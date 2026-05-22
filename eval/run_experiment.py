# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experiment runner: evaluates the RAG chatbot against a curated question dataset.

Bypasses Chainlit entirely and drives the ChatOrchestrator directly.  Each
dataset example gets a *fresh* orchestrator instance to prevent history
contamination across unrelated questions.

Usage::

    # Default: reads eval/datasets/rag_questions.jsonl
    uv run --group eval python eval/run_experiment.py

    # Custom paths
    uv run --group eval python eval/run_experiment.py \\
        --dataset-file eval/datasets/rag_questions.jsonl \\
        --dataset-name rag-questions-v1 \\
        --experiment-name "retrieval-top-k-5"

    # Quick sanity check (1 example, no Phoenix upload)
    uv run --group eval python eval/run_experiment.py --dry-run

See eval/README.md for setup instructions and a Phoenix feature overview.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never, cast
from uuid import uuid4

if TYPE_CHECKING:
    from phoenix.client import Client  # type: ignore[import-not-found]
    from phoenix.client.resources.datasets import Dataset  # type: ignore[import-not-found]

import structlog

from src.chatbot.app.chat_prompts import DEFAULT_PROMPTS
from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.build_from_settings import build_chat_model_with_profile, build_retrieval_tool
from src.chatbot.contracts.chat import ThinkingContent
from src.chatbot.contracts.citation import (
    HallucinatedCitation,
    NumberedCitation,
    UnsubstantiatedClaim,
)
from src.chatbot.contracts.process import AuthRequiredEvent, ToolCallFinished, ToolCallStarted
from src.chatbot.ui.citation_view import (
    build_citation_markdown,
    format_citation_marker,
    format_text_chunk,
)
from src.chatbot.ui.i18n_messages import resolve_message
from src.shared.observability import configure_tracing
from src.shared.settings import Settings, get_settings

logger = structlog.get_logger(__name__)

_settings = get_settings()


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------


def _git_commit_hash() -> str | None:
    """Return the short HEAD commit hash, or ``None`` if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _collect_experiment_metadata(settings: Settings) -> dict[str, Any]:
    """Assemble a metadata dict describing the full configuration of this run.

    Attached to every Phoenix experiment so that runs with different model
    configurations can be filtered and compared in the UI.  Includes:

    - ``chat_model`` / ``chat_model_provider``: the generation model
    - ``embedding_model`` / ``embedding_model_provider``: the retrieval embedding
    - ``eval_judge_model`` / ``eval_judge_provider``: the LLM-as-judge model
    - ``git_commit``: short HEAD hash (``None`` when git is unavailable)
    """
    return {
        "chat_model": settings.chat_model,
        "chat_model_provider": settings.chat_model_provider,
        "embedding_model": settings.embedding_model,
        "embedding_model_provider": settings.embedding_model_provider,
        "eval_judge_model": settings.eval_judge_model,
        "eval_judge_provider": settings.eval_judge_provider,
        "git_commit": _git_commit_hash(),
    }


def _build_experiment_description(settings: Settings) -> str:
    """Build a concise one-line description for the Phoenix experiment UI.

    Surfaces the most important configuration at a glance:
    ``chat: <model> (<provider>) | embed: <model> | judge: <model> | <git_hash>``

    Complements the full metadata dict which carries all fields but requires
    clicking into the experiment detail to read.
    """
    parts = [
        f"chat: {settings.chat_model} ({settings.chat_model_provider})",
        f"embed: {settings.embedding_model}",
        f"judge: {settings.eval_judge_model}",
    ]
    commit = _git_commit_hash()
    if commit:
        parts.append(commit)
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Eval LLM judge factory
# ---------------------------------------------------------------------------


def _eval_with_retry(
    fn: Any, input: dict[str, Any], retries: int = 20, sleep_s: float = 15.0
) -> Any:
    """Call ``fn(input)`` retrying on any 429 rate-limit error up to *retries* times.

    Catches both ``phoenix.evals.rate_limiters.RateLimitError`` (raised by the
    Phoenix LLM judge wrapper) and ``openai.RateLimitError`` (raised by the
    OpenAI-compatible chat client used in tasks).  Phoenix's own retry layer
    does not sleep meaningfully between attempts, so we own the backoff here.
    """
    from openai import RateLimitError as OpenAIRateLimitError  # type: ignore[import-not-found]
    from phoenix.evals.rate_limiters import RateLimitError as PhoenixRateLimitError  # type: ignore[import-not-found]

    for _ in range(retries):
        try:
            return fn(input)
        except (PhoenixRateLimitError, OpenAIRateLimitError):
            time.sleep(sleep_s)
    return fn(input)  # final attempt — let any exception propagate


def _build_eval_judge_llm(settings: Settings) -> Any:
    """Construct a Phoenix ``LLM`` instance from eval judge settings.

    All providers are routed through Phoenix's ``openai`` provider, which
    uses the OpenAI-compatible wire format.  For Ollama, the ``/v1`` suffix
    is appended automatically when absent.

    Args:
        settings: Application settings carrying ``eval_judge_*`` fields.

    Returns:
        A configured ``phoenix.evals.LLM`` instance.
    """
    from phoenix.evals import LLM  # type: ignore[import-not-found]

    if settings.eval_judge_provider == "ollama":
        raw_url = (settings.eval_judge_base_url or "http://localhost:11434").rstrip("/")
        base_url = raw_url if raw_url.endswith("/v1") else raw_url + "/v1"
        api_key = settings.eval_judge_api_key or "ollama"
    else:
        base_url = settings.eval_judge_base_url
        api_key = settings.eval_judge_api_key

    return LLM(
        provider="openai",
        model=settings.eval_judge_model,
        base_url=base_url,
        api_key=api_key,
        initial_per_second_request_rate=settings.eval_judge_initial_per_second_request_rate,
    )


# ---------------------------------------------------------------------------
# Phoenix experiment task
# ---------------------------------------------------------------------------


async def task(input: dict[str, str]) -> dict[str, object]:
    """Phoenix experiment task: runs one query and returns the full answer text.

    The ``input`` parameter name is significant: Phoenix binds it to the
    ``input`` field of each dataset Example (i.e. the columns listed in
    ``input_keys`` when the dataset was created).

    A fresh orchestrator is created per invocation to prevent history leakage.

    Args:
        input: Dict of input columns from the dataset example, must contain
            a ``"query"`` key.

    Returns:
        Dict with:
        - ``"answer"``: concatenated streamed response text with citation markers
        - ``"context"``: JSON-serialized tool results concatenated across all
          tool calls; used as faithfulness context
        - ``"citation_count"``: number of distinct numbered citations emitted by the model
        - ``"hallucinated_citation_count"``: number of hallucinated citation events
        - ``"unsubstantiated_claim_count"``: number of unsubstantiated claim events
    """
    chat_model, model_profile = build_chat_model_with_profile(_settings)
    retrieval_tool = build_retrieval_tool(_settings)
    orchestrator = ChatOrchestrator.create(
        chat_model,
        tools=[retrieval_tool],
        model_profile=model_profile,
        prompts=DEFAULT_PROMPTS,
    )
    query: str = input["query"]
    text_parts: list[str] = []
    pending_whitespace: str = ""
    numbered: list[NumberedCitation] = []
    hallucinated_citation_count: int = 0
    unsubstantiated_claim_count: int = 0
    context_parts: list[str] = []
    retrieved_documents: list[dict[str, object]] = []
    async for event in orchestrator.process_message(query):
        match event:
            case str():
                tokens, pending_whitespace = format_text_chunk(event, pending_whitespace)
                text_parts.extend(tokens)
            case NumberedCitation():
                tokens, pending_whitespace = format_citation_marker(event, pending_whitespace)
                text_parts.extend(tokens)
                numbered.append(event)
            case UnsubstantiatedClaim():
                unsubstantiated_claim_count += 1
            case HallucinatedCitation():
                hallucinated_citation_count += 1
            case ToolCallFinished():
                if event.result is not None:
                    context_parts.append(json.dumps(event.result))
                    _chunks = event.result.get("chunks")
                    if isinstance(_chunks, list):
                        retrieved_documents.extend(cast(list[dict[str, object]], _chunks))
            case ToolCallStarted() | ThinkingContent() | AuthRequiredEvent():
                pass
            case _ as unreachable:
                assert_never(unreachable)
    if pending_whitespace:
        text_parts.append(pending_whitespace)
    answer = "".join(text_parts)
    seen: set[int] = set()
    unique: list[NumberedCitation] = []
    for nc in numbered:
        if nc.reference_number not in seen:
            seen.add(nc.reference_number)
            unique.append(nc)
    if unique:
        answer += build_citation_markdown(
            unique, translate=lambda msg: resolve_message(msg, lang="en")
        )
    return {
        "answer": answer,
        "context": "\n\n".join(context_parts),
        "citation_count": len(unique),
        "hallucinated_citation_count": hallucinated_citation_count,
        "unsubstantiated_claim_count": unsubstantiated_claim_count,
        "retrieved_documents": retrieved_documents,
    }


def _sync_task(input: dict[str, str]) -> dict[str, object]:
    """Sync wrapper around ``task`` for the synchronous Phoenix ``Client``.

    ``run_experiment`` on the sync client cannot accept a coroutine function.
    ``asyncio.run`` creates a fresh event loop per invocation, which is safe
    here because each call is independent and the orchestrator is stateless
    across invocations.

    Delegates to ``_eval_with_retry`` so that 429 rate-limit errors from the
    chat model are handled with the same sleep-and-retry logic used for the
    LLM judge evaluators.  A longer ``sleep_s`` is used here because chat
    completions consume more of the rate-limit budget than judge calls.
    """
    return _eval_with_retry(  # type: ignore[return-value]
        lambda inp: asyncio.run(task(inp)),  # type: ignore[arg-type]
        input,  # type: ignore[arg-type]
        retries=5,
        sleep_s=90.0,
    )


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def has_citations(output: dict[str, object]) -> bool:
    """Check that the answer contains at least one numbered citation.

    Uses the ``citation_count`` field emitted by the task (derived from the
    stream of ``NumberedCitation`` events) rather than regex-matching the
    rendered answer text — consistent with how ``no_hallucinated_citations``
    and ``no_unsubstantiated_claims`` work.
    """
    count = output.get("citation_count", 0)
    return isinstance(count, int) and count > 0


def is_non_empty(output: dict[str, object]) -> bool:
    """Check that the answer is a non-trivial response (more than 50 chars)."""
    return len(str(output.get("answer", "")).strip()) > 50


def no_unsubstantiated_claims(output: dict[str, object]) -> bool:
    """Check that the answer contains no unsubstantiated claims (true = good).

    Returns ``True`` when ``unsubstantiated_claim_count`` is zero, meaning all
    model claims are covered by a citation token.  A ``False`` result indicates
    the model made at least one claim without citing a source — a signal of
    incomplete attribution.
    """
    count = output.get("unsubstantiated_claim_count", 0)
    return not (isinstance(count, int) and count > 0)


def no_hallucinated_citations(output: dict[str, object]) -> bool:
    """Check that the answer contains no hallucinated citations (true = good).

    Returns ``True`` when ``hallucinated_citation_count`` is zero, meaning
    every citation token the model emitted could be resolved to an actual
    retrieved chunk.  A ``False`` result is a signal of fabricated sourcing.
    """
    count = output.get("hallucinated_citation_count", 0)
    return not (isinstance(count, int) and count > 0)


def _build_code_evaluators() -> list[Any]:
    """Wrap the plain code evaluator functions with Phoenix ``create_evaluator``.

    Using ``create_evaluator`` (rather than passing raw callables) ensures
    ``direction="maximize"`` is attached to every evaluator so Phoenix renders
    True as green in the UI for all of them.

    The plain functions remain importable and callable without Phoenix for unit
    tests — this factory is only called at experiment runtime after the
    phoenix.evals lazy import succeeds.
    """
    from phoenix.evals import create_evaluator  # type: ignore[import-not-found]

    return [
        create_evaluator(name="has_citations", kind="code", direction="maximize")(has_citations),
        create_evaluator(name="is_non_empty", kind="code", direction="maximize")(is_non_empty),
        create_evaluator(name="no_unsubstantiated_claims", kind="code", direction="maximize")(
            no_unsubstantiated_claims
        ),
        create_evaluator(name="no_hallucinated_citations", kind="code", direction="maximize")(
            no_hallucinated_citations
        ),
    ]


def _build_faithfulness_evaluator(llm: Any) -> Any:
    """Construct a Phoenix ``ClassificationEvaluator`` for faithfulness scoring.

    Uses the built-in Phoenix faithfulness prompt template
    (``FAITHFULNESS_CLASSIFICATION_EVALUATOR_CONFIG``) which scores responses
    as ``faithful`` (1.0) or ``unfaithful`` (0.0) relative to the retrieved
    context.  The LLM judge is configured via ``eval_judge_*`` settings.

    Phoenix's experiment framework calls ``evaluator.evaluate(kwargs)`` where
    ``kwargs`` carries experiment-level keys (``output``, ``input``,
    ``expected``, …).  The faithfulness template however expects flat keys
    ``context`` and ``output``.  ``_FaithfulnessEvaluator`` remaps before
    delegating so that both the flat and nested forms are handled correctly.

    Args:
        llm: Shared ``phoenix.evals.LLM`` instance (shared rate limiter).
    """
    from phoenix.evals import create_classifier  # type: ignore[import-not-found]
    from phoenix.evals.__generated__.classification_evaluator_configs import (  # type: ignore[import-not-found]
        FAITHFULNESS_CLASSIFICATION_EVALUATOR_CONFIG as CFG,
    )

    classifier = create_classifier(
        name="faithfulness",
        prompt_template=CFG.messages[0].content,
        llm=llm,
        choices=CFG.choices,
    )

    class _FaithfulnessEvaluator:
        """Thin remapping wrapper satisfying the ``EvalsEvaluator`` protocol.

        Translates the experiment-framework envelope
        ``{"output": {"answer": ..., "context": ...}, ...}`` into the flat
        ``{"output": <answer>, "context": <context>}`` shape expected by the
        faithfulness template.
        """

        name: str = "faithfulness"
        direction: str = classifier.direction
        source: str = "LLM"
        input_schema: Any = None  # no schema — remapping handles field access

        @staticmethod
        def _remap(input: dict[str, Any]) -> dict[str, Any]:
            output = input.get("output", {})
            dataset_input = input.get("input", {})
            if isinstance(output, dict):
                output_dict = cast(dict[str, Any], output)
                query = (
                    cast(dict[str, Any], dataset_input).get("query", "")
                    if isinstance(dataset_input, dict)
                    else str(dataset_input)
                )
                return {
                    "input": query,
                    "context": output_dict.get("context", ""),
                    "output": output_dict.get("answer", ""),
                }
            return input

        def evaluate(self, input: dict[str, Any]) -> Any:
            return _eval_with_retry(classifier.evaluate, self._remap(input))

        async def async_evaluate(self, input: dict[str, Any]) -> Any:
            return await classifier.async_evaluate(self._remap(input))

    return _FaithfulnessEvaluator()


def _build_correctness_evaluator(llm: Any) -> Any:
    """Construct a Phoenix ``CorrectnessEvaluator`` for answer quality scoring.

    Evaluates whether the chatbot's answer is factually accurate and complete
    for the given question — without requiring ground-truth reference answers.
    Complements faithfulness (which tests grounding) with a general facticity
    signal.

    Input mapping:  experiment envelope ``{"input": {"query": ...},
    "output": {"answer": ...}}``  →  flat ``{"input": query, "output": answer}``
    expected by ``CorrectnessEvaluator``.

    Args:
        llm: Shared ``phoenix.evals.LLM`` instance (shared rate limiter).
    """
    from phoenix.evals.metrics import CorrectnessEvaluator  # type: ignore[import-not-found]

    evaluator = CorrectnessEvaluator(llm=llm)

    class _CorrectnessEvaluator:
        """Thin remapping wrapper satisfying the ``EvalsEvaluator`` protocol.

        Translates the experiment-framework envelope
        ``{"input": {"query": ...}, "output": {"answer": ...}}`` into the flat
        ``{"input": <query>, "output": <answer>}`` shape expected by
        ``CorrectnessEvaluator``.
        """

        name: str = "correctness"
        direction: str = "maximize"
        source: str = "LLM"
        input_schema: Any = None  # no schema — remapping handles field access

        @staticmethod
        def _remap(input: dict[str, Any]) -> dict[str, Any]:
            output = input.get("output", {})
            dataset_input = input.get("input", {})
            if isinstance(output, dict) and isinstance(dataset_input, dict):
                return {
                    "input": cast(dict[str, Any], dataset_input).get("query", ""),
                    "output": cast(dict[str, Any], output).get("answer", ""),
                }
            return input

        def evaluate(self, input: dict[str, Any]) -> Any:
            return _eval_with_retry(evaluator.evaluate, self._remap(input))

        async def async_evaluate(self, input: dict[str, Any]) -> Any:
            return await evaluator.async_evaluate(self._remap(input))

    return _CorrectnessEvaluator()


def _build_document_relevance_evaluator(
    llm: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    """Construct a per-document relevance evaluator and a shared result collector.

    Returns a ``(evaluator, collector)`` tuple.  The evaluator is passed to
    ``run_experiment``; the collector accumulates one record per evaluated
    document and is written to a JSONL file after the experiment completes.

    Uses ``create_evaluator(kind="code")`` rather than a custom class so that
    Phoenix routes the scalar return value through the correct code-evaluator
    path — the custom-class path expects a ``list[Score]`` and crashes with
    ``TypeError: 'float' object is not subscriptable`` on scalar returns.

    For each task execution the evaluator:

    1. Receives ``input`` (dataset input: ``{"query": ...}``) and ``output``
       (task output: ``{"retrieved_documents": [...], ...}``) via Phoenix
       parameter binding by name.
    2. Calls ``DocumentRelevanceEvaluator`` once per document.
    3. Appends per-document records to the thread-safe *collector*.
    4. Returns ``mean_document_relevance`` (0.0-1.0) as the experiment scalar.

    Args:
        llm: Shared ``phoenix.evals.LLM`` instance (shared rate limiter).
    """
    from phoenix.evals import create_evaluator  # type: ignore[import-not-found]
    from phoenix.evals.metrics import DocumentRelevanceEvaluator  # type: ignore[import-not-found]

    doc_evaluator = DocumentRelevanceEvaluator(llm=llm)

    collector: list[dict[str, Any]] = []
    _lock = threading.Lock()

    def _docs_from(output: dict[str, Any]) -> list[dict[str, Any]]:
        docs = output.get("retrieved_documents", [])
        if isinstance(docs, list):
            return cast(list[dict[str, Any]], docs)
        return []

    @create_evaluator(name="mean_document_relevance", kind="code", direction="maximize")
    def mean_document_relevance(
        input: dict[str, Any],  # name bound by Phoenix parameter matching
        output: dict[str, Any],
    ) -> float:
        """Evaluate mean relevance of all retrieved documents for this example."""
        query = str(input.get("query", ""))
        valid_docs = [d for d in _docs_from(output) if str(d.get("content", "")).strip()]
        if not valid_docs:
            return 0.0
        scores: list[float] = []
        for doc in valid_docs:
            _eval_input = {"input": query, "document_text": str(doc.get("content", ""))}
            result = _eval_with_retry(doc_evaluator.evaluate, _eval_input)
            score_obj = result[0] if result else None
            score: float | None = (
                float(score_obj.score)
                if score_obj is not None and score_obj.score is not None
                else None
            )
            if score is not None:
                scores.append(score)
            with _lock:
                collector.append(
                    {
                        "query": query,
                        "document_text": str(doc.get("content", "")),
                        "document_source": str(doc.get("source", "")),
                        "relevance_score": score,
                        "relevance_label": (
                            "relevant" if score is not None and score >= 0.5 else "unrelated"
                        ),
                    }
                )
        mean = sum(scores) / len(scores) if scores else 0.0
        logger.info(
            "document_relevance.evaluated",
            mean=round(mean, 4),
            doc_count=len(scores),
        )
        return mean

    return mean_document_relevance, collector


# ---------------------------------------------------------------------------
# Per-document results export
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------------


def _fetch_replay_cache(
    client: Client,
    experiment_id: str,
) -> dict[str, dict[str, object]]:
    """Build a ``{query: task_output}`` lookup from a completed Phoenix experiment.

    Fetches the experiment's dataset examples (to resolve ``query`` from
    ``dataset_example_id``) and its task runs (to retrieve cached outputs),
    then joins on ``dataset_example_id``.

    Handles paginated runs via the ``next_cursor`` continuation token.
    Runs with ``error`` set or ``output`` absent are silently skipped.

    Args:
        client:        Authenticated Phoenix ``Client``.
        experiment_id: The Phoenix experiment ID to replay.

    Returns:
        Mapping from query string to the original task output dict.
    """
    # Resolve dataset_id so we can fetch the example inputs.
    exp_resp = client._client.get(  # pyright: ignore[reportPrivateUsage]
        f"v1/experiments/{experiment_id}",
    )
    exp_resp.raise_for_status()
    exp_data: dict[str, Any] = exp_resp.json()["data"]
    dataset_id: str = exp_data["dataset_id"]
    dataset_version_id: str = exp_data["dataset_version_id"]

    # Fetch all dataset examples to map example_id → query.
    ex_resp = client._client.get(  # pyright: ignore[reportPrivateUsage]
        f"v1/datasets/{dataset_id}/examples",
        params={"version_id": dataset_version_id},
    )
    ex_resp.raise_for_status()
    examples: list[dict[str, Any]] = ex_resp.json()["data"]["examples"]
    example_to_query: dict[str, str] = {
        ex["id"]: cast(dict[str, Any], ex["input"]).get("query", "") for ex in examples
    }

    # Fetch all task runs (paginated) to map example_id → output.
    example_to_output: dict[str, dict[str, object]] = {}
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        runs_resp = client._client.get(  # pyright: ignore[reportPrivateUsage]
            f"v1/experiments/{experiment_id}/runs",
            params=params,
        )
        runs_resp.raise_for_status()
        body: dict[str, Any] = runs_resp.json()
        for run in body.get("data", []):
            if run.get("error") or run.get("output") is None:
                continue
            example_to_output[run["dataset_example_id"]] = cast(dict[str, object], run["output"])
        cursor = body.get("next_cursor")
        if not cursor:
            break

    # Join on example_id.
    cache: dict[str, dict[str, object]] = {}
    for example_id, output in example_to_output.items():
        query = example_to_query.get(example_id, "")
        if query:
            cache[query] = output
    logger.info(
        "replay_cache.built",
        experiment_id=experiment_id,
        cached_examples=len(cache),
    )
    return cache


def _build_replay_task(
    cache: dict[str, dict[str, object]],
) -> Any:
    """Return a synchronous task function that serves cached outputs by query.

    Intended for use with ``run_experiment`` when replaying task outputs from a
    previous experiment — avoiding re-inference while still creating a fresh
    Phoenix experiment entry with new name, metadata, and evaluators.

    Queries not found in the cache return an empty stub so the experiment
    run completes rather than erroring out.

    Args:
        cache: Mapping ``{query: task_output}`` built by ``_fetch_replay_cache``.
    """
    _stub: dict[str, object] = {
        "answer": "",
        "context": "",
        "citation_count": 0,
        "hallucinated_citation_count": 0,
        "unsubstantiated_claim_count": 0,
        "retrieved_documents": [],
    }

    def replay_task(input: dict[str, str]) -> dict[str, object]:
        return cache.get(input.get("query", ""), _stub)

    return replay_task


def _write_per_document_results(
    records: list[dict[str, Any]],
    experiment_name: str,
) -> None:
    """Write per-document relevance records to ``eval/results/{name}-perdoc.jsonl``.

    Each line is a JSON object with the fields populated by the document
    relevance evaluator's collector:
    ``query``, ``document_text``, ``document_source``,
    ``relevance_score``, ``relevance_label``.

    Exporting to a local file rather than a secondary Phoenix experiment avoids
    the dataset-granularity mismatch (per-doc rows mixed with per-question
    experiments in the same Phoenix dataset view).  The JSONL can be loaded
    directly into a notebook or spreadsheet for deep-dive analysis.

    Args:
        records:         Per-document dicts from the evaluator collector.
        experiment_name: Used to derive the output file name.
    """
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment_name}-perdoc.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(
        "per_document_results.written",
        path=str(out_path),
        rows=len(records),
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    """Parse a JSONL file into a list of dicts, skipping blank lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _get_or_create_dataset(
    client: Client,
    dataset_file: Path,
    dataset_name: str,
) -> Dataset:
    """Return the named Phoenix dataset, creating it from the JSONL file if absent.

    Attempts to fetch an existing dataset by name first to avoid 409 conflicts on
    repeated runs.  Only uploads when the dataset does not yet exist in Phoenix.

    ``input_keys`` is always ``["query"]``.  ``output_keys`` is set to
    ``["reference_answer"]`` if that column exists in the file.
    """
    import pandas as pd  # eval-group dependency; imported lazily

    try:
        dataset = client.datasets.get_dataset(dataset=dataset_name)
        logger.info("dataset.found", name=dataset_name)
        return dataset
    except (ValueError, Exception) as exc:
        # ValueError: name not found; fall through to create
        if "not found" not in str(exc).lower() and "404" not in str(exc):
            raise

    rows = _load_jsonl(dataset_file)
    df = pd.DataFrame(rows)

    has_reference = "reference_answer" in df.columns
    output_keys: list[str] = ["reference_answer"] if has_reference else []

    dataset = client.datasets.create_dataset(
        name=dataset_name,
        dataframe=df,
        input_keys=["query"],
        output_keys=output_keys,
    )
    logger.info(
        "dataset.created",
        name=dataset_name,
        num_examples=len(df),
        has_reference=has_reference,
    )
    return dataset


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Phoenix experiment on the RAG chatbot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See eval/README.md for setup instructions.",
    )
    parser.add_argument(
        "--dataset-file",
        default="eval/datasets/rag_questions.jsonl",
        help="Path to the JSONL file with eval questions (default: eval/datasets/rag_questions.jsonl).",
    )
    parser.add_argument(
        "--dataset-name",
        default="rag-questions-v1",
        help="Phoenix dataset name; created if it does not exist yet (default: rag-questions-v1).",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Phoenix experiment name. Defaults to a random run ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run against 1 example only without uploading results to Phoenix.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        metavar="ID",
        help=(
            "Re-run evaluators on an existing experiment without executing tasks again. "
            "The ID is shown in the Phoenix UI experiment detail page."
        ),
    )
    parser.add_argument(
        "--replay-from",
        default=None,
        metavar="ID",
        help=(
            "Create a NEW Phoenix experiment by replaying task outputs from the given "
            "experiment ID.  The task is not re-executed; cached outputs are returned "
            "directly.  Use --experiment-name to name the new experiment.  "
            "Evaluators (and their judge model) are taken from the current .env config."
        ),
    )
    args = parser.parse_args()

    if args.replay_from and args.experiment_id:
        logger.error(
            "args.conflict",
            msg="--replay-from and --experiment-id are mutually exclusive",
        )
        sys.exit(1)

    dataset_file = Path(args.dataset_file)
    if not dataset_file.exists():
        logger.error("dataset_file.not_found", path=str(dataset_file))
        sys.exit(1)

    # Configure tracing so experiment spans are forwarded to Phoenix.
    # Jaeger export is disabled here — experiment spans belong in Phoenix only.
    # Use a dedicated "-eval" project so experiment traces don't pollute the
    # live chatbot project in the Phoenix UI.
    configure_tracing(
        enabled=_settings.otel_enabled,
        service_name=_settings.otel_service_name,
        project_name=f"{_settings.phoenix_project_name}-eval",
        deployment_environment=_settings.otel_deployment_environment,
        phoenix_otlp_endpoint=_settings.otel_phoenix_otlp_endpoint,
        phoenix_export=_settings.otel_export_phoenix,
        jaeger_otlp_endpoint=_settings.otel_jaeger_otlp_endpoint,
        jaeger_export=False,
        sample_rate=1.0,  # always sample all eval spans
        console_export=_settings.otel_console_export,
        auto_instrument_haystack=_settings.otel_auto_instrument_haystack,
    )

    # Lazy imports: phoenix.client and pandas are eval-group dependencies
    # not available in the default environment.
    from phoenix.client import Client  # type: ignore[import-not-found]
    from phoenix.evals.rate_limiters import RateLimitError  # type: ignore[import-not-found]

    client: Client = Client()
    llm = _build_eval_judge_llm(_settings)
    doc_relevance_eval, doc_records = _build_document_relevance_evaluator(llm)
    evaluators = [
        *_build_code_evaluators(),
        _build_faithfulness_evaluator(llm),
        _build_correctness_evaluator(llm),
        doc_relevance_eval,
    ]

    metadata = _collect_experiment_metadata(_settings)
    description = _build_experiment_description(_settings)

    if args.dry_run:
        logger.info("dry_run.start", **metadata)
        experiment = client.experiments.run_experiment(
            dataset=_get_or_create_dataset(client, dataset_file, args.dataset_name),
            task=_sync_task,
            evaluators=evaluators,
            experiment_metadata=metadata,
            experiment_description=description,
            dry_run=1,
            rate_limit_errors=(RateLimitError,),
            retries=12,
        )
        logger.info("dry_run.complete", experiment=str(experiment))
        return

    if args.experiment_id:
        logger.info("evaluate_only.start", experiment_id=args.experiment_id, **metadata)
        existing = client.experiments.get_experiment(experiment_id=args.experiment_id)
        result = client.experiments.evaluate_experiment(
            experiment=existing,
            evaluators=evaluators,
            rate_limit_errors=(RateLimitError,),
            retries=12,
        )
        logger.info(
            "evaluate_only.complete",
            experiment_id=getattr(result, "id", str(result)),
        )
        if doc_records:
            perdoc_name = args.experiment_name or f"eval-{args.experiment_id}"
            _write_per_document_results(doc_records, perdoc_name)
        return

    if args.replay_from:
        logger.info("replay.start", replay_from=args.replay_from, **metadata)
        replay_cache = _fetch_replay_cache(client, args.replay_from)
        task_fn = _build_replay_task(replay_cache)
        metadata = {**metadata, "replay_from": args.replay_from}
    else:
        task_fn = _sync_task

    experiment_name: str = args.experiment_name or f"chatbot-eval-{uuid4().hex[:8]}"
    dataset = _get_or_create_dataset(client, dataset_file, args.dataset_name)

    logger.info("experiment.start", name=experiment_name, **metadata)
    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task_fn,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_metadata=metadata,
        experiment_description=description,
        rate_limit_errors=(RateLimitError,),
        retries=12,
    )
    logger.info("experiment.complete", experiment_id=getattr(experiment, "id", str(experiment)))
    if doc_records:
        _write_per_document_results(doc_records, experiment_name)


if __name__ == "__main__":
    main()
