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
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never, cast
from uuid import uuid4

if TYPE_CHECKING:
    from phoenix.client import Client  # type: ignore[import-not-found]
    from phoenix.client.resources.datasets import Dataset  # type: ignore[import-not-found]

import structlog

from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.app.prompts import DEFAULT_PROMPTS
from src.chatbot.app.protocols import (
    AuthRequiredEvent,
    HallucinatedCitation,
    NumberedCitation,
    ThinkingContent,
    Tool,
    ToolCallFinished,
    ToolCallStarted,
    UnsubstantiatedClaim,
)
from src.chatbot.config import (
    build_chat_model_config,
    build_retriever_config,
    build_text_embedder_config,
)
from src.chatbot.infrastructure.chat import build_chat_model, build_chat_model_profile
from src.chatbot.infrastructure.embeddings_text import build_text_embedder
from src.chatbot.infrastructure.retrieval import build_retriever
from src.chatbot.observability import configure_tracing
from src.chatbot.tools.retrieval.tool import RetrievalTool
from src.chatbot.ui.citation_view import (
    build_citation_markdown,
    format_citation_marker,
    format_text_chunk,
)
from src.chatbot.ui.i18n_messages import resolve_message
from src.settings import Settings, get_settings

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


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------


def _build_eval_orchestrator() -> ChatOrchestrator:
    """Build a fresh, session-less ChatOrchestrator for one eval invocation.

    Mirrors the factory in ``src/chatbot/ui/app.py::_build_orchestrator`` but
    omits the vacation-days tool (irrelevant for RAG evaluation) and skips all
    Chainlit session context.  A new instance must be created per task call to
    prevent history from leaking between unrelated dataset examples.
    """
    chat_model_config = build_chat_model_config(_settings)
    model_profile = build_chat_model_profile(chat_model_config)
    chat_model = build_chat_model(
        chat_model_config,
        parse_text_tool_calls=model_profile.parse_text_tool_calls,
    )
    text_embedder = build_text_embedder(build_text_embedder_config(_settings))
    retriever = build_retriever(
        config=build_retriever_config(_settings),
        text_embedder=text_embedder,
    )
    tools: list[Tool] = [RetrievalTool(retriever=retriever)]
    return ChatOrchestrator.create(
        chat_model,
        tools=tools,
        model_profile=model_profile,
        prompts=DEFAULT_PROMPTS,
    )


# ---------------------------------------------------------------------------
# Eval LLM judge factory
# ---------------------------------------------------------------------------


def _build_eval_judge_llm(settings: Settings) -> Any:
    """Construct a Phoenix ``LLM`` instance from eval judge settings.

    Both ``ollama`` and ``openai_compatible`` providers are routed through the
    OpenAI-compatible client: Ollama exposes an OpenAI-compatible API at
    ``/v1``, and other providers (Groq, OpenAI, etc.) use the same wire
    format.

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
        - ``"hallucinated_citation_count"``: number of hallucinated citation events
        - ``"unsubstantiated_claim_count"``: number of unsubstantiated claim events
    """
    orchestrator = _build_eval_orchestrator()
    query: str = input["query"]
    text_parts: list[str] = []
    pending_whitespace: str = ""
    numbered: list[NumberedCitation] = []
    hallucinated_citation_count: int = 0
    unsubstantiated_claim_count: int = 0
    context_parts: list[str] = []
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
        "hallucinated_citation_count": hallucinated_citation_count,
        "unsubstantiated_claim_count": unsubstantiated_claim_count,
    }


def _sync_task(input: dict[str, str]) -> dict[str, object]:
    """Sync wrapper around ``task`` for the synchronous Phoenix ``Client``.

    ``run_experiment`` on the sync client cannot accept a coroutine function.
    ``asyncio.run`` creates a fresh event loop per invocation, which is safe
    here because each call is independent and the orchestrator is stateless
    across invocations.
    """
    return asyncio.run(task(input))


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def has_citations(output: dict[str, object]) -> bool:
    """Check that the answer contains at least one inline citation marker [N].

    The chatbot should always cite sources for retrieved content.  Answers
    without any ``[1]``-style marker indicate a retrieval or citation failure.
    """
    return bool(re.search(r"\[\d+\]", str(output.get("answer", ""))))


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


def _build_faithfulness_evaluator(settings: Settings) -> Any:
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
        settings: Application settings carrying ``eval_judge_*`` fields.
    """
    from phoenix.evals import create_classifier  # type: ignore[import-not-found]
    from phoenix.evals.__generated__.classification_evaluator_configs import (  # type: ignore[import-not-found]
        FAITHFULNESS_CLASSIFICATION_EVALUATOR_CONFIG as CFG,
    )

    llm = _build_eval_judge_llm(settings)
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
            return classifier.evaluate(self._remap(input))

        async def async_evaluate(self, input: dict[str, Any]) -> Any:
            return await classifier.async_evaluate(self._remap(input))

    return _FaithfulnessEvaluator()


def _build_correctness_evaluator(settings: Settings) -> Any:
    """Construct a Phoenix ``CorrectnessEvaluator`` for answer quality scoring.

    Evaluates whether the chatbot's answer is factually accurate and complete
    for the given question — without requiring ground-truth reference answers.
    Complements faithfulness (which tests grounding) with a general facticity
    signal.

    Input mapping:  experiment envelope ``{"input": {"query": ...},
    "output": {"answer": ...}}``  →  flat ``{"input": query, "output": answer}``
    expected by ``CorrectnessEvaluator``.

    Args:
        settings: Application settings carrying ``eval_judge_*`` fields.
    """
    from phoenix.evals.metrics import CorrectnessEvaluator  # type: ignore[import-not-found]

    llm = _build_eval_judge_llm(settings)
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
            return evaluator.evaluate(self._remap(input))

        async def async_evaluate(self, input: dict[str, Any]) -> Any:
            return await evaluator.async_evaluate(self._remap(input))

    return _CorrectnessEvaluator()


def _extract_document_texts_from_span(span_row: Any) -> list[str]:
    """Extract retrieval document content strings from a Phoenix spans-DataFrame row.

    OpenInference emits indexed span attributes:
    ``retrieval.documents.0.document.content``,
    ``retrieval.documents.1.document.content``, …
    Phoenix flattens these into DataFrame columns prefixed with ``attributes.``.
    """
    texts: list[str] = []
    i = 0
    while True:
        col = f"attributes.retrieval.documents.{i}.document.content"
        if col not in span_row.index:
            break
        val = span_row.get(col)
        if isinstance(val, str) and val.strip():
            texts.append(val)
        i += 1
    return texts


def _build_document_relevance_evaluator(settings: Settings) -> Any:
    """Construct a per-document relevance evaluator with span-level annotation.

    For each task execution identified by ``trace_id``:

    1. Fetches the retriever span from Phoenix for that trace.
    2. Calls ``DocumentRelevanceEvaluator`` once per retrieved document.
    3. Logs a ``document_relevance`` annotation on the retriever span in Phoenix
       (mean relevance of all its documents, ``relevant``/``unrelated`` label).
    4. Returns ``mean_document_relevance`` (0..1) as the experiment-level scalar.

    Requires the experiment runner to export spans to Phoenix
    (``phoenix_export=True`` in ``configure_tracing``) so that retriever spans
    are available for querying before evaluators run.  The retriever span
    emits ``retrieval.documents.*`` attributes (OpenInference standard) via
    ``build_retriever_attributes`` in ``_qdrant.py`` — no additional
    instrumentation is required.

    Args:
        settings: Application settings carrying ``eval_judge_*`` and
            ``phoenix_project_name`` fields.
    """
    from phoenix.evals.metrics import DocumentRelevanceEvaluator  # type: ignore[import-not-found]

    llm = _build_eval_judge_llm(settings)
    doc_evaluator = DocumentRelevanceEvaluator(llm=llm)
    project_name = settings.phoenix_project_name
    # Span name constant kept inline to avoid coupling to the infra module.
    _RETRIEVER_SPAN_NAME = "chat.retriever.qdrant.retrieve"

    class _DocumentRelevanceEvaluator:
        """Wraps ``DocumentRelevanceEvaluator`` for per-span retrieval evaluation.

        Fetches retriever spans for the current trace from Phoenix, evaluates
        each retrieved document, annotates the retriever span with the per-span
        mean relevance score, and returns the trace-level mean as the
        experiment evaluation scalar.
        """

        name: str = "mean_document_relevance"
        direction: str = "maximize"
        source: str = "LLM"
        input_schema: Any = None  # no schema — handled via trace_id and span query

        @staticmethod
        def _fetch_retriever_spans(client: Any, trace_id: str) -> Any:
            """Return the RETRIEVER spans DataFrame for ``trace_id``, or ``None``."""
            try:
                spans_df = client.spans.get_spans_dataframe(project_name=project_name)
            except Exception as exc:
                logger.warning("document_relevance.spans_fetch_failed", error=str(exc))
                return None

            if spans_df is None or spans_df.empty:
                logger.warning("document_relevance.no_spans", trace_id=trace_id)
                return None

            if "context.trace_id" in spans_df.columns:
                spans_df = spans_df[spans_df["context.trace_id"] == trace_id]
            if "name" in spans_df.columns:
                spans_df = spans_df[spans_df["name"] == _RETRIEVER_SPAN_NAME]

            if spans_df.empty:
                logger.warning("document_relevance.no_retriever_spans", trace_id=trace_id)
                return None
            return spans_df

        @staticmethod
        def _log_span_annotations(client: Any, annotation_rows: list[dict[str, Any]]) -> None:
            """Log per-span ``document_relevance`` annotations back to Phoenix."""
            import pandas as pd  # eval-group dependency; imported lazily

            try:
                ann_df: Any = pd.DataFrame(annotation_rows).set_index("context.span_id")  # type: ignore[reportUnknownMemberType]
                client.spans.log_span_annotations_dataframe(
                    dataframe=ann_df,
                    annotation_name="document_relevance",
                    annotator_kind="LLM",
                )
                logger.info("document_relevance.annotations_logged", count=len(annotation_rows))
            except Exception as exc:
                logger.warning("document_relevance.annotation_failed", error=str(exc))

        @staticmethod
        def _extract_query(input: dict[str, Any]) -> str:
            dataset_input = input.get("input", {})
            return (
                cast(dict[str, Any], dataset_input).get("query", "")
                if isinstance(dataset_input, dict)
                else str(dataset_input)
            )

        def evaluate(self, input: dict[str, Any]) -> Any:
            from phoenix.client import Client  # type: ignore[import-not-found]

            trace_id: str | None = cast(str | None, input.get("trace_id"))
            if not trace_id:
                logger.warning("document_relevance.no_trace_id")
                return None

            query = self._extract_query(input)
            client = Client()
            spans_df = self._fetch_retriever_spans(client, trace_id)
            if spans_df is None:
                return None

            all_scores: list[float] = []
            annotation_rows: list[dict[str, Any]] = []

            for _, span_row in spans_df.iterrows():
                span_id = span_row.get("context.span_id") or (
                    span_row.name if isinstance(span_row.name, str) else None
                )
                doc_texts = _extract_document_texts_from_span(span_row)
                if not doc_texts:
                    continue

                span_scores: list[float] = []
                for doc_text in doc_texts:
                    result = doc_evaluator.evaluate({"input": query, "document_text": doc_text})
                    score_obj = result[0] if result else None
                    if score_obj is not None and score_obj.score is not None:
                        score = float(score_obj.score)
                        all_scores.append(score)
                        span_scores.append(score)

                if span_id and span_scores:
                    mean_span = sum(span_scores) / len(span_scores)
                    annotation_rows.append(
                        {
                            "context.span_id": span_id,
                            "score": mean_span,
                            "label": "relevant" if mean_span >= 0.5 else "unrelated",
                        }
                    )

            if annotation_rows:
                self._log_span_annotations(client, annotation_rows)

            if not all_scores:
                return None

            mean_score = sum(all_scores) / len(all_scores)
            logger.info(
                "document_relevance.complete",
                mean_score=round(mean_score, 4),
                doc_count=len(all_scores),
            )
            return mean_score

        async def async_evaluate(self, input: dict[str, Any]) -> Any:
            from phoenix.client import Client  # type: ignore[import-not-found]

            trace_id: str | None = cast(str | None, input.get("trace_id"))
            if not trace_id:
                logger.warning("document_relevance.no_trace_id")
                return None

            query = self._extract_query(input)
            client = Client()
            spans_df = self._fetch_retriever_spans(client, trace_id)
            if spans_df is None:
                return None

            all_scores: list[float] = []
            annotation_rows: list[dict[str, Any]] = []

            for _, span_row in spans_df.iterrows():
                span_id = span_row.get("context.span_id") or (
                    span_row.name if isinstance(span_row.name, str) else None
                )
                doc_texts = _extract_document_texts_from_span(span_row)
                if not doc_texts:
                    continue

                # Evaluate all documents for this span in parallel.
                results = await asyncio.gather(
                    *[
                        doc_evaluator.async_evaluate({"input": query, "document_text": text})
                        for text in doc_texts
                    ]
                )

                span_scores: list[float] = []
                for result in results:
                    score_obj = result[0] if result else None
                    if score_obj is not None and score_obj.score is not None:
                        score = float(score_obj.score)
                        all_scores.append(score)
                        span_scores.append(score)

                if span_id and span_scores:
                    mean_span = sum(span_scores) / len(span_scores)
                    annotation_rows.append(
                        {
                            "context.span_id": span_id,
                            "score": mean_span,
                            "label": "relevant" if mean_span >= 0.5 else "unrelated",
                        }
                    )

            if annotation_rows:
                self._log_span_annotations(client, annotation_rows)

            if not all_scores:
                return None

            mean_score = sum(all_scores) / len(all_scores)
            logger.info(
                "document_relevance.complete",
                mean_score=round(mean_score, 4),
                doc_count=len(all_scores),
            )
            return mean_score

    return _DocumentRelevanceEvaluator()


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
    args = parser.parse_args()

    dataset_file = Path(args.dataset_file)
    if not dataset_file.exists():
        logger.error("dataset_file.not_found", path=str(dataset_file))
        sys.exit(1)

    # Configure tracing so experiment spans are forwarded to Phoenix.
    # Jaeger export is disabled here — experiment spans belong in Phoenix only.
    configure_tracing(
        enabled=_settings.otel_enabled,
        service_name=_settings.otel_service_name,
        project_name=_settings.phoenix_project_name,
        deployment_environment=_settings.otel_deployment_environment,
        phoenix_otlp_endpoint=_settings.otel_phoenix_otlp_endpoint,
        # Always export to Phoenix in the experiment runner: the
        # DocumentRelevanceEvaluator queries retriever spans from Phoenix
        # by trace_id after each task completes.
        phoenix_export=True,
        jaeger_otlp_endpoint=_settings.otel_jaeger_otlp_endpoint,
        jaeger_export=False,
        sample_rate=1.0,  # always sample all eval spans
        console_export=_settings.otel_console_export,
        auto_instrument_haystack=_settings.otel_auto_instrument_haystack,
    )

    # Lazy imports: phoenix.client and pandas are eval-group dependencies
    # not available in the default environment.
    from phoenix.client import Client  # type: ignore[import-not-found]

    client: Client = Client()
    evaluators = [
        *_build_code_evaluators(),
        _build_faithfulness_evaluator(_settings),
        _build_correctness_evaluator(_settings),
        _build_document_relevance_evaluator(_settings),
    ]

    metadata = _collect_experiment_metadata(_settings)

    if args.dry_run:
        logger.info("dry_run.start", **metadata)
        experiment = client.experiments.run_experiment(
            dataset=_get_or_create_dataset(client, dataset_file, args.dataset_name),
            task=_sync_task,
            evaluators=evaluators,
            experiment_metadata=metadata,
            dry_run=1,
        )
        logger.info("dry_run.complete", experiment=str(experiment))
        return

    if args.experiment_id:
        logger.info("evaluate_only.start", experiment_id=args.experiment_id, **metadata)
        existing = client.experiments.get_experiment(experiment_id=args.experiment_id)
        result = client.experiments.evaluate_experiment(
            experiment=existing,
            evaluators=evaluators,
        )
        logger.info(
            "evaluate_only.complete",
            experiment_id=getattr(result, "id", str(result)),
        )
        return

    experiment_name: str = args.experiment_name or f"chatbot-eval-{uuid4().hex[:8]}"
    dataset = _get_or_create_dataset(client, dataset_file, args.dataset_name)

    logger.info("experiment.start", name=experiment_name, **metadata)
    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=_sync_task,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_metadata=metadata,
    )
    logger.info("experiment.complete", experiment_id=getattr(experiment, "id", str(experiment)))


if __name__ == "__main__":
    main()
