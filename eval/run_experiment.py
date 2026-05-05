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
        base_url = settings.eval_judge_base_url or "http://localhost:11434/v1"
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


def has_warnings(output: dict[str, object]) -> bool:
    """Check whether the answer contains at least one unsubstantiated claim.

    A non-zero count indicates the model made claims that were not covered by
    any citation token in the response — a signal of incomplete attribution.
    """
    count = output.get("unsubstantiated_claim_count", 0)
    return isinstance(count, int) and count > 0


def has_hallucinations(output: dict[str, object]) -> bool:
    """Check whether the answer contains at least one hallucinated citation.

    A non-zero count means the model referenced a citation token that could
    not be resolved to any retrieved chunk — a signal of fabricated sourcing.
    """
    count = output.get("hallucinated_citation_count", 0)
    return isinstance(count, int) and count > 0


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

    client: Client = Client()
    evaluators = [
        has_citations,
        is_non_empty,
        has_warnings,
        has_hallucinations,
        _build_faithfulness_evaluator(_settings),
    ]

    if args.dry_run:
        logger.info("dry_run.start")
        experiment = client.experiments.run_experiment(
            dataset=_get_or_create_dataset(client, dataset_file, args.dataset_name),
            task=_sync_task,
            evaluators=evaluators,
            dry_run=1,
        )
        logger.info("dry_run.complete", experiment=str(experiment))
        return

    experiment_name: str = args.experiment_name or f"chatbot-eval-{uuid4().hex[:8]}"
    dataset = _get_or_create_dataset(client, dataset_file, args.dataset_name)

    logger.info("experiment.start", name=experiment_name)
    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=_sync_task,
        evaluators=evaluators,
        experiment_name=experiment_name,
    )
    logger.info("experiment.complete", experiment_id=getattr(experiment, "id", str(experiment)))


if __name__ == "__main__":
    main()
