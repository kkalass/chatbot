"""Chainlit entry point: session lifecycle, streaming renderer, and observability.

This module is the composition root for the application. Responsibilities:

- **Composition** — wires all infrastructure adapters, tools, and the
  orchestrator into a session-scoped object graph via :func:`on_chat_start`.
- **Streaming renderer** — the :func:`on_message` handler drives the
  orchestrator's event stream, formats text chunks and citation markers into
  Chainlit tokens, and assembles sidebar elements and source-list markdown.
- **Observability** — attaches OpenTelemetry spans and structured log events
  to every request/response cycle.

Session state keys
------------------
- ``"orchestrator"`` — the :class:`~src.chatbot.app.orchestrator.ChatOrchestrator`.
"""

from collections import Counter, defaultdict
from typing import assert_never, cast
from uuid import uuid4

import chainlit as cl
import structlog
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.chatbot.app.citation import (
    Citation,
    DocumentCitation,
    HallucinatedCitation,
    NumberedCitation,
    ToolCitation,
    UnsubstantiatedClaim,
    canonical_key,
)
from src.chatbot.app.orchestrator import ChatOrchestrator, ProcessEvent
from src.chatbot.app.prompts import DEFAULT_PROMPTS
from src.chatbot.app.protocols import Tool
from src.chatbot.config import (
    build_chat_model_config,
    build_retriever_config,
    build_text_embedder_config,
)
from src.chatbot.infrastructure.chat import build_chat_model, build_chat_model_profile
from src.chatbot.infrastructure.embeddings_text import build_text_embedder
from src.chatbot.infrastructure.retrieval import build_retriever
from src.chatbot.observability import configure_tracing, to_attribute_text
from src.chatbot.observability.openinference import (
    build_input_attributes,
    build_metadata_attributes,
    build_output_attributes,
    build_session_attributes,
    build_span_kind_attributes,
    using_session_attributes,
)
from src.chatbot.observability.schema import SPAN_CHAT_UI_ON_MESSAGE
from src.chatbot.tools.retrieval.tool import RetrievalTool
from src.chatbot.tools.vacation_days import (
    InteractiveVacationDaysAuthSession,
    SimulatedVacationDaysAdapter,
    VacationDaysTool,
)
from src.chatbot.ui.citation_view import (
    build_citation_content,
    build_citation_markdown,
    build_citation_name,
)
from src.chatbot.ui.logging_config import configure_logging
from src.settings import get_settings

_SESSION_ORCHESTRATOR = "orchestrator"
_SESSION_TRACE_ID = "trace_session_id"
_tracer = trace.get_tracer(__name__)

# Configure logging at module import time (before any logger is used).
_settings = get_settings()
configure_logging(_settings.log_format)
configure_tracing(
    enabled=_settings.otel_enabled,
    service_name=_settings.otel_service_name,
    project_name=_settings.phoenix_project_name,
    deployment_environment=_settings.otel_deployment_environment,
    phoenix_otlp_endpoint=_settings.otel_phoenix_otlp_endpoint,
    phoenix_export=_settings.otel_export_phoenix,
    jaeger_otlp_endpoint=_settings.otel_jaeger_otlp_endpoint,
    jaeger_export=_settings.otel_export_jaeger,
    sample_rate=_settings.otel_sample_rate,
    console_export=_settings.otel_console_export,
    auto_instrument_haystack=_settings.otel_auto_instrument_haystack,
)

logger = structlog.get_logger(__name__)
_DEFAULT_EVAL_RUN_ID = _settings.eval_run_id if _settings.eval_run_id else str(uuid4())


async def _ask_user(prompt: str) -> str | None:
    """Prompt the user via Chainlit and return their response, or None on cancellation."""
    response = await cl.AskUserMessage(
        content=prompt,
        timeout=120,
    ).send()
    if response is None:
        return None
    value: object = response.get("output", "")  # pyright: ignore[reportUnknownMemberType]
    return str(value).strip() if value else None


def _collect_unique_numbered_citations(
    numbered: list[NumberedCitation],
) -> list[NumberedCitation]:
    """Deduplicate by rendered reference number, keeping first occurrence."""
    seen: set[int] = set()
    unique: list[NumberedCitation] = []
    for nc in numbered:
        if nc.reference_number in seen:
            continue
        seen.add(nc.reference_number)
        unique.append(nc)
    return unique


def _format_text_chunk(
    chunk: str,
    pending_whitespace: str,
) -> tuple[list[str], str]:
    """Return stream tokens for a text chunk while buffering trailing whitespace.

    Trailing whitespace is held back so a following ``[N]`` reference can be
    rendered directly after the preceding sentence without an inserted newline.
    """
    stripped = chunk.rstrip(" \t\r\n")
    if not stripped:
        return [], f"{pending_whitespace}{chunk}"

    tokens: list[str] = []
    if pending_whitespace:
        tokens.append(pending_whitespace)

    trailing_whitespace = chunk[len(stripped) :]
    tokens.append(stripped)

    return tokens, trailing_whitespace


def _format_citation_marker(
    nc: NumberedCitation,
    pending_whitespace: str,
) -> tuple[list[str], str]:
    """Return a ``[n]`` token while keeping trailing whitespace buffered.

    This avoids inserting blank lines between consecutive citation references
    when the model emits multiple marker blocks separated by newlines.
    """
    return [f"[{nc.reference_number}]"], pending_whitespace


def _build_side_elements(unique: list[NumberedCitation]) -> list[cl.Text]:
    """Build sidebar elements with stable duplicate-label disambiguation."""
    base_names = [build_citation_name(nc) for nc in unique]
    total_counts = Counter(base_names)
    seen_counts: defaultdict[str, int] = defaultdict(int)
    elements: list[cl.Text] = []
    for nc, base_name in zip(unique, base_names, strict=True):
        seen_counts[base_name] += 1
        display_name = base_name
        if total_counts[base_name] > 1:
            display_name = f"{base_name} ({seen_counts[base_name]}/{total_counts[base_name]})"
        elements.append(
            cl.Text(
                name=display_name,
                content=build_citation_content(nc),
                display="side",
            )
        )
    return elements


def _has_renderable_side_element(citation: Citation) -> bool:
    """Side panels suppress tool citations carrying error payloads."""
    match citation:
        case DocumentCitation():
            return True
        case ToolCitation():
            return "error" not in citation.result


def _build_vacation_days_tool() -> VacationDaysTool:
    """Create the vacation-days tool and bind its service/auth collaborators."""
    service = SimulatedVacationDaysAdapter()
    auth = InteractiveVacationDaysAuthSession(
        ask_user=_ask_user,
        service_label="the vacation days service",
    )
    return VacationDaysTool(service=service, auth=auth)


def _build_retrieval_tool() -> RetrievalTool:
    """Create the retrieval tool and bind its retriever infrastructure."""
    text_embedder = build_text_embedder(build_text_embedder_config(_settings))
    retriever_config = build_retriever_config(_settings)
    retriever = build_retriever(
        config=retriever_config,
        text_embedder=text_embedder,
    )
    return RetrievalTool(retriever=retriever)


def _build_orchestrator() -> ChatOrchestrator:
    """Compose one session-scoped chat orchestrator instance."""
    chat_model_config = build_chat_model_config(_settings)
    prompt_profile = build_chat_model_profile(chat_model_config)

    chat_model = build_chat_model(
        chat_model_config, parse_text_tool_calls=prompt_profile.parse_text_tool_calls
    )
    vacation_days_tool = _build_vacation_days_tool()
    retrieval_tool = _build_retrieval_tool()
    tools: list[Tool] = [vacation_days_tool, retrieval_tool]
    return ChatOrchestrator.create(
        chat_model,
        tools=tools,
        prompt_profile=prompt_profile,
        prompts=DEFAULT_PROMPTS,
    )


def _trace_request(
    *,
    span: trace.Span,
    user_text: str,
    trace_session: str,
) -> None:
    retrieval_version = (
        _settings.eval_retrieval_version
        if _settings.eval_retrieval_version is not None
        else f"top_k={_settings.retrieval_top_k};score_threshold={_settings.retrieval_score_threshold}"
    )
    metadata: dict[str, object] = {
        "run_id": _DEFAULT_EVAL_RUN_ID,
        "trace_session_id": trace_session,
        "model_name": _settings.chat_model,
        "retrieval_version": retrieval_version,
        "temperature": _settings.model_temperature,
        "seed": _settings.model_seed,
        "environment": _settings.eval_environment,
    }
    optional_metadata: dict[str, object | None] = {
        "evaluation_name": _settings.eval_name,
        "candidate_id": _settings.eval_candidate_id,
        "prompt_version_answer": _settings.eval_prompt_version_answer,
        "prompt_version_citation": _settings.eval_prompt_version_citation,
        "corpus_version": _settings.eval_corpus_version,
        "dataset_version": _settings.eval_dataset_version,
    }
    metadata.update({key: value for key, value in optional_metadata.items() if value is not None})

    span.set_attributes(build_span_kind_attributes(OpenInferenceSpanKindValues.CHAIN))
    span.set_attributes(build_session_attributes(trace_session))
    span.set_attributes(
        build_input_attributes(user_text, mime_type=OpenInferenceMimeTypeValues.TEXT)
    )
    span.set_attributes(build_metadata_attributes(metadata))
    span.set_attribute("chat.session_id", trace_session)
    span.set_attribute("chat.user_message.length", len(user_text))
    span.set_attribute("chat.user_message.preview", to_attribute_text(user_text))


def _trace_response(
    *,
    span: trace.Span,
    final_response_text: str,
    emitted_chars: int,
    emitted_chunks: list[str],
    numbered: list[NumberedCitation],
    hallucinated: list[HallucinatedCitation],
) -> None:
    span.set_attributes(
        build_output_attributes(final_response_text, mime_type=OpenInferenceMimeTypeValues.TEXT)
    )
    span.set_attributes(
        build_metadata_attributes(
            {
                "numbered_citations": len(numbered),
                "hallucinated_citations": len(hallucinated),
            }
        )
    )
    span.set_attribute("chat.response.emitted_chars", emitted_chars)
    span.set_attribute(
        "chat.response.preview", to_attribute_text("".join(emitted_chunks), max_chars=600)
    )
    span.set_attribute("chat.response.numbered_citations", len(numbered))
    span.set_attribute("chat.response.hallucinated_citations", len(hallucinated))
    span.set_status(StatusCode.OK)


@cl.on_chat_start  # pyright: ignore[reportUnknownMemberType]  # chainlit decorators are dynamically typed
async def on_chat_start() -> None:
    """Compose and store one :class:`ChatOrchestrator` per user session."""
    orchestrator = _build_orchestrator()
    trace_session_id = str(uuid4())

    cl.user_session.set(_SESSION_ORCHESTRATOR, orchestrator)  # pyright: ignore[reportUnknownMemberType]
    cl.user_session.set(_SESSION_TRACE_ID, trace_session_id)  # pyright: ignore[reportUnknownMemberType]
    logger.info(
        "session.started", chat_model=_settings.chat_model, trace_session_id=trace_session_id
    )


@cl.on_message  # pyright: ignore[reportUnknownMemberType]  # chainlit decorators are dynamically typed
async def on_message(message: cl.Message) -> None:
    """Forward the user message to the orchestrator and stream the response."""
    raw_orchestrator: object | None = cl.user_session.get(_SESSION_ORCHESTRATOR)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(raw_orchestrator, ChatOrchestrator):
        raise RuntimeError("Chat orchestrator is missing from session state")
    orchestrator = raw_orchestrator

    raw_trace_session_id = cl.user_session.get(_SESSION_TRACE_ID)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    trace_session_id = cast(object | None, raw_trace_session_id)
    trace_session = str(trace_session_id) if trace_session_id is not None else "unknown"

    user_text = str(message.content)

    response: cl.Message | None = None
    numbered: list[NumberedCitation] = []
    hallucinated: list[HallucinatedCitation] = []
    emitted_chars = 0
    emitted_chunks: list[str] = []
    pending_whitespace = ""

    with (
        using_session_attributes(trace_session),
        _tracer.start_as_current_span(SPAN_CHAT_UI_ON_MESSAGE) as span,
    ):
        _trace_request(span=span, user_text=user_text, trace_session=trace_session)

        event: ProcessEvent

        async def _stream_response_token(token: str) -> None:
            nonlocal response, emitted_chars
            if not token:
                return
            emitted_chars += len(token)
            emitted_chunks.append(token)
            if response is None:
                response = cl.Message(content="")
                await response.send()
            await response.stream_token(token)

        async for event in orchestrator.process_message(user_text):
            match event:
                case str():
                    tokens, pending_whitespace = _format_text_chunk(event, pending_whitespace)
                    for token in tokens:
                        await _stream_response_token(token)
                case NumberedCitation():
                    numbered.append(event)
                    tokens, pending_whitespace = _format_citation_marker(event, pending_whitespace)
                    for token in tokens:
                        await _stream_response_token(token)
                    logger.debug(
                        "session.numbered_citation_rendered",
                        reference_number=event.reference_number,
                        canonical_key=canonical_key(event.citation),
                    )
                case HallucinatedCitation():
                    hallucinated.append(event)
                    logger.info(
                        "session.hallucinated_citation",
                        reason=event.reason,
                        tool_call_id=event.raw.tool_call_id,
                    )
                case UnsubstantiatedClaim():
                    tokens, pending_whitespace = _format_text_chunk(
                        # TODO: i18n: we emit "unbelegt" in german; consider making this configurable or part of the prompt instructions.
                        " _(unbelegt)_",
                        pending_whitespace,
                    )
                    for token in tokens:
                        await _stream_response_token(token)
                    logger.debug("session.unsubstantiated_claim")
                case _:
                    assert_never(event)

        if pending_whitespace:
            await _stream_response_token(pending_whitespace)
            pending_whitespace = ""

        unique_numbered = _collect_unique_numbered_citations(numbered)
        renderable = [nc for nc in unique_numbered if _has_renderable_side_element(nc.citation)]

        if unique_numbered and response is None:
            response = cl.Message(content="")
            await response.send()

        if response is not None:
            sources_markdown = build_citation_markdown(unique_numbered)
            if sources_markdown:
                response.content = f"{response.content.rstrip()}\n\n{sources_markdown}"

            if renderable:
                response.elements = _build_side_elements(renderable)  # pyright: ignore[reportAttributeAccessIssue]
                logger.info("session.sources_displayed", count=len(renderable))

            await response.update()

        trace_final_response_text = (
            response.content if response is not None else "".join(emitted_chunks)
        )
        _trace_response(
            span=span,
            final_response_text=trace_final_response_text,
            emitted_chars=emitted_chars,
            emitted_chunks=emitted_chunks,
            numbered=unique_numbered,
            hallucinated=hallucinated,
        )

    logger.info("session.message_handled", length=len(user_text))
