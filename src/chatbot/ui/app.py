"""Chainlit entry point and session lifecycle hooks.

This module is the composition root for the application: it wires together
all infrastructure adapters and the orchestrator, then stores the assembled
graph in session-scoped state.

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

from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.app.prompts import DEFAULT_PROMPTS
from src.chatbot.app.protocols import (
    ProcessEvent,
    QuoteReferenceEvent,
    Retriever,
    SourceChunk,
    SourceCitationEvent,
    Tool,
    ToolCitationEvent,
)
from src.chatbot.config import (
    build_chat_model_config,
    build_retriever_config,
    build_text_embedder_config,
)
from src.chatbot.infrastructure.chat import build_chat_model, build_chat_prompt_profile
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
    build_tool_citation_content,
    build_tool_citation_markdown,
    build_tool_citation_name,
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


def collect_unique_citation_chunks(
    citation_events: list[SourceCitationEvent],
) -> list[SourceChunk]:
    """Flatten and deduplicate citation chunks while preserving first-seen order."""
    seen_chunks: set[tuple[str, str]] = set()
    unique_chunks: list[SourceChunk] = []
    for citation_event in citation_events:
        for chunk in citation_event.validated:
            key = (chunk.source, chunk.chunk_id)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            unique_chunks.append(chunk)
    return unique_chunks


def collect_unique_tool_citations(
    tool_citation_events: list[ToolCitationEvent],
) -> list[ToolCitationEvent]:
    """Deduplicate tool citations while preserving first-seen order."""
    seen_call_ids: set[str] = set()
    unique_tool_citations: list[ToolCitationEvent] = []
    for tool_citation_event in tool_citation_events:
        if tool_citation_event.tool_call_id in seen_call_ids:
            continue
        seen_call_ids.add(tool_citation_event.tool_call_id)
        unique_tool_citations.append(tool_citation_event)
    return unique_tool_citations


def _build_source_elements(unique_chunks: list[SourceChunk]) -> list[cl.Text]:
    """Build sidebar source elements with stable duplicate disambiguation."""
    base_names = [build_citation_name(chunk) for chunk in unique_chunks]
    total_counts = Counter(base_names)
    seen_counts: defaultdict[str, int] = defaultdict(int)
    elements: list[cl.Text] = []
    for chunk, base_name in zip(unique_chunks, base_names, strict=True):
        seen_counts[base_name] += 1
        display_name = base_name
        if total_counts[base_name] > 1:
            display_name = f"{base_name} ({seen_counts[base_name]}/{total_counts[base_name]})"
        elements.append(
            cl.Text(
                name=display_name,
                content=build_citation_content(chunk),
                display="side",
            )
        )
    return elements


def _build_tool_source_elements(tool_citations: list[ToolCitationEvent]) -> list[cl.Text]:
    """Build sidebar source elements for successful (non-error) tool results."""
    elements: list[cl.Text] = []
    for tool_citation in tool_citations:
        if "error" in tool_citation.result:
            continue
        elements.append(
            cl.Text(
                name=build_tool_citation_name(tool_citation),
                content=build_tool_citation_content(tool_citation),
                display="side",
            )
        )
    return elements


def _build_vacation_days_tool() -> VacationDaysTool:
    """Create the vacation-days tool and bind its service/auth collaborators."""
    service = SimulatedVacationDaysAdapter()
    auth = InteractiveVacationDaysAuthSession(
        ask_user=_ask_user,
        service_label="the vacation days service",
    )
    return VacationDaysTool(service=service, auth=auth)


def _build_retriever() -> Retriever:
    """Construct the Qdrant retriever for this session."""
    text_embedder = build_text_embedder(build_text_embedder_config(_settings))
    retriever_config = build_retriever_config(_settings)
    return build_retriever(
        config=retriever_config,
        text_embedder=text_embedder,
    )


def _build_orchestrator() -> ChatOrchestrator:
    """Compose one session-scoped chat orchestrator instance."""
    chat_model_config = build_chat_model_config(_settings)
    prompt_profile = build_chat_prompt_profile(chat_model_config)

    chat_model = build_chat_model(chat_model_config)
    vacation_days_tool = _build_vacation_days_tool()
    retrieval_tool = RetrievalTool(retriever=_build_retriever())
    tools: list[Tool] = [vacation_days_tool, retrieval_tool]
    return ChatOrchestrator(
        model=chat_model,
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
    citation_events: list[SourceCitationEvent],
    tool_citation_events: list[ToolCitationEvent],
) -> None:
    span.set_attributes(
        build_output_attributes(final_response_text, mime_type=OpenInferenceMimeTypeValues.TEXT)
    )
    span.set_attributes(
        build_metadata_attributes(
            {
                "citation_events": len(citation_events),
                "tool_citation_events": len(tool_citation_events),
            }
        )
    )
    span.set_attribute("chat.response.emitted_chars", emitted_chars)
    span.set_attribute(
        "chat.response.preview", to_attribute_text("".join(emitted_chunks), max_chars=600)
    )
    span.set_attribute("chat.response.citation_events", len(citation_events))
    span.set_attribute("chat.response.tool_citation_events", len(tool_citation_events))
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

    # Delay creating the response bubble until the first text token arrives so
    # that interactive auth prompts (AskUserMessage) are not sandwiched between
    # an empty placeholder and the eventual answer.
    response: cl.Message | None = None
    citation_events: list[SourceCitationEvent] = []
    tool_citation_events: list[ToolCitationEvent] = []
    emitted_chars = 0
    emitted_chunks: list[str] = []

    with (
        using_session_attributes(trace_session),
        _tracer.start_as_current_span(SPAN_CHAT_UI_ON_MESSAGE) as span,
    ):
        _trace_request(
            span=span,
            user_text=user_text,
            trace_session=trace_session,
        )

        event: ProcessEvent
        async for event in orchestrator.process_message(user_text):
            match event:
                case str():
                    emitted_chars += len(event)
                    emitted_chunks.append(event)
                    if response is None:
                        response = cl.Message(content="")
                        await response.send()
                    await response.stream_token(event)
                case SourceCitationEvent():
                    citation_events.append(event)
                case ToolCitationEvent():
                    tool_citation_events.append(event)
                case QuoteReferenceEvent():
                    # The model may emit quote markers in a separate paragraph block,
                    # which would render detached "[n]" lines in the streamed text.
                    # Keep references in the sources section/sidebar only.
                    logger.debug(
                        "session.quote_reference_suppressed",
                        reference_number=event.reference_number,
                        canonical_key=event.canonical_key,
                    )
                case _:
                    assert_never(event)

        unique_chunks = collect_unique_citation_chunks(citation_events)
        unique_tool_citations = collect_unique_tool_citations(tool_citation_events)

        if (unique_chunks or unique_tool_citations) and response is None:
            # Ensure citations can still be rendered even if no text token was emitted.
            response = cl.Message(content="")
            await response.send()

        if response is not None:
            if unique_chunks or unique_tool_citations:
                appendix_sections: list[str] = []
                sources_markdown = build_citation_markdown(unique_chunks)
                if sources_markdown:
                    appendix_sections.append(sources_markdown)

                tool_citations_markdown = build_tool_citation_markdown(unique_tool_citations)
                if tool_citations_markdown:
                    appendix_sections.append(tool_citations_markdown)

                if appendix_sections:
                    response.content = (
                        f"{response.content.rstrip()}\n\n{'\n\n'.join(appendix_sections)}"
                    )

                elements = [
                    *_build_source_elements(unique_chunks),
                    *_build_tool_source_elements(unique_tool_citations),
                ]
                response.elements = elements  # pyright: ignore[reportAttributeAccessIssue]
                logger.info("session.sources_displayed", count=len(elements))

            # Finalize the streamed message even when no post-processing payload exists.
            await response.update()

        trace_final_response_text = (
            response.content if response is not None else "".join(emitted_chunks)
        )
        _trace_response(
            span=span,
            final_response_text=trace_final_response_text,
            emitted_chars=emitted_chars,
            emitted_chunks=emitted_chunks,
            citation_events=citation_events,
            tool_citation_events=tool_citation_events,
        )

    logger.info("session.message_handled", length=len(user_text))
