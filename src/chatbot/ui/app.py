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
from opentelemetry import trace

from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.app.protocols import ProcessEvent, Retriever, SourceChunk, SourceCitationEvent
from src.chatbot.config import (
    build_chat_model_config,
    build_retriever_config,
    build_text_embedder_config,
)
from src.chatbot.infrastructure.chat import build_chat_model, build_chat_prompt_profile
from src.chatbot.infrastructure.embeddings_text import build_text_embedder
from src.chatbot.infrastructure.retrieval import build_retriever
from src.chatbot.observability import configure_tracing, to_attribute_text
from src.chatbot.observability.schema import SPAN_CHAT_UI_ON_MESSAGE
from src.chatbot.tools.citation.tool import CitationTool
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
    otlp_endpoint=_settings.otel_exporter_otlp_endpoint,
    sample_rate=_settings.otel_sample_rate,
    console_export=_settings.otel_console_export,
)

logger = structlog.get_logger(__name__)


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


def _collect_unique_citation_chunks(
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
    citation_tool = CitationTool()
    return ChatOrchestrator(
        model=chat_model,
        tools=[vacation_days_tool, retrieval_tool, citation_tool],
        prompt_profile=prompt_profile,
    )


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
    emitted_chars = 0
    emitted_chunks: list[str] = []

    with _tracer.start_as_current_span(SPAN_CHAT_UI_ON_MESSAGE) as span:
        span.set_attribute("chat.session_id", trace_session)
        span.set_attribute("chat.user_message.length", len(user_text))
        span.set_attribute("chat.user_message.preview", to_attribute_text(user_text))

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
                case _:
                    assert_never(event)

        unique_chunks = _collect_unique_citation_chunks(citation_events)

        # Append a compact, deduplicated source list directly to the answer so
        # provenance remains visible even when the side panel is collapsed.
        if response is not None and unique_chunks:
            sources_markdown = build_citation_markdown(unique_chunks)
            if sources_markdown:
                response.content = f"{response.content.rstrip()}\n\n{sources_markdown}"

        # Render validated citation chunks as Chainlit Text elements attached to
        # the response message so the user can inspect the grounding context.
        if unique_chunks and response is not None:
            base_names = [build_citation_name(chunk) for chunk in unique_chunks]
            total_counts = Counter(base_names)
            seen_counts: defaultdict[str, int] = defaultdict(int)
            elements: list[cl.Text] = []
            for chunk, base_name in zip(unique_chunks, base_names, strict=True):
                seen_counts[base_name] += 1
                name = base_name
                if total_counts[base_name] > 1:
                    name = f"{base_name} ({seen_counts[base_name]}/{total_counts[base_name]})"
                elements.append(
                    cl.Text(
                        name=name,
                        content=build_citation_content(chunk),
                        display="side",
                    )
                )
            response.elements = elements  # pyright: ignore[reportAttributeAccessIssue]
            logger.info("session.sources_displayed", count=len(elements))

        if response is not None:
            await response.update()

        span.set_attribute("chat.response.emitted_chars", emitted_chars)
        span.set_attribute(
            "chat.response.preview",
            to_attribute_text("".join(emitted_chunks), max_chars=600),
        )
        span.set_attribute("chat.response.citation_events", len(citation_events))

    logger.info("session.message_handled", length=len(user_text))
