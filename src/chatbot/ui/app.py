"""Chainlit entry point and session lifecycle hooks.

This module is the composition root for the application: it wires together
all infrastructure adapters and the orchestrator, then stores the assembled
graph in session-scoped state.

Session state keys
------------------
- ``"orchestrator"`` — the :class:`~src.chatbot.app.orchestrator.ChatOrchestrator`.
"""

from typing import assert_never

import chainlit as cl
import structlog

from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.app.protocols import ProcessEvent, Retriever, SourceCitationEvent
from src.chatbot.config import (
    build_chat_model_config,
    build_retriever_config,
    build_text_embedder_config,
)
from src.chatbot.infrastructure.chat import build_chat_model, build_chat_prompt_profile
from src.chatbot.infrastructure.embeddings_text import build_text_embedder
from src.chatbot.infrastructure.retrieval import build_retriever
from src.chatbot.tools.citation.tool import CitationTool
from src.chatbot.tools.retrieval.tool import RetrievalTool
from src.chatbot.tools.vacation_days import (
    InteractiveVacationDaysAuthSession,
    SimulatedVacationDaysAdapter,
    VacationDaysTool,
)
from src.chatbot.ui.citation_view import build_citation_content, build_citation_name
from src.chatbot.ui.logging_config import configure_logging
from src.settings import get_settings

_SESSION_ORCHESTRATOR = "orchestrator"

# Configure logging at module import time (before any logger is used).
_settings = get_settings()
configure_logging(_settings.log_format)

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

    cl.user_session.set(_SESSION_ORCHESTRATOR, orchestrator)  # pyright: ignore[reportUnknownMemberType]
    logger.info("session.started", chat_model=_settings.chat_model)


@cl.on_message  # pyright: ignore[reportUnknownMemberType]  # chainlit decorators are dynamically typed
async def on_message(message: cl.Message) -> None:
    """Forward the user message to the orchestrator and stream the response."""
    raw_orchestrator: object | None = cl.user_session.get(_SESSION_ORCHESTRATOR)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(raw_orchestrator, ChatOrchestrator):
        raise RuntimeError("Chat orchestrator is missing from session state")
    orchestrator = raw_orchestrator

    # Delay creating the response bubble until the first text token arrives so
    # that interactive auth prompts (AskUserMessage) are not sandwiched between
    # an empty placeholder and the eventual answer.
    response: cl.Message | None = None
    citation_events: list[SourceCitationEvent] = []

    event: ProcessEvent
    async for event in orchestrator.process_message(message.content):
        match event:
            case str():
                if response is None:
                    response = cl.Message(content="")
                    await response.send()
                await response.stream_token(event)
            case SourceCitationEvent():
                citation_events.append(event)
            case _:
                assert_never(event)

    if response is not None:
        await response.update()

    # Render validated citation chunks as Chainlit Text elements attached to
    # the response message so the user can inspect the grounding context.
    if citation_events and response is not None:
        seen_chunks: set[tuple[str, str]] = set()
        elements: list[cl.Text] = []
        for ce in citation_events:
            for chunk in ce.validated:
                key = (chunk.source, chunk.chunk_id)
                if key in seen_chunks:
                    continue
                seen_chunks.add(key)
                elements.append(
                    cl.Text(
                        name=build_citation_name(chunk),
                        content=build_citation_content(chunk),
                        display="side",
                    )
                )
        if elements:
            response.elements = elements  # pyright: ignore[reportAttributeAccessIssue]
            await response.update()
            logger.info("session.sources_displayed", count=len(elements))

    logger.info("session.message_handled", length=len(message.content))
