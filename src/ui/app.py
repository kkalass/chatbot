"""Chainlit entry point and session lifecycle hooks.

This module is the composition root for the application: it wires together
all infrastructure adapters and the orchestrator, then stores the assembled
graph in session-scoped state.

Session state keys
------------------
- ``"orchestrator"`` — the :class:`~src.app.orchestrator.ChatOrchestrator`.
"""

import chainlit as cl
import structlog
from ollama import AsyncClient

from src.app.ollama_adapter import OllamaChatModel
from src.app.orchestrator import ChatOrchestrator
from src.app.protocols import ChatModel
from src.config.settings import get_settings
from src.tools.vacation_days import (InteractiveVacationDaysAuthSession,
                                     SimulatedVacationDaysAdapter,
                                     VacationDaysTool)
from src.ui.logging_config import configure_logging

_SESSION_ORCHESTRATOR = "orchestrator"

# Configure logging at module import time (before any logger is used).
_settings = get_settings()
configure_logging(_settings.log_format)

logger = structlog.get_logger(__name__)


async def _ask_user(prompt: str) -> str | None:
    """Prompt the user via Chainlit and return their response, or None on cancellation."""
    response = await cl.AskUserMessage(  # pyright: ignore[reportUnknownMemberType]
        content=prompt,
        timeout=120,
    ).send()
    if response is None:
        return None
    value: object = response.get("output", "")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    return str(value).strip() if value else None


def _build_chat_model() -> ChatModel:
    """Create the chat-model adapter used by the session orchestrator."""
    client = AsyncClient(host=_settings.ollama_base_url)
    return OllamaChatModel(client=client, model=_settings.chat_model)


def _build_vacation_days_tool() -> VacationDaysTool:
    """Create the vacation-days tool and bind its service/auth collaborators."""
    service = SimulatedVacationDaysAdapter()
    auth = InteractiveVacationDaysAuthSession(
        ask_user=_ask_user,
        service_label="the vacation days service",
    )
    return VacationDaysTool(service=service, auth=auth)


def _build_orchestrator() -> ChatOrchestrator:
    """Compose one session-scoped chat orchestrator instance."""
    chat_model = _build_chat_model()
    vacation_days_tool = _build_vacation_days_tool()
    return ChatOrchestrator(model=chat_model, tools=[vacation_days_tool])


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
    async for chunk in orchestrator.process_message(message.content):
        if response is None:
            response = cl.Message(content="")
            await response.send()  # pyright: ignore[reportUnknownMemberType]
        await response.stream_token(chunk)  # pyright: ignore[reportUnknownMemberType]

    if response is None:
        response = cl.Message(content="")
        await response.send()  # pyright: ignore[reportUnknownMemberType]
    else:
        await response.update()  # pyright: ignore[reportUnknownMemberType]

    logger.info("session.message_handled", length=len(message.content))
