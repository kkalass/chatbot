"""Chainlit entry point and session lifecycle hooks.

This module is the composition root for the application: it wires together
the infrastructure adapter (:class:`~src.app.ollama_adapter.OllamaChatModel`)
and the orchestrator (:class:`~src.app.orchestrator.ChatOrchestrator`) and
stores the assembled graph in session-scoped state.

All credential and session handling rules from the architecture doc apply:
per-user credentials are never sourced from env vars and are not managed here
in Phase 1 (no auth-protected tools yet).
"""

import chainlit as cl
import structlog
from ollama import AsyncClient

from src.app.ollama_adapter import OllamaChatModel
from src.app.orchestrator import ChatOrchestrator
from src.config.settings import get_settings
from src.ui.logging_config import configure_logging

_SESSION_KEY = "orchestrator"

# Configure logging at module import time (before any logger is used).
_settings = get_settings()
configure_logging(_settings.log_format)

logger = structlog.get_logger(__name__)


@cl.on_chat_start  # pyright: ignore[reportUnknownMemberType]  # chainlit decorators are dynamically typed
async def on_chat_start() -> None:
    """Compose and store one :class:`ChatOrchestrator` per user session."""
    client = AsyncClient(host=_settings.ollama_base_url)
    model = OllamaChatModel(client=client, model=_settings.chat_model)
    orchestrator = ChatOrchestrator(model=model)

    cl.user_session.set(_SESSION_KEY, orchestrator)  # pyright: ignore[reportUnknownMemberType]  # chainlit user_session has no type stubs
    logger.info("session.started", chat_model=_settings.chat_model)


@cl.on_message  # pyright: ignore[reportUnknownMemberType]  # chainlit decorators are dynamically typed
async def on_message(message: cl.Message) -> None:
    """Forward the user message to the orchestrator and stream the response."""
    raw_orchestrator: object | None = cl.user_session.get(_SESSION_KEY)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # chainlit user_session has no type stubs
    if not isinstance(raw_orchestrator, ChatOrchestrator):
        raise RuntimeError("Chat orchestrator is missing from session state")
    orchestrator = raw_orchestrator

    # Respond immediately to acknowledge receipt of the user message and start the stream.
    # The client will display a typing indicator until the first chunk is received.
    response = cl.Message(content="")
    await response.send()

    async for chunk in orchestrator.process_message(message.content):
        await response.stream_token(chunk)

    await response.update()
