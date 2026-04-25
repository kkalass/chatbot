"""Chat orchestration: conversation history management and response streaming.

The orchestrator is the single entry point for the UI layer.  It depends
exclusively on the :class:`~src.app.protocols.ChatModel` Protocol — no
concrete infrastructure is imported here.
"""

from collections.abc import AsyncIterator

import structlog

from src.app.protocols import ChatMessage, ChatModel

logger = structlog.get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer questions accurately and concisely. "
    "If you are unsure about something, say so rather than guessing."
)


class ChatOrchestrator:
    """Manages per-session conversation history and streams model responses.

    Constructed once per chat session and stored in session-scoped state.
    The ``model`` dependency is injected at construction time to keep this
    class infrastructure-agnostic and fully testable with any :class:`ChatModel`
    implementation.
    """

    def __init__(
        self,
        model: ChatModel,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._model = model
        self._system_prompt = system_prompt
        self._history: list[ChatMessage] = []

    async def process_message(self, user_text: str) -> AsyncIterator[str]:
        """Append *user_text* to history, stream the model response, and
        record the full assistant reply once the stream is exhausted.

        Yields:
            Successive text chunks of the assistant response.
        """
        self._history.append(ChatMessage(role="user", content=user_text))

        log = logger.bind(user_message=user_text[:120])
        log.info("chat.request")

        chunks: list[str] = []
        async for chunk in self._model.stream_chat(self._build_messages()):
            chunks.append(chunk)
            yield chunk

        full_response = "".join(chunks)
        self._history.append(ChatMessage(role="assistant", content=full_response))
        log.info("chat.response", response_length=len(full_response))

    def _build_messages(self) -> list[ChatMessage]:
        """Prepend the system prompt to the conversation history."""
        return [ChatMessage(role="system", content=self._system_prompt), *self._history]
