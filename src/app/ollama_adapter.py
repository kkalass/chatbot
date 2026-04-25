"""Ollama backend adapter implementing the ChatModel Protocol.

This module is the only place in the application that imports the ``ollama``
client library.  All orchestration code depends on the
:class:`~src.app.protocols.ChatModel` Protocol, never on this class directly.
"""

from collections.abc import AsyncIterator, Sequence

import structlog
from ollama import AsyncClient, Message
from ollama._types import ChatResponse

from src.app.protocols import ChatMessage

logger = structlog.get_logger(__name__)


class OllamaChatModel:
    """Implements :class:`~src.app.protocols.ChatModel` via the Ollama HTTP API.

    Args:
        client: A configured :class:`ollama.AsyncClient` instance.
        model: The Ollama model name to use for chat generation.
    """

    def __init__(self, client: AsyncClient, model: str) -> None:
        self._client = client
        self._model = model

    def stream_chat(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Convert *messages* to Ollama format and stream the response.

        Yields:
            Non-empty text chunks from the model as they arrive.
        """
        return self._stream(messages)

    async def _stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        ollama_messages: list[Message] = [
            Message(role=msg.role, content=msg.content) for msg in messages
        ]

        logger.debug("ollama.request", model=self._model, message_count=len(ollama_messages))

        response_stream: AsyncIterator[ChatResponse] = await self._client.chat(  # pyright: ignore[reportUnknownMemberType]  # ollama overload typing is partially unknown
            model=self._model,
            messages=ollama_messages,
            stream=True,
        )

        async for chunk in response_stream:
            content = chunk.message.content
            if content:
                yield content
