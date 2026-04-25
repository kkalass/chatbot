"""Shared value objects and Protocol interfaces for the application layer.

Protocol-based boundaries keep the orchestrator independent of concrete
infrastructure (Ollama, HTTP clients).  All cross-module typed contracts
that don't belong to a single subsystem live here.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    """An immutable message in a conversation turn."""

    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class ChatModel(Protocol):
    """Structural interface for a streaming chat model backend.

    Implementations must accept the full conversation history and yield
    response text incrementally.  The orchestrator never imports a concrete
    model class — it depends only on this Protocol.
    """

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[str]:
        """Stream a chat completion for the given message history.

        Args:
            messages: Full conversation history including the system prompt
                and all prior turns.

        Yields:
            Successive text chunks of the assistant response.
        """
        ...
