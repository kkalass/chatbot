"""Shared value objects and Protocol interfaces for the application layer.

Protocol-based boundaries keep the orchestrator independent of concrete
infrastructure (Ollama, HTTP clients, Chainlit). All cross-module typed
contracts that don't belong to a single subsystem live here.

Citation-specific types and the ``CiteableTool`` extension live in
:mod:`src.chatbot.app.citation` so that this module is free of citation
internals.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.chatbot.app.prompts import Prompts

# JSON object — the canonical in-memory representation of structured data at
# protocol boundaries. ``Any`` is intentional: a recursive type alias caused
# more trouble than it's worth (e.g. ``.get()`` on dicts).
type JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SourceChunk:
    """A single retrieved chunk of text with provenance metadata."""

    content: str
    source: str
    score: float
    chunk_id: str
    title: str | None = None
    author: str | None = None
    publication_date: str | None = None
    source_url: str | None = None
    page: str | None = None


@dataclass(frozen=True)
class ToolCallInfo:
    """A single tool invocation requested by the model.

    ``call_id`` is a backend-specific correlation token threading from the
    ``tool_calls`` request to the matching ``role="tool"`` result message.
    Adapters mint it (e.g. tool name for Ollama, UUID for OpenAI); the
    orchestrator is purely opaque to its format.
    """

    name: str
    arguments: JsonObject
    call_id: str = ""


@dataclass(frozen=True)
class ChatMessage:
    """An immutable wire-level chat message handed to a ``ChatModel``.

    Produced by the citation layer from
    :data:`~src.chatbot.app.citation.messages.CitationLayerMessage` entries.
    ``content`` is ``str`` for every role after the citation layer pre-computes
    ``llm_content``; ``JsonObject`` remains permitted only for backward-compat
    of the wire encoding inside the Ollama adapter.
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | JsonObject
    tool_calls: tuple[ToolCallInfo, ...] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolSchema:
    """Information about a tool exposed to the model."""

    name: str
    description: str
    parameters_schema: JsonObject


type ChatStreamItem = str | list[ToolCallInfo]


@runtime_checkable
class Tool(Protocol):
    """Structural interface for an LLM-callable tool.

    All dependencies (user-interaction callbacks, service adapters) are
    injected at construction time — tools are instantiated once per session.
    The orchestrator advertises tool schemas to the model and dispatches
    ``tool_calls`` by name. Tools never import the orchestrator or any UI
    module.

    Tools that may be cited by the model implement the
    :class:`~src.chatbot.app.citation.citeable_tool.CiteableTool` extension of
    this Protocol; non-citeable tools (e.g. side-effecting actions) implement
    only ``Tool``.
    """

    schema: ToolSchema

    async def execute(self, args: JsonObject) -> JsonObject:
        """Execute the tool with *args* decoded from the LLM's tool_call.

        Returns a structured ``JsonObject`` forwarded to the model as the tool
        result. Values must never contain raw credentials.
        """
        ...


@runtime_checkable
class ChatModel(Protocol):
    """Structural interface for a chat model backend.

    A single :meth:`stream` method handles both plain-text and tool-calling
    turns. It yields text chunks as they arrive from the model. If the model
    requests tool calls instead of a text response, a single
    ``list[ToolCallInfo]`` is yielded as the final item; otherwise the stream
    ends after the text chunks.

    A base ``ChatModel`` is **citation-agnostic** — it never interprets marker
    tokens. Marker parsing and citation validation are the job of
    :class:`~src.chatbot.app.citation.layer.CitationLayer`, which decorates a
    base ``ChatModel``.
    """

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ChatStreamItem]:
        """Stream a chat completion, optionally advertising tool schemas."""
        ...


@runtime_checkable
class PromptProfile(Protocol):
    """Structural interface for model-specific prompt and tool-schema adaptation.

    Implementations are selected at composition time based on the target model
    and applied once when wiring the orchestrator. ``PromptProfile`` is
    explicitly **not** a citation concern; it adjusts the base system prompt
    and tool schemas, both of which the orchestrator owns.
    """

    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        """Return model-adjusted prompt templates."""
        ...

    def adjust_tool_description(self, tool_name: str, description: str) -> str:
        """Return model-adjusted tool description text."""
        ...

    def adjust_parameter_schema(self, tool_name: str, schema: JsonObject) -> JsonObject:
        """Return model-adjusted JSON schema for tool parameters."""
        ...


class Retriever(Protocol):
    """Structural boundary between orchestration and retrieval infrastructure."""

    async def retrieve(self, query: str) -> list[SourceChunk]:
        """Return ranked, score-filtered chunks relevant to *query*."""
        ...
