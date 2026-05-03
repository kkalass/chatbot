# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared value objects and Protocol interfaces for the application layer.

Protocol-based boundaries keep the orchestrator independent of concrete
infrastructure (Ollama, HTTP clients, Chainlit).  All cross-module typed
contracts that don't belong to a single subsystem live here.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.chatbot.app.prompts import Prompts

# JSON object — the canonical in-memory representation of structured data at
# protocol boundaries.  ``Any`` is intentional: I tried to use a recursive
# type alias but that caused more trouble than it is worth (e.g. .get() calls on dicts were troublesome).
type JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SourceChunk:
    """A single retrieved chunk of text with provenance metadata.

    Args:
        content: The raw text of the chunk.
        source: Path or identifier of the originating document (displayed in
            citations and used for deduplication).
        score: Similarity score returned by the vector store (higher is better).
        chunk_id: Unique identifier assigned during ingestion (opaque string).
        title: Human-readable document title from sidecar metadata (optional).
        author: Author(s) from sidecar metadata (optional).
        publication_date: Publication date string from sidecar metadata (optional).
        source_url: Canonical URL of the original document (optional).
    """

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

    ``call_id`` is a backend-specific correlation token that ties this request
    to its corresponding ``role="tool"`` result message.  Adapters are
    responsible for minting it (e.g. the tool name for Ollama, a UUID returned
    by the OpenAI API); the orchestrator threads it through opaquely.
    Defaults to ``""`` so test doubles that don't exercise correlation can
    omit it.
    """

    name: str
    arguments: JsonObject  # LLM-generated JSON args have no fixed schema
    call_id: str = ""  # backend-specific; adapter fills, orchestrator threads through


@dataclass(frozen=True)
class ChatMessage:
    """An immutable message in a conversation turn."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | JsonObject  # str for all roles except "tool"; JsonObject for tool results
    tool_calls: tuple[ToolCallInfo, ...] | None = (
        None  # populated for role="assistant" tool-call requests
    )
    tool_call_id: str | None = None  # populated for role="tool" result messages


@dataclass(frozen=True)
class ToolSchema:
    """Information about a tool exposed to the model.

    This is the minimal contract needed by a model implementation to advertise
    available tools.  It separates the model's concern ("what can I call?")
    from the orchestrator's concern ("how do I dispatch and execute?").
    """

    name: str
    description: str
    parameters_schema: JsonObject  # JSON Schema object describing parameters


type ChatStreamItem = str | list[ToolCallInfo]


@runtime_checkable
class Tool(Protocol):
    """Structural interface for an LLM-callable tool.

    All dependencies (user-interaction callbacks, service adapters) are
    injected at construction time — tools are instantiated once per session.
    The orchestrator advertises tool schemas to the model and dispatches
    ``tool_calls`` by name.  Tools never import the orchestrator or any UI
    module.
    """

    schema: ToolSchema  # All metadata needed by the model (name, description, parameters)

    async def execute(self, args: JsonObject) -> JsonObject:
        """Execute the tool with *args* decoded from the LLM's tool_call.

        Returns a structured ``JsonObject`` forwarded to the model as the tool
        result.  Values must never contain raw credentials.
        """
        ...


@runtime_checkable
class ChatModel(Protocol):
    """Structural interface for a chat model backend.

    A single :meth:`stream` method handles both plain-text and tool-calling
    turns.  It yields text chunks as they arrive from the model.  If the model
    requests tool calls instead of a text response, a single
    ``list[ToolCallInfo]`` is yielded as the final item; otherwise the stream
    ends after the text chunks.  Text and tool calls are mutually exclusive
    within one turn.
    """

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ChatStreamItem]:
        """Stream a chat completion, optionally advertising tool schemas.

        Args:
            messages: Full conversation history including the new user turn.
            tools: Tool schemas to advertise.  ``None`` means no tools.

        Yields:
            ``str`` chunks while the model generates a text response, followed
            by a single ``list[ToolCallInfo]`` if the model chose to invoke
            tools rather than reply with text.
        """
        ...


@runtime_checkable
class ModelProfile(Protocol):
    """Structural interface for model-specific adaptation of prompts, tool schemas,
    and adapter-level capabilities.

    Implementations are selected at composition time based on the target model
    and applied once when wiring the orchestrator.
    """

    @property
    def parse_text_tool_calls(self) -> bool:
        """Whether the chat adapter should detect tool calls emitted as JSON text.

        Models that serialise tool invocations as plain JSON in their response
        text (e.g. qwen2.5-coder) require the adapter to buffer and parse that
        text.  Only responses whose first chunk starts with ``{`` or a fenced
        JSON block are buffered; all other responses stream through unchanged.
        """
        ...

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
