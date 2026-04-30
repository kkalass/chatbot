"""Shared value objects and Protocol interfaces for the application layer.

Protocol-based boundaries keep the orchestrator independent of concrete
infrastructure (Ollama, HTTP clients, Chainlit).  All cross-module typed
contracts that don't belong to a single subsystem live here.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

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


@dataclass(frozen=True)
class SourceCitationEvent:
    """Emitted by :class:`CitationTool` when the model successfully cites its sources.

    Carries only the validated chunks — those whose source paths were both
    claimed by the model *and* present in a prior ``search_documents`` result.
    """

    validated: tuple[SourceChunk, ...]


@dataclass(frozen=True)
class ToolCitationEvent:
    """Emitted when an inline quote is validated against a non-retrieval tool call.

    ``result`` carries the raw tool-result JSON from conversation history so the UI
    can render the authoritative data without accessing history directly.
    """

    tool_call_id: str
    tool_name: str
    result: JsonObject


@dataclass(frozen=True)
class RawAssistantText:
    """Complete raw assistant text from the model before inline-quote transformation.

    Emitted by wrapper chat models that transform inline quote markers into
    structured items. Consumers use this payload for conversation-history
    persistence, while user-facing streaming still uses transformed text/events.
    """

    text: str


class SearchResultQuote(BaseModel):
    """Quote emitted for claims grounded in ``search_documents`` output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["search_result"] = "search_result"
    claim: str | None = None
    tool_call_id: str
    source: str
    chunk_id: str
    quote_text: str | None = None


class ToolCallQuote(BaseModel):
    """Quote emitted for claims grounded in non-retrieval tool outputs.

    Only ``tool_call_id`` is used for validation. The authoritative tool name is
    always resolved from conversation history.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tool_call"] = "tool_call"
    tool_call_id: str


type Quote = SearchResultQuote | ToolCallQuote


class QuoteReferenceEvent(BaseModel):
    """Inline reference emitted by the orchestrator for a validated quote."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_number: int = Field(ge=1)
    canonical_key: str


# Union of all events that a tool may emit alongside its JSON result.
type ToolEvent = SourceCitationEvent | ToolCitationEvent

# Union of all items yielded by :meth:`~src.chatbot.app.orchestrator.ChatOrchestrator.process_message`.
# ``str`` items are streamed text chunks; ``ToolEvent`` items carry structured
# metadata (citations, etc.) that the UI renders separately.
type ProcessEvent = str | ToolEvent | QuoteReferenceEvent


type ChatStreamItem = str | list[ToolCallInfo] | Quote | RawAssistantText


@dataclass(frozen=True)
class ToolContext:
    """Snapshot of conversation history passed to every tool execution.

    Frozen at the point of the tool call so tools see a consistent view of
    history even in a concurrent or multi-round agentic loop.
    """

    history: tuple[ChatMessage, ...]


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

    async def execute(
        self, args: JsonObject, context: ToolContext
    ) -> tuple[JsonObject, list[ToolEvent]]:
        """Execute the tool with *args* decoded from the LLM's tool_call.

        Args:
            args: Decoded JSON arguments from the LLM (validated inside the
                tool implementation).
            context: Snapshot of conversation history at the time of this call.
                Tools may inspect past ``search_documents`` results to validate
                claims (e.g. :class:`CitationTool`) but must not mutate history.

        Returns:
            A two-tuple of ``(result, events)`` where ``result`` is the
            ``JsonObject`` forwarded to the model as the tool result and
            ``events`` is a (possibly empty) list of :data:`ToolEvent` items
            emitted to the caller (e.g. :class:`SourceCitationEvent`).
            Values in ``result`` must never contain raw credentials.
        """
        ...


@runtime_checkable
class ChatModel(Protocol):
    """Structural interface for a chat model backend.

    A single :meth:`stream` method handles both plain-text and tool-calling
    turns.  It yields text chunks as they arrive from the model.  If the model
    requests tool calls instead of a text response, a single
    ``list[ToolCallInfo]`` is yielded as the final item; otherwise the stream
    ends after the text chunks. Quote items may also be yielded when a model
    wrapper extracts structured quote payloads from inline markers. Wrappers may
    additionally emit :class:`RawAssistantText` for history persistence.
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
            tools rather than reply with text. ``Quote`` items may appear when
            inline quote extraction is enabled. ``RawAssistantText`` may appear
            as an auxiliary payload for persistence.
        """
        ...


@runtime_checkable
class PromptProfile(Protocol):
    """Structural interface for model-specific prompt and tool-schema adaptation.

    Implementations are selected at composition time based on the target model
    and applied once when wiring the orchestrator.
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
    """Structural boundary between orchestration and retrieval infrastructure.

    Implementations wrap a vector store and an embedding model.  The tool
    layer calls ``retrieve`` and never touches the underlying clients directly.
    """

    async def retrieve(self, query: str) -> list[SourceChunk]:
        """Return ranked, score-filtered chunks relevant to *query*.

        Args:
            query: Natural-language question or search string from the user.

        Returns:
            A list of :class:`SourceChunk` objects sorted by descending score,
            already filtered to the configured similarity threshold.  May be
            empty if no relevant chunks exist.
        """
        ...
