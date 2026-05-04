# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared value objects and Protocol interfaces for the application layer.

Protocol-based boundaries keep the orchestrator independent of concrete
infrastructure (Ollama, HTTP clients, Chainlit).  All cross-module typed
contracts that don't belong to a single subsystem live here.
"""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, assert_never, runtime_checkable

from pydantic import BaseModel, ConfigDict

from src.chatbot.app.prompts import Prompts

# JSON object — the canonical in-memory representation of structured data at
# protocol boundaries.  ``Any`` is intentional: I tried to use a recursive
# type alias but that caused more trouble than it is worth (e.g. .get() calls on dicts were troublesome).
type JsonObject = dict[str, Any]


@dataclass(frozen=True)
class I18nMessage:
    """Localizable message token — a key + interpolation args for UI rendering.

    The ``key`` identifies a message template (defined by the tool as a
    ``StrEnum`` constant); ``args`` carries the interpolation parameters.
    The UI layer resolves ``key`` + ``args`` to a human-readable string —
    keeping i18n concerns out of the tool and orchestrator layers entirely.

    Reusable beyond tool calls — suitable for tool titles, status messages,
    error descriptions, or any text that may need localisation later.

    Args:
        key: A namespaced message key defined by the producing component (e.g.
            ``"retrieval.searching"``).  The UI translation map must contain
            an entry for every key produced at runtime.
        args: Interpolation arguments for the template (e.g.
            ``{"query": "AI in vocational education"}``).
    """

    key: str
    args: JsonObject


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


# ---------------------------------------------------------------------------
# Credential contracts — shared by any tool that requires session-scoped auth.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsernamePasswordCredentials:
    """Username/password pair for a service requiring HTTP Basic-style auth."""

    username: str
    password: str


class CredentialStore(Protocol):
    """Session-scoped key-indexed credential repository.

    Each tool that requires credentials operates under a stable ``key``
    (e.g. ``"vacation_days"``).  The key is also carried in
    :class:`AuthRequiredEvent` so the UI knows which slot to fill after the
    login form is submitted.
    """

    def get_credentials(self, key: str) -> UsernamePasswordCredentials | None:
        """Return stored credentials for *key*, or ``None`` if not present."""
        ...

    def set_credentials(self, key: str, username: str, password: str) -> None:
        """Store a username/password pair under *key*, replacing any previous entry."""
        ...

    def clear_credentials(self, key: str) -> None:
        """Discard stored credentials for *key* (e.g. after an auth failure)."""
        ...


class AuthRequiredException(Exception):
    """Raised by a :class:`Tool` when credentials are required but not available.

    The orchestrator catches this and yields an ``AuthRequiredEvent``, pausing
    the tool-call loop until the UI collects credentials via the login form.
    Any tool can raise this; it is not specific to vacation days.

    Args:
        credential_key: Stable key that identifies the credential slot in the
            session-scoped :class:`CredentialStore`
            (e.g. ``"vacation_days"``).  Passed through to ``AuthRequiredEvent``
            so the UI knows which slot to fill after form submission.
        service_display_name: Localizable name of the service requiring auth,
            passed through to ``AuthRequiredEvent`` for UI display.
    """

    def __init__(self, *, credential_key: str, service_display_name: I18nMessage) -> None:
        super().__init__(str(service_display_name.key))
        self.credential_key = credential_key
        self.service_display_name = service_display_name


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
class ThinkingContent:
    """A complete thinking/reasoning block emitted by a model with extended thinking enabled.

    The adapter strips ``<think>...</think>`` tags from the response stream and
    packages the accumulated content as a single ``ThinkingContent`` item.  The
    orchestrator threads it through to callers as a ``ProcessEvent`` so that
    each consumer (UI, logger, evaluator) can decide independently what to do
    with it.

    Args:
        text: The raw thinking text between the opening and closing tags.
    """

    text: str


type ChatStreamItem = str | list[ToolCallInfo] | ThinkingContent


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
    display_name: I18nMessage  # Human-readable name for UI rendering (resolved via translation map)

    def describe_call(self, args: JsonObject) -> I18nMessage:
        """Return a localizable description of a call with *args* for UI display.

        Implementations should extract the most user-relevant argument(s) and
        return an :class:`I18nMessage` with a ``StrEnum``-defined key and
        the interpolation args.  The UI translation layer resolves the key to
        a human-readable string.
        """
        ...

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


@dataclass(frozen=True)
class ToolCallStarted:
    """Emitted just before a tool call is dispatched.

    Allows the UI to open a progress indicator (e.g. a Chainlit Step) scoped
    to exactly this invocation.  Paired with :class:`ToolCallFinished` which
    carries the same ``call_id``.
    """

    tool_name: str
    call_id: str
    call_description: I18nMessage


@dataclass(frozen=True)
class ToolCallFinished:
    """Emitted after a tool call has been dispatched and its result appended.

    Paired with :class:`ToolCallStarted`; the ``call_id`` is identical so the
    UI can close the matching progress indicator.
    """

    tool_name: str
    call_id: str


@dataclass
class AuthRequiredEvent:
    """Emitted when a tool raises :class:`AuthRequiredException`.

    The orchestrator suspends the tool-call loop and awaits
    :attr:`credential_future`. The UI is expected to:

    1. Show a login form (e.g. :class:`~chainlit.AskElementMessage`).
    2. Store the collected credentials in the session-scoped credential store
       under :attr:`credential_key`.
    3. Set ``credential_future.set_result(True)`` on success, or
       ``set_result(False)`` on cancellation.

    The generator then retries the tool call (on ``True``) or substitutes an
    error result (on ``False``) and continues normally.
    """

    tool_name: str
    credential_key: str
    service_display_name: I18nMessage
    credential_future: asyncio.Future[bool]


# ---------------------------------------------------------------------------
# Citation vocabulary — public value objects emitted by the citation layer
# and consumed by the orchestrator and UI.
# ---------------------------------------------------------------------------


class RawCitation(BaseModel):
    """Marker payload emitted by the model.

    ``ref`` is the citation token of a previously-emitted citable unit; it is
    required for all regular citations. For unsubstantiated claims the model
    emits ``{"kind": "unsubstantiated"}`` without a ``ref``.
    """

    model_config = ConfigDict(frozen=True)

    ref: str | None = None
    kind: str | None = None
    raw_marker_text: str = ""


@dataclass(frozen=True)
class DocumentCitation:
    """Validated citation backed by a retrieved document chunk."""

    raw_marker_text: str
    citation_token: str
    source: str
    chunk_id: str
    content: str
    score: float
    title: str | None = None
    author: str | None = None
    publication_date: str | None = None
    source_url: str | None = None
    page: str | None = None


@dataclass(frozen=True)
class ToolCitation:
    """Validated citation backed by a non-document tool result."""

    raw_marker_text: str
    citation_token: str
    tool_name: str
    result: JsonObject
    display_name: I18nMessage | None = None  # resolved by UI via translation map


type Citation = DocumentCitation | ToolCitation


@dataclass(frozen=True)
class HallucinatedCitation:
    """A ``RawCitation`` that failed validation by the responsible tool.

    The UI decides how (or whether) to surface it. ``raw_marker_text`` is also
    spliced back into the LLM-side history so that the model sees its own
    output verbatim on subsequent turns.
    """

    raw: RawCitation
    reason: str

    @property
    def raw_marker_text(self) -> str:
        return self.raw.raw_marker_text


@dataclass(frozen=True)
class UnsubstantiatedClaim:
    """A ``RawCitation`` with ``kind="unsubstantiated"`` — the model explicitly
    signals that no tool output supports the preceding claim.

    This is *not* a validation failure: it is correct, transparent model
    behaviour. The UI renders it as ``_(unbelegt)_`` inline at the marker
    position. ``raw_marker_text`` is spliced back into the LLM-side history
    so the model sees its own signal on subsequent turns.
    """

    raw: RawCitation

    @property
    def raw_marker_text(self) -> str:
        return self.raw.raw_marker_text


@dataclass(frozen=True)
class NumberedCitation:
    """A ``Citation`` with a stable per-turn reference number assigned by the
    orchestrator (``[N]`` in the rendered text). Reference numbers are reused
    when the same canonical key appears more than once in a turn.
    """

    reference_number: int
    citation: Citation


def canonical_key(citation: Citation) -> str:
    """Stable structural key for citation deduplication and reference reuse.

    The ``citation_token`` is a content-derived hash (or otherwise unique
    per-call identifier) that the model copied verbatim into ``ref``. Using
    it directly as the canonical key makes deduplication trivially correct:
    two citations are the same evidence iff their tokens match.
    """
    match citation:
        case DocumentCitation():
            return f"document:{citation.citation_token}"
        case ToolCitation():
            return f"tool:{citation.citation_token}"
        case _:
            assert_never(citation)


type ProcessEvent = (
    str
    | NumberedCitation
    | HallucinatedCitation
    | UnsubstantiatedClaim
    | ToolCallStarted
    | ToolCallFinished
    | AuthRequiredEvent
    | ThinkingContent
)
