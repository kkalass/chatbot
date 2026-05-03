"""``CitationLayer`` — decorator around ``ChatModel`` owning all citation concerns.

Three responsibilities:
1. Augment the system prompt with citation instructions assembled from the
   registered ``CiteableTool``s.
2. Parse marker blocks from the inner model's text stream into typed
   :class:`RawCitation`\\ s.
3. Validate each parsed ``RawCitation`` against the responsible
   ``CiteableTool`` and yield :class:`Citation` (success) or
   :class:`HallucinatedCitation` (failure) to the caller.

The orchestrator depends on ``CitationLayer`` (not on the underlying
``ChatModel``) and consumes a stream of
``str | list[ToolCallInfo] | Citation | HallucinatedCitation``.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import assert_never

import structlog
from opentelemetry import trace

from src.chatbot.app.citation._parser import (
    DEFAULT_MAX_QUOTE_BLOCK_CHARS,
    CitationStreamParser,
)
from src.chatbot.app.citation.citeable_tool import CiteableTool
from src.chatbot.app.citation.context import build_citation_context
from src.chatbot.app.citation.messages import (
    CitationLayerAssistantMessage,
    CitationLayerMessage,
    CitationLayerSystemMessage,
    CitationLayerToolMessage,
    CitationLayerUserMessage,
)
from src.chatbot.app.citation.models import (
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    Citation,
    DocumentRawCitation,
    HallucinatedCitation,
    RawCitation,
    ToolRawCitation,
)
from src.chatbot.app.protocols import (
    ChatMessage,
    ChatModel,
    JsonObject,
    ToolCallInfo,
    ToolSchema,
)

logger = structlog.get_logger(__name__)

type CitationLayerStreamItem = str | list[ToolCallInfo] | Citation | HallucinatedCitation


_REASON_NO_TOOL_CALL = "no prior tool call with that tool_call_id"
_REASON_TOOL_NOT_CITEABLE = "tool is not registered as a CiteableTool"
_REASON_TOOL_REJECTED = "validate_and_enrich returned None"


_CITATION_HEADER = f"""## Inline Citations

Whenever a statement in your answer is supported by a specific tool output,
emit a structured citation object **immediately after** that statement using
the exact marker tokens below. The markers must not appear anywhere else in
your response.

Marker tokens (use exactly):
- start: {QUOTE_START_MARKER}
- end:   {QUOTE_END_MARKER}

Each marker block contains exactly one strict JSON object describing which
prior tool result supports the immediately preceding sentence.

Per-tool citation formats:
"""

_CITATION_GENERAL_RULES = """

General rules:
- Emit a citation marker **after every individual sentence** whose content is
  grounded — do not summarise multiple sentences into a single end-of-paragraph
  marker.
- Emit exactly one JSON object per marker block.
- Only use values (tool_call_id, source, chunk_id) that appear verbatim in
  prior assistant tool calls and tool results in the conversation context.
- Never invent IDs (no timestamps, suffixes, prefixes, or reformatted variants).
- If an exact tool_call_id is not visible in the conversation context, do not
  emit a marker.
- Logical inferences and transitions derived from cited material do not require
  their own marker — only direct factual claims do.
- If you make a factual claim that you cannot back with a citation marker,
  mark it inline with **!UNBELEGT!** immediately after the claim.
- Keep all normal user-facing answer text outside the markers."""


_USER_REMINDER = f"""Reminder: when your answer uses tool outputs, emit inline
citation markers immediately after the supported sentence — one marker per
sentence, not one per paragraph. Use exactly the marker tokens
{QUOTE_START_MARKER} and {QUOTE_END_MARKER}; do not use any marker variants.
Copy tool_call_id, source, and chunk_id values exactly from prior tool calls
and tool results in the conversation context. Never invent IDs or append
suffixes. If the exact ID is not visible, emit no marker.
Never emit a standalone marker list block. A marker is only valid if it appears
immediately after the exact sentence it supports.

!!!IMPORTANT!!!
Every factual claim in your answer must either:
  (a) be immediately followed by a citation marker referencing the exact tool
      output whose content supports it, OR
  (b) be marked **!UNBELEGT!** inline if no tool output contains the
      information.
There is no third option: do not state facts without a marker or a !UNBELEGT!
flag. Logical inferences and transitional sentences derived from cited material
are exempt. Do not append markers in a separate trailing citation section.
Place each marker at the point of use, directly after the supported sentence.

The actual user message is:

"""


def _to_chat_message(msg: CitationLayerMessage) -> ChatMessage:
    """Project a citation-layer message to the wire-level ``ChatMessage``."""
    match msg:
        case CitationLayerSystemMessage():
            return ChatMessage(role="system", content=msg.llm_content)
        case CitationLayerUserMessage():
            return ChatMessage(role="user", content=msg.llm_content)
        case CitationLayerAssistantMessage():
            return ChatMessage(
                role="assistant",
                content=msg.llm_content,
                tool_calls=msg.tool_calls,
            )
        case CitationLayerToolMessage():
            return ChatMessage(
                role="tool",
                content=msg.llm_content,
                tool_call_id=msg.tool_call_id,
            )
        case _:
            assert_never(msg)


def _splice_assistant_llm_content(
    parts: Sequence[str | Citation | HallucinatedCitation],
) -> str:
    """Reconstruct the LLM-side text by replacing each citation by its raw marker."""
    pieces: list[str] = []
    for part in parts:
        if isinstance(part, str):
            pieces.append(part)
        else:
            pieces.append(part.raw_marker_text)
    return "".join(pieces)


class CitationLayer:
    """Citation decorator around a :class:`~src.chatbot.app.protocols.ChatModel`.

    Composition: instantiate once per session with the inner ``ChatModel`` and
    the explicit list of ``CiteableTool``s. The same ``CiteableTool`` instances
    are also passed to the orchestrator's tool registry so that tool dispatch
    and citation validation share a single source of truth.

    Args:
        model: Inner chat model that produces text and tool calls; must not
            interpret marker tokens itself.
        citeable_tools: All ``CiteableTool``s whose results may be cited by
            the model. Names must be unique.
        max_quote_block_chars: Safety limit on a single marker block so a
            runaway model cannot exhaust memory.
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        citeable_tools: Sequence[CiteableTool],
        max_quote_block_chars: int = DEFAULT_MAX_QUOTE_BLOCK_CHARS,
    ) -> None:
        self._model = model
        self._max_quote_block_chars = max_quote_block_chars
        self._tools_by_name: dict[str, CiteableTool] = {}
        for tool in citeable_tools:
            name = tool.schema.name
            if name in self._tools_by_name:
                raise ValueError(f"Duplicate CiteableTool name: {name!r}")
            self._tools_by_name[name] = tool

    # ------------------------------------------------------------------
    # Factory methods — orchestrator calls these to build history entries.
    # ------------------------------------------------------------------

    def make_system_message(self, adjusted_base_prompt: str) -> CitationLayerSystemMessage:
        """Append citation instructions to the orchestrator-supplied base prompt."""
        fragments = [
            tool.cite_instructions().prompt_fragment for tool in self._tools_by_name.values()
        ]
        if not fragments:
            return CitationLayerSystemMessage(llm_content=adjusted_base_prompt)
        joined_fragments = "\n\n".join(fragments)
        citation_section = f"{_CITATION_HEADER}\n{joined_fragments}{_CITATION_GENERAL_RULES}"
        return CitationLayerSystemMessage(
            llm_content=f"{adjusted_base_prompt}\n\n{citation_section}"
        )

    def make_user_message(self, user_text: str) -> CitationLayerUserMessage:
        """Prepend the per-turn citation reminder to *user_text*.

        Behavior preservation: the reminder block is identical in content and
        position to today's, with ``user_text`` no longer carrying citation
        framing of its own.
        """
        return CitationLayerUserMessage(llm_content=f"{_USER_REMINDER}{user_text}\n")

    def make_assistant_message(
        self,
        parts: Sequence[str | Citation | HallucinatedCitation],
        *,
        tool_calls: Sequence[ToolCallInfo] | None = None,
    ) -> CitationLayerAssistantMessage:
        """Build an assistant message and pre-compute its LLM-side content."""
        return CitationLayerAssistantMessage(
            parts=tuple(parts),
            llm_content=_splice_assistant_llm_content(parts),
            tool_calls=tuple(tool_calls) if tool_calls else None,
        )

    def make_tool_message(
        self,
        tool_call_id: str,
        tool_name: str,
        result: JsonObject,
    ) -> CitationLayerToolMessage:
        """Build a tool-result message; uses ``CiteableTool.format_for_history`` when available."""
        tool = self._tools_by_name.get(tool_name)
        if tool is not None:
            llm_content = tool.format_for_history(result)
        else:
            llm_content = json.dumps(result, ensure_ascii=False)
        return CitationLayerToolMessage(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result,
            llm_content=llm_content,
        )

    # ------------------------------------------------------------------
    # Streaming.
    # ------------------------------------------------------------------

    def stream(
        self,
        history: Sequence[CitationLayerMessage],
        *,
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[CitationLayerStreamItem]:
        """Stream a chat completion, validating inline citations as they arrive."""
        chat_messages = [_to_chat_message(msg) for msg in history]
        history_tuple = tuple(history)
        upstream = self._model.stream(chat_messages, tools)
        parser = CitationStreamParser(max_quote_block_chars=self._max_quote_block_chars)
        tools_by_name = self._tools_by_name

        async def _gen() -> AsyncGenerator[CitationLayerStreamItem, None]:
            ctx = build_citation_context(history_tuple)
            tool_call_lookup = _build_tool_call_lookup(history_tuple)

            async for item in upstream:
                if isinstance(item, str):
                    for parsed in parser.feed(item):
                        if isinstance(parsed, str):
                            if parsed:
                                yield parsed
                        else:
                            yield _validate(parsed, tool_call_lookup, tools_by_name, ctx)
                    continue
                # tool_calls list — flush parser first to preserve order
                for parsed in parser.finish():
                    if isinstance(parsed, str):
                        if parsed:
                            yield parsed
                    else:
                        yield _validate(parsed, tool_call_lookup, tools_by_name, ctx)
                yield item

            for parsed in parser.finish():
                if isinstance(parsed, str):
                    if parsed:
                        yield parsed
                else:
                    yield _validate(parsed, tool_call_lookup, tools_by_name, ctx)

            span = trace.get_current_span()
            span.set_attribute("citation.parsed.count", parser.stats.parsed_count)
            span.set_attribute("citation.parse_failed.count", parser.stats.parse_failed_count)

        return _gen()


def _build_tool_call_lookup(
    history: Sequence[CitationLayerMessage],
) -> dict[str, str]:
    """Map ``tool_call_id`` to ``tool_name`` from prior tool messages in history."""
    lookup: dict[str, str] = {}
    for msg in history:
        if isinstance(msg, CitationLayerToolMessage):
            lookup[msg.tool_call_id] = msg.tool_name
    return lookup


def _validate(
    raw: RawCitation,
    tool_call_lookup: dict[str, str],
    tools_by_name: dict[str, CiteableTool],
    ctx: object,
) -> Citation | HallucinatedCitation:
    """Resolve the responsible CiteableTool and run its validation."""
    tool_call_id = _tool_call_id_of(raw)
    tool_name = tool_call_lookup.get(tool_call_id)
    if tool_name is None:
        return HallucinatedCitation(raw=raw, reason=_REASON_NO_TOOL_CALL)
    tool = tools_by_name.get(tool_name)
    if tool is None:
        return HallucinatedCitation(raw=raw, reason=_REASON_TOOL_NOT_CITEABLE)
    # ctx typed as object to keep import boundary minimal here; CiteableTool
    # accepts any structural CitationContext.
    citation = tool.validate_and_enrich(raw, ctx)  # type: ignore[arg-type]
    if citation is None:
        return HallucinatedCitation(raw=raw, reason=_REASON_TOOL_REJECTED)
    return citation


def _tool_call_id_of(raw: RawCitation) -> str:
    match raw:
        case DocumentRawCitation():
            return raw.tool_call_id
        case ToolRawCitation():
            return raw.tool_call_id
        case _:
            assert_never(raw)
