# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``CitationModel`` — implements the full citation protocol between the orchestrator and the LLM.

Three responsibilities:
1. Augment the system prompt with a single, tool-agnostic citation
   instruction (``{"ref": "<token>"}``) plus any per-tool fragments
   contributed by registered ``CiteableTool``\\ s.
2. Parse marker blocks from the inner model's text stream into typed
   :class:`RawCitation`\\ s.
3. Resolve each ``RawCitation.ref`` against a global token index built from
   all prior tool results in the conversation. Resolved units are enriched
   into typed :class:`Citation`\\ s by the owning tool (or a generic
   :class:`ToolCitation` for plain ``Tool``\\ s); unresolved tokens yield a
   :class:`HallucinatedCitation`.

The orchestrator depends on ``CitationModel`` (not on the underlying
``ChatModel``) and consumes a stream of
``str | list[ToolCallInfo] | Citation | HallucinatedCitation | UnsubstantiatedClaim``.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Iterator, Sequence
from typing import assert_never, cast
from uuid import uuid4

import structlog
from opentelemetry import trace

from src.chatbot.app.citation._parser import (
    DEFAULT_MAX_QUOTE_BLOCK_CHARS,
    CitationStreamParser,
)
from src.chatbot.app.citation.messages import (
    CitationAssistantMessage,
    CitationMessage,
    CitationSystemMessage,
    CitationToolMessage,
    CitationUserMessage,
)
from src.chatbot.app.protocols import (
    ChatMessage,
    ChatModel,
    Citation,
    HallucinatedCitation,
    I18nMessage,
    JsonObject,
    RawCitation,
    ThinkingContent,
    Tool,
    ToolCallInfo,
    ToolCitation,
    ToolSchema,
    UnsubstantiatedClaim,
)
from src.chatbot.app.protocols_citeable_tool import (
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    CitableUnit,
    CiteableTool,
    ToolHistoryRendering,
)

logger = structlog.get_logger(__name__)

type CitationStreamItem = (
    str
    | list[ToolCallInfo]
    | Citation
    | HallucinatedCitation
    | UnsubstantiatedClaim
    | ThinkingContent
)


_REASON_MISSING_REF = "marker payload has no ref"
_REASON_UNKNOWN_REF = "ref does not match any prior tool-result token"


def _to_chat_message(msg: CitationMessage) -> ChatMessage:
    """Project a citation-layer message to the wire-level ``ChatMessage``."""
    match msg:
        case CitationSystemMessage():
            return ChatMessage(role="system", content=msg.llm_content)
        case CitationUserMessage():
            return ChatMessage(role="user", content=msg.llm_content)
        case CitationAssistantMessage():
            return ChatMessage(
                role="assistant",
                content=msg.llm_content,
                tool_calls=msg.tool_calls,
            )
        case CitationToolMessage():
            return ChatMessage(
                role="tool",
                content=msg.llm_content,
                tool_call_id=msg.tool_call_id,
            )
        case _:
            assert_never(msg)


def _splice_assistant_llm_content(
    parts: Sequence[str | Citation | HallucinatedCitation | UnsubstantiatedClaim],
) -> str:
    """Reconstruct the LLM-side text by replacing each citation by its raw marker."""
    pieces: list[str] = []
    for part in parts:
        if isinstance(part, str):
            pieces.append(part)
        else:
            pieces.append(part.raw_marker_text)
    return "".join(pieces)


class CitationModel:
    """Citation decorator around a :class:`~src.chatbot.app.protocols.ChatModel`.

    Composition: instantiate once per session with the inner ``ChatModel`` and
    all registered tools.  ``CiteableTool`` instances receive custom rendering
    and enrichment; plain ``Tool`` instances are auto-wrapped by the generic
    path so that every tool result is citeable.

    Args:
        model: Inner chat model that produces text and tool calls; must not
            interpret marker tokens itself.
        tools: All registered tools.  ``CiteableTool`` instances are detected
            via ``isinstance`` and forwarded to the custom citation path;
            plain ``Tool`` instances go through the generic wrapper.
            Tool names must be unique.
        max_quote_block_chars: Safety limit on a single marker block so a
            runaway model cannot exhaust memory.
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        tools: Sequence[Tool],
        max_quote_block_chars: int = DEFAULT_MAX_QUOTE_BLOCK_CHARS,
    ) -> None:
        self._model = model
        self._max_quote_block_chars = max_quote_block_chars
        self._citeable_by_name: dict[str, CiteableTool] = {}
        self._plain_by_name: dict[str, Tool] = {}
        for tool in tools:
            name = tool.schema.name
            if name in self._citeable_by_name or name in self._plain_by_name:
                raise ValueError(f"Duplicate tool name: {name!r}")
            if isinstance(tool, CiteableTool):
                self._citeable_by_name[name] = tool
            else:
                self._plain_by_name[name] = tool

    # ------------------------------------------------------------------
    # Factory methods — orchestrator calls these to build history entries.
    # ------------------------------------------------------------------

    def make_system_message(self, adjusted_base_prompt: str) -> CitationSystemMessage:
        """Append citation instructions to the orchestrator-supplied base prompt."""
        joined_fragments = "\n\n".join(
            tool.cite_instructions().prompt_fragment for tool in self._citeable_by_name.values()
        )
        if joined_fragments:
            joined_fragments = "\n\n### Per-tool citation guidance\n\n" + joined_fragments

        prompt = f"""{adjusted_base_prompt}

## Inline Citations

Whenever a statement in your answer is supported by a specific tool output,
emit a structured citation marker **immediately after** that statement using
the exact tokens below. The tokens must not appear anywhere else in your
response.

Marker tokens (use exactly):
- start: {QUOTE_START_MARKER}
- end:   {QUOTE_END_MARKER}

Each marker block contains exactly one strict JSON object of the form:

    {QUOTE_START_MARKER}{{"ref":"<citation_token>"}}{QUOTE_END_MARKER}

Where ``<citation_token>`` is the value of a ``citation_token`` attribute
that appears verbatim inside a prior tool result (e.g. as
``<chunk citation_token="...">`` or ``<tool_result citation_token="...">``).
Copy the token exactly — character for character — from the element whose
inner content actually supports your sentence. Never invent, reformat,
truncate, or compose tokens.{joined_fragments}

### General rules
- A citation marker **covers all text since the previous marker**. You may
  group consecutive sentences that are grounded in the **same** element
  under one marker — place the marker after the **last** sentence in that
  run. Split runs at source boundaries: if the next sentence draws from a
  different element, close the current run with a marker before starting
  the new one.
- Emit exactly one JSON object per marker block.
- Only use ``citation_token`` values that appear verbatim in prior tool
  results in the conversation context.
- If no exact ``citation_token`` is visible for a claim, do NOT invent one
  — see the unsubstantiated marker below.
- Logical inferences and transitions derived from cited material do not
  require their own marker — only direct factual claims do.
- If you make a factual claim that no tool result supports, emit immediately
  after that claim:
    {QUOTE_START_MARKER}{{"kind":"unsubstantiated"}}{QUOTE_END_MARKER}
- Do NOT emit an unsubstantiated marker when you explicitly decline to make
  a claim (e.g. "the documents do not contain information about X"). Those
  sentences are transparent refusals, not assertions, and need no marker.
- Keep all normal user-facing answer text outside the markers.
- Structure your answer with paragraphs where the content allows it. Each
  paragraph should cover a coherent sub-topic; do not run all sentences
  together in a single block.
"""
        return CitationSystemMessage(llm_content=prompt)

    def make_user_message(self, user_text: str) -> CitationUserMessage:
        """Prepend the per-turn citation reminder to *user_text*.

        The reminder starts with the tool-agnostic base block, then appends
        any ``reminder_fragment`` contributed by registered ``CiteableTool``s,
        keeping tool-specific nudges co-located with their tool.
        """
        tool_reminders = "\n".join(
            [
                instr.reminder_fragment
                for tool in self._citeable_by_name.values()
                if (instr := tool.cite_instructions()).reminder_fragment is not None
            ]
        )
        reminder = f"""Reminder: when your answer uses tool outputs, emit inline
citation markers. A single marker covers all text since the previous marker
— group consecutive sentences grounded in the **same** element under one
marker placed after the last such sentence. Split at source boundaries, not
sentence boundaries. Use exactly the marker tokens
{QUOTE_START_MARKER} and {QUOTE_END_MARKER}; do not use any marker variants.
The marker payload is always {{"ref":"<citation_token>"}} — copy the
``citation_token`` attribute value verbatim from the exact element whose
inner content supports the run. Never invent tokens or append suffixes.
If no exact token is visible, emit the unsubstantiated marker instead.
Never emit a standalone marker list block. A marker is only valid if it
appears immediately after the last sentence of the run it covers.

{tool_reminders}

!!!IMPORTANT!!!
Every factual claim in your answer must either:
  (a) be the last sentence (or part of a same-source run ending) immediately
      followed by a citation marker referencing the ``citation_token`` of the
      exact element whose content supports the run, OR
  (b) be followed by an unsubstantiated marker if no tool output contains
      the information: {QUOTE_START_MARKER}{{"kind":"unsubstantiated"}}{QUOTE_END_MARKER}
There is no third option: do not state facts without one of these two
markers. Logical inferences and transitional sentences derived from cited
material are exempt.
IMPORTANT EXCEPTION: if you explicitly decline to make a claim — e.g. you
explain that the retrieved documents do not contain the requested
information and therefore you cannot answer — that is a transparent
refusal, not an assertion. Do NOT append an unsubstantiated marker to a
refusal sentence.
Do not append markers in a separate trailing citation section. Place each
marker at the point of use, directly after the last sentence of the run.

The actual user message is:

{user_text}
"""
        return CitationUserMessage(llm_content=reminder)

    def make_assistant_message(
        self,
        parts: Sequence[str | Citation | HallucinatedCitation | UnsubstantiatedClaim],
        *,
        tool_calls: Sequence[ToolCallInfo] | None = None,
    ) -> CitationAssistantMessage:
        """Build an assistant message and pre-compute its LLM-side content."""
        return CitationAssistantMessage(
            parts=tuple(parts),
            llm_content=_splice_assistant_llm_content(parts),
            tool_calls=tuple(tool_calls) if tool_calls else None,
        )

    def make_tool_message(
        self,
        tool_call_id: str,
        tool_name: str,
        result: JsonObject,
    ) -> CitationToolMessage:
        """Build a tool-result message via the tool's renderer or the generic wrapper.

        Registered ``CiteableTool``s render themselves; plain ``Tool``s are
        rendered by :func:`_generic_render_for_history` which embeds a single
        UUID ``citation_token`` so that the result is still citeable.
        """
        citeable = self._citeable_by_name.get(tool_name)
        if citeable is not None:
            rendering = citeable.render_for_history(result)
        else:
            rendering = _generic_render_for_history(result)
        return CitationToolMessage(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result,
            llm_content=rendering.llm_content,
            units=rendering.units,
        )

    def make_blocked_tool_response(self, tc: ToolCallInfo) -> CitationToolMessage:
        """Build a synthetic tool-result that terminates a stuck tool-call loop.

        Called when the orchestrator detects a repeated tool-call sequence.
        The message completes the ``tool_call → tool_result`` protocol so the
        conversation history remains structurally valid, and its content
        signals that the call was blocked.
        """
        llm_content = (
            f"[BLOCKED: Tool call '{tc.name}' was not executed because you already "
            f"called this tool with identical arguments in this turn.]"
        )
        return CitationToolMessage(
            tool_call_id=tc.call_id,
            tool_name=tc.name,
            result={},
            llm_content=llm_content,
            units=(),
        )

    def make_loop_escape_message(self, original_user_content: str) -> CitationUserMessage:
        """Build a user-turn message that forces text generation after a blocked loop.

        After blocked tool responses the history ends at a ``tool`` role,
        which is insufficient for many LLMs to trigger a user-facing text
        response. This user-turn message re-states the original question with
        the citation reminder so the model has a clear, protocol-valid prompt
        to reply to.

        ``original_user_content`` must be the *already-rendered* LLM content
        of the current turn's user message (i.e. the ``llm_content`` of the
        :class:`~src.chatbot.app.citation.messages.CitationLayerUserMessage`
        produced by :meth:`make_user_message` for this turn).  Passing it
        through avoids duplicating the citation-reminder assembly logic.
        """
        content = (
            f"[IMPORTANT: Your previous tool call was blocked because it was an "
            f"exact repeat. No further tool calls are available. "
            f"You MUST now synthesize a complete answer from the tool results "
            f"already in the conversation above. Do NOT call any tools.]\n\n"
            f"{original_user_content}"
        )
        return CitationUserMessage(llm_content=content)

    # ------------------------------------------------------------------
    # Streaming.
    # ------------------------------------------------------------------

    def stream(
        self,
        history: Sequence[CitationMessage],
        *,
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[CitationStreamItem]:
        """Stream a chat completion, validating inline citations as they arrive."""
        chat_messages = [_to_chat_message(msg) for msg in history]
        history_tuple = tuple(history)
        upstream = self._model.stream(chat_messages, tools)
        parser = CitationStreamParser(max_quote_block_chars=self._max_quote_block_chars)
        citeable_by_name = self._citeable_by_name
        plain_by_name = self._plain_by_name

        async def _gen() -> AsyncGenerator[CitationStreamItem, None]:
            token_index = _build_token_index(history_tuple)

            async for item in upstream:
                if isinstance(item, str):
                    for event in _emit_parsed(
                        parser.feed(item), token_index, citeable_by_name, plain_by_name
                    ):
                        yield event
                    continue
                if isinstance(item, ThinkingContent):
                    # Thinking blocks are not citable text — pass through unchanged.
                    yield item
                    continue
                # tool_calls list — flush parser first to preserve order
                for event in _emit_parsed(
                    parser.finish(), token_index, citeable_by_name, plain_by_name
                ):
                    yield event
                yield item

            for event in _emit_parsed(
                parser.finish(), token_index, citeable_by_name, plain_by_name
            ):
                yield event

            span = trace.get_current_span()
            span.set_attribute("citation.parsed.count", parser.stats.parsed_count)
            span.set_attribute("citation.parse_failed.count", parser.stats.parse_failed_count)

        return _gen()


# Token index entry: (owning tool_name, citable unit emitted by that tool).
type _TokenIndex = dict[str, tuple[str, CitableUnit]]


def _build_token_index(history: Sequence[CitationMessage]) -> _TokenIndex:
    """Index every ``CitableUnit`` ever rendered for the LLM by its token.

    Tokens that collide across tool results overwrite earlier entries; this
    is harmless when tokens are content-derived (later occurrences carry the
    same payload), and acceptable in the rare UUID-collision pathological
    case.
    """
    index: _TokenIndex = {}
    for msg in history:
        if isinstance(msg, CitationToolMessage):
            for unit in msg.units:
                index[unit.citation_token] = (msg.tool_name, unit)
    return index


def _emit_parsed(
    parsed_items: Iterable[str | RawCitation],
    token_index: _TokenIndex,
    citeable_by_name: dict[str, CiteableTool],
    plain_by_name: dict[str, Tool],
) -> Iterator[CitationStreamItem]:
    """Validate and yield each item produced by the stream parser."""
    for parsed in parsed_items:
        if isinstance(parsed, str):
            if parsed:
                yield parsed
        else:
            yield _validate(parsed, token_index, citeable_by_name, plain_by_name)


def _validate(
    raw: RawCitation,
    token_index: _TokenIndex,
    citeable_by_name: dict[str, CiteableTool],
    plain_by_name: dict[str, Tool],
) -> Citation | HallucinatedCitation | UnsubstantiatedClaim:
    """Resolve ``raw.ref`` against the global token index and enrich the unit."""
    if raw.kind == "unsubstantiated":
        return UnsubstantiatedClaim(raw=raw)
    if not raw.ref:
        return HallucinatedCitation(raw=raw, reason=_REASON_MISSING_REF)

    entry = token_index.get(raw.ref)
    if entry is None:
        return HallucinatedCitation(raw=raw, reason=_REASON_UNKNOWN_REF)

    tool_name, unit = entry
    citeable = citeable_by_name.get(tool_name)
    if citeable is not None:
        return citeable.enrich(raw, unit)
    # Generic wrapper path — look up display_name from the plain tool if registered.
    plain = plain_by_name.get(tool_name)
    display_name = plain.display_name if plain is not None else None
    return _generic_enrich(raw, unit, tool_name, display_name)


def _generic_render_for_history(result: JsonObject) -> ToolHistoryRendering:
    """Generic fallback rendering: wrap the JSON result in a citeable element.

    Embeds a fresh UUID as ``citation_token`` so the model can cite the
    result even though the tool itself is not citation-aware.
    """
    token = str(uuid4())
    encoded = json.dumps(result, ensure_ascii=False)
    llm_content = f'<tool_result citation_token="{token}">{encoded}</tool_result>'
    unit = CitableUnit(citation_token=token, payload=result)
    return ToolHistoryRendering(llm_content=llm_content, units=(unit,))


def _generic_enrich(
    raw: RawCitation, unit: CitableUnit, tool_name: str, display_name: I18nMessage | None
) -> Citation:
    """Materialise a generic :class:`ToolCitation` from a wrapper-emitted unit."""
    payload = unit.payload
    # The generic wrapper always stores the original tool result JSON as
    # payload; the runtime check below makes that explicit and is satisfied
    # by every code path that reaches this function.
    if isinstance(payload, dict):
        result: JsonObject = cast(JsonObject, payload)
    else:
        result = {}
    return ToolCitation(
        raw_marker_text=raw.raw_marker_text,
        citation_token=unit.citation_token,
        tool_name=tool_name,
        result=result,
        display_name=display_name,
    )
