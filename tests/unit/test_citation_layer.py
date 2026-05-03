# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :class:`CitationLayer`: factory methods + streaming validation."""

from collections.abc import AsyncIterator, Sequence

import pytest

from src.chatbot.app.citation import (
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    CitationContext,
    CitationLayer,
    CiteInstructions,
    DocumentCitation,
    HallucinatedCitation,
    RawCitation,
    ToolCitation,
)
from src.chatbot.app.citation.messages import (
    CitationLayerAssistantMessage,
    CitationLayerSystemMessage,
    CitationLayerToolMessage,
    CitationLayerUserMessage,
)
from src.chatbot.app.citation.models import Citation
from src.chatbot.app.protocols import (
    ChatMessage,
    ChatStreamItem,
    JsonObject,
    ToolCallInfo,
    ToolSchema,
)

# --- Stub tools and chat model -----------------------------------------------


class _StubCiteableTool:
    def __init__(
        self,
        name: str,
        *,
        fragment: str,
        history_format: str = "RENDERED",
    ) -> None:
        self.schema = ToolSchema(name=name, description="d", parameters_schema={"type": "object"})
        self._fragment = fragment
        self._history_format = history_format
        self.format_calls: list[JsonObject] = []
        self.validate_calls: list[RawCitation] = []

    async def execute(self, args: JsonObject) -> JsonObject:
        return {"ok": True}

    def cite_instructions(self) -> CiteInstructions:
        return CiteInstructions(prompt_fragment=self._fragment)

    def format_for_history(self, result: JsonObject) -> str:
        self.format_calls.append(result)
        return self._history_format

    def validate_and_enrich(self, raw: RawCitation, context: CitationContext) -> Citation | None:
        self.validate_calls.append(raw)
        if raw.chunk_id is None:
            return None
        return DocumentCitation(
            raw_marker_text=raw.raw_marker_text,
            tool_call_id=raw.tool_call_id,
            source="resolved-source",
            chunk_id=raw.chunk_id,
            content="content",
            score=1.0,
        )


class _RejectingCiteableTool(_StubCiteableTool):
    def validate_and_enrich(self, raw: RawCitation, context: CitationContext) -> Citation | None:
        self.validate_calls.append(raw)
        return None


class _StubChatModel:
    def __init__(self, items: Sequence[ChatStreamItem]) -> None:
        self._items = list(items)
        self.received_messages: list[ChatMessage] | None = None
        self.received_tools: Sequence[ToolSchema] | None = None

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ChatStreamItem]:
        self.received_messages = list(messages)
        self.received_tools = tools

        items = self._items

        async def _gen() -> AsyncIterator[ChatStreamItem]:
            for item in items:
                yield item

        return _gen()


# --- Tests ------------------------------------------------------------------


class TestConstruction:
    def test_rejects_duplicate_tool_names(self) -> None:
        a = _StubCiteableTool("dup", fragment="A")
        b = _StubCiteableTool("dup", fragment="B")

        with pytest.raises(ValueError, match="Duplicate"):
            CitationLayer(_StubChatModel([]), citeable_tools=[a, b])


class TestMakeSystemMessage:
    def test_appends_fragments_and_general_rules(self) -> None:
        a = _StubCiteableTool("ta", fragment="FRAG-A")
        b = _StubCiteableTool("tb", fragment="FRAG-B")
        layer = CitationLayer(_StubChatModel([]), citeable_tools=[a, b])

        msg = layer.make_system_message("BASE-PROMPT")

        assert isinstance(msg, CitationLayerSystemMessage)
        assert msg.llm_content.startswith("BASE-PROMPT")
        assert "FRAG-A" in msg.llm_content
        assert "FRAG-B" in msg.llm_content
        assert "Inline Citations" in msg.llm_content
        assert "General rules" in msg.llm_content

    def test_no_tools_still_includes_default_tool_call_fragment(self) -> None:
        layer = CitationLayer(_StubChatModel([]), citeable_tools=[])
        msg = layer.make_system_message("BASE")
        assert msg.llm_content.startswith("BASE")
        assert '"tool_call_id"' in msg.llm_content
        assert "Inline Citations" in msg.llm_content


class TestMakeUserMessage:
    def test_prepends_reminder_and_appends_newline(self) -> None:
        layer = CitationLayer(_StubChatModel([]), citeable_tools=[])
        msg = layer.make_user_message("hello world")

        assert isinstance(msg, CitationLayerUserMessage)
        assert msg.llm_content.endswith("hello world\n")
        assert QUOTE_START_MARKER in msg.llm_content
        assert QUOTE_END_MARKER in msg.llm_content


class TestMakeAssistantMessage:
    def test_splices_raw_marker_text_between_string_parts(self) -> None:
        layer = CitationLayer(_StubChatModel([]), citeable_tools=[])
        cit = DocumentCitation(
            raw_marker_text="<MARKER>",
            tool_call_id="tc1",
            source="s",
            chunk_id="c",
            content="x",
            score=1.0,
        )
        msg = layer.make_assistant_message(("hello ", cit, " world"))

        assert isinstance(msg, CitationLayerAssistantMessage)
        assert msg.llm_content == "hello <MARKER> world"
        assert msg.tool_calls is None
        assert msg.parts == ("hello ", cit, " world")

    def test_propagates_tool_calls(self) -> None:
        layer = CitationLayer(_StubChatModel([]), citeable_tools=[])
        tc = ToolCallInfo(call_id="cid", name="n", arguments={"a": 1})
        msg = layer.make_assistant_message(("text",), tool_calls=[tc])

        assert msg.tool_calls == (tc,)


class TestMakeToolMessage:
    def test_uses_format_for_history_when_tool_registered(self) -> None:
        tool = _StubCiteableTool("vac", fragment="F", history_format="VAC-RENDERED")
        layer = CitationLayer(_StubChatModel([]), citeable_tools=[tool])

        msg = layer.make_tool_message("tc1", "vac", {"x": 1})

        assert isinstance(msg, CitationLayerToolMessage)
        assert msg.llm_content == "VAC-RENDERED"
        assert tool.format_calls == [{"x": 1}]

    def test_falls_back_to_json_for_unknown_tool(self) -> None:
        layer = CitationLayer(_StubChatModel([]), citeable_tools=[])
        msg = layer.make_tool_message("tc1", "unknown", {"x": 1})
        assert msg.llm_content == '{"x": 1}'


class TestStream:
    @pytest.mark.asyncio
    async def test_validated_document_citation_is_yielded(self) -> None:
        marker = f'{QUOTE_START_MARKER}{{"tool_call_id":"tc1","chunk_id":"c"}}{QUOTE_END_MARKER}'
        tool = _StubCiteableTool("search", fragment="F")
        layer = CitationLayer(_StubChatModel([f"prefix {marker} suffix"]), citeable_tools=[tool])
        history = (
            CitationLayerToolMessage(
                tool_call_id="tc1", tool_name="search", result={"chunks": []}, llm_content=""
            ),
        )

        items = [item async for item in layer.stream(history)]

        # Order: "prefix ", DocumentCitation, " suffix"
        assert isinstance(items[1], DocumentCitation)
        assert items[1].source == "resolved-source"
        assert items[1].raw_marker_text == marker
        assert tool.validate_calls
        # Validation receives the parsed RawCitation; documented invariant.
        first = tool.validate_calls[0]
        assert isinstance(first, RawCitation)
        assert first.chunk_id == "c"

    @pytest.mark.asyncio
    async def test_unknown_tool_call_id_yields_hallucination(self) -> None:
        marker = f'{QUOTE_START_MARKER}{{"tool_call_id":"missing"}}{QUOTE_END_MARKER}'
        tool = _StubCiteableTool("vac", fragment="F")
        layer = CitationLayer(_StubChatModel([marker]), citeable_tools=[tool])

        items = [item async for item in layer.stream(())]

        assert any(
            isinstance(i, HallucinatedCitation) and "no prior tool call" in i.reason for i in items
        )

    @pytest.mark.asyncio
    async def test_tool_call_citation_uses_default_validation(self) -> None:
        marker = f'{QUOTE_START_MARKER}{{"tool_call_id":"tc1"}}{QUOTE_END_MARKER}'
        layer = CitationLayer(_StubChatModel([marker]), citeable_tools=[])
        history = (
            CitationLayerToolMessage(
                tool_call_id="tc1", tool_name="not_citeable", result={"x": 1}, llm_content=""
            ),
        )
        items = [item async for item in layer.stream(history)]
        citations = [item for item in items if isinstance(item, ToolCitation)]
        assert len(citations) == 1
        assert citations[0].tool_call_id == "tc1"
        assert citations[0].tool_name == "not_citeable"
        assert citations[0].result == {"x": 1}

    @pytest.mark.asyncio
    async def test_document_citation_rejection_yields_hallucination(self) -> None:
        marker = f'{QUOTE_START_MARKER}{{"tool_call_id":"tc1","chunk_id":"c"}}{QUOTE_END_MARKER}'
        rejector = _RejectingCiteableTool("search", fragment="F")
        layer = CitationLayer(_StubChatModel([marker]), citeable_tools=[rejector])
        history = (
            CitationLayerToolMessage(
                tool_call_id="tc1", tool_name="search", result={"x": 1}, llm_content=""
            ),
        )
        items = [item async for item in layer.stream(history)]
        assert any(
            isinstance(i, HallucinatedCitation) and "validate_and_enrich" in i.reason for i in items
        )

    @pytest.mark.asyncio
    async def test_passes_tool_calls_through(self) -> None:
        tc = ToolCallInfo(call_id="cid", name="n", arguments={})
        layer = CitationLayer(_StubChatModel(["text", [tc]]), citeable_tools=[])

        items = [item async for item in layer.stream(())]
        assert any(isinstance(i, list) and i[0] is tc for i in items)
