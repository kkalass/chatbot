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
    DocumentRawCitation,
    HallucinatedCitation,
    RawCitation,
    ToolCitation,
    ToolRawCitation,
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
        accepts: type[RawCitation],
        history_format: str = "RENDERED",
    ) -> None:
        self.schema = ToolSchema(name=name, description="d", parameters_schema={"type": "object"})
        self._fragment = fragment
        self._accepts = accepts
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
        if not isinstance(raw, self._accepts):
            return None
        if isinstance(raw, DocumentRawCitation):
            return DocumentCitation(
                raw_marker_text=raw.raw_marker_text,
                tool_call_id=raw.tool_call_id,
                source=raw.source,
                chunk_id=raw.chunk_id,
                content="content",
                score=1.0,
            )
        return ToolCitation(
            raw_marker_text=raw.raw_marker_text,
            tool_call_id=raw.tool_call_id,
            tool_name=self.schema.name,
            result={"ok": True},
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
        a = _StubCiteableTool("dup", fragment="A", accepts=DocumentRawCitation)
        b = _StubCiteableTool("dup", fragment="B", accepts=ToolRawCitation)

        with pytest.raises(ValueError, match="Duplicate"):
            CitationLayer(_StubChatModel([]), citeable_tools=[a, b])


class TestMakeSystemMessage:
    def test_appends_fragments_and_general_rules(self) -> None:
        a = _StubCiteableTool("ta", fragment="FRAG-A", accepts=DocumentRawCitation)
        b = _StubCiteableTool("tb", fragment="FRAG-B", accepts=ToolRawCitation)
        layer = CitationLayer(_StubChatModel([]), citeable_tools=[a, b])

        msg = layer.make_system_message("BASE-PROMPT")

        assert isinstance(msg, CitationLayerSystemMessage)
        assert msg.llm_content.startswith("BASE-PROMPT")
        assert "FRAG-A" in msg.llm_content
        assert "FRAG-B" in msg.llm_content
        assert "Inline Citations" in msg.llm_content
        assert "General rules" in msg.llm_content

    def test_no_tools_returns_only_base_prompt(self) -> None:
        layer = CitationLayer(_StubChatModel([]), citeable_tools=[])
        msg = layer.make_system_message("BASE")
        assert msg.llm_content == "BASE"


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
        tool = _StubCiteableTool(
            "vac", fragment="F", accepts=ToolRawCitation, history_format="VAC-RENDERED"
        )
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
        marker = (
            f'{QUOTE_START_MARKER}{{"kind":"document","tool_call_id":"tc1",'
            f'"source":"s","chunk_id":"c"}}{QUOTE_END_MARKER}'
        )
        tool = _StubCiteableTool("search", fragment="F", accepts=DocumentRawCitation)
        layer = CitationLayer(_StubChatModel([f"prefix {marker} suffix"]), citeable_tools=[tool])
        history = (
            CitationLayerToolMessage(
                tool_call_id="tc1", tool_name="search", result={"chunks": []}, llm_content=""
            ),
        )

        items = [item async for item in layer.stream(history)]

        # Order: "prefix ", DocumentCitation, " suffix"
        assert isinstance(items[1], DocumentCitation)
        assert items[1].source == "s"
        assert items[1].raw_marker_text == marker
        assert tool.validate_calls
        # Validation receives the parsed RawCitation; documented invariant.
        first = tool.validate_calls[0]
        assert isinstance(first, DocumentRawCitation)

    @pytest.mark.asyncio
    async def test_unknown_tool_call_id_yields_hallucination(self) -> None:
        marker = (
            f'{QUOTE_START_MARKER}{{"kind":"tool_call","tool_call_id":"missing"}}{QUOTE_END_MARKER}'
        )
        tool = _StubCiteableTool("vac", fragment="F", accepts=ToolRawCitation)
        layer = CitationLayer(_StubChatModel([marker]), citeable_tools=[tool])

        items = [item async for item in layer.stream(())]

        assert any(
            isinstance(i, HallucinatedCitation) and "no prior tool call" in i.reason for i in items
        )

    @pytest.mark.asyncio
    async def test_tool_not_citeable_yields_hallucination(self) -> None:
        marker = (
            f'{QUOTE_START_MARKER}{{"kind":"tool_call","tool_call_id":"tc1"}}{QUOTE_END_MARKER}'
        )
        layer = CitationLayer(_StubChatModel([marker]), citeable_tools=[])
        history = (
            CitationLayerToolMessage(
                tool_call_id="tc1", tool_name="not_citeable", result={}, llm_content=""
            ),
        )
        items = [item async for item in layer.stream(history)]
        assert any(
            isinstance(i, HallucinatedCitation) and "not registered" in i.reason for i in items
        )

    @pytest.mark.asyncio
    async def test_tool_rejection_yields_hallucination(self) -> None:
        marker = (
            f'{QUOTE_START_MARKER}{{"kind":"tool_call","tool_call_id":"tc1"}}{QUOTE_END_MARKER}'
        )
        rejector = _RejectingCiteableTool("vac", fragment="F", accepts=ToolRawCitation)
        layer = CitationLayer(_StubChatModel([marker]), citeable_tools=[rejector])
        history = (
            CitationLayerToolMessage(
                tool_call_id="tc1", tool_name="vac", result={"x": 1}, llm_content=""
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
