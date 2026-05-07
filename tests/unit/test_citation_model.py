# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :class:`CitationModel`: factory methods + streaming validation."""

from collections.abc import AsyncIterator, Sequence

import pytest

from src.chatbot.app.citation import CitationModel
from src.chatbot.app.citation.messages import (
    CitationAssistantMessage,
    CitationSystemMessage,
    CitationUserMessage,
)
from src.chatbot.contracts.chat import ChatMessage, ChatStreamItem, ToolCallInfo
from src.chatbot.contracts.citation import (
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    CitableUnit,
    Citation,
    CiteInstructions,
    DocumentCitation,
    HallucinatedCitation,
    RawCitation,
    ToolCitation,
    ToolHistoryRendering,
    UnsubstantiatedClaim,
)
from src.chatbot.contracts.i18n import I18nMessage, JsonObject
from src.chatbot.contracts.tools import ToolSchema

# --- Stub tools and chat model -----------------------------------------------


class _StubCiteableTool:
    """Stub tool emitting a single citable unit per result.

    The unit's token equals ``result["token"]`` so tests can drive the layer
    deterministically. The unit payload is the raw result dict.
    """

    def __init__(self, name: str, *, fragment: str) -> None:
        self.schema = ToolSchema(name=name, description="d", parameters_schema={"type": "object"})
        self.display_name = I18nMessage(key="stub.tool", args={})
        self._fragment = fragment
        self.render_calls: list[JsonObject] = []
        self.enrich_calls: list[tuple[RawCitation, CitableUnit]] = []

    async def execute(self, args: JsonObject) -> JsonObject:
        return {"ok": True}

    def describe_call(self, args: JsonObject) -> I18nMessage:
        return I18nMessage(key="stub.call", args={})

    def cite_instructions(self) -> CiteInstructions:
        return CiteInstructions(prompt_fragment=self._fragment)

    def render_for_history(self, result: JsonObject) -> ToolHistoryRendering:
        self.render_calls.append(result)
        token = str(result.get("token", "tok"))
        unit = CitableUnit(citation_token=token, payload=result)
        return ToolHistoryRendering(
            llm_content=f'<chunk citation_token="{token}">x</chunk>',
            units=(unit,),
        )

    def enrich(self, raw: RawCitation, unit: CitableUnit) -> Citation:
        self.enrich_calls.append((raw, unit))
        return DocumentCitation(
            raw_marker_text=raw.raw_marker_text,
            citation_token=unit.citation_token,
            source="resolved-source",
            chunk_id=unit.citation_token,
            content="content",
            score=1.0,
        )


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
            CitationModel(_StubChatModel([]), tools=[a, b])


class TestMakeSystemMessage:
    def test_appends_fragments_and_general_rules(self) -> None:
        a = _StubCiteableTool("ta", fragment="FRAG-A")
        b = _StubCiteableTool("tb", fragment="FRAG-B")
        layer = CitationModel(_StubChatModel([]), tools=[a, b])

        msg = layer.make_system_message("BASE-PROMPT")

        assert isinstance(msg, CitationSystemMessage)
        assert msg.llm_content.startswith("BASE-PROMPT")
        assert "FRAG-A" in msg.llm_content
        assert "FRAG-B" in msg.llm_content
        assert "Inline Citations" in msg.llm_content
        assert "General rules" in msg.llm_content

    def test_no_tools_still_documents_universal_marker(self) -> None:
        layer = CitationModel(_StubChatModel([]), tools=[])
        msg = layer.make_system_message("BASE")
        assert msg.llm_content.startswith("BASE")
        # Universal marker schema is documented even without registered tools.
        assert '"ref"' in msg.llm_content
        assert "Inline Citations" in msg.llm_content


class TestMakeUserMessage:
    def test_prepends_reminder_and_appends_newline(self) -> None:
        layer = CitationModel(_StubChatModel([]), tools=[])
        msg = layer.make_user_message("hello world")

        assert isinstance(msg, CitationUserMessage)
        assert msg.llm_content.endswith("hello world\n")
        assert QUOTE_START_MARKER in msg.llm_content
        assert QUOTE_END_MARKER in msg.llm_content


class TestMakeAssistantMessage:
    def test_splices_raw_marker_text_between_string_parts(self) -> None:
        layer = CitationModel(_StubChatModel([]), tools=[])
        cit = DocumentCitation(
            raw_marker_text="<MARKER>",
            citation_token="tok1",
            source="s",
            chunk_id="c",
            content="x",
            score=1.0,
        )
        msg = layer.make_assistant_message(("hello ", cit, " world"))

        assert isinstance(msg, CitationAssistantMessage)
        assert msg.llm_content == "hello <MARKER> world"
        assert msg.tool_calls is None
        assert msg.parts == ("hello ", cit, " world")

    def test_propagates_tool_calls(self) -> None:
        layer = CitationModel(_StubChatModel([]), tools=[])
        tc = ToolCallInfo(call_id="cid", name="n", arguments={"a": 1})
        msg = layer.make_assistant_message(("text",), tool_calls=[tc])

        assert msg.tool_calls == (tc,)


class TestMakeToolMessage:
    def test_uses_tool_renderer_when_registered(self) -> None:
        tool = _StubCiteableTool("vac", fragment="F")
        layer = CitationModel(_StubChatModel([]), tools=[tool])

        msg = layer.make_tool_message("tc1", "vac", {"token": "tok-A"})

        assert msg.llm_content == '<chunk citation_token="tok-A">x</chunk>'
        assert tool.render_calls == [{"token": "tok-A"}]
        assert len(msg.units) == 1
        assert msg.units[0].citation_token == "tok-A"

    def test_falls_back_to_generic_wrapper_for_unknown_tool(self) -> None:
        layer = CitationModel(_StubChatModel([]), tools=[])
        msg = layer.make_tool_message("tc1", "unknown", {"x": 1})
        # Generic wrapper embeds a UUID citation_token and JSON-serialises the result.
        assert msg.llm_content.startswith('<tool_result citation_token="')
        assert '{"x": 1}' in msg.llm_content
        assert msg.llm_content.endswith("</tool_result>")
        assert len(msg.units) == 1
        # Generic UUID token is non-empty and indexed.
        assert msg.units[0].citation_token


class TestStream:
    @pytest.mark.asyncio
    async def test_validated_citation_is_yielded(self) -> None:
        tool = _StubCiteableTool("search", fragment="F")
        layer = CitationModel(
            _StubChatModel(
                [f'prefix {QUOTE_START_MARKER}{{"ref":"tok-A"}}{QUOTE_END_MARKER} suffix']
            ),
            tools=[tool],
        )
        # History must be built via the layer so `units` get populated.
        tool_msg = layer.make_tool_message("tc1", "search", {"token": "tok-A"})

        items = [item async for item in layer.stream((tool_msg,))]

        # Order: "prefix ", DocumentCitation, " suffix"
        assert isinstance(items[1], DocumentCitation)
        assert items[1].source == "resolved-source"
        assert items[1].citation_token == "tok-A"
        assert tool.enrich_calls
        raw, unit = tool.enrich_calls[0]
        assert isinstance(raw, RawCitation)
        assert raw.ref == "tok-A"
        assert unit.citation_token == "tok-A"

    @pytest.mark.asyncio
    async def test_unknown_ref_yields_hallucination(self) -> None:
        marker = f'{QUOTE_START_MARKER}{{"ref":"nope"}}{QUOTE_END_MARKER}'
        tool = _StubCiteableTool("search", fragment="F")
        layer = CitationModel(_StubChatModel([marker]), tools=[tool])

        items = [item async for item in layer.stream(())]

        assert any(
            isinstance(i, HallucinatedCitation) and "ref does not match" in i.reason for i in items
        )

    @pytest.mark.asyncio
    async def test_missing_ref_yields_hallucination(self) -> None:
        marker = f"{QUOTE_START_MARKER}{{}}{QUOTE_END_MARKER}"
        layer = CitationModel(_StubChatModel([marker]), tools=[])

        items = [item async for item in layer.stream(())]

        assert any(isinstance(i, HallucinatedCitation) and "no ref" in i.reason for i in items)

    @pytest.mark.asyncio
    async def test_unsubstantiated_marker_yields_unsubstantiated_claim(self) -> None:
        marker = f'{QUOTE_START_MARKER}{{"kind":"unsubstantiated"}}{QUOTE_END_MARKER}'
        layer = CitationModel(_StubChatModel([marker]), tools=[])

        items = [item async for item in layer.stream(())]

        assert any(isinstance(i, UnsubstantiatedClaim) for i in items)

    @pytest.mark.asyncio
    async def test_generic_tool_citation_via_wrapper(self) -> None:
        # Build the tool message via a helper layer so the generic wrapper
        # assigns a UUID token; the streaming layer below indexes that token.
        helper = CitationModel(_StubChatModel([]), tools=[])
        tool_msg = helper.make_tool_message("tc1", "not_citeable", {"x": 1})
        token = tool_msg.units[0].citation_token
        marker = f'{QUOTE_START_MARKER}{{"ref":"{token}"}}{QUOTE_END_MARKER}'

        layer = CitationModel(_StubChatModel([marker]), tools=[])
        items = [item async for item in layer.stream((tool_msg,))]

        citations = [item for item in items if isinstance(item, ToolCitation)]
        assert len(citations) == 1
        assert citations[0].tool_name == "not_citeable"
        assert citations[0].result == {"x": 1}
        assert citations[0].citation_token == token

    @pytest.mark.asyncio
    async def test_passes_tool_calls_through(self) -> None:
        tc = ToolCallInfo(call_id="cid", name="n", arguments={})
        layer = CitationModel(_StubChatModel(["text", [tc]]), tools=[])

        items = [item async for item in layer.stream(())]
        assert any(isinstance(i, list) and i[0] is tc for i in items)


class TestGenericRenderDeterministicToken:
    """_generic_render_for_history must produce a stable citation_token for plain tools."""

    def test_same_result_produces_same_token(self) -> None:
        layer = CitationModel(_StubChatModel([]), tools=[])
        msg1 = layer.make_tool_message("tc1", "plain_tool", {"days": 30, "year": 2026})
        msg2 = layer.make_tool_message("tc2", "plain_tool", {"days": 30, "year": 2026})

        assert msg1.units[0].citation_token == msg2.units[0].citation_token

    def test_different_result_produces_different_token(self) -> None:
        layer = CitationModel(_StubChatModel([]), tools=[])
        msg1 = layer.make_tool_message("tc1", "plain_tool", {"days": 30})
        msg2 = layer.make_tool_message("tc2", "plain_tool", {"days": 25})

        assert msg1.units[0].citation_token != msg2.units[0].citation_token

    def test_token_is_deterministic_regardless_of_key_order(self) -> None:
        """JSON key order in the caller's dict must not affect the token."""
        layer = CitationModel(_StubChatModel([]), tools=[])
        msg1 = layer.make_tool_message("tc1", "t", {"a": 1, "b": 2})
        msg2 = layer.make_tool_message("tc2", "t", {"b": 2, "a": 1})

        assert msg1.units[0].citation_token == msg2.units[0].citation_token

    def test_token_is_hex_string_of_expected_length(self) -> None:
        layer = CitationModel(_StubChatModel([]), tools=[])
        msg = layer.make_tool_message("tc1", "t", {"x": 1})
        token = msg.units[0].citation_token

        assert len(token) == 16
        assert all(c in "0123456789abcdef" for c in token)
