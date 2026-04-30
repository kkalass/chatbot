"""Unit tests for the inline quote parsing chat-model wrapper."""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence

from src.chatbot.app.protocols import (
    ChatMessage,
    ChatStreamItem,
    RawAssistantText,
    SearchResultQuote,
    ToolCallInfo,
    ToolCallQuote,
    ToolSchema,
)
from src.chatbot.infrastructure.chat._inline_quotes import (
    InlineQuoteParsingChatModel,
    _InlineQuoteStreamParser,  # pyright: ignore[reportPrivateUsage]
)

_SEARCH_QUOTE_JSON = (
    '{"kind":"search_result","claim":"Supported by search.",'
    '"tool_call_id":"search-1","source":"corpus/report.txt","chunk_id":"chunk-7"}'
)
_TOOL_QUOTE_JSON = '{"kind":"tool_call","tool_call_id":"get_vacation_days"}'


class _FakeChatModel:
    def __init__(self, items: list[ChatStreamItem]) -> None:
        self._items = items

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ChatStreamItem]:
        del messages, tools

        async def _gen() -> AsyncGenerator[ChatStreamItem, None]:
            for item in self._items:
                yield item

        return _gen()


async def _collect(items: AsyncIterator[ChatStreamItem]) -> list[ChatStreamItem]:
    collected: list[ChatStreamItem] = []
    async for item in items:
        collected.append(item)
    return collected


class TestInlineQuoteParsingChatModel:
    async def test_parses_quote_marker_within_single_chunk(self) -> None:
        model = InlineQuoteParsingChatModel(
            _FakeChatModel(
                [
                    f"Intro <°_quote_°>{_SEARCH_QUOTE_JSON}</°_quote_°> outro",
                ]
            )
        )

        events = await _collect(model.stream(messages=[]))

        assert events[0] == "Intro "
        assert isinstance(events[1], SearchResultQuote)
        assert events[1].kind == "search_result"
        assert events[1].tool_call_id == "search-1"
        assert events[2] == " outro"
        assert isinstance(events[3], RawAssistantText)
        assert events[3].text == f"Intro <°_quote_°>{_SEARCH_QUOTE_JSON}</°_quote_°> outro"

    async def test_parses_markers_split_across_chunk_boundaries(self) -> None:
        model = InlineQuoteParsingChatModel(
            _FakeChatModel(
                [
                    "Intro <°",
                    f"_quote_°>{_SEARCH_QUOTE_JSON}</°_quo",
                    "te_°> tail",
                ]
            )
        )

        events = await _collect(model.stream(messages=[]))

        assert events[0] == "Intro "
        assert isinstance(events[1], SearchResultQuote)
        assert events[1].kind == "search_result"
        assert events[1].chunk_id == "chunk-7"
        assert events[2] == " tail"
        assert isinstance(events[3], RawAssistantText)

    async def test_falls_back_to_raw_text_on_malformed_json(self) -> None:
        raw = "Intro <°_quote_°>{not json}</°_quote_°> outro"
        model = InlineQuoteParsingChatModel(_FakeChatModel([raw]))

        events = await _collect(model.stream(messages=[]))

        assert "".join(item for item in events if isinstance(item, str)) == raw
        assert isinstance(events[-1], RawAssistantText)
        assert events[-1].text == raw

    async def test_drops_unclosed_quote_block_at_stream_end(self) -> None:
        raw = f"Intro <°_quote_°>{_SEARCH_QUOTE_JSON}"
        model = InlineQuoteParsingChatModel(_FakeChatModel([raw]))

        events = await _collect(model.stream(messages=[]))

        assert "".join(item for item in events if isinstance(item, str)) == "Intro "
        assert isinstance(events[-1], RawAssistantText)
        assert events[-1].text == raw

    async def test_falls_back_to_raw_text_when_quote_buffer_limit_is_exceeded(self) -> None:
        raw = f"<°_quote_°>{_SEARCH_QUOTE_JSON}</°_quote_°>"
        model = InlineQuoteParsingChatModel(
            _FakeChatModel([raw]),
            max_quote_block_chars=20,
        )

        events = await _collect(model.stream(messages=[]))

        assert events == [raw, RawAssistantText(text=raw)]

    async def test_passes_through_tool_calls_after_flushing_pending_text(self) -> None:
        tool_calls = [ToolCallInfo(name="search_documents", arguments={"query": "x"})]
        model = InlineQuoteParsingChatModel(_FakeChatModel(["Hello", tool_calls]))

        events = await _collect(model.stream(messages=[]))

        assert events == ["Hello", tool_calls, RawAssistantText(text="Hello")]

    async def test_parses_tool_call_quote_without_tool_name(self) -> None:
        model = InlineQuoteParsingChatModel(
            _FakeChatModel(
                [
                    f"Text <°_quote_°>{_TOOL_QUOTE_JSON}</°_quote_°> end",
                ]
            )
        )

        events = await _collect(model.stream(messages=[]))

        assert events[0] == "Text "
        assert isinstance(events[1], ToolCallQuote)
        assert events[1].tool_call_id == "get_vacation_days"
        assert events[2] == " end"
        assert isinstance(events[3], RawAssistantText)

    async def test_emits_raw_text_payload_even_without_quote_markers(self) -> None:
        model = InlineQuoteParsingChatModel(_FakeChatModel(["Plain response."]))

        events = await _collect(model.stream(messages=[]))

        assert "".join(item for item in events if isinstance(item, str)) == "Plain response."
        assert isinstance(events[-1], RawAssistantText)
        assert events[-1].text == "Plain response."


class TestInlineQuoteStreamParserCounts:
    """Verify parsed_count and parse_failed_count are tracked on the parser."""

    def test_successful_parse_increments_parsed_count(self) -> None:
        parser = _InlineQuoteStreamParser(max_quote_block_chars=16_384)
        valid_block = f"<°_quote_°>{_SEARCH_QUOTE_JSON}</°_quote_°>"

        parser.feed(valid_block)

        assert parser.parsed_count == 1
        assert parser.parse_failed_count == 0

    def test_failed_parse_increments_parse_failed_count(self) -> None:
        parser = _InlineQuoteStreamParser(max_quote_block_chars=16_384)

        parser.feed("<°_quote_°>{not json}</°_quote_°>")

        assert parser.parsed_count == 0
        assert parser.parse_failed_count == 1

    def test_mixed_valid_and_invalid_blocks_counted_separately(self) -> None:
        parser = _InlineQuoteStreamParser(max_quote_block_chars=16_384)

        parser.feed(f"<°_quote_°>{_SEARCH_QUOTE_JSON}</°_quote_°>")
        parser.feed("<°_quote_°>{bad}</°_quote_°>")
        parser.feed(f"<°_quote_°>{_SEARCH_QUOTE_JSON}</°_quote_°>")

        assert parser.parsed_count == 2
        assert parser.parse_failed_count == 1

    def test_unclosed_block_increments_parse_failed_count(self) -> None:
        parser = _InlineQuoteStreamParser(max_quote_block_chars=16_384)

        parser.feed(f"Intro <°_quote_°>{_SEARCH_QUOTE_JSON}")
        parser.finish()

        assert parser.parsed_count == 0
        assert parser.parse_failed_count == 1
