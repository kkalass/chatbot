"""Unit tests for the inline quote parsing chat-model wrapper."""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence

from src.chatbot.app.protocols import (
    ChatMessage,
    ChatStreamItem,
    SearchResultQuote,
    ToolCallInfo,
    ToolSchema,
)
from src.chatbot.infrastructure.chat._inline_quotes import InlineQuoteParsingChatModel

_SEARCH_QUOTE_JSON = (
    '{"kind":"search_result","claim":"Supported by search.",'
    '"tool_call_id":"search-1","source":"corpus/report.txt","chunk_id":"chunk-7"}'
)


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

    async def test_falls_back_to_raw_text_on_malformed_json(self) -> None:
        raw = "Intro <°_quote_°>{not json}</°_quote_°> outro"
        model = InlineQuoteParsingChatModel(_FakeChatModel([raw]))

        events = await _collect(model.stream(messages=[]))

        assert "".join(item for item in events if isinstance(item, str)) == raw
        assert all(isinstance(item, str) for item in events)

    async def test_flushes_unclosed_quote_block_as_text_at_stream_end(self) -> None:
        raw = f"Intro <°_quote_°>{_SEARCH_QUOTE_JSON}"
        model = InlineQuoteParsingChatModel(_FakeChatModel([raw]))

        events = await _collect(model.stream(messages=[]))

        assert "".join(item for item in events if isinstance(item, str)) == raw
        assert all(isinstance(item, str) for item in events)

    async def test_falls_back_to_raw_text_when_quote_buffer_limit_is_exceeded(self) -> None:
        raw = f"<°_quote_°>{_SEARCH_QUOTE_JSON}</°_quote_°>"
        model = InlineQuoteParsingChatModel(
            _FakeChatModel([raw]),
            max_quote_block_chars=20,
        )

        events = await _collect(model.stream(messages=[]))

        assert events == [raw]

    async def test_passes_through_tool_calls_after_flushing_pending_text(self) -> None:
        tool_calls = [ToolCallInfo(name="search_documents", arguments={"query": "x"})]
        model = InlineQuoteParsingChatModel(_FakeChatModel(["Hello", tool_calls]))

        events = await _collect(model.stream(messages=[]))

        assert events == ["Hello", tool_calls]
