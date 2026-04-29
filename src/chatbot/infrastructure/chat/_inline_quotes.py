"""Streaming inline-quote parsing wrapper for chat model output."""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import cast

import structlog
from opentelemetry import trace
from pydantic import ValidationError

from src.chatbot.app.protocols import (
    ChatMessage,
    ChatModel,
    ChatStreamItem,
    Quote,
    SearchResultQuote,
    ToolCallQuote,
    ToolSchema,
)

logger = structlog.get_logger(__name__)

_QUOTE_START_MARKER = "<°_quote_°>"
_QUOTE_END_MARKER = "</°_quote_°>"
_DEFAULT_MAX_QUOTE_BLOCK_CHARS = 16_384
_MAX_LOGGED_RAW_PREVIEW_CHARS = 200


def _preview_text(text: str) -> str:
    if len(text) <= _MAX_LOGGED_RAW_PREVIEW_CHARS:
        return text
    return f"{text[:_MAX_LOGGED_RAW_PREVIEW_CHARS]}..."


def _parse_quote_block(raw_block: str) -> Quote | None:
    json_payload = raw_block[len(_QUOTE_START_MARKER) : -len(_QUOTE_END_MARKER)]

    try:
        parsed: object = json.loads(json_payload)
        if not isinstance(parsed, dict):
            raise ValueError("Quote payload must be a JSON object.")

        raw_mapping = cast(dict[object, object], parsed)
        parsed_payload: dict[str, object] = {}
        for key, value in raw_mapping.items():
            if not isinstance(key, str):
                raise ValueError("Quote payload keys must be strings.")
            parsed_payload[key] = value

        kind_value = parsed_payload.get("kind")
        if not isinstance(kind_value, str):
            raise ValueError("Quote payload kind must be a string.")

        match kind_value:
            case "search_result":
                return SearchResultQuote.model_validate(parsed_payload)
            case "tool_call":
                return ToolCallQuote.model_validate(parsed_payload)
            case _:
                raise ValueError(f"Unsupported quote kind: {kind_value!r}")
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        logger.warning(
            "chat.inline_quote.parse_failed",
            error=str(exc),
            raw_block_preview=_preview_text(raw_block),
            raw_block_chars=len(raw_block),
        )
        return None


class _InlineQuoteStreamParser:
    def __init__(self, *, max_quote_block_chars: int) -> None:
        self._max_quote_block_chars = max_quote_block_chars
        self._plain_buffer = ""
        self._quote_buffer: str | None = None
        self.parsed_count: int = 0
        self.parse_failed_count: int = 0

    def feed(self, chunk: str) -> list[ChatStreamItem]:
        outputs: list[ChatStreamItem] = []
        remaining = chunk

        while True:
            if self._quote_buffer is None:
                self._plain_buffer += remaining
                remaining = ""

                start_index = self._plain_buffer.find(_QUOTE_START_MARKER)
                if start_index == -1:
                    flush_upto = len(self._plain_buffer) - len(_QUOTE_START_MARKER) + 1
                    if flush_upto > 0:
                        outputs.append(self._plain_buffer[:flush_upto])
                        self._plain_buffer = self._plain_buffer[flush_upto:]
                    break

                if start_index > 0:
                    outputs.append(self._plain_buffer[:start_index])

                self._quote_buffer = _QUOTE_START_MARKER
                remaining = self._plain_buffer[start_index + len(_QUOTE_START_MARKER) :]
                self._plain_buffer = ""
                continue

            self._quote_buffer += remaining
            remaining = ""

            end_index = self._quote_buffer.find(_QUOTE_END_MARKER, len(_QUOTE_START_MARKER))
            if end_index == -1:
                if len(self._quote_buffer) > self._max_quote_block_chars:
                    logger.warning(
                        "chat.inline_quote.buffer_limit_exceeded",
                        quote_block_chars=len(self._quote_buffer),
                        max_quote_block_chars=self._max_quote_block_chars,
                        raw_block_preview=_preview_text(self._quote_buffer),
                    )
                    outputs.append(self._quote_buffer)
                    self._quote_buffer = None
                break

            raw_block = self._quote_buffer[: end_index + len(_QUOTE_END_MARKER)]
            trailing = self._quote_buffer[end_index + len(_QUOTE_END_MARKER) :]
            if len(raw_block) > self._max_quote_block_chars:
                logger.warning(
                    "chat.inline_quote.buffer_limit_exceeded",
                    quote_block_chars=len(raw_block),
                    max_quote_block_chars=self._max_quote_block_chars,
                    raw_block_preview=_preview_text(raw_block),
                )
                outputs.append(raw_block)
                self._quote_buffer = None
                remaining = trailing
                if not remaining:
                    break
                continue

            parsed_quote = _parse_quote_block(raw_block)
            if parsed_quote is not None:
                self.parsed_count += 1
                outputs.append(parsed_quote)
            else:
                self.parse_failed_count += 1
                outputs.append(raw_block)
            self._quote_buffer = None
            remaining = trailing
            if not remaining:
                break

        return outputs

    def finish(self) -> list[ChatStreamItem]:
        outputs: list[ChatStreamItem] = []

        if self._quote_buffer is not None:
            logger.warning(
                "chat.inline_quote.unclosed_block",
                quote_block_chars=len(self._quote_buffer),
                raw_block_preview=_preview_text(self._quote_buffer),
            )
            self.parse_failed_count += 1
            self._quote_buffer = None

        if self._plain_buffer:
            outputs.append(self._plain_buffer)
            self._plain_buffer = ""

        return outputs


class InlineQuoteParsingChatModel:
    """Wrap a chat model and convert marker-delimited quote JSON into ``Quote`` items."""

    def __init__(
        self,
        upstream: ChatModel,
        *,
        max_quote_block_chars: int = _DEFAULT_MAX_QUOTE_BLOCK_CHARS,
    ) -> None:
        self._upstream = upstream
        self._max_quote_block_chars = max_quote_block_chars

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ChatStreamItem]:
        parser = _InlineQuoteStreamParser(max_quote_block_chars=self._max_quote_block_chars)
        upstream_stream = self._upstream.stream(messages, tools)

        async def _gen() -> AsyncGenerator[ChatStreamItem, None]:
            async for item in upstream_stream:
                if isinstance(item, str):
                    for parsed_item in parser.feed(item):
                        yield parsed_item
                    continue

                for parsed_item in parser.finish():
                    yield parsed_item
                yield item

            for parsed_item in parser.finish():
                yield parsed_item

            span = trace.get_current_span()
            span.set_attribute("quote.parsed.count", parser.parsed_count)
            span.set_attribute("quote.parse_failed.count", parser.parse_failed_count)

        return _gen()


def build_inline_quote_parsing_chat_model(
    upstream: ChatModel,
    *,
    max_quote_block_chars: int = _DEFAULT_MAX_QUOTE_BLOCK_CHARS,
) -> ChatModel:
    """Wrap ``upstream`` with the inline-quote parsing adapter."""
    return InlineQuoteParsingChatModel(
        upstream,
        max_quote_block_chars=max_quote_block_chars,
    )
