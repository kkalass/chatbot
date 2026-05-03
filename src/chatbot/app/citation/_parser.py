"""Streaming marker-block parser for the citation layer.

Detects ``QUOTE_START_MARKER`` / ``QUOTE_END_MARKER`` blocks in a model text
stream, parses the embedded JSON into a typed
:class:`~src.chatbot.app.citation.models.RawCitation`, and yields parsed items
inline with surrounding marker-free text. The complete marker block (including
tokens) is preserved on ``RawCitation.raw_marker_text`` so that it can be
spliced back into the LLM-side history.
"""

import json
from dataclasses import dataclass
from typing import cast

import structlog
from pydantic import ValidationError

from src.chatbot.app.citation.models import (
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    RawCitation,
)

logger = structlog.get_logger(__name__)

DEFAULT_MAX_QUOTE_BLOCK_CHARS = 16_384
_MAX_LOGGED_RAW_PREVIEW_CHARS = 200


def _preview_text(text: str) -> str:
    if len(text) <= _MAX_LOGGED_RAW_PREVIEW_CHARS:
        return text
    return f"{text[:_MAX_LOGGED_RAW_PREVIEW_CHARS]}..."


def _parse_quote_block(raw_block: str) -> RawCitation | None:
    """Parse one complete marker block (incl. tokens) into a typed RawCitation.

    Returns ``None`` on malformed JSON, missing ``tool_call_id``, or schema
    mismatch. Unknown fields (e.g. legacy ``kind``) are silently ignored.
    """
    json_payload = raw_block[len(QUOTE_START_MARKER) : -len(QUOTE_END_MARKER)]
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

        # raw_marker_text is filled by the layer; never trust the model for it.
        parsed_payload.pop("raw_marker_text", None)
        parsed_payload["raw_marker_text"] = raw_block

        # Unsubstantiated claims carry no tool_call_id; provide sentinel so
        # model_validate succeeds — _validate() short-circuits on kind first.
        if parsed_payload.get("kind") == "unsubstantiated" and "tool_call_id" not in parsed_payload:
            parsed_payload["tool_call_id"] = ""

        return RawCitation.model_validate(parsed_payload)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        logger.warning(
            "citation_layer.parse_failed",
            error=str(exc),
            raw_block_preview=_preview_text(raw_block),
            raw_block_chars=len(raw_block),
        )
        return None


@dataclass
class _ParseStats:
    parsed_count: int = 0
    parse_failed_count: int = 0


type _ParserItem = str | RawCitation


class CitationStreamParser:
    """Stateful streaming parser; ``feed`` chunks of model text and consume items.

    Each ``feed`` call returns a list of items detected so far in stream order:
    plain ``str`` segments (marker-free, suitable for direct yield to the UI)
    and parsed :class:`RawCitation` objects. Marker blocks that are
    syntactically present but malformed are emitted as raw ``str`` so the
    caller still sees the model's exact output.

    Call :meth:`finish` after the upstream stream ends to flush any pending
    plain-text buffer and surface unclosed marker blocks.
    """

    def __init__(self, *, max_quote_block_chars: int = DEFAULT_MAX_QUOTE_BLOCK_CHARS) -> None:
        self._max_quote_block_chars = max_quote_block_chars
        self._plain_buffer = ""
        self._quote_buffer: str | None = None
        self.stats = _ParseStats()

    def feed(self, chunk: str) -> list[_ParserItem]:
        outputs: list[_ParserItem] = []
        remaining = chunk

        while True:
            if self._quote_buffer is None:
                self._plain_buffer += remaining
                remaining = ""

                start_index = self._plain_buffer.find(QUOTE_START_MARKER)
                if start_index == -1:
                    flush_upto = len(self._plain_buffer) - len(QUOTE_START_MARKER) + 1
                    if flush_upto > 0:
                        outputs.append(self._plain_buffer[:flush_upto])
                        self._plain_buffer = self._plain_buffer[flush_upto:]
                    break

                if start_index > 0:
                    outputs.append(self._plain_buffer[:start_index])

                self._quote_buffer = QUOTE_START_MARKER
                remaining = self._plain_buffer[start_index + len(QUOTE_START_MARKER) :]
                self._plain_buffer = ""
                continue

            self._quote_buffer += remaining
            remaining = ""

            end_index = self._quote_buffer.find(QUOTE_END_MARKER, len(QUOTE_START_MARKER))
            if end_index == -1:
                if len(self._quote_buffer) > self._max_quote_block_chars:
                    logger.warning(
                        "citation_layer.buffer_limit_exceeded",
                        quote_block_chars=len(self._quote_buffer),
                        max_quote_block_chars=self._max_quote_block_chars,
                        raw_block_preview=_preview_text(self._quote_buffer),
                    )
                    outputs.append(self._quote_buffer)
                    self._quote_buffer = None
                break

            raw_block = self._quote_buffer[: end_index + len(QUOTE_END_MARKER)]
            trailing = self._quote_buffer[end_index + len(QUOTE_END_MARKER) :]
            if len(raw_block) > self._max_quote_block_chars:
                logger.warning(
                    "citation_layer.buffer_limit_exceeded",
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

            parsed = _parse_quote_block(raw_block)
            if parsed is not None:
                self.stats.parsed_count += 1
                outputs.append(parsed)
            else:
                self.stats.parse_failed_count += 1
                outputs.append(raw_block)
            self._quote_buffer = None
            remaining = trailing
            if not remaining:
                break

        return outputs

    def finish(self) -> list[_ParserItem]:
        outputs: list[_ParserItem] = []

        if self._quote_buffer is not None:
            logger.warning(
                "citation_layer.unclosed_block",
                quote_block_chars=len(self._quote_buffer),
                raw_block_preview=_preview_text(self._quote_buffer),
            )
            self.stats.parse_failed_count += 1
            self._quote_buffer = None

        if self._plain_buffer:
            outputs.append(self._plain_buffer)
            self._plain_buffer = ""

        return outputs
