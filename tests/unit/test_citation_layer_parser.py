"""Tests for the streaming marker-block parser used by the CitationLayer."""

import json

from src.chatbot.app.citation import (
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    RawCitation,
)
from src.chatbot.app.citation._parser import CitationStreamParser


def _doc_marker(**fields: object) -> str:
    payload = {**fields}
    return f"{QUOTE_START_MARKER}{json.dumps(payload)}{QUOTE_END_MARKER}"


def _tool_marker(tool_call_id: str) -> str:
    payload = {"tool_call_id": tool_call_id}
    return f"{QUOTE_START_MARKER}{json.dumps(payload)}{QUOTE_END_MARKER}"


class TestCitationStreamParser:
    def test_plain_text_passes_through(self) -> None:
        parser = CitationStreamParser()
        out = parser.feed("hello ") + parser.feed("world") + parser.finish()

        assert "".join(item for item in out if isinstance(item, str)) == "hello world"

    def test_parses_complete_document_marker(self) -> None:
        parser = CitationStreamParser()
        marker = _doc_marker(tool_call_id="tc1", chunk_id="c1")

        out = parser.feed(f"prefix {marker} suffix") + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        cit = citations[0]
        assert isinstance(cit, RawCitation)
        assert cit.tool_call_id == "tc1"
        assert cit.chunk_id == "c1"
        assert cit.raw_marker_text == marker

    def test_parses_tool_marker(self) -> None:
        parser = CitationStreamParser()
        marker = _tool_marker("tc-xyz")

        out = parser.feed(marker) + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        cit = citations[0]
        assert isinstance(cit, RawCitation)
        assert cit.tool_call_id == "tc-xyz"
        assert cit.chunk_id is None

    def test_handles_marker_split_across_chunks(self) -> None:
        parser = CitationStreamParser()
        marker = _doc_marker(tool_call_id="tc1", chunk_id="c")
        midpoint = len(marker) // 2

        out = parser.feed(marker[:midpoint]) + parser.feed(marker[midpoint:]) + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        assert isinstance(citations[0], RawCitation)
        assert citations[0].chunk_id == "c"

    def test_handles_start_token_split_across_chunks(self) -> None:
        parser = CitationStreamParser()
        marker = _tool_marker("tc1")
        # split inside the start marker token itself
        split = 3
        out = parser.feed("text " + marker[:split]) + parser.feed(marker[split:]) + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        assert "text " in "".join(item for item in out if isinstance(item, str))

    def test_missing_tool_call_id_emits_raw_text_and_increments_failure(self) -> None:
        parser = CitationStreamParser()
        bad = f'{QUOTE_START_MARKER}{{"chunk_id":"c1"}}{QUOTE_END_MARKER}'
        out = parser.feed(bad) + parser.finish()

        # No typed citation produced; raw block surfaces as a string.
        assert all(isinstance(item, str) for item in out)
        assert bad in "".join(out)  # type: ignore[arg-type]
        assert parser.stats.parse_failed_count == 1
        assert parser.stats.parsed_count == 0

    def test_malformed_json_emits_raw_block_as_text(self) -> None:
        parser = CitationStreamParser()
        bad = f"{QUOTE_START_MARKER}not json{QUOTE_END_MARKER}"
        out = parser.feed(bad) + parser.finish()

        assert all(isinstance(item, str) for item in out)
        assert parser.stats.parse_failed_count == 1

    def test_unclosed_block_is_dropped_with_failure_stat(self) -> None:
        parser = CitationStreamParser()
        out = parser.feed(f'{QUOTE_START_MARKER}{{"tool_call_id":"tc1"') + parser.finish()

        # The unclosed buffer must not surface as plain text.
        assert all(isinstance(item, str) for item in out)
        assert "".join(out) == ""  # type: ignore[arg-type]
        assert parser.stats.parse_failed_count == 1

    def test_buffer_limit_emits_raw_block(self) -> None:
        parser = CitationStreamParser(max_quote_block_chars=20)
        oversized = QUOTE_START_MARKER + ("x" * 100)
        out = parser.feed(oversized) + parser.finish()

        # Oversized partial block flushed as raw text.
        joined = "".join(item for item in out if isinstance(item, str))
        assert "xxx" in joined

    def test_model_supplied_raw_marker_text_is_overridden(self) -> None:
        parser = CitationStreamParser()
        payload = {
            "tool_call_id": "tc1",
            "chunk_id": "c",
            "raw_marker_text": "MODEL-INJECTED",
        }
        marker = f"{QUOTE_START_MARKER}{json.dumps(payload)}{QUOTE_END_MARKER}"
        out = parser.feed(marker) + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        assert citations[0].raw_marker_text == marker

    def test_legacy_kind_field_ignored(self) -> None:
        """Markers with legacy 'kind' and 'source' fields must still parse cleanly."""
        parser = CitationStreamParser()
        payload = {"kind": "document", "tool_call_id": "tc1", "source": "s.md", "chunk_id": "c1"}
        marker = f"{QUOTE_START_MARKER}{json.dumps(payload)}{QUOTE_END_MARKER}"
        out = parser.feed(marker) + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        assert isinstance(citations[0], RawCitation)
        assert citations[0].tool_call_id == "tc1"
        assert citations[0].chunk_id == "c1"
