# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the streaming marker-block parser used by the CitationLayer."""

import json

from src.chatbot.app.citation._parser import CitationStreamParser
from src.chatbot.contracts.citation import QUOTE_END_MARKER, QUOTE_START_MARKER, RawCitation


def _ref_marker(token: str) -> str:
    payload = {"ref": token}
    return f"{QUOTE_START_MARKER}{json.dumps(payload)}{QUOTE_END_MARKER}"


def _unsubstantiated_marker() -> str:
    payload = {"kind": "unsubstantiated"}
    return f"{QUOTE_START_MARKER}{json.dumps(payload)}{QUOTE_END_MARKER}"


class TestCitationStreamParser:
    def test_plain_text_passes_through(self) -> None:
        parser = CitationStreamParser()
        out = parser.feed("hello ") + parser.feed("world") + parser.finish()

        assert "".join(item for item in out if isinstance(item, str)) == "hello world"

    def test_parses_complete_ref_marker(self) -> None:
        parser = CitationStreamParser()
        marker = _ref_marker("tok-1")

        out = parser.feed(f"prefix {marker} suffix") + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        cit = citations[0]
        assert isinstance(cit, RawCitation)
        assert cit.ref == "tok-1"
        assert cit.kind is None
        assert cit.raw_marker_text == marker

    def test_parses_unsubstantiated_marker(self) -> None:
        parser = CitationStreamParser()
        marker = _unsubstantiated_marker()

        out = parser.feed(marker) + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        cit = citations[0]
        assert isinstance(cit, RawCitation)
        assert cit.kind == "unsubstantiated"
        assert cit.ref is None

    def test_handles_marker_split_across_chunks(self) -> None:
        parser = CitationStreamParser()
        marker = _ref_marker("tok-x")
        midpoint = len(marker) // 2

        out = parser.feed(marker[:midpoint]) + parser.feed(marker[midpoint:]) + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        assert isinstance(citations[0], RawCitation)
        assert citations[0].ref == "tok-x"

    def test_handles_start_token_split_across_chunks(self) -> None:
        parser = CitationStreamParser()
        marker = _ref_marker("tc1")
        # split inside the start marker token itself
        split = 3
        out = parser.feed("text " + marker[:split]) + parser.feed(marker[split:]) + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        assert "text " in "".join(item for item in out if isinstance(item, str))

    def test_malformed_json_emits_raw_block_as_text(self) -> None:
        parser = CitationStreamParser()
        bad = f"{QUOTE_START_MARKER}not json{QUOTE_END_MARKER}"
        out = parser.feed(bad) + parser.finish()

        assert all(isinstance(item, str) for item in out)
        assert parser.stats.parse_failed_count == 1

    def test_non_object_payload_emits_raw_block_as_text(self) -> None:
        parser = CitationStreamParser()
        bad = f"{QUOTE_START_MARKER}[1, 2, 3]{QUOTE_END_MARKER}"
        out = parser.feed(bad) + parser.finish()

        assert all(isinstance(item, str) for item in out)
        assert bad in "".join(out)  # type: ignore[arg-type]
        assert parser.stats.parse_failed_count == 1

    def test_unclosed_block_is_dropped_with_failure_stat(self) -> None:
        parser = CitationStreamParser()
        out = parser.feed(f'{QUOTE_START_MARKER}{{"ref":"tok1"') + parser.finish()

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
            "ref": "tok1",
            "raw_marker_text": "MODEL-INJECTED",
        }
        marker = f"{QUOTE_START_MARKER}{json.dumps(payload)}{QUOTE_END_MARKER}"
        out = parser.feed(marker) + parser.finish()

        citations = [item for item in out if not isinstance(item, str)]
        assert len(citations) == 1
        assert citations[0].raw_marker_text == marker
