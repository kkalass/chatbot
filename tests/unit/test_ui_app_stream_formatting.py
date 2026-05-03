# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for UI stream token formatting helpers."""

# pyright: reportPrivateUsage=false

from src.chatbot.app.citation import NumberedCitation, ToolCitation
from src.chatbot.ui.app import (
    _format_citation_marker,
    _format_text_chunk,
)


def _numbered(reference_number: int) -> NumberedCitation:
    citation = ToolCitation(
        raw_marker_text="<marker>",
        tool_call_id="tc1",
        tool_name="search_documents",
        result={"ok": True},
    )
    return NumberedCitation(reference_number=reference_number, citation=citation)


class TestCitationMarkerFormatting:
    def test_marker_keeps_pending_whitespace_buffered(self) -> None:
        tokens, pending = _format_citation_marker(_numbered(1), "\n\n")

        assert tokens == ["[1]"]
        assert pending == "\n\n"

    def test_consecutive_markers_do_not_emit_blank_lines_between_references(self) -> None:
        pending = "\n\n"
        rendered: list[str] = []

        for ref in (1, 2, 3):
            tokens, pending = _format_citation_marker(_numbered(ref), pending)
            rendered.extend(tokens)

        # Pending whitespace is flushed once at the end of the marker run.
        rendered.extend(pending)

        assert "".join(rendered) == "[1][2][3]\n\n"

    def test_pending_whitespace_is_reinserted_before_following_text(self) -> None:
        pending = "\n\n"
        marker_tokens, pending = _format_citation_marker(_numbered(1), pending)
        text_tokens, pending = _format_text_chunk("Next paragraph", pending)

        assert marker_tokens == ["[1]"]
        assert text_tokens == ["\n\n", "Next paragraph"]
        assert pending == ""
