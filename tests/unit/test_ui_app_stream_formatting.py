# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for UI stream token formatting helpers."""

# pyright: reportPrivateUsage=false

import pytest

from src.chatbot.app.protocols import NumberedCitation, ToolCitation
from src.chatbot.ui.app import (
    ResponseManager,
    _format_citation_marker,
    _format_text_chunk,
)


def _numbered(reference_number: int) -> NumberedCitation:
    citation = ToolCitation(
        raw_marker_text="<marker>",
        citation_token="tok1",
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


class _StubMessage:
    """Minimal chainlit.Message stub for ResponseController unit tests."""

    def __init__(self, content: str = "") -> None:
        self.content = content
        self._removed = False

    async def send(self) -> None:
        pass

    async def stream_token(self, token: str) -> None:
        self.content += token

    async def remove(self) -> None:
        self._removed = True

    async def update(self) -> None:
        pass


class TestResponseController:
    """Tests for ResponseController streaming-state management."""

    def _make_controller(self, monkeypatch: pytest.MonkeyPatch) -> ResponseManager:
        import src.chatbot.ui.app as ui_app

        monkeypatch.setattr(ui_app.cl, "Message", _StubMessage)
        return ResponseManager()

    @pytest.mark.asyncio
    async def test_remove_resets_transient_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Regression: after an external remove_message() call (e.g. AuthRequiredEvent),
        _message_is_transient must be False so subsequent stream_token calls
        do NOT destroy the first text message.
        """
        ctrl = self._make_controller(monkeypatch)

        # start_tool_call marks the message as transient.
        await ctrl.start_tool_call("Fetching data…")
        assert ctrl._message_is_transient is True

        # External remove (e.g. from AuthRequiredEvent handler) clears the message
        # AND must reset the flag.
        await ctrl.remove_message()
        assert ctrl._message is None
        assert ctrl._message_is_transient is False

        # First text token after auth: should create a stable message.
        await ctrl.stream_token("I")
        first_msg = ctrl._message
        assert first_msg is not None
        assert first_msg.content == "I"

        # Second text token: must NOT remove the first message.
        await ctrl.stream_token("hr Jahresurlaubsanspruch")
        assert ctrl._message is first_msg, "first message must not be replaced"
        assert ctrl.content == "Ihr Jahresurlaubsanspruch"

    @pytest.mark.asyncio
    async def test_tool_call_message_replaced_on_first_text_token_normal_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Normal flow (no auth): the tool-call placeholder is removed on the first text token."""
        ctrl = self._make_controller(monkeypatch)

        await ctrl.start_tool_call("Fetching data…")
        tool_msg = ctrl._message

        await ctrl.stream_token("Ihr Jahresurlaubsanspruch")
        assert ctrl._message is not tool_msg, "tool placeholder must have been replaced"
        assert ctrl._message is not None
        assert ctrl._message.content == "Ihr Jahresurlaubsanspruch"

        await ctrl.stream_token(" beträgt 25 Tage")
        assert ctrl._message.content == "Ihr Jahresurlaubsanspruch beträgt 25 Tage"

    @pytest.mark.asyncio
    async def test_start_tool_call_discards_empty_dangling_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        prepare_next_response removes an empty dangling message so that
        start_tool_call always gets a clean slate for the tool-call bubble.
        """
        import src.chatbot.ui.app as ui_app

        sent_messages: list[_StubMessage] = []

        class _TrackingSendMessage(_StubMessage):
            async def send(self) -> None:
                sent_messages.append(self)

        monkeypatch.setattr(ui_app.cl, "Message", _TrackingSendMessage)
        ctrl = ResponseManager()

        # Place a dangling empty message (e.g. created at the start of a prior
        # turn but never streamed into).
        empty_msg = _TrackingSendMessage(content="")
        ctrl._message = empty_msg  # type: ignore[assignment]

        await ctrl.start_tool_call("Fetching data…")

        # The empty message must have been discarded.
        assert empty_msg._removed
        # A fresh message must have been sent for the tool-call label.
        assert len(sent_messages) == 1
        assert ctrl._message is sent_messages[0]
        assert "Fetching data" in ctrl._message.content  # type: ignore[union-attr]
        assert ctrl._message_is_transient is True
