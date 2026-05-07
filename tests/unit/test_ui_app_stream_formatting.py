# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for UI stream token formatting helpers."""

# pyright: reportPrivateUsage=false

from pathlib import Path
from typing import Any, cast

import pytest

from src.chatbot.contracts.citation import DocumentCitation, NumberedCitation, ToolCitation
from src.chatbot.ui.app import ResponseManager
from src.chatbot.ui.citation_view import format_citation_marker, format_text_chunk


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
        tokens, pending = format_citation_marker(_numbered(1), "\n\n")

        assert tokens == ["_(1)_"]
        assert pending == "\n\n"

    def test_consecutive_markers_do_not_emit_blank_lines_between_references(self) -> None:
        pending = "\n\n"
        rendered: list[str] = []

        for ref in (1, 2, 3):
            tokens, pending = format_citation_marker(_numbered(ref), pending)
            rendered.extend(tokens)

        # Pending whitespace is flushed once at the end of the marker run.
        rendered.extend(pending)

        assert "".join(rendered) == "_(1)__(2)__(3)_\n\n"

    def test_pending_whitespace_is_reinserted_before_following_text(self) -> None:
        pending = "\n\n"
        marker_tokens, pending = format_citation_marker(_numbered(1), pending)
        text_tokens, pending = format_text_chunk("Next paragraph", pending)

        assert marker_tokens == ["_(1)_"]
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


class TestSideElements:
    def test_side_elements_group_text_and_image_per_citation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.chatbot.ui.app as ui_app

        class _StubText:
            def __init__(self, *, name: str, content: str, display: str) -> None:
                self.kind = "text"
                self.name = name
                self.content = content
                self.display = display

        def _stub_image_markdown_src(_path: str) -> str:
            return "/public/citation_images/fake.png"

        monkeypatch.setattr(ui_app.cl, "Text", _StubText)
        monkeypatch.setattr(
            ui_app,
            "_image_markdown_src",
            _stub_image_markdown_src,
        )

        doc_with_image = NumberedCitation(
            reference_number=2,
            citation=DocumentCitation(
                raw_marker_text="<m>",
                citation_token="c2",
                source="corpus/a.pdf",
                chunk_id="c2",
                content="first excerpt",
                score=0.9,
                title="Doc A",
                image_path="/tmp/a.png",
                kind="image_description",
            ),
        )
        doc_without_image = NumberedCitation(
            reference_number=1,
            citation=DocumentCitation(
                raw_marker_text="<m>",
                citation_token="c1",
                source="corpus/b.pdf",
                chunk_id="c1",
                content="second excerpt",
                score=0.8,
                title="Doc B",
            ),
        )

        elements = ui_app._build_side_elements([doc_with_image, doc_without_image], lang="en")
        stub_elements = cast(list[Any], elements)

        assert len(stub_elements) == 2
        assert stub_elements[0].kind == "text"
        assert "### 1." in stub_elements[0].content
        assert "Doc B" in stub_elements[0].content
        assert stub_elements[1].kind == "text"
        assert "### 2." in stub_elements[1].content
        assert "Doc A" in stub_elements[1].content
        assert "/public/citation_images/fake.png" in stub_elements[1].content
        assert stub_elements[0].name == "(1)"
        assert stub_elements[1].name == "(2)"


class TestImageMarkdownSrc:
    def test_returns_public_url_and_copies_image(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import src.chatbot.ui.app as ui_app

        monkeypatch.chdir(tmp_path)

        source = tmp_path / "source.png"
        payload = b"\x89PNG\r\n\x1a\nFAKE"
        source.write_bytes(payload)

        src = ui_app._image_markdown_src(str(source))

        assert src is not None
        assert src.startswith("/public/citation_images/")

        written = Path(src.lstrip("/"))
        assert written.is_file()
        assert written.read_bytes() == payload

    def test_returns_none_for_missing_image(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import src.chatbot.ui.app as ui_app

        monkeypatch.chdir(tmp_path)

        src = ui_app._image_markdown_src("/tmp/definitely-does-not-exist.png")

        assert src is None
