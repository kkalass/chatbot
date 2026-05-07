# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the login form interaction in the UI app."""

# pyright: reportPrivateUsage=false

import asyncio
from typing import ClassVar

import pytest

from src.chatbot.contracts.i18n import I18nMessage
from src.chatbot.contracts.process import AuthRequiredEvent
from src.chatbot.infrastructure.tools.vacation_days import VacationDaysCallKey
from src.chatbot.ui import app as ui_app


class _StubCustomElement:
    def __init__(self, name: str, props: dict[str, object], display: str = "inline") -> None:
        self.name = name
        self.props = props
        self.display = display


class _StubAskElementMessage:
    last_element_props: ClassVar[dict[str, object]] = {}
    send_result: ClassVar[dict[str, object] | None] = {"username": "alice", "password": "secret"}

    def __init__(self, content: str, element: object, timeout: int) -> None:
        if hasattr(element, "props"):
            type(self).last_element_props = element.props  # type: ignore[union-attr]

    async def send(self) -> dict[str, object] | None:
        return type(self).send_result


class TestAskLogin:
    def _make_event(self) -> AuthRequiredEvent:
        loop = asyncio.get_event_loop()
        future: asyncio.Future[bool] = loop.create_future()
        return AuthRequiredEvent(
            tool_name="get_vacation_days",
            credential_key="vacation_days",
            service_display_name=I18nMessage(key=VacationDaysCallKey.DISPLAY_NAME, args={}),
            credential_future=future,
        )

    @pytest.mark.asyncio
    async def test_returns_credentials_on_submit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ui_app.cl, "CustomElement", _StubCustomElement)
        monkeypatch.setattr(ui_app.cl, "AskElementMessage", _StubAskElementMessage)

        _StubAskElementMessage.send_result = {"username": " alice ", "password": "s3cr3t"}
        event = self._make_event()

        result = await ui_app._ask_login(event, lang="en")

        assert result == ("alice", "s3cr3t")
        assert _StubAskElementMessage.last_element_props["lang"] == "en"
        assert "Vacation Days" in str(_StubAskElementMessage.last_element_props["service_name"])

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ui_app.cl, "CustomElement", _StubCustomElement)
        monkeypatch.setattr(ui_app.cl, "AskElementMessage", _StubAskElementMessage)

        _StubAskElementMessage.send_result = None
        event = self._make_event()

        result = await ui_app._ask_login(event, lang="de")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_fields_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ui_app.cl, "CustomElement", _StubCustomElement)
        monkeypatch.setattr(ui_app.cl, "AskElementMessage", _StubAskElementMessage)

        _StubAskElementMessage.send_result = {"username": "", "password": ""}
        event = self._make_event()

        result = await ui_app._ask_login(event, lang="en")

        assert result is None
