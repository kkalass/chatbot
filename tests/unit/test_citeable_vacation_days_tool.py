# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :class:`VacationDaysTool`."""

from dataclasses import dataclass

import pytest

from src.chatbot.tools.vacation_days.auth import UsernamePasswordCredentials
from src.chatbot.tools.vacation_days.service import (
    ToolAuthenticationError,
    VacationDaysInput,
    VacationDaysOutput,
)
from src.chatbot.tools.vacation_days.tool import VacationDaysTool


@dataclass
class _StubAuth:
    creds: UsernamePasswordCredentials | None
    cleared: bool = False

    async def get_credentials(self) -> UsernamePasswordCredentials | None:
        return self.creds

    def clear_credentials(self) -> None:
        self.cleared = True


class _StubService:
    def __init__(
        self,
        *,
        result: VacationDaysOutput | None = None,
        raise_auth_error: bool = False,
    ) -> None:
        self._result = result
        self._raise = raise_auth_error
        self.calls: list[tuple[VacationDaysInput, str, str]] = []

    async def get_vacation_days(
        self, tool_input: VacationDaysInput, username: str, password: str
    ) -> VacationDaysOutput:
        self.calls.append((tool_input, username, password))
        if self._raise:
            raise ToolAuthenticationError
        assert self._result is not None
        return self._result


class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_balance_for_valid_args(self) -> None:
        service = _StubService(
            result=VacationDaysOutput(total_days=30, used_days=10, remaining_days=20)
        )
        auth = _StubAuth(creds=UsernamePasswordCredentials(username="u", password="p"))
        tool = VacationDaysTool(service, auth)

        result = await tool.execute({"year": 2026})

        assert result == {"total_days": 30, "used_days": 10, "remaining_days": 20}
        assert service.calls[0][0].year == 2026
        assert auth.cleared is False

    @pytest.mark.asyncio
    async def test_invalid_arguments_return_error(self) -> None:
        tool = VacationDaysTool(
            _StubService(result=VacationDaysOutput(total_days=0, used_days=0, remaining_days=0)),
            _StubAuth(creds=None),
        )
        result = await tool.execute({"year": "not-a-number"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_credential_cancellation_returns_error(self) -> None:
        tool = VacationDaysTool(
            _StubService(result=VacationDaysOutput(total_days=0, used_days=0, remaining_days=0)),
            _StubAuth(creds=None),
        )
        result = await tool.execute({"year": 2026})
        assert "canceled" in str(result.get("error", "")).lower()

    @pytest.mark.asyncio
    async def test_auth_error_clears_credentials(self) -> None:
        auth = _StubAuth(creds=UsernamePasswordCredentials(username="u", password="p"))
        tool = VacationDaysTool(_StubService(raise_auth_error=True), auth)

        result = await tool.execute({"year": 2026})

        assert "error" in result
        assert auth.cleared is True
