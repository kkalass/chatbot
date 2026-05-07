# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :class:`VacationDaysTool`."""

from dataclasses import dataclass

import pytest

from src.chatbot.contracts.credentials import AuthRequiredException, UsernamePasswordCredentials
from src.chatbot.infrastructure.tools.vacation_days import (
    ToolAuthenticationError,
    VacationDaysInput,
    VacationDaysOutput,
)
from src.chatbot.infrastructure.tools.vacation_days import VacationDaysTool


@dataclass
class _StubCredentialStore:
    creds: UsernamePasswordCredentials | None
    cleared: bool = False

    def get_credentials(self, key: str) -> UsernamePasswordCredentials | None:
        return self.creds

    def set_credentials(self, key: str, username: str, password: str) -> None:
        self.creds = UsernamePasswordCredentials(username=username, password=password)

    def clear_credentials(self, key: str) -> None:
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
        auth = _StubCredentialStore(creds=UsernamePasswordCredentials(username="u", password="p"))
        tool = VacationDaysTool(service, auth)

        result = await tool.execute({"year": 2026})

        assert result == {"total_days": 30, "used_days": 10, "remaining_days": 20}
        assert service.calls[0][0].year == 2026
        assert auth.cleared is False

    @pytest.mark.asyncio
    async def test_invalid_arguments_return_error(self) -> None:
        tool = VacationDaysTool(
            _StubService(result=VacationDaysOutput(total_days=0, used_days=0, remaining_days=0)),
            _StubCredentialStore(creds=None),
        )
        result = await tool.execute({"year": "not-a-number"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_auth_required_raises(self) -> None:
        tool = VacationDaysTool(
            _StubService(result=VacationDaysOutput(total_days=0, used_days=0, remaining_days=0)),
            _StubCredentialStore(creds=None),
        )
        with pytest.raises(AuthRequiredException):
            await tool.execute({"year": 2026})

    @pytest.mark.asyncio
    async def test_auth_error_clears_credentials(self) -> None:
        auth = _StubCredentialStore(creds=UsernamePasswordCredentials(username="u", password="p"))
        tool = VacationDaysTool(_StubService(raise_auth_error=True), auth)

        result = await tool.execute({"year": 2026})

        assert "error" in result
        assert auth.cleared is True
