"""Unit tests for vacation_days: schema validation, auth, adapter, and tool."""

import pytest
from pydantic import ValidationError

from src.chatbot.app.protocols import ToolContext
from src.chatbot.tools.vacation_days import (
    InteractiveVacationDaysAuthSession,
    SimulatedVacationDaysAdapter,
    VacationDaysTool,
)
from src.chatbot.tools.vacation_days.auth import (
    AskUser,
    UsernamePasswordCredentials,
    VacationDaysAuth,
)
from src.chatbot.tools.vacation_days.service import (
    ToolAuthenticationError,
    VacationDaysInput,
    VacationDaysOutput,
    VacationDaysService,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ask_user(responses: list[str | None]) -> tuple[AskUser, list[str]]:
    """Return an ``AskUser`` callable that yields scripted responses, plus a prompts log."""
    prompts: list[str] = []
    idx = 0

    async def ask_user(prompt: str) -> str | None:
        nonlocal idx
        prompts.append(prompt)
        if idx < len(responses):
            resp = responses[idx]
            idx += 1
            return resp
        return None

    return ask_user, prompts


def _make_tool(
    *,
    ask_responses: list[str | None] | None = None,
    auth: VacationDaysAuth | None = None,
    service: VacationDaysService | None = None,
) -> VacationDaysTool:
    ask_user, _ = _make_ask_user(ask_responses or [])
    vacation_days_service = service or SimulatedVacationDaysAdapter()
    tool_auth = auth or InteractiveVacationDaysAuthSession(
        ask_user=ask_user,
        service_label="test service",
    )
    return VacationDaysTool(service=vacation_days_service, auth=tool_auth)


class _FakeVacationDaysAuth:
    def __init__(self, credentials: UsernamePasswordCredentials | None) -> None:
        self._credentials = credentials
        self.clear_called = False
        self.get_credentials_calls = 0

    async def get_credentials(self) -> UsernamePasswordCredentials | None:
        self.get_credentials_calls += 1
        return self._credentials

    def clear_credentials(self) -> None:
        self.clear_called = True
        self._credentials = None


class _FakeVacationDaysService:
    def __init__(
        self,
        *,
        result: VacationDaysOutput | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[VacationDaysInput, str, str]] = []

    async def get_vacation_days(
        self,
        tool_input: VacationDaysInput,
        username: str,
        password: str,
    ) -> VacationDaysOutput:
        self.calls.append((tool_input, username, password))
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise AssertionError("Fake service requires a result or an error")
        return self._result


# ---------------------------------------------------------------------------
# Pydantic schema tests
# ---------------------------------------------------------------------------


class TestVacationDaysInput:
    def test_accepts_explicit_year(self) -> None:
        model = VacationDaysInput(year=2025)
        assert model.year == 2025

    def test_rejects_missing_year(self) -> None:
        with pytest.raises(ValidationError):
            VacationDaysInput()  # type: ignore[call-arg]


class TestVacationDaysOutput:
    def test_constructs_with_all_fields(self) -> None:
        output = VacationDaysOutput(
            employee_username="alice",
            year=2025,
            total_days=25,
            used_days=8,
            remaining_days=17,
        )
        assert output.remaining_days == 17
        assert output.employee_username == "alice"

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            VacationDaysOutput(  # type: ignore[call-arg]
                employee_username="alice",
                year=2025,
            )


# ---------------------------------------------------------------------------
# SimulatedVacationDaysAdapter
# ---------------------------------------------------------------------------


class TestSimulatedVacationDaysAdapter:
    @pytest.mark.asyncio
    async def test_returns_correct_data_for_known_user(self) -> None:
        adapter = SimulatedVacationDaysAdapter()
        result = await adapter.get_vacation_days(
            VacationDaysInput(year=2025), username="alice", password="alice123"
        )
        assert result.employee_username == "alice"
        assert result.year == 2025
        assert result.total_days == 25
        assert result.used_days == 8
        assert result.remaining_days == 17

    @pytest.mark.asyncio
    async def test_remaining_days_is_total_minus_used(self) -> None:
        adapter = SimulatedVacationDaysAdapter()
        result = await adapter.get_vacation_days(
            VacationDaysInput(year=2024), username="bob", password="bob123"
        )
        assert result.remaining_days == result.total_days - result.used_days

    @pytest.mark.asyncio
    async def test_raises_auth_error_for_wrong_password(self) -> None:
        adapter = SimulatedVacationDaysAdapter()
        with pytest.raises(ToolAuthenticationError):
            await adapter.get_vacation_days(
                VacationDaysInput(year=2025), username="alice", password="wrong"
            )

    @pytest.mark.asyncio
    async def test_raises_auth_error_for_unknown_user(self) -> None:
        adapter = SimulatedVacationDaysAdapter()
        with pytest.raises(ToolAuthenticationError):
            await adapter.get_vacation_days(
                VacationDaysInput(year=2025), username="nonexistent", password="anything"
            )

    @pytest.mark.asyncio
    async def test_year_is_reflected_in_output(self) -> None:
        adapter = SimulatedVacationDaysAdapter()
        result = await adapter.get_vacation_days(
            VacationDaysInput(year=2023), username="demo", password="demo"
        )
        assert result.year == 2023

    @pytest.mark.asyncio
    async def test_all_simulated_users_authenticate(self) -> None:
        adapter = SimulatedVacationDaysAdapter()
        for username, password in [
            ("demo", "demo"),
            ("alice", "alice123"),
            ("bob", "bob123"),
        ]:
            result = await adapter.get_vacation_days(
                VacationDaysInput(year=2025), username=username, password=password
            )
            assert result.employee_username == username


# ---------------------------------------------------------------------------
# InteractiveVacationDaysAuthSession
# ---------------------------------------------------------------------------


class TestInteractiveVacationDaysAuthSession:
    @pytest.mark.asyncio
    async def test_reprompts_when_collection_is_canceled(self) -> None:
        ask_user, prompts = _make_ask_user([])
        auth = InteractiveVacationDaysAuthSession(ask_user=ask_user)

        first_result = await auth.get_credentials()
        second_result = await auth.get_credentials()

        assert first_result is None
        assert second_result is None
        assert prompts == [
            "To access the vacation days service I need your credentials. Please enter your **username**:",
            "To access the vacation days service I need your credentials. Please enter your **username**:",
        ]

    @pytest.mark.asyncio
    async def test_collects_and_caches_credentials_on_first_use(self) -> None:
        ask_user, _ = _make_ask_user(["alice", "mypassword"])
        auth = InteractiveVacationDaysAuthSession(ask_user=ask_user)

        first_result = await auth.get_credentials()
        second_result = await auth.get_credentials()

        expected = UsernamePasswordCredentials(username="alice", password="mypassword")
        assert first_result == expected
        assert second_result == expected

    @pytest.mark.asyncio
    async def test_returns_none_when_user_cancels_username(self) -> None:
        ask_user, _ = _make_ask_user([None])
        auth = InteractiveVacationDaysAuthSession(ask_user=ask_user)

        result = await auth.get_credentials()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_user_cancels_password(self) -> None:
        ask_user, _ = _make_ask_user(["alice", None])
        auth = InteractiveVacationDaysAuthSession(ask_user=ask_user)

        result = await auth.get_credentials()

        assert result is None

    @pytest.mark.asyncio
    async def test_clear_removes_cached_credentials(self) -> None:
        ask_user, prompts = _make_ask_user(["u", "p", "u2", "p2"])
        auth = InteractiveVacationDaysAuthSession(ask_user=ask_user)

        first_result = await auth.get_credentials()
        auth.clear_credentials()
        second_result = await auth.get_credentials()

        assert first_result == UsernamePasswordCredentials(username="u", password="p")
        assert second_result == UsernamePasswordCredentials(username="u2", password="p2")
        assert len(prompts) == 4


# ---------------------------------------------------------------------------
# VacationDaysTool
# ---------------------------------------------------------------------------


class TestVacationDaysTool:
    @pytest.mark.asyncio
    async def test_happy_path_returns_formatted_result(self) -> None:
        auth = _FakeVacationDaysAuth(
            UsernamePasswordCredentials(username="alice", password="alice123")
        )
        service = _FakeVacationDaysService(
            result=VacationDaysOutput(
                employee_username="alice",
                year=2025,
                total_days=25,
                used_days=8,
                remaining_days=17,
            )
        )
        tool = _make_tool(auth=auth, service=service)
        result, events = await tool.execute({"year": 2025}, ToolContext(history=()))

        assert result["employee_username"] == "alice"
        assert result["year"] == 2025
        assert result["total_days"] == 25
        assert result["remaining_days"] == 17
        assert events == []
        assert auth.clear_called is False
        assert service.calls == [(VacationDaysInput(year=2025), "alice", "alice123")]

    @pytest.mark.asyncio
    async def test_auth_failure_clears_credentials_and_returns_message(self) -> None:
        auth = _FakeVacationDaysAuth(
            UsernamePasswordCredentials(username="alice", password="wrongpassword")
        )
        service = _FakeVacationDaysService(error=ToolAuthenticationError("bad credentials"))
        tool = _make_tool(auth=auth, service=service)
        result, events = await tool.execute({"year": 2025}, ToolContext(history=()))

        assert "error" in result
        assert "Authentication failed" in str(result["error"])
        assert events == []
        assert auth.clear_called is True

    @pytest.mark.asyncio
    async def test_invalid_args_returns_error_message(self) -> None:
        auth = _FakeVacationDaysAuth(
            UsernamePasswordCredentials(username="alice", password="alice123")
        )
        service = _FakeVacationDaysService(
            result=VacationDaysOutput(
                employee_username="alice",
                year=2025,
                total_days=25,
                used_days=8,
                remaining_days=17,
            )
        )
        tool = _make_tool(auth=auth, service=service)
        result, events = await tool.execute({"year": "not_an_int"}, ToolContext(history=()))

        assert "error" in result
        assert "Invalid arguments" in str(result["error"])
        assert events == []
        assert auth.get_credentials_calls == 0
        assert service.calls == []

    @pytest.mark.asyncio
    async def test_credential_cancellation_returns_message(self) -> None:
        tool = _make_tool(ask_responses=[None])
        result, events = await tool.execute({"year": 2025}, ToolContext(history=()))

        assert "error" in result
        assert "canceled" in str(result["error"]).lower()
        assert events == []

    @pytest.mark.asyncio
    async def test_collects_credentials_when_not_cached(self) -> None:
        service = _FakeVacationDaysService(
            result=VacationDaysOutput(
                employee_username="alice",
                year=2025,
                total_days=25,
                used_days=8,
                remaining_days=17,
            )
        )
        tool = _make_tool(ask_responses=["alice", "alice123"], service=service)
        result, events = await tool.execute({"year": 2025}, ToolContext(history=()))

        assert result["employee_username"] == "alice"
        assert result["remaining_days"] == 17
        assert events == []


# ---------------------------------------------------------------------------
# Shared fake
# ---------------------------------------------------------------------------
