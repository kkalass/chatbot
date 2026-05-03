"""Tests for the citeable :class:`VacationDaysTool`."""

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from src.chatbot.app.citation import (
    DocumentRawCitation,
    ToolCitation,
    ToolRawCitation,
)
from src.chatbot.app.protocols import JsonObject
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


@dataclass(frozen=True)
class _StaticContext:
    by_id: dict[str, JsonObject]

    def tool_result_for(self, tool_call_id: str) -> JsonObject | None:
        return self.by_id.get(tool_call_id)

    def tool_results_for(self, tool_name: str) -> Sequence[JsonObject]:
        return tuple(self.by_id.values())


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


class TestCiteInstructions:
    def test_fragment_documents_required_fields(self) -> None:
        tool = VacationDaysTool(
            _StubService(result=VacationDaysOutput(total_days=0, used_days=0, remaining_days=0)),
            _StubAuth(creds=None),
        )
        fragment = tool.cite_instructions().prompt_fragment

        assert "get_vacation_days" in fragment
        assert "tool_call_id" in fragment
        assert '"tool_call"' in fragment


class TestFormatForHistory:
    def test_returns_compact_json(self) -> None:
        tool = VacationDaysTool(
            _StubService(result=VacationDaysOutput(total_days=0, used_days=0, remaining_days=0)),
            _StubAuth(creds=None),
        )
        rendered = tool.format_for_history({"total_days": 30, "used_days": 10})
        assert rendered == '{"total_days": 30, "used_days": 10}'


class TestValidateAndEnrich:
    def _tool(self) -> VacationDaysTool:
        return VacationDaysTool(
            _StubService(result=VacationDaysOutput(total_days=0, used_days=0, remaining_days=0)),
            _StubAuth(creds=None),
        )

    def test_returns_none_for_wrong_raw_kind(self) -> None:
        tool = self._tool()
        raw = DocumentRawCitation(tool_call_id="tc1", source="s", chunk_id="c", raw_marker_text="m")
        assert tool.validate_and_enrich(raw, _StaticContext(by_id={})) is None

    def test_returns_tool_citation_when_result_present(self) -> None:
        tool = self._tool()
        ctx = _StaticContext(by_id={"tc1": {"total_days": 30, "remaining_days": 20}})
        raw = ToolRawCitation(tool_call_id="tc1", raw_marker_text="m")

        result = tool.validate_and_enrich(raw, ctx)

        assert isinstance(result, ToolCitation)
        assert result.tool_call_id == "tc1"
        assert result.tool_name == "get_vacation_days"
        assert result.result == {"total_days": 30, "remaining_days": 20}
        assert result.raw_marker_text == "m"

    def test_returns_none_when_no_matching_tool_result(self) -> None:
        tool = self._tool()
        ctx = _StaticContext(by_id={})
        raw = ToolRawCitation(tool_call_id="missing", raw_marker_text="m")
        assert tool.validate_and_enrich(raw, ctx) is None
