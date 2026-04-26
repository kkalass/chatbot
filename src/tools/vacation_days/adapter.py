"""Simulated HR service adapter with Pydantic boundary models.

This module is the only place in the application that knows about the
simulated vacation days data.  It is a stand-in for a real HTTP client and
must remain free of LLM, orchestrator, and UI imports.
"""

import structlog

from src.tools.vacation_days.service import (
    ToolAuthenticationError,
    VacationDaysInput,
    VacationDaysOutput,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Simulated HR service adapter
# ---------------------------------------------------------------------------

_SIMULATED_USERS: dict[str, tuple[str, int, int]] = {
    # username -> (password, total_days, used_days)
    "demo": ("demo", 25, 10),
    "alice": ("alice123", 25, 8),
    "bob": ("bob123", 20, 15),
}


class SimulatedVacationDaysAdapter:
    """Satisfies :class:`~src.tools.vacation_days.service.VacationDaysService`.

    Implements the vacation days lookup with hardcoded data.
    Intended for local development and demo use only.
    Concrete binding is in ``_build_vacation_days_tool`` in ``src/ui/app.py``.

    Raises:
        ToolAuthenticationError: When ``username`` is unknown or ``password``
            does not match.
    """

    async def get_vacation_days(
        self,
        tool_input: VacationDaysInput,
        username: str,
        password: str,
    ) -> VacationDaysOutput:
        log = logger.bind(username=username, year=tool_input.year)
        record = _SIMULATED_USERS.get(username)
        if record is None or password != record[0]:
            log.warning("adapter.auth_failed")
            raise ToolAuthenticationError(f"Authentication failed for user '{username}'")

        _, total_days, used_days = record
        result = VacationDaysOutput(
            employee_username=username,
            year=tool_input.year,
            total_days=total_days,
            used_days=used_days,
            remaining_days=total_days - used_days,
        )
        log.info("adapter.vacation_days_retrieved", remaining=result.remaining_days)
        return result
