"""Public surface of the vacation_days tool package.

Exports the package-level types that other modules may reasonably depend on:
the tool itself, its adapter/service boundary, and the vacation-days-specific
auth contract.
"""

from src.tools.vacation_days.adapter import SimulatedVacationDaysAdapter
from src.tools.vacation_days.auth import (
    InteractiveVacationDaysAuthSession,
    UsernamePasswordCredentials,
    VacationDaysAuth,
)
from src.tools.vacation_days.service import (
    ToolAuthenticationError,
    VacationDaysInput,
    VacationDaysOutput,
    VacationDaysService,
)
from src.tools.vacation_days.tool import VacationDaysTool

__all__ = [
    "InteractiveVacationDaysAuthSession",
    "SimulatedVacationDaysAdapter",
    "ToolAuthenticationError",
    "UsernamePasswordCredentials",
    "VacationDaysAuth",
    "VacationDaysInput",
    "VacationDaysOutput",
    "VacationDaysService",
    "VacationDaysTool",
]
