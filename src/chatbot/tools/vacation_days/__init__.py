# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public surface of the vacation_days tool package.

Exports the package-level types that other modules may reasonably depend on:
the tool itself, its adapter/service boundary, and the vacation-days-specific
auth contract.
"""

from src.chatbot.tools.vacation_days.adapter import SimulatedVacationDaysAdapter
from src.chatbot.tools.vacation_days.auth import (
    UsernamePasswordCredentials,
    VacationDaysAuth,
    VacationDaysCredentialStore,
)
from src.chatbot.tools.vacation_days.keys import VacationDaysCallKey
from src.chatbot.tools.vacation_days.service import (
    ToolAuthenticationError,
    VacationDaysInput,
    VacationDaysOutput,
    VacationDaysService,
)
from src.chatbot.tools.vacation_days.tool import VacationDaysTool

__all__ = [
    "SimulatedVacationDaysAdapter",
    "ToolAuthenticationError",
    "UsernamePasswordCredentials",
    "VacationDaysAuth",
    "VacationDaysCallKey",
    "VacationDaysCredentialStore",
    "VacationDaysInput",
    "VacationDaysOutput",
    "VacationDaysService",
    "VacationDaysTool",
]
