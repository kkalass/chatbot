# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public surface of the vacation_days tool package.

Exports the package-level types that other modules may reasonably depend on:
the tool itself, its adapter/service boundary, and the vacation-days-specific
key constants.
"""

from ._adapter import SimulatedVacationDaysAdapter
from ._keys import VacationDaysCallKey
from ._service import (
    ToolAuthenticationError,
    VacationDaysInput,
    VacationDaysOutput,
    VacationDaysService,
)
from ._tool import VacationDaysTool

__all__ = [
    "SimulatedVacationDaysAdapter",
    "ToolAuthenticationError",
    "VacationDaysCallKey",
    "VacationDaysInput",
    "VacationDaysOutput",
    "VacationDaysService",
    "VacationDaysTool",
]
