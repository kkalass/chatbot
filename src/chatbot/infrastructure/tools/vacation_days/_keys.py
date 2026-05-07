# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""I18n message keys for the vacation_days tool.

Kept in a dedicated module so both :mod:`tool` and :mod:`auth` can import
the keys without introducing a circular dependency.
"""

from enum import StrEnum


class VacationDaysCallKey(StrEnum):
    """Message keys for :meth:`VacationDaysTool.describe_call` results and UI messages.

    The UI translation map must contain an entry for every value defined here.
    """

    DISPLAY_NAME = "vacation_days.display_name"
    QUERYING = "vacation_days.querying"
