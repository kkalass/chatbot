# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Localizable message contract and JSON value type.

``JsonObject`` lives here (rather than in :mod:`tools`) because
:class:`I18nMessage` consumes it directly and would otherwise import from
``tools`` — the dependency direction needs to flow towards primitives.
"""

from dataclasses import dataclass
from typing import Any

# JSON object — the canonical in-memory representation of structured data at
# protocol boundaries. ``Any`` is intentional: a recursive type alias caused
# more friction than it was worth (e.g. .get() calls on dicts).
type JsonObject = dict[str, Any]


@dataclass(frozen=True)
class I18nMessage:
    """Localizable message token — a key + interpolation args for UI rendering."""

    key: str
    args: JsonObject
