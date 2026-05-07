# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pure-type contracts for ingestion.

No framework imports beyond Haystack value types (Document, ByteStream); no
I/O, no orchestration logic. Both ``app/`` and ``infrastructure/`` import
from here.
"""
