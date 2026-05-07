# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared Qdrant infrastructure — used by both ingest (write) and chatbot (read).

Centralising the Qdrant client construction and BM25 sparse embedding here
removes the previous ``chatbot → ingest`` dependency by hoisting the few
collaborators that genuinely span both features into a horizontal package.
"""

from ._document_store import build_qdrant_document_store

__all__ = ["build_qdrant_document_store"]
