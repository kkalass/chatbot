# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BM25 sparse embedding helpers used by both ingest and chatbot.

The model is fastembed's ``Qdrant/bm25`` (~25 kB IDF table). The query side
needs :func:`build_sparse_text` to embed the user query; the ingest side needs
:class:`SparseDocumentEmbedder` to embed the corpus in batches.
"""

from ._impl import SparseDocumentEmbedder, build_sparse_text

__all__ = ["SparseDocumentEmbedder", "build_sparse_text"]
