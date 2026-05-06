# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BM25 sparse embedding for hybrid retrieval in Qdrant.

Uses fastembed's ``Qdrant/bm25`` model which ships pre-computed IDF weights
from a large multilingual web corpus.  This gives real BM25 scoring (TF*IDF)
with proper Unicode tokenisation for both German and English — without building
a corpus-specific IDF index or implementing a custom tokenizer.

The model is ~25 kB (BM25 coefficient table, no neural network).  It is
downloaded once to ``~/.cache/fastembed`` and loaded from disk on subsequent
calls.
"""

from dataclasses import replace
from typing import Any

import structlog
from fastembed.sparse import SparseTextEmbedding as _FastembedModel  # type: ignore[import-untyped]
from haystack.dataclasses import Document, SparseEmbedding

logger = structlog.get_logger(__name__)

_BM25_MODEL_NAME = "Qdrant/bm25"

# Module-level singleton — model is tiny and loads instantly after first download.
_model: _FastembedModel | None = None  # type: ignore[valid-type]


def _get_model() -> _FastembedModel:  # type: ignore[valid-type]
    global _model
    if _model is None:
        _model = _FastembedModel(model_name=_BM25_MODEL_NAME)
    return _model


def _document_to_text(doc: Document) -> str:
    return doc.content or ""


def build_sparse_text(text: str) -> SparseEmbedding | None:
    """Build a BM25 sparse embedding from plain text.

    Delegates to the fastembed ``Qdrant/bm25`` model so that both ingestion
    and query paths use identical tokenisation and IDF weights.
    """
    results = list(_get_model().embed([text]))
    if not results:
        return None
    indices: list[int] = results[0].indices.tolist()  # type: ignore[attr-defined]
    if not indices:
        return None
    values: list[float] = results[0].values.tolist()  # type: ignore[attr-defined]
    return SparseEmbedding(indices=indices, values=values)


class SparseDocumentEmbedder:
    """Ingestion-time BM25 sparse embedding using fastembed's ``Qdrant/bm25`` model.

    Embeds a batch of documents in one fastembed call and attaches the resulting
    sparse vectors as ``sparse_embedding`` for storage in Qdrant.
    """

    def run(self, documents: list[Document]) -> dict[str, Any]:
        """Embed *documents* in batch and attach BM25 sparse vectors."""
        if not documents:
            return {"documents": []}
        texts = [_document_to_text(doc) for doc in documents]
        model = _get_model()
        embeddings = list(model.embed(texts))
        result: list[Document] = []
        for doc, emb in zip(documents, embeddings, strict=True):
            indices: list[int] = emb.indices.tolist()  # type: ignore[attr-defined]
            if indices:
                result.append(
                    replace(
                        doc,
                        sparse_embedding=SparseEmbedding(
                            indices=indices,
                            values=emb.values.tolist(),  # type: ignore[attr-defined]
                        ),
                    )
                )
            else:
                result.append(doc)
        return {"documents": result}
