# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid (dense + sparse) retriever with RRF fusion for Qdrant."""

from typing import Any, cast

import structlog
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.dataclasses import Document, SparseEmbedding
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantSparseEmbeddingRetriever,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry import trace

from src.chatbot.contracts.observability import SPAN_CHAT_RETRIEVER_QDRANT_RETRIEVE
from src.chatbot.contracts.retrieval import SourceChunk
from src.chatbot.infrastructure.embeddings_text import TextEmbedder
from src.chatbot.infrastructure.observability import (
    build_retriever_attributes,
)
from src.shared.observability import build_span_kind_attributes, to_attribute_text
from src.shared.qdrant.embeddings_sparse import build_sparse_text

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


def _trace_request(
    *,
    span: trace.Span,
    top_k: int,
) -> None:
    span.set_attributes(build_span_kind_attributes(OpenInferenceSpanKindValues.RETRIEVER))
    span.set_attribute("chat.retriever.top_k", top_k)


def _trace_response(
    *,
    span: trace.Span,
    query: str,
    chunks: list[SourceChunk],
) -> None:
    span.set_attributes(build_retriever_attributes(query=query, documents=chunks))
    span.set_attribute("chat.retriever.result_count", len(chunks))
    span.set_attribute(
        "chat.retriever.top_scores",
        to_attribute_text([round(chunk.score, 4) for chunk in chunks[:5]]),
    )


class QdrantHybridRetriever:
    """Hybrid retriever combining dense and sparse (BM25-style) search with RRF fusion."""

    def __init__(
        self,
        *,
        top_k: int,
        llm_top_k: int | None = None,
        document_store: QdrantDocumentStore,
        text_embedder: TextEmbedder,
    ) -> None:
        self._top_k = top_k
        self._embedder = text_embedder
        self._dense_retriever = QdrantEmbeddingRetriever(
            document_store=document_store,
            top_k=top_k,
            score_threshold=None,  # No threshold pre-fusion
        )
        self._sparse_retriever = QdrantSparseEmbeddingRetriever(
            document_store=document_store,
            top_k=top_k,
            score_threshold=None,
        )
        self._joiner = DocumentJoiner(
            join_mode="reciprocal_rank_fusion",
            sort_by_score=True,
            top_k=llm_top_k if llm_top_k is not None else top_k,
        )

    async def retrieve(
        self,
        query_dense: str,
        *,
        query_sparse: str | None = None,
    ) -> list[SourceChunk]:
        """Retrieve using both dense and sparse retrieval, fuse with RRF.

        *query_dense* is routed to the dense retriever. *query_sparse* is routed
        to sparse retrieval and falls back to *query_dense* when omitted.
        """
        with tracer.start_as_current_span(SPAN_CHAT_RETRIEVER_QDRANT_RETRIEVE) as span:
            _trace_request(span=span, top_k=self._top_k)

            # Note that embedding and qdrant is currently so much faster than the LLM later,
            # that we simply run serially. If we should get performance issues here,
            # we should parallelise dense/sparse embedding + retrieval.
            effective_dense = query_dense
            effective_sparse = query_sparse or query_dense
            logger.debug("retriever.embedding")
            embed_result: dict[str, Any] = self._embedder.run(text=effective_dense)
            embedding: list[float] = embed_result["embedding"]

            logger.debug("retriever.querying_hybrid", mode="dense+sparse+rrf")
            # Dense retrieval
            dense_result: dict[str, Any] = self._dense_retriever.run(query_embedding=embedding)
            dense_docs = cast(list[Document], dense_result.get("documents", []))

            sparse_docs: list[Document] = []
            sparse_query_embedding: SparseEmbedding | None = build_sparse_text(effective_sparse)
            if sparse_query_embedding is not None:
                sparse_result: dict[str, Any] = self._sparse_retriever.run(
                    query_sparse_embedding=sparse_query_embedding
                )
                sparse_docs = cast(list[Document], sparse_result.get("documents", []))

            logger.debug(
                "retriever.fusion",
                dense_hits=len(dense_docs),
                sparse_hits=len(sparse_docs),
            )
            # RRF fusion via DocumentJoiner
            join_result: dict[str, Any] = self._joiner.run(documents=[dense_docs, sparse_docs])
            fused_docs = join_result.get("documents", [])

            chunks = [
                SourceChunk(
                    content=doc.content or "",
                    source=doc.meta.get("source", "unknown"),
                    score=doc.score if doc.score is not None else 0.0,
                    chunk_id=doc.id,
                    title=doc.meta.get("title"),
                    author=doc.meta.get("author"),
                    publication_date=doc.meta.get("publication_date"),
                    source_url=doc.meta.get("source_url"),
                    page=doc.meta.get("page"),
                    kind=str(doc.meta.get("kind") or "text"),
                    image_path=doc.meta.get("image_path"),
                )
                for doc in fused_docs
                if doc.content
            ]
            logger.info("retriever.done", chunks_returned=len(chunks))
            _trace_response(span=span, query=query_dense, chunks=chunks)
            return chunks
