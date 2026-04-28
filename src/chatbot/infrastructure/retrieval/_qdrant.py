"""Qdrant retrieval implementation."""

from typing import Any

import structlog
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from opentelemetry import trace

from src.chatbot.app.protocols import Retriever, SourceChunk
from src.chatbot.infrastructure.embeddings_text import TextEmbedder
from src.chatbot.observability import to_attribute_text

from ._config import RetrieverConfig

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class QdrantRetriever:
    """Retriever backed by Qdrant vector search and an injected text embedder."""

    def __init__(
        self,
        config: "RetrieverConfig",
        document_store: QdrantDocumentStore,
        text_embedder: TextEmbedder,
    ) -> None:
        self._config = config
        self._embedder = text_embedder
        self._retriever = QdrantEmbeddingRetriever(
            document_store=document_store,
            top_k=config.top_k,
            score_threshold=config.score_threshold,
        )

    async def retrieve(self, query: str) -> list[SourceChunk]:
        """Embed a query and return ranked, filtered chunks from Qdrant."""
        with tracer.start_as_current_span("chat.retriever.qdrant.retrieve") as span:
            span.set_attribute("chat.query.preview", to_attribute_text(query))
            span.set_attribute("chat.retriever.top_k", self._config.top_k)
            span.set_attribute("chat.retriever.score_threshold", self._config.score_threshold)

            logger.debug("retriever.embedding")
            embed_result: dict[str, Any] = self._embedder.run(text=query)
            embedding: list[float] = embed_result["embedding"]

            logger.debug(
                "retriever.querying",
                top_k=self._config.top_k,
                score_threshold=self._config.score_threshold,
            )
            retrieval_result: dict[str, Any] = self._retriever.run(query_embedding=embedding)
            docs = retrieval_result["documents"]

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
                )
                for doc in docs
                if doc.content
            ]
            logger.info("retriever.done", chunks_returned=len(chunks))
            span.set_attribute("chat.retriever.result_count", len(chunks))
            span.set_attribute(
                "chat.retriever.result_preview",
                to_attribute_text(
                    [
                        {
                            "source": chunk.source,
                            "chunk_id": chunk.chunk_id,
                            "score": chunk.score,
                        }
                        for chunk in chunks[:5]
                    ]
                ),
            )
            return chunks


def build_qdrant_retriever(
    *,
    config: "RetrieverConfig",
    text_embedder: TextEmbedder,
) -> Retriever:
    """Build a Qdrant-backed retriever, constructing the document store from config."""
    document_store = QdrantDocumentStore(
        host=config.store_host,
        port=config.store_port,
        index=config.store_collection,
        embedding_dim=config.embedding_dim,
        similarity=config.store_similarity,
    )
    return QdrantRetriever(
        config=config,
        document_store=document_store,
        text_embedder=text_embedder,
    )
