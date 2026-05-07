# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ingestion pipeline: load, chunk, embed, and index txt/md/pdf documents.

This module owns the Haystack orchestration logic for ingestion: format dispatch,
batching, splitting, embedding, and writing.  Format handlers are injected by
the composition root (``ingest/cli/composition.py``) so that the pipeline does
not depend on converter-specific construction logic.

The public surface exposed to CLI and tests is :class:`IngestionPipeline`,
:class:`IngestionConfig`, and :class:`FormatHandler`.
"""

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import structlog
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from src.ingest.contracts.converters import FileConverter
from src.ingest.infrastructure.embeddings_document import DocumentEmbedder
from src.shared.qdrant.embeddings_sparse import SparseDocumentEmbedder

logger = structlog.get_logger(__name__)


_MICRO_BATCH_SIZE = 32


@dataclass(frozen=True)
class IngestionConfig:
    """Construction-time config for the ingestion pipeline.

    Args:
        split_length: Number of units (words or sentences) per chunk.
        split_overlap: Overlap in units between adjacent chunks.
        batch_size: Number of source files processed per micro-batch in
            :meth:`~IngestionPipeline.ingest_corpus`.
    """

    split_length: int = 200
    split_overlap: int = 20
    batch_size: int = _MICRO_BATCH_SIZE


@dataclass(frozen=True)
class FormatHandler:
    """One per-suffix dispatch entry for the ingestion pipeline.

    Constructed by the composition root and injected into
    :class:`IngestionPipeline`; the pipeline only consumes these and never
    knows which converter implementations are wired in.
    """

    suffix: str
    converter_factory: Callable[[], FileConverter]
    splitter_factory: Callable[[IngestionConfig], DocumentSplitter]


def load_sidecar_meta(doc_path: Path) -> dict[str, str]:
    """Load ``<doc_path>.meta.json`` sidecar if present, return empty dict otherwise.

    All values are coerced to strings for uniform Haystack metadata handling.
    Malformed files are logged and silently ignored.
    """
    meta_path = Path(str(doc_path) + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        raw: object = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("ingestion.sidecar_meta_not_a_dict", path=str(meta_path))
            return {}
        pairs = cast(dict[str, object], raw)
        return {str(k): str(v) for k, v in pairs.items()}
    except Exception as exc:
        logger.warning("ingestion.sidecar_meta_load_failed", path=str(meta_path), error=str(exc))
        return {}


def _convert(converter: FileConverter, paths: list[Path]) -> list[Document]:
    meta = [{**load_sidecar_meta(p), "source": str(p)} for p in paths]
    result = converter.run(sources=list(paths), meta=meta)
    return result["documents"]  # type: ignore[return-value]


def _split_documents(splitter: DocumentSplitter, docs: list[Document]) -> list[Document]:
    result = splitter.run(documents=docs)
    return result["documents"]  # type: ignore[arg-type]


class IngestionPipeline:
    """Orchestrates loading, chunking, embedding, and indexing of text documents.

    Each call to :meth:`ingest` runs the full pipeline for the supplied file
    paths.  Documents with unchanged IDs are overwritten (OVERWRITE policy ensures
    re-ingestion replaces stale vectors rather than duplicating them).

    The embedder is injected at construction time so that the ingestion logic
    is independent of the embedding backend.  Use
    :func:`~src.infrastructure.embeddings.build_document_embedder` to build
    the configured embedder and pass it here.

    Args:
        config: Validated construction-time configuration.
        document_store: Pre-constructed Haystack document store.
        embedder: Document embedder satisfying the
            :class:`~src.infrastructure.embeddings.DocumentEmbedder` Protocol.
        format_handlers: Per-suffix dispatch table assembled by the composition
            root; the pipeline only consults this and never constructs
            converters itself.
    """

    def __init__(
        self,
        config: IngestionConfig,
        document_store: DocumentStore,
        embedder: DocumentEmbedder,
        format_handlers: Sequence[FormatHandler],
        sparse_embedder: SparseDocumentEmbedder | None = None,
    ) -> None:
        self._config = config
        self._document_store = document_store
        self._embedder = embedder
        self._sparse_embedder = sparse_embedder
        self._format_handlers = tuple(format_handlers)
        self._format_handler_by_suffix = {h.suffix: h for h in self._format_handlers}

    def _ingest_batch(self, file_paths: list[Path]) -> int:
        """Ingest one bounded file batch and return the number of chunks written."""
        by_suffix: dict[str, list[Path]] = {}
        for p in file_paths:
            by_suffix.setdefault(p.suffix, []).append(p)

        unsupported_suffixes = by_suffix.keys() - self._format_handler_by_suffix.keys()
        if unsupported_suffixes:
            logger.info(
                "ingestion.skipping_unsupported",
                count=sum(len(by_suffix[s]) for s in unsupported_suffixes),
                suffixes=list(unsupported_suffixes),
            )

        total_supported = sum(
            len(by_suffix.get(handler.suffix, [])) for handler in self._format_handlers
        )
        if total_supported == 0:
            logger.info("ingestion.no_supported_files", total=len(file_paths))
            return 0

        logger.info(
            "ingestion.converting",
            counts={
                handler.suffix: len(by_suffix[handler.suffix])
                for handler in self._format_handlers
                if handler.suffix in by_suffix
            },
        )
        raw_document_count = 0
        chunks: list[Document] = []
        for handler in self._format_handlers:
            paths = by_suffix.get(handler.suffix, [])
            if paths:
                converted = _convert(handler.converter_factory(), paths)
                raw_document_count += len(converted)
                splitter = handler.splitter_factory(self._config)
                chunks.extend(_split_documents(splitter, converted))

        logger.info("ingestion.splitting", document_count=raw_document_count)

        logger.info("ingestion.embedding_and_writing", chunk_count=len(chunks))
        embed_result = self._embedder.run(documents=chunks)
        embedded_chunks = cast(list[Document], embed_result["documents"])

        if self._sparse_embedder:
            logger.info("ingestion.computing_sparse_vectors", chunk_count=len(embedded_chunks))
            sparse_result = self._sparse_embedder.run(documents=embedded_chunks)
            embedded_chunks = cast(list[Document], sparse_result["documents"])
            sparse_chunk_count = sum(
                1 for chunk in embedded_chunks if chunk.sparse_embedding is not None
            )
            logger.info("ingestion.sparse_vectors_computed", chunk_count=sparse_chunk_count)

        written: int = self._document_store.write_documents(
            embedded_chunks, policy=DuplicatePolicy.OVERWRITE
        )
        logger.info("ingestion.done", chunks_written=written)
        return written

    def ingest(self, file_paths: Iterable[Path]) -> int:
        """Ingest files from any iterable source using bounded file micro-batches.

        The iterable may be a list, generator, or ``Path.rglob`` iterator.
        Non-files and ingestion sidecars (``*.meta.json`` + corpus README) are
        filtered out before batching.
        """
        total_written = 0
        batch_size = self._config.batch_size
        batch: list[Path] = []
        discovered_files = 0

        for candidate in file_paths:
            if not candidate.is_file():
                continue
            if candidate.name.endswith(".meta.json") or candidate.name == "README.md":
                continue
            discovered_files += 1
            batch.append(candidate)

            if len(batch) < batch_size:
                continue

            written = self._ingest_batch(batch)
            total_written += written
            logger.debug(
                "ingestion.batch_done",
                batch_start=discovered_files - len(batch),
                batch_size=len(batch),
                chunks_written=written,
            )

            batch = []

        if batch:
            written = self._ingest_batch(batch)
            total_written += written
            logger.debug(
                "ingestion.batch_done",
                batch_start=discovered_files - len(batch),
                batch_size=len(batch),
                chunks_written=written,
            )

        logger.info("ingestion.discovered", total=discovered_files)
        logger.info("ingestion.corpus_done", total_chunks_written=total_written)
        return total_written

    def ingest_corpus(self, corpus_path: Path) -> int:
        """Discover and ingest all supported files under *corpus_path*.

        Discovery stays in this method; filtering + bounded batching are handled
        by :meth:`ingest`.
        """
        if not corpus_path.exists():
            logger.warning("ingestion.corpus_path_missing", path=str(corpus_path))
            return 0
        return self.ingest(corpus_path.rglob("*"))
