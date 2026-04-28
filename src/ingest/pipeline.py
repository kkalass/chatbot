"""Ingestion pipeline: load, chunk, embed, and index txt/md/pdf documents.

This module owns the Haystack pipeline logic for ingestion.  Infrastructure
concerns (which embedder backend to use, how to build it) live in
``src.infrastructure.embeddings``.  The pipeline accepts a
:class:`~src.infrastructure.embeddings.DocumentEmbedder` injected at
construction time so that tests can run without any external services.

The public surface exposed to CLI and tests is :class:`IngestionPipeline`
and :class:`IngestionConfig`.
"""

import io
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import structlog
from haystack.components.converters import MarkdownToDocument, TextFileToDocument
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ByteStream, Document
from haystack.document_stores.types import DuplicatePolicy
from pypdf import PdfReader

from src.ingest.infrastructure.document_store import DocumentStore
from src.ingest.infrastructure.embeddings_document import DocumentEmbedder

logger = structlog.get_logger(__name__)


_MICRO_BATCH_SIZE = 32


class _FileConverter(Protocol):
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...


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


def _build_word_splitter(config: IngestionConfig) -> DocumentSplitter:
    return DocumentSplitter(
        split_by="word",
        split_length=config.split_length,
        split_overlap=config.split_overlap,
    )


@dataclass(frozen=True)
class _FormatHandler:
    suffix: str
    converter_factory: Callable[[], _FileConverter]
    splitter_factory: Callable[[IngestionConfig], DocumentSplitter]


class _PdfPageConverter:
    """Extraction-first PDF converter: extracts text per page as individual documents.

    Each non-empty page becomes one document with page-level provenance metadata
    (``page``, ``total_pages``).  Extraction-first means the PDF is treated as a
    structured text document routed through the standard text-chunking pipeline,
    rather than embedded as a multi-modal artefact.

    Uses Haystack's ``get_bytestream_from_source`` / ``normalize_metadata``
    utilities for source loading and meta normalisation so that ``ByteStream``
    inputs (including their embedded meta) work correctly — the same contract
    the built-in Haystack converters (``PyPDFToDocument``, etc.) honour.

    Note that we do not use the built-in Haystack ``PyPDFToDocument`` converter because it does not
    preserve page-level provenance metadata, which is essential for accurate source citation in the UI.

    Conforms to :class:`_FileConverter`.
    """

    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Convert PDF sources to per-page :class:`~haystack.dataclasses.Document` instances."""
        meta_list: list[dict[str, Any]] = normalize_metadata(meta, sources_count=len(sources))

        documents: list[Document] = []
        for source, caller_meta in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as exc:
                logger.warning(
                    "pdf_converter.source_load_failed",
                    source=str(source),
                    error=str(exc),
                )
                continue
            # Merge ByteStream-embedded meta first (lower priority than caller meta).
            base_meta: dict[str, Any] = {**bytestream.meta, **caller_meta}
            try:
                reader = PdfReader(io.BytesIO(bytestream.data))
                total_pages = len(reader.pages)
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if not text.strip():
                        logger.debug(
                            "pdf_converter.skipping_empty_page",
                            source=str(source),
                            page=page_idx + 1,
                        )
                        continue
                    page_meta: dict[str, Any] = {
                        **base_meta,
                        "page": str(page_idx + 1),
                        "total_pages": str(total_pages),
                    }
                    documents.append(Document(content=text, meta=page_meta))
            except Exception as exc:
                logger.warning(
                    "pdf_converter.extraction_failed",
                    source=str(source),
                    error=str(exc),
                )
        return {"documents": documents}


# Maps each supported file suffix to both converter and splitting strategy.
# Extend here to add new format support — no other code needs to change.
_FORMAT_HANDLERS: tuple[_FormatHandler, ...] = (
    _FormatHandler(
        suffix=".txt",
        converter_factory=TextFileToDocument,
        splitter_factory=_build_word_splitter,
    ),
    _FormatHandler(
        suffix=".md",
        converter_factory=MarkdownToDocument,
        splitter_factory=_build_word_splitter,
    ),
    _FormatHandler(
        suffix=".pdf",
        converter_factory=_PdfPageConverter,
        splitter_factory=_build_word_splitter,
    ),
)
_FORMAT_HANDLER_BY_SUFFIX = {handler.suffix: handler for handler in _FORMAT_HANDLERS}


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


def _convert(converter: _FileConverter, paths: list[Path]) -> list[Document]:
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
    """

    def __init__(
        self,
        config: IngestionConfig,
        document_store: DocumentStore,
        embedder: DocumentEmbedder,
    ) -> None:
        self._config = config
        self._document_store = document_store
        self._embedder = embedder

    def _ingest_batch(self, file_paths: list[Path]) -> int:
        """Ingest one bounded file batch and return the number of chunks written."""
        by_suffix: dict[str, list[Path]] = {}
        for p in file_paths:
            by_suffix.setdefault(p.suffix, []).append(p)

        unsupported_suffixes = by_suffix.keys() - _FORMAT_HANDLER_BY_SUFFIX.keys()
        if unsupported_suffixes:
            logger.info(
                "ingestion.skipping_unsupported",
                count=sum(len(by_suffix[s]) for s in unsupported_suffixes),
                suffixes=list(unsupported_suffixes),
            )

        total_supported = sum(
            len(by_suffix.get(handler.suffix, [])) for handler in _FORMAT_HANDLERS
        )
        if total_supported == 0:
            logger.info("ingestion.no_supported_files", total=len(file_paths))
            return 0

        logger.info(
            "ingestion.converting",
            counts={
                handler.suffix: len(by_suffix[handler.suffix])
                for handler in _FORMAT_HANDLERS
                if handler.suffix in by_suffix
            },
        )
        raw_document_count = 0
        chunks: list[Document] = []
        for handler in _FORMAT_HANDLERS:
            paths = by_suffix.get(handler.suffix, [])
            if paths:
                converted = _convert(handler.converter_factory(), paths)
                raw_document_count += len(converted)
                splitter = handler.splitter_factory(self._config)
                chunks.extend(_split_documents(splitter, converted))

        logger.info("ingestion.splitting", document_count=raw_document_count)

        logger.info("ingestion.embedding_and_writing", chunk_count=len(chunks))
        embed_result = self._embedder.run(documents=chunks)
        embedded = cast(list[Document], embed_result["documents"])
        written: int = self._document_store.write_documents(
            embedded, policy=DuplicatePolicy.OVERWRITE
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
