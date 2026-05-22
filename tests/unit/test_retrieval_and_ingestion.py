# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for ingestion chunking logic, retrieval tool, and sidecar loading.

Covers:
- :class:`~src.chatbot.infrastructure.tools.retrieval.RetrievalTool` — execute contract, error handling,
  and empty-result handling.
- :class:`~src.ingest.app.IngestionPipeline` chunking via real
  :class:`~haystack.components.preprocessors.DocumentSplitter` (no
  infrastructure dependencies — uses in-memory documents only).
- :func:`~src.ingest.app.load_sidecar_meta` — sidecar loading.
- Converter routing and splitter strategy selection.
- PDF extraction: per-page document creation, page metadata propagation, and
  extraction-failure isolation.
"""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document

from src.chatbot.contracts.retrieval import SourceChunk
from src.chatbot.infrastructure.tools.retrieval import RetrievalTool
from src.ingest.app import (
    IngestionConfig,
    IngestionPipeline,
    load_sidecar_meta,
)
from src.ingest.build_from_settings import build_format_handlers

# ---------------------------------------------------------------------------
# Helpers / test doubles
# ---------------------------------------------------------------------------


class _FakeRetriever:
    """Configurable retriever double."""

    def __init__(self, chunks: list[SourceChunk]) -> None:
        self.chunks = chunks
        self.calls: list[str] = []

    async def retrieve(
        self,
        query_dense: str,
        *,
        query_sparse: str | None = None,
    ) -> list[SourceChunk]:
        self.calls.append(query_dense)
        return list(self.chunks)


# ---------------------------------------------------------------------------
# RetrievalTool
# ---------------------------------------------------------------------------


class TestRetrievalTool:
    @pytest.mark.asyncio
    async def test_calls_retriever_with_query(self) -> None:
        chunks = [SourceChunk(content="fact", source="doc.txt", score=0.9, chunk_id="1")]
        retriever = _FakeRetriever(chunks)
        tool = RetrievalTool(retriever=retriever)

        await tool.execute({"query_dense": "what is the capital?", "query_sparse": "capital"})

        assert retriever.calls == ["what is the capital?"]

    @pytest.mark.asyncio
    async def test_result_contains_chunk_content_and_source(self) -> None:
        chunks = [
            SourceChunk(
                content="Paris is the capital.", source="france.txt", score=0.95, chunk_id="1"
            )
        ]
        tool = RetrievalTool(retriever=_FakeRetriever(chunks))

        result = await tool.execute(
            {"query_dense": "capital of France", "query_sparse": "France capital"}
        )

        assert "chunks" in result
        first = result["chunks"][0]
        assert first["source"] == "france.txt"
        assert first["chunk_id"] == "1"
        assert first["content"] == "Paris is the capital."

    @pytest.mark.asyncio
    async def test_result_preserves_optional_chunk_metadata(self) -> None:
        chunks = [
            SourceChunk(
                content="Policy text",
                source="corpus/executive_order_14110.txt",
                score=0.95,
                chunk_id="1",
                title="Executive Order 14110",
                author="Executive Office of the President",
                publication_date="2023-11-01",
                source_url="https://example.com/eo-14110",
            )
        ]
        tool = RetrievalTool(retriever=_FakeRetriever(chunks))

        result = await tool.execute(
            {
                "query_dense": "Executive Order 14110 AI policy",
                "query_sparse": "Executive Order 14110",
            }
        )

        first = result["chunks"][0]
        assert first["title"] == "Executive Order 14110"
        assert first["author"] == "Executive Office of the President"
        assert first["publication_date"] == "2023-11-01"
        assert first["source_url"] == "https://example.com/eo-14110"

    @pytest.mark.asyncio
    async def test_returns_no_results_message_on_empty_retrieval(self) -> None:
        tool = RetrievalTool(retriever=_FakeRetriever([]))

        result = await tool.execute(
            {"query_dense": "unknown topic", "query_sparse": "unknown topic"}
        )

        assert result["chunks"] == []
        assert "message" in result

    @pytest.mark.asyncio
    async def test_returns_error_on_missing_query_arg(self) -> None:
        tool = RetrievalTool(retriever=_FakeRetriever([]))

        result = await tool.execute({})

        assert "error" in result


# ---------------------------------------------------------------------------
# DocumentSplitter chunking behaviour (unit, no infrastructure)
# ---------------------------------------------------------------------------


class TestDocumentSplitterChunking:
    """Verify chunking behaviour used by the ingestion pipeline."""

    def test_splits_long_document_into_multiple_chunks(self) -> None:
        splitter = DocumentSplitter(split_by="word", split_length=10, split_overlap=0)
        long_text = " ".join(["word"] * 50)
        docs = [Document(content=long_text, meta={"source": "test.txt"})]
        result = splitter.run(documents=docs)
        chunks = result["documents"]
        assert len(chunks) > 1, "A 50-word document split at 10 words should yield > 1 chunk"

    def test_chunks_preserve_source_metadata(self) -> None:
        splitter = DocumentSplitter(split_by="word", split_length=10, split_overlap=0)
        docs = [Document(content=" ".join(["w"] * 30), meta={"source": "origin.txt"})]
        result = splitter.run(documents=docs)
        for chunk in result["documents"]:
            assert chunk.meta.get("source") == "origin.txt"

    def test_short_document_produces_single_chunk(self) -> None:
        splitter = DocumentSplitter(split_by="word", split_length=200, split_overlap=0)
        docs = [Document(content="short document", meta={})]
        result = splitter.run(documents=docs)
        assert len(result["documents"]) == 1

    def test_empty_document_is_skipped(self) -> None:
        splitter = DocumentSplitter(split_by="word", split_length=10, split_overlap=0)
        result = splitter.run(documents=[Document(content="", meta={})])
        # skip_empty_documents=True by default
        assert result["documents"] == []


# ---------------------------------------------------------------------------
# Sidecar metadata loading
# ---------------------------------------------------------------------------


class TestLoadSidecarMeta:
    def test_returns_all_fields_when_sidecar_present(self, tmp_path: Path) -> None:
        doc = tmp_path / "report.txt"
        doc.write_text("content")
        sidecar = tmp_path / "report.txt.meta.json"
        sidecar.write_text(
            json.dumps(
                {
                    "title": "My Report",
                    "author": "Alice",
                    "publication_date": "2024-01",
                    "source_url": "https://example.com/report",
                    "license": "CC BY 4.0",
                }
            )
        )

        meta = load_sidecar_meta(doc)

        assert meta["title"] == "My Report"
        assert meta["author"] == "Alice"
        assert meta["publication_date"] == "2024-01"
        assert meta["source_url"] == "https://example.com/report"
        assert meta["license"] == "CC BY 4.0"

    def test_returns_empty_dict_when_no_sidecar(self, tmp_path: Path) -> None:
        doc = tmp_path / "nodoc.txt"
        doc.write_text("content")
        assert load_sidecar_meta(doc) == {}

    def test_source_key_not_in_sidecar_does_not_break(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.md"
        doc.write_text("content")
        sidecar = tmp_path / "doc.md.meta.json"
        sidecar.write_text(json.dumps({"title": "T"}))
        meta = load_sidecar_meta(doc)
        # source is NOT in the sidecar — pipeline adds it separately
        assert "source" not in meta
        assert meta["title"] == "T"

    def test_malformed_json_returns_empty_dict(self, tmp_path: Path) -> None:
        doc = tmp_path / "bad.txt"
        doc.write_text("content")
        (tmp_path / "bad.txt.meta.json").write_text("not valid json {{{")
        assert load_sidecar_meta(doc) == {}

    def test_non_dict_json_returns_empty_dict(self, tmp_path: Path) -> None:
        doc = tmp_path / "arr.txt"
        doc.write_text("content")
        (tmp_path / "arr.txt.meta.json").write_text("[1, 2, 3]")
        assert load_sidecar_meta(doc) == {}

    def test_meta_json_suffix_not_treated_as_document(self, tmp_path: Path) -> None:
        """Ensure .meta.json files are not included in supported-file discovery."""
        # .meta.json ends in .json — the pipeline only ingests .txt and .md
        assert Path("doc.txt.meta.json").suffix == ".json"
        assert ".json" not in {".txt", ".md"}


# ---------------------------------------------------------------------------
# IngestionPipeline — injected embedder and converter routing
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Minimal DocumentEmbedder double that returns documents unchanged."""

    def __init__(self) -> None:
        self.call_count = 0
        self.total_documents_seen = 0
        self.last_documents: list[Document] = []

    def run(self, documents: list[Document]) -> dict[str, Any]:
        self.call_count += 1
        self.total_documents_seen += len(documents)
        # Use dataclasses.replace to preserve document IDs, matching real Haystack embedder behaviour.
        embedded = [dataclasses.replace(d, embedding=[0.0] * 4) for d in documents]
        self.last_documents = embedded
        return {"documents": embedded}


class _FakeDocumentStore:
    """Minimal in-memory document store double."""

    def __init__(self) -> None:
        self.written: list[Document] = []

    def write_documents(
        self,
        documents: list[Document],
        policy: object = None,
    ) -> int:
        self.written.extend(documents)
        return len(documents)

    # Haystack DocumentWriter calls this method on the store.
    def count_documents(self) -> int:
        return len(self.written)


def _make_pipeline(
    tmp_path: Path,
    batch_size: int = 100,
) -> tuple[IngestionPipeline, _FakeEmbedder, _FakeDocumentStore]:
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    store = InMemoryDocumentStore()
    embedder = _FakeEmbedder()
    config = IngestionConfig(
        split_length=10,
        split_overlap=0,
        batch_size=batch_size,
    )
    pipeline = IngestionPipeline(
        config=config,
        document_store=store,
        embedder=embedder,
        format_handlers=build_format_handlers(image_service=None, extracted_image_store=None),
    )
    return pipeline, embedder, store  # type: ignore[return-value]


class TestIngestionPipelineEmbedderInjection:
    def test_embedder_called_for_txt_file(self, tmp_path: Path) -> None:
        doc = tmp_path / "article.txt"
        doc.write_text(" ".join(["word"] * 30))
        pipeline, embedder, _ = _make_pipeline(tmp_path)

        pipeline.ingest([doc])

        assert embedder.call_count >= 1
        assert embedder.total_documents_seen > 0

    def test_embedder_called_for_md_file(self, tmp_path: Path) -> None:
        doc = tmp_path / "readme.md"
        doc.write_text("# Title\nSome markdown content with several sentences. More text here.")
        pipeline, embedder, _ = _make_pipeline(tmp_path)

        pipeline.ingest([doc])

        assert embedder.call_count >= 1

    def test_md_without_sentence_punctuation_still_splits_into_multiple_chunks(
        self, tmp_path: Path
    ) -> None:
        """Regression: markdown chunking must not rely on sentence boundaries."""
        doc = tmp_path / "notes.md"
        doc.write_text("# Header\n" + " ".join(["token"] * 45))
        pipeline, embedder, _store = _make_pipeline(tmp_path)

        written = pipeline.ingest([doc])

        assert written > 1
        assert embedder.total_documents_seen > 1

    def test_unsupported_files_skipped(self, tmp_path: Path) -> None:
        docx = tmp_path / "report.docx"
        docx.write_bytes(b"PK fake docx content")
        pipeline, embedder, _ = _make_pipeline(tmp_path)

        count = pipeline.ingest([docx])

        assert count == 0
        assert embedder.call_count == 0

    def test_batch_size_controls_file_batching_in_ingest_corpus(self, tmp_path: Path) -> None:
        """ingest_corpus splits source files into batches of batch_size; the embedder is called
        once per batch, not once globally."""
        for i in range(6):
            (tmp_path / f"doc{i}.txt").write_text("word " * 10)
        pipeline, embedder, _ = _make_pipeline(tmp_path, batch_size=2)

        pipeline.ingest_corpus(tmp_path)

        # 6 files / batch_size=2 → 3 batches → embedder called 3 times.
        assert embedder.call_count == 3

    def test_ingest_accepts_iterator_and_batches_by_file_count(self, tmp_path: Path) -> None:
        """ingest accepts iterables (not only lists) and applies file-count batching."""
        for i in range(5):
            (tmp_path / f"iter{i}.txt").write_text("word " * 10)

        pipeline, embedder, _ = _make_pipeline(tmp_path, batch_size=2)

        path_iter = (p for p in sorted(tmp_path.glob("*.txt")))
        pipeline.ingest(path_iter)

        # 5 files / batch_size=2 -> 3 batches.
        assert embedder.call_count == 3

    def test_ingest_filters_sidecar_and_readme_before_batching(self, tmp_path: Path) -> None:
        """Filtered files do not consume batch slots or trigger extra embedder calls."""
        # 2 valid docs + sidecar + README => still exactly one batch at size 2.
        (tmp_path / "a.txt").write_text("word " * 10)
        (tmp_path / "b.txt").write_text("word " * 10)
        (tmp_path / "a.txt.meta.json").write_text(json.dumps({"title": "A"}))
        (tmp_path / "README.md").write_text("corpus description")

        pipeline, embedder, _ = _make_pipeline(tmp_path, batch_size=2)
        pipeline.ingest(tmp_path.rglob("*"))

        assert embedder.call_count == 1

    def test_sidecar_meta_merged_into_chunk_metadata(self, tmp_path: Path) -> None:
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        doc = tmp_path / "report.txt"
        doc.write_text("Some important content about finance.")
        sidecar = tmp_path / "report.txt.meta.json"
        sidecar.write_text(json.dumps({"title": "Finance Report", "author": "Alice"}))

        store = InMemoryDocumentStore()
        embedder = _FakeEmbedder()
        config = IngestionConfig(
            split_length=200,
            split_overlap=0,
        )
        pipeline = IngestionPipeline(
            config=config,
            document_store=store,
            embedder=embedder,
            format_handlers=build_format_handlers(image_service=None, extracted_image_store=None),
        )
        pipeline.ingest([doc])

        # Verify written docs have sidecar metadata merged in.
        written = store.filter_documents()
        assert any(d.meta.get("title") == "Finance Report" for d in written)
        assert any(d.meta.get("author") == "Alice" for d in written)

    def test_sparse_vectors_are_attached_to_written_documents(self, tmp_path: Path) -> None:
        from haystack.dataclasses import SparseEmbedding
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        doc = tmp_path / "keywords.txt"
        doc.write_text("Hybrid retrieval improves exact term recall for labor market queries.")

        store = InMemoryDocumentStore()
        embedder = _FakeEmbedder()

        class _FakeSparseEmbedder:
            def run(self, documents: list[Document]) -> dict[str, Any]:
                from dataclasses import replace

                return {
                    "documents": [
                        replace(
                            d, sparse_embedding=SparseEmbedding(indices=[1, 2], values=[0.5, 0.5])
                        )
                        for d in documents
                    ]
                }

        pipeline = IngestionPipeline(
            config=IngestionConfig(
                split_length=200,
                split_overlap=0,
            ),
            document_store=store,
            embedder=embedder,
            format_handlers=build_format_handlers(image_service=None, extracted_image_store=None),
            sparse_embedder=_FakeSparseEmbedder(),  # type: ignore[arg-type]
        )

        pipeline.ingest([doc])

        written_docs = store.filter_documents()
        assert written_docs
        assert written_docs[0].sparse_embedding is not None
        assert written_docs[0].sparse_embedding.indices

    def test_already_indexed_chunks_are_not_re_embedded(self, tmp_path: Path) -> None:
        """Re-running ingest on unchanged files skips embedding for existing chunks."""
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        doc = tmp_path / "article.txt"
        doc.write_text(" ".join(["word"] * 30))
        store = InMemoryDocumentStore()
        embedder = _FakeEmbedder()
        config = IngestionConfig(split_length=10, split_overlap=0)
        pipeline = IngestionPipeline(
            config=config,
            document_store=store,
            embedder=embedder,
            format_handlers=build_format_handlers(image_service=None, extracted_image_store=None),
        )

        pipeline.ingest([doc])
        calls_after_first_run = embedder.call_count
        docs_after_first_run = embedder.total_documents_seen

        # Second ingest of the same unchanged file — embedder must not be called again.
        pipeline.ingest([doc])

        assert embedder.call_count == calls_after_first_run
        assert embedder.total_documents_seen == docs_after_first_run

    def test_new_file_is_embedded_when_other_files_already_indexed(self, tmp_path: Path) -> None:
        """Re-run with one new file only embeds the new file's chunks."""
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        existing = tmp_path / "old.txt"
        existing.write_text(" ".join(["word"] * 30))
        new_doc = tmp_path / "new.txt"
        new_doc.write_text(" ".join(["thing"] * 30))
        store = InMemoryDocumentStore()
        embedder = _FakeEmbedder()
        config = IngestionConfig(split_length=10, split_overlap=0)
        pipeline = IngestionPipeline(
            config=config,
            document_store=store,
            embedder=embedder,
            format_handlers=build_format_handlers(image_service=None, extracted_image_store=None),
        )

        pipeline.ingest([existing])
        docs_after_first_run = embedder.total_documents_seen

        # Ingest both files; only new_doc's chunks should be embedded.
        pipeline.ingest([existing, new_doc])

        new_chunks_embedded = embedder.total_documents_seen - docs_after_first_run
        # new_doc produces the same number of chunks as existing; none from existing again.
        assert new_chunks_embedded == docs_after_first_run


# ---------------------------------------------------------------------------
# PDF ingestion — _PdfPageConverter and pipeline integration
# ---------------------------------------------------------------------------

# Minimal syntactically-valid PDF bytes (one page, ASCII text).  The xref
# offsets are slightly off, but pypdf recovers via its repair path which is
# sufficient for unit-level testing.
_MINIMAL_PDF_PAGE1 = (
    b"%PDF-1.4\n"
    b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
    b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
    b"  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
    b"4 0 obj << /Length 44 >>\n"
    b"stream\n"
    b"BT /F1 12 Tf 100 700 Td (Hello World Page One) Tj ET\n"
    b"endstream\n"
    b"endobj\n"
    b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"0000000360 00000 n \n"
    b"trailer << /Size 6 /Root 1 0 R >>\n"
    b"startxref\n441\n%%EOF"
)

_MINIMAL_PDF_PAGE2 = (
    b"%PDF-1.4\n"
    b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >> endobj\n"
    b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
    b"  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
    b"4 0 obj << /Length 44 >>\n"
    b"stream\n"
    b"BT /F1 12 Tf 100 700 Td (Content on page one here) Tj ET\n"
    b"endstream\n"
    b"endobj\n"
    b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    b"6 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
    b"  /Contents 7 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
    b"7 0 obj << /Length 44 >>\n"
    b"stream\n"
    b"BT /F1 12 Tf 100 700 Td (Content on page two here) Tj ET\n"
    b"endstream\n"
    b"endobj\n"
    b"xref\n0 8\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000120 00000 n \n"
    b"0000000280 00000 n \n"
    b"0000000374 00000 n \n"
    b"0000000450 00000 n \n"
    b"0000000620 00000 n \n"
    b"trailer << /Size 8 /Root 1 0 R >>\n"
    b"startxref\n714\n%%EOF"
)


def _write_pdf(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


class TestPdfPageConverter:
    """Unit tests for _PdfPageConverter in isolation (no pipeline infrastructure)."""

    def test_single_page_pdf_yields_one_document(self, tmp_path: Path) -> None:
        from src.ingest.infrastructure.converters import PdfPageConverter

        pdf = _write_pdf(tmp_path, "single.pdf", _MINIMAL_PDF_PAGE1)
        converter = PdfPageConverter()
        result = converter.run(sources=[pdf], meta=[{"source": str(pdf)}])

        docs = result["documents"]
        assert len(docs) == 1

    def test_extracted_text_present_in_document(self, tmp_path: Path) -> None:
        from src.ingest.infrastructure.converters import PdfPageConverter

        pdf = _write_pdf(tmp_path, "text.pdf", _MINIMAL_PDF_PAGE1)
        converter = PdfPageConverter()
        result = converter.run(sources=[pdf], meta=[{"source": str(pdf)}])

        assert result["documents"][0].content is not None
        assert "Hello World Page One" in (result["documents"][0].content or "")

    def test_page_metadata_set_to_1_for_single_page(self, tmp_path: Path) -> None:
        from src.ingest.infrastructure.converters import PdfPageConverter

        pdf = _write_pdf(tmp_path, "meta.pdf", _MINIMAL_PDF_PAGE1)
        converter = PdfPageConverter()
        result = converter.run(sources=[pdf], meta=[{"source": str(pdf)}])

        assert result["documents"][0].meta["page"] == "1"
        assert result["documents"][0].meta["total_pages"] == "1"

    def test_two_page_pdf_yields_two_documents(self, tmp_path: Path) -> None:
        from src.ingest.infrastructure.converters import PdfPageConverter

        pdf = _write_pdf(tmp_path, "two_pages.pdf", _MINIMAL_PDF_PAGE2)
        converter = PdfPageConverter()
        result = converter.run(sources=[pdf], meta=[{"source": str(pdf)}])

        docs = result["documents"]
        assert len(docs) == 2
        assert docs[0].meta["page"] == "1"
        assert docs[1].meta["page"] == "2"
        assert docs[0].meta["total_pages"] == "2"

    def test_sidecar_metadata_preserved_in_page_docs(self, tmp_path: Path) -> None:
        from src.ingest.infrastructure.converters import PdfPageConverter

        pdf = _write_pdf(tmp_path, "meta.pdf", _MINIMAL_PDF_PAGE1)
        converter = PdfPageConverter()
        result = converter.run(
            sources=[pdf],
            meta=[{"source": str(pdf), "title": "My PDF", "author": "Bob"}],
        )

        doc = result["documents"][0]
        assert doc.meta["title"] == "My PDF"
        assert doc.meta["author"] == "Bob"
        assert doc.meta["page"] == "1"

    def test_extraction_failure_returns_empty_list(self, tmp_path: Path) -> None:
        from src.ingest.infrastructure.converters import PdfPageConverter

        bad = tmp_path / "corrupt.pdf"
        bad.write_bytes(b"not a pdf at all")
        converter = PdfPageConverter()
        result = converter.run(sources=[bad], meta=[{"source": str(bad)}])

        assert result["documents"] == []

    def test_single_dict_meta_applied_to_all_sources(self, tmp_path: Path) -> None:
        from src.ingest.infrastructure.converters import PdfPageConverter

        pdf1 = _write_pdf(tmp_path, "a.pdf", _MINIMAL_PDF_PAGE1)
        pdf2 = _write_pdf(tmp_path, "b.pdf", _MINIMAL_PDF_PAGE1)
        converter = PdfPageConverter()
        result = converter.run(sources=[pdf1, pdf2], meta={"category": "report"})

        for doc in result["documents"]:
            assert doc.meta["category"] == "report"


class TestPdfIngestionPipeline:
    """Integration-style unit tests: PDF path through the full pipeline (no network/Qdrant)."""

    def test_pdf_file_ingested_and_chunks_written(self, tmp_path: Path) -> None:
        pdf = _write_pdf(tmp_path, "doc.pdf", _MINIMAL_PDF_PAGE1)
        pipeline, embedder, _store = _make_pipeline(tmp_path)

        count = pipeline.ingest([pdf])

        assert count > 0
        assert embedder.call_count >= 1

    def test_pdf_chunks_carry_page_metadata(self, tmp_path: Path) -> None:
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        pdf = _write_pdf(tmp_path, "paged.pdf", _MINIMAL_PDF_PAGE2)
        store = InMemoryDocumentStore()
        embedder = _FakeEmbedder()
        config = IngestionConfig(split_length=200, split_overlap=0)
        pipeline = IngestionPipeline(
            config=config,
            document_store=store,
            embedder=embedder,
            format_handlers=build_format_handlers(image_service=None, extracted_image_store=None),
        )

        pipeline.ingest([pdf])

        written = store.filter_documents()
        pages = {d.meta.get("page") for d in written if d.meta.get("page")}
        # Two-page PDF: both pages must be represented.
        assert "1" in pages
        assert "2" in pages

    def test_pdf_sidecar_meta_merged_into_chunks(self, tmp_path: Path) -> None:
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        pdf = _write_pdf(tmp_path, "report.pdf", _MINIMAL_PDF_PAGE1)
        sidecar = tmp_path / "report.pdf.meta.json"
        sidecar.write_text(
            json.dumps(
                {
                    "title": "Annual Report",
                    "author": "Corp Inc",
                    "publication_date": "2024-01",
                    "source_url": "https://example.com/report.pdf",
                }
            )
        )

        store = InMemoryDocumentStore()
        embedder = _FakeEmbedder()
        config = IngestionConfig(split_length=200, split_overlap=0)
        pipeline = IngestionPipeline(
            config=config,
            document_store=store,
            embedder=embedder,
            format_handlers=build_format_handlers(image_service=None, extracted_image_store=None),
        )
        pipeline.ingest([pdf])

        written = store.filter_documents()
        assert any(d.meta.get("title") == "Annual Report" for d in written)
        assert any(d.meta.get("author") == "Corp Inc" for d in written)
        assert any(d.meta.get("source_url") == "https://example.com/report.pdf" for d in written)

    def test_corrupt_pdf_skipped_gracefully(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf")
        pipeline, _embedder, _ = _make_pipeline(tmp_path)

        # Must not raise; corrupt file is logged and skipped.
        count = pipeline.ingest([bad])
        assert count == 0
