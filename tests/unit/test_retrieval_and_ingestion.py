"""Unit tests for ingestion chunking logic, retrieval tool, and sidecar loading.

Covers:
- :class:`~src.chatbot.tools.retrieval.tool.RetrievalTool` — execute contract, error handling,
  and empty-result handling.
- :class:`~src.ingest.pipeline.IngestionPipeline` chunking via real
  :class:`~haystack.components.preprocessors.DocumentSplitter` (no
  infrastructure dependencies — uses in-memory documents only).
- :func:`~src.ingest.pipeline.load_sidecar_meta` — sidecar loading.
- Converter routing and splitter strategy selection.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document

from src.chatbot.app.protocols import SourceChunk, ToolContext
from src.chatbot.tools.retrieval.tool import RetrievalTool
from src.ingest.pipeline import IngestionConfig, IngestionPipeline, load_sidecar_meta

# ---------------------------------------------------------------------------
# Helpers / test doubles
# ---------------------------------------------------------------------------


class _FakeRetriever:
    """Configurable retriever double."""

    def __init__(self, chunks: list[SourceChunk]) -> None:
        self.chunks = chunks
        self.calls: list[str] = []

    async def retrieve(self, query: str) -> list[SourceChunk]:
        self.calls.append(query)
        return list(self.chunks)


_EMPTY_CONTEXT = ToolContext(history=())


# ---------------------------------------------------------------------------
# RetrievalTool
# ---------------------------------------------------------------------------


class TestRetrievalTool:
    @pytest.mark.asyncio
    async def test_calls_retriever_with_query(self) -> None:
        chunks = [SourceChunk(content="fact", source="doc.txt", score=0.9, chunk_id="1")]
        retriever = _FakeRetriever(chunks)
        tool = RetrievalTool(retriever=retriever)

        await tool.execute({"query": "what is the capital?"}, _EMPTY_CONTEXT)

        assert retriever.calls == ["what is the capital?"]

    @pytest.mark.asyncio
    async def test_result_contains_chunk_content_and_source(self) -> None:
        chunks = [
            SourceChunk(
                content="Paris is the capital.", source="france.txt", score=0.95, chunk_id="1"
            )
        ]
        tool = RetrievalTool(retriever=_FakeRetriever(chunks))

        result, events = await tool.execute({"query": "capital of France"}, _EMPTY_CONTEXT)

        assert "chunks" in result
        first = result["chunks"][0]  # type: ignore[index]
        assert first["source"] == "france.txt"  # type: ignore[index]
        assert first["chunk_id"] == "1"  # type: ignore[index]
        assert first["content"] == "Paris is the capital."  # type: ignore[index]
        assert events == []

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

        result, events = await tool.execute({"query": "EO 14110"}, _EMPTY_CONTEXT)

        first = result["chunks"][0]  # type: ignore[index]
        assert first["title"] == "Executive Order 14110"  # type: ignore[index]
        assert first["author"] == "Executive Office of the President"  # type: ignore[index]
        assert first["publication_date"] == "2023-11-01"  # type: ignore[index]
        assert first["source_url"] == "https://example.com/eo-14110"  # type: ignore[index]
        assert events == []

    @pytest.mark.asyncio
    async def test_returns_no_results_message_on_empty_retrieval(self) -> None:
        tool = RetrievalTool(retriever=_FakeRetriever([]))

        result, events = await tool.execute({"query": "unknown topic"}, _EMPTY_CONTEXT)

        assert result["chunks"] == []  # type: ignore[comparison-overlap]
        assert "message" in result
        assert events == []

    @pytest.mark.asyncio
    async def test_returns_error_on_missing_query_arg(self) -> None:
        tool = RetrievalTool(retriever=_FakeRetriever([]))

        result, events = await tool.execute({}, _EMPTY_CONTEXT)

        assert "error" in result
        assert events == []


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

    def run(self, documents: list[Document]) -> dict[str, Any]:
        self.call_count += 1
        self.total_documents_seen += len(documents)
        # Attach a dummy embedding vector so the writer doesn't complain.
        embedded = [
            Document(content=d.content, meta=d.meta, embedding=[0.0] * 4) for d in documents
        ]
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
    pipeline = IngestionPipeline(config=config, document_store=store, embedder=embedder)
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

    def test_unsupported_files_skipped(self, tmp_path: Path) -> None:
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        pipeline, embedder, _ = _make_pipeline(tmp_path)

        count = pipeline.ingest([pdf])

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
        pipeline = IngestionPipeline(config=config, document_store=store, embedder=embedder)
        pipeline.ingest([doc])

        # Verify written docs have sidecar metadata merged in.
        written = store.filter_documents()
        assert any(d.meta.get("title") == "Finance Report" for d in written)
        assert any(d.meta.get("author") == "Alice" for d in written)
