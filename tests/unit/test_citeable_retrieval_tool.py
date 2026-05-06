# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the citeable :class:`RetrievalTool`."""

import pytest

from src.chatbot.app.protocols import DocumentCitation, JsonObject, RawCitation, SourceChunk
from src.chatbot.app.protocols_citeable_tool import CitableUnit
from src.chatbot.tools.retrieval.tool import RetrievalTool


class _StubRetriever:
    def __init__(self, chunks: list[SourceChunk]) -> None:
        self._chunks = chunks
        self.calls: list[str] = []
        self.sparse_calls: list[str | None] = []

    async def retrieve(
        self,
        query_dense: str,
        *,
        query_sparse: str | None = None,
    ) -> list[SourceChunk]:
        self.calls.append(query_dense)
        self.sparse_calls.append(query_sparse)
        return self._chunks


def _chunk(
    *,
    source: str,
    chunk_id: str,
    content: str = "txt",
    score: float = 0.9,
    title: str | None = None,
) -> SourceChunk:
    return SourceChunk(
        content=content,
        source=source,
        score=score,
        chunk_id=chunk_id,
        title=title,
        author=None,
        publication_date=None,
        source_url=None,
        page=None,
    )


def _result_with(*chunks: SourceChunk) -> JsonObject:
    return {
        "chunks": [
            {
                "source": c.source,
                "chunk_id": c.chunk_id,
                "content": c.content,
                "score": c.score,
                "title": c.title,
                "author": c.author,
                "publication_date": c.publication_date,
                "source_url": c.source_url,
                "page": c.page,
            }
            for c in chunks
        ]
    }


class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_chunks_for_valid_query(self) -> None:
        retriever = _StubRetriever([_chunk(source="s.md", chunk_id="c1")])
        tool = RetrievalTool(retriever)

        result = await tool.execute(
            {"query_dense": "what is RAG?", "query_sparse": "RAG retrieval augmented generation"}
        )

        assert "chunks" in result
        assert retriever.calls == ["what is RAG?"]

    @pytest.mark.asyncio
    async def test_empty_results_include_message(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))

        result = await tool.execute({"query_dense": "no hits", "query_sparse": "no hits"})

        assert result == {"chunks": [], "message": "No relevant documents found."}

    @pytest.mark.asyncio
    async def test_routes_dense_and_sparse_independently(self) -> None:
        retriever = _StubRetriever([_chunk(source="s.md", chunk_id="c1")])
        tool = RetrievalTool(retriever)

        await tool.execute(
            {
                "query_dense": "impact of AI on employment",
                "query_sparse": "BIBB KI Arbeitsmarkt",
            }
        )

        assert retriever.calls == ["impact of AI on employment"]
        assert retriever.sparse_calls == ["BIBB KI Arbeitsmarkt"]

    @pytest.mark.asyncio
    async def test_invalid_arguments_return_error(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))

        result = await tool.execute({})

        assert "error" in result


class TestDescribeCall:
    def test_uses_query_dense_for_display(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))
        msg = tool.describe_call(
            {"query_dense": "impact of AI on labour market", "query_sparse": "KI Arbeitsmarkt"}
        )
        assert msg.args["query"] == "impact of AI on labour market"


class TestCiteInstructions:
    def test_fragment_documents_required_fields(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))
        fragment = tool.cite_instructions().prompt_fragment

        assert "search_documents" in fragment
        assert "citation_token" in fragment
        assert '"ref"' in fragment


class TestRenderForHistory:
    def test_renders_xml_with_chunk_metadata_and_units(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))
        result = _result_with(_chunk(source="s.md", chunk_id="c1", content="hello"))

        rendering = tool.render_for_history(result)

        assert "<search_results>" in rendering.llm_content
        assert 'source_path="s.md"' in rendering.llm_content
        assert 'citation_token="c1"' in rendering.llm_content
        assert "hello" in rendering.llm_content
        assert len(rendering.units) == 1
        unit = rendering.units[0]
        assert unit.citation_token == "c1"
        assert isinstance(unit.payload, SourceChunk)
        assert unit.payload.chunk_id == "c1"

    def test_renders_no_chunks_message(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))
        rendering = tool.render_for_history(
            {"chunks": [], "message": "No relevant documents found."}
        )
        assert rendering.llm_content == "No relevant documents found."
        assert rendering.units == ()


class TestEnrich:
    def test_builds_document_citation_from_unit(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))
        chunk = _chunk(source="docs/a.md", chunk_id="c1", content="X", score=0.7)
        unit = CitableUnit(citation_token="c1", payload=chunk)
        raw = RawCitation(ref="c1", raw_marker_text="m")

        citation = tool.enrich(raw, unit)

        assert isinstance(citation, DocumentCitation)
        assert citation.source == "docs/a.md"
        assert citation.chunk_id == "c1"
        assert citation.content == "X"
        assert citation.score == 0.7
        assert citation.raw_marker_text == "m"
        assert citation.citation_token == "c1"

    def test_payload_must_be_source_chunk(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))
        unit = CitableUnit(citation_token="c1", payload={"not": "a chunk"})
        raw = RawCitation(ref="c1", raw_marker_text="m")

        with pytest.raises(AssertionError):
            tool.enrich(raw, unit)
