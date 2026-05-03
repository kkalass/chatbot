# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the citeable :class:`RetrievalTool`."""

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from src.chatbot.app.citation import (
    DocumentCitation,
    RawCitation,
)
from src.chatbot.app.protocols import JsonObject, SourceChunk
from src.chatbot.tools.retrieval.tool import RetrievalTool


@dataclass(frozen=True)
class _StaticContext:
    results: tuple[JsonObject, ...]

    def tool_result_for(self, tool_call_id: str) -> JsonObject | None:
        return None  # not used by retrieval validation

    def tool_results_for(self, tool_name: str) -> Sequence[JsonObject]:
        return self.results


class _StubRetriever:
    def __init__(self, chunks: list[SourceChunk]) -> None:
        self._chunks = chunks
        self.calls: list[str] = []

    async def retrieve(self, query: str) -> list[SourceChunk]:
        self.calls.append(query)
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

        result = await tool.execute({"query": "what is RAG?"})

        assert "chunks" in result
        assert retriever.calls == ["what is RAG?"]

    @pytest.mark.asyncio
    async def test_empty_results_include_message(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))

        result = await tool.execute({"query": "no hits"})

        assert result == {"chunks": [], "message": "No relevant documents found."}

    @pytest.mark.asyncio
    async def test_invalid_arguments_return_error(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))

        result = await tool.execute({})

        assert "error" in result


class TestCiteInstructions:
    def test_fragment_documents_required_fields(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))
        fragment = tool.cite_instructions().prompt_fragment

        assert "search_documents" in fragment
        assert "tool_call_id" in fragment
        assert "chunk_id" in fragment


class TestFormatForHistory:
    def test_renders_xml_with_chunk_metadata(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))
        result = _result_with(_chunk(source="s.md", chunk_id="c1", content="hello"))

        rendered = tool.format_for_history(result)

        assert "<search_results>" in rendered
        assert 'source_path="s.md"' in rendered
        assert 'chunk_id="c1"' in rendered
        assert "hello" in rendered

    def test_renders_no_chunks_message(self) -> None:
        tool = RetrievalTool(_StubRetriever([]))
        rendered = tool.format_for_history(
            {"chunks": [], "message": "No relevant documents found."}
        )
        assert rendered == "No relevant documents found."


class TestValidateAndEnrich:
    def _tool(self) -> RetrievalTool:
        return RetrievalTool(_StubRetriever([]))

    def test_returns_none_when_chunk_id_missing(self) -> None:
        tool = self._tool()
        raw = RawCitation(tool_call_id="tc1", raw_marker_text="m")
        ctx = _StaticContext(results=())

        assert tool.validate_and_enrich(raw, ctx) is None

    def test_exact_match_returns_document_citation(self) -> None:
        tool = self._tool()
        ctx = _StaticContext(
            results=(
                _result_with(_chunk(source="docs/a.md", chunk_id="c1", content="X", score=0.7)),
            )
        )
        raw = RawCitation(
            tool_call_id="tc1",
            chunk_id="c1",
            raw_marker_text="m",
        )

        result = tool.validate_and_enrich(raw, ctx)

        assert isinstance(result, DocumentCitation)
        assert result.source == "docs/a.md"
        assert result.chunk_id == "c1"
        assert result.content == "X"
        assert result.score == 0.7
        assert result.raw_marker_text == "m"

    def test_chunk_id_resolves_regardless_of_source(self) -> None:
        """chunk_id is a content hash — authoritative source is resolved from stored results."""
        tool = self._tool()
        ctx = _StaticContext(
            results=(_result_with(_chunk(source="docs/a.md", chunk_id="unique-hash")),),
        )
        raw = RawCitation(
            tool_call_id="tc1",
            chunk_id="unique-hash",
            raw_marker_text="m",
        )

        result = tool.validate_and_enrich(raw, ctx)
        assert isinstance(result, DocumentCitation)
        assert result.source == "docs/a.md"

    def test_unknown_chunk_id_returns_none(self) -> None:
        tool = self._tool()
        ctx = _StaticContext(
            results=(_result_with(_chunk(source="docs/a.md", chunk_id="known")),),
        )
        raw = RawCitation(tool_call_id="tc1", chunk_id="unknown", raw_marker_text="m")
        assert tool.validate_and_enrich(raw, ctx) is None

    def test_no_search_results_returns_none(self) -> None:
        tool = self._tool()
        ctx = _StaticContext(results=())
        raw = RawCitation(tool_call_id="tc1", chunk_id="y", raw_marker_text="m")
        assert tool.validate_and_enrich(raw, ctx) is None

    def test_higher_scoring_duplicate_wins(self) -> None:
        tool = self._tool()
        ctx = _StaticContext(
            results=(
                _result_with(_chunk(source="s", chunk_id="c", content="low", score=0.1)),
                _result_with(_chunk(source="s", chunk_id="c", content="high", score=0.9)),
            )
        )
        raw = RawCitation(tool_call_id="tc1", chunk_id="c", raw_marker_text="m")
        result = tool.validate_and_enrich(raw, ctx)
        assert isinstance(result, DocumentCitation)
        assert result.content == "high"
        assert result.score == 0.9
