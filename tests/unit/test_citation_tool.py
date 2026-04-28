"""Unit tests for CitationTool — citation validation and SourceCitationEvent emission."""

import pytest

from src.chatbot.app.protocols import (
    ChatMessage,
    SourceChunk,
    SourceCitationEvent,
    ToolCallInfo,
    ToolContext,
)
from src.chatbot.tools.citation.tool import CitationTool


def _make_context(
    chunks: list[dict[str, object]], call_id: str = "search_documents"
) -> ToolContext:
    """Build a ToolContext with a search_documents tool call + result in history."""
    tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id=call_id)
    assistant_msg = ChatMessage(role="assistant", content="", tool_calls=(tc,))
    tool_result = ChatMessage(
        role="tool",
        content={"chunks": chunks},
        tool_call_id=call_id,
    )
    return ToolContext(history=(assistant_msg, tool_result))


def _chunk_dict(
    source: str,
    *,
    chunk_id: str = "1",
    content: str = "text",
    score: float = 0.9,
) -> dict[str, object]:
    return {"source": source, "chunk_id": chunk_id, "content": content, "score": score}


def _citation(source: str, chunk_id: str) -> dict[str, str]:
    return {"source": source, "chunk_id": chunk_id}


class TestCitationToolExecute:
    @pytest.mark.asyncio
    async def test_validates_claimed_citation_present_in_history(self) -> None:
        ctx = _make_context([_chunk_dict("a.txt")])
        tool = CitationTool()

        result, events = await tool.execute(
            {"citations": [_citation("a.txt", "1")]},
            ctx,
        )

        assert result["validated"] == [_citation("a.txt", "1")]
        assert result["unvalidated"] == []
        assert len(events) == 1
        assert isinstance(events[0], SourceCitationEvent)
        assert events[0].validated[0].source == "a.txt"
        assert events[0].validated[0].chunk_id == "1"

    @pytest.mark.asyncio
    async def test_unvalidated_citations_not_in_history(self) -> None:
        ctx = _make_context([_chunk_dict("a.txt")])
        tool = CitationTool()

        result, events = await tool.execute(
            {
                "citations": [
                    _citation("a.txt", "1"),
                    _citation("missing.txt", "9"),
                ]
            },
            ctx,
        )

        assert result["validated"] == [_citation("a.txt", "1")]
        assert result["unvalidated"] == [_citation("missing.txt", "9")]
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_no_event_emitted_when_nothing_validated(self) -> None:
        ctx = _make_context([_chunk_dict("a.txt")])
        tool = CitationTool()

        result, events = await tool.execute(
            {"citations": [_citation("ghost.txt", "99")]},
            ctx,
        )

        assert result["validated"] == []
        assert result["unvalidated"] == [_citation("ghost.txt", "99")]
        assert events == []

    @pytest.mark.asyncio
    async def test_empty_citations_list_returns_no_events(self) -> None:
        ctx = _make_context([_chunk_dict("a.txt")])
        tool = CitationTool()

        result, events = await tool.execute({"citations": []}, ctx)

        assert result["validated"] == []
        assert result["unvalidated"] == []
        assert events == []

    @pytest.mark.asyncio
    async def test_empty_history_all_unvalidated(self) -> None:
        ctx = ToolContext(history=())
        tool = CitationTool()

        result, events = await tool.execute(
            {"citations": [_citation("a.txt", "1")]},
            ctx,
        )

        assert result["validated"] == []
        assert result["unvalidated"] == [_citation("a.txt", "1")]
        assert events == []

    @pytest.mark.asyncio
    async def test_multiple_validated_chunks_in_single_event(self) -> None:
        ctx = _make_context(
            [_chunk_dict("a.txt", chunk_id="1"), _chunk_dict("b.txt", chunk_id="2")]
        )
        tool = CitationTool()

        result, events = await tool.execute(
            {
                "citations": [
                    _citation("a.txt", "1"),
                    _citation("b.txt", "2"),
                ]
            },
            ctx,
        )

        assert len(result["validated"]) == 2  # type: ignore[arg-type]
        assert len(events) == 1
        validated_sources = {c.source for c in events[0].validated}  # type: ignore[union-attr]
        assert validated_sources == {"a.txt", "b.txt"}

    @pytest.mark.asyncio
    async def test_invalid_args_returns_error(self) -> None:
        ctx = ToolContext(history=())
        tool = CitationTool()

        result, events = await tool.execute({"wrong_field": 42}, ctx)

        assert "error" in result
        assert events == []

    @pytest.mark.asyncio
    async def test_accepts_serialized_json_list_for_citations(self) -> None:
        chunks = [
            _chunk_dict(
                "corpus/executive_order_14110.txt",
                chunk_id="6ac85fb662c1fe7507171392be9e89350244bae72bc5f554c4fa44165288bad1",
            ),
            _chunk_dict(
                "corpus/executive_order_14110.txt",
                chunk_id="0db76f76dc7db112978587556183ed103d9c80fe59d94607d8a8f01570b33a8b",
            ),
            _chunk_dict(
                "corpus/executive_order_14110.txt",
                chunk_id="cca903fccdfab94d901e890cfc0b22a4493f5d67b00e2956034250aa4ce718ef",
            ),
            _chunk_dict(
                "corpus/executive_order_14110.txt",
                chunk_id="06a2700c410ff046f9101d91386e0dc61d0a643044fcfce91761b0ca2ff66c2b",
            ),
        ]
        ctx = _make_context(chunks)
        tool = CitationTool()

        serialized_citations = (
            "["
            '{"source":"corpus/executive_order_14110.txt","chunk_id":"6ac85fb662c1fe7507171392be9e89350244bae72bc5f554c4fa44165288bad1"},'
            '{"source":"corpus/executive_order_14110.txt","chunk_id":"0db76f76dc7db112978587556183ed103d9c80fe59d94607d8a8f01570b33a8b"},'
            '{"source":"corpus/executive_order_14110.txt","chunk_id":"cca903fccdfab94d901e890cfc0b22a4493f5d67b00e2956034250aa4ce718ef"},'
            '{"source":"corpus/executive_order_14110.txt","chunk_id":"06a2700c410ff046f9101d91386e0dc61d0a643044fcfce91761b0ca2ff66c2b"}'
            "]"
        )

        result, events = await tool.execute({"citations": serialized_citations}, ctx)

        assert len(result["validated"]) == 4  # type: ignore[arg-type]
        assert result["unvalidated"] == []
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_tool_call_id_must_match_search_results(self) -> None:
        """Tool results with a different call_id are not treated as search_documents results."""
        tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="id-1")
        assistant_msg = ChatMessage(role="assistant", content="", tool_calls=(tc,))
        # Tool result has mismatched call_id.
        tool_result = ChatMessage(
            role="tool",
            content={"chunks": [_chunk_dict("a.txt")]},
            tool_call_id="different-id",
        )
        ctx = ToolContext(history=(assistant_msg, tool_result))
        tool = CitationTool()

        result, _ = await tool.execute(
            {"citations": [_citation("a.txt", "1")]},
            ctx,
        )

        assert result["validated"] == []
        assert result["unvalidated"] == [_citation("a.txt", "1")]

    @pytest.mark.asyncio
    async def test_chunk_metadata_preserved_in_source_citation_event(self) -> None:
        chunk = {
            **_chunk_dict("report.txt", content="Finance data", score=0.85),
            "title": "Finance Report",
            "author": "Alice",
            "publication_date": "2024-01-01",
            "source_url": "https://example.com/report",
        }
        ctx = _make_context([chunk])
        tool = CitationTool()

        _, events = await tool.execute(
            {"citations": [_citation("report.txt", "1")]},
            ctx,
        )

        c: SourceChunk = events[0].validated[0]  # type: ignore[union-attr]
        assert c.content == "Finance data"
        assert c.score == pytest.approx(0.85)  # type: ignore[arg-type]
        assert c.title == "Finance Report"
        assert c.author == "Alice"
        assert c.publication_date == "2024-01-01"
        assert c.source_url == "https://example.com/report"

    @pytest.mark.asyncio
    async def test_duplicate_chunk_key_keeps_highest_score(self) -> None:
        """When duplicate (source, chunk_id) appears, the highest-scored chunk is kept."""
        chunks = [
            _chunk_dict("doc.pdf", chunk_id="c1", content="lower", score=0.5),
            _chunk_dict("doc.pdf", chunk_id="c1", content="higher", score=0.9),
            _chunk_dict("doc.pdf", chunk_id="c1", content="middle", score=0.7),
        ]
        ctx = _make_context(chunks)
        tool = CitationTool()

        _, events = await tool.execute(
            {"citations": [_citation("doc.pdf", "c1")]},
            ctx,
        )

        assert len(events) == 1
        c: SourceChunk = events[0].validated[0]  # type: ignore[union-attr]
        assert c.content == "higher"
        assert c.score == pytest.approx(0.9)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_duplicate_chunk_key_across_multiple_search_calls_keeps_highest_score(
        self,
    ) -> None:
        """Same (source, chunk_id) across search calls: best chunk wins."""
        tc1 = ToolCallInfo(name="search_documents", arguments={"query": "q1"}, call_id="c1")
        tc2 = ToolCallInfo(name="search_documents", arguments={"query": "q2"}, call_id="c2")
        history = (
            ChatMessage(role="assistant", content="", tool_calls=(tc1,)),
            ChatMessage(
                role="tool",
                content={
                    "chunks": [_chunk_dict("shared.pdf", chunk_id="c9", content="low", score=0.4)]
                },
                tool_call_id="c1",
            ),
            ChatMessage(role="assistant", content="", tool_calls=(tc2,)),
            ChatMessage(
                role="tool",
                content={
                    "chunks": [_chunk_dict("shared.pdf", chunk_id="c9", content="high", score=0.95)]
                },
                tool_call_id="c2",
            ),
        )
        ctx = ToolContext(history=history)
        tool = CitationTool()

        _, events = await tool.execute(
            {"citations": [_citation("shared.pdf", "c9")]},
            ctx,
        )

        assert len(events) == 1
        c: SourceChunk = events[0].validated[0]  # type: ignore[union-attr]
        assert c.content == "high"
        assert c.score == pytest.approx(0.95)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_multiple_citations_for_same_source_with_distinct_chunk_ids(self) -> None:
        ctx = _make_context(
            [
                _chunk_dict("doc.pdf", chunk_id="c1", content="first"),
                _chunk_dict("doc.pdf", chunk_id="c2", content="second"),
            ]
        )
        tool = CitationTool()

        result, events = await tool.execute(
            {
                "citations": [
                    _citation("doc.pdf", "c1"),
                    _citation("doc.pdf", "c2"),
                ]
            },
            ctx,
        )

        assert len(result["validated"]) == 2  # type: ignore[arg-type]
        assert result["unvalidated"] == []
        assert len(events) == 1
        ids = {c.chunk_id for c in events[0].validated}  # type: ignore[union-attr]
        assert ids == {"c1", "c2"}
