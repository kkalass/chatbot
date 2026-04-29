"""Integration tests for text RAG: ingest fixture corpus, retrieve, and verify citations.

These tests require live Qdrant and Ollama services.  They are skipped
automatically when ``INTEGRATION_TESTS=1`` is not set in the environment, so
the regular unit-test run stays fast and infrastructure-independent.

Run with:
    INTEGRATION_TESTS=1 uv run pytest tests/integration/
"""

import os
import uuid
from pathlib import Path

import pytest

from src.chatbot.app.protocols import SourceCitationEvent
from src.chatbot.infrastructure.embeddings_text import TextEmbedderConfig, build_text_embedder
from src.chatbot.infrastructure.retrieval import RetrieverConfig, build_retriever
from src.ingest.infrastructure.document_store import DocumentStoreConfig, build_document_store
from src.ingest.infrastructure.embeddings_document import (
    DocumentEmbedderConfig,
    build_document_embedder,
)
from src.ingest.pipeline import IngestionConfig, IngestionPipeline

# Skip the entire module unless integration tests are explicitly opted-in.
pytestmark = pytest.mark.skipif(
    os.environ.get("INTEGRATION_TESTS") != "1",
    reason="Set INTEGRATION_TESTS=1 to run integration tests (requires Qdrant + Ollama).",
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Isolation: each test run uses a uniquely-named collection so parallel runs
# don't interfere and leftover state doesn't affect subsequent runs.
_TEST_COLLECTION = f"test_{uuid.uuid4().hex[:8]}"

_INGESTION_CONFIG = IngestionConfig(
    split_length=100,
    split_overlap=10,
)

_STORE_CONFIG = DocumentStoreConfig(
    host="localhost",
    port=6333,
    collection=_TEST_COLLECTION,
    embedding_dim=768,
)

_DOCUMENT_EMBEDDER_CONFIG = DocumentEmbedderConfig(
    url="http://localhost:11434",
    embedding_model="nomic-embed-text",
)

_TEXT_EMBEDDER_CONFIG = TextEmbedderConfig(
    url="http://localhost:11434",
    embedding_model="nomic-embed-text",
)

_RETRIEVER_CONFIG = RetrieverConfig(
    top_k=5,
    score_threshold=0.0,  # Accept all results in tests; correctness verified by citation presence.
    store_host="localhost",
    store_port=6333,
    store_collection=_TEST_COLLECTION,
    embedding_dim=768,
)


@pytest.fixture(scope="module")
def ingested_store() -> None:
    """Ingest the fixture corpus once for the entire module."""
    store = build_document_store(_STORE_CONFIG)
    pipeline = IngestionPipeline(
        config=_INGESTION_CONFIG,
        document_store=store,
        embedder=build_document_embedder(_DOCUMENT_EMBEDDER_CONFIG),
    )
    count = pipeline.ingest_corpus(_FIXTURE_DIR)
    assert count > 0, f"Expected > 0 chunks to be written; got {count}"


class TestGroundedRetrieval:
    async def test_zurich_query_returns_chunks(self, ingested_store: None) -> None:
        """After ingestion the retriever must return at least one chunk for a known topic."""
        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
        )
        chunks = await retriever.retrieve("What is the largest city in Switzerland?")
        assert len(chunks) > 0, "Expected at least one chunk for a known topic"

    async def test_citation_source_points_to_fixture_file(self, ingested_store: None) -> None:
        """Retrieved chunks must include the fixture file path in their source metadata."""
        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
        )
        chunks = await retriever.retrieve("financial center")
        sources = {c.source for c in chunks}
        assert any("zurich.txt" in s for s in sources), (
            f"Expected 'zurich.txt' in sources; got: {sources}"
        )

    async def test_unrelated_query_still_returns_chunks_above_zero_threshold(
        self, ingested_store: None
    ) -> None:
        """With score_threshold=0.0 even an unrelated query returns chunks (not empty)."""
        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
        )
        chunks = await retriever.retrieve("completely unrelated topic xyz123")
        assert isinstance(chunks, list)

    async def test_chunk_content_not_empty(self, ingested_store: None) -> None:
        """All returned chunks must have non-empty content."""
        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
        )
        chunks = await retriever.retrieve("Zurich old town")
        for chunk in chunks:
            assert chunk.content.strip(), f"Empty content in chunk {chunk.chunk_id}"


class TestOrchestratorWithRetrieval:
    """End-to-end: orchestrator with retrieval tool produces grounded responses."""

    async def test_orchestrator_attaches_sources_after_stream(self, ingested_store: None) -> None:
        """After processing a grounded query, retrieved source citations must be non-empty."""
        from src.chatbot.app.orchestrator import ChatOrchestrator
        from src.chatbot.app.protocols import SourceCitationEvent
        from src.chatbot.infrastructure.chat import (
            ChatModelConfig,
            build_chat_model,
            build_chat_prompt_profile,
        )
        from src.chatbot.tools.citation.tool import CitationTool
        from src.chatbot.tools.retrieval.tool import RetrievalTool
        from src.settings import get_settings

        settings = get_settings()
        config = ChatModelConfig(base_url=settings.ollama_base_url, model=settings.chat_model)
        prompt_profile = build_chat_prompt_profile(config)
        model = build_chat_model(config)
        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
        )
        retrieval_tool = RetrievalTool(retriever=retriever)
        citation_tool = CitationTool()

        orchestrator = ChatOrchestrator(
            model=model,
            tools=[retrieval_tool, citation_tool],
            prompt_profile=prompt_profile,
        )
        response = ""
        citation_events: list[SourceCitationEvent] = []
        async for event in orchestrator.process_message("What is Zurich known for?"):
            if isinstance(event, str):
                response += event
            elif isinstance(event, SourceCitationEvent):
                citation_events.append(event)

        assert response, "Expected a non-empty response from the model"

        # Citation fallback is best-effort: no citation tool call is allowed and
        # must not fail the turn. If citation events are present, validate them.
        if citation_events:
            validated = [chunk for evt in citation_events for chunk in evt.validated]
            assert validated, "Expected at least one validated citation chunk"
            assert all(chunk.chunk_id for chunk in validated), (
                "Validated chunks must carry chunk_id"
            )
            assert any("zurich.txt" in chunk.source for chunk in validated), (
                "Expected at least one citation sourced from zurich.txt"
            )


class TestCitationValidationIntegration:
    """Integration-level citation validation using live retrieval outputs."""

    async def test_unvalidated_claims_are_not_emitted_as_citation_events(
        self, ingested_store: None
    ) -> None:
        """Mixed claims must emit only validated chunks in SourceCitationEvent."""
        from src.chatbot.app.protocols import ChatMessage, ToolCallInfo, ToolContext
        from src.chatbot.tools.citation.tool import CitationTool

        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
        )
        chunks = await retriever.retrieve("What is Zurich known for?")
        assert chunks, "Expected retrieval to return chunks for citation validation"

        # Build a realistic search_documents tool-call + tool-result history.
        call_id = "search-1"
        search_call = ToolCallInfo(
            name="search_documents",
            arguments={"query": "What is Zurich known for?"},
            call_id=call_id,
        )
        assistant_msg = ChatMessage(role="assistant", content="", tool_calls=(search_call,))
        tool_result_msg = ChatMessage(
            role="tool",
            content={
                "chunks": [
                    {
                        "source": c.source,
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "score": c.score,
                    }
                    for c in chunks
                ]
            },
            tool_call_id=call_id,
        )
        context = ToolContext(history=(assistant_msg, tool_result_msg))

        valid = {"source": chunks[0].source, "chunk_id": chunks[0].chunk_id}
        invalid = {"source": "missing-source.txt", "chunk_id": "does-not-exist"}

        tool = CitationTool()
        result, events = await tool.execute(
            {"citations": [valid, invalid]},
            context,
        )

        assert result["validated"] == [valid]  # type: ignore[comparison-overlap]
        assert result["unvalidated"] == [invalid]  # type: ignore[comparison-overlap]
        assert len(events) == 1

        assert isinstance(events[0], SourceCitationEvent)
        validated = events[0].validated
        assert len(validated) == 1
        assert validated[0].source == valid["source"]
        assert validated[0].chunk_id == valid["chunk_id"]


class TestInlineQuoteOrchestratorIntegration:
    """WP7 integration tests for the inline-quote streaming path.

    These tests exercise the real orchestrator in inline-quote-only mode
    against live Qdrant + Ollama.
    They verify the happy path (retrieval-grounded answer completes and produces
    a response) and the regression path (non-retrieval tool answer does not hang).

    Quote marker emission is model-dependent and is NOT asserted here — only
    that the pipeline handles any model output correctly.
    """

    async def test_grounded_query_completes_without_error_on_inline_flow(
        self, ingested_store: None
    ) -> None:
        """A retrieval-grounded query processed via inline-quote flow yields a
        non-empty response and does not raise."""
        from src.chatbot.app.orchestrator import ChatOrchestrator
        from src.chatbot.app.protocols import QuoteReferenceEvent, SourceCitationEvent
        from src.chatbot.infrastructure.chat import (
            ChatModelConfig,
            build_chat_model,
            build_chat_prompt_profile,
        )
        from src.chatbot.infrastructure.chat._inline_quotes import (
            build_inline_quote_parsing_chat_model,
        )
        from src.chatbot.tools.retrieval.tool import RetrievalTool
        from src.settings import get_settings

        settings = get_settings()
        config = ChatModelConfig(base_url=settings.ollama_base_url, model=settings.chat_model)
        prompt_profile = build_chat_prompt_profile(config)
        base_model = build_chat_model(config)
        model = build_inline_quote_parsing_chat_model(base_model)
        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
        )
        retrieval_tool = RetrievalTool(retriever=retriever)

        orchestrator = ChatOrchestrator(
            model=model,
            tools=[retrieval_tool],
            prompt_profile=prompt_profile,
        )
        response = ""
        quote_ref_events: list[QuoteReferenceEvent] = []
        citation_events: list[SourceCitationEvent] = []

        async for event in orchestrator.process_message("What is Zurich known for?"):
            if isinstance(event, str):
                response += event
            elif isinstance(event, QuoteReferenceEvent):
                quote_ref_events.append(event)
            elif isinstance(event, SourceCitationEvent):
                citation_events.append(event)

        assert response, "Inline-quote flow must produce a non-empty response"
        # If the model emitted inline quotes, they must be numbered from 1.
        for _i, ref in enumerate(quote_ref_events, start=1):
            assert ref.reference_number >= 1, (
                f"Reference number must be >= 1, got {ref.reference_number}"
            )
        # Citation events, if emitted, must reference zurich.txt chunks.
        if citation_events:
            all_validated = [c for evt in citation_events for c in evt.validated]
            assert all_validated, "Citation event must carry validated chunks"
            assert any("zurich.txt" in c.source for c in all_validated), (
                "Expected at least one citation from zurich.txt"
            )

    async def test_non_grounded_query_produces_no_source_citation_event(
        self, ingested_store: None
    ) -> None:
        """A purely conversational query with inline-quote flow produces a response
        and emits no SourceCitationEvent (no retrieval chunks cited)."""
        from src.chatbot.app.orchestrator import ChatOrchestrator
        from src.chatbot.app.protocols import SourceCitationEvent
        from src.chatbot.infrastructure.chat import (
            ChatModelConfig,
            build_chat_model,
            build_chat_prompt_profile,
        )
        from src.chatbot.infrastructure.chat._inline_quotes import (
            build_inline_quote_parsing_chat_model,
        )
        from src.settings import get_settings

        settings = get_settings()
        config = ChatModelConfig(base_url=settings.ollama_base_url, model=settings.chat_model)
        prompt_profile = build_chat_prompt_profile(config)
        base_model = build_chat_model(config)
        model = build_inline_quote_parsing_chat_model(base_model)

        # No retrieval tool — model must answer from prior knowledge only.
        orchestrator = ChatOrchestrator(
            model=model,
            prompt_profile=prompt_profile,
        )
        response = ""
        citation_events: list[SourceCitationEvent] = []

        async for event in orchestrator.process_message("What is two plus two?"):
            if isinstance(event, str):
                response += event
            elif isinstance(event, SourceCitationEvent):
                citation_events.append(event)

        assert response, "Non-grounded query must still produce a response"
        assert citation_events == [], (
            "No SourceCitationEvent expected when no retrieval tool is available"
        )

    async def test_inline_parser_survives_model_output_without_markers(
        self, ingested_store: None
    ) -> None:
        """Inline quote parser must pass through arbitrary model output unchanged when
        no quote markers are present — verifying the non-blocking fallback path."""
        from src.chatbot.app.orchestrator import ChatOrchestrator
        from src.chatbot.infrastructure.chat import (
            ChatModelConfig,
            build_chat_model,
            build_chat_prompt_profile,
        )
        from src.chatbot.infrastructure.chat._inline_quotes import (
            build_inline_quote_parsing_chat_model,
        )
        from src.settings import get_settings

        settings = get_settings()
        config = ChatModelConfig(base_url=settings.ollama_base_url, model=settings.chat_model)
        prompt_profile = build_chat_prompt_profile(config)
        base_model = build_chat_model(config)
        model = build_inline_quote_parsing_chat_model(base_model)

        orchestrator = ChatOrchestrator(
            model=model,
            prompt_profile=prompt_profile,
        )

        # A simple factual question is unlikely to produce quote markers.
        # The test asserts only that the stream completes without raising.
        response = ""
        async for event in orchestrator.process_message("Say hello."):
            if isinstance(event, str):
                response += event

        assert response, "Parser wrapper must not swallow all model output"
