# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integration tests for text RAG: ingest fixture corpus, retrieve, and verify citations.

These tests require live Qdrant and Ollama services. They are skipped
automatically when ``INTEGRATION_TESTS=1`` is not set in the environment, so
the regular unit-test run stays fast and infrastructure-independent.

Run with:
    INTEGRATION_TESTS=1 uv run pytest tests/integration/
"""

import os
import uuid
from pathlib import Path

import pytest
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.chatbot.infrastructure.embeddings_text import TextEmbedderConfig, build_text_embedder
from src.chatbot.infrastructure.retrieval import build_qdrant_retriever
from src.ingest.app import IngestionConfig, IngestionPipeline
from src.ingest.build_from_settings import build_format_handlers
from src.ingest.infrastructure.embeddings_document import (
    build_ollama_document_embedder,
)
from src.shared.qdrant import build_qdrant_document_store

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


def _build_test_store() -> QdrantDocumentStore:
    """Build a test-scoped Qdrant document store."""
    return build_qdrant_document_store(
        host="localhost",
        port=6333,
        collection=_TEST_COLLECTION,
        embedding_dim=768,
    )


_TEXT_EMBEDDER_CONFIG = TextEmbedderConfig(
    url="http://localhost:11434",
    embedding_model="nomic-embed-text",
)

_RETRIEVAL_TOP_K = 5


@pytest.fixture(scope="module")
def ingested_store() -> None:
    """Ingest the fixture corpus once for the entire module."""
    store = _build_test_store()
    pipeline = IngestionPipeline(
        config=_INGESTION_CONFIG,
        document_store=store,
        embedder=build_ollama_document_embedder(
            model="nomic-embed-text",
            url="http://localhost:11434",
        ),
        format_handlers=build_format_handlers(image_service=None, extracted_image_store=None),
    )
    count = pipeline.ingest_corpus(_FIXTURE_DIR)
    assert count > 0, f"Expected > 0 chunks to be written; got {count}"


class TestGroundedRetrieval:
    async def test_zurich_query_returns_chunks(self, ingested_store: None) -> None:
        retriever = build_qdrant_retriever(
            top_k=_RETRIEVAL_TOP_K,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
            document_store=_build_test_store(),
        )
        chunks = await retriever.retrieve("What is the largest city in Switzerland?")
        assert len(chunks) > 0, "Expected at least one chunk for a known topic"

    async def test_citation_source_points_to_fixture_file(self, ingested_store: None) -> None:
        retriever = build_qdrant_retriever(
            top_k=_RETRIEVAL_TOP_K,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
            document_store=_build_test_store(),
        )
        chunks = await retriever.retrieve("financial center")
        sources = {c.source for c in chunks}
        assert any("zurich.txt" in s for s in sources), (
            f"Expected 'zurich.txt' in sources; got: {sources}"
        )

    async def test_chunk_content_not_empty(self, ingested_store: None) -> None:
        retriever = build_qdrant_retriever(
            top_k=_RETRIEVAL_TOP_K,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
            document_store=_build_test_store(),
        )
        chunks = await retriever.retrieve("Zurich old town")
        for chunk in chunks:
            assert chunk.content.strip(), f"Empty content in chunk {chunk.chunk_id}"


class TestOrchestratorWithCitationModel:
    """End-to-end: orchestrator wired with CitationModel + RetrievalTool surfaces
    validated citations and any hallucinated ones produced by the live model.

    Citation emission is model-dependent; the test asserts only structural
    properties of any events the model produces, plus that the response
    completes without errors.
    """

    async def test_grounded_query_completes_and_surfaces_citations(
        self, ingested_store: None
    ) -> None:
        from src.chatbot.app.citation import CitationModel
        from src.chatbot.app.orchestrator import ChatOrchestrator
        from src.chatbot.build_from_settings import build_chat_model_with_profile
        from src.chatbot.contracts.citation import (
            HallucinatedCitation,
            NumberedCitation,
            UnsubstantiatedClaim,
        )
        from src.chatbot.infrastructure.tools.retrieval import RetrievalTool
        from src.shared.settings import get_settings

        settings = get_settings()
        chat_model, model_profile = build_chat_model_with_profile(settings)
        retriever = build_qdrant_retriever(
            top_k=_RETRIEVAL_TOP_K,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
            document_store=_build_test_store(),
        )
        retrieval_tool = RetrievalTool(retriever=retriever)
        citation_layer = CitationModel(chat_model, tools=[retrieval_tool])

        orchestrator = ChatOrchestrator(
            citation_layer,
            tools=[retrieval_tool],
            model_profile=model_profile,
        )

        response = ""
        numbered: list[NumberedCitation] = []
        hallucinated: list[HallucinatedCitation] = []
        unsubstantiated: list[UnsubstantiatedClaim] = []
        async for event in orchestrator.process_message("What is Zurich known for?"):
            if isinstance(event, str):
                response += event
            elif isinstance(event, NumberedCitation):
                numbered.append(event)
            elif isinstance(event, UnsubstantiatedClaim):
                unsubstantiated.append(event)
            elif isinstance(event, HallucinatedCitation):
                hallucinated.append(event)

        assert response, "Expected a non-empty response from the model"

        seen: set[int] = set()
        for item in numbered:
            assert item.reference_number >= 1
            seen.add(item.reference_number)
        if seen:
            assert min(seen) == 1

        for h in hallucinated:
            assert h.reason
            assert h.raw_marker_text
