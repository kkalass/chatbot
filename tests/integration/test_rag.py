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
        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
        )
        chunks = await retriever.retrieve("What is the largest city in Switzerland?")
        assert len(chunks) > 0, "Expected at least one chunk for a known topic"

    async def test_citation_source_points_to_fixture_file(self, ingested_store: None) -> None:
        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
        )
        chunks = await retriever.retrieve("financial center")
        sources = {c.source for c in chunks}
        assert any("zurich.txt" in s for s in sources), (
            f"Expected 'zurich.txt' in sources; got: {sources}"
        )

    async def test_chunk_content_not_empty(self, ingested_store: None) -> None:
        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
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
        from src.chatbot.app.protocols import (
            HallucinatedCitation,
            NumberedCitation,
            UnsubstantiatedClaim,
        )
        from src.chatbot.infrastructure.chat import (
            ChatModelConfig,
            build_chat_model,
            build_chat_model_profile,
        )
        from src.chatbot.tools.retrieval.tool import RetrievalTool
        from src.settings import get_settings

        settings = get_settings()
        config = ChatModelConfig(base_url=settings.chat_base_url, model=settings.chat_model)
        model_profile = build_chat_model_profile(config)
        chat_model = build_chat_model(
            config, parse_text_tool_calls=model_profile.parse_text_tool_calls
        )
        retriever = build_retriever(
            config=_RETRIEVER_CONFIG,
            text_embedder=build_text_embedder(_TEXT_EMBEDDER_CONFIG),
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
