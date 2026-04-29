"""Unit tests for quote/event models introduced in Phase 7 WP1."""

import pytest
from pydantic import ValidationError

from src.chatbot.app.protocols import QuoteReferenceEvent, SearchResultQuote, ToolCallQuote


class TestQuoteModels:
    def test_search_result_quote_validates_expected_fields(self) -> None:
        quote = SearchResultQuote(
            tool_call_id="search-1",
            source="corpus/report.txt",
            chunk_id="chunk-42",
            quote_text="automation impact is moderate",
        )

        assert quote.kind == "search_result"
        assert quote.tool_call_id == "search-1"

    def test_search_result_quote_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            SearchResultQuote.model_validate(
                {
                    "claim": "x",
                    "tool_call_id": "search-1",
                    "source": "corpus/report.txt",
                    "chunk_id": "chunk-42",
                    "unsupported": True,
                }
            )

    def test_tool_call_quote_validates_expected_fields(self) -> None:
        quote = ToolCallQuote(
            tool_call_id="tool-1",
        )

        assert quote.kind == "tool_call"
        assert quote.tool_call_id == "tool-1"

    def test_quote_reference_event_requires_positive_reference_number(self) -> None:
        with pytest.raises(ValidationError):
            QuoteReferenceEvent(reference_number=0, canonical_key="search:search-1:doc:1")

    def test_quote_reference_event_accepts_valid_payload(self) -> None:
        event = QuoteReferenceEvent(reference_number=1, canonical_key="search:search-1:doc:1")

        assert event.reference_number == 1
