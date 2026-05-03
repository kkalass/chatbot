"""Tests for the RawCitation Pydantic model."""

import pytest
from pydantic import ValidationError

from src.chatbot.app.citation import RawCitation


class TestRawCitation:
    def test_minimal_payload_without_chunk_id(self) -> None:
        cit = RawCitation.model_validate({"tool_call_id": "tc1"})
        assert cit.tool_call_id == "tc1"
        assert cit.chunk_id is None
        assert cit.raw_marker_text == ""

    def test_payload_with_chunk_id(self) -> None:
        cit = RawCitation.model_validate({"tool_call_id": "tc1", "chunk_id": "c1"})
        assert cit.tool_call_id == "tc1"
        assert cit.chunk_id == "c1"

    def test_unknown_fields_ignored(self) -> None:
        # Legacy 'kind', 'source', 'quote_text', 'claim' must not cause failures.
        cit = RawCitation.model_validate(
            {
                "tool_call_id": "tc1",
                "chunk_id": "c1",
                "kind": "document",
                "source": "s.md",
                "quote_text": "verbatim",
                "claim": "the claim",
            }
        )
        assert cit.tool_call_id == "tc1"
        assert cit.chunk_id == "c1"

    def test_missing_tool_call_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawCitation.model_validate({"chunk_id": "c1"})

    def test_is_frozen(self) -> None:
        cit = RawCitation.model_validate({"tool_call_id": "tc1"})
        with pytest.raises(ValidationError):
            cit.tool_call_id = "other"  # type: ignore[misc]
