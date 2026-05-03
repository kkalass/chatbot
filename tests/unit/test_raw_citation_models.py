"""Tests for the RawCitation Pydantic models."""

import pytest
from pydantic import ValidationError

from src.chatbot.app.citation import DocumentRawCitation, ToolRawCitation


class TestDocumentRawCitation:
    def test_minimal_valid_payload(self) -> None:
        cit = DocumentRawCitation.model_validate(
            {"kind": "document", "tool_call_id": "tc1", "source": "s.md", "chunk_id": "c1"}
        )
        assert cit.kind == "document"
        assert cit.tool_call_id == "tc1"
        assert cit.source == "s.md"
        assert cit.chunk_id == "c1"
        assert cit.quote_text is None
        assert cit.claim is None
        assert cit.raw_marker_text == ""

    def test_optional_quote_text_and_claim_carried(self) -> None:
        cit = DocumentRawCitation.model_validate(
            {
                "kind": "document",
                "tool_call_id": "tc1",
                "source": "s.md",
                "chunk_id": "c1",
                "quote_text": "verbatim",
                "claim": "the claim",
            }
        )
        assert cit.quote_text == "verbatim"
        assert cit.claim == "the claim"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DocumentRawCitation.model_validate(
                {
                    "kind": "document",
                    "tool_call_id": "tc1",
                    "source": "s",
                    "chunk_id": "c",
                    "unknown": "x",
                }
            )

    def test_missing_required_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DocumentRawCitation.model_validate({"kind": "document", "tool_call_id": "tc1"})

    def test_is_frozen(self) -> None:
        cit = DocumentRawCitation.model_validate(
            {"kind": "document", "tool_call_id": "tc1", "source": "s", "chunk_id": "c"}
        )
        with pytest.raises(ValidationError):
            cit.tool_call_id = "other"  # type: ignore[misc]


class TestToolRawCitation:
    def test_minimal_valid_payload(self) -> None:
        cit = ToolRawCitation.model_validate({"kind": "tool_call", "tool_call_id": "tc-1"})
        assert cit.kind == "tool_call"
        assert cit.tool_call_id == "tc-1"
        assert cit.raw_marker_text == ""

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolRawCitation.model_validate(
                {"kind": "tool_call", "tool_call_id": "tc", "tool_name": "x"}
            )

    def test_missing_tool_call_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolRawCitation.model_validate({"kind": "tool_call"})
