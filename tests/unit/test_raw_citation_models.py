# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the RawCitation Pydantic model."""

import pytest
from pydantic import ValidationError

from src.chatbot.contracts.citation import RawCitation


class TestRawCitation:
    def test_minimal_payload_with_ref(self) -> None:
        cit = RawCitation.model_validate({"ref": "tok1"})
        assert cit.ref == "tok1"
        assert cit.kind is None
        assert cit.raw_marker_text == ""

    def test_unsubstantiated_payload_without_ref(self) -> None:
        cit = RawCitation.model_validate({"kind": "unsubstantiated"})
        assert cit.ref is None
        assert cit.kind == "unsubstantiated"

    def test_empty_payload_is_valid(self) -> None:
        # All fields are optional; semantic validation is the layer's job.
        cit = RawCitation.model_validate({})
        assert cit.ref is None
        assert cit.kind is None

    def test_is_frozen(self) -> None:
        cit = RawCitation.model_validate({"ref": "tok1"})
        with pytest.raises(ValidationError):
            cit.ref = "other"  # type: ignore[misc]
