# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for ToolInputModel JSON-string coercion behavior."""

from pydantic import ValidationError

from src.chatbot.infrastructure.tools._input_model import ToolInputModel


class _DummyInput(ToolInputModel):
    items: list[int]
    query: str


def test_coerces_json_string_for_list_field() -> None:
    parsed = _DummyInput.model_validate({"items": "[1, 2, 3]", "query": "hello"})
    assert parsed.items == [1, 2, 3]


def test_keeps_plain_string_for_string_field() -> None:
    parsed = _DummyInput.model_validate({"items": [7], "query": "[not json to parse]"})
    assert parsed.query == "[not json to parse]"


def test_invalid_json_string_still_fails_normal_validation() -> None:
    try:
        _DummyInput.model_validate({"items": "[1,2", "query": "hello"})
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected ValidationError for malformed serialized list")
