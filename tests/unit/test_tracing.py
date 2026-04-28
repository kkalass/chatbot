"""Unit tests for tracing helpers."""

from src.chatbot.observability.tracing import to_attribute_text


def test_to_attribute_text_serializes_non_string_values() -> None:
    text = to_attribute_text({"a": 1, "b": ["x", "y"]})

    assert '"a": 1' in text
    assert '"b": ["x", "y"]' in text


def test_to_attribute_text_truncates_long_values() -> None:
    text = to_attribute_text("x" * 20, max_chars=8)

    assert text == "xxxxxxxx...<truncated>"
