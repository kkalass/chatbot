"""Unit tests for the text-encoded tool call parser in the Ollama chat model."""

from src.chatbot.infrastructure.chat._ollama import (
    _try_parse_text_tool_call,  # pyright: ignore[reportPrivateUsage]
)


class TestTryParseTextToolCall:
    def test_valid_tool_call_returns_tool_call_info(self) -> None:
        result = _try_parse_text_tool_call(
            '{"name": "search_documents", "arguments": {"query": "AI labour market"}}'
        )
        assert result is not None
        assert result.name == "search_documents"
        assert result.arguments == {"query": "AI labour market"}
        assert result.call_id == "search_documents"

    def test_call_id_equals_name(self) -> None:
        result = _try_parse_text_tool_call('{"name": "get_vacation_days", "arguments": {}}')
        assert result is not None
        assert result.call_id == result.name

    def test_leading_and_trailing_whitespace_is_stripped(self) -> None:
        result = _try_parse_text_tool_call('  \n{"name": "fn", "arguments": {"x": 1}}\n  ')
        assert result is not None
        assert result.name == "fn"

    def test_fenced_json_block_is_parsed(self) -> None:
        result = _try_parse_text_tool_call(
            '```json\n{"name": "cite_sources", "arguments": {"citations": []}}\n```'
        )
        assert result is not None
        assert result.name == "cite_sources"
        assert result.arguments == {"citations": []}

    def test_plain_fenced_json_block_is_parsed(self) -> None:
        result = _try_parse_text_tool_call(
            '```\n{"name": "cite_sources", "arguments": {"citations": []}}\n```'
        )
        assert result is not None
        assert result.name == "cite_sources"

    def test_parameters_alias_is_accepted(self) -> None:
        result = _try_parse_text_tool_call(
            '{"name": "cite_sources", "parameters": {"citations": []}}'
        )
        assert result is not None
        assert result.name == "cite_sources"
        assert result.arguments == {"citations": []}

    def test_string_encoded_arguments_object_is_decoded(self) -> None:
        result = _try_parse_text_tool_call(
            '{"name": "cite_sources", "arguments": "{\\"citations\\": []}"}'
        )
        assert result is not None
        assert result.arguments == {"citations": []}

    def test_plain_text_returns_none(self) -> None:
        assert _try_parse_text_tool_call("Here is my answer.") is None

    def test_invalid_json_returns_none(self) -> None:
        assert _try_parse_text_tool_call("{not valid json}") is None

    def test_json_array_returns_none(self) -> None:
        assert _try_parse_text_tool_call('[{"name": "fn", "arguments": {}}]') is None

    def test_missing_name_field_returns_none(self) -> None:
        assert _try_parse_text_tool_call('{"arguments": {"x": 1}}') is None

    def test_missing_arguments_field_returns_none(self) -> None:
        assert _try_parse_text_tool_call('{"name": "fn"}') is None

    def test_name_not_string_returns_none(self) -> None:
        assert _try_parse_text_tool_call('{"name": 42, "arguments": {}}') is None

    def test_arguments_not_dict_returns_none(self) -> None:
        assert _try_parse_text_tool_call('{"name": "fn", "arguments": "bad"}') is None

    def test_empty_string_returns_none(self) -> None:
        assert _try_parse_text_tool_call("") is None

    def test_nested_arguments_are_preserved(self) -> None:
        result = _try_parse_text_tool_call(
            '{"name": "fn", "arguments": {"filters": {"year": 2024}, "top_k": 5}}'
        )
        assert result is not None
        assert result.arguments == {"filters": {"year": 2024}, "top_k": 5}
