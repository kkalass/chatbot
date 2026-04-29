"""Unit tests for infrastructure chat prompt profiles."""

from datetime import datetime

from src.chatbot.app.prompts import (
    DEFAULT_PROMPTS,
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    build_default_prompts,
)
from src.chatbot.infrastructure.chat import (
    ChatModelConfig,
    DefaultChatPromptProfile,
    SmallModelPromptProfile,
    build_chat_prompt_profile,
)


class TestBuildDefaultPrompts:
    def test_inline_quotes_disabled_returns_default_prompts(self) -> None:
        prompts = build_default_prompts(inline_quotes_enabled=False)

        assert prompts is DEFAULT_PROMPTS

    def test_inline_quotes_enabled_includes_start_marker(self) -> None:
        prompts = build_default_prompts(inline_quotes_enabled=True)
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        assert QUOTE_START_MARKER in system_text

    def test_inline_quotes_enabled_includes_end_marker(self) -> None:
        prompts = build_default_prompts(inline_quotes_enabled=True)
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        assert QUOTE_END_MARKER in system_text

    def test_inline_quotes_enabled_includes_search_result_kind(self) -> None:
        prompts = build_default_prompts(inline_quotes_enabled=True)
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        assert "search_result" in system_text

    def test_inline_quotes_enabled_includes_tool_call_kind(self) -> None:
        prompts = build_default_prompts(inline_quotes_enabled=True)
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        assert "tool_call" in system_text

    def test_inline_quotes_enabled_includes_tool_call_id_field(self) -> None:
        prompts = build_default_prompts(inline_quotes_enabled=True)
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        assert "tool_call_id" in system_text

    def test_inline_quotes_enabled_preserves_base_assistant_instructions(self) -> None:
        prompts = build_default_prompts(inline_quotes_enabled=True)
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        # Base instructions must still be present.
        assert "You are a helpful assistant" in system_text

    def test_inline_quotes_enabled_preserves_citation_prompts_unchanged(self) -> None:
        base = build_default_prompts(inline_quotes_enabled=False)
        with_quotes = build_default_prompts(inline_quotes_enabled=True)
        now = datetime(2026, 4, 29)

        assert base.citation_system_prompt(now) == with_quotes.citation_system_prompt(now)
        assert base.citation_request_message("results", "answer") == with_quotes.citation_request_message("results", "answer")


class TestChatPromptProfiles:
    def test_build_chat_prompt_profile_returns_small_for_llama_model(self) -> None:
        config = ChatModelConfig(base_url="http://localhost:11434", model="llama3.1:8b")

        profile = build_chat_prompt_profile(config)

        assert isinstance(profile, SmallModelPromptProfile)

    def test_build_chat_prompt_profile_returns_default_for_unknown_model(
        self,
    ) -> None:
        config = ChatModelConfig(base_url="http://localhost:11434", model="some-unknown-model")

        profile = build_chat_prompt_profile(config)

        assert isinstance(profile, DefaultChatPromptProfile)

    def test_default_profile_adjust_prompts_is_identity(self) -> None:
        profile = DefaultChatPromptProfile()

        assert profile.adjust_prompts(DEFAULT_PROMPTS) is DEFAULT_PROMPTS

    def test_default_profile_adjust_tool_description_is_identity(self) -> None:
        profile = DefaultChatPromptProfile()
        description = "Original description"

        assert profile.adjust_tool_description("search_documents", description) == description

    def test_default_profile_adjust_parameter_schema_is_identity(self) -> None:
        profile = DefaultChatPromptProfile()
        schema = {"type": "object", "properties": {"query": {"type": "string"}}}

        assert profile.adjust_parameter_schema("search_documents", schema) == schema

    def test_small_profile_appends_prompt_guidance(self) -> None:
        profile = SmallModelPromptProfile()

        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 28))
        citation_system_text = adjusted.citation_system_prompt(datetime(2026, 4, 28))
        citation_text = adjusted.citation_request_message(
            "- source: a.txt\n  chunk_id: 1\n  content: |\n    text",
            "Answer text.",
        )

        assert "never as JSON-encoded strings" in system_text
        assert "native JSON values" in citation_system_text
        assert "top-level key must be exactly: citations" in citation_text
        assert "do not invent other top-level keys such as d" in citation_text

    def test_small_profile_adds_inline_quote_json_hardening(self) -> None:
        profile = SmallModelPromptProfile()
        prompts_with_quotes = build_default_prompts(inline_quotes_enabled=True)

        adjusted = profile.adjust_prompts(prompts_with_quotes)
        system_text = adjusted.system_prompt(datetime(2026, 4, 29))

        # Hardening language must be present alongside the marker contract.
        assert "Inline citation JSON rules" in system_text
        assert "no extra fields" in system_text
        assert "tool_call_id" in system_text

    def test_small_profile_inline_quote_hardening_present_even_without_base_markers(
        self,
    ) -> None:
        # Hardening should be appended regardless of base prompt content —
        # it is harmless when inline quotes are disabled and reinforcing when enabled.
        profile = SmallModelPromptProfile()

        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 29))

        assert "Inline citation JSON rules" in system_text

    def test_small_profile_appends_only_citation_tool_description(self) -> None:
        profile = SmallModelPromptProfile()
        citation_description = "Declare citation pairs."
        retrieval_description = "Search docs."

        adjusted_citation = profile.adjust_tool_description("cite_sources", citation_description)
        adjusted_retrieval = profile.adjust_tool_description(
            "search_documents", retrieval_description
        )

        assert "JSON object with a citations array" in adjusted_citation
        assert "do not invent other top-level keys such as d" in adjusted_citation
        assert adjusted_retrieval == retrieval_description

    def test_small_profile_simplifies_citation_parameter_schema(self) -> None:
        profile = SmallModelPromptProfile()
        schema = {
            "type": "object",
            "properties": {"citations": {"type": "array", "items": {"$ref": "#/$defs/X"}}},
            "$defs": {"X": {"type": "object"}},
        }

        adjusted = profile.adjust_parameter_schema("cite_sources", schema)

        assert adjusted["type"] == "object"
        assert adjusted["additionalProperties"] is False
        assert adjusted["required"] == ["citations"]
        assert "$defs" not in adjusted
        citations = adjusted["properties"]["citations"]  # type: ignore[index]
        assert citations["type"] == "array"  # type: ignore[index]
        assert citations["items"]["type"] == "object"  # type: ignore[index]

    def test_small_profile_keeps_other_parameter_schemas_unchanged(self) -> None:
        profile = SmallModelPromptProfile()
        schema = {"type": "object", "properties": {"query": {"type": "string"}}}

        adjusted = profile.adjust_parameter_schema("search_documents", schema)

        assert adjusted == schema
