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
    def test_build_default_prompts_returns_default_prompts(self) -> None:
        prompts = build_default_prompts()

        assert prompts is DEFAULT_PROMPTS

    def test_includes_start_marker(self) -> None:
        prompts = build_default_prompts()
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        assert QUOTE_START_MARKER in system_text

    def test_includes_end_marker(self) -> None:
        prompts = build_default_prompts()
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        assert QUOTE_END_MARKER in system_text

    def test_includes_search_result_kind(self) -> None:
        prompts = build_default_prompts()
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        assert "search_result" in system_text

    def test_includes_tool_call_kind(self) -> None:
        prompts = build_default_prompts()
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        assert "tool_call" in system_text

    def test_includes_tool_call_id_field(self) -> None:
        prompts = build_default_prompts()
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        assert "tool_call_id" in system_text

    def test_preserves_base_assistant_instructions(self) -> None:
        prompts = build_default_prompts()
        system_text = prompts.system_prompt(datetime(2026, 4, 29))

        # Base instructions must still be present.
        assert "You are a helpful assistant" in system_text


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

        assert "never as JSON-encoded strings" in system_text

    def test_small_profile_adds_inline_quote_json_hardening(self) -> None:
        profile = SmallModelPromptProfile()
        prompts_with_quotes = build_default_prompts()

        adjusted = profile.adjust_prompts(prompts_with_quotes)
        system_text = adjusted.system_prompt(datetime(2026, 4, 29))

        # Hardening language must be present alongside the marker contract.
        assert "Inline citation JSON rules" in system_text
        assert "no extra fields" in system_text
        assert "tool_call_id" in system_text

    def test_small_profile_keeps_other_parameter_schemas_unchanged(self) -> None:
        profile = SmallModelPromptProfile()
        schema = {"type": "object", "properties": {"query": {"type": "string"}}}

        adjusted = profile.adjust_parameter_schema("search_documents", schema)

        assert adjusted == schema
