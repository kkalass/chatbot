"""Unit tests for infrastructure chat prompt profiles."""

from datetime import datetime

from src.chatbot.app.prompts import (
    DEFAULT_PROMPTS,
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
)
from src.chatbot.infrastructure.chat import (
    ChatModelConfig,
    DefaultChatPromptProfile,
    QwenCoderPromptProfile,
    SmallModelPromptProfile,
    build_chat_prompt_profile,
)


class TestDefaultPrompts:
    def test_includes_start_marker(self) -> None:
        system_text = DEFAULT_PROMPTS.system_prompt(datetime(2026, 4, 29))

        assert QUOTE_START_MARKER in system_text

    def test_includes_end_marker(self) -> None:
        system_text = DEFAULT_PROMPTS.system_prompt(datetime(2026, 4, 29))

        assert QUOTE_END_MARKER in system_text

    def test_includes_search_result_kind(self) -> None:
        system_text = DEFAULT_PROMPTS.system_prompt(datetime(2026, 4, 29))

        assert "search_result" in system_text

    def test_includes_tool_call_kind(self) -> None:
        system_text = DEFAULT_PROMPTS.system_prompt(datetime(2026, 4, 29))

        assert "tool_call" in system_text

    def test_includes_tool_call_id_field(self) -> None:
        system_text = DEFAULT_PROMPTS.system_prompt(datetime(2026, 4, 29))

        assert "tool_call_id" in system_text

    def test_preserves_base_assistant_instructions(self) -> None:
        system_text = DEFAULT_PROMPTS.system_prompt(datetime(2026, 4, 29))

        # Base instructions must still be present.
        assert "You are a helpful assistant" in system_text

    def test_default_user_message_reminder_includes_exact_quote_markers(self) -> None:
        user_text = DEFAULT_PROMPTS.user_message("How many vacation days do I have?")

        assert QUOTE_START_MARKER in user_text
        assert QUOTE_END_MARKER in user_text
        assert "do not use any marker variants" in user_text

    def test_default_user_message_reminder_requires_tool_call_fields(self) -> None:
        user_text = DEFAULT_PROMPTS.user_message("How many vacation days do I have?")

        assert "kind=tool_call" in user_text
        assert "tool_call_id" in user_text
        assert "source" in user_text
        assert "chunk_id" in user_text
        assert "include kind, claim," not in user_text
        # tool_name is no longer required in tool_call markers (model output is untrusted)
        assert "tool_name" not in user_text or "tool_call_id" in user_text

    def test_default_user_message_reminder_disallows_invented_tool_call_ids(self) -> None:
        user_text = DEFAULT_PROMPTS.user_message("How many vacation days do I have?")

        assert "Never invent IDs" in user_text
        assert "If the exact ID is not visible, emit no marker" in user_text

    def test_default_user_message_reminder_encourages_single_marker_per_tool_call(self) -> None:
        user_text = DEFAULT_PROMPTS.user_message("How many vacation days do I have?")

        assert "one marker\nper sentence, not one per paragraph" in user_text


class TestChatPromptProfiles:
    def test_build_chat_prompt_profile_returns_small_for_llama_model(self) -> None:
        config = ChatModelConfig(base_url="http://localhost:11434", model="llama3.1:8b")

        profile = build_chat_prompt_profile(config)

        assert isinstance(profile, SmallModelPromptProfile)

    def test_build_chat_prompt_profile_returns_small_for_qwen_model(self) -> None:
        config = ChatModelConfig(base_url="http://localhost:11434", model="qwen3.5:9b")

        profile = build_chat_prompt_profile(config)

        assert isinstance(profile, SmallModelPromptProfile)

    def test_build_chat_prompt_profile_returns_qwen_coder_profile_for_qwen_coder(
        self,
    ) -> None:
        config = ChatModelConfig(base_url="http://localhost:11434", model="qwen2.5-coder:14b")

        profile = build_chat_prompt_profile(config)

        assert isinstance(profile, QwenCoderPromptProfile)

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
        prompts_with_quotes = DEFAULT_PROMPTS

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

    def test_qwen_coder_profile_requires_explicit_tool_name(self) -> None:
        profile = QwenCoderPromptProfile()

        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 30))

        assert 'NEVER output a bare arguments object such as {"citations": [...]}' in system_text
        assert (
            'must be a single object of the form {"name": "tool_name", "arguments": {...}}'
            in system_text
        )
        assert 'tool name must be exactly "cite_sources"' in system_text
