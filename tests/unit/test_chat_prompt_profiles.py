"""Unit tests for infrastructure chat prompt profiles."""

from src.chatbot.app.prompts import DEFAULT_PROMPTS
from src.chatbot.infrastructure.chat import (
    ChatModelConfig,
    DefaultChatPromptProfile,
    build_chat_prompt_profile,
)


class TestChatPromptProfiles:
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
