"""Unit tests for prompt profiles after the citation-layer refactor.

The base ``Prompts`` no longer carry any citation framing — those concerns
live exclusively in the :class:`CitationLayer`. Profiles must therefore not
reintroduce them.
"""

from datetime import datetime

from src.chatbot.app.prompts import DEFAULT_PROMPTS
from src.chatbot.infrastructure.chat import (
    ChatModelConfig,
    DefaultChatPromptProfile,
    QwenCoderPromptProfile,
    SmallModelPromptProfile,
    build_chat_prompt_profile,
)


class TestDefaultPrompts:
    def test_base_system_prompt_is_citation_agnostic(self) -> None:
        system_text = DEFAULT_PROMPTS.system_prompt(datetime(2026, 4, 29))

        assert "You are a helpful assistant" in system_text
        assert "<°_quote_°>" not in system_text
        assert "kind=" not in system_text
        assert "tool_call_id" not in system_text

    def test_default_user_message_is_pure_passthrough(self) -> None:
        text = "How many vacation days do I have?"
        assert DEFAULT_PROMPTS.user_message(text) == text


class TestProfileSelection:
    def test_llama_selects_small_profile(self) -> None:
        config = ChatModelConfig(base_url="http://localhost:11434", model="llama3.1:8b")
        assert isinstance(build_chat_prompt_profile(config), SmallModelPromptProfile)

    def test_qwen_selects_small_profile(self) -> None:
        config = ChatModelConfig(base_url="http://localhost:11434", model="qwen3.5:9b")
        assert isinstance(build_chat_prompt_profile(config), SmallModelPromptProfile)

    def test_qwen_coder_selects_qwen_coder_profile(self) -> None:
        config = ChatModelConfig(base_url="http://localhost:11434", model="qwen2.5-coder:14b")
        assert isinstance(build_chat_prompt_profile(config), QwenCoderPromptProfile)

    def test_unknown_model_selects_default_profile(self) -> None:
        config = ChatModelConfig(base_url="http://localhost:11434", model="some-unknown-model")
        assert isinstance(build_chat_prompt_profile(config), DefaultChatPromptProfile)


class TestDefaultChatPromptProfile:
    def test_adjust_prompts_is_identity(self) -> None:
        profile = DefaultChatPromptProfile()
        assert profile.adjust_prompts(DEFAULT_PROMPTS) is DEFAULT_PROMPTS

    def test_adjust_tool_description_is_identity(self) -> None:
        profile = DefaultChatPromptProfile()
        assert profile.adjust_tool_description("search_documents", "desc") == "desc"

    def test_adjust_parameter_schema_is_identity(self) -> None:
        profile = DefaultChatPromptProfile()
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        assert profile.adjust_parameter_schema("search_documents", schema) == schema


class TestSmallModelPromptProfile:
    def test_appends_tool_call_guidance(self) -> None:
        profile = SmallModelPromptProfile()
        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 28))

        assert "never as JSON-encoded strings" in system_text
        assert "Tool calls must be submitted via the tool-calling API" in system_text

    def test_does_not_reintroduce_citation_concerns(self) -> None:
        # Citation guidance is owned exclusively by the CitationLayer; the
        # profile must stay free of it after the Phase 9 refactor.
        profile = SmallModelPromptProfile()
        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 29))

        assert "Inline citation JSON rules" not in system_text
        assert "<°_quote_°>" not in system_text


class TestQwenCoderPromptProfile:
    def test_inherits_small_profile_guidance(self) -> None:
        profile = QwenCoderPromptProfile()
        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 30))

        assert "Tool calls must be submitted via the tool-calling API" in system_text

    def test_appends_qwen_coder_specific_rules(self) -> None:
        profile = QwenCoderPromptProfile()
        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 30))

        assert "Qwen-coder specific tool-call rules" in system_text
        assert "NEVER omit the tool name when calling a tool." in system_text

    def test_does_not_mention_removed_cite_sources_tool(self) -> None:
        profile = QwenCoderPromptProfile()
        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 30))

        assert "cite_sources" not in system_text
