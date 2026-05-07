# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for model profiles after the citation-layer refactor.

The base ``Prompts`` no longer carry any citation framing — those concerns
live exclusively in the :class:`CitationLayer`. Profiles must therefore not
reintroduce them.
"""

from datetime import datetime

from src.chatbot.app.chat_prompts import DEFAULT_PROMPTS
from src.chatbot.build_from_settings import build_model_profile
from src.chatbot.infrastructure.chat import (
    DefaultChatModelProfile,
    QwenCoderModelProfile,
    SmallModelProfile,
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
        assert isinstance(build_model_profile("llama3.1:8b"), SmallModelProfile)

    def test_qwen_selects_small_profile(self) -> None:
        assert isinstance(build_model_profile("qwen3.5:9b"), SmallModelProfile)

    def test_qwen_coder_selects_qwen_coder_profile(self) -> None:
        assert isinstance(build_model_profile("qwen2.5-coder:14b"), QwenCoderModelProfile)

    def test_unknown_model_selects_default_profile(self) -> None:
        assert isinstance(build_model_profile("some-unknown-model"), DefaultChatModelProfile)


class TestParseTextToolCallsFlag:
    def test_default_profile_returns_false(self) -> None:
        assert DefaultChatModelProfile().parse_text_tool_calls is False

    def test_default_profile_allows_constructor_override(self) -> None:
        assert DefaultChatModelProfile(parse_text_tool_calls=True).parse_text_tool_calls is True

    def test_small_model_profile_returns_false(self) -> None:
        assert SmallModelProfile().parse_text_tool_calls is False

    def test_small_model_profile_allows_constructor_override(self) -> None:
        assert SmallModelProfile(parse_text_tool_calls=True).parse_text_tool_calls is True

    def test_qwen_coder_profile_returns_true(self) -> None:
        assert QwenCoderModelProfile().parse_text_tool_calls is True


class TestDefaultChatModelProfile:
    def test_adjust_prompts_is_identity(self) -> None:
        profile = DefaultChatModelProfile()
        assert profile.adjust_prompts(DEFAULT_PROMPTS) is DEFAULT_PROMPTS

    def test_adjust_tool_description_is_identity(self) -> None:
        profile = DefaultChatModelProfile()
        assert profile.adjust_tool_description("search_documents", "desc") == "desc"

    def test_adjust_parameter_schema_is_identity(self) -> None:
        profile = DefaultChatModelProfile()
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        assert profile.adjust_parameter_schema("search_documents", schema) == schema


class TestSmallModelProfile:
    def test_appends_tool_call_guidance(self) -> None:
        profile = SmallModelProfile()
        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 28))

        assert "never as JSON-encoded strings" in system_text
        assert "Tool calls must be submitted via the tool-calling API" in system_text

    def test_does_not_reintroduce_citation_concerns(self) -> None:
        # Citation guidance is owned exclusively by the CitationLayer; the
        # profile must stay free of it after the Phase 9 refactor.
        profile = SmallModelProfile()
        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 29))

        assert "Inline citation JSON rules" not in system_text
        assert "<°_quote_°>" not in system_text


class TestQwenCoderModelProfile:
    def test_inherits_small_profile_guidance(self) -> None:
        profile = QwenCoderModelProfile()
        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 30))

        assert "Tool calls must be submitted via the tool-calling API" in system_text

    def test_appends_qwen_coder_specific_rules(self) -> None:
        profile = QwenCoderModelProfile()
        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 30))

        assert "Qwen-coder specific tool-call rules" in system_text
        assert "NEVER omit the tool name when calling a tool." in system_text

    def test_does_not_mention_removed_cite_sources_tool(self) -> None:
        profile = QwenCoderModelProfile()
        adjusted = profile.adjust_prompts(DEFAULT_PROMPTS)
        system_text = adjusted.system_prompt(datetime(2026, 4, 30))

        assert "cite_sources" not in system_text
