"""Default prompt profile implementation.

The prompt-profile interface lives in ``src.chatbot.app.protocols.PromptProfile``.
This module provides the default identity implementation used by models that
require no prompt-level tuning.
"""

from dataclasses import dataclass, replace
from datetime import datetime

from src.chatbot.app.prompts import Prompts
from src.chatbot.app.protocols import JsonObject, PromptProfile


@dataclass(frozen=True)
class DefaultChatPromptProfile(PromptProfile):
    """Identity prompt profile — passes all app-level prompts and schemas through unchanged.

    Used as the baseline for models that do not require prompt-level tuning.
    """

    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        return prompts

    def adjust_tool_description(self, tool_name: str, description: str) -> str:
        return description

    def adjust_parameter_schema(self, tool_name: str, schema: JsonObject) -> JsonObject:
        return schema


@dataclass(frozen=True)
class SmallModelPromptProfile(DefaultChatPromptProfile):
    """Prompt profile with stricter tool-calling guidance for weaker tool-call models."""

    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        def _system_prompt(now: datetime) -> str:
            return f"""{prompts.system_prompt(now)}

When calling tools, provide arguments as native JSON values (arrays/objects),
never as JSON-encoded strings and never as schema metadata.

Tool call rules:
- NEVER write a tool call as a code block or as JSON in your response text.
- NEVER describe what you intend to do with a tool instead of doing it.
- Tool calls must be submitted via the tool-calling API, not written as prose."""

        return replace(prompts, system_prompt=_system_prompt)


@dataclass(frozen=True)
class QwenCoderPromptProfile(SmallModelPromptProfile):
    """Prompt profile for qwen-coder variants that leak tool-call JSON into text.

    These models often serialise only the arguments object or an incomplete JSON
    fragment instead of using the native tool-calling channel. The prompt must
    make the required call shape explicit.
    """

    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        base_prompts = super().adjust_prompts(prompts)

        def _system_prompt(now: datetime) -> str:
            return f"""{base_prompts.system_prompt(now)}

Qwen-coder specific tool-call rules:
- NEVER output a bare arguments object as your response text.
- NEVER omit the tool name when calling a tool.
- If you accidentally express a tool call as JSON text, it must be a single object of the form {{"name": "tool_name", "arguments": {{...}}}}."""

        return replace(base_prompts, system_prompt=_system_prompt)
