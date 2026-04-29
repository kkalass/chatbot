"""Default prompt profile implementation.

The prompt-profile interface lives in ``src.chatbot.app.protocols.PromptProfile``.
This module provides the default identity implementation used by models that
require no prompt-level tuning.
"""

from dataclasses import dataclass
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

Inline citation JSON rules (when emitting quote markers):
- The JSON inside markers must be a single strict object with no extra fields.
- Copy tool_call_id, source, and chunk_id values exactly as they
  appear in tool results — do not paraphrase or abbreviate them."""

        return Prompts(system_prompt=_system_prompt)
