"""Default prompt profile implementation.

The prompt-profile interface lives in ``src.chatbot.app.protocols.PromptProfile``.
This module provides the default identity implementation used by models that
require no prompt-level tuning.
"""

from dataclasses import dataclass

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
