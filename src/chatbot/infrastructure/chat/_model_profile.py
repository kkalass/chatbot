"""Model profile implementations.

The model-profile interface lives in ``src.chatbot.app.protocols.ModelProfile``.
This module provides the concrete profiles used for model-specific adaptation
of prompts, tool schemas, and adapter-level capabilities.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime

from src.chatbot.app.prompts import Prompts
from src.chatbot.app.protocols import JsonObject, ModelProfile


@dataclass(frozen=True)
class DefaultChatModelProfile(ModelProfile):
    """Identity model profile — passes all app-level prompts and schemas through unchanged.

    Used as the baseline for models that do not require any tuning.
    """

    _parse_text_tool_calls: bool = field(default=False, repr=False)

    def __init__(self, parse_text_tool_calls: bool = False) -> None:
        object.__setattr__(self, "_parse_text_tool_calls", parse_text_tool_calls)

    @property
    def parse_text_tool_calls(self) -> bool:
        return self._parse_text_tool_calls

    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        return prompts

    def adjust_tool_description(self, tool_name: str, description: str) -> str:
        return description

    def adjust_parameter_schema(self, tool_name: str, schema: JsonObject) -> JsonObject:
        return schema


@dataclass(frozen=True)
class SmallModelProfile(DefaultChatModelProfile):
    """Model profile with stricter tool-calling guidance for weaker tool-call models."""

    def __init__(self, parse_text_tool_calls: bool = False) -> None:
        super().__init__(parse_text_tool_calls=parse_text_tool_calls)

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
class QwenCoderModelProfile(SmallModelProfile):
    """Model profile for qwen-coder variants that leak tool-call JSON into text.

    These models often serialise only the arguments object or an incomplete JSON
    fragment instead of using the native tool-calling channel. The prompt must
    make the required call shape explicit, and the adapter must parse text-encoded
    tool calls from the response stream.
    """

    def __init__(self) -> None:
        super().__init__(parse_text_tool_calls=True)

    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        base_prompts = super().adjust_prompts(prompts)

        def _system_prompt(now: datetime) -> str:
            return f"""{base_prompts.system_prompt(now)}

Qwen-coder specific tool-call rules:
- NEVER output a bare arguments object as your response text.
- NEVER omit the tool name when calling a tool.
- If you accidentally express a tool call as JSON text, it must be a single object of the form {{"name": "tool_name", "arguments": {{...}}}}."""

        return replace(base_prompts, system_prompt=_system_prompt)
