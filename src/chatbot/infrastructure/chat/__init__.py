"""Public chat infrastructure API.

This package exposes provider-agnostic construction primitives while keeping
provider-specific integrations in internal modules.
"""

from dataclasses import dataclass
from typing import Literal, assert_never

from src.chatbot.app.protocols import ChatModel, PromptProfile

from ._ollama import build_ollama_chat_model
from ._prompt_profile import (
    DefaultChatPromptProfile,
    QwenCoderPromptProfile,
    SmallModelPromptProfile,
)


@dataclass(frozen=True)
class ChatModelConfig:
    """Construction-time config for the chat model adapter.

    Args:
        base_url: Base URL of the provider server.
        model: Model identifier used for chat generation.
        temperature: Optional temperature override passed to the provider.
        seed: Optional deterministic seed passed to the provider.
        provider: Chat provider backend identifier.
        parse_text_tool_calls: Enable text-encoded tool call detection.
            Set to ``True`` only for models that serialise tool calls as JSON
            in their response text (e.g. qwen2.5-coder). Off by default.
    """

    base_url: str
    model: str
    temperature: float | None = None
    seed: int | None = None
    provider: Literal["ollama"] = "ollama"
    parse_text_tool_calls: bool = False


def build_chat_prompt_profile(config: ChatModelConfig) -> PromptProfile:
    """Build a prompt-adaptation profile for the configured model.

    Prompt behaviour is model-specific (not provider-specific): two different
    models on the same Ollama server may require very different tuning.  Add a
    case here when a particular model needs it, e.g.:

        case model if model.startswith("llama3.1"):
            return Llama31PromptProfile()
    """
    model_name = config.model.lower()
    match model_name:
        case name if "llama" in name:
            return SmallModelPromptProfile()
        case name if "qwen" in name and "coder" in name:
            return QwenCoderPromptProfile()
        case name if "qwen" in name:
            return SmallModelPromptProfile()
        case _:
            return DefaultChatPromptProfile()


def build_chat_model(
    config: ChatModelConfig,
) -> ChatModel:
    """Construct the chat model prescribed by ``config.provider``."""
    match config.provider:
        case "ollama":
            return build_ollama_chat_model(
                base_url=config.base_url,
                model=config.model,
                temperature=config.temperature,
                seed=config.seed,
                parse_text_tool_calls=config.parse_text_tool_calls,
            )
        case _:
            assert_never(config.provider)


__all__ = [
    "ChatModelConfig",
    "DefaultChatPromptProfile",
    "PromptProfile",
    "QwenCoderPromptProfile",
    "SmallModelPromptProfile",
    "build_chat_model",
    "build_chat_prompt_profile",
]
