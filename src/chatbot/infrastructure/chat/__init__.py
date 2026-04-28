"""Public chat infrastructure API.

This package exposes provider-agnostic construction primitives while keeping
provider-specific integrations in internal modules.
"""

from dataclasses import dataclass
from typing import Literal, assert_never

from src.chatbot.app.protocols import ChatModel, PromptProfile

from ._ollama import build_ollama_chat_model
from ._prompt_profile import DefaultChatPromptProfile


@dataclass(frozen=True)
class ChatModelConfig:
    """Construction-time config for the chat model adapter.

    Args:
        base_url: Base URL of the provider server.
        model: Model identifier used for chat generation.
        provider: Chat provider backend identifier.
    """

    base_url: str
    model: str
    provider: Literal["ollama"] = "ollama"


def build_chat_prompt_profile(config: ChatModelConfig) -> PromptProfile:
    """Build a prompt-adaptation profile for the configured model.

    Prompt behaviour is model-specific (not provider-specific): two different
    models on the same Ollama server may require very different tuning.  Add a
    case here when a particular model needs it, e.g.:

        case model if model.startswith("llama3.1"):
            return Llama31PromptProfile()
    """
    match config.model:
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
            )
        case _:
            assert_never(config.provider)


__all__ = [
    "ChatModelConfig",
    "DefaultChatPromptProfile",
    "PromptProfile",
    "build_chat_model",
    "build_chat_prompt_profile",
]
