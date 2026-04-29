"""Public chat infrastructure API.

This package exposes provider-agnostic construction primitives while keeping
provider-specific integrations in internal modules.
"""

from dataclasses import dataclass
from typing import Literal, assert_never

from src.chatbot.app.protocols import ChatModel, PromptProfile

from ._inline_quotes import build_inline_quote_parsing_chat_model
from ._ollama import build_ollama_chat_model
from ._prompt_profile import DefaultChatPromptProfile, SmallModelPromptProfile


@dataclass(frozen=True)
class ChatModelConfig:
    """Construction-time config for the chat model adapter.

    Args:
        base_url: Base URL of the provider server.
        model: Model identifier used for chat generation.
        temperature: Optional temperature override passed to the provider.
        seed: Optional deterministic seed passed to the provider.
        provider: Chat provider backend identifier.
    """

    base_url: str
    model: str
    temperature: float | None = None
    seed: int | None = None
    provider: Literal["ollama"] = "ollama"


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
        case _:
            return DefaultChatPromptProfile()


def build_chat_model(
    config: ChatModelConfig,
    *,
    inline_quotes_enabled: bool = False,
) -> ChatModel:
    """Construct the chat model prescribed by ``config.provider``."""
    match config.provider:
        case "ollama":
            model = build_ollama_chat_model(
                base_url=config.base_url,
                model=config.model,
                temperature=config.temperature,
                seed=config.seed,
            )
            if inline_quotes_enabled:
                return build_inline_quote_parsing_chat_model(model)
            return model
        case _:
            assert_never(config.provider)


__all__ = [
    "ChatModelConfig",
    "DefaultChatPromptProfile",
    "PromptProfile",
    "SmallModelPromptProfile",
    "build_chat_model",
    "build_chat_prompt_profile",
]
