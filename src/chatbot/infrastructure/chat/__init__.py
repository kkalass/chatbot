"""Public chat infrastructure API.

This package exposes provider-agnostic construction primitives while keeping
provider-specific integrations in internal modules.
"""

from dataclasses import dataclass
from typing import Literal, assert_never

from src.chatbot.app.protocols import ChatModel, ModelProfile

from ._model_profile import (
    DefaultChatModelProfile,
    QwenCoderModelProfile,
    SmallModelProfile,
)
from ._ollama import build_ollama_chat_model
from ._text_tool_call_wrapper import TextToolCallParsingWrapper


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


def build_chat_model_profile(config: ChatModelConfig) -> ModelProfile:
    """Build a model profile for the configured model.

    The profile encapsulates all model-specific adaptations: prompt tuning,
    tool-schema adjustments, and adapter-level capability flags such as
    ``parse_text_tool_calls``.  Add a case here when a particular model needs
    non-default behaviour, e.g.::

        case name if name.startswith("llama3.1"):
            return CustomLlamaModelProfile()
    """
    model_name = config.model.lower()
    match model_name:
        case name if "llama" in name:
            return SmallModelProfile()
        case name if "qwen" in name and "coder" in name:
            return QwenCoderModelProfile()
        case name if "qwen" in name:
            return SmallModelProfile()
        case _:
            return DefaultChatModelProfile()


def build_chat_model(
    config: ChatModelConfig,
    *,
    parse_text_tool_calls: bool = False,
) -> ChatModel:
    """Construct the chat model prescribed by ``config.provider``.

    Args:
        parse_text_tool_calls: Enable detection of text-encoded tool calls
            for models that don't use the native tool_calls field. Determined
            by the active ModelProfile at the call site.
    """
    match config.provider:
        case "ollama":
            model = build_ollama_chat_model(
                base_url=config.base_url,
                model=config.model,
                temperature=config.temperature,
                seed=config.seed,
            )
        case _:
            assert_never(config.provider)
    if parse_text_tool_calls:
        return TextToolCallParsingWrapper(model)
    return model


__all__ = [
    "ChatModelConfig",
    "DefaultChatModelProfile",
    "ModelProfile",
    "QwenCoderModelProfile",
    "SmallModelProfile",
    "build_chat_model",
    "build_chat_model_profile",
]
