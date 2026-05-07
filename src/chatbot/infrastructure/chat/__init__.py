# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public chat infrastructure API.

This package exposes provider-specific construction primitives; provider
selection and model-profile dispatch live in the composition layer
(``src.chatbot.build_from_settings``), keeping this package free of
cross-cutting decision logic.
"""

from src.chatbot.contracts.chat import ModelProfile

from ._model_profile import (
    DefaultChatModelProfile,
    QwenCoderModelProfile,
    SmallModelProfile,
)
from ._ollama import build_ollama_chat_model
from ._openai_compatible import build_openai_compatible_chat_model
from ._text_tool_call_wrapper import TextToolCallParsingWrapper

__all__ = [
    "DefaultChatModelProfile",
    "ModelProfile",
    "QwenCoderModelProfile",
    "SmallModelProfile",
    "TextToolCallParsingWrapper",
    "build_ollama_chat_model",
    "build_openai_compatible_chat_model",
]
