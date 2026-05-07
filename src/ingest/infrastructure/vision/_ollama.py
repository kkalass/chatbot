# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ollama-backed vision describer."""

from typing import Any, cast

import structlog
from ollama import Client

from src.ingest.infrastructure.vision.contracts import VisionPromptBuilder

logger = structlog.get_logger(__name__)


class _OllamaVisionDescriber:
    """Vision describer backed by an Ollama vision-capable model.

    Prompt policy is injected from composition roots so this adapter remains
    transport-focused (request/response contract only).
    """

    def __init__(
        self,
        *,
        model: str,
        url: str,
        prompt_builder: VisionPromptBuilder,
    ) -> None:
        self._model = model
        self._client = Client(host=url)
        self._prompt_builder = prompt_builder

    def describe(
        self,
        image_bytes: bytes,
        *,
        language_hint: str | None = None,
    ) -> str:
        prompt = self._prompt_builder(language_hint=language_hint)
        # The ollama python client accepts ``images`` as a list of raw bytes
        # objects on user messages.
        # ollama.Client.chat is overloaded; the typed overload returns ChatResponse
        # but pyright cannot narrow it through the dynamic kwargs. The runtime
        # contract is stable.
        response: Any = self._client.chat(  # pyright: ignore[reportUnknownMemberType]
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }
            ],
            options={"temperature": 0.0},
        )
        message: dict[str, Any] = cast(dict[str, Any], response["message"])
        content_obj: object = message.get("content", "")
        return str(content_obj).strip()


def build_ollama_vision_describer(
    *,
    model: str,
    url: str,
    prompt_builder: VisionPromptBuilder,
) -> _OllamaVisionDescriber:
    """Build an Ollama-backed vision describer."""
    logger.debug("vision.ollama.build", model=model, url=url)
    return _OllamaVisionDescriber(model=model, url=url, prompt_builder=prompt_builder)
