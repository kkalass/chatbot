# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for Ollama vision describer adapter."""

from typing import Any

import pytest

import src.ingest.infrastructure.vision._ollama as ollama_vision


class _FakeClient:
    def __init__(self, *, host: str) -> None:
        self.host = host
        self.last_kwargs: dict[str, Any] | None = None

    def chat(self, **kwargs: Any) -> dict[str, object]:
        self.last_kwargs = kwargs
        return {"message": {"content": "  described  "}}


class TestOllamaVisionDescriber:
    def test_uses_injected_prompt_builder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_hint: list[str | None] = []
        fake_client = _FakeClient(host="http://localhost:11434")

        def _fake_client_factory(*, host: str) -> _FakeClient:
            assert host == "http://localhost:11434"
            return fake_client

        def _prompt_builder(*, language_hint: str | None = None) -> str:
            captured_hint.append(language_hint)
            return f"PROMPT::{language_hint}"

        monkeypatch.setattr(ollama_vision, "Client", _fake_client_factory)

        describer = ollama_vision.build_ollama_vision_describer(
            model="qwen2.5vl:7b",
            url="http://localhost:11434",
            prompt_builder=_prompt_builder,
        )

        result = describer.describe(b"img", language_hint="de")

        assert result == "described"
        assert captured_hint == ["de"]
        assert fake_client.last_kwargs is not None
        assert fake_client.last_kwargs["messages"][0]["content"] == "PROMPT::de"
