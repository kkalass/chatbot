# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ingestion vision-prompt policy."""

from src.ingest.app import build_image_description_prompt


class TestBuildImageDescriptionPrompt:
    def test_defaults_to_english_without_hint(self) -> None:
        prompt = build_image_description_prompt(language_hint=None)

        assert "Write the description in English." in prompt
        assert "three short paragraphs" in prompt

    def test_uses_german_prompt_for_de_language_hint(self) -> None:
        prompt = build_image_description_prompt(language_hint="de")

        assert "Schreibe die Beschreibung auf Deutsch." in prompt
        assert "drei kurzen Absaetzen" in prompt

    def test_uses_german_prompt_for_regional_de_hint(self) -> None:
        prompt = build_image_description_prompt(language_hint="de-DE")

        assert "Schreibe die Beschreibung auf Deutsch." in prompt

    def test_falls_back_to_english_for_non_de_hint(self) -> None:
        prompt = build_image_description_prompt(language_hint="fr")

        assert "Write the description in English." in prompt
