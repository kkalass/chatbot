# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`src.chatbot.ui.i18n_messages`."""

import pytest

from src.chatbot.app.protocols import I18nMessage
from src.chatbot.tools.retrieval.tool import RetrievalCallKey
from src.chatbot.tools.vacation_days.keys import VacationDaysCallKey
from src.chatbot.ui.i18n_messages import (
    SUPPORTED_LANGUAGES,
    TRANSLATIONS,
    detect_language,
    resolve_message,
)


class TestTranslationCoverage:
    """Every StrEnum key defined by a tool must have an entry for every supported language."""

    @pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGUAGES))
    @pytest.mark.parametrize("key", list(RetrievalCallKey))
    def test_retrieval_keys_covered(self, lang: str, key: RetrievalCallKey) -> None:
        assert key in TRANSLATIONS[lang], f"Missing translation for {key!r} in lang={lang!r}"

    @pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGUAGES))
    @pytest.mark.parametrize("key", list(VacationDaysCallKey))
    def test_vacation_days_keys_covered(self, lang: str, key: VacationDaysCallKey) -> None:
        assert key in TRANSLATIONS[lang], f"Missing translation for {key!r} in lang={lang!r}"


class TestDetectLanguage:
    def test_exact_primary_tag(self) -> None:
        assert detect_language("en") == "en"

    def test_german_with_region(self) -> None:
        assert detect_language("de-DE") == "de"

    def test_weighted_list_uses_first(self) -> None:
        assert detect_language("de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7") == "de"

    def test_english_weighted_list(self) -> None:
        assert detect_language("en-US,en;q=0.9") == "en"

    def test_unsupported_lang_falls_back_to_en(self) -> None:
        assert detect_language("fr-FR,fr;q=0.9") == "en"

    def test_empty_falls_back_to_en(self) -> None:
        assert detect_language("") == "en"


class TestResolveMessage:
    def test_known_key_formats_args_english(self) -> None:
        msg = I18nMessage(key=RetrievalCallKey.SEARCHING, args={"query": "AI trends"})
        assert resolve_message(msg, lang="en") == "Searching for: AI trends"

    def test_known_key_formats_args_german(self) -> None:
        msg = I18nMessage(key=RetrievalCallKey.SEARCHING, args={"query": "KI-Trends"})
        assert resolve_message(msg, lang="de") == "Suche nach: KI-Trends"

    def test_vacation_days_english(self) -> None:
        msg = I18nMessage(key=VacationDaysCallKey.QUERYING, args={"year": "2026"})
        assert resolve_message(msg, lang="en") == "Querying vacation days for 2026"

    def test_vacation_days_german(self) -> None:
        msg = I18nMessage(key=VacationDaysCallKey.QUERYING, args={"year": "2026"})
        assert resolve_message(msg, lang="de") == "Urlaubstage für 2026 abfragen"

    def test_display_name_english(self) -> None:
        msg = I18nMessage(key=RetrievalCallKey.DISPLAY_NAME, args={})
        assert resolve_message(msg, lang="en") == "Document Search"

    def test_display_name_german(self) -> None:
        msg = I18nMessage(key=RetrievalCallKey.DISPLAY_NAME, args={})
        assert resolve_message(msg, lang="de") == "Dokumentensuche"

    def test_default_lang_is_english(self) -> None:
        msg = I18nMessage(key=RetrievalCallKey.DISPLAY_NAME, args={})
        assert resolve_message(msg) == "Document Search"

    def test_unsupported_lang_falls_back_to_english(self) -> None:
        msg = I18nMessage(key=VacationDaysCallKey.DISPLAY_NAME, args={})
        assert resolve_message(msg, lang="fr") == "Vacation Days Service"

    def test_unknown_key_returns_key_string(self) -> None:
        msg = I18nMessage(key="some.unknown.key", args={})
        assert resolve_message(msg) == "some.unknown.key"

    def test_missing_arg_returns_unformatted_template(self) -> None:
        msg = I18nMessage(key=RetrievalCallKey.SEARCHING, args={})  # query arg missing
        assert resolve_message(msg) == TRANSLATIONS["en"][RetrievalCallKey.SEARCHING]
