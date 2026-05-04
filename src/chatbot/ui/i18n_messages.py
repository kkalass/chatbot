# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Internationalisation: translation map and message resolver for the Chainlit UI.

Each tool defines its message keys as a ``StrEnum`` subclass.  This module maps
every key to a localised format-string template and exposes
:func:`resolve_message` as the single resolution point for the UI.

Adding a new tool requires:
1. Defining a ``StrEnum`` subclass in the tool module with one value per
   ``I18nMessage`` key the tool emits.
2. Adding one entry per key **and language** to :data:`TRANSLATIONS`.
"""

from src.chatbot.app.protocols import I18nMessage
from src.chatbot.tools.retrieval.tool import RetrievalCallKey
from src.chatbot.tools.vacation_days.keys import VacationDaysCallKey

# ---------------------------------------------------------------------------
# Translation map
# ---------------------------------------------------------------------------
# Outer key: BCP 47 primary language subtag ("en", "de", …).
# Inner key:  I18nMessage.key — a StrEnum value (plain string at runtime).
# Value:      Python str.format-compatible template; named slots map to
#             I18nMessage.args keys.
TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        RetrievalCallKey.DISPLAY_NAME: "Document Search",
        RetrievalCallKey.SEARCHING: "Searching for: {query}",
        VacationDaysCallKey.DISPLAY_NAME: "Vacation Days Service",
        VacationDaysCallKey.QUERYING: "Querying vacation days for {year}",
    },
    "de": {
        RetrievalCallKey.DISPLAY_NAME: "Dokumentensuche",
        RetrievalCallKey.SEARCHING: "Suche nach: {query}",
        VacationDaysCallKey.DISPLAY_NAME: "Urlaubstage-Dienst",
        VacationDaysCallKey.QUERYING: "Urlaubstage für {year} abfragen",
    },
}

#: Supported primary language tags — derived from :data:`TRANSLATIONS` keys.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(TRANSLATIONS)

_FALLBACK_LANG = "en"


def detect_language(accept_language: str) -> str:
    """Extract the primary language tag from an ``Accept-Language`` header value.

    Parses ``"de-DE,de;q=0.9,en-US;q=0.8"`` → ``"de"``.  Returns ``"en"``
    for unsupported primary tags or unparseable input.

    Args:
        accept_language: Raw ``Accept-Language`` header value.

    Returns:
        A supported primary language tag, or ``"en"`` as fallback.
    """
    primary = accept_language.split(",")[0].split(";")[0].split("-")[0].strip().lower()
    return primary if primary in SUPPORTED_LANGUAGES else _FALLBACK_LANG


def resolve_message(msg: I18nMessage, *, lang: str = _FALLBACK_LANG) -> str:
    """Resolve an :class:`~src.chatbot.app.protocols.I18nMessage` to a display string.

    Looks up the template in ``TRANSLATIONS[lang]``, falling back to the
    ``"en"`` map for unsupported languages.  Returns the raw ``msg.key`` when
    no template exists so that missing translations are always visible rather
    than silently suppressed.

    Args:
        msg: The message to resolve.
        lang: BCP 47 primary language tag (e.g. ``"en"``, ``"de"``).

    Returns:
        The formatted string, or the unformatted template if a required arg is
        missing, or the raw key if no template exists.
    """
    lang_map = TRANSLATIONS.get(lang) or TRANSLATIONS.get(_FALLBACK_LANG, {})
    template = lang_map.get(msg.key)
    if template is None:
        return msg.key
    try:
        return template.format(**msg.args)
    except KeyError:
        return template
