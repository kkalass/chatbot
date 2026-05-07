# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt policy for image-description generation during ingestion.

This module intentionally lives outside provider adapters. Prompt wording is
domain policy, while transport and model invocation belong to infrastructure
adapters (e.g. Ollama HTTP contract).

Why not English-only prompts?
- Smaller vision models often follow output-language constraints more reliably
  when the instruction language matches the expected output language.
- For mixed corpora (e.g. German and English documents), language-matching
  prompts improve retrieval quality because generated descriptions stay close
  to document language and terminology.
"""

_DESCRIPTION_PROMPT_EN = """You are an image-description assistant for a retrieval system.

Produce a plain-text description of the supplied image as exactly three short paragraphs:

1. WHAT IT IS — chart type, diagram type, photo, screenshot, or table.
2. WHAT IT SHOWS — axes, categories, key entities, dominant data points, headings, or visible labels. Reproduce numeric values and percentages verbatim where present.
3. WHAT IT CLAIMS — the takeaway a careful reader would extract.

Rules:
- Plain text, no JSON, no markdown headings.
- Write the description in English.
- Do not invent numbers or labels you cannot read. If uncertain, say so explicitly.
- Be concise. The whole description should fit in roughly 200 words.
"""

_DESCRIPTION_PROMPT_DE = """Du bist ein Assistent zur Bildbeschreibung fuer ein Retrieval-System.

Erstelle eine reine Textbeschreibung des Bildes in genau drei kurzen Absaetzen:

1. WAS ES IST — Art des Diagramms, Skizze, Foto, Screenshot oder Tabelle.
2. WAS ES ZEIGT — Achsen, Kategorien, zentrale Entitaeten, dominante Datenpunkte, Ueberschriften oder sichtbare Beschriftungen. Uebernimm Zahlenwerte und Prozentangaben woertlich, wenn vorhanden.
3. WAS ES AUSSAGT — die Kernaussage, die ein aufmerksamer Leser ableiten wuerde.

Regeln:
- Reiner Text, kein JSON, keine Markdown-Ueberschriften.
- Schreibe die Beschreibung auf Deutsch.
- Keine erfundenen Zahlen oder Beschriftungen. Bei Unsicherheit explizit kennzeichnen.
- Knapp halten. Die gesamte Beschreibung soll etwa 200 Woerter umfassen.
"""


def build_image_description_prompt(*, language_hint: str | None = None) -> str:
    """Return the vision prompt selected by the document language hint.

    Args:
        language_hint: Optional BCP 47 language tag from source metadata.
            Values starting with ``"de"`` select the German prompt. All other
            values (or ``None``) fall back to English.
    """
    normalized = (language_hint or "").strip().lower()
    if normalized.startswith("de"):
        return _DESCRIPTION_PROMPT_DE
    return _DESCRIPTION_PROMPT_EN


__all__ = ["build_image_description_prompt"]
