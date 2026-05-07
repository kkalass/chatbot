# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contracts for vision-model ingestion infrastructure.

This module intentionally contains only provider-agnostic types used across
vision infrastructure adapters and composition roots.
"""

from typing import Protocol, runtime_checkable


class VisionPromptBuilder(Protocol):
    """Build a provider-agnostic prompt text from a document language hint."""

    def __call__(self, *, language_hint: str | None = None) -> str:
        """Return prompt text for the supplied language hint."""
        ...


@runtime_checkable
class VisionDescriber(Protocol):
    """Structural boundary for vision-model description generation.

    Implementations must be deterministic with respect to ``image_bytes`` only
    in expectation (vision-model output varies); upstream callers MUST cache
    descriptions keyed by image content hash so reindexing of unchanged
    images costs zero vision-model calls.
    """

    def describe(
        self,
        image_bytes: bytes,
        *,
        language_hint: str | None = None,
    ) -> str:
        """Return a textual description of the supplied image.

        Args:
            image_bytes: Raw image bytes (any format the underlying model
                accepts; PNG/JPEG are the canonical choices).
            language_hint: BCP 47 primary language tag (e.g. ``"de"``,
                ``"en"``) used to nudge the model towards the surrounding
                document's language. ``None`` falls back to English.
        """
        ...


__all__ = ["VisionDescriber", "VisionPromptBuilder"]
