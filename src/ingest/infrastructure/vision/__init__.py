# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public vision-model infrastructure API and factory."""

from ._ollama import build_ollama_vision_describer
from .contracts import VisionDescriber, VisionPromptBuilder

__all__ = [
    "VisionDescriber",
    "VisionPromptBuilder",
    "build_ollama_vision_describer",
]
