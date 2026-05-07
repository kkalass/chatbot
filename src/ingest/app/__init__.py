# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ingestion application/orchestration modules."""

from ._pipeline import FormatHandler, IngestionConfig, IngestionPipeline, load_sidecar_meta
from ._vision_prompts import build_image_description_prompt

__all__ = [
    "FormatHandler",
    "IngestionConfig",
    "IngestionPipeline",
    "build_image_description_prompt",
    "load_sidecar_meta",
]
