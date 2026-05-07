# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Converter Protocol and the framework-neutral image-description payload."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from haystack.dataclasses import ByteStream


@dataclass(frozen=True)
class ImageDescriptionPayload:
    """Adapter-layer payload for one described image with ingestion provenance."""

    description: str
    source: str
    image_path: Path
    image_hash: str
    base_meta: dict[str, Any]  # caller/meta boundary data
    extra_meta: dict[str, Any] | None = None


class FileConverter(Protocol):
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,  # Haystack convention
    ) -> dict[str, Any]: ...  # Haystack convention: {"documents": list[Document]}
