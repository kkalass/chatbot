# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Standalone image-file converter."""

from pathlib import Path
from typing import Any

import structlog
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream, Document

from src.ingest.contracts.converters import ImageDescriptionPayload
from src.ingest.infrastructure.image_description import ImageDescriptionService

from ._shared import extract_language_hint, to_haystack_document

logger = structlog.get_logger(__name__)


class ImageFileConverter:
    """Standalone-image converter producing image-description :class:`Document`s."""

    def __init__(self, service: ImageDescriptionService) -> None:
        self._service = service

    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,  # Haystack convention
    ) -> dict[str, Any]:  # Haystack convention: {"documents": list[Document]}
        meta_list: list[dict[str, Any]] = normalize_metadata(meta, sources_count=len(sources))

        documents: list[Document] = []
        for source, caller_meta in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as exc:
                logger.warning(
                    "image_converter.source_load_failed",
                    source=str(source),
                    error=str(exc),
                )
                continue

            base_meta: dict[str, Any] = {**bytestream.meta, **caller_meta}  # Haystack meta merge
            language_hint = extract_language_hint(base_meta)
            result = self._service.describe(bytestream.data, language_hint=language_hint)
            if result is None:
                logger.info("image_converter.skipped", source=str(source))
                continue

            source_str = str(source)
            payload = ImageDescriptionPayload(
                description=result.description,
                source=source_str,
                image_path=Path(source_str),
                image_hash=result.image_hash,
                base_meta=base_meta,
            )
            documents.append(to_haystack_document(payload))

        return {"documents": documents}
