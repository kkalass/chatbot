# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PDF converter and PDF image processing helpers."""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream, Document
from pypdf import PdfReader

from src.ingest.contracts.converters import ImageDescriptionPayload
from src.ingest.infrastructure.image_cache import ExtractedImageStore, compute_image_hash
from src.ingest.infrastructure.image_description import ImageDescriptionService

from ._shared import extract_language_hint, to_haystack_document

logger = structlog.get_logger(__name__)


def _extract_page_images(
    page: Any,  # pypdf.PageObject has no stable public type
) -> list[tuple[int, str, bytes]]:
    """Return per-page image triples ``(image_index, suffix, bytes)``."""
    extracted: list[tuple[int, str, bytes]] = []
    try:
        images_iter = list(page.images)
    except Exception as exc:
        logger.debug("pdf_converter.image_iteration_failed", error=str(exc))
        return extracted
    for idx, image_file in enumerate(images_iter):
        try:
            data: bytes = image_file.data
            name: str = image_file.name or f"image_{idx}"
        except Exception as exc:
            logger.debug("pdf_converter.image_load_failed", image_index=idx, error=str(exc))
            continue
        suffix = Path(name).suffix.lower() or ".png"
        extracted.append((idx, suffix, data))
    return extracted


@dataclass(frozen=True)
class PdfImageProcessor:
    """Adapter helper that extracts, describes, and persists embedded PDF images."""

    service: ImageDescriptionService
    store: ExtractedImageStore

    def emit_for_page(
        self,
        *,
        pdf_source: str,
        pdf_hash: str,
        page_number: int,
        page_images: list[tuple[int, str, bytes]],
        base_meta: dict[str, Any],
        seen_hashes: set[str],
        language_hint: str | None,
    ) -> list[ImageDescriptionPayload]:
        """Return image-description payloads for one PDF page."""
        payloads: list[ImageDescriptionPayload] = []
        for image_index, suffix, image_bytes in page_images:
            image_hash = compute_image_hash(image_bytes)
            if image_hash in seen_hashes:
                logger.debug(
                    "image_pipeline.dedup_within_pdf",
                    pdf=pdf_source,
                    page=page_number,
                    image_index=image_index,
                    hash=image_hash,
                )
                continue
            result = self.service.describe(image_bytes, language_hint=language_hint)
            if result is None:
                continue
            seen_hashes.add(result.image_hash)
            stored_path = self.store.store(
                pdf_hash=pdf_hash,
                page=page_number,
                image_index=image_index,
                image_bytes=image_bytes,
                suffix=suffix,
            )
            payloads.append(
                ImageDescriptionPayload(
                    description=result.description,
                    source=pdf_source,
                    image_path=stored_path,
                    image_hash=result.image_hash,
                    base_meta=base_meta,
                    extra_meta={
                        "page": str(page_number),
                        "image_index": str(image_index),
                    },
                )
            )
        return payloads


class PdfPageConverter:
    """Extraction-first PDF converter: extracts text per page and optional image docs."""

    def __init__(
        self,
        *,
        image_service: ImageDescriptionService | None = None,
        extracted_image_store: ExtractedImageStore | None = None,
    ) -> None:
        self._image_processor: PdfImageProcessor | None = None
        if image_service is not None and extracted_image_store is not None:
            self._image_processor = PdfImageProcessor(
                service=image_service,
                store=extracted_image_store,
            )

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
                    "pdf_converter.source_load_failed", source=str(source), error=str(exc)
                )
                continue

            base_meta: dict[str, Any] = {**bytestream.meta, **caller_meta}  # Haystack meta merge
            language_hint = extract_language_hint(base_meta)
            pdf_hash = hashlib.sha256(bytestream.data).hexdigest()
            seen_image_hashes: set[str] = set()

            try:
                reader = PdfReader(io.BytesIO(bytestream.data))
                total_pages = len(reader.pages)
                for page_idx, page in enumerate(reader.pages):
                    page_number = page_idx + 1
                    text = page.extract_text() or ""
                    if text.strip():
                        page_meta: dict[str, Any] = {
                            **base_meta,
                            "page": str(page_number),
                            "total_pages": str(total_pages),
                        }
                        documents.append(Document(content=text, meta=page_meta))
                    else:
                        logger.debug(
                            "pdf_converter.skipping_empty_page",
                            source=str(source),
                            page=page_number,
                        )

                    if self._image_processor is not None:
                        page_images = _extract_page_images(page)
                        if page_images:
                            image_payloads = self._image_processor.emit_for_page(
                                pdf_source=str(source),
                                pdf_hash=pdf_hash,
                                page_number=page_number,
                                page_images=page_images,
                                base_meta={
                                    **base_meta,
                                    "total_pages": str(total_pages),
                                },
                                seen_hashes=seen_image_hashes,
                                language_hint=language_hint,
                            )
                            documents.extend(
                                to_haystack_document(payload) for payload in image_payloads
                            )
            except Exception as exc:
                logger.warning(
                    "pdf_converter.extraction_failed", source=str(source), error=str(exc)
                )

        return {"documents": documents}
