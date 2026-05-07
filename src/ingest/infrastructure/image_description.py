# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image-to-description orchestration used during ingestion.

Composes the size/dimension filter, the cache, and the vision describer into
a single :class:`ImageDescriptionService` consumed by the ingestion pipeline.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import structlog
from PIL import Image

from src.ingest.infrastructure.image_cache import (
    ImageDescriptionCache,
    compute_image_hash,
)
from src.ingest.infrastructure.vision import VisionDescriber

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ImageDescriptionResult:
    """Outcome of describing one image."""

    description: str
    image_hash: str


@dataclass(frozen=True)
class ImageFilterConfig:
    """Filter thresholds applied before/after vision-model calls."""

    min_dimension: int
    min_description_length: int


def _image_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    """Return ``(width, height)`` for *image_bytes* or ``None`` on parse failure."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.size
    except Exception as exc:
        logger.debug("image_pipeline.dimensions_failed", error=str(exc))
        return None


class ImageDescriptionService:
    """Cache-aware vision-description orchestrator.

    Responsibilities:
    - Compute content hash once per image.
    - Short-circuit on cache hit.
    - Filter trivially small images before any vision-model call.
    - Invoke the vision describer.
    - Apply a length floor to drop uninformative descriptions (decorative
      logos, dividers).
    - Persist the description in the cache on success.

    Args:
        describer: Vision-model describer.
        cache: Description cache keyed by image content hash.
        filter_config: Dimension and length thresholds.
    """

    def __init__(
        self,
        *,
        describer: VisionDescriber,
        cache: ImageDescriptionCache,
        filter_config: ImageFilterConfig,
    ) -> None:
        self._describer = describer
        self._cache = cache
        self._filter = filter_config
        self._vision_calls = 0

    @property
    def vision_call_count(self) -> int:
        """Total number of vision-model invocations since construction.

        Useful for tests asserting cache effectiveness.
        """
        return self._vision_calls

    def describe(
        self,
        image_bytes: bytes,
        *,
        language_hint: str | None = None,
    ) -> ImageDescriptionResult | None:
        """Return a description for *image_bytes* or ``None`` if filtered out."""
        image_hash = compute_image_hash(image_bytes)
        cached = self._cache.get(image_hash)
        if cached is not None:
            logger.debug("image_pipeline.cache_hit", hash=image_hash)
            # Cached descriptions have already passed the length filter when
            # they were originally written — do not re-filter here so a
            # threshold change does not silently invalidate prior work.
            return ImageDescriptionResult(description=cached, image_hash=image_hash)

        dims = _image_dimensions(image_bytes)
        if dims is None:
            logger.info("image_pipeline.skipping_unparseable", hash=image_hash)
            return None
        width, height = dims
        if width < self._filter.min_dimension or height < self._filter.min_dimension:
            logger.debug(
                "image_pipeline.skipping_too_small",
                hash=image_hash,
                width=width,
                height=height,
                min_dimension=self._filter.min_dimension,
            )
            return None

        try:
            description = self._describer.describe(image_bytes, language_hint=language_hint)
        except Exception as exc:
            logger.warning(
                "image_pipeline.vision_call_failed",
                hash=image_hash,
                error=str(exc),
            )
            return None
        self._vision_calls += 1

        normalized = description.strip()
        if len(normalized) < self._filter.min_description_length:
            logger.debug(
                "image_pipeline.skipping_short_description",
                hash=image_hash,
                length=len(normalized),
                min_length=self._filter.min_description_length,
            )
            return None

        self._cache.put(image_hash, normalized)
        logger.info(
            "image_pipeline.described",
            hash=image_hash,
            description_length=len(normalized),
        )
        return ImageDescriptionResult(description=normalized, image_hash=image_hash)


__all__ = [
    "ImageDescriptionResult",
    "ImageDescriptionService",
    "ImageFilterConfig",
]
