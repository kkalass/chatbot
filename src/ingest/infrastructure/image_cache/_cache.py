# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Filesystem-keyed cache for vision-model image descriptions.

Keying is by SHA-256 of the image bytes. Cache files are plain UTF-8 text;
the cache directory is shared across reindex runs so unchanged images cost
zero vision-model calls.
"""

import hashlib
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def compute_image_hash(image_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of *image_bytes*."""
    return hashlib.sha256(image_bytes).hexdigest()


class ImageDescriptionCache:
    """On-disk cache for textual image descriptions, keyed by content hash.

    The cache is intentionally single-file-per-key with no index: this keeps
    the cache directory inspectable, lock-free, and trivial to invalidate by
    deletion. Concurrent writes to the same key are safe — the file content
    is identical when produced for the same hash.

    Args:
        cache_dir: Root directory in which description files live. Created
            on construction if missing.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, image_hash: str) -> Path:
        return self._cache_dir / f"{image_hash}.txt"

    def get(self, image_hash: str) -> str | None:
        """Return the cached description for *image_hash* or ``None`` on miss."""
        path = self._path_for(image_hash)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "image_cache.read_failed",
                hash=image_hash,
                path=str(path),
                error=str(exc),
            )
            return None

    def put(self, image_hash: str, description: str) -> None:
        """Persist *description* under *image_hash*. Overwrites silently."""
        path = self._path_for(image_hash)
        try:
            path.write_text(description, encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "image_cache.write_failed",
                hash=image_hash,
                path=str(path),
                error=str(exc),
            )


class ExtractedImageStore:
    """Persists PDF-extracted image bytes under a stable on-disk path.

    The on-disk path is what the citation layer surfaces to the UI, so it
    must outlive the ingestion run. Files are organised by source-PDF hash
    to keep deletion of one source's artefacts trivial.

    Args:
        root_dir: Root directory under which extracted images are written.
    """

    def __init__(self, root_dir: Path) -> None:
        self._root = root_dir
        self._root.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        *,
        pdf_hash: str,
        page: int,
        image_index: int,
        image_bytes: bytes,
        suffix: str,
    ) -> Path:
        """Write *image_bytes* to a deterministic path and return it.

        Args:
            pdf_hash: Content hash of the source PDF (groups extractions per
                document).
            page: 1-based PDF page number.
            image_index: Index of the image within the page.
            image_bytes: Raw image bytes to persist.
            suffix: File suffix including the leading dot (e.g. ``".png"``).
        """
        normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        target_dir = self._root / pdf_hash
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"page{page:04d}_img{image_index:03d}{normalized_suffix}"
        if not path.exists():
            try:
                path.write_bytes(image_bytes)
            except OSError as exc:
                logger.warning(
                    "extracted_image_store.write_failed",
                    path=str(path),
                    error=str(exc),
                )
        return path
