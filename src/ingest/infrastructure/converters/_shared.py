# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adapter-internal helpers shared by converters.

The public Protocol (:class:`~src.ingest.contracts.converters.FileConverter`)
and value object (:class:`~src.ingest.contracts.converters.ImageDescriptionPayload`)
live in :mod:`src.ingest.contracts.converters`. Helpers here only depend on
those contracts.
"""

from typing import Any

from haystack.dataclasses import Document

from src.ingest.contracts.converters import ImageDescriptionPayload
from src.ingest.contracts.images import IMAGE_KIND_DESCRIPTION


def extract_language_hint(meta: dict[str, Any]) -> str | None:
    """Best-effort extraction of a language hint from sidecar metadata."""
    raw = meta.get("language") or meta.get("lang")
    if not raw:
        return None
    return str(raw).strip().lower() or None


def to_haystack_document(payload: ImageDescriptionPayload) -> Document:
    """Map framework-neutral image payloads into Haystack documents."""
    meta: dict[str, Any] = {
        **payload.base_meta,
        "source": payload.source,
        "kind": IMAGE_KIND_DESCRIPTION,
        "image_hash": payload.image_hash,
        "image_path": str(payload.image_path),
    }
    if payload.extra_meta:
        meta.update(payload.extra_meta)
    return Document(content=payload.description, meta=meta)
