# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image-related ingestion constants.

Constants only — implementations consuming these (the description service,
the filter config) live in :mod:`src.ingest.infrastructure.image_description`.
"""

#: Identifier for chunk content kind ``"image_description"``. Set as the
#: ``kind`` metadata field on Documents whose content is the vision-model
#: description of an image (standalone image file or PDF-embedded figure).
IMAGE_KIND_DESCRIPTION = "image_description"

#: Suffixes recognised as standalone image files at ingestion discovery time.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})
