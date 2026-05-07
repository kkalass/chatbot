# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Retrieval boundary: SourceChunk value object and Retriever Protocol."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SourceChunk:
    """A single retrieved chunk of text with provenance metadata.

    Args:
        content: The raw text of the chunk.
        source: Path or identifier of the originating document (displayed in
            citations and used for deduplication).
        score: Similarity score returned by the vector store (higher is better).
        chunk_id: Unique identifier assigned during ingestion (opaque string).
        title: Human-readable document title from sidecar metadata (optional).
        author: Author(s) from sidecar metadata (optional).
        publication_date: Publication date string from sidecar metadata (optional).
        source_url: Canonical URL of the original document (optional).
        page: Page label (PDF page number as string) when applicable.
        kind: Chunk content kind. ``"text"`` for plain text chunks (default);
            ``"image_description"`` for chunks whose content is the
            vision-model description of an image (standalone image file or
            PDF-embedded figure).
        image_path: On-disk path to the source image when ``kind
            == "image_description"``. The UI surfaces this as a viewable
            artefact alongside the description.
    """

    content: str
    source: str
    score: float
    chunk_id: str
    title: str | None = None
    author: str | None = None
    publication_date: str | None = None
    source_url: str | None = None
    page: str | None = None
    kind: str = "text"
    image_path: str | None = None


class Retriever(Protocol):
    """Structural boundary between orchestration and retrieval infrastructure."""

    async def retrieve(
        self,
        query_dense: str,
        *,
        query_sparse: str | None = None,
    ) -> list[SourceChunk]:
        """Return ranked, score-filtered chunks relevant to the retrieval queries.

        *query_dense* is used for embedding-based retrieval. If *query_sparse* is
        omitted, implementations should fall back to *query_dense* for sparse retrieval.
        """
        ...
