"""Citation rendering helpers for the Chainlit UI.

Keeps the primary citation label compact while enriching the side panel content
with optional metadata when available.
"""

from src.chatbot.app.protocols import SourceChunk


def build_citation_name(chunk: SourceChunk) -> str:
    """Return a compact citation element name shown in the side panel list."""
    label = chunk.title or chunk.source
    return f"{label} [{chunk.chunk_id}]"


def build_citation_content(chunk: SourceChunk) -> str:
    """Return citation side-panel content with optional metadata and chunk text."""
    lines: list[str] = []

    if chunk.author:
        lines.append(f"Author: {chunk.author}")
    if chunk.publication_date:
        lines.append(f"Publication date: {chunk.publication_date}")
    if chunk.source_url:
        lines.append(f"Source link: {chunk.source_url}")
    lines.append(f"Source file: {chunk.source}")

    return "\n".join([*lines, "", chunk.content])
