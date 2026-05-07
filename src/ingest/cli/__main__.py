# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Developer CLI for corpus management operations.

Usage (from repository root):

    uv run python -m src.ingest.cli reindex            # ingest all corpus files
    uv run python -m src.ingest.cli reset              # wipe collection and re-ingest
    uv run python -m src.ingest.cli reset --wipe-only  # wipe without re-ingesting

These commands are the FR-06 developer-operations workflows. They read
configuration from the same ``.env`` / environment variables used by the
application; all infrastructure construction happens in
:mod:`src.ingest.build_from_settings`.
"""

import argparse
import sys
from pathlib import Path

import structlog

from src.ingest.build_from_settings import (
    build_ingestion_pipeline,
)
from src.shared.observability.logging import configure_logging
from src.shared.qdrant import build_qdrant_document_store
from src.shared.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


def cmd_reindex(corpus_path: Path, settings: Settings) -> int:
    """Ingest (or re-ingest) all supported files under *corpus_path*."""
    logger.info("cli.reindex.start", corpus=str(corpus_path))
    pipeline = build_ingestion_pipeline(settings)
    try:
        count = pipeline.ingest_corpus(corpus_path)
        logger.info("cli.reindex.done", chunks_written=count)
        return 0
    except Exception:
        logger.exception("cli.reindex.failed")
        return 1


def cmd_reset(
    corpus_path: Path,
    settings: Settings,
    *,
    wipe_only: bool,
) -> int:
    """Wipe the Qdrant collection and optionally re-ingest the corpus."""
    logger.info("cli.reset.start", wipe_only=wipe_only)
    try:
        wiped_store = build_qdrant_document_store(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=settings.qdrant_collection,
            embedding_dim=settings.embedding_dim,
            recreate_index=True,
        )
        wiped_store.count_documents()
        logger.info("cli.reset.collection_recreated", collection=settings.qdrant_collection)
    except Exception:
        logger.exception("cli.reset.wipe_failed")
        return 1

    if wipe_only:
        logger.info("cli.reset.wipe_only_done")
        return 0

    # Re-use the freshly recreated store rather than connecting again.
    pipeline = build_ingestion_pipeline(
        settings,
        document_store_factory=lambda: wiped_store,
    )
    try:
        count = pipeline.ingest_corpus(corpus_path)
        logger.info("cli.reindex.done", chunks_written=count)
        return 0
    except Exception:
        logger.exception("cli.reindex.failed")
        return 1


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="Corpus management CLI for the RAG chatbot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("reindex", help="Ingest (or re-ingest) all corpus files.")

    reset_parser = subparsers.add_parser(
        "reset", help="Wipe the vector collection and optionally re-ingest."
    )
    reset_parser.add_argument(
        "--wipe-only",
        action="store_true",
        help="Only wipe the collection; skip re-ingestion.",
    )

    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_format)
    corpus_path = Path(settings.corpus_path)

    if args.command == "reindex":
        return cmd_reindex(corpus_path, settings)
    elif args.command == "reset":
        return cmd_reset(corpus_path, settings, wipe_only=args.wipe_only)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
