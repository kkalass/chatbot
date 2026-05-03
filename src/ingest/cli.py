# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Developer CLI for corpus management operations.

Usage (from repository root):

    uv run python -m src.ingest.cli reindex        # ingest all corpus files
    uv run python -m src.ingest.cli reset          # wipe collection and re-ingest
    uv run python -m src.ingest.cli reset --wipe-only  # wipe without re-ingesting

These commands are the FR-06 developer-operations workflows.  They read
configuration from the same ``.env`` / environment variables used by the
application.
"""

import argparse
import sys
from pathlib import Path

import structlog

from src.chatbot.ui.logging_config import configure_logging
from src.ingest.config import (
    build_document_embedder_config,
    build_document_store_config,
    build_ingestion_config,
)
from src.ingest.infrastructure.document_store import DocumentStoreConfig, build_document_store
from src.ingest.infrastructure.embeddings_document import build_document_embedder
from src.ingest.pipeline import IngestionConfig, IngestionPipeline
from src.settings import get_settings

logger = structlog.get_logger(__name__)


def cmd_reindex(
    corpus_path: Path,
    ingestion_config: IngestionConfig,
    store_config: DocumentStoreConfig,
) -> int:
    """Ingest (or re-ingest) all supported files under *corpus_path*.

    Returns:
        Exit code — 0 on success, 1 on error.
    """
    logger.info("cli.reindex.start", corpus=str(corpus_path))
    store = build_document_store(store_config)
    embedder = build_document_embedder(build_document_embedder_config(get_settings()))
    pipeline = IngestionPipeline(config=ingestion_config, document_store=store, embedder=embedder)
    try:
        count = pipeline.ingest_corpus(corpus_path)
        logger.info("cli.reindex.done", chunks_written=count)
        return 0
    except Exception:
        logger.exception("cli.reindex.failed")
        return 1


def cmd_reset(
    corpus_path: Path,
    ingestion_config: IngestionConfig,
    store_config: DocumentStoreConfig,
    *,
    wipe_only: bool,
) -> int:
    """Wipe the Qdrant collection and optionally re-ingest the corpus.

    Args:
        corpus_path: Root corpus directory.
        ingestion_config: Ingestion configuration.
        store_config: Document store configuration.
        wipe_only: When ``True``, skip re-ingestion after wiping.

    Returns:
        Exit code — 0 on success, 1 on error.
    """
    logger.info("cli.reset.start", wipe_only=wipe_only)
    try:
        wipe_store = build_document_store(store_config, recreate_index=True)
        wipe_store.count_documents()
        logger.info("cli.reset.collection_recreated", collection=store_config.collection)
    except Exception:
        logger.exception("cli.reset.wipe_failed")
        return 1

    if wipe_only:
        logger.info("cli.reset.wipe_only_done")
        return 0

    return cmd_reindex(corpus_path, ingestion_config, store_config)


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
    ingestion_config = build_ingestion_config(settings)
    store_config = build_document_store_config(settings)
    corpus_path = Path(settings.corpus_path)

    if args.command == "reindex":
        return cmd_reindex(corpus_path, ingestion_config, store_config)
    elif args.command == "reset":
        return cmd_reset(
            corpus_path,
            ingestion_config,
            store_config,
            wipe_only=args.wipe_only,
        )
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
