"""structlog configuration for the application.

Call :func:`configure_logging` exactly once at process startup (before any
logger is used).  Log format is driven by ``Settings.log_format`` so that
development gets human-readable output and CI/production gets JSON suitable
for log aggregation.
"""

import logging

import structlog


def configure_logging(log_format: str) -> None:
    """Configure structlog with shared processors and the requested renderer.

    Uses ``structlog.stdlib`` integration so that third-party libraries that
    emit via the standard ``logging`` module are captured and formatted
    consistently.

    Args:
        log_format: ``"console"`` for :class:`~structlog.dev.ConsoleRenderer`,
            ``"json"`` for :class:`~structlog.processors.JSONRenderer`.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    if log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
