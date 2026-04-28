"""OpenTelemetry setup and trace-safe attribute serialization helpers."""

from __future__ import annotations

import json

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

logger = structlog.get_logger(__name__)

_MAX_ATTRIBUTE_CHARS = 4096
_tracing_configured = False


def configure_tracing(
    *,
    enabled: bool,
    service_name: str,
    otlp_endpoint: str,
    sample_rate: float,
    console_export: bool,
) -> None:
    """Configure global OpenTelemetry tracing once for the running process.

    Args:
        enabled: Enables trace exporting when ``True``.
        service_name: Value emitted as ``service.name`` in all spans.
        otlp_endpoint: OTLP/HTTP traces endpoint, e.g. ``http://localhost:4318/v1/traces``.
        sample_rate: Fraction of root traces to sample in ``[0.0, 1.0]``.
        console_export: Enables an additional console exporter for local debugging.
    """
    global _tracing_configured
    if _tracing_configured:
        return

    if not enabled:
        logger.info("tracing.disabled")
        _tracing_configured = True
        return

    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: service_name}),
        sampler=ParentBased(TraceIdRatioBased(sample_rate)),
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))

    if console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracing_configured = True
    logger.info(
        "tracing.enabled",
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        sample_rate=sample_rate,
        console_export=console_export,
    )


def to_attribute_text(value: object, *, max_chars: int = _MAX_ATTRIBUTE_CHARS) -> str:
    """Return a bounded string representation suitable for span attributes."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, default=str)
        except Exception:
            text = str(value)

    if len(text) <= max_chars:
        return text

    return f"{text[:max_chars]}...<truncated>"
