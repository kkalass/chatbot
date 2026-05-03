# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenTelemetry setup and trace-safe attribute serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

import structlog
from openinference.instrumentation.haystack import HaystackInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcOTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpOTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.semconv._incubating.attributes.deployment_attributes import (
    DEPLOYMENT_ENVIRONMENT_NAME,
)
from opentelemetry.semconv.attributes.service_attributes import SERVICE_NAME
from phoenix.otel import register

logger = structlog.get_logger(__name__)

_MAX_ATTRIBUTE_CHARS = 4096
_tracing_configured = False
_haystack_instrumented = False


def build_jaeger_exporter(endpoint: str) -> SpanExporter:
    """Build a Jaeger exporter from endpoint format.

    Endpoints ending with ``/v1/traces`` are treated as OTLP/HTTP targets
    (the default mode in this project).
    Otherwise, OTLP/gRPC is used as an explicit opt-in fallback.
    """
    if endpoint.rstrip("/").endswith("/v1/traces"):
        return HttpOTLPSpanExporter(endpoint=endpoint)

    insecure = endpoint.startswith("http://")
    return GrpcOTLPSpanExporter(endpoint=endpoint, insecure=insecure)


def configure_tracing(
    *,
    enabled: bool,
    service_name: str,
    project_name: str,
    deployment_environment: str,
    phoenix_otlp_endpoint: str,
    phoenix_export: bool,
    jaeger_otlp_endpoint: str,
    jaeger_export: bool,
    sample_rate: float,
    console_export: bool,
    auto_instrument_haystack: bool,
) -> None:
    """Configure global OpenTelemetry tracing once for the running process.

    Args:
        enabled: Enables trace exporting when ``True``.
        service_name: Value emitted as ``service.name`` in all spans.
        project_name: Phoenix project used to group local traces.
        deployment_environment: Value emitted as ``deployment.environment`` in resources.
        phoenix_otlp_endpoint: Phoenix OTLP/HTTP endpoint, e.g. ``http://localhost:6006/v1/traces``.
        phoenix_export: Enables exporting traces to Phoenix.
        jaeger_otlp_endpoint: Jaeger OTLP endpoint.
            ``.../v1/traces`` selects OTLP/HTTP (default), otherwise OTLP/gRPC is used.
        jaeger_export: Enables exporting traces to Jaeger.
        sample_rate: Fraction of root traces to sample in ``[0.0, 1.0]``.
        console_export: Enables an additional console exporter for local debugging.
        auto_instrument_haystack: Enables Haystack auto-instrumentation.
    """
    global _haystack_instrumented, _tracing_configured
    if _tracing_configured:
        return

    if not enabled:
        logger.info("tracing.disabled")
        _tracing_configured = True
        return

    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            DEPLOYMENT_ENVIRONMENT_NAME: deployment_environment,
        }
    )
    sampler = ParentBased(TraceIdRatioBased(sample_rate))
    if phoenix_export:
        provider = register(
            endpoint=phoenix_otlp_endpoint,
            project_name=project_name,
            batch=True,
            protocol="http/protobuf",
            verbose=False,
            auto_instrument=False,
            resource=resource,
            sampler=sampler,
        )
    else:
        provider = TracerProvider(resource=resource, sampler=sampler)
        trace.set_tracer_provider(provider)

    if jaeger_export:
        jaeger_processor = BatchSpanProcessor(build_jaeger_exporter(jaeger_otlp_endpoint))
        if phoenix_export:
            # Keep Phoenix's default processor and add Jaeger as an additional sink.
            add_span_processor = cast(Callable[..., None], provider.add_span_processor)
            add_span_processor(jaeger_processor, replace_default_processor=False)
        else:
            provider.add_span_processor(jaeger_processor)

    if auto_instrument_haystack and not _haystack_instrumented:
        HaystackInstrumentor().instrument(tracer_provider=provider)
        _haystack_instrumented = True

    if console_export:
        if phoenix_export:
            # Phoenix extends add_span_processor with replace_default_processor,
            # but its public type currently exposes the base OTel signature only.
            add_span_processor = cast(Callable[..., None], provider.add_span_processor)
            add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter()),
                replace_default_processor=False,
            )
        else:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    _tracing_configured = True
    logger.info(
        "tracing.enabled",
        service_name=service_name,
        project_name=project_name,
        deployment_environment=deployment_environment,
        phoenix_otlp_endpoint=phoenix_otlp_endpoint,
        phoenix_export=phoenix_export,
        jaeger_otlp_endpoint=jaeger_otlp_endpoint,
        jaeger_export=jaeger_export,
        sample_rate=sample_rate,
        console_export=console_export,
        auto_instrument_haystack=auto_instrument_haystack,
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
