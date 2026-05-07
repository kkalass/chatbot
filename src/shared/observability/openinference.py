# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generic OpenInference helpers for Phoenix-friendly manual span annotation.

Centralizes the OpenInference semantic-convention attribute construction so
callers do not handcraft attribute keys. This module contains only generic,
subsystem-agnostic builders. Subsystem-specific builders that translate
chatbot/ingest domain types into OpenInference payloads live in
``src/chatbot/infrastructure/observability/``.
"""

from __future__ import annotations

from openinference.instrumentation import (
    get_input_attributes,
    get_output_attributes,
    get_tool_attributes,
    using_session,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.util.types import AttributeValue

from src.shared.observability.tracing import to_attribute_text

type TraceAttributes = dict[str, AttributeValue]


def build_span_kind_attributes(kind: OpenInferenceSpanKindValues) -> TraceAttributes:
    return {SpanAttributes.OPENINFERENCE_SPAN_KIND: kind.value}


def build_session_attributes(session_id: str) -> TraceAttributes:
    return {SpanAttributes.SESSION_ID: session_id}


def build_metadata_attributes(metadata: dict[str, object]) -> TraceAttributes:
    return {SpanAttributes.METADATA: to_attribute_text(metadata)}


def build_input_attributes(
    value: object,
    *,
    mime_type: OpenInferenceMimeTypeValues,
) -> TraceAttributes:
    return dict(get_input_attributes(value, mime_type=mime_type))


def build_output_attributes(
    value: object,
    *,
    mime_type: OpenInferenceMimeTypeValues,
) -> TraceAttributes:
    return dict(get_output_attributes(value, mime_type=mime_type))


def build_tool_execution_attributes(
    *,
    tool_name: str,
    parameters: dict[str, object],
) -> TraceAttributes:
    return dict(get_tool_attributes(name=tool_name, parameters=parameters))


def using_session_attributes(session_id: str) -> using_session:
    return using_session(session_id)
