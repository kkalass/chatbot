# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared observability primitives: tracing, OpenInference, structlog setup.

Cross-cutting between the chatbot UI and the ingest CLI; both entry points need
to configure logging and tracing identically. Lives under `shared/` because it
has no feature-specific knowledge.
"""

from .logging import configure_logging
from .openinference import (
    TraceAttributes,
    build_input_attributes,
    build_metadata_attributes,
    build_output_attributes,
    build_session_attributes,
    build_span_kind_attributes,
    build_tool_execution_attributes,
    using_session_attributes,
)
from .tracing import configure_tracing, to_attribute_text

__all__ = [
    "TraceAttributes",
    "build_input_attributes",
    "build_metadata_attributes",
    "build_output_attributes",
    "build_session_attributes",
    "build_span_kind_attributes",
    "build_tool_execution_attributes",
    "configure_logging",
    "configure_tracing",
    "to_attribute_text",
    "using_session_attributes",
]
