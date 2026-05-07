# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stable span operation names for chatbot tracing.

These constants are part of the system's telemetry contract: trace queries,
Phoenix dashboards and alerting depend on the exact strings. They live in
``contracts/`` (rather than next to the OpenInference attribute builders in
``infrastructure/observability/``) so the ``app`` layer can reference them
without taking a dependency on ``infrastructure``.

Attribute ownership rules (high-level):
- UI span: user input preview + final emitted response preview + evaluation metadata.
- Orchestrator spans: control-flow state (steps, tool dispatch, citation pass).
- Model span: compact request/response summaries.
- Tool spans: tool-specific input/result summaries.
- Retriever span: infrastructure-only — params (top_k, score_threshold), result_count, top_scores.
  No content previews, no query text (those belong to the tool span one level up).
"""

# Root/user interaction span
SPAN_CHAT_UI_ON_MESSAGE = "chat.ui.on_message"

# Orchestration spans
SPAN_CHAT_ORCHESTRATOR_STEP = "chat.orchestrator.step"
SPAN_CHAT_ORCHESTRATOR_TOOL_DISPATCH = "chat.orchestrator.tool_dispatch"

# Model adapter spans
SPAN_CHAT_MODEL_OLLAMA_STREAM = "chat.model.ollama.stream"
SPAN_CHAT_MODEL_OPENAI_COMPATIBLE_STREAM = "chat.model.openai_compatible.stream"

# Tool spans
SPAN_CHAT_TOOL_SEARCH_DOCUMENTS = "chat.tool.search_documents"

# Retrieval infrastructure spans
SPAN_CHAT_RETRIEVER_QDRANT_RETRIEVE = "chat.retriever.qdrant.retrieve"
