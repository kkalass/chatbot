"""Tracing schema contract for span names and attribute ownership.

This module defines stable span operation names used across the chatbot.
Keep these constants as the single source of truth to avoid naming drift
and to make trace queries deterministic.

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
SPAN_CHAT_ORCHESTRATOR_CITATION_PASS = "chat.orchestrator.citation_pass"

# Model adapter spans
SPAN_CHAT_MODEL_OLLAMA_STREAM = "chat.model.ollama.stream"

# Tool spans
SPAN_CHAT_TOOL_SEARCH_DOCUMENTS = "chat.tool.search_documents"
SPAN_CHAT_TOOL_CITE_SOURCES = "chat.tool.cite_sources"

# Retrieval infrastructure spans
SPAN_CHAT_RETRIEVER_QDRANT_RETRIEVE = "chat.retriever.qdrant.retrieve"
