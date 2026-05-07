# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chatbot-specific OpenInference payload builders.

OpenInference attribute builders that translate chatbot ``ChatMessage`` /
``ToolCallInfo`` / ``ToolSchema`` / ``SourceChunk`` into the typed
dictionaries expected by the OpenInference SDK, plus compact human-readable
span-attribute previews of conversation state.

Span operation names live in :mod:`src.chatbot.contracts.observability` so
the ``app`` layer can reference them without importing infrastructure.
Generic, feature-agnostic OpenInference helpers live in
:mod:`src.shared.observability.openinference`.
"""

from ._attrs import (
    build_document,
    build_llm_attributes,
    build_message,
    build_retriever_attributes,
    build_tool,
    build_tool_call,
    summarize_messages,
    summarize_search_result,
)

__all__ = [
    "build_document",
    "build_llm_attributes",
    "build_message",
    "build_retriever_attributes",
    "build_tool",
    "build_tool_call",
    "summarize_messages",
    "summarize_search_result",
]
