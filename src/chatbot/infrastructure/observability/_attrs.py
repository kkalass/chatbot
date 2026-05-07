# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tracing serialization helpers for chatbot app-layer types.

Two responsibilities, both feature-specific (chatbot domain types):

1. Compact human-readable span-attribute previews of conversation state
   (:func:`summarize_messages`, :func:`summarize_search_result`).
2. OpenInference payload builders that translate chatbot ``ChatMessage`` /
   ``ToolCallInfo`` / ``ToolSchema`` / ``SourceChunk`` into the typed
   dictionaries expected by the OpenInference SDK.

Generic, feature-agnostic OpenInference helpers live in
:mod:`src.shared.observability.openinference`. The split keeps the shared
layer free of imports from chatbot ``contracts``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

from openinference.instrumentation import (
    Document,
    Message,
    Tool,
    ToolCall,
    get_input_attributes,
    get_llm_attributes,
    get_retriever_attributes,
)
from openinference.semconv.trace import OpenInferenceMimeTypeValues

from src.chatbot.contracts.chat import ChatMessage, ToolCallInfo
from src.chatbot.contracts.i18n import JsonObject
from src.chatbot.contracts.retrieval import SourceChunk
from src.chatbot.contracts.tools import ToolSchema
from src.shared.observability import TraceAttributes, to_attribute_text

_DEFAULT_MAX_MESSAGES = 8
_DEFAULT_MAX_CHUNKS = 3


def summarize_messages(
    messages: Sequence[ChatMessage],
    *,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
) -> list[JsonObject]:
    """Return a compact, human-readable summary of the last *max_messages* messages.

    Tool-result messages containing ``search_documents`` chunks are expanded to
    show source + content preview rather than a raw JSON blob.

    Args:
        messages: Full conversation snapshot as passed to the model.
        max_messages: How many messages from the tail to include.

    Returns:
        A list of dicts suitable for ``to_attribute_text()`` serialization.
    """
    summary: list[JsonObject] = []
    for msg in messages[-max_messages:]:
        entry: JsonObject = {
            "role": msg.role,
            "tool_call_id": msg.tool_call_id,
            "tool_calls": [tc.name for tc in msg.tool_calls] if msg.tool_calls else [],
        }
        if isinstance(msg.content, str):
            entry["content_chars"] = len(msg.content)
            entry["content_preview"] = to_attribute_text(msg.content, max_chars=240)
        else:
            entry["content_keys"] = sorted(msg.content.keys())
            chunks = cast(object, msg.content.get("chunks"))
            if isinstance(chunks, list):
                chunk_list = cast(list[object], chunks)
                entry["chunk_count"] = len(chunk_list)
                chunk_items: list[JsonObject] = []
                for chunk_obj in chunk_list[:_DEFAULT_MAX_CHUNKS]:
                    if isinstance(chunk_obj, dict):
                        chunk = cast(dict[str, object], chunk_obj)
                        source = cast(object, chunk.get("source"))
                        chunk_id = cast(object, chunk.get("chunk_id"))
                        content = cast(object, chunk.get("content"))
                        chunk_items.append(
                            {
                                "source": str(source) if source is not None else "",
                                "chunk_id": str(chunk_id) if chunk_id is not None else "",
                                "content_preview": to_attribute_text(
                                    str(content) if content is not None else "",
                                    max_chars=160,
                                ),
                            }
                        )
                entry["chunks"] = chunk_items
        summary.append(entry)
    return summary


def summarize_search_result(
    result: JsonObject,
    *,
    max_chunks: int = _DEFAULT_MAX_CHUNKS,
) -> list[JsonObject]:
    """Return a compact, human-readable preview of a ``search_documents`` tool result.

    Args:
        result: Raw JSON returned by :class:`~src.chatbot.infrastructure.tools.retrieval.RetrievalTool`.
        max_chunks: Number of top chunks to include in the preview.

    Returns:
        A list of dicts with ``source``, ``chunk_id``, ``score``, and
        ``content_preview`` for each chunk, suitable for span attributes.
    """
    chunks = cast(object, result.get("chunks"))
    if not isinstance(chunks, list):
        return []
    chunk_list = cast(list[object], chunks)

    preview: list[JsonObject] = []
    for chunk_obj in chunk_list[:max_chunks]:
        if not isinstance(chunk_obj, dict):
            continue
        chunk = cast(dict[str, object], chunk_obj)
        source = cast(object, chunk.get("source"))
        chunk_id = cast(object, chunk.get("chunk_id"))
        score = cast(object, chunk.get("score"))
        content = cast(object, chunk.get("content"))
        preview.append(
            {
                "source": str(source) if source is not None else "",
                "chunk_id": str(chunk_id) if chunk_id is not None else "",
                "score": score,
                "content_preview": to_attribute_text(
                    str(content) if content is not None else "",
                    max_chars=180,
                ),
            }
        )
    return preview


def build_llm_attributes(
    *,
    provider: str,
    model_name: str,
    messages: Sequence[ChatMessage],
    tools: Sequence[ToolSchema] | None,
    response_text: str,
    tool_calls: Sequence[ToolCallInfo],
    invocation_parameters: dict[str, object] | None = None,
) -> TraceAttributes:
    output_messages: list[Message] = []
    if response_text or tool_calls:
        output_messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "tool_calls": [build_tool_call(tool_call) for tool_call in tool_calls],
            }
        )

    advertised_tools = [build_tool(tool_schema) for tool_schema in tools] if tools else None
    return dict(
        get_llm_attributes(
            provider=provider,
            model_name=model_name,
            invocation_parameters=invocation_parameters,
            input_messages=[build_message(message) for message in messages],
            output_messages=output_messages or None,
            tools=advertised_tools,
        )
    )


def build_retriever_attributes(
    *,
    query: str,
    documents: Sequence[SourceChunk],
) -> TraceAttributes:
    attributes = dict(get_input_attributes(query, mime_type=OpenInferenceMimeTypeValues.TEXT))
    attributes.update(
        get_retriever_attributes(documents=[build_document(document) for document in documents])
    )
    return attributes


def build_message(message: ChatMessage) -> Message:
    payload: Message = {
        "role": message.role,
        "content": _content_to_text(message.content),
    }
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [build_tool_call(tool_call) for tool_call in message.tool_calls]
    return payload


def build_tool_call(tool_call: ToolCallInfo) -> ToolCall:
    return {
        "id": tool_call.call_id,
        "function": {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
        },
    }


def build_tool(tool_schema: ToolSchema) -> Tool:
    return {
        "json_schema": {
            "name": tool_schema.name,
            "description": tool_schema.description,
            "parameters": tool_schema.parameters_schema,
        }
    }


def build_document(chunk: SourceChunk) -> Document:
    metadata = {
        "source": chunk.source,
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "author": chunk.author,
        "publication_date": chunk.publication_date,
        "source_url": chunk.source_url,
        "page": chunk.page,
    }
    return {
        "id": chunk.chunk_id,
        "content": chunk.content,
        "score": chunk.score,
        "metadata": metadata,
    }


def _content_to_text(content: str | dict[str, object]) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=True, default=str)
