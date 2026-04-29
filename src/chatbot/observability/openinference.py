"""OpenInference helpers for Phoenix-friendly manual span annotation.

This module centralizes all OpenInference-specific constants and helper API
usage so the rest of the application does not handcraft semantic-convention
attribute keys.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from openinference.instrumentation import (
    Document,
    Message,
    Tool,
    ToolCall,
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
    get_retriever_attributes,
    get_tool_attributes,
    using_session,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.util.types import AttributeValue

from src.chatbot.app.protocols import ChatMessage, SourceChunk, ToolCallInfo, ToolSchema
from src.chatbot.observability.tracing import to_attribute_text

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


def using_session_attributes(session_id: str) -> using_session:
    return using_session(session_id)


def _content_to_text(content: str | dict[str, object]) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=True, default=str)
