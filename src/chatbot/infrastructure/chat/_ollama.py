"""Ollama-backed chat model implementation.

This module is internal to ``src.chatbot.infrastructure.chat`` and should not be
imported outside the infrastructure package.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence

import structlog
from ollama import AsyncClient, Message
from ollama._types import ChatResponse
from ollama._types import Tool as OllamaTool
from opentelemetry import trace

from src.chatbot.app.protocols import ChatMessage, ChatModel, ToolCallInfo, ToolSchema
from src.chatbot.app.tracing import summarize_messages
from src.chatbot.observability import to_attribute_text
from src.chatbot.observability.schema import SPAN_CHAT_MODEL_OLLAMA_STREAM

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


def _to_ollama_tool(tool_schema: ToolSchema) -> OllamaTool:
    """Convert a ``ToolSchema`` to an Ollama function-tool schema."""
    return OllamaTool.model_validate(
        {
            "type": "function",
            "function": {
                "name": tool_schema.name,
                "description": tool_schema.description,
                "parameters": tool_schema.parameters_schema,
            },
        }
    )


def _to_ollama_message(msg: ChatMessage) -> Message:
    content: str = json.dumps(msg.content) if isinstance(msg.content, dict) else msg.content
    if msg.role == "tool":
        return Message(role="tool", content=content, tool_name=msg.tool_call_id)
    if msg.tool_calls:
        ollama_tool_calls = [
            Message.ToolCall(
                function=Message.ToolCall.Function(
                    name=tc.name,
                    arguments=tc.arguments,  # type: ignore[arg-type]
                )
            )
            for tc in msg.tool_calls
        ]
        return Message(role=msg.role, content=content, tool_calls=ollama_tool_calls)
    return Message(role=msg.role, content=content)


class OllamaChatModel:
    """Implements ``ChatModel`` via the Ollama HTTP API."""

    def __init__(
        self,
        client: AsyncClient,
        model: str,
    ) -> None:
        self._client = client
        self._model = model

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[str | list[ToolCallInfo]]:
        """Stream model output and optional tool calls."""
        ollama_messages = [_to_ollama_message(m) for m in messages]
        ollama_tools = [_to_ollama_tool(t) for t in tools] if tools else None
        model = self._model
        client = self._client

        async def _gen() -> AsyncGenerator[str | list[ToolCallInfo], None]:
            with tracer.start_as_current_span(SPAN_CHAT_MODEL_OLLAMA_STREAM) as span:
                logger.debug(
                    "ollama.stream.request",
                    model=model,
                    message_count=len(ollama_messages),
                    tool_count=len(ollama_tools) if ollama_tools else 0,
                )
                span.set_attribute("llm.provider", "ollama")
                span.set_attribute("llm.model", model)
                span.set_attribute("llm.request.message_count", len(ollama_messages))
                span.set_attribute(
                    "llm.request.tool_count", len(ollama_tools) if ollama_tools else 0
                )
                span.set_attribute(
                    "llm.request.message_summary",
                    to_attribute_text(summarize_messages(messages)),
                )

                response_stream: AsyncIterator[ChatResponse] = await client.chat(  # pyright: ignore[reportUnknownMemberType]
                    model=model,
                    messages=ollama_messages,
                    tools=ollama_tools,
                    stream=True,
                )

                tool_calls: list[ToolCallInfo] = []
                streamed_text_chars = 0
                response_text_parts: list[str] = []
                async for chunk in response_stream:
                    content = chunk.message.content
                    if content:
                        streamed_text_chars += len(content)
                        if len("".join(response_text_parts)) < 1200:
                            response_text_parts.append(content)
                        yield content
                    if chunk.message.tool_calls:
                        for tc in chunk.message.tool_calls:
                            tool_calls.append(
                                ToolCallInfo(
                                    name=tc.function.name,
                                    arguments=dict(tc.function.arguments),
                                    call_id=tc.function.name,
                                )
                            )

                span.set_attribute("llm.response.streamed_text_chars", streamed_text_chars)
                span.set_attribute(
                    "llm.response.text_preview",
                    to_attribute_text("".join(response_text_parts), max_chars=600),
                )
                span.set_attribute("llm.response.tool_call_count", len(tool_calls))
                span.set_attribute(
                    "llm.response.tool_calls",
                    to_attribute_text(
                        [
                            {
                                "name": tc.name,
                                "call_id": tc.call_id,
                                "arguments": tc.arguments,
                            }
                            for tc in tool_calls
                        ]
                    ),
                )
                if tool_calls:
                    yield tool_calls

        return _gen()


def build_ollama_chat_model(
    *,
    base_url: str,
    model: str,
) -> ChatModel:
    """Build an Ollama-backed chat model."""
    client = AsyncClient(host=base_url)
    return OllamaChatModel(
        client=client,
        model=model,
    )
