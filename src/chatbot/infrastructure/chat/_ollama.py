"""Ollama-backed chat model implementation.

This module is internal to ``src.chatbot.infrastructure.chat`` and should not be
imported outside the infrastructure package.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import cast

import structlog
from ollama import AsyncClient, Message
from ollama._types import ChatResponse
from ollama._types import Tool as OllamaTool
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.chatbot.app.protocols import (
    ChatMessage,
    ChatModel,
    ChatStreamItem,
    ToolCallInfo,
    ToolSchema,
)
from src.chatbot.app.tracing import summarize_messages
from src.chatbot.observability import to_attribute_text
from src.chatbot.observability.openinference import (
    build_input_attributes,
    build_llm_attributes,
    build_output_attributes,
    build_span_kind_attributes,
    build_tool_call,
)
from src.chatbot.observability.schema import SPAN_CHAT_MODEL_OLLAMA_STREAM

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


def _normalize_text_tool_call_payload(text: str) -> str:
    """Strip markdown code fences from text-encoded tool calls.

    Some models wrap the JSON payload in a fenced block such as
    `````json ... ````` before emitting it as text.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    if not lines[0].startswith("```"):
        return stripped
    if lines[-1].strip() != "```":
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _looks_like_text_tool_call_start(text: str) -> bool:
    """Return whether the first streamed text chunk could be a text tool call."""
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("```")


def _coerce_text_tool_call_arguments(value: object) -> dict[str, object] | None:
    """Coerce tool-call argument payloads to a JSON object when possible."""
    if isinstance(value, dict):
        return value  # type: ignore[return-value]
    if not isinstance(value, str):
        return None
    try:
        parsed: object = json.loads(_normalize_text_tool_call_payload(value))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed  # type: ignore[return-value]


def _try_parse_text_tool_call(text: str) -> ToolCallInfo | None:
    """Parse a text-encoded tool call emitted by models that don't use the native tool_calls field.

    Some models (e.g. qwen2.5-coder) serialise tool invocations as plain JSON text:
    ``{"name": "fn", "arguments": {...}}``
    instead of populating ``message.tool_calls``.  Returns ``None`` if ``text``
    does not match that shape, so the caller can fall back to yielding text.
    """
    normalized = _normalize_text_tool_call_payload(text)
    try:
        # Annotate as object so isinstance narrows cleanly without Unknown propagation.
        raw: object = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    payload = cast(dict[str, object], raw)
    name: object = payload.get("name")
    raw_arguments: object = payload.get("arguments")
    if raw_arguments is None:
        # Some models emit OpenAI-style "parameters" instead of "arguments".
        raw_arguments = payload.get("parameters")
    arguments = _coerce_text_tool_call_arguments(raw_arguments)
    if not isinstance(name, str) or arguments is None:
        return None
    return ToolCallInfo(name=name, arguments=arguments, call_id=name)


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


def _trace_request(
    *,
    span: trace.Span,
    messages: Sequence[ChatMessage],
    tools: Sequence[ToolSchema] | None,
    ollama_options: dict[str, object],
) -> None:
    request_payload = {
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "tool_calls": message.tool_calls,
                "tool_call_id": message.tool_call_id,
            }
            for message in messages
        ],
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_schema": tool.parameters_schema,
            }
            for tool in tools or []
        ],
        "options": ollama_options,
        "stream": True,
    }
    span.set_attributes(build_span_kind_attributes(OpenInferenceSpanKindValues.LLM))
    span.set_attributes(
        build_input_attributes(
            request_payload,
            mime_type=OpenInferenceMimeTypeValues.JSON,
        )
    )
    span.set_attribute("llm.request.message_count", len(messages))
    span.set_attribute("llm.request.tool_count", len(tools) if tools else 0)
    span.set_attribute(
        "llm.request.message_summary",
        to_attribute_text(summarize_messages(messages)),
    )


def _trace_response(
    *,
    span: trace.Span,
    model: str,
    messages: Sequence[ChatMessage],
    tools: Sequence[ToolSchema] | None,
    tool_calls: Sequence[ToolCallInfo],
    response_text: str,
    streamed_text_chars: int,
    ollama_options: dict[str, object],
) -> None:
    response_payload = {
        "text": response_text,
        "tool_calls": [build_tool_call(tool_call) for tool_call in tool_calls],
    }
    span.set_attributes(
        build_llm_attributes(
            provider="ollama",
            model_name=model,
            messages=messages,
            tools=tools,
            response_text=response_text,
            tool_calls=tool_calls,
            invocation_parameters={
                "stream": True,
                **ollama_options,
            },
        )
    )
    span.set_attributes(
        build_output_attributes(
            response_payload,
            mime_type=OpenInferenceMimeTypeValues.JSON,
        )
    )
    span.set_attribute("llm.response.streamed_text_chars", streamed_text_chars)
    span.set_attribute(
        "llm.response.text_preview",
        to_attribute_text(response_text, max_chars=600),
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


class OllamaChatModel:
    """Implements ``ChatModel`` via the Ollama HTTP API."""

    def __init__(
        self,
        client: AsyncClient,
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
        parse_text_tool_calls: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._seed = seed
        self._parse_text_tool_calls = parse_text_tool_calls
        self._ollama_options: dict[str, object] = {
            **({"temperature": temperature} if temperature is not None else {}),
            **({"seed": seed} if seed is not None else {}),
        }

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ChatStreamItem]:
        """Stream model output and optional tool calls."""
        ollama_messages = [_to_ollama_message(m) for m in messages]
        ollama_tools = [_to_ollama_tool(t) for t in tools] if tools else None
        ollama_options = self._ollama_options
        model = self._model
        client = self._client

        async def _gen() -> AsyncGenerator[ChatStreamItem, None]:
            with tracer.start_as_current_span(SPAN_CHAT_MODEL_OLLAMA_STREAM) as span:
                _trace_request(
                    span=span,
                    messages=messages,
                    tools=tools,
                    ollama_options=ollama_options,
                )

                logger.debug(
                    "ollama.stream.request",
                    model=model,
                    message_count=len(ollama_messages),
                    tool_count=len(ollama_tools) if ollama_tools else 0,
                )
                response_stream: AsyncIterator[ChatResponse] = await client.chat(  # pyright: ignore[reportUnknownMemberType]
                    model=model,
                    messages=ollama_messages,
                    tools=ollama_tools,
                    options=ollama_options,
                    stream=True,
                )

                trace_streamed_text_chars = 0
                trace_response_text_parts: list[str] = []

                tool_calls: list[ToolCallInfo] = []
                # Buffer used when the first content chunk starts with '{' — the model
                # may be emitting a text-encoded tool call instead of using tool_calls.
                # Only active when parse_text_tool_calls=True (opt-in, model-specific).
                parse_text_tool_calls = self._parse_text_tool_calls
                text_tool_call_buffer: list[str] | None = None
                try:
                    async for chunk in response_stream:
                        content = chunk.message.content
                        if content:
                            trace_streamed_text_chars += len(content)
                            if text_tool_call_buffer is not None:
                                # Already in buffering mode — keep collecting.
                                text_tool_call_buffer.append(content)
                            elif (
                                parse_text_tool_calls
                                and not trace_response_text_parts
                                and _looks_like_text_tool_call_start(content)
                            ):
                                # First content chunk looks like JSON — enter buffering mode.
                                text_tool_call_buffer = [content]
                            else:
                                trace_response_text_parts.append(content)
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

                    # Resolve text-tool-call buffer: try to parse, fall back to text.
                    if text_tool_call_buffer is not None:
                        accumulated = "".join(text_tool_call_buffer)
                        parsed = _try_parse_text_tool_call(accumulated)
                        if parsed is not None:
                            logger.debug(
                                "ollama.stream.text_tool_call_detected",
                                name=parsed.name,
                            )
                            tool_calls.append(parsed)
                        else:
                            # Not a tool call — emit as regular text.
                            trace_response_text_parts.extend(text_tool_call_buffer)
                            for part in text_tool_call_buffer:
                                yield part

                    if tool_calls:
                        yield tool_calls

                    _trace_response(
                        span=span,
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_calls=tool_calls,
                        response_text="".join(trace_response_text_parts),
                        streamed_text_chars=trace_streamed_text_chars,
                        ollama_options=ollama_options,
                    )
                    span.set_status(StatusCode.OK)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        return _gen()


def build_ollama_chat_model(
    *,
    base_url: str,
    model: str,
    temperature: float | None = None,
    seed: int | None = None,
    parse_text_tool_calls: bool = False,
) -> ChatModel:
    """Build an Ollama-backed chat model.

    Args:
        parse_text_tool_calls: Enable detection of text-encoded tool calls
            (JSON in response text) for models that don't use the native
            tool_calls field (e.g. qwen2.5-coder). Off by default — enabling
            it buffers the entire first response chunk when it starts with
            ``{`` or a fenced code block, which degrades streaming UX for
            normal text models.
    """
    client = AsyncClient(host=base_url)
    return OllamaChatModel(
        client=client,
        model=model,
        temperature=temperature,
        seed=seed,
        parse_text_tool_calls=parse_text_tool_calls,
    )
