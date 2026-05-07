# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenAI-API-compatible chat model implementation.

Supports any provider that exposes an OpenAI-compatible ``/chat/completions``
endpoint (Groq, Together AI, Fireworks AI, …).  The adapter handles:

- Streaming text via ``AsyncOpenAI.chat.completions.create(stream=True)``.
- Incremental tool-call assembly (OpenAI streams name and arguments in pieces).
- Extraction of ``<think>…</think>`` reasoning blocks — yielded as
  :class:`~src.chatbot.app.protocols.ThinkingContent` items so that callers
  can decide what to do with them without the adapter making UI decisions.

This module is internal to ``src.chatbot.infrastructure.chat`` and must not be
imported outside the infrastructure package.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence

import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.chatbot.contracts.chat import (
    ChatMessage,
    ChatModel,
    ChatStreamItem,
    ThinkingContent,
    ToolCallInfo,
)
from src.chatbot.contracts.observability import SPAN_CHAT_MODEL_OPENAI_COMPATIBLE_STREAM
from src.chatbot.contracts.tools import ToolSchema
from src.chatbot.infrastructure.observability import (
    build_llm_attributes,
    build_tool_call,
    summarize_messages,
)
from src.shared.observability import (
    build_input_attributes,
    build_output_attributes,
    build_span_kind_attributes,
    to_attribute_text,
)

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_OPEN_LEN = len(_THINK_OPEN)
_THINK_CLOSE_LEN = len(_THINK_CLOSE)


# ---------------------------------------------------------------------------
# Protocol conversion helpers
# ---------------------------------------------------------------------------


def _to_openai_tool(tool_schema: ToolSchema) -> ChatCompletionToolParam:
    """Convert a ``ToolSchema`` to an OpenAI function-tool parameter."""
    return {
        "type": "function",
        "function": {
            "name": tool_schema.name,
            "description": tool_schema.description,
            "parameters": tool_schema.parameters_schema,
        },
    }


def _to_openai_message(msg: ChatMessage) -> ChatCompletionMessageParam:
    """Convert a ``ChatMessage`` to an OpenAI-compatible message dict."""
    content: str = json.dumps(msg.content) if isinstance(msg.content, dict) else msg.content
    if msg.role == "tool":
        # OpenAI requires tool_call_id for tool result messages.
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": msg.tool_call_id or "",
        }
    if msg.tool_calls:
        openai_tool_calls = [
            {
                "id": tc.call_id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in msg.tool_calls
        ]
        return {
            "role": "assistant",  # type: ignore[return-value]
            "content": content,
            "tool_calls": openai_tool_calls,  # type: ignore[typeddict-item]
        }
    role = msg.role  # "system" | "user" | "assistant"
    return {"role": role, "content": content}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Think-tag streaming parser
# ---------------------------------------------------------------------------


async def _parse_think_tags(
    inner: AsyncIterator[ChatStreamItem],
) -> AsyncGenerator[ChatStreamItem, None]:
    """Strip ``<think>…</think>`` blocks from a chat stream.

    Text between the tags is packaged as :class:`ThinkingContent`; everything
    outside the tags passes through as plain ``str`` chunks.  Tool-call items
    pass through unchanged.

    The parser is stateful across chunks: an opening tag can arrive mid-chunk
    or be split across two consecutive chunks.  A single call to this function
    produces one ``ThinkingContent`` per complete ``<think>`` block; if the
    stream ends while inside a block (malformed response) the accumulated text
    is still emitted as ``ThinkingContent``.
    """
    in_think = False
    buf = ""  # normal text buffer (NORMAL state) or thinking text (IN_THINK state)

    async for item in inner:
        if not isinstance(item, str):
            # Non-text item: flush any open think block, then pass through.
            if in_think and buf:
                yield ThinkingContent(text=buf)
                buf = ""
                in_think = False
            yield item
            continue

        buf += item

        # Process buf until we can't make further progress without more data.
        while True:
            if not in_think:
                pos = buf.find(_THINK_OPEN)
                if pos >= 0:
                    # Emit text before the tag.
                    if pos > 0:
                        yield buf[:pos]
                    buf = buf[pos + _THINK_OPEN_LEN :]
                    in_think = True
                    # Continue the while loop to handle content after the tag.
                else:
                    # No opening tag found; keep trailing bytes that could be a
                    # partial tag in the buffer, emit the safe prefix.
                    safe = max(0, len(buf) - (_THINK_OPEN_LEN - 1))
                    if safe > 0:
                        yield buf[:safe]
                        buf = buf[safe:]
                    break
            else:
                pos = buf.find(_THINK_CLOSE)
                if pos >= 0:
                    # Complete think block found.
                    yield ThinkingContent(text=buf[:pos])
                    buf = buf[pos + _THINK_CLOSE_LEN :]
                    in_think = False
                    # Continue the while loop: normal text may follow immediately.
                else:
                    # Closing tag not yet complete; keep buffering.
                    # Emit nothing here — thinking content is held until the block closes.
                    break

    # Flush remaining buffer after stream ends.
    if buf:
        if in_think:
            yield ThinkingContent(text=buf)
        else:
            yield buf


# ---------------------------------------------------------------------------
# Tracing helpers
# ---------------------------------------------------------------------------


def _trace_request(
    *,
    span: trace.Span,
    messages: Sequence[ChatMessage],
    tools: Sequence[ToolSchema] | None,
    model: str,
    temperature: float | None,
    seed: int | None,
) -> None:
    request_payload: dict[str, object] = {
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
        "model": model,
        "temperature": temperature,
        "seed": seed,
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
    thinking_chars: int,
    streamed_text_chars: int,
    temperature: float | None,
    seed: int | None,
) -> None:
    response_payload = {
        "text": response_text,
        "tool_calls": [build_tool_call(tool_call) for tool_call in tool_calls],
    }
    invocation_params: dict[str, object] = {"stream": True}
    if temperature is not None:
        invocation_params["temperature"] = temperature
    if seed is not None:
        invocation_params["seed"] = seed
    span.set_attributes(
        build_llm_attributes(
            provider="openai_compatible",
            model_name=model,
            messages=messages,
            tools=tools,
            response_text=response_text,
            tool_calls=tool_calls,
            invocation_parameters=invocation_params,
        )
    )
    span.set_attributes(
        build_output_attributes(
            response_payload,
            mime_type=OpenInferenceMimeTypeValues.JSON,
        )
    )
    span.set_attribute("llm.response.streamed_text_chars", streamed_text_chars)
    span.set_attribute("llm.response.thinking_chars", thinking_chars)
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


# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------


class OpenAICompatibleChatModel:
    """Implements ``ChatModel`` via any OpenAI-compatible ``/chat/completions`` API."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._seed = seed

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ChatStreamItem]:
        """Stream model output and optional tool calls."""
        openai_messages = [_to_openai_message(m) for m in messages]
        openai_tools = [_to_openai_tool(t) for t in tools] if tools else None
        model = self._model
        temperature = self._temperature
        seed = self._seed
        client = self._client

        async def _raw_stream() -> AsyncGenerator[ChatStreamItem, None]:
            """Drive the OpenAI stream and accumulate incremental tool-call deltas."""
            with tracer.start_as_current_span(SPAN_CHAT_MODEL_OPENAI_COMPATIBLE_STREAM) as span:
                _trace_request(
                    span=span,
                    messages=messages,
                    tools=tools,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                )

                logger.debug(
                    "openai_compatible.stream.request",
                    model=model,
                    message_count=len(openai_messages),
                    tool_count=len(openai_tools) if openai_tools else 0,
                )

                # Accumulates incremental tool-call deltas: index → {id, name, arguments}.
                tool_call_acc: dict[int, dict[str, str]] = {}
                trace_response_text_parts: list[str] = []
                trace_streamed_text_chars = 0

                try:
                    # Two branches to avoid NOT_GIVEN/Omit sentinel type mismatch —
                    # the openai 2.x stubs use Omit as sentinel, not NotGiven, so
                    # passing NOT_GIVEN for tools causes a reportArgumentType error.
                    if openai_tools:
                        response_stream = await client.chat.completions.create(  # pyright: ignore[reportCallIssue, reportArgumentType]
                            model=model,
                            messages=openai_messages,
                            tools=openai_tools,
                            temperature=temperature,
                            seed=seed,
                            stream=True,
                        )
                    else:
                        response_stream = await client.chat.completions.create(  # pyright: ignore[reportCallIssue]
                            model=model,
                            messages=openai_messages,
                            temperature=temperature,
                            seed=seed,
                            stream=True,
                        )

                    async for chunk in response_stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta

                        # Accumulate tool-call name/arguments fragments.
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in tool_call_acc:
                                    tool_call_acc[idx] = {
                                        "id": "",
                                        "name": "",
                                        "arguments": "",
                                    }
                                entry = tool_call_acc[idx]
                                if tc_delta.id:
                                    entry["id"] += tc_delta.id
                                if tc_delta.function and tc_delta.function.name:
                                    entry["name"] += tc_delta.function.name
                                if tc_delta.function and tc_delta.function.arguments:
                                    entry["arguments"] += tc_delta.function.arguments

                        content = delta.content
                        if content:
                            trace_streamed_text_chars += len(content)
                            trace_response_text_parts.append(content)
                            yield content

                    # After stream: assemble tool calls from accumulated deltas.
                    tool_calls: list[ToolCallInfo] = []
                    for _idx, entry in sorted(tool_call_acc.items()):
                        try:
                            parsed_args: object = json.loads(entry["arguments"])
                        except json.JSONDecodeError:
                            logger.warning(
                                "openai_compatible.tool_call.invalid_arguments",
                                name=entry["name"],
                                raw_arguments=entry["arguments"],
                            )
                            parsed_args = {}
                        if not isinstance(parsed_args, dict):
                            parsed_args = {}
                        tool_calls.append(
                            ToolCallInfo(
                                name=entry["name"],
                                arguments=parsed_args,  # type: ignore[arg-type]
                                call_id=entry["id"] or entry["name"],
                            )
                        )

                    if tool_calls:
                        yield tool_calls

                    _trace_response(
                        span=span,
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_calls=tool_calls,
                        response_text="".join(trace_response_text_parts),
                        thinking_chars=0,  # updated after think-tag parsing wraps this
                        streamed_text_chars=trace_streamed_text_chars,
                        temperature=temperature,
                        seed=seed,
                    )
                    span.set_status(StatusCode.OK)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise

        # Wrap the raw stream with think-tag parsing so ThinkingContent items
        # emerge before reaching the citation layer.
        return _parse_think_tags(_raw_stream())


def build_openai_compatible_chat_model(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> ChatModel:
    """Build an OpenAI-API-compatible chat model adapter.

    Args:
        base_url: Provider API root (e.g. ``https://api.groq.com/openai/v1``).
        model: Model identifier as accepted by the provider (e.g. ``qwen3-32b``).
        api_key: Provider API key.  Defaults to ``"none"`` (satisfies the client's
            non-empty requirement while still sending the header; some providers
            ignore it entirely for local deployments).
        temperature: Sampling temperature forwarded to the provider.
        seed: Deterministic seed forwarded to the provider.
    """
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or "none",
    )
    return OpenAICompatibleChatModel(
        client=client,
        model=model,
        temperature=temperature,
        seed=seed,
    )
