"""Provider-agnostic decorator that detects text-encoded tool calls in a chat stream.

Some models (e.g. qwen2.5-coder) serialise tool invocations as plain JSON in
their response text instead of using the native tool-calling channel.  This
wrapper intercepts the stream from any ``ChatModel``, buffers the first chunk
when it looks like a JSON tool call, and either promotes it to a real
``ToolCallInfo`` or re-emits the buffered text if parsing fails.

The wrapper is composed at construction time by ``build_chat_model()`` when
the active ``ModelProfile`` declares ``parse_text_tool_calls = True``.  For
all other models the inner ``ChatModel`` is used directly, so there is zero
overhead.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import cast

import structlog

from src.chatbot.app.protocols import (
    ChatMessage,
    ChatModel,
    ChatStreamItem,
    ToolCallInfo,
    ToolSchema,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _normalize_text_tool_call_payload(text: str) -> str:
    """Strip markdown code fences from a text-encoded tool call payload.

    Some models wrap the JSON in a fenced block such as ``\\`\\`\\`json ... \\`\\`\\```.
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
    """Return whether the first streamed text chunk could be a text-encoded tool call."""
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("```")


def _coerce_text_tool_call_arguments(value: object) -> dict[str, object] | None:
    """Coerce a tool-call argument payload to a JSON object when possible."""
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
    """Parse a text-encoded tool call, returning ``None`` if the text is not a valid call.

    Accepts bare JSON objects and fenced code blocks, with either ``arguments``
    or OpenAI-style ``parameters`` as the key.  String-encoded argument objects
    are decoded automatically.
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


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class TextToolCallParsingWrapper:
    """``ChatModel`` decorator that promotes text-encoded tool calls to native ones.

    Wraps the inner model's stream and applies a single-pass heuristic:

    1. If the **first** text chunk looks like JSON (starts with ``{`` or
       a fenced code block), buffering mode is entered.
    2. All subsequent text chunks are accumulated in the buffer.
    3. After the stream ends, the buffer is parsed via
       :func:`_try_parse_text_tool_call`.  On success the result is yielded as
       a ``list[ToolCallInfo]``; on failure the buffered text is re-emitted
       verbatim.

    Native ``list[ToolCallInfo]`` items from the inner stream always pass
    through unchanged.
    """

    def __init__(self, inner: ChatModel) -> None:
        self._inner = inner

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ChatStreamItem]:
        """Wrap the inner stream, detecting and promoting text-encoded tool calls."""
        inner = self._inner

        async def _gen() -> AsyncGenerator[ChatStreamItem, None]:
            text_seen = False
            buffer: list[str] | None = None

            async for item in inner.stream(messages, tools):
                if isinstance(item, str):
                    if buffer is not None:
                        buffer.append(item)
                    elif not text_seen and _looks_like_text_tool_call_start(item):
                        buffer = [item]
                    else:
                        text_seen = True
                        yield item
                else:
                    # Native list[ToolCallInfo] — pass through unconditionally.
                    yield item

            if buffer is not None:
                accumulated = "".join(buffer)
                parsed = _try_parse_text_tool_call(accumulated)
                if parsed is not None:
                    logger.debug(
                        "chat.text_tool_call_detected",
                        name=parsed.name,
                    )
                    yield [parsed]
                else:
                    for part in buffer:
                        yield part

        return _gen()
