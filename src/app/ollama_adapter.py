"""Ollama backend adapter implementing the ChatModel Protocol.

This module is the only place in the application that imports the ``ollama``
client library.  All orchestration code depends on the
:class:`~src.app.protocols.ChatModel` Protocol, never on this class directly.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence

import structlog
from ollama import AsyncClient, Message
from ollama._types import ChatResponse
from ollama._types import Tool as OllamaTool

from src.app.protocols import ChatMessage, ToolCallInfo, ToolSchema

logger = structlog.get_logger(__name__)


def _to_ollama_tool(tool_schema: ToolSchema) -> OllamaTool:
    """Convert a :class:`~src.app.protocols.ToolSchema` to an Ollama tool schema.

    Delegates structural coercion to Pydantic's ``model_validate``, which maps
    the JSON Schema dict from ``parameters_schema`` directly onto Ollama's
    nested ``Tool.Function.Parameters`` model without lossy manual mapping.
    """
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
    # Serialise structured tool results to JSON at the wire boundary;
    # all other roles always carry plain text content.
    content: str = json.dumps(msg.content) if isinstance(msg.content, dict) else msg.content
    if msg.role == "tool":
        # Ollama correlates tool results by tool_name, not a UUID-style id.
        # Our call_id for Ollama is the tool name, so we pass it verbatim.
        return Message(role="tool", content=content, tool_name=msg.tool_call_id)
    if msg.tool_calls:
        ollama_tool_calls = [
            Message.ToolCall(
                function=Message.ToolCall.Function(
                    name=tc.name,
                    arguments=tc.arguments,  # type: ignore[arg-type]  # Ollama accepts dict[str, Any]
                )
            )
            for tc in msg.tool_calls
        ]
        return Message(role=msg.role, content=content, tool_calls=ollama_tool_calls)
    return Message(role=msg.role, content=content)


class OllamaChatModel:
    """Implements :class:`~src.app.protocols.ChatModel` via the Ollama HTTP API.

    Args:
        client: A configured :class:`ollama.AsyncClient` instance.
        model: The Ollama model name to use for chat generation.
    """

    def __init__(self, client: AsyncClient, model: str) -> None:
        self._client = client
        self._model = model

    # ------------------------------------------------------------------
    # ChatModel Protocol
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[str | list[ToolCallInfo]]:
        """Stream a completion, optionally advertising tool schemas.

        Yields text chunks as they arrive.  If the model requests tool calls
        instead of generating text, a single ``list[ToolCallInfo]`` is yielded
        as the final item after any (typically empty) content chunks.
        """
        ollama_messages = [_to_ollama_message(m) for m in messages]
        ollama_tools = [_to_ollama_tool(t) for t in tools] if tools else None
        model = self._model
        client = self._client

        async def _gen() -> AsyncGenerator[str | list[ToolCallInfo], None]:
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
                stream=True,
            )

            tool_calls: list[ToolCallInfo] = []
            async for chunk in response_stream:
                content = chunk.message.content
                if content:
                    yield content
                if chunk.message.tool_calls:
                    for tc in chunk.message.tool_calls:
                        tool_calls.append(
                            ToolCallInfo(
                                name=tc.function.name,
                                arguments=dict(tc.function.arguments),
                                # Ollama has no UUID per call; use the tool name as the
                                # correlation token.  The _to_ollama_message serialiser
                                # will pass this back as Message.tool_name.
                                call_id=tc.function.name,
                            )
                        )

            if tool_calls:
                logger.debug("ollama.stream.tool_calls", calls=[t.name for t in tool_calls])
                yield tool_calls
            else:
                logger.debug("ollama.stream.done")

        return _gen()
