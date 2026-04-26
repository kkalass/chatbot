"""Chat orchestration: conversation history and agentic tool-call loop.

The orchestrator is the single entry point for the UI layer.  It depends
exclusively on Protocol interfaces — no concrete infrastructure is imported here.

Each turn runs the same loop regardless of whether tools are registered:
stream from the model, yield text chunks to the caller, execute any tool calls
that arrive at the end of the stream, and repeat until the model produces a
plain-text response.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime

import structlog

from src.app.protocols import (ChatMessage, ChatModel, JsonObject, Tool,
                               ToolCallInfo, ToolSchema)

logger = structlog.get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer questions accurately and concisely. "
    "If you are unsure about something, say so rather than guessing. "
    f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
)

_MAX_TOOL_ROUNDS = 10  # safety limit to prevent infinite agentic loops


class ChatOrchestrator:
    """Manages per-session conversation history and the agentic tool-call loop.

    Constructed once per chat session and stored in session-scoped state.
    All dependencies are injected at construction time; no infrastructure is
    instantiated internally.

    Args:
        model: Chat model backend (Protocol).
        tools: Tool implementations advertised to the model.  The orchestrator
            has no knowledge of individual tool semantics — dispatch is by name.
        system_prompt: System instruction prepended to every model call.
    """

    def __init__(
        self,
        model: ChatModel,
        tools: list[Tool] | None = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._model = model
        _tools = tools or []
        self._tool_map: dict[str, Tool] = {t.schema.name: t for t in _tools}
        self._tool_schemas: list[ToolSchema] | None = [t.schema for t in _tools] if _tools else None
        self._system_prompt = system_prompt
        self._history: list[ChatMessage] = []

    def process_message(self, user_text: str) -> AsyncIterator[str]:
        """Process *user_text* and return an async iterator of response chunks.

        Appends the user message to history eagerly (before iteration begins),
        then returns a lazy async iterator that runs the agentic loop.  Tool
        result messages are appended to history during tool rounds but are not
        yielded to the caller — only text responses are streamed back.

        Args:
            user_text: Raw message text from the user.

        Returns:
            An async iterator that yields non-empty string chunks.
        """
        self._history.append(ChatMessage(role="user", content=user_text))
        history = self._history
        model = self._model
        tool_schemas = self._tool_schemas
        tool_map = self._tool_map
        system_prompt = self._system_prompt

        async def _gen() -> AsyncGenerator[str, None]:
            for round_num in range(_MAX_TOOL_ROUNDS):
                messages = [ChatMessage(role="system", content=system_prompt), *history]
                collected: list[str] = []
                tool_calls: list[ToolCallInfo] = []

                async for item in model.stream(messages, tools=tool_schemas):
                    if isinstance(item, str):
                        collected.append(item)
                        yield item
                    else:
                        tool_calls = item

                if not tool_calls:
                    # Plain text response — record and finish.
                    history.append(ChatMessage(role="assistant", content="".join(collected)))
                    return

                logger.info(
                    "orchestrator.tool_round",
                    round=round_num,
                    calls=[tc.name for tc in tool_calls],
                )
                # Record the assistant turn with the tool-call request so the
                # model has the full request→result context on the next round.
                history.append(
                    ChatMessage(
                        role="assistant",
                        content="".join(collected),
                        tool_calls=tuple(tool_calls),
                    )
                )
                for tc in tool_calls:
                    result = await _dispatch(tc, tool_map)
                    history.append(
                        ChatMessage(role="tool", content=result, tool_call_id=tc.call_id)
                    )

            # Safety: exceeded max rounds — stream one final response without tools.
            logger.warning("orchestrator.max_tool_rounds_exceeded", limit=_MAX_TOOL_ROUNDS)
            messages = [ChatMessage(role="system", content=system_prompt), *history]
            final: list[str] = []
            async for item in model.stream(messages, tools=None):
                if isinstance(item, str):
                    final.append(item)
                    yield item
            history.append(ChatMessage(role="assistant", content="".join(final)))

        return _gen()


async def _dispatch(tc: ToolCallInfo, tool_map: dict[str, Tool]) -> JsonObject:
    """Look up the named tool, execute it, and return its structured result."""
    tool = tool_map.get(tc.name)
    if tool is None:
        logger.error("orchestrator.unknown_tool", name=tc.name)
        return {"error": f"unknown tool '{tc.name}'"}
    try:
        return await tool.execute(tc.arguments)
    except Exception as exc:
        logger.exception("orchestrator.tool_error", name=tc.name, exc=str(exc))
        return {"error": f"Tool '{tc.name}' raised an error: {exc}"}
