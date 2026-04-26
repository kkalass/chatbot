"""Unit tests for ChatOrchestrator — streaming chat path and agentic tool-call loop."""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence

import pytest

from src.app.orchestrator import ChatOrchestrator
from src.app.protocols import ChatMessage, ToolCallInfo, ToolSchema

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeChatModel:
    """Configurable fake implementing the ChatModel protocol.

    Each turn is ``(text_chunks, tool_calls)``.  For a plain-text response
    supply non-empty ``text_chunks`` and ``[]`` for tool_calls.  For a
    tool-call response supply ``[]`` for text_chunks and the desired calls.
    Mirrors the real ``stream()`` contract: yields str chunks then, if any
    tool_calls are present, yields the list as the final item.
    """

    def __init__(
        self,
        turns: list[tuple[list[str], list[ToolCallInfo]]] | None = None,
    ) -> None:
        # Default: single plain-text turn.
        self.turns: list[tuple[list[str], list[ToolCallInfo]]] = turns or [
            (["default response"], [])
        ]
        self._turn_idx = 0
        self.stream_calls: list[list[ChatMessage]] = []

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[str | list[ToolCallInfo]]:
        self.stream_calls.append(list(messages))
        idx = min(self._turn_idx, len(self.turns) - 1)
        self._turn_idx += 1
        chunks, tool_calls = self.turns[idx]
        return self._gen(chunks, tool_calls)

    def _gen(
        self,
        chunks: list[str],
        tool_calls: list[ToolCallInfo],
    ) -> AsyncIterator[str | list[ToolCallInfo]]:
        async def _inner() -> AsyncGenerator[str | list[ToolCallInfo], None]:
            for chunk in chunks:
                yield chunk
            if tool_calls:
                yield tool_calls

        return _inner()


class _FakeTool:
    def __init__(self, name: str, result: dict[str, object] | None = None) -> None:
        self.schema = ToolSchema(
            name=name,
            description="A fake tool",
            parameters_schema={"type": "object", "properties": {}},
        )
        self.result: dict[str, object] = result if result is not None else {"output": "tool_result"}
        self.calls: list[dict[str, object]] = []

    async def execute(self, args: dict[str, object]) -> dict[str, object]:
        self.calls.append(dict(args))
        return self.result


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


class TestChatOrchestratorStreaming:
    @pytest.mark.asyncio
    async def test_yields_model_chunks(self) -> None:
        model = _FakeChatModel(turns=[(["Hello", ", ", "world!"], [])])
        orchestrator = ChatOrchestrator(model)

        stream = orchestrator.process_message("Hi")
        assert await _collect(stream) == "Hello, world!"

    @pytest.mark.asyncio
    async def test_stream_called_once_per_message(self) -> None:
        model = _FakeChatModel(turns=[(["ok"], [])])
        orchestrator = ChatOrchestrator(model)

        stream = orchestrator.process_message("Hi")
        await _collect(stream)

        assert len(model.stream_calls) == 1

    @pytest.mark.asyncio
    async def test_user_message_appended_to_history(self) -> None:
        model = _FakeChatModel(turns=[(["ok"], [])])
        orchestrator = ChatOrchestrator(model)

        stream = orchestrator.process_message("Test question")
        await _collect(stream)

        sent = model.stream_calls[0]
        assert any(m.role == "user" and m.content == "Test question" for m in sent)

    @pytest.mark.asyncio
    async def test_assistant_reply_recorded_in_history(self) -> None:
        model = _FakeChatModel(turns=[(["chunk1", "chunk2"], []), (["second reply"], [])])
        orchestrator = ChatOrchestrator(model)

        stream = orchestrator.process_message("question")
        await _collect(stream)

        stream2 = orchestrator.process_message("follow-up")
        await _collect(stream2)

        second_call = model.stream_calls[1]
        assistant_messages = [m for m in second_call if m.role == "assistant"]
        assert assistant_messages[-1].content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_system_prompt_is_first_message(self) -> None:
        model = _FakeChatModel(turns=[(["response"], [])])
        system = "You are a test bot."
        orchestrator = ChatOrchestrator(model, system_prompt=system)

        stream = orchestrator.process_message("hi")
        await _collect(stream)

        first = model.stream_calls[0][0]
        assert first.role == "system"
        assert first.content == system

    @pytest.mark.asyncio
    async def test_history_grows_across_turns(self) -> None:
        model = _FakeChatModel(turns=[(["reply"], []), (["second reply"], [])])
        orchestrator = ChatOrchestrator(model)

        stream1 = orchestrator.process_message("turn 1")
        await _collect(stream1)
        stream2 = orchestrator.process_message("turn 2")
        await _collect(stream2)

        roles = [m.role for m in model.stream_calls[1]]
        assert roles == ["system", "user", "assistant", "user"]

    @pytest.mark.asyncio
    async def test_empty_response_handled(self) -> None:
        model = _FakeChatModel(turns=[([], [])])
        orchestrator = ChatOrchestrator(model)

        stream = orchestrator.process_message("silence")
        assert await _collect(stream) == ""


# ---------------------------------------------------------------------------
# Agentic loop path
# ---------------------------------------------------------------------------


class TestChatOrchestratorAgenticLoop:
    @pytest.mark.asyncio
    async def test_stream_called_when_tools_configured(self) -> None:
        model = _FakeChatModel(turns=[(["response text"], [])])
        tool = _FakeTool("my_tool")
        orchestrator = ChatOrchestrator(model, tools=[tool])

        stream = orchestrator.process_message("hi")
        await _collect(stream)

        assert len(model.stream_calls) == 1

    @pytest.mark.asyncio
    async def test_executes_tool_call_and_yields_final_text(self) -> None:
        tool = _FakeTool("get_data", result={"value": 42})
        tc = ToolCallInfo(name="get_data", arguments={})
        model = _FakeChatModel(
            turns=[
                ([], [tc]),  # round 1: model requests tool
                (["Here is 42!"], []),  # round 2: final text response
            ]
        )
        orchestrator = ChatOrchestrator(model, tools=[tool])

        stream = orchestrator.process_message("give me data")
        result = await _collect(stream)

        assert result == "Here is 42!"
        assert len(tool.calls) == 1

    @pytest.mark.asyncio
    async def test_tool_result_appended_before_second_complete_call(self) -> None:
        tool = _FakeTool("get_data", result={"value": 99})
        tc = ToolCallInfo(name="get_data", arguments={"x": 1}, call_id="call-abc")
        model = _FakeChatModel(
            turns=[
                ([], [tc]),
                (["Done!"], []),
            ]
        )
        orchestrator = ChatOrchestrator(model, tools=[tool])

        stream = orchestrator.process_message("query")
        await _collect(stream)

        # Second stream() call must include a tool result message (JSON-serialised).
        second_call_messages = model.stream_calls[1]
        tool_msgs = [m for m in second_call_messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == {"value": 99}
        # Correlation token must be threaded through opaquely.
        assert tool_msgs[0].tool_call_id == "call-abc"

    @pytest.mark.asyncio
    async def test_unknown_tool_name_does_not_raise(self) -> None:
        tc = ToolCallInfo(name="nonexistent", arguments={})
        model = _FakeChatModel(
            turns=[
                ([], [tc]),
                (["Handled."], []),
            ]
        )
        known_tool = _FakeTool("known_tool")
        orchestrator = ChatOrchestrator(model, tools=[known_tool])

        stream = orchestrator.process_message("trigger unknown")
        result = await _collect(stream)
        assert result == "Handled."

    @pytest.mark.asyncio
    async def test_final_text_recorded_in_history(self) -> None:
        tc = ToolCallInfo(name="my_tool", arguments={})
        tool = _FakeTool("my_tool")
        model = _FakeChatModel(
            turns=[([], [tc]), (["Final answer."], [])],
        )
        orchestrator = ChatOrchestrator(model, tools=[tool])

        stream = orchestrator.process_message("ask")
        await _collect(stream)

        # The history must include the final assistant reply.
        history = orchestrator._history  # pyright: ignore[reportPrivateUsage]
        assert history[-1] == ChatMessage(role="assistant", content="Final answer.")

    @pytest.mark.asyncio
    async def test_tool_call_assistant_message_recorded_in_history(self) -> None:
        tc = ToolCallInfo(name="my_tool", arguments={"x": 1})
        tool = _FakeTool("my_tool")
        model = _FakeChatModel(
            turns=[([], [tc]), (["Done."], [])],
        )
        orchestrator = ChatOrchestrator(model, tools=[tool])

        stream = orchestrator.process_message("ask")
        await _collect(stream)

        history = orchestrator._history  # pyright: ignore[reportPrivateUsage]
        # The assistant tool-call request must appear before the tool result.
        assistant_msgs = [m for m in history if m.role == "assistant" and m.tool_calls]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].tool_calls == (tc,)
