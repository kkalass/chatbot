"""Unit tests for ChatOrchestrator."""

from collections.abc import AsyncIterator, Sequence

import pytest

from src.app.orchestrator import ChatOrchestrator
from src.app.protocols import ChatMessage


class _FakeChatModel:
    """Test double that returns a configurable stream of chunks."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.calls: list[list[ChatMessage]] = []

    def stream_chat(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        return self._stream()

    async def _stream(self) -> AsyncIterator[str]:
        for chunk in self.chunks:
            yield chunk


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


class TestChatOrchestrator:
    @pytest.mark.asyncio
    async def test_yields_model_chunks(self) -> None:
        model = _FakeChatModel(["Hello", ", ", "world!"])
        orchestrator = ChatOrchestrator(model)

        result = await _collect(orchestrator.process_message("Hi"))

        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_user_message_appended_to_history(self) -> None:
        model = _FakeChatModel(["ok"])
        orchestrator = ChatOrchestrator(model)

        await _collect(orchestrator.process_message("Test question"))

        sent_messages = model.calls[0]
        user_messages = [m for m in sent_messages if m.role == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].content == "Test question"

    @pytest.mark.asyncio
    async def test_assistant_reply_recorded_in_history(self) -> None:
        model = _FakeChatModel(["chunk1", "chunk2"])
        orchestrator = ChatOrchestrator(model)

        await _collect(orchestrator.process_message("question"))

        # Second message includes the assistant reply from the previous turn
        model.chunks = ["second reply"]
        await _collect(orchestrator.process_message("follow-up"))
        second_call_messages = model.calls[1]
        assistant_messages = [m for m in second_call_messages if m.role == "assistant"]
        assert assistant_messages[-1].content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_system_prompt_is_first_message(self) -> None:
        model = _FakeChatModel(["response"])
        system = "You are a test bot."
        orchestrator = ChatOrchestrator(model, system_prompt=system)

        await _collect(orchestrator.process_message("hi"))

        first_message = model.calls[0][0]
        assert first_message.role == "system"
        assert first_message.content == system

    @pytest.mark.asyncio
    async def test_history_grows_across_turns(self) -> None:
        model = _FakeChatModel(["reply"])
        orchestrator = ChatOrchestrator(model)

        await _collect(orchestrator.process_message("turn 1"))
        model.chunks = ["second reply"]
        await _collect(orchestrator.process_message("turn 2"))

        # Second call should include system + user1 + assistant1 + user2
        second_call = model.calls[1]
        roles = [m.role for m in second_call]
        assert roles == ["system", "user", "assistant", "user"]

    @pytest.mark.asyncio
    async def test_empty_response_handled(self) -> None:
        model = _FakeChatModel([])
        orchestrator = ChatOrchestrator(model)

        result = await _collect(orchestrator.process_message("silence"))

        assert result == ""
