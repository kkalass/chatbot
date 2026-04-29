"""Unit tests for ChatOrchestrator — streaming chat path and agentic tool-call loop."""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import replace
from datetime import datetime

import pytest

from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.app.prompts import DEFAULT_PROMPTS, Prompts
from src.chatbot.app.protocols import (
    ChatMessage,
    JsonObject,
    ProcessEvent,
    PromptProfile,
    SourceChunk,
    SourceCitationEvent,
    ToolCallInfo,
    ToolContext,
    ToolEvent,
    ToolSchema,
)

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
        self.stream_tools: list[list[ToolSchema] | None] = []

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[str | list[ToolCallInfo]]:
        self.stream_calls.append(list(messages))
        self.stream_tools.append(list(tools) if tools is not None else None)
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
    def __init__(
        self,
        name: str,
        result: dict[str, object] | None = None,
        events: list[ToolEvent] | None = None,
    ) -> None:
        self.schema = ToolSchema(
            name=name,
            description="A fake tool",
            parameters_schema={"type": "object", "properties": {}},
        )
        self.result: dict[str, object] = result if result is not None else {"output": "tool_result"}
        self.events: list[ToolEvent] = events or []
        self.calls: list[dict[str, object]] = []
        self.contexts: list[ToolContext] = []

    async def execute(
        self, args: dict[str, object], context: ToolContext
    ) -> tuple[dict[str, object], list[ToolEvent]]:
        self.calls.append(dict(args))
        self.contexts.append(context)
        return self.result, list(self.events)


class _IdentityPromptProfile(PromptProfile):
    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        return prompts

    def adjust_tool_description(self, tool_name: str, description: str) -> str:
        return description

    def adjust_parameter_schema(self, tool_name: str, schema: JsonObject) -> JsonObject:
        return schema


class _FixedPromptsProfile(_IdentityPromptProfile):
    def __init__(self, prompts: Prompts) -> None:
        self._prompts = prompts

    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        return self._prompts


class _CountingToolProfile(_IdentityPromptProfile):
    def __init__(self) -> None:
        self.description_calls = 0
        self.schema_calls = 0

    def adjust_tool_description(self, tool_name: str, description: str) -> str:
        self.description_calls += 1
        return f"adapted: {description}"

    def adjust_parameter_schema(self, tool_name: str, schema: JsonObject) -> JsonObject:
        self.schema_calls += 1
        return {
            **schema,
            "x-profile": "counting",
        }


async def _collect(stream: AsyncIterator[ProcessEvent]) -> str:
    parts: list[str] = []
    async for event in stream:
        if isinstance(event, str):
            parts.append(event)
    return "".join(parts)


async def _collect_all(stream: AsyncIterator[ProcessEvent]) -> tuple[str, list[ToolEvent]]:
    text_parts: list[str] = []
    tool_events: list[ToolEvent] = []
    async for event in stream:
        if isinstance(event, str):
            text_parts.append(event)
        else:
            tool_events.append(event)
    return "".join(text_parts), tool_events


class TestChatOrchestratorStreaming:
    @pytest.mark.asyncio
    async def test_yields_model_chunks(self) -> None:
        model = _FakeChatModel(turns=[(["Hello", ", ", "world!"], [])])
        orchestrator = ChatOrchestrator(model, prompt_profile=_IdentityPromptProfile())

        stream = orchestrator.process_message("Hi")
        assert await _collect(stream) == "Hello, world!"

    @pytest.mark.asyncio
    async def test_stream_called_once_per_message(self) -> None:
        model = _FakeChatModel(turns=[(["ok"], [])])
        orchestrator = ChatOrchestrator(model, prompt_profile=_IdentityPromptProfile())

        stream = orchestrator.process_message("Hi")
        await _collect(stream)

        assert len(model.stream_calls) == 1

    @pytest.mark.asyncio
    async def test_user_message_appended_to_history(self) -> None:
        model = _FakeChatModel(turns=[(["ok"], [])])
        orchestrator = ChatOrchestrator(model, prompt_profile=_IdentityPromptProfile())

        stream = orchestrator.process_message("Test question")
        await _collect(stream)

        sent = model.stream_calls[0]
        assert any(m.role == "user" and m.content == "Test question" for m in sent)

    @pytest.mark.asyncio
    async def test_assistant_reply_recorded_in_history(self) -> None:
        model = _FakeChatModel(turns=[(["chunk1", "chunk2"], []), (["second reply"], [])])
        orchestrator = ChatOrchestrator(model, prompt_profile=_IdentityPromptProfile())

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

        def _fixed_prompt(_dt: datetime) -> str:
            return system

        prompts = replace(DEFAULT_PROMPTS, system_prompt=_fixed_prompt)
        orchestrator = ChatOrchestrator(
            model,
            prompt_profile=_FixedPromptsProfile(prompts),
        )

        stream = orchestrator.process_message("hi")
        await _collect(stream)

        first = model.stream_calls[0][0]
        assert first.role == "system"
        assert first.content == system

    @pytest.mark.asyncio
    async def test_history_grows_across_turns(self) -> None:
        model = _FakeChatModel(turns=[(["reply"], []), (["second reply"], [])])
        orchestrator = ChatOrchestrator(model, prompt_profile=_IdentityPromptProfile())

        stream1 = orchestrator.process_message("turn 1")
        await _collect(stream1)
        stream2 = orchestrator.process_message("turn 2")
        await _collect(stream2)

        roles = [m.role for m in model.stream_calls[1]]
        assert roles == ["system", "user", "assistant", "user"]

    @pytest.mark.asyncio
    async def test_empty_response_handled(self) -> None:
        model = _FakeChatModel(turns=[([], [])])
        orchestrator = ChatOrchestrator(model, prompt_profile=_IdentityPromptProfile())

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
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        stream = orchestrator.process_message("hi")
        await _collect(stream)

        assert len(model.stream_calls) == 1

    @pytest.mark.asyncio
    async def test_tool_schema_adaptation_happens_once_at_construction(self) -> None:
        tool = _FakeTool("get_data")
        tc = ToolCallInfo(name="get_data", arguments={})
        model = _FakeChatModel(turns=[([], [tc]), (["Done."], [])])
        profile = _CountingToolProfile()
        orchestrator = ChatOrchestrator(model, tools=[tool], prompt_profile=profile)

        await _collect(orchestrator.process_message("ask"))

        assert profile.description_calls == 1
        assert profile.schema_calls == 1
        assert model.stream_tools[0] is not None
        first_tools = model.stream_tools[0]
        assert first_tools is not None
        assert first_tools[0].description == "adapted: A fake tool"
        assert first_tools[0].parameters_schema["x-profile"] == "counting"

    @pytest.mark.asyncio
    async def test_executes_tool_call_and_yields_final_text(self) -> None:
        tool = _FakeTool("get_data", result={"value": 42})
        tc = ToolCallInfo(name="get_data", arguments={})
        model = _FakeChatModel(
            turns=[
                ([], [tc]),  # step 1: model requests tool
                (["Here is 42!"], []),  # step 2: final text response
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        stream = orchestrator.process_message("give me data")
        result = await _collect(stream)

        assert result == "Here is 42!"
        assert len(tool.calls) == 1

    @pytest.mark.asyncio
    async def test_yields_text_before_tool_call_immediately(self) -> None:
        citation_event = SourceCitationEvent(validated=())
        tool = _FakeTool("get_data", result={"value": 42}, events=[citation_event])
        tc = ToolCallInfo(name="get_data", arguments={})
        model = _FakeChatModel(
            turns=[
                (["Working on it..."], [tc]),
                (["Done."], []),
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        events: list[ProcessEvent] = []
        async for event in orchestrator.process_message("give me data"):
            events.append(event)

        assert events == ["Working on it...", citation_event, "Done."]

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
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
        )

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
        orchestrator = ChatOrchestrator(
            model,
            tools=[known_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

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
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
        )

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
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        stream = orchestrator.process_message("ask")
        await _collect(stream)

        history = orchestrator._history  # pyright: ignore[reportPrivateUsage]
        # The assistant tool-call request must appear before the tool result.
        assistant_msgs = [m for m in history if m.role == "assistant" and m.tool_calls]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].tool_calls == (tc,)

    @pytest.mark.asyncio
    async def test_tool_events_yielded_in_stream(self) -> None:
        chunk = SourceChunk(content="text", source="a.txt", score=0.9, chunk_id="1")
        citation_event = SourceCitationEvent(validated=(chunk,))
        tool = _FakeTool("get_data", result={"ok": True}, events=[citation_event])
        tc = ToolCallInfo(name="get_data", arguments={})
        model = _FakeChatModel(turns=[([], [tc]), (["Done."], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        _, events = await _collect_all(orchestrator.process_message("ask"))

        assert len(events) == 1
        assert events[0] is citation_event

    @pytest.mark.asyncio
    async def test_context_history_snapshot_passed_to_tool(self) -> None:
        tool = _FakeTool("my_tool")
        tc = ToolCallInfo(name="my_tool", arguments={})
        model = _FakeChatModel(turns=[([], [tc]), (["reply"], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        await _collect(orchestrator.process_message("hello"))

        assert len(tool.contexts) == 1
        # Context is a snapshot (tuple), not the live list.
        assert isinstance(tool.contexts[0].history, tuple)
        # History at dispatch time must contain at least the user message.
        roles = [m.role for m in tool.contexts[0].history]
        assert "user" in roles


# ---------------------------------------------------------------------------
# Citation pass
# ---------------------------------------------------------------------------


class TestCitationPass:
    """Verify the citation pass is triggered and emits SourceCitationEvent."""

    def _make_search_result_msg(self, call_id: str) -> dict[str, object]:
        return {"chunks": [{"source": "doc.txt", "content": "some content", "score": 0.9}]}

    @pytest.mark.asyncio
    async def test_citation_pass_triggered_when_search_results_in_history(self) -> None:
        """When search_documents was called, citation pass uses dedicated prompt + cite_sources."""
        chunk = SourceChunk(content="some content", source="doc.txt", score=0.9, chunk_id="1")
        cite_event = SourceCitationEvent(validated=(chunk,))

        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        cite_tc = ToolCallInfo(
            name="cite_sources",
            arguments={"citations": [{"source": "doc.txt", "chunk_id": "1"}]},
            call_id="c1",
        )

        search_tool = _FakeTool(
            "search_documents",
            result={
                "chunks": [
                    {
                        "source": "doc.txt",
                        "chunk_id": "1",
                        "content": "some content",
                        "score": 0.9,
                    }
                ]
            },
        )
        cite_tool = _FakeTool(
            "cite_sources",
            result={
                "validated": [{"source": "doc.txt", "chunk_id": "1"}],
                "unvalidated": [],
            },
            events=[cite_event],
        )

        model = _FakeChatModel(
            turns=[
                ([], [search_tc]),  # main loop: call search
                (["Based on sources."], []),  # main loop: final text
                ([], [cite_tc]),  # citation pass: call cite_sources
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool, cite_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        text, events = await _collect_all(orchestrator.process_message("find me info"))

        assert text == "Based on sources."
        assert len(events) == 1
        assert isinstance(events[0], SourceCitationEvent)
        # Main loop (2 steps) + single citation step.
        assert len(model.stream_calls) == 3
        # Citation pass receives a dedicated two-message prompt (system + citation request).
        citation_messages = model.stream_calls[2]
        assert [m.role for m in citation_messages] == ["system", "user"]
        assert isinstance(citation_messages[1].content, str)
        assert "<search_results>" in citation_messages[1].content
        assert "<answer>" in citation_messages[1].content
        assert "Based on sources." in citation_messages[1].content

    @pytest.mark.asyncio
    async def test_citation_pass_uses_dedicated_citation_system_prompt(self) -> None:
        chunk = SourceChunk(content="some content", source="doc.txt", score=0.9, chunk_id="1")
        cite_event = SourceCitationEvent(validated=(chunk,))

        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        cite_tc = ToolCallInfo(
            name="cite_sources",
            arguments={"citations": [{"source": "doc.txt", "chunk_id": "1"}]},
            call_id="c1",
        )

        prompts = Prompts(
            system_prompt=lambda _now: "GENERAL SYSTEM",
            citation_system_prompt=lambda _now: "CITATION SYSTEM",
            citation_request_message=lambda _search_results, _answer: "citation request",
        )

        search_tool = _FakeTool(
            "search_documents",
            result={
                "chunks": [
                    {
                        "source": "doc.txt",
                        "chunk_id": "1",
                        "content": "some content",
                        "score": 0.9,
                    }
                ]
            },
        )
        cite_tool = _FakeTool(
            "cite_sources",
            result={
                "validated": [{"source": "doc.txt", "chunk_id": "1"}],
                "unvalidated": [],
            },
            events=[cite_event],
        )

        model = _FakeChatModel(
            turns=[
                ([], [search_tc]),
                (["Based on sources."], []),
                ([], [cite_tc]),
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool, cite_tool],
            prompt_profile=_FixedPromptsProfile(prompts),
        )

        await _collect_all(orchestrator.process_message("find me info"))

        assert model.stream_calls[0][0].content == "GENERAL SYSTEM"
        assert model.stream_calls[1][0].content == "GENERAL SYSTEM"
        assert model.stream_calls[2][0].content == "CITATION SYSTEM"

    @pytest.mark.asyncio
    async def test_citation_pass_not_triggered_when_no_search_results(self) -> None:
        """When no search_documents tool call occurred, citation pass is skipped."""
        cite_tool = _FakeTool("cite_sources")
        model = _FakeChatModel(turns=[(["plain answer"], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[cite_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        text, events = await _collect_all(orchestrator.process_message("hi"))

        assert text == "plain answer"
        assert events == []
        # cite_sources must not have been called.
        assert cite_tool.calls == []

    @pytest.mark.asyncio
    async def test_citation_pass_skipped_when_already_cited_in_main_loop(self) -> None:
        """If cite_sources fires in main loop, citation pass does not fire again."""
        chunk = SourceChunk(content="c", source="f.txt", score=0.9, chunk_id="1")
        cite_event = SourceCitationEvent(validated=(chunk,))

        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        cite_tc = ToolCallInfo(
            name="cite_sources",
            arguments={"citations": [{"source": "f.txt", "chunk_id": "1"}]},
            call_id="c1",
        )

        search_tool = _FakeTool(
            "search_documents",
            result={
                "chunks": [
                    {
                        "source": "f.txt",
                        "chunk_id": "1",
                        "content": "c",
                        "score": 0.9,
                    }
                ]
            },
        )
        cite_tool = _FakeTool(
            "cite_sources",
            result={
                "validated": [{"source": "f.txt", "chunk_id": "1"}],
                "unvalidated": [],
            },
            events=[cite_event],
        )

        model = _FakeChatModel(
            turns=[
                ([], [search_tc, cite_tc]),  # main loop: both tools in one step
                (["answer"], []),  # main loop: final text
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool, cite_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        _, events = await _collect_all(orchestrator.process_message("find info"))

        # Only one citation event from the main-loop call — no second citation pass.
        assert len(events) == 1
        # cite_sources called exactly once (during main loop, not again in pass).
        assert len(cite_tool.calls) == 1

    @pytest.mark.asyncio
    async def test_citation_pass_ignores_non_citation_tool_calls(self) -> None:
        """Citation pass should not dispatch tools other than cite_sources."""
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        wrong_tc = ToolCallInfo(name="search_documents", arguments={"query": "again"}, call_id="s2")

        search_tool = _FakeTool(
            "search_documents",
            result={
                "chunks": [
                    {
                        "source": "doc.txt",
                        "chunk_id": "1",
                        "content": "c",
                        "score": 0.9,
                    }
                ]
            },
        )
        cite_tool = _FakeTool(
            "cite_sources",
            result={"validated": [], "unvalidated": []},
            events=[],
        )

        model = _FakeChatModel(
            turns=[
                ([], [search_tc]),
                (["answer"], []),
                ([], [wrong_tc]),
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool, cite_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        _, events = await _collect_all(orchestrator.process_message("find info"))

        assert events == []
        assert len(search_tool.calls) == 1
        assert cite_tool.calls == []

    @pytest.mark.asyncio
    async def test_citation_pass_recovers_serialized_citation_tool_call(self) -> None:
        """If model emits serialized cite_sources JSON as text, pass should recover it."""
        chunk = SourceChunk(content="c", source="doc.txt", score=0.9, chunk_id="1")
        cite_event = SourceCitationEvent(validated=(chunk,))

        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        search_tool = _FakeTool(
            "search_documents",
            result={
                "chunks": [
                    {
                        "source": "doc.txt",
                        "chunk_id": "1",
                        "content": "c",
                        "score": 0.9,
                    }
                ]
            },
        )
        cite_tool = _FakeTool(
            "cite_sources",
            result={
                "validated": [{"source": "doc.txt", "chunk_id": "1"}],
                "unvalidated": [],
            },
            events=[cite_event],
        )

        serialized = (
            '{"name":"cite_sources","parameters":{"citations":['
            '{"source":"doc.txt","chunk_id":"1"}]}}'
        )
        model = _FakeChatModel(
            turns=[
                ([], [search_tc]),
                (["answer"], []),
                ([serialized], []),
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool, cite_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        _, events = await _collect_all(orchestrator.process_message("find info"))

        assert len(events) == 1
        assert len(cite_tool.calls) == 1
        assert cite_tool.calls[0] == {"citations": [{"source": "doc.txt", "chunk_id": "1"}]}

    @pytest.mark.asyncio
    async def test_citation_pass_recovers_malformed_serialized_citation_tool_call(self) -> None:
        """Recovery should also work for lightly malformed serialized JSON payloads."""
        chunk = SourceChunk(content="c", source="doc.txt", score=0.9, chunk_id="1")
        cite_event = SourceCitationEvent(validated=(chunk,))

        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        search_tool = _FakeTool(
            "search_documents",
            result={
                "chunks": [
                    {
                        "source": "doc.txt",
                        "chunk_id": "1",
                        "content": "c",
                        "score": 0.9,
                    }
                ]
            },
        )
        cite_tool = _FakeTool(
            "cite_sources",
            result={
                "validated": [{"source": "doc.txt", "chunk_id": "1"}],
                "unvalidated": [],
            },
            events=[cite_event],
        )

        malformed_serialized = (
            '{"name":"cite_sources","parameters":{"citations":[{"source":"doc.txt","chunk_id":"1"}]'
        )
        model = _FakeChatModel(
            turns=[
                ([], [search_tc]),
                (["answer"], []),
                ([malformed_serialized], []),
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool, cite_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        _, events = await _collect_all(orchestrator.process_message("find info"))

        assert len(events) == 1
        assert len(cite_tool.calls) == 1
        assert cite_tool.calls[0] == {"citations": [{"source": "doc.txt", "chunk_id": "1"}]}
