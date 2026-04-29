"""Unit tests for ChatOrchestrator — streaming chat path and agentic tool-call loop."""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import replace
from datetime import datetime

import pytest

from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.app.prompts import DEFAULT_PROMPTS, Prompts
from src.chatbot.app.protocols import (
    ChatMessage,
    ChatRuntimeFlags,
    JsonObject,
    ProcessEvent,
    PromptProfile,
    QuoteReferenceEvent,
    SearchResultQuote,
    SourceChunk,
    SourceCitationEvent,
    ToolCallInfo,
    ToolCallQuote,
    ToolContext,
    ToolEvent,
    ToolSchema,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeChatModel:
    """Configurable fake implementing the ChatModel protocol.

    Each turn is ``(stream_items, tool_calls)``.  ``stream_items`` may contain
    ``str`` chunks or ``Quote`` objects (for testing inline-quote handling).
    For a plain-text response supply non-empty ``stream_items`` and ``[]`` for
    tool_calls.  For a tool-call response supply ``[]`` for stream_items and the
    desired calls.
    Mirrors the real ``stream()`` contract: yields items then, if any
    tool_calls are present, yields the list as the final item.
    """

    def __init__(
        self,
        turns: list[tuple[list[str | SearchResultQuote | ToolCallQuote], list[ToolCallInfo]]]
        | None = None,
    ) -> None:
        # Default: single plain-text turn.
        self.turns: list[
            tuple[list[str | SearchResultQuote | ToolCallQuote], list[ToolCallInfo]]
        ] = turns or [(["default response"], [])]
        self._turn_idx = 0
        self.stream_calls: list[list[ChatMessage]] = []
        self.stream_tools: list[list[ToolSchema] | None] = []

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[str | list[ToolCallInfo] | SearchResultQuote | ToolCallQuote]:
        self.stream_calls.append(list(messages))
        self.stream_tools.append(list(tools) if tools is not None else None)
        idx = min(self._turn_idx, len(self.turns) - 1)
        self._turn_idx += 1
        chunks, tool_calls = self.turns[idx]
        return self._gen(chunks, tool_calls)

    def _gen(
        self,
        chunks: list[str | SearchResultQuote | ToolCallQuote],
        tool_calls: list[ToolCallInfo],
    ) -> AsyncIterator[str | list[ToolCallInfo] | SearchResultQuote | ToolCallQuote]:
        async def _inner() -> AsyncGenerator[
            str | list[ToolCallInfo] | SearchResultQuote | ToolCallQuote, None
        ]:
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


async def _collect_all(stream: AsyncIterator[ProcessEvent]) -> tuple[str, list[ProcessEvent]]:
    text_parts: list[str] = []
    non_text_events: list[ProcessEvent] = []
    async for event in stream:
        if isinstance(event, str):
            text_parts.append(event)
        else:
            non_text_events.append(event)
    return "".join(text_parts), non_text_events


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


# ---------------------------------------------------------------------------
# Convenience flag constants for WP4 tests
# ---------------------------------------------------------------------------

_FLAGS_INLINE_ONLY = ChatRuntimeFlags(inline_quotes_enabled=True, citation_round_trip_enabled=False)
_FLAGS_ROUND_TRIP_ONLY = ChatRuntimeFlags(
    inline_quotes_enabled=False, citation_round_trip_enabled=True
)
_FLAGS_BOTH = ChatRuntimeFlags(inline_quotes_enabled=True, citation_round_trip_enabled=True)
_FLAGS_NEITHER = ChatRuntimeFlags(inline_quotes_enabled=False, citation_round_trip_enabled=False)


def _make_search_tool_result(
    source: str = "doc.txt",
    chunk_id: str = "c1",
    content: str = "body",
) -> dict[str, object]:
    return {"chunks": [{"source": source, "chunk_id": chunk_id, "content": content, "score": 0.9}]}


class TestInlineQuotePipeline:
    """WP4: orchestrator inline-quote collection, dedup, and citation finalisation."""

    @pytest.mark.asyncio
    async def test_valid_search_quote_emits_reference_event(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        search_tool = _FakeTool("search_documents", result=_make_search_tool_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["Answer. ", quote, " more text."], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_INLINE_ONLY,
        )

        text, events = await _collect_all(orchestrator.process_message("ask"))

        ref_events = [e for e in events if isinstance(e, QuoteReferenceEvent)]
        assert len(ref_events) == 1
        assert ref_events[0].reference_number == 1
        assert "s1" in ref_events[0].canonical_key
        assert "doc.txt" in ref_events[0].canonical_key
        assert "c1" in ref_events[0].canonical_key
        assert text == "Answer.  more text."

    @pytest.mark.asyncio
    async def test_invalid_search_quote_not_in_history_is_dropped(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        bad_quote = SearchResultQuote(
            claim="claim", tool_call_id="s1", source="unknown.txt", chunk_id="no-such-chunk"
        )
        search_tool = _FakeTool("search_documents", result=_make_search_tool_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["Text.", bad_quote], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_INLINE_ONLY,
        )

        text, events = await _collect_all(orchestrator.process_message("ask"))

        assert text == "Text."
        assert not any(isinstance(e, QuoteReferenceEvent) for e in events)

    @pytest.mark.asyncio
    async def test_duplicate_quote_reuses_reference_number(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        search_tool = _FakeTool("search_documents", result=_make_search_tool_result())
        model = _FakeChatModel(
            turns=[([], [search_tc]), (["A ", quote, " and again ", quote, "."], [])]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_INLINE_ONLY,
        )

        _, events = await _collect_all(orchestrator.process_message("ask"))

        ref_events = [e for e in events if isinstance(e, QuoteReferenceEvent)]
        assert len(ref_events) == 2
        assert ref_events[0].reference_number == 1
        assert ref_events[1].reference_number == 1
        assert ref_events[0].canonical_key == ref_events[1].canonical_key

    @pytest.mark.asyncio
    async def test_two_distinct_quotes_get_increasing_reference_numbers(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote_a = SearchResultQuote(claim="a", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        quote_b = SearchResultQuote(claim="b", tool_call_id="s1", source="doc.txt", chunk_id="c2")
        search_tool = _FakeTool(
            "search_documents",
            result={
                "chunks": [
                    {"source": "doc.txt", "chunk_id": "c1", "content": "body1", "score": 0.9},
                    {"source": "doc.txt", "chunk_id": "c2", "content": "body2", "score": 0.8},
                ]
            },
        )
        model = _FakeChatModel(
            turns=[([], [search_tc]), (["A ", quote_a, " B ", quote_b, "."], [])]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_INLINE_ONLY,
        )

        _, events = await _collect_all(orchestrator.process_message("ask"))

        ref_events = [e for e in events if isinstance(e, QuoteReferenceEvent)]
        assert len(ref_events) == 2
        assert ref_events[0].reference_number == 1
        assert ref_events[1].reference_number == 2
        assert ref_events[0].canonical_key != ref_events[1].canonical_key

    @pytest.mark.asyncio
    async def test_valid_search_quotes_produce_source_citation_event(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        search_tool = _FakeTool("search_documents", result=_make_search_tool_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["Answer.", quote], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_INLINE_ONLY,
        )

        _, events = await _collect_all(orchestrator.process_message("ask"))

        citation_events = [e for e in events if isinstance(e, SourceCitationEvent)]
        assert len(citation_events) == 1
        assert len(citation_events[0].validated) == 1
        assert citation_events[0].validated[0].source == "doc.txt"
        assert citation_events[0].validated[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_inline_citation_suppresses_legacy_citation_pass(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        search_tool = _FakeTool("search_documents", result=_make_search_tool_result())
        cite_tool = _FakeTool("cite_sources", result={}, events=[])
        model = _FakeChatModel(turns=[([], [search_tc]), (["Answer.", quote], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool, cite_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_BOTH,
        )

        await _collect_all(orchestrator.process_message("ask"))

        assert len(model.stream_calls) == 2  # search step + final; no citation pass
        assert cite_tool.calls == []

    @pytest.mark.asyncio
    async def test_citation_pass_disabled_via_flag(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        search_tool = _FakeTool("search_documents", result=_make_search_tool_result())
        cite_tool = _FakeTool("cite_sources", result={}, events=[])
        model = _FakeChatModel(turns=[([], [search_tc]), (["Answer."], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool, cite_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_NEITHER,
        )

        _, events = await _collect_all(orchestrator.process_message("ask"))

        assert len(model.stream_calls) == 2
        assert cite_tool.calls == []
        assert events == []

    @pytest.mark.asyncio
    async def test_valid_tool_call_quote_emits_reference_event(self) -> None:
        vacation_tc = ToolCallInfo(name="get_vacation_days", arguments={}, call_id="v1")
        quote = ToolCallQuote(
            claim="27 days remaining", tool_call_id="v1", tool_name="get_vacation_days"
        )
        vacation_tool = _FakeTool("get_vacation_days", result={"remaining_days": 27})
        model = _FakeChatModel(
            turns=[([], [vacation_tc]), (["You have ", quote, " days left."], [])]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[vacation_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_INLINE_ONLY,
        )

        _, events = await _collect_all(orchestrator.process_message("ask"))

        ref_events = [e for e in events if isinstance(e, QuoteReferenceEvent)]
        assert len(ref_events) == 1
        assert ref_events[0].reference_number == 1
        assert "v1" in ref_events[0].canonical_key
        assert "get_vacation_days" in ref_events[0].canonical_key

    @pytest.mark.asyncio
    async def test_invalid_tool_call_quote_unknown_call_id_is_dropped(self) -> None:
        quote = ToolCallQuote(
            claim="something", tool_call_id="nonexistent", tool_name="get_vacation_days"
        )
        model = _FakeChatModel(turns=[(["Answer ", quote, "."], [])])
        orchestrator = ChatOrchestrator(
            model, prompt_profile=_IdentityPromptProfile(), runtime_flags=_FLAGS_INLINE_ONLY
        )

        text, events = await _collect_all(orchestrator.process_message("ask"))

        assert text == "Answer ."
        assert not any(isinstance(e, QuoteReferenceEvent) for e in events)

    @pytest.mark.asyncio
    async def test_mixed_stream_text_and_quotes_interleaved(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote_a = SearchResultQuote(claim="a", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        quote_b = SearchResultQuote(claim="b", tool_call_id="s1", source="doc.txt", chunk_id="c2")
        search_tool = _FakeTool(
            "search_documents",
            result={
                "chunks": [
                    {"source": "doc.txt", "chunk_id": "c1", "content": "body1", "score": 0.9},
                    {"source": "doc.txt", "chunk_id": "c2", "content": "body2", "score": 0.8},
                ]
            },
        )
        model = _FakeChatModel(
            turns=[([], [search_tc]), (["First ", quote_a, " second ", quote_b, " end."], [])]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_INLINE_ONLY,
        )

        text, events = await _collect_all(orchestrator.process_message("ask"))

        assert text == "First  second  end."
        ref_events = [e for e in events if isinstance(e, QuoteReferenceEvent)]
        assert len(ref_events) == 2
        citation_events = [e for e in events if isinstance(e, SourceCitationEvent)]
        assert len(citation_events) == 1
        assert len(citation_events[0].validated) == 2

    @pytest.mark.asyncio
    async def test_inline_quotes_disabled_skips_all_quote_processing(self) -> None:
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        model = _FakeChatModel(turns=[(["Text ", quote, " more."], [])])
        orchestrator = ChatOrchestrator(
            model, prompt_profile=_IdentityPromptProfile(), runtime_flags=_FLAGS_ROUND_TRIP_ONLY
        )

        text, events = await _collect_all(orchestrator.process_message("ask"))

        assert text == "Text  more."
        assert not any(isinstance(e, QuoteReferenceEvent) for e in events)

    @pytest.mark.asyncio
    async def test_no_source_citation_event_when_no_valid_search_quotes(self) -> None:
        model = _FakeChatModel(turns=[(["plain answer"], [])])
        orchestrator = ChatOrchestrator(
            model, prompt_profile=_IdentityPromptProfile(), runtime_flags=_FLAGS_INLINE_ONLY
        )

        _, events = await _collect_all(orchestrator.process_message("ask"))

        assert not any(isinstance(e, SourceCitationEvent) for e in events)

    @pytest.mark.asyncio
    async def test_quote_reference_events_emitted_in_stream_order(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        search_tool = _FakeTool("search_documents", result=_make_search_tool_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["before", quote, "after"], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_INLINE_ONLY,
        )

        all_events: list[ProcessEvent] = []
        async for event in orchestrator.process_message("ask"):
            all_events.append(event)

        inline_events = [e for e in all_events if isinstance(e, (str, QuoteReferenceEvent))]
        expected_ref = QuoteReferenceEvent(reference_number=1, canonical_key="search:s1:doc.txt:c1")
        assert inline_events == ["before", expected_ref, "after"]


# ---------------------------------------------------------------------------
# WP6: Migration and Legacy Path Sunset
# ---------------------------------------------------------------------------


class TestWP6FlagGatedBehavior:
    """WP6: verify that both flows can be toggled and that inline flow runs
    end-to-end without invoking the citation round-trip."""

    @pytest.mark.asyncio
    async def test_inline_only_flag_produces_no_citation_pass(self) -> None:
        """With inline_quotes_enabled=True, citation_round_trip_enabled=False:
        search results are cited inline and no citation pass is invoked."""
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        search_tool = _FakeTool("search_documents", result=_make_search_tool_result())
        # CitationTool is intentionally NOT registered — mirrors the new production default.
        model = _FakeChatModel(turns=[([], [search_tc]), (["Answer.", quote], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_INLINE_ONLY,
        )

        text, events = await _collect_all(orchestrator.process_message("find info"))

        assert text == "Answer."
        # Exactly two model stream calls — no citation pass step.
        assert len(model.stream_calls) == 2
        citation_events = [e for e in events if isinstance(e, SourceCitationEvent)]
        assert len(citation_events) == 1
        assert citation_events[0].validated[0].source == "doc.txt"

    @pytest.mark.asyncio
    async def test_round_trip_only_flag_triggers_citation_pass(self) -> None:
        """With inline_quotes_enabled=False, citation_round_trip_enabled=True:
        the legacy citation pass fires as before WP4."""
        chunk = SourceChunk(content="body", source="doc.txt", score=0.9, chunk_id="c1")
        cite_event = SourceCitationEvent(validated=(chunk,))
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        cite_tc = ToolCallInfo(
            name="cite_sources",
            arguments={"citations": [{"source": "doc.txt", "chunk_id": "c1"}]},
            call_id="c1",
        )
        search_tool = _FakeTool("search_documents", result=_make_search_tool_result())
        cite_tool = _FakeTool(
            "cite_sources",
            result={"validated": [{"source": "doc.txt", "chunk_id": "c1"}], "unvalidated": []},
            events=[cite_event],
        )
        model = _FakeChatModel(
            turns=[
                ([], [search_tc]),
                (["Based on results."], []),
                ([], [cite_tc]),
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool, cite_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_ROUND_TRIP_ONLY,
        )

        text, events = await _collect_all(orchestrator.process_message("find info"))

        assert text == "Based on results."
        # Three stream calls: search step + final text + citation pass.
        assert len(model.stream_calls) == 3
        citation_events = [e for e in events if isinstance(e, SourceCitationEvent)]
        assert len(citation_events) == 1

    @pytest.mark.asyncio
    async def test_both_flags_disabled_yields_no_citations(self) -> None:
        """With both flags off: no inline quotes and no citation pass — search result
        is consumed but no citation event is produced."""
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        search_tool = _FakeTool("search_documents", result=_make_search_tool_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["Answer."], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_NEITHER,
        )

        text, events = await _collect_all(orchestrator.process_message("find info"))

        assert text == "Answer."
        assert len(model.stream_calls) == 2
        assert not any(isinstance(e, SourceCitationEvent) for e in events)

    @pytest.mark.asyncio
    async def test_inline_only_no_search_results_produces_no_citations(self) -> None:
        """Inline flow with no retrieval turn: no citation event, no citation pass."""
        model = _FakeChatModel(turns=[(["Plain conversational answer."], [])])
        orchestrator = ChatOrchestrator(
            model,
            prompt_profile=_IdentityPromptProfile(),
            runtime_flags=_FLAGS_INLINE_ONLY,
        )

        text, events = await _collect_all(orchestrator.process_message("how are you?"))

        assert text == "Plain conversational answer."
        assert len(model.stream_calls) == 1
        assert not any(isinstance(e, SourceCitationEvent) for e in events)
        assert not any(isinstance(e, QuoteReferenceEvent) for e in events)
