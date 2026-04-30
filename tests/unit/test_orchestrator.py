"""Unit tests for ChatOrchestrator in inline-quote-only mode."""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import replace
from datetime import datetime

import pytest

from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.app.prompts import DEFAULT_PROMPTS, Prompts
from src.chatbot.app.protocols import (
    ChatMessage,
    ChatStreamItem,
    JsonObject,
    ProcessEvent,
    PromptProfile,
    QuoteReferenceEvent,
    RawAssistantText,
    SearchResultQuote,
    SourceCitationEvent,
    ToolCallInfo,
    ToolCallQuote,
    ToolCitationEvent,
    ToolContext,
    ToolEvent,
    ToolSchema,
)


class _FakeChatModel:
    def __init__(
        self,
        turns: list[tuple[list[str | SearchResultQuote | ToolCallQuote], list[ToolCallInfo]]]
        | None = None,
    ) -> None:
        self.turns = turns or [(["default response"], [])]
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
            str | list[ToolCallInfo] | SearchResultQuote | ToolCallQuote,
            None,
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
        self.events = events or []
        self.calls: list[dict[str, object]] = []
        self.contexts: list[ToolContext] = []

    async def execute(
        self,
        args: dict[str, object],
        context: ToolContext,
    ) -> tuple[dict[str, object], list[ToolEvent]]:
        self.calls.append(dict(args))
        self.contexts.append(context)
        return self.result, list(self.events)


class _FakeRawTextChatModel:
    def __init__(
        self,
        turns: list[tuple[list[ChatStreamItem], list[ToolCallInfo]]] | None = None,
    ) -> None:
        self.turns = turns or [
            (["default response", RawAssistantText(text="default response")], [])
        ]
        self._turn_idx = 0
        self.stream_calls: list[list[ChatMessage]] = []
        self.stream_tools: list[list[ToolSchema] | None] = []

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ChatStreamItem]:
        self.stream_calls.append(list(messages))
        self.stream_tools.append(list(tools) if tools is not None else None)
        idx = min(self._turn_idx, len(self.turns) - 1)
        self._turn_idx += 1
        chunks, tool_calls = self.turns[idx]
        return self._gen(chunks, tool_calls)

    def _gen(
        self,
        chunks: list[ChatStreamItem],
        tool_calls: list[ToolCallInfo],
    ) -> AsyncIterator[ChatStreamItem]:
        async def _inner() -> AsyncGenerator[ChatStreamItem, None]:
            for chunk in chunks:
                yield chunk
            if tool_calls:
                yield tool_calls

        return _inner()


class _IdentityPromptProfile(PromptProfile):
    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        return prompts

    def adjust_tool_description(self, tool_name: str, description: str) -> str:
        return description

    def adjust_parameter_schema(self, tool_name: str, schema: JsonObject) -> JsonObject:
        return schema


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

        assert await _collect(orchestrator.process_message("Hi")) == "Hello, world!"

    @pytest.mark.asyncio
    async def test_system_prompt_is_first_message(self) -> None:
        model = _FakeChatModel(turns=[(["response"], [])])

        def _fixed_prompt(_dt: datetime) -> str:
            return "You are a test bot."

        prompts = replace(DEFAULT_PROMPTS, system_prompt=_fixed_prompt)
        orchestrator = ChatOrchestrator(
            model,
            prompt_profile=_IdentityPromptProfile(),
            prompts=prompts,
        )

        await _collect(orchestrator.process_message("Hi"))

        assert model.stream_calls[0][0].role == "system"
        assert model.stream_calls[0][0].content == "You are a test bot."

    @pytest.mark.asyncio
    async def test_current_user_turn_is_formatted_for_model_call_only(self) -> None:
        model = _FakeChatModel(turns=[(["response"], []), (["next response"], [])])

        def _format_user_message(user_text: str) -> str:
            return f"QUESTION:\n{user_text}\n\nRemember citations."

        prompts = replace(
            DEFAULT_PROMPTS,
            user_message=_format_user_message,
        )
        orchestrator = ChatOrchestrator(
            model,
            prompt_profile=_IdentityPromptProfile(),
            prompts=prompts,
        )

        await _collect(orchestrator.process_message("Hi"))

        assert model.stream_calls[0][1].role == "user"
        assert model.stream_calls[0][1].content == "QUESTION:\nHi\n\nRemember citations."

        await _collect(orchestrator.process_message("Next"))

        second_turn_user_messages = [
            message.content for message in model.stream_calls[1] if message.role == "user"
        ]
        assert second_turn_user_messages == [
            "Hi",
            "QUESTION:\nNext\n\nRemember citations.",
        ]


class TestAgenticLoop:
    @pytest.mark.asyncio
    async def test_executes_tool_then_continues_with_tool_result_in_history(self) -> None:
        tool = _FakeTool("echo", result={"value": "from_tool"})
        tc = ToolCallInfo(name="echo", arguments={"x": 1}, call_id="c1")
        model = _FakeChatModel(turns=[([], [tc]), (["Done."], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        text = await _collect(orchestrator.process_message("do thing"))

        assert text == "Done."
        assert tool.calls == [{"x": 1}]
        second_call_messages = model.stream_calls[1]
        tool_msgs = [m for m in second_call_messages if m.role == "tool"]
        assert tool_msgs and tool_msgs[-1].tool_call_id == "c1"

    @pytest.mark.asyncio
    async def test_follow_up_step_keeps_formatting_latest_user_turn_without_extra_message(
        self,
    ) -> None:
        tool = _FakeTool("echo", result={"value": "from_tool"})
        tc = ToolCallInfo(name="echo", arguments={"x": 1}, call_id="c1")

        def _format_user_message(user_text: str) -> str:
            return f"{user_text}\n\nCite tool-backed claims inline."

        prompts = replace(
            DEFAULT_PROMPTS,
            user_message=_format_user_message,
        )
        model = _FakeChatModel(turns=[([], [tc]), (["Done."], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
            prompts=prompts,
        )

        await _collect(orchestrator.process_message("do thing"))

        second_call_messages = model.stream_calls[1]
        user_messages = [m for m in second_call_messages if m.role == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].content == "do thing\n\nCite tool-backed claims inline."

    @pytest.mark.asyncio
    async def test_breaks_to_no_tools_fallback_on_repeated_equivalent_tool_calls(self) -> None:
        tool = _FakeTool("echo", result={"value": "from_tool"})
        first = ToolCallInfo(name="echo", arguments={"x": 1}, call_id="c1")
        repeated = ToolCallInfo(name="echo", arguments={"x": 1}, call_id="c2")
        model = _FakeChatModel(
            turns=[
                ([], [first]),
                ([], [repeated]),
                (["Fallback answer."], []),
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        text = await _collect(orchestrator.process_message("do thing"))

        assert text == "Fallback answer."
        assert tool.calls == [{"x": 1}]
        assert model.stream_tools[2] is None


class TestInlineQuotePipeline:
    def _search_result(self) -> dict[str, object]:
        return {
            "chunks": [
                {
                    "source": "doc.txt",
                    "chunk_id": "c1",
                    "content": "body",
                    "score": 0.9,
                    "title": "Doc Title",
                    "author": "A",
                    "publication_date": "2023-03-10",
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_valid_search_quote_emits_reference_and_source_citation(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        search_tool = _FakeTool("search_documents", result=self._search_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["Answer. ", quote, " end."], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        text, events = await _collect_all(orchestrator.process_message("ask"))

        assert text == "Answer.  end."
        ref_events = [e for e in events if isinstance(e, QuoteReferenceEvent)]
        assert len(ref_events) == 1
        assert ref_events[0].reference_number == 1
        citation_events = [e for e in events if isinstance(e, SourceCitationEvent)]
        assert len(citation_events) == 1
        assert citation_events[0].validated[0].source == "doc.txt"

    @pytest.mark.asyncio
    async def test_duplicate_quote_reuses_reference_number(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        search_tool = _FakeTool("search_documents", result=self._search_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["A", quote, "B", quote], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        _, events = await _collect_all(orchestrator.process_message("ask"))

        refs = [e for e in events if isinstance(e, QuoteReferenceEvent)]
        assert len(refs) == 2
        assert refs[0].reference_number == 1
        assert refs[1].reference_number == 1

    @pytest.mark.asyncio
    async def test_invalid_search_quote_is_dropped(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        bad_quote = SearchResultQuote(
            claim="claim",
            tool_call_id="s1",
            source="unknown.txt",
            chunk_id="bad",
        )
        search_tool = _FakeTool("search_documents", result=self._search_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["Text.", bad_quote], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        text, events = await _collect_all(orchestrator.process_message("ask"))

        assert text == "Text."
        assert not any(isinstance(e, QuoteReferenceEvent) for e in events)
        assert not any(isinstance(e, SourceCitationEvent) for e in events)

    @pytest.mark.asyncio
    async def test_search_quote_with_leading_slash_source_is_accepted(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(
            claim="claim",
            tool_call_id="get_search_results",
            source="/doc.txt",
            chunk_id="c1",
        )
        search_tool = _FakeTool("search_documents", result=self._search_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["Answer. ", quote, " end."], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        text, events = await _collect_all(orchestrator.process_message("ask"))

        assert text == "Answer.  end."
        ref_events = [e for e in events if isinstance(e, QuoteReferenceEvent)]
        assert len(ref_events) == 1
        citation_events = [e for e in events if isinstance(e, SourceCitationEvent)]
        assert len(citation_events) == 1
        assert citation_events[0].validated[0].source == "doc.txt"

    @pytest.mark.asyncio
    async def test_tool_call_quote_emits_reference_and_tool_citation(self) -> None:
        vacation_tc = ToolCallInfo(name="get_vacation_days", arguments={}, call_id="v1")
        quote = ToolCallQuote(
            tool_call_id="v1",
        )
        vacation_tool = _FakeTool("get_vacation_days", result={"remaining_days": 27})
        model = _FakeChatModel(turns=[([], [vacation_tc]), (["You have ", quote, " days."], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[vacation_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        text, events = await _collect_all(orchestrator.process_message("ask"))

        assert text == "You have  days."
        assert len([e for e in events if isinstance(e, QuoteReferenceEvent)]) == 1
        assert not any(isinstance(e, SourceCitationEvent) for e in events)
        tool_citation_events = [e for e in events if isinstance(e, ToolCitationEvent)]
        assert len(tool_citation_events) == 1
        assert tool_citation_events[0].tool_name == "get_vacation_days"
        assert tool_citation_events[0].result == {"remaining_days": 27}

    @pytest.mark.asyncio
    async def test_quote_reference_events_keep_stream_order(self) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        search_tool = _FakeTool("search_documents", result=self._search_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["before", quote, "after"], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        all_events: list[ProcessEvent] = []
        async for event in orchestrator.process_message("ask"):
            all_events.append(event)

        expected_ref = QuoteReferenceEvent(reference_number=1, canonical_key="search:s1:doc.txt:c1")
        inline_events = [e for e in all_events if isinstance(e, (str, QuoteReferenceEvent))]
        assert inline_events == ["before", expected_ref, "after"]

    @pytest.mark.asyncio
    async def test_follow_up_turn_receives_assistant_history_with_rendered_reference_tokens(
        self,
    ) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        search_tool = _FakeTool("search_documents", result=self._search_result())
        model = _FakeChatModel(
            turns=[
                ([], [search_tc]),
                (["Answer.", quote], []),
                (["Follow-up reply."], []),
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        await _collect(orchestrator.process_message("ask"))
        await _collect(orchestrator.process_message("next"))

        follow_up_messages = model.stream_calls[2]
        assistant_messages = [
            message.content
            for message in follow_up_messages
            if message.role == "assistant" and isinstance(message.content, str)
        ]
        assert "Answer.[1]" in assistant_messages

    @pytest.mark.asyncio
    async def test_follow_up_turn_prefers_raw_assistant_text_with_quote_markers_in_history(
        self,
    ) -> None:
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        quote = SearchResultQuote(claim="claim", tool_call_id="s1", source="doc.txt", chunk_id="c1")
        raw_answer = (
            "Answer."
            '<°_quote_°>{"kind":"search_result","tool_call_id":"s1",'
            '"source":"doc.txt","chunk_id":"c1"}</°_quote_°>'
        )
        search_tool = _FakeTool("search_documents", result=self._search_result())
        model = _FakeRawTextChatModel(
            turns=[
                ([], [search_tc]),
                (["Answer.", quote, RawAssistantText(text=raw_answer)], []),
                (["Follow-up reply.", RawAssistantText(text="Follow-up reply.")], []),
            ]
        )
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        await _collect(orchestrator.process_message("ask"))
        await _collect(orchestrator.process_message("next"))

        follow_up_messages = model.stream_calls[2]
        assistant_messages = [
            message.content
            for message in follow_up_messages
            if message.role == "assistant" and isinstance(message.content, str)
        ]
        assert raw_answer in assistant_messages

    @pytest.mark.asyncio
    async def test_search_chunks_are_rendered_as_markdown_for_model(self) -> None:
        """History keeps original JSON; model sees markdown with chunk_id adjacent to content."""
        search_tc = ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="s1")
        search_tool = _FakeTool("search_documents", result=self._search_result())
        model = _FakeChatModel(turns=[([], [search_tc]), (["Reply."], [])])
        orchestrator = ChatOrchestrator(
            model,
            tools=[search_tool],
            prompt_profile=_IdentityPromptProfile(),
        )

        await _collect(orchestrator.process_message("ask"))

        # Step 2 messages — model sees markdown, not raw JSON.
        step2_messages = model.stream_calls[1]
        tool_msgs = [m for m in step2_messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        tool_content = tool_msgs[0].content
        assert isinstance(tool_content, str), "model should receive str, not dict"
        assert "<search_results>" in tool_content
        assert (
            '<source title="Doc Title" source_path="doc.txt" year="2023" author="A">'
            in tool_content
        )
        assert '<chunk chunk_id="c1" page="unknown">' in tool_content
        assert "body" in tool_content
        assert "</search_results>" in tool_content
        # Original JSON dict must NOT be in the message sent to the model.
        assert "{" not in tool_content or "chunk_id" in tool_content  # no raw JSON keys

        # History must still carry the original JSON for citation_support.
        history_tool_msgs = [m for m in orchestrator._history if m.role == "tool"]  # pyright: ignore[reportPrivateUsage]
        assert len(history_tool_msgs) == 1
        assert isinstance(history_tool_msgs[0].content, dict)
        assert "chunks" in history_tool_msgs[0].content
