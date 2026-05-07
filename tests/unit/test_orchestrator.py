# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :class:`ChatOrchestrator`: per-turn ref numbering, dispatch, fallbacks."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import pytest

from src.chatbot.app.citation import CitationMessage
from src.chatbot.app.citation.citation_model import CitationStreamItem
from src.chatbot.app.citation.messages import (
    CitationAssistantMessage,
    CitationSystemMessage,
    CitationToolMessage,
    CitationUserMessage,
)
from src.chatbot.app.orchestrator import ChatOrchestrator, ToolCallFinished, ToolCallStarted
from src.chatbot.contracts.chat import ModelProfile, ToolCallInfo
from src.chatbot.contracts.citation import (
    Citation,
    DocumentCitation,
    HallucinatedCitation,
    NumberedCitation,
    RawCitation,
)
from src.chatbot.contracts.i18n import I18nMessage, JsonObject
from src.chatbot.contracts.tools import ToolSchema

# --- Stubs ------------------------------------------------------------------


@dataclass(frozen=True)
class _IdentityProfile:
    @property
    def parse_text_tool_calls(self) -> bool:
        return False

    def adjust_prompts(self, prompts: object) -> object:
        return prompts

    def adjust_tool_description(self, tool_name: str, description: str) -> str:
        return description

    def adjust_parameter_schema(self, tool_name: str, schema: JsonObject) -> JsonObject:
        return schema


class _StubCitationModel:
    """Minimal CitationModel stand-in honouring the surface used by the orchestrator."""

    def __init__(self, scripted_streams: list[list[CitationStreamItem]]) -> None:
        self._streams = scripted_streams
        self._index = 0
        self.received_histories: list[list[CitationMessage]] = []
        self.received_tool_lists: list[Sequence[ToolSchema] | None] = []

    def make_system_message(self, base_prompt: str) -> CitationSystemMessage:
        return CitationSystemMessage(llm_content=base_prompt)

    def make_user_message(self, user_text: str) -> CitationUserMessage:
        return CitationUserMessage(llm_content=user_text)

    def make_assistant_message(
        self,
        parts: Sequence[str | Citation | HallucinatedCitation],
        *,
        tool_calls: Sequence[ToolCallInfo] | None = None,
    ) -> CitationAssistantMessage:
        return CitationAssistantMessage(
            parts=tuple(parts),
            llm_content="".join(p if isinstance(p, str) else "" for p in parts),
            tool_calls=tuple(tool_calls) if tool_calls else None,
        )

    def make_tool_message(self, call_id: str, name: str, result: JsonObject) -> CitationToolMessage:
        return CitationToolMessage(
            tool_call_id=call_id, tool_name=name, result=result, llm_content=""
        )

    def make_blocked_tool_response(self, tc: ToolCallInfo) -> CitationToolMessage:
        return CitationToolMessage(
            tool_call_id=tc.call_id,
            tool_name=tc.name,
            result={},
            llm_content="[BLOCKED]",
        )

    def make_loop_escape_message(self, original_user_content: str) -> CitationUserMessage:
        return CitationUserMessage(llm_content=f"[ESCAPE]{original_user_content}")

    def make_max_steps_escape_message(self, original_user_content: str) -> CitationUserMessage:
        return CitationUserMessage(llm_content=f"[MAXSTEPS]{original_user_content}")

    def stream(
        self,
        history: Sequence[CitationMessage],
        *,
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[CitationStreamItem]:
        self.received_histories.append(list(history))
        self.received_tool_lists.append(tools)
        items = self._streams[self._index]
        self._index += 1

        async def _gen() -> AsyncIterator[CitationStreamItem]:
            for item in items:
                yield item

        return _gen()


class _StubTool:
    def __init__(self, name: str, *, result: JsonObject) -> None:
        self.schema = ToolSchema(name=name, description="d", parameters_schema={"type": "object"})
        self.display_name = I18nMessage(key="stub.tool", args={})
        self._result = result
        self.calls: list[JsonObject] = []

    def describe_call(self, args: JsonObject) -> I18nMessage:
        return I18nMessage(key="stub.call", args=dict(args))

    async def execute(self, args: JsonObject) -> JsonObject:
        self.calls.append(args)
        return self._result


def _doc_citation(*, marker: str = "[M]", chunk_id: str = "c1") -> DocumentCitation:
    return DocumentCitation(
        raw_marker_text=marker,
        citation_token=chunk_id,
        source="s",
        chunk_id=chunk_id,
        content="x",
        score=1.0,
    )


def _profile() -> ModelProfile:
    return _IdentityProfile()  # type: ignore[return-value]


# --- Tests ------------------------------------------------------------------


class TestSingleTurn:
    @pytest.mark.asyncio
    async def test_yields_streamed_text_and_appends_history(self) -> None:
        layer = _StubCitationModel([["hello ", "world"]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        events = [e async for e in orch.process_message("hi")]

        assert events == ["hello ", "world"]

    @pytest.mark.asyncio
    async def test_assigns_sequential_ref_numbers(self) -> None:
        c1 = _doc_citation(chunk_id="c1")
        c2 = _doc_citation(chunk_id="c2")
        layer = _StubCitationModel([["a", c1, "b", c2]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        events = [e async for e in orch.process_message("hi")]
        numbered = [e for e in events if isinstance(e, NumberedCitation)]

        assert [n.reference_number for n in numbered] == [1, 2]

    @pytest.mark.asyncio
    async def test_reuses_ref_number_for_same_canonical_key(self) -> None:
        c1a = _doc_citation(chunk_id="c1")
        c1b = _doc_citation(chunk_id="c1")  # same canonical key
        layer = _StubCitationModel([[c1a, " then ", c1b]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        events = [e async for e in orch.process_message("hi")]
        numbered = [e for e in events if isinstance(e, NumberedCitation)]

        assert [n.reference_number for n in numbered] == [1, 1]

    @pytest.mark.asyncio
    async def test_passes_through_hallucinated_citation(self) -> None:
        h = HallucinatedCitation(
            raw=RawCitation(ref="missing", raw_marker_text="<m>"),
            reason="x",
        )
        layer = _StubCitationModel([["a", h, "b"]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        events = [e async for e in orch.process_message("hi")]
        assert h in events


class TestToolDispatchLoop:
    @pytest.mark.asyncio
    async def test_dispatches_then_continues_to_next_step(self) -> None:
        tc = ToolCallInfo(call_id="cid1", name="vac", arguments={"year": 2026})
        tool = _StubTool("vac", result={"days": 30})
        layer = _StubCitationModel(
            [
                ["thinking ", [tc]],
                ["final answer"],
            ]
        )
        orch = ChatOrchestrator(
            layer,  # type: ignore[arg-type]
            model_profile=_profile(),
            tools=[tool],
        )

        events = [e async for e in orch.process_message("how many days?")]

        assert "thinking " in events
        assert "final answer" in events
        assert tool.calls == [{"year": 2026}]

    @pytest.mark.asyncio
    async def test_emits_tool_call_started_and_finished(self) -> None:
        tc = ToolCallInfo(call_id="cid1", name="vac", arguments={"year": 2026})
        tool = _StubTool("vac", result={"days": 30})
        layer = _StubCitationModel(
            [
                [[tc]],
                ["final answer"],
            ]
        )
        orch = ChatOrchestrator(
            layer,  # type: ignore[arg-type]
            model_profile=_profile(),
            tools=[tool],
        )

        events = [e async for e in orch.process_message("how many days?")]

        started = [e for e in events if isinstance(e, ToolCallStarted)]
        finished = [e for e in events if isinstance(e, ToolCallFinished)]
        assert len(started) == 1
        assert started[0].tool_name == "vac"
        assert started[0].call_id == "cid1"
        assert started[0].call_description == I18nMessage(key="stub.call", args={"year": 2026})
        assert len(finished) == 1
        assert finished[0].tool_name == "vac"
        assert finished[0].call_id == "cid1"
        assert finished[0].result == {"days": 30}
        # Started must precede Finished in the event stream.
        assert events.index(started[0]) < events.index(finished[0])
        # Second stream sees the tool result in history
        second_history = layer.received_histories[1]
        assert any(isinstance(m, CitationToolMessage) for m in second_history)


class TestRepeatedToolCallSafety:
    @pytest.mark.asyncio
    async def test_repeated_signature_triggers_fallback_without_tools(self) -> None:
        tc1 = ToolCallInfo(call_id="cid1", name="vac", arguments={"year": 2026})
        tc2 = ToolCallInfo(call_id="cid2", name="vac", arguments={"year": 2026})
        tool = _StubTool("vac", result={"days": 30})
        layer = _StubCitationModel(
            [
                [[tc1]],
                [[tc2]],
                ["fallback answer"],
            ]
        )
        orch = ChatOrchestrator(
            layer,  # type: ignore[arg-type]
            model_profile=_profile(),
            tools=[tool],
        )

        events = [e async for e in orch.process_message("hi")]

        assert "fallback answer" in events
        # Final stream should have been requested without tools.
        assert layer.received_tool_lists[-1] is None

    @pytest.mark.asyncio
    async def test_history_has_no_pending_tool_calls_at_fallback_step(self) -> None:
        """The duplicate-detected assistant message must be followed by a blocked
        tool response before the fallback step, so the history ends at a valid
        tool-result boundary — never at an unanswered assistant tool-call."""
        tc1 = ToolCallInfo(call_id="cid1", name="vac", arguments={"year": 2026})
        tc2 = ToolCallInfo(call_id="cid2", name="vac", arguments={"year": 2026})
        tool = _StubTool("vac", result={"days": 30})
        layer = _StubCitationModel(
            [
                [[tc1]],
                [[tc2]],
                ["fallback answer"],
            ]
        )
        orch = ChatOrchestrator(
            layer,  # type: ignore[arg-type]
            model_profile=_profile(),
            tools=[tool],
        )

        events = [e async for e in orch.process_message("hi")]
        assert "fallback answer" in events

        # The last message passed to the fallback stream must not be an assistant
        # message with unanswered tool_calls. (An earlier assistant message with
        # tool_calls is fine, provided a tool result follows it.)
        final_history = layer.received_histories[-1]
        # Skip system message (index 0), inspect the rest.
        tail = [m for m in final_history if not isinstance(m, CitationSystemMessage)]
        assert tail, "History must not be empty"
        assert not (
            isinstance(tail[-1], CitationAssistantMessage) and tail[-1].tool_calls is not None
        ), "Last message in fallback-step history must not be an unanswered assistant tool-call."

    @pytest.mark.asyncio
    async def test_blocked_tool_response_appended_for_duplicate_calls(self) -> None:
        """A synthetic blocked tool-result must be appended for every pending call
        in the duplicate-detected step so the history ends at a tool-result."""
        tc1 = ToolCallInfo(call_id="cid1", name="vac", arguments={"year": 2026})
        tc2 = ToolCallInfo(call_id="cid2", name="vac", arguments={"year": 2026})
        tool = _StubTool("vac", result={"days": 30})
        layer = _StubCitationModel(
            [
                [[tc1]],
                [[tc2]],
                ["fallback answer"],
            ]
        )
        orch = ChatOrchestrator(
            layer,  # type: ignore[arg-type]
            model_profile=_profile(),
            tools=[tool],
        )

        events = [e async for e in orch.process_message("hi")]
        _ = events  # we care about history side-effects, not yielded events

        # The fallback step history must contain a blocked tool-result for cid2.
        final_history = layer.received_histories[-1]
        blocked = [
            m
            for m in final_history
            if isinstance(m, CitationToolMessage) and m.llm_content == "[BLOCKED]"
        ]
        assert len(blocked) >= 1, "Blocked tool-result message must appear in fallback-step history"
        assert blocked[-1].tool_call_id == "cid2"

    @pytest.mark.asyncio
    async def test_loop_escape_user_message_appended_after_blocked_responses(self) -> None:
        """After blocked tool responses, a loop-escape user message must be
        appended so the fallback step ends on a user turn — required by many
        LLMs to trigger user-facing text generation."""
        tc1 = ToolCallInfo(call_id="cid1", name="vac", arguments={"year": 2026})
        tc2 = ToolCallInfo(call_id="cid2", name="vac", arguments={"year": 2026})
        tool = _StubTool("vac", result={"days": 30})
        layer = _StubCitationModel(
            [
                [[tc1]],
                [[tc2]],
                ["fallback answer"],
            ]
        )
        orch = ChatOrchestrator(
            layer,  # type: ignore[arg-type]
            model_profile=_profile(),
            tools=[tool],
        )

        events = [e async for e in orch.process_message("the question")]
        _ = events

        # The fallback step history must end with the loop-escape user message.
        final_history = layer.received_histories[-1]
        tail = [m for m in final_history if not isinstance(m, CitationSystemMessage)]
        assert tail and isinstance(tail[-1], CitationUserMessage), (
            "Fallback-step history must end with a user message"
        )
        assert tail[-1].llm_content.startswith("[ESCAPE]"), (
            "Last message must be the loop-escape user message"
        )

    @pytest.mark.asyncio
    async def test_max_steps_escape_user_message_appended_before_final_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the agentic loop hits its safety limit, a max-steps escape user
        message must be appended before the forced no-tools final step. Without
        it the history would end on a ``tool`` role and many LLMs emit empty
        completions."""
        # Shrink the limit so the test exercises the path quickly.
        from src.chatbot.app import orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "_MAX_TOOL_STEPS", 2)

        # Each step except the final one produces a fresh, non-repeating tool
        # call so the loop-detection path is not triggered.
        tc1 = ToolCallInfo(call_id="cid1", name="vac", arguments={"year": 2026})
        tc2 = ToolCallInfo(call_id="cid2", name="vac", arguments={"year": 2027})
        tool = _StubTool("vac", result={"days": 30})
        layer = _StubCitationModel(
            [
                [[tc1]],
                [[tc2]],
                ["final answer after max steps"],
            ]
        )
        orch = ChatOrchestrator(
            layer,  # type: ignore[arg-type]
            model_profile=_profile(),
            tools=[tool],
        )

        _ = [e async for e in orch.process_message("the question")]

        final_history = layer.received_histories[-1]
        tail = [m for m in final_history if not isinstance(m, CitationSystemMessage)]
        assert tail and isinstance(tail[-1], CitationUserMessage), (
            "Final-step history must end with a user message"
        )
        assert tail[-1].llm_content.startswith("[MAXSTEPS]"), (
            "Last message must be the max-steps escape user message"
        )
        # The forced final step must not advertise any tools.
        assert layer.received_tool_lists[-1] is None


class TestSessionStableNumbering:
    """Reference numbers persist across turns within the same orchestrator session."""

    @pytest.mark.asyncio
    async def test_same_chunk_reuses_number_across_turns(self) -> None:
        c1 = _doc_citation(chunk_id="c1")
        layer = _StubCitationModel([[c1], [c1]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        events1 = [e async for e in orch.process_message("first question")]
        events2 = [e async for e in orch.process_message("second question")]

        num1 = [e for e in events1 if isinstance(e, NumberedCitation)]
        num2 = [e for e in events2 if isinstance(e, NumberedCitation)]
        assert num1[0].reference_number == 1
        assert num2[0].reference_number == 1, "Same chunk must reuse ref [1] in turn 2"

    @pytest.mark.asyncio
    async def test_new_chunk_in_turn2_gets_next_number(self) -> None:
        c1 = _doc_citation(chunk_id="c1")
        c2 = _doc_citation(chunk_id="c2")
        layer = _StubCitationModel([[c1], [c2]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        events1 = [e async for e in orch.process_message("first question")]
        events2 = [e async for e in orch.process_message("second question")]

        num1 = [e for e in events1 if isinstance(e, NumberedCitation)]
        num2 = [e for e in events2 if isinstance(e, NumberedCitation)]
        assert num1[0].reference_number == 1
        assert num2[0].reference_number == 2, "New chunk in turn 2 must get ref [2]"

    @pytest.mark.asyncio
    async def test_mixed_reuse_and_new_in_turn2(self) -> None:
        c1 = _doc_citation(chunk_id="c1")
        c2 = _doc_citation(chunk_id="c2")
        layer = _StubCitationModel([[c1], [c1, c2]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        await orch.process_message("turn 1").__anext__()
        # drain turn 1
        async for _ in orch.process_message("turn 1"):
            pass
        # Re-create with fresh layer to avoid index exhaustion
        layer2 = _StubCitationModel([[c1], [c1, c2]])
        orch2 = ChatOrchestrator(layer2, model_profile=_profile())  # type: ignore[arg-type]

        events1 = [e async for e in orch2.process_message("turn 1")]
        events2 = [e async for e in orch2.process_message("turn 2")]

        num1 = [e for e in events1 if isinstance(e, NumberedCitation)]
        num2 = [e for e in events2 if isinstance(e, NumberedCitation)]
        assert num1 == [NumberedCitation(reference_number=1, citation=c1)]
        assert num2[0].reference_number == 1, "c1 reuses ref [1]"
        assert num2[1].reference_number == 2, "c2 gets new ref [2]"
