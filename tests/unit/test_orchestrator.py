"""Tests for :class:`ChatOrchestrator`: per-turn ref numbering, dispatch, fallbacks."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import pytest

from src.chatbot.app.citation import (
    Citation,
    CitationLayerMessage,
    DocumentCitation,
    HallucinatedCitation,
    NumberedCitation,
    RawCitation,
)
from src.chatbot.app.citation.layer import CitationLayerStreamItem
from src.chatbot.app.citation.messages import (
    CitationLayerAssistantMessage,
    CitationLayerSystemMessage,
    CitationLayerToolMessage,
    CitationLayerUserMessage,
)
from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.app.protocols import (
    JsonObject,
    ModelProfile,
    ToolCallInfo,
    ToolSchema,
)

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


class _StubCitationLayer:
    """Minimal CitationLayer stand-in honouring the surface used by the orchestrator."""

    def __init__(self, scripted_streams: list[list[CitationLayerStreamItem]]) -> None:
        self._streams = scripted_streams
        self._index = 0
        self.received_histories: list[list[CitationLayerMessage]] = []
        self.received_tool_lists: list[Sequence[ToolSchema] | None] = []

    def make_system_message(self, base_prompt: str) -> CitationLayerSystemMessage:
        return CitationLayerSystemMessage(llm_content=base_prompt)

    def make_user_message(self, user_text: str) -> CitationLayerUserMessage:
        return CitationLayerUserMessage(llm_content=user_text)

    def make_assistant_message(
        self,
        parts: Sequence[str | Citation | HallucinatedCitation],
        *,
        tool_calls: Sequence[ToolCallInfo] | None = None,
    ) -> CitationLayerAssistantMessage:
        return CitationLayerAssistantMessage(
            parts=tuple(parts),
            llm_content="".join(p if isinstance(p, str) else "" for p in parts),
            tool_calls=tuple(tool_calls) if tool_calls else None,
        )

    def make_tool_message(
        self, call_id: str, name: str, result: JsonObject
    ) -> CitationLayerToolMessage:
        return CitationLayerToolMessage(
            tool_call_id=call_id, tool_name=name, result=result, llm_content=""
        )

    def stream(
        self,
        history: Sequence[CitationLayerMessage],
        *,
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[CitationLayerStreamItem]:
        self.received_histories.append(list(history))
        self.received_tool_lists.append(tools)
        items = self._streams[self._index]
        self._index += 1

        async def _gen() -> AsyncIterator[CitationLayerStreamItem]:
            for item in items:
                yield item

        return _gen()


class _StubTool:
    def __init__(self, name: str, *, result: JsonObject) -> None:
        self.schema = ToolSchema(name=name, description="d", parameters_schema={"type": "object"})
        self._result = result
        self.calls: list[JsonObject] = []

    async def execute(self, args: JsonObject) -> JsonObject:
        self.calls.append(args)
        return self._result


def _doc_citation(*, marker: str = "[M]", chunk_id: str = "c1") -> DocumentCitation:
    return DocumentCitation(
        raw_marker_text=marker,
        tool_call_id="tc1",
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
        layer = _StubCitationLayer([["hello ", "world"]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        events = [e async for e in orch.process_message("hi")]

        assert events == ["hello ", "world"]

    @pytest.mark.asyncio
    async def test_assigns_sequential_ref_numbers(self) -> None:
        c1 = _doc_citation(chunk_id="c1")
        c2 = _doc_citation(chunk_id="c2")
        layer = _StubCitationLayer([["a", c1, "b", c2]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        events = [e async for e in orch.process_message("hi")]
        numbered = [e for e in events if isinstance(e, NumberedCitation)]

        assert [n.reference_number for n in numbered] == [1, 2]

    @pytest.mark.asyncio
    async def test_reuses_ref_number_for_same_canonical_key(self) -> None:
        c1a = _doc_citation(chunk_id="c1")
        c1b = _doc_citation(chunk_id="c1")  # same canonical key
        layer = _StubCitationLayer([[c1a, " then ", c1b]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        events = [e async for e in orch.process_message("hi")]
        numbered = [e for e in events if isinstance(e, NumberedCitation)]

        assert [n.reference_number for n in numbered] == [1, 1]

    @pytest.mark.asyncio
    async def test_passes_through_hallucinated_citation(self) -> None:
        h = HallucinatedCitation(
            raw=RawCitation(tool_call_id="missing", raw_marker_text="<m>"),
            reason="x",
        )
        layer = _StubCitationLayer([["a", h, "b"]])
        orch = ChatOrchestrator(layer, model_profile=_profile())  # type: ignore[arg-type]

        events = [e async for e in orch.process_message("hi")]
        assert h in events


class TestToolDispatchLoop:
    @pytest.mark.asyncio
    async def test_dispatches_then_continues_to_next_step(self) -> None:
        tc = ToolCallInfo(call_id="cid1", name="vac", arguments={"year": 2026})
        tool = _StubTool("vac", result={"days": 30})
        layer = _StubCitationLayer(
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
        # Second stream sees the tool result in history
        second_history = layer.received_histories[1]
        assert any(isinstance(m, CitationLayerToolMessage) for m in second_history)


class TestRepeatedToolCallSafety:
    @pytest.mark.asyncio
    async def test_repeated_signature_triggers_fallback_without_tools(self) -> None:
        tc1 = ToolCallInfo(call_id="cid1", name="vac", arguments={"year": 2026})
        tc2 = ToolCallInfo(call_id="cid2", name="vac", arguments={"year": 2026})
        tool = _StubTool("vac", result={"days": 30})
        layer = _StubCitationLayer(
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
