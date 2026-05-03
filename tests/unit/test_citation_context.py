# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the default :class:`CitationContext` adapter."""

from src.chatbot.app.citation import (
    CitationContext,
    build_citation_context,
)
from src.chatbot.app.citation.messages import (
    CitationLayerMessage,
    CitationLayerSystemMessage,
    CitationLayerToolMessage,
    CitationLayerUserMessage,
)


def _tool_msg(call_id: str, name: str, result: dict[str, object]) -> CitationLayerToolMessage:
    return CitationLayerToolMessage(
        tool_call_id=call_id,
        tool_name=name,
        result=result,
        llm_content="",
    )


class TestDefaultCitationContext:
    def test_returns_protocol_compatible_object(self) -> None:
        ctx = build_citation_context(())
        assert isinstance(ctx, CitationContext)

    def test_tool_result_for_returns_match(self) -> None:
        history: tuple[CitationLayerMessage, ...] = (
            CitationLayerSystemMessage(llm_content="sys"),
            _tool_msg("tc1", "search_documents", {"chunks": [{"id": 1}]}),
            CitationLayerUserMessage(llm_content="hi"),
            _tool_msg("tc2", "vacation_days", {"days_remaining": 12}),
        )
        ctx = build_citation_context(history)

        assert ctx.tool_result_for("tc1") == {"chunks": [{"id": 1}]}
        assert ctx.tool_result_for("tc2") == {"days_remaining": 12}

    def test_tool_result_for_returns_none_when_missing(self) -> None:
        ctx = build_citation_context((_tool_msg("tc1", "x", {"a": 1}),))
        assert ctx.tool_result_for("does-not-exist") is None

    def test_tool_results_for_returns_chronological_order(self) -> None:
        history: tuple[CitationLayerMessage, ...] = (
            _tool_msg("tc1", "search_documents", {"step": 1}),
            _tool_msg("tc2", "vacation_days", {"unrelated": True}),
            _tool_msg("tc3", "search_documents", {"step": 2}),
        )
        ctx = build_citation_context(history)

        results = list(ctx.tool_results_for("search_documents"))
        assert results == [{"step": 1}, {"step": 2}]

    def test_tool_results_for_empty_when_no_match(self) -> None:
        ctx = build_citation_context((_tool_msg("tc1", "other", {}),))
        assert list(ctx.tool_results_for("missing")) == []
