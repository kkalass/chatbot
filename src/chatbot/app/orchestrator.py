"""Chat orchestration: conversation history and agentic tool-call loop.

The orchestrator is the single entry point for the UI layer.  It depends
exclusively on Protocol interfaces — no concrete infrastructure is imported here.

Document retrieval is modelled as a regular tool: the LLM decides *if* and
*when* to call ``search_documents`` and formulates the query itself.  This
enables multi-hop retrieval and query reformulation from conversation context.

Each turn runs the same loop regardless of which tools are registered:
stream from the model, yield text chunks to the caller, execute any tool calls
that arrive at the end of the stream, and repeat until the model produces a
plain-text response.  After the main loop a citation pass is triggered when
search results were consumed during the turn — the model is asked to call
``cite_sources`` so grounding provenance is captured as a
:class:`~src.chatbot.app.protocols.SourceCitationEvent`.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime

import structlog
from opentelemetry import trace

from src.chatbot.app.prompts import DEFAULT_PROMPTS
from src.chatbot.app.protocols import (
    ChatMessage,
    ChatModel,
    JsonObject,
    ProcessEvent,
    PromptProfile,
    SourceCitationEvent,
    Tool,
    ToolCallInfo,
    ToolContext,
    ToolEvent,
    ToolSchema,
)
from src.chatbot.observability import to_attribute_text

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

_MAX_TOOL_ROUNDS = 10  # safety limit to prevent infinite agentic loops
_SEARCH_TOOL_NAME = "search_documents"
_CITATION_TOOL_NAME = "cite_sources"


class ChatOrchestrator:
    """Manages per-session conversation history and the agentic tool-call loop.

    Constructed once per chat session and stored in session-scoped state.
    All dependencies are injected at construction time; no infrastructure is
    instantiated internally.

    Args:
        model: Chat model backend (Protocol).
        tools: Tool implementations registered for dispatch and advertised to
            the model.
        prompt_profile: Model-specific prompt profile. Used once at construction
            time to derive adjusted prompts and adjusted tool schemas.
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        prompt_profile: PromptProfile,
        tools: list[Tool] | None = None,
    ) -> None:
        self._model = model
        self._prompts = prompt_profile.adjust_prompts(DEFAULT_PROMPTS)
        _tools = tools or []
        self._tool_map: dict[str, Tool] = {t.schema.name: t for t in _tools}
        adjusted_schemas = [
            _adjust_tool_schema(t.schema, prompt_profile=prompt_profile) for t in _tools
        ]
        self._tool_schemas: list[ToolSchema] | None = adjusted_schemas if adjusted_schemas else None
        self._tool_schema_map: dict[str, ToolSchema] = {s.name: s for s in adjusted_schemas}
        self._history: list[ChatMessage] = []
        # Session-scoped flag: once search results exist in history, citation fallback
        # can be considered on subsequent turns as well.
        self._has_search_results_in_session = False

    def process_message(self, user_text: str) -> AsyncIterator[ProcessEvent]:
        """Process *user_text* and return an async iterator of :data:`~src.chatbot.app.protocols.ProcessEvent` items.

        Appends the user message to history eagerly (before iteration begins),
        then returns a lazy async iterator that runs the agentic loop.  Tool
        result messages are appended to history during tool rounds but are not
        yielded to the caller — only text chunks (``str``) and
        :class:`~src.chatbot.app.protocols.ToolEvent` items are streamed back.

        Args:
            user_text: Raw message text from the user.

        Returns:
            An async iterator yielding :data:`ProcessEvent` items —
            ``str`` for streaming text and :class:`ToolEvent` subclasses for
            structured metadata (e.g. :class:`~src.chatbot.app.protocols.SourceCitationEvent`).
        """
        self._history.append(ChatMessage(role="user", content=user_text))
        history = self._history
        model = self._model
        tool_schemas = self._tool_schemas
        tool_schema_map = self._tool_schema_map
        tool_map = self._tool_map
        prompts = self._prompts

        async def _gen() -> AsyncGenerator[ProcessEvent, None]:
            emitted_citation_events: list[SourceCitationEvent] = []
            with tracer.start_as_current_span("chat.orchestrator.process_message") as turn_span:
                turn_span.set_attribute("chat.user_message.length", len(user_text))
                turn_span.set_attribute("chat.user_message.preview", to_attribute_text(user_text))
                turn_span.set_attribute("chat.tool_count", len(tool_map))

                for round_num in range(_MAX_TOOL_ROUNDS):
                    with tracer.start_as_current_span("chat.orchestrator.round") as round_span:
                        round_span.set_attribute("chat.round", round_num)
                        system_text = prompts.system_prompt(datetime.now(tz=UTC))
                        messages = [ChatMessage(role="system", content=system_text), *history]
                        collected: list[str] = []
                        tool_calls: list[ToolCallInfo] = []

                        async for item in model.stream(messages, tools=tool_schemas):
                            if isinstance(item, str):
                                collected.append(item)
                                yield item
                            else:
                                tool_calls.extend(item)

                        round_span.set_attribute("chat.round.text_chars", len("".join(collected)))
                        round_span.set_attribute("chat.round.tool_call_count", len(tool_calls))
                        round_span.set_attribute(
                            "chat.round.tool_calls",
                            to_attribute_text([tc.name for tc in tool_calls]),
                        )

                        if not tool_calls:
                            history.append(
                                ChatMessage(role="assistant", content="".join(collected))
                            )
                            break

                        logger.info(
                            "orchestrator.tool_round",
                            round=round_num,
                            calls=[tc.name for tc in tool_calls],
                        )
                        history.append(
                            ChatMessage(
                                role="assistant",
                                content="".join(collected),
                                tool_calls=tuple(tool_calls),
                            )
                        )
                        if any(tc.name == _SEARCH_TOOL_NAME for tc in tool_calls):
                            self._has_search_results_in_session = True
                        async for event in _dispatch_tool_calls(tool_calls, history, tool_map):
                            yield event
                            match event:
                                case SourceCitationEvent():
                                    emitted_citation_events.append(event)
                else:
                    # Safety: exceeded max rounds — one final response without tools.
                    logger.warning("orchestrator.max_tool_rounds_exceeded", limit=_MAX_TOOL_ROUNDS)
                    system_text = prompts.system_prompt(datetime.now(tz=UTC))
                    messages = [ChatMessage(role="system", content=system_text), *history]
                    final: list[str] = []
                    async for item in model.stream(messages, tools=None):
                        if isinstance(item, str):
                            final.append(item)
                            yield item
                    history.append(ChatMessage(role="assistant", content="".join(final)))

                # Citation pass: triggered only when the main loop consumed search results
                # AND no citation event was already emitted (e.g. model called cite_sources
                # voluntarily during the main loop).
                if not emitted_citation_events and self._has_search_results_in_session:
                    with tracer.start_as_current_span("chat.orchestrator.citation_pass") as span:
                        citation_tool = tool_map.get(_CITATION_TOOL_NAME)
                        citation_schema = tool_schema_map.get(_CITATION_TOOL_NAME)
                        span.set_attribute(
                            "chat.citation_tool_available",
                            citation_tool is not None and citation_schema is not None,
                        )
                        if citation_tool is not None and citation_schema is not None:
                            citation_schemas: list[ToolSchema] = [citation_schema]
                            history.append(
                                ChatMessage(role="user", content=prompts.citation_fallback_message)
                            )
                            system_text = prompts.system_prompt(datetime.now(tz=UTC))
                            messages = [ChatMessage(role="system", content=system_text), *history]
                            cite_calls: list[ToolCallInfo] = []
                            cite_text: list[str] = []

                            async for item in model.stream(messages, tools=citation_schemas):
                                if isinstance(item, str):
                                    cite_text.append(item)
                                else:
                                    cite_calls.extend(item)

                            span.set_attribute(
                                "chat.citation_pass.tool_call_count", len(cite_calls)
                            )
                            span.set_attribute(
                                "chat.citation_pass.tool_calls",
                                to_attribute_text([tc.name for tc in cite_calls]),
                            )
                            span.set_attribute(
                                "chat.citation_pass.text_preview",
                                to_attribute_text("".join(cite_text)),
                            )

                            if not cite_calls:
                                logger.warning("orchestrator.citation_pass_no_tool_call")
                                history.append(
                                    ChatMessage(role="assistant", content="".join(cite_text))
                                )
                                return

                            history.append(
                                ChatMessage(
                                    role="assistant",
                                    content="".join(cite_text),
                                    tool_calls=tuple(cite_calls),
                                )
                            )

                            async for event in _dispatch_tool_calls(cite_calls, history, tool_map):
                                yield event

                turn_span.set_attribute("chat.citation_events", len(emitted_citation_events))
                turn_span.set_attribute("chat.history_entries", len(history))

        return _gen()


def _adjust_tool_schema(schema: ToolSchema, *, prompt_profile: PromptProfile) -> ToolSchema:
    """Apply model-specific prompt-profile adaptations to a tool schema once."""
    return ToolSchema(
        name=schema.name,
        description=prompt_profile.adjust_tool_description(schema.name, schema.description),
        parameters_schema=prompt_profile.adjust_parameter_schema(
            schema.name, schema.parameters_schema
        ),
    )


def _dispatch_tool_calls(
    tool_calls: list[ToolCallInfo],
    history: list[ChatMessage],
    tool_map: dict[str, Tool],
) -> AsyncIterator[ToolEvent]:
    """Dispatch *tool_calls*, append tool results to history, and stream tool events."""

    async def _gen() -> AsyncGenerator[ToolEvent, None]:
        for tc in tool_calls:
            context = ToolContext(history=tuple(history))
            result, events = await _dispatch(tc, tool_map, context)
            for event in events:
                yield event
            history.append(ChatMessage(role="tool", content=result, tool_call_id=tc.call_id))

    return _gen()


async def _dispatch(
    tc: ToolCallInfo,
    tool_map: dict[str, Tool],
    context: ToolContext,
) -> tuple[JsonObject, list[ToolEvent]]:
    """Look up the named tool, execute it, and return its structured result and events."""
    with tracer.start_as_current_span("chat.orchestrator.tool_dispatch") as span:
        span.set_attribute("chat.tool.name", tc.name)
        span.set_attribute("chat.tool.call_id", tc.call_id)
        span.set_attribute("chat.tool.arguments", to_attribute_text(tc.arguments))

        tool = tool_map.get(tc.name)
        if tool is None:
            logger.error("orchestrator.unknown_tool", name=tc.name)
            span.set_attribute("chat.tool.error", True)
            return {"error": f"unknown tool '{tc.name}'"}, []
        try:
            result, events = await tool.execute(tc.arguments, context)
            span.set_attribute("chat.tool.result", to_attribute_text(result))
            span.set_attribute("chat.tool.events", len(events))
            return result, events
        except Exception as exc:
            logger.exception("orchestrator.tool_error", name=tc.name, exc=str(exc))
            span.record_exception(exc)
            span.set_attribute("chat.tool.error", True)
            return {"error": f"Tool '{tc.name}' raised an error: {exc}"}, []
