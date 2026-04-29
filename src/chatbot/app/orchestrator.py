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
search results were consumed during the turn. The citation pass uses a
dedicated prompt containing only (a) search result data from this turn and
(b) the final answer text, then asks the model to call ``cite_sources`` so
grounding provenance is captured as a
:class:`~src.chatbot.app.protocols.SourceCitationEvent`.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime

import structlog
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.chatbot.app.citation_support import (
    collect_search_chunks,
    parse_serialized_citation_tool_call,
    render_search_results_for_prompt,
)
from src.chatbot.app.prompts import DEFAULT_PROMPTS, Prompts
from src.chatbot.app.protocols import (
    ChatMessage,
    ChatModel,
    ChatStreamItem,
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
from src.chatbot.observability.openinference import (
    build_input_attributes,
    build_output_attributes,
    build_span_kind_attributes,
    build_tool_execution_attributes,
)
from src.chatbot.observability.schema import (
    SPAN_CHAT_ORCHESTRATOR_CITATION_PASS,
    SPAN_CHAT_ORCHESTRATOR_STEP,
    SPAN_CHAT_ORCHESTRATOR_TOOL_DISPATCH,
)

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

_MAX_TOOL_STEPS = 10  # safety limit to prevent infinite agentic loops
_SEARCH_TOOL_NAME = "search_documents"
_CITATION_TOOL_NAME = "cite_sources"


def _append_stream_item(
    *,
    item: ChatStreamItem,
    collected_text: list[str],
    tool_calls: list[ToolCallInfo],
) -> None:
    """Append one chat stream item to text/tool buffers.

    Quote items are intentionally ignored in WP1 and handled in a later phase.
    """
    if isinstance(item, str):
        collected_text.append(item)
        return
    if isinstance(item, list):
        tool_calls.extend(item)
        return
    logger.debug("orchestrator.quote_item_ignored_wp1", kind=item.kind)


def _trace_step_request(
    *,
    span: trace.Span,
    step_num: int,
    messages: list[ChatMessage],
    tool_schemas: list[ToolSchema] | None,
) -> None:
    span.set_attribute("chat.step", step_num)
    span.set_attributes(
        build_input_attributes(
            {
                "message_count": len(messages),
                "step_num": step_num,
                "tool_count": len(tool_schemas) if tool_schemas else 0,
            },
            mime_type=OpenInferenceMimeTypeValues.JSON,
        )
    )


def _trace_step_response(
    *,
    span: trace.Span,
    collected: list[str],
    tool_calls: list[ToolCallInfo],
    assistant_text: str | None = None,
) -> None:
    span.set_attribute("chat.step.text_chars", len("".join(collected)))
    span.set_attribute("chat.step.tool_call_count", len(tool_calls))
    span.set_attribute(
        "chat.step.tool_calls",
        to_attribute_text([tc.name for tc in tool_calls]),
    )
    if assistant_text is not None:
        span.set_attribute(
            "chat.step.output_preview",
            to_attribute_text(assistant_text),
        )


def _trace_citation_pass_request(
    *,
    span: trace.Span,
    citation_tool_available: bool,
    search_call_ids_in_turn: set[str],
    available_chunk_count: int,
    citation_user_prompt: str,
    messages: list[ChatMessage],
) -> None:
    span.set_attribute("chat.citation_tool_available", citation_tool_available)
    span.set_attribute("chat.citation_pass.recovery_attempted", False)
    span.set_attribute("chat.citation_pass.recovery_succeeded", False)
    span.set_attribute(
        "chat.citation_pass.search_call_ids",
        to_attribute_text(sorted(search_call_ids_in_turn)),
    )
    span.set_attribute("chat.citation_pass.available_chunk_count", available_chunk_count)
    span.set_attribute(
        "chat.citation_pass.request_prompt",
        to_attribute_text(citation_user_prompt),
    )
    span.set_attributes(
        build_input_attributes(
            {
                "search_call_count": len(search_call_ids_in_turn),
                "available_chunk_count": available_chunk_count,
                "message_count": len(messages),
            },
            mime_type=OpenInferenceMimeTypeValues.JSON,
        )
    )


def _trace_citation_pass_response(
    *,
    span: trace.Span,
    cite_calls: list[ToolCallInfo],
    cite_text: list[str],
) -> None:
    model_output = "".join(cite_text)
    span.set_attribute("chat.citation_pass.tool_call_count", len(cite_calls))
    span.set_attribute(
        "chat.citation_pass.tool_calls",
        to_attribute_text([tc.name for tc in cite_calls]),
    )
    span.set_attribute("chat.citation_pass.model_output_chars", len(model_output))
    span.set_attribute(
        "chat.citation_pass.model_output_preview",
        to_attribute_text(model_output),
    )


def _trace_citation_pass_error(
    *,
    span: trace.Span,
    failure_reason: str,
    no_tool_call: bool = False,
) -> None:
    if no_tool_call:
        span.set_attribute("chat.citation_pass.no_tool_call", True)
    span.set_attribute("chat.citation_pass.failure_reason", failure_reason)


def _trace_citation_pass_recovery(
    *,
    span: trace.Span,
    recovery_attempted: bool,
    recovery_succeeded: bool,
    recovered_serialized_tool_call: bool = False,
) -> None:
    span.set_attribute("chat.citation_pass.recovery_attempted", recovery_attempted)
    span.set_attribute("chat.citation_pass.recovery_succeeded", recovery_succeeded)
    if recovered_serialized_tool_call:
        span.set_attribute("chat.citation_pass.recovered_serialized_tool_call", True)


def _trace_citation_pass_dispatch_result(
    *,
    span: trace.Span,
    result: JsonObject,
) -> None:
    span.set_attribute(
        "chat.citation_pass.dispatch_result_preview",
        to_attribute_text(result),
    )


def _trace_citation_pass_missing_tool(*, span: trace.Span, available: bool) -> None:
    span.set_attribute("chat.citation_tool_available", available)
    span.set_attribute("chat.citation_pass.recovery_attempted", False)
    span.set_attribute("chat.citation_pass.recovery_succeeded", False)


def _trace_citation_pass_non_citation_calls(
    *,
    span: trace.Span,
    cite_calls: list[ToolCallInfo],
    allowed_calls: list[ToolCallInfo],
) -> None:
    span.set_attribute(
        "chat.citation_pass.non_citation_tool_call_count",
        len(cite_calls) - len(allowed_calls),
    )


def _trace_tool_dispatch_request(*, span: trace.Span, tc: ToolCallInfo) -> None:
    span.set_attributes(build_span_kind_attributes(OpenInferenceSpanKindValues.CHAIN))
    span.set_attribute("chat.tool.call_id", tc.call_id)
    span.set_attributes(
        build_input_attributes(
            {
                "tool_name": tc.name,
                "call_id": tc.call_id,
                "arguments": tc.arguments,
            },
            mime_type=OpenInferenceMimeTypeValues.JSON,
        )
    )
    span.set_attributes(build_tool_execution_attributes(tool_name=tc.name, parameters=tc.arguments))


def _trace_tool_dispatch_response(
    *,
    span: trace.Span,
    result: JsonObject,
    events: list[ToolEvent],
) -> None:
    span.set_attributes(
        build_output_attributes(
            {
                "status": "ok",
                "result_keys": sorted(result.keys()),
                "event_count": len(events),
            },
            mime_type=OpenInferenceMimeTypeValues.JSON,
        )
    )


def _trace_tool_dispatch_error(
    *,
    span: trace.Span,
    error_msg: str,
    exc: Exception | None = None,
) -> None:
    if exc is not None:
        span.record_exception(exc)
    span.set_attributes(
        build_output_attributes(
            {
                "status": "error",
                "message": error_msg,
            },
            mime_type=OpenInferenceMimeTypeValues.JSON,
        )
    )
    span.set_status(StatusCode.ERROR, error_msg)


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
        prompts: Prompts = DEFAULT_PROMPTS,
    ) -> None:
        self._model = model
        self._prompts = prompt_profile.adjust_prompts(prompts)
        _tools = tools or []
        self._tool_map: dict[str, Tool] = {t.schema.name: t for t in _tools}
        adjusted_schemas = [
            _adjust_tool_schema(t.schema, prompt_profile=prompt_profile) for t in _tools
        ]
        self._tool_schemas: list[ToolSchema] | None = adjusted_schemas if adjusted_schemas else None
        self._tool_schema_map: dict[str, ToolSchema] = {s.name: s for s in adjusted_schemas}
        self._history: list[ChatMessage] = []

    def process_message(self, user_text: str) -> AsyncIterator[ProcessEvent]:
        """Process *user_text* and return an async iterator of :data:`~src.chatbot.app.protocols.ProcessEvent` items.

        Appends the user message to history eagerly (before iteration begins),
        then returns a lazy async iterator that runs the agentic loop.  Tool
        result messages are appended to history during tool steps but are not
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
            search_call_ids_in_turn: set[str] = set()
            final_assistant_text = ""

            for step_num in range(_MAX_TOOL_STEPS):
                with tracer.start_as_current_span(SPAN_CHAT_ORCHESTRATOR_STEP) as step_span:
                    system_text = prompts.system_prompt(datetime.now(tz=UTC))
                    messages = [ChatMessage(role="system", content=system_text), *history]
                    _trace_step_request(
                        span=step_span,
                        step_num=step_num,
                        messages=messages,
                        tool_schemas=tool_schemas,
                    )
                    collected: list[str] = []
                    tool_calls: list[ToolCallInfo] = []

                    async for item in model.stream(messages, tools=tool_schemas):
                        if isinstance(item, str):
                            collected.append(item)
                            yield item
                        else:
                            _append_stream_item(
                                item=item,
                                collected_text=collected,
                                tool_calls=tool_calls,
                            )

                    if not tool_calls:
                        assistant_text = "".join(collected)
                        final_assistant_text = assistant_text
                        _trace_step_response(
                            span=step_span,
                            collected=collected,
                            tool_calls=tool_calls,
                            assistant_text=assistant_text,
                        )
                        history.append(ChatMessage(role="assistant", content=assistant_text))
                        break

                    _trace_step_response(
                        span=step_span,
                        collected=collected,
                        tool_calls=tool_calls,
                    )

                    logger.info(
                        "orchestrator.tool_step",
                        step=step_num,
                        calls=[tc.name for tc in tool_calls],
                    )
                    history.append(
                        ChatMessage(
                            role="assistant",
                            content="".join(collected),
                            tool_calls=tuple(tool_calls),
                        )
                    )
                    for tc in tool_calls:
                        if tc.name == _SEARCH_TOOL_NAME:
                            search_call_ids_in_turn.add(tc.call_id)
                    async for event in _dispatch_tool_calls(tool_calls, history, tool_map):
                        yield event
                        match event:
                            case SourceCitationEvent():
                                emitted_citation_events.append(event)
            else:
                # Safety: exceeded max steps — one final response without tools.
                logger.warning("orchestrator.max_tool_steps_exceeded", limit=_MAX_TOOL_STEPS)
                system_text = prompts.system_prompt(datetime.now(tz=UTC))
                messages = [ChatMessage(role="system", content=system_text), *history]
                final: list[str] = []
                async for item in model.stream(messages, tools=None):
                    if isinstance(item, str):
                        final.append(item)
                        yield item
                    else:
                        _append_stream_item(
                            item=item,
                            collected_text=final,
                            tool_calls=[],
                        )
                final_assistant_text = "".join(final)
                history.append(ChatMessage(role="assistant", content=final_assistant_text))

            # Citation pass: only for the current turn when search results were
            # consumed and no citation event was emitted during the main loop.
            if not emitted_citation_events and search_call_ids_in_turn:
                with tracer.start_as_current_span(SPAN_CHAT_ORCHESTRATOR_CITATION_PASS) as span:
                    citation_tool = tool_map.get(_CITATION_TOOL_NAME)
                    citation_schema = tool_schema_map.get(_CITATION_TOOL_NAME)
                    citation_tool_available = (
                        citation_tool is not None and citation_schema is not None
                    )
                    _trace_citation_pass_missing_tool(
                        span=span,
                        available=citation_tool_available,
                    )
                    if citation_tool_available:
                        assert citation_tool is not None
                        assert citation_schema is not None
                        available_chunks = collect_search_chunks(
                            tuple(history),
                            search_call_ids=search_call_ids_in_turn,
                        )
                        if not available_chunks:
                            _trace_citation_pass_error(
                                span=span,
                                failure_reason="no search chunks correlated to current-turn search calls",
                            )
                            logger.warning("orchestrator.citation_pass_no_search_chunks")
                        else:
                            rendered_search_results = render_search_results_for_prompt(
                                tuple(available_chunks.values())
                            )
                            citation_user_prompt = prompts.citation_request_message(
                                rendered_search_results,
                                final_assistant_text,
                            )
                            citation_schemas: list[ToolSchema] = [citation_schema]
                            system_text = prompts.citation_system_prompt(datetime.now(tz=UTC))
                            messages = [
                                ChatMessage(role="system", content=system_text),
                                ChatMessage(role="user", content=citation_user_prompt),
                            ]
                            _trace_citation_pass_request(
                                span=span,
                                citation_tool_available=citation_tool_available,
                                search_call_ids_in_turn=search_call_ids_in_turn,
                                available_chunk_count=len(available_chunks),
                                citation_user_prompt=citation_user_prompt,
                                messages=messages,
                            )
                            cite_calls: list[ToolCallInfo] = []
                            cite_text: list[str] = []

                            async for item in model.stream(messages, tools=citation_schemas):
                                if isinstance(item, str):
                                    cite_text.append(item)
                                else:
                                    _append_stream_item(
                                        item=item,
                                        collected_text=cite_text,
                                        tool_calls=cite_calls,
                                    )

                            _trace_citation_pass_response(
                                span=span,
                                cite_calls=cite_calls,
                                cite_text=cite_text,
                            )

                            if not cite_calls:
                                _trace_citation_pass_recovery(
                                    span=span,
                                    recovery_attempted=True,
                                    recovery_succeeded=False,
                                )
                                parsed_call = parse_serialized_citation_tool_call(
                                    "".join(cite_text)
                                )
                                if parsed_call is None:
                                    logger.warning("orchestrator.citation_pass_no_tool_call")
                                    _trace_citation_pass_error(
                                        span=span,
                                        failure_reason="model returned text instead of cite_sources tool call",
                                        no_tool_call=True,
                                    )
                                else:
                                    _trace_citation_pass_recovery(
                                        span=span,
                                        recovery_attempted=True,
                                        recovery_succeeded=True,
                                        recovered_serialized_tool_call=True,
                                    )
                                    result, events = await _dispatch(
                                        parsed_call,
                                        tool_map,
                                        ToolContext(history=tuple(history)),
                                    )
                                    _trace_citation_pass_dispatch_result(span=span, result=result)
                                    for event in events:
                                        emitted_citation_events.append(event)
                                        yield event
                            else:
                                allowed_calls = [
                                    tc for tc in cite_calls if tc.name == _CITATION_TOOL_NAME
                                ]
                                _trace_citation_pass_non_citation_calls(
                                    span=span,
                                    cite_calls=cite_calls,
                                    allowed_calls=allowed_calls,
                                )
                                if not allowed_calls:
                                    _trace_citation_pass_error(
                                        span=span,
                                        failure_reason="no cite_sources call in citation pass output",
                                    )
                                    logger.warning(
                                        "orchestrator.citation_pass_wrong_tool_calls",
                                        calls=[tc.name for tc in cite_calls],
                                    )
                                else:
                                    for tc in allowed_calls[:1]:
                                        result, events = await _dispatch(
                                            tc,
                                            tool_map,
                                            ToolContext(history=tuple(history)),
                                        )
                                        _trace_citation_pass_dispatch_result(
                                            span=span,
                                            result=result,
                                        )
                                        for event in events:
                                            emitted_citation_events.append(event)
                                            yield event

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
    with tracer.start_as_current_span(SPAN_CHAT_ORCHESTRATOR_TOOL_DISPATCH) as span:
        _trace_tool_dispatch_request(span=span, tc=tc)

        tool = tool_map.get(tc.name)
        if tool is None:
            logger.error("orchestrator.unknown_tool", name=tc.name)
            error_msg = f"unknown tool '{tc.name}'"
            _trace_tool_dispatch_error(span=span, error_msg=error_msg)
            return {"error": error_msg}, []
        try:
            result, events = await tool.execute(tc.arguments, context)
            _trace_tool_dispatch_response(span=span, result=result, events=events)
            return result, events
        except Exception as exc:
            logger.exception("orchestrator.tool_error", name=tc.name, exc=str(exc))
            _trace_tool_dispatch_error(span=span, error_msg=str(exc), exc=exc)
            return {"error": f"Tool '{tc.name}' raised an error: {exc}"}, []
