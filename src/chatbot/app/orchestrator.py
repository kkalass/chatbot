"""Chat orchestration: conversation history and agentic tool-call loop.

The orchestrator is the single entry point for the UI layer.  It depends
exclusively on Protocol interfaces — no concrete infrastructure is imported here.

Document retrieval is modelled as a regular tool: the LLM decides *if* and
*when* to call ``search_documents`` and formulates the query itself.  This
enables multi-hop retrieval and query reformulation from conversation context.

Each turn runs the same loop regardless of which tools are registered:
stream from the model, yield text chunks to the caller, execute any tool calls
that arrive at the end of the stream, and repeat until the model produces a
plain-text response. Grounding provenance is produced inline from parsed quote
items and emitted as :class:`~src.chatbot.app.protocols.SourceCitationEvent`.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime

import structlog
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.chatbot.app.citation_support import (
    build_canonical_key,
    validate_search_quote,
    validate_tool_call_quote,
)
from src.chatbot.app.prompts import DEFAULT_PROMPTS, Prompts
from src.chatbot.app.protocols import (
    ChatMessage,
    ChatModel,
    JsonObject,
    ProcessEvent,
    PromptProfile,
    Quote,
    QuoteReferenceEvent,
    SearchResultQuote,
    SourceChunk,
    SourceCitationEvent,
    Tool,
    ToolCallInfo,
    ToolCallQuote,
    ToolCitationEvent,
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
    SPAN_CHAT_ORCHESTRATOR_STEP,
    SPAN_CHAT_ORCHESTRATOR_TOOL_DISPATCH,
)

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

_MAX_TOOL_STEPS = 10  # safety limit to prevent infinite agentic loops


def _build_step_messages(
    *,
    system_text: str,
    history: list[ChatMessage],
    prompts: Prompts,
) -> list[ChatMessage]:
    """Build one step's model input without mutating persisted history."""
    latest_user_index = next(
        (index for index in range(len(history) - 1, -1, -1) if history[index].role == "user"),
        None,
    )
    step_history: list[ChatMessage] = []
    for index, message in enumerate(history):
        if index == latest_user_index and isinstance(message.content, str):
            step_history.append(
                ChatMessage(role=message.role, content=prompts.user_message(message.content))
            )
            continue
        step_history.append(message)
    return [ChatMessage(role="system", content=system_text), *step_history]


def _resolve_quote(
    quote: Quote,
    history: tuple[ChatMessage, ...],
) -> SourceChunk | ToolCallInfo | None:
    """Validate a quote against conversation history.

    Returns:
        A ``SourceChunk`` for a valid ``SearchResultQuote``.
        A ``ToolCallInfo`` (the actual call from history) for a valid ``ToolCallQuote``.
        ``None`` when validation fails.
    """
    if isinstance(quote, SearchResultQuote):
        return validate_search_quote(quote, history)  # SourceChunk | None
    # ToolCallQuote — returns authoritative ToolCallInfo so tool_name cannot be spoofed
    return validate_tool_call_quote(quote, history)


def _find_tool_result(call_id: str, history: tuple[ChatMessage, ...]) -> JsonObject | None:
    """Return the tool-result JSON for *call_id* from history, or ``None`` if not found."""
    for msg in history:
        if msg.role == "tool" and msg.tool_call_id == call_id and isinstance(msg.content, dict):
            return msg.content  # type: ignore[return-value]  # JsonObject is dict[str, object]
    return None


def _trace_quote_counts(
    *,
    detected: int,
    validated: int,
    invalid: int,
    duplicate: int,
) -> None:
    span = trace.get_current_span()
    span.set_attribute("quote.detected.count", detected)
    span.set_attribute("quote.validated.count", validated)
    span.set_attribute("quote.invalid.count", invalid)
    span.set_attribute("quote.duplicate.count", duplicate)


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
        tool_map = self._tool_map
        prompts = self._prompts

        async def _gen() -> AsyncGenerator[ProcessEvent, None]:
            emitted_citation_events: list[SourceCitationEvent] = []
            emitted_tool_citation_events: list[ToolCitationEvent] = []

            # Inline-quote tracking.
            quote_dedup: dict[str, int] = {}  # canonical_key -> reference_number
            quote_ref_counter = 0
            validated_search_chunks: list[SourceChunk] = []
            quote_detected = 0
            quote_validated = 0
            quote_invalid = 0
            quote_duplicate = 0

            for step_num in range(_MAX_TOOL_STEPS):
                with tracer.start_as_current_span(SPAN_CHAT_ORCHESTRATOR_STEP) as step_span:
                    system_text = prompts.system_prompt(datetime.now(tz=UTC))
                    messages = _build_step_messages(
                        system_text=system_text,
                        history=history,
                        prompts=prompts,
                    )
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
                        elif isinstance(item, list):
                            tool_calls.extend(item)
                        else:
                            quote_detected += 1
                            canonical_key = build_canonical_key(item)
                            if canonical_key in quote_dedup:
                                quote_duplicate += 1
                                yield QuoteReferenceEvent(
                                    reference_number=quote_dedup[canonical_key],
                                    canonical_key=canonical_key,
                                )
                            else:
                                chunk = _resolve_quote(item, tuple(history))
                                if chunk is not None:
                                    quote_validated += 1
                                    quote_ref_counter += 1
                                    quote_dedup[canonical_key] = quote_ref_counter
                                    if isinstance(chunk, SourceChunk):
                                        validated_search_chunks.append(chunk)
                                    elif isinstance(item, ToolCallQuote):
                                        # Use authoritative name from history, not the model's (potentially hallucinated) tool_name
                                        tool_result = _find_tool_result(
                                            item.tool_call_id, tuple(history)
                                        )
                                        emitted_tool_citation_events.append(
                                            ToolCitationEvent(
                                                tool_call_id=item.tool_call_id,
                                                tool_name=chunk.name,
                                                result=tool_result or {},
                                            )
                                        )
                                    yield QuoteReferenceEvent(
                                        reference_number=quote_ref_counter,
                                        canonical_key=canonical_key,
                                    )
                                else:
                                    quote_invalid += 1
                                    logger.debug(
                                        "orchestrator.quote_invalid",
                                        kind=item.kind,
                                        canonical_key=canonical_key,
                                    )

                    # If this is the final, text message (e.g. no tool calls)
                    if not tool_calls:
                        assistant_text = "".join(collected)
                        _trace_step_response(
                            span=step_span,
                            collected=collected,
                            tool_calls=tool_calls,
                            assistant_text=assistant_text,
                        )
                        history.append(ChatMessage(role="assistant", content=assistant_text))
                        break

                    # tool calls present — emit step response and loop to execute tools and continue.
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
                    async for event in _dispatch_tool_calls(tool_calls, history, tool_map):
                        # FIXME: we don't have the cite tool any more, right? So I think this events might never need dispatch?
                        # Can we simplify this code by inlinig _dispatch_tool_calls and not yielding events, only appending to history?
                        # Or can we think of other uses for tools to dispatch events?
                        yield event
                        match event:
                            case SourceCitationEvent():
                                emitted_citation_events.append(event)
                            case ToolCitationEvent():
                                emitted_tool_citation_events.append(event)
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

                history.append(ChatMessage(role="assistant", content="".join(final)))

            # Emit a SourceCitationEvent from validated inline search quotes.
            if validated_search_chunks:
                inline_citation = SourceCitationEvent(validated=tuple(validated_search_chunks))
                emitted_citation_events.append(inline_citation)
                yield inline_citation

            for tool_citation_event in emitted_tool_citation_events:
                yield tool_citation_event

            # Tracing — quote counts collected across all steps of this turn.
            _trace_quote_counts(
                detected=quote_detected,
                validated=quote_validated,
                invalid=quote_invalid,
                duplicate=quote_duplicate,
            )

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
