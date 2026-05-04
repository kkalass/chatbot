# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat orchestration: conversation history and agentic tool-call loop.

The orchestrator is the single entry point for the UI layer. It depends
exclusively on Protocol interfaces — no concrete infrastructure is imported
here.

Document retrieval is modelled as a regular tool: the LLM decides *if* and
*when* to call ``search_documents`` and formulates the query itself. This
enables multi-hop retrieval and query reformulation from conversation context.

Each turn runs the same loop regardless of which tools are registered:
stream from the :class:`~src.chatbot.app.citation.CitationLayer`, yield text
chunks and citation events to the caller, execute any tool calls that arrive
at the end of the stream, and repeat until the model produces a plain-text
response. Citation marker parsing and validation are entirely owned by the
citation layer; the orchestrator only assigns stable per-turn reference numbers
to validated citations and surfaces hallucinated ones.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime
from typing import assert_never

import structlog
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.chatbot.app.citation import (
    CitationMessage,
    CitationModel,
)
from src.chatbot.app.prompts import DEFAULT_PROMPTS, Prompts
from src.chatbot.app.protocols import (
    AuthRequiredEvent,
    AuthRequiredException,
    ChatModel,
    Citation,
    DocumentCitation,
    HallucinatedCitation,
    JsonObject,
    ModelProfile,
    NumberedCitation,
    ProcessEvent,
    ThinkingContent,
    Tool,
    ToolCallFinished,
    ToolCallInfo,
    ToolCallStarted,
    ToolCitation,
    ToolSchema,
    UnsubstantiatedClaim,
    canonical_key,
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


def _tool_call_sequence_signature(tool_calls: list[ToolCallInfo]) -> tuple[tuple[str, str], ...]:
    """Stable signature for a tool-call sequence, excluding backend call IDs.

    Comparing only ``(name, arguments)`` lets us detect repeated calls even
    when the backend mints fresh correlation IDs per step.
    """
    return tuple(
        (tc.name, json.dumps(tc.arguments, sort_keys=True, separators=(",", ":")))
        for tc in tool_calls
    )


def _trace_step_request(
    *,
    span: trace.Span,
    step_num: int,
    history_size: int,
    tool_schemas: list[ToolSchema] | None,
) -> None:
    span.set_attribute("chat.step", step_num)
    span.set_attributes(
        build_input_attributes(
            {
                "history_size": history_size,
                "step_num": step_num,
                "tool_count": len(tool_schemas) if tool_schemas else 0,
            },
            mime_type=OpenInferenceMimeTypeValues.JSON,
        )
    )


def _trace_step_response(
    *,
    span: trace.Span,
    text_chars: int,
    tool_calls: list[ToolCallInfo],
    assistant_text: str,
) -> None:
    span.set_attribute("chat.step.text_chars", text_chars)
    span.set_attribute("chat.step.tool_call_count", len(tool_calls))
    span.set_attribute(
        "chat.step.tool_calls",
        to_attribute_text([tc.name for tc in tool_calls]),
    )
    span.set_attribute("chat.step.output_preview", to_attribute_text(assistant_text))


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


def _trace_tool_dispatch_response(*, span: trace.Span, result: JsonObject) -> None:
    span.set_attributes(
        build_output_attributes(
            {"status": "ok", "result_keys": sorted(result.keys())},
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
            {"status": "error", "message": error_msg},
            mime_type=OpenInferenceMimeTypeValues.JSON,
        )
    )
    span.set_status(StatusCode.ERROR, error_msg)


def _trace_citation_counts(
    *,
    validated: int,
    duplicate: int,
    hallucinated: int,
) -> None:
    span = trace.get_current_span()
    span.set_attribute("citation.validated.count", validated)
    span.set_attribute("citation.duplicate.count", duplicate)
    span.set_attribute("citation.hallucinated.count", hallucinated)


class _CitationNumberer:
    """Assigns stable per-turn reference numbers to validated citations, deduplicating by canonical key.

    Shared across the main and fallback streaming loops so numbers remain
    consistent within a single turn regardless of which path produced the citation.
    """

    def __init__(self) -> None:
        self._ref_by_key: dict[str, int] = {}
        self._ref_counter: int = 0
        self.validated: int = 0
        self.duplicate: int = 0
        self.hallucinated: int = 0

    def assign(self, item: Citation) -> NumberedCitation:
        key = canonical_key(item)
        existing = self._ref_by_key.get(key)
        if existing is not None:
            self.duplicate += 1
            return NumberedCitation(reference_number=existing, citation=item)
        self.validated += 1
        self._ref_counter += 1
        self._ref_by_key[key] = self._ref_counter
        return NumberedCitation(reference_number=self._ref_counter, citation=item)


class ChatOrchestrator:
    """Manages per-session conversation history and the agentic tool-call loop.

    Constructed once per chat session and stored in session-scoped state.
    All dependencies are injected at construction time; no infrastructure is
    instantiated internally.

    Prefer the :meth:`create` factory for production use — it filters
    ``CiteableTool``\\s automatically and builds the ``CitationLayer``.
    The ``__init__`` constructor is intentionally kept injectable for unit
    tests that supply a ``CitationLayer`` stub.

    Args:
        citation_layer: Citation-aware decorator over the underlying chat
            model. Owns marker parsing, prompt fragment assembly, and citation
            validation. The orchestrator never talks to the raw ``ChatModel``.
        tools: Tool implementations registered for dispatch and advertised to
            the model.
        model_profile: Model-specific prompt profile. Used once at
            construction time to derive adjusted prompts and adjusted tool
            schemas.
        prompts: Custom base prompt configuration. Mainly for testing — allows
            switching between prompt variants for evaluation purposes.
    """

    def __init__(
        self,
        citation_layer: CitationModel,
        *,
        model_profile: ModelProfile,
        tools: list[Tool] | None = None,
        prompts: Prompts = DEFAULT_PROMPTS,
    ) -> None:
        self._citation_layer = citation_layer
        self._prompts = model_profile.adjust_prompts(prompts)
        _tools = tools or []
        self._tool_map: dict[str, Tool] = {t.schema.name: t for t in _tools}
        adjusted_schemas = [
            _adjust_tool_schema(t.schema, model_profile=model_profile) for t in _tools
        ]
        self._tool_schemas: list[ToolSchema] | None = adjusted_schemas if adjusted_schemas else None
        self._history: list[CitationMessage] = []

    @classmethod
    def create(
        cls,
        model: ChatModel,
        *,
        tools: list[Tool] | None = None,
        model_profile: ModelProfile,
        prompts: Prompts = DEFAULT_PROMPTS,
    ) -> "ChatOrchestrator":
        """Construct a :class:`ChatOrchestrator` from a raw ``ChatModel`` and tools.

        Filters ``CiteableTool``\\s from *tools* and wires them into a
        :class:`~src.chatbot.app.citation.CitationLayer` internally, so
        callers do not need to split the tool list manually.

        Args:
            model: Inner chat model supplying text and tool-call streams.
            tools: All tools to register. ``CiteableTool`` instances are
                automatically forwarded to the citation layer.
            model_profile: Model-specific prompt adjustments.
            prompts: Base prompt configuration; defaults to
                :data:`~src.chatbot.app.prompts.DEFAULT_PROMPTS`.
        """
        _tools = tools or []
        citation_layer = CitationModel(model, tools=_tools)
        return cls(citation_layer, tools=_tools, model_profile=model_profile, prompts=prompts)

    def process_message(self, user_text: str) -> AsyncIterator[ProcessEvent]:
        """Process *user_text* and return an async iterator of :data:`ProcessEvent` items.

        Appends the user message to history eagerly (before iteration begins),
        then returns a lazy async iterator that runs the agentic loop.

        Args:
            user_text: Raw message text from the user.

        Returns:
            Async iterator yielding ``str`` for streaming text and
            :class:`NumberedCitation` / :class:`HallucinatedCitation` / :class:`UnsubstantiatedClaim` for
            citation events.
        """
        user_msg = self._citation_layer.make_user_message(self._prompts.user_message(user_text))
        self._history.append(user_msg)

        history = self._history
        citation_layer = self._citation_layer
        tool_schemas = self._tool_schemas
        tool_map = self._tool_map
        prompts = self._prompts
        user_msg_content = user_msg.llm_content

        async def _gen() -> AsyncGenerator[ProcessEvent, None]:
            previous_tool_call_signature: tuple[tuple[str, str], ...] | None = None
            last_step = False
            numberer = _CitationNumberer()

            for step_num in range(_MAX_TOOL_STEPS + 1):
                with tracer.start_as_current_span(SPAN_CHAT_ORCHESTRATOR_STEP) as step_span:
                    # Force a final no-tools step once the safety limit is reached.
                    if not last_step and step_num == _MAX_TOOL_STEPS:
                        logger.warning(
                            "orchestrator.max_tool_steps_exceeded", limit=_MAX_TOOL_STEPS
                        )
                        last_step = True

                    effective_tools = None if last_step else tool_schemas
                    system_msg = citation_layer.make_system_message(
                        prompts.system_prompt(datetime.now(tz=UTC))
                    )
                    _trace_step_request(
                        span=step_span,
                        step_num=step_num,
                        history_size=1 + len(history),
                        tool_schemas=effective_tools,
                    )

                    parts: list[str | Citation | HallucinatedCitation | UnsubstantiatedClaim] = []
                    tool_calls: list[ToolCallInfo] = []
                    text_chars = 0

                    async for item in citation_layer.stream(
                        [system_msg, *history], tools=effective_tools
                    ):
                        match item:
                            case str():
                                text_chars += len(item)
                                parts.append(item)
                                yield item
                            case ThinkingContent():
                                # Thread through to the caller; consumed/logged at UI layer.
                                yield item
                            case list():
                                if not last_step:
                                    tool_calls.extend(item)
                                # else: no tools advertised — defensively ignore any spurious calls.
                            case HallucinatedCitation():
                                numberer.hallucinated += 1
                                parts.append(item)
                                yield item
                            case UnsubstantiatedClaim():
                                parts.append(item)
                                yield item
                            case DocumentCitation() | ToolCitation():
                                numbered = numberer.assign(item)
                                parts.append(item)
                                yield numbered
                            case _:
                                assert_never(item)

                    assistant_message = citation_layer.make_assistant_message(
                        parts, tool_calls=None if last_step else tool_calls
                    )
                    _trace_step_response(
                        span=step_span,
                        text_chars=text_chars,
                        tool_calls=tool_calls,
                        assistant_text=assistant_message.llm_content,
                    )
                    history.append(assistant_message)

                    # A text-only response or the forced final step terminates the loop.
                    if not tool_calls or last_step:
                        break

                    # Detect if the model is stuck in a loop and trigger a final no-tools step.
                    current_signature = _tool_call_sequence_signature(tool_calls)
                    if previous_tool_call_signature == current_signature:
                        logger.warning(
                            "orchestrator.repeated_tool_calls_detected",
                            step=step_num,
                            calls=[tc.name for tc in tool_calls],
                        )
                        # Keep the duplicate assistant message so the conversation
                        # structure is valid (tool_call must be followed by
                        # tool_result). Append a synthetic blocked-tool response
                        # for each pending call, then inject a user-turn message
                        # that re-states the original question with citation
                        # instructions. Many LLMs only generate user-facing text
                        # in response to a user turn; ending at a tool result
                        # causes them to produce empty output.
                        for tc in tool_calls:
                            history.append(citation_layer.make_blocked_tool_response(tc))
                        history.append(citation_layer.make_loop_escape_message(user_msg_content))
                        last_step = True
                        continue
                    previous_tool_call_signature = current_signature

                    # All good — dispatch tools and continue to the next step.
                    logger.info(
                        "orchestrator.tool_step",
                        step=step_num,
                        calls=[tc.name for tc in tool_calls],
                    )
                    for tc in tool_calls:
                        tool = tool_map[tc.name]
                        yield ToolCallStarted(
                            tool_name=tc.name,
                            call_id=tc.call_id,
                            call_description=tool.describe_call(tc.arguments),
                        )
                        try:
                            result = await _dispatch(tc, tool_map)
                        except AuthRequiredException as exc:
                            future: asyncio.Future[bool] = (
                                asyncio.get_running_loop().create_future()
                            )
                            yield AuthRequiredEvent(
                                tool_name=tc.name,
                                credential_key=exc.credential_key,
                                service_display_name=exc.service_display_name,
                                credential_future=future,
                            )
                            success = await future
                            if success:
                                result = await _dispatch(tc, tool_map)
                            else:
                                result = {"error": "Authentication was canceled by the user."}
                        history.append(
                            citation_layer.make_tool_message(tc.call_id, tc.name, result)
                        )
                        yield ToolCallFinished(tool_name=tc.name, call_id=tc.call_id)

            _trace_citation_counts(
                validated=numberer.validated,
                duplicate=numberer.duplicate,
                hallucinated=numberer.hallucinated,
            )

        return _gen()


def _adjust_tool_schema(schema: ToolSchema, *, model_profile: ModelProfile) -> ToolSchema:
    """Apply model-specific profile adaptations to a tool schema once."""
    return ToolSchema(
        name=schema.name,
        description=model_profile.adjust_tool_description(schema.name, schema.description),
        parameters_schema=model_profile.adjust_parameter_schema(
            schema.name, schema.parameters_schema
        ),
    )


async def _dispatch(tc: ToolCallInfo, tool_map: dict[str, Tool]) -> JsonObject:
    """Look up the named tool, execute it, and return its structured result."""
    with tracer.start_as_current_span(SPAN_CHAT_ORCHESTRATOR_TOOL_DISPATCH) as span:
        _trace_tool_dispatch_request(span=span, tc=tc)

        tool = tool_map.get(tc.name)
        if tool is None:
            logger.error("orchestrator.unknown_tool", name=tc.name)
            error_msg = f"unknown tool '{tc.name}'"
            _trace_tool_dispatch_error(span=span, error_msg=error_msg)
            return {"error": error_msg}

        try:
            result = await tool.execute(tc.arguments)
            _trace_tool_dispatch_response(span=span, result=result)
            return result
        except AuthRequiredException:
            raise  # propagate to the orchestrator's tool-call loop for UI handling
        except Exception as exc:
            logger.exception("orchestrator.tool_error", name=tc.name, exc=str(exc))
            _trace_tool_dispatch_error(span=span, error_msg=str(exc), exc=exc)
            return {"error": f"Tool '{tc.name}' raised an error: {exc}"}
