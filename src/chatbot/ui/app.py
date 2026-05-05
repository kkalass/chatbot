# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chainlit entry point: session lifecycle, streaming renderer, and observability.

This module is the composition root for the application. Responsibilities:

- **Composition** — wires all infrastructure adapters, tools, and the
  orchestrator into a session-scoped object graph via :func:`on_chat_start`.
- **Streaming renderer** — the :func:`on_message` handler drives the
  orchestrator's event stream, formats text chunks and citation markers into
  Chainlit tokens, and assembles sidebar elements and source-list markdown.
- **Observability** — attaches OpenTelemetry spans and structured log events
  to every request/response cycle.

Session state keys
------------------
- ``"orchestrator"`` — the :class:`~src.chatbot.app.orchestrator.ChatOrchestrator`.
"""

from typing import assert_never, cast
from uuid import uuid4

import chainlit as cl
import structlog
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.chatbot.app.credential_store import InMemoryCredentialStore
from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.app.prompts import DEFAULT_PROMPTS
from src.chatbot.app.protocols import (
    AuthRequiredEvent,
    Citation,
    CredentialStore,
    DocumentCitation,
    HallucinatedCitation,
    NumberedCitation,
    ProcessEvent,
    ThinkingContent,
    Tool,
    ToolCallFinished,
    ToolCallStarted,
    ToolCitation,
    UnsubstantiatedClaim,
    canonical_key,
)
from src.chatbot.config import (
    build_chat_model_config,
    build_retriever_config,
    build_text_embedder_config,
)
from src.chatbot.infrastructure.chat import build_chat_model, build_chat_model_profile
from src.chatbot.infrastructure.embeddings_text import build_text_embedder
from src.chatbot.infrastructure.retrieval import build_retriever
from src.chatbot.observability import configure_tracing, to_attribute_text
from src.chatbot.observability.openinference import (
    build_input_attributes,
    build_metadata_attributes,
    build_output_attributes,
    build_session_attributes,
    build_span_kind_attributes,
    using_session_attributes,
)
from src.chatbot.observability.schema import SPAN_CHAT_UI_ON_MESSAGE
from src.chatbot.tools.retrieval.tool import RetrievalTool
from src.chatbot.tools.vacation_days import (
    SimulatedVacationDaysAdapter,
    VacationDaysTool,
)
from src.chatbot.ui.citation_view import (
    build_citation_content,
    build_citation_markdown,
    build_side_panel_label,
    format_citation_marker,
    format_text_chunk,
)
from src.chatbot.ui.i18n_messages import detect_language, resolve_message
from src.chatbot.ui.logging_config import configure_logging
from src.settings import get_settings

_SESSION_ORCHESTRATOR = "orchestrator"
_SESSION_TRACE_ID = "trace_session_id"
_SESSION_LANG = "lang"
_SESSION_CREDENTIAL_STORE = "credential_store"
_SESSION_SHOWN_SIDEBAR_REFS = "shown_sidebar_refs"
_tracer = trace.get_tracer(__name__)

# Configure logging at module import time (before any logger is used).
_settings = get_settings()
configure_logging(_settings.log_format)
configure_tracing(
    enabled=_settings.otel_enabled,
    service_name=_settings.otel_service_name,
    project_name=_settings.phoenix_project_name,
    deployment_environment=_settings.otel_deployment_environment,
    phoenix_otlp_endpoint=_settings.otel_phoenix_otlp_endpoint,
    phoenix_export=_settings.otel_export_phoenix,
    jaeger_otlp_endpoint=_settings.otel_jaeger_otlp_endpoint,
    jaeger_export=_settings.otel_export_jaeger,
    sample_rate=_settings.otel_sample_rate,
    console_export=_settings.otel_console_export,
    auto_instrument_haystack=_settings.otel_auto_instrument_haystack,
)

logger = structlog.get_logger(__name__)
_DEFAULT_EVAL_RUN_ID = _settings.eval_run_id if _settings.eval_run_id else str(uuid4())


def _detect_session_lang() -> str:
    """Derive a supported language tag from the Chainlit session.

    ``WebsocketSession.language`` is set from the ``HTTP_ACCEPT_LANGUAGE``
    WSGI environ key during socket handshake, e.g. ``"de-DE"`` or ``"en-US"``.
    :func:`~src.chatbot.ui.i18n_messages.detect_language` maps that to a
    supported primary subtag (``"de"``, ``"en"``, …).
    """
    try:
        lang_header: str = cl.context.session.language  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportUnknownMemberType, reportUnknownArgumentType]
    except Exception:
        return "en"
    return detect_language(str(lang_header))  # pyright: ignore[reportUnknownArgumentType]


def _collect_unique_numbered_citations(
    numbered: list[NumberedCitation],
) -> list[NumberedCitation]:
    """Deduplicate by rendered reference number, keeping first occurrence."""
    seen: set[int] = set()
    unique: list[NumberedCitation] = []
    for nc in numbered:
        if nc.reference_number in seen:
            continue
        seen.add(nc.reference_number)
        unique.append(nc)
    return unique


def _build_side_elements(unique: list[NumberedCitation], *, lang: str) -> list[cl.Text]:
    """Aggregate all citations into a single side-panel element.

    One ``cl.Text`` element named with the localised panel title (e.g.
    "Quellenangaben") is returned.  Each citation's content section begins
    with a ``[N]``-prefixed heading so the reference numbers visible in the
    answer text map directly to entries in the panel.
    """
    panel_title = build_side_panel_label(translate=lambda msg: resolve_message(msg, lang=lang))
    content = "\n\n---\n\n".join(
        build_citation_content(nc, translate=lambda msg: resolve_message(msg, lang=lang))
        for nc in unique
    )
    return [cl.Text(name=panel_title, content=content, display="side")]


def _has_renderable_side_element(citation: Citation) -> bool:
    """Side panels suppress tool citations carrying error payloads."""
    match citation:
        case DocumentCitation():
            return True
        case ToolCitation():
            return "error" not in citation.result


def _build_credential_store() -> CredentialStore:
    """Create the credential store."""
    return InMemoryCredentialStore()


def _build_vacation_days_tool(credential_store: CredentialStore) -> VacationDaysTool:
    """Create the vacation-days tool."""
    service = SimulatedVacationDaysAdapter()
    return VacationDaysTool(service=service, credential_store=credential_store)


def _build_retrieval_tool() -> RetrievalTool:
    """Create the retrieval tool and bind its retriever infrastructure."""
    text_embedder = build_text_embedder(build_text_embedder_config(_settings))
    retriever_config = build_retriever_config(_settings)
    retriever = build_retriever(
        config=retriever_config,
        text_embedder=text_embedder,
    )
    return RetrievalTool(retriever=retriever)


def _build_orchestrator() -> tuple[ChatOrchestrator, CredentialStore]:
    """Compose one session-scoped chat orchestrator instance."""
    chat_model_config = build_chat_model_config(_settings)
    model_profile = build_chat_model_profile(chat_model_config)

    chat_model = build_chat_model(
        chat_model_config, parse_text_tool_calls=model_profile.parse_text_tool_calls
    )
    credential_store = _build_credential_store()
    vacation_days_tool = _build_vacation_days_tool(credential_store)
    retrieval_tool = _build_retrieval_tool()
    tools: list[Tool] = [vacation_days_tool, retrieval_tool]
    orchestrator = ChatOrchestrator.create(
        chat_model,
        tools=tools,
        model_profile=model_profile,
        prompts=DEFAULT_PROMPTS,
    )
    return orchestrator, credential_store


def _trace_request(
    *,
    span: trace.Span,
    user_text: str,
    trace_session: str,
) -> None:
    retrieval_version = (
        _settings.eval_retrieval_version
        if _settings.eval_retrieval_version is not None
        else f"top_k={_settings.retrieval_top_k};score_threshold={_settings.retrieval_score_threshold}"
    )
    metadata: dict[str, object] = {
        "run_id": _DEFAULT_EVAL_RUN_ID,
        "trace_session_id": trace_session,
        "model_name": _settings.chat_model,
        "retrieval_version": retrieval_version,
        "temperature": _settings.model_temperature,
        "seed": _settings.model_seed,
        "environment": _settings.eval_environment,
    }
    optional_metadata: dict[str, object | None] = {
        "evaluation_name": _settings.eval_name,
        "candidate_id": _settings.eval_candidate_id,
        "prompt_version_answer": _settings.eval_prompt_version_answer,
        "prompt_version_citation": _settings.eval_prompt_version_citation,
        "corpus_version": _settings.eval_corpus_version,
        "dataset_version": _settings.eval_dataset_version,
    }
    metadata.update({key: value for key, value in optional_metadata.items() if value is not None})

    span.set_attributes(build_span_kind_attributes(OpenInferenceSpanKindValues.CHAIN))
    span.set_attributes(build_session_attributes(trace_session))
    span.set_attributes(
        build_input_attributes(user_text, mime_type=OpenInferenceMimeTypeValues.TEXT)
    )
    span.set_attributes(build_metadata_attributes(metadata))
    span.set_attribute("chat.session_id", trace_session)
    span.set_attribute("chat.user_message.length", len(user_text))
    span.set_attribute("chat.user_message.preview", to_attribute_text(user_text))


def _trace_response(
    *,
    span: trace.Span,
    final_response_text: str,
    numbered: list[NumberedCitation],
    hallucinated: list[HallucinatedCitation],
) -> None:
    span.set_attributes(
        build_output_attributes(final_response_text, mime_type=OpenInferenceMimeTypeValues.TEXT)
    )
    span.set_attributes(
        build_metadata_attributes(
            {
                "numbered_citations": len(numbered),
                "hallucinated_citations": len(hallucinated),
            }
        )
    )
    span.set_attribute("chat.response.numbered_citations", len(numbered))
    span.set_attribute("chat.response.hallucinated_citations", len(hallucinated))
    span.set_status(StatusCode.OK)


@cl.on_chat_start  # pyright: ignore[reportUnknownMemberType]  # chainlit decorators are dynamically typed
async def on_chat_start() -> None:
    """Compose and store one :class:`ChatOrchestrator` per user session."""
    lang = _detect_session_lang()
    orchestrator, credential_store = _build_orchestrator()
    trace_session_id = str(uuid4())

    cl.user_session.set(_SESSION_ORCHESTRATOR, orchestrator)  # pyright: ignore[reportUnknownMemberType]
    cl.user_session.set(_SESSION_TRACE_ID, trace_session_id)  # pyright: ignore[reportUnknownMemberType]
    cl.user_session.set(_SESSION_LANG, lang)  # pyright: ignore[reportUnknownMemberType]
    cl.user_session.set(_SESSION_CREDENTIAL_STORE, credential_store)  # pyright: ignore[reportUnknownMemberType]
    logger.info(
        "session.started", chat_model=_settings.chat_model, trace_session_id=trace_session_id
    )


async def _ask_login(event: AuthRequiredEvent, *, lang: str) -> tuple[str, str] | None:
    """Show a masked login form and return ``(username, password)`` or ``None``.

    Uses :class:`~chainlit.AskElementMessage` with ``LoginForm.jsx`` so the
    password field renders as ``<input type="password">`` — credentials never
    appear in plain text in the UI.
    """
    service_name = resolve_message(event.service_display_name, lang=lang)
    element = cl.CustomElement(
        name="LoginForm",
        props={"service_name": service_name, "lang": lang},
        display="inline",
    )
    res = await cl.AskElementMessage(content="", element=element, timeout=120).send()  # pyright: ignore[reportUnknownMemberType]
    if not res:
        return None
    username = str(res.get("username", "")).strip()  # pyright: ignore[reportUnknownMemberType]
    password = str(res.get("password", "")).strip()  # pyright: ignore[reportUnknownMemberType]
    if not username or not password:
        return None
    return username, password


class ResponseManager:
    """Helper for managing the response message lifecycle within the event loop.

    The orchestrator may emit events that require the current response message to
    be removed (e.g. when a tool call starts) or updated with new content and
    elements.  This manager encapsulates that logic so the event loop can
    focus on formatting and observability.
    """

    def __init__(self):
        self._message: cl.Message | None = None
        self._message_is_transient = False

    @property
    def content(self) -> str:
        return self._message.content if self._message is not None else ""

    async def _get_or_create_message(self, *, create_transient: bool = False) -> cl.Message:
        if self._message is None:
            self._message = cl.Message(content="")
            await self._message.send()
            self._message_is_transient = create_transient
        return self._message

    async def stream_token(self, token: str) -> None:
        if not token:
            return

        if self._message_is_transient:
            # While we do want to append to an ongoing normal response, if the current response
            # is marked as transient, we need to get rid of it and start a fresh message.
            await self.remove_message()

        msg = await self._get_or_create_message()

        await msg.stream_token(token)

    async def start_tool_call(self, tool_call_description: str) -> None:
        # flush pending content and remove dangling empty message if any
        await self.finalize_current_message()
        msg = await self._get_or_create_message(create_transient=True)
        await msg.stream_token(f"⚙️ {tool_call_description}\n")

    async def remove_message(self):
        if self._message is not None:
            await self._message.remove()
            self._message = None
        self._message_is_transient = False

    async def finalize_current_message(self):
        if self._message_is_transient:
            await self.remove_message()
        if self._message is not None and not self._message.content.strip():
            # empty, dangling message — replace it entirely.
            await self.remove_message()
        if self._message is not None:
            await self._message.update()  # send accumulated content immediately
            self._message = None

    async def set_content(self, content: str, elements: list[cl.Text]) -> None:
        msg = await self._get_or_create_message()
        msg.content = content
        msg.elements = elements  # pyright: ignore[reportAttributeAccessIssue]
        await msg.update()


@cl.on_message  # pyright: ignore[reportUnknownMemberType]  # chainlit decorators are dynamically typed
async def on_message(message: cl.Message) -> None:
    """Forward the user message to the orchestrator and stream the response."""
    orchestrator = _get_session_orchestrator()
    trace_session = _get_session_trace()
    lang = _get_session_lang()

    user_text = str(message.content)

    response: ResponseManager = ResponseManager()
    numbered: list[NumberedCitation] = []
    hallucinated: list[HallucinatedCitation] = []
    pending_whitespace = ""

    with (
        using_session_attributes(trace_session),
        _tracer.start_as_current_span(SPAN_CHAT_UI_ON_MESSAGE) as span,
    ):
        _trace_request(span=span, user_text=user_text, trace_session=trace_session)

        event: ProcessEvent

        async for event in orchestrator.process_message(user_text):
            match event:
                case str():
                    tokens, pending_whitespace = format_text_chunk(event, pending_whitespace)
                    for token in tokens:
                        await response.stream_token(token)
                case NumberedCitation():
                    numbered.append(event)
                    tokens, pending_whitespace = format_citation_marker(event, pending_whitespace)
                    for token in tokens:
                        await response.stream_token(token)
                    logger.debug(
                        "session.numbered_citation_rendered",
                        reference_number=event.reference_number,
                        canonical_key=canonical_key(event.citation),
                    )
                case HallucinatedCitation():
                    hallucinated.append(event)
                    logger.info(
                        "session.hallucinated_citation",
                        reason=event.reason,
                        ref=event.raw.ref,
                    )
                case UnsubstantiatedClaim():
                    # Emit inline (like a citation marker) — do not flush pending_whitespace
                    # so the marker stays attached to the preceding sentence without leading newlines.
                    await response.stream_token(" ⚠️")
                    logger.debug("session.unsubstantiated_claim")
                case ToolCallStarted():
                    tool_call_description = resolve_message(event.call_description, lang=lang)
                    await response.start_tool_call(tool_call_description)
                case ToolCallFinished():
                    logger.debug("session.tool_call_finished", tool=event.tool_name)
                case ThinkingContent():
                    # Thinking blocks are silently consumed for now; the model profile
                    # decides whether thinking is enabled, and this match site is the
                    # right place to add UI rendering (e.g. a collapsible Step) later.
                    logger.debug(
                        "session.thinking_content_received",
                        chars=len(event.text),
                        preview=event.text[:120] if event.text else "",
                    )
                case AuthRequiredEvent():
                    credential_store = _get_session_credential_store()
                    await response.finalize_current_message()  # flush pending content and remove dangling empty message before the blocking login flow
                    creds = await _ask_login(event, lang=lang)
                    if creds is not None:
                        credential_store.set_credentials(event.credential_key, *creds)
                        event.credential_future.set_result(True)
                    else:
                        event.credential_future.set_result(False)
                case _:
                    assert_never(event)

        if pending_whitespace:
            await response.stream_token(pending_whitespace)
            pending_whitespace = ""

        unique_numbered = _collect_unique_numbered_citations(numbered)
        renderable = [nc for nc in unique_numbered if _has_renderable_side_element(nc.citation)]
        shown_refs = _get_session_shown_sidebar_refs()
        new_renderable = [nc for nc in renderable if nc.reference_number not in shown_refs]

        if unique_numbered:
            sources_markdown = build_citation_markdown(
                unique_numbered, translate=lambda msg: resolve_message(msg, lang=lang)
            )
            elements: list[cl.Text]
            if new_renderable:
                elements = _build_side_elements(new_renderable, lang=lang)  # pyright: ignore[reportAttributeAccessIssue]
                cl.user_session.set(  # pyright: ignore[reportUnknownMemberType]
                    _SESSION_SHOWN_SIDEBAR_REFS,
                    shown_refs | {nc.reference_number for nc in new_renderable},
                )
                logger.info("session.sources_displayed", count=len(new_renderable))
            else:
                elements = []
            await response.set_content(
                content=f"{response.content.rstrip()}\n\n{sources_markdown}", elements=elements
            )

        _trace_response(
            span=span,
            final_response_text=response.content,
            numbered=unique_numbered,
            hallucinated=hallucinated,
        )

        # Finalize the open streaming message.  When citations were present,
        # set_content() already called msg.update() — this is a harmless
        # second update.  When there were no citations (e.g. unsubstantiated
        # claim only, or a pure-text answer), this is the only update call,
        # and without it the Chainlit "pulsating dot" never disappears.
        await response.finalize_current_message()

    logger.info("session.message_handled", length=len(user_text))


def _get_session_credential_store() -> InMemoryCredentialStore:
    raw = cl.user_session.get(_SESSION_CREDENTIAL_STORE)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(raw, InMemoryCredentialStore):
        raise RuntimeError("InMemoryCredentialStore is missing from session state")
    return raw


def _get_session_lang():
    raw_lang = cl.user_session.get(_SESSION_LANG)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    lang = str(raw_lang) if isinstance(raw_lang, str) else "en"
    return lang


def _get_session_trace():
    raw_trace_session_id = cl.user_session.get(_SESSION_TRACE_ID)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    trace_session_id = cast(object | None, raw_trace_session_id)
    trace_session = str(trace_session_id) if trace_session_id is not None else "unknown"
    return trace_session


def _get_session_orchestrator():
    raw_orchestrator: object | None = cl.user_session.get(_SESSION_ORCHESTRATOR)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(raw_orchestrator, ChatOrchestrator):
        raise RuntimeError("Chat orchestrator is missing from session state")
    orchestrator = raw_orchestrator
    return orchestrator


def _get_session_shown_sidebar_refs() -> set[int]:
    raw = cl.user_session.get(_SESSION_SHOWN_SIDEBAR_REFS)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return raw if isinstance(raw, set) else set()  # type: ignore[return-value]  # Chainlit session is untyped; set[int] guaranteed by _set call
