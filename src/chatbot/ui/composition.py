# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chatbot UI composition root — wires UI-specific concerns at session startup.

Observability bootstrap, credential store, and UI-only tools (vacation days)
live here.  Settings → infrastructure mapping lives in
:mod:`src.chatbot.build_from_settings`; this module delegates to it and adds
the Chainlit-session-scoped layer on top.
"""

from src.chatbot.app.chat_prompts import DEFAULT_PROMPTS
from src.chatbot.app.credential_store import InMemoryCredentialStore
from src.chatbot.app.orchestrator import ChatOrchestrator
from src.chatbot.build_from_settings import build_chat_model_with_profile, build_retrieval_tool
from src.chatbot.contracts.credentials import CredentialStore
from src.chatbot.contracts.tools import Tool
from src.chatbot.infrastructure.tools.vacation_days import (
    SimulatedVacationDaysAdapter,
    VacationDaysTool,
)
from src.shared.observability import configure_tracing
from src.shared.observability.logging import configure_logging
from src.shared.settings import Settings


def bootstrap_observability(settings: Settings) -> None:
    """Configure logging and tracing once at process start.

    Must be called before any logger is used or any span is created.
    """
    configure_logging(settings.log_format)
    configure_tracing(
        enabled=settings.otel_enabled,
        service_name=settings.otel_service_name,
        project_name=settings.phoenix_project_name,
        deployment_environment=settings.otel_deployment_environment,
        phoenix_otlp_endpoint=settings.otel_phoenix_otlp_endpoint,
        phoenix_export=settings.otel_export_phoenix,
        jaeger_otlp_endpoint=settings.otel_jaeger_otlp_endpoint,
        jaeger_export=settings.otel_export_jaeger,
        sample_rate=settings.otel_sample_rate,
        console_export=settings.otel_console_export,
        auto_instrument_haystack=settings.otel_auto_instrument_haystack,
    )


def build_credential_store() -> CredentialStore:
    """Create the credential store."""
    return InMemoryCredentialStore()


def _build_vacation_days_tool(credential_store: CredentialStore) -> VacationDaysTool:
    service = SimulatedVacationDaysAdapter()
    return VacationDaysTool(service=service, credential_store=credential_store)


def build_orchestrator(settings: Settings) -> tuple[ChatOrchestrator, CredentialStore]:
    """Compose one session-scoped chat orchestrator instance.

    Returns the orchestrator and its bound credential store; both are owned by
    the caller (the Chainlit session) and discarded together at session end.
    """
    chat_model, model_profile = build_chat_model_with_profile(settings)
    credential_store = build_credential_store()
    vacation_days_tool = _build_vacation_days_tool(credential_store)
    retrieval_tool = build_retrieval_tool(settings)
    tools: list[Tool] = [vacation_days_tool, retrieval_tool]
    orchestrator = ChatOrchestrator.create(
        chat_model,
        tools=tools,
        model_profile=model_profile,
        prompts=DEFAULT_PROMPTS,
    )
    return orchestrator, credential_store
