# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vacation-days credential store and auth protocol.

This module intentionally lives inside the ``vacation_days`` package because
its semantics are service-local: it manages one session-scoped username/
password pair for the vacation-days flow.

Binding point:
    The composition root creates a :class:`VacationDaysCredentialStore`,
    stores it in the Chainlit user session, and injects it into
    ``VacationDaysTool`` via the ``VacationDaysAuth`` protocol.
    The UI fills the store through :func:`set_credentials` after the user
    submits the login form.
"""

from dataclasses import dataclass
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class UsernamePasswordCredentials:
    """Username/password pair for the vacation-days service."""

    username: str
    password: str


class VacationDaysAuth(Protocol):
    """Session-scoped auth collaborator used by :class:`VacationDaysTool`.

    ``VacationDaysTool`` depends on this Protocol rather than a concrete
    implementation to keep credential storage replaceable.
    """

    def get_credentials(self) -> UsernamePasswordCredentials | None:
        """Return cached credentials, or ``None`` if no credentials are stored."""

    def clear_credentials(self) -> None:
        """Discard any cached credentials for the current session."""


class VacationDaysCredentialStore:
    """Satisfies :class:`~src.chatbot.tools.vacation_days.auth.VacationDaysAuth`.

    Holds one session-scoped username/password pair. Credentials are set
    explicitly by the UI via :meth:`set_credentials` after the user submits the
    login form — no interactive prompting happens here.

    Instantiated once in ``on_chat_start`` and stored in the Chainlit user
    session so the login handler and the tool share the same instance.
    """

    def __init__(self) -> None:
        self._cached_credentials: UsernamePasswordCredentials | None = None

    def set_credentials(self, username: str, password: str) -> None:
        """Store a username/password pair, replacing any previous credentials."""
        self._cached_credentials = UsernamePasswordCredentials(
            username=username,
            password=password,
        )
        logger.info("vacation_days.auth.stored", username=username)

    def get_credentials(self) -> UsernamePasswordCredentials | None:
        """Return the stored credentials, or ``None`` if none are available."""
        return self._cached_credentials

    def clear_credentials(self) -> None:
        """Discard cached credentials on auth failure so the next call re-prompts."""
        self._cached_credentials = None
        logger.info("vacation_days.auth.cleared")
