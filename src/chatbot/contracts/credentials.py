# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session-scoped credential contracts."""

from dataclasses import dataclass
from typing import Protocol

from src.chatbot.contracts.i18n import I18nMessage


@dataclass(frozen=True)
class UsernamePasswordCredentials:
    """Username/password pair for a service requiring HTTP Basic-style auth."""

    username: str
    password: str


class CredentialStore(Protocol):
    """Session-scoped key-indexed credential repository.

    Each tool that requires credentials operates under a stable ``key``
    (e.g. ``"vacation_days"``).  The key is also carried in
    :class:`~src.chatbot.contracts.process.AuthRequiredEvent` so the UI knows
    which slot to fill after the login form is submitted.
    """

    def get_credentials(self, key: str) -> UsernamePasswordCredentials | None:
        """Return stored credentials for *key*, or ``None`` if not present."""
        ...

    def set_credentials(self, key: str, username: str, password: str) -> None:
        """Store a username/password pair under *key*, replacing any previous entry."""
        ...

    def clear_credentials(self, key: str) -> None:
        """Discard stored credentials for *key* (e.g. after an auth failure)."""
        ...


class AuthRequiredException(Exception):
    """Raised by a Tool when credentials are required but not available.

    The orchestrator catches this and yields an
    :class:`~src.chatbot.contracts.process.AuthRequiredEvent`, pausing the
    tool-call loop until the UI collects credentials via the login form.

    Args:
        credential_key: Stable key that identifies the credential slot in the
            session-scoped :class:`CredentialStore` (e.g. ``"vacation_days"``).
        service_display_name: Localizable name of the service requiring auth.
    """

    def __init__(self, *, credential_key: str, service_display_name: I18nMessage) -> None:
        super().__init__(str(service_display_name.key))
        self.credential_key = credential_key
        self.service_display_name = service_display_name
