# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Concrete in-memory credential store implementation.

:class:`CredentialStore` and :class:`UsernamePasswordCredentials` live in
:mod:`src.chatbot.app.protocols`.  This module contains only the session-scoped
in-memory implementation used by the composition root.
"""

import structlog

from src.chatbot.app.protocols import UsernamePasswordCredentials

logger = structlog.get_logger(__name__)

__all__ = ["InMemoryCredentialStore"]


class InMemoryCredentialStore:
    """Satisfies :class:`~src.chatbot.app.protocols.CredentialStore`; holds one session's credentials in memory.

    Instantiated once per chat session in ``on_chat_start`` and stored in the
    Chainlit user session so the login handler and all tools share the same
    instance.
    """

    def __init__(self) -> None:
        self._store: dict[str, UsernamePasswordCredentials] = {}

    def get_credentials(self, key: str) -> UsernamePasswordCredentials | None:
        """Return credentials stored under *key*, or ``None`` if absent."""
        return self._store.get(key)

    def set_credentials(self, key: str, username: str, password: str) -> None:
        """Store username/password under *key*, replacing any previous entry."""
        self._store[key] = UsernamePasswordCredentials(username=username, password=password)
        logger.info("credential_store.stored", key=key, username=username)

    def clear_credentials(self, key: str) -> None:
        """Discard credentials for *key*, typically after an auth failure."""
        self._store.pop(key, None)
        logger.info("credential_store.cleared", key=key)
