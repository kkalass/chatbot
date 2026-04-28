"""Vacation-days-specific interactive username/password authentication.

This module intentionally lives inside the ``vacation_days`` package because
its semantics are service-local: it manages one session-scoped username/
password pair for the vacation-days flow.

Binding point:
    The composition root injects ``InteractiveVacationDaysAuthSession`` into
    ``VacationDaysTool`` through the ``VacationDaysAuth`` Protocol.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)

AskUser = Callable[[str], Awaitable[str | None]]


@dataclass(frozen=True)
class UsernamePasswordCredentials:
    """Username/password pair for the vacation-days service."""

    username: str
    password: str


class VacationDaysAuth(Protocol):
    """Session-scoped auth collaborator used by :class:`VacationDaysTool`.

    ``VacationDaysTool`` depends on this Protocol rather than a concrete auth
    implementation to keep auth policy and prompting strategy replaceable.
    """

    async def get_credentials(self) -> UsernamePasswordCredentials | None:
        """Return cached credentials or interactively collect them."""

    def clear_credentials(self) -> None:
        """Discard any cached credentials for the current session."""


class InteractiveVacationDaysAuthSession:
    """Satisfies :class:`~src.chatbot.tools.vacation_days.auth.VacationDaysAuth`.

    Collects and caches one username/password pair for the vacation-days flow.
    Concrete binding is in ``_build_vacation_days_tool`` in ``src/ui/app.py``.

    The object itself is session-scoped because it is instantiated once in
    ``on_chat_start`` and injected into ``VacationDaysTool``.

    Args:
        ask_user: Async callback that prompts the user and returns their input,
            or ``None`` on cancellation/timeout.
        service_label: Human-readable label shown in the collection prompts.
    """

    def __init__(
        self,
        ask_user: AskUser,
        service_label: str = "the vacation days service",
    ) -> None:
        self._ask_user = ask_user
        self._label = service_label
        self._cached_credentials: UsernamePasswordCredentials | None = None

    async def get_credentials(self) -> UsernamePasswordCredentials | None:
        """Return cached credentials or collect them from the user."""
        if self._cached_credentials is not None:
            return self._cached_credentials

        username = await self._ask_user(
            f"To access {self._label} I need your credentials. Please enter your **username**:"
        )
        if not username:
            logger.info("vacation_days.auth.canceled")
            return None

        password = await self._ask_user(
            "Please enter your **password** (sorry, it will not be hidden yet):"
        )
        if not password:
            logger.info("vacation_days.auth.canceled")
            return None

        self._cached_credentials = UsernamePasswordCredentials(
            username=username,
            password=password,
        )
        logger.info("vacation_days.auth.stored", username=username)
        return self._cached_credentials

    def clear_credentials(self) -> None:
        """Discard cached credentials so the next call re-prompts the user."""
        self._cached_credentials = None
        logger.info("vacation_days.auth.cleared")
