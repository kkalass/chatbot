# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Process events emitted by the orchestrator to the UI / consumers."""

import asyncio
from dataclasses import dataclass

from src.chatbot.contracts.chat import ThinkingContent
from src.chatbot.contracts.citation import (
    HallucinatedCitation,
    NumberedCitation,
    UnsubstantiatedClaim,
)
from src.chatbot.contracts.i18n import I18nMessage, JsonObject


@dataclass(frozen=True)
class ToolCallStarted:
    """Emitted just before a tool call is dispatched.

    Allows the UI to open a progress indicator (e.g. a Chainlit Step) scoped
    to exactly this invocation.  Paired with :class:`ToolCallFinished` which
    carries the same ``call_id``.
    """

    tool_name: str
    call_id: str
    call_description: I18nMessage


@dataclass(frozen=True)
class ToolCallFinished:
    """Emitted after a tool call has been dispatched and its result appended.

    ``result`` carries the raw tool return value serialized as JSON.  It is
    ``None`` when dispatch failed (error path).  Consumers like the eval runner
    use this to build a faithfulness context without re-querying infrastructure.
    """

    tool_name: str
    call_id: str
    result: JsonObject | None = None


@dataclass
class AuthRequiredEvent:
    """Emitted when a tool raises
    :class:`~src.chatbot.contracts.credentials.AuthRequiredException`.

    The orchestrator suspends the tool-call loop and awaits
    :attr:`credential_future`. The UI is expected to:

    1. Show a login form (e.g. :class:`~chainlit.AskElementMessage`).
    2. Store the collected credentials in the session-scoped credential store
       under :attr:`credential_key`.
    3. Set ``credential_future.set_result(True)`` on success, or
       ``set_result(False)`` on cancellation.

    The generator then retries the tool call (on ``True``) or substitutes an
    error result (on ``False``) and continues normally.
    """

    tool_name: str
    credential_key: str
    service_display_name: I18nMessage
    credential_future: asyncio.Future[bool]


type ProcessEvent = (
    str
    | NumberedCitation
    | HallucinatedCitation
    | UnsubstantiatedClaim
    | ToolCallStarted
    | ToolCallFinished
    | AuthRequiredEvent
    | ThinkingContent
)
