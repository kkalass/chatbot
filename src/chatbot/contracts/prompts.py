# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt template contract.

The :class:`Prompts` dataclass is a contract — it appears in the
:meth:`~src.chatbot.contracts.chat.ModelProfile.adjust_prompts` signature.
Default content lives in :mod:`src.chatbot.app.chat_prompts`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Prompts:
    """Immutable base prompt configuration.

    Args:
        system_prompt: Callable that receives the current :class:`~datetime.datetime`
            and returns the base system instruction string.
        user_message: Callable used to format the current user turn.  Defaults
            to a pure pass-through.
    """

    system_prompt: Callable[[datetime], str]
    user_message: Callable[[str], str]
