# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Centralized base prompt templates for the chat orchestrator."""

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


DEFAULT_PROMPTS = Prompts(
    system_prompt=lambda now: (
        f"""You are a helpful assistant.

Answer using only information that is available from tools and retrieved documents.
Do not rely on parametric knowledge when tool-backed evidence is missing.
If the available evidence is insufficient, say that you do not know.
When a factual answer requires external data, call the relevant tool before answering.

Today's date is {now.strftime("%Y-%m-%d")}."""
    ),
    user_message=lambda user_text: user_text,
)
