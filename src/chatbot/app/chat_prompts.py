# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Default chatbot system + user prompt content.

The :class:`~src.chatbot.contracts.prompts.Prompts` *type* lives in
``contracts/`` because it is part of the
:meth:`~src.chatbot.contracts.chat.ModelProfile.adjust_prompts` signature.
The default *content* (this module) is app-level policy: what the chatbot
does by default. Model profiles modify it; they do not define it.
"""

from src.chatbot.contracts.prompts import Prompts

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
