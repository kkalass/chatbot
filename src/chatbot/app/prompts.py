"""Centralized base prompt templates for the chat orchestrator.

After the Phase 9 refactor, this module is intentionally free of all citation
concerns; the citation marker policy and per-tool citation fragments are owned
by :class:`~src.chatbot.app.citation.layer.CitationLayer`, which appends them
to the orchestrator-supplied base system prompt.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Prompts:
    """Immutable base prompt configuration.

    Args:
        system_prompt: Callable that receives the current :class:`~datetime.datetime`
            and returns the base system instruction string. The citation layer
            appends its own section to this string before sending to the model.
        user_message: Callable used to format the current user turn.  Defaults
            to a pure pass-through. Application-level framing may wrap the user
            text without duplicating the citation reminder, which the citation
            layer prepends in :meth:`CitationLayer.make_user_message`.
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
