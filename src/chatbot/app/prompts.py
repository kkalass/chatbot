"""Centralized prompt templates for the chat orchestrator.

All text that the orchestrator sends to the model as instruction or as a
user-facing follow-up request is defined here.  Callers customize prompts by
constructing a :class:`Prompts` instance via :func:`dataclasses.replace`
rather than subclassing or patching globals.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Prompts:
    """Immutable prompt configuration injected into :class:`~src.chatbot.app.orchestrator.ChatOrchestrator`.

    Args:
        system_prompt: Callable that receives the current :class:`~datetime.datetime`
            and returns the system instruction string.  Accepting ``datetime`` as a
            parameter ensures the date is evaluated lazily at request time rather
            than at module import time.
        citation_fallback_message: User-turn message appended to history when the
            citation pass is triggered, prompting the model to call ``cite_sources``.
    """

    system_prompt: Callable[[datetime], str]
    citation_fallback_message: str


def _default_system_prompt(now: datetime) -> str:
    return (
        "You are a helpful assistant. "
        "Restrict answers to information available from tools and retrieved documents. "
        "If you don't know the answer, say you don't know. "
        "Express uncertainty when evidence is insufficient rather than drawing on parametric knowledge. "
        "To fulfill those instructions above, you will probably always choose at least one tool to call until you have data to base your answer on."
        f"Today's date is {now.strftime('%Y-%m-%d')}."
    )


_DEFAULT_CITATION_FALLBACK = (
    "Please identify which exact search chunks you used in your last answer. "
    "Then call the cite_sources tool with source+chunk_id citation pairs."
)

DEFAULT_PROMPTS = Prompts(
    system_prompt=_default_system_prompt,
    citation_fallback_message=_DEFAULT_CITATION_FALLBACK,
)
