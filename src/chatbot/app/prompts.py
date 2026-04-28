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
        citation_system_prompt: Callable that receives the current
            :class:`~datetime.datetime` and returns the dedicated system
            instruction for the citation pass.
        citation_request_message: Callable that builds the dedicated user message
            for the citation pass from rendered search results and the final
            assistant answer text.
    """

    system_prompt: Callable[[datetime], str]
    citation_system_prompt: Callable[[datetime], str]
    citation_request_message: Callable[[str, str], str]


DEFAULT_PROMPTS = Prompts(
    system_prompt=lambda now: (
        f"""You are a helpful assistant.

Answer using only information that is available from tools and retrieved documents.
Do not rely on parametric knowledge when tool-backed evidence is missing.
If the available evidence is insufficient, say that you do not know.
When a factual answer requires external data, call the relevant tool before answering.

Today's date is {now.strftime("%Y-%m-%d")}."""
    ),
    citation_system_prompt=lambda now: (
        f"""You are a citation alignment assistant.

Your only job is to map the already-written answer to supporting retrieved chunks.
Do not answer the user, do not rewrite the answer, and do not add commentary.
Use only the search results provided in the citation request.
If support is missing, return an empty citations list.

Today's date is {now.strftime("%Y-%m-%d")}."""
    ),
    citation_request_message=lambda search_results, answer: (
        f"""You are performing a citation-only mapping task.

You previously returned the following search results:
<search_results>
{search_results}
</search_results>

Determine which of these search results support the following answer:
<answer>
{answer}
</answer>

Call the cite_sources tool exactly once with the supporting source+chunk_id pairs.
Rules:
- use only source and chunk_id values that appear in <search_results>
- do not call any tool other than cite_sources
- do not output explanatory text
- if no result supports the answer, call cite_sources with citations=[]"""
    ),
)
