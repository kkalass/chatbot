"""Centralized prompt templates for the chat orchestrator.

All text that the orchestrator sends to the model as instruction or as a
user-facing follow-up request is defined here.  Callers customize prompts by
constructing a :class:`Prompts` instance via :func:`dataclasses.replace`
rather than subclassing or patching globals.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

# Marker tokens for inline quote extraction (Phase 7).
# These rare sentinel strings are intentionally distinct from any normal prose.
QUOTE_START_MARKER = "<°_quote_°>"
QUOTE_END_MARKER = "</°_quote_°>"

# System prompt addendum that teaches the model the inline quote marker contract.
# Appended only when inline_quotes_enabled=True so the prompt remains clean on
# the legacy citation path.
_INLINE_QUOTE_SYSTEM_PROMPT_ADDENDUM = f"""
## Inline Citations

Whenever a statement in your answer is supported by a specific search result or
tool output, emit a structured citation object **immediately after** that
statement using the exact marker tokens below.  The markers must not appear
anywhere else in your response.

For a statement grounded in a `search_documents` result:
{QUOTE_START_MARKER}
{{"kind":"search_result","claim":"<brief statement being supported>","tool_call_id":"<exact call_id from search_documents result>","source":"<exact source path from search result>","chunk_id":"<exact chunk_id from search result>"}}
{QUOTE_END_MARKER}

For a statement grounded in another tool output:
{QUOTE_START_MARKER}
{{"kind":"tool_call","claim":"<brief statement being supported>","tool_call_id":"<exact call_id from the tool result>","tool_name":"<exact tool name>"}}
{QUOTE_END_MARKER}

Rules:
- Emit exactly one JSON object per marker block.
- Only use `tool_call_id`, `source`, `chunk_id`, and `tool_name` values that
  appear verbatim in tool results already present in the conversation context.
- The optional `quote_text` field may carry a short verbatim extract.
- Do not emit markers for unsupported, inferred, or uncertain claims.
- Keep all normal user-facing answer text outside the markers."""


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


def _base_system_prompt(now: datetime) -> str:
    return f"""You are a helpful assistant.

Answer using only information that is available from tools and retrieved documents.
Do not rely on parametric knowledge when tool-backed evidence is missing.
If the available evidence is insufficient, say that you do not know.
When a factual answer requires external data, call the relevant tool before answering.

Today's date is {now.strftime("%Y-%m-%d")}."""


DEFAULT_PROMPTS = Prompts(
    system_prompt=_base_system_prompt,
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


def build_default_prompts(*, inline_quotes_enabled: bool) -> Prompts:
    """Return the default :class:`Prompts` variant matching the given feature flags.

    When *inline_quotes_enabled* is ``True`` the system prompt is extended with
    the marker contract that instructs the model to emit structured quote blocks.
    Otherwise, :data:`DEFAULT_PROMPTS` is returned unchanged so the legacy
    citation round-trip path is unaffected.
    """
    if not inline_quotes_enabled:
        return DEFAULT_PROMPTS

    def _system_prompt_with_quotes(now: datetime) -> str:
        return _base_system_prompt(now) + _INLINE_QUOTE_SYSTEM_PROMPT_ADDENDUM

    from dataclasses import replace

    return replace(DEFAULT_PROMPTS, system_prompt=_system_prompt_with_quotes)
