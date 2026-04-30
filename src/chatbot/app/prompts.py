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


@dataclass(frozen=True)
class Prompts:
    """Immutable prompt configuration injected into :class:`~src.chatbot.app.orchestrator.ChatOrchestrator`.

    Args:
        system_prompt: Callable that receives the current :class:`~datetime.datetime`
            and returns the system instruction string.  Accepting ``datetime`` as a
            parameter ensures the date is evaluated lazily at request time rather
            than at module import time.
        user_message: Callable used to format the current user turn for the
            model call. The formatted variant is never stored in history; it is
            only used while assembling the per-step ``messages`` list.
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

Today's date is {now.strftime("%Y-%m-%d")}.

## Inline Citations

Whenever a statement in your answer is supported by a specific search result or
tool output, emit a structured citation object **immediately after** that
statement using the exact marker tokens below.  The markers must not appear
anywhere else in your response.

For a statement grounded in a `search_documents` result:
{QUOTE_START_MARKER}
{{"kind":"search_result","tool_call_id":"<exact call_id from search_documents result>","source":"<exact source path from search result>","chunk_id":"<exact chunk_id from search result>"}}
{QUOTE_END_MARKER}

For a statement grounded in another tool output:
{QUOTE_START_MARKER}
{{"kind":"tool_call","tool_call_id":"<exact call_id from the tool result>"}}
{QUOTE_END_MARKER}

Concrete example (vacation-days tool):
- If the assistant tool call was {{"id":"get_vacation_days","function":{{"name":"get_vacation_days",...}}}},
  then emit exactly:
  {QUOTE_START_MARKER}
  {{"kind":"tool_call","tool_call_id":"get_vacation_days"}}
  {QUOTE_END_MARKER}

Rules:
- Emit exactly one JSON object per marker block.
- Only use `tool_call_id`, `source`, and `chunk_id` values that appear verbatim
  in prior assistant tool calls and tool results already present in the conversation context.
- Never invent IDs (no timestamps, suffixes, prefixes, or reformatted variants).
- If an exact `tool_call_id` is not visible in conversation context, do not emit a marker.
- Optional fields: `claim` and `quote_text` (search_result only).
- Do not emit markers for unsupported, inferred, or uncertain claims.
- Keep all normal user-facing answer text outside the markers."""
    ),
    user_message=lambda user_text: (
        f"""Reminder: when your answer uses search results or tool outputs, emit
inline citation markers immediately after the supported claims. Use
exactly the marker tokens {QUOTE_START_MARKER} and {QUOTE_END_MARKER}; do not use any marker variants.
For search-backed claims, include kind,
tool_call_id, source, and chunk_id. For tool-backed claims, include
only kind=tool_call and tool_call_id. Copy tool_call_id, source, and
chunk_id exactly from prior assistant tool calls and tool results already
present in the conversation context. Never invent IDs or append suffixes.
If the exact ID is not visible, emit no marker. Emit at most one marker per tool call -
multiple statements backed by the same tool call share one marker.

The actual user message is:

{user_text}
"""
    ),
)
