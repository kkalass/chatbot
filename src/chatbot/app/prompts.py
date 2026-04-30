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
- Emit a citation marker **after every individual sentence** whose content is grounded in the content
  of a specific chunk — do not summarise multiple sentences into a single end-of-paragraph marker.
- Emit exactly one JSON object per marker block.
- Only use `tool_call_id`, `source`, and `chunk_id` values that appear verbatim
  in prior assistant tool calls and tool results already present in the conversation context and that correspond to the content you are citing.
- Never invent IDs (no timestamps, suffixes, prefixes, or reformatted variants).
- If an exact `tool_call_id` is not visible in conversation context, do not emit a marker.
- Only cite a chunk if the claim is directly supported by text in that chunk's `content` field.
  Do not cite a chunk whose content does not contain the information being stated.
- Search results are grouped as `<source ...><chunk chunk_id="..."> ... </chunk>...</source>`.
  The `chunk_id` you copy must come from the `chunk_id` attribute of the **specific `<chunk>` tag
  whose inner text supports the sentence** — not from the first chunk of that source, and not
  from a sibling chunk of the same source. Read the chunk content first, then copy that chunk's id.
- A common failure mode is to default to the first chunk of a source. Do not do this.

Correct vs. incorrect attribution example:

  Given:
    <source title="Example" source_path="ex.pdf">
      <chunk chunk_id="AAA">Job losses are likely in some sectors.</chunk>
      <chunk chunk_id="BBB">AI can also create new jobs and raise productivity.</chunk>
    </source>

  CORRECT (claim about new jobs cites BBB, whose content supports it):
    AI may create new jobs and raise productivity. {QUOTE_START_MARKER}{{"kind":"search_result","tool_call_id":"...","source":"ex.pdf","chunk_id":"BBB"}}{QUOTE_END_MARKER}

  INCORRECT (claim about new jobs cites AAA, the first chunk of the source, even though AAA is about job losses):
    AI may create new jobs and raise productivity. {QUOTE_START_MARKER}{{"kind":"search_result","tool_call_id":"...","source":"ex.pdf","chunk_id":"AAA"}}{QUOTE_END_MARKER}

- `claim` and `quote_text` are optional.
- Logical inferences and transitions derived from cited material do not require
  their own marker — only direct factual claims do.
- If you make a factual claim that you cannot back with a citation marker from
  retrieved chunks or tool output, mark it inline with **!UNBELEGT!** immediately
  after the claim.
- Keep all normal user-facing answer text outside the markers."""
    ),
    user_message=lambda user_text: (
        f"""Reminder: when your answer uses search results or tool outputs, emit
inline citation markers immediately after the supported sentence — one marker
per sentence, not one per paragraph. Use exactly the marker tokens
{QUOTE_START_MARKER} and {QUOTE_END_MARKER}; do not use any marker variants.
For search-backed claims, include kind, tool_call_id, source, and chunk_id.
Only cite a chunk whose `content` field actually contains the information stated.
The `chunk_id` must come from the specific `<chunk>` tag whose inner text supports
the sentence — not the first `<chunk>` of the source. Verify the chunk content
matches the claim before copying its `chunk_id`.
For tool-backed claims, include only kind=tool_call and tool_call_id.
Copy tool_call_id, source, and chunk_id
exactly from prior assistant tool calls and tool results already present in
the conversation context. Never invent IDs or append suffixes.
If the exact ID is not visible, emit no marker.
Never emit a standalone marker list block. A marker is only valid if it appears
immediately after the exact sentence it supports.

!!!IMPORTANT!!!
Every factual claim in your answer must either:
  (a) be immediately followed by a citation marker referencing the exact chunk or
      tool output whose content supports it, OR
  (b) be marked **!UNBELEGT!** inline if no retrieved chunk or tool output
      contains the information.
There is no third option: do not state facts without a marker or a !UNBELEGT! flag.
Logical inferences and transitional sentences derived from cited material are exempt.
Do not append markers in a separate trailing citation section. Place each marker
at the point of use, directly after the supported sentence.
Do not skip markers or !UNBELEGT! flags on follow-up answers.

The actual user message is:

{user_text}
"""
    ),
)
