"""VacationDaysTool: bridges the LLM tool-calling contract and the HR adapter.

Implements :class:`~src.chatbot.app.citation.CiteableTool` so that vacation-day
balances returned by this tool can be cited inline by the model.

Nothing in this module imports ``chainlit`` or any UI module.  All
session-specific dependencies are injected at construction time.
"""

import json

import structlog
from pydantic import ValidationError

from src.chatbot.app.citation import (
    QUOTE_END_MARKER,
    QUOTE_START_MARKER,
    CitationContext,
    CiteInstructions,
    RawCitation,
    ToolCitation,
    ToolRawCitation,
)
from src.chatbot.app.citation.models import Citation
from src.chatbot.app.protocols import JsonObject, ToolSchema
from src.chatbot.tools.vacation_days.auth import VacationDaysAuth
from src.chatbot.tools.vacation_days.service import (
    ToolAuthenticationError,
    VacationDaysInput,
    VacationDaysService,
)

logger = structlog.get_logger(__name__)

_TOOL_NAME = "get_vacation_days"


_CITE_FRAGMENT = f"""### {_TOOL_NAME}

When a sentence is grounded in the result of a `{_TOOL_NAME}` call, emit
immediately after that sentence a marker block of the form:

  {QUOTE_START_MARKER}{{"kind":"tool_call","tool_call_id":"<id>"}}{QUOTE_END_MARKER}

Required fields (all strings, copied **verbatim** from the conversation
context):
  - kind: must be the literal string "tool_call"
  - tool_call_id: the `tool_call_id` attribute of the assistant's tool call
    that produced the supporting `{_TOOL_NAME}` result

Do not invent IDs. If the exact `tool_call_id` is not visible in the
conversation context, do not emit a marker for that sentence."""


class VacationDaysTool:
    """LLM-callable, citeable tool wrapping a vacation-days service boundary.

    Vacation-days credentials are managed by an injected auth collaborator;
    they never appear in LLM-visible arguments or results.

    Binding points:
        - ``service`` is injected via ``VacationDaysService``.
        - ``auth`` is injected via ``VacationDaysAuth``.
    Concrete bindings are defined in the composition root (`on_chat_start`).

    Args:
        service: The vacation-days service to call after successful auth.
        auth: Handles interactive credential collection and session-scoped
            caching.
    """

    def __init__(
        self,
        service: VacationDaysService,
        auth: VacationDaysAuth,
    ) -> None:
        self._service = service
        self._auth = auth
        self.schema = ToolSchema(
            name=_TOOL_NAME,
            description="""Retrieve the vacation day balance (Urlaubstage / Resturlaub) for the current employee directly from the HR system.

Use this tool whenever the user asks about vacation days, remaining leave, used days, or annual leave entitlement — do NOT use the document search tool for this.

Parameter: year (integer) — the calendar year to query (e.g. 2026).
Returns: total_days (annual entitlement), used_days, remaining_days.""",
            parameters_schema=VacationDaysInput.model_json_schema(mode="validation"),  # type: ignore[arg-type]
        )

    async def execute(self, args: JsonObject) -> JsonObject:
        """Validate *args*, collect credentials if needed, and call the adapter."""
        try:
            tool_input = VacationDaysInput.model_validate(args)
        except ValidationError as exc:
            return {"error": f"Invalid arguments: {exc}"}

        credentials = await self._auth.get_credentials()
        if credentials is None:
            return {"error": "Credential collection was canceled by the user."}

        try:
            result = await self._service.get_vacation_days(
                tool_input,
                username=credentials.username,
                password=credentials.password,
            )
        except ToolAuthenticationError:
            self._auth.clear_credentials()
            logger.warning("tool.auth_failed", username=credentials.username)
            return {
                "error": (
                    "Authentication failed. The stored credentials were cleared. "
                    "Please try again — you will be prompted for your credentials."
                )
            }

        logger.info("tool.success", username=credentials.username, year=tool_input.year)
        return result.model_dump()

    # --- CiteableTool ------------------------------------------------

    def cite_instructions(self) -> CiteInstructions:
        return CiteInstructions(prompt_fragment=_CITE_FRAGMENT)

    def format_for_history(self, result: JsonObject) -> str:
        return json.dumps(result, ensure_ascii=False)

    def validate_and_enrich(
        self,
        raw: RawCitation,
        context: CitationContext,
    ) -> Citation | None:
        if not isinstance(raw, ToolRawCitation):
            return None
        result = context.tool_result_for(raw.tool_call_id)
        if result is None:
            return None
        return ToolCitation(
            raw_marker_text=raw.raw_marker_text,
            tool_call_id=raw.tool_call_id,
            tool_name=_TOOL_NAME,
            result=result,
        )
