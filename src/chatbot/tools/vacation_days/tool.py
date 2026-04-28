"""VacationDaysTool: bridges the LLM tool-calling contract and the HR adapter.

Nothing in this module imports ``chainlit`` or any UI module.  All
session-specific dependencies are injected at construction time.
"""

import structlog
from pydantic import ValidationError

from src.chatbot.app.protocols import JsonObject, ToolContext, ToolEvent, ToolSchema
from src.chatbot.tools.vacation_days.auth import VacationDaysAuth
from src.chatbot.tools.vacation_days.service import (
    ToolAuthenticationError,
    VacationDaysInput,
    VacationDaysService,
)

logger = structlog.get_logger(__name__)

_TOOL_NAME = "get_vacation_days"


class VacationDaysTool:
    """LLM-callable tool wrapping a vacation-days service boundary.

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
        # Generate schema from Pydantic model to ensure consistency between
        # validation and advertisement.  The Pydantic schema is the single
        # source of truth.
        self.schema = ToolSchema(
            name=_TOOL_NAME,
            description="""Look up the vacation day balance for the currently authenticated employee.

Provide the calendar year to query.
Returns total allocation, used days, and remaining days for that year.""",
            parameters_schema=VacationDaysInput.model_json_schema(mode="validation"),  # type: ignore[arg-type]
        )

    async def execute(
        self, args: JsonObject, context: ToolContext
    ) -> tuple[JsonObject, list[ToolEvent]]:
        """Validate *args*, collect credentials if needed, and call the adapter."""
        try:
            tool_input = VacationDaysInput.model_validate(args)
        except ValidationError as exc:
            return {"error": f"Invalid arguments: {exc}"}, []

        credentials = await self._auth.get_credentials()
        if credentials is None:
            return {"error": "Credential collection was canceled by the user."}, []

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
            }, []

        logger.info("tool.success", username=credentials.username, year=tool_input.year)
        return result.model_dump(), []
