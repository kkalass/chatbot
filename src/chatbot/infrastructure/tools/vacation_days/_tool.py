# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""VacationDaysTool: bridges the LLM tool-calling contract and the HR adapter.

Nothing in this module imports ``chainlit`` or any UI module.  All
session-specific dependencies are injected at construction time.
"""

import structlog
from pydantic import ValidationError

from src.chatbot.contracts.credentials import AuthRequiredException, CredentialStore
from src.chatbot.contracts.i18n import I18nMessage, JsonObject
from src.chatbot.contracts.tools import ToolSchema

from ._keys import VacationDaysCallKey
from ._service import (
    ToolAuthenticationError,
    VacationDaysInput,
    VacationDaysService,
)

logger = structlog.get_logger(__name__)

_TOOL_NAME = "get_vacation_days"
_CREDENTIAL_KEY = "vacation_days"


class VacationDaysTool:
    """LLM-callable tool wrapping a vacation-days service boundary.

    Credentials are retrieved from the injected :class:`CredentialStore` under
    the stable key :data:`_CREDENTIAL_KEY` (``"vacation_days"``).  They never
    appear in LLM-visible arguments or results.

    Binding points:
        - ``service`` is injected via ``VacationDaysService``.
        - ``credential_store`` is injected via ``CredentialStore``.
    Concrete bindings are defined in the composition root (``on_chat_start``).

    Args:
        service: The vacation-days service to call after successful auth.
        credential_store: Session-scoped key-indexed credential repository.
    """

    def __init__(
        self,
        service: VacationDaysService,
        credential_store: CredentialStore,
    ) -> None:
        self._service = service
        self._credential_store = credential_store
        self.display_name = I18nMessage(key=VacationDaysCallKey.DISPLAY_NAME, args={})
        self.schema = ToolSchema(
            name=_TOOL_NAME,
            description="""Retrieve the vacation day balance (Urlaubstage / Resturlaub) for the current employee directly from the HR system.

Use this tool whenever the user asks about vacation days, remaining leave, used days, or annual leave entitlement — do NOT use the document search tool for this.

Parameter: year (integer) — the calendar year to query (e.g. 2026).
Returns: total_days (annual entitlement), used_days, remaining_days.""",
            parameters_schema=VacationDaysInput.model_json_schema(mode="validation"),  # type: ignore[arg-type]
        )

    def describe_call(self, args: JsonObject) -> I18nMessage:
        year = str(args.get("year", ""))
        return I18nMessage(key=VacationDaysCallKey.QUERYING, args={"year": year})

    async def execute(self, args: JsonObject) -> JsonObject:
        """Validate *args*, check credentials, and call the adapter.

        Raises:
            AuthRequiredException: When no credentials are stored. The
                orchestrator catches this and yields an ``AuthRequiredEvent``
                so the UI can collect credentials without involving the LLM.
        """
        try:
            tool_input = VacationDaysInput.model_validate(args)
        except ValidationError as exc:
            return {"error": f"Invalid arguments: {exc}"}

        credentials = self._credential_store.get_credentials(_CREDENTIAL_KEY)
        if credentials is None:
            raise AuthRequiredException(
                credential_key=_CREDENTIAL_KEY,
                service_display_name=self.display_name,
            )

        try:
            result = await self._service.get_vacation_days(
                tool_input,
                username=credentials.username,
                password=credentials.password,
            )
        except ToolAuthenticationError:
            self._credential_store.clear_credentials(_CREDENTIAL_KEY)
            logger.warning("tool.auth_failed", username=credentials.username)
            return {
                "error": (
                    "Authentication failed. The stored credentials were cleared. "
                    "Please try again — you will be prompted for your credentials."
                )
            }

        logger.info("tool.success", username=credentials.username, year=tool_input.year)
        return result.model_dump()
