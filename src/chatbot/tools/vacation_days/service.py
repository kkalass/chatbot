"""Typed boundaries for the vacation-days integration.

These Protocols describe the collaboration points around the vacation-days
tool without coupling callers to concrete adapter implementations.

Binding point:
    The composition root wires a concrete implementation (currently
    ``SimulatedVacationDaysAdapter``) into ``VacationDaysTool`` via the
    ``VacationDaysService`` Protocol.
"""

from typing import Protocol

from pydantic import BaseModel, Field

from src.chatbot.tools._input_model import ToolInputModel


class VacationDaysInput(ToolInputModel):
    """Decoded and validated arguments from the LLM's tool_call."""

    year: int = Field(description="Calendar year to query vacation days for.")


class VacationDaysOutput(BaseModel):
    """Structured response returned by a vacation-days service."""

    employee_username: str
    year: int
    total_days: int
    used_days: int
    remaining_days: int


class ToolAuthenticationError(Exception):
    """Raised when provided credentials are rejected by the service."""


class VacationDaysService(Protocol):
    """Capability required by :class:`VacationDaysTool` to fetch balances.

    Consumers type against this Protocol; implementations may be simulated,
    HTTP-backed, or any other backend that satisfies the method contract.
    """

    async def get_vacation_days(
        self,
        tool_input: VacationDaysInput,
        username: str,
        password: str,
    ) -> VacationDaysOutput:
        """Return the vacation-day balance for the authenticated employee."""
        ...
