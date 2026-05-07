# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tool Protocol and tool schema."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.chatbot.contracts.i18n import I18nMessage, JsonObject


@dataclass(frozen=True)
class ToolSchema:
    """Information about a tool exposed to the model.

    This is the minimal contract needed by a model implementation to advertise
    available tools.  It separates the model's concern ("what can I call?")
    from the orchestrator's concern ("how do I dispatch and execute?").
    """

    name: str
    description: str
    parameters_schema: JsonObject  # JSON Schema object describing parameters


@runtime_checkable
class Tool(Protocol):
    """Structural interface for an LLM-callable tool.

    All dependencies (user-interaction callbacks, service adapters) are
    injected at construction time — tools are instantiated once per session.
    The orchestrator advertises tool schemas to the model and dispatches
    ``tool_calls`` by name.  Tools never import the orchestrator or any UI
    module.
    """

    schema: ToolSchema  # All metadata needed by the model (name, description, parameters)
    display_name: I18nMessage  # Human-readable name for UI rendering (resolved via translation map)

    def describe_call(self, args: JsonObject) -> I18nMessage:
        """Return a localizable description of a call with *args* for UI display.

        Implementations should extract the most user-relevant argument(s) and
        return an :class:`I18nMessage` with a ``StrEnum``-defined key and
        the interpolation args.  The UI translation layer resolves the key to
        a human-readable string.
        """
        ...

    async def execute(self, args: JsonObject) -> JsonObject:
        """Execute the tool with *args* decoded from the LLM's tool_call.

        Returns a structured ``JsonObject`` forwarded to the model as the tool
        result.  Values must never contain raw credentials.
        """
        ...
