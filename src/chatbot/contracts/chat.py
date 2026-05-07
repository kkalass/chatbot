# SPDX-FileCopyrightText: 2026 Klas Kalaß
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat-model boundary: messages, streaming items, model + profile Protocols."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.chatbot.contracts.i18n import JsonObject
from src.chatbot.contracts.prompts import Prompts
from src.chatbot.contracts.tools import ToolSchema


@dataclass(frozen=True)
class ToolCallInfo:
    """A single tool invocation requested by the model.

    ``call_id`` is a backend-specific correlation token that ties this request
    to its corresponding ``role="tool"`` result message.  Adapters are
    responsible for minting it (e.g. the tool name for Ollama, a UUID returned
    by the OpenAI API); the orchestrator threads it through opaquely.
    Defaults to ``""`` so test doubles that don't exercise correlation can
    omit it.
    """

    name: str
    arguments: JsonObject  # LLM-generated JSON args have no fixed schema
    call_id: str = ""  # backend-specific; adapter fills, orchestrator threads through


@dataclass(frozen=True)
class ChatMessage:
    """An immutable message in a conversation turn."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | JsonObject  # str for all roles except "tool"; JsonObject for tool results
    tool_calls: tuple[ToolCallInfo, ...] | None = (
        None  # populated for role="assistant" tool-call requests
    )
    tool_call_id: str | None = None  # populated for role="tool" result messages


@dataclass(frozen=True)
class ThinkingContent:
    """A complete thinking/reasoning block emitted by a model with extended thinking enabled.

    The adapter strips ``<think>...</think>`` tags from the response stream and
    packages the accumulated content as a single ``ThinkingContent`` item.  The
    orchestrator threads it through to callers as a ``ProcessEvent`` so that
    each consumer (UI, logger, evaluator) can decide independently what to do
    with it.
    """

    text: str


type ChatStreamItem = str | list[ToolCallInfo] | ThinkingContent


@runtime_checkable
class ChatModel(Protocol):
    """Structural interface for a chat model backend.

    A single :meth:`stream` method handles both plain-text and tool-calling
    turns.  It yields text chunks as they arrive from the model.  If the model
    requests tool calls instead of a text response, a single
    ``list[ToolCallInfo]`` is yielded as the final item; otherwise the stream
    ends after the text chunks.  Text and tool calls are mutually exclusive
    within one turn.
    """

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[ChatStreamItem]:
        """Stream a chat completion, optionally advertising tool schemas."""
        ...


@runtime_checkable
class ModelProfile(Protocol):
    """Structural interface for model-specific adaptation of prompts, tool schemas,
    and adapter-level capabilities.
    """

    @property
    def parse_text_tool_calls(self) -> bool:
        """Whether the chat adapter should detect tool calls emitted as JSON text.

        Models that serialise tool invocations as plain JSON in their response
        text (e.g. qwen2.5-coder) require the adapter to buffer and parse that
        text.  Only responses whose first chunk starts with ``{`` or a fenced
        JSON block are buffered; all other responses stream through unchanged.
        """
        ...

    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        """Return model-adjusted prompt templates."""
        ...

    def adjust_tool_description(self, tool_name: str, description: str) -> str:
        """Return model-adjusted tool description text."""
        ...

    def adjust_parameter_schema(self, tool_name: str, schema: JsonObject) -> JsonObject:
        """Return model-adjusted JSON schema for tool parameters."""
        ...
