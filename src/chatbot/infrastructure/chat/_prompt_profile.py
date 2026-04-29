"""Default prompt profile implementation.

The prompt-profile interface lives in ``src.chatbot.app.protocols.PromptProfile``.
This module provides the default identity implementation used by models that
require no prompt-level tuning.
"""

from dataclasses import dataclass
from datetime import datetime

from src.chatbot.app.prompts import Prompts
from src.chatbot.app.protocols import JsonObject, PromptProfile


@dataclass(frozen=True)
class DefaultChatPromptProfile(PromptProfile):
    """Identity prompt profile — passes all app-level prompts and schemas through unchanged.

    Used as the baseline for models that do not require prompt-level tuning.
    """

    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        return prompts

    def adjust_tool_description(self, tool_name: str, description: str) -> str:
        return description

    def adjust_parameter_schema(self, tool_name: str, schema: JsonObject) -> JsonObject:
        return schema


@dataclass(frozen=True)
class SmallModelPromptProfile(DefaultChatPromptProfile):
    """Prompt profile with stricter tool-calling guidance for weaker tool-call models."""

    def adjust_prompts(self, prompts: Prompts) -> Prompts:
        def _system_prompt(now: datetime) -> str:
            return f"""{prompts.system_prompt(now)}

When calling tools, provide arguments as native JSON values (arrays/objects),
never as JSON-encoded strings and never as schema metadata.

Inline citation JSON rules (when emitting quote markers):
- The JSON inside markers must be a single strict object with no extra fields.
- Copy tool_call_id, source, chunk_id, and tool_name values exactly as they
  appear in tool results — do not paraphrase or abbreviate them."""

        def _citation_system_prompt(now: datetime) -> str:
            return f"""{prompts.citation_system_prompt(now)}

When calling cite_sources, provide arguments as native JSON values
(arrays/objects), never as JSON-encoded strings and never as schema metadata."""

        def _citation_request_message(search_results: str, answer: str) -> str:
            return f"""{prompts.citation_request_message(search_results, answer)}

Important formatting constraints for cite_sources arguments:
- arguments must be a JSON object
- top-level key must be exactly: citations
- citations must be a JSON array value (not a string)
- do not invent other top-level keys such as d"""

        return Prompts(
            system_prompt=_system_prompt,
            citation_system_prompt=_citation_system_prompt,
            citation_request_message=_citation_request_message,
        )

    def adjust_tool_description(self, tool_name: str, description: str) -> str:
        match tool_name:
            case "cite_sources":
                return f"""{description}

Use arguments as a JSON object with a citations array, for example:
{{"citations":[{{"source":"corpus/doc.txt","chunk_id":"abc123"}}]}}

do not invent other top-level keys such as d."""
            case _:
                return description

    def adjust_parameter_schema(self, tool_name: str, schema: JsonObject) -> JsonObject:
        match tool_name:
            case "cite_sources":
                return {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["citations"],
                    "properties": {
                        "citations": {
                            "type": "array",
                            "description": "List of the exact source+chunk_id pairs used in the answer.",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["source", "chunk_id"],
                                "properties": {
                                    "source": {
                                        "type": "string",
                                        "description": "Source path copied exactly from search_documents output.",
                                    },
                                    "chunk_id": {
                                        "type": "string",
                                        "description": "Chunk identifier copied exactly from search_documents output.",
                                    },
                                },
                            },
                        }
                    },
                }
            case _:
                return schema
