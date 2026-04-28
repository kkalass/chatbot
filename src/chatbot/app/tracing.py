"""Tracing serialization helpers for app-layer types.

These helpers produce bounded, human-readable span attribute values from
domain types (ChatMessage, JsonObject).  They belong in the ``app`` layer
because they introspect ChatMessage structure, but they carry no business
logic — their only concern is formatting for observability.

Keeping them here (rather than in the orchestrator module) ensures the
orchestrator stays focused on control-flow and that any future model adapter
or tool can reuse the same serialization without reaching into orchestration
code.
"""

from collections.abc import Sequence
from typing import cast

from src.chatbot.app.protocols import ChatMessage, JsonObject
from src.chatbot.observability import to_attribute_text

_DEFAULT_MAX_MESSAGES = 8
_DEFAULT_MAX_CHUNKS = 3


def summarize_messages(
    messages: Sequence[ChatMessage],
    *,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
) -> list[JsonObject]:
    """Return a compact, human-readable summary of the last *max_messages* messages.

    Tool-result messages containing ``search_documents`` chunks are expanded to
    show source + content preview rather than a raw JSON blob.

    Args:
        messages: Full conversation snapshot as passed to the model.
        max_messages: How many messages from the tail to include.

    Returns:
        A list of dicts suitable for ``to_attribute_text()`` serialization.
    """
    summary: list[JsonObject] = []
    for msg in messages[-max_messages:]:
        entry: JsonObject = {
            "role": msg.role,
            "tool_call_id": msg.tool_call_id,
            "tool_calls": [tc.name for tc in msg.tool_calls] if msg.tool_calls else [],
        }
        if isinstance(msg.content, str):
            entry["content_chars"] = len(msg.content)
            entry["content_preview"] = to_attribute_text(msg.content, max_chars=240)
        else:
            entry["content_keys"] = sorted(msg.content.keys())
            chunks = cast(object, msg.content.get("chunks"))
            if isinstance(chunks, list):
                chunk_list = cast(list[object], chunks)
                entry["chunk_count"] = len(chunk_list)
                chunk_items: list[JsonObject] = []
                for chunk_obj in chunk_list[:_DEFAULT_MAX_CHUNKS]:
                    if isinstance(chunk_obj, dict):
                        chunk = cast(dict[str, object], chunk_obj)
                        source = cast(object, chunk.get("source"))
                        chunk_id = cast(object, chunk.get("chunk_id"))
                        content = cast(object, chunk.get("content"))
                        chunk_items.append(
                            {
                                "source": str(source) if source is not None else "",
                                "chunk_id": str(chunk_id) if chunk_id is not None else "",
                                "content_preview": to_attribute_text(
                                    str(content) if content is not None else "",
                                    max_chars=160,
                                ),
                            }
                        )
                entry["chunks"] = chunk_items
        summary.append(entry)
    return summary


def summarize_search_result(
    result: JsonObject,
    *,
    max_chunks: int = _DEFAULT_MAX_CHUNKS,
) -> list[JsonObject]:
    """Return a compact, human-readable preview of a ``search_documents`` tool result.

    Args:
        result: Raw JSON returned by :class:`~src.chatbot.tools.retrieval.tool.RetrievalTool`.
        max_chunks: Number of top chunks to include in the preview.

    Returns:
        A list of dicts with ``source``, ``chunk_id``, ``score``, and
        ``content_preview`` for each chunk, suitable for span attributes.
    """
    chunks = cast(object, result.get("chunks"))
    if not isinstance(chunks, list):
        return []
    chunk_list = cast(list[object], chunks)

    preview: list[JsonObject] = []
    for chunk_obj in chunk_list[:max_chunks]:
        if not isinstance(chunk_obj, dict):
            continue
        chunk = cast(dict[str, object], chunk_obj)
        source = cast(object, chunk.get("source"))
        chunk_id = cast(object, chunk.get("chunk_id"))
        score = cast(object, chunk.get("score"))
        content = cast(object, chunk.get("content"))
        preview.append(
            {
                "source": str(source) if source is not None else "",
                "chunk_id": str(chunk_id) if chunk_id is not None else "",
                "score": score,
                "content_preview": to_attribute_text(
                    str(content) if content is not None else "",
                    max_chars=180,
                ),
            }
        )
    return preview
