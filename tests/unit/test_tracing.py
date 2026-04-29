"""Unit tests for tracing helpers."""

from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues

from src.chatbot.app.protocols import ChatMessage, SourceChunk, ToolCallInfo, ToolSchema
from src.chatbot.observability.openinference import (
    build_document,
    build_input_attributes,
    build_llm_attributes,
    build_message,
    build_output_attributes,
    build_retriever_attributes,
    build_session_attributes,
    build_span_kind_attributes,
    build_tool_call,
    build_tool_execution_attributes,
)
from src.chatbot.observability.tracing import to_attribute_text


def test_to_attribute_text_serializes_non_string_values() -> None:
    text = to_attribute_text({"a": 1, "b": ["x", "y"]})

    assert '"a": 1' in text
    assert '"b": ["x", "y"]' in text


def test_to_attribute_text_truncates_long_values() -> None:
    text = to_attribute_text("x" * 20, max_chars=8)

    assert text == "xxxxxxxx...<truncated>"


def test_build_span_kind_attributes_uses_openinference_key() -> None:
    attributes = build_span_kind_attributes(OpenInferenceSpanKindValues.LLM)

    assert attributes == {"openinference.span.kind": "LLM"}


def test_build_input_and_output_attributes_use_openinference_helpers() -> None:
    input_attributes = build_input_attributes(
        {"query": "Where is Zurich?"},
        mime_type=OpenInferenceMimeTypeValues.JSON,
    )
    output_attributes = build_output_attributes(
        "Zurich is in Switzerland.",
        mime_type=OpenInferenceMimeTypeValues.TEXT,
    )

    assert input_attributes["input.value"] == '{"query": "Where is Zurich?"}'
    assert input_attributes["input.mime_type"] == "application/json"
    assert output_attributes["output.value"] == "Zurich is in Switzerland."
    assert output_attributes["output.mime_type"] == "text/plain"


def test_build_message_serializes_tool_results_and_tool_calls() -> None:
    message = ChatMessage(
        role="assistant",
        content={"status": "ok"},
        tool_calls=(ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="c1"),),
    )

    payload = build_message(message)

    assert payload.get("role") == "assistant"
    assert payload.get("content") == '{"status": "ok"}'
    assert payload.get("tool_calls") == [
        {
            "id": "c1",
            "function": {"name": "search_documents", "arguments": {"query": "q"}},
        }
    ]


def test_build_llm_attributes_includes_messages_and_tool_calls() -> None:
    tool_schema = ToolSchema(
        name="search_documents",
        description="Search docs",
        parameters_schema={"type": "object"},
    )
    attributes = build_llm_attributes(
        provider="ollama",
        model_name="qwen2.5-coder:14b",
        messages=(ChatMessage(role="user", content="hi"),),
        tools=(tool_schema,),
        response_text="answer",
        tool_calls=(
            ToolCallInfo(name="search_documents", arguments={"query": "q"}, call_id="tc1"),
        ),
        invocation_parameters={"stream": True},
    )

    assert attributes["llm.provider"] == "ollama"
    assert attributes["llm.model_name"] == "qwen2.5-coder:14b"
    assert attributes["llm.invocation_parameters"] == '{"stream": true}'
    assert attributes["llm.input_messages.0.message.role"] == "user"
    assert attributes["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == "tc1"


def test_build_tool_execution_attributes_sets_tool_name_and_parameters() -> None:
    attributes = build_tool_execution_attributes(
        tool_name="cite_sources",
        parameters={"citations": [{"source": "doc.txt", "chunk_id": "1"}]},
    )

    assert attributes["tool.name"] == "cite_sources"
    assert (
        attributes["tool.parameters"] == '{"citations": [{"source": "doc.txt", "chunk_id": "1"}]}'
    )


def test_build_document_and_retriever_attributes_include_document_payloads() -> None:
    chunk = SourceChunk(content="content", source="doc.txt", score=0.9, chunk_id="1")

    document = build_document(chunk)
    attributes = build_retriever_attributes(query="question", documents=(chunk,))

    assert document.get("id") == "1"
    assert document.get("metadata") == {
        "source": "doc.txt",
        "chunk_id": "1",
        "title": None,
        "author": None,
        "publication_date": None,
        "source_url": None,
        "page": None,
    }
    assert attributes["input.value"] == "question"
    assert attributes["retrieval.documents.0.document.id"] == "1"


def test_build_session_attributes_uses_openinference_session_key() -> None:
    attributes = build_session_attributes("session-123")

    assert attributes == {"session.id": "session-123"}


def test_build_tool_call_preserves_call_id_and_arguments() -> None:
    payload = build_tool_call(ToolCallInfo(name="tool", arguments={"x": 1}, call_id="abc"))

    assert payload == {"id": "abc", "function": {"name": "tool", "arguments": {"x": 1}}}
