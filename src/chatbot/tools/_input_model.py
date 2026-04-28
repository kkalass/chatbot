"""Common Pydantic base model for LLM-facing tool argument parsing.

LLM tool-calls sometimes serialize structured arguments as JSON strings
(e.g. a list field arrives as '[]' text). This base model applies a
field-level pre-validation coercion for structured fields only.
"""

from __future__ import annotations

import json
from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ValidationInfo, field_validator


class ToolInputModel(BaseModel):
    """Base model that coerces JSON-serialized structured field values."""

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_json_string_for_structured_fields(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if not isinstance(value, str):
            return value

        field_name = info.field_name
        if field_name is None:
            return value

        model_field = cls.model_fields.get(field_name)
        if model_field is None:
            return value

        if not _expects_structured_value(model_field.annotation):
            return value

        stripped = value.strip()
        if not stripped or stripped[0] not in "[{":
            return value

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value


def _expects_structured_value(annotation: Any) -> bool:
    origin = get_origin(annotation)

    if origin is None:
        return annotation in (list, dict)

    if origin is Union or origin is UnionType:
        return any(
            _expects_structured_value(arg) for arg in get_args(annotation) if arg is not type(None)
        )

    if origin is list or origin is dict:
        return True

    if str(origin) == "typing.Annotated":
        args = get_args(annotation)
        return _expects_structured_value(args[0]) if args else False

    return False
