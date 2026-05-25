"""
SchemaDescriptor — validate values against a JSON-schema-like dict.
"""

from __future__ import annotations

from typing import Any, Optional

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

__all__ = ["SchemaDescriptor"]

_TYPE_MAP = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": (int, float),
    "bool": bool,
    "boolean": bool,
    "list": list,
    "array": list,
    "dict": dict,
    "object": dict,
}


class SchemaDescriptor(BaseDescriptor):
    """Descriptor that validates values against a schema dict."""

    __slots__ = ("_schema",)

    descriptor_id: str = "schema"

    def __init__(
        self,
        schema: dict,
        field_type: type = object,
        inner: Optional[BaseDescriptor] = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._schema = schema

    def pre_set(self, obj: Any, value: Any) -> Any:
        self._validate(value)
        return value

    def _validate(self, value: Any) -> None:
        schema_type = self._schema.get("type")
        if schema_type is not None:
            expected = _TYPE_MAP.get(schema_type)
            if expected is not None and not isinstance(value, expected):
                raise TypeError(
                    f"Expected type '{schema_type}', got {type(value).__name__}"
                )

        required = self._schema.get("required")
        if required and isinstance(value, dict):
            for key in required:
                if key not in value:
                    raise ValueError(f"Missing required key: '{key}'")

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.VALIDATE, {"constraint": "schema"}),
            Step(Op.DESCRIBE, {}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["schema"] = self._schema
        return meta
