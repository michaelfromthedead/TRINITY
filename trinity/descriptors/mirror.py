"""
Mirror descriptor - reads value from another field, denies direct writes.
"""

from __future__ import annotations

from typing import Any, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")

__all__ = ["MirrorDescriptor"]


class MirrorDescriptor(BaseDescriptor[T]):
    """
    Mirrors the value of another field on the same object.

    Reading returns the source field's value. Writing raises AttributeError.
    """

    __slots__ = ("_source_field",)

    descriptor_id = "mirror"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        source_field: str = "",
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        if not source_field:
            raise ValueError("source_field must be non-empty")
        self._source_field = source_field

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T:
        if obj is None:
            return self  # type: ignore
        self.pre_get(obj)
        raw_value = getattr(obj, self._source_field)
        return self.post_get(obj, raw_value)

    def __set__(self, obj: Any, value: T) -> None:
        raise AttributeError("Cannot set mirrored field directly")

    @property
    def descriptor_steps(self) -> list[Step]:
        return [Step(Op.INTERCEPT, {"get": "mirror_read", "set": "deny"})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["source_field"] = self._source_field
        return meta
