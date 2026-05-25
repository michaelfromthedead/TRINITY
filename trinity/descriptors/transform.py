"""
Transform descriptor - apply read/write transformations.

Transforms values on read and/or write through user-supplied callables.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class TransformDescriptor(BaseDescriptor[T]):
    """
    Applies transformations on read and/or write.

    - write_transform: called in pre_set to transform value before storage.
    - read_transform: called in post_get to transform value on retrieval.
    """

    __slots__ = ("_read_transform", "_write_transform")

    descriptor_id = "transform"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        read_transform: Optional[Callable] = None,
        write_transform: Optional[Callable] = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._read_transform = read_transform
        self._write_transform = write_transform

    def pre_set(self, obj: Any, value: T) -> T:
        """Apply write transform before storing."""
        if self._write_transform is not None:
            return self._write_transform(value)
        return value

    def post_get(self, obj: Any, value: T) -> T:
        """Apply read transform after retrieval."""
        if self._read_transform is not None:
            return self._read_transform(value)
        return value

    @property
    def descriptor_steps(self) -> list[Step]:
        return [Step(Op.INTERCEPT, {"get": "read_transform", "set": "write_transform"})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["has_read_transform"] = self._read_transform is not None
        meta["has_write_transform"] = self._write_transform is not None
        return meta


__all__ = ["TransformDescriptor"]
