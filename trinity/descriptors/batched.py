"""
Batched descriptor - collects writes and flushes in batches.
"""

from __future__ import annotations

from typing import Any, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")

__all__ = ["BatchedDescriptor", "flush_batch"]

_DEFAULT_BATCH_SIZE = 10


class BatchedDescriptor(BaseDescriptor[T]):
    """
    Collects field writes and triggers a flush when batch_size is reached.

    The owner object can implement ``_flush_batch(field_name)`` to handle
    the batch flush.
    """

    __slots__ = ("_batch_size", "_batch_key", "_count_key")

    descriptor_id = "batched"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._batch_size = batch_size
        self._batch_key = ""
        self._count_key = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._batch_key = f"_batch_{name}"
        self._count_key = f"_batch_count_{name}"

    def _ensure_keys(self, obj: Any) -> None:
        if self._batch_key not in obj.__dict__:
            obj.__dict__[self._batch_key] = []
        if self._count_key not in obj.__dict__:
            obj.__dict__[self._count_key] = 0

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        self._ensure_keys(obj)
        bk = self._batch_key
        ck = self._count_key
        obj.__dict__[bk].append(value)
        obj.__dict__[ck] += 1
        if obj.__dict__[ck] >= self._batch_size:
            if hasattr(obj, "_flush_batch"):
                obj._flush_batch(self._name)
            obj.__dict__[ck] = 0
            obj.__dict__[bk] = []

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.INTERCEPT, {"set": "batch_collect"}),
            Step(Op.TAG, {"key": "batched", "value": True}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["batch_size"] = self._batch_size
        return meta


def flush_batch(obj: Any, field: str) -> None:
    """Manually trigger a batch flush for the given field."""
    if hasattr(obj, "_flush_batch"):
        obj._flush_batch(field)
    bk = f"_batch_{field}"
    ck = f"_batch_count_{field}"
    obj.__dict__[bk] = []
    obj.__dict__[ck] = 0
