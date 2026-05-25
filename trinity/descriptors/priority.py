"""
Priority descriptor - tags entities with a processing priority.
"""

from __future__ import annotations

from typing import Any, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")

__all__ = ["PriorityDescriptor"]


class PriorityDescriptor(BaseDescriptor[T]):
    """
    Tags the field with a numeric priority (higher = processed first).

    This is a simple marker descriptor with no special get/set behavior.
    """

    __slots__ = ("_priority",)

    descriptor_id = "priority"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        priority: int = 0,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._priority = priority

    @property
    def descriptor_steps(self) -> list[Step]:
        return [Step(Op.TAG, {"key": "priority", "value": self._priority})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["priority"] = self._priority
        return meta
