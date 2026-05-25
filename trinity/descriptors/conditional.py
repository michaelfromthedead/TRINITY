"""
Conditional descriptor - gate writes with a predicate.

Only allows writes when a user-supplied predicate returns True.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class WriteConditionError(Exception):
    """Raised when a write condition predicate returns False."""
    pass


class ConditionalDescriptor(BaseDescriptor[T]):
    """
    Gates writes behind a predicate function.

    The predicate receives (obj, name, old_value, new_value) and must
    return True to allow the write.
    """

    __slots__ = ("_predicate",)

    descriptor_id = "conditional"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ("conditional",)

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        predicate: Optional[Callable[[Any, str, Any, Any], bool]] = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        if predicate is None:
            raise ValueError("ConditionalDescriptor requires a predicate")
        self._predicate = predicate

    def pre_set(self, obj: Any, value: T) -> T:
        """Check predicate before allowing write."""
        old_value = self._get_stored_safe(obj)
        if not self._predicate(obj, self._name, old_value, value):
            raise WriteConditionError(
                f"Write condition failed for '{self._name}': "
                f"old={old_value!r}, new={value!r}"
            )
        return value

    @property
    def descriptor_steps(self) -> list[Step]:
        return [Step(Op.INTERCEPT, {"set": "condition_check"})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["predicate"] = getattr(self._predicate, "__name__", repr(self._predicate))
        return meta


__all__ = ["ConditionalDescriptor", "WriteConditionError"]
