"""
Sparse descriptor - stores only non-default values.

Optimizes memory by not storing values equal to the default,
using a class-level dictionary keyed by (object_id, field_name).
"""

from __future__ import annotations

from typing import Any, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class SparseDescriptor(BaseDescriptor[T]):
    """
    Descriptor that only stores non-default values.

    Uses a class-level sparse store dictionary keyed by (id(obj), field_name).
    When a value equals the default, it is removed from storage.
    """

    __slots__ = ("_default",)

    descriptor_id = "sparse"
    accepts_inner: tuple[str, ...] = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        default: Any = None,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._default = default

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        attr = f"_sparse_{name}"
        if not hasattr(owner, attr):
            setattr(owner, attr, {})

    def _get_store(self, obj: Any) -> dict:
        return getattr(type(obj), f"_sparse_{self._name}")

    def _get_stored(self, obj: Any) -> T:
        """Retrieve from the sparse store instead of obj.__dict__."""
        store = self._get_store(obj)
        return store.get((id(obj), self._name), self._default)

    def _get_stored_safe(self, obj: Any) -> Optional[T]:
        """Retrieve from sparse store, returning None if absent."""
        store = self._get_store(obj)
        return store.get((id(obj), self._name))

    def _set_stored(self, obj: Any, value: T) -> None:
        """Store in the sparse store; remove entry if value equals default."""
        store = self._get_store(obj)
        key = (id(obj), self._name)
        if value == self._default:
            store.pop(key, None)
        else:
            store[key] = value

    def _delete_stored(self, obj: Any) -> None:
        """Remove entry from the sparse store."""
        store = self._get_store(obj)
        store.pop((id(obj), self._name), None)

    @property
    def descriptor_steps(self) -> list[Step]:
        return [Step(Op.INTERCEPT, {"get": "sparse_get", "set": "sparse_set"})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["default"] = self._default
        return meta


def sparse_count(cls: type, field: str) -> int:
    """Count non-default entries for a sparse field across all instances."""
    store = getattr(cls, f"_sparse_{field}", {})
    return sum(1 for k in store if k[1] == field)


__all__ = ["SparseDescriptor", "sparse_count"]
