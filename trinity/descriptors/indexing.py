"""
Indexed descriptor - maintains per-value indexes for fast lookup.

Provides indexing and optional uniqueness constraints for Trinity fields.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional, TypeVar

from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class IndexedDescriptor(BaseDescriptor[T]):
    """
    Maintains an index mapping field values to object ids.

    Supports optional uniqueness enforcement.
    """

    __slots__ = ("_unique",)

    descriptor_id = "indexed"

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        unique: bool = False,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._unique = unique

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        index_attr = f"_index_{name}"
        if not hasattr(owner, index_attr):
            setattr(owner, index_attr, defaultdict(set))

    def _get_index(self, obj: Any) -> defaultdict[Any, set[int]]:
        return getattr(type(obj), f"_index_{self._name}")

    def pre_set(self, obj: Any, value: T) -> T:
        # Remove obj from old value's index entry
        old_value = self._get_stored_safe(obj)
        if old_value is not None:
            index = self._get_index(obj)
            index[old_value].discard(id(obj))
        return value

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        index = self._get_index(obj)
        if self._unique and index[value] and id(obj) not in index[value]:
            # Rollback: remove from new, restore old
            index[value].discard(id(obj))
            if old_value is not None:
                index[old_value].add(id(obj))
                self._set_stored(obj, old_value)
            raise ValueError(
                f"Unique constraint violated for field '{self._name}' "
                f"with value {value!r}"
            )
        index[value].add(id(obj))

    @classmethod
    def find_by(cls, owner_cls: type, field_name: str, value: Any) -> set[int]:
        """Return set of object ids indexed under the given value."""
        index_attr = f"_index_{field_name}"
        index = getattr(owner_cls, index_attr, {})
        return set(index.get(value, set()))

    @property
    def descriptor_steps(self) -> list:
        from trinity.decorators.ops import Op, Step
        return [
            Step(Op.INTERCEPT, {"set": "index_update"}),
            Step(Op.REGISTER, {"registry": f"index:{self._name}"}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["unique"] = self._unique
        return meta


def find_by_index(cls: type, field_name: str, value: Any) -> set[int]:
    """Module-level helper to query an indexed field."""
    return IndexedDescriptor.find_by(cls, field_name, value)


__all__ = ["IndexedDescriptor", "find_by_index"]
