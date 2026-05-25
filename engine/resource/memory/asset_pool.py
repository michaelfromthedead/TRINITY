"""Typed object pool for asset instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

from engine.resource.constants import DEFAULT_POOL_CAPACITY

T = TypeVar("T")


@dataclass
class PoolSlot:
    """A single slot in the asset pool."""

    __slots__ = ("slot_id", "is_active", "data")

    slot_id: int
    is_active: bool
    data: Any

    def __init__(self, slot_id: int, is_active: bool = False, data: Any = None) -> None:
        self.slot_id = slot_id
        self.is_active = is_active
        self.data = data


class AssetPool(Generic[T]):
    """Generic object pool that pre-allocates slots for asset instances."""

    __slots__ = ("_capacity", "_slots", "_free_ids")

    def __init__(self, capacity: int = DEFAULT_POOL_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("Pool capacity must be positive")
        self._capacity = capacity
        self._slots: list[PoolSlot] = [
            PoolSlot(slot_id=i) for i in range(capacity)
        ]
        self._free_ids: list[int] = list(range(capacity - 1, -1, -1))

    def acquire(self, obj: T) -> tuple[int, T]:
        """Acquire a slot, returning (slot_id, object). Raises RuntimeError if full."""
        if not self._free_ids:
            raise RuntimeError("Asset pool is full")
        slot_id = self._free_ids.pop()
        slot = self._slots[slot_id]
        slot.is_active = True
        slot.data = obj
        return slot_id, obj

    def release(self, slot_id: int) -> None:
        """Release a slot back to the pool."""
        if slot_id < 0 or slot_id >= self._capacity:
            raise IndexError(f"Invalid slot_id: {slot_id}")
        slot = self._slots[slot_id]
        if not slot.is_active:
            raise ValueError(f"Slot {slot_id} is not active")
        slot.is_active = False
        slot.data = None
        self._free_ids.append(slot_id)

    def get(self, slot_id: int) -> Optional[T]:
        """Get object at slot_id, or None if inactive/invalid."""
        if slot_id < 0 or slot_id >= self._capacity:
            return None
        slot = self._slots[slot_id]
        if not slot.is_active:
            return None
        return slot.data

    def active_count(self) -> int:
        """Number of currently active slots."""
        return self._capacity - len(self._free_ids)

    def capacity(self) -> int:
        """Total pool capacity."""
        return self._capacity

    def is_full(self) -> bool:
        """Whether the pool has no free slots."""
        return len(self._free_ids) == 0

    def reset(self) -> None:
        """Release all slots."""
        for slot in self._slots:
            slot.is_active = False
            slot.data = None
        self._free_ids = list(range(self._capacity - 1, -1, -1))
