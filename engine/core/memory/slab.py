"""Slab allocator — routes allocations to the smallest fitting size class."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from engine.core.constants import DEFAULT_SLAB_SIZE_CLASSES, DEFAULT_SLAB_SLOTS_PER_CLASS
from engine.core.memory.allocator import Allocator
from engine.core.memory.pool import PoolAllocator

logger = logging.getLogger(__name__)

_DEFAULT_SIZE_CLASSES: List[int] = list(DEFAULT_SLAB_SIZE_CLASSES)
_DEFAULT_SLOTS_PER_CLASS = DEFAULT_SLAB_SLOTS_PER_CLASS


class SlabAllocator(Allocator):
    """Multi-pool allocator with power-of-two size classes.

    Each size class is backed by a :class:`PoolAllocator`.  An allocation is
    routed to the smallest class that can satisfy the request.
    """

    def __init__(
        self,
        size_classes: Optional[List[int]] = None,
        slots_per_class: int = _DEFAULT_SLOTS_PER_CLASS,
    ) -> None:
        self._size_classes = sorted(size_classes or _DEFAULT_SIZE_CLASSES)
        if not self._size_classes:
            raise ValueError("at least one size class required")
        self._pools: Dict[int, PoolAllocator] = {
            sc: PoolAllocator(element_size=sc, count=slots_per_class)
            for sc in self._size_classes
        }

    # -- Allocator interface ---------------------------------------------------

    def allocate(self, size: int) -> int:
        """Allocate *size* bytes. Returns encoded offset (size_class_index << 32 | slot_index)."""
        sc = self._pick_class(size)
        if sc is None:
            raise MemoryError(
                f"SlabAllocator: no size class fits {size} "
                f"(max class {self._size_classes[-1]})"
            )
        index = self._pools[sc].allocate()
        sc_idx = self._size_classes.index(sc)
        logger.debug("slab alloc size=%d class=%d index=%d", size, sc, index)
        return (sc_idx << 32) | index

    def free(self, offset: int) -> None:
        """Free an encoded offset returned by :meth:`allocate`."""
        sc_idx = (offset >> 32) & 0xFFFFFFFF
        index = offset & 0xFFFFFFFF
        if sc_idx >= len(self._size_classes):
            raise ValueError(f"invalid encoded offset: bad size class index {sc_idx}")
        size_class = self._size_classes[sc_idx]
        self._pools[size_class].free(index)
        logger.debug("slab free class=%d index=%d", size_class, index)

    def reset(self) -> None:
        for pool in self._pools.values():
            pool.reset()

    @property
    def used_bytes(self) -> int:
        return sum(pool.used_bytes for pool in self._pools.values())

    @property
    def capacity(self) -> int:
        return sum(pool.capacity for pool in self._pools.values())

    # -- Legacy tuple interface ------------------------------------------------

    def allocate_slab(self, size: int) -> Tuple[int, int]:
        """Allocate *size* bytes and return ``(size_class, slot_index)``."""
        sc = self._pick_class(size)
        if sc is None:
            raise MemoryError(
                f"SlabAllocator: no size class fits {size} "
                f"(max class {self._size_classes[-1]})"
            )
        index = self._pools[sc].allocate()
        logger.debug("slab alloc size=%d class=%d index=%d", size, sc, index)
        return sc, index

    def free_slab(self, size_class: int, index: int) -> None:
        """Return a slot to its size-class pool."""
        if size_class not in self._pools:
            raise ValueError(f"unknown size class {size_class}")
        self._pools[size_class].free(index)
        logger.debug("slab free class=%d index=%d", size_class, index)

    # -- Properties -----------------------------------------------------------

    @property
    def size_classes(self) -> List[int]:
        return list(self._size_classes)

    def pool_for(self, size_class: int) -> PoolAllocator:
        """Return the underlying pool for a given size class."""
        return self._pools[size_class]

    # -- Internal -------------------------------------------------------------

    def _pick_class(self, size: int) -> Optional[int]:
        for sc in self._size_classes:
            if sc >= size:
                return sc
        return None
