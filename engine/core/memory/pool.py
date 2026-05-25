"""Fixed-size pool allocator — O(1) allocate / free via embedded free list."""

from __future__ import annotations

import logging
from typing import List

from engine.core.memory.allocator import Allocator

logger = logging.getLogger(__name__)


class PoolAllocator(Allocator):
    """Pre-allocated pool of fixed-size slots.

    Slots are referenced by **index** (0-based).  A free list tracks which
    indices are available.
    """

    def __init__(self, element_size: int, count: int) -> None:
        if element_size <= 0:
            raise ValueError("element_size must be positive")
        if count <= 0:
            raise ValueError("count must be positive")
        self._element_size = element_size
        self._count = count
        self._buffer = bytearray(element_size * count)
        # Free list: available indices (LIFO order for cache friendliness)
        self._free_list: List[int] = list(range(count - 1, -1, -1))
        self._free_set: set[int] = set(self._free_list)
        self._allocated = 0

    # -- Allocator interface --------------------------------------------------

    def allocate(self, size: int = 0) -> int:  # noqa: ARG002
        """Return the index of a free slot.

        *size* is ignored (each slot is ``element_size`` bytes).
        """
        if not self._free_list:
            raise MemoryError("PoolAllocator is full")
        index = self._free_list.pop()
        self._free_set.discard(index)
        self._allocated += 1
        logger.debug("pool alloc index=%d", index)
        return index

    def free(self, offset: int) -> None:
        """Return the slot at *offset* (index) to the free list."""
        if offset < 0 or offset >= self._count:
            raise ValueError(f"invalid slot index {offset}")
        if offset in self._free_set:
            raise ValueError(f"double free on slot {offset}")
        self._free_list.append(offset)
        self._free_set.add(offset)
        self._allocated -= 1
        logger.debug("pool free index=%d", offset)

    def reset(self) -> None:
        self._free_list = list(range(self._count - 1, -1, -1))
        self._free_set = set(self._free_list)
        self._allocated = 0
        logger.debug("pool reset")

    @property
    def used_bytes(self) -> int:
        return self._allocated * self._element_size

    @property
    def capacity(self) -> int:
        return self._count * self._element_size

    # -- Pool-specific --------------------------------------------------------

    @property
    def element_size(self) -> int:
        return self._element_size

    @property
    def is_full(self) -> bool:
        return len(self._free_list) == 0

    @property
    def available_count(self) -> int:
        return len(self._free_list)

    @property
    def buffer(self) -> bytearray:
        return self._buffer
