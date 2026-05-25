"""Stack (LIFO) allocator with marker-based unwinding."""

from __future__ import annotations

import logging

from engine.core.memory.allocator import Allocator

logger = logging.getLogger(__name__)


class StackAllocator(Allocator):
    """LIFO allocator that supports marker-based bulk free.

    Use :meth:`get_marker` before a group of allocations and
    :meth:`free_to_marker` to unwind them all at once.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer = bytearray(capacity)
        self._offset = 0

    # -- Allocator interface --------------------------------------------------

    def allocate(self, size: int) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        if self._offset + size > self._capacity:
            raise MemoryError(
                f"StackAllocator out of memory: requested {size}, "
                f"available {self._capacity - self._offset}"
            )
        offset = self._offset
        self._offset += size
        logger.debug("stack alloc size=%d offset=%d", size, offset)
        return offset

    def free(self, offset: int) -> None:
        """Free back to *offset* (LIFO semantics)."""
        if offset < 0 or offset > self._offset:
            raise ValueError(f"invalid free offset {offset}")
        self._offset = offset
        logger.debug("stack free to offset=%d", offset)

    def reset(self) -> None:
        self._offset = 0
        logger.debug("stack reset")

    @property
    def used_bytes(self) -> int:
        return self._offset

    @property
    def capacity(self) -> int:
        return self._capacity

    # -- Marker API -----------------------------------------------------------

    def get_marker(self) -> int:
        """Return the current stack position as a marker."""
        return self._offset

    def free_to_marker(self, marker: int) -> None:
        """Unwind all allocations made after *marker*."""
        if marker < 0 or marker > self._offset:
            raise ValueError(f"invalid marker {marker}")
        self._offset = marker
        logger.debug("stack free_to_marker=%d", marker)

    @property
    def buffer(self) -> bytearray:
        return self._buffer
