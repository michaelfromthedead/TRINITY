"""Linear (bump) allocator — O(1) allocate, reset-only free."""

from __future__ import annotations

import logging

from engine.core.memory.allocator import Allocator

logger = logging.getLogger(__name__)


class LinearAllocator(Allocator):
    """Bump-pointer allocator backed by a fixed ``bytearray``.

    Individual allocations cannot be freed; call :meth:`reset` to reclaim all
    memory at once.  Ideal for per-frame scratch data.
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
                f"LinearAllocator out of memory: requested {size}, "
                f"available {self._capacity - self._offset}"
            )
        offset = self._offset
        self._offset += size
        logger.debug("linear alloc size=%d offset=%d", size, offset)
        return offset

    def free(self, offset: int) -> None:  # noqa: ARG002
        """No-op — linear allocators do not support individual frees."""
        logger.warning("linear free ignored (reset-only)")

    def reset(self) -> None:
        self._offset = 0
        logger.debug("linear reset")

    @property
    def used_bytes(self) -> int:
        return self._offset

    @property
    def capacity(self) -> int:
        return self._capacity

    # -- Convenience ----------------------------------------------------------

    @property
    def buffer(self) -> bytearray:
        """Direct access to the backing buffer."""
        return self._buffer
