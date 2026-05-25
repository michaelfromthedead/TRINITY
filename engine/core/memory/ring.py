"""Ring (circular) buffer allocator for streaming data."""

from __future__ import annotations

import logging

from engine.core.memory.allocator import Allocator

logger = logging.getLogger(__name__)


class RingAllocator(Allocator):
    """Circular allocator that wraps around when it reaches the end.

    Suitable for streaming or double-buffered data where old allocations
    are implicitly overwritten.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer = bytearray(capacity)
        self._head = 0
        self._used = 0

    # -- Allocator interface --------------------------------------------------

    def allocate(self, size: int) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        if size > self._capacity:
            raise MemoryError(
                f"RingAllocator: allocation {size} exceeds capacity {self._capacity}"
            )
        offset = self._head
        self._head = (self._head + size) % self._capacity
        self._used = min(self._used + size, self._capacity)
        logger.debug("ring alloc size=%d offset=%d head=%d", size, offset, self._head)
        return offset

    def free(self, offset: int) -> None:  # noqa: ARG002
        """No-op — ring allocators free implicitly by overwriting."""
        logger.warning("ring free ignored (implicit)")

    def reset(self) -> None:
        self._head = 0
        self._used = 0
        logger.debug("ring reset")

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def head(self) -> int:
        """Current write position."""
        return self._head

    @property
    def buffer(self) -> bytearray:
        return self._buffer
