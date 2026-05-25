"""Base allocator ABC and shared types for the memory subsystem."""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class MemoryTag(enum.Enum):
    """Categories for memory allocations, used by MemoryTracker for budgeting."""
    UNKNOWN = 0
    CORE = 1
    RENDERING = 2
    PHYSICS = 3
    ANIMATION = 4
    AUDIO = 5
    GAMEPLAY = 6
    UI = 7
    NETWORK = 8


@dataclass(slots=True)
class AllocationInfo:
    """Describes a single allocation for tracking purposes."""
    offset: int
    size: int
    tag: MemoryTag = MemoryTag.UNKNOWN


class Allocator(ABC):
    """Abstract base for all memory allocators.

    Allocators manage a backing ``bytearray`` and hand out integer offsets.
    """

    @abstractmethod
    def allocate(self, size: int) -> int:
        """Reserve *size* bytes and return the offset into the backing buffer."""
        ...

    @abstractmethod
    def free(self, offset: int) -> None:
        """Release a previous allocation at *offset* (if supported)."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Bulk-free all allocations."""
        ...

    @property
    @abstractmethod
    def used_bytes(self) -> int:
        """Number of bytes currently in use."""
        ...

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Total capacity in bytes."""
        ...
