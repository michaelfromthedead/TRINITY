"""engine.core.memory — Memory allocators and tracking for the game engine."""

from engine.core.memory.allocator import Allocator, AllocationInfo, MemoryTag
from engine.core.memory.linear import LinearAllocator
from engine.core.memory.stack import StackAllocator
from engine.core.memory.pool import PoolAllocator
from engine.core.memory.ring import RingAllocator
from engine.core.memory.slab import SlabAllocator
from engine.core.memory.tlsf import TLSFAllocator
from engine.core.memory.tracker import MemoryTracker, MemoryStats
from engine.core.memory.object_pool import ObjectPool

__all__ = [
    "Allocator",
    "AllocationInfo",
    "MemoryTag",
    "LinearAllocator",
    "StackAllocator",
    "PoolAllocator",
    "RingAllocator",
    "SlabAllocator",
    "TLSFAllocator",
    "MemoryTracker",
    "MemoryStats",
    "ObjectPool",
]
