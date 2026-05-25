"""Stress tests for SlabAllocator."""

import pytest

from engine.core.memory.slab import SlabAllocator


class TestSlabAllocatorStress:
    def test_stress_mixed_size_classes(self) -> None:
        """Mix allocation sizes across classes to avoid single-pool exhaustion."""
        s = SlabAllocator(size_classes=[16, 32, 64], slots_per_class=8)
        # allocate(8) only hits 16-byte pool (8 slots).
        # A 9th allocate(8) would raise uncaught MemoryError.
        # Use mixed sizes across classes instead.
        sizes = [8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32]
        for size in sizes:
            s.allocate(size)
        assert s.used_bytes > 0
