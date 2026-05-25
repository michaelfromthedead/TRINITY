"""Tests for SlabAllocator."""

import pytest

from engine.core.memory.slab import SlabAllocator


class TestSlabAllocator:
    def test_size_class_selection(self):
        s = SlabAllocator(size_classes=[16, 64, 256])
        sc, _ = s.allocate_slab(10)
        assert sc == 16
        sc, _ = s.allocate_slab(17)
        assert sc == 64
        sc, _ = s.allocate_slab(65)
        assert sc == 256

    def test_exact_size_class(self):
        s = SlabAllocator(size_classes=[32, 64])
        sc, _ = s.allocate_slab(32)
        assert sc == 32

    def test_too_large_raises(self):
        s = SlabAllocator(size_classes=[16, 32])
        with pytest.raises(MemoryError):
            s.allocate_slab(64)

    def test_free(self):
        s = SlabAllocator(size_classes=[32], slots_per_class=2)
        sc, i0 = s.allocate_slab(10)
        _, i1 = s.allocate_slab(10)
        s.free_slab(sc, i0)
        # Can allocate again
        _, i2 = s.allocate_slab(10)
        assert i2 == i0

    def test_unknown_class_free_raises(self):
        s = SlabAllocator(size_classes=[16])
        with pytest.raises(ValueError):
            s.free_slab(999, 0)

    def test_reset(self):
        s = SlabAllocator(size_classes=[32], slots_per_class=2)
        s.allocate_slab(10)
        s.allocate_slab(10)
        s.reset()
        pool = s.pool_for(32)
        assert pool.available_count == 2

    def test_allocator_interface(self):
        """Test the Allocator ABC interface (encoded int offsets)."""
        s = SlabAllocator(size_classes=[32, 64], slots_per_class=4)
        off = s.allocate(10)
        assert isinstance(off, int)
        assert s.used_bytes > 0
        s.free(off)

    def test_allocator_capacity(self):
        s = SlabAllocator(size_classes=[32, 64], slots_per_class=4)
        assert s.capacity == 32 * 4 + 64 * 4
