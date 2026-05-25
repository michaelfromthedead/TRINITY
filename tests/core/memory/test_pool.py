"""Tests for PoolAllocator."""

import pytest

from engine.core.memory.pool import PoolAllocator


class TestPoolAllocator:
    def test_allocate_returns_indices(self):
        p = PoolAllocator(element_size=32, count=4)
        indices = {p.allocate() for _ in range(4)}
        assert indices == {0, 1, 2, 3}

    def test_full_pool_raises(self):
        p = PoolAllocator(element_size=16, count=2)
        p.allocate()
        p.allocate()
        assert p.is_full
        with pytest.raises(MemoryError):
            p.allocate()

    def test_free_and_reuse(self):
        p = PoolAllocator(element_size=16, count=2)
        i0 = p.allocate()
        i1 = p.allocate()
        p.free(i0)
        assert p.available_count == 1
        reused = p.allocate()
        assert reused == i0

    def test_double_free_raises(self):
        p = PoolAllocator(element_size=16, count=4)
        idx = p.allocate()
        p.free(idx)
        with pytest.raises(ValueError):
            p.free(idx)

    def test_invalid_free_raises(self):
        p = PoolAllocator(element_size=16, count=4)
        with pytest.raises(ValueError):
            p.free(-1)
        with pytest.raises(ValueError):
            p.free(4)

    def test_reset(self):
        p = PoolAllocator(element_size=16, count=4)
        for _ in range(4):
            p.allocate()
        p.reset()
        assert p.available_count == 4
        assert p.used_bytes == 0

    def test_used_bytes(self):
        p = PoolAllocator(element_size=32, count=4)
        p.allocate()
        p.allocate()
        assert p.used_bytes == 64

    def test_capacity(self):
        p = PoolAllocator(element_size=32, count=4)
        assert p.capacity == 128
