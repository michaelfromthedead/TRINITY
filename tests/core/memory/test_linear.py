"""Tests for LinearAllocator."""

import pytest

from engine.core.memory.linear import LinearAllocator


class TestLinearAllocator:
    def test_allocate_returns_sequential_offsets(self):
        a = LinearAllocator(256)
        assert a.allocate(10) == 0
        assert a.allocate(20) == 10
        assert a.allocate(30) == 30

    def test_used_bytes_tracks_allocations(self):
        a = LinearAllocator(256)
        a.allocate(64)
        a.allocate(32)
        assert a.used_bytes == 96

    def test_capacity(self):
        a = LinearAllocator(512)
        assert a.capacity == 512

    def test_overflow_raises(self):
        a = LinearAllocator(32)
        a.allocate(30)
        with pytest.raises(MemoryError):
            a.allocate(10)

    def test_exact_fit(self):
        a = LinearAllocator(64)
        a.allocate(64)
        assert a.used_bytes == 64

    def test_reset_reclaims_all(self):
        a = LinearAllocator(128)
        a.allocate(100)
        a.reset()
        assert a.used_bytes == 0
        # Can allocate again after reset
        assert a.allocate(128) == 0

    def test_free_is_noop(self):
        a = LinearAllocator(64)
        off = a.allocate(32)
        a.free(off)
        # used_bytes unchanged — free is a no-op
        assert a.used_bytes == 32

    def test_zero_or_negative_size_raises(self):
        a = LinearAllocator(64)
        with pytest.raises(ValueError):
            a.allocate(0)
        with pytest.raises(ValueError):
            a.allocate(-1)

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            LinearAllocator(0)
