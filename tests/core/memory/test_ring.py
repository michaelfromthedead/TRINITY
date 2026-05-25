"""Tests for RingAllocator."""

import pytest

from engine.core.memory.ring import RingAllocator


class TestRingAllocator:
    def test_allocate_sequential(self):
        r = RingAllocator(128)
        assert r.allocate(32) == 0
        assert r.allocate(32) == 32

    def test_wrap_around(self):
        r = RingAllocator(64)
        r.allocate(50)
        # Next allocation wraps
        off = r.allocate(30)
        assert off == 50
        assert r.head == (50 + 30) % 64  # == 16

    def test_used_capped_at_capacity(self):
        r = RingAllocator(64)
        r.allocate(40)
        r.allocate(40)
        assert r.used_bytes == 64  # capped

    def test_exceeds_capacity_raises(self):
        r = RingAllocator(32)
        with pytest.raises(MemoryError):
            r.allocate(64)

    def test_reset(self):
        r = RingAllocator(128)
        r.allocate(100)
        r.reset()
        assert r.used_bytes == 0
        assert r.head == 0

    def test_free_is_noop(self):
        r = RingAllocator(64)
        off = r.allocate(32)
        r.free(off)
        assert r.used_bytes == 32
