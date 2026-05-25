"""Tests for TLSFAllocator."""

import pytest

from engine.core.memory.tlsf import TLSFAllocator


class TestTLSFAllocator:
    def test_basic_allocate(self):
        t = TLSFAllocator(1024)
        off = t.allocate(64)
        assert off == 0
        assert t.used_bytes == 64

    def test_multiple_allocations(self):
        t = TLSFAllocator(1024)
        offsets = [t.allocate(32) for _ in range(4)]
        # All offsets should be unique
        assert len(set(offsets)) == 4

    def test_free_and_reuse(self):
        t = TLSFAllocator(256)
        off = t.allocate(64)
        assert t.used_bytes == 64
        t.free(off, 64)
        assert t.used_bytes == 0
        # Re-allocate same size; freed space should be reused
        off2 = t.allocate(64)
        assert off2 == off
        assert t.used_bytes == 64

    def test_overflow_raises(self):
        t = TLSFAllocator(64)
        t.allocate(48)
        with pytest.raises(MemoryError):
            t.allocate(48)

    def test_reset(self):
        t = TLSFAllocator(256)
        t.allocate(100)
        t.allocate(100)
        t.reset()
        assert t.used_bytes == 0
        # Full capacity available again
        t.allocate(200)

    def test_fragmentation_handling(self):
        """Allocate, free alternating blocks, then allocate a larger block."""
        t = TLSFAllocator(1024)
        offsets = []
        for _ in range(8):
            offsets.append(t.allocate(64))
        # Free every other block
        for i in range(0, 8, 2):
            t.free(offsets[i], 64)
        # Should be able to allocate a 64-byte block from freed space
        freed_offsets = {offsets[i] for i in range(0, 8, 2)}
        used_before = t.used_bytes
        off = t.allocate(64)
        assert off in freed_offsets
        assert t.used_bytes == used_before + 64

    def test_small_capacity_raises(self):
        with pytest.raises(ValueError):
            TLSFAllocator(4)
