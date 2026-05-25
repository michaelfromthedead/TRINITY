"""Tests for MemoryTracker."""

import pytest

from engine.core.memory.allocator import AllocationInfo, MemoryTag
from engine.core.memory.tracker import MemoryTracker, MemoryStats


class TestMemoryTracker:
    def test_track_allocation(self):
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=64, tag=MemoryTag.GAMEPLAY))
        stats = t.get_stats(MemoryTag.GAMEPLAY)
        assert stats.allocated == 64
        assert stats.current == 64
        assert stats.peak == 64

    def test_track_free(self):
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=64, tag=MemoryTag.CORE))
        t.track_free(0)
        stats = t.get_stats(MemoryTag.CORE)
        assert stats.freed == 64
        assert stats.current == 0
        assert stats.peak == 64

    def test_peak_tracking(self):
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=100, tag=MemoryTag.RENDERING))
        t.track_allocation(AllocationInfo(offset=100, size=50, tag=MemoryTag.RENDERING))
        t.track_free(0)
        stats = t.get_stats(MemoryTag.RENDERING)
        assert stats.peak == 150
        assert stats.current == 50

    def test_get_live_allocations(self):
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=32, tag=MemoryTag.UI))
        t.track_allocation(AllocationInfo(offset=32, size=64, tag=MemoryTag.AUDIO))
        t.track_free(0)
        live = t.get_live_allocations()
        assert len(live) == 1
        assert live[0].offset == 32

    def test_get_total_stats(self):
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=50, tag=MemoryTag.CORE))
        t.track_allocation(AllocationInfo(offset=50, size=30, tag=MemoryTag.PHYSICS))
        total = t.get_total_stats()
        assert total.allocated == 80
        assert total.current == 80

    def test_unknown_free_is_safe(self):
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=64, tag=MemoryTag.CORE))
        stats_before = t.get_total_stats()
        # Freeing unknown offset should not change stats
        t.track_free(999)
        stats_after = t.get_total_stats()
        assert stats_after.allocated == stats_before.allocated
        assert stats_after.freed == stats_before.freed
        assert stats_after.current == stats_before.current

    def test_leak_detection_via_live(self):
        """Live allocations after expected cleanup indicate leaks."""
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=128, tag=MemoryTag.NETWORK))
        # "Forgot" to free
        leaks = t.get_live_allocations()
        assert len(leaks) == 1
        assert leaks[0].tag == MemoryTag.NETWORK
