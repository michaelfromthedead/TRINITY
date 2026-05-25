"""Stress tests for RingAllocator."""

from engine.core.memory.ring import RingAllocator


class TestRingAllocatorStress:
    def test_stress_wrap_offset(self) -> None:
        """Third allocation wraps head back to 0."""
        r = RingAllocator(1024)
        off1 = r.allocate(512)
        assert off1 == 0
        off2 = r.allocate(512)
        assert off2 == 512
        # head is now (512 + 512) % 1024 = 0
        off3 = r.allocate(512)
        # third alloc wraps head to 0
        assert off3 == 0
