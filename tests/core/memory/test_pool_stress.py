"""Stress tests for PoolAllocator."""

from engine.core.memory.pool import PoolAllocator


class TestPoolAllocatorStress:
    def test_stress_lifo_reuse_order(self) -> None:
        """After freeing all in LIFO order, re-allocation returns original indices."""
        p = PoolAllocator(element_size=16, count=1000)
        indices = [p.allocate() for _ in range(1000)]
        # Free in reverse (LIFO) order
        for idx in reversed(indices):
            p.free(idx)
        reused = [p.allocate() for _ in range(1000)]
        # LIFO pop returns original order, not reversed
        assert reused == indices
