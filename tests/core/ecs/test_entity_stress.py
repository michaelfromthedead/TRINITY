"""Stress tests for EntityAllocator generation lifecycle."""

from engine.core.ecs.entity import EntityAllocator


class TestEntityAllocatorStress:
    def test_stress_stale_handle_detection(self) -> None:
        """Stale handle is detected as dead before generation wraps."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        stale = e  # save handle
        for _ in range(255):
            alloc.deallocate(e)
            e = alloc.allocate()
        # After 256 cycles gen would wrap to 0, making stale indistinguishable.
        # With 255 cycles stale gen=0 != current gen=255, so still detected dead.
        assert not alloc.is_alive(stale)
        assert e.generation == 255
