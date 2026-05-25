"""Blackbox tests: entity lifecycle survives 256+ recycle cycles.

This is a strict blackbox test suite. No implementation details are read or
relied upon. We test only the public API surface:

    EntityAllocator()
        .allocate() -> Entity
        .deallocate(entity)
        .is_alive(entity) -> bool

    Entity(index, generation)
        .is_valid() -> bool
        .null() -> Entity
        .index, .generation

Acceptance criteria:
    - Entity generation handles 256+ cycles without stale detection failure
    - All existing memory allocator tests still pass
"""

import pytest

from engine.core.ecs.entity import Entity, EntityAllocator


class TestEntityLifecycle256Plus:
    """Entity lifecycle survives 256+ recycle cycles (blackbox).

    Tests focus on stale handle detection across the 256-cycle generation
    boundary. A stale handle (saved at an earlier generation) must never
    be reported as alive after the slot has been recycled past it, even
    when the generation counter wraps.
    """

    # ------------------------------------------------------------------
    # 1 -- Exact boundary: 256 cycles
    # ------------------------------------------------------------------

    def test_alive_after_256_cycles(self):
        """A recycled entity is alive after exactly 256 cycles."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        for _ in range(256):
            alloc.deallocate(e)
            e = alloc.allocate()
        assert alloc.is_alive(e), "current entity should be alive after 256 cycles"

    def test_stale_handle_dead_after_256_cycles(self):
        """Stale handle from before any recycling is dead at 256 cycles.

        This tests the 8-bit wrap boundary: a handle saved at gen=0 must
        NOT appear alive after 256 cycles return gen to 0. If the generation
        counter is 8-bit this fails; a 16-bit counter (or wider) makes it pass.
        """
        alloc = EntityAllocator()
        e = alloc.allocate()
        stale = e
        for _ in range(256):
            alloc.deallocate(e)
            e = alloc.allocate()
        assert not alloc.is_alive(stale), (
            "stale handle from before 256 cycles must be dead "
            "(handle gen=0 should not match recycled gen=256)"
        )

    def test_stale_handle_generation_collision_after_256(self):
        """A stale saved mid-cycle, when gen wraps back to same value, is dead.

        Save a handle at generation X, then cycle 256 more times so gen
        returns to X. With an 8-bit counter the stale would appear alive
        (generation collision). With a 16-bit counter gen = X+256 > X, so
        the stale is correctly dead.
        """
        alloc = EntityAllocator()
        e = alloc.allocate()

        # Cycle to generation 100
        for _ in range(100):
            alloc.deallocate(e)
            e = alloc.allocate()
        assert e.generation == 100
        mid_stale = e  # saved at gen=100

        # Cycle exactly 256 more times (gen returns to 100 if 8-bit)
        for _ in range(256):
            alloc.deallocate(e)
            e = alloc.allocate()

        assert not alloc.is_alive(mid_stale), (
            "stale saved at gen=100 must be dead after 256 more cycles "
            "(no collision on wrap)"
        )

    # ------------------------------------------------------------------
    # 2 -- Beyond one wrap: 257, 512, 1000 cycles
    # ------------------------------------------------------------------

    def test_alive_after_257_cycles(self):
        """Entity is alive after 257 cycles (one past full 8-bit wrap)."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        for _ in range(257):
            alloc.deallocate(e)
            e = alloc.allocate()
        assert alloc.is_alive(e)

    def test_stale_dead_after_257_cycles(self):
        """Stale handle dead after 257 cycles."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        stale = e
        for _ in range(257):
            alloc.deallocate(e)
            e = alloc.allocate()
        assert not alloc.is_alive(stale)

    def test_alive_after_512_cycles(self):
        """Entity is alive after 512 cycles (two full 8-bit wraps)."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        for _ in range(512):
            alloc.deallocate(e)
            e = alloc.allocate()
        assert alloc.is_alive(e)

    def test_stale_dead_after_512_cycles(self):
        """Stale handle dead after 512 cycles."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        stale = e
        for _ in range(512):
            alloc.deallocate(e)
            e = alloc.allocate()
        assert not alloc.is_alive(stale)

    def test_alive_after_1000_cycles(self):
        """Entity survives 1000 recycle cycles."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        for _ in range(1000):
            alloc.deallocate(e)
            e = alloc.allocate()
        assert alloc.is_alive(e)

    def test_stale_dead_after_1000_cycles(self):
        """Stale handle dead after 1000 cycles."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        stale = e
        for _ in range(1000):
            alloc.deallocate(e)
            e = alloc.allocate()
        assert not alloc.is_alive(stale)

    # ------------------------------------------------------------------
    # 3 -- Multiple independent entities, different indices
    # ------------------------------------------------------------------

    def test_multiple_entities_independent_cycles(self):
        """Multiple entities at different indices cycling independently."""
        alloc = EntityAllocator()
        entities = [alloc.allocate() for _ in range(10)]
        stales = list(entities)

        for i, e in enumerate(entities):
            cycles = 200 + i * 15  # 200, 215, 230, ..., 335
            for _ in range(cycles):
                alloc.deallocate(e)
                e = alloc.allocate()
            entities[i] = e

        for i, e in enumerate(entities):
            assert alloc.is_alive(e), (
                f"entity[{i}] should be alive after {200 + i * 15} cycles"
            )

        for i, s in enumerate(stales):
            assert not alloc.is_alive(s), (
                f"stale[{i}] should be dead after {200 + i * 15} cycles"
            )

    # ------------------------------------------------------------------
    # 4 -- Stale handles saved mid-cycle (not just at start)
    # ------------------------------------------------------------------

    def test_stale_handle_from_mid_cycle_dead(self):
        """A handle saved partway through cycling is dead after more cycles."""
        alloc = EntityAllocator()
        e = alloc.allocate()

        for _ in range(100):
            alloc.deallocate(e)
            e = alloc.allocate()
        mid_stale = e  # saved at gen=100

        for _ in range(200):
            alloc.deallocate(e)
            e = alloc.allocate()

        assert alloc.is_alive(e), "current entity should be alive"
        assert not alloc.is_alive(mid_stale), "mid-cycle stale should be dead"

    def test_stale_from_earlier_generation_after_wrap(self):
        """Stale handle from an earlier generation is dead after wrap."""
        alloc = EntityAllocator()
        e = alloc.allocate()

        for _ in range(250):
            alloc.deallocate(e)
            e = alloc.allocate()
        late_stale = e  # saved at gen=250

        for _ in range(20):
            alloc.deallocate(e)
            e = alloc.allocate()

        assert alloc.is_alive(e), "current entity alive"
        assert not alloc.is_alive(late_stale), "stale from gen=250 should be dead"

    # ------------------------------------------------------------------
    # 5 -- Multiple stale handles from same index, different generations
    # ------------------------------------------------------------------

    def test_multiple_stales_same_index(self):
        """Multiple stale handles from the same index are all dead after 256+."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        stales = []

        # Save stale handle every 50 cycles (save before next deallocate)
        for i in range(300):
            if i > 0 and i % 50 == 0:
                stales.append(e)
            alloc.deallocate(e)
            e = alloc.allocate()

        for idx, s in enumerate(stales):
            assert not alloc.is_alive(s), (
                f"stale saved at cycle {idx * 50} should be dead"
            )

        assert alloc.is_alive(e), "current entity should be alive after 300 cycles"

    # ------------------------------------------------------------------
    # 6 -- Property stability through cycles
    # ------------------------------------------------------------------

    def test_entity_properties_stable_after_256_cycles(self):
        """Entity properties (index, is_valid) remain correct after cycling."""
        alloc = EntityAllocator()
        indices = set()
        for _ in range(260):
            e = alloc.allocate()
            indices.add(e.index)
            assert isinstance(e.index, int)
            assert e.is_valid(), "every allocated entity should be valid"
            alloc.deallocate(e)

        assert len(indices) < 260, "indices should be recycled, not all unique"
