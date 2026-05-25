"""Additional whitebox entity tests exercising internal code paths not covered
by test_ecs_whitebox.py or test_entity_stress.py.

Targets (T-CORE-1.5 whitebox acceptance criteria)
--------------------------------------------------
1. Entity: null sentinel, is_valid edge cases, __repr__, from_packed edge
   values, __hash__ stability, equality with non-Entity types
2. EntityAllocator: double-deallocate is safe, deallocate with no prior
   allocate, free list emptiness after full drain, next_index boundary
3. Generation: concurrent generation wrap safety (Python GIL level),
   generation tracking across many entity slots simultaneously
"""

from __future__ import annotations

import threading

import pytest

from engine.core.ecs.entity import (
    ENTITY_GENERATION_BITS,
    ENTITY_INDEX_BITS,
    GENERATION_MASK,
    INDEX_MASK,
    Entity,
    EntityAllocator,
)


# ===========================================================================
# 1.  Entity internals
# ===========================================================================

class TestEntityExtraWhitebox:
    """Entity packing edge cases not covered by prior whitebox tests."""

    def test_null_packed_value(self) -> None:
        """Entity.null() uses _NULL_INDEX (= INDEX_MASK) and generation 0."""
        e = Entity.null()
        assert not e.is_valid()
        # _NULL_INDEX = INDEX_MASK; packed = (0 << INDEX_BITS) | INDEX_MASK
        assert e.index == INDEX_MASK
        assert e.generation == 0

    def test_null_is_valid_false(self) -> None:
        assert not Entity.null().is_valid()

    def test_valid_entity_is_valid_true(self) -> None:
        e = Entity(0, 0)
        assert e.is_valid()

    def test_entity_with_max_index_is_valid(self) -> None:
        e = Entity(INDEX_MASK, 0)
        assert not e.is_valid()  # INDEX_MASK is the null sentinel

    def test_entity_with_index_below_max_is_valid(self) -> None:
        e = Entity(INDEX_MASK - 1, 0)
        assert e.is_valid()

    def test_repr_null(self) -> None:
        assert repr(Entity.null()) == "Entity(null)"

    def test_repr_valid(self) -> None:
        r = repr(Entity(42, 7))
        assert "index=42" in r
        assert "gen=7" in r

    def test_equality_with_non_entity_returns_not_implemented(self) -> None:
        e = Entity(0, 0)
        assert e.__eq__(42) is NotImplemented
        assert e.__eq__("string") is NotImplemented
        assert e.__eq__(None) is NotImplemented

    def test_hash_stability(self) -> None:
        """Entity hash must be stable and match _packed."""
        e = Entity(12345, 200)
        assert hash(e) == e._packed

    def test_from_packed_all_ones_index(self) -> None:
        """from_packed with index bits all 1 should be the null entity."""
        packed = INDEX_MASK  # index = all 1s, gen = 0
        e = Entity.from_packed(packed)
        assert e.index == INDEX_MASK
        assert not e.is_valid()

    def test_from_packed_all_ones_gen(self) -> None:
        """from_packed with generation bits all 1."""
        gen_mask = GENERATION_MASK  # 0xFF for 8 bits
        packed = (gen_mask << ENTITY_INDEX_BITS) | 42
        e = Entity.from_packed(packed)
        assert e.index == 42
        assert e.generation == gen_mask

    def test_from_packed_full_max(self) -> None:
        """All bits set: max index + max generation."""
        max_gen = GENERATION_MASK
        max_idx = INDEX_MASK
        packed = (max_gen << ENTITY_INDEX_BITS) | max_idx
        e = Entity.from_packed(packed)
        assert e.index == max_idx
        assert e.generation == max_gen
        assert not e.is_valid()  # max_idx is the null sentinel

    def test_index_masking_overflow(self) -> None:
        """Index wider than ENTITY_INDEX_BITS gets masked."""
        wide_index = 1 << ENTITY_INDEX_BITS | 0xABCDEF
        e = Entity(wide_index, 0)
        assert e.index == (wide_index & INDEX_MASK)

    def test_generation_masking_overflow(self) -> None:
        """Generation wider than ENTITY_GENERATION_BITS gets masked."""
        wide_gen = 1 << ENTITY_GENERATION_BITS | 0xAB
        e = Entity(0, wide_gen)
        assert e.generation == (wide_gen & GENERATION_MASK)

    def test_zero_packed_is_entity_zero_zero(self) -> None:
        e = Entity.from_packed(0)
        assert e.index == 0
        assert e.generation == 0
        assert e.is_valid()

    def test_packed_round_trip_with_all_edges(self) -> None:
        """Round-trip through from_packed for boundary values."""
        test_vals = [
            0,
            1,
            INDEX_MASK,
            (1 << ENTITY_INDEX_BITS) - 2,
            (GENERATION_MASK << ENTITY_INDEX_BITS),
            (GENERATION_MASK << ENTITY_INDEX_BITS) | INDEX_MASK,
        ]
        for packed in test_vals:
            e1 = Entity.from_packed(packed)
            e2 = Entity.from_packed(e1._packed)
            assert e1 == e2
            assert e1.index == e2.index
            assert e1.generation == e2.generation


# ===========================================================================
# 2.  EntityAllocator internals
# ===========================================================================

class TestEntityAllocatorExtraWhitebox:
    """EntityAllocator internals not covered by prior tests."""

    def test_double_deallocate_safe(self) -> None:
        """Deallocating an already-deallocated entity must not raise."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        alloc.deallocate(e)
        alloc.deallocate(e)  # second deallocate: generation bumps again
        # Entity is no longer alive
        assert not alloc.is_alive(e)
        # The generation was bumped on first deallocate; second deallocate
        # bumps it again, so the original handle is doubly stale
        e2 = alloc.allocate()
        assert e2.index == e.index
        assert e2.generation == 2  # bumped twice

    def test_deallocate_entity_never_allocated(self) -> None:
        """Deallocating an entity with index beyond _generations is safe."""
        alloc = EntityAllocator()
        e = Entity(999_999, 0)  # never allocated
        alloc.deallocate(e)  # must not raise

    def test_allocate_after_full_deallocate_cycle(self) -> None:
        """Allocate many, deallocate all, then allocate again."""
        alloc = EntityAllocator()
        batch = [alloc.allocate() for _ in range(100)]
        for e in batch:
            alloc.deallocate(e)
        # Free list is populated; new allocations use it
        recycled = [alloc.allocate() for _ in range(100)]
        assert all(e.index < 100 for e in recycled)
        # All should have generation=1 (bumped once)
        assert all(e.generation == 1 for e in recycled)

    def test_free_list_depth_after_mixed_ops(self) -> None:
        """Free list grows and shrinks correctly with mixed allocate/deallocate."""
        alloc = EntityAllocator()
        # Allocate 50 entities
        batch = [alloc.allocate() for _ in range(50)]
        assert alloc._next_index == 50
        assert len(alloc._free_list) == 0

        # Deallocate 20 specific ones
        for e in batch[:20]:
            alloc.deallocate(e)
        assert len(alloc._free_list) == 20

        # Allocate 10 new ones -- should consume 10 from free list
        new_batch = [alloc.allocate() for _ in range(10)]
        assert len(alloc._free_list) == 10
        assert alloc._next_index == 50  # unchanged

        # Deallocate all
        for e in batch[20:] + new_batch:
            alloc.deallocate(e)
        assert len(alloc._free_list) == 50

    def test_next_index_advances_past_free_list_consumption(self) -> None:
        """After consuming all free list entries, next_index advances."""
        alloc = EntityAllocator()
        # Allocate 5, free 5
        batch = [alloc.allocate() for _ in range(5)]
        for e in batch:
            alloc.deallocate(e)
        assert alloc._next_index == 5

        # Allocate 10 -- first 5 from free list, next 5 advance next_index
        batch2 = [alloc.allocate() for _ in range(10)]
        assert alloc._next_index == 10
        assert all(e.index < 10 for e in batch2)
        # First 5 should have generation 1 (recycled), last 5 generation 0
        for e in batch2:
            if e.index < 5:
                assert e.generation == 1
            else:
                assert e.generation == 0

    def test_deallocate_then_reuse_generation_bumps_chain(self) -> None:
        """Each deallocate/allocate cycle bumps generation, tracking chain."""
        alloc = EntityAllocator()
        e = alloc.allocate()
        for expected_gen in range(256):
            assert e.generation == expected_gen
            alloc.deallocate(e)
            if expected_gen < 255:
                e = alloc.allocate()

    def test_is_alive_with_non_entity(self) -> None:
        """is_alive must handle non-Entity types gracefully."""
        alloc = EntityAllocator()
        with pytest.raises(AttributeError):
            alloc.is_alive(None)  # type: ignore[arg-type]
        with pytest.raises(AttributeError):
            alloc.is_alive(42)  # type: ignore[arg-type]

    def test_allocator_empty_init_state(self) -> None:
        """Initial state: no generations, empty free list, next_index=0."""
        alloc = EntityAllocator()
        assert alloc._generations == []
        assert alloc._free_list == []
        assert alloc._next_index == 0

    def test_generation_list_grows_with_allocations(self) -> None:
        """_generations list length equals _next_index."""
        alloc = EntityAllocator()
        for i in range(1, 101):
            alloc.allocate()
            assert len(alloc._generations) == i

    def test_concurrent_allocate(self) -> None:
        """Multiple threads allocating from the same allocator sequentially."""
        alloc = EntityAllocator()
        results: list[list[Entity]] = [[] for _ in range(4)]
        errors: list[str] = []
        done = threading.Event()

        def worker(worker_id: int) -> None:
            while not done.is_set():
                try:
                    e = alloc.allocate()
                    results[worker_id].append(e)
                    # Verify alive immediately
                    assert alloc.is_alive(e)
                except RuntimeError:
                    # Max entities reached -- stop
                    break
                except Exception as exc:
                    errors.append(f"worker {worker_id}: {exc}")
                    break

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        done.wait(0.3)
        done.set()
        for t in threads:
            t.join(timeout=2)
        assert not errors, f"concurrent errors: {errors}"
        # All allocated entities should have unique indices
        all_entities = [e for batch in results for e in batch]
        indices = [e.index for e in all_entities]
        assert len(indices) == len(set(indices))  # no duplicates

    def test_concurrent_allocate_deallocate(self) -> None:
        """Multiple threads allocating and deallocating concurrently."""
        alloc = EntityAllocator()
        errors: list[str] = []
        lock = threading.Lock()
        done = threading.Event()

        def worker() -> None:
            while not done.is_set():
                try:
                    with lock:
                        e = alloc.allocate()
                    with lock:
                        alloc.deallocate(e)
                except RuntimeError:
                    pass
                except Exception as exc:
                    errors.append(str(exc))
                    break

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        done.wait(0.3)
        done.set()
        for t in threads:
            t.join(timeout=2)
        assert not errors, f"concurrent errors: {errors}"
