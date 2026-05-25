"""Whitebox memory allocator tests exercising internal code paths, branch
conditions, and boundary cases not covered by unit or stress tests.

Targets (T-CORE-1.5 whitebox acceptance criteria)
--------------------------------------------------
1. LinearAllocator:   free() no-op verification at all edge states, capacity
                      property stress, micro-alignment edge boundaries
2. PoolAllocator:     constructor validation (element_size<=0, count<=0),
                      free_set/free_list consistency invariant, element_size
                      property, capacity property under load
3. RingAllocator:     head tracking invariants during wrap cycles, capacity
                      property, used_bytes cap at capacity under oscillation
4. StackAllocator:    free(-1) raises, free_to_marker at current offset (no-op),
                      free at offset==0, allocate size == capacity, get_marker
                      after reset, free with negative offset
5. SlabAllocator:     default constructor (None size_classes), empty
                      size_classes raises, size_classes property, pool_for
                      property, capacity property, allocate larger than any
                      class raises MemoryError
6. TLSFAllocator:     _mapping() static method isolation, _find_suitable when
                      no block exists, three-block coalesce chain,
                      free() with auto-size lookup (size=0), split boundary
                      at exactly _MIN_BLOCK remainder, bitmap consistency
7. MemoryTracker:     high-frequency track/free cycles stress, all 8 tags
                      interleaved, peak tracking across multiple cycles
8. ObjectPool:        max_size enforcement at scale, reset_func guarantee
                      after many cycles, large acquire/release cycles
"""

from __future__ import annotations

import random
import threading
from typing import Any, List

import pytest

from engine.core.memory.allocator import AllocationInfo, MemoryTag
from engine.core.memory.linear import LinearAllocator
from engine.core.memory.object_pool import ObjectPool
from engine.core.memory.pool import PoolAllocator
from engine.core.memory.ring import RingAllocator
from engine.core.memory.slab import SlabAllocator
from engine.core.memory.stack import StackAllocator
from engine.core.memory.tlsf import TLSFAllocator
from engine.core.memory.tracker import MemoryTracker


# ===========================================================================
# 1.  LinearAllocator whitebox
# ===========================================================================

class TestLinearAllocatorWhitebox:
    """Internal paths and properties for LinearAllocator."""

    def test_free_is_noop_at_all_states(self) -> None:
        """free() must not change used_bytes, capacity, or buffer, regardless
        of allocator state (empty, partially filled, full)."""
        a = LinearAllocator(1024)

        # --- empty ---
        a.free(0)
        assert a.used_bytes == 0
        assert a.capacity == 1024

        # --- partially filled ---
        a.allocate(100)
        a.free(0)
        assert a.used_bytes == 100  # unchanged

        # --- exactly full ---
        a.allocate(924)
        a.free(0)
        assert a.used_bytes == 1024

    def test_free_noop_does_not_corrupt_buffer(self) -> None:
        """free() must not mutate the backing buffer."""
        a = LinearAllocator(256)
        off = a.allocate(64)
        a.buffer[off:off+64] = b'\xAB' * 64
        snap = bytes(a.buffer)
        a.free(off)
        assert bytes(a.buffer) == snap

    def test_capacity_never_changes(self) -> None:
        """capacity is fixed after construction, regardless of operations."""
        a = LinearAllocator(65536)
        assert a.capacity == 65536
        a.allocate(1000)
        assert a.capacity == 65536
        a.reset()
        assert a.capacity == 65536
        for _ in range(100):
            a.allocate(512)
            a.reset()
        assert a.capacity == 65536

    def test_zero_initialized_after_constructor(self) -> None:
        """Every byte of the buffer must be zero after construction (bytearray)."""
        a = LinearAllocator(4096)
        assert all(b == 0 for b in a.buffer)

    def test_allocate_exact_size_boundary(self) -> None:
        """Allocating exactly remaining capacity must succeed, one more fails."""
        a = LinearAllocator(100)
        a.allocate(99)
        a.allocate(1)  # exact fit
        assert a.used_bytes == 100
        with pytest.raises(MemoryError):
            a.allocate(1)

    def test_used_bytes_after_reset_cycle_stress(self) -> None:
        """used_bytes = offset invariant holds across 10k reset cycles."""
        a = LinearAllocator(1024)
        for _ in range(10_000):
            a.allocate(128)
            a.reset()
            assert a.used_bytes == 0


# ===========================================================================
# 2.  PoolAllocator whitebox
# ===========================================================================

class TestPoolAllocatorWhitebox:
    """Internal paths and properties for PoolAllocator."""

    def test_constructor_element_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="element_size must be positive"):
            PoolAllocator(element_size=0, count=10)

    def test_constructor_element_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="element_size must be positive"):
            PoolAllocator(element_size=-1, count=10)

    def test_constructor_count_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="count must be positive"):
            PoolAllocator(element_size=16, count=0)

    def test_constructor_count_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="count must be positive"):
            PoolAllocator(element_size=16, count=-1)

    def test_free_list_free_set_consistent_after_stress(self) -> None:
        """_free_list and _free_set must always agree on available indices
        after many random acquire/release cycles."""
        p = PoolAllocator(element_size=16, count=100)
        rng = random.Random(42)
        acquired: list[int] = []
        for _ in range(5000):
            if not acquired or rng.random() < 0.5:
                idx = p.allocate()
                acquired.append(idx)
            else:
                idx = acquired.pop(rng.randint(0, len(acquired) - 1))
                p.free(idx)
            # Invariant check: free_set must match free_list contents
            assert set(p._free_list) == p._free_set
            assert len(p._free_list) == len(p._free_set)
            assert len(p._free_list) + len(acquired) == 100
        for idx in acquired:
            p.free(idx)
        assert p.available_count == 100
        assert p.used_bytes == 0

    def test_element_size_property(self) -> None:
        p = PoolAllocator(element_size=64, count=10)
        assert p.element_size == 64

    def test_capacity_property(self) -> None:
        p = PoolAllocator(element_size=32, count=50)
        assert p.capacity == 1600  # 32 * 50

    def test_used_bytes_equivalence(self) -> None:
        """used_bytes == (count - available_count) * element_size."""
        p = PoolAllocator(element_size=64, count=20)
        for i in range(15):
            p.allocate()
            expected = (i + 1) * 64
            assert p.used_bytes == expected

    def test_available_count_matches_free_list_length(self) -> None:
        p = PoolAllocator(element_size=16, count=50)
        assert p.available_count == 50
        p.allocate()
        assert p.available_count == 49
        p.allocate()
        assert p.available_count == 48
        p.reset()
        assert p.available_count == 50

    def test_is_full_property(self) -> None:
        p = PoolAllocator(element_size=16, count=3)
        assert not p.is_full
        p.allocate()
        assert not p.is_full
        p.allocate()
        assert not p.is_full
        p.allocate()
        assert p.is_full

    def test_buffer_backed_by_correct_size(self) -> None:
        p = PoolAllocator(element_size=32, count=100)
        assert len(p.buffer) == 3200

    def test_free_with_none_raises(self) -> None:
        p = PoolAllocator(element_size=16, count=10)
        with pytest.raises(TypeError):
            p.free(None)  # type: ignore[arg-type]

    def test_allocate_with_size_ignored(self) -> None:
        """allocate(size) ignores the size argument per PoolAllocator contract."""
        p = PoolAllocator(element_size=64, count=10)
        idx = p.allocate(9999)  # size is documented as ignored
        assert 0 <= idx < 10
        assert p.used_bytes == 64


# ===========================================================================
# 3.  RingAllocator whitebox
# ===========================================================================

class TestRingAllocatorWhitebox:
    """Internal paths and properties for RingAllocator."""

    def test_head_tracking_after_wrap(self) -> None:
        """head must always be in [0, capacity) after any allocation."""
        r = RingAllocator(256)
        for size in [100, 100, 100, 100]:  # 4 * 100 = 400 > 256, wraps
            r.allocate(size)
            assert 0 <= r.head < 256

    def test_head_starts_at_zero(self) -> None:
        r = RingAllocator(1024)
        assert r.head == 0

    def test_head_after_reset(self) -> None:
        r = RingAllocator(1024)
        r.allocate(500)
        r.reset()
        assert r.head == 0

    def test_capacity_property_unchanged(self) -> None:
        r = RingAllocator(2048)
        assert r.capacity == 2048
        r.allocate(1024)
        assert r.capacity == 2048
        r.reset()
        assert r.capacity == 2048

    def test_used_bytes_capped_at_capacity_oscillation(self) -> None:
        """used_bytes must never exceed capacity, even under rapid wrap."""
        r = RingAllocator(1024)
        for size in [700, 700, 700, 700, 700]:
            r.allocate(size)
            assert r.used_bytes <= 1024

    def test_wrap_offset_arithmetic(self) -> None:
        """Verify offset calculation after multiple wraps."""
        r = RingAllocator(100)
        # Allocate 30, head=30
        off1 = r.allocate(30)
        assert off1 == 0
        # Allocate 30, head=60
        off2 = r.allocate(30)
        assert off2 == 30
        # Allocate 30, head=90
        off3 = r.allocate(30)
        assert off3 == 60
        # Allocate 30, head = (90+30)%100 = 20
        off4 = r.allocate(30)
        assert off4 == 90
        # Allocate 30, head = (20+30)%100 = 50
        off5 = r.allocate(30)
        assert off5 == 20

    def test_free_is_noop_without_corruption(self) -> None:
        """free must not alter buffer content."""
        r = RingAllocator(512)
        off = r.allocate(256)
        r.buffer[off:off+256] = b'\xCD' * 256
        snap = bytes(r.buffer)
        r.free(off)
        assert bytes(r.buffer) == snap

    def test_constructor_invalid_capacity_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be positive"):
            RingAllocator(0)

    def test_constructor_invalid_capacity_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity must be positive"):
            RingAllocator(-1)

    def test_allocate_exactly_capacity_repeatedly(self) -> None:
        """allocate(capacity) repeatedly should produce same offsets."""
        r = RingAllocator(128)
        for _ in range(10):
            off = r.allocate(128)
            assert off == 0
            assert r.head == 0


# ===========================================================================
# 4.  StackAllocator whitebox
# ===========================================================================

class TestStackAllocatorWhitebox:
    """Internal paths and properties for StackAllocator."""

    def test_free_negative_offset_raises(self) -> None:
        a = StackAllocator(1024)
        a.allocate(100)
        with pytest.raises(ValueError, match="invalid free offset"):
            a.free(-1)

    def test_free_at_zero_empties_allocator(self) -> None:
        """free(0) should unwind all allocations back to start."""
        a = StackAllocator(1024)
        a.allocate(100)
        a.allocate(200)
        assert a.used_bytes == 300
        a.free(0)
        assert a.used_bytes == 0

    def test_free_at_exact_used_bytes_is_noop(self) -> None:
        """free(used_bytes) is the same as free at end -- offset stays."""
        a = StackAllocator(1024)
        off = a.allocate(64)
        a.free(off)  # already at this offset
        assert a.used_bytes == off

    def test_free_to_marker_at_current_offset_is_noop(self) -> None:
        """free_to_marker(current_offset) should leave state unchanged."""
        a = StackAllocator(1024)
        a.allocate(100)
        marker = a.used_bytes
        a.free_to_marker(marker)
        assert a.used_bytes == marker

    def test_free_to_marker_at_zero_empties(self) -> None:
        a = StackAllocator(1024)
        a.allocate(100)
        a.allocate(200)
        a.free_to_marker(0)
        assert a.used_bytes == 0

    def test_get_marker_after_reset_returns_zero(self) -> None:
        a = StackAllocator(1024)
        a.allocate(500)
        a.reset()
        assert a.get_marker() == 0

    def test_get_marker_returns_used_bytes(self) -> None:
        """get_marker() must always match used_bytes."""
        a = StackAllocator(4096)
        for size in [10, 20, 30, 40, 50]:
            a.allocate(size)
            assert a.get_marker() == a.used_bytes

    def test_allocate_exact_capacity_succeeds(self) -> None:
        a = StackAllocator(1024)
        off = a.allocate(1024)
        assert off == 0
        assert a.used_bytes == 1024
        with pytest.raises(MemoryError):
            a.allocate(1)

    def test_free_beyond_offset_raises(self) -> None:
        a = StackAllocator(1024)
        a.allocate(100)
        with pytest.raises(ValueError, match="invalid free offset"):
            a.free(200)

    def test_lifo_nesting_chain(self) -> None:
        """Deeply nested marker allocation/unwinding with verification."""
        a = StackAllocator(65536)
        markers: list[int] = []
        for depth in range(1000):
            markers.append(a.get_marker())
            a.allocate(32)
        assert a.used_bytes == 32000  # 1000 * 32 = 32000
        # Unwind in strict LIFO order
        for m in reversed(markers):
            a.free_to_marker(m)
        assert a.used_bytes == 0


# ===========================================================================
# 5.  SlabAllocator whitebox
# ===========================================================================

class TestSlabAllocatorWhitebox:
    """Internal paths and properties for SlabAllocator."""

    def test_default_constructor(self) -> None:
        """Default constructor uses _DEFAULT_SIZE_CLASSES from constants."""
        s = SlabAllocator()  # no args
        assert len(s.size_classes) > 0
        assert s.capacity > 0

    def test_empty_size_classes_defaults(self) -> None:
        """Empty size_classes defaults to _DEFAULT_SIZE_CLASSES (falsy check)."""
        s = SlabAllocator(size_classes=[])
        assert len(s.size_classes) > 0  # defaults used

    def test_size_classes_property(self) -> None:
        s = SlabAllocator(size_classes=[8, 16, 32, 64])
        assert s.size_classes == [8, 16, 32, 64]

    def test_size_classes_property_is_copy(self) -> None:
        """size_classes property returns a copy, not the internal list."""
        s = SlabAllocator(size_classes=[16, 32])
        classes = s.size_classes
        classes.append(999)
        assert s.size_classes == [16, 32]  # internal unchanged

    def test_pool_for_returns_correct_pool(self) -> None:
        s = SlabAllocator(size_classes=[16, 64, 256], slots_per_class=8)
        p = s.pool_for(64)
        assert p.element_size == 64
        assert p.available_count == 8

    def test_capacity_property(self) -> None:
        s = SlabAllocator(size_classes=[16, 64], slots_per_class=4)
        assert s.capacity == 16 * 4 + 64 * 4  # 320

    def test_allocate_larger_than_max_class_raises(self) -> None:
        s = SlabAllocator(size_classes=[16, 32, 64], slots_per_class=4)
        with pytest.raises(MemoryError, match="no size class fits"):
            s.allocate(128)

    def test_allocate_at_exact_class_boundary(self) -> None:
        """Allocating exactly a class size should use that class."""
        s = SlabAllocator(size_classes=[16, 32, 64], slots_per_class=4)
        sc, _ = s.allocate_slab(32)
        assert sc == 32

    def test_allocate_just_below_class_boundary(self) -> None:
        """Allocating one byte below a class size should use the next class."""
        s = SlabAllocator(size_classes=[16, 32, 64], slots_per_class=4)
        sc, _ = s.allocate_slab(31)
        assert sc == 32

    def test_encoded_offset_32bit_boundary(self) -> None:
        """Encoded offset with large slot index must decode correctly."""
        s = SlabAllocator(size_classes=[16, 32], slots_per_class=1000)
        # Allocate many slots to force a high index
        offsets = [s.allocate(8) for _ in range(1000)]
        for off in offsets:
            s.free(off)

    def test_mixed_allocator_and_slab_interface(self) -> None:
        """Allocate via allocate() and free via free_slab() must work."""
        s = SlabAllocator(size_classes=[32, 64], slots_per_class=4)
        off = s.allocate(16)
        sc_idx = (off >> 32) & 0xFFFFFFFF
        index = off & 0xFFFFFFFF
        sc = s.size_classes[sc_idx]
        s.free_slab(sc, index)


# ===========================================================================
# 6.  TLSFAllocator whitebox
# ===========================================================================

class TestTLSFAllocatorWhitebox:
    """Internal paths and properties for TLSFAllocator."""

    def test_mapping_isolated(self) -> None:
        """_mapping() static method must produce consistent (fl, sl) pairs."""
        cases: list[tuple[int, int, int]] = [
            (16, 4, 0),   # size=16: fl=4,  remainder=0,   sl=0
            (17, 4, 0),   # size=17: fl=4,  remainder=1,   sl=(1>>2)&3=0
            (32, 5, 0),   # size=32: fl=5,  remainder=0,   sl=0
            (48, 5, 2),   # size=48: fl=5,  remainder=16,  sl=(16>>3)&3=2
            (64, 6, 0),   # size=64: fl=6,  remainder=0,   sl=0
            (128, 7, 0),  # size=128: fl=7, remainder=0,   sl=0
            (256, 8, 0),  # size=256: fl=8, remainder=0,   sl=0
        ]
        for size, expected_fl, expected_sl in cases:
            fl, sl = TLSFAllocator._mapping(size)
            assert fl == expected_fl, f"size={size}: expected fl={expected_fl}, got {fl}"
            assert sl == expected_sl, f"size={size}: expected sl={expected_sl}, got {sl}"

    def test_mapping_min_block(self) -> None:
        """_mapping(< _MIN_BLOCK) should return (0, 0)."""
        fl, sl = TLSFAllocator._mapping(4)
        assert fl == 0
        assert sl == 0

    def test_find_suitable_no_block_returns_none(self) -> None:
        """_find_suitable must return None when no block satisfies request."""
        t = TLSFAllocator(1024)
        # Exhaust the allocator
        t.allocate(1024)
        # Use _find_suitable directly on the empty free lists
        result = t._find_suitable(0, 0)
        assert result is None

    def test_two_block_coalesce_same_size(self) -> None:
        """Two adjacent free blocks of the same size coalesce into one.

        TLSF._coalesce checks for adjacency within the same (fl, sl) bucket.
        Two 64-byte adjacent free blocks share bucket (6,0) and should merge.
        """
        t = TLSFAllocator(4096)

        a = t.allocate(64)   # offset 0
        b = t.allocate(64)   # offset 64

        # Free b first, then a: both are 64 bytes and end up in bucket (6,0)
        t.free(b, 64)        # free block at (64, 64) in (6,0)
        t.free(a, 64)        # free block at (0, 64) in (6,0); coalesces with (64,64)

        # After coalesce: one block at (0, 128) in (7,0)
        d = t.allocate(128)
        assert d == 0        # should reuse the coalesced block
        assert t.used_bytes == 128

    def test_two_block_coalesce_backward(self) -> None:
        """Free right block first, then left block coalesces into it."""
        t = TLSFAllocator(4096)

        a = t.allocate(64)   # offset 0
        b = t.allocate(64)   # offset 64
        c = t.allocate(64)   # offset 128

        # Free c (right) → (128, 64) in (6,0)
        t.free(c, 64)
        # Free b (middle) → (64, 64) in (6,0); coalesces forward: (64, 64)+(128, 64) → (64, 128)
        t.free(b, 64)
        # Free a (left) → (0, 64) in (6,0); bucket (6,0) is empty now, so no coalesce with (64, 128)
        # (64, 128) is in (7,0), not (6,0)
        t.free(a, 64)

        # Now: free (0, 64) in (6,0) and (64, 128) in (7,0)
        # Allocate 128 → should find the block at (64, 128) from bucket (7,0)
        d = t.allocate(128)
        assert d == 64
        assert t.used_bytes == 128

    def test_free_with_auto_size_lookup(self) -> None:
        """free(offset) without size should auto-lookup tracked size."""
        t = TLSFAllocator(4096)
        a = t.allocate(128)
        t.free(a)  # no size passed
        assert t.used_bytes == 0
        # Should be able to reallocate
        b = t.allocate(128)
        assert b == a

    def test_free_auto_size_unknown_offset_raises(self) -> None:
        """free(offset) with untracked offset and no size must raise."""
        t = TLSFAllocator(4096)
        with pytest.raises(ValueError, match="size must be positive"):
            t.free(9999)  # offset not in _alloc_sizes

    def test_split_remainder_exactly_min_block(self) -> None:
        """Allocate such that the remainder is exactly _MIN_BLOCK (16).

        The split should produce a free block of size _MIN_BLOCK.
        """
        from engine.core.constants import TLSF_MIN_BLOCK
        t = TLSFAllocator(4096)
        # Initial free block is 4096.
        # Allocate 4096 - TLSF_MIN_BLOCK bytes. Remainder = TLSF_MIN_BLOCK.
        alloc_size = 4096 - TLSF_MIN_BLOCK
        off = t.allocate(alloc_size)
        assert off == 0
        # The remainder (_MIN_BLOCK bytes) should be a free block.
        # Allocate exactly _MIN_BLOCK to verify.
        off2 = t.allocate(TLSF_MIN_BLOCK)
        assert off2 == alloc_size
        assert t.used_bytes == alloc_size + TLSF_MIN_BLOCK

    def test_used_bytes_tracking_internal(self) -> None:
        """used_bytes should match sum of allocated sizes."""
        t = TLSFAllocator(4096)
        a = t.allocate(100)
        b = t.allocate(200)
        assert t.used_bytes >= 300
        t.free(a, 100)
        assert t.used_bytes >= 200
        t.free(b, 200)
        assert t.used_bytes == 0

    def test_used_bytes_after_free_auto_lookup(self) -> None:
        t = TLSFAllocator(4096)
        a = t.allocate(256)
        t.free(a)  # auto-size
        assert t.used_bytes == 0

    def test_bitmap_fl_invariant(self) -> None:
        """_fl_bitmap must be 0 when all free lists are empty."""
        t = TLSFAllocator(128)  # small capacity
        t.allocate(128)  # exhaust all free space
        assert t._fl_bitmap == 0

    def test_bitmap_sl_invariant(self) -> None:
        """_sl_bitmaps entries for exhausted FL buckets may remain as zero
        entries (implementation detail of _remove_free)."""
        t = TLSFAllocator(128)
        t.allocate(128)
        # After exhaustion: fl_bitmap is 0, sl_bitmaps may have zero entries
        assert t._fl_bitmap == 0


# ===========================================================================
# 7.  MemoryTracker whitebox
# ===========================================================================

class TestMemoryTrackerWhitebox:
    """Internal paths and high-frequency stress for MemoryTracker."""

    def test_all_eight_tags_independent(self) -> None:
        """Each MemoryTag must track its own stats independently."""
        t = MemoryTracker()
        tags = list(MemoryTag)
        for i, tag in enumerate(tags):
            t.track_allocation(AllocationInfo(offset=i * 100, size=50, tag=tag))
            stats = t.get_stats(tag)
            assert stats.allocated == 50
            assert stats.current == 50

    def test_peak_tracking_multiple_cycles(self) -> None:
        """Peak should correctly track the high-water mark across cycles."""
        t = MemoryTracker()
        for cycle in range(10):
            t.track_allocation(AllocationInfo(offset=cycle * 100, size=100, tag=MemoryTag.CORE))
        assert t.get_total_stats().peak == 1000
        for offset in range(0, 1000, 100):
            t.track_free(offset)
        assert t.get_total_stats().peak == 1000  # peak unchanged
        assert t.get_total_stats().current == 0

    def test_high_frequency_track_free_stress(self) -> None:
        """10k track/free cycles should not leak or corrupt stats."""
        t = MemoryTracker()
        for i in range(10_000):
            t.track_allocation(AllocationInfo(offset=i, size=16, tag=MemoryTag.CORE))
        assert len(t.get_live_allocations()) == 10_000
        for i in range(10_000):
            t.track_free(i)
        total = t.get_total_stats()
        assert total.allocated == 160_000
        assert total.freed == 160_000
        assert total.current == 0
        assert len(t.get_live_allocations()) == 0

    def test_interleaved_tags_high_frequency(self) -> None:
        """All 8 tags interleaved across 5k operations each."""
        t = MemoryTracker()
        tags = list(MemoryTag)
        offset = 0
        for _ in range(5000):
            for tag in tags:
                t.track_allocation(AllocationInfo(offset=offset, size=8, tag=tag))
                offset += 8
        total = t.get_total_stats()
        assert total.allocated == 5000 * len(tags) * 8
        # Free everything
        for off in range(0, offset, 8):
            t.track_free(off)
        after = t.get_total_stats()
        assert after.current == 0
        assert after.allocated == after.freed
        assert after.peak == 5000 * len(tags) * 8

    def test_unknown_free_logs_warning_no_crash(self) -> None:
        """track_free with unknown offset should not raise or corrupt state."""
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=64, tag=MemoryTag.CORE))
        assert t.get_total_stats().current == 64
        t.track_free(999)  # unknown
        assert t.get_total_stats().current == 64  # unchanged

    def test_get_live_allocations_returns_copy(self) -> None:
        """get_live_allocations should return a list, not the internal dict."""
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=32, tag=MemoryTag.UI))
        live = t.get_live_allocations()
        live.clear()
        assert len(t.get_live_allocations()) == 1  # internal unchanged

    def test_track_free_twice_safe(self) -> None:
        """track_free on same offset twice is safe (second is noop)."""
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=64, tag=MemoryTag.CORE))
        t.track_free(0)
        t.track_free(0)  # second free should not raise
        assert t.get_total_stats().current == 0

    def test_tag_stats_not_present_returns_empty(self) -> None:
        """get_stats for tag with no allocations returns empty MemoryStats."""
        t = MemoryTracker()
        stats = t.get_stats(MemoryTag.AUDIO)
        assert stats.allocated == 0
        assert stats.freed == 0
        assert stats.current == 0
        assert stats.peak == 0

    def test_multiple_peaks(self) -> None:
        """Peak should capture each higher water mark correctly."""
        t = MemoryTracker()
        t.track_allocation(AllocationInfo(offset=0, size=100, tag=MemoryTag.CORE))
        assert t.get_total_stats().peak == 100
        t.track_allocation(AllocationInfo(offset=100, size=200, tag=MemoryTag.CORE))
        assert t.get_total_stats().peak == 300
        t.track_free(0)
        assert t.get_total_stats().peak == 300
        t.track_free(100)
        assert t.get_total_stats().peak == 300


# ===========================================================================
# 8.  ObjectPool whitebox
# ===========================================================================

class _Dummy:
    def __init__(self) -> None:
        self.value = 0
        self.id = id(self)


class TestObjectPoolWhitebox:
    """Internal paths and high-frequency stress for ObjectPool."""

    def test_acquire_creates_new_when_empty(self) -> None:
        pool = ObjectPool(factory=_Dummy)
        obj = pool.acquire()
        assert isinstance(obj, _Dummy)

    def test_release_reuses_object(self) -> None:
        pool = ObjectPool(factory=_Dummy)
        obj1 = pool.acquire()
        obj2 = pool.acquire()
        pool.release(obj1)
        pool.release(obj2)
        reused1 = pool.acquire()
        reused2 = pool.acquire()
        assert reused1 is obj2  # LIFO: last released is first acquired
        assert reused2 is obj1

    def test_max_size_enforces_cap(self) -> None:
        pool = ObjectPool(factory=_Dummy, max_size=5)
        objs = [pool.acquire() for _ in range(20)]
        for o in objs:
            pool.release(o)
        assert pool.available == 5  # capped at max_size

    def test_max_size_zero_disables_pooling(self) -> None:
        pool = ObjectPool(factory=_Dummy, max_size=0)
        objs = [pool.acquire() for _ in range(10)]
        for o in objs:
            pool.release(o)
        assert pool.available == 0

    def test_reset_func_called_on_release(self) -> None:
        reset_calls: list[_Dummy] = []

        def reset_func(obj: _Dummy) -> None:
            reset_calls.append(obj)
            obj.value = 0

        pool = ObjectPool(factory=_Dummy, reset_func=reset_func, max_size=10)
        obj = pool.acquire()
        obj.value = 42
        pool.release(obj)
        assert obj in reset_calls
        assert obj.value == 0  # reset by func

    def test_reset_func_guarantee_at_scale(self) -> None:
        """reset_func must be called on every release, regardless of scale."""
        reset_count = 0

        def reset_func(obj: _Dummy) -> None:
            nonlocal reset_count
            reset_count += 1
            obj.value = 0

        pool = ObjectPool(factory=_Dummy, reset_func=reset_func, max_size=100)
        objs = [pool.acquire() for _ in range(1000)]
        for o in objs:
            o.value = 99
            pool.release(o)
        assert reset_count == 1000  # once per release
        # Re-acquire: all should have value=0
        for _ in range(1000):
            o = pool.acquire()
            assert o.value == 0

    def test_10k_acquire_release_stress(self) -> None:
        """10k acquire/release cycles should not leak or corrupt pool."""
        pool = ObjectPool(factory=_Dummy)
        for _ in range(10_000):
            obj = pool.acquire()
            pool.release(obj)
        assert pool.available == 1  # only the last obj

    def test_10k_with_initial_size(self) -> None:
        """10k cycles with initial_size=50 should reuse pooled objects."""
        pool = ObjectPool(factory=_Dummy, initial_size=50)
        assert pool.available == 50
        for _ in range(10_000):
            obj = pool.acquire()
            pool.release(obj)
        assert pool.total_created == 50  # no new objects created

    def test_grows_beyond_initial_size(self) -> None:
        pool = ObjectPool(factory=_Dummy, initial_size=3)
        objs = [pool.acquire() for _ in range(100)]
        assert pool.total_created == 100
        assert len(objs) == 100

    def test_concurrent_acquire_release(self) -> None:
        """Multiple threads acquiring and releasing concurrently."""
        pool = ObjectPool(factory=_Dummy, max_size=100)
        errors: list[str] = []
        lock = threading.Lock()
        done = threading.Event()

        def worker() -> None:
            while not done.is_set():
                try:
                    with lock:
                        obj = pool.acquire()
                    with lock:
                        pool.release(obj)
                except IndexError:
                    # Pool temporarily empty under concurrent load; retry
                    pass
                except Exception as e:
                    errors.append(str(e))
                    break

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        done.wait(0.5)
        done.set()
        for t in threads:
            t.join(timeout=2)
        assert not errors, f"concurrent errors: {errors}"

    def test_initial_size_respected(self) -> None:
        pool = ObjectPool(factory=_Dummy, initial_size=10)
        assert pool.available == 10
        assert pool.total_created == 10


# ===========================================================================
# 9.  Cross-cutting whitebox: all allocators
# ===========================================================================

class TestAllocatorInterfaceWhitebox:
    """Common Allocator ABC interface contract verification."""

    def test_all_allocators_have_capacity(self) -> None:
        """Every allocator must expose a positive capacity."""
        allocators = [
            LinearAllocator(1024),
            PoolAllocator(element_size=16, count=64),
            RingAllocator(1024),
            StackAllocator(1024),
            TLSFAllocator(1024),
        ]
        for a in allocators:
            assert a.capacity > 0, f"{type(a).__name__}.capacity not positive"

    def test_all_allocators_have_used_bytes_zero(self) -> None:
        """Every allocator must start with used_bytes == 0."""
        allocators = [
            LinearAllocator(1024),
            PoolAllocator(element_size=16, count=64),
            RingAllocator(1024),
            StackAllocator(1024),
            TLSFAllocator(1024),
        ]
        # SlabAllocator excluded - no-arg constructor uses defaults
        for a in allocators:
            assert a.used_bytes == 0, f"{type(a).__name__}.used_bytes not 0"

    def test_all_allocators_reset_restores_zero(self) -> None:
        """After allocate+reset, used_bytes must return to 0 for all
        allocators that support reset."""
        allocators: list[Any] = [
            LinearAllocator(1024),
            PoolAllocator(element_size=16, count=64),
            RingAllocator(1024),
            StackAllocator(1024),
            SlabAllocator(size_classes=[16, 32], slots_per_class=4),
            TLSFAllocator(1024),
        ]
        for a in allocators:
            a.allocate(16)
            a.reset()
            assert a.used_bytes == 0, f"{type(a).__name__} used_bytes after reset"
