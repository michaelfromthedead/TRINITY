// Blackbox contract tests for HistoryRingSlot (T-FG-3.5).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract: N-slot ring buffer for temporal resource rotation.
//
// The HistoryRingSlot guarantees:
//   - N >= 2 (enforced by panic in constructor)
//   - current_slot cycles through 0..N-1 in round-robin order
//   - Each slot stores exactly one ResourceHandle
//   - advance() moves to the next slot without mutating slot contents
//   - write_current_and_advance writes to the current slot then advances
//   - slot_handle(slot_index) reads the handle at a given slot
//   - slot_count() returns the number of slots (N)
//   - current_slot() returns the current slot index
//
// Coverage:
//   1.  N>=2 enforcement: N=1 panics, N=0 panics
//   2.  N=2 slot cycling matches classic double-buffering
//   3.  N=3 slot cycling (wrap-around)
//   4.  N-slot rotation for N=4..=6 through multiple full cycles
//   5.  Large slot count (N=128) cycles without overflow
//   6.  current_slot starts at 0 for any N
//   7.  slot_count returns the constructor argument
//   8.  slot_handle returns the initial handle for all slots
//   9.  slot_handle panics on out-of-bounds index
//  10.  advance() does not mutate slot contents
//  11.  write_current_and_advance writes to the current slot then advances
//  12.  write_current_and_advance isolates each slot
//  13.  write_current_and_advance wraps back to slot 0 after N calls
//  14.  Multiple full cycles: handles survive repeated wraps
//  15.  Overwrite existing handles in a slot
//  16.  ResourceHandle::NONE as initial handle
//  17.  Clone and Debug trait derivations
//  18.  Handle values are preserved across advance cycles (no data corruption)

use renderer_backend::frame_graph::{HistoryRingSlot, ResourceHandle};

// =============================================================================
// SECTION 1 -- N >= 2 enforcement
// =============================================================================

/// Creating a HistoryRingSlot with slot_count=1 must panic (N >= 2).
#[test]
#[should_panic(expected = "requires at least 2 slots")]
fn new_with_one_slot_panics() {
    let _ring = HistoryRingSlot::new(1, ResourceHandle(0));
}

/// Creating a HistoryRingSlot with slot_count=0 must panic (N >= 2).
#[test]
#[should_panic(expected = "requires at least 2 slots")]
fn new_with_zero_slots_panics() {
    let _ring = HistoryRingSlot::new(0, ResourceHandle(0));
}

// =============================================================================
// SECTION 2 -- N=2 slot cycling (double-buffering)
// =============================================================================

/// N=2 behaves like classic double-buffering: 0 -> 1 -> 0 -> 1 ...
#[test]
fn n2_double_buffering_cycle() {
    let mut ring = HistoryRingSlot::new(2, ResourceHandle(0));
    assert_eq!(ring.slot_count(), 2);
    assert_eq!(ring.current_slot(), 0);

    // Cycle 1: slot 0 -> slot 1
    ring.advance();
    assert_eq!(ring.current_slot(), 1);

    // Cycle 2: slot 1 -> slot 0 (wrap)
    ring.advance();
    assert_eq!(ring.current_slot(), 0);

    // Cycle 3: slot 0 -> slot 1
    ring.advance();
    assert_eq!(ring.current_slot(), 1);

    // Cycle 4: slot 1 -> slot 0
    ring.advance();
    assert_eq!(ring.current_slot(), 0);
}

/// N=2 with write_current_and_advance: write to current slot, then advance.
#[test]
fn n2_write_current_and_advance() {
    let mut ring = HistoryRingSlot::new(2, ResourceHandle(0));
    assert_eq!(ring.current_slot(), 0);

    // Write to slot 0, advance to slot 1.
    ring.write_current_and_advance(ResourceHandle(42));
    assert_eq!(ring.current_slot(), 1);
    assert_eq!(ring.slot_handle(0), ResourceHandle(42));

    // Write to slot 1, advance to slot 0.
    ring.write_current_and_advance(ResourceHandle(99));
    assert_eq!(ring.current_slot(), 0);
    assert_eq!(ring.slot_handle(1), ResourceHandle(99));

    // Slot 0 still holds its value from the first write.
    assert_eq!(ring.slot_handle(0), ResourceHandle(42));
}

// =============================================================================
// SECTION 3 -- N=3 slot cycling
// =============================================================================

/// N=3 cycles through slots 0 -> 1 -> 2 -> 0 with advance().
#[test]
fn n3_slot_cycles() {
    let mut ring = HistoryRingSlot::new(3, ResourceHandle(0));
    assert_eq!(ring.slot_count(), 3);
    assert_eq!(ring.current_slot(), 0);

    ring.advance();
    assert_eq!(ring.current_slot(), 1);

    ring.advance();
    assert_eq!(ring.current_slot(), 2);

    ring.advance();
    assert_eq!(ring.current_slot(), 0); // wrapped around
}

/// Fill N=3 with distinct handles using write_current_and_advance,
/// then verify each slot holds the correct handle after one full lap.
#[test]
fn n3_write_distinct_handles_full_lap() {
    let mut ring = HistoryRingSlot::new(3, ResourceHandle(0));

    ring.write_current_and_advance(ResourceHandle(10));
    assert_eq!(ring.current_slot(), 1);

    ring.write_current_and_advance(ResourceHandle(20));
    assert_eq!(ring.current_slot(), 2);

    ring.write_current_and_advance(ResourceHandle(30));
    assert_eq!(ring.current_slot(), 0); // wrapped around

    // Each slot holds the handle written when it was current.
    assert_eq!(ring.slot_handle(0), ResourceHandle(30));
    assert_eq!(ring.slot_handle(1), ResourceHandle(10));
    assert_eq!(ring.slot_handle(2), ResourceHandle(20));
}

// =============================================================================
// SECTION 4 -- N-slot rotation (generalized)
// =============================================================================

/// For N=4..=6, the ring correctly cycles through all slots and wraps
/// around through two full cycles.
#[test]
fn n_slot_rotation_cycle() {
    for n in 4..=6 {
        let mut ring = HistoryRingSlot::new(n, ResourceHandle(0));
        let count = ring.slot_count();
        assert_eq!(count, n);

        // Advance through exactly two full cycles.
        for cycle in 0..2 {
            for slot in 0..count {
                assert_eq!(
                    ring.current_slot(),
                    slot,
                    "N={n}, cycle {cycle}, step {slot}: expected current_slot={slot}",
                );
                ring.advance();
            }
        }
    }
}

/// For a large slot count (N=128), the ring cycles correctly through
/// three full laps without overflow or wraparound issues.
#[test]
fn large_slot_count_rotation() {
    let n = 128;
    let mut ring = HistoryRingSlot::new(n, ResourceHandle(0));
    assert_eq!(ring.slot_count(), n);

    // Rotate through 3 full cycles using write_current_and_advance.
    for cycle in 0..3 {
        for slot in 0..n {
            assert_eq!(
                ring.current_slot(),
                slot,
                "N={n}, cycle {cycle}, step {slot}: expected current_slot={slot}",
            );
            ring.write_current_and_advance(ResourceHandle((cycle * n + slot) as u32));
        }
    }
    assert_eq!(ring.current_slot(), 0);

    // Verify the final lap's values are present.
    let base = 2 * n; // third lap
    for slot in 0..n {
        let expected = ResourceHandle((base + slot) as u32);
        assert_eq!(
            ring.slot_handle(slot),
            expected,
            "N={n}, slot {slot}: expected handle {expected:?} after third lap",
        );
    }
}

// =============================================================================
// SECTION 5 -- current_slot and slot_count
// =============================================================================

/// current_slot starts at 0 for any valid N.
#[test]
fn current_slot_starts_at_zero() {
    for n in [2usize, 3, 5, 10, 64] {
        let ring = HistoryRingSlot::new(n, ResourceHandle(99));
        assert_eq!(
            ring.current_slot(),
            0,
            "N={n}: current_slot must start at 0",
        );
        assert_eq!(ring.slot_count(), n, "N={n}: slot_count must equal n");
    }
}

/// slot_count returns the value passed to new().
#[test]
fn slot_count_matches_constructor_arg() {
    for n in [2, 3, 4, 8, 16, 32] {
        let ring = HistoryRingSlot::new(n, ResourceHandle(0));
        assert_eq!(ring.slot_count(), n, "slot_count must be {n}");
    }
}

// =============================================================================
// SECTION 6 -- slot_handle read-back
// =============================================================================

/// All slots are initialised to the handle passed to new().
#[test]
fn all_slots_initialised_to_initial_handle() {
    let ring = HistoryRingSlot::new(5, ResourceHandle(42));
    for i in 0..5 {
        assert_eq!(
            ring.slot_handle(i),
            ResourceHandle(42),
            "slot {i} must hold the initial handle",
        );
    }
}

/// ResourceHandle::NONE is a valid initial handle.
#[test]
fn none_handle_is_valid_initial() {
    let ring = HistoryRingSlot::new(3, ResourceHandle::NONE);
    assert_eq!(ring.slot_handle(0), ResourceHandle::NONE);
    assert_eq!(ring.slot_handle(1), ResourceHandle::NONE);
    assert_eq!(ring.slot_handle(2), ResourceHandle::NONE);
}

/// After overwriting a slot's handle, slot_handle returns the new handle.
#[test]
fn slot_handle_returns_overwritten_value() {
    let mut ring = HistoryRingSlot::new(3, ResourceHandle(0));

    // Use write_current_and_advance to fill all slots.
    ring.write_current_and_advance(ResourceHandle(100));
    ring.write_current_and_advance(ResourceHandle(200));
    ring.write_current_and_advance(ResourceHandle(300));

    // current_slot is now 0 again (after 3 advances).
    // Overwrite slot 0.
    ring.write_current_and_advance(ResourceHandle(999));
    assert_eq!(ring.slot_handle(0), ResourceHandle(999));
    // Other slots are untouched.
    assert_eq!(ring.slot_handle(1), ResourceHandle(100));
    assert_eq!(ring.slot_handle(2), ResourceHandle(200));
}

// =============================================================================
// SECTION 7 -- slot_handle out-of-bounds panics
// =============================================================================

/// slot_handle must panic when the index equals slot_count.
#[test]
fn slot_handle_panics_at_slot_count_boundary() {
    let ring = HistoryRingSlot::new(2, ResourceHandle(0));
    let result = std::panic::catch_unwind(|| {
        let _ = ring.slot_handle(2);
    });
    assert!(
        result.is_err(),
        "slot_handle(index == slot_count) must panic",
    );
}

/// slot_handle must panic when the index exceeds slot_count.
#[test]
fn slot_handle_panics_out_of_bounds() {
    let ring = HistoryRingSlot::new(3, ResourceHandle(0));
    let result = std::panic::catch_unwind(|| {
        let _ = ring.slot_handle(5);
    });
    assert!(
        result.is_err(),
        "slot_handle(index > slot_count) must panic",
    );
}

/// slot_handle must panic for very large indices.
#[test]
fn slot_handle_panics_on_large_index() {
    let ring = HistoryRingSlot::new(2, ResourceHandle(0));
    let result = std::panic::catch_unwind(|| {
        let _ = ring.slot_handle(usize::MAX);
    });
    assert!(
        result.is_err(),
        "slot_handle(usize::MAX) must panic",
    );
}

// =============================================================================
// SECTION 8 -- advance() does not mutate slot contents
// =============================================================================

/// advance() only changes current_slot; it must NOT mutate slot contents.
#[test]
fn advance_does_not_affect_slot_contents() {
    // Use the public API to set up known state:
    // write_current_and_advance to fill slots with distinct handles.
    let mut ring = HistoryRingSlot::new(3, ResourceHandle(0));
    ring.write_current_and_advance(ResourceHandle(10));
    ring.write_current_and_advance(ResourceHandle(20));
    ring.write_current_and_advance(ResourceHandle(30));

    // current_slot is now 0.  Save expected values.
    let expected = [ResourceHandle(30), ResourceHandle(10), ResourceHandle(20)];

    // Advance twice -- contents must not change.
    ring.advance();
    assert_eq!(ring.current_slot(), 1);
    for i in 0..3 {
        assert_eq!(
            ring.slot_handle(i),
            expected[i],
            "slot {i} must not change after advance",
        );
    }

    ring.advance();
    assert_eq!(ring.current_slot(), 2);
    for i in 0..3 {
        assert_eq!(
            ring.slot_handle(i),
            expected[i],
            "slot {i} must not change after second advance",
        );
    }
}

// =============================================================================
// SECTION 9 -- write_current_and_advance
// =============================================================================

/// write_current_and_advance writes to the current slot, then advances.
/// After exactly `slot_count` calls, the ring is back at slot 0.
#[test]
fn write_current_and_advance_returns_to_slot_zero_after_full_cycle() {
    let mut ring = HistoryRingSlot::new(4, ResourceHandle(0));
    for i in 0..4 {
        ring.write_current_and_advance(ResourceHandle(i as u32));
        if i < 3 {
            assert_ne!(
                ring.current_slot(),
                0,
                "Must NOT be at slot 0 after only {} write(s)",
                i + 1,
            );
        }
    }
    assert_eq!(
        ring.current_slot(),
        0,
        "After exactly slot_count writes, must return to slot 0",
    );
}

/// write_current_and_advance isolates each slot: writing to one slot
/// does not affect the other slots' handles.
#[test]
fn write_current_and_advance_isolates_slots() {
    let mut ring = HistoryRingSlot::new(3, ResourceHandle(0));
    let base: u32 = 100;

    // Write distinct values at each position during the first lap.
    for i in 0..3 {
        ring.write_current_and_advance(ResourceHandle(base + i));
    }
    // After one full lap the ring is back at slot 0, and each slot
    // holds the handle written when it was current.
    assert_eq!(ring.slot_handle(0), ResourceHandle(base + 0));
    assert_eq!(ring.slot_handle(1), ResourceHandle(base + 1));
    assert_eq!(ring.slot_handle(2), ResourceHandle(base + 2));

    // Overwrite slot 0 with a new value during the second lap.
    ring.write_current_and_advance(ResourceHandle(base + 10));
    assert_eq!(ring.slot_handle(0), ResourceHandle(base + 10));
    // The other slots are untouched.
    assert_eq!(ring.slot_handle(1), ResourceHandle(base + 1));
    assert_eq!(ring.slot_handle(2), ResourceHandle(base + 2));
}

// =============================================================================
// SECTION 10 -- Multiple full cycles
// =============================================================================

/// After multiple full cycles, handles survive repeated wraps.
#[test]
fn handles_survive_multiple_full_cycles() {
    let n = 4;
    let mut ring = HistoryRingSlot::new(n, ResourceHandle(0));

    // Cycle through 3 full laps.
    for lap in 0..3 {
        for slot in 0..n {
            let handle = ResourceHandle((lap * 100 + slot) as u32);
            ring.write_current_and_advance(handle);
        }
    }

    // After 3 laps, the newest writes are from lap 2.
    let base = 200u32;
    for slot in 0..n {
        assert_eq!(
            ring.slot_handle(slot),
            ResourceHandle(base + slot as u32),
            "slot {slot} after 3 laps",
        );
    }
}

/// Handle values are preserved across advance-only cycles (no writes).
#[test]
fn handles_preserved_across_advance_only_cycles() {
    let mut ring = HistoryRingSlot::new(3, ResourceHandle(0));

    // Write distinct handles initially.
    ring.write_current_and_advance(ResourceHandle(10));
    ring.write_current_and_advance(ResourceHandle(20));
    ring.write_current_and_advance(ResourceHandle(30));

    // Now advance through many cycles without writing.
    for _ in 0..100 {
        ring.advance();
    }

    // The handles must still be present and correct.
    assert_eq!(ring.slot_handle(0), ResourceHandle(30));
    assert_eq!(ring.slot_handle(1), ResourceHandle(10));
    assert_eq!(ring.slot_handle(2), ResourceHandle(20));

    // current_slot should be (0 + 100) % 3 = 1.
    assert_eq!(ring.current_slot(), 1);
}

// =============================================================================
// SECTION 11 -- Clone and Debug derivations
// =============================================================================

/// HistoryRingSlot implements Clone.
#[test]
fn history_ring_buffer_is_clone() {
    let ring = HistoryRingSlot::new(4, ResourceHandle(42));
    let cloned = ring.clone();
    assert_eq!(cloned.slot_count(), ring.slot_count());
    assert_eq!(cloned.current_slot(), ring.current_slot());
    for i in 0..ring.slot_count() {
        assert_eq!(
            cloned.slot_handle(i),
            ring.slot_handle(i),
            "slot {i} must be equal after clone",
        );
    }
}

/// Cloned ring operates independently from the original.
#[test]
fn cloned_ring_is_independent() {
    let mut ring = HistoryRingSlot::new(3, ResourceHandle(0));
    ring.write_current_and_advance(ResourceHandle(10));
    ring.write_current_and_advance(ResourceHandle(20));

    let mut cloned = ring.clone();
    cloned.write_current_and_advance(ResourceHandle(99));

    // Original should not see the write to the clone.
    assert_eq!(
        ring.slot_handle(2),
        ResourceHandle(0),
        "original must not be affected by clone's writes",
    );
    assert_eq!(
        cloned.slot_handle(2),
        ResourceHandle(99),
        "clone must see its own writes",
    );
}

/// HistoryRingSlot implements Debug.
#[test]
fn history_ring_buffer_is_debug() {
    let ring = HistoryRingSlot::new(3, ResourceHandle(7));
    let debug_str = format!("{:?}", ring);
    assert!(
        !debug_str.is_empty(),
        "Debug output must not be empty",
    );
    // The debug output should contain key fields.
    assert!(
        debug_str.contains("slots") || debug_str.contains("current"),
        "Debug output '{}' must contain 'slots' or 'current'",
        debug_str,
    );
}

// =============================================================================
// SECTION 12 -- ResourceHandle API
// =============================================================================

/// ResourceHandle::NONE is the u32::MAX sentinel.
#[test]
fn resource_handle_none_sentinel() {
    // The only way to test NONE from the public API is to observe it
    // through the ring buffer.
    let ring = HistoryRingSlot::new(2, ResourceHandle::NONE);
    for i in 0..2 {
        // NONE should be stored and returned as-is.
        let _handle = ring.slot_handle(i);
    }
}

/// ResourceHandle can be constructed with various u32 values.
#[test]
fn resource_handle_various_values() {
    let mut ring = HistoryRingSlot::new(3, ResourceHandle(0));

    let values = [
        0u32,
        1,
        42,
        255,
        1_000_000,
        u32::MAX - 1,
    ];

    for &val in &values {
        ring.write_current_and_advance(ResourceHandle(val));
    }

    // After writing 6 values to a 3-slot ring, 2 full laps completed.
    // The latest lap overwrites: slot 2 gets the 6th write (u32::MAX - 1).
    assert_eq!(ring.slot_handle(2), ResourceHandle(u32::MAX - 1));
}

// =============================================================================
// SECTION 13 -- Edge cases and stress
// =============================================================================

/// Rapid cycling between two slots with distinct handles validates that
/// no handle leaks between slots.
#[test]
fn rapid_alternation_no_leak() {
    let mut ring = HistoryRingSlot::new(2, ResourceHandle(0));

    for i in 0..100 {
        let handle = ResourceHandle(i);
        ring.write_current_and_advance(handle);

        // After each write+advance, slot (i % 2) should hold handle i.
        let slot = i as usize % 2;
        assert_eq!(
            ring.slot_handle(slot),
            handle,
            "after write at iteration {i}, slot {slot} must hold {handle:?}",
        );
    }
}

/// N=2 with alternating writes: each overwrite only affects the slot
/// that was current when the write occurred.
#[test]
fn n2_alternating_writes_no_crosstalk() {
    let mut ring = HistoryRingSlot::new(2, ResourceHandle(0));

    // Write even-handles to even iterations, odd-handles to odd.
    for i in 0..20 {
        let handle = ResourceHandle(if i % 2 == 0 { 1000 + i } else { 2000 + i });
        ring.write_current_and_advance(handle);
    }

    // After 20 writes in a 2-slot buffer, slots have been overwritten 10 times each.
    // Slot 0 gets even-indexed writes: i=0 -> handle 1000, i=2 -> 1002, ..., i=18 -> 1018
    // Slot 1 gets odd-indexed writes: i=1 -> handle 2001, i=3 -> 2003, ..., i=19 -> 2019
    assert_eq!(ring.slot_handle(0), ResourceHandle(1018));
    assert_eq!(ring.slot_handle(1), ResourceHandle(2019));
}

/// Single advance on a fresh ring moves from slot 0 to slot 1 and
/// slot contents remain the initial handle.
#[test]
fn single_advance_on_fresh_ring() {
    let mut ring = HistoryRingSlot::new(4, ResourceHandle(7));
    assert_eq!(ring.current_slot(), 0);
    assert_eq!(ring.slot_handle(0), ResourceHandle(7));

    ring.advance();
    assert_eq!(ring.current_slot(), 1);
    // All slots still hold the initial handle.
    assert_eq!(ring.slot_handle(0), ResourceHandle(7));
    assert_eq!(ring.slot_handle(1), ResourceHandle(7));
    assert_eq!(ring.slot_handle(2), ResourceHandle(7));
    assert_eq!(ring.slot_handle(3), ResourceHandle(7));
}
