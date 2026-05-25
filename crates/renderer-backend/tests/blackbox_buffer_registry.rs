// Blackbox contract tests for BufferRegistry (T-GPU-1.2).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::gpu_driven::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criterion (T-GPU-1.2):
//   CPU writes frame N while GPU reads frame N-1, NO SYNC STALLS.
//   Back-pressure only when all 3 staging slots are occupied, at which
//   point the application may throttle gracefully instead of stalling.
//
// The BufferRegistry guarantees:
//   - No sync stalls in the common case: the CPU always has at least one
//     free slot to write into.
//   - Back-pressure signals via AcquireResult::NoSlotAvailable when all
//     three slots are occupied by pending GPU work.
//
// Coverage:
//   1.  Core double-buffer guarantee: CPU writes frame N, GPU reads N-1
//   2.  No-sync-stall in steady-state frame loop (6+ iterations)
//   3.  Stall only when all 3 slots occupied (back-pressure boundary)
//   4.  acquire_reading returns newest Ready frame (highest frame_index)
//   5.  Concurrent-frame data integrity (written data survives pipeline)
//   6.  Cannot submit/release an unacquired slot (InvalidSlot)
//   7.  acquire_reading returns None when no Ready slot exists
//   8.  Release recycles the slot for CPU reuse
//   9.  Reset restores all slots to Free with zero frame_count
//  10.  ensure_capacity grows only Free slots
//  11.  StagingBufferDesc alignment (default + custom)
//  12.  Zero-capacity constructor panics
//  13.  Large data through the pipeline (1 MB)
//  14.  Zero-byte submit is valid
//  15.  Exact-capacity submit is valid
//  16.  Slot index bounds (out-of-range is InvalidSlot)

use renderer_backend::gpu_driven::{
    AcquireResult, BufferRegistry, BufferSlot, ReleaseResult, SlotState,
    StagingBufferDesc, SubmitResult, NUM_STAGING_SLOTS,
};

// SlotState is used in full_cycle_twice_on_same_slot for state assertions.

// =============================================================================
// Helpers
// =============================================================================

/// Acquire a staging slot, panicking if none available.
fn acquire(reg: &mut BufferRegistry) -> usize {
    match reg.acquire_staging() {
        AcquireResult::Acquired { slot_index } => slot_index,
        AcquireResult::NoSlotAvailable => {
            panic!("acquire_staging() returned NoSlotAvailable when we expected a slot")
        }
    }
}

/// Submit a slot with `written_size` bytes and verify it succeeded.
fn submit(reg: &mut BufferRegistry, idx: usize, written_size: usize) {
    match reg.submit_staging(idx, written_size) {
        SubmitResult::Submitted => {}
        SubmitResult::InvalidSlot => {
            panic!("submit_staging({}) returned InvalidSlot", idx);
        }
    }
}

/// Acquire the newest Ready slot for reading, panicking if none.
fn acquire_read(reg: &mut BufferRegistry) -> usize {
    reg.acquire_reading()
        .expect("acquire_reading() returned None when we expected a Ready slot")
}

/// Release a slot and verify it succeeded.
fn release(reg: &mut BufferRegistry, idx: usize) {
    match reg.release_staging(idx) {
        ReleaseResult::Released => {}
        ReleaseResult::InvalidSlot => {
            panic!("release_staging({}) returned InvalidSlot", idx);
        }
    }
}

/// Write a deterministic byte pattern into a slot's mutable slice.
fn write_pattern(slot: &mut BufferSlot, len: usize, seed: u8) {
    let buf = slot.as_mut_slice();
    assert!(
        len <= buf.len(),
        "write_pattern: len {} exceeds capacity {}",
        len,
        buf.len()
    );
    for i in 0..len {
        buf[i] = seed.wrapping_add(i as u8);
    }
}

/// Verify a slot's readable data matches the expected pattern.
fn assert_pattern(slot: &BufferSlot, expected_len: usize, seed: u8) {
    assert_eq!(
        slot.size(),
        expected_len,
        "pattern length mismatch: expected {}, got {}",
        expected_len,
        slot.size()
    );
    let data = slot.as_slice();
    for i in 0..expected_len {
        let expected = seed.wrapping_add(i as u8);
        assert_eq!(
            data[i], expected,
            "pattern mismatch at byte {}: expected 0x{:02x}, got 0x{:02x}",
            i, expected, data[i]
        );
    }
}

// =============================================================================
// SECTION 1 -- Core double-buffer guarantee (T-GPU-1.2 acceptance criteria)
// =============================================================================

/// THE ACCEPTANCE TEST: CPU writes frame N while GPU reads frame N-1.
///
/// This validates the core double-buffer contract (T-GPU-1.2):
///
///   Flow:
///     CPU: acquire slot 0, write frame 0, submit             [frame_count=1]
///     CPU: acquire slot 1, write frame 1, submit             [frame_count=2]
///     GPU: acquire_read -> slot 1 (newest, frame 1)
///     CPU: acquire slot 2, write frame 2, submit             [frame_count=3]
///          ^-- CPU writes frame 2 WHILE GPU reads frame 1
///     GPU: release(slot 1)
///     GPU: acquire_read -> slot 2 (newest, frame 2)
///     GPU: release(slot 2)
///     GPU: acquire_read -> slot 0 (frame 0, remaining Ready)
///     GPU: release(slot 0)
///
///   All 3 frames verified for data integrity.
///   No sync stall: the CPU always had a Free slot to write into.
#[test]
fn cpu_writes_frame_n_while_gpu_reads_frame_n_minus_1() {
    let mut reg = BufferRegistry::new(4096);

    // --- Frame 0: CPU writes -------------------------------------------------
    let slot0 = acquire(&mut reg);
    write_pattern(reg.slot_mut(slot0).unwrap(), 128, 0xA0);
    submit(&mut reg, slot0, 128); // frame_count becomes 1

    // --- Frame 1: CPU writes -------------------------------------------------
    let slot1 = acquire(&mut reg);
    write_pattern(reg.slot_mut(slot1).unwrap(), 256, 0xB0);
    submit(&mut reg, slot1, 256); // frame_count becomes 2

    // --- GPU begins reading the newest frame (frame 1) -----------------------
    let read_idx = acquire_read(&mut reg);
    assert_eq!(read_idx, slot1, "GPU must read the newest frame (slot 1)");
    assert_eq!(reg.free_slots(), 1, "Slot 2 remains Free for CPU writes");

    // --- Frame 2: CPU writes WHILE GPU reads frame 1 -------------------------
    // THIS IS THE CORE GUARANTEE: CPU gets a Free slot without stalling.
    let slot_for_frame2 = acquire(&mut reg);
    assert!(
        slot_for_frame2 < NUM_STAGING_SLOTS,
        "CPU must acquire a slot without stalling (T-GPU-1.2)"
    );
    write_pattern(reg.slot_mut(slot_for_frame2).unwrap(), 64, 0xC0);
    submit(&mut reg, slot_for_frame2, 64); // frame_count becomes 3

    // Data integrity: frame 1 data intact while GPU held it.
    assert_pattern(reg.slot(read_idx).unwrap(), 256, 0xB0);

    // --- GPU finishes frame 1, releases --------------------------------------
    release(&mut reg, read_idx);

    // --- GPU reads frame 2 (now the newest Ready) ----------------------------
    let read_idx2 = acquire_read(&mut reg);
    assert_eq!(
        read_idx2, slot_for_frame2,
        "GPU must read the newest Ready frame (frame 2, slot {})",
        slot_for_frame2
    );
    assert_pattern(reg.slot(read_idx2).unwrap(), 64, 0xC0);
    release(&mut reg, read_idx2);

    // --- GPU reads frame 0 (remaining Ready) ---------------------------------
    let read_idx3 = acquire_read(&mut reg);
    assert_eq!(read_idx3, slot0, "GPU must read frame 0 (slot 0)");
    assert_pattern(reg.slot(read_idx3).unwrap(), 128, 0xA0);
    release(&mut reg, read_idx3);

    // Final state: all slots free, frame count is 3.
    assert_eq!(
        reg.free_slots(),
        NUM_STAGING_SLOTS,
        "All slots must be Free at end of clean cycle"
    );
    assert_eq!(reg.frame_count(), 3, "Three frames submitted");
}

// =============================================================================
// SECTION 2 -- No-sync-stall guarantee
// =============================================================================

/// The CPU never stalls in steady-state double-buffering.
///
/// Simulates 6 full frame cycles where the CPU writes a frame, submits it,
/// the GPU acquires it for reading, and releases it. In the steady state the
/// CPU always has a Free slot available and never sees NoSlotAvailable.
#[test]
fn no_sync_stall_in_steady_state_frame_loop() {
    let mut reg = BufferRegistry::new(1024);
    const ITERATIONS: usize = 6;

    for frame in 0..ITERATIONS {
        let seed = 0x10 + (frame as u8) * 0x10;
        let size = 64 + frame * 16;

        // CPU: acquire, write, submit
        let idx = reg.acquire_staging();
        let slot_idx = match idx {
            AcquireResult::Acquired { slot_index } => slot_index,
            AcquireResult::NoSlotAvailable => {
                panic!(
                    "CPU STALLED at frame {}: no slot available. \
                     T-GPU-1.2 requires no sync stalls in steady state.",
                    frame
                );
            }
        };

        write_pattern(reg.slot_mut(slot_idx).unwrap(), size, seed);
        submit(&mut reg, slot_idx, size);

        // GPU: acquire latest ready, read, release
        let read_idx = acquire_read(&mut reg);
        assert_pattern(reg.slot(read_idx).unwrap(), size, seed);
        release(&mut reg, read_idx);
    }

    assert_eq!(
        reg.frame_count(),
        ITERATIONS as u64,
        "All {} frames submitted successfully without stall",
        ITERATIONS
    );
}

/// Back-pressure (NoSlotAvailable) only triggers when all 3 slots are
/// occupied by pending GPU work. The application can then throttle
/// gracefully instead of stalling the pipeline.
#[test]
fn stall_only_when_all_three_slots_occupied() {
    let mut reg = BufferRegistry::new(256);

    // Acquire all 3 slots for CPU writing.
    let s0 = acquire(&mut reg);
    assert_eq!(reg.free_slots(), 2, "1 slot acquired, 2 remain Free");
    assert!(!reg.is_stalled(), "1 slot acquired -- not stalled");

    let s1 = acquire(&mut reg);
    assert_eq!(reg.free_slots(), 1, "2 slots acquired, 1 remains Free");
    assert!(!reg.is_stalled(), "2 slots acquired -- not stalled");

    let s2 = acquire(&mut reg);
    assert_eq!(reg.free_slots(), 0, "All 3 slots acquired -- 0 Free");
    assert!(
        reg.is_stalled(),
        "is_stalled() must be true when all 3 slots are occupied \
         (even if they are all in Writing state)"
    );

    // Submit all 3 (Writing -> Ready).
    submit(&mut reg, s0, 32);
    submit(&mut reg, s1, 32);
    submit(&mut reg, s2, 32);

    // Still 0 Free (all 3 are Ready). Attempting another acquire_staging
    // must return NoSlotAvailable.
    assert_eq!(reg.free_slots(), 0, "Still 0 Free after submit");
    match reg.acquire_staging() {
        AcquireResult::NoSlotAvailable => { /* expected back-pressure */ }
        AcquireResult::Acquired { slot_index } => {
            panic!(
                "Acquired slot {} when all 3 slots are occupied. \
                 T-GPU-1.2 demands back-pressure via NoSlotAvailable.",
                slot_index
            );
        }
    }

    // acquire_reading consumes 1 (Ready -> Reading). Still 0 Free.
    let r0 = acquire_read(&mut reg);
    assert_eq!(reg.free_slots(), 0, "Still 0 Free after one read");

    // Release the completed slot (Reading -> Free).
    release(&mut reg, r0);
    assert_eq!(reg.free_slots(), 1, "1 slot freed");
    assert!(!reg.is_stalled(), "1 Free slot available -- not stalled");

    // Now CPU can acquire again without stalling.
    let recycled = acquire(&mut reg);
    assert!(recycled < NUM_STAGING_SLOTS, "Must get a valid recycled slot");
    assert_eq!(reg.free_slots(), 0, "Re-acquired slot consumes the Free slot");
}

// =============================================================================
// SECTION 3 -- acquire_reading returns newest frame
// =============================================================================

/// acquire_reading must return the slot with the highest frame_index
/// (i.e., the most recently submitted frame) among all Ready slots.
#[test]
fn acquire_reading_returns_newest_submitted_frame() {
    let mut reg = BufferRegistry::new(512);

    // Submit frames in order: s1, s0 (so s1 has a higher frame_index).
    let s0 = acquire(&mut reg);
    let s1 = acquire(&mut reg);

    submit(&mut reg, s0, 16); // frame_count becomes 1
    submit(&mut reg, s1, 16); // frame_count becomes 2

    // Both s0 and s1 are Ready. acquire_reading must return the newest (s1).
    let read = acquire_read(&mut reg);
    assert_eq!(
        read, s1,
        "acquire_reading must return the newest slot (highest frame_index), \
         got slot {} instead of {}",
        read, s1
    );
}

/// After consuming the newest frame, the next acquire_reading returns
/// the second-newest frame.
#[test]
fn acquire_reading_returns_second_newest_after_consuming_newest() {
    let mut reg = BufferRegistry::new(512);

    let s0 = acquire(&mut reg);
    let s1 = acquire(&mut reg);

    submit(&mut reg, s0, 16); // frame 1
    submit(&mut reg, s1, 16); // frame 2

    // Consume newest (frame 2).
    let first = acquire_read(&mut reg);
    assert_eq!(first, s1, "First read must be newest slot");

    // After consuming newest, the remaining Ready slot is s0 (frame 1).
    release(&mut reg, first);

    let second = acquire_read(&mut reg);
    assert_eq!(second, s0, "Second read must be slot 0 (frame 1)");
    release(&mut reg, second);
}

// =============================================================================
// SECTION 4 -- Concurrent-frame data integrity
// =============================================================================

/// When CPU writes frame N while GPU reads frame N-1, each frame's data
/// must remain independent and uncorrupted.
#[test]
fn concurrent_frame_data_integrity() {
    let mut reg = BufferRegistry::new(8192);
    let patterns: [(u8, usize); 4] = [
        (0xAA, 64),  // Frame 0: 64 bytes of 0xAA-seeded data
        (0xBB, 128), // Frame 1: 128 bytes
        (0xCC, 256), // Frame 2: 256 bytes
        (0xDD, 512), // Frame 3: 512 bytes
    ];

    // --- Submit frame 0 ------------------------------------------------------
    let s0 = acquire(&mut reg);
    write_pattern(reg.slot_mut(s0).unwrap(), patterns[0].1, patterns[0].0);
    submit(&mut reg, s0, patterns[0].1);

    // --- Submit frame 1 (while frame 0 is Ready) -----------------------------
    let s1 = acquire(&mut reg);
    write_pattern(reg.slot_mut(s1).unwrap(), patterns[1].1, patterns[1].0);
    submit(&mut reg, s1, patterns[1].1);

    // --- GPU reads frame 1 (newest), CPU writes frame 2 ----------------------
    let read1 = acquire_read(&mut reg);
    assert_eq!(read1, s1, "GPU must read newest (frame 1)");

    // CPU writes frame 2 WHILE GPU reads frame 1.
    let s2 = acquire(&mut reg);
    write_pattern(reg.slot_mut(s2).unwrap(), patterns[2].1, patterns[2].0);
    submit(&mut reg, s2, patterns[2].1);

    // Verify frame 1 data is intact while GPU reads it.
    assert_pattern(reg.slot(read1).unwrap(), patterns[1].1, patterns[1].0);
    release(&mut reg, read1);

    // --- GPU reads frame 2, CPU writes frame 3 -------------------------------
    let read2 = acquire_read(&mut reg);
    assert_eq!(read2, s2, "GPU must read newest (frame 2)");

    let s3 = acquire(&mut reg);
    write_pattern(reg.slot_mut(s3).unwrap(), patterns[3].1, patterns[3].0);
    submit(&mut reg, s3, patterns[3].1);

    assert_pattern(reg.slot(read2).unwrap(), patterns[2].1, patterns[2].0);
    release(&mut reg, read2);

    // --- GPU reads frame 3 (last frame) --------------------------------------
    let read3 = acquire_read(&mut reg);
    assert_eq!(read3, s3, "GPU must read newest (frame 3)");
    assert_pattern(reg.slot(read3).unwrap(), patterns[3].1, patterns[3].0);
    release(&mut reg, read3);

    // --- GPU reads frame 0 (oldest, still Ready) -----------------------------
    let read0 = acquire_read(&mut reg);
    assert_eq!(read0, s0, "GPU must read remaining slot (frame 0)");
    assert_pattern(reg.slot(read0).unwrap(), patterns[0].1, patterns[0].0);
    release(&mut reg, read0);

    // All 4 frames verified. No data corruption across concurrent writes/reads.
    assert_eq!(reg.free_slots(), NUM_STAGING_SLOTS);
    assert_eq!(reg.frame_count(), 4);
}

// =============================================================================
// SECTION 5 -- Error paths: invalid slot operations
// =============================================================================

/// Submitting an unacquired slot (Free -> Submit) must fail.
#[test]
fn submit_unacquired_slot_fails() {
    let mut reg = BufferRegistry::new(256);
    // Slot 0 was never acquired -- submitting it should fail.
    assert!(
        matches!(reg.submit_staging(0, 16), SubmitResult::InvalidSlot),
        "Submitting a Free slot must return InvalidSlot"
    );
}

/// Releasing an unacquired slot (Free -> Release) must fail.
#[test]
fn release_unacquired_slot_fails() {
    let mut reg = BufferRegistry::new(256);
    assert!(
        matches!(reg.release_staging(0), ReleaseResult::InvalidSlot),
        "Releasing a Free slot must return InvalidSlot"
    );
}

/// Submitting a slot that is in Writing state but with an out-of-range
/// index must return InvalidSlot.
#[test]
fn submit_out_of_range_index_fails() {
    let mut reg = BufferRegistry::new(256);
    assert!(
        matches!(
            reg.submit_staging(NUM_STAGING_SLOTS, 16),
            SubmitResult::InvalidSlot
        ),
        "Out-of-range slot index must return InvalidSlot"
    );
    assert!(
        matches!(
            reg.submit_staging(usize::MAX, 16),
            SubmitResult::InvalidSlot
        ),
        "Very large slot index must return InvalidSlot"
    );
}

/// Releasing an out-of-range index must return InvalidSlot.
#[test]
fn release_out_of_range_index_fails() {
    let mut reg = BufferRegistry::new(256);
    assert!(
        matches!(
            reg.release_staging(NUM_STAGING_SLOTS),
            ReleaseResult::InvalidSlot
        ),
        "Out-of-range release index must return InvalidSlot"
    );
}

/// Releasing a slot in Writing state must fail (must be Ready or Reading).
#[test]
fn release_writing_slot_fails() {
    let mut reg = BufferRegistry::new(256);
    let idx = acquire(&mut reg);
    // Slot is now Writing. Releasing it directly should fail.
    assert!(
        matches!(reg.release_staging(idx), ReleaseResult::InvalidSlot),
        "Releasing a Writing slot must return InvalidSlot"
    );
}

// =============================================================================
// SECTION 6 -- acquire_reading edge cases
// =============================================================================

/// When no slot has been submitted, acquire_reading returns None.
#[test]
fn acquire_reading_returns_none_when_nothing_ready() {
    let mut reg = BufferRegistry::new(256);
    assert!(
        reg.acquire_reading().is_none(),
        "acquire_reading must return None when no Ready slot exists"
    );
}

/// After all Ready slots have been consumed, acquire_reading returns None.
#[test]
fn acquire_reading_returns_none_after_all_consumed() {
    let mut reg = BufferRegistry::new(256);
    let idx = acquire(&mut reg);
    submit(&mut reg, idx, 8);

    let _read = acquire_read(&mut reg);
    // Slot is now Reading, not Ready. acquire_reading should return None.
    assert!(
        reg.acquire_reading().is_none(),
        "acquire_reading must return None when all Ready slots have been consumed"
    );
}

// =============================================================================
// SECTION 7 -- Release recycles slots
// =============================================================================

/// After a slot is released (Reading -> Free), the CPU can re-acquire it.
/// Note: the released slot may differ from the acquired index because the
/// round-robin pointer advances. The valid assertion is that any slot is
/// returned (i.e., acquire does not stall).
#[test]
fn released_slot_is_recycled_for_cpu_write() {
    let mut reg = BufferRegistry::new(256);

    let idx = acquire(&mut reg);
    submit(&mut reg, idx, 8);
    let read = acquire_read(&mut reg);
    release(&mut reg, read);

    // The freed slot should be immediately reusable by the CPU.
    let free_before = reg.free_slots();
    let recycled = acquire(&mut reg);
    assert!(
        recycled < NUM_STAGING_SLOTS,
        "After release, acquire must return a valid slot index"
    );
    assert_eq!(
        reg.free_slots(),
        free_before - 1,
        "After acquire, free_slots must decrease by 1"
    );
}

/// Passing slots through the full cycle twice validates state machine
/// reset: Free -> Writing -> Ready -> Reading -> Free -> (repeat).
/// The round-robin pointer may change which slot is returned, so we
/// verify the cycle semantics, not the specific index.
#[test]
fn full_cycle_twice_on_same_slot() {
    let mut reg = BufferRegistry::new(256);

    // Cycle 1.
    let idx = acquire(&mut reg);
    write_pattern(reg.slot_mut(idx).unwrap(), 32, 0xE0);
    submit(&mut reg, idx, 32);
    let read = acquire_read(&mut reg);
    assert_pattern(reg.slot(read).unwrap(), 32, 0xE0);
    release(&mut reg, read);

    // Cycle 2: acquire another slot (round-robin moves forward).
    let idx2 = acquire(&mut reg);
    assert!(
        idx2 < NUM_STAGING_SLOTS,
        "Second cycle acquire must return a valid slot index"
    );
    // The re-acquired slot should have size=0 (reset by release).
    assert_eq!(
        reg.slot(idx2).unwrap().size(),
        0,
        "Re-acquired slot must have size 0 (reset by prior release)"
    );
    assert_eq!(
        reg.slot(idx2).unwrap().state(),
        SlotState::Writing,
        "Re-acquired slot must be in Writing state"
    );

    write_pattern(reg.slot_mut(idx2).unwrap(), 64, 0xF0);
    submit(&mut reg, idx2, 64);
    let read2 = acquire_read(&mut reg);
    assert_pattern(reg.slot(read2).unwrap(), 64, 0xF0);
    release(&mut reg, read2);

    assert_eq!(
        reg.frame_count(),
        2,
        "Two full cycles must produce frame_count=2"
    );
}

// =============================================================================
// SECTION 8 -- Reset
// =============================================================================

/// Reset restores all slots to Free and zeroes frame_count.
#[test]
fn reset_restores_all_slots_to_free() {
    let mut reg = BufferRegistry::new(1024);

    // Put the registry into a used state.
    let s0 = acquire(&mut reg);
    let s1 = acquire(&mut reg);
    submit(&mut reg, s0, 32);
    submit(&mut reg, s1, 64);
    let _r = acquire_read(&mut reg);
    // s0/s1 are Ready->Consumed, s2 is still Writing.

    reg.reset();

    assert_eq!(
        reg.free_slots(),
        NUM_STAGING_SLOTS,
        "After reset, all {} slots must be Free",
        NUM_STAGING_SLOTS
    );
    assert_eq!(reg.ready_slots(), 0, "After reset, no Ready slots");
    assert_eq!(reg.frame_count(), 0, "After reset, frame_count must be 0");
    assert!(!reg.is_stalled(), "After reset, must not be stalled");

    // After reset, acquire should work immediately.
    let fresh = acquire(&mut reg);
    assert!(
        fresh < NUM_STAGING_SLOTS,
        "After reset, acquire must return a valid slot"
    );
}

/// After reset, the frame counter starts from 0.
#[test]
fn reset_resets_frame_count() {
    let mut reg = BufferRegistry::new(256);
    for _ in 0..5 {
        let idx = acquire(&mut reg);
        submit(&mut reg, idx, 8);
        let r = acquire_read(&mut reg);
        release(&mut reg, r);
    }
    assert_eq!(reg.frame_count(), 5, "Five frames submitted");

    reg.reset();
    assert_eq!(reg.frame_count(), 0, "Frame count must reset to 0");

    // First submission after reset should set frame_count to 1.
    let idx = acquire(&mut reg);
    submit(&mut reg, idx, 8);
    assert_eq!(reg.frame_count(), 1, "First post-reset submit must set frame_count to 1");
}

// =============================================================================
// SECTION 9 -- ensure_capacity
// =============================================================================

/// ensure_capacity grows only Free slots; non-Free slots keep their size.
#[test]
fn ensure_capacity_grows_only_free_slots() {
    let mut reg = BufferRegistry::new(128);

    let idx = acquire(&mut reg); // Slot idx moves to Writing.
    let original_capacity = reg.slot(idx).unwrap().capacity();

    reg.ensure_capacity(4096);

    // The acquired (Writing) slot must still have its original capacity.
    assert_eq!(
        reg.slot(idx).unwrap().capacity(),
        original_capacity,
        "Non-Free slot must NOT be resized by ensure_capacity"
    );

    // The remaining Free slots should have >= 4096 capacity.
    for i in 0..NUM_STAGING_SLOTS {
        let slot = reg.slot(i).unwrap();
        if i == idx {
            continue; // Skipped, checked above.
        }
        assert!(
            slot.capacity() >= 4096,
            "Free slot {} must have capacity >= 4096, got {}",
            i,
            slot.capacity()
        );
    }
}

/// ensure_capacity is a no-op when all slots already meet the minimum.
#[test]
fn ensure_capacity_noop_when_sufficient() {
    let mut reg = BufferRegistry::new(1024);
    let caps_before: Vec<usize> = (0..NUM_STAGING_SLOTS)
        .map(|i| reg.slot(i).unwrap().capacity())
        .collect();

    reg.ensure_capacity(512); // Below current capacity.

    for i in 0..NUM_STAGING_SLOTS {
        assert_eq!(
            reg.slot(i).unwrap().capacity(),
            caps_before[i],
            "ensure_capacity must not shrink slots"
        );
    }
}

// =============================================================================
// SECTION 10 -- StagingBufferDesc alignment
// =============================================================================

/// Default alignment rounds up to MIN_GPU_ALIGNMENT (256).
#[test]
fn staging_desc_default_alignment() {
    let desc = StagingBufferDesc::new(100);
    assert_eq!(
        desc.aligned_size(),
        256,
        "StagingBufferDesc::new(100) must round up to 256"
    );

    let desc = StagingBufferDesc::new(256);
    assert_eq!(
        desc.aligned_size(),
        256,
        "StagingBufferDesc::new(256) is already aligned"
    );

    let desc = StagingBufferDesc::new(257);
    assert_eq!(
        desc.aligned_size(),
        512,
        "StagingBufferDesc::new(257) must round up to 512"
    );
}

/// Custom alignment is respected.
#[test]
fn staging_desc_custom_alignment() {
    let desc = StagingBufferDesc::with_alignment(100, 64);
    assert_eq!(
        desc.aligned_size(),
        128,
        "100 with alignment 64 must round to 128"
    );

    let desc = StagingBufferDesc::with_alignment(100, 128);
    assert_eq!(
        desc.aligned_size(),
        128,
        "100 with alignment 128 must round to 128"
    );

    let desc = StagingBufferDesc::with_alignment(100, 1);
    assert_eq!(
        desc.aligned_size(),
        100,
        "Alignment 1 must be a no-op"
    );

    // Alignment 0 should use MIN_GPU_ALIGNMENT.
    let desc = StagingBufferDesc::with_alignment(100, 0);
    assert_eq!(
        desc.aligned_size(),
        256,
        "Alignment 0 must use MIN_GPU_ALIGNMENT (256)"
    );
}

/// StagingBufferDesc::default() has size 0 and MIN_GPU_ALIGNMENT.
/// aligned_size() of 0 bytes with any alignment is 0.
#[test]
fn staging_desc_default() {
    let desc = StagingBufferDesc::default();
    assert_eq!(desc.size, 0, "Default size must be 0");
    assert_eq!(
        desc.alignment,
        256,
        "Default alignment must be MIN_GPU_ALIGNMENT (256)"
    );
    // aligned_size of 0 bytes is 0, regardless of alignment.
    assert_eq!(
        desc.aligned_size(),
        0,
        "aligned_size() of size 0 must return 0"
    );
}

// =============================================================================
// SECTION 11 -- Edge cases
// =============================================================================

/// Zero-capacity must panic.
#[test]
#[should_panic(expected = "requires a positive default capacity")]
fn new_with_zero_capacity_panics() {
    let _reg = BufferRegistry::new(0);
}

/// Large data (1 MB) through the pipeline validates capacity growth and
/// data integrity at scale.
#[test]
fn large_data_through_pipeline() {
    let mut reg = BufferRegistry::new(64); // Start small.

    // ensure_capacity grows Free slots to handle large data.
    reg.ensure_capacity(1_048_576); // 1 MB

    let idx = acquire(&mut reg);
    let slot = reg.slot_mut(idx).unwrap();
    assert!(
        slot.capacity() >= 1_048_576,
        "Slot capacity must be at least 1 MB after ensure_capacity, got {}",
        slot.capacity()
    );

    // Write 1 MB of patterned data.
    let buf = slot.as_mut_slice();
    for (i, byte) in buf.iter_mut().enumerate().take(1_048_576) {
        *byte = (i & 0xFF) as u8;
    }
    submit(&mut reg, idx, 1_048_576);

    // Read it back.
    let read = acquire_read(&mut reg);
    let data = reg.slot(read).unwrap().as_slice();
    assert_eq!(data.len(), 1_048_576, "Large data must preserve length");
    for i in 0..data.len() {
        assert_eq!(
            data[i],
            (i & 0xFF) as u8,
            "Large data corruption at byte {}",
            i
        );
    }
    release(&mut reg, read);
}

/// Zero-byte submit is valid: the slot transitions from Writing to Ready
/// with size 0.
#[test]
fn zero_byte_submit_is_valid() {
    let mut reg = BufferRegistry::new(256);
    let idx = acquire(&mut reg);

    // Submit with 0 bytes written.
    match reg.submit_staging(idx, 0) {
        SubmitResult::Submitted => {}
        SubmitResult::InvalidSlot => {
            panic!("Submitting 0 bytes on an acquired slot must succeed");
        }
    }

    // Slot is now Ready with size 0.
    let read = acquire_read(&mut reg);
    assert_eq!(
        reg.slot(read).unwrap().size(),
        0,
        "Zero-byte submit must produce size 0"
    );
    release(&mut reg, read);
}

/// Submitting exactly capacity bytes is valid (boundary condition).
#[test]
fn exact_capacity_submit_is_valid() {
    let mut reg = BufferRegistry::new(256);
    let idx = acquire(&mut reg);

    // Fill the entire slot.
    let slot = reg.slot_mut(idx).unwrap();
    let cap = slot.capacity();
    for i in 0..cap {
        slot.as_mut_slice()[i] = 0xFF;
    }
    submit(&mut reg, idx, cap);

    let read = acquire_read(&mut reg);
    assert_eq!(
        reg.slot(read).unwrap().size(),
        cap,
        "Submit with capacity bytes must preserve size"
    );
    assert!(
        reg.slot(read).unwrap().as_slice().iter().all(|&b| b == 0xFF),
        "All bytes must be 0xFF after capacity fill"
    );
    release(&mut reg, read);
}

/// Submitting the same byte value across multiple frames tests that
/// residual data from a prior frame does not leak into a new frame.
#[test]
fn no_data_leak_between_frames() {
    let mut reg = BufferRegistry::new(1024);

    // Write frame 0 with all 0xFF.
    let s0 = acquire(&mut reg);
    {
        let buf = reg.slot_mut(s0).unwrap().as_mut_slice();
        buf[..512].fill(0xFF);
    }
    submit(&mut reg, s0, 512);

    // Write frame 1 with all 0x00 in a DIFFERENT slot.
    let s1 = acquire(&mut reg);
    // Should be a different slot since s0 is now Ready.
    assert_ne!(s0, s1, "Must be different slots");
    {
        let buf = reg.slot_mut(s1).unwrap().as_mut_slice();
        buf[..256].fill(0x00);
    }
    submit(&mut reg, s1, 256);

    // Read frame 1 and verify no 0xFF leakage.
    let r1 = acquire_read(&mut reg); // Gets newest (s1).
    assert_eq!(r1, s1);
    let data = reg.slot(r1).unwrap().as_slice();
    assert!(data.iter().all(|&b| b == 0x00), "Frame 1 must not contain 0xFF from frame 0");
    release(&mut reg, r1);

    // Read frame 0 and verify it kept 0xFF.
    let r0 = acquire_read(&mut reg);
    assert_eq!(r0, s0);
    let data = reg.slot(r0).unwrap().as_slice();
    assert!(data.iter().all(|&b| b == 0xFF), "Frame 0 must retain its 0xFF pattern");
    release(&mut reg, r0);
}

// =============================================================================
// SECTION 12 -- Slot accessors and edge conditions
// =============================================================================

/// slot() returns None for out-of-range indices.
#[test]
fn slot_access_out_of_range() {
    let reg = BufferRegistry::new(256);
    assert!(reg.slot(NUM_STAGING_SLOTS).is_none(), "slot(NUM_STAGING_SLOTS) must be None");
    assert!(reg.slot(usize::MAX).is_none(), "slot(usize::MAX) must be None");
}

/// slot_mut() returns None for out-of-range indices.
#[test]
fn slot_mut_access_out_of_range() {
    let mut reg = BufferRegistry::new(256);
    assert!(reg.slot_mut(NUM_STAGING_SLOTS).is_none(), "slot_mut(NUM_STAGING_SLOTS) must be None");
}

/// frame_count() is monotonically increasing with each successful submit.
#[test]
fn frame_count_monotonically_increasing() {
    let mut reg = BufferRegistry::new(256);
    let mut last_count = 0u64;

    for i in 1..=5 {
        let idx = acquire(&mut reg);
        submit(&mut reg, idx, 8);
        let current = reg.frame_count();
        assert!(
            current > last_count,
            "frame_count must increase: {} -> {}",
            last_count,
            current
        );
        assert_eq!(current, i, "frame_count must equal number of submissions");

        // Consume and release so slots are available.
        let r = acquire_read(&mut reg);
        release(&mut reg, r);
        last_count = current;
    }
}

/// ready_slots() counts only slots in Ready state.
#[test]
fn ready_slots_tracking() {
    let mut reg = BufferRegistry::new(256);

    assert_eq!(reg.ready_slots(), 0, "Initially no Ready slots");

    let s0 = acquire(&mut reg);
    assert_eq!(reg.ready_slots(), 0, "Writing slot is not Ready");

    submit(&mut reg, s0, 8);
    assert_eq!(reg.ready_slots(), 1, "One Ready slot after submit");

    let s1 = acquire(&mut reg);
    submit(&mut reg, s1, 8);
    assert_eq!(reg.ready_slots(), 2, "Two Ready slots after second submit");

    let r = acquire_read(&mut reg);
    assert_eq!(reg.ready_slots(), 1, "One Ready after consuming one");
    release(&mut reg, r);

    assert_eq!(reg.ready_slots(), 1, "Slot still Ready after unrelated release");
}

// =============================================================================
// SECTION 13 -- is_stalled edge conditions
// =============================================================================

/// is_stalled() returns false when any slot is Free.
#[test]
fn is_stalled_false_when_any_free() {
    let mut reg = BufferRegistry::new(256);
    assert!(!reg.is_stalled(), "Brand-new registry must not be stalled");

    let _ = acquire(&mut reg);
    assert!(!reg.is_stalled(), "1 slot taken, 2 free -- not stalled");

    let _ = acquire(&mut reg);
    assert!(!reg.is_stalled(), "2 slots taken, 1 free -- not stalled");
}

/// is_stalled() returns true only when all 3 slots are occupied (none Free).
#[test]
fn is_stalled_true_when_all_occupied() {
    let mut reg = BufferRegistry::new(256);
    let _ = acquire(&mut reg);
    let _ = acquire(&mut reg);
    let _ = acquire(&mut reg);

    assert!(
        reg.is_stalled(),
        "All 3 slots acquired (Writing) -- is_stalled must be true"
    );
}

/// is_stalled() returns false after release restores a Free slot.
#[test]
fn is_stalled_recovers_after_release() {
    let mut reg = BufferRegistry::new(256);
    let s0 = acquire(&mut reg);
    let _s1 = acquire(&mut reg);
    let _s2 = acquire(&mut reg);
    assert!(reg.is_stalled(), "All 3 acquired -- stalled");

    submit(&mut reg, s0, 8);
    let r0 = acquire_read(&mut reg);
    // s0 is now Reading. Still no Free slots.
    assert!(reg.is_stalled(), "All occupied (1 Reading, 2 Writing) -- stalled");

    release(&mut reg, r0);
    assert!(!reg.is_stalled(), "After release, 1 Free -- not stalled");
}

// =============================================================================
// SECTION 14 -- Display formatting
// =============================================================================

/// Display output contains key state indicators.
#[test]
fn display_contains_state() {
    let reg = BufferRegistry::new(64);
    let s = format!("{}", reg);
    assert!(s.starts_with("BufferRegistry("), "Display must start with BufferRegistry(");
    assert!(s.contains("free="), "Display must contain free= count");
    assert!(s.contains("ready="), "Display must contain ready= count");
    assert!(s.contains("stalled="), "Display must contain stalled= flag");
    assert!(s.contains("frame="), "Display must contain frame= count");
    assert!(s.contains(']'), "Display must end bracket");

    // After some operations, confirm display updates.
    let mut reg2 = BufferRegistry::new(128);
    let s_initial = format!("{}", reg2);
    assert!(s_initial.contains("free=3"), "Initial display must show free=3");

    let idx = acquire(&mut reg2);
    let s_after_acquire = format!("{}", reg2);
    assert!(s_after_acquire.contains("Writing"), "Display must show Writing state");

    submit(&mut reg2, idx, 16);
    let s_after_submit = format!("{}", reg2);
    assert!(s_after_submit.contains("Ready"), "Display must show Ready state");
    assert!(s_after_submit.contains("frame=1"), "Display must show frame=1");
}

// =============================================================================
// SECTION 15 -- NUM_STAGING_SLOTS constant
// =============================================================================

/// The constant must be exactly 3 (triple-buffering).
#[test]
fn num_staging_slots_is_three() {
    assert_eq!(
        NUM_STAGING_SLOTS, 3,
        "NUM_STAGING_SLOTS must be 3 for triple-buffered staging"
    );
}

/// All indices 0..NUM_STAGING_SLOTS are valid.
#[test]
fn all_slot_indices_valid() {
    let reg = BufferRegistry::new(256);
    for i in 0..NUM_STAGING_SLOTS {
        assert!(
            reg.slot(i).is_some(),
            "Slot {} must be accessible",
            i
        );
    }
}

// =============================================================================
// SECTION 16 -- Round-robin acquire distribution
// =============================================================================

/// Repeated acquire+release cycles should distribute across all 3 slots
/// in round-robin order.
#[test]
fn round_robin_distributes_across_all_slots() {
    let mut reg = BufferRegistry::new(256);
    let mut seen = [false; NUM_STAGING_SLOTS];

    for _ in 0..6 {
        let idx = acquire(&mut reg);
        seen[idx] = true;
        submit(&mut reg, idx, 8);
        let r = acquire_read(&mut reg);
        release(&mut reg, r);
    }

    assert!(
        seen.iter().all(|&s| s),
        "Round-robin acquire must visit all {} slots: {:?}",
        NUM_STAGING_SLOTS,
        seen
    );
}

/// Simulate the exact user-doc example from the BufferRegistry doc comment
/// to confirm the documented API usage compiles and behaves correctly.
#[test]
fn documented_frame_loop_example() {
    // From the doc comment on BufferRegistry:
    //   let mut reg = BufferRegistry::new(1 << 20);
    //   if let AcquireResult::Acquired { slot_index: idx } = reg.acquire_staging() {
    //       let slot = reg.slot_mut(idx).unwrap();
    //       // ... write data into slot.as_mut_slice() ...
    //       reg.submit_staging(idx, written_bytes);
    //   }
    //   // GPU frame
    //   if let Some(idx) = reg.acquire_reading() {
    //       let slot = reg.slot(idx).unwrap();
    //       // ... submit slot.as_slice() to GPU ...
    //       // ... on GPU completion callback:
    //       reg.release_staging(idx);
    //   }

    let mut reg = BufferRegistry::new(1 << 20); // 1 MiB

    // CPU frame
    if let AcquireResult::Acquired { slot_index: idx } = reg.acquire_staging() {
        let slot = reg.slot_mut(idx).unwrap();
        let buf = slot.as_mut_slice();
        buf[..4].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        assert!(matches!(
            reg.submit_staging(idx, 4),
            SubmitResult::Submitted
        ));
    } else {
        panic!("Documented example: acquire_staging must succeed on fresh registry");
    }

    // GPU frame
    if let Some(idx) = reg.acquire_reading() {
        let slot = reg.slot(idx).unwrap();
        assert_eq!(&slot.as_slice()[..4], &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert!(matches!(
            reg.release_staging(idx),
            ReleaseResult::Released
        ));
    } else {
        panic!("Documented example: acquire_reading must find the submitted slot");
    }

    assert_eq!(reg.frame_count(), 1);
    assert_eq!(reg.free_slots(), NUM_STAGING_SLOTS);
}
