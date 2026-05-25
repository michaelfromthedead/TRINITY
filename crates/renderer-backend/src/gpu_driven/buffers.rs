//! GPU-driven BufferRegistry with triple-buffered staging.
//!
//! Eliminates CPU-GPU sync stalls by maintaining three rotating staging
//! slots:
//!
//! - **Frame N**   - CPU writes into the back-buffer slot.
//! - **Frame N-1** - GPU reads from the front-buffer slot.
//! - **Frame N-2** - Slot being reclaimed / recycled.
//!
//! Back-pressure is signalled only when all three slots are occupied by
//! pending GPU work, enabling the application to throttle submission
//! gracefully rather than stalling the pipeline.

use core::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of staging slots in the triple-buffer ring.
pub const NUM_STAGING_SLOTS: usize = 3;

/// Minimum alignment for GPU staging buffers (256 B -- the Vulkan
/// `minStorageBufferOffsetAlignment` guarantee across desktop hardware).
pub const MIN_GPU_ALIGNMENT: usize = 256;

// ---------------------------------------------------------------------------
// Slot state machine
// ---------------------------------------------------------------------------

/// Lifecycle state of a single staging-buffer slot.
///
/// Valid transitions:
/// ```text
///   Free  --acquire-->  Writing  --submit-->  Ready  --start_read-->  Reading
///   ^                                                                       |
///   +------------------------ release ---------------------------------------+
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Slot is available for CPU writes.
    Free,
    /// CPU is actively writing into this slot.
    Writing,
    /// CPU has finished writing; the data is ready for GPU consumption.
    Ready,
    /// GPU is currently reading / processing this slot.
    Reading,
}

impl SlotState {
    /// Returns `true` when the slot may transition to `Writing`.
    pub fn can_acquire(self) -> bool {
        matches!(self, Self::Free)
    }

    /// Returns `true` when the slot may transition to `Ready`.
    pub fn can_submit(self) -> bool {
        matches!(self, Self::Writing)
    }

    /// Returns `true` when the slot may transition to `Reading` or back to
    /// `Free`.
    pub fn can_release(self) -> bool {
        matches!(self, Self::Ready | Self::Reading)
    }
}

impl fmt::Display for SlotState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Free => write!(f, "Free"),
            Self::Writing => write!(f, "Writing"),
            Self::Ready => write!(f, "Ready"),
            Self::Reading => write!(f, "Reading"),
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer slot
// ---------------------------------------------------------------------------

/// A single staging buffer within the triple-buffer ring.
#[derive(Debug)]
pub struct BufferSlot {
    /// Current lifecycle state.
    state: SlotState,
    /// Logical number of bytes written by the CPU.
    size: usize,
    /// Allocated capacity of the backing `Vec<u8>`.
    capacity: usize,
    /// Raw byte storage.
    data: Vec<u8>,
    /// Monotonically increasing frame index at which this slot was last
    /// submitted. 0 means never-submitted.
    frame_index: u64,
}

impl BufferSlot {
    /// Create a new slot with the given capacity. The slot starts in `Free`
    /// state and the backing store is zeroed.
    pub fn new(capacity: usize) -> Self {
        Self {
            state: SlotState::Free,
            size: 0,
            capacity,
            data: vec![0u8; capacity],
            frame_index: 0,
        }
    }

    // -- Accessors ---------------------------------------------------------

    pub fn state(&self) -> SlotState {
        self.state
    }

    /// Number of bytes the CPU has written into this slot.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Total allocated capacity of the backing store.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// View the written payload as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.size]
    }

    /// Mutable view of the entire backing store (for CPU writes).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data[..self.capacity]
    }

    /// The frame index recorded at submission time.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    // -- State transitions -------------------------------------------------

    /// Try to transition from `Free` -> `Writing`.
    ///
    /// Returns `false` when the slot is not in `Free` state.
    pub fn acquire(&mut self) -> bool {
        if !self.state.can_acquire() {
            return false;
        }
        self.state = SlotState::Writing;
        self.size = 0;
        true
    }

    /// Transition from `Writing` -> `Ready`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the slot is not in `Writing` state or if
    /// `written_size` exceeds the slot capacity.
    pub fn submit(&mut self, written_size: usize, frame_index: u64) {
        debug_assert!(
            self.state.can_submit(),
            "submit() called on slot in {:?} state",
            self.state
        );
        debug_assert!(
            written_size <= self.capacity,
            "submit(): written_size {} exceeds capacity {}",
            written_size,
            self.capacity
        );
        self.size = written_size;
        self.frame_index = frame_index;
        self.state = SlotState::Ready;
    }

    /// Transition from `Ready` -> `Reading`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the slot cannot be released.
    pub fn start_read(&mut self) {
        debug_assert!(
            self.state.can_release(),
            "start_read() called on slot in {:?} state",
            self.state
        );
        self.state = SlotState::Reading;
    }

    /// Transition from `Reading` -> `Free`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the slot cannot be released.
    pub fn release(&mut self) {
        debug_assert!(
            self.state.can_release(),
            "release() called on slot in {:?} state",
            self.state
        );
        self.state = SlotState::Free;
        self.size = 0;
    }

    /// Grow the backing store to `new_capacity` if it is larger than the
    /// current capacity. Existing data is preserved up to the current size.
    pub fn resize(&mut self, new_capacity: usize) {
        if new_capacity > self.capacity {
            self.data.resize(new_capacity, 0u8);
            self.capacity = new_capacity;
        }
    }
}

// ---------------------------------------------------------------------------
// Staging-buffer descriptor
// ---------------------------------------------------------------------------

/// Descriptor used to allocate or describe a staged buffer.
#[derive(Debug, Clone, Copy)]
pub struct StagingBufferDesc {
    /// Requested size in bytes.
    pub size: usize,
    /// Alignment in bytes (0 means `MIN_GPU_ALIGNMENT`).
    pub alignment: usize,
}

impl StagingBufferDesc {
    /// Create a descriptor with default GPU alignment.
    pub const fn new(size: usize) -> Self {
        Self {
            size,
            alignment: MIN_GPU_ALIGNMENT,
        }
    }

    /// Create a descriptor with an explicit alignment.
    pub const fn with_alignment(size: usize, alignment: usize) -> Self {
        Self { size, alignment }
    }

    /// Return `size` rounded up to the next multiple of `alignment`.
    pub fn aligned_size(&self) -> usize {
        let align = if self.alignment == 0 {
            MIN_GPU_ALIGNMENT
        } else {
            self.alignment
        };
        ((self.size + align - 1) / align) * align
    }
}

impl Default for StagingBufferDesc {
    fn default() -> Self {
        Self {
            size: 0,
            alignment: MIN_GPU_ALIGNMENT,
        }
    }
}

// ---------------------------------------------------------------------------
// Result enums
// ---------------------------------------------------------------------------

/// Result of `BufferRegistry::acquire_staging`.
#[derive(Debug)]
pub enum AcquireResult {
    /// A staging slot was successfully acquired.
    Acquired {
        /// Index of the acquired slot (`0 .. NUM_STAGING_SLOTS`).
        slot_index: usize,
    },
    /// All slots are currently occupied by pending GPU work. The caller
    /// should wait for `release_staging` and retry.
    NoSlotAvailable,
}

/// Result of `BufferRegistry::submit_staging`.
#[derive(Debug)]
pub enum SubmitResult {
    /// The slot was successfully submitted for GPU consumption.
    Submitted,
    /// The slot index is out of range or the slot is not in `Writing` state.
    InvalidSlot,
}

/// Result of `BufferRegistry::release_staging`.
#[derive(Debug)]
pub enum ReleaseResult {
    /// The slot was released back to the free pool.
    Released,
    /// The slot index is out of range or the slot is not in
    /// `Ready`/`Reading` state.
    InvalidSlot,
}

// ---------------------------------------------------------------------------
// BufferRegistry
// ---------------------------------------------------------------------------

/// GPU-driven BufferRegistry with triple-buffered staging.
///
/// Manages three rotating staging slots such that the CPU can write frame N
/// while the GPU concurrently reads frame N-1. The guarantee:
///
/// - **No sync stalls** in the common case: the CPU always has at least one
///   free slot to write into.
/// - **Back-pressure** only triggers when all three slots are occupied by
///   pending GPU work, at which point the application should throttle.
///
/// # Typical frame loop
///
/// ```ignore
/// let mut reg = BufferRegistry::new(1 << 20);
///
/// // CPU frame
/// if let AcquireResult::Acquired { slot_index: idx } = reg.acquire_staging() {
///     let slot = reg.slot_mut(idx).unwrap();
///     // ... write data into slot.as_mut_slice() ...
///     reg.submit_staging(idx, written_bytes);
/// }
///
/// // GPU frame (may run concurrently with the next CPU frame)
/// if let Some(idx) = reg.acquire_reading() {
///     let slot = reg.slot(idx).unwrap();
///     // ... submit slot.as_slice() to GPU ...
///     // ... on GPU completion callback:
///     reg.release_staging(idx);
/// }
/// ```
pub struct BufferRegistry {
    /// The three staging slots.
    slots: [BufferSlot; NUM_STAGING_SLOTS],
    /// Round-robin pointer for the next `acquire_staging` probe.
    write_index: usize,
    /// Monotonically increasing frame counter.
    frame_count: u64,
}

impl BufferRegistry {
    /// Create a new `BufferRegistry` where every staging slot has at least
    /// `default_capacity` bytes of backing storage.
    ///
    /// # Panics
    ///
    /// Panics if `default_capacity == 0`.
    pub fn new(default_capacity: usize) -> Self {
        assert!(
            default_capacity > 0,
            "BufferRegistry requires a positive default capacity"
        );
        Self {
            slots: [
                BufferSlot::new(default_capacity),
                BufferSlot::new(default_capacity),
                BufferSlot::new(default_capacity),
            ],
            write_index: 0,
            frame_count: 0,
        }
    }

    // -- Accessors ---------------------------------------------------------

    /// Current frame count (incremented on every successful submit).
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Shared reference to a slot by index, or `None` if out of range.
    pub fn slot(&self, index: usize) -> Option<&BufferSlot> {
        self.slots.get(index)
    }

    /// Mutable reference to a slot by index, or `None`.
    ///
    /// # Safety
    ///
    /// The caller must ensure the GPU is not currently reading this slot
    /// before taking a mutable reference.
    pub fn slot_mut(&mut self, index: usize) -> Option<&mut BufferSlot> {
        self.slots.get_mut(index)
    }

    /// Number of slots currently in `Free` state (available for CPU writes).
    pub fn free_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state() == SlotState::Free)
            .count()
    }

    /// Number of slots currently in `Ready` state (available for GPU reads).
    pub fn ready_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state() == SlotState::Ready)
            .count()
    }

    /// Returns `true` when all three slots are occupied.
    pub fn is_stalled(&self) -> bool {
        self.free_slots() == 0
    }

    // -- Core operations ---------------------------------------------------

    /// Acquire a staging slot for CPU writes.
    ///
    /// Probes slots in round-robin order starting from the last-used index.
    /// Returns `AcquireResult::NoSlotAvailable` when all three slots are
    /// occupied.
    pub fn acquire_staging(&mut self) -> AcquireResult {
        for offset in 0..NUM_STAGING_SLOTS {
            let idx = (self.write_index + offset) % NUM_STAGING_SLOTS;
            if self.slots[idx].acquire() {
                self.write_index = (idx + 1) % NUM_STAGING_SLOTS;
                return AcquireResult::Acquired { slot_index: idx };
            }
        }
        AcquireResult::NoSlotAvailable
    }

    /// Mark a previously acquired staging slot as ready for GPU consumption.
    ///
    /// `written_size` is the number of bytes actually written by the CPU
    /// (may be less than the slot capacity). On success the internal frame
    /// counter is incremented.
    pub fn submit_staging(&mut self, slot_index: usize, written_size: usize) -> SubmitResult {
        let slot = match self.slots.get_mut(slot_index) {
            Some(s) => s,
            None => return SubmitResult::InvalidSlot,
        };
        if slot.state() != SlotState::Writing {
            return SubmitResult::InvalidSlot;
        }
        self.frame_count += 1;
        slot.submit(written_size, self.frame_count);
        SubmitResult::Submitted
    }

    /// Obtain the most recently submitted slot for GPU consumption.
    ///
    /// Scans all slots for the one in `Ready` state with the highest
    /// `frame_index`. Transitions it to `Reading` and returns its index.
    ///
    /// Returns `None` when no slot is ready.
    pub fn acquire_reading(&mut self) -> Option<usize> {
        let mut best: Option<(usize, u64)> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.state() == SlotState::Ready {
                let frame = slot.frame_index();
                match best {
                    None => best = Some((i, frame)),
                    Some((_, best_frame)) if frame > best_frame => {
                        best = Some((i, frame));
                    }
                    _ => {}
                }
            }
        }
        if let Some((idx, _)) = best {
            self.slots[idx].start_read();
            Some(idx)
        } else {
            None
        }
    }

    /// Release a staging slot back to the free pool after the GPU has
    /// finished reading it.
    pub fn release_staging(&mut self, slot_index: usize) -> ReleaseResult {
        let slot = match self.slots.get_mut(slot_index) {
            Some(s) => s,
            None => return ReleaseResult::InvalidSlot,
        };
        if !slot.state().can_release() {
            return ReleaseResult::InvalidSlot;
        }
        slot.release();
        ReleaseResult::Released
    }

    // -- Bulk / maintenance ------------------------------------------------

    /// Reset *all* slots to `Free` and zero the frame counter.
    ///
    /// Intended use: device-loss recovery, or a full pipeline flush.
    pub fn reset(&mut self) {
        for slot in self.slots.iter_mut() {
            slot.state = SlotState::Free;
            slot.size = 0;
            slot.frame_index = 0;
        }
        self.write_index = 0;
        self.frame_count = 0;
    }

    /// Ensure each `Free` slot has at least `min_capacity` bytes.
    ///
    /// Slots in non-`Free` states are skipped and will be sized on their
    /// next cycle through `Free`.
    pub fn ensure_capacity(&mut self, min_capacity: usize) {
        for slot in self.slots.iter_mut() {
            if slot.state() == SlotState::Free && slot.capacity() < min_capacity {
                slot.resize(min_capacity);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl fmt::Display for BufferRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BufferRegistry(frame={}, free={}, ready={}, stalled={}, slots=[",
            self.frame_count,
            self.free_slots(),
            self.ready_slots(),
            self.is_stalled(),
        )?;
        for (i, slot) in self.slots.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(
                f,
                "[{} size={}/{} frame={}]",
                slot.state(),
                slot.size(),
                slot.capacity(),
                slot.frame_index(),
            )?;
        }
        write!(f, "])")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers -----------------------------------------------------------

    fn acquire_one(reg: &mut BufferRegistry) -> usize {
        match reg.acquire_staging() {
            AcquireResult::Acquired { slot_index } => slot_index,
            _ => panic!("expected Acquired"),
        }
    }

    // -- Registry lifecycle ------------------------------------------------

    #[test]
    fn test_new_registry_all_free() {
        let reg = BufferRegistry::new(4096);
        assert_eq!(reg.free_slots(), 3);
        assert_eq!(reg.ready_slots(), 0);
        assert!(!reg.is_stalled());
        assert_eq!(reg.frame_count(), 0);
    }

    #[test]
    fn test_acquire_submit_cycle() {
        let mut reg = BufferRegistry::new(256);

        let idx = acquire_one(&mut reg);
        assert_eq!(idx, 0);
        assert_eq!(reg.free_slots(), 2);

        // Write a little data.
        let slot = reg.slot_mut(idx).unwrap();
        slot.as_mut_slice()[..4].copy_from_slice(&[1u8, 2, 3, 4]);

        assert!(matches!(
            reg.submit_staging(idx, 4),
            SubmitResult::Submitted
        ));
        assert_eq!(reg.ready_slots(), 1);
        assert_eq!(reg.frame_count(), 1);
    }

    // -- Triple-buffer rotation -------------------------------------------

    #[test]
    fn test_triple_buffer_acquire_all() {
        let mut reg = BufferRegistry::new(64);

        let _ = acquire_one(&mut reg);
        let _ = acquire_one(&mut reg);
        let _ = acquire_one(&mut reg);
        assert!(matches!(reg.acquire_staging(), AcquireResult::NoSlotAvailable));
        assert!(reg.is_stalled());
    }

    #[test]
    fn test_acquire_reading_returns_newest() {
        let mut reg = BufferRegistry::new(64);

        let s0 = acquire_one(&mut reg);
        let s1 = acquire_one(&mut reg);

        assert!(matches!(reg.submit_staging(s0, 8), SubmitResult::Submitted));
        assert!(matches!(reg.submit_staging(s1, 8), SubmitResult::Submitted));

        // acquire_reading must return the slot with the higher frame_index.
        let read = reg.acquire_reading().unwrap();
        assert_eq!(read, s1);
    }

    #[test]
    fn test_full_rotate() {
        let mut reg = BufferRegistry::new(64);

        // Fill, submit, consume, release -- full cycle.
        let a0 = acquire_one(&mut reg);
        let a1 = acquire_one(&mut reg);
        let a2 = acquire_one(&mut reg);

        assert!(matches!(reg.submit_staging(a0, 8), SubmitResult::Submitted));
        assert!(matches!(reg.submit_staging(a1, 8), SubmitResult::Submitted));
        assert!(matches!(reg.submit_staging(a2, 8), SubmitResult::Submitted));

        let read = reg.acquire_reading().unwrap();
        assert_eq!(read, a2);
        assert_eq!(reg.slot(read).unwrap().state(), SlotState::Reading);

        assert!(matches!(reg.release_staging(read), ReleaseResult::Released));
        assert_eq!(reg.free_slots(), 1);

        // The recycled slot should be a2.
        let recycled = acquire_one(&mut reg);
        assert_eq!(recycled, a2);
    }

    // -- Error paths -------------------------------------------------------

    #[test]
    fn test_submit_unacquired_slot_fails() {
        let mut reg = BufferRegistry::new(64);
        assert!(matches!(
            reg.submit_staging(0, 4),
            SubmitResult::InvalidSlot
        ));
    }

    #[test]
    fn test_release_unacquired_slot_fails() {
        let mut reg = BufferRegistry::new(64);
        assert!(matches!(
            reg.release_staging(0),
            ReleaseResult::InvalidSlot
        ));
    }

    #[test]
    fn test_double_acquire_fails() {
        let mut reg = BufferRegistry::new(64);
        let idx = acquire_one(&mut reg);
        // Second acquire on the same logical slot (round-robin reaches a
        // different index) -- should still work, but explicitly acquiring a
        // specific busy slot is not possible via the public API.
        // Verify that the slot itself rejects a second acquire.
        assert!(!reg.slot_mut(idx).unwrap().acquire());
    }

    // -- Maintenance -------------------------------------------------------

    #[test]
    fn test_reset() {
        let mut reg = BufferRegistry::new(1024);
        let idx = acquire_one(&mut reg);
        assert!(matches!(reg.submit_staging(idx, 16), SubmitResult::Submitted));

        reg.reset();
        assert_eq!(reg.free_slots(), 3);
        assert_eq!(reg.frame_count(), 0);
    }

    #[test]
    fn test_ensure_capacity_only_resizes_free_slots() {
        let mut reg = BufferRegistry::new(128);
        // Put one slot into Writing so resize skips it.
        let idx = acquire_one(&mut reg);

        reg.ensure_capacity(1024);

        // The two Free slots should be >= 1024.
        let mut seen_free = 0usize;
        let mut seen_busy = 0usize;
        for (i, slot) in reg.slots.iter().enumerate() {
            if i == idx {
                assert_eq!(slot.capacity(), 128);
                seen_busy += 1;
            } else {
                assert!(slot.capacity() >= 1024);
                seen_free += 1;
            }
        }
        assert_eq!(seen_free, 2);
        assert_eq!(seen_busy, 1);
    }

    // -- StagingBufferDesc ------------------------------------------------

    #[test]
    fn test_staging_buffer_desc_alignment() {
        let desc = StagingBufferDesc::new(100);
        // Rounds up to MIN_GPU_ALIGNMENT (256).
        assert_eq!(desc.aligned_size(), 256);

        let desc = StagingBufferDesc::with_alignment(100, 64);
        assert_eq!(desc.aligned_size(), 128);
    }

    // -- Display -----------------------------------------------------------

    #[test]
    fn test_display() {
        let reg = BufferRegistry::new(64);
        let s = format!("{}", reg);
        assert!(s.contains("free=3"));
        assert!(s.contains("stalled=false"));
        assert!(s.starts_with("BufferRegistry("));
    }

    // -- Slot-level unit tests ---------------------------------------------

    #[test]
    fn test_slot_state_transitions() {
        let mut slot = BufferSlot::new(128);
        assert_eq!(slot.state(), SlotState::Free);

        assert!(slot.acquire());
        assert_eq!(slot.state(), SlotState::Writing);

        slot.submit(64, 1);
        assert_eq!(slot.state(), SlotState::Ready);
        assert_eq!(slot.size(), 64);
        assert_eq!(slot.frame_index(), 1);

        slot.start_read();
        assert_eq!(slot.state(), SlotState::Reading);

        slot.release();
        assert_eq!(slot.state(), SlotState::Free);
        assert_eq!(slot.size(), 0);
    }

    #[test]
    fn test_buffer_slot_resize() {
        let mut slot = BufferSlot::new(128);
        assert_eq!(slot.capacity(), 128);

        slot.resize(256);
        assert_eq!(slot.capacity(), 256);

        // Shrink is a no-op.
        slot.resize(64);
        assert_eq!(slot.capacity(), 256);
    }

    // ===================================================================
    // Slot state machine lifecycle (9 additional tests)
    // ===================================================================

    #[test]
    fn test_slot_acquire_rejected_on_non_free() {
        let mut slot = BufferSlot::new(128);
        assert!(slot.acquire()); // Free -> Writing
        assert!(!slot.acquire()); // Writing -> rejected
        slot.submit(8, 1);
        assert!(!slot.acquire()); // Ready -> rejected
        slot.start_read();
        assert!(!slot.acquire()); // Reading -> rejected
    }

    #[test]
    fn test_slot_release_guard_via_can_release() {
        let mut slot = BufferSlot::new(64);
        // Free is not releasable.
        assert!(!slot.state().can_release());
        // Writing is not releasable.
        slot.acquire();
        assert!(!slot.state().can_release());
        // Ready is releasable.
        slot.submit(16, 1);
        assert!(slot.state().can_release());
        // Reading is releasable.
        slot.start_read();
        assert!(slot.state().can_release());
        // Back to Free is not releasable.
        slot.release();
        assert!(!slot.state().can_release());
    }

    #[test]
    fn test_slot_can_submit_guards_correctly() {
        let mut slot = BufferSlot::new(64);
        assert!(!slot.state().can_submit());
        slot.acquire();
        assert!(slot.state().can_submit());
        slot.submit(8, 1);
        assert!(!slot.state().can_submit());
        slot.start_read();
        assert!(!slot.state().can_submit());
        slot.release();
        assert!(!slot.state().can_submit());
    }

    #[test]
    fn test_slot_release_from_ready_skips_reading() {
        // A Ready slot can be released directly to Free without going through
        // Reading first (e.g. discarded work).
        let mut slot = BufferSlot::new(64);
        slot.acquire();
        slot.submit(32, 1);
        assert_eq!(slot.state(), SlotState::Ready);

        slot.release();
        assert_eq!(slot.state(), SlotState::Free);
        assert_eq!(slot.size(), 0);
    }

    #[test]
    fn test_slot_full_lifecycle_repeat() {
        let mut slot = BufferSlot::new(128);
        for cycle in 0..5 {
            assert!(slot.acquire());
            assert_eq!(slot.state(), SlotState::Writing);

            slot.as_mut_slice()[0] = (cycle + 10) as u8;
            slot.submit(1, cycle as u64 + 1);
            assert_eq!(slot.state(), SlotState::Ready);

            slot.start_read();
            assert_eq!(slot.state(), SlotState::Reading);

            slot.release();
            assert_eq!(slot.state(), SlotState::Free);
            assert_eq!(slot.size(), 0);
        }
    }

    #[test]
    fn test_slot_submit_metadata() {
        let mut slot = BufferSlot::new(256);
        slot.acquire();
        slot.as_mut_slice()[..4].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        slot.submit(4, 42);
        assert_eq!(slot.size(), 4);
        assert_eq!(slot.frame_index(), 42);
        assert_eq!(&slot.as_slice(), &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn test_slot_data_integrity_after_submit() {
        let mut slot = BufferSlot::new(1024);
        slot.as_mut_slice()[..8].copy_from_slice(&[1u8, 2, 3, 4, 5, 6, 7, 8]);
        slot.acquire();
        // Write partial data.
        slot.as_mut_slice()[..6].copy_from_slice(&[10, 20, 30, 40, 50, 60]);
        slot.submit(6, 1);
        assert_eq!(slot.as_slice(), &[10, 20, 30, 40, 50, 60]);

        // After release, data is "gone" (size=0) but backing store remains.
        slot.start_read();
        slot.release();
        assert_eq!(slot.size(), 0);
        assert_eq!(slot.capacity(), 1024);
    }

    #[test]
    fn test_slot_new_slot_zeroed() {
        let mut slot = BufferSlot::new(256);
        assert_eq!(slot.as_mut_slice(), &[0u8; 256]);
    }

    #[test]
    fn test_slot_preserve_data_on_resize() {
        let mut slot = BufferSlot::new(128);
        slot.acquire();
        slot.as_mut_slice()[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        slot.submit(8, 1);

        // Resize larger preserves the data.
        slot.resize(256);
        assert_eq!(slot.capacity(), 256);
        assert_eq!(&slot.as_slice()[..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    // ===================================================================
    // Round-robin (3 tests)
    // ===================================================================

    #[test]
    fn test_round_robin_wraparound() {
        let mut reg = BufferRegistry::new(64);
        // Acquire all three; round-robin pointer should cycle through 0,1,2.
        let s0 = acquire_one(&mut reg);
        assert_eq!(s0, 0);
        let s1 = acquire_one(&mut reg);
        assert_eq!(s1, 1);
        let s2 = acquire_one(&mut reg);
        assert_eq!(s2, 2);
        assert!(matches!(reg.acquire_staging(), AcquireResult::NoSlotAvailable));

        // Submit slot 2, then release so next acquire wraps.
        assert!(matches!(reg.submit_staging(2, 4), SubmitResult::Submitted));
        assert!(matches!(reg.release_staging(2), ReleaseResult::Released));
        let s_next = acquire_one(&mut reg);
        assert_eq!(s_next, 2);
    }

    #[test]
    fn test_round_robin_skips_busy_slots() {
        let mut reg = BufferRegistry::new(64);
        // Take slot 0, leave 1 and 2 free.
        let s0 = acquire_one(&mut reg); // slot 0
        assert_eq!(s0, 0);

        // Submit 0 so it's no longer Writing (just to mix states).
        assert!(matches!(reg.submit_staging(0, 8), SubmitResult::Submitted));

        // Next acquire: starts at 1 -> slot 1.
        let s1 = acquire_one(&mut reg);
        assert_eq!(s1, 1);

        // Release slot 0 (Ready) and slot 1 (need to submit first).
        assert!(matches!(reg.release_staging(0), ReleaseResult::Released));
        assert!(matches!(reg.submit_staging(1, 8), SubmitResult::Submitted));
        assert!(matches!(reg.release_staging(1), ReleaseResult::Released));

        // Acquire: pointer at 2 -> slot 2 (free)
        let s2 = acquire_one(&mut reg);
        assert_eq!(s2, 2);
    }

    #[test]
    fn test_round_robin_mixed_ops_pointer_advances() {
        let mut reg = BufferRegistry::new(64);
        // Take slot 0, submit, release.
        let _s0 = acquire_one(&mut reg); // slot 0, pointer -> 1
        assert!(matches!(reg.submit_staging(0, 4), SubmitResult::Submitted));
        assert!(matches!(reg.release_staging(0), ReleaseResult::Released));

        // Take slot 1, submit, release.
        let _s1 = acquire_one(&mut reg); // slot 1, pointer -> 2
        assert!(matches!(reg.submit_staging(1, 4), SubmitResult::Submitted));
        assert!(matches!(reg.release_staging(1), ReleaseResult::Released));

        // Take slot 2, submit, release.
        let _s2 = acquire_one(&mut reg); // slot 2, pointer -> 0
        assert!(matches!(reg.submit_staging(2, 4), SubmitResult::Submitted));
        assert!(matches!(reg.release_staging(2), ReleaseResult::Released));

        // Next acquire wraps to slot 0.
        let s0_again = acquire_one(&mut reg);
        assert_eq!(s0_again, 0);
    }

    // ===================================================================
    // Back-pressure (4 tests)
    // ===================================================================

    #[test]
    fn test_back_pressure_acquire_all_no_submit() {
        let mut reg = BufferRegistry::new(64);
        let _ = acquire_one(&mut reg);
        let _ = acquire_one(&mut reg);
        let _ = acquire_one(&mut reg);
        assert!(matches!(reg.acquire_staging(), AcquireResult::NoSlotAvailable));
        assert_eq!(reg.free_slots(), 0);
    }

    #[test]
    fn test_back_pressure_clears_on_release() {
        let mut reg = BufferRegistry::new(64);
        let _s0 = acquire_one(&mut reg);
        let _s1 = acquire_one(&mut reg);
        let s2 = acquire_one(&mut reg);
        assert!(reg.is_stalled());

        // Submit and release one slot.
        assert!(matches!(reg.submit_staging(s2, 8), SubmitResult::Submitted));
        assert!(matches!(reg.release_staging(s2), ReleaseResult::Released));

        assert!(!reg.is_stalled());
        assert_eq!(reg.free_slots(), 1);
    }

    #[test]
    fn test_back_pressure_submit_does_not_free() {
        let mut reg = BufferRegistry::new(64);
        let s0 = acquire_one(&mut reg);
        let s1 = acquire_one(&mut reg);
        let s2 = acquire_one(&mut reg);

        // Submit all three -- they become Ready, not Free.
        assert!(matches!(reg.submit_staging(s0, 4), SubmitResult::Submitted));
        assert!(matches!(reg.submit_staging(s1, 4), SubmitResult::Submitted));
        assert!(matches!(reg.submit_staging(s2, 4), SubmitResult::Submitted));

        // Still stalled because Free == 0.
        assert_eq!(reg.free_slots(), 0);
        assert!(reg.is_stalled());
    }

    #[test]
    fn test_back_pressure_full_cycle() {
        let mut reg = BufferRegistry::new(64);
        // Acquire all 3.
        let slots: Vec<_> = (0..3).map(|_| acquire_one(&mut reg)).collect();
        assert!(reg.is_stalled());

        // Submit all 3.
        for &s in &slots {
            assert!(matches!(reg.submit_staging(s, 8), SubmitResult::Submitted));
        }

        // Read newest.
        let read = reg.acquire_reading().unwrap();
        assert!(matches!(reg.release_staging(read), ReleaseResult::Released));

        // Now one slot is free and stalling should be resolved.
        assert!(!reg.is_stalled());
        assert_eq!(reg.free_slots(), 1);
    }

    // ===================================================================
    // Newest-slot reading (4 tests)
    // ===================================================================

    #[test]
    fn test_acquire_reading_none_when_no_ready() {
        let mut reg = BufferRegistry::new(64);
        // All Free -- nothing to read.
        assert!(reg.acquire_reading().is_none());

        // Slot acquired but not submitted -- still nothing to read.
        let s0 = acquire_one(&mut reg);
        assert!(reg.acquire_reading().is_none());

        // After release, still no Ready slots.
        assert!(matches!(reg.release_staging(s0), ReleaseResult::InvalidSlot));
    }

    #[test]
    fn test_acquire_reading_submit_out_of_order() {
        let mut reg = BufferRegistry::new(64);
        let s0 = acquire_one(&mut reg);
        let s1 = acquire_one(&mut reg);

        // Submit in reverse order: slot 1 first, then slot 0.
        assert!(matches!(reg.submit_staging(s1, 8), SubmitResult::Submitted));
        assert!(matches!(reg.submit_staging(s0, 8), SubmitResult::Submitted));

        // frame_count is 2 for slot 0 (submitted last), 1 for slot 1.
        // acquire_reading should return the highest frame_index = slot 0.
        let read = reg.acquire_reading().unwrap();
        assert_eq!(read, s0);
        assert_eq!(reg.slot(s0).unwrap().state(), SlotState::Reading);
    }

    #[test]
    fn test_acquire_reading_consumes_all_sequentially() {
        let mut reg = BufferRegistry::new(64);
        let s0 = acquire_one(&mut reg);
        let s1 = acquire_one(&mut reg);
        let s2 = acquire_one(&mut reg);

        assert!(matches!(reg.submit_staging(s0, 8), SubmitResult::Submitted));
        assert!(matches!(reg.submit_staging(s1, 8), SubmitResult::Submitted));
        assert!(matches!(reg.submit_staging(s2, 8), SubmitResult::Submitted));

        // First read: newest = s2 (frame 3).
        let r0 = reg.acquire_reading().unwrap();
        assert_eq!(r0, s2);

        // Second read: newest = s1 (frame 2).
        let r1 = reg.acquire_reading().unwrap();
        assert_eq!(r1, s1);

        // Third read: newest = s0 (frame 1).
        let r2 = reg.acquire_reading().unwrap();
        assert_eq!(r2, s0);

        // No more Ready slots.
        assert!(reg.acquire_reading().is_none());
    }

    #[test]
    fn test_acquire_reading_with_release() {
        let mut reg = BufferRegistry::new(64);
        let s0 = acquire_one(&mut reg);
        assert!(matches!(reg.submit_staging(s0, 8), SubmitResult::Submitted));

        let read = reg.acquire_reading().unwrap();
        assert_eq!(read, s0);
        assert_eq!(reg.slot(s0).unwrap().state(), SlotState::Reading);

        // Release the read slot.
        assert!(matches!(reg.release_staging(read), ReleaseResult::Released));
        assert_eq!(reg.slot(s0).unwrap().state(), SlotState::Free);
    }

    // ===================================================================
    // Reset / recovery (3 tests)
    // ===================================================================

    #[test]
    fn test_reset_from_mixed_states() {
        let mut reg = BufferRegistry::new(64);
        let _s0 = acquire_one(&mut reg); // Writing
        let s1 = acquire_one(&mut reg); // Writing
        assert!(matches!(reg.submit_staging(s1, 8), SubmitResult::Submitted)); // Ready
        let _s2 = acquire_one(&mut reg); // Writing

        reg.reset();
        assert_eq!(reg.free_slots(), 3);
        assert_eq!(reg.frame_count(), 0);
        for i in 0..3 {
            assert_eq!(reg.slot(i).unwrap().state(), SlotState::Free);
            assert_eq!(reg.slot(i).unwrap().frame_index(), 0);
            assert_eq!(reg.slot(i).unwrap().size(), 0);
        }
    }

    #[test]
    fn test_reset_and_resume_operation() {
        let mut reg = BufferRegistry::new(64);
        let s0 = acquire_one(&mut reg);
        assert!(matches!(reg.submit_staging(s0, 8), SubmitResult::Submitted));
        reg.reset();

        // Resume: should work as if freshly created.
        let s = acquire_one(&mut reg);
        assert_eq!(s, 0);
        assert!(matches!(reg.submit_staging(s, 4), SubmitResult::Submitted));
        assert_eq!(reg.frame_count(), 1);
        let read = reg.acquire_reading().unwrap();
        assert_eq!(read, s);
    }

    #[test]
    fn test_reset_zeroes_write_index() {
        let mut reg = BufferRegistry::new(64);
        // Acquire and release slots so write_index ends up at 2.
        let s0 = acquire_one(&mut reg); // write_index -> 1
        let s1 = acquire_one(&mut reg); // write_index -> 2
        assert!(matches!(reg.submit_staging(s0, 4), SubmitResult::Submitted));
        assert!(matches!(reg.submit_staging(s1, 4), SubmitResult::Submitted));
        assert!(matches!(reg.release_staging(s0), ReleaseResult::Released));
        assert!(matches!(reg.release_staging(s1), ReleaseResult::Released));
        let s2 = acquire_one(&mut reg); // write_index -> 0
        assert_eq!(s2, 2);
        assert!(matches!(reg.submit_staging(s2, 4), SubmitResult::Submitted));
        assert!(matches!(reg.release_staging(s2), ReleaseResult::Released));

        // write_index is now at 0.
        let check = acquire_one(&mut reg);
        assert_eq!(check, 0);

        reg.reset();
        // After reset, write_index should be 0.
        let after = acquire_one(&mut reg);
        assert_eq!(after, 0);
    }

    // ===================================================================
    // Capacity growth (5 tests)
    // ===================================================================

    #[test]
    fn test_ensure_capacity_all_busy_no_resize() {
        let mut reg = BufferRegistry::new(64);
        let _ = acquire_one(&mut reg);
        let _ = acquire_one(&mut reg);
        let _ = acquire_one(&mut reg);
        // All three are Writing -- no Free slots to resize.
        reg.ensure_capacity(4096);
        for i in 0..3 {
            assert_eq!(reg.slot(i).unwrap().capacity(), 64);
        }
    }

    #[test]
    fn test_ensure_capacity_exact_no_change() {
        let mut reg = BufferRegistry::new(1024);
        // All are Free with exactly 1024 capacity.
        reg.ensure_capacity(1024);
        for i in 0..3 {
            assert_eq!(reg.slot(i).unwrap().capacity(), 1024);
        }
    }

    #[test]
    fn test_ensure_capacity_multiple_growth() {
        let mut reg = BufferRegistry::new(128);
        reg.ensure_capacity(256);
        for i in 0..3 {
            assert_eq!(reg.slot(i).unwrap().capacity(), 256);
        }
        reg.ensure_capacity(512);
        for i in 0..3 {
            assert_eq!(reg.slot(i).unwrap().capacity(), 512);
        }
    }

    #[test]
    fn test_ensure_capacity_never_shrinks() {
        let mut reg = BufferRegistry::new(1024);
        reg.ensure_capacity(128); // smaller than current
        for i in 0..3 {
            assert_eq!(reg.slot(i).unwrap().capacity(), 1024);
        }
    }

    #[test]
    fn test_ensure_capacity_skips_busy_slot() {
        let mut reg = BufferRegistry::new(128);
        let _idx = acquire_one(&mut reg); // slot 0 -> Writing

        reg.ensure_capacity(512);
        // Slot 0 should still have 128 (skipped), others should be >= 512.
        assert_eq!(reg.slot(0).unwrap().capacity(), 128);
        assert!(reg.slot(1).unwrap().capacity() >= 512);
        assert!(reg.slot(2).unwrap().capacity() >= 512);
    }

    // ===================================================================
    // Alignment (5 tests)
    // ===================================================================

    #[test]
    fn test_staging_buffer_desc_zero_alignment() {
        // Zero alignment should default to MIN_GPU_ALIGNMENT.
        let desc = StagingBufferDesc {
            size: 100,
            alignment: 0,
        };
        assert_eq!(desc.aligned_size(), MIN_GPU_ALIGNMENT);
    }

    #[test]
    fn test_staging_buffer_desc_exact_multiple() {
        let desc = StagingBufferDesc::with_alignment(256, 256);
        assert_eq!(desc.aligned_size(), 256);
    }

    #[test]
    fn test_staging_buffer_desc_already_aligned() {
        let desc = StagingBufferDesc::with_alignment(512, 256);
        assert_eq!(desc.aligned_size(), 512);
    }

    #[test]
    fn test_staging_buffer_desc_large_alignment() {
        let desc = StagingBufferDesc::with_alignment(1, 1024);
        assert_eq!(desc.aligned_size(), 1024);
    }

    #[test]
    fn test_staging_buffer_desc_power_of_two_boundaries() {
        // Test values just below, at, and above alignment boundaries.
        for align in [64usize, 128, 256, 512] {
            let desc_just_below = StagingBufferDesc::with_alignment(align - 1, align);
            assert_eq!(desc_just_below.aligned_size(), align);

            let desc_exact = StagingBufferDesc::with_alignment(align, align);
            assert_eq!(desc_exact.aligned_size(), align);

            let desc_just_above = StagingBufferDesc::with_alignment(align + 1, align);
            assert_eq!(desc_just_above.aligned_size(), align * 2);
        }
    }

    // ===================================================================
    // Frame counter (3 tests)
    // ===================================================================

    #[test]
    fn test_frame_counter_monotonic() {
        let mut reg = BufferRegistry::new(8192);
        let mut last_frame = 0u64;
        for _ in 0..50 {
            // Acquire, submit, read, release to keep slots recycling.
            if let AcquireResult::Acquired { slot_index: idx } = reg.acquire_staging() {
                assert!(matches!(reg.submit_staging(idx, 4), SubmitResult::Submitted));
                assert!(reg.frame_count() > last_frame);
                last_frame = reg.frame_count();
            }
            // Consume the newly submitted slot to avoid stalling.
            if let Some(read) = reg.acquire_reading() {
                assert!(matches!(reg.release_staging(read), ReleaseResult::Released));
            }
        }
        assert!(reg.frame_count() >= 50);
    }

    #[test]
    fn test_frame_counter_not_incremented_failed() {
        let mut reg = BufferRegistry::new(64);
        // Submit to an out-of-range slot does not increment.
        assert!(matches!(reg.submit_staging(99, 4), SubmitResult::InvalidSlot));
        assert_eq!(reg.frame_count(), 0);
    }

    #[test]
    fn test_frame_counter_not_incremented_other() {
        let mut reg = BufferRegistry::new(64);
        let _ = acquire_one(&mut reg);
        // acquire alone does not increment.
        assert_eq!(reg.frame_count(), 0);

        let s0 = acquire_one(&mut reg);
        assert_eq!(reg.frame_count(), 0);

        // release alone does not increment.
        assert!(matches!(reg.submit_staging(s0, 8), SubmitResult::Submitted));
        assert_eq!(reg.frame_count(), 1);

        let read = reg.acquire_reading().unwrap();
        assert_eq!(reg.frame_count(), 1);

        assert!(matches!(reg.release_staging(read), ReleaseResult::Released));
        assert_eq!(reg.frame_count(), 1);
    }

    // ===================================================================
    // Slot isolation (1 test)
    // ===================================================================

    #[test]
    fn test_slot_isolation() {
        let mut reg = BufferRegistry::new(64);
        let s0 = acquire_one(&mut reg);
        let s1 = acquire_one(&mut reg);

        // Write distinct patterns to each slot.
        reg.slot_mut(s0).unwrap().as_mut_slice()[..2].copy_from_slice(&[0xAA, 0xBB]);
        reg.slot_mut(s1).unwrap().as_mut_slice()[..2].copy_from_slice(&[0xCC, 0xDD]);

        // Submit with correct written_size so as_slice() matches exactly.
        assert!(matches!(reg.submit_staging(s0, 2), SubmitResult::Submitted));
        assert!(matches!(reg.submit_staging(s1, 2), SubmitResult::Submitted));

        // Verify each slot's data is distinct and not corrupted.
        assert_eq!(reg.slot(s0).unwrap().as_slice(), &[0xAA, 0xBB]);
        assert_eq!(reg.slot(s1).unwrap().as_slice(), &[0xCC, 0xDD]);
    }

    // ===================================================================
    // Out-of-range / edge cases (6 tests)
    // ===================================================================

    #[test]
    fn test_slot_access_out_of_range() {
        let reg = BufferRegistry::new(64);
        assert!(reg.slot(3).is_none());
        assert!(reg.slot(100).is_none());
        assert!(reg.slot(usize::MAX).is_none());
    }

    #[test]
    fn test_slot_mut_out_of_range() {
        let mut reg = BufferRegistry::new(64);
        assert!(reg.slot_mut(3).is_none());
        assert!(reg.slot_mut(100).is_none());
    }

    #[test]
    fn test_submit_staging_out_of_range() {
        let mut reg = BufferRegistry::new(64);
        assert!(matches!(
            reg.submit_staging(3, 4),
            SubmitResult::InvalidSlot
        ));
        assert!(matches!(
            reg.submit_staging(100, 4),
            SubmitResult::InvalidSlot
        ));
    }

    #[test]
    fn test_release_staging_out_of_range() {
        let mut reg = BufferRegistry::new(64);
        assert!(matches!(
            reg.release_staging(3),
            ReleaseResult::InvalidSlot
        ));
        assert!(matches!(
            reg.release_staging(100),
            ReleaseResult::InvalidSlot
        ));
    }

    #[test]
    fn test_submit_staging_zero_size() {
        let mut reg = BufferRegistry::new(64);
        let idx = acquire_one(&mut reg);
        // Submitting with zero bytes is valid.
        assert!(matches!(
            reg.submit_staging(idx, 0),
            SubmitResult::Submitted
        ));
        assert_eq!(reg.slot(idx).unwrap().size(), 0);
        assert!(reg.slot(idx).unwrap().as_slice().is_empty());
    }

    #[test]
    fn test_submit_staging_full_capacity() {
        let mut reg = BufferRegistry::new(64);
        let idx = acquire_one(&mut reg);
        // Fill the entire slot.
        let slot = reg.slot_mut(idx).unwrap();
        slot.as_mut_slice().fill(0xFF);
        assert!(matches!(
            reg.submit_staging(idx, 64),
            SubmitResult::Submitted
        ));
        assert_eq!(reg.slot(idx).unwrap().size(), 64);
        assert_eq!(reg.slot(idx).unwrap().as_slice(), &[0xFFu8; 64]);
    }

    // ===================================================================
    // Release guard (2 tests)
    // ===================================================================

    #[test]
    fn test_release_free_slot_via_registry() {
        let mut reg = BufferRegistry::new(64);
        // Slot 0 is Free -- release should fail.
        assert!(matches!(
            reg.release_staging(0),
            ReleaseResult::InvalidSlot
        ));
    }

    #[test]
    fn test_release_writing_slot_via_registry() {
        let mut reg = BufferRegistry::new(64);
        let idx = acquire_one(&mut reg);
        // Slot is Writing -- release should fail.
        assert!(matches!(
            reg.release_staging(idx),
            ReleaseResult::InvalidSlot
        ));
    }

    // ===================================================================
    // Repeated cycles (1 test)
    // ===================================================================

    #[test]
    fn test_acquire_release_reacquire_cycles() {
        let mut reg = BufferRegistry::new(64);
        for cycle in 0..10 {
            let idx = acquire_one(&mut reg);
            assert!(matches!(reg.submit_staging(idx, 4), SubmitResult::Submitted));
            let read = reg.acquire_reading().unwrap();
            assert!(matches!(reg.release_staging(read), ReleaseResult::Released));
            assert_eq!(reg.frame_count(), cycle as u64 + 1);
        }
    }

    // ===================================================================
    // Zero-capacity constructor (1 test)
    // ===================================================================

    #[test]
    #[should_panic(expected = "BufferRegistry requires a positive default capacity")]
    fn test_new_registry_zero_capacity_panics() {
        let _reg = BufferRegistry::new(0);
    }

    // ===================================================================
    // Display (2 additional tests)
    // ===================================================================

    #[test]
    fn test_display_mixed_states() {
        let mut reg = BufferRegistry::new(64);
        let _s0 = acquire_one(&mut reg);
        let s1 = acquire_one(&mut reg);
        assert!(matches!(reg.submit_staging(s1, 8), SubmitResult::Submitted));

        let s = format!("{}", reg);
        assert!(s.contains("free=1")); // slot 2 is Free
        assert!(s.contains("ready=1")); // slot 1 is Ready
    }

    #[test]
    fn test_display_stalled() {
        let mut reg = BufferRegistry::new(64);
        let _ = acquire_one(&mut reg);
        let _ = acquire_one(&mut reg);
        let _ = acquire_one(&mut reg);

        let s = format!("{}", reg);
        assert!(s.contains("free=0"));
        assert!(s.contains("stalled=true"));
    }
}
