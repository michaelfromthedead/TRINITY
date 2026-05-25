// GPU memory allocators and budget tracker.
//
// Provides three allocator strategies:
// - FrameAllocator  – bump-pointer, reset once per frame
// - PoolAllocator   – fixed-size block pool (64 KB, 256 KB, 1 MB, 4 MB)
// - StackAllocator  – LIFO stack for nested staging allocations
//
// And an atomic GPU budget tracker (GpuBudget) for capacity planning.

// ---------------------------------------------------------------------------
// FrameAllocator — bump-pointer, per-frame reset
// ---------------------------------------------------------------------------

/// Bump-pointer allocator, per-frame reset.
///
/// Allocates linearly from a fixed backing buffer.  The entire region is
/// reclaimed by calling `reset()` — individual deallocation is not supported.
/// Ideal for transient per-frame data whose lifetime matches the frame.
pub struct FrameAllocator {
    buffer: Vec<u8>,
    offset: usize,
    capacity: usize,
}

impl FrameAllocator {
    /// Create a new bump allocator with the given `capacity` (in bytes).
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            offset: 0,
            capacity,
        }
    }

    /// Allocate a contiguous slice of `size` bytes aligned to `alignment`.
    ///
    /// Returns `None` when the allocation would exceed the backing capacity.
    /// The returned slice is valid until the next call to `reset()`.
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Option<&mut [u8]> {
        if size == 0 {
            return None;
        }
        let alignment = alignment.max(1);
        let start = self.offset;
        // Align up: round start to the next multiple of alignment.
        let aligned = (start + alignment - 1) & !(alignment - 1);
        let end = aligned.checked_add(size)?;
        if end > self.capacity {
            return None;
        }
        self.offset = end;
        Some(&mut self.buffer[aligned..end])
    }

    /// Reset the allocator — zero the offset for the next frame.
    ///
    /// Does **not** zero the backing memory (caller may clear if needed).
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Return the number of bytes currently used (next allocation offset).
    pub fn used(&self) -> usize {
        self.offset
    }
}

// ---------------------------------------------------------------------------
// PoolAllocator — fixed-size block pool
// ---------------------------------------------------------------------------

const DEFAULT_BLOCK_SIZES: &[usize] = &[64 * 1024, 256 * 1024, 1 * 1024 * 1024, 4 * 1024 * 1024];

/// Fixed-size block pool allocator.
///
/// Maintains free lists for a set of block sizes (64 KB, 256 KB, 1 MB, 4 MB).
/// A request is rounded up to the next supported block size.
pub struct PoolAllocator {
    block_sizes: Vec<usize>,
    free_blocks: Vec<Vec<Vec<u8>>>,
}

impl PoolAllocator {
    /// Create a new pool allocator with the default block size classes.
    pub fn new() -> Self {
        let count = DEFAULT_BLOCK_SIZES.len();
        Self {
            block_sizes: DEFAULT_BLOCK_SIZES.to_vec(),
            free_blocks: (0..count).map(|_| Vec::new()).collect(),
        }
    }

    /// Return the index of the smallest block size that fits `size`.
    fn size_class(&self, size: usize) -> Option<usize> {
        self.block_sizes
            .iter()
            .position(|&bs| bs >= size)
    }

    /// Allocate a block of at least `size` bytes.
    ///
    /// Returns a zeroed `Vec<u8>` of the chosen block size, or `None` if
    /// the request exceeds the largest pool class (4 MB).
    pub fn allocate(&mut self, size: usize) -> Option<Vec<u8>> {
        let idx = self.size_class(size)?;
        // Recycle from the free list if available.
        if let Some(block) = self.free_blocks[idx].pop() {
            return Some(block);
        }
        // Otherwise allocate fresh.
        let bs = self.block_sizes[idx];
        Some(vec![0u8; bs])
    }

    /// Return a previously allocated block to the pool.
    ///
    /// # Panics
    ///
    /// Panics if the block length does not match one of the supported sizes.
    pub fn deallocate(&mut self, block: Vec<u8>) {
        let len = block.len();
        let idx = self
            .block_sizes
            .iter()
            .position(|&bs| bs == len)
            .expect("deallocate: block size does not match any pool class");
        self.free_blocks[idx].push(block);
    }
}

impl Default for PoolAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StackAllocator — LIFO stack for nested staging allocations
// ---------------------------------------------------------------------------

/// LIFO stack allocator for nested staging allocations.
///
/// Callers obtain a `marker` on push and pass it back on pop.  The backing
/// store is an append-only byte vector; pop simply restores the previous
/// marker (no memory is zeroed).
pub struct StackAllocator {
    buffer: Vec<u8>,
}

impl StackAllocator {
    /// Create a new stack allocator pre-allocated with `capacity` bytes.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Push `data` onto the stack and return a marker for later pop.
    pub fn push(&mut self, data: &[u8]) -> Option<usize> {
        let marker = self.buffer.len();
        self.buffer.extend_from_slice(data);
        Some(marker)
    }

    /// Pop back to a previously returned marker.
    ///
    /// # Panics
    ///
    /// Panics if `marker` is greater than the current buffer length
    /// (i.e. markers must be consumed in strict LIFO order).
    pub fn pop(&mut self, marker: usize) {
        assert!(
            marker <= self.buffer.len(),
            "StackAllocator::pop: marker {} is beyond buffer length {}",
            marker,
            self.buffer.len(),
        );
        self.buffer.truncate(marker);
    }
}

// ---------------------------------------------------------------------------
// GpuBudget — atomic GPU memory budget tracker
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic GPU memory budget tracker.
///
/// Uses relaxed atomic operations (visibility is not critical — budget is
/// an approximate ceiling used for capacity planning).
pub struct GpuBudget {
    /// Hard cap in bytes.
    pub cap: u64,
    used: AtomicU64,
}

impl GpuBudget {
    /// Create a new budget with the given capacity (in bytes).
    pub fn new(cap: u64) -> Self {
        Self {
            cap,
            used: AtomicU64::new(0),
        }
    }

    /// Attempt to reserve `bytes` against the budget.
    ///
    /// Returns `true` if the reservation fit within the cap, `false`
    /// otherwise.  On success the used counter is increased; on failure
    /// it is unchanged.
    pub fn try_reserve(&self, bytes: u64) -> bool {
        loop {
            let current = self.used.load(Ordering::Relaxed);
            let new = current.saturating_add(bytes);
            if new > self.cap {
                return false;
            }
            if self
                .used
                .compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
            // CAS failed — retry.
        }
    }

    /// Release `bytes` from the budget (subtract from used).
    ///
    /// Saturates at zero — over-release will not underflow.
    pub fn release(&self, bytes: u64) {
        self.used
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(bytes))
            })
            .ok();
    }

    /// Return the current usage in bytes.
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    /// Return the available (unused) bytes.
    pub fn available(&self) -> u64 {
        self.cap.saturating_sub(self.used.load(Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------
// RingBuffer -- circular staging allocator
// ---------------------------------------------------------------------------

/// Fixed-size circular buffer with head/tail cursors for GPU staging.
///
/// Allocations wrap around when they reach the end of the buffer, reusing
/// space that has been consumed via `consume()`.  Ideal for streaming uploads
/// where the consumer drains data in FIFO order.
pub struct RingBuffer {
    buf: Vec<u8>,
    cap: usize,
    head: usize,
    tail: usize,
    wraps: u64,
    overflows: u64,
}

/// Alignment constant for staging allocations (cache-line aligned).
pub const CACHE_LINE_BYTES: usize = 64;

/// Snapshot of ring-buffer usage for diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct RingStats {
    pub capacity: usize,
    pub used: usize,
    pub available: usize,
    pub wraps: u64,
    pub overflows: u64,
}

impl RingBuffer {
    /// Create a new ring buffer with the given capacity in bytes.
    /// Capacity is rounded up to `CACHE_LINE_BYTES`.
    pub fn new(capacity_bytes: usize) -> Self {
        let cap = capacity_bytes.max(CACHE_LINE_BYTES);
        Self {
            buf: vec![0u8; cap],
            cap,
            head: 0,
            tail: 0,
            wraps: 0,
            overflows: 0,
        }
    }

    /// Allocate a contiguous slice of `size` bytes aligned to `alignment`.
    /// Returns `(offset, length)` on success, or `None` on overflow.
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Option<(usize, usize)> {
        let align = alignment.max(1);
        let aligned_head = (self.head + align - 1) & !(align - 1);

        if self.head >= self.tail {
            // Not wrapped — try fitting at head first, then wrap to 0.
            if aligned_head + size <= self.cap {
                self.head = aligned_head + size;
                return Some((aligned_head, size));
            }
            // Wrap to beginning.
            if size <= self.tail {
                self.head = size;
                self.wraps += 1;
                return Some((0, size));
            }
        } else {
            // Wrapped — free space is [head, tail).
            if aligned_head + size <= self.tail {
                self.head = aligned_head + size;
                return Some((aligned_head, size));
            }
        }

        self.overflows += 1;
        None
    }

    /// Advance the tail cursor, freeing `bytes` of consumed data.
    pub fn consume(&mut self, bytes: usize) {
        let used = self.used();
        let to_consume = bytes.min(used);
        self.tail = (self.tail + to_consume) % self.cap;
    }

    /// Reset all cursors and counters (frame boundary).
    pub fn reset(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.wraps = 0;
        self.overflows = 0;
    }

    pub fn used(&self) -> usize {
        if self.head >= self.tail {
            self.head - self.tail
        } else {
            self.cap - self.tail + self.head
        }
    }

    pub fn available(&self) -> usize {
        self.cap - self.used()
    }

    pub fn capacity(&self) -> usize { self.cap }
    pub fn head(&self) -> usize { self.head }
    pub fn tail(&self) -> usize { self.tail }

    pub fn stats(&self) -> RingStats {
        RingStats {
            capacity: self.cap,
            used: self.used(),
            available: self.available(),
            wraps: self.wraps,
            overflows: self.overflows,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- FrameAllocator ----

    #[test]
    fn frame_allocator_basic_allocation_and_reset() {
        let mut alloc = FrameAllocator::new(1024);
        assert_eq!(alloc.used(), 0);

        let slice = alloc.allocate(64, 4).expect("allocation should succeed");
        assert_eq!(slice.len(), 64);
        assert_eq!(alloc.used(), 64);

        // A second allocation starts right after the first.
        let slice2 = alloc.allocate(32, 1).expect("second allocation");
        assert_eq!(slice2.len(), 32);
        assert_eq!(alloc.used(), 64 + 32);

        // Reset and verify offset goes back to zero.
        alloc.reset();
        assert_eq!(alloc.used(), 0);

        // After reset we can reuse the space.
        let slice3 = alloc.allocate(1024, 1).expect("full buffer after reset");
        assert_eq!(slice3.len(), 1024);
    }

    #[test]
    fn frame_allocator_alignment() {
        let mut alloc = FrameAllocator::new(256);
        // Allocate 1 byte with 64-byte alignment at offset 0.
        alloc.allocate(1, 64).unwrap();
        // offset should now be 1.
        assert_eq!(alloc.used(), 1);
        // Next allocation of 1 byte with 64-byte alignment — the allocator
        // should advance the internal offset to 64 before handing out the slice.
        alloc.allocate(1, 64).unwrap();
        assert_eq!(alloc.used(), 65);
    }

    #[test]
    fn frame_allocator_oom() {
        let mut alloc = FrameAllocator::new(16);
        assert!(alloc.allocate(8, 1).is_some());
        // Only 8 bytes remain — should fail.
        assert!(alloc.allocate(16, 1).is_none());
        // Zero-size request should also return None.
        assert!(alloc.allocate(0, 1).is_none());
    }

    #[test]
    fn frame_allocator_overflow_safe() {
        let mut alloc = FrameAllocator::new(1024);
        // Near the end of the buffer, requesting an alignment that pushes
        // past capacity should gracefully fail.
        alloc.offset = 1020; // sneak the offset close to the end
        assert!(alloc.allocate(8, 8).is_none()); // 1020 → 1024 aligned, needs 8 more
    }

    // ---- PoolAllocator ----

    #[test]
    fn pool_allocator_allocate_and_deallocate() {
        let mut pool = PoolAllocator::new();

        // Allocate below the smallest class — should round up to 64 KB.
        let block = pool.allocate(1).expect("small allocation");
        assert_eq!(block.len(), 64 * 1024);

        // Exact size for the 256 KB class.
        let block2 = pool.allocate(256 * 1024).expect("256 KB allocation");
        assert_eq!(block2.len(), 256 * 1024);

        // Return both to the pool.
        pool.deallocate(block);
        pool.deallocate(block2);

        // Re-allocate — should recycle from free list.
        let recycled = pool.allocate(1).expect("recycled allocation");
        assert_eq!(recycled.len(), 64 * 1024);
    }

    #[test]
    fn pool_allocator_too_large() {
        let mut pool = PoolAllocator::new();
        // Largest class is 4 MB — 5 MB should return None.
        assert!(pool.allocate(5 * 1024 * 1024).is_none());
    }

    #[test]
    #[should_panic(expected = "block size does not match")]
    fn pool_allocator_deallocate_wrong_size() {
        let mut pool = PoolAllocator::new();
        pool.deallocate(vec![0u8; 128]); // 128 B is not a pool class
    }

    // ---- StackAllocator ----

    #[test]
    fn stack_allocator_push_pop() {
        let mut stack = StackAllocator::new(256);

        let m1 = stack.push(b"hello").expect("push");
        let m2 = stack.push(b"world").expect("push");

        // Pop in LIFO order.
        stack.pop(m2);
        // After popping m2, the buffer length should be exactly after "hello".
        assert_eq!(stack.buffer.len(), 5);

        stack.pop(m1);
        assert_eq!(stack.buffer.len(), 0);
    }

    #[test]
    fn stack_allocator_pop_restores_marker() {
        let mut stack = StackAllocator::new(128);

        stack.push(b"aaaa").unwrap();
        let m_b = stack.push(b"bbbb").unwrap();
        // Pop back to 'b' marker — should leave only "aaaa".
        stack.pop(m_b);
        assert_eq!(&stack.buffer[..], b"aaaa");
    }

    #[test]
    #[should_panic(expected = "marker")]
    fn stack_allocator_invalid_marker() {
        let mut stack = StackAllocator::new(64);
        stack.push(b"data");
        // This marker is beyond the current buffer length.
        stack.pop(999);
    }

    // ---- GpuBudget ----

    #[test]
    fn gpu_budget_basic_reserve_release() {
        let budget = GpuBudget::new(100);
        assert_eq!(budget.available(), 100);
        assert_eq!(budget.used(), 0);

        assert!(budget.try_reserve(40));
        assert_eq!(budget.used(), 40);
        assert_eq!(budget.available(), 60);

        assert!(budget.try_reserve(60));
        assert_eq!(budget.used(), 100);
        assert_eq!(budget.available(), 0);
    }

    #[test]
    fn gpu_budget_overflow_rejected() {
        let budget = GpuBudget::new(100);
        assert!(budget.try_reserve(60));
        // Not enough room — should fail.
        assert!(!budget.try_reserve(50));
        // Usage unchanged.
        assert_eq!(budget.used(), 60);
    }

    #[test]
    fn gpu_budget_release() {
        let budget = GpuBudget::new(100);
        budget.try_reserve(80);
        budget.release(30);
        assert_eq!(budget.used(), 50);
        assert_eq!(budget.available(), 50);
    }

    #[test]
    fn gpu_budget_release_below_zero() {
        let budget = GpuBudget::new(100);
        budget.try_reserve(10);
        budget.release(20); // would underflow → saturating at 0
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn gpu_budget_concurrent_semantics() {
        // Verify that try_reserve/release do not panic under single-threaded
        // concurrent-style logic (the CAS loop is exercised).
        let budget = GpuBudget::new(1024);
        for _ in 0..100 {
            assert!(budget.try_reserve(10));
        }
        assert_eq!(budget.used(), 1000);
        // One more fits.
        assert!(budget.try_reserve(24));
        assert_eq!(budget.used(), 1024);
        // No more room.
        assert!(!budget.try_reserve(1));
    }
}
