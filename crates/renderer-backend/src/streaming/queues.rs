// SPDX-License-Identifier: MIT
//
// queues.rs -- Lock-free queues for 3-thread streaming architecture (T-AS-5.1)
//
// Provides:
// - SPSC queue for request submission (game thread -> streaming thread)
// - MPSC queue for priority updates (multiple threads -> streaming thread)
// - Ring buffer for GPU upload commands (streaming thread -> render thread)

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// SpscQueue -- Single-Producer Single-Consumer Lock-Free Queue
// ---------------------------------------------------------------------------

/// A bounded, lock-free SPSC (single-producer, single-consumer) queue.
///
/// This queue is designed for the game thread to submit streaming requests
/// to the streaming thread without blocking.
///
/// # Safety
///
/// - Only one thread may call `push()` (the producer)
/// - Only one thread may call `pop()` (the consumer)
/// - Multiple threads calling push or pop simultaneously is undefined behavior
///
/// # Implementation
///
/// Uses a bounded ring buffer with atomic head/tail indices. The queue is
/// lock-free and wait-free for both push and pop operations.
pub struct SpscQueue<T> {
    buffer: Box<[UnsafeCell<Option<T>>]>,
    capacity: usize,
    /// Write index (owned by producer)
    head: AtomicUsize,
    /// Read index (owned by consumer)
    tail: AtomicUsize,
}

// Safety: SpscQueue is Send+Sync because:
// - The buffer is accessed through atomic indices
// - Producer only writes to head, consumer only writes to tail
// - Data races are prevented by the SPSC protocol
unsafe impl<T: Send> Send for SpscQueue<T> {}
unsafe impl<T: Send> Sync for SpscQueue<T> {}

impl<T> SpscQueue<T> {
    /// Creates a new SPSC queue with the given capacity.
    ///
    /// The actual capacity is rounded up to the next power of 2 for efficient
    /// modulo operations.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(2);
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(UnsafeCell::new(None));
        }
        Self {
            buffer: buffer.into_boxed_slice(),
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Returns the capacity of the queue.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Attempts to push an item to the queue.
    ///
    /// Returns `Ok(())` if successful, or `Err(item)` if the queue is full.
    ///
    /// # Safety
    ///
    /// This must only be called from a single producer thread.
    pub fn push(&self, item: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        // Check if queue is full
        let next_head = (head + 1) & (self.capacity - 1);
        if next_head == tail {
            return Err(item);
        }

        // Safety: We own the head slot since we're the only producer
        unsafe {
            *self.buffer[head].get() = Some(item);
        }

        // Publish the write
        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Attempts to pop an item from the queue.
    ///
    /// Returns `Some(item)` if successful, or `None` if the queue is empty.
    ///
    /// # Safety
    ///
    /// This must only be called from a single consumer thread.
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        // Check if queue is empty
        if tail == head {
            return None;
        }

        // Safety: We own the tail slot since we're the only consumer
        let item = unsafe { (*self.buffer[tail].get()).take() };

        // Publish the read
        let next_tail = (tail + 1) & (self.capacity - 1);
        self.tail.store(next_tail, Ordering::Release);

        item
    }

    /// Returns the number of items currently in the queue.
    ///
    /// This is an approximation in concurrent scenarios.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        (head.wrapping_sub(tail)) & (self.capacity - 1)
    }

    /// Returns true if the queue is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head == tail
    }

    /// Returns true if the queue is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        ((head + 1) & (self.capacity - 1)) == tail
    }

    /// Clears all items from the queue.
    ///
    /// # Safety
    ///
    /// This must only be called when no other threads are accessing the queue.
    pub fn clear(&self) {
        while self.pop().is_some() {}
    }
}

impl<T> Drop for SpscQueue<T> {
    fn drop(&mut self) {
        // Drop any remaining items
        while self.pop().is_some() {}
    }
}

// ---------------------------------------------------------------------------
// MpscQueue -- Multi-Producer Single-Consumer Lock-Free Queue
// ---------------------------------------------------------------------------

/// A node in the MPSC queue.
struct MpscNode<T> {
    value: UnsafeCell<Option<T>>,
    next: AtomicPtr<MpscNode<T>>,
}

use std::ptr;
use std::sync::atomic::AtomicPtr;

impl<T> MpscNode<T> {
    fn new(value: Option<T>) -> *mut Self {
        Box::into_raw(Box::new(Self {
            value: UnsafeCell::new(value),
            next: AtomicPtr::new(ptr::null_mut()),
        }))
    }
}

/// A lock-free MPSC (multi-producer, single-consumer) queue.
///
/// This queue is designed for multiple threads (e.g., game thread, streaming
/// thread) to submit priority updates to the streaming thread.
///
/// # Safety
///
/// - Multiple threads may call `push()` concurrently
/// - Only one thread may call `pop()` (the consumer)
///
/// # Implementation
///
/// Uses a linked list with atomic CAS operations for lock-free push.
/// Based on the Michael-Scott queue algorithm.
pub struct MpscQueue<T> {
    head: AtomicPtr<MpscNode<T>>,
    tail: AtomicPtr<MpscNode<T>>,
    len: AtomicUsize,
}

// Safety: MpscQueue is Send+Sync because all operations use atomics
unsafe impl<T: Send> Send for MpscQueue<T> {}
unsafe impl<T: Send> Sync for MpscQueue<T> {}

impl<T> MpscQueue<T> {
    /// Creates a new empty MPSC queue.
    pub fn new() -> Self {
        // Create a dummy/sentinel node
        let dummy = MpscNode::new(None);
        Self {
            head: AtomicPtr::new(dummy),
            tail: AtomicPtr::new(dummy),
            len: AtomicUsize::new(0),
        }
    }

    /// Pushes an item to the queue.
    ///
    /// This operation is lock-free and can be called from multiple threads.
    pub fn push(&self, value: T) {
        let node = MpscNode::new(Some(value));

        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let tail_next = unsafe { (*tail).next.load(Ordering::Acquire) };

            if tail_next.is_null() {
                // Try to link the new node
                if unsafe {
                    (*tail)
                        .next
                        .compare_exchange(
                            ptr::null_mut(),
                            node,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                } {
                    // Successfully linked, try to swing tail
                    let _ = self.tail.compare_exchange(
                        tail,
                        node,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                    self.len.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            } else {
                // Help swing tail to the next node
                let _ = self.tail.compare_exchange(
                    tail,
                    tail_next,
                    Ordering::Release,
                    Ordering::Relaxed,
                );
            }
        }
    }

    /// Pops an item from the queue.
    ///
    /// Returns `Some(item)` if successful, or `None` if the queue is empty.
    ///
    /// # Safety
    ///
    /// This must only be called from a single consumer thread.
    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let head_next = unsafe { (*head).next.load(Ordering::Acquire) };

            if head == tail {
                if head_next.is_null() {
                    // Queue is empty
                    return None;
                }
                // Tail is lagging, help move it
                let _ = self.tail.compare_exchange(
                    tail,
                    head_next,
                    Ordering::Release,
                    Ordering::Relaxed,
                );
            } else if !head_next.is_null() {
                // Read value before CAS to avoid data race
                let value = unsafe { (*head_next.cast::<MpscNode<T>>()).value.get().read() };

                if self
                    .head
                    .compare_exchange(head, head_next, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    // Successfully dequeued, free the old head
                    unsafe {
                        drop(Box::from_raw(head));
                    }
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    return value;
                }
            }
        }
    }

    /// Returns the approximate number of items in the queue.
    #[inline]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Returns true if the queue is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for MpscQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for MpscQueue<T> {
    fn drop(&mut self) {
        // Pop all remaining items
        while self.pop().is_some() {}

        // Free the dummy node
        let head = self.head.load(Ordering::Relaxed);
        unsafe {
            drop(Box::from_raw(head));
        }
    }
}

// ---------------------------------------------------------------------------
// RingBuffer -- Bounded Ring Buffer for GPU Upload Commands
// ---------------------------------------------------------------------------

/// A slot in the ring buffer with atomic state tracking.
#[repr(C)]
struct RingSlot<T> {
    /// The data in this slot.
    data: UnsafeCell<Option<T>>,
    /// Sequence number for this slot (for ABA prevention).
    sequence: AtomicU64,
}

/// A bounded, lock-free ring buffer for streaming -> render thread communication.
///
/// This buffer is designed for the streaming thread to queue GPU upload commands
/// that the render thread will consume.
///
/// # Design
///
/// - Fixed capacity (power of 2)
/// - Sequence numbers for ABA prevention
/// - Supports wrap-around
/// - Single producer, single consumer
///
/// # Memory Layout
///
/// The buffer is cache-line padded to avoid false sharing.
pub struct RingBuffer<T> {
    buffer: Box<[RingSlot<T>]>,
    capacity: usize,
    mask: usize,
    /// Producer position
    write_pos: AtomicU64,
    /// Consumer position
    read_pos: AtomicU64,
}

// Safety: RingBuffer uses atomic operations for synchronization
unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

impl<T> RingBuffer<T> {
    /// Creates a new ring buffer with the given capacity.
    ///
    /// Capacity is rounded up to the next power of 2.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(2);
        let mask = capacity - 1;

        let mut buffer = Vec::with_capacity(capacity);
        for i in 0..capacity {
            buffer.push(RingSlot {
                data: UnsafeCell::new(None),
                sequence: AtomicU64::new(i as u64),
            });
        }

        Self {
            buffer: buffer.into_boxed_slice(),
            capacity,
            mask,
            write_pos: AtomicU64::new(0),
            read_pos: AtomicU64::new(0),
        }
    }

    /// Returns the capacity of the ring buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Attempts to push an item to the ring buffer.
    ///
    /// Returns `Ok(())` if successful, or `Err(item)` if the buffer is full.
    pub fn push(&self, item: T) -> Result<(), T> {
        let mut pos = self.write_pos.load(Ordering::Relaxed);

        loop {
            let slot = &self.buffer[(pos as usize) & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq as i64 - pos as i64;

            if diff == 0 {
                // Slot is ready for writing
                match self.write_pos.compare_exchange_weak(
                    pos,
                    pos + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // Write the data
                        unsafe {
                            *slot.data.get() = Some(item);
                        }
                        // Publish
                        slot.sequence.store(pos + 1, Ordering::Release);
                        return Ok(());
                    }
                    Err(new_pos) => {
                        pos = new_pos;
                    }
                }
            } else if diff < 0 {
                // Buffer is full
                return Err(item);
            } else {
                // Another producer advanced, retry
                pos = self.write_pos.load(Ordering::Relaxed);
            }
        }
    }

    /// Attempts to pop an item from the ring buffer.
    ///
    /// Returns `Some(item)` if successful, or `None` if the buffer is empty.
    pub fn pop(&self) -> Option<T> {
        let mut pos = self.read_pos.load(Ordering::Relaxed);

        loop {
            let slot = &self.buffer[(pos as usize) & self.mask];
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq as i64 - (pos as i64 + 1);

            if diff == 0 {
                // Slot is ready for reading
                match self.read_pos.compare_exchange_weak(
                    pos,
                    pos + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // Read the data
                        let item = unsafe { (*slot.data.get()).take() };
                        // Reset sequence for next round
                        slot.sequence
                            .store(pos + self.capacity as u64, Ordering::Release);
                        return item;
                    }
                    Err(new_pos) => {
                        pos = new_pos;
                    }
                }
            } else if diff < 0 {
                // Buffer is empty
                return None;
            } else {
                // Another consumer advanced, retry
                pos = self.read_pos.load(Ordering::Relaxed);
            }
        }
    }

    /// Returns the approximate number of items in the buffer.
    pub fn len(&self) -> usize {
        let write = self.write_pos.load(Ordering::Relaxed);
        let read = self.read_pos.load(Ordering::Relaxed);
        (write.wrapping_sub(read)) as usize
    }

    /// Returns true if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Clears all items from the buffer.
    pub fn clear(&self) {
        while self.pop().is_some() {}
    }

    /// Returns the number of times the buffer has wrapped around.
    pub fn wrap_count(&self) -> u64 {
        self.write_pos.load(Ordering::Relaxed) / self.capacity as u64
    }
}

impl<T> Drop for RingBuffer<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

// ---------------------------------------------------------------------------
// AssetLoadState -- Atomic state flags for asset loading stages
// ---------------------------------------------------------------------------

/// Atomic state flags for tracking asset loading stages.
///
/// Each asset progresses through: Queued -> Loading -> Uploading -> Ready
/// or may transition to Failed at any point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LoadState {
    /// Asset is not yet queued for loading.
    None = 0,
    /// Asset is queued for loading.
    Queued = 1,
    /// Asset is being loaded from disk.
    Loading = 2,
    /// Asset data is being uploaded to GPU.
    Uploading = 3,
    /// Asset is ready for use.
    Ready = 4,
    /// Asset loading failed.
    Failed = 5,
    /// Asset is being unloaded.
    Unloading = 6,
}

impl LoadState {
    /// Creates a LoadState from a u8 value.
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => LoadState::None,
            1 => LoadState::Queued,
            2 => LoadState::Loading,
            3 => LoadState::Uploading,
            4 => LoadState::Ready,
            5 => LoadState::Failed,
            6 => LoadState::Unloading,
            _ => LoadState::None,
        }
    }
}

/// Atomic wrapper for LoadState with CAS operations.
pub struct AtomicLoadState {
    state: AtomicU8,
}

use std::sync::atomic::AtomicU8;

impl AtomicLoadState {
    /// Creates a new atomic load state.
    #[inline]
    pub const fn new(state: LoadState) -> Self {
        Self {
            state: AtomicU8::new(state as u8),
        }
    }

    /// Loads the current state.
    #[inline]
    pub fn load(&self, ordering: Ordering) -> LoadState {
        LoadState::from_u8(self.state.load(ordering))
    }

    /// Stores a new state.
    #[inline]
    pub fn store(&self, state: LoadState, ordering: Ordering) {
        self.state.store(state as u8, ordering);
    }

    /// Attempts to transition from one state to another.
    ///
    /// Returns `Ok(())` if the transition was successful, or `Err(current)`
    /// if the current state didn't match the expected state.
    #[inline]
    pub fn transition(
        &self,
        from: LoadState,
        to: LoadState,
        success: Ordering,
        failure: Ordering,
    ) -> Result<(), LoadState> {
        match self
            .state
            .compare_exchange(from as u8, to as u8, success, failure)
        {
            Ok(_) => Ok(()),
            Err(current) => Err(LoadState::from_u8(current)),
        }
    }

    /// Attempts a weak transition (may spuriously fail).
    #[inline]
    pub fn transition_weak(
        &self,
        from: LoadState,
        to: LoadState,
        success: Ordering,
        failure: Ordering,
    ) -> Result<(), LoadState> {
        match self
            .state
            .compare_exchange_weak(from as u8, to as u8, success, failure)
        {
            Ok(_) => Ok(()),
            Err(current) => Err(LoadState::from_u8(current)),
        }
    }

    /// Returns true if the asset is in a terminal state (Ready or Failed).
    #[inline]
    pub fn is_terminal(&self) -> bool {
        let state = self.load(Ordering::Relaxed);
        matches!(state, LoadState::Ready | LoadState::Failed)
    }

    /// Returns true if the asset is currently being processed.
    #[inline]
    pub fn is_in_progress(&self) -> bool {
        let state = self.load(Ordering::Relaxed);
        matches!(
            state,
            LoadState::Queued | LoadState::Loading | LoadState::Uploading
        )
    }
}

impl Default for AtomicLoadState {
    fn default() -> Self {
        Self::new(LoadState::None)
    }
}

impl std::fmt::Debug for AtomicLoadState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicLoadState")
            .field("state", &self.load(Ordering::Relaxed))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// AtomicRefCount -- Thread-safe reference counting
// ---------------------------------------------------------------------------

/// Atomic reference count for asset lifetime management.
///
/// Uses acquire/release semantics to ensure proper synchronization when
/// the last reference is dropped.
pub struct AtomicRefCount {
    count: AtomicU32,
}

impl AtomicRefCount {
    /// Creates a new reference count initialized to 1.
    #[inline]
    pub const fn new() -> Self {
        Self {
            count: AtomicU32::new(1),
        }
    }

    /// Creates a new reference count with the given initial value.
    #[inline]
    pub const fn with_count(count: u32) -> Self {
        Self {
            count: AtomicU32::new(count),
        }
    }

    /// Increments the reference count.
    ///
    /// Returns the previous count.
    #[inline]
    pub fn increment(&self) -> u32 {
        // Relaxed is fine for increment since we don't need to synchronize
        // with any other operation.
        self.count.fetch_add(1, Ordering::Relaxed)
    }

    /// Decrements the reference count.
    ///
    /// Returns `true` if this was the last reference (count reached 0).
    #[inline]
    pub fn decrement(&self) -> bool {
        // Use Release ordering so that all previous writes are visible
        // to the thread that observes the count reaching 0.
        let prev = self.count.fetch_sub(1, Ordering::Release);

        if prev == 1 {
            // This was the last reference. Use Acquire to synchronize
            // with all previous Release operations.
            std::sync::atomic::fence(Ordering::Acquire);
            true
        } else {
            false
        }
    }

    /// Returns the current reference count.
    #[inline]
    pub fn get(&self) -> u32 {
        self.count.load(Ordering::Relaxed)
    }

    /// Returns true if the reference count is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.get() == 0
    }

    /// Attempts to increment only if the count is non-zero.
    ///
    /// Returns `true` if the increment was successful.
    #[inline]
    pub fn try_increment(&self) -> bool {
        let mut current = self.count.load(Ordering::Relaxed);
        loop {
            if current == 0 {
                return false;
            }
            match self.count.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(new_current) => current = new_current,
            }
        }
    }
}

impl Default for AtomicRefCount {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AtomicRefCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicRefCount")
            .field("count", &self.get())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// StreamRequest -- Request submitted to streaming thread
// ---------------------------------------------------------------------------

/// A streaming request submitted by the game thread.
#[derive(Debug, Clone)]
pub struct StreamRequest {
    /// Unique identifier for the asset.
    pub asset_id: u64,
    /// Priority of the request (lower = higher priority).
    pub priority: u32,
    /// Path or identifier for the asset data.
    pub path: String,
    /// Size hint in bytes (0 if unknown).
    pub size_hint: u64,
    /// Timestamp when the request was created.
    pub timestamp: u64,
}

impl StreamRequest {
    /// Creates a new streaming request.
    pub fn new(asset_id: u64, priority: u32, path: String) -> Self {
        Self {
            asset_id,
            priority,
            path,
            size_hint: 0,
            timestamp: 0,
        }
    }

    /// Sets the size hint.
    pub fn with_size_hint(mut self, size: u64) -> Self {
        self.size_hint = size;
        self
    }

    /// Sets the timestamp.
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }
}

// ---------------------------------------------------------------------------
// PriorityUpdate -- Priority update message
// ---------------------------------------------------------------------------

/// A priority update message sent to the streaming thread.
#[derive(Debug, Clone, Copy)]
pub struct PriorityUpdate {
    /// The asset ID to update.
    pub asset_id: u64,
    /// The new priority value.
    pub new_priority: u32,
    /// Whether to cancel the request entirely.
    pub cancel: bool,
}

impl PriorityUpdate {
    /// Creates a priority update.
    pub fn new(asset_id: u64, new_priority: u32) -> Self {
        Self {
            asset_id,
            new_priority,
            cancel: false,
        }
    }

    /// Creates a cancellation request.
    pub fn cancel(asset_id: u64) -> Self {
        Self {
            asset_id,
            new_priority: u32::MAX,
            cancel: true,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuUploadCommand -- Command sent to render thread
// ---------------------------------------------------------------------------

/// GPU upload command types.
#[derive(Debug, Clone)]
pub enum GpuUploadCommand {
    /// Upload texture data to GPU.
    UploadTexture {
        asset_id: u64,
        data: Arc<[u8]>,
        width: u32,
        height: u32,
        format: u32,
        mip_level: u32,
    },
    /// Upload buffer data to GPU.
    UploadBuffer {
        asset_id: u64,
        data: Arc<[u8]>,
        offset: u64,
    },
    /// Signal that an asset is ready.
    AssetReady { asset_id: u64 },
    /// Signal that an asset failed to load.
    AssetFailed { asset_id: u64, error_code: u32 },
    /// Fence command for synchronization.
    Fence { fence_id: u64 },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::thread;

    // ── SPSC Queue Tests ────────────────────────────────────────────────

    #[test]
    fn spsc_queue_basic_push_pop() {
        let queue = SpscQueue::<u32>::new(8);
        assert!(queue.is_empty());

        queue.push(1).unwrap();
        queue.push(2).unwrap();
        queue.push(3).unwrap();

        assert_eq!(queue.len(), 3);
        assert!(!queue.is_empty());

        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.pop(), None);
        assert!(queue.is_empty());
    }

    #[test]
    fn spsc_queue_full_behavior() {
        let queue = SpscQueue::<u32>::new(4); // Rounds to 4

        // Fill the queue (capacity - 1 items due to ring buffer)
        assert!(queue.push(1).is_ok());
        assert!(queue.push(2).is_ok());
        assert!(queue.push(3).is_ok());

        // Should be full now
        assert!(queue.is_full());
        assert!(queue.push(4).is_err());

        // Pop one and push should work
        assert_eq!(queue.pop(), Some(1));
        assert!(queue.push(4).is_ok());
    }

    #[test]
    fn spsc_queue_wrap_around() {
        let queue = SpscQueue::<u32>::new(4);

        // Fill and empty multiple times to test wrap-around
        for round in 0..3 {
            for i in 0..3 {
                queue.push(round * 10 + i).unwrap();
            }
            for i in 0..3 {
                assert_eq!(queue.pop(), Some(round * 10 + i));
            }
        }
    }

    #[test]
    fn spsc_queue_concurrent_producer_consumer() {
        let queue = Arc::new(SpscQueue::<u32>::new(1024));
        let queue_producer = Arc::clone(&queue);
        let queue_consumer = Arc::clone(&queue);

        let items: u32 = 10000;
        let producer = thread::spawn(move || {
            for i in 0..items {
                while queue_producer.push(i).is_err() {
                    thread::yield_now();
                }
            }
        });

        let consumer = thread::spawn(move || {
            let mut sum: u64 = 0;
            let mut count = 0;
            while count < items {
                if let Some(v) = queue_consumer.pop() {
                    sum += v as u64;
                    count += 1;
                } else {
                    thread::yield_now();
                }
            }
            sum
        });

        producer.join().unwrap();
        let sum = consumer.join().unwrap();

        // Sum of 0..items
        let expected: u64 = (items as u64 * (items as u64 - 1)) / 2;
        assert_eq!(sum, expected);
    }

    #[test]
    fn spsc_queue_clear() {
        let queue = SpscQueue::<u32>::new(8);
        queue.push(1).unwrap();
        queue.push(2).unwrap();
        queue.push(3).unwrap();

        queue.clear();
        assert!(queue.is_empty());
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn spsc_queue_capacity_power_of_two() {
        let queue = SpscQueue::<u32>::new(5);
        assert_eq!(queue.capacity(), 8); // Rounded up to next power of 2

        let queue = SpscQueue::<u32>::new(16);
        assert_eq!(queue.capacity(), 16); // Already power of 2
    }

    // ── MPSC Queue Tests ────────────────────────────────────────────────

    #[test]
    fn mpsc_queue_basic_push_pop() {
        let queue = MpscQueue::<u32>::new();
        assert!(queue.is_empty());

        queue.push(1);
        queue.push(2);
        queue.push(3);

        assert_eq!(queue.len(), 3);

        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn mpsc_queue_concurrent_producers() {
        let queue = Arc::new(MpscQueue::<u32>::new());
        let mut handles = vec![];

        // Spawn 4 producer threads
        for t in 0..4 {
            let q = Arc::clone(&queue);
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    q.push(t * 1000 + i);
                }
            }));
        }

        // Wait for producers
        for h in handles {
            h.join().unwrap();
        }

        // Consume all items
        let mut count = 0;
        while queue.pop().is_some() {
            count += 1;
        }

        assert_eq!(count, 4000);
    }

    #[test]
    fn mpsc_queue_concurrent_push_pop() {
        let queue = Arc::new(MpscQueue::<u32>::new());
        let total_items: u32 = 10000;

        let queue_producer = Arc::clone(&queue);
        let queue_consumer = Arc::clone(&queue);

        let producer = thread::spawn(move || {
            for i in 0..total_items {
                queue_producer.push(i);
            }
        });

        let consumer = thread::spawn(move || {
            let mut collected = vec![];
            let mut attempts = 0;
            while collected.len() < total_items as usize {
                if let Some(v) = queue_consumer.pop() {
                    collected.push(v);
                    attempts = 0;
                } else {
                    attempts += 1;
                    if attempts > 1000000 {
                        break; // Timeout
                    }
                    thread::yield_now();
                }
            }
            collected
        });

        producer.join().unwrap();
        let collected = consumer.join().unwrap();

        assert_eq!(collected.len(), total_items as usize);
    }

    // ── Ring Buffer Tests ───────────────────────────────────────────────

    #[test]
    fn ring_buffer_basic_push_pop() {
        let buffer = RingBuffer::<u32>::new(8);
        assert!(buffer.is_empty());

        buffer.push(1).unwrap();
        buffer.push(2).unwrap();
        buffer.push(3).unwrap();

        assert_eq!(buffer.len(), 3);
        assert!(!buffer.is_empty());

        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn ring_buffer_full_behavior() {
        let buffer = RingBuffer::<u32>::new(4);

        buffer.push(1).unwrap();
        buffer.push(2).unwrap();
        buffer.push(3).unwrap();
        buffer.push(4).unwrap();

        assert!(buffer.is_full());
        assert!(buffer.push(5).is_err());

        // Pop one and push should work
        assert_eq!(buffer.pop(), Some(1));
        assert!(buffer.push(5).is_ok());
    }

    #[test]
    fn ring_buffer_wrap_around() {
        let buffer = RingBuffer::<u32>::new(4);

        // Fill and empty multiple times to test wrap-around
        for round in 0..5 {
            for i in 0..4 {
                buffer.push(round * 10 + i).unwrap();
            }
            for i in 0..4 {
                assert_eq!(buffer.pop(), Some(round * 10 + i));
            }
        }

        // Verify wrap count increased
        assert!(buffer.wrap_count() > 0);
    }

    #[test]
    fn ring_buffer_wrap_count() {
        let buffer = RingBuffer::<u32>::new(4);

        // Initial wrap count should be 0
        assert_eq!(buffer.wrap_count(), 0);

        // Fill and empty several rounds
        for _ in 0..10 {
            for i in 0..4 {
                buffer.push(i).unwrap();
            }
            for _ in 0..4 {
                buffer.pop();
            }
        }

        assert!(buffer.wrap_count() >= 10);
    }

    #[test]
    fn ring_buffer_concurrent_access() {
        let buffer = Arc::new(RingBuffer::<u32>::new(256));
        let buffer_producer = Arc::clone(&buffer);
        let buffer_consumer = Arc::clone(&buffer);

        let items: u32 = 5000;
        let producer = thread::spawn(move || {
            for i in 0..items {
                while buffer_producer.push(i).is_err() {
                    thread::yield_now();
                }
            }
        });

        let consumer = thread::spawn(move || {
            let mut sum: u64 = 0;
            let mut count = 0;
            while count < items {
                if let Some(v) = buffer_consumer.pop() {
                    sum += v as u64;
                    count += 1;
                } else {
                    thread::yield_now();
                }
            }
            sum
        });

        producer.join().unwrap();
        let sum = consumer.join().unwrap();

        let expected: u64 = (items as u64 * (items as u64 - 1)) / 2;
        assert_eq!(sum, expected);
    }

    // ── Load State Tests ────────────────────────────────────────────────

    #[test]
    fn load_state_transitions() {
        let state = AtomicLoadState::new(LoadState::None);
        assert_eq!(state.load(Ordering::Relaxed), LoadState::None);

        // Valid transition: None -> Queued
        assert!(state
            .transition(LoadState::None, LoadState::Queued, Ordering::AcqRel, Ordering::Acquire)
            .is_ok());
        assert_eq!(state.load(Ordering::Relaxed), LoadState::Queued);

        // Valid transition: Queued -> Loading
        assert!(state
            .transition(
                LoadState::Queued,
                LoadState::Loading,
                Ordering::AcqRel,
                Ordering::Acquire
            )
            .is_ok());
        assert_eq!(state.load(Ordering::Relaxed), LoadState::Loading);

        // Invalid transition: wrong expected state
        let result = state.transition(
            LoadState::Queued,
            LoadState::Ready,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LoadState::Loading);
    }

    #[test]
    fn load_state_terminal_check() {
        let state = AtomicLoadState::new(LoadState::Ready);
        assert!(state.is_terminal());

        let state = AtomicLoadState::new(LoadState::Failed);
        assert!(state.is_terminal());

        let state = AtomicLoadState::new(LoadState::Loading);
        assert!(!state.is_terminal());
    }

    #[test]
    fn load_state_in_progress_check() {
        let state = AtomicLoadState::new(LoadState::Loading);
        assert!(state.is_in_progress());

        let state = AtomicLoadState::new(LoadState::Uploading);
        assert!(state.is_in_progress());

        let state = AtomicLoadState::new(LoadState::Ready);
        assert!(!state.is_in_progress());
    }

    // ── Atomic Ref Count Tests ──────────────────────────────────────────

    #[test]
    fn ref_count_basic_operations() {
        let rc = AtomicRefCount::new();
        assert_eq!(rc.get(), 1);

        rc.increment();
        assert_eq!(rc.get(), 2);

        rc.increment();
        assert_eq!(rc.get(), 3);

        assert!(!rc.decrement()); // Not last ref
        assert_eq!(rc.get(), 2);

        assert!(!rc.decrement()); // Not last ref
        assert_eq!(rc.get(), 1);

        assert!(rc.decrement()); // Last ref
        assert_eq!(rc.get(), 0);
    }

    #[test]
    fn ref_count_try_increment() {
        let rc = AtomicRefCount::new();

        // Should succeed when count > 0
        assert!(rc.try_increment());
        assert_eq!(rc.get(), 2);

        // Decrement to 0
        rc.decrement();
        rc.decrement();
        assert_eq!(rc.get(), 0);

        // Should fail when count == 0
        assert!(!rc.try_increment());
        assert_eq!(rc.get(), 0);
    }

    #[test]
    fn ref_count_concurrent_increment() {
        let rc = Arc::new(AtomicRefCount::new());
        let mut handles = vec![];

        // Spawn 10 threads that each increment 1000 times
        for _ in 0..10 {
            let rc = Arc::clone(&rc);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    rc.increment();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // 1 initial + 10 * 1000 increments
        assert_eq!(rc.get(), 10001);
    }

    #[test]
    fn ref_count_concurrent_decrement() {
        let rc = Arc::new(AtomicRefCount::with_count(10000));
        let last_ref = Arc::new(AtomicBool::new(false));
        let mut handles = vec![];

        // Spawn 10 threads that each decrement 1000 times
        for _ in 0..10 {
            let rc = Arc::clone(&rc);
            let last_ref = Arc::clone(&last_ref);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    if rc.decrement() {
                        last_ref.store(true, Ordering::Relaxed);
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(rc.get(), 0);
        assert!(last_ref.load(Ordering::Relaxed));
    }

    // ── Stream Request Tests ────────────────────────────────────────────

    #[test]
    fn stream_request_builder() {
        let request = StreamRequest::new(42, 5, "textures/hero.dds".to_string())
            .with_size_hint(1024 * 1024)
            .with_timestamp(12345);

        assert_eq!(request.asset_id, 42);
        assert_eq!(request.priority, 5);
        assert_eq!(request.path, "textures/hero.dds");
        assert_eq!(request.size_hint, 1024 * 1024);
        assert_eq!(request.timestamp, 12345);
    }

    // ── Priority Update Tests ───────────────────────────────────────────

    #[test]
    fn priority_update_creation() {
        let update = PriorityUpdate::new(42, 10);
        assert_eq!(update.asset_id, 42);
        assert_eq!(update.new_priority, 10);
        assert!(!update.cancel);

        let cancel = PriorityUpdate::cancel(99);
        assert_eq!(cancel.asset_id, 99);
        assert!(cancel.cancel);
    }

    // ── GPU Upload Command Tests ────────────────────────────────────────

    #[test]
    fn gpu_upload_command_texture() {
        let data: Arc<[u8]> = Arc::from(vec![0u8; 1024]);
        let cmd = GpuUploadCommand::UploadTexture {
            asset_id: 1,
            data,
            width: 256,
            height: 256,
            format: 87, // RGBA8
            mip_level: 0,
        };

        if let GpuUploadCommand::UploadTexture {
            asset_id,
            width,
            height,
            ..
        } = cmd
        {
            assert_eq!(asset_id, 1);
            assert_eq!(width, 256);
            assert_eq!(height, 256);
        } else {
            panic!("Expected UploadTexture");
        }
    }

    #[test]
    fn gpu_upload_command_fence() {
        let cmd = GpuUploadCommand::Fence { fence_id: 42 };
        if let GpuUploadCommand::Fence { fence_id } = cmd {
            assert_eq!(fence_id, 42);
        } else {
            panic!("Expected Fence");
        }
    }
}
