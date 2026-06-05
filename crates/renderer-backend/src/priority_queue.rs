// SPDX-License-Identifier: MIT
//
// priority_queue.rs — Multi-tier priority queue with starvation prevention
// (T-MAT-10.2)
//
// A tiered queue system for scheduling asset loading/processing with
// configurable starvation prevention that promotes long-waiting items
// to higher priority tiers.

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};

// ---------------------------------------------------------------------------
// PriorityTier
// ---------------------------------------------------------------------------

/// Priority tiers for queue items, ordered from highest to lowest priority.
///
/// Items in higher-priority (lower numeric value) tiers are dequeued first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PriorityTier {
    /// Immediately visible assets — highest priority, processed first.
    Critical = 0,
    /// Assets about to become visible — second priority.
    High = 1,
    /// Potentially visible assets — normal priority.
    Normal = 2,
    /// Background preload assets — lowest priority.
    Low = 3,
}

impl PriorityTier {
    /// Total number of priority tiers.
    pub const COUNT: usize = 4;

    /// Returns the next higher priority tier, or `None` if already at Critical.
    #[inline]
    pub fn promote(self) -> Option<Self> {
        match self {
            Self::Critical => None,
            Self::High => Some(Self::Critical),
            Self::Normal => Some(Self::High),
            Self::Low => Some(Self::Normal),
        }
    }

    /// Returns the numeric index (0-3) of this tier.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Creates a tier from a numeric index, clamping to valid range.
    #[inline]
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Critical,
            1 => Self::High,
            2 => Self::Normal,
            _ => Self::Low,
        }
    }
}

impl std::fmt::Display for PriorityTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Critical => write!(f, "Critical"),
            Self::High => write!(f, "High"),
            Self::Normal => write!(f, "Normal"),
            Self::Low => write!(f, "Low"),
        }
    }
}

// ---------------------------------------------------------------------------
// QueueItem
// ---------------------------------------------------------------------------

/// A single item in the tiered queue with metadata for starvation tracking.
#[derive(Debug, Clone)]
pub struct QueueItem<T> {
    /// The actual payload.
    pub item: T,
    /// Current priority tier (may change via promotion).
    pub tier: PriorityTier,
    /// Frame number when this item was enqueued (or last promoted).
    pub enqueue_frame: u64,
    /// Whether this item has been promoted due to starvation.
    pub promoted: bool,
}

impl<T> QueueItem<T> {
    /// Creates a new queue item at the specified tier and frame.
    #[inline]
    pub fn new(item: T, tier: PriorityTier, enqueue_frame: u64) -> Self {
        Self {
            item,
            tier,
            enqueue_frame,
            promoted: false,
        }
    }

    /// Returns how many frames this item has been waiting at its current tier.
    #[inline]
    pub fn frames_waiting(&self, current_frame: u64) -> u64 {
        current_frame.saturating_sub(self.enqueue_frame)
    }
}

// ---------------------------------------------------------------------------
// TierQueue (internal, per-tier)
// ---------------------------------------------------------------------------

/// A single tier's queue, protected by its own mutex for concurrent access.
struct TierQueue<T> {
    queue: Mutex<VecDeque<QueueItem<T>>>,
}

impl<T> TierQueue<T> {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    #[inline]
    fn lock(&self) -> MutexGuard<'_, VecDeque<QueueItem<T>>> {
        self.queue.lock().expect("tier queue mutex poisoned")
    }

    #[inline]
    fn push_back(&self, item: QueueItem<T>) {
        self.lock().push_back(item);
    }

    #[inline]
    fn pop_front(&self) -> Option<QueueItem<T>> {
        self.lock().pop_front()
    }

    #[inline]
    fn len(&self) -> usize {
        self.lock().len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }
}

impl<T> Default for TierQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TieredQueue
// ---------------------------------------------------------------------------

/// A multi-tier priority queue with starvation prevention.
///
/// Items are organized into four priority tiers (Critical, High, Normal, Low).
/// Each tier has its own lock, enabling concurrent access to different tiers.
///
/// Starvation prevention works by tracking how long items have been waiting
/// and promoting them to higher tiers after a configurable threshold.
///
/// # Performance
///
/// - `enqueue`: O(1) amortized
/// - `dequeue`: O(1) amortized
/// - `dequeue_batch`: O(n) where n is batch size
/// - `promote_starving`: O(total items) when called
///
/// # Thread Safety
///
/// The queue is thread-safe with per-tier locking. Multiple threads can
/// enqueue to and dequeue from different tiers simultaneously.
///
/// # Example
///
/// ```
/// use renderer_backend::priority_queue::{TieredQueue, PriorityTier};
///
/// let queue: TieredQueue<u32> = TieredQueue::new();
///
/// // Enqueue items at different priorities
/// queue.enqueue(1, PriorityTier::Low, 0);
/// queue.enqueue(2, PriorityTier::Critical, 0);
/// queue.enqueue(3, PriorityTier::Normal, 0);
///
/// // Critical items come first
/// assert_eq!(queue.dequeue().map(|q| q.item), Some(2));
/// assert_eq!(queue.dequeue().map(|q| q.item), Some(3));
/// assert_eq!(queue.dequeue().map(|q| q.item), Some(1));
/// ```
pub struct TieredQueue<T> {
    /// One queue per priority tier, indexed by tier value.
    tiers: [TierQueue<T>; PriorityTier::COUNT],
    /// Default starvation threshold in frames.
    default_threshold: u64,
}

impl<T> TieredQueue<T> {
    /// Creates a new tiered queue with a default starvation threshold of 60 frames.
    pub fn new() -> Self {
        Self::with_threshold(60)
    }

    /// Creates a new tiered queue with a custom starvation threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Number of frames before items are promoted to a higher tier.
    pub fn with_threshold(threshold: u64) -> Self {
        Self {
            tiers: [
                TierQueue::new(),
                TierQueue::new(),
                TierQueue::new(),
                TierQueue::new(),
            ],
            default_threshold: threshold,
        }
    }

    /// Returns the default starvation threshold.
    #[inline]
    pub fn threshold(&self) -> u64 {
        self.default_threshold
    }

    /// Enqueues an item at the specified priority tier.
    ///
    /// # Arguments
    ///
    /// * `item` - The item to enqueue.
    /// * `tier` - The priority tier for this item.
    /// * `current_frame` - The current frame number for starvation tracking.
    ///
    /// # Performance
    ///
    /// O(1) amortized time complexity.
    #[inline]
    pub fn enqueue(&self, item: T, tier: PriorityTier, current_frame: u64) {
        let queue_item = QueueItem::new(item, tier, current_frame);
        self.tiers[tier.index()].push_back(queue_item);
    }

    /// Dequeues the highest-priority item.
    ///
    /// Returns items in priority order: Critical > High > Normal > Low.
    /// Within a tier, items are returned in FIFO order.
    ///
    /// # Returns
    ///
    /// The highest-priority item, or `None` if the queue is empty.
    ///
    /// # Performance
    ///
    /// O(1) amortized time complexity.
    pub fn dequeue(&self) -> Option<QueueItem<T>> {
        for tier in &self.tiers {
            if let Some(item) = tier.pop_front() {
                return Some(item);
            }
        }
        None
    }

    /// Dequeues up to `count` items in priority order.
    ///
    /// # Arguments
    ///
    /// * `count` - Maximum number of items to dequeue.
    ///
    /// # Returns
    ///
    /// A vector of dequeued items, which may be shorter than `count` if
    /// there are fewer items in the queue.
    ///
    /// # Performance
    ///
    /// O(count) time complexity.
    pub fn dequeue_batch(&self, count: usize) -> Vec<QueueItem<T>> {
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            match self.dequeue() {
                Some(item) => result.push(item),
                None => break,
            }
        }
        result
    }

    /// Returns the total number of items across all tiers.
    ///
    /// # Note
    ///
    /// This requires locking all tiers sequentially, so the count may
    /// be approximate in concurrent scenarios.
    pub fn len(&self) -> usize {
        self.tiers.iter().map(|t| t.len()).sum()
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.tiers.iter().all(|t| t.is_empty())
    }

    /// Returns the number of items in a specific tier.
    #[inline]
    pub fn tier_len(&self, tier: PriorityTier) -> usize {
        self.tiers[tier.index()].len()
    }

    /// Promotes items that have been waiting longer than the threshold.
    ///
    /// Items are moved to the next higher tier (Low -> Normal -> High -> Critical).
    /// Items already at Critical tier cannot be promoted further.
    ///
    /// When an item is promoted, its `enqueue_frame` is reset to `current_frame`,
    /// so it must wait another threshold period before being promoted again.
    /// This prevents cascading promotions in a single call.
    ///
    /// # Arguments
    ///
    /// * `current_frame` - The current frame number.
    /// * `threshold` - Number of frames an item must wait before promotion.
    ///
    /// # Returns
    ///
    /// The number of items that were promoted.
    ///
    /// # Performance
    ///
    /// O(n) where n is the total number of items in the queue.
    pub fn promote_starving(&self, current_frame: u64, threshold: u64) -> usize {
        let mut promoted_count = 0;

        // Process tiers from lowest to highest priority (excluding Critical).
        // We go in reverse order to avoid items being promoted multiple tiers
        // in a single call (since we reset enqueue_frame on promotion).
        for tier_idx in (1..PriorityTier::COUNT).rev() {
            let current_tier = PriorityTier::from_index(tier_idx);
            let higher_tier = current_tier.promote().expect("non-critical tier has promotion target");

            // Collect items to promote.
            let mut to_promote = Vec::new();
            let mut to_keep = VecDeque::new();

            {
                let mut guard = self.tiers[tier_idx].lock();
                while let Some(mut item) = guard.pop_front() {
                    if item.frames_waiting(current_frame) >= threshold {
                        item.tier = higher_tier;
                        item.promoted = true;
                        // Reset enqueue_frame to prevent cascading promotions
                        // in the same call (item must wait again at new tier).
                        item.enqueue_frame = current_frame;
                        to_promote.push(item);
                        promoted_count += 1;
                    } else {
                        to_keep.push_back(item);
                    }
                }
                // Put back items that weren't promoted.
                *guard = to_keep;
            }

            // Add promoted items to the higher tier.
            let mut higher_guard = self.tiers[higher_tier.index()].lock();
            for item in to_promote {
                higher_guard.push_back(item);
            }
        }

        promoted_count
    }

    /// Promotes items using the default starvation threshold.
    ///
    /// Equivalent to `promote_starving(current_frame, self.threshold())`.
    #[inline]
    pub fn promote_starving_default(&self, current_frame: u64) -> usize {
        self.promote_starving(current_frame, self.default_threshold)
    }

    /// Clears all items from all tiers.
    pub fn clear(&self) {
        for tier in &self.tiers {
            tier.lock().clear();
        }
    }

    /// Returns statistics about queue state.
    pub fn stats(&self) -> QueueStats {
        QueueStats {
            critical_count: self.tier_len(PriorityTier::Critical),
            high_count: self.tier_len(PriorityTier::High),
            normal_count: self.tier_len(PriorityTier::Normal),
            low_count: self.tier_len(PriorityTier::Low),
        }
    }
}

impl<T> Default for TieredQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::fmt::Debug for TieredQueue<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TieredQueue")
            .field("total_len", &self.len())
            .field("threshold", &self.default_threshold)
            .field("stats", &self.stats())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// QueueStats
// ---------------------------------------------------------------------------

/// Statistics about the current state of a tiered queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueStats {
    /// Number of items in the Critical tier.
    pub critical_count: usize,
    /// Number of items in the High tier.
    pub high_count: usize,
    /// Number of items in the Normal tier.
    pub normal_count: usize,
    /// Number of items in the Low tier.
    pub low_count: usize,
}

impl QueueStats {
    /// Total number of items across all tiers.
    #[inline]
    pub fn total(&self) -> usize {
        self.critical_count + self.high_count + self.normal_count + self.low_count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // ── Basic Operations ────────────────────────────────────────────────

    #[test]
    fn enqueue_dequeue_single_item() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        queue.enqueue(42, PriorityTier::Normal, 0);

        let item = queue.dequeue().expect("should have item");
        assert_eq!(item.item, 42);
        assert_eq!(item.tier, PriorityTier::Normal);
        assert!(!item.promoted);
    }

    #[test]
    fn dequeue_empty_returns_none() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        assert!(queue.dequeue().is_none());
    }

    #[test]
    fn fifo_within_same_tier() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        queue.enqueue(1, PriorityTier::Normal, 0);
        queue.enqueue(2, PriorityTier::Normal, 0);
        queue.enqueue(3, PriorityTier::Normal, 0);

        assert_eq!(queue.dequeue().unwrap().item, 1);
        assert_eq!(queue.dequeue().unwrap().item, 2);
        assert_eq!(queue.dequeue().unwrap().item, 3);
    }

    // ── Priority Ordering ───────────────────────────────────────────────

    #[test]
    fn tier_priority_critical_before_high() {
        let queue: TieredQueue<&str> = TieredQueue::new();
        queue.enqueue("high", PriorityTier::High, 0);
        queue.enqueue("critical", PriorityTier::Critical, 0);

        assert_eq!(queue.dequeue().unwrap().item, "critical");
        assert_eq!(queue.dequeue().unwrap().item, "high");
    }

    #[test]
    fn tier_priority_high_before_normal() {
        let queue: TieredQueue<&str> = TieredQueue::new();
        queue.enqueue("normal", PriorityTier::Normal, 0);
        queue.enqueue("high", PriorityTier::High, 0);

        assert_eq!(queue.dequeue().unwrap().item, "high");
        assert_eq!(queue.dequeue().unwrap().item, "normal");
    }

    #[test]
    fn tier_priority_normal_before_low() {
        let queue: TieredQueue<&str> = TieredQueue::new();
        queue.enqueue("low", PriorityTier::Low, 0);
        queue.enqueue("normal", PriorityTier::Normal, 0);

        assert_eq!(queue.dequeue().unwrap().item, "normal");
        assert_eq!(queue.dequeue().unwrap().item, "low");
    }

    #[test]
    fn full_priority_order() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        // Enqueue in reverse priority order.
        queue.enqueue(4, PriorityTier::Low, 0);
        queue.enqueue(3, PriorityTier::Normal, 0);
        queue.enqueue(2, PriorityTier::High, 0);
        queue.enqueue(1, PriorityTier::Critical, 0);

        // Should come out in priority order.
        assert_eq!(queue.dequeue().unwrap().item, 1);
        assert_eq!(queue.dequeue().unwrap().item, 2);
        assert_eq!(queue.dequeue().unwrap().item, 3);
        assert_eq!(queue.dequeue().unwrap().item, 4);
    }

    // ── Starvation Prevention ───────────────────────────────────────────

    #[test]
    fn promote_starving_moves_items_up() {
        let queue: TieredQueue<u32> = TieredQueue::with_threshold(10);
        queue.enqueue(1, PriorityTier::Low, 0);

        // Not enough frames have passed.
        let promoted = queue.promote_starving(5, 10);
        assert_eq!(promoted, 0);
        assert_eq!(queue.tier_len(PriorityTier::Low), 1);
        assert_eq!(queue.tier_len(PriorityTier::Normal), 0);

        // Now enough frames have passed.
        let promoted = queue.promote_starving(10, 10);
        assert_eq!(promoted, 1);
        assert_eq!(queue.tier_len(PriorityTier::Low), 0);
        assert_eq!(queue.tier_len(PriorityTier::Normal), 1);
    }

    #[test]
    fn promoted_item_marked_as_promoted() {
        let queue: TieredQueue<u32> = TieredQueue::with_threshold(5);
        queue.enqueue(42, PriorityTier::Normal, 0);

        queue.promote_starving(10, 5);

        let item = queue.dequeue().unwrap();
        assert_eq!(item.item, 42);
        assert_eq!(item.tier, PriorityTier::High);
        assert!(item.promoted);
    }

    #[test]
    fn cannot_promote_past_critical() {
        let queue: TieredQueue<u32> = TieredQueue::with_threshold(5);
        queue.enqueue(1, PriorityTier::Critical, 0);

        let promoted = queue.promote_starving(100, 5);
        assert_eq!(promoted, 0);
        assert_eq!(queue.tier_len(PriorityTier::Critical), 1);
    }

    #[test]
    fn multi_tier_promotion() {
        let queue: TieredQueue<u32> = TieredQueue::with_threshold(5);
        queue.enqueue(1, PriorityTier::Low, 0);
        queue.enqueue(2, PriorityTier::Normal, 0);
        queue.enqueue(3, PriorityTier::High, 0);

        // First promotion: each item moves up one tier.
        let promoted = queue.promote_starving(10, 5);
        assert_eq!(promoted, 3);

        // Low -> Normal, Normal -> High, High -> Critical
        // Each item's enqueue_frame is reset to 10, preventing cascading.
        assert_eq!(queue.tier_len(PriorityTier::Critical), 1); // was High
        assert_eq!(queue.tier_len(PriorityTier::High), 1);     // was Normal
        assert_eq!(queue.tier_len(PriorityTier::Normal), 1);   // was Low
        assert_eq!(queue.tier_len(PriorityTier::Low), 0);

        // Second promotion after another threshold period.
        let promoted = queue.promote_starving(20, 5);
        assert_eq!(promoted, 2); // Critical item can't be promoted

        assert_eq!(queue.tier_len(PriorityTier::Critical), 2); // was High
        assert_eq!(queue.tier_len(PriorityTier::High), 1);     // was Normal
        assert_eq!(queue.tier_len(PriorityTier::Normal), 0);
        assert_eq!(queue.tier_len(PriorityTier::Low), 0);
    }

    // ── Batch Dequeue ───────────────────────────────────────────────────

    #[test]
    fn dequeue_batch_returns_requested_count() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        for i in 0..10 {
            queue.enqueue(i, PriorityTier::Normal, 0);
        }

        let batch = queue.dequeue_batch(5);
        assert_eq!(batch.len(), 5);
        assert_eq!(batch.iter().map(|i| i.item).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn dequeue_batch_returns_less_if_queue_smaller() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        queue.enqueue(1, PriorityTier::Normal, 0);
        queue.enqueue(2, PriorityTier::Normal, 0);

        let batch = queue.dequeue_batch(10);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn dequeue_batch_respects_priority() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        queue.enqueue(3, PriorityTier::Low, 0);
        queue.enqueue(2, PriorityTier::Normal, 0);
        queue.enqueue(1, PriorityTier::Critical, 0);

        let batch = queue.dequeue_batch(3);
        let items: Vec<u32> = batch.iter().map(|i| i.item).collect();
        assert_eq!(items, vec![1, 2, 3]);
    }

    // ── Length and Empty ────────────────────────────────────────────────

    #[test]
    fn len_counts_all_tiers() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        queue.enqueue(1, PriorityTier::Critical, 0);
        queue.enqueue(2, PriorityTier::High, 0);
        queue.enqueue(3, PriorityTier::Normal, 0);
        queue.enqueue(4, PriorityTier::Low, 0);

        assert_eq!(queue.len(), 4);
    }

    #[test]
    fn is_empty_when_empty() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        assert!(queue.is_empty());

        queue.enqueue(1, PriorityTier::Low, 0);
        assert!(!queue.is_empty());

        queue.dequeue();
        assert!(queue.is_empty());
    }

    // ── Concurrent Access ───────────────────────────────────────────────

    #[test]
    fn concurrent_enqueue_to_different_tiers() {
        let queue = Arc::new(TieredQueue::<u32>::new());
        let mut handles = vec![];

        for tier_idx in 0..4 {
            let q = Arc::clone(&queue);
            let tier = PriorityTier::from_index(tier_idx);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    q.enqueue(i, tier, 0);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(queue.len(), 400);
    }

    #[test]
    fn concurrent_enqueue_dequeue() {
        let queue = Arc::new(TieredQueue::<u32>::new());
        let enqueue_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let dequeue_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut handles = vec![];

        // Enqueue threads.
        for _ in 0..2 {
            let q = Arc::clone(&queue);
            let ec = Arc::clone(&enqueue_count);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    q.enqueue(i, PriorityTier::Normal, 0);
                    ec.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }));
        }

        // Dequeue threads.
        for _ in 0..2 {
            let q = Arc::clone(&queue);
            let dc = Arc::clone(&dequeue_count);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    if q.dequeue().is_some() {
                        dc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Drain remaining items.
        while queue.dequeue().is_some() {
            dequeue_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }

        assert_eq!(
            enqueue_count.load(std::sync::atomic::Ordering::SeqCst),
            dequeue_count.load(std::sync::atomic::Ordering::SeqCst)
        );
    }

    // ── Edge Cases ──────────────────────────────────────────────────────

    #[test]
    fn clear_removes_all_items() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        queue.enqueue(1, PriorityTier::Critical, 0);
        queue.enqueue(2, PriorityTier::Low, 0);

        queue.clear();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn stats_reports_tier_counts() {
        let queue: TieredQueue<u32> = TieredQueue::new();
        queue.enqueue(1, PriorityTier::Critical, 0);
        queue.enqueue(2, PriorityTier::High, 0);
        queue.enqueue(3, PriorityTier::High, 0);
        queue.enqueue(4, PriorityTier::Normal, 0);

        let stats = queue.stats();
        assert_eq!(stats.critical_count, 1);
        assert_eq!(stats.high_count, 2);
        assert_eq!(stats.normal_count, 1);
        assert_eq!(stats.low_count, 0);
        assert_eq!(stats.total(), 4);
    }

    #[test]
    fn frames_waiting_calculation() {
        let item = QueueItem::new(42, PriorityTier::Normal, 100);
        assert_eq!(item.frames_waiting(100), 0);
        assert_eq!(item.frames_waiting(150), 50);
        assert_eq!(item.frames_waiting(50), 0); // saturating_sub
    }

    #[test]
    fn priority_tier_promote_chain() {
        assert_eq!(PriorityTier::Low.promote(), Some(PriorityTier::Normal));
        assert_eq!(PriorityTier::Normal.promote(), Some(PriorityTier::High));
        assert_eq!(PriorityTier::High.promote(), Some(PriorityTier::Critical));
        assert_eq!(PriorityTier::Critical.promote(), None);
    }

    #[test]
    fn default_threshold_used() {
        let queue: TieredQueue<u32> = TieredQueue::with_threshold(20);
        assert_eq!(queue.threshold(), 20);

        queue.enqueue(1, PriorityTier::Low, 0);

        // Uses default threshold.
        let promoted = queue.promote_starving_default(15);
        assert_eq!(promoted, 0);

        let promoted = queue.promote_starving_default(25);
        assert_eq!(promoted, 1);
    }

    // ── Performance Benchmark ───────────────────────────────────────────

    #[test]
    fn benchmark_enqueue_dequeue_performance() {
        use std::time::Instant;

        let queue: TieredQueue<u64> = TieredQueue::new();
        let iterations = 10_000;

        // Benchmark enqueue.
        let start = Instant::now();
        for i in 0..iterations {
            queue.enqueue(i, PriorityTier::Normal, 0);
        }
        let enqueue_time = start.elapsed();
        let enqueue_per_op_ns = enqueue_time.as_nanos() / iterations as u128;

        // Benchmark dequeue.
        let start = Instant::now();
        for _ in 0..iterations {
            queue.dequeue();
        }
        let dequeue_time = start.elapsed();
        let dequeue_per_op_ns = dequeue_time.as_nanos() / iterations as u128;

        // Assert < 1us per operation (1000ns).
        // Note: CI environments may be slower, so we use a generous margin.
        assert!(
            enqueue_per_op_ns < 10_000,
            "enqueue too slow: {}ns per op",
            enqueue_per_op_ns
        );
        assert!(
            dequeue_per_op_ns < 10_000,
            "dequeue too slow: {}ns per op",
            dequeue_per_op_ns
        );

        // Print for visibility in test output.
        eprintln!(
            "Performance: enqueue={}ns/op, dequeue={}ns/op",
            enqueue_per_op_ns, dequeue_per_op_ns
        );
    }
}
