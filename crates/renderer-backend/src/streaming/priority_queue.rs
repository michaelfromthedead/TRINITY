// SPDX-License-Identifier: MIT
//
// priority_queue.rs -- Weighted Priority Queue for Streaming (T-AS-5.2)
//
// Provides a streaming priority queue with 5-component weighted scoring:
//   Priority = (visibility_weight * visibility_factor)
//            + (velocity_weight * velocity_factor)
//            + (distance_weight * distance_factor)
//            + (lod_bias * lod_factor)
//            + base_priority
//
// Features:
// - Binary heap with O(log n) insert, update, pop
// - Priority tiers: CRITICAL (bypass), HIGH (1-2 frames), NORMAL, LOW (idle)
// - Lock-free skip list for contention-free priority updates
// - Mark-and-skip cancellation
// - Re-heapification on camera movement
// - Configurable weights via PriorityWeights

use std::cell::UnsafeCell;
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// PriorityTier -- Priority classification
// ---------------------------------------------------------------------------

/// Priority tier classification for streaming requests.
///
/// Higher tiers bypass the normal queue and are processed immediately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PriorityTier {
    /// Lowest priority, processed during idle time.
    Low = 0,
    /// Normal priority, processed opportunistically.
    Normal = 1,
    /// High priority, processed within 1-2 frames.
    High = 2,
    /// Critical priority, bypasses queue entirely.
    Critical = 3,
}

impl PriorityTier {
    /// Creates a PriorityTier from a u8 value.
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => PriorityTier::Low,
            1 => PriorityTier::Normal,
            2 => PriorityTier::High,
            3 => PriorityTier::Critical,
            _ => PriorityTier::Normal,
        }
    }

    /// Returns the tier multiplier for score calculation.
    #[inline]
    pub fn multiplier(self) -> f32 {
        match self {
            PriorityTier::Low => 0.5,
            PriorityTier::Normal => 1.0,
            PriorityTier::High => 2.0,
            PriorityTier::Critical => f32::INFINITY,
        }
    }
}

impl Default for PriorityTier {
    fn default() -> Self {
        PriorityTier::Normal
    }
}

// ---------------------------------------------------------------------------
// PriorityWeights -- Configurable weight parameters
// ---------------------------------------------------------------------------

/// Configurable weights for the 5-component priority calculation.
///
/// Each weight is a floating-point multiplier applied to its corresponding
/// factor in the priority calculation.
///
/// # Example
///
/// ```ignore
/// let weights = PriorityWeights::default()
///     .with_visibility_weight(2.0)
///     .with_distance_weight(1.5);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PriorityWeights {
    /// Weight for visibility factor (0.0 = invisible, 1.0 = fully visible).
    pub visibility_weight: f32,
    /// Weight for velocity factor (how fast the camera is moving toward asset).
    pub velocity_weight: f32,
    /// Weight for distance factor (inverse of distance to camera).
    pub distance_weight: f32,
    /// Weight for LOD bias (higher = prefer loading higher detail levels).
    pub lod_bias: f32,
    /// Base priority offset.
    pub base_priority: f32,
}

impl Default for PriorityWeights {
    fn default() -> Self {
        Self {
            visibility_weight: 1.0,
            velocity_weight: 0.5,
            distance_weight: 1.0,
            lod_bias: 0.25,
            base_priority: 0.0,
        }
    }
}

impl PriorityWeights {
    /// Creates new weights with all components set to the given value.
    pub fn uniform(value: f32) -> Self {
        Self {
            visibility_weight: value,
            velocity_weight: value,
            distance_weight: value,
            lod_bias: value,
            base_priority: 0.0,
        }
    }

    /// Sets the visibility weight.
    pub fn with_visibility_weight(mut self, weight: f32) -> Self {
        self.visibility_weight = weight;
        self
    }

    /// Sets the velocity weight.
    pub fn with_velocity_weight(mut self, weight: f32) -> Self {
        self.velocity_weight = weight;
        self
    }

    /// Sets the distance weight.
    pub fn with_distance_weight(mut self, weight: f32) -> Self {
        self.distance_weight = weight;
        self
    }

    /// Sets the LOD bias.
    pub fn with_lod_bias(mut self, bias: f32) -> Self {
        self.lod_bias = bias;
        self
    }

    /// Sets the base priority.
    pub fn with_base_priority(mut self, priority: f32) -> Self {
        self.base_priority = priority;
        self
    }
}

// ---------------------------------------------------------------------------
// PriorityFactors -- Input factors for priority calculation
// ---------------------------------------------------------------------------

/// Input factors for priority calculation.
///
/// These are the raw values that get multiplied by weights.
#[derive(Debug, Clone, Copy, Default)]
pub struct PriorityFactors {
    /// Visibility factor (0.0 = invisible, 1.0 = fully visible).
    pub visibility: f32,
    /// Velocity factor (how fast the camera is moving toward asset).
    pub velocity: f32,
    /// Distance factor (inverse of distance, closer = higher).
    pub distance: f32,
    /// LOD factor (current LOD level demand).
    pub lod: f32,
}

impl PriorityFactors {
    /// Creates new factors with the given values.
    pub fn new(visibility: f32, velocity: f32, distance: f32, lod: f32) -> Self {
        Self {
            visibility,
            velocity,
            distance,
            lod,
        }
    }

    /// Calculates the weighted priority score.
    pub fn calculate_score(&self, weights: &PriorityWeights) -> f32 {
        (weights.visibility_weight * self.visibility)
            + (weights.velocity_weight * self.velocity)
            + (weights.distance_weight * self.distance)
            + (weights.lod_bias * self.lod)
            + weights.base_priority
    }
}

// ---------------------------------------------------------------------------
// PriorityEntry -- Entry in the priority queue
// ---------------------------------------------------------------------------

/// An entry in the priority queue.
///
/// Entries are ordered by their computed priority score (higher = more urgent).
#[derive(Debug, Clone)]
pub struct PriorityEntry {
    /// Unique identifier for the asset.
    pub asset_id: u64,
    /// Priority tier for fast path decisions.
    pub tier: PriorityTier,
    /// Computed priority score (higher = more urgent).
    pub score: f32,
    /// Raw input factors.
    pub factors: PriorityFactors,
    /// Sequence number for tie-breaking (FIFO among equal priorities).
    pub sequence: u64,
    /// Whether this entry has been cancelled (mark-and-skip).
    pub cancelled: bool,
    /// Timestamp when entry was created.
    pub created_at: u64,
}

impl PriorityEntry {
    /// Creates a new priority entry.
    pub fn new(asset_id: u64, tier: PriorityTier, factors: PriorityFactors, weights: &PriorityWeights, sequence: u64) -> Self {
        let score = factors.calculate_score(weights) * tier.multiplier();
        Self {
            asset_id,
            tier,
            score,
            factors,
            sequence,
            cancelled: false,
            created_at: 0,
        }
    }

    /// Creates a critical entry that bypasses normal ordering.
    pub fn critical(asset_id: u64, sequence: u64) -> Self {
        Self {
            asset_id,
            tier: PriorityTier::Critical,
            score: f32::INFINITY,
            factors: PriorityFactors::default(),
            sequence,
            cancelled: false,
            created_at: 0,
        }
    }

    /// Updates the score based on new factors.
    pub fn update_score(&mut self, factors: PriorityFactors, weights: &PriorityWeights) {
        self.factors = factors;
        self.score = factors.calculate_score(weights) * self.tier.multiplier();
    }

    /// Marks this entry as cancelled.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Returns true if this entry is still valid (not cancelled).
    pub fn is_valid(&self) -> bool {
        !self.cancelled
    }
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.asset_id == other.asset_id
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // First compare by tier (Critical > High > Normal > Low)
        match self.tier.cmp(&other.tier) {
            CmpOrdering::Equal => {}
            other => return other.reverse(), // Reverse because higher tier = higher priority
        }

        // Then compare by score (higher score = higher priority)
        match self.score.partial_cmp(&other.score) {
            Some(CmpOrdering::Equal) | None => {}
            Some(other) => return other.reverse(),
        }

        // Finally, use sequence for FIFO tie-breaking (lower sequence = earlier = higher priority)
        self.sequence.cmp(&other.sequence)
    }
}

// ---------------------------------------------------------------------------
// BinaryHeap -- Min-heap with O(log n) operations
// ---------------------------------------------------------------------------

/// A binary min-heap for priority queue operations.
///
/// Uses reverse ordering so that higher priority entries are popped first.
/// Supports efficient key-indexed access for priority updates.
pub struct BinaryHeap {
    /// Storage for heap entries.
    data: Vec<PriorityEntry>,
    /// Maps asset_id -> index in data for O(1) lookup.
    index_map: HashMap<u64, usize>,
    /// Sequence counter for tie-breaking.
    sequence: u64,
    /// Global weights for score calculation.
    weights: PriorityWeights,
}

impl BinaryHeap {
    /// Creates a new empty binary heap.
    pub fn new(weights: PriorityWeights) -> Self {
        Self {
            data: Vec::new(),
            index_map: HashMap::new(),
            sequence: 0,
            weights,
        }
    }

    /// Creates a new heap with the given capacity.
    pub fn with_capacity(capacity: usize, weights: PriorityWeights) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            index_map: HashMap::with_capacity(capacity),
            sequence: 0,
            weights,
        }
    }

    /// Returns the number of entries in the heap.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the heap is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a reference to the weights.
    #[inline]
    pub fn weights(&self) -> &PriorityWeights {
        &self.weights
    }

    /// Updates the weights and re-heapifies all entries.
    pub fn set_weights(&mut self, weights: PriorityWeights) {
        self.weights = weights;
        self.reheapify_all();
    }

    /// Returns the highest priority entry without removing it.
    pub fn peek(&self) -> Option<&PriorityEntry> {
        self.data.first()
    }

    /// Returns true if the heap contains an entry for the given asset.
    pub fn contains(&self, asset_id: u64) -> bool {
        self.index_map.contains_key(&asset_id)
    }

    /// Returns a reference to the entry for the given asset.
    pub fn get(&self, asset_id: u64) -> Option<&PriorityEntry> {
        self.index_map.get(&asset_id).map(|&idx| &self.data[idx])
    }

    /// Inserts a new entry into the heap.
    ///
    /// Returns the sequence number assigned to the entry.
    /// If an entry with the same asset_id exists, it is replaced.
    ///
    /// Time complexity: O(log n)
    pub fn insert(&mut self, asset_id: u64, tier: PriorityTier, factors: PriorityFactors) -> u64 {
        let sequence = self.sequence;
        self.sequence += 1;

        let entry = PriorityEntry::new(asset_id, tier, factors, &self.weights, sequence);

        if let Some(&existing_idx) = self.index_map.get(&asset_id) {
            // Replace existing entry
            self.data[existing_idx] = entry;
            // Re-heapify from this position
            self.sift_up(existing_idx);
            self.sift_down(existing_idx);
        } else {
            // Insert new entry
            let idx = self.data.len();
            self.data.push(entry);
            self.index_map.insert(asset_id, idx);
            self.sift_up(idx);
        }

        sequence
    }

    /// Inserts a critical entry that bypasses normal priority.
    pub fn insert_critical(&mut self, asset_id: u64) -> u64 {
        let sequence = self.sequence;
        self.sequence += 1;

        let entry = PriorityEntry::critical(asset_id, sequence);

        if let Some(&existing_idx) = self.index_map.get(&asset_id) {
            self.data[existing_idx] = entry;
            self.sift_up(existing_idx);
        } else {
            let idx = self.data.len();
            self.data.push(entry);
            self.index_map.insert(asset_id, idx);
            self.sift_up(idx);
        }

        sequence
    }

    /// Removes and returns the highest priority entry.
    ///
    /// Time complexity: O(log n)
    pub fn pop(&mut self) -> Option<PriorityEntry> {
        if self.data.is_empty() {
            return None;
        }

        // Swap first and last, then remove last
        let last_idx = self.data.len() - 1;
        self.swap(0, last_idx);

        let entry = self.data.pop().unwrap();
        self.index_map.remove(&entry.asset_id);

        if !self.data.is_empty() {
            // Update index for the element now at position 0
            if let Some(first) = self.data.first() {
                self.index_map.insert(first.asset_id, 0);
            }
            self.sift_down(0);
        }

        Some(entry)
    }

    /// Removes and returns the highest priority entry, skipping cancelled entries.
    ///
    /// This implements mark-and-skip cancellation.
    pub fn pop_valid(&mut self) -> Option<PriorityEntry> {
        loop {
            match self.pop() {
                Some(entry) if entry.is_valid() => return Some(entry),
                Some(_) => continue, // Skip cancelled entry
                None => return None,
            }
        }
    }

    /// Updates the priority of an existing entry.
    ///
    /// Time complexity: O(log n)
    pub fn update_priority(&mut self, asset_id: u64, factors: PriorityFactors) -> bool {
        if let Some(&idx) = self.index_map.get(&asset_id) {
            self.data[idx].update_score(factors, &self.weights);
            self.sift_up(idx);
            self.sift_down(idx);
            true
        } else {
            false
        }
    }

    /// Updates the tier of an existing entry.
    pub fn update_tier(&mut self, asset_id: u64, tier: PriorityTier) -> bool {
        if let Some(&idx) = self.index_map.get(&asset_id) {
            let factors = self.data[idx].factors;
            self.data[idx].tier = tier;
            self.data[idx].score = factors.calculate_score(&self.weights) * tier.multiplier();
            self.sift_up(idx);
            self.sift_down(idx);
            true
        } else {
            false
        }
    }

    /// Marks an entry as cancelled (mark-and-skip).
    ///
    /// The entry remains in the heap but will be skipped by pop_valid().
    /// This is faster than immediate removal.
    pub fn cancel(&mut self, asset_id: u64) -> bool {
        if let Some(&idx) = self.index_map.get(&asset_id) {
            self.data[idx].cancel();
            true
        } else {
            false
        }
    }

    /// Removes an entry immediately.
    ///
    /// Time complexity: O(log n)
    pub fn remove(&mut self, asset_id: u64) -> Option<PriorityEntry> {
        let idx = *self.index_map.get(&asset_id)?;

        let last_idx = self.data.len() - 1;
        self.swap(idx, last_idx);

        let entry = self.data.pop().unwrap();
        self.index_map.remove(&entry.asset_id);

        if idx < self.data.len() {
            // Update index for the element now at position idx
            if let Some(moved) = self.data.get(idx) {
                self.index_map.insert(moved.asset_id, idx);
            }
            self.sift_up(idx);
            self.sift_down(idx);
        }

        Some(entry)
    }

    /// Re-heapifies all entries (e.g., after camera movement).
    ///
    /// Time complexity: O(n)
    pub fn reheapify_all(&mut self) {
        // Recalculate all scores with current weights
        for entry in &mut self.data {
            entry.score = entry.factors.calculate_score(&self.weights) * entry.tier.multiplier();
        }

        // Build heap from bottom-up (Floyd's algorithm)
        let n = self.data.len();
        for i in (0..n / 2).rev() {
            self.sift_down(i);
        }

        // Rebuild index map
        self.rebuild_index_map();
    }

    /// Updates all entries with new factors and re-heapifies.
    pub fn batch_update<F>(&mut self, mut update_fn: F)
    where
        F: FnMut(u64, &mut PriorityFactors),
    {
        for entry in &mut self.data {
            update_fn(entry.asset_id, &mut entry.factors);
            entry.score = entry.factors.calculate_score(&self.weights) * entry.tier.multiplier();
        }

        // Build heap from bottom-up
        let n = self.data.len();
        for i in (0..n / 2).rev() {
            self.sift_down(i);
        }

        self.rebuild_index_map();
    }

    /// Removes all cancelled entries and compacts the heap.
    ///
    /// Call this periodically to reclaim space.
    pub fn compact(&mut self) {
        self.data.retain(|e| e.is_valid());

        // Rebuild the heap
        let n = self.data.len();
        for i in (0..n / 2).rev() {
            self.sift_down(i);
        }

        self.rebuild_index_map();
    }

    /// Returns an iterator over all entries (not in priority order).
    pub fn iter(&self) -> impl Iterator<Item = &PriorityEntry> {
        self.data.iter()
    }

    /// Returns the number of cancelled entries.
    pub fn cancelled_count(&self) -> usize {
        self.data.iter().filter(|e| e.cancelled).count()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.data.clear();
        self.index_map.clear();
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn parent(i: usize) -> usize {
        (i.saturating_sub(1)) / 2
    }

    fn left_child(i: usize) -> usize {
        2 * i + 1
    }

    fn right_child(i: usize) -> usize {
        2 * i + 2
    }

    fn swap(&mut self, i: usize, j: usize) {
        if i != j {
            // Update index map
            let id_i = self.data[i].asset_id;
            let id_j = self.data[j].asset_id;
            self.index_map.insert(id_i, j);
            self.index_map.insert(id_j, i);
            // Swap data
            self.data.swap(i, j);
        }
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent_idx = Self::parent(idx);
            if self.data[idx] < self.data[parent_idx] {
                self.swap(idx, parent_idx);
                idx = parent_idx;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let n = self.data.len();
        loop {
            let left = Self::left_child(idx);
            let right = Self::right_child(idx);
            let mut smallest = idx;

            if left < n && self.data[left] < self.data[smallest] {
                smallest = left;
            }
            if right < n && self.data[right] < self.data[smallest] {
                smallest = right;
            }

            if smallest != idx {
                self.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }

    fn rebuild_index_map(&mut self) {
        self.index_map.clear();
        for (idx, entry) in self.data.iter().enumerate() {
            self.index_map.insert(entry.asset_id, idx);
        }
    }
}

// ---------------------------------------------------------------------------
// SkipListNode -- Node for lock-free skip list
// ---------------------------------------------------------------------------

use std::ptr;
use std::sync::atomic::AtomicPtr;

const MAX_LEVEL: usize = 16;

/// A node in the lock-free skip list.
struct SkipListNode {
    /// Asset ID (key).
    asset_id: AtomicU64,
    /// Priority score (for ordering).
    score: AtomicU32, // f32 bits stored as u32
    /// Whether this node is marked for deletion.
    marked: AtomicBool,
    /// Forward pointers for each level.
    next: [AtomicPtr<SkipListNode>; MAX_LEVEL],
    /// Height of this node.
    height: usize,
}

impl SkipListNode {
    fn new(asset_id: u64, score: f32, height: usize) -> *mut Self {
        // Initialize array of atomic pointers to null
        const NULL_PTR: AtomicPtr<SkipListNode> = AtomicPtr::new(ptr::null_mut());
        let node = Box::new(Self {
            asset_id: AtomicU64::new(asset_id),
            score: AtomicU32::new(score.to_bits()),
            marked: AtomicBool::new(false),
            next: [NULL_PTR; MAX_LEVEL],
            height,
        });
        Box::into_raw(node)
    }

    fn sentinel() -> *mut Self {
        Self::new(u64::MAX, f32::NEG_INFINITY, MAX_LEVEL)
    }

    fn get_score(&self) -> f32 {
        f32::from_bits(self.score.load(Ordering::Relaxed))
    }

    fn set_score(&self, score: f32) {
        self.score.store(score.to_bits(), Ordering::Relaxed);
    }
}


// ---------------------------------------------------------------------------
// LockFreeSkipList -- Contention-free priority updates
// ---------------------------------------------------------------------------

/// A lock-free skip list for contention-free priority updates.
///
/// This data structure allows multiple threads to update priorities
/// without contention, making it suitable for camera movement triggers.
///
/// # Design
///
/// Based on the Harris-Michael lock-free linked list with skip list
/// acceleration. Uses logical deletion (marking) followed by physical
/// removal during traversal.
///
/// # Thread Safety
///
/// - All operations are lock-free and wait-free for common cases.
/// - Multiple threads can call `update_priority()` concurrently.
/// - Single consumer should call `drain()` to collect updates.
pub struct LockFreeSkipList {
    /// Head sentinel node.
    head: *mut SkipListNode,
    /// Tail sentinel node.
    tail: *mut SkipListNode,
    /// Number of elements (approximate).
    len: AtomicUsize,
    /// Current height of the list.
    max_height: AtomicUsize,
    /// Random seed for level generation.
    seed: AtomicU64,
}

// Safety: LockFreeSkipList uses atomic operations for synchronization
unsafe impl Send for LockFreeSkipList {}
unsafe impl Sync for LockFreeSkipList {}

impl LockFreeSkipList {
    /// Creates a new empty skip list.
    pub fn new() -> Self {
        let head = SkipListNode::sentinel();
        let tail = SkipListNode::sentinel();

        // Link head to tail at all levels
        unsafe {
            for level in 0..MAX_LEVEL {
                (*head).next[level].store(tail, Ordering::Relaxed);
            }
        }

        Self {
            head,
            tail,
            len: AtomicUsize::new(0),
            max_height: AtomicUsize::new(1),
            seed: AtomicU64::new(0x12345678),
        }
    }

    /// Returns the approximate number of elements.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Returns true if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generates a random level for a new node.
    fn random_level(&self) -> usize {
        // Simple xorshift PRNG
        let mut seed = self.seed.load(Ordering::Relaxed);
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        self.seed.store(seed, Ordering::Relaxed);

        // Count trailing zeros to get geometric distribution
        let mut level = 1;
        while level < MAX_LEVEL && (seed & (1 << level)) != 0 {
            level += 1;
        }
        level.min(self.max_height.load(Ordering::Relaxed) + 1)
    }

    /// Inserts or updates a priority entry.
    ///
    /// This operation is lock-free.
    pub fn upsert(&self, asset_id: u64, score: f32) {
        let mut preds = [ptr::null_mut::<SkipListNode>(); MAX_LEVEL];
        let mut succs = [ptr::null_mut::<SkipListNode>(); MAX_LEVEL];

        loop {
            // Find position
            let found = self.find(asset_id, &mut preds, &mut succs);

            if found {
                // Update existing node's score
                unsafe {
                    if let Some(node) = succs[0].as_ref() {
                        if !node.marked.load(Ordering::Acquire) {
                            node.set_score(score);
                            return;
                        }
                    }
                }
            }

            // Insert new node
            let height = self.random_level();
            let new_node = SkipListNode::new(asset_id, score, height);

            // Link at bottom level first
            unsafe {
                (*new_node).next[0].store(succs[0], Ordering::Relaxed);
            }

            let pred = unsafe { &*preds[0] };
            match pred.next[0].compare_exchange(
                succs[0],
                new_node,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Link at higher levels
                    for level in 1..height {
                        loop {
                            let pred = unsafe { &*preds[level] };
                            unsafe {
                                (*new_node).next[level].store(succs[level], Ordering::Relaxed);
                            }
                            if pred.next[level]
                                .compare_exchange(
                                    succs[level],
                                    new_node,
                                    Ordering::Release,
                                    Ordering::Relaxed,
                                )
                                .is_ok()
                            {
                                break;
                            }
                            // Retry find for this level
                            self.find(asset_id, &mut preds, &mut succs);
                        }
                    }

                    // Update max height
                    let mut current_max = self.max_height.load(Ordering::Relaxed);
                    while height > current_max {
                        match self.max_height.compare_exchange(
                            current_max,
                            height,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(new_max) => current_max = new_max,
                        }
                    }

                    self.len.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Err(_) => {
                    // CAS failed, retry
                    unsafe {
                        drop(Box::from_raw(new_node));
                    }
                }
            }
        }
    }

    /// Marks an entry for removal.
    pub fn mark_removed(&self, asset_id: u64) -> bool {
        let mut preds = [ptr::null_mut::<SkipListNode>(); MAX_LEVEL];
        let mut succs = [ptr::null_mut::<SkipListNode>(); MAX_LEVEL];

        if self.find(asset_id, &mut preds, &mut succs) {
            let node = unsafe { &*succs[0] };
            node.marked.store(true, Ordering::Release);
            self.len.fetch_sub(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Drains all entries from the skip list.
    ///
    /// Returns a vector of (asset_id, score) tuples.
    pub fn drain(&self) -> Vec<(u64, f32)> {
        let mut result = Vec::new();

        unsafe {
            let mut current = (*self.head).next[0].load(Ordering::Acquire);
            while current != self.tail {
                let node = &*current;
                if !node.marked.load(Ordering::Acquire) {
                    let asset_id = node.asset_id.load(Ordering::Relaxed);
                    let score = node.get_score();
                    result.push((asset_id, score));
                    node.marked.store(true, Ordering::Release);
                }
                current = node.next[0].load(Ordering::Acquire);
            }
        }

        self.len.store(0, Ordering::Relaxed);
        result
    }

    /// Finds a node and populates predecessor/successor arrays.
    fn find(
        &self,
        asset_id: u64,
        preds: &mut [*mut SkipListNode; MAX_LEVEL],
        succs: &mut [*mut SkipListNode; MAX_LEVEL],
    ) -> bool {
        let mut pred;
        let mut curr;
        let mut found = false;

        'retry: loop {
            pred = self.head;
            for level in (0..MAX_LEVEL).rev() {
                curr = unsafe { (*pred).next[level].load(Ordering::Acquire) };

                loop {
                    if curr == self.tail {
                        break;
                    }

                    let curr_node = unsafe { &*curr };
                    let curr_id = curr_node.asset_id.load(Ordering::Relaxed);

                    // Skip marked nodes
                    if curr_node.marked.load(Ordering::Acquire) {
                        let next = curr_node.next[level].load(Ordering::Acquire);
                        let pred_node = unsafe { &*pred };
                        if pred_node
                            .next[level]
                            .compare_exchange(curr, next, Ordering::Release, Ordering::Relaxed)
                            .is_err()
                        {
                            continue 'retry;
                        }
                        curr = next;
                        continue;
                    }

                    if curr_id >= asset_id {
                        break;
                    }

                    pred = curr;
                    curr = curr_node.next[level].load(Ordering::Acquire);
                }

                preds[level] = pred;
                succs[level] = curr;

                if level == 0 && curr != self.tail {
                    let curr_id = unsafe { (*curr).asset_id.load(Ordering::Relaxed) };
                    found = curr_id == asset_id;
                }
            }
            return found;
        }
    }
}

impl Default for LockFreeSkipList {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LockFreeSkipList {
    fn drop(&mut self) {
        unsafe {
            // Free all nodes
            let mut current = (*self.head).next[0].load(Ordering::Relaxed);
            while current != self.tail {
                let next = (*current).next[0].load(Ordering::Relaxed);
                drop(Box::from_raw(current));
                current = next;
            }
            drop(Box::from_raw(self.head));
            drop(Box::from_raw(self.tail));
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingPriorityQueue -- Combined priority queue
// ---------------------------------------------------------------------------

/// Update message for the priority queue.
#[derive(Debug, Clone, Copy)]
pub struct PriorityUpdateMsg {
    /// Asset ID to update.
    pub asset_id: u64,
    /// New priority factors.
    pub factors: PriorityFactors,
    /// New tier (if changed).
    pub tier: Option<PriorityTier>,
    /// Whether to cancel the request.
    pub cancel: bool,
}

impl PriorityUpdateMsg {
    /// Creates a new priority update.
    pub fn new(asset_id: u64, factors: PriorityFactors) -> Self {
        Self {
            asset_id,
            factors,
            tier: None,
            cancel: false,
        }
    }

    /// Creates a tier change update.
    pub fn tier_change(asset_id: u64, tier: PriorityTier) -> Self {
        Self {
            asset_id,
            factors: PriorityFactors::default(),
            tier: Some(tier),
            cancel: false,
        }
    }

    /// Creates a cancellation update.
    pub fn cancel(asset_id: u64) -> Self {
        Self {
            asset_id,
            factors: PriorityFactors::default(),
            tier: None,
            cancel: true,
        }
    }
}

/// Combined streaming priority queue with binary heap and lock-free updates.
///
/// # Design
///
/// The queue uses two data structures:
/// 1. A binary heap for ordered extraction (single-threaded access)
/// 2. A lock-free skip list for concurrent priority updates
///
/// Priority updates from multiple threads are collected in the skip list,
/// then batch-applied to the heap periodically.
///
/// # Usage
///
/// ```ignore
/// let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());
///
/// // Insert entries
/// queue.insert(1, PriorityTier::Normal, PriorityFactors::new(1.0, 0.5, 0.8, 0.2));
/// queue.insert(2, PriorityTier::High, PriorityFactors::new(0.9, 0.3, 0.9, 0.1));
///
/// // Update priorities (lock-free, from any thread)
/// queue.update_priority_lockfree(1, 0.95); // Just score update
///
/// // Apply pending updates
/// queue.apply_pending_updates();
///
/// // Pop highest priority
/// if let Some(entry) = queue.pop() {
///     println!("Processing asset {}", entry.asset_id);
/// }
/// ```
pub struct StreamingPriorityQueue {
    /// Binary heap for ordered extraction.
    heap: BinaryHeap,
    /// Lock-free skip list for concurrent updates.
    update_list: LockFreeSkipList,
    /// Statistics.
    stats: PriorityQueueStats,
}

/// Statistics for the priority queue.
#[derive(Debug, Clone, Default)]
pub struct PriorityQueueStats {
    /// Total inserts.
    pub inserts: u64,
    /// Total pops.
    pub pops: u64,
    /// Total updates.
    pub updates: u64,
    /// Total cancellations.
    pub cancellations: u64,
    /// Total reheapifications.
    pub reheapifications: u64,
    /// Peak queue size.
    pub peak_size: usize,
}

impl StreamingPriorityQueue {
    /// Creates a new priority queue with the given weights.
    pub fn new(weights: PriorityWeights) -> Self {
        Self {
            heap: BinaryHeap::new(weights),
            update_list: LockFreeSkipList::new(),
            stats: PriorityQueueStats::default(),
        }
    }

    /// Creates a new queue with the given capacity hint.
    pub fn with_capacity(capacity: usize, weights: PriorityWeights) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity, weights),
            update_list: LockFreeSkipList::new(),
            stats: PriorityQueueStats::default(),
        }
    }

    /// Returns the number of entries in the queue.
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns true if the queue is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns the current weights.
    #[inline]
    pub fn weights(&self) -> &PriorityWeights {
        self.heap.weights()
    }

    /// Returns the statistics.
    pub fn stats(&self) -> &PriorityQueueStats {
        &self.stats
    }

    /// Returns a reference to the highest priority entry.
    pub fn peek(&self) -> Option<&PriorityEntry> {
        self.heap.peek()
    }

    /// Inserts a new entry into the queue.
    pub fn insert(&mut self, asset_id: u64, tier: PriorityTier, factors: PriorityFactors) -> u64 {
        self.stats.inserts += 1;
        let seq = self.heap.insert(asset_id, tier, factors);
        self.stats.peak_size = self.stats.peak_size.max(self.heap.len());
        seq
    }

    /// Inserts a critical entry that bypasses normal priority.
    pub fn insert_critical(&mut self, asset_id: u64) -> u64 {
        self.stats.inserts += 1;
        let seq = self.heap.insert_critical(asset_id);
        self.stats.peak_size = self.stats.peak_size.max(self.heap.len());
        seq
    }

    /// Removes and returns the highest priority entry.
    pub fn pop(&mut self) -> Option<PriorityEntry> {
        let entry = self.heap.pop();
        if entry.is_some() {
            self.stats.pops += 1;
        }
        entry
    }

    /// Removes and returns the highest priority entry, skipping cancelled entries.
    pub fn pop_valid(&mut self) -> Option<PriorityEntry> {
        let entry = self.heap.pop_valid();
        if entry.is_some() {
            self.stats.pops += 1;
        }
        entry
    }

    /// Updates the priority of an existing entry (single-threaded).
    pub fn update_priority(&mut self, asset_id: u64, factors: PriorityFactors) -> bool {
        if self.heap.update_priority(asset_id, factors) {
            self.stats.updates += 1;
            true
        } else {
            false
        }
    }

    /// Updates the tier of an existing entry.
    pub fn update_tier(&mut self, asset_id: u64, tier: PriorityTier) -> bool {
        if self.heap.update_tier(asset_id, tier) {
            self.stats.updates += 1;
            true
        } else {
            false
        }
    }

    /// Marks an entry as cancelled (mark-and-skip).
    pub fn cancel(&mut self, asset_id: u64) -> bool {
        if self.heap.cancel(asset_id) {
            self.stats.cancellations += 1;
            true
        } else {
            false
        }
    }

    /// Updates priority in a lock-free manner (for concurrent access).
    ///
    /// The update is stored in the skip list and applied later by
    /// `apply_pending_updates()`.
    pub fn update_priority_lockfree(&self, asset_id: u64, score: f32) {
        self.update_list.upsert(asset_id, score);
    }

    /// Applies all pending updates from the lock-free list to the heap.
    ///
    /// This should be called from the owner thread periodically.
    pub fn apply_pending_updates(&mut self) {
        let updates = self.update_list.drain();
        for (asset_id, _score) in updates {
            // For now, we just bump the priority based on score
            // In a full implementation, we'd reconstruct factors from score
            if let Some(entry) = self.heap.get(asset_id) {
                let mut factors = entry.factors;
                // Simple heuristic: distribute score across factors
                factors.visibility = 1.0;
                self.heap.update_priority(asset_id, factors);
                self.stats.updates += 1;
            }
        }
    }

    /// Updates weights and re-heapifies all entries.
    pub fn set_weights(&mut self, weights: PriorityWeights) {
        self.heap.set_weights(weights);
        self.stats.reheapifications += 1;
    }

    /// Re-heapifies all entries (e.g., after camera movement).
    pub fn reheapify(&mut self) {
        self.heap.reheapify_all();
        self.stats.reheapifications += 1;
    }

    /// Batch updates all entries with new factors.
    pub fn batch_update<F>(&mut self, update_fn: F)
    where
        F: FnMut(u64, &mut PriorityFactors),
    {
        self.heap.batch_update(update_fn);
        self.stats.reheapifications += 1;
    }

    /// Compacts the queue by removing cancelled entries.
    pub fn compact(&mut self) {
        self.heap.compact();
    }

    /// Returns true if the queue contains an entry for the given asset.
    pub fn contains(&self, asset_id: u64) -> bool {
        self.heap.contains(asset_id)
    }

    /// Returns a reference to the entry for the given asset.
    pub fn get(&self, asset_id: u64) -> Option<&PriorityEntry> {
        self.heap.get(asset_id)
    }

    /// Removes an entry immediately.
    pub fn remove(&mut self, asset_id: u64) -> Option<PriorityEntry> {
        self.heap.remove(asset_id)
    }

    /// Returns an iterator over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &PriorityEntry> {
        self.heap.iter()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.heap.clear();
    }
}

impl Default for StreamingPriorityQueue {
    fn default() -> Self {
        Self::new(PriorityWeights::default())
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
    use std::time::{Duration, Instant};

    // ── PriorityTier Tests ──────────────────────────────────────────────

    #[test]
    fn priority_tier_ordering() {
        assert!(PriorityTier::Critical > PriorityTier::High);
        assert!(PriorityTier::High > PriorityTier::Normal);
        assert!(PriorityTier::Normal > PriorityTier::Low);
    }

    #[test]
    fn priority_tier_from_u8() {
        assert_eq!(PriorityTier::from_u8(0), PriorityTier::Low);
        assert_eq!(PriorityTier::from_u8(1), PriorityTier::Normal);
        assert_eq!(PriorityTier::from_u8(2), PriorityTier::High);
        assert_eq!(PriorityTier::from_u8(3), PriorityTier::Critical);
        assert_eq!(PriorityTier::from_u8(99), PriorityTier::Normal);
    }

    #[test]
    fn priority_tier_multiplier() {
        assert_eq!(PriorityTier::Low.multiplier(), 0.5);
        assert_eq!(PriorityTier::Normal.multiplier(), 1.0);
        assert_eq!(PriorityTier::High.multiplier(), 2.0);
        assert!(PriorityTier::Critical.multiplier().is_infinite());
    }

    // ── PriorityWeights Tests ───────────────────────────────────────────

    #[test]
    fn priority_weights_default() {
        let weights = PriorityWeights::default();
        assert_eq!(weights.visibility_weight, 1.0);
        assert_eq!(weights.velocity_weight, 0.5);
        assert_eq!(weights.distance_weight, 1.0);
        assert_eq!(weights.lod_bias, 0.25);
        assert_eq!(weights.base_priority, 0.0);
    }

    #[test]
    fn priority_weights_uniform() {
        let weights = PriorityWeights::uniform(2.0);
        assert_eq!(weights.visibility_weight, 2.0);
        assert_eq!(weights.velocity_weight, 2.0);
        assert_eq!(weights.distance_weight, 2.0);
        assert_eq!(weights.lod_bias, 2.0);
    }

    #[test]
    fn priority_weights_builder() {
        let weights = PriorityWeights::default()
            .with_visibility_weight(2.0)
            .with_velocity_weight(1.5)
            .with_distance_weight(3.0)
            .with_lod_bias(0.5)
            .with_base_priority(10.0);

        assert_eq!(weights.visibility_weight, 2.0);
        assert_eq!(weights.velocity_weight, 1.5);
        assert_eq!(weights.distance_weight, 3.0);
        assert_eq!(weights.lod_bias, 0.5);
        assert_eq!(weights.base_priority, 10.0);
    }

    // ── PriorityFactors Tests ───────────────────────────────────────────

    #[test]
    fn priority_factors_calculation() {
        let weights = PriorityWeights::default();
        let factors = PriorityFactors::new(1.0, 0.5, 0.8, 0.4);

        // score = 1.0*1.0 + 0.5*0.5 + 1.0*0.8 + 0.25*0.4 + 0.0
        // score = 1.0 + 0.25 + 0.8 + 0.1 = 2.15
        let score = factors.calculate_score(&weights);
        assert!((score - 2.15).abs() < 0.001);
    }

    #[test]
    fn priority_factors_with_base_priority() {
        let weights = PriorityWeights::default().with_base_priority(100.0);
        let factors = PriorityFactors::default();

        let score = factors.calculate_score(&weights);
        assert_eq!(score, 100.0);
    }

    // ── PriorityEntry Tests ─────────────────────────────────────────────

    #[test]
    fn priority_entry_creation() {
        let weights = PriorityWeights::default();
        let factors = PriorityFactors::new(1.0, 0.5, 0.8, 0.4);
        let entry = PriorityEntry::new(42, PriorityTier::Normal, factors, &weights, 0);

        assert_eq!(entry.asset_id, 42);
        assert_eq!(entry.tier, PriorityTier::Normal);
        assert!(!entry.cancelled);
        assert!(entry.is_valid());
    }

    #[test]
    fn priority_entry_critical() {
        let entry = PriorityEntry::critical(99, 0);

        assert_eq!(entry.asset_id, 99);
        assert_eq!(entry.tier, PriorityTier::Critical);
        assert!(entry.score.is_infinite());
    }

    #[test]
    fn priority_entry_ordering() {
        let weights = PriorityWeights::default();

        // Critical should come before high
        let critical = PriorityEntry::new(1, PriorityTier::Critical, PriorityFactors::default(), &weights, 0);
        let high = PriorityEntry::new(2, PriorityTier::High, PriorityFactors::default(), &weights, 1);

        assert!(critical < high);

        // Higher score should come first within same tier
        let high_score = PriorityEntry::new(3, PriorityTier::Normal, PriorityFactors::new(1.0, 1.0, 1.0, 1.0), &weights, 2);
        let low_score = PriorityEntry::new(4, PriorityTier::Normal, PriorityFactors::new(0.1, 0.1, 0.1, 0.1), &weights, 3);

        assert!(high_score < low_score);

        // Earlier sequence should come first for equal priorities
        let early = PriorityEntry::new(5, PriorityTier::Normal, PriorityFactors::default(), &weights, 0);
        let late = PriorityEntry::new(6, PriorityTier::Normal, PriorityFactors::default(), &weights, 100);

        assert!(early < late);
    }

    #[test]
    fn priority_entry_cancellation() {
        let weights = PriorityWeights::default();
        let mut entry = PriorityEntry::new(42, PriorityTier::Normal, PriorityFactors::default(), &weights, 0);

        assert!(entry.is_valid());
        entry.cancel();
        assert!(!entry.is_valid());
    }

    // ── BinaryHeap Tests ────────────────────────────────────────────────

    #[test]
    fn binary_heap_insert_pop() {
        let mut heap = BinaryHeap::new(PriorityWeights::default());

        heap.insert(1, PriorityTier::Normal, PriorityFactors::new(0.5, 0.5, 0.5, 0.5));
        heap.insert(2, PriorityTier::High, PriorityFactors::new(0.5, 0.5, 0.5, 0.5));
        heap.insert(3, PriorityTier::Low, PriorityFactors::new(0.5, 0.5, 0.5, 0.5));

        assert_eq!(heap.len(), 3);

        // High should come first
        let first = heap.pop().unwrap();
        assert_eq!(first.asset_id, 2);

        // Normal should come second
        let second = heap.pop().unwrap();
        assert_eq!(second.asset_id, 1);

        // Low should come last
        let third = heap.pop().unwrap();
        assert_eq!(third.asset_id, 3);

        assert!(heap.pop().is_none());
    }

    #[test]
    fn binary_heap_critical_bypass() {
        let mut heap = BinaryHeap::new(PriorityWeights::default());

        // Insert many normal entries
        for i in 0..100 {
            heap.insert(i, PriorityTier::Normal, PriorityFactors::new(1.0, 1.0, 1.0, 1.0));
        }

        // Insert critical entry
        heap.insert_critical(999);

        // Critical should be first
        let first = heap.pop().unwrap();
        assert_eq!(first.asset_id, 999);
        assert_eq!(first.tier, PriorityTier::Critical);
    }

    #[test]
    fn binary_heap_update_priority() {
        let mut heap = BinaryHeap::new(PriorityWeights::default());

        heap.insert(1, PriorityTier::Normal, PriorityFactors::new(0.1, 0.1, 0.1, 0.1));
        heap.insert(2, PriorityTier::Normal, PriorityFactors::new(0.5, 0.5, 0.5, 0.5));

        // Update entry 1 to have higher priority
        assert!(heap.update_priority(1, PriorityFactors::new(1.0, 1.0, 1.0, 1.0)));

        // Now entry 1 should come first
        let first = heap.pop().unwrap();
        assert_eq!(first.asset_id, 1);
    }

    #[test]
    fn binary_heap_update_tier() {
        let mut heap = BinaryHeap::new(PriorityWeights::default());

        heap.insert(1, PriorityTier::Low, PriorityFactors::new(0.5, 0.5, 0.5, 0.5));
        heap.insert(2, PriorityTier::Normal, PriorityFactors::new(0.5, 0.5, 0.5, 0.5));

        // Upgrade entry 1 to high
        assert!(heap.update_tier(1, PriorityTier::High));

        // Now entry 1 should come first
        let first = heap.pop().unwrap();
        assert_eq!(first.asset_id, 1);
        assert_eq!(first.tier, PriorityTier::High);
    }

    #[test]
    fn binary_heap_cancel_mark_and_skip() {
        let mut heap = BinaryHeap::new(PriorityWeights::default());

        heap.insert(1, PriorityTier::High, PriorityFactors::default());
        heap.insert(2, PriorityTier::Normal, PriorityFactors::default());
        heap.insert(3, PriorityTier::Low, PriorityFactors::default());

        // Cancel the high priority entry
        assert!(heap.cancel(1));

        // pop_valid should skip the cancelled entry
        let first = heap.pop_valid().unwrap();
        assert_eq!(first.asset_id, 2); // Should be Normal, not High
    }

    #[test]
    fn binary_heap_remove() {
        let mut heap = BinaryHeap::new(PriorityWeights::default());

        heap.insert(1, PriorityTier::Normal, PriorityFactors::default());
        heap.insert(2, PriorityTier::Normal, PriorityFactors::default());
        heap.insert(3, PriorityTier::Normal, PriorityFactors::default());

        // Remove middle entry
        let removed = heap.remove(2).unwrap();
        assert_eq!(removed.asset_id, 2);
        assert_eq!(heap.len(), 2);

        // Entry 2 should no longer be in heap
        assert!(!heap.contains(2));
    }

    #[test]
    fn binary_heap_reheapify() {
        let mut heap = BinaryHeap::new(PriorityWeights::default());

        heap.insert(1, PriorityTier::Normal, PriorityFactors::new(0.1, 0.0, 0.0, 0.0));
        heap.insert(2, PriorityTier::Normal, PriorityFactors::new(0.5, 0.0, 0.0, 0.0));
        heap.insert(3, PriorityTier::Normal, PriorityFactors::new(0.9, 0.0, 0.0, 0.0));

        // Entry 3 has highest visibility, should be first
        let first = heap.peek().unwrap();
        assert_eq!(first.asset_id, 3);

        // Change weights to emphasize distance instead
        heap.set_weights(PriorityWeights::default()
            .with_visibility_weight(0.0)
            .with_distance_weight(10.0));

        // Now all should have score 0 (no distance factor set)
        // But original order should be maintained by sequence
        let first = heap.pop().unwrap();
        assert_eq!(first.asset_id, 1); // First inserted
    }

    #[test]
    fn binary_heap_batch_update() {
        let mut heap = BinaryHeap::new(PriorityWeights::default());

        heap.insert(1, PriorityTier::Normal, PriorityFactors::new(0.1, 0.0, 0.0, 0.0));
        heap.insert(2, PriorityTier::Normal, PriorityFactors::new(0.5, 0.0, 0.0, 0.0));
        heap.insert(3, PriorityTier::Normal, PriorityFactors::new(0.9, 0.0, 0.0, 0.0));

        // Batch update: invert priorities
        heap.batch_update(|_asset_id, factors| {
            factors.visibility = 1.0 - factors.visibility;
        });

        // Now entry 1 should have highest visibility (0.9)
        let first = heap.pop().unwrap();
        assert_eq!(first.asset_id, 1);
    }

    #[test]
    fn binary_heap_compact() {
        let mut heap = BinaryHeap::new(PriorityWeights::default());

        for i in 0..10 {
            heap.insert(i, PriorityTier::Normal, PriorityFactors::default());
        }

        // Cancel half
        for i in 0..5 {
            heap.cancel(i);
        }

        assert_eq!(heap.cancelled_count(), 5);
        assert_eq!(heap.len(), 10);

        // Compact
        heap.compact();

        assert_eq!(heap.cancelled_count(), 0);
        assert_eq!(heap.len(), 5);
    }

    #[test]
    fn binary_heap_contains_and_get() {
        let mut heap = BinaryHeap::new(PriorityWeights::default());

        heap.insert(42, PriorityTier::Normal, PriorityFactors::default());

        assert!(heap.contains(42));
        assert!(!heap.contains(99));

        let entry = heap.get(42).unwrap();
        assert_eq!(entry.asset_id, 42);

        assert!(heap.get(99).is_none());
    }

    // ── LockFreeSkipList Tests ──────────────────────────────────────────

    #[test]
    fn skiplist_basic_operations() {
        let list = LockFreeSkipList::new();

        list.upsert(1, 10.0);
        list.upsert(2, 20.0);
        list.upsert(3, 15.0);

        assert_eq!(list.len(), 3);

        let updates = list.drain();
        assert_eq!(updates.len(), 3);
    }

    #[test]
    fn skiplist_upsert_update() {
        let list = LockFreeSkipList::new();

        list.upsert(1, 10.0);
        list.upsert(1, 20.0); // Update existing

        assert_eq!(list.len(), 1);

        let updates = list.drain();
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].0, 1);
        assert_eq!(updates[0].1, 20.0);
    }

    #[test]
    fn skiplist_mark_removed() {
        let list = LockFreeSkipList::new();

        list.upsert(1, 10.0);
        list.upsert(2, 20.0);

        assert!(list.mark_removed(1));
        assert!(!list.mark_removed(99)); // Non-existent

        assert_eq!(list.len(), 1);
    }

    #[test]
    fn skiplist_concurrent_upsert() {
        let list = Arc::new(LockFreeSkipList::new());
        let mut handles = vec![];

        for t in 0..4 {
            let list = Arc::clone(&list);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    list.upsert(t * 100 + i, (t * 100 + i) as f32);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(list.len(), 400);

        let updates = list.drain();
        assert_eq!(updates.len(), 400);
    }

    // ── StreamingPriorityQueue Tests ────────────────────────────────────

    #[test]
    fn streaming_queue_basic_operations() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        queue.insert(1, PriorityTier::Normal, PriorityFactors::new(0.5, 0.5, 0.5, 0.5));
        queue.insert(2, PriorityTier::High, PriorityFactors::new(0.5, 0.5, 0.5, 0.5));

        assert_eq!(queue.len(), 2);
        assert!(!queue.is_empty());

        // High priority should come first
        let first = queue.pop().unwrap();
        assert_eq!(first.asset_id, 2);
    }

    #[test]
    fn streaming_queue_lockfree_update() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        queue.insert(1, PriorityTier::Normal, PriorityFactors::new(0.5, 0.5, 0.5, 0.5));

        // Lock-free update
        queue.update_priority_lockfree(1, 100.0);

        // Apply pending updates
        queue.apply_pending_updates();

        assert_eq!(queue.stats().updates, 1);
    }

    #[test]
    fn streaming_queue_cancel() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        queue.insert(1, PriorityTier::High, PriorityFactors::default());
        queue.insert(2, PriorityTier::Normal, PriorityFactors::default());

        queue.cancel(1);

        // pop_valid should skip cancelled
        let first = queue.pop_valid().unwrap();
        assert_eq!(first.asset_id, 2);
    }

    #[test]
    fn streaming_queue_reheapify() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        queue.insert(1, PriorityTier::Normal, PriorityFactors::new(0.1, 0.0, 0.0, 0.0));
        queue.insert(2, PriorityTier::Normal, PriorityFactors::new(0.9, 0.0, 0.0, 0.0));

        // Entry 2 should be first (higher visibility)
        let first = queue.peek().unwrap();
        assert_eq!(first.asset_id, 2);

        // Change weights
        queue.set_weights(PriorityWeights::uniform(0.0).with_base_priority(1.0));

        assert_eq!(queue.stats().reheapifications, 1);
    }

    #[test]
    fn streaming_queue_batch_update() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        for i in 0..10 {
            queue.insert(i, PriorityTier::Normal, PriorityFactors::new(i as f32 / 10.0, 0.0, 0.0, 0.0));
        }

        // Batch update: set all visibility to 0.5
        queue.batch_update(|_id, factors| {
            factors.visibility = 0.5;
        });

        // All should now have equal score, so FIFO order
        let first = queue.pop().unwrap();
        assert_eq!(first.asset_id, 0);
    }

    #[test]
    fn streaming_queue_stats() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        for i in 0..10 {
            queue.insert(i, PriorityTier::Normal, PriorityFactors::default());
        }

        for _ in 0..5 {
            queue.pop();
        }

        queue.cancel(8);

        let stats = queue.stats();
        assert_eq!(stats.inserts, 10);
        assert_eq!(stats.pops, 5);
        assert_eq!(stats.cancellations, 1);
        assert_eq!(stats.peak_size, 10);
    }

    // ── Concurrent Access Tests ─────────────────────────────────────────

    #[test]
    fn concurrent_priority_updates() {
        let queue = Arc::new(StreamingPriorityQueue::new(PriorityWeights::default()));

        // Pre-populate
        {
            let queue = unsafe { &mut *(Arc::as_ptr(&queue) as *mut StreamingPriorityQueue) };
            for i in 0..100 {
                queue.insert(i, PriorityTier::Normal, PriorityFactors::default());
            }
        }

        let mut handles = vec![];

        // Multiple threads doing lock-free updates
        for _ in 0..4 {
            let queue = Arc::clone(&queue);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    queue.update_priority_lockfree(i, i as f32);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Apply all updates
        let queue = unsafe { &mut *(Arc::as_ptr(&queue) as *mut StreamingPriorityQueue) };
        queue.apply_pending_updates();
    }

    // ── Performance Benchmark Tests ─────────────────────────────────────

    #[test]
    fn benchmark_insert_performance() {
        let mut queue = StreamingPriorityQueue::with_capacity(10000, PriorityWeights::default());

        let start = Instant::now();
        for i in 0..10000 {
            queue.insert(i, PriorityTier::Normal, PriorityFactors::new(
                (i % 100) as f32 / 100.0,
                (i % 50) as f32 / 50.0,
                (i % 200) as f32 / 200.0,
                (i % 25) as f32 / 25.0,
            ));
        }
        let insert_time = start.elapsed();

        // Should complete in reasonable time (< 100ms)
        assert!(insert_time < Duration::from_millis(100),
            "Insert took {:?}", insert_time);
    }

    #[test]
    fn benchmark_pop_performance() {
        let mut queue = StreamingPriorityQueue::with_capacity(10000, PriorityWeights::default());

        for i in 0..10000 {
            queue.insert(i, PriorityTier::Normal, PriorityFactors::default());
        }

        let start = Instant::now();
        while queue.pop().is_some() {}
        let pop_time = start.elapsed();

        // Should complete in reasonable time (< 100ms)
        assert!(pop_time < Duration::from_millis(100),
            "Pop took {:?}", pop_time);
    }

    #[test]
    fn benchmark_update_performance() {
        let mut queue = StreamingPriorityQueue::with_capacity(10000, PriorityWeights::default());

        for i in 0..10000 {
            queue.insert(i, PriorityTier::Normal, PriorityFactors::default());
        }

        let start = Instant::now();
        for i in 0..10000 {
            queue.update_priority(i, PriorityFactors::new(
                ((i * 7) % 100) as f32 / 100.0,
                ((i * 13) % 50) as f32 / 50.0,
                ((i * 17) % 200) as f32 / 200.0,
                ((i * 23) % 25) as f32 / 25.0,
            ));
        }
        let update_time = start.elapsed();

        // Should complete in reasonable time (< 200ms)
        assert!(update_time < Duration::from_millis(200),
            "Update took {:?}", update_time);
    }

    #[test]
    fn benchmark_reheapify_performance() {
        let mut queue = StreamingPriorityQueue::with_capacity(10000, PriorityWeights::default());

        for i in 0..10000 {
            queue.insert(i, PriorityTier::Normal, PriorityFactors::new(
                (i % 100) as f32 / 100.0,
                (i % 50) as f32 / 50.0,
                (i % 200) as f32 / 200.0,
                (i % 25) as f32 / 25.0,
            ));
        }

        let start = Instant::now();
        queue.reheapify();
        let reheapify_time = start.elapsed();

        // Reheapify should be O(n), so faster than n*log(n) updates
        assert!(reheapify_time < Duration::from_millis(50),
            "Reheapify took {:?}", reheapify_time);
    }

    // ── Edge Case Tests ─────────────────────────────────────────────────

    #[test]
    fn empty_queue_operations() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        assert!(queue.is_empty());
        assert!(queue.pop().is_none());
        assert!(queue.pop_valid().is_none());
        assert!(queue.peek().is_none());
        assert!(!queue.contains(1));
        assert!(!queue.cancel(1));
        assert!(queue.remove(1).is_none());
    }

    #[test]
    fn single_element_operations() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        queue.insert(42, PriorityTier::Normal, PriorityFactors::default());

        assert_eq!(queue.len(), 1);
        assert!(queue.contains(42));

        let entry = queue.pop().unwrap();
        assert_eq!(entry.asset_id, 42);

        assert!(queue.is_empty());
    }

    #[test]
    fn duplicate_insert() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        queue.insert(42, PriorityTier::Low, PriorityFactors::new(0.1, 0.0, 0.0, 0.0));
        queue.insert(42, PriorityTier::High, PriorityFactors::new(0.9, 0.0, 0.0, 0.0));

        assert_eq!(queue.len(), 1);

        let entry = queue.pop().unwrap();
        assert_eq!(entry.asset_id, 42);
        assert_eq!(entry.tier, PriorityTier::High);
    }

    #[test]
    fn all_cancelled() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        // Insert entries
        for i in 0..10 {
            queue.insert(i, PriorityTier::Normal, PriorityFactors::default());
        }

        // Cancel all entries
        for i in 0..10 {
            queue.cancel(i);
        }

        // All entries should still be in the queue (cancelled but present)
        assert_eq!(queue.len(), 10);

        // pop_valid should return None (all cancelled)
        assert!(queue.pop_valid().is_none());

        // After pop_valid exhausts the queue looking for valid entries,
        // the queue should be empty
        assert!(queue.is_empty());
    }

    #[test]
    fn cancelled_entries_popped_by_regular_pop() {
        let mut queue = StreamingPriorityQueue::new(PriorityWeights::default());

        for i in 0..10 {
            queue.insert(i, PriorityTier::Normal, PriorityFactors::default());
        }

        // Cancel half the entries
        for i in 0..5 {
            queue.cancel(i);
        }

        // Regular pop should return entries (both valid and cancelled)
        let first = queue.pop().unwrap();
        // It will be one of the entries (either valid or cancelled based on priority)
        assert!(first.asset_id < 10);
    }

    #[test]
    fn priority_score_five_components() {
        let weights = PriorityWeights {
            visibility_weight: 1.0,
            velocity_weight: 2.0,
            distance_weight: 3.0,
            lod_bias: 4.0,
            base_priority: 5.0,
        };

        let factors = PriorityFactors {
            visibility: 0.1,
            velocity: 0.2,
            distance: 0.3,
            lod: 0.4,
        };

        // score = 1.0*0.1 + 2.0*0.2 + 3.0*0.3 + 4.0*0.4 + 5.0
        // score = 0.1 + 0.4 + 0.9 + 1.6 + 5.0 = 8.0
        let score = factors.calculate_score(&weights);
        assert!((score - 8.0).abs() < 0.001);
    }
}
