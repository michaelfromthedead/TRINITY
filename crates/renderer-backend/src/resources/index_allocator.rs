//! Generic index allocator with free list for efficient resource management.
//!
//! This module provides [`IndexAllocator`] for managing index allocation with recycling,
//! commonly used by bindless registries and other resource managers.
//!
//! # Features
//!
//! - **Free list recycling**: LIFO stack for cache-friendly index reuse
//! - **Capacity limits**: Configurable maximum index count
//! - **Double-free protection**: Detects and rejects duplicate frees
//! - **Generation tracking**: Optional validation to detect use-after-free
//!
//! # Example
//!
//! ```
//! use renderer_backend::resources::index_allocator::{IndexAllocator, GenerationalIndex};
//!
//! // Basic allocation
//! let mut allocator = IndexAllocator::new(1024);
//! let idx1 = allocator.allocate().unwrap();
//! let idx2 = allocator.allocate().unwrap();
//! assert_eq!(idx1, 0);
//! assert_eq!(idx2, 1);
//!
//! // Free and recycle
//! allocator.free(idx1);
//! let idx3 = allocator.allocate().unwrap(); // Returns 0 (recycled)
//!
//! // With generation tracking
//! let mut gen_allocator = IndexAllocator::with_generations(1024);
//! let gen_idx = gen_allocator.allocate_generational().unwrap();
//! gen_allocator.free(gen_idx.index);
//! let gen_idx2 = gen_allocator.allocate_generational().unwrap();
//! assert!(!gen_allocator.is_valid(gen_idx)); // Stale!
//! assert!(gen_allocator.is_valid(gen_idx2)); // Valid
//! ```

use std::collections::HashSet;
use std::fmt;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during index allocation operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocatorError {
    /// Allocator has reached its capacity limit.
    AtCapacity {
        /// The maximum capacity of the allocator.
        capacity: u32,
    },
    /// The provided index is out of bounds.
    InvalidIndex {
        /// The invalid index that was provided.
        index: u32,
        /// The capacity of the allocator.
        capacity: u32,
    },
    /// Attempted to free an index that was already free.
    DoubleFree(u32),
    /// Generation mismatch indicates use-after-free.
    StaleGeneration {
        /// The expected (current) generation.
        expected: u32,
        /// The generation that was provided.
        found: u32,
    },
}

impl fmt::Display for AllocatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtCapacity { capacity } => {
                write!(f, "allocator at capacity ({})", capacity)
            }
            Self::InvalidIndex { index, capacity } => {
                write!(f, "invalid index {} (capacity: {})", index, capacity)
            }
            Self::DoubleFree(index) => {
                write!(f, "double free detected for index {}", index)
            }
            Self::StaleGeneration { expected, found } => {
                write!(
                    f,
                    "stale generation: expected {}, found {}",
                    expected, found
                )
            }
        }
    }
}

impl std::error::Error for AllocatorError {}

// ============================================================================
// GenerationalIndex
// ============================================================================

/// A handle that combines an index with a generation for use-after-free detection.
///
/// When an index is freed and reallocated, its generation is incremented.
/// Attempting to use a [`GenerationalIndex`] with an outdated generation
/// indicates a use-after-free bug.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GenerationalIndex {
    /// The slot index in the allocator.
    pub index: u32,
    /// The generation at allocation time.
    pub generation: u32,
}

impl GenerationalIndex {
    /// Creates a new generational index.
    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Returns the raw index.
    #[inline]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Returns the generation.
    #[inline]
    pub const fn generation(&self) -> u32 {
        self.generation
    }

    /// Creates an invalid/null generational index.
    #[inline]
    pub const fn null() -> Self {
        Self {
            index: u32::MAX,
            generation: u32::MAX,
        }
    }

    /// Checks if this is a null/invalid index.
    #[inline]
    pub const fn is_null(&self) -> bool {
        self.index == u32::MAX && self.generation == u32::MAX
    }
}

impl Default for GenerationalIndex {
    fn default() -> Self {
        Self::null()
    }
}

impl fmt::Display for GenerationalIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            write!(f, "GenerationalIndex(null)")
        } else {
            write!(f, "GenerationalIndex({}, gen={})", self.index, self.generation)
        }
    }
}

// ============================================================================
// IndexAllocator
// ============================================================================

/// A generic index allocator with free list for efficient resource management.
///
/// The allocator maintains a free list (LIFO stack) for recycling freed indices,
/// which provides cache-friendly allocation patterns. Optionally, generation
/// tracking can be enabled to detect use-after-free bugs.
///
/// # Thread Safety
///
/// `IndexAllocator` is `Send + Sync` when generation tracking is disabled.
/// External synchronization is required for concurrent access.
///
/// # Performance
///
/// - `allocate()`: O(1) - pops from free list or increments counter
/// - `free()`: O(1) - pushes to free list (or O(n) with double-free check)
/// - `is_allocated()`: O(n) worst case (checks free list membership)
#[derive(Debug)]
pub struct IndexAllocator {
    /// Maximum number of indices that can be allocated.
    capacity: u32,
    /// Next fresh index to allocate (when free list is empty).
    next_index: u32,
    /// Stack of freed indices for recycling (LIFO order).
    free_indices: Vec<u32>,
    /// Optional generation tracking per slot.
    generations: Option<Vec<u32>>,
    /// Set of allocated indices for O(1) is_allocated checks.
    /// This is the inverse of free_indices for efficient lookup.
    allocated_set: HashSet<u32>,
}

impl IndexAllocator {
    /// Creates a new index allocator with the given capacity.
    ///
    /// Generation tracking is disabled by default.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of indices that can be allocated.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::index_allocator::IndexAllocator;
    ///
    /// let allocator = IndexAllocator::new(1024);
    /// assert_eq!(allocator.capacity(), 1024);
    /// assert_eq!(allocator.count(), 0);
    /// ```
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            next_index: 0,
            free_indices: Vec::new(),
            generations: None,
            allocated_set: HashSet::new(),
        }
    }

    /// Creates a new index allocator with generation tracking enabled.
    ///
    /// Generation tracking allows detection of use-after-free bugs by
    /// incrementing a generation counter each time a slot is reused.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of indices that can be allocated.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::index_allocator::IndexAllocator;
    ///
    /// let allocator = IndexAllocator::with_generations(1024);
    /// assert_eq!(allocator.capacity(), 1024);
    /// assert!(allocator.has_generations());
    /// ```
    pub fn with_generations(capacity: u32) -> Self {
        Self {
            capacity,
            next_index: 0,
            free_indices: Vec::new(),
            generations: Some(vec![0; capacity as usize]),
            allocated_set: HashSet::new(),
        }
    }

    /// Returns `true` if generation tracking is enabled.
    #[inline]
    pub fn has_generations(&self) -> bool {
        self.generations.is_some()
    }

    /// Returns the maximum capacity of the allocator.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Returns the number of currently allocated indices.
    #[inline]
    pub fn count(&self) -> u32 {
        self.allocated_set.len() as u32
    }

    /// Returns the number of indices available in the free list.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_indices.len()
    }

    /// Returns the number of remaining allocatable slots.
    ///
    /// This includes both fresh indices (not yet allocated) and recycled indices.
    #[inline]
    pub fn available(&self) -> u32 {
        let fresh = self.capacity.saturating_sub(self.next_index);
        fresh + self.free_indices.len() as u32
    }

    /// Allocates the next available index.
    ///
    /// Returns a recycled index from the free list (LIFO order) if available,
    /// otherwise allocates a fresh index. Returns `None` if at capacity.
    ///
    /// # Returns
    ///
    /// * `Some(index)` - The allocated index.
    /// * `None` - The allocator is at capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::index_allocator::IndexAllocator;
    ///
    /// let mut allocator = IndexAllocator::new(3);
    /// assert_eq!(allocator.allocate(), Some(0));
    /// assert_eq!(allocator.allocate(), Some(1));
    /// assert_eq!(allocator.allocate(), Some(2));
    /// assert_eq!(allocator.allocate(), None); // At capacity
    /// ```
    pub fn allocate(&mut self) -> Option<u32> {
        // First try recycling from free list (LIFO for cache friendliness)
        if let Some(index) = self.free_indices.pop() {
            self.allocated_set.insert(index);
            return Some(index);
        }

        // Fall back to fresh allocation
        if self.next_index < self.capacity {
            let index = self.next_index;
            self.next_index += 1;
            self.allocated_set.insert(index);
            Some(index)
        } else {
            None
        }
    }

    /// Allocates the next available index, returning an error if at capacity.
    ///
    /// This is the fallible version of [`allocate()`](Self::allocate).
    ///
    /// # Errors
    ///
    /// Returns [`AllocatorError::AtCapacity`] if the allocator is full.
    pub fn try_allocate(&mut self) -> Result<u32, AllocatorError> {
        self.allocate()
            .ok_or(AllocatorError::AtCapacity { capacity: self.capacity })
    }

    /// Allocates an index with generation tracking.
    ///
    /// Returns a [`GenerationalIndex`] that includes both the slot index and
    /// the current generation. This allows detection of use-after-free bugs.
    ///
    /// # Returns
    ///
    /// * `Some(GenerationalIndex)` - The allocated index with generation.
    /// * `None` - The allocator is at capacity or generations are not enabled.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::index_allocator::IndexAllocator;
    ///
    /// let mut allocator = IndexAllocator::with_generations(1024);
    /// let gen_idx = allocator.allocate_generational().unwrap();
    /// assert_eq!(gen_idx.index, 0);
    /// assert_eq!(gen_idx.generation, 0);
    /// ```
    pub fn allocate_generational(&mut self) -> Option<GenerationalIndex> {
        // Check if generations are enabled first
        if self.generations.is_none() {
            return None;
        }

        // Allocate the index
        let index = self.allocate()?;

        // Now get the generation (safe because we checked is_some above)
        let generation = self.generations.as_ref().unwrap()[index as usize];
        Some(GenerationalIndex::new(index, generation))
    }

    /// Allocates an index with generation tracking, returning an error on failure.
    ///
    /// # Errors
    ///
    /// Returns [`AllocatorError::AtCapacity`] if the allocator is full.
    pub fn try_allocate_generational(&mut self) -> Result<GenerationalIndex, AllocatorError> {
        self.allocate_generational()
            .ok_or(AllocatorError::AtCapacity { capacity: self.capacity })
    }

    /// Frees an index, returning it to the free list for recycling.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to free.
    ///
    /// # Returns
    ///
    /// * `true` - The index was successfully freed.
    /// * `false` - The index was invalid or already free (double-free).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::index_allocator::IndexAllocator;
    ///
    /// let mut allocator = IndexAllocator::new(1024);
    /// let idx = allocator.allocate().unwrap();
    /// assert!(allocator.free(idx));
    /// assert!(!allocator.free(idx)); // Double-free returns false
    /// ```
    pub fn free(&mut self, index: u32) -> bool {
        // Validate index bounds
        if index >= self.next_index || index >= self.capacity {
            return false;
        }

        // Check for double-free
        if !self.allocated_set.remove(&index) {
            return false;
        }

        // Increment generation if tracking is enabled
        if let Some(generations) = &mut self.generations {
            generations[index as usize] = generations[index as usize].wrapping_add(1);
        }

        // Add to free list (LIFO)
        self.free_indices.push(index);
        true
    }

    /// Frees an index, returning an error on failure.
    ///
    /// # Errors
    ///
    /// * [`AllocatorError::InvalidIndex`] - The index is out of bounds.
    /// * [`AllocatorError::DoubleFree`] - The index was already free.
    pub fn try_free(&mut self, index: u32) -> Result<(), AllocatorError> {
        // Validate index bounds
        if index >= self.capacity {
            return Err(AllocatorError::InvalidIndex {
                index,
                capacity: self.capacity,
            });
        }
        if index >= self.next_index {
            return Err(AllocatorError::InvalidIndex {
                index,
                capacity: self.next_index, // Use next_index as effective capacity
            });
        }

        // Check for double-free
        if !self.allocated_set.remove(&index) {
            return Err(AllocatorError::DoubleFree(index));
        }

        // Increment generation if tracking is enabled
        if let Some(generations) = &mut self.generations {
            generations[index as usize] = generations[index as usize].wrapping_add(1);
        }

        // Add to free list (LIFO)
        self.free_indices.push(index);
        Ok(())
    }

    /// Frees a generational index with validation.
    ///
    /// Validates that the generation matches before freeing.
    ///
    /// # Errors
    ///
    /// * [`AllocatorError::InvalidIndex`] - The index is out of bounds.
    /// * [`AllocatorError::DoubleFree`] - The index was already free.
    /// * [`AllocatorError::StaleGeneration`] - The generation doesn't match.
    pub fn try_free_generational(&mut self, gen_idx: GenerationalIndex) -> Result<(), AllocatorError> {
        // Check generation first
        if let Some(generations) = &self.generations {
            if gen_idx.index < self.capacity {
                let current = generations[gen_idx.index as usize];
                if current != gen_idx.generation {
                    return Err(AllocatorError::StaleGeneration {
                        expected: current,
                        found: gen_idx.generation,
                    });
                }
            }
        }

        self.try_free(gen_idx.index)
    }

    /// Checks if an index is currently allocated.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to check.
    ///
    /// # Returns
    ///
    /// `true` if the index is allocated, `false` if free or out of bounds.
    #[inline]
    pub fn is_allocated(&self, index: u32) -> bool {
        self.allocated_set.contains(&index)
    }

    /// Validates a generational index.
    ///
    /// Returns `true` if the index is currently allocated and the generation
    /// matches. Returns `false` if:
    /// - The index is out of bounds
    /// - The index is not allocated
    /// - The generation doesn't match (stale reference)
    /// - Generation tracking is not enabled
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::index_allocator::IndexAllocator;
    ///
    /// let mut allocator = IndexAllocator::with_generations(1024);
    /// let gen_idx = allocator.allocate_generational().unwrap();
    /// assert!(allocator.is_valid(gen_idx));
    ///
    /// allocator.free(gen_idx.index);
    /// assert!(!allocator.is_valid(gen_idx)); // Stale
    ///
    /// let gen_idx2 = allocator.allocate_generational().unwrap();
    /// assert_eq!(gen_idx2.index, gen_idx.index); // Same slot
    /// assert_ne!(gen_idx2.generation, gen_idx.generation); // Different generation
    /// assert!(allocator.is_valid(gen_idx2));
    /// ```
    pub fn is_valid(&self, gen_idx: GenerationalIndex) -> bool {
        // Check if null
        if gen_idx.is_null() {
            return false;
        }

        // Check bounds and allocation
        if !self.is_allocated(gen_idx.index) {
            return false;
        }

        // Check generation if tracking is enabled
        if let Some(generations) = &self.generations {
            if gen_idx.index as usize >= generations.len() {
                return false;
            }
            generations[gen_idx.index as usize] == gen_idx.generation
        } else {
            // No generation tracking, just check allocation
            true
        }
    }

    /// Returns the current generation for an index.
    ///
    /// Returns `None` if generation tracking is disabled or index is out of bounds.
    pub fn generation(&self, index: u32) -> Option<u32> {
        self.generations
            .as_ref()
            .and_then(|g| g.get(index as usize).copied())
    }

    /// Clears all allocations, resetting to initial state.
    ///
    /// All indices are freed and the next allocation will start from 0.
    /// Generations are preserved if tracking is enabled.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::index_allocator::IndexAllocator;
    ///
    /// let mut allocator = IndexAllocator::new(1024);
    /// allocator.allocate();
    /// allocator.allocate();
    /// assert_eq!(allocator.count(), 2);
    ///
    /// allocator.clear();
    /// assert_eq!(allocator.count(), 0);
    /// assert_eq!(allocator.allocate(), Some(0));
    /// ```
    pub fn clear(&mut self) {
        self.next_index = 0;
        self.free_indices.clear();
        self.allocated_set.clear();
        // Note: generations are intentionally preserved to catch stale references
    }

    /// Resets the allocator completely, including generations.
    ///
    /// Unlike [`clear()`](Self::clear), this also resets all generations to 0.
    pub fn reset(&mut self) {
        self.next_index = 0;
        self.free_indices.clear();
        self.allocated_set.clear();
        if let Some(generations) = &mut self.generations {
            generations.fill(0);
        }
    }

    /// Returns an iterator over all currently allocated indices.
    pub fn allocated_indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.allocated_set.iter().copied()
    }

    /// Returns the peak number of simultaneous allocations.
    ///
    /// This is the highest value `next_index` has reached.
    #[inline]
    pub fn peak_allocations(&self) -> u32 {
        self.next_index
    }

    /// Returns the fragmentation ratio (0.0 = none, 1.0 = all slots are recycled).
    ///
    /// High fragmentation means many slots have been freed and recycled.
    pub fn fragmentation(&self) -> f32 {
        if self.next_index == 0 {
            0.0
        } else {
            self.free_indices.len() as f32 / self.next_index as f32
        }
    }

    /// Returns the utilization ratio (allocated / capacity).
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            0.0
        } else {
            self.count() as f32 / self.capacity as f32
        }
    }

    /// Compacts the free list by removing invalid entries.
    ///
    /// This is a no-op for the current implementation but provided for
    /// future extensibility.
    pub fn compact(&mut self) {
        // Current implementation maintains invariants automatically
        // This method exists for API completeness
    }

    /// Reserves capacity in the free list for future frees.
    ///
    /// Useful when you know many indices will be freed.
    pub fn reserve_free_list(&mut self, additional: usize) {
        self.free_indices.reserve(additional);
    }

    /// Shrinks the free list capacity to match its length.
    pub fn shrink_free_list(&mut self) {
        self.free_indices.shrink_to_fit();
    }
}

// Safety: IndexAllocator contains no interior mutability or thread-local state
unsafe impl Send for IndexAllocator {}
unsafe impl Sync for IndexAllocator {}

impl Clone for IndexAllocator {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            next_index: self.next_index,
            free_indices: self.free_indices.clone(),
            generations: self.generations.clone(),
            allocated_set: self.allocated_set.clone(),
        }
    }
}

impl Default for IndexAllocator {
    /// Creates an allocator with capacity of 1024.
    fn default() -> Self {
        Self::new(1024)
    }
}

// ============================================================================
// Metrics
// ============================================================================

/// Metrics for monitoring allocator state.
#[derive(Clone, Debug, Default)]
pub struct IndexAllocatorMetrics {
    /// Current number of allocated indices.
    pub allocated_count: u32,
    /// Maximum capacity.
    pub capacity: u32,
    /// Number of indices in the free list.
    pub free_list_size: usize,
    /// Peak allocations ever reached.
    pub peak_allocations: u32,
    /// Fragmentation ratio (0.0-1.0).
    pub fragmentation: f32,
    /// Utilization ratio (0.0-1.0).
    pub utilization: f32,
    /// Whether generation tracking is enabled.
    pub has_generations: bool,
}

impl IndexAllocator {
    /// Returns current metrics for monitoring.
    pub fn metrics(&self) -> IndexAllocatorMetrics {
        IndexAllocatorMetrics {
            allocated_count: self.count(),
            capacity: self.capacity,
            free_list_size: self.free_indices.len(),
            peak_allocations: self.next_index,
            fragmentation: self.fragmentation(),
            utilization: self.utilization(),
            has_generations: self.generations.is_some(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Basic Allocation Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_new_allocator() {
        let allocator = IndexAllocator::new(100);
        assert_eq!(allocator.capacity(), 100);
        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.free_count(), 0);
        assert!(!allocator.has_generations());
    }

    #[test]
    fn test_with_generations() {
        let allocator = IndexAllocator::with_generations(100);
        assert_eq!(allocator.capacity(), 100);
        assert!(allocator.has_generations());
    }

    #[test]
    fn test_default() {
        let allocator = IndexAllocator::default();
        assert_eq!(allocator.capacity(), 1024);
    }

    #[test]
    fn test_sequential_allocation() {
        let mut allocator = IndexAllocator::new(10);
        assert_eq!(allocator.allocate(), Some(0));
        assert_eq!(allocator.allocate(), Some(1));
        assert_eq!(allocator.allocate(), Some(2));
        assert_eq!(allocator.count(), 3);
    }

    #[test]
    fn test_allocation_at_capacity() {
        let mut allocator = IndexAllocator::new(3);
        assert_eq!(allocator.allocate(), Some(0));
        assert_eq!(allocator.allocate(), Some(1));
        assert_eq!(allocator.allocate(), Some(2));
        assert_eq!(allocator.allocate(), None);
        assert_eq!(allocator.count(), 3);
    }

    #[test]
    fn test_try_allocate_at_capacity() {
        let mut allocator = IndexAllocator::new(1);
        assert!(allocator.try_allocate().is_ok());
        assert!(matches!(
            allocator.try_allocate(),
            Err(AllocatorError::AtCapacity { capacity: 1 })
        ));
    }

    // ------------------------------------------------------------------------
    // Free List Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_free_basic() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        assert_eq!(allocator.count(), 1);
        assert!(allocator.free(idx));
        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.free_count(), 1);
    }

    #[test]
    fn test_free_list_lifo_order() {
        let mut allocator = IndexAllocator::new(10);
        let idx0 = allocator.allocate().unwrap(); // 0
        let idx1 = allocator.allocate().unwrap(); // 1
        let idx2 = allocator.allocate().unwrap(); // 2

        allocator.free(idx0);
        allocator.free(idx1);
        allocator.free(idx2);

        // LIFO: last freed (2) comes out first
        assert_eq!(allocator.allocate(), Some(2));
        assert_eq!(allocator.allocate(), Some(1));
        assert_eq!(allocator.allocate(), Some(0));
    }

    #[test]
    fn test_free_and_reallocate() {
        let mut allocator = IndexAllocator::new(5);
        let idx0 = allocator.allocate().unwrap();
        let idx1 = allocator.allocate().unwrap();

        allocator.free(idx0);
        let idx2 = allocator.allocate().unwrap();

        assert_eq!(idx2, idx0); // Recycled!
        assert_eq!(allocator.count(), 2);
    }

    #[test]
    fn test_double_free_returns_false() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        assert!(allocator.free(idx));
        assert!(!allocator.free(idx)); // Double-free
    }

    #[test]
    fn test_try_free_double_free() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        assert!(allocator.try_free(idx).is_ok());
        assert!(matches!(
            allocator.try_free(idx),
            Err(AllocatorError::DoubleFree(0))
        ));
    }

    #[test]
    fn test_free_invalid_index() {
        let mut allocator = IndexAllocator::new(10);
        assert!(!allocator.free(100)); // Out of bounds
        assert!(!allocator.free(0)); // Never allocated
    }

    #[test]
    fn test_try_free_invalid_index() {
        let mut allocator = IndexAllocator::new(10);
        assert!(matches!(
            allocator.try_free(100),
            Err(AllocatorError::InvalidIndex { index: 100, capacity: 10 })
        ));
    }

    #[test]
    fn test_try_free_never_allocated() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate(); // Allocate 0
        // Index 5 is within next_index conceptually valid but never allocated
        assert!(matches!(
            allocator.try_free(5),
            Err(AllocatorError::InvalidIndex { .. })
        ));
    }

    // ------------------------------------------------------------------------
    // is_allocated Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_is_allocated() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        assert!(allocator.is_allocated(idx));
        allocator.free(idx);
        assert!(!allocator.is_allocated(idx));
    }

    #[test]
    fn test_is_allocated_out_of_bounds() {
        let allocator = IndexAllocator::new(10);
        assert!(!allocator.is_allocated(100));
    }

    #[test]
    fn test_is_allocated_never_allocated() {
        let allocator = IndexAllocator::new(10);
        assert!(!allocator.is_allocated(0));
    }

    // ------------------------------------------------------------------------
    // Generation Tracking Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_allocate_generational() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx = allocator.allocate_generational().unwrap();
        assert_eq!(gen_idx.index, 0);
        assert_eq!(gen_idx.generation, 0);
    }

    #[test]
    fn test_generation_increments_on_reuse() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx1 = allocator.allocate_generational().unwrap();
        allocator.free(gen_idx1.index);
        let gen_idx2 = allocator.allocate_generational().unwrap();

        assert_eq!(gen_idx1.index, gen_idx2.index);
        assert_eq!(gen_idx1.generation, 0);
        assert_eq!(gen_idx2.generation, 1);
    }

    #[test]
    fn test_is_valid_current() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx = allocator.allocate_generational().unwrap();
        assert!(allocator.is_valid(gen_idx));
    }

    #[test]
    fn test_is_valid_stale() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx1 = allocator.allocate_generational().unwrap();
        allocator.free(gen_idx1.index);
        let _gen_idx2 = allocator.allocate_generational().unwrap();

        assert!(!allocator.is_valid(gen_idx1)); // Stale!
    }

    #[test]
    fn test_is_valid_freed() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx = allocator.allocate_generational().unwrap();
        allocator.free(gen_idx.index);
        assert!(!allocator.is_valid(gen_idx));
    }

    #[test]
    fn test_is_valid_null() {
        let allocator = IndexAllocator::with_generations(10);
        assert!(!allocator.is_valid(GenerationalIndex::null()));
    }

    #[test]
    fn test_is_valid_without_generations() {
        let mut allocator = IndexAllocator::new(10);
        let idx = allocator.allocate().unwrap();
        // Create a fake generational index
        let gen_idx = GenerationalIndex::new(idx, 0);
        // Without generation tracking, is_valid just checks allocation
        assert!(allocator.is_valid(gen_idx));
    }

    #[test]
    fn test_generation_method() {
        let mut allocator = IndexAllocator::with_generations(10);
        assert_eq!(allocator.generation(0), Some(0));
        allocator.allocate();
        allocator.free(0);
        assert_eq!(allocator.generation(0), Some(1));
    }

    #[test]
    fn test_generation_without_tracking() {
        let allocator = IndexAllocator::new(10);
        assert_eq!(allocator.generation(0), None);
    }

    #[test]
    fn test_try_free_generational_valid() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx = allocator.allocate_generational().unwrap();
        assert!(allocator.try_free_generational(gen_idx).is_ok());
    }

    #[test]
    fn test_try_free_generational_stale() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx1 = allocator.allocate_generational().unwrap();
        allocator.free(gen_idx1.index);
        let _gen_idx2 = allocator.allocate_generational().unwrap();

        assert!(matches!(
            allocator.try_free_generational(gen_idx1),
            Err(AllocatorError::StaleGeneration { expected: 1, found: 0 })
        ));
    }

    #[test]
    fn test_try_allocate_generational_at_capacity() {
        let mut allocator = IndexAllocator::with_generations(1);
        assert!(allocator.try_allocate_generational().is_ok());
        assert!(matches!(
            allocator.try_allocate_generational(),
            Err(AllocatorError::AtCapacity { capacity: 1 })
        ));
    }

    // ------------------------------------------------------------------------
    // Clear and Reset Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_clear() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();
        allocator.allocate();
        allocator.clear();

        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.free_count(), 0);
        assert_eq!(allocator.allocate(), Some(0));
    }

    #[test]
    fn test_clear_preserves_generations() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx = allocator.allocate_generational().unwrap();
        allocator.free(gen_idx.index);

        // Generation is now 1
        assert_eq!(allocator.generation(0), Some(1));

        allocator.clear();

        // Clear preserves generations
        assert_eq!(allocator.generation(0), Some(1));

        // New allocation at same slot has generation 1
        let gen_idx2 = allocator.allocate_generational().unwrap();
        assert_eq!(gen_idx2.generation, 1);
    }

    #[test]
    fn test_reset() {
        let mut allocator = IndexAllocator::with_generations(10);
        let gen_idx = allocator.allocate_generational().unwrap();
        allocator.free(gen_idx.index);

        allocator.reset();

        assert_eq!(allocator.count(), 0);
        assert_eq!(allocator.generation(0), Some(0)); // Reset to 0
    }

    // ------------------------------------------------------------------------
    // Edge Cases
    // ------------------------------------------------------------------------

    #[test]
    fn test_zero_capacity() {
        let mut allocator = IndexAllocator::new(0);
        assert_eq!(allocator.capacity(), 0);
        assert_eq!(allocator.allocate(), None);
        assert!(!allocator.free(0));
    }

    #[test]
    fn test_max_capacity() {
        // Don't actually allocate u32::MAX, just verify construction
        let allocator = IndexAllocator::new(u32::MAX);
        assert_eq!(allocator.capacity(), u32::MAX);
    }

    #[test]
    fn test_generation_wrapping() {
        let mut allocator = IndexAllocator::with_generations(1);

        // Manually set generation to MAX to test wrapping
        if let Some(gens) = &mut allocator.generations {
            gens[0] = u32::MAX;
        }

        allocator.allocate();
        allocator.free(0);

        // Should wrap to 0
        assert_eq!(allocator.generation(0), Some(0));
    }

    // ------------------------------------------------------------------------
    // Metrics and Statistics Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_available() {
        let mut allocator = IndexAllocator::new(10);
        assert_eq!(allocator.available(), 10);

        allocator.allocate();
        assert_eq!(allocator.available(), 9);

        allocator.free(0);
        assert_eq!(allocator.available(), 10); // 9 fresh + 1 recycled
    }

    #[test]
    fn test_peak_allocations() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();
        allocator.allocate();
        allocator.allocate();

        assert_eq!(allocator.peak_allocations(), 3);

        allocator.free(0);
        allocator.free(1);

        // Peak doesn't decrease
        assert_eq!(allocator.peak_allocations(), 3);
    }

    #[test]
    fn test_fragmentation() {
        let mut allocator = IndexAllocator::new(10);
        assert_eq!(allocator.fragmentation(), 0.0);

        for _ in 0..4 {
            allocator.allocate();
        }
        // 4 allocated, 0 free
        assert_eq!(allocator.fragmentation(), 0.0);

        allocator.free(0);
        allocator.free(1);
        // 2 allocated, 2 free, peak = 4
        assert_eq!(allocator.fragmentation(), 0.5);
    }

    #[test]
    fn test_utilization() {
        let mut allocator = IndexAllocator::new(10);
        assert_eq!(allocator.utilization(), 0.0);

        for _ in 0..5 {
            allocator.allocate();
        }
        assert_eq!(allocator.utilization(), 0.5);

        for _ in 0..5 {
            allocator.allocate();
        }
        assert_eq!(allocator.utilization(), 1.0);
    }

    #[test]
    fn test_metrics() {
        let mut allocator = IndexAllocator::with_generations(100);
        allocator.allocate();
        allocator.allocate();
        allocator.free(0);

        let metrics = allocator.metrics();
        assert_eq!(metrics.allocated_count, 1);
        assert_eq!(metrics.capacity, 100);
        assert_eq!(metrics.free_list_size, 1);
        assert_eq!(metrics.peak_allocations, 2);
        assert!(metrics.has_generations);
    }

    // ------------------------------------------------------------------------
    // Iterator Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_allocated_indices_iterator() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate(); // 0
        allocator.allocate(); // 1
        allocator.allocate(); // 2
        allocator.free(1);

        let mut indices: Vec<_> = allocator.allocated_indices().collect();
        indices.sort();
        assert_eq!(indices, vec![0, 2]);
    }

    // ------------------------------------------------------------------------
    // Memory Management Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_reserve_free_list() {
        let mut allocator = IndexAllocator::new(1000);
        allocator.reserve_free_list(500);
        // Just verify it doesn't panic
        assert!(allocator.free_indices.capacity() >= 500);
    }

    #[test]
    fn test_shrink_free_list() {
        let mut allocator = IndexAllocator::new(100);
        for i in 0..50 {
            allocator.allocate();
        }
        for i in 0..50 {
            allocator.free(i);
        }

        let before = allocator.free_indices.capacity();
        allocator.shrink_free_list();
        let after = allocator.free_indices.capacity();
        assert!(after <= before);
    }

    #[test]
    fn test_compact() {
        let mut allocator = IndexAllocator::new(10);
        allocator.allocate();
        allocator.free(0);
        allocator.compact(); // Should be no-op but not panic
        assert_eq!(allocator.free_count(), 1);
    }

    // ------------------------------------------------------------------------
    // Thread Safety Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<IndexAllocator>();
        assert_sync::<IndexAllocator>();
        assert_send::<GenerationalIndex>();
        assert_sync::<GenerationalIndex>();
    }

    // ------------------------------------------------------------------------
    // GenerationalIndex Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_generational_index_new() {
        let idx = GenerationalIndex::new(42, 7);
        assert_eq!(idx.index(), 42);
        assert_eq!(idx.generation(), 7);
    }

    #[test]
    fn test_generational_index_null() {
        let idx = GenerationalIndex::null();
        assert!(idx.is_null());
        assert_eq!(idx.index, u32::MAX);
        assert_eq!(idx.generation, u32::MAX);
    }

    #[test]
    fn test_generational_index_default() {
        let idx = GenerationalIndex::default();
        assert!(idx.is_null());
    }

    #[test]
    fn test_generational_index_display() {
        let idx = GenerationalIndex::new(5, 3);
        assert_eq!(format!("{}", idx), "GenerationalIndex(5, gen=3)");

        let null = GenerationalIndex::null();
        assert_eq!(format!("{}", null), "GenerationalIndex(null)");
    }

    #[test]
    fn test_generational_index_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(GenerationalIndex::new(0, 0));
        set.insert(GenerationalIndex::new(0, 1));
        set.insert(GenerationalIndex::new(1, 0));

        assert_eq!(set.len(), 3);
        assert!(set.contains(&GenerationalIndex::new(0, 0)));
    }

    // ------------------------------------------------------------------------
    // Error Display Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        assert_eq!(
            AllocatorError::AtCapacity { capacity: 100 }.to_string(),
            "allocator at capacity (100)"
        );
        assert_eq!(
            AllocatorError::InvalidIndex { index: 50, capacity: 10 }.to_string(),
            "invalid index 50 (capacity: 10)"
        );
        assert_eq!(
            AllocatorError::DoubleFree(5).to_string(),
            "double free detected for index 5"
        );
        assert_eq!(
            AllocatorError::StaleGeneration { expected: 2, found: 1 }.to_string(),
            "stale generation: expected 2, found 1"
        );
    }

    // ------------------------------------------------------------------------
    // Clone Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_clone() {
        let mut allocator = IndexAllocator::with_generations(10);
        allocator.allocate();
        allocator.allocate();
        allocator.free(0);

        let cloned = allocator.clone();

        assert_eq!(cloned.count(), allocator.count());
        assert_eq!(cloned.capacity(), allocator.capacity());
        assert_eq!(cloned.free_count(), allocator.free_count());
        assert_eq!(cloned.generation(0), allocator.generation(0));
    }

    // ------------------------------------------------------------------------
    // Stress Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_many_allocations() {
        let mut allocator = IndexAllocator::new(10000);

        for i in 0..10000 {
            assert_eq!(allocator.allocate(), Some(i));
        }
        assert_eq!(allocator.allocate(), None);
        assert_eq!(allocator.count(), 10000);
    }

    #[test]
    fn test_allocate_free_cycle() {
        let mut allocator = IndexAllocator::with_generations(100);

        // Allocate all
        let indices: Vec<_> = (0..100).map(|_| allocator.allocate_generational().unwrap()).collect();

        // Free all (reverse order)
        for gen_idx in indices.iter().rev() {
            assert!(allocator.try_free_generational(*gen_idx).is_ok());
        }

        // Reallocate all (should all have generation 1)
        for i in 0..100 {
            let gen_idx = allocator.allocate_generational().unwrap();
            assert_eq!(gen_idx.generation, 1);
        }
    }

    #[test]
    fn test_interleaved_alloc_free() {
        let mut allocator = IndexAllocator::new(10);

        let a = allocator.allocate().unwrap();
        let b = allocator.allocate().unwrap();
        allocator.free(a);
        let c = allocator.allocate().unwrap();
        let d = allocator.allocate().unwrap();
        allocator.free(b);
        allocator.free(c);
        let e = allocator.allocate().unwrap();

        // Verify LIFO behavior
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 0); // Recycled from a
        assert_eq!(d, 2);
        assert_eq!(e, 0); // Recycled from c (LIFO: c freed after b)
    }
}
