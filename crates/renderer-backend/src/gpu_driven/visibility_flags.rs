//! Visibility Flags Buffer for GPU-driven culling (T-WGPU-P6.2.3).
//!
//! This module provides a packed bitfield buffer where each bit represents
//! whether an object passed GPU culling. The buffer is designed for:
//!
//! - **Atomic writes**: Culling compute shaders atomically OR bits to set visibility
//! - **Sequential reads**: Compaction shaders read bits to filter visible objects
//! - **Frame-level clearing**: All bits cleared to 0 at frame start
//!
//! # Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      VisibilityFlagsBuffer                              │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  word 0:      [31][30]...[1][0]   ← objects 0-31                        │
//! │  word 1:      [31][30]...[1][0]   ← objects 32-63                       │
//! │  word 2:      [31][30]...[1][0]   ← objects 64-95                       │
//! │  ...                                                                    │
//! │  word N-1:    [31][30]...[1][0]   ← objects (N-1)*32 to N*32-1          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! - **Memory**: 1 bit per object (32 objects per u32 word)
//! - **Culling**: Atomic OR (`atomicOr`) in compute shader (no contention)
//! - **Compaction**: Sequential read, bit extraction
//! - **Clear**: Full buffer write to 0 each frame
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::VisibilityFlagsBuffer;
//!
//! // Create buffer for 100,000 objects
//! let mut flags = VisibilityFlagsBuffer::new(&device, 100_000, Some("main_scene"));
//!
//! // Each frame:
//! flags.clear(&queue);  // Reset all bits to 0
//!
//! // In culling compute shader (WGSL):
//! // atomicOr(&visibility_flags[word_index], bit_mask);
//!
//! // In compaction compute shader (WGSL):
//! // let word = visibility_flags[word_index];
//! // if (word & bit_mask) != 0 { /* object is visible */ }
//! ```

use std::mem;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Number of bits per word (u32).
pub const BITS_PER_WORD: usize = 32;

/// Default initial object capacity.
pub const DEFAULT_VISIBILITY_FLAGS_CAPACITY: usize = 65536;

/// Minimum capacity to avoid edge cases.
pub const MIN_VISIBILITY_FLAGS_CAPACITY: usize = 32;

/// Size of a single word in bytes.
pub const WORD_SIZE: usize = mem::size_of::<u32>();

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Calculate number of u32 words needed for N objects.
///
/// Each word holds 32 objects (1 bit per object).
///
/// # Examples
///
/// ```ignore
/// assert_eq!(words_for_objects(0), 0);
/// assert_eq!(words_for_objects(1), 1);
/// assert_eq!(words_for_objects(32), 1);
/// assert_eq!(words_for_objects(33), 2);
/// ```
#[inline]
pub const fn words_for_objects(object_count: usize) -> usize {
    (object_count + BITS_PER_WORD - 1) / BITS_PER_WORD
}

/// Calculate word index and bit mask for an object.
///
/// Returns `(word_index, bit_mask)` where:
/// - `word_index`: Index into the u32 array
/// - `bit_mask`: Single-bit mask to OR/AND with the word
///
/// # Examples
///
/// ```ignore
/// assert_eq!(bit_location(0), (0, 1));        // Object 0 -> word 0, bit 0
/// assert_eq!(bit_location(31), (0, 1 << 31)); // Object 31 -> word 0, bit 31
/// assert_eq!(bit_location(32), (1, 1));       // Object 32 -> word 1, bit 0
/// ```
#[inline]
pub const fn bit_location(object_index: usize) -> (usize, u32) {
    let word = object_index / BITS_PER_WORD;
    let bit = 1u32 << (object_index % BITS_PER_WORD);
    (word, bit)
}

/// Check if an object is visible in the flags buffer (CPU-side).
///
/// # Arguments
///
/// * `flags` - Slice of u32 words containing visibility bits
/// * `object_index` - Index of the object to check
///
/// # Returns
///
/// `true` if the object's visibility bit is set.
#[inline]
pub fn is_visible(flags: &[u32], object_index: usize) -> bool {
    let (word, bit) = bit_location(object_index);
    if word < flags.len() {
        (flags[word] & bit) != 0
    } else {
        false
    }
}

/// Set an object's visibility bit (CPU-side).
///
/// # Arguments
///
/// * `flags` - Mutable slice of u32 words
/// * `object_index` - Index of the object to mark visible
#[inline]
pub fn set_visible(flags: &mut [u32], object_index: usize) {
    let (word, bit) = bit_location(object_index);
    if word < flags.len() {
        flags[word] |= bit;
    }
}

/// Clear an object's visibility bit (CPU-side).
///
/// # Arguments
///
/// * `flags` - Mutable slice of u32 words
/// * `object_index` - Index of the object to mark invisible
#[inline]
pub fn clear_visible(flags: &mut [u32], object_index: usize) {
    let (word, bit) = bit_location(object_index);
    if word < flags.len() {
        flags[word] &= !bit;
    }
}

/// Count visible objects in the flags buffer (CPU-side).
///
/// Uses `count_ones()` for efficient population count.
#[inline]
pub fn count_visible(flags: &[u32]) -> usize {
    flags.iter().map(|w| w.count_ones() as usize).sum()
}

// =============================================================================
// VISIBILITY FLAGS BUFFER
// =============================================================================

/// GPU visibility flags buffer for culling results.
///
/// Stores 1 bit per object as a packed bitfield in u32 words.
/// Designed for atomic writes in culling shaders and sequential
/// reads in compaction shaders.
///
/// # GPU Usage (WGSL)
///
/// ```wgsl
/// @group(0) @binding(0) var<storage, read_write> visibility_flags: array<atomic<u32>>;
///
/// // In culling shader (set visible):
/// let obj_idx = global_id.x;
/// let word_idx = obj_idx / 32u;
/// let bit_mask = 1u << (obj_idx % 32u);
/// atomicOr(&visibility_flags[word_idx], bit_mask);
///
/// // In compaction shader (read visibility):
/// @group(0) @binding(0) var<storage, read> visibility_flags: array<u32>;
/// let word = visibility_flags[word_idx];
/// if (word & bit_mask) != 0u {
///     // Object is visible
/// }
/// ```
pub struct VisibilityFlagsBuffer {
    /// GPU storage buffer for visibility bits.
    buffer: wgpu::Buffer,

    /// Number of objects this buffer can track.
    object_count: usize,

    /// Number of u32 words in the buffer.
    word_count: usize,

    /// Debug label.
    label: Option<String>,
}

impl VisibilityFlagsBuffer {
    /// Create a new visibility flags buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `object_count` - Maximum number of objects to track
    /// * `label` - Optional debug label
    ///
    /// # Example
    ///
    /// ```ignore
    /// let flags = VisibilityFlagsBuffer::new(&device, 100_000, Some("main_scene"));
    /// ```
    pub fn new(device: &wgpu::Device, object_count: usize, label: Option<&str>) -> Self {
        let object_count = object_count.max(MIN_VISIBILITY_FLAGS_CAPACITY);
        let word_count = words_for_objects(object_count);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: label.map(|l| format!("{}_visibility_flags", l)).as_deref(),
            size: (word_count * WORD_SIZE) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            object_count,
            word_count,
            label: label.map(String::from),
        }
    }

    /// Create with default capacity.
    pub fn with_default_capacity(device: &wgpu::Device, label: Option<&str>) -> Self {
        Self::new(device, DEFAULT_VISIBILITY_FLAGS_CAPACITY, label)
    }

    /// Clear all visibility bits to 0.
    ///
    /// Call at frame start before culling dispatches.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue for buffer writes
    pub fn clear(&self, queue: &wgpu::Queue) {
        let zeros = vec![0u32; self.word_count];
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&zeros));
    }

    /// Clear using a command encoder (for batching).
    ///
    /// Uses `clear_buffer` command for potentially better performance
    /// on some GPU backends.
    pub fn clear_with_encoder(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.buffer, 0, None);
    }

    /// Get the GPU storage buffer for binding.
    ///
    /// Use with `BufferBindingType::Storage { read_only: false }` for
    /// atomic writes in culling shaders, or `read_only: true` for
    /// compaction reads.
    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get buffer binding resource.
    #[inline]
    pub fn buffer_binding(&self) -> wgpu::BufferBinding<'_> {
        self.buffer.as_entire_buffer_binding()
    }

    /// Get the number of objects this buffer can track.
    #[inline]
    pub fn object_count(&self) -> usize {
        self.object_count
    }

    /// Get the number of u32 words in the buffer.
    #[inline]
    pub fn word_count(&self) -> usize {
        self.word_count
    }

    /// Get buffer size in bytes.
    #[inline]
    pub fn buffer_size(&self) -> u64 {
        (self.word_count * WORD_SIZE) as u64
    }

    /// Get the debug label.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Resize buffer for new object count.
    ///
    /// Creates a new buffer if the new count exceeds current capacity.
    /// The old buffer is dropped and the new buffer is uninitialized
    /// (should be cleared before use).
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `new_object_count` - New maximum object count
    ///
    /// # Returns
    ///
    /// `true` if buffer was resized, `false` if existing capacity was sufficient.
    pub fn resize(&mut self, device: &wgpu::Device, new_object_count: usize) -> bool {
        let new_object_count = new_object_count.max(MIN_VISIBILITY_FLAGS_CAPACITY);

        if new_object_count > self.object_count {
            let new_word_count = words_for_objects(new_object_count);

            let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: self
                    .label
                    .as_ref()
                    .map(|l| format!("{}_visibility_flags", l))
                    .as_deref(),
                size: (new_word_count * WORD_SIZE) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            self.buffer = new_buffer;
            self.object_count = new_object_count;
            self.word_count = new_word_count;

            return true;
        }

        false
    }

    /// Read visibility flags back to CPU (synchronous).
    ///
    /// This is primarily for debugging/testing. Blocks until GPU completes.
    ///
    /// # Returns
    ///
    /// Vector of u32 words containing visibility bits.
    pub fn read_back(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_flags_staging"),
            size: self.buffer_size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.buffer_size());
        queue.submit([encoder.finish()]);

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let words: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        words
    }

    /// Upload CPU flags to GPU (for testing).
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue
    /// * `flags` - Slice of u32 words to upload
    pub fn upload(&self, queue: &wgpu::Queue, flags: &[u32]) {
        let upload_count = flags.len().min(self.word_count);
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&flags[..upload_count]),
        );
    }
}

impl std::fmt::Debug for VisibilityFlagsBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisibilityFlagsBuffer")
            .field("object_count", &self.object_count)
            .field("word_count", &self.word_count)
            .field("buffer_size", &self.buffer_size())
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

// =============================================================================
// CPU REFERENCE IMPLEMENTATION
// =============================================================================

/// CPU reference implementation for clearing visibility flags.
pub fn cpu_clear_visibility_flags(flags: &mut [u32]) {
    flags.fill(0);
}

/// CPU reference implementation for atomic OR (simulated).
///
/// In GPU, this would be `atomicOr(&visibility_flags[word], bit)`.
pub fn cpu_atomic_or_visibility(flags: &mut [u32], object_index: usize) {
    set_visible(flags, object_index);
}

/// CPU reference implementation for compaction pass.
///
/// Returns indices of all visible objects.
pub fn cpu_compact_visible(flags: &[u32], object_count: usize) -> Vec<usize> {
    (0..object_count)
        .filter(|&i| is_visible(flags, i))
        .collect()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Words For Objects Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_words_for_objects_zero() {
        assert_eq!(words_for_objects(0), 0);
    }

    #[test]
    fn test_words_for_objects_one() {
        assert_eq!(words_for_objects(1), 1);
    }

    #[test]
    fn test_words_for_objects_boundary() {
        assert_eq!(words_for_objects(32), 1);
        assert_eq!(words_for_objects(33), 2);
    }

    #[test]
    fn test_words_for_objects_exact_multiple() {
        assert_eq!(words_for_objects(64), 2);
        assert_eq!(words_for_objects(128), 4);
        assert_eq!(words_for_objects(1024), 32);
    }

    #[test]
    fn test_words_for_objects_off_by_one() {
        assert_eq!(words_for_objects(65), 3);
        assert_eq!(words_for_objects(63), 2);
    }

    #[test]
    fn test_words_for_objects_large() {
        // 1 million objects
        assert_eq!(words_for_objects(1_000_000), 31250);
    }

    // -------------------------------------------------------------------------
    // Bit Location Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bit_location_object_zero() {
        let (word, bit) = bit_location(0);
        assert_eq!(word, 0);
        assert_eq!(bit, 1);
    }

    #[test]
    fn test_bit_location_object_31() {
        let (word, bit) = bit_location(31);
        assert_eq!(word, 0);
        assert_eq!(bit, 1 << 31);
    }

    #[test]
    fn test_bit_location_object_32() {
        let (word, bit) = bit_location(32);
        assert_eq!(word, 1);
        assert_eq!(bit, 1);
    }

    #[test]
    fn test_bit_location_object_63() {
        let (word, bit) = bit_location(63);
        assert_eq!(word, 1);
        assert_eq!(bit, 1 << 31);
    }

    #[test]
    fn test_bit_location_object_64() {
        let (word, bit) = bit_location(64);
        assert_eq!(word, 2);
        assert_eq!(bit, 1);
    }

    #[test]
    fn test_bit_location_arbitrary() {
        // Object 100 should be in word 3 (100/32=3), bit 4 (100%32=4)
        let (word, bit) = bit_location(100);
        assert_eq!(word, 3);
        assert_eq!(bit, 1 << 4);
    }

    // -------------------------------------------------------------------------
    // Visibility Flag Operations Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_visible_empty() {
        let flags = vec![0u32; 4];
        assert!(!is_visible(&flags, 0));
        assert!(!is_visible(&flags, 31));
        assert!(!is_visible(&flags, 64));
    }

    #[test]
    fn test_set_visible_single() {
        let mut flags = vec![0u32; 4];
        set_visible(&mut flags, 0);
        assert!(is_visible(&flags, 0));
        assert!(!is_visible(&flags, 1));
    }

    #[test]
    fn test_set_visible_multiple() {
        let mut flags = vec![0u32; 4];
        set_visible(&mut flags, 5);
        set_visible(&mut flags, 32);
        set_visible(&mut flags, 100);

        assert!(is_visible(&flags, 5));
        assert!(is_visible(&flags, 32));
        assert!(is_visible(&flags, 100));
        assert!(!is_visible(&flags, 0));
        assert!(!is_visible(&flags, 33));
    }

    #[test]
    fn test_clear_visible() {
        let mut flags = vec![0u32; 4];
        set_visible(&mut flags, 10);
        assert!(is_visible(&flags, 10));

        clear_visible(&mut flags, 10);
        assert!(!is_visible(&flags, 10));
    }

    #[test]
    fn test_clear_visible_preserves_others() {
        let mut flags = vec![0u32; 4];
        set_visible(&mut flags, 10);
        set_visible(&mut flags, 11);

        clear_visible(&mut flags, 10);
        assert!(!is_visible(&flags, 10));
        assert!(is_visible(&flags, 11));
    }

    #[test]
    fn test_is_visible_out_of_bounds() {
        let flags = vec![0u32; 2]; // Only 64 objects
        // Should return false for out-of-bounds, not panic
        assert!(!is_visible(&flags, 100));
    }

    #[test]
    fn test_set_visible_out_of_bounds() {
        let mut flags = vec![0u32; 2];
        // Should not panic for out-of-bounds
        set_visible(&mut flags, 100);
        // Flags should be unchanged
        assert_eq!(flags, vec![0u32; 2]);
    }

    // -------------------------------------------------------------------------
    // Count Visible Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_count_visible_empty() {
        let flags = vec![0u32; 4];
        assert_eq!(count_visible(&flags), 0);
    }

    #[test]
    fn test_count_visible_all_set() {
        let flags = vec![u32::MAX; 4];
        assert_eq!(count_visible(&flags), 128); // 4 * 32 = 128
    }

    #[test]
    fn test_count_visible_some_set() {
        let mut flags = vec![0u32; 4];
        set_visible(&mut flags, 0);
        set_visible(&mut flags, 10);
        set_visible(&mut flags, 50);
        assert_eq!(count_visible(&flags), 3);
    }

    #[test]
    fn test_count_visible_pattern() {
        // Alternating bits: 0b10101010...
        let flags = vec![0xAAAAAAAA; 4];
        assert_eq!(count_visible(&flags), 64); // Half of 128
    }

    // -------------------------------------------------------------------------
    // CPU Reference Implementation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_clear_visibility_flags() {
        let mut flags = vec![0xFFFFFFFF; 4];
        cpu_clear_visibility_flags(&mut flags);
        assert_eq!(flags, vec![0u32; 4]);
    }

    #[test]
    fn test_cpu_atomic_or_visibility() {
        let mut flags = vec![0u32; 4];
        cpu_atomic_or_visibility(&mut flags, 42);
        assert!(is_visible(&flags, 42));
    }

    #[test]
    fn test_cpu_compact_visible() {
        let mut flags = vec![0u32; 4];
        set_visible(&mut flags, 5);
        set_visible(&mut flags, 20);
        set_visible(&mut flags, 100);

        let visible = cpu_compact_visible(&flags, 128);
        assert_eq!(visible, vec![5, 20, 100]);
    }

    #[test]
    fn test_cpu_compact_visible_empty() {
        let flags = vec![0u32; 4];
        let visible = cpu_compact_visible(&flags, 128);
        assert!(visible.is_empty());
    }

    #[test]
    fn test_cpu_compact_visible_all() {
        let flags = vec![u32::MAX; 2];
        let visible = cpu_compact_visible(&flags, 64);
        assert_eq!(visible.len(), 64);
        assert_eq!(visible[0], 0);
        assert_eq!(visible[63], 63);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants() {
        assert_eq!(BITS_PER_WORD, 32);
        assert_eq!(WORD_SIZE, 4);
        assert!(DEFAULT_VISIBILITY_FLAGS_CAPACITY >= MIN_VISIBILITY_FLAGS_CAPACITY);
    }

    // -------------------------------------------------------------------------
    // Buffer Sizing Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffer_size_calculation() {
        // 1000 objects needs 32 words (ceiling of 1000/32)
        let words = words_for_objects(1000);
        assert_eq!(words, 32);

        // Buffer size in bytes
        let size = words * WORD_SIZE;
        assert_eq!(size, 128);
    }

    #[test]
    fn test_buffer_size_large() {
        // 100,000 objects
        let words = words_for_objects(100_000);
        assert_eq!(words, 3125);

        let size = words * WORD_SIZE;
        assert_eq!(size, 12500); // ~12KB
    }

    // -------------------------------------------------------------------------
    // Bit Pattern Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bit_pattern_sequential() {
        let mut flags = vec![0u32; 4];

        // Set first 8 objects
        for i in 0..8 {
            set_visible(&mut flags, i);
        }

        // First word should be 0x000000FF
        assert_eq!(flags[0], 0x000000FF);
    }

    #[test]
    fn test_bit_pattern_every_other() {
        let mut flags = vec![0u32; 1];

        // Set every other object in first word
        for i in (0..32).step_by(2) {
            set_visible(&mut flags, i);
        }

        // Should be alternating bits: 0b01010101...
        assert_eq!(flags[0], 0x55555555);
    }

    #[test]
    fn test_bit_pattern_cross_word() {
        let mut flags = vec![0u32; 2];

        // Set objects 30, 31, 32, 33 (crosses word boundary)
        set_visible(&mut flags, 30);
        set_visible(&mut flags, 31);
        set_visible(&mut flags, 32);
        set_visible(&mut flags, 33);

        assert_eq!(flags[0], 0xC0000000); // Bits 30, 31 set
        assert_eq!(flags[1], 0x00000003); // Bits 0, 1 set
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_single_object_capacity() {
        // Minimum practical case
        let words = words_for_objects(1);
        assert_eq!(words, 1);

        let mut flags = vec![0u32; words];
        set_visible(&mut flags, 0);
        assert!(is_visible(&flags, 0));
    }

    #[test]
    fn test_exactly_one_word() {
        let words = words_for_objects(32);
        assert_eq!(words, 1);

        let mut flags = vec![0u32; words];
        set_visible(&mut flags, 31);
        assert!(is_visible(&flags, 31));
    }

    #[test]
    fn test_idempotent_set() {
        let mut flags = vec![0u32; 1];

        // Setting same bit multiple times should be idempotent
        set_visible(&mut flags, 5);
        let word1 = flags[0];

        set_visible(&mut flags, 5);
        let word2 = flags[0];

        assert_eq!(word1, word2);
    }
}
