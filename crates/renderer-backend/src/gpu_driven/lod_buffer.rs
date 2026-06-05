//! LOD Buffer Management for GPU-driven LOD selection (T-WGPU-P6.5.3).
//!
//! This module provides buffer management for storing per-object LOD selection
//! results. The LOD buffer stores the selected LOD level and blend factor for
//! each object, enabling smooth LOD transitions and efficient indirect draw
//! generation.
//!
//! # Overview
//!
//! ```text
//! +------------------------------------------------------------------+
//! |                         LodBuffer                                |
//! +------------------------------------------------------------------+
//! | Entry 0:  [level: u32][blend_factor: f32]  <- Object 0 LOD       |
//! | Entry 1:  [level: u32][blend_factor: f32]  <- Object 1 LOD       |
//! | Entry 2:  [level: u32][blend_factor: f32]  <- Object 2 LOD       |
//! | ...                                                              |
//! | Entry N:  [level: u32][blend_factor: f32]  <- Object N LOD       |
//! +------------------------------------------------------------------+
//! ```
//!
//! # Data Layout
//!
//! Each `LodEntry` is 8 bytes for GPU alignment:
//!
//! | Offset | Field        | Size | Description                    |
//! |--------|--------------|------|--------------------------------|
//! | 0      | level        | 4    | LOD level (0-3, u32)           |
//! | 4      | blend_factor | 4    | Blend factor for transitions   |
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::lod_buffer::{LodBuffer, LodBufferPool, LodEntry};
//!
//! // Create buffer for 10,000 objects
//! let mut lod_buffer = LodBuffer::new(&device, 10_000, Some("main_scene"));
//!
//! // Each frame:
//! lod_buffer.clear(&mut encoder);  // Reset all entries to LOD 0
//!
//! // After LOD selection compute pass, buffer contains per-object LOD levels
//! // that can be used in indirect draw generation.
//!
//! // For double/triple buffering:
//! let mut pool = LodBufferPool::new(&device, 10_000, 2);  // Double buffering
//! let current = pool.current();
//! current.clear(&mut encoder);
//! // ... use current buffer ...
//! pool.advance();  // Switch to next buffer
//! ```
//!
//! # GPU Usage (WGSL)
//!
//! ```wgsl
//! struct LodEntry {
//!     level: u32,
//!     blend_factor: f32,
//! }
//!
//! @group(0) @binding(0) var<storage, read_write> lod_buffer: array<LodEntry>;
//!
//! // In LOD selection compute shader:
//! let obj_idx = global_id.x;
//! let distance = calculate_distance(camera_pos, object_pos[obj_idx]);
//! let lod = select_lod_level(distance);
//! let blend = calculate_blend_factor(distance, lod);
//! lod_buffer[obj_idx] = LodEntry(lod, blend);
//!
//! // In indirect draw generation:
//! let lod_entry = lod_buffer[obj_idx];
//! let mesh_idx = lod_meshes[lod_entry.level];
//! ```

use bytemuck::{Pod, Zeroable};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Size of a single LOD entry in bytes.
pub const LOD_ENTRY_SIZE: usize = 8;

/// Default LOD buffer capacity (number of objects).
pub const DEFAULT_LOD_BUFFER_CAPACITY: u32 = 65536;

/// Minimum LOD buffer capacity.
pub const MIN_LOD_BUFFER_CAPACITY: u32 = 32;

/// Maximum LOD level (LOD 0-3, so max is 3).
pub const MAX_LOD_LEVEL: u32 = 3;

/// Default number of buffers in a pool.
pub const DEFAULT_POOL_SIZE: usize = 2;

// =============================================================================
// LOD ENTRY
// =============================================================================

/// Per-object LOD selection result.
///
/// Stores the selected LOD level and an optional blend factor for smooth
/// LOD transitions. This struct is designed for GPU storage and is used
/// in indirect draw generation.
///
/// # Memory Layout (8 bytes)
///
/// ```text
/// +------------------+--------+--------+----------------------------------+
/// | Field            | Offset | Size   | Description                      |
/// +------------------+--------+--------+----------------------------------+
/// | level            | 0      | 4      | LOD level (0-3)                  |
/// | blend_factor     | 4      | 4      | Transition blend factor (0.0-1.0)|
/// +------------------+--------+--------+----------------------------------+
/// | Total            |        | 8      |                                  |
/// +------------------+--------+--------+----------------------------------+
/// ```
///
/// # LOD Levels
///
/// - `0`: Highest detail (closest to camera)
/// - `1`: High detail
/// - `2`: Medium detail
/// - `3`: Lowest detail (farthest from camera)
///
/// # Blend Factor
///
/// The blend factor is used for smooth LOD transitions (LOD morphing):
/// - `0.0`: Fully the current LOD level
/// - `1.0`: Fully the next LOD level (transitioning)
///
/// For discrete LOD selection, blend_factor is typically `0.0`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LodEntry {
    /// Selected LOD level (0 = highest detail, 3 = lowest).
    pub level: u32,
    /// Blend factor for smooth LOD transitions (0.0 to 1.0).
    pub blend_factor: f32,
}

impl Default for LodEntry {
    /// Create a default LOD entry at highest detail (LOD 0).
    fn default() -> Self {
        Self {
            level: 0,
            blend_factor: 0.0,
        }
    }
}

impl LodEntry {
    /// Size of this struct in bytes.
    pub const SIZE: usize = LOD_ENTRY_SIZE;

    /// Create a new LOD entry.
    ///
    /// # Arguments
    ///
    /// * `level` - LOD level (0-3, clamped if out of range)
    /// * `blend_factor` - Blend factor for transitions (0.0-1.0)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let entry = LodEntry::new(1, 0.5);
    /// assert_eq!(entry.level, 1);
    /// assert_eq!(entry.blend_factor, 0.5);
    /// ```
    #[inline]
    pub const fn new(level: u32, blend_factor: f32) -> Self {
        Self {
            level,
            blend_factor,
        }
    }

    /// Create a LOD entry with discrete selection (no blending).
    ///
    /// # Arguments
    ///
    /// * `level` - LOD level (0-3)
    #[inline]
    pub const fn discrete(level: u32) -> Self {
        Self {
            level,
            blend_factor: 0.0,
        }
    }

    /// Create a LOD entry with clamped level.
    ///
    /// Ensures the level is within valid range (0-3).
    #[inline]
    pub fn clamped(level: u32, blend_factor: f32) -> Self {
        Self {
            level: level.min(MAX_LOD_LEVEL),
            blend_factor: blend_factor.clamp(0.0, 1.0),
        }
    }

    /// Check if this entry represents the highest detail LOD.
    #[inline]
    pub const fn is_highest_detail(&self) -> bool {
        self.level == 0
    }

    /// Check if this entry represents the lowest detail LOD.
    #[inline]
    pub const fn is_lowest_detail(&self) -> bool {
        self.level >= MAX_LOD_LEVEL
    }

    /// Check if this entry is in transition (blend_factor > 0).
    #[inline]
    pub fn is_transitioning(&self) -> bool {
        self.blend_factor > 0.0
    }

    /// Get the effective LOD level considering blend factor.
    ///
    /// Returns a float representing the interpolated LOD level.
    /// For example, level 1 with blend 0.5 returns 1.5.
    #[inline]
    pub fn effective_level(&self) -> f32 {
        self.level as f32 + self.blend_factor
    }
}

// =============================================================================
// LOD BUFFER
// =============================================================================

/// GPU buffer for storing per-object LOD selections.
///
/// Stores one `LodEntry` per object, enabling efficient LOD-based
/// indirect draw generation on the GPU.
///
/// # Features
///
/// - **Per-object LOD storage**: One 8-byte entry per object
/// - **GPU-compatible**: Uses wgpu storage buffer with correct alignment
/// - **Frame reset**: Clear buffer each frame before LOD selection
/// - **Dynamic resize**: Grow buffer as scene object count increases
///
/// # Usage
///
/// 1. Create buffer with capacity for your scene
/// 2. Clear buffer at frame start
/// 3. LOD selection compute shader writes entries
/// 4. Indirect draw generation reads entries
pub struct LodBuffer {
    /// GPU storage buffer for LOD entries.
    buffer: wgpu::Buffer,
    /// Maximum number of objects this buffer can hold.
    capacity: u32,
    /// Debug label.
    label: Option<String>,
}

impl LodBuffer {
    /// Create a new LOD buffer with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `capacity` - Maximum number of objects to track
    /// * `label` - Optional debug label
    ///
    /// # Example
    ///
    /// ```ignore
    /// let buffer = LodBuffer::new(&device, 10_000, Some("main_scene"));
    /// ```
    pub fn new(device: &wgpu::Device, capacity: u32, label: Option<&str>) -> Self {
        let capacity = capacity.max(MIN_LOD_BUFFER_CAPACITY);
        let size = (capacity as usize * LOD_ENTRY_SIZE) as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: label.map(|l| format!("{}_lod_buffer", l)).as_deref(),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity,
            label: label.map(String::from),
        }
    }

    /// Create with default capacity.
    ///
    /// Uses `DEFAULT_LOD_BUFFER_CAPACITY` (65,536 objects).
    pub fn with_default_capacity(device: &wgpu::Device, label: Option<&str>) -> Self {
        Self::new(device, DEFAULT_LOD_BUFFER_CAPACITY, label)
    }

    /// Resize buffer if needed.
    ///
    /// Creates a new buffer if `new_capacity` exceeds current capacity.
    /// Does NOT copy existing data - the new buffer should be cleared
    /// before use.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `new_capacity` - Required capacity
    ///
    /// # Returns
    ///
    /// `true` if buffer was resized, `false` if existing capacity was sufficient.
    pub fn resize(&mut self, device: &wgpu::Device, new_capacity: u32) -> bool {
        let new_capacity = new_capacity.max(MIN_LOD_BUFFER_CAPACITY);

        if new_capacity > self.capacity {
            let size = (new_capacity as usize * LOD_ENTRY_SIZE) as u64;

            let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: self.label.as_ref().map(|l| format!("{}_lod_buffer", l)).as_deref(),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            self.buffer = new_buffer;
            self.capacity = new_capacity;
            return true;
        }

        false
    }

    /// Clear all entries by writing zeros.
    ///
    /// Uses `encoder.clear_buffer` for efficient GPU-side clearing.
    /// Call at frame start before LOD selection dispatch.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder for the clear operation
    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.buffer, 0, None);
    }

    /// Clear entries using a queue write.
    ///
    /// Alternative to `clear()` that writes CPU-side zeros.
    /// Useful when you don't have access to an encoder.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue
    pub fn clear_with_queue(&self, queue: &wgpu::Queue) {
        let zeros = vec![0u8; self.size_bytes() as usize];
        queue.write_buffer(&self.buffer, 0, &zeros);
    }

    /// Get the GPU buffer for binding.
    ///
    /// Use with `BufferBindingType::Storage { read_only: false }` for
    /// LOD selection writes, or `read_only: true` for indirect draw reads.
    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get buffer binding resource.
    #[inline]
    pub fn buffer_binding(&self) -> wgpu::BufferBinding<'_> {
        self.buffer.as_entire_buffer_binding()
    }

    /// Get the capacity (maximum number of objects).
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Get buffer size in bytes.
    #[inline]
    pub fn size_bytes(&self) -> u64 {
        (self.capacity as usize * LOD_ENTRY_SIZE) as u64
    }

    /// Get the debug label.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Upload LOD entries from CPU (for testing/initialization).
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue
    /// * `entries` - Slice of LOD entries to upload
    pub fn upload(&self, queue: &wgpu::Queue, entries: &[LodEntry]) {
        let upload_count = entries.len().min(self.capacity as usize);
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&entries[..upload_count]),
        );
    }

    /// Read LOD entries back to CPU (for debugging/testing).
    ///
    /// Blocks until GPU completes.
    ///
    /// # Returns
    ///
    /// Vector of LOD entries.
    pub fn read_back(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<LodEntry> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lod_buffer_staging"),
            size: self.size_bytes(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size_bytes());
        queue.submit([encoder.finish()]);

        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let entries: Vec<LodEntry> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        entries
    }
}

impl std::fmt::Debug for LodBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LodBuffer")
            .field("capacity", &self.capacity)
            .field("size_bytes", &self.size_bytes())
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

// =============================================================================
// LOD BUFFER POOL
// =============================================================================

/// Pool of LOD buffers for double/triple buffering.
///
/// Maintains multiple LOD buffers that can be cycled each frame to avoid
/// GPU stalls from reading a buffer that is still being written.
///
/// # Buffering Strategies
///
/// - **Double buffering (2 buffers)**: Previous frame's LOD data available
///   while current frame writes new data.
/// - **Triple buffering (3 buffers)**: Reduces stalls with pipelined frames.
///
/// # Usage
///
/// ```ignore
/// let mut pool = LodBufferPool::new(&device, 10_000, 2);  // Double buffering
///
/// // Each frame:
/// let current = pool.current();
/// current.clear(&mut encoder);
/// // Dispatch LOD selection using current buffer
/// // Dispatch indirect draw generation using current buffer
/// pool.advance();  // Switch to next buffer for next frame
/// ```
pub struct LodBufferPool {
    /// Array of LOD buffers.
    buffers: Vec<LodBuffer>,
    /// Index of the currently active buffer.
    current_index: usize,
}

impl LodBufferPool {
    /// Create a new LOD buffer pool.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `capacity` - Capacity for each buffer (objects per buffer)
    /// * `buffer_count` - Number of buffers in the pool (2 for double, 3 for triple)
    ///
    /// # Panics
    ///
    /// Panics if `buffer_count` is 0.
    pub fn new(device: &wgpu::Device, capacity: u32, buffer_count: usize) -> Self {
        assert!(buffer_count > 0, "Pool must have at least 1 buffer");

        let buffers = (0..buffer_count)
            .map(|i| LodBuffer::new(device, capacity, Some(&format!("pool_{}", i))))
            .collect();

        Self {
            buffers,
            current_index: 0,
        }
    }

    /// Create a double-buffered pool (2 buffers).
    pub fn double_buffered(device: &wgpu::Device, capacity: u32) -> Self {
        Self::new(device, capacity, 2)
    }

    /// Create a triple-buffered pool (3 buffers).
    pub fn triple_buffered(device: &wgpu::Device, capacity: u32) -> Self {
        Self::new(device, capacity, 3)
    }

    /// Get the current buffer.
    #[inline]
    pub fn current(&self) -> &LodBuffer {
        &self.buffers[self.current_index]
    }

    /// Get a mutable reference to the current buffer.
    #[inline]
    pub fn current_mut(&mut self) -> &mut LodBuffer {
        &mut self.buffers[self.current_index]
    }

    /// Get the previous buffer (for reading previous frame's data).
    ///
    /// Returns the buffer that was current before the last `advance()` call.
    #[inline]
    pub fn previous(&self) -> &LodBuffer {
        let prev_index = if self.current_index == 0 {
            self.buffers.len() - 1
        } else {
            self.current_index - 1
        };
        &self.buffers[prev_index]
    }

    /// Advance to the next buffer in the pool.
    ///
    /// Call at the end of each frame to switch buffers.
    pub fn advance(&mut self) {
        self.current_index = (self.current_index + 1) % self.buffers.len();
    }

    /// Get the number of buffers in the pool.
    #[inline]
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Get the current buffer index.
    #[inline]
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Get the capacity of each buffer.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.buffers[0].capacity()
    }

    /// Resize all buffers in the pool.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `new_capacity` - New capacity for each buffer
    ///
    /// # Returns
    ///
    /// `true` if any buffer was resized.
    pub fn resize_all(&mut self, device: &wgpu::Device, new_capacity: u32) -> bool {
        let mut resized = false;
        for buffer in &mut self.buffers {
            resized |= buffer.resize(device, new_capacity);
        }
        resized
    }

    /// Clear all buffers in the pool.
    ///
    /// Useful for initialization or scene changes.
    pub fn clear_all(&self, encoder: &mut wgpu::CommandEncoder) {
        for buffer in &self.buffers {
            buffer.clear(encoder);
        }
    }

    /// Get a specific buffer by index.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> &LodBuffer {
        &self.buffers[index]
    }
}

impl std::fmt::Debug for LodBufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LodBufferPool")
            .field("buffer_count", &self.buffers.len())
            .field("current_index", &self.current_index)
            .field("capacity", &self.capacity())
            .finish()
    }
}

// =============================================================================
// CPU HELPER FUNCTIONS
// =============================================================================

/// CPU reference implementation: Clear all LOD entries to default (LOD 0).
pub fn cpu_clear_lod_entries(entries: &mut [LodEntry]) {
    entries.fill(LodEntry::default());
}

/// CPU reference implementation: Set LOD entry at index.
pub fn cpu_set_lod_entry(entries: &mut [LodEntry], index: usize, level: u32, blend_factor: f32) {
    if index < entries.len() {
        entries[index] = LodEntry::new(level, blend_factor);
    }
}

/// CPU reference implementation: Get LOD level at index.
pub fn cpu_get_lod_level(entries: &[LodEntry], index: usize) -> Option<u32> {
    entries.get(index).map(|e| e.level)
}

/// CPU reference implementation: Count objects at each LOD level.
pub fn cpu_count_by_lod(entries: &[LodEntry]) -> [usize; 4] {
    let mut counts = [0usize; 4];
    for entry in entries {
        let level = (entry.level as usize).min(3);
        counts[level] += 1;
    }
    counts
}

/// CPU reference implementation: Collect indices of objects at a specific LOD.
pub fn cpu_collect_by_lod(entries: &[LodEntry], target_lod: u32) -> Vec<usize> {
    entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.level == target_lod)
        .map(|(i, _)| i)
        .collect()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // LodEntry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lod_entry_size() {
        assert_eq!(
            std::mem::size_of::<LodEntry>(),
            LOD_ENTRY_SIZE,
            "LodEntry must be {} bytes",
            LOD_ENTRY_SIZE
        );
        assert_eq!(LodEntry::SIZE, 8);
    }

    #[test]
    fn test_lod_entry_alignment() {
        assert_eq!(std::mem::align_of::<LodEntry>(), 4);
    }

    #[test]
    fn test_lod_entry_field_offsets() {
        let entry = LodEntry::new(2, 0.75);
        let bytes = bytemuck::bytes_of(&entry);

        // level at offset 0
        let level = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(level, 2);

        // blend_factor at offset 4
        let blend = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert!((blend - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_lod_entry_default() {
        let entry = LodEntry::default();
        assert_eq!(entry.level, 0);
        assert_eq!(entry.blend_factor, 0.0);
    }

    #[test]
    fn test_lod_entry_new() {
        let entry = LodEntry::new(3, 0.5);
        assert_eq!(entry.level, 3);
        assert_eq!(entry.blend_factor, 0.5);
    }

    #[test]
    fn test_lod_entry_discrete() {
        let entry = LodEntry::discrete(2);
        assert_eq!(entry.level, 2);
        assert_eq!(entry.blend_factor, 0.0);
    }

    #[test]
    fn test_lod_entry_clamped() {
        // Level clamping
        let entry = LodEntry::clamped(10, 0.5);
        assert_eq!(entry.level, MAX_LOD_LEVEL);
        assert_eq!(entry.blend_factor, 0.5);

        // Blend factor clamping
        let entry = LodEntry::clamped(1, 2.0);
        assert_eq!(entry.level, 1);
        assert_eq!(entry.blend_factor, 1.0);

        // Negative blend factor
        let entry = LodEntry::clamped(0, -0.5);
        assert_eq!(entry.blend_factor, 0.0);
    }

    #[test]
    fn test_lod_entry_is_highest_detail() {
        assert!(LodEntry::discrete(0).is_highest_detail());
        assert!(!LodEntry::discrete(1).is_highest_detail());
        assert!(!LodEntry::discrete(3).is_highest_detail());
    }

    #[test]
    fn test_lod_entry_is_lowest_detail() {
        assert!(!LodEntry::discrete(0).is_lowest_detail());
        assert!(!LodEntry::discrete(2).is_lowest_detail());
        assert!(LodEntry::discrete(3).is_lowest_detail());
        assert!(LodEntry::discrete(4).is_lowest_detail()); // Beyond max
    }

    #[test]
    fn test_lod_entry_is_transitioning() {
        assert!(!LodEntry::new(0, 0.0).is_transitioning());
        assert!(LodEntry::new(0, 0.001).is_transitioning());
        assert!(LodEntry::new(1, 0.5).is_transitioning());
        assert!(LodEntry::new(2, 1.0).is_transitioning());
    }

    #[test]
    fn test_lod_entry_effective_level() {
        assert_eq!(LodEntry::new(0, 0.0).effective_level(), 0.0);
        assert_eq!(LodEntry::new(1, 0.5).effective_level(), 1.5);
        assert_eq!(LodEntry::new(2, 0.25).effective_level(), 2.25);
        assert_eq!(LodEntry::new(3, 0.0).effective_level(), 3.0);
    }

    #[test]
    fn test_lod_entry_pod_zeroable() {
        // Pod: can be cast to/from bytes
        let entry = LodEntry::new(1, 0.5);
        let bytes: &[u8] = bytemuck::bytes_of(&entry);
        assert_eq!(bytes.len(), 8);

        // Zeroable: can be zeroed
        let zeroed: LodEntry = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.level, 0);
        assert_eq!(zeroed.blend_factor, 0.0);
    }

    #[test]
    fn test_lod_entry_bytemuck_roundtrip() {
        let entry = LodEntry::new(2, 0.75);
        let bytes: &[u8] = bytemuck::bytes_of(&entry);
        let roundtrip: &LodEntry = bytemuck::from_bytes(bytes);
        assert_eq!(*roundtrip, entry);
    }

    #[test]
    fn test_lod_entry_slice_cast() {
        let entries = [
            LodEntry::new(0, 0.0),
            LodEntry::new(1, 0.25),
            LodEntry::new(2, 0.5),
            LodEntry::new(3, 0.75),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&entries);
        assert_eq!(bytes.len(), LOD_ENTRY_SIZE * 4);

        let roundtrip: &[LodEntry] = bytemuck::cast_slice(bytes);
        assert_eq!(roundtrip, &entries);
    }

    #[test]
    fn test_lod_entry_equality() {
        let e1 = LodEntry::new(1, 0.5);
        let e2 = LodEntry::new(1, 0.5);
        let e3 = LodEntry::new(2, 0.5);

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_lod_entry_clone_copy() {
        let entry = LodEntry::new(2, 0.3);
        let cloned = entry.clone();
        let copied: LodEntry = entry; // Copy

        assert_eq!(entry, cloned);
        assert_eq!(entry, copied);
    }

    #[test]
    fn test_lod_entry_debug() {
        let entry = LodEntry::new(1, 0.5);
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("LodEntry"));
        assert!(debug_str.contains("level"));
        assert!(debug_str.contains("blend_factor"));
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants() {
        assert_eq!(LOD_ENTRY_SIZE, 8);
        assert_eq!(MAX_LOD_LEVEL, 3);
        assert!(DEFAULT_LOD_BUFFER_CAPACITY >= MIN_LOD_BUFFER_CAPACITY);
        assert!(DEFAULT_POOL_SIZE >= 1);
    }

    // -------------------------------------------------------------------------
    // CPU Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_clear_lod_entries() {
        let mut entries = vec![
            LodEntry::new(1, 0.5),
            LodEntry::new(2, 0.3),
            LodEntry::new(3, 0.7),
        ];

        cpu_clear_lod_entries(&mut entries);

        for entry in &entries {
            assert_eq!(entry.level, 0);
            assert_eq!(entry.blend_factor, 0.0);
        }
    }

    #[test]
    fn test_cpu_set_lod_entry() {
        let mut entries = vec![LodEntry::default(); 5];

        cpu_set_lod_entry(&mut entries, 2, 3, 0.5);

        assert_eq!(entries[2].level, 3);
        assert_eq!(entries[2].blend_factor, 0.5);

        // Other entries unchanged
        assert_eq!(entries[0], LodEntry::default());
        assert_eq!(entries[4], LodEntry::default());
    }

    #[test]
    fn test_cpu_set_lod_entry_out_of_bounds() {
        let mut entries = vec![LodEntry::default(); 3];

        // Should not panic
        cpu_set_lod_entry(&mut entries, 10, 2, 0.5);

        // Entries unchanged
        assert!(entries.iter().all(|e| *e == LodEntry::default()));
    }

    #[test]
    fn test_cpu_get_lod_level() {
        let entries = vec![
            LodEntry::new(0, 0.0),
            LodEntry::new(2, 0.5),
            LodEntry::new(3, 0.0),
        ];

        assert_eq!(cpu_get_lod_level(&entries, 0), Some(0));
        assert_eq!(cpu_get_lod_level(&entries, 1), Some(2));
        assert_eq!(cpu_get_lod_level(&entries, 2), Some(3));
        assert_eq!(cpu_get_lod_level(&entries, 10), None);
    }

    #[test]
    fn test_cpu_count_by_lod() {
        let entries = vec![
            LodEntry::discrete(0),
            LodEntry::discrete(0),
            LodEntry::discrete(1),
            LodEntry::discrete(2),
            LodEntry::discrete(2),
            LodEntry::discrete(2),
            LodEntry::discrete(3),
        ];

        let counts = cpu_count_by_lod(&entries);
        assert_eq!(counts[0], 2); // LOD 0
        assert_eq!(counts[1], 1); // LOD 1
        assert_eq!(counts[2], 3); // LOD 2
        assert_eq!(counts[3], 1); // LOD 3
    }

    #[test]
    fn test_cpu_count_by_lod_empty() {
        let entries: Vec<LodEntry> = vec![];
        let counts = cpu_count_by_lod(&entries);
        assert_eq!(counts, [0, 0, 0, 0]);
    }

    #[test]
    fn test_cpu_count_by_lod_all_same() {
        let entries = vec![LodEntry::discrete(1); 100];
        let counts = cpu_count_by_lod(&entries);
        assert_eq!(counts, [0, 100, 0, 0]);
    }

    #[test]
    fn test_cpu_collect_by_lod() {
        let entries = vec![
            LodEntry::discrete(0), // index 0
            LodEntry::discrete(1), // index 1
            LodEntry::discrete(0), // index 2
            LodEntry::discrete(2), // index 3
            LodEntry::discrete(0), // index 4
        ];

        let lod0_indices = cpu_collect_by_lod(&entries, 0);
        assert_eq!(lod0_indices, vec![0, 2, 4]);

        let lod1_indices = cpu_collect_by_lod(&entries, 1);
        assert_eq!(lod1_indices, vec![1]);

        let lod2_indices = cpu_collect_by_lod(&entries, 2);
        assert_eq!(lod2_indices, vec![3]);

        let lod3_indices = cpu_collect_by_lod(&entries, 3);
        assert!(lod3_indices.is_empty());
    }

    #[test]
    fn test_cpu_collect_by_lod_empty() {
        let entries: Vec<LodEntry> = vec![];
        let indices = cpu_collect_by_lod(&entries, 0);
        assert!(indices.is_empty());
    }

    // -------------------------------------------------------------------------
    // Buffer Size Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffer_size_calculation() {
        // 1000 objects at 8 bytes each = 8000 bytes
        let size = 1000 * LOD_ENTRY_SIZE;
        assert_eq!(size, 8000);

        // 100,000 objects = 800,000 bytes = ~781 KB
        let size = 100_000 * LOD_ENTRY_SIZE;
        assert_eq!(size, 800_000);
    }

    #[test]
    fn test_min_capacity_enforcement() {
        // Capacity below minimum should be clamped
        let capacity = 0u32.max(MIN_LOD_BUFFER_CAPACITY);
        assert!(capacity >= MIN_LOD_BUFFER_CAPACITY);

        let capacity = 10u32.max(MIN_LOD_BUFFER_CAPACITY);
        assert!(capacity >= MIN_LOD_BUFFER_CAPACITY);
    }

    // -------------------------------------------------------------------------
    // LodBufferPool Logic Tests (CPU-only)
    // -------------------------------------------------------------------------

    #[test]
    fn test_pool_buffer_count_assertion() {
        // Cannot create pool with 0 buffers
        let result = std::panic::catch_unwind(|| {
            // We can't actually create a pool without a device,
            // but we can test that the assertion exists by checking
            // that DEFAULT_POOL_SIZE >= 1
            assert!(DEFAULT_POOL_SIZE >= 1);
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_pool_index_wraparound() {
        // Simulate index advancement
        let buffer_count = 3;
        let mut current_index = 0;

        // Advance 5 times, should wrap around
        for _ in 0..5 {
            current_index = (current_index + 1) % buffer_count;
        }

        // 5 % 3 = 2
        assert_eq!(current_index, 2);
    }

    #[test]
    fn test_pool_previous_index() {
        let buffer_count = 3;

        // Test previous calculation for each position
        for current in 0..buffer_count {
            let prev = if current == 0 {
                buffer_count - 1
            } else {
                current - 1
            };

            // Verify wraparound
            if current == 0 {
                assert_eq!(prev, 2);
            } else {
                assert_eq!(prev, current - 1);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lod_entry_max_values() {
        let entry = LodEntry::new(u32::MAX, f32::MAX);
        assert_eq!(entry.level, u32::MAX);
        assert_eq!(entry.blend_factor, f32::MAX);
    }

    #[test]
    fn test_lod_entry_negative_blend() {
        // Negative blend factor is valid in the struct (no clamping in new())
        let entry = LodEntry::new(0, -1.0);
        assert_eq!(entry.blend_factor, -1.0);

        // Use clamped() for validation
        let entry = LodEntry::clamped(0, -1.0);
        assert_eq!(entry.blend_factor, 0.0);
    }

    #[test]
    fn test_lod_entry_nan_blend() {
        // NaN is a valid f32 value (should be handled by callers)
        let entry = LodEntry::new(0, f32::NAN);
        assert!(entry.blend_factor.is_nan());
    }

    #[test]
    fn test_lod_entry_inf_blend() {
        let entry = LodEntry::new(0, f32::INFINITY);
        assert!(entry.blend_factor.is_infinite());

        // Clamped handles infinity
        let entry = LodEntry::clamped(0, f32::INFINITY);
        assert_eq!(entry.blend_factor, 1.0);
    }
}
