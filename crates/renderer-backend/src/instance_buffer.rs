//! Instance Buffer Management for TLAS Construction
//!
//! This module provides CPU-to-GPU instance buffer upload infrastructure
//! for per-frame TLAS construction in ray tracing pipelines.
//!
//! # Architecture
//!
//! - `InstanceBufferConfig`: Configuration for buffer capacity and double-buffering
//! - `InstanceBuffer`: Ping-pong buffer for GPU upload without stalls
//! - `InstanceCollector`: Scene traversal helper for gathering instances
//!
//! # Double-Buffering
//!
//! The ping-pong double-buffer strategy allows CPU writes to one buffer
//! while the GPU reads from the other. This avoids GPU stalls when
//! uploading per-frame instance transforms.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::instance_buffer::{InstanceBuffer, InstanceBufferConfig, InstanceCollector};
//! use renderer_backend::tlas::TlasInstance;
//!
//! // Create double-buffered instance buffer
//! let config = InstanceBufferConfig::default()
//!     .with_max_instances(1000)
//!     .with_double_buffer(true);
//!
//! let mut buffer = InstanceBuffer::new(config);
//!
//! // Each frame:
//! buffer.begin_frame();
//!
//! // Collect instances from scene
//! let mut collector = InstanceCollector::new();
//! collector.collect_from_mesh(blas_handle, &transform, 0);
//! let instances = collector.finish();
//!
//! // Add to buffer
//! buffer.add_instances(&instances)?;
//!
//! // Upload to GPU (current_buffer slice)
//! let data = buffer.current_buffer();
//! ```

use crate::blas_pool::BlasHandle;
use crate::tlas::TlasInstance;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum instances for the buffer.
const DEFAULT_MAX_INSTANCES: u32 = 65536;

/// Size of TlasInstance in bytes (64 bytes for GPU layout).
const TLAS_INSTANCE_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// InstanceBufferError
// ---------------------------------------------------------------------------

/// Errors that can occur during instance buffer operations.
#[derive(Debug)]
pub enum InstanceBufferError {
    /// Buffer capacity exceeded.
    BufferFull { max: u32 },
    /// Instance transform contains invalid values (NaN, Inf).
    InvalidTransform,
}

impl std::fmt::Display for InstanceBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferFull { max } => {
                write!(f, "instance buffer full: max capacity is {}", max)
            }
            Self::InvalidTransform => {
                write!(f, "instance transform contains invalid values (NaN or Inf)")
            }
        }
    }
}

impl std::error::Error for InstanceBufferError {}

// ---------------------------------------------------------------------------
// InstanceBufferConfig
// ---------------------------------------------------------------------------

/// Configuration for instance buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceBufferConfig {
    /// Maximum number of instances the buffer can hold.
    pub max_instances: u32,
    /// Enable double-buffering (ping-pong) to avoid GPU stalls.
    pub double_buffer: bool,
}

impl InstanceBufferConfig {
    /// Create a new config with default settings.
    pub const fn new() -> Self {
        Self {
            max_instances: DEFAULT_MAX_INSTANCES,
            double_buffer: true,
        }
    }

    /// Set maximum instance capacity.
    pub const fn with_max_instances(mut self, max: u32) -> Self {
        self.max_instances = max;
        self
    }

    /// Enable or disable double-buffering.
    pub const fn with_double_buffer(mut self, enable: bool) -> Self {
        self.double_buffer = enable;
        self
    }

    /// Configuration for static scenes (single buffer, large capacity).
    pub const fn for_static_scene() -> Self {
        Self {
            max_instances: DEFAULT_MAX_INSTANCES,
            double_buffer: false,
        }
    }

    /// Configuration for dynamic scenes (double buffer, moderate capacity).
    pub const fn for_dynamic_scene() -> Self {
        Self {
            max_instances: DEFAULT_MAX_INSTANCES,
            double_buffer: true,
        }
    }
}

impl Default for InstanceBufferConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// InstanceBuffer
// ---------------------------------------------------------------------------

/// Double-buffered instance buffer for per-frame TLAS construction.
///
/// Uses ping-pong buffering to allow CPU writes while GPU reads,
/// avoiding synchronization stalls.
#[derive(Debug)]
pub struct InstanceBuffer {
    /// Configuration.
    config: InstanceBufferConfig,
    /// Ping-pong buffers (only [0] used if double_buffer is false).
    buffers: [Vec<TlasInstance>; 2],
    /// Current buffer index (0 or 1).
    current_buffer: usize,
    /// Number of instances in the current buffer.
    instance_count: u32,
    /// Current frame number.
    frame: u64,
}

impl InstanceBuffer {
    /// Create a new instance buffer with the given configuration.
    pub fn new(config: InstanceBufferConfig) -> Self {
        let capacity = config.max_instances as usize;
        Self {
            config,
            buffers: [
                Vec::with_capacity(capacity),
                Vec::with_capacity(capacity),
            ],
            current_buffer: 0,
            instance_count: 0,
            frame: 0,
        }
    }

    /// Begin a new frame: swap buffers and reset count.
    ///
    /// If double-buffering is enabled, this swaps to the other buffer.
    /// The previous buffer remains intact for GPU reads.
    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);

        if self.config.double_buffer {
            // Swap to the other buffer
            self.current_buffer = 1 - self.current_buffer;
        }

        // Clear current buffer for new frame
        self.buffers[self.current_buffer].clear();
        self.instance_count = 0;
    }

    /// Add a single instance to the buffer.
    ///
    /// # Returns
    ///
    /// The index of the added instance on success.
    ///
    /// # Errors
    ///
    /// Returns `InstanceBufferError::BufferFull` if capacity is exceeded.
    /// Returns `InstanceBufferError::InvalidTransform` if transform contains NaN or Inf.
    pub fn add_instance(&mut self, instance: TlasInstance) -> Result<u32, InstanceBufferError> {
        if self.instance_count >= self.config.max_instances {
            return Err(InstanceBufferError::BufferFull {
                max: self.config.max_instances,
            });
        }

        // Validate transform
        if !Self::is_valid_transform(&instance.transform) {
            return Err(InstanceBufferError::InvalidTransform);
        }

        let index = self.instance_count;
        self.buffers[self.current_buffer].push(instance);
        self.instance_count += 1;

        Ok(index)
    }

    /// Add multiple instances to the buffer.
    ///
    /// # Errors
    ///
    /// Returns `InstanceBufferError::BufferFull` if adding all instances
    /// would exceed capacity. In this case, no instances are added.
    pub fn add_instances(&mut self, instances: &[TlasInstance]) -> Result<(), InstanceBufferError> {
        let new_count = self.instance_count + instances.len() as u32;

        if new_count > self.config.max_instances {
            return Err(InstanceBufferError::BufferFull {
                max: self.config.max_instances,
            });
        }

        // Validate all transforms first
        for instance in instances {
            if !Self::is_valid_transform(&instance.transform) {
                return Err(InstanceBufferError::InvalidTransform);
            }
        }

        self.buffers[self.current_buffer].extend_from_slice(instances);
        self.instance_count = new_count;

        Ok(())
    }

    /// Get the current instance count.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Get the current buffer slice for GPU upload.
    pub fn current_buffer(&self) -> &[TlasInstance] {
        &self.buffers[self.current_buffer]
    }

    /// Get the previous buffer slice (GPU may still be reading).
    ///
    /// Returns the current buffer if double-buffering is disabled.
    pub fn previous_buffer(&self) -> &[TlasInstance] {
        if self.config.double_buffer {
            &self.buffers[1 - self.current_buffer]
        } else {
            &self.buffers[self.current_buffer]
        }
    }

    /// Calculate the upload size in bytes.
    pub fn upload_size(&self) -> usize {
        self.instance_count as usize * TLAS_INSTANCE_SIZE
    }

    /// Clear the current buffer.
    pub fn clear(&mut self) {
        self.buffers[self.current_buffer].clear();
        self.instance_count = 0;
    }

    /// Get the current frame number.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Get the configuration.
    pub fn config(&self) -> &InstanceBufferConfig {
        &self.config
    }

    /// Get the current buffer index.
    pub fn current_buffer_index(&self) -> usize {
        self.current_buffer
    }

    /// Get remaining capacity.
    pub fn remaining_capacity(&self) -> u32 {
        self.config.max_instances.saturating_sub(self.instance_count)
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.instance_count == 0
    }

    /// Check if buffer is full.
    pub fn is_full(&self) -> bool {
        self.instance_count >= self.config.max_instances
    }

    /// Validate that a transform matrix contains no NaN or Inf values.
    fn is_valid_transform(transform: &[[f32; 4]; 3]) -> bool {
        for row in transform {
            for val in row {
                if !val.is_finite() {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// InstanceCollector
// ---------------------------------------------------------------------------

/// Helper for collecting TLAS instances from scene traversal.
///
/// Provides a convenient API for gathering instances from meshes
/// during scene graph traversal.
#[derive(Debug, Default)]
pub struct InstanceCollector {
    /// Collected instances.
    instances: Vec<TlasInstance>,
    /// Visibility mask filter (instances must match this mask).
    mask_filter: u8,
}

impl InstanceCollector {
    /// Create a new collector with no mask filter.
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            mask_filter: 0xFF, // Accept all by default
        }
    }

    /// Create a collector with a visibility mask filter.
    ///
    /// Only instances whose mask ANDs non-zero with the filter are collected.
    pub fn with_mask_filter(mask: u8) -> Self {
        Self {
            instances: Vec::new(),
            mask_filter: mask,
        }
    }

    /// Collect an instance from a mesh.
    ///
    /// # Arguments
    ///
    /// * `blas_handle` - Handle to the mesh's BLAS in the pool
    /// * `transform` - 3x4 row-major affine transform (as flat array)
    /// * `custom_index` - Per-instance custom index (shader binding)
    pub fn collect_from_mesh(
        &mut self,
        blas_handle: BlasHandle,
        transform: &[f32; 12],
        custom_index: u32,
    ) {
        self.collect_from_mesh_raw(blas_handle.id() as u64, transform, custom_index);
    }

    /// Collect an instance using a raw BLAS address.
    ///
    /// # Arguments
    ///
    /// * `blas_address` - GPU address of the BLAS
    /// * `transform` - 3x4 row-major affine transform (as flat array)
    /// * `custom_index` - Per-instance custom index (shader binding)
    pub fn collect_from_mesh_raw(
        &mut self,
        blas_address: u64,
        transform: &[f32; 12],
        custom_index: u32,
    ) {
        // Convert flat array to 3x4 matrix
        let transform_matrix: [[f32; 4]; 3] = [
            [transform[0], transform[1], transform[2], transform[3]],
            [transform[4], transform[5], transform[6], transform[7]],
            [transform[8], transform[9], transform[10], transform[11]],
        ];

        let instance = TlasInstance::new()
            .with_transform(transform_matrix)
            .with_blas_address(blas_address)
            .with_custom_index(custom_index)
            .with_mask(self.mask_filter);

        self.instances.push(instance);
    }

    /// Collect an instance with a pre-built TlasInstance.
    pub fn collect_instance(&mut self, instance: TlasInstance) {
        // Apply mask filter
        if (instance.mask & self.mask_filter) != 0 {
            self.instances.push(instance);
        }
    }

    /// Collect multiple instances.
    pub fn collect_instances(&mut self, instances: &[TlasInstance]) {
        for instance in instances {
            self.collect_instance(*instance);
        }
    }

    /// Get the current collected count.
    pub fn count(&self) -> usize {
        self.instances.len()
    }

    /// Check if collector is empty.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Get the mask filter.
    pub fn mask_filter(&self) -> u8 {
        self.mask_filter
    }

    /// Finish collection and return instances.
    ///
    /// Consumes the collector.
    pub fn finish(self) -> Vec<TlasInstance> {
        self.instances
    }

    /// Clear collected instances without consuming.
    pub fn clear(&mut self) {
        self.instances.clear();
    }

    /// Get a slice of collected instances.
    pub fn instances(&self) -> &[TlasInstance] {
        &self.instances
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------------------------

    /// Create a test instance with a given BLAS address.
    fn make_test_instance(blas_address: u64) -> TlasInstance {
        TlasInstance::new().with_blas_address(blas_address)
    }

    /// Create test instances.
    fn make_test_instances(count: usize) -> Vec<TlasInstance> {
        (0..count)
            .map(|i| make_test_instance((i + 1) as u64))
            .collect()
    }

    /// Identity transform as flat array.
    fn identity_flat() -> [f32; 12] {
        [
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
        ]
    }

    /// Create an instance with invalid transform (NaN).
    fn make_invalid_instance() -> TlasInstance {
        TlasInstance::new()
            .with_blas_address(1)
            .with_transform([
                [f32::NAN, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ])
    }

    /// Create a mock BLAS address for testing (simulates BlasHandle::id()).
    fn mock_blas_address(id: u32) -> u64 {
        id as u64
    }

    // -------------------------------------------------------------------------
    // Test: Config defaults
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = InstanceBufferConfig::default();

        assert_eq!(config.max_instances, DEFAULT_MAX_INSTANCES);
        assert!(config.double_buffer);
    }

    #[test]
    fn test_config_new() {
        let config = InstanceBufferConfig::new();

        assert_eq!(config.max_instances, DEFAULT_MAX_INSTANCES);
        assert!(config.double_buffer);
    }

    #[test]
    fn test_config_with_max_instances() {
        let config = InstanceBufferConfig::new().with_max_instances(1000);

        assert_eq!(config.max_instances, 1000);
    }

    #[test]
    fn test_config_with_double_buffer() {
        let config = InstanceBufferConfig::new().with_double_buffer(false);

        assert!(!config.double_buffer);
    }

    #[test]
    fn test_config_for_static_scene() {
        let config = InstanceBufferConfig::for_static_scene();

        assert!(!config.double_buffer);
    }

    #[test]
    fn test_config_for_dynamic_scene() {
        let config = InstanceBufferConfig::for_dynamic_scene();

        assert!(config.double_buffer);
    }

    // -------------------------------------------------------------------------
    // Test: Single instance add
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_single_instance() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        let instance = make_test_instance(1);
        let index = buffer.add_instance(instance).expect("Should succeed");

        assert_eq!(index, 0);
        assert_eq!(buffer.instance_count(), 1);
    }

    #[test]
    fn test_add_multiple_single_instances() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        for i in 0..5 {
            let instance = make_test_instance((i + 1) as u64);
            let index = buffer.add_instance(instance).expect("Should succeed");
            assert_eq!(index, i as u32);
        }

        assert_eq!(buffer.instance_count(), 5);
    }

    // -------------------------------------------------------------------------
    // Test: Batch instance add
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_instances_batch() {
        let config = InstanceBufferConfig::new().with_max_instances(100);
        let mut buffer = InstanceBuffer::new(config);

        let instances = make_test_instances(50);
        buffer.add_instances(&instances).expect("Should succeed");

        assert_eq!(buffer.instance_count(), 50);
    }

    #[test]
    fn test_add_instances_empty_batch() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        let instances: Vec<TlasInstance> = Vec::new();
        buffer.add_instances(&instances).expect("Should succeed");

        assert_eq!(buffer.instance_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Buffer full error
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffer_full_single_add() {
        let config = InstanceBufferConfig::new().with_max_instances(5);
        let mut buffer = InstanceBuffer::new(config);

        // Fill buffer
        for i in 0..5 {
            buffer
                .add_instance(make_test_instance((i + 1) as u64))
                .expect("Should succeed");
        }

        // One more should fail
        let result = buffer.add_instance(make_test_instance(100));
        assert!(matches!(
            result,
            Err(InstanceBufferError::BufferFull { max: 5 })
        ));
    }

    #[test]
    fn test_buffer_full_batch_add() {
        let config = InstanceBufferConfig::new().with_max_instances(5);
        let mut buffer = InstanceBuffer::new(config);

        // Add 3 instances
        buffer
            .add_instances(&make_test_instances(3))
            .expect("Should succeed");

        // Try to add 5 more (would exceed capacity)
        let result = buffer.add_instances(&make_test_instances(5));
        assert!(matches!(
            result,
            Err(InstanceBufferError::BufferFull { max: 5 })
        ));

        // Original 3 should still be there
        assert_eq!(buffer.instance_count(), 3);
    }

    // -------------------------------------------------------------------------
    // Test: Invalid transform
    // -------------------------------------------------------------------------

    #[test]
    fn test_invalid_transform_nan() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        let invalid = make_invalid_instance();
        let result = buffer.add_instance(invalid);

        assert!(matches!(result, Err(InstanceBufferError::InvalidTransform)));
    }

    #[test]
    fn test_invalid_transform_inf() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        let invalid = TlasInstance::new()
            .with_blas_address(1)
            .with_transform([
                [f32::INFINITY, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]);
        let result = buffer.add_instance(invalid);

        assert!(matches!(result, Err(InstanceBufferError::InvalidTransform)));
    }

    #[test]
    fn test_invalid_transform_batch() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        let instances = vec![make_test_instance(1), make_invalid_instance()];
        let result = buffer.add_instances(&instances);

        assert!(matches!(result, Err(InstanceBufferError::InvalidTransform)));
        // No instances should be added on error
        assert_eq!(buffer.instance_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Ping-pong buffer swap
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffer_swap() {
        let config = InstanceBufferConfig::new()
            .with_max_instances(10)
            .with_double_buffer(true);
        let mut buffer = InstanceBuffer::new(config);

        // Frame 0: buffer index 0
        assert_eq!(buffer.current_buffer_index(), 0);

        // Add instance to buffer 0
        buffer.add_instance(make_test_instance(1)).unwrap();
        assert_eq!(buffer.instance_count(), 1);

        // Frame 1: swap to buffer 1
        buffer.begin_frame();
        assert_eq!(buffer.current_buffer_index(), 1);
        assert_eq!(buffer.instance_count(), 0); // New buffer is empty

        // Previous buffer should still have data
        assert_eq!(buffer.previous_buffer().len(), 1);
    }

    #[test]
    fn test_double_buffer_preserves_previous() {
        let config = InstanceBufferConfig::new()
            .with_max_instances(10)
            .with_double_buffer(true);
        let mut buffer = InstanceBuffer::new(config);

        // Frame 0
        buffer.add_instance(make_test_instance(1)).unwrap();
        buffer.add_instance(make_test_instance(2)).unwrap();

        // Frame 1
        buffer.begin_frame();
        buffer.add_instance(make_test_instance(10)).unwrap();

        // Current buffer has 1, previous has 2
        assert_eq!(buffer.current_buffer().len(), 1);
        assert_eq!(buffer.previous_buffer().len(), 2);
    }

    #[test]
    fn test_single_buffer_no_swap() {
        let config = InstanceBufferConfig::new()
            .with_max_instances(10)
            .with_double_buffer(false);
        let mut buffer = InstanceBuffer::new(config);

        // Add instance
        buffer.add_instance(make_test_instance(1)).unwrap();

        // Begin frame should clear (no double buffer)
        buffer.begin_frame();
        assert_eq!(buffer.current_buffer_index(), 0);
        assert_eq!(buffer.instance_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Frame begin resets count
    // -------------------------------------------------------------------------

    #[test]
    fn test_begin_frame_resets_count() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        buffer
            .add_instances(&make_test_instances(5))
            .expect("Should succeed");
        assert_eq!(buffer.instance_count(), 5);

        buffer.begin_frame();
        assert_eq!(buffer.instance_count(), 0);
    }

    #[test]
    fn test_begin_frame_increments_frame() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        assert_eq!(buffer.frame(), 0);

        buffer.begin_frame();
        assert_eq!(buffer.frame(), 1);

        buffer.begin_frame();
        assert_eq!(buffer.frame(), 2);
    }

    // -------------------------------------------------------------------------
    // Test: Instance count tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_instance_count_tracking() {
        let config = InstanceBufferConfig::new().with_max_instances(100);
        let mut buffer = InstanceBuffer::new(config);

        assert_eq!(buffer.instance_count(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());

        buffer.add_instance(make_test_instance(1)).unwrap();
        assert_eq!(buffer.instance_count(), 1);
        assert!(!buffer.is_empty());

        buffer.add_instances(&make_test_instances(10)).unwrap();
        assert_eq!(buffer.instance_count(), 11);
    }

    #[test]
    fn test_remaining_capacity() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        assert_eq!(buffer.remaining_capacity(), 10);

        buffer.add_instances(&make_test_instances(3)).unwrap();
        assert_eq!(buffer.remaining_capacity(), 7);

        buffer.add_instances(&make_test_instances(7)).unwrap();
        assert_eq!(buffer.remaining_capacity(), 0);
        assert!(buffer.is_full());
    }

    // -------------------------------------------------------------------------
    // Test: Upload size calculation
    // -------------------------------------------------------------------------

    #[test]
    fn test_upload_size_empty() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let buffer = InstanceBuffer::new(config);

        assert_eq!(buffer.upload_size(), 0);
    }

    #[test]
    fn test_upload_size_single() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        buffer.add_instance(make_test_instance(1)).unwrap();
        assert_eq!(buffer.upload_size(), TLAS_INSTANCE_SIZE);
    }

    #[test]
    fn test_upload_size_multiple() {
        let config = InstanceBufferConfig::new().with_max_instances(100);
        let mut buffer = InstanceBuffer::new(config);

        buffer.add_instances(&make_test_instances(50)).unwrap();
        assert_eq!(buffer.upload_size(), 50 * TLAS_INSTANCE_SIZE);
    }

    // -------------------------------------------------------------------------
    // Test: Clear
    // -------------------------------------------------------------------------

    #[test]
    fn test_clear() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        buffer.add_instances(&make_test_instances(5)).unwrap();
        assert_eq!(buffer.instance_count(), 5);

        buffer.clear();
        assert_eq!(buffer.instance_count(), 0);
        assert!(buffer.is_empty());
    }

    // -------------------------------------------------------------------------
    // Test: Collector with mask filter
    // -------------------------------------------------------------------------

    #[test]
    fn test_collector_new() {
        let collector = InstanceCollector::new();

        assert!(collector.is_empty());
        assert_eq!(collector.count(), 0);
        assert_eq!(collector.mask_filter(), 0xFF);
    }

    #[test]
    fn test_collector_with_mask_filter() {
        let collector = InstanceCollector::with_mask_filter(0b10101010);

        assert_eq!(collector.mask_filter(), 0b10101010);
    }

    #[test]
    fn test_collector_collect_from_mesh() {
        let mut collector = InstanceCollector::new();
        let blas_address = mock_blas_address(42);
        let transform = identity_flat();

        collector.collect_from_mesh_raw(blas_address, &transform, 100);

        assert_eq!(collector.count(), 1);
        let instances = collector.finish();
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].blas_address, 42);
        assert_eq!(instances[0].custom_index(), 100);
    }

    #[test]
    fn test_collector_mask_filter_accepts() {
        let mut collector = InstanceCollector::with_mask_filter(0b00001111);

        // Instance with mask 0x0F should be accepted
        let instance = TlasInstance::new()
            .with_blas_address(1)
            .with_mask(0b00001111);
        collector.collect_instance(instance);

        assert_eq!(collector.count(), 1);
    }

    #[test]
    fn test_collector_mask_filter_rejects() {
        let mut collector = InstanceCollector::with_mask_filter(0b00001111);

        // Instance with mask 0xF0 should be rejected (no overlap)
        let instance = TlasInstance::new()
            .with_blas_address(1)
            .with_mask(0b11110000);
        collector.collect_instance(instance);

        assert_eq!(collector.count(), 0);
    }

    #[test]
    fn test_collector_collect_instances() {
        let mut collector = InstanceCollector::new();

        let instances = make_test_instances(10);
        collector.collect_instances(&instances);

        assert_eq!(collector.count(), 10);
    }

    #[test]
    fn test_collector_clear() {
        let mut collector = InstanceCollector::new();

        collector.collect_instances(&make_test_instances(5));
        assert_eq!(collector.count(), 5);

        collector.clear();
        assert!(collector.is_empty());
    }

    #[test]
    fn test_collector_finish() {
        let mut collector = InstanceCollector::new();

        collector.collect_from_mesh_raw(mock_blas_address(1), &identity_flat(), 0);
        collector.collect_from_mesh_raw(mock_blas_address(2), &identity_flat(), 1);

        let instances = collector.finish();
        assert_eq!(instances.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Test: Large batch (1000 instances)
    // -------------------------------------------------------------------------

    #[test]
    fn test_large_batch_1000_instances() {
        let config = InstanceBufferConfig::new().with_max_instances(2000);
        let mut buffer = InstanceBuffer::new(config);

        let instances = make_test_instances(1000);
        buffer.add_instances(&instances).expect("Should succeed");

        assert_eq!(buffer.instance_count(), 1000);
        assert_eq!(buffer.upload_size(), 1000 * TLAS_INSTANCE_SIZE);
        assert_eq!(buffer.current_buffer().len(), 1000);
    }

    #[test]
    fn test_large_batch_collector() {
        let mut collector = InstanceCollector::new();

        for i in 0..1000 {
            collector.collect_from_mesh_raw(mock_blas_address(i as u32), &identity_flat(), i as u32);
        }

        assert_eq!(collector.count(), 1000);
        let instances = collector.finish();
        assert_eq!(instances.len(), 1000);
    }

    // -------------------------------------------------------------------------
    // Test: Error display
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display_buffer_full() {
        let err = InstanceBufferError::BufferFull { max: 100 };
        let msg = format!("{}", err);

        assert!(msg.contains("100"));
        assert!(msg.contains("full"));
    }

    #[test]
    fn test_error_display_invalid_transform() {
        let err = InstanceBufferError::InvalidTransform;
        let msg = format!("{}", err);

        assert!(msg.contains("invalid") || msg.contains("NaN") || msg.contains("Inf"));
    }

    // -------------------------------------------------------------------------
    // Test: Edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_zero_max_instances() {
        let config = InstanceBufferConfig::new().with_max_instances(0);
        let mut buffer = InstanceBuffer::new(config);

        let result = buffer.add_instance(make_test_instance(1));
        assert!(matches!(
            result,
            Err(InstanceBufferError::BufferFull { max: 0 })
        ));
    }

    #[test]
    fn test_current_buffer_empty() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let buffer = InstanceBuffer::new(config);

        assert!(buffer.current_buffer().is_empty());
    }

    #[test]
    fn test_frame_wrapping() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        // Force near max u64
        buffer.frame = u64::MAX;
        buffer.begin_frame();

        assert_eq!(buffer.frame(), 0); // Wrapped
    }

    #[test]
    fn test_transform_validation_negative_infinity() {
        let config = InstanceBufferConfig::new().with_max_instances(10);
        let mut buffer = InstanceBuffer::new(config);

        let invalid = TlasInstance::new()
            .with_blas_address(1)
            .with_transform([
                [f32::NEG_INFINITY, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]);
        let result = buffer.add_instance(invalid);

        assert!(matches!(result, Err(InstanceBufferError::InvalidTransform)));
    }
}
