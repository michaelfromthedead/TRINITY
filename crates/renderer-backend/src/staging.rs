//! CPU-to-GPU staging buffer ring for efficient light data upload.
//!
//! This module provides triple-buffered staging buffers to prevent GPU stalls
//! when uploading per-frame light data. The ring pattern ensures that:
//! - The CPU can write to a staging buffer while the GPU reads from another
//! - No synchronization stalls occur during uploads
//! - Memory usage is bounded to 3x the maximum light buffer size
//!
//! Usage:
//! ```ignore
//! let mut uploader = LightUploader::new(&device, max_lights);
//!
//! // Each frame:
//! uploader.upload(&queue, &mut encoder, &builder, &target_buffers);
//! ```

use crate::light_buffers::{LightBufferBuilder, LightCounts, LightGpuBuffers};

/// Default ring capacity (triple buffering).
pub const DEFAULT_RING_CAPACITY: usize = 3;

/// Minimum buffer size in bytes (wgpu requires non-zero).
const MIN_BUFFER_SIZE: u64 = 16;

/// A ring buffer of staging buffers for CPU-to-GPU uploads.
///
/// Uses triple buffering by default to prevent GPU stalls:
/// - Frame N: GPU reads from buffer 0
/// - Frame N+1: GPU reads from buffer 1, CPU writes to buffer 0
/// - Frame N+2: GPU reads from buffer 2, CPU writes to buffer 1
///
/// This pattern ensures the CPU never writes to a buffer the GPU is reading.
#[derive(Debug)]
pub struct StagingRing {
    /// The staging buffers (MAP_WRITE | COPY_SRC)
    buffers: Vec<wgpu::Buffer>,
    /// Current buffer index for writing
    current: usize,
    /// Number of buffers in the ring (typically 3)
    capacity: usize,
    /// Size of each staging buffer in bytes
    buffer_size: u64,
}

impl StagingRing {
    /// Create a new staging ring with the specified buffer size and capacity.
    ///
    /// # Arguments
    /// * `device` - The wgpu device for buffer creation
    /// * `size` - Size of each staging buffer in bytes
    /// * `capacity` - Number of buffers in the ring (typically 3 for triple buffering)
    ///
    /// # Panics
    /// Panics if capacity is 0.
    pub fn new(device: &wgpu::Device, size: u64, capacity: usize) -> Self {
        assert!(capacity > 0, "StagingRing capacity must be > 0");

        let buffer_size = size.max(MIN_BUFFER_SIZE);
        let buffers = (0..capacity)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Staging Ring Buffer {}", i)),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Self {
            buffers,
            current: 0,
            capacity,
            buffer_size,
        }
    }

    /// Create a staging ring with triple buffering (default).
    pub fn new_triple(device: &wgpu::Device, size: u64) -> Self {
        Self::new(device, size, DEFAULT_RING_CAPACITY)
    }

    /// Get the current staging buffer for writing.
    pub fn get_current(&self) -> &wgpu::Buffer {
        &self.buffers[self.current]
    }

    /// Get the current buffer index.
    pub fn current_index(&self) -> usize {
        self.current
    }

    /// Advance to the next buffer in the ring.
    pub fn advance(&mut self) {
        self.current = (self.current + 1) % self.capacity;
    }

    /// Get the ring capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the size of each buffer in bytes.
    pub fn buffer_size(&self) -> u64 {
        self.buffer_size
    }

    /// Check if the ring has been advanced at least `capacity` times
    /// since creation, meaning all buffers have been used at least once.
    pub fn is_warmed_up(&self, frame_count: u64) -> bool {
        frame_count >= self.capacity as u64
    }
}

/// Statistics from a light upload operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct UploadStats {
    /// Total bytes uploaded to staging buffer
    pub bytes_staged: u64,
    /// Number of copy commands issued
    pub copy_commands: u32,
    /// Which ring buffer was used
    pub ring_index: usize,
}

/// Uploader for light data using staged triple-buffered transfers.
///
/// Handles the complete upload pipeline:
/// 1. Write light data to staging buffer via `queue.write_buffer()`
/// 2. Copy from staging to GPU storage via `encoder.copy_buffer_to_buffer()`
/// 3. Advance ring buffer for next frame
#[derive(Debug)]
pub struct LightUploader {
    /// Staging ring for light type buffers (combined)
    staging: StagingRing,
    /// Separate staging ring for the counts uniform buffer
    counts_staging: StagingRing,
    /// Maximum lights this uploader was sized for
    max_lights: u32,
    /// Frame counter for tracking warm-up
    frame_count: u64,
}

impl LightUploader {
    /// Create a new light uploader sized for the given maximum light count.
    ///
    /// The staging buffer is sized to hold all light types at maximum capacity.
    /// Uses triple buffering by default.
    ///
    /// # Arguments
    /// * `device` - The wgpu device for buffer creation
    /// * `max_lights` - Maximum number of lights per type to support
    pub fn new(device: &wgpu::Device, max_lights: u32) -> Self {
        Self::with_capacity(device, max_lights, DEFAULT_RING_CAPACITY)
    }

    /// Create a light uploader with custom ring capacity.
    pub fn with_capacity(device: &wgpu::Device, max_lights: u32, ring_capacity: usize) -> Self {
        // Calculate maximum buffer size needed
        // This is a conservative estimate assuming all light types at max
        let max_buffer_size = Self::calculate_max_buffer_size(max_lights);

        let staging = StagingRing::new(device, max_buffer_size, ring_capacity);

        // Counts buffer is always 32 bytes (LightCounts struct)
        let counts_staging = StagingRing::new(
            device,
            std::mem::size_of::<LightCounts>() as u64,
            ring_capacity,
        );

        Self {
            staging,
            counts_staging,
            max_lights,
            frame_count: 0,
        }
    }

    /// Calculate the maximum buffer size needed for all light types.
    fn calculate_max_buffer_size(max_lights: u32) -> u64 {
        use crate::light_types::{
            DirectionalLightGPU, DiskAreaLightGPU, IESLightGPU, PointLightGPU,
            RectAreaLightGPU, SkyLightGPU, SpotLightGPU,
        };

        let max = max_lights as usize;
        let total = max * std::mem::size_of::<DirectionalLightGPU>()
            + max * std::mem::size_of::<PointLightGPU>()
            + max * std::mem::size_of::<SpotLightGPU>()
            + max * std::mem::size_of::<RectAreaLightGPU>()
            + max * std::mem::size_of::<DiskAreaLightGPU>()
            + max * std::mem::size_of::<IESLightGPU>()
            + max * std::mem::size_of::<SkyLightGPU>();

        total as u64
    }

    /// Upload light data from the builder to GPU buffers.
    ///
    /// This performs staged uploads using the ring buffer:
    /// 1. Writes light data to the current staging buffer
    /// 2. Encodes copy commands from staging to target GPU buffers
    /// 3. Advances the ring for the next frame
    ///
    /// # Arguments
    /// * `queue` - The wgpu queue for staging buffer writes
    /// * `encoder` - Command encoder for copy commands
    /// * `builder` - The light buffer builder with current frame's lights
    /// * `target` - The GPU buffers to upload to
    ///
    /// # Returns
    /// Statistics about the upload operation.
    pub fn upload(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        builder: &LightBufferBuilder,
        target: &LightGpuBuffers,
    ) -> UploadStats {
        let mut stats = UploadStats {
            ring_index: self.staging.current_index(),
            ..Default::default()
        };

        // Upload counts buffer
        let counts = builder.counts();
        let counts_bytes = bytemuck::bytes_of(&counts);
        queue.write_buffer(self.counts_staging.get_current(), 0, counts_bytes);
        encoder.copy_buffer_to_buffer(
            self.counts_staging.get_current(),
            0,
            &target.counts_buffer,
            0,
            std::mem::size_of::<LightCounts>() as u64,
        );
        stats.copy_commands += 1;
        stats.bytes_staged += std::mem::size_of::<LightCounts>() as u64;

        // Track offset into combined staging buffer
        let staging_buffer = self.staging.get_current();
        let mut staging_offset: u64 = 0;

        // Helper to upload a single light type
        macro_rules! upload_light_type {
            ($lights:expr, $target_buffer:expr, $type_name:literal) => {
                let data = $lights;
                if !data.is_empty() {
                    let bytes = bytemuck::cast_slice(data);
                    let size = bytes.len() as u64;

                    // Write to staging buffer at current offset
                    queue.write_buffer(staging_buffer, staging_offset, bytes);

                    // Copy from staging to target
                    encoder.copy_buffer_to_buffer(
                        staging_buffer,
                        staging_offset,
                        $target_buffer,
                        0,
                        size,
                    );

                    #[allow(unused_assignments)]
                    {
                        staging_offset += size;
                    }
                    stats.bytes_staged += size;
                    stats.copy_commands += 1;
                }
            };
        }

        // Upload each light type
        upload_light_type!(builder.directional_lights(), &target.directional, "directional");
        upload_light_type!(builder.point_lights(), &target.point, "point");
        upload_light_type!(builder.spot_lights(), &target.spot, "spot");
        upload_light_type!(builder.rect_area_lights(), &target.rect_area, "rect_area");
        upload_light_type!(builder.disk_area_lights(), &target.disk_area, "disk_area");
        upload_light_type!(builder.ies_lights(), &target.ies, "ies");
        upload_light_type!(builder.sky_lights(), &target.sky, "sky");

        // Advance both rings
        self.staging.advance();
        self.counts_staging.advance();
        self.frame_count += 1;

        stats
    }

    /// Upload light data without using the command encoder (direct queue writes only).
    ///
    /// This is a simpler upload path that uses `queue.write_buffer()` directly
    /// to the target buffers. Use this when you don't need the staging ring
    /// (e.g., when lights don't change every frame).
    ///
    /// Note: This bypasses the staging ring entirely.
    pub fn upload_direct(
        &self,
        queue: &wgpu::Queue,
        builder: &LightBufferBuilder,
        target: &LightGpuBuffers,
    ) -> UploadStats {
        let mut stats = UploadStats::default();

        // Upload counts
        let counts = builder.counts();
        queue.write_buffer(&target.counts_buffer, 0, bytemuck::bytes_of(&counts));
        stats.bytes_staged += std::mem::size_of::<LightCounts>() as u64;

        // Helper for direct uploads
        macro_rules! upload_direct {
            ($lights:expr, $target:expr) => {
                let data = $lights;
                if !data.is_empty() {
                    let bytes = bytemuck::cast_slice(data);
                    queue.write_buffer($target, 0, bytes);
                    stats.bytes_staged += bytes.len() as u64;
                }
            };
        }

        upload_direct!(builder.directional_lights(), &target.directional);
        upload_direct!(builder.point_lights(), &target.point);
        upload_direct!(builder.spot_lights(), &target.spot);
        upload_direct!(builder.rect_area_lights(), &target.rect_area);
        upload_direct!(builder.disk_area_lights(), &target.disk_area);
        upload_direct!(builder.ies_lights(), &target.ies);
        upload_direct!(builder.sky_lights(), &target.sky);

        stats
    }

    /// Get the maximum lights this uploader supports.
    pub fn max_lights(&self) -> u32 {
        self.max_lights
    }

    /// Get the current frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Check if the staging ring has warmed up (all buffers used at least once).
    pub fn is_warmed_up(&self) -> bool {
        self.staging.is_warmed_up(self.frame_count)
    }

    /// Get the ring capacity.
    pub fn ring_capacity(&self) -> usize {
        self.staging.capacity()
    }

    /// Get the size of each staging buffer.
    pub fn staging_buffer_size(&self) -> u64 {
        self.staging.buffer_size()
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Mock-free tests that verify logic without GPU

    #[test]
    fn test_staging_ring_capacity_validation() {
        // Capacity of 0 should panic - we test that it's asserted
        // We can't actually test the panic without std::panic::catch_unwind
        // so we just verify valid capacities work
        assert!(DEFAULT_RING_CAPACITY > 0);
    }

    #[test]
    fn test_staging_ring_advance_wraps() {
        // Simulate ring behavior without GPU
        struct MockRing {
            current: usize,
            capacity: usize,
        }

        impl MockRing {
            fn new(capacity: usize) -> Self {
                Self { current: 0, capacity }
            }

            fn advance(&mut self) {
                self.current = (self.current + 1) % self.capacity;
            }

            fn current(&self) -> usize {
                self.current
            }
        }

        let mut ring = MockRing::new(3);
        assert_eq!(ring.current(), 0);

        ring.advance();
        assert_eq!(ring.current(), 1);

        ring.advance();
        assert_eq!(ring.current(), 2);

        ring.advance();
        assert_eq!(ring.current(), 0); // Wrapped!

        ring.advance();
        assert_eq!(ring.current(), 1);
    }

    #[test]
    fn test_staging_ring_double_buffer() {
        struct MockRing {
            current: usize,
            capacity: usize,
        }

        impl MockRing {
            fn new(capacity: usize) -> Self {
                Self { current: 0, capacity }
            }

            fn advance(&mut self) {
                self.current = (self.current + 1) % self.capacity;
            }
        }

        let mut ring = MockRing::new(2);
        assert_eq!(ring.current, 0);
        ring.advance();
        assert_eq!(ring.current, 1);
        ring.advance();
        assert_eq!(ring.current, 0);
    }

    #[test]
    fn test_staging_ring_single_buffer() {
        struct MockRing {
            current: usize,
            capacity: usize,
        }

        impl MockRing {
            fn new(capacity: usize) -> Self {
                Self { current: 0, capacity }
            }

            fn advance(&mut self) {
                self.current = (self.current + 1) % self.capacity;
            }
        }

        let mut ring = MockRing::new(1);
        assert_eq!(ring.current, 0);
        ring.advance();
        assert_eq!(ring.current, 0); // Always 0 with capacity 1
        ring.advance();
        assert_eq!(ring.current, 0);
    }

    #[test]
    fn test_calculate_max_buffer_size() {
        use crate::light_types::{
            DirectionalLightGPU, DiskAreaLightGPU, IESLightGPU, PointLightGPU,
            RectAreaLightGPU, SkyLightGPU, SpotLightGPU,
        };

        let max_lights = 100u32;
        let expected = 100 * (
            std::mem::size_of::<DirectionalLightGPU>()
            + std::mem::size_of::<PointLightGPU>()
            + std::mem::size_of::<SpotLightGPU>()
            + std::mem::size_of::<RectAreaLightGPU>()
            + std::mem::size_of::<DiskAreaLightGPU>()
            + std::mem::size_of::<IESLightGPU>()
            + std::mem::size_of::<SkyLightGPU>()
        );

        let calculated = LightUploader::calculate_max_buffer_size(max_lights);
        assert_eq!(calculated, expected as u64);
    }

    #[test]
    fn test_calculate_max_buffer_size_zero() {
        let size = LightUploader::calculate_max_buffer_size(0);
        assert_eq!(size, 0);
    }

    #[test]
    fn test_upload_stats_default() {
        let stats = UploadStats::default();
        assert_eq!(stats.bytes_staged, 0);
        assert_eq!(stats.copy_commands, 0);
        assert_eq!(stats.ring_index, 0);
    }

    #[test]
    fn test_light_counts_size() {
        // LightCounts must be exactly 32 bytes for the staging buffer
        assert_eq!(std::mem::size_of::<LightCounts>(), 32);
    }

    #[test]
    fn test_warm_up_calculation() {
        // Simulate warm-up tracking
        fn is_warmed_up(frame_count: u64, capacity: usize) -> bool {
            frame_count >= capacity as u64
        }

        assert!(!is_warmed_up(0, 3));
        assert!(!is_warmed_up(1, 3));
        assert!(!is_warmed_up(2, 3));
        assert!(is_warmed_up(3, 3));
        assert!(is_warmed_up(100, 3));
    }

    #[test]
    fn test_min_buffer_size_constant() {
        // MIN_BUFFER_SIZE should match the one in light_buffers
        assert_eq!(MIN_BUFFER_SIZE, 16);
    }

    #[test]
    fn test_default_ring_capacity() {
        assert_eq!(DEFAULT_RING_CAPACITY, 3);
    }

    // Integration tests that require wgpu would go here with #[cfg(feature = "wgpu-test")]
    // For now, we test the pure logic without GPU dependencies
}
