//! Structure-of-Arrays (SoA) light buffer builder for GPU upload.
//!
//! This module provides efficient batched upload of heterogeneous light types
//! to GPU storage buffers. Each light type gets its own buffer, enabling:
//! - Efficient SIMD-friendly shader access
//! - Per-type buffer updates without full re-upload
//! - Type-specific culling and sorting
//!
//! Usage:
//! ```ignore
//! let mut builder = LightBufferBuilder::new();
//! builder.add_point(&point_light);
//! builder.add_directional(&sun_light);
//! let buffers = builder.build_gpu_buffers(&device);
//! ```

use crate::light_types::{
    DirectionalLightGPU, DiskAreaLightGPU, IESLightGPU, PointLightGPU,
    RectAreaLightGPU, SkyLightGPU, SpotLightGPU,
};
use wgpu::util::DeviceExt;

/// Minimum buffer size in bytes for empty light buffers.
/// Required because wgpu does not allow zero-sized buffers.
const MIN_BUFFER_SIZE: u64 = 16;

/// Light counts for each type, used by shaders to iterate.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCounts {
    pub directional_count: u32,
    pub point_count: u32,
    pub spot_count: u32,
    pub rect_area_count: u32,
    pub disk_area_count: u32,
    pub ies_count: u32,
    pub sky_count: u32,
    /// Padding for 16-byte alignment
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<LightCounts>() == 32);
const _: () = assert!(std::mem::size_of::<LightCounts>() % 16 == 0);

impl LightCounts {
    /// Returns total number of lights across all types.
    pub fn total(&self) -> u32 {
        self.directional_count
            + self.point_count
            + self.spot_count
            + self.rect_area_count
            + self.disk_area_count
            + self.ies_count
            + self.sky_count
    }

    /// Returns true if there are no lights.
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }
}

/// Result of building GPU light buffers.
///
/// Contains one storage buffer per light type, plus a counts buffer
/// for shader iteration. All buffers use STORAGE | COPY_DST usage.
pub struct LightGpuBuffers {
    /// Storage buffer for directional lights
    pub directional: wgpu::Buffer,
    /// Storage buffer for point lights
    pub point: wgpu::Buffer,
    /// Storage buffer for spot lights
    pub spot: wgpu::Buffer,
    /// Storage buffer for rectangular area lights
    pub rect_area: wgpu::Buffer,
    /// Storage buffer for disk area lights
    pub disk_area: wgpu::Buffer,
    /// Storage buffer for IES profile lights
    pub ies: wgpu::Buffer,
    /// Storage buffer for sky lights
    pub sky: wgpu::Buffer,
    /// Uniform buffer containing light counts
    pub counts_buffer: wgpu::Buffer,
    /// CPU-side copy of light counts
    pub counts: LightCounts,
}

impl LightGpuBuffers {
    /// Returns the total byte size of all light buffers (excluding counts).
    pub fn total_light_bytes(&self) -> u64 {
        self.directional.size()
            + self.point.size()
            + self.spot.size()
            + self.rect_area.size()
            + self.disk_area.size()
            + self.ies.size()
            + self.sky.size()
    }
}

/// Structure-of-Arrays buffer builder for efficient GPU light upload.
///
/// Collects lights by type into separate vectors, then builds
/// GPU storage buffers for each type.
#[derive(Debug, Default)]
pub struct LightBufferBuilder {
    directional: Vec<DirectionalLightGPU>,
    point: Vec<PointLightGPU>,
    spot: Vec<SpotLightGPU>,
    rect_area: Vec<RectAreaLightGPU>,
    disk_area: Vec<DiskAreaLightGPU>,
    ies: Vec<IESLightGPU>,
    sky: Vec<SkyLightGPU>,
}

impl LightBufferBuilder {
    /// Create an empty buffer builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a buffer builder with pre-allocated capacity for each type.
    pub fn with_capacity(
        directional: usize,
        point: usize,
        spot: usize,
        rect_area: usize,
        disk_area: usize,
        ies: usize,
        sky: usize,
    ) -> Self {
        Self {
            directional: Vec::with_capacity(directional),
            point: Vec::with_capacity(point),
            spot: Vec::with_capacity(spot),
            rect_area: Vec::with_capacity(rect_area),
            disk_area: Vec::with_capacity(disk_area),
            ies: Vec::with_capacity(ies),
            sky: Vec::with_capacity(sky),
        }
    }

    /// Add a directional light to the builder.
    pub fn add_directional(&mut self, light: &DirectionalLightGPU) -> &mut Self {
        self.directional.push(*light);
        self
    }

    /// Add a point light to the builder.
    pub fn add_point(&mut self, light: &PointLightGPU) -> &mut Self {
        self.point.push(*light);
        self
    }

    /// Add a spot light to the builder.
    pub fn add_spot(&mut self, light: &SpotLightGPU) -> &mut Self {
        self.spot.push(*light);
        self
    }

    /// Add a rectangular area light to the builder.
    pub fn add_rect_area(&mut self, light: &RectAreaLightGPU) -> &mut Self {
        self.rect_area.push(*light);
        self
    }

    /// Add a disk area light to the builder.
    pub fn add_disk_area(&mut self, light: &DiskAreaLightGPU) -> &mut Self {
        self.disk_area.push(*light);
        self
    }

    /// Add an IES profile light to the builder.
    pub fn add_ies(&mut self, light: &IESLightGPU) -> &mut Self {
        self.ies.push(*light);
        self
    }

    /// Add a sky light to the builder.
    pub fn add_sky(&mut self, light: &SkyLightGPU) -> &mut Self {
        self.sky.push(*light);
        self
    }

    /// Clear all light buffers, resetting to empty state.
    pub fn clear(&mut self) {
        self.directional.clear();
        self.point.clear();
        self.spot.clear();
        self.rect_area.clear();
        self.disk_area.clear();
        self.ies.clear();
        self.sky.clear();
    }

    /// Returns the current light counts without building buffers.
    pub fn counts(&self) -> LightCounts {
        LightCounts {
            directional_count: self.directional.len() as u32,
            point_count: self.point.len() as u32,
            spot_count: self.spot.len() as u32,
            rect_area_count: self.rect_area.len() as u32,
            disk_area_count: self.disk_area.len() as u32,
            ies_count: self.ies.len() as u32,
            sky_count: self.sky.len() as u32,
            _pad: 0,
        }
    }

    /// Returns true if no lights have been added.
    pub fn is_empty(&self) -> bool {
        self.directional.is_empty()
            && self.point.is_empty()
            && self.spot.is_empty()
            && self.rect_area.is_empty()
            && self.disk_area.is_empty()
            && self.ies.is_empty()
            && self.sky.is_empty()
    }

    /// Returns the total number of lights across all types.
    pub fn total_lights(&self) -> usize {
        self.directional.len()
            + self.point.len()
            + self.spot.len()
            + self.rect_area.len()
            + self.disk_area.len()
            + self.ies.len()
            + self.sky.len()
    }

    /// Calculate the exact byte size needed for all light buffers.
    ///
    /// Returns a tuple of (directional, point, spot, rect_area, disk_area, ies, sky) sizes.
    pub fn calculate_buffer_sizes(&self) -> BufferSizes {
        BufferSizes {
            directional: Self::buffer_size_for::<DirectionalLightGPU>(self.directional.len()),
            point: Self::buffer_size_for::<PointLightGPU>(self.point.len()),
            spot: Self::buffer_size_for::<SpotLightGPU>(self.spot.len()),
            rect_area: Self::buffer_size_for::<RectAreaLightGPU>(self.rect_area.len()),
            disk_area: Self::buffer_size_for::<DiskAreaLightGPU>(self.disk_area.len()),
            ies: Self::buffer_size_for::<IESLightGPU>(self.ies.len()),
            sky: Self::buffer_size_for::<SkyLightGPU>(self.sky.len()),
        }
    }

    /// Calculate buffer size for a given light type and count.
    /// Returns MIN_BUFFER_SIZE for empty buffers (wgpu requires non-zero size).
    fn buffer_size_for<T>(count: usize) -> u64 {
        let bytes = count * std::mem::size_of::<T>();
        if bytes == 0 {
            MIN_BUFFER_SIZE
        } else {
            bytes as u64
        }
    }

    /// Build GPU storage buffers for all light types.
    ///
    /// Creates wgpu::Buffer for each light type with STORAGE | COPY_DST usage.
    /// Empty light types get a minimum-sized placeholder buffer.
    pub fn build_gpu_buffers(&self, device: &wgpu::Device) -> LightGpuBuffers {
        let counts = self.counts();

        let directional = self.create_storage_buffer(
            device,
            "Directional Lights",
            &self.directional,
        );

        let point = self.create_storage_buffer(
            device,
            "Point Lights",
            &self.point,
        );

        let spot = self.create_storage_buffer(
            device,
            "Spot Lights",
            &self.spot,
        );

        let rect_area = self.create_storage_buffer(
            device,
            "RectArea Lights",
            &self.rect_area,
        );

        let disk_area = self.create_storage_buffer(
            device,
            "DiskArea Lights",
            &self.disk_area,
        );

        let ies = self.create_storage_buffer(
            device,
            "IES Lights",
            &self.ies,
        );

        let sky = self.create_storage_buffer(
            device,
            "Sky Lights",
            &self.sky,
        );

        let counts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Counts"),
            contents: bytemuck::bytes_of(&counts),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        LightGpuBuffers {
            directional,
            point,
            spot,
            rect_area,
            disk_area,
            ies,
            sky,
            counts_buffer,
            counts,
        }
    }

    /// Create a storage buffer from a slice of Pod types.
    /// Creates a minimum-sized buffer if the slice is empty.
    fn create_storage_buffer<T: bytemuck::Pod>(
        &self,
        device: &wgpu::Device,
        label: &str,
        data: &[T],
    ) -> wgpu::Buffer {
        if data.is_empty() {
            // Create minimum-sized placeholder buffer
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: MIN_BUFFER_SIZE,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        }
    }

    /// Get immutable reference to directional lights.
    pub fn directional_lights(&self) -> &[DirectionalLightGPU] {
        &self.directional
    }

    /// Get immutable reference to point lights.
    pub fn point_lights(&self) -> &[PointLightGPU] {
        &self.point
    }

    /// Get immutable reference to spot lights.
    pub fn spot_lights(&self) -> &[SpotLightGPU] {
        &self.spot
    }

    /// Get immutable reference to rectangular area lights.
    pub fn rect_area_lights(&self) -> &[RectAreaLightGPU] {
        &self.rect_area
    }

    /// Get immutable reference to disk area lights.
    pub fn disk_area_lights(&self) -> &[DiskAreaLightGPU] {
        &self.disk_area
    }

    /// Get immutable reference to IES lights.
    pub fn ies_lights(&self) -> &[IESLightGPU] {
        &self.ies
    }

    /// Get immutable reference to sky lights.
    pub fn sky_lights(&self) -> &[SkyLightGPU] {
        &self.sky
    }
}

/// Buffer sizes for each light type in bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct BufferSizes {
    pub directional: u64,
    pub point: u64,
    pub spot: u64,
    pub rect_area: u64,
    pub disk_area: u64,
    pub ies: u64,
    pub sky: u64,
}

impl BufferSizes {
    /// Returns total size of all light buffers in bytes.
    pub fn total(&self) -> u64 {
        self.directional + self.point + self.spot + self.rect_area
            + self.disk_area + self.ies + self.sky
    }
}

// =============================================================================
// Implement bytemuck traits for light types (required for buffer creation)
// =============================================================================

// SAFETY: All light GPU types are #[repr(C)] with only f32/u32 fields
unsafe impl bytemuck::Pod for DirectionalLightGPU {}
unsafe impl bytemuck::Zeroable for DirectionalLightGPU {}

unsafe impl bytemuck::Pod for PointLightGPU {}
unsafe impl bytemuck::Zeroable for PointLightGPU {}

unsafe impl bytemuck::Pod for SpotLightGPU {}
unsafe impl bytemuck::Zeroable for SpotLightGPU {}

unsafe impl bytemuck::Pod for RectAreaLightGPU {}
unsafe impl bytemuck::Zeroable for RectAreaLightGPU {}

unsafe impl bytemuck::Pod for DiskAreaLightGPU {}
unsafe impl bytemuck::Zeroable for DiskAreaLightGPU {}

unsafe impl bytemuck::Pod for IESLightGPU {}
unsafe impl bytemuck::Zeroable for IESLightGPU {}

unsafe impl bytemuck::Pod for SkyLightGPU {}
unsafe impl bytemuck::Zeroable for SkyLightGPU {}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_light_counts_default() {
        let counts = LightCounts::default();
        assert_eq!(counts.total(), 0);
        assert!(counts.is_empty());
    }

    #[test]
    fn test_light_counts_total() {
        let counts = LightCounts {
            directional_count: 1,
            point_count: 5,
            spot_count: 3,
            rect_area_count: 2,
            disk_area_count: 1,
            ies_count: 4,
            sky_count: 1,
            _pad: 0,
        };
        assert_eq!(counts.total(), 17);
        assert!(!counts.is_empty());
    }

    #[test]
    fn test_light_counts_size_alignment() {
        // LightCounts must be 16-byte aligned for GPU uniform buffers
        assert_eq!(std::mem::size_of::<LightCounts>(), 32);
        assert_eq!(std::mem::size_of::<LightCounts>() % 16, 0);
    }

    #[test]
    fn test_builder_new_is_empty() {
        let builder = LightBufferBuilder::new();
        assert!(builder.is_empty());
        assert_eq!(builder.total_lights(), 0);
        assert!(builder.counts().is_empty());
    }

    #[test]
    fn test_builder_add_directional() {
        let mut builder = LightBufferBuilder::new();
        let light = DirectionalLightGPU::new_sun(
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
            100000.0,
        );
        builder.add_directional(&light);

        assert!(!builder.is_empty());
        assert_eq!(builder.total_lights(), 1);
        assert_eq!(builder.counts().directional_count, 1);
        assert_eq!(builder.directional_lights().len(), 1);
    }

    #[test]
    fn test_builder_add_point() {
        let mut builder = LightBufferBuilder::new();
        let light = PointLightGPU::new(
            [0.0, 5.0, 0.0],
            [1.0, 0.8, 0.6],
            800.0,
            10.0,
        );
        builder.add_point(&light);

        assert_eq!(builder.counts().point_count, 1);
        assert_eq!(builder.point_lights().len(), 1);
    }

    #[test]
    fn test_builder_add_spot() {
        let mut builder = LightBufferBuilder::new();
        let light = SpotLightGPU::new(
            [0.0, 5.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
            1000.0,
            0.4363,
            0.7854,
            20.0,
        );
        builder.add_spot(&light);

        assert_eq!(builder.counts().spot_count, 1);
        assert_eq!(builder.spot_lights().len(), 1);
    }

    #[test]
    fn test_builder_add_rect_area() {
        let mut builder = LightBufferBuilder::new();
        let light = RectAreaLightGPU::new(
            [0.0, 3.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            2.0,
            1.0,
            [1.0, 1.0, 1.0],
            500.0,
        );
        builder.add_rect_area(&light);

        assert_eq!(builder.counts().rect_area_count, 1);
        assert_eq!(builder.rect_area_lights().len(), 1);
    }

    #[test]
    fn test_builder_add_disk_area() {
        let mut builder = LightBufferBuilder::new();
        let light = DiskAreaLightGPU::new(
            [0.0, 2.0, 0.0],
            [0.0, -1.0, 0.0],
            0.5,
            [1.0, 0.8, 0.6],
            1000.0,
        );
        builder.add_disk_area(&light);

        assert_eq!(builder.counts().disk_area_count, 1);
        assert_eq!(builder.disk_area_lights().len(), 1);
    }

    #[test]
    fn test_builder_add_ies() {
        let mut builder = LightBufferBuilder::new();
        let light = IESLightGPU::new(
            [0.0, 3.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
            1000.0,
            15.0,
            0,
        );
        builder.add_ies(&light);

        assert_eq!(builder.counts().ies_count, 1);
        assert_eq!(builder.ies_lights().len(), 1);
    }

    #[test]
    fn test_builder_add_sky() {
        let mut builder = LightBufferBuilder::new();
        let light = SkyLightGPU::new([0.5, 0.7, 1.0], 1.0, 0);
        builder.add_sky(&light);

        assert_eq!(builder.counts().sky_count, 1);
        assert_eq!(builder.sky_lights().len(), 1);
    }

    #[test]
    fn test_builder_clear() {
        let mut builder = LightBufferBuilder::new();

        // Add one of each type
        builder.add_directional(&DirectionalLightGPU::default());
        builder.add_point(&PointLightGPU::default());
        builder.add_spot(&SpotLightGPU::default());
        builder.add_rect_area(&RectAreaLightGPU::default());
        builder.add_disk_area(&DiskAreaLightGPU::default());
        builder.add_ies(&IESLightGPU::default());
        builder.add_sky(&SkyLightGPU::default());

        assert_eq!(builder.total_lights(), 7);

        builder.clear();

        assert!(builder.is_empty());
        assert_eq!(builder.total_lights(), 0);
        assert!(builder.counts().is_empty());
    }

    #[test]
    fn test_builder_with_capacity() {
        let builder = LightBufferBuilder::with_capacity(2, 10, 5, 3, 2, 4, 1);

        // Capacity should be set, but builder still empty
        assert!(builder.is_empty());
        assert_eq!(builder.directional.capacity(), 2);
        assert_eq!(builder.point.capacity(), 10);
        assert_eq!(builder.spot.capacity(), 5);
        assert_eq!(builder.rect_area.capacity(), 3);
        assert_eq!(builder.disk_area.capacity(), 2);
        assert_eq!(builder.ies.capacity(), 4);
        assert_eq!(builder.sky.capacity(), 1);
    }

    #[test]
    fn test_builder_chaining() {
        let mut builder = LightBufferBuilder::new();

        builder
            .add_point(&PointLightGPU::default())
            .add_point(&PointLightGPU::default())
            .add_spot(&SpotLightGPU::default());

        assert_eq!(builder.total_lights(), 3);
        assert_eq!(builder.counts().point_count, 2);
        assert_eq!(builder.counts().spot_count, 1);
    }

    #[test]
    fn test_calculate_buffer_sizes_empty() {
        let builder = LightBufferBuilder::new();
        let sizes = builder.calculate_buffer_sizes();

        // Empty buffers should use MIN_BUFFER_SIZE
        assert_eq!(sizes.directional, MIN_BUFFER_SIZE);
        assert_eq!(sizes.point, MIN_BUFFER_SIZE);
        assert_eq!(sizes.spot, MIN_BUFFER_SIZE);
        assert_eq!(sizes.rect_area, MIN_BUFFER_SIZE);
        assert_eq!(sizes.disk_area, MIN_BUFFER_SIZE);
        assert_eq!(sizes.ies, MIN_BUFFER_SIZE);
        assert_eq!(sizes.sky, MIN_BUFFER_SIZE);
        assert_eq!(sizes.total(), MIN_BUFFER_SIZE * 7);
    }

    #[test]
    fn test_calculate_buffer_sizes_with_lights() {
        let mut builder = LightBufferBuilder::new();
        builder.add_directional(&DirectionalLightGPU::default());
        builder.add_point(&PointLightGPU::default());
        builder.add_point(&PointLightGPU::default());

        let sizes = builder.calculate_buffer_sizes();

        // DirectionalLightGPU = 64 bytes
        assert_eq!(sizes.directional, 64);
        // 2 * PointLightGPU = 2 * 48 = 96 bytes
        assert_eq!(sizes.point, 96);
        // Empty types use MIN_BUFFER_SIZE
        assert_eq!(sizes.spot, MIN_BUFFER_SIZE);
    }

    #[test]
    fn test_buffer_sizes_total() {
        let sizes = BufferSizes {
            directional: 64,
            point: 96,
            spot: 128,
            rect_area: 80,
            disk_area: 48,
            ies: 64,
            sky: 48,
        };
        assert_eq!(sizes.total(), 528);
    }

    #[test]
    fn test_bytemuck_traits() {
        // Verify that bytemuck can cast light types to bytes
        let light = PointLightGPU::new([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 1000.0, 10.0);
        let bytes: &[u8] = bytemuck::bytes_of(&light);
        assert_eq!(bytes.len(), 48);

        // Verify we can create zeroed instances
        let zeroed: PointLightGPU = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.position, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_all_light_types_bytemuck_sizes() {
        // Verify bytemuck sizes match expected struct sizes
        assert_eq!(
            std::mem::size_of::<DirectionalLightGPU>(),
            bytemuck::bytes_of(&DirectionalLightGPU::default()).len()
        );
        assert_eq!(
            std::mem::size_of::<PointLightGPU>(),
            bytemuck::bytes_of(&PointLightGPU::default()).len()
        );
        assert_eq!(
            std::mem::size_of::<SpotLightGPU>(),
            bytemuck::bytes_of(&SpotLightGPU::default()).len()
        );
        assert_eq!(
            std::mem::size_of::<RectAreaLightGPU>(),
            bytemuck::bytes_of(&RectAreaLightGPU::default()).len()
        );
        assert_eq!(
            std::mem::size_of::<DiskAreaLightGPU>(),
            bytemuck::bytes_of(&DiskAreaLightGPU::default()).len()
        );
        assert_eq!(
            std::mem::size_of::<IESLightGPU>(),
            bytemuck::bytes_of(&IESLightGPU::default()).len()
        );
        assert_eq!(
            std::mem::size_of::<SkyLightGPU>(),
            bytemuck::bytes_of(&SkyLightGPU::default()).len()
        );
    }
}
