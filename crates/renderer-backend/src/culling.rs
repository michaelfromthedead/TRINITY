//! Light culling compute pass for Forward+ clustered rendering (T-LIT-2.4).
//!
//! This module provides GPU dispatch for the froxel-based light culling shader.
//! The culling pass partitions the view frustum into 3D froxels (frustum voxels)
//! and assigns lights to each froxel for efficient per-tile light iteration.
//!
//! # Overview
//!
//! The [`LightCullingPass`] manages the compute pipeline and bind group layout
//! for the light culling shader. It dispatches one workgroup per screen tile,
//! where each workgroup:
//!
//! 1. Reduces depth bounds within the tile to find min/max Z
//! 2. Iterates through depth slices covered by the tile
//! 3. Tests each light against the froxel AABB
//! 4. Writes intersecting light indices to the froxel light list
//!
//! # Usage
//!
//! ```ignore
//! let culling_pass = LightCullingPass::new(&device);
//! let grid_config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
//!
//! // Create output buffer
//! let output_buffer = CullingOutputBuffers::new(&device, &grid_config);
//!
//! // Dispatch during frame recording
//! culling_pass.dispatch(
//!     &mut encoder,
//!     &light_buffers,
//!     &depth_texture_view,
//!     &grid_config,
//!     &output_buffer,
//! );
//! ```

use crate::light_bindings::LightBuffers;
use std::mem;

// ---------------------------------------------------------------------------
// Constants (must match light_culling.wgsl)
// ---------------------------------------------------------------------------

/// Tile size in pixels (16x16 workgroup).
pub const TILE_SIZE: u32 = 16;

/// Maximum lights that can be assigned to a single froxel.
pub const MAX_LIGHTS_PER_FROXEL: u32 = 64;

/// Default number of depth slices for exponential froxel distribution.
pub const DEFAULT_DEPTH_SLICES: u32 = 32;

// ---------------------------------------------------------------------------
// FroxelGridConfig — CPU-side configuration
// ---------------------------------------------------------------------------

/// Configuration for the 3D froxel grid used in light culling.
///
/// This struct describes the grid dimensions and depth distribution parameters.
/// It is designed to be uploaded to the GPU as a uniform buffer.
///
/// # Memory Layout
///
/// The struct is `repr(C)` with 48 bytes total, aligned to 16 bytes for
/// std140/std430 compatibility:
///
/// | Offset | Field         | Size    |
/// |--------|---------------|---------|
/// | 0      | grid_size     | 12 bytes |
/// | 12     | tile_size     | 4 bytes  |
/// | 16     | near          | 4 bytes  |
/// | 20     | far           | 4 bytes  |
/// | 24     | screen_size   | 8 bytes  |
/// | 32     | depth_scale   | 4 bytes  |
/// | 36     | depth_bias    | 4 bytes  |
/// | 40     | _padding      | 8 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FroxelGridConfig {
    /// Grid dimensions: [tiles_x, tiles_y, depth_slices].
    pub grid_size: [u32; 3],
    /// Tile size in pixels (typically 16).
    pub tile_size: u32,
    /// Near plane distance.
    pub near: f32,
    /// Far plane distance.
    pub far: f32,
    /// Screen resolution: [width, height].
    pub screen_size: [u32; 2],
    /// Logarithmic depth slice scale factor.
    pub depth_scale: f32,
    /// Logarithmic depth slice bias.
    pub depth_bias: f32,
    /// Padding to 16-byte alignment.
    pub _padding: [u32; 2],
}

impl FroxelGridConfig {
    /// Create a froxel grid configuration for the given resolution.
    ///
    /// Uses 16x16 pixel tiles and 32 depth slices by default.
    ///
    /// # Arguments
    ///
    /// * `width` - Screen width in pixels.
    /// * `height` - Screen height in pixels.
    /// * `near` - Near plane distance.
    /// * `far` - Far plane distance.
    pub fn for_resolution(width: u32, height: u32, near: f32, far: f32) -> Self {
        Self::new(width, height, near, far, TILE_SIZE, DEFAULT_DEPTH_SLICES)
    }

    /// Create a froxel grid configuration with custom tile size and depth slices.
    ///
    /// # Arguments
    ///
    /// * `width` - Screen width in pixels.
    /// * `height` - Screen height in pixels.
    /// * `near` - Near plane distance.
    /// * `far` - Far plane distance.
    /// * `tile_size` - Tile size in pixels (must be power of 2).
    /// * `depth_slices` - Number of depth slices.
    pub fn new(
        width: u32,
        height: u32,
        near: f32,
        far: f32,
        tile_size: u32,
        depth_slices: u32,
    ) -> Self {
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;

        // Compute logarithmic depth distribution parameters.
        // depth_scale and depth_bias are used to convert linear depth to slice index:
        //   slice = log2(linear_depth) * depth_scale + depth_bias
        let depth_scale = depth_slices as f32 / (far / near).ln();
        let depth_bias = -(near.ln() * depth_scale);

        Self {
            grid_size: [tiles_x, tiles_y, depth_slices],
            tile_size,
            near,
            far,
            screen_size: [width, height],
            depth_scale,
            depth_bias,
            _padding: [0; 2],
        }
    }

    /// Total number of tiles (tiles_x * tiles_y).
    #[inline]
    pub fn total_tiles(&self) -> u32 {
        self.grid_size[0] * self.grid_size[1]
    }

    /// Total number of froxels (tiles_x * tiles_y * depth_slices).
    #[inline]
    pub fn total_froxels(&self) -> u32 {
        self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
    }

    /// Compute workgroup counts for the light culling dispatch.
    ///
    /// Returns (x, y, z) workgroup counts.
    #[inline]
    pub fn workgroup_counts(&self) -> (u32, u32, u32) {
        (self.grid_size[0], self.grid_size[1], 1)
    }

    /// Calculate the required size of the froxel light index buffer in bytes.
    ///
    /// Each froxel can store up to `MAX_LIGHTS_PER_FROXEL` light indices (u32).
    #[inline]
    pub fn light_index_buffer_size(&self) -> u64 {
        (self.total_froxels() as u64) * (MAX_LIGHTS_PER_FROXEL as u64) * (mem::size_of::<u32>() as u64)
    }

    /// Calculate the required size of the froxel metadata buffer in bytes.
    ///
    /// Each froxel has a `Froxel` struct (8 bytes: offset + count).
    #[inline]
    pub fn froxel_buffer_size(&self) -> u64 {
        (self.total_froxels() as u64) * 8
    }
}

impl Default for FroxelGridConfig {
    fn default() -> Self {
        Self::for_resolution(1920, 1080, 0.1, 1000.0)
    }
}

// ---------------------------------------------------------------------------
// CullingParams — GPU-side uniform (matches shader struct)
// ---------------------------------------------------------------------------

/// GPU-side culling parameters uniform (matches WGSL `CullingParams` struct).
///
/// 32 bytes, aligned to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullingParams {
    /// Screen width in pixels.
    pub screen_width: u32,
    /// Screen height in pixels.
    pub screen_height: u32,
    /// Number of depth slices.
    pub num_depth_slices: u32,
    /// Tile size in pixels.
    pub tile_size: u32,
    /// Depth slice scale for logarithmic distribution.
    pub depth_slice_scale: f32,
    /// Depth slice bias for logarithmic distribution.
    pub depth_slice_bias: f32,
    /// Near plane distance.
    pub near_plane: f32,
    /// Far plane distance.
    pub far_plane: f32,
}

impl From<&FroxelGridConfig> for CullingParams {
    fn from(config: &FroxelGridConfig) -> Self {
        Self {
            screen_width: config.screen_size[0],
            screen_height: config.screen_size[1],
            num_depth_slices: config.grid_size[2],
            tile_size: config.tile_size,
            depth_slice_scale: config.depth_scale,
            depth_slice_bias: config.depth_bias,
            near_plane: config.near,
            far_plane: config.far,
        }
    }
}

// ---------------------------------------------------------------------------
// LightCounts — GPU-side uniform
// ---------------------------------------------------------------------------

/// GPU-side light counts uniform (matches WGSL `LightCounts` struct).
///
/// 16 bytes, aligned to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightCounts {
    /// Number of active point lights.
    pub num_point: u32,
    /// Number of active spot lights.
    pub num_spot: u32,
    /// Padding for alignment.
    pub _pad0: u32,
    pub _pad1: u32,
}

impl LightCounts {
    /// Create light counts from point and spot light counts.
    pub fn new(num_point: u32, num_spot: u32) -> Self {
        Self {
            num_point,
            num_spot,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// CullingOutputBuffers — output buffers for culling results
// ---------------------------------------------------------------------------

/// Output buffers for the light culling pass.
///
/// Contains the froxel metadata and light index list buffers that store
/// the culling results.
pub struct CullingOutputBuffers {
    /// Froxel metadata buffer (offset + count per froxel).
    pub froxel_grid: wgpu::Buffer,
    /// Light index list buffer.
    pub light_index_list: wgpu::Buffer,
    /// Atomic counter buffer for global light allocation.
    pub global_counter: wgpu::Buffer,
    /// Total number of froxels.
    pub froxel_count: u32,
}

impl CullingOutputBuffers {
    /// Create output buffers for the given grid configuration.
    pub fn new(device: &wgpu::Device, config: &FroxelGridConfig) -> Self {
        let froxel_count = config.total_froxels();

        let froxel_grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("culling_froxel_grid"),
            size: config.froxel_buffer_size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_index_list = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("culling_light_index_list"),
            size: config.light_index_buffer_size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Atomic counter buffer (single u32).
        let global_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("culling_global_counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            froxel_grid,
            light_index_list,
            global_counter,
            froxel_count,
        }
    }
}

// ---------------------------------------------------------------------------
// LightCullingPass — compute pipeline and dispatch
// ---------------------------------------------------------------------------

/// Light culling compute pass.
///
/// Manages the compute pipeline, bind group layout, and dispatch logic for
/// the froxel-based light culling shader.
pub struct LightCullingPass {
    /// Compute pipeline for light culling.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for culling resources.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl LightCullingPass {
    /// Create a new light culling pass.
    ///
    /// Loads the light culling shader and creates the compute pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline = Self::create_pipeline(device, &bind_group_layout);

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Get the bind group layout for external bind group creation.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create the bind group layout for light culling.
    ///
    /// Layout matches the WGSL shader bindings:
    ///
    /// | Binding | Type              | Content                  |
    /// |---------|-------------------|--------------------------|
    /// | 0       | uniform           | CullingParams            |
    /// | 1       | texture_depth_2d  | Depth buffer             |
    /// | 2       | uniform           | LightCounts              |
    /// | 3       | storage<read>     | Point lights             |
    /// | 4       | storage<read>     | Spot lights              |
    /// | 5       | storage<rw>       | Froxel grid              |
    /// | 6       | storage<rw>       | Light index list         |
    /// | 7       | storage<rw>       | Global counter (atomic)  |
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light_culling_bind_group_layout"),
            entries: &[
                // Binding 0: CullingParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: LightCounts uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Point lights storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Spot lights storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Froxel grid storage (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 6: Light index list storage (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 7: Global counter storage (read-write, atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create the compute pipeline for light culling.
    fn create_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        // Load shader source.
        let shader_source = include_str!("../shaders/light_culling.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("light_culling_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light_culling_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("light_culling_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Create a bind group for the culling pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `params_buffer` - Uniform buffer containing `CullingParams`.
    /// * `depth_texture` - Depth texture view.
    /// * `light_counts_buffer` - Uniform buffer containing `LightCounts`.
    /// * `point_lights_buffer` - Storage buffer containing point lights.
    /// * `spot_lights_buffer` - Storage buffer containing spot lights.
    /// * `output` - Output buffers for culling results.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
        depth_texture: &wgpu::TextureView,
        light_counts_buffer: &wgpu::Buffer,
        point_lights_buffer: &wgpu::Buffer,
        spot_lights_buffer: &wgpu::Buffer,
        output: &CullingOutputBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_culling_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: light_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: point_lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: spot_lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output.froxel_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output.light_index_list.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: output.global_counter.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch the light culling compute shader.
    ///
    /// Records commands to the encoder that execute the culling pass.
    /// Workgroup counts are derived from the grid configuration.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record to.
    /// * `bind_group` - Pre-created bind group with all resources.
    /// * `grid_config` - Froxel grid configuration for workgroup sizing.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        grid_config: &FroxelGridConfig,
    ) {
        let (wg_x, wg_y, wg_z) = grid_config.workgroup_counts();

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("light_culling_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }

    /// Convenience method to dispatch with inline bind group creation.
    ///
    /// Creates a temporary bind group and dispatches the culling pass.
    /// For better performance when dispatching multiple times per frame,
    /// pre-create the bind group using [`create_bind_group`].
    pub fn dispatch_with_resources(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params_buffer: &wgpu::Buffer,
        depth_texture: &wgpu::TextureView,
        light_counts_buffer: &wgpu::Buffer,
        light_buffers: &LightBuffers,
        output: &CullingOutputBuffers,
        grid_config: &FroxelGridConfig,
    ) {
        let bind_group = self.create_bind_group(
            device,
            params_buffer,
            depth_texture,
            light_counts_buffer,
            &light_buffers.point_lights,
            &light_buffers.spot_lights,
            output,
        );

        self.dispatch(encoder, &bind_group, grid_config);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── FroxelGridConfig struct sizes and alignment ─────────────────────────

    #[test]
    fn test_froxel_grid_config_size() {
        // 48 bytes: 3*u32 + u32 + 2*f32 + 2*u32 + 2*f32 + 2*u32 padding
        assert_eq!(mem::size_of::<FroxelGridConfig>(), 48);
    }

    #[test]
    fn test_froxel_grid_config_alignment() {
        assert_eq!(mem::align_of::<FroxelGridConfig>(), 4);
    }

    #[test]
    fn test_culling_params_size() {
        // 32 bytes: 4*u32 + 4*f32
        assert_eq!(mem::size_of::<CullingParams>(), 32);
    }

    #[test]
    fn test_light_counts_size() {
        // 16 bytes: 4*u32
        assert_eq!(mem::size_of::<LightCounts>(), 16);
    }

    // ── FroxelGridConfig::for_resolution ────────────────────────────────────

    #[test]
    fn test_grid_config_1080p() {
        let config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
        assert_eq!(config.grid_size[0], 120); // 1920 / 16
        assert_eq!(config.grid_size[1], 68);  // ceil(1080 / 16)
        assert_eq!(config.grid_size[2], DEFAULT_DEPTH_SLICES);
        assert_eq!(config.tile_size, TILE_SIZE);
        assert!((config.near - 0.1).abs() < f32::EPSILON);
        assert!((config.far - 1000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_grid_config_720p() {
        let config = FroxelGridConfig::for_resolution(1280, 720, 0.1, 500.0);
        assert_eq!(config.grid_size[0], 80);  // 1280 / 16
        assert_eq!(config.grid_size[1], 45);  // 720 / 16
        assert_eq!(config.screen_size, [1280, 720]);
    }

    #[test]
    fn test_grid_config_4k() {
        let config = FroxelGridConfig::for_resolution(3840, 2160, 0.01, 10000.0);
        assert_eq!(config.grid_size[0], 240); // 3840 / 16
        assert_eq!(config.grid_size[1], 135); // 2160 / 16
    }

    #[test]
    fn test_grid_config_non_divisible() {
        // 1000x600: not evenly divisible by 16
        let config = FroxelGridConfig::for_resolution(1000, 600, 0.1, 100.0);
        assert_eq!(config.grid_size[0], 63);  // ceil(1000 / 16) = 63
        assert_eq!(config.grid_size[1], 38);  // ceil(600 / 16) = 38
    }

    #[test]
    fn test_grid_config_custom_depth_slices() {
        let config = FroxelGridConfig::new(1920, 1080, 0.1, 1000.0, 16, 64);
        assert_eq!(config.grid_size[2], 64);
    }

    // ── FroxelGridConfig helper methods ─────────────────────────────────────

    #[test]
    fn test_total_tiles() {
        let config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
        assert_eq!(config.total_tiles(), 120 * 68);
    }

    #[test]
    fn test_total_froxels() {
        let config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
        assert_eq!(config.total_froxels(), 120 * 68 * 32);
    }

    #[test]
    fn test_workgroup_counts() {
        let config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
        let (wg_x, wg_y, wg_z) = config.workgroup_counts();
        assert_eq!(wg_x, 120);
        assert_eq!(wg_y, 68);
        assert_eq!(wg_z, 1);
    }

    #[test]
    fn test_light_index_buffer_size() {
        let config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
        let expected = config.total_froxels() as u64 * MAX_LIGHTS_PER_FROXEL as u64 * 4;
        assert_eq!(config.light_index_buffer_size(), expected);
    }

    #[test]
    fn test_froxel_buffer_size() {
        let config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
        // Each froxel has 8 bytes (offset + count)
        let expected = config.total_froxels() as u64 * 8;
        assert_eq!(config.froxel_buffer_size(), expected);
    }

    // ── Depth distribution parameters ───────────────────────────────────────

    #[test]
    fn test_depth_scale_positive() {
        let config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
        assert!(config.depth_scale > 0.0, "depth_scale must be positive");
    }

    #[test]
    fn test_depth_distribution_parameters() {
        let config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
        // Verify the scale and bias are consistent with log-space slicing
        // slice = log2(depth) * scale + bias should map [near, far] to [0, num_slices]
        let depth_scale = config.depth_scale;
        let depth_bias = config.depth_bias;

        // At near plane, slice should be ~0
        let slice_at_near = config.near.ln() * depth_scale + depth_bias;
        assert!(slice_at_near.abs() < 1.0, "slice at near should be ~0");
    }

    // ── CullingParams conversion ────────────────────────────────────────────

    #[test]
    fn test_culling_params_from_config() {
        let config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
        let params = CullingParams::from(&config);

        assert_eq!(params.screen_width, 1920);
        assert_eq!(params.screen_height, 1080);
        assert_eq!(params.num_depth_slices, 32);
        assert_eq!(params.tile_size, 16);
        assert!((params.near_plane - 0.1).abs() < f32::EPSILON);
        assert!((params.far_plane - 1000.0).abs() < f32::EPSILON);
    }

    // ── LightCounts ─────────────────────────────────────────────────────────

    #[test]
    fn test_light_counts_new() {
        let counts = LightCounts::new(100, 50);
        assert_eq!(counts.num_point, 100);
        assert_eq!(counts.num_spot, 50);
    }

    #[test]
    fn test_light_counts_default() {
        let counts = LightCounts::default();
        assert_eq!(counts.num_point, 0);
        assert_eq!(counts.num_spot, 0);
    }

    // ── FroxelGridConfig default ────────────────────────────────────────────

    #[test]
    fn test_grid_config_default() {
        let config = FroxelGridConfig::default();
        assert_eq!(config.screen_size, [1920, 1080]);
        assert_eq!(config.tile_size, TILE_SIZE);
        assert_eq!(config.grid_size[2], DEFAULT_DEPTH_SLICES);
    }

    // ── Constants match shader ──────────────────────────────────────────────

    #[test]
    fn test_tile_size_matches_shader() {
        assert_eq!(TILE_SIZE, 16);
    }

    #[test]
    fn test_max_lights_per_froxel_matches_shader() {
        assert_eq!(MAX_LIGHTS_PER_FROXEL, 64);
    }

    // ── GPU integration tests (require device) ──────────────────────────────

    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .expect("device creation"),
        )
    }

    #[test]
    fn test_culling_output_buffers_creation() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let config = FroxelGridConfig::for_resolution(1920, 1080, 0.1, 1000.0);
        let output = CullingOutputBuffers::new(&device, &config);

        assert_eq!(output.froxel_count, config.total_froxels());
    }

    #[test]
    fn test_light_culling_pass_creation() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // This will fail if the shader doesn't compile
        let _pass = LightCullingPass::new(&device);
    }

    #[test]
    fn test_bind_group_layout_creation() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let pass = LightCullingPass::new(&device);
        let _layout = pass.bind_group_layout();
    }

    // ── Bind group layout test (independent of shader) ──────────────────────

    #[test]
    fn test_bind_group_layout_creation_independent() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Test bind group layout creation directly (doesn't require shader)
        let layout = LightCullingPass::create_bind_group_layout(&device);
        let _ = layout;
    }
}
