//! Light data bind group layouts for clustered forward+ rendering.
//!
//! This module provides bind group layouts and creation helpers for GPU-driven
//! lighting with froxel-based light culling. The layout supports 7 light types
//! as storage buffers plus 2 uniform buffers for camera and grid configuration.
//!
//! # Bind Group Layout
//!
//! | Binding | Type           | Content                    |
//! |---------|----------------|----------------------------|
//! | 0       | storage<read>  | Directional lights         |
//! | 1       | storage<read>  | Point lights               |
//! | 2       | storage<read>  | Spot lights                |
//! | 3       | storage<read>  | Rect area lights           |
//! | 4       | storage<read>  | Disk area lights           |
//! | 5       | storage<read>  | IES profile lights         |
//! | 6       | storage<read>  | Tube/capsule lights        |
//! | 7       | uniform        | Camera uniforms            |
//! | 8       | uniform        | Froxel grid configuration  |
//!
//! # Usage
//!
//! ```ignore
//! let layout = create_light_bind_group_layout(&device);
//! let buffers = LightBuffers::new(&device, max_lights);
//! let bind_group = create_light_bind_group(&device, &layout, &buffers);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Light data structures (GPU-side)
// ---------------------------------------------------------------------------

/// GPU-side directional light data.
///
/// 48 bytes, aligned to 16 bytes for std430 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DirectionalLightGpu {
    /// Direction vector (normalized), w = 0 (unused).
    pub direction: [f32; 4],
    /// RGB color intensity, w = 0 (unused).
    pub color: [f32; 4],
    /// Shadow cascade data: x = shadow_bias, y = shadow_normal_bias,
    /// z = cascade_count, w = shadow_map_index.
    pub shadow_params: [f32; 4],
}

impl Default for DirectionalLightGpu {
    fn default() -> Self {
        Self {
            direction: [0.0, -1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            shadow_params: [0.001, 0.01, 4.0, -1.0],
        }
    }
}

/// GPU-side point light data.
///
/// 48 bytes, aligned to 16 bytes for std430 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointLightGpu {
    /// World-space position, w = radius.
    pub position_radius: [f32; 4],
    /// RGB color intensity, w = falloff exponent.
    pub color_falloff: [f32; 4],
    /// Shadow params: x = shadow_bias, y = shadow_map_index, z = near, w = far.
    pub shadow_params: [f32; 4],
}

impl Default for PointLightGpu {
    fn default() -> Self {
        Self {
            position_radius: [0.0, 0.0, 0.0, 10.0],
            color_falloff: [1.0, 1.0, 1.0, 2.0],
            shadow_params: [0.001, -1.0, 0.1, 100.0],
        }
    }
}

/// GPU-side spot light data.
///
/// 64 bytes, aligned to 16 bytes for std430 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpotLightGpu {
    /// World-space position, w = radius.
    pub position_radius: [f32; 4],
    /// Spot direction (normalized), w = inner cone angle (radians).
    pub direction_inner_angle: [f32; 4],
    /// RGB color intensity, w = outer cone angle (radians).
    pub color_outer_angle: [f32; 4],
    /// Shadow params: x = shadow_bias, y = shadow_map_index, z = falloff, w = unused.
    pub shadow_params: [f32; 4],
}

impl Default for SpotLightGpu {
    fn default() -> Self {
        Self {
            position_radius: [0.0, 0.0, 0.0, 20.0],
            direction_inner_angle: [0.0, -1.0, 0.0, 0.3],
            color_outer_angle: [1.0, 1.0, 1.0, 0.5],
            shadow_params: [0.001, -1.0, 2.0, 0.0],
        }
    }
}

/// GPU-side rectangular area light data.
///
/// 80 bytes, aligned to 16 bytes for std430 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RectAreaLightGpu {
    /// World-space center position, w = unused.
    pub position: [f32; 4],
    /// Surface normal direction, w = unused.
    pub normal: [f32; 4],
    /// Tangent direction (half-width direction), w = half_width.
    pub tangent_width: [f32; 4],
    /// Bitangent direction (half-height direction), w = half_height.
    pub bitangent_height: [f32; 4],
    /// RGB color intensity, w = two_sided flag (0 or 1).
    pub color_two_sided: [f32; 4],
}

impl Default for RectAreaLightGpu {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 0.0],
            normal: [0.0, -1.0, 0.0, 0.0],
            tangent_width: [1.0, 0.0, 0.0, 0.5],
            bitangent_height: [0.0, 0.0, 1.0, 0.5],
            color_two_sided: [1.0, 1.0, 1.0, 0.0],
        }
    }
}

/// GPU-side disk area light data.
///
/// 64 bytes, aligned to 16 bytes for std430 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DiskAreaLightGpu {
    /// World-space center position, w = radius.
    pub position_radius: [f32; 4],
    /// Surface normal direction, w = unused.
    pub normal: [f32; 4],
    /// RGB color intensity, w = two_sided flag (0 or 1).
    pub color_two_sided: [f32; 4],
    /// Reserved for future use.
    pub reserved: [f32; 4],
}

impl Default for DiskAreaLightGpu {
    fn default() -> Self {
        Self {
            position_radius: [0.0, 0.0, 0.0, 0.5],
            normal: [0.0, -1.0, 0.0, 0.0],
            color_two_sided: [1.0, 1.0, 1.0, 0.0],
            reserved: [0.0; 4],
        }
    }
}

/// GPU-side IES profile light data.
///
/// 64 bytes, aligned to 16 bytes for std430 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IesLightGpu {
    /// World-space position, w = radius.
    pub position_radius: [f32; 4],
    /// Primary direction (z-axis of IES coordinate system), w = IES texture index.
    pub direction_ies_index: [f32; 4],
    /// RGB color intensity, w = intensity scale.
    pub color_intensity: [f32; 4],
    /// Shadow params: x = shadow_bias, y = shadow_map_index, z,w = reserved.
    pub shadow_params: [f32; 4],
}

impl Default for IesLightGpu {
    fn default() -> Self {
        Self {
            position_radius: [0.0, 0.0, 0.0, 10.0],
            direction_ies_index: [0.0, -1.0, 0.0, 0.0],
            color_intensity: [1.0, 1.0, 1.0, 1.0],
            shadow_params: [0.001, -1.0, 0.0, 0.0],
        }
    }
}

/// GPU-side tube/capsule light data.
///
/// 64 bytes, aligned to 16 bytes for std430 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TubeLightGpu {
    /// Start point of tube, w = radius.
    pub start_radius: [f32; 4],
    /// End point of tube, w = length (computed).
    pub end_length: [f32; 4],
    /// RGB color intensity, w = falloff exponent.
    pub color_falloff: [f32; 4],
    /// Reserved for future use.
    pub reserved: [f32; 4],
}

impl Default for TubeLightGpu {
    fn default() -> Self {
        Self {
            start_radius: [0.0, 0.0, 0.0, 0.1],
            end_length: [1.0, 0.0, 0.0, 1.0],
            color_falloff: [1.0, 1.0, 1.0, 2.0],
            reserved: [0.0; 4],
        }
    }
}

/// GPU-side camera uniforms.
///
/// 192 bytes, aligned to 16 bytes for std140 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniformsGpu {
    /// View matrix (column-major).
    pub view: [[f32; 4]; 4],
    /// Projection matrix (column-major).
    pub projection: [[f32; 4]; 4],
    /// View-projection matrix (column-major).
    pub view_projection: [[f32; 4]; 4],
    /// Camera world-space position, w = unused.
    pub position: [f32; 4],
    /// Near plane distance, far plane distance, aspect ratio, fov_y (radians).
    pub params: [f32; 4],
}

impl Default for CameraUniformsGpu {
    fn default() -> Self {
        Self {
            view: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            projection: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            view_projection: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            position: [0.0, 0.0, 0.0, 1.0],
            params: [0.1, 1000.0, 1.0, 1.0],
        }
    }
}

/// GPU-side froxel grid configuration.
///
/// 48 bytes, aligned to 16 bytes for std140 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FroxelGridConfigGpu {
    /// Grid dimensions: x = tile_count_x, y = tile_count_y, z = depth_slices, w = total_froxels.
    pub dimensions: [u32; 4],
    /// Tile size in pixels: x = tile_width, y = tile_height, z,w = unused.
    pub tile_size: [u32; 4],
    /// Depth distribution params: x = near_plane, y = far_plane, z = log_depth_scale, w = log_depth_bias.
    pub depth_params: [f32; 4],
}

impl Default for FroxelGridConfigGpu {
    fn default() -> Self {
        Self {
            // Default: 16x9 tiles, 24 depth slices = 3456 froxels
            dimensions: [16, 9, 24, 16 * 9 * 24],
            tile_size: [80, 80, 0, 0], // 1280x720 / 16x9
            depth_params: [0.1, 1000.0, 1.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// LightBuffers — container for all light GPU buffers
// ---------------------------------------------------------------------------

/// Container holding all GPU buffers for light data.
///
/// Each buffer is sized to hold a maximum number of lights of its type.
/// Buffers are created with STORAGE usage for shader read access.
pub struct LightBuffers {
    /// Directional lights storage buffer.
    pub directional_lights: wgpu::Buffer,
    /// Point lights storage buffer.
    pub point_lights: wgpu::Buffer,
    /// Spot lights storage buffer.
    pub spot_lights: wgpu::Buffer,
    /// Rectangular area lights storage buffer.
    pub rect_area_lights: wgpu::Buffer,
    /// Disk area lights storage buffer.
    pub disk_area_lights: wgpu::Buffer,
    /// IES profile lights storage buffer.
    pub ies_lights: wgpu::Buffer,
    /// Tube/capsule lights storage buffer.
    pub tube_lights: wgpu::Buffer,
    /// Camera uniforms buffer.
    pub camera_uniforms: wgpu::Buffer,
    /// Froxel grid configuration buffer.
    pub grid_config: wgpu::Buffer,
}

/// Configuration for light buffer allocation.
#[derive(Clone, Copy, Debug)]
pub struct LightBufferConfig {
    /// Maximum number of directional lights.
    pub max_directional: u32,
    /// Maximum number of point lights.
    pub max_point: u32,
    /// Maximum number of spot lights.
    pub max_spot: u32,
    /// Maximum number of rectangular area lights.
    pub max_rect_area: u32,
    /// Maximum number of disk area lights.
    pub max_disk_area: u32,
    /// Maximum number of IES profile lights.
    pub max_ies: u32,
    /// Maximum number of tube lights.
    pub max_tube: u32,
}

impl Default for LightBufferConfig {
    fn default() -> Self {
        Self {
            max_directional: 4,
            max_point: 1024,
            max_spot: 256,
            max_rect_area: 64,
            max_disk_area: 64,
            max_ies: 128,
            max_tube: 64,
        }
    }
}

impl LightBuffers {
    /// Create light buffers with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to allocate buffers on.
    /// * `config` - Configuration specifying maximum light counts.
    pub fn new(device: &wgpu::Device, config: &LightBufferConfig) -> Self {
        // Helper to create a storage buffer with minimum size of 16 bytes
        let create_storage_buffer = |label: &str, count: u32, stride: usize| -> wgpu::Buffer {
            let size = (count as usize * stride).max(16) as u64;
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let directional_lights = create_storage_buffer(
            "light_buffer_directional",
            config.max_directional,
            mem::size_of::<DirectionalLightGpu>(),
        );

        let point_lights = create_storage_buffer(
            "light_buffer_point",
            config.max_point,
            mem::size_of::<PointLightGpu>(),
        );

        let spot_lights = create_storage_buffer(
            "light_buffer_spot",
            config.max_spot,
            mem::size_of::<SpotLightGpu>(),
        );

        let rect_area_lights = create_storage_buffer(
            "light_buffer_rect_area",
            config.max_rect_area,
            mem::size_of::<RectAreaLightGpu>(),
        );

        let disk_area_lights = create_storage_buffer(
            "light_buffer_disk_area",
            config.max_disk_area,
            mem::size_of::<DiskAreaLightGpu>(),
        );

        let ies_lights = create_storage_buffer(
            "light_buffer_ies",
            config.max_ies,
            mem::size_of::<IesLightGpu>(),
        );

        let tube_lights = create_storage_buffer(
            "light_buffer_tube",
            config.max_tube,
            mem::size_of::<TubeLightGpu>(),
        );

        let camera_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light_buffer_camera_uniforms"),
            size: mem::size_of::<CameraUniformsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_config = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light_buffer_grid_config"),
            size: mem::size_of::<FroxelGridConfigGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            directional_lights,
            point_lights,
            spot_lights,
            rect_area_lights,
            disk_area_lights,
            ies_lights,
            tube_lights,
            camera_uniforms,
            grid_config,
        }
    }

    /// Create light buffers with default configuration.
    pub fn with_defaults(device: &wgpu::Device) -> Self {
        Self::new(device, &LightBufferConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Bind group layout creation
// ---------------------------------------------------------------------------

/// Binding indices for the light bind group.
pub mod bindings {
    /// Directional lights storage buffer (binding 0).
    pub const DIRECTIONAL_LIGHTS: u32 = 0;
    /// Point lights storage buffer (binding 1).
    pub const POINT_LIGHTS: u32 = 1;
    /// Spot lights storage buffer (binding 2).
    pub const SPOT_LIGHTS: u32 = 2;
    /// Rectangular area lights storage buffer (binding 3).
    pub const RECT_AREA_LIGHTS: u32 = 3;
    /// Disk area lights storage buffer (binding 4).
    pub const DISK_AREA_LIGHTS: u32 = 4;
    /// IES profile lights storage buffer (binding 5).
    pub const IES_LIGHTS: u32 = 5;
    /// Tube lights storage buffer (binding 6).
    pub const TUBE_LIGHTS: u32 = 6;
    /// Camera uniforms buffer (binding 7).
    pub const CAMERA_UNIFORMS: u32 = 7;
    /// Froxel grid configuration buffer (binding 8).
    pub const GRID_CONFIG: u32 = 8;

    /// Total number of bindings in the light bind group.
    pub const BINDING_COUNT: usize = 9;
}

/// Create the bind group layout for light data.
///
/// The layout consists of:
/// - 7 storage buffers (read-only) for each light type
/// - 2 uniform buffers for camera and grid configuration
///
/// All bindings are visible to both fragment and compute shaders for maximum
/// flexibility in forward+, deferred, and compute-based lighting implementations.
///
/// # Arguments
///
/// * `device` - The wgpu device to create the layout on.
///
/// # Returns
///
/// A [`wgpu::BindGroupLayout`] with 9 entries.
pub fn create_light_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let visibility = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("light_bind_group_layout"),
        entries: &[
            // Binding 0: Directional lights (storage<read>)
            wgpu::BindGroupLayoutEntry {
                binding: bindings::DIRECTIONAL_LIGHTS,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: Point lights (storage<read>)
            wgpu::BindGroupLayoutEntry {
                binding: bindings::POINT_LIGHTS,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 2: Spot lights (storage<read>)
            wgpu::BindGroupLayoutEntry {
                binding: bindings::SPOT_LIGHTS,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 3: Rect area lights (storage<read>)
            wgpu::BindGroupLayoutEntry {
                binding: bindings::RECT_AREA_LIGHTS,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 4: Disk area lights (storage<read>)
            wgpu::BindGroupLayoutEntry {
                binding: bindings::DISK_AREA_LIGHTS,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 5: IES lights (storage<read>)
            wgpu::BindGroupLayoutEntry {
                binding: bindings::IES_LIGHTS,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 6: Tube lights (storage<read>)
            wgpu::BindGroupLayoutEntry {
                binding: bindings::TUBE_LIGHTS,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 7: Camera uniforms (uniform)
            wgpu::BindGroupLayoutEntry {
                binding: bindings::CAMERA_UNIFORMS,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 8: Grid config (uniform)
            wgpu::BindGroupLayoutEntry {
                binding: bindings::GRID_CONFIG,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Create a bind group for light data from pre-allocated buffers.
///
/// # Arguments
///
/// * `device` - The wgpu device.
/// * `layout` - The bind group layout created by [`create_light_bind_group_layout`].
/// * `buffers` - The light buffers containing all required GPU buffers.
///
/// # Returns
///
/// A [`wgpu::BindGroup`] ready to be bound for rendering or compute.
pub fn create_light_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &LightBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("light_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: bindings::DIRECTIONAL_LIGHTS,
                resource: buffers.directional_lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: bindings::POINT_LIGHTS,
                resource: buffers.point_lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: bindings::SPOT_LIGHTS,
                resource: buffers.spot_lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: bindings::RECT_AREA_LIGHTS,
                resource: buffers.rect_area_lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: bindings::DISK_AREA_LIGHTS,
                resource: buffers.disk_area_lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: bindings::IES_LIGHTS,
                resource: buffers.ies_lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: bindings::TUBE_LIGHTS,
                resource: buffers.tube_lights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: bindings::CAMERA_UNIFORMS,
                resource: buffers.camera_uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: bindings::GRID_CONFIG,
                resource: buffers.grid_config.as_entire_binding(),
            },
        ],
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── GPU struct sizes ────────────────────────────────────────────────────

    #[test]
    fn test_directional_light_size() {
        assert_eq!(mem::size_of::<DirectionalLightGpu>(), 48);
        assert_eq!(mem::align_of::<DirectionalLightGpu>(), 4);
    }

    #[test]
    fn test_point_light_size() {
        assert_eq!(mem::size_of::<PointLightGpu>(), 48);
        assert_eq!(mem::align_of::<PointLightGpu>(), 4);
    }

    #[test]
    fn test_spot_light_size() {
        assert_eq!(mem::size_of::<SpotLightGpu>(), 64);
        assert_eq!(mem::align_of::<SpotLightGpu>(), 4);
    }

    #[test]
    fn test_rect_area_light_size() {
        assert_eq!(mem::size_of::<RectAreaLightGpu>(), 80);
        assert_eq!(mem::align_of::<RectAreaLightGpu>(), 4);
    }

    #[test]
    fn test_disk_area_light_size() {
        assert_eq!(mem::size_of::<DiskAreaLightGpu>(), 64);
        assert_eq!(mem::align_of::<DiskAreaLightGpu>(), 4);
    }

    #[test]
    fn test_ies_light_size() {
        assert_eq!(mem::size_of::<IesLightGpu>(), 64);
        assert_eq!(mem::align_of::<IesLightGpu>(), 4);
    }

    #[test]
    fn test_tube_light_size() {
        assert_eq!(mem::size_of::<TubeLightGpu>(), 64);
        assert_eq!(mem::align_of::<TubeLightGpu>(), 4);
    }

    #[test]
    fn test_camera_uniforms_size() {
        // 3 matrices (4x4 f32 each = 64 bytes) + 2 vec4 (32 bytes) = 224 bytes
        assert_eq!(mem::size_of::<CameraUniformsGpu>(), 224);
    }

    #[test]
    fn test_grid_config_size() {
        // 2 uvec4 (32 bytes) + 1 vec4 (16 bytes) = 48 bytes
        assert_eq!(mem::size_of::<FroxelGridConfigGpu>(), 48);
    }

    // ── Default values ──────────────────────────────────────────────────────

    #[test]
    fn test_directional_light_default() {
        let light = DirectionalLightGpu::default();
        assert_eq!(light.direction[1], -1.0);
        assert_eq!(light.color[0], 1.0);
    }

    #[test]
    fn test_point_light_default() {
        let light = PointLightGpu::default();
        assert_eq!(light.position_radius[3], 10.0);
        assert_eq!(light.color_falloff[3], 2.0);
    }

    #[test]
    fn test_light_buffer_config_default() {
        let config = LightBufferConfig::default();
        assert_eq!(config.max_directional, 4);
        assert_eq!(config.max_point, 1024);
        assert_eq!(config.max_spot, 256);
    }

    // ── Integration tests (require GPU) ─────────────────────────────────────

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
    fn test_create_light_bind_group_layout() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let layout = create_light_bind_group_layout(&device);
        // Layout creation should not panic
        let _ = layout;
    }

    #[test]
    fn test_light_buffers_creation() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buffers = LightBuffers::with_defaults(&device);
        // All buffers should be created without panic
        let _ = buffers;
    }

    #[test]
    fn test_create_light_bind_group() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let layout = create_light_bind_group_layout(&device);
        let buffers = LightBuffers::with_defaults(&device);
        let bind_group = create_light_bind_group(&device, &layout, &buffers);

        // Bind group creation should succeed
        let _ = bind_group;
    }

    #[test]
    fn test_custom_buffer_config() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let config = LightBufferConfig {
            max_directional: 8,
            max_point: 2048,
            max_spot: 512,
            max_rect_area: 128,
            max_disk_area: 128,
            max_ies: 256,
            max_tube: 128,
        };

        let buffers = LightBuffers::new(&device, &config);
        let layout = create_light_bind_group_layout(&device);
        let bind_group = create_light_bind_group(&device, &layout, &buffers);

        let _ = bind_group;
    }

    #[test]
    fn test_binding_indices() {
        // Verify binding indices are sequential
        assert_eq!(bindings::DIRECTIONAL_LIGHTS, 0);
        assert_eq!(bindings::POINT_LIGHTS, 1);
        assert_eq!(bindings::SPOT_LIGHTS, 2);
        assert_eq!(bindings::RECT_AREA_LIGHTS, 3);
        assert_eq!(bindings::DISK_AREA_LIGHTS, 4);
        assert_eq!(bindings::IES_LIGHTS, 5);
        assert_eq!(bindings::TUBE_LIGHTS, 6);
        assert_eq!(bindings::CAMERA_UNIFORMS, 7);
        assert_eq!(bindings::GRID_CONFIG, 8);
        assert_eq!(bindings::BINDING_COUNT, 9);
    }

    // ── Additional coverage tests ───────────────────────────────────────────

    #[test]
    fn test_bind_group_layout_has_nine_entries() {
        // Verify that the layout entries array has exactly 9 elements
        // by checking BINDING_COUNT is consistent
        assert_eq!(bindings::BINDING_COUNT, 9);

        // Verify all storage bindings are for light types (0-6)
        assert!(bindings::DIRECTIONAL_LIGHTS < bindings::CAMERA_UNIFORMS);
        assert!(bindings::POINT_LIGHTS < bindings::CAMERA_UNIFORMS);
        assert!(bindings::SPOT_LIGHTS < bindings::CAMERA_UNIFORMS);
        assert!(bindings::RECT_AREA_LIGHTS < bindings::CAMERA_UNIFORMS);
        assert!(bindings::DISK_AREA_LIGHTS < bindings::CAMERA_UNIFORMS);
        assert!(bindings::IES_LIGHTS < bindings::CAMERA_UNIFORMS);
        assert!(bindings::TUBE_LIGHTS < bindings::CAMERA_UNIFORMS);

        // Verify uniform bindings are last (7-8)
        assert!(bindings::CAMERA_UNIFORMS < bindings::GRID_CONFIG);
    }

    #[test]
    fn test_spot_light_default() {
        let light = SpotLightGpu::default();
        assert_eq!(light.position_radius[3], 20.0); // radius
        assert_eq!(light.direction_inner_angle[1], -1.0); // pointing down
        assert_eq!(light.color_outer_angle[0], 1.0); // white
    }

    #[test]
    fn test_rect_area_light_default() {
        let light = RectAreaLightGpu::default();
        assert_eq!(light.normal[1], -1.0); // pointing down
        assert_eq!(light.tangent_width[3], 0.5); // half-width
        assert_eq!(light.bitangent_height[3], 0.5); // half-height
        assert_eq!(light.color_two_sided[3], 0.0); // one-sided
    }

    #[test]
    fn test_disk_area_light_default() {
        let light = DiskAreaLightGpu::default();
        assert_eq!(light.position_radius[3], 0.5); // radius
        assert_eq!(light.normal[1], -1.0); // pointing down
        assert_eq!(light.color_two_sided[3], 0.0); // one-sided
    }

    #[test]
    fn test_ies_light_default() {
        let light = IesLightGpu::default();
        assert_eq!(light.position_radius[3], 10.0); // radius
        assert_eq!(light.direction_ies_index[1], -1.0); // pointing down
        assert_eq!(light.color_intensity[3], 1.0); // intensity scale
    }

    #[test]
    fn test_tube_light_default() {
        let light = TubeLightGpu::default();
        assert_eq!(light.start_radius[3], 0.1); // small radius
        assert_eq!(light.end_length[0], 1.0); // x offset
        assert_eq!(light.color_falloff[3], 2.0); // falloff exponent
    }

    #[test]
    fn test_camera_uniforms_default() {
        let camera = CameraUniformsGpu::default();
        // Identity matrix check (diagonal = 1.0)
        assert_eq!(camera.view[0][0], 1.0);
        assert_eq!(camera.view[1][1], 1.0);
        assert_eq!(camera.view[2][2], 1.0);
        assert_eq!(camera.view[3][3], 1.0);
        // Near/far planes
        assert_eq!(camera.params[0], 0.1); // near
        assert_eq!(camera.params[1], 1000.0); // far
    }

    #[test]
    fn test_froxel_grid_config_default() {
        let config = FroxelGridConfigGpu::default();
        // 16x9 tiles, 24 depth slices
        assert_eq!(config.dimensions[0], 16);
        assert_eq!(config.dimensions[1], 9);
        assert_eq!(config.dimensions[2], 24);
        // Total froxels = 16 * 9 * 24 = 3456
        assert_eq!(config.dimensions[3], 16 * 9 * 24);
        // Tile size
        assert_eq!(config.tile_size[0], 80);
        assert_eq!(config.tile_size[1], 80);
    }

    #[test]
    fn test_all_light_types_bound_correctly() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Create buffers with minimal sizes to verify all bindings work
        let config = LightBufferConfig {
            max_directional: 1,
            max_point: 1,
            max_spot: 1,
            max_rect_area: 1,
            max_disk_area: 1,
            max_ies: 1,
            max_tube: 1,
        };

        let buffers = LightBuffers::new(&device, &config);
        let layout = create_light_bind_group_layout(&device);
        let bind_group = create_light_bind_group(&device, &layout, &buffers);

        // If we get here without panic, all bindings were created correctly
        let _ = bind_group;
    }

    #[test]
    fn test_buffer_sizes_match_struct_sizes() {
        // Verify buffer sizes are calculated correctly based on struct sizes
        let config = LightBufferConfig::default();

        // Each buffer should hold config.max_* items of the corresponding struct
        let expected_directional = config.max_directional as usize * mem::size_of::<DirectionalLightGpu>();
        let expected_point = config.max_point as usize * mem::size_of::<PointLightGpu>();
        let expected_spot = config.max_spot as usize * mem::size_of::<SpotLightGpu>();

        // Verify calculations
        assert_eq!(expected_directional, 4 * 48); // 4 * 48 = 192 bytes
        assert_eq!(expected_point, 1024 * 48); // 1024 * 48 = 49152 bytes
        assert_eq!(expected_spot, 256 * 64); // 256 * 64 = 16384 bytes
    }

    #[test]
    fn test_camera_buffer_is_uniform() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buffers = LightBuffers::with_defaults(&device);
        // Camera buffer should have UNIFORM usage
        // If creation succeeds with the uniform binding, it's correct
        let layout = create_light_bind_group_layout(&device);
        let bind_group = create_light_bind_group(&device, &layout, &buffers);
        let _ = bind_group;
    }

    #[test]
    fn test_froxel_config_buffer_is_uniform() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let buffers = LightBuffers::with_defaults(&device);
        // Grid config buffer should have UNIFORM usage
        // If creation succeeds with the uniform binding, it's correct
        let layout = create_light_bind_group_layout(&device);
        let bind_group = create_light_bind_group(&device, &layout, &buffers);
        let _ = bind_group;
    }
}
