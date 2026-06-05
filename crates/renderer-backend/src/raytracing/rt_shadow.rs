//! RT Shadow Ray Compute Pipeline (T-RT-P1.5).
//!
//! This module provides a compute pipeline for hardware-accelerated shadow rays
//! using inline ray queries. It traces rays from visible surfaces toward lights
//! to compute per-pixel shadow factors.
//!
//! # Overview
//!
//! Shadow rays are traced from each visible pixel toward each light source.
//! The TLAS (Top-Level Acceleration Structure) is queried to determine if
//! any geometry blocks the path, resulting in a shadow.
//!
//! # Performance
//!
//! Ray tracing shadows provides:
//! - Pixel-perfect shadow accuracy
//! - No shadow map resolution artifacts
//! - Natural support for all light types
//! - Efficient early termination on first hit
//!
//! Trade-offs:
//! - Higher GPU cost than shadow mapping
//! - Requires RT-capable hardware
//! - Memory for acceleration structures
//!
//! # Usage
//!
//! ```ignore
//! let pipeline = RTShadowPipeline::new(&device);
//! let params = ShadowRayParams::new(camera, width, height, 4);
//!
//! let bind_group = pipeline.create_bind_group(
//!     &device,
//!     &tlas_buffer,
//!     &depth_view,
//!     &normal_view,
//!     &lights_buffer,
//!     &shadow_output_buffer,
//!     &params_buffer,
//! );
//!
//! pipeline.dispatch(&mut encoder, &bind_group, width, height);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (8x8 threads).
pub const WORKGROUP_SIZE: u32 = 8;

/// Default normal bias to prevent self-intersection.
pub const DEFAULT_BIAS: f32 = 0.001;

/// Maximum supported lights per dispatch.
pub const MAX_LIGHTS: u32 = 256;

// ---------------------------------------------------------------------------
// Light Type
// ---------------------------------------------------------------------------

/// Light type enumeration for shadow ray tracing.
///
/// Matches the GPU shader constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum LightType {
    /// Directional light (sun/moon). Rays trace toward infinity.
    Directional = 0,
    /// Point light. Rays trace toward light position.
    Point = 1,
    /// Spot light. Rays trace toward light position.
    Spot = 2,
}

impl LightType {
    /// Convert from u32 value.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Directional),
            1 => Some(Self::Point),
            2 => Some(Self::Spot),
            _ => None,
        }
    }
}

impl Default for LightType {
    fn default() -> Self {
        Self::Point
    }
}

// ---------------------------------------------------------------------------
// Light
// ---------------------------------------------------------------------------

/// GPU-compatible light structure for shadow ray tracing.
///
/// This is a compact representation optimized for GPU access, packing
/// position, type, direction, range, color, and intensity into 48 bytes.
///
/// # Memory Layout
///
/// | Offset | Field            | Size   | Description                    |
/// |--------|------------------|--------|--------------------------------|
/// | 0      | position_type    | 16     | xyz=position/dir, w=type       |
/// | 16     | direction_range  | 16     | xyz=direction, w=range         |
/// | 32     | color_intensity  | 16     | xyz=color, w=intensity         |
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Light {
    /// xyz = position (point/spot) or direction (directional)
    /// w = light type (0=directional, 1=point, 2=spot)
    pub position_type: [f32; 4],

    /// xyz = direction (spot only), w = range (point/spot)
    pub direction_range: [f32; 4],

    /// xyz = color RGB, w = intensity
    pub color_intensity: [f32; 4],
}

// Compile-time size assertion: 48 bytes
const _: () = assert!(mem::size_of::<Light>() == 48);
const _: () = assert!(mem::align_of::<Light>() == 4);

impl Light {
    /// Create a directional light.
    ///
    /// # Arguments
    ///
    /// * `direction` - Light direction (normalized, pointing from light)
    /// * `color` - RGB color [0, 1]
    /// * `intensity` - Light intensity multiplier
    pub fn directional(direction: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self {
            position_type: [direction[0], direction[1], direction[2], 0.0],
            direction_range: [0.0, 0.0, 0.0, 10000.0],
            color_intensity: [color[0], color[1], color[2], intensity],
        }
    }

    /// Create a point light.
    ///
    /// # Arguments
    ///
    /// * `position` - World position of the light
    /// * `range` - Maximum influence range
    /// * `color` - RGB color [0, 1]
    /// * `intensity` - Light intensity multiplier
    pub fn point(position: [f32; 3], range: f32, color: [f32; 3], intensity: f32) -> Self {
        Self {
            position_type: [position[0], position[1], position[2], 1.0],
            direction_range: [0.0, 0.0, 0.0, range],
            color_intensity: [color[0], color[1], color[2], intensity],
        }
    }

    /// Create a spot light.
    ///
    /// # Arguments
    ///
    /// * `position` - World position of the light
    /// * `direction` - Light direction (normalized)
    /// * `range` - Maximum influence range
    /// * `color` - RGB color [0, 1]
    /// * `intensity` - Light intensity multiplier
    pub fn spot(
        position: [f32; 3],
        direction: [f32; 3],
        range: f32,
        color: [f32; 3],
        intensity: f32,
    ) -> Self {
        Self {
            position_type: [position[0], position[1], position[2], 2.0],
            direction_range: [direction[0], direction[1], direction[2], range],
            color_intensity: [color[0], color[1], color[2], intensity],
        }
    }

    /// Get the light type.
    pub fn light_type(&self) -> LightType {
        LightType::from_u32(self.position_type[3] as u32).unwrap_or_default()
    }

    /// Get the light position (for point/spot lights).
    pub fn position(&self) -> [f32; 3] {
        [self.position_type[0], self.position_type[1], self.position_type[2]]
    }

    /// Get the light direction (for directional/spot lights).
    pub fn direction(&self) -> [f32; 3] {
        match self.light_type() {
            LightType::Directional => {
                [self.position_type[0], self.position_type[1], self.position_type[2]]
            }
            LightType::Spot => {
                [self.direction_range[0], self.direction_range[1], self.direction_range[2]]
            }
            LightType::Point => [0.0, -1.0, 0.0], // Default down
        }
    }

    /// Get the light range.
    pub fn range(&self) -> f32 {
        self.direction_range[3]
    }

    /// Get the light color.
    pub fn color(&self) -> [f32; 3] {
        [self.color_intensity[0], self.color_intensity[1], self.color_intensity[2]]
    }

    /// Get the light intensity.
    pub fn intensity(&self) -> f32 {
        self.color_intensity[3]
    }
}

impl Default for Light {
    fn default() -> Self {
        Self::point([0.0, 0.0, 0.0], 10.0, [1.0, 1.0, 1.0], 1.0)
    }
}

// ---------------------------------------------------------------------------
// ShadowRayParams
// ---------------------------------------------------------------------------

/// GPU-side shadow ray parameters.
///
/// This struct is uploaded to a uniform buffer for the compute shader.
/// Contains the inverse view-projection matrix and dispatch parameters.
///
/// # Memory Layout
///
/// 80 bytes total, std140 compatible:
///
/// | Offset | Field            | Size   | Description                    |
/// |--------|------------------|--------|--------------------------------|
/// | 0      | inverse_view_proj| 64     | Inverse VP matrix (mat4x4)     |
/// | 64     | light_count      | 4      | Number of lights               |
/// | 68     | width            | 4      | Output width in pixels         |
/// | 72     | height           | 4      | Output height in pixels        |
/// | 76     | bias             | 4      | Normal bias for self-shadow    |
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowRayParams {
    /// Inverse view-projection matrix for world position reconstruction.
    pub inverse_view_proj: [[f32; 4]; 4],

    /// Number of lights to process.
    pub light_count: u32,

    /// Output texture width in pixels.
    pub width: u32,

    /// Output texture height in pixels.
    pub height: u32,

    /// Normal bias offset to prevent self-intersection.
    pub bias: f32,
}

// Compile-time size assertion: 80 bytes
const _: () = assert!(mem::size_of::<ShadowRayParams>() == 80);
const _: () = assert!(mem::align_of::<ShadowRayParams>() == 4);

impl ShadowRayParams {
    /// Create new shadow ray parameters.
    ///
    /// # Arguments
    ///
    /// * `inverse_view_proj` - Inverse view-projection matrix
    /// * `width` - Output width in pixels
    /// * `height` - Output height in pixels
    /// * `light_count` - Number of lights
    pub fn new(
        inverse_view_proj: [[f32; 4]; 4],
        width: u32,
        height: u32,
        light_count: u32,
    ) -> Self {
        Self {
            inverse_view_proj,
            light_count,
            width,
            height,
            bias: DEFAULT_BIAS,
        }
    }

    /// Create parameters with custom bias.
    pub fn with_bias(mut self, bias: f32) -> Self {
        self.bias = bias;
        self
    }

    /// Create an identity matrix for testing.
    pub fn identity_matrix() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Compute total output buffer size in bytes.
    ///
    /// The output buffer stores one f32 per pixel per light.
    pub fn output_buffer_size(&self) -> u64 {
        (self.width as u64) * (self.height as u64) * (self.light_count as u64) * 4
    }
}

impl Default for ShadowRayParams {
    fn default() -> Self {
        Self {
            inverse_view_proj: Self::identity_matrix(),
            light_count: 0,
            width: 0,
            height: 0,
            bias: DEFAULT_BIAS,
        }
    }
}

// ---------------------------------------------------------------------------
// RTShadowPipeline
// ---------------------------------------------------------------------------

/// RT shadow ray compute pipeline.
///
/// Manages the compute pipeline and bind group layout for hardware-accelerated
/// shadow ray tracing using inline ray queries.
///
/// # Bind Group Layout
///
/// | Binding | Type           | Content                          |
/// |---------|----------------|----------------------------------|
/// | 0       | storage (read) | TLAS placeholder / AS            |
/// | 1       | texture_depth  | Depth buffer                     |
/// | 2       | texture_2d     | Normal buffer (world-space)      |
/// | 3       | storage (read) | Light array                      |
/// | 4       | storage (rw)   | Shadow output array              |
/// | 5       | uniform        | ShadowRayParams                  |
pub struct RTShadowPipeline {
    /// Compute pipeline for shadow ray tracing.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for shadow ray resources.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl RTShadowPipeline {
    /// Create a new RT shadow pipeline.
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
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create the bind group layout for RT shadow resources.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt_shadow_bind_group_layout"),
            entries: &[
                // Binding 0: TLAS placeholder (storage buffer)
                // Note: Real RT would use acceleration_structure type
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(4),
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
                // Binding 2: Normal texture (world-space)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 3: Lights array (storage buffer)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            mem::size_of::<Light>() as u64,
                        ),
                    },
                    count: None,
                },
                // Binding 4: Shadow output (storage buffer, read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(4),
                    },
                    count: None,
                },
                // Binding 5: ShadowRayParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            mem::size_of::<ShadowRayParams>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create the compute pipeline for RT shadows.
    fn create_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader_source = include_str!("../../shaders/raytracing/rt_shadow.comp.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rt_shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt_shadow_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rt_shadow_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Create a bind group for the RT shadow pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `tlas_buffer` - TLAS placeholder buffer (minimum 4 bytes).
    /// * `depth_view` - Depth buffer texture view.
    /// * `normal_view` - World-space normal buffer texture view.
    /// * `lights_buffer` - Buffer containing Light array.
    /// * `shadow_output_buffer` - Output buffer for shadow factors.
    /// * `params_buffer` - Uniform buffer containing ShadowRayParams.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        tlas_buffer: &wgpu::Buffer,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        lights_buffer: &wgpu::Buffer,
        shadow_output_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt_shadow_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tlas_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: shadow_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch the RT shadow compute shader.
    ///
    /// Records compute commands to the encoder. Workgroup counts are
    /// calculated from the output resolution.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record to.
    /// * `bind_group` - Pre-created bind group with all resources.
    /// * `width` - Output width in pixels.
    /// * `height` - Output height in pixels.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let wg_x = (width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let wg_y = (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rt_shadow_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// Calculate workgroup counts for a given resolution.
    ///
    /// # Arguments
    ///
    /// * `width` - Output width in pixels.
    /// * `height` - Output height in pixels.
    ///
    /// # Returns
    ///
    /// Tuple of (x, y, z) workgroup counts.
    #[inline]
    pub fn workgroup_counts(width: u32, height: u32) -> (u32, u32, u32) {
        let wg_x = (width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let wg_y = (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        (wg_x, wg_y, 1)
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Create a params uniform buffer.
///
/// # Arguments
///
/// * `device` - The wgpu device.
/// * `params` - Shadow ray parameters.
///
/// # Returns
///
/// A GPU buffer containing the parameters.
pub fn create_params_buffer(device: &wgpu::Device, params: &ShadowRayParams) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt_shadow_params"),
        size: mem::size_of::<ShadowRayParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::bytes_of(params));
    buffer.unmap();

    buffer
}

/// Create a lights storage buffer.
///
/// # Arguments
///
/// * `device` - The wgpu device.
/// * `lights` - Array of lights.
///
/// # Returns
///
/// A GPU buffer containing the lights.
pub fn create_lights_buffer(device: &wgpu::Device, lights: &[Light]) -> wgpu::Buffer {
    let size = (lights.len() * mem::size_of::<Light>()).max(mem::size_of::<Light>()) as u64;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt_shadow_lights"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    if !lights.is_empty() {
        buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(lights));
    }
    buffer.unmap();

    buffer
}

/// Create a shadow output storage buffer.
///
/// # Arguments
///
/// * `device` - The wgpu device.
/// * `width` - Output width in pixels.
/// * `height` - Output height in pixels.
/// * `light_count` - Number of lights.
///
/// # Returns
///
/// A GPU buffer for shadow output.
pub fn create_shadow_output_buffer(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    light_count: u32,
) -> wgpu::Buffer {
    let size = (width as u64) * (height as u64) * (light_count as u64) * 4;
    let size = size.max(4); // Minimum 4 bytes

    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt_shadow_output"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Create a TLAS placeholder buffer for validation.
///
/// # Arguments
///
/// * `device` - The wgpu device.
///
/// # Returns
///
/// A minimal GPU buffer satisfying the TLAS binding.
pub fn create_tlas_placeholder(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt_shadow_tlas_placeholder"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

// ---------------------------------------------------------------------------
// Shader Validation (naga)
// ---------------------------------------------------------------------------

/// Validate the RT shadow shader using naga.
///
/// This function parses and validates the WGSL shader source.
///
/// # Returns
///
/// Ok(()) if validation passes, Err with message if it fails.
#[cfg(test)]
pub fn validate_shader() -> Result<(), String> {
    use naga::front::wgsl;

    let shader_source = include_str!("../../shaders/raytracing/rt_shadow.comp.wgsl");

    let module = wgsl::parse_str(shader_source)
        .map_err(|e| format!("WGSL parse error: {:?}", e))?;

    // Validate the module
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    validator
        .validate(&module)
        .map_err(|e| format!("WGSL validation error: {:?}", e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Struct Size and Alignment Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_light_size_is_48_bytes() {
        assert_eq!(mem::size_of::<Light>(), 48);
    }

    #[test]
    fn test_light_alignment_is_4_bytes() {
        assert_eq!(mem::align_of::<Light>(), 4);
    }

    #[test]
    fn test_shadow_ray_params_size_is_80_bytes() {
        assert_eq!(mem::size_of::<ShadowRayParams>(), 80);
    }

    #[test]
    fn test_shadow_ray_params_alignment_is_4_bytes() {
        assert_eq!(mem::align_of::<ShadowRayParams>(), 4);
    }

    // -------------------------------------------------------------------------
    // Light Creation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_directional_light_creation() {
        let light = Light::directional([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 1.0);

        assert_eq!(light.light_type(), LightType::Directional);
        assert_eq!(light.direction(), [0.0, -1.0, 0.0]);
        assert_eq!(light.color(), [1.0, 1.0, 1.0]);
        assert_eq!(light.intensity(), 1.0);
    }

    #[test]
    fn test_point_light_creation() {
        let light = Light::point([1.0, 2.0, 3.0], 10.0, [1.0, 0.5, 0.0], 100.0);

        assert_eq!(light.light_type(), LightType::Point);
        assert_eq!(light.position(), [1.0, 2.0, 3.0]);
        assert_eq!(light.range(), 10.0);
        assert_eq!(light.color(), [1.0, 0.5, 0.0]);
        assert_eq!(light.intensity(), 100.0);
    }

    #[test]
    fn test_spot_light_creation() {
        let light = Light::spot(
            [5.0, 10.0, 5.0],
            [0.0, -1.0, 0.0],
            20.0,
            [1.0, 1.0, 0.8],
            500.0,
        );

        assert_eq!(light.light_type(), LightType::Spot);
        assert_eq!(light.position(), [5.0, 10.0, 5.0]);
        assert_eq!(light.direction(), [0.0, -1.0, 0.0]);
        assert_eq!(light.range(), 20.0);
        assert_eq!(light.color(), [1.0, 1.0, 0.8]);
        assert_eq!(light.intensity(), 500.0);
    }

    #[test]
    fn test_light_default() {
        let light = Light::default();
        assert_eq!(light.light_type(), LightType::Point);
    }

    // -------------------------------------------------------------------------
    // LightType Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_light_type_from_u32() {
        assert_eq!(LightType::from_u32(0), Some(LightType::Directional));
        assert_eq!(LightType::from_u32(1), Some(LightType::Point));
        assert_eq!(LightType::from_u32(2), Some(LightType::Spot));
        assert_eq!(LightType::from_u32(3), None);
        assert_eq!(LightType::from_u32(100), None);
    }

    #[test]
    fn test_light_type_default() {
        assert_eq!(LightType::default(), LightType::Point);
    }

    // -------------------------------------------------------------------------
    // ShadowRayParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shadow_ray_params_new() {
        let params = ShadowRayParams::new(
            ShadowRayParams::identity_matrix(),
            1920,
            1080,
            4,
        );

        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        assert_eq!(params.light_count, 4);
        assert_eq!(params.bias, DEFAULT_BIAS);
    }

    #[test]
    fn test_shadow_ray_params_with_bias() {
        let params = ShadowRayParams::new(
            ShadowRayParams::identity_matrix(),
            1920,
            1080,
            4,
        ).with_bias(0.005);

        assert_eq!(params.bias, 0.005);
    }

    #[test]
    fn test_shadow_ray_params_output_buffer_size() {
        let params = ShadowRayParams::new(
            ShadowRayParams::identity_matrix(),
            1920,
            1080,
            4,
        );

        // 1920 * 1080 * 4 lights * 4 bytes = 33,177,600 bytes
        assert_eq!(params.output_buffer_size(), 1920 * 1080 * 4 * 4);
    }

    #[test]
    fn test_shadow_ray_params_default() {
        let params = ShadowRayParams::default();
        assert_eq!(params.light_count, 0);
        assert_eq!(params.width, 0);
        assert_eq!(params.height, 0);
        assert_eq!(params.bias, DEFAULT_BIAS);
    }

    #[test]
    fn test_identity_matrix() {
        let mat = ShadowRayParams::identity_matrix();
        assert_eq!(mat[0][0], 1.0);
        assert_eq!(mat[1][1], 1.0);
        assert_eq!(mat[2][2], 1.0);
        assert_eq!(mat[3][3], 1.0);
        assert_eq!(mat[0][1], 0.0);
        assert_eq!(mat[1][0], 0.0);
    }

    // -------------------------------------------------------------------------
    // Workgroup Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_workgroup_counts_exact() {
        let (x, y, z) = RTShadowPipeline::workgroup_counts(1920, 1080);
        assert_eq!(x, 240); // 1920 / 8 = 240
        assert_eq!(y, 135); // 1080 / 8 = 135
        assert_eq!(z, 1);
    }

    #[test]
    fn test_workgroup_counts_rounded() {
        let (x, y, z) = RTShadowPipeline::workgroup_counts(1921, 1081);
        assert_eq!(x, 241); // ceil(1921 / 8) = 241
        assert_eq!(y, 136); // ceil(1081 / 8) = 136
        assert_eq!(z, 1);
    }

    #[test]
    fn test_workgroup_counts_small() {
        let (x, y, z) = RTShadowPipeline::workgroup_counts(8, 8);
        assert_eq!(x, 1);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_workgroup_counts_minimum() {
        let (x, y, z) = RTShadowPipeline::workgroup_counts(1, 1);
        assert_eq!(x, 1);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    // -------------------------------------------------------------------------
    // Shader Validation Tests (naga)
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_parses_successfully() {
        use naga::front::wgsl;

        let shader_source = include_str!("../../shaders/raytracing/rt_shadow.comp.wgsl");
        let result = wgsl::parse_str(shader_source);

        assert!(result.is_ok(), "Shader parsing failed: {:?}", result.err());
    }

    #[test]
    fn test_shader_validates_successfully() {
        let result = validate_shader();
        assert!(result.is_ok(), "Shader validation failed: {:?}", result.err());
    }

    #[test]
    fn test_shader_has_main_entry_point() {
        use naga::front::wgsl;

        let shader_source = include_str!("../../shaders/raytracing/rt_shadow.comp.wgsl");
        let module = wgsl::parse_str(shader_source).expect("Failed to parse shader");

        let has_main = module.entry_points.iter().any(|ep| ep.name == "main");
        assert!(has_main, "Shader must have 'main' entry point");
    }

    #[test]
    fn test_shader_is_compute_shader() {
        use naga::front::wgsl;

        let shader_source = include_str!("../../shaders/raytracing/rt_shadow.comp.wgsl");
        let module = wgsl::parse_str(shader_source).expect("Failed to parse shader");

        let main_ep = module.entry_points.iter().find(|ep| ep.name == "main");
        assert!(main_ep.is_some(), "Shader must have 'main' entry point");

        let ep = main_ep.unwrap();
        assert_eq!(ep.stage, naga::ShaderStage::Compute, "Entry point must be compute stage");
    }

    #[test]
    fn test_shader_workgroup_size() {
        use naga::front::wgsl;

        let shader_source = include_str!("../../shaders/raytracing/rt_shadow.comp.wgsl");
        let module = wgsl::parse_str(shader_source).expect("Failed to parse shader");

        let main_ep = module.entry_points.iter().find(|ep| ep.name == "main").unwrap();
        assert_eq!(main_ep.workgroup_size, [8, 8, 1], "Workgroup size must be 8x8x1");
    }

    // -------------------------------------------------------------------------
    // Bytemuck Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_light_is_pod() {
        let light = Light::default();
        let bytes = bytemuck::bytes_of(&light);
        assert_eq!(bytes.len(), 48);
    }

    #[test]
    fn test_shadow_ray_params_is_pod() {
        let params = ShadowRayParams::default();
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 80);
    }

    #[test]
    fn test_light_array_cast() {
        let lights = vec![
            Light::point([0.0, 0.0, 0.0], 10.0, [1.0, 1.0, 1.0], 1.0),
            Light::directional([0.0, -1.0, 0.0], [1.0, 1.0, 0.9], 1.0),
        ];

        let bytes: &[u8] = bytemuck::cast_slice(&lights);
        assert_eq!(bytes.len(), 96); // 2 * 48 bytes
    }

    // -------------------------------------------------------------------------
    // Constant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_workgroup_size_constant() {
        assert_eq!(WORKGROUP_SIZE, 8);
    }

    #[test]
    fn test_default_bias_constant() {
        assert_eq!(DEFAULT_BIAS, 0.001);
    }

    #[test]
    fn test_max_lights_constant() {
        assert_eq!(MAX_LIGHTS, 256);
    }

    // -------------------------------------------------------------------------
    // Light Field Access Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_light_position_type_field() {
        let light = Light::point([1.0, 2.0, 3.0], 10.0, [1.0, 1.0, 1.0], 1.0);
        assert_eq!(light.position_type[0], 1.0);
        assert_eq!(light.position_type[1], 2.0);
        assert_eq!(light.position_type[2], 3.0);
        assert_eq!(light.position_type[3], 1.0); // Point type
    }

    #[test]
    fn test_light_direction_range_field() {
        let light = Light::spot([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], 15.0, [1.0, 1.0, 1.0], 1.0);
        assert_eq!(light.direction_range[0], 0.0);
        assert_eq!(light.direction_range[1], -1.0);
        assert_eq!(light.direction_range[2], 0.0);
        assert_eq!(light.direction_range[3], 15.0); // Range
    }

    #[test]
    fn test_light_color_intensity_field() {
        let light = Light::point([0.0, 0.0, 0.0], 10.0, [1.0, 0.5, 0.2], 250.0);
        assert_eq!(light.color_intensity[0], 1.0);
        assert_eq!(light.color_intensity[1], 0.5);
        assert_eq!(light.color_intensity[2], 0.2);
        assert_eq!(light.color_intensity[3], 250.0); // Intensity
    }

    #[test]
    fn test_point_light_direction_default() {
        let light = Light::point([0.0, 0.0, 0.0], 10.0, [1.0, 1.0, 1.0], 1.0);
        // Point lights have no meaningful direction, returns default
        assert_eq!(light.direction(), [0.0, -1.0, 0.0]);
    }
}
