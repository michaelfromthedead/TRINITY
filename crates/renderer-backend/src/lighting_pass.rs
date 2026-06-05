//! Deferred lighting pass dispatch (T-LIT-3.7).
//!
//! This module provides Rust dispatch logic for the deferred PBR lighting
//! compute shader (`lighting_pass.comp.wgsl`). The shader reads G-Buffer
//! textures, evaluates Cook-Torrance BRDF for all lights using froxel-based
//! culling, and outputs HDR lighting.
//!
//! # Bind Groups
//!
//! | Group | Contents |
//! |-------|----------|
//! | 0     | G-Buffer textures (albedo, normal, roughness_metallic, depth, sampler) |
//! | 1     | Light data buffers + froxel indices + camera + grid config |
//! | 2     | Output HDR texture (storage, write) |
//! | 3     | Shadow resources (cascade data, shadow maps, sampler, params) |
//!
//! # Usage
//!
//! ```ignore
//! let lighting_pass = LightingPass::new(&device, &shader_source);
//!
//! let bind_groups = lighting_pass.create_bind_groups(
//!     &device,
//!     &g_buffer,
//!     &light_buffers,
//!     &froxel_grid,
//!     &froxel_light_indices,
//!     &shadow_resources,
//!     &output_hdr_view,
//! );
//!
//! let config = LightingPassConfig {
//!     screen_size: [1920, 1080],
//!     workgroup_size: [8, 8],
//! };
//!
//! lighting_pass.dispatch(&mut encoder, &bind_groups, &config);
//! ```

use std::borrow::Cow;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the lighting pass dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightingPassConfig {
    /// Screen dimensions in pixels [width, height].
    pub screen_size: [u32; 2],

    /// Workgroup size matching the shader's @workgroup_size.
    /// Typically [8, 8] for the lighting_pass.comp.wgsl shader.
    pub workgroup_size: [u32; 2],
}

impl Default for LightingPassConfig {
    fn default() -> Self {
        Self {
            screen_size: [1920, 1080],
            workgroup_size: [8, 8],
        }
    }
}

impl LightingPassConfig {
    /// Create a new configuration with the given screen size.
    ///
    /// Uses default workgroup size of [8, 8].
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            screen_size: [width, height],
            workgroup_size: [8, 8],
        }
    }

    /// Create a configuration with custom workgroup size.
    pub fn with_workgroup_size(width: u32, height: u32, wg_x: u32, wg_y: u32) -> Self {
        debug_assert!(wg_x > 0 && wg_y > 0, "workgroup size must be non-zero");
        Self {
            screen_size: [width, height],
            workgroup_size: [wg_x, wg_y],
        }
    }

    /// Calculate the number of workgroups to dispatch.
    ///
    /// Returns `(workgroups_x, workgroups_y)` using ceiling division
    /// to ensure full screen coverage.
    #[inline]
    pub fn workgroup_count(&self) -> (u32, u32) {
        let x = (self.screen_size[0] + self.workgroup_size[0] - 1) / self.workgroup_size[0];
        let y = (self.screen_size[1] + self.workgroup_size[1] - 1) / self.workgroup_size[1];
        (x, y)
    }
}

// ---------------------------------------------------------------------------
// G-Buffer Textures
// ---------------------------------------------------------------------------

/// References to G-Buffer textures for the lighting pass.
///
/// These textures are produced by the geometry pass and consumed
/// by the deferred lighting shader.
pub struct GBufferTextures<'a> {
    /// Albedo texture (RGBA8 or RGBA16Float).
    /// RGB = base color, A = unused or AO.
    pub albedo: &'a wgpu::TextureView,

    /// World-space normal texture (RGBA16Float or RGBA8Snorm).
    /// RGB = (normal * 0.5 + 0.5), A = unused.
    pub normal: &'a wgpu::TextureView,

    /// Roughness/metallic/AO packed texture.
    /// R = roughness, G = metallic, B = ambient occlusion.
    pub roughness_metallic: &'a wgpu::TextureView,

    /// Depth buffer (Depth32Float).
    pub depth: &'a wgpu::TextureView,

    /// Sampler for G-Buffer textures.
    pub sampler: &'a wgpu::Sampler,
}

// ---------------------------------------------------------------------------
// Light GPU Buffers
// ---------------------------------------------------------------------------

/// References to GPU buffers containing light data and froxel grid.
///
/// These buffers are produced by the light culling pass and consumed
/// by the deferred lighting shader.
pub struct LightGpuBuffers<'a> {
    /// Point lights storage buffer.
    pub point_lights: &'a wgpu::Buffer,

    /// Spot lights storage buffer.
    pub spot_lights: &'a wgpu::Buffer,

    /// Directional lights storage buffer.
    pub directional_lights: &'a wgpu::Buffer,

    /// Light counts uniform buffer (num_directional, num_point, num_spot, pad).
    pub light_counts: &'a wgpu::Buffer,

    /// Froxel grid storage buffer (array of Froxel structs).
    pub froxel_grid: &'a wgpu::Buffer,

    /// Froxel light indices storage buffer (packed light type + index).
    pub froxel_light_indices: &'a wgpu::Buffer,

    /// Camera uniforms buffer (view, projection, inverse matrices, position).
    pub camera: &'a wgpu::Buffer,

    /// Froxel grid configuration uniform buffer.
    pub grid_config: &'a wgpu::Buffer,
}

// ---------------------------------------------------------------------------
// Shadow Resources
// ---------------------------------------------------------------------------

/// GPU-side cascade data for CSM shadows.
///
/// Matches the shader's CascadeData struct (80 bytes per cascade).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CascadeDataGpu {
    /// Light view-projection matrix (column-major).
    pub light_view_proj: [[f32; 4]; 4],
    /// View-space split depth for cascade selection.
    pub split_depth: f32,
    /// Shadow map array layer index.
    pub shadow_map_index: u32,
    /// Padding for 16-byte alignment.
    pub _pad: [f32; 2],
}

impl Default for CascadeDataGpu {
    fn default() -> Self {
        Self {
            light_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            split_depth: 1000.0,
            shadow_map_index: 0,
            _pad: [0.0; 2],
        }
    }
}

// Size assertion: 80 bytes per cascade
const _: () = assert!(std::mem::size_of::<CascadeDataGpu>() == 80);

/// References to shadow mapping resources for the lighting pass.
pub struct ShadowResources<'a> {
    /// Cascade data uniform buffer (array of 4 CascadeDataGpu).
    pub cascade_data: &'a wgpu::Buffer,

    /// Shadow map depth texture array (4 layers for CSM).
    pub shadow_maps: &'a wgpu::TextureView,

    /// Comparison sampler for shadow lookups.
    pub shadow_sampler: &'a wgpu::Sampler,

    /// Shadow parameters uniform buffer.
    /// vec4: (bias, slope_bias, normal_bias, pcf_radius).
    pub shadow_params: &'a wgpu::Buffer,

    /// T-LIT-4.4: Cascade blend range uniform buffer.
    /// f32: distance in world units over which cascades blend (default 2.0).
    pub cascade_blend_range: &'a wgpu::Buffer,
}

// ---------------------------------------------------------------------------
// Contact Shadow Resources (T-LIT-8.2)
// ---------------------------------------------------------------------------

/// References to contact shadow resources for the lighting pass.
///
/// Contact shadows enhance shadow quality by ray-marching in screen space
/// to detect small-scale occlusions that traditional shadow maps miss.
pub struct ContactShadowResources<'a> {
    /// Contact shadow texture from the contact shadow pass.
    ///
    /// Format: R8Unorm, where 0 = full shadow, 1 = no shadow.
    pub contact_shadow_texture: &'a wgpu::TextureView,

    /// Sampler for the contact shadow texture.
    pub contact_shadow_sampler: &'a wgpu::Sampler,

    /// Contact shadow blend configuration uniform buffer.
    ///
    /// Contains blend mode, intensity, and fallback value.
    pub blend_config: &'a wgpu::Buffer,
}

impl<'a> ContactShadowResources<'a> {
    /// Create new contact shadow resources.
    pub fn new(
        contact_shadow_texture: &'a wgpu::TextureView,
        contact_shadow_sampler: &'a wgpu::Sampler,
        blend_config: &'a wgpu::Buffer,
    ) -> Self {
        Self {
            contact_shadow_texture,
            contact_shadow_sampler,
            blend_config,
        }
    }
}

// ---------------------------------------------------------------------------
// LightingPass
// ---------------------------------------------------------------------------

/// Deferred lighting pass compute pipeline and bind group layouts.
///
/// Encapsulates the compute pipeline for evaluating PBR lighting from
/// G-Buffer data using froxel-based light culling.
pub struct LightingPass {
    /// The compute pipeline for the lighting pass shader.
    pipeline: wgpu::ComputePipeline,

    /// Bind group layouts for the 4 shader groups.
    bind_group_layouts: [wgpu::BindGroupLayout; 4],
}

impl LightingPass {
    /// Create a new lighting pass from WGSL shader source.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `shader_source` - WGSL source code for the lighting_pass.comp.wgsl shader.
    ///
    /// # Panics
    ///
    /// Panics if shader compilation fails.
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        // Create bind group layouts for all 4 groups
        let bind_group_layouts = Self::create_bind_group_layouts(device);

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LightingPass Pipeline Layout"),
            bind_group_layouts: &[
                &bind_group_layouts[0],
                &bind_group_layouts[1],
                &bind_group_layouts[2],
                &bind_group_layouts[3],
            ],
            push_constant_ranges: &[],
        });

        // Compile shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LightingPass Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LightingPass Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layouts,
        }
    }

    /// Create bind group layouts matching the shader's 4 groups.
    fn create_bind_group_layouts(device: &wgpu::Device) -> [wgpu::BindGroupLayout; 4] {
        let visibility = wgpu::ShaderStages::COMPUTE;

        // Group 0: G-Buffer textures
        let group0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LightingPass Group 0 (G-Buffer)"),
            entries: &[
                // @binding(0) var g_albedo: texture_2d<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(1) var g_normal: texture_2d<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2) var g_roughness_metallic: texture_2d<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(3) var g_depth: texture_depth_2d
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(4) var g_sampler: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Group 1: Light data and froxel grid
        let group1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LightingPass Group 1 (Lights)"),
            entries: &[
                // @binding(0) var<storage, read> light_buffer_point: array<PointLight>
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1) var<storage, read> light_buffer_spot: array<SpotLight>
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2) var<storage, read> light_buffer_directional: array<DirectionalLight>
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3) var<uniform> light_counts: LightCounts
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(4) var<storage, read> froxel_grid: array<Froxel>
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(5) var<storage, read> froxel_light_indices: array<u32>
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(6) var<uniform> camera: CameraUniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(7) var<uniform> grid_config: FroxelGridConfig
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Group 2: Output HDR texture
        let group2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LightingPass Group 2 (Output)"),
            entries: &[
                // @binding(0) var output_hdr: texture_storage_2d<rgba16float, write>
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Group 3: Shadow resources
        let group3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LightingPass Group 3 (Shadows)"),
            entries: &[
                // @binding(0) var<uniform> cascade_data: array<CascadeData, 4>
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1) var shadow_maps: texture_depth_2d_array
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2) var shadow_sampler: sampler_comparison
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // @binding(3) var<uniform> shadow_params: vec4<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // T-LIT-4.4: @binding(4) var<uniform> cascade_blend_range: f32
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        [group0, group1, group2, group3]
    }

    /// Create bind groups for all 4 shader groups.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `g_buffer` - G-Buffer texture references.
    /// * `light_buffers` - Light data and froxel grid buffers.
    /// * `shadow_resources` - Shadow mapping resources.
    /// * `output_hdr` - Output HDR texture view (storage, write).
    ///
    /// # Returns
    ///
    /// An array of 4 bind groups matching the shader layout.
    pub fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        g_buffer: &GBufferTextures<'_>,
        light_buffers: &LightGpuBuffers<'_>,
        shadow_resources: &ShadowResources<'_>,
        output_hdr: &wgpu::TextureView,
    ) -> [wgpu::BindGroup; 4] {
        // Group 0: G-Buffer textures
        let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LightingPass BindGroup 0"),
            layout: &self.bind_group_layouts[0],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(g_buffer.albedo),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(g_buffer.normal),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(g_buffer.roughness_metallic),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(g_buffer.depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(g_buffer.sampler),
                },
            ],
        });

        // Group 1: Light data and froxel grid
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LightingPass BindGroup 1"),
            layout: &self.bind_group_layouts[1],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffers.point_lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffers.spot_lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: light_buffers.directional_lights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: light_buffers.light_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: light_buffers.froxel_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: light_buffers.froxel_light_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: light_buffers.camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: light_buffers.grid_config.as_entire_binding(),
                },
            ],
        });

        // Group 2: Output HDR texture
        let group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LightingPass BindGroup 2"),
            layout: &self.bind_group_layouts[2],
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(output_hdr),
            }],
        });

        // Group 3: Shadow resources
        let group3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LightingPass BindGroup 3"),
            layout: &self.bind_group_layouts[3],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_resources.cascade_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(shadow_resources.shadow_maps),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(shadow_resources.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: shadow_resources.shadow_params.as_entire_binding(),
                },
                // T-LIT-4.4: Cascade blend range
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: shadow_resources.cascade_blend_range.as_entire_binding(),
                },
            ],
        });

        [group0, group1, group2, group3]
    }

    /// Dispatch the lighting pass compute shader.
    ///
    /// Records a compute pass to the given command encoder that evaluates
    /// PBR lighting for all pixels based on the G-Buffer and light data.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record the compute pass.
    /// * `bind_groups` - The 4 bind groups created by `create_bind_groups`.
    /// * `config` - Lighting pass configuration (screen size, workgroup size).
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_groups: &[wgpu::BindGroup; 4],
        config: &LightingPassConfig,
    ) {
        let (wg_x, wg_y) = config.workgroup_count();

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LightingPass Compute"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_groups[0], &[]);
        compute_pass.set_bind_group(1, &bind_groups[1], &[]);
        compute_pass.set_bind_group(2, &bind_groups[2], &[]);
        compute_pass.set_bind_group(3, &bind_groups[3], &[]);
        compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// Access the underlying compute pipeline.
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Access the bind group layouts.
    pub fn bind_group_layouts(&self) -> &[wgpu::BindGroupLayout; 4] {
        &self.bind_group_layouts
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Calculate workgroup count for a given screen dimension.
///
/// Uses ceiling division to ensure full coverage.
#[inline]
pub fn calculate_workgroup_count(screen_dim: u32, workgroup_dim: u32) -> u32 {
    (screen_dim + workgroup_dim - 1) / workgroup_dim
}

/// Create a default shadow params buffer for testing.
///
/// Returns (bias, slope_bias, normal_bias, pcf_radius).
pub fn default_shadow_params() -> [f32; 4] {
    [
        0.001,  // bias
        0.001,  // slope_bias
        0.005,  // normal_bias
        1.0,    // pcf_radius (1 = 3x3 kernel)
    ]
}

/// T-LIT-4.4: Default cascade blend range in world units.
///
/// Returns the default blend range (2.0 world units) matching
/// the Python CascadedShadowMap.cascade_blend_range default.
pub const DEFAULT_CASCADE_BLEND_RANGE: f32 = 2.0;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- LightingPassConfig ---------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = LightingPassConfig::default();
        assert_eq!(config.screen_size, [1920, 1080]);
        assert_eq!(config.workgroup_size, [8, 8]);
    }

    #[test]
    fn test_config_new() {
        let config = LightingPassConfig::new(1280, 720);
        assert_eq!(config.screen_size, [1280, 720]);
        assert_eq!(config.workgroup_size, [8, 8]);
    }

    #[test]
    fn test_config_with_workgroup_size() {
        let config = LightingPassConfig::with_workgroup_size(2560, 1440, 16, 16);
        assert_eq!(config.screen_size, [2560, 1440]);
        assert_eq!(config.workgroup_size, [16, 16]);
    }

    #[test]
    fn test_workgroup_count_exact() {
        // Screen size exactly divisible by workgroup size
        let config = LightingPassConfig::new(1920, 1080);
        let (wg_x, wg_y) = config.workgroup_count();
        assert_eq!(wg_x, 240); // 1920 / 8 = 240
        assert_eq!(wg_y, 135); // 1080 / 8 = 135
    }

    #[test]
    fn test_workgroup_count_ceiling() {
        // Screen size not exactly divisible - needs ceiling
        let config = LightingPassConfig::new(1921, 1081);
        let (wg_x, wg_y) = config.workgroup_count();
        assert_eq!(wg_x, 241); // ceil(1921 / 8) = 241
        assert_eq!(wg_y, 136); // ceil(1081 / 8) = 136
    }

    #[test]
    fn test_workgroup_count_small() {
        // Very small screen (edge case)
        let config = LightingPassConfig::new(1, 1);
        let (wg_x, wg_y) = config.workgroup_count();
        assert_eq!(wg_x, 1);
        assert_eq!(wg_y, 1);
    }

    #[test]
    fn test_workgroup_count_large_workgroup() {
        // Larger workgroup size
        let config = LightingPassConfig::with_workgroup_size(1920, 1080, 32, 32);
        let (wg_x, wg_y) = config.workgroup_count();
        assert_eq!(wg_x, 60);  // 1920 / 32 = 60
        assert_eq!(wg_y, 34);  // ceil(1080 / 32) = 33.75 -> 34
    }

    #[test]
    fn test_workgroup_count_4k() {
        // 4K resolution
        let config = LightingPassConfig::new(3840, 2160);
        let (wg_x, wg_y) = config.workgroup_count();
        assert_eq!(wg_x, 480); // 3840 / 8 = 480
        assert_eq!(wg_y, 270); // 2160 / 8 = 270
    }

    // -- CascadeDataGpu -------------------------------------------------------

    #[test]
    fn test_cascade_data_size() {
        // CascadeData should be 80 bytes (4x4 matrix + 2 floats + 2 padding)
        assert_eq!(std::mem::size_of::<CascadeDataGpu>(), 80);
    }

    #[test]
    fn test_cascade_data_default() {
        let cascade = CascadeDataGpu::default();
        assert_eq!(cascade.split_depth, 1000.0);
        assert_eq!(cascade.shadow_map_index, 0);
        // Identity matrix check
        assert_eq!(cascade.light_view_proj[0][0], 1.0);
        assert_eq!(cascade.light_view_proj[1][1], 1.0);
        assert_eq!(cascade.light_view_proj[2][2], 1.0);
        assert_eq!(cascade.light_view_proj[3][3], 1.0);
    }

    // -- Helper functions -----------------------------------------------------

    #[test]
    fn test_calculate_workgroup_count() {
        assert_eq!(calculate_workgroup_count(1920, 8), 240);
        assert_eq!(calculate_workgroup_count(1921, 8), 241);
        assert_eq!(calculate_workgroup_count(8, 8), 1);
        assert_eq!(calculate_workgroup_count(9, 8), 2);
        assert_eq!(calculate_workgroup_count(1, 8), 1);
    }

    #[test]
    fn test_default_shadow_params() {
        let params = default_shadow_params();
        assert_eq!(params.len(), 4);
        assert!(params[0] > 0.0); // bias
        assert!(params[1] > 0.0); // slope_bias
        assert!(params[2] > 0.0); // normal_bias
        assert!(params[3] >= 0.0); // pcf_radius
    }

    // -- GPU integration tests ------------------------------------------------

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
    fn test_bind_group_layout_creation() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let layouts = LightingPass::create_bind_group_layouts(&device);
        assert_eq!(layouts.len(), 4);
    }

    #[test]
    fn test_workgroup_count_non_square() {
        // Non-square screen with non-square workgroup
        let config = LightingPassConfig::with_workgroup_size(800, 600, 16, 8);
        let (wg_x, wg_y) = config.workgroup_count();
        assert_eq!(wg_x, 50);  // 800 / 16 = 50
        assert_eq!(wg_y, 75);  // 600 / 8 = 75
    }

    #[test]
    fn test_config_equality() {
        let c1 = LightingPassConfig::new(1920, 1080);
        let c2 = LightingPassConfig::new(1920, 1080);
        let c3 = LightingPassConfig::new(1280, 720);

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_cascade_data_array_size() {
        // 4 cascades should fit in 320 bytes
        let cascades = [CascadeDataGpu::default(); 4];
        assert_eq!(std::mem::size_of_val(&cascades), 320);
    }

    #[test]
    fn test_shadow_params_size() {
        // Shadow params should be vec4 = 16 bytes
        let params = default_shadow_params();
        assert_eq!(std::mem::size_of_val(&params), 16);
    }

    // -- ContactShadowResources -----------------------------------------------

    #[test]
    fn test_contact_shadow_resources_creation() {
        // This is a compile-time test to ensure the struct can be created
        // Actual GPU tests require wgpu device
        fn _check_contact_shadow_resources_api<'a>(
            texture: &'a wgpu::TextureView,
            sampler: &'a wgpu::Sampler,
            buffer: &'a wgpu::Buffer,
        ) {
            let _resources = ContactShadowResources::new(texture, sampler, buffer);
        }
    }

    // GPU integration test that creates the full lighting pass pipeline
    #[test]
    fn test_lighting_pass_pipeline_creation() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Minimal valid compute shader for testing pipeline creation
        let minimal_shader = r#"
            @group(0) @binding(0) var g_albedo: texture_2d<f32>;
            @group(0) @binding(1) var g_normal: texture_2d<f32>;
            @group(0) @binding(2) var g_roughness_metallic: texture_2d<f32>;
            @group(0) @binding(3) var g_depth: texture_depth_2d;
            @group(0) @binding(4) var g_sampler: sampler;

            @group(1) @binding(0) var<storage, read> light_buffer_point: array<vec4<f32>>;
            @group(1) @binding(1) var<storage, read> light_buffer_spot: array<vec4<f32>>;
            @group(1) @binding(2) var<storage, read> light_buffer_directional: array<vec4<f32>>;
            @group(1) @binding(3) var<uniform> light_counts: vec4<u32>;
            @group(1) @binding(4) var<storage, read> froxel_grid: array<vec2<u32>>;
            @group(1) @binding(5) var<storage, read> froxel_light_indices: array<u32>;
            @group(1) @binding(6) var<uniform> camera: mat4x4<f32>;
            @group(1) @binding(7) var<uniform> grid_config: vec4<f32>;

            @group(2) @binding(0) var output_hdr: texture_storage_2d<rgba16float, write>;

            @group(3) @binding(0) var<uniform> cascade_data: array<mat4x4<f32>, 4>;
            @group(3) @binding(1) var shadow_maps: texture_depth_2d_array;
            @group(3) @binding(2) var shadow_sampler: sampler_comparison;
            @group(3) @binding(3) var<uniform> shadow_params: vec4<f32>;
            @group(3) @binding(4) var<uniform> cascade_blend_range: f32;

            @compute @workgroup_size(8, 8, 1)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                // Minimal implementation for testing
            }
        "#;

        let lighting_pass = LightingPass::new(&device, minimal_shader);

        // Verify pipeline and layouts were created
        assert_eq!(lighting_pass.bind_group_layouts().len(), 4);
        let _ = lighting_pass.pipeline(); // Should not panic
    }
}
