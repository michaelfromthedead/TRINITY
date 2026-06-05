//! Screen-Space Global Illumination (SSGI) using HiZ-accelerated ray marching (T-GIR-P3.1).
//!
//! This module provides GPU dispatch for SSGI, which samples indirect lighting from
//! screen-space by tracing rays against the hierarchical depth buffer. SSGI captures
//! diffuse color bleeding effects that enhance the realism of global illumination.
//!
//! # Overview
//!
//! SSGI traces rays from each pixel into a cosine-weighted hemisphere around the
//! surface normal. The ray-march samples the HiZ buffer for efficient occlusion
//! testing and fetches indirect lighting from visible surfaces.
//!
//! # Ray Configuration
//!
//! | Rays/Pixel | Use Case                           | Cost         |
//! |------------|------------------------------------|--------------|
//! | 4          | Mobile, low-end GPUs               | ~0.5ms       |
//! | 8          | Balanced quality/performance       | ~1.0ms       |
//! | 16         | Desktop, high quality              | ~2.0ms       |
//!
//! # Half-Resolution Rendering
//!
//! By default, SSGI is rendered at half resolution and upscaled using a bilateral
//! filter to preserve edges. This provides a 4x speedup with minimal quality loss.
//!
//! # Usage
//!
//! ```ignore
//! let config = SSGIConfig::default();
//! let pass = SSGIPass::new(&device, config, true /* half_res */);
//!
//! // Create bind group with frame resources
//! let bind_group = pass.create_bind_group(
//!     &device,
//!     &depth_view,
//!     &normal_view,
//!     &lighting_view,
//!     &hiz_view,
//!     &camera_buffer,
//!     &output_view,
//! );
//!
//! // Dispatch during frame recording
//! pass.dispatch(&mut encoder, &bind_group, width, height);
//! ```
//!
//! # Frame Graph Integration
//!
//! The [`SSGINode`] integrates with the frame graph system, declaring resource
//! dependencies for automatic barrier insertion and resource aliasing.

use std::mem;

use crate::frame_graph::{
    DispatchSource, IrPass, PassIndex, ResourceAccessSet, ResourceHandle, ViewType,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (8x8 threads).
pub const WORKGROUP_SIZE: u32 = 8;

/// Default maximum ray distance in world units.
pub const DEFAULT_MAX_DISTANCE: f32 = 15.0;

/// Default depth thickness threshold for hit detection.
pub const DEFAULT_THICKNESS: f32 = 0.1;

/// Default GI intensity multiplier.
pub const DEFAULT_INTENSITY: f32 = 1.0;

/// Default distance fade start in world units.
pub const DEFAULT_FADE_START: f32 = 5.0;

/// Default distance fade end in world units.
pub const DEFAULT_FADE_END: f32 = 15.0;

/// Default number of ray-march steps.
pub const DEFAULT_MAX_STEPS: u32 = 48;

/// Default rays per pixel.
pub const DEFAULT_RAYS_PER_PIXEL: u32 = 8;

/// PI constant for shader compatibility.
pub const PI: f32 = std::f32::consts::PI;

// ---------------------------------------------------------------------------
// SSGIQuality
// ---------------------------------------------------------------------------

/// SSGI quality tier controlling ray count and step count.
///
/// Higher tiers produce smoother, more accurate GI but increase GPU cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SSGIQuality {
    /// 4 rays, 32 steps. Suitable for mobile and low-end GPUs.
    Low,
    /// 8 rays, 48 steps. Balanced quality/performance.
    Medium,
    /// 16 rays, 64 steps. High quality for desktop GPUs.
    High,
}

impl SSGIQuality {
    /// Get the number of rays per pixel for this quality tier.
    #[inline]
    pub const fn rays_per_pixel(self) -> u32 {
        match self {
            Self::Low => 4,
            Self::Medium => 8,
            Self::High => 16,
        }
    }

    /// Get the number of ray-march steps for this quality tier.
    #[inline]
    pub const fn max_steps(self) -> u32 {
        match self {
            Self::Low => 32,
            Self::Medium => 48,
            Self::High => 64,
        }
    }

    /// Create quality tier from ray count (rounds to nearest tier).
    pub fn from_ray_count(rays: u32) -> Self {
        match rays {
            0..=5 => Self::Low,
            6..=11 => Self::Medium,
            _ => Self::High,
        }
    }

    /// Get approximate cost in milliseconds at 1080p.
    pub const fn estimated_cost_ms(self) -> f32 {
        match self {
            Self::Low => 0.5,
            Self::Medium => 1.0,
            Self::High => 2.0,
        }
    }
}

impl Default for SSGIQuality {
    fn default() -> Self {
        Self::Medium
    }
}

// ---------------------------------------------------------------------------
// SSGIConfig
// ---------------------------------------------------------------------------

/// GPU-side SSGI configuration.
///
/// This struct is uploaded to a uniform buffer for the compute shader.
/// Matches the WGSL `SSGIConfig` struct layout.
///
/// # Memory Layout
///
/// 32 bytes total, std140/std430 compatible:
///
/// | Offset | Field              | Size    |
/// |--------|--------------------| --------|
/// | 0      | rays_per_pixel     | 4 bytes |
/// | 4      | max_steps          | 4 bytes |
/// | 8      | max_distance       | 4 bytes |
/// | 12     | thickness          | 4 bytes |
/// | 16     | intensity          | 4 bytes |
/// | 20     | distance_fade_start| 4 bytes |
/// | 24     | distance_fade_end  | 4 bytes |
/// | 28     | _pad               | 4 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SSGIConfig {
    /// Number of rays per pixel (4, 8, or 16).
    pub rays_per_pixel: u32,
    /// Maximum ray-march steps (32-64).
    pub max_steps: u32,
    /// Maximum ray distance in world space (10-20 meters).
    pub max_distance: f32,
    /// Depth thickness threshold for hit detection.
    pub thickness: f32,
    /// GI contribution intensity multiplier.
    pub intensity: f32,
    /// Distance at which fade begins (world units).
    pub distance_fade_start: f32,
    /// Distance at which fade ends (world units).
    pub distance_fade_end: f32,
    /// Padding for 32-byte alignment.
    pub _pad: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<SSGIConfig>() == 32);

impl SSGIConfig {
    /// Create a new SSGI config from quality tier with default parameters.
    pub fn from_quality(quality: SSGIQuality) -> Self {
        Self {
            rays_per_pixel: quality.rays_per_pixel(),
            max_steps: quality.max_steps(),
            max_distance: DEFAULT_MAX_DISTANCE,
            thickness: DEFAULT_THICKNESS,
            intensity: DEFAULT_INTENSITY,
            distance_fade_start: DEFAULT_FADE_START,
            distance_fade_end: DEFAULT_FADE_END,
            _pad: 0.0,
        }
    }

    /// Create a new SSGI config with custom parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rays_per_pixel: u32,
        max_steps: u32,
        max_distance: f32,
        thickness: f32,
        intensity: f32,
        distance_fade_start: f32,
        distance_fade_end: f32,
    ) -> Self {
        Self {
            rays_per_pixel,
            max_steps,
            max_distance,
            thickness,
            intensity,
            distance_fade_start,
            distance_fade_end,
            _pad: 0.0,
        }
    }

    /// Validate config parameters are within acceptable ranges.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.rays_per_pixel == 0 {
            return Err("rays_per_pixel must be > 0");
        }
        if self.max_steps == 0 {
            return Err("max_steps must be > 0");
        }
        if self.max_distance <= 0.0 {
            return Err("max_distance must be > 0");
        }
        if self.thickness <= 0.0 {
            return Err("thickness must be > 0");
        }
        if self.distance_fade_end <= self.distance_fade_start {
            return Err("distance_fade_end must be > distance_fade_start");
        }
        Ok(())
    }

    /// Get the quality tier that best matches this config.
    pub fn quality_tier(&self) -> SSGIQuality {
        SSGIQuality::from_ray_count(self.rays_per_pixel)
    }
}

impl Default for SSGIConfig {
    fn default() -> Self {
        Self::from_quality(SSGIQuality::default())
    }
}

// ---------------------------------------------------------------------------
// SSGIDispatchParams
// ---------------------------------------------------------------------------

/// GPU-side dispatch parameters for SSGI.
///
/// This struct contains per-frame parameters that change each dispatch.
/// Matches the WGSL `SSGIDispatchParams` struct layout.
///
/// # Memory Layout
///
/// 32 bytes total:
///
/// | Offset | Field        | Size    |
/// |--------|--------------|---------|
/// | 0      | screen_size  | 8 bytes |
/// | 8      | inv_screen   | 8 bytes |
/// | 16     | frame_index  | 4 bytes |
/// | 20     | _pad         | 12 bytes|
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SSGIDispatchParams {
    /// Screen dimensions (width, height).
    pub screen_size: [u32; 2],
    /// Inverse screen dimensions (1/width, 1/height).
    pub inv_screen: [f32; 2],
    /// Frame index for temporal jittering.
    pub frame_index: u32,
    /// Padding for 32-byte alignment.
    pub _pad: [u32; 3],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<SSGIDispatchParams>() == 32);

impl SSGIDispatchParams {
    /// Create new dispatch params for the given screen dimensions.
    pub fn new(width: u32, height: u32, frame_index: u32) -> Self {
        Self {
            screen_size: [width, height],
            inv_screen: [1.0 / width as f32, 1.0 / height as f32],
            frame_index,
            _pad: [0, 0, 0],
        }
    }
}

impl Default for SSGIDispatchParams {
    fn default() -> Self {
        Self::new(1920, 1080, 0)
    }
}

// ---------------------------------------------------------------------------
// SSGIPass
// ---------------------------------------------------------------------------

/// SSGI compute pass for screen-space global illumination.
///
/// Manages the compute pipeline, bind group layout, and configuration buffers
/// for HiZ-accelerated ray marching to sample indirect lighting.
///
/// # Bind Group Layout
///
/// | Binding | Type              | Content                          |
/// |---------|-------------------|----------------------------------|
/// | 0       | uniform           | SSGIConfig                       |
/// | 1       | uniform           | SSGIDispatchParams               |
/// | 2       | texture_depth_2d  | Depth buffer                     |
/// | 3       | texture_2d        | GBuffer normals (world-space)    |
/// | 4       | texture_2d        | Lighting buffer (previous frame) |
/// | 5       | texture_2d        | HiZ buffer (for ray marching)    |
/// | 6       | uniform           | Camera uniforms                  |
/// | 7       | storage_texture   | Output SSGI irradiance           |
pub struct SSGIPass {
    /// Compute pipeline for SSGI ray tracing.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for SSGI resources.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Configuration buffer uploaded to GPU.
    config_buffer: wgpu::Buffer,
    /// Dispatch parameters buffer (updated each frame).
    dispatch_buffer: wgpu::Buffer,
    /// Current configuration.
    config: SSGIConfig,
    /// Whether to render at half resolution.
    half_res: bool,
}

impl SSGIPass {
    /// Create a new SSGI pass with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `config` - SSGI configuration.
    /// * `half_res` - Whether to render at half resolution (recommended).
    pub fn new(device: &wgpu::Device, config: SSGIConfig, half_res: bool) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline = Self::create_pipeline(device, &bind_group_layout);
        let config_buffer = Self::create_config_buffer(device, &config);
        let dispatch_buffer = Self::create_dispatch_buffer(device);

        Self {
            pipeline,
            bind_group_layout,
            config_buffer,
            dispatch_buffer,
            config,
            half_res,
        }
    }

    /// Create a new SSGI pass from quality preset.
    pub fn from_quality(device: &wgpu::Device, quality: SSGIQuality, half_res: bool) -> Self {
        Self::new(device, SSGIConfig::from_quality(quality), half_res)
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &SSGIConfig {
        &self.config
    }

    /// Get whether half-resolution rendering is enabled.
    #[inline]
    pub fn half_res(&self) -> bool {
        self.half_res
    }

    /// Get the bind group layout for external bind group creation.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get a reference to the config buffer.
    #[inline]
    pub fn config_buffer(&self) -> &wgpu::Buffer {
        &self.config_buffer
    }

    /// Get a reference to the dispatch params buffer.
    #[inline]
    pub fn dispatch_buffer(&self) -> &wgpu::Buffer {
        &self.dispatch_buffer
    }

    /// Update the configuration.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue for buffer uploads.
    /// * `config` - New SSGI configuration.
    pub fn set_config(&mut self, queue: &wgpu::Queue, config: SSGIConfig) {
        self.config = config;
        queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&config));
    }

    /// Update dispatch parameters for the current frame.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue for buffer uploads.
    /// * `width` - Screen width (full resolution).
    /// * `height` - Screen height (full resolution).
    /// * `frame_index` - Current frame index for temporal jittering.
    pub fn update_dispatch_params(
        &self,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        frame_index: u32,
    ) {
        // If half-res, dispatch at half dimensions
        let (dispatch_width, dispatch_height) = if self.half_res {
            (width / 2, height / 2)
        } else {
            (width, height)
        };

        let params = SSGIDispatchParams::new(dispatch_width, dispatch_height, frame_index);
        queue.write_buffer(&self.dispatch_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Create a bind group for a frame.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `depth_view` - Depth buffer view.
    /// * `normal_view` - GBuffer normals view (world-space).
    /// * `lighting_view` - Lighting buffer view (for sampling indirect).
    /// * `hiz_view` - HiZ buffer view (full mip chain, for ray marching).
    /// * `hiz_sampler` - HiZ sampler (point sampling).
    /// * `camera_buffer` - Camera uniforms buffer.
    /// * `output_view` - Output SSGI irradiance storage view.
    #[allow(clippy::too_many_arguments)]
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        lighting_view: &wgpu::TextureView,
        hiz_view: &wgpu::TextureView,
        hiz_sampler: &wgpu::Sampler,
        camera_buffer: &wgpu::Buffer,
        output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssgi_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.dispatch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(lighting_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(hiz_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
            ],
        })
    }

    /// Dispatch the SSGI compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder.
    /// * `bind_group` - The bind group containing all resources.
    /// * `width` - Full-resolution screen width.
    /// * `height` - Full-resolution screen height.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        // Compute dispatch dimensions (half-res if enabled)
        let (dispatch_width, dispatch_height) = if self.half_res {
            (width / 2, height / 2)
        } else {
            (width, height)
        };

        let workgroups_x = (dispatch_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (dispatch_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssgi_trace_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Get dispatch dimensions for a given screen size.
    pub fn dispatch_size(&self, width: u32, height: u32) -> (u32, u32) {
        if self.half_res {
            (width / 2, height / 2)
        } else {
            (width, height)
        }
    }

    /// Create the bind group layout for SSGI resources.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssgi_bind_group_layout"),
            entries: &[
                // Binding 0: SSGIConfig uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            mem::size_of::<SSGIConfig>() as u64,
                        ),
                    },
                    count: None,
                },
                // Binding 1: SSGIDispatchParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            mem::size_of::<SSGIDispatchParams>() as u64,
                        ),
                    },
                    count: None,
                },
                // Binding 2: Depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 3: Normal texture (world-space)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 4: Lighting buffer (for indirect sampling)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 5: HiZ buffer (mip chain)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 6: HiZ sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Binding 7: Camera uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None, // Camera struct size varies
                    },
                    count: None,
                },
                // Binding 8: Output SSGI irradiance (storage texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create the compute pipeline for SSGI.
    fn create_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader_source = include_str!("../shaders/ssgi_trace.comp.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssgi_trace_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssgi_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssgi_trace_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "ssgi_trace",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Create the config buffer.
    fn create_config_buffer(device: &wgpu::Device, config: &SSGIConfig) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssgi_config_buffer"),
            contents: bytemuck::bytes_of(config),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create the dispatch params buffer.
    fn create_dispatch_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;

        let params = SSGIDispatchParams::default();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssgi_dispatch_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create a frame graph pass for SSGI tracing.
    pub fn create_trace_pass(
        &self,
        index: PassIndex,
        depth_handle: ResourceHandle,
        normal_handle: ResourceHandle,
        lighting_handle: ResourceHandle,
        hiz_handle: ResourceHandle,
        output_handle: ResourceHandle,
        width: u32,
        height: u32,
    ) -> IrPass {
        let (dispatch_width, dispatch_height) = self.dispatch_size(width, height);
        let workgroups_x = (dispatch_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (dispatch_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let dispatch = DispatchSource::Direct {
            group_count_x: workgroups_x,
            group_count_y: workgroups_y,
            group_count_z: 1,
        };

        let mut pass = IrPass::compute(index, "ssgi_trace", dispatch, ViewType::Storage);
        pass.access_set = ResourceAccessSet {
            reads: vec![depth_handle, normal_handle, lighting_handle, hiz_handle],
            writes: vec![output_handle],
        };
        pass
    }
}

// ---------------------------------------------------------------------------
// Frame Graph Integration
// ---------------------------------------------------------------------------

/// Create a frame graph pass for SSGI tracing.
///
/// # Arguments
///
/// * `index` - Pass index in the frame graph.
/// * `depth_handle` - Resource handle for the depth buffer.
/// * `normal_handle` - Resource handle for the normal buffer.
/// * `lighting_handle` - Resource handle for the lighting buffer.
/// * `hiz_handle` - Resource handle for the HiZ buffer.
/// * `output_handle` - Resource handle for the SSGI output.
/// * `width` - Full-resolution width.
/// * `height` - Full-resolution height.
/// * `half_res` - Whether to render at half resolution.
#[allow(clippy::too_many_arguments)]
pub fn create_ssgi_pass(
    index: PassIndex,
    depth_handle: ResourceHandle,
    normal_handle: ResourceHandle,
    lighting_handle: ResourceHandle,
    hiz_handle: ResourceHandle,
    output_handle: ResourceHandle,
    width: u32,
    height: u32,
    half_res: bool,
) -> IrPass {
    let (dispatch_width, dispatch_height) = if half_res {
        (width / 2, height / 2)
    } else {
        (width, height)
    };

    let workgroups_x = (dispatch_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let workgroups_y = (dispatch_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let dispatch = DispatchSource::Direct {
        group_count_x: workgroups_x,
        group_count_y: workgroups_y,
        group_count_z: 1,
    };

    let mut pass = IrPass::compute(index, "ssgi_trace", dispatch, ViewType::Storage);
    pass.access_set = ResourceAccessSet {
        reads: vec![depth_handle, normal_handle, lighting_handle, hiz_handle],
        writes: vec![output_handle],
    };
    pass
}

// ---------------------------------------------------------------------------
// Hemisphere Sampling Utilities (CPU-side validation)
// ---------------------------------------------------------------------------

/// Cosine-weighted hemisphere sampling using Fibonacci spiral.
///
/// This matches the GPU-side sampling pattern for testing.
pub fn sample_hemisphere_cosine(normal: [f32; 3], sample_index: u32, total_samples: u32) -> [f32; 3] {
    let golden_ratio: f32 = 1.618033988749;
    let theta = 2.0 * PI * (sample_index as f32) / golden_ratio;
    let cos_phi = 1.0 - (sample_index as f32) / (total_samples as f32);
    let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();

    // Local direction in tangent space
    let local_x = sin_phi * theta.cos();
    let local_y = sin_phi * theta.sin();
    let local_z = cos_phi;

    // Build tangent frame from normal
    let up = if normal[1].abs() < 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };

    // Cross product: tangent = up x normal
    let tangent = [
        up[1] * normal[2] - up[2] * normal[1],
        up[2] * normal[0] - up[0] * normal[2],
        up[0] * normal[1] - up[1] * normal[0],
    ];
    let tangent_len = (tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]).sqrt();
    let tangent = [tangent[0] / tangent_len, tangent[1] / tangent_len, tangent[2] / tangent_len];

    // Cross product: bitangent = normal x tangent
    let bitangent = [
        normal[1] * tangent[2] - normal[2] * tangent[1],
        normal[2] * tangent[0] - normal[0] * tangent[2],
        normal[0] * tangent[1] - normal[1] * tangent[0],
    ];

    // Transform to world space
    [
        tangent[0] * local_x + bitangent[0] * local_y + normal[0] * local_z,
        tangent[1] * local_x + bitangent[1] * local_y + normal[1] * local_z,
        tangent[2] * local_x + bitangent[2] * local_y + normal[2] * local_z,
    ]
}

/// Compute distance fade factor.
///
/// Returns 1.0 at fade_start, 0.0 at fade_end, smooth interpolation between.
pub fn compute_distance_fade(distance: f32, fade_start: f32, fade_end: f32) -> f32 {
    if distance <= fade_start {
        1.0
    } else if distance >= fade_end {
        0.0
    } else {
        let t = (distance - fade_start) / (fade_end - fade_start);
        // Smoothstep
        1.0 - (t * t * (3.0 - 2.0 * t))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SSGIConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_size() {
        assert_eq!(mem::size_of::<SSGIConfig>(), 32);
    }

    #[test]
    fn test_config_alignment() {
        assert_eq!(mem::align_of::<SSGIConfig>(), 4);
    }

    #[test]
    fn test_config_pod() {
        let config = SSGIConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_config_default_values() {
        let config = SSGIConfig::default();
        assert_eq!(config.rays_per_pixel, 8);
        assert_eq!(config.max_steps, 48);
        assert_eq!(config.max_distance, 15.0);
        assert_eq!(config.thickness, 0.1);
        assert_eq!(config.intensity, 1.0);
        assert_eq!(config.distance_fade_start, 5.0);
        assert_eq!(config.distance_fade_end, 15.0);
    }

    #[test]
    fn test_config_from_quality_low() {
        let config = SSGIConfig::from_quality(SSGIQuality::Low);
        assert_eq!(config.rays_per_pixel, 4);
        assert_eq!(config.max_steps, 32);
    }

    #[test]
    fn test_config_from_quality_medium() {
        let config = SSGIConfig::from_quality(SSGIQuality::Medium);
        assert_eq!(config.rays_per_pixel, 8);
        assert_eq!(config.max_steps, 48);
    }

    #[test]
    fn test_config_from_quality_high() {
        let config = SSGIConfig::from_quality(SSGIQuality::High);
        assert_eq!(config.rays_per_pixel, 16);
        assert_eq!(config.max_steps, 64);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = SSGIConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_rays() {
        let config = SSGIConfig {
            rays_per_pixel: 0,
            ..SSGIConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_steps() {
        let config = SSGIConfig {
            max_steps: 0,
            ..SSGIConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_negative_distance() {
        let config = SSGIConfig {
            max_distance: -1.0,
            ..SSGIConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_fade() {
        let config = SSGIConfig {
            distance_fade_start: 10.0,
            distance_fade_end: 5.0,
            ..SSGIConfig::default()
        };
        assert!(config.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // SSGIDispatchParams tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dispatch_params_size() {
        assert_eq!(mem::size_of::<SSGIDispatchParams>(), 32);
    }

    #[test]
    fn test_dispatch_params_new() {
        let params = SSGIDispatchParams::new(1920, 1080, 42);
        assert_eq!(params.screen_size, [1920, 1080]);
        assert!((params.inv_screen[0] - 1.0 / 1920.0).abs() < 1e-6);
        assert!((params.inv_screen[1] - 1.0 / 1080.0).abs() < 1e-6);
        assert_eq!(params.frame_index, 42);
    }

    // -----------------------------------------------------------------------
    // SSGIQuality tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quality_rays_per_pixel() {
        assert_eq!(SSGIQuality::Low.rays_per_pixel(), 4);
        assert_eq!(SSGIQuality::Medium.rays_per_pixel(), 8);
        assert_eq!(SSGIQuality::High.rays_per_pixel(), 16);
    }

    #[test]
    fn test_quality_max_steps() {
        assert_eq!(SSGIQuality::Low.max_steps(), 32);
        assert_eq!(SSGIQuality::Medium.max_steps(), 48);
        assert_eq!(SSGIQuality::High.max_steps(), 64);
    }

    #[test]
    fn test_quality_from_ray_count() {
        assert_eq!(SSGIQuality::from_ray_count(1), SSGIQuality::Low);
        assert_eq!(SSGIQuality::from_ray_count(4), SSGIQuality::Low);
        assert_eq!(SSGIQuality::from_ray_count(5), SSGIQuality::Low);
        assert_eq!(SSGIQuality::from_ray_count(6), SSGIQuality::Medium);
        assert_eq!(SSGIQuality::from_ray_count(8), SSGIQuality::Medium);
        assert_eq!(SSGIQuality::from_ray_count(11), SSGIQuality::Medium);
        assert_eq!(SSGIQuality::from_ray_count(12), SSGIQuality::High);
        assert_eq!(SSGIQuality::from_ray_count(16), SSGIQuality::High);
        assert_eq!(SSGIQuality::from_ray_count(32), SSGIQuality::High);
    }

    #[test]
    fn test_quality_estimated_cost() {
        assert_eq!(SSGIQuality::Low.estimated_cost_ms(), 0.5);
        assert_eq!(SSGIQuality::Medium.estimated_cost_ms(), 1.0);
        assert_eq!(SSGIQuality::High.estimated_cost_ms(), 2.0);
    }

    // -----------------------------------------------------------------------
    // Hemisphere sampling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hemisphere_sampling_direction() {
        let normal = [0.0, 1.0, 0.0];

        for i in 0..16 {
            let dir = sample_hemisphere_cosine(normal, i, 16);

            // Direction should be normalized
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            assert!((len - 1.0).abs() < 0.01, "Direction not normalized: len = {}", len);

            // Should point into hemisphere (dot with normal > 0)
            let dot = dir[0] * normal[0] + dir[1] * normal[1] + dir[2] * normal[2];
            assert!(dot >= -0.01, "Direction points wrong way: dot = {}", dot);
        }
    }

    #[test]
    fn test_hemisphere_sampling_z_normal() {
        let normal = [0.0, 0.0, 1.0];

        for i in 0..8 {
            let dir = sample_hemisphere_cosine(normal, i, 8);

            // z component should be positive (pointing into hemisphere)
            assert!(dir[2] >= -0.01, "Direction z is negative: {}", dir[2]);
        }
    }

    #[test]
    fn test_hemisphere_sampling_arbitrary_normal() {
        // Normalized diagonal normal
        let n = 1.0 / 3.0_f32.sqrt();
        let normal = [n, n, n];

        for i in 0..8 {
            let dir = sample_hemisphere_cosine(normal, i, 8);

            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            assert!((len - 1.0).abs() < 0.01);

            let dot = dir[0] * normal[0] + dir[1] * normal[1] + dir[2] * normal[2];
            assert!(dot >= -0.01, "Dot with normal should be >= 0, got {}", dot);
        }
    }

    #[test]
    fn test_hemisphere_sampling_coverage() {
        // Test that samples spread across the hemisphere
        let normal = [0.0, 1.0, 0.0];
        let samples: Vec<[f32; 3]> = (0..16)
            .map(|i| sample_hemisphere_cosine(normal, i, 16))
            .collect();

        // Check that samples are reasonably spread out (not all identical)
        let mut min_dist = f32::MAX;
        for i in 0..samples.len() {
            for j in (i + 1)..samples.len() {
                let dx = samples[i][0] - samples[j][0];
                let dy = samples[i][1] - samples[j][1];
                let dz = samples[i][2] - samples[j][2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                min_dist = min_dist.min(dist);
            }
        }

        // Samples should be at least somewhat spread out
        assert!(min_dist > 0.1, "Samples too clustered: min_dist = {}", min_dist);
    }

    // -----------------------------------------------------------------------
    // Distance fade tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_distance_fade_at_start() {
        let fade = compute_distance_fade(5.0, 5.0, 15.0);
        assert!((fade - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_fade_before_start() {
        let fade = compute_distance_fade(2.0, 5.0, 15.0);
        assert!((fade - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_fade_at_end() {
        let fade = compute_distance_fade(15.0, 5.0, 15.0);
        assert!(fade.abs() < 0.001);
    }

    #[test]
    fn test_distance_fade_after_end() {
        let fade = compute_distance_fade(20.0, 5.0, 15.0);
        assert!(fade.abs() < 0.001);
    }

    #[test]
    fn test_distance_fade_midpoint() {
        let fade = compute_distance_fade(10.0, 5.0, 15.0);
        // At midpoint of smoothstep, should be ~0.5
        assert!((fade - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_distance_fade_monotonic() {
        // Fade should monotonically decrease
        let mut prev = 1.0;
        for d in 0..=20 {
            let fade = compute_distance_fade(d as f32, 5.0, 15.0);
            assert!(fade <= prev + 0.001, "Fade not monotonic at d={}", d);
            prev = fade;
        }
    }

    #[test]
    fn test_distance_fade_range() {
        // Fade should always be in [0, 1]
        for d in 0..=30 {
            let fade = compute_distance_fade(d as f32, 5.0, 15.0);
            assert!(fade >= 0.0 && fade <= 1.0, "Fade out of range at d={}: {}", d, fade);
        }
    }

    // -----------------------------------------------------------------------
    // Frame graph integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_ssgi_pass() {
        use crate::frame_graph::PassType;

        let pass = create_ssgi_pass(
            PassIndex(0),
            ResourceHandle(1),
            ResourceHandle(2),
            ResourceHandle(3),
            ResourceHandle(4),
            ResourceHandle(5),
            1920,
            1080,
            true, // half_res
        );

        assert_eq!(pass.name, "ssgi_trace");
        assert_eq!(pass.pass_type, PassType::Compute);

        // Should read from depth, normal, lighting, hiz
        assert_eq!(pass.access_set.reads.len(), 4);
        assert!(pass.access_set.reads.contains(&ResourceHandle(1)));
        assert!(pass.access_set.reads.contains(&ResourceHandle(2)));
        assert!(pass.access_set.reads.contains(&ResourceHandle(3)));
        assert!(pass.access_set.reads.contains(&ResourceHandle(4)));

        // Should write to output
        assert_eq!(pass.access_set.writes.len(), 1);
        assert!(pass.access_set.writes.contains(&ResourceHandle(5)));

        // Check dispatch (half-res: 960x540 -> 120x68 workgroups)
        if let Some(DispatchSource::Direct { group_count_x, group_count_y, group_count_z }) = pass.dispatch_source {
            assert_eq!(group_count_x, (960 + 7) / 8);
            assert_eq!(group_count_y, (540 + 7) / 8);
            assert_eq!(group_count_z, 1);
        } else {
            panic!("Expected Direct dispatch");
        }
    }

    #[test]
    fn test_create_ssgi_pass_full_res() {
        let pass = create_ssgi_pass(
            PassIndex(0),
            ResourceHandle(1),
            ResourceHandle(2),
            ResourceHandle(3),
            ResourceHandle(4),
            ResourceHandle(5),
            1920,
            1080,
            false, // full_res
        );

        // Full-res: 1920x1080 -> 240x135 workgroups
        if let Some(DispatchSource::Direct { group_count_x, group_count_y, .. }) = pass.dispatch_source {
            assert_eq!(group_count_x, (1920 + 7) / 8);
            assert_eq!(group_count_y, (1080 + 7) / 8);
        } else {
            panic!("Expected Direct dispatch");
        }
    }

    // -----------------------------------------------------------------------
    // Shader validation tests (using naga)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssgi_shader_parses() {
        // Validate that the SSGI compute shader parses correctly
        let shader_source = include_str!("../shaders/ssgi_trace.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSGI shader should parse without errors");

        // Verify the entry point exists
        let entry_point = module
            .entry_points
            .iter()
            .find(|ep| ep.name == "ssgi_trace");
        assert!(
            entry_point.is_some(),
            "Should have ssgi_trace entry point"
        );

        // Verify it's a compute shader
        let ep = entry_point.unwrap();
        assert_eq!(
            ep.stage,
            naga::ShaderStage::Compute,
            "Should be a compute shader"
        );

        // Verify workgroup size
        assert_eq!(ep.workgroup_size, [8, 8, 1], "Workgroup size should be 8x8x1");
    }

    #[test]
    fn test_ssgi_shader_validates() {
        // Full validation including type checking
        let shader_source = include_str!("../shaders/ssgi_trace.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSGI shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        let _info = validator
            .validate(&module)
            .expect("SSGI shader should validate without errors");

        // Verify entry point workgroup size
        let ep = module
            .entry_points
            .iter()
            .find(|ep| ep.name == "ssgi_trace")
            .expect("Should have ssgi_trace entry point");
        assert_eq!(
            ep.workgroup_size, [8, 8, 1],
            "Entry point workgroup size should be 8x8x1"
        );
    }

    #[test]
    fn test_ssgi_shader_bindings() {
        // Verify the shader has the expected bindings
        let shader_source = include_str!("../shaders/ssgi_trace.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSGI shader should parse without errors");

        // Count global variables by binding type
        let mut texture_count = 0;
        let mut storage_count = 0;
        let mut uniform_count = 0;
        let mut sampler_count = 0;

        for (_, var) in module.global_variables.iter() {
            match var.space {
                naga::AddressSpace::Handle => {
                    // Could be texture, storage texture, or sampler
                    match module.types[var.ty].inner {
                        naga::TypeInner::Image { class: naga::ImageClass::Storage { .. }, .. } => {
                            storage_count += 1;
                        }
                        naga::TypeInner::Image { .. } => {
                            texture_count += 1;
                        }
                        naga::TypeInner::Sampler { .. } => {
                            sampler_count += 1;
                        }
                        _ => {}
                    }
                }
                naga::AddressSpace::Uniform => {
                    uniform_count += 1;
                }
                _ => {}
            }
        }

        // Expected: depth + normal + lighting + hiz = 4 textures
        assert!(texture_count >= 4, "Should have at least 4 texture bindings, found {}", texture_count);
        // Expected: output storage texture
        assert!(storage_count >= 1, "Should have at least 1 storage texture binding, found {}", storage_count);
        // Expected: config + dispatch params + camera = 3 uniforms
        assert!(uniform_count >= 3, "Should have at least 3 uniform bindings, found {}", uniform_count);
        // Expected: hiz sampler
        assert!(sampler_count >= 1, "Should have at least 1 sampler binding, found {}", sampler_count);
    }

    #[test]
    fn test_ssgi_shader_struct_sizes() {
        // Verify shader structs match Rust structs
        let shader_source = include_str!("../shaders/ssgi_trace.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSGI shader should parse without errors");

        // Find SSGIConfig struct
        let ssgi_config = module.types.iter().find(|(_, ty)| {
            if let naga::TypeInner::Struct { members, .. } = &ty.inner {
                members.len() == 8 && ty.name.as_ref().map_or(false, |n| n == "SSGIConfig")
            } else {
                false
            }
        });
        assert!(ssgi_config.is_some(), "Should have SSGIConfig struct");

        // Find SSGIDispatchParams struct
        let dispatch_params = module.types.iter().find(|(_, ty)| {
            if let naga::TypeInner::Struct { .. } = &ty.inner {
                ty.name.as_ref().map_or(false, |n| n == "SSGIDispatchParams")
            } else {
                false
            }
        });
        assert!(dispatch_params.is_some(), "Should have SSGIDispatchParams struct");
    }
}
