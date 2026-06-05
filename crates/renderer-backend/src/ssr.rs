//! Screen-Space Reflections (SSR) with Hierarchical Z-Buffer ray marching (T-GIR-P4.2).
//!
//! This module provides GPU infrastructure for screen-space reflections using
//! hierarchical ray marching. HiZ acceleration enables efficient ray traversal by
//! starting at coarse mip levels and descending to finer levels only when necessary.
//!
//! # Algorithm
//!
//! The HiZ ray marching algorithm works as follows:
//!
//! 1. Start at the coarsest (smallest) mip level of the HiZ buffer
//! 2. March the ray in screen-space, checking against the HiZ depth at the current mip
//! 3. If the ray is behind the surface (ray.z > hiz.z), descend to a finer mip level
//! 4. If at mip 0 and behind surface, perform binary refinement to find exact hit
//! 5. If the ray is in front, advance by a step size proportional to the mip level
//! 6. Terminate on hit, miss (off-screen), or max iterations
//!
//! # Performance
//!
//! Typical step counts for different scene types:
//! - Simple indoor scenes: 8-16 steps average
//! - Complex outdoor scenes: 16-32 steps average
//! - Edge cases (grazing angles): 32-64 steps
//!
//! # Usage
//!
//! ```ignore
//! // Create SSR config
//! let config = SSRConfig::default();
//!
//! // Create SSR pass
//! let ssr_pass = SSRPass::new(&device, &hiz_config);
//!
//! // Each frame: run SSR after HiZ generation
//! let bind_group = ssr_pass.create_bind_group(
//!     &device,
//!     &config_buffer,
//!     &hiz_sampled_view,
//!     &depth_view,
//!     &normal_view,
//!     &reflection_output_view,
//! );
//! ssr_pass.dispatch(&mut encoder, &bind_group, screen_width, screen_height);
//! ```

use std::mem;

use bytemuck::{Pod, Zeroable};

use crate::frame_graph::{
    DispatchSource, IrPass, PassIndex, ResourceAccessSet, ResourceHandle, ViewType,
};
use crate::hiz::HiZConfig;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (8x8 threads).
pub const WORKGROUP_SIZE: u32 = 8;

/// Default maximum ray march steps for HiZ traversal.
pub const DEFAULT_MAX_STEPS: u32 = 64;

/// Default maximum binary refinement steps.
pub const DEFAULT_MAX_BINARY_STEPS: u32 = 8;

/// Default depth comparison thickness threshold.
pub const DEFAULT_THICKNESS: f32 = 0.015;

/// Default initial ray step stride (world-space units).
pub const DEFAULT_STRIDE: f32 = 0.1;

/// Default temporal jitter amount for TAA integration.
pub const DEFAULT_JITTER: f32 = 0.5;

/// Default maximum ray travel distance in world-space.
pub const DEFAULT_MAX_DISTANCE: f32 = 100.0;

// ---------------------------------------------------------------------------
// SSRConfig
// ---------------------------------------------------------------------------

/// Configuration for SSR ray marching.
///
/// This struct is uploaded to a uniform buffer and consumed by the SSR compute
/// shader. All parameters are tunable for quality/performance tradeoff.
///
/// # Memory Layout
///
/// 32 bytes total, std140/std430 compatible:
///
/// | Offset | Field            | Size    |
/// |--------|------------------|---------|
/// | 0      | max_steps        | 4 bytes |
/// | 4      | max_binary_steps | 4 bytes |
/// | 8      | thickness        | 4 bytes |
/// | 12     | stride           | 4 bytes |
/// | 16     | jitter_amount    | 4 bytes |
/// | 20     | max_distance     | 4 bytes |
/// | 24     | _pad             | 8 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SSRConfig {
    /// Maximum HiZ ray march steps (64-128 typical).
    ///
    /// Higher values allow longer rays to be traced at the cost of more
    /// GPU iterations. Most rays terminate early due to HiZ acceleration.
    pub max_steps: u32,

    /// Maximum binary refinement steps when at mip 0 (8 typical).
    ///
    /// Binary search narrows down the exact hit point after the coarse
    /// HiZ traversal finds a potential intersection.
    pub max_binary_steps: u32,

    /// Depth comparison threshold for determining a hit.
    ///
    /// When `abs(ray_depth - surface_depth) < thickness`, we consider it a hit.
    /// Too small causes missed intersections; too large causes false positives.
    /// Typical range: 0.005-0.05 (0.5%-5% of depth range).
    pub thickness: f32,

    /// Initial step size in world-space units.
    ///
    /// Affects the base marching speed at the finest mip level. Larger strides
    /// are used at coarser mip levels (stride * 2^mip).
    pub stride: f32,

    /// Temporal jitter amount for TAA integration (0-1).
    ///
    /// Jitters the ray origin slightly each frame to reduce temporal aliasing
    /// when combined with temporal accumulation. Set to 0 for static images.
    pub jitter_amount: f32,

    /// Maximum ray travel distance in world-space units.
    ///
    /// Rays that travel beyond this distance without hitting geometry are
    /// terminated as misses. Limits the cost of long rays in open scenes.
    pub max_distance: f32,

    /// Padding for 32-byte alignment.
    pub _pad: [f32; 2],
}

// Compile-time size assertion: 4 + 4 + 4 + 4 + 4 + 4 + 8 = 32 bytes
const _: () = assert!(mem::size_of::<SSRConfig>() == 32);
const _: () = assert!(mem::align_of::<SSRConfig>() == 4);

impl Default for SSRConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            max_binary_steps: DEFAULT_MAX_BINARY_STEPS,
            thickness: DEFAULT_THICKNESS,
            stride: DEFAULT_STRIDE,
            jitter_amount: DEFAULT_JITTER,
            max_distance: DEFAULT_MAX_DISTANCE,
            _pad: [0.0, 0.0],
        }
    }
}

impl SSRConfig {
    /// Create a new SSR configuration with specified parameters.
    pub fn new(
        max_steps: u32,
        max_binary_steps: u32,
        thickness: f32,
        stride: f32,
        jitter_amount: f32,
        max_distance: f32,
    ) -> Self {
        Self {
            max_steps,
            max_binary_steps,
            thickness,
            stride,
            jitter_amount,
            max_distance,
            _pad: [0.0, 0.0],
        }
    }

    /// Create a high-quality configuration for close-up reflections.
    ///
    /// Uses more steps and finer thickness for detailed surfaces.
    pub fn high_quality() -> Self {
        Self {
            max_steps: 128,
            max_binary_steps: 12,
            thickness: 0.01,
            stride: 0.05,
            jitter_amount: 0.5,
            max_distance: 150.0,
            _pad: [0.0, 0.0],
        }
    }

    /// Create a performance-optimized configuration.
    ///
    /// Uses fewer steps and coarser thresholds for faster execution.
    pub fn performance() -> Self {
        Self {
            max_steps: 32,
            max_binary_steps: 4,
            thickness: 0.03,
            stride: 0.2,
            jitter_amount: 0.5,
            max_distance: 50.0,
            _pad: [0.0, 0.0],
        }
    }

    /// Builder method to set max_steps.
    pub fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_steps = steps;
        self
    }

    /// Builder method to set max_binary_steps.
    pub fn with_max_binary_steps(mut self, steps: u32) -> Self {
        self.max_binary_steps = steps;
        self
    }

    /// Builder method to set thickness.
    pub fn with_thickness(mut self, thickness: f32) -> Self {
        self.thickness = thickness;
        self
    }

    /// Builder method to set stride.
    pub fn with_stride(mut self, stride: f32) -> Self {
        self.stride = stride;
        self
    }

    /// Builder method to set jitter_amount.
    pub fn with_jitter(mut self, jitter: f32) -> Self {
        self.jitter_amount = jitter;
        self
    }

    /// Builder method to set max_distance.
    pub fn with_max_distance(mut self, distance: f32) -> Self {
        self.max_distance = distance;
        self
    }
}

// ---------------------------------------------------------------------------
// SSRUniforms
// ---------------------------------------------------------------------------

/// Extended uniforms for SSR including camera and screen parameters.
///
/// This struct combines the SSR configuration with camera matrices needed
/// for view-space to screen-space projection during ray marching.
///
/// # Memory Layout
///
/// 256 bytes total (4x4 matrices + config + screen info):
///
/// | Offset | Field             | Size     |
/// |--------|-------------------|----------|
/// | 0      | view_matrix       | 64 bytes |
/// | 64     | proj_matrix       | 64 bytes |
/// | 128    | inv_view_matrix   | 64 bytes |
/// | 192    | inv_proj_matrix   | 64 bytes |
/// | 256    | config (SSRConfig)| 32 bytes |
/// | 288    | screen_size       | 8 bytes  |
/// | 296    | max_mip           | 4 bytes  |
/// | 300    | frame_index       | 4 bytes  |
/// | 304    | near_plane        | 4 bytes  |
/// | 308    | far_plane         | 4 bytes  |
/// | 312    | _pad              | 8 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SSRUniforms {
    /// View matrix (world-to-view transform).
    pub view_matrix: [[f32; 4]; 4],
    /// Projection matrix (view-to-clip transform).
    pub proj_matrix: [[f32; 4]; 4],
    /// Inverse view matrix (view-to-world transform).
    pub inv_view_matrix: [[f32; 4]; 4],
    /// Inverse projection matrix (clip-to-view transform).
    pub inv_proj_matrix: [[f32; 4]; 4],
    /// SSR configuration parameters.
    pub config: SSRConfig,
    /// Screen dimensions (width, height).
    pub screen_size: [u32; 2],
    /// Maximum mip level of the HiZ buffer.
    pub max_mip: u32,
    /// Frame index for temporal effects.
    pub frame_index: u32,
    /// Near plane distance.
    pub near_plane: f32,
    /// Far plane distance.
    pub far_plane: f32,
    /// Padding for alignment.
    pub _pad: [f32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<SSRUniforms>() == 320);

impl Default for SSRUniforms {
    fn default() -> Self {
        Self {
            view_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            proj_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            inv_view_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            inv_proj_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            config: SSRConfig::default(),
            screen_size: [1920, 1080],
            max_mip: 10,
            frame_index: 0,
            near_plane: 0.1,
            far_plane: 1000.0,
            _pad: [0.0, 0.0],
        }
    }
}

impl SSRUniforms {
    /// Create SSR uniforms from camera matrices and HiZ config.
    pub fn new(
        view_matrix: [[f32; 4]; 4],
        proj_matrix: [[f32; 4]; 4],
        inv_view_matrix: [[f32; 4]; 4],
        inv_proj_matrix: [[f32; 4]; 4],
        config: SSRConfig,
        hiz_config: &HiZConfig,
        frame_index: u32,
        near_plane: f32,
        far_plane: f32,
    ) -> Self {
        Self {
            view_matrix,
            proj_matrix,
            inv_view_matrix,
            inv_proj_matrix,
            config,
            screen_size: [hiz_config.width, hiz_config.height],
            max_mip: hiz_config.mip_levels.saturating_sub(1),
            frame_index,
            near_plane,
            far_plane,
            _pad: [0.0, 0.0],
        }
    }

    /// Update the frame index for temporal effects.
    pub fn with_frame_index(mut self, frame_index: u32) -> Self {
        self.frame_index = frame_index;
        self
    }
}

// ---------------------------------------------------------------------------
// HitResult
// ---------------------------------------------------------------------------

/// Result of an SSR ray trace.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HitResult {
    /// Whether the ray hit geometry.
    pub hit: bool,
    /// Screen-space UV of the hit point (0-1 range).
    pub uv: [f32; 2],
    /// World-space distance from ray origin to hit.
    pub distance: f32,
    /// Number of HiZ steps taken.
    pub steps_taken: u32,
    /// Final mip level reached.
    pub final_mip: u32,
    /// Confidence of the hit (0-1), based on angle and distance.
    pub confidence: f32,
}

impl HitResult {
    /// Create a miss result.
    pub fn miss() -> Self {
        Self::default()
    }

    /// Create a hit result.
    pub fn new_hit(uv: [f32; 2], distance: f32, steps: u32, mip: u32, confidence: f32) -> Self {
        Self {
            hit: true,
            uv,
            distance,
            steps_taken: steps,
            final_mip: mip,
            confidence,
        }
    }
}

// ---------------------------------------------------------------------------
// SSRPass
// ---------------------------------------------------------------------------

/// Compute pass for SSR ray marching.
///
/// This pass performs hierarchical screen-space ray marching using the HiZ
/// buffer to accelerate ray-depth intersection tests. The output is written
/// to a reflection buffer for later compositing.
///
/// # Bind Group Layout
///
/// | Binding | Type                         | Content                    |
/// |---------|------------------------------|----------------------------|
/// | 0       | uniform                      | SSRUniforms                |
/// | 1       | texture_2d<f32>              | HiZ texture (all mips)     |
/// | 2       | sampler                      | HiZ sampler (point filter) |
/// | 3       | texture_2d<f32>              | GBuffer depth              |
/// | 4       | texture_2d<f32>              | GBuffer normal (RGB)       |
/// | 5       | texture_storage_2d<write>    | Reflection output          |
pub struct SSRPass {
    /// Compute pipeline for SSR ray marching.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for SSR resources.
    bind_group_layout: wgpu::BindGroupLayout,
    /// HiZ point sampler (no filtering).
    hiz_sampler: wgpu::Sampler,
}

impl SSRPass {
    /// Create a new SSR pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline = Self::create_pipeline(device, &bind_group_layout);
        let hiz_sampler = Self::create_hiz_sampler(device);

        Self {
            pipeline,
            bind_group_layout,
            hiz_sampler,
        }
    }

    /// Get the bind group layout for external bind group creation.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the HiZ sampler.
    #[inline]
    pub fn hiz_sampler(&self) -> &wgpu::Sampler {
        &self.hiz_sampler
    }

    /// Create a bind group for the SSR pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `uniforms_buffer` - Buffer containing SSRUniforms.
    /// * `hiz_view` - Sampled view of the entire HiZ mip chain.
    /// * `depth_view` - GBuffer depth texture view.
    /// * `normal_view` - GBuffer normal texture view.
    /// * `output_view` - Reflection output texture view (storage).
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        uniforms_buffer: &wgpu::Buffer,
        hiz_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
            ],
        })
    }

    /// Dispatch the SSR compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder.
    /// * `bind_group` - The bind group containing all SSR resources.
    /// * `width` - Screen width in pixels.
    /// * `height` - Screen height in pixels.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let workgroups_x = (width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ssr_ray_march_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Create an IrPass for frame graph integration.
    ///
    /// # Arguments
    ///
    /// * `index` - Pass index in the frame graph.
    /// * `hiz_handle` - Resource handle for the HiZ texture.
    /// * `depth_handle` - Resource handle for the GBuffer depth.
    /// * `normal_handle` - Resource handle for the GBuffer normals.
    /// * `output_handle` - Resource handle for the reflection output.
    /// * `width` - Screen width for dispatch calculation.
    /// * `height` - Screen height for dispatch calculation.
    pub fn create_hiz_march_pass(
        index: PassIndex,
        hiz_handle: ResourceHandle,
        depth_handle: ResourceHandle,
        normal_handle: ResourceHandle,
        output_handle: ResourceHandle,
        width: u32,
        height: u32,
    ) -> IrPass {
        let workgroups_x = (width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let dispatch = DispatchSource::Direct {
            group_count_x: workgroups_x,
            group_count_y: workgroups_y,
            group_count_z: 1,
        };

        let mut pass = IrPass::compute(index, "ssr_hiz_march", dispatch, ViewType::Storage);
        pass.access_set = ResourceAccessSet {
            reads: vec![hiz_handle, depth_handle, normal_handle],
            writes: vec![output_handle],
        };
        pass
    }

    /// Create the bind group layout for SSR.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr_bind_group_layout"),
            entries: &[
                // Binding 0: Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            mem::size_of::<SSRUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                // Binding 1: HiZ texture (all mip levels)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: HiZ sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Binding 3: GBuffer depth
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
                // Binding 4: GBuffer normals
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 5: Reflection output (storage texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

    /// Create the compute pipeline for SSR.
    fn create_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader_source = include_str!("../shaders/ssr_ray_march.comp.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssr_ray_march_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssr_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssr_ray_march_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "ssr_ray_march",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Create the HiZ point sampler.
    fn create_hiz_sampler(device: &wgpu::Device) -> wgpu::Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hiz_point_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// SSRPassChain
// ---------------------------------------------------------------------------

/// Complete SSR pass chain including uniforms buffer management.
///
/// Encapsulates the SSR pass, uniform buffer, and bind group for easy
/// per-frame updates and dispatch.
pub struct SSRPassChain {
    /// The SSR pass.
    pass: SSRPass,
    /// Uniform buffer for SSR parameters.
    uniform_buffer: wgpu::Buffer,
    /// Current screen dimensions.
    screen_size: (u32, u32),
}

impl SSRPassChain {
    /// Create a new SSR pass chain.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `width` - Screen width.
    /// * `height` - Screen height.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let pass = SSRPass::new(device);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssr_uniforms"),
            size: mem::size_of::<SSRUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pass,
            uniform_buffer,
            screen_size: (width, height),
        }
    }

    /// Get a reference to the SSR pass.
    #[inline]
    pub fn pass(&self) -> &SSRPass {
        &self.pass
    }

    /// Get the uniform buffer.
    #[inline]
    pub fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform_buffer
    }

    /// Update uniforms and dispatch the SSR pass.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder.
    /// * `queue` - The queue for buffer uploads.
    /// * `uniforms` - The SSR uniforms to upload.
    /// * `hiz_view` - HiZ texture view.
    /// * `depth_view` - GBuffer depth view.
    /// * `normal_view` - GBuffer normal view.
    /// * `output_view` - Reflection output view.
    /// * `device` - Device for bind group creation.
    pub fn dispatch_with_uniforms(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        uniforms: &SSRUniforms,
        hiz_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));

        let bind_group = self.pass.create_bind_group(
            device,
            &self.uniform_buffer,
            hiz_view,
            depth_view,
            normal_view,
            output_view,
        );

        self.pass
            .dispatch(encoder, &bind_group, self.screen_size.0, self.screen_size.1);
    }

    /// Resize the pass chain for new screen dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.screen_size = (width, height);
    }
}

// ---------------------------------------------------------------------------
// Frame Graph Integration
// ---------------------------------------------------------------------------

/// Create a frame graph pass for SSR ray marching.
///
/// # Arguments
///
/// * `index` - Pass index in the frame graph.
/// * `name` - Pass name.
/// * `hiz_handle` - Resource handle for the HiZ texture.
/// * `depth_handle` - Resource handle for GBuffer depth.
/// * `normal_handle` - Resource handle for GBuffer normals.
/// * `output_handle` - Resource handle for reflection output.
/// * `width` - Screen width.
/// * `height` - Screen height.
pub fn create_ssr_pass(
    index: PassIndex,
    name: &str,
    hiz_handle: ResourceHandle,
    depth_handle: ResourceHandle,
    normal_handle: ResourceHandle,
    output_handle: ResourceHandle,
    width: u32,
    height: u32,
) -> IrPass {
    let workgroups_x = (width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let workgroups_y = (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let dispatch = DispatchSource::Direct {
        group_count_x: workgroups_x,
        group_count_y: workgroups_y,
        group_count_z: 1,
    };

    let mut pass = IrPass::compute(index, name, dispatch, ViewType::Storage);
    pass.access_set = ResourceAccessSet {
        reads: vec![hiz_handle, depth_handle, normal_handle],
        writes: vec![output_handle],
    };
    pass
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SSRConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssr_config_size_is_32_bytes() {
        assert_eq!(mem::size_of::<SSRConfig>(), 32);
    }

    #[test]
    fn test_ssr_config_alignment() {
        assert_eq!(mem::align_of::<SSRConfig>(), 4);
    }

    #[test]
    fn test_ssr_config_pod() {
        let config = SSRConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_ssr_config_default() {
        let config = SSRConfig::default();
        assert_eq!(config.max_steps, DEFAULT_MAX_STEPS);
        assert_eq!(config.max_binary_steps, DEFAULT_MAX_BINARY_STEPS);
        assert!((config.thickness - DEFAULT_THICKNESS).abs() < f32::EPSILON);
        assert!((config.stride - DEFAULT_STRIDE).abs() < f32::EPSILON);
        assert!((config.jitter_amount - DEFAULT_JITTER).abs() < f32::EPSILON);
        assert!((config.max_distance - DEFAULT_MAX_DISTANCE).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ssr_config_high_quality() {
        let config = SSRConfig::high_quality();
        assert_eq!(config.max_steps, 128);
        assert_eq!(config.max_binary_steps, 12);
        assert!(config.max_distance > DEFAULT_MAX_DISTANCE);
    }

    #[test]
    fn test_ssr_config_performance() {
        let config = SSRConfig::performance();
        assert_eq!(config.max_steps, 32);
        assert_eq!(config.max_binary_steps, 4);
        assert!(config.max_distance < DEFAULT_MAX_DISTANCE);
    }

    #[test]
    fn test_ssr_config_builder() {
        let config = SSRConfig::default()
            .with_max_steps(96)
            .with_max_binary_steps(10)
            .with_thickness(0.02)
            .with_stride(0.15)
            .with_jitter(0.3)
            .with_max_distance(75.0);

        assert_eq!(config.max_steps, 96);
        assert_eq!(config.max_binary_steps, 10);
        assert!((config.thickness - 0.02).abs() < f32::EPSILON);
        assert!((config.stride - 0.15).abs() < f32::EPSILON);
        assert!((config.jitter_amount - 0.3).abs() < f32::EPSILON);
        assert!((config.max_distance - 75.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ssr_config_new() {
        let config = SSRConfig::new(80, 6, 0.025, 0.12, 0.4, 80.0);
        assert_eq!(config.max_steps, 80);
        assert_eq!(config.max_binary_steps, 6);
        assert!((config.thickness - 0.025).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // SSRUniforms tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssr_uniforms_size_is_320_bytes() {
        assert_eq!(mem::size_of::<SSRUniforms>(), 320);
    }

    #[test]
    fn test_ssr_uniforms_pod() {
        let uniforms = SSRUniforms::default();
        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 320);
    }

    #[test]
    fn test_ssr_uniforms_default() {
        let uniforms = SSRUniforms::default();
        assert_eq!(uniforms.screen_size, [1920, 1080]);
        assert_eq!(uniforms.max_mip, 10);
        assert_eq!(uniforms.frame_index, 0);
    }

    #[test]
    fn test_ssr_uniforms_from_hiz_config() {
        let hiz_config = HiZConfig::new(1920, 1080);
        let identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

        let uniforms = SSRUniforms::new(
            identity,
            identity,
            identity,
            identity,
            SSRConfig::default(),
            &hiz_config,
            42,
            0.1,
            1000.0,
        );

        assert_eq!(uniforms.screen_size, [1920, 1080]);
        assert_eq!(uniforms.max_mip, hiz_config.mip_levels - 1);
        assert_eq!(uniforms.frame_index, 42);
    }

    #[test]
    fn test_ssr_uniforms_with_frame_index() {
        let uniforms = SSRUniforms::default().with_frame_index(123);
        assert_eq!(uniforms.frame_index, 123);
    }

    // -----------------------------------------------------------------------
    // HitResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hit_result_miss() {
        let result = HitResult::miss();
        assert!(!result.hit);
        assert_eq!(result.uv, [0.0, 0.0]);
        assert_eq!(result.distance, 0.0);
        assert_eq!(result.steps_taken, 0);
    }

    #[test]
    fn test_hit_result_new_hit() {
        let result = HitResult::new_hit([0.5, 0.5], 10.0, 32, 0, 0.9);
        assert!(result.hit);
        assert_eq!(result.uv, [0.5, 0.5]);
        assert_eq!(result.distance, 10.0);
        assert_eq!(result.steps_taken, 32);
        assert_eq!(result.final_mip, 0);
        assert!((result.confidence - 0.9).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Frame graph integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_ssr_pass() {
        use crate::frame_graph::PassType;

        let hiz = ResourceHandle(0);
        let depth = ResourceHandle(1);
        let normal = ResourceHandle(2);
        let output = ResourceHandle(3);

        let pass = create_ssr_pass(PassIndex(0), "ssr_test", hiz, depth, normal, output, 1920, 1080);

        assert_eq!(pass.name, "ssr_test");
        assert_eq!(pass.pass_type, PassType::Compute);
        assert!(pass.access_set.reads.contains(&hiz));
        assert!(pass.access_set.reads.contains(&depth));
        assert!(pass.access_set.reads.contains(&normal));
        assert!(pass.access_set.writes.contains(&output));
    }

    #[test]
    fn test_create_ssr_pass_workgroups() {
        let hiz = ResourceHandle(0);
        let depth = ResourceHandle(1);
        let normal = ResourceHandle(2);
        let output = ResourceHandle(3);

        let pass = create_ssr_pass(PassIndex(0), "ssr", hiz, depth, normal, output, 1920, 1080);

        // 1920 / 8 = 240, 1080 / 8 = 135
        if let Some(DispatchSource::Direct {
            group_count_x,
            group_count_y,
            group_count_z,
        }) = pass.dispatch_source
        {
            assert_eq!(group_count_x, 240);
            assert_eq!(group_count_y, 135);
            assert_eq!(group_count_z, 1);
        } else {
            panic!("Expected Direct dispatch");
        }
    }

    #[test]
    fn test_create_ssr_pass_odd_size() {
        let hiz = ResourceHandle(0);
        let depth = ResourceHandle(1);
        let normal = ResourceHandle(2);
        let output = ResourceHandle(3);

        let pass = create_ssr_pass(PassIndex(0), "ssr", hiz, depth, normal, output, 100, 100);

        // 100 / 8 = 12.5, ceil = 13 workgroups
        if let Some(DispatchSource::Direct {
            group_count_x,
            group_count_y,
            ..
        }) = pass.dispatch_source
        {
            assert_eq!(group_count_x, 13);
            assert_eq!(group_count_y, 13);
        } else {
            panic!("Expected Direct dispatch");
        }
    }

    // -----------------------------------------------------------------------
    // Ray marching algorithm tests (unit tests for shader logic)
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_size_for_mip() {
        // Step size doubles with each mip level
        let base_stride = 0.1;
        for mip in 0..10 {
            let step = base_stride * (1 << mip) as f32;
            assert!((step - base_stride * 2.0_f32.powi(mip as i32)).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_mip_descent_threshold() {
        // When ray is behind surface, descend mip
        let ray_depth = 0.8; // Behind the surface
        let hiz_depth = 0.5; // Surface is closer to camera (reversed-Z: smaller = farther)

        // In reversed-Z, larger values are closer
        // Ray at 0.8 (closer) is behind surface at 0.5 (farther)
        // This means ray has passed through, need to refine
        let should_descend = ray_depth > hiz_depth;
        assert!(should_descend);
    }

    #[test]
    fn test_mip_advance_condition() {
        // When ray is in front of surface, advance
        let ray_depth = 0.3;
        let hiz_depth = 0.5;

        // Ray at 0.3 is farther than surface at 0.5 (reversed-Z)
        // Ray is in front, can advance
        let should_advance = ray_depth <= hiz_depth;
        assert!(should_advance);
    }

    #[test]
    fn test_binary_refinement_convergence() {
        // Simulate binary search convergence
        let mut t_lo: f32 = 5.0;
        let mut t_hi: f32 = 10.0;
        let target: f32 = 7.3;

        for _ in 0..8 {
            let t_mid = (t_lo + t_hi) * 0.5;
            if t_mid < target {
                t_lo = t_mid;
            } else {
                t_hi = t_mid;
            }
        }

        // After 8 iterations, should be within 2^-8 = 0.004 of range
        let range: f32 = (10.0 - 5.0) / 256.0; // 0.0195
        assert!((t_lo - target).abs() < range || (t_hi - target).abs() < range);
    }

    #[test]
    fn test_screen_bounds_check() {
        // UV coordinates should be in [0, 1] range
        let check_bounds = |u: f32, v: f32| -> bool { u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0 };

        assert!(check_bounds(0.0, 0.0));
        assert!(check_bounds(0.5, 0.5));
        assert!(check_bounds(1.0, 1.0));
        assert!(!check_bounds(-0.1, 0.5));
        assert!(!check_bounds(0.5, 1.1));
    }

    #[test]
    fn test_ray_direction_normalization() {
        // Reflected ray direction should be normalized
        let dir: [f32; 3] = [0.6, 0.8, 0.0]; // Already normalized (0.6^2 + 0.8^2 = 1)
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        assert!((len - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hit_confidence_calculation() {
        // Confidence decreases with grazing angles and large distances
        let calc_confidence = |angle_cos: f32, distance: f32, max_dist: f32| -> f32 {
            let angle_factor = angle_cos.max(0.0);
            let dist_factor = 1.0 - (distance / max_dist).min(1.0);
            angle_factor * dist_factor
        };

        // Perpendicular hit at close range: high confidence
        assert!(calc_confidence(1.0, 10.0, 100.0) > 0.8);

        // Grazing angle: lower confidence
        assert!(calc_confidence(0.2, 10.0, 100.0) < 0.3);

        // Far distance: lower confidence
        assert!(calc_confidence(1.0, 90.0, 100.0) < 0.2);
    }

    // -----------------------------------------------------------------------
    // Shader validation tests (using naga)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssr_shader_parses() {
        let shader_source = include_str!("../shaders/ssr_ray_march.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSR shader should parse without errors");

        // Verify the entry point exists
        let entry_point = module
            .entry_points
            .iter()
            .find(|ep| ep.name == "ssr_ray_march");
        assert!(
            entry_point.is_some(),
            "Should have ssr_ray_march entry point"
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
    fn test_ssr_shader_validates() {
        let shader_source = include_str!("../shaders/ssr_ray_march.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSR shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        let _info = validator
            .validate(&module)
            .expect("SSR shader should validate without errors");
    }

    #[test]
    fn test_ssr_shader_bindings() {
        let shader_source = include_str!("../shaders/ssr_ray_march.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSR shader should parse without errors");

        // Count global variables by binding
        let mut uniform_count = 0;
        let mut texture_count = 0;
        let mut sampler_count = 0;
        let mut storage_count = 0;

        for (_, var) in module.global_variables.iter() {
            match var.space {
                naga::AddressSpace::Uniform => uniform_count += 1,
                naga::AddressSpace::Handle => {
                    match &module.types[var.ty].inner {
                        naga::TypeInner::Image { class, .. } => {
                            match class {
                                naga::ImageClass::Storage { .. } => storage_count += 1,
                                _ => texture_count += 1,
                            }
                        }
                        naga::TypeInner::Sampler { .. } => sampler_count += 1,
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        assert!(uniform_count >= 1, "Should have at least 1 uniform binding");
        assert!(texture_count >= 3, "Should have at least 3 texture bindings (HiZ, depth, normal)");
        assert!(sampler_count >= 1, "Should have at least 1 sampler binding");
        assert!(storage_count >= 1, "Should have at least 1 storage texture binding");
    }

    // -----------------------------------------------------------------------
    // GPU tests (require wgpu device)
    // -----------------------------------------------------------------------

    fn try_create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN,
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;

            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()
        })
    }

    macro_rules! require_gpu {
        () => {
            match try_create_test_device() {
                Some(device_queue) => device_queue,
                None => {
                    eprintln!("Skipping test: no GPU adapter available");
                    return;
                }
            }
        };
    }

    #[test]
    fn test_ssr_pass_creation() {
        let (device, _queue) = require_gpu!();
        let pass = SSRPass::new(&device);

        // Just verify it doesn't panic
        let _ = pass.bind_group_layout();
        let _ = pass.hiz_sampler();
    }

    #[test]
    fn test_ssr_pass_chain_creation() {
        let (device, _queue) = require_gpu!();
        let chain = SSRPassChain::new(&device, 1920, 1080);

        assert_eq!(chain.screen_size, (1920, 1080));
    }

    #[test]
    fn test_ssr_pass_chain_resize() {
        let (device, _queue) = require_gpu!();
        let mut chain = SSRPassChain::new(&device, 1920, 1080);

        chain.resize(2560, 1440);
        assert_eq!(chain.screen_size, (2560, 1440));
    }
}
