//! Contact shadow compute dispatch for screen-space shadow rays (T-LIT-8.3).
//!
//! This module provides GPU dispatch for contact shadows, which enhance shadow
//! quality by ray-marching in screen space to detect small-scale occlusions
//! that traditional shadow maps miss.
//!
//! # Overview
//!
//! Contact shadows trace rays from each pixel toward the light direction in
//! screen space. The ray-march samples the depth buffer to detect intersections,
//! producing soft shadow contacts for fine details like grass, hair, and mesh
//! self-shadowing.
//!
//! # Quality Tiers
//!
//! | Tier   | Steps | Use Case                           |
//! |--------|-------|------------------------------------|
//! | Low    | 8     | Mobile, low-end GPUs               |
//! | Medium | 16    | Balanced quality/performance       |
//! | High   | 32    | Desktop, quality-focused           |
//! | Ultra  | 64    | High-end GPUs, maximum quality     |
//!
//! # Usage
//!
//! ```ignore
//! let pass = ContactShadowPass::new(&device, ContactShadowQuality::High);
//!
//! // Create bind group with frame resources
//! let bind_group = pass.create_bind_group(
//!     &device,
//!     &depth_view,
//!     &normal_view,
//!     &camera_buffer,
//!     &light_buffer,
//!     &output_view,
//! );
//!
//! // Dispatch during frame recording
//! pass.dispatch(&mut encoder, &bind_group, width, height);
//! ```
//!
//! # Frame Graph Integration
//!
//! The [`ContactShadowNode`] integrates with the frame graph system, declaring
//! resource dependencies for automatic barrier insertion and resource aliasing.

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (8x8 threads).
pub const WORKGROUP_SIZE: u32 = 8;

/// Default maximum ray distance in world units.
pub const DEFAULT_MAX_DISTANCE: f32 = 0.5;

/// Default occlusion thickness threshold.
pub const DEFAULT_THICKNESS: f32 = 0.05;

/// Default normal bias to prevent self-shadowing.
pub const DEFAULT_NORMAL_BIAS: f32 = 0.01;

// ---------------------------------------------------------------------------
// Quality Tiers
// ---------------------------------------------------------------------------

/// Contact shadow quality tier controlling ray-march step count.
///
/// Higher step counts produce smoother shadows but increase GPU cost.
/// Each tier doubles the step count from the previous.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContactShadowQuality {
    /// 8 ray-march steps. Suitable for mobile and low-end GPUs.
    Low,
    /// 16 ray-march steps. Balanced quality/performance.
    Medium,
    /// 32 ray-march steps. Good quality for desktop GPUs.
    High,
    /// 64 ray-march steps. Maximum quality for high-end GPUs.
    Ultra,
}

impl ContactShadowQuality {
    /// Get the number of ray-march steps for this quality tier.
    #[inline]
    pub const fn step_count(self) -> u32 {
        match self {
            Self::Low => 8,
            Self::Medium => 16,
            Self::High => 32,
            Self::Ultra => 64,
        }
    }

    /// Create quality tier from step count (rounds to nearest tier).
    pub fn from_step_count(steps: u32) -> Self {
        match steps {
            0..=11 => Self::Low,
            12..=23 => Self::Medium,
            24..=47 => Self::High,
            _ => Self::Ultra,
        }
    }
}

impl Default for ContactShadowQuality {
    fn default() -> Self {
        Self::Medium
    }
}

// ---------------------------------------------------------------------------
// GPU Configuration
// ---------------------------------------------------------------------------

/// GPU-side contact shadow configuration.
///
/// This struct is uploaded to a uniform buffer for the compute shader.
/// Matches the WGSL `ContactShadowConfig` struct layout.
///
/// # Memory Layout
///
/// 16 bytes total, std140/std430 compatible:
///
/// | Offset | Field        | Size    |
/// |--------|--------------|---------|
/// | 0      | step_count   | 4 bytes |
/// | 4      | max_distance | 4 bytes |
/// | 8      | thickness    | 4 bytes |
/// | 12     | normal_bias  | 4 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContactShadowConfigGpu {
    /// Number of ray-march steps (8, 16, 32, or 64).
    pub step_count: u32,
    /// Maximum ray distance in world space (default 0.5).
    pub max_distance: f32,
    /// Occlusion thickness threshold (default 0.05).
    pub thickness: f32,
    /// Normal offset bias to prevent self-shadowing (default 0.01).
    pub normal_bias: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ContactShadowConfigGpu>() == 16);

impl ContactShadowConfigGpu {
    /// Create a new GPU config from quality tier with default parameters.
    pub fn from_quality(quality: ContactShadowQuality) -> Self {
        Self {
            step_count: quality.step_count(),
            max_distance: DEFAULT_MAX_DISTANCE,
            thickness: DEFAULT_THICKNESS,
            normal_bias: DEFAULT_NORMAL_BIAS,
        }
    }

    /// Create a new GPU config with custom parameters.
    pub fn new(
        quality: ContactShadowQuality,
        max_distance: f32,
        thickness: f32,
        normal_bias: f32,
    ) -> Self {
        Self {
            step_count: quality.step_count(),
            max_distance,
            thickness,
            normal_bias,
        }
    }
}

impl Default for ContactShadowConfigGpu {
    fn default() -> Self {
        Self::from_quality(ContactShadowQuality::default())
    }
}

// ---------------------------------------------------------------------------
// ContactShadowPass
// ---------------------------------------------------------------------------

/// Contact shadow compute pass.
///
/// Manages the compute pipeline, bind group layout, and configuration buffer
/// for screen-space contact shadow ray-marching.
///
/// # Bind Group Layout
///
/// | Binding | Type              | Content                          |
/// |---------|-------------------|----------------------------------|
/// | 0       | uniform           | ContactShadowConfigGpu           |
/// | 1       | texture_depth_2d  | Depth buffer                     |
/// | 2       | texture_2d        | Normal buffer (view-space)       |
/// | 3       | uniform           | Camera uniforms                  |
/// | 4       | uniform           | Light direction uniforms         |
/// | 5       | storage_texture   | Output contact shadow texture    |
pub struct ContactShadowPass {
    /// Compute pipeline for contact shadow ray-marching.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for contact shadow resources.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Configuration buffer uploaded to GPU.
    config_buffer: wgpu::Buffer,
    /// Current quality tier.
    quality: ContactShadowQuality,
}

impl ContactShadowPass {
    /// Create a new contact shadow pass with the specified quality.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `quality` - Quality tier controlling ray-march step count.
    pub fn new(device: &wgpu::Device, quality: ContactShadowQuality) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline = Self::create_pipeline(device, &bind_group_layout);
        let config_buffer = Self::create_config_buffer(device, quality);

        Self {
            pipeline,
            bind_group_layout,
            config_buffer,
            quality,
        }
    }

    /// Get the current quality tier.
    #[inline]
    pub fn quality(&self) -> ContactShadowQuality {
        self.quality
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

    /// Update the quality tier and re-upload configuration.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue for buffer uploads.
    /// * `quality` - New quality tier.
    pub fn set_quality(&mut self, queue: &wgpu::Queue, quality: ContactShadowQuality) {
        if self.quality != quality {
            self.quality = quality;
            let config = ContactShadowConfigGpu::from_quality(quality);
            queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&config));
        }
    }

    /// Update the full configuration.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue for buffer uploads.
    /// * `config` - New GPU configuration.
    pub fn set_config(&mut self, queue: &wgpu::Queue, config: &ContactShadowConfigGpu) {
        self.quality = ContactShadowQuality::from_step_count(config.step_count);
        queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(config));
    }

    /// Create the bind group layout for contact shadow resources.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("contact_shadow_bind_group_layout"),
            entries: &[
                // Binding 0: ContactShadowConfigGpu uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            mem::size_of::<ContactShadowConfigGpu>() as u64,
                        ),
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
                // Binding 2: Normal texture (view-space)
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
                // Binding 3: Camera uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Light direction uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Output contact shadow texture (storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create the compute pipeline for contact shadows.
    fn create_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader_source = include_str!("../shaders/contact_shadow.comp.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact_shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("contact_shadow_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("contact_shadow_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Create the configuration buffer with initial values.
    fn create_config_buffer(device: &wgpu::Device, quality: ContactShadowQuality) -> wgpu::Buffer {
        let config = ContactShadowConfigGpu::from_quality(quality);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact_shadow_config"),
            size: mem::size_of::<ContactShadowConfigGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        // Initialize buffer with config data
        buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::bytes_of(&config));
        buffer.unmap();

        buffer
    }

    /// Create a bind group for the contact shadow pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `depth_view` - Depth buffer texture view.
    /// * `normal_view` - View-space normal buffer texture view.
    /// * `camera_buffer` - Camera uniforms buffer (view/proj matrices).
    /// * `light_buffer` - Light direction uniforms buffer.
    /// * `output_view` - Output contact shadow texture view (R8Unorm).
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        camera_buffer: &wgpu::Buffer,
        light_buffer: &wgpu::Buffer,
        output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact_shadow_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.config_buffer.as_entire_binding(),
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
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
            ],
        })
    }

    /// Dispatch the contact shadow compute shader.
    ///
    /// Records compute commands to the encoder. Workgroup counts are
    /// calculated from the output resolution.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record to.
    /// * `bind_group` - Pre-created bind group with all resources.
    /// * `width` - Output texture width in pixels.
    /// * `height` - Output texture height in pixels.
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
            label: Some("contact_shadow_pass"),
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
    /// * `width` - Output texture width in pixels.
    /// * `height` - Output texture height in pixels.
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
// Frame Graph Integration
// ---------------------------------------------------------------------------

/// Resource identifier for frame graph dependency tracking.
///
/// These are symbolic identifiers used by the frame graph to track
/// resource dependencies between passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(pub u32);

impl ResourceId {
    /// Depth buffer resource.
    pub const DEPTH_BUFFER: Self = Self(0);
    /// View-space normal buffer resource.
    pub const NORMAL_BUFFER: Self = Self(1);
    /// Camera uniforms resource.
    pub const CAMERA_UNIFORMS: Self = Self(2);
    /// Light direction uniforms resource.
    pub const LIGHT_DIRECTION: Self = Self(3);
    /// Contact shadow output texture resource.
    pub const CONTACT_SHADOW_TEXTURE: Self = Self(4);
}

/// Frame graph context for pass execution.
///
/// Provides access to frame-specific data and resource bindings.
pub struct FrameGraphContext<'a> {
    /// Current frame index.
    pub frame_index: u64,
    /// Command encoder for recording GPU commands.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The wgpu device.
    pub device: &'a wgpu::Device,
    /// Output resolution width.
    pub width: u32,
    /// Output resolution height.
    pub height: u32,
}

/// Trait for frame graph pass nodes.
///
/// Defines the interface for passes that participate in the frame graph
/// system for automatic resource management and barrier insertion.
pub trait FrameGraphNode {
    /// Execute the pass, recording GPU commands.
    fn execute(&self, ctx: &mut FrameGraphContext, bind_group: &wgpu::BindGroup);

    /// Get the list of resources this pass reads.
    fn reads(&self) -> &[ResourceId];

    /// Get the list of resources this pass writes.
    fn writes(&self) -> &[ResourceId];

    /// Get the pass name for debugging.
    fn name(&self) -> &str;
}

/// Contact shadow frame graph node.
///
/// Wraps [`ContactShadowPass`] for integration with the frame graph system.
/// Declares resource dependencies for automatic barrier scheduling.
pub struct ContactShadowNode {
    /// The underlying contact shadow pass.
    pass: ContactShadowPass,
    /// Cached read dependencies.
    reads: Vec<ResourceId>,
    /// Cached write dependencies.
    writes: Vec<ResourceId>,
}

impl ContactShadowNode {
    /// Create a new contact shadow frame graph node.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `quality` - Quality tier controlling ray-march step count.
    pub fn new(device: &wgpu::Device, quality: ContactShadowQuality) -> Self {
        Self {
            pass: ContactShadowPass::new(device, quality),
            reads: vec![
                ResourceId::DEPTH_BUFFER,
                ResourceId::NORMAL_BUFFER,
                ResourceId::CAMERA_UNIFORMS,
                ResourceId::LIGHT_DIRECTION,
            ],
            writes: vec![ResourceId::CONTACT_SHADOW_TEXTURE],
        }
    }

    /// Get a reference to the underlying pass.
    #[inline]
    pub fn pass(&self) -> &ContactShadowPass {
        &self.pass
    }

    /// Get a mutable reference to the underlying pass.
    #[inline]
    pub fn pass_mut(&mut self) -> &mut ContactShadowPass {
        &mut self.pass
    }
}

impl FrameGraphNode for ContactShadowNode {
    fn execute(&self, ctx: &mut FrameGraphContext, bind_group: &wgpu::BindGroup) {
        self.pass.dispatch(ctx.encoder, bind_group, ctx.width, ctx.height);
    }

    fn reads(&self) -> &[ResourceId] {
        &self.reads
    }

    fn writes(&self) -> &[ResourceId] {
        &self.writes
    }

    fn name(&self) -> &str {
        "ContactShadow"
    }
}

// ---------------------------------------------------------------------------
// Helper: Create Contact Shadow Texture
// ---------------------------------------------------------------------------

/// Create a texture suitable for contact shadow output.
///
/// The texture uses R8Unorm format for efficient storage of the
/// binary/smooth shadow factor.
///
/// # Arguments
///
/// * `device` - The wgpu device.
/// * `width` - Texture width in pixels.
/// * `height` - Texture height in pixels.
///
/// # Returns
///
/// A texture configured for contact shadow storage texture access.
pub fn create_contact_shadow_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("contact_shadow_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

// ---------------------------------------------------------------------------
// Contact Shadow Blending (T-LIT-8.2)
// ---------------------------------------------------------------------------

/// Blend mode for combining contact shadows with shadow map results.
///
/// Contact shadows enhance shadow quality by ray-marching in screen space.
/// The blend mode determines how the contact shadow result is combined with
/// the traditional shadow map result.
///
/// # Blend Modes
///
/// | Mode     | Formula                      | Use Case                    |
/// |----------|------------------------------|-----------------------------|
/// | Min      | min(shadow_map, contact)     | Standard, conservative      |
/// | Multiply | shadow_map * contact         | Softer blending             |
/// | Replace  | contact                      | Debug mode, visualization   |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ContactShadowBlendMode {
    /// Minimum blend: `min(shadow_map, contact)`.
    ///
    /// The standard blend mode. Takes the minimum of shadow map and contact
    /// shadow results, ensuring both contribute to the final shadow. A pixel
    /// is only lit if both the shadow map AND contact shadow say it's lit.
    ///
    /// This is the most conservative approach and prevents light leaks.
    #[default]
    Min,

    /// Multiply blend: `shadow_map * contact`.
    ///
    /// Produces softer shadows by multiplying the two shadow factors.
    /// Results in darker shadows where both methods detect occlusion,
    /// but allows partial light through when only one method sees shadow.
    ///
    /// Useful for artistic effects or when contact shadows are too harsh.
    Multiply,

    /// Replace blend: `contact` only.
    ///
    /// Ignores the shadow map entirely and uses only contact shadows.
    /// Primarily useful for debugging and visualizing contact shadow
    /// coverage independent of the shadow map.
    Replace,
}

impl ContactShadowBlendMode {
    /// Blend shadow map and contact shadow results according to this mode.
    ///
    /// Both input values should be in the range [0, 1] where:
    /// - 0.0 = fully shadowed (no light)
    /// - 1.0 = fully lit (no shadow)
    ///
    /// # Arguments
    ///
    /// * `shadow_map` - Shadow factor from the shadow map (0.0 = shadow, 1.0 = lit)
    /// * `contact` - Shadow factor from contact shadow pass (0.0 = shadow, 1.0 = lit)
    ///
    /// # Returns
    ///
    /// Combined shadow factor in [0, 1].
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::contact_shadow::ContactShadowBlendMode;
    ///
    /// let mode = ContactShadowBlendMode::Min;
    /// let result = mode.blend(0.5, 0.8);
    /// assert_eq!(result, 0.5); // min(0.5, 0.8) = 0.5
    /// ```
    #[inline]
    pub fn blend(self, shadow_map: f32, contact: f32) -> f32 {
        match self {
            Self::Min => shadow_map.min(contact),
            Self::Multiply => shadow_map * contact,
            Self::Replace => contact,
        }
    }

    /// Get the WGSL expression for this blend mode.
    ///
    /// Returns a string template that can be substituted into a shader,
    /// using `{shadow_map}` and `{contact}` as placeholders.
    pub fn wgsl_expression(&self) -> &'static str {
        match self {
            Self::Min => "min({shadow_map}, {contact})",
            Self::Multiply => "{shadow_map} * {contact}",
            Self::Replace => "{contact}",
        }
    }

    /// Shader constant value for use in uniform buffers.
    ///
    /// Used to switch blend modes at runtime without shader recompilation.
    pub fn as_u32(&self) -> u32 {
        match self {
            Self::Min => 0,
            Self::Multiply => 1,
            Self::Replace => 2,
        }
    }

    /// Create blend mode from shader constant value.
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::Min,
            1 => Self::Multiply,
            2 => Self::Replace,
            _ => Self::Min, // Default fallback
        }
    }
}

impl std::fmt::Display for ContactShadowBlendMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Min => write!(f, "Min"),
            Self::Multiply => write!(f, "Multiply"),
            Self::Replace => write!(f, "Replace"),
        }
    }
}

// ---------------------------------------------------------------------------
// Contact Shadow Blending Resources
// ---------------------------------------------------------------------------

/// GPU configuration for contact shadow blending in the lighting pass.
///
/// This struct is uploaded to a uniform buffer and read by the lighting shader
/// to control how contact shadows are combined with shadow maps.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContactShadowBlendConfigGpu {
    /// Blend mode (0 = Min, 1 = Multiply, 2 = Replace).
    pub blend_mode: u32,
    /// Strength/intensity of contact shadows [0, 1].
    /// 0.0 = no contact shadow contribution, 1.0 = full contribution.
    pub intensity: f32,
    /// Fallback value when contact shadow texture is not available.
    /// Typically 1.0 (fully lit) to gracefully degrade.
    pub fallback_value: f32,
    /// Padding for 16-byte alignment.
    pub _pad: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ContactShadowBlendConfigGpu>() == 16);

impl Default for ContactShadowBlendConfigGpu {
    fn default() -> Self {
        Self {
            blend_mode: ContactShadowBlendMode::Min.as_u32(),
            intensity: 1.0,
            fallback_value: 1.0,
            _pad: 0.0,
        }
    }
}

impl ContactShadowBlendConfigGpu {
    /// Create config from blend mode with default intensity.
    pub fn from_mode(mode: ContactShadowBlendMode) -> Self {
        Self {
            blend_mode: mode.as_u32(),
            ..Default::default()
        }
    }

    /// Create config with custom intensity.
    pub fn with_intensity(mode: ContactShadowBlendMode, intensity: f32) -> Self {
        Self {
            blend_mode: mode.as_u32(),
            intensity: intensity.clamp(0.0, 1.0),
            fallback_value: 1.0,
            _pad: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ContactShadowPass Extensions for Lighting Integration
// ---------------------------------------------------------------------------

impl ContactShadowPass {
    /// Returns the bind group entry for the contact shadow output texture.
    ///
    /// This is used to bind the contact shadow texture to the lighting pass
    /// for blending with shadow map results.
    ///
    /// # Arguments
    ///
    /// * `contact_shadow_view` - The texture view of the contact shadow output.
    ///
    /// # Returns
    ///
    /// A `BindingResource` wrapping the texture view.
    #[inline]
    pub fn output_texture_binding<'a>(contact_shadow_view: &'a wgpu::TextureView) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::TextureView(contact_shadow_view)
    }

    /// Create a bind group layout entry for the contact shadow texture in the lighting pass.
    ///
    /// This layout entry should be added to the lighting pass's bind group layout
    /// to enable contact shadow sampling.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding slot for the contact shadow texture.
    ///
    /// # Returns
    ///
    /// A `BindGroupLayoutEntry` for the contact shadow texture.
    pub fn lighting_pass_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    /// Create a bind group layout entry for the contact shadow blend config.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding slot for the blend config uniform.
    ///
    /// # Returns
    ///
    /// A `BindGroupLayoutEntry` for the blend config uniform buffer.
    pub fn blend_config_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(
                    mem::size_of::<ContactShadowBlendConfigGpu>() as u64,
                ),
            },
            count: None,
        }
    }

    /// Create a combined shadow bind group for the lighting pass.
    ///
    /// This bind group contains both the contact shadow texture and blend
    /// configuration for use in the lighting pass shader.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `layout` - The bind group layout (should include entries from
    ///   `lighting_pass_layout_entry` and `blend_config_layout_entry`).
    /// * `contact_shadow_view` - The contact shadow output texture view.
    /// * `blend_config_buffer` - The blend configuration uniform buffer.
    /// * `sampler` - A sampler for the contact shadow texture.
    ///
    /// # Returns
    ///
    /// A bind group ready for use in the lighting pass.
    pub fn create_combined_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        contact_shadow_view: &wgpu::TextureView,
        blend_config_buffer: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact_shadow_combined_bind_group"),
            layout,
            entries: &[
                // Binding 0: Contact shadow texture
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(contact_shadow_view),
                },
                // Binding 1: Sampler
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                // Binding 2: Blend config uniform
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blend_config_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create the bind group layout for contact shadow integration in the lighting pass.
    ///
    /// This layout contains:
    /// - Binding 0: Contact shadow texture
    /// - Binding 1: Sampler
    /// - Binding 2: Blend config uniform
    pub fn create_combined_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("contact_shadow_combined_bind_group_layout"),
            entries: &[
                Self::lighting_pass_layout_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                Self::blend_config_layout_entry(2),
            ],
        })
    }

    /// Create a uniform buffer for contact shadow blend configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `config` - Initial blend configuration.
    ///
    /// # Returns
    ///
    /// A GPU buffer containing the blend configuration.
    pub fn create_blend_config_buffer(
        device: &wgpu::Device,
        config: &ContactShadowBlendConfigGpu,
    ) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact_shadow_blend_config"),
            size: mem::size_of::<ContactShadowBlendConfigGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::bytes_of(config));
        buffer.unmap();

        buffer
    }
}

// ---------------------------------------------------------------------------
// Fallback Texture for Missing Contact Shadows
// ---------------------------------------------------------------------------

/// Create a 1x1 white texture for use when contact shadows are disabled.
///
/// This fallback texture returns 1.0 (fully lit) for all samples, allowing
/// the lighting pass to gracefully degrade when contact shadows are not
/// available.
///
/// # Arguments
///
/// * `device` - The wgpu device.
/// * `queue` - The wgpu queue for uploading texture data.
///
/// # Returns
///
/// A 1x1 R8Unorm texture containing white (1.0).
pub fn create_fallback_contact_shadow_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("contact_shadow_fallback"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Write white (1.0 = 255 in R8Unorm)
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255u8],
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(1),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    texture
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ContactShadowQuality -------------------------------------------------

    #[test]
    fn test_quality_step_counts() {
        assert_eq!(ContactShadowQuality::Low.step_count(), 8);
        assert_eq!(ContactShadowQuality::Medium.step_count(), 16);
        assert_eq!(ContactShadowQuality::High.step_count(), 32);
        assert_eq!(ContactShadowQuality::Ultra.step_count(), 64);
    }

    #[test]
    fn test_quality_from_step_count_low() {
        assert_eq!(ContactShadowQuality::from_step_count(1), ContactShadowQuality::Low);
        assert_eq!(ContactShadowQuality::from_step_count(8), ContactShadowQuality::Low);
        assert_eq!(ContactShadowQuality::from_step_count(11), ContactShadowQuality::Low);
    }

    #[test]
    fn test_quality_from_step_count_medium() {
        assert_eq!(ContactShadowQuality::from_step_count(12), ContactShadowQuality::Medium);
        assert_eq!(ContactShadowQuality::from_step_count(16), ContactShadowQuality::Medium);
        assert_eq!(ContactShadowQuality::from_step_count(23), ContactShadowQuality::Medium);
    }

    #[test]
    fn test_quality_from_step_count_high() {
        assert_eq!(ContactShadowQuality::from_step_count(24), ContactShadowQuality::High);
        assert_eq!(ContactShadowQuality::from_step_count(32), ContactShadowQuality::High);
        assert_eq!(ContactShadowQuality::from_step_count(47), ContactShadowQuality::High);
    }

    #[test]
    fn test_quality_from_step_count_ultra() {
        assert_eq!(ContactShadowQuality::from_step_count(48), ContactShadowQuality::Ultra);
        assert_eq!(ContactShadowQuality::from_step_count(64), ContactShadowQuality::Ultra);
        assert_eq!(ContactShadowQuality::from_step_count(100), ContactShadowQuality::Ultra);
    }

    #[test]
    fn test_quality_default() {
        assert_eq!(ContactShadowQuality::default(), ContactShadowQuality::Medium);
    }

    // -- ContactShadowConfigGpu -----------------------------------------------

    #[test]
    fn test_config_gpu_size() {
        // Must be exactly 16 bytes for GPU alignment
        assert_eq!(mem::size_of::<ContactShadowConfigGpu>(), 16);
    }

    #[test]
    fn test_config_gpu_alignment() {
        // Verify 4-byte field alignment
        assert_eq!(mem::align_of::<ContactShadowConfigGpu>(), 4);
    }

    #[test]
    fn test_config_gpu_from_quality_low() {
        let config = ContactShadowConfigGpu::from_quality(ContactShadowQuality::Low);
        assert_eq!(config.step_count, 8);
        assert_eq!(config.max_distance, DEFAULT_MAX_DISTANCE);
        assert_eq!(config.thickness, DEFAULT_THICKNESS);
        assert_eq!(config.normal_bias, DEFAULT_NORMAL_BIAS);
    }

    #[test]
    fn test_config_gpu_from_quality_medium() {
        let config = ContactShadowConfigGpu::from_quality(ContactShadowQuality::Medium);
        assert_eq!(config.step_count, 16);
    }

    #[test]
    fn test_config_gpu_from_quality_high() {
        let config = ContactShadowConfigGpu::from_quality(ContactShadowQuality::High);
        assert_eq!(config.step_count, 32);
    }

    #[test]
    fn test_config_gpu_from_quality_ultra() {
        let config = ContactShadowConfigGpu::from_quality(ContactShadowQuality::Ultra);
        assert_eq!(config.step_count, 64);
    }

    #[test]
    fn test_config_gpu_custom_params() {
        let config = ContactShadowConfigGpu::new(
            ContactShadowQuality::High,
            1.0,
            0.1,
            0.02,
        );
        assert_eq!(config.step_count, 32);
        assert_eq!(config.max_distance, 1.0);
        assert_eq!(config.thickness, 0.1);
        assert_eq!(config.normal_bias, 0.02);
    }

    #[test]
    fn test_config_gpu_default() {
        let config = ContactShadowConfigGpu::default();
        assert_eq!(config.step_count, 16); // Medium quality default
        assert_eq!(config.max_distance, DEFAULT_MAX_DISTANCE);
    }

    // -- Workgroup Calculations -----------------------------------------------

    #[test]
    fn test_workgroup_counts_exact() {
        // 1920x1080 with 8x8 workgroups
        let (x, y, z) = ContactShadowPass::workgroup_counts(1920, 1080);
        assert_eq!(x, 240); // 1920 / 8 = 240
        assert_eq!(y, 135); // 1080 / 8 = 135
        assert_eq!(z, 1);
    }

    #[test]
    fn test_workgroup_counts_rounded() {
        // Non-divisible resolution
        let (x, y, z) = ContactShadowPass::workgroup_counts(1921, 1081);
        assert_eq!(x, 241); // ceil(1921 / 8) = 241
        assert_eq!(y, 136); // ceil(1081 / 8) = 136
        assert_eq!(z, 1);
    }

    #[test]
    fn test_workgroup_counts_small() {
        let (x, y, z) = ContactShadowPass::workgroup_counts(8, 8);
        assert_eq!(x, 1);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_workgroup_counts_minimum() {
        // Even 1x1 should dispatch at least 1 workgroup
        let (x, y, z) = ContactShadowPass::workgroup_counts(1, 1);
        assert_eq!(x, 1);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    // -- ResourceId -----------------------------------------------------------

    #[test]
    fn test_resource_id_constants() {
        // Verify resource IDs are distinct
        assert_ne!(ResourceId::DEPTH_BUFFER, ResourceId::NORMAL_BUFFER);
        assert_ne!(ResourceId::NORMAL_BUFFER, ResourceId::CAMERA_UNIFORMS);
        assert_ne!(ResourceId::CAMERA_UNIFORMS, ResourceId::LIGHT_DIRECTION);
        assert_ne!(ResourceId::LIGHT_DIRECTION, ResourceId::CONTACT_SHADOW_TEXTURE);
    }

    #[test]
    fn test_resource_id_equality() {
        let id1 = ResourceId(42);
        let id2 = ResourceId(42);
        let id3 = ResourceId(43);
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    // -- ContactShadowNode ----------------------------------------------------

    #[test]
    fn test_node_reads_dependencies() {
        // We can't create a real node without GPU, but we can verify the constants
        let expected_reads = [
            ResourceId::DEPTH_BUFFER,
            ResourceId::NORMAL_BUFFER,
            ResourceId::CAMERA_UNIFORMS,
            ResourceId::LIGHT_DIRECTION,
        ];

        // Verify all expected resources are defined
        for resource in &expected_reads {
            assert!(resource.0 <= 4, "Resource ID should be in valid range");
        }
    }

    #[test]
    fn test_node_writes_dependencies() {
        let expected_writes = [ResourceId::CONTACT_SHADOW_TEXTURE];

        // Verify write resource is defined
        assert_eq!(expected_writes.len(), 1);
        assert_eq!(expected_writes[0], ResourceId::CONTACT_SHADOW_TEXTURE);
    }

    // -- Constants ------------------------------------------------------------

    #[test]
    fn test_workgroup_size() {
        assert_eq!(WORKGROUP_SIZE, 8);
    }

    #[test]
    fn test_default_constants() {
        assert_eq!(DEFAULT_MAX_DISTANCE, 0.5);
        assert_eq!(DEFAULT_THICKNESS, 0.05);
        assert_eq!(DEFAULT_NORMAL_BIAS, 0.01);
    }

    // -- ContactShadowBlendMode -----------------------------------------------

    #[test]
    fn test_blend_mode_min_produces_correct_result() {
        let mode = ContactShadowBlendMode::Min;

        // shadow_map=0.5, contact=0.8 -> 0.5 (takes the darker one)
        assert_eq!(mode.blend(0.5, 0.8), 0.5);

        // shadow_map=0.8, contact=0.5 -> 0.5
        assert_eq!(mode.blend(0.8, 0.5), 0.5);

        // Both same value
        assert_eq!(mode.blend(0.7, 0.7), 0.7);

        // Edge cases
        assert_eq!(mode.blend(0.0, 1.0), 0.0);
        assert_eq!(mode.blend(1.0, 0.0), 0.0);
        assert_eq!(mode.blend(1.0, 1.0), 1.0);
        assert_eq!(mode.blend(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_blend_mode_multiply_produces_correct_result() {
        let mode = ContactShadowBlendMode::Multiply;

        // 0.5 * 0.8 = 0.4
        let result = mode.blend(0.5, 0.8);
        assert!((result - 0.4).abs() < 0.0001);

        // 0.8 * 0.5 = 0.4
        let result2 = mode.blend(0.8, 0.5);
        assert!((result2 - 0.4).abs() < 0.0001);

        // Edge cases
        assert_eq!(mode.blend(0.0, 1.0), 0.0);
        assert_eq!(mode.blend(1.0, 0.0), 0.0);
        assert_eq!(mode.blend(1.0, 1.0), 1.0);

        // 0.5 * 0.5 = 0.25
        let result3 = mode.blend(0.5, 0.5);
        assert!((result3 - 0.25).abs() < 0.0001);
    }

    #[test]
    fn test_blend_mode_replace_ignores_shadow_map() {
        let mode = ContactShadowBlendMode::Replace;

        // Should always return contact shadow value, ignoring shadow_map
        assert_eq!(mode.blend(0.5, 0.8), 0.8);
        assert_eq!(mode.blend(0.0, 0.8), 0.8);
        assert_eq!(mode.blend(1.0, 0.8), 0.8);
        assert_eq!(mode.blend(0.3, 0.0), 0.0);
        assert_eq!(mode.blend(0.7, 1.0), 1.0);
    }

    #[test]
    fn test_blend_mode_default_is_min() {
        assert_eq!(ContactShadowBlendMode::default(), ContactShadowBlendMode::Min);
    }

    #[test]
    fn test_blend_mode_u32_roundtrip() {
        for mode in [
            ContactShadowBlendMode::Min,
            ContactShadowBlendMode::Multiply,
            ContactShadowBlendMode::Replace,
        ] {
            let value = mode.as_u32();
            let restored = ContactShadowBlendMode::from_u32(value);
            assert_eq!(mode, restored);
        }
    }

    #[test]
    fn test_blend_mode_from_u32_fallback() {
        // Invalid values should fall back to Min
        assert_eq!(ContactShadowBlendMode::from_u32(3), ContactShadowBlendMode::Min);
        assert_eq!(ContactShadowBlendMode::from_u32(100), ContactShadowBlendMode::Min);
        assert_eq!(ContactShadowBlendMode::from_u32(u32::MAX), ContactShadowBlendMode::Min);
    }

    #[test]
    fn test_blend_mode_display() {
        assert_eq!(format!("{}", ContactShadowBlendMode::Min), "Min");
        assert_eq!(format!("{}", ContactShadowBlendMode::Multiply), "Multiply");
        assert_eq!(format!("{}", ContactShadowBlendMode::Replace), "Replace");
    }

    #[test]
    fn test_blend_mode_wgsl_expression() {
        assert_eq!(
            ContactShadowBlendMode::Min.wgsl_expression(),
            "min({shadow_map}, {contact})"
        );
        assert_eq!(
            ContactShadowBlendMode::Multiply.wgsl_expression(),
            "{shadow_map} * {contact}"
        );
        assert_eq!(
            ContactShadowBlendMode::Replace.wgsl_expression(),
            "{contact}"
        );
    }

    // -- ContactShadowBlendConfigGpu ------------------------------------------

    #[test]
    fn test_blend_config_gpu_size() {
        // Must be 16 bytes for GPU alignment
        assert_eq!(mem::size_of::<ContactShadowBlendConfigGpu>(), 16);
    }

    #[test]
    fn test_blend_config_gpu_default() {
        let config = ContactShadowBlendConfigGpu::default();
        assert_eq!(config.blend_mode, ContactShadowBlendMode::Min.as_u32());
        assert_eq!(config.intensity, 1.0);
        assert_eq!(config.fallback_value, 1.0);
    }

    #[test]
    fn test_blend_config_gpu_from_mode() {
        let config = ContactShadowBlendConfigGpu::from_mode(ContactShadowBlendMode::Multiply);
        assert_eq!(config.blend_mode, ContactShadowBlendMode::Multiply.as_u32());
        assert_eq!(config.intensity, 1.0);
    }

    #[test]
    fn test_blend_config_gpu_with_intensity() {
        let config = ContactShadowBlendConfigGpu::with_intensity(
            ContactShadowBlendMode::Replace,
            0.5,
        );
        assert_eq!(config.blend_mode, ContactShadowBlendMode::Replace.as_u32());
        assert_eq!(config.intensity, 0.5);
    }

    #[test]
    fn test_blend_config_intensity_clamping() {
        // Test clamping below 0
        let config1 = ContactShadowBlendConfigGpu::with_intensity(ContactShadowBlendMode::Min, -0.5);
        assert_eq!(config1.intensity, 0.0);

        // Test clamping above 1
        let config2 = ContactShadowBlendConfigGpu::with_intensity(ContactShadowBlendMode::Min, 1.5);
        assert_eq!(config2.intensity, 1.0);

        // Test valid range
        let config3 = ContactShadowBlendConfigGpu::with_intensity(ContactShadowBlendMode::Min, 0.75);
        assert_eq!(config3.intensity, 0.75);
    }

    // -- GPU Tests (require wgpu) ---------------------------------------------

    #[cfg(feature = "gpu_tests")]
    mod gpu_tests {
        use super::*;

        fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))?;

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor::default(),
                None,
            ))
            .ok()?;

            Some((device, queue))
        }

        #[test]
        fn test_pass_creation() {
            let Some((device, _)) = create_test_device() else {
                return;
            };

            let pass = ContactShadowPass::new(&device, ContactShadowQuality::High);
            assert_eq!(pass.quality(), ContactShadowQuality::High);
        }

        #[test]
        fn test_quality_update() {
            let Some((device, queue)) = create_test_device() else {
                return;
            };

            let mut pass = ContactShadowPass::new(&device, ContactShadowQuality::Low);
            assert_eq!(pass.quality(), ContactShadowQuality::Low);

            pass.set_quality(&queue, ContactShadowQuality::Ultra);
            assert_eq!(pass.quality(), ContactShadowQuality::Ultra);
        }

        #[test]
        fn test_node_creation() {
            let Some((device, _)) = create_test_device() else {
                return;
            };

            let node = ContactShadowNode::new(&device, ContactShadowQuality::Medium);
            assert_eq!(node.pass().quality(), ContactShadowQuality::Medium);
            assert_eq!(node.name(), "ContactShadow");
            assert_eq!(node.reads().len(), 4);
            assert_eq!(node.writes().len(), 1);
        }

        #[test]
        fn test_contact_shadow_texture_creation() {
            let Some((device, _)) = create_test_device() else {
                return;
            };

            let texture = create_contact_shadow_texture(&device, 1920, 1080);
            // Texture creation succeeded without panic
            drop(texture);
        }

        #[test]
        fn test_bind_group_layout_entries() {
            let Some((device, _)) = create_test_device() else {
                return;
            };

            let pass = ContactShadowPass::new(&device, ContactShadowQuality::Medium);
            let _layout = pass.bind_group_layout();
            // Layout creation succeeded (validated by wgpu)
        }

        #[test]
        fn test_combined_bind_group_layout_creation() {
            let Some((device, _)) = create_test_device() else {
                return;
            };

            let layout = ContactShadowPass::create_combined_bind_group_layout(&device);
            // Layout creation succeeded without panic
            drop(layout);
        }

        #[test]
        fn test_blend_config_buffer_creation() {
            let Some((device, _)) = create_test_device() else {
                return;
            };

            let config = ContactShadowBlendConfigGpu::default();
            let buffer = ContactShadowPass::create_blend_config_buffer(&device, &config);
            // Buffer creation succeeded
            drop(buffer);
        }

        #[test]
        fn test_fallback_texture_creation() {
            let Some((device, queue)) = create_test_device() else {
                return;
            };

            let texture = create_fallback_contact_shadow_texture(&device, &queue);
            // Texture creation succeeded
            drop(texture);
        }

        #[test]
        fn test_combined_bind_group_creation() {
            let Some((device, queue)) = create_test_device() else {
                return;
            };

            // Create resources
            let contact_texture = create_contact_shadow_texture(&device, 64, 64);
            let contact_view = contact_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let config = ContactShadowBlendConfigGpu::default();
            let config_buffer = ContactShadowPass::create_blend_config_buffer(&device, &config);

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("contact_shadow_sampler"),
                ..Default::default()
            });

            let layout = ContactShadowPass::create_combined_bind_group_layout(&device);

            // Create bind group
            let bind_group = ContactShadowPass::create_combined_bind_group(
                &device,
                &layout,
                &contact_view,
                &config_buffer,
                &sampler,
            );

            // Bind group creation succeeded
            drop(bind_group);
        }

        #[test]
        fn test_lighting_pass_layout_entry() {
            let Some((device, _)) = create_test_device() else {
                return;
            };

            // Create a layout with the contact shadow entry
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("test_layout"),
                entries: &[ContactShadowPass::lighting_pass_layout_entry(0)],
            });

            // Layout creation succeeded
            drop(layout);
        }

        #[test]
        fn test_blend_config_layout_entry() {
            let Some((device, _)) = create_test_device() else {
                return;
            };

            // Create a layout with the blend config entry
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("test_blend_config_layout"),
                entries: &[ContactShadowPass::blend_config_layout_entry(0)],
            });

            // Layout creation succeeded
            drop(layout);
        }
    }
}
