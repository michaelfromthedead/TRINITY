//! Multi-Pass SDF Rendering (T-DEMO-6.7)
//!
//! Implements opaque and transparent SDF render passes with correct ordering
//! for integration with raster geometry. The multi-pass system enables:
//!
//! - Opaque SDF pass: Writes depth, alpha test, early-Z
//! - Transparent SDF pass: Reads depth, alpha blending, sorted
//!
//! # Pass Ordering
//!
//! ```text
//! 1. Raster Opaque    - Traditional geometry, depth write
//! 2. SDF Opaque       - Ray-marched solids, depth write
//! 3. SDF Transparent  - Ray-marched glass/fog, depth read only
//! 4. Raster Transparent - Traditional alpha geometry
//! ```
//!
//! # Blend Modes
//!
//! - Opaque: No blending, depth write enabled
//! - Transparent: src_alpha * src + (1 - src_alpha) * dst

use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Workgroup size for multi-pass shaders.
pub const MULTIPASS_WORKGROUP_SIZE: u32 = 8;

/// Maximum number of transparent SDF objects that can be sorted.
pub const MAX_TRANSPARENT_OBJECTS: usize = 256;

/// Default alpha threshold for opaque pass (alpha > threshold = opaque).
pub const OPAQUE_ALPHA_THRESHOLD: f32 = 0.99;

/// Minimum alpha for transparent pass to avoid invisible fragments.
pub const TRANSPARENT_ALPHA_MIN: f32 = 0.001;

// ---------------------------------------------------------------------------
// Blend Mode
// ---------------------------------------------------------------------------

/// Blend mode for SDF rendering passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SdfBlendMode {
    /// No blending, overwrites destination.
    Opaque,
    /// Standard alpha blending: src_alpha * src + (1 - src_alpha) * dst.
    AlphaBlend,
    /// Additive blending: src + dst.
    Additive,
    /// Pre-multiplied alpha: src + (1 - src_alpha) * dst.
    Premultiplied,
}

impl SdfBlendMode {
    /// Convert to wgpu blend state.
    pub fn to_wgpu_blend_state(&self) -> Option<wgpu::BlendState> {
        match self {
            Self::Opaque => None,
            Self::AlphaBlend => Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            Self::Additive => Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            }),
            Self::Premultiplied => Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
        }
    }
}

impl Default for SdfBlendMode {
    fn default() -> Self {
        Self::Opaque
    }
}

// ---------------------------------------------------------------------------
// Depth Mode
// ---------------------------------------------------------------------------

/// Depth buffer configuration for SDF passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SdfDepthMode {
    /// Full depth testing and writing (opaque objects).
    WriteAndTest,
    /// Depth testing only, no writing (transparent objects).
    TestOnly,
    /// No depth operations (overlay effects).
    Disabled,
}

impl SdfDepthMode {
    /// Returns true if depth writing is enabled.
    #[inline]
    pub fn writes_depth(&self) -> bool {
        matches!(self, Self::WriteAndTest)
    }

    /// Returns true if depth testing is enabled.
    #[inline]
    pub fn tests_depth(&self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

impl Default for SdfDepthMode {
    fn default() -> Self {
        Self::WriteAndTest
    }
}

// ---------------------------------------------------------------------------
// Pass Type
// ---------------------------------------------------------------------------

/// SDF render pass type identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SdfPassType {
    /// Opaque SDF geometry pass.
    Opaque,
    /// Transparent SDF geometry pass.
    Transparent,
}

impl SdfPassType {
    /// Get the default blend mode for this pass type.
    pub fn default_blend_mode(&self) -> SdfBlendMode {
        match self {
            Self::Opaque => SdfBlendMode::Opaque,
            Self::Transparent => SdfBlendMode::AlphaBlend,
        }
    }

    /// Get the default depth mode for this pass type.
    pub fn default_depth_mode(&self) -> SdfDepthMode {
        match self {
            Self::Opaque => SdfDepthMode::WriteAndTest,
            Self::Transparent => SdfDepthMode::TestOnly,
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-Pass Uniforms
// ---------------------------------------------------------------------------

/// Uniform buffer data for multi-pass SDF rendering.
///
/// Extended uniforms include pass-specific parameters for blending,
/// depth configuration, and alpha thresholds.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct MultiPassUniforms {
    /// Animation time in seconds.
    pub time: f32,
    /// Output width in pixels.
    pub resolution_x: f32,
    /// Output height in pixels.
    pub resolution_y: f32,
    /// Pass type: 0 = opaque, 1 = transparent.
    pub pass_type: u32,
    /// Alpha threshold for opaque pass.
    pub alpha_threshold: f32,
    /// Minimum alpha for transparent pass.
    pub alpha_min: f32,
    /// Depth bias for transparent objects.
    pub depth_bias: f32,
    /// Padding for vec4 alignment.
    pub _padding: f32,
}

impl Default for MultiPassUniforms {
    fn default() -> Self {
        Self {
            time: 0.0,
            resolution_x: 800.0,
            resolution_y: 600.0,
            pass_type: 0,
            alpha_threshold: OPAQUE_ALPHA_THRESHOLD,
            alpha_min: TRANSPARENT_ALPHA_MIN,
            depth_bias: 0.0,
            _padding: 0.0,
        }
    }
}

impl MultiPassUniforms {
    /// Create uniforms for opaque pass.
    pub fn opaque(width: u32, height: u32, time: f32) -> Self {
        Self {
            time,
            resolution_x: width as f32,
            resolution_y: height as f32,
            pass_type: 0,
            alpha_threshold: OPAQUE_ALPHA_THRESHOLD,
            alpha_min: 0.0,
            depth_bias: 0.0,
            _padding: 0.0,
        }
    }

    /// Create uniforms for transparent pass.
    pub fn transparent(width: u32, height: u32, time: f32) -> Self {
        Self {
            time,
            resolution_x: width as f32,
            resolution_y: height as f32,
            pass_type: 1,
            alpha_threshold: 0.0,
            alpha_min: TRANSPARENT_ALPHA_MIN,
            depth_bias: 0.001,
            _padding: 0.0,
        }
    }

    /// Get data as bytes for GPU upload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

// ---------------------------------------------------------------------------
// Pass Configuration
// ---------------------------------------------------------------------------

/// Configuration for an SDF render pass.
#[derive(Clone, Debug, PartialEq)]
pub struct SdfPassConfig {
    /// Pass type (opaque or transparent).
    pub pass_type: SdfPassType,
    /// Blend mode for output.
    pub blend_mode: SdfBlendMode,
    /// Depth buffer configuration.
    pub depth_mode: SdfDepthMode,
    /// Alpha threshold for discard (opaque only).
    pub alpha_threshold: f32,
    /// Minimum alpha to render (transparent only).
    pub alpha_min: f32,
    /// Enable back-to-front sorting for transparent objects.
    pub sort_transparent: bool,
}

impl SdfPassConfig {
    /// Create configuration for opaque pass.
    pub fn opaque() -> Self {
        Self {
            pass_type: SdfPassType::Opaque,
            blend_mode: SdfBlendMode::Opaque,
            depth_mode: SdfDepthMode::WriteAndTest,
            alpha_threshold: OPAQUE_ALPHA_THRESHOLD,
            alpha_min: 0.0,
            sort_transparent: false,
        }
    }

    /// Create configuration for transparent pass.
    pub fn transparent() -> Self {
        Self {
            pass_type: SdfPassType::Transparent,
            blend_mode: SdfBlendMode::AlphaBlend,
            depth_mode: SdfDepthMode::TestOnly,
            alpha_threshold: 0.0,
            alpha_min: TRANSPARENT_ALPHA_MIN,
            sort_transparent: true,
        }
    }

    /// Create configuration for additive effects.
    pub fn additive() -> Self {
        Self {
            pass_type: SdfPassType::Transparent,
            blend_mode: SdfBlendMode::Additive,
            depth_mode: SdfDepthMode::TestOnly,
            alpha_threshold: 0.0,
            alpha_min: TRANSPARENT_ALPHA_MIN,
            sort_transparent: false,
        }
    }
}

impl Default for SdfPassConfig {
    fn default() -> Self {
        Self::opaque()
    }
}

// ---------------------------------------------------------------------------
// Multi-Pass Renderer
// ---------------------------------------------------------------------------

/// Multi-pass SDF renderer supporting opaque and transparent passes.
///
/// Manages two compute pipelines (opaque + transparent) that write to
/// separate output textures for compositing with raster geometry.
pub struct MultiPassSdfRenderer {
    /// Configuration for opaque pass.
    opaque_config: SdfPassConfig,
    /// Configuration for transparent pass.
    transparent_config: SdfPassConfig,
    /// Uniform buffer for opaque pass.
    opaque_uniform_buffer: wgpu::Buffer,
    /// Uniform buffer for transparent pass.
    transparent_uniform_buffer: wgpu::Buffer,
    /// Opaque pass compute pipeline.
    opaque_pipeline: wgpu::ComputePipeline,
    /// Transparent pass compute pipeline.
    transparent_pipeline: wgpu::ComputePipeline,
    /// Bind group layout (shared).
    bind_group_layout: wgpu::BindGroupLayout,
    /// Output texture for opaque SDF (RGBA8, depth in alpha).
    opaque_output: wgpu::Texture,
    /// Output texture for transparent SDF (RGBA16F for HDR).
    transparent_output: wgpu::Texture,
    /// Depth texture (shared between passes).
    depth_texture: wgpu::Texture,
    /// Opaque output view.
    opaque_output_view: wgpu::TextureView,
    /// Transparent output view.
    transparent_output_view: wgpu::TextureView,
    /// Depth texture view.
    depth_view: wgpu::TextureView,
    /// Bind group for opaque pass.
    opaque_bind_group: wgpu::BindGroup,
    /// Bind group for transparent pass.
    transparent_bind_group: wgpu::BindGroup,
    /// Render dimensions.
    width: u32,
    height: u32,
}

impl MultiPassSdfRenderer {
    /// Create a new multi-pass SDF renderer.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device
    /// * `width` - render target width
    /// * `height` - render target height
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let opaque_config = SdfPassConfig::opaque();
        let transparent_config = SdfPassConfig::transparent();

        // Create shader modules
        let opaque_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Opaque Pass Shader"),
            source: wgpu::ShaderSource::Wgsl(MULTIPASS_OPAQUE_SHADER.into()),
        });

        let transparent_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Transparent Pass Shader"),
            source: wgpu::ShaderSource::Wgsl(MULTIPASS_TRANSPARENT_SHADER.into()),
        });

        // Create uniform buffers
        let opaque_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Opaque Uniform Buffer"),
            size: std::mem::size_of::<MultiPassUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let transparent_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Transparent Uniform Buffer"),
            size: std::mem::size_of::<MultiPassUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output textures
        let opaque_output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SDF Opaque Output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let transparent_output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SDF Transparent Output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SDF Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let opaque_output_view = opaque_output.create_view(&wgpu::TextureViewDescriptor::default());
        let transparent_output_view =
            transparent_output.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Multi-Pass Bind Group Layout"),
            entries: &[
                // Uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<MultiPassUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                // Color output (storage texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Depth texture (storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Separate layout for transparent pass (reads depth, writes HDR)
        let transparent_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SDF Transparent Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<MultiPassUniforms>() as u64,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // Create pipeline layouts
        let opaque_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SDF Opaque Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let transparent_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SDF Transparent Pipeline Layout"),
                bind_group_layouts: &[&transparent_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create compute pipelines
        let opaque_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Opaque Compute Pipeline"),
            layout: Some(&opaque_pipeline_layout),
            module: &opaque_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let transparent_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SDF Transparent Compute Pipeline"),
                layout: Some(&transparent_pipeline_layout),
                module: &transparent_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Create bind groups
        let opaque_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Opaque Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &opaque_uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<MultiPassUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&opaque_output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
            ],
        });

        let transparent_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SDF Transparent Bind Group"),
            layout: &transparent_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &transparent_uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<MultiPassUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&transparent_output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
            ],
        });

        Self {
            opaque_config,
            transparent_config,
            opaque_uniform_buffer,
            transparent_uniform_buffer,
            opaque_pipeline,
            transparent_pipeline,
            bind_group_layout,
            opaque_output,
            transparent_output,
            depth_texture,
            opaque_output_view,
            transparent_output_view,
            depth_view,
            opaque_bind_group,
            transparent_bind_group,
            width,
            height,
        }
    }

    /// Update uniforms and upload to GPU.
    pub fn update(&self, queue: &wgpu::Queue, time: f32) {
        let opaque_uniforms = MultiPassUniforms::opaque(self.width, self.height, time);
        let transparent_uniforms = MultiPassUniforms::transparent(self.width, self.height, time);

        queue.write_buffer(&self.opaque_uniform_buffer, 0, opaque_uniforms.as_bytes());
        queue.write_buffer(
            &self.transparent_uniform_buffer,
            0,
            transparent_uniforms.as_bytes(),
        );
    }

    /// Dispatch the opaque SDF pass.
    pub fn dispatch_opaque(&self, encoder: &mut wgpu::CommandEncoder) {
        let (gx, gy) = self.dispatch_size();

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SDF Opaque Pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.opaque_pipeline);
        cpass.set_bind_group(0, &self.opaque_bind_group, &[]);
        cpass.dispatch_workgroups(gx, gy, 1);
    }

    /// Dispatch the transparent SDF pass.
    pub fn dispatch_transparent(&self, encoder: &mut wgpu::CommandEncoder) {
        let (gx, gy) = self.dispatch_size();

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SDF Transparent Pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.transparent_pipeline);
        cpass.set_bind_group(0, &self.transparent_bind_group, &[]);
        cpass.dispatch_workgroups(gx, gy, 1);
    }

    /// Dispatch both passes in correct order.
    ///
    /// Order: opaque first (writes depth), transparent second (reads depth).
    pub fn dispatch_all(&self, encoder: &mut wgpu::CommandEncoder) {
        self.dispatch_opaque(encoder);
        self.dispatch_transparent(encoder);
    }

    /// Get dispatch dimensions for compute shaders.
    #[inline]
    fn dispatch_size(&self) -> (u32, u32) {
        let gx = (self.width + MULTIPASS_WORKGROUP_SIZE - 1) / MULTIPASS_WORKGROUP_SIZE;
        let gy = (self.height + MULTIPASS_WORKGROUP_SIZE - 1) / MULTIPASS_WORKGROUP_SIZE;
        (gx, gy)
    }

    /// Get the opaque output texture view.
    #[inline]
    pub fn opaque_output_view(&self) -> &wgpu::TextureView {
        &self.opaque_output_view
    }

    /// Get the transparent output texture view.
    #[inline]
    pub fn transparent_output_view(&self) -> &wgpu::TextureView {
        &self.transparent_output_view
    }

    /// Get the depth texture view.
    #[inline]
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Get the opaque output texture.
    #[inline]
    pub fn opaque_output(&self) -> &wgpu::Texture {
        &self.opaque_output
    }

    /// Get the transparent output texture.
    #[inline]
    pub fn transparent_output(&self) -> &wgpu::Texture {
        &self.transparent_output
    }

    /// Get render dimensions.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the opaque pass configuration.
    #[inline]
    pub fn opaque_config(&self) -> &SdfPassConfig {
        &self.opaque_config
    }

    /// Get the transparent pass configuration.
    #[inline]
    pub fn transparent_config(&self) -> &SdfPassConfig {
        &self.transparent_config
    }
}

// ---------------------------------------------------------------------------
// Pass Ordering
// ---------------------------------------------------------------------------

/// Render pass ordering for correct composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderPassOrder {
    /// Raster opaque geometry (depth write).
    RasterOpaque = 0,
    /// SDF opaque geometry (depth write).
    SdfOpaque = 1,
    /// SDF transparent geometry (depth read only).
    SdfTransparent = 2,
    /// Raster transparent geometry (depth read only).
    RasterTransparent = 3,
}

impl RenderPassOrder {
    /// Get all passes in correct rendering order.
    pub fn all_ordered() -> [Self; 4] {
        [
            Self::RasterOpaque,
            Self::SdfOpaque,
            Self::SdfTransparent,
            Self::RasterTransparent,
        ]
    }

    /// Check if this pass writes depth.
    #[inline]
    pub fn writes_depth(&self) -> bool {
        matches!(self, Self::RasterOpaque | Self::SdfOpaque)
    }

    /// Check if this pass is transparent.
    #[inline]
    pub fn is_transparent(&self) -> bool {
        matches!(self, Self::SdfTransparent | Self::RasterTransparent)
    }

    /// Check if this is an SDF pass.
    #[inline]
    pub fn is_sdf(&self) -> bool {
        matches!(self, Self::SdfOpaque | Self::SdfTransparent)
    }
}

// ---------------------------------------------------------------------------
// Embedded Shaders
// ---------------------------------------------------------------------------

/// Embedded WGSL shader for opaque SDF pass.
pub const MULTIPASS_OPAQUE_SHADER: &str = r#"
// SDF Opaque Pass - writes depth, discards transparent fragments

struct MultiPassUniforms {
    time: f32,
    resolution_x: f32,
    resolution_y: f32,
    pass_type: u32,
    alpha_threshold: f32,
    alpha_min: f32,
    depth_bias: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: MultiPassUniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var depth_texture: texture_storage_2d<r32float, read_write>;

// SDF primitives
fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// Scene SDF with material ID
fn scene_sdf(p: vec3<f32>) -> vec2<f32> {
    let t = uniforms.time;

    // Opaque sphere
    let sphere_pos = vec3<f32>(sin(t) * 0.8, 0.0, cos(t) * 0.8);
    let d_sphere = sdf_sphere(p - sphere_pos, 0.4);

    // Opaque box
    let angle = t * 0.5;
    let c = cos(angle);
    let s = sin(angle);
    let rotated_p = vec3<f32>(p.x * c - p.z * s, p.y, p.x * s + p.z * c);
    let d_box = sdf_box(rotated_p, vec3<f32>(0.3, 0.3, 0.3));

    // Ground plane
    let d_ground = p.y + 0.8;

    var d = d_sphere;
    var mat_id = 1.0;

    if (d_box < d) {
        d = d_box;
        mat_id = 2.0;
    }

    if (d_ground < d) {
        d = d_ground;
        mat_id = 0.0;
    }

    return vec2<f32>(d, mat_id);
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.001, 0.0);
    return normalize(vec3<f32>(
        scene_sdf(p + e.xyy).x - scene_sdf(p - e.xyy).x,
        scene_sdf(p + e.yxy).x - scene_sdf(p - e.yxy).x,
        scene_sdf(p + e.yyx).x - scene_sdf(p - e.yyx).x
    ));
}

const MAX_STEPS: i32 = 64;
const MAX_DIST: f32 = 20.0;
const SURF_DIST: f32 = 0.001;

fn ray_march(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    var t = 0.0;
    var mat_id = -1.0;

    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let result = scene_sdf(p);
        let d = result.x;
        mat_id = result.y;

        if (d < SURF_DIST) {
            break;
        }

        t += d;

        if (t > MAX_DIST) {
            mat_id = -1.0;
            break;
        }
    }

    return vec3<f32>(t, mat_id, 1.0); // t, material_id, alpha (opaque = 1.0)
}

fn get_material_color(mat_id: f32, p: vec3<f32>) -> vec3<f32> {
    if (mat_id < 0.5) {
        // Ground - checkerboard
        let checker = floor(p.x * 2.0) + floor(p.z * 2.0);
        if (fract(checker * 0.5) < 0.5) {
            return vec3<f32>(0.4, 0.4, 0.4);
        }
        return vec3<f32>(0.6, 0.6, 0.6);
    } else if (mat_id < 1.5) {
        // Sphere - red
        return vec3<f32>(0.8, 0.2, 0.3);
    } else {
        // Box - blue
        return vec3<f32>(0.3, 0.5, 0.9);
    }
}

fn calc_lighting(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let light_pos = vec3<f32>(
        sin(uniforms.time * 0.7) * 3.0,
        2.0 + sin(uniforms.time * 0.3),
        cos(uniforms.time * 0.7) * 3.0
    );

    let light_dir = normalize(light_pos - p);
    let diff = max(dot(n, light_dir), 0.0);
    let ao = 0.5 + 0.5 * n.y;

    let ambient = vec3<f32>(0.1, 0.12, 0.15);
    let diffuse = vec3<f32>(0.9, 0.8, 0.7) * diff;

    return ambient * ao + diffuse;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let resolution = vec2<f32>(uniforms.resolution_x, uniforms.resolution_y);
    let coords = vec2<f32>(f32(global_id.x), f32(global_id.y));

    if (coords.x >= resolution.x || coords.y >= resolution.y) {
        return;
    }

    let uv = (coords - 0.5 * resolution) / min(resolution.x, resolution.y);

    // Camera
    let cam_pos = vec3<f32>(0.0, 0.5, 3.0);
    let cam_target = vec3<f32>(0.0, 0.0, 0.0);
    let cam_up = vec3<f32>(0.0, 1.0, 0.0);

    let forward = normalize(cam_target - cam_pos);
    let right = normalize(cross(forward, cam_up));
    let up = cross(right, forward);

    let rd = normalize(forward + uv.x * right + uv.y * up);

    // Ray march
    let result = ray_march(cam_pos, rd);
    let t = result.x;
    let mat_id = result.y;
    let alpha = result.z;

    // Discard if alpha below threshold (opaque pass)
    if (alpha < uniforms.alpha_threshold) {
        return;
    }

    if (mat_id >= 0.0 && t < MAX_DIST) {
        let p = cam_pos + rd * t;
        let n = calc_normal(p);
        let lighting = calc_lighting(p, n);
        let base_color = get_material_color(mat_id, p);
        var color = base_color * lighting;

        // Gamma correction
        color = pow(color, vec3<f32>(1.0 / 2.2));

        // Write color
        textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(color, 1.0));

        // Write depth (normalized)
        let depth = t / MAX_DIST;
        textureStore(depth_texture, vec2<i32>(global_id.xy), vec4<f32>(depth, 0.0, 0.0, 0.0));
    }
}
"#;

/// Embedded WGSL shader for transparent SDF pass.
pub const MULTIPASS_TRANSPARENT_SHADER: &str = r#"
// SDF Transparent Pass - reads depth, alpha blending

struct MultiPassUniforms {
    time: f32,
    resolution_x: f32,
    resolution_y: f32,
    pass_type: u32,
    alpha_threshold: f32,
    alpha_min: f32,
    depth_bias: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: MultiPassUniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var depth_texture: texture_2d<f32>;

// SDF primitives
fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_torus(p: vec3<f32>, r: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xz) - r.x, p.y);
    return length(q) - r.y;
}

// Transparent scene (glass torus, fog volumes)
fn scene_transparent_sdf(p: vec3<f32>) -> vec3<f32> {
    let t = uniforms.time;

    // Glass torus (transparent)
    let d_torus = sdf_torus(p - vec3<f32>(0.0, 0.2, -1.0), vec2<f32>(0.5, 0.15));

    // Fog sphere
    let fog_pos = vec3<f32>(sin(t * 0.3) * 1.5, 0.3, cos(t * 0.5) * 1.5);
    let d_fog = sdf_sphere(p - fog_pos, 0.6);

    var d = d_torus;
    var mat_id = 10.0; // Glass
    var alpha = 0.5;

    if (d_fog < d) {
        d = d_fog;
        mat_id = 11.0; // Fog
        // Distance-based alpha for volumetric effect
        alpha = 0.3 * exp(-abs(d_fog) * 2.0);
    }

    return vec3<f32>(d, mat_id, alpha);
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.001, 0.0);
    return normalize(vec3<f32>(
        scene_transparent_sdf(p + e.xyy).x - scene_transparent_sdf(p - e.xyy).x,
        scene_transparent_sdf(p + e.yxy).x - scene_transparent_sdf(p - e.yxy).x,
        scene_transparent_sdf(p + e.yyx).x - scene_transparent_sdf(p - e.yyx).x
    ));
}

const MAX_STEPS: i32 = 64;
const MAX_DIST: f32 = 20.0;
const SURF_DIST: f32 = 0.001;

fn ray_march_transparent(ro: vec3<f32>, rd: vec3<f32>, max_depth: f32) -> vec4<f32> {
    var t = 0.0;
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;

    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let result = scene_transparent_sdf(p);
        let d = result.x;
        let mat_id = result.y;
        let local_alpha = result.z;

        // Stop at opaque geometry depth
        if (t / MAX_DIST > max_depth - uniforms.depth_bias) {
            break;
        }

        if (d < SURF_DIST && local_alpha > uniforms.alpha_min) {
            // Calculate contribution
            let n = calc_normal(p);
            var color: vec3<f32>;

            if (mat_id > 10.5) {
                // Fog - bluish
                color = vec3<f32>(0.6, 0.7, 0.9);
            } else {
                // Glass - refractive tint
                let fresnel = 0.1 + 0.9 * pow(1.0 - abs(dot(n, -rd)), 3.0);
                color = vec3<f32>(0.9, 0.95, 1.0) * fresnel;
            }

            // Front-to-back compositing
            let contrib_alpha = local_alpha * (1.0 - accumulated_alpha);
            accumulated_color += color * contrib_alpha;
            accumulated_alpha += contrib_alpha;

            if (accumulated_alpha > 0.99) {
                break;
            }
        }

        t += max(abs(d), 0.01);

        if (t > MAX_DIST) {
            break;
        }
    }

    return vec4<f32>(accumulated_color, accumulated_alpha);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let resolution = vec2<f32>(uniforms.resolution_x, uniforms.resolution_y);
    let coords = vec2<f32>(f32(global_id.x), f32(global_id.y));

    if (coords.x >= resolution.x || coords.y >= resolution.y) {
        return;
    }

    let uv = (coords - 0.5 * resolution) / min(resolution.x, resolution.y);

    // Camera
    let cam_pos = vec3<f32>(0.0, 0.5, 3.0);
    let cam_target = vec3<f32>(0.0, 0.0, 0.0);
    let cam_up = vec3<f32>(0.0, 1.0, 0.0);

    let forward = normalize(cam_target - cam_pos);
    let right = normalize(cross(forward, cam_up));
    let up = cross(right, forward);

    let rd = normalize(forward + uv.x * right + uv.y * up);

    // Read existing depth from opaque pass
    let existing_depth = textureLoad(depth_texture, vec2<i32>(global_id.xy), 0).r;

    // Ray march transparent objects (stop at opaque depth)
    let result = ray_march_transparent(cam_pos, rd, existing_depth);

    // Only write if we have visible transparency
    if (result.w > uniforms.alpha_min) {
        // Output HDR color with alpha for compositing
        textureStore(output_texture, vec2<i32>(global_id.xy), result);
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SdfBlendMode Tests
    // =========================================================================

    #[test]
    fn test_blend_mode_opaque() {
        let mode = SdfBlendMode::Opaque;
        assert!(mode.to_wgpu_blend_state().is_none());
    }

    #[test]
    fn test_blend_mode_alpha_blend() {
        let mode = SdfBlendMode::AlphaBlend;
        let state = mode.to_wgpu_blend_state();
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(state.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }

    #[test]
    fn test_blend_mode_additive() {
        let mode = SdfBlendMode::Additive;
        let state = mode.to_wgpu_blend_state();
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(state.color.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_blend_mode_premultiplied() {
        let mode = SdfBlendMode::Premultiplied;
        let state = mode.to_wgpu_blend_state();
        assert!(state.is_some());
        let state = state.unwrap();
        assert_eq!(state.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(state.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }

    #[test]
    fn test_blend_mode_default() {
        assert_eq!(SdfBlendMode::default(), SdfBlendMode::Opaque);
    }

    // =========================================================================
    // SdfDepthMode Tests
    // =========================================================================

    #[test]
    fn test_depth_mode_write_and_test() {
        let mode = SdfDepthMode::WriteAndTest;
        assert!(mode.writes_depth());
        assert!(mode.tests_depth());
    }

    #[test]
    fn test_depth_mode_test_only() {
        let mode = SdfDepthMode::TestOnly;
        assert!(!mode.writes_depth());
        assert!(mode.tests_depth());
    }

    #[test]
    fn test_depth_mode_disabled() {
        let mode = SdfDepthMode::Disabled;
        assert!(!mode.writes_depth());
        assert!(!mode.tests_depth());
    }

    #[test]
    fn test_depth_mode_default() {
        assert_eq!(SdfDepthMode::default(), SdfDepthMode::WriteAndTest);
    }

    // =========================================================================
    // SdfPassType Tests
    // =========================================================================

    #[test]
    fn test_pass_type_opaque_defaults() {
        let pass_type = SdfPassType::Opaque;
        assert_eq!(pass_type.default_blend_mode(), SdfBlendMode::Opaque);
        assert_eq!(pass_type.default_depth_mode(), SdfDepthMode::WriteAndTest);
    }

    #[test]
    fn test_pass_type_transparent_defaults() {
        let pass_type = SdfPassType::Transparent;
        assert_eq!(pass_type.default_blend_mode(), SdfBlendMode::AlphaBlend);
        assert_eq!(pass_type.default_depth_mode(), SdfDepthMode::TestOnly);
    }

    // =========================================================================
    // MultiPassUniforms Tests
    // =========================================================================

    #[test]
    fn test_multipass_uniforms_default() {
        let uniforms = MultiPassUniforms::default();
        assert_eq!(uniforms.time, 0.0);
        assert_eq!(uniforms.resolution_x, 800.0);
        assert_eq!(uniforms.resolution_y, 600.0);
        assert_eq!(uniforms.pass_type, 0);
        assert_eq!(uniforms.alpha_threshold, OPAQUE_ALPHA_THRESHOLD);
    }

    #[test]
    fn test_multipass_uniforms_opaque() {
        let uniforms = MultiPassUniforms::opaque(1920, 1080, 5.0);
        assert_eq!(uniforms.time, 5.0);
        assert_eq!(uniforms.resolution_x, 1920.0);
        assert_eq!(uniforms.resolution_y, 1080.0);
        assert_eq!(uniforms.pass_type, 0);
        assert_eq!(uniforms.alpha_threshold, OPAQUE_ALPHA_THRESHOLD);
    }

    #[test]
    fn test_multipass_uniforms_transparent() {
        let uniforms = MultiPassUniforms::transparent(1920, 1080, 5.0);
        assert_eq!(uniforms.time, 5.0);
        assert_eq!(uniforms.pass_type, 1);
        assert_eq!(uniforms.alpha_min, TRANSPARENT_ALPHA_MIN);
        assert!(uniforms.depth_bias > 0.0);
    }

    #[test]
    fn test_multipass_uniforms_memory_layout() {
        assert_eq!(std::mem::size_of::<MultiPassUniforms>(), 32);
    }

    #[test]
    fn test_multipass_uniforms_as_bytes() {
        let uniforms = MultiPassUniforms::default();
        let bytes = uniforms.as_bytes();
        assert_eq!(bytes.len(), 32);
    }

    // =========================================================================
    // SdfPassConfig Tests
    // =========================================================================

    #[test]
    fn test_pass_config_opaque() {
        let config = SdfPassConfig::opaque();
        assert_eq!(config.pass_type, SdfPassType::Opaque);
        assert_eq!(config.blend_mode, SdfBlendMode::Opaque);
        assert_eq!(config.depth_mode, SdfDepthMode::WriteAndTest);
        assert!(!config.sort_transparent);
    }

    #[test]
    fn test_pass_config_transparent() {
        let config = SdfPassConfig::transparent();
        assert_eq!(config.pass_type, SdfPassType::Transparent);
        assert_eq!(config.blend_mode, SdfBlendMode::AlphaBlend);
        assert_eq!(config.depth_mode, SdfDepthMode::TestOnly);
        assert!(config.sort_transparent);
    }

    #[test]
    fn test_pass_config_additive() {
        let config = SdfPassConfig::additive();
        assert_eq!(config.pass_type, SdfPassType::Transparent);
        assert_eq!(config.blend_mode, SdfBlendMode::Additive);
        assert!(!config.sort_transparent);
    }

    #[test]
    fn test_pass_config_default() {
        let config = SdfPassConfig::default();
        assert_eq!(config.pass_type, SdfPassType::Opaque);
    }

    // =========================================================================
    // RenderPassOrder Tests
    // =========================================================================

    #[test]
    fn test_render_pass_order_all_ordered() {
        let order = RenderPassOrder::all_ordered();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], RenderPassOrder::RasterOpaque);
        assert_eq!(order[1], RenderPassOrder::SdfOpaque);
        assert_eq!(order[2], RenderPassOrder::SdfTransparent);
        assert_eq!(order[3], RenderPassOrder::RasterTransparent);
    }

    #[test]
    fn test_render_pass_order_writes_depth() {
        assert!(RenderPassOrder::RasterOpaque.writes_depth());
        assert!(RenderPassOrder::SdfOpaque.writes_depth());
        assert!(!RenderPassOrder::SdfTransparent.writes_depth());
        assert!(!RenderPassOrder::RasterTransparent.writes_depth());
    }

    #[test]
    fn test_render_pass_order_is_transparent() {
        assert!(!RenderPassOrder::RasterOpaque.is_transparent());
        assert!(!RenderPassOrder::SdfOpaque.is_transparent());
        assert!(RenderPassOrder::SdfTransparent.is_transparent());
        assert!(RenderPassOrder::RasterTransparent.is_transparent());
    }

    #[test]
    fn test_render_pass_order_is_sdf() {
        assert!(!RenderPassOrder::RasterOpaque.is_sdf());
        assert!(RenderPassOrder::SdfOpaque.is_sdf());
        assert!(RenderPassOrder::SdfTransparent.is_sdf());
        assert!(!RenderPassOrder::RasterTransparent.is_sdf());
    }

    #[test]
    fn test_render_pass_order_comparison() {
        assert!(RenderPassOrder::RasterOpaque < RenderPassOrder::SdfOpaque);
        assert!(RenderPassOrder::SdfOpaque < RenderPassOrder::SdfTransparent);
        assert!(RenderPassOrder::SdfTransparent < RenderPassOrder::RasterTransparent);
    }

    // =========================================================================
    // Shader Validation Tests
    // =========================================================================

    #[test]
    fn test_opaque_shader_not_empty() {
        assert!(!MULTIPASS_OPAQUE_SHADER.is_empty());
        assert!(MULTIPASS_OPAQUE_SHADER.len() > 1000);
    }

    #[test]
    fn test_opaque_shader_has_entry_point() {
        assert!(MULTIPASS_OPAQUE_SHADER.contains("@compute"));
        assert!(MULTIPASS_OPAQUE_SHADER.contains("fn main("));
        assert!(MULTIPASS_OPAQUE_SHADER.contains("@workgroup_size(8, 8, 1)"));
    }

    #[test]
    fn test_opaque_shader_has_uniforms() {
        assert!(MULTIPASS_OPAQUE_SHADER.contains("MultiPassUniforms"));
        assert!(MULTIPASS_OPAQUE_SHADER.contains("alpha_threshold"));
        assert!(MULTIPASS_OPAQUE_SHADER.contains("pass_type"));
    }

    #[test]
    fn test_opaque_shader_has_depth_write() {
        assert!(MULTIPASS_OPAQUE_SHADER.contains("depth_texture"));
        assert!(MULTIPASS_OPAQUE_SHADER.contains("textureStore"));
    }

    #[test]
    fn test_transparent_shader_not_empty() {
        assert!(!MULTIPASS_TRANSPARENT_SHADER.is_empty());
        assert!(MULTIPASS_TRANSPARENT_SHADER.len() > 1000);
    }

    #[test]
    fn test_transparent_shader_has_entry_point() {
        assert!(MULTIPASS_TRANSPARENT_SHADER.contains("@compute"));
        assert!(MULTIPASS_TRANSPARENT_SHADER.contains("fn main("));
        assert!(MULTIPASS_TRANSPARENT_SHADER.contains("@workgroup_size(8, 8, 1)"));
    }

    #[test]
    fn test_transparent_shader_has_uniforms() {
        assert!(MULTIPASS_TRANSPARENT_SHADER.contains("MultiPassUniforms"));
        assert!(MULTIPASS_TRANSPARENT_SHADER.contains("alpha_min"));
        assert!(MULTIPASS_TRANSPARENT_SHADER.contains("depth_bias"));
    }

    #[test]
    fn test_transparent_shader_reads_depth() {
        assert!(MULTIPASS_TRANSPARENT_SHADER.contains("depth_texture: texture_2d<f32>"));
        assert!(MULTIPASS_TRANSPARENT_SHADER.contains("textureLoad"));
    }

    #[test]
    fn test_transparent_shader_has_hdr_output() {
        assert!(MULTIPASS_TRANSPARENT_SHADER.contains("rgba16float"));
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_workgroup_size_constant() {
        assert_eq!(MULTIPASS_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn test_max_transparent_objects() {
        assert!(MAX_TRANSPARENT_OBJECTS >= 64);
    }

    #[test]
    fn test_alpha_thresholds() {
        assert!(OPAQUE_ALPHA_THRESHOLD > 0.9);
        assert!(OPAQUE_ALPHA_THRESHOLD <= 1.0);
        assert!(TRANSPARENT_ALPHA_MIN > 0.0);
        assert!(TRANSPARENT_ALPHA_MIN < 0.1);
    }

    // =========================================================================
    // Shader WGSL Parse Tests (using naga)
    // =========================================================================

    #[test]
    fn test_opaque_shader_parses() {
        use naga::front::wgsl;
        let result = wgsl::parse_str(MULTIPASS_OPAQUE_SHADER);
        assert!(result.is_ok(), "Opaque shader parse error: {:?}", result.err());
    }

    #[test]
    fn test_transparent_shader_parses() {
        use naga::front::wgsl;
        let result = wgsl::parse_str(MULTIPASS_TRANSPARENT_SHADER);
        assert!(result.is_ok(), "Transparent shader parse error: {:?}", result.err());
    }
}
