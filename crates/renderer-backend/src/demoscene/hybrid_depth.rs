//! Hybrid Mode Depth Buffer (T-DEMO-6.3 / T-DEMO-6.4)
//!
//! This module implements hybrid rendering where ray-marched SDF content is
//! composited with rasterized geometry using depth buffer comparison.
//!
//! # Overview
//!
//! In hybrid mode, the rasterizer first renders opaque geometry to a depth buffer.
//! The demoscene ray marcher then:
//! 1. Samples the rasterization depth buffer (T-DEMO-6.3)
//! 2. Converts NDC depth to linear depth for comparison
//! 3. Ray marches only up to the rasterized depth
//! 4. Writes color only where ray march hit is closer (T-DEMO-6.4)
//!
//! # Depth Buffer Format
//!
//! The depth buffer uses `TextureFormat::Depth32Float` for maximum precision.
//! NDC depth (0.0 = near, 1.0 = far) is converted to linear depth using the
//! camera's near/far planes.
//!
//! # Usage
//!
//! ```ignore
//! let hybrid = HybridDepthRenderer::new(&device, 800, 600);
//! hybrid.bind_depth_buffer(&depth_texture_view);
//! hybrid.dispatch(&mut encoder);
//! ```

use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default near plane distance for depth linearization.
pub const DEFAULT_NEAR_PLANE: f32 = 0.1;

/// Default far plane distance for depth linearization.
pub const DEFAULT_FAR_PLANE: f32 = 100.0;

/// Maximum ray march distance (should match shader MAX_DIST).
pub const MAX_RAY_MARCH_DIST: f32 = 20.0;

/// Depth buffer format used for hybrid rendering.
pub const DEPTH_BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Depth comparison epsilon for floating-point precision.
pub const DEPTH_EPSILON: f32 = 0.0001;

// ---------------------------------------------------------------------------
// HybridUniforms
// ---------------------------------------------------------------------------

/// Uniform buffer data for hybrid depth shader.
///
/// Extends the basic demoscene uniforms with depth buffer parameters.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field        | Size    |
/// |--------|--------------|---------|
/// | 0      | time         | 4 bytes |
/// | 4      | resolution_x | 4 bytes |
/// | 8      | resolution_y | 4 bytes |
/// | 12     | near_plane   | 4 bytes |
/// | 16     | far_plane    | 4 bytes |
/// | 20     | depth_enabled| 4 bytes (u32 as f32) |
/// | 24     | _padding0    | 4 bytes |
/// | 28     | _padding1    | 4 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct HybridUniforms {
    /// Animation time in seconds.
    pub time: f32,

    /// Output texture width in pixels.
    pub resolution_x: f32,

    /// Output texture height in pixels.
    pub resolution_y: f32,

    /// Camera near plane distance.
    pub near_plane: f32,

    /// Camera far plane distance.
    pub far_plane: f32,

    /// Whether depth buffer is enabled (1.0 = enabled, 0.0 = disabled).
    pub depth_enabled: f32,

    /// Padding for vec4 alignment.
    pub _padding0: f32,

    /// Padding for vec4 alignment.
    pub _padding1: f32,
}

impl Default for HybridUniforms {
    fn default() -> Self {
        Self {
            time: 0.0,
            resolution_x: 800.0,
            resolution_y: 600.0,
            near_plane: DEFAULT_NEAR_PLANE,
            far_plane: DEFAULT_FAR_PLANE,
            depth_enabled: 0.0,
            _padding0: 0.0,
            _padding1: 0.0,
        }
    }
}

impl HybridUniforms {
    /// Create new uniforms with the given resolution.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            resolution_x: width as f32,
            resolution_y: height as f32,
            ..Default::default()
        }
    }

    /// Create new uniforms with resolution and depth parameters.
    pub fn with_depth(width: u32, height: u32, near: f32, far: f32) -> Self {
        Self {
            resolution_x: width as f32,
            resolution_y: height as f32,
            near_plane: near,
            far_plane: far,
            depth_enabled: 1.0,
            ..Default::default()
        }
    }

    /// Update the animation time.
    #[inline]
    pub fn set_time(&mut self, time: f32) {
        self.time = time;
    }

    /// Update the resolution.
    #[inline]
    pub fn set_resolution(&mut self, width: u32, height: u32) {
        self.resolution_x = width as f32;
        self.resolution_y = height as f32;
    }

    /// Set the camera near and far planes.
    #[inline]
    pub fn set_depth_planes(&mut self, near: f32, far: f32) {
        self.near_plane = near.max(DEPTH_EPSILON);
        // Use clamped near_plane to ensure far > near
        self.far_plane = far.max(self.near_plane + DEPTH_EPSILON);
    }

    /// Enable or disable depth buffer usage.
    #[inline]
    pub fn set_depth_enabled(&mut self, enabled: bool) {
        self.depth_enabled = if enabled { 1.0 } else { 0.0 };
    }

    /// Check if depth buffer is enabled.
    #[inline]
    pub fn is_depth_enabled(&self) -> bool {
        self.depth_enabled > 0.5
    }

    /// Get the current resolution as (width, height).
    #[inline]
    pub fn resolution(&self) -> (u32, u32) {
        (self.resolution_x as u32, self.resolution_y as u32)
    }

    /// Get the uniform data as bytes for GPU upload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Convert NDC depth to linear depth.
    ///
    /// Uses the reverse-Z formula for better precision at distance.
    #[inline]
    pub fn ndc_to_linear_depth(&self, ndc_depth: f32) -> f32 {
        ndc_to_linear(ndc_depth, self.near_plane, self.far_plane)
    }

    /// Convert linear depth to NDC depth.
    #[inline]
    pub fn linear_to_ndc_depth(&self, linear_depth: f32) -> f32 {
        linear_to_ndc(linear_depth, self.near_plane, self.far_plane)
    }
}

// ---------------------------------------------------------------------------
// Depth Conversion Functions
// ---------------------------------------------------------------------------

/// Convert NDC depth (0.0 = near, 1.0 = far) to linear view-space depth.
///
/// Uses standard perspective projection formula:
/// ```text
/// linear = (near * far) / (far - ndc_depth * (far - near))
/// ```
#[inline]
pub fn ndc_to_linear(ndc_depth: f32, near: f32, far: f32) -> f32 {
    if ndc_depth >= 1.0 - DEPTH_EPSILON {
        return far;
    }
    if ndc_depth <= DEPTH_EPSILON {
        return near;
    }
    (near * far) / (far - ndc_depth * (far - near))
}

/// Convert linear view-space depth to NDC depth (0.0 = near, 1.0 = far).
///
/// Inverse of `ndc_to_linear`.
#[inline]
pub fn linear_to_ndc(linear_depth: f32, near: f32, far: f32) -> f32 {
    if linear_depth <= near {
        return 0.0;
    }
    if linear_depth >= far {
        return 1.0;
    }
    (far * (linear_depth - near)) / (linear_depth * (far - near))
}

/// Convert reverse-Z NDC depth to linear depth.
///
/// In reverse-Z, 1.0 = near and 0.0 = far for better precision.
#[inline]
pub fn reverse_z_to_linear(ndc_depth: f32, near: f32, far: f32) -> f32 {
    ndc_to_linear(1.0 - ndc_depth, near, far)
}

/// Convert linear depth to reverse-Z NDC depth.
#[inline]
pub fn linear_to_reverse_z(linear_depth: f32, near: f32, far: f32) -> f32 {
    1.0 - linear_to_ndc(linear_depth, near, far)
}

// ---------------------------------------------------------------------------
// DepthCompareResult
// ---------------------------------------------------------------------------

/// Result of comparing ray march hit with rasterized depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthCompareResult {
    /// Ray march hit is closer than raster depth - write ray march color.
    RayMarchCloser,
    /// Raster geometry is closer - discard ray march fragment.
    RasterCloser,
    /// Ray march missed entirely - show raster/sky.
    RayMarchMiss,
    /// Depths are equal within epsilon - prefer ray march for consistency.
    Equal,
}

impl DepthCompareResult {
    /// Determine comparison result from distances.
    ///
    /// # Arguments
    ///
    /// * `ray_march_dist` - Linear distance from camera to ray march hit (or MAX if miss)
    /// * `raster_depth` - Linear depth from rasterized depth buffer
    /// * `epsilon` - Tolerance for equality comparison
    pub fn compare(ray_march_dist: f32, raster_depth: f32, epsilon: f32) -> Self {
        // Ray march missed entirely
        if ray_march_dist >= MAX_RAY_MARCH_DIST - epsilon {
            return Self::RayMarchMiss;
        }

        let diff = ray_march_dist - raster_depth;

        if diff.abs() < epsilon {
            Self::Equal
        } else if diff < 0.0 {
            Self::RayMarchCloser
        } else {
            Self::RasterCloser
        }
    }

    /// Should the ray march color be written?
    #[inline]
    pub fn should_write_ray_march(self) -> bool {
        matches!(self, Self::RayMarchCloser | Self::Equal)
    }

    /// Should the raster color be preserved?
    #[inline]
    pub fn should_preserve_raster(self) -> bool {
        matches!(self, Self::RasterCloser | Self::RayMarchMiss)
    }
}

// ---------------------------------------------------------------------------
// HybridDepthConfig
// ---------------------------------------------------------------------------

/// Configuration for hybrid depth rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridDepthConfig {
    /// Render target width in pixels.
    pub width: u32,

    /// Render target height in pixels.
    pub height: u32,

    /// Camera near plane distance.
    pub near_plane: f32,

    /// Camera far plane distance.
    pub far_plane: f32,

    /// Workgroup size X (must match shader).
    pub workgroup_size_x: u32,

    /// Workgroup size Y (must match shader).
    pub workgroup_size_y: u32,

    /// Use reverse-Z depth buffer.
    pub reverse_z: bool,

    /// Depth comparison epsilon.
    pub depth_epsilon: f32,
}

impl Default for HybridDepthConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            near_plane: DEFAULT_NEAR_PLANE,
            far_plane: DEFAULT_FAR_PLANE,
            workgroup_size_x: 8,
            workgroup_size_y: 8,
            reverse_z: false,
            depth_epsilon: DEPTH_EPSILON,
        }
    }
}

impl HybridDepthConfig {
    /// Create a new configuration with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Create a configuration with depth planes.
    pub fn with_depth_planes(mut self, near: f32, far: f32) -> Self {
        self.near_plane = near.max(DEPTH_EPSILON);
        self.far_plane = far.max(near + DEPTH_EPSILON);
        self
    }

    /// Enable reverse-Z depth buffer.
    pub fn with_reverse_z(mut self, reverse_z: bool) -> Self {
        self.reverse_z = reverse_z;
        self
    }

    /// Set depth comparison epsilon.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.depth_epsilon = epsilon.max(0.0);
        self
    }

    /// Calculate the number of workgroups needed for dispatch.
    #[inline]
    pub fn dispatch_size(&self) -> (u32, u32, u32) {
        let x = (self.width + self.workgroup_size_x - 1) / self.workgroup_size_x;
        let y = (self.height + self.workgroup_size_y - 1) / self.workgroup_size_y;
        (x, y, 1)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.width == 0 {
            return Err("width cannot be zero");
        }
        if self.height == 0 {
            return Err("height cannot be zero");
        }
        if self.near_plane <= 0.0 {
            return Err("near plane must be positive");
        }
        if self.far_plane <= self.near_plane {
            return Err("far plane must be greater than near plane");
        }
        if self.workgroup_size_x == 0 || self.workgroup_size_y == 0 {
            return Err("workgroup size cannot be zero");
        }
        Ok(())
    }

    /// Create uniforms from this configuration.
    pub fn to_uniforms(&self) -> HybridUniforms {
        HybridUniforms::with_depth(self.width, self.height, self.near_plane, self.far_plane)
    }
}

// ---------------------------------------------------------------------------
// DepthBufferBinding
// ---------------------------------------------------------------------------

/// Represents a bound depth buffer for hybrid rendering.
#[derive(Debug)]
pub struct DepthBufferBinding {
    /// Width of the depth buffer.
    pub width: u32,

    /// Height of the depth buffer.
    pub height: u32,

    /// Whether the binding is valid.
    pub is_bound: bool,

    /// Whether this is a reverse-Z depth buffer.
    pub is_reverse_z: bool,
}

impl Default for DepthBufferBinding {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            is_bound: false,
            is_reverse_z: false,
        }
    }
}

impl DepthBufferBinding {
    /// Create a new binding for a depth buffer.
    pub fn new(width: u32, height: u32, reverse_z: bool) -> Self {
        Self {
            width,
            height,
            is_bound: true,
            is_reverse_z: reverse_z,
        }
    }

    /// Check if the binding matches the expected dimensions.
    pub fn matches_dimensions(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }

    /// Clear the binding (no depth buffer bound).
    pub fn clear(&mut self) {
        self.is_bound = false;
        self.width = 0;
        self.height = 0;
    }
}

// ---------------------------------------------------------------------------
// HybridDepthRenderer
// ---------------------------------------------------------------------------

/// Hybrid depth renderer for compositing ray-marched SDF with rasterized geometry.
///
/// This renderer samples a depth buffer from rasterization and uses it to
/// correctly composite ray-marched content.
pub struct HybridDepthRenderer {
    /// Configuration parameters.
    config: HybridDepthConfig,

    /// Uniform buffer for shader parameters.
    uniform_buffer: wgpu::Buffer,

    /// Current uniform data (CPU-side).
    uniforms: HybridUniforms,

    /// Compute pipeline for hybrid ray marching.
    compute_pipeline: wgpu::ComputePipeline,

    /// Bind group layout for the pipeline.
    bind_group_layout: wgpu::BindGroupLayout,

    /// Output texture for rendering (storage texture).
    output_texture: wgpu::Texture,

    /// Output texture view.
    output_view: wgpu::TextureView,

    /// Depth buffer binding state.
    depth_binding: DepthBufferBinding,

    /// Current bind group (rebuilt when depth buffer changes).
    bind_group: Option<wgpu::BindGroup>,

    /// Sampler for depth buffer.
    depth_sampler: wgpu::Sampler,

    /// Placeholder depth texture for when no depth buffer is bound.
    placeholder_depth: wgpu::Texture,

    /// Placeholder depth texture view.
    placeholder_depth_view: wgpu::TextureView,
}

/// Embedded hybrid depth shader source.
pub const HYBRID_DEPTH_SHADER: &str = include_str!("hybrid_depth.wgsl");

impl HybridDepthRenderer {
    /// Create a new hybrid depth renderer.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device for resource creation
    /// * `width` - render target width
    /// * `height` - render target height
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let config = HybridDepthConfig::new(width, height);
        Self::with_config(device, config)
    }

    /// Create a new hybrid depth renderer with custom configuration.
    pub fn with_config(device: &wgpu::Device, config: HybridDepthConfig) -> Self {
        config.validate().expect("invalid configuration");

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hybrid Depth Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(HYBRID_DEPTH_SHADER.into()),
        });

        // Create uniform buffer
        let uniforms = config.to_uniforms();
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hybrid Depth Uniform Buffer"),
            size: std::mem::size_of::<HybridUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output storage texture
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Hybrid Depth Output Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth sampler
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Hybrid Depth Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: None, // No depth comparison in sampler - we do it manually
            ..Default::default()
        });

        // Create placeholder depth texture (1x1, far plane value)
        let placeholder_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Hybrid Depth Placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_BUFFER_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create two views from the placeholder depth texture:
        // - One for the initial bind group creation
        // - One to store in the struct for later use
        let placeholder_depth_view_for_bind =
            placeholder_depth.create_view(&wgpu::TextureViewDescriptor::default());
        let placeholder_depth_view =
            placeholder_depth.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Hybrid Depth Bind Group Layout"),
            entries: &[
                // Uniform buffer binding (0)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<HybridUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // Output storage texture binding (1)
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
                // Depth texture binding (2)
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
                // Depth sampler binding (3)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // Create compute pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hybrid Depth Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hybrid Depth Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create initial bind group with placeholder depth BEFORE moving view into struct
        let initial_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hybrid Depth Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<HybridUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&placeholder_depth_view_for_bind),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&depth_sampler),
                },
            ],
        });

        Self {
            config,
            uniform_buffer,
            uniforms,
            compute_pipeline,
            bind_group_layout,
            output_texture,
            output_view,
            depth_binding: DepthBufferBinding::default(),
            bind_group: Some(initial_bind_group),
            depth_sampler,
            placeholder_depth,
            placeholder_depth_view,
        }
    }

    /// Rebuild the bind group with a new depth texture view.
    fn rebuild_bind_group(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hybrid Depth Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<HybridUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        }));
    }

    /// Rebuild bind group using the placeholder depth view.
    fn rebuild_bind_group_with_placeholder(&mut self, device: &wgpu::Device) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hybrid Depth Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<HybridUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.placeholder_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        }));
    }

    /// Bind a depth buffer for hybrid rendering.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device
    /// * `depth_view` - Depth texture view from rasterization
    /// * `width` - Depth buffer width
    /// * `height` - Depth buffer height
    /// * `reverse_z` - Whether the depth buffer uses reverse-Z
    pub fn bind_depth_buffer(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        reverse_z: bool,
    ) {
        self.depth_binding = DepthBufferBinding::new(width, height, reverse_z);
        self.uniforms.set_depth_enabled(true);
        self.config.reverse_z = reverse_z;
        self.rebuild_bind_group(device, depth_view);
    }

    /// Unbind the depth buffer (switch to full-screen mode).
    pub fn unbind_depth_buffer(&mut self, device: &wgpu::Device) {
        self.depth_binding.clear();
        self.uniforms.set_depth_enabled(false);
        // Recreate bind group with placeholder depth
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hybrid Depth Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<HybridUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.placeholder_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                },
            ],
        }));
    }

    /// Check if a depth buffer is currently bound.
    #[inline]
    pub fn has_depth_buffer(&self) -> bool {
        self.depth_binding.is_bound
    }

    /// Get the depth buffer binding state.
    #[inline]
    pub fn depth_binding(&self) -> &DepthBufferBinding {
        &self.depth_binding
    }

    /// Update animation uniforms with the given time.
    #[inline]
    pub fn update(&mut self, time: f32) {
        self.uniforms.set_time(time);
    }

    /// Update uniforms and upload to GPU.
    pub fn update_and_upload(&mut self, queue: &wgpu::Queue, time: f32) {
        self.update(time);
        queue.write_buffer(&self.uniform_buffer, 0, self.uniforms.as_bytes());
    }

    /// Dispatch the compute shader.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let (x, y, z) = self.config.dispatch_size();

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Hybrid Depth Compute Pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.compute_pipeline);
        if let Some(ref bind_group) = self.bind_group {
            cpass.set_bind_group(0, bind_group, &[]);
        }
        cpass.dispatch_workgroups(x, y, z);
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &HybridDepthConfig {
        &self.config
    }

    /// Get the current uniforms.
    #[inline]
    pub fn uniforms(&self) -> &HybridUniforms {
        &self.uniforms
    }

    /// Get the output texture view.
    #[inline]
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// Get the output texture.
    #[inline]
    pub fn output_texture(&self) -> &wgpu::Texture {
        &self.output_texture
    }

    /// Get the render dimensions.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Resize the render target.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.config.width && height == self.config.height {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.uniforms.set_resolution(width, height);

        // Recreate output texture
        self.output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Hybrid Depth Output Texture"),
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
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        self.output_view = self
            .output_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Rebuild bind group with new output view
        if self.depth_binding.is_bound {
            // Keep current depth view (caller must rebind if depth buffer size changed)
            return;
        }
        // Use placeholder depth view for rebuild
        self.rebuild_bind_group_with_placeholder(device);
    }

    /// Set depth planes for linearization.
    pub fn set_depth_planes(&mut self, near: f32, far: f32) {
        self.uniforms.set_depth_planes(near, far);
        self.config.near_plane = self.uniforms.near_plane;
        self.config.far_plane = self.uniforms.far_plane;
    }
}

// ---------------------------------------------------------------------------
// Shader Validation
// ---------------------------------------------------------------------------

/// Validate that the hybrid depth shader compiles successfully.
#[cfg(test)]
pub fn validate_hybrid_shader() -> Result<(), String> {
    use naga::front::wgsl;

    match wgsl::parse_str(HYBRID_DEPTH_SHADER) {
        Ok(_module) => Ok(()),
        Err(err) => Err(format!("Shader parse error: {}", err)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhi_device::RhiDevice;

    // =========================================================================
    // Depth Conversion Tests
    // =========================================================================

    #[test]
    fn test_ndc_to_linear_near_plane() {
        let linear = ndc_to_linear(0.0, 0.1, 100.0);
        assert!((linear - 0.1).abs() < 0.001, "Expected 0.1, got {}", linear);
    }

    #[test]
    fn test_ndc_to_linear_far_plane() {
        let linear = ndc_to_linear(1.0, 0.1, 100.0);
        assert!((linear - 100.0).abs() < 0.001, "Expected 100.0, got {}", linear);
    }

    #[test]
    fn test_ndc_to_linear_midpoint() {
        let linear = ndc_to_linear(0.5, 0.1, 100.0);
        // At NDC 0.5, linear depth should be around 0.2 with these planes
        assert!(linear > 0.1 && linear < 100.0);
    }

    #[test]
    fn test_linear_to_ndc_near_plane() {
        let ndc = linear_to_ndc(0.1, 0.1, 100.0);
        assert!(ndc.abs() < 0.001, "Expected 0.0, got {}", ndc);
    }

    #[test]
    fn test_linear_to_ndc_far_plane() {
        let ndc = linear_to_ndc(100.0, 0.1, 100.0);
        assert!((ndc - 1.0).abs() < 0.001, "Expected 1.0, got {}", ndc);
    }

    #[test]
    fn test_ndc_linear_roundtrip() {
        let near = 0.1;
        let far = 100.0;
        for ndc_input in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let linear = ndc_to_linear(ndc_input, near, far);
            let ndc_output = linear_to_ndc(linear, near, far);
            assert!(
                (ndc_input - ndc_output).abs() < 0.001,
                "Roundtrip failed: {} -> {} -> {}",
                ndc_input,
                linear,
                ndc_output
            );
        }
    }

    #[test]
    fn test_reverse_z_to_linear_near() {
        // In reverse-Z, 1.0 is near
        let linear = reverse_z_to_linear(1.0, 0.1, 100.0);
        assert!((linear - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_reverse_z_to_linear_far() {
        // In reverse-Z, 0.0 is far
        let linear = reverse_z_to_linear(0.0, 0.1, 100.0);
        assert!((linear - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_to_reverse_z_near() {
        let reverse_z = linear_to_reverse_z(0.1, 0.1, 100.0);
        assert!((reverse_z - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_to_reverse_z_far() {
        let reverse_z = linear_to_reverse_z(100.0, 0.1, 100.0);
        assert!(reverse_z.abs() < 0.001);
    }

    #[test]
    fn test_reverse_z_roundtrip() {
        let near = 0.5;
        let far = 50.0;
        for reverse_z_input in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let linear = reverse_z_to_linear(reverse_z_input, near, far);
            let reverse_z_output = linear_to_reverse_z(linear, near, far);
            assert!(
                (reverse_z_input - reverse_z_output).abs() < 0.001,
                "Roundtrip failed: {} -> {} -> {}",
                reverse_z_input,
                linear,
                reverse_z_output
            );
        }
    }

    // =========================================================================
    // HybridUniforms Tests
    // =========================================================================

    #[test]
    fn test_hybrid_uniforms_default() {
        let uniforms = HybridUniforms::default();
        assert_eq!(uniforms.time, 0.0);
        assert_eq!(uniforms.resolution_x, 800.0);
        assert_eq!(uniforms.resolution_y, 600.0);
        assert_eq!(uniforms.near_plane, DEFAULT_NEAR_PLANE);
        assert_eq!(uniforms.far_plane, DEFAULT_FAR_PLANE);
        assert!(!uniforms.is_depth_enabled());
    }

    #[test]
    fn test_hybrid_uniforms_new() {
        let uniforms = HybridUniforms::new(1920, 1080);
        assert_eq!(uniforms.resolution_x, 1920.0);
        assert_eq!(uniforms.resolution_y, 1080.0);
    }

    #[test]
    fn test_hybrid_uniforms_with_depth() {
        let uniforms = HybridUniforms::with_depth(800, 600, 0.5, 200.0);
        assert_eq!(uniforms.near_plane, 0.5);
        assert_eq!(uniforms.far_plane, 200.0);
        assert!(uniforms.is_depth_enabled());
    }

    #[test]
    fn test_hybrid_uniforms_set_time() {
        let mut uniforms = HybridUniforms::default();
        uniforms.set_time(5.5);
        assert_eq!(uniforms.time, 5.5);
    }

    #[test]
    fn test_hybrid_uniforms_set_resolution() {
        let mut uniforms = HybridUniforms::default();
        uniforms.set_resolution(1280, 720);
        assert_eq!(uniforms.resolution_x, 1280.0);
        assert_eq!(uniforms.resolution_y, 720.0);
    }

    #[test]
    fn test_hybrid_uniforms_set_depth_planes() {
        let mut uniforms = HybridUniforms::default();
        uniforms.set_depth_planes(0.01, 500.0);
        assert_eq!(uniforms.near_plane, 0.01);
        assert_eq!(uniforms.far_plane, 500.0);
    }

    #[test]
    fn test_hybrid_uniforms_set_depth_planes_clamp() {
        let mut uniforms = HybridUniforms::default();
        uniforms.set_depth_planes(-1.0, -5.0);
        assert!(uniforms.near_plane > 0.0);
        assert!(uniforms.far_plane > uniforms.near_plane);
    }

    #[test]
    fn test_hybrid_uniforms_depth_enabled() {
        let mut uniforms = HybridUniforms::default();
        assert!(!uniforms.is_depth_enabled());

        uniforms.set_depth_enabled(true);
        assert!(uniforms.is_depth_enabled());

        uniforms.set_depth_enabled(false);
        assert!(!uniforms.is_depth_enabled());
    }

    #[test]
    fn test_hybrid_uniforms_resolution() {
        let uniforms = HybridUniforms::new(1024, 768);
        let (w, h) = uniforms.resolution();
        assert_eq!(w, 1024);
        assert_eq!(h, 768);
    }

    #[test]
    fn test_hybrid_uniforms_as_bytes() {
        let uniforms = HybridUniforms::default();
        let bytes = uniforms.as_bytes();
        assert_eq!(bytes.len(), std::mem::size_of::<HybridUniforms>());
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_hybrid_uniforms_memory_layout() {
        assert_eq!(std::mem::size_of::<HybridUniforms>(), 32);
        assert_eq!(std::mem::align_of::<HybridUniforms>(), 4);
    }

    #[test]
    fn test_hybrid_uniforms_ndc_to_linear() {
        let uniforms = HybridUniforms::with_depth(800, 600, 0.1, 100.0);
        let linear = uniforms.ndc_to_linear_depth(0.0);
        assert!((linear - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_hybrid_uniforms_linear_to_ndc() {
        let uniforms = HybridUniforms::with_depth(800, 600, 0.1, 100.0);
        let ndc = uniforms.linear_to_ndc_depth(100.0);
        assert!((ndc - 1.0).abs() < 0.001);
    }

    // =========================================================================
    // DepthCompareResult Tests
    // =========================================================================

    #[test]
    fn test_depth_compare_ray_march_closer() {
        let result = DepthCompareResult::compare(5.0, 10.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RayMarchCloser);
        assert!(result.should_write_ray_march());
        assert!(!result.should_preserve_raster());
    }

    #[test]
    fn test_depth_compare_raster_closer() {
        let result = DepthCompareResult::compare(10.0, 5.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RasterCloser);
        assert!(!result.should_write_ray_march());
        assert!(result.should_preserve_raster());
    }

    #[test]
    fn test_depth_compare_ray_march_miss() {
        let result = DepthCompareResult::compare(MAX_RAY_MARCH_DIST, 5.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RayMarchMiss);
        assert!(!result.should_write_ray_march());
        assert!(result.should_preserve_raster());
    }

    #[test]
    fn test_depth_compare_equal() {
        let result = DepthCompareResult::compare(5.0, 5.0, 0.1);
        assert_eq!(result, DepthCompareResult::Equal);
        assert!(result.should_write_ray_march());
        assert!(!result.should_preserve_raster());
    }

    #[test]
    fn test_depth_compare_equal_within_epsilon() {
        let result = DepthCompareResult::compare(5.0, 5.05, 0.1);
        assert_eq!(result, DepthCompareResult::Equal);
    }

    #[test]
    fn test_depth_compare_boundary_near_miss() {
        let result = DepthCompareResult::compare(MAX_RAY_MARCH_DIST - 0.1, 5.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RasterCloser);
    }

    // =========================================================================
    // HybridDepthConfig Tests
    // =========================================================================

    #[test]
    fn test_hybrid_depth_config_default() {
        let config = HybridDepthConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.near_plane, DEFAULT_NEAR_PLANE);
        assert_eq!(config.far_plane, DEFAULT_FAR_PLANE);
        assert!(!config.reverse_z);
    }

    #[test]
    fn test_hybrid_depth_config_new() {
        let config = HybridDepthConfig::new(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn test_hybrid_depth_config_with_depth_planes() {
        let config = HybridDepthConfig::new(800, 600).with_depth_planes(0.5, 500.0);
        assert_eq!(config.near_plane, 0.5);
        assert_eq!(config.far_plane, 500.0);
    }

    #[test]
    fn test_hybrid_depth_config_with_reverse_z() {
        let config = HybridDepthConfig::new(800, 600).with_reverse_z(true);
        assert!(config.reverse_z);
    }

    #[test]
    fn test_hybrid_depth_config_with_epsilon() {
        let config = HybridDepthConfig::new(800, 600).with_epsilon(0.01);
        assert_eq!(config.depth_epsilon, 0.01);
    }

    #[test]
    fn test_hybrid_depth_config_dispatch_size() {
        let config = HybridDepthConfig::new(800, 600);
        let (x, y, z) = config.dispatch_size();
        assert_eq!(x, 100);
        assert_eq!(y, 75);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_hybrid_depth_config_dispatch_size_non_aligned() {
        let config = HybridDepthConfig::new(801, 601);
        let (x, y, z) = config.dispatch_size();
        assert_eq!(x, 101);
        assert_eq!(y, 76);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_hybrid_depth_config_validate_ok() {
        let config = HybridDepthConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_hybrid_depth_config_validate_width_zero() {
        let mut config = HybridDepthConfig::default();
        config.width = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_hybrid_depth_config_validate_height_zero() {
        let mut config = HybridDepthConfig::default();
        config.height = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_hybrid_depth_config_validate_near_negative() {
        let mut config = HybridDepthConfig::default();
        config.near_plane = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_hybrid_depth_config_validate_far_less_than_near() {
        let mut config = HybridDepthConfig::default();
        config.near_plane = 10.0;
        config.far_plane = 5.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_hybrid_depth_config_to_uniforms() {
        let config = HybridDepthConfig::new(1024, 768).with_depth_planes(0.5, 200.0);
        let uniforms = config.to_uniforms();
        assert_eq!(uniforms.resolution_x, 1024.0);
        assert_eq!(uniforms.resolution_y, 768.0);
        assert_eq!(uniforms.near_plane, 0.5);
        assert_eq!(uniforms.far_plane, 200.0);
    }

    // =========================================================================
    // DepthBufferBinding Tests
    // =========================================================================

    #[test]
    fn test_depth_buffer_binding_default() {
        let binding = DepthBufferBinding::default();
        assert!(!binding.is_bound);
        assert_eq!(binding.width, 0);
        assert_eq!(binding.height, 0);
    }

    #[test]
    fn test_depth_buffer_binding_new() {
        let binding = DepthBufferBinding::new(1920, 1080, false);
        assert!(binding.is_bound);
        assert_eq!(binding.width, 1920);
        assert_eq!(binding.height, 1080);
        assert!(!binding.is_reverse_z);
    }

    #[test]
    fn test_depth_buffer_binding_reverse_z() {
        let binding = DepthBufferBinding::new(800, 600, true);
        assert!(binding.is_reverse_z);
    }

    #[test]
    fn test_depth_buffer_binding_matches_dimensions() {
        let binding = DepthBufferBinding::new(800, 600, false);
        assert!(binding.matches_dimensions(800, 600));
        assert!(!binding.matches_dimensions(1024, 768));
    }

    #[test]
    fn test_depth_buffer_binding_clear() {
        let mut binding = DepthBufferBinding::new(800, 600, false);
        binding.clear();
        assert!(!binding.is_bound);
        assert_eq!(binding.width, 0);
        assert_eq!(binding.height, 0);
    }

    // =========================================================================
    // Shader Tests
    // =========================================================================

    #[test]
    fn test_hybrid_shader_embedded_constant() {
        assert!(!HYBRID_DEPTH_SHADER.is_empty());
        assert!(HYBRID_DEPTH_SHADER.len() > 100);
    }

    #[test]
    fn test_hybrid_shader_contains_entry_point() {
        assert!(HYBRID_DEPTH_SHADER.contains("fn main"));
        assert!(HYBRID_DEPTH_SHADER.contains("@compute"));
    }

    #[test]
    fn test_hybrid_shader_contains_workgroup_size() {
        assert!(HYBRID_DEPTH_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_hybrid_shader_contains_uniforms() {
        assert!(HYBRID_DEPTH_SHADER.contains("HybridUniforms"));
        assert!(HYBRID_DEPTH_SHADER.contains("near_plane"));
        assert!(HYBRID_DEPTH_SHADER.contains("far_plane"));
        assert!(HYBRID_DEPTH_SHADER.contains("depth_enabled"));
    }

    #[test]
    fn test_hybrid_shader_contains_depth_bindings() {
        assert!(HYBRID_DEPTH_SHADER.contains("@group(0) @binding(2)"));
        assert!(HYBRID_DEPTH_SHADER.contains("@group(0) @binding(3)"));
    }

    #[test]
    fn test_hybrid_shader_contains_depth_functions() {
        assert!(HYBRID_DEPTH_SHADER.contains("ndc_to_linear"));
        assert!(HYBRID_DEPTH_SHADER.contains("sample_depth"));
    }

    #[test]
    fn test_hybrid_shader_contains_ray_march() {
        assert!(HYBRID_DEPTH_SHADER.contains("ray_march"));
        assert!(HYBRID_DEPTH_SHADER.contains("MAX_STEPS"));
    }

    #[test]
    fn test_hybrid_shader_validation_naga() {
        let result = validate_hybrid_shader();
        assert!(result.is_ok(), "Shader should parse: {:?}", result.err());
    }

    // =========================================================================
    // HybridDepthRenderer GPU Tests
    // =========================================================================

    #[test]
    fn test_hybrid_renderer_creation() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            assert_eq!(renderer.size(), (800, 600));
            assert!(!renderer.has_depth_buffer());
        }
    }

    #[test]
    fn test_hybrid_renderer_with_config() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let config = HybridDepthConfig::new(1280, 720).with_depth_planes(0.5, 200.0);
            let renderer = HybridDepthRenderer::with_config(&device.device, config);
            assert_eq!(renderer.size(), (1280, 720));
            assert_eq!(renderer.uniforms().near_plane, 0.5);
            assert_eq!(renderer.uniforms().far_plane, 200.0);
        }
    }

    #[test]
    fn test_hybrid_renderer_update() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.update(1.5);
            assert_eq!(renderer.uniforms().time, 1.5);
        }
    }

    #[test]
    fn test_hybrid_renderer_update_and_upload() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.update_and_upload(&device.queue, 2.0);
            assert_eq!(renderer.uniforms().time, 2.0);
        }
    }

    #[test]
    fn test_hybrid_renderer_dispatch() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HybridDepthRenderer::new(&device.device, 800, 600);

            let mut encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Test Encoder"),
                    });

            renderer.dispatch(&mut encoder);

            device.queue.submit(std::iter::once(encoder.finish()));
            device.wait_idle();
        }
    }

    #[test]
    fn test_hybrid_renderer_resize() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.resize(&device.device, 1920, 1080);
            assert_eq!(renderer.size(), (1920, 1080));
        }
    }

    #[test]
    fn test_hybrid_renderer_resize_same_size() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.resize(&device.device, 800, 600);
            assert_eq!(renderer.size(), (800, 600));
        }
    }

    #[test]
    fn test_hybrid_renderer_depth_planes() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.set_depth_planes(0.01, 500.0);
            assert_eq!(renderer.uniforms().near_plane, 0.01);
            assert_eq!(renderer.uniforms().far_plane, 500.0);
        }
    }

    #[test]
    fn test_hybrid_renderer_bind_depth_buffer() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);

            // Create a depth texture
            let depth_texture = device.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Test Depth Texture"),
                size: wgpu::Extent3d {
                    width: 800,
                    height: 600,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_BUFFER_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            renderer.bind_depth_buffer(&device.device, &depth_view, 800, 600, false);

            assert!(renderer.has_depth_buffer());
            assert!(renderer.uniforms().is_depth_enabled());
        }
    }

    #[test]
    fn test_hybrid_renderer_unbind_depth_buffer() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);

            // Create and bind depth buffer
            let depth_texture = device.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Test Depth Texture"),
                size: wgpu::Extent3d {
                    width: 800,
                    height: 600,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_BUFFER_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            renderer.bind_depth_buffer(&device.device, &depth_view, 800, 600, false);

            // Unbind
            renderer.unbind_depth_buffer(&device.device);

            assert!(!renderer.has_depth_buffer());
            assert!(!renderer.uniforms().is_depth_enabled());
        }
    }

    #[test]
    fn test_hybrid_renderer_dispatch_with_depth() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);

            // Create depth texture
            let depth_texture = device.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Test Depth Texture"),
                size: wgpu::Extent3d {
                    width: 800,
                    height: 600,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_BUFFER_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            renderer.bind_depth_buffer(&device.device, &depth_view, 800, 600, false);

            renderer.update_and_upload(&device.queue, 1.0);

            let mut encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Test Encoder"),
                    });

            renderer.dispatch(&mut encoder);

            device.queue.submit(std::iter::once(encoder.finish()));
            device.wait_idle();
        }
    }

    #[test]
    fn test_hybrid_renderer_output_view() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            let _view = renderer.output_view();
        }
    }

    #[test]
    fn test_hybrid_renderer_output_texture() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            let texture = renderer.output_texture();
            assert_eq!(texture.width(), 800);
            assert_eq!(texture.height(), 600);
        }
    }

    // =========================================================================
    // Edge Cases and Integration Tests
    // =========================================================================

    #[test]
    fn test_depth_conversion_extreme_near() {
        let linear = ndc_to_linear(0.001, 0.001, 1000.0);
        assert!(linear >= 0.001 && linear < 0.01);
    }

    #[test]
    fn test_depth_conversion_extreme_far() {
        // NDC depth near 1.0 (far plane) with large depth range
        // The perspective projection concentrates precision near the near plane,
        // so linear depth increases rapidly only near NDC 1.0
        let linear = ndc_to_linear(0.999, 0.1, 10000.0);
        // At 0.999 NDC with these planes, linear depth is around 99
        // because of the non-linear distribution of depth values
        assert!(linear > 50.0 && linear < 200.0, "Got {}", linear);

        // At NDC 1.0 (or close to it), we get the far plane
        let linear_at_far = ndc_to_linear(1.0, 0.1, 10000.0);
        assert!((linear_at_far - 10000.0).abs() < 1.0, "Far plane: {}", linear_at_far);
    }

    #[test]
    fn test_depth_compare_many_cases() {
        // Test various distance combinations with small epsilon
        let epsilon = DEPTH_EPSILON;
        let test_cases = [
            (1.0, 10.0, DepthCompareResult::RayMarchCloser),
            (10.0, 1.0, DepthCompareResult::RasterCloser),
            (MAX_RAY_MARCH_DIST, 5.0, DepthCompareResult::RayMarchMiss),
            // With small epsilon, this should be RasterCloser
            (MAX_RAY_MARCH_DIST - 0.1, 5.0, DepthCompareResult::RasterCloser),
        ];

        for (ray_dist, raster_depth, expected) in test_cases {
            let result = DepthCompareResult::compare(ray_dist, raster_depth, epsilon);
            assert_eq!(
                result, expected,
                "Failed for ray_dist={}, raster_depth={}",
                ray_dist, raster_depth
            );
        }

        // Test equal case separately with appropriate epsilon
        let result = DepthCompareResult::compare(5.0, 5.0, 0.1);
        assert_eq!(result, DepthCompareResult::Equal);
    }

    #[test]
    fn test_config_chain_methods() {
        let config = HybridDepthConfig::new(1920, 1080)
            .with_depth_planes(0.01, 1000.0)
            .with_reverse_z(true)
            .with_epsilon(0.001);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.near_plane, 0.01);
        assert_eq!(config.far_plane, 1000.0);
        assert!(config.reverse_z);
        assert_eq!(config.depth_epsilon, 0.001);
    }

    #[test]
    fn test_uniforms_clone_copy() {
        let uniforms1 = HybridUniforms::with_depth(800, 600, 0.1, 100.0);
        let uniforms2 = uniforms1;
        assert_eq!(uniforms1, uniforms2);
    }

    #[test]
    fn test_config_clone_copy() {
        let config1 = HybridDepthConfig::new(800, 600);
        let config2 = config1;
        assert_eq!(config1, config2);
    }

    #[test]
    fn test_depth_compare_result_exhaustive() {
        // Ensure all variants are testable
        let variants = [
            DepthCompareResult::RayMarchCloser,
            DepthCompareResult::RasterCloser,
            DepthCompareResult::RayMarchMiss,
            DepthCompareResult::Equal,
        ];

        for v in variants {
            let _ = format!("{:?}", v);
            let _ = v.should_write_ray_march();
            let _ = v.should_preserve_raster();
        }
    }

    #[test]
    fn test_hybrid_uniforms_pod_zeroable() {
        let zeroed: HybridUniforms = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.time, 0.0);
        assert_eq!(zeroed.resolution_x, 0.0);
        assert_eq!(zeroed.near_plane, 0.0);
    }

    #[test]
    fn test_constants_values() {
        assert_eq!(DEFAULT_NEAR_PLANE, 0.1);
        assert_eq!(DEFAULT_FAR_PLANE, 100.0);
        assert_eq!(MAX_RAY_MARCH_DIST, 20.0);
        assert_eq!(DEPTH_BUFFER_FORMAT, wgpu::TextureFormat::Depth32Float);
        assert!(DEPTH_EPSILON > 0.0);
    }
}
