//! Reflection Buffer for SSR, RT reflections, and probe blending.
//!
//! This module provides a standardized buffer format for storing reflection data
//! from multiple techniques (SSR, ray-traced reflections, reflection probes, and
//! environment maps). The buffer is typically rendered at half resolution and
//! upscaled using a bilateral filter to preserve sharp reflection boundaries.
//!
//! # Usage
//!
//! ```ignore
//! // Create reflection buffer config at half resolution
//! let config = ReflectionBufferConfig::half_res(1920, 1080);
//!
//! // Create buffer resource
//! let buffer = ReflectionBuffer::new(&device, &config);
//!
//! // After reflection passes write to buffer, upscale to full res
//! let upscale_pass = BilateralUpscalePass::new(&device, &config);
//! upscale_pass.dispatch(&mut encoder, &buffer, &depth_texture, &normal_texture, &output);
//! ```
//!
//! # Technique Blending
//!
//! The `technique_mask` field in each pixel indicates which reflection
//! techniques contributed to that pixel. This allows the compositor to:
//!
//! - Blend multiple techniques based on confidence/quality
//! - Fall back to lower-quality techniques when SSR misses
//! - Debug which technique is being used per-pixel
//!
//! Technique priority (highest to lowest):
//! 1. Ray-traced reflections (RT) - highest quality
//! 2. Screen-space reflections (SSR) - good quality, limited range
//! 3. Reflection probes (Probe) - pre-baked, parallax corrected
//! 4. Environment map (Env) - fallback for sky/distant reflections

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Technique Flags
// ---------------------------------------------------------------------------

/// Technique flags for the `technique_mask` field in `ReflectionBufferPixel`.
///
/// Multiple flags can be combined to indicate hybrid techniques or fallback chains.
pub mod technique_flags {
    /// Screen-space reflections (SSR).
    pub const SSR: u32 = 1 << 0;
    /// Ray-traced reflections (hardware or software RT).
    pub const RT: u32 = 1 << 1;
    /// Reflection probe (parallax-corrected cubemap).
    pub const PROBE: u32 = 1 << 2;
    /// Environment map fallback (sky, distant geometry).
    pub const ENV: u32 = 1 << 3;

    /// Returns a human-readable string for the technique mask.
    pub fn to_string(mask: u32) -> String {
        let mut parts = Vec::new();
        if mask & SSR != 0 {
            parts.push("SSR");
        }
        if mask & RT != 0 {
            parts.push("RT");
        }
        if mask & PROBE != 0 {
            parts.push("Probe");
        }
        if mask & ENV != 0 {
            parts.push("Env");
        }
        if parts.is_empty() {
            "None".to_string()
        } else {
            parts.join("|")
        }
    }

    /// Returns the highest priority technique from a mask.
    ///
    /// Priority order: RT > SSR > Probe > Env
    pub fn highest_priority(mask: u32) -> u32 {
        if mask & RT != 0 {
            RT
        } else if mask & SSR != 0 {
            SSR
        } else if mask & PROBE != 0 {
            PROBE
        } else if mask & ENV != 0 {
            ENV
        } else {
            0
        }
    }
}

// Re-export at module level for convenience
pub use technique_flags as TechniqueFlags;

// ---------------------------------------------------------------------------
// ReflectionBufferPixel
// ---------------------------------------------------------------------------

/// Reflection buffer pixel data for GPU storage.
///
/// This struct contains all per-pixel reflection data needed for compositing:
/// - Reflection color (linear RGB)
/// - Source roughness at hit point (for blur filtering)
/// - World-space hit distance (negative = miss, used for temporal stability)
/// - Technique mask (which reflection method produced this pixel)
///
/// The struct is 32 bytes with proper alignment for GPU access.
///
/// # Memory Layout
///
/// | Offset | Size | Field          | Description                          |
/// |--------|------|----------------|--------------------------------------|
/// | 0      | 12   | color          | RGB reflection color (linear)        |
/// | 12     | 4    | roughness      | Source roughness at hit point        |
/// | 16     | 4    | hit_distance   | World-space distance (neg = miss)    |
/// | 20     | 4    | technique_mask | Bit flags for technique(s) used      |
/// | 24     | 8    | _pad           | Padding for 32-byte alignment        |
///
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ReflectionBufferPixel {
    /// RGB reflection color (linear color space).
    ///
    /// Values are HDR (can exceed 1.0) and should be tonemapped during
    /// final compositing.
    pub color: [f32; 3],

    /// Source roughness at the reflection hit point.
    ///
    /// Used for roughness-based blur filtering during upscaling.
    /// Range: 0.0 (mirror) to 1.0 (fully rough).
    pub roughness: f32,

    /// World-space distance to the reflection hit point.
    ///
    /// - Positive values: valid hit at this distance
    /// - Negative values: miss (use fallback technique)
    /// - Zero: degenerate case (treat as miss)
    ///
    /// Used for temporal reprojection and depth-aware filtering.
    pub hit_distance: f32,

    /// Bitmask indicating which reflection technique(s) contributed.
    ///
    /// See [`TechniqueFlags`] for flag values:
    /// - Bit 0: SSR
    /// - Bit 1: RT
    /// - Bit 2: Probe
    /// - Bit 3: Env
    pub technique_mask: u32,

    /// Padding for 32-byte alignment.
    ///
    /// Required for optimal GPU cache line alignment.
    pub _pad: [f32; 2],
}

// Compile-time size assertion: 3*4 + 4 + 4 + 4 + 2*4 = 32 bytes
const _: () = assert!(std::mem::size_of::<ReflectionBufferPixel>() == 32);
const _: () = assert!(std::mem::align_of::<ReflectionBufferPixel>() == 4);

impl Default for ReflectionBufferPixel {
    fn default() -> Self {
        Self {
            color: [0.0, 0.0, 0.0],
            roughness: 0.0,
            hit_distance: -1.0, // Indicate miss by default
            technique_mask: 0,
            _pad: [0.0, 0.0],
        }
    }
}

impl ReflectionBufferPixel {
    /// Create a new reflection pixel with the given values.
    pub fn new(
        color: [f32; 3],
        roughness: f32,
        hit_distance: f32,
        technique_mask: u32,
    ) -> Self {
        Self {
            color,
            roughness,
            hit_distance,
            technique_mask,
            _pad: [0.0, 0.0],
        }
    }

    /// Create a miss pixel (no valid reflection).
    pub fn miss() -> Self {
        Self {
            color: [0.0, 0.0, 0.0],
            roughness: 0.0,
            hit_distance: -1.0,
            technique_mask: 0,
            _pad: [0.0, 0.0],
        }
    }

    /// Create a pixel from SSR hit.
    pub fn from_ssr(color: [f32; 3], roughness: f32, hit_distance: f32) -> Self {
        Self::new(color, roughness, hit_distance, TechniqueFlags::SSR)
    }

    /// Create a pixel from ray-traced hit.
    pub fn from_rt(color: [f32; 3], roughness: f32, hit_distance: f32) -> Self {
        Self::new(color, roughness, hit_distance, TechniqueFlags::RT)
    }

    /// Create a pixel from reflection probe.
    pub fn from_probe(color: [f32; 3], roughness: f32) -> Self {
        // Probes don't have a specific hit distance
        Self::new(color, roughness, 0.0, TechniqueFlags::PROBE)
    }

    /// Create a pixel from environment map fallback.
    pub fn from_env(color: [f32; 3]) -> Self {
        Self::new(color, 1.0, 0.0, TechniqueFlags::ENV)
    }

    /// Returns `true` if this pixel represents a valid hit.
    #[inline]
    pub fn is_hit(&self) -> bool {
        self.hit_distance > 0.0 || self.technique_mask != 0
    }

    /// Returns `true` if this pixel is a miss (no reflection).
    #[inline]
    pub fn is_miss(&self) -> bool {
        !self.is_hit()
    }

    /// Returns the primary technique used for this pixel.
    pub fn primary_technique(&self) -> u32 {
        TechniqueFlags::highest_priority(self.technique_mask)
    }

    /// Returns a human-readable technique string.
    pub fn technique_string(&self) -> String {
        TechniqueFlags::to_string(self.technique_mask)
    }
}

// ---------------------------------------------------------------------------
// ReflectionBufferConfig
// ---------------------------------------------------------------------------

/// Configuration for reflection buffer creation.
///
/// Typically reflections are rendered at half resolution to save bandwidth
/// and then upscaled using a bilateral filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectionBufferConfig {
    /// Width of the reflection buffer (typically half screen width).
    pub width: u32,

    /// Height of the reflection buffer (typically half screen height).
    pub height: u32,

    /// Resolution scale relative to full screen.
    ///
    /// - 0.5 = half resolution (default, recommended)
    /// - 1.0 = full resolution (expensive)
    /// - 0.25 = quarter resolution (very fast, lower quality)
    pub resolution_scale: f32,
}

impl Default for ReflectionBufferConfig {
    fn default() -> Self {
        Self {
            width: 960,
            height: 540,
            resolution_scale: 0.5,
        }
    }
}

impl ReflectionBufferConfig {
    /// Create a config for half-resolution rendering.
    ///
    /// # Arguments
    /// * `full_width` - Full screen width
    /// * `full_height` - Full screen height
    pub fn half_res(full_width: u32, full_height: u32) -> Self {
        Self {
            width: full_width / 2,
            height: full_height / 2,
            resolution_scale: 0.5,
        }
    }

    /// Create a config for full-resolution rendering.
    ///
    /// Warning: This is expensive and usually unnecessary.
    pub fn full_res(full_width: u32, full_height: u32) -> Self {
        Self {
            width: full_width,
            height: full_height,
            resolution_scale: 1.0,
        }
    }

    /// Create a config for quarter-resolution rendering.
    ///
    /// Useful for very rough reflections or performance-critical scenarios.
    pub fn quarter_res(full_width: u32, full_height: u32) -> Self {
        Self {
            width: full_width / 4,
            height: full_height / 4,
            resolution_scale: 0.25,
        }
    }

    /// Create a config with custom resolution scale.
    ///
    /// # Arguments
    /// * `full_width` - Full screen width
    /// * `full_height` - Full screen height
    /// * `scale` - Resolution scale (0.0-1.0)
    pub fn with_scale(full_width: u32, full_height: u32, scale: f32) -> Self {
        let scale = scale.clamp(0.1, 1.0);
        Self {
            width: ((full_width as f32) * scale).max(1.0) as u32,
            height: ((full_height as f32) * scale).max(1.0) as u32,
            resolution_scale: scale,
        }
    }

    /// Returns the total number of pixels in the reflection buffer.
    #[inline]
    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }

    /// Returns the buffer size in bytes.
    #[inline]
    pub fn buffer_size(&self) -> u64 {
        (self.pixel_count() as u64) * (std::mem::size_of::<ReflectionBufferPixel>() as u64)
    }

    /// Returns the full resolution dimensions this buffer upscales to.
    pub fn full_resolution(&self) -> (u32, u32) {
        if self.resolution_scale > 0.0 {
            let full_w = (self.width as f32 / self.resolution_scale).round() as u32;
            let full_h = (self.height as f32 / self.resolution_scale).round() as u32;
            (full_w, full_h)
        } else {
            (self.width, self.height)
        }
    }
}

// ---------------------------------------------------------------------------
// ReflectionBuffer
// ---------------------------------------------------------------------------

/// Minimum buffer size in bytes (wgpu requires non-zero buffers).
const MIN_BUFFER_SIZE: u64 = 32;

/// GPU buffer for reflection data.
///
/// Stores per-pixel reflection information from SSR, RT, probes, and env maps.
/// The buffer is typically half-resolution and upscaled during compositing.
pub struct ReflectionBuffer {
    /// GPU storage buffer containing reflection pixel data.
    buffer: wgpu::Buffer,
    /// Configuration for this buffer.
    config: ReflectionBufferConfig,
    /// CPU-side cache for debugging and readback.
    cpu_data: Option<Vec<ReflectionBufferPixel>>,
}

impl ReflectionBuffer {
    /// Create a new reflection buffer with the specified configuration.
    ///
    /// # Arguments
    /// * `device` - wgpu device for buffer creation
    /// * `config` - Buffer configuration
    pub fn new(device: &wgpu::Device, config: &ReflectionBufferConfig) -> Self {
        let buffer_size = config.buffer_size().max(MIN_BUFFER_SIZE);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reflection Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            config: *config,
            cpu_data: None,
        }
    }

    /// Create a reflection buffer with CPU-side caching enabled.
    ///
    /// Useful for debugging and readback, but uses more memory.
    pub fn new_with_cpu_cache(device: &wgpu::Device, config: &ReflectionBufferConfig) -> Self {
        let mut buffer = Self::new(device, config);
        buffer.cpu_data = Some(vec![
            ReflectionBufferPixel::default();
            config.pixel_count() as usize
        ]);
        buffer
    }

    /// Get the GPU buffer.
    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the buffer configuration.
    #[inline]
    pub fn config(&self) -> &ReflectionBufferConfig {
        &self.config
    }

    /// Get buffer width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Get buffer height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.config.height
    }

    /// Get total pixel count.
    #[inline]
    pub fn pixel_count(&self) -> u32 {
        self.config.pixel_count()
    }

    /// Clear the buffer to all misses.
    pub fn clear(&mut self, queue: &wgpu::Queue) {
        let clear_data = vec![ReflectionBufferPixel::miss(); self.pixel_count() as usize];
        let bytes = bytemuck::cast_slice(&clear_data);
        queue.write_buffer(&self.buffer, 0, bytes);

        if let Some(ref mut cpu_data) = self.cpu_data {
            cpu_data.fill(ReflectionBufferPixel::miss());
        }
    }

    /// Resize the buffer to new dimensions.
    ///
    /// Creates a new GPU buffer; the old data is discarded.
    pub fn resize(&mut self, device: &wgpu::Device, config: &ReflectionBufferConfig) {
        if self.config == *config {
            return;
        }

        let buffer_size = config.buffer_size().max(MIN_BUFFER_SIZE);

        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reflection Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        if self.cpu_data.is_some() {
            self.cpu_data = Some(vec![
                ReflectionBufferPixel::default();
                config.pixel_count() as usize
            ]);
        }

        self.config = *config;
    }

    /// Get the GPU buffer size in bytes.
    #[inline]
    pub fn buffer_size(&self) -> u64 {
        self.buffer.size()
    }

    /// Returns `true` if CPU caching is enabled.
    #[inline]
    pub fn has_cpu_cache(&self) -> bool {
        self.cpu_data.is_some()
    }

    /// Get read access to the CPU cache (if enabled).
    pub fn cpu_cache(&self) -> Option<&[ReflectionBufferPixel]> {
        self.cpu_data.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Frame Graph Pass Builders
// ---------------------------------------------------------------------------

/// Create a reflection buffer texture resource for the frame graph.
///
/// Returns a texture descriptor suitable for half-resolution reflection storage.
/// The texture format is RGBA32Float for HDR reflection data.
///
/// # Arguments
/// * `config` - Reflection buffer configuration
///
/// # Returns
/// A wgpu TextureDescriptor for the reflection buffer texture.
pub fn create_reflection_buffer_texture_desc(
    config: &ReflectionBufferConfig,
) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("Reflection Buffer Texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // RGBA32Float for HDR color + roughness + hit distance + technique mask
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

/// Uniforms for the bilateral upscale pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BilateralUpscaleUniforms {
    /// Source (half-res) dimensions.
    pub src_dims: [u32; 2],
    /// Destination (full-res) dimensions.
    pub dst_dims: [u32; 2],
    /// Depth similarity threshold for bilateral weighting.
    pub depth_threshold: f32,
    /// Normal similarity threshold (dot product) for bilateral weighting.
    pub normal_threshold: f32,
    /// Edge sharpness factor (higher = sharper edges).
    pub edge_sharpness: f32,
    /// Padding for 16-byte alignment.
    pub _pad: f32,
}

// Compile-time size assertion: 2*4 + 2*4 + 4 + 4 + 4 + 4 = 32 bytes
const _: () = assert!(std::mem::size_of::<BilateralUpscaleUniforms>() == 32);

impl Default for BilateralUpscaleUniforms {
    fn default() -> Self {
        Self {
            src_dims: [960, 540],
            dst_dims: [1920, 1080],
            depth_threshold: 0.01, // 1% depth difference
            normal_threshold: 0.9, // ~25 degree angle tolerance
            edge_sharpness: 4.0,
            _pad: 0.0,
        }
    }
}

impl BilateralUpscaleUniforms {
    /// Create uniforms for the given source and destination dimensions.
    pub fn new(src_width: u32, src_height: u32, dst_width: u32, dst_height: u32) -> Self {
        Self {
            src_dims: [src_width, src_height],
            dst_dims: [dst_width, dst_height],
            ..Default::default()
        }
    }

    /// Create uniforms from a reflection buffer config.
    pub fn from_config(config: &ReflectionBufferConfig) -> Self {
        let (dst_w, dst_h) = config.full_resolution();
        Self::new(config.width, config.height, dst_w, dst_h)
    }

    /// Set the depth threshold for bilateral weighting.
    pub fn with_depth_threshold(mut self, threshold: f32) -> Self {
        self.depth_threshold = threshold;
        self
    }

    /// Set the normal threshold for bilateral weighting.
    pub fn with_normal_threshold(mut self, threshold: f32) -> Self {
        self.normal_threshold = threshold;
        self
    }

    /// Set the edge sharpness factor.
    pub fn with_edge_sharpness(mut self, sharpness: f32) -> Self {
        self.edge_sharpness = sharpness;
        self
    }
}

// ---------------------------------------------------------------------------
// BilateralUpscalePass
// ---------------------------------------------------------------------------

/// Bilateral upscale pass for reflection buffer.
///
/// Upscales half-resolution reflection data to full resolution using
/// depth-aware and normal-aware bilateral filtering to preserve edges.
pub struct BilateralUpscalePass {
    /// Compute pipeline for the upscale shader.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for shader bindings.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffer for pass parameters.
    uniform_buffer: wgpu::Buffer,
}

impl BilateralUpscalePass {
    /// Create a new bilateral upscale pass.
    ///
    /// # Arguments
    /// * `device` - wgpu device
    /// * `shader_source` - WGSL shader source code
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reflection Bilateral Upscale Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bilateral Upscale Bind Group Layout"),
            entries: &[
                // Uniforms
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
                // Source reflection texture (half-res)
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
                // Full-res depth texture
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
                // Full-res normal texture
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
                // Output texture (full-res)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bilateral Upscale Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bilateral Upscale Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bilateral Upscale Uniforms"),
            contents: bytemuck::bytes_of(&BilateralUpscaleUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
        }
    }

    /// Get the bind group layout for creating bind groups.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the compute pipeline.
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Get the uniform buffer.
    pub fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform_buffer
    }

    /// Update the uniform buffer with new parameters.
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &BilateralUpscaleUniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Create a bind group for a specific set of textures.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        src_reflection: &wgpu::TextureView,
        depth_texture: &wgpu::TextureView,
        normal_texture: &wgpu::TextureView,
        output_texture: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bilateral Upscale Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src_reflection),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(output_texture),
                },
            ],
        })
    }

    /// Dispatch the upscale compute shader.
    ///
    /// # Arguments
    /// * `encoder` - Command encoder to record dispatch
    /// * `bind_group` - Bind group with textures
    /// * `output_width` - Full-res output width
    /// * `output_height` - Full-res output height
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        output_width: u32,
        output_height: u32,
    ) {
        const WORKGROUP_SIZE: u32 = 8;

        let workgroups_x = (output_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (output_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Bilateral Upscale Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- ReflectionBufferPixel Tests ---

    #[test]
    fn test_reflection_buffer_pixel_size_is_32_bytes() {
        assert_eq!(std::mem::size_of::<ReflectionBufferPixel>(), 32);
    }

    #[test]
    fn test_reflection_buffer_pixel_alignment() {
        assert_eq!(std::mem::align_of::<ReflectionBufferPixel>(), 4);
    }

    #[test]
    fn test_pod_zeroable_traits() {
        // Verify Zeroable
        let zeroed: ReflectionBufferPixel = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.color, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed.roughness, 0.0);
        assert_eq!(zeroed.hit_distance, 0.0);
        assert_eq!(zeroed.technique_mask, 0);

        // Verify Pod - can cast to/from bytes
        let pixel = ReflectionBufferPixel::new([1.0, 0.5, 0.25], 0.3, 5.0, TechniqueFlags::SSR);
        let bytes: &[u8] = bytemuck::bytes_of(&pixel);
        assert_eq!(bytes.len(), 32);

        // Verify we can cast back
        let restored: &ReflectionBufferPixel = bytemuck::from_bytes(bytes);
        assert_eq!(restored.color, [1.0, 0.5, 0.25]);
        assert_eq!(restored.roughness, 0.3);
    }

    #[test]
    fn test_reflection_buffer_pixel_default() {
        let pixel = ReflectionBufferPixel::default();
        assert_eq!(pixel.color, [0.0, 0.0, 0.0]);
        assert_eq!(pixel.roughness, 0.0);
        assert_eq!(pixel.hit_distance, -1.0); // Miss
        assert_eq!(pixel.technique_mask, 0);
        assert!(pixel.is_miss());
    }

    #[test]
    fn test_reflection_buffer_pixel_miss() {
        let pixel = ReflectionBufferPixel::miss();
        assert!(pixel.is_miss());
        assert!(!pixel.is_hit());
        assert_eq!(pixel.hit_distance, -1.0);
        assert_eq!(pixel.technique_mask, 0);
    }

    #[test]
    fn test_reflection_buffer_pixel_from_ssr() {
        let pixel = ReflectionBufferPixel::from_ssr([1.0, 0.8, 0.6], 0.2, 3.5);
        assert!(pixel.is_hit());
        assert_eq!(pixel.color, [1.0, 0.8, 0.6]);
        assert_eq!(pixel.roughness, 0.2);
        assert_eq!(pixel.hit_distance, 3.5);
        assert_eq!(pixel.technique_mask, TechniqueFlags::SSR);
        assert_eq!(pixel.primary_technique(), TechniqueFlags::SSR);
    }

    #[test]
    fn test_reflection_buffer_pixel_from_rt() {
        let pixel = ReflectionBufferPixel::from_rt([0.5, 0.5, 0.5], 0.1, 10.0);
        assert!(pixel.is_hit());
        assert_eq!(pixel.technique_mask, TechniqueFlags::RT);
        assert_eq!(pixel.primary_technique(), TechniqueFlags::RT);
    }

    #[test]
    fn test_reflection_buffer_pixel_from_probe() {
        let pixel = ReflectionBufferPixel::from_probe([0.3, 0.4, 0.5], 0.5);
        assert!(pixel.is_hit());
        assert_eq!(pixel.technique_mask, TechniqueFlags::PROBE);
        assert_eq!(pixel.primary_technique(), TechniqueFlags::PROBE);
    }

    #[test]
    fn test_reflection_buffer_pixel_from_env() {
        let pixel = ReflectionBufferPixel::from_env([0.1, 0.2, 0.8]);
        assert!(pixel.is_hit());
        assert_eq!(pixel.color, [0.1, 0.2, 0.8]);
        assert_eq!(pixel.roughness, 1.0);
        assert_eq!(pixel.technique_mask, TechniqueFlags::ENV);
    }

    #[test]
    fn test_reflection_buffer_pixel_technique_string() {
        let pixel = ReflectionBufferPixel::new(
            [1.0, 1.0, 1.0],
            0.0,
            1.0,
            TechniqueFlags::SSR | TechniqueFlags::RT,
        );
        assert_eq!(pixel.technique_string(), "SSR|RT");
    }

    #[test]
    fn test_bytemuck_cast_slice() {
        let pixels = vec![
            ReflectionBufferPixel::from_ssr([1.0, 0.0, 0.0], 0.1, 1.0),
            ReflectionBufferPixel::from_rt([0.0, 1.0, 0.0], 0.2, 2.0),
            ReflectionBufferPixel::from_probe([0.0, 0.0, 1.0], 0.3),
        ];

        let bytes: &[u8] = bytemuck::cast_slice(&pixels);
        assert_eq!(bytes.len(), 96); // 3 * 32 bytes
    }

    // --- TechniqueFlags Tests ---

    #[test]
    fn test_technique_flags_values() {
        assert_eq!(TechniqueFlags::SSR, 1);
        assert_eq!(TechniqueFlags::RT, 2);
        assert_eq!(TechniqueFlags::PROBE, 4);
        assert_eq!(TechniqueFlags::ENV, 8);
    }

    #[test]
    fn test_technique_flags_to_string() {
        assert_eq!(TechniqueFlags::to_string(0), "None");
        assert_eq!(TechniqueFlags::to_string(TechniqueFlags::SSR), "SSR");
        assert_eq!(TechniqueFlags::to_string(TechniqueFlags::RT), "RT");
        assert_eq!(TechniqueFlags::to_string(TechniqueFlags::PROBE), "Probe");
        assert_eq!(TechniqueFlags::to_string(TechniqueFlags::ENV), "Env");
        assert_eq!(
            TechniqueFlags::to_string(TechniqueFlags::SSR | TechniqueFlags::PROBE),
            "SSR|Probe"
        );
        assert_eq!(
            TechniqueFlags::to_string(
                TechniqueFlags::SSR | TechniqueFlags::RT | TechniqueFlags::PROBE | TechniqueFlags::ENV
            ),
            "SSR|RT|Probe|Env"
        );
    }

    #[test]
    fn test_technique_flags_highest_priority() {
        // RT has highest priority
        assert_eq!(
            TechniqueFlags::highest_priority(TechniqueFlags::SSR | TechniqueFlags::RT),
            TechniqueFlags::RT
        );

        // SSR over Probe
        assert_eq!(
            TechniqueFlags::highest_priority(TechniqueFlags::SSR | TechniqueFlags::PROBE),
            TechniqueFlags::SSR
        );

        // Probe over Env
        assert_eq!(
            TechniqueFlags::highest_priority(TechniqueFlags::PROBE | TechniqueFlags::ENV),
            TechniqueFlags::PROBE
        );

        // Env alone
        assert_eq!(
            TechniqueFlags::highest_priority(TechniqueFlags::ENV),
            TechniqueFlags::ENV
        );

        // None
        assert_eq!(TechniqueFlags::highest_priority(0), 0);
    }

    // --- ReflectionBufferConfig Tests ---

    #[test]
    fn test_config_defaults() {
        let config = ReflectionBufferConfig::default();
        assert_eq!(config.width, 960);
        assert_eq!(config.height, 540);
        assert_eq!(config.resolution_scale, 0.5);
    }

    #[test]
    fn test_config_half_res() {
        let config = ReflectionBufferConfig::half_res(1920, 1080);
        assert_eq!(config.width, 960);
        assert_eq!(config.height, 540);
        assert_eq!(config.resolution_scale, 0.5);
    }

    #[test]
    fn test_config_full_res() {
        let config = ReflectionBufferConfig::full_res(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.resolution_scale, 1.0);
    }

    #[test]
    fn test_config_quarter_res() {
        let config = ReflectionBufferConfig::quarter_res(1920, 1080);
        assert_eq!(config.width, 480);
        assert_eq!(config.height, 270);
        assert_eq!(config.resolution_scale, 0.25);
    }

    #[test]
    fn test_config_with_scale() {
        let config = ReflectionBufferConfig::with_scale(1920, 1080, 0.75);
        assert_eq!(config.width, 1440);
        assert_eq!(config.height, 810);
        assert_eq!(config.resolution_scale, 0.75);
    }

    #[test]
    fn test_config_with_scale_clamping() {
        // Scale below minimum (0.1) should clamp
        let config = ReflectionBufferConfig::with_scale(1920, 1080, 0.05);
        assert_eq!(config.resolution_scale, 0.1);

        // Scale above maximum (1.0) should clamp
        let config = ReflectionBufferConfig::with_scale(1920, 1080, 1.5);
        assert_eq!(config.resolution_scale, 1.0);
    }

    #[test]
    fn test_config_pixel_count() {
        let config = ReflectionBufferConfig::half_res(1920, 1080);
        assert_eq!(config.pixel_count(), 960 * 540);
    }

    #[test]
    fn test_config_buffer_size() {
        let config = ReflectionBufferConfig::half_res(1920, 1080);
        let expected_size = (960 * 540 * 32) as u64; // 32 bytes per pixel
        assert_eq!(config.buffer_size(), expected_size);
    }

    #[test]
    fn test_config_full_resolution() {
        let config = ReflectionBufferConfig::half_res(1920, 1080);
        let (full_w, full_h) = config.full_resolution();
        assert_eq!(full_w, 1920);
        assert_eq!(full_h, 1080);
    }

    // --- BilateralUpscaleUniforms Tests ---

    #[test]
    fn test_bilateral_upscale_uniforms_size() {
        assert_eq!(std::mem::size_of::<BilateralUpscaleUniforms>(), 32);
    }

    #[test]
    fn test_bilateral_upscale_uniforms_default() {
        let uniforms = BilateralUpscaleUniforms::default();
        assert_eq!(uniforms.src_dims, [960, 540]);
        assert_eq!(uniforms.dst_dims, [1920, 1080]);
        assert_eq!(uniforms.depth_threshold, 0.01);
        assert_eq!(uniforms.normal_threshold, 0.9);
        assert_eq!(uniforms.edge_sharpness, 4.0);
    }

    #[test]
    fn test_bilateral_upscale_uniforms_new() {
        let uniforms = BilateralUpscaleUniforms::new(480, 270, 1920, 1080);
        assert_eq!(uniforms.src_dims, [480, 270]);
        assert_eq!(uniforms.dst_dims, [1920, 1080]);
    }

    #[test]
    fn test_bilateral_upscale_uniforms_from_config() {
        let config = ReflectionBufferConfig::half_res(1920, 1080);
        let uniforms = BilateralUpscaleUniforms::from_config(&config);
        assert_eq!(uniforms.src_dims, [960, 540]);
        assert_eq!(uniforms.dst_dims, [1920, 1080]);
    }

    #[test]
    fn test_bilateral_upscale_uniforms_builder_pattern() {
        let uniforms = BilateralUpscaleUniforms::default()
            .with_depth_threshold(0.02)
            .with_normal_threshold(0.8)
            .with_edge_sharpness(8.0);

        assert_eq!(uniforms.depth_threshold, 0.02);
        assert_eq!(uniforms.normal_threshold, 0.8);
        assert_eq!(uniforms.edge_sharpness, 8.0);
    }

    #[test]
    fn test_bilateral_upscale_uniforms_pod() {
        let uniforms = BilateralUpscaleUniforms::default();
        let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 32);
    }

    // --- Texture Descriptor Tests ---

    #[test]
    fn test_create_reflection_buffer_texture_desc() {
        let config = ReflectionBufferConfig::half_res(1920, 1080);
        let desc = create_reflection_buffer_texture_desc(&config);

        assert_eq!(desc.size.width, 960);
        assert_eq!(desc.size.height, 540);
        assert_eq!(desc.size.depth_or_array_layers, 1);
        assert_eq!(desc.format, wgpu::TextureFormat::Rgba32Float);
        assert_eq!(desc.mip_level_count, 1);
        assert_eq!(desc.sample_count, 1);
    }

    // --- GPU Buffer Tests (require wgpu device) ---

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
    fn test_reflection_buffer_creation() {
        let (device, _queue) = require_gpu!();
        let config = ReflectionBufferConfig::half_res(1920, 1080);
        let buffer = ReflectionBuffer::new(&device, &config);

        assert_eq!(buffer.width(), 960);
        assert_eq!(buffer.height(), 540);
        assert_eq!(buffer.pixel_count(), 960 * 540);
        assert_eq!(buffer.buffer_size(), config.buffer_size());
        assert!(!buffer.has_cpu_cache());
    }

    #[test]
    fn test_reflection_buffer_with_cpu_cache() {
        let (device, _queue) = require_gpu!();
        let config = ReflectionBufferConfig::half_res(1920, 1080);
        let buffer = ReflectionBuffer::new_with_cpu_cache(&device, &config);

        assert!(buffer.has_cpu_cache());
        assert!(buffer.cpu_cache().is_some());
        assert_eq!(buffer.cpu_cache().unwrap().len(), config.pixel_count() as usize);
    }

    #[test]
    fn test_reflection_buffer_resize() {
        let (device, _queue) = require_gpu!();
        let config1 = ReflectionBufferConfig::half_res(1920, 1080);
        let config2 = ReflectionBufferConfig::quarter_res(1920, 1080);

        let mut buffer = ReflectionBuffer::new(&device, &config1);
        assert_eq!(buffer.width(), 960);

        buffer.resize(&device, &config2);
        assert_eq!(buffer.width(), 480);
        assert_eq!(buffer.height(), 270);
    }

    #[test]
    fn test_reflection_buffer_clear() {
        let (device, queue) = require_gpu!();
        let config = ReflectionBufferConfig::with_scale(100, 100, 1.0);
        let mut buffer = ReflectionBuffer::new_with_cpu_cache(&device, &config);

        buffer.clear(&queue);

        // All pixels should be miss
        let cache = buffer.cpu_cache().unwrap();
        for pixel in cache {
            assert!(pixel.is_miss());
        }
    }
}
