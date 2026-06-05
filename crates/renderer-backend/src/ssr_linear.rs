//! Linear SSR ray marching fallback for screen-space reflections (T-GIR-P4.3).
//!
//! This module provides a simpler linear SSR fallback for when HiZ isn't available
//! or for quality comparison. Linear ray marching steps through screen-space at a
//! fixed stride, making it predictable but potentially slower than HiZ-accelerated
//! marching.
//!
//! # Overview
//!
//! The linear ray marcher:
//! 1. Steps along the reflected ray in screen-space at a fixed stride
//! 2. Tests depth at each step against the depth buffer
//! 3. When an intersection is detected, performs binary refinement
//! 4. Applies configurable fade functions for edge/distance/roughness falloff
//!
//! # Usage
//!
//! ```ignore
//! // Configure linear SSR
//! let config = SSRLinearConfig::default();
//! let fade = SSRFadeConfig::default();
//!
//! // Create the compute pass
//! let pass = SSRLinearPass::new(&device, &config, &fade);
//!
//! // Dispatch each frame
//! pass.dispatch(&mut encoder, &bind_group, width, height);
//! ```
//!
//! # Fade Functions
//!
//! The module provides three fade functions that can be combined:
//!
//! - **Edge fade**: Reduces reflection strength near screen edges where rays exit
//! - **Distance fade**: Reduces reflection strength for distant hits
//! - **Roughness fade**: Reduces reflection strength for rough surfaces
//!
//! All fades use smooth Hermite interpolation (smoothstep) for natural falloff.

use bytemuck::{Pod, Zeroable};
use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size.
pub const WORKGROUP_SIZE: u32 = 8;

/// Default maximum ray marching steps.
pub const DEFAULT_MAX_STEPS: u32 = 64;

/// Default stride in pixels.
pub const DEFAULT_STRIDE_PIXELS: f32 = 4.0;

/// Default depth thickness threshold.
pub const DEFAULT_THICKNESS: f32 = 0.1;

/// Default temporal jitter amount.
pub const DEFAULT_JITTER: f32 = 0.0;

/// Number of binary refinement iterations.
pub const BINARY_REFINE_ITERATIONS: u32 = 4;

// ---------------------------------------------------------------------------
// SSRLinearConfig
// ---------------------------------------------------------------------------

/// Configuration for linear SSR ray marching.
///
/// Controls the quality/performance tradeoff for linear ray marching.
/// Higher step counts and lower strides produce better quality at the
/// cost of performance.
///
/// # Memory Layout
///
/// 16 bytes total, GPU-aligned:
///
/// | Offset | Field        | Size    | Description                    |
/// |--------|--------------|---------|--------------------------------|
/// | 0      | max_steps    | 4 bytes | Maximum ray marching steps     |
/// | 4      | stride_pixels| 4 bytes | Step size in pixels            |
/// | 8      | thickness    | 4 bytes | Depth comparison threshold     |
/// | 12     | jitter       | 4 bytes | Temporal jitter factor         |
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SSRLinearConfig {
    /// Maximum number of ray marching steps (64-256 recommended).
    ///
    /// Higher values increase quality but cost more performance.
    /// Use 64 for fast preview, 128 for balanced, 256 for high quality.
    pub max_steps: u32,

    /// Step size in screen pixels (4-8 recommended).
    ///
    /// Smaller strides catch thin objects but require more steps.
    /// Larger strides are faster but may miss thin geometry.
    pub stride_pixels: f32,

    /// Depth thickness threshold for hit detection.
    ///
    /// Controls how thick surfaces appear for intersection testing.
    /// Too small: rays pass through thin surfaces.
    /// Too large: false positives from depth discontinuities.
    pub thickness: f32,

    /// Temporal jitter factor (0.0-1.0).
    ///
    /// Randomizes ray start position per-frame for temporal stability.
    /// Combined with TAA, this reduces stepping artifacts.
    pub jitter: f32,
}

// Compile-time size assertion: 4 + 4 + 4 + 4 = 16 bytes
const _: () = assert!(mem::size_of::<SSRLinearConfig>() == 16);
const _: () = assert!(mem::align_of::<SSRLinearConfig>() == 4);

impl Default for SSRLinearConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            stride_pixels: DEFAULT_STRIDE_PIXELS,
            thickness: DEFAULT_THICKNESS,
            jitter: DEFAULT_JITTER,
        }
    }
}

impl SSRLinearConfig {
    /// Create a new linear SSR configuration.
    ///
    /// # Arguments
    ///
    /// * `max_steps` - Maximum ray marching steps (64-256 recommended)
    /// * `stride_pixels` - Step size in pixels (4-8 recommended)
    /// * `thickness` - Depth threshold for hit detection
    /// * `jitter` - Temporal jitter factor (0.0-1.0)
    pub fn new(max_steps: u32, stride_pixels: f32, thickness: f32, jitter: f32) -> Self {
        Self {
            max_steps,
            stride_pixels,
            thickness,
            jitter,
        }
    }

    /// Create a fast/preview quality configuration.
    ///
    /// Uses fewer steps and larger stride for real-time preview.
    pub fn fast() -> Self {
        Self {
            max_steps: 32,
            stride_pixels: 8.0,
            thickness: 0.15,
            jitter: 0.5,
        }
    }

    /// Create a balanced quality configuration.
    ///
    /// Good balance between quality and performance.
    pub fn balanced() -> Self {
        Self {
            max_steps: 64,
            stride_pixels: 4.0,
            thickness: 0.1,
            jitter: 0.5,
        }
    }

    /// Create a high quality configuration.
    ///
    /// Maximum quality at the cost of performance.
    pub fn high_quality() -> Self {
        Self {
            max_steps: 128,
            stride_pixels: 2.0,
            thickness: 0.05,
            jitter: 0.25,
        }
    }

    /// Create an ultra quality configuration.
    ///
    /// For offline rendering or benchmarking.
    pub fn ultra() -> Self {
        Self {
            max_steps: 256,
            stride_pixels: 1.0,
            thickness: 0.02,
            jitter: 0.1,
        }
    }

    /// Set the maximum steps with builder pattern.
    pub fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_steps = steps;
        self
    }

    /// Set the stride with builder pattern.
    pub fn with_stride(mut self, stride_pixels: f32) -> Self {
        self.stride_pixels = stride_pixels;
        self
    }

    /// Set the thickness with builder pattern.
    pub fn with_thickness(mut self, thickness: f32) -> Self {
        self.thickness = thickness;
        self
    }

    /// Set the jitter with builder pattern.
    pub fn with_jitter(mut self, jitter: f32) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// Validate the configuration, clamping to valid ranges.
    pub fn validate(self) -> Self {
        Self {
            max_steps: self.max_steps.clamp(1, 1024),
            stride_pixels: self.stride_pixels.clamp(0.5, 64.0),
            thickness: self.thickness.clamp(0.001, 10.0),
            jitter: self.jitter.clamp(0.0, 1.0),
        }
    }

    /// Estimate the approximate cost of this configuration.
    ///
    /// Returns a relative cost factor (higher = more expensive).
    pub fn estimated_cost(&self) -> f32 {
        // Cost is roughly proportional to steps / stride
        (self.max_steps as f32) / self.stride_pixels.max(0.5)
    }
}

// ---------------------------------------------------------------------------
// SSRFadeConfig
// ---------------------------------------------------------------------------

/// Fade parameters for edge/distance/roughness falloff.
///
/// Controls how reflections fade out at screen edges, with distance,
/// and for rough surfaces. All fade functions use smooth Hermite
/// interpolation (smoothstep) for natural-looking falloff.
///
/// # Memory Layout
///
/// 32 bytes total, GPU-aligned:
///
/// | Offset | Field               | Size    | Description              |
/// |--------|---------------------|---------|--------------------------|
/// | 0      | edge_fade_start     | 4 bytes | Start screen edge fade   |
/// | 4      | edge_fade_end       | 4 bytes | End screen edge fade     |
/// | 8      | distance_fade_start | 4 bytes | Start distance fade      |
/// | 12     | distance_fade_end   | 4 bytes | End distance fade        |
/// | 16     | roughness_fade_start| 4 bytes | Start roughness fade     |
/// | 20     | roughness_fade_end  | 4 bytes | End roughness fade       |
/// | 24     | _pad                | 8 bytes | Padding for alignment    |
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SSRFadeConfig {
    /// Screen edge fade start (0.0-1.0, as fraction from center).
    ///
    /// At this distance from screen center, fade begins (1.0 = full strength).
    /// Default: 0.8 (fade starts 80% from center to edge).
    pub edge_fade_start: f32,

    /// Screen edge fade end (0.0-1.0, as fraction from center).
    ///
    /// At this distance from screen center, fade is complete (0.0 = no reflection).
    /// Default: 1.0 (fade completes at screen edge).
    pub edge_fade_end: f32,

    /// Distance fade start in world units (meters).
    ///
    /// Hits closer than this distance have full reflection strength.
    /// Default: 50.0 meters.
    pub distance_fade_start: f32,

    /// Distance fade end in world units (meters).
    ///
    /// Hits beyond this distance have no reflection.
    /// Default: 100.0 meters.
    pub distance_fade_end: f32,

    /// Roughness fade start (0.0-1.0).
    ///
    /// Surfaces smoother than this have full reflection strength.
    /// Default: 0.5.
    pub roughness_fade_start: f32,

    /// Roughness fade end (0.0-1.0).
    ///
    /// Surfaces rougher than this have no reflection.
    /// Default: 0.8.
    pub roughness_fade_end: f32,

    /// Padding for 32-byte alignment.
    pub _pad: [f32; 2],
}

// Compile-time size assertion: 4 * 6 + 8 = 32 bytes
const _: () = assert!(mem::size_of::<SSRFadeConfig>() == 32);
const _: () = assert!(mem::align_of::<SSRFadeConfig>() == 4);

impl Default for SSRFadeConfig {
    fn default() -> Self {
        Self {
            edge_fade_start: 0.8,
            edge_fade_end: 1.0,
            distance_fade_start: 50.0,
            distance_fade_end: 100.0,
            roughness_fade_start: 0.5,
            roughness_fade_end: 0.8,
            _pad: [0.0, 0.0],
        }
    }
}

impl SSRFadeConfig {
    /// Create a new fade configuration with all parameters.
    pub fn new(
        edge_fade_start: f32,
        edge_fade_end: f32,
        distance_fade_start: f32,
        distance_fade_end: f32,
        roughness_fade_start: f32,
        roughness_fade_end: f32,
    ) -> Self {
        Self {
            edge_fade_start,
            edge_fade_end,
            distance_fade_start,
            distance_fade_end,
            roughness_fade_start,
            roughness_fade_end,
            _pad: [0.0, 0.0],
        }
    }

    /// Create a configuration with no fade (full reflections everywhere).
    pub fn no_fade() -> Self {
        Self {
            edge_fade_start: 1.0,
            edge_fade_end: 1.0,
            distance_fade_start: 10000.0,
            distance_fade_end: 10000.0,
            roughness_fade_start: 1.0,
            roughness_fade_end: 1.0,
            _pad: [0.0, 0.0],
        }
    }

    /// Create an aggressive fade configuration for stylized rendering.
    pub fn aggressive() -> Self {
        Self {
            edge_fade_start: 0.6,
            edge_fade_end: 0.9,
            distance_fade_start: 20.0,
            distance_fade_end: 50.0,
            roughness_fade_start: 0.3,
            roughness_fade_end: 0.6,
            _pad: [0.0, 0.0],
        }
    }

    /// Create a subtle fade configuration for maximum reflection coverage.
    pub fn subtle() -> Self {
        Self {
            edge_fade_start: 0.9,
            edge_fade_end: 1.0,
            distance_fade_start: 100.0,
            distance_fade_end: 200.0,
            roughness_fade_start: 0.7,
            roughness_fade_end: 0.95,
            _pad: [0.0, 0.0],
        }
    }

    /// Set edge fade parameters with builder pattern.
    pub fn with_edge_fade(mut self, start: f32, end: f32) -> Self {
        self.edge_fade_start = start.clamp(0.0, 1.0);
        self.edge_fade_end = end.clamp(0.0, 1.0);
        self
    }

    /// Set distance fade parameters with builder pattern.
    pub fn with_distance_fade(mut self, start: f32, end: f32) -> Self {
        self.distance_fade_start = start.max(0.0);
        self.distance_fade_end = end.max(start);
        self
    }

    /// Set roughness fade parameters with builder pattern.
    pub fn with_roughness_fade(mut self, start: f32, end: f32) -> Self {
        self.roughness_fade_start = start.clamp(0.0, 1.0);
        self.roughness_fade_end = end.clamp(0.0, 1.0);
        self
    }

    /// Validate the configuration, ensuring start < end.
    pub fn validate(self) -> Self {
        // First clamp start values to valid ranges
        let edge_start = self.edge_fade_start.clamp(0.0, 1.0);
        let distance_start = self.distance_fade_start.max(0.0);
        let roughness_start = self.roughness_fade_start.clamp(0.0, 1.0);

        // Then clamp end values to be at least as large as start values
        Self {
            edge_fade_start: edge_start,
            edge_fade_end: self.edge_fade_end.clamp(0.0, 1.0).max(edge_start),
            distance_fade_start: distance_start,
            distance_fade_end: self.distance_fade_end.max(0.0).max(distance_start),
            roughness_fade_start: roughness_start,
            roughness_fade_end: self.roughness_fade_end.clamp(0.0, 1.0).max(roughness_start),
            _pad: [0.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-side Fade Functions
// ---------------------------------------------------------------------------

/// Hermite interpolation (smoothstep).
///
/// Returns 0 for x <= edge0, 1 for x >= edge1,
/// and smooth interpolation between.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Compute edge fade based on screen UV.
///
/// Returns 1.0 at screen center, fading to 0.0 at edges.
///
/// # Arguments
///
/// * `screen_uv` - Screen-space UV coordinates (0.0-1.0, 0.0-1.0)
/// * `config` - Fade configuration
///
/// # Returns
///
/// Fade factor in range [0.0, 1.0]
pub fn compute_edge_fade(screen_uv: [f32; 2], config: &SSRFadeConfig) -> f32 {
    // Distance from center (0.0 at center, ~0.707 at corners)
    let centered_uv = [screen_uv[0] - 0.5, screen_uv[1] - 0.5];
    let distance_from_center = (centered_uv[0] * centered_uv[0] + centered_uv[1] * centered_uv[1]).sqrt() * 2.0;

    // Map to [0, 1] where 0 = edge_fade_end, 1 = edge_fade_start
    1.0 - smoothstep(config.edge_fade_start, config.edge_fade_end, distance_from_center)
}

/// Compute distance fade based on hit distance.
///
/// Returns 1.0 for close hits, fading to 0.0 for distant hits.
///
/// # Arguments
///
/// * `hit_distance` - World-space distance to reflection hit
/// * `config` - Fade configuration
///
/// # Returns
///
/// Fade factor in range [0.0, 1.0]
pub fn compute_distance_fade(hit_distance: f32, config: &SSRFadeConfig) -> f32 {
    1.0 - smoothstep(config.distance_fade_start, config.distance_fade_end, hit_distance)
}

/// Compute roughness fade based on surface roughness.
///
/// Returns 1.0 for smooth surfaces, fading to 0.0 for rough surfaces.
///
/// # Arguments
///
/// * `roughness` - Surface roughness (0.0-1.0)
/// * `config` - Fade configuration
///
/// # Returns
///
/// Fade factor in range [0.0, 1.0]
pub fn compute_roughness_fade(roughness: f32, config: &SSRFadeConfig) -> f32 {
    1.0 - smoothstep(config.roughness_fade_start, config.roughness_fade_end, roughness)
}

/// Compute combined fade factor from all sources.
///
/// Multiplies edge, distance, and roughness fades together.
///
/// # Arguments
///
/// * `screen_uv` - Screen-space UV coordinates
/// * `hit_distance` - World-space distance to hit
/// * `roughness` - Surface roughness
/// * `config` - Fade configuration
///
/// # Returns
///
/// Combined fade factor in range [0.0, 1.0]
pub fn compute_combined_fade(
    screen_uv: [f32; 2],
    hit_distance: f32,
    roughness: f32,
    config: &SSRFadeConfig,
) -> f32 {
    let edge = compute_edge_fade(screen_uv, config);
    let distance = compute_distance_fade(hit_distance, config);
    let rough = compute_roughness_fade(roughness, config);
    edge * distance * rough
}

// ---------------------------------------------------------------------------
// SSRLinearUniforms
// ---------------------------------------------------------------------------

/// Combined uniforms for SSR linear pass.
///
/// Combines SSRLinearConfig and SSRFadeConfig into a single GPU-uploadable struct.
///
/// # Memory Layout
///
/// 64 bytes total (16 + 32 + 16 padding for 64-byte alignment):
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SSRLinearUniforms {
    /// Linear ray marching configuration.
    pub linear_config: SSRLinearConfig,
    /// Fade configuration.
    pub fade_config: SSRFadeConfig,
    /// Screen dimensions (width, height).
    pub screen_size: [u32; 2],
    /// Frame index for temporal jitter.
    pub frame_index: u32,
    /// Padding for alignment.
    pub _pad: u32,
}

// Compile-time size assertion: 16 + 32 + 8 + 4 + 4 = 64 bytes
const _: () = assert!(mem::size_of::<SSRLinearUniforms>() == 64);

impl Default for SSRLinearUniforms {
    fn default() -> Self {
        Self {
            linear_config: SSRLinearConfig::default(),
            fade_config: SSRFadeConfig::default(),
            screen_size: [1920, 1080],
            frame_index: 0,
            _pad: 0,
        }
    }
}

impl SSRLinearUniforms {
    /// Create uniforms for the given screen dimensions.
    pub fn new(
        linear_config: SSRLinearConfig,
        fade_config: SSRFadeConfig,
        width: u32,
        height: u32,
        frame_index: u32,
    ) -> Self {
        Self {
            linear_config,
            fade_config,
            screen_size: [width, height],
            frame_index,
            _pad: 0,
        }
    }

    /// Create uniforms with default configs for the given dimensions.
    pub fn for_screen(width: u32, height: u32) -> Self {
        Self {
            screen_size: [width, height],
            ..Default::default()
        }
    }

    /// Update frame index for next frame.
    pub fn next_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

// ---------------------------------------------------------------------------
// HitResult
// ---------------------------------------------------------------------------

/// Result of a linear ray march.
///
/// Contains information about whether a hit was found and its properties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitResult {
    /// Whether the ray hit geometry.
    pub hit: bool,
    /// Screen-space UV of the hit point (0.0-1.0).
    pub screen_uv: [f32; 2],
    /// World-space distance to the hit.
    pub hit_distance: f32,
    /// Number of steps taken.
    pub steps: u32,
}

impl Default for HitResult {
    fn default() -> Self {
        Self {
            hit: false,
            screen_uv: [0.0, 0.0],
            hit_distance: -1.0,
            steps: 0,
        }
    }
}

impl HitResult {
    /// Create a miss result.
    pub fn miss(steps: u32) -> Self {
        Self {
            hit: false,
            screen_uv: [0.0, 0.0],
            hit_distance: -1.0,
            steps,
        }
    }

    /// Create a hit result.
    pub fn new(screen_uv: [f32; 2], hit_distance: f32, steps: u32) -> Self {
        Self {
            hit: true,
            screen_uv,
            hit_distance,
            steps,
        }
    }
}

// ---------------------------------------------------------------------------
// SSRLinearPass
// ---------------------------------------------------------------------------

/// Linear SSR ray marching compute pass.
///
/// Performs screen-space reflections using simple linear ray marching.
/// This is a fallback for when HiZ isn't available or for quality comparison.
pub struct SSRLinearPass {
    /// Compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl SSRLinearPass {
    /// Create a new linear SSR pass.
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

    /// Get the bind group layout.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get the compute pipeline.
    #[inline]
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Create a bind group for the SSR pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `uniforms_buffer` - Buffer containing SSRLinearUniforms.
    /// * `depth_texture` - Scene depth buffer.
    /// * `normal_texture` - Scene normal buffer (world-space).
    /// * `color_texture` - Scene color buffer to sample.
    /// * `output_texture` - Output reflection buffer (storage).
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        uniforms_buffer: &wgpu::Buffer,
        depth_texture: &wgpu::TextureView,
        normal_texture: &wgpu::TextureView,
        color_texture: &wgpu::TextureView,
        output_texture: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr_linear_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(normal_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(color_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(output_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Dispatch the SSR linear compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder.
    /// * `bind_group` - The bind group.
    /// * `width` - Output width.
    /// * `height` - Output height.
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
            label: Some("ssr_linear_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Create the bind group layout.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr_linear_bind_group_layout"),
            entries: &[
                // Binding 0: Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            mem::size_of::<SSRLinearUniforms>() as u64,
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
                // Binding 2: Normal texture
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
                // Binding 3: Color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 4: Output texture (storage)
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
                // Binding 5: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    /// Create the compute pipeline.
    fn create_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader_source = include_str!("../shaders/ssr_ray_march_linear.comp.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssr_linear_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssr_linear_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssr_linear_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "ssr_linear_march",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Frame Graph Integration
// ---------------------------------------------------------------------------

use crate::frame_graph::{
    DispatchSource, IrPass, PassIndex, ResourceAccessSet, ResourceHandle, ViewType,
};

/// Create a frame graph pass for linear SSR.
///
/// # Arguments
///
/// * `index` - Pass index in the frame graph.
/// * `name` - Pass name.
/// * `depth_handle` - Resource handle for the depth buffer.
/// * `normal_handle` - Resource handle for the normal buffer.
/// * `color_handle` - Resource handle for the color buffer.
/// * `output_handle` - Resource handle for the output reflection buffer.
/// * `width` - Output width.
/// * `height` - Output height.
pub fn create_ssr_linear_pass(
    index: PassIndex,
    name: &str,
    depth_handle: ResourceHandle,
    normal_handle: ResourceHandle,
    color_handle: ResourceHandle,
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
        reads: vec![depth_handle, normal_handle, color_handle],
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
    // SSRLinearConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssr_linear_config_size() {
        assert_eq!(mem::size_of::<SSRLinearConfig>(), 16);
    }

    #[test]
    fn test_ssr_linear_config_alignment() {
        assert_eq!(mem::align_of::<SSRLinearConfig>(), 4);
    }

    #[test]
    fn test_ssr_linear_config_pod() {
        let config = SSRLinearConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_ssr_linear_config_default() {
        let config = SSRLinearConfig::default();
        assert_eq!(config.max_steps, DEFAULT_MAX_STEPS);
        assert_eq!(config.stride_pixels, DEFAULT_STRIDE_PIXELS);
        assert_eq!(config.thickness, DEFAULT_THICKNESS);
        assert_eq!(config.jitter, DEFAULT_JITTER);
    }

    #[test]
    fn test_ssr_linear_config_new() {
        let config = SSRLinearConfig::new(128, 2.0, 0.05, 0.5);
        assert_eq!(config.max_steps, 128);
        assert_eq!(config.stride_pixels, 2.0);
        assert_eq!(config.thickness, 0.05);
        assert_eq!(config.jitter, 0.5);
    }

    #[test]
    fn test_ssr_linear_config_presets() {
        let fast = SSRLinearConfig::fast();
        assert_eq!(fast.max_steps, 32);
        assert_eq!(fast.stride_pixels, 8.0);

        let balanced = SSRLinearConfig::balanced();
        assert_eq!(balanced.max_steps, 64);
        assert_eq!(balanced.stride_pixels, 4.0);

        let high = SSRLinearConfig::high_quality();
        assert_eq!(high.max_steps, 128);
        assert_eq!(high.stride_pixels, 2.0);

        let ultra = SSRLinearConfig::ultra();
        assert_eq!(ultra.max_steps, 256);
        assert_eq!(ultra.stride_pixels, 1.0);
    }

    #[test]
    fn test_ssr_linear_config_builder() {
        let config = SSRLinearConfig::default()
            .with_max_steps(128)
            .with_stride(2.5)
            .with_thickness(0.08)
            .with_jitter(0.3);

        assert_eq!(config.max_steps, 128);
        assert_eq!(config.stride_pixels, 2.5);
        assert_eq!(config.thickness, 0.08);
        assert_eq!(config.jitter, 0.3);
    }

    #[test]
    fn test_ssr_linear_config_jitter_clamped() {
        let config = SSRLinearConfig::default().with_jitter(1.5);
        assert_eq!(config.jitter, 1.0);

        let config = SSRLinearConfig::default().with_jitter(-0.5);
        assert_eq!(config.jitter, 0.0);
    }

    #[test]
    fn test_ssr_linear_config_validate() {
        let invalid = SSRLinearConfig {
            max_steps: 10000,
            stride_pixels: 0.1,
            thickness: 100.0,
            jitter: 2.0,
        };

        let valid = invalid.validate();
        assert_eq!(valid.max_steps, 1024);
        assert_eq!(valid.stride_pixels, 0.5);
        assert_eq!(valid.thickness, 10.0);
        assert_eq!(valid.jitter, 1.0);
    }

    #[test]
    fn test_ssr_linear_config_estimated_cost() {
        let fast = SSRLinearConfig::fast();
        let ultra = SSRLinearConfig::ultra();

        // Ultra should be more expensive than fast
        assert!(ultra.estimated_cost() > fast.estimated_cost());
    }

    // -----------------------------------------------------------------------
    // SSRFadeConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssr_fade_config_size() {
        assert_eq!(mem::size_of::<SSRFadeConfig>(), 32);
    }

    #[test]
    fn test_ssr_fade_config_alignment() {
        assert_eq!(mem::align_of::<SSRFadeConfig>(), 4);
    }

    #[test]
    fn test_ssr_fade_config_pod() {
        let config = SSRFadeConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_ssr_fade_config_default() {
        let config = SSRFadeConfig::default();
        assert_eq!(config.edge_fade_start, 0.8);
        assert_eq!(config.edge_fade_end, 1.0);
        assert_eq!(config.distance_fade_start, 50.0);
        assert_eq!(config.distance_fade_end, 100.0);
        assert_eq!(config.roughness_fade_start, 0.5);
        assert_eq!(config.roughness_fade_end, 0.8);
    }

    #[test]
    fn test_ssr_fade_config_no_fade() {
        let config = SSRFadeConfig::no_fade();
        assert_eq!(config.edge_fade_start, 1.0);
        assert_eq!(config.edge_fade_end, 1.0);
        assert_eq!(config.distance_fade_start, 10000.0);
        assert_eq!(config.roughness_fade_start, 1.0);
    }

    #[test]
    fn test_ssr_fade_config_presets() {
        let aggressive = SSRFadeConfig::aggressive();
        assert!(aggressive.edge_fade_start < 0.8);
        assert!(aggressive.distance_fade_end < 100.0);

        let subtle = SSRFadeConfig::subtle();
        assert!(subtle.edge_fade_start > 0.8);
        assert!(subtle.distance_fade_end > 100.0);
    }

    #[test]
    fn test_ssr_fade_config_builder() {
        let config = SSRFadeConfig::default()
            .with_edge_fade(0.7, 0.95)
            .with_distance_fade(30.0, 80.0)
            .with_roughness_fade(0.4, 0.7);

        assert_eq!(config.edge_fade_start, 0.7);
        assert_eq!(config.edge_fade_end, 0.95);
        assert_eq!(config.distance_fade_start, 30.0);
        assert_eq!(config.distance_fade_end, 80.0);
        assert_eq!(config.roughness_fade_start, 0.4);
        assert_eq!(config.roughness_fade_end, 0.7);
    }

    #[test]
    fn test_ssr_fade_config_validate() {
        let invalid = SSRFadeConfig {
            edge_fade_start: 1.5,
            edge_fade_end: 0.5,
            distance_fade_start: -10.0,
            distance_fade_end: -20.0,
            roughness_fade_start: 0.9,
            roughness_fade_end: 0.3,
            _pad: [0.0, 0.0],
        };

        let valid = invalid.validate();
        assert!(valid.edge_fade_start <= 1.0);
        assert!(valid.edge_fade_end >= valid.edge_fade_start);
        assert!(valid.distance_fade_start >= 0.0);
        assert!(valid.distance_fade_end >= valid.distance_fade_start);
        assert!(valid.roughness_fade_end >= valid.roughness_fade_start);
    }

    // -----------------------------------------------------------------------
    // Fade function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_smoothstep_boundaries() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 0.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 1.0), 1.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
    }

    #[test]
    fn test_smoothstep_midpoint() {
        let mid = smoothstep(0.0, 1.0, 0.5);
        assert!((mid - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_smoothstep_monotonic() {
        // Smoothstep should be monotonically increasing
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let x = i as f32 / 100.0;
            let val = smoothstep(0.0, 1.0, x);
            assert!(val >= prev);
            prev = val;
        }
    }

    #[test]
    fn test_edge_fade_center() {
        let config = SSRFadeConfig::default();
        let fade = compute_edge_fade([0.5, 0.5], &config);
        assert!((fade - 1.0).abs() < 0.01, "Center should have full fade");
    }

    #[test]
    fn test_edge_fade_corner() {
        let config = SSRFadeConfig::default();
        let fade = compute_edge_fade([0.0, 0.0], &config);
        assert!(fade < 0.5, "Corner should have reduced fade");
    }

    #[test]
    fn test_edge_fade_edge() {
        let config = SSRFadeConfig::default();
        let fade = compute_edge_fade([0.0, 0.5], &config);
        assert!(fade < 1.0, "Edge should have some fade");
    }

    #[test]
    fn test_distance_fade_close() {
        let config = SSRFadeConfig::default();
        let fade = compute_distance_fade(10.0, &config);
        assert_eq!(fade, 1.0, "Close distances should have full fade");
    }

    #[test]
    fn test_distance_fade_far() {
        let config = SSRFadeConfig::default();
        let fade = compute_distance_fade(200.0, &config);
        assert_eq!(fade, 0.0, "Far distances should have no fade");
    }

    #[test]
    fn test_distance_fade_mid() {
        let config = SSRFadeConfig::default();
        let fade = compute_distance_fade(75.0, &config);
        assert!(fade > 0.0 && fade < 1.0, "Mid distance should partial fade");
    }

    #[test]
    fn test_roughness_fade_smooth() {
        let config = SSRFadeConfig::default();
        let fade = compute_roughness_fade(0.0, &config);
        assert_eq!(fade, 1.0, "Smooth surfaces should have full fade");
    }

    #[test]
    fn test_roughness_fade_rough() {
        let config = SSRFadeConfig::default();
        let fade = compute_roughness_fade(1.0, &config);
        assert_eq!(fade, 0.0, "Rough surfaces should have no fade");
    }

    #[test]
    fn test_roughness_fade_mid() {
        let config = SSRFadeConfig::default();
        let fade = compute_roughness_fade(0.65, &config);
        assert!(fade > 0.0 && fade < 1.0, "Mid roughness should partial fade");
    }

    #[test]
    fn test_combined_fade_all_max() {
        let config = SSRFadeConfig::no_fade();
        let fade = compute_combined_fade([0.5, 0.5], 10.0, 0.0, &config);
        assert!((fade - 1.0).abs() < 0.01, "No fade config should give full fade");
    }

    #[test]
    fn test_combined_fade_multiplicative() {
        let config = SSRFadeConfig::default();
        let edge = compute_edge_fade([0.5, 0.5], &config);
        let distance = compute_distance_fade(75.0, &config);
        let roughness = compute_roughness_fade(0.65, &config);
        let combined = compute_combined_fade([0.5, 0.5], 75.0, 0.65, &config);

        let expected = edge * distance * roughness;
        assert!((combined - expected).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // SSRLinearUniforms tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssr_linear_uniforms_size() {
        assert_eq!(mem::size_of::<SSRLinearUniforms>(), 64);
    }

    #[test]
    fn test_ssr_linear_uniforms_pod() {
        let uniforms = SSRLinearUniforms::default();
        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn test_ssr_linear_uniforms_new() {
        let uniforms = SSRLinearUniforms::new(
            SSRLinearConfig::balanced(),
            SSRFadeConfig::aggressive(),
            2560,
            1440,
            42,
        );

        assert_eq!(uniforms.screen_size, [2560, 1440]);
        assert_eq!(uniforms.frame_index, 42);
    }

    #[test]
    fn test_ssr_linear_uniforms_for_screen() {
        let uniforms = SSRLinearUniforms::for_screen(3840, 2160);
        assert_eq!(uniforms.screen_size, [3840, 2160]);
    }

    #[test]
    fn test_ssr_linear_uniforms_next_frame() {
        let mut uniforms = SSRLinearUniforms::default();
        assert_eq!(uniforms.frame_index, 0);
        uniforms.next_frame();
        assert_eq!(uniforms.frame_index, 1);
        uniforms.next_frame();
        assert_eq!(uniforms.frame_index, 2);
    }

    // -----------------------------------------------------------------------
    // HitResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hit_result_miss() {
        let result = HitResult::miss(64);
        assert!(!result.hit);
        assert_eq!(result.hit_distance, -1.0);
        assert_eq!(result.steps, 64);
    }

    #[test]
    fn test_hit_result_hit() {
        let result = HitResult::new([0.5, 0.5], 25.0, 32);
        assert!(result.hit);
        assert_eq!(result.screen_uv, [0.5, 0.5]);
        assert_eq!(result.hit_distance, 25.0);
        assert_eq!(result.steps, 32);
    }

    #[test]
    fn test_hit_result_default() {
        let result = HitResult::default();
        assert!(!result.hit);
    }

    // -----------------------------------------------------------------------
    // Frame graph integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_ssr_linear_pass() {
        use crate::frame_graph::PassType;

        let pass = create_ssr_linear_pass(
            PassIndex(5),
            "ssr_linear",
            ResourceHandle(0),
            ResourceHandle(1),
            ResourceHandle(2),
            ResourceHandle(3),
            1920,
            1080,
        );

        assert_eq!(pass.name, "ssr_linear");
        assert_eq!(pass.pass_type, PassType::Compute);
        assert_eq!(pass.access_set.reads.len(), 3);
        assert_eq!(pass.access_set.writes.len(), 1);

        // Verify workgroup calculation
        if let Some(DispatchSource::Direct {
            group_count_x,
            group_count_y,
            group_count_z,
        }) = pass.dispatch_source
        {
            assert_eq!(group_count_x, (1920 + 7) / 8); // 240
            assert_eq!(group_count_y, (1080 + 7) / 8); // 135
            assert_eq!(group_count_z, 1);
        } else {
            panic!("Expected Direct dispatch");
        }
    }

    // -----------------------------------------------------------------------
    // Shader validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ssr_linear_shader_parses() {
        let shader_source = include_str!("../shaders/ssr_ray_march_linear.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSR linear shader should parse without errors");

        let entry_point = module
            .entry_points
            .iter()
            .find(|ep| ep.name == "ssr_linear_march");
        assert!(entry_point.is_some(), "Should have ssr_linear_march entry point");

        let ep = entry_point.unwrap();
        assert_eq!(ep.stage, naga::ShaderStage::Compute);
        assert_eq!(ep.workgroup_size, [8, 8, 1]);
    }

    #[test]
    fn test_ssr_linear_shader_validates() {
        let shader_source = include_str!("../shaders/ssr_ray_march_linear.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSR linear shader should parse");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("SSR linear shader should validate");
    }

    #[test]
    fn test_ssr_fade_shader_parses() {
        let shader_source = include_str!("../shaders/ssr_fade.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSR fade shader should parse without errors");

        // Should have fade functions
        assert!(!module.functions.is_empty(), "Should have functions");
    }

    #[test]
    fn test_ssr_fade_shader_validates() {
        let shader_source = include_str!("../shaders/ssr_fade.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("SSR fade shader should parse");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("SSR fade shader should validate");
    }
}
