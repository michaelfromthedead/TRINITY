//! Demoscene Renderer (T-DEMO-5.3 / T-DEMO-5.4)
//!
//! This module provides a compute-based ray marching renderer for demoscene effects.
//! It implements embedded WGSL shaders and a render loop suitable for 4K intros.
//!
//! # Overview
//!
//! The `DemoRenderer` struct encapsulates:
//! - **Embedded Shader**: WGSL source compiled at runtime from `include_str!`
//! - **Compute Pipeline**: GPU pipeline for ray marching dispatch
//! - **Uniform Buffer**: Time and resolution parameters for animation
//! - **Render Loop**: Frame-rate independent animation with vsync support
//!
//! # Usage
//!
//! ```ignore
//! let mut renderer = DemoRenderer::new(&device, width, height);
//! loop {
//!     renderer.update(elapsed_time);
//!     renderer.dispatch(&mut encoder, &output_view);
//!     // ... present frame ...
//!     renderer.poll_events(&device);
//! }
//! ```
//!
//! # Design
//!
//! The compute shader uses @workgroup_size(8, 8, 1) for efficient GPU utilization.
//! Each thread computes one pixel via ray marching through an SDF scene.
//!
//! Frame timing is managed externally; `update()` accepts elapsed time in seconds.
//! The shader interpolates animations based on this time value.

use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Embedded Shader
// ---------------------------------------------------------------------------

/// Embedded WGSL shader source for demoscene ray marching.
///
/// This shader is loaded at compile time via `include_str!` for the 4K constraint.
/// It implements:
/// - SDF primitives (sphere, box, torus)
/// - Smooth boolean operations (smin)
/// - Ray marching with 64 steps
/// - Phong-like lighting with soft shadows
/// - Sky gradient and fog effects
pub const DEMO_SHADER: &str = include_str!("demoscene/demo.wgsl");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default workgroup size X dimension.
pub const WORKGROUP_SIZE_X: u32 = 8;

/// Default workgroup size Y dimension.
pub const WORKGROUP_SIZE_Y: u32 = 8;

/// Minimum supported render width.
pub const MIN_RENDER_WIDTH: u32 = 1;

/// Minimum supported render height.
pub const MIN_RENDER_HEIGHT: u32 = 1;

/// Maximum supported render dimension (4K).
pub const MAX_RENDER_DIMENSION: u32 = 4096;

/// Target frame time for vsync at 60 FPS (in seconds).
pub const TARGET_FRAME_TIME_60FPS: f32 = 1.0 / 60.0;

/// Target frame time for vsync at 144 FPS (in seconds).
pub const TARGET_FRAME_TIME_144FPS: f32 = 1.0 / 144.0;

// ---------------------------------------------------------------------------
// DemoUniforms
// ---------------------------------------------------------------------------

/// Uniform buffer data for the demoscene shader.
///
/// This struct is uploaded to the GPU each frame via the uniform buffer.
/// Layout matches the WGSL struct `DemoUniforms`.
///
/// # Memory Layout (16 bytes)
///
/// | Offset | Field        | Size    |
/// |--------|--------------|---------|
/// | 0      | time         | 4 bytes |
/// | 4      | resolution_x | 4 bytes |
/// | 8      | resolution_y | 4 bytes |
/// | 12     | _padding     | 4 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct DemoUniforms {
    /// Animation time in seconds (monotonically increasing).
    pub time: f32,

    /// Output texture width in pixels.
    pub resolution_x: f32,

    /// Output texture height in pixels.
    pub resolution_y: f32,

    /// Padding for vec4 alignment.
    pub _padding: f32,
}

impl Default for DemoUniforms {
    fn default() -> Self {
        Self {
            time: 0.0,
            resolution_x: 800.0,
            resolution_y: 600.0,
            _padding: 0.0,
        }
    }
}

impl DemoUniforms {
    /// Create new uniforms with the given resolution.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            time: 0.0,
            resolution_x: width as f32,
            resolution_y: height as f32,
            _padding: 0.0,
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
}

// ---------------------------------------------------------------------------
// DemoRenderConfig
// ---------------------------------------------------------------------------

/// Configuration for the demoscene renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemoRenderConfig {
    /// Render target width in pixels.
    pub width: u32,

    /// Render target height in pixels.
    pub height: u32,

    /// Workgroup size X (must match shader).
    pub workgroup_size_x: u32,

    /// Workgroup size Y (must match shader).
    pub workgroup_size_y: u32,

    /// Enable vertical sync (limits frame rate).
    pub vsync: bool,

    /// Target frame rate when vsync is enabled.
    pub target_fps: u32,
}

impl Default for DemoRenderConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            workgroup_size_x: WORKGROUP_SIZE_X,
            workgroup_size_y: WORKGROUP_SIZE_Y,
            vsync: true,
            target_fps: 60,
        }
    }
}

impl DemoRenderConfig {
    /// Create a new configuration with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.clamp(MIN_RENDER_WIDTH, MAX_RENDER_DIMENSION),
            height: height.clamp(MIN_RENDER_HEIGHT, MAX_RENDER_DIMENSION),
            ..Default::default()
        }
    }

    /// Set vsync mode.
    #[inline]
    pub fn with_vsync(mut self, vsync: bool) -> Self {
        self.vsync = vsync;
        self
    }

    /// Set target FPS.
    #[inline]
    pub fn with_target_fps(mut self, fps: u32) -> Self {
        self.target_fps = fps.max(1);
        self
    }

    /// Calculate the number of workgroups needed for dispatch.
    #[inline]
    pub fn dispatch_size(&self) -> (u32, u32, u32) {
        let x = (self.width + self.workgroup_size_x - 1) / self.workgroup_size_x;
        let y = (self.height + self.workgroup_size_y - 1) / self.workgroup_size_y;
        (x, y, 1)
    }

    /// Get the target frame time in seconds.
    #[inline]
    pub fn target_frame_time(&self) -> f32 {
        1.0 / self.target_fps as f32
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.width < MIN_RENDER_WIDTH || self.width > MAX_RENDER_DIMENSION {
            return Err("width out of range");
        }
        if self.height < MIN_RENDER_HEIGHT || self.height > MAX_RENDER_DIMENSION {
            return Err("height out of range");
        }
        if self.workgroup_size_x == 0 || self.workgroup_size_y == 0 {
            return Err("workgroup size cannot be zero");
        }
        if self.target_fps == 0 {
            return Err("target FPS cannot be zero");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FrameTiming
// ---------------------------------------------------------------------------

/// Frame timing statistics for the render loop.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameTiming {
    /// Current frame number.
    pub frame_number: u64,

    /// Elapsed time since start in seconds.
    pub elapsed_time: f32,

    /// Delta time since last frame in seconds.
    pub delta_time: f32,

    /// Instantaneous frames per second.
    pub fps: f32,

    /// Average frame time over recent frames (smoothed).
    pub avg_frame_time: f32,
}

impl FrameTiming {
    /// Create new frame timing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update timing for a new frame.
    pub fn update(&mut self, delta: f32) {
        self.frame_number += 1;
        self.delta_time = delta;
        self.elapsed_time += delta;

        if delta > 0.0 {
            self.fps = 1.0 / delta;
        }

        // Exponential moving average for smooth frame time
        const SMOOTHING: f32 = 0.9;
        self.avg_frame_time = SMOOTHING * self.avg_frame_time + (1.0 - SMOOTHING) * delta;
    }

    /// Reset timing to initial state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// DemoRenderer
// ---------------------------------------------------------------------------

/// Demoscene ray marching renderer.
///
/// Encapsulates a compute pipeline for real-time SDF ray marching.
/// Suitable for 4K intros and demoscene effects.
pub struct DemoRenderer {
    /// Configuration parameters.
    config: DemoRenderConfig,

    /// Uniform buffer for shader parameters.
    uniform_buffer: wgpu::Buffer,

    /// Current uniform data (CPU-side).
    uniforms: DemoUniforms,

    /// Compute pipeline for ray marching.
    compute_pipeline: wgpu::ComputePipeline,

    /// Bind group layout for the pipeline.
    bind_group_layout: wgpu::BindGroupLayout,

    /// Output texture for rendering (storage texture).
    output_texture: wgpu::Texture,

    /// Output texture view.
    output_view: wgpu::TextureView,

    /// Bind group with uniforms and output texture.
    bind_group: wgpu::BindGroup,

    /// Frame timing statistics.
    timing: FrameTiming,

    /// Start instant for elapsed time calculation.
    start_time: Instant,

    /// Last frame instant for delta calculation.
    last_frame_time: Instant,
}

impl DemoRenderer {
    /// Create a new demoscene renderer.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device for resource creation
    /// * `width` - render target width
    /// * `height` - render target height
    ///
    /// # Panics
    ///
    /// Panics if shader compilation fails or resource creation fails.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let config = DemoRenderConfig::new(width, height);
        Self::with_config(device, config)
    }

    /// Create a new demoscene renderer with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device for resource creation
    /// * `config` - render configuration
    ///
    /// # Panics
    ///
    /// Panics if shader compilation fails or resource creation fails.
    pub fn with_config(device: &wgpu::Device, config: DemoRenderConfig) -> Self {
        config.validate().expect("invalid configuration");

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Demoscene Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(DEMO_SHADER.into()),
        });

        // Create uniform buffer
        let uniforms = DemoUniforms::new(config.width, config.height);
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Demoscene Uniform Buffer"),
            size: std::mem::size_of::<DemoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output storage texture
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Demoscene Output Texture"),
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

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Demoscene Bind Group Layout"),
            entries: &[
                // Uniform buffer binding
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<DemoUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                // Output storage texture binding
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
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Demoscene Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<DemoUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
            ],
        });

        // Create compute pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Demoscene Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Demoscene Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let now = Instant::now();

        Self {
            config,
            uniform_buffer,
            uniforms,
            compute_pipeline,
            bind_group_layout,
            output_texture,
            output_view,
            bind_group,
            timing: FrameTiming::new(),
            start_time: now,
            last_frame_time: now,
        }
    }

    /// Update animation uniforms with the given time.
    ///
    /// # Arguments
    ///
    /// * `time` - Animation time in seconds (typically elapsed since start)
    #[inline]
    pub fn update(&mut self, time: f32) {
        self.uniforms.set_time(time);
    }

    /// Update uniforms and upload to GPU.
    ///
    /// # Arguments
    ///
    /// * `queue` - wgpu queue for buffer writes
    /// * `time` - Animation time in seconds
    pub fn update_and_upload(&mut self, queue: &wgpu::Queue, time: f32) {
        self.update(time);
        queue.write_buffer(&self.uniform_buffer, 0, self.uniforms.as_bytes());
    }

    /// Dispatch the compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record dispatch commands
    ///
    /// The output is written to the internal storage texture.
    /// Use `output_view()` to access the result.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let (x, y, z) = self.config.dispatch_size();

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Demoscene Compute Pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(x, y, z);
    }

    /// Present the frame (no-op for compute-only renderer).
    ///
    /// This method exists for API consistency with windowed renderers.
    /// For actual presentation, copy the output texture to a swapchain texture.
    #[inline]
    pub fn present(&self, _surface: &()) {
        // No-op for compute renderer
        // Real presentation would copy output_texture to swapchain
    }

    /// Poll device events and update frame timing.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device to poll
    ///
    /// Call this once per frame after dispatch and submit.
    pub fn poll_events(&mut self, device: &wgpu::Device) {
        device.poll(wgpu::Maintain::Poll);

        let now = Instant::now();
        let delta = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        self.timing.update(delta);
    }

    /// Run one complete frame iteration.
    ///
    /// This is a convenience method that:
    /// 1. Updates uniforms with current elapsed time
    /// 2. Uploads uniforms to GPU
    /// 3. Creates and dispatches compute pass
    /// 4. Submits commands to queue
    /// 5. Polls device events
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device
    /// * `queue` - wgpu queue
    ///
    /// # Returns
    ///
    /// The elapsed time in seconds since the renderer was created.
    pub fn render_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        self.update_and_upload(queue, elapsed);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Demoscene Frame Encoder"),
        });

        self.dispatch(&mut encoder);

        queue.submit(std::iter::once(encoder.finish()));

        self.poll_events(device);

        elapsed
    }

    /// Get the current frame timing statistics.
    #[inline]
    pub fn timing(&self) -> &FrameTiming {
        &self.timing
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &DemoRenderConfig {
        &self.config
    }

    /// Get the current uniforms.
    #[inline]
    pub fn uniforms(&self) -> &DemoUniforms {
        &self.uniforms
    }

    /// Get the output texture view for reading the result.
    #[inline]
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.output_view
    }

    /// Get the output texture for copy operations.
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
    ///
    /// Creates new output texture and bind group with updated dimensions.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.clamp(MIN_RENDER_WIDTH, MAX_RENDER_DIMENSION);
        let height = height.clamp(MIN_RENDER_HEIGHT, MAX_RENDER_DIMENSION);

        if width == self.config.width && height == self.config.height {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.uniforms.set_resolution(width, height);

        // Recreate output texture
        self.output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Demoscene Output Texture"),
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

        // Recreate bind group with new texture view
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Demoscene Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<DemoUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
            ],
        });
    }

    /// Reset the animation time to zero.
    pub fn reset_time(&mut self) {
        self.start_time = Instant::now();
        self.last_frame_time = self.start_time;
        self.timing.reset();
        self.uniforms.set_time(0.0);
    }

    /// Get elapsed time since renderer creation.
    #[inline]
    pub fn elapsed(&self) -> f32 {
        self.start_time.elapsed().as_secs_f32()
    }

    /// Copy the output texture to a target texture.
    ///
    /// Useful for presenting to a swapchain or readback.
    pub fn copy_output_to(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
    ) {
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Shader Validation
// ---------------------------------------------------------------------------

/// Validate that the embedded shader compiles successfully.
///
/// This function is useful for tests to verify shader correctness
/// without requiring a full GPU context.
///
/// Uses the `naga` crate (dev-dependency) for offline validation.
#[cfg(test)]
pub fn validate_demo_shader() -> Result<(), String> {
    use naga::front::wgsl;

    match wgsl::parse_str(DEMO_SHADER) {
        Ok(_module) => Ok(()),
        Err(err) => Err(format!("Shader parse error: {}", err)),
    }
}

/// Calculate dispatch dimensions for a given resolution.
#[inline]
pub fn calculate_dispatch_size(
    width: u32,
    height: u32,
    workgroup_x: u32,
    workgroup_y: u32,
) -> (u32, u32, u32) {
    let x = (width + workgroup_x - 1) / workgroup_x;
    let y = (height + workgroup_y - 1) / workgroup_y;
    (x, y, 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhi_device::RhiDevice;

    // =========================================================================
    // Shader Compilation Tests
    // =========================================================================

    #[test]
    fn test_shader_embedded_constant() {
        // Verify the shader constant is non-empty
        assert!(!DEMO_SHADER.is_empty(), "DEMO_SHADER should not be empty");
        assert!(
            DEMO_SHADER.len() > 100,
            "DEMO_SHADER should have substantial content"
        );
    }

    #[test]
    fn test_shader_contains_entry_point() {
        assert!(
            DEMO_SHADER.contains("fn main"),
            "Shader should contain main entry point"
        );
        assert!(
            DEMO_SHADER.contains("@compute"),
            "Shader should have @compute attribute"
        );
    }

    #[test]
    fn test_shader_contains_workgroup_size() {
        assert!(
            DEMO_SHADER.contains("@workgroup_size"),
            "Shader should specify workgroup size"
        );
        assert!(
            DEMO_SHADER.contains("@workgroup_size(8, 8, 1)"),
            "Shader should use 8x8x1 workgroup size"
        );
    }

    #[test]
    fn test_shader_contains_uniforms() {
        assert!(
            DEMO_SHADER.contains("DemoUniforms"),
            "Shader should define DemoUniforms struct"
        );
        assert!(
            DEMO_SHADER.contains("time: f32"),
            "Shader should have time uniform"
        );
        assert!(
            DEMO_SHADER.contains("resolution_x"),
            "Shader should have resolution_x uniform"
        );
        assert!(
            DEMO_SHADER.contains("resolution_y"),
            "Shader should have resolution_y uniform"
        );
    }

    #[test]
    fn test_shader_contains_bindings() {
        assert!(
            DEMO_SHADER.contains("@group(0) @binding(0)"),
            "Shader should have binding 0"
        );
        assert!(
            DEMO_SHADER.contains("@group(0) @binding(1)"),
            "Shader should have binding 1"
        );
    }

    #[test]
    fn test_shader_contains_output_texture() {
        assert!(
            DEMO_SHADER.contains("texture_storage_2d"),
            "Shader should use storage texture"
        );
        assert!(
            DEMO_SHADER.contains("textureStore"),
            "Shader should write to storage texture"
        );
    }

    #[test]
    fn test_shader_validation_naga() {
        // Validate shader with naga parser
        let result = validate_demo_shader();
        assert!(result.is_ok(), "Shader should parse: {:?}", result.err());
    }

    #[test]
    fn test_shader_contains_sdf_primitives() {
        assert!(
            DEMO_SHADER.contains("sdf_sphere"),
            "Shader should have sphere SDF"
        );
        assert!(
            DEMO_SHADER.contains("sdf_box"),
            "Shader should have box SDF"
        );
        assert!(
            DEMO_SHADER.contains("sdf_torus"),
            "Shader should have torus SDF"
        );
    }

    #[test]
    fn test_shader_contains_raymarching() {
        assert!(
            DEMO_SHADER.contains("ray_march"),
            "Shader should have ray_march function"
        );
        assert!(
            DEMO_SHADER.contains("MAX_STEPS"),
            "Shader should define MAX_STEPS"
        );
        assert!(
            DEMO_SHADER.contains("MAX_DIST"),
            "Shader should define MAX_DIST"
        );
    }

    // =========================================================================
    // DemoUniforms Tests
    // =========================================================================

    #[test]
    fn test_demo_uniforms_default() {
        let uniforms = DemoUniforms::default();
        assert_eq!(uniforms.time, 0.0);
        assert_eq!(uniforms.resolution_x, 800.0);
        assert_eq!(uniforms.resolution_y, 600.0);
        assert_eq!(uniforms._padding, 0.0);
    }

    #[test]
    fn test_demo_uniforms_new() {
        let uniforms = DemoUniforms::new(1920, 1080);
        assert_eq!(uniforms.time, 0.0);
        assert_eq!(uniforms.resolution_x, 1920.0);
        assert_eq!(uniforms.resolution_y, 1080.0);
    }

    #[test]
    fn test_demo_uniforms_set_time() {
        let mut uniforms = DemoUniforms::default();
        uniforms.set_time(5.5);
        assert_eq!(uniforms.time, 5.5);
    }

    #[test]
    fn test_demo_uniforms_set_resolution() {
        let mut uniforms = DemoUniforms::default();
        uniforms.set_resolution(3840, 2160);
        assert_eq!(uniforms.resolution_x, 3840.0);
        assert_eq!(uniforms.resolution_y, 2160.0);
    }

    #[test]
    fn test_demo_uniforms_resolution() {
        let uniforms = DemoUniforms::new(1280, 720);
        let (w, h) = uniforms.resolution();
        assert_eq!(w, 1280);
        assert_eq!(h, 720);
    }

    #[test]
    fn test_demo_uniforms_as_bytes() {
        let uniforms = DemoUniforms::new(100, 200);
        let bytes = uniforms.as_bytes();
        assert_eq!(bytes.len(), std::mem::size_of::<DemoUniforms>());
        assert_eq!(bytes.len(), 16); // 4 floats * 4 bytes
    }

    #[test]
    fn test_demo_uniforms_pod_zeroable() {
        // Verify Pod and Zeroable traits work
        let zeroed: DemoUniforms = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.time, 0.0);
        assert_eq!(zeroed.resolution_x, 0.0);
        assert_eq!(zeroed.resolution_y, 0.0);
    }

    #[test]
    fn test_demo_uniforms_memory_layout() {
        assert_eq!(std::mem::size_of::<DemoUniforms>(), 16);
        assert_eq!(std::mem::align_of::<DemoUniforms>(), 4);
    }

    // =========================================================================
    // DemoRenderConfig Tests
    // =========================================================================

    #[test]
    fn test_render_config_default() {
        let config = DemoRenderConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.workgroup_size_x, WORKGROUP_SIZE_X);
        assert_eq!(config.workgroup_size_y, WORKGROUP_SIZE_Y);
        assert!(config.vsync);
        assert_eq!(config.target_fps, 60);
    }

    #[test]
    fn test_render_config_new() {
        let config = DemoRenderConfig::new(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn test_render_config_clamp_min() {
        let config = DemoRenderConfig::new(0, 0);
        assert_eq!(config.width, MIN_RENDER_WIDTH);
        assert_eq!(config.height, MIN_RENDER_HEIGHT);
    }

    #[test]
    fn test_render_config_clamp_max() {
        let config = DemoRenderConfig::new(10000, 10000);
        assert_eq!(config.width, MAX_RENDER_DIMENSION);
        assert_eq!(config.height, MAX_RENDER_DIMENSION);
    }

    #[test]
    fn test_render_config_with_vsync() {
        let config = DemoRenderConfig::default().with_vsync(false);
        assert!(!config.vsync);
    }

    #[test]
    fn test_render_config_with_target_fps() {
        let config = DemoRenderConfig::default().with_target_fps(144);
        assert_eq!(config.target_fps, 144);
    }

    #[test]
    fn test_render_config_target_fps_min() {
        let config = DemoRenderConfig::default().with_target_fps(0);
        assert_eq!(config.target_fps, 1);
    }

    #[test]
    fn test_render_config_dispatch_size() {
        let config = DemoRenderConfig::new(800, 600);
        let (x, y, z) = config.dispatch_size();
        // 800 / 8 = 100, 600 / 8 = 75
        assert_eq!(x, 100);
        assert_eq!(y, 75);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_render_config_dispatch_size_non_aligned() {
        let config = DemoRenderConfig::new(801, 601);
        let (x, y, z) = config.dispatch_size();
        // (801 + 7) / 8 = 101, (601 + 7) / 8 = 76
        assert_eq!(x, 101);
        assert_eq!(y, 76);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_render_config_target_frame_time() {
        let config = DemoRenderConfig::default().with_target_fps(60);
        let frame_time = config.target_frame_time();
        assert!((frame_time - TARGET_FRAME_TIME_60FPS).abs() < 0.0001);
    }

    #[test]
    fn test_render_config_validate_ok() {
        let config = DemoRenderConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_render_config_validate_width_zero() {
        let mut config = DemoRenderConfig::default();
        config.width = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_render_config_validate_height_zero() {
        let mut config = DemoRenderConfig::default();
        config.height = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_render_config_validate_workgroup_zero() {
        let mut config = DemoRenderConfig::default();
        config.workgroup_size_x = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_render_config_validate_fps_zero() {
        let mut config = DemoRenderConfig::default();
        config.target_fps = 0;
        assert!(config.validate().is_err());
    }

    // =========================================================================
    // FrameTiming Tests
    // =========================================================================

    #[test]
    fn test_frame_timing_new() {
        let timing = FrameTiming::new();
        assert_eq!(timing.frame_number, 0);
        assert_eq!(timing.elapsed_time, 0.0);
        assert_eq!(timing.delta_time, 0.0);
        assert_eq!(timing.fps, 0.0);
    }

    #[test]
    fn test_frame_timing_update() {
        let mut timing = FrameTiming::new();
        timing.update(0.016667); // ~60 FPS

        assert_eq!(timing.frame_number, 1);
        assert!((timing.delta_time - 0.016667).abs() < 0.0001);
        assert!(timing.fps > 50.0 && timing.fps < 70.0);
    }

    #[test]
    fn test_frame_timing_elapsed_accumulates() {
        let mut timing = FrameTiming::new();
        timing.update(0.01);
        timing.update(0.02);
        timing.update(0.03);

        assert_eq!(timing.frame_number, 3);
        assert!((timing.elapsed_time - 0.06).abs() < 0.0001);
    }

    #[test]
    fn test_frame_timing_reset() {
        let mut timing = FrameTiming::new();
        timing.update(1.0);
        timing.reset();

        assert_eq!(timing.frame_number, 0);
        assert_eq!(timing.elapsed_time, 0.0);
    }

    #[test]
    fn test_frame_timing_avg_smoothing() {
        let mut timing = FrameTiming::new();
        // Simulate varying frame times
        for _ in 0..10 {
            timing.update(0.016);
        }
        // avg_frame_time should be close to 0.016
        assert!(timing.avg_frame_time > 0.01 && timing.avg_frame_time < 0.02);
    }

    // =========================================================================
    // Dispatch Dimension Tests
    // =========================================================================

    #[test]
    fn test_calculate_dispatch_size_aligned() {
        let (x, y, z) = calculate_dispatch_size(800, 600, 8, 8);
        assert_eq!(x, 100);
        assert_eq!(y, 75);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_calculate_dispatch_size_unaligned() {
        let (x, y, z) = calculate_dispatch_size(1920, 1080, 8, 8);
        assert_eq!(x, 240); // 1920 / 8 = 240
        assert_eq!(y, 135); // 1080 / 8 = 135
        assert_eq!(z, 1);
    }

    #[test]
    fn test_calculate_dispatch_size_small() {
        let (x, y, z) = calculate_dispatch_size(1, 1, 8, 8);
        assert_eq!(x, 1);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_calculate_dispatch_size_4k() {
        let (x, y, z) = calculate_dispatch_size(3840, 2160, 8, 8);
        assert_eq!(x, 480); // 3840 / 8 = 480
        assert_eq!(y, 270); // 2160 / 8 = 270
        assert_eq!(z, 1);
    }

    #[test]
    fn test_calculate_dispatch_size_odd() {
        let (x, y, z) = calculate_dispatch_size(7, 5, 8, 8);
        assert_eq!(x, 1); // (7 + 7) / 8 = 1
        assert_eq!(y, 1); // (5 + 7) / 8 = 1
        assert_eq!(z, 1);
    }

    // =========================================================================
    // Compute Pipeline Creation Tests (require GPU)
    // =========================================================================

    #[test]
    fn test_demo_renderer_creation() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = DemoRenderer::new(&device.device, 800, 600);
            assert_eq!(renderer.size(), (800, 600));
        }
    }

    #[test]
    fn test_demo_renderer_with_config() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let config = DemoRenderConfig::new(1280, 720).with_target_fps(144);
            let renderer = DemoRenderer::with_config(&device.device, config);
            assert_eq!(renderer.size(), (1280, 720));
            assert_eq!(renderer.config().target_fps, 144);
        }
    }

    #[test]
    fn test_demo_renderer_update() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = DemoRenderer::new(&device.device, 800, 600);
            renderer.update(1.5);
            assert_eq!(renderer.uniforms().time, 1.5);
        }
    }

    #[test]
    fn test_demo_renderer_update_and_upload() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = DemoRenderer::new(&device.device, 800, 600);
            renderer.update_and_upload(&device.queue, 2.0);
            assert_eq!(renderer.uniforms().time, 2.0);
        }
    }

    #[test]
    fn test_demo_renderer_dispatch() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = DemoRenderer::new(&device.device, 800, 600);

            let mut encoder = device.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("Test Encoder"),
                },
            );

            renderer.dispatch(&mut encoder);

            // Should not panic
            device.queue.submit(std::iter::once(encoder.finish()));
            device.wait_idle();
        }
    }

    #[test]
    fn test_demo_renderer_resize() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = DemoRenderer::new(&device.device, 800, 600);
            renderer.resize(&device.device, 1920, 1080);
            assert_eq!(renderer.size(), (1920, 1080));
        }
    }

    #[test]
    fn test_demo_renderer_resize_same_size() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = DemoRenderer::new(&device.device, 800, 600);
            renderer.resize(&device.device, 800, 600);
            // Should be a no-op
            assert_eq!(renderer.size(), (800, 600));
        }
    }

    #[test]
    fn test_demo_renderer_resize_clamp() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = DemoRenderer::new(&device.device, 800, 600);
            renderer.resize(&device.device, 10000, 10000);
            assert_eq!(renderer.size(), (MAX_RENDER_DIMENSION, MAX_RENDER_DIMENSION));
        }
    }

    #[test]
    fn test_demo_renderer_reset_time() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = DemoRenderer::new(&device.device, 800, 600);
            renderer.update(10.0);
            renderer.reset_time();
            assert_eq!(renderer.uniforms().time, 0.0);
            assert_eq!(renderer.timing().frame_number, 0);
        }
    }

    #[test]
    fn test_demo_renderer_render_frame() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = DemoRenderer::new(&device.device, 800, 600);
            let elapsed = renderer.render_frame(&device.device, &device.queue);
            assert!(elapsed >= 0.0);
            assert_eq!(renderer.timing().frame_number, 1);
        }
    }

    #[test]
    fn test_demo_renderer_multiple_frames() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = DemoRenderer::new(&device.device, 800, 600);

            for _ in 0..5 {
                renderer.render_frame(&device.device, &device.queue);
            }

            assert_eq!(renderer.timing().frame_number, 5);
        }
    }

    #[test]
    fn test_demo_renderer_output_view() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = DemoRenderer::new(&device.device, 800, 600);
            // Should not panic
            let _view = renderer.output_view();
        }
    }

    #[test]
    fn test_demo_renderer_output_texture() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = DemoRenderer::new(&device.device, 800, 600);
            let texture = renderer.output_texture();
            assert_eq!(texture.width(), 800);
            assert_eq!(texture.height(), 600);
        }
    }

    #[test]
    fn test_demo_renderer_copy_output() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = DemoRenderer::new(&device.device, 800, 600);

            let target = device.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Copy Target"),
                size: wgpu::Extent3d {
                    width: 800,
                    height: 600,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let mut encoder = device.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Encoder"),
                },
            );

            renderer.copy_output_to(&mut encoder, &target);

            device.queue.submit(std::iter::once(encoder.finish()));
            device.wait_idle();
        }
    }

    // =========================================================================
    // Uniform Buffer Update Tests
    // =========================================================================

    #[test]
    fn test_uniform_buffer_update_multiple() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = DemoRenderer::new(&device.device, 800, 600);

            for i in 0..10 {
                renderer.update_and_upload(&device.queue, i as f32 * 0.1);
            }

            assert!((renderer.uniforms().time - 0.9).abs() < 0.0001);
        }
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_constants_workgroup_size() {
        assert_eq!(WORKGROUP_SIZE_X, 8);
        assert_eq!(WORKGROUP_SIZE_Y, 8);
    }

    #[test]
    fn test_constants_render_limits() {
        assert_eq!(MIN_RENDER_WIDTH, 1);
        assert_eq!(MIN_RENDER_HEIGHT, 1);
        assert_eq!(MAX_RENDER_DIMENSION, 4096);
    }

    #[test]
    fn test_constants_frame_times() {
        assert!((TARGET_FRAME_TIME_60FPS - 1.0 / 60.0).abs() < 0.0001);
        assert!((TARGET_FRAME_TIME_144FPS - 1.0 / 144.0).abs() < 0.0001);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_demo_uniforms_extreme_values() {
        let mut uniforms = DemoUniforms::default();
        uniforms.set_time(f32::MAX);
        assert_eq!(uniforms.time, f32::MAX);

        uniforms.set_time(-1.0);
        assert_eq!(uniforms.time, -1.0);
    }

    #[test]
    fn test_dispatch_size_power_of_two() {
        // Common demoscene resolutions
        let cases = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
        ];

        for (w, h) in cases {
            let (x, y, z) = calculate_dispatch_size(w, h, 8, 8);
            assert_eq!(x, w / 8);
            assert_eq!(y, h / 8);
            assert_eq!(z, 1);
        }
    }

    #[test]
    fn test_render_config_clone_eq() {
        let config1 = DemoRenderConfig::new(800, 600);
        let config2 = config1;
        assert_eq!(config1, config2);
    }

    #[test]
    fn test_frame_timing_clone_copy() {
        let timing1 = FrameTiming {
            frame_number: 100,
            elapsed_time: 5.0,
            delta_time: 0.016,
            fps: 60.0,
            avg_frame_time: 0.016,
        };
        let timing2 = timing1;
        assert_eq!(timing1.frame_number, timing2.frame_number);
    }
}
