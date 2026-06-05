//! A Trous spatial denoiser for ray tracing (T-RT-P1.9).
//!
//! This module provides GPU dispatch for edge-aware spatial denoising using
//! the A Trous wavelet transform. The algorithm runs multiple iterations with
//! increasing step sizes (1, 2, 4, 8, ...) to filter at different spatial
//! frequencies while preserving edges.
//!
//! # Overview
//!
//! A Trous denoising is a fundamental component of modern ray tracing
//! denoisers like SVGF and ASVGF. It applies a 5-tap wavelet kernel with
//! edge-stopping functions based on depth, normal, and luminance similarity.
//!
//! # Iterations
//!
//! | Iteration | Step Size | Effective Radius |
//! |-----------|-----------|------------------|
//! | 0         | 1         | 2 pixels         |
//! | 1         | 2         | 4 pixels         |
//! | 2         | 4         | 8 pixels         |
//! | 3         | 8         | 16 pixels        |
//! | 4         | 16        | 32 pixels        |
//!
//! # Usage
//!
//! ```ignore
//! let pipeline = DenoiserPipeline::new(&device);
//!
//! // Run 5 iterations for full denoising
//! for iteration in 0..5 {
//!     let params = pipeline.create_params(iteration, width, height);
//!     // Create bind group with input/output textures...
//!     // Dispatch compute shader...
//!     // Swap input/output for next iteration (ping-pong)
//! }
//! ```
//!
//! # References
//!
//! - Dammertz et al., "Edge-Avoiding A-Trous Wavelet Transform for fast
//!   Global Illumination Filtering" (HPG 2010)
//! - NVIDIA SVGF (Spatiotemporal Variance-Guided Filtering)

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (8x8 threads).
pub const WORKGROUP_SIZE: u32 = 8;

/// Default sigma for color/luminance edge stopping.
/// Controls how much luminance variation is allowed before edge stopping.
pub const DEFAULT_SIGMA_COLOR: f32 = 4.0;

/// Default sigma for depth edge stopping.
/// Controls sensitivity to depth discontinuities.
pub const DEFAULT_SIGMA_DEPTH: f32 = 1.0;

/// Default sigma for normal edge stopping.
/// Higher values allow more blur across normal variations.
pub const DEFAULT_SIGMA_NORMAL: f32 = 128.0;

/// Maximum recommended iterations for A Trous denoising.
/// Beyond this, the filter radius exceeds practical limits.
pub const MAX_ITERATIONS: u32 = 5;

// ---------------------------------------------------------------------------
// DenoiseParams
// ---------------------------------------------------------------------------

/// GPU-side denoising parameters for an A Trous iteration.
///
/// This struct is uploaded to a uniform buffer for the compute shader.
/// Must match the WGSL `DenoiseParams` struct layout exactly.
///
/// # Memory Layout
///
/// 32 bytes total (8 x u32/f32), std140/std430 compatible:
///
/// | Offset | Field        | Size    |
/// |--------|--------------|---------|
/// | 0      | step_size    | 4 bytes |
/// | 4      | sigma_color  | 4 bytes |
/// | 8      | sigma_depth  | 4 bytes |
/// | 12     | sigma_normal | 4 bytes |
/// | 16     | width        | 4 bytes |
/// | 20     | height       | 4 bytes |
/// | 24     | _padding     | 8 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DenoiseParams {
    /// Wavelet step size (1, 2, 4, 8, ...). Each iteration doubles the step.
    pub step_size: u32,
    /// Color/luminance similarity sigma. Higher = more color blur allowed.
    pub sigma_color: f32,
    /// Depth similarity sigma. Lower = stronger depth edge preservation.
    pub sigma_depth: f32,
    /// Normal similarity sigma. Higher = more blur across normal variations.
    pub sigma_normal: f32,
    /// Input texture width in pixels.
    pub width: u32,
    /// Input texture height in pixels.
    pub height: u32,
    /// Padding to 32-byte alignment.
    pub _padding: [u32; 2],
}

// Compile-time size assertion - must be exactly 32 bytes
const _: () = assert!(mem::size_of::<DenoiseParams>() == 32);

impl DenoiseParams {
    /// Create parameters for a specific A Trous iteration.
    ///
    /// Step size doubles with each iteration: 1, 2, 4, 8, 16, ...
    ///
    /// # Arguments
    ///
    /// * `iteration` - Zero-based iteration index (0, 1, 2, ...)
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::raytracing::denoiser::DenoiseParams;
    ///
    /// let params = DenoiseParams::for_iteration(2, 1920, 1080);
    /// assert_eq!(params.step_size, 4); // 1 << 2 = 4
    /// ```
    pub fn for_iteration(iteration: u32, width: u32, height: u32) -> Self {
        Self {
            step_size: 1 << iteration,
            sigma_color: DEFAULT_SIGMA_COLOR,
            sigma_depth: DEFAULT_SIGMA_DEPTH,
            sigma_normal: DEFAULT_SIGMA_NORMAL,
            width,
            height,
            _padding: [0; 2],
        }
    }

    /// Create parameters with custom sigma values.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Zero-based iteration index
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `sigma_color` - Luminance edge-stopping sigma
    /// * `sigma_depth` - Depth edge-stopping sigma
    /// * `sigma_normal` - Normal edge-stopping sigma
    pub fn with_sigmas(
        iteration: u32,
        width: u32,
        height: u32,
        sigma_color: f32,
        sigma_depth: f32,
        sigma_normal: f32,
    ) -> Self {
        Self {
            step_size: 1 << iteration,
            sigma_color,
            sigma_depth,
            sigma_normal,
            width,
            height,
            _padding: [0; 2],
        }
    }

    /// Get the effective filter radius for the current step size.
    ///
    /// The A Trous kernel samples at offsets [-2, -1, 0, 1, 2] * step_size,
    /// so the effective radius is 2 * step_size.
    #[inline]
    pub fn effective_radius(&self) -> u32 {
        2 * self.step_size
    }

    /// Calculate step size for a given iteration.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Zero-based iteration index
    ///
    /// # Returns
    ///
    /// Step size as power of two: 1, 2, 4, 8, 16, ...
    #[inline]
    pub const fn step_size_for_iteration(iteration: u32) -> u32 {
        1u32 << iteration
    }
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self::for_iteration(0, 1920, 1080)
    }
}

// ---------------------------------------------------------------------------
// DenoiserPipeline
// ---------------------------------------------------------------------------

/// A Trous denoiser compute pipeline.
///
/// Manages the compute pipeline, bind group layout, and parameter buffer
/// for edge-aware spatial denoising.
///
/// # Bind Group Layout
///
/// | Binding | Type               | Content                          |
/// |---------|--------------------|----------------------------------|
/// | 0       | texture_2d         | Input color (noisy)              |
/// | 1       | texture_depth_2d   | Depth buffer                     |
/// | 2       | texture_2d         | Normal buffer                    |
/// | 3       | storage_texture    | Output color (denoised)          |
/// | 4       | uniform            | DenoiseParams                    |
pub struct DenoiserPipeline {
    /// Compute pipeline for A Trous denoising.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for denoiser resources.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Parameter buffer (reusable across iterations).
    params_buffer: wgpu::Buffer,
}

impl DenoiserPipeline {
    /// Create a new A Trous denoiser pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline = Self::create_pipeline(device, &bind_group_layout);
        let params_buffer = Self::create_params_buffer(device);

        Self {
            pipeline,
            bind_group_layout,
            params_buffer,
        }
    }

    /// Get the bind group layout for external bind group creation.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get a reference to the params buffer.
    #[inline]
    pub fn params_buffer(&self) -> &wgpu::Buffer {
        &self.params_buffer
    }

    /// Create parameters for a specific iteration with default sigmas.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Zero-based iteration index (0-4 recommended)
    /// * `width` - Texture width
    /// * `height` - Texture height
    #[inline]
    pub fn create_params(&self, iteration: u32, width: u32, height: u32) -> DenoiseParams {
        DenoiseParams::for_iteration(iteration, width, height)
    }

    /// Upload parameters to the GPU buffer.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue for buffer uploads.
    /// * `params` - Denoising parameters for this iteration.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &DenoiseParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Create the bind group layout.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atrous_denoise_bind_group_layout"),
            entries: &[
                // Binding 0: Input color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
                // Binding 3: Output storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Binding 4: DenoiseParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            mem::size_of::<DenoiseParams>() as u64,
                        ),
                    },
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
        let shader_source = include_str!("../../shaders/raytracing/atrous_denoise.comp.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("atrous_denoise_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("atrous_denoise_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("atrous_denoise_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Create the parameters buffer.
    fn create_params_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        let params = DenoiseParams::default();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atrous_denoise_params"),
            size: mem::size_of::<DenoiseParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::bytes_of(&params));
        buffer.unmap();

        buffer
    }

    /// Create a bind group for an A Trous iteration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `input_view` - Input color texture view (noisy).
    /// * `depth_view` - Depth buffer texture view.
    /// * `normal_view` - Normal buffer texture view.
    /// * `output_view` - Output storage texture view (denoised).
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atrous_denoise_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
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
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch the A Trous denoising compute shader.
    ///
    /// Records compute commands to the encoder. Call this once per iteration,
    /// with ping-pong texture swapping between iterations.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record to.
    /// * `bind_group` - Pre-created bind group with textures.
    /// * `width` - Output texture width in pixels.
    /// * `height` - Output texture height in pixels.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let (wg_x, wg_y, wg_z) = Self::workgroup_counts(width, height);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("atrous_denoise_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }

    /// Calculate workgroup counts for a given resolution.
    ///
    /// # Arguments
    ///
    /// * `width` - Texture width in pixels.
    /// * `height` - Texture height in pixels.
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
// Ping-Pong Buffer Helper
// ---------------------------------------------------------------------------

/// Manages ping-pong buffers for multi-pass A Trous denoising.
///
/// A Trous denoising requires multiple passes, each reading from the previous
/// pass's output. This helper manages the buffer swap.
pub struct PingPongBuffers {
    /// Texture A
    texture_a: wgpu::Texture,
    /// Texture B
    texture_b: wgpu::Texture,
    /// View for texture A
    view_a: wgpu::TextureView,
    /// View for texture B
    view_b: wgpu::TextureView,
    /// Current write target (0 = A, 1 = B)
    current_write: usize,
}

impl PingPongBuffers {
    /// Create ping-pong buffers for denoising.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `width` - Texture width.
    /// * `height` - Texture height.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let create_texture = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
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
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };

        let texture_a = create_texture("atrous_ping");
        let texture_b = create_texture("atrous_pong");

        let view_a = texture_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = texture_b.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            texture_a,
            texture_b,
            view_a,
            view_b,
            current_write: 0,
        }
    }

    /// Get the current input view (read source).
    pub fn input_view(&self) -> &wgpu::TextureView {
        if self.current_write == 0 {
            &self.view_b
        } else {
            &self.view_a
        }
    }

    /// Get the current output view (write target).
    pub fn output_view(&self) -> &wgpu::TextureView {
        if self.current_write == 0 {
            &self.view_a
        } else {
            &self.view_b
        }
    }

    /// Get the current output texture.
    #[allow(dead_code)]
    pub fn output_texture(&self) -> &wgpu::Texture {
        if self.current_write == 0 {
            &self.texture_a
        } else {
            &self.texture_b
        }
    }

    /// Swap buffers for the next iteration.
    pub fn swap(&mut self) {
        self.current_write = 1 - self.current_write;
    }

    /// Get the final result view after all iterations.
    /// This is the last write target.
    pub fn final_view(&self) -> &wgpu::TextureView {
        // After swap(), current_write points to next write target
        // So the result is in the other buffer
        if self.current_write == 0 {
            &self.view_b
        } else {
            &self.view_a
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- DenoiseParams --------------------------------------------------------

    #[test]
    fn test_denoise_params_size() {
        // Must be exactly 32 bytes for GPU alignment
        assert_eq!(mem::size_of::<DenoiseParams>(), 32);
    }

    #[test]
    fn test_denoise_params_alignment() {
        // Verify 4-byte field alignment
        assert_eq!(mem::align_of::<DenoiseParams>(), 4);
    }

    #[test]
    fn test_denoise_params_iteration_0() {
        let params = DenoiseParams::for_iteration(0, 1920, 1080);
        assert_eq!(params.step_size, 1);
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
        assert_eq!(params.sigma_color, DEFAULT_SIGMA_COLOR);
        assert_eq!(params.sigma_depth, DEFAULT_SIGMA_DEPTH);
        assert_eq!(params.sigma_normal, DEFAULT_SIGMA_NORMAL);
    }

    #[test]
    fn test_denoise_params_iteration_1() {
        let params = DenoiseParams::for_iteration(1, 1920, 1080);
        assert_eq!(params.step_size, 2);
    }

    #[test]
    fn test_denoise_params_iteration_2() {
        let params = DenoiseParams::for_iteration(2, 1920, 1080);
        assert_eq!(params.step_size, 4);
    }

    #[test]
    fn test_denoise_params_iteration_3() {
        let params = DenoiseParams::for_iteration(3, 1920, 1080);
        assert_eq!(params.step_size, 8);
    }

    #[test]
    fn test_denoise_params_iteration_4() {
        let params = DenoiseParams::for_iteration(4, 1920, 1080);
        assert_eq!(params.step_size, 16);
    }

    #[test]
    fn test_denoise_params_custom_sigmas() {
        let params = DenoiseParams::with_sigmas(2, 1280, 720, 2.0, 0.5, 64.0);
        assert_eq!(params.step_size, 4);
        assert_eq!(params.width, 1280);
        assert_eq!(params.height, 720);
        assert_eq!(params.sigma_color, 2.0);
        assert_eq!(params.sigma_depth, 0.5);
        assert_eq!(params.sigma_normal, 64.0);
    }

    #[test]
    fn test_denoise_params_effective_radius() {
        let params = DenoiseParams::for_iteration(0, 1920, 1080);
        assert_eq!(params.effective_radius(), 2); // 2 * 1

        let params = DenoiseParams::for_iteration(2, 1920, 1080);
        assert_eq!(params.effective_radius(), 8); // 2 * 4

        let params = DenoiseParams::for_iteration(4, 1920, 1080);
        assert_eq!(params.effective_radius(), 32); // 2 * 16
    }

    #[test]
    fn test_denoise_params_step_size_for_iteration() {
        assert_eq!(DenoiseParams::step_size_for_iteration(0), 1);
        assert_eq!(DenoiseParams::step_size_for_iteration(1), 2);
        assert_eq!(DenoiseParams::step_size_for_iteration(2), 4);
        assert_eq!(DenoiseParams::step_size_for_iteration(3), 8);
        assert_eq!(DenoiseParams::step_size_for_iteration(4), 16);
        assert_eq!(DenoiseParams::step_size_for_iteration(5), 32);
    }

    #[test]
    fn test_denoise_params_default() {
        let params = DenoiseParams::default();
        assert_eq!(params.step_size, 1);
        assert_eq!(params.width, 1920);
        assert_eq!(params.height, 1080);
    }

    // -- Workgroup Calculations -----------------------------------------------

    #[test]
    fn test_workgroup_counts_exact() {
        let (x, y, z) = DenoiserPipeline::workgroup_counts(1920, 1080);
        assert_eq!(x, 240); // 1920 / 8 = 240
        assert_eq!(y, 135); // 1080 / 8 = 135
        assert_eq!(z, 1);
    }

    #[test]
    fn test_workgroup_counts_rounded() {
        let (x, y, z) = DenoiserPipeline::workgroup_counts(1921, 1081);
        assert_eq!(x, 241); // ceil(1921 / 8) = 241
        assert_eq!(y, 136); // ceil(1081 / 8) = 136
        assert_eq!(z, 1);
    }

    #[test]
    fn test_workgroup_counts_small() {
        let (x, y, z) = DenoiserPipeline::workgroup_counts(8, 8);
        assert_eq!(x, 1);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_workgroup_counts_minimum() {
        let (x, y, z) = DenoiserPipeline::workgroup_counts(1, 1);
        assert_eq!(x, 1);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    // -- Constants ------------------------------------------------------------

    #[test]
    fn test_workgroup_size() {
        assert_eq!(WORKGROUP_SIZE, 8);
    }

    #[test]
    fn test_default_sigmas() {
        assert_eq!(DEFAULT_SIGMA_COLOR, 4.0);
        assert_eq!(DEFAULT_SIGMA_DEPTH, 1.0);
        assert_eq!(DEFAULT_SIGMA_NORMAL, 128.0);
    }

    #[test]
    fn test_max_iterations() {
        assert_eq!(MAX_ITERATIONS, 5);
    }

    // -- Shader Validation (using naga) ---------------------------------------

    #[test]
    fn test_atrous_shader_parses() {
        let shader_source = include_str!("../../shaders/raytracing/atrous_denoise.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("A Trous shader should parse without errors");

        // Verify the entry point exists
        let entry_point = module.entry_points.iter().find(|ep| ep.name == "main");
        assert!(entry_point.is_some(), "Should have main entry point");

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
    fn test_atrous_shader_validates() {
        let shader_source = include_str!("../../shaders/raytracing/atrous_denoise.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("A Trous shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        let _info = validator
            .validate(&module)
            .expect("A Trous shader should validate without errors");
    }

    #[test]
    fn test_atrous_shader_bindings() {
        let shader_source = include_str!("../../shaders/raytracing/atrous_denoise.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("A Trous shader should parse without errors");

        // Count global variables by binding type
        let mut texture_count = 0;
        let mut depth_texture_count = 0;
        let mut storage_count = 0;
        let mut uniform_count = 0;

        for (_, var) in module.global_variables.iter() {
            match var.space {
                naga::AddressSpace::Handle => {
                    match &module.types[var.ty].inner {
                        naga::TypeInner::Image {
                            class: naga::ImageClass::Storage { .. },
                            ..
                        } => {
                            storage_count += 1;
                        }
                        naga::TypeInner::Image {
                            class: naga::ImageClass::Depth { .. },
                            ..
                        } => {
                            depth_texture_count += 1;
                        }
                        naga::TypeInner::Image { .. } => {
                            texture_count += 1;
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

        // Expected: input_texture + normal_texture = 2 textures
        assert_eq!(texture_count, 2, "Should have 2 texture bindings");
        // Expected: depth_texture = 1 depth texture
        assert_eq!(depth_texture_count, 1, "Should have 1 depth texture binding");
        // Expected: output_texture = 1 storage texture
        assert_eq!(storage_count, 1, "Should have 1 storage texture binding");
        // Expected: params = 1 uniform
        assert_eq!(uniform_count, 1, "Should have 1 uniform binding");
    }

    #[test]
    fn test_atrous_shader_struct_denoise_params() {
        let shader_source = include_str!("../../shaders/raytracing/atrous_denoise.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("A Trous shader should parse without errors");

        // Find DenoiseParams struct
        let denoise_params = module.types.iter().find(|(_, ty)| {
            ty.name.as_ref().map_or(false, |n| n == "DenoiseParams")
        });
        assert!(denoise_params.is_some(), "Should have DenoiseParams struct");

        // Verify struct has expected number of members
        if let Some((_, ty)) = denoise_params {
            if let naga::TypeInner::Struct { members, .. } = &ty.inner {
                // step_size, sigma_color, sigma_depth, sigma_normal, width, height, _padding
                assert_eq!(members.len(), 7, "DenoiseParams should have 7 members");
            } else {
                panic!("DenoiseParams should be a struct");
            }
        }
    }

    #[test]
    fn test_atrous_shader_kernel_constants() {
        let shader_source = include_str!("../../shaders/raytracing/atrous_denoise.comp.wgsl");

        // Parse and verify the shader contains expected kernel weights
        assert!(
            shader_source.contains("KERNEL_WEIGHTS"),
            "Should define KERNEL_WEIGHTS constant"
        );
        assert!(
            shader_source.contains("KERNEL_OFFSETS"),
            "Should define KERNEL_OFFSETS constant"
        );
        assert!(
            shader_source.contains("0.0625"),
            "Should have 1/16 weight"
        );
        assert!(
            shader_source.contains("0.375"),
            "Should have 6/16 weight"
        );
    }

    #[test]
    fn test_atrous_shader_edge_stopping_functions() {
        let shader_source = include_str!("../../shaders/raytracing/atrous_denoise.comp.wgsl");

        // Verify edge-stopping functions are defined
        assert!(
            shader_source.contains("fn depth_weight"),
            "Should define depth_weight function"
        );
        assert!(
            shader_source.contains("fn normal_weight"),
            "Should define normal_weight function"
        );
        assert!(
            shader_source.contains("fn luminance_weight"),
            "Should define luminance_weight function"
        );
        assert!(
            shader_source.contains("fn luminance"),
            "Should define luminance helper function"
        );
    }

    // -- Bytemuck traits ------------------------------------------------------

    #[test]
    fn test_denoise_params_pod() {
        // Verify DenoiseParams implements Pod correctly
        let params = DenoiseParams::default();
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 32);

        // Verify we can cast back
        let restored: DenoiseParams = *bytemuck::from_bytes(bytes);
        assert_eq!(restored.step_size, params.step_size);
        assert_eq!(restored.width, params.width);
    }

    #[test]
    fn test_denoise_params_zeroable() {
        // Verify DenoiseParams implements Zeroable correctly
        let zeroed: DenoiseParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.step_size, 0);
        assert_eq!(zeroed.sigma_color, 0.0);
        assert_eq!(zeroed.width, 0);
    }
}
