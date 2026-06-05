//! Image processing compute shaders for wgpu 25.x (T-WGPU-P3.10.5).
//!
//! This module provides GPU infrastructure for common image processing operations:
//! - Gaussian blur (separable horizontal/vertical)
//! - 2x downsampling with multiple filter modes
//! - Luminance histogram computation
//! - HDR tonemapping with ACES and other curves
//!
//! # Overview
//!
//! All operations use compute shaders with shared memory optimizations for
//! efficient global memory access patterns. The module provides both low-level
//! dispatch helpers and a high-level `ImageProcessor` type that manages
//! pipelines and bind groups.
//!
//! # Usage
//!
//! ```ignore
//! // Create processor from device
//! let processor = ImageProcessor::new(&device);
//!
//! // Blur an image (separable 9-tap Gaussian)
//! processor.blur(&mut encoder, &input_view, &output_view, &uniforms, 1.0);
//!
//! // Downsample for mip generation
//! processor.downsample(&mut encoder, &src_view, &dst_view, &uniforms, FilterMode::Box);
//!
//! // Compute luminance histogram
//! let histogram_buffer = processor.compute_histogram(&mut encoder, &input_view, &uniforms);
//!
//! // Apply HDR tonemapping
//! processor.tonemap(&mut encoder, &hdr_view, &ldr_view, &uniforms, 0.0);
//! ```
//!
//! # Frame Graph Integration
//!
//! Each operation can be expressed as an `IrPass` for automatic barrier insertion:
//!
//! ```ignore
//! let blur_pass = create_blur_pass(index, input_handle, output_handle, width, height);
//! frame_graph.add_pass(blur_pass);
//! ```

use std::borrow::Cow;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size for blur operations.
pub const BLUR_WORKGROUP_SIZE: u32 = 128;

/// Compute shader workgroup size for downsample/tonemap operations.
pub const IMAGE_WORKGROUP_SIZE: u32 = 8;

/// Number of bins in the luminance histogram.
pub const HISTOGRAM_BINS: u32 = 256;

/// Kernel radius for 9-tap Gaussian blur.
pub const BLUR_KERNEL_RADIUS: u32 = 4;

// ---------------------------------------------------------------------------
// Filter Modes
// ---------------------------------------------------------------------------

/// Filter mode for downsampling operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum FilterMode {
    /// Box filter (2x2 average). Fastest, good for most use cases.
    #[default]
    Box = 0,
    /// Bilinear filter using hardware sampler. Smooth interpolation.
    Bilinear = 1,
    /// Karis average (luminance-weighted). Reduces fireflies in HDR bloom.
    Karis = 2,
}

impl FilterMode {
    /// Get the u32 value for shader uniform.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// Tonemapping Modes
// ---------------------------------------------------------------------------

/// Tonemapping curve selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum TonemapMode {
    /// ACES filmic (Narkowicz approximation). Industry standard.
    #[default]
    Aces = 0,
    /// Reinhard (simple). Can look washed out.
    Reinhard = 1,
    /// Uncharted 2 (John Hable). Good filmic look.
    Uncharted2 = 2,
    /// ACES fitted (Hill/Epic). More accurate, higher cost.
    AcesFitted = 3,
}

impl TonemapMode {
    /// Get the u32 value for shader uniform.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// Uniform Structures
// ---------------------------------------------------------------------------

/// Uniform buffer layout for blur operations.
///
/// Matches the WGSL `BlurUniforms` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlurUniforms {
    /// Source texture dimensions.
    pub src_dims: [u32; 2],
    /// Destination texture dimensions (same as source for blur).
    pub dst_dims: [u32; 2],
    /// Scale factor for blur radius (1.0 = standard).
    pub blur_scale: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl BlurUniforms {
    /// Create blur uniforms for the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            src_dims: [width, height],
            dst_dims: [width, height],
            blur_scale: 1.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Create blur uniforms with custom scale.
    pub fn with_scale(width: u32, height: u32, scale: f32) -> Self {
        Self {
            src_dims: [width, height],
            dst_dims: [width, height],
            blur_scale: scale,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

/// Uniform buffer layout for downsample operations.
///
/// Matches the WGSL `DownsampleUniforms` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DownsampleUniforms {
    /// Source texture dimensions.
    pub src_dims: [u32; 2],
    /// Destination texture dimensions (src / 2).
    pub dst_dims: [u32; 2],
    /// Filter mode (0 = box, 1 = bilinear, 2 = karis).
    pub filter_mode: u32,
    /// Current mip level being generated.
    pub mip_level: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl DownsampleUniforms {
    /// Create downsample uniforms for the given dimensions.
    pub fn new(src_width: u32, src_height: u32, filter: FilterMode, mip: u32) -> Self {
        Self {
            src_dims: [src_width, src_height],
            dst_dims: [src_width / 2, src_height / 2],
            filter_mode: filter.as_u32(),
            mip_level: mip,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create downsample uniforms with explicit destination size.
    pub fn with_dst_size(
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        filter: FilterMode,
        mip: u32,
    ) -> Self {
        Self {
            src_dims: [src_width, src_height],
            dst_dims: [dst_width, dst_height],
            filter_mode: filter.as_u32(),
            mip_level: mip,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Uniform buffer layout for histogram operations.
///
/// Matches the WGSL `HistogramUniforms` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HistogramUniforms {
    /// Source texture dimensions.
    pub src_dims: [u32; 2],
    /// Total pixel count (width * height).
    pub num_pixels: u32,
    /// Minimum log luminance for mapping.
    pub min_luminance: f32,
    /// Maximum log luminance for mapping.
    pub max_luminance: f32,
    pub _pad0: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl HistogramUniforms {
    /// Create histogram uniforms for the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            src_dims: [width, height],
            num_pixels: width * height,
            min_luminance: -10.0, // log2(~0.001)
            max_luminance: 4.0,   // log2(16)
            _pad0: 0.0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    /// Create histogram uniforms with custom luminance range.
    pub fn with_range(width: u32, height: u32, min_lum: f32, max_lum: f32) -> Self {
        Self {
            src_dims: [width, height],
            num_pixels: width * height,
            min_luminance: min_lum,
            max_luminance: max_lum,
            _pad0: 0.0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Uniform buffer layout for tonemapping operations.
///
/// Matches the WGSL `TonemapUniforms` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TonemapUniforms {
    /// Source texture dimensions.
    pub src_dims: [u32; 2],
    /// Destination texture dimensions.
    pub dst_dims: [u32; 2],
    /// Exposure adjustment in stops (EV).
    pub exposure: f32,
    /// Gamma correction value (typically 2.2).
    pub gamma: f32,
    /// Tonemapping curve selection.
    pub mode: u32,
    /// White point for some curves (default 4.0).
    pub white_point: f32,
}

impl TonemapUniforms {
    /// Create tonemap uniforms with default settings.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            src_dims: [width, height],
            dst_dims: [width, height],
            exposure: 0.0,
            gamma: 2.2,
            mode: TonemapMode::Aces.as_u32(),
            white_point: 4.0,
        }
    }

    /// Create tonemap uniforms with exposure adjustment.
    pub fn with_exposure(width: u32, height: u32, exposure: f32) -> Self {
        Self {
            src_dims: [width, height],
            dst_dims: [width, height],
            exposure,
            gamma: 2.2,
            mode: TonemapMode::Aces.as_u32(),
            white_point: 4.0,
        }
    }

    /// Create tonemap uniforms with full configuration.
    pub fn full(
        width: u32,
        height: u32,
        exposure: f32,
        gamma: f32,
        mode: TonemapMode,
        white_point: f32,
    ) -> Self {
        Self {
            src_dims: [width, height],
            dst_dims: [width, height],
            exposure,
            gamma,
            mode: mode.as_u32(),
            white_point,
        }
    }
}

// ---------------------------------------------------------------------------
// ImageProcessor
// ---------------------------------------------------------------------------

/// High-level image processing interface.
///
/// Manages compute pipelines, bind group layouts, and provides methods for
/// common image processing operations. Create once per device and reuse.
///
/// # Thread Safety
///
/// `ImageProcessor` is `Send + Sync` and can be shared across threads.
/// Individual operations require mutable access to a command encoder.
pub struct ImageProcessor {
    /// Horizontal blur pipeline.
    blur_horizontal_pipeline: wgpu::ComputePipeline,
    /// Vertical blur pipeline.
    blur_vertical_pipeline: wgpu::ComputePipeline,
    /// Downsample pipeline.
    downsample_pipeline: wgpu::ComputePipeline,
    /// Histogram compute pipeline.
    histogram_pipeline: wgpu::ComputePipeline,
    /// Histogram clear pipeline.
    histogram_clear_pipeline: wgpu::ComputePipeline,
    /// Tonemapping pipeline.
    tonemap_pipeline: wgpu::ComputePipeline,

    /// Bind group layout for blur operations.
    blur_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for downsample operations.
    downsample_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for histogram operations.
    histogram_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for tonemap operations.
    tonemap_bind_group_layout: wgpu::BindGroupLayout,

    /// Linear sampler for bilinear filtering.
    linear_sampler: wgpu::Sampler,
}

impl ImageProcessor {
    /// Shader source for blur_horizontal.wgsl.
    const BLUR_HORIZONTAL_WGSL: &'static str = include_str!("../../shaders/blur_horizontal.wgsl");
    /// Shader source for blur_vertical.wgsl.
    const BLUR_VERTICAL_WGSL: &'static str = include_str!("../../shaders/blur_vertical.wgsl");
    /// Shader source for downsample.wgsl.
    const DOWNSAMPLE_WGSL: &'static str = include_str!("../../shaders/downsample.wgsl");
    /// Shader source for histogram.wgsl.
    const HISTOGRAM_WGSL: &'static str = include_str!("../../shaders/histogram.wgsl");
    /// Shader source for tonemapping.wgsl.
    const TONEMAPPING_WGSL: &'static str = include_str!("../../shaders/tonemapping.wgsl");

    /// Create a new image processor with all required pipelines.
    ///
    /// This compiles all shaders and creates pipeline layouts. The operation
    /// may take a few milliseconds on first call.
    pub fn new(device: &wgpu::Device) -> Self {
        // Create shader modules
        let blur_h_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur_horizontal.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(Self::BLUR_HORIZONTAL_WGSL)),
        });

        let blur_v_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur_vertical.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(Self::BLUR_VERTICAL_WGSL)),
        });

        let downsample_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("downsample.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(Self::DOWNSAMPLE_WGSL)),
        });

        let histogram_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("histogram.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(Self::HISTOGRAM_WGSL)),
        });

        let tonemap_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tonemapping.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(Self::TONEMAPPING_WGSL)),
        });

        // Create linear sampler for bilinear filtering
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("image_processing_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create bind group layouts
        let blur_bind_group_layout = Self::create_blur_bind_group_layout(device);
        let downsample_bind_group_layout = Self::create_downsample_bind_group_layout(device);
        let histogram_bind_group_layout = Self::create_histogram_bind_group_layout(device);
        let tonemap_bind_group_layout = Self::create_tonemap_bind_group_layout(device);

        // Create pipeline layouts
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blur_pipeline_layout"),
            bind_group_layouts: &[&blur_bind_group_layout],
            push_constant_ranges: &[],
        });

        let downsample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("downsample_pipeline_layout"),
                bind_group_layouts: &[&downsample_bind_group_layout],
                push_constant_ranges: &[],
            });

        let histogram_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("histogram_pipeline_layout"),
                bind_group_layouts: &[&histogram_bind_group_layout],
                push_constant_ranges: &[],
            });

        let tonemap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tonemap_pipeline_layout"),
                bind_group_layouts: &[&tonemap_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create compute pipelines
        let blur_horizontal_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("blur_horizontal_pipeline"),
                layout: Some(&blur_pipeline_layout),
                module: &blur_h_module,
                entry_point: "blur_horizontal",
                compilation_options: Default::default(),
                cache: None,
            });

        let blur_vertical_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("blur_vertical_pipeline"),
                layout: Some(&blur_pipeline_layout),
                module: &blur_v_module,
                entry_point: "blur_vertical",
                compilation_options: Default::default(),
                cache: None,
            });

        let downsample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("downsample_pipeline"),
                layout: Some(&downsample_pipeline_layout),
                module: &downsample_module,
                entry_point: "downsample",
                compilation_options: Default::default(),
                cache: None,
            });

        let histogram_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("histogram_pipeline"),
                layout: Some(&histogram_pipeline_layout),
                module: &histogram_module,
                entry_point: "compute_histogram",
                compilation_options: Default::default(),
                cache: None,
            });

        let histogram_clear_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("histogram_clear_pipeline"),
                layout: Some(&histogram_pipeline_layout),
                module: &histogram_module,
                entry_point: "clear_histogram",
                compilation_options: Default::default(),
                cache: None,
            });

        let tonemap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tonemap_pipeline"),
            layout: Some(&tonemap_pipeline_layout),
            module: &tonemap_module,
            entry_point: "tonemap",
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            blur_horizontal_pipeline,
            blur_vertical_pipeline,
            downsample_pipeline,
            histogram_pipeline,
            histogram_clear_pipeline,
            tonemap_pipeline,
            blur_bind_group_layout,
            downsample_bind_group_layout,
            histogram_bind_group_layout,
            tonemap_bind_group_layout,
            linear_sampler,
        }
    }

    /// Create bind group layout for blur operations.
    fn create_blur_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur_bind_group_layout"),
            entries: &[
                // @binding(0): Source texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(1): Destination storage texture
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
                // @binding(2): Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for downsample operations.
    fn create_downsample_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("downsample_bind_group_layout"),
            entries: &[
                // @binding(0): Source texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(1): Destination storage texture
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
                // @binding(2): Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3): Linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for histogram operations.
    fn create_histogram_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("histogram_bind_group_layout"),
            entries: &[
                // @binding(0): Source texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(1): Histogram storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2): Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for tonemap operations.
    fn create_tonemap_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap_bind_group_layout"),
            entries: &[
                // @binding(0): Source HDR texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(1): Destination LDR storage texture
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
                // @binding(2): Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Get the blur bind group layout for external bind group creation.
    pub fn blur_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.blur_bind_group_layout
    }

    /// Get the downsample bind group layout for external bind group creation.
    pub fn downsample_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.downsample_bind_group_layout
    }

    /// Get the histogram bind group layout for external bind group creation.
    pub fn histogram_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.histogram_bind_group_layout
    }

    /// Get the tonemap bind group layout for external bind group creation.
    pub fn tonemap_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.tonemap_bind_group_layout
    }

    /// Get the linear sampler.
    pub fn linear_sampler(&self) -> &wgpu::Sampler {
        &self.linear_sampler
    }

    // -------------------------------------------------------------------------
    // Blur Operations
    // -------------------------------------------------------------------------

    /// Create a bind group for blur operations.
    pub fn create_blur_bind_group(
        &self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
        dst_view: &wgpu::TextureView,
        uniforms: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_bind_group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        })
    }

    /// Apply horizontal Gaussian blur.
    ///
    /// Dispatches the horizontal blur compute shader. For a full 2D blur,
    /// call this followed by `blur_vertical` with an intermediate texture.
    pub fn blur_horizontal(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("blur_horizontal_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.blur_horizontal_pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        // Dispatch: ceil(width / 128) x height workgroups
        let workgroups_x = (width + BLUR_WORKGROUP_SIZE - 1) / BLUR_WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups_x, height, 1);
    }

    /// Apply vertical Gaussian blur.
    ///
    /// Dispatches the vertical blur compute shader. For a full 2D blur,
    /// call `blur_horizontal` first with an intermediate texture.
    pub fn blur_vertical(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("blur_vertical_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.blur_vertical_pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        // Dispatch: width x ceil(height / 128) workgroups
        let workgroups_y = (height + BLUR_WORKGROUP_SIZE - 1) / BLUR_WORKGROUP_SIZE;
        pass.dispatch_workgroups(width, workgroups_y, 1);
    }

    /// Apply full 2D Gaussian blur (horizontal + vertical passes).
    ///
    /// Requires an intermediate texture for the horizontal pass output.
    /// The blur radius is controlled by `scale` in the uniforms.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record to.
    /// * `input` - Input texture view.
    /// * `intermediate` - Intermediate texture view (must be writable).
    /// * `output` - Output texture view.
    /// * `uniforms_h` - Uniform buffer for horizontal pass.
    /// * `uniforms_v` - Uniform buffer for vertical pass.
    /// * `width` - Texture width.
    /// * `height` - Texture height.
    pub fn blur(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        intermediate: &wgpu::TextureView,
        output: &wgpu::TextureView,
        uniforms_h: &wgpu::Buffer,
        uniforms_v: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        // Horizontal pass: input -> intermediate
        let bind_group_h = self.create_blur_bind_group(device, input, intermediate, uniforms_h);
        self.blur_horizontal(encoder, &bind_group_h, width, height);

        // Vertical pass: intermediate -> output
        let bind_group_v = self.create_blur_bind_group(device, intermediate, output, uniforms_v);
        self.blur_vertical(encoder, &bind_group_v, width, height);
    }

    // -------------------------------------------------------------------------
    // Downsample Operations
    // -------------------------------------------------------------------------

    /// Create a bind group for downsample operations.
    pub fn create_downsample_bind_group(
        &self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
        dst_view: &wgpu::TextureView,
        uniforms: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("downsample_bind_group"),
            layout: &self.downsample_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        })
    }

    /// Apply 2x downsampling.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record to.
    /// * `bind_group` - Pre-created bind group.
    /// * `dst_width` - Destination width (src_width / 2).
    /// * `dst_height` - Destination height (src_height / 2).
    pub fn downsample(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        dst_width: u32,
        dst_height: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("downsample_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.downsample_pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        // Dispatch: ceil(width / 8) x ceil(height / 8) workgroups
        let workgroups_x = (dst_width + IMAGE_WORKGROUP_SIZE - 1) / IMAGE_WORKGROUP_SIZE;
        let workgroups_y = (dst_height + IMAGE_WORKGROUP_SIZE - 1) / IMAGE_WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // -------------------------------------------------------------------------
    // Histogram Operations
    // -------------------------------------------------------------------------

    /// Create a bind group for histogram operations.
    pub fn create_histogram_bind_group(
        &self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
        histogram_buffer: &wgpu::Buffer,
        uniforms: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histogram_bind_group"),
            layout: &self.histogram_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        })
    }

    /// Clear the histogram buffer to zero.
    ///
    /// Must be called before `compute_histogram`.
    pub fn clear_histogram(&self, encoder: &mut wgpu::CommandEncoder, bind_group: &wgpu::BindGroup) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram_clear_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.histogram_clear_pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        // Single workgroup of 256 threads clears 256 bins
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Compute luminance histogram.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record to.
    /// * `bind_group` - Pre-created bind group.
    /// * `num_pixels` - Total number of pixels (width * height).
    ///
    /// # Note
    ///
    /// Call `clear_histogram` before this to reset the histogram buffer.
    pub fn compute_histogram(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        num_pixels: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram_compute_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.histogram_pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        // Grid-stride loop handles arbitrary pixel counts
        // Use enough workgroups to cover all pixels with reasonable parallelism
        let workgroups = (num_pixels + HISTOGRAM_BINS - 1) / HISTOGRAM_BINS;
        let workgroups = workgroups.min(1024); // Cap at 1024 workgroups
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    // -------------------------------------------------------------------------
    // Tonemap Operations
    // -------------------------------------------------------------------------

    /// Create a bind group for tonemap operations.
    pub fn create_tonemap_bind_group(
        &self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
        dst_view: &wgpu::TextureView,
        uniforms: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap_bind_group"),
            layout: &self.tonemap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        })
    }

    /// Apply HDR tonemapping.
    ///
    /// Converts HDR input to LDR output using the selected tonemapping curve
    /// with exposure adjustment and gamma correction.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record to.
    /// * `bind_group` - Pre-created bind group.
    /// * `width` - Texture width.
    /// * `height` - Texture height.
    pub fn tonemap(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tonemap_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.tonemap_pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        // Dispatch: ceil(width / 8) x ceil(height / 8) workgroups
        let workgroups_x = (width + IMAGE_WORKGROUP_SIZE - 1) / IMAGE_WORKGROUP_SIZE;
        let workgroups_y = (height + IMAGE_WORKGROUP_SIZE - 1) / IMAGE_WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Create a histogram buffer with 256 u32 bins.
pub fn create_histogram_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram_buffer"),
        size: (HISTOGRAM_BINS * std::mem::size_of::<u32>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Compute the number of workgroups needed for an 8x8 workgroup size.
#[inline]
pub fn compute_workgroups_8x8(width: u32, height: u32) -> (u32, u32) {
    let x = (width + 7) / 8;
    let y = (height + 7) / 8;
    (x, y)
}

/// Compute the number of workgroups needed for horizontal blur (128x1).
#[inline]
pub fn compute_workgroups_blur_h(width: u32, height: u32) -> (u32, u32) {
    let x = (width + 127) / 128;
    (x, height)
}

/// Compute the number of workgroups needed for vertical blur (1x128).
#[inline]
pub fn compute_workgroups_blur_v(width: u32, height: u32) -> (u32, u32) {
    let y = (height + 127) / 128;
    (width, y)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_mode_values() {
        assert_eq!(FilterMode::Box.as_u32(), 0);
        assert_eq!(FilterMode::Bilinear.as_u32(), 1);
        assert_eq!(FilterMode::Karis.as_u32(), 2);
    }

    #[test]
    fn test_tonemap_mode_values() {
        assert_eq!(TonemapMode::Aces.as_u32(), 0);
        assert_eq!(TonemapMode::Reinhard.as_u32(), 1);
        assert_eq!(TonemapMode::Uncharted2.as_u32(), 2);
        assert_eq!(TonemapMode::AcesFitted.as_u32(), 3);
    }

    #[test]
    fn test_blur_uniforms() {
        let uniforms = BlurUniforms::new(1920, 1080);
        assert_eq!(uniforms.src_dims, [1920, 1080]);
        assert_eq!(uniforms.dst_dims, [1920, 1080]);
        assert_eq!(uniforms.blur_scale, 1.0);
    }

    #[test]
    fn test_blur_uniforms_with_scale() {
        let uniforms = BlurUniforms::with_scale(1920, 1080, 2.0);
        assert_eq!(uniforms.blur_scale, 2.0);
    }

    #[test]
    fn test_downsample_uniforms() {
        let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Box, 0);
        assert_eq!(uniforms.src_dims, [1920, 1080]);
        assert_eq!(uniforms.dst_dims, [960, 540]);
        assert_eq!(uniforms.filter_mode, 0);
        assert_eq!(uniforms.mip_level, 0);
    }

    #[test]
    fn test_downsample_uniforms_karis() {
        let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Karis, 0);
        assert_eq!(uniforms.filter_mode, 2);
    }

    #[test]
    fn test_histogram_uniforms() {
        let uniforms = HistogramUniforms::new(1920, 1080);
        assert_eq!(uniforms.src_dims, [1920, 1080]);
        assert_eq!(uniforms.num_pixels, 1920 * 1080);
        assert_eq!(uniforms.min_luminance, -10.0);
        assert_eq!(uniforms.max_luminance, 4.0);
    }

    #[test]
    fn test_histogram_uniforms_with_range() {
        let uniforms = HistogramUniforms::with_range(1920, 1080, -5.0, 10.0);
        assert_eq!(uniforms.min_luminance, -5.0);
        assert_eq!(uniforms.max_luminance, 10.0);
    }

    #[test]
    fn test_tonemap_uniforms() {
        let uniforms = TonemapUniforms::new(1920, 1080);
        assert_eq!(uniforms.src_dims, [1920, 1080]);
        assert_eq!(uniforms.exposure, 0.0);
        assert_eq!(uniforms.gamma, 2.2);
        assert_eq!(uniforms.mode, 0); // ACES
        assert_eq!(uniforms.white_point, 4.0);
    }

    #[test]
    fn test_tonemap_uniforms_with_exposure() {
        let uniforms = TonemapUniforms::with_exposure(1920, 1080, 1.5);
        assert_eq!(uniforms.exposure, 1.5);
    }

    #[test]
    fn test_tonemap_uniforms_full() {
        let uniforms = TonemapUniforms::full(
            1920,
            1080,
            2.0,
            2.4,
            TonemapMode::Uncharted2,
            6.0,
        );
        assert_eq!(uniforms.exposure, 2.0);
        assert_eq!(uniforms.gamma, 2.4);
        assert_eq!(uniforms.mode, 2);
        assert_eq!(uniforms.white_point, 6.0);
    }

    #[test]
    fn test_compute_workgroups_8x8() {
        assert_eq!(compute_workgroups_8x8(1920, 1080), (240, 135));
        assert_eq!(compute_workgroups_8x8(1, 1), (1, 1));
        assert_eq!(compute_workgroups_8x8(8, 8), (1, 1));
        assert_eq!(compute_workgroups_8x8(9, 9), (2, 2));
    }

    #[test]
    fn test_compute_workgroups_blur_h() {
        assert_eq!(compute_workgroups_blur_h(1920, 1080), (15, 1080));
        assert_eq!(compute_workgroups_blur_h(128, 100), (1, 100));
        assert_eq!(compute_workgroups_blur_h(129, 100), (2, 100));
    }

    #[test]
    fn test_compute_workgroups_blur_v() {
        assert_eq!(compute_workgroups_blur_v(1920, 1080), (1920, 9));
        assert_eq!(compute_workgroups_blur_v(100, 128), (100, 1));
        assert_eq!(compute_workgroups_blur_v(100, 129), (100, 2));
    }

    #[test]
    fn test_uniform_sizes_match_16byte_alignment() {
        // wgpu requires uniform buffers to be 16-byte aligned
        assert_eq!(std::mem::size_of::<BlurUniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<DownsampleUniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<HistogramUniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<TonemapUniforms>() % 16, 0);
    }

    // =========================================================================
    // WHITEBOX TESTS: T-WGPU-P3.10.5 (Image Processing)
    // =========================================================================

    // -------------------------------------------------------------------------
    // 1. Uniform structs: bytemuck::Pod/Zeroable verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_blur_uniforms_pod_zeroable() {
        // Verify bytemuck traits work correctly
        let zeroed: BlurUniforms = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.src_dims, [0, 0]);
        assert_eq!(zeroed.dst_dims, [0, 0]);
        assert_eq!(zeroed.blur_scale, 0.0);
        assert_eq!(zeroed._pad0, 0.0);
        assert_eq!(zeroed._pad1, 0.0);
        assert_eq!(zeroed._pad2, 0.0);

        // Verify Pod cast works
        let uniforms = BlurUniforms::new(100, 200);
        let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), std::mem::size_of::<BlurUniforms>());

        // Round-trip through bytes
        let restored: &BlurUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(restored.src_dims, uniforms.src_dims);
        assert_eq!(restored.dst_dims, uniforms.dst_dims);
    }

    #[test]
    fn test_downsample_uniforms_pod_zeroable() {
        let zeroed: DownsampleUniforms = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.src_dims, [0, 0]);
        assert_eq!(zeroed.dst_dims, [0, 0]);
        assert_eq!(zeroed.filter_mode, 0);
        assert_eq!(zeroed.mip_level, 0);

        let uniforms = DownsampleUniforms::new(512, 256, FilterMode::Bilinear, 3);
        let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), std::mem::size_of::<DownsampleUniforms>());

        let restored: &DownsampleUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(restored.filter_mode, FilterMode::Bilinear.as_u32());
        assert_eq!(restored.mip_level, 3);
    }

    #[test]
    fn test_histogram_uniforms_pod_zeroable() {
        let zeroed: HistogramUniforms = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.src_dims, [0, 0]);
        assert_eq!(zeroed.num_pixels, 0);
        assert_eq!(zeroed.min_luminance, 0.0);
        assert_eq!(zeroed.max_luminance, 0.0);

        let uniforms = HistogramUniforms::with_range(800, 600, -8.0, 8.0);
        let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), std::mem::size_of::<HistogramUniforms>());

        let restored: &HistogramUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(restored.num_pixels, 800 * 600);
        assert!((restored.min_luminance - (-8.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tonemap_uniforms_pod_zeroable() {
        let zeroed: TonemapUniforms = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.src_dims, [0, 0]);
        assert_eq!(zeroed.dst_dims, [0, 0]);
        assert_eq!(zeroed.exposure, 0.0);
        assert_eq!(zeroed.gamma, 0.0);
        assert_eq!(zeroed.mode, 0);
        assert_eq!(zeroed.white_point, 0.0);

        let uniforms = TonemapUniforms::full(1280, 720, 1.5, 2.2, TonemapMode::AcesFitted, 5.0);
        let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), std::mem::size_of::<TonemapUniforms>());

        let restored: &TonemapUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(restored.mode, TonemapMode::AcesFitted.as_u32());
    }

    // -------------------------------------------------------------------------
    // 2. FilterMode enum: Clone, Copy, PartialEq, Eq, Default, Hash
    // -------------------------------------------------------------------------

    #[test]
    fn test_filter_mode_clone() {
        let mode = FilterMode::Karis;
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_filter_mode_copy() {
        let mode = FilterMode::Bilinear;
        let copied = mode; // Copy trait
        assert_eq!(mode, copied);
        // Both are still usable (Copy, not Move)
        assert_eq!(mode.as_u32(), copied.as_u32());
    }

    #[test]
    fn test_filter_mode_partial_eq() {
        assert_eq!(FilterMode::Box, FilterMode::Box);
        assert_ne!(FilterMode::Box, FilterMode::Bilinear);
        assert_ne!(FilterMode::Bilinear, FilterMode::Karis);
    }

    #[test]
    fn test_filter_mode_default() {
        let default_mode: FilterMode = Default::default();
        assert_eq!(default_mode, FilterMode::Box);
    }

    #[test]
    fn test_filter_mode_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FilterMode::Box);
        set.insert(FilterMode::Bilinear);
        set.insert(FilterMode::Karis);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&FilterMode::Karis));
    }

    #[test]
    fn test_filter_mode_debug() {
        let debug_str = format!("{:?}", FilterMode::Bilinear);
        assert!(debug_str.contains("Bilinear"));
    }

    // -------------------------------------------------------------------------
    // 3. TonemapMode enum: Clone, Copy, PartialEq, Eq, Default, Hash
    // -------------------------------------------------------------------------

    #[test]
    fn test_tonemap_mode_clone() {
        let mode = TonemapMode::Uncharted2;
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_tonemap_mode_copy() {
        let mode = TonemapMode::Reinhard;
        let copied = mode;
        assert_eq!(mode, copied);
        assert_eq!(mode.as_u32(), copied.as_u32());
    }

    #[test]
    fn test_tonemap_mode_partial_eq() {
        assert_eq!(TonemapMode::Aces, TonemapMode::Aces);
        assert_ne!(TonemapMode::Aces, TonemapMode::AcesFitted);
        assert_ne!(TonemapMode::Reinhard, TonemapMode::Uncharted2);
    }

    #[test]
    fn test_tonemap_mode_default() {
        let default_mode: TonemapMode = Default::default();
        assert_eq!(default_mode, TonemapMode::Aces);
    }

    #[test]
    fn test_tonemap_mode_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TonemapMode::Aces);
        set.insert(TonemapMode::Reinhard);
        set.insert(TonemapMode::Uncharted2);
        set.insert(TonemapMode::AcesFitted);
        assert_eq!(set.len(), 4);
        assert!(set.contains(&TonemapMode::AcesFitted));
    }

    #[test]
    fn test_tonemap_mode_debug() {
        let debug_str = format!("{:?}", TonemapMode::AcesFitted);
        assert!(debug_str.contains("AcesFitted"));
    }

    #[test]
    fn test_tonemap_mode_all_variants() {
        // Ensure all variants map to unique values
        let modes = [
            TonemapMode::Aces,
            TonemapMode::Reinhard,
            TonemapMode::Uncharted2,
            TonemapMode::AcesFitted,
        ];
        for (i, mode) in modes.iter().enumerate() {
            assert_eq!(mode.as_u32(), i as u32);
        }
    }

    // -------------------------------------------------------------------------
    // 4. Workgroup calculations: edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_workgroups_small_images() {
        // Single pixel
        assert_eq!(compute_workgroups_8x8(1, 1), (1, 1));
        assert_eq!(compute_workgroups_blur_h(1, 1), (1, 1));
        assert_eq!(compute_workgroups_blur_v(1, 1), (1, 1));

        // Less than workgroup size
        assert_eq!(compute_workgroups_8x8(4, 4), (1, 1));
        assert_eq!(compute_workgroups_blur_h(64, 50), (1, 50));
        assert_eq!(compute_workgroups_blur_v(50, 64), (50, 1));
    }

    #[test]
    fn test_workgroups_exact_boundaries() {
        // Exactly on boundary
        assert_eq!(compute_workgroups_8x8(8, 8), (1, 1));
        assert_eq!(compute_workgroups_8x8(16, 16), (2, 2));
        assert_eq!(compute_workgroups_8x8(64, 64), (8, 8));

        assert_eq!(compute_workgroups_blur_h(128, 100), (1, 100));
        assert_eq!(compute_workgroups_blur_h(256, 100), (2, 100));

        assert_eq!(compute_workgroups_blur_v(100, 128), (100, 1));
        assert_eq!(compute_workgroups_blur_v(100, 256), (100, 2));
    }

    #[test]
    fn test_workgroups_non_power_of_two() {
        // Non-power-of-two dimensions
        assert_eq!(compute_workgroups_8x8(100, 75), (13, 10)); // ceil(100/8)=13, ceil(75/8)=10
        assert_eq!(compute_workgroups_8x8(1920, 1080), (240, 135)); // Standard HD

        assert_eq!(compute_workgroups_blur_h(1000, 500), (8, 500)); // ceil(1000/128)=8
        assert_eq!(compute_workgroups_blur_v(500, 1000), (500, 8)); // ceil(1000/128)=8
    }

    #[test]
    fn test_workgroups_odd_dimensions() {
        // Odd numbers
        assert_eq!(compute_workgroups_8x8(7, 7), (1, 1));
        assert_eq!(compute_workgroups_8x8(9, 9), (2, 2));
        assert_eq!(compute_workgroups_8x8(127, 127), (16, 16));

        assert_eq!(compute_workgroups_blur_h(127, 99), (1, 99));
        assert_eq!(compute_workgroups_blur_h(129, 99), (2, 99));
    }

    #[test]
    fn test_workgroups_large_images() {
        // 4K resolution
        assert_eq!(compute_workgroups_8x8(3840, 2160), (480, 270));
        // 8K resolution
        assert_eq!(compute_workgroups_8x8(7680, 4320), (960, 540));

        // Blur workgroups for 4K
        assert_eq!(compute_workgroups_blur_h(3840, 2160), (30, 2160)); // ceil(3840/128)=30
        assert_eq!(compute_workgroups_blur_v(3840, 2160), (3840, 17)); // ceil(2160/128)=17
    }

    // -------------------------------------------------------------------------
    // 5. DownsampleUniforms with explicit destination size
    // -------------------------------------------------------------------------

    #[test]
    fn test_downsample_uniforms_explicit_dst() {
        let uniforms = DownsampleUniforms::with_dst_size(
            1920, 1080, // src
            960, 540,   // dst (half)
            FilterMode::Bilinear,
            1,
        );
        assert_eq!(uniforms.src_dims, [1920, 1080]);
        assert_eq!(uniforms.dst_dims, [960, 540]);
        assert_eq!(uniforms.filter_mode, 1);
        assert_eq!(uniforms.mip_level, 1);
    }

    #[test]
    fn test_downsample_uniforms_non_half_dst() {
        // Non-standard downsampling (e.g., for custom aspect ratios)
        let uniforms = DownsampleUniforms::with_dst_size(
            1920, 1080, // src
            1280, 720,  // dst (not exactly half)
            FilterMode::Box,
            0,
        );
        assert_eq!(uniforms.dst_dims, [1280, 720]);
    }

    #[test]
    fn test_downsample_uniforms_quarter_resolution() {
        // Quarter resolution (two mip levels down)
        let uniforms = DownsampleUniforms::with_dst_size(
            2048, 2048,
            512, 512,
            FilterMode::Karis,
            2,
        );
        assert_eq!(uniforms.src_dims, [2048, 2048]);
        assert_eq!(uniforms.dst_dims, [512, 512]);
        assert_eq!(uniforms.mip_level, 2);
    }

    // -------------------------------------------------------------------------
    // 6. Edge cases: HDR value ranges and special values
    // -------------------------------------------------------------------------

    #[test]
    fn test_tonemap_exposure_range() {
        // Negative exposure (darken)
        let dark = TonemapUniforms::with_exposure(100, 100, -3.0);
        assert_eq!(dark.exposure, -3.0);

        // Zero exposure (neutral)
        let neutral = TonemapUniforms::with_exposure(100, 100, 0.0);
        assert_eq!(neutral.exposure, 0.0);

        // Positive exposure (brighten)
        let bright = TonemapUniforms::with_exposure(100, 100, 5.0);
        assert_eq!(bright.exposure, 5.0);
    }

    #[test]
    fn test_histogram_luminance_range() {
        // Very dark scene
        let dark = HistogramUniforms::with_range(100, 100, -20.0, -5.0);
        assert_eq!(dark.min_luminance, -20.0);
        assert_eq!(dark.max_luminance, -5.0);

        // Very bright HDR scene
        let bright = HistogramUniforms::with_range(100, 100, 0.0, 20.0);
        assert_eq!(bright.min_luminance, 0.0);
        assert_eq!(bright.max_luminance, 20.0);

        // Wide range (extreme HDR)
        let wide = HistogramUniforms::with_range(100, 100, -15.0, 15.0);
        assert!((wide.max_luminance - wide.min_luminance - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tonemap_gamma_values() {
        // sRGB gamma
        let srgb = TonemapUniforms::full(100, 100, 0.0, 2.2, TonemapMode::Aces, 4.0);
        assert!((srgb.gamma - 2.2).abs() < f32::EPSILON);

        // Linear (gamma 1.0)
        let linear = TonemapUniforms::full(100, 100, 0.0, 1.0, TonemapMode::Aces, 4.0);
        assert!((linear.gamma - 1.0).abs() < f32::EPSILON);

        // Rec.709 gamma
        let rec709 = TonemapUniforms::full(100, 100, 0.0, 2.4, TonemapMode::Aces, 4.0);
        assert!((rec709.gamma - 2.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tonemap_white_point_values() {
        // Low white point (more aggressive tonemapping)
        let low = TonemapUniforms::full(100, 100, 0.0, 2.2, TonemapMode::Reinhard, 2.0);
        assert_eq!(low.white_point, 2.0);

        // High white point (preserves more highlights)
        let high = TonemapUniforms::full(100, 100, 0.0, 2.2, TonemapMode::Reinhard, 16.0);
        assert_eq!(high.white_point, 16.0);
    }

    // -------------------------------------------------------------------------
    // 7. Constants verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants() {
        assert_eq!(BLUR_WORKGROUP_SIZE, 128);
        assert_eq!(IMAGE_WORKGROUP_SIZE, 8);
        assert_eq!(HISTOGRAM_BINS, 256);
        assert_eq!(BLUR_KERNEL_RADIUS, 4);
    }

    // -------------------------------------------------------------------------
    // 8. Uniform struct copy and debug traits
    // -------------------------------------------------------------------------

    #[test]
    fn test_blur_uniforms_copy_debug() {
        let uniforms = BlurUniforms::new(640, 480);
        let copied = uniforms; // Copy
        assert_eq!(copied.src_dims, uniforms.src_dims);

        let debug_str = format!("{:?}", uniforms);
        assert!(debug_str.contains("BlurUniforms"));
        assert!(debug_str.contains("640"));
    }

    #[test]
    fn test_downsample_uniforms_copy_debug() {
        let uniforms = DownsampleUniforms::new(1024, 768, FilterMode::Box, 0);
        let copied = uniforms;
        assert_eq!(copied.src_dims, uniforms.src_dims);

        let debug_str = format!("{:?}", uniforms);
        assert!(debug_str.contains("DownsampleUniforms"));
    }

    #[test]
    fn test_histogram_uniforms_copy_debug() {
        let uniforms = HistogramUniforms::new(800, 600);
        let copied = uniforms;
        assert_eq!(copied.num_pixels, 800 * 600);

        let debug_str = format!("{:?}", uniforms);
        assert!(debug_str.contains("HistogramUniforms"));
    }

    #[test]
    fn test_tonemap_uniforms_copy_debug() {
        let uniforms = TonemapUniforms::new(1920, 1080);
        let copied = uniforms;
        assert_eq!(copied.gamma, 2.2);

        let debug_str = format!("{:?}", uniforms);
        assert!(debug_str.contains("TonemapUniforms"));
    }

    // -------------------------------------------------------------------------
    // 9. Edge cases: zero and minimum dimensions
    // -------------------------------------------------------------------------

    #[test]
    fn test_uniforms_zero_dimensions() {
        // Zero dimensions (edge case)
        let blur = BlurUniforms::new(0, 0);
        assert_eq!(blur.src_dims, [0, 0]);

        let histogram = HistogramUniforms::new(0, 0);
        assert_eq!(histogram.num_pixels, 0);
    }

    #[test]
    fn test_workgroups_zero_dimensions() {
        // Zero width/height shouldn't panic (ceil division)
        let (x, y) = compute_workgroups_8x8(0, 0);
        assert_eq!((x, y), (0, 0));

        let (x, y) = compute_workgroups_blur_h(0, 0);
        assert_eq!((x, y), (0, 0));

        let (x, y) = compute_workgroups_blur_v(0, 0);
        assert_eq!((x, y), (0, 0));
    }

    // -------------------------------------------------------------------------
    // 10. Downsample odd dimension handling
    // -------------------------------------------------------------------------

    #[test]
    fn test_downsample_odd_dimensions() {
        // Odd source dimensions (integer division truncates)
        let uniforms = DownsampleUniforms::new(1921, 1081, FilterMode::Box, 0);
        assert_eq!(uniforms.src_dims, [1921, 1081]);
        assert_eq!(uniforms.dst_dims, [960, 540]); // 1921/2=960, 1081/2=540

        // Single pixel source
        let single = DownsampleUniforms::new(1, 1, FilterMode::Box, 0);
        assert_eq!(single.dst_dims, [0, 0]); // 1/2=0
    }

    #[test]
    fn test_filter_mode_all_variants() {
        let modes = [FilterMode::Box, FilterMode::Bilinear, FilterMode::Karis];
        for (i, mode) in modes.iter().enumerate() {
            assert_eq!(mode.as_u32(), i as u32);
        }
    }
}
