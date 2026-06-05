//! Mipmap generation for TRINITY textures (T-WGPU-P2.3.4).
//!
//! This module provides GPU-accelerated mipmap generation using compute shaders.
//! It supports both power-of-two and non-power-of-two (NPOT) textures, with
//! configurable filter modes for different quality/performance tradeoffs.
//!
//! # Overview
//!
//! Mipmaps are pre-computed lower-resolution versions of textures that improve
//! rendering quality and performance when textures are viewed at smaller sizes.
//! This module generates mipmaps via GPU compute shaders rather than CPU,
//! providing significant performance benefits for runtime texture processing.
//!
//! # Filter Modes
//!
//! | Filter | Quality | Performance | Use Case |
//! |--------|---------|-------------|----------|
//! | Box | Good | Fast | Default for most textures |
//! | Bilinear | Better | Moderate | UI, text, detailed textures |
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::resources::mip_generator::{MipGenerator, MipFilter};
//!
//! // Create the generator once at startup
//! let generator = MipGenerator::new(&device);
//!
//! // Generate mips for a texture
//! generator.generate_mips(&mut encoder, &texture, MipFilter::Box);
//!
//! // Or generate a specific range
//! generator.generate_mip_range(&mut encoder, &texture, 0, 3, MipFilter::Bilinear);
//! ```
//!
//! # Format Support
//!
//! The generator supports common color formats. Use [`is_format_supported`] to
//! check if a format can be processed.
//!
//! Supported formats:
//! - Rgba8Unorm, Rgba8UnormSrgb
//! - Bgra8Unorm, Bgra8UnormSrgb
//! - Rgba16Float
//! - R8Unorm, Rg8Unorm
//! - R16Float, Rg16Float
//!
//! # NPOT Handling
//!
//! Non-power-of-two textures are handled correctly:
//! - Dimensions are floor-divided by 2 at each mip level
//! - Minimum dimension is 1 at the final mip level
//! - Edge texels are clamped to prevent artifacts

use std::borrow::Cow;
use std::num::NonZeroU64;

use log::debug;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, CommandEncoder,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, SamplerBindingType,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess,
    Texture, TextureFormat, TextureSampleType, TextureView, TextureViewDescriptor,
    TextureViewDimension,
};

// ============================================================================
// Constants
// ============================================================================

/// Compute shader workgroup size (8x8 threads).
pub const WORKGROUP_SIZE: u32 = 8;

/// Minimum texture dimension before stopping mip generation.
pub const MIN_MIP_DIMENSION: u32 = 1;

/// Uniform buffer size for mip generation parameters.
const UNIFORM_BUFFER_SIZE: u64 = 32; // 8 u32s = 32 bytes

// ============================================================================
// MipFilter
// ============================================================================

/// Filter mode for mipmap generation.
///
/// The filter mode determines how source texels are combined when
/// downsampling to create the next mip level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum MipFilter {
    /// Box filter (2x2 average).
    ///
    /// Simple average of 4 source texels. Fast and produces good results
    /// for most textures. This is the default and recommended filter.
    ///
    /// - Quality: Good
    /// - Performance: Best
    /// - Use for: General textures, diffuse maps, normal maps
    #[default]
    Box = 0,

    /// Bilinear filter.
    ///
    /// Uses hardware texture sampling for smoother gradients. Produces
    /// slightly better results than box filter at a small performance cost.
    ///
    /// - Quality: Better
    /// - Performance: Good
    /// - Use for: UI textures, text, detailed color gradients
    Bilinear = 1,
}

impl MipFilter {
    /// Returns the shader uniform value for this filter mode.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Returns a human-readable name for this filter mode.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            MipFilter::Box => "Box",
            MipFilter::Bilinear => "Bilinear",
        }
    }

    /// Returns all available filter modes.
    #[inline]
    pub const fn all() -> [MipFilter; 2] {
        [MipFilter::Box, MipFilter::Bilinear]
    }
}

// ============================================================================
// Format Support
// ============================================================================

/// Storage format information for a texture format.
#[derive(Debug, Clone, Copy)]
pub struct StorageFormatInfo {
    /// The storage texture format to use.
    pub storage_format: TextureFormat,
    /// Whether the format requires a separate pipeline.
    pub needs_custom_pipeline: bool,
}

/// Checks if a texture format is supported for mip generation.
///
/// # Arguments
///
/// * `format` - The texture format to check.
///
/// # Returns
///
/// `true` if the format can be used with the mip generator.
///
/// # Supported Formats
///
/// - Rgba8Unorm, Rgba8UnormSrgb
/// - Bgra8Unorm, Bgra8UnormSrgb
/// - Rgba16Float
/// - R8Unorm, Rg8Unorm
/// - R16Float, Rg16Float
/// - R32Float, Rg32Float, Rgba32Float
pub const fn is_format_supported(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::Rgba16Float
            | TextureFormat::R8Unorm
            | TextureFormat::Rg8Unorm
            | TextureFormat::R16Float
            | TextureFormat::Rg16Float
            | TextureFormat::R32Float
            | TextureFormat::Rg32Float
            | TextureFormat::Rgba32Float
            | TextureFormat::Rg11b10Float
    )
}

/// Returns the storage texture format for a given input format.
///
/// Some formats map to different storage formats due to wgpu storage
/// texture format restrictions.
///
/// # Arguments
///
/// * `format` - The texture format to query.
///
/// # Returns
///
/// The storage format information, or `None` if the format is not supported.
pub const fn storage_format_for(format: TextureFormat) -> Option<StorageFormatInfo> {
    match format {
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => Some(StorageFormatInfo {
            storage_format: TextureFormat::Rgba8Unorm,
            needs_custom_pipeline: false,
        }),
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => Some(StorageFormatInfo {
            storage_format: TextureFormat::Rgba8Unorm, // BGRA not supported as storage
            needs_custom_pipeline: true,               // May need channel swizzle
        }),
        TextureFormat::Rgba16Float => Some(StorageFormatInfo {
            storage_format: TextureFormat::Rgba16Float,
            needs_custom_pipeline: true,
        }),
        TextureFormat::R8Unorm => Some(StorageFormatInfo {
            storage_format: TextureFormat::R8Unorm,
            needs_custom_pipeline: true,
        }),
        TextureFormat::Rg8Unorm => Some(StorageFormatInfo {
            storage_format: TextureFormat::Rg8Unorm,
            needs_custom_pipeline: true,
        }),
        TextureFormat::R16Float => Some(StorageFormatInfo {
            storage_format: TextureFormat::R16Float,
            needs_custom_pipeline: true,
        }),
        TextureFormat::Rg16Float => Some(StorageFormatInfo {
            storage_format: TextureFormat::Rg16Float,
            needs_custom_pipeline: true,
        }),
        TextureFormat::R32Float => Some(StorageFormatInfo {
            storage_format: TextureFormat::R32Float,
            needs_custom_pipeline: true,
        }),
        TextureFormat::Rg32Float => Some(StorageFormatInfo {
            storage_format: TextureFormat::Rg32Float,
            needs_custom_pipeline: true,
        }),
        TextureFormat::Rgba32Float => Some(StorageFormatInfo {
            storage_format: TextureFormat::Rgba32Float,
            needs_custom_pipeline: true,
        }),
        TextureFormat::Rg11b10Float => Some(StorageFormatInfo {
            storage_format: TextureFormat::Rg11b10Float,
            needs_custom_pipeline: true,
        }),
        _ => None,
    }
}

/// Checks if a format is filterable (can use texture sampling).
///
/// Filterable formats support bilinear sampling, which is required
/// for the bilinear mip filter mode.
pub const fn is_filterable(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::R8Unorm
            | TextureFormat::Rg8Unorm
            | TextureFormat::Rgba8Unorm
            | TextureFormat::Rgba8UnormSrgb
            | TextureFormat::Bgra8Unorm
            | TextureFormat::Bgra8UnormSrgb
            | TextureFormat::R16Float
            | TextureFormat::Rg16Float
            | TextureFormat::Rgba16Float
            | TextureFormat::R32Float
            | TextureFormat::Rg32Float
            | TextureFormat::Rgba32Float
            | TextureFormat::Rg11b10Float
    )
}

// ============================================================================
// Mip Size Calculation
// ============================================================================

/// Calculates the size of a specific mip level.
///
/// Each mip level is half the size of the previous level, with a minimum
/// dimension of 1. This handles both power-of-two and non-power-of-two
/// textures correctly using floor division.
///
/// # Arguments
///
/// * `width` - Base texture width in pixels.
/// * `height` - Base texture height in pixels.
/// * `mip_level` - The mip level (0 = base resolution).
///
/// # Returns
///
/// (width, height) at the specified mip level.
///
/// # Examples
///
/// ```
/// use renderer_backend::resources::mip_generator::calculate_mip_size;
///
/// // Power-of-two texture
/// assert_eq!(calculate_mip_size(1024, 1024, 0), (1024, 1024));
/// assert_eq!(calculate_mip_size(1024, 1024, 1), (512, 512));
/// assert_eq!(calculate_mip_size(1024, 1024, 10), (1, 1));
///
/// // Non-power-of-two texture
/// assert_eq!(calculate_mip_size(100, 100, 0), (100, 100));
/// assert_eq!(calculate_mip_size(100, 100, 1), (50, 50));
/// assert_eq!(calculate_mip_size(100, 100, 2), (25, 25));
/// assert_eq!(calculate_mip_size(100, 100, 3), (12, 12));
///
/// // Non-square texture
/// assert_eq!(calculate_mip_size(256, 64, 0), (256, 64));
/// assert_eq!(calculate_mip_size(256, 64, 1), (128, 32));
/// assert_eq!(calculate_mip_size(256, 64, 6), (4, 1));
///
/// // Minimum dimension is 1
/// assert_eq!(calculate_mip_size(4, 4, 10), (1, 1));
/// ```
#[inline]
pub const fn calculate_mip_size(width: u32, height: u32, mip_level: u32) -> (u32, u32) {
    let shifted_w = width >> mip_level;
    let shifted_h = height >> mip_level;
    let mip_width = if shifted_w > 0 { shifted_w } else { 1 };
    let mip_height = if shifted_h > 0 { shifted_h } else { 1 };
    (mip_width, mip_height)
}

/// Calculates the number of mip levels for a texture.
///
/// The mip chain continues until the smallest dimension reaches 1.
///
/// # Arguments
///
/// * `width` - Texture width in pixels.
/// * `height` - Texture height in pixels.
///
/// # Returns
///
/// The number of mip levels including the base level.
///
/// # Examples
///
/// ```
/// use renderer_backend::resources::mip_generator::calculate_mip_levels;
///
/// assert_eq!(calculate_mip_levels(1024, 1024), 11);  // 1024 -> 512 -> ... -> 1
/// assert_eq!(calculate_mip_levels(512, 256), 10);   // 512 -> 256 -> ... -> 1
/// assert_eq!(calculate_mip_levels(1, 1), 1);        // Already minimum
/// assert_eq!(calculate_mip_levels(100, 100), 7);    // 100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1
/// assert_eq!(calculate_mip_levels(4, 4), 3);        // 4 -> 2 -> 1
/// ```
#[inline]
pub const fn calculate_mip_levels(width: u32, height: u32) -> u32 {
    let max_dim = if width > height { width } else { height };
    if max_dim == 0 {
        return 1;
    }
    // log2(max_dim) + 1 = number of mip levels
    32 - max_dim.leading_zeros()
}

// ============================================================================
// MipUniforms
// ============================================================================

/// Uniform data for mip generation shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MipUniforms {
    /// Source mip level dimensions.
    src_width: u32,
    src_height: u32,
    /// Destination mip level dimensions.
    dst_width: u32,
    dst_height: u32,
    /// Filter mode (0 = box, 1 = bilinear).
    filter_mode: u32,
    /// Padding for 16-byte alignment.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ============================================================================
// MipGenerator
// ============================================================================

/// GPU-accelerated mipmap generator.
///
/// This struct holds the compute pipeline and bind group layout needed
/// for mip generation. Create it once at startup and reuse it for all
/// mip generation operations.
///
/// # Example
///
/// ```ignore
/// let generator = MipGenerator::new(&device);
///
/// // Generate all mips for a texture
/// generator.generate_mips(&mut encoder, &queue, &texture, MipFilter::Box);
///
/// // Generate a specific range of mips
/// generator.generate_mip_range(&mut encoder, &queue, &texture, 0, 5, MipFilter::Bilinear);
/// ```
pub struct MipGenerator {
    /// Compute pipeline for mip generation.
    pipeline: ComputePipeline,
    /// Bind group layout for shader resources.
    bind_group_layout: BindGroupLayout,
    /// Linear sampler for bilinear filtering.
    sampler: wgpu::Sampler,
}

impl MipGenerator {
    /// Creates a new mip generator.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let generator = MipGenerator::new(&device);
    /// ```
    pub fn new(device: &Device) -> Self {
        // Create shader module
        let shader_source = include_str!("../../shaders/mip_generate.comp.wgsl");
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mip_generate_shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mip_generate_bind_group_layout"),
            entries: &[
                // Source texture (sampled)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Destination texture (storage)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Uniforms
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(UNIFORM_BUFFER_SIZE),
                    },
                    count: None,
                },
                // Linear sampler
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mip_generate_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("mip_generate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "mip_downsample",
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create linear sampler for bilinear filtering
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("mip_generate_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        debug!("MipGenerator created with {} workgroup size", WORKGROUP_SIZE);

        Self {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }

    /// Generates all mip levels for a texture.
    ///
    /// This method generates all mip levels from level 0 (base) down to
    /// the smallest mip level. The texture must have been created with
    /// the appropriate mip level count.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (for creating bind groups).
    /// * `queue` - The wgpu queue (for uploading uniforms).
    /// * `encoder` - The command encoder to record commands to.
    /// * `texture` - The texture to generate mips for.
    /// * `filter` - The filter mode to use.
    ///
    /// # Panics
    ///
    /// Panics if the texture format is not supported.
    ///
    /// # Example
    ///
    /// ```ignore
    /// generator.generate_mips(&device, &queue, &mut encoder, &texture, MipFilter::Box);
    /// ```
    pub fn generate_mips(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        texture: &Texture,
        filter: MipFilter,
    ) {
        let mip_count = texture.mip_level_count();
        if mip_count <= 1 {
            return; // No mips to generate
        }

        self.generate_mip_range(device, queue, encoder, texture, 0, mip_count - 1, filter);
    }

    /// Generates a specific range of mip levels.
    ///
    /// This method generates mip levels starting from `base_mip` and
    /// continuing for `mip_count` levels. The source for each mip level
    /// is the previous mip level.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (for creating bind groups).
    /// * `queue` - The wgpu queue (for uploading uniforms).
    /// * `encoder` - The command encoder to record commands to.
    /// * `texture` - The texture to generate mips for.
    /// * `base_mip` - The source mip level (destination is base_mip + 1).
    /// * `mip_count` - Number of mip levels to generate.
    /// * `filter` - The filter mode to use.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The texture format is not supported
    /// - base_mip + mip_count exceeds the texture's mip level count
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Generate mips 1-3 from mip 0
    /// generator.generate_mip_range(
    ///     &device, &queue, &mut encoder, &texture,
    ///     0, 3, MipFilter::Bilinear
    /// );
    /// ```
    pub fn generate_mip_range(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        texture: &Texture,
        base_mip: u32,
        mip_count: u32,
        filter: MipFilter,
    ) {
        let texture_mip_count = texture.mip_level_count();
        let texture_size = texture.size();
        let format = texture.format();

        // Validate format
        assert!(
            is_format_supported(format),
            "Texture format {:?} is not supported for mip generation",
            format
        );

        // Validate mip range
        assert!(
            base_mip + mip_count < texture_mip_count,
            "Mip range {}..{} exceeds texture mip count {}",
            base_mip,
            base_mip + mip_count,
            texture_mip_count
        );

        debug!(
            "Generating {} mip levels for {}x{} texture (filter: {})",
            mip_count,
            texture_size.width,
            texture_size.height,
            filter.name()
        );

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mip_generate_uniforms"),
            size: UNIFORM_BUFFER_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Generate each mip level
        for i in 0..mip_count {
            let src_mip = base_mip + i;
            let dst_mip = src_mip + 1;

            let (src_width, src_height) =
                calculate_mip_size(texture_size.width, texture_size.height, src_mip);
            let (dst_width, dst_height) =
                calculate_mip_size(texture_size.width, texture_size.height, dst_mip);

            // Create views for source and destination mip levels
            let src_view = texture.create_view(&TextureViewDescriptor {
                label: Some(&format!("mip_src_view_{}", src_mip)),
                format: Some(format),
                dimension: Some(TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: src_mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });

            let dst_view = texture.create_view(&TextureViewDescriptor {
                label: Some(&format!("mip_dst_view_{}", dst_mip)),
                format: Some(format),
                dimension: Some(TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: dst_mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });

            // Update uniforms
            let uniforms = MipUniforms {
                src_width,
                src_height,
                dst_width,
                dst_height,
                filter_mode: filter.as_u32(),
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

            // Create bind group
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some(&format!("mip_generate_bind_group_{}", dst_mip)),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&src_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&dst_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            // Dispatch compute shader
            let workgroups_x = (dst_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let workgroups_y = (dst_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

            {
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some(&format!("mip_generate_pass_{}", dst_mip)),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            debug!(
                "Generated mip {} ({}x{}) -> mip {} ({}x{}), workgroups: {}x{}",
                src_mip, src_width, src_height, dst_mip, dst_width, dst_height, workgroups_x, workgroups_y
            );
        }
    }

    /// Creates a bind group for a single mip generation pass.
    ///
    /// This is a lower-level API for cases where you need more control
    /// over the mip generation process.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `src_view` - View of the source mip level.
    /// * `dst_view` - View of the destination mip level.
    /// * `uniform_buffer` - Buffer containing MipUniforms.
    ///
    /// # Returns
    ///
    /// A bind group ready for use with the mip generation pipeline.
    pub fn create_bind_group(
        &self,
        device: &Device,
        src_view: &TextureView,
        dst_view: &TextureView,
        uniform_buffer: &wgpu::Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("mip_generate_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(src_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(dst_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Returns the compute pipeline for custom dispatch.
    ///
    /// Use this for advanced use cases where you need to integrate
    /// mip generation into a custom render graph.
    #[inline]
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    /// Returns the bind group layout.
    ///
    /// Use this for advanced use cases where you need to create
    /// custom bind groups.
    #[inline]
    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }

    /// Dispatches a mip generation pass with a pre-created bind group.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder.
    /// * `bind_group` - The bind group containing resources.
    /// * `dst_width` - Width of the destination mip level.
    /// * `dst_height` - Height of the destination mip level.
    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
        dst_width: u32,
        dst_height: u32,
    ) {
        let workgroups_x = (dst_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (dst_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("mip_generate_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}

// ============================================================================
// MipChainInfo
// ============================================================================

/// Information about a mip chain for planning purposes.
///
/// Use this to pre-calculate mip chain properties without a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipChainInfo {
    /// Base texture width.
    pub width: u32,
    /// Base texture height.
    pub height: u32,
    /// Number of mip levels.
    pub mip_levels: u32,
}

impl MipChainInfo {
    /// Creates mip chain info for the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `width` - Base texture width.
    /// * `height` - Base texture height.
    ///
    /// # Returns
    ///
    /// Mip chain information including the calculated mip level count.
    #[inline]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            mip_levels: calculate_mip_levels(width, height),
        }
    }

    /// Creates mip chain info with explicit mip count.
    ///
    /// # Arguments
    ///
    /// * `width` - Base texture width.
    /// * `height` - Base texture height.
    /// * `mip_levels` - Number of mip levels.
    #[inline]
    pub const fn with_mip_count(width: u32, height: u32, mip_levels: u32) -> Self {
        Self {
            width,
            height,
            mip_levels,
        }
    }

    /// Gets the size at a specific mip level.
    ///
    /// # Arguments
    ///
    /// * `mip_level` - The mip level (0 = base).
    ///
    /// # Returns
    ///
    /// (width, height) at the specified level.
    #[inline]
    pub const fn mip_size(&self, mip_level: u32) -> (u32, u32) {
        calculate_mip_size(self.width, self.height, mip_level)
    }

    /// Calculates total texels across all mip levels.
    ///
    /// Useful for memory estimation.
    pub const fn total_texels(&self) -> u64 {
        let mut total = 0u64;
        let mut mip = 0u32;
        while mip < self.mip_levels {
            let (w, h) = self.mip_size(mip);
            total += (w as u64) * (h as u64);
            mip += 1;
        }
        total
    }

    /// Checks if this is a power-of-two texture.
    #[inline]
    pub const fn is_power_of_two(&self) -> bool {
        self.width.is_power_of_two() && self.height.is_power_of_two()
    }

    /// Returns the smallest dimension at any mip level.
    ///
    /// This is always 1 for a complete mip chain.
    #[inline]
    pub const fn smallest_dimension(&self) -> u32 {
        let (w, h) = self.mip_size(self.mip_levels.saturating_sub(1));
        if w < h {
            w
        } else {
            h
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_mip_size_power_of_two() {
        // 1024x1024 texture
        assert_eq!(calculate_mip_size(1024, 1024, 0), (1024, 1024));
        assert_eq!(calculate_mip_size(1024, 1024, 1), (512, 512));
        assert_eq!(calculate_mip_size(1024, 1024, 2), (256, 256));
        assert_eq!(calculate_mip_size(1024, 1024, 10), (1, 1));

        // 512x256 texture
        assert_eq!(calculate_mip_size(512, 256, 0), (512, 256));
        assert_eq!(calculate_mip_size(512, 256, 1), (256, 128));
        assert_eq!(calculate_mip_size(512, 256, 8), (2, 1));
        assert_eq!(calculate_mip_size(512, 256, 9), (1, 1));
    }

    #[test]
    fn test_calculate_mip_size_npot() {
        // 100x100 NPOT texture
        assert_eq!(calculate_mip_size(100, 100, 0), (100, 100));
        assert_eq!(calculate_mip_size(100, 100, 1), (50, 50));
        assert_eq!(calculate_mip_size(100, 100, 2), (25, 25));
        assert_eq!(calculate_mip_size(100, 100, 3), (12, 12));
        assert_eq!(calculate_mip_size(100, 100, 4), (6, 6));
        assert_eq!(calculate_mip_size(100, 100, 5), (3, 3));
        assert_eq!(calculate_mip_size(100, 100, 6), (1, 1));

        // 300x200 NPOT texture
        assert_eq!(calculate_mip_size(300, 200, 0), (300, 200));
        assert_eq!(calculate_mip_size(300, 200, 1), (150, 100));
        assert_eq!(calculate_mip_size(300, 200, 2), (75, 50));
        assert_eq!(calculate_mip_size(300, 200, 3), (37, 25));
    }

    #[test]
    fn test_calculate_mip_size_non_square() {
        // Wide texture
        assert_eq!(calculate_mip_size(256, 64, 0), (256, 64));
        assert_eq!(calculate_mip_size(256, 64, 1), (128, 32));
        assert_eq!(calculate_mip_size(256, 64, 2), (64, 16));
        assert_eq!(calculate_mip_size(256, 64, 6), (4, 1));
        assert_eq!(calculate_mip_size(256, 64, 8), (1, 1));

        // Tall texture
        assert_eq!(calculate_mip_size(64, 256, 0), (64, 256));
        assert_eq!(calculate_mip_size(64, 256, 1), (32, 128));
        assert_eq!(calculate_mip_size(64, 256, 6), (1, 4));
    }

    #[test]
    fn test_calculate_mip_size_minimum_dimension() {
        // Dimension stays at 1 once reached
        assert_eq!(calculate_mip_size(4, 4, 10), (1, 1));
        assert_eq!(calculate_mip_size(1, 1, 0), (1, 1));
        assert_eq!(calculate_mip_size(1, 1, 5), (1, 1));
        assert_eq!(calculate_mip_size(2, 1, 1), (1, 1));
    }

    #[test]
    fn test_calculate_mip_levels_power_of_two() {
        assert_eq!(calculate_mip_levels(1024, 1024), 11);
        assert_eq!(calculate_mip_levels(512, 512), 10);
        assert_eq!(calculate_mip_levels(256, 256), 9);
        assert_eq!(calculate_mip_levels(128, 128), 8);
        assert_eq!(calculate_mip_levels(64, 64), 7);
        assert_eq!(calculate_mip_levels(4, 4), 3);
        assert_eq!(calculate_mip_levels(2, 2), 2);
        assert_eq!(calculate_mip_levels(1, 1), 1);
    }

    #[test]
    fn test_calculate_mip_levels_npot() {
        // NPOT textures: log2(max_dim) + 1
        assert_eq!(calculate_mip_levels(100, 100), 7); // log2(100) ~ 6.64, so 7
        assert_eq!(calculate_mip_levels(300, 200), 9); // log2(300) ~ 8.23, so 9
        assert_eq!(calculate_mip_levels(1920, 1080), 11); // log2(1920) ~ 10.9, so 11
    }

    #[test]
    fn test_calculate_mip_levels_non_square() {
        // Non-square uses max dimension
        assert_eq!(calculate_mip_levels(512, 256), 10); // Based on 512
        assert_eq!(calculate_mip_levels(256, 512), 10); // Based on 512
        assert_eq!(calculate_mip_levels(1024, 1), 11); // Based on 1024
        assert_eq!(calculate_mip_levels(1, 1024), 11); // Based on 1024
    }

    #[test]
    fn test_mip_filter_enum() {
        assert_eq!(MipFilter::Box.as_u32(), 0);
        assert_eq!(MipFilter::Bilinear.as_u32(), 1);
        assert_eq!(MipFilter::Box.name(), "Box");
        assert_eq!(MipFilter::Bilinear.name(), "Bilinear");
        assert_eq!(MipFilter::default(), MipFilter::Box);
        assert_eq!(MipFilter::all(), [MipFilter::Box, MipFilter::Bilinear]);
    }

    #[test]
    fn test_is_format_supported() {
        // Supported formats
        assert!(is_format_supported(TextureFormat::Rgba8Unorm));
        assert!(is_format_supported(TextureFormat::Rgba8UnormSrgb));
        assert!(is_format_supported(TextureFormat::Bgra8Unorm));
        assert!(is_format_supported(TextureFormat::Bgra8UnormSrgb));
        assert!(is_format_supported(TextureFormat::Rgba16Float));
        assert!(is_format_supported(TextureFormat::R8Unorm));
        assert!(is_format_supported(TextureFormat::Rg8Unorm));
        assert!(is_format_supported(TextureFormat::R32Float));

        // Unsupported formats
        assert!(!is_format_supported(TextureFormat::Depth32Float));
        assert!(!is_format_supported(TextureFormat::Bc1RgbaUnorm));
        assert!(!is_format_supported(TextureFormat::R8Uint));
    }

    #[test]
    fn test_is_filterable() {
        // Filterable formats
        assert!(is_filterable(TextureFormat::Rgba8Unorm));
        assert!(is_filterable(TextureFormat::Rgba16Float));
        assert!(is_filterable(TextureFormat::R32Float));

        // Non-filterable formats
        assert!(!is_filterable(TextureFormat::R8Uint));
        assert!(!is_filterable(TextureFormat::Rgba8Uint));
    }

    #[test]
    fn test_mip_chain_info() {
        let info = MipChainInfo::new(1024, 1024);
        assert_eq!(info.width, 1024);
        assert_eq!(info.height, 1024);
        assert_eq!(info.mip_levels, 11);
        assert!(info.is_power_of_two());

        let npot_info = MipChainInfo::new(100, 100);
        assert_eq!(npot_info.mip_levels, 7);
        assert!(!npot_info.is_power_of_two());
    }

    #[test]
    fn test_mip_chain_info_mip_sizes() {
        let info = MipChainInfo::new(256, 128);
        assert_eq!(info.mip_size(0), (256, 128));
        assert_eq!(info.mip_size(1), (128, 64));
        assert_eq!(info.mip_size(2), (64, 32));
        assert_eq!(info.mip_size(7), (2, 1));
        assert_eq!(info.mip_size(8), (1, 1));
    }

    #[test]
    fn test_mip_chain_info_total_texels() {
        // 4x4 with 3 mips: 16 + 4 + 1 = 21
        let small = MipChainInfo::new(4, 4);
        assert_eq!(small.total_texels(), 21);

        // 2x2 with 2 mips: 4 + 1 = 5
        let tiny = MipChainInfo::new(2, 2);
        assert_eq!(tiny.total_texels(), 5);

        // 1x1: just 1 texel
        let single = MipChainInfo::new(1, 1);
        assert_eq!(single.total_texels(), 1);
    }

    #[test]
    fn test_mip_chain_info_smallest_dimension() {
        let info = MipChainInfo::new(1024, 1024);
        assert_eq!(info.smallest_dimension(), 1);

        let info = MipChainInfo::with_mip_count(1024, 1024, 5);
        let (w, h) = info.mip_size(4);
        assert_eq!(info.smallest_dimension(), w.min(h));
    }

    #[test]
    fn test_storage_format_for() {
        // RGBA8 formats
        let rgba8 = storage_format_for(TextureFormat::Rgba8Unorm).unwrap();
        assert_eq!(rgba8.storage_format, TextureFormat::Rgba8Unorm);
        assert!(!rgba8.needs_custom_pipeline);

        let rgba8_srgb = storage_format_for(TextureFormat::Rgba8UnormSrgb).unwrap();
        assert_eq!(rgba8_srgb.storage_format, TextureFormat::Rgba8Unorm);

        // BGRA8 formats (need custom handling)
        let bgra8 = storage_format_for(TextureFormat::Bgra8Unorm).unwrap();
        assert!(bgra8.needs_custom_pipeline);

        // Float formats
        let rgba16f = storage_format_for(TextureFormat::Rgba16Float).unwrap();
        assert_eq!(rgba16f.storage_format, TextureFormat::Rgba16Float);
        assert!(rgba16f.needs_custom_pipeline);

        // Unsupported format
        assert!(storage_format_for(TextureFormat::Depth32Float).is_none());
    }

    #[test]
    fn test_mip_uniforms_size() {
        // Verify struct size matches UNIFORM_BUFFER_SIZE
        assert_eq!(std::mem::size_of::<MipUniforms>(), 32);
        assert_eq!(UNIFORM_BUFFER_SIZE, 32);
    }

    #[test]
    fn test_workgroup_dispatch_calculation() {
        // Test dispatch calculation matches shader workgroup size
        let dst_width = 100u32;
        let dst_height = 50u32;
        let workgroups_x = (dst_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (dst_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // 100 / 8 = 12.5 -> 13 workgroups
        assert_eq!(workgroups_x, 13);
        // 50 / 8 = 6.25 -> 7 workgroups
        assert_eq!(workgroups_y, 7);
    }

    #[test]
    fn test_mip_filter_all() {
        let all_filters = MipFilter::all();
        assert_eq!(all_filters.len(), 2);
        assert!(all_filters.contains(&MipFilter::Box));
        assert!(all_filters.contains(&MipFilter::Bilinear));
    }
}
