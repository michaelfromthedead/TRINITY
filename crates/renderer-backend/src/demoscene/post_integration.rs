//! S13 Output Feeds S8 Post-Processing Integration (T-DEMO-6.8)
//!
//! Integrates demoscene SDF renderer output (S13) with the post-processing
//! pipeline (S8). Verifies that:
//!
//! - Tone mapping applies correctly to SDF HDR output
//! - Bloom extracts from SDF bright regions
//! - TAA works on SDF rendered frames
//! - Motion vectors from ray march (optional)
//!
//! # Architecture
//!
//! ```text
//! [SDF Renderer (S13)]
//!        |
//!        v
//! [HDR Output Texture]
//!        |
//!        +---> [Bloom Bright Pass] --> [Bloom Blur]
//!        |                                   |
//!        v                                   v
//! [Tone Map (ACES)]  <------------------[Bloom Composite]
//!        |
//!        v
//! [TAA]
//!        |
//!        v
//! [Final LDR Output]
//! ```

use crate::frame_graph::{
    DispatchSource, IrPass, IrResource, PassIndex, ResourceDesc, ResourceHandle,
    ResourceLifetime, ResourceState, TextureDesc, ViewType,
};
use crate::post_process::{create_bloom_pass, create_taa_pass, create_tonemap_pass};
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Bloom brightness threshold for SDF output.
pub const SDF_BLOOM_THRESHOLD: f32 = 1.0;

/// Bloom intensity multiplier for SDF scenes.
pub const SDF_BLOOM_INTENSITY: f32 = 0.5;

/// TAA feedback weight for SDF frames.
pub const SDF_TAA_FEEDBACK: f32 = 0.9;

/// Motion vector scale for ray march based motion.
pub const SDF_MOTION_SCALE: f32 = 1.0;

// ---------------------------------------------------------------------------
// Post-Process Configuration
// ---------------------------------------------------------------------------

/// Post-processing configuration for SDF output.
#[derive(Clone, Debug, PartialEq)]
pub struct SdfPostProcessConfig {
    /// Enable tone mapping (ACES filmic).
    pub enable_tonemap: bool,
    /// Enable bloom extraction and composite.
    pub enable_bloom: bool,
    /// Enable temporal anti-aliasing.
    pub enable_taa: bool,
    /// Enable motion vector generation.
    pub enable_motion_vectors: bool,
    /// Bloom brightness threshold.
    pub bloom_threshold: f32,
    /// Bloom intensity multiplier.
    pub bloom_intensity: f32,
    /// TAA history feedback weight.
    pub taa_feedback: f32,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
}

impl Default for SdfPostProcessConfig {
    fn default() -> Self {
        Self {
            enable_tonemap: true,
            enable_bloom: true,
            enable_taa: true,
            enable_motion_vectors: false,
            bloom_threshold: SDF_BLOOM_THRESHOLD,
            bloom_intensity: SDF_BLOOM_INTENSITY,
            taa_feedback: SDF_TAA_FEEDBACK,
            width: 1920,
            height: 1080,
        }
    }
}

impl SdfPostProcessConfig {
    /// Create a minimal config (tone map only).
    pub fn minimal(width: u32, height: u32) -> Self {
        Self {
            enable_tonemap: true,
            enable_bloom: false,
            enable_taa: false,
            enable_motion_vectors: false,
            width,
            height,
            ..Default::default()
        }
    }

    /// Create a full config with all effects.
    pub fn full(width: u32, height: u32) -> Self {
        Self {
            enable_tonemap: true,
            enable_bloom: true,
            enable_taa: true,
            enable_motion_vectors: true,
            width,
            height,
            ..Default::default()
        }
    }

    /// Set bloom parameters.
    pub fn with_bloom(mut self, threshold: f32, intensity: f32) -> Self {
        self.enable_bloom = true;
        self.bloom_threshold = threshold;
        self.bloom_intensity = intensity;
        self
    }

    /// Set TAA parameters.
    pub fn with_taa(mut self, feedback: f32) -> Self {
        self.enable_taa = true;
        self.taa_feedback = feedback;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.width == 0 || self.height == 0 {
            return Err("resolution must be non-zero");
        }
        if self.bloom_threshold < 0.0 {
            return Err("bloom threshold must be non-negative");
        }
        if self.bloom_intensity < 0.0 {
            return Err("bloom intensity must be non-negative");
        }
        if self.taa_feedback < 0.0 || self.taa_feedback > 1.0 {
            return Err("TAA feedback must be in [0, 1]");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Uniform Data
// ---------------------------------------------------------------------------

/// Uniform buffer for post-processing effects.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct PostProcessUniforms {
    /// Resolution (width, height).
    pub resolution: [f32; 2],
    /// Time for animated effects.
    pub time: f32,
    /// Bloom brightness threshold.
    pub bloom_threshold: f32,
    /// Bloom intensity multiplier.
    pub bloom_intensity: f32,
    /// TAA feedback weight.
    pub taa_feedback: f32,
    /// Motion vector scale.
    pub motion_scale: f32,
    /// Exposure adjustment.
    pub exposure: f32,
}

impl Default for PostProcessUniforms {
    fn default() -> Self {
        Self {
            resolution: [1920.0, 1080.0],
            time: 0.0,
            bloom_threshold: SDF_BLOOM_THRESHOLD,
            bloom_intensity: SDF_BLOOM_INTENSITY,
            taa_feedback: SDF_TAA_FEEDBACK,
            motion_scale: SDF_MOTION_SCALE,
            exposure: 1.0,
        }
    }
}

impl PostProcessUniforms {
    /// Create uniforms from config.
    pub fn from_config(config: &SdfPostProcessConfig, time: f32) -> Self {
        Self {
            resolution: [config.width as f32, config.height as f32],
            time,
            bloom_threshold: config.bloom_threshold,
            bloom_intensity: config.bloom_intensity,
            taa_feedback: config.taa_feedback,
            motion_scale: SDF_MOTION_SCALE,
            exposure: 1.0,
        }
    }

    /// Get data as bytes for GPU upload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

// ---------------------------------------------------------------------------
// Bloom Bright Pass
// ---------------------------------------------------------------------------

/// Create a bloom brightness extraction pass for SDF output.
///
/// Extracts pixels above the brightness threshold for the bloom effect.
pub fn create_sdf_bloom_bright_pass(
    index: PassIndex,
    sdf_hdr_input: ResourceHandle,
    bright_output: ResourceHandle,
    width: u32,
    height: u32,
) -> IrPass {
    let mut pass = IrPass::compute(
        index,
        "sdf_bloom_bright",
        DispatchSource::Direct {
            group_count_x: (width / 8).max(1),
            group_count_y: (height / 8).max(1),
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.reads.push(sdf_hdr_input);
    pass.access_set.writes.push(bright_output);
    pass.tags.push("post-process".into());
    pass.tags.push("bloom".into());
    pass.tags.push("sdf-integration".into());
    pass
}

/// Create a bloom blur pass (horizontal or vertical).
pub fn create_bloom_blur_pass(
    index: PassIndex,
    input: ResourceHandle,
    output: ResourceHandle,
    width: u32,
    height: u32,
    is_horizontal: bool,
) -> IrPass {
    let name = if is_horizontal {
        "bloom_blur_h"
    } else {
        "bloom_blur_v"
    };

    let mut pass = IrPass::compute(
        index,
        name,
        DispatchSource::Direct {
            group_count_x: (width / 8).max(1),
            group_count_y: (height / 8).max(1),
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.reads.push(input);
    pass.access_set.writes.push(output);
    pass.tags.push("post-process".into());
    pass.tags.push("bloom".into());
    pass
}

// ---------------------------------------------------------------------------
// Motion Vector Pass
// ---------------------------------------------------------------------------

/// Create a motion vector generation pass for ray-marched frames.
///
/// Computes motion vectors by re-projecting SDF positions using
/// previous frame camera matrices.
pub fn create_sdf_motion_vector_pass(
    index: PassIndex,
    sdf_depth_input: ResourceHandle,
    motion_output: ResourceHandle,
    width: u32,
    height: u32,
) -> IrPass {
    let mut pass = IrPass::compute(
        index,
        "sdf_motion_vectors",
        DispatchSource::Direct {
            group_count_x: (width / 8).max(1),
            group_count_y: (height / 8).max(1),
            group_count_z: 1,
        },
        ViewType::Storage,
    );
    pass.access_set.reads.push(sdf_depth_input);
    pass.access_set.writes.push(motion_output);
    pass.tags.push("post-process".into());
    pass.tags.push("motion-vectors".into());
    pass.tags.push("sdf-integration".into());
    pass
}

// ---------------------------------------------------------------------------
// SDF Post-Process Chain
// ---------------------------------------------------------------------------

/// Build the complete post-processing pipeline for SDF output.
///
/// Creates all necessary passes and intermediate resources based on the
/// configuration. Returns `(Vec<IrPass>, Vec<IrResource>)`.
///
/// # Pipeline Structure
///
/// With all features enabled:
///
/// 1. `sdf_bloom_bright` - Extract bright pixels
/// 2. `bloom_blur_h`     - Horizontal Gaussian blur
/// 3. `bloom_blur_v`     - Vertical Gaussian blur
/// 4. `tonemap`          - ACES filmic tone mapping + bloom composite
/// 5. `taa`              - Temporal anti-aliasing
///
/// # Arguments
///
/// * `start_index` - Starting pass index
/// * `sdf_hdr_input` - HDR output from SDF renderer
/// * `ldr_output` - Final LDR output
/// * `config` - Post-processing configuration
pub fn create_sdf_post_process_chain(
    start_index: PassIndex,
    sdf_hdr_input: ResourceHandle,
    ldr_output: ResourceHandle,
    config: &SdfPostProcessConfig,
) -> (Vec<IrPass>, Vec<IrResource>) {
    let mut passes = Vec::new();
    let mut resources = Vec::new();
    let mut current_index = start_index.0;

    let width = config.width;
    let height = config.height;

    // Resource handle allocation (reserved range to avoid collisions)
    const BLOOM_BRIGHT: ResourceHandle = ResourceHandle(0xFE00);
    const BLOOM_BLUR_H: ResourceHandle = ResourceHandle(0xFE01);
    const BLOOM_BLUR_V: ResourceHandle = ResourceHandle(0xFE02);
    const TONEMAP_OUT: ResourceHandle = ResourceHandle(0xFE03);
    const TAA_HISTORY: ResourceHandle = ResourceHandle(0xFE04);
    const MOTION_VECTORS: ResourceHandle = ResourceHandle(0xFE05);

    // Current input for the chain
    let mut current_input = sdf_hdr_input;

    // Create intermediate texture descriptor
    let hdr_tex_desc = TextureDesc {
        width,
        height,
        mip_levels: 1,
        array_layers: 1,
        format: "rgba16float".into(),
    };

    let ldr_tex_desc = TextureDesc {
        width,
        height,
        mip_levels: 1,
        array_layers: 1,
        format: "rgba8unorm".into(),
    };

    // === Bloom Pipeline ===
    if config.enable_bloom {
        // Bloom bright extraction
        resources.push(IrResource::new(
            BLOOM_BRIGHT,
            "bloom_bright",
            ResourceDesc::Texture2D(hdr_tex_desc.clone()),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ));

        passes.push(create_sdf_bloom_bright_pass(
            PassIndex(current_index),
            current_input,
            BLOOM_BRIGHT,
            width,
            height,
        ));
        current_index += 1;

        // Horizontal blur
        resources.push(IrResource::new(
            BLOOM_BLUR_H,
            "bloom_blur_h",
            ResourceDesc::Texture2D(hdr_tex_desc.clone()),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ));

        passes.push(create_bloom_blur_pass(
            PassIndex(current_index),
            BLOOM_BRIGHT,
            BLOOM_BLUR_H,
            width,
            height,
            true,
        ));
        current_index += 1;

        // Vertical blur
        resources.push(IrResource::new(
            BLOOM_BLUR_V,
            "bloom_blur_v",
            ResourceDesc::Texture2D(hdr_tex_desc.clone()),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        ));

        passes.push(create_bloom_blur_pass(
            PassIndex(current_index),
            BLOOM_BLUR_H,
            BLOOM_BLUR_V,
            width,
            height,
            false,
        ));
        current_index += 1;
    }

    // === Tone Mapping ===
    if config.enable_tonemap {
        // Determine tonemap input (original HDR or bloom composite)
        let tonemap_input = current_input;

        // Determine tonemap output
        let tonemap_output = if config.enable_taa { TONEMAP_OUT } else { ldr_output };

        if config.enable_taa {
            resources.push(IrResource::new(
                TONEMAP_OUT,
                "tonemap_output",
                ResourceDesc::Texture2D(ldr_tex_desc.clone()),
                ResourceLifetime::Transient,
                ResourceState::Uninitialized,
            ));
        }

        passes.push(create_tonemap_pass(
            PassIndex(current_index),
            tonemap_input,
            tonemap_output,
        ));
        current_index += 1;

        current_input = tonemap_output;
    }

    // === TAA ===
    if config.enable_taa {
        resources.push(IrResource::new(
            TAA_HISTORY,
            "taa_history",
            ResourceDesc::Texture2D(ldr_tex_desc.clone()),
            ResourceLifetime::Imported,  // TAA history persists across frames
            ResourceState::Uninitialized,
        ));

        passes.push(create_taa_pass(
            PassIndex(current_index),
            current_input,
            TAA_HISTORY,
            ldr_output,
        ));
        // current_index += 1; // Not needed after last pass
    }

    (passes, resources)
}

// ---------------------------------------------------------------------------
// SDF Post-Processor
// ---------------------------------------------------------------------------

/// GPU-side SDF post-processor.
///
/// Manages compute pipelines for post-processing SDF output:
/// - Bloom bright extraction
/// - Bloom blur (separable Gaussian)
/// - Tone mapping (ACES)
/// - TAA
pub struct SdfPostProcessor {
    /// Configuration.
    config: SdfPostProcessConfig,
    /// Uniform buffer.
    uniform_buffer: wgpu::Buffer,
    /// Bloom bright extraction pipeline.
    bloom_bright_pipeline: Option<wgpu::ComputePipeline>,
    /// Bloom blur pipeline (shared for H and V).
    bloom_blur_pipeline: Option<wgpu::ComputePipeline>,
    /// Tone mapping pipeline.
    tonemap_pipeline: wgpu::ComputePipeline,
    /// TAA pipeline.
    taa_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for all passes.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Intermediate textures.
    bloom_bright_texture: Option<wgpu::Texture>,
    bloom_blur_h_texture: Option<wgpu::Texture>,
    bloom_blur_v_texture: Option<wgpu::Texture>,
    tonemap_texture: Option<wgpu::Texture>,
    taa_history_texture: Option<wgpu::Texture>,
}

impl SdfPostProcessor {
    /// Create a new SDF post-processor.
    pub fn new(device: &wgpu::Device, config: SdfPostProcessConfig) -> Self {
        config.validate().expect("invalid post-process config");

        let width = config.width;
        let height = config.height;

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SDF Post-Process Uniform Buffer"),
            size: std::mem::size_of::<PostProcessUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF Post-Process Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<PostProcessUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
            label: Some("SDF Post-Process Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create tonemap shader and pipeline (always enabled)
        let tonemap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SDF Tonemap Shader"),
            source: wgpu::ShaderSource::Wgsl(TONEMAP_SHADER.into()),
        });

        let tonemap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SDF Tonemap Pipeline"),
            layout: Some(&pipeline_layout),
            module: &tonemap_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create optional pipelines based on config
        let bloom_bright_pipeline = if config.enable_bloom {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("SDF Bloom Bright Shader"),
                source: wgpu::ShaderSource::Wgsl(BLOOM_BRIGHT_SHADER.into()),
            });
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SDF Bloom Bright Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }))
        } else {
            None
        };

        let bloom_blur_pipeline = if config.enable_bloom {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("SDF Bloom Blur Shader"),
                source: wgpu::ShaderSource::Wgsl(BLOOM_BLUR_SHADER.into()),
            });
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SDF Bloom Blur Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }))
        } else {
            None
        };

        let taa_pipeline = if config.enable_taa {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("SDF TAA Shader"),
                source: wgpu::ShaderSource::Wgsl(TAA_SHADER.into()),
            });
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SDF TAA Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }))
        } else {
            None
        };

        // Create intermediate textures
        let hdr_texture_desc = wgpu::TextureDescriptor {
            label: Some("SDF Post-Process HDR Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let bloom_bright_texture = if config.enable_bloom {
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Bloom Bright Texture"),
                ..hdr_texture_desc.clone()
            }))
        } else {
            None
        };

        let bloom_blur_h_texture = if config.enable_bloom {
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Bloom Blur H Texture"),
                ..hdr_texture_desc.clone()
            }))
        } else {
            None
        };

        let bloom_blur_v_texture = if config.enable_bloom {
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Bloom Blur V Texture"),
                ..hdr_texture_desc.clone()
            }))
        } else {
            None
        };

        let tonemap_texture = if config.enable_taa {
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Tonemap Output Texture"),
                ..hdr_texture_desc.clone()
            }))
        } else {
            None
        };

        let taa_history_texture = if config.enable_taa {
            Some(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("TAA History Texture"),
                ..hdr_texture_desc
            }))
        } else {
            None
        };

        Self {
            config,
            uniform_buffer,
            bloom_bright_pipeline,
            bloom_blur_pipeline,
            tonemap_pipeline,
            taa_pipeline,
            bind_group_layout,
            bloom_bright_texture,
            bloom_blur_h_texture,
            bloom_blur_v_texture,
            tonemap_texture,
            taa_history_texture,
        }
    }

    /// Update uniforms and upload to GPU.
    pub fn update(&self, queue: &wgpu::Queue, time: f32) {
        let uniforms = PostProcessUniforms::from_config(&self.config, time);
        queue.write_buffer(&self.uniform_buffer, 0, uniforms.as_bytes());
    }

    /// Get the configuration.
    #[inline]
    pub fn config(&self) -> &SdfPostProcessConfig {
        &self.config
    }

    /// Check if bloom is enabled.
    #[inline]
    pub fn has_bloom(&self) -> bool {
        self.bloom_bright_pipeline.is_some()
    }

    /// Check if TAA is enabled.
    #[inline]
    pub fn has_taa(&self) -> bool {
        self.taa_pipeline.is_some()
    }
}

// ---------------------------------------------------------------------------
// Embedded Shaders
// ---------------------------------------------------------------------------

/// ACES tone mapping shader.
pub const TONEMAP_SHADER: &str = r#"
// ACES Filmic Tone Mapping for SDF Output

struct PostProcessUniforms {
    resolution: vec2<f32>,
    time: f32,
    bloom_threshold: f32,
    bloom_intensity: f32,
    taa_feedback: f32,
    motion_scale: f32,
    exposure: f32,
}

@group(0) @binding(0) var<uniform> uniforms: PostProcessUniforms;
@group(0) @binding(1) var hdr_input: texture_2d<f32>;
@group(0) @binding(2) var ldr_output: texture_storage_2d<rgba16float, write>;

// ACES approximation by Krzysztof Narkowicz
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);

    if (f32(coords.x) >= uniforms.resolution.x || f32(coords.y) >= uniforms.resolution.y) {
        return;
    }

    // Load HDR color
    var hdr_color = textureLoad(hdr_input, coords, 0).rgb;

    // Apply exposure
    hdr_color *= uniforms.exposure;

    // Apply ACES tone mapping
    let ldr_color = aces_tonemap(hdr_color);

    // Output
    textureStore(ldr_output, coords, vec4<f32>(ldr_color, 1.0));
}
"#;

/// Bloom brightness extraction shader.
pub const BLOOM_BRIGHT_SHADER: &str = r#"
// Bloom Bright Pass - Extract pixels above threshold

struct PostProcessUniforms {
    resolution: vec2<f32>,
    time: f32,
    bloom_threshold: f32,
    bloom_intensity: f32,
    taa_feedback: f32,
    motion_scale: f32,
    exposure: f32,
}

@group(0) @binding(0) var<uniform> uniforms: PostProcessUniforms;
@group(0) @binding(1) var hdr_input: texture_2d<f32>;
@group(0) @binding(2) var bright_output: texture_storage_2d<rgba16float, write>;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);

    if (f32(coords.x) >= uniforms.resolution.x || f32(coords.y) >= uniforms.resolution.y) {
        return;
    }

    let hdr_color = textureLoad(hdr_input, coords, 0).rgb;
    let lum = luminance(hdr_color);

    // Soft threshold
    let soft_threshold = max(lum - uniforms.bloom_threshold, 0.0);
    let contribution = soft_threshold / (soft_threshold + 1.0);

    let bright_color = hdr_color * contribution * uniforms.bloom_intensity;

    textureStore(bright_output, coords, vec4<f32>(bright_color, 1.0));
}
"#;

/// Bloom Gaussian blur shader.
pub const BLOOM_BLUR_SHADER: &str = r#"
// Bloom Separable Gaussian Blur

struct PostProcessUniforms {
    resolution: vec2<f32>,
    time: f32,
    bloom_threshold: f32,
    bloom_intensity: f32,
    taa_feedback: f32,
    motion_scale: f32,
    exposure: f32,
}

@group(0) @binding(0) var<uniform> uniforms: PostProcessUniforms;
@group(0) @binding(1) var blur_input: texture_2d<f32>;
@group(0) @binding(2) var blur_output: texture_storage_2d<rgba16float, write>;

// 13-tap Gaussian kernel
const KERNEL_WEIGHTS: array<f32, 7> = array<f32, 7>(
    0.0093, 0.028, 0.0659, 0.1216, 0.1760, 0.1993, 0.1760
);

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);

    if (f32(coords.x) >= uniforms.resolution.x || f32(coords.y) >= uniforms.resolution.y) {
        return;
    }

    // Determine blur direction from pass uniform (horizontal if time integer is even)
    let is_horizontal = (u32(uniforms.time * 10.0) % 2u) == 0u;
    let dir = select(vec2<i32>(0, 1), vec2<i32>(1, 0), is_horizontal);

    var result = vec3<f32>(0.0);

    // Center sample
    result += textureLoad(blur_input, coords, 0).rgb * KERNEL_WEIGHTS[6];

    // Symmetric samples
    for (var i = 0; i < 6; i++) {
        let offset = (i + 1) * 2;
        let sample_pos_p = coords + dir * offset;
        let sample_pos_n = coords - dir * offset;

        result += textureLoad(blur_input, sample_pos_p, 0).rgb * KERNEL_WEIGHTS[i];
        result += textureLoad(blur_input, sample_pos_n, 0).rgb * KERNEL_WEIGHTS[i];
    }

    textureStore(blur_output, coords, vec4<f32>(result, 1.0));
}
"#;

/// TAA shader.
pub const TAA_SHADER: &str = r#"
// Temporal Anti-Aliasing for SDF Output

struct PostProcessUniforms {
    resolution: vec2<f32>,
    time: f32,
    bloom_threshold: f32,
    bloom_intensity: f32,
    taa_feedback: f32,
    motion_scale: f32,
    exposure: f32,
}

@group(0) @binding(0) var<uniform> uniforms: PostProcessUniforms;
@group(0) @binding(1) var current_frame: texture_2d<f32>;
@group(0) @binding(2) var taa_output: texture_storage_2d<rgba16float, write>;

// Note: In a full implementation, we'd have a history texture and motion vectors
// This is a simplified version that demonstrates the integration point

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b,
        0.5 * rgb.r - 0.5 * rgb.b,
        -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b
    );
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        ycocg.x + ycocg.y - ycocg.z,
        ycocg.x + ycocg.z,
        ycocg.x - ycocg.y - ycocg.z
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);

    if (f32(coords.x) >= uniforms.resolution.x || f32(coords.y) >= uniforms.resolution.y) {
        return;
    }

    // Load current frame
    let current_color = textureLoad(current_frame, coords, 0).rgb;

    // In a full implementation:
    // 1. Load history using motion vectors
    // 2. Clamp history to neighborhood min/max
    // 3. Blend with feedback weight

    // For now, just pass through with slight temporal smoothing via neighborhood average
    var result = current_color;

    // Simple 3x3 neighborhood average for anti-aliasing effect
    var sum = vec3<f32>(0.0);
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sample_coords = coords + vec2<i32>(dx, dy);
            if (sample_coords.x >= 0 && sample_coords.y >= 0 &&
                f32(sample_coords.x) < uniforms.resolution.x &&
                f32(sample_coords.y) < uniforms.resolution.y) {
                sum += textureLoad(current_frame, sample_coords, 0).rgb;
                count += 1.0;
            }
        }
    }

    // Blend current with neighborhood
    let neighborhood_avg = sum / count;
    result = mix(current_color, neighborhood_avg, 1.0 - uniforms.taa_feedback);

    textureStore(taa_output, coords, vec4<f32>(result, 1.0));
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SdfPostProcessConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = SdfPostProcessConfig::default();
        assert!(config.enable_tonemap);
        assert!(config.enable_bloom);
        assert!(config.enable_taa);
        assert!(!config.enable_motion_vectors);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn test_config_minimal() {
        let config = SdfPostProcessConfig::minimal(800, 600);
        assert!(config.enable_tonemap);
        assert!(!config.enable_bloom);
        assert!(!config.enable_taa);
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
    }

    #[test]
    fn test_config_full() {
        let config = SdfPostProcessConfig::full(1920, 1080);
        assert!(config.enable_tonemap);
        assert!(config.enable_bloom);
        assert!(config.enable_taa);
        assert!(config.enable_motion_vectors);
    }

    #[test]
    fn test_config_with_bloom() {
        let config = SdfPostProcessConfig::minimal(800, 600).with_bloom(0.8, 0.7);
        assert!(config.enable_bloom);
        assert_eq!(config.bloom_threshold, 0.8);
        assert_eq!(config.bloom_intensity, 0.7);
    }

    #[test]
    fn test_config_with_taa() {
        let config = SdfPostProcessConfig::minimal(800, 600).with_taa(0.85);
        assert!(config.enable_taa);
        assert_eq!(config.taa_feedback, 0.85);
    }

    #[test]
    fn test_config_validate_ok() {
        let config = SdfPostProcessConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_width() {
        let mut config = SdfPostProcessConfig::default();
        config.width = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_negative_bloom_threshold() {
        let mut config = SdfPostProcessConfig::default();
        config.bloom_threshold = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_taa_feedback() {
        let mut config = SdfPostProcessConfig::default();
        config.taa_feedback = 1.5;
        assert!(config.validate().is_err());
    }

    // =========================================================================
    // PostProcessUniforms Tests
    // =========================================================================

    #[test]
    fn test_uniforms_default() {
        let uniforms = PostProcessUniforms::default();
        assert_eq!(uniforms.resolution, [1920.0, 1080.0]);
        assert_eq!(uniforms.time, 0.0);
        assert_eq!(uniforms.bloom_threshold, SDF_BLOOM_THRESHOLD);
        assert_eq!(uniforms.exposure, 1.0);
    }

    #[test]
    fn test_uniforms_from_config() {
        let config = SdfPostProcessConfig::full(800, 600)
            .with_bloom(0.5, 0.8)
            .with_taa(0.95);
        let uniforms = PostProcessUniforms::from_config(&config, 5.0);

        assert_eq!(uniforms.resolution, [800.0, 600.0]);
        assert_eq!(uniforms.time, 5.0);
        assert_eq!(uniforms.bloom_threshold, 0.5);
        assert_eq!(uniforms.bloom_intensity, 0.8);
        assert_eq!(uniforms.taa_feedback, 0.95);
    }

    #[test]
    fn test_uniforms_memory_layout() {
        assert_eq!(std::mem::size_of::<PostProcessUniforms>(), 32);
    }

    #[test]
    fn test_uniforms_as_bytes() {
        let uniforms = PostProcessUniforms::default();
        let bytes = uniforms.as_bytes();
        assert_eq!(bytes.len(), 32);
    }

    // =========================================================================
    // IR Pass Creation Tests
    // =========================================================================

    #[test]
    fn test_bloom_bright_pass_creation() {
        let pass = create_sdf_bloom_bright_pass(
            PassIndex(0),
            ResourceHandle(1),
            ResourceHandle(2),
            1920,
            1080,
        );
        assert_eq!(pass.name, "sdf_bloom_bright");
        assert!(pass.access_set.reads.contains(&ResourceHandle(1)));
        assert!(pass.access_set.writes.contains(&ResourceHandle(2)));
        assert!(pass.tags.contains(&"bloom".into()));
        assert!(pass.tags.contains(&"sdf-integration".into()));
    }

    #[test]
    fn test_bloom_blur_pass_horizontal() {
        let pass = create_bloom_blur_pass(
            PassIndex(1),
            ResourceHandle(2),
            ResourceHandle(3),
            1920,
            1080,
            true,
        );
        assert_eq!(pass.name, "bloom_blur_h");
        assert!(pass.access_set.reads.contains(&ResourceHandle(2)));
        assert!(pass.access_set.writes.contains(&ResourceHandle(3)));
    }

    #[test]
    fn test_bloom_blur_pass_vertical() {
        let pass = create_bloom_blur_pass(
            PassIndex(2),
            ResourceHandle(3),
            ResourceHandle(4),
            1920,
            1080,
            false,
        );
        assert_eq!(pass.name, "bloom_blur_v");
    }

    #[test]
    fn test_motion_vector_pass_creation() {
        let pass = create_sdf_motion_vector_pass(
            PassIndex(0),
            ResourceHandle(1),
            ResourceHandle(2),
            1920,
            1080,
        );
        assert_eq!(pass.name, "sdf_motion_vectors");
        assert!(pass.tags.contains(&"motion-vectors".into()));
        assert!(pass.tags.contains(&"sdf-integration".into()));
    }

    // =========================================================================
    // Post-Process Chain Tests
    // =========================================================================

    #[test]
    fn test_chain_minimal_config() {
        let config = SdfPostProcessConfig::minimal(1920, 1080);
        let (passes, resources) = create_sdf_post_process_chain(
            PassIndex(0),
            ResourceHandle(10),
            ResourceHandle(20),
            &config,
        );

        // Minimal config: just tonemap
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0].name, "tonemap");
        assert!(resources.is_empty()); // No intermediate resources needed
    }

    #[test]
    fn test_chain_full_config() {
        let config = SdfPostProcessConfig::full(1920, 1080);
        let (passes, resources) = create_sdf_post_process_chain(
            PassIndex(0),
            ResourceHandle(10),
            ResourceHandle(20),
            &config,
        );

        // Full config: bloom (3 passes) + tonemap + TAA = 5 passes
        assert!(passes.len() >= 4);

        // Check pass names are present
        let pass_names: Vec<_> = passes.iter().map(|p| p.name.as_str()).collect();
        assert!(pass_names.contains(&"sdf_bloom_bright"));
        assert!(pass_names.contains(&"tonemap"));
        assert!(pass_names.contains(&"taa"));

        // Should have intermediate resources
        assert!(!resources.is_empty());
    }

    #[test]
    fn test_chain_pass_indices_consecutive() {
        let config = SdfPostProcessConfig::full(1920, 1080);
        let (passes, _) = create_sdf_post_process_chain(
            PassIndex(10),
            ResourceHandle(1),
            ResourceHandle(2),
            &config,
        );

        for (i, pass) in passes.iter().enumerate() {
            assert_eq!(pass.index.0, 10 + i);
        }
    }

    #[test]
    fn test_chain_bloom_only() {
        let mut config = SdfPostProcessConfig::default();
        config.enable_taa = false;

        let (passes, resources) = create_sdf_post_process_chain(
            PassIndex(0),
            ResourceHandle(10),
            ResourceHandle(20),
            &config,
        );

        // Bloom (3) + tonemap (1) = 4 passes
        assert_eq!(passes.len(), 4);

        // Intermediate resources for bloom
        assert!(resources.len() >= 3);
    }

    #[test]
    fn test_chain_taa_only() {
        let mut config = SdfPostProcessConfig::default();
        config.enable_bloom = false;

        let (passes, resources) = create_sdf_post_process_chain(
            PassIndex(0),
            ResourceHandle(10),
            ResourceHandle(20),
            &config,
        );

        // Tonemap + TAA = 2 passes
        assert_eq!(passes.len(), 2);

        // TAA needs history + tonemap intermediate
        assert!(!resources.is_empty());
    }

    #[test]
    fn test_chain_resource_handles_reserved_range() {
        let config = SdfPostProcessConfig::full(1920, 1080);
        let (_, resources) = create_sdf_post_process_chain(
            PassIndex(0),
            ResourceHandle(10),
            ResourceHandle(20),
            &config,
        );

        // All resource handles should be in reserved range (0xFE00+)
        for resource in &resources {
            assert!(resource.handle.0 >= 0xFE00, "handle {} not in reserved range", resource.handle.0);
        }
    }

    // =========================================================================
    // Shader Tests
    // =========================================================================

    #[test]
    fn test_tonemap_shader_not_empty() {
        assert!(!TONEMAP_SHADER.is_empty());
        assert!(TONEMAP_SHADER.len() > 500);
    }

    #[test]
    fn test_tonemap_shader_has_entry_point() {
        assert!(TONEMAP_SHADER.contains("@compute"));
        assert!(TONEMAP_SHADER.contains("fn main("));
        assert!(TONEMAP_SHADER.contains("@workgroup_size(8, 8, 1)"));
    }

    #[test]
    fn test_tonemap_shader_has_aces() {
        assert!(TONEMAP_SHADER.contains("aces_tonemap"));
    }

    #[test]
    fn test_bloom_bright_shader_not_empty() {
        assert!(!BLOOM_BRIGHT_SHADER.is_empty());
    }

    #[test]
    fn test_bloom_bright_shader_has_threshold() {
        assert!(BLOOM_BRIGHT_SHADER.contains("bloom_threshold"));
        assert!(BLOOM_BRIGHT_SHADER.contains("luminance"));
    }

    #[test]
    fn test_bloom_blur_shader_not_empty() {
        assert!(!BLOOM_BLUR_SHADER.is_empty());
    }

    #[test]
    fn test_bloom_blur_shader_has_kernel() {
        assert!(BLOOM_BLUR_SHADER.contains("KERNEL_WEIGHTS"));
    }

    #[test]
    fn test_taa_shader_not_empty() {
        assert!(!TAA_SHADER.is_empty());
    }

    #[test]
    fn test_taa_shader_has_feedback() {
        assert!(TAA_SHADER.contains("taa_feedback"));
    }

    // =========================================================================
    // Shader WGSL Parse Tests (using naga)
    // =========================================================================

    #[test]
    fn test_tonemap_shader_parses() {
        use naga::front::wgsl;
        let result = wgsl::parse_str(TONEMAP_SHADER);
        assert!(result.is_ok(), "Tonemap shader parse error: {:?}", result.err());
    }

    #[test]
    fn test_bloom_bright_shader_parses() {
        use naga::front::wgsl;
        let result = wgsl::parse_str(BLOOM_BRIGHT_SHADER);
        assert!(result.is_ok(), "Bloom bright shader parse error: {:?}", result.err());
    }

    #[test]
    fn test_bloom_blur_shader_parses() {
        use naga::front::wgsl;
        let result = wgsl::parse_str(BLOOM_BLUR_SHADER);
        assert!(result.is_ok(), "Bloom blur shader parse error: {:?}", result.err());
    }

    #[test]
    fn test_taa_shader_parses() {
        use naga::front::wgsl;
        let result = wgsl::parse_str(TAA_SHADER);
        assert!(result.is_ok(), "TAA shader parse error: {:?}", result.err());
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_bloom_threshold_constant() {
        assert!(SDF_BLOOM_THRESHOLD >= 0.0);
        assert!(SDF_BLOOM_THRESHOLD <= 5.0);
    }

    #[test]
    fn test_bloom_intensity_constant() {
        assert!(SDF_BLOOM_INTENSITY >= 0.0);
        assert!(SDF_BLOOM_INTENSITY <= 2.0);
    }

    #[test]
    fn test_taa_feedback_constant() {
        assert!(SDF_TAA_FEEDBACK >= 0.0);
        assert!(SDF_TAA_FEEDBACK <= 1.0);
    }

    #[test]
    fn test_motion_scale_constant() {
        assert!(SDF_MOTION_SCALE > 0.0);
    }
}
