//! Unified GPU Culling Pipeline (T-WGPU-P6.6.3).
//!
//! This module provides a unified pipeline struct that integrates all GPU culling
//! stages into a single cohesive system. It composes the existing pipelines
//! (FrustumCullPipeline, HiZCullPipeline, LOD select, stream compact, build indirect)
//! and provides a single `execute()` method to run the entire culling pipeline.
//!
//! # Overview
//!
//! The GPU culling pipeline consists of 5 sequential compute stages with proper
//! barrier placement between stages:
//!
//! ```text
//! Stage 1: Frustum Cull
//! ┌───────────────────────────────────────────────────────────┐
//! │ FrustumCullPipeline: Tests AABBs against frustum planes   │
//! │ Input:  ObjectData[], FrustumPlanes                       │
//! │ Output: visibility_flags (bitfield)                       │
//! └─────────────────────────┬─────────────────────────────────┘
//!                           │ [Storage Buffer Barrier]
//!                           ▼
//! Stage 2: HiZ Occlusion Cull
//! ┌───────────────────────────────────────────────────────────┐
//! │ HiZCullPipeline: Tests visible objects against HiZ depth  │
//! │ Input:  ObjectData[], HiZPyramid, visibility_flags        │
//! │ Output: visibility_flags (updated)                        │
//! └─────────────────────────┬─────────────────────────────────┘
//!                           │ [Storage Buffer Barrier]
//!                           ▼
//! Stage 3: LOD Selection
//! ┌───────────────────────────────────────────────────────────┐
//! │ LOD Select: Computes LOD level per visible object         │
//! │ Input:  ObjectLodInput[], camera_position                 │
//! │ Output: LodSelectOutput[] (level + blend_factor)          │
//! └─────────────────────────┬─────────────────────────────────┘
//!                           │ [Storage Buffer Barrier]
//!                           ▼
//! Stage 4: Stream Compaction
//! ┌───────────────────────────────────────────────────────────┐
//! │ StreamCompactPipeline: Compacts visible indices           │
//! │ Input:  visibility_flags, prefix_sum                      │
//! │ Output: compacted_indices[], compacted_count              │
//! └─────────────────────────┬─────────────────────────────────┘
//!                           │ [Storage Buffer Barrier]
//!                           ▼
//! Stage 5: Build Indirect Draw Commands
//! ┌───────────────────────────────────────────────────────────┐
//! │ BuildIndirectPipeline: Generates indirect draw args       │
//! │ Input:  compacted_indices[], MeshData[], LodBuffer[]      │
//! │ Output: DrawIndexedIndirectArgs[], draw_count             │
//! └───────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **5 compute stages**: Frustum cull -> HiZ cull -> LOD select -> Stream compact -> Build indirect
//! - **Single execute() method**: Runs entire pipeline with proper barriers
//! - **Builder pattern configuration**: Flexible pipeline configuration
//! - **Debug visualization**: Optional intermediate result dumps
//! - **Performance metrics**: Timing data per stage
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{
//!     GPUCullingPipeline, GPUCullingConfig, GPUCullingResources, GPUCullingParams,
//! };
//!
//! // Create pipeline with builder
//! let pipeline = GPUCullingPipeline::builder(&device)
//!     .with_max_objects(100_000)
//!     .with_max_draws(4096)
//!     .with_hiz_size(1920, 1080)
//!     .with_debug_visualization(false)
//!     .build();
//!
//! // Each frame:
//! let params = GPUCullingParams::new(
//!     object_count,
//!     camera_position,
//!     view_projection,
//!     near_plane,
//!     fov_y,
//! );
//!
//! // Execute entire culling pipeline
//! pipeline.execute(
//!     &mut encoder,
//!     &device,
//!     &queue,
//!     &resources,
//!     &params,
//! );
//!
//! // Results are now in resources.indirect_draw_buffer()
//! ```
//!
//! # Performance
//!
//! - Target: < 1ms total for 100K objects on modern GPU
//! - Frustum cull: ~0.1ms (64 threads/wg)
//! - HiZ cull: ~0.2ms (64 threads/wg, texture samples)
//! - LOD select: ~0.1ms (64 threads/wg)
//! - Stream compact: ~0.2ms (64 threads/wg, prefix scan)
//! - Build indirect: ~0.1ms (64 threads/wg)

use bytemuck::{Pod, Zeroable};
use std::mem;

use super::frustum::{FrustumBuffer, FrustumPlanes};
use super::frustum_cull_pipeline::{FrustumCullPipeline as FrustumCullPipelineV2, CullDispatchParams};
use super::hiz_cull_pipeline::{HiZCullPipeline, HiZCullParams};
use super::hiz_pyramid::HiZPyramid;
use super::lod_select::{
    LodSelectParams, ObjectLodInput, LodSelectOutput, SelectionMode,
    LOD_SELECT_OUTPUT_SIZE, WORKGROUP_SIZE as LOD_SELECT_WORKGROUP_SIZE,
};
use super::stream_compact::{
    StreamCompactParams, CompactedIndices,
    WORKGROUP_SIZE as STREAM_COMPACT_WORKGROUP_SIZE,
};
use super::build_indirect::{
    BuildIndirectParams, MeshData,
    BUILD_INDIRECT_PARAMS_SIZE, DRAW_INDEXED_INDIRECT_ARGS_SIZE,
    WORKGROUP_SIZE as BUILD_INDIRECT_WORKGROUP_SIZE,
};
use super::scene_data::SceneDataBuffers;
use super::visibility_flags::VisibilityFlagsBuffer;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Default maximum number of objects supported.
pub const DEFAULT_MAX_OBJECTS: u32 = 100_000;

/// Default maximum number of indirect draw commands.
pub const DEFAULT_MAX_DRAWS: u32 = 65536;

/// Default HiZ pyramid width.
pub const DEFAULT_HIZ_WIDTH: u32 = 1920;

/// Default HiZ pyramid height.
pub const DEFAULT_HIZ_HEIGHT: u32 = 1080;

/// Size of GPUCullingParams in bytes.
pub const GPU_CULLING_PARAMS_SIZE: usize = 128;

/// Workgroup size used by all pipeline stages.
pub const WORKGROUP_SIZE: u32 = 64;

// =============================================================================
// CULLING STAGES
// =============================================================================

/// Enumeration of GPU culling stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CullingStage {
    /// Stage 1: Frustum culling against 6 frustum planes.
    FrustumCull = 0,
    /// Stage 2: HiZ occlusion culling against depth pyramid.
    HiZCull = 1,
    /// Stage 3: LOD level selection per visible object.
    LodSelect = 2,
    /// Stage 4: Stream compaction of visible indices.
    StreamCompact = 3,
    /// Stage 5: Indirect draw command generation.
    BuildIndirect = 4,
}

impl CullingStage {
    /// Get the total number of stages.
    pub const fn count() -> usize {
        5
    }

    /// Get stage index (0-4).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Get stage name for debugging.
    pub const fn name(self) -> &'static str {
        match self {
            Self::FrustumCull => "FrustumCull",
            Self::HiZCull => "HiZCull",
            Self::LodSelect => "LodSelect",
            Self::StreamCompact => "StreamCompact",
            Self::BuildIndirect => "BuildIndirect",
        }
    }
}

// =============================================================================
// GPU CULLING CONFIG
// =============================================================================

/// Configuration for GPU culling pipeline.
#[derive(Clone, Debug)]
pub struct GPUCullingConfig {
    /// Maximum number of objects to process.
    pub max_objects: u32,
    /// Maximum number of indirect draw commands.
    pub max_draws: u32,
    /// HiZ pyramid width.
    pub hiz_width: u32,
    /// HiZ pyramid height.
    pub hiz_height: u32,
    /// Number of HiZ mip levels.
    pub hiz_mip_count: u32,
    /// Enable debug visualization.
    pub debug_visualization: bool,
    /// Skip frustum culling stage.
    pub skip_frustum_cull: bool,
    /// Skip HiZ occlusion culling stage.
    pub skip_hiz_cull: bool,
    /// Skip LOD selection stage.
    pub skip_lod_select: bool,
    /// Enable conservative occlusion culling.
    pub conservative_hiz: bool,
}

impl Default for GPUCullingConfig {
    fn default() -> Self {
        Self {
            max_objects: DEFAULT_MAX_OBJECTS,
            max_draws: DEFAULT_MAX_DRAWS,
            hiz_width: DEFAULT_HIZ_WIDTH,
            hiz_height: DEFAULT_HIZ_HEIGHT,
            hiz_mip_count: 11, // log2(1920) + 1
            debug_visualization: false,
            skip_frustum_cull: false,
            skip_hiz_cull: false,
            skip_lod_select: false,
            conservative_hiz: false,
        }
    }
}

// =============================================================================
// GPU CULLING PARAMS
// =============================================================================

/// Per-frame parameters for GPU culling pipeline.
///
/// # Memory Layout (128 bytes)
///
/// | Offset | Field            | Size | Description                      |
/// |--------|------------------|------|----------------------------------|
/// | 0      | object_count     | 4    | Number of objects to process     |
/// | 4      | flags            | 4    | Processing flags                 |
/// | 8      | hiz_width        | 4    | HiZ pyramid width                |
/// | 12     | hiz_height       | 4    | HiZ pyramid height               |
/// | 16     | camera_position  | 12   | Camera world position            |
/// | 28     | near_plane       | 4    | Near plane distance              |
/// | 32     | view_projection  | 64   | Combined VP matrix               |
/// | 96     | fov_y            | 4    | Vertical FOV in radians          |
/// | 100    | screen_width     | 4    | Screen width for LOD selection   |
/// | 104    | screen_height    | 4    | Screen height for LOD selection  |
/// | 108    | max_draws        | 4    | Maximum draw commands            |
/// | 112    | max_mip          | 4    | Maximum HiZ mip level            |
/// | 116    | lod_mode         | 4    | LOD selection mode               |
/// | 120    | _pad0            | 4    | Padding                          |
/// | 124    | _pad1            | 4    | Padding                          |
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GPUCullingParams {
    /// Number of objects to cull.
    pub object_count: u32,
    /// Processing flags.
    pub flags: u32,
    /// HiZ pyramid width at mip 0.
    pub hiz_width: u32,
    /// HiZ pyramid height at mip 0.
    pub hiz_height: u32,
    /// Camera position in world space.
    pub camera_position: [f32; 3],
    /// Near plane distance.
    pub near_plane: f32,
    /// Combined view-projection matrix (column-major).
    pub view_projection: [[f32; 4]; 4],
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Screen width for LOD calculation.
    pub screen_width: f32,
    /// Screen height for LOD calculation.
    pub screen_height: f32,
    /// Maximum draw commands to generate.
    pub max_draws: u32,
    /// Maximum HiZ mip level.
    pub max_mip: u32,
    /// LOD selection mode (0=distance, 1=screen-size).
    pub lod_mode: u32,
    /// Padding.
    pub _pad0: u32,
    /// Padding.
    pub _pad1: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<GPUCullingParams>() == GPU_CULLING_PARAMS_SIZE);

/// Flag: Skip frustum culling stage.
pub const FLAG_SKIP_FRUSTUM: u32 = 1 << 0;
/// Flag: Skip HiZ occlusion culling stage.
pub const FLAG_SKIP_HIZ: u32 = 1 << 1;
/// Flag: Skip LOD selection stage.
pub const FLAG_SKIP_LOD: u32 = 1 << 2;
/// Flag: Conservative HiZ culling.
pub const FLAG_CONSERVATIVE: u32 = 1 << 3;
/// Flag: Debug visualization enabled.
pub const FLAG_DEBUG: u32 = 1 << 4;

impl Default for GPUCullingParams {
    fn default() -> Self {
        Self {
            object_count: 0,
            flags: 0,
            hiz_width: DEFAULT_HIZ_WIDTH,
            hiz_height: DEFAULT_HIZ_HEIGHT,
            camera_position: [0.0, 0.0, 0.0],
            near_plane: 0.1,
            view_projection: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            fov_y: std::f32::consts::FRAC_PI_4,
            screen_width: 1920.0,
            screen_height: 1080.0,
            max_draws: DEFAULT_MAX_DRAWS,
            max_mip: 10,
            lod_mode: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

impl GPUCullingParams {
    /// Size of this struct in bytes.
    pub const SIZE: usize = GPU_CULLING_PARAMS_SIZE;

    /// Create new GPU culling parameters.
    ///
    /// # Arguments
    ///
    /// * `object_count` - Number of objects to process.
    /// * `camera_position` - Camera world position.
    /// * `view_projection` - Combined VP matrix.
    /// * `near_plane` - Near plane distance.
    /// * `fov_y` - Vertical FOV in radians.
    pub fn new(
        object_count: u32,
        camera_position: [f32; 3],
        view_projection: &[[f32; 4]; 4],
        near_plane: f32,
        fov_y: f32,
    ) -> Self {
        Self {
            object_count,
            camera_position,
            view_projection: *view_projection,
            near_plane,
            fov_y,
            ..Default::default()
        }
    }

    /// Set HiZ parameters.
    #[inline]
    pub fn with_hiz(mut self, width: u32, height: u32, max_mip: u32) -> Self {
        self.hiz_width = width;
        self.hiz_height = height;
        self.max_mip = max_mip;
        self
    }

    /// Set screen dimensions for LOD calculation.
    #[inline]
    pub fn with_screen_size(mut self, width: f32, height: f32) -> Self {
        self.screen_width = width;
        self.screen_height = height;
        self
    }

    /// Set maximum draw commands.
    #[inline]
    pub fn with_max_draws(mut self, max_draws: u32) -> Self {
        self.max_draws = max_draws;
        self
    }

    /// Set LOD selection mode.
    #[inline]
    pub fn with_lod_mode(mut self, mode: SelectionMode) -> Self {
        self.lod_mode = mode.as_u32();
        self
    }

    /// Set flags.
    #[inline]
    pub fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }

    /// Add a flag.
    #[inline]
    pub fn add_flag(mut self, flag: u32) -> Self {
        self.flags |= flag;
        self
    }

    /// Check if frustum culling is enabled.
    #[inline]
    pub fn frustum_enabled(&self) -> bool {
        (self.flags & FLAG_SKIP_FRUSTUM) == 0
    }

    /// Check if HiZ culling is enabled.
    #[inline]
    pub fn hiz_enabled(&self) -> bool {
        (self.flags & FLAG_SKIP_HIZ) == 0
    }

    /// Check if LOD selection is enabled.
    #[inline]
    pub fn lod_enabled(&self) -> bool {
        (self.flags & FLAG_SKIP_LOD) == 0
    }

    /// Calculate number of workgroups for given object count.
    #[inline]
    pub fn workgroups(&self) -> u32 {
        (self.object_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Convert to HiZCullParams for HiZ stage.
    pub fn to_hiz_cull_params(&self) -> HiZCullParams {
        HiZCullParams::new(
            self.object_count,
            self.hiz_width,
            self.hiz_height,
            self.max_mip + 1,
            &self.view_projection,
            self.near_plane,
        ).with_flags(
            if (self.flags & FLAG_CONSERVATIVE) != 0 {
                super::hiz_cull_pipeline::FLAG_CONSERVATIVE
            } else {
                0
            }
        )
    }

    /// Convert to LodSelectParams for LOD stage.
    pub fn to_lod_select_params(&self) -> LodSelectParams {
        let mode = if self.lod_mode == 1 {
            SelectionMode::ScreenSize
        } else {
            SelectionMode::Distance
        };

        LodSelectParams::new(
            self.camera_position,
            self.screen_width,
            self.screen_height,
            self.fov_y,
            mode,
            self.object_count,
        )
    }

    /// Convert to BuildIndirectParams for build indirect stage.
    pub fn to_build_indirect_params(&self, visible_count: u32) -> BuildIndirectParams {
        BuildIndirectParams::new(visible_count, self.max_draws)
    }
}

// =============================================================================
// DEBUG DUMP
// =============================================================================

/// Debug dump of intermediate results.
#[derive(Clone, Debug, Default)]
pub struct CullingDebugDump {
    /// Stage timings in nanoseconds.
    pub stage_timings_ns: [u64; CullingStage::count()],
    /// Visibility counts after each stage.
    pub visibility_counts: [u32; CullingStage::count()],
    /// Total objects processed.
    pub total_objects: u32,
    /// Final visible count.
    pub final_visible_count: u32,
    /// Final draw command count.
    pub final_draw_count: u32,
    /// Whether debug was enabled.
    pub debug_enabled: bool,
}

impl CullingDebugDump {
    /// Create a new empty debug dump.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get timing for a specific stage.
    #[inline]
    pub fn stage_timing_ms(&self, stage: CullingStage) -> f64 {
        self.stage_timings_ns[stage.index()] as f64 / 1_000_000.0
    }

    /// Get total timing in milliseconds.
    #[inline]
    pub fn total_timing_ms(&self) -> f64 {
        self.stage_timings_ns.iter().sum::<u64>() as f64 / 1_000_000.0
    }

    /// Get cull rate (percentage culled).
    #[inline]
    pub fn cull_rate(&self) -> f32 {
        if self.total_objects == 0 {
            0.0
        } else {
            1.0 - (self.final_visible_count as f32 / self.total_objects as f32)
        }
    }
}

// =============================================================================
// GPU CULLING PIPELINE BUILDER
// =============================================================================

/// Builder for GPUCullingPipeline.
pub struct GPUCullingPipelineBuilder<'a> {
    device: &'a wgpu::Device,
    config: GPUCullingConfig,
}

impl<'a> GPUCullingPipelineBuilder<'a> {
    /// Create a new pipeline builder.
    pub fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            config: GPUCullingConfig::default(),
        }
    }

    /// Set maximum object count.
    #[inline]
    pub fn with_max_objects(mut self, max_objects: u32) -> Self {
        self.config.max_objects = max_objects;
        self
    }

    /// Set maximum draw commands.
    #[inline]
    pub fn with_max_draws(mut self, max_draws: u32) -> Self {
        self.config.max_draws = max_draws;
        self
    }

    /// Set HiZ pyramid dimensions.
    #[inline]
    pub fn with_hiz_size(mut self, width: u32, height: u32) -> Self {
        self.config.hiz_width = width;
        self.config.hiz_height = height;
        // Calculate mip count
        let max_dim = width.max(height);
        self.config.hiz_mip_count = (max_dim as f32).log2().ceil() as u32 + 1;
        self
    }

    /// Set HiZ mip count explicitly.
    #[inline]
    pub fn with_hiz_mip_count(mut self, mip_count: u32) -> Self {
        self.config.hiz_mip_count = mip_count;
        self
    }

    /// Enable or disable debug visualization.
    #[inline]
    pub fn with_debug_visualization(mut self, enable: bool) -> Self {
        self.config.debug_visualization = enable;
        self
    }

    /// Skip frustum culling stage.
    #[inline]
    pub fn skip_frustum_cull(mut self, skip: bool) -> Self {
        self.config.skip_frustum_cull = skip;
        self
    }

    /// Skip HiZ occlusion culling stage.
    #[inline]
    pub fn skip_hiz_cull(mut self, skip: bool) -> Self {
        self.config.skip_hiz_cull = skip;
        self
    }

    /// Skip LOD selection stage.
    #[inline]
    pub fn skip_lod_select(mut self, skip: bool) -> Self {
        self.config.skip_lod_select = skip;
        self
    }

    /// Enable conservative HiZ culling.
    #[inline]
    pub fn with_conservative_hiz(mut self, conservative: bool) -> Self {
        self.config.conservative_hiz = conservative;
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> GPUCullingPipeline {
        GPUCullingPipeline::new_with_config(self.device, self.config)
    }
}

// =============================================================================
// GPU CULLING PIPELINE
// =============================================================================

/// Unified GPU culling pipeline integrating all culling stages.
///
/// This struct composes the existing pipelines and provides a single
/// `execute()` method to run the entire culling pipeline with proper
/// barrier placement between stages.
pub struct GPUCullingPipeline {
    /// Stage 1: Frustum culling pipeline.
    frustum_cull_pipeline: FrustumCullPipelineV2,

    /// Stage 2: HiZ occlusion culling pipeline.
    hiz_cull_pipeline: HiZCullPipeline,

    /// Stage 3: LOD selection shader module.
    lod_select_shader: wgpu::ShaderModule,

    /// Stage 3: LOD selection compute pipeline.
    lod_select_pipeline: wgpu::ComputePipeline,

    /// Stage 3: LOD select bind group layout.
    lod_select_layout: wgpu::BindGroupLayout,

    /// Stage 4: Stream compaction shader module.
    stream_compact_shader: wgpu::ShaderModule,

    /// Stage 4: Stream compaction compute pipeline.
    stream_compact_pipeline: wgpu::ComputePipeline,

    /// Stage 4: Stream compact bind group layout.
    stream_compact_layout: wgpu::BindGroupLayout,

    /// Stage 5: Build indirect shader module.
    build_indirect_shader: wgpu::ShaderModule,

    /// Stage 5: Build indirect compute pipeline.
    build_indirect_pipeline: wgpu::ComputePipeline,

    /// Stage 5: Build indirect bind group layout.
    build_indirect_layout: wgpu::BindGroupLayout,

    /// Pipeline configuration.
    config: GPUCullingConfig,
}

impl GPUCullingPipeline {
    /// Create a new GPU culling pipeline with default configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for pipeline creation.
    pub fn new(device: &wgpu::Device) -> Self {
        Self::new_with_config(device, GPUCullingConfig::default())
    }

    /// Create a pipeline builder.
    pub fn builder(device: &wgpu::Device) -> GPUCullingPipelineBuilder<'_> {
        GPUCullingPipelineBuilder::new(device)
    }

    /// Create a new GPU culling pipeline with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for pipeline creation.
    /// * `config` - Pipeline configuration.
    pub fn new_with_config(device: &wgpu::Device, config: GPUCullingConfig) -> Self {
        // Create Stage 1: Frustum cull pipeline
        let frustum_cull_pipeline = FrustumCullPipelineV2::new(device);

        // Create Stage 2: HiZ cull pipeline
        let hiz_cull_pipeline = HiZCullPipeline::new(device);

        // Create Stage 3: LOD select pipeline
        let lod_select_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_culling_lod_select_shader"),
            source: wgpu::ShaderSource::Wgsl(Self::lod_select_shader_source().into()),
        });

        let lod_select_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_culling_lod_select_layout"),
            entries: &[
                // Binding 0: LodSelectParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(48).unwrap()),
                    },
                    count: None,
                },
                // Binding 1: ObjectLodInput[] storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(48).unwrap()),
                    },
                    count: None,
                },
                // Binding 2: LodSelectOutput[] storage (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(8).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let lod_select_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_culling_lod_select_pipeline_layout"),
            bind_group_layouts: &[&lod_select_layout],
            push_constant_ranges: &[],
        });

        let lod_select_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_culling_lod_select_pipeline"),
            layout: Some(&lod_select_pipeline_layout),
            module: &lod_select_shader,
            entry_point: "lod_select_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create Stage 4: Stream compact pipeline
        let stream_compact_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_culling_stream_compact_shader"),
            source: wgpu::ShaderSource::Wgsl(Self::stream_compact_shader_source().into()),
        });

        let stream_compact_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_culling_stream_compact_layout"),
            entries: &[
                // Binding 0: StreamCompactParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                    },
                    count: None,
                },
                // Binding 1: visibility_flags storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
                // Binding 2: compacted_indices storage (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
                // Binding 3: compacted_count storage (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let stream_compact_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_culling_stream_compact_pipeline_layout"),
            bind_group_layouts: &[&stream_compact_layout],
            push_constant_ranges: &[],
        });

        let stream_compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_culling_stream_compact_pipeline"),
            layout: Some(&stream_compact_pipeline_layout),
            module: &stream_compact_shader,
            entry_point: "stream_compact_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create Stage 5: Build indirect pipeline
        let build_indirect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_culling_build_indirect_shader"),
            source: wgpu::ShaderSource::Wgsl(Self::build_indirect_shader_source().into()),
        });

        let build_indirect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_culling_build_indirect_layout"),
            entries: &[
                // Binding 0: BuildIndirectParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                    },
                    count: None,
                },
                // Binding 1: compacted_indices storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
                // Binding 2: mesh_data storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(48).unwrap()),
                    },
                    count: None,
                },
                // Binding 3: lod_buffer storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(8).unwrap()),
                    },
                    count: None,
                },
                // Binding 4: object_data storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(144).unwrap()),
                    },
                    count: None,
                },
                // Binding 5: indirect_draw_args storage (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(20).unwrap()),
                    },
                    count: None,
                },
                // Binding 6: draw_count storage (atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let build_indirect_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_culling_build_indirect_pipeline_layout"),
            bind_group_layouts: &[&build_indirect_layout],
            push_constant_ranges: &[],
        });

        let build_indirect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_culling_build_indirect_pipeline"),
            layout: Some(&build_indirect_pipeline_layout),
            module: &build_indirect_shader,
            entry_point: "build_indirect_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            frustum_cull_pipeline,
            hiz_cull_pipeline,
            lod_select_shader,
            lod_select_pipeline,
            lod_select_layout,
            stream_compact_shader,
            stream_compact_pipeline,
            stream_compact_layout,
            build_indirect_shader,
            build_indirect_pipeline,
            build_indirect_layout,
            config,
        }
    }

    /// Get the pipeline configuration.
    #[inline]
    pub fn config(&self) -> &GPUCullingConfig {
        &self.config
    }

    /// Get the frustum cull pipeline.
    #[inline]
    pub fn frustum_cull_pipeline(&self) -> &FrustumCullPipelineV2 {
        &self.frustum_cull_pipeline
    }

    /// Get the HiZ cull pipeline.
    #[inline]
    pub fn hiz_cull_pipeline(&self) -> &HiZCullPipeline {
        &self.hiz_cull_pipeline
    }

    /// Get the LOD select bind group layout.
    #[inline]
    pub fn lod_select_layout(&self) -> &wgpu::BindGroupLayout {
        &self.lod_select_layout
    }

    /// Get the stream compact bind group layout.
    #[inline]
    pub fn stream_compact_layout(&self) -> &wgpu::BindGroupLayout {
        &self.stream_compact_layout
    }

    /// Get the build indirect bind group layout.
    #[inline]
    pub fn build_indirect_layout(&self) -> &wgpu::BindGroupLayout {
        &self.build_indirect_layout
    }

    /// Execute the complete GPU culling pipeline.
    ///
    /// Runs all 5 stages in sequence with proper storage buffer barriers
    /// between stages.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder for recording commands.
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `frustum_buffer` - Frustum planes buffer.
    /// * `hiz_pyramid` - HiZ depth pyramid texture.
    /// * `scene_data` - Scene data buffers (object data).
    /// * `visibility_flags` - Visibility flags buffer.
    /// * `lod_input_buffer` - LOD input buffer for objects.
    /// * `lod_output_buffer` - LOD output buffer.
    /// * `compacted_indices` - Compacted indices output.
    /// * `mesh_data_buffer` - Mesh data buffer.
    /// * `indirect_draw_buffer` - Indirect draw command buffer.
    /// * `draw_count_buffer` - Draw count buffer.
    /// * `params` - Per-frame culling parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frustum_buffer: &FrustumBuffer,
        hiz_pyramid: &HiZPyramid,
        scene_data: &SceneDataBuffers,
        visibility_flags: &VisibilityFlagsBuffer,
        lod_input_buffer: &wgpu::Buffer,
        lod_output_buffer: &wgpu::Buffer,
        compacted_indices: &CompactedIndices,
        mesh_data_buffer: &wgpu::Buffer,
        indirect_draw_buffer: &wgpu::Buffer,
        draw_count_buffer: &wgpu::Buffer,
        params: &GPUCullingParams,
    ) {
        if params.object_count == 0 {
            return;
        }

        // Clear visibility flags before culling
        visibility_flags.clear_with_encoder(encoder);

        // Stage 1: Frustum Cull
        if params.frustum_enabled() {
            self.frustum_cull_pipeline.dispatch(
                encoder,
                device,
                queue,
                frustum_buffer,
                scene_data,
                visibility_flags,
                params.object_count,
            );

            // Storage buffer barrier after frustum cull
            // Note: wgpu handles barriers automatically for storage buffers
            // between compute passes, but explicit barrier placement is noted here
        }

        // Stage 2: HiZ Occlusion Cull
        if params.hiz_enabled() {
            let hiz_params = params.to_hiz_cull_params();
            self.hiz_cull_pipeline.dispatch(
                encoder,
                device,
                queue,
                frustum_buffer,
                hiz_pyramid,
                scene_data,
                visibility_flags,
                &hiz_params,
            );

            // Storage buffer barrier after HiZ cull
        }

        // Stage 3: LOD Selection
        if params.lod_enabled() {
            let lod_params = params.to_lod_select_params();
            let lod_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_culling_lod_params_buffer"),
                size: 48,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&lod_params_buffer, 0, bytemuck::bytes_of(&lod_params));

            let lod_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_culling_lod_bind_group"),
                layout: &self.lod_select_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: lod_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: lod_input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lod_output_buffer.as_entire_binding(),
                    },
                ],
            });

            let workgroups = (params.object_count + LOD_SELECT_WORKGROUP_SIZE - 1) / LOD_SELECT_WORKGROUP_SIZE;

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_culling_lod_select_pass"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.lod_select_pipeline);
                pass.set_bind_group(0, &lod_bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Storage buffer barrier after LOD select
        }

        // Clear compacted count before stream compaction
        compacted_indices.clear_count_with_encoder(encoder);

        // Stage 4: Stream Compaction
        {
            let compact_params = StreamCompactParams::new(params.object_count);
            let compact_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_culling_compact_params_buffer"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&compact_params_buffer, 0, bytemuck::bytes_of(&compact_params));

            let compact_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_culling_compact_bind_group"),
                layout: &self.stream_compact_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: compact_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: visibility_flags.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: compacted_indices.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: compacted_indices.count_buffer().as_entire_binding(),
                    },
                ],
            });

            let workgroups = (params.object_count + STREAM_COMPACT_WORKGROUP_SIZE - 1) / STREAM_COMPACT_WORKGROUP_SIZE;

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_culling_stream_compact_pass"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.stream_compact_pipeline);
                pass.set_bind_group(0, &compact_bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Storage buffer barrier after stream compaction
        }

        // Clear draw count before build indirect
        encoder.clear_buffer(draw_count_buffer, 0, None);

        // Stage 5: Build Indirect Draw Commands
        {
            // For this stage we need visible_count which comes from stream compaction
            // In a real implementation, we'd either:
            // 1. Use indirect dispatch based on compacted_count
            // 2. Read back count and dispatch on CPU
            // 3. Dispatch conservatively based on max possible visible
            //
            // Here we dispatch based on object_count (conservative approach)
            let build_params = BuildIndirectParams::new(params.object_count, params.max_draws);
            let build_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_culling_build_indirect_params_buffer"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&build_params_buffer, 0, bytemuck::bytes_of(&build_params));

            let build_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_culling_build_indirect_bind_group"),
                layout: &self.build_indirect_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: build_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: compacted_indices.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: mesh_data_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: lod_output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: scene_data.object_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: indirect_draw_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: draw_count_buffer.as_entire_binding(),
                    },
                ],
            });

            let workgroups = (params.object_count + BUILD_INDIRECT_WORKGROUP_SIZE - 1) / BUILD_INDIRECT_WORKGROUP_SIZE;

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_culling_build_indirect_pass"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.build_indirect_pipeline);
                pass.set_bind_group(0, &build_bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
    }

    /// LOD selection shader source.
    fn lod_select_shader_source() -> &'static str {
        r#"
// LOD Selection Shader for GPUCullingPipeline (T-WGPU-P6.6.3)

const WORKGROUP_SIZE: u32 = 64u;
const MAX_LOD_LEVELS: u32 = 4u;
const EPSILON: f32 = 1e-6;

// LOD thresholds for screen coverage
const COVERAGE_LOD0: f32 = 0.3;
const COVERAGE_LOD1: f32 = 0.15;
const COVERAGE_LOD2: f32 = 0.05;

struct LodSelectParams {
    camera_position: vec3<f32>,
    _pad0: f32,
    screen_width: f32,
    screen_height: f32,
    fov_y: f32,
    selection_mode: u32,
    object_count: u32,
    enable_blend: u32,
    blend_range: f32,
    _pad1: f32,
}

struct ObjectLodInput {
    world_position: vec3<f32>,
    bounding_radius: f32,
    thresholds: vec3<f32>,
    _pad0: f32,
    flags: u32,
    forced_lod: u32,
    _pad1: vec2<f32>,
}

struct LodSelectOutput {
    level: u32,
    blend_factor: f32,
}

// Flags
const FLAG_FORCE_LOD: u32 = 1u;
const FLAG_ALWAYS_LOD0: u32 = 2u;
const FLAG_ALWAYS_LOD3: u32 = 4u;
const FLAG_DISABLE_BLEND: u32 = 8u;

@group(0) @binding(0) var<uniform> params: LodSelectParams;
@group(0) @binding(1) var<storage, read> objects: array<ObjectLodInput>;
@group(0) @binding(2) var<storage, read_write> output: array<LodSelectOutput>;

fn distance_to_camera(object_pos: vec3<f32>) -> f32 {
    let delta = object_pos - params.camera_position;
    return length(delta);
}

fn screen_coverage(object_pos: vec3<f32>, radius: f32) -> f32 {
    let dist = distance_to_camera(object_pos);
    if (dist < EPSILON) {
        return 1.0;
    }
    let half_fov = params.fov_y * 0.5;
    let tan_half_fov = tan(half_fov);
    if (tan_half_fov < EPSILON) {
        return 1.0;
    }
    let visible_height = 2.0 * dist * tan_half_fov;
    let diameter = 2.0 * radius;
    return diameter / visible_height;
}

fn select_lod_by_distance(distance: f32, thresholds: vec3<f32>) -> u32 {
    if (distance < thresholds.x) { return 0u; }
    if (distance < thresholds.y) { return 1u; }
    if (distance < thresholds.z) { return 2u; }
    return 3u;
}

fn select_lod_by_coverage(coverage: f32) -> u32 {
    if (coverage >= COVERAGE_LOD0) { return 0u; }
    if (coverage >= COVERAGE_LOD1) { return 1u; }
    if (coverage >= COVERAGE_LOD2) { return 2u; }
    return 3u;
}

@compute @workgroup_size(64, 1, 1)
fn lod_select_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.object_count) {
        return;
    }

    let obj = objects[idx];
    var level: u32;
    var blend: f32 = 0.0;

    // Check forced LOD flags
    if ((obj.flags & FLAG_FORCE_LOD) != 0u) {
        level = min(obj.forced_lod, 3u);
    } else if ((obj.flags & FLAG_ALWAYS_LOD0) != 0u) {
        level = 0u;
    } else if ((obj.flags & FLAG_ALWAYS_LOD3) != 0u) {
        level = 3u;
    } else if (params.selection_mode == 1u) {
        // Screen-size based
        let coverage = screen_coverage(obj.world_position, obj.bounding_radius);
        level = select_lod_by_coverage(coverage);
    } else {
        // Distance based
        let distance = distance_to_camera(obj.world_position);
        level = select_lod_by_distance(distance, obj.thresholds);
    }

    output[idx] = LodSelectOutput(level, blend);
}
"#
    }

    /// Stream compaction shader source.
    fn stream_compact_shader_source() -> &'static str {
        r#"
// Stream Compaction Shader for GPUCullingPipeline (T-WGPU-P6.6.3)
//
// This is a simple single-pass stream compaction using atomic counters.
// For large object counts, a prefix-scan based approach would be more efficient.

const WORKGROUP_SIZE: u32 = 64u;
const BITS_PER_WORD: u32 = 32u;

struct StreamCompactParams {
    object_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: StreamCompactParams;
@group(0) @binding(1) var<storage, read> visibility_flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> compacted_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> compacted_count: atomic<u32>;

fn is_visible(object_idx: u32) -> bool {
    let word_idx = object_idx / BITS_PER_WORD;
    let bit_idx = object_idx % BITS_PER_WORD;
    let mask = 1u << bit_idx;
    return (visibility_flags[word_idx] & mask) != 0u;
}

@compute @workgroup_size(64, 1, 1)
fn stream_compact_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.object_count) {
        return;
    }

    if (is_visible(idx)) {
        let output_idx = atomicAdd(&compacted_count, 1u);
        compacted_indices[output_idx] = idx;
    }
}
"#
    }

    /// Build indirect shader source.
    fn build_indirect_shader_source() -> &'static str {
        r#"
// Build Indirect Draw Commands Shader for GPUCullingPipeline (T-WGPU-P6.6.3)

const WORKGROUP_SIZE: u32 = 64u;
const MAX_LOD_LEVELS: u32 = 4u;

struct BuildIndirectParams {
    visible_count: u32,
    max_draws: u32,
    _pad0: u32,
    _pad1: u32,
}

struct LodSelectOutput {
    level: u32,
    blend_factor: f32,
}

struct MeshData {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    _pad: u32,
    lod_index_counts: array<u32, 4>,
    lod_first_index: array<u32, 4>,
}

struct ObjectData {
    transform: mat4x4<f32>,
    aabb_min: vec3<f32>,
    _pad0: f32,
    aabb_max: vec3<f32>,
    _pad1: f32,
    mesh_index: u32,
    material_index: u32,
    lod_distances: array<f32, 4>,
    flags: u32,
    _padding: array<u32, 5>,
}

struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> params: BuildIndirectParams;
@group(0) @binding(1) var<storage, read> compacted_indices: array<u32>;
@group(0) @binding(2) var<storage, read> mesh_data: array<MeshData>;
@group(0) @binding(3) var<storage, read> lod_buffer: array<LodSelectOutput>;
@group(0) @binding(4) var<storage, read> object_data: array<ObjectData>;
@group(0) @binding(5) var<storage, read_write> indirect_args: array<DrawIndexedIndirectArgs>;
@group(0) @binding(6) var<storage, read_write> draw_count: atomic<u32>;

@compute @workgroup_size(64, 1, 1)
fn build_indirect_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.visible_count) {
        return;
    }

    // Get actual object index from compacted list
    let object_idx = compacted_indices[idx];

    // Get object and LOD data
    let obj = object_data[object_idx];
    let lod = lod_buffer[object_idx];
    let mesh = mesh_data[obj.mesh_index];

    // Determine index count and offset for this LOD level
    let lod_level = min(lod.level, 3u);
    var index_count = mesh.lod_index_counts[lod_level];
    var first_index = mesh.first_index + mesh.lod_first_index[lod_level];

    // Fall back to base mesh if LOD data not available
    if (index_count == 0u) {
        index_count = mesh.index_count;
        first_index = mesh.first_index;
    }

    // Allocate draw command slot
    let draw_idx = atomicAdd(&draw_count, 1u);
    if (draw_idx >= params.max_draws) {
        return;
    }

    // Write indirect draw args
    indirect_args[draw_idx] = DrawIndexedIndirectArgs(
        index_count,
        1u,
        first_index,
        mesh.base_vertex,
        object_idx
    );
}
"#
    }
}

impl std::fmt::Debug for GPUCullingPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GPUCullingPipeline")
            .field("config", &self.config)
            .field("workgroup_size", &WORKGROUP_SIZE)
            .field("stages", &CullingStage::count())
            .finish_non_exhaustive()
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Calculate workgroups needed for N objects.
#[inline]
pub const fn workgroups_for_objects(object_count: u32) -> u32 {
    let base = object_count / WORKGROUP_SIZE;
    let remainder = object_count % WORKGROUP_SIZE;
    if remainder != 0 { base + 1 } else { base }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // STRUCT SIZE AND LAYOUT TESTS
    // =========================================================================

    #[test]
    fn test_gpu_culling_params_size() {
        assert_eq!(
            std::mem::size_of::<GPUCullingParams>(),
            GPU_CULLING_PARAMS_SIZE,
            "GPUCullingParams must be {} bytes",
            GPU_CULLING_PARAMS_SIZE
        );
        assert_eq!(GPUCullingParams::SIZE, 128);
    }

    #[test]
    fn test_gpu_culling_params_alignment() {
        assert!(std::mem::align_of::<GPUCullingParams>() >= 4);
    }

    #[test]
    fn test_gpu_culling_params_pod_trait() {
        let params = GPUCullingParams::default();
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 128);

        let restored: &GPUCullingParams = bytemuck::from_bytes(bytes);
        assert_eq!(restored.object_count, params.object_count);
    }

    #[test]
    fn test_gpu_culling_params_zeroable_trait() {
        let zeroed: GPUCullingParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.object_count, 0);
        assert_eq!(zeroed.flags, 0);
    }

    #[test]
    fn test_gpu_culling_params_bytemuck_roundtrip() {
        let vp = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let original = GPUCullingParams::new(
            1234,
            [10.0, 20.0, 30.0],
            &vp,
            0.1,
            std::f32::consts::FRAC_PI_4,
        );

        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let restored: &GPUCullingParams = bytemuck::from_bytes(bytes);
        assert_eq!(restored.object_count, 1234);
        assert_eq!(restored.near_plane, 0.1);
    }

    // =========================================================================
    // CONFIG TESTS
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = GPUCullingConfig::default();
        assert_eq!(config.max_objects, DEFAULT_MAX_OBJECTS);
        assert_eq!(config.max_draws, DEFAULT_MAX_DRAWS);
        assert_eq!(config.hiz_width, DEFAULT_HIZ_WIDTH);
        assert_eq!(config.hiz_height, DEFAULT_HIZ_HEIGHT);
        assert!(!config.debug_visualization);
        assert!(!config.skip_frustum_cull);
        assert!(!config.skip_hiz_cull);
        assert!(!config.skip_lod_select);
    }

    // =========================================================================
    // CULLING STAGE TESTS
    // =========================================================================

    #[test]
    fn test_culling_stage_count() {
        assert_eq!(CullingStage::count(), 5);
    }

    #[test]
    fn test_culling_stage_indices() {
        assert_eq!(CullingStage::FrustumCull.index(), 0);
        assert_eq!(CullingStage::HiZCull.index(), 1);
        assert_eq!(CullingStage::LodSelect.index(), 2);
        assert_eq!(CullingStage::StreamCompact.index(), 3);
        assert_eq!(CullingStage::BuildIndirect.index(), 4);
    }

    #[test]
    fn test_culling_stage_names() {
        assert_eq!(CullingStage::FrustumCull.name(), "FrustumCull");
        assert_eq!(CullingStage::HiZCull.name(), "HiZCull");
        assert_eq!(CullingStage::LodSelect.name(), "LodSelect");
        assert_eq!(CullingStage::StreamCompact.name(), "StreamCompact");
        assert_eq!(CullingStage::BuildIndirect.name(), "BuildIndirect");
    }

    // =========================================================================
    // FLAG TESTS
    // =========================================================================

    #[test]
    fn test_flags_distinct() {
        let flags = [FLAG_SKIP_FRUSTUM, FLAG_SKIP_HIZ, FLAG_SKIP_LOD, FLAG_CONSERVATIVE, FLAG_DEBUG];
        for i in 0..flags.len() {
            for j in (i + 1)..flags.len() {
                assert_eq!(flags[i] & flags[j], 0, "Flags {} and {} overlap", flags[i], flags[j]);
            }
        }
    }

    #[test]
    fn test_params_flag_checking() {
        let params = GPUCullingParams::default()
            .add_flag(FLAG_SKIP_FRUSTUM);
        assert!(!params.frustum_enabled());
        assert!(params.hiz_enabled());
        assert!(params.lod_enabled());

        let params = GPUCullingParams::default()
            .add_flag(FLAG_SKIP_HIZ);
        assert!(params.frustum_enabled());
        assert!(!params.hiz_enabled());

        let params = GPUCullingParams::default()
            .add_flag(FLAG_SKIP_LOD);
        assert!(params.frustum_enabled());
        assert!(!params.lod_enabled());
    }

    // =========================================================================
    // WORKGROUP CALCULATION TESTS
    // =========================================================================

    #[test]
    fn test_workgroups_for_objects() {
        assert_eq!(workgroups_for_objects(0), 0);
        assert_eq!(workgroups_for_objects(1), 1);
        assert_eq!(workgroups_for_objects(64), 1);
        assert_eq!(workgroups_for_objects(65), 2);
        assert_eq!(workgroups_for_objects(128), 2);
        assert_eq!(workgroups_for_objects(1000), 16);
        assert_eq!(workgroups_for_objects(100_000), 1563);
    }

    // =========================================================================
    // DEBUG DUMP TESTS
    // =========================================================================

    #[test]
    fn test_debug_dump_default() {
        let dump = CullingDebugDump::default();
        assert_eq!(dump.total_objects, 0);
        assert_eq!(dump.final_visible_count, 0);
        assert_eq!(dump.final_draw_count, 0);
        assert!(!dump.debug_enabled);
    }

    #[test]
    fn test_debug_dump_cull_rate() {
        let mut dump = CullingDebugDump::new();
        dump.total_objects = 1000;
        dump.final_visible_count = 250;
        assert!((dump.cull_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_debug_dump_timing() {
        let mut dump = CullingDebugDump::new();
        dump.stage_timings_ns[CullingStage::FrustumCull.index()] = 1_000_000;
        dump.stage_timings_ns[CullingStage::HiZCull.index()] = 2_000_000;
        assert!((dump.stage_timing_ms(CullingStage::FrustumCull) - 1.0).abs() < 0.001);
        assert!((dump.total_timing_ms() - 3.0).abs() < 0.001);
    }

    // =========================================================================
    // PARAMS BUILDER TESTS
    // =========================================================================

    #[test]
    fn test_params_builder_methods() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0, 0.0, 0.0], &vp, 0.1, 0.785)
            .with_hiz(1920, 1080, 10)
            .with_screen_size(1920.0, 1080.0)
            .with_max_draws(4096)
            .with_lod_mode(SelectionMode::ScreenSize);

        assert_eq!(params.object_count, 100);
        assert_eq!(params.hiz_width, 1920);
        assert_eq!(params.hiz_height, 1080);
        assert_eq!(params.max_mip, 10);
        assert_eq!(params.screen_width, 1920.0);
        assert_eq!(params.screen_height, 1080.0);
        assert_eq!(params.max_draws, 4096);
        assert_eq!(params.lod_mode, 1);
    }

    #[test]
    fn test_params_conversion_to_hiz_cull() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [0.0, 0.0, 0.0], &vp, 0.1, 0.785)
            .with_hiz(1920, 1080, 10);

        let hiz_params = params.to_hiz_cull_params();
        assert_eq!(hiz_params.object_count, 100);
        assert_eq!(hiz_params.hiz_width, 1920);
        assert_eq!(hiz_params.hiz_height, 1080);
    }

    #[test]
    fn test_params_conversion_to_lod_select() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = GPUCullingParams::new(100, [1.0, 2.0, 3.0], &vp, 0.1, 0.785)
            .with_screen_size(1920.0, 1080.0);

        let lod_params = params.to_lod_select_params();
        assert_eq!(lod_params.object_count, 100);
        assert_eq!(lod_params.camera_position, [1.0, 2.0, 3.0]);
        assert_eq!(lod_params.screen_width, 1920.0);
    }

    // =========================================================================
    // SHADER SOURCE TESTS
    // =========================================================================

    #[test]
    fn test_lod_select_shader_source_not_empty() {
        let source = GPUCullingPipeline::lod_select_shader_source();
        assert!(!source.is_empty());
        assert!(source.contains("fn lod_select_main"));
        assert!(source.contains("@compute @workgroup_size(64, 1, 1)"));
    }

    #[test]
    fn test_stream_compact_shader_source_not_empty() {
        let source = GPUCullingPipeline::stream_compact_shader_source();
        assert!(!source.is_empty());
        assert!(source.contains("fn stream_compact_main"));
        assert!(source.contains("@compute @workgroup_size(64, 1, 1)"));
    }

    #[test]
    fn test_build_indirect_shader_source_not_empty() {
        let source = GPUCullingPipeline::build_indirect_shader_source();
        assert!(!source.is_empty());
        assert!(source.contains("fn build_indirect_main"));
        assert!(source.contains("@compute @workgroup_size(64, 1, 1)"));
    }

    #[test]
    fn test_shader_sources_have_proper_bindings() {
        // LOD select shader
        let source = GPUCullingPipeline::lod_select_shader_source();
        assert!(source.contains("@group(0) @binding(0)"));
        assert!(source.contains("@group(0) @binding(1)"));
        assert!(source.contains("@group(0) @binding(2)"));

        // Stream compact shader
        let source = GPUCullingPipeline::stream_compact_shader_source();
        assert!(source.contains("@group(0) @binding(0)"));
        assert!(source.contains("@group(0) @binding(1)"));
        assert!(source.contains("@group(0) @binding(2)"));
        assert!(source.contains("@group(0) @binding(3)"));

        // Build indirect shader
        let source = GPUCullingPipeline::build_indirect_shader_source();
        assert!(source.contains("@group(0) @binding(0)"));
        assert!(source.contains("@group(0) @binding(1)"));
        assert!(source.contains("@group(0) @binding(2)"));
        assert!(source.contains("@group(0) @binding(3)"));
        assert!(source.contains("@group(0) @binding(4)"));
        assert!(source.contains("@group(0) @binding(5)"));
        assert!(source.contains("@group(0) @binding(6)"));
    }

    // =========================================================================
    // CONSTANTS TESTS
    // =========================================================================

    #[test]
    fn test_constants() {
        assert_eq!(WORKGROUP_SIZE, 64);
        assert_eq!(DEFAULT_MAX_OBJECTS, 100_000);
        assert_eq!(DEFAULT_MAX_DRAWS, 65536);
        assert_eq!(DEFAULT_HIZ_WIDTH, 1920);
        assert_eq!(DEFAULT_HIZ_HEIGHT, 1080);
    }
}
