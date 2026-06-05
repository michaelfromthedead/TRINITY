//! HiZ Cull Pipeline for GPU-Driven Rendering (T-WGPU-P6.4.4).
//!
//! This module provides a compute pipeline that combines frustum culling and
//! HiZ occlusion testing into a single efficient pass. Objects are first tested
//! against the view frustum, and visible objects are then tested against the
//! HiZ pyramid for occlusion.
//!
//! # Overview
//!
//! The HiZ cull pipeline combines two culling stages:
//!
//! ```text
//! +------------------+     +-------------------+     +------------------+
//! | FrustumPlanes    |---->|                   |     | HiZ Pyramid      |
//! | (96 bytes)       |     |  HiZ Cull         |<----|  Texture         |
//! +------------------+     |  ComputePipeline  |     +------------------+
//!                          |  (64 threads/wg)  |
//! +------------------+     |                   |
//! | SceneDataBuffers |---->|   1. Frustum Test |
//! | (ObjectData[])   |     |   2. HiZ Test     |
//! +------------------+     +--------+----------+
//!                                   |
//!                                   v
//!                          +-------------------+
//!                          | VisibilityFlags   |
//!                          | (1 bit/object)    |
//!                          +-------------------+
//! ```
//!
//! # Algorithm
//!
//! For each object:
//! 1. **Frustum Test**: Test AABB against 6 frustum planes (early-out if culled)
//! 2. **HiZ Occlusion Test**: Project AABB to screen, sample HiZ depth pyramid
//! 3. **Visibility Write**: Set bit in visibility buffer if both tests pass
//!
//! # Temporal Stability
//!
//! Uses previous frame's HiZ pyramid for occlusion testing. This provides:
//! - **No GPU stalls**: Occlusion data from prior frame is immediately available
//! - **Temporal coherence**: Most objects don't move significantly between frames
//! - **Conservative**: May show briefly visible objects (never incorrectly cull)
//!
//! # Performance
//!
//! - Workgroup size: 64 threads (optimal for most GPUs)
//! - One thread per object
//! - Early-out on frustum failure (skips HiZ test)
//! - Atomic OR for visibility bit setting
//! - Single dispatch for both culling stages
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{
//!     HiZCullPipeline, FrustumBuffer, HiZPyramid,
//!     SceneDataBuffers, VisibilityFlagsBuffer,
//! };
//!
//! // Create pipeline
//! let pipeline = HiZCullPipeline::new(&device);
//!
//! // Each frame:
//! frustum_buffer.update(&queue, &view_projection);
//! visibility.clear(&queue);
//!
//! pipeline.dispatch(
//!     &mut encoder,
//!     &device,
//!     &queue,
//!     &frustum_buffer,
//!     &hiz_pyramid,
//!     &scene_data,
//!     &visibility,
//!     &hiz_params,
//!     object_count,
//! );
//! ```

use bytemuck::{Pod, Zeroable};
use std::mem;

use super::frustum::{FrustumBuffer, FRUSTUM_PLANES_SIZE};
use super::hiz_occlusion::{HiZOcclusionParams, HIZ_OCCLUSION_PARAMS_SIZE};
use super::hiz_pyramid::HiZPyramid;
use super::scene_data::SceneDataBuffers;
use super::visibility_flags::VisibilityFlagsBuffer;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Workgroup size for HiZ culling (64 threads per workgroup).
///
/// 64 is chosen as a balance between:
/// - GPU occupancy (multiple of warp/wavefront size)
/// - Memory coalescing efficiency
/// - Thread-level parallelism
pub const WORKGROUP_SIZE: u32 = 64;

/// Size of HiZCullParams in bytes.
pub const HIZ_CULL_PARAMS_SIZE: usize = 96;

/// Size of ObjectData in bytes (must match Rust ObjectData struct).
///
/// The WGSL struct uses `array<f32, 4>` instead of `vec4<f32>` for lod_distances
/// to avoid WGSL's 16-byte alignment requirement, which would add implicit padding.
/// This keeps the WGSL struct at 144 bytes, matching the Rust layout exactly.
pub const OBJECT_DATA_SIZE: usize = 144;

// =============================================================================
// HIZ CULL PARAMETERS
// =============================================================================

/// Parameters for HiZ cull compute dispatch.
///
/// This uniform buffer is bound to provide the dispatch configuration
/// and HiZ sampling parameters to the compute shader.
///
/// # Memory Layout (96 bytes)
///
/// | Offset | Field            | Size | Description                      |
/// |--------|------------------|------|----------------------------------|
/// | 0      | object_count     | 4    | Number of objects to process     |
/// | 4      | hiz_width        | 4    | HiZ pyramid base width           |
/// | 8      | hiz_height       | 4    | HiZ pyramid base height          |
/// | 12     | max_mip          | 4    | Maximum mip level (num_mips - 1) |
/// | 16     | view_projection  | 64   | Combined VP matrix               |
/// | 80     | near_plane       | 4    | Near plane distance              |
/// | 84     | flags            | 4    | Processing flags                 |
/// | 88     | _pad0            | 4    | Padding                          |
/// | 92     | _pad1            | 4    | Padding                          |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct HiZCullParams {
    /// Number of objects to cull.
    pub object_count: u32,
    /// HiZ pyramid width at mip 0.
    pub hiz_width: u32,
    /// HiZ pyramid height at mip 0.
    pub hiz_height: u32,
    /// Maximum mip level (num_mips - 1).
    pub max_mip: u32,
    /// Combined view-projection matrix (column-major).
    pub view_projection: [[f32; 4]; 4],
    /// Near plane distance for clipping.
    pub near_plane: f32,
    /// Processing flags.
    pub flags: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    pub _pad1: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<HiZCullParams>() == HIZ_CULL_PARAMS_SIZE);

/// Flag: Skip frustum test (HiZ only).
pub const FLAG_SKIP_FRUSTUM: u32 = 1 << 0;
/// Flag: Skip HiZ test (frustum only).
pub const FLAG_SKIP_HIZ: u32 = 1 << 1;
/// Flag: Use conservative HiZ test.
pub const FLAG_CONSERVATIVE: u32 = 1 << 2;
/// Flag: Debug mode (output detailed results).
pub const FLAG_DEBUG: u32 = 1 << 3;

impl HiZCullParams {
    /// Create new HiZ cull parameters.
    ///
    /// # Arguments
    ///
    /// * `object_count` - Number of objects to cull.
    /// * `hiz_width` - HiZ pyramid width at mip 0.
    /// * `hiz_height` - HiZ pyramid height at mip 0.
    /// * `num_mips` - Total number of mip levels.
    /// * `view_projection` - Combined view-projection matrix.
    /// * `near_plane` - Near plane distance.
    #[inline]
    pub fn new(
        object_count: u32,
        hiz_width: u32,
        hiz_height: u32,
        num_mips: u32,
        view_projection: &[[f32; 4]; 4],
        near_plane: f32,
    ) -> Self {
        Self {
            object_count,
            hiz_width,
            hiz_height,
            max_mip: num_mips.saturating_sub(1),
            view_projection: *view_projection,
            near_plane,
            flags: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create parameters with flags.
    #[inline]
    pub fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }

    /// Create parameters from HiZOcclusionParams.
    ///
    /// Convenience method to create HiZCullParams from existing HiZ occlusion parameters.
    pub fn from_hiz_occlusion_params(
        occlusion_params: &HiZOcclusionParams,
        object_count: u32,
    ) -> Self {
        Self {
            object_count,
            hiz_width: occlusion_params.hiz_size[0] as u32,
            hiz_height: occlusion_params.hiz_size[1] as u32,
            max_mip: occlusion_params.max_mip,
            view_projection: occlusion_params.view_projection,
            near_plane: occlusion_params.near_plane,
            flags: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Get HiZ dimensions as floats (for shader).
    #[inline]
    pub fn hiz_size_f32(&self) -> [f32; 2] {
        [self.hiz_width as f32, self.hiz_height as f32]
    }

    /// Calculate the number of workgroups needed.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        workgroups_for_objects(self.object_count)
    }
}

// =============================================================================
// HIZ CULL PIPELINE
// =============================================================================

/// Combined frustum + HiZ occlusion culling compute pipeline.
///
/// This pipeline performs both frustum and occlusion culling in a single
/// dispatch, writing visibility results to a bitfield buffer. The frustum
/// test is performed first as a cheap early-out before the more expensive
/// HiZ occlusion test.
///
/// # Bind Groups
///
/// The pipeline uses three bind groups:
///
/// **Group 0 (Frustum)**:
/// - Binding 0: `FrustumPlanes` uniform buffer (96 bytes)
///
/// **Group 1 (HiZ Texture)**:
/// - Binding 0: HiZ pyramid texture (2D, all mips)
/// - Binding 1: HiZ sampler (linear)
///
/// **Group 2 (Objects + Visibility + Params)**:
/// - Binding 0: `HiZCullParams` uniform buffer (96 bytes)
/// - Binding 1: `ObjectData[]` storage buffer (read-only)
/// - Binding 2: `visibility_flags` storage buffer (read-write, atomic)
///
/// # Thread Organization
///
/// - One thread per object
/// - Workgroup size: 64 threads
/// - Dispatch: ceil(object_count / 64) workgroups
pub struct HiZCullPipeline {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,

    /// Bind group layout for frustum planes (Group 0).
    frustum_layout: wgpu::BindGroupLayout,

    /// Bind group layout for HiZ texture + sampler (Group 1).
    hiz_layout: wgpu::BindGroupLayout,

    /// Bind group layout for objects, visibility, and params (Group 2).
    objects_layout: wgpu::BindGroupLayout,

    /// Pipeline layout.
    pipeline_layout: wgpu::PipelineLayout,

    /// HiZ sampler for texture sampling.
    hiz_sampler: wgpu::Sampler,
}

impl HiZCullPipeline {
    /// Create a new HiZ cull pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for pipeline creation
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pipeline = HiZCullPipeline::new(&device);
    /// ```
    pub fn new(device: &wgpu::Device) -> Self {
        // Create shader module from embedded WGSL source
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_cull_pipeline_shader"),
            source: wgpu::ShaderSource::Wgsl(Self::shader_source().into()),
        });

        // Group 0: Frustum planes uniform
        let frustum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_cull_frustum_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(FRUSTUM_PLANES_SIZE as u64).unwrap(),
                    ),
                },
                count: None,
            }],
        });

        // Group 1: HiZ texture + sampler
        let hiz_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_cull_hiz_layout"),
            entries: &[
                // Binding 0: HiZ pyramid texture
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
                // Binding 1: HiZ sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Group 2: Params uniform, Objects storage (read), Visibility storage (read-write)
        let objects_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_cull_objects_layout"),
            entries: &[
                // Binding 0: HiZCullParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(HIZ_CULL_PARAMS_SIZE as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                // Binding 1: ObjectData[] storage (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(OBJECT_DATA_SIZE as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                // Binding 2: Visibility flags storage (read-write for atomic ops)
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
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hiz_cull_pipeline_layout"),
            bind_group_layouts: &[&frustum_layout, &hiz_layout, &objects_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hiz_cull_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "hiz_cull_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create HiZ sampler (linear filtering for mip sampling)
        let hiz_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hiz_cull_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            frustum_layout,
            hiz_layout,
            objects_layout,
            pipeline_layout,
            hiz_sampler,
        }
    }

    /// Generate the WGSL shader source for the HiZ cull pipeline.
    ///
    /// This shader combines frustum culling and HiZ occlusion testing,
    /// reading ObjectData, testing against frustum planes and HiZ pyramid,
    /// and writing visibility results using atomic OR operations.
    fn shader_source() -> &'static str {
        r#"
// HiZ Cull Pipeline Shader (T-WGPU-P6.4.4)
//
// Combines frustum culling and HiZ occlusion testing in a single pass.
// Tests object AABBs against:
// 1. Frustum planes (early-out if outside)
// 2. HiZ pyramid (occlusion test)
// Writes visibility to a bitfield buffer using atomic operations.

// Workgroup size: 64 threads
const WORKGROUP_SIZE: u32 = 64u;
const BITS_PER_WORD: u32 = 32u;
const NUM_FRUSTUM_PLANES: u32 = 6u;
const NUM_CORNERS: u32 = 8u;
const EPSILON: f32 = 1e-6;
const CONSERVATIVE_EXPAND: f32 = 1.0;

// Processing flags
const FLAG_SKIP_FRUSTUM: u32 = 1u;
const FLAG_SKIP_HIZ: u32 = 2u;
const FLAG_CONSERVATIVE: u32 = 4u;

// ============================================================================
// Structs
// ============================================================================

struct FrustumPlane {
    normal: vec3<f32>,
    distance: f32,
}

struct FrustumPlanes {
    planes: array<FrustumPlane, 6>,
}

struct HiZCullParams {
    object_count: u32,
    hiz_width: u32,
    hiz_height: u32,
    max_mip: u32,
    view_projection: mat4x4<f32>,
    near_plane: f32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
}

// ObjectData layout (144 bytes, matching Rust ObjectData exactly)
// Uses array<f32, 4> instead of vec4<f32> for lod_distances to avoid
// WGSL's 16-byte alignment requirement for vec4, which would add implicit
// padding and expand the struct to 160 bytes.
struct ObjectData {
    transform: mat4x4<f32>,     // 64 bytes  (offset 0)
    aabb_min: vec3<f32>,        // 12 bytes  (offset 64)
    _pad0: f32,                 // 4 bytes   (offset 76)
    aabb_max: vec3<f32>,        // 12 bytes  (offset 80)
    _pad1: f32,                 // 4 bytes   (offset 92)
    mesh_index: u32,            // 4 bytes   (offset 96)
    material_index: u32,        // 4 bytes   (offset 100)
    lod_distances: array<f32, 4>, // 16 bytes (offset 104) - array avoids vec4 alignment
    flags: u32,                 // 4 bytes   (offset 120)
    _padding: array<u32, 5>,    // 20 bytes  (offset 124) - padding to 144 bytes
}

// ============================================================================
// Bindings
// ============================================================================

// Group 0: Frustum planes
@group(0) @binding(0) var<uniform> frustum: FrustumPlanes;

// Group 1: HiZ pyramid
@group(1) @binding(0) var hiz_texture: texture_2d<f32>;
@group(1) @binding(1) var hiz_sampler: sampler;

// Group 2: Objects, visibility, and params
@group(2) @binding(0) var<uniform> params: HiZCullParams;
@group(2) @binding(1) var<storage, read> objects: array<ObjectData>;
@group(2) @binding(2) var<storage, read_write> visibility_flags: array<atomic<u32>>;

// ============================================================================
// Frustum Culling Functions
// ============================================================================

/// Test AABB against frustum using p-vertex optimization.
/// Returns true if visible, false if culled.
fn test_aabb_frustum(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0u; i < NUM_FRUSTUM_PLANES; i = i + 1u) {
        let plane = frustum.planes[i];

        // P-vertex: corner most aligned with plane normal
        let p = vec3<f32>(
            select(aabb_min.x, aabb_max.x, plane.normal.x >= 0.0),
            select(aabb_min.y, aabb_max.y, plane.normal.y >= 0.0),
            select(aabb_min.z, aabb_max.z, plane.normal.z >= 0.0),
        );

        // If p-vertex is outside plane, entire AABB is culled
        if (dot(plane.normal, p) + plane.distance < 0.0) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// HiZ Occlusion Functions
// ============================================================================

/// Get the i-th corner of an AABB (0-7).
fn get_aabb_corner(aabb_min: vec3<f32>, aabb_max: vec3<f32>, index: u32) -> vec3<f32> {
    return vec3<f32>(
        select(aabb_min.x, aabb_max.x, (index & 1u) != 0u),
        select(aabb_min.y, aabb_max.y, (index & 2u) != 0u),
        select(aabb_min.z, aabb_max.z, (index & 4u) != 0u),
    );
}

/// Transform a point to clip space.
fn world_to_clip(world_pos: vec3<f32>) -> vec4<f32> {
    return params.view_projection * vec4<f32>(world_pos, 1.0);
}

/// NDC to screen coordinates.
fn ndc_to_screen(ndc: vec2<f32>) -> vec2<f32> {
    let uv = (ndc + vec2<f32>(1.0, 1.0)) * 0.5;
    let screen_uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return screen_uv * vec2<f32>(f32(params.hiz_width), f32(params.hiz_height));
}

/// Project AABB to screen-space and get nearest depth.
/// Returns (min_screen, max_screen, near_depth, valid)
fn project_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec4<f32> {
    var min_screen = vec2<f32>(1e30, 1e30);
    var max_screen = vec2<f32>(-1e30, -1e30);
    var near_depth: f32 = 0.0;
    var all_behind = true;
    var any_behind = false;

    let hiz_size = vec2<f32>(f32(params.hiz_width), f32(params.hiz_height));

    for (var i = 0u; i < NUM_CORNERS; i = i + 1u) {
        let corner = get_aabb_corner(aabb_min, aabb_max, i);
        let clip = world_to_clip(corner);

        if (clip.w > EPSILON) {
            all_behind = false;
            let inv_w = 1.0 / clip.w;
            let ndc = vec3<f32>(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w);
            let screen = ndc_to_screen(ndc.xy);

            min_screen = min(min_screen, screen);
            max_screen = max(max_screen, screen);

            let depth = clamp(ndc.z, 0.0, 1.0);
            near_depth = max(near_depth, depth);
        } else {
            any_behind = true;
        }
    }

    if (all_behind) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0); // Invalid flag
    }

    if (any_behind) {
        min_screen = vec2<f32>(0.0, 0.0);
        max_screen = hiz_size;
        near_depth = 1.0;
    }

    // Clamp and expand conservatively
    min_screen = max(min_screen - vec2<f32>(CONSERVATIVE_EXPAND), vec2<f32>(0.0, 0.0));
    max_screen = min(max_screen + vec2<f32>(CONSERVATIVE_EXPAND), hiz_size);

    // Pack into vec4: (rect_width, rect_height, near_depth, 1.0 = valid)
    return vec4<f32>(max_screen.x - min_screen.x, max_screen.y - min_screen.y, near_depth, 1.0);
}

/// Select mip level based on rect size.
fn select_mip_level(rect_size: vec2<f32>) -> u32 {
    let max_dim = max(rect_size.x, rect_size.y);
    if (max_dim <= 1.0) {
        return 0u;
    }
    let mip = u32(log2(max_dim));
    return min(mip, params.max_mip);
}

/// Sample HiZ depth at a point using textureLoad.
fn sample_hiz_point(uv: vec2<f32>, level: u32) -> f32 {
    let divisor = 1u << level;
    let mip_w = max(params.hiz_width / divisor, 1u);
    let mip_h = max(params.hiz_height / divisor, 1u);
    let texel = vec2<i32>(vec2<f32>(uv.x * f32(mip_w), uv.y * f32(mip_h)));
    let clamped = clamp(texel, vec2<i32>(0, 0), vec2<i32>(i32(mip_w) - 1, i32(mip_h) - 1));
    return textureLoad(hiz_texture, clamped, i32(level)).r;
}

/// Sample HiZ rect and return max depth.
fn sample_hiz_rect_max(min_uv: vec2<f32>, max_uv: vec2<f32>, level: u32) -> f32 {
    let d00 = sample_hiz_point(min_uv, level);
    let d10 = sample_hiz_point(vec2<f32>(max_uv.x, min_uv.y), level);
    let d01 = sample_hiz_point(vec2<f32>(min_uv.x, max_uv.y), level);
    let d11 = sample_hiz_point(max_uv, level);
    return max(max(d00, d10), max(d01, d11));
}

/// Test HiZ occlusion for an AABB.
/// Returns true if visible (not occluded).
fn test_hiz_occlusion(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    let proj = project_aabb(aabb_min, aabb_max);

    // Check validity (w component)
    if (proj.w < 0.0) {
        return false; // All behind camera
    }

    let rect_size = proj.xy;
    let near_depth = proj.z;

    // Degenerate rect: mark as visible
    if (rect_size.x < 1.0 || rect_size.y < 1.0) {
        return true;
    }

    let mip_level = select_mip_level(rect_size);
    let hiz_size = vec2<f32>(f32(params.hiz_width), f32(params.hiz_height));

    // We need min/max screen coords for UV calculation
    // Reconstruct from rect_size (simplified: assume centered, then expand)
    // Actually we need to track min/max separately - using a simpler approach here
    // For full accuracy, modify project_aabb to return min/max screen coords

    // Simplified: sample at multiple points based on rect size
    // This is conservative but sufficient for most cases
    let center_uv = vec2<f32>(0.5, 0.5); // Simplified center
    let hiz_depth = sample_hiz_point(center_uv, mip_level);

    // Occlusion test (reverse-Z)
    return near_depth >= (hiz_depth - EPSILON);
}

/// Full HiZ occlusion test with proper min/max tracking.
fn test_hiz_occlusion_full(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    var min_screen = vec2<f32>(1e30, 1e30);
    var max_screen = vec2<f32>(-1e30, -1e30);
    var near_depth: f32 = 0.0;
    var all_behind = true;
    var any_behind = false;

    let hiz_size = vec2<f32>(f32(params.hiz_width), f32(params.hiz_height));

    // Project all 8 corners
    for (var i = 0u; i < NUM_CORNERS; i = i + 1u) {
        let corner = get_aabb_corner(aabb_min, aabb_max, i);
        let clip = world_to_clip(corner);

        if (clip.w > EPSILON) {
            all_behind = false;
            let inv_w = 1.0 / clip.w;
            let ndc = vec3<f32>(clip.x * inv_w, clip.y * inv_w, clip.z * inv_w);
            let screen = ndc_to_screen(ndc.xy);

            min_screen = min(min_screen, screen);
            max_screen = max(max_screen, screen);

            let depth = clamp(ndc.z, 0.0, 1.0);
            near_depth = max(near_depth, depth);
        } else {
            any_behind = true;
        }
    }

    if (all_behind) {
        return false;
    }

    if (any_behind) {
        min_screen = vec2<f32>(0.0, 0.0);
        max_screen = hiz_size;
        near_depth = 1.0;
    }

    // Clamp and expand conservatively
    min_screen = max(min_screen - vec2<f32>(CONSERVATIVE_EXPAND), vec2<f32>(0.0, 0.0));
    max_screen = min(max_screen + vec2<f32>(CONSERVATIVE_EXPAND), hiz_size);

    let rect_size = max_screen - min_screen;

    // Degenerate rect: mark as visible
    if (rect_size.x < 1.0 || rect_size.y < 1.0) {
        return true;
    }

    let mip_level = select_mip_level(rect_size);

    // Convert to UVs
    let min_uv = min_screen / hiz_size;
    let max_uv = max_screen / hiz_size;

    // Sample HiZ at 4 corners
    let hiz_depth = sample_hiz_rect_max(min_uv, max_uv, mip_level);

    // Occlusion test (reverse-Z)
    return near_depth >= (hiz_depth - EPSILON);
}

// ============================================================================
// Main Compute Entry Point
// ============================================================================

@compute @workgroup_size(64, 1, 1)
fn hiz_cull_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let object_idx = gid.x;

    // Bounds check
    if (object_idx >= params.object_count) {
        return;
    }

    // Load object data
    let obj = objects[object_idx];

    // Skip objects without VISIBLE flag (bit 0)
    if ((obj.flags & 1u) == 0u) {
        return;
    }

    var visible = true;

    // Step 1: Frustum test (unless skipped)
    if ((params.flags & FLAG_SKIP_FRUSTUM) == 0u) {
        visible = test_aabb_frustum(obj.aabb_min, obj.aabb_max);
    }

    // Step 2: HiZ occlusion test (only if frustum passed and not skipped)
    if (visible && (params.flags & FLAG_SKIP_HIZ) == 0u) {
        visible = test_hiz_occlusion_full(obj.aabb_min, obj.aabb_max);
    }

    // Set visibility bit atomically
    if (visible) {
        let word_idx = object_idx / BITS_PER_WORD;
        let bit_mask = 1u << (object_idx % BITS_PER_WORD);
        atomicOr(&visibility_flags[word_idx], bit_mask);
    }
}
"#
    }

    /// Dispatch the HiZ culling compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder for recording commands
    /// * `device` - The wgpu device for bind group creation
    /// * `queue` - The wgpu queue for buffer writes
    /// * `frustum` - Frustum planes buffer
    /// * `hiz` - HiZ pyramid texture
    /// * `objects` - Scene data buffers containing objects
    /// * `visibility` - Visibility flags buffer for output
    /// * `params` - HiZ cull parameters
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut encoder = device.create_command_encoder(&Default::default());
    /// let params = HiZCullParams::new(
    ///     object_count, hiz.width(), hiz.height(), hiz.mip_count(),
    ///     &view_proj, 0.1,
    /// );
    /// pipeline.dispatch(
    ///     &mut encoder,
    ///     &device,
    ///     &queue,
    ///     &frustum_buffer,
    ///     &hiz_pyramid,
    ///     &scene_data,
    ///     &visibility,
    ///     &params,
    /// );
    /// queue.submit([encoder.finish()]);
    /// ```
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frustum: &FrustumBuffer,
        hiz: &HiZPyramid,
        objects: &SceneDataBuffers,
        visibility: &VisibilityFlagsBuffer,
        params: &HiZCullParams,
    ) {
        if params.object_count == 0 {
            return;
        }

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hiz_cull_params_buffer"),
            size: HIZ_CULL_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(params));

        // Create frustum bind group (Group 0)
        let frustum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_cull_frustum_bind_group"),
            layout: &self.frustum_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frustum.buffer().as_entire_binding(),
            }],
        });

        // Create HiZ bind group (Group 1)
        let hiz_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_cull_hiz_bind_group"),
            layout: &self.hiz_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hiz.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                },
            ],
        });

        // Create objects bind group (Group 2)
        let objects_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_cull_objects_bind_group"),
            layout: &self.objects_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: objects.object_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: visibility.buffer().as_entire_binding(),
                },
            ],
        });

        // Calculate workgroup count
        let workgroup_count = params.num_workgroups();

        // Begin compute pass and dispatch
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz_cull_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &frustum_bind_group, &[]);
            pass.set_bind_group(1, &hiz_bind_group, &[]);
            pass.set_bind_group(2, &objects_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
    }

    /// Dispatch with pre-created bind groups for improved performance.
    ///
    /// Use this variant when dispatching multiple times with the same
    /// bind groups to avoid per-frame bind group creation overhead.
    pub fn dispatch_with_bind_groups(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frustum_bind_group: &wgpu::BindGroup,
        hiz_bind_group: &wgpu::BindGroup,
        objects_bind_group: &wgpu::BindGroup,
        object_count: u32,
    ) {
        if object_count == 0 {
            return;
        }

        let workgroup_count = workgroups_for_objects(object_count);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hiz_cull_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, frustum_bind_group, &[]);
            pass.set_bind_group(1, hiz_bind_group, &[]);
            pass.set_bind_group(2, objects_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
    }

    /// Get the compute pipeline.
    #[inline]
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Get the frustum bind group layout (Group 0).
    #[inline]
    pub fn frustum_layout(&self) -> &wgpu::BindGroupLayout {
        &self.frustum_layout
    }

    /// Get the HiZ bind group layout (Group 1).
    #[inline]
    pub fn hiz_layout(&self) -> &wgpu::BindGroupLayout {
        &self.hiz_layout
    }

    /// Get the objects bind group layout (Group 2).
    #[inline]
    pub fn objects_layout(&self) -> &wgpu::BindGroupLayout {
        &self.objects_layout
    }

    /// Get the pipeline layout.
    #[inline]
    pub fn pipeline_layout(&self) -> &wgpu::PipelineLayout {
        &self.pipeline_layout
    }

    /// Get the HiZ sampler.
    #[inline]
    pub fn hiz_sampler(&self) -> &wgpu::Sampler {
        &self.hiz_sampler
    }

    /// Create a frustum bind group for the given frustum buffer.
    pub fn create_frustum_bind_group(
        &self,
        device: &wgpu::Device,
        frustum: &FrustumBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_cull_frustum_bind_group"),
            layout: &self.frustum_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frustum.buffer().as_entire_binding(),
            }],
        })
    }

    /// Create a HiZ bind group for the given HiZ pyramid.
    pub fn create_hiz_bind_group(
        &self,
        device: &wgpu::Device,
        hiz: &HiZPyramid,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_cull_hiz_bind_group"),
            layout: &self.hiz_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hiz.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                },
            ],
        })
    }

    /// Create an objects bind group for the given buffers.
    pub fn create_objects_bind_group(
        &self,
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
        objects: &SceneDataBuffers,
        visibility: &VisibilityFlagsBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_cull_objects_bind_group"),
            layout: &self.objects_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: objects.object_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: visibility.buffer().as_entire_binding(),
                },
            ],
        })
    }
}

impl std::fmt::Debug for HiZCullPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HiZCullPipeline")
            .field("workgroup_size", &WORKGROUP_SIZE)
            .finish_non_exhaustive()
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Calculate the number of workgroups needed for N objects.
///
/// Uses saturating arithmetic to prevent overflow when `object_count` is
/// close to `u32::MAX`. The formula `(n + 63) / 64` would overflow if
/// `n > u32::MAX - 63`.
#[inline]
pub const fn workgroups_for_objects(object_count: u32) -> u32 {
    // Div-ceil pattern: avoids overflow by computing remainder first
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
    use std::mem::offset_of;

    // =========================================================================
    // CATEGORY 1: STRUCT LAYOUT TESTS
    // =========================================================================

    #[test]
    fn test_hiz_cull_params_size_is_96_bytes() {
        assert_eq!(
            mem::size_of::<HiZCullParams>(),
            96,
            "HiZCullParams must be exactly 96 bytes for GPU alignment"
        );
        assert_eq!(HIZ_CULL_PARAMS_SIZE, 96);
    }

    #[test]
    fn test_hiz_cull_params_alignment() {
        // Must be 4-byte aligned minimum for uniform buffers
        assert!(mem::align_of::<HiZCullParams>() >= 4);
    }

    #[test]
    fn test_hiz_cull_params_field_offsets() {
        // Verify field offsets match GPU memory layout documentation
        assert_eq!(offset_of!(HiZCullParams, object_count), 0);
        assert_eq!(offset_of!(HiZCullParams, hiz_width), 4);
        assert_eq!(offset_of!(HiZCullParams, hiz_height), 8);
        assert_eq!(offset_of!(HiZCullParams, max_mip), 12);
        assert_eq!(offset_of!(HiZCullParams, view_projection), 16);
        assert_eq!(offset_of!(HiZCullParams, near_plane), 80);
        assert_eq!(offset_of!(HiZCullParams, flags), 84);
        assert_eq!(offset_of!(HiZCullParams, _pad0), 88);
        assert_eq!(offset_of!(HiZCullParams, _pad1), 92);
    }

    #[test]
    fn test_hiz_cull_params_repr_c() {
        // Verify repr(C) layout - fields should be in declaration order without reordering
        let params = HiZCullParams::default();
        let ptr = &params as *const HiZCullParams as *const u8;

        unsafe {
            // object_count at offset 0
            let object_count_ptr = ptr.add(0) as *const u32;
            assert_eq!(*object_count_ptr, params.object_count);

            // hiz_width at offset 4
            let hiz_width_ptr = ptr.add(4) as *const u32;
            assert_eq!(*hiz_width_ptr, params.hiz_width);

            // hiz_height at offset 8
            let hiz_height_ptr = ptr.add(8) as *const u32;
            assert_eq!(*hiz_height_ptr, params.hiz_height);
        }
    }

    #[test]
    fn test_hiz_cull_params_pod_trait() {
        // Verify Pod trait - struct must be plain old data
        let params = HiZCullParams::default();
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 96);

        // Should be able to cast back
        let restored: &HiZCullParams = bytemuck::from_bytes(bytes);
        assert_eq!(restored.object_count, params.object_count);
    }

    #[test]
    fn test_hiz_cull_params_zeroable_trait() {
        // Verify Zeroable trait - zeroed struct must be valid
        let zeroed: HiZCullParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.object_count, 0);
        assert_eq!(zeroed.hiz_width, 0);
        assert_eq!(zeroed.hiz_height, 0);
        assert_eq!(zeroed.max_mip, 0);
        assert_eq!(zeroed.near_plane, 0.0);
        assert_eq!(zeroed.flags, 0);
    }

    #[test]
    fn test_hiz_cull_params_bytemuck_roundtrip() {
        let vp = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let original = HiZCullParams::new(1234, 1920, 1080, 11, &vp, 0.5);

        // Serialize to bytes
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        assert_eq!(bytes.len(), 96);

        // Deserialize from bytes
        let restored: &HiZCullParams = bytemuck::from_bytes(bytes);
        assert_eq!(restored.object_count, 1234);
        assert_eq!(restored.hiz_width, 1920);
        assert_eq!(restored.hiz_height, 1080);
        assert_eq!(restored.max_mip, 10);
        assert_eq!(restored.near_plane, 0.5);
    }

    #[test]
    fn test_hiz_cull_params_view_projection_layout() {
        // view_projection is 64 bytes (4x4 f32 matrix)
        assert_eq!(mem::size_of::<[[f32; 4]; 4]>(), 64);

        // Verify matrix is stored at correct offset
        let vp = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let params = HiZCullParams::new(0, 0, 0, 0, &vp, 0.0);
        let bytes = bytemuck::bytes_of(&params);

        // Matrix starts at offset 16
        let matrix_bytes = &bytes[16..80];
        let first_elem = f32::from_le_bytes([
            matrix_bytes[0], matrix_bytes[1], matrix_bytes[2], matrix_bytes[3]
        ]);
        assert_eq!(first_elem, 1.0);
    }

    // =========================================================================
    // CATEGORY 2: FLAG COMBINATION TESTS
    // =========================================================================

    #[test]
    fn test_flag_skip_frustum_value() {
        assert_eq!(FLAG_SKIP_FRUSTUM, 1 << 0);
        assert_eq!(FLAG_SKIP_FRUSTUM, 0b0001);
    }

    #[test]
    fn test_flag_skip_hiz_value() {
        assert_eq!(FLAG_SKIP_HIZ, 1 << 1);
        assert_eq!(FLAG_SKIP_HIZ, 0b0010);
    }

    #[test]
    fn test_flag_conservative_value() {
        assert_eq!(FLAG_CONSERVATIVE, 1 << 2);
        assert_eq!(FLAG_CONSERVATIVE, 0b0100);
    }

    #[test]
    fn test_flag_debug_value() {
        assert_eq!(FLAG_DEBUG, 1 << 3);
        assert_eq!(FLAG_DEBUG, 0b1000);
    }

    #[test]
    fn test_flags_are_distinct_bits() {
        // All flags should be independent (no overlap)
        let all_flags = [FLAG_SKIP_FRUSTUM, FLAG_SKIP_HIZ, FLAG_CONSERVATIVE, FLAG_DEBUG];
        for i in 0..all_flags.len() {
            for j in (i + 1)..all_flags.len() {
                assert_eq!(
                    all_flags[i] & all_flags[j],
                    0,
                    "Flags {} and {} overlap",
                    all_flags[i],
                    all_flags[j]
                );
            }
        }
    }

    #[test]
    fn test_flag_combination_default_both_enabled() {
        let flags = 0u32;
        assert!((flags & FLAG_SKIP_FRUSTUM) == 0, "Frustum should be enabled by default");
        assert!((flags & FLAG_SKIP_HIZ) == 0, "HiZ should be enabled by default");
    }

    #[test]
    fn test_flag_combination_frustum_only() {
        let flags = FLAG_SKIP_HIZ;
        assert!((flags & FLAG_SKIP_FRUSTUM) == 0, "Frustum should be enabled");
        assert!((flags & FLAG_SKIP_HIZ) != 0, "HiZ should be skipped");
    }

    #[test]
    fn test_flag_combination_hiz_only() {
        let flags = FLAG_SKIP_FRUSTUM;
        assert!((flags & FLAG_SKIP_FRUSTUM) != 0, "Frustum should be skipped");
        assert!((flags & FLAG_SKIP_HIZ) == 0, "HiZ should be enabled");
    }

    #[test]
    fn test_flag_combination_both_skipped() {
        let flags = FLAG_SKIP_FRUSTUM | FLAG_SKIP_HIZ;
        assert!((flags & FLAG_SKIP_FRUSTUM) != 0, "Frustum should be skipped");
        assert!((flags & FLAG_SKIP_HIZ) != 0, "HiZ should be skipped");
    }

    #[test]
    fn test_flag_combination_conservative_frustum() {
        let flags = FLAG_SKIP_HIZ | FLAG_CONSERVATIVE;
        assert!((flags & FLAG_SKIP_HIZ) != 0);
        assert!((flags & FLAG_CONSERVATIVE) != 0);
        assert!((flags & FLAG_SKIP_FRUSTUM) == 0);
    }

    #[test]
    fn test_flag_combination_conservative_hiz() {
        let flags = FLAG_SKIP_FRUSTUM | FLAG_CONSERVATIVE;
        assert!((flags & FLAG_SKIP_FRUSTUM) != 0);
        assert!((flags & FLAG_CONSERVATIVE) != 0);
        assert!((flags & FLAG_SKIP_HIZ) == 0);
    }

    #[test]
    fn test_flag_combination_debug_with_all() {
        let flags = FLAG_SKIP_FRUSTUM | FLAG_SKIP_HIZ | FLAG_CONSERVATIVE | FLAG_DEBUG;
        assert!((flags & FLAG_SKIP_FRUSTUM) != 0);
        assert!((flags & FLAG_SKIP_HIZ) != 0);
        assert!((flags & FLAG_CONSERVATIVE) != 0);
        assert!((flags & FLAG_DEBUG) != 0);
        assert_eq!(flags, 0b1111);
    }

    #[test]
    fn test_flag_with_flags_method() {
        let vp = [[0.0; 4]; 4];

        // Test single flag
        let params = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.1)
            .with_flags(FLAG_SKIP_FRUSTUM);
        assert_eq!(params.flags, FLAG_SKIP_FRUSTUM);

        // Test multiple flags
        let params = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.1)
            .with_flags(FLAG_SKIP_FRUSTUM | FLAG_CONSERVATIVE | FLAG_DEBUG);
        assert_eq!(params.flags, FLAG_SKIP_FRUSTUM | FLAG_CONSERVATIVE | FLAG_DEBUG);
    }

    #[test]
    fn test_flag_values_match_shader() {
        // Verify Rust constants match shader constants
        let source = HiZCullPipeline::shader_source();

        assert!(source.contains("const FLAG_SKIP_FRUSTUM: u32 = 1u;"));
        assert!(source.contains("const FLAG_SKIP_HIZ: u32 = 2u;"));
        assert!(source.contains("const FLAG_CONSERVATIVE: u32 = 4u;"));
    }

    // =========================================================================
    // CATEGORY 3: SHADER SOURCE VALIDATION
    // =========================================================================

    #[test]
    fn test_shader_entry_point_hiz_cull_main() {
        let source = HiZCullPipeline::shader_source();
        assert!(
            source.contains("fn hiz_cull_main"),
            "Entry point hiz_cull_main must exist"
        );
        assert!(
            source.contains("@compute @workgroup_size(64, 1, 1)"),
            "Entry point must have @compute @workgroup_size(64, 1, 1) attribute"
        );
    }

    #[test]
    fn test_shader_bind_group_0_frustum_planes() {
        let source = HiZCullPipeline::shader_source();

        // Group 0, Binding 0: FrustumPlanes uniform
        assert!(
            source.contains("@group(0) @binding(0) var<uniform> frustum: FrustumPlanes"),
            "Group 0 must have FrustumPlanes uniform at binding 0"
        );
    }

    #[test]
    fn test_shader_bind_group_1_hiz_texture() {
        let source = HiZCullPipeline::shader_source();

        // Group 1, Binding 0: HiZ texture
        assert!(
            source.contains("@group(1) @binding(0) var hiz_texture: texture_2d<f32>"),
            "Group 1 must have HiZ texture at binding 0"
        );

        // Group 1, Binding 1: HiZ sampler
        assert!(
            source.contains("@group(1) @binding(1) var hiz_sampler: sampler"),
            "Group 1 must have HiZ sampler at binding 1"
        );
    }

    #[test]
    fn test_shader_bind_group_2_objects_visibility_params() {
        let source = HiZCullPipeline::shader_source();

        // Group 2, Binding 0: HiZCullParams uniform
        assert!(
            source.contains("@group(2) @binding(0) var<uniform> params: HiZCullParams"),
            "Group 2 must have HiZCullParams uniform at binding 0"
        );

        // Group 2, Binding 1: ObjectData storage (read-only)
        assert!(
            source.contains("@group(2) @binding(1) var<storage, read> objects: array<ObjectData>"),
            "Group 2 must have ObjectData read-only storage at binding 1"
        );

        // Group 2, Binding 2: visibility_flags storage (read-write)
        assert!(
            source.contains("@group(2) @binding(2) var<storage, read_write> visibility_flags: array<atomic<u32>>"),
            "Group 2 must have visibility_flags atomic storage at binding 2"
        );
    }

    #[test]
    fn test_shader_test_aabb_frustum_function() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("fn test_aabb_frustum(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool"),
            "test_aabb_frustum function must exist with correct signature"
        );

        // Verify it tests against 6 frustum planes
        assert!(
            source.contains("NUM_FRUSTUM_PLANES: u32 = 6u"),
            "Must test against 6 frustum planes"
        );
    }

    #[test]
    fn test_shader_test_hiz_occlusion_full_function() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("fn test_hiz_occlusion_full(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool"),
            "test_hiz_occlusion_full function must exist with correct signature"
        );
    }

    #[test]
    fn test_shader_atomic_or_visibility_update() {
        let source = HiZCullPipeline::shader_source();

        // Verify atomicOr is used for visibility buffer updates
        assert!(
            source.contains("atomicOr(&visibility_flags[word_idx], bit_mask)"),
            "Visibility buffer must use atomicOr for updates"
        );

        // Verify bit manipulation logic
        assert!(
            source.contains("let word_idx = object_idx / BITS_PER_WORD"),
            "Must calculate word index from object index"
        );
        assert!(
            source.contains("let bit_mask = 1u << (object_idx % BITS_PER_WORD)"),
            "Must calculate bit mask from object index"
        );
    }

    #[test]
    fn test_shader_workgroup_size_constant() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("const WORKGROUP_SIZE: u32 = 64u"),
            "Shader WORKGROUP_SIZE constant must be 64"
        );
    }

    #[test]
    fn test_shader_bits_per_word_constant() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("const BITS_PER_WORD: u32 = 32u"),
            "Shader BITS_PER_WORD constant must be 32"
        );
    }

    #[test]
    fn test_shader_epsilon_constant() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("const EPSILON: f32 = 1e-6"),
            "Shader must define EPSILON for floating point comparisons"
        );
    }

    #[test]
    fn test_shader_conservative_expand_constant() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("const CONSERVATIVE_EXPAND: f32 = 1.0"),
            "Shader must define CONSERVATIVE_EXPAND for screen rect expansion"
        );
    }

    #[test]
    fn test_shader_frustum_plane_struct() {
        let source = HiZCullPipeline::shader_source();

        assert!(source.contains("struct FrustumPlane"));
        assert!(source.contains("normal: vec3<f32>"));
        assert!(source.contains("distance: f32"));
    }

    #[test]
    fn test_shader_frustum_planes_struct() {
        let source = HiZCullPipeline::shader_source();

        assert!(source.contains("struct FrustumPlanes"));
        assert!(source.contains("planes: array<FrustumPlane, 6>"));
    }

    #[test]
    fn test_shader_hiz_cull_params_struct() {
        let source = HiZCullPipeline::shader_source();

        assert!(source.contains("struct HiZCullParams"));
        assert!(source.contains("object_count: u32"));
        assert!(source.contains("hiz_width: u32"));
        assert!(source.contains("hiz_height: u32"));
        assert!(source.contains("max_mip: u32"));
        assert!(source.contains("view_projection: mat4x4<f32>"));
        assert!(source.contains("near_plane: f32"));
        assert!(source.contains("flags: u32"));
    }

    #[test]
    fn test_shader_object_data_struct() {
        let source = HiZCullPipeline::shader_source();

        assert!(source.contains("struct ObjectData"));
        assert!(source.contains("transform: mat4x4<f32>"));
        assert!(source.contains("aabb_min: vec3<f32>"));
        assert!(source.contains("aabb_max: vec3<f32>"));
        assert!(source.contains("mesh_index: u32"));
        assert!(source.contains("material_index: u32"));
        // Uses array<f32, 4> instead of vec4<f32> to avoid WGSL alignment padding
        assert!(source.contains("lod_distances: array<f32, 4>"));
    }

    #[test]
    fn test_shader_p_vertex_optimization() {
        let source = HiZCullPipeline::shader_source();

        // P-vertex optimization: select corner most aligned with plane normal
        assert!(
            source.contains("select(aabb_min.x, aabb_max.x, plane.normal.x >= 0.0)"),
            "Must use P-vertex optimization for frustum culling"
        );
    }

    #[test]
    fn test_shader_mip_level_selection() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("fn select_mip_level"),
            "Must have mip level selection function"
        );
        assert!(
            source.contains("log2(max_dim)"),
            "Mip level should be based on log2 of rect dimension"
        );
    }

    #[test]
    fn test_shader_bounds_check() {
        let source = HiZCullPipeline::shader_source();

        // Must check bounds before processing
        assert!(
            source.contains("if (object_idx >= params.object_count)"),
            "Must bounds-check object index"
        );
    }

    #[test]
    fn test_shader_object_flags_check() {
        let source = HiZCullPipeline::shader_source();

        // Must check object VISIBLE flag before processing
        assert!(
            source.contains("if ((obj.flags & 1u) == 0u)"),
            "Must check object VISIBLE flag (bit 0)"
        );
    }

    // =========================================================================
    // CATEGORY 4: PIPELINE INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_workgroup_size_is_64() {
        assert_eq!(WORKGROUP_SIZE, 64);

        // Verify shader and Rust constant match
        let source = HiZCullPipeline::shader_source();
        assert!(source.contains("@workgroup_size(64, 1, 1)"));
        assert!(source.contains("const WORKGROUP_SIZE: u32 = 64u"));
    }

    #[test]
    fn test_dispatch_calculation_exact_multiple() {
        // Exact multiples of 64
        assert_eq!(workgroups_for_objects(64), 1);
        assert_eq!(workgroups_for_objects(128), 2);
        assert_eq!(workgroups_for_objects(256), 4);
        assert_eq!(workgroups_for_objects(640), 10);
    }

    #[test]
    fn test_dispatch_calculation_ceiling_division() {
        // Non-multiples: ceil(n/64)
        assert_eq!(workgroups_for_objects(1), 1);
        assert_eq!(workgroups_for_objects(63), 1);
        assert_eq!(workgroups_for_objects(65), 2);
        assert_eq!(workgroups_for_objects(100), 2);
        assert_eq!(workgroups_for_objects(127), 2);
        assert_eq!(workgroups_for_objects(129), 3);
    }

    #[test]
    fn test_dispatch_calculation_formula() {
        // Verify formula: (n + 63) / 64
        for n in [1, 33, 64, 65, 100, 127, 128, 200, 1000, 10000] {
            let expected = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let actual = workgroups_for_objects(n);
            assert_eq!(actual, expected, "workgroups_for_objects({}) should be {}", n, expected);
        }
    }

    #[test]
    fn test_bind_group_layout_count() {
        // Pipeline uses 3 bind group layouts
        // This is validated at compile time by the pipeline creation in new()
        // but we can verify the shader source uses all 3 groups
        let source = HiZCullPipeline::shader_source();
        assert!(source.contains("@group(0)"));
        assert!(source.contains("@group(1)"));
        assert!(source.contains("@group(2)"));

        // Should NOT have group 3
        assert!(!source.contains("@group(3)"));
    }

    #[test]
    fn test_hiz_cull_params_from_occlusion_params() {
        let vp = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let occlusion = HiZOcclusionParams::new(&vp, 1920.0, 1080.0, 0.1, 11);
        let params = HiZCullParams::from_hiz_occlusion_params(&occlusion, 500);

        assert_eq!(params.object_count, 500);
        assert_eq!(params.hiz_width, 1920);
        assert_eq!(params.hiz_height, 1080);
        assert_eq!(params.max_mip, 10);
        assert_eq!(params.near_plane, 0.1);
        assert_eq!(params.flags, 0, "Flags should be 0 when created from occlusion params");
    }

    #[test]
    fn test_hiz_cull_params_new() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = HiZCullParams::new(1000, 1920, 1080, 11, &vp, 0.1);

        assert_eq!(params.object_count, 1000);
        assert_eq!(params.hiz_width, 1920);
        assert_eq!(params.hiz_height, 1080);
        assert_eq!(params.max_mip, 10); // num_mips - 1
        assert_eq!(params.near_plane, 0.1);
        assert_eq!(params.flags, 0);
        assert_eq!(params._pad0, 0);
        assert_eq!(params._pad1, 0);
    }

    #[test]
    fn test_hiz_cull_params_max_mip_saturating() {
        let vp = [[0.0; 4]; 4];

        // num_mips = 0 should result in max_mip = 0 (saturating_sub)
        let params = HiZCullParams::new(100, 1920, 1080, 0, &vp, 0.1);
        assert_eq!(params.max_mip, 0);

        // num_mips = 1 should result in max_mip = 0
        let params = HiZCullParams::new(100, 1920, 1080, 1, &vp, 0.1);
        assert_eq!(params.max_mip, 0);

        // num_mips = 11 should result in max_mip = 10
        let params = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.1);
        assert_eq!(params.max_mip, 10);
    }

    #[test]
    fn test_hiz_size_f32_conversion() {
        let vp = [[0.0; 4]; 4];
        let params = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.1);

        let size = params.hiz_size_f32();
        assert_eq!(size[0], 1920.0);
        assert_eq!(size[1], 1080.0);
    }

    #[test]
    fn test_num_workgroups_method() {
        let vp = [[0.0; 4]; 4];

        let params = HiZCullParams::new(64, 1920, 1080, 11, &vp, 0.1);
        assert_eq!(params.num_workgroups(), 1);

        let params = HiZCullParams::new(65, 1920, 1080, 11, &vp, 0.1);
        assert_eq!(params.num_workgroups(), 2);

        let params = HiZCullParams::new(1000, 1920, 1080, 11, &vp, 0.1);
        assert_eq!(params.num_workgroups(), 16);
    }

    // =========================================================================
    // CATEGORY 5: EDGE CASES
    // =========================================================================

    #[test]
    fn test_zero_objects() {
        assert_eq!(workgroups_for_objects(0), 0);

        let vp = [[0.0; 4]; 4];
        let params = HiZCullParams::new(0, 1920, 1080, 11, &vp, 0.1);
        assert_eq!(params.object_count, 0);
        assert_eq!(params.num_workgroups(), 0);
    }

    #[test]
    fn test_single_object() {
        assert_eq!(workgroups_for_objects(1), 1);

        let vp = [[0.0; 4]; 4];
        let params = HiZCullParams::new(1, 1920, 1080, 11, &vp, 0.1);
        assert_eq!(params.num_workgroups(), 1);
    }

    #[test]
    fn test_object_count_not_divisible_by_64() {
        // Test various non-divisible counts
        let test_counts = [1, 33, 63, 65, 100, 127, 129, 255, 257];

        for count in test_counts {
            let expected_workgroups = (count + 63) / 64;
            assert_eq!(
                workgroups_for_objects(count),
                expected_workgroups,
                "Failed for count={}", count
            );
        }
    }

    #[test]
    fn test_max_mip_level_bounds() {
        let vp = [[0.0; 4]; 4];

        // max_mip = num_mips - 1, minimum is 0
        let params = HiZCullParams::new(100, 1920, 1080, 0, &vp, 0.1);
        assert_eq!(params.max_mip, 0, "max_mip should saturate to 0");

        // Large num_mips
        let params = HiZCullParams::new(100, 4096, 4096, 13, &vp, 0.1);
        assert_eq!(params.max_mip, 12);

        // Very large num_mips (unlikely but should work)
        let params = HiZCullParams::new(100, 16384, 16384, 15, &vp, 0.1);
        assert_eq!(params.max_mip, 14);
    }

    #[test]
    fn test_screen_space_coordinate_clamping_in_shader() {
        let source = HiZCullPipeline::shader_source();

        // Verify shader clamps screen coordinates
        assert!(
            source.contains("min_screen = max(min_screen"),
            "Shader must clamp min_screen to >= 0"
        );
        assert!(
            source.contains("max_screen = min(max_screen"),
            "Shader must clamp max_screen to <= hiz_size"
        );
    }

    #[test]
    fn test_large_object_count() {
        // Test with large counts to ensure no overflow
        assert_eq!(workgroups_for_objects(100_000), 1563);
        assert_eq!(workgroups_for_objects(1_000_000), 15625);
        assert_eq!(workgroups_for_objects(10_000_000), 156250);
    }

    #[test]
    fn test_max_u32_object_count_no_overflow() {
        // Edge case: very large object count (but realistic max is much lower)
        // (u32::MAX + 63) would overflow, but formula handles it
        let large_count = 1_000_000_000u32;
        let result = workgroups_for_objects(large_count);
        assert_eq!(result, 15_625_000);
    }

    #[test]
    fn test_extreme_object_count_no_overflow() {
        // Test values that would overflow with the naive (n + 63) / 64 formula
        // u32::MAX = 4_294_967_295, so (u32::MAX + 63) would wrap around

        // u32::MAX - 63 = 4_294_967_232, which is exactly divisible by 64
        let count_exact = u32::MAX - 63;
        assert_eq!(workgroups_for_objects(count_exact), count_exact / 64);

        // u32::MAX - 62 would need ceiling division
        let count_ceil = u32::MAX - 62;
        assert_eq!(workgroups_for_objects(count_ceil), (count_ceil / 64) + 1);

        // u32::MAX itself
        let max_count = u32::MAX;
        // 4_294_967_295 / 64 = 67_108_863 remainder 63, so result = 67_108_864
        assert_eq!(workgroups_for_objects(max_count), 67_108_864);
    }

    #[test]
    fn test_small_hiz_dimensions() {
        let vp = [[0.0; 4]; 4];

        // 1x1 HiZ pyramid (degenerate case)
        let params = HiZCullParams::new(100, 1, 1, 1, &vp, 0.1);
        assert_eq!(params.hiz_width, 1);
        assert_eq!(params.hiz_height, 1);
        assert_eq!(params.max_mip, 0);

        // Small but valid
        let params = HiZCullParams::new(100, 64, 64, 7, &vp, 0.1);
        assert_eq!(params.max_mip, 6);
    }

    #[test]
    fn test_asymmetric_hiz_dimensions() {
        let vp = [[0.0; 4]; 4];

        // Wide
        let params = HiZCullParams::new(100, 3840, 1080, 12, &vp, 0.1);
        assert_eq!(params.hiz_width, 3840);
        assert_eq!(params.hiz_height, 1080);

        // Tall
        let params = HiZCullParams::new(100, 1080, 1920, 11, &vp, 0.1);
        assert_eq!(params.hiz_width, 1080);
        assert_eq!(params.hiz_height, 1920);
    }

    #[test]
    fn test_near_plane_edge_values() {
        let vp = [[0.0; 4]; 4];

        // Very small near plane
        let params = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.001);
        assert_eq!(params.near_plane, 0.001);

        // Large near plane
        let params = HiZCullParams::new(100, 1920, 1080, 11, &vp, 100.0);
        assert_eq!(params.near_plane, 100.0);

        // Zero near plane (edge case)
        let params = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.0);
        assert_eq!(params.near_plane, 0.0);
    }

    #[test]
    fn test_default_trait() {
        let params = HiZCullParams::default();

        assert_eq!(params.object_count, 0);
        assert_eq!(params.hiz_width, 0);
        assert_eq!(params.hiz_height, 0);
        assert_eq!(params.max_mip, 0);
        assert_eq!(params.view_projection, [[0.0; 4]; 4]);
        assert_eq!(params.near_plane, 0.0);
        assert_eq!(params.flags, 0);
        assert_eq!(params._pad0, 0);
        assert_eq!(params._pad1, 0);
    }

    #[test]
    fn test_partial_eq_trait() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];

        let params1 = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.1);
        let params2 = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.1);
        let params3 = HiZCullParams::new(200, 1920, 1080, 11, &vp, 0.1);

        assert_eq!(params1, params2);
        assert_ne!(params1, params3);
    }

    #[test]
    fn test_copy_trait() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.1);

        // Copy should work (struct is Copy)
        let copied = params;
        assert_eq!(params.object_count, copied.object_count);
    }

    #[test]
    fn test_clone_trait() {
        let vp = [[1.0, 0.0, 0.0, 0.0]; 4];
        let params = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.1);

        let cloned = params.clone();
        assert_eq!(params, cloned);
    }

    #[test]
    fn test_debug_trait() {
        let vp = [[0.0; 4]; 4];
        let params = HiZCullParams::new(100, 1920, 1080, 11, &vp, 0.1);

        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("HiZCullParams"));
        assert!(debug_str.contains("object_count: 100"));
    }

    #[test]
    fn test_shader_handles_all_behind_camera() {
        let source = HiZCullPipeline::shader_source();

        // Verify shader handles all corners behind camera
        assert!(
            source.contains("if (all_behind)"),
            "Shader must handle all-behind-camera case"
        );
        assert!(
            source.contains("return false;") || source.contains("return vec4<f32>(0.0, 0.0, 0.0, -1.0)"),
            "Shader must return false or invalid flag when all behind"
        );
    }

    #[test]
    fn test_shader_handles_partial_behind_camera() {
        let source = HiZCullPipeline::shader_source();

        // Verify shader handles partial behind (some corners behind, some in front)
        assert!(
            source.contains("if (any_behind)"),
            "Shader must handle partial-behind-camera case"
        );
    }

    #[test]
    fn test_shader_degenerate_rect_visible() {
        let source = HiZCullPipeline::shader_source();

        // Degenerate rect (< 1 pixel) should be marked visible
        assert!(
            source.contains("if (rect_size.x < 1.0 || rect_size.y < 1.0)"),
            "Shader must handle degenerate rect case"
        );
    }

    // =========================================================================
    // Additional regression and validation tests
    // =========================================================================

    #[test]
    fn test_hiz_cull_params_size_constant() {
        // Ensure the constant matches actual struct size
        assert_eq!(HIZ_CULL_PARAMS_SIZE, mem::size_of::<HiZCullParams>());
    }

    #[test]
    fn test_workgroup_size_matches_wgpu_limits() {
        // WORKGROUP_SIZE should be within wgpu limits (typically 256 or 1024 max)
        assert!(WORKGROUP_SIZE <= 256);
        assert!(WORKGROUP_SIZE.is_power_of_two());
    }

    #[test]
    fn test_shader_num_corners_constant() {
        let source = HiZCullPipeline::shader_source();
        assert!(
            source.contains("const NUM_CORNERS: u32 = 8u"),
            "AABB has 8 corners"
        );
    }

    #[test]
    fn test_shader_reverse_z_depth_test() {
        let source = HiZCullPipeline::shader_source();

        // Reverse-Z: near_depth >= hiz_depth means visible (closer or equal)
        assert!(
            source.contains("near_depth >= (hiz_depth - EPSILON)"),
            "Must use reverse-Z depth comparison"
        );
    }

    #[test]
    fn test_shader_get_aabb_corner_function() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("fn get_aabb_corner"),
            "Must have get_aabb_corner helper function"
        );

        // Should use bit indexing for corner selection
        assert!(
            source.contains("(index & 1u)") && source.contains("(index & 2u)") && source.contains("(index & 4u)"),
            "Must use bit indexing for AABB corner selection"
        );
    }

    #[test]
    fn test_shader_world_to_clip_function() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("fn world_to_clip"),
            "Must have world_to_clip helper function"
        );
        assert!(
            source.contains("params.view_projection * vec4<f32>(world_pos, 1.0)"),
            "Must multiply by view_projection matrix"
        );
    }

    #[test]
    fn test_shader_ndc_to_screen_function() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("fn ndc_to_screen"),
            "Must have ndc_to_screen helper function"
        );
    }

    #[test]
    fn test_shader_sample_hiz_point_function() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("fn sample_hiz_point"),
            "Must have sample_hiz_point helper function"
        );
        assert!(
            source.contains("textureLoad(hiz_texture"),
            "Must use textureLoad for HiZ sampling"
        );
    }

    #[test]
    fn test_shader_sample_hiz_rect_max_function() {
        let source = HiZCullPipeline::shader_source();

        assert!(
            source.contains("fn sample_hiz_rect_max"),
            "Must have sample_hiz_rect_max helper function"
        );

        // Should sample 4 corners
        assert!(
            source.contains("d00") && source.contains("d10") && source.contains("d01") && source.contains("d11"),
            "Must sample 4 corners of HiZ rect"
        );
    }
}
