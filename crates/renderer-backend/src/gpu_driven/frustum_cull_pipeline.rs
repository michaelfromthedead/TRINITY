//! Frustum Cull Pipeline for GPU-driven rendering (T-WGPU-P6.3.3).
//!
//! This module provides a compute pipeline that performs frustum culling on
//! object data, writing visibility results to a bitfield buffer. It binds
//! the frustum planes, object storage buffer, and visibility flags buffer.
//!
//! # Overview
//!
//! The frustum cull pipeline tests each object's AABB against the view frustum
//! and sets the corresponding bit in the visibility flags buffer:
//!
//! ```text
//! +------------------+     +-------------------+
//! | FrustumBuffer    |---->|                   |
//! | (96 bytes)       |     |  FrustumCull      |
//! +------------------+     |  ComputePipeline  |
//!                          |  (64 threads/wg)  |
//! +------------------+     |                   |
//! | SceneDataBuffers |---->|                   |
//! | (ObjectData[])   |     +--------+----------+
//! +------------------+              |
//!                                   v
//!                          +-------------------+
//!                          | VisibilityFlags   |
//!                          | (1 bit/object)    |
//!                          +-------------------+
//! ```
//!
//! # Performance
//!
//! - Workgroup size: 64 threads (optimal for most GPUs)
//! - One thread per object
//! - Early-out culling on first plane failure
//! - Atomic OR for visibility bit setting
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{
//!     FrustumCullPipeline, FrustumBuffer, SceneDataBuffers, VisibilityFlagsBuffer,
//! };
//!
//! // Create pipeline
//! let pipeline = FrustumCullPipeline::new(&device);
//!
//! // Each frame:
//! frustum_buffer.update(&queue, &view_projection);
//! visibility.clear(&queue);
//!
//! pipeline.dispatch(
//!     &mut encoder,
//!     &device,
//!     &frustum_buffer,
//!     &scene_data,
//!     &visibility,
//!     object_count,
//! );
//! ```

use bytemuck::{Pod, Zeroable};
use std::mem;

use super::frustum::{FrustumBuffer, FRUSTUM_CULL_SHADER, FRUSTUM_PLANES_SIZE};
use super::object_data::OBJECT_DATA_SIZE;
use super::scene_data::SceneDataBuffers;
use super::visibility_flags::VisibilityFlagsBuffer;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Workgroup size for frustum culling (64 threads per workgroup).
///
/// 64 is chosen as a balance between:
/// - GPU occupancy (multiple of warp/wavefront size)
/// - Memory coalescing efficiency
/// - Thread-level parallelism
pub const WORKGROUP_SIZE: u32 = 64;

/// Size of CullDispatchParams in bytes.
pub const CULL_DISPATCH_PARAMS_SIZE: usize = 16;

// =============================================================================
// DISPATCH PARAMETERS
// =============================================================================

/// Parameters for frustum cull compute dispatch.
///
/// This uniform buffer is bound to provide the dispatch configuration
/// to the compute shader.
///
/// # Memory Layout (16 bytes)
///
/// | Offset | Field        | Size | Description                    |
/// |--------|--------------|------|--------------------------------|
/// | 0      | object_count | 4    | Number of objects to process   |
/// | 4      | flags        | 4    | Culling flags (reserved)       |
/// | 8      | _pad0        | 4    | Padding for alignment          |
/// | 12     | _pad1        | 4    | Padding for alignment          |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct CullDispatchParams {
    /// Number of objects to cull.
    pub object_count: u32,
    /// Flags (reserved for future extensions).
    pub flags: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    pub _pad1: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<CullDispatchParams>() == CULL_DISPATCH_PARAMS_SIZE);

impl CullDispatchParams {
    /// Create new dispatch parameters.
    #[inline]
    pub fn new(object_count: u32) -> Self {
        Self {
            object_count,
            flags: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create dispatch parameters with flags.
    #[inline]
    pub fn with_flags(object_count: u32, flags: u32) -> Self {
        Self {
            object_count,
            flags,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

// =============================================================================
// FRUSTUM CULL PIPELINE
// =============================================================================

/// Frustum culling compute pipeline.
///
/// This pipeline performs AABB-frustum intersection testing on GPU,
/// reading object data from a storage buffer and writing visibility
/// results to a bitfield buffer.
///
/// # Bind Groups
///
/// The pipeline uses two bind groups:
///
/// **Group 0 (Frustum)**:
/// - Binding 0: `FrustumPlanes` uniform buffer (96 bytes)
///
/// **Group 1 (Objects + Visibility)**:
/// - Binding 0: `CullDispatchParams` uniform buffer (16 bytes)
/// - Binding 1: `ObjectData[]` storage buffer (read-only)
/// - Binding 2: `visibility_flags` storage buffer (read-write, atomic)
///
/// # Thread Organization
///
/// - One thread per object
/// - Workgroup size: 64 threads
/// - Dispatch: ceil(object_count / 64) workgroups
pub struct FrustumCullPipeline {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,

    /// Bind group layout for frustum planes (Group 0).
    frustum_layout: wgpu::BindGroupLayout,

    /// Bind group layout for objects and visibility (Group 1).
    objects_layout: wgpu::BindGroupLayout,

    /// Pipeline layout.
    pipeline_layout: wgpu::PipelineLayout,
}

impl FrustumCullPipeline {
    /// Create a new frustum cull pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for pipeline creation
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pipeline = FrustumCullPipeline::new(&device);
    /// ```
    pub fn new(device: &wgpu::Device) -> Self {
        // Create shader module from embedded WGSL source
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frustum_cull_pipeline_shader"),
            source: wgpu::ShaderSource::Wgsl(Self::shader_source().into()),
        });

        // Group 0: Frustum planes uniform
        let frustum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frustum_cull_frustum_layout"),
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

        // Group 1: Params uniform, Objects storage (read), Visibility storage (read-write)
        let objects_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frustum_cull_objects_layout"),
            entries: &[
                // Binding 0: CullDispatchParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(CULL_DISPATCH_PARAMS_SIZE as u64).unwrap(),
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
            label: Some("frustum_cull_pipeline_layout"),
            bind_group_layouts: &[&frustum_layout, &objects_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frustum_cull_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "frustum_cull_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            frustum_layout,
            objects_layout,
            pipeline_layout,
        }
    }

    /// Generate the WGSL shader source for the frustum cull pipeline.
    ///
    /// This shader reads ObjectData, tests AABB against frustum planes,
    /// and writes visibility results using atomic OR operations.
    fn shader_source() -> &'static str {
        r#"
// Frustum Cull Pipeline Shader (T-WGPU-P6.3.3)
//
// Tests object AABBs against frustum planes and writes visibility
// to a bitfield buffer using atomic operations.

// Workgroup size: 64 threads
const WORKGROUP_SIZE: u32 = 64u;
const BITS_PER_WORD: u32 = 32u;

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

struct CullDispatchParams {
    object_count: u32,
    flags: u32,
    _pad0: u32,
    _pad1: u32,
}

// ObjectData layout (144 bytes)
// Must match Rust ObjectData struct exactly
// Uses array<f32, 4> instead of vec4<f32> to avoid WGSL 16-byte alignment for vec4
struct ObjectData {
    transform: mat4x4<f32>,     // 64 bytes  (offset 0)
    aabb_min: vec3<f32>,        // 12 bytes  (offset 64)
    _pad0: f32,                 // 4 bytes   (offset 76)
    aabb_max: vec3<f32>,        // 12 bytes  (offset 80)
    _pad1: f32,                 // 4 bytes   (offset 92)
    mesh_index: u32,            // 4 bytes   (offset 96)
    material_index: u32,        // 4 bytes   (offset 100)
    lod_distances: array<f32, 4>, // 16 bytes (offset 104)
    flags: u32,                 // 4 bytes   (offset 120)
    _padding: array<u32, 5>,    // 20 bytes  (offset 124)
}

// ============================================================================
// Bindings
// ============================================================================

// Group 0: Frustum planes
@group(0) @binding(0) var<uniform> frustum: FrustumPlanes;

// Group 1: Objects and visibility
@group(1) @binding(0) var<uniform> params: CullDispatchParams;
@group(1) @binding(1) var<storage, read> objects: array<ObjectData>;
@group(1) @binding(2) var<storage, read_write> visibility_flags: array<atomic<u32>>;

// ============================================================================
// Frustum Culling Functions
// ============================================================================

/// Test AABB against frustum using p-vertex optimization.
/// Returns true if visible, false if culled.
fn test_aabb_frustum(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
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
// Main Compute Entry Point
// ============================================================================

@compute @workgroup_size(64, 1, 1)
fn frustum_cull_main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Test AABB against frustum
    let visible = test_aabb_frustum(obj.aabb_min, obj.aabb_max);

    // Set visibility bit atomically
    if (visible) {
        let word_idx = object_idx / BITS_PER_WORD;
        let bit_mask = 1u << (object_idx % BITS_PER_WORD);
        atomicOr(&visibility_flags[word_idx], bit_mask);
    }
}
"#
    }

    /// Dispatch the frustum culling compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder for recording commands
    /// * `device` - The wgpu device for bind group creation
    /// * `queue` - The wgpu queue for buffer writes
    /// * `frustum` - Frustum planes buffer
    /// * `objects` - Scene data buffers containing objects
    /// * `visibility` - Visibility flags buffer for output
    /// * `object_count` - Number of objects to process
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut encoder = device.create_command_encoder(&Default::default());
    /// pipeline.dispatch(
    ///     &mut encoder,
    ///     &device,
    ///     &queue,
    ///     &frustum_buffer,
    ///     &scene_data,
    ///     &visibility,
    ///     1000,
    /// );
    /// queue.submit([encoder.finish()]);
    /// ```
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frustum: &FrustumBuffer,
        objects: &SceneDataBuffers,
        visibility: &VisibilityFlagsBuffer,
        object_count: u32,
    ) {
        if object_count == 0 {
            return;
        }

        // Create params buffer
        let params = CullDispatchParams::new(object_count);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frustum_cull_params_buffer"),
            size: CULL_DISPATCH_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create frustum bind group (Group 0)
        let frustum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frustum_cull_frustum_bind_group"),
            layout: &self.frustum_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frustum.buffer().as_entire_binding(),
            }],
        });

        // Create objects bind group (Group 1)
        let objects_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frustum_cull_objects_bind_group"),
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
        let workgroup_count = (object_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // Begin compute pass and dispatch
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frustum_cull_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &frustum_bind_group, &[]);
            pass.set_bind_group(1, &objects_bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
    }

    /// Dispatch with pre-created bind groups for improved performance.
    ///
    /// Use this variant when dispatching multiple times with the same
    /// bind groups to avoid per-frame bind group creation overhead.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder for recording commands
    /// * `frustum_bind_group` - Pre-created frustum bind group
    /// * `objects_bind_group` - Pre-created objects bind group
    /// * `object_count` - Number of objects to process
    pub fn dispatch_with_bind_groups(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frustum_bind_group: &wgpu::BindGroup,
        objects_bind_group: &wgpu::BindGroup,
        object_count: u32,
    ) {
        if object_count == 0 {
            return;
        }

        let workgroup_count = (object_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frustum_cull_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, frustum_bind_group, &[]);
            pass.set_bind_group(1, objects_bind_group, &[]);
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

    /// Get the objects bind group layout (Group 1).
    #[inline]
    pub fn objects_layout(&self) -> &wgpu::BindGroupLayout {
        &self.objects_layout
    }

    /// Get the pipeline layout.
    #[inline]
    pub fn pipeline_layout(&self) -> &wgpu::PipelineLayout {
        &self.pipeline_layout
    }

    /// Create a frustum bind group for the given frustum buffer.
    ///
    /// Use with `dispatch_with_bind_groups()` for optimal performance.
    pub fn create_frustum_bind_group(
        &self,
        device: &wgpu::Device,
        frustum: &FrustumBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frustum_cull_frustum_bind_group"),
            layout: &self.frustum_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frustum.buffer().as_entire_binding(),
            }],
        })
    }

    /// Create an objects bind group for the given buffers.
    ///
    /// Use with `dispatch_with_bind_groups()` for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `params_buffer` - Uniform buffer containing CullDispatchParams
    /// * `objects` - Scene data buffers
    /// * `visibility` - Visibility flags buffer
    pub fn create_objects_bind_group(
        &self,
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
        objects: &SceneDataBuffers,
        visibility: &VisibilityFlagsBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frustum_cull_objects_bind_group"),
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

impl std::fmt::Debug for FrustumCullPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrustumCullPipeline")
            .field("workgroup_size", &WORKGROUP_SIZE)
            .finish_non_exhaustive()
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Calculate the number of workgroups needed for N objects.
#[inline]
pub const fn workgroups_for_objects(object_count: u32) -> u32 {
    (object_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // CullDispatchParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cull_dispatch_params_size() {
        assert_eq!(
            mem::size_of::<CullDispatchParams>(),
            16,
            "CullDispatchParams must be 16 bytes"
        );
        assert_eq!(CULL_DISPATCH_PARAMS_SIZE, 16);
    }

    #[test]
    fn test_cull_dispatch_params_new() {
        let params = CullDispatchParams::new(1000);
        assert_eq!(params.object_count, 1000);
        assert_eq!(params.flags, 0);
    }

    #[test]
    fn test_cull_dispatch_params_with_flags() {
        let params = CullDispatchParams::with_flags(500, 0x01);
        assert_eq!(params.object_count, 500);
        assert_eq!(params.flags, 0x01);
    }

    #[test]
    fn test_cull_dispatch_params_bytemuck() {
        let params = CullDispatchParams::new(42);
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);

        // First 4 bytes should be object_count (little-endian)
        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(count, 42);
    }

    // -------------------------------------------------------------------------
    // Workgroup Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_workgroups_for_objects_zero() {
        assert_eq!(workgroups_for_objects(0), 0);
    }

    #[test]
    fn test_workgroups_for_objects_one() {
        assert_eq!(workgroups_for_objects(1), 1);
    }

    #[test]
    fn test_workgroups_for_objects_exact() {
        assert_eq!(workgroups_for_objects(64), 1);
        assert_eq!(workgroups_for_objects(128), 2);
        assert_eq!(workgroups_for_objects(256), 4);
    }

    #[test]
    fn test_workgroups_for_objects_boundary() {
        assert_eq!(workgroups_for_objects(63), 1);
        assert_eq!(workgroups_for_objects(65), 2);
        assert_eq!(workgroups_for_objects(127), 2);
        assert_eq!(workgroups_for_objects(129), 3);
    }

    #[test]
    fn test_workgroups_for_objects_large() {
        // 100,000 objects should need 1563 workgroups (ceil(100000/64))
        assert_eq!(workgroups_for_objects(100_000), 1563);

        // 1 million objects
        assert_eq!(workgroups_for_objects(1_000_000), 15625);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_workgroup_size_power_of_two() {
        // Workgroup size should be power of 2 for optimal performance
        assert!(WORKGROUP_SIZE.is_power_of_two());
    }

    #[test]
    fn test_workgroup_size_value() {
        // Verify the documented workgroup size
        assert_eq!(WORKGROUP_SIZE, 64);
    }

    // -------------------------------------------------------------------------
    // Shader Source Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_source_not_empty() {
        let source = FrustumCullPipeline::shader_source();
        assert!(!source.is_empty());
    }

    #[test]
    fn test_shader_source_contains_entry_point() {
        let source = FrustumCullPipeline::shader_source();
        assert!(source.contains("fn frustum_cull_main"));
    }

    #[test]
    fn test_shader_source_contains_workgroup_size() {
        let source = FrustumCullPipeline::shader_source();
        assert!(source.contains("@workgroup_size(64, 1, 1)"));
    }

    #[test]
    fn test_shader_source_contains_bindings() {
        let source = FrustumCullPipeline::shader_source();
        assert!(source.contains("@group(0) @binding(0)"));
        assert!(source.contains("@group(1) @binding(0)"));
        assert!(source.contains("@group(1) @binding(1)"));
        assert!(source.contains("@group(1) @binding(2)"));
    }

    #[test]
    fn test_shader_source_contains_frustum_test() {
        let source = FrustumCullPipeline::shader_source();
        assert!(source.contains("fn test_aabb_frustum"));
    }

    #[test]
    fn test_shader_source_contains_atomic_or() {
        let source = FrustumCullPipeline::shader_source();
        assert!(source.contains("atomicOr"));
    }

    // -------------------------------------------------------------------------
    // Integration Tests (require wgpu device - skipped in unit tests)
    // -------------------------------------------------------------------------

    // Note: Full pipeline tests require a wgpu::Device and should be
    // run as integration tests in tests/blackbox_frustum_cull_pipeline.rs
}
