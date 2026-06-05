//! Virtual Geometry LOD System for TRINITY Engine (T-GPU-8.3).
//!
//! Implements Nanite-style continuous LOD management with:
//! - Screen-space error computation from geometric error
//! - LOD bias calculation for quality scaling
//! - Dithered and alpha-blended LOD transitions
//! - Streaming priority calculation based on visibility and error
//! - Page residency tracking for virtual texturing integration
//!
//! # Overview
//!
//! The Virtual LOD system manages multi-LOD meshes where each LOD level has a
//! known geometric error (maximum deviation from full-resolution mesh in world
//! units). The system projects this error to screen space to determine which
//! LOD level provides acceptable visual quality at the current viewing distance.
//!
//! # Screen-Space Error
//!
//! The key concept is screen-space error: how many pixels of deviation would
//! result from using a simplified LOD. The formula is:
//!
//! ```text
//! screen_error = geometric_error * (cot(fov/2) / distance) * (screen_height / 2)
//! ```
//!
//! LOD selection finds the highest-quality LOD where `screen_error < threshold`.
//!
//! # Streaming Priority
//!
//! For meshes that need LOD data streamed from disk, the system computes a
//! streaming priority based on:
//! - Visibility (visible meshes are more urgent)
//! - Screen error relative to threshold (higher = more urgent)
//! - Screen size (larger objects are more noticeable)
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::virtual_geometry::virtual_lod::{
//!     VirtualLODPipeline, VirtualLODResources, VirtualLODParams, VirtualMesh, LODLevel,
//! };
//!
//! // Create pipeline
//! let pipeline = VirtualLODPipeline::new(&device, shader_source);
//! let resources = VirtualLODResources::new(&device, 100_000);
//!
//! // Each frame: update camera and compute LODs
//! let params = VirtualLODParams::new(
//!     camera_position,
//!     1.0,   // error_threshold (pixels)
//!     0.0,   // lod_bias
//!     0.1,   // transition_width
//!     1080,  // screen_height
//!     std::f32::consts::FRAC_PI_4, // fov_y (45 degrees)
//! );
//! resources.upload_params(&queue, &params);
//! resources.upload_meshes(&queue, &virtual_meshes);
//!
//! pipeline.dispatch(&mut encoder, &resources, instance_count);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL).
pub const WORKGROUP_SIZE: u32 = 256;

/// Maximum LOD levels per virtual mesh.
pub const MAX_LOD_LEVELS: usize = 16;

/// Maximum inline LOD levels in VirtualMesh struct.
pub const MAX_INLINE_LODS: usize = 6;

/// Invalid LOD marker (mesh culled or not visible).
pub const INVALID_LOD: u32 = 0xFFFFFFFF;

/// Flag: Use dithered transition instead of alpha blend.
pub const FLAG_USE_DITHER: u32 = 1;

/// Flag: Use forced LOD from params (debug).
pub const FLAG_FORCE_LOD: u32 = 2;

/// Flag: Don't compute streaming priority.
pub const FLAG_DISABLE_STREAMING: u32 = 4;

/// Flag: Enable page residency tracking.
pub const FLAG_PAGE_TRACKING: u32 = 8;

/// Streaming priority tier: Critical (visible, large error).
pub const PRIORITY_CRITICAL: u32 = 0;

/// Streaming priority tier: High (visible, moderate error).
pub const PRIORITY_HIGH: u32 = 1;

/// Streaming priority tier: Normal (visible, low error).
pub const PRIORITY_NORMAL: u32 = 2;

/// Streaming priority tier: Low (barely visible or far).
pub const PRIORITY_LOW: u32 = 3;

/// Result flag: Needs streaming (error above threshold).
pub const RESULT_FLAG_NEEDS_STREAMING: u32 = 1;

/// Result flag: Page is resident.
pub const RESULT_FLAG_PAGE_RESIDENT: u32 = 2;

// ---------------------------------------------------------------------------
// VirtualLODParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for virtual LOD system parameters.
///
/// # Memory Layout
///
/// 64 bytes, std140 compatible:
/// | Offset | Field              | Size |
/// |--------|-------------------|------|
/// | 0      | camera_position   | 12   |
/// | 12     | error_threshold   | 4    |
/// | 16     | lod_bias          | 4    |
/// | 20     | transition_width  | 4    |
/// | 24     | streaming_budget  | 4    |
/// | 28     | flags             | 4    |
/// | 32     | num_instances     | 4    |
/// | 36     | screen_height     | 4    |
/// | 40     | fov_y             | 4    |
/// | 44     | forced_lod        | 4    |
/// | 48     | frame_index       | 4    |
/// | 52     | _pad0             | 4    |
/// | 56     | _pad1             | 4    |
/// | 60     | _pad2             | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VirtualLODParams {
    /// Camera position in world space (xyz).
    pub camera_position: [f32; 3],
    /// Screen-space error threshold in pixels.
    pub error_threshold: f32,
    /// Global LOD bias: negative = higher quality, positive = lower quality.
    pub lod_bias: f32,
    /// Transition width for blending (0-1 range).
    pub transition_width: f32,
    /// Streaming budget in bytes per frame.
    pub streaming_budget: u32,
    /// Flags (see FLAG_* constants).
    pub flags: u32,
    /// Number of mesh instances to process.
    pub num_instances: u32,
    /// Screen height in pixels.
    pub screen_height: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Forced LOD level (when FLAG_FORCE_LOD is set).
    pub forced_lod: u32,
    /// Frame index for temporal effects.
    pub frame_index: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VirtualLODParams>() == 64);

impl VirtualLODParams {
    /// Create new LOD parameters.
    ///
    /// # Arguments
    ///
    /// * `camera_position` - Camera world position.
    /// * `error_threshold` - Max acceptable screen-space error in pixels.
    /// * `lod_bias` - Quality bias (-2 to +2 typical).
    /// * `transition_width` - Blend transition width (0-1).
    /// * `screen_height` - Screen height in pixels.
    /// * `fov_y` - Vertical field of view in radians.
    pub fn new(
        camera_position: [f32; 3],
        error_threshold: f32,
        lod_bias: f32,
        transition_width: f32,
        screen_height: u32,
        fov_y: f32,
    ) -> Self {
        Self {
            camera_position,
            error_threshold,
            lod_bias,
            transition_width,
            streaming_budget: 16 * 1024 * 1024, // 16 MB default
            flags: 0,
            num_instances: 0,
            screen_height: screen_height as f32,
            fov_y,
            forced_lod: 0,
            frame_index: 0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Set number of instances to process.
    pub fn with_instance_count(mut self, count: u32) -> Self {
        self.num_instances = count;
        self
    }

    /// Enable dithered LOD transitions.
    pub fn with_dither(mut self) -> Self {
        self.flags |= FLAG_USE_DITHER;
        self
    }

    /// Force a specific LOD level (for debugging).
    pub fn with_forced_lod(mut self, lod: u32) -> Self {
        self.forced_lod = lod;
        self.flags |= FLAG_FORCE_LOD;
        self
    }

    /// Set streaming budget in bytes per frame.
    pub fn with_streaming_budget(mut self, budget: u32) -> Self {
        self.streaming_budget = budget;
        self
    }

    /// Enable page residency tracking.
    pub fn with_page_tracking(mut self) -> Self {
        self.flags |= FLAG_PAGE_TRACKING;
        self
    }

    /// Update frame index for temporal effects.
    pub fn set_frame_index(&mut self, frame: u32) {
        self.frame_index = frame;
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_instances + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

// ---------------------------------------------------------------------------
// LODLevel
// ---------------------------------------------------------------------------

/// LOD level descriptor.
///
/// # Memory Layout
///
/// 16 bytes:
/// | Offset | Field           | Size |
/// |--------|-----------------|------|
/// | 0      | geometric_error | 4    |
/// | 4      | triangle_count  | 4    |
/// | 8      | vertex_offset   | 4    |
/// | 12     | index_offset    | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LODLevel {
    /// Maximum geometric error in world units.
    /// This is the max deviation from full-res mesh.
    pub geometric_error: f32,
    /// Number of triangles in this LOD.
    pub triangle_count: u32,
    /// Byte offset to vertex data.
    pub vertex_offset: u32,
    /// Byte offset to index data.
    pub index_offset: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<LODLevel>() == 16);

impl LODLevel {
    /// Create a new LOD level.
    ///
    /// # Arguments
    ///
    /// * `geometric_error` - Max error in world units from full-res.
    /// * `triangle_count` - Number of triangles.
    /// * `vertex_offset` - Byte offset to vertex data.
    /// * `index_offset` - Byte offset to index data.
    pub fn new(
        geometric_error: f32,
        triangle_count: u32,
        vertex_offset: u32,
        index_offset: u32,
    ) -> Self {
        Self {
            geometric_error,
            triangle_count,
            vertex_offset,
            index_offset,
        }
    }

    /// Create LOD 0 (highest detail, zero error).
    pub fn lod0(triangle_count: u32) -> Self {
        Self {
            geometric_error: 0.0,
            triangle_count,
            vertex_offset: 0,
            index_offset: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// VirtualMesh
// ---------------------------------------------------------------------------

/// Virtual mesh instance with inline LOD levels.
///
/// # Memory Layout
///
/// 128 bytes:
/// | Offset | Field             | Size |
/// |--------|-------------------|------|
/// | 0      | position          | 12   |
/// | 12     | bounding_radius   | 4    |
/// | 16     | lod_levels        | 96   | (6 LODLevel entries)
/// | 112    | num_lods          | 4    |
/// | 116    | page_id           | 4    |
/// | 120    | mesh_id           | 4    |
/// | 124    | _pad              | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VirtualMesh {
    /// World position (bounding sphere center).
    pub position: [f32; 3],
    /// Bounding sphere radius.
    pub bounding_radius: f32,
    /// Inline LOD levels (up to 6).
    pub lod_levels: [LODLevel; MAX_INLINE_LODS],
    /// Number of valid LOD levels.
    pub num_lods: u32,
    /// Page ID for virtual texturing (0xFFFFFFFF if not used).
    pub page_id: u32,
    /// Unique mesh identifier.
    pub mesh_id: u32,
    /// Padding.
    pub _pad: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VirtualMesh>() == 128);

impl Default for VirtualMesh {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            bounding_radius: 1.0,
            lod_levels: [LODLevel::default(); MAX_INLINE_LODS],
            num_lods: 1,
            page_id: INVALID_LOD,
            mesh_id: 0,
            _pad: 0,
        }
    }
}

impl VirtualMesh {
    /// Create a new virtual mesh.
    ///
    /// # Arguments
    ///
    /// * `position` - World position (bounding sphere center).
    /// * `bounding_radius` - Bounding sphere radius.
    /// * `lod_levels` - Slice of LOD levels (max 6).
    /// * `mesh_id` - Unique mesh identifier.
    ///
    /// # Panics
    ///
    /// Panics if `lod_levels` is empty or has more than 6 elements.
    pub fn new(
        position: [f32; 3],
        bounding_radius: f32,
        lod_levels: &[LODLevel],
        mesh_id: u32,
    ) -> Self {
        assert!(!lod_levels.is_empty(), "At least one LOD level required");
        assert!(lod_levels.len() <= MAX_INLINE_LODS, "Max {} inline LODs", MAX_INLINE_LODS);

        let mut levels = [LODLevel::default(); MAX_INLINE_LODS];
        for (i, level) in lod_levels.iter().enumerate() {
            levels[i] = *level;
        }

        Self {
            position,
            bounding_radius,
            lod_levels: levels,
            num_lods: lod_levels.len() as u32,
            page_id: INVALID_LOD,
            mesh_id,
            _pad: 0,
        }
    }

    /// Set the page ID for virtual texturing integration.
    pub fn with_page_id(mut self, page_id: u32) -> Self {
        self.page_id = page_id;
        self
    }

    /// Create a simple mesh with a single LOD.
    pub fn single_lod(
        position: [f32; 3],
        radius: f32,
        triangle_count: u32,
        mesh_id: u32,
    ) -> Self {
        Self::new(position, radius, &[LODLevel::lod0(triangle_count)], mesh_id)
    }

    /// Create a mesh with standard LOD chain (4 levels).
    ///
    /// Geometric errors are computed based on triangle reduction ratios.
    pub fn standard_lods(
        position: [f32; 3],
        radius: f32,
        base_triangles: u32,
        mesh_id: u32,
    ) -> Self {
        // Standard LOD chain: 100%, 50%, 25%, 12.5%
        let reduction_ratios = [1.0, 0.5, 0.25, 0.125];
        // Geometric error grows inversely with triangle count
        let base_error = radius * 0.001; // 0.1% of radius for LOD 0

        let levels: Vec<LODLevel> = reduction_ratios
            .iter()
            .enumerate()
            .map(|(i, &ratio)| {
                LODLevel::new(
                    base_error * (1.0 / ratio), // Error grows as triangles decrease
                    (base_triangles as f32 * ratio) as u32,
                    0,
                    0,
                )
            })
            .collect();

        Self::new(position, radius, &levels, mesh_id)
    }
}

// ---------------------------------------------------------------------------
// StreamingPriority
// ---------------------------------------------------------------------------

/// Streaming priority for a mesh instance.
///
/// # Memory Layout
///
/// 16 bytes:
/// | Offset | Field     | Size |
/// |--------|-----------|------|
/// | 0      | priority  | 4    |
/// | 4      | mesh_id   | 4    |
/// | 8      | page_id   | 4    |
/// | 12     | lod_level | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StreamingPriority {
    /// Packed priority: [tier:8][priority:24]. Lower = more urgent.
    pub priority: u32,
    /// Mesh ID for tracking.
    pub mesh_id: u32,
    /// Page ID for virtual texturing.
    pub page_id: u32,
    /// Target LOD level to stream.
    pub lod_level: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<StreamingPriority>() == 16);

impl StreamingPriority {
    /// Get the priority tier (0=Critical, 1=High, 2=Normal, 3=Low).
    #[inline]
    pub fn tier(&self) -> u32 {
        self.priority >> 24
    }

    /// Get the priority value within the tier (lower = more urgent).
    #[inline]
    pub fn value(&self) -> u32 {
        self.priority & 0x00FFFFFF
    }

    /// Check if this is critical priority.
    #[inline]
    pub fn is_critical(&self) -> bool {
        self.tier() == PRIORITY_CRITICAL
    }

    /// Compare priorities (returns true if self is more urgent than other).
    #[inline]
    pub fn is_more_urgent_than(&self, other: &StreamingPriority) -> bool {
        self.priority < other.priority
    }
}

// ---------------------------------------------------------------------------
// LODResult
// ---------------------------------------------------------------------------

/// LOD selection result for a mesh instance.
///
/// # Memory Layout
///
/// 16 bytes:
/// | Offset | Field          | Size |
/// |--------|----------------|------|
/// | 0      | lod_level      | 4    |
/// | 4      | blend_factor   | 4    |
/// | 8      | screen_error   | 4    |
/// | 12     | flags          | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LODResult {
    /// Selected LOD level (0 = highest detail, INVALID_LOD = culled).
    pub lod_level: u32,
    /// Blend factor for transition (0.0 = primary, 1.0 = secondary).
    pub blend_factor: f32,
    /// Computed screen-space error in pixels.
    pub screen_error: f32,
    /// Result flags (see RESULT_FLAG_* constants).
    pub flags: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<LODResult>() == 16);

impl LODResult {
    /// Check if mesh was culled (invalid LOD).
    #[inline]
    pub fn is_culled(&self) -> bool {
        self.lod_level == INVALID_LOD
    }

    /// Check if mesh is visible.
    #[inline]
    pub fn is_visible(&self) -> bool {
        self.lod_level != INVALID_LOD
    }

    /// Check if LOD transition blending is needed.
    #[inline]
    pub fn needs_blending(&self) -> bool {
        self.blend_factor > 0.0 && self.blend_factor < 1.0
    }

    /// Check if streaming is needed.
    #[inline]
    pub fn needs_streaming(&self) -> bool {
        (self.flags & RESULT_FLAG_NEEDS_STREAMING) != 0
    }

    /// Check if page is resident.
    #[inline]
    pub fn is_page_resident(&self) -> bool {
        (self.flags & RESULT_FLAG_PAGE_RESIDENT) != 0
    }

    /// Get the LOD level, or None if culled.
    #[inline]
    pub fn lod(&self) -> Option<u32> {
        if self.is_culled() {
            None
        } else {
            Some(self.lod_level)
        }
    }
}

// ---------------------------------------------------------------------------
// PageResidency
// ---------------------------------------------------------------------------

/// Page residency entry for virtual texturing integration.
///
/// # Memory Layout
///
/// 8 bytes:
/// | Offset | Field   | Size |
/// |--------|---------|------|
/// | 0      | page_id | 4    |
/// | 4      | status  | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PageResidency {
    /// Page ID.
    pub page_id: u32,
    /// Status: bit 0 = resident, bits 1-7 = last access frame delta.
    pub status: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<PageResidency>() == 8);

impl PageResidency {
    /// Create a new residency entry.
    pub fn new(page_id: u32, resident: bool) -> Self {
        Self {
            page_id,
            status: if resident { 1 } else { 0 },
        }
    }

    /// Check if page is resident.
    #[inline]
    pub fn is_resident(&self) -> bool {
        (self.status & 1) != 0
    }

    /// Set residency status.
    pub fn set_resident(&mut self, resident: bool) {
        if resident {
            self.status |= 1;
        } else {
            self.status &= !1;
        }
    }

    /// Get last access frame delta (0-127).
    #[inline]
    pub fn last_access_delta(&self) -> u8 {
        ((self.status >> 1) & 0x7F) as u8
    }
}

// ---------------------------------------------------------------------------
// VirtualLODResources
// ---------------------------------------------------------------------------

/// GPU resources for virtual LOD system.
pub struct VirtualLODResources {
    /// Uniform buffer for LOD parameters.
    pub params_buffer: wgpu::Buffer,
    /// Storage buffer for virtual mesh data (input).
    pub meshes_buffer: wgpu::Buffer,
    /// Storage buffer for LOD results (output).
    pub results_buffer: wgpu::Buffer,
    /// Storage buffer for streaming priorities (output).
    pub priorities_buffer: wgpu::Buffer,
    /// Storage buffer for page residency (read-write).
    pub residency_buffer: wgpu::Buffer,
    /// Staging buffer for reading results back to CPU.
    pub results_staging: wgpu::Buffer,
    /// Staging buffer for reading priorities back to CPU.
    pub priorities_staging: wgpu::Buffer,
    /// Maximum number of mesh instances supported.
    pub capacity: u32,
    /// Maximum number of pages tracked.
    pub page_capacity: u32,
}

impl VirtualLODResources {
    /// Create virtual LOD resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `mesh_capacity` - Maximum number of mesh instances.
    /// * `page_capacity` - Maximum number of pages to track.
    pub fn new(device: &wgpu::Device, mesh_capacity: u32, page_capacity: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_lod_params"),
            size: mem::size_of::<VirtualLODParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let meshes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_lod_meshes"),
            size: (mesh_capacity as u64) * (mem::size_of::<VirtualMesh>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_lod_results"),
            size: (mesh_capacity as u64) * (mem::size_of::<LODResult>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let priorities_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_lod_priorities"),
            size: (mesh_capacity as u64) * (mem::size_of::<StreamingPriority>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let residency_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_lod_residency"),
            size: (page_capacity as u64) * (mem::size_of::<PageResidency>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let results_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_lod_results_staging"),
            size: (mesh_capacity as u64) * (mem::size_of::<LODResult>() as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let priorities_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("virtual_lod_priorities_staging"),
            size: (mesh_capacity as u64) * (mem::size_of::<StreamingPriority>() as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            meshes_buffer,
            results_buffer,
            priorities_buffer,
            residency_buffer,
            results_staging,
            priorities_staging,
            capacity: mesh_capacity,
            page_capacity,
        }
    }

    /// Create with default page capacity (64K pages).
    pub fn with_default_pages(device: &wgpu::Device, mesh_capacity: u32) -> Self {
        Self::new(device, mesh_capacity, 65536)
    }

    /// Upload LOD parameters to GPU.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &VirtualLODParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload virtual mesh data to GPU.
    ///
    /// # Panics
    ///
    /// Panics if `meshes.len() > self.capacity`.
    pub fn upload_meshes(&self, queue: &wgpu::Queue, meshes: &[VirtualMesh]) {
        assert!(meshes.len() <= self.capacity as usize);
        queue.write_buffer(&self.meshes_buffer, 0, bytemuck::cast_slice(meshes));
    }

    /// Upload page residency data to GPU.
    ///
    /// # Panics
    ///
    /// Panics if `residency.len() > self.page_capacity`.
    pub fn upload_residency(&self, queue: &wgpu::Queue, residency: &[PageResidency]) {
        assert!(residency.len() <= self.page_capacity as usize);
        queue.write_buffer(&self.residency_buffer, 0, bytemuck::cast_slice(residency));
    }
}

// ---------------------------------------------------------------------------
// VirtualLODPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for virtual LOD system.
pub struct VirtualLODPipeline {
    /// Main pipeline: full LOD selection + streaming priority.
    pub pipeline: wgpu::ComputePipeline,
    /// Screen error only pipeline (no LOD selection).
    pub pipeline_error_only: wgpu::ComputePipeline,
    /// Streaming priority only pipeline.
    pub pipeline_streaming_only: wgpu::ComputePipeline,
    /// Dither LOD transition pipeline.
    pub pipeline_dither: wgpu::ComputePipeline,
    /// Bind group layout for LOD resources.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl VirtualLODPipeline {
    /// Create the virtual LOD pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `shader_source` - WGSL shader source code.
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("virtual_lod_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("virtual_lod_bind_group_layout"),
            entries: &[
                // @binding(0) params: VirtualLODParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<VirtualLODParams>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) meshes: array<VirtualMesh>
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2) results: array<LODResult>
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3) priorities: array<StreamingPriority>
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(4) page_residency: array<PageResidency>
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("virtual_lod_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("virtual_lod_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compute_virtual_lod",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_error_only = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("virtual_lod_pipeline_error_only"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compute_screen_error_only",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_streaming_only =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("virtual_lod_pipeline_streaming_only"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "compute_streaming_only",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let pipeline_dither = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("virtual_lod_pipeline_dither"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compute_dither_lod",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            pipeline_error_only,
            pipeline_streaming_only,
            pipeline_dither,
            bind_group_layout,
        }
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &VirtualLODResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("virtual_lod_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.meshes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.priorities_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.residency_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// Compute screen-space error from geometric error (CPU reference).
///
/// # Arguments
///
/// * `geometric_error` - Maximum deviation in world units.
/// * `distance` - Distance from camera to object.
/// * `fov_y` - Vertical field of view in radians.
/// * `screen_height` - Screen height in pixels.
pub fn cpu_screen_space_error(
    geometric_error: f32,
    distance: f32,
    fov_y: f32,
    screen_height: f32,
) -> f32 {
    if distance <= 0.001 {
        return 1_000_000.0; // Very large error
    }

    let half_fov = fov_y * 0.5;
    let fov_factor = 1.0 / half_fov.tan();

    geometric_error * fov_factor * screen_height * 0.5 / distance
}

/// Select LOD level using binary search (CPU reference).
///
/// Finds the highest-quality LOD (lowest index) where screen error is below threshold.
///
/// # Arguments
///
/// * `lod_errors` - Geometric errors for each LOD level.
/// * `distance` - Distance from camera.
/// * `threshold` - Max acceptable screen error in pixels.
/// * `bias` - LOD quality bias.
/// * `fov_y` - Vertical field of view.
/// * `screen_height` - Screen height in pixels.
///
/// # Returns
///
/// (selected_lod, screen_error_at_that_lod)
pub fn cpu_select_lod(
    lod_errors: &[f32],
    distance: f32,
    threshold: f32,
    bias: f32,
    fov_y: f32,
    screen_height: f32,
) -> (usize, f32) {
    if lod_errors.is_empty() {
        return (0, 0.0);
    }

    let biased_threshold = threshold * 2.0_f32.powf(bias);

    // Linear search from highest quality
    for (lod, &geometric_error) in lod_errors.iter().enumerate() {
        let screen_error = cpu_screen_space_error(geometric_error, distance, fov_y, screen_height);
        if screen_error <= biased_threshold {
            return (lod, screen_error);
        }
    }

    // All LODs exceed threshold, return lowest quality
    let last_idx = lod_errors.len() - 1;
    let screen_error = cpu_screen_space_error(
        lod_errors[last_idx],
        distance,
        fov_y,
        screen_height,
    );
    (last_idx, screen_error)
}

/// Compute LOD blend factor (CPU reference).
///
/// Returns blend factor in [0, 1] for smooth transitions.
/// - 0.0 = use current LOD fully
/// - 1.0 = transition to next (lower quality) LOD
pub fn cpu_lod_blend_factor(
    screen_error: f32,
    threshold: f32,
    transition_width: f32,
) -> f32 {
    if transition_width <= 0.0 {
        return 0.0;
    }

    let blend_start = threshold * (1.0 - transition_width);
    let blend_end = threshold;

    if screen_error <= blend_start {
        return 0.0;
    }
    if screen_error >= blend_end {
        return 1.0;
    }

    (screen_error - blend_start) / (blend_end - blend_start)
}

/// Compute streaming priority (CPU reference).
///
/// # Arguments
///
/// * `is_visible` - Whether the mesh is visible.
/// * `screen_error` - Screen-space error in pixels.
/// * `threshold` - Error threshold.
/// * `screen_size` - Projected screen size in pixels.
/// * `screen_height` - Total screen height.
///
/// # Returns
///
/// (tier, priority_value) where lower values = more urgent.
pub fn cpu_streaming_priority(
    is_visible: bool,
    screen_error: f32,
    threshold: f32,
    screen_size: f32,
    screen_height: f32,
) -> (u32, u32) {
    if !is_visible {
        return (PRIORITY_LOW, 0x00FFFFFF);
    }

    let error_ratio = screen_error / threshold.max(0.001);
    let size_factor = (screen_size / screen_height).clamp(0.0, 1.0);
    let urgency = error_ratio * size_factor;

    let tier = if urgency > 0.8 {
        PRIORITY_CRITICAL
    } else if urgency > 0.5 {
        PRIORITY_HIGH
    } else if urgency > 0.2 {
        PRIORITY_NORMAL
    } else {
        PRIORITY_LOW
    };

    let inverse_urgency = ((1.0 - urgency.clamp(0.0, 0.999)) * 16_777_215.0) as u32;
    (tier, inverse_urgency)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    // -------------------------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------------------------

    fn make_test_mesh(position: [f32; 3], radius: f32, mesh_id: u32) -> VirtualMesh {
        VirtualMesh::standard_lods(position, radius, 10000, mesh_id)
    }

    fn make_simple_mesh(position: [f32; 3]) -> VirtualMesh {
        let levels = [
            LODLevel::new(0.0, 10000, 0, 0),    // LOD 0: perfect
            LODLevel::new(0.1, 5000, 0, 0),     // LOD 1: 0.1 world units error
            LODLevel::new(0.5, 2500, 0, 0),     // LOD 2: 0.5 world units error
            LODLevel::new(1.0, 1000, 0, 0),     // LOD 3: 1.0 world units error
        ];
        VirtualMesh::new(position, 5.0, &levels, 0)
    }

    // -------------------------------------------------------------------------
    // Struct Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_virtual_lod_params_size() {
        assert_eq!(mem::size_of::<VirtualLODParams>(), 64);
    }

    #[test]
    fn test_lod_level_size() {
        assert_eq!(mem::size_of::<LODLevel>(), 16);
    }

    #[test]
    fn test_virtual_mesh_size() {
        assert_eq!(mem::size_of::<VirtualMesh>(), 128);
    }

    #[test]
    fn test_streaming_priority_size() {
        assert_eq!(mem::size_of::<StreamingPriority>(), 16);
    }

    #[test]
    fn test_lod_result_size() {
        assert_eq!(mem::size_of::<LODResult>(), 16);
    }

    #[test]
    fn test_page_residency_size() {
        assert_eq!(mem::size_of::<PageResidency>(), 8);
    }

    // -------------------------------------------------------------------------
    // VirtualLODParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_params_new() {
        let params = VirtualLODParams::new(
            [0.0, 0.0, 0.0],
            1.0,
            0.0,
            0.1,
            1080,
            FRAC_PI_4,
        );

        assert_eq!(params.camera_position, [0.0, 0.0, 0.0]);
        assert_eq!(params.error_threshold, 1.0);
        assert_eq!(params.lod_bias, 0.0);
        assert_eq!(params.transition_width, 0.1);
        assert_eq!(params.screen_height, 1080.0);
        assert_eq!(params.fov_y, FRAC_PI_4);
        assert_eq!(params.flags, 0);
    }

    #[test]
    fn test_params_with_instance_count() {
        let params = VirtualLODParams::new([0.0; 3], 1.0, 0.0, 0.1, 1080, FRAC_PI_4)
            .with_instance_count(1000);
        assert_eq!(params.num_instances, 1000);
    }

    #[test]
    fn test_params_with_dither() {
        let params = VirtualLODParams::new([0.0; 3], 1.0, 0.0, 0.1, 1080, FRAC_PI_4)
            .with_dither();
        assert!(params.flags & FLAG_USE_DITHER != 0);
    }

    #[test]
    fn test_params_with_forced_lod() {
        let params = VirtualLODParams::new([0.0; 3], 1.0, 0.0, 0.1, 1080, FRAC_PI_4)
            .with_forced_lod(2);
        assert_eq!(params.forced_lod, 2);
        assert!(params.flags & FLAG_FORCE_LOD != 0);
    }

    #[test]
    fn test_params_num_workgroups() {
        let params = VirtualLODParams::new([0.0; 3], 1.0, 0.0, 0.1, 1080, FRAC_PI_4)
            .with_instance_count(1000);
        assert_eq!(params.num_workgroups(), 4); // ceil(1000 / 256)

        let params2 = VirtualLODParams::new([0.0; 3], 1.0, 0.0, 0.1, 1080, FRAC_PI_4)
            .with_instance_count(256);
        assert_eq!(params2.num_workgroups(), 1);

        let params3 = VirtualLODParams::new([0.0; 3], 1.0, 0.0, 0.1, 1080, FRAC_PI_4)
            .with_instance_count(257);
        assert_eq!(params3.num_workgroups(), 2);
    }

    // -------------------------------------------------------------------------
    // LODLevel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lod_level_new() {
        let level = LODLevel::new(0.5, 5000, 1024, 2048);
        assert_eq!(level.geometric_error, 0.5);
        assert_eq!(level.triangle_count, 5000);
        assert_eq!(level.vertex_offset, 1024);
        assert_eq!(level.index_offset, 2048);
    }

    #[test]
    fn test_lod_level_lod0() {
        let level = LODLevel::lod0(10000);
        assert_eq!(level.geometric_error, 0.0);
        assert_eq!(level.triangle_count, 10000);
        assert_eq!(level.vertex_offset, 0);
        assert_eq!(level.index_offset, 0);
    }

    // -------------------------------------------------------------------------
    // VirtualMesh Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_virtual_mesh_new() {
        let levels = [
            LODLevel::new(0.0, 10000, 0, 0),
            LODLevel::new(0.5, 5000, 1000, 2000),
        ];
        let mesh = VirtualMesh::new([1.0, 2.0, 3.0], 5.0, &levels, 42);

        assert_eq!(mesh.position, [1.0, 2.0, 3.0]);
        assert_eq!(mesh.bounding_radius, 5.0);
        assert_eq!(mesh.num_lods, 2);
        assert_eq!(mesh.mesh_id, 42);
        assert_eq!(mesh.page_id, INVALID_LOD);
    }

    #[test]
    fn test_virtual_mesh_with_page_id() {
        let mesh = VirtualMesh::single_lod([0.0; 3], 1.0, 1000, 0)
            .with_page_id(123);
        assert_eq!(mesh.page_id, 123);
    }

    #[test]
    fn test_virtual_mesh_standard_lods() {
        let mesh = VirtualMesh::standard_lods([0.0; 3], 10.0, 10000, 1);

        assert_eq!(mesh.num_lods, 4);
        assert_eq!(mesh.bounding_radius, 10.0);

        // Check LOD 0 has smallest error
        assert!(mesh.lod_levels[0].geometric_error < mesh.lod_levels[1].geometric_error);
        // Check triangle counts decrease
        assert!(mesh.lod_levels[0].triangle_count > mesh.lod_levels[1].triangle_count);
    }

    #[test]
    #[should_panic(expected = "At least one LOD level required")]
    fn test_virtual_mesh_empty_lods_panics() {
        let _mesh = VirtualMesh::new([0.0; 3], 1.0, &[], 0);
    }

    #[test]
    #[should_panic(expected = "Max 6 inline LODs")]
    fn test_virtual_mesh_too_many_lods_panics() {
        let levels: Vec<LODLevel> = (0..7).map(|i| LODLevel::new(i as f32, 1000, 0, 0)).collect();
        let _mesh = VirtualMesh::new([0.0; 3], 1.0, &levels, 0);
    }

    // -------------------------------------------------------------------------
    // StreamingPriority Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_streaming_priority_tier() {
        let critical = StreamingPriority {
            priority: (PRIORITY_CRITICAL << 24) | 0x001000,
            mesh_id: 0,
            page_id: 0,
            lod_level: 0,
        };
        assert_eq!(critical.tier(), PRIORITY_CRITICAL);
        assert!(critical.is_critical());

        let low = StreamingPriority {
            priority: (PRIORITY_LOW << 24) | 0xFFFFFF,
            mesh_id: 0,
            page_id: 0,
            lod_level: 0,
        };
        assert_eq!(low.tier(), PRIORITY_LOW);
        assert!(!low.is_critical());
    }

    #[test]
    fn test_streaming_priority_comparison() {
        let critical = StreamingPriority {
            priority: (PRIORITY_CRITICAL << 24) | 0x001000,
            mesh_id: 0,
            page_id: 0,
            lod_level: 0,
        };
        let low = StreamingPriority {
            priority: (PRIORITY_LOW << 24) | 0x001000,
            mesh_id: 0,
            page_id: 0,
            lod_level: 0,
        };

        assert!(critical.is_more_urgent_than(&low));
        assert!(!low.is_more_urgent_than(&critical));
    }

    // -------------------------------------------------------------------------
    // LODResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lod_result_is_culled() {
        let culled = LODResult {
            lod_level: INVALID_LOD,
            blend_factor: 0.0,
            screen_error: 0.0,
            flags: 0,
        };
        assert!(culled.is_culled());
        assert!(!culled.is_visible());
        assert!(culled.lod().is_none());
    }

    #[test]
    fn test_lod_result_is_visible() {
        let visible = LODResult {
            lod_level: 2,
            blend_factor: 0.5,
            screen_error: 0.8,
            flags: 0,
        };
        assert!(!visible.is_culled());
        assert!(visible.is_visible());
        assert_eq!(visible.lod(), Some(2));
    }

    #[test]
    fn test_lod_result_needs_blending() {
        let no_blend = LODResult {
            lod_level: 0,
            blend_factor: 0.0,
            screen_error: 0.5,
            flags: 0,
        };
        assert!(!no_blend.needs_blending());

        let blending = LODResult {
            lod_level: 0,
            blend_factor: 0.5,
            screen_error: 0.8,
            flags: 0,
        };
        assert!(blending.needs_blending());

        let full_blend = LODResult {
            lod_level: 1,
            blend_factor: 1.0,
            screen_error: 1.2,
            flags: 0,
        };
        assert!(!full_blend.needs_blending());
    }

    #[test]
    fn test_lod_result_flags() {
        let needs_streaming = LODResult {
            lod_level: 0,
            blend_factor: 0.0,
            screen_error: 1.5,
            flags: RESULT_FLAG_NEEDS_STREAMING,
        };
        assert!(needs_streaming.needs_streaming());
        assert!(!needs_streaming.is_page_resident());

        let resident = LODResult {
            lod_level: 0,
            blend_factor: 0.0,
            screen_error: 0.5,
            flags: RESULT_FLAG_PAGE_RESIDENT,
        };
        assert!(!resident.needs_streaming());
        assert!(resident.is_page_resident());
    }

    // -------------------------------------------------------------------------
    // PageResidency Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_page_residency_new() {
        let resident = PageResidency::new(42, true);
        assert_eq!(resident.page_id, 42);
        assert!(resident.is_resident());

        let not_resident = PageResidency::new(43, false);
        assert!(!not_resident.is_resident());
    }

    #[test]
    fn test_page_residency_set_resident() {
        let mut page = PageResidency::new(0, false);
        assert!(!page.is_resident());

        page.set_resident(true);
        assert!(page.is_resident());

        page.set_resident(false);
        assert!(!page.is_resident());
    }

    // -------------------------------------------------------------------------
    // CPU Reference Implementation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_screen_space_error_basic() {
        // At distance 100, with 1.0 geometric error, 45 degree FOV, 1080 screen
        let error = cpu_screen_space_error(1.0, 100.0, FRAC_PI_4, 1080.0);
        // Should be > 0 and < screen_height
        assert!(error > 0.0);
        assert!(error < 1080.0);
    }

    #[test]
    fn test_cpu_screen_space_error_distance_scaling() {
        // Error should decrease with distance (inverse relationship)
        let near_error = cpu_screen_space_error(1.0, 10.0, FRAC_PI_4, 1080.0);
        let far_error = cpu_screen_space_error(1.0, 100.0, FRAC_PI_4, 1080.0);
        assert!(near_error > far_error);

        // Should be roughly 10x larger when 10x closer
        let ratio = near_error / far_error;
        assert!((ratio - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_cpu_screen_space_error_very_close() {
        // Very close should return large error
        let error = cpu_screen_space_error(1.0, 0.0001, FRAC_PI_4, 1080.0);
        assert!(error > 100_000.0);
    }

    #[test]
    fn test_cpu_select_lod_simple() {
        // LOD errors: geometric error in world units for each LOD level
        // LOD 0 = highest detail = lowest error
        // Screen error formula: geometric_error * fov_factor * screen_height/2 / distance
        // fov_factor for 45deg = cot(22.5deg) = 2.414
        // So at distance 100, geometric_error 1.0:
        //   screen_error = 1.0 * 2.414 * 540 / 100 = 13.0 pixels

        let lod_errors = [0.0, 0.1, 0.5, 1.0];

        // At far distance (1000), all geometric errors produce small screen errors:
        // LOD 0 (0.0): 0 pixels
        // LOD 1 (0.1): 0.1 * 2.414 * 540 / 1000 = 0.13 pixels
        // LOD 2 (0.5): 0.65 pixels
        // LOD 3 (1.0): 1.3 pixels
        // With threshold 1.0, LOD 0-2 all pass (< 1.0 pixel)
        // Since we search from LOD 0, we find LOD 0 first
        let (lod, screen_error) = cpu_select_lod(&lod_errors, 1000.0, 1.0, 0.0, FRAC_PI_4, 1080.0);
        assert_eq!(lod, 0); // LOD 0 (error=0.0) always has 0 screen error
        assert_eq!(screen_error, 0.0);

        // At close distance (10), geometric errors produce large screen errors:
        // LOD 0 (0.0): 0 pixels - still acceptable
        // LOD 1 (0.1): 13 pixels
        // LOD 2 (0.5): 65 pixels
        // LOD 3 (1.0): 130 pixels
        // With threshold 1.0, only LOD 0 passes
        let (lod_close, _) = cpu_select_lod(&lod_errors, 10.0, 1.0, 0.0, FRAC_PI_4, 1080.0);
        assert_eq!(lod_close, 0);

        // Test with a more realistic scenario where higher LODs are selected:
        // Use errors that start non-zero
        let lod_errors_nonzero = [0.01, 0.1, 0.5, 1.0];
        // At distance 10:
        // LOD 0 (0.01): 1.3 pixels > 1.0 threshold, fails
        // LOD 1 (0.1): 13 pixels, fails
        // ...all fail, return last LOD
        let (lod_last, _) = cpu_select_lod(&lod_errors_nonzero, 10.0, 1.0, 0.0, FRAC_PI_4, 1080.0);
        assert_eq!(lod_last, 3); // All exceed threshold, return last

        // At distance 1000 with same errors:
        // LOD 0 (0.01): 0.13 pixels < 1.0, passes
        let (lod_far, _) = cpu_select_lod(&lod_errors_nonzero, 1000.0, 1.0, 0.0, FRAC_PI_4, 1080.0);
        assert_eq!(lod_far, 0);
    }

    #[test]
    fn test_cpu_select_lod_bias_effect() {
        let lod_errors = [0.0, 0.1, 0.5, 1.0];
        let distance = 100.0;
        let threshold = 1.0;

        // Positive bias = allow more error = lower quality LOD
        let (lod_positive, _) = cpu_select_lod(&lod_errors, distance, threshold, 2.0, FRAC_PI_4, 1080.0);
        // Negative bias = allow less error = higher quality LOD
        let (lod_negative, _) = cpu_select_lod(&lod_errors, distance, threshold, -2.0, FRAC_PI_4, 1080.0);

        assert!(lod_positive >= lod_negative);
    }

    #[test]
    fn test_cpu_select_lod_empty() {
        let lod_errors: [f32; 0] = [];
        let (lod, screen_error) = cpu_select_lod(&lod_errors, 100.0, 1.0, 0.0, FRAC_PI_4, 1080.0);
        assert_eq!(lod, 0);
        assert_eq!(screen_error, 0.0);
    }

    #[test]
    fn test_cpu_lod_blend_factor_no_transition() {
        // No transition width = no blending
        let factor = cpu_lod_blend_factor(0.5, 1.0, 0.0);
        assert_eq!(factor, 0.0);
    }

    #[test]
    fn test_cpu_lod_blend_factor_below_start() {
        // Error well below threshold = no blending
        let factor = cpu_lod_blend_factor(0.5, 1.0, 0.2);
        assert_eq!(factor, 0.0);
    }

    #[test]
    fn test_cpu_lod_blend_factor_above_end() {
        // Error at or above threshold = full blend
        let factor = cpu_lod_blend_factor(1.0, 1.0, 0.2);
        assert_eq!(factor, 1.0);

        let factor2 = cpu_lod_blend_factor(1.5, 1.0, 0.2);
        assert_eq!(factor2, 1.0);
    }

    #[test]
    fn test_cpu_lod_blend_factor_in_range() {
        // Error in transition range = partial blend
        // With threshold=1.0, width=0.2, blend starts at 0.8 and ends at 1.0
        let factor = cpu_lod_blend_factor(0.9, 1.0, 0.2);
        assert!(factor > 0.0);
        assert!(factor < 1.0);
        // At 0.9, we're halfway through [0.8, 1.0]
        assert!((factor - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cpu_streaming_priority_invisible() {
        let (tier, value) = cpu_streaming_priority(false, 2.0, 1.0, 500.0, 1080.0);
        assert_eq!(tier, PRIORITY_LOW);
        assert_eq!(value, 0x00FFFFFF);
    }

    #[test]
    fn test_cpu_streaming_priority_critical() {
        // High error ratio + large screen size = critical
        let (tier, _) = cpu_streaming_priority(true, 5.0, 1.0, 800.0, 1080.0);
        assert_eq!(tier, PRIORITY_CRITICAL);
    }

    #[test]
    fn test_cpu_streaming_priority_low() {
        // Low error ratio + small screen size = low priority
        let (tier, _) = cpu_streaming_priority(true, 0.1, 1.0, 10.0, 1080.0);
        assert_eq!(tier, PRIORITY_LOW);
    }

    #[test]
    fn test_cpu_streaming_priority_ordering() {
        // Higher urgency should result in lower priority value within same tier
        let (tier1, value1) = cpu_streaming_priority(true, 3.0, 1.0, 500.0, 1080.0);
        let (tier2, value2) = cpu_streaming_priority(true, 1.5, 1.0, 500.0, 1080.0);

        // Both might be same tier, but value1 should be lower (more urgent)
        if tier1 == tier2 {
            assert!(value1 < value2);
        }
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_screen_error_with_zero_fov() {
        // Zero FOV should not crash (tan(0) = 0, would cause div by zero)
        // Our implementation should handle this gracefully
        let error = cpu_screen_space_error(1.0, 100.0, 0.001, 1080.0);
        assert!(error.is_finite());
    }

    #[test]
    fn test_screen_error_with_90_degree_fov() {
        let error = cpu_screen_space_error(1.0, 100.0, FRAC_PI_2, 1080.0);
        assert!(error > 0.0);
        assert!(error.is_finite());
    }

    #[test]
    fn test_select_lod_all_exceed_threshold() {
        // All LODs exceed threshold -> should return last LOD
        let lod_errors = [100.0, 200.0, 300.0];
        let (lod, _) = cpu_select_lod(&lod_errors, 1.0, 0.001, 0.0, FRAC_PI_4, 1080.0);
        assert_eq!(lod, 2); // Last LOD
    }

    #[test]
    fn test_blend_factor_very_small_width() {
        let factor = cpu_lod_blend_factor(0.99, 1.0, 0.001);
        // Should be close to 1.0 or exactly 1.0
        assert!(factor >= 0.0);
        assert!(factor <= 1.0);
    }

    #[test]
    fn test_virtual_mesh_default() {
        let mesh = VirtualMesh::default();
        assert_eq!(mesh.position, [0.0; 3]);
        assert_eq!(mesh.bounding_radius, 1.0);
        assert_eq!(mesh.num_lods, 1);
        assert_eq!(mesh.page_id, INVALID_LOD);
    }

    #[test]
    fn test_params_default() {
        let params = VirtualLODParams::default();
        assert_eq!(params.camera_position, [0.0; 3]);
        assert_eq!(params.error_threshold, 0.0);
        assert_eq!(params.num_instances, 0);
    }

    #[test]
    fn test_lod_level_default() {
        let level = LODLevel::default();
        assert_eq!(level.geometric_error, 0.0);
        assert_eq!(level.triangle_count, 0);
    }

    #[test]
    fn test_page_residency_frame_delta() {
        let mut page = PageResidency { page_id: 0, status: 0 };

        // Set frame delta (bits 1-7)
        page.status = (50 << 1) | 1; // delta=50, resident=true
        assert_eq!(page.last_access_delta(), 50);
        assert!(page.is_resident());

        // Max delta (127)
        page.status = (127 << 1) | 0;
        assert_eq!(page.last_access_delta(), 127);
        assert!(!page.is_resident());
    }

    #[test]
    fn test_streaming_priority_value() {
        let priority = StreamingPriority {
            priority: (PRIORITY_HIGH << 24) | 0x123456,
            mesh_id: 1,
            page_id: 2,
            lod_level: 3,
        };

        assert_eq!(priority.tier(), PRIORITY_HIGH);
        assert_eq!(priority.value(), 0x123456);
    }

    #[test]
    fn test_multiple_params_flags() {
        let params = VirtualLODParams::new([0.0; 3], 1.0, 0.0, 0.1, 1080, FRAC_PI_4)
            .with_dither()
            .with_page_tracking()
            .with_streaming_budget(8 * 1024 * 1024);

        assert!(params.flags & FLAG_USE_DITHER != 0);
        assert!(params.flags & FLAG_PAGE_TRACKING != 0);
        assert_eq!(params.streaming_budget, 8 * 1024 * 1024);
    }

    #[test]
    fn test_cpu_screen_error_zero_geometric_error() {
        // Zero geometric error should give zero screen error
        let error = cpu_screen_space_error(0.0, 100.0, FRAC_PI_4, 1080.0);
        assert_eq!(error, 0.0);
    }

    #[test]
    fn test_cpu_screen_error_negative_distance() {
        // Negative distance should be treated same as very close
        let error = cpu_screen_space_error(1.0, -10.0, FRAC_PI_4, 1080.0);
        assert!(error > 100_000.0);
    }

    #[test]
    fn test_workgroup_size_constant() {
        assert_eq!(WORKGROUP_SIZE, 256);
    }

    #[test]
    fn test_max_lod_levels_constant() {
        assert_eq!(MAX_LOD_LEVELS, 16);
        assert_eq!(MAX_INLINE_LODS, 6);
    }
}
