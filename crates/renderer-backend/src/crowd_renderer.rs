//! GPU Instanced Crowd Renderer (T-AN-8.2)
//!
//! This module provides efficient GPU-driven instanced rendering for crowd/character
//! rendering. It handles per-instance data storage, frustum culling, LOD selection,
//! animation time offsets, and indirect draw command generation.
//!
//! # Architecture
//!
//! - `CrowdInstance`: Per-instance GPU data (position, rotation, scale, animation, LOD)
//! - `CrowdBatch`: Collection of instances sharing the same mesh
//! - `CrowdRenderConfig`: Configuration for LOD distances, culling, etc.
//! - `CrowdRenderer`: Main renderer managing batches and GPU buffers
//!
//! # GPU Memory Layout
//!
//! The `CrowdInstance` struct is `repr(C)` with 64 bytes total, aligned for
//! efficient GPU access:
//!
//! | Offset | Field          | Size     |
//! |--------|----------------|----------|
//! | 0      | position       | 12 bytes |
//! | 12     | scale          | 4 bytes  |
//! | 16     | rotation       | 16 bytes |
//! | 32     | animation_id   | 2 bytes  |
//! | 34     | lod_level      | 1 byte   |
//! | 35     | flags          | 1 byte   |
//! | 36     | animation_time | 4 bytes  |
//! | 40     | _padding       | 24 bytes |
//!
//! # Per-Instance Animation
//!
//! Each instance can have:
//! - Unique animation ID (selects which animation to play)
//! - Animation time offset (phase shift for variety)
//! - Phase synchronization flag (for coordinated crowds)
//!
//! # LOD System
//!
//! Three LOD levels are supported:
//! - **LOD 0**: Full skeleton with all bones
//! - **LOD 1**: Simplified skeleton (major bones only)
//! - **LOD 2**: Billboard impostor (single quad)
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::crowd_renderer::{CrowdRenderer, CrowdRenderConfig, CrowdInstance, CrowdBatch};
//! use renderer_backend::animation_textures::AnimationTextureAtlas;
//!
//! // Create renderer with LOD distances
//! let config = CrowdRenderConfig::new()
//!     .with_lod_distances([20.0, 50.0, 100.0])
//!     .with_max_instances(10000);
//! let mut renderer = CrowdRenderer::new(config);
//!
//! // Add a batch for a character mesh
//! let batch_id = renderer.add_batch(mesh_id);
//!
//! // Add instances
//! let instance = CrowdInstance::new([10.0, 0.0, 5.0])
//!     .with_animation(0, 0.5)  // Animation 0, phase offset 0.5
//!     .with_scale(1.0);
//! renderer.add_instance(batch_id, instance);
//!
//! // Update LODs based on camera position
//! renderer.update_lods([0.0, 5.0, 0.0]);
//!
//! // Cull against frustum
//! renderer.cull_frustum(&frustum);
//!
//! // Get draw commands for rendering
//! let commands = renderer.get_draw_commands();
//! ```

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum instances for the crowd renderer.
pub const DEFAULT_MAX_INSTANCES: u32 = 10_000;

/// Size of CrowdInstance in bytes (64 bytes for GPU alignment).
pub const CROWD_INSTANCE_SIZE: usize = 64;

/// Maximum LOD levels supported.
pub const MAX_LOD_LEVELS: usize = 3;

/// LOD level for full skeleton.
pub const LOD_FULL_SKELETON: u8 = 0;

/// LOD level for simplified skeleton.
pub const LOD_SIMPLIFIED: u8 = 1;

/// LOD level for impostor/billboard.
pub const LOD_IMPOSTOR: u8 = 2;

/// Instance flag: Visible (passes frustum culling).
pub const FLAG_VISIBLE: u8 = 0x01;

/// Instance flag: Phase synchronized with group.
pub const FLAG_PHASE_SYNC: u8 = 0x02;

/// Instance flag: Cast shadows.
pub const FLAG_CAST_SHADOW: u8 = 0x04;

/// Instance flag: Receive shadows.
pub const FLAG_RECEIVE_SHADOW: u8 = 0x08;

/// Default buffer grow factor when resizing.
pub const BUFFER_GROW_FACTOR: f32 = 1.5;

// ---------------------------------------------------------------------------
// CrowdError
// ---------------------------------------------------------------------------

/// Errors that can occur during crowd rendering operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CrowdError {
    /// Buffer capacity exceeded and cannot grow.
    BufferFull { max: u32 },
    /// Instance data contains invalid values (NaN, Inf).
    InvalidInstance { reason: &'static str },
    /// Invalid LOD threshold configuration.
    InvalidLodDistances { reason: &'static str },
    /// Batch index out of bounds.
    BatchNotFound { batch_id: usize },
    /// Instance index out of bounds.
    InstanceNotFound { batch_id: usize, instance_id: usize },
}

impl std::fmt::Display for CrowdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferFull { max } => {
                write!(f, "crowd buffer full: max capacity is {}", max)
            }
            Self::InvalidInstance { reason } => {
                write!(f, "invalid crowd instance: {}", reason)
            }
            Self::InvalidLodDistances { reason } => {
                write!(f, "invalid LOD distances: {}", reason)
            }
            Self::BatchNotFound { batch_id } => {
                write!(f, "batch {} not found", batch_id)
            }
            Self::InstanceNotFound { batch_id, instance_id } => {
                write!(f, "instance {} not found in batch {}", instance_id, batch_id)
            }
        }
    }
}

impl std::error::Error for CrowdError {}

// ---------------------------------------------------------------------------
// Vec3 / Quat helper types (inline to avoid external dependencies)
// ---------------------------------------------------------------------------

/// 3D vector type for crowd rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Create a new Vec3.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Calculate squared distance to another point.
    #[inline]
    pub fn distance_squared(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Calculate distance to another point.
    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        self.distance_squared(other).sqrt()
    }

    /// Convert to array.
    #[inline]
    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Create from array.
    #[inline]
    pub fn from_array(arr: [f32; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    /// Check if all components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(arr: [f32; 3]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Vec3> for [f32; 3] {
    fn from(v: Vec3) -> Self {
        v.to_array()
    }
}

/// Quaternion type for crowd instance rotation.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    /// Create a new quaternion.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Identity quaternion.
    pub const IDENTITY: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    /// Convert to array [x, y, z, w].
    #[inline]
    pub fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Create from array [x, y, z, w].
    #[inline]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }

    /// Create rotation from Y-axis angle (radians).
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let half = angle * 0.5;
        Self::new(0.0, half.sin(), 0.0, half.cos())
    }

    /// Normalize the quaternion.
    #[inline]
    pub fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if len > 0.0001 {
            Self::new(self.x / len, self.y / len, self.z / len, self.w / len)
        } else {
            Self::IDENTITY
        }
    }

    /// Check if all components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<[f32; 4]> for Quat {
    fn from(arr: [f32; 4]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Quat> for [f32; 4] {
    fn from(q: Quat) -> Self {
        q.to_array()
    }
}

// ---------------------------------------------------------------------------
// Frustum
// ---------------------------------------------------------------------------

/// Frustum for culling, defined by 6 planes.
///
/// Each plane is stored as [A, B, C, D] where Ax + By + Cz + D >= 0 is inside.
#[derive(Clone, Copy, Debug, Default)]
pub struct Frustum {
    /// Six frustum planes: left, right, bottom, top, near, far.
    pub planes: [[f32; 4]; 6],
}

impl Frustum {
    /// Create a new frustum from 6 planes.
    pub fn new(planes: [[f32; 4]; 6]) -> Self {
        Self { planes }
    }

    /// Create an empty frustum that accepts everything.
    pub fn unbounded() -> Self {
        Self {
            planes: [[0.0, 0.0, 0.0, f32::MAX]; 6],
        }
    }

    /// Create a simple box frustum from bounds.
    pub fn from_bounds(min: Vec3, max: Vec3) -> Self {
        Self {
            planes: [
                [1.0, 0.0, 0.0, -min.x],   // Left: x >= min.x
                [-1.0, 0.0, 0.0, max.x],   // Right: x <= max.x
                [0.0, 1.0, 0.0, -min.y],   // Bottom: y >= min.y
                [0.0, -1.0, 0.0, max.y],   // Top: y <= max.y
                [0.0, 0.0, 1.0, -min.z],   // Near: z >= min.z
                [0.0, 0.0, -1.0, max.z],   // Far: z <= max.z
            ],
        }
    }

    /// Test if a sphere is at least partially inside the frustum.
    #[inline]
    pub fn contains_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let distance = plane[0] * center.x + plane[1] * center.y + plane[2] * center.z + plane[3];
            if distance < -radius {
                return false;
            }
        }
        true
    }

    /// Test if a point is inside the frustum.
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        self.contains_sphere(point, 0.0)
    }
}

// ---------------------------------------------------------------------------
// BufferHandle / GpuBuffer (placeholder types)
// ---------------------------------------------------------------------------

/// Handle to a GPU buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u32);

impl BufferHandle {
    /// Create a new buffer handle.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Invalid/null buffer handle.
    pub const INVALID: Self = Self(u32::MAX);

    /// Check if this is a valid handle.
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

/// Prepared GPU buffer data.
#[derive(Clone, Debug)]
pub struct GpuBuffer {
    /// Buffer handle.
    pub handle: BufferHandle,
    /// Buffer data as bytes.
    pub data: Vec<u8>,
    /// Buffer size in bytes.
    pub size: usize,
    /// Buffer usage flags.
    pub usage: GpuBufferUsage,
}

impl GpuBuffer {
    /// Create a new GPU buffer.
    pub fn new(handle: BufferHandle, data: Vec<u8>, usage: GpuBufferUsage) -> Self {
        let size = data.len();
        Self { handle, data, size, usage }
    }

    /// Create an empty buffer with a given size.
    pub fn empty(handle: BufferHandle, size: usize, usage: GpuBufferUsage) -> Self {
        Self {
            handle,
            data: vec![0u8; size],
            size,
            usage,
        }
    }
}

/// GPU buffer usage flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBufferUsage(pub u32);

impl GpuBufferUsage {
    /// Vertex buffer usage.
    pub const VERTEX: Self = Self(0x01);
    /// Storage buffer usage.
    pub const STORAGE: Self = Self(0x02);
    /// Indirect buffer usage.
    pub const INDIRECT: Self = Self(0x04);
    /// Copy destination usage.
    pub const COPY_DST: Self = Self(0x08);
}

// ---------------------------------------------------------------------------
// AnimationTextureAtlas (reference type)
// ---------------------------------------------------------------------------

/// Reference to an animation texture atlas (from animation_textures module).
///
/// This is a simplified reference type; the actual atlas is stored elsewhere.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnimationTextureAtlas {
    /// Atlas handle/identifier.
    pub handle: u32,
    /// Number of clips in the atlas.
    pub clip_count: u32,
    /// Texture width (frames).
    pub texture_width: u32,
    /// Texture height (bones * 2).
    pub texture_height: u32,
}

impl AnimationTextureAtlas {
    /// Create a new atlas reference.
    pub fn new(handle: u32, clip_count: u32, width: u32, height: u32) -> Self {
        Self {
            handle,
            clip_count,
            texture_width: width,
            texture_height: height,
        }
    }
}

// ---------------------------------------------------------------------------
// CrowdInstance — GPU per-instance data
// ---------------------------------------------------------------------------

/// Per-instance data for GPU crowd rendering.
///
/// This struct is designed for efficient GPU upload with 64-byte alignment.
/// All fields are tightly packed with explicit padding for std430 compatibility.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CrowdInstance {
    /// World-space position.
    pub position: Vec3,
    /// Uniform scale factor.
    pub scale: f32,
    /// Rotation quaternion.
    pub rotation: Quat,
    /// Animation clip ID (index into texture atlas).
    pub animation_id: u16,
    /// Current LOD level (0=full, 1=simplified, 2=impostor).
    pub lod_level: u8,
    /// Instance flags (visibility, shadow, sync).
    pub flags: u8,
    /// Animation time/phase offset (seconds).
    pub animation_time: f32,
    /// Padding for 64-byte alignment.
    pub _padding: [f32; 6],
}

impl CrowdInstance {
    /// Create a new crowd instance at the given position.
    ///
    /// Rotation is identity, scale is 1.0, animation is 0.
    #[inline]
    pub fn new(position: impl Into<Vec3>) -> Self {
        Self {
            position: position.into(),
            scale: 1.0,
            rotation: Quat::IDENTITY,
            animation_id: 0,
            lod_level: LOD_FULL_SKELETON,
            flags: FLAG_VISIBLE | FLAG_CAST_SHADOW | FLAG_RECEIVE_SHADOW,
            animation_time: 0.0,
            _padding: [0.0; 6],
        }
    }

    /// Create an instance with full parameters.
    #[inline]
    pub fn with_full_params(
        position: impl Into<Vec3>,
        rotation: impl Into<Quat>,
        scale: f32,
        animation_id: u16,
        animation_time: f32,
        lod_level: u8,
        flags: u8,
    ) -> Self {
        Self {
            position: position.into(),
            scale,
            rotation: rotation.into(),
            animation_id,
            lod_level,
            flags,
            animation_time,
            _padding: [0.0; 6],
        }
    }

    /// Set the rotation.
    #[inline]
    pub fn with_rotation(mut self, rotation: impl Into<Quat>) -> Self {
        self.rotation = rotation.into();
        self
    }

    /// Set the uniform scale.
    #[inline]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set the animation parameters.
    #[inline]
    pub fn with_animation(mut self, animation_id: u16, time_offset: f32) -> Self {
        self.animation_id = animation_id;
        self.animation_time = time_offset;
        self
    }

    /// Set the LOD level.
    #[inline]
    pub fn with_lod(mut self, lod_level: u8) -> Self {
        self.lod_level = lod_level.min(LOD_IMPOSTOR);
        self
    }

    /// Set instance flags.
    #[inline]
    pub fn with_flags(mut self, flags: u8) -> Self {
        self.flags = flags;
        self
    }

    /// Enable phase synchronization.
    #[inline]
    pub fn with_phase_sync(mut self, sync: bool) -> Self {
        if sync {
            self.flags |= FLAG_PHASE_SYNC;
        } else {
            self.flags &= !FLAG_PHASE_SYNC;
        }
        self
    }

    /// Check if the instance is visible.
    #[inline]
    pub fn is_visible(&self) -> bool {
        (self.flags & FLAG_VISIBLE) != 0
    }

    /// Set visibility flag.
    #[inline]
    pub fn set_visible(&mut self, visible: bool) {
        if visible {
            self.flags |= FLAG_VISIBLE;
        } else {
            self.flags &= !FLAG_VISIBLE;
        }
    }

    /// Check if phase synchronization is enabled.
    #[inline]
    pub fn is_phase_synced(&self) -> bool {
        (self.flags & FLAG_PHASE_SYNC) != 0
    }

    /// Validate that all fields contain finite values.
    pub fn validate(&self) -> Result<(), CrowdError> {
        if !self.position.is_finite() {
            return Err(CrowdError::InvalidInstance {
                reason: "position contains non-finite values",
            });
        }

        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(CrowdError::InvalidInstance {
                reason: "scale must be positive and finite",
            });
        }

        if !self.rotation.is_finite() {
            return Err(CrowdError::InvalidInstance {
                reason: "rotation quaternion contains non-finite values",
            });
        }

        if !self.animation_time.is_finite() {
            return Err(CrowdError::InvalidInstance {
                reason: "animation_time is not finite",
            });
        }

        Ok(())
    }

    /// Calculate squared distance to a point.
    #[inline]
    pub fn distance_squared(&self, point: Vec3) -> f32 {
        self.position.distance_squared(point)
    }

    /// Calculate distance to a point.
    #[inline]
    pub fn distance(&self, point: Vec3) -> f32 {
        self.position.distance(point)
    }

    /// Compute bounding sphere radius for frustum culling.
    ///
    /// Assumes a unit sphere at the origin scaled by `scale`.
    #[inline]
    pub fn bounding_radius(&self) -> f32 {
        // Character height is typically ~2 units, so radius is ~1 * scale
        self.scale
    }
}

impl Default for CrowdInstance {
    fn default() -> Self {
        Self::new(Vec3::ZERO)
    }
}

// ---------------------------------------------------------------------------
// CrowdBatch
// ---------------------------------------------------------------------------

/// A batch of crowd instances sharing the same mesh.
#[derive(Clone, Debug)]
pub struct CrowdBatch {
    /// Mesh ID for this batch.
    pub mesh_id: usize,
    /// All instances in this batch.
    pub instances: Vec<CrowdInstance>,
    /// GPU instance buffer handle (if uploaded).
    pub instance_buffer: Option<BufferHandle>,
    /// GPU indirect draw buffer handle (if uploaded).
    pub indirect_buffer: Option<BufferHandle>,
    /// Number of visible instances (after culling).
    visible_count: usize,
    /// Dirty flag for GPU upload.
    dirty: bool,
}

impl CrowdBatch {
    /// Create a new batch for a mesh.
    pub fn new(mesh_id: usize) -> Self {
        Self {
            mesh_id,
            instances: Vec::new(),
            instance_buffer: None,
            indirect_buffer: None,
            visible_count: 0,
            dirty: false,
        }
    }

    /// Create a batch with pre-allocated capacity.
    pub fn with_capacity(mesh_id: usize, capacity: usize) -> Self {
        Self {
            mesh_id,
            instances: Vec::with_capacity(capacity),
            instance_buffer: None,
            indirect_buffer: None,
            visible_count: 0,
            dirty: false,
        }
    }

    /// Add an instance to this batch.
    pub fn add_instance(&mut self, instance: CrowdInstance) -> usize {
        let index = self.instances.len();
        self.instances.push(instance);
        self.dirty = true;
        index
    }

    /// Remove an instance by swapping with the last.
    ///
    /// Returns the index that was swapped into the removed slot, if any.
    pub fn remove_instance(&mut self, index: usize) -> Option<usize> {
        if index >= self.instances.len() {
            return None;
        }

        let last_index = self.instances.len() - 1;
        if index != last_index {
            self.instances.swap(index, last_index);
        }
        self.instances.pop();
        self.dirty = true;

        if index != last_index && !self.instances.is_empty() {
            Some(last_index)
        } else {
            None
        }
    }

    /// Update an existing instance.
    pub fn update_instance(&mut self, index: usize, instance: CrowdInstance) -> Result<(), CrowdError> {
        if index >= self.instances.len() {
            return Err(CrowdError::InstanceNotFound {
                batch_id: self.mesh_id,
                instance_id: index,
            });
        }
        self.instances[index] = instance;
        self.dirty = true;
        Ok(())
    }

    /// Get an instance by index.
    pub fn get(&self, index: usize) -> Option<&CrowdInstance> {
        self.instances.get(index)
    }

    /// Get a mutable reference to an instance.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut CrowdInstance> {
        self.dirty = true;
        self.instances.get_mut(index)
    }

    /// Get the number of instances in this batch.
    #[inline]
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Get the number of visible instances.
    #[inline]
    pub fn visible_count(&self) -> usize {
        self.visible_count
    }

    /// Check if the batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Check if the batch needs GPU upload.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the batch as clean after GPU upload.
    #[inline]
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }

    /// Clear all instances.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.visible_count = 0;
        self.dirty = true;
    }

    /// Update LOD levels based on camera position.
    pub fn update_lods(&mut self, camera_pos: Vec3, lod_distances: &[f32; 3]) {
        let lod_sq = [
            lod_distances[0] * lod_distances[0],
            lod_distances[1] * lod_distances[1],
            lod_distances[2] * lod_distances[2],
        ];

        for instance in &mut self.instances {
            let dist_sq = instance.distance_squared(camera_pos);
            instance.lod_level = if dist_sq < lod_sq[0] {
                LOD_FULL_SKELETON
            } else if dist_sq < lod_sq[1] {
                LOD_SIMPLIFIED
            } else if dist_sq < lod_sq[2] {
                LOD_IMPOSTOR
            } else {
                // Beyond impostor distance, still use impostor
                LOD_IMPOSTOR
            };
        }
        self.dirty = true;
    }

    /// Cull instances against a frustum.
    ///
    /// Updates visibility flags and returns the number of visible instances.
    pub fn cull_frustum(&mut self, frustum: &Frustum) -> usize {
        self.visible_count = 0;

        for instance in &mut self.instances {
            let visible = frustum.contains_sphere(instance.position, instance.bounding_radius());
            instance.set_visible(visible);
            if visible {
                self.visible_count += 1;
            }
        }

        self.dirty = true;
        self.visible_count
    }

    /// Get raw bytes for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.instances)
    }

    /// Get only visible instances.
    pub fn visible_instances(&self) -> impl Iterator<Item = &CrowdInstance> {
        self.instances.iter().filter(|i| i.is_visible())
    }

    /// Collect visible instances into a vector.
    pub fn collect_visible(&self) -> Vec<CrowdInstance> {
        self.instances.iter().filter(|i| i.is_visible()).copied().collect()
    }

    /// Get buffer size in bytes.
    pub fn buffer_size(&self) -> usize {
        self.instances.len() * CROWD_INSTANCE_SIZE
    }
}

// ---------------------------------------------------------------------------
// CrowdRenderConfig
// ---------------------------------------------------------------------------

/// Configuration for crowd rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct CrowdRenderConfig {
    /// Distance thresholds for LOD levels [full, simplified, impostor].
    /// Beyond lod_distances[2], instances are still rendered as impostors.
    pub lod_distances: [f32; 3],
    /// Enable frustum culling.
    pub frustum_cull: bool,
    /// Maximum instances across all batches.
    pub max_instances: u32,
    /// Reference to animation texture atlas.
    pub animation_texture: AnimationTextureAtlas,
    /// Enable smooth LOD transitions (requires shader support).
    pub smooth_lod_transitions: bool,
    /// LOD transition range (world units).
    pub lod_transition_range: f32,
    /// Validate instances on add.
    pub validate_on_add: bool,
}

impl CrowdRenderConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self {
            lod_distances: [20.0, 50.0, 100.0],
            frustum_cull: true,
            max_instances: DEFAULT_MAX_INSTANCES,
            animation_texture: AnimationTextureAtlas::default(),
            smooth_lod_transitions: false,
            lod_transition_range: 2.0,
            validate_on_add: false,
        }
    }

    /// Set LOD distance thresholds.
    pub fn with_lod_distances(mut self, distances: [f32; 3]) -> Self {
        self.lod_distances = distances;
        self
    }

    /// Set frustum culling enabled/disabled.
    pub fn with_frustum_cull(mut self, enable: bool) -> Self {
        self.frustum_cull = enable;
        self
    }

    /// Set maximum instances.
    pub fn with_max_instances(mut self, max: u32) -> Self {
        self.max_instances = max;
        self
    }

    /// Set animation texture atlas.
    pub fn with_animation_texture(mut self, atlas: AnimationTextureAtlas) -> Self {
        self.animation_texture = atlas;
        self
    }

    /// Enable smooth LOD transitions.
    pub fn with_smooth_lod_transitions(mut self, enable: bool) -> Self {
        self.smooth_lod_transitions = enable;
        self
    }

    /// Set LOD transition range.
    pub fn with_lod_transition_range(mut self, range: f32) -> Self {
        self.lod_transition_range = range;
        self
    }

    /// Enable validation on instance add.
    pub fn with_validation(mut self, enable: bool) -> Self {
        self.validate_on_add = enable;
        self
    }

    /// Validate LOD distances.
    pub fn validate(&self) -> Result<(), CrowdError> {
        for (i, &d) in self.lod_distances.iter().enumerate() {
            if !d.is_finite() || d <= 0.0 {
                return Err(CrowdError::InvalidLodDistances {
                    reason: "LOD distances must be positive and finite",
                });
            }
            if i > 0 && d <= self.lod_distances[i - 1] {
                return Err(CrowdError::InvalidLodDistances {
                    reason: "LOD distances must be strictly increasing",
                });
            }
        }
        Ok(())
    }

    /// Configuration for large crowds (10k+).
    pub fn for_large_crowd() -> Self {
        Self {
            lod_distances: [15.0, 40.0, 80.0],
            frustum_cull: true,
            max_instances: 50_000,
            animation_texture: AnimationTextureAtlas::default(),
            smooth_lod_transitions: true,
            lod_transition_range: 3.0,
            validate_on_add: false,
        }
    }

    /// Configuration for small crowds (<1k).
    pub fn for_small_crowd() -> Self {
        Self {
            lod_distances: [30.0, 60.0, 120.0],
            frustum_cull: true,
            max_instances: 1_000,
            animation_texture: AnimationTextureAtlas::default(),
            smooth_lod_transitions: false,
            lod_transition_range: 2.0,
            validate_on_add: true,
        }
    }
}

impl Default for CrowdRenderConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// IndirectDrawCommand
// ---------------------------------------------------------------------------

/// Indirect draw command for GPU-driven rendering.
///
/// This matches the Vulkan/WebGPU indirect draw command layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct IndirectDrawCommand {
    /// Number of vertices per instance.
    pub vertex_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// First vertex index.
    pub first_vertex: u32,
    /// First instance index (base instance).
    pub first_instance: u32,
}

impl IndirectDrawCommand {
    /// Create a new indirect draw command.
    pub fn new(vertex_count: u32, instance_count: u32) -> Self {
        Self {
            vertex_count,
            instance_count,
            first_vertex: 0,
            first_instance: 0,
        }
    }

    /// Create a command with offsets.
    pub fn with_offsets(
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Self {
        Self {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        }
    }

    /// Check if this is an empty draw (no instances).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instance_count == 0 || self.vertex_count == 0
    }
}

/// Indirect draw command for indexed geometry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct IndirectDrawIndexedCommand {
    /// Number of indices per instance.
    pub index_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// First index offset.
    pub first_index: u32,
    /// Vertex offset added to each index.
    pub vertex_offset: i32,
    /// First instance index (base instance).
    pub first_instance: u32,
}

impl IndirectDrawIndexedCommand {
    /// Create a new indexed indirect draw command.
    pub fn new(index_count: u32, instance_count: u32) -> Self {
        Self {
            index_count,
            instance_count,
            first_index: 0,
            vertex_offset: 0,
            first_instance: 0,
        }
    }

    /// Check if this is an empty draw.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instance_count == 0 || self.index_count == 0
    }
}

// ---------------------------------------------------------------------------
// LodBatch
// ---------------------------------------------------------------------------

/// Batch of instances for a specific LOD level.
#[derive(Clone, Debug)]
pub struct LodBatch {
    /// LOD level for this batch.
    pub lod_level: u8,
    /// Mesh/batch ID.
    pub batch_id: usize,
    /// Instance indices within the parent batch.
    pub instance_indices: Vec<u32>,
    /// Instance count.
    pub instance_count: u32,
}

impl LodBatch {
    /// Create a new LOD batch.
    pub fn new(lod_level: u8, batch_id: usize) -> Self {
        Self {
            lod_level,
            batch_id,
            instance_indices: Vec::new(),
            instance_count: 0,
        }
    }

    /// Check if the batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instance_count == 0
    }
}

// ---------------------------------------------------------------------------
// CrowdRenderer
// ---------------------------------------------------------------------------

/// GPU instanced crowd renderer.
///
/// Manages multiple batches of crowd instances with:
/// - Per-instance animation offsets
/// - Distance-based LOD selection
/// - Frustum culling
/// - Indirect draw command generation
#[derive(Debug)]
pub struct CrowdRenderer {
    /// Renderer configuration.
    pub config: CrowdRenderConfig,
    /// All crowd batches.
    pub batches: Vec<CrowdBatch>,
    /// Total instance count across all batches.
    total_instances: usize,
    /// Total visible instance count.
    total_visible: usize,
    /// Current frame number.
    frame: u64,
    /// Global animation time.
    global_time: f32,
    /// Next buffer handle ID.
    next_buffer_id: u32,
}

impl CrowdRenderer {
    /// Create a new crowd renderer with the given configuration.
    pub fn new(config: CrowdRenderConfig) -> Self {
        Self {
            config,
            batches: Vec::new(),
            total_instances: 0,
            total_visible: 0,
            frame: 0,
            global_time: 0.0,
            next_buffer_id: 1,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CrowdRenderConfig::default())
    }

    /// Add a new batch for a mesh.
    ///
    /// Returns the batch ID.
    pub fn add_batch(&mut self, mesh_id: usize) -> usize {
        let batch_id = self.batches.len();
        self.batches.push(CrowdBatch::new(mesh_id));
        batch_id
    }

    /// Add a batch with pre-allocated capacity.
    pub fn add_batch_with_capacity(&mut self, mesh_id: usize, capacity: usize) -> usize {
        let batch_id = self.batches.len();
        self.batches.push(CrowdBatch::with_capacity(mesh_id, capacity));
        batch_id
    }

    /// Add an instance to a batch.
    pub fn add_instance(&mut self, batch_id: usize, instance: CrowdInstance) -> Result<usize, CrowdError> {
        if batch_id >= self.batches.len() {
            return Err(CrowdError::BatchNotFound { batch_id });
        }

        if self.total_instances >= self.config.max_instances as usize {
            return Err(CrowdError::BufferFull {
                max: self.config.max_instances,
            });
        }

        if self.config.validate_on_add {
            instance.validate()?;
        }

        let index = self.batches[batch_id].add_instance(instance);
        self.total_instances += 1;

        Ok(index)
    }

    /// Remove an instance from a batch.
    pub fn remove_instance(&mut self, batch_id: usize, instance_id: usize) -> Result<Option<usize>, CrowdError> {
        if batch_id >= self.batches.len() {
            return Err(CrowdError::BatchNotFound { batch_id });
        }

        let result = self.batches[batch_id].remove_instance(instance_id);
        if result.is_some() || instance_id < self.batches[batch_id].instance_count() + 1 {
            self.total_instances = self.total_instances.saturating_sub(1);
        }

        Ok(result)
    }

    /// Update an instance in a batch.
    pub fn update_instance(
        &mut self,
        batch_id: usize,
        instance_id: usize,
        instance: CrowdInstance,
    ) -> Result<(), CrowdError> {
        if batch_id >= self.batches.len() {
            return Err(CrowdError::BatchNotFound { batch_id });
        }

        if self.config.validate_on_add {
            instance.validate()?;
        }

        self.batches[batch_id].update_instance(instance_id, instance)
    }

    /// Get a batch by ID.
    pub fn get_batch(&self, batch_id: usize) -> Option<&CrowdBatch> {
        self.batches.get(batch_id)
    }

    /// Get a mutable batch by ID.
    pub fn get_batch_mut(&mut self, batch_id: usize) -> Option<&mut CrowdBatch> {
        self.batches.get_mut(batch_id)
    }

    /// Update LOD levels for all instances based on camera position.
    pub fn update_lods(&mut self, camera_pos: impl Into<Vec3>) {
        let camera = camera_pos.into();
        for batch in &mut self.batches {
            batch.update_lods(camera, &self.config.lod_distances);
        }
    }

    /// Cull all instances against a frustum.
    ///
    /// Updates visibility flags and recalculates visible counts.
    pub fn cull_frustum(&mut self, frustum: &Frustum) {
        if !self.config.frustum_cull {
            // Mark all visible if culling is disabled
            self.total_visible = self.total_instances;
            for batch in &mut self.batches {
                for instance in &mut batch.instances {
                    instance.set_visible(true);
                }
                batch.visible_count = batch.instances.len();
            }
            return;
        }

        self.total_visible = 0;
        for batch in &mut self.batches {
            let visible = batch.cull_frustum(frustum);
            self.total_visible += visible;
        }
    }

    /// Prepare GPU buffers for all batches.
    ///
    /// Returns a list of buffers that need to be uploaded.
    pub fn prepare_gpu_buffers(&mut self) -> Vec<GpuBuffer> {
        let mut buffers = Vec::new();

        for batch in &mut self.batches {
            if batch.is_dirty() && !batch.is_empty() {
                // Create instance buffer
                let handle = BufferHandle::new(self.next_buffer_id);
                self.next_buffer_id += 1;

                let data = batch.as_bytes().to_vec();
                let buffer = GpuBuffer::new(
                    handle,
                    data,
                    GpuBufferUsage::VERTEX,
                );
                buffers.push(buffer);

                batch.instance_buffer = Some(handle);
                batch.mark_clean();
            }
        }

        buffers
    }

    /// Get draw commands for all visible instances.
    ///
    /// Groups instances by LOD level for efficient rendering.
    pub fn get_draw_commands(&self) -> Vec<IndirectDrawCommand> {
        let mut commands = Vec::new();
        let mut first_instance: u32 = 0;

        for batch in &self.batches {
            if batch.visible_count() == 0 {
                continue;
            }

            // Group by LOD level
            let mut lod_counts = [0u32; MAX_LOD_LEVELS];
            for instance in batch.visible_instances() {
                let lod = (instance.lod_level as usize).min(MAX_LOD_LEVELS - 1);
                lod_counts[lod] += 1;
            }

            // Create a command for each LOD level with instances
            for (lod, &count) in lod_counts.iter().enumerate() {
                if count > 0 {
                    // Vertex count depends on LOD level (simplified for this implementation)
                    let vertex_count = match lod {
                        0 => 1000,   // Full skeleton mesh
                        1 => 200,    // Simplified mesh
                        2 => 4,      // Billboard quad
                        _ => 4,
                    };

                    commands.push(IndirectDrawCommand::with_offsets(
                        vertex_count,
                        count,
                        0,
                        first_instance,
                    ));

                    first_instance += count;
                }
            }
        }

        commands
    }

    /// Get draw commands grouped by batch and LOD.
    pub fn get_draw_commands_by_batch(&self) -> Vec<(usize, Vec<IndirectDrawCommand>)> {
        let mut result = Vec::new();

        for (batch_id, batch) in self.batches.iter().enumerate() {
            if batch.visible_count() == 0 {
                continue;
            }

            let mut commands = Vec::new();
            let mut lod_counts = [0u32; MAX_LOD_LEVELS];

            for instance in batch.visible_instances() {
                let lod = (instance.lod_level as usize).min(MAX_LOD_LEVELS - 1);
                lod_counts[lod] += 1;
            }

            for (lod, &count) in lod_counts.iter().enumerate() {
                if count > 0 {
                    let vertex_count = match lod {
                        0 => 1000,
                        1 => 200,
                        2 => 4,
                        _ => 4,
                    };
                    commands.push(IndirectDrawCommand::new(vertex_count, count));
                }
            }

            if !commands.is_empty() {
                result.push((batch_id, commands));
            }
        }

        result
    }

    /// Get LOD batches (instances grouped by LOD level).
    pub fn get_lod_batches(&self) -> Vec<LodBatch> {
        let mut lod_batches = Vec::new();

        for (batch_id, batch) in self.batches.iter().enumerate() {
            let mut lod_indices: [Vec<u32>; MAX_LOD_LEVELS] = Default::default();

            for (i, instance) in batch.instances.iter().enumerate() {
                if instance.is_visible() {
                    let lod = (instance.lod_level as usize).min(MAX_LOD_LEVELS - 1);
                    lod_indices[lod].push(i as u32);
                }
            }

            for (lod, indices) in lod_indices.into_iter().enumerate() {
                if !indices.is_empty() {
                    lod_batches.push(LodBatch {
                        lod_level: lod as u8,
                        batch_id,
                        instance_count: indices.len() as u32,
                        instance_indices: indices,
                    });
                }
            }
        }

        lod_batches
    }

    /// Get total instance count across all batches.
    #[inline]
    pub fn instance_count(&self) -> usize {
        self.total_instances
    }

    /// Get total visible instance count.
    #[inline]
    pub fn visible_count(&self) -> usize {
        self.total_visible
    }

    /// Get the number of batches.
    #[inline]
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Update animation time for all instances.
    ///
    /// Advances the global time and optionally syncs phase-synchronized instances.
    pub fn update_animation_time(&mut self, delta_time: f32) {
        self.global_time += delta_time;

        for batch in &mut self.batches {
            for instance in &mut batch.instances {
                if instance.is_phase_synced() {
                    // Phase-synced instances use global time
                    instance.animation_time = self.global_time;
                } else {
                    // Non-synced instances advance independently
                    instance.animation_time += delta_time;
                }
            }
            batch.dirty = true;
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }

    /// Get the current frame number.
    #[inline]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Get the global animation time.
    #[inline]
    pub fn global_time(&self) -> f32 {
        self.global_time
    }

    /// Clear all batches.
    pub fn clear(&mut self) {
        for batch in &mut self.batches {
            batch.clear();
        }
        self.total_instances = 0;
        self.total_visible = 0;
    }

    /// Remove all batches.
    pub fn clear_batches(&mut self) {
        self.batches.clear();
        self.total_instances = 0;
        self.total_visible = 0;
    }

    /// Get statistics about the current state.
    pub fn stats(&self) -> CrowdRendererStats {
        let mut lod_counts = [0usize; MAX_LOD_LEVELS];
        let mut total_buffer_size = 0usize;

        for batch in &self.batches {
            total_buffer_size += batch.buffer_size();
            for instance in &batch.instances {
                if instance.is_visible() {
                    let lod = (instance.lod_level as usize).min(MAX_LOD_LEVELS - 1);
                    lod_counts[lod] += 1;
                }
            }
        }

        CrowdRendererStats {
            batch_count: self.batches.len(),
            total_instances: self.total_instances,
            visible_instances: self.total_visible,
            lod_0_count: lod_counts[0],
            lod_1_count: lod_counts[1],
            lod_2_count: lod_counts[2],
            total_buffer_size,
            frame: self.frame,
        }
    }
}

impl Default for CrowdRenderer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// CrowdRendererStats
// ---------------------------------------------------------------------------

/// Statistics about the crowd renderer state.
#[derive(Clone, Debug, Default)]
pub struct CrowdRendererStats {
    /// Number of batches.
    pub batch_count: usize,
    /// Total instance count.
    pub total_instances: usize,
    /// Visible instance count.
    pub visible_instances: usize,
    /// Instances at LOD 0 (full skeleton).
    pub lod_0_count: usize,
    /// Instances at LOD 1 (simplified).
    pub lod_1_count: usize,
    /// Instances at LOD 2 (impostor).
    pub lod_2_count: usize,
    /// Total buffer size in bytes.
    pub total_buffer_size: usize,
    /// Current frame number.
    pub frame: u64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Vec3 Tests
    // ========================================================================

    #[test]
    fn test_vec3_new() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3_distance_squared() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(a.distance_squared(b), 25.0);
    }

    #[test]
    fn test_vec3_distance() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        assert!((a.distance(b) - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec3_from_array() {
        let v: Vec3 = [1.0, 2.0, 3.0].into();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3_to_array() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let arr: [f32; 3] = v.into();
        assert_eq!(arr, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vec3_is_finite() {
        assert!(Vec3::new(1.0, 2.0, 3.0).is_finite());
        assert!(!Vec3::new(f32::NAN, 0.0, 0.0).is_finite());
        assert!(!Vec3::new(0.0, f32::INFINITY, 0.0).is_finite());
    }

    // ========================================================================
    // Quat Tests
    // ========================================================================

    #[test]
    fn test_quat_identity() {
        let q = Quat::IDENTITY;
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
        assert_eq!(q.w, 1.0);
    }

    #[test]
    fn test_quat_from_rotation_y() {
        let q = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        // 90 degree rotation around Y
        assert!((q.y - 0.707).abs() < 0.01);
        assert!((q.w - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_quat_normalize() {
        let q = Quat::new(0.0, 0.0, 0.0, 2.0);
        let n = q.normalize();
        assert!((n.w - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_quat_is_finite() {
        assert!(Quat::IDENTITY.is_finite());
        assert!(!Quat::new(f32::NAN, 0.0, 0.0, 1.0).is_finite());
    }

    // ========================================================================
    // Frustum Tests
    // ========================================================================

    #[test]
    fn test_frustum_from_bounds() {
        let frustum = Frustum::from_bounds(
            Vec3::new(-100.0, -100.0, -100.0),
            Vec3::new(100.0, 100.0, 100.0),
        );
        assert!(frustum.contains_point(Vec3::ZERO));
        assert!(!frustum.contains_point(Vec3::new(200.0, 0.0, 0.0)));
    }

    #[test]
    fn test_frustum_contains_sphere() {
        let frustum = Frustum::from_bounds(
            Vec3::new(-100.0, -100.0, -100.0),
            Vec3::new(100.0, 100.0, 100.0),
        );

        // Center inside
        assert!(frustum.contains_sphere(Vec3::ZERO, 10.0));

        // Edge overlapping
        assert!(frustum.contains_sphere(Vec3::new(95.0, 0.0, 0.0), 10.0));

        // Completely outside
        assert!(!frustum.contains_sphere(Vec3::new(200.0, 0.0, 0.0), 10.0));
    }

    #[test]
    fn test_frustum_unbounded() {
        let frustum = Frustum::unbounded();
        assert!(frustum.contains_point(Vec3::new(1e10, 1e10, 1e10)));
    }

    // ========================================================================
    // CrowdInstance Tests
    // ========================================================================

    #[test]
    fn test_instance_struct_size() {
        assert_eq!(std::mem::size_of::<CrowdInstance>(), CROWD_INSTANCE_SIZE);
    }

    #[test]
    fn test_instance_new() {
        let instance = CrowdInstance::new([1.0, 2.0, 3.0]);
        assert_eq!(instance.position.x, 1.0);
        assert_eq!(instance.position.y, 2.0);
        assert_eq!(instance.position.z, 3.0);
        assert_eq!(instance.scale, 1.0);
        assert_eq!(instance.animation_id, 0);
        assert_eq!(instance.lod_level, LOD_FULL_SKELETON);
    }

    #[test]
    fn test_instance_builder_pattern() {
        let instance = CrowdInstance::new([0.0, 0.0, 0.0])
            .with_rotation(Quat::from_rotation_y(1.0))
            .with_scale(2.0)
            .with_animation(5, 0.5)
            .with_lod(LOD_SIMPLIFIED)
            .with_phase_sync(true);

        assert_eq!(instance.scale, 2.0);
        assert_eq!(instance.animation_id, 5);
        assert!((instance.animation_time - 0.5).abs() < 0.0001);
        assert_eq!(instance.lod_level, LOD_SIMPLIFIED);
        assert!(instance.is_phase_synced());
    }

    #[test]
    fn test_instance_visibility() {
        let mut instance = CrowdInstance::new([0.0, 0.0, 0.0]);
        assert!(instance.is_visible()); // Default is visible

        instance.set_visible(false);
        assert!(!instance.is_visible());

        instance.set_visible(true);
        assert!(instance.is_visible());
    }

    #[test]
    fn test_instance_validate_valid() {
        let instance = CrowdInstance::new([1.0, 2.0, 3.0]).with_scale(1.0);
        assert!(instance.validate().is_ok());
    }

    #[test]
    fn test_instance_validate_nan_position() {
        let mut instance = CrowdInstance::new([0.0, 0.0, 0.0]);
        instance.position = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_validate_zero_scale() {
        let instance = CrowdInstance::new([0.0, 0.0, 0.0]).with_scale(0.0);
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_validate_negative_scale() {
        let instance = CrowdInstance::new([0.0, 0.0, 0.0]).with_scale(-1.0);
        assert!(instance.validate().is_err());
    }

    #[test]
    fn test_instance_distance() {
        let instance = CrowdInstance::new([3.0, 4.0, 0.0]);
        let dist = instance.distance(Vec3::ZERO);
        assert!((dist - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_instance_bounding_radius() {
        let instance = CrowdInstance::new([0.0, 0.0, 0.0]).with_scale(2.5);
        assert_eq!(instance.bounding_radius(), 2.5);
    }

    #[test]
    fn test_instance_pod_zeroable() {
        let instance = CrowdInstance::default();
        let bytes: &[u8] = bytemuck::bytes_of(&instance);
        assert_eq!(bytes.len(), CROWD_INSTANCE_SIZE);
    }

    // ========================================================================
    // CrowdBatch Tests
    // ========================================================================

    #[test]
    fn test_batch_new() {
        let batch = CrowdBatch::new(42);
        assert_eq!(batch.mesh_id, 42);
        assert!(batch.is_empty());
        assert_eq!(batch.instance_count(), 0);
    }

    #[test]
    fn test_batch_add_instance() {
        let mut batch = CrowdBatch::new(0);
        let idx = batch.add_instance(CrowdInstance::new([1.0, 0.0, 0.0]));
        assert_eq!(idx, 0);
        assert_eq!(batch.instance_count(), 1);
        assert!(batch.is_dirty());
    }

    #[test]
    fn test_batch_add_multiple_instances() {
        let mut batch = CrowdBatch::new(0);
        for i in 0..10 {
            let idx = batch.add_instance(CrowdInstance::new([i as f32, 0.0, 0.0]));
            assert_eq!(idx, i);
        }
        assert_eq!(batch.instance_count(), 10);
    }

    #[test]
    fn test_batch_remove_instance() {
        let mut batch = CrowdBatch::new(0);
        for i in 0..5 {
            batch.add_instance(CrowdInstance::new([i as f32, 0.0, 0.0]));
        }

        // Remove middle instance
        let swapped = batch.remove_instance(2);
        assert_eq!(swapped, Some(4)); // Last instance swapped to index 2

        assert_eq!(batch.instance_count(), 4);

        // Check that position at index 2 is now 4.0 (was last)
        assert_eq!(batch.get(2).unwrap().position.x, 4.0);
    }

    #[test]
    fn test_batch_remove_last_instance() {
        let mut batch = CrowdBatch::new(0);
        batch.add_instance(CrowdInstance::new([0.0, 0.0, 0.0]));
        batch.add_instance(CrowdInstance::new([1.0, 0.0, 0.0]));

        let swapped = batch.remove_instance(1);
        assert_eq!(swapped, None); // No swap when removing last

        assert_eq!(batch.instance_count(), 1);
    }

    #[test]
    fn test_batch_update_lods() {
        let mut batch = CrowdBatch::new(0);
        batch.add_instance(CrowdInstance::new([5.0, 0.0, 0.0]));   // Close
        batch.add_instance(CrowdInstance::new([30.0, 0.0, 0.0]));  // Medium
        batch.add_instance(CrowdInstance::new([80.0, 0.0, 0.0]));  // Far

        let lod_distances = [20.0, 50.0, 100.0];
        batch.update_lods(Vec3::ZERO, &lod_distances);

        assert_eq!(batch.get(0).unwrap().lod_level, LOD_FULL_SKELETON);
        assert_eq!(batch.get(1).unwrap().lod_level, LOD_SIMPLIFIED);
        assert_eq!(batch.get(2).unwrap().lod_level, LOD_IMPOSTOR);
    }

    #[test]
    fn test_batch_cull_frustum() {
        let mut batch = CrowdBatch::new(0);
        batch.add_instance(CrowdInstance::new([0.0, 0.0, 0.0]));     // Inside
        batch.add_instance(CrowdInstance::new([500.0, 0.0, 0.0]));   // Outside

        let frustum = Frustum::from_bounds(
            Vec3::new(-100.0, -100.0, -100.0),
            Vec3::new(100.0, 100.0, 100.0),
        );

        let visible = batch.cull_frustum(&frustum);
        assert_eq!(visible, 1);
        assert!(batch.get(0).unwrap().is_visible());
        assert!(!batch.get(1).unwrap().is_visible());
    }

    #[test]
    fn test_batch_visible_instances() {
        let mut batch = CrowdBatch::new(0);
        batch.add_instance(CrowdInstance::new([0.0, 0.0, 0.0]));
        batch.add_instance(CrowdInstance::new([1.0, 0.0, 0.0]));
        batch.add_instance(CrowdInstance::new([2.0, 0.0, 0.0]));

        batch.instances[1].set_visible(false);

        let visible: Vec<_> = batch.visible_instances().collect();
        assert_eq!(visible.len(), 2);
    }

    #[test]
    fn test_batch_as_bytes() {
        let mut batch = CrowdBatch::new(0);
        batch.add_instance(CrowdInstance::new([1.0, 0.0, 0.0]));

        let bytes = batch.as_bytes();
        assert_eq!(bytes.len(), CROWD_INSTANCE_SIZE);
    }

    #[test]
    fn test_batch_clear() {
        let mut batch = CrowdBatch::new(0);
        for _ in 0..5 {
            batch.add_instance(CrowdInstance::default());
        }

        batch.clear();
        assert!(batch.is_empty());
        assert_eq!(batch.visible_count(), 0);
    }

    // ========================================================================
    // CrowdRenderConfig Tests
    // ========================================================================

    #[test]
    fn test_config_default() {
        let config = CrowdRenderConfig::default();
        assert_eq!(config.lod_distances, [20.0, 50.0, 100.0]);
        assert!(config.frustum_cull);
        assert_eq!(config.max_instances, DEFAULT_MAX_INSTANCES);
    }

    #[test]
    fn test_config_builder() {
        let config = CrowdRenderConfig::new()
            .with_lod_distances([10.0, 30.0, 60.0])
            .with_max_instances(5000)
            .with_frustum_cull(false)
            .with_smooth_lod_transitions(true);

        assert_eq!(config.lod_distances, [10.0, 30.0, 60.0]);
        assert_eq!(config.max_instances, 5000);
        assert!(!config.frustum_cull);
        assert!(config.smooth_lod_transitions);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = CrowdRenderConfig::new();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_non_increasing() {
        let config = CrowdRenderConfig::new()
            .with_lod_distances([50.0, 30.0, 100.0]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_distance() {
        let config = CrowdRenderConfig::new()
            .with_lod_distances([0.0, 50.0, 100.0]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_for_large_crowd() {
        let config = CrowdRenderConfig::for_large_crowd();
        assert_eq!(config.max_instances, 50_000);
        assert!(config.smooth_lod_transitions);
    }

    #[test]
    fn test_config_for_small_crowd() {
        let config = CrowdRenderConfig::for_small_crowd();
        assert_eq!(config.max_instances, 1_000);
        assert!(config.validate_on_add);
    }

    // ========================================================================
    // IndirectDrawCommand Tests
    // ========================================================================

    #[test]
    fn test_draw_command_new() {
        let cmd = IndirectDrawCommand::new(1000, 50);
        assert_eq!(cmd.vertex_count, 1000);
        assert_eq!(cmd.instance_count, 50);
        assert_eq!(cmd.first_vertex, 0);
        assert_eq!(cmd.first_instance, 0);
    }

    #[test]
    fn test_draw_command_with_offsets() {
        let cmd = IndirectDrawCommand::with_offsets(1000, 50, 10, 20);
        assert_eq!(cmd.first_vertex, 10);
        assert_eq!(cmd.first_instance, 20);
    }

    #[test]
    fn test_draw_command_is_empty() {
        assert!(IndirectDrawCommand::new(0, 10).is_empty());
        assert!(IndirectDrawCommand::new(100, 0).is_empty());
        assert!(!IndirectDrawCommand::new(100, 10).is_empty());
    }

    #[test]
    fn test_indexed_draw_command() {
        let cmd = IndirectDrawIndexedCommand::new(3000, 100);
        assert_eq!(cmd.index_count, 3000);
        assert_eq!(cmd.instance_count, 100);
        assert!(!cmd.is_empty());
    }

    // ========================================================================
    // CrowdRenderer Tests
    // ========================================================================

    #[test]
    fn test_renderer_new() {
        let renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        assert_eq!(renderer.batch_count(), 0);
        assert_eq!(renderer.instance_count(), 0);
        assert_eq!(renderer.visible_count(), 0);
    }

    #[test]
    fn test_renderer_add_batch() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(42);
        assert_eq!(batch_id, 0);
        assert_eq!(renderer.batch_count(), 1);
    }

    #[test]
    fn test_renderer_add_instance() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        let idx = renderer.add_instance(batch_id, CrowdInstance::new([1.0, 0.0, 0.0])).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(renderer.instance_count(), 1);
    }

    #[test]
    fn test_renderer_add_instance_invalid_batch() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let result = renderer.add_instance(999, CrowdInstance::default());
        assert!(matches!(result, Err(CrowdError::BatchNotFound { .. })));
    }

    #[test]
    fn test_renderer_max_instances() {
        let config = CrowdRenderConfig::new().with_max_instances(5);
        let mut renderer = CrowdRenderer::new(config);
        let batch_id = renderer.add_batch(0);

        for _ in 0..5 {
            renderer.add_instance(batch_id, CrowdInstance::default()).unwrap();
        }

        let result = renderer.add_instance(batch_id, CrowdInstance::default());
        assert!(matches!(result, Err(CrowdError::BufferFull { .. })));
    }

    #[test]
    fn test_renderer_update_lods() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        renderer.add_instance(batch_id, CrowdInstance::new([5.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([30.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([80.0, 0.0, 0.0])).unwrap();

        renderer.update_lods(Vec3::ZERO);

        let batch = renderer.get_batch(batch_id).unwrap();
        assert_eq!(batch.get(0).unwrap().lod_level, LOD_FULL_SKELETON);
        assert_eq!(batch.get(1).unwrap().lod_level, LOD_SIMPLIFIED);
        assert_eq!(batch.get(2).unwrap().lod_level, LOD_IMPOSTOR);
    }

    #[test]
    fn test_renderer_cull_frustum() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        renderer.add_instance(batch_id, CrowdInstance::new([0.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([500.0, 0.0, 0.0])).unwrap();

        let frustum = Frustum::from_bounds(
            Vec3::new(-100.0, -100.0, -100.0),
            Vec3::new(100.0, 100.0, 100.0),
        );

        renderer.cull_frustum(&frustum);

        assert_eq!(renderer.visible_count(), 1);
    }

    #[test]
    fn test_renderer_cull_frustum_disabled() {
        let config = CrowdRenderConfig::new().with_frustum_cull(false);
        let mut renderer = CrowdRenderer::new(config);
        let batch_id = renderer.add_batch(0);

        renderer.add_instance(batch_id, CrowdInstance::new([0.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([500.0, 0.0, 0.0])).unwrap();

        let frustum = Frustum::from_bounds(
            Vec3::new(-100.0, -100.0, -100.0),
            Vec3::new(100.0, 100.0, 100.0),
        );

        renderer.cull_frustum(&frustum);

        // Both visible when culling is disabled
        assert_eq!(renderer.visible_count(), 2);
    }

    #[test]
    fn test_renderer_prepare_gpu_buffers() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        renderer.add_instance(batch_id, CrowdInstance::new([0.0, 0.0, 0.0])).unwrap();

        let buffers = renderer.prepare_gpu_buffers();
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].size, CROWD_INSTANCE_SIZE);
    }

    #[test]
    fn test_renderer_get_draw_commands() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        // Add instances at different LOD levels
        renderer.add_instance(batch_id, CrowdInstance::new([5.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([30.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([80.0, 0.0, 0.0])).unwrap();

        renderer.update_lods(Vec3::ZERO);
        renderer.cull_frustum(&Frustum::unbounded());

        let commands = renderer.get_draw_commands();

        // Should have 3 commands (one per LOD level)
        assert_eq!(commands.len(), 3);
    }

    #[test]
    fn test_renderer_get_draw_commands_empty() {
        let renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let commands = renderer.get_draw_commands();
        assert!(commands.is_empty());
    }

    #[test]
    fn test_renderer_get_lod_batches() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        renderer.add_instance(batch_id, CrowdInstance::new([5.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([6.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([30.0, 0.0, 0.0])).unwrap();

        renderer.update_lods(Vec3::ZERO);
        renderer.cull_frustum(&Frustum::unbounded());

        let lod_batches = renderer.get_lod_batches();

        // Should have 2 batches: LOD 0 (2 instances) and LOD 1 (1 instance)
        assert_eq!(lod_batches.len(), 2);

        let lod0 = lod_batches.iter().find(|b| b.lod_level == LOD_FULL_SKELETON).unwrap();
        assert_eq!(lod0.instance_count, 2);

        let lod1 = lod_batches.iter().find(|b| b.lod_level == LOD_SIMPLIFIED).unwrap();
        assert_eq!(lod1.instance_count, 1);
    }

    #[test]
    fn test_renderer_update_animation_time() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        renderer.add_instance(batch_id, CrowdInstance::new([0.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(
            batch_id,
            CrowdInstance::new([1.0, 0.0, 0.0]).with_phase_sync(true)
        ).unwrap();

        renderer.update_animation_time(0.5);

        {
            let batch = renderer.get_batch(batch_id).unwrap();

            // Non-synced instance advances by delta
            assert!((batch.get(0).unwrap().animation_time - 0.5).abs() < 0.0001);

            // Synced instance uses global time
            assert!((batch.get(1).unwrap().animation_time - 0.5).abs() < 0.0001);
        }

        renderer.update_animation_time(0.3);

        {
            let batch = renderer.get_batch(batch_id).unwrap();

            // Non-synced: 0.5 + 0.3 = 0.8
            assert!((batch.get(0).unwrap().animation_time - 0.8).abs() < 0.0001);

            // Synced: global time = 0.8
            assert!((batch.get(1).unwrap().animation_time - 0.8).abs() < 0.0001);
        }
    }

    #[test]
    fn test_renderer_stats() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        renderer.add_instance(batch_id, CrowdInstance::new([5.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_id, CrowdInstance::new([30.0, 0.0, 0.0])).unwrap();

        renderer.update_lods(Vec3::ZERO);
        renderer.cull_frustum(&Frustum::unbounded());

        let stats = renderer.stats();

        assert_eq!(stats.batch_count, 1);
        assert_eq!(stats.total_instances, 2);
        assert_eq!(stats.visible_instances, 2);
        assert_eq!(stats.lod_0_count, 1);
        assert_eq!(stats.lod_1_count, 1);
        assert_eq!(stats.total_buffer_size, 2 * CROWD_INSTANCE_SIZE);
    }

    #[test]
    fn test_renderer_clear() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let batch_id = renderer.add_batch(0);

        for _ in 0..10 {
            renderer.add_instance(batch_id, CrowdInstance::default()).unwrap();
        }

        renderer.clear();

        assert_eq!(renderer.instance_count(), 0);
        assert_eq!(renderer.visible_count(), 0);
        assert_eq!(renderer.batch_count(), 1); // Batches remain
    }

    #[test]
    fn test_renderer_clear_batches() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        renderer.add_batch(0);
        renderer.add_batch(1);

        renderer.clear_batches();

        assert_eq!(renderer.batch_count(), 0);
    }

    #[test]
    fn test_renderer_begin_frame() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        assert_eq!(renderer.frame(), 0);

        renderer.begin_frame();
        assert_eq!(renderer.frame(), 1);

        renderer.begin_frame();
        assert_eq!(renderer.frame(), 2);
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_empty_batch_culling() {
        let mut batch = CrowdBatch::new(0);
        let frustum = Frustum::unbounded();
        let visible = batch.cull_frustum(&frustum);
        assert_eq!(visible, 0);
    }

    #[test]
    fn test_empty_renderer_draw_commands() {
        let renderer = CrowdRenderer::new(CrowdRenderConfig::default());
        let commands = renderer.get_draw_commands();
        assert!(commands.is_empty());
    }

    #[test]
    fn test_validation_on_add() {
        let config = CrowdRenderConfig::new().with_validation(true);
        let mut renderer = CrowdRenderer::new(config);
        let batch_id = renderer.add_batch(0);

        // Valid instance
        let result = renderer.add_instance(batch_id, CrowdInstance::new([1.0, 0.0, 0.0]));
        assert!(result.is_ok());

        // Invalid instance (NaN position)
        let mut invalid = CrowdInstance::new([0.0, 0.0, 0.0]);
        invalid.position = Vec3::new(f32::NAN, 0.0, 0.0);
        let result = renderer.add_instance(batch_id, invalid);
        assert!(matches!(result, Err(CrowdError::InvalidInstance { .. })));
    }

    #[test]
    fn test_lod_beyond_max_distance() {
        let mut batch = CrowdBatch::new(0);
        batch.add_instance(CrowdInstance::new([1000.0, 0.0, 0.0])); // Very far

        let lod_distances = [20.0, 50.0, 100.0];
        batch.update_lods(Vec3::ZERO, &lod_distances);

        // Should still be impostor even beyond max distance
        assert_eq!(batch.get(0).unwrap().lod_level, LOD_IMPOSTOR);
    }

    #[test]
    fn test_buffer_handle() {
        let handle = BufferHandle::new(42);
        assert!(handle.is_valid());
        assert!(!BufferHandle::INVALID.is_valid());
    }

    #[test]
    fn test_gpu_buffer() {
        let buffer = GpuBuffer::new(
            BufferHandle::new(1),
            vec![0u8; 64],
            GpuBufferUsage::VERTEX,
        );
        assert_eq!(buffer.size, 64);
        assert_eq!(buffer.handle.0, 1);
    }

    #[test]
    fn test_animation_texture_atlas() {
        let atlas = AnimationTextureAtlas::new(1, 5, 100, 64);
        assert_eq!(atlas.handle, 1);
        assert_eq!(atlas.clip_count, 5);
    }

    // ========================================================================
    // Error Display Tests
    // ========================================================================

    #[test]
    fn test_error_display_buffer_full() {
        let err = CrowdError::BufferFull { max: 1000 };
        assert!(err.to_string().contains("1000"));
    }

    #[test]
    fn test_error_display_invalid_instance() {
        let err = CrowdError::InvalidInstance { reason: "test reason" };
        assert!(err.to_string().contains("test reason"));
    }

    #[test]
    fn test_error_display_batch_not_found() {
        let err = CrowdError::BatchNotFound { batch_id: 42 };
        assert!(err.to_string().contains("42"));
    }

    #[test]
    fn test_error_display_invalid_lod_distances() {
        let err = CrowdError::InvalidLodDistances { reason: "must be increasing" };
        assert!(err.to_string().contains("increasing"));
    }

    // ========================================================================
    // Concurrency Simulation Tests
    // ========================================================================

    #[test]
    fn test_large_crowd_simulation() {
        let config = CrowdRenderConfig::for_large_crowd();
        let mut renderer = CrowdRenderer::new(config);
        let batch_id = renderer.add_batch(0);

        // Add 1000 instances in a grid
        for i in 0..1000 {
            let x = (i % 32) as f32 * 2.0;
            let z = (i / 32) as f32 * 2.0;
            renderer.add_instance(
                batch_id,
                CrowdInstance::new([x, 0.0, z])
                    .with_animation((i % 5) as u16, (i as f32) * 0.1)
            ).unwrap();
        }

        assert_eq!(renderer.instance_count(), 1000);

        // Update LODs and cull
        renderer.update_lods([16.0, 0.0, 16.0]);
        renderer.cull_frustum(&Frustum::unbounded());

        // All should be visible
        assert_eq!(renderer.visible_count(), 1000);

        // Get stats
        let stats = renderer.stats();
        assert!(stats.lod_0_count > 0);
        assert!(stats.lod_1_count > 0);
        assert!(stats.lod_2_count > 0);
    }

    #[test]
    fn test_multiple_batches() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());

        let batch_a = renderer.add_batch(0);
        let batch_b = renderer.add_batch(1);

        for _ in 0..5 {
            renderer.add_instance(batch_a, CrowdInstance::default()).unwrap();
        }
        for _ in 0..3 {
            renderer.add_instance(batch_b, CrowdInstance::default()).unwrap();
        }

        assert_eq!(renderer.batch_count(), 2);
        assert_eq!(renderer.instance_count(), 8);
        assert_eq!(renderer.get_batch(batch_a).unwrap().instance_count(), 5);
        assert_eq!(renderer.get_batch(batch_b).unwrap().instance_count(), 3);
    }

    #[test]
    fn test_draw_commands_by_batch() {
        let mut renderer = CrowdRenderer::new(CrowdRenderConfig::default());

        let batch_a = renderer.add_batch(0);
        let batch_b = renderer.add_batch(1);

        renderer.add_instance(batch_a, CrowdInstance::new([0.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_b, CrowdInstance::new([0.0, 0.0, 0.0])).unwrap();
        renderer.add_instance(batch_b, CrowdInstance::new([0.0, 0.0, 0.0])).unwrap();

        renderer.cull_frustum(&Frustum::unbounded());

        let commands = renderer.get_draw_commands_by_batch();

        assert_eq!(commands.len(), 2);
    }
}
