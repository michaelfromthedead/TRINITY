//! Blend Shape / Morph Target System for TRINITY Engine (T-AN-7.1).
//!
//! This module provides a complete morph target and blend shape system for
//! facial animation, character customization, and mesh deformation:
//!
//! - Sparse vertex delta storage (only affected vertices)
//! - Multiple shapes per mesh (50-200 typical for facial rigs)
//! - Linear combination blending with weight clamping
//! - Face region masking (eyes, mouth, brow, etc.)
//! - Pre/post-skinning application modes
//! - GPU-ready buffer preparation
//!
//! # Memory Layout
//!
//! Blend shapes use sparse storage to minimize memory footprint:
//!
//! | Type           | Size   | Description                          |
//! |----------------|--------|--------------------------------------|
//! | VertexDelta    | 28B    | Position + normal + optional tangent |
//! | DeltaEntry     | 32B    | Vertex index + delta (GPU aligned)   |
//! | BlendWeight    | 4B     | Per-shape weight (f32)               |
//!
//! # Application Order
//!
//! Blend shapes can be applied in two modes:
//!
//! - **PreSkinning**: Applied before bone transforms. Deltas are in bind pose space.
//!   Best for: facial expressions, body morphs, character customization.
//!
//! - **PostSkinning**: Applied after bone transforms. Deltas are in world space.
//!   Best for: muscle bulging, clothing deformation, dynamic correctives.
//!
//! # Performance Targets
//!
//! - CPU application: < 0.1ms for 100 shapes on 10K vertices
//! - GPU buffer prep: < 0.05ms for 200 shapes
//! - Memory: ~40KB per 1000 affected vertices
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::blend_shapes::{
//!     BlendShapeSet, BlendShapeTarget, VertexDelta, FaceRegion, ApplicationMode,
//! };
//! use glam::Vec3;
//!
//! // Create a blend shape set for a mesh
//! let mut shapes = BlendShapeSet::new(10000); // 10K vertices
//!
//! // Add a smile shape
//! let smile = BlendShapeTarget::new("smile")
//!     .with_delta(100, VertexDelta::position(Vec3::new(0.1, 0.2, 0.0)))
//!     .with_delta(101, VertexDelta::position(Vec3::new(-0.1, 0.2, 0.0)))
//!     .with_region(FaceRegion::Mouth);
//!
//! let smile_idx = shapes.add_target(smile);
//!
//! // Animate the shape
//! shapes.set_weight(smile_idx, 0.8);
//!
//! // Apply to mesh vertices
//! let mut positions = vec![Vec3::ZERO; 10000];
//! let mut normals = vec![Vec3::Y; 10000];
//! shapes.apply(&mut positions, &mut normals);
//!
//! // Or apply only to specific face region
//! shapes.apply_masked(&mut positions, FaceRegion::Mouth);
//!
//! // Prepare GPU buffers for compute shader
//! let (delta_buffer, weight_buffer) = shapes.prepare_gpu_buffers();
//! ```

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::mem;

use glam::Vec3;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of blend shapes per mesh.
pub const MAX_BLEND_SHAPES: usize = 256;

/// Default weight value for new shapes.
pub const DEFAULT_WEIGHT: f32 = 0.0;

/// Minimum weight threshold for considering a shape "active".
pub const WEIGHT_THRESHOLD: f32 = 0.0001;

/// GPU delta entry size in bytes (vertex index + position + normal + tangent flag).
pub const GPU_DELTA_ENTRY_SIZE: usize = 32;

/// GPU workgroup size for blend shape compute shaders.
pub const WORKGROUP_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// FaceRegion
// ---------------------------------------------------------------------------

/// Face regions for masked blend shape application.
///
/// Used to limit blend shape effects to specific areas of the face,
/// enabling independent control of different facial features.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FaceRegion {
    /// Left eye region (eyelid, brow, surrounding skin).
    LeftEye,
    /// Right eye region (eyelid, brow, surrounding skin).
    RightEye,
    /// Mouth region (lips, corners, surrounding skin).
    Mouth,
    /// Left eyebrow region.
    LeftBrow,
    /// Right eyebrow region.
    RightBrow,
    /// Nose region (nostrils, bridge, tip).
    Nose,
    /// Jaw region (lower jaw, chin).
    Jaw,
    /// Cheeks region (both sides).
    Cheeks,
}

impl FaceRegion {
    /// Get all face regions as an array.
    pub fn all() -> &'static [FaceRegion] {
        &[
            FaceRegion::LeftEye,
            FaceRegion::RightEye,
            FaceRegion::Mouth,
            FaceRegion::LeftBrow,
            FaceRegion::RightBrow,
            FaceRegion::Nose,
            FaceRegion::Jaw,
            FaceRegion::Cheeks,
        ]
    }

    /// Get a human-readable name for this region.
    pub fn name(&self) -> &'static str {
        match self {
            FaceRegion::LeftEye => "Left Eye",
            FaceRegion::RightEye => "Right Eye",
            FaceRegion::Mouth => "Mouth",
            FaceRegion::LeftBrow => "Left Brow",
            FaceRegion::RightBrow => "Right Brow",
            FaceRegion::Nose => "Nose",
            FaceRegion::Jaw => "Jaw",
            FaceRegion::Cheeks => "Cheeks",
        }
    }

    /// Check if this is an eye-related region.
    pub fn is_eye(&self) -> bool {
        matches!(self, FaceRegion::LeftEye | FaceRegion::RightEye)
    }

    /// Check if this is a brow-related region.
    pub fn is_brow(&self) -> bool {
        matches!(self, FaceRegion::LeftBrow | FaceRegion::RightBrow)
    }

    /// Get the opposite/mirrored region if applicable.
    pub fn mirror(&self) -> Option<FaceRegion> {
        match self {
            FaceRegion::LeftEye => Some(FaceRegion::RightEye),
            FaceRegion::RightEye => Some(FaceRegion::LeftEye),
            FaceRegion::LeftBrow => Some(FaceRegion::RightBrow),
            FaceRegion::RightBrow => Some(FaceRegion::LeftBrow),
            _ => None,
        }
    }
}

impl Default for FaceRegion {
    fn default() -> Self {
        FaceRegion::Mouth
    }
}

// ---------------------------------------------------------------------------
// BodyRegion
// ---------------------------------------------------------------------------

/// Body regions for masked blend shape application.
///
/// Used for body-wide blend shapes like muscle definition,
/// body type morphs, or clothing fit adjustments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BodyRegion {
    /// Left hand region.
    LeftHand,
    /// Right hand region.
    RightHand,
    /// Left foot region.
    LeftFoot,
    /// Right foot region.
    RightFoot,
    /// Torso/chest region.
    Torso,
    /// Left arm region.
    LeftArm,
    /// Right arm region.
    RightArm,
    /// Left leg region.
    LeftLeg,
    /// Right leg region.
    RightLeg,
    /// Head/neck region.
    Head,
}

impl BodyRegion {
    /// Get all body regions as an array.
    pub fn all() -> &'static [BodyRegion] {
        &[
            BodyRegion::LeftHand,
            BodyRegion::RightHand,
            BodyRegion::LeftFoot,
            BodyRegion::RightFoot,
            BodyRegion::Torso,
            BodyRegion::LeftArm,
            BodyRegion::RightArm,
            BodyRegion::LeftLeg,
            BodyRegion::RightLeg,
            BodyRegion::Head,
        ]
    }

    /// Get a human-readable name for this region.
    pub fn name(&self) -> &'static str {
        match self {
            BodyRegion::LeftHand => "Left Hand",
            BodyRegion::RightHand => "Right Hand",
            BodyRegion::LeftFoot => "Left Foot",
            BodyRegion::RightFoot => "Right Foot",
            BodyRegion::Torso => "Torso",
            BodyRegion::LeftArm => "Left Arm",
            BodyRegion::RightArm => "Right Arm",
            BodyRegion::LeftLeg => "Left Leg",
            BodyRegion::RightLeg => "Right Leg",
            BodyRegion::Head => "Head",
        }
    }

    /// Get the opposite/mirrored region if applicable.
    pub fn mirror(&self) -> Option<BodyRegion> {
        match self {
            BodyRegion::LeftHand => Some(BodyRegion::RightHand),
            BodyRegion::RightHand => Some(BodyRegion::LeftHand),
            BodyRegion::LeftFoot => Some(BodyRegion::RightFoot),
            BodyRegion::RightFoot => Some(BodyRegion::LeftFoot),
            BodyRegion::LeftArm => Some(BodyRegion::RightArm),
            BodyRegion::RightArm => Some(BodyRegion::LeftArm),
            BodyRegion::LeftLeg => Some(BodyRegion::RightLeg),
            BodyRegion::RightLeg => Some(BodyRegion::LeftLeg),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ApplicationMode
// ---------------------------------------------------------------------------

/// When to apply blend shape deltas in the skinning pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApplicationMode {
    /// Apply before bone transforms (bind pose space).
    ///
    /// Deltas are in the mesh's bind pose coordinate system.
    /// The result is then transformed by skeletal animation.
    /// Best for: facial expressions, body morphs, character customization.
    #[default]
    PreSkinning,

    /// Apply after bone transforms (world/skinned space).
    ///
    /// Deltas are applied to already-skinned vertices.
    /// Best for: muscle bulging, dynamic correctives, cloth simulation.
    PostSkinning,
}

impl ApplicationMode {
    /// Check if this is pre-skinning mode.
    #[inline]
    pub fn is_pre_skinning(&self) -> bool {
        matches!(self, ApplicationMode::PreSkinning)
    }

    /// Check if this is post-skinning mode.
    #[inline]
    pub fn is_post_skinning(&self) -> bool {
        matches!(self, ApplicationMode::PostSkinning)
    }

    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            ApplicationMode::PreSkinning => "Pre-Skinning",
            ApplicationMode::PostSkinning => "Post-Skinning",
        }
    }
}

// ---------------------------------------------------------------------------
// VertexDelta
// ---------------------------------------------------------------------------

/// Vertex delta for blend shape deformation.
///
/// Represents the change to apply to a single vertex when a blend shape
/// is active. Stores position, normal, and optionally tangent deltas.
///
/// # Memory Layout
///
/// Position and normal are always present (24 bytes).
/// Tangent is optional to save memory for shapes that don't affect it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VertexDelta {
    /// Position delta (offset from bind pose position).
    pub position: Vec3,
    /// Normal delta (offset from bind pose normal).
    pub normal: Vec3,
    /// Tangent delta (offset from bind pose tangent), if applicable.
    pub tangent: Option<Vec3>,
}

impl VertexDelta {
    /// Create a zero delta (no change).
    pub const ZERO: Self = Self {
        position: Vec3::ZERO,
        normal: Vec3::ZERO,
        tangent: None,
    };

    /// Create a delta with only position change.
    #[inline]
    pub fn position(position: Vec3) -> Self {
        Self {
            position,
            normal: Vec3::ZERO,
            tangent: None,
        }
    }

    /// Create a delta with position and normal change.
    #[inline]
    pub fn position_normal(position: Vec3, normal: Vec3) -> Self {
        Self {
            position,
            normal,
            tangent: None,
        }
    }

    /// Create a delta with all components.
    #[inline]
    pub fn new(position: Vec3, normal: Vec3, tangent: Option<Vec3>) -> Self {
        Self {
            position,
            normal,
            tangent,
        }
    }

    /// Create a delta with all components including tangent.
    #[inline]
    pub fn full(position: Vec3, normal: Vec3, tangent: Vec3) -> Self {
        Self {
            position,
            normal,
            tangent: Some(tangent),
        }
    }

    /// Check if this delta has any effect.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.position.length_squared() < f32::EPSILON
            && self.normal.length_squared() < f32::EPSILON
            && self.tangent.map_or(true, |t| t.length_squared() < f32::EPSILON)
    }

    /// Check if this delta has a tangent component.
    #[inline]
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Scale the delta by a weight factor.
    #[inline]
    pub fn scaled(&self, weight: f32) -> Self {
        Self {
            position: self.position * weight,
            normal: self.normal * weight,
            tangent: self.tangent.map(|t| t * weight),
        }
    }

    /// Add another delta to this one (for combining multiple shapes).
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            position: self.position + other.position,
            normal: self.normal + other.normal,
            tangent: match (self.tangent, other.tangent) {
                (Some(a), Some(b)) => Some(a + b),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
        }
    }

    /// Linearly interpolate between two deltas.
    #[inline]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(other.position, t),
            normal: self.normal.lerp(other.normal, t),
            tangent: match (self.tangent, other.tangent) {
                (Some(a), Some(b)) => Some(a.lerp(b, t)),
                (Some(a), None) => Some(a * (1.0 - t)),
                (None, Some(b)) => Some(b * t),
                (None, None) => None,
            },
        }
    }
}

impl Default for VertexDelta {
    fn default() -> Self {
        Self::ZERO
    }
}

// ---------------------------------------------------------------------------
// SparseVertexDelta
// ---------------------------------------------------------------------------

/// Sparse vertex delta with vertex index for efficient storage.
///
/// Used internally to store only affected vertices rather than
/// allocating deltas for every vertex in the mesh.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseVertexDelta {
    /// Index of the affected vertex in the mesh.
    pub vertex_index: u32,
    /// The delta to apply.
    pub delta: VertexDelta,
}

impl SparseVertexDelta {
    /// Create a new sparse delta.
    #[inline]
    pub fn new(vertex_index: u32, delta: VertexDelta) -> Self {
        Self {
            vertex_index,
            delta,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuDeltaEntry
// ---------------------------------------------------------------------------

/// GPU-aligned delta entry for compute shader processing.
///
/// Packed format for efficient GPU memory access:
/// - 4 bytes: vertex index
/// - 12 bytes: position delta
/// - 12 bytes: normal delta
/// - 4 bytes: tangent flag + padding
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDeltaEntry {
    /// Vertex index in the mesh.
    pub vertex_index: u32,
    /// Position delta (x, y, z).
    pub position: [f32; 3],
    /// Normal delta (x, y, z).
    pub normal: [f32; 3],
    /// Flags: bit 0 = has tangent (remaining bits reserved).
    pub flags: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<GpuDeltaEntry>() == GPU_DELTA_ENTRY_SIZE);

impl GpuDeltaEntry {
    /// Create from a sparse delta.
    pub fn from_sparse(sparse: &SparseVertexDelta) -> Self {
        Self {
            vertex_index: sparse.vertex_index,
            position: sparse.delta.position.to_array(),
            normal: sparse.delta.normal.to_array(),
            flags: if sparse.delta.has_tangent() { 1 } else { 0 },
        }
    }

    /// Create a zero entry (for padding).
    pub const fn zero() -> Self {
        Self {
            vertex_index: 0,
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 0.0],
            flags: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuTangentEntry
// ---------------------------------------------------------------------------

/// Separate tangent delta entry for shapes that need tangent deltas.
///
/// Stored in a separate buffer to avoid wasting memory for shapes
/// that don't modify tangents.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTangentEntry {
    /// Vertex index (matches GpuDeltaEntry).
    pub vertex_index: u32,
    /// Tangent delta (x, y, z).
    pub tangent: [f32; 3],
}

impl GpuTangentEntry {
    /// Create from a sparse delta with tangent.
    pub fn from_sparse(sparse: &SparseVertexDelta) -> Option<Self> {
        sparse.delta.tangent.map(|t| Self {
            vertex_index: sparse.vertex_index,
            tangent: t.to_array(),
        })
    }
}

// ---------------------------------------------------------------------------
// BlendShapeTarget
// ---------------------------------------------------------------------------

/// A single blend shape target (morph target).
///
/// Contains the vertex deltas and metadata for one shape.
/// Shapes are combined using weighted linear blending.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlendShapeTarget {
    /// Human-readable name (e.g., "smile", "blink_L", "jaw_open").
    pub name: String,
    /// Sparse vertex deltas (only affected vertices).
    pub deltas: Vec<SparseVertexDelta>,
    /// Affected vertex indices for quick lookup.
    pub affected_vertices: Vec<u32>,
    /// Face region this shape primarily affects (for masking).
    pub face_region: Option<FaceRegion>,
    /// Body region this shape primarily affects (for masking).
    pub body_region: Option<BodyRegion>,
    /// When to apply this shape in the skinning pipeline.
    pub application_mode: ApplicationMode,
    /// Category for organization (e.g., "viseme", "expression", "corrective").
    pub category: Option<String>,
    /// Minimum weight clamp (usually 0.0).
    pub min_weight: f32,
    /// Maximum weight clamp (usually 1.0).
    pub max_weight: f32,
}

impl BlendShapeTarget {
    /// Create a new empty blend shape target.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            deltas: Vec::new(),
            affected_vertices: Vec::new(),
            face_region: None,
            body_region: None,
            application_mode: ApplicationMode::PreSkinning,
            category: None,
            min_weight: 0.0,
            max_weight: 1.0,
        }
    }

    /// Builder: add a vertex delta.
    pub fn with_delta(mut self, vertex_index: u32, delta: VertexDelta) -> Self {
        self.add_delta(vertex_index, delta);
        self
    }

    /// Builder: set face region.
    pub fn with_region(mut self, region: FaceRegion) -> Self {
        self.face_region = Some(region);
        self
    }

    /// Builder: set body region.
    pub fn with_body_region(mut self, region: BodyRegion) -> Self {
        self.body_region = Some(region);
        self
    }

    /// Builder: set application mode.
    pub fn with_mode(mut self, mode: ApplicationMode) -> Self {
        self.application_mode = mode;
        self
    }

    /// Builder: set category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Builder: set weight range.
    pub fn with_weight_range(mut self, min: f32, max: f32) -> Self {
        self.min_weight = min;
        self.max_weight = max;
        self
    }

    /// Add a vertex delta to this shape.
    pub fn add_delta(&mut self, vertex_index: u32, delta: VertexDelta) {
        // Skip zero deltas
        if delta.is_zero() {
            return;
        }

        // Check if vertex already has a delta
        if let Some(existing) = self
            .deltas
            .iter_mut()
            .find(|d| d.vertex_index == vertex_index)
        {
            // Combine with existing delta
            existing.delta = existing.delta.add(&delta);
        } else {
            // Add new delta
            self.deltas.push(SparseVertexDelta::new(vertex_index, delta));
            self.affected_vertices.push(vertex_index);
        }
    }

    /// Remove a vertex delta.
    pub fn remove_delta(&mut self, vertex_index: u32) -> bool {
        if let Some(pos) = self
            .deltas
            .iter()
            .position(|d| d.vertex_index == vertex_index)
        {
            self.deltas.remove(pos);
            self.affected_vertices.retain(|&v| v != vertex_index);
            true
        } else {
            false
        }
    }

    /// Get the delta for a specific vertex.
    pub fn get_delta(&self, vertex_index: u32) -> Option<&VertexDelta> {
        self.deltas
            .iter()
            .find(|d| d.vertex_index == vertex_index)
            .map(|d| &d.delta)
    }

    /// Check if a vertex is affected by this shape.
    #[inline]
    pub fn affects_vertex(&self, vertex_index: u32) -> bool {
        self.affected_vertices.contains(&vertex_index)
    }

    /// Get the number of affected vertices.
    #[inline]
    pub fn affected_count(&self) -> usize {
        self.deltas.len()
    }

    /// Check if this shape is empty (no deltas).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    /// Check if this shape affects the given face region.
    #[inline]
    pub fn is_face_region(&self, region: FaceRegion) -> bool {
        self.face_region == Some(region)
    }

    /// Check if this shape affects the given body region.
    #[inline]
    pub fn is_body_region(&self, region: BodyRegion) -> bool {
        self.body_region == Some(region)
    }

    /// Check if this shape should be applied pre-skinning.
    #[inline]
    pub fn is_pre_skinning(&self) -> bool {
        self.application_mode.is_pre_skinning()
    }

    /// Check if this shape should be applied post-skinning.
    #[inline]
    pub fn is_post_skinning(&self) -> bool {
        self.application_mode.is_post_skinning()
    }

    /// Clamp a weight value to this shape's valid range.
    #[inline]
    pub fn clamp_weight(&self, weight: f32) -> f32 {
        weight.clamp(self.min_weight, self.max_weight)
    }

    /// Sort deltas by vertex index for better cache coherence.
    pub fn sort_deltas(&mut self) {
        self.deltas.sort_by_key(|d| d.vertex_index);
        self.affected_vertices.sort();
    }

    /// Prepare GPU delta entries for this shape.
    pub fn prepare_gpu_deltas(&self) -> Vec<GpuDeltaEntry> {
        self.deltas.iter().map(GpuDeltaEntry::from_sparse).collect()
    }

    /// Prepare GPU tangent entries (only for deltas with tangents).
    pub fn prepare_gpu_tangents(&self) -> Vec<GpuTangentEntry> {
        self.deltas
            .iter()
            .filter_map(GpuTangentEntry::from_sparse)
            .collect()
    }

    /// Check if any deltas have tangent components.
    pub fn has_tangent_deltas(&self) -> bool {
        self.deltas.iter().any(|d| d.delta.has_tangent())
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.name.len()
            + self.deltas.len() * mem::size_of::<SparseVertexDelta>()
            + self.affected_vertices.len() * mem::size_of::<u32>()
            + self.category.as_ref().map_or(0, |c| c.len())
            + mem::size_of::<Self>()
    }
}

impl Default for BlendShapeTarget {
    fn default() -> Self {
        Self::new("default")
    }
}

// ---------------------------------------------------------------------------
// BlendShapeSet
// ---------------------------------------------------------------------------

/// Collection of blend shapes for a single mesh.
///
/// Manages multiple blend shape targets with their weights,
/// providing efficient application to vertex data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlendShapeSet {
    /// All blend shape targets.
    pub targets: Vec<BlendShapeTarget>,
    /// Current weight for each target (same length as targets).
    pub weights: Vec<f32>,
    /// Total vertex count of the target mesh.
    pub vertex_count: u32,
    /// Name-to-index lookup for fast weight setting by name.
    #[serde(skip)]
    name_lookup: HashMap<String, usize>,
    /// Cached list of active (non-zero weight) shape indices.
    #[serde(skip)]
    active_shapes: RefCell<Vec<usize>>,
    /// Flag indicating active_shapes needs rebuilding.
    #[serde(skip)]
    active_dirty: Cell<bool>,
}

impl BlendShapeSet {
    /// Create a new empty blend shape set for a mesh.
    pub fn new(vertex_count: u32) -> Self {
        Self {
            targets: Vec::new(),
            weights: Vec::new(),
            vertex_count,
            name_lookup: HashMap::new(),
            active_shapes: RefCell::new(Vec::new()),
            active_dirty: Cell::new(false),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(vertex_count: u32, shape_capacity: usize) -> Self {
        Self {
            targets: Vec::with_capacity(shape_capacity),
            weights: Vec::with_capacity(shape_capacity),
            vertex_count,
            name_lookup: HashMap::with_capacity(shape_capacity),
            active_shapes: RefCell::new(Vec::with_capacity(shape_capacity)),
            active_dirty: Cell::new(false),
        }
    }

    /// Add a blend shape target.
    ///
    /// Returns the index of the added shape.
    pub fn add_target(&mut self, target: BlendShapeTarget) -> usize {
        let index = self.targets.len();
        self.name_lookup.insert(target.name.clone(), index);
        self.targets.push(target);
        self.weights.push(DEFAULT_WEIGHT);
        index
    }

    /// Remove a blend shape target by index.
    ///
    /// Returns the removed target, or None if index is out of bounds.
    pub fn remove_target(&mut self, index: usize) -> Option<BlendShapeTarget> {
        if index >= self.targets.len() {
            return None;
        }

        let target = self.targets.remove(index);
        self.weights.remove(index);
        self.name_lookup.remove(&target.name);

        // Rebuild name lookup with corrected indices
        self.name_lookup.clear();
        for (i, t) in self.targets.iter().enumerate() {
            self.name_lookup.insert(t.name.clone(), i);
        }

        self.active_dirty.set(true);
        Some(target)
    }

    /// Get a blend shape target by index.
    #[inline]
    pub fn get_target(&self, index: usize) -> Option<&BlendShapeTarget> {
        self.targets.get(index)
    }

    /// Get a mutable blend shape target by index.
    #[inline]
    pub fn get_target_mut(&mut self, index: usize) -> Option<&mut BlendShapeTarget> {
        self.targets.get_mut(index)
    }

    /// Find a blend shape target by name.
    pub fn find_target(&self, name: &str) -> Option<(usize, &BlendShapeTarget)> {
        self.name_lookup
            .get(name)
            .map(|&idx| (idx, &self.targets[idx]))
    }

    /// Get the number of blend shape targets.
    #[inline]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Check if the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Set the weight for a blend shape by index.
    ///
    /// Weight is clamped to the shape's valid range.
    pub fn set_weight(&mut self, index: usize, weight: f32) {
        if let Some(target) = self.targets.get(index) {
            let clamped = target.clamp_weight(weight);
            if let Some(w) = self.weights.get_mut(index) {
                if (*w - clamped).abs() > WEIGHT_THRESHOLD {
                    self.active_dirty.set(true);
                }
                *w = clamped;
            }
        }
    }

    /// Set the weight for a blend shape by name.
    ///
    /// Returns true if the shape was found and weight was set.
    pub fn set_weight_by_name(&mut self, name: &str, weight: f32) -> bool {
        if let Some(&index) = self.name_lookup.get(name) {
            self.set_weight(index, weight);
            true
        } else {
            false
        }
    }

    /// Get the weight for a blend shape by index.
    #[inline]
    pub fn get_weight(&self, index: usize) -> Option<f32> {
        self.weights.get(index).copied()
    }

    /// Get the weight for a blend shape by name.
    pub fn get_weight_by_name(&self, name: &str) -> Option<f32> {
        self.name_lookup
            .get(name)
            .and_then(|&idx| self.weights.get(idx).copied())
    }

    /// Set all weights to zero.
    pub fn reset_weights(&mut self) {
        for w in &mut self.weights {
            *w = 0.0;
        }
        self.active_shapes.borrow_mut().clear();
        self.active_dirty.set(false);
    }

    /// Get the number of shapes with non-zero weights.
    pub fn active_count(&self) -> usize {
        self.ensure_active_cache();
        self.active_shapes.borrow().len()
    }

    /// Check if any shapes are active.
    pub fn has_active_shapes(&self) -> bool {
        self.active_count() > 0
    }

    /// Get indices of active shapes (returns a clone for safety).
    pub fn active_indices(&self) -> Vec<usize> {
        self.ensure_active_cache();
        self.active_shapes.borrow().clone()
    }

    /// Ensure the active shape cache is up to date.
    fn ensure_active_cache(&self) {
        if self.active_dirty.get() {
            let mut shapes = self.active_shapes.borrow_mut();
            shapes.clear();
            for (i, &w) in self.weights.iter().enumerate() {
                if w.abs() > WEIGHT_THRESHOLD {
                    shapes.push(i);
                }
            }
            self.active_dirty.set(false);
        }
    }

    /// Apply all active blend shapes to vertex positions and normals.
    ///
    /// This applies weighted linear combination of all active shape deltas.
    /// Normals are accumulated but not renormalized (caller should normalize if needed).
    pub fn apply(&self, positions: &mut [Vec3], normals: &mut [Vec3]) {
        self.ensure_active_cache();

        for &shape_idx in self.active_shapes.borrow().iter() {
            let weight = self.weights[shape_idx];
            let target = &self.targets[shape_idx];

            for sparse in &target.deltas {
                let vi = sparse.vertex_index as usize;
                if vi < positions.len() {
                    positions[vi] += sparse.delta.position * weight;
                }
                if vi < normals.len() {
                    normals[vi] += sparse.delta.normal * weight;
                }
            }
        }
    }

    /// Apply blend shapes with tangent output.
    pub fn apply_with_tangents(
        &self,
        positions: &mut [Vec3],
        normals: &mut [Vec3],
        tangents: &mut [Vec3],
    ) {
        self.ensure_active_cache();

        for &shape_idx in self.active_shapes.borrow().iter() {
            let weight = self.weights[shape_idx];
            let target = &self.targets[shape_idx];

            for sparse in &target.deltas {
                let vi = sparse.vertex_index as usize;
                if vi < positions.len() {
                    positions[vi] += sparse.delta.position * weight;
                }
                if vi < normals.len() {
                    normals[vi] += sparse.delta.normal * weight;
                }
                if let Some(t) = sparse.delta.tangent {
                    if vi < tangents.len() {
                        tangents[vi] += t * weight;
                    }
                }
            }
        }
    }

    /// Apply only shapes targeting a specific face region.
    pub fn apply_masked(&self, positions: &mut [Vec3], region: FaceRegion) {
        self.ensure_active_cache();

        for &shape_idx in self.active_shapes.borrow().iter() {
            let target = &self.targets[shape_idx];
            if target.face_region != Some(region) {
                continue;
            }

            let weight = self.weights[shape_idx];
            for sparse in &target.deltas {
                let vi = sparse.vertex_index as usize;
                if vi < positions.len() {
                    positions[vi] += sparse.delta.position * weight;
                }
            }
        }
    }

    /// Apply only shapes targeting a specific body region.
    pub fn apply_body_masked(&self, positions: &mut [Vec3], region: BodyRegion) {
        self.ensure_active_cache();

        for &shape_idx in self.active_shapes.borrow().iter() {
            let target = &self.targets[shape_idx];
            if target.body_region != Some(region) {
                continue;
            }

            let weight = self.weights[shape_idx];
            for sparse in &target.deltas {
                let vi = sparse.vertex_index as usize;
                if vi < positions.len() {
                    positions[vi] += sparse.delta.position * weight;
                }
            }
        }
    }

    /// Apply only pre-skinning shapes.
    pub fn apply_pre_skinning(&self, positions: &mut [Vec3], normals: &mut [Vec3]) {
        self.ensure_active_cache();

        for &shape_idx in self.active_shapes.borrow().iter() {
            let target = &self.targets[shape_idx];
            if !target.is_pre_skinning() {
                continue;
            }

            let weight = self.weights[shape_idx];
            for sparse in &target.deltas {
                let vi = sparse.vertex_index as usize;
                if vi < positions.len() {
                    positions[vi] += sparse.delta.position * weight;
                }
                if vi < normals.len() {
                    normals[vi] += sparse.delta.normal * weight;
                }
            }
        }
    }

    /// Apply only post-skinning shapes.
    pub fn apply_post_skinning(&self, positions: &mut [Vec3], normals: &mut [Vec3]) {
        self.ensure_active_cache();

        for &shape_idx in self.active_shapes.borrow().iter() {
            let target = &self.targets[shape_idx];
            if !target.is_post_skinning() {
                continue;
            }

            let weight = self.weights[shape_idx];
            for sparse in &target.deltas {
                let vi = sparse.vertex_index as usize;
                if vi < positions.len() {
                    positions[vi] += sparse.delta.position * weight;
                }
                if vi < normals.len() {
                    normals[vi] += sparse.delta.normal * weight;
                }
            }
        }
    }

    /// Prepare GPU buffers for compute shader processing.
    ///
    /// Returns (delta_buffer, weight_buffer) as flat f32 arrays.
    ///
    /// # Delta Buffer Layout
    ///
    /// For each active shape, deltas are packed sequentially:
    /// ```text
    /// [shape_0_delta_count, delta_0, delta_1, ..., shape_1_delta_count, ...]
    /// ```
    ///
    /// Each delta is 8 f32s: [vertex_index, pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, flags]
    ///
    /// # Weight Buffer Layout
    ///
    /// Simple array of weights: [w0, w1, w2, ...]
    pub fn prepare_gpu_buffers(&self) -> (Vec<f32>, Vec<f32>) {
        self.ensure_active_cache();

        // Estimate buffer sizes
        let active = self.active_shapes.borrow();
        let total_deltas: usize = active
            .iter()
            .map(|&i| self.targets[i].deltas.len())
            .sum();
        let mut delta_buffer = Vec::with_capacity(total_deltas * 8 + active.len());
        drop(active);
        let mut weight_buffer = Vec::with_capacity(self.weights.len());

        // Pack delta buffer
        for &shape_idx in self.active_shapes.borrow().iter() {
            let target = &self.targets[shape_idx];

            // Write delta count for this shape
            delta_buffer.push(target.deltas.len() as f32);

            // Write each delta
            for sparse in &target.deltas {
                delta_buffer.push(sparse.vertex_index as f32);
                delta_buffer.push(sparse.delta.position.x);
                delta_buffer.push(sparse.delta.position.y);
                delta_buffer.push(sparse.delta.position.z);
                delta_buffer.push(sparse.delta.normal.x);
                delta_buffer.push(sparse.delta.normal.y);
                delta_buffer.push(sparse.delta.normal.z);
                delta_buffer.push(if sparse.delta.has_tangent() {
                    1.0
                } else {
                    0.0
                });
            }
        }

        // Pack weight buffer
        weight_buffer.extend_from_slice(&self.weights);

        (delta_buffer, weight_buffer)
    }

    /// Prepare GPU delta entries for a specific shape.
    pub fn prepare_shape_gpu_deltas(&self, index: usize) -> Option<Vec<GpuDeltaEntry>> {
        self.targets.get(index).map(|t| t.prepare_gpu_deltas())
    }

    /// Get all targets in a specific category.
    pub fn targets_in_category(&self, category: &str) -> Vec<(usize, &BlendShapeTarget)> {
        self.targets
            .iter()
            .enumerate()
            .filter(|(_, t)| t.category.as_deref() == Some(category))
            .collect()
    }

    /// Get all targets affecting a specific face region.
    pub fn targets_for_face_region(&self, region: FaceRegion) -> Vec<(usize, &BlendShapeTarget)> {
        self.targets
            .iter()
            .enumerate()
            .filter(|(_, t)| t.face_region == Some(region))
            .collect()
    }

    /// Get all targets affecting a specific body region.
    pub fn targets_for_body_region(&self, region: BodyRegion) -> Vec<(usize, &BlendShapeTarget)> {
        self.targets
            .iter()
            .enumerate()
            .filter(|(_, t)| t.body_region == Some(region))
            .collect()
    }

    /// Get total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let targets_mem: usize = self.targets.iter().map(|t| t.memory_usage()).sum();
        targets_mem
            + self.weights.len() * mem::size_of::<f32>()
            + self.name_lookup.len() * (mem::size_of::<String>() + mem::size_of::<usize>())
            + self.active_shapes.borrow().capacity() * mem::size_of::<usize>()
            + mem::size_of::<Self>()
    }

    /// Sort all target deltas for better cache coherence.
    pub fn optimize(&mut self) {
        for target in &mut self.targets {
            target.sort_deltas();
        }
    }

    /// Rebuild the name lookup table (call after deserialization).
    pub fn rebuild_lookup(&mut self) {
        self.name_lookup.clear();
        for (i, target) in self.targets.iter().enumerate() {
            self.name_lookup.insert(target.name.clone(), i);
        }
        self.active_dirty.set(true);
    }
}

impl Default for BlendShapeSet {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// BlendShapeChannel
// ---------------------------------------------------------------------------

/// A named group of blend shapes that are animated together.
///
/// Useful for organizing related shapes (e.g., all visemes, all brow shapes).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlendShapeChannel {
    /// Channel name (e.g., "visemes", "expressions").
    pub name: String,
    /// Indices of shapes in this channel.
    pub shape_indices: Vec<usize>,
    /// Global multiplier for all shapes in this channel.
    pub multiplier: f32,
    /// Whether this channel is enabled.
    pub enabled: bool,
}

impl BlendShapeChannel {
    /// Create a new channel.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            shape_indices: Vec::new(),
            multiplier: 1.0,
            enabled: true,
        }
    }

    /// Add a shape to this channel.
    pub fn add_shape(&mut self, index: usize) {
        if !self.shape_indices.contains(&index) {
            self.shape_indices.push(index);
        }
    }

    /// Remove a shape from this channel.
    pub fn remove_shape(&mut self, index: usize) {
        self.shape_indices.retain(|&i| i != index);
    }

    /// Apply channel multiplier to a blend shape set.
    pub fn apply_multiplier(&self, shapes: &mut BlendShapeSet) {
        if !self.enabled {
            // Zero out all shapes in this channel
            for &idx in &self.shape_indices {
                shapes.set_weight(idx, 0.0);
            }
        } else if (self.multiplier - 1.0).abs() > f32::EPSILON {
            // Scale weights by multiplier
            for &idx in &self.shape_indices {
                if let Some(w) = shapes.get_weight(idx) {
                    shapes.set_weight(idx, w * self.multiplier);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BlendShapeAnimation
// ---------------------------------------------------------------------------

/// Keyframe for blend shape animation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BlendShapeKeyframe {
    /// Time in seconds.
    pub time: f32,
    /// Weight value at this keyframe.
    pub weight: f32,
}

impl BlendShapeKeyframe {
    /// Create a new keyframe.
    pub fn new(time: f32, weight: f32) -> Self {
        Self { time, weight }
    }
}

/// Animation track for a single blend shape.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlendShapeTrack {
    /// Name of the target blend shape.
    pub shape_name: String,
    /// Keyframes sorted by time.
    pub keyframes: Vec<BlendShapeKeyframe>,
}

impl BlendShapeTrack {
    /// Create a new track.
    pub fn new(shape_name: impl Into<String>) -> Self {
        Self {
            shape_name: shape_name.into(),
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe (maintains sorted order).
    pub fn add_keyframe(&mut self, keyframe: BlendShapeKeyframe) {
        let pos = self
            .keyframes
            .iter()
            .position(|k| k.time > keyframe.time)
            .unwrap_or(self.keyframes.len());
        self.keyframes.insert(pos, keyframe);
    }

    /// Sample the track at a given time using linear interpolation.
    pub fn sample(&self, time: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }

        // Before first keyframe
        if time <= self.keyframes[0].time {
            return self.keyframes[0].weight;
        }

        // After last keyframe
        let last = self.keyframes.last().unwrap();
        if time >= last.time {
            return last.weight;
        }

        // Find surrounding keyframes
        for i in 0..self.keyframes.len() - 1 {
            let k0 = &self.keyframes[i];
            let k1 = &self.keyframes[i + 1];

            if time >= k0.time && time <= k1.time {
                let t = (time - k0.time) / (k1.time - k0.time);
                return k0.weight + (k1.weight - k0.weight) * t;
            }
        }

        0.0
    }

    /// Get the duration of this track.
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map_or(0.0, |k| k.time)
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Create a mirrored blend shape by flipping X coordinates.
///
/// Useful for creating symmetric shapes (e.g., blink_L -> blink_R).
pub fn mirror_shape(shape: &BlendShapeTarget, new_name: impl Into<String>) -> BlendShapeTarget {
    let mut mirrored = BlendShapeTarget::new(new_name);
    mirrored.application_mode = shape.application_mode;
    mirrored.category = shape.category.clone();
    mirrored.min_weight = shape.min_weight;
    mirrored.max_weight = shape.max_weight;

    // Mirror face region if applicable
    mirrored.face_region = shape.face_region.and_then(|r| r.mirror()).or(shape.face_region);
    mirrored.body_region = shape.body_region.and_then(|r| r.mirror()).or(shape.body_region);

    // Mirror deltas (flip X coordinate)
    for sparse in &shape.deltas {
        let mirrored_delta = VertexDelta {
            position: Vec3::new(
                -sparse.delta.position.x,
                sparse.delta.position.y,
                sparse.delta.position.z,
            ),
            normal: Vec3::new(
                -sparse.delta.normal.x,
                sparse.delta.normal.y,
                sparse.delta.normal.z,
            ),
            tangent: sparse.delta.tangent.map(|t| Vec3::new(-t.x, t.y, t.z)),
        };
        mirrored.deltas.push(SparseVertexDelta::new(
            sparse.vertex_index,
            mirrored_delta,
        ));
        mirrored.affected_vertices.push(sparse.vertex_index);
    }

    mirrored
}

/// Combine two shapes into a new additive shape.
pub fn combine_shapes(
    a: &BlendShapeTarget,
    b: &BlendShapeTarget,
    new_name: impl Into<String>,
) -> BlendShapeTarget {
    let mut combined = BlendShapeTarget::new(new_name);
    combined.application_mode = a.application_mode;

    // Collect all affected vertices
    let mut vertex_deltas: HashMap<u32, VertexDelta> = HashMap::new();

    for sparse in &a.deltas {
        vertex_deltas.insert(sparse.vertex_index, sparse.delta);
    }

    for sparse in &b.deltas {
        vertex_deltas
            .entry(sparse.vertex_index)
            .and_modify(|d| *d = d.add(&sparse.delta))
            .or_insert(sparse.delta);
    }

    for (vertex_index, delta) in vertex_deltas {
        combined.add_delta(vertex_index, delta);
    }

    combined.sort_deltas();
    combined
}

/// Calculate the "strength" of a shape (maximum delta magnitude).
pub fn shape_strength(shape: &BlendShapeTarget) -> f32 {
    shape
        .deltas
        .iter()
        .map(|d| d.delta.position.length())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // VertexDelta Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertex_delta_creation() {
        let d = VertexDelta::position(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(d.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(d.normal, Vec3::ZERO);
        assert!(d.tangent.is_none());
    }

    #[test]
    fn test_vertex_delta_position_normal() {
        let d = VertexDelta::position_normal(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(d.position, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(d.normal, Vec3::new(0.0, 1.0, 0.0));
        assert!(!d.has_tangent());
    }

    #[test]
    fn test_vertex_delta_full() {
        let d = VertexDelta::full(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!(d.has_tangent());
        assert_eq!(d.tangent, Some(Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn test_vertex_delta_is_zero() {
        assert!(VertexDelta::ZERO.is_zero());
        assert!(!VertexDelta::position(Vec3::new(0.1, 0.0, 0.0)).is_zero());
    }

    #[test]
    fn test_vertex_delta_scaled() {
        let d = VertexDelta::position(Vec3::new(2.0, 4.0, 6.0));
        let scaled = d.scaled(0.5);
        assert_eq!(scaled.position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vertex_delta_add() {
        let a = VertexDelta::position(Vec3::new(1.0, 0.0, 0.0));
        let b = VertexDelta::position(Vec3::new(0.0, 1.0, 0.0));
        let sum = a.add(&b);
        assert_eq!(sum.position, Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn test_vertex_delta_add_with_tangent() {
        let a = VertexDelta::full(Vec3::X, Vec3::Y, Vec3::Z);
        let b = VertexDelta::full(Vec3::X, Vec3::Y, Vec3::Z);
        let sum = a.add(&b);
        assert_eq!(sum.tangent, Some(Vec3::new(0.0, 0.0, 2.0)));
    }

    #[test]
    fn test_vertex_delta_lerp() {
        let a = VertexDelta::position(Vec3::ZERO);
        let b = VertexDelta::position(Vec3::new(10.0, 0.0, 0.0));
        let lerped = a.lerp(&b, 0.5);
        assert_eq!(lerped.position, Vec3::new(5.0, 0.0, 0.0));
    }

    // -----------------------------------------------------------------------
    // BlendShapeTarget Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_shape_target_creation() {
        let shape = BlendShapeTarget::new("test_shape");
        assert_eq!(shape.name, "test_shape");
        assert!(shape.is_empty());
        assert_eq!(shape.affected_count(), 0);
    }

    #[test]
    fn test_blend_shape_target_builder() {
        let shape = BlendShapeTarget::new("smile")
            .with_delta(0, VertexDelta::position(Vec3::X))
            .with_delta(1, VertexDelta::position(Vec3::Y))
            .with_region(FaceRegion::Mouth)
            .with_mode(ApplicationMode::PreSkinning)
            .with_category("expression");

        assert_eq!(shape.affected_count(), 2);
        assert_eq!(shape.face_region, Some(FaceRegion::Mouth));
        assert_eq!(shape.category, Some("expression".to_string()));
        assert!(shape.is_pre_skinning());
    }

    #[test]
    fn test_blend_shape_target_add_delta() {
        let mut shape = BlendShapeTarget::new("test");
        shape.add_delta(5, VertexDelta::position(Vec3::new(1.0, 2.0, 3.0)));

        assert_eq!(shape.affected_count(), 1);
        assert!(shape.affects_vertex(5));
        assert!(!shape.affects_vertex(0));

        let delta = shape.get_delta(5).unwrap();
        assert_eq!(delta.position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_blend_shape_target_add_delta_combines() {
        let mut shape = BlendShapeTarget::new("test");
        shape.add_delta(5, VertexDelta::position(Vec3::new(1.0, 0.0, 0.0)));
        shape.add_delta(5, VertexDelta::position(Vec3::new(0.0, 1.0, 0.0)));

        assert_eq!(shape.affected_count(), 1); // Still just one vertex
        let delta = shape.get_delta(5).unwrap();
        assert_eq!(delta.position, Vec3::new(1.0, 1.0, 0.0)); // Combined
    }

    #[test]
    fn test_blend_shape_target_skip_zero_delta() {
        let mut shape = BlendShapeTarget::new("test");
        shape.add_delta(0, VertexDelta::ZERO);
        assert!(shape.is_empty());
    }

    #[test]
    fn test_blend_shape_target_remove_delta() {
        let mut shape = BlendShapeTarget::new("test")
            .with_delta(0, VertexDelta::position(Vec3::X))
            .with_delta(1, VertexDelta::position(Vec3::Y));

        assert!(shape.remove_delta(0));
        assert_eq!(shape.affected_count(), 1);
        assert!(!shape.affects_vertex(0));
        assert!(shape.affects_vertex(1));
    }

    #[test]
    fn test_blend_shape_target_weight_clamp() {
        let shape = BlendShapeTarget::new("test").with_weight_range(0.0, 1.0);

        assert_eq!(shape.clamp_weight(-0.5), 0.0);
        assert_eq!(shape.clamp_weight(0.5), 0.5);
        assert_eq!(shape.clamp_weight(1.5), 1.0);
    }

    #[test]
    fn test_blend_shape_target_custom_weight_range() {
        let shape = BlendShapeTarget::new("test").with_weight_range(-1.0, 2.0);

        assert_eq!(shape.clamp_weight(-2.0), -1.0);
        assert_eq!(shape.clamp_weight(0.0), 0.0);
        assert_eq!(shape.clamp_weight(3.0), 2.0);
    }

    #[test]
    fn test_blend_shape_target_sort_deltas() {
        let mut shape = BlendShapeTarget::new("test")
            .with_delta(5, VertexDelta::position(Vec3::X))
            .with_delta(1, VertexDelta::position(Vec3::Y))
            .with_delta(3, VertexDelta::position(Vec3::Z));

        shape.sort_deltas();

        assert_eq!(shape.deltas[0].vertex_index, 1);
        assert_eq!(shape.deltas[1].vertex_index, 3);
        assert_eq!(shape.deltas[2].vertex_index, 5);
    }

    #[test]
    fn test_blend_shape_target_face_region() {
        let shape = BlendShapeTarget::new("blink_L").with_region(FaceRegion::LeftEye);

        assert!(shape.is_face_region(FaceRegion::LeftEye));
        assert!(!shape.is_face_region(FaceRegion::RightEye));
    }

    #[test]
    fn test_blend_shape_target_body_region() {
        let shape = BlendShapeTarget::new("fist_L").with_body_region(BodyRegion::LeftHand);

        assert!(shape.is_body_region(BodyRegion::LeftHand));
        assert!(!shape.is_body_region(BodyRegion::RightHand));
    }

    #[test]
    fn test_blend_shape_target_application_mode() {
        let pre = BlendShapeTarget::new("test").with_mode(ApplicationMode::PreSkinning);
        let post = BlendShapeTarget::new("test").with_mode(ApplicationMode::PostSkinning);

        assert!(pre.is_pre_skinning());
        assert!(!pre.is_post_skinning());
        assert!(!post.is_pre_skinning());
        assert!(post.is_post_skinning());
    }

    #[test]
    fn test_blend_shape_target_has_tangent_deltas() {
        let mut shape = BlendShapeTarget::new("test");
        shape.add_delta(0, VertexDelta::position(Vec3::X));
        assert!(!shape.has_tangent_deltas());

        shape.add_delta(1, VertexDelta::full(Vec3::X, Vec3::Y, Vec3::Z));
        assert!(shape.has_tangent_deltas());
    }

    #[test]
    fn test_blend_shape_target_gpu_deltas() {
        let shape = BlendShapeTarget::new("test")
            .with_delta(0, VertexDelta::position(Vec3::new(1.0, 2.0, 3.0)))
            .with_delta(1, VertexDelta::position_normal(Vec3::X, Vec3::Y));

        let gpu_deltas = shape.prepare_gpu_deltas();
        assert_eq!(gpu_deltas.len(), 2);
        assert_eq!(gpu_deltas[0].vertex_index, 0);
        assert_eq!(gpu_deltas[0].position, [1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // BlendShapeSet Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_shape_set_creation() {
        let set = BlendShapeSet::new(1000);
        assert_eq!(set.vertex_count, 1000);
        assert!(set.is_empty());
        assert_eq!(set.target_count(), 0);
    }

    #[test]
    fn test_blend_shape_set_add_target() {
        let mut set = BlendShapeSet::new(1000);
        let idx = set.add_target(BlendShapeTarget::new("smile"));

        assert_eq!(idx, 0);
        assert_eq!(set.target_count(), 1);
        assert!(!set.is_empty());
    }

    #[test]
    fn test_blend_shape_set_multiple_targets() {
        let mut set = BlendShapeSet::new(1000);
        let idx0 = set.add_target(BlendShapeTarget::new("smile"));
        let idx1 = set.add_target(BlendShapeTarget::new("frown"));
        let idx2 = set.add_target(BlendShapeTarget::new("blink"));

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        assert_eq!(set.target_count(), 3);
    }

    #[test]
    fn test_blend_shape_set_remove_target() {
        let mut set = BlendShapeSet::new(1000);
        set.add_target(BlendShapeTarget::new("smile"));
        set.add_target(BlendShapeTarget::new("frown"));

        let removed = set.remove_target(0);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "smile");
        assert_eq!(set.target_count(), 1);
    }

    #[test]
    fn test_blend_shape_set_weight() {
        let mut set = BlendShapeSet::new(1000);
        set.add_target(BlendShapeTarget::new("smile"));

        set.set_weight(0, 0.5);
        assert_eq!(set.get_weight(0), Some(0.5));
    }

    #[test]
    fn test_blend_shape_set_weight_by_name() {
        let mut set = BlendShapeSet::new(1000);
        set.add_target(BlendShapeTarget::new("smile"));

        assert!(set.set_weight_by_name("smile", 0.75));
        assert_eq!(set.get_weight_by_name("smile"), Some(0.75));
    }

    #[test]
    fn test_blend_shape_set_weight_by_name_not_found() {
        let mut set = BlendShapeSet::new(1000);
        set.add_target(BlendShapeTarget::new("smile"));

        assert!(!set.set_weight_by_name("frown", 0.5));
        assert_eq!(set.get_weight_by_name("frown"), None);
    }

    #[test]
    fn test_blend_shape_set_weight_clamping() {
        let mut set = BlendShapeSet::new(1000);
        set.add_target(BlendShapeTarget::new("smile"));

        set.set_weight(0, 1.5);
        assert_eq!(set.get_weight(0), Some(1.0));

        set.set_weight(0, -0.5);
        assert_eq!(set.get_weight(0), Some(0.0));
    }

    #[test]
    fn test_blend_shape_set_reset_weights() {
        let mut set = BlendShapeSet::new(1000);
        set.add_target(BlendShapeTarget::new("smile"));
        set.add_target(BlendShapeTarget::new("frown"));

        set.set_weight(0, 0.5);
        set.set_weight(1, 0.8);
        set.reset_weights();

        assert_eq!(set.get_weight(0), Some(0.0));
        assert_eq!(set.get_weight(1), Some(0.0));
    }

    #[test]
    fn test_blend_shape_set_active_count() {
        let mut set = BlendShapeSet::new(1000);
        set.add_target(BlendShapeTarget::new("smile"));
        set.add_target(BlendShapeTarget::new("frown"));
        set.add_target(BlendShapeTarget::new("blink"));

        assert_eq!(set.active_count(), 0);

        set.set_weight(0, 0.5);
        assert_eq!(set.active_count(), 1);

        set.set_weight(2, 0.3);
        assert_eq!(set.active_count(), 2);

        set.set_weight(0, 0.0);
        assert_eq!(set.active_count(), 1);
    }

    #[test]
    fn test_blend_shape_set_apply_single() {
        let mut set = BlendShapeSet::new(10);
        let shape = BlendShapeTarget::new("test")
            .with_delta(0, VertexDelta::position(Vec3::new(1.0, 0.0, 0.0)));
        set.add_target(shape);
        set.set_weight(0, 0.5);

        let mut positions = vec![Vec3::ZERO; 10];
        let mut normals = vec![Vec3::Y; 10];

        set.apply(&mut positions, &mut normals);

        assert_eq!(positions[0], Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn test_blend_shape_set_apply_multiple() {
        let mut set = BlendShapeSet::new(10);

        let shape1 = BlendShapeTarget::new("x")
            .with_delta(0, VertexDelta::position(Vec3::new(2.0, 0.0, 0.0)));
        let shape2 = BlendShapeTarget::new("y")
            .with_delta(0, VertexDelta::position(Vec3::new(0.0, 4.0, 0.0)));

        set.add_target(shape1);
        set.add_target(shape2);
        set.set_weight(0, 0.5); // 0.5 * 2.0 = 1.0
        set.set_weight(1, 0.25); // 0.25 * 4.0 = 1.0

        let mut positions = vec![Vec3::ZERO; 10];
        let mut normals = vec![Vec3::Y; 10];

        set.apply(&mut positions, &mut normals);

        assert_eq!(positions[0], Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn test_blend_shape_set_apply_zero_weight() {
        let mut set = BlendShapeSet::new(10);
        let shape = BlendShapeTarget::new("test")
            .with_delta(0, VertexDelta::position(Vec3::new(1.0, 0.0, 0.0)));
        set.add_target(shape);
        set.set_weight(0, 0.0);

        let mut positions = vec![Vec3::ZERO; 10];
        let mut normals = vec![Vec3::Y; 10];

        set.apply(&mut positions, &mut normals);

        assert_eq!(positions[0], Vec3::ZERO);
    }

    #[test]
    fn test_blend_shape_set_apply_all_weights() {
        let mut set = BlendShapeSet::new(10);

        for i in 0..5 {
            let shape = BlendShapeTarget::new(format!("shape_{}", i))
                .with_delta(0, VertexDelta::position(Vec3::new(0.2, 0.0, 0.0)));
            set.add_target(shape);
            set.set_weight(i, 1.0);
        }

        let mut positions = vec![Vec3::ZERO; 10];
        let mut normals = vec![Vec3::Y; 10];

        set.apply(&mut positions, &mut normals);

        // 5 shapes * 0.2 = 1.0
        assert!((positions[0].x - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_blend_shape_set_apply_empty_shapes() {
        let mut set = BlendShapeSet::new(10);
        let shape = BlendShapeTarget::new("empty"); // No deltas
        set.add_target(shape);
        set.set_weight(0, 1.0);

        let mut positions = vec![Vec3::ZERO; 10];
        let mut normals = vec![Vec3::Y; 10];

        set.apply(&mut positions, &mut normals);

        assert_eq!(positions[0], Vec3::ZERO);
    }

    #[test]
    fn test_blend_shape_set_apply_with_tangents() {
        let mut set = BlendShapeSet::new(10);
        let shape = BlendShapeTarget::new("test")
            .with_delta(
                0,
                VertexDelta::full(
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.0, 1.0, 0.0),
                ),
            );
        set.add_target(shape);
        set.set_weight(0, 1.0);

        let mut positions = vec![Vec3::ZERO; 10];
        let mut normals = vec![Vec3::Y; 10];
        let mut tangents = vec![Vec3::X; 10];

        set.apply_with_tangents(&mut positions, &mut normals, &mut tangents);

        assert_eq!(positions[0], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(normals[0], Vec3::new(0.0, 1.0, 1.0)); // Y + Z
        assert_eq!(tangents[0], Vec3::new(1.0, 1.0, 0.0)); // X + Y
    }

    #[test]
    fn test_blend_shape_set_apply_masked() {
        let mut set = BlendShapeSet::new(10);

        let mouth_shape = BlendShapeTarget::new("smile")
            .with_delta(0, VertexDelta::position(Vec3::X))
            .with_region(FaceRegion::Mouth);

        let eye_shape = BlendShapeTarget::new("blink")
            .with_delta(1, VertexDelta::position(Vec3::Y))
            .with_region(FaceRegion::LeftEye);

        set.add_target(mouth_shape);
        set.add_target(eye_shape);
        set.set_weight(0, 1.0);
        set.set_weight(1, 1.0);

        let mut positions = vec![Vec3::ZERO; 10];

        // Apply only mouth shapes
        set.apply_masked(&mut positions, FaceRegion::Mouth);

        assert_eq!(positions[0], Vec3::X); // Mouth affected
        assert_eq!(positions[1], Vec3::ZERO); // Eye not affected
    }

    #[test]
    fn test_blend_shape_set_apply_pre_skinning() {
        let mut set = BlendShapeSet::new(10);

        let pre = BlendShapeTarget::new("pre")
            .with_delta(0, VertexDelta::position(Vec3::X))
            .with_mode(ApplicationMode::PreSkinning);

        let post = BlendShapeTarget::new("post")
            .with_delta(1, VertexDelta::position(Vec3::Y))
            .with_mode(ApplicationMode::PostSkinning);

        set.add_target(pre);
        set.add_target(post);
        set.set_weight(0, 1.0);
        set.set_weight(1, 1.0);

        let mut positions = vec![Vec3::ZERO; 10];
        let mut normals = vec![Vec3::Y; 10];

        set.apply_pre_skinning(&mut positions, &mut normals);

        assert_eq!(positions[0], Vec3::X);
        assert_eq!(positions[1], Vec3::ZERO); // Post-skinning shape not applied
    }

    #[test]
    fn test_blend_shape_set_apply_post_skinning() {
        let mut set = BlendShapeSet::new(10);

        let pre = BlendShapeTarget::new("pre")
            .with_delta(0, VertexDelta::position(Vec3::X))
            .with_mode(ApplicationMode::PreSkinning);

        let post = BlendShapeTarget::new("post")
            .with_delta(1, VertexDelta::position(Vec3::Y))
            .with_mode(ApplicationMode::PostSkinning);

        set.add_target(pre);
        set.add_target(post);
        set.set_weight(0, 1.0);
        set.set_weight(1, 1.0);

        let mut positions = vec![Vec3::ZERO; 10];
        let mut normals = vec![Vec3::Y; 10];

        set.apply_post_skinning(&mut positions, &mut normals);

        assert_eq!(positions[0], Vec3::ZERO); // Pre-skinning shape not applied
        assert_eq!(positions[1], Vec3::Y);
    }

    #[test]
    fn test_blend_shape_set_sparse_storage() {
        let mut set = BlendShapeSet::new(100000);

        // Create shape affecting only 10 vertices out of 100K
        let mut shape = BlendShapeTarget::new("sparse");
        for i in 0..10 {
            shape.add_delta(i * 10000, VertexDelta::position(Vec3::X));
        }

        set.add_target(shape);
        set.set_weight(0, 1.0);

        let mut positions = vec![Vec3::ZERO; 100000];
        let mut normals = vec![Vec3::Y; 100000];

        set.apply(&mut positions, &mut normals);

        // Only the 10 affected vertices should be modified
        for i in 0..10 {
            assert_eq!(positions[i * 10000], Vec3::X);
        }
        assert_eq!(positions[1], Vec3::ZERO);
    }

    #[test]
    fn test_blend_shape_set_gpu_buffers() {
        let mut set = BlendShapeSet::new(10);

        let shape = BlendShapeTarget::new("test")
            .with_delta(0, VertexDelta::position(Vec3::new(1.0, 2.0, 3.0)))
            .with_delta(1, VertexDelta::position(Vec3::new(4.0, 5.0, 6.0)));

        set.add_target(shape);
        set.set_weight(0, 0.5);

        let (delta_buffer, weight_buffer) = set.prepare_gpu_buffers();

        // Delta buffer: [count, v0_idx, v0_pos.x, v0_pos.y, v0_pos.z, v0_norm.x, v0_norm.y, v0_norm.z, v0_flags, ...]
        assert!(!delta_buffer.is_empty());
        assert_eq!(delta_buffer[0], 2.0); // 2 deltas

        // Weight buffer
        assert_eq!(weight_buffer.len(), 1);
        assert_eq!(weight_buffer[0], 0.5);
    }

    #[test]
    fn test_blend_shape_set_find_target() {
        let mut set = BlendShapeSet::new(10);
        set.add_target(BlendShapeTarget::new("smile"));
        set.add_target(BlendShapeTarget::new("frown"));

        let (idx, target) = set.find_target("frown").unwrap();
        assert_eq!(idx, 1);
        assert_eq!(target.name, "frown");

        assert!(set.find_target("nonexistent").is_none());
    }

    #[test]
    fn test_blend_shape_set_targets_in_category() {
        let mut set = BlendShapeSet::new(10);
        set.add_target(BlendShapeTarget::new("smile").with_category("expression"));
        set.add_target(BlendShapeTarget::new("frown").with_category("expression"));
        set.add_target(BlendShapeTarget::new("aa").with_category("viseme"));

        let expressions = set.targets_in_category("expression");
        assert_eq!(expressions.len(), 2);

        let visemes = set.targets_in_category("viseme");
        assert_eq!(visemes.len(), 1);
    }

    #[test]
    fn test_blend_shape_set_targets_for_face_region() {
        let mut set = BlendShapeSet::new(10);
        set.add_target(BlendShapeTarget::new("blink_L").with_region(FaceRegion::LeftEye));
        set.add_target(BlendShapeTarget::new("blink_R").with_region(FaceRegion::RightEye));
        set.add_target(BlendShapeTarget::new("smile").with_region(FaceRegion::Mouth));

        let eyes = set.targets_for_face_region(FaceRegion::LeftEye);
        assert_eq!(eyes.len(), 1);
        assert_eq!(eyes[0].1.name, "blink_L");
    }

    #[test]
    fn test_blend_shape_set_optimize() {
        let mut set = BlendShapeSet::new(10);
        let shape = BlendShapeTarget::new("test")
            .with_delta(5, VertexDelta::position(Vec3::X))
            .with_delta(1, VertexDelta::position(Vec3::Y))
            .with_delta(3, VertexDelta::position(Vec3::Z));

        set.add_target(shape);
        set.optimize();

        let target = set.get_target(0).unwrap();
        assert_eq!(target.deltas[0].vertex_index, 1);
        assert_eq!(target.deltas[1].vertex_index, 3);
        assert_eq!(target.deltas[2].vertex_index, 5);
    }

    #[test]
    fn test_blend_shape_set_rebuild_lookup() {
        let mut set = BlendShapeSet::new(10);
        set.add_target(BlendShapeTarget::new("smile"));
        set.add_target(BlendShapeTarget::new("frown"));

        // Simulate deserialization clearing the lookup
        set.name_lookup.clear();

        // Rebuild should restore functionality
        set.rebuild_lookup();

        assert!(set.set_weight_by_name("smile", 0.5));
        assert_eq!(set.get_weight_by_name("smile"), Some(0.5));
    }

    // -----------------------------------------------------------------------
    // FaceRegion Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_region_all() {
        let all = FaceRegion::all();
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn test_face_region_mirror() {
        assert_eq!(FaceRegion::LeftEye.mirror(), Some(FaceRegion::RightEye));
        assert_eq!(FaceRegion::RightEye.mirror(), Some(FaceRegion::LeftEye));
        assert_eq!(FaceRegion::Mouth.mirror(), None);
    }

    #[test]
    fn test_face_region_is_eye() {
        assert!(FaceRegion::LeftEye.is_eye());
        assert!(FaceRegion::RightEye.is_eye());
        assert!(!FaceRegion::Mouth.is_eye());
    }

    #[test]
    fn test_face_region_is_brow() {
        assert!(FaceRegion::LeftBrow.is_brow());
        assert!(FaceRegion::RightBrow.is_brow());
        assert!(!FaceRegion::Jaw.is_brow());
    }

    // -----------------------------------------------------------------------
    // BodyRegion Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_body_region_all() {
        let all = BodyRegion::all();
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn test_body_region_mirror() {
        assert_eq!(BodyRegion::LeftHand.mirror(), Some(BodyRegion::RightHand));
        assert_eq!(BodyRegion::RightFoot.mirror(), Some(BodyRegion::LeftFoot));
        assert_eq!(BodyRegion::Torso.mirror(), None);
    }

    // -----------------------------------------------------------------------
    // ApplicationMode Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_application_mode_default() {
        let mode = ApplicationMode::default();
        assert!(mode.is_pre_skinning());
    }

    #[test]
    fn test_application_mode_checks() {
        assert!(ApplicationMode::PreSkinning.is_pre_skinning());
        assert!(!ApplicationMode::PreSkinning.is_post_skinning());
        assert!(ApplicationMode::PostSkinning.is_post_skinning());
        assert!(!ApplicationMode::PostSkinning.is_pre_skinning());
    }

    // -----------------------------------------------------------------------
    // GPU Data Structure Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_delta_entry_size() {
        assert_eq!(std::mem::size_of::<GpuDeltaEntry>(), 32);
    }

    #[test]
    fn test_gpu_delta_entry_from_sparse() {
        let sparse = SparseVertexDelta::new(42, VertexDelta::position(Vec3::new(1.0, 2.0, 3.0)));
        let gpu = GpuDeltaEntry::from_sparse(&sparse);

        assert_eq!(gpu.vertex_index, 42);
        assert_eq!(gpu.position, [1.0, 2.0, 3.0]);
        assert_eq!(gpu.normal, [0.0, 0.0, 0.0]);
        assert_eq!(gpu.flags, 0);
    }

    #[test]
    fn test_gpu_delta_entry_with_tangent() {
        let sparse = SparseVertexDelta::new(0, VertexDelta::full(Vec3::X, Vec3::Y, Vec3::Z));
        let gpu = GpuDeltaEntry::from_sparse(&sparse);

        assert_eq!(gpu.flags, 1); // Has tangent flag
    }

    // -----------------------------------------------------------------------
    // Utility Function Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mirror_shape() {
        let original = BlendShapeTarget::new("blink_L")
            .with_delta(
                0,
                VertexDelta::position_normal(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 0.0, 0.0)),
            )
            .with_region(FaceRegion::LeftEye);

        let mirrored = mirror_shape(&original, "blink_R");

        assert_eq!(mirrored.name, "blink_R");
        assert_eq!(mirrored.face_region, Some(FaceRegion::RightEye));

        let delta = mirrored.get_delta(0).unwrap();
        assert_eq!(delta.position.x, -1.0); // X flipped
        assert_eq!(delta.position.y, 2.0); // Y preserved
        assert_eq!(delta.normal.x, -0.5); // Normal X flipped
    }

    #[test]
    fn test_combine_shapes() {
        let a = BlendShapeTarget::new("a")
            .with_delta(0, VertexDelta::position(Vec3::new(1.0, 0.0, 0.0)));
        let b = BlendShapeTarget::new("b")
            .with_delta(0, VertexDelta::position(Vec3::new(0.0, 1.0, 0.0)))
            .with_delta(1, VertexDelta::position(Vec3::new(0.0, 0.0, 1.0)));

        let combined = combine_shapes(&a, &b, "combined");

        assert_eq!(combined.affected_count(), 2);

        let d0 = combined.get_delta(0).unwrap();
        assert_eq!(d0.position, Vec3::new(1.0, 1.0, 0.0));

        let d1 = combined.get_delta(1).unwrap();
        assert_eq!(d1.position, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_shape_strength() {
        let shape = BlendShapeTarget::new("test")
            .with_delta(0, VertexDelta::position(Vec3::new(1.0, 0.0, 0.0)))
            .with_delta(1, VertexDelta::position(Vec3::new(0.0, 2.0, 0.0)))
            .with_delta(2, VertexDelta::position(Vec3::new(0.0, 0.0, 0.5)));

        let strength = shape_strength(&shape);
        assert!((strength - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_shape_strength_empty() {
        let shape = BlendShapeTarget::new("empty");
        assert_eq!(shape_strength(&shape), 0.0);
    }

    // -----------------------------------------------------------------------
    // BlendShapeChannel Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_shape_channel() {
        let mut channel = BlendShapeChannel::new("expressions");
        channel.add_shape(0);
        channel.add_shape(1);

        assert_eq!(channel.shape_indices.len(), 2);

        channel.remove_shape(0);
        assert_eq!(channel.shape_indices.len(), 1);
        assert!(channel.shape_indices.contains(&1));
    }

    #[test]
    fn test_blend_shape_channel_multiplier() {
        let mut set = BlendShapeSet::new(10);
        set.add_target(BlendShapeTarget::new("a"));
        set.add_target(BlendShapeTarget::new("b"));

        set.set_weight(0, 0.5);
        set.set_weight(1, 0.8);

        let mut channel = BlendShapeChannel::new("test");
        channel.add_shape(0);
        channel.add_shape(1);
        channel.multiplier = 0.5;

        channel.apply_multiplier(&mut set);

        assert_eq!(set.get_weight(0), Some(0.25)); // 0.5 * 0.5
        assert_eq!(set.get_weight(1), Some(0.4)); // 0.8 * 0.5
    }

    #[test]
    fn test_blend_shape_channel_disabled() {
        let mut set = BlendShapeSet::new(10);
        set.add_target(BlendShapeTarget::new("a"));
        set.set_weight(0, 0.5);

        let mut channel = BlendShapeChannel::new("test");
        channel.add_shape(0);
        channel.enabled = false;

        channel.apply_multiplier(&mut set);

        assert_eq!(set.get_weight(0), Some(0.0)); // Zeroed when disabled
    }

    // -----------------------------------------------------------------------
    // BlendShapeTrack Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_shape_track() {
        let mut track = BlendShapeTrack::new("smile");
        track.add_keyframe(BlendShapeKeyframe::new(0.0, 0.0));
        track.add_keyframe(BlendShapeKeyframe::new(1.0, 1.0));

        assert_eq!(track.duration(), 1.0);
        assert_eq!(track.sample(0.0), 0.0);
        assert_eq!(track.sample(0.5), 0.5);
        assert_eq!(track.sample(1.0), 1.0);
    }

    #[test]
    fn test_blend_shape_track_before_first() {
        let mut track = BlendShapeTrack::new("test");
        track.add_keyframe(BlendShapeKeyframe::new(1.0, 0.5));
        track.add_keyframe(BlendShapeKeyframe::new(2.0, 1.0));

        assert_eq!(track.sample(0.0), 0.5); // Clamp to first keyframe
    }

    #[test]
    fn test_blend_shape_track_after_last() {
        let mut track = BlendShapeTrack::new("test");
        track.add_keyframe(BlendShapeKeyframe::new(0.0, 0.0));
        track.add_keyframe(BlendShapeKeyframe::new(1.0, 0.5));

        assert_eq!(track.sample(2.0), 0.5); // Clamp to last keyframe
    }

    #[test]
    fn test_blend_shape_track_empty() {
        let track = BlendShapeTrack::new("empty");
        assert_eq!(track.sample(0.5), 0.0);
        assert_eq!(track.duration(), 0.0);
    }

    #[test]
    fn test_blend_shape_track_insert_sorted() {
        let mut track = BlendShapeTrack::new("test");
        track.add_keyframe(BlendShapeKeyframe::new(2.0, 0.0));
        track.add_keyframe(BlendShapeKeyframe::new(0.0, 0.0));
        track.add_keyframe(BlendShapeKeyframe::new(1.0, 0.0));

        assert_eq!(track.keyframes[0].time, 0.0);
        assert_eq!(track.keyframes[1].time, 1.0);
        assert_eq!(track.keyframes[2].time, 2.0);
    }

    // -----------------------------------------------------------------------
    // Edge Case Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_vertex_count() {
        let set = BlendShapeSet::new(u32::MAX);
        assert_eq!(set.vertex_count, u32::MAX);
    }

    #[test]
    fn test_many_shapes() {
        let mut set = BlendShapeSet::new(100);
        for i in 0..200 {
            set.add_target(BlendShapeTarget::new(format!("shape_{}", i)));
        }
        assert_eq!(set.target_count(), 200);
    }

    #[test]
    fn test_apply_out_of_bounds_vertex() {
        let mut set = BlendShapeSet::new(10);
        let shape = BlendShapeTarget::new("test")
            .with_delta(999, VertexDelta::position(Vec3::X)); // Way out of bounds
        set.add_target(shape);
        set.set_weight(0, 1.0);

        let mut positions = vec![Vec3::ZERO; 10];
        let mut normals = vec![Vec3::Y; 10];

        // Should not panic, just skip out-of-bounds vertices
        set.apply(&mut positions, &mut normals);

        // All vertices should be unchanged
        for p in &positions {
            assert_eq!(*p, Vec3::ZERO);
        }
    }

    #[test]
    fn test_memory_usage() {
        let mut set = BlendShapeSet::new(1000);
        let shape = BlendShapeTarget::new("test")
            .with_delta(0, VertexDelta::position(Vec3::X));
        set.add_target(shape);

        let usage = set.memory_usage();
        assert!(usage > 0);
    }

    #[test]
    fn test_default_impls() {
        let _delta = VertexDelta::default();
        let _region = FaceRegion::default();
        let _mode = ApplicationMode::default();
        let _target = BlendShapeTarget::default();
        let _set = BlendShapeSet::default();
    }
}
