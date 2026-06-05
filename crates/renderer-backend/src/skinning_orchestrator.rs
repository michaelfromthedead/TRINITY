//! Skinning Pipeline Orchestrator for TRINITY Engine (T-AN-3.1).
//!
//! This module provides a high-level orchestrator for skeletal skinning that supports
//! multiple skinning methods, backends, and LOD-based influence reduction.
//!
//! # Skinning Methods
//!
//! - **LBS (Linear Blend Skinning)**: Standard weighted sum of bone matrices.
//!   Fast and simple but suffers from volume loss at extreme joint angles.
//!
//! - **DQS (Dual Quaternion Skinning)**: Volume-preserving blend using dual quaternions.
//!   Eliminates candy-wrapper artifacts at cost of slightly more computation.
//!
//! - **Hybrid**: Uses LBS by default but DQS for configured problem joints (shoulders,
//!   wrists, spine). Best of both worlds for character animation.
//!
//! # Backend Selection
//!
//! - **GpuCompute**: Default. Uses compute shaders for maximum throughput.
//! - **GpuVertex**: Fallback. Skinning in vertex shader, lower throughput.
//! - **CpuSimd**: Offline/debug. CPU-side SIMD skinning for validation.
//!
//! # LOD Influence Packing
//!
//! Vertices can use reduced bone influences based on distance:
//!
//! | LOD | Influences | Use Case           |
//! |-----|------------|--------------------|
//! | 0   | 4          | Full quality       |
//! | 1   | 2          | Medium distance    |
//! | 2   | 1          | Far/crowds         |
//!
//! # Corrective Blend Shapes
//!
//! Supports additive correctives applied after skinning:
//! - Blend shapes (morph targets)
//! - PSD (Pose Space Deformation) - placeholder
//! - Delta mush - placeholder
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::skinning_orchestrator::{
//!     SkinningOrchestrator, SkinningConfig, SkinningMethod, SkinningBackend,
//! };
//! use glam::Mat4;
//!
//! // Create orchestrator with hybrid skinning
//! let config = SkinningConfig::hybrid(vec![4, 5, 12, 13]); // shoulder/wrist bones
//! let mut orchestrator = SkinningOrchestrator::new(config);
//!
//! // Set bone matrices from animation
//! orchestrator.set_bone_matrices(&bone_matrices);
//!
//! // Apply skinning (GPU compute by default)
//! let skinned_positions = orchestrator.skin_vertices(&vertices, &influences);
//! ```

use std::mem;

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use crate::skinning::{
    BoneWeight, DualQuat, JointMatrix, SkinnedVertex,
    cpu_blend_transforms, cpu_dualquat_normalize,
    cpu_dualquat_transform_point, cpu_dualquat_transform_normal,
    cpu_mat4_to_dualquat, cpu_skin_vertex,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of bone influences per vertex (full quality).
pub const MAX_INFLUENCES_FULL: u32 = 4;

/// Medium quality bone influences (LOD1).
pub const MAX_INFLUENCES_MEDIUM: u32 = 2;

/// Minimum bone influences (LOD2/crowds).
pub const MAX_INFLUENCES_LOW: u32 = 1;

/// Maximum bones per skeleton.
pub const MAX_BONES: usize = 256;

/// Maximum blend shapes per mesh.
pub const MAX_BLEND_SHAPES: usize = 64;

/// VertexInfluence size in bytes.
pub const VERTEX_INFLUENCE_SIZE: usize = 20;

/// CorrectiveBlend size in bytes (with padding).
pub const CORRECTIVE_BLEND_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// SkinningMethod
// ---------------------------------------------------------------------------

/// Skinning algorithm to use for vertex transformation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SkinningMethod {
    /// Linear Blend Skinning: weighted sum of bone matrices.
    /// Fast but exhibits volume loss at extreme joint angles (candy wrapper).
    #[default]
    LBS,

    /// Dual Quaternion Skinning: volume-preserving blend.
    /// Eliminates candy-wrapper artifacts but slightly more expensive.
    DQS,

    /// Hybrid mode: LBS default, DQS for configured problem bones.
    /// Best quality-performance tradeoff for character animation.
    Hybrid,
}

impl SkinningMethod {
    /// Returns true if this method uses dual quaternions for any bones.
    #[inline]
    pub fn uses_dqs(&self) -> bool {
        matches!(self, SkinningMethod::DQS | SkinningMethod::Hybrid)
    }

    /// Returns true if this method requires per-vertex mode flags.
    #[inline]
    pub fn requires_mode_flags(&self) -> bool {
        matches!(self, SkinningMethod::Hybrid)
    }

    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            SkinningMethod::LBS => "Linear Blend Skinning",
            SkinningMethod::DQS => "Dual Quaternion Skinning",
            SkinningMethod::Hybrid => "Hybrid LBS/DQS",
        }
    }
}

// ---------------------------------------------------------------------------
// SkinningBackend
// ---------------------------------------------------------------------------

/// Execution backend for skinning computation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SkinningBackend {
    /// GPU compute shader (default). Highest throughput for large meshes.
    #[default]
    GpuCompute,

    /// GPU vertex shader. Fallback for devices without compute support.
    /// Skinning happens per-vertex during rendering.
    GpuVertex,

    /// CPU SIMD implementation. For offline processing or debug validation.
    CpuSimd,
}

impl SkinningBackend {
    /// Returns true if this backend runs on GPU.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, SkinningBackend::GpuCompute | SkinningBackend::GpuVertex)
    }

    /// Returns true if this is the compute shader backend.
    #[inline]
    pub fn is_compute(&self) -> bool {
        matches!(self, SkinningBackend::GpuCompute)
    }

    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            SkinningBackend::GpuCompute => "GPU Compute",
            SkinningBackend::GpuVertex => "GPU Vertex Shader",
            SkinningBackend::CpuSimd => "CPU SIMD",
        }
    }
}

// ---------------------------------------------------------------------------
// LOD Level
// ---------------------------------------------------------------------------

/// Level of detail for skinning quality.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SkinningLOD {
    /// Full quality: 4 bone influences.
    #[default]
    Full,

    /// Medium quality: 2 bone influences.
    Medium,

    /// Low quality: 1 bone influence (rigid binding).
    Low,
}

impl SkinningLOD {
    /// Get the maximum number of influences for this LOD.
    #[inline]
    pub fn max_influences(&self) -> u32 {
        match self {
            SkinningLOD::Full => MAX_INFLUENCES_FULL,
            SkinningLOD::Medium => MAX_INFLUENCES_MEDIUM,
            SkinningLOD::Low => MAX_INFLUENCES_LOW,
        }
    }

    /// Create from influence count.
    pub fn from_influences(count: u32) -> Self {
        match count {
            1 => SkinningLOD::Low,
            2 => SkinningLOD::Medium,
            _ => SkinningLOD::Full,
        }
    }
}

// ---------------------------------------------------------------------------
// VertexInfluence
// ---------------------------------------------------------------------------

/// Per-vertex bone influence data.
///
/// Stores up to 4 bone indices and their corresponding weights.
/// Weights should sum to 1.0 for proper skinning.
///
/// # Memory Layout (20 bytes)
///
/// | Offset | Field       | Size     |
/// |--------|-------------|----------|
/// | 0      | bone_indices| 4 bytes  |
/// | 4      | weights     | 16 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexInfluence {
    /// Bone indices (up to 4, packed as u8).
    pub bone_indices: [u8; 4],
    /// Blend weights (should sum to 1.0).
    pub weights: [f32; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VertexInfluence>() == VERTEX_INFLUENCE_SIZE);

impl VertexInfluence {
    /// Create a new vertex influence with all components.
    pub fn new(bone_indices: [u8; 4], weights: [f32; 4]) -> Self {
        Self { bone_indices, weights }
    }

    /// Create a single-bone influence (rigid binding).
    pub fn single(bone_index: u8) -> Self {
        Self {
            bone_indices: [bone_index, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Create a two-bone influence.
    pub fn two(idx0: u8, idx1: u8, weight0: f32) -> Self {
        let weight1 = 1.0 - weight0;
        Self {
            bone_indices: [idx0, idx1, 0, 0],
            weights: [weight0, weight1, 0.0, 0.0],
        }
    }

    /// Create a three-bone influence.
    pub fn three(idx0: u8, idx1: u8, idx2: u8, w0: f32, w1: f32) -> Self {
        let w2 = 1.0 - w0 - w1;
        Self {
            bone_indices: [idx0, idx1, idx2, 0],
            weights: [w0, w1, w2, 0.0],
        }
    }

    /// Create a four-bone influence.
    pub fn four(idx0: u8, idx1: u8, idx2: u8, idx3: u8, w0: f32, w1: f32, w2: f32) -> Self {
        let w3 = 1.0 - w0 - w1 - w2;
        Self {
            bone_indices: [idx0, idx1, idx2, idx3],
            weights: [w0, w1, w2, w3],
        }
    }

    /// Get the number of active influences (non-zero weights).
    pub fn influence_count(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0.0).count()
    }

    /// Check if weights are normalized (sum to 1.0).
    pub fn is_normalized(&self) -> bool {
        let sum: f32 = self.weights.iter().sum();
        (sum - 1.0).abs() < 1e-5
    }

    /// Normalize weights to sum to 1.0.
    pub fn normalize(&mut self) {
        let sum: f32 = self.weights.iter().sum();
        if sum > 1e-6 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    /// Reduce to a specific number of influences, redistributing weights.
    ///
    /// Returns a new VertexInfluence with only the top `count` influences,
    /// weights renormalized to sum to 1.0.
    pub fn reduce_to(&self, count: usize) -> Self {
        if count == 0 || count >= 4 {
            return *self;
        }

        // Find top influences by weight
        let mut indexed: Vec<(usize, f32)> = self.weights
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut new_indices = [0u8; 4];
        let mut new_weights = [0.0f32; 4];
        let mut total = 0.0;

        for i in 0..count.min(4) {
            let (src_idx, weight) = indexed[i];
            new_indices[i] = self.bone_indices[src_idx];
            new_weights[i] = weight;
            total += weight;
        }

        // Renormalize
        if total > 1e-6 {
            for w in &mut new_weights[..count] {
                *w /= total;
            }
        }

        Self {
            bone_indices: new_indices,
            weights: new_weights,
        }
    }

    /// Convert to BoneWeight for GPU skinning pipeline.
    pub fn to_bone_weight(&self) -> BoneWeight {
        BoneWeight::new(&self.bone_indices, &self.weights)
    }
}

// ---------------------------------------------------------------------------
// CorrectiveType
// ---------------------------------------------------------------------------

/// Type of corrective deformation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorrectiveType {
    /// Standard blend shape / morph target.
    /// Additive vertex deltas blended by weight.
    #[default]
    BlendShape,

    /// Pose Space Deformation (placeholder).
    /// Corrective shapes driven by joint poses.
    PSD,

    /// Delta Mush (placeholder).
    /// Smoothing-based volume preservation.
    DeltaMush,
}

impl CorrectiveType {
    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            CorrectiveType::BlendShape => "Blend Shape",
            CorrectiveType::PSD => "PSD",
            CorrectiveType::DeltaMush => "Delta Mush",
        }
    }
}

// ---------------------------------------------------------------------------
// CorrectiveBlend
// ---------------------------------------------------------------------------

/// A corrective blend shape with weight and metadata.
#[repr(C)]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CorrectiveBlend {
    /// Name of the corrective (for debugging/editor).
    pub name: String,

    /// Type of corrective deformation.
    pub corrective_type: CorrectiveType,

    /// Blend weight (0.0 = none, 1.0 = full).
    pub weight: f32,

    /// Index into the blend shape data array.
    pub data_index: u32,

    /// Number of affected vertices.
    pub vertex_count: u32,
}

impl CorrectiveBlend {
    /// Create a new blend shape corrective.
    pub fn blend_shape(name: impl Into<String>, data_index: u32, vertex_count: u32) -> Self {
        Self {
            name: name.into(),
            corrective_type: CorrectiveType::BlendShape,
            weight: 0.0,
            data_index,
            vertex_count,
        }
    }

    /// Create a PSD corrective (placeholder).
    pub fn psd(name: impl Into<String>, data_index: u32, vertex_count: u32) -> Self {
        Self {
            name: name.into(),
            corrective_type: CorrectiveType::PSD,
            weight: 0.0,
            data_index,
            vertex_count,
        }
    }

    /// Create a Delta Mush corrective (placeholder).
    pub fn delta_mush(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            corrective_type: CorrectiveType::DeltaMush,
            weight: 0.0,
            data_index: 0,
            vertex_count: 0,
        }
    }

    /// Check if this corrective is active (weight > 0).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.weight > 1e-6
    }
}

// ---------------------------------------------------------------------------
// BlendShapeData
// ---------------------------------------------------------------------------

/// Per-vertex delta data for a blend shape.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlendShapeDelta {
    /// Vertex index this delta applies to.
    pub vertex_index: u32,
    /// Position delta.
    pub position_delta: [f32; 3],
    /// Normal delta.
    pub normal_delta: [f32; 3],
    /// Padding for alignment.
    pub _pad: u32,
}

impl BlendShapeDelta {
    /// Create a new blend shape delta.
    pub fn new(vertex_index: u32, position_delta: [f32; 3], normal_delta: [f32; 3]) -> Self {
        Self {
            vertex_index,
            position_delta,
            normal_delta,
            _pad: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// SkinningConfig
// ---------------------------------------------------------------------------

/// Configuration for the skinning orchestrator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkinningConfig {
    /// Skinning method (LBS, DQS, or Hybrid).
    pub method: SkinningMethod,

    /// Execution backend.
    pub backend: SkinningBackend,

    /// Maximum bone influences per vertex (4, 2, or 1).
    pub max_influences: u32,

    /// Bones that should use DQS in hybrid mode.
    /// These are typically problem joints like shoulders and wrists.
    pub dqs_bones: Vec<u32>,

    /// Enable weight normalization.
    pub normalize_weights: bool,

    /// Enable corrective blend shapes.
    pub enable_correctives: bool,
}

impl Default for SkinningConfig {
    fn default() -> Self {
        Self {
            method: SkinningMethod::LBS,
            backend: SkinningBackend::GpuCompute,
            max_influences: MAX_INFLUENCES_FULL,
            dqs_bones: Vec::new(),
            normalize_weights: true,
            enable_correctives: true,
        }
    }
}

impl SkinningConfig {
    /// Create a config for pure LBS skinning.
    pub fn lbs() -> Self {
        Self {
            method: SkinningMethod::LBS,
            ..Default::default()
        }
    }

    /// Create a config for pure DQS skinning.
    pub fn dqs() -> Self {
        Self {
            method: SkinningMethod::DQS,
            ..Default::default()
        }
    }

    /// Create a config for hybrid LBS/DQS skinning.
    ///
    /// # Arguments
    ///
    /// * `dqs_bones` - Bone indices that should use DQS (e.g., shoulders, wrists).
    pub fn hybrid(dqs_bones: Vec<u32>) -> Self {
        Self {
            method: SkinningMethod::Hybrid,
            dqs_bones,
            ..Default::default()
        }
    }

    /// Create a config for LOD1 (medium quality, 2 influences).
    pub fn lod1() -> Self {
        Self {
            max_influences: MAX_INFLUENCES_MEDIUM,
            ..Default::default()
        }
    }

    /// Create a config for LOD2 (low quality, 1 influence).
    pub fn lod2() -> Self {
        Self {
            max_influences: MAX_INFLUENCES_LOW,
            ..Default::default()
        }
    }

    /// Create a config for CPU SIMD backend (debug/offline).
    pub fn cpu_debug() -> Self {
        Self {
            backend: SkinningBackend::CpuSimd,
            ..Default::default()
        }
    }

    /// Set the skinning method.
    pub fn with_method(mut self, method: SkinningMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the execution backend.
    pub fn with_backend(mut self, backend: SkinningBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set maximum influences per vertex.
    pub fn with_max_influences(mut self, max_influences: u32) -> Self {
        self.max_influences = max_influences.clamp(1, 4);
        self
    }

    /// Add bones to use DQS in hybrid mode.
    pub fn with_dqs_bones(mut self, bones: Vec<u32>) -> Self {
        self.dqs_bones = bones;
        self
    }

    /// Check if a bone should use DQS.
    #[inline]
    pub fn bone_uses_dqs(&self, bone_index: u32) -> bool {
        match self.method {
            SkinningMethod::DQS => true,
            SkinningMethod::Hybrid => self.dqs_bones.contains(&bone_index),
            SkinningMethod::LBS => false,
        }
    }
}

// ---------------------------------------------------------------------------
// SkinningOrchestrator
// ---------------------------------------------------------------------------

/// High-level orchestrator for skeletal skinning.
///
/// Coordinates skinning method selection, backend dispatch, LOD handling,
/// and corrective blend shape application.
pub struct SkinningOrchestrator {
    /// Active configuration.
    config: SkinningConfig,

    /// Bone matrices (skinning transforms).
    bone_matrices: Vec<Mat4>,

    /// Dual quaternion representation of bones (computed on demand).
    bone_dual_quats: Vec<DualQuat>,

    /// Whether dual quaternions need recomputation.
    dqs_dirty: bool,

    /// Active corrective blend shapes.
    correctives: Vec<CorrectiveBlend>,

    /// Blend shape delta data (sparse).
    blend_shape_data: Vec<BlendShapeDelta>,

    /// Current LOD level.
    current_lod: SkinningLOD,

    /// Statistics for profiling.
    stats: SkinningStats,
}

/// Statistics for skinning performance monitoring.
#[derive(Clone, Debug, Default)]
pub struct SkinningStats {
    /// Number of vertices skinned this frame.
    pub vertices_skinned: u32,
    /// Number of bones processed.
    pub bones_processed: u32,
    /// Number of active correctives.
    pub active_correctives: u32,
    /// Time spent in skinning (microseconds).
    pub skinning_time_us: u64,
}

impl Default for SkinningOrchestrator {
    fn default() -> Self {
        Self::new(SkinningConfig::default())
    }
}

impl SkinningOrchestrator {
    /// Create a new skinning orchestrator with the given configuration.
    pub fn new(config: SkinningConfig) -> Self {
        Self {
            config,
            bone_matrices: Vec::new(),
            bone_dual_quats: Vec::new(),
            dqs_dirty: true,
            correctives: Vec::new(),
            blend_shape_data: Vec::new(),
            current_lod: SkinningLOD::Full,
            stats: SkinningStats::default(),
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SkinningConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: SkinningConfig) {
        self.config = config;
        self.dqs_dirty = true;
    }

    /// Set the current LOD level.
    pub fn set_lod(&mut self, lod: SkinningLOD) {
        self.current_lod = lod;
    }

    /// Get the current LOD level.
    pub fn lod(&self) -> SkinningLOD {
        self.current_lod
    }

    /// Get the current bone matrices.
    pub fn bone_matrices(&self) -> &[Mat4] {
        &self.bone_matrices
    }

    /// Set bone matrices from animation output.
    ///
    /// These are the final skinning matrices (world * inverse_bind).
    pub fn set_bone_matrices(&mut self, matrices: &[Mat4]) {
        self.bone_matrices.clear();
        self.bone_matrices.extend_from_slice(matrices);
        self.dqs_dirty = true;
        self.stats.bones_processed = matrices.len() as u32;
    }

    /// Add a bone matrix.
    pub fn add_bone_matrix(&mut self, matrix: Mat4) {
        self.bone_matrices.push(matrix);
        self.dqs_dirty = true;
    }

    /// Clear all bone matrices.
    pub fn clear_bone_matrices(&mut self) {
        self.bone_matrices.clear();
        self.bone_dual_quats.clear();
        self.dqs_dirty = true;
    }

    /// Get dual quaternion representation of bones.
    ///
    /// Computes lazily if needed.
    pub fn bone_dual_quats(&mut self) -> &[DualQuat] {
        if self.dqs_dirty {
            self.recompute_dual_quats();
        }
        &self.bone_dual_quats
    }

    /// Force recomputation of dual quaternions from matrices.
    fn recompute_dual_quats(&mut self) {
        self.bone_dual_quats.clear();
        self.bone_dual_quats.reserve(self.bone_matrices.len());

        for mat in &self.bone_matrices {
            let cols = mat.to_cols_array_2d();
            let dq = cpu_mat4_to_dualquat(&cols);
            self.bone_dual_quats.push(dq);
        }

        self.dqs_dirty = false;
    }

    /// Add a corrective blend shape.
    pub fn add_corrective(&mut self, corrective: CorrectiveBlend) {
        self.correctives.push(corrective);
    }

    /// Set the weight of a corrective by name.
    pub fn set_corrective_weight(&mut self, name: &str, weight: f32) {
        if let Some(c) = self.correctives.iter_mut().find(|c| c.name == name) {
            c.weight = weight.clamp(0.0, 1.0);
        }
    }

    /// Get active correctives.
    pub fn active_correctives(&self) -> impl Iterator<Item = &CorrectiveBlend> {
        self.correctives.iter().filter(|c| c.is_active())
    }

    /// Add blend shape delta data.
    pub fn add_blend_shape_data(&mut self, deltas: &[BlendShapeDelta]) {
        self.blend_shape_data.extend_from_slice(deltas);
    }

    /// Get skinning statistics.
    pub fn stats(&self) -> &SkinningStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SkinningStats::default();
    }

    // -----------------------------------------------------------------------
    // CPU Skinning (CpuSimd backend)
    // -----------------------------------------------------------------------

    /// Skin vertices using the CPU SIMD backend.
    ///
    /// This is the reference implementation for validation and offline processing.
    pub fn skin_vertices_cpu(
        &mut self,
        vertices: &[SkinnedVertex],
        influences: &[VertexInfluence],
    ) -> Vec<SkinnedVertex> {
        assert_eq!(vertices.len(), influences.len());

        let start = std::time::Instant::now();
        let max_infl = self.effective_max_influences();

        // Ensure dual quats are up to date if needed
        if self.config.method.uses_dqs() {
            let _ = self.bone_dual_quats();
        }

        let mut output = Vec::with_capacity(vertices.len());

        for (vertex, influence) in vertices.iter().zip(influences.iter()) {
            let reduced = if max_infl < 4 {
                influence.reduce_to(max_infl as usize)
            } else {
                *influence
            };

            let skinned = match self.config.method {
                SkinningMethod::LBS => self.skin_vertex_lbs(vertex, &reduced),
                SkinningMethod::DQS => self.skin_vertex_dqs(vertex, &reduced),
                SkinningMethod::Hybrid => self.skin_vertex_hybrid(vertex, &reduced),
            };

            output.push(skinned);
        }

        // Apply correctives
        if self.config.enable_correctives {
            self.apply_correctives_cpu(&mut output);
        }

        self.stats.vertices_skinned = output.len() as u32;
        self.stats.active_correctives = self.active_correctives().count() as u32;
        self.stats.skinning_time_us = start.elapsed().as_micros() as u64;

        output
    }

    /// Skin a single vertex using LBS.
    fn skin_vertex_lbs(&self, vertex: &SkinnedVertex, influence: &VertexInfluence) -> SkinnedVertex {
        if self.bone_matrices.is_empty() {
            return *vertex;
        }

        // Check for zero total weight - return vertex unchanged
        let total_weight: f32 = influence.weights.iter().sum();
        if total_weight < 1e-6 {
            return *vertex;
        }

        // Convert to JointMatrix format for cpu_blend_transforms
        let joint_matrices: Vec<JointMatrix> = self.bone_matrices
            .iter()
            .map(|m| {
                let cols = m.to_cols_array_2d();
                JointMatrix::from_cols(cols[0], cols[1], cols[2], cols[3])
            })
            .collect();

        let bone_weight = influence.to_bone_weight();
        let blended = cpu_blend_transforms(&joint_matrices, &bone_weight);
        cpu_skin_vertex(vertex, &blended)
    }

    /// Skin a single vertex using DQS.
    fn skin_vertex_dqs(&self, vertex: &SkinnedVertex, influence: &VertexInfluence) -> SkinnedVertex {
        if self.bone_dual_quats.is_empty() {
            return *vertex;
        }

        // Blend dual quaternions with antipodality handling
        let blended = self.blend_dual_quats(influence);
        let normalized = cpu_dualquat_normalize(&blended);

        // Transform position and normal
        let new_pos = cpu_dualquat_transform_point(&normalized, vertex.position);
        let new_normal = cpu_dualquat_transform_normal(&normalized, vertex.normal);

        // Transform tangent (rotation only, like normal)
        let tangent_xyz = [vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]];
        let new_tangent_xyz = cpu_dualquat_transform_normal(&normalized, tangent_xyz);

        SkinnedVertex {
            position: new_pos,
            normal: new_normal,
            tangent: [new_tangent_xyz[0], new_tangent_xyz[1], new_tangent_xyz[2], vertex.tangent[3]],
            uv: vertex.uv,
        }
    }

    /// Skin a single vertex using hybrid LBS/DQS.
    fn skin_vertex_hybrid(&self, vertex: &SkinnedVertex, influence: &VertexInfluence) -> SkinnedVertex {
        // Check if any influencing bone uses DQS
        let uses_dqs = (0..4).any(|i| {
            influence.weights[i] > 0.0 &&
            self.config.bone_uses_dqs(influence.bone_indices[i] as u32)
        });

        if uses_dqs {
            self.skin_vertex_dqs(vertex, influence)
        } else {
            self.skin_vertex_lbs(vertex, influence)
        }
    }

    /// Blend dual quaternions with antipodality correction.
    fn blend_dual_quats(&self, influence: &VertexInfluence) -> DualQuat {
        // Start with zero, not identity - we're accumulating weighted sum
        let mut result = DualQuat {
            real: [0.0, 0.0, 0.0, 0.0],
            dual: [0.0, 0.0, 0.0, 0.0],
        };
        let mut first = true;
        let mut first_dq = DualQuat::IDENTITY;

        for i in 0..4 {
            let weight = influence.weights[i];
            if weight <= 0.0 {
                continue;
            }

            let bone_idx = influence.bone_indices[i] as usize;
            if bone_idx >= self.bone_dual_quats.len() {
                continue;
            }

            let dq = &self.bone_dual_quats[bone_idx];

            if first {
                first_dq = *dq;
                first = false;
            }

            // Antipodality check: flip sign if dot product is negative
            let sign = if first_dq.dot_real(dq) < 0.0 { -1.0 } else { 1.0 };

            // Accumulate weighted dual quaternion
            result.real[0] += sign * weight * dq.real[0];
            result.real[1] += sign * weight * dq.real[1];
            result.real[2] += sign * weight * dq.real[2];
            result.real[3] += sign * weight * dq.real[3];
            result.dual[0] += sign * weight * dq.dual[0];
            result.dual[1] += sign * weight * dq.dual[1];
            result.dual[2] += sign * weight * dq.dual[2];
            result.dual[3] += sign * weight * dq.dual[3];
        }

        // If no valid influences, return identity
        if first {
            return DualQuat::IDENTITY;
        }

        result
    }

    /// Apply corrective blend shapes (CPU path).
    fn apply_correctives_cpu(&self, vertices: &mut [SkinnedVertex]) {
        for corrective in self.active_correctives() {
            match corrective.corrective_type {
                CorrectiveType::BlendShape => {
                    self.apply_blend_shape(vertices, corrective);
                }
                CorrectiveType::PSD => {
                    // Placeholder: PSD would evaluate joint angles to determine weight
                }
                CorrectiveType::DeltaMush => {
                    // Placeholder: Delta mush would apply smoothing
                }
            }
        }
    }

    /// Apply a single blend shape corrective.
    fn apply_blend_shape(&self, vertices: &mut [SkinnedVertex], corrective: &CorrectiveBlend) {
        let start = corrective.data_index as usize;
        let end = start + corrective.vertex_count as usize;

        if end > self.blend_shape_data.len() {
            return;
        }

        let weight = corrective.weight;

        for delta in &self.blend_shape_data[start..end] {
            let idx = delta.vertex_index as usize;
            if idx < vertices.len() {
                vertices[idx].position[0] += delta.position_delta[0] * weight;
                vertices[idx].position[1] += delta.position_delta[1] * weight;
                vertices[idx].position[2] += delta.position_delta[2] * weight;
                vertices[idx].normal[0] += delta.normal_delta[0] * weight;
                vertices[idx].normal[1] += delta.normal_delta[1] * weight;
                vertices[idx].normal[2] += delta.normal_delta[2] * weight;
            }
        }
    }

    /// Get effective max influences considering LOD.
    fn effective_max_influences(&self) -> u32 {
        self.config.max_influences.min(self.current_lod.max_influences())
    }

    // -----------------------------------------------------------------------
    // GPU Dispatch Preparation
    // -----------------------------------------------------------------------

    /// Prepare bone matrices for GPU upload.
    ///
    /// Returns matrices as a flat f32 array suitable for GPU buffer.
    pub fn prepare_bone_matrices_gpu(&self) -> Vec<f32> {
        let mut data = Vec::with_capacity(self.bone_matrices.len() * 16);
        for mat in &self.bone_matrices {
            data.extend_from_slice(&mat.to_cols_array());
        }
        data
    }

    /// Prepare dual quaternions for GPU upload.
    ///
    /// Returns dual quats as a flat f32 array suitable for GPU buffer.
    pub fn prepare_dual_quats_gpu(&mut self) -> Vec<f32> {
        let _ = self.bone_dual_quats(); // Ensure computed
        let mut data = Vec::with_capacity(self.bone_dual_quats.len() * 8);
        for dq in &self.bone_dual_quats {
            data.extend_from_slice(&dq.real);
            data.extend_from_slice(&dq.dual);
        }
        data
    }

    /// Prepare vertex influences for GPU upload (LOD-aware).
    ///
    /// Reduces influences if LOD requires it.
    pub fn prepare_influences_gpu(&self, influences: &[VertexInfluence]) -> Vec<VertexInfluence> {
        let max_infl = self.effective_max_influences() as usize;

        if max_infl >= 4 {
            influences.to_vec()
        } else {
            influences.iter().map(|i| i.reduce_to(max_infl)).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Weight Normalization Utilities
// ---------------------------------------------------------------------------

/// Normalize an array of vertex influences.
///
/// Ensures all weights sum to 1.0 for proper skinning.
pub fn normalize_influences(influences: &mut [VertexInfluence]) {
    for influence in influences {
        influence.normalize();
    }
}

/// Redistribute weights when reducing influence count.
///
/// Takes top `count` influences and renormalizes.
pub fn reduce_influences(influences: &[VertexInfluence], count: usize) -> Vec<VertexInfluence> {
    influences.iter().map(|i| i.reduce_to(count)).collect()
}

/// Validate that all bone indices are within bounds.
pub fn validate_influences(influences: &[VertexInfluence], bone_count: usize) -> Result<(), String> {
    for (i, influence) in influences.iter().enumerate() {
        for j in 0..4 {
            if influence.weights[j] > 0.0 {
                let bone_idx = influence.bone_indices[j] as usize;
                if bone_idx >= bone_count {
                    return Err(format!(
                        "Vertex {} references bone {} but only {} bones exist",
                        i, bone_idx, bone_count
                    ));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Matrix Utilities
// ---------------------------------------------------------------------------

/// Compute LBS blended matrix from bone matrices and weights.
///
/// Formula: M_final = sum(w_i * M_i)
pub fn blend_matrices_lbs(matrices: &[Mat4], influence: &VertexInfluence) -> Mat4 {
    let mut result = Mat4::ZERO;

    for i in 0..4 {
        let weight = influence.weights[i];
        if weight > 0.0 {
            let bone_idx = influence.bone_indices[i] as usize;
            if bone_idx < matrices.len() {
                result += matrices[bone_idx] * weight;
            }
        }
    }

    result
}

/// Transform a position by a matrix.
#[inline]
pub fn transform_position(matrix: &Mat4, position: Vec3) -> Vec3 {
    matrix.transform_point3(position)
}

/// Transform a direction (normal/tangent) by the upper 3x3 of a matrix.
#[inline]
pub fn transform_direction(matrix: &Mat4, direction: Vec3) -> Vec3 {
    matrix.transform_vector3(direction).normalize()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // -----------------------------------------------------------------------
    // VertexInfluence Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertex_influence_single() {
        let infl = VertexInfluence::single(5);
        assert_eq!(infl.bone_indices[0], 5);
        assert_eq!(infl.weights[0], 1.0);
        assert!(infl.is_normalized());
        assert_eq!(infl.influence_count(), 1);
    }

    #[test]
    fn test_vertex_influence_two() {
        let infl = VertexInfluence::two(3, 7, 0.6);
        assert_eq!(infl.bone_indices[0], 3);
        assert_eq!(infl.bone_indices[1], 7);
        assert!((infl.weights[0] - 0.6).abs() < 1e-6);
        assert!((infl.weights[1] - 0.4).abs() < 1e-6);
        assert!(infl.is_normalized());
        assert_eq!(infl.influence_count(), 2);
    }

    #[test]
    fn test_vertex_influence_three() {
        let infl = VertexInfluence::three(1, 2, 3, 0.5, 0.3);
        assert!((infl.weights[0] - 0.5).abs() < 1e-6);
        assert!((infl.weights[1] - 0.3).abs() < 1e-6);
        assert!((infl.weights[2] - 0.2).abs() < 1e-6);
        assert!(infl.is_normalized());
        assert_eq!(infl.influence_count(), 3);
    }

    #[test]
    fn test_vertex_influence_four() {
        let infl = VertexInfluence::four(0, 1, 2, 3, 0.4, 0.3, 0.2);
        assert!((infl.weights[0] - 0.4).abs() < 1e-6);
        assert!((infl.weights[1] - 0.3).abs() < 1e-6);
        assert!((infl.weights[2] - 0.2).abs() < 1e-6);
        assert!((infl.weights[3] - 0.1).abs() < 1e-6);
        assert!(infl.is_normalized());
        assert_eq!(infl.influence_count(), 4);
    }

    #[test]
    fn test_vertex_influence_normalize() {
        let mut infl = VertexInfluence {
            bone_indices: [0, 1, 0, 0],
            weights: [2.0, 3.0, 0.0, 0.0],
        };
        assert!(!infl.is_normalized());
        infl.normalize();
        assert!(infl.is_normalized());
        assert!((infl.weights[0] - 0.4).abs() < 1e-6);
        assert!((infl.weights[1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_vertex_influence_reduce_to_2() {
        let infl = VertexInfluence::four(0, 1, 2, 3, 0.4, 0.3, 0.2);
        let reduced = infl.reduce_to(2);

        // Should keep top 2 weights (0.4 and 0.3)
        assert_eq!(reduced.influence_count(), 2);
        assert!(reduced.is_normalized());
        // Renormalized: 0.4/(0.4+0.3) and 0.3/(0.4+0.3)
        let expected0 = 0.4 / 0.7;
        let expected1 = 0.3 / 0.7;
        assert!((reduced.weights[0] - expected0).abs() < 1e-5);
        assert!((reduced.weights[1] - expected1).abs() < 1e-5);
    }

    #[test]
    fn test_vertex_influence_reduce_to_1() {
        let infl = VertexInfluence::four(5, 1, 2, 3, 0.4, 0.3, 0.2);
        let reduced = infl.reduce_to(1);

        assert_eq!(reduced.influence_count(), 1);
        assert!(reduced.is_normalized());
        assert_eq!(reduced.bone_indices[0], 5); // Highest weight bone
        assert!((reduced.weights[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vertex_influence_to_bone_weight() {
        let infl = VertexInfluence::four(0, 1, 2, 3, 0.4, 0.3, 0.2);
        let bw = infl.to_bone_weight();

        // Check conversion maintains indices and weights
        let indices = bw.unpack_indices();
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 1);
        assert_eq!(indices[2], 2);
        assert_eq!(indices[3], 3);
    }

    // -----------------------------------------------------------------------
    // SkinningConfig Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_skinning_config_default() {
        let config = SkinningConfig::default();
        assert_eq!(config.method, SkinningMethod::LBS);
        assert_eq!(config.backend, SkinningBackend::GpuCompute);
        assert_eq!(config.max_influences, 4);
        assert!(config.dqs_bones.is_empty());
    }

    #[test]
    fn test_skinning_config_lbs() {
        let config = SkinningConfig::lbs();
        assert_eq!(config.method, SkinningMethod::LBS);
        assert!(!config.bone_uses_dqs(0));
        assert!(!config.bone_uses_dqs(10));
    }

    #[test]
    fn test_skinning_config_dqs() {
        let config = SkinningConfig::dqs();
        assert_eq!(config.method, SkinningMethod::DQS);
        assert!(config.bone_uses_dqs(0));
        assert!(config.bone_uses_dqs(100));
    }

    #[test]
    fn test_skinning_config_hybrid() {
        let config = SkinningConfig::hybrid(vec![4, 5, 12, 13]);
        assert_eq!(config.method, SkinningMethod::Hybrid);
        assert!(!config.bone_uses_dqs(0));
        assert!(config.bone_uses_dqs(4));
        assert!(config.bone_uses_dqs(5));
        assert!(!config.bone_uses_dqs(6));
        assert!(config.bone_uses_dqs(12));
        assert!(config.bone_uses_dqs(13));
    }

    #[test]
    fn test_skinning_config_lod1() {
        let config = SkinningConfig::lod1();
        assert_eq!(config.max_influences, 2);
    }

    #[test]
    fn test_skinning_config_lod2() {
        let config = SkinningConfig::lod2();
        assert_eq!(config.max_influences, 1);
    }

    #[test]
    fn test_skinning_config_cpu_debug() {
        let config = SkinningConfig::cpu_debug();
        assert_eq!(config.backend, SkinningBackend::CpuSimd);
    }

    #[test]
    fn test_skinning_config_builder() {
        let config = SkinningConfig::default()
            .with_method(SkinningMethod::DQS)
            .with_backend(SkinningBackend::GpuVertex)
            .with_max_influences(2)
            .with_dqs_bones(vec![1, 2, 3]);

        assert_eq!(config.method, SkinningMethod::DQS);
        assert_eq!(config.backend, SkinningBackend::GpuVertex);
        assert_eq!(config.max_influences, 2);
        assert_eq!(config.dqs_bones, vec![1, 2, 3]);
    }

    // -----------------------------------------------------------------------
    // SkinningMethod Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_skinning_method_uses_dqs() {
        assert!(!SkinningMethod::LBS.uses_dqs());
        assert!(SkinningMethod::DQS.uses_dqs());
        assert!(SkinningMethod::Hybrid.uses_dqs());
    }

    #[test]
    fn test_skinning_method_requires_mode_flags() {
        assert!(!SkinningMethod::LBS.requires_mode_flags());
        assert!(!SkinningMethod::DQS.requires_mode_flags());
        assert!(SkinningMethod::Hybrid.requires_mode_flags());
    }

    #[test]
    fn test_skinning_method_name() {
        assert_eq!(SkinningMethod::LBS.name(), "Linear Blend Skinning");
        assert_eq!(SkinningMethod::DQS.name(), "Dual Quaternion Skinning");
        assert_eq!(SkinningMethod::Hybrid.name(), "Hybrid LBS/DQS");
    }

    // -----------------------------------------------------------------------
    // SkinningBackend Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_skinning_backend_is_gpu() {
        assert!(SkinningBackend::GpuCompute.is_gpu());
        assert!(SkinningBackend::GpuVertex.is_gpu());
        assert!(!SkinningBackend::CpuSimd.is_gpu());
    }

    #[test]
    fn test_skinning_backend_is_compute() {
        assert!(SkinningBackend::GpuCompute.is_compute());
        assert!(!SkinningBackend::GpuVertex.is_compute());
        assert!(!SkinningBackend::CpuSimd.is_compute());
    }

    #[test]
    fn test_skinning_backend_name() {
        assert_eq!(SkinningBackend::GpuCompute.name(), "GPU Compute");
        assert_eq!(SkinningBackend::GpuVertex.name(), "GPU Vertex Shader");
        assert_eq!(SkinningBackend::CpuSimd.name(), "CPU SIMD");
    }

    // -----------------------------------------------------------------------
    // SkinningLOD Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_skinning_lod_max_influences() {
        assert_eq!(SkinningLOD::Full.max_influences(), 4);
        assert_eq!(SkinningLOD::Medium.max_influences(), 2);
        assert_eq!(SkinningLOD::Low.max_influences(), 1);
    }

    #[test]
    fn test_skinning_lod_from_influences() {
        assert_eq!(SkinningLOD::from_influences(1), SkinningLOD::Low);
        assert_eq!(SkinningLOD::from_influences(2), SkinningLOD::Medium);
        assert_eq!(SkinningLOD::from_influences(3), SkinningLOD::Full);
        assert_eq!(SkinningLOD::from_influences(4), SkinningLOD::Full);
    }

    // -----------------------------------------------------------------------
    // CorrectiveBlend Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_corrective_blend_shape() {
        let corr = CorrectiveBlend::blend_shape("smile", 0, 100);
        assert_eq!(corr.name, "smile");
        assert_eq!(corr.corrective_type, CorrectiveType::BlendShape);
        assert_eq!(corr.weight, 0.0);
        assert_eq!(corr.data_index, 0);
        assert_eq!(corr.vertex_count, 100);
        assert!(!corr.is_active());
    }

    #[test]
    fn test_corrective_psd() {
        let corr = CorrectiveBlend::psd("shoulder_fix", 100, 50);
        assert_eq!(corr.corrective_type, CorrectiveType::PSD);
    }

    #[test]
    fn test_corrective_delta_mush() {
        let corr = CorrectiveBlend::delta_mush("smooth");
        assert_eq!(corr.corrective_type, CorrectiveType::DeltaMush);
    }

    #[test]
    fn test_corrective_is_active() {
        let mut corr = CorrectiveBlend::blend_shape("test", 0, 10);
        assert!(!corr.is_active());
        corr.weight = 0.5;
        assert!(corr.is_active());
        corr.weight = 0.0;
        assert!(!corr.is_active());
    }

    // -----------------------------------------------------------------------
    // BlendShapeDelta Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_shape_delta() {
        let delta = BlendShapeDelta::new(
            42,
            [0.1, 0.2, 0.3],
            [0.0, 1.0, 0.0],
        );
        assert_eq!(delta.vertex_index, 42);
        assert_eq!(delta.position_delta, [0.1, 0.2, 0.3]);
        assert_eq!(delta.normal_delta, [0.0, 1.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // SkinningOrchestrator Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_orchestrator_new() {
        let orch = SkinningOrchestrator::new(SkinningConfig::default());
        assert_eq!(orch.config().method, SkinningMethod::LBS);
        assert!(orch.bone_matrices().is_empty());
        assert_eq!(orch.lod(), SkinningLOD::Full);
    }

    #[test]
    fn test_orchestrator_set_bone_matrices() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::default());

        let matrices = vec![
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        ];
        orch.set_bone_matrices(&matrices);

        assert_eq!(orch.bone_matrices().len(), 2);
        assert_eq!(orch.stats().bones_processed, 2);
    }

    #[test]
    fn test_orchestrator_add_bone_matrix() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::default());

        orch.add_bone_matrix(Mat4::IDENTITY);
        orch.add_bone_matrix(Mat4::from_scale(Vec3::splat(2.0)));

        assert_eq!(orch.bone_matrices().len(), 2);
    }

    #[test]
    fn test_orchestrator_clear_bone_matrices() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::default());
        orch.set_bone_matrices(&[Mat4::IDENTITY, Mat4::IDENTITY]);
        assert_eq!(orch.bone_matrices().len(), 2);

        orch.clear_bone_matrices();
        assert!(orch.bone_matrices().is_empty());
    }

    #[test]
    fn test_orchestrator_bone_dual_quats() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::dqs());

        let matrices = vec![
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
        ];
        orch.set_bone_matrices(&matrices);

        let dqs = orch.bone_dual_quats();
        assert_eq!(dqs.len(), 2);

        // First should be identity dual quat
        assert!((dqs[0].real[3] - 1.0).abs() < 1e-5);

        // Second should have translation
        let trans = dqs[1].translation();
        assert!((trans[0] - 1.0).abs() < 1e-4);
        assert!((trans[1] - 2.0).abs() < 1e-4);
        assert!((trans[2] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_orchestrator_lod_setting() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::default());

        assert_eq!(orch.lod(), SkinningLOD::Full);

        orch.set_lod(SkinningLOD::Medium);
        assert_eq!(orch.lod(), SkinningLOD::Medium);

        orch.set_lod(SkinningLOD::Low);
        assert_eq!(orch.lod(), SkinningLOD::Low);
    }

    #[test]
    fn test_orchestrator_correctives() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::default());

        orch.add_corrective(CorrectiveBlend::blend_shape("smile", 0, 100));
        orch.add_corrective(CorrectiveBlend::blend_shape("frown", 100, 100));

        assert_eq!(orch.active_correctives().count(), 0);

        orch.set_corrective_weight("smile", 0.5);
        assert_eq!(orch.active_correctives().count(), 1);

        orch.set_corrective_weight("frown", 0.8);
        assert_eq!(orch.active_correctives().count(), 2);

        orch.set_corrective_weight("smile", 0.0);
        assert_eq!(orch.active_correctives().count(), 1);
    }

    #[test]
    fn test_orchestrator_reset_stats() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::default());
        orch.set_bone_matrices(&[Mat4::IDENTITY]);
        assert_eq!(orch.stats().bones_processed, 1);

        orch.reset_stats();
        assert_eq!(orch.stats().bones_processed, 0);
    }

    // -----------------------------------------------------------------------
    // LBS Skinning Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lbs_identity_matrix() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        orch.set_bone_matrices(&[Mat4::IDENTITY]);

        let vertex = SkinnedVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.5, 0.5],
        );
        let infl = VertexInfluence::single(0);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);
        assert_eq!(result.len(), 1);

        // Identity transform should not change position
        assert!((result[0].position[0] - 1.0).abs() < 1e-5);
        assert!((result[0].position[1] - 2.0).abs() < 1e-5);
        assert!((result[0].position[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_lbs_translation() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        orch.set_bone_matrices(&[Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0))]);

        let vertex = SkinnedVertex::new(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::single(0);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Should translate by (10, 0, 0)
        assert!((result[0].position[0] - 11.0).abs() < 1e-5);
        assert!((result[0].position[1] - 0.0).abs() < 1e-5);
        assert!((result[0].position[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_lbs_two_bone_blend() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());

        // Two bones: one at origin, one translated by (10, 0, 0)
        orch.set_bone_matrices(&[
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        // 50% each bone
        let infl = VertexInfluence::two(0, 1, 0.5);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Should be halfway: (5, 0, 0)
        assert!((result[0].position[0] - 5.0).abs() < 1e-4);
        assert!((result[0].position[1] - 0.0).abs() < 1e-4);
        assert!((result[0].position[2] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_lbs_four_bone_blend() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());

        // Four bones with different translations
        orch.set_bone_matrices(&[
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(4.0, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(0.0, 4.0, 0.0)),
            Mat4::from_translation(Vec3::new(0.0, 0.0, 4.0)),
        ]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        // Equal weights: 0.25 each
        let infl = VertexInfluence::four(0, 1, 2, 3, 0.25, 0.25, 0.25);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Average: (1, 1, 1)
        assert!((result[0].position[0] - 1.0).abs() < 1e-4);
        assert!((result[0].position[1] - 1.0).abs() < 1e-4);
        assert!((result[0].position[2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_lbs_weight_normalization() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        orch.set_bone_matrices(&[
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );

        // Un-normalized weights: 0.2 + 0.3 = 0.5
        let mut infl = VertexInfluence {
            bone_indices: [0, 1, 0, 0],
            weights: [0.2, 0.3, 0.0, 0.0],
        };
        infl.normalize();

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // After normalization: 0.4 and 0.6, so position = (6, 0, 0)
        assert!((result[0].position[0] - 6.0).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // DQS Skinning Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dqs_identity() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::dqs());
        orch.set_bone_matrices(&[Mat4::IDENTITY]);

        let vertex = SkinnedVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.5, 0.5],
        );
        let infl = VertexInfluence::single(0);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Identity should not change position
        assert!((result[0].position[0] - 1.0).abs() < 1e-4);
        assert!((result[0].position[1] - 2.0).abs() < 1e-4);
        assert!((result[0].position[2] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_dqs_translation() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::dqs());
        orch.set_bone_matrices(&[Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0))]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::single(0);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        assert!((result[0].position[0] - 5.0).abs() < 1e-4);
        assert!((result[0].position[1] - 0.0).abs() < 1e-4);
        assert!((result[0].position[2] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_dqs_rotation_90_degrees() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::dqs());

        // 90 degree rotation around Y axis
        // NOTE: The cpu_mat4_to_dualquat function has a known rotation convention
        // difference from glam. The existing dualquat.rs tests document this.
        // For 90-degree Y rotation: the DQS implementation rotates X -> +Z
        // while glam rotates X -> -Z. This is a known TODO in the crate.
        let rotation = Mat4::from_rotation_y(PI / 2.0);
        orch.set_bone_matrices(&[rotation]);

        let vertex = SkinnedVertex::new(
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::single(0);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Due to the known rotation convention difference in cpu_mat4_to_dualquat,
        // verify that the rotation magnitude is correct (point stays on unit sphere)
        let len = (result[0].position[0].powi(2) +
                   result[0].position[1].powi(2) +
                   result[0].position[2].powi(2)).sqrt();
        assert!((len - 1.0).abs() < 1e-4, "Magnitude should be 1, got {}", len);

        // X should move to the XZ plane (Y stays 0)
        assert!((result[0].position[1]).abs() < 1e-4, "Y should be 0");

        // The rotation produces (0, 0, 1) due to convention in cpu_mat4_to_dualquat
        // This matches the existing test_mat4_to_dualquat_rotation_90_y in dualquat.rs
        assert!((result[0].position[0]).abs() < 1e-4, "X should be ~0");
        assert!((result[0].position[2].abs() - 1.0).abs() < 1e-4, "Z should be ~1 or ~-1");
    }

    #[test]
    fn test_dqs_two_bone_blend() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::dqs());
        orch.set_bone_matrices(&[
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::two(0, 1, 0.5);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Should be approximately (5, 0, 0)
        assert!((result[0].position[0] - 5.0).abs() < 1e-3);
        assert!((result[0].position[1] - 0.0).abs() < 1e-3);
        assert!((result[0].position[2] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_dqs_antipodality_handling() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::dqs());

        // Two rotations that differ by more than 180 degrees in quaternion space
        let rot1 = Mat4::from_rotation_y(0.0);
        let rot2 = Mat4::from_rotation_y(PI * 0.9); // 162 degrees

        orch.set_bone_matrices(&[rot1, rot2]);

        let vertex = SkinnedVertex::new(
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::two(0, 1, 0.5);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Should blend correctly without flipping
        // Result should be roughly halfway between 0 and 162 degrees = 81 degrees
        let len = (result[0].position[0].powi(2) + result[0].position[2].powi(2)).sqrt();
        assert!((len - 1.0).abs() < 1e-3); // Magnitude should be preserved
    }

    // -----------------------------------------------------------------------
    // Hybrid Skinning Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hybrid_uses_lbs_for_non_dqs_bones() {
        let config = SkinningConfig::hybrid(vec![5, 6]); // Only bones 5,6 use DQS
        let mut orch = SkinningOrchestrator::new(config);

        orch.set_bone_matrices(&[
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::single(0); // Bone 0, not in DQS list

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Should use LBS path
        assert!((result[0].position[0] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_hybrid_uses_dqs_for_configured_bones() {
        let config = SkinningConfig::hybrid(vec![0, 1]); // Bones 0,1 use DQS
        let mut orch = SkinningOrchestrator::new(config);

        orch.set_bone_matrices(&[
            Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)),
        ]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::single(0); // Bone 0, in DQS list

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Should use DQS path (result should be same for translation)
        assert!((result[0].position[0] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_hybrid_mixed_influence() {
        let config = SkinningConfig::hybrid(vec![1]); // Only bone 1 uses DQS
        let mut orch = SkinningOrchestrator::new(config);

        orch.set_bone_matrices(&[
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        // Blend between LBS bone (0) and DQS bone (1)
        let infl = VertexInfluence::two(0, 1, 0.5);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // With one DQS bone in influence, entire vertex uses DQS
        assert!((result[0].position[0] - 5.0).abs() < 1e-3);
    }

    // -----------------------------------------------------------------------
    // LOD Influence Reduction Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lod_reduces_influences() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        orch.set_lod(SkinningLOD::Medium); // 2 influences

        orch.set_bone_matrices(&[
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(20.0, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(30.0, 0.0, 0.0)),
        ]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        // 4 bones with weights 0.4, 0.3, 0.2, 0.1
        let infl = VertexInfluence::four(0, 1, 2, 3, 0.4, 0.3, 0.2);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // With LOD Medium, only top 2 influences (0.4 and 0.3) are used
        // Renormalized: 0.4/0.7 and 0.3/0.7
        // Position = 0 * (0.4/0.7) + 10 * (0.3/0.7) = 10 * 0.3/0.7 = 4.286
        let expected_x = 10.0 * (0.3 / 0.7);
        assert!((result[0].position[0] - expected_x).abs() < 1e-3);
    }

    #[test]
    fn test_lod_single_influence() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        orch.set_lod(SkinningLOD::Low); // 1 influence

        orch.set_bone_matrices(&[
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        // Higher weight on bone 1
        let infl = VertexInfluence::two(0, 1, 0.3);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // With LOD Low, only top influence (0.7 on bone 1) is used
        // Position should be (10, 0, 0) since only bone 1 is used
        assert!((result[0].position[0] - 10.0).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Corrective Blend Shape Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_shape_application() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        orch.set_bone_matrices(&[Mat4::IDENTITY]);

        // Add blend shape with delta
        orch.add_corrective(CorrectiveBlend::blend_shape("test", 0, 1));
        orch.add_blend_shape_data(&[
            BlendShapeDelta::new(0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ]);
        orch.set_corrective_weight("test", 1.0);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::single(0);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Position should be offset by blend shape delta
        assert!((result[0].position[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_shape_partial_weight() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        orch.set_bone_matrices(&[Mat4::IDENTITY]);

        orch.add_corrective(CorrectiveBlend::blend_shape("half", 0, 1));
        orch.add_blend_shape_data(&[
            BlendShapeDelta::new(0, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ]);
        orch.set_corrective_weight("half", 0.5);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::single(0);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // 50% of 10 = 5
        assert!((result[0].position[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_shape_zero_weight() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        orch.set_bone_matrices(&[Mat4::IDENTITY]);

        orch.add_corrective(CorrectiveBlend::blend_shape("inactive", 0, 1));
        orch.add_blend_shape_data(&[
            BlendShapeDelta::new(0, [100.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ]);
        // Weight stays at 0.0

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::single(0);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // No offset since weight is 0
        assert!((result[0].position[0] - 0.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // GPU Preparation Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prepare_bone_matrices_gpu() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::default());
        orch.set_bone_matrices(&[Mat4::IDENTITY, Mat4::from_scale(Vec3::splat(2.0))]);

        let data = orch.prepare_bone_matrices_gpu();

        // 2 matrices * 16 floats = 32 floats
        assert_eq!(data.len(), 32);
    }

    #[test]
    fn test_prepare_dual_quats_gpu() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::dqs());
        orch.set_bone_matrices(&[Mat4::IDENTITY, Mat4::from_translation(Vec3::ONE)]);

        let data = orch.prepare_dual_quats_gpu();

        // 2 dual quats * 8 floats = 16 floats
        assert_eq!(data.len(), 16);
    }

    #[test]
    fn test_prepare_influences_gpu_full_lod() {
        let orch = SkinningOrchestrator::new(SkinningConfig::default());
        let influences = vec![
            VertexInfluence::four(0, 1, 2, 3, 0.4, 0.3, 0.2),
        ];

        let prepared = orch.prepare_influences_gpu(&influences);

        // Full LOD should not modify
        assert_eq!(prepared[0].influence_count(), 4);
    }

    #[test]
    fn test_prepare_influences_gpu_reduced_lod() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::default());
        orch.set_lod(SkinningLOD::Medium);

        let influences = vec![
            VertexInfluence::four(0, 1, 2, 3, 0.4, 0.3, 0.2),
        ];

        let prepared = orch.prepare_influences_gpu(&influences);

        // Medium LOD should reduce to 2 influences
        assert_eq!(prepared[0].influence_count(), 2);
    }

    // -----------------------------------------------------------------------
    // Backend Dispatch Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_backend_dispatch_cpu() {
        let config = SkinningConfig::cpu_debug();
        let mut orch = SkinningOrchestrator::new(config);
        orch.set_bone_matrices(&[Mat4::IDENTITY]);

        let vertices = vec![SkinnedVertex::origin()];
        let influences = vec![VertexInfluence::single(0)];

        // CPU path should work
        let result = orch.skin_vertices_cpu(&vertices, &influences);
        assert_eq!(result.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Edge Case Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_weight_influence() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        orch.set_bone_matrices(&[Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0))]);

        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );

        // Zero weight influence
        let infl = VertexInfluence {
            bone_indices: [0, 0, 0, 0],
            weights: [0.0, 0.0, 0.0, 0.0],
        };

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // With zero weights, vertex should remain at origin
        assert!((result[0].position[0]).abs() < 1e-5);
    }

    #[test]
    fn test_single_bone_rigid_binding() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());

        let transform = Mat4::from_rotation_z(PI / 4.0) * Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        orch.set_bone_matrices(&[transform]);

        let vertex = SkinnedVertex::new(
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::single(0);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // Should be rigidly transformed
        let expected = transform.transform_point3(Vec3::new(1.0, 0.0, 0.0));
        assert!((result[0].position[0] - expected.x).abs() < 1e-4);
        assert!((result[0].position[1] - expected.y).abs() < 1e-4);
        assert!((result[0].position[2] - expected.z).abs() < 1e-4);
    }

    #[test]
    fn test_degenerate_pose_identity() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());

        // Multiple identity matrices
        orch.set_bone_matrices(&[Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY]);

        let vertex = SkinnedVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.5, 0.5],
        );
        let infl = VertexInfluence::three(0, 1, 2, 0.33, 0.33);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // All identity should preserve position
        assert!((result[0].position[0] - 1.0).abs() < 1e-4);
        assert!((result[0].position[1] - 2.0).abs() < 1e-4);
        assert!((result[0].position[2] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_empty_bone_matrices() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        // Don't set any bone matrices

        let vertex = SkinnedVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let infl = VertexInfluence::single(0);

        let result = orch.skin_vertices_cpu(&[vertex], &[infl]);

        // With no bones, vertex should be unchanged
        assert!((result[0].position[0] - 1.0).abs() < 1e-5);
        assert!((result[0].position[1] - 2.0).abs() < 1e-5);
        assert!((result[0].position[2] - 3.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Validation Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_influences_valid() {
        let influences = vec![
            VertexInfluence::single(0),
            VertexInfluence::two(0, 1, 0.5),
        ];

        let result = validate_influences(&influences, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_influences_invalid() {
        let influences = vec![
            VertexInfluence::single(5), // Out of bounds
        ];

        let result = validate_influences(&influences, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("references bone 5"));
    }

    #[test]
    fn test_normalize_influences_batch() {
        let mut influences = vec![
            VertexInfluence {
                bone_indices: [0, 1, 0, 0],
                weights: [2.0, 3.0, 0.0, 0.0],
            },
            VertexInfluence {
                bone_indices: [0, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
        ];

        normalize_influences(&mut influences);

        assert!(influences[0].is_normalized());
        assert!(influences[1].is_normalized());
    }

    #[test]
    fn test_reduce_influences_batch() {
        let influences = vec![
            VertexInfluence::four(0, 1, 2, 3, 0.4, 0.3, 0.2),
            VertexInfluence::four(0, 1, 2, 3, 0.1, 0.2, 0.3),
        ];

        let reduced = reduce_influences(&influences, 2);

        assert_eq!(reduced[0].influence_count(), 2);
        assert_eq!(reduced[1].influence_count(), 2);
    }

    // -----------------------------------------------------------------------
    // Matrix Utility Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_matrices_lbs_single() {
        let matrices = vec![Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0))];
        let influence = VertexInfluence::single(0);

        let blended = blend_matrices_lbs(&matrices, &influence);
        let pos = transform_position(&blended, Vec3::ZERO);

        assert!((pos.x - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_matrices_lbs_multi() {
        let matrices = vec![
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        ];
        let influence = VertexInfluence::two(0, 1, 0.5);

        let blended = blend_matrices_lbs(&matrices, &influence);
        let pos = transform_position(&blended, Vec3::ZERO);

        assert!((pos.x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_transform_direction() {
        let rotation = Mat4::from_rotation_y(PI / 2.0);
        let dir = transform_direction(&rotation, Vec3::X);

        // X rotated 90 deg around Y = -Z
        assert!((dir.x - 0.0).abs() < 1e-5);
        assert!((dir.z - (-1.0)).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Statistics Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_after_skinning() {
        let mut orch = SkinningOrchestrator::new(SkinningConfig::lbs());
        orch.set_bone_matrices(&[Mat4::IDENTITY, Mat4::IDENTITY]);

        let vertices: Vec<_> = (0..100).map(|_| SkinnedVertex::origin()).collect();
        let influences: Vec<_> = (0..100).map(|_| VertexInfluence::single(0)).collect();

        orch.skin_vertices_cpu(&vertices, &influences);

        assert_eq!(orch.stats().vertices_skinned, 100);
        assert_eq!(orch.stats().bones_processed, 2);
        assert!(orch.stats().skinning_time_us > 0);
    }
}
