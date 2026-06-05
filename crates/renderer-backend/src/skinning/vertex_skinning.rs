//! GPU Vertex Skinning for TRINITY Engine (T-GPU-7.2).
//!
//! This module implements GPU-based vertex skinning using compute shaders.
//! The skinning shader applies skeletal animation transforms to mesh vertices,
//! supporting up to 4 bone influences per vertex with proper normal transformation.
//!
//! # Overview
//!
//! The skinning pipeline:
//! 1. CPU uploads skinning matrices from joint_transform output
//! 2. GPU dispatches vertex skinning compute shader with ceil(vertex_count / 256) workgroups
//! 3. Each thread transforms one vertex using weighted bone blending
//! 4. Skinned vertices are written to output buffer for rendering
//!
//! # Normal Transformation
//!
//! For correct lighting under non-uniform scale, normals are transformed using
//! the cofactor matrix (transpose of inverse). This is more expensive than a
//! simple 3x3 multiply but ensures correct results for all skeletal animations.
//!
//! # Data Layout
//!
//! The `SkinnedVertex` struct is 48 bytes:
//!
//! ```text
//! struct SkinnedVertex {
//!     position: [f32; 3],   // 12 bytes
//!     normal: [f32; 3],     // 12 bytes
//!     tangent: [f32; 4],    // 16 bytes (xyz + handedness)
//!     uv: [f32; 2],         // 8 bytes
//! }                         // Total: 48 bytes
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::skinning::{
//!     VertexSkinningPipeline, SkinnedMesh, SkinnedVertex, BoneWeight,
//! };
//!
//! // Create pipeline
//! let pipeline = VertexSkinningPipeline::new(&device);
//!
//! // Create skinned mesh from vertex/weight data
//! let mesh = SkinnedMesh::new(
//!     &device, vertices, weights, joint_count,
//!     &pipeline.bind_group_layout,
//! );
//!
//! // Each frame: upload joint matrices and dispatch
//! mesh.update_joint_matrices(&queue, &joint_matrices);
//! pipeline.dispatch(&mut encoder, &mesh);
//!
//! // Use mesh.dst_vertex_buffer for rendering
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// SkinnedVertex size in bytes (must match WGSL SourceVertex/SkinnedVertex).
pub const SKINNED_VERTEX_SIZE: usize = 48;

/// BoneWeight size in bytes (must match WGSL BoneWeight).
pub const BONE_WEIGHT_SIZE: usize = 16;

/// VertexSkinningParams size in bytes.
pub const SKINNING_PARAMS_SIZE: usize = 16;

/// JointMatrix size in bytes (4x4 f32 matrix).
pub const JOINT_MATRIX_SIZE: usize = 64;

/// Maximum bone influences per vertex.
pub const MAX_BONE_INFLUENCES: usize = 4;

/// Skinning flag: use dual quaternion blending.
pub const FLAG_DUAL_QUATERNION: u32 = 1;

/// Skinning flag: normalize weights.
pub const FLAG_NORMALIZE_WEIGHTS: u32 = 2;

// ---------------------------------------------------------------------------
// SkinnedVertex
// ---------------------------------------------------------------------------

/// GPU vertex data structure for skinning (48 bytes).
///
/// Matches the WGSL `SourceVertex` and `SkinnedVertex` struct layout.
/// Used for both input (bind pose) and output (skinned) vertices.
///
/// # Memory Layout (48 bytes)
///
/// | Offset | Field    | Size     |
/// |--------|----------|----------|
/// | 0      | position | 12 bytes |
/// | 12     | normal   | 12 bytes |
/// | 24     | tangent  | 16 bytes |
/// | 40     | uv       | 8 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    /// Object-space or skinned position.
    pub position: [f32; 3],
    /// Vertex normal (unit length).
    pub normal: [f32; 3],
    /// Vertex tangent (xyz) and handedness (w).
    pub tangent: [f32; 4],
    /// Texture coordinates.
    pub uv: [f32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<SkinnedVertex>() == SKINNED_VERTEX_SIZE);

impl SkinnedVertex {
    /// Create a new vertex with all components.
    pub fn new(
        position: [f32; 3],
        normal: [f32; 3],
        tangent: [f32; 4],
        uv: [f32; 2],
    ) -> Self {
        Self {
            position,
            normal,
            tangent,
            uv,
        }
    }

    /// Create a vertex at the origin with default normal (up).
    pub const fn origin() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        }
    }

    /// Create a vertex with just position and normal.
    pub fn from_position_normal(position: [f32; 3], normal: [f32; 3]) -> Self {
        Self {
            position,
            normal,
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        }
    }
}

impl Default for SkinnedVertex {
    fn default() -> Self {
        Self::origin()
    }
}

// ---------------------------------------------------------------------------
// BoneWeight
// ---------------------------------------------------------------------------

/// Bone weight data for a single vertex (16 bytes).
///
/// Supports up to 4 bone influences per vertex. The indices are packed into
/// a single u32 (4 x u8), and weights are stored as 3 f32s with the fourth
/// weight computed as 1 - sum(weights[0..3]).
///
/// # Memory Layout (16 bytes)
///
/// | Offset | Field   | Size     |
/// |--------|---------|----------|
/// | 0      | indices | 4 bytes  |
/// | 4      | weights | 12 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoneWeight {
    /// Packed bone indices (4 x u8).
    /// Index 0 in bits 0-7, index 1 in bits 8-15, etc.
    pub indices: u32,
    /// Blend weights for first 3 bones.
    /// Fourth weight = 1.0 - weights[0] - weights[1] - weights[2].
    pub weights: [f32; 3],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<BoneWeight>() == BONE_WEIGHT_SIZE);

impl BoneWeight {
    /// Create bone weights from indices and normalized weights.
    ///
    /// # Arguments
    ///
    /// * `indices` - Up to 4 bone indices (0-255 each).
    /// * `weights` - Up to 4 weights (must sum to 1.0 for correct results).
    ///
    /// # Panics
    ///
    /// Panics if indices or weights have more than 4 elements.
    pub fn new(indices: &[u8], weights: &[f32]) -> Self {
        assert!(
            indices.len() <= MAX_BONE_INFLUENCES,
            "Too many bone influences"
        );
        assert!(
            weights.len() <= MAX_BONE_INFLUENCES,
            "Too many bone weights"
        );

        let mut packed_indices = 0u32;
        for (i, &idx) in indices.iter().take(4).enumerate() {
            packed_indices |= (idx as u32) << (i * 8);
        }

        let mut packed_weights = [0.0f32; 3];
        for (i, &w) in weights.iter().take(3).enumerate() {
            packed_weights[i] = w;
        }

        Self {
            indices: packed_indices,
            weights: packed_weights,
        }
    }

    /// Create a single-bone weight (100% influence from one bone).
    pub fn single_bone(bone_index: u8) -> Self {
        Self {
            indices: bone_index as u32,
            weights: [1.0, 0.0, 0.0],
        }
    }

    /// Create a two-bone blend.
    pub fn two_bones(idx0: u8, idx1: u8, weight0: f32) -> Self {
        let weight1 = 1.0 - weight0;
        Self {
            indices: (idx0 as u32) | ((idx1 as u32) << 8),
            weights: [weight0, weight1, 0.0],
        }
    }

    /// Create a three-bone blend.
    pub fn three_bones(idx0: u8, idx1: u8, idx2: u8, weight0: f32, weight1: f32) -> Self {
        Self {
            indices: (idx0 as u32) | ((idx1 as u32) << 8) | ((idx2 as u32) << 16),
            weights: [weight0, weight1, 1.0 - weight0 - weight1],
        }
    }

    /// Create a four-bone blend.
    pub fn four_bones(
        idx0: u8,
        idx1: u8,
        idx2: u8,
        idx3: u8,
        weight0: f32,
        weight1: f32,
        weight2: f32,
    ) -> Self {
        Self {
            indices: (idx0 as u32)
                | ((idx1 as u32) << 8)
                | ((idx2 as u32) << 16)
                | ((idx3 as u32) << 24),
            weights: [weight0, weight1, weight2],
        }
    }

    /// Unpack bone indices.
    pub fn unpack_indices(&self) -> [u8; 4] {
        [
            (self.indices & 0xFF) as u8,
            ((self.indices >> 8) & 0xFF) as u8,
            ((self.indices >> 16) & 0xFF) as u8,
            ((self.indices >> 24) & 0xFF) as u8,
        ]
    }

    /// Get all 4 weights (fourth computed from sum).
    pub fn all_weights(&self) -> [f32; 4] {
        let w3 = (1.0 - self.weights[0] - self.weights[1] - self.weights[2]).max(0.0);
        [self.weights[0], self.weights[1], self.weights[2], w3]
    }

    /// Check if weights are normalized (sum to 1.0).
    pub fn is_normalized(&self) -> bool {
        let sum = self.weights[0] + self.weights[1] + self.weights[2];
        sum >= 0.0 && sum <= 1.0 + f32::EPSILON
    }

    /// Normalize weights to sum to 1.0.
    pub fn normalize(&mut self) {
        let w3 = (1.0 - self.weights[0] - self.weights[1] - self.weights[2]).max(0.0);
        let sum = self.weights[0] + self.weights[1] + self.weights[2] + w3;
        if sum > f32::EPSILON {
            self.weights[0] /= sum;
            self.weights[1] /= sum;
            self.weights[2] /= sum;
        }
    }

    /// Count number of active bone influences (non-zero weight).
    pub fn influence_count(&self) -> usize {
        let weights = self.all_weights();
        weights.iter().filter(|&&w| w > f32::EPSILON).count()
    }
}

impl Default for BoneWeight {
    fn default() -> Self {
        Self::single_bone(0)
    }
}

// ---------------------------------------------------------------------------
// VertexSkinningParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for skinning parameters.
///
/// Matches the WGSL `VertexSkinningParams` struct layout.
///
/// # Memory Layout (16 bytes, std140 compatible)
///
/// | Offset | Field        | Size    |
/// |--------|--------------|---------|
/// | 0      | vertex_count | 4 bytes |
/// | 4      | joint_count  | 4 bytes |
/// | 8      | flags        | 4 bytes |
/// | 12     | _padding     | 4 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexSkinningParams {
    /// Number of vertices to process.
    pub vertex_count: u32,
    /// Number of joints in the skeleton.
    pub joint_count: u32,
    /// Flags (see FLAG_* constants).
    pub flags: u32,
    /// Padding for 16-byte alignment.
    pub _padding: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VertexSkinningParams>() == SKINNING_PARAMS_SIZE);

impl VertexSkinningParams {
    /// Create skinning parameters.
    pub fn new(vertex_count: u32, joint_count: u32) -> Self {
        Self {
            vertex_count,
            joint_count,
            flags: 0,
            _padding: 0,
        }
    }

    /// Create parameters with flags.
    pub fn with_flags(vertex_count: u32, joint_count: u32, flags: u32) -> Self {
        Self {
            vertex_count,
            joint_count,
            flags,
            _padding: 0,
        }
    }

    /// Get number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.vertex_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

impl Default for VertexSkinningParams {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

// ---------------------------------------------------------------------------
// JointMatrix
// ---------------------------------------------------------------------------

/// 4x4 transformation matrix for a joint (64 bytes).
///
/// Column-major layout matching WGSL mat4x4<f32>.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct JointMatrix {
    /// Column 0 of the matrix.
    pub col0: [f32; 4],
    /// Column 1 of the matrix.
    pub col1: [f32; 4],
    /// Column 2 of the matrix.
    pub col2: [f32; 4],
    /// Column 3 of the matrix.
    pub col3: [f32; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<JointMatrix>() == JOINT_MATRIX_SIZE);

impl JointMatrix {
    /// Create from column vectors.
    pub fn from_cols(col0: [f32; 4], col1: [f32; 4], col2: [f32; 4], col3: [f32; 4]) -> Self {
        Self {
            col0,
            col1,
            col2,
            col3,
        }
    }

    /// Identity matrix.
    pub const IDENTITY: Self = Self {
        col0: [1.0, 0.0, 0.0, 0.0],
        col1: [0.0, 1.0, 0.0, 0.0],
        col2: [0.0, 0.0, 1.0, 0.0],
        col3: [0.0, 0.0, 0.0, 1.0],
    };

    /// Create a translation matrix.
    pub fn translation(x: f32, y: f32, z: f32) -> Self {
        Self {
            col0: [1.0, 0.0, 0.0, 0.0],
            col1: [0.0, 1.0, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
            col3: [x, y, z, 1.0],
        }
    }

    /// Create a uniform scale matrix.
    pub fn scale(s: f32) -> Self {
        Self {
            col0: [s, 0.0, 0.0, 0.0],
            col1: [0.0, s, 0.0, 0.0],
            col2: [0.0, 0.0, s, 0.0],
            col3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a non-uniform scale matrix.
    pub fn scale_xyz(x: f32, y: f32, z: f32) -> Self {
        Self {
            col0: [x, 0.0, 0.0, 0.0],
            col1: [0.0, y, 0.0, 0.0],
            col2: [0.0, 0.0, z, 0.0],
            col3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a rotation matrix around the X axis.
    pub fn rotation_x(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            col0: [1.0, 0.0, 0.0, 0.0],
            col1: [0.0, c, s, 0.0],
            col2: [0.0, -s, c, 0.0],
            col3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a rotation matrix around the Y axis.
    pub fn rotation_y(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            col0: [c, 0.0, -s, 0.0],
            col1: [0.0, 1.0, 0.0, 0.0],
            col2: [s, 0.0, c, 0.0],
            col3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a rotation matrix around the Z axis.
    pub fn rotation_z(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            col0: [c, s, 0.0, 0.0],
            col1: [-s, c, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
            col3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Multiply two matrices.
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = [[0.0f32; 4]; 4];
        let a = [self.col0, self.col1, self.col2, self.col3];
        let b = [other.col0, other.col1, other.col2, other.col3];

        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = a[0][j] * b[i][0]
                    + a[1][j] * b[i][1]
                    + a[2][j] * b[i][2]
                    + a[3][j] * b[i][3];
            }
        }

        Self {
            col0: result[0],
            col1: result[1],
            col2: result[2],
            col3: result[3],
        }
    }

    /// Transform a point (position with w=1).
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        [
            self.col0[0] * p[0] + self.col1[0] * p[1] + self.col2[0] * p[2] + self.col3[0],
            self.col0[1] * p[0] + self.col1[1] * p[1] + self.col2[1] * p[2] + self.col3[1],
            self.col0[2] * p[0] + self.col1[2] * p[1] + self.col2[2] * p[2] + self.col3[2],
        ]
    }

    /// Transform a vector (direction with w=0).
    pub fn transform_vector(&self, v: [f32; 3]) -> [f32; 3] {
        [
            self.col0[0] * v[0] + self.col1[0] * v[1] + self.col2[0] * v[2],
            self.col0[1] * v[0] + self.col1[1] * v[1] + self.col2[1] * v[2],
            self.col0[2] * v[0] + self.col1[2] * v[1] + self.col2[2] * v[2],
        ]
    }
}

impl Default for JointMatrix {
    fn default() -> Self {
        Self::IDENTITY
    }
}

// ---------------------------------------------------------------------------
// SkinnedMesh
// ---------------------------------------------------------------------------

/// GPU resources for a skinned mesh.
///
/// Contains source vertices, bone weights, joint matrices, and output buffer.
pub struct SkinnedMesh {
    /// Uniform buffer for skinning parameters.
    pub params_buffer: wgpu::Buffer,
    /// Storage buffer for source (bind pose) vertices.
    pub src_vertex_buffer: wgpu::Buffer,
    /// Storage buffer for bone weights.
    pub bone_weight_buffer: wgpu::Buffer,
    /// Storage buffer for joint matrices.
    pub joint_matrix_buffer: wgpu::Buffer,
    /// Storage buffer for skinned output vertices.
    pub dst_vertex_buffer: wgpu::Buffer,
    /// Number of vertices in the mesh.
    pub vertex_count: u32,
    /// Number of joints in the skeleton.
    pub joint_count: u32,
    /// Bind group for skinning shader.
    pub bind_group: wgpu::BindGroup,
}

impl SkinnedMesh {
    /// Create a new skinned mesh from vertex and weight data.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `vertices` - Source vertices (bind pose).
    /// * `weights` - Bone weights per vertex.
    /// * `joint_count` - Number of joints in skeleton.
    /// * `bind_group_layout` - Layout from `VertexSkinningPipeline`.
    pub fn new(
        device: &wgpu::Device,
        vertices: &[SkinnedVertex],
        weights: &[BoneWeight],
        joint_count: u32,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        assert_eq!(
            vertices.len(),
            weights.len(),
            "Vertex and weight counts must match"
        );

        let vertex_count = vertices.len() as u32;

        // Create params buffer
        let params = VertexSkinningParams::new(vertex_count, joint_count);
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning_params"),
            size: SKINNING_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create source vertex buffer
        let src_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning_src_vertices"),
            size: (vertex_count as u64) * (SKINNED_VERTEX_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bone weight buffer
        let bone_weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning_bone_weights"),
            size: (vertex_count as u64) * (BONE_WEIGHT_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create joint matrix buffer (allocate for max joints)
        let joint_matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning_joint_matrices"),
            size: (joint_count as u64).max(1) * (JOINT_MATRIX_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create destination vertex buffer
        let dst_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning_dst_vertices"),
            size: (vertex_count as u64) * (SKINNED_VERTEX_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinning_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bone_weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: joint_matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dst_vertex_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            params_buffer,
            src_vertex_buffer,
            bone_weight_buffer,
            joint_matrix_buffer,
            dst_vertex_buffer,
            vertex_count,
            joint_count,
            bind_group,
        }
    }

    /// Upload source vertices and weights.
    pub fn upload_mesh_data(
        &self,
        queue: &wgpu::Queue,
        vertices: &[SkinnedVertex],
        weights: &[BoneWeight],
    ) {
        queue.write_buffer(&self.src_vertex_buffer, 0, bytemuck::cast_slice(vertices));
        queue.write_buffer(&self.bone_weight_buffer, 0, bytemuck::cast_slice(weights));
    }

    /// Update joint matrices for this frame.
    pub fn update_joint_matrices(&self, queue: &wgpu::Queue, matrices: &[JointMatrix]) {
        assert!(
            matrices.len() <= self.joint_count as usize,
            "Too many joint matrices"
        );
        queue.write_buffer(
            &self.joint_matrix_buffer,
            0,
            bytemuck::cast_slice(matrices),
        );
    }

    /// Update skinning parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &VertexSkinningParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }
}

// ---------------------------------------------------------------------------
// VertexSkinningPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for vertex skinning.
///
/// Encapsulates the shader module, pipeline, and bind group layout.
pub struct VertexSkinningPipeline {
    /// Bind group layout for skinning shader.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Compute pipeline for multi-bone skinning.
    pub pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for single-bone skinning (optimized).
    pub pipeline_single: wgpu::ComputePipeline,
}

impl VertexSkinningPipeline {
    /// Create the vertex skinning pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vertex_skinning_bind_group_layout"),
            entries: &[
                // binding 0: VertexSkinningParams (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: src_vertices (storage, read)
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
                // binding 2: bone_weights (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: joint_matrices (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: dst_vertices (storage, read_write)
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vertex_skinning_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader module
        let shader_source =
            include_str!("../../shaders/skinning/vertex_skinning.comp.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vertex_skinning_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create multi-bone skinning pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vertex_skinning_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "skin_vertices",
            compilation_options: Default::default(),
            cache: None,
        });

        // Create single-bone skinning pipeline (optimized)
        let pipeline_single = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vertex_skinning_single_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "skin_vertices_single",
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            bind_group_layout,
            pipeline,
            pipeline_single,
        }
    }

    /// Dispatch the vertex skinning compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record dispatch.
    /// * `mesh` - Skinned mesh with bind group.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, mesh: &SkinnedMesh) {
        if mesh.vertex_count == 0 {
            return;
        }

        let num_workgroups = (mesh.vertex_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vertex_skinning_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &mesh.bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    /// Dispatch the single-bone skinning compute shader.
    ///
    /// Use when all vertices have only one bone influence for better performance.
    pub fn dispatch_single(&self, encoder: &mut wgpu::CommandEncoder, mesh: &SkinnedMesh) {
        if mesh.vertex_count == 0 {
            return;
        }

        let num_workgroups = (mesh.vertex_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vertex_skinning_single_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline_single);
        pass.set_bind_group(0, &mesh.bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// Normalize a 3D vector.
#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > f32::EPSILON {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Compute the normal transformation matrix (transpose of inverse).
///
/// For non-uniform scale, normals must be transformed by (M^-1)^T.
/// For orthogonal matrices (pure rotation), this equals the original matrix.
///
/// Input matrix m is in column-major order: m[col][row].
fn cpu_normal_matrix(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    // Convert to row-major for clarity in determinant/inverse calculation
    // r[row][col] = m[col][row]
    let a00 = m[0][0]; let a01 = m[1][0]; let a02 = m[2][0];
    let a10 = m[0][1]; let a11 = m[1][1]; let a12 = m[2][1];
    let a20 = m[0][2]; let a21 = m[1][2]; let a22 = m[2][2];

    // Compute determinant using row expansion
    let det = a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20);

    if det.abs() < f32::EPSILON {
        // Degenerate matrix, return identity
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let inv_det = 1.0 / det;

    // Compute inverse using adjugate formula
    // inv[i][j] = cofactor[j][i] / det (adjugate is transpose of cofactor matrix)
    let inv00 = (a11 * a22 - a12 * a21) * inv_det;
    let inv01 = (a02 * a21 - a01 * a22) * inv_det;
    let inv02 = (a01 * a12 - a02 * a11) * inv_det;

    let inv10 = (a12 * a20 - a10 * a22) * inv_det;
    let inv11 = (a00 * a22 - a02 * a20) * inv_det;
    let inv12 = (a02 * a10 - a00 * a12) * inv_det;

    let inv20 = (a10 * a21 - a11 * a20) * inv_det;
    let inv21 = (a01 * a20 - a00 * a21) * inv_det;
    let inv22 = (a00 * a11 - a01 * a10) * inv_det;

    // Normal matrix = (M^-1)^T
    // Transpose: result[row][col] = inv[col][row]
    // Convert back to column-major for output
    // output[col][row] = transpose[row][col] = inv[col][row]
    [
        [inv00, inv01, inv02],  // column 0 = row 0 of inverse
        [inv10, inv11, inv12],  // column 1 = row 1 of inverse
        [inv20, inv21, inv22],  // column 2 = row 2 of inverse
    ]
}

/// Multiply a 3x3 matrix by a 3D vector.
fn mat3_mul_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Blend multiple joint transforms with given weights (CPU reference).
///
/// # Arguments
///
/// * `matrices` - All joint matrices.
/// * `weight` - Bone weight for this vertex.
///
/// # Returns
///
/// Blended 4x4 transformation matrix.
pub fn cpu_blend_transforms(matrices: &[JointMatrix], weight: &BoneWeight) -> JointMatrix {
    let indices = weight.unpack_indices();
    let weights = weight.all_weights();

    let mut result = JointMatrix {
        col0: [0.0; 4],
        col1: [0.0; 4],
        col2: [0.0; 4],
        col3: [0.0; 4],
    };

    for i in 0..4 {
        let idx = indices[i] as usize;
        let w = weights[i];

        if w > f32::EPSILON && idx < matrices.len() {
            let m = &matrices[idx];
            for j in 0..4 {
                result.col0[j] += m.col0[j] * w;
                result.col1[j] += m.col1[j] * w;
                result.col2[j] += m.col2[j] * w;
                result.col3[j] += m.col3[j] * w;
            }
        }
    }

    result
}

/// Skin a single vertex using blended joint transform (CPU reference).
///
/// # Arguments
///
/// * `vertex` - Source vertex (bind pose).
/// * `blended` - Blended joint matrix.
///
/// # Returns
///
/// Skinned vertex with transformed position, normal, and tangent.
pub fn cpu_skin_vertex(vertex: &SkinnedVertex, blended: &JointMatrix) -> SkinnedVertex {
    // Transform position
    let position = blended.transform_point(vertex.position);

    // Extract 3x3 for normal/tangent transformation
    let m3 = [
        [blended.col0[0], blended.col0[1], blended.col0[2]],
        [blended.col1[0], blended.col1[1], blended.col1[2]],
        [blended.col2[0], blended.col2[1], blended.col2[2]],
    ];

    // Compute normal matrix (transpose of inverse) for proper normal transformation
    let normal_matrix = cpu_normal_matrix(m3);
    let normal = normalize(mat3_mul_vec3(normal_matrix, vertex.normal));

    // Transform tangent (just rotation/scale)
    let tangent_xyz = normalize(mat3_mul_vec3(m3, [
        vertex.tangent[0],
        vertex.tangent[1],
        vertex.tangent[2],
    ]));

    SkinnedVertex {
        position,
        normal,
        tangent: [tangent_xyz[0], tangent_xyz[1], tangent_xyz[2], vertex.tangent[3]],
        uv: vertex.uv,
    }
}

/// Skin a batch of vertices (CPU reference).
///
/// # Arguments
///
/// * `vertices` - Source vertices.
/// * `weights` - Bone weights per vertex.
/// * `matrices` - Joint matrices.
/// * `output` - Output buffer for skinned vertices.
pub fn cpu_skin_vertices(
    vertices: &[SkinnedVertex],
    weights: &[BoneWeight],
    matrices: &[JointMatrix],
    output: &mut [SkinnedVertex],
) {
    assert_eq!(vertices.len(), weights.len());
    assert_eq!(vertices.len(), output.len());

    for i in 0..vertices.len() {
        let blended = cpu_blend_transforms(matrices, &weights[i]);
        output[i] = cpu_skin_vertex(&vertices[i], &blended);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SkinnedVertex ───────────────────────────────────────────────────────

    #[test]
    fn test_skinned_vertex_size() {
        assert_eq!(mem::size_of::<SkinnedVertex>(), 48);
    }

    #[test]
    fn test_skinned_vertex_origin() {
        let v = SkinnedVertex::origin();
        assert_eq!(v.position, [0.0, 0.0, 0.0]);
        assert_eq!(v.normal, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_skinned_vertex_new() {
        let v = SkinnedVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.5, 0.5],
        );
        assert_eq!(v.position, [1.0, 2.0, 3.0]);
        assert_eq!(v.uv, [0.5, 0.5]);
    }

    // ── BoneWeight ──────────────────────────────────────────────────────────

    #[test]
    fn test_bone_weight_size() {
        assert_eq!(mem::size_of::<BoneWeight>(), 16);
    }

    #[test]
    fn test_bone_weight_single() {
        let bw = BoneWeight::single_bone(5);
        let indices = bw.unpack_indices();
        let weights = bw.all_weights();

        assert_eq!(indices[0], 5);
        assert!((weights[0] - 1.0).abs() < f32::EPSILON);
        assert!(weights[1].abs() < f32::EPSILON);
    }

    #[test]
    fn test_bone_weight_two_bones() {
        let bw = BoneWeight::two_bones(1, 2, 0.7);
        let indices = bw.unpack_indices();
        let weights = bw.all_weights();

        assert_eq!(indices[0], 1);
        assert_eq!(indices[1], 2);
        assert!((weights[0] - 0.7).abs() < f32::EPSILON);
        assert!((weights[1] - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_bone_weight_three_bones() {
        let bw = BoneWeight::three_bones(0, 1, 2, 0.5, 0.3);
        let indices = bw.unpack_indices();
        let weights = bw.all_weights();

        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 1);
        assert_eq!(indices[2], 2);
        assert!((weights[0] - 0.5).abs() < f32::EPSILON);
        assert!((weights[1] - 0.3).abs() < f32::EPSILON);
        assert!((weights[2] - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_bone_weight_four_bones() {
        let bw = BoneWeight::four_bones(0, 1, 2, 3, 0.4, 0.3, 0.2);
        let indices = bw.unpack_indices();
        let weights = bw.all_weights();

        assert_eq!(indices, [0, 1, 2, 3]);
        assert!((weights[0] - 0.4).abs() < f32::EPSILON);
        assert!((weights[1] - 0.3).abs() < f32::EPSILON);
        assert!((weights[2] - 0.2).abs() < f32::EPSILON);
        assert!((weights[3] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_bone_weight_normalization() {
        let mut bw = BoneWeight {
            indices: 0,
            weights: [0.5, 0.5, 0.5], // Sum > 1
        };
        bw.normalize();
        let weights = bw.all_weights();
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bone_weight_influence_count() {
        assert_eq!(BoneWeight::single_bone(0).influence_count(), 1);
        assert_eq!(BoneWeight::two_bones(0, 1, 0.5).influence_count(), 2);
        assert_eq!(BoneWeight::three_bones(0, 1, 2, 0.4, 0.3).influence_count(), 3);
        assert_eq!(BoneWeight::four_bones(0, 1, 2, 3, 0.3, 0.3, 0.2).influence_count(), 4);
    }

    // ── VertexSkinningParams ────────────────────────────────────────────────

    #[test]
    fn test_skinning_params_size() {
        assert_eq!(mem::size_of::<VertexSkinningParams>(), 16);
    }

    #[test]
    fn test_skinning_params_workgroups() {
        assert_eq!(VertexSkinningParams::new(1, 10).num_workgroups(), 1);
        assert_eq!(VertexSkinningParams::new(256, 10).num_workgroups(), 1);
        assert_eq!(VertexSkinningParams::new(257, 10).num_workgroups(), 2);
        assert_eq!(VertexSkinningParams::new(512, 10).num_workgroups(), 2);
        assert_eq!(VertexSkinningParams::new(513, 10).num_workgroups(), 3);
    }

    // ── JointMatrix ─────────────────────────────────────────────────────────

    #[test]
    fn test_joint_matrix_size() {
        assert_eq!(mem::size_of::<JointMatrix>(), 64);
    }

    #[test]
    fn test_joint_matrix_identity() {
        let m = JointMatrix::IDENTITY;
        assert_eq!(m.col0, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(m.col1, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(m.col2, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(m.col3, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_joint_matrix_translation() {
        let m = JointMatrix::translation(1.0, 2.0, 3.0);
        let p = m.transform_point([0.0, 0.0, 0.0]);
        assert_eq!(p, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_joint_matrix_scale() {
        let m = JointMatrix::scale(2.0);
        let p = m.transform_point([1.0, 1.0, 1.0]);
        assert_eq!(p, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_joint_matrix_nonuniform_scale() {
        let m = JointMatrix::scale_xyz(2.0, 3.0, 4.0);
        let p = m.transform_point([1.0, 1.0, 1.0]);
        assert_eq!(p, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_joint_matrix_rotation_z() {
        let m = JointMatrix::rotation_z(std::f32::consts::FRAC_PI_2);
        let p = m.transform_point([1.0, 0.0, 0.0]);
        assert!((p[0]).abs() < 0.001);
        assert!((p[1] - 1.0).abs() < 0.001);
        assert!((p[2]).abs() < 0.001);
    }

    #[test]
    fn test_joint_matrix_mul() {
        let t = JointMatrix::translation(1.0, 0.0, 0.0);
        let s = JointMatrix::scale(2.0);
        let ts = t.mul(&s);
        let p = ts.transform_point([1.0, 0.0, 0.0]);
        // First scale (2.0), then translate (+1.0) = 3.0
        assert!((p[0] - 3.0).abs() < 0.001);
    }

    // ── CPU Reference: Single Bone ──────────────────────────────────────────

    #[test]
    fn test_cpu_skin_identity_single_bone() {
        let vertex = SkinnedVertex::new(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [JointMatrix::IDENTITY];
        let weight = BoneWeight::single_bone(0);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        assert_eq!(skinned.position, vertex.position);
        assert_eq!(skinned.normal, vertex.normal);
    }

    #[test]
    fn test_cpu_skin_translation_single_bone() {
        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [JointMatrix::translation(5.0, 0.0, 0.0)];
        let weight = BoneWeight::single_bone(0);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        assert!((skinned.position[0] - 5.0).abs() < 0.001);
        assert_eq!(skinned.normal, vertex.normal);
    }

    #[test]
    fn test_cpu_skin_rotation_single_bone() {
        let vertex = SkinnedVertex::new(
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [JointMatrix::rotation_z(std::f32::consts::FRAC_PI_2)];
        let weight = BoneWeight::single_bone(0);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        // Position rotated 90 degrees around Z: [1,0,0] -> [0,1,0]
        assert!(
            (skinned.position[0]).abs() < 0.001,
            "position[0] should be ~0, got {}",
            skinned.position[0]
        );
        assert!(
            (skinned.position[1] - 1.0).abs() < 0.001,
            "position[1] should be ~1, got {}",
            skinned.position[1]
        );

        // Normal also rotated around Z: [1,0,0] -> [0,1,0]
        // For pure rotation, cofactor matrix equals the rotation matrix
        assert!(
            (skinned.normal[0]).abs() < 0.01,
            "normal[0] should be ~0, got {}",
            skinned.normal[0]
        );
        assert!(
            (skinned.normal[1] - 1.0).abs() < 0.01,
            "normal[1] should be ~1, got {}",
            skinned.normal[1]
        );
    }

    // ── CPU Reference: Two Bones ────────────────────────────────────────────

    #[test]
    fn test_cpu_skin_two_bones_equal() {
        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [
            JointMatrix::translation(2.0, 0.0, 0.0),
            JointMatrix::translation(-2.0, 0.0, 0.0),
        ];
        let weight = BoneWeight::two_bones(0, 1, 0.5);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        // 50% of +2 and 50% of -2 = 0
        assert!((skinned.position[0]).abs() < 0.001);
    }

    #[test]
    fn test_cpu_skin_two_bones_weighted() {
        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [
            JointMatrix::translation(10.0, 0.0, 0.0),
            JointMatrix::IDENTITY,
        ];
        let weight = BoneWeight::two_bones(0, 1, 0.3);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        // 30% of 10 + 70% of 0 = 3
        assert!((skinned.position[0] - 3.0).abs() < 0.001);
    }

    // ── CPU Reference: Three Bones ──────────────────────────────────────────

    #[test]
    fn test_cpu_skin_three_bones() {
        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [
            JointMatrix::translation(10.0, 0.0, 0.0),
            JointMatrix::translation(0.0, 10.0, 0.0),
            JointMatrix::translation(0.0, 0.0, 10.0),
        ];
        // 0.4 + 0.3 + 0.3 = 1.0
        let weight = BoneWeight::three_bones(0, 1, 2, 0.4, 0.3);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        assert!((skinned.position[0] - 4.0).abs() < 0.001);
        assert!((skinned.position[1] - 3.0).abs() < 0.001);
        assert!((skinned.position[2] - 3.0).abs() < 0.001);
    }

    // ── CPU Reference: Four Bones ───────────────────────────────────────────

    #[test]
    fn test_cpu_skin_four_bones() {
        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [
            JointMatrix::translation(10.0, 0.0, 0.0),
            JointMatrix::translation(0.0, 10.0, 0.0),
            JointMatrix::translation(0.0, 0.0, 10.0),
            JointMatrix::translation(0.0, 0.0, 0.0),
        ];
        // 0.25 each
        let weight = BoneWeight::four_bones(0, 1, 2, 3, 0.25, 0.25, 0.25);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        assert!((skinned.position[0] - 2.5).abs() < 0.001);
        assert!((skinned.position[1] - 2.5).abs() < 0.001);
        assert!((skinned.position[2] - 2.5).abs() < 0.001);
    }

    // ── CPU Reference: Normal Transformation ────────────────────────────────

    #[test]
    fn test_cpu_skin_normal_uniform_scale() {
        let vertex = SkinnedVertex::new(
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [JointMatrix::scale(2.0)];
        let weight = BoneWeight::single_bone(0);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        // Position scaled
        assert!((skinned.position[0] - 2.0).abs() < 0.001);

        // Normal unchanged (uniform scale preserves direction)
        assert!((skinned.normal[0] - 1.0).abs() < 0.001);
        assert!((skinned.normal[1]).abs() < 0.001);
    }

    #[test]
    fn test_cpu_skin_normal_nonuniform_scale() {
        let vertex = SkinnedVertex::new(
            [1.0, 1.0, 0.0],
            [0.707, 0.707, 0.0], // 45-degree normal
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0],
        );
        // Scale X by 2, Y by 0.5
        let matrices = [JointMatrix::scale_xyz(2.0, 0.5, 1.0)];
        let weight = BoneWeight::single_bone(0);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        // Position scaled
        assert!((skinned.position[0] - 2.0).abs() < 0.001);
        assert!((skinned.position[1] - 0.5).abs() < 0.001);

        // Normal should be transformed by inverse transpose
        // For scale(2, 0.5, 1), inverse is scale(0.5, 2, 1)
        // Normal (0.707, 0.707) -> (0.707*0.5, 0.707*2) = (0.35, 1.41) normalized
        let nx = skinned.normal[0];
        let ny = skinned.normal[1];
        let len = (nx * nx + ny * ny).sqrt();
        assert!((len - 1.0).abs() < 0.01); // Should be normalized
        // Y component should be larger than X after this scale
        assert!(ny.abs() > nx.abs());
    }

    // ── CPU Reference: Weight Normalization ─────────────────────────────────

    #[test]
    fn test_cpu_skin_partial_weights() {
        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [JointMatrix::translation(10.0, 0.0, 0.0)];

        // Weight [0.5, 0, 0] with all indices = 0 means:
        // - w0=0.5 applied to bone 0
        // - w3=0.5 (computed as 1-0.5) also applied to bone 0
        // Total: 1.0 * translation(10,0,0) = [10, 0, 0]
        let weight = BoneWeight {
            indices: 0, // All indices are 0
            weights: [0.5, 0.0, 0.0],
        };

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        // All weight goes to bone 0, so full translation applies
        assert!(
            (skinned.position[0] - 10.0).abs() < 0.001,
            "Expected 10.0, got {}",
            skinned.position[0]
        );
    }

    #[test]
    fn test_cpu_skin_invalid_fourth_bone() {
        let vertex = SkinnedVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [JointMatrix::translation(10.0, 0.0, 0.0)];

        // First bone has 50% weight, fourth bone (index 255) is invalid
        // with 50% weight that won't be applied
        let weight = BoneWeight {
            indices: 0xFF000000, // Indices [0, 0, 0, 255]
            weights: [0.5, 0.0, 0.0],
        };

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        // Only 50% of weight is applied (invalid 4th bone is skipped)
        assert!(
            (skinned.position[0] - 5.0).abs() < 0.001,
            "Expected 5.0, got {}",
            skinned.position[0]
        );
    }

    // ── CPU Reference: Batch Processing ─────────────────────────────────────

    #[test]
    fn test_cpu_skin_vertices_batch() {
        let vertices = vec![
            SkinnedVertex::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0]),
            SkinnedVertex::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [0.5, 0.0]),
            SkinnedVertex::new([2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0]),
        ];
        let weights = vec![
            BoneWeight::single_bone(0),
            BoneWeight::single_bone(0),
            BoneWeight::single_bone(0),
        ];
        let matrices = vec![JointMatrix::translation(0.0, 5.0, 0.0)];
        let mut output = vec![SkinnedVertex::default(); 3];

        cpu_skin_vertices(&vertices, &weights, &matrices, &mut output);

        for (i, v) in output.iter().enumerate() {
            assert!((v.position[0] - vertices[i].position[0]).abs() < 0.001);
            assert!((v.position[1] - 5.0).abs() < 0.001);
            assert_eq!(v.uv, vertices[i].uv);
        }
    }

    // ── WGSL Shader Validation ──────────────────────────────────────────────

    #[test]
    fn test_shader_validation() {
        let shader_source =
            include_str!("../../shaders/skinning/vertex_skinning.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("Failed to parse vertex_skinning.comp.wgsl");

        // Validate the module
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("Shader validation failed");

        // Check for expected entry points
        let entry_points: Vec<_> = module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();
        assert!(entry_points.contains(&"skin_vertices"));
        assert!(entry_points.contains(&"skin_vertices_single"));
    }

    // ── Edge Cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_cpu_skin_zero_weight() {
        let vertex = SkinnedVertex::origin();
        let matrices = [JointMatrix::translation(100.0, 0.0, 0.0)];
        let weight = BoneWeight {
            indices: 0,
            weights: [0.0, 0.0, 0.0], // All zero (degenerate)
        };

        let blended = cpu_blend_transforms(&matrices, &weight);
        // Blended matrix will be all zeros, which means identity-like behavior
        // after normalization in skin_vertex
        let skinned = cpu_skin_vertex(&vertex, &blended);

        // With zero matrix, position becomes [0,0,0] and normal [0,0,0]
        // This is an edge case - in practice weights should sum to 1
        assert!(!skinned.position[0].is_nan());
    }

    #[test]
    fn test_cpu_skin_invalid_bone_index() {
        let vertex = SkinnedVertex::new(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0],
        );
        let matrices = [JointMatrix::IDENTITY]; // Only bone 0 exists

        // Reference invalid bone index
        let weight = BoneWeight::two_bones(0, 99, 0.5);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        // Only valid bone (50%) should contribute
        assert!((skinned.position[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_tangent_handedness_preserved() {
        let vertex = SkinnedVertex::new(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, -1.0], // Negative handedness
            [0.0, 0.0],
        );
        let matrices = [JointMatrix::rotation_y(std::f32::consts::FRAC_PI_2)];
        let weight = BoneWeight::single_bone(0);

        let blended = cpu_blend_transforms(&matrices, &weight);
        let skinned = cpu_skin_vertex(&vertex, &blended);

        // Handedness should be preserved
        assert!((skinned.tangent[3] - (-1.0)).abs() < f32::EPSILON);
    }
}
