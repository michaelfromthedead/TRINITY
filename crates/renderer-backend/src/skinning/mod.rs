//! GPU Compute Skinning subsystem for TRINITY Engine.
//!
//! This module provides GPU-accelerated skeletal animation computation,
//! including joint transform computation, vertex skinning, and dual quaternion blending.
//!
//! # Sub-modules
//!
//! - `joint_transform` -- Joint Transform Compute (T-GPU-7.1)
//!   - World-space skinning matrix computation from local transforms
//!   - Joint hierarchy traversal on GPU
//!   - Inverse bind pose multiplication
//!
//! - `dualquat` -- Dual Quaternion Skinning (T-GPU-7.3)
//!   - Volume-preserving joint blending (no "candy wrapper" artifact)
//!   - Mat4 to dual quaternion conversion
//!   - Antipodal quaternion handling for shortest path interpolation
//!
//! # Overview
//!
//! Skeletal animation involves two main GPU operations:
//!
//! 1. **Joint Transform Compute**: For each joint in a skeleton:
//!    - Traverse parent chain to accumulate world transform
//!    - Multiply by inverse bind pose to get final skinning matrix
//!
//! 2. **Vertex Skinning** (future): For each skinned vertex:
//!    - Blend multiple joint matrices using bone weights
//!    - Transform position and normal
//!
//! # Memory Layout
//!
//! All structures are GPU-aligned (vec4, 16-byte boundaries):
//!
//! | Type            | Size  | Description                      |
//! |-----------------|-------|----------------------------------|
//! | JointData       | 64B   | Local transform + hierarchy info |
//! | BindPose        | 48B   | Inverse bind matrix (mat4x3)     |
//! | SkinningMatrix  | 48B   | Output skinning matrix (mat4x3)  |
//!
//! # Performance Targets
//!
//! - Joint transforms: < 0.05ms for 256 joints
//! - Hierarchy depth: Up to 64 levels supported
//! - Memory bandwidth: ~160 bytes per joint
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::skinning::{
//!     JointData, BindPose, JointTransformPipeline, JointTransformResources,
//!     JointTransformParams, cpu_compute_joint_matrix,
//! };
//!
//! // Set up skeleton data
//! let joints = vec![
//!     JointData::from_translation([0.0, 0.0, 0.0], 0, -1),  // Root
//!     JointData::from_translation([1.0, 0.0, 0.0], 1, 0),   // Child
//! ];
//! let bind_poses = vec![BindPose::identity(), BindPose::identity()];
//!
//! // GPU path: create pipeline and resources
//! let pipeline = JointTransformPipeline::new(&device, SHADER_SOURCE);
//! let resources = JointTransformResources::new(&device, 256);
//!
//! resources.upload_joints(&queue, &joints);
//! resources.upload_bind_poses(&queue, &bind_poses);
//! resources.upload_params(&queue, &JointTransformParams::new(2));
//!
//! // Dispatch compute
//! let bind_group = pipeline.create_bind_group(&device, &resources);
//! encoder.dispatch_workgroups(params.num_workgroups(), 1, 1);
//!
//! // CPU fallback for testing
//! let matrices = cpu_compute_skinning_matrices(&joints, &bind_poses);
//! ```

pub mod dualquat;
pub mod joint_transform;
pub mod vertex_skinning;

// Re-export joint_transform core types (T-GPU-7.1)
pub use joint_transform::{
    // Data structures
    BindPose,
    JointData,
    JointTransformParams,
    SkinningMatrix,

    // Hierarchy management
    JointHierarchy,
    HierarchyError,

    // GPU resources and pipeline
    JointTransformPipeline,
    JointTransformResources,

    // CPU reference implementations
    cpu_compute_joint_matrix,
    cpu_compute_skinning_matrices,
    cpu_concat_transforms,

    // Constants
    BIND_POSE_SIZE,
    FLAG_SKIP_INVERSE_BIND,
    FLAG_TOPOLOGICAL_ORDER,
    INVALID_PARENT,
    JOINT_DATA_SIZE,
    JOINT_TRANSFORM_PARAMS_SIZE,
    MAX_HIERARCHY_DEPTH,
    MAX_JOINTS,
    SKINNING_MATRIX_SIZE,
    WORKGROUP_SIZE,
};

// Re-export dualquat module types (T-GPU-7.3)
pub use dualquat::{
    // Data structures
    DualQuat,
    DualQuatBuffer,
    DualQuatSkinningParams,
    DualQuatSkinningPipeline,
    DualQuatSkinningResources,

    // Constants
    DUALQUAT_SIZE,
    MAX_JOINTS as DUALQUAT_MAX_JOINTS,
    MAX_WEIGHTS_PER_VERTEX,
    SKINNING_PARAMS_SIZE as DUALQUAT_SKINNING_PARAMS_SIZE,
    WORKGROUP_SIZE as DUALQUAT_WORKGROUP_SIZE,

    // CPU reference functions
    cpu_dualquat_blend,
    cpu_dualquat_get_translation,
    cpu_dualquat_normalize,
    cpu_dualquat_transform_normal,
    cpu_dualquat_transform_point,
    cpu_mat4_to_dualquat,
    cpu_quat_rotate_vector,
};

// Re-export vertex_skinning module types (T-GPU-7.2)
pub use vertex_skinning::{
    // Data structures
    BoneWeight,
    JointMatrix,
    SkinnedMesh,
    SkinnedVertex,
    VertexSkinningParams,
    VertexSkinningPipeline,

    // Constants
    BONE_WEIGHT_SIZE,
    FLAG_DUAL_QUATERNION,
    FLAG_NORMALIZE_WEIGHTS,
    JOINT_MATRIX_SIZE,
    MAX_BONE_INFLUENCES,
    SKINNED_VERTEX_SIZE,
    SKINNING_PARAMS_SIZE,
    WORKGROUP_SIZE as SKINNING_WORKGROUP_SIZE,

    // CPU reference functions
    cpu_blend_transforms,
    cpu_skin_vertex,
    cpu_skin_vertices,
};
