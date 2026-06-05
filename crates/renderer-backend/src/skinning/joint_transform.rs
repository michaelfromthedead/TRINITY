//! GPU Joint Transform Compute for TRINITY Engine (T-GPU-7.1).
//!
//! This module provides GPU-based computation of world-space skinning matrices
//! from local joint transforms and skeleton hierarchy.
//!
//! # Overview
//!
//! Skeletal animation requires computing final skinning matrices for each joint:
//!
//! 1. **Local Transform**: Each joint has a local transform relative to its parent
//! 2. **World Transform**: Accumulated transform from root to joint
//! 3. **Skinning Matrix**: World transform * inverse bind pose
//!
//! The GPU compute shader traverses the joint hierarchy to build world transforms,
//! then multiplies by inverse bind poses to produce final skinning matrices.
//!
//! # Memory Layout
//!
//! - `JointData`: 64 bytes per joint (mat4x3 + hierarchy info)
//! - `BindPose`: 48 bytes per joint (mat4x3 inverse bind matrix)
//! - `SkinningMatrix`: 48 bytes per joint (mat4x3 output)
//!
//! # Performance
//!
//! - Work complexity: O(n * d) where n = joints, d = hierarchy depth
//! - Target: < 0.05ms for 256 joints with depth <= 16
//! - Memory bandwidth: ~160 bytes per joint (read + write)
//!
//! # Usage
//!
//! ```ignore
//! // Create pipeline and resources
//! let pipeline = JointTransformPipeline::new(&device, SHADER_SOURCE);
//! let resources = JointTransformResources::new(&device, 256);
//!
//! // Each frame: update joint transforms
//! resources.upload_joints(&queue, &joint_data);
//! resources.upload_params(&queue, &params);
//! pipeline.dispatch(&mut encoder, &resources, joint_count);
//!
//! // Read skinning matrices for vertex skinning
//! let matrices = resources.read_skinning_matrices(&device, &queue);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 64;

/// Maximum number of joints supported per skeleton.
pub const MAX_JOINTS: u32 = 256;

/// Maximum hierarchy depth for parent traversal.
pub const MAX_HIERARCHY_DEPTH: u32 = 64;

/// Invalid parent index (root joints have no parent).
pub const INVALID_PARENT: i32 = -1;

/// Size of JointData in bytes.
pub const JOINT_DATA_SIZE: usize = 64;

/// Size of BindPose in bytes.
pub const BIND_POSE_SIZE: usize = 48;

/// Size of SkinningMatrix in bytes.
pub const SKINNING_MATRIX_SIZE: usize = 48;

/// Size of JointTransformParams in bytes.
pub const JOINT_TRANSFORM_PARAMS_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------

/// Flag: Use topological order optimization (joints must be sorted).
pub const FLAG_TOPOLOGICAL_ORDER: u32 = 1;

/// Flag: Skip inverse bind pose multiplication (output world transforms only).
pub const FLAG_SKIP_INVERSE_BIND: u32 = 2;

// ---------------------------------------------------------------------------
// JointData
// ---------------------------------------------------------------------------

/// Joint data containing local transform and hierarchy information.
///
/// The local transform is stored as a mat4x3 in a compact format where
/// rotation/scale occupies the xyz components and translation is packed
/// into the w components of the three column vectors.
///
/// # Memory Layout
///
/// 64 bytes, vec4 aligned:
/// | Offset | Field           | Size |
/// |--------|-----------------|------|
/// | 0      | local_transform | 48   | (3 x vec4 = mat4x3)
/// | 48     | parent_index    | 4    |
/// | 52     | joint_index     | 4    |
/// | 56     | _pad0           | 4    |
/// | 60     | _pad1           | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct JointData {
    /// Local transform column 0: (m00, m10, m20, tx)
    /// Where m00, m10, m20 are the X axis and tx is translation.x
    pub local_col0: [f32; 4],
    /// Local transform column 1: (m01, m11, m21, ty)
    /// Where m01, m11, m21 are the Y axis and ty is translation.y
    pub local_col1: [f32; 4],
    /// Local transform column 2: (m02, m12, m22, tz)
    /// Where m02, m12, m22 are the Z axis and tz is translation.z
    pub local_col2: [f32; 4],
    /// Parent joint index (-1 for root joints).
    pub parent_index: i32,
    /// This joint's index in the skeleton.
    pub joint_index: u32,
    /// Padding for 64-byte alignment.
    pub _pad0: u32,
    pub _pad1: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<JointData>() == JOINT_DATA_SIZE);

impl JointData {
    /// Create joint data with identity transform.
    pub fn identity(joint_index: u32, parent_index: i32) -> Self {
        Self {
            local_col0: [1.0, 0.0, 0.0, 0.0], // X axis, tx=0
            local_col1: [0.0, 1.0, 0.0, 0.0], // Y axis, ty=0
            local_col2: [0.0, 0.0, 1.0, 0.0], // Z axis, tz=0
            parent_index,
            joint_index,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create joint data from a mat4x3 transform.
    ///
    /// # Arguments
    ///
    /// * `transform` - 4x3 matrix stored as row-major [row][col]
    /// * `joint_index` - Index of this joint in the skeleton
    /// * `parent_index` - Index of parent joint (-1 for root)
    pub fn from_mat4x3(transform: &[[f32; 3]; 4], joint_index: u32, parent_index: i32) -> Self {
        // transform is row-major: transform[row][col]
        // We need column-major for GPU: col0 = (m00, m10, m20), col1 = (m01, m11, m21), etc.
        Self {
            local_col0: [transform[0][0], transform[1][0], transform[2][0], transform[3][0]],
            local_col1: [transform[0][1], transform[1][1], transform[2][1], transform[3][1]],
            local_col2: [transform[0][2], transform[1][2], transform[2][2], transform[3][2]],
            parent_index,
            joint_index,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create joint data from translation only.
    pub fn from_translation(translation: [f32; 3], joint_index: u32, parent_index: i32) -> Self {
        Self {
            local_col0: [1.0, 0.0, 0.0, translation[0]],
            local_col1: [0.0, 1.0, 0.0, translation[1]],
            local_col2: [0.0, 0.0, 1.0, translation[2]],
            parent_index,
            joint_index,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create joint data from rotation (quaternion) and translation.
    pub fn from_rotation_translation(
        quat: [f32; 4],
        translation: [f32; 3],
        joint_index: u32,
        parent_index: i32,
    ) -> Self {
        // Convert quaternion to rotation matrix
        let [x, y, z, w] = quat;
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        Self {
            local_col0: [1.0 - (yy + zz), xy + wz, xz - wy, translation[0]],
            local_col1: [xy - wz, 1.0 - (xx + zz), yz + wx, translation[1]],
            local_col2: [xz + wy, yz - wx, 1.0 - (xx + yy), translation[2]],
            parent_index,
            joint_index,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Extract the translation component.
    pub fn translation(&self) -> [f32; 3] {
        [self.local_col0[3], self.local_col1[3], self.local_col2[3]]
    }

    /// Extract the rotation matrix (3x3) as row-major.
    pub fn rotation_matrix(&self) -> [[f32; 3]; 3] {
        [
            [self.local_col0[0], self.local_col1[0], self.local_col2[0]],
            [self.local_col0[1], self.local_col1[1], self.local_col2[1]],
            [self.local_col0[2], self.local_col1[2], self.local_col2[2]],
        ]
    }

    /// Check if this joint is a root joint (no parent).
    pub fn is_root(&self) -> bool {
        self.parent_index < 0
    }
}

// ---------------------------------------------------------------------------
// BindPose
// ---------------------------------------------------------------------------

/// Inverse bind pose matrix for skinning.
///
/// Stored as mat4x3 in the same compact format as JointData transforms.
///
/// # Memory Layout
///
/// 48 bytes, vec4 aligned:
/// | Offset | Field | Size |
/// |--------|-------|------|
/// | 0      | col0  | 16   |
/// | 16     | col1  | 16   |
/// | 32     | col2  | 16   |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BindPose {
    /// Column 0: (m00, m10, m20, tx)
    pub col0: [f32; 4],
    /// Column 1: (m01, m11, m21, ty)
    pub col1: [f32; 4],
    /// Column 2: (m02, m12, m22, tz)
    pub col2: [f32; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<BindPose>() == BIND_POSE_SIZE);

impl BindPose {
    /// Create identity inverse bind pose.
    pub fn identity() -> Self {
        Self {
            col0: [1.0, 0.0, 0.0, 0.0],
            col1: [0.0, 1.0, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Create from mat4x3 (row-major input).
    pub fn from_mat4x3(transform: &[[f32; 3]; 4]) -> Self {
        Self {
            col0: [transform[0][0], transform[1][0], transform[2][0], transform[3][0]],
            col1: [transform[0][1], transform[1][1], transform[2][1], transform[3][1]],
            col2: [transform[0][2], transform[1][2], transform[2][2], transform[3][2]],
        }
    }

    /// Create from translation only.
    pub fn from_translation(translation: [f32; 3]) -> Self {
        Self {
            col0: [1.0, 0.0, 0.0, translation[0]],
            col1: [0.0, 1.0, 0.0, translation[1]],
            col2: [0.0, 0.0, 1.0, translation[2]],
        }
    }
}

// ---------------------------------------------------------------------------
// SkinningMatrix
// ---------------------------------------------------------------------------

/// Output skinning matrix computed by the shader.
///
/// Same format as BindPose: mat4x3 with translation in w components.
///
/// # Memory Layout
///
/// 48 bytes, vec4 aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinningMatrix {
    /// Column 0: (m00, m10, m20, tx)
    pub col0: [f32; 4],
    /// Column 1: (m01, m11, m21, ty)
    pub col1: [f32; 4],
    /// Column 2: (m02, m12, m22, tz)
    pub col2: [f32; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<SkinningMatrix>() == SKINNING_MATRIX_SIZE);

impl SkinningMatrix {
    /// Create identity skinning matrix.
    pub fn identity() -> Self {
        Self {
            col0: [1.0, 0.0, 0.0, 0.0],
            col1: [0.0, 1.0, 0.0, 0.0],
            col2: [0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Extract translation component.
    pub fn translation(&self) -> [f32; 3] {
        [self.col0[3], self.col1[3], self.col2[3]]
    }

    /// Transform a point (applies rotation and translation).
    pub fn transform_point(&self, point: [f32; 3]) -> [f32; 3] {
        [
            self.col0[0] * point[0] + self.col1[0] * point[1] + self.col2[0] * point[2] + self.col0[3],
            self.col0[1] * point[0] + self.col1[1] * point[1] + self.col2[1] * point[2] + self.col1[3],
            self.col0[2] * point[0] + self.col1[2] * point[1] + self.col2[2] * point[2] + self.col2[3],
        ]
    }

    /// Transform a vector (applies rotation only, no translation).
    pub fn transform_vector(&self, vec: [f32; 3]) -> [f32; 3] {
        [
            self.col0[0] * vec[0] + self.col1[0] * vec[1] + self.col2[0] * vec[2],
            self.col0[1] * vec[0] + self.col1[1] * vec[1] + self.col2[1] * vec[2],
            self.col0[2] * vec[0] + self.col1[2] * vec[1] + self.col2[2] * vec[2],
        ]
    }
}

// ---------------------------------------------------------------------------
// JointTransformParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for joint transform computation parameters.
///
/// # Memory Layout
///
/// 16 bytes, std140/std430 compatible:
/// | Offset | Field      | Size |
/// |--------|------------|------|
/// | 0      | num_joints | 4    |
/// | 4      | flags      | 4    |
/// | 8      | _pad0      | 4    |
/// | 12     | _pad1      | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct JointTransformParams {
    /// Number of joints to process.
    pub num_joints: u32,
    /// Computation flags (see FLAG_* constants).
    pub flags: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    pub _pad1: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<JointTransformParams>() == JOINT_TRANSFORM_PARAMS_SIZE);

impl JointTransformParams {
    /// Create parameters for the given joint count.
    pub fn new(num_joints: u32) -> Self {
        Self {
            num_joints,
            flags: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create parameters with flags.
    pub fn with_flags(num_joints: u32, flags: u32) -> Self {
        Self {
            num_joints,
            flags,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_joints + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

// ---------------------------------------------------------------------------
// JointHierarchy
// ---------------------------------------------------------------------------

/// Helper struct for managing joint hierarchy.
///
/// Provides utilities for validating and sorting joints in topological order.
#[derive(Clone, Debug)]
pub struct JointHierarchy {
    /// Parent index for each joint (-1 for root).
    pub parents: Vec<i32>,
    /// Depth of each joint in the hierarchy.
    pub depths: Vec<u32>,
    /// Joints sorted in topological order (roots first).
    pub topological_order: Vec<u32>,
}

impl JointHierarchy {
    /// Create a hierarchy from parent indices.
    pub fn new(parents: &[i32]) -> Self {
        let num_joints = parents.len();
        let mut depths = vec![0u32; num_joints];
        let mut topological_order = Vec::with_capacity(num_joints);

        // Compute depths and find roots
        let mut roots = Vec::new();
        for (i, &parent) in parents.iter().enumerate() {
            if parent < 0 {
                roots.push(i as u32);
            }
        }

        // BFS to compute depths and topological order
        let mut queue = roots.clone();
        topological_order.extend(&roots);

        while let Some(joint_idx) = queue.first().cloned() {
            queue.remove(0);
            let parent_depth = depths[joint_idx as usize];

            // Find children
            for (child_idx, &parent) in parents.iter().enumerate() {
                if parent >= 0 && parent as usize == joint_idx as usize {
                    depths[child_idx] = parent_depth + 1;
                    queue.push(child_idx as u32);
                    topological_order.push(child_idx as u32);
                }
            }
        }

        Self {
            parents: parents.to_vec(),
            depths,
            topological_order,
        }
    }

    /// Get the maximum depth in the hierarchy.
    pub fn max_depth(&self) -> u32 {
        self.depths.iter().copied().max().unwrap_or(0)
    }

    /// Validate the hierarchy (check for cycles, invalid indices).
    pub fn validate(&self) -> Result<(), HierarchyError> {
        let num_joints = self.parents.len();

        for (i, &parent) in self.parents.iter().enumerate() {
            if parent >= 0 {
                let parent_idx = parent as usize;
                if parent_idx >= num_joints {
                    return Err(HierarchyError::InvalidParent {
                        joint: i as u32,
                        parent: parent as u32,
                    });
                }
                if parent_idx == i {
                    return Err(HierarchyError::SelfReference { joint: i as u32 });
                }
            }
        }

        // Check for cycles by verifying topological order covers all joints
        if self.topological_order.len() != num_joints {
            return Err(HierarchyError::Cycle);
        }

        Ok(())
    }

    /// Reorder joints array to topological order.
    pub fn reorder_joints(&self, joints: &[JointData]) -> Vec<JointData> {
        let mut reordered = Vec::with_capacity(joints.len());
        let mut old_to_new: Vec<u32> = vec![0; joints.len()];

        for (new_idx, &old_idx) in self.topological_order.iter().enumerate() {
            old_to_new[old_idx as usize] = new_idx as u32;
        }

        for &old_idx in &self.topological_order {
            let mut joint = joints[old_idx as usize];
            joint.joint_index = old_to_new[old_idx as usize];
            if joint.parent_index >= 0 {
                joint.parent_index = old_to_new[joint.parent_index as usize] as i32;
            }
            reordered.push(joint);
        }

        reordered
    }
}

/// Errors in joint hierarchy validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HierarchyError {
    /// Parent index is out of bounds.
    InvalidParent { joint: u32, parent: u32 },
    /// Joint references itself as parent.
    SelfReference { joint: u32 },
    /// Hierarchy contains a cycle.
    Cycle,
}

impl std::fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HierarchyError::InvalidParent { joint, parent } => {
                write!(f, "Joint {} has invalid parent index {}", joint, parent)
            }
            HierarchyError::SelfReference { joint } => {
                write!(f, "Joint {} references itself as parent", joint)
            }
            HierarchyError::Cycle => write!(f, "Joint hierarchy contains a cycle"),
        }
    }
}

impl std::error::Error for HierarchyError {}

// ---------------------------------------------------------------------------
// JointTransformResources
// ---------------------------------------------------------------------------

/// GPU resources for joint transform computation.
pub struct JointTransformResources {
    /// Uniform buffer for computation parameters.
    pub params_buffer: wgpu::Buffer,
    /// Storage buffer for joint data (input).
    pub joints_buffer: wgpu::Buffer,
    /// Storage buffer for inverse bind poses (input).
    pub bind_poses_buffer: wgpu::Buffer,
    /// Storage buffer for skinning matrices (output).
    pub skinning_buffer: wgpu::Buffer,
    /// Staging buffer for reading skinning matrices back to CPU.
    pub skinning_staging: wgpu::Buffer,
    /// Maximum number of joints supported.
    pub capacity: u32,
}

impl JointTransformResources {
    /// Create resources for the given joint capacity.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("joint_transform_params"),
            size: mem::size_of::<JointTransformParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let joints_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("joint_transform_joints"),
            size: (capacity as u64) * (JOINT_DATA_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_poses_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("joint_transform_bind_poses"),
            size: (capacity as u64) * (BIND_POSE_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let skinning_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("joint_transform_skinning"),
            size: (capacity as u64) * (SKINNING_MATRIX_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let skinning_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("joint_transform_skinning_staging"),
            size: (capacity as u64) * (SKINNING_MATRIX_SIZE as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            joints_buffer,
            bind_poses_buffer,
            skinning_buffer,
            skinning_staging,
            capacity,
        }
    }

    /// Upload computation parameters to GPU.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &JointTransformParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload joint data to GPU.
    ///
    /// # Panics
    ///
    /// Panics if `joints.len() > self.capacity`.
    pub fn upload_joints(&self, queue: &wgpu::Queue, joints: &[JointData]) {
        assert!(joints.len() <= self.capacity as usize);
        queue.write_buffer(&self.joints_buffer, 0, bytemuck::cast_slice(joints));
    }

    /// Upload inverse bind poses to GPU.
    ///
    /// # Panics
    ///
    /// Panics if `bind_poses.len() > self.capacity`.
    pub fn upload_bind_poses(&self, queue: &wgpu::Queue, bind_poses: &[BindPose]) {
        assert!(bind_poses.len() <= self.capacity as usize);
        queue.write_buffer(&self.bind_poses_buffer, 0, bytemuck::cast_slice(bind_poses));
    }
}

// ---------------------------------------------------------------------------
// JointTransformPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for joint transform computation.
pub struct JointTransformPipeline {
    /// Main pipeline (parent chain traversal).
    pub pipeline: wgpu::ComputePipeline,
    /// Topological order pipeline (requires sorted joints).
    pub pipeline_topological: wgpu::ComputePipeline,
    /// World transform only pipeline (skips inverse bind).
    pub pipeline_world_only: wgpu::ComputePipeline,
    /// Bind group layout for resources.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl JointTransformPipeline {
    /// Create the joint transform pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `shader_source` - WGSL shader source code.
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("joint_transform_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("joint_transform_bind_group_layout"),
            entries: &[
                // @binding(0) params: JointTransformParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(JOINT_TRANSFORM_PARAMS_SIZE as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) joints: array<JointData>
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
                // @binding(2) inverse_bind_poses: array<InverseBindPose>
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
                // @binding(3) skinning_matrices: array<SkinningMatrix>
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("joint_transform_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("joint_transform_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compute_joint_transforms",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_topological = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("joint_transform_pipeline_topological"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compute_joint_transforms_topological",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_world_only = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("joint_transform_pipeline_world_only"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compute_world_transforms_only",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            pipeline_topological,
            pipeline_world_only,
            bind_group_layout,
        }
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &JointTransformResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("joint_transform_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.joints_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.bind_poses_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.skinning_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation: compute world-space joint matrix.
///
/// Traverses the parent chain to accumulate transforms from root to joint.
pub fn cpu_compute_joint_matrix(joints: &[JointData], joint_idx: usize) -> SkinningMatrix {
    if joint_idx >= joints.len() {
        return SkinningMatrix::identity();
    }

    // Start with this joint's local transform
    let mut result = joint_to_skinning_matrix(&joints[joint_idx]);

    // Traverse parent chain
    let mut current = &joints[joint_idx];
    let mut depth = 0;

    while current.parent_index >= 0 && depth < MAX_HIERARCHY_DEPTH as usize {
        let parent_idx = current.parent_index as usize;
        if parent_idx >= joints.len() {
            break;
        }

        let parent = &joints[parent_idx];
        let parent_mat = joint_to_skinning_matrix(parent);

        // result = parent * result
        result = cpu_concat_transforms(&parent_mat, &result);

        current = parent;
        depth += 1;
    }

    result
}

/// Convert JointData to SkinningMatrix (extracts the mat4x3).
fn joint_to_skinning_matrix(joint: &JointData) -> SkinningMatrix {
    SkinningMatrix {
        col0: joint.local_col0,
        col1: joint.local_col1,
        col2: joint.local_col2,
    }
}

/// CPU reference implementation: concatenate two mat4x3 transforms.
///
/// Computes A * B treating both as affine transforms (with implicit [0,0,0,1] row).
pub fn cpu_concat_transforms(a: &SkinningMatrix, b: &SkinningMatrix) -> SkinningMatrix {
    // Extract rotation/scale and translation from A
    let a_col0 = [a.col0[0], a.col0[1], a.col0[2]];
    let a_col1 = [a.col1[0], a.col1[1], a.col1[2]];
    let a_col2 = [a.col2[0], a.col2[1], a.col2[2]];
    let a_trans = [a.col0[3], a.col1[3], a.col2[3]];

    // Extract from B
    let b_col0 = [b.col0[0], b.col0[1], b.col0[2]];
    let b_col1 = [b.col1[0], b.col1[1], b.col1[2]];
    let b_col2 = [b.col2[0], b.col2[1], b.col2[2]];
    let b_trans = [b.col0[3], b.col1[3], b.col2[3]];

    // Compute result columns (rotation/scale part)
    let r_col0 = [
        a_col0[0] * b_col0[0] + a_col1[0] * b_col0[1] + a_col2[0] * b_col0[2],
        a_col0[1] * b_col0[0] + a_col1[1] * b_col0[1] + a_col2[1] * b_col0[2],
        a_col0[2] * b_col0[0] + a_col1[2] * b_col0[1] + a_col2[2] * b_col0[2],
    ];
    let r_col1 = [
        a_col0[0] * b_col1[0] + a_col1[0] * b_col1[1] + a_col2[0] * b_col1[2],
        a_col0[1] * b_col1[0] + a_col1[1] * b_col1[1] + a_col2[1] * b_col1[2],
        a_col0[2] * b_col1[0] + a_col1[2] * b_col1[1] + a_col2[2] * b_col1[2],
    ];
    let r_col2 = [
        a_col0[0] * b_col2[0] + a_col1[0] * b_col2[1] + a_col2[0] * b_col2[2],
        a_col0[1] * b_col2[0] + a_col1[1] * b_col2[1] + a_col2[1] * b_col2[2],
        a_col0[2] * b_col2[0] + a_col1[2] * b_col2[1] + a_col2[2] * b_col2[2],
    ];

    // Translation: A * B_trans + A_trans
    let r_trans = [
        a_col0[0] * b_trans[0] + a_col1[0] * b_trans[1] + a_col2[0] * b_trans[2] + a_trans[0],
        a_col0[1] * b_trans[0] + a_col1[1] * b_trans[1] + a_col2[1] * b_trans[2] + a_trans[1],
        a_col0[2] * b_trans[0] + a_col1[2] * b_trans[1] + a_col2[2] * b_trans[2] + a_trans[2],
    ];

    SkinningMatrix {
        col0: [r_col0[0], r_col0[1], r_col0[2], r_trans[0]],
        col1: [r_col1[0], r_col1[1], r_col1[2], r_trans[1]],
        col2: [r_col2[0], r_col2[1], r_col2[2], r_trans[2]],
    }
}

/// CPU reference: compute all skinning matrices for a skeleton.
pub fn cpu_compute_skinning_matrices(
    joints: &[JointData],
    bind_poses: &[BindPose],
) -> Vec<SkinningMatrix> {
    joints
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            let world = cpu_compute_joint_matrix(joints, idx);
            let inv_bind = &bind_poses[idx];
            let inv_bind_mat = SkinningMatrix {
                col0: inv_bind.col0,
                col1: inv_bind.col1,
                col2: inv_bind.col2,
            };
            cpu_concat_transforms(&world, &inv_bind_mat)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Size assertions
    // -------------------------------------------------------------------------

    #[test]
    fn test_joint_data_size() {
        assert_eq!(mem::size_of::<JointData>(), 64);
    }

    #[test]
    fn test_bind_pose_size() {
        assert_eq!(mem::size_of::<BindPose>(), 48);
    }

    #[test]
    fn test_skinning_matrix_size() {
        assert_eq!(mem::size_of::<SkinningMatrix>(), 48);
    }

    #[test]
    fn test_joint_transform_params_size() {
        assert_eq!(mem::size_of::<JointTransformParams>(), 16);
    }

    // -------------------------------------------------------------------------
    // JointData tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_joint_data_identity() {
        let joint = JointData::identity(0, INVALID_PARENT);
        assert_eq!(joint.local_col0, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(joint.local_col1, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(joint.local_col2, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(joint.parent_index, INVALID_PARENT);
        assert!(joint.is_root());
    }

    #[test]
    fn test_joint_data_translation() {
        let joint = JointData::from_translation([1.0, 2.0, 3.0], 1, 0);
        assert_eq!(joint.translation(), [1.0, 2.0, 3.0]);
        assert_eq!(joint.parent_index, 0);
        assert!(!joint.is_root());
    }

    #[test]
    fn test_joint_data_rotation_translation() {
        // Identity quaternion: (0, 0, 0, 1)
        let joint = JointData::from_rotation_translation(
            [0.0, 0.0, 0.0, 1.0],
            [5.0, 6.0, 7.0],
            2,
            1,
        );
        let rot = joint.rotation_matrix();

        // Should be identity rotation
        assert!((rot[0][0] - 1.0).abs() < 1e-6);
        assert!((rot[1][1] - 1.0).abs() < 1e-6);
        assert!((rot[2][2] - 1.0).abs() < 1e-6);

        assert_eq!(joint.translation(), [5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_joint_data_from_mat4x3() {
        let transform = [
            [1.0, 0.0, 0.0], // Row 0
            [0.0, 1.0, 0.0], // Row 1
            [0.0, 0.0, 1.0], // Row 2
            [3.0, 4.0, 5.0], // Translation row
        ];
        let joint = JointData::from_mat4x3(&transform, 0, INVALID_PARENT);
        assert_eq!(joint.translation(), [3.0, 4.0, 5.0]);
    }

    // -------------------------------------------------------------------------
    // BindPose tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bind_pose_identity() {
        let bp = BindPose::identity();
        assert_eq!(bp.col0, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(bp.col1, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(bp.col2, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_bind_pose_from_translation() {
        let bp = BindPose::from_translation([-1.0, -2.0, -3.0]);
        assert_eq!(bp.col0[3], -1.0);
        assert_eq!(bp.col1[3], -2.0);
        assert_eq!(bp.col2[3], -3.0);
    }

    // -------------------------------------------------------------------------
    // SkinningMatrix tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_skinning_matrix_identity() {
        let sm = SkinningMatrix::identity();
        assert_eq!(sm.translation(), [0.0, 0.0, 0.0]);

        // Transform point should not change
        let point = [1.0, 2.0, 3.0];
        let result = sm.transform_point(point);
        assert_eq!(result, point);
    }

    #[test]
    fn test_skinning_matrix_transform_point() {
        // Translation only
        let sm = SkinningMatrix {
            col0: [1.0, 0.0, 0.0, 10.0],
            col1: [0.0, 1.0, 0.0, 20.0],
            col2: [0.0, 0.0, 1.0, 30.0],
        };
        let result = sm.transform_point([1.0, 2.0, 3.0]);
        assert_eq!(result, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_skinning_matrix_transform_vector() {
        // Translation only matrix should not affect vectors
        let sm = SkinningMatrix {
            col0: [1.0, 0.0, 0.0, 10.0],
            col1: [0.0, 1.0, 0.0, 20.0],
            col2: [0.0, 0.0, 1.0, 30.0],
        };
        let result = sm.transform_vector([1.0, 2.0, 3.0]);
        assert_eq!(result, [1.0, 2.0, 3.0]);
    }

    // -------------------------------------------------------------------------
    // JointTransformParams tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_params_num_workgroups() {
        assert_eq!(JointTransformParams::new(1).num_workgroups(), 1);
        assert_eq!(JointTransformParams::new(64).num_workgroups(), 1);
        assert_eq!(JointTransformParams::new(65).num_workgroups(), 2);
        assert_eq!(JointTransformParams::new(128).num_workgroups(), 2);
        assert_eq!(JointTransformParams::new(256).num_workgroups(), 4);
    }

    #[test]
    fn test_params_with_flags() {
        let params = JointTransformParams::with_flags(100, FLAG_TOPOLOGICAL_ORDER);
        assert_eq!(params.num_joints, 100);
        assert_eq!(params.flags, FLAG_TOPOLOGICAL_ORDER);
    }

    // -------------------------------------------------------------------------
    // JointHierarchy tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hierarchy_single_root() {
        let parents = vec![-1];
        let hierarchy = JointHierarchy::new(&parents);

        assert_eq!(hierarchy.depths, vec![0]);
        assert_eq!(hierarchy.topological_order, vec![0]);
        assert!(hierarchy.validate().is_ok());
    }

    #[test]
    fn test_hierarchy_linear_chain() {
        // Root -> Joint1 -> Joint2 -> Joint3
        let parents = vec![-1, 0, 1, 2];
        let hierarchy = JointHierarchy::new(&parents);

        assert_eq!(hierarchy.depths, vec![0, 1, 2, 3]);
        assert_eq!(hierarchy.max_depth(), 3);
        assert_eq!(hierarchy.topological_order, vec![0, 1, 2, 3]);
        assert!(hierarchy.validate().is_ok());
    }

    #[test]
    fn test_hierarchy_branching() {
        //       Root(0)
        //      /       \
        //   Joint1(1)  Joint2(2)
        //     |
        //   Joint3(3)
        let parents = vec![-1, 0, 0, 1];
        let hierarchy = JointHierarchy::new(&parents);

        assert_eq!(hierarchy.depths, vec![0, 1, 1, 2]);
        assert_eq!(hierarchy.max_depth(), 2);
        assert!(hierarchy.validate().is_ok());

        // Topological order: root first, then children
        assert_eq!(hierarchy.topological_order[0], 0); // Root
        // Children 1 and 2 can be in any order
        assert!(hierarchy.topological_order.contains(&1));
        assert!(hierarchy.topological_order.contains(&2));
        // Joint 3 must be after Joint 1
        let pos_1 = hierarchy.topological_order.iter().position(|&x| x == 1).unwrap();
        let pos_3 = hierarchy.topological_order.iter().position(|&x| x == 3).unwrap();
        assert!(pos_1 < pos_3);
    }

    #[test]
    fn test_hierarchy_invalid_parent() {
        let parents = vec![-1, 100]; // Joint 1 has invalid parent 100
        let hierarchy = JointHierarchy::new(&parents);

        let result = hierarchy.validate();
        assert!(matches!(result, Err(HierarchyError::InvalidParent { .. })));
    }

    #[test]
    fn test_hierarchy_self_reference() {
        let parents = vec![-1, 1]; // Joint 1 references itself
        let hierarchy = JointHierarchy::new(&parents);

        let result = hierarchy.validate();
        assert!(matches!(result, Err(HierarchyError::SelfReference { .. })));
    }

    #[test]
    fn test_hierarchy_reorder_joints() {
        // Out of order: Joint1(1) is a child of Joint0(0), but Joint1 comes first in array
        // We'll test reordering with reversed joint data
        let parents = vec![-1, 0, 1]; // Linear: 0 -> 1 -> 2
        let hierarchy = JointHierarchy::new(&parents);

        let joints = vec![
            JointData::from_translation([1.0, 0.0, 0.0], 0, -1),
            JointData::from_translation([2.0, 0.0, 0.0], 1, 0),
            JointData::from_translation([3.0, 0.0, 0.0], 2, 1),
        ];

        let reordered = hierarchy.reorder_joints(&joints);

        // After reordering, indices should be updated
        assert_eq!(reordered.len(), 3);
        assert_eq!(reordered[0].parent_index, -1); // Root
        assert_eq!(reordered[1].parent_index, 0);  // Child of new index 0
        assert_eq!(reordered[2].parent_index, 1);  // Child of new index 1
    }

    // -------------------------------------------------------------------------
    // CPU Reference Implementation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_compute_joint_matrix_single_root() {
        let joints = vec![JointData::from_translation([5.0, 6.0, 7.0], 0, -1)];

        let result = cpu_compute_joint_matrix(&joints, 0);
        assert_eq!(result.translation(), [5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_cpu_compute_joint_matrix_chain() {
        // Root at (1,0,0), Child at local (2,0,0)
        // World position of child should be (3,0,0)
        let joints = vec![
            JointData::from_translation([1.0, 0.0, 0.0], 0, -1),
            JointData::from_translation([2.0, 0.0, 0.0], 1, 0),
        ];

        let result = cpu_compute_joint_matrix(&joints, 1);
        let trans = result.translation();
        assert!((trans[0] - 3.0).abs() < 1e-6);
        assert!((trans[1] - 0.0).abs() < 1e-6);
        assert!((trans[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_compute_joint_matrix_deep_chain() {
        // Chain of 5 joints, each translated by (1,0,0) locally
        // Final joint should be at (5,0,0)
        let joints: Vec<JointData> = (0..5)
            .map(|i| {
                JointData::from_translation(
                    [1.0, 0.0, 0.0],
                    i as u32,
                    if i == 0 { -1 } else { i as i32 - 1 },
                )
            })
            .collect();

        let result = cpu_compute_joint_matrix(&joints, 4);
        let trans = result.translation();
        assert!((trans[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_concat_transforms_identity() {
        let a = SkinningMatrix::identity();
        let b = SkinningMatrix::identity();
        let result = cpu_concat_transforms(&a, &b);

        assert!((result.col0[0] - 1.0).abs() < 1e-6);
        assert!((result.col1[1] - 1.0).abs() < 1e-6);
        assert!((result.col2[2] - 1.0).abs() < 1e-6);
        assert_eq!(result.translation(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cpu_concat_transforms_translation() {
        let a = SkinningMatrix {
            col0: [1.0, 0.0, 0.0, 1.0],
            col1: [0.0, 1.0, 0.0, 2.0],
            col2: [0.0, 0.0, 1.0, 3.0],
        };
        let b = SkinningMatrix {
            col0: [1.0, 0.0, 0.0, 4.0],
            col1: [0.0, 1.0, 0.0, 5.0],
            col2: [0.0, 0.0, 1.0, 6.0],
        };

        let result = cpu_concat_transforms(&a, &b);
        let trans = result.translation();

        // Translation should be sum
        assert!((trans[0] - 5.0).abs() < 1e-6);
        assert!((trans[1] - 7.0).abs() < 1e-6);
        assert!((trans[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_compute_skinning_matrices() {
        let joints = vec![
            JointData::from_translation([1.0, 0.0, 0.0], 0, -1),
            JointData::from_translation([2.0, 0.0, 0.0], 1, 0),
        ];
        let bind_poses = vec![
            BindPose::identity(),
            BindPose::from_translation([-3.0, 0.0, 0.0]), // Inverse offset
        ];

        let results = cpu_compute_skinning_matrices(&joints, &bind_poses);

        // Joint 0: world (1,0,0) * identity = (1,0,0)
        let t0 = results[0].translation();
        assert!((t0[0] - 1.0).abs() < 1e-6);

        // Joint 1: world (3,0,0) * inv_bind (-3,0,0) = (0,0,0)
        let t1 = results[1].translation();
        assert!((t1[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_compute_joint_matrix_out_of_bounds() {
        let joints = vec![JointData::identity(0, -1)];
        let result = cpu_compute_joint_matrix(&joints, 100); // Out of bounds
        // Should return identity
        assert_eq!(result.col0, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_joint_rotation_90_degrees() {
        // 90 degree rotation around Z axis
        // Quaternion for 90 deg Z: (0, 0, sin(45), cos(45)) = (0, 0, 0.707, 0.707)
        let sin45 = std::f32::consts::FRAC_PI_4.sin();
        let cos45 = std::f32::consts::FRAC_PI_4.cos();
        let joint = JointData::from_rotation_translation(
            [0.0, 0.0, sin45, cos45],
            [0.0, 0.0, 0.0],
            0,
            -1,
        );

        let rot = joint.rotation_matrix();

        // Standard rotation matrix for angle theta around Z:
        // | cos(theta) -sin(theta)  0 |
        // | sin(theta)  cos(theta)  0 |
        // |     0           0       1 |
        //
        // For 90 degrees (theta = pi/2):
        // | 0  -1  0 |
        // | 1   0  0 |
        // | 0   0  1 |
        assert!((rot[0][0] - 0.0).abs() < 1e-5);  // cos(90) = 0
        assert!((rot[0][1] + 1.0).abs() < 1e-5);  // -sin(90) = -1
        assert!((rot[1][0] - 1.0).abs() < 1e-5);  // sin(90) = 1
        assert!((rot[1][1] - 0.0).abs() < 1e-5);  // cos(90) = 0
        assert!((rot[2][2] - 1.0).abs() < 1e-5);  // Z axis unchanged
    }

    #[test]
    fn test_complex_hierarchy() {
        //       Root(0) at origin
        //      /         \
        //   Arm(1)      Leg(2)
        //   at (1,0,0)  at (-1,0,0)
        //     |
        //   Hand(3)
        //   at (1,0,0) local

        let joints = vec![
            JointData::from_translation([0.0, 0.0, 0.0], 0, -1), // Root
            JointData::from_translation([1.0, 0.0, 0.0], 1, 0),  // Arm
            JointData::from_translation([-1.0, 0.0, 0.0], 2, 0), // Leg
            JointData::from_translation([1.0, 0.0, 0.0], 3, 1),  // Hand
        ];

        // Root world position = (0,0,0)
        let root_world = cpu_compute_joint_matrix(&joints, 0);
        assert_eq!(root_world.translation(), [0.0, 0.0, 0.0]);

        // Arm world position = (1,0,0)
        let arm_world = cpu_compute_joint_matrix(&joints, 1);
        let t1 = arm_world.translation();
        assert!((t1[0] - 1.0).abs() < 1e-6);

        // Leg world position = (-1,0,0)
        let leg_world = cpu_compute_joint_matrix(&joints, 2);
        let t2 = leg_world.translation();
        assert!((t2[0] + 1.0).abs() < 1e-6);

        // Hand world position = (2,0,0) (Arm + Hand local)
        let hand_world = cpu_compute_joint_matrix(&joints, 3);
        let t3 = hand_world.translation();
        assert!((t3[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_joints_constant() {
        assert_eq!(MAX_JOINTS, 256);
    }

    #[test]
    fn test_workgroup_size_constant() {
        assert_eq!(WORKGROUP_SIZE, 64);
    }

    // -------------------------------------------------------------------------
    // Shader validation test
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_parses() {
        let shader_source = include_str!("../../shaders/skinning/joint_transform.comp.wgsl");

        let result = naga::front::wgsl::parse_str(shader_source);
        assert!(
            result.is_ok(),
            "Shader failed to parse: {:?}",
            result.err()
        );

        let module = result.unwrap();

        // Verify entry points exist
        let entry_points: Vec<&str> = module.entry_points.iter().map(|ep| ep.name.as_str()).collect();
        assert!(entry_points.contains(&"compute_joint_transforms"));
        assert!(entry_points.contains(&"compute_joint_transforms_topological"));
        assert!(entry_points.contains(&"compute_world_transforms_only"));
    }
}
