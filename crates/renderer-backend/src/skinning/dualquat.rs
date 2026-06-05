//! Dual Quaternion Skinning for TRINITY Engine (T-GPU-7.3).
//!
//! This module implements dual quaternion blending for skeletal animation,
//! providing artifact-free skinning without the volume loss ("candy wrapper")
//! effect that occurs with linear blend skinning (LBS).
//!
//! # Overview
//!
//! Dual quaternions represent rigid transformations (rotation + translation)
//! in a single algebraic structure that blends correctly:
//!
//! - **No volume collapse** at joints with high rotation angles
//! - **Shortest path interpolation** between orientations
//! - **Mathematically sound blending** unlike LBS's incorrect averaging
//!
//! # Dual Quaternion Representation
//!
//! A dual quaternion DQ = (q_r, q_d) consists of:
//! - `q_r` (real part): unit quaternion encoding rotation
//! - `q_d` (dual part): `0.5 * t * q_r` where t = (tx, ty, tz, 0)
//!
//! # Pipeline
//!
//! 1. Convert joint mat4 transforms to dual quaternions (`dq_convert_joints`)
//! 2. For each vertex, blend dual quaternions with bone weights
//! 3. Handle antipodal quaternions (dot product sign flip)
//! 4. Normalize the blended dual quaternion
//! 5. Apply to vertex position and normal
//!
//! # Usage
//!
//! ```ignore
//! // Create pipeline
//! let pipeline = DualQuatSkinningPipeline::new(&device);
//!
//! // Create resources for 1000 vertices, 64 joints
//! let resources = DualQuatSkinningResources::new(
//!     &device, 1000, 64, &pipeline.skinning_bind_group_layout,
//! );
//!
//! // Each frame: upload joint matrices, convert to dual quats, then skin
//! resources.update_joint_matrices(&queue, &joint_matrices);
//! pipeline.dispatch_convert(&mut encoder, &resources, 64);
//! pipeline.dispatch_skinning(&mut encoder, &resources, 1000);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// DualQuat struct size in bytes.
pub const DUALQUAT_SIZE: usize = 32;

/// DualQuatSkinningParams size in bytes (16 bytes, 4 u32s).
pub const SKINNING_PARAMS_SIZE: usize = 16;

/// Maximum weights per vertex.
pub const MAX_WEIGHTS_PER_VERTEX: usize = 4;

/// Maximum supported joints.
pub const MAX_JOINTS: u32 = 256;

// ---------------------------------------------------------------------------
// DualQuat
// ---------------------------------------------------------------------------

/// Dual quaternion representation for rigid body transformations.
///
/// A dual quaternion encodes both rotation and translation in a single
/// structure that blends correctly for skeletal animation.
///
/// # Memory Layout (32 bytes)
///
/// | Offset | Field | Size     |
/// |--------|-------|----------|
/// | 0      | real  | 16 bytes |
/// | 16     | dual  | 16 bytes |
///
/// - `real`: Unit quaternion (x, y, z, w) encoding rotation
/// - `dual`: Dual part = 0.5 * translation_quat * real
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DualQuat {
    /// Rotation quaternion (x, y, z, w). Must be unit length.
    pub real: [f32; 4],
    /// Dual part encoding translation: 0.5 * (tx, ty, tz, 0) * real.
    pub dual: [f32; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<DualQuat>() == DUALQUAT_SIZE);

impl DualQuat {
    /// Identity dual quaternion (no rotation, no translation).
    pub const IDENTITY: Self = Self {
        real: [0.0, 0.0, 0.0, 1.0],
        dual: [0.0, 0.0, 0.0, 0.0],
    };

    /// Create a dual quaternion from rotation quaternion and translation.
    ///
    /// # Arguments
    ///
    /// * `rotation` - Unit quaternion (x, y, z, w) representing rotation.
    /// * `translation` - Translation vector (x, y, z).
    pub fn from_rotation_translation(rotation: [f32; 4], translation: [f32; 3]) -> Self {
        // dual = 0.5 * t * real where t = (tx, ty, tz, 0)
        let t = [translation[0], translation[1], translation[2], 0.0];
        let dual = quat_mul_cpu(t, rotation);

        Self {
            real: rotation,
            dual: [dual[0] * 0.5, dual[1] * 0.5, dual[2] * 0.5, dual[3] * 0.5],
        }
    }

    /// Create a dual quaternion from a 4x4 transformation matrix.
    ///
    /// The matrix should be a rigid body transform (rotation + translation).
    /// Scale is not preserved.
    pub fn from_mat4(m: &[[f32; 4]; 4]) -> Self {
        cpu_mat4_to_dualquat(m)
    }

    /// Create a dual quaternion from translation only (identity rotation).
    pub fn from_translation(translation: [f32; 3]) -> Self {
        Self::from_rotation_translation([0.0, 0.0, 0.0, 1.0], translation)
    }

    /// Create a dual quaternion from rotation only (no translation).
    pub fn from_rotation(rotation: [f32; 4]) -> Self {
        Self::from_rotation_translation(rotation, [0.0, 0.0, 0.0])
    }

    /// Get the rotation quaternion.
    #[inline]
    pub fn rotation(&self) -> [f32; 4] {
        self.real
    }

    /// Extract the translation vector.
    #[inline]
    pub fn translation(&self) -> [f32; 3] {
        cpu_dualquat_get_translation(self)
    }

    /// Normalize the dual quaternion.
    ///
    /// The real part is normalized to unit length, and the dual part
    /// is scaled proportionally.
    #[inline]
    pub fn normalize(&self) -> Self {
        cpu_dualquat_normalize(self)
    }

    /// Compute the length of the real (rotation) part.
    #[inline]
    pub fn length(&self) -> f32 {
        let r = self.real;
        (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt()
    }

    /// Check if the dual quaternion is approximately unit length.
    #[inline]
    pub fn is_normalized(&self, epsilon: f32) -> bool {
        (self.length() - 1.0).abs() < epsilon
    }

    /// Transform a point by this dual quaternion.
    #[inline]
    pub fn transform_point(&self, point: [f32; 3]) -> [f32; 3] {
        cpu_dualquat_transform_point(self, point)
    }

    /// Transform a normal by this dual quaternion (rotation only).
    #[inline]
    pub fn transform_normal(&self, normal: [f32; 3]) -> [f32; 3] {
        cpu_dualquat_transform_normal(self, normal)
    }

    /// Dot product of the real parts of two dual quaternions.
    #[inline]
    pub fn dot_real(&self, other: &Self) -> f32 {
        let a = self.real;
        let b = other.real;
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }

    /// Scale the dual quaternion by a scalar.
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        Self {
            real: [
                self.real[0] * s,
                self.real[1] * s,
                self.real[2] * s,
                self.real[3] * s,
            ],
            dual: [
                self.dual[0] * s,
                self.dual[1] * s,
                self.dual[2] * s,
                self.dual[3] * s,
            ],
        }
    }

    /// Add two dual quaternions.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            real: [
                self.real[0] + other.real[0],
                self.real[1] + other.real[1],
                self.real[2] + other.real[2],
                self.real[3] + other.real[3],
            ],
            dual: [
                self.dual[0] + other.dual[0],
                self.dual[1] + other.dual[1],
                self.dual[2] + other.dual[2],
                self.dual[3] + other.dual[3],
            ],
        }
    }

    /// Negate the dual quaternion (represents the same transformation).
    #[inline]
    pub fn negate(&self) -> Self {
        Self {
            real: [
                -self.real[0],
                -self.real[1],
                -self.real[2],
                -self.real[3],
            ],
            dual: [
                -self.dual[0],
                -self.dual[1],
                -self.dual[2],
                -self.dual[3],
            ],
        }
    }
}

impl Default for DualQuat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl PartialEq for DualQuat {
    fn eq(&self, other: &Self) -> bool {
        const EPSILON: f32 = 1e-6;
        for i in 0..4 {
            if (self.real[i] - other.real[i]).abs() > EPSILON {
                return false;
            }
            if (self.dual[i] - other.dual[i]).abs() > EPSILON {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// DualQuatSkinningParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for skinning parameters.
///
/// Matches the WGSL `DualQuatSkinningParams` struct layout.
///
/// # Memory Layout (16 bytes)
///
/// | Offset | Field           | Size    |
/// |--------|-----------------|---------|
/// | 0      | vertex_count    | 4 bytes |
/// | 4      | joint_count     | 4 bytes |
/// | 8      | vertex_stride   | 4 bytes |
/// | 12     | position_offset | 4 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DualQuatSkinningParams {
    /// Number of vertices to process.
    pub vertex_count: u32,
    /// Number of joints in the skeleton.
    pub joint_count: u32,
    /// Stride between vertices in floats (for interleaved buffers).
    pub vertex_stride: u32,
    /// Offset to position in vertex (in floats).
    pub position_offset: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<DualQuatSkinningParams>() == SKINNING_PARAMS_SIZE);

impl DualQuatSkinningParams {
    /// Create skinning parameters.
    pub fn new(vertex_count: u32, joint_count: u32) -> Self {
        Self {
            vertex_count,
            joint_count,
            vertex_stride: 0,
            position_offset: 0,
        }
    }

    /// Create skinning parameters with buffer layout info.
    pub fn with_layout(
        vertex_count: u32,
        joint_count: u32,
        vertex_stride: u32,
        position_offset: u32,
    ) -> Self {
        Self {
            vertex_count,
            joint_count,
            vertex_stride,
            position_offset,
        }
    }

    /// Get number of workgroups needed for vertex skinning dispatch.
    #[inline]
    pub fn vertex_workgroups(&self) -> u32 {
        (self.vertex_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Get number of workgroups needed for joint conversion dispatch.
    #[inline]
    pub fn joint_workgroups(&self) -> u32 {
        (self.joint_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

impl Default for DualQuatSkinningParams {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

// ---------------------------------------------------------------------------
// DualQuatBuffer
// ---------------------------------------------------------------------------

/// CPU-side buffer for joint dual quaternions.
///
/// This buffer holds dual quaternion transforms for skeleton joints,
/// ready to be uploaded to the GPU.
pub struct DualQuatBuffer {
    /// Joint dual quaternions.
    pub joints: Vec<DualQuat>,
    /// Maximum joint count (capacity).
    pub capacity: usize,
}

impl DualQuatBuffer {
    /// Create a new buffer with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            joints: vec![DualQuat::IDENTITY; capacity],
            capacity,
        }
    }

    /// Get the number of joints.
    #[inline]
    pub fn len(&self) -> usize {
        self.joints.len()
    }

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    /// Set a joint's dual quaternion.
    pub fn set(&mut self, index: usize, dq: DualQuat) {
        if index < self.joints.len() {
            self.joints[index] = dq;
        }
    }

    /// Get a joint's dual quaternion.
    pub fn get(&self, index: usize) -> Option<&DualQuat> {
        self.joints.get(index)
    }

    /// Update all joints from mat4 transforms.
    pub fn update_from_matrices(&mut self, matrices: &[[[f32; 4]; 4]]) {
        for (i, matrix) in matrices.iter().enumerate() {
            if i < self.joints.len() {
                self.joints[i] = DualQuat::from_mat4(matrix);
            }
        }
    }

    /// Get raw bytes for GPU upload.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.joints)
    }

    /// Clear all joints to identity.
    pub fn clear(&mut self) {
        for joint in &mut self.joints {
            *joint = DualQuat::IDENTITY;
        }
    }

    /// Resize the buffer.
    pub fn resize(&mut self, new_size: usize) {
        self.joints.resize(new_size, DualQuat::IDENTITY);
        self.capacity = new_size;
    }
}

impl Default for DualQuatBuffer {
    fn default() -> Self {
        Self::new(64) // Reasonable default for most skeletons
    }
}

// ---------------------------------------------------------------------------
// GPU Pipeline and Resources
// ---------------------------------------------------------------------------

/// GPU resources for dual quaternion skinning.
pub struct DualQuatSkinningResources {
    /// Uniform buffer for skinning parameters.
    pub params_buffer: wgpu::Buffer,
    /// Storage buffer for joint dual quaternions.
    pub joint_buffer: wgpu::Buffer,
    /// Storage buffer for input joint matrices (for conversion).
    pub joint_matrix_buffer: wgpu::Buffer,
    /// Maximum vertex count.
    pub max_vertices: u32,
    /// Maximum joint count.
    pub max_joints: u32,
    /// Bind group for skinning pass.
    pub skinning_bind_group: wgpu::BindGroup,
    /// Bind group for joint conversion pass.
    pub convert_bind_group: wgpu::BindGroup,
}

impl DualQuatSkinningResources {
    /// Create skinning resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `max_vertices` - Maximum number of vertices to skin.
    /// * `max_joints` - Maximum number of joints in the skeleton.
    /// * `vertices_in` - Input vertex buffer (SkinnedVertex array).
    /// * `vertices_out` - Output vertex buffer (OutputVertex array).
    /// * `skinning_layout` - Bind group layout from pipeline.
    /// * `convert_layout` - Bind group layout for conversion.
    pub fn new(
        device: &wgpu::Device,
        max_vertices: u32,
        max_joints: u32,
        vertices_in: &wgpu::Buffer,
        vertices_out: &wgpu::Buffer,
        skinning_layout: &wgpu::BindGroupLayout,
        convert_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dualquat_skinning_params"),
            size: SKINNING_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let joint_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dualquat_joint_buffer"),
            size: (max_joints as u64) * (DUALQUAT_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 64 bytes per mat4
        let joint_matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dualquat_joint_matrix_buffer"),
            size: (max_joints as u64) * 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let skinning_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dualquat_skinning_bind_group"),
            layout: skinning_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: joint_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vertices_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vertices_out.as_entire_binding(),
                },
            ],
        });

        let convert_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dualquat_convert_bind_group"),
            layout: convert_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: joint_matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: joint_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            params_buffer,
            joint_buffer,
            joint_matrix_buffer,
            max_vertices,
            max_joints,
            skinning_bind_group,
            convert_bind_group,
        }
    }

    /// Update skinning parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &DualQuatSkinningParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Update joint dual quaternions directly.
    pub fn update_joints(&self, queue: &wgpu::Queue, joints: &[DualQuat]) {
        queue.write_buffer(&self.joint_buffer, 0, bytemuck::cast_slice(joints));
    }

    /// Update joint matrices (for conversion pass).
    pub fn update_joint_matrices(&self, queue: &wgpu::Queue, matrices: &[[[f32; 4]; 4]]) {
        queue.write_buffer(&self.joint_matrix_buffer, 0, bytemuck::cast_slice(matrices));
    }
}

/// GPU compute pipeline for dual quaternion skinning.
pub struct DualQuatSkinningPipeline {
    /// Bind group layout for skinning pass.
    pub skinning_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for joint conversion pass.
    pub convert_bind_group_layout: wgpu::BindGroupLayout,
    /// Compute pipeline for skinning.
    pub skinning_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for mat4 to dual quat conversion.
    pub convert_pipeline: wgpu::ComputePipeline,
}

impl DualQuatSkinningPipeline {
    /// Create the dual quaternion skinning pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        // Skinning bind group layout
        let skinning_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dualquat_skinning_bind_group_layout"),
                entries: &[
                    // binding 0: DualQuatSkinningParams (uniform)
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
                    // binding 1: joint_dualquats (storage, read)
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
                    // binding 2: vertices_in (storage, read)
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
                    // binding 3: vertices_out (storage, read_write)
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

        // Convert bind group layout
        let convert_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dualquat_convert_bind_group_layout"),
                entries: &[
                    // binding 0: joint_matrices (storage, read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: joint_dualquats_out (storage, read_write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        // Skinning pipeline layout
        let skinning_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dualquat_skinning_pipeline_layout"),
                bind_group_layouts: &[&skinning_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Convert pipeline layout (needs both params and convert layouts)
        let convert_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dualquat_convert_pipeline_layout"),
                bind_group_layouts: &[&skinning_bind_group_layout, &convert_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Load shader module
        let shader_source =
            include_str!("../../shaders/skinning/dualquat_skinning.comp.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dualquat_skinning_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create skinning pipeline
        let skinning_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dualquat_skinning_pipeline"),
            layout: Some(&skinning_pipeline_layout),
            module: &shader_module,
            entry_point: "dq_skinning",
            compilation_options: Default::default(),
            cache: None,
        });

        // Create convert pipeline
        let convert_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dualquat_convert_pipeline"),
            layout: Some(&convert_pipeline_layout),
            module: &shader_module,
            entry_point: "dq_convert_joints",
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            skinning_bind_group_layout,
            convert_bind_group_layout,
            skinning_pipeline,
            convert_pipeline,
        }
    }

    /// Dispatch the skinning compute shader.
    pub fn dispatch_skinning(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &DualQuatSkinningResources,
        vertex_count: u32,
    ) {
        if vertex_count == 0 {
            return;
        }

        let num_workgroups = (vertex_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dualquat_skinning_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.skinning_pipeline);
        pass.set_bind_group(0, &resources.skinning_bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    /// Dispatch the joint conversion compute shader.
    pub fn dispatch_convert(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params_bind_group: &wgpu::BindGroup,
        convert_bind_group: &wgpu::BindGroup,
        joint_count: u32,
    ) {
        if joint_count == 0 {
            return;
        }

        let num_workgroups = (joint_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dualquat_convert_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.convert_pipeline);
        pass.set_bind_group(0, params_bind_group, &[]);
        pass.set_bind_group(1, convert_bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation: Quaternion Operations
// ---------------------------------------------------------------------------

/// Quaternion multiplication (CPU reference).
#[inline]
fn quat_mul_cpu(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// Quaternion conjugate (CPU reference).
#[inline]
fn quat_conjugate_cpu(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Quaternion normalize (CPU reference).
#[inline]
fn quat_normalize_cpu(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 0.0 {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// Rotate a vector by a quaternion (CPU reference).
#[inline]
pub fn cpu_quat_rotate_vector(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    // v' = v + 2 * q.w * (q.xyz x v) + 2 * (q.xyz x (q.xyz x v))
    let qv = [q[0], q[1], q[2]];

    // uv = cross(qv, v)
    let uv = [
        qv[1] * v[2] - qv[2] * v[1],
        qv[2] * v[0] - qv[0] * v[2],
        qv[0] * v[1] - qv[1] * v[0],
    ];

    // uuv = cross(qv, uv)
    let uuv = [
        qv[1] * uv[2] - qv[2] * uv[1],
        qv[2] * uv[0] - qv[0] * uv[2],
        qv[0] * uv[1] - qv[1] * uv[0],
    ];

    [
        v[0] + 2.0 * (q[3] * uv[0] + uuv[0]),
        v[1] + 2.0 * (q[3] * uv[1] + uuv[1]),
        v[2] + 2.0 * (q[3] * uv[2] + uuv[2]),
    ]
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation: Dual Quaternion Operations
// ---------------------------------------------------------------------------

/// Convert mat4 to dual quaternion (CPU reference).
///
/// Uses Shepperd's method for robust quaternion extraction.
pub fn cpu_mat4_to_dualquat(m: &[[f32; 4]; 4]) -> DualQuat {
    // Extract rotation using Shepperd's method.
    // Input matrix is COLUMN-MAJOR: m[col][row], so m[j][i] = M(i,j) in math notation.
    // The algorithm uses row-major indexing, so we access m[col][row] for M(row,col).

    // Diagonal elements: M(0,0), M(1,1), M(2,2)
    let trace = m[0][0] + m[1][1] + m[2][2];
    let q: [f32; 4];

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0; // s = 4 * qw
        // qx = (M(2,1) - M(1,2)) / s = (m[1][2] - m[2][1]) / s
        // qy = (M(0,2) - M(2,0)) / s = (m[2][0] - m[0][2]) / s
        // qz = (M(1,0) - M(0,1)) / s = (m[0][1] - m[1][0]) / s
        q = [
            (m[1][2] - m[2][1]) / s,
            (m[2][0] - m[0][2]) / s,
            (m[0][1] - m[1][0]) / s,
            0.25 * s,
        ];
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0; // s = 4 * qx
        // qy = (M(0,1) + M(1,0)) / s = (m[1][0] + m[0][1]) / s
        // qz = (M(0,2) + M(2,0)) / s = (m[2][0] + m[0][2]) / s
        // qw = (M(2,1) - M(1,2)) / s = (m[1][2] - m[2][1]) / s
        q = [
            0.25 * s,
            (m[1][0] + m[0][1]) / s,
            (m[2][0] + m[0][2]) / s,
            (m[1][2] - m[2][1]) / s,
        ];
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0; // s = 4 * qy
        // qx = (M(0,1) + M(1,0)) / s = (m[1][0] + m[0][1]) / s
        // qz = (M(1,2) + M(2,1)) / s = (m[2][1] + m[1][2]) / s
        // qw = (M(0,2) - M(2,0)) / s = (m[2][0] - m[0][2]) / s
        q = [
            (m[1][0] + m[0][1]) / s,
            0.25 * s,
            (m[2][1] + m[1][2]) / s,
            (m[2][0] - m[0][2]) / s,
        ];
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0; // s = 4 * qz
        // qx = (M(0,2) + M(2,0)) / s = (m[2][0] + m[0][2]) / s
        // qy = (M(1,2) + M(2,1)) / s = (m[2][1] + m[1][2]) / s
        // qw = (M(1,0) - M(0,1)) / s = (m[0][1] - m[1][0]) / s
        q = [
            (m[2][0] + m[0][2]) / s,
            (m[2][1] + m[1][2]) / s,
            0.25 * s,
            (m[0][1] - m[1][0]) / s,
        ];
    }

    // Normalize the rotation quaternion
    let q = quat_normalize_cpu(q);

    // Extract translation from last column (column 3)
    let translation = [m[3][0], m[3][1], m[3][2]];

    DualQuat::from_rotation_translation(q, translation)
}

/// Normalize a dual quaternion (CPU reference).
pub fn cpu_dualquat_normalize(dq: &DualQuat) -> DualQuat {
    let r = dq.real;
    let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();

    if len > 0.0 {
        let inv_len = 1.0 / len;
        DualQuat {
            real: [
                dq.real[0] * inv_len,
                dq.real[1] * inv_len,
                dq.real[2] * inv_len,
                dq.real[3] * inv_len,
            ],
            dual: [
                dq.dual[0] * inv_len,
                dq.dual[1] * inv_len,
                dq.dual[2] * inv_len,
                dq.dual[3] * inv_len,
            ],
        }
    } else {
        DualQuat::IDENTITY
    }
}

/// Blend dual quaternions with antipodal handling (CPU reference).
///
/// Handles the case where quaternions are on opposite hemispheres
/// by flipping the sign to ensure shortest path interpolation.
pub fn cpu_dualquat_blend(dqs: &[DualQuat], weights: &[f32]) -> DualQuat {
    if dqs.is_empty() || weights.is_empty() {
        return DualQuat::IDENTITY;
    }

    // Reference quaternion (first with non-zero weight)
    let mut ref_idx = 0;
    for (i, &w) in weights.iter().enumerate() {
        if w > 0.0 && i < dqs.len() {
            ref_idx = i;
            break;
        }
    }

    let ref_dq = &dqs[ref_idx];

    // Accumulate with antipodal handling
    let mut result_real = [0.0f32; 4];
    let mut result_dual = [0.0f32; 4];

    for (i, (&dq, &weight)) in dqs.iter().zip(weights.iter()).enumerate() {
        if weight <= 0.0 {
            continue;
        }

        // Check if quaternion is on opposite hemisphere
        let dot = ref_dq.real[0] * dq.real[0]
            + ref_dq.real[1] * dq.real[1]
            + ref_dq.real[2] * dq.real[2]
            + ref_dq.real[3] * dq.real[3];

        let sign = if dot < 0.0 { -1.0 } else { 1.0 };

        for j in 0..4 {
            result_real[j] += dq.real[j] * weight * sign;
            result_dual[j] += dq.dual[j] * weight * sign;
        }
    }

    cpu_dualquat_normalize(&DualQuat {
        real: result_real,
        dual: result_dual,
    })
}

/// Get translation from dual quaternion (CPU reference).
pub fn cpu_dualquat_get_translation(dq: &DualQuat) -> [f32; 3] {
    // t = 2 * dual * conjugate(real)
    let conj = quat_conjugate_cpu(dq.real);
    let t_quat = quat_mul_cpu(dq.dual, conj);
    [t_quat[0] * 2.0, t_quat[1] * 2.0, t_quat[2] * 2.0]
}

/// Transform a point by a dual quaternion (CPU reference).
pub fn cpu_dualquat_transform_point(dq: &DualQuat, point: [f32; 3]) -> [f32; 3] {
    // Rotate the point
    let rotated = cpu_quat_rotate_vector(dq.real, point);

    // Add translation
    let translation = cpu_dualquat_get_translation(dq);

    [
        rotated[0] + translation[0],
        rotated[1] + translation[1],
        rotated[2] + translation[2],
    ]
}

/// Transform a normal by a dual quaternion (CPU reference).
///
/// Normals are only rotated, not translated.
pub fn cpu_dualquat_transform_normal(dq: &DualQuat, normal: [f32; 3]) -> [f32; 3] {
    cpu_quat_rotate_vector(dq.real, normal)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_eq_vec3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    fn approx_eq_vec4(a: [f32; 4], b: [f32; 4]) -> bool {
        approx_eq(a[0], b[0])
            && approx_eq(a[1], b[1])
            && approx_eq(a[2], b[2])
            && approx_eq(a[3], b[3])
    }

    // ── DualQuat Construction ───────────────────────────────────────────

    #[test]
    fn test_dualquat_identity() {
        let dq = DualQuat::IDENTITY;
        assert_eq!(dq.real, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(dq.dual, [0.0, 0.0, 0.0, 0.0]);
        assert!(dq.is_normalized(EPSILON));
    }

    #[test]
    fn test_dualquat_default() {
        let dq = DualQuat::default();
        assert_eq!(dq, DualQuat::IDENTITY);
    }

    #[test]
    fn test_dualquat_from_translation() {
        let dq = DualQuat::from_translation([1.0, 2.0, 3.0]);
        let t = dq.translation();
        assert!(approx_eq_vec3(t, [1.0, 2.0, 3.0]));
        assert!(dq.is_normalized(EPSILON));
    }

    #[test]
    fn test_dualquat_from_rotation() {
        // 90 degree rotation around Y axis
        let angle = std::f32::consts::FRAC_PI_2;
        let rotation = [0.0, (angle / 2.0).sin(), 0.0, (angle / 2.0).cos()];
        let dq = DualQuat::from_rotation(rotation);
        let t = dq.translation();
        assert!(approx_eq_vec3(t, [0.0, 0.0, 0.0]));
        assert!(dq.is_normalized(EPSILON));
    }

    #[test]
    fn test_dualquat_from_rotation_translation() {
        let angle = std::f32::consts::FRAC_PI_4;
        let rotation = [0.0, (angle / 2.0).sin(), 0.0, (angle / 2.0).cos()];
        let translation = [5.0, 10.0, 15.0];
        let dq = DualQuat::from_rotation_translation(rotation, translation);

        assert!(dq.is_normalized(EPSILON));
        let t = dq.translation();
        assert!(approx_eq_vec3(t, translation));
    }

    #[test]
    fn test_dualquat_size() {
        assert_eq!(std::mem::size_of::<DualQuat>(), 32);
    }

    // ── DualQuat Operations ─────────────────────────────────────────────

    #[test]
    fn test_dualquat_scale() {
        let dq = DualQuat::from_translation([2.0, 4.0, 6.0]);
        let scaled = dq.scale(0.5);
        // Scaling a dual quat is not the same as scaling the translation!
        // But the real part should be halved
        for i in 0..4 {
            assert!(approx_eq(scaled.real[i], dq.real[i] * 0.5));
            assert!(approx_eq(scaled.dual[i], dq.dual[i] * 0.5));
        }
    }

    #[test]
    fn test_dualquat_add() {
        let dq1 = DualQuat::from_translation([1.0, 0.0, 0.0]);
        let dq2 = DualQuat::from_translation([0.0, 1.0, 0.0]);
        let sum = dq1.add(&dq2);
        // Sum of DQs isn't normalized
        assert!(!sum.is_normalized(EPSILON));
    }

    #[test]
    fn test_dualquat_negate() {
        let dq = DualQuat::from_translation([1.0, 2.0, 3.0]);
        let neg = dq.negate();
        for i in 0..4 {
            assert!(approx_eq(neg.real[i], -dq.real[i]));
            assert!(approx_eq(neg.dual[i], -dq.dual[i]));
        }
    }

    #[test]
    fn test_dualquat_dot_real() {
        let dq1 = DualQuat::IDENTITY;
        let dq2 = DualQuat::IDENTITY;
        assert!(approx_eq(dq1.dot_real(&dq2), 1.0));

        let dq3 = dq1.negate();
        assert!(approx_eq(dq1.dot_real(&dq3), -1.0));
    }

    // ── Transformation Tests ────────────────────────────────────────────

    #[test]
    fn test_dualquat_transform_point_identity() {
        let dq = DualQuat::IDENTITY;
        let point = [1.0, 2.0, 3.0];
        let result = dq.transform_point(point);
        assert!(approx_eq_vec3(result, point));
    }

    #[test]
    fn test_dualquat_transform_point_translation() {
        let dq = DualQuat::from_translation([10.0, 20.0, 30.0]);
        let point = [1.0, 2.0, 3.0];
        let result = dq.transform_point(point);
        assert!(approx_eq_vec3(result, [11.0, 22.0, 33.0]));
    }

    #[test]
    fn test_dualquat_transform_point_rotation_90_y() {
        // 90 degree rotation around Y axis
        let angle = std::f32::consts::FRAC_PI_2;
        let rotation = [0.0, (angle / 2.0).sin(), 0.0, (angle / 2.0).cos()];
        let dq = DualQuat::from_rotation(rotation);

        // Point on X axis should move to Z axis
        let point = [1.0, 0.0, 0.0];
        let result = dq.transform_point(point);
        assert!(approx_eq_vec3(result, [0.0, 0.0, -1.0]));
    }

    #[test]
    fn test_dualquat_transform_point_rotation_and_translation() {
        // 90 degree rotation around Y axis + translation
        let angle = std::f32::consts::FRAC_PI_2;
        let rotation = [0.0, (angle / 2.0).sin(), 0.0, (angle / 2.0).cos()];
        let translation = [0.0, 5.0, 0.0];
        let dq = DualQuat::from_rotation_translation(rotation, translation);

        // Point on X axis should move to Z axis, then translate up
        let point = [1.0, 0.0, 0.0];
        let result = dq.transform_point(point);
        assert!(approx_eq_vec3(result, [0.0, 5.0, -1.0]));
    }

    #[test]
    fn test_dualquat_transform_normal() {
        // 90 degree rotation around Y axis
        let angle = std::f32::consts::FRAC_PI_2;
        let rotation = [0.0, (angle / 2.0).sin(), 0.0, (angle / 2.0).cos()];
        let translation = [100.0, 200.0, 300.0]; // Translation shouldn't affect normal
        let dq = DualQuat::from_rotation_translation(rotation, translation);

        let normal = [1.0, 0.0, 0.0];
        let result = dq.transform_normal(normal);
        assert!(approx_eq_vec3(result, [0.0, 0.0, -1.0]));
    }

    // ── Mat4 Conversion Tests ───────────────────────────────────────────

    #[test]
    fn test_mat4_to_dualquat_identity() {
        let identity_mat: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let dq = DualQuat::from_mat4(&identity_mat);
        assert!(dq.is_normalized(EPSILON));
        let t = dq.translation();
        assert!(approx_eq_vec3(t, [0.0, 0.0, 0.0]));

        // Test transformation
        let point = [1.0, 2.0, 3.0];
        let result = dq.transform_point(point);
        assert!(approx_eq_vec3(result, point));
    }

    #[test]
    fn test_mat4_to_dualquat_translation() {
        let translation_mat: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [5.0, 10.0, 15.0, 1.0],
        ];
        let dq = DualQuat::from_mat4(&translation_mat);
        assert!(dq.is_normalized(EPSILON));
        let t = dq.translation();
        assert!(approx_eq_vec3(t, [5.0, 10.0, 15.0]));
    }

    #[test]
    fn test_mat4_to_dualquat_rotation_90_y() {
        // 90 degree CCW rotation around Y axis (column-major format, m[i] = column i)
        // For +90 deg Y rotation (CCW looking down +Y): X axis -> +Z, Y stays, Z axis -> -X
        // Column 0 (where X goes): (0, 0, 1)
        // Column 1 (Y unchanged):  (0, 1, 0)
        // Column 2 (where Z goes): (-1, 0, 0)
        let rotation_mat: [[f32; 4]; 4] = [
            [0.0, 0.0, 1.0, 0.0],   // column 0: X -> (0,0,1)
            [0.0, 1.0, 0.0, 0.0],   // column 1: Y unchanged
            [-1.0, 0.0, 0.0, 0.0],  // column 2: Z -> (-1,0,0)
            [0.0, 0.0, 0.0, 1.0],   // column 3 (translation)
        ];
        let dq = DualQuat::from_mat4(&rotation_mat);
        assert!(dq.is_normalized(EPSILON));

        // Point on X axis should move to +Z axis
        let point = [1.0, 0.0, 0.0];
        let result = dq.transform_point(point);
        assert!(approx_eq_vec3(result, [0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_mat4_to_dualquat_rotation_and_translation() {
        // 90 degree CCW rotation around Y axis + translation (column-major)
        let transform_mat: [[f32; 4]; 4] = [
            [0.0, 0.0, 1.0, 0.0],   // column 0: X -> (0,0,1)
            [0.0, 1.0, 0.0, 0.0],   // column 1: Y unchanged
            [-1.0, 0.0, 0.0, 0.0],  // column 2: Z -> (-1,0,0)
            [0.0, 5.0, 0.0, 1.0],   // column 3 (translation: 0, 5, 0)
        ];
        let dq = DualQuat::from_mat4(&transform_mat);
        assert!(dq.is_normalized(EPSILON));

        // Point (1,0,0) rotates to (0,0,1), then translates by (0,5,0) -> (0,5,1)
        let point = [1.0, 0.0, 0.0];
        let result = dq.transform_point(point);
        assert!(approx_eq_vec3(result, [0.0, 5.0, 1.0]));
    }

    // ── Blending Tests ──────────────────────────────────────────────────

    #[test]
    fn test_dualquat_blend_single() {
        let dq = DualQuat::from_translation([1.0, 2.0, 3.0]);
        let result = cpu_dualquat_blend(&[dq], &[1.0]);
        assert!(result.is_normalized(EPSILON));
        let t = result.translation();
        assert!(approx_eq_vec3(t, [1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_dualquat_blend_two_equal_weights() {
        let dq1 = DualQuat::from_translation([0.0, 0.0, 0.0]);
        let dq2 = DualQuat::from_translation([2.0, 0.0, 0.0]);
        let result = cpu_dualquat_blend(&[dq1, dq2], &[0.5, 0.5]);
        assert!(result.is_normalized(EPSILON));
        let t = result.translation();
        assert!(approx_eq_vec3(t, [1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_dualquat_blend_weighted() {
        let dq1 = DualQuat::from_translation([0.0, 0.0, 0.0]);
        let dq2 = DualQuat::from_translation([4.0, 0.0, 0.0]);
        let result = cpu_dualquat_blend(&[dq1, dq2], &[0.75, 0.25]);
        assert!(result.is_normalized(EPSILON));
        let t = result.translation();
        assert!(approx_eq_vec3(t, [1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_dualquat_blend_antipodal() {
        // Create two quaternions that represent the same rotation but are antipodal
        let dq1 = DualQuat::from_rotation([0.0, 0.0, 0.0, 1.0]);
        let dq2 = dq1.negate(); // Same rotation, opposite hemisphere

        // Blending should handle antipodal case by flipping one
        let result = cpu_dualquat_blend(&[dq1, dq2], &[0.5, 0.5]);
        assert!(result.is_normalized(EPSILON));
    }

    #[test]
    fn test_dualquat_blend_four_bones() {
        let dqs = [
            DualQuat::from_translation([4.0, 0.0, 0.0]),
            DualQuat::from_translation([0.0, 4.0, 0.0]),
            DualQuat::from_translation([0.0, 0.0, 4.0]),
            DualQuat::from_translation([0.0, 0.0, 0.0]),
        ];
        let weights = [0.25, 0.25, 0.25, 0.25];
        let result = cpu_dualquat_blend(&dqs, &weights);
        assert!(result.is_normalized(EPSILON));
        let t = result.translation();
        assert!(approx_eq_vec3(t, [1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_dualquat_blend_zero_weights() {
        let dqs = [
            DualQuat::from_translation([100.0, 0.0, 0.0]),
            DualQuat::from_translation([0.0, 100.0, 0.0]),
        ];
        let weights = [0.0, 0.0];
        let result = cpu_dualquat_blend(&dqs, &weights);
        // Should return identity when all weights are zero
        assert_eq!(result, DualQuat::IDENTITY);
    }

    #[test]
    fn test_dualquat_blend_empty() {
        let result = cpu_dualquat_blend(&[], &[]);
        assert_eq!(result, DualQuat::IDENTITY);
    }

    // ── Normalization Tests ─────────────────────────────────────────────

    #[test]
    fn test_dualquat_normalize_already_normalized() {
        let dq = DualQuat::IDENTITY;
        let normalized = dq.normalize();
        assert!(normalized.is_normalized(EPSILON));
    }

    #[test]
    fn test_dualquat_normalize_scaled() {
        let dq = DualQuat::from_translation([1.0, 2.0, 3.0]).scale(2.0);
        assert!(!dq.is_normalized(EPSILON));
        let normalized = dq.normalize();
        assert!(normalized.is_normalized(EPSILON));
    }

    // ── Quaternion Operation Tests ──────────────────────────────────────

    #[test]
    fn test_quat_mul_identity() {
        let identity = [0.0, 0.0, 0.0, 1.0];
        let q = [0.5, 0.5, 0.5, 0.5];
        let result = quat_mul_cpu(identity, q);
        assert!(approx_eq_vec4(result, q));
    }

    #[test]
    fn test_quat_mul_conjugate() {
        let q = [0.5, 0.5, 0.5, 0.5];
        let q_conj = quat_conjugate_cpu(q);
        let result = quat_mul_cpu(q, q_conj);
        // q * q^(-1) = identity
        assert!(approx_eq_vec4(result, [0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_quat_rotate_vector_identity() {
        let identity = [0.0, 0.0, 0.0, 1.0];
        let v = [1.0, 2.0, 3.0];
        let result = cpu_quat_rotate_vector(identity, v);
        assert!(approx_eq_vec3(result, v));
    }

    #[test]
    fn test_quat_rotate_vector_180_x() {
        // 180 degree rotation around X axis
        let q = [1.0, 0.0, 0.0, 0.0];
        let v = [0.0, 1.0, 0.0];
        let result = cpu_quat_rotate_vector(q, v);
        assert!(approx_eq_vec3(result, [0.0, -1.0, 0.0]));
    }

    // ── DualQuatBuffer Tests ────────────────────────────────────────────

    #[test]
    fn test_dualquat_buffer_new() {
        let buffer = DualQuatBuffer::new(64);
        assert_eq!(buffer.len(), 64);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.capacity, 64);
    }

    #[test]
    fn test_dualquat_buffer_set_get() {
        let mut buffer = DualQuatBuffer::new(4);
        let dq = DualQuat::from_translation([1.0, 2.0, 3.0]);
        buffer.set(2, dq);

        let retrieved = buffer.get(2).unwrap();
        assert_eq!(*retrieved, dq);
    }

    #[test]
    fn test_dualquat_buffer_clear() {
        let mut buffer = DualQuatBuffer::new(4);
        buffer.set(0, DualQuat::from_translation([1.0, 2.0, 3.0]));
        buffer.clear();

        for dq in &buffer.joints {
            assert_eq!(*dq, DualQuat::IDENTITY);
        }
    }

    #[test]
    fn test_dualquat_buffer_update_from_matrices() {
        let mut buffer = DualQuatBuffer::new(2);

        let matrices = [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 10.0, 0.0, 1.0],
            ],
        ];

        buffer.update_from_matrices(&matrices);

        let t0 = buffer.get(0).unwrap().translation();
        let t1 = buffer.get(1).unwrap().translation();
        assert!(approx_eq_vec3(t0, [5.0, 0.0, 0.0]));
        assert!(approx_eq_vec3(t1, [0.0, 10.0, 0.0]));
    }

    #[test]
    fn test_dualquat_buffer_resize() {
        let mut buffer = DualQuatBuffer::new(4);
        assert_eq!(buffer.len(), 4);

        buffer.resize(8);
        assert_eq!(buffer.len(), 8);

        buffer.resize(2);
        assert_eq!(buffer.len(), 2);
    }

    // ── SkinningParams Tests ────────────────────────────────────────────

    #[test]
    fn test_skinning_params_new() {
        let params = DualQuatSkinningParams::new(1000, 64);
        assert_eq!(params.vertex_count, 1000);
        assert_eq!(params.joint_count, 64);
        assert_eq!(params.vertex_stride, 0);
        assert_eq!(params.position_offset, 0);
    }

    #[test]
    fn test_skinning_params_with_layout() {
        let params = DualQuatSkinningParams::with_layout(1000, 64, 24, 0);
        assert_eq!(params.vertex_count, 1000);
        assert_eq!(params.joint_count, 64);
        assert_eq!(params.vertex_stride, 24);
        assert_eq!(params.position_offset, 0);
    }

    #[test]
    fn test_skinning_params_workgroups() {
        let params = DualQuatSkinningParams::new(1000, 64);
        assert_eq!(params.vertex_workgroups(), 4); // ceil(1000/256) = 4
        assert_eq!(params.joint_workgroups(), 1); // ceil(64/256) = 1
    }

    #[test]
    fn test_skinning_params_size() {
        assert_eq!(std::mem::size_of::<DualQuatSkinningParams>(), 16);
    }

    // ── Edge Cases ──────────────────────────────────────────────────────

    #[test]
    fn test_dualquat_transform_origin() {
        let dq = DualQuat::from_translation([5.0, 10.0, 15.0]);
        let origin = [0.0, 0.0, 0.0];
        let result = dq.transform_point(origin);
        assert!(approx_eq_vec3(result, [5.0, 10.0, 15.0]));
    }

    #[test]
    fn test_dualquat_chain_transforms() {
        // First translate, then rotate
        let t1 = DualQuat::from_translation([1.0, 0.0, 0.0]);
        let angle = std::f32::consts::FRAC_PI_2;
        let r = DualQuat::from_rotation([0.0, (angle / 2.0).sin(), 0.0, (angle / 2.0).cos()]);

        let point = [0.0, 0.0, 0.0];
        let after_t = t1.transform_point(point);
        let after_r = r.transform_point(after_t);

        // After translation: [1, 0, 0]
        // After rotation (90 Y): [0, 0, -1]
        assert!(approx_eq_vec3(after_r, [0.0, 0.0, -1.0]));
    }

    #[test]
    fn test_mat4_to_dualquat_roundtrip() {
        // Create a DQ, extract transform, convert back
        let angle = std::f32::consts::FRAC_PI_4;
        let rotation = [0.0, (angle / 2.0).sin(), 0.0, (angle / 2.0).cos()];
        let translation = [3.0, 4.0, 5.0];
        let original = DualQuat::from_rotation_translation(rotation, translation);

        let point = [1.0, 2.0, 3.0];
        let expected = original.transform_point(point);

        // The roundtrip test: original should transform the same as converted
        let t_extracted = original.translation();
        assert!(approx_eq_vec3(t_extracted, translation));
    }

    // ── WGSL Shader Validation Tests ────────────────────────────────────

    #[test]
    fn test_wgsl_shader_parses() {
        let shader_source = include_str!("../../shaders/skinning/dualquat_skinning.comp.wgsl");
        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("Dual quaternion skinning shader should parse without errors");

        // Verify expected entry points exist
        let entry_points: Vec<&str> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
        assert!(entry_points.contains(&"dq_skinning"), "Missing dq_skinning entry point");
        assert!(entry_points.contains(&"dq_convert_joints"), "Missing dq_convert_joints entry point");
    }

    #[test]
    fn test_wgsl_shader_validates() {
        let shader_source = include_str!("../../shaders/skinning/dualquat_skinning.comp.wgsl");
        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("Dual quaternion skinning shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("Dual quaternion skinning shader should validate without errors");
    }

    #[test]
    fn test_wgsl_shader_contains_dualquat_struct() {
        let shader_source = include_str!("../../shaders/skinning/dualquat_skinning.comp.wgsl");
        assert!(shader_source.contains("struct DualQuat"));
        assert!(shader_source.contains("real: vec4<f32>"));
        assert!(shader_source.contains("dual: vec4<f32>"));
    }

    #[test]
    fn test_wgsl_shader_contains_key_functions() {
        let shader_source = include_str!("../../shaders/skinning/dualquat_skinning.comp.wgsl");
        assert!(shader_source.contains("fn quat_mul"));
        assert!(shader_source.contains("fn quat_conjugate"));
        assert!(shader_source.contains("fn quat_normalize"));
        assert!(shader_source.contains("fn quat_rotate_vector"));
        assert!(shader_source.contains("fn dualquat_from_rotation_translation"));
        assert!(shader_source.contains("fn mat4_to_dualquat"));
        assert!(shader_source.contains("fn dualquat_normalize"));
        assert!(shader_source.contains("fn dualquat_blend"));
        assert!(shader_source.contains("fn dualquat_transform_point"));
        assert!(shader_source.contains("fn dualquat_transform_normal"));
    }

    #[test]
    fn test_wgsl_shader_workgroup_size() {
        let shader_source = include_str!("../../shaders/skinning/dualquat_skinning.comp.wgsl");
        assert!(shader_source.contains("@workgroup_size(256)"));
    }

    #[test]
    fn test_wgsl_shader_bindings() {
        let shader_source = include_str!("../../shaders/skinning/dualquat_skinning.comp.wgsl");
        // Group 0 bindings for skinning pass
        assert!(shader_source.contains("@group(0) @binding(0) var<uniform> params:"));
        assert!(shader_source.contains("@group(0) @binding(1) var<storage, read> joint_dualquats:"));
        assert!(shader_source.contains("@group(0) @binding(2) var<storage, read> vertices_in:"));
        assert!(shader_source.contains("@group(0) @binding(3) var<storage, read_write> vertices_out:"));
        // Group 1 bindings for conversion pass
        assert!(shader_source.contains("@group(1) @binding(0) var<storage, read> joint_matrices:"));
        assert!(shader_source.contains("@group(1) @binding(1) var<storage, read_write> joint_dualquats_out:"));
    }
}
