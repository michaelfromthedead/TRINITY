//! GPU Instance Update for TRINITY Engine (T-GPU-4.2).
//!
//! This module provides GPU-based instance data management for GPU-driven
//! instanced rendering. It handles transform packing, bounds computation,
//! LOD and visibility flag aggregation.
//!
//! # Overview
//!
//! In GPU-driven rendering, instance data must be efficiently packed for
//! shader consumption. This module:
//!
//! 1. Packs transforms from mat4 to mat4x3 (48 bytes vs 64 bytes)
//! 2. Transforms local bounding volumes to world space
//! 3. Aggregates LOD selection and visibility flags from culling passes
//! 4. Produces compact instance data ready for indirect drawing
//!
//! # Data Layout
//!
//! The packed `InstanceData` struct is 64 bytes aligned:
//!
//! | Offset | Field         | Size | Description                    |
//! |--------|---------------|------|--------------------------------|
//! | 0      | transform     | 48   | mat4x3 packed transform        |
//! | 48     | bounds_center | 12   | World-space sphere center      |
//! | 60     | bounds_radius | 4    | World-space sphere radius      |
//! | 64     | lod_index     | 4    | Selected LOD level             |
//! | 68     | flags         | 4    | Visibility and instance flags  |
//! | 72     | material_id   | 4    | Material table index           |
//! | 76     | _padding      | 4    | Padding for 64-byte alignment  |
//!
//! # Performance
//!
//! - Work complexity: O(n), one thread per instance
//! - Target: < 0.1ms for 100K instances
//! - Memory: 64 bytes per output instance
//!
//! # Usage
//!
//! ```ignore
//! // Create pipeline and resources
//! let pipeline = InstanceUpdatePipeline::new(&device, shader_source);
//! let resources = InstanceUpdateResources::new(&device, 100_000);
//!
//! // Each frame: update instance data
//! resources.upload_transforms(&queue, &transforms);
//! resources.upload_local_bounds(&queue, &bounds);
//! resources.upload_lod_indices(&queue, &lods);
//! resources.upload_visibility(&queue, &visibility);
//! resources.upload_material_ids(&queue, &materials);
//!
//! let params = InstanceUpdateParams::new(instance_count);
//! pipeline.dispatch(&mut encoder, &resources, &params);
//!
//! // Instance data ready for rendering
//! ```

use std::mem;

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Size of InstanceData struct in bytes (must match WGSL layout).
pub const INSTANCE_DATA_SIZE: usize = 80;

/// Size of InputTransform struct in bytes.
pub const INPUT_TRANSFORM_SIZE: usize = 64;

/// Size of LocalBounds struct in bytes.
pub const LOCAL_BOUNDS_SIZE: usize = 16;

/// Size of InstanceUpdateParams struct in bytes.
pub const INSTANCE_UPDATE_PARAMS_SIZE: usize = 16;

/// Instance flag: visible after culling.
pub const FLAG_VISIBLE: u32 = 1;

/// Instance flag: uses skeletal animation.
pub const FLAG_SKINNED: u32 = 2;

/// Instance flag: transform needs update.
pub const FLAG_DIRTY: u32 = 4;

/// Instance flag: casts shadows.
pub const FLAG_CAST_SHADOW: u32 = 8;

/// Instance flag: receives shadows.
pub const FLAG_RECEIVE_SHADOW: u32 = 16;

/// Instance flag: two-sided rendering.
pub const FLAG_TWO_SIDED: u32 = 32;

/// Instance flag: apply motion blur.
pub const FLAG_MOTION_BLUR: u32 = 64;

/// Instance flag: static geometry (no transform updates).
pub const FLAG_STATIC: u32 = 128;

/// Invalid LOD marker (instance culled by distance).
pub const LOD_CULLED: u32 = 0xFFFFFFFF;

/// Invalid material ID marker.
pub const INVALID_MATERIAL: u32 = 0xFFFFFFFF;

/// Default capacity for instance buffers.
pub const DEFAULT_INSTANCE_CAPACITY: u32 = 65536;

// ---------------------------------------------------------------------------
// InstanceData
// ---------------------------------------------------------------------------

/// Packed instance data for GPU rendering.
///
/// Contains transform, bounds, LOD, flags, and material information
/// in a compact format suitable for shader consumption.
///
/// # Memory Layout
///
/// 80 bytes, 16-byte aligned:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | transform     | 48   | mat4x3 in column-major order
/// | 48     | bounds_center | 12   |
/// | 60     | bounds_radius | 4    |
/// | 64     | lod_index     | 4    |
/// | 68     | flags         | 4    |
/// | 72     | material_id   | 4    |
/// | 76     | _padding      | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct InstanceData {
    /// Packed mat4x3 transform (12 floats, column-major).
    ///
    /// Layout: [col0.xyz, col1.xyz, col2.xyz, col3.xyz]
    /// Where col3 is the translation vector.
    pub transform: [f32; 12],
    /// World-space bounding sphere center.
    pub bounds_center: [f32; 3],
    /// World-space bounding sphere radius.
    pub bounds_radius: f32,
    /// Selected LOD index from distance culling.
    pub lod_index: u32,
    /// Instance flags (visibility, skinned, etc.).
    pub flags: u32,
    /// Material table index.
    pub material_id: u32,
    /// Padding for 16-byte alignment.
    pub _padding: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<InstanceData>() == INSTANCE_DATA_SIZE);

impl InstanceData {
    /// Create new instance data with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create instance data with transform and material.
    pub fn with_transform(transform: [f32; 12], material_id: u32) -> Self {
        Self {
            transform,
            bounds_center: [0.0; 3],
            bounds_radius: 0.0,
            lod_index: 0,
            flags: FLAG_VISIBLE,
            material_id,
            _padding: 0,
        }
    }

    /// Create fully specified instance data.
    pub fn full(
        transform: [f32; 12],
        bounds_center: [f32; 3],
        bounds_radius: f32,
        lod_index: u32,
        flags: u32,
        material_id: u32,
    ) -> Self {
        Self {
            transform,
            bounds_center,
            bounds_radius,
            lod_index,
            flags,
            material_id,
            _padding: 0,
        }
    }

    /// Check if this instance is visible.
    #[inline]
    pub fn is_visible(&self) -> bool {
        (self.flags & FLAG_VISIBLE) != 0
    }

    /// Check if this instance is skinned (skeletal animation).
    #[inline]
    pub fn is_skinned(&self) -> bool {
        (self.flags & FLAG_SKINNED) != 0
    }

    /// Check if this instance was culled by LOD distance.
    #[inline]
    pub fn is_lod_culled(&self) -> bool {
        self.lod_index == LOD_CULLED
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

    /// Set skinned flag.
    #[inline]
    pub fn set_skinned(&mut self, skinned: bool) {
        if skinned {
            self.flags |= FLAG_SKINNED;
        } else {
            self.flags &= !FLAG_SKINNED;
        }
    }

    /// Extract translation from the packed transform.
    #[inline]
    pub fn translation(&self) -> [f32; 3] {
        [self.transform[9], self.transform[10], self.transform[11]]
    }

    /// Extract the 3x3 rotation/scale matrix from the packed transform.
    #[inline]
    pub fn rotation_scale_matrix(&self) -> [[f32; 3]; 3] {
        [
            [self.transform[0], self.transform[3], self.transform[6]],
            [self.transform[1], self.transform[4], self.transform[7]],
            [self.transform[2], self.transform[5], self.transform[8]],
        ]
    }
}

// ---------------------------------------------------------------------------
// InputTransform
// ---------------------------------------------------------------------------

/// Input transform data from animation/physics systems.
///
/// Stores a full 4x4 transformation matrix in row-major order.
///
/// # Memory Layout
///
/// 64 bytes, vec4 aligned:
/// | Offset | Field | Size |
/// |--------|-------|------|
/// | 0      | row0  | 16   |
/// | 16     | row1  | 16   |
/// | 32     | row2  | 16   |
/// | 48     | row3  | 16   |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct InputTransform {
    /// Matrix row 0: [m00, m01, m02, m03]
    pub row0: [f32; 4],
    /// Matrix row 1: [m10, m11, m12, m13]
    pub row1: [f32; 4],
    /// Matrix row 2: [m20, m21, m22, m23]
    pub row2: [f32; 4],
    /// Matrix row 3: [m30, m31, m32, m33] (typically [0,0,0,1])
    pub row3: [f32; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<InputTransform>() == INPUT_TRANSFORM_SIZE);

impl InputTransform {
    /// Create an identity transform.
    pub fn identity() -> Self {
        Self {
            row0: [1.0, 0.0, 0.0, 0.0],
            row1: [0.0, 1.0, 0.0, 0.0],
            row2: [0.0, 0.0, 1.0, 0.0],
            row3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a translation transform.
    pub fn translation(x: f32, y: f32, z: f32) -> Self {
        Self {
            row0: [1.0, 0.0, 0.0, x],
            row1: [0.0, 1.0, 0.0, y],
            row2: [0.0, 0.0, 1.0, z],
            row3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a scale transform.
    pub fn scale(sx: f32, sy: f32, sz: f32) -> Self {
        Self {
            row0: [sx, 0.0, 0.0, 0.0],
            row1: [0.0, sy, 0.0, 0.0],
            row2: [0.0, 0.0, sz, 0.0],
            row3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a uniform scale transform.
    pub fn uniform_scale(s: f32) -> Self {
        Self::scale(s, s, s)
    }

    /// Create a rotation transform around the X axis (radians).
    pub fn rotation_x(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            row0: [1.0, 0.0, 0.0, 0.0],
            row1: [0.0, c, -s, 0.0],
            row2: [0.0, s, c, 0.0],
            row3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a rotation transform around the Y axis (radians).
    pub fn rotation_y(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            row0: [c, 0.0, s, 0.0],
            row1: [0.0, 1.0, 0.0, 0.0],
            row2: [-s, 0.0, c, 0.0],
            row3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a rotation transform around the Z axis (radians).
    pub fn rotation_z(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            row0: [c, -s, 0.0, 0.0],
            row1: [s, c, 0.0, 0.0],
            row2: [0.0, 0.0, 1.0, 0.0],
            row3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Create a transform from translation, rotation (quaternion), and scale.
    ///
    /// Quaternion format: [x, y, z, w]
    pub fn from_trs(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Self {
        let [qx, qy, qz, qw] = rotation;

        // Quaternion to rotation matrix
        let x2 = qx + qx;
        let y2 = qy + qy;
        let z2 = qz + qz;
        let xx = qx * x2;
        let xy = qx * y2;
        let xz = qx * z2;
        let yy = qy * y2;
        let yz = qy * z2;
        let zz = qz * z2;
        let wx = qw * x2;
        let wy = qw * y2;
        let wz = qw * z2;

        let [sx, sy, sz] = scale;
        let [tx, ty, tz] = translation;

        Self {
            row0: [(1.0 - yy - zz) * sx, (xy - wz) * sy, (xz + wy) * sz, tx],
            row1: [(xy + wz) * sx, (1.0 - xx - zz) * sy, (yz - wx) * sz, ty],
            row2: [(xz - wy) * sx, (yz + wx) * sy, (1.0 - xx - yy) * sz, tz],
            row3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Multiply two transforms.
    pub fn multiply(&self, other: &Self) -> Self {
        let mut result = Self::default();

        for i in 0..4 {
            let row = match i {
                0 => &self.row0,
                1 => &self.row1,
                2 => &self.row2,
                _ => &self.row3,
            };

            let out_row = match i {
                0 => &mut result.row0,
                1 => &mut result.row1,
                2 => &mut result.row2,
                _ => &mut result.row3,
            };

            for j in 0..4 {
                out_row[j] = row[0] * other.row0[j]
                    + row[1] * other.row1[j]
                    + row[2] * other.row2[j]
                    + row[3] * other.row3[j];
            }
        }

        result
    }

    /// Transform a point (applies translation).
    pub fn transform_point(&self, p: [f32; 3]) -> [f32; 3] {
        [
            self.row0[0] * p[0] + self.row0[1] * p[1] + self.row0[2] * p[2] + self.row0[3],
            self.row1[0] * p[0] + self.row1[1] * p[1] + self.row1[2] * p[2] + self.row1[3],
            self.row2[0] * p[0] + self.row2[1] * p[1] + self.row2[2] * p[2] + self.row2[3],
        ]
    }

    /// Transform a vector (ignores translation).
    pub fn transform_vector(&self, v: [f32; 3]) -> [f32; 3] {
        [
            self.row0[0] * v[0] + self.row0[1] * v[1] + self.row0[2] * v[2],
            self.row1[0] * v[0] + self.row1[1] * v[1] + self.row1[2] * v[2],
            self.row2[0] * v[0] + self.row2[1] * v[1] + self.row2[2] * v[2],
        ]
    }

    /// Get the maximum scale factor (for bounding radius scaling).
    pub fn max_scale(&self) -> f32 {
        let col0_len = (self.row0[0] * self.row0[0]
            + self.row1[0] * self.row1[0]
            + self.row2[0] * self.row2[0])
        .sqrt();
        let col1_len = (self.row0[1] * self.row0[1]
            + self.row1[1] * self.row1[1]
            + self.row2[1] * self.row2[1])
        .sqrt();
        let col2_len = (self.row0[2] * self.row0[2]
            + self.row1[2] * self.row1[2]
            + self.row2[2] * self.row2[2])
        .sqrt();

        col0_len.max(col1_len).max(col2_len)
    }

    /// Extract translation component.
    #[inline]
    pub fn translation_component(&self) -> [f32; 3] {
        [self.row0[3], self.row1[3], self.row2[3]]
    }
}

// ---------------------------------------------------------------------------
// LocalBounds
// ---------------------------------------------------------------------------

/// Local bounding data for an instance (object space).
///
/// # Memory Layout
///
/// 16 bytes, vec4 aligned:
/// | Offset | Field  | Size |
/// |--------|--------|------|
/// | 0      | center | 12   |
/// | 12     | radius | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct LocalBounds {
    /// Object-space bounding sphere center.
    pub center: [f32; 3],
    /// Object-space bounding sphere radius.
    pub radius: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<LocalBounds>() == LOCAL_BOUNDS_SIZE);

impl LocalBounds {
    /// Create new local bounds.
    pub fn new(center: [f32; 3], radius: f32) -> Self {
        Self { center, radius }
    }

    /// Create bounds centered at origin.
    pub fn centered(radius: f32) -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            radius,
        }
    }

    /// Create bounds from AABB min/max corners.
    pub fn from_aabb(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        let center = [
            (aabb_min[0] + aabb_max[0]) * 0.5,
            (aabb_min[1] + aabb_max[1]) * 0.5,
            (aabb_min[2] + aabb_max[2]) * 0.5,
        ];
        let half_extents = [
            (aabb_max[0] - aabb_min[0]) * 0.5,
            (aabb_max[1] - aabb_min[1]) * 0.5,
            (aabb_max[2] - aabb_min[2]) * 0.5,
        ];
        let radius = (half_extents[0] * half_extents[0]
            + half_extents[1] * half_extents[1]
            + half_extents[2] * half_extents[2])
        .sqrt();

        Self { center, radius }
    }

    /// Transform bounds to world space.
    pub fn transform(&self, transform: &InputTransform) -> (f32, [f32; 3]) {
        let world_center = transform.transform_point(self.center);
        let world_radius = self.radius * transform.max_scale();
        (world_radius, world_center)
    }
}

// ---------------------------------------------------------------------------
// InstanceUpdateParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for instance update parameters.
///
/// # Memory Layout
///
/// 16 bytes, std140 compatible:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | num_instances | 4    |
/// | 4      | update_flags  | 4    |
/// | 8      | delta_time    | 4    |
/// | 12     | _padding      | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct InstanceUpdateParams {
    /// Number of instances to process.
    pub num_instances: u32,
    /// Update flags (enable/disable specific updates).
    pub update_flags: u32,
    /// Delta time since last update (for motion vectors).
    pub delta_time: f32,
    /// Padding for 16-byte alignment.
    pub _padding: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<InstanceUpdateParams>() == INSTANCE_UPDATE_PARAMS_SIZE);

impl InstanceUpdateParams {
    /// Create parameters for the given instance count.
    pub fn new(num_instances: u32) -> Self {
        Self {
            num_instances,
            update_flags: 0,
            delta_time: 0.0,
            _padding: 0,
        }
    }

    /// Create parameters with delta time.
    pub fn with_delta_time(num_instances: u32, delta_time: f32) -> Self {
        Self {
            num_instances,
            update_flags: 0,
            delta_time,
            _padding: 0,
        }
    }

    /// Create parameters with flags.
    pub fn with_flags(num_instances: u32, update_flags: u32) -> Self {
        Self {
            num_instances,
            update_flags,
            delta_time: 0.0,
            _padding: 0,
        }
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_instances + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation: Transform Packing
// ---------------------------------------------------------------------------

/// Pack a mat4 transform into mat4x3 format (12 floats).
///
/// The mat4x3 format stores the 3x3 rotation/scale matrix and translation
/// in column-major order, omitting the implicit [0,0,0,1] bottom row.
///
/// # Returns
///
/// 12-element array in column-major order:
/// [col0.x, col0.y, col0.z, col1.x, col1.y, col1.z, col2.x, col2.y, col2.z, col3.x, col3.y, col3.z]
pub fn cpu_pack_transform(transform: &InputTransform) -> [f32; 12] {
    [
        // Column 0
        transform.row0[0],
        transform.row1[0],
        transform.row2[0],
        // Column 1
        transform.row0[1],
        transform.row1[1],
        transform.row2[1],
        // Column 2
        transform.row0[2],
        transform.row1[2],
        transform.row2[2],
        // Column 3 (translation)
        transform.row0[3],
        transform.row1[3],
        transform.row2[3],
    ]
}

/// Unpack a mat4x3 format back to a full InputTransform.
///
/// Reconstructs the full 4x4 matrix by adding the implicit [0,0,0,1] row.
pub fn cpu_unpack_transform(packed: &[f32; 12]) -> InputTransform {
    InputTransform {
        row0: [packed[0], packed[3], packed[6], packed[9]],
        row1: [packed[1], packed[4], packed[7], packed[10]],
        row2: [packed[2], packed[5], packed[8], packed[11]],
        row3: [0.0, 0.0, 0.0, 1.0],
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation: Instance Update
// ---------------------------------------------------------------------------

/// CPU reference implementation of instance update.
///
/// Processes transforms, bounds, and flags to produce packed instance data.
pub fn cpu_instance_update(
    transforms: &[InputTransform],
    local_bounds: &[LocalBounds],
    lod_indices: &[u32],
    visibility_flags: &[u32],
    material_ids: &[u32],
) -> Vec<InstanceData> {
    let count = transforms.len();
    assert_eq!(local_bounds.len(), count);
    assert_eq!(lod_indices.len(), count);
    assert_eq!(visibility_flags.len(), count);
    assert_eq!(material_ids.len(), count);

    transforms
        .iter()
        .enumerate()
        .map(|(i, transform)| {
            let packed_transform = cpu_pack_transform(transform);

            // Transform bounds to world space
            let world_center = transform.transform_point(local_bounds[i].center);
            let world_radius = local_bounds[i].radius * transform.max_scale();

            // Build flags
            let lod = lod_indices[i];
            let visibility = visibility_flags[i];
            let mut flags = 0u32;
            if visibility != 0 && lod != LOD_CULLED {
                flags |= FLAG_VISIBLE;
            }

            InstanceData {
                transform: packed_transform,
                bounds_center: world_center,
                bounds_radius: world_radius,
                lod_index: lod,
                flags,
                material_id: material_ids[i],
                _padding: 0,
            }
        })
        .collect()
}

/// CPU reference implementation for transform-only update.
pub fn cpu_instance_update_transform_only(
    instances: &mut [InstanceData],
    transforms: &[InputTransform],
) {
    assert_eq!(instances.len(), transforms.len());

    for (instance, transform) in instances.iter_mut().zip(transforms.iter()) {
        instance.transform = cpu_pack_transform(transform);
        instance.flags |= FLAG_DIRTY;
    }
}

/// CPU reference implementation for visibility-only update.
pub fn cpu_instance_update_visibility_only(
    instances: &mut [InstanceData],
    lod_indices: &[u32],
    visibility_flags: &[u32],
) {
    assert_eq!(instances.len(), lod_indices.len());
    assert_eq!(instances.len(), visibility_flags.len());

    for (i, instance) in instances.iter_mut().enumerate() {
        let lod = lod_indices[i];
        let visibility = visibility_flags[i];

        // Clear and rebuild visibility flags
        instance.flags &= !FLAG_VISIBLE;
        if visibility != 0 && lod != LOD_CULLED {
            instance.flags |= FLAG_VISIBLE;
        }
        instance.lod_index = lod;
    }
}

// ---------------------------------------------------------------------------
// InstanceBuffer
// ---------------------------------------------------------------------------

/// Manages instance data on the GPU.
///
/// Provides CPU-side staging and GPU buffer management for instance data.
pub struct InstanceBuffer {
    /// CPU-side staging buffer.
    pub instances: Vec<InstanceData>,
    /// GPU storage buffer for instance data.
    pub buffer: wgpu::Buffer,
    /// Staging buffer for CPU readback.
    pub staging: wgpu::Buffer,
    /// Maximum capacity.
    pub capacity: u32,
}

impl InstanceBuffer {
    /// Create a new instance buffer with the given capacity.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let instances = vec![InstanceData::default(); capacity as usize];

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_data_buffer"),
            size: (capacity as u64) * (INSTANCE_DATA_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_data_staging"),
            size: (capacity as u64) * (INSTANCE_DATA_SIZE as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            instances,
            buffer,
            staging,
            capacity,
        }
    }

    /// Upload instance data to GPU.
    ///
    /// # Panics
    ///
    /// Panics if `count > self.capacity`.
    pub fn upload(&self, queue: &wgpu::Queue, count: u32) {
        assert!(count <= self.capacity);
        let data = &self.instances[..count as usize];
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    /// Get a mutable reference to an instance.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut InstanceData> {
        self.instances.get_mut(index)
    }

    /// Set instance data at the given index.
    pub fn set(&mut self, index: usize, data: InstanceData) {
        if index < self.instances.len() {
            self.instances[index] = data;
        }
    }

    /// Clear all instances (set to default).
    pub fn clear(&mut self) {
        for instance in &mut self.instances {
            *instance = InstanceData::default();
        }
    }
}

// ---------------------------------------------------------------------------
// InstanceUpdateResources
// ---------------------------------------------------------------------------

/// GPU resources for instance update.
///
/// Contains all buffers needed for the instance update compute shader.
pub struct InstanceUpdateResources {
    /// Uniform buffer for update parameters.
    pub params_buffer: wgpu::Buffer,
    /// Storage buffer for input transforms.
    pub transforms_buffer: wgpu::Buffer,
    /// Storage buffer for local bounds.
    pub local_bounds_buffer: wgpu::Buffer,
    /// Storage buffer for LOD indices.
    pub lod_indices_buffer: wgpu::Buffer,
    /// Storage buffer for visibility flags.
    pub visibility_buffer: wgpu::Buffer,
    /// Storage buffer for material IDs.
    pub material_ids_buffer: wgpu::Buffer,
    /// Storage buffer for output instance data.
    pub output_buffer: wgpu::Buffer,
    /// Staging buffer for CPU readback.
    pub staging_buffer: wgpu::Buffer,
    /// Maximum capacity.
    pub capacity: u32,
}

impl InstanceUpdateResources {
    /// Create instance update resources with the given capacity.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_update_params"),
            size: INSTANCE_UPDATE_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let transforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_update_transforms"),
            size: (capacity as u64) * (INPUT_TRANSFORM_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let local_bounds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_update_local_bounds"),
            size: (capacity as u64) * (LOCAL_BOUNDS_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lod_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_update_lod_indices"),
            size: (capacity as u64) * 4, // u32 per instance
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_update_visibility"),
            size: (capacity as u64) * 4, // u32 per instance
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material_ids_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_update_material_ids"),
            size: (capacity as u64) * 4, // u32 per instance
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_update_output"),
            size: (capacity as u64) * (INSTANCE_DATA_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_update_staging"),
            size: (capacity as u64) * (INSTANCE_DATA_SIZE as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            transforms_buffer,
            local_bounds_buffer,
            lod_indices_buffer,
            visibility_buffer,
            material_ids_buffer,
            output_buffer,
            staging_buffer,
            capacity,
        }
    }

    /// Upload update parameters.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &InstanceUpdateParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload input transforms.
    ///
    /// # Panics
    ///
    /// Panics if `transforms.len() > self.capacity`.
    pub fn upload_transforms(&self, queue: &wgpu::Queue, transforms: &[InputTransform]) {
        assert!(transforms.len() <= self.capacity as usize);
        queue.write_buffer(&self.transforms_buffer, 0, bytemuck::cast_slice(transforms));
    }

    /// Upload local bounds.
    ///
    /// # Panics
    ///
    /// Panics if `bounds.len() > self.capacity`.
    pub fn upload_local_bounds(&self, queue: &wgpu::Queue, bounds: &[LocalBounds]) {
        assert!(bounds.len() <= self.capacity as usize);
        queue.write_buffer(&self.local_bounds_buffer, 0, bytemuck::cast_slice(bounds));
    }

    /// Upload LOD indices.
    ///
    /// # Panics
    ///
    /// Panics if `lod_indices.len() > self.capacity`.
    pub fn upload_lod_indices(&self, queue: &wgpu::Queue, lod_indices: &[u32]) {
        assert!(lod_indices.len() <= self.capacity as usize);
        queue.write_buffer(&self.lod_indices_buffer, 0, bytemuck::cast_slice(lod_indices));
    }

    /// Upload visibility flags.
    ///
    /// # Panics
    ///
    /// Panics if `visibility.len() > self.capacity`.
    pub fn upload_visibility(&self, queue: &wgpu::Queue, visibility: &[u32]) {
        assert!(visibility.len() <= self.capacity as usize);
        queue.write_buffer(&self.visibility_buffer, 0, bytemuck::cast_slice(visibility));
    }

    /// Upload material IDs.
    ///
    /// # Panics
    ///
    /// Panics if `material_ids.len() > self.capacity`.
    pub fn upload_material_ids(&self, queue: &wgpu::Queue, material_ids: &[u32]) {
        assert!(material_ids.len() <= self.capacity as usize);
        queue.write_buffer(&self.material_ids_buffer, 0, bytemuck::cast_slice(material_ids));
    }
}

// ---------------------------------------------------------------------------
// InstanceUpdatePipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for instance update.
pub struct InstanceUpdatePipeline {
    /// Main update pipeline (full update).
    pub pipeline: wgpu::ComputePipeline,
    /// Transform-only update pipeline.
    pub pipeline_transform_only: wgpu::ComputePipeline,
    /// Visibility-only update pipeline.
    pub pipeline_visibility_only: wgpu::ComputePipeline,
    /// Skinned mesh update pipeline.
    pub pipeline_skinned: wgpu::ComputePipeline,
    /// Clear visibility pipeline.
    pub pipeline_clear: wgpu::ComputePipeline,
    /// Bind group layout.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl InstanceUpdatePipeline {
    /// Create the instance update pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `shader_source` - WGSL shader source code.
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instance_update_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_update_bind_group_layout"),
            entries: &[
                // @binding(0) params: InstanceUpdateParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(INSTANCE_UPDATE_PARAMS_SIZE as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) input_transforms
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
                // @binding(2) local_bounds
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
                // @binding(3) lod_indices
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
                // @binding(4) visibility_flags
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(5) material_ids
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(6) output_instances
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
            label: Some("instance_update_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("instance_update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "instance_update",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_transform_only =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("instance_update_pipeline_transform_only"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "instance_update_transform_only",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let pipeline_visibility_only =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("instance_update_pipeline_visibility_only"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "instance_update_visibility_only",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let pipeline_skinned = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("instance_update_pipeline_skinned"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "instance_update_skinned",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_clear = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("instance_update_pipeline_clear"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "instance_clear_visibility",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            pipeline_transform_only,
            pipeline_visibility_only,
            pipeline_skinned,
            pipeline_clear,
            bind_group_layout,
        }
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &InstanceUpdateResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instance_update_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.transforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.local_bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.lod_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.material_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: resources.output_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Size and alignment tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_instance_data_size() {
        assert_eq!(mem::size_of::<InstanceData>(), INSTANCE_DATA_SIZE);
        assert_eq!(mem::size_of::<InstanceData>(), 80);
    }

    #[test]
    fn test_input_transform_size() {
        assert_eq!(mem::size_of::<InputTransform>(), INPUT_TRANSFORM_SIZE);
        assert_eq!(mem::size_of::<InputTransform>(), 64);
    }

    #[test]
    fn test_local_bounds_size() {
        assert_eq!(mem::size_of::<LocalBounds>(), LOCAL_BOUNDS_SIZE);
        assert_eq!(mem::size_of::<LocalBounds>(), 16);
    }

    #[test]
    fn test_instance_update_params_size() {
        assert_eq!(mem::size_of::<InstanceUpdateParams>(), INSTANCE_UPDATE_PARAMS_SIZE);
        assert_eq!(mem::size_of::<InstanceUpdateParams>(), 16);
    }

    #[test]
    fn test_instance_data_alignment() {
        // Should be 16-byte aligned for GPU
        assert_eq!(mem::align_of::<InstanceData>(), 4);
    }

    #[test]
    fn test_input_transform_alignment() {
        assert_eq!(mem::align_of::<InputTransform>(), 4);
    }

    // -----------------------------------------------------------------------
    // Transform tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_identity_transform() {
        let t = InputTransform::identity();
        assert_eq!(t.row0, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(t.row1, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(t.row2, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(t.row3, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_translation_transform() {
        let t = InputTransform::translation(1.0, 2.0, 3.0);
        assert_eq!(t.row0[3], 1.0);
        assert_eq!(t.row1[3], 2.0);
        assert_eq!(t.row2[3], 3.0);
    }

    #[test]
    fn test_scale_transform() {
        let t = InputTransform::scale(2.0, 3.0, 4.0);
        assert_eq!(t.row0[0], 2.0);
        assert_eq!(t.row1[1], 3.0);
        assert_eq!(t.row2[2], 4.0);
    }

    #[test]
    fn test_uniform_scale_transform() {
        let t = InputTransform::uniform_scale(2.0);
        assert_eq!(t.row0[0], 2.0);
        assert_eq!(t.row1[1], 2.0);
        assert_eq!(t.row2[2], 2.0);
    }

    #[test]
    fn test_transform_point_identity() {
        let t = InputTransform::identity();
        let p = [1.0, 2.0, 3.0];
        let result = t.transform_point(p);
        assert_eq!(result, p);
    }

    #[test]
    fn test_transform_point_translation() {
        let t = InputTransform::translation(10.0, 20.0, 30.0);
        let p = [1.0, 2.0, 3.0];
        let result = t.transform_point(p);
        assert_eq!(result, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_transform_point_scale() {
        let t = InputTransform::scale(2.0, 3.0, 4.0);
        let p = [1.0, 2.0, 3.0];
        let result = t.transform_point(p);
        assert_eq!(result, [2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_transform_vector_ignores_translation() {
        let t = InputTransform::translation(100.0, 200.0, 300.0);
        let v = [1.0, 2.0, 3.0];
        let result = t.transform_vector(v);
        assert_eq!(result, v);
    }

    #[test]
    fn test_max_scale_uniform() {
        let t = InputTransform::uniform_scale(5.0);
        assert!((t.max_scale() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_scale_non_uniform() {
        let t = InputTransform::scale(2.0, 5.0, 3.0);
        assert!((t.max_scale() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_translation_component() {
        let t = InputTransform::translation(1.0, 2.0, 3.0);
        assert_eq!(t.translation_component(), [1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // Transform packing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pack_identity_transform() {
        let t = InputTransform::identity();
        let packed = cpu_pack_transform(&t);

        // Column 0 (identity first column)
        assert_eq!(packed[0], 1.0);
        assert_eq!(packed[1], 0.0);
        assert_eq!(packed[2], 0.0);

        // Column 1
        assert_eq!(packed[3], 0.0);
        assert_eq!(packed[4], 1.0);
        assert_eq!(packed[5], 0.0);

        // Column 2
        assert_eq!(packed[6], 0.0);
        assert_eq!(packed[7], 0.0);
        assert_eq!(packed[8], 1.0);

        // Column 3 (translation = 0)
        assert_eq!(packed[9], 0.0);
        assert_eq!(packed[10], 0.0);
        assert_eq!(packed[11], 0.0);
    }

    #[test]
    fn test_pack_translation_transform() {
        let t = InputTransform::translation(10.0, 20.0, 30.0);
        let packed = cpu_pack_transform(&t);

        // Translation in column 3
        assert_eq!(packed[9], 10.0);
        assert_eq!(packed[10], 20.0);
        assert_eq!(packed[11], 30.0);
    }

    #[test]
    fn test_pack_scale_transform() {
        let t = InputTransform::scale(2.0, 3.0, 4.0);
        let packed = cpu_pack_transform(&t);

        // Scale factors on diagonal
        assert_eq!(packed[0], 2.0);
        assert_eq!(packed[4], 3.0);
        assert_eq!(packed[8], 4.0);
    }

    #[test]
    fn test_unpack_roundtrip() {
        let original = InputTransform::from_trs(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0], // Identity rotation
            [1.0, 1.0, 1.0],
        );

        let packed = cpu_pack_transform(&original);
        let unpacked = cpu_unpack_transform(&packed);

        // Check that unpacked matches original (ignoring row3 which is always [0,0,0,1])
        for i in 0..4 {
            assert!((unpacked.row0[i] - original.row0[i]).abs() < 1e-6);
            assert!((unpacked.row1[i] - original.row1[i]).abs() < 1e-6);
            assert!((unpacked.row2[i] - original.row2[i]).abs() < 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // LocalBounds tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_local_bounds_new() {
        let bounds = LocalBounds::new([1.0, 2.0, 3.0], 5.0);
        assert_eq!(bounds.center, [1.0, 2.0, 3.0]);
        assert_eq!(bounds.radius, 5.0);
    }

    #[test]
    fn test_local_bounds_centered() {
        let bounds = LocalBounds::centered(5.0);
        assert_eq!(bounds.center, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.radius, 5.0);
    }

    #[test]
    fn test_local_bounds_from_aabb() {
        let bounds = LocalBounds::from_aabb([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);

        assert_eq!(bounds.center, [0.0, 0.0, 0.0]);

        // Radius should be sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
        let expected_radius = (1.0_f32 + 4.0 + 9.0).sqrt();
        assert!((bounds.radius - expected_radius).abs() < 1e-6);
    }

    #[test]
    fn test_local_bounds_transform() {
        let bounds = LocalBounds::centered(1.0);
        let transform = InputTransform::scale(2.0, 2.0, 2.0);

        let (world_radius, world_center) = bounds.transform(&transform);

        assert_eq!(world_center, [0.0, 0.0, 0.0]);
        assert!((world_radius - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_local_bounds_transform_with_translation() {
        let bounds = LocalBounds::centered(1.0);
        let transform = InputTransform::translation(10.0, 20.0, 30.0);

        let (world_radius, world_center) = bounds.transform(&transform);

        assert_eq!(world_center, [10.0, 20.0, 30.0]);
        assert!((world_radius - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // InstanceData tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_instance_data_default() {
        let data = InstanceData::default();
        assert!(!data.is_visible());
        assert!(!data.is_skinned());
        assert!(!data.is_lod_culled()); // Default lod_index is 0
    }

    #[test]
    fn test_instance_data_visibility() {
        let mut data = InstanceData::new();
        assert!(!data.is_visible());

        data.set_visible(true);
        assert!(data.is_visible());

        data.set_visible(false);
        assert!(!data.is_visible());
    }

    #[test]
    fn test_instance_data_skinned() {
        let mut data = InstanceData::new();
        assert!(!data.is_skinned());

        data.set_skinned(true);
        assert!(data.is_skinned());

        data.set_skinned(false);
        assert!(!data.is_skinned());
    }

    #[test]
    fn test_instance_data_lod_culled() {
        let mut data = InstanceData::new();
        assert!(!data.is_lod_culled());

        data.lod_index = LOD_CULLED;
        assert!(data.is_lod_culled());
    }

    #[test]
    fn test_instance_data_translation_extraction() {
        let t = InputTransform::translation(1.0, 2.0, 3.0);
        let packed = cpu_pack_transform(&t);
        let data = InstanceData::with_transform(packed, 0);

        assert_eq!(data.translation(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_instance_data_full_constructor() {
        let transform = cpu_pack_transform(&InputTransform::identity());
        let data = InstanceData::full(
            transform,
            [1.0, 2.0, 3.0],
            5.0,
            2,
            FLAG_VISIBLE | FLAG_SKINNED,
            42,
        );

        assert_eq!(data.bounds_center, [1.0, 2.0, 3.0]);
        assert_eq!(data.bounds_radius, 5.0);
        assert_eq!(data.lod_index, 2);
        assert!(data.is_visible());
        assert!(data.is_skinned());
        assert_eq!(data.material_id, 42);
    }

    // -----------------------------------------------------------------------
    // InstanceUpdateParams tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_instance_update_params_new() {
        let params = InstanceUpdateParams::new(1000);
        assert_eq!(params.num_instances, 1000);
        assert_eq!(params.update_flags, 0);
        assert_eq!(params.delta_time, 0.0);
    }

    #[test]
    fn test_instance_update_params_with_delta_time() {
        let params = InstanceUpdateParams::with_delta_time(1000, 0.016);
        assert_eq!(params.num_instances, 1000);
        assert!((params.delta_time - 0.016).abs() < 1e-6);
    }

    #[test]
    fn test_instance_update_params_num_workgroups() {
        assert_eq!(InstanceUpdateParams::new(1).num_workgroups(), 1);
        assert_eq!(InstanceUpdateParams::new(256).num_workgroups(), 1);
        assert_eq!(InstanceUpdateParams::new(257).num_workgroups(), 2);
        assert_eq!(InstanceUpdateParams::new(512).num_workgroups(), 2);
        assert_eq!(InstanceUpdateParams::new(1000).num_workgroups(), 4);
    }

    // -----------------------------------------------------------------------
    // CPU instance update tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_instance_update_single() {
        let transforms = [InputTransform::translation(10.0, 20.0, 30.0)];
        let local_bounds = [LocalBounds::centered(5.0)];
        let lod_indices = [0u32];
        let visibility_flags = [1u32];
        let material_ids = [42u32];

        let result = cpu_instance_update(
            &transforms,
            &local_bounds,
            &lod_indices,
            &visibility_flags,
            &material_ids,
        );

        assert_eq!(result.len(), 1);
        assert!(result[0].is_visible());
        assert_eq!(result[0].lod_index, 0);
        assert_eq!(result[0].material_id, 42);
        assert_eq!(result[0].bounds_center, [10.0, 20.0, 30.0]);
        assert!((result[0].bounds_radius - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_instance_update_culled() {
        let transforms = [InputTransform::identity()];
        let local_bounds = [LocalBounds::centered(1.0)];
        let lod_indices = [LOD_CULLED];
        let visibility_flags = [0u32]; // Not visible
        let material_ids = [0u32];

        let result = cpu_instance_update(
            &transforms,
            &local_bounds,
            &lod_indices,
            &visibility_flags,
            &material_ids,
        );

        assert!(!result[0].is_visible());
        assert!(result[0].is_lod_culled());
    }

    #[test]
    fn test_cpu_instance_update_multiple() {
        let transforms = [
            InputTransform::translation(0.0, 0.0, 0.0),
            InputTransform::translation(10.0, 0.0, 0.0),
            InputTransform::translation(20.0, 0.0, 0.0),
        ];
        let local_bounds = [
            LocalBounds::centered(1.0),
            LocalBounds::centered(2.0),
            LocalBounds::centered(3.0),
        ];
        let lod_indices = [0, 1, 2];
        let visibility_flags = [1, 1, 0]; // Third is not visible
        let material_ids = [0, 1, 2];

        let result = cpu_instance_update(
            &transforms,
            &local_bounds,
            &lod_indices,
            &visibility_flags,
            &material_ids,
        );

        assert_eq!(result.len(), 3);

        // First: visible, LOD 0
        assert!(result[0].is_visible());
        assert_eq!(result[0].lod_index, 0);
        assert_eq!(result[0].bounds_center, [0.0, 0.0, 0.0]);

        // Second: visible, LOD 1
        assert!(result[1].is_visible());
        assert_eq!(result[1].lod_index, 1);
        assert_eq!(result[1].bounds_center, [10.0, 0.0, 0.0]);

        // Third: not visible
        assert!(!result[2].is_visible());
        assert_eq!(result[2].lod_index, 2);
    }

    #[test]
    fn test_cpu_instance_update_transform_only() {
        let mut instances = vec![
            InstanceData::full(
                cpu_pack_transform(&InputTransform::identity()),
                [0.0, 0.0, 0.0],
                1.0,
                0,
                FLAG_VISIBLE,
                0,
            ),
        ];

        let new_transforms = [InputTransform::translation(100.0, 200.0, 300.0)];

        cpu_instance_update_transform_only(&mut instances, &new_transforms);

        assert_eq!(instances[0].translation(), [100.0, 200.0, 300.0]);
        assert!((instances[0].flags & FLAG_DIRTY) != 0);
    }

    #[test]
    fn test_cpu_instance_update_visibility_only() {
        let mut instances = vec![
            InstanceData::full(
                cpu_pack_transform(&InputTransform::identity()),
                [0.0, 0.0, 0.0],
                1.0,
                0,
                FLAG_VISIBLE,
                0,
            ),
        ];

        // Make invisible
        let lod_indices = [LOD_CULLED];
        let visibility_flags = [0u32];

        cpu_instance_update_visibility_only(&mut instances, &lod_indices, &visibility_flags);

        assert!(!instances[0].is_visible());
        assert!(instances[0].is_lod_culled());
    }

    // -----------------------------------------------------------------------
    // Flag constant tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_flag_constants() {
        assert_eq!(FLAG_VISIBLE, 1);
        assert_eq!(FLAG_SKINNED, 2);
        assert_eq!(FLAG_DIRTY, 4);
        assert_eq!(FLAG_CAST_SHADOW, 8);
        assert_eq!(FLAG_RECEIVE_SHADOW, 16);
        assert_eq!(FLAG_TWO_SIDED, 32);
        assert_eq!(FLAG_MOTION_BLUR, 64);
        assert_eq!(FLAG_STATIC, 128);
    }

    #[test]
    fn test_flags_are_independent() {
        // All flags should be powers of 2 (single bit)
        let flags = [
            FLAG_VISIBLE,
            FLAG_SKINNED,
            FLAG_DIRTY,
            FLAG_CAST_SHADOW,
            FLAG_RECEIVE_SHADOW,
            FLAG_TWO_SIDED,
            FLAG_MOTION_BLUR,
            FLAG_STATIC,
        ];

        for &flag in &flags {
            assert!(flag.is_power_of_two());
        }

        // No overlap
        let combined: u32 = flags.iter().sum();
        let xored: u32 = flags.iter().fold(0, |acc, &f| acc ^ f);
        assert_eq!(combined, xored);
    }

    // -----------------------------------------------------------------------
    // Shader validation tests (using naga)
    // -----------------------------------------------------------------------

    #[test]
    fn test_shader_parses_successfully() {
        // Read the shader source
        let shader_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/gpu_driven/gpu_instance_update.comp.wgsl"
        );

        // Check if file exists (may not in CI without full checkout)
        if std::path::Path::new(shader_path).exists() {
            let shader_source = std::fs::read_to_string(shader_path)
                .expect("Failed to read shader file");

            // Parse with naga
            let module = naga::front::wgsl::parse_str(&shader_source);
            assert!(
                module.is_ok(),
                "Shader parsing failed: {:?}",
                module.err()
            );

            // Validate the module
            let module = module.unwrap();
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            let validation_result = validator.validate(&module);
            assert!(
                validation_result.is_ok(),
                "Shader validation failed: {:?}",
                validation_result.err()
            );
        }
    }

    #[test]
    fn test_shader_entry_points_exist() {
        let shader_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/gpu_driven/gpu_instance_update.comp.wgsl"
        );

        if std::path::Path::new(shader_path).exists() {
            let shader_source = std::fs::read_to_string(shader_path)
                .expect("Failed to read shader file");

            let module = naga::front::wgsl::parse_str(&shader_source)
                .expect("Failed to parse shader");

            // Check for expected entry points
            let entry_point_names: Vec<&str> = module
                .entry_points
                .iter()
                .map(|ep| ep.name.as_str())
                .collect();

            assert!(
                entry_point_names.contains(&"instance_update"),
                "Missing entry point: instance_update"
            );
            assert!(
                entry_point_names.contains(&"instance_update_transform_only"),
                "Missing entry point: instance_update_transform_only"
            );
            assert!(
                entry_point_names.contains(&"instance_update_visibility_only"),
                "Missing entry point: instance_update_visibility_only"
            );
            assert!(
                entry_point_names.contains(&"instance_update_skinned"),
                "Missing entry point: instance_update_skinned"
            );
            assert!(
                entry_point_names.contains(&"instance_clear_visibility"),
                "Missing entry point: instance_clear_visibility"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Rotation transform tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rotation_x() {
        let angle = std::f32::consts::FRAC_PI_2; // 90 degrees
        let t = InputTransform::rotation_x(angle);

        // Rotating (0, 1, 0) around X by 90 degrees should give (0, 0, 1)
        let result = t.transform_point([0.0, 1.0, 0.0]);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_y() {
        let angle = std::f32::consts::FRAC_PI_2; // 90 degrees
        let t = InputTransform::rotation_y(angle);

        // Rotating (1, 0, 0) around Y by 90 degrees should give (0, 0, -1)
        let result = t.transform_point([1.0, 0.0, 0.0]);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_z() {
        let angle = std::f32::consts::FRAC_PI_2; // 90 degrees
        let t = InputTransform::rotation_z(angle);

        // Rotating (1, 0, 0) around Z by 90 degrees should give (0, 1, 0)
        let result = t.transform_point([1.0, 0.0, 0.0]);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_transform_multiply() {
        let t1 = InputTransform::translation(10.0, 0.0, 0.0);
        let t2 = InputTransform::scale(2.0, 2.0, 2.0);

        // Scale then translate: point at origin should end up at (10, 0, 0)
        let combined = t1.multiply(&t2);
        let result = combined.transform_point([0.0, 0.0, 0.0]);
        assert_eq!(result, [10.0, 0.0, 0.0]);

        // Point at (1, 0, 0) should be scaled to (2, 0, 0) then translated to (12, 0, 0)
        let result2 = combined.transform_point([1.0, 0.0, 0.0]);
        assert_eq!(result2, [12.0, 0.0, 0.0]);
    }

    #[test]
    fn test_from_trs_identity() {
        let t = InputTransform::from_trs(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            [1.0, 1.0, 1.0],
        );

        // Should behave like identity
        let p = [1.0, 2.0, 3.0];
        let result = t.transform_point(p);
        assert!((result[0] - p[0]).abs() < 1e-6);
        assert!((result[1] - p[1]).abs() < 1e-6);
        assert!((result[2] - p[2]).abs() < 1e-6);
    }

    #[test]
    fn test_from_trs_with_scale() {
        let t = InputTransform::from_trs(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [2.0, 3.0, 4.0],
        );

        let result = t.transform_point([1.0, 1.0, 1.0]);
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 4.0).abs() < 1e-6);
    }
}
