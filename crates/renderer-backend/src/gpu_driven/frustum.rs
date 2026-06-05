//! Frustum Plane Extraction for GPU-Driven Culling (T-WGPU-P6.3.1).
//!
//! This module provides frustum plane extraction from a view-projection matrix
//! for use in GPU-driven frustum culling. It implements the Gribb-Hartmann method
//! for efficient plane extraction and provides GPU buffer management.
//!
//! # Overview
//!
//! Frustum culling requires testing objects against 6 planes that define the
//! view frustum. This module extracts those planes from the combined view-projection
//! matrix and uploads them to the GPU for compute shader access.
//!
//! # Plane Ordering
//!
//! | Index | Plane  | Description                    |
//! |-------|--------|--------------------------------|
//! | 0     | Left   | Left side of view frustum     |
//! | 1     | Right  | Right side of view frustum    |
//! | 2     | Bottom | Bottom of view frustum        |
//! | 3     | Top    | Top of view frustum           |
//! | 4     | Near   | Near clipping plane           |
//! | 5     | Far    | Far clipping plane            |
//!
//! # Memory Layout
//!
//! The `FrustumPlanes` struct is 96 bytes (6 planes x 16 bytes each):
//!
//! ```text
//! struct FrustumPlanes {
//!     planes: array<FrustumPlane, 6>,  // 96 bytes total
//! }
//!
//! struct FrustumPlane {
//!     normal: vec3<f32>,   // 12 bytes
//!     distance: f32,       // 4 bytes
//! }
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::frustum::{FrustumPlanes, FrustumBuffer};
//!
//! // Extract planes from view-projection matrix
//! let vp = view_matrix * projection_matrix;
//! let planes = FrustumPlanes::from_view_projection(&vp);
//!
//! // Create GPU buffer
//! let mut buffer = FrustumBuffer::new(&device);
//!
//! // Update each frame
//! buffer.update(&queue, &new_vp_matrix);
//!
//! // Bind for compute shader
//! let binding = wgpu::BindGroupEntry {
//!     binding: 0,
//!     resource: buffer.buffer().as_entire_binding(),
//! };
//! ```
//!
//! # WGSL Reference
//!
//! ```wgsl
//! struct FrustumPlane {
//!     normal: vec3<f32>,
//!     distance: f32,
//! }
//!
//! struct FrustumPlanes {
//!     planes: array<FrustumPlane, 6>,
//! }
//!
//! @group(0) @binding(0) var<uniform> frustum: FrustumPlanes;
//!
//! fn is_sphere_visible(center: vec3<f32>, radius: f32) -> bool {
//!     for (var i = 0u; i < 6u; i++) {
//!         let plane = frustum.planes[i];
//!         let dist = dot(plane.normal, center) + plane.distance;
//!         if (dist < -radius) {
//!             return false;
//!         }
//!     }
//!     return true;
//! }
//! ```

use bytemuck::{Pod, Zeroable};
use std::mem;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Number of frustum planes (left, right, bottom, top, near, far).
pub const NUM_FRUSTUM_PLANES: usize = 6;

/// Size of FrustumPlane in bytes (must match GPU struct).
pub const FRUSTUM_PLANE_SIZE: usize = 16;

/// Size of FrustumPlanes in bytes (6 planes x 16 bytes).
pub const FRUSTUM_PLANES_SIZE: usize = 96;

/// Plane index constants for array access.
pub const PLANE_LEFT: usize = 0;
pub const PLANE_RIGHT: usize = 1;
pub const PLANE_BOTTOM: usize = 2;
pub const PLANE_TOP: usize = 3;
pub const PLANE_NEAR: usize = 4;
pub const PLANE_FAR: usize = 5;

/// Epsilon for plane normalization to avoid division by zero.
const NORMALIZE_EPSILON: f32 = 1e-8;

// =============================================================================
// FRUSTUM PLANE
// =============================================================================

/// A frustum plane in Hessian normal form (ax + by + cz + d = 0).
///
/// The plane equation is: `dot(normal, point) + distance = 0`
///
/// Points where `dot(normal, point) + distance > 0` are on the positive
/// (inside) side of the plane. All plane normals point inward toward the
/// interior of the frustum.
///
/// # Memory Layout (16 bytes)
///
/// ```text
/// | Offset | Field    | Type | Size |
/// |--------|----------|------|------|
/// | 0      | normal.x | f32  | 4    |
/// | 4      | normal.y | f32  | 4    |
/// | 8      | normal.z | f32  | 4    |
/// | 12     | distance | f32  | 4    |
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct FrustumPlane {
    /// Normalized plane normal (pointing into frustum).
    pub normal: [f32; 3],
    /// Signed distance from origin (d in ax+by+cz+d=0).
    pub distance: f32,
}

// Compile-time size assertions
const _: () = assert!(mem::size_of::<FrustumPlane>() == FRUSTUM_PLANE_SIZE);
const _: () = assert!(mem::size_of::<FrustumPlane>() == 16);

impl FrustumPlane {
    /// Create a new frustum plane from normal and distance.
    ///
    /// The normal is automatically normalized. If the normal has near-zero
    /// length, defaults to (0, 0, 1) with distance 0.
    ///
    /// # Arguments
    ///
    /// * `normal` - Plane normal direction (will be normalized)
    /// * `distance` - Signed distance from origin
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create a plane at z = -5, facing +Z
    /// let plane = FrustumPlane::new([0.0, 0.0, 1.0], 5.0);
    /// ```
    #[inline]
    pub fn new(normal: [f32; 3], distance: f32) -> Self {
        let mut plane = Self { normal, distance };
        plane.normalize();
        plane
    }

    /// Create a plane from a point and normal.
    ///
    /// The distance is computed as -dot(normal, point) after normalization.
    ///
    /// # Arguments
    ///
    /// * `point` - A point on the plane
    /// * `normal` - Plane normal direction (will be normalized)
    #[inline]
    pub fn from_point_normal(point: [f32; 3], normal: [f32; 3]) -> Self {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len < NORMALIZE_EPSILON {
            return Self {
                normal: [0.0, 0.0, 1.0],
                distance: 0.0,
            };
        }
        let inv_len = 1.0 / len;
        let n = [
            normal[0] * inv_len,
            normal[1] * inv_len,
            normal[2] * inv_len,
        ];
        let d = -(n[0] * point[0] + n[1] * point[1] + n[2] * point[2]);
        Self { normal: n, distance: d }
    }

    /// Normalize the plane equation (ensure unit normal).
    ///
    /// Both normal and distance are scaled by 1/||normal||.
    #[inline]
    pub fn normalize(&mut self) {
        let len = (self.normal[0] * self.normal[0]
            + self.normal[1] * self.normal[1]
            + self.normal[2] * self.normal[2])
        .sqrt();

        if len < NORMALIZE_EPSILON {
            self.normal = [0.0, 0.0, 1.0];
            self.distance = 0.0;
            return;
        }

        let inv_len = 1.0 / len;
        self.normal[0] *= inv_len;
        self.normal[1] *= inv_len;
        self.normal[2] *= inv_len;
        self.distance *= inv_len;
    }

    /// Check if the plane normal is normalized (unit length).
    #[inline]
    pub fn is_normalized(&self) -> bool {
        let len_sq = self.normal[0] * self.normal[0]
            + self.normal[1] * self.normal[1]
            + self.normal[2] * self.normal[2];
        (len_sq - 1.0).abs() < 1e-5
    }

    /// Compute signed distance from a point to the plane.
    ///
    /// Returns positive if point is on the inside (normal side),
    /// negative if outside, zero if on the plane.
    #[inline]
    pub fn distance_to_point(&self, point: [f32; 3]) -> f32 {
        self.normal[0] * point[0]
            + self.normal[1] * point[1]
            + self.normal[2] * point[2]
            + self.distance
    }

    /// Test if a sphere intersects or is inside the plane's positive half-space.
    ///
    /// Returns `true` if the sphere is at least partially inside the frustum
    /// relative to this plane.
    #[inline]
    pub fn test_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        self.distance_to_point(center) >= -radius
    }
}

// =============================================================================
// FRUSTUM PLANES
// =============================================================================

/// The 6 frustum planes extracted from a view-projection matrix.
///
/// Planes are ordered: left, right, bottom, top, near, far.
/// All plane normals point inward (toward the interior of the frustum).
///
/// # Memory Layout (96 bytes)
///
/// ```text
/// | Offset | Field     | Size |
/// |--------|-----------|------|
/// | 0      | planes[0] | 16   | (left)
/// | 16     | planes[1] | 16   | (right)
/// | 32     | planes[2] | 16   | (bottom)
/// | 48     | planes[3] | 16   | (top)
/// | 64     | planes[4] | 16   | (near)
/// | 80     | planes[5] | 16   | (far)
/// ```
///
/// # WGSL Binding
///
/// ```wgsl
/// struct FrustumPlane {
///     normal: vec3<f32>,
///     distance: f32,
/// }
///
/// struct FrustumPlanes {
///     planes: array<FrustumPlane, 6>,
/// }
///
/// @group(0) @binding(0) var<uniform> frustum: FrustumPlanes;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct FrustumPlanes {
    /// The 6 frustum planes in order: left, right, bottom, top, near, far.
    pub planes: [FrustumPlane; NUM_FRUSTUM_PLANES],
}

// Compile-time size assertions
const _: () = assert!(mem::size_of::<FrustumPlanes>() == FRUSTUM_PLANES_SIZE);
const _: () = assert!(mem::size_of::<FrustumPlanes>() == 96);

impl FrustumPlanes {
    /// Create frustum planes from a view-projection matrix.
    ///
    /// Uses the Gribb-Hartmann method to extract planes from the combined
    /// view-projection matrix. The matrix should be in column-major order
    /// (OpenGL/wgpu convention).
    ///
    /// # Arguments
    ///
    /// * `vp` - 4x4 view-projection matrix in column-major order
    ///
    /// # Algorithm
    ///
    /// The Gribb-Hartmann method extracts planes by combining matrix rows:
    /// - Left:   row3 + row0
    /// - Right:  row3 - row0
    /// - Bottom: row3 + row1
    /// - Top:    row3 - row1
    /// - Near:   row3 + row2 (for [0,1] depth like wgpu)
    /// - Far:    row3 - row2
    ///
    /// # Example
    ///
    /// ```ignore
    /// let view = compute_view_matrix(eye, target, up);
    /// let proj = compute_perspective_matrix(fovy, aspect, near, far);
    /// let vp = multiply_matrices(&view, &proj);
    /// let frustum = FrustumPlanes::from_view_projection(&vp);
    /// ```
    pub fn from_view_projection(vp: &[[f32; 4]; 4]) -> Self {
        // Gribb-Hartmann plane extraction for column-major matrices.
        // In column-major: vp[col][row]
        // Row i of the matrix is [vp[0][i], vp[1][i], vp[2][i], vp[3][i]]

        let row0 = [vp[0][0], vp[1][0], vp[2][0], vp[3][0]];
        let row1 = [vp[0][1], vp[1][1], vp[2][1], vp[3][1]];
        let row2 = [vp[0][2], vp[1][2], vp[2][2], vp[3][2]];
        let row3 = [vp[0][3], vp[1][3], vp[2][3], vp[3][3]];

        let mut planes = [FrustumPlane::default(); NUM_FRUSTUM_PLANES];

        // Left: row3 + row0
        planes[PLANE_LEFT] = FrustumPlane::new(
            [row3[0] + row0[0], row3[1] + row0[1], row3[2] + row0[2]],
            row3[3] + row0[3],
        );

        // Right: row3 - row0
        planes[PLANE_RIGHT] = FrustumPlane::new(
            [row3[0] - row0[0], row3[1] - row0[1], row3[2] - row0[2]],
            row3[3] - row0[3],
        );

        // Bottom: row3 + row1
        planes[PLANE_BOTTOM] = FrustumPlane::new(
            [row3[0] + row1[0], row3[1] + row1[1], row3[2] + row1[2]],
            row3[3] + row1[3],
        );

        // Top: row3 - row1
        planes[PLANE_TOP] = FrustumPlane::new(
            [row3[0] - row1[0], row3[1] - row1[1], row3[2] - row1[2]],
            row3[3] - row1[3],
        );

        // Near: row3 + row2 (for [0,1] depth range like wgpu)
        planes[PLANE_NEAR] = FrustumPlane::new(
            [row3[0] + row2[0], row3[1] + row2[1], row3[2] + row2[2]],
            row3[3] + row2[3],
        );

        // Far: row3 - row2
        planes[PLANE_FAR] = FrustumPlane::new(
            [row3[0] - row2[0], row3[1] - row2[1], row3[2] - row2[2]],
            row3[3] - row2[3],
        );

        Self { planes }
    }

    /// Create frustum planes for an identity view-projection.
    ///
    /// This creates a canonical frustum aligned with the coordinate axes,
    /// useful for testing.
    pub fn identity() -> Self {
        Self::from_view_projection(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Get a reference to a specific plane.
    #[inline]
    pub fn plane(&self, index: usize) -> &FrustumPlane {
        &self.planes[index]
    }

    /// Get the left plane.
    #[inline]
    pub fn left(&self) -> &FrustumPlane {
        &self.planes[PLANE_LEFT]
    }

    /// Get the right plane.
    #[inline]
    pub fn right(&self) -> &FrustumPlane {
        &self.planes[PLANE_RIGHT]
    }

    /// Get the bottom plane.
    #[inline]
    pub fn bottom(&self) -> &FrustumPlane {
        &self.planes[PLANE_BOTTOM]
    }

    /// Get the top plane.
    #[inline]
    pub fn top(&self) -> &FrustumPlane {
        &self.planes[PLANE_TOP]
    }

    /// Get the near plane.
    #[inline]
    pub fn near(&self) -> &FrustumPlane {
        &self.planes[PLANE_NEAR]
    }

    /// Get the far plane.
    #[inline]
    pub fn far(&self) -> &FrustumPlane {
        &self.planes[PLANE_FAR]
    }

    /// Check if all planes are properly normalized.
    pub fn all_normalized(&self) -> bool {
        self.planes.iter().all(|p| p.is_normalized())
    }

    /// Test if a sphere is visible (inside or intersecting the frustum).
    ///
    /// This is a CPU reference implementation for testing and fallback.
    pub fn test_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        for plane in &self.planes {
            if !plane.test_sphere(center, radius) {
                return false;
            }
        }
        true
    }

    /// Test if an AABB is visible (inside or intersecting the frustum).
    ///
    /// Uses the p-vertex optimization: for each plane, only test the corner
    /// most aligned with the plane normal.
    pub fn test_aabb(&self, aabb_min: [f32; 3], aabb_max: [f32; 3]) -> bool {
        for plane in &self.planes {
            // P-vertex: corner most aligned with normal
            let p = [
                if plane.normal[0] >= 0.0 { aabb_max[0] } else { aabb_min[0] },
                if plane.normal[1] >= 0.0 { aabb_max[1] } else { aabb_min[1] },
                if plane.normal[2] >= 0.0 { aabb_max[2] } else { aabb_min[2] },
            ];

            if plane.distance_to_point(p) < 0.0 {
                return false;
            }
        }
        true
    }
}

// =============================================================================
// FRUSTUM BUFFER
// =============================================================================

/// GPU buffer wrapper for FrustumPlanes uniform data.
///
/// Manages a wgpu uniform buffer containing the 6 frustum planes for
/// GPU-driven culling compute shaders.
///
/// # Usage
///
/// ```ignore
/// // Create buffer
/// let mut buffer = FrustumBuffer::new(&device);
///
/// // Update each frame
/// buffer.update(&queue, &view_projection_matrix);
///
/// // Create bind group
/// let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
///     layout: &layout,
///     entries: &[wgpu::BindGroupEntry {
///         binding: 0,
///         resource: buffer.buffer().as_entire_binding(),
///     }],
///     label: Some("frustum_bind_group"),
/// });
/// ```
pub struct FrustumBuffer {
    /// The wgpu uniform buffer.
    buffer: wgpu::Buffer,
    /// Current frustum planes (CPU-side cache).
    planes: FrustumPlanes,
}

impl FrustumBuffer {
    /// Create a new FrustumBuffer.
    ///
    /// The buffer is initialized with identity frustum planes.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for buffer creation
    pub fn new(device: &wgpu::Device) -> Self {
        let planes = FrustumPlanes::identity();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frustum_planes_buffer"),
            size: FRUSTUM_PLANES_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { buffer, planes }
    }

    /// Create a new FrustumBuffer with a custom label.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for buffer creation
    /// * `label` - Debug label for the buffer
    pub fn with_label(device: &wgpu::Device, label: &str) -> Self {
        let planes = FrustumPlanes::identity();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: FRUSTUM_PLANES_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { buffer, planes }
    }

    /// Update the frustum planes from a view-projection matrix.
    ///
    /// Extracts planes using the Gribb-Hartmann method and uploads
    /// them to the GPU buffer.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue for buffer writes
    /// * `view_projection` - 4x4 VP matrix in column-major order
    pub fn update(&mut self, queue: &wgpu::Queue, view_projection: &[[f32; 4]; 4]) {
        self.planes = FrustumPlanes::from_view_projection(view_projection);
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&self.planes));
    }

    /// Update with pre-extracted frustum planes.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue for buffer writes
    /// * `planes` - The frustum planes to upload
    pub fn update_planes(&mut self, queue: &wgpu::Queue, planes: &FrustumPlanes) {
        self.planes = *planes;
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&self.planes));
    }

    /// Get the GPU buffer.
    #[inline]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the buffer as a binding resource.
    #[inline]
    pub fn as_entire_binding(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }

    /// Get the current frustum planes (CPU-side cache).
    #[inline]
    pub fn planes(&self) -> &FrustumPlanes {
        &self.planes
    }

    /// Get buffer size in bytes.
    #[inline]
    pub const fn size() -> u64 {
        FRUSTUM_PLANES_SIZE as u64
    }
}

impl std::fmt::Debug for FrustumBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrustumBuffer")
            .field("planes", &self.planes)
            .finish_non_exhaustive()
    }
}

// =============================================================================
// WGSL SHADER INTEGRATION (T-WGPU-P6.3.2)
// =============================================================================

/// WGSL shader source for AABB-frustum intersection testing.
///
/// Contains reusable functions for GPU-based frustum culling:
/// - `test_aabb_frustum()` - Basic visibility test with early-out
/// - `test_aabb_frustum_detailed()` - Returns outside/intersecting/inside
/// - `test_aabb_plane()` - Single plane test
/// - `test_obb_frustum()` - Oriented bounding box test
///
/// Also includes compute shader entry points for batch culling:
/// - `cull_aabb_batch` - Batch AABB culling (outputs 0/1)
/// - `cull_aabb_batch_detailed` - Batch culling with detailed result
///
/// # WGSL Usage
///
/// ```wgsl
/// // Include this shader and call the functions:
/// let visible = test_aabb_frustum(aabb_min, aabb_max);
/// let result = test_aabb_frustum_detailed(aabb_min, aabb_max);
/// ```
pub const FRUSTUM_CULL_SHADER: &str = include_str!("../../shaders/frustum_cull.wgsl");

/// Create a bind group layout for the frustum planes uniform buffer.
///
/// This layout is used by shaders that need to access the frustum planes
/// for visibility testing. The buffer should contain a `FrustumPlanes`
/// struct (96 bytes).
///
/// # Binding Layout
///
/// | Binding | Type    | Stage   | Description                    |
/// |---------|---------|---------|--------------------------------|
/// | 0       | Uniform | Compute | FrustumPlanes (6 planes, 96B)  |
///
/// # Example
///
/// ```ignore
/// let layout = create_frustum_cull_bind_group_layout(&device);
/// let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
///     layout: &layout,
///     entries: &[wgpu::BindGroupEntry {
///         binding: 0,
///         resource: frustum_buffer.as_entire_binding(),
///     }],
///     label: Some("frustum_cull_bind_group"),
/// });
/// ```
pub fn create_frustum_cull_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("frustum_cull_bind_group_layout"),
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
    })
}

/// Create a bind group layout for batch AABB frustum culling.
///
/// This layout is used by the `cull_aabb_batch` compute shader entry point.
/// It requires input AABBs and outputs visibility flags.
///
/// # Binding Layout
///
/// | Binding | Type      | Stage   | Description                       |
/// |---------|-----------|---------|-----------------------------------|
/// | 0       | Uniform   | Compute | CullParams (num_objects, flags)   |
/// | 1       | Storage R | Compute | Input AABBs array                 |
/// | 2       | Storage RW| Compute | Output visibility flags           |
///
/// # Example
///
/// ```ignore
/// let layout = create_frustum_cull_batch_bind_group_layout(&device);
/// ```
pub fn create_frustum_cull_batch_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("frustum_cull_batch_bind_group_layout"),
        entries: &[
            // CullParams uniform (num_objects, flags, padding)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                },
                count: None,
            },
            // Input AABBs (storage, read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(std::num::NonZeroU64::new(32).unwrap()),
                },
                count: None,
            },
            // Output visibility flags (storage, read-write)
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
    })
}

/// AABB struct for batch frustum culling input.
///
/// Matches the `InputAABB` struct in frustum_cull.wgsl.
/// Each AABB is 32 bytes (with padding for GPU alignment).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct CullAABB {
    /// Minimum corner of AABB in world space
    pub min: [f32; 3],
    /// Padding for 16-byte alignment
    pub _pad0: f32,
    /// Maximum corner of AABB in world space
    pub max: [f32; 3],
    /// Padding for 16-byte alignment
    pub _pad1: f32,
}

/// Size of CullAABB in bytes.
pub const CULL_AABB_SIZE: usize = 32;

// Compile-time size assertion
const _: () = assert!(mem::size_of::<CullAABB>() == CULL_AABB_SIZE);

impl CullAABB {
    /// Create a new CullAABB from min/max corners.
    #[inline]
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self {
            min,
            _pad0: 0.0,
            max,
            _pad1: 0.0,
        }
    }

    /// Create from center and half-extents.
    #[inline]
    pub fn from_center_extents(center: [f32; 3], half_extents: [f32; 3]) -> Self {
        Self::new(
            [
                center[0] - half_extents[0],
                center[1] - half_extents[1],
                center[2] - half_extents[2],
            ],
            [
                center[0] + half_extents[0],
                center[1] + half_extents[1],
                center[2] + half_extents[2],
            ],
        )
    }
}

/// Parameters for batch AABB frustum culling compute shader.
///
/// Matches the `CullParams` struct in frustum_cull.wgsl.
/// Named `FrustumCullParams` to avoid collision with `CullParams` in frustum_cull.rs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct FrustumCullParams {
    /// Number of AABBs to process
    pub num_objects: u32,
    /// Flags (reserved for future use)
    pub flags: u32,
    /// Padding for 16-byte alignment
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Size of FrustumCullParams in bytes.
pub const FRUSTUM_CULL_PARAMS_SIZE: usize = 16;

// Compile-time size assertion
const _: () = assert!(mem::size_of::<FrustumCullParams>() == FRUSTUM_CULL_PARAMS_SIZE);

impl FrustumCullParams {
    /// Create new cull parameters.
    #[inline]
    pub fn new(num_objects: u32) -> Self {
        Self {
            num_objects,
            flags: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Visibility result constants matching WGSL shader.
pub const VISIBILITY_OUTSIDE: u32 = 0;
pub const VISIBILITY_INTERSECTING: u32 = 1;
pub const VISIBILITY_INSIDE: u32 = 2;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create a perspective projection matrix.
///
/// This helper creates a standard perspective matrix for testing.
/// For production use, prefer a full math library.
///
/// # Arguments
///
/// * `fovy` - Vertical field of view in radians
/// * `aspect` - Aspect ratio (width / height)
/// * `near` - Near clipping plane distance
/// * `far` - Far clipping plane distance
///
/// # Returns
///
/// A 4x4 perspective projection matrix in column-major order.
pub fn perspective_matrix(fovy: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fovy * 0.5).tan();

    // wgpu uses [0,1] depth range (not [-1,1])
    let range_inv = 1.0 / (near - far);

    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far * range_inv, -1.0],
        [0.0, 0.0, near * far * range_inv, 0.0],
    ]
}

/// Create a look-at view matrix.
///
/// This helper creates a standard view matrix for testing.
///
/// # Arguments
///
/// * `eye` - Camera position
/// * `target` - Look-at target point
/// * `up` - World up vector
///
/// # Returns
///
/// A 4x4 view matrix in column-major order.
pub fn look_at_matrix(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize([
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    ]);
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [
            -dot(s, eye),
            -dot(u, eye),
            dot(f, eye),
            1.0,
        ],
    ]
}

/// Multiply two 4x4 matrices.
pub fn multiply_matrices(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[k][j] * b[i][k];
            }
        }
    }
    result
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < NORMALIZE_EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Size and Layout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frustum_plane_size() {
        assert_eq!(
            mem::size_of::<FrustumPlane>(),
            16,
            "FrustumPlane must be 16 bytes"
        );
        assert_eq!(
            FRUSTUM_PLANE_SIZE,
            16,
            "FRUSTUM_PLANE_SIZE constant must be 16"
        );
    }

    #[test]
    fn test_frustum_planes_size() {
        assert_eq!(
            mem::size_of::<FrustumPlanes>(),
            96,
            "FrustumPlanes must be 96 bytes (6 * 16)"
        );
        assert_eq!(
            FRUSTUM_PLANES_SIZE,
            96,
            "FRUSTUM_PLANES_SIZE constant must be 96"
        );
    }

    #[test]
    fn test_bytemuck_size() {
        // Verify bytemuck compatibility
        let planes = FrustumPlanes::identity();
        let bytes: &[u8] = bytemuck::bytes_of(&planes);
        assert_eq!(bytes.len(), 96, "Bytemuck bytes should be 96");
    }

    // -------------------------------------------------------------------------
    // Plane Extraction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_extract_identity_matrix() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let planes = FrustumPlanes::from_view_projection(&identity);

        // All planes should be normalized
        assert!(planes.all_normalized(), "All planes should be normalized");

        // Check plane count
        assert_eq!(planes.planes.len(), 6);
    }

    #[test]
    fn test_extract_perspective_matrix() {
        // Create a realistic perspective matrix
        let fovy = std::f32::consts::FRAC_PI_4; // 45 degrees
        let aspect = 16.0 / 9.0;
        let near = 0.1;
        let far = 100.0;

        let proj = perspective_matrix(fovy, aspect, near, far);

        // Create a simple view matrix (looking down -Z)
        let view = look_at_matrix(
            [0.0, 0.0, 5.0],  // eye
            [0.0, 0.0, 0.0],  // target
            [0.0, 1.0, 0.0],  // up
        );

        let vp = multiply_matrices(&view, &proj);
        let planes = FrustumPlanes::from_view_projection(&vp);

        // Verify all planes are normalized
        assert!(planes.all_normalized(), "All planes should be normalized");

        // Test that points inside frustum are visible
        // Object at origin should be visible
        let visible = planes.test_sphere([0.0, 0.0, 0.0], 1.0);
        assert!(visible, "Sphere at origin should be visible");

        // Object far behind camera should be culled
        let culled = planes.test_sphere([0.0, 0.0, 100.0], 1.0);
        assert!(!culled, "Sphere behind camera should be culled");
    }

    // -------------------------------------------------------------------------
    // Normalization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_plane_normalization() {
        // Non-normalized input
        let plane = FrustumPlane::new([2.0, 0.0, 0.0], 4.0);

        // Normal should be unit length
        let normal_len = (plane.normal[0].powi(2)
            + plane.normal[1].powi(2)
            + plane.normal[2].powi(2))
        .sqrt();
        assert!(
            (normal_len - 1.0).abs() < 1e-5,
            "Normal should be unit length: {}",
            normal_len
        );

        // Check actual values
        assert!((plane.normal[0] - 1.0).abs() < 1e-6, "normal.x should be 1.0");
        assert!((plane.distance - 2.0).abs() < 1e-6, "distance should be scaled");
    }

    #[test]
    fn test_plane_is_normalized() {
        let normalized = FrustumPlane::new([1.0, 0.0, 0.0], 1.0);
        assert!(normalized.is_normalized());

        // Manually create unnormalized plane
        let mut unnormalized = FrustumPlane {
            normal: [2.0, 0.0, 0.0],
            distance: 1.0,
        };
        assert!(!unnormalized.is_normalized());

        unnormalized.normalize();
        assert!(unnormalized.is_normalized());
    }

    #[test]
    fn test_zero_normal_handling() {
        // Zero normal should default to (0,0,1)
        let plane = FrustumPlane::new([0.0, 0.0, 0.0], 1.0);
        assert_eq!(plane.normal, [0.0, 0.0, 1.0]);
        assert_eq!(plane.distance, 0.0);
    }

    // -------------------------------------------------------------------------
    // Culling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sphere_inside_frustum() {
        let planes = FrustumPlanes::from_view_projection(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        // Small sphere at origin
        let visible = planes.test_sphere([0.0, 0.0, 0.0], 0.1);
        assert!(visible, "Small sphere at origin should be visible");
    }

    #[test]
    fn test_aabb_culling() {
        // Create frustum looking down -Z
        let view = look_at_matrix([0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = perspective_matrix(1.0, 1.0, 1.0, 100.0);
        let vp = multiply_matrices(&view, &proj);
        let planes = FrustumPlanes::from_view_projection(&vp);

        // AABB in front of camera
        let visible = planes.test_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert!(visible, "AABB at origin should be visible");

        // AABB way to the side
        let culled = planes.test_aabb([100.0, 100.0, 0.0], [102.0, 102.0, 2.0]);
        assert!(!culled, "AABB far to the side should be culled");
    }

    // -------------------------------------------------------------------------
    // Plane Accessor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_plane_accessors() {
        let planes = FrustumPlanes::identity();

        // Test index accessors
        assert!(std::ptr::eq(planes.plane(PLANE_LEFT), planes.left()));
        assert!(std::ptr::eq(planes.plane(PLANE_RIGHT), planes.right()));
        assert!(std::ptr::eq(planes.plane(PLANE_BOTTOM), planes.bottom()));
        assert!(std::ptr::eq(planes.plane(PLANE_TOP), planes.top()));
        assert!(std::ptr::eq(planes.plane(PLANE_NEAR), planes.near()));
        assert!(std::ptr::eq(planes.plane(PLANE_FAR), planes.far()));
    }

    // -------------------------------------------------------------------------
    // Distance Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_distance_to_point() {
        // Plane at z = 0, facing +Z
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);

        // Point in front (positive side)
        assert!(plane.distance_to_point([0.0, 0.0, 5.0]) > 0.0);

        // Point behind (negative side)
        assert!(plane.distance_to_point([0.0, 0.0, -5.0]) < 0.0);

        // Point on plane
        assert!((plane.distance_to_point([0.0, 0.0, 0.0])).abs() < 1e-6);
    }

    #[test]
    fn test_from_point_normal() {
        let point = [0.0, 0.0, 5.0];
        let normal = [0.0, 0.0, 1.0];
        let plane = FrustumPlane::from_point_normal(point, normal);

        // Point should be on the plane
        let dist = plane.distance_to_point(point);
        assert!(dist.abs() < 1e-5, "Point should be on plane, dist = {}", dist);
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_perspective_matrix() {
        let proj = perspective_matrix(1.0, 1.0, 0.1, 100.0);

        // Verify it's a valid perspective matrix
        // The [3][2] element should be -1 for right-handed coords
        assert!((proj[2][3] - (-1.0)).abs() < 1e-6);

        // The [3][3] element should be 0
        assert!(proj[3][3].abs() < 1e-6);
    }

    #[test]
    fn test_look_at_matrix() {
        let view = look_at_matrix([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

        // The view matrix should place the camera at (0,0,5) looking at origin
        // Translation component should be related to eye position
        assert!(view[3][3].abs() - 1.0 < 1e-6, "w component should be 1");
    }

    #[test]
    fn test_matrix_multiply() {
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let result = multiply_matrices(&identity, &identity);

        // Identity * Identity = Identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result[i][j] - expected).abs() < 1e-6,
                    "result[{}][{}] = {}, expected {}",
                    i, j, result[i][j], expected
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_plane_indices() {
        assert_eq!(PLANE_LEFT, 0);
        assert_eq!(PLANE_RIGHT, 1);
        assert_eq!(PLANE_BOTTOM, 2);
        assert_eq!(PLANE_TOP, 3);
        assert_eq!(PLANE_NEAR, 4);
        assert_eq!(PLANE_FAR, 5);
        assert_eq!(NUM_FRUSTUM_PLANES, 6);
    }
}
