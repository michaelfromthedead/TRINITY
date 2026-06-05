//! GPU Frustum Culling for TRINITY Engine (T-GPU-3.1).
//!
//! This module provides GPU-based frustum culling using compute shaders.
//! It tests instance bounding volumes (sphere + AABB) against the view
//! frustum to determine visibility.
//!
//! # Overview
//!
//! Frustum culling eliminates objects outside the camera's view frustum
//! before rendering. This module implements a two-phase approach:
//!
//! 1. **Sphere Test**: Quick bounding sphere test (cheap, conservative)
//! 2. **AABB Test**: Precise axis-aligned bounding box test (more accurate)
//!
//! The sphere test catches most culled objects early, while the AABB test
//! refines false positives from the conservative sphere test.
//!
//! # Performance
//!
//! - Work complexity: O(n), one thread per instance
//! - Target: < 0.1ms for 100K instances
//! - Memory: 48 bytes per instance bounds
//!
//! # Usage
//!
//! ```ignore
//! // Create pipeline and resources
//! let pipeline = FrustumCullPipeline::new(&device);
//! let resources = FrustumCullResources::new(&device, 100_000);
//!
//! // Each frame: update frustum and cull
//! resources.upload_frustum(&queue, &frustum);
//! resources.upload_instances(&queue, &instance_bounds);
//! pipeline.dispatch(&mut encoder, &resources, instance_count);
//!
//! // Read visibility results
//! let visible = resources.read_visibility(&device, &queue);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Number of frustum planes (left, right, bottom, top, near, far).
pub const NUM_FRUSTUM_PLANES: usize = 6;

/// Culling flag: Use sphere test when available.
pub const FLAG_USE_SPHERE: u32 = 1;

/// Culling flag: Debug mode (always mark visible).
pub const FLAG_DEBUG_VISIBLE: u32 = 2;

// ---------------------------------------------------------------------------
// FrustumPlane
// ---------------------------------------------------------------------------

/// A frustum plane in Hessian normal form.
///
/// The plane equation is: `dot(normal, point) + distance = 0`
///
/// Points where `dot(normal, point) + distance > 0` are on the positive
/// (inside) side of the plane.
///
/// # Memory Layout
///
/// 16 bytes, vec4 aligned:
/// | Offset | Field    | Size |
/// |--------|----------|------|
/// | 0      | normal.x | 4    |
/// | 4      | normal.y | 4    |
/// | 8      | normal.z | 4    |
/// | 12     | distance | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrustumPlane {
    /// Normalized plane normal (pointing into frustum).
    pub normal: [f32; 3],
    /// Signed distance from origin.
    pub distance: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<FrustumPlane>() == 16);

impl FrustumPlane {
    /// Create a new frustum plane.
    ///
    /// The normal is automatically normalized.
    pub fn new(normal: [f32; 3], distance: f32) -> Self {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len < 1e-8 {
            return Self {
                normal: [0.0, 0.0, 1.0],
                distance: 0.0,
            };
        }
        let inv_len = 1.0 / len;
        Self {
            normal: [normal[0] * inv_len, normal[1] * inv_len, normal[2] * inv_len],
            distance: distance * inv_len,
        }
    }

    /// Create a plane from a point and normal.
    pub fn from_point_normal(point: [f32; 3], normal: [f32; 3]) -> Self {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len < 1e-8 {
            return Self {
                normal: [0.0, 0.0, 1.0],
                distance: 0.0,
            };
        }
        let inv_len = 1.0 / len;
        let n = [normal[0] * inv_len, normal[1] * inv_len, normal[2] * inv_len];
        let d = -(n[0] * point[0] + n[1] * point[1] + n[2] * point[2]);
        Self { normal: n, distance: d }
    }

    /// Compute signed distance from a point to the plane.
    #[inline]
    pub fn distance_to_point(&self, point: [f32; 3]) -> f32 {
        self.normal[0] * point[0] + self.normal[1] * point[1] + self.normal[2] * point[2] + self.distance
    }
}

// ---------------------------------------------------------------------------
// Frustum
// ---------------------------------------------------------------------------

/// Plane indices in the frustum array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum FrustumPlaneIndex {
    Left = 0,
    Right = 1,
    Bottom = 2,
    Top = 3,
    Near = 4,
    Far = 5,
}

/// View frustum defined by 6 planes.
///
/// Planes are ordered: left, right, bottom, top, near, far.
/// All plane normals point inward (toward the interior of the frustum).
///
/// # Memory Layout
///
/// 96 bytes (6 * 16 bytes per plane):
/// | Offset | Field     | Size |
/// |--------|-----------|------|
/// | 0      | left      | 16   |
/// | 16     | right     | 16   |
/// | 32     | bottom    | 16   |
/// | 48     | top       | 16   |
/// | 64     | near      | 16   |
/// | 80     | far       | 16   |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Frustum {
    /// The 6 frustum planes.
    pub planes: [FrustumPlane; 6],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<Frustum>() == 96);

impl Frustum {
    /// Create a frustum from a view-projection matrix.
    ///
    /// Uses the Gribb-Hartmann method to extract planes from the combined
    /// view-projection matrix.
    ///
    /// # Arguments
    ///
    /// * `vp` - 4x4 view-projection matrix in column-major order (OpenGL/wgpu style)
    ///
    /// # Returns
    ///
    /// A `Frustum` with all planes normalized and normals pointing inward.
    pub fn from_view_projection(vp: &[[f32; 4]; 4]) -> Self {
        // Gribb-Hartmann plane extraction for column-major matrices
        // plane[i] = row3 op row[i] where op is + or -

        // Column-major access: vp[col][row]
        // Row i of the matrix is [vp[0][i], vp[1][i], vp[2][i], vp[3][i]]

        let row0 = [vp[0][0], vp[1][0], vp[2][0], vp[3][0]];
        let row1 = [vp[0][1], vp[1][1], vp[2][1], vp[3][1]];
        let row2 = [vp[0][2], vp[1][2], vp[2][2], vp[3][2]];
        let row3 = [vp[0][3], vp[1][3], vp[2][3], vp[3][3]];

        let mut planes = [FrustumPlane::default(); 6];

        // Left: row3 + row0
        planes[FrustumPlaneIndex::Left as usize] = FrustumPlane::new(
            [row3[0] + row0[0], row3[1] + row0[1], row3[2] + row0[2]],
            row3[3] + row0[3],
        );

        // Right: row3 - row0
        planes[FrustumPlaneIndex::Right as usize] = FrustumPlane::new(
            [row3[0] - row0[0], row3[1] - row0[1], row3[2] - row0[2]],
            row3[3] - row0[3],
        );

        // Bottom: row3 + row1
        planes[FrustumPlaneIndex::Bottom as usize] = FrustumPlane::new(
            [row3[0] + row1[0], row3[1] + row1[1], row3[2] + row1[2]],
            row3[3] + row1[3],
        );

        // Top: row3 - row1
        planes[FrustumPlaneIndex::Top as usize] = FrustumPlane::new(
            [row3[0] - row1[0], row3[1] - row1[1], row3[2] - row1[2]],
            row3[3] - row1[3],
        );

        // Near: row3 + row2 (for depth [0,1] like wgpu)
        planes[FrustumPlaneIndex::Near as usize] = FrustumPlane::new(
            [row3[0] + row2[0], row3[1] + row2[1], row3[2] + row2[2]],
            row3[3] + row2[3],
        );

        // Far: row3 - row2
        planes[FrustumPlaneIndex::Far as usize] = FrustumPlane::new(
            [row3[0] - row2[0], row3[1] - row2[1], row3[2] - row2[2]],
            row3[3] - row2[3],
        );

        Self { planes }
    }

    /// Test if a sphere is visible (inside or intersecting the frustum).
    ///
    /// This is the CPU reference implementation for testing.
    pub fn test_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        for plane in &self.planes {
            let dist = plane.distance_to_point(center);
            if dist < -radius {
                return false; // Sphere is entirely outside this plane
            }
        }
        true
    }

    /// Test if an AABB is visible (inside or intersecting the frustum).
    ///
    /// Uses the p-vertex optimization: for each plane, test only the corner
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
                return false; // AABB entirely outside this plane
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// CullParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for culling parameters.
///
/// # Memory Layout
///
/// 16 bytes, std140/std430 compatible:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | num_instances | 4    |
/// | 4      | flags         | 4    |
/// | 8      | _pad0         | 4    |
/// | 12     | _pad1         | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullParams {
    /// Number of instances to process.
    pub num_instances: u32,
    /// Culling flags (see FLAG_* constants).
    pub flags: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    pub _pad1: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<CullParams>() == 16);

impl CullParams {
    /// Create parameters for the given instance count.
    pub fn new(num_instances: u32) -> Self {
        Self {
            num_instances,
            flags: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Create parameters with flags.
    pub fn with_flags(num_instances: u32, flags: u32) -> Self {
        Self {
            num_instances,
            flags,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_instances + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

// ---------------------------------------------------------------------------
// InstanceBounds
// ---------------------------------------------------------------------------

/// Bounding data for a single instance.
///
/// Contains both a bounding sphere (for fast rejection) and an AABB
/// (for precise culling).
///
/// # Memory Layout
///
/// 48 bytes, vec4 aligned:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | sphere_center | 12   |
/// | 12     | sphere_radius | 4    |
/// | 16     | aabb_min      | 12   |
/// | 28     | _pad0         | 4    |
/// | 32     | aabb_max      | 12   |
/// | 44     | _pad1         | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceBounds {
    /// Center of bounding sphere in world space.
    pub sphere_center: [f32; 3],
    /// Radius of bounding sphere. Set to 0 to use AABB only.
    pub sphere_radius: f32,
    /// Minimum corner of AABB in world space.
    pub aabb_min: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad0: f32,
    /// Maximum corner of AABB in world space.
    pub aabb_max: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad1: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<InstanceBounds>() == 48);

impl InstanceBounds {
    /// Create instance bounds from sphere and AABB.
    pub fn new(
        sphere_center: [f32; 3],
        sphere_radius: f32,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
    ) -> Self {
        Self {
            sphere_center,
            sphere_radius,
            aabb_min,
            _pad0: 0.0,
            aabb_max,
            _pad1: 0.0,
        }
    }

    /// Create instance bounds from AABB only (sphere radius = 0).
    pub fn from_aabb(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        // Compute center as midpoint
        let center = [
            (aabb_min[0] + aabb_max[0]) * 0.5,
            (aabb_min[1] + aabb_max[1]) * 0.5,
            (aabb_min[2] + aabb_max[2]) * 0.5,
        ];
        Self {
            sphere_center: center,
            sphere_radius: 0.0, // Use AABB only
            aabb_min,
            _pad0: 0.0,
            aabb_max,
            _pad1: 0.0,
        }
    }

    /// Create instance bounds from AABB with auto-computed bounding sphere.
    pub fn from_aabb_with_sphere(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
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

        Self {
            sphere_center: center,
            sphere_radius: radius,
            aabb_min,
            _pad0: 0.0,
            aabb_max,
            _pad1: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// FrustumCullResources
// ---------------------------------------------------------------------------

/// GPU resources for frustum culling.
///
/// Contains all buffers needed for the culling compute shader.
pub struct FrustumCullResources {
    /// Uniform buffer for culling parameters.
    pub params_buffer: wgpu::Buffer,
    /// Uniform buffer for frustum planes.
    pub frustum_buffer: wgpu::Buffer,
    /// Storage buffer for instance bounds (input).
    pub instances_buffer: wgpu::Buffer,
    /// Storage buffer for visibility flags (output).
    pub visibility_buffer: wgpu::Buffer,
    /// Staging buffer for reading visibility back to CPU.
    pub visibility_staging: wgpu::Buffer,
    /// Maximum number of instances supported.
    pub capacity: u32,
}

impl FrustumCullResources {
    /// Create culling resources for the given capacity.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frustum_cull_params"),
            size: mem::size_of::<CullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frustum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frustum_cull_frustum"),
            size: mem::size_of::<Frustum>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frustum_cull_instances"),
            size: (capacity as u64) * (mem::size_of::<InstanceBounds>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frustum_cull_visibility"),
            size: (capacity as u64) * 4, // u32 per instance
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let visibility_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frustum_cull_visibility_staging"),
            size: (capacity as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            frustum_buffer,
            instances_buffer,
            visibility_buffer,
            visibility_staging,
            capacity,
        }
    }

    /// Upload culling parameters to GPU.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &CullParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload frustum to GPU.
    pub fn upload_frustum(&self, queue: &wgpu::Queue, frustum: &Frustum) {
        queue.write_buffer(&self.frustum_buffer, 0, bytemuck::bytes_of(frustum));
    }

    /// Upload instance bounds to GPU.
    ///
    /// # Panics
    ///
    /// Panics if `instances.len() > self.capacity`.
    pub fn upload_instances(&self, queue: &wgpu::Queue, instances: &[InstanceBounds]) {
        assert!(instances.len() <= self.capacity as usize);
        queue.write_buffer(&self.instances_buffer, 0, bytemuck::cast_slice(instances));
    }
}

// ---------------------------------------------------------------------------
// FrustumCullPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for frustum culling.
pub struct FrustumCullPipeline {
    /// Main culling pipeline (sphere + AABB).
    pub pipeline: wgpu::ComputePipeline,
    /// Sphere-only culling pipeline.
    pub pipeline_sphere_only: wgpu::ComputePipeline,
    /// AABB-only culling pipeline.
    pub pipeline_aabb_only: wgpu::ComputePipeline,
    /// Bind group layout for culling resources.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl FrustumCullPipeline {
    /// Create the culling pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `shader_source` - WGSL shader source code.
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frustum_cull_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frustum_cull_bind_group_layout"),
            entries: &[
                // @binding(0) params: CullParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<CullParams>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) frustum: Frustum
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<Frustum>() as u64).unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(2) instances: array<InstanceBounds>
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
                // @binding(3) visibility: array<u32>
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
            label: Some("frustum_cull_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frustum_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_frustum",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_sphere_only = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frustum_cull_pipeline_sphere_only"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_frustum_sphere_only",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_aabb_only = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frustum_cull_pipeline_aabb_only"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_frustum_aabb_only",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            pipeline_sphere_only,
            pipeline_aabb_only,
            bind_group_layout,
        }
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &FrustumCullResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frustum_cull_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.frustum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.instances_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.visibility_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation of frustum culling.
///
/// Used for testing and fallback when GPU is not available.
pub fn cpu_frustum_cull(frustum: &Frustum, instances: &[InstanceBounds]) -> Vec<u32> {
    instances
        .iter()
        .map(|bounds| {
            let visible = if bounds.sphere_radius <= 0.0 {
                // No sphere: AABB only
                frustum.test_aabb(bounds.aabb_min, bounds.aabb_max)
            } else {
                // Sphere test first
                if frustum.test_sphere(bounds.sphere_center, bounds.sphere_radius) {
                    // Sphere passed, refine with AABB
                    frustum.test_aabb(bounds.aabb_min, bounds.aabb_max)
                } else {
                    false
                }
            };
            if visible { 1 } else { 0 }
        })
        .collect()
}

/// CPU reference implementation for sphere-only culling.
pub fn cpu_frustum_cull_sphere_only(frustum: &Frustum, instances: &[InstanceBounds]) -> Vec<u32> {
    instances
        .iter()
        .map(|bounds| {
            let visible = if bounds.sphere_radius <= 0.0 {
                frustum.test_aabb(bounds.aabb_min, bounds.aabb_max)
            } else {
                frustum.test_sphere(bounds.sphere_center, bounds.sphere_radius)
            };
            if visible { 1 } else { 0 }
        })
        .collect()
}

/// CPU reference implementation for AABB-only culling.
pub fn cpu_frustum_cull_aabb_only(frustum: &Frustum, instances: &[InstanceBounds]) -> Vec<u32> {
    instances
        .iter()
        .map(|bounds| {
            if frustum.test_aabb(bounds.aabb_min, bounds.aabb_max) {
                1
            } else {
                0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: Create a simple perspective frustum for testing.
    /// Camera at origin, looking down -Z, with a 90-degree FOV.
    fn make_test_frustum() -> Frustum {
        // Simple frustum: left, right, bottom, top at 45 degrees, near=1, far=100
        Frustum {
            planes: [
                // Left: normal points right (+X) at 45 degrees into frustum
                FrustumPlane::new([1.0, 0.0, -1.0], 0.0),
                // Right: normal points left (-X) at 45 degrees into frustum
                FrustumPlane::new([-1.0, 0.0, -1.0], 0.0),
                // Bottom: normal points up (+Y) at 45 degrees into frustum
                FrustumPlane::new([0.0, 1.0, -1.0], 0.0),
                // Top: normal points down (-Y) at 45 degrees into frustum
                FrustumPlane::new([0.0, -1.0, -1.0], 0.0),
                // Near: normal points forward (-Z), at z = -1
                FrustumPlane::new([0.0, 0.0, -1.0], -1.0),
                // Far: normal points backward (+Z), at z = -100
                FrustumPlane::new([0.0, 0.0, 1.0], 100.0),
            ],
        }
    }

    #[test]
    fn test_sphere_inside_frustum_visible() {
        let frustum = make_test_frustum();
        // Sphere at (0, 0, -10), radius 1: should be visible
        let center = [0.0, 0.0, -10.0];
        let radius = 1.0;
        assert!(frustum.test_sphere(center, radius));
    }

    #[test]
    fn test_sphere_outside_frustum_culled() {
        let frustum = make_test_frustum();
        // Sphere at (100, 0, -10), radius 1: way outside right side
        let center = [100.0, 0.0, -10.0];
        let radius = 1.0;
        assert!(!frustum.test_sphere(center, radius));

        // Sphere at (0, 0, -200), radius 1: beyond far plane
        let center = [0.0, 0.0, -200.0];
        assert!(!frustum.test_sphere(center, radius));

        // Sphere at (0, 0, 10), radius 1: behind camera (in front of near plane)
        let center = [0.0, 0.0, 10.0];
        assert!(!frustum.test_sphere(center, radius));
    }

    #[test]
    fn test_sphere_intersecting_frustum_visible() {
        let frustum = make_test_frustum();
        // Sphere touching near plane from inside
        let center = [0.0, 0.0, -2.0];
        let radius = 1.5; // Extends past near plane
        assert!(frustum.test_sphere(center, radius));
    }

    #[test]
    fn test_aabb_inside_frustum_visible() {
        let frustum = make_test_frustum();
        // Small AABB centered at (0, 0, -10)
        let aabb_min = [-1.0, -1.0, -11.0];
        let aabb_max = [1.0, 1.0, -9.0];
        assert!(frustum.test_aabb(aabb_min, aabb_max));
    }

    #[test]
    fn test_aabb_outside_frustum_culled() {
        let frustum = make_test_frustum();
        // AABB far to the right
        let aabb_min = [50.0, -1.0, -11.0];
        let aabb_max = [52.0, 1.0, -9.0];
        assert!(!frustum.test_aabb(aabb_min, aabb_max));
    }

    #[test]
    fn test_aabb_partial_overlap_visible() {
        let frustum = make_test_frustum();
        // Large AABB that partially overlaps frustum
        let aabb_min = [-5.0, -5.0, -20.0];
        let aabb_max = [5.0, 5.0, -10.0];
        assert!(frustum.test_aabb(aabb_min, aabb_max));
    }

    #[test]
    fn test_zero_radius_uses_aabb() {
        let frustum = make_test_frustum();

        // Instance with zero radius: should use AABB test
        let bounds = InstanceBounds::from_aabb([-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]);
        assert_eq!(bounds.sphere_radius, 0.0);

        let visibility = cpu_frustum_cull(&frustum, &[bounds]);
        assert_eq!(visibility[0], 1); // Should be visible via AABB

        // Instance with zero radius outside frustum
        let bounds_outside = InstanceBounds::from_aabb([50.0, 50.0, -11.0], [52.0, 52.0, -9.0]);
        let visibility = cpu_frustum_cull(&frustum, &[bounds_outside]);
        assert_eq!(visibility[0], 0); // Should be culled
    }

    #[test]
    fn test_camera_inside_bounds() {
        let frustum = make_test_frustum();

        // Large bounds that contain the camera origin
        let bounds = InstanceBounds::new(
            [0.0, 0.0, 0.0],   // sphere center at origin
            50.0,              // large radius
            [-50.0, -50.0, -50.0],
            [50.0, 50.0, 50.0],
        );

        // This should be visible because the camera is inside
        let visibility = cpu_frustum_cull(&frustum, &[bounds]);
        assert_eq!(visibility[0], 1);
    }

    #[test]
    fn test_cull_params_size() {
        assert_eq!(mem::size_of::<CullParams>(), 16);
    }

    #[test]
    fn test_frustum_planes_size() {
        assert_eq!(mem::size_of::<FrustumPlane>(), 16);
        assert_eq!(mem::size_of::<Frustum>(), 96);
    }

    #[test]
    fn test_instance_bounds_size() {
        assert_eq!(mem::size_of::<InstanceBounds>(), 48);
    }

    #[test]
    fn test_cpu_cull_multiple_instances() {
        let frustum = make_test_frustum();

        let instances = vec![
            // Visible: inside frustum
            InstanceBounds::new([0.0, 0.0, -10.0], 1.0, [-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]),
            // Culled: outside right
            InstanceBounds::new([100.0, 0.0, -10.0], 1.0, [99.0, -1.0, -11.0], [101.0, 1.0, -9.0]),
            // Visible: on edge
            InstanceBounds::new([0.0, 0.0, -5.0], 2.0, [-2.0, -2.0, -7.0], [2.0, 2.0, -3.0]),
            // Culled: behind camera
            InstanceBounds::new([0.0, 0.0, 5.0], 1.0, [-1.0, -1.0, 4.0], [1.0, 1.0, 6.0]),
        ];

        let visibility = cpu_frustum_cull(&frustum, &instances);
        assert_eq!(visibility, vec![1, 0, 1, 0]);
    }

    #[test]
    fn test_cull_params_num_workgroups() {
        assert_eq!(CullParams::new(1).num_workgroups(), 1);
        assert_eq!(CullParams::new(256).num_workgroups(), 1);
        assert_eq!(CullParams::new(257).num_workgroups(), 2);
        assert_eq!(CullParams::new(512).num_workgroups(), 2);
        assert_eq!(CullParams::new(1000).num_workgroups(), 4);
    }

    #[test]
    fn test_frustum_plane_normalization() {
        // Non-normalized input
        let plane = FrustumPlane::new([2.0, 0.0, 0.0], 4.0);
        // Should be normalized
        assert!((plane.normal[0] - 1.0).abs() < 1e-6);
        assert!((plane.distance - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_instance_bounds_from_aabb_with_sphere() {
        let bounds = InstanceBounds::from_aabb_with_sphere(
            [-1.0, -2.0, -3.0],
            [1.0, 2.0, 3.0],
        );

        // Center should be midpoint
        assert!((bounds.sphere_center[0] - 0.0).abs() < 1e-6);
        assert!((bounds.sphere_center[1] - 0.0).abs() < 1e-6);
        assert!((bounds.sphere_center[2] - 0.0).abs() < 1e-6);

        // Radius should be sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
        let expected_radius = (1.0_f32 + 4.0 + 9.0).sqrt();
        assert!((bounds.sphere_radius - expected_radius).abs() < 1e-6);
    }
}
