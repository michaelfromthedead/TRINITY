//! GPU Meshlet Culling for TRINITY Engine (T-GPU-4.4).
//!
//! This module provides per-meshlet culling using frustum, normal cone, and
//! optional HZB tests. It enables fine-grained visibility determination for
//! mesh shading pipelines.
//!
//! # Overview
//!
//! Meshlet culling operates on individual meshlets (small clusters of triangles)
//! rather than entire meshes. This enables:
//!
//! 1. **Frustum Culling**: Test meshlet bounding spheres against view frustum
//! 2. **Normal Cone Culling**: Early backface rejection using precomputed cones
//! 3. **HZB Occlusion Culling**: Fine-grained occlusion using hierarchical-Z
//!
//! # Dispatch Model
//!
//! - **Per-Mesh Dispatch**: One workgroup per mesh, threads process meshlets
//! - **Flat Dispatch**: One thread per meshlet (alternative layout)
//!
//! # Performance
//!
//! - Work complexity: O(meshlets)
//! - Target: < 0.15ms for 1M meshlets
//! - Memory: 32 bytes per meshlet bounds
//!
//! # Usage
//!
//! ```ignore
//! // Create pipeline and resources
//! let pipeline = MeshletCullPipeline::new(&device, &shader_source);
//! let resources = MeshletCullResources::new(&device, 100_000, 1_000);
//!
//! // Each frame: upload data and dispatch
//! resources.upload_params(&queue, &params);
//! resources.upload_meshes(&queue, &mesh_infos);
//! resources.upload_bounds(&queue, &meshlet_bounds);
//! pipeline.dispatch(&mut encoder, &resources, num_meshes);
//!
//! // Read visibility results
//! let visible = resources.read_visibility(&device, &queue);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 64;

/// Flat dispatch workgroup size for per-meshlet processing.
pub const FLAT_WORKGROUP_SIZE: u32 = 256;

/// Number of frustum planes (left, right, bottom, top, near, far).
pub const NUM_FRUSTUM_PLANES: usize = 6;

/// Maximum meshlets per mesh (workgroup size limit).
pub const MAX_MESHLETS_PER_MESH: u32 = 64;

/// Default maximum total meshlets.
pub const DEFAULT_MAX_MESHLETS: u32 = 100_000;

/// Default maximum meshes.
pub const DEFAULT_MAX_MESHES: u32 = 10_000;

// ---------------------------------------------------------------------------
// MeshletCullParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for meshlet culling parameters.
///
/// # Memory Layout
///
/// 112 bytes, std140 compatible:
/// | Offset | Field               | Size |
/// |--------|---------------------|------|
/// | 0      | num_meshes          | 4    |
/// | 4      | enable_frustum_cull | 4    |
/// | 8      | enable_cone_cull    | 4    |
/// | 12     | enable_hzb_cull     | 4    |
/// | 16     | view_proj           | 64   |
/// | 80     | camera_position     | 12   |
/// | 92     | hzb_width           | 4    |
/// | 96     | hzb_height          | 4    |
/// | 100    | num_mips            | 4    |
/// | 104    | near_plane          | 4    |
/// | 108    | far_plane           | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshletCullParams {
    /// Number of meshes to process.
    pub num_meshes: u32,
    /// Enable frustum culling (1 = enabled, 0 = disabled).
    pub enable_frustum_cull: u32,
    /// Enable normal cone culling (1 = enabled, 0 = disabled).
    pub enable_cone_cull: u32,
    /// Enable HZB occlusion culling (1 = enabled, 0 = disabled).
    pub enable_hzb_cull: u32,
    /// Combined view-projection matrix (column-major).
    pub view_proj: [[f32; 4]; 4],
    /// Camera position in world space.
    pub camera_position: [f32; 3],
    /// HZB texture width (mip 0).
    pub hzb_width: u32,
    /// HZB texture height (mip 0).
    pub hzb_height: u32,
    /// Number of mip levels in HZB texture.
    pub num_mips: u32,
    /// Near plane distance.
    pub near_plane: f32,
    /// Far plane distance.
    pub far_plane: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshletCullParams>() == 112);

impl MeshletCullParams {
    /// Create meshlet culling parameters with all culling enabled.
    pub fn new(
        num_meshes: u32,
        view_proj: &[[f32; 4]; 4],
        camera_position: [f32; 3],
    ) -> Self {
        Self {
            num_meshes,
            enable_frustum_cull: 1,
            enable_cone_cull: 1,
            enable_hzb_cull: 0, // HZB disabled by default
            view_proj: *view_proj,
            camera_position,
            hzb_width: 0,
            hzb_height: 0,
            num_mips: 0,
            near_plane: 0.1,
            far_plane: 1000.0,
        }
    }

    /// Create parameters with HZB culling enabled.
    pub fn with_hzb(
        num_meshes: u32,
        view_proj: &[[f32; 4]; 4],
        camera_position: [f32; 3],
        hzb_width: u32,
        hzb_height: u32,
        num_mips: u32,
        near_plane: f32,
        far_plane: f32,
    ) -> Self {
        Self {
            num_meshes,
            enable_frustum_cull: 1,
            enable_cone_cull: 1,
            enable_hzb_cull: 1,
            view_proj: *view_proj,
            camera_position,
            hzb_width,
            hzb_height,
            num_mips,
            near_plane,
            far_plane,
        }
    }

    /// Enable or disable frustum culling.
    pub fn set_frustum_cull(&mut self, enabled: bool) {
        self.enable_frustum_cull = if enabled { 1 } else { 0 };
    }

    /// Enable or disable cone culling.
    pub fn set_cone_cull(&mut self, enabled: bool) {
        self.enable_cone_cull = if enabled { 1 } else { 0 };
    }

    /// Enable or disable HZB culling.
    pub fn set_hzb_cull(&mut self, enabled: bool) {
        self.enable_hzb_cull = if enabled { 1 } else { 0 };
    }

    /// Get the number of workgroups needed for per-mesh dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        self.num_meshes
    }
}

// ---------------------------------------------------------------------------
// FrustumPlane
// ---------------------------------------------------------------------------

/// A frustum plane in Hessian normal form.
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

    /// Compute signed distance from a point to the plane.
    #[inline]
    pub fn distance_to_point(&self, point: [f32; 3]) -> f32 {
        self.normal[0] * point[0] + self.normal[1] * point[1] + self.normal[2] * point[2]
            + self.distance
    }
}

// ---------------------------------------------------------------------------
// MeshInfo
// ---------------------------------------------------------------------------

/// Per-mesh metadata for meshlet access.
///
/// # Memory Layout
///
/// 16 bytes, vec4 aligned:
/// | Offset | Field          | Size |
/// |--------|----------------|------|
/// | 0      | meshlet_offset | 4    |
/// | 4      | meshlet_count  | 4    |
/// | 8      | instance_id    | 4    |
/// | 12     | _pad           | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshInfo {
    /// Starting index in the global meshlet array.
    pub meshlet_offset: u32,
    /// Number of meshlets in this mesh.
    pub meshlet_count: u32,
    /// Instance ID for this mesh.
    pub instance_id: u32,
    /// Padding for alignment.
    pub _pad: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshInfo>() == 16);

impl MeshInfo {
    /// Create mesh info.
    pub fn new(meshlet_offset: u32, meshlet_count: u32, instance_id: u32) -> Self {
        Self {
            meshlet_offset,
            meshlet_count,
            instance_id,
            _pad: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// MeshletBounds
// ---------------------------------------------------------------------------

/// Bounding data for a single meshlet.
///
/// Contains a bounding sphere and normal cone for culling.
///
/// # Normal Cone
///
/// The normal cone represents the spread of normals within a meshlet:
/// - `cone_axis`: Average normal direction of the meshlet
/// - `cone_cutoff`: cos(half_angle) of the cone that contains all normals
///
/// A meshlet is backfacing if:
///   dot(normalize(camera_pos - center), cone_axis) > cone_cutoff
///
/// # Memory Layout
///
/// 32 bytes, vec4 aligned:
/// | Offset | Field       | Size |
/// |--------|-------------|------|
/// | 0      | center      | 12   |
/// | 12     | radius      | 4    |
/// | 16     | cone_axis   | 12   |
/// | 28     | cone_cutoff | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshletBounds {
    /// Center of bounding sphere in world space.
    pub center: [f32; 3],
    /// Radius of bounding sphere.
    pub radius: f32,
    /// Normal cone axis (average normal direction).
    pub cone_axis: [f32; 3],
    /// Cone cutoff value (cos of half-angle). Values > 1.0 disable cone test.
    pub cone_cutoff: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshletBounds>() == 32);

impl MeshletBounds {
    /// Create meshlet bounds with sphere and cone.
    pub fn new(
        center: [f32; 3],
        radius: f32,
        cone_axis: [f32; 3],
        cone_cutoff: f32,
    ) -> Self {
        Self {
            center,
            radius,
            cone_axis,
            cone_cutoff,
        }
    }

    /// Create bounds with only a bounding sphere (no cone culling).
    pub fn sphere_only(center: [f32; 3], radius: f32) -> Self {
        Self {
            center,
            radius,
            cone_axis: [0.0, 0.0, 1.0],
            cone_cutoff: 2.0, // > 1.0 disables cone test
        }
    }

    /// Create bounds facing along a specific axis.
    pub fn facing(center: [f32; 3], radius: f32, normal: [f32; 3], spread_angle: f32) -> Self {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let cone_axis = if len > 1e-8 {
            [normal[0] / len, normal[1] / len, normal[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        };
        Self {
            center,
            radius,
            cone_axis,
            cone_cutoff: spread_angle.cos(),
        }
    }
}

// ---------------------------------------------------------------------------
// MeshletVisibility
// ---------------------------------------------------------------------------

/// Visibility result for each meshlet.
///
/// # Memory Layout
///
/// 4 bytes:
/// | Offset | Field   | Size |
/// |--------|---------|------|
/// | 0      | visible | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshletVisibility {
    /// Visibility flag: 1 = visible, 0 = culled.
    pub visible: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshletVisibility>() == 4);

impl MeshletVisibility {
    /// Check if this meshlet is visible.
    #[inline]
    pub fn is_visible(&self) -> bool {
        self.visible != 0
    }

    /// Check if this meshlet is culled.
    #[inline]
    pub fn is_culled(&self) -> bool {
        self.visible == 0
    }
}

// ---------------------------------------------------------------------------
// MeshletCullResources
// ---------------------------------------------------------------------------

/// GPU resources for meshlet culling.
pub struct MeshletCullResources {
    /// Uniform buffer for culling parameters.
    pub params_buffer: wgpu::Buffer,
    /// Storage buffer for per-mesh info.
    pub meshes_buffer: wgpu::Buffer,
    /// Storage buffer for meshlet bounds.
    pub bounds_buffer: wgpu::Buffer,
    /// Storage buffer for visibility output.
    pub visibility_buffer: wgpu::Buffer,
    /// Storage buffer for frustum planes.
    pub planes_buffer: wgpu::Buffer,
    /// Staging buffer for reading visibility back to CPU.
    pub visibility_staging: wgpu::Buffer,
    /// Maximum number of meshlets supported.
    pub max_meshlets: u32,
    /// Maximum number of meshes supported.
    pub max_meshes: u32,
}

impl MeshletCullResources {
    /// Create meshlet culling resources.
    pub fn new(device: &wgpu::Device, max_meshlets: u32, max_meshes: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_cull_params"),
            size: mem::size_of::<MeshletCullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let meshes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_cull_meshes"),
            size: (max_meshes as u64) * (mem::size_of::<MeshInfo>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bounds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_cull_bounds"),
            size: (max_meshlets as u64) * (mem::size_of::<MeshletBounds>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_cull_visibility"),
            size: (max_meshlets as u64) * (mem::size_of::<MeshletVisibility>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let planes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_cull_planes"),
            size: (NUM_FRUSTUM_PLANES as u64) * (mem::size_of::<FrustumPlane>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visibility_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_cull_visibility_staging"),
            size: (max_meshlets as u64) * (mem::size_of::<MeshletVisibility>() as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            meshes_buffer,
            bounds_buffer,
            visibility_buffer,
            planes_buffer,
            visibility_staging,
            max_meshlets,
            max_meshes,
        }
    }

    /// Upload culling parameters to GPU.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &MeshletCullParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload mesh info array to GPU.
    pub fn upload_meshes(&self, queue: &wgpu::Queue, meshes: &[MeshInfo]) {
        assert!(meshes.len() <= self.max_meshes as usize);
        queue.write_buffer(&self.meshes_buffer, 0, bytemuck::cast_slice(meshes));
    }

    /// Upload meshlet bounds to GPU.
    pub fn upload_bounds(&self, queue: &wgpu::Queue, bounds: &[MeshletBounds]) {
        assert!(bounds.len() <= self.max_meshlets as usize);
        queue.write_buffer(&self.bounds_buffer, 0, bytemuck::cast_slice(bounds));
    }

    /// Upload frustum planes to GPU.
    pub fn upload_planes(&self, queue: &wgpu::Queue, planes: &[FrustumPlane; 6]) {
        queue.write_buffer(&self.planes_buffer, 0, bytemuck::cast_slice(planes));
    }
}

// ---------------------------------------------------------------------------
// MeshletCullPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for meshlet culling.
pub struct MeshletCullPipeline {
    /// Main culling pipeline (frustum + cone + HZB).
    pub pipeline: wgpu::ComputePipeline,
    /// Frustum-only culling pipeline.
    pub pipeline_frustum_only: wgpu::ComputePipeline,
    /// Frustum + cone culling pipeline (no HZB).
    pub pipeline_no_hzb: wgpu::ComputePipeline,
    /// Cone-only culling pipeline.
    pub pipeline_cone_only: wgpu::ComputePipeline,
    /// Flat dispatch pipeline.
    pub pipeline_flat: wgpu::ComputePipeline,
    /// Bind group layout for culling resources.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for culling with HZB.
    pub bind_group_layout_with_hzb: wgpu::BindGroupLayout,
}

impl MeshletCullPipeline {
    /// Create the meshlet culling pipeline.
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meshlet_cull_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Bind group layout without HZB
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet_cull_bind_group_layout"),
            entries: &[
                // @binding(0) params: MeshletCullParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<MeshletCullParams>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) meshes: array<MeshInfo>
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
                // @binding(2) meshlet_bounds: array<MeshletBounds>
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
                // @binding(3) visibility: array<MeshletVisibility>
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
                // @binding(4) frustum_planes: array<FrustumPlane>
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
            ],
        });

        // Bind group layout with HZB
        let bind_group_layout_with_hzb =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("meshlet_cull_bind_group_layout_with_hzb"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(
                                std::num::NonZeroU64::new(
                                    mem::size_of::<MeshletCullParams>() as u64,
                                )
                                .unwrap(),
                            ),
                        },
                        count: None,
                    },
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
                    // @binding(5) hzb_texture: texture_2d<f32>
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meshlet_cull_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_layout_with_hzb =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("meshlet_cull_pipeline_layout_with_hzb"),
                bind_group_layouts: &[&bind_group_layout_with_hzb],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("meshlet_cull_pipeline"),
            layout: Some(&pipeline_layout_with_hzb),
            module: &shader_module,
            entry_point: "cull_meshlet",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_frustum_only =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("meshlet_cull_pipeline_frustum_only"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "cull_meshlet_frustum_only",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let pipeline_no_hzb = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("meshlet_cull_pipeline_no_hzb"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_meshlet_no_hzb",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_cone_only = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("meshlet_cull_pipeline_cone_only"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_meshlet_cone_only",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_flat = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("meshlet_cull_pipeline_flat"),
            layout: Some(&pipeline_layout_with_hzb),
            module: &shader_module,
            entry_point: "cull_meshlet_flat",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            pipeline_frustum_only,
            pipeline_no_hzb,
            pipeline_cone_only,
            pipeline_flat,
            bind_group_layout,
            bind_group_layout_with_hzb,
        }
    }

    /// Create a bind group for the given resources (without HZB).
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &MeshletCullResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet_cull_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.meshes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.planes_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create a bind group for the given resources with HZB texture.
    pub fn create_bind_group_with_hzb(
        &self,
        device: &wgpu::Device,
        resources: &MeshletCullResources,
        hzb_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("meshlet_cull_bind_group_with_hzb"),
            layout: &self.bind_group_layout_with_hzb,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.meshes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.planes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(hzb_view),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation of frustum culling for a bounding sphere.
pub fn cpu_frustum_cull_sphere(
    center: [f32; 3],
    radius: f32,
    planes: &[FrustumPlane; 6],
) -> bool {
    for plane in planes {
        let dist = plane.distance_to_point(center);
        if dist < -radius {
            return false; // Sphere is entirely outside this plane
        }
    }
    true
}

/// CPU reference implementation of normal cone culling.
///
/// The cone_axis represents the average normal direction of the meshlet.
/// The cone_cutoff is cos(half_angle) where half_angle is the spread of normals.
///
/// A meshlet is backfacing (should be culled) when the view direction is mostly
/// opposite to the cone_axis, specifically when:
///   dot(view_direction, cone_axis) < -cone_cutoff
///
/// This means the camera is "behind" the meshlet relative to its normal direction.
///
/// Returns `true` if the meshlet should be CULLED (is backfacing).
pub fn cpu_cone_cull(
    center: [f32; 3],
    cone_axis: [f32; 3],
    cone_cutoff: f32,
    camera_pos: [f32; 3],
) -> bool {
    // Skip cone test if cutoff is invalid (>= 1.0 means disabled)
    if cone_cutoff >= 1.0 {
        return false; // Don't cull
    }

    // Compute view direction from meshlet center to camera
    let to_camera = [
        camera_pos[0] - center[0],
        camera_pos[1] - center[1],
        camera_pos[2] - center[2],
    ];

    let dist_sq = to_camera[0] * to_camera[0]
        + to_camera[1] * to_camera[1]
        + to_camera[2] * to_camera[2];

    // Camera at meshlet center - can't determine facing, don't cull
    if dist_sq < 1e-8 {
        return false;
    }

    let inv_dist = 1.0 / dist_sq.sqrt();
    let view_dir = [
        to_camera[0] * inv_dist,
        to_camera[1] * inv_dist,
        to_camera[2] * inv_dist,
    ];

    // Dot product of view direction (meshlet->camera) and cone axis (normal direction)
    let cone_dot =
        view_dir[0] * cone_axis[0] + view_dir[1] * cone_axis[1] + view_dir[2] * cone_axis[2];

    // If dot < -cutoff, the view direction is opposite to the normal cone,
    // meaning the meshlet is backfacing and should be culled.
    // For a tight cone (cutoff = cos(small_angle) near 1.0), this requires
    // the view to be almost directly opposite to the normals.
    // For a wide cone (cutoff = cos(large_angle) near 0.0), even side views get culled.
    cone_dot < -cone_cutoff
}

/// CPU reference implementation of meshlet culling.
pub fn cpu_meshlet_cull(
    bounds: &[MeshletBounds],
    meshes: &[MeshInfo],
    planes: &[FrustumPlane; 6],
    camera_pos: [f32; 3],
    enable_frustum: bool,
    enable_cone: bool,
) -> Vec<MeshletVisibility> {
    let mut visibility = vec![MeshletVisibility { visible: 0 }; bounds.len()];

    for mesh in meshes {
        for i in 0..mesh.meshlet_count {
            let global_idx = (mesh.meshlet_offset + i) as usize;
            if global_idx >= bounds.len() {
                continue;
            }

            let b = &bounds[global_idx];
            let mut visible = true;

            // Frustum culling
            if enable_frustum && visible {
                visible = cpu_frustum_cull_sphere(b.center, b.radius, planes);
            }

            // Cone culling
            if enable_cone && visible {
                let culled = cpu_cone_cull(b.center, b.cone_axis, b.cone_cutoff, camera_pos);
                visible = !culled;
            }

            visibility[global_idx].visible = if visible { 1 } else { 0 };
        }
    }

    visibility
}

/// CPU reference implementation of flat meshlet culling.
pub fn cpu_meshlet_cull_flat(
    bounds: &[MeshletBounds],
    planes: &[FrustumPlane; 6],
    camera_pos: [f32; 3],
    enable_frustum: bool,
    enable_cone: bool,
) -> Vec<MeshletVisibility> {
    bounds
        .iter()
        .map(|b| {
            // Skip invalid entries
            if b.radius <= 0.0 {
                return MeshletVisibility { visible: 0 };
            }

            let mut visible = true;

            // Frustum culling
            if enable_frustum && visible {
                visible = cpu_frustum_cull_sphere(b.center, b.radius, planes);
            }

            // Cone culling
            if enable_cone && visible {
                let culled = cpu_cone_cull(b.center, b.cone_axis, b.cone_cutoff, camera_pos);
                visible = !culled;
            }

            MeshletVisibility {
                visible: if visible { 1 } else { 0 },
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
    use std::f32::consts::PI;

    /// Helper: Create a test frustum (camera at origin, looking down -Z, 90-degree FOV).
    fn make_test_frustum() -> [FrustumPlane; 6] {
        [
            // Left: normal points right (+X) at 45 degrees
            FrustumPlane::new([1.0, 0.0, -1.0], 0.0),
            // Right: normal points left (-X) at 45 degrees
            FrustumPlane::new([-1.0, 0.0, -1.0], 0.0),
            // Bottom: normal points up (+Y) at 45 degrees
            FrustumPlane::new([0.0, 1.0, -1.0], 0.0),
            // Top: normal points down (-Y) at 45 degrees
            FrustumPlane::new([0.0, -1.0, -1.0], 0.0),
            // Near: normal points forward (-Z), at z = -1
            FrustumPlane::new([0.0, 0.0, -1.0], -1.0),
            // Far: normal points backward (+Z), at z = -100
            FrustumPlane::new([0.0, 0.0, 1.0], 100.0),
        ]
    }

    // -------------------------------------------------------------------------
    // Struct Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_meshlet_cull_params_size() {
        assert_eq!(
            mem::size_of::<MeshletCullParams>(),
            112,
            "MeshletCullParams must be 112 bytes"
        );
    }

    #[test]
    fn test_mesh_info_size() {
        assert_eq!(
            mem::size_of::<MeshInfo>(),
            16,
            "MeshInfo must be 16 bytes"
        );
    }

    #[test]
    fn test_meshlet_bounds_size() {
        assert_eq!(
            mem::size_of::<MeshletBounds>(),
            32,
            "MeshletBounds must be 32 bytes"
        );
    }

    #[test]
    fn test_meshlet_visibility_size() {
        assert_eq!(
            mem::size_of::<MeshletVisibility>(),
            4,
            "MeshletVisibility must be 4 bytes"
        );
    }

    #[test]
    fn test_frustum_plane_size() {
        assert_eq!(
            mem::size_of::<FrustumPlane>(),
            16,
            "FrustumPlane must be 16 bytes"
        );
    }

    // -------------------------------------------------------------------------
    // Frustum Culling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_meshlet_inside_frustum_visible() {
        let planes = make_test_frustum();

        // Meshlet inside frustum at z=-10
        let bounds = MeshletBounds::sphere_only([0.0, 0.0, -10.0], 1.0);
        let visible = cpu_frustum_cull_sphere(bounds.center, bounds.radius, &planes);

        assert!(visible, "Meshlet inside frustum should be visible");
    }

    #[test]
    fn test_meshlet_outside_frustum_culled() {
        let planes = make_test_frustum();

        // Meshlet way outside to the right
        let bounds = MeshletBounds::sphere_only([100.0, 0.0, -10.0], 1.0);
        let visible = cpu_frustum_cull_sphere(bounds.center, bounds.radius, &planes);

        assert!(!visible, "Meshlet outside frustum should be culled");
    }

    #[test]
    fn test_meshlet_behind_camera_culled() {
        let planes = make_test_frustum();

        // Meshlet behind camera (positive Z)
        let bounds = MeshletBounds::sphere_only([0.0, 0.0, 10.0], 1.0);
        let visible = cpu_frustum_cull_sphere(bounds.center, bounds.radius, &planes);

        assert!(!visible, "Meshlet behind camera should be culled");
    }

    #[test]
    fn test_meshlet_beyond_far_culled() {
        let planes = make_test_frustum();

        // Meshlet beyond far plane
        let bounds = MeshletBounds::sphere_only([0.0, 0.0, -200.0], 1.0);
        let visible = cpu_frustum_cull_sphere(bounds.center, bounds.radius, &planes);

        assert!(!visible, "Meshlet beyond far plane should be culled");
    }

    #[test]
    fn test_meshlet_intersecting_frustum_visible() {
        let planes = make_test_frustum();

        // Meshlet partially inside frustum (sphere intersects near plane)
        let bounds = MeshletBounds::sphere_only([0.0, 0.0, -2.0], 1.5);
        let visible = cpu_frustum_cull_sphere(bounds.center, bounds.radius, &planes);

        assert!(visible, "Meshlet intersecting frustum should be visible");
    }

    // -------------------------------------------------------------------------
    // Normal Cone Culling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_backfacing_cone_culled() {
        let camera_pos = [0.0, 0.0, 0.0];

        // Meshlet facing away from camera (cone axis points away)
        let bounds = MeshletBounds::new(
            [0.0, 0.0, -10.0],  // center
            1.0,                // radius
            [0.0, 0.0, -1.0],   // cone_axis: pointing away from camera
            0.0,                // cone_cutoff: cos(90deg) = 0, tight cone
        );

        let culled = cpu_cone_cull(bounds.center, bounds.cone_axis, bounds.cone_cutoff, camera_pos);

        assert!(culled, "Backfacing meshlet should be culled");
    }

    #[test]
    fn test_frontfacing_cone_visible() {
        let camera_pos = [0.0, 0.0, 0.0];

        // Meshlet facing toward camera (cone axis points toward camera)
        let bounds = MeshletBounds::new(
            [0.0, 0.0, -10.0],  // center
            1.0,                // radius
            [0.0, 0.0, 1.0],    // cone_axis: pointing toward camera
            0.0,                // cone_cutoff
        );

        let culled = cpu_cone_cull(bounds.center, bounds.cone_axis, bounds.cone_cutoff, camera_pos);

        assert!(!culled, "Front-facing meshlet should not be culled");
    }

    #[test]
    fn test_sideways_cone_visible() {
        let camera_pos = [0.0, 0.0, 0.0];

        // Meshlet facing sideways (perpendicular to view)
        let bounds = MeshletBounds::new(
            [0.0, 0.0, -10.0],  // center
            1.0,                // radius
            [1.0, 0.0, 0.0],    // cone_axis: pointing sideways
            0.0,                // cone_cutoff
        );

        let culled = cpu_cone_cull(bounds.center, bounds.cone_axis, bounds.cone_cutoff, camera_pos);

        assert!(!culled, "Sideways meshlet should not be culled (dot=0, cutoff=0)");
    }

    #[test]
    fn test_wide_cone_visible() {
        let camera_pos = [0.0, 0.0, 0.0];

        // Wide cone with normals mostly facing away, but with 60-degree spread
        // This means some normals could face sideways, making parts potentially visible.
        // cutoff = cos(60 deg) = 0.5
        // A meshlet at [0, 0, -10] with normals averaging [0, 0, -1] but spread 60 degrees
        // means some normals point at angles like [sin(60), 0, -cos(60)] = [0.866, 0, -0.5]
        //
        // View direction from meshlet to camera: [0, 0, 1]
        // dot([0, 0, 1], [0, 0, -1]) = -1
        // Culling condition: dot < -cutoff => -1 < -0.5 => TRUE, culled
        //
        // Actually, let's use a case where it's NOT culled:
        // Meshlet at [10, 0, -10], camera at origin.
        // View direction: [-10, 0, 10] normalized = [-0.707, 0, 0.707]
        // Cone axis pointing mostly away: [0, 0, -1]
        // dot([-0.707, 0, 0.707], [0, 0, -1]) = -0.707
        // With cutoff = 0.5: -0.707 < -0.5 => TRUE, still culled
        //
        // Let's use a tighter check: make the cone WIDER (smaller cutoff)
        // If cutoff = 0.2 (cos(~78 degrees)):
        // -0.707 < -0.2 => TRUE, still culled
        //
        // The issue is our backfacing meshlet IS backfacing no matter the cone width.
        // Let's test a case where normals point forward but with spread:
        let bounds = MeshletBounds::new(
            [0.0, 0.0, -10.0],
            1.0,
            [0.0, 0.0, 1.0],    // cone_axis: pointing TOWARD camera (front-facing)
            0.1,                // cutoff = cos(~84 deg), wide spread of normals
        );

        let culled = cpu_cone_cull(bounds.center, bounds.cone_axis, bounds.cone_cutoff, camera_pos);

        // View direction: [0, 0, 1], cone_axis: [0, 0, 1]
        // dot = 1, cutoff = 0.1
        // 1 < -0.1 => FALSE, not culled
        assert!(!culled, "Front-facing wide cone should not be culled");
    }

    #[test]
    fn test_disabled_cone_not_culled() {
        let camera_pos = [0.0, 0.0, 0.0];

        // Disabled cone (cutoff >= 1.0)
        let bounds = MeshletBounds::sphere_only([0.0, 0.0, -10.0], 1.0);
        // sphere_only sets cone_cutoff to 2.0

        let culled = cpu_cone_cull(bounds.center, bounds.cone_axis, bounds.cone_cutoff, camera_pos);

        assert!(!culled, "Disabled cone should not be culled");
    }

    // -------------------------------------------------------------------------
    // Combined Culling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multiple_meshes_mixed_visibility() {
        let planes = make_test_frustum();
        let camera_pos = [0.0, 0.0, 0.0];

        // Create multiple meshes with different meshlet configurations
        let meshes = vec![
            MeshInfo::new(0, 2, 0),  // Mesh 0: meshlets 0-1
            MeshInfo::new(2, 2, 1),  // Mesh 1: meshlets 2-3
        ];

        let bounds = vec![
            // Mesh 0, meshlet 0: visible (inside frustum, front-facing)
            MeshletBounds::new([0.0, 0.0, -10.0], 1.0, [0.0, 0.0, 1.0], 0.0),
            // Mesh 0, meshlet 1: culled (outside frustum)
            MeshletBounds::new([100.0, 0.0, -10.0], 1.0, [0.0, 0.0, 1.0], 0.0),
            // Mesh 1, meshlet 0: culled (backfacing)
            MeshletBounds::new([0.0, 0.0, -15.0], 1.0, [0.0, 0.0, -1.0], 0.0),
            // Mesh 1, meshlet 1: visible (inside frustum, front-facing)
            MeshletBounds::new([0.0, 0.0, -20.0], 1.0, [0.0, 0.0, 1.0], 0.0),
        ];

        let visibility = cpu_meshlet_cull(&bounds, &meshes, &planes, camera_pos, true, true);

        assert_eq!(visibility[0].visible, 1, "Meshlet 0 should be visible");
        assert_eq!(visibility[1].visible, 0, "Meshlet 1 should be culled (frustum)");
        assert_eq!(visibility[2].visible, 0, "Meshlet 2 should be culled (cone)");
        assert_eq!(visibility[3].visible, 1, "Meshlet 3 should be visible");
    }

    #[test]
    fn test_frustum_only_ignores_cone() {
        let planes = make_test_frustum();
        let camera_pos = [0.0, 0.0, 0.0];

        let meshes = vec![MeshInfo::new(0, 1, 0)];

        // Backfacing meshlet inside frustum
        let bounds = vec![MeshletBounds::new(
            [0.0, 0.0, -10.0],
            1.0,
            [0.0, 0.0, -1.0],
            0.0,
        )];

        // Frustum only (cone disabled)
        let visibility = cpu_meshlet_cull(&bounds, &meshes, &planes, camera_pos, true, false);

        assert_eq!(
            visibility[0].visible, 1,
            "Backfacing meshlet should be visible with frustum-only culling"
        );
    }

    #[test]
    fn test_cone_only_ignores_frustum() {
        let planes = make_test_frustum();
        let camera_pos = [0.0, 0.0, 0.0];

        let meshes = vec![MeshInfo::new(0, 1, 0)];

        // Front-facing meshlet outside frustum
        let bounds = vec![MeshletBounds::new(
            [100.0, 0.0, -10.0],
            1.0,
            [0.0, 0.0, 1.0],
            0.0,
        )];

        // Cone only (frustum disabled)
        let visibility = cpu_meshlet_cull(&bounds, &meshes, &planes, camera_pos, false, true);

        assert_eq!(
            visibility[0].visible, 1,
            "Meshlet outside frustum should be visible with cone-only culling"
        );
    }

    // -------------------------------------------------------------------------
    // Flat Dispatch Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_flat_cull_skips_invalid() {
        let planes = make_test_frustum();
        let camera_pos = [0.0, 0.0, 0.0];

        let bounds = vec![
            MeshletBounds::sphere_only([0.0, 0.0, -10.0], 1.0),  // Valid
            MeshletBounds::sphere_only([0.0, 0.0, -10.0], 0.0),  // Invalid (radius = 0)
            MeshletBounds::sphere_only([0.0, 0.0, -10.0], -1.0), // Invalid (radius < 0)
        ];

        let visibility = cpu_meshlet_cull_flat(&bounds, &planes, camera_pos, true, true);

        assert_eq!(visibility[0].visible, 1, "Valid meshlet should be visible");
        assert_eq!(visibility[1].visible, 0, "Zero radius meshlet should be culled");
        assert_eq!(visibility[2].visible, 0, "Negative radius meshlet should be culled");
    }

    // -------------------------------------------------------------------------
    // Enable/Disable Flags Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_culling_disabled() {
        let planes = make_test_frustum();
        let camera_pos = [0.0, 0.0, 0.0];

        let meshes = vec![MeshInfo::new(0, 2, 0)];

        let bounds = vec![
            // Outside frustum and backfacing
            MeshletBounds::new([100.0, 0.0, -10.0], 1.0, [0.0, 0.0, -1.0], 0.0),
            // Behind camera and backfacing
            MeshletBounds::new([0.0, 0.0, 10.0], 1.0, [0.0, 0.0, 1.0], 0.0),
        ];

        // All culling disabled
        let visibility = cpu_meshlet_cull(&bounds, &meshes, &planes, camera_pos, false, false);

        assert_eq!(visibility[0].visible, 1, "Should be visible with all culling disabled");
        assert_eq!(visibility[1].visible, 1, "Should be visible with all culling disabled");
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_mesh_list() {
        let planes = make_test_frustum();
        let camera_pos = [0.0, 0.0, 0.0];

        let meshes: Vec<MeshInfo> = vec![];
        let bounds: Vec<MeshletBounds> = vec![];

        let visibility = cpu_meshlet_cull(&bounds, &meshes, &planes, camera_pos, true, true);

        assert!(visibility.is_empty(), "Empty input should produce empty output");
    }

    #[test]
    fn test_mesh_with_zero_meshlets() {
        let planes = make_test_frustum();
        let camera_pos = [0.0, 0.0, 0.0];

        let meshes = vec![MeshInfo::new(0, 0, 0)]; // Zero meshlets
        let bounds: Vec<MeshletBounds> = vec![];

        let visibility = cpu_meshlet_cull(&bounds, &meshes, &planes, camera_pos, true, true);

        assert!(visibility.is_empty(), "Mesh with zero meshlets should produce empty visibility");
    }

    #[test]
    fn test_camera_at_meshlet_center() {
        let planes = make_test_frustum();

        // Camera at meshlet center
        let bounds = MeshletBounds::new([5.0, 5.0, 5.0], 1.0, [0.0, 0.0, -1.0], 0.0);
        let camera_pos = [5.0, 5.0, 5.0];

        let culled = cpu_cone_cull(bounds.center, bounds.cone_axis, bounds.cone_cutoff, camera_pos);

        assert!(!culled, "Camera at meshlet center should not trigger cone cull");
    }

    #[test]
    fn test_large_meshlet_count() {
        let planes = make_test_frustum();
        let camera_pos = [0.0, 0.0, 0.0];

        // Create 1000 meshlets
        let meshes = vec![MeshInfo::new(0, 1000, 0)];
        let bounds: Vec<_> = (0..1000)
            .map(|i| {
                let z = -5.0 - (i as f32 * 0.1);
                MeshletBounds::sphere_only([0.0, 0.0, z], 0.5)
            })
            .collect();

        let visibility = cpu_meshlet_cull(&bounds, &meshes, &planes, camera_pos, true, false);

        assert_eq!(visibility.len(), 1000, "Should process all 1000 meshlets");

        // Count visible
        let visible_count = visibility.iter().filter(|v| v.is_visible()).count();
        assert!(visible_count > 0, "Some meshlets should be visible");
    }

    // -------------------------------------------------------------------------
    // MeshletBounds Helper Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_meshlet_bounds_facing() {
        let spread_angle = PI / 4.0; // 45 degrees
        let bounds = MeshletBounds::facing([0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0], spread_angle);

        assert!((bounds.cone_axis[1] - 1.0).abs() < 1e-6, "Cone axis should be normalized");
        assert!(
            (bounds.cone_cutoff - (PI / 4.0).cos()).abs() < 1e-6,
            "Cutoff should be cos of spread angle"
        );
    }

    #[test]
    fn test_meshlet_bounds_sphere_only() {
        let bounds = MeshletBounds::sphere_only([1.0, 2.0, 3.0], 5.0);

        assert_eq!(bounds.center, [1.0, 2.0, 3.0]);
        assert_eq!(bounds.radius, 5.0);
        assert!(bounds.cone_cutoff > 1.0, "cone_cutoff should be > 1.0 to disable cone test");
    }

    // -------------------------------------------------------------------------
    // MeshletCullParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_params_default_flags() {
        let vp = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let params = MeshletCullParams::new(10, &vp, [0.0, 0.0, 0.0]);

        assert_eq!(params.num_meshes, 10);
        assert_eq!(params.enable_frustum_cull, 1, "Frustum cull should be enabled by default");
        assert_eq!(params.enable_cone_cull, 1, "Cone cull should be enabled by default");
        assert_eq!(params.enable_hzb_cull, 0, "HZB cull should be disabled by default");
    }

    #[test]
    fn test_params_with_hzb() {
        let vp = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let params = MeshletCullParams::with_hzb(10, &vp, [0.0, 0.0, 0.0], 1920, 1080, 10, 0.1, 1000.0);

        assert_eq!(params.enable_hzb_cull, 1, "HZB cull should be enabled");
        assert_eq!(params.hzb_width, 1920);
        assert_eq!(params.hzb_height, 1080);
        assert_eq!(params.num_mips, 10);
    }

    #[test]
    fn test_params_set_flags() {
        let vp = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let mut params = MeshletCullParams::new(10, &vp, [0.0, 0.0, 0.0]);

        params.set_frustum_cull(false);
        params.set_cone_cull(false);
        params.set_hzb_cull(true);

        assert_eq!(params.enable_frustum_cull, 0);
        assert_eq!(params.enable_cone_cull, 0);
        assert_eq!(params.enable_hzb_cull, 1);
    }

    #[test]
    fn test_params_num_workgroups() {
        let vp = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

        let params = MeshletCullParams::new(10, &vp, [0.0, 0.0, 0.0]);
        assert_eq!(params.num_workgroups(), 10);

        let params = MeshletCullParams::new(0, &vp, [0.0, 0.0, 0.0]);
        assert_eq!(params.num_workgroups(), 0);
    }

    // -------------------------------------------------------------------------
    // MeshInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mesh_info_creation() {
        let info = MeshInfo::new(100, 50, 5);

        assert_eq!(info.meshlet_offset, 100);
        assert_eq!(info.meshlet_count, 50);
        assert_eq!(info.instance_id, 5);
        assert_eq!(info._pad, 0);
    }

    // -------------------------------------------------------------------------
    // MeshletVisibility Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_visibility_helpers() {
        let visible = MeshletVisibility { visible: 1 };
        assert!(visible.is_visible());
        assert!(!visible.is_culled());

        let culled = MeshletVisibility { visible: 0 };
        assert!(!culled.is_visible());
        assert!(culled.is_culled());
    }
}
