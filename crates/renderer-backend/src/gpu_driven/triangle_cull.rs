//! GPU Triangle Culling for TRINITY Engine (T-GPU-3.7).
//!
//! Per-triangle culling in clip space: backface rejection, degenerate detection,
//! and frustum culling. Operates after instance-level frustum culling to reject
//! individual triangles before rasterization.
//!
//! # Overview
//!
//! Triangle culling eliminates invisible triangles at the GPU level:
//!
//! 1. **Backface Culling**: Reject triangles facing away from camera (configurable winding)
//! 2. **Degenerate Detection**: Reject zero-area or micro triangles
//! 3. **Frustum Culling**: Reject triangles entirely outside clip volume
//!
//! # Performance
//!
//! - Work complexity: O(n), one thread per triangle
//! - Target: < 0.1ms for 1M triangles
//! - Memory: 64 bytes per input triangle, 8 bytes per result
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{TriangleCullPipeline, TriangleCullResources};
//!
//! // Create pipeline and resources
//! let pipeline = TriangleCullPipeline::new(&device);
//! let resources = TriangleCullResources::new(&device, 1_000_000);
//!
//! // Each frame: upload triangles and cull
//! resources.upload_params(&queue, &params);
//! resources.upload_triangles(&queue, &triangles);
//! pipeline.dispatch(&mut encoder, &bind_group, &params);
//!
//! // Read visibility results
//! let results = resources.read_results(&device, &queue, triangle_count);
//! ```

use std::mem;

use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferUsages, Device, Queue};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Default degenerate triangle area threshold (in screen pixels squared).
pub const DEFAULT_DEGENERATE_THRESHOLD: f32 = 1e-6;

/// Cull reason: Triangle is visible (not culled).
pub const CULL_REASON_NONE: u32 = 0;

/// Cull reason: Triangle is backfacing.
pub const CULL_REASON_BACKFACE: u32 = 1;

/// Cull reason: Triangle is degenerate (zero/micro area).
pub const CULL_REASON_DEGENERATE: u32 = 2;

/// Cull reason: Triangle is outside frustum.
pub const CULL_REASON_FRUSTUM: u32 = 3;

// =============================================================================
// CULL MODE
// =============================================================================

/// Backface culling mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum CullMode {
    /// No backface culling - render both sides.
    #[default]
    None = 0,
    /// Cull counter-clockwise faces (front = clockwise).
    CounterClockwise = 1,
    /// Cull clockwise faces (front = counter-clockwise).
    Clockwise = 2,
}

impl CullMode {
    /// Convert to u32 for GPU uniform.
    #[inline]
    pub const fn to_u32(self) -> u32 {
        self as u32
    }

    /// Create from u32 value.
    #[inline]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::CounterClockwise,
            2 => Self::Clockwise,
            _ => Self::None,
        }
    }
}

// =============================================================================
// CULL REASON
// =============================================================================

/// Reason a triangle was culled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum CullReason {
    /// Triangle is visible (not culled).
    #[default]
    None = 0,
    /// Triangle is backfacing.
    Backface = 1,
    /// Triangle is degenerate (zero or micro area).
    Degenerate = 2,
    /// Triangle is entirely outside frustum.
    Frustum = 3,
}

impl CullReason {
    /// Convert to u32 for GPU uniform.
    #[inline]
    pub const fn to_u32(self) -> u32 {
        self as u32
    }

    /// Create from u32 value.
    #[inline]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Backface,
            2 => Self::Degenerate,
            3 => Self::Frustum,
            _ => Self::None,
        }
    }

    /// Check if this reason indicates the triangle was culled.
    #[inline]
    pub const fn is_culled(self) -> bool {
        !matches!(self, Self::None)
    }
}

// =============================================================================
// TRIANGLE CULL PARAMS
// =============================================================================

/// GPU uniform buffer for triangle culling parameters.
///
/// # Memory Layout
///
/// 96 bytes total (params + 4x4 matrix):
/// | Offset | Field               | Size |
/// |--------|---------------------|------|
/// | 0      | num_triangles       | 4    |
/// | 4      | cull_backface       | 4    |
/// | 8      | degenerate_threshold| 4    |
/// | 12     | viewport_width      | 4    |
/// | 16     | viewport_height     | 4    |
/// | 20     | _pad0               | 4    |
/// | 24     | _pad1               | 4    |
/// | 28     | _pad2               | 4    |
/// | 32     | view_proj           | 64   |
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TriangleCullParams {
    /// Number of triangles to process.
    pub num_triangles: u32,
    /// Backface culling mode (0 = none, 1 = CCW, 2 = CW).
    pub cull_backface: u32,
    /// Area threshold for degenerate triangle detection.
    pub degenerate_threshold: f32,
    /// Viewport width in pixels.
    pub viewport_width: f32,
    /// Viewport height in pixels.
    pub viewport_height: f32,
    /// Padding for alignment.
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    /// View-projection matrix (column-major, 4x4).
    pub view_proj: [[f32; 4]; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<TriangleCullParams>() == 96);

impl Default for TriangleCullParams {
    fn default() -> Self {
        Self {
            num_triangles: 0,
            cull_backface: CullMode::CounterClockwise.to_u32(),
            degenerate_threshold: DEFAULT_DEGENERATE_THRESHOLD,
            viewport_width: 1920.0,
            viewport_height: 1080.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

impl TriangleCullParams {
    /// Create parameters for the given triangle count.
    pub fn new(num_triangles: u32, view_proj: [[f32; 4]; 4]) -> Self {
        Self {
            num_triangles,
            view_proj,
            ..Default::default()
        }
    }

    /// Builder: set backface cull mode.
    #[inline]
    pub fn with_cull_mode(mut self, mode: CullMode) -> Self {
        self.cull_backface = mode.to_u32();
        self
    }

    /// Builder: set degenerate threshold.
    #[inline]
    pub fn with_degenerate_threshold(mut self, threshold: f32) -> Self {
        self.degenerate_threshold = threshold;
        self
    }

    /// Builder: set viewport dimensions.
    #[inline]
    pub fn with_viewport(mut self, width: f32, height: f32) -> Self {
        self.viewport_width = width;
        self.viewport_height = height;
        self
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_triangles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Get the cull mode.
    #[inline]
    pub fn cull_mode(&self) -> CullMode {
        CullMode::from_u32(self.cull_backface)
    }
}

// =============================================================================
// TRIANGLE INPUT
// =============================================================================

/// Input triangle with 3 vertices and identification.
///
/// Vertices are in world space; the shader transforms to clip space.
///
/// # Memory Layout
///
/// 64 bytes:
/// | Offset | Field        | Size |
/// |--------|--------------|------|
/// | 0      | v0           | 12   |
/// | 12     | _pad0        | 4    |
/// | 16     | v1           | 12   |
/// | 28     | _pad1        | 4    |
/// | 32     | v2           | 12   |
/// | 44     | instance_id  | 4    |
/// | 48     | primitive_id | 4    |
/// | 52     | _pad2        | 12   |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TriangleInput {
    /// Vertex 0 position (world space).
    pub v0: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad0: f32,
    /// Vertex 1 position (world space).
    pub v1: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad1: f32,
    /// Vertex 2 position (world space).
    pub v2: [f32; 3],
    /// Instance ID this triangle belongs to.
    pub instance_id: u32,
    /// Primitive ID within the instance.
    pub primitive_id: u32,
    /// Padding for struct alignment.
    pub _pad2: [f32; 3],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<TriangleInput>() == 64);

impl TriangleInput {
    /// Create a new triangle input.
    pub const fn new(
        v0: [f32; 3],
        v1: [f32; 3],
        v2: [f32; 3],
        instance_id: u32,
        primitive_id: u32,
    ) -> Self {
        Self {
            v0,
            _pad0: 0.0,
            v1,
            _pad1: 0.0,
            v2,
            instance_id,
            primitive_id,
            _pad2: [0.0, 0.0, 0.0],
        }
    }

    /// Create from array of vertices.
    pub const fn from_vertices(vertices: [[f32; 3]; 3], instance_id: u32, primitive_id: u32) -> Self {
        Self::new(vertices[0], vertices[1], vertices[2], instance_id, primitive_id)
    }

    /// Compute the geometric normal (unnormalized).
    pub fn compute_normal(&self) -> [f32; 3] {
        let e1 = [
            self.v1[0] - self.v0[0],
            self.v1[1] - self.v0[1],
            self.v1[2] - self.v0[2],
        ];
        let e2 = [
            self.v2[0] - self.v0[0],
            self.v2[1] - self.v0[1],
            self.v2[2] - self.v0[2],
        ];
        // Cross product: e1 x e2
        [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ]
    }

    /// Compute the triangle area.
    pub fn compute_area(&self) -> f32 {
        let normal = self.compute_normal();
        let len_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        len_sq.sqrt() * 0.5
    }
}

// =============================================================================
// CULL RESULT
// =============================================================================

/// Culling result for a single triangle.
///
/// # Memory Layout
///
/// 8 bytes:
/// | Offset | Field       | Size |
/// |--------|-------------|------|
/// | 0      | visible     | 4    |
/// | 4      | cull_reason | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct CullResult {
    /// Visibility flag: 0 = culled, 1 = visible.
    pub visible: u32,
    /// Cull reason: 0 = none, 1 = backface, 2 = degenerate, 3 = frustum.
    pub cull_reason: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<CullResult>() == 8);

impl CullResult {
    /// Create a visible result.
    pub const fn visible() -> Self {
        Self {
            visible: 1,
            cull_reason: CULL_REASON_NONE,
        }
    }

    /// Create a culled result with reason.
    pub const fn culled(reason: CullReason) -> Self {
        Self {
            visible: 0,
            cull_reason: reason.to_u32(),
        }
    }

    /// Check if triangle is visible.
    #[inline]
    pub const fn is_visible(&self) -> bool {
        self.visible != 0
    }

    /// Get the cull reason.
    #[inline]
    pub fn reason(&self) -> CullReason {
        CullReason::from_u32(self.cull_reason)
    }
}

// =============================================================================
// TRIANGLE CULL RESOURCES
// =============================================================================

/// GPU resources for triangle culling.
pub struct TriangleCullResources {
    /// Uniform buffer for parameters.
    pub params_buffer: Buffer,
    /// Storage buffer for input triangles.
    pub triangles_buffer: Buffer,
    /// Storage buffer for cull results.
    pub results_buffer: Buffer,
    /// Staging buffer for reading results.
    pub results_staging: Buffer,
    /// Maximum triangles supported.
    pub capacity: u32,
}

impl TriangleCullResources {
    /// Create resources for the given triangle capacity.
    pub fn new(device: &Device, capacity: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("triangle_cull_params"),
            size: mem::size_of::<TriangleCullParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("triangle_cull_triangles"),
            size: (capacity as u64) * (mem::size_of::<TriangleInput>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("triangle_cull_results"),
            size: (capacity as u64) * (mem::size_of::<CullResult>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let results_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("triangle_cull_results_staging"),
            size: (capacity as u64) * (mem::size_of::<CullResult>() as u64),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            triangles_buffer,
            results_buffer,
            results_staging,
            capacity,
        }
    }

    /// Upload culling parameters to GPU.
    pub fn upload_params(&self, queue: &Queue, params: &TriangleCullParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload triangles to GPU.
    ///
    /// # Panics
    ///
    /// Panics if `triangles.len() > self.capacity`.
    pub fn upload_triangles(&self, queue: &Queue, triangles: &[TriangleInput]) {
        assert!(triangles.len() <= self.capacity as usize);
        queue.write_buffer(&self.triangles_buffer, 0, bytemuck::cast_slice(triangles));
    }

    /// Read cull results back from GPU.
    ///
    /// This is a synchronous operation that waits for GPU completion.
    pub fn read_results(&self, device: &Device, queue: &Queue, count: u32) -> Vec<CullResult> {
        let count = count.min(self.capacity) as usize;
        let byte_size = count * mem::size_of::<CullResult>();

        // Copy from GPU buffer to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("triangle_cull_read_results"),
        });
        encoder.copy_buffer_to_buffer(
            &self.results_buffer,
            0,
            &self.results_staging,
            0,
            byte_size as u64,
        );
        queue.submit([encoder.finish()]);

        // Map and read
        let buffer_slice = self.results_staging.slice(..byte_size as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let results: Vec<CullResult> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.results_staging.unmap();

        results
    }
}

// =============================================================================
// TRIANGLE CULL PIPELINE
// =============================================================================

/// Compute pipeline for triangle culling.
pub struct TriangleCullPipeline {
    /// Main culling pipeline (all tests).
    pub pipeline: wgpu::ComputePipeline,
    /// Backface-only pipeline.
    pub pipeline_backface_only: wgpu::ComputePipeline,
    /// Frustum-only pipeline.
    pub pipeline_frustum_only: wgpu::ComputePipeline,
    /// No-frustum pipeline (backface + degenerate).
    pub pipeline_no_frustum: wgpu::ComputePipeline,
    /// Bind group layout.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl TriangleCullPipeline {
    /// Create the triangle culling pipeline.
    pub fn new(device: &Device) -> Self {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_cull_triangle.comp.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("triangle_cull_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("triangle_cull_bind_group_layout"),
            entries: &[
                // @binding(0) params: TriangleCullParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<TriangleCullParams>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) triangles: array<TriangleInput>
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
                // @binding(2) results: array<CullResult>
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
            label: Some("triangle_cull_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("triangle_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_triangle",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_backface_only = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("triangle_cull_pipeline_backface_only"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_triangle_backface_only",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_frustum_only = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("triangle_cull_pipeline_frustum_only"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_triangle_frustum_only",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_no_frustum = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("triangle_cull_pipeline_no_frustum"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_triangle_no_frustum",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            pipeline_backface_only,
            pipeline_frustum_only,
            pipeline_no_frustum,
            bind_group_layout,
        }
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &Device,
        resources: &TriangleCullResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("triangle_cull_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.triangles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.results_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch the main culling pipeline.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &TriangleCullParams,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("triangle_cull_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(params.num_workgroups(), 1, 1);
    }

    /// Dispatch backface-only culling.
    pub fn dispatch_backface_only(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &TriangleCullParams,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("triangle_cull_backface_only_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_backface_only);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(params.num_workgroups(), 1, 1);
    }

    /// Dispatch frustum-only culling.
    pub fn dispatch_frustum_only(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &TriangleCullParams,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("triangle_cull_frustum_only_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_frustum_only);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(params.num_workgroups(), 1, 1);
    }

    /// Dispatch no-frustum culling (backface + degenerate).
    pub fn dispatch_no_frustum(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &TriangleCullParams,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("triangle_cull_no_frustum_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_no_frustum);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(params.num_workgroups(), 1, 1);
    }
}

// =============================================================================
// CPU REFERENCE IMPLEMENTATION
// =============================================================================

/// Transform a 3D point by a 4x4 matrix (column-major) to clip space.
fn transform_to_clip(point: [f32; 3], view_proj: &[[f32; 4]; 4]) -> [f32; 4] {
    [
        view_proj[0][0] * point[0]
            + view_proj[1][0] * point[1]
            + view_proj[2][0] * point[2]
            + view_proj[3][0],
        view_proj[0][1] * point[0]
            + view_proj[1][1] * point[1]
            + view_proj[2][1] * point[2]
            + view_proj[3][1],
        view_proj[0][2] * point[0]
            + view_proj[1][2] * point[1]
            + view_proj[2][2] * point[2]
            + view_proj[3][2],
        view_proj[0][3] * point[0]
            + view_proj[1][3] * point[1]
            + view_proj[2][3] * point[2]
            + view_proj[3][3],
    ]
}

/// Compute outcode for a clip-space vertex.
fn compute_outcode(clip: [f32; 4]) -> u32 {
    let w = clip[3];
    let mut code = 0u32;

    if clip[0] < -w { code |= 1; } // LEFT
    if clip[0] > w { code |= 2; }  // RIGHT
    if clip[1] < -w { code |= 4; } // BOTTOM
    if clip[1] > w { code |= 8; }  // TOP
    if clip[2] < 0.0 { code |= 16; } // NEAR
    if clip[2] > w { code |= 32; }   // FAR

    code
}

/// Check if triangle is entirely outside frustum using outcodes.
fn cpu_is_frustum_culled(c0: [f32; 4], c1: [f32; 4], c2: [f32; 4]) -> bool {
    let code0 = compute_outcode(c0);
    let code1 = compute_outcode(c1);
    let code2 = compute_outcode(c2);
    (code0 & code1 & code2) != 0
}

/// Check if triangle is backfacing in clip space.
fn cpu_is_backfacing(c0: [f32; 4], c1: [f32; 4], c2: [f32; 4], cull_mode: CullMode) -> bool {
    if cull_mode == CullMode::None {
        return false;
    }

    // Handle vertices behind camera
    if c0[3] <= 0.0 || c1[3] <= 0.0 || c2[3] <= 0.0 {
        return false;
    }

    // Convert to NDC
    let ndc0 = [c0[0] / c0[3], c0[1] / c0[3]];
    let ndc1 = [c1[0] / c1[3], c1[1] / c1[3]];
    let ndc2 = [c2[0] / c2[3], c2[1] / c2[3]];

    // 2D cross product z component
    let e1 = [ndc1[0] - ndc0[0], ndc1[1] - ndc0[1]];
    let e2 = [ndc2[0] - ndc0[0], ndc2[1] - ndc0[1]];
    let cross_z = e1[0] * e2[1] - e1[1] * e2[0];

    match cull_mode {
        CullMode::CounterClockwise => cross_z > 0.0,
        CullMode::Clockwise => cross_z < 0.0,
        CullMode::None => false,
    }
}

/// Check if triangle is degenerate based on screen-space area.
fn cpu_is_degenerate(
    c0: [f32; 4],
    c1: [f32; 4],
    c2: [f32; 4],
    threshold: f32,
    viewport_width: f32,
    viewport_height: f32,
) -> bool {
    if threshold <= 0.0 {
        return false;
    }

    // Handle vertices behind camera
    if c0[3] <= 0.0 || c1[3] <= 0.0 || c2[3] <= 0.0 {
        return false;
    }

    // Convert to screen pixels
    let half_w = viewport_width * 0.5;
    let half_h = viewport_height * 0.5;

    let p0 = [(c0[0] / c0[3]) * half_w, (c0[1] / c0[3]) * half_h];
    let p1 = [(c1[0] / c1[3]) * half_w, (c1[1] / c1[3]) * half_h];
    let p2 = [(c2[0] / c2[3]) * half_w, (c2[1] / c2[3]) * half_h];

    let e1 = [p1[0] - p0[0], p1[1] - p0[1]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1]];
    let area = (e1[0] * e2[1] - e1[1] * e2[0]).abs() * 0.5;

    area < threshold
}

/// CPU reference implementation of triangle culling.
///
/// Returns a vector of cull results matching the input triangles.
pub fn cpu_triangle_cull(params: &TriangleCullParams, triangles: &[TriangleInput]) -> Vec<CullResult> {
    let cull_mode = CullMode::from_u32(params.cull_backface);

    triangles
        .iter()
        .map(|tri| {
            let c0 = transform_to_clip(tri.v0, &params.view_proj);
            let c1 = transform_to_clip(tri.v1, &params.view_proj);
            let c2 = transform_to_clip(tri.v2, &params.view_proj);

            // Frustum test first (cheapest rejection)
            if cpu_is_frustum_culled(c0, c1, c2) {
                return CullResult::culled(CullReason::Frustum);
            }

            // Backface test
            if cpu_is_backfacing(c0, c1, c2, cull_mode) {
                return CullResult::culled(CullReason::Backface);
            }

            // Degenerate test
            if cpu_is_degenerate(
                c0,
                c1,
                c2,
                params.degenerate_threshold,
                params.viewport_width,
                params.viewport_height,
            ) {
                return CullResult::culled(CullReason::Degenerate);
            }

            CullResult::visible()
        })
        .collect()
}

/// CPU reference: backface culling only.
pub fn cpu_triangle_cull_backface_only(
    params: &TriangleCullParams,
    triangles: &[TriangleInput],
) -> Vec<CullResult> {
    let cull_mode = CullMode::from_u32(params.cull_backface);

    triangles
        .iter()
        .map(|tri| {
            let c0 = transform_to_clip(tri.v0, &params.view_proj);
            let c1 = transform_to_clip(tri.v1, &params.view_proj);
            let c2 = transform_to_clip(tri.v2, &params.view_proj);

            if cpu_is_backfacing(c0, c1, c2, cull_mode) {
                CullResult::culled(CullReason::Backface)
            } else {
                CullResult::visible()
            }
        })
        .collect()
}

/// CPU reference: frustum culling only.
pub fn cpu_triangle_cull_frustum_only(
    params: &TriangleCullParams,
    triangles: &[TriangleInput],
) -> Vec<CullResult> {
    triangles
        .iter()
        .map(|tri| {
            let c0 = transform_to_clip(tri.v0, &params.view_proj);
            let c1 = transform_to_clip(tri.v1, &params.view_proj);
            let c2 = transform_to_clip(tri.v2, &params.view_proj);

            if cpu_is_frustum_culled(c0, c1, c2) {
                CullResult::culled(CullReason::Frustum)
            } else {
                CullResult::visible()
            }
        })
        .collect()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Size/Layout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_triangle_cull_params_size() {
        assert_eq!(mem::size_of::<TriangleCullParams>(), 96);
    }

    #[test]
    fn test_triangle_input_size() {
        assert_eq!(mem::size_of::<TriangleInput>(), 64);
    }

    #[test]
    fn test_cull_result_size() {
        assert_eq!(mem::size_of::<CullResult>(), 8);
    }

    #[test]
    fn test_params_pod() {
        let params = TriangleCullParams::default();
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 96);
    }

    #[test]
    fn test_triangle_input_pod() {
        let tri = TriangleInput::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0, 0);
        let bytes = bytemuck::bytes_of(&tri);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn test_cull_result_pod() {
        let result = CullResult::visible();
        let bytes = bytemuck::bytes_of(&result);
        assert_eq!(bytes.len(), 8);
    }

    // -------------------------------------------------------------------------
    // CullMode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cull_mode_roundtrip() {
        assert_eq!(CullMode::from_u32(CullMode::None.to_u32()), CullMode::None);
        assert_eq!(
            CullMode::from_u32(CullMode::CounterClockwise.to_u32()),
            CullMode::CounterClockwise
        );
        assert_eq!(CullMode::from_u32(CullMode::Clockwise.to_u32()), CullMode::Clockwise);
    }

    #[test]
    fn test_cull_mode_invalid() {
        assert_eq!(CullMode::from_u32(99), CullMode::None);
    }

    // -------------------------------------------------------------------------
    // CullReason Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cull_reason_roundtrip() {
        assert_eq!(CullReason::from_u32(CullReason::None.to_u32()), CullReason::None);
        assert_eq!(CullReason::from_u32(CullReason::Backface.to_u32()), CullReason::Backface);
        assert_eq!(CullReason::from_u32(CullReason::Degenerate.to_u32()), CullReason::Degenerate);
        assert_eq!(CullReason::from_u32(CullReason::Frustum.to_u32()), CullReason::Frustum);
    }

    #[test]
    fn test_cull_reason_is_culled() {
        assert!(!CullReason::None.is_culled());
        assert!(CullReason::Backface.is_culled());
        assert!(CullReason::Degenerate.is_culled());
        assert!(CullReason::Frustum.is_culled());
    }

    // -------------------------------------------------------------------------
    // TriangleCullParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_params_num_workgroups() {
        let params = TriangleCullParams { num_triangles: 1, ..Default::default() };
        assert_eq!(params.num_workgroups(), 1);

        let params = TriangleCullParams { num_triangles: 256, ..Default::default() };
        assert_eq!(params.num_workgroups(), 1);

        let params = TriangleCullParams { num_triangles: 257, ..Default::default() };
        assert_eq!(params.num_workgroups(), 2);

        let params = TriangleCullParams { num_triangles: 1000, ..Default::default() };
        assert_eq!(params.num_workgroups(), 4);
    }

    #[test]
    fn test_params_builder() {
        let params = TriangleCullParams::new(100, [[1.0, 0.0, 0.0, 0.0]; 4])
            .with_cull_mode(CullMode::Clockwise)
            .with_degenerate_threshold(0.5)
            .with_viewport(800.0, 600.0);

        assert_eq!(params.num_triangles, 100);
        assert_eq!(params.cull_mode(), CullMode::Clockwise);
        assert!((params.degenerate_threshold - 0.5).abs() < 1e-6);
        assert!((params.viewport_width - 800.0).abs() < 1e-6);
        assert!((params.viewport_height - 600.0).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // TriangleInput Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_triangle_input_new() {
        let tri = TriangleInput::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], 42, 17);
        assert_eq!(tri.v0, [1.0, 2.0, 3.0]);
        assert_eq!(tri.v1, [4.0, 5.0, 6.0]);
        assert_eq!(tri.v2, [7.0, 8.0, 9.0]);
        assert_eq!(tri.instance_id, 42);
        assert_eq!(tri.primitive_id, 17);
    }

    #[test]
    fn test_triangle_input_from_vertices() {
        let verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let tri = TriangleInput::from_vertices(verts, 0, 0);
        assert_eq!(tri.v0, verts[0]);
        assert_eq!(tri.v1, verts[1]);
        assert_eq!(tri.v2, verts[2]);
    }

    #[test]
    fn test_triangle_compute_area() {
        // Unit right triangle: area = 0.5
        let tri = TriangleInput::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0, 0);
        let area = tri.compute_area();
        assert!((area - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_triangle_compute_normal() {
        // XY plane triangle, normal should point +Z
        let tri = TriangleInput::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0, 0);
        let normal = tri.compute_normal();
        // Cross product of (1,0,0) x (0,1,0) = (0,0,1)
        assert!((normal[0] - 0.0).abs() < 1e-6);
        assert!((normal[1] - 0.0).abs() < 1e-6);
        assert!((normal[2] - 1.0).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // CullResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cull_result_visible() {
        let result = CullResult::visible();
        assert!(result.is_visible());
        assert_eq!(result.reason(), CullReason::None);
    }

    #[test]
    fn test_cull_result_culled() {
        let result = CullResult::culled(CullReason::Backface);
        assert!(!result.is_visible());
        assert_eq!(result.reason(), CullReason::Backface);
    }

    // -------------------------------------------------------------------------
    // CPU Culling Tests - Frontfacing Triangle
    // -------------------------------------------------------------------------

    /// Identity view-projection for simple tests
    fn identity_vp() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Simple orthographic-like projection (no perspective)
    fn ortho_vp() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0], // z maps [0,2] to [0,1]
            [0.0, 0.0, 0.5, 1.0],
        ]
    }

    #[test]
    fn test_frontfacing_triangle_visible() {
        // CCW triangle in XY plane at z=0.5 (inside frustum)
        // When viewed from +Z, CCW winding is frontfacing
        let tri = TriangleInput::new([0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::Clockwise); // Cull CW, keep CCW

        let results = cpu_triangle_cull(&params, &[tri]);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_visible(), "Frontfacing CCW triangle should be visible");
    }

    // -------------------------------------------------------------------------
    // CPU Culling Tests - Backfacing Triangle
    // -------------------------------------------------------------------------

    #[test]
    fn test_backfacing_triangle_culled_ccw_mode() {
        // CCW triangle - should be culled when CCW mode is active
        let tri = TriangleInput::new([0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::CounterClockwise);

        let results = cpu_triangle_cull(&params, &[tri]);
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_visible());
        assert_eq!(results[0].reason(), CullReason::Backface);
    }

    #[test]
    fn test_backfacing_triangle_culled_cw_mode() {
        // CW triangle - should be culled when CW mode is active
        let tri = TriangleInput::new([0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::Clockwise);

        let results = cpu_triangle_cull(&params, &[tri]);
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_visible());
        assert_eq!(results[0].reason(), CullReason::Backface);
    }

    #[test]
    fn test_backface_cull_disabled() {
        // Any triangle should be visible when backface culling is disabled
        let tri = TriangleInput::new([0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::None);

        let results = cpu_triangle_cull(&params, &[tri]);
        assert!(results[0].is_visible());
    }

    // -------------------------------------------------------------------------
    // CPU Culling Tests - Degenerate Triangle
    // -------------------------------------------------------------------------

    #[test]
    fn test_degenerate_triangle_zero_area() {
        // Collinear points = zero area
        let tri = TriangleInput::new([0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [1.0, 0.0, 0.5], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::None)
            .with_degenerate_threshold(1.0) // Generous threshold
            .with_viewport(1920.0, 1080.0);

        let results = cpu_triangle_cull(&params, &[tri]);
        assert!(!results[0].is_visible());
        assert_eq!(results[0].reason(), CullReason::Degenerate);
    }

    #[test]
    fn test_degenerate_triangle_same_point() {
        // All vertices at same point
        let tri = TriangleInput::new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::None)
            .with_degenerate_threshold(0.001)
            .with_viewport(1920.0, 1080.0);

        let results = cpu_triangle_cull(&params, &[tri]);
        assert!(!results[0].is_visible());
        assert_eq!(results[0].reason(), CullReason::Degenerate);
    }

    #[test]
    fn test_degenerate_threshold_zero_disabled() {
        // Degenerate test should be skipped when threshold is 0
        let tri = TriangleInput::new([0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [1.0, 0.0, 0.5], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::None)
            .with_degenerate_threshold(0.0);

        let results = cpu_triangle_cull(&params, &[tri]);
        // Should be visible because degenerate test is disabled
        assert!(results[0].is_visible());
    }

    // -------------------------------------------------------------------------
    // CPU Culling Tests - Frustum Culling
    // -------------------------------------------------------------------------

    #[test]
    fn test_triangle_outside_left() {
        // Triangle entirely left of frustum (x < -w for all vertices)
        let tri = TriangleInput::new([-3.0, 0.0, 0.5], [-2.5, 0.5, 0.5], [-2.5, -0.5, 0.5], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::None);

        let results = cpu_triangle_cull(&params, &[tri]);
        assert!(!results[0].is_visible());
        assert_eq!(results[0].reason(), CullReason::Frustum);
    }

    #[test]
    fn test_triangle_outside_right() {
        let tri = TriangleInput::new([3.0, 0.0, 0.5], [2.5, 0.5, 0.5], [2.5, -0.5, 0.5], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::None);

        let results = cpu_triangle_cull(&params, &[tri]);
        assert!(!results[0].is_visible());
        assert_eq!(results[0].reason(), CullReason::Frustum);
    }

    #[test]
    fn test_triangle_outside_near() {
        // Triangle behind near plane (z < 0 in clip space)
        // With ortho_vp: z_clip = 0.5 * z + 0.5, so z < -1 gives z_clip < 0
        let tri = TriangleInput::new([0.0, 0.0, -2.0], [0.5, 0.0, -2.0], [0.0, 0.5, -2.0], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::None);

        let results = cpu_triangle_cull(&params, &[tri]);
        assert!(!results[0].is_visible());
        assert_eq!(results[0].reason(), CullReason::Frustum);
    }

    #[test]
    fn test_triangle_outside_far() {
        // Triangle beyond far plane (z > w in clip space)
        let tri = TriangleInput::new([0.0, 0.0, 5.0], [0.5, 0.0, 5.0], [0.0, 0.5, 5.0], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::None);

        let results = cpu_triangle_cull(&params, &[tri]);
        assert!(!results[0].is_visible());
        assert_eq!(results[0].reason(), CullReason::Frustum);
    }

    #[test]
    fn test_triangle_spanning_frustum_visible() {
        // Triangle that spans the frustum boundary should be visible
        let tri = TriangleInput::new([-0.5, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], 0, 0);

        let params = TriangleCullParams::new(1, ortho_vp())
            .with_cull_mode(CullMode::None);

        let results = cpu_triangle_cull(&params, &[tri]);
        assert!(results[0].is_visible());
    }

    // -------------------------------------------------------------------------
    // CPU Culling Tests - Cull Reason Tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_cull_reason_tracking_mixed() {
        let triangles = vec![
            // Visible triangle
            TriangleInput::new([0.0, 0.0, 0.5], [0.3, 0.0, 0.5], [0.0, 0.3, 0.5], 0, 0),
            // Backfacing (CCW when culling CCW)
            TriangleInput::new([0.0, 0.0, 0.5], [0.3, 0.0, 0.5], [0.0, 0.3, 0.5], 1, 0),
            // Outside frustum
            TriangleInput::new([-5.0, 0.0, 0.5], [-4.5, 0.0, 0.5], [-5.0, 0.5, 0.5], 2, 0),
        ];

        let params = TriangleCullParams::new(3, ortho_vp())
            .with_cull_mode(CullMode::CounterClockwise);

        let results = cpu_triangle_cull(&params, &triangles);

        // First is backface culled (CCW mode culls CCW triangles)
        assert!(!results[0].is_visible());
        assert_eq!(results[0].reason(), CullReason::Backface);

        // Second same
        assert!(!results[1].is_visible());
        assert_eq!(results[1].reason(), CullReason::Backface);

        // Third is frustum culled
        assert!(!results[2].is_visible());
        assert_eq!(results[2].reason(), CullReason::Frustum);
    }

    // -------------------------------------------------------------------------
    // Shader Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_triangle_cull_shader_parses() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_cull_triangle.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("triangle cull shader should parse without errors");

        // Verify expected entry points
        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "cull_triangle"),
            "Should have cull_triangle entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "cull_triangle_backface_only"),
            "Should have cull_triangle_backface_only entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "cull_triangle_frustum_only"),
            "Should have cull_triangle_frustum_only entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "cull_triangle_no_frustum"),
            "Should have cull_triangle_no_frustum entry point"
        );
    }

    #[test]
    fn test_triangle_cull_shader_validates() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_cull_triangle.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("triangle cull shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("triangle cull shader should validate without errors");
    }

    #[test]
    fn test_triangle_cull_shader_workgroup_size() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_cull_triangle.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("triangle cull shader should parse without errors");

        for ep in &module.entry_points {
            if ep.stage == naga::ShaderStage::Compute {
                assert_eq!(
                    ep.workgroup_size,
                    [256, 1, 1],
                    "Entry point {} should have workgroup size 256x1x1",
                    ep.name
                );
            }
        }
    }

    #[test]
    fn test_triangle_cull_shader_entry_points_are_compute() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_cull_triangle.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("triangle cull shader should parse without errors");

        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                naga::ShaderStage::Compute,
                "Entry point {} should be a compute shader",
                ep.name
            );
        }
    }
}
