//! GPU Small Triangle Culling for TRINITY Engine (T-GPU-3.8).
//!
//! This module provides GPU-based small triangle culling using compute shaders.
//! It culls triangles that project to sub-pixel sizes, avoiding wasted
//! rasterization resources on geometry that won't contribute to the final image.
//!
//! # Overview
//!
//! Small triangle culling eliminates triangles whose screen-space area is below
//! a configurable threshold (default: 1 pixel). This is particularly beneficial
//! for:
//!
//! - High-detail meshes viewed at distance
//! - Dense foliage and vegetation
//! - Particle systems with billboard quads
//! - LOD transitions where triangles shrink below visible size
//!
//! # Algorithm
//!
//! 1. Input triangles are pre-projected to NDC space
//! 2. NDC coordinates are converted to pixel coordinates
//! 3. Triangle area is computed using the 2D cross product (shoelace formula)
//! 4. Triangles below the pixel threshold are marked as culled
//!
//! # Performance
//!
//! - Work complexity: O(n), one thread per triangle
//! - Target: < 0.02ms for 100K triangles
//! - Memory: 32 bytes per projected triangle
//!
//! # Usage
//!
//! ```ignore
//! // Create pipeline and resources
//! let pipeline = SmallTriangleCullPipeline::new(&device);
//! let resources = SmallTriangleCullResources::new(&device, 100_000);
//!
//! // Each frame: upload triangles and cull
//! let params = SmallTriangleCullParams::new(
//!     triangle_count,
//!     &view_proj_matrix,
//!     1920.0, 1080.0,  // viewport dimensions
//!     1.0,             // min_pixel_area
//! );
//! resources.upload_params(&queue, &params);
//! resources.upload_triangles(&queue, &projected_triangles);
//! pipeline.dispatch(&mut encoder, &resources, triangle_count);
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

/// Default minimum pixel area threshold.
pub const DEFAULT_MIN_PIXEL_AREA: f32 = 1.0;

/// Epsilon for degenerate triangle detection.
pub const DEGENERATE_EPSILON: f32 = 1e-10;

// ---------------------------------------------------------------------------
// SmallTriangleCullParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for small triangle culling parameters.
///
/// # Memory Layout
///
/// 96 bytes, std140 compatible:
/// | Offset | Field             | Size |
/// |--------|-------------------|------|
/// | 0      | num_triangles     | 4    |
/// | 4      | _pad0             | 4    |
/// | 8      | _pad1             | 4    |
/// | 12     | _pad2             | 4    |
/// | 16     | view_proj         | 64   |
/// | 80     | viewport_width    | 4    |
/// | 84     | viewport_height   | 4    |
/// | 88     | min_pixel_area    | 4    |
/// | 92     | _pad3             | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SmallTriangleCullParams {
    /// Number of triangles to process.
    pub num_triangles: u32,
    /// Padding for mat4 alignment.
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    /// View-projection matrix in column-major order.
    pub view_proj: [[f32; 4]; 4],
    /// Viewport width in pixels.
    pub viewport_width: f32,
    /// Viewport height in pixels.
    pub viewport_height: f32,
    /// Minimum visible area in pixels.
    pub min_pixel_area: f32,
    /// Padding for 16-byte alignment.
    pub _pad3: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<SmallTriangleCullParams>() == 96);

impl Default for SmallTriangleCullParams {
    fn default() -> Self {
        Self {
            num_triangles: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            viewport_width: 1920.0,
            viewport_height: 1080.0,
            min_pixel_area: DEFAULT_MIN_PIXEL_AREA,
            _pad3: 0.0,
        }
    }
}

impl SmallTriangleCullParams {
    /// Create parameters for small triangle culling.
    ///
    /// # Arguments
    ///
    /// * `num_triangles` - Number of triangles to process.
    /// * `view_proj` - View-projection matrix in column-major order.
    /// * `viewport_width` - Viewport width in pixels.
    /// * `viewport_height` - Viewport height in pixels.
    /// * `min_pixel_area` - Minimum visible area in pixels (default: 1.0).
    pub fn new(
        num_triangles: u32,
        view_proj: &[[f32; 4]; 4],
        viewport_width: f32,
        viewport_height: f32,
        min_pixel_area: f32,
    ) -> Self {
        Self {
            num_triangles,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            view_proj: *view_proj,
            viewport_width,
            viewport_height,
            min_pixel_area,
            _pad3: 0.0,
        }
    }

    /// Create parameters with default min_pixel_area (1.0).
    pub fn with_defaults(
        num_triangles: u32,
        view_proj: &[[f32; 4]; 4],
        viewport_width: f32,
        viewport_height: f32,
    ) -> Self {
        Self::new(
            num_triangles,
            view_proj,
            viewport_width,
            viewport_height,
            DEFAULT_MIN_PIXEL_AREA,
        )
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_triangles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

// ---------------------------------------------------------------------------
// ProjectedTriangle
// ---------------------------------------------------------------------------

/// Pre-projected triangle with screen-space positions.
///
/// Contains NDC coordinates (after perspective divide) for each vertex,
/// plus instance and primitive IDs for draw call lookup.
///
/// # Memory Layout
///
/// 32 bytes:
/// | Offset | Field        | Size |
/// |--------|--------------|------|
/// | 0      | p0           | 8    |
/// | 8      | p1           | 8    |
/// | 16     | p2           | 8    |
/// | 24     | instance_id  | 4    |
/// | 28     | primitive_id | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProjectedTriangle {
    /// Screen-space position of vertex 0 (NDC, after perspective divide).
    pub p0: [f32; 2],
    /// Screen-space position of vertex 1 (NDC, after perspective divide).
    pub p1: [f32; 2],
    /// Screen-space position of vertex 2 (NDC, after perspective divide).
    pub p2: [f32; 2],
    /// Instance ID for draw call lookup.
    pub instance_id: u32,
    /// Primitive ID within the mesh.
    pub primitive_id: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ProjectedTriangle>() == 32);

impl ProjectedTriangle {
    /// Create a projected triangle from NDC coordinates and IDs.
    pub fn new(
        p0: [f32; 2],
        p1: [f32; 2],
        p2: [f32; 2],
        instance_id: u32,
        primitive_id: u32,
    ) -> Self {
        Self {
            p0,
            p1,
            p2,
            instance_id,
            primitive_id,
        }
    }

    /// Create a projected triangle from world positions and view-projection matrix.
    ///
    /// Projects each vertex to clip space, performs perspective divide to get NDC.
    pub fn from_world_positions(
        v0: [f32; 3],
        v1: [f32; 3],
        v2: [f32; 3],
        view_proj: &[[f32; 4]; 4],
        instance_id: u32,
        primitive_id: u32,
    ) -> Self {
        let p0 = project_to_ndc(v0, view_proj);
        let p1 = project_to_ndc(v1, view_proj);
        let p2 = project_to_ndc(v2, view_proj);

        Self {
            p0,
            p1,
            p2,
            instance_id,
            primitive_id,
        }
    }
}

// ---------------------------------------------------------------------------
// SmallTriangleCullResult
// ---------------------------------------------------------------------------

/// Result of small triangle culling operation.
#[derive(Clone, Debug, Default)]
pub struct SmallTriangleCullResult {
    /// Visibility flags for each triangle (1 = visible, 0 = culled).
    pub visible_flags: Vec<u32>,
    /// Number of triangles that were culled.
    pub culled_count: u32,
}

impl SmallTriangleCullResult {
    /// Check if a specific triangle is visible.
    #[inline]
    pub fn is_visible(&self, index: usize) -> bool {
        self.visible_flags.get(index).copied().unwrap_or(0) != 0
    }

    /// Get the number of visible triangles.
    #[inline]
    pub fn visible_count(&self) -> u32 {
        self.visible_flags.len() as u32 - self.culled_count
    }

    /// Get the cull ratio (0.0 = none culled, 1.0 = all culled).
    #[inline]
    pub fn cull_ratio(&self) -> f32 {
        if self.visible_flags.is_empty() {
            0.0
        } else {
            self.culled_count as f32 / self.visible_flags.len() as f32
        }
    }
}

// ---------------------------------------------------------------------------
// SmallTriangleCullResources
// ---------------------------------------------------------------------------

/// GPU resources for small triangle culling.
///
/// Contains all buffers needed for the culling compute shader.
pub struct SmallTriangleCullResources {
    /// Uniform buffer for culling parameters.
    pub params_buffer: wgpu::Buffer,
    /// Storage buffer for projected triangles (input).
    pub triangles_buffer: wgpu::Buffer,
    /// Storage buffer for visibility flags (output).
    pub visibility_buffer: wgpu::Buffer,
    /// Storage buffer for atomic culled count (output).
    pub culled_count_buffer: wgpu::Buffer,
    /// Staging buffer for reading visibility back to CPU.
    pub visibility_staging: wgpu::Buffer,
    /// Staging buffer for reading culled count back to CPU.
    pub culled_count_staging: wgpu::Buffer,
    /// Maximum number of triangles supported.
    pub capacity: u32,
}

impl SmallTriangleCullResources {
    /// Create culling resources for the given capacity.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("small_triangle_cull_params"),
            size: mem::size_of::<SmallTriangleCullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("small_triangle_cull_triangles"),
            size: (capacity as u64) * (mem::size_of::<ProjectedTriangle>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("small_triangle_cull_visibility"),
            size: (capacity as u64) * 4, // u32 per triangle
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let culled_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("small_triangle_cull_count"),
            size: 4, // single u32 atomic
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visibility_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("small_triangle_cull_visibility_staging"),
            size: (capacity as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let culled_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("small_triangle_cull_count_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            triangles_buffer,
            visibility_buffer,
            culled_count_buffer,
            visibility_staging,
            culled_count_staging,
            capacity,
        }
    }

    /// Upload culling parameters to GPU.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &SmallTriangleCullParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload projected triangles to GPU.
    ///
    /// # Panics
    ///
    /// Panics if `triangles.len() > self.capacity`.
    pub fn upload_triangles(&self, queue: &wgpu::Queue, triangles: &[ProjectedTriangle]) {
        assert!(triangles.len() <= self.capacity as usize);
        queue.write_buffer(&self.triangles_buffer, 0, bytemuck::cast_slice(triangles));
    }

    /// Reset the culled count to zero before dispatch.
    pub fn reset_culled_count(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.culled_count_buffer, 0, &[0u8; 4]);
    }
}

// ---------------------------------------------------------------------------
// SmallTriangleCullPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for small triangle culling.
pub struct SmallTriangleCullPipeline {
    /// Main pipeline: small triangle culling with statistics.
    pub pipeline: wgpu::ComputePipeline,
    /// Degenerate-only pipeline (zero-area triangles only).
    pub pipeline_degenerate_only: wgpu::ComputePipeline,
    /// No-stats pipeline (faster, no atomic counter).
    pub pipeline_no_stats: wgpu::ComputePipeline,
    /// Bind group layout for culling resources.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl SmallTriangleCullPipeline {
    /// Create the small triangle culling pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `shader_source` - WGSL shader source code.
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("small_triangle_cull_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("small_triangle_cull_bind_group_layout"),
            entries: &[
                // @binding(0) params: SmallTriangleCullParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<SmallTriangleCullParams>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) triangles: array<ProjectedTriangle>
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
                // @binding(2) visible_flags: array<u32>
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
                // @binding(3) culled_count: atomic<u32>
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("small_triangle_cull_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("small_triangle_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_small_triangle",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_degenerate_only =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("small_triangle_cull_pipeline_degenerate"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "cull_degenerate_only",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let pipeline_no_stats = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("small_triangle_cull_pipeline_no_stats"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_small_triangle_no_stats",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            pipeline_degenerate_only,
            pipeline_no_stats,
            bind_group_layout,
        }
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &SmallTriangleCullResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("small_triangle_cull_bind_group"),
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
                    resource: resources.visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.culled_count_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Project a 3D world position to 2D NDC coordinates.
///
/// Performs matrix multiplication and perspective divide.
fn project_to_ndc(pos: [f32; 3], view_proj: &[[f32; 4]; 4]) -> [f32; 2] {
    // Multiply by view-projection matrix (column-major)
    let clip_x = view_proj[0][0] * pos[0]
        + view_proj[1][0] * pos[1]
        + view_proj[2][0] * pos[2]
        + view_proj[3][0];
    let clip_y = view_proj[0][1] * pos[0]
        + view_proj[1][1] * pos[1]
        + view_proj[2][1] * pos[2]
        + view_proj[3][1];
    let clip_w = view_proj[0][3] * pos[0]
        + view_proj[1][3] * pos[1]
        + view_proj[2][3] * pos[2]
        + view_proj[3][3];

    // Perspective divide
    if clip_w.abs() < 1e-8 {
        [0.0, 0.0] // Degenerate case
    } else {
        [clip_x / clip_w, clip_y / clip_w]
    }
}

/// Convert NDC coordinates to pixel coordinates.
pub fn ndc_to_pixels(ndc: [f32; 2], viewport_width: f32, viewport_height: f32) -> [f32; 2] {
    [
        (ndc[0] + 1.0) * 0.5 * viewport_width,
        (ndc[1] + 1.0) * 0.5 * viewport_height,
    ]
}

/// Compute 2D triangle area using cross product (shoelace formula).
///
/// Returns the absolute area of the triangle in the coordinate system
/// of the input points.
pub fn triangle_area_2d(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> f32 {
    let e1 = [p1[0] - p0[0], p1[1] - p0[1]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1]];
    // 2D cross product gives signed parallelogram area
    let cross = e1[0] * e2[1] - e1[1] * e2[0];
    // Triangle is half the parallelogram
    cross.abs() * 0.5
}

/// Check if a triangle is degenerate (zero or near-zero area).
pub fn is_degenerate(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> bool {
    triangle_area_2d(p0, p1, p2) < DEGENERATE_EPSILON
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation of small triangle culling.
///
/// Used for testing and fallback when GPU is not available.
pub fn cpu_small_triangle_cull(
    triangles: &[ProjectedTriangle],
    viewport_width: f32,
    viewport_height: f32,
    min_pixel_area: f32,
) -> SmallTriangleCullResult {
    let mut visible_flags = Vec::with_capacity(triangles.len());
    let mut culled_count = 0u32;

    for tri in triangles {
        // Convert NDC to pixel coordinates
        let p0_pixels = ndc_to_pixels(tri.p0, viewport_width, viewport_height);
        let p1_pixels = ndc_to_pixels(tri.p1, viewport_width, viewport_height);
        let p2_pixels = ndc_to_pixels(tri.p2, viewport_width, viewport_height);

        // Compute area in pixels
        let area = triangle_area_2d(p0_pixels, p1_pixels, p2_pixels);

        if area < min_pixel_area {
            visible_flags.push(0);
            culled_count += 1;
        } else {
            visible_flags.push(1);
        }
    }

    SmallTriangleCullResult {
        visible_flags,
        culled_count,
    }
}

/// CPU reference implementation for degenerate-only culling.
pub fn cpu_cull_degenerate_only(triangles: &[ProjectedTriangle]) -> SmallTriangleCullResult {
    let mut visible_flags = Vec::with_capacity(triangles.len());
    let mut culled_count = 0u32;

    for tri in triangles {
        if is_degenerate(tri.p0, tri.p1, tri.p2) {
            visible_flags.push(0);
            culled_count += 1;
        } else {
            visible_flags.push(1);
        }
    }

    SmallTriangleCullResult {
        visible_flags,
        culled_count,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: Create identity view-projection matrix.
    fn identity_matrix() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Helper: Create a triangle with given NDC coordinates.
    fn make_triangle(
        p0: [f32; 2],
        p1: [f32; 2],
        p2: [f32; 2],
    ) -> ProjectedTriangle {
        ProjectedTriangle::new(p0, p1, p2, 0, 0)
    }

    // -----------------------------------------------------------------------
    // Large triangle tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_triangle_visible() {
        // Large triangle: 100+ pixels in 1920x1080 viewport
        // NDC coords: triangle spanning significant portion of screen
        let triangles = vec![make_triangle(
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.0, 0.5],
        )];

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        assert_eq!(result.visible_flags[0], 1, "Large triangle should be visible");
        assert_eq!(result.culled_count, 0);
    }

    #[test]
    fn test_very_large_triangle_visible() {
        // Full-screen triangle
        let triangles = vec![make_triangle(
            [-1.0, -1.0],
            [1.0, -1.0],
            [0.0, 1.0],
        )];

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        assert_eq!(result.visible_flags[0], 1, "Full-screen triangle should be visible");

        // Calculate expected area: half of 1920*1080 = 1,036,800 pixels
        let p0 = ndc_to_pixels([-1.0, -1.0], 1920.0, 1080.0);
        let p1 = ndc_to_pixels([1.0, -1.0], 1920.0, 1080.0);
        let p2 = ndc_to_pixels([0.0, 1.0], 1920.0, 1080.0);
        let area = triangle_area_2d(p0, p1, p2);
        assert!(area > 1_000_000.0, "Full-screen triangle area should be > 1M pixels");
    }

    // -----------------------------------------------------------------------
    // Small triangle tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_small_triangle_culled() {
        // Tiny triangle: < 1 pixel area
        // In a 1920x1080 viewport, NDC range is [-1, 1]
        // 1 pixel width = 2.0 / 1920 = 0.00104
        // For triangle area < 1 pixel:
        let triangles = vec![make_triangle(
            [0.0, 0.0],
            [0.0001, 0.0],
            [0.0, 0.0001],
        )];

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        assert_eq!(result.visible_flags[0], 0, "Sub-pixel triangle should be culled");
        assert_eq!(result.culled_count, 1);
    }

    #[test]
    fn test_sub_pixel_triangle_culled() {
        // Triangle smaller than a single pixel
        let triangles = vec![make_triangle(
            [0.0, 0.0],
            [0.00001, 0.0],
            [0.0, 0.00001],
        )];

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        assert_eq!(result.visible_flags[0], 0, "Sub-pixel triangle should be culled");
        assert_eq!(result.culled_count, 1);
    }

    // -----------------------------------------------------------------------
    // Edge case: exactly threshold area
    // -----------------------------------------------------------------------

    #[test]
    fn test_exactly_threshold_area() {
        // Create a triangle with exactly 1 pixel area
        // For 1920x1080: 1 pixel = (2/1920) * (2/1080) = ~1.93e-6 NDC^2
        // Triangle area = base * height / 2
        // Want: (width * height / 2) * (960 * 540) = 1 pixel
        // So in NDC: area = 1 / (960 * 540) = 1.93e-6
        // base * height = 2 * 1.93e-6 = 3.86e-6
        // Use base = 0.001964, height = 0.001964 (sqrt)

        // Actually, let's compute it properly:
        // Pixel area = NDC area * (viewport_width/2) * (viewport_height/2)
        // 1 pixel = NDC_area * 960 * 540 = NDC_area * 518400
        // NDC_area = 1 / 518400 = 1.93e-6

        // For triangle: area = |cross| / 2
        // We need: 0.5 * base * height = 1.93e-6
        // Let base = height = sqrt(2 * 1.93e-6) = 0.00196

        let side = 0.00197; // Slightly above threshold
        let triangles = vec![make_triangle(
            [0.0, 0.0],
            [side, 0.0],
            [0.0, side],
        )];

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        // Should be visible (slightly > 1 pixel)
        assert_eq!(result.visible_flags[0], 1, "Triangle at threshold should be visible");
    }

    #[test]
    fn test_just_below_threshold() {
        let side = 0.00195; // Slightly below threshold
        let triangles = vec![make_triangle(
            [0.0, 0.0],
            [side, 0.0],
            [0.0, side],
        )];

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        // Should be culled (slightly < 1 pixel)
        assert_eq!(result.visible_flags[0], 0, "Triangle below threshold should be culled");
    }

    // -----------------------------------------------------------------------
    // Degenerate triangle tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_degenerate_zero_area_culled() {
        // All three vertices at same point
        let triangles = vec![make_triangle(
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        )];

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        assert_eq!(result.visible_flags[0], 0, "Degenerate point triangle should be culled");
        assert_eq!(result.culled_count, 1);
    }

    #[test]
    fn test_degenerate_line_culled() {
        // Collinear vertices (line, not triangle)
        let triangles = vec![make_triangle(
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0], // All on same line
        )];

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        assert_eq!(result.visible_flags[0], 0, "Degenerate line triangle should be culled");
        assert_eq!(result.culled_count, 1);
    }

    #[test]
    fn test_degenerate_two_vertices_same() {
        // Two vertices at same position
        let triangles = vec![make_triangle(
            [0.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.5], // Same as p1
        )];

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        assert_eq!(result.visible_flags[0], 0, "Degenerate (two same vertices) should be culled");
    }

    // -----------------------------------------------------------------------
    // Multiple triangles mixed visibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_triangles_mixed_visibility() {
        let triangles = vec![
            // Large: visible
            make_triangle([-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]),
            // Tiny: culled
            make_triangle([0.0, 0.0], [0.00001, 0.0], [0.0, 0.00001]),
            // Large: visible
            make_triangle([-0.3, -0.3], [0.3, -0.3], [0.0, 0.3]),
            // Degenerate: culled
            make_triangle([0.1, 0.1], [0.1, 0.1], [0.1, 0.1]),
            // Medium: visible
            make_triangle([0.0, 0.0], [0.1, 0.0], [0.0, 0.1]),
        ];

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        assert_eq!(result.visible_flags, vec![1, 0, 1, 0, 1]);
        assert_eq!(result.culled_count, 2);
        assert_eq!(result.visible_count(), 3);
    }

    // -----------------------------------------------------------------------
    // Atomic counter accuracy
    // -----------------------------------------------------------------------

    #[test]
    fn test_culled_count_accuracy() {
        let mut triangles = Vec::new();

        // Add 100 visible triangles
        for i in 0..100 {
            let offset = i as f32 * 0.001;
            triangles.push(make_triangle(
                [-0.5 + offset, -0.5],
                [0.5 + offset, -0.5],
                [0.0 + offset, 0.5],
            ));
        }

        // Add 50 culled triangles (tiny)
        for _ in 0..50 {
            triangles.push(make_triangle(
                [0.0, 0.0],
                [0.00001, 0.0],
                [0.0, 0.00001],
            ));
        }

        let result = cpu_small_triangle_cull(&triangles, 1920.0, 1080.0, 1.0);

        assert_eq!(result.culled_count, 50, "Should count exactly 50 culled triangles");
        assert_eq!(result.visible_count(), 100, "Should have 100 visible triangles");
        assert!((result.cull_ratio() - 0.333).abs() < 0.01, "Cull ratio should be ~33%");
    }

    // -----------------------------------------------------------------------
    // Struct size verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_params_struct_size() {
        assert_eq!(
            mem::size_of::<SmallTriangleCullParams>(),
            96,
            "SmallTriangleCullParams must be 96 bytes for GPU alignment"
        );
    }

    #[test]
    fn test_projected_triangle_struct_size() {
        assert_eq!(
            mem::size_of::<ProjectedTriangle>(),
            32,
            "ProjectedTriangle must be 32 bytes"
        );
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_triangle_area_2d() {
        // Right triangle with legs 1, 1 -> area = 0.5
        let area = triangle_area_2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]);
        assert!((area - 0.5).abs() < 1e-6);

        // Right triangle with legs 2, 3 -> area = 3.0
        let area = triangle_area_2d([0.0, 0.0], [2.0, 0.0], [0.0, 3.0]);
        assert!((area - 3.0).abs() < 1e-6);

        // Reversed winding (should still be positive)
        let area = triangle_area_2d([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]);
        assert!((area - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ndc_to_pixels() {
        // Center of screen
        let pixels = ndc_to_pixels([0.0, 0.0], 1920.0, 1080.0);
        assert!((pixels[0] - 960.0).abs() < 1e-6);
        assert!((pixels[1] - 540.0).abs() < 1e-6);

        // Top-left corner
        let pixels = ndc_to_pixels([-1.0, -1.0], 1920.0, 1080.0);
        assert!((pixels[0] - 0.0).abs() < 1e-6);
        assert!((pixels[1] - 0.0).abs() < 1e-6);

        // Bottom-right corner
        let pixels = ndc_to_pixels([1.0, 1.0], 1920.0, 1080.0);
        assert!((pixels[0] - 1920.0).abs() < 1e-6);
        assert!((pixels[1] - 1080.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_degenerate() {
        // Point
        assert!(is_degenerate([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]));

        // Line
        assert!(is_degenerate([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]));

        // Valid triangle
        assert!(!is_degenerate([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]));
    }

    #[test]
    fn test_project_to_ndc() {
        let identity = identity_matrix();

        // Identity matrix should pass through
        let ndc = project_to_ndc([0.5, 0.5, 0.5], &identity);
        assert!((ndc[0] - 0.5).abs() < 1e-6);
        assert!((ndc[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_projected_triangle_from_world() {
        let identity = identity_matrix();

        let tri = ProjectedTriangle::from_world_positions(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            &identity,
            42,
            7,
        );

        assert_eq!(tri.instance_id, 42);
        assert_eq!(tri.primitive_id, 7);
        assert!((tri.p0[0] - 0.0).abs() < 1e-6);
        assert!((tri.p1[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cull_result_helpers() {
        let result = SmallTriangleCullResult {
            visible_flags: vec![1, 0, 1, 1, 0],
            culled_count: 2,
        };

        assert!(result.is_visible(0));
        assert!(!result.is_visible(1));
        assert!(result.is_visible(2));
        assert_eq!(result.visible_count(), 3);
        assert!((result.cull_ratio() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_num_workgroups() {
        let params1 = SmallTriangleCullParams::new(1, &identity_matrix(), 1920.0, 1080.0, 1.0);
        assert_eq!(params1.num_workgroups(), 1);

        let params256 = SmallTriangleCullParams::new(256, &identity_matrix(), 1920.0, 1080.0, 1.0);
        assert_eq!(params256.num_workgroups(), 1);

        let params257 = SmallTriangleCullParams::new(257, &identity_matrix(), 1920.0, 1080.0, 1.0);
        assert_eq!(params257.num_workgroups(), 2);

        let params1000 = SmallTriangleCullParams::new(1000, &identity_matrix(), 1920.0, 1080.0, 1.0);
        assert_eq!(params1000.num_workgroups(), 4);
    }

    #[test]
    fn test_different_viewport_sizes() {
        // Same NDC triangle, different viewports
        let triangles = vec![make_triangle(
            [0.0, 0.0],
            [0.01, 0.0],
            [0.0, 0.01],
        )];

        // Small viewport (320x240): triangle should be smaller in pixels
        let result_small = cpu_small_triangle_cull(&triangles, 320.0, 240.0, 1.0);

        // Large viewport (4K): triangle should be larger in pixels
        let result_large = cpu_small_triangle_cull(&triangles, 3840.0, 2160.0, 1.0);

        // In small viewport, this might be culled; in large viewport, visible
        // The same NDC size appears larger on a higher-res screen
        assert_eq!(result_large.visible_flags[0], 1, "Should be visible on 4K");
    }

    #[test]
    fn test_degenerate_only_mode() {
        let triangles = vec![
            // Valid triangle (visible in degenerate-only mode)
            make_triangle([-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]),
            // Tiny but valid (visible in degenerate-only mode)
            make_triangle([0.0, 0.0], [0.0001, 0.0], [0.0, 0.0001]),
            // Degenerate point (culled)
            make_triangle([0.1, 0.1], [0.1, 0.1], [0.1, 0.1]),
            // Degenerate line (culled)
            make_triangle([0.0, 0.0], [0.5, 0.5], [1.0, 1.0]),
        ];

        let result = cpu_cull_degenerate_only(&triangles);

        // Only truly degenerate triangles should be culled
        assert_eq!(result.visible_flags, vec![1, 1, 0, 0]);
        assert_eq!(result.culled_count, 2);
    }

    #[test]
    fn test_params_default() {
        let params = SmallTriangleCullParams::default();

        assert_eq!(params.num_triangles, 0);
        assert_eq!(params.viewport_width, 1920.0);
        assert_eq!(params.viewport_height, 1080.0);
        assert_eq!(params.min_pixel_area, DEFAULT_MIN_PIXEL_AREA);
    }

    #[test]
    fn test_params_with_defaults() {
        let vp = identity_matrix();
        let params = SmallTriangleCullParams::with_defaults(100, &vp, 1280.0, 720.0);

        assert_eq!(params.num_triangles, 100);
        assert_eq!(params.viewport_width, 1280.0);
        assert_eq!(params.viewport_height, 720.0);
        assert_eq!(params.min_pixel_area, DEFAULT_MIN_PIXEL_AREA);
    }
}
