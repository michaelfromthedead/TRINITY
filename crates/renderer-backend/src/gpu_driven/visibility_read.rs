//! GPU Visibility Buffer Read for TRINITY Engine (T-GPU-3.6).
//!
//! This module reads visibility buffer data and outputs interpolated vertex
//! attributes for material shading. It reconstructs world-space positions,
//! normals, UVs, and tangent space from triangle barycentrics.
//!
//! # Overview
//!
//! The visibility buffer read pass is the bridge between visibility buffer
//! rendering and deferred material shading:
//!
//! 1. **Input**: Visibility buffer with (instance_id, primitive_id, barycentrics)
//! 2. **Process**: Look up triangle vertices, interpolate attributes, transform
//! 3. **Output**: Shading inputs (world_pos, normal, uv, tangent space, material_id)
//!
//! # Key Features
//!
//! - Correct normal transformation using the normal matrix (transpose inverse)
//! - Gram-Schmidt orthogonalization for tangent space
//! - Handles denormalized vectors gracefully
//! - Per-pixel material ID output for material table lookup
//!
//! # Performance
//!
//! - Work complexity: O(pixels), one thread per pixel in 16x16 tiles
//! - Target: <0.2ms for 1080p (2M pixels)
//! - Memory: 64 bytes per output shading input
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{VisibilityReadPipeline, VisibilityReadResources};
//!
//! // Create pipeline and resources
//! let pipeline = VisibilityReadPipeline::new(&device);
//! let resources = VisibilityReadResources::new(&device, 1920, 1080, max_instances);
//!
//! // Each frame: read visibility and generate shading inputs
//! let params = VisibilityReadParams::new(1920, 1080);
//! resources.upload_params(&queue, &params);
//! pipeline.dispatch(&mut encoder, &resources, &params);
//!
//! // Shading inputs are now ready for material evaluation
//! ```

use std::mem;

use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferUsages, Device, Queue};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Compute shader tile size (16x16 = 256 threads per workgroup).
pub const TILE_SIZE: u32 = 16;

/// Workgroup size for linear operations (clear pass).
pub const WORKGROUP_SIZE: u32 = 256;

/// Invalid instance ID marker (background pixels).
pub const INVALID_INSTANCE: u32 = 0xFFFFFFFF;

/// Invalid primitive ID marker.
pub const INVALID_PRIMITIVE: u32 = 0xFFFFFFFF;

/// Small epsilon for safe normalization.
pub const EPSILON: f32 = 1e-7;

// =============================================================================
// VISIBILITY READ PARAMS
// =============================================================================

/// GPU uniform buffer for visibility read parameters.
///
/// Matches the WGSL `VisibilityReadParams` struct layout.
///
/// # Memory Layout
///
/// 16 bytes, std140/std430 compatible:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | screen_width  | 4    |
/// | 4      | screen_height | 4    |
/// | 8      | tile_offset_x | 4    |
/// | 12     | tile_offset_y | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct VisibilityReadParams {
    /// Screen width in pixels.
    pub screen_width: u32,
    /// Screen height in pixels.
    pub screen_height: u32,
    /// Tile offset X for tiled dispatch.
    pub tile_offset_x: u32,
    /// Tile offset Y for tiled dispatch.
    pub tile_offset_y: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VisibilityReadParams>() == 16);

impl VisibilityReadParams {
    /// Create parameters for the given screen size.
    pub const fn new(screen_width: u32, screen_height: u32) -> Self {
        Self {
            screen_width,
            screen_height,
            tile_offset_x: 0,
            tile_offset_y: 0,
        }
    }

    /// Create parameters with a specific tile offset.
    pub const fn with_tile_offset(
        screen_width: u32,
        screen_height: u32,
        tile_offset_x: u32,
        tile_offset_y: u32,
    ) -> Self {
        Self {
            screen_width,
            screen_height,
            tile_offset_x,
            tile_offset_y,
        }
    }

    /// Get the number of tiles in X direction.
    #[inline]
    pub const fn num_tiles_x(&self) -> u32 {
        (self.screen_width + TILE_SIZE - 1) / TILE_SIZE
    }

    /// Get the number of tiles in Y direction.
    #[inline]
    pub const fn num_tiles_y(&self) -> u32 {
        (self.screen_height + TILE_SIZE - 1) / TILE_SIZE
    }

    /// Get the total pixel count.
    #[inline]
    pub const fn pixel_count(&self) -> u32 {
        self.screen_width * self.screen_height
    }

    /// Get the number of workgroups for the clear pass.
    #[inline]
    pub const fn num_clear_workgroups(&self) -> u32 {
        (self.pixel_count() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

// =============================================================================
// VISIBILITY DATA
// =============================================================================

/// Visibility data per pixel (output from visibility buffer write pass).
///
/// # Memory Layout
///
/// 16 bytes:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | instance_id   | 4    |
/// | 4      | primitive_id  | 4    |
/// | 8      | barycentrics  | 8    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct VisibilityData {
    /// Instance ID (INVALID_INSTANCE = background).
    pub instance_id: u32,
    /// Primitive/triangle ID within the mesh.
    pub primitive_id: u32,
    /// Barycentric coordinates (beta, gamma); alpha = 1 - beta - gamma.
    pub barycentrics: [f32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VisibilityData>() == 16);

impl VisibilityData {
    /// Create visibility data for a valid pixel.
    pub const fn new(instance_id: u32, primitive_id: u32, bary_beta: f32, bary_gamma: f32) -> Self {
        Self {
            instance_id,
            primitive_id,
            barycentrics: [bary_beta, bary_gamma],
        }
    }

    /// Create visibility data for a background pixel.
    pub const fn invalid() -> Self {
        Self {
            instance_id: INVALID_INSTANCE,
            primitive_id: INVALID_PRIMITIVE,
            barycentrics: [0.0, 0.0],
        }
    }

    /// Check if this pixel is valid (has geometry).
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.instance_id != INVALID_INSTANCE && self.primitive_id != INVALID_PRIMITIVE
    }

    /// Get the barycentric weight for vertex 0 (alpha).
    #[inline]
    pub fn bary_alpha(&self) -> f32 {
        1.0 - self.barycentrics[0] - self.barycentrics[1]
    }

    /// Get the barycentric weight for vertex 1 (beta).
    #[inline]
    pub fn bary_beta(&self) -> f32 {
        self.barycentrics[0]
    }

    /// Get the barycentric weight for vertex 2 (gamma).
    #[inline]
    pub fn bary_gamma(&self) -> f32 {
        self.barycentrics[1]
    }
}

// =============================================================================
// INSTANCE TRANSFORM
// =============================================================================

/// Per-instance transform data for world-space calculations.
///
/// Contains both the world matrix and the normal matrix (transpose inverse
/// of the upper-left 3x3). The normal matrix is stored as three columns
/// to match WGSL's vec3 alignment requirements.
///
/// # Memory Layout
///
/// 112 bytes:
/// | Offset | Field              | Size |
/// |--------|--------------------|------|
/// | 0      | world_matrix       | 64   |
/// | 64     | normal_matrix_col0 | 12   |
/// | 76     | _pad0              | 4    |
/// | 80     | normal_matrix_col1 | 12   |
/// | 92     | _pad1              | 4    |
/// | 96     | normal_matrix_col2 | 12   |
/// | 108    | _pad2              | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct InstanceTransform {
    /// World matrix (model-to-world transform), column-major.
    pub world_matrix: [[f32; 4]; 4],
    /// Normal matrix column 0.
    pub normal_matrix_col0: [f32; 3],
    pub _pad0: f32,
    /// Normal matrix column 1.
    pub normal_matrix_col1: [f32; 3],
    pub _pad1: f32,
    /// Normal matrix column 2.
    pub normal_matrix_col2: [f32; 3],
    pub _pad2: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<InstanceTransform>() == 112);

impl Default for InstanceTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl InstanceTransform {
    /// Create an identity transform.
    pub const fn identity() -> Self {
        Self {
            world_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            normal_matrix_col0: [1.0, 0.0, 0.0],
            _pad0: 0.0,
            normal_matrix_col1: [0.0, 1.0, 0.0],
            _pad1: 0.0,
            normal_matrix_col2: [0.0, 0.0, 1.0],
            _pad2: 0.0,
        }
    }

    /// Create a transform from a world matrix.
    ///
    /// Automatically computes the normal matrix (transpose inverse of 3x3).
    pub fn from_world_matrix(world_matrix: [[f32; 4]; 4]) -> Self {
        let normal_matrix = compute_normal_matrix(&world_matrix);
        Self {
            world_matrix,
            normal_matrix_col0: [normal_matrix[0][0], normal_matrix[0][1], normal_matrix[0][2]],
            _pad0: 0.0,
            normal_matrix_col1: [normal_matrix[1][0], normal_matrix[1][1], normal_matrix[1][2]],
            _pad1: 0.0,
            normal_matrix_col2: [normal_matrix[2][0], normal_matrix[2][1], normal_matrix[2][2]],
            _pad2: 0.0,
        }
    }

    /// Create a translation transform.
    pub fn translation(x: f32, y: f32, z: f32) -> Self {
        Self::from_world_matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, y, z, 1.0],
        ])
    }

    /// Create a uniform scale transform.
    pub fn scale(s: f32) -> Self {
        Self::from_world_matrix([
            [s, 0.0, 0.0, 0.0],
            [0.0, s, 0.0, 0.0],
            [0.0, 0.0, s, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create a non-uniform scale transform.
    pub fn scale_xyz(sx: f32, sy: f32, sz: f32) -> Self {
        Self::from_world_matrix([
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Get the translation component.
    #[inline]
    pub fn translation_xyz(&self) -> [f32; 3] {
        [
            self.world_matrix[3][0],
            self.world_matrix[3][1],
            self.world_matrix[3][2],
        ]
    }
}

/// Compute the normal matrix (transpose inverse of upper-left 3x3).
///
/// The normal matrix is used to correctly transform normal vectors when
/// the model matrix contains non-uniform scaling or shearing.
fn compute_normal_matrix(world_matrix: &[[f32; 4]; 4]) -> [[f32; 3]; 3] {
    // Extract upper-left 3x3
    let m = [
        [world_matrix[0][0], world_matrix[0][1], world_matrix[0][2]],
        [world_matrix[1][0], world_matrix[1][1], world_matrix[1][2]],
        [world_matrix[2][0], world_matrix[2][1], world_matrix[2][2]],
    ];

    // Compute determinant
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    // Handle singular matrix (fall back to identity)
    if det.abs() < EPSILON {
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
    }

    let inv_det = 1.0 / det;

    // Compute inverse
    let inv = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ];

    // Return transpose of inverse (transposed for column-major shader access)
    inv
}

// =============================================================================
// VERTEX DATA
// =============================================================================

/// Vertex data for triangle reconstruction.
///
/// # Memory Layout
///
/// 64 bytes:
/// | Offset | Field    | Size |
/// |--------|----------|------|
/// | 0      | position | 12   |
/// | 12     | _pad0    | 4    |
/// | 16     | normal   | 12   |
/// | 28     | _pad1    | 4    |
/// | 32     | uv       | 8    |
/// | 40     | _pad2    | 8    |
/// | 48     | tangent  | 16   |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct VertexData {
    /// Object-space position.
    pub position: [f32; 3],
    pub _pad0: f32,
    /// Object-space normal (unit vector).
    pub normal: [f32; 3],
    pub _pad1: f32,
    /// Texture coordinates.
    pub uv: [f32; 2],
    /// Padding for alignment.
    pub _pad2: [f32; 2],
    /// Tangent with handedness in w component.
    pub tangent: [f32; 4],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VertexData>() == 64);

impl VertexData {
    /// Create new vertex data.
    pub const fn new(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        tangent: [f32; 4],
    ) -> Self {
        Self {
            position,
            _pad0: 0.0,
            normal,
            _pad1: 0.0,
            uv,
            _pad2: [0.0, 0.0],
            tangent,
        }
    }

    /// Create vertex data with default tangent (+X, handedness +1).
    pub const fn with_default_tangent(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
    ) -> Self {
        Self::new(position, normal, uv, [1.0, 0.0, 0.0, 1.0])
    }
}

// =============================================================================
// INSTANCE METADATA
// =============================================================================

/// Per-instance metadata for vertex/index buffer lookups.
///
/// # Memory Layout
///
/// 16 bytes:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | index_offset  | 4    |
/// | 4      | vertex_offset | 4    |
/// | 8      | material_id   | 4    |
/// | 12     | _pad          | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct InstanceMetadata {
    /// Base index into the global index buffer.
    pub index_offset: u32,
    /// Base vertex offset into the global vertex buffer.
    pub vertex_offset: u32,
    /// Material ID for this instance.
    pub material_id: u32,
    /// Reserved for future use.
    pub _pad: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<InstanceMetadata>() == 16);

impl InstanceMetadata {
    /// Create new instance metadata.
    pub const fn new(index_offset: u32, vertex_offset: u32, material_id: u32) -> Self {
        Self {
            index_offset,
            vertex_offset,
            material_id,
            _pad: 0,
        }
    }
}

// =============================================================================
// SHADING INPUT
// =============================================================================

/// Output shading inputs for material evaluation.
///
/// # Memory Layout
///
/// 80 bytes:
/// | Offset | Field        | Size |
/// |--------|--------------|------|
/// | 0      | world_pos    | 12   |
/// | 12     | _pad0        | 4    |
/// | 16     | world_normal | 12   |
/// | 28     | _pad1        | 4    |
/// | 32     | uv           | 8    |
/// | 40     | instance_id  | 4    |
/// | 44     | material_id  | 4    |
/// | 48     | tangent      | 12   |
/// | 60     | _pad2        | 4    |
/// | 64     | bitangent    | 12   |
/// | 76     | _pad3        | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ShadingInput {
    /// World-space position.
    pub world_pos: [f32; 3],
    pub _pad0: f32,
    /// World-space unit normal.
    pub world_normal: [f32; 3],
    pub _pad1: f32,
    /// Interpolated texture coordinates.
    pub uv: [f32; 2],
    /// Instance ID for additional lookups.
    pub instance_id: u32,
    /// Material ID for material table lookup.
    pub material_id: u32,
    /// World-space unit tangent.
    pub tangent: [f32; 3],
    pub _pad2: f32,
    /// World-space unit bitangent.
    pub bitangent: [f32; 3],
    pub _pad3: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ShadingInput>() == 80);

impl ShadingInput {
    /// Create an invalid/empty shading input (for background pixels).
    pub const fn invalid() -> Self {
        Self {
            world_pos: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            world_normal: [0.0, 1.0, 0.0],
            _pad1: 0.0,
            uv: [0.0, 0.0],
            instance_id: INVALID_INSTANCE,
            material_id: 0,
            tangent: [1.0, 0.0, 0.0],
            _pad2: 0.0,
            bitangent: [0.0, 0.0, 1.0],
            _pad3: 0.0,
        }
    }

    /// Check if this shading input is valid.
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.instance_id != INVALID_INSTANCE
    }

    /// Create a new shading input with all fields.
    pub const fn new(
        world_pos: [f32; 3],
        world_normal: [f32; 3],
        uv: [f32; 2],
        instance_id: u32,
        material_id: u32,
        tangent: [f32; 3],
        bitangent: [f32; 3],
    ) -> Self {
        Self {
            world_pos,
            _pad0: 0.0,
            world_normal,
            _pad1: 0.0,
            uv,
            instance_id,
            material_id,
            tangent,
            _pad2: 0.0,
            bitangent,
            _pad3: 0.0,
        }
    }
}

// =============================================================================
// VISIBILITY READ RESOURCES
// =============================================================================

/// GPU resources for visibility buffer read pass.
///
/// Contains all buffers needed for the visibility read algorithm.
pub struct VisibilityReadResources {
    /// Uniform buffer for parameters.
    pub params_buffer: Buffer,
    /// Visibility buffer (one entry per pixel).
    pub visibility_buffer: Buffer,
    /// Per-instance transforms.
    pub instance_transforms_buffer: Buffer,
    /// Per-instance metadata.
    pub instance_metadata_buffer: Buffer,
    /// Global vertex buffer.
    pub vertex_buffer: Buffer,
    /// Global index buffer.
    pub index_buffer: Buffer,
    /// Output shading inputs (one per pixel).
    pub shading_inputs_buffer: Buffer,
    /// Staging buffer for reading shading inputs back to CPU.
    pub shading_inputs_staging: Buffer,
    /// Screen width.
    pub screen_width: u32,
    /// Screen height.
    pub screen_height: u32,
    /// Maximum instances supported.
    pub max_instances: u32,
    /// Maximum vertices supported.
    pub max_vertices: u32,
    /// Maximum indices supported.
    pub max_indices: u32,
}

impl VisibilityReadResources {
    /// Create resources for visibility buffer read.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `screen_width` - Screen width in pixels
    /// * `screen_height` - Screen height in pixels
    /// * `max_instances` - Maximum number of instances
    /// * `max_vertices` - Maximum number of vertices
    /// * `max_indices` - Maximum number of indices
    pub fn new(
        device: &Device,
        screen_width: u32,
        screen_height: u32,
        max_instances: u32,
        max_vertices: u32,
        max_indices: u32,
    ) -> Self {
        let pixel_count = (screen_width * screen_height) as u64;

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_read_params"),
            size: mem::size_of::<VisibilityReadParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_buffer"),
            size: pixel_count * (mem::size_of::<VisibilityData>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_transforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_transforms"),
            size: (max_instances as u64) * (mem::size_of::<InstanceTransform>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_metadata_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_metadata"),
            size: (max_instances as u64) * (mem::size_of::<InstanceMetadata>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_read_vertices"),
            size: (max_vertices as u64) * (mem::size_of::<VertexData>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_read_indices"),
            size: (max_indices as u64) * (mem::size_of::<u32>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shading_inputs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shading_inputs"),
            size: pixel_count * (mem::size_of::<ShadingInput>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shading_inputs_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shading_inputs_staging"),
            size: pixel_count * (mem::size_of::<ShadingInput>() as u64),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            visibility_buffer,
            instance_transforms_buffer,
            instance_metadata_buffer,
            vertex_buffer,
            index_buffer,
            shading_inputs_buffer,
            shading_inputs_staging,
            screen_width,
            screen_height,
            max_instances,
            max_vertices,
            max_indices,
        }
    }

    /// Upload parameters to the GPU.
    pub fn upload_params(&self, queue: &Queue, params: &VisibilityReadParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload visibility data to the GPU.
    pub fn upload_visibility(&self, queue: &Queue, visibility: &[VisibilityData]) {
        let byte_len = visibility.len() * mem::size_of::<VisibilityData>();
        assert!(byte_len <= self.visibility_buffer.size() as usize);
        queue.write_buffer(&self.visibility_buffer, 0, bytemuck::cast_slice(visibility));
    }

    /// Upload instance transforms to the GPU.
    pub fn upload_instance_transforms(&self, queue: &Queue, transforms: &[InstanceTransform]) {
        let byte_len = transforms.len() * mem::size_of::<InstanceTransform>();
        assert!(byte_len <= self.instance_transforms_buffer.size() as usize);
        queue.write_buffer(
            &self.instance_transforms_buffer,
            0,
            bytemuck::cast_slice(transforms),
        );
    }

    /// Upload instance metadata to the GPU.
    pub fn upload_instance_metadata(&self, queue: &Queue, metadata: &[InstanceMetadata]) {
        let byte_len = metadata.len() * mem::size_of::<InstanceMetadata>();
        assert!(byte_len <= self.instance_metadata_buffer.size() as usize);
        queue.write_buffer(
            &self.instance_metadata_buffer,
            0,
            bytemuck::cast_slice(metadata),
        );
    }

    /// Upload vertex data to the GPU.
    pub fn upload_vertices(&self, queue: &Queue, vertices: &[VertexData]) {
        let byte_len = vertices.len() * mem::size_of::<VertexData>();
        assert!(byte_len <= self.vertex_buffer.size() as usize);
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
    }

    /// Upload index data to the GPU.
    pub fn upload_indices(&self, queue: &Queue, indices: &[u32]) {
        let byte_len = indices.len() * mem::size_of::<u32>();
        assert!(byte_len <= self.index_buffer.size() as usize);
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(indices));
    }

    /// Read shading inputs back to the CPU.
    ///
    /// This is a synchronous operation that waits for GPU completion.
    pub fn read_shading_inputs(&self, device: &Device, queue: &Queue) -> Vec<ShadingInput> {
        let pixel_count = (self.screen_width * self.screen_height) as usize;
        let byte_size = pixel_count * mem::size_of::<ShadingInput>();

        // Copy from GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_shading_inputs"),
        });
        encoder.copy_buffer_to_buffer(
            &self.shading_inputs_buffer,
            0,
            &self.shading_inputs_staging,
            0,
            byte_size as u64,
        );
        queue.submit([encoder.finish()]);

        // Map staging buffer and read
        let buffer_slice = self.shading_inputs_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let inputs: Vec<ShadingInput> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.shading_inputs_staging.unmap();

        inputs
    }

    /// Get the pixel count.
    #[inline]
    pub fn pixel_count(&self) -> u32 {
        self.screen_width * self.screen_height
    }

    /// Get the shading inputs buffer.
    #[inline]
    pub fn shading_inputs_buffer(&self) -> &Buffer {
        &self.shading_inputs_buffer
    }

    /// Get the visibility buffer.
    #[inline]
    pub fn visibility_buffer(&self) -> &Buffer {
        &self.visibility_buffer
    }
}

// =============================================================================
// VISIBILITY READ PIPELINE
// =============================================================================

/// Compute pipeline for visibility buffer read.
pub struct VisibilityReadPipeline {
    /// Main visibility read pipeline.
    read_pipeline: wgpu::ComputePipeline,
    /// Single-tile optimized pipeline.
    read_single_tile_pipeline: wgpu::ComputePipeline,
    /// Clear shading inputs pipeline.
    clear_pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl VisibilityReadPipeline {
    /// Create a new visibility read pipeline.
    pub fn new(device: &Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline_layout = Self::create_pipeline_layout(device, &bind_group_layout);

        let shader_source = include_str!("../../shaders/gpu_driven/gpu_visibility_read.comp.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_visibility_read_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let read_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("visibility_read_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "visibility_read",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let read_single_tile_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("visibility_read_single_tile_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "visibility_read_single_tile",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("visibility_read_clear_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "clear_shading_inputs",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            read_pipeline,
            read_single_tile_pipeline,
            clear_pipeline,
            bind_group_layout,
        }
    }

    /// Get the bind group layout.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &Device,
        resources: &VisibilityReadResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility_read_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.instance_transforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.instance_metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: resources.shading_inputs_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch visibility read.
    ///
    /// Uses tiled dispatch for optimal GPU utilization.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &VisibilityReadParams,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("visibility_read_pass"),
            timestamp_writes: None,
        });

        // For small screens, use single-tile variant
        if params.screen_width <= TILE_SIZE && params.screen_height <= TILE_SIZE {
            pass.set_pipeline(&self.read_single_tile_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        } else {
            pass.set_pipeline(&self.read_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(params.num_tiles_x(), params.num_tiles_y(), 1);
        }
    }

    /// Dispatch clear pass to initialize shading inputs.
    pub fn dispatch_clear(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &VisibilityReadParams,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("visibility_read_clear_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.clear_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(params.num_clear_workgroups(), 1, 1);
    }

    /// Create the bind group layout.
    fn create_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("visibility_read_bind_group_layout"),
            entries: &[
                // binding 0: params (uniform)
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
                // binding 1: visibility_buffer (storage, read)
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
                // binding 2: instance_transforms (storage, read)
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
                // binding 3: instance_metadata (storage, read)
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
                // binding 4: vertex_buffer (storage, read)
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
                // binding 5: index_buffer (storage, read)
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
                // binding 6: shading_inputs (storage, read_write)
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
        })
    }

    /// Create the pipeline layout.
    fn create_pipeline_layout(
        device: &Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::PipelineLayout {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("visibility_read_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        })
    }
}

// =============================================================================
// CPU REFERENCE IMPLEMENTATIONS
// =============================================================================

/// Interpolate a vec3 using barycentric coordinates (CPU reference).
#[inline]
pub fn cpu_interpolate_vec3(
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    bary_beta: f32,
    bary_gamma: f32,
) -> [f32; 3] {
    let alpha = 1.0 - bary_beta - bary_gamma;
    [
        v0[0] * alpha + v1[0] * bary_beta + v2[0] * bary_gamma,
        v0[1] * alpha + v1[1] * bary_beta + v2[1] * bary_gamma,
        v0[2] * alpha + v1[2] * bary_beta + v2[2] * bary_gamma,
    ]
}

/// Interpolate a vec2 using barycentric coordinates (CPU reference).
#[inline]
pub fn cpu_interpolate_vec2(
    v0: [f32; 2],
    v1: [f32; 2],
    v2: [f32; 2],
    bary_beta: f32,
    bary_gamma: f32,
) -> [f32; 2] {
    let alpha = 1.0 - bary_beta - bary_gamma;
    [
        v0[0] * alpha + v1[0] * bary_beta + v2[0] * bary_gamma,
        v0[1] * alpha + v1[1] * bary_beta + v2[1] * bary_gamma,
    ]
}

/// Interpolate a vec4 using barycentric coordinates (CPU reference).
#[inline]
pub fn cpu_interpolate_vec4(
    v0: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
    bary_beta: f32,
    bary_gamma: f32,
) -> [f32; 4] {
    let alpha = 1.0 - bary_beta - bary_gamma;
    [
        v0[0] * alpha + v1[0] * bary_beta + v2[0] * bary_gamma,
        v0[1] * alpha + v1[1] * bary_beta + v2[1] * bary_gamma,
        v0[2] * alpha + v1[2] * bary_beta + v2[2] * bary_gamma,
        v0[3] * alpha + v1[3] * bary_beta + v2[3] * bary_gamma,
    ]
}

/// Normalize a vector, handling denormalized inputs gracefully (CPU reference).
#[inline]
pub fn cpu_safe_normalize(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < EPSILON * EPSILON {
        return [0.0, 1.0, 0.0]; // Default up vector
    }
    let inv_len = 1.0 / len_sq.sqrt();
    [v[0] * inv_len, v[1] * inv_len, v[2] * inv_len]
}

/// Transform position from object space to world space (CPU reference).
pub fn cpu_transform_position(pos: [f32; 3], world_matrix: &[[f32; 4]; 4]) -> [f32; 3] {
    let w = world_matrix[0][3] * pos[0]
        + world_matrix[1][3] * pos[1]
        + world_matrix[2][3] * pos[2]
        + world_matrix[3][3];

    let inv_w = if w.abs() > EPSILON { 1.0 / w } else { 1.0 };

    [
        (world_matrix[0][0] * pos[0]
            + world_matrix[1][0] * pos[1]
            + world_matrix[2][0] * pos[2]
            + world_matrix[3][0])
            * inv_w,
        (world_matrix[0][1] * pos[0]
            + world_matrix[1][1] * pos[1]
            + world_matrix[2][1] * pos[2]
            + world_matrix[3][1])
            * inv_w,
        (world_matrix[0][2] * pos[0]
            + world_matrix[1][2] * pos[1]
            + world_matrix[2][2] * pos[2]
            + world_matrix[3][2])
            * inv_w,
    ]
}

/// Transform normal from object space to world space using normal matrix (CPU reference).
pub fn cpu_transform_normal(normal: [f32; 3], transform: &InstanceTransform) -> [f32; 3] {
    let transformed = [
        transform.normal_matrix_col0[0] * normal[0]
            + transform.normal_matrix_col1[0] * normal[1]
            + transform.normal_matrix_col2[0] * normal[2],
        transform.normal_matrix_col0[1] * normal[0]
            + transform.normal_matrix_col1[1] * normal[1]
            + transform.normal_matrix_col2[1] * normal[2],
        transform.normal_matrix_col0[2] * normal[0]
            + transform.normal_matrix_col1[2] * normal[1]
            + transform.normal_matrix_col2[2] * normal[2],
    ];
    cpu_safe_normalize(transformed)
}

/// Compute dot product of two vec3.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute cross product of two vec3.
#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Scale a vec3 by a scalar.
#[inline]
fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Subtract two vec3.
#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Transform tangent vector using the world matrix upper-left 3x3 (CPU reference).
pub fn cpu_transform_tangent(tangent: [f32; 3], world_matrix: &[[f32; 4]; 4]) -> [f32; 3] {
    let transformed = [
        world_matrix[0][0] * tangent[0]
            + world_matrix[1][0] * tangent[1]
            + world_matrix[2][0] * tangent[2],
        world_matrix[0][1] * tangent[0]
            + world_matrix[1][1] * tangent[1]
            + world_matrix[2][1] * tangent[2],
        world_matrix[0][2] * tangent[0]
            + world_matrix[1][2] * tangent[1]
            + world_matrix[2][2] * tangent[2],
    ];
    cpu_safe_normalize(transformed)
}

/// Compute orthonormal tangent space from normal and tangent (CPU reference).
pub fn cpu_compute_tangent_space(
    world_normal: [f32; 3],
    interpolated_tangent: [f32; 4],
    world_matrix: &[[f32; 4]; 4],
) -> ([f32; 3], [f32; 3]) {
    // Transform tangent to world space
    let tangent_xyz = [
        interpolated_tangent[0],
        interpolated_tangent[1],
        interpolated_tangent[2],
    ];
    let world_tangent_raw = cpu_transform_tangent(tangent_xyz, world_matrix);

    // Gram-Schmidt orthogonalization: remove normal component from tangent
    let n_dot_t = dot3(world_normal, world_tangent_raw);
    let world_tangent = cpu_safe_normalize(sub3(world_tangent_raw, scale3(world_normal, n_dot_t)));

    // Compute bitangent using handedness from tangent.w
    let bitangent_raw = cross3(world_normal, world_tangent);
    let bitangent = cpu_safe_normalize(scale3(bitangent_raw, interpolated_tangent[3]));

    (world_tangent, bitangent)
}

/// CPU reference implementation for visibility buffer read.
///
/// Processes a single pixel and returns the shading input.
pub fn cpu_visibility_read(
    vis: &VisibilityData,
    instance_transforms: &[InstanceTransform],
    instance_metadata: &[InstanceMetadata],
    vertices: &[VertexData],
    indices: &[u32],
) -> ShadingInput {
    if !vis.is_valid() {
        return ShadingInput::invalid();
    }

    let transform = &instance_transforms[vis.instance_id as usize];
    let metadata = &instance_metadata[vis.instance_id as usize];

    // Calculate triangle vertex indices
    let tri_base = (metadata.index_offset + vis.primitive_id * 3) as usize;
    let i0 = (indices[tri_base] + metadata.vertex_offset) as usize;
    let i1 = (indices[tri_base + 1] + metadata.vertex_offset) as usize;
    let i2 = (indices[tri_base + 2] + metadata.vertex_offset) as usize;

    // Fetch triangle vertices
    let v0 = &vertices[i0];
    let v1 = &vertices[i1];
    let v2 = &vertices[i2];

    // Interpolate vertex attributes
    let bary_beta = vis.barycentrics[0];
    let bary_gamma = vis.barycentrics[1];

    let obj_pos = cpu_interpolate_vec3(v0.position, v1.position, v2.position, bary_beta, bary_gamma);
    let obj_normal = cpu_interpolate_vec3(v0.normal, v1.normal, v2.normal, bary_beta, bary_gamma);
    let interp_uv = cpu_interpolate_vec2(v0.uv, v1.uv, v2.uv, bary_beta, bary_gamma);
    let interp_tangent =
        cpu_interpolate_vec4(v0.tangent, v1.tangent, v2.tangent, bary_beta, bary_gamma);

    // Transform to world space
    let world_pos = cpu_transform_position(obj_pos, &transform.world_matrix);
    let world_normal = cpu_transform_normal(obj_normal, transform);

    // Compute orthonormal tangent space
    let (tangent, bitangent) =
        cpu_compute_tangent_space(world_normal, interp_tangent, &transform.world_matrix);

    ShadingInput::new(
        world_pos,
        world_normal,
        interp_uv,
        vis.instance_id,
        metadata.material_id,
        tangent,
        bitangent,
    )
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
    fn test_visibility_read_params_size() {
        assert_eq!(mem::size_of::<VisibilityReadParams>(), 16);
    }

    #[test]
    fn test_visibility_data_size() {
        assert_eq!(mem::size_of::<VisibilityData>(), 16);
    }

    #[test]
    fn test_instance_transform_size() {
        assert_eq!(mem::size_of::<InstanceTransform>(), 112);
    }

    #[test]
    fn test_vertex_data_size() {
        assert_eq!(mem::size_of::<VertexData>(), 64);
    }

    #[test]
    fn test_instance_metadata_size() {
        assert_eq!(mem::size_of::<InstanceMetadata>(), 16);
    }

    #[test]
    fn test_shading_input_size() {
        assert_eq!(mem::size_of::<ShadingInput>(), 80);
    }

    #[test]
    fn test_structs_are_pod() {
        // These should compile without error if Pod is implemented correctly
        let params = VisibilityReadParams::new(1920, 1080);
        let _ = bytemuck::bytes_of(&params);

        let vis = VisibilityData::new(0, 0, 0.5, 0.25);
        let _ = bytemuck::bytes_of(&vis);

        let transform = InstanceTransform::identity();
        let _ = bytemuck::bytes_of(&transform);

        let vertex = VertexData::new([0.0; 3], [0.0, 1.0, 0.0], [0.0; 2], [1.0, 0.0, 0.0, 1.0]);
        let _ = bytemuck::bytes_of(&vertex);

        let metadata = InstanceMetadata::new(0, 0, 0);
        let _ = bytemuck::bytes_of(&metadata);

        let shading = ShadingInput::invalid();
        let _ = bytemuck::bytes_of(&shading);
    }

    // -------------------------------------------------------------------------
    // Barycentric Interpolation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_interpolate_vec3_at_vertex0() {
        let v0 = [1.0, 2.0, 3.0];
        let v1 = [4.0, 5.0, 6.0];
        let v2 = [7.0, 8.0, 9.0];

        // beta=0, gamma=0 => alpha=1, so we get v0
        let result = cpu_interpolate_vec3(v0, v1, v2, 0.0, 0.0);
        assert!((result[0] - 1.0).abs() < EPSILON);
        assert!((result[1] - 2.0).abs() < EPSILON);
        assert!((result[2] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_interpolate_vec3_at_vertex1() {
        let v0 = [1.0, 2.0, 3.0];
        let v1 = [4.0, 5.0, 6.0];
        let v2 = [7.0, 8.0, 9.0];

        // beta=1, gamma=0 => alpha=0, so we get v1
        let result = cpu_interpolate_vec3(v0, v1, v2, 1.0, 0.0);
        assert!((result[0] - 4.0).abs() < EPSILON);
        assert!((result[1] - 5.0).abs() < EPSILON);
        assert!((result[2] - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_interpolate_vec3_at_vertex2() {
        let v0 = [1.0, 2.0, 3.0];
        let v1 = [4.0, 5.0, 6.0];
        let v2 = [7.0, 8.0, 9.0];

        // beta=0, gamma=1 => alpha=0, so we get v2
        let result = cpu_interpolate_vec3(v0, v1, v2, 0.0, 1.0);
        assert!((result[0] - 7.0).abs() < EPSILON);
        assert!((result[1] - 8.0).abs() < EPSILON);
        assert!((result[2] - 9.0).abs() < EPSILON);
    }

    #[test]
    fn test_interpolate_vec3_centroid() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [3.0, 0.0, 0.0];
        let v2 = [0.0, 3.0, 0.0];

        // Centroid: alpha=beta=gamma=1/3
        let result = cpu_interpolate_vec3(v0, v1, v2, 1.0 / 3.0, 1.0 / 3.0);
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 1.0).abs() < 0.001);
        assert!((result[2] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_vec2_accuracy() {
        let v0 = [0.0, 0.0];
        let v1 = [1.0, 0.0];
        let v2 = [0.0, 1.0];

        let result = cpu_interpolate_vec2(v0, v1, v2, 0.5, 0.25);
        // Expected: (0.25 * 0 + 0.5 * 1 + 0.25 * 0, 0.25 * 0 + 0.5 * 0 + 0.25 * 1)
        assert!((result[0] - 0.5).abs() < EPSILON);
        assert!((result[1] - 0.25).abs() < EPSILON);
    }

    // -------------------------------------------------------------------------
    // World Position Reconstruction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transform_position_identity() {
        let pos = [1.0, 2.0, 3.0];
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let result = cpu_transform_position(pos, &identity);
        assert!((result[0] - 1.0).abs() < EPSILON);
        assert!((result[1] - 2.0).abs() < EPSILON);
        assert!((result[2] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_transform_position_translation() {
        let pos = [1.0, 2.0, 3.0];
        let translation = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ];

        let result = cpu_transform_position(pos, &translation);
        assert!((result[0] - 11.0).abs() < EPSILON);
        assert!((result[1] - 22.0).abs() < EPSILON);
        assert!((result[2] - 33.0).abs() < EPSILON);
    }

    #[test]
    fn test_transform_position_scale() {
        let pos = [1.0, 2.0, 3.0];
        let scale = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let result = cpu_transform_position(pos, &scale);
        assert!((result[0] - 2.0).abs() < EPSILON);
        assert!((result[1] - 6.0).abs() < EPSILON);
        assert!((result[2] - 12.0).abs() < EPSILON);
    }

    // -------------------------------------------------------------------------
    // Normal Transformation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transform_normal_identity() {
        let normal = [0.0, 1.0, 0.0];
        let transform = InstanceTransform::identity();

        let result = cpu_transform_normal(normal, &transform);
        assert!((result[0] - 0.0).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_transform_normal_with_scale() {
        // Non-uniform scale: the normal matrix should handle this correctly
        let normal = [0.0, 1.0, 0.0]; // Up normal
        let transform = InstanceTransform::scale_xyz(2.0, 1.0, 1.0);

        let result = cpu_transform_normal(normal, &transform);
        // With non-uniform scale in X, normal in Y should remain in Y
        assert!((result[0]).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2]).abs() < EPSILON);
    }

    #[test]
    fn test_transform_normal_is_normalized() {
        let normal = [1.0, 1.0, 1.0]; // Not normalized
        let transform = InstanceTransform::identity();

        let result = cpu_transform_normal(normal, &transform);
        let len = (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
        assert!((len - 1.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // UV Interpolation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_uv_interpolation_full_range() {
        let v0 = [0.0, 0.0];
        let v1 = [1.0, 0.0];
        let v2 = [1.0, 1.0];

        // At v0
        let r0 = cpu_interpolate_vec2(v0, v1, v2, 0.0, 0.0);
        assert!((r0[0] - 0.0).abs() < EPSILON);
        assert!((r0[1] - 0.0).abs() < EPSILON);

        // At v1
        let r1 = cpu_interpolate_vec2(v0, v1, v2, 1.0, 0.0);
        assert!((r1[0] - 1.0).abs() < EPSILON);
        assert!((r1[1] - 0.0).abs() < EPSILON);

        // At v2
        let r2 = cpu_interpolate_vec2(v0, v1, v2, 0.0, 1.0);
        assert!((r2[0] - 1.0).abs() < EPSILON);
        assert!((r2[1] - 1.0).abs() < EPSILON);
    }

    // -------------------------------------------------------------------------
    // Tangent/Bitangent Orthonormalization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_tangent_space_orthogonal() {
        let normal = [0.0, 1.0, 0.0];
        let tangent = [1.0, 0.0, 0.0, 1.0]; // Handedness +1
        let world_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let (t, b) = cpu_compute_tangent_space(normal, tangent, &world_matrix);

        // Tangent should be orthogonal to normal
        let n_dot_t = dot3(normal, t);
        assert!(n_dot_t.abs() < 0.001);

        // Bitangent should be orthogonal to both
        let n_dot_b = dot3(normal, b);
        let t_dot_b = dot3(t, b);
        assert!(n_dot_b.abs() < 0.001);
        assert!(t_dot_b.abs() < 0.001);
    }

    #[test]
    fn test_tangent_space_normalized() {
        let normal = [0.0, 1.0, 0.0];
        let tangent = [1.0, 0.1, 0.0, 1.0]; // Slightly non-orthogonal
        let world_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let (t, b) = cpu_compute_tangent_space(normal, tangent, &world_matrix);

        let t_len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        let b_len = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();

        assert!((t_len - 1.0).abs() < 0.001);
        assert!((b_len - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_tangent_space_handedness() {
        let normal = [0.0, 1.0, 0.0];
        let tangent_pos = [1.0, 0.0, 0.0, 1.0]; // Handedness +1
        let tangent_neg = [1.0, 0.0, 0.0, -1.0]; // Handedness -1
        let world_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let (_, b_pos) = cpu_compute_tangent_space(normal, tangent_pos, &world_matrix);
        let (_, b_neg) = cpu_compute_tangent_space(normal, tangent_neg, &world_matrix);

        // Bitangents should be opposite
        assert!((b_pos[0] + b_neg[0]).abs() < 0.001);
        assert!((b_pos[1] + b_neg[1]).abs() < 0.001);
        assert!((b_pos[2] + b_neg[2]).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Denormalized Normal Handling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_safe_normalize_zero_vector() {
        let zero = [0.0, 0.0, 0.0];
        let result = cpu_safe_normalize(zero);

        // Should return default up vector
        assert!((result[0] - 0.0).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_safe_normalize_tiny_vector() {
        let tiny = [1e-10, 1e-10, 1e-10];
        let result = cpu_safe_normalize(tiny);

        // Should return default up vector due to denormalization
        assert!((result[0] - 0.0).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] - 0.0).abs() < EPSILON);
    }

    // -------------------------------------------------------------------------
    // Visibility Data Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_visibility_data_invalid() {
        let vis = VisibilityData::invalid();
        assert!(!vis.is_valid());
        assert_eq!(vis.instance_id, INVALID_INSTANCE);
        assert_eq!(vis.primitive_id, INVALID_PRIMITIVE);
    }

    #[test]
    fn test_visibility_data_valid() {
        let vis = VisibilityData::new(42, 7, 0.3, 0.4);
        assert!(vis.is_valid());
        assert!((vis.bary_alpha() - 0.3).abs() < EPSILON);
        assert!((vis.bary_beta() - 0.3).abs() < EPSILON);
        assert!((vis.bary_gamma() - 0.4).abs() < EPSILON);
    }

    // -------------------------------------------------------------------------
    // Full Pipeline CPU Reference Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_visibility_read_full_pipeline() {
        // Set up test data
        let transforms = vec![InstanceTransform::translation(10.0, 0.0, 0.0)];
        let metadata = vec![InstanceMetadata::new(0, 0, 42)];

        // Simple triangle
        let vertices = vec![
            VertexData::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
            VertexData::new([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
            VertexData::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
        ];
        let indices = vec![0, 1, 2];

        // Hit at centroid
        let vis = VisibilityData::new(0, 0, 1.0 / 3.0, 1.0 / 3.0);

        let result = cpu_visibility_read(&vis, &transforms, &metadata, &vertices, &indices);

        assert!(result.is_valid());
        assert_eq!(result.instance_id, 0);
        assert_eq!(result.material_id, 42);

        // World position should be centroid + translation
        // Centroid of triangle is (1/3, 1/3, 0), translated by (10, 0, 0)
        assert!((result.world_pos[0] - (1.0 / 3.0 + 10.0)).abs() < 0.01);
        assert!((result.world_pos[1] - 1.0 / 3.0).abs() < 0.01);
        assert!((result.world_pos[2] - 0.0).abs() < 0.01);

        // Normal should be +Z
        assert!((result.world_normal[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cpu_visibility_read_invalid_pixel() {
        let transforms = vec![InstanceTransform::identity()];
        let metadata = vec![InstanceMetadata::new(0, 0, 0)];
        let vertices = vec![VertexData::default(); 3];
        let indices = vec![0, 1, 2];

        let vis = VisibilityData::invalid();
        let result = cpu_visibility_read(&vis, &transforms, &metadata, &vertices, &indices);

        assert!(!result.is_valid());
        assert_eq!(result.instance_id, INVALID_INSTANCE);
    }

    // -------------------------------------------------------------------------
    // Shader Validation Tests (using naga)
    // -------------------------------------------------------------------------

    #[test]
    fn test_visibility_read_shader_parses() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_visibility_read.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("visibility read shader should parse without errors");

        // Verify expected entry points exist
        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "visibility_read"),
            "Should have visibility_read entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "visibility_read_single_tile"),
            "Should have visibility_read_single_tile entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "clear_shading_inputs"),
            "Should have clear_shading_inputs entry point"
        );
    }

    #[test]
    fn test_visibility_read_shader_validates() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_visibility_read.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("visibility read shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("visibility read shader should validate without errors");
    }

    #[test]
    fn test_visibility_read_shader_workgroup_size() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_visibility_read.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("visibility read shader should parse without errors");

        for ep in &module.entry_points {
            if ep.stage == naga::ShaderStage::Compute {
                if ep.name == "visibility_read" || ep.name == "visibility_read_single_tile" {
                    assert_eq!(
                        ep.workgroup_size,
                        [16, 16, 1],
                        "Entry point {} should have workgroup size 16x16x1",
                        ep.name
                    );
                } else if ep.name == "clear_shading_inputs" {
                    assert_eq!(
                        ep.workgroup_size,
                        [256, 1, 1],
                        "Entry point {} should have workgroup size 256x1x1",
                        ep.name
                    );
                }
            }
        }
    }

    #[test]
    fn test_visibility_read_shader_entry_points_are_compute() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_visibility_read.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("visibility read shader should parse without errors");

        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                naga::ShaderStage::Compute,
                "Entry point {} should be a compute shader",
                ep.name
            );
        }
    }

    // -------------------------------------------------------------------------
    // VisibilityReadParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_params_tile_counts() {
        let params = VisibilityReadParams::new(1920, 1080);
        assert_eq!(params.num_tiles_x(), 120); // 1920 / 16 = 120
        assert_eq!(params.num_tiles_y(), 68); // ceil(1080 / 16) = 68
    }

    #[test]
    fn test_params_clear_workgroups() {
        let params = VisibilityReadParams::new(100, 100);
        let pixel_count = 100 * 100;
        let expected = (pixel_count + 255) / 256;
        assert_eq!(params.num_clear_workgroups(), expected);
    }
}
