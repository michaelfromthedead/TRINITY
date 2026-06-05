//! Dynamic uniform buffer support for TRINITY.
//!
//! This module provides helpers for dynamic uniform buffers, which allow multiple
//! objects to share a single buffer with different data at aligned offsets.
//!
//! # Overview
//!
//! Dynamic uniform buffers are essential for efficient rendering of multiple objects.
//! Instead of creating separate uniform buffers for each object's transform, all
//! transforms are packed into a single buffer with proper alignment.
//!
//! # WebGPU Alignment Requirements
//!
//! The WebGPU specification requires uniform buffer offsets to be aligned to at least
//! 256 bytes (the `minUniformBufferOffsetAlignment` limit). This module provides
//! helpers to calculate these aligned offsets.
//!
//! # Architecture
//!
//! ```text
//! +------------------+------------------+------------------+
//! | Object 0 Data    | Object 1 Data    | Object 2 Data    |
//! | (padded to 256B) | (padded to 256B) | (padded to 256B) |
//! +------------------+------------------+------------------+
//! ^                  ^                  ^
//! offset=0           offset=256        offset=512
//! ```
//!
//! When binding, pass the object's offset as a dynamic offset to `set_bind_group`.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::uniform::{
//!     UNIFORM_ALIGNMENT, align_uniform_offset, dynamic_offset_for_object,
//!     uniform_buffer_size_for_objects, ObjectTransform,
//! };
//!
//! // Calculate buffer size for 100 objects
//! let buffer_size = uniform_buffer_size_for_objects(100, ObjectTransform::SIZE);
//!
//! // Get offset for object #42
//! let offset = dynamic_offset_for_object(42, ObjectTransform::SIZE);
//!
//! // In render pass:
//! // render_pass.set_bind_group(1, &bind_group, &[offset]);
//! ```

use wgpu::{BindingType, BufferBindingType, DynamicOffset};

// ============================================================================
// Constants
// ============================================================================

/// Minimum uniform buffer offset alignment (256 bytes per WebGPU spec).
///
/// This is the `minUniformBufferOffsetAlignment` limit from the WebGPU specification.
/// All dynamic offsets must be multiples of this value.
///
/// Note: Some implementations may support smaller alignments, but 256 is the
/// guaranteed minimum that works everywhere.
pub const UNIFORM_ALIGNMENT: u64 = 256;

// ============================================================================
// Alignment Helpers
// ============================================================================

/// Calculate aligned offset for dynamic uniform buffers.
///
/// Rounds the given offset up to the nearest multiple of [`UNIFORM_ALIGNMENT`].
///
/// # Arguments
///
/// * `offset` - The raw offset in bytes
///
/// # Returns
///
/// The aligned offset, which is >= `offset` and a multiple of 256.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::uniform::align_uniform_offset;
///
/// assert_eq!(align_uniform_offset(0), 0);
/// assert_eq!(align_uniform_offset(1), 256);
/// assert_eq!(align_uniform_offset(48), 256);
/// assert_eq!(align_uniform_offset(256), 256);
/// assert_eq!(align_uniform_offset(300), 512);
/// ```
#[inline]
pub const fn align_uniform_offset(offset: u64) -> u64 {
    // Handle zero case: 0 should remain 0
    if offset == 0 {
        return 0;
    }
    // Round up to next multiple of UNIFORM_ALIGNMENT
    (offset + UNIFORM_ALIGNMENT - 1) & !(UNIFORM_ALIGNMENT - 1)
}

/// Calculate the aligned size for uniform data.
///
/// Given the actual data size, returns the padded size that satisfies alignment.
/// This is used to calculate stride between objects in a dynamic uniform buffer.
///
/// # Arguments
///
/// * `data_size` - The actual size of the uniform data in bytes
///
/// # Returns
///
/// The aligned size, which is >= `data_size` and a multiple of 256.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::uniform::aligned_uniform_size;
///
/// // A 64-byte struct needs 256 bytes with alignment
/// assert_eq!(aligned_uniform_size(64), 256);
///
/// // A 256-byte struct stays 256 bytes
/// assert_eq!(aligned_uniform_size(256), 256);
///
/// // A 300-byte struct needs 512 bytes
/// assert_eq!(aligned_uniform_size(300), 512);
/// ```
#[inline]
pub const fn aligned_uniform_size(data_size: u64) -> u64 {
    if data_size == 0 {
        return 0;
    }
    (data_size + UNIFORM_ALIGNMENT - 1) & !(UNIFORM_ALIGNMENT - 1)
}

/// Calculate dynamic offset for the N-th object in a uniform buffer.
///
/// Given an object index and per-object data size, returns the byte offset
/// that satisfies uniform buffer alignment requirements.
///
/// # Arguments
///
/// * `object_index` - Zero-based index of the object (0, 1, 2, ...)
/// * `data_size` - The actual size of each object's data in bytes
///
/// # Returns
///
/// The byte offset for the given object, suitable for use with `set_bind_group`.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::uniform::dynamic_offset_for_object;
///
/// // Each object uses 64 bytes, aligned to 256
/// assert_eq!(dynamic_offset_for_object(0, 64), 0);
/// assert_eq!(dynamic_offset_for_object(1, 64), 256);
/// assert_eq!(dynamic_offset_for_object(2, 64), 512);
/// assert_eq!(dynamic_offset_for_object(10, 64), 2560);
/// ```
#[inline]
pub const fn dynamic_offset_for_object(object_index: u32, data_size: u64) -> DynamicOffset {
    let aligned_size = aligned_uniform_size(data_size);
    (object_index as u64 * aligned_size) as DynamicOffset
}

/// Calculate required buffer size for N objects with alignment.
///
/// This determines the minimum buffer size needed to store `object_count` objects,
/// each with `data_size` bytes of data, properly aligned for dynamic uniform access.
///
/// # Arguments
///
/// * `object_count` - Number of objects to store
/// * `data_size` - Actual size of each object's data in bytes
///
/// # Returns
///
/// Total buffer size in bytes needed for all objects.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::uniform::uniform_buffer_size_for_objects;
///
/// // 100 objects at 64 bytes each = 100 * 256 = 25600 bytes
/// assert_eq!(uniform_buffer_size_for_objects(100, 64), 25600);
///
/// // 1 object needs at least 256 bytes
/// assert_eq!(uniform_buffer_size_for_objects(1, 64), 256);
///
/// // 0 objects needs 0 bytes
/// assert_eq!(uniform_buffer_size_for_objects(0, 64), 0);
/// ```
#[inline]
pub const fn uniform_buffer_size_for_objects(object_count: u32, data_size: u64) -> u64 {
    let aligned_size = aligned_uniform_size(data_size);
    object_count as u64 * aligned_size
}

// ============================================================================
// Binding Type Helpers
// ============================================================================

/// Create a uniform binding type with dynamic offset support.
///
/// Use this when creating bind group layouts for dynamic uniform buffers
/// where the offset will be specified at bind time via `set_bind_group`.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::uniform::uniform_binding_type_dynamic;
/// use wgpu::BindGroupLayoutEntry;
///
/// let entry = BindGroupLayoutEntry {
///     binding: 0,
///     visibility: wgpu::ShaderStages::VERTEX,
///     ty: uniform_binding_type_dynamic(),
///     count: None,
/// };
/// ```
#[inline]
pub const fn uniform_binding_type_dynamic() -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: true,
        min_binding_size: None,
    }
}

/// Create a uniform binding type without dynamic offset (static binding).
///
/// Use this when the buffer offset is fixed at bind group creation time.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::uniform::uniform_binding_type_static;
/// use wgpu::BindGroupLayoutEntry;
///
/// let entry = BindGroupLayoutEntry {
///     binding: 0,
///     visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
///     ty: uniform_binding_type_static(),
///     count: None,
/// };
/// ```
#[inline]
pub const fn uniform_binding_type_static() -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

// ============================================================================
// Standard Transform Types
// ============================================================================

/// Standard transform data for per-object uniforms.
///
/// This struct represents the typical per-object data needed for rendering:
/// - Model matrix for world-space transformation
/// - Normal matrix for correct lighting (inverse-transpose of upper-left 3x3)
/// - Object ID for picking and debugging
///
/// # Memory Layout (std140/std430 compatible)
///
/// ```text
/// Offset  Size  Field
/// 0       64    model (mat4x4<f32>)
/// 64      48    normal (mat3x3 padded to mat3x4 for std140)
/// 112     4     object_id (u32)
/// 116     12    _padding (align to 16 bytes)
/// ----
/// 128 bytes total
/// ```
///
/// Note: This struct is 128 bytes, but when used in a dynamic uniform buffer,
/// each instance will be aligned to 256 bytes. The alignment helpers handle this.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::uniform::ObjectTransform;
///
/// let transform = ObjectTransform::identity();
/// assert_eq!(ObjectTransform::SIZE, 128);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectTransform {
    /// Model matrix (mat4x4<f32>) - transforms from local to world space.
    pub model: [[f32; 4]; 4],

    /// Normal matrix (mat3x3 padded to mat3x4 for std140 layout).
    /// This is the inverse-transpose of the upper-left 3x3 of the model matrix.
    /// The 4th column of each row is padding.
    pub normal: [[f32; 4]; 3],

    /// Object ID for picking, debugging, or instance identification.
    pub object_id: u32,

    /// Padding to align struct to 16 bytes (GPU struct alignment).
    pub _padding: [u32; 3],
}

impl ObjectTransform {
    /// Size of ObjectTransform in bytes (128 bytes).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Create an identity transform.
    ///
    /// Returns a transform with identity model matrix, identity normal matrix,
    /// and object_id of 0.
    #[inline]
    pub const fn identity() -> Self {
        Self {
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            normal: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            object_id: 0,
            _padding: [0; 3],
        }
    }

    /// Create a transform with the given model matrix and object ID.
    ///
    /// Note: This sets the normal matrix to identity. For correct lighting with
    /// non-uniform scaling, you should compute the proper normal matrix
    /// (inverse-transpose of upper-left 3x3).
    #[inline]
    pub const fn with_model(model: [[f32; 4]; 4], object_id: u32) -> Self {
        Self {
            model,
            normal: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            object_id,
            _padding: [0; 3],
        }
    }

    /// Create a transform with full control over all fields.
    #[inline]
    pub const fn new(
        model: [[f32; 4]; 4],
        normal: [[f32; 4]; 3],
        object_id: u32,
    ) -> Self {
        Self {
            model,
            normal,
            object_id,
            _padding: [0; 3],
        }
    }
}

impl Default for ObjectTransform {
    fn default() -> Self {
        Self::identity()
    }
}

// ============================================================================
// View/Projection Uniform
// ============================================================================

/// Camera/view uniform data.
///
/// This struct represents the per-frame camera data typically bound at
/// binding group 0 (global uniforms).
///
/// # Memory Layout (std140/std430 compatible)
///
/// ```text
/// Offset  Size  Field
/// 0       64    view (mat4x4<f32>)
/// 64      64    projection (mat4x4<f32>)
/// 128     64    view_projection (mat4x4<f32>)
/// 192     16    camera_position (vec4<f32>, w unused)
/// 208     16    viewport (vec4<f32>: width, height, near, far)
/// 224     4     frame_index (u32)
/// 228     4     time (f32)
/// 232     24    _padding
/// ----
/// 256 bytes total (aligned to UNIFORM_ALIGNMENT)
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// View matrix (world to camera space).
    pub view: [[f32; 4]; 4],

    /// Projection matrix (camera to clip space).
    pub projection: [[f32; 4]; 4],

    /// Pre-multiplied view-projection matrix.
    pub view_projection: [[f32; 4]; 4],

    /// Camera position in world space (xyz), w is unused.
    pub camera_position: [f32; 4],

    /// Viewport parameters: width, height, near plane, far plane.
    pub viewport: [f32; 4],

    /// Frame index (monotonically increasing).
    pub frame_index: u32,

    /// Time in seconds since start.
    pub time: f32,

    /// Padding to align to 256 bytes.
    pub _padding: [f32; 6],
}

impl CameraUniform {
    /// Size of CameraUniform in bytes (256 bytes, aligned).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Create a default camera uniform (identity matrices, origin camera).
    pub const fn identity() -> Self {
        Self {
            view: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            projection: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            view_projection: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_position: [0.0, 0.0, 0.0, 1.0],
            viewport: [1920.0, 1080.0, 0.1, 1000.0],
            frame_index: 0,
            time: 0.0,
            _padding: [0.0; 6],
        }
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::identity()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_alignment_constant() {
        // WebGPU spec requires 256 byte alignment
        assert_eq!(UNIFORM_ALIGNMENT, 256);
        // Must be a power of 2
        assert!(UNIFORM_ALIGNMENT.is_power_of_two());
    }

    #[test]
    fn test_align_uniform_offset() {
        // Zero stays zero
        assert_eq!(align_uniform_offset(0), 0);

        // Small values round up to 256
        assert_eq!(align_uniform_offset(1), 256);
        assert_eq!(align_uniform_offset(48), 256);
        assert_eq!(align_uniform_offset(128), 256);
        assert_eq!(align_uniform_offset(255), 256);

        // Exact multiples stay the same
        assert_eq!(align_uniform_offset(256), 256);
        assert_eq!(align_uniform_offset(512), 512);
        assert_eq!(align_uniform_offset(1024), 1024);

        // Values between multiples round up
        assert_eq!(align_uniform_offset(257), 512);
        assert_eq!(align_uniform_offset(300), 512);
        assert_eq!(align_uniform_offset(511), 512);
    }

    #[test]
    fn test_aligned_uniform_size() {
        // Zero stays zero
        assert_eq!(aligned_uniform_size(0), 0);

        // Small sizes round up to 256
        assert_eq!(aligned_uniform_size(64), 256);
        assert_eq!(aligned_uniform_size(128), 256);
        assert_eq!(aligned_uniform_size(200), 256);

        // Exact 256 stays 256
        assert_eq!(aligned_uniform_size(256), 256);

        // Larger sizes round up appropriately
        assert_eq!(aligned_uniform_size(257), 512);
        assert_eq!(aligned_uniform_size(300), 512);
        assert_eq!(aligned_uniform_size(512), 512);
        assert_eq!(aligned_uniform_size(513), 768);
    }

    #[test]
    fn test_dynamic_offset_for_object() {
        // Object 0 is always at offset 0
        assert_eq!(dynamic_offset_for_object(0, 64), 0);
        assert_eq!(dynamic_offset_for_object(0, 128), 0);
        assert_eq!(dynamic_offset_for_object(0, 256), 0);

        // Small data (64 bytes) - aligned to 256
        assert_eq!(dynamic_offset_for_object(1, 64), 256);
        assert_eq!(dynamic_offset_for_object(2, 64), 512);
        assert_eq!(dynamic_offset_for_object(10, 64), 2560);

        // Data exactly 256 bytes
        assert_eq!(dynamic_offset_for_object(1, 256), 256);
        assert_eq!(dynamic_offset_for_object(2, 256), 512);
        assert_eq!(dynamic_offset_for_object(5, 256), 1280);

        // Larger data (300 bytes) - aligned to 512
        assert_eq!(dynamic_offset_for_object(1, 300), 512);
        assert_eq!(dynamic_offset_for_object(2, 300), 1024);
    }

    #[test]
    fn test_uniform_buffer_size_for_objects() {
        // Zero objects needs zero bytes
        assert_eq!(uniform_buffer_size_for_objects(0, 64), 0);
        assert_eq!(uniform_buffer_size_for_objects(0, 256), 0);

        // One object needs one aligned block
        assert_eq!(uniform_buffer_size_for_objects(1, 64), 256);
        assert_eq!(uniform_buffer_size_for_objects(1, 256), 256);
        assert_eq!(uniform_buffer_size_for_objects(1, 300), 512);

        // Multiple objects
        assert_eq!(uniform_buffer_size_for_objects(10, 64), 2560);
        assert_eq!(uniform_buffer_size_for_objects(100, 64), 25600);
        assert_eq!(uniform_buffer_size_for_objects(100, 256), 25600);

        // Large data per object
        assert_eq!(uniform_buffer_size_for_objects(10, 300), 5120); // 10 * 512
    }

    #[test]
    fn test_object_transform_size() {
        // ObjectTransform should be 128 bytes
        assert_eq!(std::mem::size_of::<ObjectTransform>(), 128);
        assert_eq!(ObjectTransform::SIZE, 128);

        // Verify individual field sizes
        // model: 4x4 f32 = 64 bytes
        // normal: 3x4 f32 = 48 bytes
        // object_id: 4 bytes
        // _padding: 12 bytes
        // Total: 128 bytes
    }

    #[test]
    fn test_object_transform_identity() {
        let transform = ObjectTransform::identity();

        // Check identity model matrix
        assert_eq!(transform.model[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(transform.model[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(transform.model[2], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(transform.model[3], [0.0, 0.0, 0.0, 1.0]);

        // Check identity normal matrix
        assert_eq!(transform.normal[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(transform.normal[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(transform.normal[2], [0.0, 0.0, 1.0, 0.0]);

        // Check defaults
        assert_eq!(transform.object_id, 0);
        assert_eq!(transform._padding, [0, 0, 0]);
    }

    #[test]
    fn test_object_transform_with_model() {
        let model = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ];
        let transform = ObjectTransform::with_model(model, 42);

        assert_eq!(transform.model, model);
        assert_eq!(transform.object_id, 42);
        // Normal should be identity (user should compute proper normal matrix)
        assert_eq!(transform.normal[0], [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_object_transform_new() {
        let model = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let normal = [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
        ];
        let transform = ObjectTransform::new(model, normal, 123);

        assert_eq!(transform.model, model);
        assert_eq!(transform.normal, normal);
        assert_eq!(transform.object_id, 123);
    }

    #[test]
    fn test_object_transform_default() {
        let transform: ObjectTransform = Default::default();
        let identity = ObjectTransform::identity();

        assert_eq!(transform.model, identity.model);
        assert_eq!(transform.normal, identity.normal);
        assert_eq!(transform.object_id, identity.object_id);
    }

    #[test]
    fn test_object_transform_bytemuck() {
        // Verify bytemuck traits work correctly
        let transform = ObjectTransform::identity();

        // Cast to bytes
        let bytes: &[u8] = bytemuck::bytes_of(&transform);
        assert_eq!(bytes.len(), 128);

        // Cast back
        let recovered: &ObjectTransform = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.model, transform.model);
        assert_eq!(recovered.object_id, transform.object_id);
    }

    #[test]
    fn test_camera_uniform_size() {
        // CameraUniform should be exactly 256 bytes (aligned)
        assert_eq!(std::mem::size_of::<CameraUniform>(), 256);
        assert_eq!(CameraUniform::SIZE, 256);
    }

    #[test]
    fn test_camera_uniform_identity() {
        let camera = CameraUniform::identity();

        // Check identity matrices
        assert_eq!(camera.view[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(camera.projection[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(camera.view_projection[0], [1.0, 0.0, 0.0, 0.0]);

        // Check defaults
        assert_eq!(camera.camera_position, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(camera.viewport, [1920.0, 1080.0, 0.1, 1000.0]);
        assert_eq!(camera.frame_index, 0);
        assert_eq!(camera.time, 0.0);
    }

    #[test]
    fn test_camera_uniform_bytemuck() {
        let camera = CameraUniform::identity();
        let bytes: &[u8] = bytemuck::bytes_of(&camera);
        assert_eq!(bytes.len(), 256);
    }

    #[test]
    fn test_binding_type_dynamic() {
        let binding = uniform_binding_type_dynamic();

        match binding {
            BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(matches!(ty, BufferBindingType::Uniform));
                assert!(has_dynamic_offset);
                assert!(min_binding_size.is_none());
            }
            _ => panic!("Expected Buffer binding type"),
        }
    }

    #[test]
    fn test_binding_type_static() {
        let binding = uniform_binding_type_static();

        match binding {
            BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(matches!(ty, BufferBindingType::Uniform));
                assert!(!has_dynamic_offset);
                assert!(min_binding_size.is_none());
            }
            _ => panic!("Expected Buffer binding type"),
        }
    }

    #[test]
    fn test_alignment_consistency() {
        // Verify that offset calculation and size calculation are consistent
        for size in [64u64, 128, 200, 256, 300, 400, 512] {
            for count in 1..10u32 {
                let buffer_size = uniform_buffer_size_for_objects(count, size);
                let last_offset = dynamic_offset_for_object(count - 1, size);
                let last_aligned_size = aligned_uniform_size(size);

                // Buffer size should accommodate the last object plus its data
                assert!(
                    buffer_size >= last_offset as u64 + last_aligned_size,
                    "Buffer too small for {} objects of size {}: buffer_size={}, last_offset={}, aligned_size={}",
                    count, size, buffer_size, last_offset, last_aligned_size
                );
            }
        }
    }
}
