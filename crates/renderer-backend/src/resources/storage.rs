//! Storage buffer support for TRINITY.
//!
//! This module provides helpers for storage buffers, which allow shaders to read
//! and/or write arbitrary structured data. Unlike uniform buffers, storage buffers
//! can be much larger and support read-write access from compute shaders.
//!
//! # Overview
//!
//! Storage buffers are essential for:
//! - GPU-driven rendering (instance data, draw commands)
//! - Compute shader workloads (particle systems, simulations)
//! - Large data sets that exceed uniform buffer limits
//! - Read-write access patterns (accumulation, atomics)
//!
//! # Alignment Requirements
//!
//! Storage buffers have more relaxed alignment than uniform buffers:
//! - Minimum storage buffer offset alignment: typically 16-256 bytes (device-dependent)
//! - Struct alignment follows std430 rules (vec4 = 16 bytes)
//!
//! For maximum compatibility, this module uses 16-byte alignment (vec4 size).
//!
//! # Read-Only vs Read-Write
//!
//! ```wgsl
//! // Read-only storage (can be shared across invocations)
//! @group(0) @binding(0) var<storage, read> instances: array<InstanceData>;
//!
//! // Read-write storage (for compute shaders)
//! @group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::storage::{
//!     storage_binding_type_readonly, storage_binding_type_readwrite,
//!     StorageHeader, STORAGE_ALIGNMENT,
//! };
//! use wgpu::BindGroupLayoutEntry;
//!
//! // Create a read-only storage binding for instance data
//! let instance_binding = BindGroupLayoutEntry {
//!     binding: 0,
//!     visibility: wgpu::ShaderStages::VERTEX,
//!     ty: storage_binding_type_readonly(),
//!     count: None,
//! };
//!
//! // Create a read-write storage binding for compute output
//! let output_binding = BindGroupLayoutEntry {
//!     binding: 1,
//!     visibility: wgpu::ShaderStages::COMPUTE,
//!     ty: storage_binding_type_readwrite(),
//!     count: None,
//! };
//! ```

use std::num::NonZeroU64;
use wgpu::{BindingType, BufferBindingType};

// ============================================================================
// Constants
// ============================================================================

/// Minimum storage buffer alignment (16 bytes for vec4).
///
/// While the WebGPU spec allows device-specific alignment requirements for
/// storage buffers (typically 16-256 bytes), 16 bytes (vec4 size) is the
/// most common and provides good compatibility.
///
/// For dynamic offset storage buffers, use [`STORAGE_DYNAMIC_ALIGNMENT`]
/// which may be larger on some devices.
pub const STORAGE_ALIGNMENT: u64 = 16;

/// Minimum alignment for storage buffer dynamic offsets.
///
/// This is the `minStorageBufferOffsetAlignment` limit from WebGPU.
/// The actual value is device-dependent, but 256 bytes is a safe maximum
/// that works on all conformant implementations.
///
/// Note: Query `device.limits().min_storage_buffer_offset_alignment` for
/// the actual device limit.
pub const STORAGE_DYNAMIC_ALIGNMENT: u64 = 256;

// ============================================================================
// Alignment Helpers
// ============================================================================

/// Align a size to storage buffer requirements.
///
/// Rounds up to the nearest multiple of [`STORAGE_ALIGNMENT`] (16 bytes).
///
/// # Arguments
///
/// * `size` - The size in bytes to align
///
/// # Returns
///
/// The aligned size, which is >= `size` and a multiple of 16.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::storage::align_storage_size;
///
/// assert_eq!(align_storage_size(0), 0);
/// assert_eq!(align_storage_size(1), 16);
/// assert_eq!(align_storage_size(16), 16);
/// assert_eq!(align_storage_size(17), 32);
/// assert_eq!(align_storage_size(48), 48);
/// ```
#[inline]
pub const fn align_storage_size(size: u64) -> u64 {
    if size == 0 {
        return 0;
    }
    (size + STORAGE_ALIGNMENT - 1) & !(STORAGE_ALIGNMENT - 1)
}

/// Align an offset for dynamic storage buffer binding.
///
/// Rounds up to the nearest multiple of [`STORAGE_DYNAMIC_ALIGNMENT`] (256 bytes).
///
/// # Arguments
///
/// * `offset` - The offset in bytes to align
///
/// # Returns
///
/// The aligned offset, which is >= `offset` and a multiple of 256.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::storage::align_storage_dynamic_offset;
///
/// assert_eq!(align_storage_dynamic_offset(0), 0);
/// assert_eq!(align_storage_dynamic_offset(1), 256);
/// assert_eq!(align_storage_dynamic_offset(256), 256);
/// assert_eq!(align_storage_dynamic_offset(300), 512);
/// ```
#[inline]
pub const fn align_storage_dynamic_offset(offset: u64) -> u64 {
    if offset == 0 {
        return 0;
    }
    (offset + STORAGE_DYNAMIC_ALIGNMENT - 1) & !(STORAGE_DYNAMIC_ALIGNMENT - 1)
}

/// Calculate required buffer size for N elements with alignment.
///
/// This determines the minimum buffer size needed to store `element_count`
/// elements, each with `element_size` bytes, properly aligned.
///
/// # Arguments
///
/// * `element_count` - Number of elements to store
/// * `element_size` - Size of each element in bytes
///
/// # Returns
///
/// Total buffer size in bytes needed for all elements.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::storage::storage_buffer_size;
///
/// // 100 elements at 64 bytes each
/// assert_eq!(storage_buffer_size(100, 64), 6400);
///
/// // Elements need to be aligned to 16 bytes
/// assert_eq!(storage_buffer_size(100, 12), 1600); // 12 -> 16 bytes each
/// ```
#[inline]
pub const fn storage_buffer_size(element_count: u32, element_size: u64) -> u64 {
    let aligned_size = align_storage_size(element_size);
    element_count as u64 * aligned_size
}

// ============================================================================
// Binding Type Helpers - Basic
// ============================================================================

/// Create a read-only storage binding type.
///
/// Use this for storage buffers that shaders only read from, such as:
/// - Instance data arrays
/// - Lookup tables
/// - Pre-computed data
///
/// Read-only storage buffers can be more efficiently cached by the GPU.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::storage::storage_binding_type_readonly;
/// use wgpu::BindGroupLayoutEntry;
///
/// let entry = BindGroupLayoutEntry {
///     binding: 0,
///     visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
///     ty: storage_binding_type_readonly(),
///     count: None,
/// };
/// ```
#[inline]
pub const fn storage_binding_type_readonly() -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

/// Create a read-write storage binding type.
///
/// Use this for storage buffers that shaders both read from and write to:
/// - Compute shader outputs
/// - Particle system state
/// - Accumulation buffers
/// - Atomic counters
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::storage::storage_binding_type_readwrite;
/// use wgpu::BindGroupLayoutEntry;
///
/// let entry = BindGroupLayoutEntry {
///     binding: 0,
///     visibility: wgpu::ShaderStages::COMPUTE,
///     ty: storage_binding_type_readwrite(),
///     count: None,
/// };
/// ```
#[inline]
pub const fn storage_binding_type_readwrite() -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

// ============================================================================
// Binding Type Helpers - With min_binding_size
// ============================================================================

/// Create a read-only storage binding type with minimum binding size.
///
/// Specifying `min_binding_size` enables validation that the buffer is large
/// enough, and can improve performance by allowing the driver to optimize.
///
/// # Arguments
///
/// * `size` - Minimum required buffer size in bytes. Must be > 0.
///
/// # Panics
///
/// Panics if `size` is 0 (use [`storage_binding_type_readonly`] instead).
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::storage::storage_binding_type_readonly_sized;
/// use wgpu::BindGroupLayoutEntry;
///
/// // Require at least 1024 bytes
/// let entry = BindGroupLayoutEntry {
///     binding: 0,
///     visibility: wgpu::ShaderStages::VERTEX,
///     ty: storage_binding_type_readonly_sized(1024),
///     count: None,
/// };
/// ```
#[inline]
pub fn storage_binding_type_readonly_sized(size: u64) -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: Some(
            NonZeroU64::new(size).expect("min_binding_size must be > 0"),
        ),
    }
}

/// Create a read-write storage binding type with minimum binding size.
///
/// Specifying `min_binding_size` enables validation that the buffer is large
/// enough, and can improve performance by allowing the driver to optimize.
///
/// # Arguments
///
/// * `size` - Minimum required buffer size in bytes. Must be > 0.
///
/// # Panics
///
/// Panics if `size` is 0 (use [`storage_binding_type_readwrite`] instead).
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::storage::storage_binding_type_readwrite_sized;
/// use wgpu::BindGroupLayoutEntry;
///
/// // Require at least 4096 bytes for compute output
/// let entry = BindGroupLayoutEntry {
///     binding: 0,
///     visibility: wgpu::ShaderStages::COMPUTE,
///     ty: storage_binding_type_readwrite_sized(4096),
///     count: None,
/// };
/// ```
#[inline]
pub fn storage_binding_type_readwrite_sized(size: u64) -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: Some(
            NonZeroU64::new(size).expect("min_binding_size must be > 0"),
        ),
    }
}

// ============================================================================
// Binding Type Helpers - Dynamic Offset
// ============================================================================

/// Create a read-only storage binding type with dynamic offset.
///
/// Use this when multiple data sets are packed into a single buffer and
/// the offset will be specified at bind time via `set_bind_group`.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::storage::storage_binding_type_dynamic_readonly;
/// use wgpu::BindGroupLayoutEntry;
///
/// let entry = BindGroupLayoutEntry {
///     binding: 0,
///     visibility: wgpu::ShaderStages::VERTEX,
///     ty: storage_binding_type_dynamic_readonly(),
///     count: None,
/// };
///
/// // In render pass:
/// // render_pass.set_bind_group(0, &bind_group, &[dynamic_offset]);
/// ```
#[inline]
pub const fn storage_binding_type_dynamic_readonly() -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: true,
        min_binding_size: None,
    }
}

/// Create a read-write storage binding type with dynamic offset.
///
/// Use this when multiple data sets are packed into a single buffer and
/// the offset will be specified at bind time via `set_bind_group`.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::storage::storage_binding_type_dynamic_readwrite;
/// use wgpu::BindGroupLayoutEntry;
///
/// let entry = BindGroupLayoutEntry {
///     binding: 0,
///     visibility: wgpu::ShaderStages::COMPUTE,
///     ty: storage_binding_type_dynamic_readwrite(),
///     count: None,
/// };
/// ```
#[inline]
pub const fn storage_binding_type_dynamic_readwrite() -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: true,
        min_binding_size: None,
    }
}

// ============================================================================
// Binding Type Helpers - Dynamic Offset with min_binding_size
// ============================================================================

/// Create a read-only storage binding type with dynamic offset and minimum size.
///
/// Combines dynamic offset support with size validation.
///
/// # Arguments
///
/// * `size` - Minimum required buffer size in bytes. Must be > 0.
///
/// # Panics
///
/// Panics if `size` is 0.
#[inline]
pub fn storage_binding_type_dynamic_readonly_sized(size: u64) -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: true,
        min_binding_size: Some(
            NonZeroU64::new(size).expect("min_binding_size must be > 0"),
        ),
    }
}

/// Create a read-write storage binding type with dynamic offset and minimum size.
///
/// Combines dynamic offset support with size validation.
///
/// # Arguments
///
/// * `size` - Minimum required buffer size in bytes. Must be > 0.
///
/// # Panics
///
/// Panics if `size` is 0.
#[inline]
pub fn storage_binding_type_dynamic_readwrite_sized(size: u64) -> BindingType {
    BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: true,
        min_binding_size: Some(
            NonZeroU64::new(size).expect("min_binding_size must be > 0"),
        ),
    }
}

// ============================================================================
// GPU Structs
// ============================================================================

/// Common header for GPU storage buffers.
///
/// This header is useful for storage buffers that contain dynamic arrays,
/// allowing the shader to know how many elements are valid.
///
/// # Memory Layout (std430 compatible)
///
/// ```text
/// Offset  Size  Field
/// 0       4     count (u32) - number of valid elements
/// 4       4     capacity (u32) - maximum number of elements
/// 8       4     flags (u32) - application-specific flags
/// 12      4     _padding (u32) - align to 16 bytes
/// ----
/// 16 bytes total
/// ```
///
/// # WGSL Declaration
///
/// ```wgsl
/// struct StorageHeader {
///     count: u32,
///     capacity: u32,
///     flags: u32,
///     _padding: u32,
/// }
///
/// struct MyStorage {
///     header: StorageHeader,
///     data: array<MyElement>,
/// }
/// ```
///
/// # Example
///
/// ```
/// use renderer_backend::resources::storage::StorageHeader;
///
/// let header = StorageHeader::new(100, 1024);
/// assert_eq!(header.count, 100);
/// assert_eq!(header.capacity, 1024);
/// assert_eq!(StorageHeader::SIZE, 16);
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StorageHeader {
    /// Number of valid elements in the storage buffer.
    pub count: u32,

    /// Maximum capacity (number of elements that can be stored).
    pub capacity: u32,

    /// Application-specific flags (e.g., dirty bits, update generation).
    pub flags: u32,

    /// Padding to align struct to 16 bytes.
    pub _padding: u32,
}

impl StorageHeader {
    /// Size of StorageHeader in bytes (16 bytes).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Create a new storage header.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of valid elements
    /// * `capacity` - Maximum number of elements
    #[inline]
    pub const fn new(count: u32, capacity: u32) -> Self {
        Self {
            count,
            capacity,
            flags: 0,
            _padding: 0,
        }
    }

    /// Create a storage header with flags.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of valid elements
    /// * `capacity` - Maximum number of elements
    /// * `flags` - Application-specific flags
    #[inline]
    pub const fn with_flags(count: u32, capacity: u32, flags: u32) -> Self {
        Self {
            count,
            capacity,
            flags,
            _padding: 0,
        }
    }

    /// Create an empty header with the given capacity.
    #[inline]
    pub const fn empty(capacity: u32) -> Self {
        Self::new(0, capacity)
    }
}

/// GPU instance data for instanced rendering.
///
/// This struct is commonly used for GPU-driven instanced rendering where
/// each instance has its own transform and material properties.
///
/// # Memory Layout (std430 compatible)
///
/// ```text
/// Offset  Size  Field
/// 0       64    model (mat4x4<f32>)
/// 64      4     material_id (u32)
/// 68      4     flags (u32)
/// 72      8     _padding (align to 16)
/// ----
/// 80 bytes total
/// ```
///
/// # WGSL Declaration
///
/// ```wgsl
/// struct InstanceData {
///     model: mat4x4<f32>,
///     material_id: u32,
///     flags: u32,
///     _padding: vec2<u32>,
/// }
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    /// Model matrix (local to world transform).
    pub model: [[f32; 4]; 4],

    /// Material index for material array lookup.
    pub material_id: u32,

    /// Instance flags (visibility, LOD level, etc.).
    pub flags: u32,

    /// Padding to align to 16 bytes.
    pub _padding: [u32; 2],
}

impl InstanceData {
    /// Size of InstanceData in bytes (80 bytes).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Flag: instance is visible.
    pub const FLAG_VISIBLE: u32 = 1 << 0;

    /// Flag: instance casts shadows.
    pub const FLAG_CAST_SHADOW: u32 = 1 << 1;

    /// Flag: instance receives shadows.
    pub const FLAG_RECEIVE_SHADOW: u32 = 1 << 2;

    /// Create an identity instance.
    #[inline]
    pub const fn identity() -> Self {
        Self {
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            material_id: 0,
            flags: Self::FLAG_VISIBLE | Self::FLAG_CAST_SHADOW | Self::FLAG_RECEIVE_SHADOW,
            _padding: [0; 2],
        }
    }

    /// Create an instance with the given model matrix.
    #[inline]
    pub const fn with_model(model: [[f32; 4]; 4], material_id: u32) -> Self {
        Self {
            model,
            material_id,
            flags: Self::FLAG_VISIBLE | Self::FLAG_CAST_SHADOW | Self::FLAG_RECEIVE_SHADOW,
            _padding: [0; 2],
        }
    }
}

impl Default for InstanceData {
    fn default() -> Self {
        Self::identity()
    }
}

/// Draw command for indirect rendering.
///
/// This struct matches `wgpu::util::DrawIndirectArgs` and can be used for
/// indirect draw calls where the GPU determines draw parameters.
///
/// # Memory Layout
///
/// ```text
/// Offset  Size  Field
/// 0       4     vertex_count (u32)
/// 4       4     instance_count (u32)
/// 8       4     first_vertex (u32)
/// 12      4     first_instance (u32)
/// ----
/// 16 bytes total
/// ```
///
/// # WGSL Declaration
///
/// ```wgsl
/// struct DrawIndirectArgs {
///     vertex_count: u32,
///     instance_count: u32,
///     first_vertex: u32,
///     first_instance: u32,
/// }
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirectArgs {
    /// Number of vertices to draw.
    pub vertex_count: u32,

    /// Number of instances to draw.
    pub instance_count: u32,

    /// First vertex to start drawing from.
    pub first_vertex: u32,

    /// First instance to start drawing from.
    pub first_instance: u32,
}

impl DrawIndirectArgs {
    /// Size of DrawIndirectArgs in bytes (16 bytes).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Create new draw arguments.
    #[inline]
    pub const fn new(
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Self {
        Self {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        }
    }
}

/// Indexed draw command for indirect rendering.
///
/// This struct matches `wgpu::util::DrawIndexedIndirectArgs` and can be used
/// for indirect indexed draw calls.
///
/// # Memory Layout
///
/// ```text
/// Offset  Size  Field
/// 0       4     index_count (u32)
/// 4       4     instance_count (u32)
/// 8       4     first_index (u32)
/// 12      4     base_vertex (i32)
/// 16      4     first_instance (u32)
/// ----
/// 20 bytes total (padded to 32 for alignment)
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    /// Number of indices to draw.
    pub index_count: u32,

    /// Number of instances to draw.
    pub instance_count: u32,

    /// First index to start drawing from.
    pub first_index: u32,

    /// Base vertex offset added to each index.
    pub base_vertex: i32,

    /// First instance to start drawing from.
    pub first_instance: u32,

    /// Padding to align to 8 bytes (required for some implementations).
    pub _padding: u32,
}

impl DrawIndexedIndirectArgs {
    /// Size of DrawIndexedIndirectArgs in bytes (24 bytes).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Create new indexed draw arguments.
    #[inline]
    pub const fn new(
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Self {
        Self {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
            _padding: 0,
        }
    }
}

/// Dispatch command for indirect compute.
///
/// This struct matches `wgpu::util::DispatchIndirectArgs` and can be used
/// for indirect dispatch calls where the GPU determines workgroup counts.
///
/// # Memory Layout
///
/// ```text
/// Offset  Size  Field
/// 0       4     x (u32) - workgroups in X
/// 4       4     y (u32) - workgroups in Y
/// 8       4     z (u32) - workgroups in Z
/// 12      4     _padding (u32)
/// ----
/// 16 bytes total
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispatchIndirectArgs {
    /// Number of workgroups in X dimension.
    pub x: u32,

    /// Number of workgroups in Y dimension.
    pub y: u32,

    /// Number of workgroups in Z dimension.
    pub z: u32,

    /// Padding to align to 16 bytes.
    pub _padding: u32,
}

impl DispatchIndirectArgs {
    /// Size of DispatchIndirectArgs in bytes (16 bytes).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Create new dispatch arguments.
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self {
            x,
            y,
            z,
            _padding: 0,
        }
    }

    /// Create dispatch arguments for a 1D workload.
    #[inline]
    pub const fn linear(count: u32) -> Self {
        Self::new(count, 1, 1)
    }

    /// Create dispatch arguments for a 2D workload.
    #[inline]
    pub const fn grid_2d(x: u32, y: u32) -> Self {
        Self::new(x, y, 1)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Constants Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_storage_alignment_constant() {
        // Storage alignment is 16 bytes (vec4)
        assert_eq!(STORAGE_ALIGNMENT, 16);
        assert!(STORAGE_ALIGNMENT.is_power_of_two());
    }

    #[test]
    fn test_storage_dynamic_alignment_constant() {
        // Dynamic offset alignment is 256 bytes
        assert_eq!(STORAGE_DYNAMIC_ALIGNMENT, 256);
        assert!(STORAGE_DYNAMIC_ALIGNMENT.is_power_of_two());
    }

    // ------------------------------------------------------------------------
    // Alignment Helper Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_align_storage_size() {
        // Zero stays zero
        assert_eq!(align_storage_size(0), 0);

        // Small values round up to 16
        assert_eq!(align_storage_size(1), 16);
        assert_eq!(align_storage_size(12), 16);
        assert_eq!(align_storage_size(15), 16);

        // Exact multiples stay the same
        assert_eq!(align_storage_size(16), 16);
        assert_eq!(align_storage_size(32), 32);
        assert_eq!(align_storage_size(64), 64);

        // Values between multiples round up
        assert_eq!(align_storage_size(17), 32);
        assert_eq!(align_storage_size(48), 48);
        assert_eq!(align_storage_size(49), 64);
    }

    #[test]
    fn test_align_storage_dynamic_offset() {
        // Zero stays zero
        assert_eq!(align_storage_dynamic_offset(0), 0);

        // Small values round up to 256
        assert_eq!(align_storage_dynamic_offset(1), 256);
        assert_eq!(align_storage_dynamic_offset(128), 256);
        assert_eq!(align_storage_dynamic_offset(255), 256);

        // Exact multiples stay the same
        assert_eq!(align_storage_dynamic_offset(256), 256);
        assert_eq!(align_storage_dynamic_offset(512), 512);

        // Values between multiples round up
        assert_eq!(align_storage_dynamic_offset(257), 512);
        assert_eq!(align_storage_dynamic_offset(300), 512);
    }

    #[test]
    fn test_storage_buffer_size() {
        // Zero elements needs zero bytes
        assert_eq!(storage_buffer_size(0, 64), 0);

        // Elements already aligned
        assert_eq!(storage_buffer_size(10, 16), 160);
        assert_eq!(storage_buffer_size(10, 32), 320);
        assert_eq!(storage_buffer_size(10, 64), 640);

        // Elements needing alignment
        assert_eq!(storage_buffer_size(10, 12), 160); // 12 -> 16
        assert_eq!(storage_buffer_size(10, 20), 320); // 20 -> 32
        assert_eq!(storage_buffer_size(100, 12), 1600); // 12 -> 16, 100 * 16
    }

    // ------------------------------------------------------------------------
    // Binding Type Tests - Basic
    // ------------------------------------------------------------------------

    #[test]
    fn test_storage_binding_type_readonly() {
        let binding = storage_binding_type_readonly();

        match binding {
            BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(matches!(ty, BufferBindingType::Storage { read_only: true }));
                assert!(!has_dynamic_offset);
                assert!(min_binding_size.is_none());
            }
            _ => panic!("Expected Buffer binding type"),
        }
    }

    #[test]
    fn test_storage_binding_type_readwrite() {
        let binding = storage_binding_type_readwrite();

        match binding {
            BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(matches!(ty, BufferBindingType::Storage { read_only: false }));
                assert!(!has_dynamic_offset);
                assert!(min_binding_size.is_none());
            }
            _ => panic!("Expected Buffer binding type"),
        }
    }

    // ------------------------------------------------------------------------
    // Binding Type Tests - With min_binding_size
    // ------------------------------------------------------------------------

    #[test]
    fn test_storage_binding_type_readonly_sized() {
        let binding = storage_binding_type_readonly_sized(1024);

        match binding {
            BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(matches!(ty, BufferBindingType::Storage { read_only: true }));
                assert!(!has_dynamic_offset);
                assert_eq!(min_binding_size, NonZeroU64::new(1024));
            }
            _ => panic!("Expected Buffer binding type"),
        }
    }

    #[test]
    fn test_storage_binding_type_readwrite_sized() {
        let binding = storage_binding_type_readwrite_sized(4096);

        match binding {
            BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(matches!(ty, BufferBindingType::Storage { read_only: false }));
                assert!(!has_dynamic_offset);
                assert_eq!(min_binding_size, NonZeroU64::new(4096));
            }
            _ => panic!("Expected Buffer binding type"),
        }
    }

    #[test]
    #[should_panic(expected = "min_binding_size must be > 0")]
    fn test_storage_binding_type_readonly_sized_zero_panics() {
        storage_binding_type_readonly_sized(0);
    }

    #[test]
    #[should_panic(expected = "min_binding_size must be > 0")]
    fn test_storage_binding_type_readwrite_sized_zero_panics() {
        storage_binding_type_readwrite_sized(0);
    }

    // ------------------------------------------------------------------------
    // Binding Type Tests - Dynamic Offset
    // ------------------------------------------------------------------------

    #[test]
    fn test_storage_binding_type_dynamic_readonly() {
        let binding = storage_binding_type_dynamic_readonly();

        match binding {
            BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(matches!(ty, BufferBindingType::Storage { read_only: true }));
                assert!(has_dynamic_offset);
                assert!(min_binding_size.is_none());
            }
            _ => panic!("Expected Buffer binding type"),
        }
    }

    #[test]
    fn test_storage_binding_type_dynamic_readwrite() {
        let binding = storage_binding_type_dynamic_readwrite();

        match binding {
            BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(matches!(ty, BufferBindingType::Storage { read_only: false }));
                assert!(has_dynamic_offset);
                assert!(min_binding_size.is_none());
            }
            _ => panic!("Expected Buffer binding type"),
        }
    }

    // ------------------------------------------------------------------------
    // Binding Type Tests - Dynamic Offset with min_binding_size
    // ------------------------------------------------------------------------

    #[test]
    fn test_storage_binding_type_dynamic_readonly_sized() {
        let binding = storage_binding_type_dynamic_readonly_sized(512);

        match binding {
            BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(matches!(ty, BufferBindingType::Storage { read_only: true }));
                assert!(has_dynamic_offset);
                assert_eq!(min_binding_size, NonZeroU64::new(512));
            }
            _ => panic!("Expected Buffer binding type"),
        }
    }

    #[test]
    fn test_storage_binding_type_dynamic_readwrite_sized() {
        let binding = storage_binding_type_dynamic_readwrite_sized(2048);

        match binding {
            BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(matches!(ty, BufferBindingType::Storage { read_only: false }));
                assert!(has_dynamic_offset);
                assert_eq!(min_binding_size, NonZeroU64::new(2048));
            }
            _ => panic!("Expected Buffer binding type"),
        }
    }

    // ------------------------------------------------------------------------
    // StorageHeader Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_storage_header_size() {
        assert_eq!(std::mem::size_of::<StorageHeader>(), 16);
        assert_eq!(StorageHeader::SIZE, 16);
    }

    #[test]
    fn test_storage_header_new() {
        let header = StorageHeader::new(100, 1024);
        assert_eq!(header.count, 100);
        assert_eq!(header.capacity, 1024);
        assert_eq!(header.flags, 0);
        assert_eq!(header._padding, 0);
    }

    #[test]
    fn test_storage_header_with_flags() {
        let header = StorageHeader::with_flags(50, 200, 0xFF);
        assert_eq!(header.count, 50);
        assert_eq!(header.capacity, 200);
        assert_eq!(header.flags, 0xFF);
    }

    #[test]
    fn test_storage_header_empty() {
        let header = StorageHeader::empty(512);
        assert_eq!(header.count, 0);
        assert_eq!(header.capacity, 512);
    }

    #[test]
    fn test_storage_header_bytemuck() {
        let header = StorageHeader::new(42, 100);
        let bytes: &[u8] = bytemuck::bytes_of(&header);
        assert_eq!(bytes.len(), 16);

        let recovered: &StorageHeader = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.count, 42);
        assert_eq!(recovered.capacity, 100);
    }

    // ------------------------------------------------------------------------
    // InstanceData Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_instance_data_size() {
        assert_eq!(std::mem::size_of::<InstanceData>(), 80);
        assert_eq!(InstanceData::SIZE, 80);
    }

    #[test]
    fn test_instance_data_identity() {
        let instance = InstanceData::identity();

        // Check identity matrix
        assert_eq!(instance.model[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(instance.model[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(instance.model[2], [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(instance.model[3], [0.0, 0.0, 0.0, 1.0]);

        // Check defaults
        assert_eq!(instance.material_id, 0);
        assert_eq!(
            instance.flags,
            InstanceData::FLAG_VISIBLE
                | InstanceData::FLAG_CAST_SHADOW
                | InstanceData::FLAG_RECEIVE_SHADOW
        );
    }

    #[test]
    fn test_instance_data_with_model() {
        let model = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ];
        let instance = InstanceData::with_model(model, 42);

        assert_eq!(instance.model, model);
        assert_eq!(instance.material_id, 42);
    }

    #[test]
    fn test_instance_data_flags() {
        assert_eq!(InstanceData::FLAG_VISIBLE, 1);
        assert_eq!(InstanceData::FLAG_CAST_SHADOW, 2);
        assert_eq!(InstanceData::FLAG_RECEIVE_SHADOW, 4);
    }

    #[test]
    fn test_instance_data_bytemuck() {
        let instance = InstanceData::identity();
        let bytes: &[u8] = bytemuck::bytes_of(&instance);
        assert_eq!(bytes.len(), 80);
    }

    // ------------------------------------------------------------------------
    // DrawIndirectArgs Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_size() {
        assert_eq!(std::mem::size_of::<DrawIndirectArgs>(), 16);
        assert_eq!(DrawIndirectArgs::SIZE, 16);
    }

    #[test]
    fn test_draw_indirect_args_new() {
        let args = DrawIndirectArgs::new(100, 10, 0, 5);
        assert_eq!(args.vertex_count, 100);
        assert_eq!(args.instance_count, 10);
        assert_eq!(args.first_vertex, 0);
        assert_eq!(args.first_instance, 5);
    }

    #[test]
    fn test_draw_indirect_args_bytemuck() {
        let args = DrawIndirectArgs::new(36, 1, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 16);
    }

    // ------------------------------------------------------------------------
    // DrawIndexedIndirectArgs Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_draw_indexed_indirect_args_size() {
        assert_eq!(std::mem::size_of::<DrawIndexedIndirectArgs>(), 24);
        assert_eq!(DrawIndexedIndirectArgs::SIZE, 24);
    }

    #[test]
    fn test_draw_indexed_indirect_args_new() {
        let args = DrawIndexedIndirectArgs::new(36, 10, 0, -5, 0);
        assert_eq!(args.index_count, 36);
        assert_eq!(args.instance_count, 10);
        assert_eq!(args.first_index, 0);
        assert_eq!(args.base_vertex, -5);
        assert_eq!(args.first_instance, 0);
    }

    #[test]
    fn test_draw_indexed_indirect_args_bytemuck() {
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 24);
    }

    // ------------------------------------------------------------------------
    // DispatchIndirectArgs Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_dispatch_indirect_args_size() {
        assert_eq!(std::mem::size_of::<DispatchIndirectArgs>(), 16);
        assert_eq!(DispatchIndirectArgs::SIZE, 16);
    }

    #[test]
    fn test_dispatch_indirect_args_new() {
        let args = DispatchIndirectArgs::new(8, 8, 1);
        assert_eq!(args.x, 8);
        assert_eq!(args.y, 8);
        assert_eq!(args.z, 1);
    }

    #[test]
    fn test_dispatch_indirect_args_linear() {
        let args = DispatchIndirectArgs::linear(256);
        assert_eq!(args.x, 256);
        assert_eq!(args.y, 1);
        assert_eq!(args.z, 1);
    }

    #[test]
    fn test_dispatch_indirect_args_grid_2d() {
        let args = DispatchIndirectArgs::grid_2d(16, 16);
        assert_eq!(args.x, 16);
        assert_eq!(args.y, 16);
        assert_eq!(args.z, 1);
    }

    #[test]
    fn test_dispatch_indirect_args_bytemuck() {
        let args = DispatchIndirectArgs::new(64, 64, 1);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 16);
    }
}
