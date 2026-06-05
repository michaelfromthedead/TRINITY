//! Indirect buffer support for TRINITY.
//!
//! This module provides indirect draw and dispatch argument structs that exactly
//! match the wgpu::util layout, along with validation helpers and buffer creation
//! utilities for GPU-driven rendering.
//!
//! # Overview
//!
//! Indirect buffers allow the GPU to specify draw/dispatch parameters, enabling:
//! - GPU-driven rendering (culling decides what to draw)
//! - Multi-draw indirect (batch many draws in one call)
//! - Compute-driven dispatch (previous pass determines workload)
//!
//! # Struct Layouts
//!
//! The structs in this module match wgpu::util exactly:
//!
//! | Struct | Size | wgpu Type |
//! |--------|------|-----------|
//! | [`DrawIndirectArgs`] | 16 bytes | `wgpu::util::DrawIndirectArgs` |
//! | [`DrawIndexedIndirectArgs`] | 20 bytes | `wgpu::util::DrawIndexedIndirectArgs` |
//! | [`DispatchIndirectArgs`] | 12 bytes | `wgpu::util::DispatchIndirectArgs` |
//!
//! # Padded vs Exact Layouts
//!
//! When storing indirect args in storage buffer arrays, you may need 16-byte
//! aligned versions. Use the padded versions from [`storage`](super::storage):
//!
//! - `storage::DrawIndexedIndirectArgs` (24 bytes, padded)
//! - `storage::DispatchIndirectArgs` (16 bytes, padded)
//!
//! The exact-size versions in this module are for direct use with wgpu's
//! `draw_indirect`, `draw_indexed_indirect`, and `dispatch_indirect` calls.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::indirect::{
//!     DrawIndirectArgs, DrawIndexedIndirectArgs, DispatchIndirectArgs,
//!     create_indirect_buffer, indirect_buffer_size,
//! };
//!
//! # fn example(device: &wgpu::Device) {
//! // Create a buffer for 100 draw commands
//! let draw_commands = vec![
//!     DrawIndirectArgs::new(36, 1, 0, 0),
//!     DrawIndirectArgs::new(24, 1, 36, 0),
//!     // ...
//! ];
//!
//! let buffer = create_indirect_buffer(
//!     device,
//!     "draw_commands",
//!     bytemuck::cast_slice(&draw_commands),
//! );
//!
//! // Calculate buffer size for N indexed draw commands
//! let size = indirect_buffer_size::<DrawIndexedIndirectArgs>(100);
//! assert_eq!(size, 2000); // 100 * 20 bytes
//! # }
//! ```

use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

// ============================================================================
// Re-exports from storage (padded versions)
// ============================================================================

// Re-export padded versions for storage buffer arrays
pub use super::storage::{
    DispatchIndirectArgs as DispatchIndirectArgsPadded,
    DrawIndexedIndirectArgs as DrawIndexedIndirectArgsPadded,
};

// Re-export the 16-byte DrawIndirectArgs directly (same layout in both)
pub use super::storage::DrawIndirectArgs;

// ============================================================================
// Exact wgpu-compatible structs
// ============================================================================

/// Indexed draw command for indirect rendering (exact wgpu layout).
///
/// This struct matches `wgpu::util::DrawIndexedIndirectArgs` exactly at 20 bytes.
/// Use this for direct indirect draw calls. For storage buffer arrays where
/// 16-byte alignment is needed, use [`DrawIndexedIndirectArgsPadded`].
///
/// # Memory Layout (20 bytes)
///
/// ```text
/// Offset  Size  Field
/// 0       4     index_count (u32)
/// 4       4     instance_count (u32)
/// 8       4     first_index (u32)
/// 12      4     base_vertex (i32) - NOTE: signed!
/// 16      4     first_instance (u32)
/// ----
/// 20 bytes total
/// ```
///
/// # WGSL Declaration
///
/// ```wgsl
/// struct DrawIndexedIndirectArgs {
///     index_count: u32,
///     instance_count: u32,
///     first_index: u32,
///     base_vertex: i32,
///     first_instance: u32,
/// }
/// ```
///
/// # Note on base_vertex
///
/// The `base_vertex` field is a signed integer (i32), not unsigned. This allows
/// negative offsets which can be useful when combining meshes with different
/// vertex buffer layouts.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    /// Number of indices to draw.
    pub index_count: u32,

    /// Number of instances to draw.
    pub instance_count: u32,

    /// First index to start drawing from (offset into index buffer).
    pub first_index: u32,

    /// Base vertex offset added to each index value.
    /// This is **signed** (i32) to allow negative offsets.
    pub base_vertex: i32,

    /// First instance to start drawing from.
    pub first_instance: u32,
}

impl DrawIndexedIndirectArgs {
    /// Size of DrawIndexedIndirectArgs in bytes (20 bytes).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Create new indexed draw arguments.
    ///
    /// # Arguments
    ///
    /// * `index_count` - Number of indices to draw
    /// * `instance_count` - Number of instances to draw
    /// * `first_index` - Starting index in the index buffer
    /// * `base_vertex` - Value added to each index (can be negative)
    /// * `first_instance` - Starting instance ID
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::indirect::DrawIndexedIndirectArgs;
    ///
    /// // Draw a mesh with 36 indices, single instance
    /// let args = DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0);
    ///
    /// // Draw with negative base vertex (useful for packed vertex buffers)
    /// let args_offset = DrawIndexedIndirectArgs::new(36, 1, 0, -100, 0);
    /// ```
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
        }
    }

    /// Create a zeroed (no-op) draw command.
    ///
    /// A draw with `instance_count = 0` or `index_count = 0` draws nothing.
    #[inline]
    pub const fn zeroed() -> Self {
        Self {
            index_count: 0,
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }
    }

    /// Convert to padded version for storage buffer arrays.
    #[inline]
    pub const fn to_padded(self) -> DrawIndexedIndirectArgsPadded {
        DrawIndexedIndirectArgsPadded::new(
            self.index_count,
            self.instance_count,
            self.first_index,
            self.base_vertex,
            self.first_instance,
        )
    }
}

/// Dispatch command for indirect compute (exact wgpu layout).
///
/// This struct matches `wgpu::util::DispatchIndirectArgs` exactly at 12 bytes.
/// Use this for direct indirect dispatch calls. For storage buffer arrays where
/// 16-byte alignment is needed, use [`DispatchIndirectArgsPadded`].
///
/// # Memory Layout (12 bytes)
///
/// ```text
/// Offset  Size  Field
/// 0       4     x (u32) - workgroups in X
/// 4       4     y (u32) - workgroups in Y
/// 8       4     z (u32) - workgroups in Z
/// ----
/// 12 bytes total
/// ```
///
/// # WGSL Declaration
///
/// ```wgsl
/// struct DispatchIndirectArgs {
///     x: u32,
///     y: u32,
///     z: u32,
/// }
/// ```
///
/// # Workgroup Limits
///
/// The maximum workgroup count per dimension is typically 65535. Use
/// [`validate_dispatch_indirect_args`] to check against device limits.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispatchIndirectArgs {
    /// Number of workgroups in X dimension.
    pub x: u32,

    /// Number of workgroups in Y dimension.
    pub y: u32,

    /// Number of workgroups in Z dimension.
    pub z: u32,
}

impl DispatchIndirectArgs {
    /// Size of DispatchIndirectArgs in bytes (12 bytes).
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Maximum workgroup count per dimension (WebGPU limit).
    pub const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535;

    /// Create new dispatch arguments.
    ///
    /// # Arguments
    ///
    /// * `x` - Workgroups in X dimension
    /// * `y` - Workgroups in Y dimension
    /// * `z` - Workgroups in Z dimension
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::indirect::DispatchIndirectArgs;
    ///
    /// // 8x8x1 workgroups for a 2D compute pass
    /// let args = DispatchIndirectArgs::new(8, 8, 1);
    /// ```
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Create dispatch arguments for a 1D workload.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::indirect::DispatchIndirectArgs;
    ///
    /// // 256 workgroups in a single dimension
    /// let args = DispatchIndirectArgs::linear(256);
    /// assert_eq!(args.x, 256);
    /// assert_eq!(args.y, 1);
    /// assert_eq!(args.z, 1);
    /// ```
    #[inline]
    pub const fn linear(count: u32) -> Self {
        Self::new(count, 1, 1)
    }

    /// Create dispatch arguments for a 2D workload.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::indirect::DispatchIndirectArgs;
    ///
    /// // 16x16 grid for image processing
    /// let args = DispatchIndirectArgs::grid_2d(16, 16);
    /// assert_eq!(args.x, 16);
    /// assert_eq!(args.y, 16);
    /// assert_eq!(args.z, 1);
    /// ```
    #[inline]
    pub const fn grid_2d(x: u32, y: u32) -> Self {
        Self::new(x, y, 1)
    }

    /// Create a zeroed (no-op) dispatch command.
    ///
    /// A dispatch with any dimension = 0 does nothing.
    #[inline]
    pub const fn zeroed() -> Self {
        Self { x: 0, y: 0, z: 0 }
    }

    /// Convert to padded version for storage buffer arrays.
    #[inline]
    pub const fn to_padded(self) -> DispatchIndirectArgsPadded {
        DispatchIndirectArgsPadded::new(self.x, self.y, self.z)
    }

    /// Total number of workgroups to dispatch.
    ///
    /// Returns `None` if the multiplication would overflow.
    #[inline]
    pub fn total_workgroups(&self) -> Option<u64> {
        (self.x as u64)
            .checked_mul(self.y as u64)?
            .checked_mul(self.z as u64)
    }
}

// ============================================================================
// Validation
// ============================================================================

/// Error type for indirect argument validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndirectValidationError {
    /// Draw count exceeds reasonable limits.
    DrawCountTooLarge {
        field: &'static str,
        value: u32,
        max: u32,
    },
    /// Workgroup count exceeds device limits.
    WorkgroupCountTooLarge {
        dimension: char,
        value: u32,
        max: u32,
    },
    /// Total workgroup count would overflow.
    WorkgroupOverflow { x: u32, y: u32, z: u32 },
    /// Zero instances or vertices (likely a bug).
    ZeroCount { field: &'static str },
}

impl std::fmt::Display for IndirectValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndirectValidationError::DrawCountTooLarge { field, value, max } => {
                write!(f, "{} ({}) exceeds maximum ({})", field, value, max)
            }
            IndirectValidationError::WorkgroupCountTooLarge {
                dimension,
                value,
                max,
            } => {
                write!(
                    f,
                    "workgroup count in {} ({}) exceeds maximum ({})",
                    dimension, value, max
                )
            }
            IndirectValidationError::WorkgroupOverflow { x, y, z } => {
                write!(
                    f,
                    "total workgroup count ({} * {} * {}) would overflow",
                    x, y, z
                )
            }
            IndirectValidationError::ZeroCount { field } => {
                write!(f, "{} is zero (no-op draw/dispatch)", field)
            }
        }
    }
}

impl std::error::Error for IndirectValidationError {}

/// Validation options for indirect argument checks.
#[derive(Debug, Clone, Copy)]
pub struct ValidationOptions {
    /// Maximum allowed vertex/index count (default: 16M).
    pub max_vertex_count: u32,
    /// Maximum allowed instance count (default: 1M).
    pub max_instance_count: u32,
    /// Maximum workgroups per dimension (default: 65535).
    pub max_workgroups_per_dim: u32,
    /// Warn on zero counts (default: false).
    pub warn_on_zero: bool,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            max_vertex_count: 16 * 1024 * 1024,     // 16M
            max_instance_count: 1024 * 1024,        // 1M
            max_workgroups_per_dim: 65535,          // WebGPU limit
            warn_on_zero: false,
        }
    }
}

/// Validate draw indirect arguments.
///
/// Checks that vertex and instance counts are within reasonable limits.
/// This helps catch bugs where uninitialized or corrupted data would cause
/// the GPU to hang or crash.
///
/// # Arguments
///
/// * `args` - The draw arguments to validate
/// * `options` - Validation options (use `Default::default()` for standard limits)
///
/// # Example
///
/// ```
/// use renderer_backend::resources::indirect::{
///     DrawIndirectArgs, validate_draw_indirect_args, ValidationOptions,
/// };
///
/// let args = DrawIndirectArgs::new(36, 1, 0, 0);
/// assert!(validate_draw_indirect_args(&args, &ValidationOptions::default()).is_ok());
///
/// // Suspiciously large count
/// let bad_args = DrawIndirectArgs::new(u32::MAX, 1, 0, 0);
/// assert!(validate_draw_indirect_args(&bad_args, &ValidationOptions::default()).is_err());
/// ```
pub fn validate_draw_indirect_args(
    args: &DrawIndirectArgs,
    options: &ValidationOptions,
) -> Result<(), IndirectValidationError> {
    if options.warn_on_zero && args.instance_count == 0 {
        return Err(IndirectValidationError::ZeroCount {
            field: "instance_count",
        });
    }

    if options.warn_on_zero && args.vertex_count == 0 {
        return Err(IndirectValidationError::ZeroCount {
            field: "vertex_count",
        });
    }

    if args.vertex_count > options.max_vertex_count {
        return Err(IndirectValidationError::DrawCountTooLarge {
            field: "vertex_count",
            value: args.vertex_count,
            max: options.max_vertex_count,
        });
    }

    if args.instance_count > options.max_instance_count {
        return Err(IndirectValidationError::DrawCountTooLarge {
            field: "instance_count",
            value: args.instance_count,
            max: options.max_instance_count,
        });
    }

    Ok(())
}

/// Validate indexed draw indirect arguments.
///
/// Checks that index and instance counts are within reasonable limits.
///
/// # Arguments
///
/// * `args` - The indexed draw arguments to validate
/// * `options` - Validation options
///
/// # Example
///
/// ```
/// use renderer_backend::resources::indirect::{
///     DrawIndexedIndirectArgs, validate_draw_indexed_indirect_args, ValidationOptions,
/// };
///
/// let args = DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0);
/// assert!(validate_draw_indexed_indirect_args(&args, &ValidationOptions::default()).is_ok());
/// ```
pub fn validate_draw_indexed_indirect_args(
    args: &DrawIndexedIndirectArgs,
    options: &ValidationOptions,
) -> Result<(), IndirectValidationError> {
    if options.warn_on_zero && args.instance_count == 0 {
        return Err(IndirectValidationError::ZeroCount {
            field: "instance_count",
        });
    }

    if options.warn_on_zero && args.index_count == 0 {
        return Err(IndirectValidationError::ZeroCount {
            field: "index_count",
        });
    }

    if args.index_count > options.max_vertex_count {
        return Err(IndirectValidationError::DrawCountTooLarge {
            field: "index_count",
            value: args.index_count,
            max: options.max_vertex_count,
        });
    }

    if args.instance_count > options.max_instance_count {
        return Err(IndirectValidationError::DrawCountTooLarge {
            field: "instance_count",
            value: args.instance_count,
            max: options.max_instance_count,
        });
    }

    Ok(())
}

/// Validate dispatch indirect arguments.
///
/// Checks that workgroup counts are within device limits and won't overflow.
///
/// # Arguments
///
/// * `args` - The dispatch arguments to validate
/// * `options` - Validation options
///
/// # Example
///
/// ```
/// use renderer_backend::resources::indirect::{
///     DispatchIndirectArgs, validate_dispatch_indirect_args, ValidationOptions,
/// };
///
/// let args = DispatchIndirectArgs::new(64, 64, 1);
/// assert!(validate_dispatch_indirect_args(&args, &ValidationOptions::default()).is_ok());
///
/// // Exceeds WebGPU limit
/// let bad_args = DispatchIndirectArgs::new(100000, 1, 1);
/// assert!(validate_dispatch_indirect_args(&bad_args, &ValidationOptions::default()).is_err());
/// ```
pub fn validate_dispatch_indirect_args(
    args: &DispatchIndirectArgs,
    options: &ValidationOptions,
) -> Result<(), IndirectValidationError> {
    if options.warn_on_zero && (args.x == 0 || args.y == 0 || args.z == 0) {
        return Err(IndirectValidationError::ZeroCount {
            field: "workgroup dimension",
        });
    }

    if args.x > options.max_workgroups_per_dim {
        return Err(IndirectValidationError::WorkgroupCountTooLarge {
            dimension: 'X',
            value: args.x,
            max: options.max_workgroups_per_dim,
        });
    }

    if args.y > options.max_workgroups_per_dim {
        return Err(IndirectValidationError::WorkgroupCountTooLarge {
            dimension: 'Y',
            value: args.y,
            max: options.max_workgroups_per_dim,
        });
    }

    if args.z > options.max_workgroups_per_dim {
        return Err(IndirectValidationError::WorkgroupCountTooLarge {
            dimension: 'Z',
            value: args.z,
            max: options.max_workgroups_per_dim,
        });
    }

    // Check for overflow
    if args.total_workgroups().is_none() {
        return Err(IndirectValidationError::WorkgroupOverflow {
            x: args.x,
            y: args.y,
            z: args.z,
        });
    }

    Ok(())
}

// ============================================================================
// Buffer Creation Helpers
// ============================================================================

/// Calculate buffer size for N indirect commands.
///
/// # Type Parameters
///
/// * `T` - The indirect args type (must be Pod)
///
/// # Arguments
///
/// * `count` - Number of commands
///
/// # Example
///
/// ```
/// use renderer_backend::resources::indirect::{
///     DrawIndirectArgs, DrawIndexedIndirectArgs, DispatchIndirectArgs,
///     indirect_buffer_size,
/// };
///
/// assert_eq!(indirect_buffer_size::<DrawIndirectArgs>(10), 160);        // 10 * 16
/// assert_eq!(indirect_buffer_size::<DrawIndexedIndirectArgs>(10), 200); // 10 * 20
/// assert_eq!(indirect_buffer_size::<DispatchIndirectArgs>(10), 120);    // 10 * 12
/// ```
#[inline]
pub const fn indirect_buffer_size<T: bytemuck::Pod>(count: u32) -> u64 {
    std::mem::size_of::<T>() as u64 * count as u64
}

/// Create an indirect buffer with INDIRECT usage.
///
/// Creates a buffer suitable for use with `draw_indirect`, `draw_indexed_indirect`,
/// or `dispatch_indirect` commands.
///
/// # Arguments
///
/// * `device` - The wgpu device
/// * `label` - Debug label for the buffer
/// * `data` - Initial buffer contents (as bytes)
///
/// # Usage Flags
///
/// The buffer is created with `INDIRECT | COPY_DST | STORAGE` flags, allowing:
/// - Direct use with indirect draw/dispatch calls
/// - CPU updates via staging buffer
/// - GPU updates via compute shader (for GPU-driven rendering)
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::indirect::{DrawIndirectArgs, create_indirect_buffer};
///
/// # fn example(device: &wgpu::Device) {
/// let commands = vec![
///     DrawIndirectArgs::new(36, 1, 0, 0),
///     DrawIndirectArgs::new(24, 1, 36, 0),
/// ];
///
/// let buffer = create_indirect_buffer(
///     device,
///     "draw_commands",
///     bytemuck::cast_slice(&commands),
/// );
/// # }
/// ```
pub fn create_indirect_buffer(device: &Device, label: &str, data: &[u8]) -> Buffer {
    use wgpu::util::DeviceExt;

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    })
}

/// Create an empty indirect buffer of the specified size.
///
/// # Arguments
///
/// * `device` - The wgpu device
/// * `label` - Debug label for the buffer
/// * `size` - Buffer size in bytes
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::indirect::{
///     DrawIndirectArgs, create_empty_indirect_buffer, indirect_buffer_size,
/// };
///
/// # fn example(device: &wgpu::Device) {
/// // Create buffer for 100 draw commands
/// let size = indirect_buffer_size::<DrawIndirectArgs>(100);
/// let buffer = create_empty_indirect_buffer(device, "indirect_commands", size);
/// # }
/// ```
pub fn create_empty_indirect_buffer(device: &Device, label: &str, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

/// Create a typed indirect buffer from a slice of commands.
///
/// Convenience wrapper around [`create_indirect_buffer`] that handles the
/// bytemuck cast automatically.
///
/// # Type Parameters
///
/// * `T` - The indirect args type (must be Pod)
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::indirect::{DrawIndirectArgs, create_typed_indirect_buffer};
///
/// # fn example(device: &wgpu::Device) {
/// let commands = [
///     DrawIndirectArgs::new(36, 1, 0, 0),
///     DrawIndirectArgs::new(24, 1, 36, 0),
/// ];
///
/// let buffer = create_typed_indirect_buffer(device, "draws", &commands);
/// # }
/// ```
pub fn create_typed_indirect_buffer<T: bytemuck::Pod>(
    device: &Device,
    label: &str,
    data: &[T],
) -> Buffer {
    create_indirect_buffer(device, label, bytemuck::cast_slice(data))
}

// ============================================================================
// Multi-Draw Support
// ============================================================================

/// Information for multi-draw indirect calls.
///
/// When using `multi_draw_indirect` or `multi_draw_indexed_indirect`,
/// you need to know the command stride and count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiDrawInfo {
    /// Offset into the indirect buffer (in bytes).
    pub offset: u64,
    /// Number of draw commands.
    pub count: u32,
}

impl MultiDrawInfo {
    /// Create multi-draw info for non-indexed draws.
    ///
    /// # Arguments
    ///
    /// * `offset_commands` - Number of commands to skip (0 = start of buffer)
    /// * `count` - Number of draw commands to execute
    #[inline]
    pub const fn draw(offset_commands: u32, count: u32) -> Self {
        Self {
            offset: DrawIndirectArgs::SIZE * offset_commands as u64,
            count,
        }
    }

    /// Create multi-draw info for indexed draws.
    ///
    /// # Arguments
    ///
    /// * `offset_commands` - Number of commands to skip (0 = start of buffer)
    /// * `count` - Number of draw commands to execute
    #[inline]
    pub const fn draw_indexed(offset_commands: u32, count: u32) -> Self {
        Self {
            offset: DrawIndexedIndirectArgs::SIZE * offset_commands as u64,
            count,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // Size Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_size() {
        assert_eq!(std::mem::size_of::<DrawIndirectArgs>(), 16);
        assert_eq!(DrawIndirectArgs::SIZE, 16);
    }

    #[test]
    fn test_draw_indexed_indirect_args_size() {
        // Exact wgpu size: 20 bytes (no padding)
        assert_eq!(std::mem::size_of::<DrawIndexedIndirectArgs>(), 20);
        assert_eq!(DrawIndexedIndirectArgs::SIZE, 20);
    }

    #[test]
    fn test_dispatch_indirect_args_size() {
        // Exact wgpu size: 12 bytes (no padding)
        assert_eq!(std::mem::size_of::<DispatchIndirectArgs>(), 12);
        assert_eq!(DispatchIndirectArgs::SIZE, 12);
    }

    // ------------------------------------------------------------------------
    // DrawIndexedIndirectArgs Tests
    // ------------------------------------------------------------------------

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
    fn test_draw_indexed_indirect_args_zeroed() {
        let args = DrawIndexedIndirectArgs::zeroed();
        assert_eq!(args.index_count, 0);
        assert_eq!(args.instance_count, 0);
        assert_eq!(args.first_index, 0);
        assert_eq!(args.base_vertex, 0);
        assert_eq!(args.first_instance, 0);
    }

    #[test]
    fn test_draw_indexed_indirect_args_bytemuck() {
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 20);

        // Verify layout matches expected
        let recovered: &DrawIndexedIndirectArgs = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.index_count, 36);
    }

    #[test]
    fn test_draw_indexed_indirect_args_negative_base_vertex() {
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, -100, 0);
        assert_eq!(args.base_vertex, -100);

        // Verify bytemuck round-trip preserves sign
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        let recovered: &DrawIndexedIndirectArgs = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.base_vertex, -100);
    }

    #[test]
    fn test_draw_indexed_to_padded() {
        let args = DrawIndexedIndirectArgs::new(36, 10, 5, -3, 2);
        let padded = args.to_padded();
        assert_eq!(padded.index_count, 36);
        assert_eq!(padded.instance_count, 10);
        assert_eq!(padded.first_index, 5);
        assert_eq!(padded.base_vertex, -3);
        assert_eq!(padded.first_instance, 2);
    }

    // ------------------------------------------------------------------------
    // DispatchIndirectArgs Tests
    // ------------------------------------------------------------------------

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
    fn test_dispatch_indirect_args_zeroed() {
        let args = DispatchIndirectArgs::zeroed();
        assert_eq!(args.x, 0);
        assert_eq!(args.y, 0);
        assert_eq!(args.z, 0);
    }

    #[test]
    fn test_dispatch_indirect_args_bytemuck() {
        let args = DispatchIndirectArgs::new(64, 64, 1);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 12);

        let recovered: &DispatchIndirectArgs = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.x, 64);
        assert_eq!(recovered.y, 64);
        assert_eq!(recovered.z, 1);
    }

    #[test]
    fn test_dispatch_indirect_args_total_workgroups() {
        let args = DispatchIndirectArgs::new(8, 8, 2);
        assert_eq!(args.total_workgroups(), Some(128));

        let args_2d = DispatchIndirectArgs::grid_2d(1000, 1000);
        assert_eq!(args_2d.total_workgroups(), Some(1_000_000));
    }

    #[test]
    fn test_dispatch_indirect_args_total_workgroups_overflow() {
        // Very large values that would overflow u64
        let args = DispatchIndirectArgs::new(u32::MAX, u32::MAX, u32::MAX);
        assert!(args.total_workgroups().is_none());
    }

    #[test]
    fn test_dispatch_to_padded() {
        let args = DispatchIndirectArgs::new(64, 32, 16);
        let padded = args.to_padded();
        assert_eq!(padded.x, 64);
        assert_eq!(padded.y, 32);
        assert_eq!(padded.z, 16);
    }

    // ------------------------------------------------------------------------
    // Validation Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_validate_draw_indirect_args_valid() {
        let args = DrawIndirectArgs::new(36, 1, 0, 0);
        assert!(validate_draw_indirect_args(&args, &ValidationOptions::default()).is_ok());
    }

    #[test]
    fn test_validate_draw_indirect_args_too_large() {
        let args = DrawIndirectArgs::new(u32::MAX, 1, 0, 0);
        let result = validate_draw_indirect_args(&args, &ValidationOptions::default());
        assert!(matches!(
            result,
            Err(IndirectValidationError::DrawCountTooLarge { field: "vertex_count", .. })
        ));
    }

    #[test]
    fn test_validate_draw_indirect_args_zero_warn() {
        let args = DrawIndirectArgs::new(36, 0, 0, 0);
        let options = ValidationOptions {
            warn_on_zero: true,
            ..Default::default()
        };
        let result = validate_draw_indirect_args(&args, &options);
        assert!(matches!(
            result,
            Err(IndirectValidationError::ZeroCount { field: "instance_count" })
        ));
    }

    #[test]
    fn test_validate_draw_indexed_indirect_args_valid() {
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0);
        assert!(validate_draw_indexed_indirect_args(&args, &ValidationOptions::default()).is_ok());
    }

    #[test]
    fn test_validate_draw_indexed_indirect_args_negative_base_vertex() {
        // Negative base_vertex is valid
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, -100, 0);
        assert!(validate_draw_indexed_indirect_args(&args, &ValidationOptions::default()).is_ok());
    }

    #[test]
    fn test_validate_dispatch_indirect_args_valid() {
        let args = DispatchIndirectArgs::new(64, 64, 1);
        assert!(validate_dispatch_indirect_args(&args, &ValidationOptions::default()).is_ok());
    }

    #[test]
    fn test_validate_dispatch_indirect_args_too_large() {
        let args = DispatchIndirectArgs::new(100000, 1, 1);
        let result = validate_dispatch_indirect_args(&args, &ValidationOptions::default());
        assert!(matches!(
            result,
            Err(IndirectValidationError::WorkgroupCountTooLarge { dimension: 'X', .. })
        ));
    }

    #[test]
    fn test_validate_dispatch_indirect_args_overflow() {
        let args = DispatchIndirectArgs::new(u32::MAX, u32::MAX, u32::MAX);
        let result = validate_dispatch_indirect_args(&args, &ValidationOptions::default());
        // First hits the size limit, not overflow
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------------
    // Buffer Size Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_indirect_buffer_size() {
        assert_eq!(indirect_buffer_size::<DrawIndirectArgs>(0), 0);
        assert_eq!(indirect_buffer_size::<DrawIndirectArgs>(1), 16);
        assert_eq!(indirect_buffer_size::<DrawIndirectArgs>(10), 160);

        assert_eq!(indirect_buffer_size::<DrawIndexedIndirectArgs>(0), 0);
        assert_eq!(indirect_buffer_size::<DrawIndexedIndirectArgs>(1), 20);
        assert_eq!(indirect_buffer_size::<DrawIndexedIndirectArgs>(10), 200);

        assert_eq!(indirect_buffer_size::<DispatchIndirectArgs>(0), 0);
        assert_eq!(indirect_buffer_size::<DispatchIndirectArgs>(1), 12);
        assert_eq!(indirect_buffer_size::<DispatchIndirectArgs>(10), 120);
    }

    // ------------------------------------------------------------------------
    // MultiDrawInfo Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_multi_draw_info_draw() {
        let info = MultiDrawInfo::draw(0, 100);
        assert_eq!(info.offset, 0);
        assert_eq!(info.count, 100);

        let info_offset = MultiDrawInfo::draw(10, 50);
        assert_eq!(info_offset.offset, 160); // 10 * 16 bytes
        assert_eq!(info_offset.count, 50);
    }

    #[test]
    fn test_multi_draw_info_draw_indexed() {
        let info = MultiDrawInfo::draw_indexed(0, 100);
        assert_eq!(info.offset, 0);
        assert_eq!(info.count, 100);

        let info_offset = MultiDrawInfo::draw_indexed(10, 50);
        assert_eq!(info_offset.offset, 200); // 10 * 20 bytes
        assert_eq!(info_offset.count, 50);
    }

    // ------------------------------------------------------------------------
    // Padded vs Exact Size Comparison
    // ------------------------------------------------------------------------

    #[test]
    fn test_padded_vs_exact_sizes() {
        // DrawIndexedIndirectArgs: exact is 20, padded is 24
        assert_eq!(DrawIndexedIndirectArgs::SIZE, 20);
        assert_eq!(DrawIndexedIndirectArgsPadded::SIZE, 24);

        // DispatchIndirectArgs: exact is 12, padded is 16
        assert_eq!(DispatchIndirectArgs::SIZE, 12);
        assert_eq!(DispatchIndirectArgsPadded::SIZE, 16);

        // DrawIndirectArgs: same size (16 bytes, already aligned)
        assert_eq!(DrawIndirectArgs::SIZE, 16);
    }
}
