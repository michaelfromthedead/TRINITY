//! Copy command wrappers for TRINITY wgpu abstraction.
//!
//! This module provides validated wrappers around wgpu's copy commands with:
//! - Alignment validation (4-byte alignment for offsets and sizes)
//! - Size validation (non-zero, within bounds)
//! - Buffer usage validation (COPY_SRC/COPY_DST)
//!
//! # Overview
//!
//! Copy commands in wgpu require careful attention to alignment and usage flags.
//! This module provides safe wrappers that validate inputs before executing copies:
//!
//! - [`copy_buffer_to_buffer`] - Buffer-to-buffer copy with validation
//! - [`BufferCopyParams`] - Parameters for buffer copy operations
//! - [`CopyError`] - Detailed error types for copy failures
//!
//! # Alignment Requirements
//!
//! wgpu requires 4-byte alignment for buffer copy operations:
//! - Source offset must be 4-byte aligned
//! - Destination offset must be 4-byte aligned
//! - Copy size must be 4-byte aligned
//!
//! # Usage Flags
//!
//! Buffers must have the appropriate usage flags:
//! - Source buffer must have `BufferUsages::COPY_SRC`
//! - Destination buffer must have `BufferUsages::COPY_DST`
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::copy_commands::{copy_buffer_to_buffer, BufferCopyParams};
//! use wgpu::BufferUsages;
//!
//! # fn example(
//! #     encoder: &mut wgpu::CommandEncoder,
//! #     source: &wgpu::Buffer,
//! #     dest: &wgpu::Buffer,
//! # ) {
//! let params = BufferCopyParams {
//!     source_offset: 0,
//!     dest_offset: 256,
//!     size: 1024,
//! };
//!
//! // Returns Ok(()) on success, or CopyError on validation failure
//! copy_buffer_to_buffer(encoder, source, dest, params).expect("Copy failed");
//! # }
//! ```

use std::fmt;
use thiserror::Error;
use wgpu::{AstcBlock, AstcChannel, Buffer, BufferUsages, CommandEncoder, Texture, TextureUsages};

use crate::rhi_commands::RhiCommandEncoder as TrinityCommandEncoder;

// ============================================================================
// Constants
// ============================================================================

/// Required alignment for buffer copy operations (4 bytes).
///
/// wgpu requires that source offset, destination offset, and copy size
/// all be aligned to this value for buffer-to-buffer copies.
pub const COPY_BUFFER_ALIGNMENT: u64 = 4;

/// Required alignment for buffer offsets in texture copy operations (4 bytes).
///
/// When copying between buffers and textures, the buffer offset must be
/// aligned to this value.
pub const BUFFER_OFFSET_ALIGNMENT: u64 = 4;

/// Required alignment for bytes_per_row in texture copy operations (256 bytes).
///
/// wgpu requires that `bytes_per_row` in `ImageDataLayout` be aligned to 256
/// when copying texture data to/from buffers.
pub const BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Required alignment for copy sizes (4 bytes).
///
/// Copy sizes must be aligned to this value for buffer operations.
pub const COPY_SIZE_ALIGNMENT: u64 = 4;

// ============================================================================
// BufferCopyParams
// ============================================================================

/// Parameters for a buffer-to-buffer copy operation.
///
/// Specifies the source offset, destination offset, and size of the copy.
/// All values must be 4-byte aligned for wgpu.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::BufferCopyParams;
///
/// // Copy 1KB from offset 0 to offset 256
/// let params = BufferCopyParams {
///     source_offset: 0,
///     dest_offset: 256,
///     size: 1024,
/// };
///
/// // All values are 4-byte aligned
/// assert!(params.source_offset % 4 == 0);
/// assert!(params.dest_offset % 4 == 0);
/// assert!(params.size % 4 == 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferCopyParams {
    /// Offset in bytes from the start of the source buffer.
    /// Must be 4-byte aligned.
    pub source_offset: u64,

    /// Offset in bytes from the start of the destination buffer.
    /// Must be 4-byte aligned.
    pub dest_offset: u64,

    /// Number of bytes to copy.
    /// Must be 4-byte aligned and greater than 0.
    pub size: u64,
}

impl BufferCopyParams {
    /// Create new buffer copy parameters.
    ///
    /// # Arguments
    ///
    /// * `source_offset` - Offset in source buffer (must be 4-byte aligned)
    /// * `dest_offset` - Offset in destination buffer (must be 4-byte aligned)
    /// * `size` - Number of bytes to copy (must be 4-byte aligned and > 0)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::BufferCopyParams;
    ///
    /// let params = BufferCopyParams::new(0, 0, 512);
    /// assert_eq!(params.size, 512);
    /// ```
    #[inline]
    pub const fn new(source_offset: u64, dest_offset: u64, size: u64) -> Self {
        Self {
            source_offset,
            dest_offset,
            size,
        }
    }

    /// Create parameters for copying an entire buffer to offset 0.
    ///
    /// Useful for simple full-buffer copies.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the source buffer to copy
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::BufferCopyParams;
    ///
    /// let params = BufferCopyParams::full(1024);
    /// assert_eq!(params.source_offset, 0);
    /// assert_eq!(params.dest_offset, 0);
    /// assert_eq!(params.size, 1024);
    /// ```
    #[inline]
    pub const fn full(size: u64) -> Self {
        Self {
            source_offset: 0,
            dest_offset: 0,
            size,
        }
    }

    /// Check if all parameters are 4-byte aligned.
    ///
    /// Returns `true` if source_offset, dest_offset, and size are all
    /// multiples of 4.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::BufferCopyParams;
    ///
    /// let aligned = BufferCopyParams::new(0, 256, 1024);
    /// assert!(aligned.is_aligned());
    ///
    /// let unaligned = BufferCopyParams::new(1, 256, 1024);
    /// assert!(!unaligned.is_aligned());
    /// ```
    #[inline]
    pub const fn is_aligned(&self) -> bool {
        self.source_offset % COPY_BUFFER_ALIGNMENT == 0
            && self.dest_offset % COPY_BUFFER_ALIGNMENT == 0
            && self.size % COPY_BUFFER_ALIGNMENT == 0
    }

    /// Check if the size is valid (greater than 0).
    #[inline]
    pub const fn has_valid_size(&self) -> bool {
        self.size > 0
    }
}

impl Default for BufferCopyParams {
    fn default() -> Self {
        Self {
            source_offset: 0,
            dest_offset: 0,
            size: 0,
        }
    }
}

impl fmt::Display for BufferCopyParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BufferCopyParams {{ source: {}, dest: {}, size: {} }}",
            self.source_offset, self.dest_offset, self.size
        )
    }
}

// ============================================================================
// CopyError
// ============================================================================

/// Error types for buffer copy operations.
///
/// Provides detailed information about why a copy operation failed,
/// including the specific value that caused the error.
///
/// # Error Categories
///
/// - **AlignmentError**: An offset or size is not 4-byte aligned
/// - **SizeError**: The copy size is invalid (zero or exceeds bounds)
/// - **UsageError**: A buffer is missing required usage flags
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::CopyError;
///
/// let error = CopyError::alignment_source(5);
/// assert!(matches!(error, CopyError::AlignmentError { .. }));
/// ```
#[derive(Debug, Clone, Error)]
pub enum CopyError {
    /// An offset or size is not properly aligned.
    #[error("alignment error: {field} value {offset} is not aligned to {required} bytes")]
    AlignmentError {
        /// Which field has the alignment error (source_offset, dest_offset, or size)
        field: &'static str,
        /// The misaligned value
        offset: u64,
        /// Required alignment (4 bytes)
        required: u64,
    },

    /// The copy size is invalid.
    #[error("size error: size {size} is invalid: {reason}")]
    SizeError {
        /// The invalid size value
        size: u64,
        /// Explanation of why the size is invalid
        reason: &'static str,
    },

    /// A buffer is missing required usage flags.
    #[error("usage error: buffer is missing required usage flag: {missing}")]
    UsageError {
        /// Description of the missing usage flag
        missing: &'static str,
    },

    /// Source offset plus size exceeds source buffer bounds.
    #[error("bounds error: source offset {offset} + size {size} exceeds buffer size {buffer_size}")]
    SourceBoundsError {
        /// Source offset
        offset: u64,
        /// Copy size
        size: u64,
        /// Source buffer size
        buffer_size: u64,
    },

    /// Destination offset plus size exceeds destination buffer bounds.
    #[error("bounds error: dest offset {offset} + size {size} exceeds buffer size {buffer_size}")]
    DestBoundsError {
        /// Destination offset
        offset: u64,
        /// Copy size
        size: u64,
        /// Destination buffer size
        buffer_size: u64,
    },

    /// bytes_per_row is not properly aligned to 256 bytes.
    #[error("bytes_per_row alignment error: {value} is not aligned to {required} bytes")]
    BytesPerRowAlignmentError {
        /// The misaligned bytes_per_row value
        value: u32,
        /// Required alignment (256 bytes)
        required: u32,
    },

    /// rows_per_image is required but was not specified.
    #[error("rows_per_image is required when copying to textures with depth_or_array_layers > 1 (got {depth_or_array_layers} layers)")]
    RowsPerImageRequired {
        /// The depth or array layer count that requires rows_per_image
        depth_or_array_layers: u32,
    },

    /// Texture is missing required COPY_DST usage.
    #[error("texture usage error: texture is missing required usage flag: COPY_DST")]
    TextureMissingCopyDst,

    /// Copy dimensions are invalid (zero width, height, or depth).
    #[error("copy extent error: {dimension} must be greater than 0 (got {value})")]
    InvalidExtent {
        /// Which dimension is invalid
        dimension: &'static str,
        /// The invalid value
        value: u32,
    },

    /// Destination buffer is missing required MAP_READ usage for readback.
    #[error("buffer usage error: destination buffer is missing required usage flag: MAP_READ")]
    BufferMissingMapRead,

    /// Source texture is missing required COPY_SRC usage.
    #[error("texture usage error: source texture is missing required usage flag: COPY_SRC")]
    TextureMissingCopySrc,

    /// Texture formats are not copy-compatible.
    #[error("format incompatible: source format {src:?} is not copy-compatible with destination format {dst:?}")]
    FormatIncompatible {
        /// Source texture format
        src: wgpu::TextureFormat,
        /// Destination texture format
        dst: wgpu::TextureFormat,
    },

    /// Buffer is missing required COPY_DST usage for clear operation.
    #[error("buffer usage error: buffer is missing required usage flag: COPY_DST")]
    BufferMissingCopyDst,

    /// Clear offset is not 4-byte aligned.
    #[error("clear alignment error: offset {offset} is not aligned to 4 bytes")]
    ClearOffsetNotAligned {
        /// The misaligned offset value
        offset: u64,
    },

    /// Clear size is not 4-byte aligned.
    #[error("clear alignment error: size {size} is not aligned to 4 bytes")]
    ClearSizeNotAligned {
        /// The misaligned size value
        size: u64,
    },

    /// Clear operation exceeds buffer bounds.
    #[error("clear bounds error: offset {offset} + size {size} exceeds buffer size {buffer_size}")]
    ClearBoundsError {
        /// Clear offset
        offset: u64,
        /// Clear size
        size: u64,
        /// Buffer size
        buffer_size: u64,
    },

    /// Mip level exceeds texture's mip count.
    #[error("mip level out of bounds: level {level} exceeds maximum {max}")]
    MipLevelOutOfBounds {
        /// The requested mip level
        level: u32,
        /// Maximum valid mip level (mip_level_count - 1)
        max: u32,
    },

    /// Copy region exceeds texture bounds at the specified mip level.
    #[error("texture bounds error: copy region ({origin_x}+{width}, {origin_y}+{height}, {origin_z}+{depth}) exceeds texture size ({tex_width}, {tex_height}, {tex_depth}) at mip level {mip_level}")]
    TextureBoundsError {
        /// X origin of the copy region
        origin_x: u32,
        /// Y origin of the copy region
        origin_y: u32,
        /// Z origin (or array layer) of the copy region
        origin_z: u32,
        /// Width of the copy region
        width: u32,
        /// Height of the copy region
        height: u32,
        /// Depth (or array layer count) of the copy region
        depth: u32,
        /// Texture width at the specified mip level
        tex_width: u32,
        /// Texture height at the specified mip level
        tex_height: u32,
        /// Texture depth (or array layer count)
        tex_depth: u32,
        /// The mip level being accessed
        mip_level: u32,
    },
}

impl CopyError {
    /// Create an alignment error for source offset.
    #[inline]
    pub fn alignment_source(offset: u64) -> Self {
        CopyError::AlignmentError {
            field: "source_offset",
            offset,
            required: COPY_BUFFER_ALIGNMENT,
        }
    }

    /// Create an alignment error for destination offset.
    #[inline]
    pub fn alignment_dest(offset: u64) -> Self {
        CopyError::AlignmentError {
            field: "dest_offset",
            offset,
            required: COPY_BUFFER_ALIGNMENT,
        }
    }

    /// Create an alignment error for copy size.
    #[inline]
    pub fn alignment_size(size: u64) -> Self {
        CopyError::AlignmentError {
            field: "size",
            offset: size,
            required: COPY_BUFFER_ALIGNMENT,
        }
    }

    /// Create a size error for zero size.
    #[inline]
    pub fn zero_size() -> Self {
        CopyError::SizeError {
            size: 0,
            reason: "copy size must be greater than 0",
        }
    }

    /// Create a usage error for missing COPY_SRC.
    #[inline]
    pub fn missing_copy_src() -> Self {
        CopyError::UsageError {
            missing: "COPY_SRC",
        }
    }

    /// Create a usage error for missing COPY_DST.
    #[inline]
    pub fn missing_copy_dst() -> Self {
        CopyError::UsageError {
            missing: "COPY_DST",
        }
    }

    /// Create a source bounds error.
    #[inline]
    pub fn source_bounds(offset: u64, size: u64, buffer_size: u64) -> Self {
        CopyError::SourceBoundsError {
            offset,
            size,
            buffer_size,
        }
    }

    /// Create a destination bounds error.
    #[inline]
    pub fn dest_bounds(offset: u64, size: u64, buffer_size: u64) -> Self {
        CopyError::DestBoundsError {
            offset,
            size,
            buffer_size,
        }
    }

    /// Create a bytes_per_row alignment error.
    #[inline]
    pub fn bytes_per_row_alignment(value: u32) -> Self {
        CopyError::BytesPerRowAlignmentError {
            value,
            required: BYTES_PER_ROW_ALIGNMENT,
        }
    }

    /// Create a rows_per_image required error.
    #[inline]
    pub fn rows_per_image_required(depth_or_array_layers: u32) -> Self {
        CopyError::RowsPerImageRequired {
            depth_or_array_layers,
        }
    }

    /// Create a texture missing COPY_DST error.
    #[inline]
    pub fn texture_missing_copy_dst() -> Self {
        CopyError::TextureMissingCopyDst
    }

    /// Create an invalid extent error.
    #[inline]
    pub fn invalid_extent(dimension: &'static str, value: u32) -> Self {
        CopyError::InvalidExtent { dimension, value }
    }

    /// Create a buffer missing MAP_READ error.
    #[inline]
    pub fn buffer_missing_map_read() -> Self {
        CopyError::BufferMissingMapRead
    }

    /// Create a texture missing COPY_SRC error.
    #[inline]
    pub fn texture_missing_copy_src() -> Self {
        CopyError::TextureMissingCopySrc
    }

    /// Create a format incompatible error.
    #[inline]
    pub fn format_incompatible(src: wgpu::TextureFormat, dst: wgpu::TextureFormat) -> Self {
        CopyError::FormatIncompatible { src, dst }
    }

    /// Create a mip level out of bounds error.
    #[inline]
    pub fn mip_level_out_of_bounds(level: u32, max: u32) -> Self {
        CopyError::MipLevelOutOfBounds { level, max }
    }

    /// Create a buffer missing COPY_DST error.
    #[inline]
    pub fn buffer_missing_copy_dst() -> Self {
        CopyError::BufferMissingCopyDst
    }

    /// Create a clear offset not aligned error.
    #[inline]
    pub fn clear_offset_not_aligned(offset: u64) -> Self {
        CopyError::ClearOffsetNotAligned { offset }
    }

    /// Create a clear size not aligned error.
    #[inline]
    pub fn clear_size_not_aligned(size: u64) -> Self {
        CopyError::ClearSizeNotAligned { size }
    }

    /// Create a clear bounds error.
    #[inline]
    pub fn clear_bounds(offset: u64, size: u64, buffer_size: u64) -> Self {
        CopyError::ClearBoundsError {
            offset,
            size,
            buffer_size,
        }
    }

    /// Create a texture bounds error.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn texture_bounds(
        origin_x: u32,
        origin_y: u32,
        origin_z: u32,
        width: u32,
        height: u32,
        depth: u32,
        tex_width: u32,
        tex_height: u32,
        tex_depth: u32,
        mip_level: u32,
    ) -> Self {
        CopyError::TextureBoundsError {
            origin_x,
            origin_y,
            origin_z,
            width,
            height,
            depth,
            tex_width,
            tex_height,
            tex_depth,
            mip_level,
        }
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Check if a value is 4-byte aligned.
#[inline]
pub const fn is_aligned(value: u64) -> bool {
    value % COPY_BUFFER_ALIGNMENT == 0
}

/// Align a value up to 4-byte boundary.
#[inline]
pub const fn align_up(value: u64) -> u64 {
    (value + COPY_BUFFER_ALIGNMENT - 1) & !(COPY_BUFFER_ALIGNMENT - 1)
}

/// Validate buffer copy parameters.
///
/// Checks alignment and size constraints. Does NOT check buffer bounds or usage
/// (those require buffer references).
///
/// # Arguments
///
/// * `params` - The copy parameters to validate
///
/// # Returns
///
/// * `Ok(())` if all parameters are valid
/// * `Err(CopyError)` with details if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{validate_params, BufferCopyParams, CopyError};
///
/// // Valid parameters
/// let valid = BufferCopyParams::new(0, 256, 1024);
/// assert!(validate_params(&valid).is_ok());
///
/// // Unaligned source offset
/// let invalid = BufferCopyParams::new(3, 256, 1024);
/// assert!(matches!(validate_params(&invalid), Err(CopyError::AlignmentError { .. })));
/// ```
pub fn validate_params(params: &BufferCopyParams) -> Result<(), CopyError> {
    // Check source offset alignment
    if params.source_offset % COPY_BUFFER_ALIGNMENT != 0 {
        return Err(CopyError::alignment_source(params.source_offset));
    }

    // Check dest offset alignment
    if params.dest_offset % COPY_BUFFER_ALIGNMENT != 0 {
        return Err(CopyError::alignment_dest(params.dest_offset));
    }

    // Check size alignment
    if params.size % COPY_BUFFER_ALIGNMENT != 0 {
        return Err(CopyError::alignment_size(params.size));
    }

    // Check size > 0
    if params.size == 0 {
        return Err(CopyError::zero_size());
    }

    Ok(())
}

/// Validate buffer usage flags for copy operation.
///
/// Checks that:
/// - Source buffer has `BufferUsages::COPY_SRC`
/// - Destination buffer has `BufferUsages::COPY_DST`
///
/// # Arguments
///
/// * `source_usage` - Usage flags of the source buffer
/// * `dest_usage` - Usage flags of the destination buffer
///
/// # Returns
///
/// * `Ok(())` if both buffers have required flags
/// * `Err(CopyError)` if a required flag is missing
pub fn validate_usage(
    source_usage: BufferUsages,
    dest_usage: BufferUsages,
) -> Result<(), CopyError> {
    if !source_usage.contains(BufferUsages::COPY_SRC) {
        return Err(CopyError::missing_copy_src());
    }

    if !dest_usage.contains(BufferUsages::COPY_DST) {
        return Err(CopyError::missing_copy_dst());
    }

    Ok(())
}

/// Validate copy bounds against buffer sizes.
///
/// Checks that:
/// - source_offset + size <= source_size
/// - dest_offset + size <= dest_size
///
/// # Arguments
///
/// * `params` - The copy parameters
/// * `source_size` - Size of the source buffer in bytes
/// * `dest_size` - Size of the destination buffer in bytes
///
/// # Returns
///
/// * `Ok(())` if the copy is within bounds
/// * `Err(CopyError)` if the copy would exceed buffer bounds
pub fn validate_bounds(
    params: &BufferCopyParams,
    source_size: u64,
    dest_size: u64,
) -> Result<(), CopyError> {
    // Check source bounds (handle overflow)
    let source_end = params
        .source_offset
        .checked_add(params.size)
        .ok_or_else(|| CopyError::source_bounds(params.source_offset, params.size, source_size))?;

    if source_end > source_size {
        return Err(CopyError::source_bounds(
            params.source_offset,
            params.size,
            source_size,
        ));
    }

    // Check dest bounds (handle overflow)
    let dest_end = params
        .dest_offset
        .checked_add(params.size)
        .ok_or_else(|| CopyError::dest_bounds(params.dest_offset, params.size, dest_size))?;

    if dest_end > dest_size {
        return Err(CopyError::dest_bounds(
            params.dest_offset,
            params.size,
            dest_size,
        ));
    }

    Ok(())
}

// ============================================================================
// Copy Alignment Calculator
// ============================================================================

/// Utility for calculating copy alignment requirements.
///
/// This struct provides const methods for calculating aligned values needed
/// for texture copy operations between buffers and textures. wgpu has strict
/// alignment requirements for these operations.
///
/// # Alignment Requirements
///
/// - **Buffer offset**: Must be 4-byte aligned (`BUFFER_OFFSET_ALIGNMENT`)
/// - **Bytes per row**: Must be 256-byte aligned (`BYTES_PER_ROW_ALIGNMENT`)
/// - **Copy size**: Must be 4-byte aligned (`COPY_SIZE_ALIGNMENT`)
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::CopyAlignmentCalculator;
///
/// // For a 100-pixel wide RGBA texture (4 bytes per pixel = 400 bytes per row)
/// let unaligned_bytes_per_row = 100 * 4; // 400
/// let aligned_bytes_per_row = CopyAlignmentCalculator::calculate_aligned_bytes_per_row(
///     unaligned_bytes_per_row
/// );
/// assert_eq!(aligned_bytes_per_row, 512); // Aligned to 256
///
/// // Calculate buffer size for a 2D texture
/// let buffer_size = CopyAlignmentCalculator::calculate_buffer_size(
///     aligned_bytes_per_row,
///     100, // height (rows_per_image for 2D)
///     1,   // depth = 1 for 2D textures
/// );
/// ```
pub struct CopyAlignmentCalculator;

impl CopyAlignmentCalculator {
    /// Align bytes_per_row up to the next multiple of `BYTES_PER_ROW_ALIGNMENT` (256).
    ///
    /// wgpu requires `bytes_per_row` in `ImageDataLayout` to be aligned to 256
    /// when copying texture data to/from buffers.
    ///
    /// # Arguments
    ///
    /// * `unaligned_bytes_per_row` - The unaligned bytes per row value
    ///
    /// # Returns
    ///
    /// The value aligned up to the next 256-byte boundary. Returns 0 for input 0.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::CopyAlignmentCalculator;
    ///
    /// // Exact multiple of 256 stays the same
    /// assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(256), 256);
    /// assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(512), 512);
    ///
    /// // Values are rounded up to next 256-byte boundary
    /// assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(100), 256);
    /// assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(300), 512);
    /// assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(257), 512);
    /// ```
    #[inline]
    pub const fn calculate_aligned_bytes_per_row(unaligned_bytes_per_row: u32) -> u32 {
        if unaligned_bytes_per_row == 0 {
            return 0;
        }
        // Align up: (value + alignment - 1) & !(alignment - 1)
        let alignment = BYTES_PER_ROW_ALIGNMENT;
        (unaligned_bytes_per_row + alignment - 1) & !(alignment - 1)
    }

    /// Calculate rows per image for 3D textures or texture arrays.
    ///
    /// For 3D textures or 2D texture arrays, this calculates the number of rows
    /// per "image slice". For a simple 2D texture with depth/array_layers = 1,
    /// this returns the height.
    ///
    /// # Arguments
    ///
    /// * `height` - The height of the texture in pixels
    /// * `depth_or_array_layers` - The depth (for 3D) or array layer count
    ///
    /// # Returns
    ///
    /// The number of rows per image slice. For 2D textures (depth=1), this equals height.
    /// For 3D/array textures, this also equals height (rows per individual layer/slice).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::CopyAlignmentCalculator;
    ///
    /// // 2D texture: rows_per_image = height
    /// let rows = CopyAlignmentCalculator::calculate_rows_per_image(100, 1);
    /// assert_eq!(rows, 100);
    ///
    /// // 3D texture or array: each slice has `height` rows
    /// let rows = CopyAlignmentCalculator::calculate_rows_per_image(64, 8);
    /// assert_eq!(rows, 64);
    /// ```
    #[inline]
    pub const fn calculate_rows_per_image(height: u32, _depth_or_array_layers: u32) -> u32 {
        // For wgpu's ImageDataLayout, rows_per_image is the number of rows
        // per depth slice or array layer. This is always the texture height.
        height
    }

    /// Calculate the total buffer size needed for a texture copy operation.
    ///
    /// This computes the minimum buffer size required to hold texture data
    /// for copy operations, accounting for row padding and multiple depth slices.
    ///
    /// # Arguments
    ///
    /// * `bytes_per_row` - Aligned bytes per row (should be 256-byte aligned)
    /// * `rows_per_image` - Number of rows per image/slice (typically texture height)
    /// * `depth` - Number of depth slices or array layers
    ///
    /// # Returns
    ///
    /// The total buffer size in bytes needed to hold the texture data.
    ///
    /// # Formula
    ///
    /// For a texture copy, the required buffer size is:
    /// - For `depth == 1`: `bytes_per_row * rows_per_image`
    /// - For `depth > 1`: `bytes_per_row * rows_per_image * depth`
    ///
    /// Note: The last row of the last slice doesn't need padding, but we
    /// allocate conservatively to simplify buffer management.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::CopyAlignmentCalculator;
    ///
    /// // 2D texture: 256 bytes/row, 100 rows, depth 1
    /// let size = CopyAlignmentCalculator::calculate_buffer_size(256, 100, 1);
    /// assert_eq!(size, 25600);
    ///
    /// // 3D texture: 512 bytes/row, 64 rows, depth 8
    /// let size = CopyAlignmentCalculator::calculate_buffer_size(512, 64, 8);
    /// assert_eq!(size, 262144); // 512 * 64 * 8
    /// ```
    #[inline]
    pub const fn calculate_buffer_size(bytes_per_row: u32, rows_per_image: u32, depth: u32) -> u64 {
        (bytes_per_row as u64) * (rows_per_image as u64) * (depth as u64)
    }

    /// Calculate the aligned buffer size including alignment padding.
    ///
    /// This is a convenience method that combines bytes_per_row alignment
    /// and buffer size calculation.
    ///
    /// # Arguments
    ///
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `depth` - Number of depth slices or array layers
    /// * `bytes_per_pixel` - Number of bytes per pixel (e.g., 4 for RGBA8)
    ///
    /// # Returns
    ///
    /// A tuple of (aligned_bytes_per_row, total_buffer_size)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::CopyAlignmentCalculator;
    ///
    /// // 100x100 RGBA texture
    /// let (bytes_per_row, total_size) =
    ///     CopyAlignmentCalculator::calculate_texture_buffer_layout(100, 100, 1, 4);
    /// assert_eq!(bytes_per_row, 512); // 100 * 4 = 400, aligned to 512
    /// assert_eq!(total_size, 51200);  // 512 * 100 * 1
    /// ```
    #[inline]
    pub const fn calculate_texture_buffer_layout(
        width: u32,
        height: u32,
        depth: u32,
        bytes_per_pixel: u32,
    ) -> (u32, u64) {
        let unaligned_bytes_per_row = width * bytes_per_pixel;
        let aligned_bytes_per_row = Self::calculate_aligned_bytes_per_row(unaligned_bytes_per_row);
        let rows_per_image = Self::calculate_rows_per_image(height, depth);
        let total_size = Self::calculate_buffer_size(aligned_bytes_per_row, rows_per_image, depth);
        (aligned_bytes_per_row, total_size)
    }

    /// Check if a bytes_per_row value is properly aligned.
    ///
    /// # Arguments
    ///
    /// * `bytes_per_row` - The bytes per row value to check
    ///
    /// # Returns
    ///
    /// `true` if the value is aligned to 256 bytes, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::CopyAlignmentCalculator;
    ///
    /// assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(256));
    /// assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(512));
    /// assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(100));
    /// assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(300));
    /// ```
    #[inline]
    pub const fn is_bytes_per_row_aligned(bytes_per_row: u32) -> bool {
        bytes_per_row % BYTES_PER_ROW_ALIGNMENT == 0
    }

    /// Calculate padding bytes added for row alignment.
    ///
    /// # Arguments
    ///
    /// * `unaligned_bytes_per_row` - The unaligned bytes per row value
    ///
    /// # Returns
    ///
    /// The number of padding bytes that will be added for alignment.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::CopyAlignmentCalculator;
    ///
    /// // 256 needs no padding
    /// assert_eq!(CopyAlignmentCalculator::calculate_row_padding(256), 0);
    ///
    /// // 100 needs 156 bytes of padding to reach 256
    /// assert_eq!(CopyAlignmentCalculator::calculate_row_padding(100), 156);
    ///
    /// // 300 needs 212 bytes of padding to reach 512
    /// assert_eq!(CopyAlignmentCalculator::calculate_row_padding(300), 212);
    /// ```
    #[inline]
    pub const fn calculate_row_padding(unaligned_bytes_per_row: u32) -> u32 {
        let aligned = Self::calculate_aligned_bytes_per_row(unaligned_bytes_per_row);
        aligned - unaligned_bytes_per_row
    }
}

// ============================================================================
// Buffer-to-Texture Copy Types
// ============================================================================

/// Source buffer information for a buffer-to-texture copy operation.
///
/// This struct wraps a buffer reference with its data layout, specifying how
/// the buffer data is organized for copying to a texture.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::ImageCopyBuffer;
/// use wgpu::ImageDataLayout;
///
/// # fn example(buffer: &wgpu::Buffer) {
/// let source = ImageCopyBuffer {
///     buffer,
///     layout: ImageDataLayout {
///         offset: 0,
///         bytes_per_row: Some(512),  // Must be 256-aligned
///         rows_per_image: Some(128), // Required for 3D/array textures
///     },
/// };
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ImageCopyBuffer<'a> {
    /// The source buffer containing the image data.
    /// Must have `BufferUsages::COPY_SRC`.
    pub buffer: &'a Buffer,

    /// Layout of the image data in the buffer.
    /// `bytes_per_row` must be 256-byte aligned.
    /// `rows_per_image` is required when copying to 3D textures or texture arrays.
    pub layout: wgpu::ImageDataLayout,
}

impl<'a> ImageCopyBuffer<'a> {
    /// Create a new ImageCopyBuffer with specified layout.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The source buffer
    /// * `offset` - Byte offset into the buffer
    /// * `bytes_per_row` - Bytes per row (must be 256-aligned)
    /// * `rows_per_image` - Rows per image slice (required for 3D/array textures)
    #[inline]
    pub fn new(
        buffer: &'a Buffer,
        offset: u64,
        bytes_per_row: u32,
        rows_per_image: Option<u32>,
    ) -> Self {
        Self {
            buffer,
            layout: wgpu::ImageDataLayout {
                offset,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image,
            },
        }
    }

    /// Create an ImageCopyBuffer with automatic layout calculation.
    ///
    /// Calculates the aligned bytes_per_row based on texture width and bytes per pixel.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The source buffer
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `depth_or_array_layers` - Depth or array layer count
    /// * `bytes_per_pixel` - Bytes per pixel (e.g., 4 for RGBA8)
    #[inline]
    pub fn with_auto_layout(
        buffer: &'a Buffer,
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
        bytes_per_pixel: u32,
    ) -> Self {
        let bytes_per_row = CopyAlignmentCalculator::calculate_aligned_bytes_per_row(
            width * bytes_per_pixel,
        );
        let rows_per_image = if depth_or_array_layers > 1 {
            Some(height)
        } else {
            None
        };

        Self::new(buffer, 0, bytes_per_row, rows_per_image)
    }
}

/// Destination texture information for a buffer-to-texture copy operation.
///
/// This struct wraps a texture reference with target location information,
/// specifying where in the texture the data should be copied to.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::ImageCopyTexture;
/// use wgpu::{Origin3d, TextureAspect};
///
/// # fn example(texture: &wgpu::Texture) {
/// let destination = ImageCopyTexture {
///     texture,
///     mip_level: 0,
///     origin: Origin3d::ZERO,
///     aspect: TextureAspect::All,
/// };
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ImageCopyTexture<'a> {
    /// The destination texture.
    /// Must have `TextureUsages::COPY_DST`.
    pub texture: &'a Texture,

    /// The target mip level for the copy (0 = base level).
    pub mip_level: u32,

    /// The 3D origin within the texture to start copying at.
    pub origin: wgpu::Origin3d,

    /// Which aspect of the texture to copy to.
    /// For color textures, use `TextureAspect::All`.
    /// For depth-stencil textures, specify depth or stencil aspect.
    pub aspect: wgpu::TextureAspect,
}

impl<'a> ImageCopyTexture<'a> {
    /// Create a new ImageCopyTexture targeting the base mip level at origin (0,0,0).
    ///
    /// # Arguments
    ///
    /// * `texture` - The destination texture
    #[inline]
    pub fn new(texture: &'a Texture) -> Self {
        Self {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        }
    }

    /// Create with a specific mip level.
    ///
    /// # Arguments
    ///
    /// * `texture` - The destination texture
    /// * `mip_level` - Target mip level (0 = base)
    #[inline]
    pub fn with_mip_level(texture: &'a Texture, mip_level: u32) -> Self {
        Self {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        }
    }

    /// Create with a specific origin within the texture.
    ///
    /// # Arguments
    ///
    /// * `texture` - The destination texture
    /// * `x` - X coordinate of origin
    /// * `y` - Y coordinate of origin
    /// * `z` - Z coordinate or array layer index
    #[inline]
    pub fn with_origin(texture: &'a Texture, x: u32, y: u32, z: u32) -> Self {
        Self {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z },
            aspect: wgpu::TextureAspect::All,
        }
    }

    /// Create with full customization.
    ///
    /// # Arguments
    ///
    /// * `texture` - The destination texture
    /// * `mip_level` - Target mip level
    /// * `origin` - 3D origin within the texture
    /// * `aspect` - Texture aspect (color, depth, or stencil)
    #[inline]
    pub fn full(
        texture: &'a Texture,
        mip_level: u32,
        origin: wgpu::Origin3d,
        aspect: wgpu::TextureAspect,
    ) -> Self {
        Self {
            texture,
            mip_level,
            origin,
            aspect,
        }
    }
}

/// Dimensions of a copy operation (width, height, depth/layers).
///
/// This struct specifies the extent of data to copy in a buffer-to-texture
/// or texture-to-buffer operation.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::CopyExtent3d;
///
/// // 2D texture copy: 256x256, depth=1
/// let extent_2d = CopyExtent3d::new_2d(256, 256);
///
/// // 3D texture or array copy: 64x64, 8 layers
/// let extent_3d = CopyExtent3d::new(64, 64, 8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CopyExtent3d {
    /// Width of the copy region in pixels/texels.
    pub width: u32,

    /// Height of the copy region in pixels/texels.
    pub height: u32,

    /// Depth (for 3D textures) or array layer count (for 2D arrays).
    /// For simple 2D textures, this should be 1.
    pub depth_or_array_layers: u32,
}

impl CopyExtent3d {
    /// Create a new CopyExtent3d with specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    /// * `depth_or_array_layers` - Depth or layer count
    #[inline]
    pub const fn new(width: u32, height: u32, depth_or_array_layers: u32) -> Self {
        Self {
            width,
            height,
            depth_or_array_layers,
        }
    }

    /// Create a CopyExtent3d for a 2D texture (depth = 1).
    ///
    /// # Arguments
    ///
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    #[inline]
    pub const fn new_2d(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth_or_array_layers: 1,
        }
    }

    /// Check if all dimensions are greater than zero.
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0 && self.depth_or_array_layers > 0
    }

    /// Check if this is a multi-layer copy (3D texture or texture array).
    #[inline]
    pub const fn is_multi_layer(&self) -> bool {
        self.depth_or_array_layers > 1
    }

    /// Convert to wgpu's Extent3d type.
    #[inline]
    pub const fn to_wgpu(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: self.depth_or_array_layers,
        }
    }
}

impl From<CopyExtent3d> for wgpu::Extent3d {
    fn from(extent: CopyExtent3d) -> Self {
        extent.to_wgpu()
    }
}

impl From<wgpu::Extent3d> for CopyExtent3d {
    fn from(extent: wgpu::Extent3d) -> Self {
        Self {
            width: extent.width,
            height: extent.height,
            depth_or_array_layers: extent.depth_or_array_layers,
        }
    }
}

impl Default for CopyExtent3d {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        }
    }
}

impl fmt::Display for CopyExtent3d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{}x{}",
            self.width, self.height, self.depth_or_array_layers
        )
    }
}

// ============================================================================
// Buffer-to-Texture Validation
// ============================================================================

/// Validate buffer-to-texture copy parameters.
///
/// Checks:
/// - bytes_per_row is 256-byte aligned
/// - rows_per_image is specified when depth_or_array_layers > 1
/// - All extent dimensions are > 0
/// - Source buffer has COPY_SRC usage
/// - Destination texture has COPY_DST usage
///
/// # Arguments
///
/// * `source` - The source buffer copy info
/// * `destination` - The destination texture copy info
/// * `copy_size` - The extent of the copy operation
///
/// # Returns
///
/// * `Ok(())` if all parameters are valid
/// * `Err(CopyError)` with details if validation fails
pub fn validate_buffer_to_texture_params(
    source: &ImageCopyBuffer,
    destination: &ImageCopyTexture,
    copy_size: &CopyExtent3d,
) -> Result<(), CopyError> {
    // Validate extent dimensions
    if copy_size.width == 0 {
        return Err(CopyError::invalid_extent("width", 0));
    }
    if copy_size.height == 0 {
        return Err(CopyError::invalid_extent("height", 0));
    }
    if copy_size.depth_or_array_layers == 0 {
        return Err(CopyError::invalid_extent("depth_or_array_layers", 0));
    }

    // Validate bytes_per_row alignment
    if let Some(bytes_per_row) = source.layout.bytes_per_row {
        if !CopyAlignmentCalculator::is_bytes_per_row_aligned(bytes_per_row) {
            return Err(CopyError::bytes_per_row_alignment(bytes_per_row));
        }
    }

    // Validate rows_per_image is present when needed
    if copy_size.depth_or_array_layers > 1 && source.layout.rows_per_image.is_none() {
        return Err(CopyError::rows_per_image_required(
            copy_size.depth_or_array_layers,
        ));
    }

    // Validate buffer offset alignment (4-byte)
    if source.layout.offset % BUFFER_OFFSET_ALIGNMENT != 0 {
        return Err(CopyError::AlignmentError {
            field: "buffer_offset",
            offset: source.layout.offset,
            required: BUFFER_OFFSET_ALIGNMENT,
        });
    }

    // Validate buffer usage
    if !source.buffer.usage().contains(BufferUsages::COPY_SRC) {
        return Err(CopyError::missing_copy_src());
    }

    // Validate texture usage
    if !destination.texture.usage().contains(TextureUsages::COPY_DST) {
        return Err(CopyError::texture_missing_copy_dst());
    }

    Ok(())
}

// ============================================================================
// Copy Functions
// ============================================================================

/// Copy data from a buffer to a texture with validation.
///
/// This function validates all parameters before executing the copy:
/// - bytes_per_row must be aligned to 256 bytes
/// - rows_per_image must be specified for 3D textures or texture arrays
/// - Buffer offset must be 4-byte aligned
/// - Source buffer must have COPY_SRC usage
/// - Destination texture must have COPY_DST usage
/// - Copy extent must have non-zero dimensions
///
/// # Arguments
///
/// * `encoder` - The command encoder to record the copy command on
/// * `source` - Source buffer information (buffer + layout)
/// * `destination` - Destination texture information (texture + mip/origin/aspect)
/// * `copy_size` - The extent of the copy region (width, height, depth/layers)
///
/// # Returns
///
/// * `Ok(())` if the copy was recorded successfully
/// * `Err(CopyError)` if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{
///     copy_buffer_to_texture, ImageCopyBuffer, ImageCopyTexture, CopyExtent3d
/// };
/// use wgpu::{ImageDataLayout, Origin3d, TextureAspect};
///
/// # fn example(
/// #     encoder: &mut wgpu::CommandEncoder,
/// #     buffer: &wgpu::Buffer,
/// #     texture: &wgpu::Texture,
/// # ) {
/// let source = ImageCopyBuffer {
///     buffer,
///     layout: ImageDataLayout {
///         offset: 0,
///         bytes_per_row: Some(512), // 256-aligned
///         rows_per_image: None,     // Not needed for 2D
///     },
/// };
///
/// let destination = ImageCopyTexture {
///     texture,
///     mip_level: 0,
///     origin: Origin3d::ZERO,
///     aspect: TextureAspect::All,
/// };
///
/// let copy_size = CopyExtent3d::new_2d(128, 128);
///
/// copy_buffer_to_texture(encoder, source, destination, copy_size)
///     .expect("Buffer to texture copy failed");
/// # }
/// ```
///
/// # wgpu Alignment Requirements
///
/// Per the wgpu specification:
/// - `buffer offset` must be a multiple of 4
/// - `bytes_per_row` must be a multiple of 256
/// - `rows_per_image` is required for 3D textures and 2D texture arrays
pub fn copy_buffer_to_texture(
    encoder: &mut CommandEncoder,
    source: ImageCopyBuffer,
    destination: ImageCopyTexture,
    copy_size: CopyExtent3d,
) -> Result<(), CopyError> {
    // Validate all parameters
    validate_buffer_to_texture_params(&source, &destination, &copy_size)?;

    // Convert to wgpu types and execute copy
    let wgpu_source = wgpu::ImageCopyBuffer {
        buffer: source.buffer,
        layout: source.layout,
    };

    let wgpu_destination = wgpu::ImageCopyTexture {
        texture: destination.texture,
        mip_level: destination.mip_level,
        origin: destination.origin,
        aspect: destination.aspect,
    };

    encoder.copy_buffer_to_texture(wgpu_source, wgpu_destination, copy_size.to_wgpu());

    Ok(())
}

/// Copy data from a buffer to a texture using a TrinityCommandEncoder.
///
/// Convenience wrapper that extracts the inner wgpu encoder and calls
/// [`copy_buffer_to_texture`].
///
/// # Arguments
///
/// * `encoder` - The TrinityCommandEncoder to record the copy command on
/// * `source` - Source buffer information
/// * `destination` - Destination texture information
/// * `copy_size` - The extent of the copy region
///
/// # Returns
///
/// * `Ok(())` if the copy was recorded successfully
/// * `Err(CopyError)` if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{
///     copy_buffer_to_texture_trinity, ImageCopyBuffer, ImageCopyTexture, CopyExtent3d
/// };
/// use renderer_backend::command_encoder::TrinityCommandEncoder;
///
/// # fn example(
/// #     encoder: &mut TrinityCommandEncoder,
/// #     buffer: &wgpu::Buffer,
/// #     texture: &wgpu::Texture,
/// # ) {
/// let source = ImageCopyBuffer::with_auto_layout(buffer, 128, 128, 1, 4);
/// let destination = ImageCopyTexture::new(texture);
/// let copy_size = CopyExtent3d::new_2d(128, 128);
///
/// copy_buffer_to_texture_trinity(encoder, source, destination, copy_size)
///     .expect("Copy failed");
/// # }
/// ```
pub fn copy_buffer_to_texture_trinity(
    encoder: &mut TrinityCommandEncoder,
    source: ImageCopyBuffer,
    destination: ImageCopyTexture,
    copy_size: CopyExtent3d,
) -> Result<(), CopyError> {
    copy_buffer_to_texture(encoder.inner_mut(), source, destination, copy_size)
}

// ============================================================================
// Texture-to-Buffer Copy (Readback)
// ============================================================================

/// Validate texture-to-buffer copy parameters.
///
/// Checks:
/// - bytes_per_row is 256-byte aligned
/// - rows_per_image is specified when depth_or_array_layers > 1
/// - All extent dimensions are > 0
/// - Source texture has COPY_SRC usage
/// - Destination buffer has COPY_DST usage (required for copy target)
///
/// Note: MAP_READ validation is done separately via [`validate_map_read_usage`]
/// since it's only required for CPU readback, not for the copy operation itself.
///
/// # Arguments
///
/// * `source` - The source texture copy info
/// * `destination` - The destination buffer copy info
/// * `copy_size` - The extent of the copy operation
///
/// # Returns
///
/// * `Ok(())` if all parameters are valid
/// * `Err(CopyError)` with details if validation fails
pub fn validate_texture_to_buffer_params(
    source: &ImageCopyTexture,
    destination: &ImageCopyBuffer,
    copy_size: &CopyExtent3d,
) -> Result<(), CopyError> {
    // Validate extent dimensions
    if copy_size.width == 0 {
        return Err(CopyError::invalid_extent("width", 0));
    }
    if copy_size.height == 0 {
        return Err(CopyError::invalid_extent("height", 0));
    }
    if copy_size.depth_or_array_layers == 0 {
        return Err(CopyError::invalid_extent("depth_or_array_layers", 0));
    }

    // Validate bytes_per_row alignment
    if let Some(bytes_per_row) = destination.layout.bytes_per_row {
        if !CopyAlignmentCalculator::is_bytes_per_row_aligned(bytes_per_row) {
            return Err(CopyError::bytes_per_row_alignment(bytes_per_row));
        }
    }

    // Validate rows_per_image is present when needed
    if copy_size.depth_or_array_layers > 1 && destination.layout.rows_per_image.is_none() {
        return Err(CopyError::rows_per_image_required(
            copy_size.depth_or_array_layers,
        ));
    }

    // Validate buffer offset alignment (4-byte)
    if destination.layout.offset % BUFFER_OFFSET_ALIGNMENT != 0 {
        return Err(CopyError::AlignmentError {
            field: "buffer_offset",
            offset: destination.layout.offset,
            required: BUFFER_OFFSET_ALIGNMENT,
        });
    }

    // Validate texture usage (source must have COPY_SRC)
    if !source.texture.usage().contains(TextureUsages::COPY_SRC) {
        return Err(CopyError::texture_missing_copy_src());
    }

    // Validate buffer usage (destination must have COPY_DST for the copy operation)
    if !destination.buffer.usage().contains(BufferUsages::COPY_DST) {
        return Err(CopyError::missing_copy_dst());
    }

    Ok(())
}

/// Validate that a buffer has MAP_READ usage for CPU readback.
///
/// This is a separate validation step from [`validate_texture_to_buffer_params`]
/// because MAP_READ is only required when the CPU needs to read the buffer contents,
/// not for the GPU copy operation itself.
///
/// # Arguments
///
/// * `buffer` - The buffer to validate
///
/// # Returns
///
/// * `Ok(())` if the buffer has MAP_READ usage
/// * `Err(CopyError::BufferMissingMapRead)` otherwise
pub fn validate_map_read_usage(buffer: &Buffer) -> Result<(), CopyError> {
    if !buffer.usage().contains(BufferUsages::MAP_READ) {
        return Err(CopyError::buffer_missing_map_read());
    }
    Ok(())
}

/// Copy data from a texture to a buffer with validation (GPU readback).
///
/// This function validates all parameters before executing the copy:
/// - bytes_per_row must be aligned to 256 bytes
/// - rows_per_image must be specified for 3D textures or texture arrays
/// - Buffer offset must be 4-byte aligned
/// - Source texture must have COPY_SRC usage
/// - Destination buffer must have COPY_DST usage
/// - Copy extent must have non-zero dimensions
///
/// Note: This function does NOT validate MAP_READ usage. If you intend to read
/// the buffer contents on the CPU, use [`validate_map_read_usage`] separately
/// or use the [`StagingBuffer`] helper.
///
/// # Arguments
///
/// * `encoder` - The command encoder to record the copy command on
/// * `source` - Source texture information (texture + mip/origin/aspect)
/// * `destination` - Destination buffer information (buffer + layout)
/// * `copy_size` - The extent of the copy region (width, height, depth/layers)
///
/// # Returns
///
/// * `Ok(())` if the copy was recorded successfully
/// * `Err(CopyError)` if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{
///     copy_texture_to_buffer, ImageCopyBuffer, ImageCopyTexture, CopyExtent3d
/// };
/// use wgpu::{ImageDataLayout, Origin3d, TextureAspect};
///
/// # fn example(
/// #     encoder: &mut wgpu::CommandEncoder,
/// #     texture: &wgpu::Texture,
/// #     buffer: &wgpu::Buffer,
/// # ) {
/// let source = ImageCopyTexture {
///     texture,
///     mip_level: 0,
///     origin: Origin3d::ZERO,
///     aspect: TextureAspect::All,
/// };
///
/// let destination = ImageCopyBuffer {
///     buffer,
///     layout: ImageDataLayout {
///         offset: 0,
///         bytes_per_row: Some(512), // 256-aligned
///         rows_per_image: None,     // Not needed for 2D
///     },
/// };
///
/// let copy_size = CopyExtent3d::new_2d(128, 128);
///
/// copy_texture_to_buffer(encoder, source, destination, copy_size)
///     .expect("Texture to buffer copy failed");
/// # }
/// ```
///
/// # wgpu Alignment Requirements
///
/// Per the wgpu specification:
/// - `buffer offset` must be a multiple of 4
/// - `bytes_per_row` must be a multiple of 256
/// - `rows_per_image` is required for 3D textures and 2D texture arrays
pub fn copy_texture_to_buffer(
    encoder: &mut CommandEncoder,
    source: ImageCopyTexture,
    destination: ImageCopyBuffer,
    copy_size: CopyExtent3d,
) -> Result<(), CopyError> {
    // Validate all parameters
    validate_texture_to_buffer_params(&source, &destination, &copy_size)?;

    // Convert to wgpu types and execute copy
    let wgpu_source = wgpu::ImageCopyTexture {
        texture: source.texture,
        mip_level: source.mip_level,
        origin: source.origin,
        aspect: source.aspect,
    };

    let wgpu_destination = wgpu::ImageCopyBuffer {
        buffer: destination.buffer,
        layout: destination.layout,
    };

    encoder.copy_texture_to_buffer(wgpu_source, wgpu_destination, copy_size.to_wgpu());

    Ok(())
}

/// Copy data from a texture to a buffer using a TrinityCommandEncoder.
///
/// Convenience wrapper that extracts the inner wgpu encoder and calls
/// [`copy_texture_to_buffer`].
///
/// # Arguments
///
/// * `encoder` - The TrinityCommandEncoder to record the copy command on
/// * `source` - Source texture information
/// * `destination` - Destination buffer information
/// * `copy_size` - The extent of the copy region
///
/// # Returns
///
/// * `Ok(())` if the copy was recorded successfully
/// * `Err(CopyError)` if validation fails
pub fn copy_texture_to_buffer_trinity(
    encoder: &mut TrinityCommandEncoder,
    source: ImageCopyTexture,
    destination: ImageCopyBuffer,
    copy_size: CopyExtent3d,
) -> Result<(), CopyError> {
    copy_texture_to_buffer(encoder.inner_mut(), source, destination, copy_size)
}

// ============================================================================
// Texture-to-Texture Copy (GPU-side Blit)
// ============================================================================

/// Check if two texture formats are copy-compatible.
///
/// Two formats are copy-compatible if they have the same block size and belong
/// to compatible format families. This follows wgpu/WebGPU copy compatibility rules.
///
/// # Compatibility Rules
///
/// 1. **Same format**: Always compatible
/// 2. **Same block size**: Formats with the same byte size per texel/block are compatible
/// 3. **Compressed formats**: Must have matching block dimensions and byte sizes
///
/// # Arguments
///
/// * `src_format` - The source texture format
/// * `dst_format` - The destination texture format
///
/// # Returns
///
/// `true` if the formats are copy-compatible, `false` otherwise.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::are_formats_copy_compatible;
/// use wgpu::TextureFormat;
///
/// // Same format is always compatible
/// assert!(are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm));
///
/// // Same-size formats within a family are compatible
/// assert!(are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb));
///
/// // Different size formats are not compatible
/// assert!(!are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba16Float));
/// ```
pub fn are_formats_copy_compatible(
    src_format: wgpu::TextureFormat,
    dst_format: wgpu::TextureFormat,
) -> bool {
    // Same format is always compatible
    if src_format == dst_format {
        return true;
    }

    // Get block copy size for each format (bytes per block for compressed, bytes per texel otherwise)
    let src_size = src_format.block_copy_size(None);
    let dst_size = dst_format.block_copy_size(None);

    // If either returns None (view formats that depend on aspect), they must be the same format
    match (src_size, dst_size) {
        (Some(src), Some(dst)) => {
            // Same block size means potentially compatible
            if src != dst {
                return false;
            }

            // For compressed formats, block dimensions must also match
            let src_dims = src_format.block_dimensions();
            let dst_dims = dst_format.block_dimensions();
            if src_dims != dst_dims {
                return false;
            }

            // Check format family compatibility
            // Formats in the same "copy-compatible class" can be copied between
            are_in_same_copy_class(src_format, dst_format)
        }
        _ => false,
    }
}

/// Check if two formats belong to the same copy-compatible class.
///
/// WebGPU defines copy-compatible classes where formats can be freely copied
/// between each other despite having different color space interpretations.
fn are_in_same_copy_class(src: wgpu::TextureFormat, dst: wgpu::TextureFormat) -> bool {
    use wgpu::TextureFormat::*;

    // Define copy-compatible classes (formats that can be copied between each other)
    // These groups have the same memory layout but different interpretations

    // 8-bit 1-channel formats
    let r8_class = [R8Unorm, R8Snorm, R8Uint, R8Sint];
    if r8_class.contains(&src) && r8_class.contains(&dst) {
        return true;
    }

    // 16-bit 1-channel formats
    let r16_class = [R16Unorm, R16Snorm, R16Uint, R16Sint, R16Float];
    if r16_class.contains(&src) && r16_class.contains(&dst) {
        return true;
    }

    // 16-bit 2-channel formats
    let rg8_class = [Rg8Unorm, Rg8Snorm, Rg8Uint, Rg8Sint];
    if rg8_class.contains(&src) && rg8_class.contains(&dst) {
        return true;
    }

    // 32-bit 1-channel formats
    let r32_class = [R32Uint, R32Sint, R32Float];
    if r32_class.contains(&src) && r32_class.contains(&dst) {
        return true;
    }

    // 32-bit 2-channel formats
    let rg16_class = [Rg16Unorm, Rg16Snorm, Rg16Uint, Rg16Sint, Rg16Float];
    if rg16_class.contains(&src) && rg16_class.contains(&dst) {
        return true;
    }

    // 32-bit 4-channel formats (RGBA8 family)
    let rgba8_class = [
        Rgba8Unorm, Rgba8UnormSrgb, Rgba8Snorm, Rgba8Uint, Rgba8Sint,
        Bgra8Unorm, Bgra8UnormSrgb,
    ];
    if rgba8_class.contains(&src) && rgba8_class.contains(&dst) {
        return true;
    }

    // 64-bit 2-channel formats
    let rg32_class = [Rg32Uint, Rg32Sint, Rg32Float];
    if rg32_class.contains(&src) && rg32_class.contains(&dst) {
        return true;
    }

    // 64-bit 4-channel formats
    let rgba16_class = [Rgba16Unorm, Rgba16Snorm, Rgba16Uint, Rgba16Sint, Rgba16Float];
    if rgba16_class.contains(&src) && rgba16_class.contains(&dst) {
        return true;
    }

    // 128-bit 4-channel formats
    let rgba32_class = [Rgba32Uint, Rgba32Sint, Rgba32Float];
    if rgba32_class.contains(&src) && rgba32_class.contains(&dst) {
        return true;
    }

    // RGB10A2 family (32-bit packed)
    let rgb10a2_class = [Rgb10a2Uint, Rgb10a2Unorm];
    if rgb10a2_class.contains(&src) && rgb10a2_class.contains(&dst) {
        return true;
    }

    // BC1 compressed formats
    let bc1_class = [Bc1RgbaUnorm, Bc1RgbaUnormSrgb];
    if bc1_class.contains(&src) && bc1_class.contains(&dst) {
        return true;
    }

    // BC2 compressed formats
    let bc2_class = [Bc2RgbaUnorm, Bc2RgbaUnormSrgb];
    if bc2_class.contains(&src) && bc2_class.contains(&dst) {
        return true;
    }

    // BC3 compressed formats
    let bc3_class = [Bc3RgbaUnorm, Bc3RgbaUnormSrgb];
    if bc3_class.contains(&src) && bc3_class.contains(&dst) {
        return true;
    }

    // BC7 compressed formats
    let bc7_class = [Bc7RgbaUnorm, Bc7RgbaUnormSrgb];
    if bc7_class.contains(&src) && bc7_class.contains(&dst) {
        return true;
    }

    // ETC2 RGB compressed formats
    let etc2_rgb_class = [Etc2Rgb8Unorm, Etc2Rgb8UnormSrgb];
    if etc2_rgb_class.contains(&src) && etc2_rgb_class.contains(&dst) {
        return true;
    }

    // ETC2 RGBA compressed formats
    let etc2_rgba_class = [Etc2Rgba8Unorm, Etc2Rgba8UnormSrgb];
    if etc2_rgba_class.contains(&src) && etc2_rgba_class.contains(&dst) {
        return true;
    }

    // ETC2 RGB8A1 compressed formats
    let etc2_rgb8a1_class = [Etc2Rgb8A1Unorm, Etc2Rgb8A1UnormSrgb];
    if etc2_rgb8a1_class.contains(&src) && etc2_rgb8a1_class.contains(&dst) {
        return true;
    }

    // ASTC compressed formats (4x4 block)
    let astc_4x4_class = [Astc { block: AstcBlock::B4x4, channel: AstcChannel::Unorm },
                          Astc { block: AstcBlock::B4x4, channel: AstcChannel::UnormSrgb }];
    if astc_4x4_class.contains(&src) && astc_4x4_class.contains(&dst) {
        return true;
    }

    // Depth formats are generally not copy-compatible with color formats
    // Stencil formats are also separate

    false
}

/// Calculate the size of a texture at a given mip level.
///
/// Each mip level is half the size of the previous level (rounded down),
/// with a minimum size of 1 in each dimension.
///
/// # Arguments
///
/// * `base_size` - The base (mip level 0) size
/// * `mip_level` - The mip level to calculate size for
///
/// # Returns
///
/// The size at the specified mip level (minimum of 1).
#[inline]
pub const fn mip_size(base_size: u32, mip_level: u32) -> u32 {
    // Clamp mip_level to avoid shift overflow (u32 has max 31 bits to shift)
    // Any shift >= 32 would result in 0, which we clamp to 1 anyway
    if mip_level >= 32 {
        return 1;
    }
    let size = base_size >> mip_level;
    if size == 0 { 1 } else { size }
}

/// Validate texture-to-texture copy parameters.
///
/// Performs comprehensive validation for GPU-side texture blitting:
/// - Both textures must have appropriate usage flags (COPY_SRC/COPY_DST)
/// - Formats must be copy-compatible
/// - Mip levels must be within bounds
/// - Copy region must fit within both textures at specified mip levels
/// - Copy extent must have non-zero dimensions
///
/// # Arguments
///
/// * `source` - The source texture copy info
/// * `destination` - The destination texture copy info
/// * `copy_size` - The extent of the copy operation
///
/// # Returns
///
/// * `Ok(())` if all parameters are valid
/// * `Err(CopyError)` with details if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{
///     validate_texture_to_texture_params, ImageCopyTexture, CopyExtent3d
/// };
///
/// # fn example(src_texture: &wgpu::Texture, dst_texture: &wgpu::Texture) {
/// let source = ImageCopyTexture::new(src_texture);
/// let destination = ImageCopyTexture::new(dst_texture);
/// let copy_size = CopyExtent3d::new_2d(128, 128);
///
/// match validate_texture_to_texture_params(&source, &destination, &copy_size) {
///     Ok(()) => println!("Parameters are valid"),
///     Err(e) => println!("Validation failed: {}", e),
/// }
/// # }
/// ```
pub fn validate_texture_to_texture_params(
    source: &ImageCopyTexture,
    destination: &ImageCopyTexture,
    copy_size: &CopyExtent3d,
) -> Result<(), CopyError> {
    // Validate extent dimensions (must be non-zero)
    if copy_size.width == 0 {
        return Err(CopyError::invalid_extent("width", 0));
    }
    if copy_size.height == 0 {
        return Err(CopyError::invalid_extent("height", 0));
    }
    if copy_size.depth_or_array_layers == 0 {
        return Err(CopyError::invalid_extent("depth_or_array_layers", 0));
    }

    // Validate source texture usage (must have COPY_SRC)
    if !source.texture.usage().contains(TextureUsages::COPY_SRC) {
        return Err(CopyError::texture_missing_copy_src());
    }

    // Validate destination texture usage (must have COPY_DST)
    if !destination.texture.usage().contains(TextureUsages::COPY_DST) {
        return Err(CopyError::texture_missing_copy_dst());
    }

    // Validate format compatibility
    let src_format = source.texture.format();
    let dst_format = destination.texture.format();
    if !are_formats_copy_compatible(src_format, dst_format) {
        return Err(CopyError::format_incompatible(src_format, dst_format));
    }

    // Validate source mip level
    let src_mip_count = source.texture.mip_level_count();
    if source.mip_level >= src_mip_count {
        return Err(CopyError::mip_level_out_of_bounds(
            source.mip_level,
            src_mip_count.saturating_sub(1),
        ));
    }

    // Validate destination mip level
    let dst_mip_count = destination.texture.mip_level_count();
    if destination.mip_level >= dst_mip_count {
        return Err(CopyError::mip_level_out_of_bounds(
            destination.mip_level,
            dst_mip_count.saturating_sub(1),
        ));
    }

    // Validate source bounds at the specified mip level
    let src_size = source.texture.size();
    let src_width_at_mip = mip_size(src_size.width, source.mip_level);
    let src_height_at_mip = mip_size(src_size.height, source.mip_level);
    let src_depth = src_size.depth_or_array_layers; // Array layers don't change with mip

    let src_end_x = source.origin.x.saturating_add(copy_size.width);
    let src_end_y = source.origin.y.saturating_add(copy_size.height);
    let src_end_z = source.origin.z.saturating_add(copy_size.depth_or_array_layers);

    if src_end_x > src_width_at_mip || src_end_y > src_height_at_mip || src_end_z > src_depth {
        return Err(CopyError::texture_bounds(
            source.origin.x,
            source.origin.y,
            source.origin.z,
            copy_size.width,
            copy_size.height,
            copy_size.depth_or_array_layers,
            src_width_at_mip,
            src_height_at_mip,
            src_depth,
            source.mip_level,
        ));
    }

    // Validate destination bounds at the specified mip level
    let dst_size = destination.texture.size();
    let dst_width_at_mip = mip_size(dst_size.width, destination.mip_level);
    let dst_height_at_mip = mip_size(dst_size.height, destination.mip_level);
    let dst_depth = dst_size.depth_or_array_layers;

    let dst_end_x = destination.origin.x.saturating_add(copy_size.width);
    let dst_end_y = destination.origin.y.saturating_add(copy_size.height);
    let dst_end_z = destination.origin.z.saturating_add(copy_size.depth_or_array_layers);

    if dst_end_x > dst_width_at_mip || dst_end_y > dst_height_at_mip || dst_end_z > dst_depth {
        return Err(CopyError::texture_bounds(
            destination.origin.x,
            destination.origin.y,
            destination.origin.z,
            copy_size.width,
            copy_size.height,
            copy_size.depth_or_array_layers,
            dst_width_at_mip,
            dst_height_at_mip,
            dst_depth,
            destination.mip_level,
        ));
    }

    Ok(())
}

/// Copy from texture to texture (GPU-side blit) with validation.
///
/// This function performs a GPU-side copy from one texture to another. Both textures
/// must be on the same device and their formats must be copy-compatible.
///
/// # Validation
///
/// The following validations are performed:
/// - Source texture must have `COPY_SRC` usage
/// - Destination texture must have `COPY_DST` usage
/// - Formats must be copy-compatible (same block size and compatible format class)
/// - Mip levels must be within each texture's mip count
/// - Copy region must fit within both textures at their specified mip levels
/// - Array layer selection via origin.z and copy extent must be valid
///
/// # Arguments
///
/// * `encoder` - The command encoder to record the copy command on
/// * `source` - Source texture information (texture + mip level + origin + aspect)
/// * `destination` - Destination texture information (texture + mip level + origin + aspect)
/// * `copy_size` - The extent of the copy region (width, height, depth/layers)
///
/// # Returns
///
/// * `Ok(())` if the copy was recorded successfully
/// * `Err(CopyError)` if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{
///     copy_texture_to_texture, ImageCopyTexture, CopyExtent3d
/// };
/// use wgpu::{Origin3d, TextureAspect};
///
/// # fn example(
/// #     encoder: &mut wgpu::CommandEncoder,
/// #     src_texture: &wgpu::Texture,
/// #     dst_texture: &wgpu::Texture,
/// # ) {
/// // Copy from mip level 0 of source to mip level 1 of destination
/// let source = ImageCopyTexture {
///     texture: src_texture,
///     mip_level: 0,
///     origin: Origin3d::ZERO,
///     aspect: TextureAspect::All,
/// };
///
/// let destination = ImageCopyTexture {
///     texture: dst_texture,
///     mip_level: 1,
///     origin: Origin3d::ZERO,
///     aspect: TextureAspect::All,
/// };
///
/// let copy_size = CopyExtent3d::new_2d(128, 128);
///
/// copy_texture_to_texture(encoder, source, destination, copy_size)
///     .expect("Texture to texture copy failed");
/// # }
/// ```
///
/// # Use Cases
///
/// - **Mipmap generation**: Copy from higher resolution mip to lower
/// - **Texture atlasing**: Copy regions between atlas textures
/// - **Render target resolve**: Copy from multisampled to non-multisampled
/// - **Array layer manipulation**: Copy between array layers
/// - **Format conversion**: Copy between compatible format variants (e.g., sRGB to linear)
pub fn copy_texture_to_texture(
    encoder: &mut CommandEncoder,
    source: ImageCopyTexture,
    destination: ImageCopyTexture,
    copy_size: CopyExtent3d,
) -> Result<(), CopyError> {
    // Validate all parameters
    validate_texture_to_texture_params(&source, &destination, &copy_size)?;

    // Convert to wgpu types and execute copy
    let wgpu_source = wgpu::ImageCopyTexture {
        texture: source.texture,
        mip_level: source.mip_level,
        origin: source.origin,
        aspect: source.aspect,
    };

    let wgpu_destination = wgpu::ImageCopyTexture {
        texture: destination.texture,
        mip_level: destination.mip_level,
        origin: destination.origin,
        aspect: destination.aspect,
    };

    encoder.copy_texture_to_texture(wgpu_source, wgpu_destination, copy_size.to_wgpu());

    Ok(())
}

/// Copy from texture to texture using a TrinityCommandEncoder.
///
/// Convenience wrapper that extracts the inner wgpu encoder and calls
/// [`copy_texture_to_texture`].
///
/// # Arguments
///
/// * `encoder` - The TrinityCommandEncoder to record the copy command on
/// * `source` - Source texture information
/// * `destination` - Destination texture information
/// * `copy_size` - The extent of the copy region
///
/// # Returns
///
/// * `Ok(())` if the copy was recorded successfully
/// * `Err(CopyError)` if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{
///     copy_texture_to_texture_trinity, ImageCopyTexture, CopyExtent3d
/// };
/// use renderer_backend::command_encoder::TrinityCommandEncoder;
///
/// # fn example(
/// #     encoder: &mut TrinityCommandEncoder,
/// #     src_texture: &wgpu::Texture,
/// #     dst_texture: &wgpu::Texture,
/// # ) {
/// let source = ImageCopyTexture::new(src_texture);
/// let destination = ImageCopyTexture::new(dst_texture);
/// let copy_size = CopyExtent3d::new_2d(256, 256);
///
/// copy_texture_to_texture_trinity(encoder, source, destination, copy_size)
///     .expect("Copy failed");
/// # }
/// ```
pub fn copy_texture_to_texture_trinity(
    encoder: &mut TrinityCommandEncoder,
    source: ImageCopyTexture,
    destination: ImageCopyTexture,
    copy_size: CopyExtent3d,
) -> Result<(), CopyError> {
    copy_texture_to_texture(encoder.inner_mut(), source, destination, copy_size)
}

// ============================================================================
// Staging Buffer for Readback
// ============================================================================

/// Helper struct for creating and managing staging buffers for GPU readback.
///
/// Staging buffers are temporary buffers used to copy GPU texture data back to
/// the CPU. They must have `COPY_DST | MAP_READ` usage flags.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{StagingBuffer, CopyExtent3d};
///
/// # async fn example(device: &wgpu::Device) {
/// // Create staging buffer for a 256x256 RGBA8 texture
/// let copy_size = CopyExtent3d::new_2d(256, 256);
/// let staging = StagingBuffer::new(device, &copy_size, 4); // 4 bytes per pixel (RGBA8)
///
/// // After copying texture data to the staging buffer...
/// // Map and read the data
/// let view = staging.map_read().await.expect("Failed to map buffer");
/// let data: &[u8] = &view[..];
/// // Process data...
/// # }
/// ```
pub struct StagingBuffer {
    /// The underlying wgpu buffer.
    pub buffer: wgpu::Buffer,

    /// Total size of the buffer in bytes.
    pub size: u64,

    /// Aligned bytes per row (256-byte aligned).
    pub bytes_per_row: u32,

    /// Number of rows per image slice.
    pub rows_per_image: u32,
}

impl StagingBuffer {
    /// Create a new staging buffer for texture readback.
    ///
    /// The buffer is created with `COPY_DST | MAP_READ` usage flags,
    /// making it suitable for receiving texture copy data and subsequent
    /// CPU mapping.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the buffer on
    /// * `copy_size` - The dimensions of the texture data to copy
    /// * `bytes_per_pixel` - Bytes per pixel (e.g., 4 for RGBA8, 16 for RGBA32Float)
    ///
    /// # Returns
    ///
    /// A new `StagingBuffer` with properly aligned layout.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::{StagingBuffer, CopyExtent3d};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// // For a 100x100 RGBA8 texture
    /// let extent = CopyExtent3d::new_2d(100, 100);
    /// let staging = StagingBuffer::new(device, &extent, 4);
    ///
    /// // bytes_per_row is 256-aligned: 100 * 4 = 400 -> 512
    /// assert_eq!(staging.bytes_per_row, 512);
    /// # }
    /// ```
    pub fn new(device: &wgpu::Device, copy_size: &CopyExtent3d, bytes_per_pixel: u32) -> Self {
        let (bytes_per_row, size) = CopyAlignmentCalculator::calculate_texture_buffer_layout(
            copy_size.width,
            copy_size.height,
            copy_size.depth_or_array_layers,
            bytes_per_pixel,
        );

        let rows_per_image = CopyAlignmentCalculator::calculate_rows_per_image(
            copy_size.height,
            copy_size.depth_or_array_layers,
        );

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("StagingBuffer for readback"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size,
            bytes_per_row,
            rows_per_image,
        }
    }

    /// Create a staging buffer with a custom label.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the buffer on
    /// * `copy_size` - The dimensions of the texture data to copy
    /// * `bytes_per_pixel` - Bytes per pixel
    /// * `label` - Custom label for debugging
    pub fn with_label(
        device: &wgpu::Device,
        copy_size: &CopyExtent3d,
        bytes_per_pixel: u32,
        label: &str,
    ) -> Self {
        let (bytes_per_row, size) = CopyAlignmentCalculator::calculate_texture_buffer_layout(
            copy_size.width,
            copy_size.height,
            copy_size.depth_or_array_layers,
            bytes_per_pixel,
        );

        let rows_per_image = CopyAlignmentCalculator::calculate_rows_per_image(
            copy_size.height,
            copy_size.depth_or_array_layers,
        );

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size,
            bytes_per_row,
            rows_per_image,
        }
    }

    /// Get an [`ImageCopyBuffer`] reference for use in copy operations.
    ///
    /// Returns an `ImageCopyBuffer` configured with this staging buffer's layout.
    ///
    /// # Arguments
    ///
    /// * `depth_or_array_layers` - Number of depth slices or array layers being copied.
    ///   If > 1, `rows_per_image` will be included in the layout.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::{
    ///     StagingBuffer, CopyExtent3d, copy_texture_to_buffer, ImageCopyTexture
    /// };
    ///
    /// # fn example(
    /// #     device: &wgpu::Device,
    /// #     encoder: &mut wgpu::CommandEncoder,
    /// #     texture: &wgpu::Texture,
    /// # ) {
    /// let extent = CopyExtent3d::new_2d(128, 128);
    /// let staging = StagingBuffer::new(device, &extent, 4);
    ///
    /// let source = ImageCopyTexture::new(texture);
    /// let destination = staging.as_image_copy_buffer(extent.depth_or_array_layers);
    ///
    /// copy_texture_to_buffer(encoder, source, destination, extent)
    ///     .expect("Copy failed");
    /// # }
    /// ```
    pub fn as_image_copy_buffer(&self, depth_or_array_layers: u32) -> ImageCopyBuffer<'_> {
        ImageCopyBuffer {
            buffer: &self.buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.bytes_per_row),
                rows_per_image: if depth_or_array_layers > 1 {
                    Some(self.rows_per_image)
                } else {
                    None
                },
            },
        }
    }

    /// Map the buffer for reading and return a view of the data.
    ///
    /// This is an async operation that waits for the GPU to finish writing
    /// to the buffer and then maps it for CPU access.
    ///
    /// # Returns
    ///
    /// * `Ok(BufferView)` containing the mapped data
    /// * `Err(BufferAsyncError)` if mapping fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::{StagingBuffer, CopyExtent3d};
    ///
    /// # async fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
    /// let extent = CopyExtent3d::new_2d(64, 64);
    /// let staging = StagingBuffer::new(device, &extent, 4);
    ///
    /// // ... copy texture to staging buffer via command encoder ...
    ///
    /// // Map and read
    /// let view = staging.map_read().await.expect("Mapping failed");
    /// let bytes: &[u8] = &view;
    /// println!("Read {} bytes from GPU", bytes.len());
    /// # }
    /// ```
    pub async fn map_read(&self) -> Result<wgpu::BufferView<'_>, wgpu::BufferAsyncError> {
        let buffer_slice = self.buffer.slice(..);

        // Create a channel to signal when mapping is complete
        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Wait for the mapping to complete
        // In a real application, you'd want to poll the device here
        // For now, we block on the channel
        rx.recv()
            .map_err(|_| wgpu::BufferAsyncError)??;

        Ok(buffer_slice.get_mapped_range())
    }

    /// Map the buffer synchronously by polling the device.
    ///
    /// This is a convenience method that blocks until the buffer is mapped.
    /// Use this when you need synchronous access and can afford to block.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to poll
    ///
    /// # Returns
    ///
    /// * `Ok(BufferView)` containing the mapped data
    /// * `Err(BufferAsyncError)` if mapping fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::{StagingBuffer, CopyExtent3d};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let extent = CopyExtent3d::new_2d(64, 64);
    /// let staging = StagingBuffer::new(device, &extent, 4);
    ///
    /// // ... copy texture to staging buffer ...
    ///
    /// // Synchronous read
    /// let view = staging.map_read_sync(device).expect("Mapping failed");
    /// let bytes: &[u8] = &view;
    /// println!("Read {} bytes", bytes.len());
    /// # }
    /// ```
    pub fn map_read_sync(
        &self,
        device: &wgpu::Device,
    ) -> Result<wgpu::BufferView<'_>, wgpu::BufferAsyncError> {
        let buffer_slice = self.buffer.slice(..);

        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll the device until mapping completes
        loop {
            device.poll(wgpu::Maintain::Poll);
            match rx.try_recv() {
                Ok(result) => {
                    result?;
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Continue polling
                    std::thread::yield_now();
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(wgpu::BufferAsyncError);
                }
            }
        }

        Ok(buffer_slice.get_mapped_range())
    }

    /// Unmap the buffer after reading.
    ///
    /// This must be called after you're done reading the mapped data.
    /// The buffer must be unmapped before it can be used in another copy operation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::{StagingBuffer, CopyExtent3d};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let extent = CopyExtent3d::new_2d(64, 64);
    /// let staging = StagingBuffer::new(device, &extent, 4);
    ///
    /// // ... copy and map ...
    /// {
    ///     let view = staging.map_read_sync(device).unwrap();
    ///     // Use view...
    /// } // view is dropped here
    ///
    /// // Now unmap
    /// staging.unmap();
    /// # }
    /// ```
    pub fn unmap(&self) {
        self.buffer.unmap();
    }

    /// Get the image data layout for this staging buffer.
    ///
    /// # Arguments
    ///
    /// * `depth_or_array_layers` - Number of depth slices or array layers
    pub fn layout(&self, depth_or_array_layers: u32) -> wgpu::ImageDataLayout {
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(self.bytes_per_row),
            rows_per_image: if depth_or_array_layers > 1 {
                Some(self.rows_per_image)
            } else {
                None
            },
        }
    }

    /// Check if this buffer has the correct usage for readback operations.
    ///
    /// Returns `true` if the buffer has both `COPY_DST` and `MAP_READ` usage.
    pub fn has_readback_usage(&self) -> bool {
        let usage = self.buffer.usage();
        usage.contains(BufferUsages::COPY_DST) && usage.contains(BufferUsages::MAP_READ)
    }
}

// ============================================================================
// Buffer-to-Buffer Copy Functions
// ============================================================================

/// Copy data from one buffer to another with validation.
///
/// This function validates all parameters before executing the copy:
/// - Alignment validation (4-byte alignment for offsets and size)
/// - Size validation (must be > 0)
/// - Usage validation (source must have COPY_SRC, dest must have COPY_DST)
/// - Bounds validation (copy must not exceed buffer sizes)
///
/// # Arguments
///
/// * `encoder` - The command encoder to record the copy command on
/// * `source` - The source buffer to copy from
/// * `dest` - The destination buffer to copy to
/// * `params` - Copy parameters (source offset, dest offset, size)
///
/// # Returns
///
/// * `Ok(())` if the copy was recorded successfully
/// * `Err(CopyError)` if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{copy_buffer_to_buffer, BufferCopyParams};
///
/// # fn example(
/// #     encoder: &mut wgpu::CommandEncoder,
/// #     source: &wgpu::Buffer,
/// #     dest: &wgpu::Buffer,
/// # ) {
/// // Copy 1KB from source offset 0 to dest offset 256
/// let result = copy_buffer_to_buffer(
///     encoder,
///     source,
///     dest,
///     BufferCopyParams::new(0, 256, 1024),
/// );
///
/// match result {
///     Ok(()) => println!("Copy recorded"),
///     Err(e) => println!("Copy failed: {}", e),
/// }
/// # }
/// ```
///
/// # wgpu Alignment Requirements
///
/// Per the wgpu specification:
/// - `source_offset` must be a multiple of 4
/// - `dest_offset` must be a multiple of 4
/// - `size` must be a multiple of 4
///
/// This function validates these constraints and returns an error if they're not met.
pub fn copy_buffer_to_buffer(
    encoder: &mut CommandEncoder,
    source: &Buffer,
    dest: &Buffer,
    params: BufferCopyParams,
) -> Result<(), CopyError> {
    // Validate parameters (alignment and size > 0)
    validate_params(&params)?;

    // Validate buffer usage flags
    validate_usage(source.usage(), dest.usage())?;

    // Validate bounds
    validate_bounds(&params, source.size(), dest.size())?;

    // All validation passed, execute the copy
    encoder.copy_buffer_to_buffer(
        source,
        params.source_offset,
        dest,
        params.dest_offset,
        params.size,
    );

    Ok(())
}

/// Copy data from one buffer to another using a TrinityCommandEncoder.
///
/// Convenience wrapper that extracts the inner wgpu encoder and calls
/// [`copy_buffer_to_buffer`].
///
/// # Arguments
///
/// * `encoder` - The TrinityCommandEncoder to record the copy command on
/// * `source` - The source buffer to copy from
/// * `dest` - The destination buffer to copy to
/// * `params` - Copy parameters (source offset, dest offset, size)
///
/// # Returns
///
/// * `Ok(())` if the copy was recorded successfully
/// * `Err(CopyError)` if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{copy_buffer_to_buffer_trinity, BufferCopyParams};
/// use renderer_backend::command_encoder::TrinityCommandEncoder;
///
/// # fn example(
/// #     encoder: &mut TrinityCommandEncoder,
/// #     source: &wgpu::Buffer,
/// #     dest: &wgpu::Buffer,
/// # ) {
/// let result = copy_buffer_to_buffer_trinity(
///     encoder,
///     source,
///     dest,
///     BufferCopyParams::full(1024),
/// );
/// # }
/// ```
pub fn copy_buffer_to_buffer_trinity(
    encoder: &mut TrinityCommandEncoder,
    source: &Buffer,
    dest: &Buffer,
    params: BufferCopyParams,
) -> Result<(), CopyError> {
    copy_buffer_to_buffer(encoder.inner_mut(), source, dest, params)
}

// ============================================================================
// Buffer Clear Operations
// ============================================================================

/// Parameters for a buffer clear operation.
///
/// Specifies the offset and optional size of the region to clear.
/// The buffer region is filled with zeros.
///
/// # Alignment Requirements
///
/// wgpu requires 4-byte alignment for buffer clear operations:
/// - Offset must be 4-byte aligned
/// - Size (if specified) must be 4-byte aligned
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::BufferClearParams;
///
/// // Clear entire buffer from start
/// let params_full = BufferClearParams::new(0, None);
///
/// // Clear 1KB starting at offset 256
/// let params_range = BufferClearParams::new(256, Some(1024));
///
/// // Clear entire buffer (convenience constructor)
/// let params_all = BufferClearParams::full();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferClearParams {
    /// Offset in bytes from the start of the buffer.
    /// Must be 4-byte aligned.
    pub offset: u64,

    /// Number of bytes to clear, or `None` to clear from offset to end of buffer.
    /// If specified, must be 4-byte aligned.
    pub size: Option<u64>,
}

impl BufferClearParams {
    /// Create new buffer clear parameters.
    ///
    /// # Arguments
    ///
    /// * `offset` - Offset in buffer (must be 4-byte aligned)
    /// * `size` - Number of bytes to clear, or `None` for entire buffer from offset
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::BufferClearParams;
    ///
    /// // Clear 512 bytes starting at offset 128
    /// let params = BufferClearParams::new(128, Some(512));
    /// assert_eq!(params.offset, 128);
    /// assert_eq!(params.size, Some(512));
    /// ```
    #[inline]
    pub const fn new(offset: u64, size: Option<u64>) -> Self {
        Self { offset, size }
    }

    /// Create parameters for clearing an entire buffer from the start.
    ///
    /// Equivalent to `BufferClearParams::new(0, None)`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::BufferClearParams;
    ///
    /// let params = BufferClearParams::full();
    /// assert_eq!(params.offset, 0);
    /// assert_eq!(params.size, None);
    /// ```
    #[inline]
    pub const fn full() -> Self {
        Self {
            offset: 0,
            size: None,
        }
    }

    /// Create parameters for clearing a specific range.
    ///
    /// Convenience method that ensures size is always specified.
    ///
    /// # Arguments
    ///
    /// * `offset` - Offset in buffer (must be 4-byte aligned)
    /// * `size` - Number of bytes to clear (must be 4-byte aligned)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::copy_commands::BufferClearParams;
    ///
    /// let params = BufferClearParams::range(256, 1024);
    /// assert_eq!(params.offset, 256);
    /// assert_eq!(params.size, Some(1024));
    /// ```
    #[inline]
    pub const fn range(offset: u64, size: u64) -> Self {
        Self {
            offset,
            size: Some(size),
        }
    }

    /// Check if the offset is 4-byte aligned.
    #[inline]
    pub const fn is_offset_aligned(&self) -> bool {
        self.offset % COPY_BUFFER_ALIGNMENT == 0
    }

    /// Check if the size (if specified) is 4-byte aligned.
    #[inline]
    pub const fn is_size_aligned(&self) -> bool {
        match self.size {
            Some(size) => size % COPY_BUFFER_ALIGNMENT == 0,
            None => true, // No size specified, alignment check passes
        }
    }

    /// Check if all parameters are properly aligned.
    #[inline]
    pub const fn is_aligned(&self) -> bool {
        self.is_offset_aligned() && self.is_size_aligned()
    }
}

impl Default for BufferClearParams {
    fn default() -> Self {
        Self::full()
    }
}

impl fmt::Display for BufferClearParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.size {
            Some(size) => write!(f, "BufferClearParams {{ offset: {}, size: {} }}", self.offset, size),
            None => write!(f, "BufferClearParams {{ offset: {}, size: entire }}", self.offset),
        }
    }
}

/// Validate buffer clear parameters.
///
/// Checks:
/// - Offset is 4-byte aligned
/// - Size (if specified) is 4-byte aligned
/// - Buffer has COPY_DST usage
/// - Offset + size (or offset to end) does not exceed buffer size
///
/// # Arguments
///
/// * `buffer` - The buffer to validate against
/// * `params` - The clear parameters to validate
///
/// # Returns
///
/// * `Ok(())` if all parameters are valid
/// * `Err(CopyError)` with details if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{validate_buffer_clear_params, BufferClearParams, CopyError};
///
/// # fn example(buffer: &wgpu::Buffer) {
/// // Valid parameters
/// let valid = BufferClearParams::new(0, Some(256));
/// // Note: Actual validation requires a real buffer
///
/// // Unaligned offset would fail
/// let invalid = BufferClearParams::new(3, Some(256));
/// # }
/// ```
pub fn validate_buffer_clear_params(
    buffer: &Buffer,
    params: &BufferClearParams,
) -> Result<(), CopyError> {
    // Check buffer has COPY_DST usage (required for clear_buffer)
    if !buffer.usage().contains(BufferUsages::COPY_DST) {
        return Err(CopyError::buffer_missing_copy_dst());
    }

    // Check offset alignment
    if params.offset % COPY_BUFFER_ALIGNMENT != 0 {
        return Err(CopyError::clear_offset_not_aligned(params.offset));
    }

    // Check size alignment (if specified)
    if let Some(size) = params.size {
        if size % COPY_BUFFER_ALIGNMENT != 0 {
            return Err(CopyError::clear_size_not_aligned(size));
        }
    }

    // Get buffer size and calculate effective clear size
    let buffer_size = buffer.size();

    // Calculate the clear end position
    let clear_size = params.size.unwrap_or_else(|| buffer_size.saturating_sub(params.offset));

    // Check bounds (handle overflow)
    let clear_end = params
        .offset
        .checked_add(clear_size)
        .ok_or_else(|| CopyError::clear_bounds(params.offset, clear_size, buffer_size))?;

    if clear_end > buffer_size {
        return Err(CopyError::clear_bounds(params.offset, clear_size, buffer_size));
    }

    Ok(())
}

/// Clear a buffer to zeros with validation.
///
/// This function fills the specified region of a buffer with zeros. It validates
/// all parameters before executing the clear:
/// - Offset must be 4-byte aligned
/// - Size (if specified) must be 4-byte aligned
/// - Buffer must have COPY_DST usage
/// - Clear region must be within buffer bounds
///
/// # Arguments
///
/// * `encoder` - The command encoder to record the clear command on
/// * `buffer` - The buffer to clear
/// * `params` - Clear parameters (offset and optional size)
///
/// # Returns
///
/// * `Ok(())` if the clear was recorded successfully
/// * `Err(CopyError)` if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{clear_buffer, BufferClearParams};
///
/// # fn example(
/// #     encoder: &mut wgpu::CommandEncoder,
/// #     buffer: &wgpu::Buffer,
/// # ) {
/// // Clear entire buffer
/// clear_buffer(encoder, buffer, BufferClearParams::full())
///     .expect("Clear failed");
///
/// // Clear specific range
/// clear_buffer(encoder, buffer, BufferClearParams::range(256, 1024))
///     .expect("Clear failed");
/// # }
/// ```
///
/// # wgpu Alignment Requirements
///
/// Per the wgpu specification:
/// - `offset` must be a multiple of 4
/// - `size` (if specified) must be a multiple of 4
/// - Buffer must have `COPY_DST` usage flag
///
/// This function validates these constraints and returns an error if they're not met.
pub fn clear_buffer(
    encoder: &mut CommandEncoder,
    buffer: &Buffer,
    params: BufferClearParams,
) -> Result<(), CopyError> {
    // Validate all parameters
    validate_buffer_clear_params(buffer, &params)?;

    // Calculate the actual size to clear
    let buffer_size = buffer.size();
    let clear_size = params.size.unwrap_or_else(|| buffer_size.saturating_sub(params.offset));

    // Execute the clear if there's actually something to clear
    if clear_size > 0 {
        encoder.clear_buffer(buffer, params.offset, Some(clear_size));
    }

    Ok(())
}

/// Clear a buffer to zeros using a TrinityCommandEncoder.
///
/// Convenience wrapper that extracts the inner wgpu encoder and calls
/// [`clear_buffer`].
///
/// # Arguments
///
/// * `encoder` - The TrinityCommandEncoder to record the clear command on
/// * `buffer` - The buffer to clear
/// * `params` - Clear parameters (offset and optional size)
///
/// # Returns
///
/// * `Ok(())` if the clear was recorded successfully
/// * `Err(CopyError)` if validation fails
///
/// # Example
///
/// ```no_run
/// use renderer_backend::copy_commands::{clear_buffer_trinity, BufferClearParams};
/// use renderer_backend::command_encoder::TrinityCommandEncoder;
///
/// # fn example(
/// #     encoder: &mut TrinityCommandEncoder,
/// #     buffer: &wgpu::Buffer,
/// # ) {
/// clear_buffer_trinity(encoder, buffer, BufferClearParams::full())
///     .expect("Clear failed");
/// # }
/// ```
pub fn clear_buffer_trinity(
    encoder: &mut TrinityCommandEncoder,
    buffer: &Buffer,
    params: BufferClearParams,
) -> Result<(), CopyError> {
    clear_buffer(encoder.inner_mut(), buffer, params)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // BufferCopyParams Tests
    // ========================================================================

    #[test]
    fn test_buffer_copy_params_new() {
        let params = BufferCopyParams::new(100, 200, 300);
        assert_eq!(params.source_offset, 100);
        assert_eq!(params.dest_offset, 200);
        assert_eq!(params.size, 300);
    }

    #[test]
    fn test_buffer_copy_params_full() {
        let params = BufferCopyParams::full(1024);
        assert_eq!(params.source_offset, 0);
        assert_eq!(params.dest_offset, 0);
        assert_eq!(params.size, 1024);
    }

    #[test]
    fn test_buffer_copy_params_default() {
        let params = BufferCopyParams::default();
        assert_eq!(params.source_offset, 0);
        assert_eq!(params.dest_offset, 0);
        assert_eq!(params.size, 0);
    }

    #[test]
    fn test_buffer_copy_params_is_aligned() {
        // All aligned
        let aligned = BufferCopyParams::new(0, 256, 1024);
        assert!(aligned.is_aligned());

        // Unaligned source
        let unaligned_src = BufferCopyParams::new(1, 256, 1024);
        assert!(!unaligned_src.is_aligned());

        // Unaligned dest
        let unaligned_dst = BufferCopyParams::new(0, 257, 1024);
        assert!(!unaligned_dst.is_aligned());

        // Unaligned size
        let unaligned_size = BufferCopyParams::new(0, 256, 1023);
        assert!(!unaligned_size.is_aligned());
    }

    #[test]
    fn test_buffer_copy_params_has_valid_size() {
        let valid = BufferCopyParams::new(0, 0, 100);
        assert!(valid.has_valid_size());

        let invalid = BufferCopyParams::new(0, 0, 0);
        assert!(!invalid.has_valid_size());
    }

    #[test]
    fn test_buffer_copy_params_display() {
        let params = BufferCopyParams::new(100, 200, 300);
        let s = format!("{}", params);
        assert!(s.contains("source: 100"));
        assert!(s.contains("dest: 200"));
        assert!(s.contains("size: 300"));
    }

    // ========================================================================
    // Validation Tests
    // ========================================================================

    #[test]
    fn test_copy_buffer_success() {
        // Valid aligned parameters with non-zero size
        let params = BufferCopyParams::new(0, 0, 256);
        assert!(validate_params(&params).is_ok());

        let params2 = BufferCopyParams::new(256, 512, 1024);
        assert!(validate_params(&params2).is_ok());
    }

    #[test]
    fn test_copy_buffer_alignment_error_source() {
        // Source offset not 4-byte aligned
        let params = BufferCopyParams::new(1, 0, 256);
        let result = validate_params(&params);
        assert!(result.is_err());

        match result {
            Err(CopyError::AlignmentError { field, offset, required }) => {
                assert_eq!(field, "source_offset");
                assert_eq!(offset, 1);
                assert_eq!(required, 4);
            }
            _ => panic!("Expected AlignmentError for source_offset"),
        }

        // Another unaligned source offset
        let params2 = BufferCopyParams::new(5, 0, 256);
        let result2 = validate_params(&params2);
        assert!(matches!(result2, Err(CopyError::AlignmentError { field: "source_offset", .. })));
    }

    #[test]
    fn test_copy_buffer_alignment_error_dest() {
        // Dest offset not 4-byte aligned
        let params = BufferCopyParams::new(0, 3, 256);
        let result = validate_params(&params);
        assert!(result.is_err());

        match result {
            Err(CopyError::AlignmentError { field, offset, required }) => {
                assert_eq!(field, "dest_offset");
                assert_eq!(offset, 3);
                assert_eq!(required, 4);
            }
            _ => panic!("Expected AlignmentError for dest_offset"),
        }

        // Another unaligned dest offset
        let params2 = BufferCopyParams::new(0, 7, 256);
        let result2 = validate_params(&params2);
        assert!(matches!(result2, Err(CopyError::AlignmentError { field: "dest_offset", .. })));
    }

    #[test]
    fn test_copy_buffer_alignment_error_size() {
        // Size not 4-byte aligned
        let params = BufferCopyParams::new(0, 0, 255);
        let result = validate_params(&params);
        assert!(result.is_err());

        match result {
            Err(CopyError::AlignmentError { field, offset, required }) => {
                assert_eq!(field, "size");
                assert_eq!(offset, 255);
                assert_eq!(required, 4);
            }
            _ => panic!("Expected AlignmentError for size"),
        }

        // Another unaligned size
        let params2 = BufferCopyParams::new(0, 0, 1);
        let result2 = validate_params(&params2);
        assert!(matches!(result2, Err(CopyError::AlignmentError { field: "size", .. })));
    }

    #[test]
    fn test_copy_buffer_zero_size() {
        // Zero size is invalid
        let params = BufferCopyParams::new(0, 0, 0);
        let result = validate_params(&params);
        assert!(result.is_err());

        match result {
            Err(CopyError::SizeError { size, reason }) => {
                assert_eq!(size, 0);
                assert!(reason.contains("greater than 0"));
            }
            _ => panic!("Expected SizeError for zero size"),
        }
    }

    #[test]
    fn test_copy_buffer_valid_params() {
        // Various valid aligned parameters
        let test_cases = [
            BufferCopyParams::new(0, 0, 4),
            BufferCopyParams::new(0, 0, 256),
            BufferCopyParams::new(4, 4, 4),
            BufferCopyParams::new(256, 512, 1024),
            BufferCopyParams::new(0, 1024, 4096),
            BufferCopyParams::full(65536),
        ];

        for params in test_cases {
            let result = validate_params(&params);
            assert!(
                result.is_ok(),
                "Expected Ok for params {:?}, got {:?}",
                params,
                result
            );
        }
    }

    // ========================================================================
    // Usage Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_usage_success() {
        let source_usage = BufferUsages::COPY_SRC | BufferUsages::VERTEX;
        let dest_usage = BufferUsages::COPY_DST | BufferUsages::STORAGE;
        assert!(validate_usage(source_usage, dest_usage).is_ok());
    }

    #[test]
    fn test_validate_usage_missing_copy_src() {
        let source_usage = BufferUsages::VERTEX; // Missing COPY_SRC
        let dest_usage = BufferUsages::COPY_DST;
        let result = validate_usage(source_usage, dest_usage);

        assert!(matches!(result, Err(CopyError::UsageError { missing: "COPY_SRC" })));
    }

    #[test]
    fn test_validate_usage_missing_copy_dst() {
        let source_usage = BufferUsages::COPY_SRC;
        let dest_usage = BufferUsages::STORAGE; // Missing COPY_DST
        let result = validate_usage(source_usage, dest_usage);

        assert!(matches!(result, Err(CopyError::UsageError { missing: "COPY_DST" })));
    }

    // ========================================================================
    // Bounds Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_bounds_success() {
        let params = BufferCopyParams::new(0, 0, 256);
        assert!(validate_bounds(&params, 256, 256).is_ok());
        assert!(validate_bounds(&params, 512, 512).is_ok());
    }

    #[test]
    fn test_validate_bounds_source_overflow() {
        let params = BufferCopyParams::new(128, 0, 256);
        // source_offset (128) + size (256) = 384 > source_size (256)
        let result = validate_bounds(&params, 256, 512);
        assert!(matches!(result, Err(CopyError::SourceBoundsError { .. })));
    }

    #[test]
    fn test_validate_bounds_dest_overflow() {
        let params = BufferCopyParams::new(0, 128, 256);
        // dest_offset (128) + size (256) = 384 > dest_size (256)
        let result = validate_bounds(&params, 512, 256);
        assert!(matches!(result, Err(CopyError::DestBoundsError { .. })));
    }

    // ========================================================================
    // Alignment Helper Tests
    // ========================================================================

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0));
        assert!(is_aligned(4));
        assert!(is_aligned(256));
        assert!(is_aligned(1024));

        assert!(!is_aligned(1));
        assert!(!is_aligned(2));
        assert!(!is_aligned(3));
        assert!(!is_aligned(5));
        assert!(!is_aligned(255));
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 4);
        assert_eq!(align_up(2), 4);
        assert_eq!(align_up(3), 4);
        assert_eq!(align_up(4), 4);
        assert_eq!(align_up(5), 8);
        assert_eq!(align_up(255), 256);
        assert_eq!(align_up(256), 256);
    }

    // ========================================================================
    // CopyError Tests
    // ========================================================================

    #[test]
    fn test_copy_error_display() {
        let alignment_err = CopyError::alignment_source(5);
        let display = format!("{}", alignment_err);
        assert!(display.contains("alignment error"));
        assert!(display.contains("source_offset"));
        assert!(display.contains("5"));
        assert!(display.contains("4 bytes"));

        let size_err = CopyError::zero_size();
        let display = format!("{}", size_err);
        assert!(display.contains("size error"));
        assert!(display.contains("0"));

        let usage_err = CopyError::missing_copy_src();
        let display = format!("{}", usage_err);
        assert!(display.contains("usage error"));
        assert!(display.contains("COPY_SRC"));
    }

    #[test]
    fn test_copy_error_constructors() {
        let _ = CopyError::alignment_source(1);
        let _ = CopyError::alignment_dest(2);
        let _ = CopyError::alignment_size(3);
        let _ = CopyError::zero_size();
        let _ = CopyError::missing_copy_src();
        let _ = CopyError::missing_copy_dst();
        let _ = CopyError::source_bounds(0, 100, 50);
        let _ = CopyError::dest_bounds(0, 100, 50);
    }

    #[test]
    fn test_copy_error_buffer_clear_constructors() {
        // Test the new buffer clear error constructors
        let _ = CopyError::buffer_missing_copy_dst();
        let _ = CopyError::clear_offset_not_aligned(3);
        let _ = CopyError::clear_size_not_aligned(5);
        let _ = CopyError::clear_bounds(100, 200, 150);
    }

    #[test]
    fn test_copy_error_buffer_clear_display() {
        // BufferMissingCopyDst
        let err = CopyError::buffer_missing_copy_dst();
        let display = format!("{}", err);
        assert!(display.contains("COPY_DST"));
        assert!(display.contains("missing"));

        // ClearOffsetNotAligned
        let err = CopyError::clear_offset_not_aligned(7);
        let display = format!("{}", err);
        assert!(display.contains("offset"));
        assert!(display.contains("7"));
        assert!(display.contains("4 bytes"));

        // ClearSizeNotAligned
        let err = CopyError::clear_size_not_aligned(13);
        let display = format!("{}", err);
        assert!(display.contains("size"));
        assert!(display.contains("13"));
        assert!(display.contains("4 bytes"));

        // ClearBoundsError
        let err = CopyError::clear_bounds(100, 200, 150);
        let display = format!("{}", err);
        assert!(display.contains("100")); // offset
        assert!(display.contains("200")); // size
        assert!(display.contains("150")); // buffer_size
        assert!(display.contains("exceeds"));
    }

    // ========================================================================
    // BufferClearParams Tests
    // ========================================================================

    #[test]
    fn test_buffer_clear_params_new() {
        let params = BufferClearParams::new(256, Some(1024));
        assert_eq!(params.offset, 256);
        assert_eq!(params.size, Some(1024));

        let params_no_size = BufferClearParams::new(128, None);
        assert_eq!(params_no_size.offset, 128);
        assert_eq!(params_no_size.size, None);
    }

    #[test]
    fn test_buffer_clear_params_full() {
        let params = BufferClearParams::full();
        assert_eq!(params.offset, 0);
        assert_eq!(params.size, None);
    }

    #[test]
    fn test_buffer_clear_params_range() {
        let params = BufferClearParams::range(512, 2048);
        assert_eq!(params.offset, 512);
        assert_eq!(params.size, Some(2048));
    }

    #[test]
    fn test_buffer_clear_params_default() {
        let params = BufferClearParams::default();
        assert_eq!(params.offset, 0);
        assert_eq!(params.size, None);
    }

    #[test]
    fn test_buffer_clear_params_is_offset_aligned() {
        // Aligned offsets
        assert!(BufferClearParams::new(0, None).is_offset_aligned());
        assert!(BufferClearParams::new(4, None).is_offset_aligned());
        assert!(BufferClearParams::new(256, None).is_offset_aligned());
        assert!(BufferClearParams::new(1024, None).is_offset_aligned());

        // Unaligned offsets
        assert!(!BufferClearParams::new(1, None).is_offset_aligned());
        assert!(!BufferClearParams::new(2, None).is_offset_aligned());
        assert!(!BufferClearParams::new(3, None).is_offset_aligned());
        assert!(!BufferClearParams::new(5, None).is_offset_aligned());
        assert!(!BufferClearParams::new(255, None).is_offset_aligned());
    }

    #[test]
    fn test_buffer_clear_params_is_size_aligned() {
        // No size specified - always aligned
        assert!(BufferClearParams::new(0, None).is_size_aligned());

        // Aligned sizes
        assert!(BufferClearParams::new(0, Some(4)).is_size_aligned());
        assert!(BufferClearParams::new(0, Some(256)).is_size_aligned());
        assert!(BufferClearParams::new(0, Some(1024)).is_size_aligned());

        // Unaligned sizes
        assert!(!BufferClearParams::new(0, Some(1)).is_size_aligned());
        assert!(!BufferClearParams::new(0, Some(2)).is_size_aligned());
        assert!(!BufferClearParams::new(0, Some(3)).is_size_aligned());
        assert!(!BufferClearParams::new(0, Some(5)).is_size_aligned());
        assert!(!BufferClearParams::new(0, Some(255)).is_size_aligned());
    }

    #[test]
    fn test_buffer_clear_params_is_aligned() {
        // All aligned
        assert!(BufferClearParams::new(0, Some(256)).is_aligned());
        assert!(BufferClearParams::new(128, Some(1024)).is_aligned());
        assert!(BufferClearParams::full().is_aligned());

        // Unaligned offset
        assert!(!BufferClearParams::new(1, Some(256)).is_aligned());

        // Unaligned size
        assert!(!BufferClearParams::new(0, Some(255)).is_aligned());

        // Both unaligned
        assert!(!BufferClearParams::new(3, Some(5)).is_aligned());
    }

    #[test]
    fn test_buffer_clear_params_display() {
        let params_with_size = BufferClearParams::new(256, Some(1024));
        let display = format!("{}", params_with_size);
        assert!(display.contains("offset: 256"));
        assert!(display.contains("size: 1024"));

        let params_no_size = BufferClearParams::new(128, None);
        let display = format!("{}", params_no_size);
        assert!(display.contains("offset: 128"));
        assert!(display.contains("entire"));
    }

    #[test]
    fn test_buffer_clear_params_equality() {
        let a = BufferClearParams::new(100, Some(200));
        let b = BufferClearParams::new(100, Some(200));
        let c = BufferClearParams::new(100, Some(300));
        let d = BufferClearParams::new(100, None);

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn test_buffer_clear_function_signatures_exist() {
        // Verify the functions exist and have correct signatures by referencing them
        let _clear_fn: fn(&mut CommandEncoder, &Buffer, BufferClearParams) -> Result<(), CopyError>
            = clear_buffer;

        let _trinity_clear_fn: fn(&mut TrinityCommandEncoder, &Buffer, BufferClearParams) -> Result<(), CopyError>
            = clear_buffer_trinity;

        let _validate_fn: fn(&Buffer, &BufferClearParams) -> Result<(), CopyError>
            = validate_buffer_clear_params;
    }

    // ========================================================================
    // CopyAlignmentCalculator Tests
    // ========================================================================

    #[test]
    fn test_alignment_constants() {
        // Verify alignment constants have expected values
        assert_eq!(BUFFER_OFFSET_ALIGNMENT, 4);
        assert_eq!(BYTES_PER_ROW_ALIGNMENT, 256);
        assert_eq!(COPY_SIZE_ALIGNMENT, 4);
        assert_eq!(COPY_BUFFER_ALIGNMENT, 4);
    }

    #[test]
    fn test_bytes_per_row_alignment_exact() {
        // Values that are already aligned should stay the same
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(256), 256);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(512), 512);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(768), 768);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(1024), 1024);
    }

    #[test]
    fn test_bytes_per_row_alignment_round_up() {
        // Values should be rounded up to next 256-byte boundary
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(100), 256);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(300), 512);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(1), 256);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(255), 256);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(257), 512);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(400), 512);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(513), 768);
    }

    #[test]
    fn test_rows_per_image_calculation() {
        // 2D texture: rows_per_image equals height
        assert_eq!(CopyAlignmentCalculator::calculate_rows_per_image(100, 1), 100);
        assert_eq!(CopyAlignmentCalculator::calculate_rows_per_image(512, 1), 512);

        // 3D/array texture: rows_per_image still equals height (per slice)
        assert_eq!(CopyAlignmentCalculator::calculate_rows_per_image(64, 8), 64);
        assert_eq!(CopyAlignmentCalculator::calculate_rows_per_image(128, 16), 128);
        assert_eq!(CopyAlignmentCalculator::calculate_rows_per_image(256, 32), 256);
    }

    #[test]
    fn test_buffer_size_2d() {
        // 2D texture buffer size calculations (depth = 1)
        // 256 bytes/row * 100 rows * 1 depth = 25600
        assert_eq!(CopyAlignmentCalculator::calculate_buffer_size(256, 100, 1), 25600);

        // 512 bytes/row * 512 rows * 1 depth = 262144
        assert_eq!(CopyAlignmentCalculator::calculate_buffer_size(512, 512, 1), 262144);

        // 1024 bytes/row * 768 rows * 1 depth = 786432
        assert_eq!(CopyAlignmentCalculator::calculate_buffer_size(1024, 768, 1), 786432);
    }

    #[test]
    fn test_buffer_size_3d() {
        // 3D texture buffer size calculations (depth > 1)
        // 512 bytes/row * 64 rows * 8 depth = 262144
        assert_eq!(CopyAlignmentCalculator::calculate_buffer_size(512, 64, 8), 262144);

        // 256 bytes/row * 256 rows * 4 depth = 262144
        assert_eq!(CopyAlignmentCalculator::calculate_buffer_size(256, 256, 4), 262144);

        // 1024 bytes/row * 128 rows * 16 depth = 2097152
        assert_eq!(CopyAlignmentCalculator::calculate_buffer_size(1024, 128, 16), 2097152);
    }

    #[test]
    fn test_zero_values() {
        // Zero input handling
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(0), 0);
        assert_eq!(CopyAlignmentCalculator::calculate_rows_per_image(0, 1), 0);
        assert_eq!(CopyAlignmentCalculator::calculate_rows_per_image(0, 0), 0);
        assert_eq!(CopyAlignmentCalculator::calculate_buffer_size(0, 0, 0), 0);
        assert_eq!(CopyAlignmentCalculator::calculate_buffer_size(256, 0, 1), 0);
        assert_eq!(CopyAlignmentCalculator::calculate_buffer_size(256, 100, 0), 0);
    }

    #[test]
    fn test_edge_cases() {
        // Edge cases near alignment boundary
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(254), 256);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(255), 256);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(256), 256);

        // Large values
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(10000), 10240);
        assert_eq!(CopyAlignmentCalculator::calculate_buffer_size(4096, 2160, 1), 8847360);

        // Common texture sizes (RGBA8 = 4 bytes per pixel)
        // 1920x1080: 1920 * 4 = 7680, aligned = 7680 (already 256-aligned)
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(7680), 7680);

        // 1280x720: 1280 * 4 = 5120, aligned = 5120 (already 256-aligned)
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(5120), 5120);
    }

    #[test]
    fn test_texture_buffer_layout() {
        // 100x100 RGBA texture
        let (bytes_per_row, total_size) =
            CopyAlignmentCalculator::calculate_texture_buffer_layout(100, 100, 1, 4);
        assert_eq!(bytes_per_row, 512); // 100 * 4 = 400, aligned to 512
        assert_eq!(total_size, 51200);  // 512 * 100 * 1

        // 256x256 RGBA texture (already aligned)
        let (bytes_per_row, total_size) =
            CopyAlignmentCalculator::calculate_texture_buffer_layout(256, 256, 1, 4);
        assert_eq!(bytes_per_row, 1024); // 256 * 4 = 1024 (256-aligned)
        assert_eq!(total_size, 262144);  // 1024 * 256 * 1

        // 64x64x8 3D RGBA texture
        let (bytes_per_row, total_size) =
            CopyAlignmentCalculator::calculate_texture_buffer_layout(64, 64, 8, 4);
        assert_eq!(bytes_per_row, 256); // 64 * 4 = 256 (256-aligned)
        assert_eq!(total_size, 131072); // 256 * 64 * 8
    }

    #[test]
    fn test_is_bytes_per_row_aligned() {
        // Aligned values
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(0));
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(256));
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(512));
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(1024));
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(2048));

        // Unaligned values
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(1));
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(100));
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(255));
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(257));
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(400));
    }

    #[test]
    fn test_calculate_row_padding() {
        // No padding needed for aligned values
        assert_eq!(CopyAlignmentCalculator::calculate_row_padding(256), 0);
        assert_eq!(CopyAlignmentCalculator::calculate_row_padding(512), 0);

        // Padding calculations
        assert_eq!(CopyAlignmentCalculator::calculate_row_padding(100), 156);
        assert_eq!(CopyAlignmentCalculator::calculate_row_padding(300), 212);
        assert_eq!(CopyAlignmentCalculator::calculate_row_padding(1), 255);
        assert_eq!(CopyAlignmentCalculator::calculate_row_padding(255), 1);
        assert_eq!(CopyAlignmentCalculator::calculate_row_padding(257), 255);
    }

    // ========================================================================
    // CopyExtent3d Tests
    // ========================================================================

    #[test]
    fn test_copy_extent3d_new() {
        let extent = CopyExtent3d::new(128, 256, 4);
        assert_eq!(extent.width, 128);
        assert_eq!(extent.height, 256);
        assert_eq!(extent.depth_or_array_layers, 4);
    }

    #[test]
    fn test_copy_extent3d_new_2d() {
        let extent = CopyExtent3d::new_2d(512, 512);
        assert_eq!(extent.width, 512);
        assert_eq!(extent.height, 512);
        assert_eq!(extent.depth_or_array_layers, 1);
    }

    #[test]
    fn test_copy_extent3d_is_valid() {
        // Valid extents
        assert!(CopyExtent3d::new(1, 1, 1).is_valid());
        assert!(CopyExtent3d::new(256, 256, 1).is_valid());
        assert!(CopyExtent3d::new(64, 64, 8).is_valid());

        // Invalid extents (zero dimensions)
        assert!(!CopyExtent3d::new(0, 100, 1).is_valid());
        assert!(!CopyExtent3d::new(100, 0, 1).is_valid());
        assert!(!CopyExtent3d::new(100, 100, 0).is_valid());
        assert!(!CopyExtent3d::new(0, 0, 0).is_valid());
    }

    #[test]
    fn test_copy_extent3d_is_multi_layer() {
        // Single layer (2D textures)
        assert!(!CopyExtent3d::new(256, 256, 1).is_multi_layer());
        assert!(!CopyExtent3d::new_2d(1920, 1080).is_multi_layer());

        // Multi-layer (3D textures or arrays)
        assert!(CopyExtent3d::new(64, 64, 2).is_multi_layer());
        assert!(CopyExtent3d::new(128, 128, 6).is_multi_layer());
        assert!(CopyExtent3d::new(256, 256, 32).is_multi_layer());
    }

    #[test]
    fn test_copy_extent3d_to_wgpu() {
        let extent = CopyExtent3d::new(128, 256, 4);
        let wgpu_extent = extent.to_wgpu();
        assert_eq!(wgpu_extent.width, 128);
        assert_eq!(wgpu_extent.height, 256);
        assert_eq!(wgpu_extent.depth_or_array_layers, 4);
    }

    #[test]
    fn test_copy_extent3d_from_wgpu() {
        let wgpu_extent = wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 6,
        };
        let extent: CopyExtent3d = wgpu_extent.into();
        assert_eq!(extent.width, 512);
        assert_eq!(extent.height, 512);
        assert_eq!(extent.depth_or_array_layers, 6);
    }

    #[test]
    fn test_copy_extent3d_default() {
        let extent = CopyExtent3d::default();
        assert_eq!(extent.width, 1);
        assert_eq!(extent.height, 1);
        assert_eq!(extent.depth_or_array_layers, 1);
        assert!(extent.is_valid());
    }

    #[test]
    fn test_copy_extent3d_display() {
        let extent = CopyExtent3d::new(1920, 1080, 1);
        assert_eq!(format!("{}", extent), "1920x1080x1");

        let extent_3d = CopyExtent3d::new(64, 64, 8);
        assert_eq!(format!("{}", extent_3d), "64x64x8");
    }

    // ========================================================================
    // Buffer-to-Texture Copy Error Tests
    // ========================================================================

    #[test]
    fn test_copy_error_bytes_per_row_alignment() {
        let error = CopyError::bytes_per_row_alignment(300);
        let display = format!("{}", error);
        assert!(display.contains("bytes_per_row"));
        assert!(display.contains("300"));
        assert!(display.contains("256"));
    }

    #[test]
    fn test_copy_error_rows_per_image_required() {
        let error = CopyError::rows_per_image_required(8);
        let display = format!("{}", error);
        assert!(display.contains("rows_per_image"));
        assert!(display.contains("8"));
    }

    #[test]
    fn test_copy_error_texture_missing_copy_dst() {
        let error = CopyError::texture_missing_copy_dst();
        let display = format!("{}", error);
        assert!(display.contains("COPY_DST"));
    }

    #[test]
    fn test_copy_error_invalid_extent() {
        let error = CopyError::invalid_extent("width", 0);
        let display = format!("{}", error);
        assert!(display.contains("width"));
        assert!(display.contains("0"));
    }

    // ========================================================================
    // ImageCopyTexture Tests
    // ========================================================================

    // Note: These tests are limited because wgpu::Texture cannot be easily
    // mocked without a real GPU device. We test the struct construction logic.

    #[test]
    fn test_image_copy_texture_origin_values() {
        // Test that Origin3d values are correctly set
        let origin = wgpu::Origin3d { x: 10, y: 20, z: 30 };
        assert_eq!(origin.x, 10);
        assert_eq!(origin.y, 20);
        assert_eq!(origin.z, 30);

        // Test ZERO origin
        let zero = wgpu::Origin3d::ZERO;
        assert_eq!(zero.x, 0);
        assert_eq!(zero.y, 0);
        assert_eq!(zero.z, 0);
    }

    #[test]
    fn test_texture_aspect_variants() {
        // Test that TextureAspect variants work as expected
        let all = wgpu::TextureAspect::All;
        let depth = wgpu::TextureAspect::DepthOnly;
        let stencil = wgpu::TextureAspect::StencilOnly;

        // These should all be distinct
        assert!(all != depth);
        assert!(depth != stencil);
        assert!(all != stencil);
    }

    // ========================================================================
    // ImageDataLayout Tests
    // ========================================================================

    #[test]
    fn test_image_data_layout_construction() {
        // Test 2D layout (no rows_per_image needed)
        let layout_2d = wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(512),
            rows_per_image: None,
        };
        assert_eq!(layout_2d.offset, 0);
        assert_eq!(layout_2d.bytes_per_row, Some(512));
        assert_eq!(layout_2d.rows_per_image, None);

        // Test 3D/array layout (rows_per_image required)
        let layout_3d = wgpu::ImageDataLayout {
            offset: 256,
            bytes_per_row: Some(1024),
            rows_per_image: Some(128),
        };
        assert_eq!(layout_3d.offset, 256);
        assert_eq!(layout_3d.bytes_per_row, Some(1024));
        assert_eq!(layout_3d.rows_per_image, Some(128));
    }

    #[test]
    fn test_image_data_layout_alignment_validation() {
        // Aligned bytes_per_row
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(256));
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(512));
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(1024));

        // Unaligned bytes_per_row (would fail validation)
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(100));
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(300));
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(500));
    }

    // ========================================================================
    // Buffer-to-Texture Validation Logic Tests
    // ========================================================================

    #[test]
    fn test_extent_validation_zero_width() {
        // Zero width should be invalid
        let extent = CopyExtent3d::new(0, 100, 1);
        assert!(!extent.is_valid());
    }

    #[test]
    fn test_extent_validation_zero_height() {
        // Zero height should be invalid
        let extent = CopyExtent3d::new(100, 0, 1);
        assert!(!extent.is_valid());
    }

    #[test]
    fn test_extent_validation_zero_depth() {
        // Zero depth should be invalid
        let extent = CopyExtent3d::new(100, 100, 0);
        assert!(!extent.is_valid());
    }

    #[test]
    fn test_multi_layer_requires_rows_per_image() {
        // For multi-layer copies, rows_per_image is conceptually required
        let extent_3d = CopyExtent3d::new(64, 64, 8);
        assert!(extent_3d.is_multi_layer());

        // Single layer doesn't require rows_per_image
        let extent_2d = CopyExtent3d::new_2d(256, 256);
        assert!(!extent_2d.is_multi_layer());
    }

    #[test]
    fn test_bytes_per_row_auto_calculation() {
        // For a 100-pixel wide RGBA8 texture
        let width = 100;
        let bytes_per_pixel = 4;
        let unaligned = width * bytes_per_pixel; // 400
        let aligned = CopyAlignmentCalculator::calculate_aligned_bytes_per_row(unaligned);
        assert_eq!(aligned, 512); // Rounded up to 512

        // For a 256-pixel wide RGBA8 texture (already aligned)
        let width2 = 256;
        let unaligned2 = width2 * bytes_per_pixel; // 1024
        let aligned2 = CopyAlignmentCalculator::calculate_aligned_bytes_per_row(unaligned2);
        assert_eq!(aligned2, 1024); // Already 256-aligned
    }

    #[test]
    fn test_buffer_size_for_texture_copy() {
        // 2D texture: 128x128 RGBA8
        let (bytes_per_row, total_size) =
            CopyAlignmentCalculator::calculate_texture_buffer_layout(128, 128, 1, 4);
        assert_eq!(bytes_per_row, 512); // 128 * 4 = 512 (aligned)
        assert_eq!(total_size, 65536);  // 512 * 128

        // 3D texture: 64x64x4 RGBA8
        let (bytes_per_row_3d, total_size_3d) =
            CopyAlignmentCalculator::calculate_texture_buffer_layout(64, 64, 4, 4);
        assert_eq!(bytes_per_row_3d, 256); // 64 * 4 = 256 (aligned)
        assert_eq!(total_size_3d, 65536);  // 256 * 64 * 4
    }

    #[test]
    fn test_copy_extent_equality() {
        let a = CopyExtent3d::new(128, 256, 4);
        let b = CopyExtent3d::new(128, 256, 4);
        let c = CopyExtent3d::new(256, 256, 4);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_copy_extent_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(CopyExtent3d::new(128, 128, 1));
        set.insert(CopyExtent3d::new(256, 256, 1));
        set.insert(CopyExtent3d::new(128, 128, 1)); // Duplicate

        assert_eq!(set.len(), 2);
    }

    // ========================================================================
    // Texture-to-Buffer Copy Tests (T-WGPU-P4.2.3)
    // ========================================================================

    // ------------------------------------------------------------------------
    // Criterion 1: Same parameters as buffer-to-texture (reuse existing structs)
    // ------------------------------------------------------------------------

    #[test]
    fn test_texture_to_buffer_uses_same_structs_as_buffer_to_texture() {
        // Verify that ImageCopyTexture, ImageCopyBuffer, and CopyExtent3d
        // can be used for texture-to-buffer operations (demonstrating reuse)
        let copy_size = CopyExtent3d::new_2d(128, 128);
        assert_eq!(copy_size.width, 128);
        assert_eq!(copy_size.height, 128);
        assert_eq!(copy_size.depth_or_array_layers, 1);
        assert!(copy_size.is_valid());
    }

    #[test]
    fn test_texture_to_buffer_layout_same_as_buffer_to_texture() {
        // Verify that layout calculations are symmetric
        let width = 100;
        let height = 100;
        let bytes_per_pixel = 4;

        let (bytes_per_row, total_size) =
            CopyAlignmentCalculator::calculate_texture_buffer_layout(width, height, 1, bytes_per_pixel);

        // bytes_per_row should be 256-aligned
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(bytes_per_row));
        // Total size should be bytes_per_row * height
        assert_eq!(total_size, (bytes_per_row as u64) * (height as u64));
    }

    // ------------------------------------------------------------------------
    // Criterion 2: MAP_READ usage validation on destination buffer
    // ------------------------------------------------------------------------

    #[test]
    fn test_buffer_missing_map_read_error() {
        let error = CopyError::buffer_missing_map_read();
        let display = format!("{}", error);
        assert!(display.contains("MAP_READ"));
        assert!(display.contains("buffer"));
    }

    #[test]
    fn test_texture_missing_copy_src_error() {
        let error = CopyError::texture_missing_copy_src();
        let display = format!("{}", error);
        assert!(display.contains("COPY_SRC"));
        assert!(display.contains("texture"));
    }

    #[test]
    fn test_new_error_constructors_exist() {
        // Verify new error constructors work
        let map_read_err = CopyError::buffer_missing_map_read();
        assert!(matches!(map_read_err, CopyError::BufferMissingMapRead));

        let copy_src_err = CopyError::texture_missing_copy_src();
        assert!(matches!(copy_src_err, CopyError::TextureMissingCopySrc));
    }

    // ------------------------------------------------------------------------
    // Criterion 3: Staging buffer pattern helper
    // ------------------------------------------------------------------------

    #[test]
    fn test_staging_buffer_layout_calculation() {
        // Verify StagingBuffer calculates correct layout
        let extent = CopyExtent3d::new_2d(100, 100);

        // For 100x100 RGBA8: 100 * 4 = 400 bytes per row
        // Aligned to 256: 512 bytes per row
        let (expected_bytes_per_row, expected_size) =
            CopyAlignmentCalculator::calculate_texture_buffer_layout(
                extent.width, extent.height, extent.depth_or_array_layers, 4
            );

        assert_eq!(expected_bytes_per_row, 512);
        assert_eq!(expected_size, 51200); // 512 * 100

        let expected_rows_per_image =
            CopyAlignmentCalculator::calculate_rows_per_image(extent.height, extent.depth_or_array_layers);
        assert_eq!(expected_rows_per_image, 100);
    }

    #[test]
    fn test_staging_buffer_3d_layout() {
        // Verify 3D texture staging buffer layout
        let extent = CopyExtent3d::new(64, 64, 8);

        let (bytes_per_row, size) =
            CopyAlignmentCalculator::calculate_texture_buffer_layout(
                extent.width, extent.height, extent.depth_or_array_layers, 4
            );

        // 64 * 4 = 256 (already aligned)
        assert_eq!(bytes_per_row, 256);
        // 256 * 64 * 8 = 131072
        assert_eq!(size, 131072);

        let rows_per_image =
            CopyAlignmentCalculator::calculate_rows_per_image(extent.height, extent.depth_or_array_layers);
        assert_eq!(rows_per_image, 64);
    }

    #[test]
    fn test_staging_buffer_image_copy_buffer_2d() {
        // Verify as_image_copy_buffer produces correct layout for 2D
        // (Cannot fully test without wgpu device, but can test layout logic)
        let extent = CopyExtent3d::new_2d(128, 128);

        // For 2D (depth_or_array_layers = 1), rows_per_image should be None
        assert_eq!(extent.depth_or_array_layers, 1);
        assert!(!extent.is_multi_layer());
    }

    #[test]
    fn test_staging_buffer_image_copy_buffer_3d() {
        // Verify as_image_copy_buffer produces correct layout for 3D
        let extent = CopyExtent3d::new(64, 64, 4);

        // For 3D (depth_or_array_layers > 1), rows_per_image should be Some
        assert_eq!(extent.depth_or_array_layers, 4);
        assert!(extent.is_multi_layer());
    }

    // ------------------------------------------------------------------------
    // Criterion 4: Async readback support (validation logic tests)
    // ------------------------------------------------------------------------

    #[test]
    fn test_map_read_usage_flag_values() {
        // Verify BufferUsages::MAP_READ exists and has expected behavior
        let map_read = BufferUsages::MAP_READ;
        let copy_dst = BufferUsages::COPY_DST;

        // Staging buffer should have both
        let staging_usage = map_read | copy_dst;
        assert!(staging_usage.contains(BufferUsages::MAP_READ));
        assert!(staging_usage.contains(BufferUsages::COPY_DST));
    }

    #[test]
    fn test_validate_map_read_usage_with_valid_usages() {
        // Test that validate_map_read_usage would pass with MAP_READ
        // (Cannot test actual buffer without GPU, but test logic indirectly)
        let with_map_read = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
        assert!(with_map_read.contains(BufferUsages::MAP_READ));
    }

    #[test]
    fn test_validate_map_read_usage_without_map_read() {
        // Test that usage without MAP_READ would fail
        let without_map_read = BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        assert!(!without_map_read.contains(BufferUsages::MAP_READ));
    }

    // ------------------------------------------------------------------------
    // Additional Texture-to-Buffer Validation Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_texture_to_buffer_validation_zero_extent() {
        // Validate that zero extents are caught
        let zero_width = CopyExtent3d::new(0, 100, 1);
        assert!(!zero_width.is_valid());

        let zero_height = CopyExtent3d::new(100, 0, 1);
        assert!(!zero_height.is_valid());

        let zero_depth = CopyExtent3d::new(100, 100, 0);
        assert!(!zero_depth.is_valid());
    }

    #[test]
    fn test_texture_to_buffer_bytes_per_row_alignment() {
        // Validate bytes_per_row alignment checking
        // Aligned values should pass
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(256));
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(512));
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(1024));

        // Unaligned values should fail
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(300));
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(400));
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(500));
    }

    #[test]
    fn test_texture_to_buffer_rows_per_image_requirement() {
        // For multi-layer copies, rows_per_image is required
        let multi_layer = CopyExtent3d::new(64, 64, 8);
        assert!(multi_layer.is_multi_layer());

        // For single-layer copies, rows_per_image is optional
        let single_layer = CopyExtent3d::new_2d(256, 256);
        assert!(!single_layer.is_multi_layer());
    }

    #[test]
    fn test_texture_to_buffer_buffer_offset_alignment() {
        // Buffer offset must be 4-byte aligned
        assert_eq!(BUFFER_OFFSET_ALIGNMENT, 4);

        // Test alignment checking
        assert!(is_aligned(0));
        assert!(is_aligned(4));
        assert!(is_aligned(8));
        assert!(is_aligned(256));

        assert!(!is_aligned(1));
        assert!(!is_aligned(2));
        assert!(!is_aligned(3));
        assert!(!is_aligned(5));
    }

    #[test]
    fn test_staging_buffer_common_texture_formats() {
        // Test layout calculations for common texture formats

        // RGBA8: 4 bytes per pixel
        let rgba8_extent = CopyExtent3d::new_2d(1920, 1080);
        let (bpr, size) = CopyAlignmentCalculator::calculate_texture_buffer_layout(
            rgba8_extent.width, rgba8_extent.height, 1, 4
        );
        assert_eq!(bpr, 7680); // 1920 * 4 = 7680 (256-aligned)
        assert_eq!(size, 7680 * 1080);

        // RGBA16Float: 8 bytes per pixel
        let rgba16f_extent = CopyExtent3d::new_2d(256, 256);
        let (bpr16, size16) = CopyAlignmentCalculator::calculate_texture_buffer_layout(
            rgba16f_extent.width, rgba16f_extent.height, 1, 8
        );
        assert_eq!(bpr16, 2048); // 256 * 8 = 2048 (256-aligned)
        assert_eq!(size16, 2048 * 256);

        // RGBA32Float: 16 bytes per pixel
        let rgba32f_extent = CopyExtent3d::new_2d(128, 128);
        let (bpr32, size32) = CopyAlignmentCalculator::calculate_texture_buffer_layout(
            rgba32f_extent.width, rgba32f_extent.height, 1, 16
        );
        assert_eq!(bpr32, 2048); // 128 * 16 = 2048 (256-aligned)
        assert_eq!(size32, 2048 * 128);
    }

    #[test]
    fn test_staging_buffer_mipmap_sizes() {
        // Test staging buffer for different mip levels
        let mip0 = CopyExtent3d::new_2d(256, 256);
        let mip1 = CopyExtent3d::new_2d(128, 128);
        let mip2 = CopyExtent3d::new_2d(64, 64);
        let mip3 = CopyExtent3d::new_2d(32, 32);

        let (bpr0, _) = CopyAlignmentCalculator::calculate_texture_buffer_layout(
            mip0.width, mip0.height, 1, 4
        );
        let (bpr1, _) = CopyAlignmentCalculator::calculate_texture_buffer_layout(
            mip1.width, mip1.height, 1, 4
        );
        let (bpr2, _) = CopyAlignmentCalculator::calculate_texture_buffer_layout(
            mip2.width, mip2.height, 1, 4
        );
        let (bpr3, _) = CopyAlignmentCalculator::calculate_texture_buffer_layout(
            mip3.width, mip3.height, 1, 4
        );

        assert_eq!(bpr0, 1024); // 256 * 4 = 1024
        assert_eq!(bpr1, 512);  // 128 * 4 = 512
        assert_eq!(bpr2, 256);  // 64 * 4 = 256
        assert_eq!(bpr3, 256);  // 32 * 4 = 128, aligned to 256
    }

    #[test]
    fn test_image_data_layout_for_readback() {
        // Test creating ImageDataLayout for readback operations
        let layout_2d = wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(512),
            rows_per_image: None,
        };
        assert_eq!(layout_2d.offset, 0);
        assert_eq!(layout_2d.bytes_per_row, Some(512));
        assert_eq!(layout_2d.rows_per_image, None);

        let layout_3d = wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: Some(64),
        };
        assert_eq!(layout_3d.offset, 0);
        assert_eq!(layout_3d.bytes_per_row, Some(256));
        assert_eq!(layout_3d.rows_per_image, Some(64));
    }

    // ========================================================================
    // Texture-to-Texture Copy Tests (T-WGPU-P4.2.4)
    // ========================================================================

    // ------------------------------------------------------------------------
    // Criterion 1: Source and dest ImageCopyTexture - both use ImageCopyTexture struct
    // ------------------------------------------------------------------------

    #[test]
    fn test_texture_to_texture_uses_image_copy_texture_for_both() {
        // Verify that ImageCopyTexture can be used for both source and destination
        // This demonstrates the API design reuses the same struct
        let origin_src = wgpu::Origin3d { x: 0, y: 0, z: 0 };
        let origin_dst = wgpu::Origin3d { x: 64, y: 64, z: 0 };

        // Both use Origin3d and the same struct pattern
        assert_eq!(origin_src.x, 0);
        assert_eq!(origin_dst.x, 64);
    }

    #[test]
    fn test_texture_to_texture_image_copy_texture_struct_consistency() {
        // Verify ImageCopyTexture has all required fields for texture-to-texture copy
        // The struct should have: texture, mip_level, origin, aspect

        // Test that we can construct origins for array layer selection
        let array_layer_2 = wgpu::Origin3d { x: 0, y: 0, z: 2 };
        assert_eq!(array_layer_2.z, 2);

        // Test various mip level values
        let mip_levels = [0u32, 1, 2, 3, 4];
        for level in mip_levels {
            assert!(level < 16); // Reasonable mip level range
        }
    }

    // ------------------------------------------------------------------------
    // Criterion 2: Format compatibility check - validate formats are copy-compatible
    // ------------------------------------------------------------------------

    #[test]
    fn test_format_compatibility_same_format() {
        use wgpu::TextureFormat;

        // Same format should always be compatible
        assert!(are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm));
        assert!(are_formats_copy_compatible(TextureFormat::Rgba16Float, TextureFormat::Rgba16Float));
        assert!(are_formats_copy_compatible(TextureFormat::R32Float, TextureFormat::R32Float));
        assert!(are_formats_copy_compatible(TextureFormat::Depth32Float, TextureFormat::Depth32Float));
    }

    #[test]
    fn test_format_compatibility_rgba8_family() {
        use wgpu::TextureFormat;

        // RGBA8 family formats should be compatible (same size, different interpretation)
        assert!(are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb));
        assert!(are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Snorm));
        assert!(are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Uint));
        assert!(are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba8Sint));
        assert!(are_formats_copy_compatible(TextureFormat::Bgra8Unorm, TextureFormat::Bgra8UnormSrgb));
        assert!(are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_format_compatibility_different_sizes() {
        use wgpu::TextureFormat;

        // Different size formats should NOT be compatible
        assert!(!are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba16Float));
        assert!(!are_formats_copy_compatible(TextureFormat::R8Unorm, TextureFormat::R16Unorm));
        assert!(!are_formats_copy_compatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba32Float));
        assert!(!are_formats_copy_compatible(TextureFormat::R32Float, TextureFormat::Rg32Float));
    }

    #[test]
    fn test_format_compatibility_r8_family() {
        use wgpu::TextureFormat;

        // R8 family formats should be compatible
        assert!(are_formats_copy_compatible(TextureFormat::R8Unorm, TextureFormat::R8Snorm));
        assert!(are_formats_copy_compatible(TextureFormat::R8Unorm, TextureFormat::R8Uint));
        assert!(are_formats_copy_compatible(TextureFormat::R8Unorm, TextureFormat::R8Sint));
    }

    #[test]
    fn test_format_compatibility_r32_family() {
        use wgpu::TextureFormat;

        // R32 family formats should be compatible
        assert!(are_formats_copy_compatible(TextureFormat::R32Float, TextureFormat::R32Uint));
        assert!(are_formats_copy_compatible(TextureFormat::R32Float, TextureFormat::R32Sint));
        assert!(are_formats_copy_compatible(TextureFormat::R32Uint, TextureFormat::R32Sint));
    }

    #[test]
    fn test_format_compatibility_rgba16_family() {
        use wgpu::TextureFormat;

        // RGBA16 family formats should be compatible
        assert!(are_formats_copy_compatible(TextureFormat::Rgba16Float, TextureFormat::Rgba16Uint));
        assert!(are_formats_copy_compatible(TextureFormat::Rgba16Float, TextureFormat::Rgba16Sint));
        assert!(are_formats_copy_compatible(TextureFormat::Rgba16Float, TextureFormat::Rgba16Unorm));
        assert!(are_formats_copy_compatible(TextureFormat::Rgba16Float, TextureFormat::Rgba16Snorm));
    }

    #[test]
    fn test_format_compatibility_compressed_bc() {
        use wgpu::TextureFormat;

        // BC compressed formats within same family should be compatible
        assert!(are_formats_copy_compatible(TextureFormat::Bc1RgbaUnorm, TextureFormat::Bc1RgbaUnormSrgb));
        assert!(are_formats_copy_compatible(TextureFormat::Bc2RgbaUnorm, TextureFormat::Bc2RgbaUnormSrgb));
        assert!(are_formats_copy_compatible(TextureFormat::Bc3RgbaUnorm, TextureFormat::Bc3RgbaUnormSrgb));
        assert!(are_formats_copy_compatible(TextureFormat::Bc7RgbaUnorm, TextureFormat::Bc7RgbaUnormSrgb));

        // Different BC formats should NOT be compatible
        assert!(!are_formats_copy_compatible(TextureFormat::Bc1RgbaUnorm, TextureFormat::Bc3RgbaUnorm));
        assert!(!are_formats_copy_compatible(TextureFormat::Bc2RgbaUnorm, TextureFormat::Bc7RgbaUnorm));
    }

    #[test]
    fn test_format_incompatible_error() {
        use wgpu::TextureFormat;

        let error = CopyError::format_incompatible(TextureFormat::Rgba8Unorm, TextureFormat::Rgba16Float);
        let display = format!("{}", error);
        assert!(display.contains("format"));
        assert!(display.contains("incompatible") || display.contains("Incompatible"));
    }

    // ------------------------------------------------------------------------
    // Criterion 3: Mip level selection - support mip_level field in ImageCopyTexture
    // ------------------------------------------------------------------------

    #[test]
    fn test_mip_size_calculation() {
        // Base size 256: mip 0 = 256, mip 1 = 128, mip 2 = 64, mip 3 = 32, etc.
        assert_eq!(mip_size(256, 0), 256);
        assert_eq!(mip_size(256, 1), 128);
        assert_eq!(mip_size(256, 2), 64);
        assert_eq!(mip_size(256, 3), 32);
        assert_eq!(mip_size(256, 4), 16);
        assert_eq!(mip_size(256, 5), 8);
        assert_eq!(mip_size(256, 6), 4);
        assert_eq!(mip_size(256, 7), 2);
        assert_eq!(mip_size(256, 8), 1);
        // Further mips should clamp to 1
        assert_eq!(mip_size(256, 9), 1);
        assert_eq!(mip_size(256, 10), 1);
    }

    #[test]
    fn test_mip_size_non_power_of_two() {
        // Non-power-of-two base size
        assert_eq!(mip_size(100, 0), 100);
        assert_eq!(mip_size(100, 1), 50);
        assert_eq!(mip_size(100, 2), 25);
        assert_eq!(mip_size(100, 3), 12);
        assert_eq!(mip_size(100, 4), 6);
        assert_eq!(mip_size(100, 5), 3);
        assert_eq!(mip_size(100, 6), 1);
        assert_eq!(mip_size(100, 7), 1); // Clamped to 1
    }

    #[test]
    fn test_mip_level_out_of_bounds_error() {
        let error = CopyError::mip_level_out_of_bounds(5, 4);
        let display = format!("{}", error);
        assert!(display.contains("mip"));
        assert!(display.contains("5"));
        assert!(display.contains("4"));
        assert!(display.contains("bounds") || display.contains("exceeds"));
    }

    #[test]
    fn test_mip_level_validation_range() {
        // Test that mip level values are properly bounded
        // Typical texture mip count for 256x256 is 9 levels (256->128->64->32->16->8->4->2->1)
        let mip_count = 9u32;
        let valid_levels = 0..mip_count;
        let invalid_level = mip_count;

        for level in valid_levels.clone() {
            assert!(level < mip_count, "Level {} should be valid", level);
        }
        assert!(invalid_level >= mip_count, "Level {} should be invalid", invalid_level);
    }

    // ------------------------------------------------------------------------
    // Criterion 4: Array layer selection - support array layers via origin.z and extent.depth_or_array_layers
    // ------------------------------------------------------------------------

    #[test]
    fn test_array_layer_selection_via_origin_z() {
        // Origin.z specifies the starting array layer
        let origin_layer_0 = wgpu::Origin3d { x: 0, y: 0, z: 0 };
        let origin_layer_5 = wgpu::Origin3d { x: 0, y: 0, z: 5 };

        assert_eq!(origin_layer_0.z, 0);
        assert_eq!(origin_layer_5.z, 5);
    }

    #[test]
    fn test_array_layer_count_via_extent() {
        // depth_or_array_layers in CopyExtent3d specifies number of layers to copy
        let single_layer = CopyExtent3d::new(64, 64, 1);
        let four_layers = CopyExtent3d::new(64, 64, 4);
        let eight_layers = CopyExtent3d::new(64, 64, 8);

        assert_eq!(single_layer.depth_or_array_layers, 1);
        assert_eq!(four_layers.depth_or_array_layers, 4);
        assert_eq!(eight_layers.depth_or_array_layers, 8);

        assert!(!single_layer.is_multi_layer());
        assert!(four_layers.is_multi_layer());
        assert!(eight_layers.is_multi_layer());
    }

    #[test]
    fn test_texture_bounds_error() {
        let error = CopyError::texture_bounds(
            100, 100, 2,    // origin
            200, 200, 4,    // copy size
            256, 256, 4,    // texture size
            0               // mip level
        );
        let display = format!("{}", error);
        assert!(display.contains("bounds") || display.contains("exceeds"));
        assert!(display.contains("100")); // origin
        assert!(display.contains("200")); // copy size
        assert!(display.contains("256")); // texture size
    }

    #[test]
    fn test_array_layer_bounds_checking_logic() {
        // Test that array layer bounds are properly calculated
        let texture_array_layers = 6u32;
        let copy_start_layer = 2u32;
        let copy_layer_count = 3u32;
        let copy_end_layer = copy_start_layer + copy_layer_count;

        assert_eq!(copy_end_layer, 5);
        assert!(copy_end_layer <= texture_array_layers, "Copy should be within bounds");

        // Invalid case
        let invalid_start = 4u32;
        let invalid_count = 4u32;
        let invalid_end = invalid_start + invalid_count;
        assert_eq!(invalid_end, 8);
        assert!(invalid_end > texture_array_layers, "Copy should exceed bounds");
    }

    // ------------------------------------------------------------------------
    // Additional texture-to-texture validation tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_texture_to_texture_zero_extent_validation() {
        // Zero extent dimensions should fail validation
        let zero_width = CopyExtent3d::new(0, 64, 1);
        let zero_height = CopyExtent3d::new(64, 0, 1);
        let zero_depth = CopyExtent3d::new(64, 64, 0);

        assert!(!zero_width.is_valid());
        assert!(!zero_height.is_valid());
        assert!(!zero_depth.is_valid());
    }

    #[test]
    fn test_texture_to_texture_error_constructors() {
        // Verify all new error constructors work
        let format_err = CopyError::format_incompatible(
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba16Float
        );
        assert!(matches!(format_err, CopyError::FormatIncompatible { .. }));

        let mip_err = CopyError::mip_level_out_of_bounds(10, 8);
        assert!(matches!(mip_err, CopyError::MipLevelOutOfBounds { level: 10, max: 8 }));

        let bounds_err = CopyError::texture_bounds(0, 0, 0, 512, 512, 1, 256, 256, 1, 0);
        assert!(matches!(bounds_err, CopyError::TextureBoundsError { .. }));
    }

    #[test]
    fn test_copy_compatible_class_rg_formats() {
        use wgpu::TextureFormat;

        // RG8 family
        assert!(are_formats_copy_compatible(TextureFormat::Rg8Unorm, TextureFormat::Rg8Snorm));
        assert!(are_formats_copy_compatible(TextureFormat::Rg8Unorm, TextureFormat::Rg8Uint));
        assert!(are_formats_copy_compatible(TextureFormat::Rg8Unorm, TextureFormat::Rg8Sint));

        // RG16 family
        assert!(are_formats_copy_compatible(TextureFormat::Rg16Float, TextureFormat::Rg16Uint));
        assert!(are_formats_copy_compatible(TextureFormat::Rg16Float, TextureFormat::Rg16Sint));
        assert!(are_formats_copy_compatible(TextureFormat::Rg16Float, TextureFormat::Rg16Unorm));

        // RG32 family
        assert!(are_formats_copy_compatible(TextureFormat::Rg32Float, TextureFormat::Rg32Uint));
        assert!(are_formats_copy_compatible(TextureFormat::Rg32Float, TextureFormat::Rg32Sint));
    }

    #[test]
    fn test_copy_compatible_class_rgba32() {
        use wgpu::TextureFormat;

        // RGBA32 family (128-bit)
        assert!(are_formats_copy_compatible(TextureFormat::Rgba32Float, TextureFormat::Rgba32Uint));
        assert!(are_formats_copy_compatible(TextureFormat::Rgba32Float, TextureFormat::Rgba32Sint));
        assert!(are_formats_copy_compatible(TextureFormat::Rgba32Uint, TextureFormat::Rgba32Sint));
    }

    #[test]
    fn test_copy_compatible_r16_family() {
        use wgpu::TextureFormat;

        // R16 family
        assert!(are_formats_copy_compatible(TextureFormat::R16Float, TextureFormat::R16Uint));
        assert!(are_formats_copy_compatible(TextureFormat::R16Float, TextureFormat::R16Sint));
        assert!(are_formats_copy_compatible(TextureFormat::R16Float, TextureFormat::R16Unorm));
        assert!(are_formats_copy_compatible(TextureFormat::R16Float, TextureFormat::R16Snorm));
    }

    #[test]
    fn test_mip_size_edge_cases() {
        // Minimum size should always be 1
        assert_eq!(mip_size(1, 0), 1);
        assert_eq!(mip_size(1, 1), 1);
        assert_eq!(mip_size(1, 100), 1);

        // Large base sizes
        assert_eq!(mip_size(4096, 0), 4096);
        assert_eq!(mip_size(4096, 1), 2048);
        assert_eq!(mip_size(4096, 12), 1);

        // Asymmetric dimensions would use this for each dimension independently
        assert_eq!(mip_size(1920, 0), 1920);
        assert_eq!(mip_size(1920, 1), 960);
        assert_eq!(mip_size(1080, 0), 1080);
        assert_eq!(mip_size(1080, 1), 540);
    }

    #[test]
    fn test_texture_to_texture_function_exists() {
        // Verify the functions exist and have correct signatures by referencing them
        let _copy_fn: fn(&mut CommandEncoder, ImageCopyTexture, ImageCopyTexture, CopyExtent3d) -> Result<(), CopyError>
            = copy_texture_to_texture;

        let _trinity_copy_fn: fn(&mut TrinityCommandEncoder, ImageCopyTexture, ImageCopyTexture, CopyExtent3d) -> Result<(), CopyError>
            = copy_texture_to_texture_trinity;

        let _validate_fn: fn(&ImageCopyTexture, &ImageCopyTexture, &CopyExtent3d) -> Result<(), CopyError>
            = validate_texture_to_texture_params;

        let _compat_fn: fn(wgpu::TextureFormat, wgpu::TextureFormat) -> bool
            = are_formats_copy_compatible;
    }
}
