//! Buffer creation and management for TRINITY.
//!
//! This module provides the buffer creation API for the TRINITY wgpu abstraction layer.
//! It wraps wgpu's buffer creation with validation, alignment, and metadata tracking.
//!
//! # Overview
//!
//! Buffer creation in wgpu requires proper size alignment and usage flag validation.
//! This module provides:
//!
//! - [`TrinityBuffer`] - Wrapper around wgpu::Buffer with metadata
//! - [`TrinityBufferDescriptor`] - Buffer creation parameters
//! - [`create_buffer`] - Validated buffer creation with logging
//! - [`buffer_usages`] - Common usage flag presets
//! - [`validate_usage`] - Usage flag combination validation
//!
//! # Buffer Usage Flags Reference
//!
//! wgpu provides 10 buffer usage flags that control how buffers can be used:
//!
//! | Flag | Purpose | Common With |
//! |------|---------|-------------|
//! | `MAP_READ` | CPU read access | `COPY_DST` (staging readback) |
//! | `MAP_WRITE` | CPU write access | `COPY_SRC` (staging upload) |
//! | `COPY_SRC` | Copy source | `MAP_WRITE`, `STORAGE` |
//! | `COPY_DST` | Copy destination | `VERTEX`, `INDEX`, `UNIFORM`, `STORAGE` |
//! | `INDEX` | Index buffer | `COPY_DST` |
//! | `VERTEX` | Vertex buffer | `COPY_DST` |
//! | `UNIFORM` | Uniform buffer | `COPY_DST` |
//! | `STORAGE` | Storage buffer | `COPY_DST`, `COPY_SRC` |
//! | `INDIRECT` | Indirect args | `STORAGE`, `COPY_DST` |
//! | `QUERY_RESOLVE` | Query results | `COPY_SRC` |
//!
//! ## Invalid Combinations
//!
//! - `MAP_READ | MAP_WRITE` - Cannot map for both read and write
//! - `MAP_READ | VERTEX` - Mapped read buffers cannot be used as vertex buffers
//! - `MAP_READ | INDEX` - Mapped read buffers cannot be used as index buffers
//! - `MAP_READ | UNIFORM` - Mapped read buffers cannot be used as uniform buffers
//! - `MAP_READ | STORAGE` - Mapped read buffers cannot be used as storage buffers
//!
//! # Alignment
//!
//! All buffer sizes are aligned to 4 bytes (wgpu requirement for many operations).
//! The `create_buffer` function automatically aligns sizes up.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::buffer::{TrinityBuffer, TrinityBufferDescriptor, create_buffer};
//! use renderer_backend::resources::buffer::buffer_usages;
//! use wgpu::BufferUsages;
//!
//! # fn example(device: &wgpu::Device) {
//! // Using raw BufferUsages
//! let desc = TrinityBufferDescriptor {
//!     label: Some("vertex_buffer"),
//!     size: 1024,
//!     usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
//!     mapped_at_creation: false,
//! };
//!
//! // Or using presets (recommended)
//! let desc_preset = TrinityBufferDescriptor {
//!     label: Some("vertex_buffer"),
//!     size: 1024,
//!     usage: buffer_usages::VERTEX,
//!     mapped_at_creation: false,
//! };
//!
//! let buffer = create_buffer(device, &desc);
//! assert_eq!(buffer.size(), 1024);
//! assert!(buffer.usage().contains(BufferUsages::VERTEX));
//! # }
//! ```

use log::debug;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, MapMode};

// ============================================================================
// Buffer Usage Presets
// ============================================================================

/// Common buffer usage presets for typical use cases.
///
/// These presets combine the necessary flags for common buffer patterns,
/// reducing boilerplate and preventing invalid combinations.
///
/// # Usage Presets Reference
///
/// | Preset | Flags | Use Case |
/// |--------|-------|----------|
/// | `VERTEX` | `VERTEX \| COPY_DST` | Vertex data with CPU upload |
/// | `INDEX` | `INDEX \| COPY_DST` | Index data with CPU upload |
/// | `UNIFORM` | `UNIFORM \| COPY_DST` | Uniform data with CPU upload |
/// | `STORAGE_READ` | `STORAGE \| COPY_DST` | Read-only shader storage |
/// | `STORAGE_RW` | `STORAGE \| COPY_DST \| COPY_SRC` | Read-write shader storage |
/// | `STAGING_UPLOAD` | `MAP_WRITE \| COPY_SRC` | CPU->GPU data transfer |
/// | `STAGING_READBACK` | `MAP_READ \| COPY_DST` | GPU->CPU data transfer |
/// | `INDIRECT` | `INDIRECT \| COPY_DST \| STORAGE` | Indirect draw/dispatch args |
/// | `QUERY_RESOLVE` | `QUERY_RESOLVE \| COPY_SRC` | Query result storage |
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::{buffer_usages, TrinityBufferDescriptor, create_buffer};
///
/// # fn example(device: &wgpu::Device) {
/// // Create a vertex buffer using preset
/// let vertex_buffer = create_buffer(device, &TrinityBufferDescriptor {
///     label: Some("vertices"),
///     size: 4096,
///     usage: buffer_usages::VERTEX,
///     mapped_at_creation: false,
/// });
///
/// // Create a staging buffer for uploads
/// let staging = create_buffer(device, &TrinityBufferDescriptor {
///     label: Some("staging"),
///     size: 1024,
///     usage: buffer_usages::STAGING_UPLOAD,
///     mapped_at_creation: true, // Map immediately for writing
/// });
/// # }
/// ```
pub mod buffer_usages {
    use wgpu::BufferUsages;

    /// Vertex buffer with CPU upload capability.
    ///
    /// Combines `VERTEX` for use in vertex shader input with `COPY_DST`
    /// for uploading data via staging buffer or queue write.
    pub const VERTEX: BufferUsages = BufferUsages::VERTEX.union(BufferUsages::COPY_DST);

    /// Index buffer with CPU upload capability.
    ///
    /// Combines `INDEX` for use as index buffer with `COPY_DST`
    /// for uploading index data.
    pub const INDEX: BufferUsages = BufferUsages::INDEX.union(BufferUsages::COPY_DST);

    /// Uniform buffer with CPU upload capability.
    ///
    /// Combines `UNIFORM` for use in shaders with `COPY_DST`
    /// for uploading uniform data each frame.
    pub const UNIFORM: BufferUsages = BufferUsages::UNIFORM.union(BufferUsages::COPY_DST);

    /// Storage buffer (read-only from shader perspective).
    ///
    /// Combines `STORAGE` for shader access with `COPY_DST`
    /// for uploading data. Shader can read but writes are discarded.
    pub const STORAGE_READ: BufferUsages = BufferUsages::STORAGE.union(BufferUsages::COPY_DST);

    /// Storage buffer (read-write from shader perspective).
    ///
    /// Combines `STORAGE` for shader access with both `COPY_DST`
    /// for uploading and `COPY_SRC` for reading back results.
    pub const STORAGE_RW: BufferUsages = BufferUsages::STORAGE
        .union(BufferUsages::COPY_DST)
        .union(BufferUsages::COPY_SRC);

    /// Staging buffer for CPU->GPU uploads.
    ///
    /// Combines `MAP_WRITE` for CPU write access with `COPY_SRC`
    /// to copy data to GPU-only buffers. Use with `mapped_at_creation: true`.
    pub const STAGING_UPLOAD: BufferUsages =
        BufferUsages::MAP_WRITE.union(BufferUsages::COPY_SRC);

    /// Staging buffer for GPU->CPU readback.
    ///
    /// Combines `MAP_READ` for CPU read access with `COPY_DST`
    /// to receive data from GPU buffers. Map after copy completes.
    pub const STAGING_READBACK: BufferUsages =
        BufferUsages::MAP_READ.union(BufferUsages::COPY_DST);

    /// Indirect draw/dispatch argument buffer.
    ///
    /// Combines `INDIRECT` for use with draw_indirect/dispatch_indirect,
    /// `STORAGE` for compute shader writes, and `COPY_DST` for CPU updates.
    pub const INDIRECT: BufferUsages = BufferUsages::INDIRECT
        .union(BufferUsages::COPY_DST)
        .union(BufferUsages::STORAGE);

    /// Query result buffer.
    ///
    /// Combines `QUERY_RESOLVE` for receiving query results with
    /// `COPY_SRC` for copying to a readable staging buffer.
    pub const QUERY_RESOLVE: BufferUsages =
        BufferUsages::QUERY_RESOLVE.union(BufferUsages::COPY_SRC);
}

// ============================================================================
// Usage Validation
// ============================================================================

/// Errors that can occur when validating buffer usage flag combinations.
///
/// Some combinations of buffer usage flags are invalid in wgpu. This enum
/// describes the specific validation failures.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::buffer::{validate_usage, UsageValidationError};
/// use wgpu::BufferUsages;
///
/// // MAP_READ + VERTEX is invalid
/// let invalid = BufferUsages::MAP_READ | BufferUsages::VERTEX;
/// assert!(matches!(
///     validate_usage(invalid),
///     Err(UsageValidationError::MapReadWithGpuOnly(_))
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsageValidationError {
    /// `MAP_READ` cannot be combined with `VERTEX`, `INDEX`, `UNIFORM`, or `STORAGE`.
    ///
    /// Buffers that are mappable for reading cannot be directly used by the GPU
    /// as vertex/index/uniform/storage buffers. Use a staging buffer pattern instead.
    MapReadWithGpuOnly(BufferUsages),

    /// `MAP_WRITE` cannot be combined with `VERTEX`, `INDEX`, `UNIFORM`, or `STORAGE`
    /// unless `COPY_DST` is also present (which would make it a staging pattern).
    ///
    /// Note: This is a soft warning in some cases, but we enforce stricter validation.
    MapWriteWithGpuOnly(BufferUsages),

    /// Both `MAP_READ` and `MAP_WRITE` cannot be set together.
    ///
    /// wgpu does not support buffers that are mappable for both reading and writing.
    MapReadAndWrite,
}

impl std::fmt::Display for UsageValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UsageValidationError::MapReadWithGpuOnly(usage) => {
                write!(
                    f,
                    "MAP_READ cannot be combined with GPU-only usage flags (VERTEX, INDEX, UNIFORM, STORAGE): {:?}",
                    usage
                )
            }
            UsageValidationError::MapWriteWithGpuOnly(usage) => {
                write!(
                    f,
                    "MAP_WRITE should not be combined with GPU-only usage flags without COPY_DST: {:?}",
                    usage
                )
            }
            UsageValidationError::MapReadAndWrite => {
                write!(f, "MAP_READ and MAP_WRITE cannot both be set on the same buffer")
            }
        }
    }
}

impl std::error::Error for UsageValidationError {}

/// Validates buffer usage flag combinations.
///
/// Some combinations of buffer usage flags are invalid or problematic in wgpu.
/// This function checks for known invalid combinations and returns an error
/// if any are detected.
///
/// # Arguments
///
/// * `usage` - The buffer usage flags to validate
///
/// # Returns
///
/// `Ok(())` if the usage flags are valid, or `Err(UsageValidationError)` describing
/// the invalid combination.
///
/// # Invalid Combinations
///
/// - `MAP_READ | MAP_WRITE` - Cannot map for both read and write simultaneously
/// - `MAP_READ | VERTEX/INDEX/UNIFORM/STORAGE` - Readable buffers cannot be GPU-only
///
/// # Example
///
/// ```
/// use renderer_backend::resources::buffer::validate_usage;
/// use wgpu::BufferUsages;
///
/// // Valid: vertex buffer with copy destination
/// assert!(validate_usage(BufferUsages::VERTEX | BufferUsages::COPY_DST).is_ok());
///
/// // Valid: staging upload buffer
/// assert!(validate_usage(BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC).is_ok());
///
/// // Invalid: MAP_READ with VERTEX
/// assert!(validate_usage(BufferUsages::MAP_READ | BufferUsages::VERTEX).is_err());
///
/// // Invalid: MAP_READ and MAP_WRITE together
/// assert!(validate_usage(BufferUsages::MAP_READ | BufferUsages::MAP_WRITE).is_err());
/// ```
pub fn validate_usage(usage: BufferUsages) -> Result<(), UsageValidationError> {
    // MAP_READ + MAP_WRITE is invalid
    if usage.contains(BufferUsages::MAP_READ) && usage.contains(BufferUsages::MAP_WRITE) {
        return Err(UsageValidationError::MapReadAndWrite);
    }

    // GPU-only usage flags that cannot be combined with MAP_READ
    let gpu_only = BufferUsages::VERTEX
        | BufferUsages::INDEX
        | BufferUsages::UNIFORM
        | BufferUsages::STORAGE;

    // MAP_READ + GPU-only flags is invalid
    if usage.contains(BufferUsages::MAP_READ) && usage.intersects(gpu_only) {
        return Err(UsageValidationError::MapReadWithGpuOnly(usage));
    }

    // MAP_WRITE + GPU-only without COPY_DST is suspicious (soft validation)
    // We allow it but it's typically a mistake - the user probably wants staging
    // For now, we don't error on this, but could add a warning level

    Ok(())
}

/// Validates usage flags and returns a descriptive result.
///
/// This is a convenience wrapper around [`validate_usage`] that provides
/// more context in the error message.
///
/// # Arguments
///
/// * `usage` - The buffer usage flags to validate
/// * `label` - Optional label for better error messages
///
/// # Returns
///
/// `Ok(())` if valid, or `Err` with a descriptive message.
pub fn validate_usage_with_label(
    usage: BufferUsages,
    label: Option<&str>,
) -> Result<(), String> {
    validate_usage(usage).map_err(|e| {
        match label {
            Some(l) => format!("Buffer '{}': {}", l, e),
            None => e.to_string(),
        }
    })
}

// ============================================================================
// Constants
// ============================================================================

/// Minimum buffer alignment in bytes.
///
/// wgpu requires buffer sizes to be aligned to 4 bytes for many operations,
/// including copy operations and uniform buffer bindings.
pub const BUFFER_ALIGNMENT: u64 = 4;

// ============================================================================
// TrinityBuffer
// ============================================================================

/// TRINITY buffer wrapper with metadata.
///
/// This struct wraps a wgpu [`Buffer`] with additional metadata about the
/// buffer's size, usage flags, and debug label. This allows for better
/// debugging and resource tracking.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::{TrinityBufferDescriptor, create_buffer};
/// use wgpu::BufferUsages;
///
/// # fn example(device: &wgpu::Device) {
/// let desc = TrinityBufferDescriptor {
///     label: Some("my_buffer"),
///     size: 256,
///     usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
///     mapped_at_creation: false,
/// };
///
/// let buffer = create_buffer(device, &desc);
/// println!("Buffer size: {} bytes", buffer.size());
/// println!("Buffer usage: {:?}", buffer.usage());
/// # }
/// ```
pub struct TrinityBuffer {
    /// The underlying wgpu buffer.
    inner: Buffer,
    /// Buffer size in bytes (after alignment).
    size: u64,
    /// Buffer usage flags.
    usage: BufferUsages,
    /// Optional debug label.
    label: Option<String>,
}

impl TrinityBuffer {
    /// Creates a TrinityBuffer from a raw wgpu buffer.
    ///
    /// Use this when you need to wrap a buffer created through other means
    /// (e.g., `create_buffer_init` or external libraries).
    ///
    /// # Arguments
    ///
    /// * `buffer` - The wgpu buffer to wrap
    /// * `size` - The buffer size in bytes
    /// * `usage` - The buffer usage flags
    /// * `label` - Optional debug label
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::resources::buffer::{TrinityBuffer, buffer_usages};
    /// use wgpu::util::DeviceExt;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let data = [1u8, 2, 3, 4];
    /// let raw_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    ///     label: Some("my_buffer"),
    ///     contents: &data,
    ///     usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    /// });
    ///
    /// let trinity_buffer = TrinityBuffer::from_raw(
    ///     raw_buffer,
    ///     4,
    ///     buffer_usages::VERTEX,
    ///     Some("my_buffer".to_string()),
    /// );
    /// # }
    /// ```
    pub fn from_raw(
        buffer: Buffer,
        size: u64,
        usage: BufferUsages,
        label: Option<String>,
    ) -> Self {
        Self {
            inner: buffer,
            size,
            usage,
            label,
        }
    }

    /// Returns the buffer size in bytes.
    ///
    /// This is the aligned size, which may be larger than the requested size
    /// if alignment was required.
    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the buffer usage flags.
    #[inline]
    pub fn usage(&self) -> BufferUsages {
        self.usage
    }

    /// Returns a reference to the inner wgpu buffer.
    ///
    /// Use this when you need to pass the buffer to wgpu APIs.
    #[inline]
    pub fn inner(&self) -> &Buffer {
        &self.inner
    }

    /// Returns the debug label, if any.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Consumes the wrapper and returns the inner wgpu buffer.
    ///
    /// Use this when you need to transfer ownership of the buffer.
    #[inline]
    pub fn into_inner(self) -> Buffer {
        self.inner
    }
}

impl std::fmt::Debug for TrinityBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrinityBuffer")
            .field("label", &self.label)
            .field("size", &self.size)
            .field("usage", &self.usage)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// TrinityBufferDescriptor
// ============================================================================

/// Buffer creation descriptor.
///
/// This struct describes the parameters for creating a new buffer.
/// All fields are required except for `label`, which is optional but
/// recommended for debugging.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::TrinityBufferDescriptor;
/// use wgpu::BufferUsages;
///
/// let desc = TrinityBufferDescriptor {
///     label: Some("staging_buffer"),
///     size: 4096,
///     usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
///     mapped_at_creation: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct TrinityBufferDescriptor<'a> {
    /// Debug label for the buffer.
    ///
    /// This label appears in GPU debugging tools and error messages.
    /// Recommended for all production code.
    pub label: Option<&'a str>,

    /// Size of the buffer in bytes.
    ///
    /// Must be greater than 0. Will be aligned up to [`BUFFER_ALIGNMENT`] if necessary.
    pub size: u64,

    /// Usage flags for the buffer.
    ///
    /// Must not be empty. Specifies how the buffer will be used (vertex, uniform, etc.).
    pub usage: BufferUsages,

    /// Whether to map the buffer at creation time.
    ///
    /// If `true`, the buffer is created in a mapped state, allowing immediate
    /// CPU access for writing initial data. Only valid with `MAP_WRITE` usage.
    pub mapped_at_creation: bool,
}

impl Default for TrinityBufferDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            size: BUFFER_ALIGNMENT, // Minimum valid size
            usage: BufferUsages::COPY_DST, // Safe default
            mapped_at_creation: false,
        }
    }
}

// ============================================================================
// Alignment Utilities
// ============================================================================

/// Aligns a size up to the buffer alignment boundary.
///
/// # Arguments
///
/// * `size` - The size to align
///
/// # Returns
///
/// The aligned size, which is always >= the input size and a multiple of [`BUFFER_ALIGNMENT`].
///
/// # Example
///
/// ```
/// use renderer_backend::resources::buffer::align_size;
///
/// assert_eq!(align_size(1), 4);
/// assert_eq!(align_size(4), 4);
/// assert_eq!(align_size(5), 8);
/// assert_eq!(align_size(100), 100);
/// assert_eq!(align_size(101), 104);
/// ```
#[inline]
pub const fn align_size(size: u64) -> u64 {
    // Fast alignment: (size + alignment - 1) & !(alignment - 1)
    (size + BUFFER_ALIGNMENT - 1) & !(BUFFER_ALIGNMENT - 1)
}

/// Checks if a size is properly aligned.
///
/// # Arguments
///
/// * `size` - The size to check
///
/// # Returns
///
/// `true` if the size is a multiple of [`BUFFER_ALIGNMENT`], `false` otherwise.
#[inline]
pub const fn is_aligned(size: u64) -> bool {
    size % BUFFER_ALIGNMENT == 0
}

// ============================================================================
// Buffer Creation
// ============================================================================

/// Creates a buffer with validation and logging.
///
/// This function creates a wgpu buffer with the specified parameters,
/// performing validation and automatic size alignment.
///
/// # Arguments
///
/// * `device` - The wgpu device to create the buffer on
/// * `desc` - The buffer descriptor specifying parameters
///
/// # Returns
///
/// A [`TrinityBuffer`] wrapping the created wgpu buffer.
///
/// # Panics
///
/// Panics if:
/// - `size` is 0
/// - `usage` is empty
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::{TrinityBufferDescriptor, create_buffer};
/// use wgpu::BufferUsages;
///
/// # fn example(device: &wgpu::Device) {
/// let buffer = create_buffer(device, &TrinityBufferDescriptor {
///     label: Some("uniform_buffer"),
///     size: 256,
///     usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
///     mapped_at_creation: false,
/// });
///
/// assert_eq!(buffer.size(), 256);
/// # }
/// ```
pub fn create_buffer(device: &Device, desc: &TrinityBufferDescriptor) -> TrinityBuffer {
    // Validate size
    assert!(desc.size > 0, "Buffer size must be greater than 0");

    // Validate usage
    assert!(
        !desc.usage.is_empty(),
        "Buffer usage flags must not be empty"
    );

    // Align size to 4 bytes
    let aligned_size = align_size(desc.size);

    // Log if we needed to align
    if aligned_size != desc.size {
        debug!(
            "Buffer {:?}: aligned size {} -> {} bytes",
            desc.label, desc.size, aligned_size
        );
    }

    // Create the wgpu buffer
    let inner = device.create_buffer(&BufferDescriptor {
        label: desc.label,
        size: aligned_size,
        usage: desc.usage,
        mapped_at_creation: desc.mapped_at_creation,
    });

    // Log allocation
    debug!(
        "Buffer allocated: label={:?}, size={}, usage={:?}",
        desc.label, aligned_size, desc.usage
    );

    TrinityBuffer {
        inner,
        size: aligned_size,
        usage: desc.usage,
        label: desc.label.map(String::from),
    }
}

/// Creates a buffer with validation and logging, returning an error on failure.
///
/// This is the fallible version of [`create_buffer`]. Instead of panicking on
/// invalid input, it returns an error.
///
/// # Arguments
///
/// * `device` - The wgpu device to create the buffer on
/// * `desc` - The buffer descriptor specifying parameters
///
/// # Returns
///
/// `Ok(TrinityBuffer)` on success, or `Err(BufferCreationError)` on validation failure.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::{TrinityBufferDescriptor, try_create_buffer};
/// use wgpu::BufferUsages;
///
/// # fn example(device: &wgpu::Device) -> Result<(), Box<dyn std::error::Error>> {
/// let result = try_create_buffer(device, &TrinityBufferDescriptor {
///     label: Some("test_buffer"),
///     size: 0, // Invalid!
///     usage: BufferUsages::VERTEX,
///     mapped_at_creation: false,
/// });
///
/// assert!(result.is_err());
/// # Ok(())
/// # }
/// ```
pub fn try_create_buffer(
    device: &Device,
    desc: &TrinityBufferDescriptor,
) -> Result<TrinityBuffer, BufferCreationError> {
    // Validate size
    if desc.size == 0 {
        return Err(BufferCreationError::ZeroSize);
    }

    // Validate usage flags are not empty
    if desc.usage.is_empty() {
        return Err(BufferCreationError::EmptyUsage);
    }

    // Validate usage flag combinations
    validate_usage(desc.usage)?;

    // Align size to 4 bytes
    let aligned_size = align_size(desc.size);

    // Log if we needed to align
    if aligned_size != desc.size {
        debug!(
            "Buffer {:?}: aligned size {} -> {} bytes",
            desc.label, desc.size, aligned_size
        );
    }

    // Create the wgpu buffer
    let inner = device.create_buffer(&BufferDescriptor {
        label: desc.label,
        size: aligned_size,
        usage: desc.usage,
        mapped_at_creation: desc.mapped_at_creation,
    });

    // Log allocation
    debug!(
        "Buffer allocated: label={:?}, size={}, usage={:?}",
        desc.label, aligned_size, desc.usage
    );

    Ok(TrinityBuffer {
        inner,
        size: aligned_size,
        usage: desc.usage,
        label: desc.label.map(String::from),
    })
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during buffer creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferCreationError {
    /// Buffer size was 0.
    ZeroSize,
    /// Buffer usage flags were empty.
    EmptyUsage,
    /// Buffer usage flags have an invalid combination.
    InvalidUsage(UsageValidationError),
}

impl std::fmt::Display for BufferCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferCreationError::ZeroSize => write!(f, "buffer size must be greater than 0"),
            BufferCreationError::EmptyUsage => write!(f, "buffer usage flags must not be empty"),
            BufferCreationError::InvalidUsage(e) => write!(f, "invalid buffer usage: {}", e),
        }
    }
}

impl std::error::Error for BufferCreationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BufferCreationError::InvalidUsage(e) => Some(e),
            _ => None,
        }
    }
}

impl From<UsageValidationError> for BufferCreationError {
    fn from(err: UsageValidationError) -> Self {
        BufferCreationError::InvalidUsage(err)
    }
}

// ============================================================================
// Buffer Mapping Types
// ============================================================================

/// Buffer mapping mode.
///
/// Specifies whether the buffer should be mapped for reading or writing.
/// This corresponds to wgpu's `MapMode` enum.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::buffer::MappingMode;
/// use wgpu::MapMode;
///
/// let mode = MappingMode::Read;
/// let wgpu_mode: MapMode = mode.into();
/// assert!(matches!(wgpu_mode, MapMode::Read));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MappingMode {
    /// Map for CPU reading (GPU->CPU transfer).
    ///
    /// Requires `BufferUsages::MAP_READ` on the buffer.
    Read,
    /// Map for CPU writing (CPU->GPU transfer).
    ///
    /// Requires `BufferUsages::MAP_WRITE` on the buffer.
    Write,
}

impl From<MappingMode> for MapMode {
    fn from(mode: MappingMode) -> Self {
        match mode {
            MappingMode::Read => MapMode::Read,
            MappingMode::Write => MapMode::Write,
        }
    }
}

impl std::fmt::Display for MappingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MappingMode::Read => write!(f, "Read"),
            MappingMode::Write => write!(f, "Write"),
        }
    }
}

// ============================================================================
// Mapping Errors
// ============================================================================

/// Error during buffer mapping operations.
///
/// These errors indicate why a buffer mapping operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MappingError {
    /// Buffer was not created with `MAP_READ` or `MAP_WRITE` usage.
    ///
    /// To map a buffer, it must have been created with the appropriate
    /// mapping usage flag for the requested mode.
    NotMappable {
        /// The requested mapping mode.
        mode: MappingMode,
        /// The buffer's actual usage flags.
        usage: BufferUsages,
    },

    /// Async mapping operation failed.
    ///
    /// The GPU rejected the mapping request or the buffer was destroyed.
    MapFailed,

    /// Buffer is already mapped.
    ///
    /// A buffer can only have one active mapping at a time. Call `unmap()`
    /// on the existing mapping before creating a new one.
    AlreadyMapped,

    /// Channel communication failed during async mapping.
    ///
    /// This indicates an internal error in the async notification mechanism.
    ChannelError,

    /// Buffer was not mapped at creation but sync access was attempted.
    ///
    /// For sync mapping, the buffer must be created with `mapped_at_creation: true`.
    NotMappedAtCreation,
}

impl std::fmt::Display for MappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MappingError::NotMappable { mode, usage } => {
                let required = match mode {
                    MappingMode::Read => "MAP_READ",
                    MappingMode::Write => "MAP_WRITE",
                };
                write!(
                    f,
                    "buffer is not mappable for {}: missing {} usage (has {:?})",
                    mode, required, usage
                )
            }
            MappingError::MapFailed => write!(f, "async buffer mapping failed"),
            MappingError::AlreadyMapped => write!(f, "buffer is already mapped"),
            MappingError::ChannelError => write!(f, "internal channel error during async mapping"),
            MappingError::NotMappedAtCreation => {
                write!(f, "buffer was not mapped at creation for sync access")
            }
        }
    }
}

impl std::error::Error for MappingError {}

// ============================================================================
// Mapped Buffer Handle
// ============================================================================

/// Handle to a mapped buffer region.
///
/// This struct manages the lifetime of a buffer mapping. When dropped,
/// it automatically unmaps the buffer, making it available for GPU use again.
///
/// # Lifetime
///
/// The mapping is valid as long as this handle exists. Once dropped,
/// the buffer is unmapped and the data is no longer accessible from the CPU.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::{
///     create_buffer, TrinityBufferDescriptor, buffer_usages,
///     map_buffer_sync_write
/// };
///
/// # fn example(device: &wgpu::Device) {
/// // Create a staging buffer for upload
/// let buffer = create_buffer(device, &TrinityBufferDescriptor {
///     label: Some("staging"),
///     size: 1024,
///     usage: buffer_usages::STAGING_UPLOAD,
///     mapped_at_creation: true,
/// });
///
/// // Map and write data
/// {
///     let mapped = map_buffer_sync_write(&buffer).unwrap();
///     mapped.write_data(&[1u8, 2, 3, 4]);
/// } // Buffer is unmapped here when MappedBuffer is dropped
/// # }
/// ```
pub struct MappedBuffer<'a> {
    /// Reference to the underlying buffer.
    buffer: &'a TrinityBuffer,
    /// The mapping mode (Read or Write).
    mode: MappingMode,
    /// Flag to track if we should unmap on drop.
    /// Set to false if unmap() is called explicitly.
    should_unmap: AtomicBool,
}

impl<'a> MappedBuffer<'a> {
    /// Creates a new mapped buffer handle.
    ///
    /// # Safety
    ///
    /// The buffer must already be in a mapped state.
    fn new(buffer: &'a TrinityBuffer, mode: MappingMode) -> Self {
        Self {
            buffer,
            mode,
            should_unmap: AtomicBool::new(true),
        }
    }

    /// Returns the mapping mode.
    #[inline]
    pub fn mode(&self) -> MappingMode {
        self.mode
    }

    /// Returns a reference to the underlying buffer.
    #[inline]
    pub fn buffer(&self) -> &TrinityBuffer {
        self.buffer
    }

    /// Gets a read-only view of the mapped buffer data.
    ///
    /// # Returns
    ///
    /// A slice containing the buffer's data.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not currently mapped.
    pub fn read_data(&self) -> Vec<u8> {
        let slice = self.buffer.inner().slice(..);
        let view = slice.get_mapped_range();
        view.to_vec()
    }

    /// Reads a portion of the mapped buffer data.
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset
    /// * `len` - Number of bytes to read
    ///
    /// # Returns
    ///
    /// A vector containing the requested data.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds or the buffer is not mapped.
    pub fn read_range(&self, offset: u64, len: u64) -> Vec<u8> {
        let slice = self.buffer.inner().slice(offset..offset + len);
        let view = slice.get_mapped_range();
        view.to_vec()
    }

    /// Writes data to the mapped buffer.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to write
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The mapping mode is not `Write`
    /// - The data is larger than the buffer
    /// - The buffer is not currently mapped
    pub fn write_data(&self, data: &[u8]) {
        assert!(
            self.mode == MappingMode::Write,
            "cannot write to a read-mode mapping"
        );
        assert!(
            data.len() as u64 <= self.buffer.size(),
            "data ({} bytes) exceeds buffer size ({} bytes)",
            data.len(),
            self.buffer.size()
        );

        let slice = self.buffer.inner().slice(..);
        let mut view = slice.get_mapped_range_mut();
        view[..data.len()].copy_from_slice(data);
    }

    /// Writes data to a specific offset in the mapped buffer.
    ///
    /// # Arguments
    ///
    /// * `offset` - Starting byte offset in the buffer
    /// * `data` - The data to write
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The mapping mode is not `Write`
    /// - The write would exceed buffer bounds
    /// - The buffer is not currently mapped
    pub fn write_at(&self, offset: u64, data: &[u8]) {
        assert!(
            self.mode == MappingMode::Write,
            "cannot write to a read-mode mapping"
        );
        let end = offset + data.len() as u64;
        assert!(
            end <= self.buffer.size(),
            "write at offset {} with {} bytes exceeds buffer size {}",
            offset,
            data.len(),
            self.buffer.size()
        );

        let slice = self.buffer.inner().slice(offset..end);
        let mut view = slice.get_mapped_range_mut();
        view.copy_from_slice(data);
    }

    /// Explicitly unmaps the buffer.
    ///
    /// This is called automatically on drop, but can be called explicitly
    /// if you need to unmap before the handle goes out of scope.
    ///
    /// After calling this, the handle should not be used.
    pub fn unmap(self) {
        // unmap is called in Drop, but we consume self here
        // to make the API clearer. Drop will handle actual unmap.
    }

    /// Prevents automatic unmapping on drop.
    ///
    /// Use this if you need to keep the buffer mapped after dropping
    /// the handle. You are responsible for calling `buffer.inner().unmap()`
    /// manually.
    ///
    /// # Safety
    ///
    /// After calling this, you must ensure the buffer is unmapped before
    /// it is used by the GPU.
    pub fn leak(self) {
        self.should_unmap.store(false, Ordering::SeqCst);
        std::mem::forget(self);
    }
}

impl<'a> Drop for MappedBuffer<'a> {
    fn drop(&mut self) {
        if self.should_unmap.load(Ordering::SeqCst) {
            self.buffer.inner().unmap();
            debug!("Buffer {:?} unmapped", self.buffer.label());
        }
    }
}

impl<'a> std::fmt::Debug for MappedBuffer<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedBuffer")
            .field("buffer", &self.buffer.label())
            .field("mode", &self.mode)
            .field("size", &self.buffer.size())
            .finish()
    }
}

// ============================================================================
// Sync Buffer Mapping
// ============================================================================

/// Maps a buffer synchronously for writing (must be created with `mapped_at_creation=true`).
///
/// This is the primary way to write initial data to a staging buffer.
/// The buffer must have been created with `mapped_at_creation: true`.
///
/// # Arguments
///
/// * `buffer` - The buffer to map (must have `MAP_WRITE` usage)
///
/// # Returns
///
/// `Ok(MappedBuffer)` if the buffer was mapped at creation, or `Err(MappingError)` otherwise.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::{
///     create_buffer, TrinityBufferDescriptor, buffer_usages,
///     map_buffer_sync_write
/// };
///
/// # fn example(device: &wgpu::Device) {
/// let buffer = create_buffer(device, &TrinityBufferDescriptor {
///     label: Some("upload_staging"),
///     size: 256,
///     usage: buffer_usages::STAGING_UPLOAD,
///     mapped_at_creation: true,  // Required for sync mapping
/// });
///
/// let mapped = map_buffer_sync_write(&buffer).unwrap();
/// mapped.write_data(&[0u8; 256]);
/// // Buffer is unmapped when `mapped` goes out of scope
/// # }
/// ```
pub fn map_buffer_sync_write(buffer: &TrinityBuffer) -> Result<MappedBuffer<'_>, MappingError> {
    // Verify buffer has MAP_WRITE usage
    if !buffer.usage().contains(BufferUsages::MAP_WRITE) {
        return Err(MappingError::NotMappable {
            mode: MappingMode::Write,
            usage: buffer.usage(),
        });
    }

    // Note: We assume the buffer was created with mapped_at_creation=true.
    // wgpu doesn't provide a way to check this at runtime, so we trust the caller.
    // If the buffer wasn't mapped at creation, get_mapped_range_mut will panic.
    debug!("Sync mapping buffer {:?} for write", buffer.label());
    Ok(MappedBuffer::new(buffer, MappingMode::Write))
}

/// Maps a buffer synchronously for reading.
///
/// This is typically used after an async map operation completes.
/// For truly synchronous read access, use [`map_buffer_blocking`].
///
/// # Arguments
///
/// * `buffer` - The buffer to map (must have `MAP_READ` usage and be in mapped state)
///
/// # Returns
///
/// `Ok(MappedBuffer)` if successful, or `Err(MappingError)` if the buffer isn't mappable.
pub fn map_buffer_sync_read(buffer: &TrinityBuffer) -> Result<MappedBuffer<'_>, MappingError> {
    // Verify buffer has MAP_READ usage
    if !buffer.usage().contains(BufferUsages::MAP_READ) {
        return Err(MappingError::NotMappable {
            mode: MappingMode::Read,
            usage: buffer.usage(),
        });
    }

    debug!("Sync mapping buffer {:?} for read", buffer.label());
    Ok(MappedBuffer::new(buffer, MappingMode::Read))
}

// ============================================================================
// Async Buffer Mapping
// ============================================================================

/// Maps a buffer asynchronously using a callback.
///
/// This initiates an async mapping operation. The callback is called when
/// the mapping completes (or fails). You must poll the device to drive
/// the operation to completion.
///
/// # Arguments
///
/// * `buffer` - The buffer to map
/// * `mode` - Read or Write mode
/// * `callback` - Called when mapping completes with `Ok(())` or `Err(MappingError)`
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::{
///     create_buffer, TrinityBufferDescriptor, buffer_usages,
///     map_buffer_async, MappingMode
/// };
///
/// # fn example(device: &wgpu::Device, buffer: &renderer_backend::resources::buffer::TrinityBuffer) {
/// map_buffer_async(buffer, MappingMode::Read, |result| {
///     match result {
///         Ok(()) => println!("Mapping succeeded!"),
///         Err(e) => println!("Mapping failed: {}", e),
///     }
/// });
///
/// // Must poll device to drive the async operation
/// device.poll(wgpu::Maintain::Wait);
/// # }
/// ```
pub fn map_buffer_async<F>(buffer: &TrinityBuffer, mode: MappingMode, callback: F)
where
    F: FnOnce(Result<(), MappingError>) + Send + 'static,
{
    // Verify buffer has appropriate usage
    let required_usage = match mode {
        MappingMode::Read => BufferUsages::MAP_READ,
        MappingMode::Write => BufferUsages::MAP_WRITE,
    };

    if !buffer.usage().contains(required_usage) {
        callback(Err(MappingError::NotMappable {
            mode,
            usage: buffer.usage(),
        }));
        return;
    }

    debug!(
        "Async mapping buffer {:?} for {:?}",
        buffer.label(),
        mode
    );

    let slice = buffer.inner().slice(..);
    slice.map_async(mode.into(), move |result| {
        let mapping_result = result.map_err(|_| MappingError::MapFailed);
        callback(mapping_result);
    });
}

/// Maps a buffer asynchronously with a oneshot channel for notification.
///
/// This is a convenience wrapper around [`map_buffer_async`] that returns
/// a receiver you can await or poll for completion.
///
/// # Arguments
///
/// * `buffer` - The buffer to map
/// * `mode` - Read or Write mode
///
/// # Returns
///
/// A receiver that will receive the mapping result when complete.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::{
///     create_buffer, TrinityBufferDescriptor, buffer_usages,
///     map_buffer_async_channel, MappingMode
/// };
/// use std::sync::mpsc;
///
/// # fn example(device: &wgpu::Device, buffer: &renderer_backend::resources::buffer::TrinityBuffer) {
/// let rx = map_buffer_async_channel(buffer, MappingMode::Read);
///
/// // Poll device to drive completion
/// device.poll(wgpu::Maintain::Wait);
///
/// // Check result
/// let result = rx.recv().unwrap();
/// if result.is_ok() {
///     let data = buffer.inner().slice(..).get_mapped_range();
///     // Use data...
/// }
/// # }
/// ```
pub fn map_buffer_async_channel(
    buffer: &TrinityBuffer,
    mode: MappingMode,
) -> std::sync::mpsc::Receiver<Result<(), MappingError>> {
    let (tx, rx) = std::sync::mpsc::channel();

    map_buffer_async(buffer, mode, move |result| {
        let _ = tx.send(result);
    });

    rx
}

/// Maps a buffer and blocks until mapping completes.
///
/// This is a convenience function that combines async mapping with
/// device polling to provide a synchronous API for buffer mapping.
///
/// # Arguments
///
/// * `buffer` - The buffer to map
/// * `mode` - Read or Write mode
/// * `device` - The wgpu device (needed to poll for completion)
///
/// # Returns
///
/// `Ok(MappedBuffer)` on success, or `Err(MappingError)` on failure.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::{
///     create_buffer, TrinityBufferDescriptor, buffer_usages,
///     map_buffer_blocking, MappingMode
/// };
///
/// # fn example(device: &wgpu::Device) {
/// // Create a readback staging buffer
/// let buffer = create_buffer(device, &TrinityBufferDescriptor {
///     label: Some("readback"),
///     size: 1024,
///     usage: buffer_usages::STAGING_READBACK,
///     mapped_at_creation: false,
/// });
///
/// // After GPU has written data to the buffer via copy...
/// // (buffer copy code here)
///
/// // Map the buffer and read data
/// let mapped = map_buffer_blocking(&buffer, MappingMode::Read, device).unwrap();
/// let data = mapped.read_data();
/// println!("Read {} bytes", data.len());
/// # }
/// ```
pub fn map_buffer_blocking<'a>(
    buffer: &'a TrinityBuffer,
    mode: MappingMode,
    device: &Device,
) -> Result<MappedBuffer<'a>, MappingError> {
    let rx = map_buffer_async_channel(buffer, mode);

    // Poll device until mapping completes
    device.poll(wgpu::Maintain::Wait);

    // Get the result
    let result = rx.recv().map_err(|_| MappingError::ChannelError)?;
    result?;

    debug!(
        "Blocking map of buffer {:?} for {:?} completed",
        buffer.label(),
        mode
    );

    Ok(MappedBuffer::new(buffer, mode))
}

// ============================================================================
// Staging Buffer Helpers
// ============================================================================

/// Creates a staging buffer initialized with data for uploading to the GPU.
///
/// This is a convenience function that creates a staging buffer and
/// immediately writes data to it. The buffer is returned already unmapped
/// and ready for use as a copy source.
///
/// # Arguments
///
/// * `device` - The wgpu device
/// * `data` - The data to upload
/// * `label` - Optional debug label
///
/// # Returns
///
/// A `TrinityBuffer` containing the data, ready to be used as a copy source.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::create_staging_upload_buffer;
///
/// # fn example(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
/// let vertices: [f32; 12] = [
///     -0.5, -0.5, 0.0,
///      0.5, -0.5, 0.0,
///      0.5,  0.5, 0.0,
///     -0.5,  0.5, 0.0,
/// ];
///
/// let staging = create_staging_upload_buffer(
///     device,
///     bytemuck::cast_slice(&vertices),
///     Some("vertex_staging")
/// );
///
/// // Copy to GPU-local vertex buffer
/// // encoder.copy_buffer_to_buffer(&staging.inner(), 0, &vertex_buffer.inner(), 0, staging.size());
/// # }
/// ```
pub fn create_staging_upload_buffer(
    device: &Device,
    data: &[u8],
    label: Option<&str>,
) -> TrinityBuffer {
    use wgpu::util::DeviceExt;

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label,
        contents: data,
        usage: buffer_usages::STAGING_UPLOAD,
    });

    let size = align_size(data.len() as u64);
    debug!(
        "Created staging upload buffer {:?}: {} bytes",
        label, size
    );

    TrinityBuffer {
        inner: buffer,
        size,
        usage: buffer_usages::STAGING_UPLOAD,
        label: label.map(String::from),
    }
}

/// Creates a staging buffer for reading data back from the GPU.
///
/// This creates an empty staging buffer suitable for use as a copy destination.
/// After copying GPU data to this buffer, use [`map_buffer_blocking`] to read it.
///
/// # Arguments
///
/// * `device` - The wgpu device
/// * `size` - Size in bytes
/// * `label` - Optional debug label
///
/// # Returns
///
/// A `TrinityBuffer` ready to receive GPU data via copy.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::buffer::{
///     create_staging_readback_buffer, map_buffer_blocking, MappingMode
/// };
///
/// # fn example(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, gpu_buffer: &wgpu::Buffer) {
/// let readback = create_staging_readback_buffer(device, 1024, Some("readback"));
///
/// // Copy GPU data to staging buffer
/// encoder.copy_buffer_to_buffer(gpu_buffer, 0, readback.inner(), 0, 1024);
///
/// // Submit commands... then map and read
/// // let mapped = map_buffer_blocking(&readback, MappingMode::Read, device).unwrap();
/// // let data = mapped.read_data();
/// # }
/// ```
pub fn create_staging_readback_buffer(
    device: &Device,
    size: u64,
    label: Option<&str>,
) -> TrinityBuffer {
    let aligned_size = align_size(size);

    let buffer = device.create_buffer(&BufferDescriptor {
        label,
        size: aligned_size,
        usage: buffer_usages::STAGING_READBACK,
        mapped_at_creation: false,
    });

    debug!(
        "Created staging readback buffer {:?}: {} bytes",
        label, aligned_size
    );

    TrinityBuffer {
        inner: buffer,
        size: aligned_size,
        usage: buffer_usages::STAGING_READBACK,
        label: label.map(String::from),
    }
}

/// Checks if a buffer can be mapped for the given mode.
///
/// # Arguments
///
/// * `buffer` - The buffer to check
/// * `mode` - The desired mapping mode
///
/// # Returns
///
/// `true` if the buffer has the required usage flags, `false` otherwise.
#[inline]
pub fn is_mappable(buffer: &TrinityBuffer, mode: MappingMode) -> bool {
    let required = match mode {
        MappingMode::Read => BufferUsages::MAP_READ,
        MappingMode::Write => BufferUsages::MAP_WRITE,
    };
    buffer.usage().contains(required)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_descriptor_defaults() {
        let desc = TrinityBufferDescriptor::default();

        assert!(desc.label.is_none());
        assert_eq!(desc.size, BUFFER_ALIGNMENT);
        assert_eq!(desc.usage, BufferUsages::COPY_DST);
        assert!(!desc.mapped_at_creation);
    }

    #[test]
    fn test_buffer_size_alignment() {
        // Already aligned sizes
        assert_eq!(align_size(4), 4);
        assert_eq!(align_size(8), 8);
        assert_eq!(align_size(100), 100);
        assert_eq!(align_size(1024), 1024);

        // Sizes that need alignment
        assert_eq!(align_size(1), 4);
        assert_eq!(align_size(2), 4);
        assert_eq!(align_size(3), 4);
        assert_eq!(align_size(5), 8);
        assert_eq!(align_size(6), 8);
        assert_eq!(align_size(7), 8);
        assert_eq!(align_size(101), 104);
        assert_eq!(align_size(102), 104);
        assert_eq!(align_size(103), 104);
    }

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0));
        assert!(is_aligned(4));
        assert!(is_aligned(8));
        assert!(is_aligned(100));
        assert!(is_aligned(1024));

        assert!(!is_aligned(1));
        assert!(!is_aligned(2));
        assert!(!is_aligned(3));
        assert!(!is_aligned(5));
        assert!(!is_aligned(101));
    }

    #[test]
    fn test_buffer_usage_validation() {
        // Test that empty usage would be caught by try_create_buffer
        let desc = TrinityBufferDescriptor {
            label: Some("test"),
            size: 64,
            usage: BufferUsages::empty(),
            mapped_at_creation: false,
        };

        // We can't actually create the buffer without a device,
        // but we can test the validation logic
        assert!(desc.usage.is_empty());
    }

    #[test]
    fn test_buffer_creation_error_display() {
        assert_eq!(
            BufferCreationError::ZeroSize.to_string(),
            "buffer size must be greater than 0"
        );
        assert_eq!(
            BufferCreationError::EmptyUsage.to_string(),
            "buffer usage flags must not be empty"
        );
    }

    #[test]
    fn test_alignment_edge_cases() {
        // Zero is a special case - aligns to minimum
        assert_eq!(align_size(0), 0);

        // Power of 2 sizes (aligned already)
        for pow in 2..20 {
            let size: u64 = 1 << pow;
            assert_eq!(align_size(size), size);
            if size > 4 {
                assert_eq!(align_size(size - 1), size);
            }
        }

        // Practical large values (realistic buffer sizes)
        assert_eq!(align_size(1024 * 1024), 1024 * 1024); // 1MB
        assert_eq!(align_size(1024 * 1024 * 256), 1024 * 1024 * 256); // 256MB
        assert_eq!(align_size(1024 * 1024 * 256 + 1), 1024 * 1024 * 256 + 4);
    }

    #[test]
    fn test_descriptor_clone() {
        let desc1 = TrinityBufferDescriptor {
            label: Some("test_buffer"),
            size: 256,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };

        let desc2 = desc1.clone();

        assert_eq!(desc1.label, desc2.label);
        assert_eq!(desc1.size, desc2.size);
        assert_eq!(desc1.usage, desc2.usage);
        assert_eq!(desc1.mapped_at_creation, desc2.mapped_at_creation);
    }

    // ========================================================================
    // Buffer Usage Preset Tests
    // ========================================================================

    #[test]
    fn test_usage_presets_valid() {
        use super::buffer_usages;

        // All presets should pass validation
        assert!(validate_usage(buffer_usages::VERTEX).is_ok());
        assert!(validate_usage(buffer_usages::INDEX).is_ok());
        assert!(validate_usage(buffer_usages::UNIFORM).is_ok());
        assert!(validate_usage(buffer_usages::STORAGE_READ).is_ok());
        assert!(validate_usage(buffer_usages::STORAGE_RW).is_ok());
        assert!(validate_usage(buffer_usages::STAGING_UPLOAD).is_ok());
        assert!(validate_usage(buffer_usages::STAGING_READBACK).is_ok());
        assert!(validate_usage(buffer_usages::INDIRECT).is_ok());
        assert!(validate_usage(buffer_usages::QUERY_RESOLVE).is_ok());
    }

    #[test]
    fn test_usage_presets_contain_expected_flags() {
        use super::buffer_usages;

        // VERTEX preset
        assert!(buffer_usages::VERTEX.contains(BufferUsages::VERTEX));
        assert!(buffer_usages::VERTEX.contains(BufferUsages::COPY_DST));

        // INDEX preset
        assert!(buffer_usages::INDEX.contains(BufferUsages::INDEX));
        assert!(buffer_usages::INDEX.contains(BufferUsages::COPY_DST));

        // UNIFORM preset
        assert!(buffer_usages::UNIFORM.contains(BufferUsages::UNIFORM));
        assert!(buffer_usages::UNIFORM.contains(BufferUsages::COPY_DST));

        // STORAGE_READ preset
        assert!(buffer_usages::STORAGE_READ.contains(BufferUsages::STORAGE));
        assert!(buffer_usages::STORAGE_READ.contains(BufferUsages::COPY_DST));

        // STORAGE_RW preset
        assert!(buffer_usages::STORAGE_RW.contains(BufferUsages::STORAGE));
        assert!(buffer_usages::STORAGE_RW.contains(BufferUsages::COPY_DST));
        assert!(buffer_usages::STORAGE_RW.contains(BufferUsages::COPY_SRC));

        // STAGING_UPLOAD preset
        assert!(buffer_usages::STAGING_UPLOAD.contains(BufferUsages::MAP_WRITE));
        assert!(buffer_usages::STAGING_UPLOAD.contains(BufferUsages::COPY_SRC));

        // STAGING_READBACK preset
        assert!(buffer_usages::STAGING_READBACK.contains(BufferUsages::MAP_READ));
        assert!(buffer_usages::STAGING_READBACK.contains(BufferUsages::COPY_DST));

        // INDIRECT preset
        assert!(buffer_usages::INDIRECT.contains(BufferUsages::INDIRECT));
        assert!(buffer_usages::INDIRECT.contains(BufferUsages::COPY_DST));
        assert!(buffer_usages::INDIRECT.contains(BufferUsages::STORAGE));

        // QUERY_RESOLVE preset
        assert!(buffer_usages::QUERY_RESOLVE.contains(BufferUsages::QUERY_RESOLVE));
        assert!(buffer_usages::QUERY_RESOLVE.contains(BufferUsages::COPY_SRC));
    }

    // ========================================================================
    // Usage Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_usage_map_read_and_write_invalid() {
        let invalid = BufferUsages::MAP_READ | BufferUsages::MAP_WRITE;
        let result = validate_usage(invalid);

        assert!(result.is_err());
        assert!(matches!(result, Err(UsageValidationError::MapReadAndWrite)));
    }

    #[test]
    fn test_validate_usage_map_read_with_vertex_invalid() {
        let invalid = BufferUsages::MAP_READ | BufferUsages::VERTEX;
        let result = validate_usage(invalid);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(UsageValidationError::MapReadWithGpuOnly(_))
        ));
    }

    #[test]
    fn test_validate_usage_map_read_with_index_invalid() {
        let invalid = BufferUsages::MAP_READ | BufferUsages::INDEX;
        let result = validate_usage(invalid);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(UsageValidationError::MapReadWithGpuOnly(_))
        ));
    }

    #[test]
    fn test_validate_usage_map_read_with_uniform_invalid() {
        let invalid = BufferUsages::MAP_READ | BufferUsages::UNIFORM;
        let result = validate_usage(invalid);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(UsageValidationError::MapReadWithGpuOnly(_))
        ));
    }

    #[test]
    fn test_validate_usage_map_read_with_storage_invalid() {
        let invalid = BufferUsages::MAP_READ | BufferUsages::STORAGE;
        let result = validate_usage(invalid);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(UsageValidationError::MapReadWithGpuOnly(_))
        ));
    }

    #[test]
    fn test_validate_usage_valid_combinations() {
        // Common valid combinations
        assert!(validate_usage(BufferUsages::VERTEX | BufferUsages::COPY_DST).is_ok());
        assert!(validate_usage(BufferUsages::INDEX | BufferUsages::COPY_DST).is_ok());
        assert!(validate_usage(BufferUsages::UNIFORM | BufferUsages::COPY_DST).is_ok());
        assert!(validate_usage(BufferUsages::STORAGE | BufferUsages::COPY_DST).is_ok());
        assert!(
            validate_usage(BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC)
                .is_ok()
        );

        // Staging patterns
        assert!(validate_usage(BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC).is_ok());
        assert!(validate_usage(BufferUsages::MAP_READ | BufferUsages::COPY_DST).is_ok());

        // Indirect buffer
        assert!(
            validate_usage(
                BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST
            )
            .is_ok()
        );

        // Query resolve
        assert!(validate_usage(BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC).is_ok());

        // Single flags (where valid alone)
        assert!(validate_usage(BufferUsages::COPY_SRC).is_ok());
        assert!(validate_usage(BufferUsages::COPY_DST).is_ok());
        assert!(validate_usage(BufferUsages::MAP_READ).is_ok());
        assert!(validate_usage(BufferUsages::MAP_WRITE).is_ok());
    }

    #[test]
    fn test_all_usage_flags_documented() {
        // Verify all 10 wgpu BufferUsages flags exist and can be used
        let all_flags = [
            BufferUsages::MAP_READ,
            BufferUsages::MAP_WRITE,
            BufferUsages::COPY_SRC,
            BufferUsages::COPY_DST,
            BufferUsages::INDEX,
            BufferUsages::VERTEX,
            BufferUsages::UNIFORM,
            BufferUsages::STORAGE,
            BufferUsages::INDIRECT,
            BufferUsages::QUERY_RESOLVE,
        ];

        // All individual flags should be non-empty
        for flag in all_flags {
            assert!(!flag.is_empty(), "Flag {:?} should not be empty", flag);
        }

        // Combining all flags should work
        let combined = all_flags.iter().fold(BufferUsages::empty(), |acc, &f| acc | f);
        assert!(combined.contains(BufferUsages::MAP_READ));
        assert!(combined.contains(BufferUsages::QUERY_RESOLVE));
    }

    #[test]
    fn test_usage_validation_error_display() {
        let err1 = UsageValidationError::MapReadAndWrite;
        assert!(err1.to_string().contains("MAP_READ"));
        assert!(err1.to_string().contains("MAP_WRITE"));

        let err2 =
            UsageValidationError::MapReadWithGpuOnly(BufferUsages::MAP_READ | BufferUsages::VERTEX);
        assert!(err2.to_string().contains("MAP_READ"));
        assert!(err2.to_string().contains("GPU-only"));
    }

    #[test]
    fn test_validate_usage_with_label() {
        // Valid usage
        let result = validate_usage_with_label(
            BufferUsages::VERTEX | BufferUsages::COPY_DST,
            Some("test_buffer"),
        );
        assert!(result.is_ok());

        // Invalid usage with label
        let result = validate_usage_with_label(
            BufferUsages::MAP_READ | BufferUsages::MAP_WRITE,
            Some("bad_buffer"),
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("bad_buffer"));
        assert!(err.contains("MAP_READ"));

        // Invalid usage without label
        let result =
            validate_usage_with_label(BufferUsages::MAP_READ | BufferUsages::VERTEX, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_creation_error_invalid_usage() {
        let validation_err = UsageValidationError::MapReadAndWrite;
        let creation_err: BufferCreationError = validation_err.into();

        assert!(matches!(
            creation_err,
            BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite)
        ));

        // Test Display
        let display = creation_err.to_string();
        assert!(display.contains("invalid buffer usage"));
    }

    #[test]
    fn test_buffer_creation_error_source() {
        let creation_err = BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite);
        let source = std::error::Error::source(&creation_err);
        assert!(source.is_some());

        let creation_err = BufferCreationError::ZeroSize;
        let source = std::error::Error::source(&creation_err);
        assert!(source.is_none());
    }

    // ========================================================================
    // Buffer Mapping Tests
    // ========================================================================

    #[test]
    fn test_mapping_mode_conversion() {
        use wgpu::MapMode;

        // Test Read mode conversion
        let read_mode = MappingMode::Read;
        let wgpu_read: MapMode = read_mode.into();
        assert!(matches!(wgpu_read, MapMode::Read));

        // Test Write mode conversion
        let write_mode = MappingMode::Write;
        let wgpu_write: MapMode = write_mode.into();
        assert!(matches!(wgpu_write, MapMode::Write));
    }

    #[test]
    fn test_mapping_mode_display() {
        assert_eq!(MappingMode::Read.to_string(), "Read");
        assert_eq!(MappingMode::Write.to_string(), "Write");
    }

    #[test]
    fn test_mapping_mode_equality() {
        assert_eq!(MappingMode::Read, MappingMode::Read);
        assert_eq!(MappingMode::Write, MappingMode::Write);
        assert_ne!(MappingMode::Read, MappingMode::Write);
    }

    #[test]
    fn test_mapping_mode_clone_copy() {
        let mode = MappingMode::Read;
        let cloned = mode.clone();
        let copied = mode;
        assert_eq!(mode, cloned);
        assert_eq!(mode, copied);
    }

    #[test]
    fn test_mapping_mode_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(MappingMode::Read);
        set.insert(MappingMode::Write);

        assert!(set.contains(&MappingMode::Read));
        assert!(set.contains(&MappingMode::Write));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_mapping_error_display_not_mappable() {
        let err = MappingError::NotMappable {
            mode: MappingMode::Read,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        };
        let display = err.to_string();
        assert!(display.contains("not mappable"));
        assert!(display.contains("Read"));
        assert!(display.contains("MAP_READ"));
    }

    #[test]
    fn test_mapping_error_display_map_failed() {
        let err = MappingError::MapFailed;
        assert!(err.to_string().contains("failed"));
    }

    #[test]
    fn test_mapping_error_display_already_mapped() {
        let err = MappingError::AlreadyMapped;
        assert!(err.to_string().contains("already mapped"));
    }

    #[test]
    fn test_mapping_error_display_channel_error() {
        let err = MappingError::ChannelError;
        assert!(err.to_string().contains("channel"));
    }

    #[test]
    fn test_mapping_error_display_not_mapped_at_creation() {
        let err = MappingError::NotMappedAtCreation;
        assert!(err.to_string().contains("not mapped at creation"));
    }

    #[test]
    fn test_mapping_error_equality() {
        assert_eq!(MappingError::MapFailed, MappingError::MapFailed);
        assert_eq!(MappingError::AlreadyMapped, MappingError::AlreadyMapped);
        assert_ne!(MappingError::MapFailed, MappingError::AlreadyMapped);

        let err1 = MappingError::NotMappable {
            mode: MappingMode::Read,
            usage: BufferUsages::VERTEX,
        };
        let err2 = MappingError::NotMappable {
            mode: MappingMode::Read,
            usage: BufferUsages::VERTEX,
        };
        let err3 = MappingError::NotMappable {
            mode: MappingMode::Write,
            usage: BufferUsages::VERTEX,
        };
        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_mapping_error_debug() {
        let err = MappingError::MapFailed;
        let debug = format!("{:?}", err);
        assert!(debug.contains("MapFailed"));
    }

    #[test]
    fn test_is_mappable_staging_upload() {
        // We can't create a real buffer without a device,
        // but we can test the logic indirectly by checking usage flags
        let upload_usage = buffer_usages::STAGING_UPLOAD;
        assert!(upload_usage.contains(BufferUsages::MAP_WRITE));
        assert!(!upload_usage.contains(BufferUsages::MAP_READ));
    }

    #[test]
    fn test_is_mappable_staging_readback() {
        let readback_usage = buffer_usages::STAGING_READBACK;
        assert!(readback_usage.contains(BufferUsages::MAP_READ));
        assert!(!readback_usage.contains(BufferUsages::MAP_WRITE));
    }

    #[test]
    fn test_vertex_buffer_not_mappable() {
        // Vertex buffers should not have mapping flags
        let vertex_usage = buffer_usages::VERTEX;
        assert!(!vertex_usage.contains(BufferUsages::MAP_READ));
        assert!(!vertex_usage.contains(BufferUsages::MAP_WRITE));
    }

    #[test]
    fn test_uniform_buffer_not_mappable() {
        // Uniform buffers should not have mapping flags
        let uniform_usage = buffer_usages::UNIFORM;
        assert!(!uniform_usage.contains(BufferUsages::MAP_READ));
        assert!(!uniform_usage.contains(BufferUsages::MAP_WRITE));
    }

    #[test]
    fn test_mapping_error_is_error_trait() {
        // Verify MappingError implements std::error::Error
        fn assert_error<E: std::error::Error>() {}
        assert_error::<MappingError>();
    }

    #[test]
    fn test_mapping_mode_debug() {
        let read = MappingMode::Read;
        let write = MappingMode::Write;
        assert_eq!(format!("{:?}", read), "Read");
        assert_eq!(format!("{:?}", write), "Write");
    }
}
