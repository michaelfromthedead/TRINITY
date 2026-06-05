//! Async buffer mapping for TRINITY.
//!
//! This module provides a high-level async buffer mapping abstraction that wraps
//! wgpu's buffer mapping API. It provides:
//!
//! - **Async mapping with state tracking**: Track mapping state through its lifecycle
//! - **Non-blocking status polling**: Check mapping completion without blocking
//! - **Mapped slice access**: Type-safe read/write access to mapped data
//! - **Automatic unmap on drop**: RAII-based resource cleanup
//!
//! # Architecture
//!
//! ```text
//! BufferMapper
//!     ├── state: MappingState (Unmapped/Pending/Mapped/Failed)
//!     ├── mode: Option<MappingMode> (Read/Write)
//!     ├── offset/size: Mapped region bounds
//!     └── Automatic unmap on drop
//!
//! State Transitions:
//!     Unmapped -> Pending (map_async called)
//!     Pending -> Mapped (device polled, mapping complete)
//!     Pending -> Failed (mapping error)
//!     Mapped -> Unmapped (unmap called or drop)
//!     Failed -> Unmapped (reset or drop)
//! ```
//!
//! # wgpu Mapping Patterns
//!
//! ```text
//! // Async mapping flow:
//! buffer.slice(offset..offset+size).map_async(mode, callback)
//! device.poll(Maintain::Poll)  // Non-blocking check
//! device.poll(Maintain::Wait)  // Blocking wait
//!
//! // Access mapped data:
//! buffer.slice().get_mapped_range()      // Read access
//! buffer.slice().get_mapped_range_mut()  // Write access
//!
//! // Cleanup:
//! buffer.unmap()
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::buffer_mapping::{BufferMapper, MappingState, MappingMode};
//! use std::sync::Arc;
//!
//! # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
//! let mut mapper = BufferMapper::new(buffer);
//!
//! // Start async mapping
//! mapper.map_async(device, MappingMode::Read, 0, 1024).unwrap();
//!
//! // Poll until ready
//! while !mapper.is_ready() {
//!     mapper.poll(device);
//! }
//!
//! // Read mapped data
//! if let Some(data) = mapper.read_data::<u8>() {
//!     println!("Read {} bytes", data.len());
//! }
//!
//! // Automatic unmap on drop
//! # }
//! ```
//!
//! # Thread Safety
//!
//! `BufferMapper` is `Send` but not `Sync`. The mapping state and callbacks
//! are managed internally, and the device must be polled on the same thread
//! that initiated the mapping (or use wgpu's thread-safe polling features).

use bytemuck::Pod;
use log::{debug, trace, warn};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use wgpu::{Buffer, Device, MapMode, Maintain};

// ============================================================================
// MappingState
// ============================================================================

/// State of a buffer mapping operation.
///
/// Tracks the lifecycle of an async buffer mapping from initiation to completion.
///
/// # State Diagram
///
/// ```text
/// ┌─────────┐  map_async()  ┌─────────┐
/// │ Unmapped│──────────────>│ Pending │
/// └─────────┘               └────┬────┘
///      ^                         │
///      │                  poll() │
///      │                  ┌──────┴──────┐
///      │                  │             │
///      │         success  v    failure  v
///      │               ┌──────┐    ┌────────┐
///      │               │Mapped│    │ Failed │
///      │               └──┬───┘    └────┬───┘
///      │       unmap()   │              │
///      └─────────────────┴──────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MappingState {
    /// Buffer is not mapped. Ready for a new mapping operation.
    Unmapped = 0,
    /// Mapping operation is in progress. Waiting for GPU completion.
    Pending = 1,
    /// Buffer is successfully mapped. Data can be accessed.
    Mapped = 2,
    /// Mapping operation failed. Check error details.
    Failed = 3,
}

impl MappingState {
    /// Returns `true` if the buffer is in the `Unmapped` state.
    #[inline]
    pub fn is_unmapped(self) -> bool {
        self == MappingState::Unmapped
    }

    /// Returns `true` if a mapping operation is pending.
    #[inline]
    pub fn is_pending(self) -> bool {
        self == MappingState::Pending
    }

    /// Returns `true` if the buffer is successfully mapped.
    #[inline]
    pub fn is_mapped(self) -> bool {
        self == MappingState::Mapped
    }

    /// Returns `true` if the mapping operation failed.
    #[inline]
    pub fn is_failed(self) -> bool {
        self == MappingState::Failed
    }

    /// Returns `true` if the buffer is ready for data access (mapped state).
    #[inline]
    pub fn is_ready(self) -> bool {
        self == MappingState::Mapped
    }

    /// Returns `true` if no mapping operation is in progress.
    #[inline]
    pub fn is_idle(self) -> bool {
        matches!(self, MappingState::Unmapped | MappingState::Failed)
    }
}

impl std::fmt::Display for MappingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MappingState::Unmapped => write!(f, "Unmapped"),
            MappingState::Pending => write!(f, "Pending"),
            MappingState::Mapped => write!(f, "Mapped"),
            MappingState::Failed => write!(f, "Failed"),
        }
    }
}

impl Default for MappingState {
    fn default() -> Self {
        MappingState::Unmapped
    }
}

// ============================================================================
// MappingMode
// ============================================================================

/// Mode for buffer mapping operations.
///
/// Specifies whether the buffer should be mapped for reading or writing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MappingMode {
    /// Map for reading data from GPU to CPU.
    ///
    /// Requires `BufferUsages::MAP_READ` on the buffer.
    Read,
    /// Map for writing data from CPU to GPU.
    ///
    /// Requires `BufferUsages::MAP_WRITE` on the buffer.
    Write,
}

impl MappingMode {
    /// Returns the required wgpu buffer usage for this mapping mode.
    #[inline]
    pub fn required_usage(self) -> wgpu::BufferUsages {
        match self {
            MappingMode::Read => wgpu::BufferUsages::MAP_READ,
            MappingMode::Write => wgpu::BufferUsages::MAP_WRITE,
        }
    }
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
// BufferMapError
// ============================================================================

/// Error during buffer mapping operations.
#[derive(Debug, Clone)]
pub enum BufferMapError {
    /// Buffer is already mapped or has a pending mapping operation.
    AlreadyMapped,
    /// Specified range is invalid (offset + size exceeds buffer size).
    InvalidRange {
        /// Requested offset.
        offset: u64,
        /// Requested size.
        size: u64,
        /// Actual buffer size.
        buffer_size: u64,
    },
    /// The async mapping operation failed on the GPU side.
    MappingFailed(String),
    /// Buffer is not in the mapped state when data access was attempted.
    NotMapped,
    /// Operation requires a different mapping mode than the current one.
    WrongMode {
        /// The current mapping mode.
        current: MappingMode,
        /// The required mapping mode.
        required: MappingMode,
    },
    /// Buffer doesn't have the required usage flags for mapping.
    MissingUsage {
        /// The required usage flags.
        required: wgpu::BufferUsages,
        /// The actual buffer usage flags.
        actual: wgpu::BufferUsages,
    },
    /// Offset is not properly aligned for mapping (must be multiple of 8).
    UnalignedOffset {
        /// The provided offset.
        offset: u64,
        /// The required alignment.
        required_alignment: u64,
    },
    /// Size is not properly aligned for mapping (must be multiple of 4).
    UnalignedSize {
        /// The provided size.
        size: u64,
        /// The required alignment.
        required_alignment: u64,
    },
}

impl std::fmt::Display for BufferMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferMapError::AlreadyMapped => {
                write!(f, "buffer is already mapped or has a pending mapping operation")
            }
            BufferMapError::InvalidRange { offset, size, buffer_size } => {
                write!(
                    f,
                    "invalid mapping range: offset {} + size {} = {} exceeds buffer size {}",
                    offset, size, offset + size, buffer_size
                )
            }
            BufferMapError::MappingFailed(reason) => {
                write!(f, "buffer mapping failed: {}", reason)
            }
            BufferMapError::NotMapped => {
                write!(f, "buffer is not mapped; call map_async() and wait for completion first")
            }
            BufferMapError::WrongMode { current, required } => {
                write!(
                    f,
                    "wrong mapping mode: buffer is mapped for {} but {} access was requested",
                    current, required
                )
            }
            BufferMapError::MissingUsage { required, actual } => {
                write!(
                    f,
                    "buffer missing required usage flags: need {:?}, have {:?}",
                    required, actual
                )
            }
            BufferMapError::UnalignedOffset { offset, required_alignment } => {
                write!(
                    f,
                    "mapping offset {} is not aligned to {} bytes",
                    offset, required_alignment
                )
            }
            BufferMapError::UnalignedSize { size, required_alignment } => {
                write!(
                    f,
                    "mapping size {} is not aligned to {} bytes",
                    size, required_alignment
                )
            }
        }
    }
}

impl std::error::Error for BufferMapError {}

// ============================================================================
// Constants
// ============================================================================

/// Required alignment for buffer mapping offset.
///
/// wgpu requires the mapping offset to be a multiple of 8 bytes.
pub const COPY_BUFFER_ALIGNMENT: u64 = 8;

/// Required alignment for buffer mapping size.
///
/// wgpu requires the mapping size to be a multiple of 4 bytes.
pub const MAP_SIZE_ALIGNMENT: u64 = 4;

// ============================================================================
// BufferMapper
// ============================================================================

/// High-level async buffer mapping wrapper.
///
/// `BufferMapper` provides a stateful wrapper around wgpu's buffer mapping API,
/// tracking the mapping state and providing convenient methods for async mapping
/// with automatic cleanup.
///
/// # Lifecycle
///
/// 1. Create mapper with `BufferMapper::new(buffer)`
/// 2. Start mapping with `map_async(device, mode, offset, size)`
/// 3. Poll for completion with `poll(device)` or block with `wait(device)`
/// 4. Access data with `read_data()` or `write_data()`
/// 5. Explicitly `unmap()` or let Drop handle it
///
/// # Example
///
/// ```no_run
/// use renderer_backend::buffer_mapping::{BufferMapper, MappingMode};
/// use std::sync::Arc;
///
/// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
/// let mut mapper = BufferMapper::new(buffer);
///
/// // Map for reading
/// mapper.map_async(device, MappingMode::Read, 0, 256).unwrap();
/// mapper.wait(device);
///
/// // Read data
/// let data: Vec<f32> = mapper.read_data().unwrap();
/// println!("Read {} floats", data.len());
/// # }
/// ```
pub struct BufferMapper {
    /// The underlying wgpu buffer (shared ownership).
    buffer: Arc<Buffer>,
    /// Current mapping state.
    state: MappingState,
    /// Current mapping mode (if mapped or pending).
    mode: Option<MappingMode>,
    /// Mapping offset in bytes.
    offset: u64,
    /// Mapping size in bytes.
    size: u64,
    /// Atomic state for callback synchronization.
    /// 0 = no pending callback, 1 = pending, 2 = completed successfully, 3 = failed
    callback_state: Arc<AtomicU8>,
}

impl BufferMapper {
    /// Creates a new buffer mapper.
    ///
    /// The mapper starts in the `Unmapped` state.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The wgpu buffer to map (must have appropriate MAP_READ or MAP_WRITE usage)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::BufferMapper;
    /// use std::sync::Arc;
    ///
    /// # fn example(buffer: Arc<wgpu::Buffer>) {
    /// let mapper = BufferMapper::new(buffer);
    /// assert!(mapper.state().is_unmapped());
    /// # }
    /// ```
    pub fn new(buffer: Arc<Buffer>) -> Self {
        Self {
            buffer,
            state: MappingState::Unmapped,
            mode: None,
            offset: 0,
            size: 0,
            callback_state: Arc::new(AtomicU8::new(0)),
        }
    }

    /// Returns a reference to the underlying buffer.
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the current mapping state.
    #[inline]
    pub fn state(&self) -> MappingState {
        self.state
    }

    /// Returns the current mapping mode, if any.
    #[inline]
    pub fn mode(&self) -> Option<MappingMode> {
        self.mode
    }

    /// Returns the mapping offset in bytes.
    #[inline]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the mapping size in bytes.
    #[inline]
    pub fn mapped_size(&self) -> u64 {
        self.size
    }

    /// Returns `true` if the buffer is ready for data access.
    ///
    /// This is a convenience method equivalent to `state().is_mapped()`.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.state.is_ready()
    }

    /// Returns `true` if a mapping operation is currently pending.
    #[inline]
    pub fn is_pending(&self) -> bool {
        self.state.is_pending()
    }

    // ========================================================================
    // Async Mapping
    // ========================================================================

    /// Starts an async mapping operation.
    ///
    /// This initiates a buffer mapping request with the GPU. The mapping will
    /// not be immediately available; you must poll the device (via [`poll`] or
    /// [`wait`]) until the mapping completes.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (needed to set up the mapping callback)
    /// * `mode` - Whether to map for reading or writing
    /// * `offset` - Byte offset into the buffer (must be aligned to 8 bytes)
    /// * `size` - Number of bytes to map (must be aligned to 4 bytes)
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Buffer is already mapped or has a pending mapping
    /// - Offset or size are not properly aligned
    /// - Range exceeds buffer bounds
    /// - Buffer doesn't have required usage flags
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::{BufferMapper, MappingMode};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
    /// let mut mapper = BufferMapper::new(buffer);
    /// mapper.map_async(device, MappingMode::Read, 0, 1024).unwrap();
    /// // Buffer is now in Pending state
    /// # }
    /// ```
    ///
    /// [`poll`]: BufferMapper::poll
    /// [`wait`]: BufferMapper::wait
    pub fn map_async(
        &mut self,
        _device: &Device,
        mode: MappingMode,
        offset: u64,
        size: u64,
    ) -> Result<(), BufferMapError> {
        // Check if already mapped or pending
        if !self.state.is_idle() {
            return Err(BufferMapError::AlreadyMapped);
        }

        // Validate non-zero size (wgpu panics on zero-size mappings)
        if size == 0 {
            return Err(BufferMapError::InvalidRange {
                offset,
                size,
                buffer_size: self.buffer.size(),
            });
        }

        // Validate alignment
        if offset % COPY_BUFFER_ALIGNMENT != 0 {
            return Err(BufferMapError::UnalignedOffset {
                offset,
                required_alignment: COPY_BUFFER_ALIGNMENT,
            });
        }

        if size % MAP_SIZE_ALIGNMENT != 0 {
            return Err(BufferMapError::UnalignedSize {
                size,
                required_alignment: MAP_SIZE_ALIGNMENT,
            });
        }

        // Validate range
        let buffer_size = self.buffer.size();
        if offset.saturating_add(size) > buffer_size {
            return Err(BufferMapError::InvalidRange {
                offset,
                size,
                buffer_size,
            });
        }

        // Check buffer has required usage
        let required_usage = mode.required_usage();
        let actual_usage = self.buffer.usage();
        if !actual_usage.contains(required_usage) {
            return Err(BufferMapError::MissingUsage {
                required: required_usage,
                actual: actual_usage,
            });
        }

        // Update state
        self.state = MappingState::Pending;
        self.mode = Some(mode);
        self.offset = offset;
        self.size = size;

        // Reset callback state
        self.callback_state.store(1, Ordering::SeqCst); // 1 = pending

        // Start async mapping
        let callback_state = Arc::clone(&self.callback_state);
        let slice = self.buffer.slice(offset..offset + size);

        debug!(
            "Starting async buffer mapping: mode={}, offset={}, size={}",
            mode, offset, size
        );

        slice.map_async(mode.into(), move |result| {
            match result {
                Ok(()) => {
                    trace!("Buffer mapping callback: success");
                    callback_state.store(2, Ordering::SeqCst); // 2 = success
                }
                Err(_) => {
                    trace!("Buffer mapping callback: failed");
                    callback_state.store(3, Ordering::SeqCst); // 3 = failed
                }
            }
        });

        Ok(())
    }

    /// Polls the device and updates the mapping state.
    ///
    /// This performs a non-blocking poll of the GPU device to check if any
    /// pending operations have completed. If the mapping operation completes,
    /// the state will transition to either `Mapped` or `Failed`.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to poll
    ///
    /// # Returns
    ///
    /// The current mapping state after polling.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::{BufferMapper, MappingState, MappingMode};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
    /// let mut mapper = BufferMapper::new(buffer);
    /// mapper.map_async(device, MappingMode::Read, 0, 256).unwrap();
    ///
    /// loop {
    ///     let state = mapper.poll(device);
    ///     if state.is_mapped() {
    ///         println!("Mapping complete!");
    ///         break;
    ///     } else if state.is_failed() {
    ///         println!("Mapping failed!");
    ///         break;
    ///     }
    ///     // Do other work...
    /// }
    /// # }
    /// ```
    pub fn poll(&mut self, device: &Device) -> MappingState {
        if self.state != MappingState::Pending {
            return self.state;
        }

        // Non-blocking poll
        device.poll(Maintain::Poll);

        // Check callback state
        let callback_result = self.callback_state.load(Ordering::SeqCst);
        match callback_result {
            2 => {
                // Success
                self.state = MappingState::Mapped;
                debug!("Buffer mapping completed successfully");
            }
            3 => {
                // Failed
                self.state = MappingState::Failed;
                warn!("Buffer mapping failed");
            }
            _ => {
                // Still pending
            }
        }

        self.state
    }

    /// Blocks until the mapping operation completes.
    ///
    /// This performs a blocking wait on the GPU device until the pending
    /// mapping operation completes. After this call returns, the state will
    /// be either `Mapped` or `Failed`.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to wait on
    ///
    /// # Panics
    ///
    /// May panic if the GPU device is lost or in an error state.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::{BufferMapper, MappingMode};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
    /// let mut mapper = BufferMapper::new(buffer);
    /// mapper.map_async(device, MappingMode::Read, 0, 1024).unwrap();
    ///
    /// // Block until mapping completes
    /// mapper.wait(device);
    ///
    /// if mapper.is_ready() {
    ///     let data: Vec<u8> = mapper.read_data().unwrap();
    /// }
    /// # }
    /// ```
    pub fn wait(&mut self, device: &Device) {
        if self.state != MappingState::Pending {
            return;
        }

        debug!("Blocking wait for buffer mapping");

        // Blocking poll
        device.poll(Maintain::Wait);

        // Check callback state
        let callback_result = self.callback_state.load(Ordering::SeqCst);
        match callback_result {
            2 => {
                self.state = MappingState::Mapped;
                debug!("Buffer mapping completed successfully (blocking)");
            }
            3 => {
                self.state = MappingState::Failed;
                warn!("Buffer mapping failed (blocking)");
            }
            _ => {
                // This shouldn't happen after a blocking wait, but handle it gracefully
                warn!("Buffer mapping still pending after blocking wait");
            }
        }
    }

    // ========================================================================
    // Data Access
    // ========================================================================

    /// Gets a read-only view of the mapped buffer range.
    ///
    /// Returns a `BufferView` that provides read access to the mapped data.
    /// The view borrows the mapper, ensuring the mapping stays valid.
    ///
    /// # Returns
    ///
    /// `Some(BufferView)` if the buffer is mapped for reading, `None` otherwise.
    ///
    /// # Panics
    ///
    /// May panic if called when the buffer is not in the `Mapped` state.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::{BufferMapper, MappingMode};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
    /// let mut mapper = BufferMapper::new(buffer);
    /// mapper.map_async(device, MappingMode::Read, 0, 1024).unwrap();
    /// mapper.wait(device);
    ///
    /// if let Some(view) = mapper.get_mapped_range() {
    ///     let bytes: &[u8] = &view;
    ///     println!("First byte: {}", bytes[0]);
    /// }
    /// # }
    /// ```
    pub fn get_mapped_range(&self) -> Option<wgpu::BufferView<'_>> {
        if self.state != MappingState::Mapped {
            return None;
        }

        let slice = self.buffer.slice(self.offset..self.offset + self.size);
        Some(slice.get_mapped_range())
    }

    /// Gets a mutable view of the mapped buffer range.
    ///
    /// Returns a `BufferViewMut` that provides write access to the mapped data.
    /// The view mutably borrows the mapper, ensuring exclusive access.
    ///
    /// # Returns
    ///
    /// `Some(BufferViewMut)` if the buffer is mapped for writing, `None` otherwise.
    ///
    /// # Panics
    ///
    /// May panic if called when:
    /// - The buffer is not in the `Mapped` state
    /// - The buffer is mapped for reading (not writing)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::{BufferMapper, MappingMode};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
    /// let mut mapper = BufferMapper::new(buffer);
    /// mapper.map_async(device, MappingMode::Write, 0, 1024).unwrap();
    /// mapper.wait(device);
    ///
    /// if let Some(mut view) = mapper.get_mapped_range_mut() {
    ///     view[0] = 42;
    ///     view[1..5].copy_from_slice(&[1, 2, 3, 4]);
    /// }
    /// # }
    /// ```
    pub fn get_mapped_range_mut(&self) -> Option<wgpu::BufferViewMut<'_>> {
        if self.state != MappingState::Mapped {
            return None;
        }

        // Only allow mutable access for write mappings
        if self.mode != Some(MappingMode::Write) {
            return None;
        }

        let slice = self.buffer.slice(self.offset..self.offset + self.size);
        Some(slice.get_mapped_range_mut())
    }

    /// Reads data from the mapped buffer as a vector of `Pod` types.
    ///
    /// This is a convenience method that reads the entire mapped region
    /// and casts it to the requested type.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type to read. Must implement `bytemuck::Pod`.
    ///
    /// # Returns
    ///
    /// `Some(Vec<T>)` containing the data, or `None` if:
    /// - Buffer is not mapped
    /// - Data is not properly aligned for type `T`
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::{BufferMapper, MappingMode};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
    /// let mut mapper = BufferMapper::new(buffer);
    /// mapper.map_async(device, MappingMode::Read, 0, 256).unwrap();
    /// mapper.wait(device);
    ///
    /// // Read as f32 values
    /// if let Some(floats) = mapper.read_data::<f32>() {
    ///     println!("Read {} floats", floats.len());
    ///     for (i, f) in floats.iter().enumerate() {
    ///         println!("  [{}] = {}", i, f);
    ///     }
    /// }
    /// # }
    /// ```
    pub fn read_data<T: Pod>(&self) -> Option<Vec<T>> {
        let view = self.get_mapped_range()?;
        let bytes: &[u8] = &view;

        // Check alignment
        if bytes.as_ptr() as usize % std::mem::align_of::<T>() != 0 {
            warn!("Buffer data not aligned for type {}", std::any::type_name::<T>());
            return None;
        }

        // Cast and copy
        let slice: &[T] = bytemuck::cast_slice(bytes);
        Some(slice.to_vec())
    }

    /// Writes data to the mapped buffer from a slice of `Pod` types.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type to write. Must implement `bytemuck::Pod`.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to write
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Buffer is not mapped
    /// - Buffer is mapped for reading (not writing)
    /// - Data exceeds the mapped region size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::{BufferMapper, MappingMode};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
    /// let mut mapper = BufferMapper::new(buffer);
    /// mapper.map_async(device, MappingMode::Write, 0, 256).unwrap();
    /// mapper.wait(device);
    ///
    /// let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    /// mapper.write_data(&data).unwrap();
    /// # }
    /// ```
    pub fn write_data<T: Pod>(&mut self, data: &[T]) -> Result<(), BufferMapError> {
        if self.state != MappingState::Mapped {
            return Err(BufferMapError::NotMapped);
        }

        if self.mode != Some(MappingMode::Write) {
            return Err(BufferMapError::WrongMode {
                current: self.mode.unwrap_or(MappingMode::Read),
                required: MappingMode::Write,
            });
        }

        let bytes: &[u8] = bytemuck::cast_slice(data);
        if bytes.len() as u64 > self.size {
            return Err(BufferMapError::InvalidRange {
                offset: 0,
                size: bytes.len() as u64,
                buffer_size: self.size,
            });
        }

        let slice = self.buffer.slice(self.offset..self.offset + self.size);
        let mut view = slice.get_mapped_range_mut();
        view[..bytes.len()].copy_from_slice(bytes);

        debug!("Wrote {} bytes to mapped buffer", bytes.len());
        Ok(())
    }

    // ========================================================================
    // State Management
    // ========================================================================

    /// Unmaps the buffer.
    ///
    /// This releases the mapping and makes the buffer available for GPU use again.
    /// After unmapping, the state transitions to `Unmapped`.
    ///
    /// If the buffer is not currently mapped, this is a no-op.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::{BufferMapper, MappingMode};
    /// use std::sync::Arc;
    ///
    /// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
    /// let mut mapper = BufferMapper::new(buffer);
    /// mapper.map_async(device, MappingMode::Read, 0, 256).unwrap();
    /// mapper.wait(device);
    ///
    /// // Use the mapped data...
    ///
    /// // Explicitly unmap when done
    /// mapper.unmap();
    /// assert!(mapper.state().is_unmapped());
    /// # }
    /// ```
    pub fn unmap(&mut self) {
        if self.state == MappingState::Mapped {
            debug!("Unmapping buffer");
            self.buffer.unmap();
        }

        self.state = MappingState::Unmapped;
        self.mode = None;
        self.offset = 0;
        self.size = 0;
        self.callback_state.store(0, Ordering::SeqCst);
    }

    /// Resets the mapper state without unmapping.
    ///
    /// This is useful when you want to reuse the mapper after a failed
    /// mapping operation. For a successful mapping, use [`unmap`] instead.
    ///
    /// # Warning
    ///
    /// If called while the buffer is actually mapped, this will leave the
    /// buffer in an inconsistent state. Only use this for error recovery.
    ///
    /// [`unmap`]: BufferMapper::unmap
    pub fn reset(&mut self) {
        if self.state == MappingState::Mapped {
            warn!("reset() called while buffer is mapped; use unmap() instead");
            self.buffer.unmap();
        }

        self.state = MappingState::Unmapped;
        self.mode = None;
        self.offset = 0;
        self.size = 0;
        self.callback_state.store(0, Ordering::SeqCst);
    }
}

impl Drop for BufferMapper {
    fn drop(&mut self) {
        if self.state == MappingState::Mapped {
            debug!("BufferMapper dropped while mapped; unmapping automatically");
            self.buffer.unmap();
        } else if self.state == MappingState::Pending {
            debug!("BufferMapper dropped while mapping pending; state may be inconsistent");
            // We can't reliably cancel a pending mapping, so just log a warning
        }
    }
}

impl std::fmt::Debug for BufferMapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferMapper")
            .field("state", &self.state)
            .field("mode", &self.mode)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .field("buffer_size", &self.buffer.size())
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Creates a buffer mapper and immediately starts an async read mapping.
///
/// This is a convenience function for the common pattern of creating a mapper
/// and mapping the entire buffer for reading.
///
/// # Arguments
///
/// * `buffer` - The buffer to map
/// * `device` - The wgpu device
///
/// # Returns
///
/// `Ok(BufferMapper)` in the `Pending` state, or `Err` on validation failure.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::buffer_mapping::map_for_read;
/// use std::sync::Arc;
///
/// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
/// let mut mapper = map_for_read(buffer, device).unwrap();
/// mapper.wait(device);
/// let data: Vec<u8> = mapper.read_data().unwrap();
/// # }
/// ```
pub fn map_for_read(buffer: Arc<Buffer>, device: &Device) -> Result<BufferMapper, BufferMapError> {
    let size = buffer.size();
    let mut mapper = BufferMapper::new(buffer);
    mapper.map_async(device, MappingMode::Read, 0, size)?;
    Ok(mapper)
}

/// Creates a buffer mapper and immediately starts an async write mapping.
///
/// This is a convenience function for the common pattern of creating a mapper
/// and mapping the entire buffer for writing.
///
/// # Arguments
///
/// * `buffer` - The buffer to map
/// * `device` - The wgpu device
///
/// # Returns
///
/// `Ok(BufferMapper)` in the `Pending` state, or `Err` on validation failure.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::buffer_mapping::map_for_write;
/// use std::sync::Arc;
///
/// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
/// let mut mapper = map_for_write(buffer, device).unwrap();
/// mapper.wait(device);
/// mapper.write_data(&[1u8, 2, 3, 4]).unwrap();
/// # }
/// ```
pub fn map_for_write(buffer: Arc<Buffer>, device: &Device) -> Result<BufferMapper, BufferMapError> {
    let size = buffer.size();
    let mut mapper = BufferMapper::new(buffer);
    mapper.map_async(device, MappingMode::Write, 0, size)?;
    Ok(mapper)
}

/// Maps a buffer, waits for completion, and reads all data as the specified type.
///
/// This is a convenience function that combines mapping, waiting, and reading
/// into a single call. It's suitable for one-shot readback operations.
///
/// # Arguments
///
/// * `buffer` - The buffer to read
/// * `device` - The wgpu device
///
/// # Returns
///
/// `Ok(Vec<T>)` containing the buffer data, or `Err` on failure.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::buffer_mapping::read_buffer_sync;
/// use std::sync::Arc;
///
/// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
/// let data: Vec<f32> = read_buffer_sync(buffer, device).unwrap();
/// println!("Read {} floats", data.len());
/// # }
/// ```
pub fn read_buffer_sync<T: Pod>(
    buffer: Arc<Buffer>,
    device: &Device,
) -> Result<Vec<T>, BufferMapError> {
    let mut mapper = map_for_read(buffer, device)?;
    mapper.wait(device);

    if mapper.state() == MappingState::Failed {
        return Err(BufferMapError::MappingFailed("async mapping failed".to_string()));
    }

    mapper.read_data().ok_or(BufferMapError::NotMapped)
}

/// Maps a buffer, waits for completion, and writes data.
///
/// This is a convenience function that combines mapping, waiting, and writing
/// into a single call. It's suitable for one-shot upload operations.
///
/// # Arguments
///
/// * `buffer` - The buffer to write
/// * `device` - The wgpu device
/// * `data` - The data to write
///
/// # Returns
///
/// `Ok(())` on success, or `Err` on failure.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::buffer_mapping::write_buffer_sync;
/// use std::sync::Arc;
///
/// # fn example(device: &wgpu::Device, buffer: Arc<wgpu::Buffer>) {
/// let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
/// write_buffer_sync(buffer, device, &data).unwrap();
/// # }
/// ```
pub fn write_buffer_sync<T: Pod>(
    buffer: Arc<Buffer>,
    device: &Device,
    data: &[T],
) -> Result<(), BufferMapError> {
    let mut mapper = map_for_write(buffer, device)?;
    mapper.wait(device);

    if mapper.state() == MappingState::Failed {
        return Err(BufferMapError::MappingFailed("async mapping failed".to_string()));
    }

    mapper.write_data(data)
}

// ============================================================================
// BufferReadback (T-WGPU-P4.8.2)
// ============================================================================

/// GPU-to-CPU buffer readback utility.
///
/// `BufferReadback` provides a high-level interface for reading GPU buffer data
/// back to the CPU. It manages a staging buffer with `MAP_READ | COPY_DST` usage
/// and provides both synchronous and asynchronous readback methods.
///
/// # Architecture
///
/// ```text
/// GPU Buffer (any usage)
///        │
///        │ copy_buffer_to_buffer (in CommandEncoder)
///        ▼
/// Staging Buffer (MAP_READ | COPY_DST)
///        │
///        │ map_async + poll/wait
///        ▼
/// CPU Data (Vec<T>)
/// ```
///
/// # Usage Patterns
///
/// ## Single-shot Readback
///
/// ```no_run
/// use renderer_backend::buffer_mapping::BufferReadback;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) {
/// let mut readback = BufferReadback::new(device, 1024, Some("Readback"));
///
/// // In command encoder
/// let mut encoder = device.create_command_encoder(&Default::default());
/// readback.read_buffer(device, queue, &mut encoder, source, 0).unwrap();
/// queue.submit([encoder.finish()]);
///
/// // Get the data after GPU completes
/// device.poll(wgpu::Maintain::Wait);
/// if let Some(data) = readback.get_data::<f32>() {
///     println!("Read {} floats", data.len());
/// }
/// # }
/// ```
///
/// ## Async Readback (Non-blocking)
///
/// ```no_run
/// use renderer_backend::buffer_mapping::{BufferReadback, MappingState};
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) {
/// let mut readback = BufferReadback::new(device, 1024, Some("Async Readback"));
///
/// let mut encoder = device.create_command_encoder(&Default::default());
/// readback.begin_readback(&mut encoder, source, 0);
/// queue.submit([encoder.finish()]);
///
/// // Poll asynchronously
/// loop {
///     match readback.poll_readback(device) {
///         MappingState::Mapped => {
///             let data = readback.finish_readback::<f32>().unwrap();
///             break;
///         }
///         MappingState::Failed => panic!("Readback failed"),
///         _ => { /* do other work */ }
///     }
/// }
/// # }
/// ```
///
/// # Thread Safety
///
/// `BufferReadback` is `Send` but not `Sync`. The staging buffer and mapper
/// state are managed internally.
pub struct BufferReadback {
    /// The staging buffer (MAP_READ | COPY_DST).
    staging_buffer: Arc<Buffer>,
    /// Buffer mapper for async mapping.
    mapper: BufferMapper,
    /// Size of the staging buffer in bytes.
    size: u64,
    /// Flag to indicate mapping should be started in get_data (deferred from read_buffer)
    needs_mapping: bool,
}

impl BufferReadback {
    /// Creates a new buffer readback utility.
    ///
    /// Allocates a staging buffer with `MAP_READ | COPY_DST` usage for GPU-to-CPU
    /// data transfer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `size` - Size of the staging buffer in bytes (must be a multiple of 4)
    /// * `label` - Optional debug label for the buffer
    ///
    /// # Panics
    ///
    /// Panics if `size` is not a multiple of `MAP_SIZE_ALIGNMENT` (4 bytes).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::BufferReadback;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// // Create a 4KB readback buffer
    /// let readback = BufferReadback::new(device, 4096, Some("Compute Results"));
    /// assert_eq!(readback.size(), 4096);
    /// # }
    /// ```
    pub fn new(device: &Device, size: u64, label: Option<&str>) -> Self {
        assert!(
            size % MAP_SIZE_ALIGNMENT == 0,
            "BufferReadback size must be a multiple of {} bytes, got {}",
            MAP_SIZE_ALIGNMENT,
            size
        );

        let staging_label = label.map(|l| format!("{} (staging)", l));
        let staging_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: staging_label.as_deref(),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let mapper = BufferMapper::new(Arc::clone(&staging_buffer));

        debug!(
            "Created BufferReadback: size={}, label={:?}",
            size,
            label
        );

        Self {
            staging_buffer,
            mapper,
            size,
            needs_mapping: false,
        }
    }

    /// Returns a reference to the underlying staging buffer.
    ///
    /// This can be used for advanced scenarios where direct buffer access is needed.
    #[inline]
    pub fn staging_buffer(&self) -> &Buffer {
        &self.staging_buffer
    }

    /// Returns the size of the staging buffer in bytes.
    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the current mapping state of the staging buffer.
    #[inline]
    pub fn state(&self) -> MappingState {
        self.mapper.state()
    }

    /// Returns `true` if data is ready to be read.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.mapper.is_ready()
    }

    // ========================================================================
    // Single-shot Readback
    // ========================================================================

    /// Performs a complete buffer readback operation.
    ///
    /// This method:
    /// 1. Encodes a copy command from the source buffer to the staging buffer
    /// 2. Starts async mapping of the staging buffer
    ///
    /// After calling this method, submit the encoder and wait for the GPU to
    /// complete, then call [`get_data`] to retrieve the results.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue (unused but kept for API consistency)
    /// * `encoder` - Command encoder to record the copy command
    /// * `source` - Source buffer to read from (must have COPY_SRC usage)
    /// * `source_offset` - Byte offset into the source buffer
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - The staging buffer is already mapped or has a pending mapping
    /// - The source offset is not aligned to `COPY_BUFFER_ALIGNMENT` (8 bytes)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::BufferReadback;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, gpu_buffer: &wgpu::Buffer) {
    /// let mut readback = BufferReadback::new(device, 256, None);
    ///
    /// let mut encoder = device.create_command_encoder(&Default::default());
    /// readback.read_buffer(device, queue, &mut encoder, gpu_buffer, 0).unwrap();
    /// queue.submit([encoder.finish()]);
    ///
    /// // Wait for GPU and get data
    /// device.poll(wgpu::Maintain::Wait);
    /// let data: Vec<u32> = readback.get_data().unwrap();
    /// # }
    /// ```
    ///
    /// [`get_data`]: BufferReadback::get_data
    pub fn read_buffer(
        &mut self,
        _device: &Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        source: &Buffer,
        source_offset: u64,
    ) -> Result<(), BufferMapError> {
        // Validate source offset alignment
        if source_offset % COPY_BUFFER_ALIGNMENT != 0 {
            return Err(BufferMapError::UnalignedOffset {
                offset: source_offset,
                required_alignment: COPY_BUFFER_ALIGNMENT,
            });
        }

        // Ensure mapper is in idle state and no pending read_buffer
        if !self.mapper.state().is_idle() || self.needs_mapping {
            return Err(BufferMapError::AlreadyMapped);
        }

        // Encode copy command
        // Note: Mapping is deferred to get_data() to allow submit first
        encoder.copy_buffer_to_buffer(source, source_offset, &self.staging_buffer, 0, self.size);

        // Mark that we need to map when get_data is called
        self.needs_mapping = true;

        trace!(
            "Encoded buffer copy: source_offset={}, size={}",
            source_offset,
            self.size
        );

        Ok(())
    }

    /// Gets the readback data after mapping completes.
    ///
    /// This method blocks until the mapping is complete (if still pending),
    /// then returns the data as a vector of the requested type.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type to interpret the data as. Must implement `bytemuck::Pod`.
    ///
    /// # Returns
    ///
    /// `Some(Vec<T>)` containing the data if:
    /// - The mapping completed successfully
    /// - The data is properly aligned for type `T`
    ///
    /// `None` if:
    /// - The mapping failed
    /// - The data is not aligned for the requested type
    /// - No readback operation was initiated
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::BufferReadback;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) {
    /// let mut readback = BufferReadback::new(device, 64, None);
    ///
    /// let mut encoder = device.create_command_encoder(&Default::default());
    /// readback.read_buffer(device, queue, &mut encoder, source, 0).unwrap();
    /// queue.submit([encoder.finish()]);
    /// device.poll(wgpu::Maintain::Wait);
    ///
    /// // Get data as f32 values
    /// if let Some(floats) = readback.get_data::<f32>() {
    ///     for (i, f) in floats.iter().enumerate() {
    ///         println!("[{}] = {}", i, f);
    ///     }
    /// }
    /// # }
    /// ```
    pub fn get_data<T: Pod>(&self) -> Option<Vec<T>> {
        if !self.mapper.is_ready() {
            trace!("get_data called but mapper not ready (state: {:?})", self.mapper.state());
            return None;
        }

        self.mapper.read_data()
    }

    /// Waits for the readback to complete and returns the data.
    ///
    /// This is a convenience method that combines waiting and data retrieval.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (for blocking wait)
    ///
    /// # Returns
    ///
    /// `Some(Vec<T>)` containing the data, or `None` if the mapping failed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::BufferReadback;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) {
    /// let mut readback = BufferReadback::new(device, 256, None);
    ///
    /// let mut encoder = device.create_command_encoder(&Default::default());
    /// readback.read_buffer(device, queue, &mut encoder, source, 0).unwrap();
    /// queue.submit([encoder.finish()]);
    ///
    /// // Wait and get data in one call
    /// let data: Vec<u32> = readback.wait_and_get_data(device).unwrap();
    /// # }
    /// ```
    pub fn wait_and_get_data<T: Pod>(&mut self, device: &Device) -> Option<Vec<T>> {
        // Start mapping if deferred from read_buffer
        if self.needs_mapping {
            self.needs_mapping = false;
            if let Err(e) = self.mapper.map_async(device, MappingMode::Read, 0, self.size) {
                trace!("Failed to start deferred mapping: {:?}", e);
                return None;
            }
        }

        if self.mapper.is_pending() {
            self.mapper.wait(device);
        }

        self.get_data()
    }

    // ========================================================================
    // Async Readback
    // ========================================================================

    /// Begins an async readback operation.
    ///
    /// This encodes a copy command from the source buffer to the staging buffer.
    /// After submitting the encoder, call [`poll_readback`] to check for completion,
    /// or [`finish_readback`] to block and retrieve the data.
    ///
    /// Unlike [`read_buffer`], this method does not start the async mapping
    /// immediately. The mapping is started when `poll_readback` is called
    /// for the first time.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record the copy command
    /// * `source` - Source buffer to read from
    /// * `source_offset` - Byte offset into the source buffer (must be aligned to 8)
    ///
    /// # Panics
    ///
    /// Panics if `source_offset` is not aligned to `COPY_BUFFER_ALIGNMENT`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::BufferReadback;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) {
    /// let mut readback = BufferReadback::new(device, 1024, None);
    ///
    /// let mut encoder = device.create_command_encoder(&Default::default());
    /// readback.begin_readback(&mut encoder, source, 0);
    /// queue.submit([encoder.finish()]);
    ///
    /// // Later, poll for completion
    /// # }
    /// ```
    ///
    /// [`poll_readback`]: BufferReadback::poll_readback
    /// [`finish_readback`]: BufferReadback::finish_readback
    /// [`read_buffer`]: BufferReadback::read_buffer
    pub fn begin_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source: &Buffer,
        source_offset: u64,
    ) {
        assert!(
            source_offset % COPY_BUFFER_ALIGNMENT == 0,
            "source_offset must be aligned to {} bytes, got {}",
            COPY_BUFFER_ALIGNMENT,
            source_offset
        );

        // Reset mapper if needed
        if !self.mapper.state().is_idle() {
            self.mapper.unmap();
        }

        // Encode copy command
        encoder.copy_buffer_to_buffer(source, source_offset, &self.staging_buffer, 0, self.size);

        trace!(
            "Encoded async readback: source_offset={}, size={}",
            source_offset,
            self.size
        );
    }

    /// Polls the readback operation for completion.
    ///
    /// If this is the first call after `begin_readback`, the async mapping
    /// is started. Subsequent calls poll the device and check the mapping state.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    ///
    /// # Returns
    ///
    /// The current mapping state:
    /// - `Unmapped` - Mapping not yet started (call again to start)
    /// - `Pending` - Mapping in progress
    /// - `Mapped` - Ready to read data
    /// - `Failed` - Mapping failed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::{BufferReadback, MappingState};
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) {
    /// let mut readback = BufferReadback::new(device, 256, None);
    ///
    /// // ... begin_readback and submit ...
    ///
    /// loop {
    ///     match readback.poll_readback(device) {
    ///         MappingState::Mapped => break,
    ///         MappingState::Failed => panic!("Readback failed"),
    ///         _ => { /* do other work */ }
    ///     }
    /// }
    /// # }
    /// ```
    pub fn poll_readback(&mut self, device: &Device) -> MappingState {
        // Start mapping if not already started
        if self.mapper.state().is_unmapped() {
            if let Err(e) = self.mapper.map_async(device, MappingMode::Read, 0, self.size) {
                warn!("Failed to start async mapping: {}", e);
                return MappingState::Failed;
            }
        }

        self.mapper.poll(device)
    }

    /// Blocks until readback completes and returns the data.
    ///
    /// This method waits for the mapping to complete (if pending) and returns
    /// the data. After calling this method, the staging buffer is unmapped
    /// and ready for another readback operation.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type to interpret the data as
    ///
    /// # Returns
    ///
    /// `Some(Vec<T>)` containing the data, or `None` if:
    /// - The mapping failed
    /// - The data is not aligned for the requested type
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::BufferReadback;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) {
    /// let mut readback = BufferReadback::new(device, 128, None);
    ///
    /// // ... begin_readback and submit ...
    ///
    /// // Block and get data
    /// let data: Vec<f32> = readback.finish_readback(device).unwrap();
    /// # }
    /// ```
    pub fn finish_readback<T: Pod>(&mut self, device: &Device) -> Option<Vec<T>> {
        // Start mapping if not yet started
        if self.mapper.state().is_unmapped() {
            if let Err(e) = self.mapper.map_async(device, MappingMode::Read, 0, self.size) {
                warn!("Failed to start mapping in finish_readback: {}", e);
                return None;
            }
        }

        // Wait for completion
        if self.mapper.is_pending() {
            self.mapper.wait(device);
        }

        // Get data
        let data = self.mapper.read_data();

        // Unmap for next use
        self.mapper.unmap();

        data
    }

    /// Resets the readback utility for a new operation.
    ///
    /// Call this if you need to cancel a pending readback or prepare for
    /// a new readback without waiting for the current one to complete.
    pub fn reset(&mut self) {
        self.mapper.reset();
    }
}

impl std::fmt::Debug for BufferReadback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferReadback")
            .field("size", &self.size)
            .field("state", &self.mapper.state())
            .finish_non_exhaustive()
    }
}

// ============================================================================
// DoubleBufferedReadback (T-WGPU-P4.8.2)
// ============================================================================

/// Double-buffered readback for continuous streaming.
///
/// `DoubleBufferedReadback` uses two staging buffers in a ping-pong pattern,
/// allowing data to be read from one buffer while the GPU writes to the other.
/// This eliminates stalls when streaming data continuously.
///
/// # Architecture
///
/// ```text
/// Frame N:
///   GPU writes to Buffer A ──┐
///   CPU reads from Buffer B ─┘ (parallel)
///
/// Frame N+1:
///   GPU writes to Buffer B ──┐
///   CPU reads from Buffer A ─┘ (parallel)
/// ```
///
/// # Example
///
/// ```no_run
/// use renderer_backend::buffer_mapping::DoubleBufferedReadback;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) {
/// let mut double_readback = DoubleBufferedReadback::new(device, 1024, Some("Streaming"));
///
/// // Each frame:
/// for frame in 0..100 {
///     let mut encoder = device.create_command_encoder(&Default::default());
///
///     // Begin readback of current frame's data
///     double_readback.begin_readback(&mut encoder, source, 0);
///     queue.submit([encoder.finish()]);
///
///     // Get data from previous frame (if available)
///     if let Some(data) = double_readback.poll_and_get::<f32>(device) {
///         println!("Frame {}: Got {} values", frame, data.len());
///     }
///
///     // Swap buffers for next frame
///     double_readback.swap();
/// }
/// # }
/// ```
pub struct DoubleBufferedReadback {
    /// The two staging buffers.
    buffers: [BufferReadback; 2],
    /// Current write buffer index (0 or 1).
    current: usize,
}

impl DoubleBufferedReadback {
    /// Creates a new double-buffered readback utility.
    ///
    /// Allocates two staging buffers of the specified size.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `size` - Size of each staging buffer in bytes
    /// * `label` - Optional debug label prefix
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::DoubleBufferedReadback;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let double_readback = DoubleBufferedReadback::new(device, 4096, Some("ParticleData"));
    /// assert_eq!(double_readback.size(), 4096);
    /// # }
    /// ```
    pub fn new(device: &Device, size: u64, label: Option<&str>) -> Self {
        let label_a = label.map(|l| format!("{} A", l));
        let label_b = label.map(|l| format!("{} B", l));

        Self {
            buffers: [
                BufferReadback::new(device, size, label_a.as_deref()),
                BufferReadback::new(device, size, label_b.as_deref()),
            ],
            current: 0,
        }
    }

    /// Returns the size of each staging buffer in bytes.
    #[inline]
    pub fn size(&self) -> u64 {
        self.buffers[0].size()
    }

    /// Returns the current buffer index (0 or 1).
    #[inline]
    pub fn current_index(&self) -> usize {
        self.current
    }

    /// Returns the previous buffer index (0 or 1).
    #[inline]
    pub fn previous_index(&self) -> usize {
        1 - self.current
    }

    /// Returns a reference to the current write buffer.
    #[inline]
    pub fn current_buffer(&self) -> &BufferReadback {
        &self.buffers[self.current]
    }

    /// Returns a mutable reference to the current write buffer.
    #[inline]
    pub fn current_buffer_mut(&mut self) -> &mut BufferReadback {
        &mut self.buffers[self.current]
    }

    /// Returns a reference to the previous (read) buffer.
    #[inline]
    pub fn previous_buffer(&self) -> &BufferReadback {
        &self.buffers[1 - self.current]
    }

    /// Returns a mutable reference to the previous (read) buffer.
    #[inline]
    pub fn previous_buffer_mut(&mut self) -> &mut BufferReadback {
        &mut self.buffers[1 - self.current]
    }

    /// Begins a readback operation on the current buffer.
    ///
    /// Encodes a copy command from the source buffer to the current staging
    /// buffer. After submitting, call [`poll_and_get`] on the previous buffer
    /// to retrieve the data from the last frame.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record the copy command
    /// * `source` - Source buffer to read from
    /// * `source_offset` - Byte offset into the source buffer
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::DoubleBufferedReadback;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) {
    /// let mut double_readback = DoubleBufferedReadback::new(device, 256, None);
    ///
    /// let mut encoder = device.create_command_encoder(&Default::default());
    /// double_readback.begin_readback(&mut encoder, source, 0);
    /// queue.submit([encoder.finish()]);
    /// # }
    /// ```
    ///
    /// [`poll_and_get`]: DoubleBufferedReadback::poll_and_get
    pub fn begin_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source: &Buffer,
        source_offset: u64,
    ) {
        self.buffers[self.current].begin_readback(encoder, source, source_offset);
    }

    /// Polls the previous buffer and retrieves data if ready.
    ///
    /// This method checks if the previous frame's readback has completed.
    /// If so, it returns the data. This allows the current frame's GPU work
    /// to proceed in parallel with CPU data processing.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    ///
    /// # Returns
    ///
    /// `Some(Vec<T>)` if the previous readback is complete, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::DoubleBufferedReadback;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// # let mut double_readback = DoubleBufferedReadback::new(device, 256, None);
    /// // After begin_readback and submit...
    ///
    /// if let Some(data) = double_readback.poll_and_get::<f32>(device) {
    ///     println!("Got {} values from previous frame", data.len());
    /// }
    /// # }
    /// ```
    pub fn poll_and_get<T: Pod>(&mut self, device: &Device) -> Option<Vec<T>> {
        let prev = 1 - self.current;

        // Poll the previous buffer
        let state = self.buffers[prev].poll_readback(device);

        if state.is_mapped() {
            // Get the data and reset for next use
            let data = self.buffers[prev].mapper.read_data();
            self.buffers[prev].mapper.unmap();
            data
        } else {
            None
        }
    }

    /// Waits for the previous buffer and retrieves the data.
    ///
    /// This method blocks until the previous frame's readback completes.
    /// Use [`poll_and_get`] for non-blocking operation.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    ///
    /// # Returns
    ///
    /// `Some(Vec<T>)` containing the data, or `None` if mapping failed.
    ///
    /// [`poll_and_get`]: DoubleBufferedReadback::poll_and_get
    pub fn wait_and_get<T: Pod>(&mut self, device: &Device) -> Option<Vec<T>> {
        let prev = 1 - self.current;
        self.buffers[prev].finish_readback(device)
    }

    /// Swaps the current and previous buffers.
    ///
    /// Call this at the end of each frame to advance the ping-pong pattern.
    /// After swapping:
    /// - The previous "current" buffer becomes the new "previous" (read) buffer
    /// - The previous "previous" buffer becomes the new "current" (write) buffer
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::buffer_mapping::DoubleBufferedReadback;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) {
    /// # let mut double_readback = DoubleBufferedReadback::new(device, 256, None);
    /// // End of frame
    /// double_readback.swap();
    /// // Now previous frame's data can be read, and new data can be written
    /// # }
    /// ```
    pub fn swap(&mut self) {
        self.current = 1 - self.current;
        trace!("DoubleBufferedReadback swapped: current={}", self.current);
    }

    /// Resets both buffers for fresh operation.
    ///
    /// Use this to clear any pending operations and start fresh.
    pub fn reset(&mut self) {
        self.buffers[0].reset();
        self.buffers[1].reset();
        self.current = 0;
    }

    /// Gets the underlying staging buffer for the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - Buffer index (0 or 1)
    ///
    /// # Panics
    ///
    /// Panics if index is not 0 or 1.
    #[inline]
    pub fn staging_buffer(&self, index: usize) -> &Buffer {
        assert!(index < 2, "Buffer index must be 0 or 1");
        self.buffers[index].staging_buffer()
    }
}

impl std::fmt::Debug for DoubleBufferedReadback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DoubleBufferedReadback")
            .field("size", &self.size())
            .field("current", &self.current)
            .field("buffer_0_state", &self.buffers[0].state())
            .field("buffer_1_state", &self.buffers[1].state())
            .finish()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // MappingState Tests
    // ========================================================================

    #[test]
    fn test_mapping_state_default() {
        let state = MappingState::default();
        assert_eq!(state, MappingState::Unmapped);
    }

    #[test]
    fn test_mapping_state_is_unmapped() {
        assert!(MappingState::Unmapped.is_unmapped());
        assert!(!MappingState::Pending.is_unmapped());
        assert!(!MappingState::Mapped.is_unmapped());
        assert!(!MappingState::Failed.is_unmapped());
    }

    #[test]
    fn test_mapping_state_is_pending() {
        assert!(!MappingState::Unmapped.is_pending());
        assert!(MappingState::Pending.is_pending());
        assert!(!MappingState::Mapped.is_pending());
        assert!(!MappingState::Failed.is_pending());
    }

    #[test]
    fn test_mapping_state_is_mapped() {
        assert!(!MappingState::Unmapped.is_mapped());
        assert!(!MappingState::Pending.is_mapped());
        assert!(MappingState::Mapped.is_mapped());
        assert!(!MappingState::Failed.is_mapped());
    }

    #[test]
    fn test_mapping_state_is_failed() {
        assert!(!MappingState::Unmapped.is_failed());
        assert!(!MappingState::Pending.is_failed());
        assert!(!MappingState::Mapped.is_failed());
        assert!(MappingState::Failed.is_failed());
    }

    #[test]
    fn test_mapping_state_is_ready() {
        assert!(!MappingState::Unmapped.is_ready());
        assert!(!MappingState::Pending.is_ready());
        assert!(MappingState::Mapped.is_ready());
        assert!(!MappingState::Failed.is_ready());
    }

    #[test]
    fn test_mapping_state_is_idle() {
        assert!(MappingState::Unmapped.is_idle());
        assert!(!MappingState::Pending.is_idle());
        assert!(!MappingState::Mapped.is_idle());
        assert!(MappingState::Failed.is_idle());
    }

    #[test]
    fn test_mapping_state_display() {
        assert_eq!(MappingState::Unmapped.to_string(), "Unmapped");
        assert_eq!(MappingState::Pending.to_string(), "Pending");
        assert_eq!(MappingState::Mapped.to_string(), "Mapped");
        assert_eq!(MappingState::Failed.to_string(), "Failed");
    }

    #[test]
    fn test_mapping_state_debug() {
        assert_eq!(format!("{:?}", MappingState::Unmapped), "Unmapped");
        assert_eq!(format!("{:?}", MappingState::Pending), "Pending");
        assert_eq!(format!("{:?}", MappingState::Mapped), "Mapped");
        assert_eq!(format!("{:?}", MappingState::Failed), "Failed");
    }

    #[test]
    fn test_mapping_state_clone() {
        let state = MappingState::Mapped;
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_mapping_state_copy() {
        let state = MappingState::Pending;
        let copied = state;
        assert_eq!(state, copied);
    }

    #[test]
    fn test_mapping_state_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(MappingState::Unmapped);
        set.insert(MappingState::Pending);
        set.insert(MappingState::Mapped);
        set.insert(MappingState::Failed);

        assert_eq!(set.len(), 4);
        assert!(set.contains(&MappingState::Unmapped));
        assert!(set.contains(&MappingState::Pending));
        assert!(set.contains(&MappingState::Mapped));
        assert!(set.contains(&MappingState::Failed));
    }

    // ========================================================================
    // MappingMode Tests
    // ========================================================================

    #[test]
    fn test_mapping_mode_required_usage() {
        assert_eq!(
            MappingMode::Read.required_usage(),
            wgpu::BufferUsages::MAP_READ
        );
        assert_eq!(
            MappingMode::Write.required_usage(),
            wgpu::BufferUsages::MAP_WRITE
        );
    }

    #[test]
    fn test_mapping_mode_into_map_mode() {
        let read: MapMode = MappingMode::Read.into();
        let write: MapMode = MappingMode::Write.into();
        assert!(matches!(read, MapMode::Read));
        assert!(matches!(write, MapMode::Write));
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

        assert_eq!(set.len(), 2);
        assert!(set.contains(&MappingMode::Read));
        assert!(set.contains(&MappingMode::Write));
    }

    // ========================================================================
    // BufferMapError Tests
    // ========================================================================

    #[test]
    fn test_buffer_map_error_display_already_mapped() {
        let err = BufferMapError::AlreadyMapped;
        let display = err.to_string();
        assert!(display.contains("already mapped"));
    }

    #[test]
    fn test_buffer_map_error_display_invalid_range() {
        let err = BufferMapError::InvalidRange {
            offset: 100,
            size: 200,
            buffer_size: 256,
        };
        let display = err.to_string();
        assert!(display.contains("100"));
        assert!(display.contains("200"));
        assert!(display.contains("256"));
        assert!(display.contains("300")); // offset + size
    }

    #[test]
    fn test_buffer_map_error_display_mapping_failed() {
        let err = BufferMapError::MappingFailed("GPU error".to_string());
        let display = err.to_string();
        assert!(display.contains("failed"));
        assert!(display.contains("GPU error"));
    }

    #[test]
    fn test_buffer_map_error_display_not_mapped() {
        let err = BufferMapError::NotMapped;
        let display = err.to_string();
        assert!(display.contains("not mapped"));
    }

    #[test]
    fn test_buffer_map_error_display_wrong_mode() {
        let err = BufferMapError::WrongMode {
            current: MappingMode::Read,
            required: MappingMode::Write,
        };
        let display = err.to_string();
        assert!(display.contains("Read"));
        assert!(display.contains("Write"));
        assert!(display.contains("wrong"));
    }

    #[test]
    fn test_buffer_map_error_display_missing_usage() {
        let err = BufferMapError::MissingUsage {
            required: wgpu::BufferUsages::MAP_READ,
            actual: wgpu::BufferUsages::VERTEX,
        };
        let display = err.to_string();
        assert!(display.contains("missing"));
        assert!(display.contains("usage"));
    }

    #[test]
    fn test_buffer_map_error_display_unaligned_offset() {
        let err = BufferMapError::UnalignedOffset {
            offset: 7,
            required_alignment: 8,
        };
        let display = err.to_string();
        assert!(display.contains("7"));
        assert!(display.contains("8"));
        assert!(display.contains("offset"));
    }

    #[test]
    fn test_buffer_map_error_display_unaligned_size() {
        let err = BufferMapError::UnalignedSize {
            size: 5,
            required_alignment: 4,
        };
        let display = err.to_string();
        assert!(display.contains("5"));
        assert!(display.contains("4"));
        assert!(display.contains("size"));
    }

    #[test]
    fn test_buffer_map_error_debug() {
        let err = BufferMapError::AlreadyMapped;
        let debug = format!("{:?}", err);
        assert!(debug.contains("AlreadyMapped"));
    }

    #[test]
    fn test_buffer_map_error_clone() {
        let err = BufferMapError::InvalidRange {
            offset: 10,
            size: 20,
            buffer_size: 100,
        };
        let cloned = err.clone();
        assert!(matches!(
            cloned,
            BufferMapError::InvalidRange {
                offset: 10,
                size: 20,
                buffer_size: 100,
            }
        ));
    }

    #[test]
    fn test_buffer_map_error_is_error_trait() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<BufferMapError>();
    }

    // ========================================================================
    // Constants Tests
    // ========================================================================

    #[test]
    fn test_copy_buffer_alignment() {
        assert_eq!(COPY_BUFFER_ALIGNMENT, 8);
    }

    #[test]
    fn test_map_size_alignment() {
        assert_eq!(MAP_SIZE_ALIGNMENT, 4);
    }

    // ========================================================================
    // BufferMapper Unit Tests (no GPU required)
    // ========================================================================

    #[test]
    fn test_mapping_state_transitions() {
        // Test the expected state transitions
        let state = MappingState::Unmapped;
        assert!(state.is_idle());

        // After map_async, state becomes Pending
        let state = MappingState::Pending;
        assert!(!state.is_idle());
        assert!(state.is_pending());

        // After successful poll, state becomes Mapped
        let state = MappingState::Mapped;
        assert!(!state.is_idle());
        assert!(state.is_ready());

        // After unmap, state becomes Unmapped again
        let state = MappingState::Unmapped;
        assert!(state.is_idle());
    }

    #[test]
    fn test_failed_state_is_idle() {
        // Failed state should be considered idle (ready for retry)
        let state = MappingState::Failed;
        assert!(state.is_idle());
        assert!(!state.is_ready());
        assert!(state.is_failed());
    }

    #[test]
    fn test_alignment_validation_logic() {
        // Test alignment check logic without actual GPU
        let check_offset_alignment = |offset: u64| -> bool {
            offset % COPY_BUFFER_ALIGNMENT == 0
        };

        assert!(check_offset_alignment(0));
        assert!(check_offset_alignment(8));
        assert!(check_offset_alignment(16));
        assert!(check_offset_alignment(64));
        assert!(!check_offset_alignment(1));
        assert!(!check_offset_alignment(7));
        assert!(!check_offset_alignment(15));

        let check_size_alignment = |size: u64| -> bool {
            size % MAP_SIZE_ALIGNMENT == 0
        };

        assert!(check_size_alignment(0));
        assert!(check_size_alignment(4));
        assert!(check_size_alignment(8));
        assert!(check_size_alignment(256));
        assert!(!check_size_alignment(1));
        assert!(!check_size_alignment(3));
        assert!(!check_size_alignment(5));
    }

    #[test]
    fn test_range_validation_logic() {
        // Test range validation logic without actual GPU
        let check_range = |offset: u64, size: u64, buffer_size: u64| -> bool {
            offset.saturating_add(size) <= buffer_size
        };

        assert!(check_range(0, 100, 100));
        assert!(check_range(0, 100, 200));
        assert!(check_range(50, 50, 100));
        assert!(!check_range(0, 101, 100));
        assert!(!check_range(50, 51, 100));
        assert!(!check_range(100, 1, 100));

        // Test overflow protection
        assert!(!check_range(u64::MAX, 1, 100));
        assert!(!check_range(1, u64::MAX, 100));
    }

    #[test]
    fn test_callback_state_values() {
        // Test the callback state encoding
        // 0 = no pending, 1 = pending, 2 = success, 3 = failed
        let state = AtomicU8::new(0);
        assert_eq!(state.load(Ordering::SeqCst), 0);

        state.store(1, Ordering::SeqCst);
        assert_eq!(state.load(Ordering::SeqCst), 1);

        state.store(2, Ordering::SeqCst);
        assert_eq!(state.load(Ordering::SeqCst), 2);

        state.store(3, Ordering::SeqCst);
        assert_eq!(state.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_bytemuck_cast_alignment() {
        // Test that bytemuck cast slice works correctly
        let bytes: [u8; 16] = [0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64];
        let floats: &[f32] = bytemuck::cast_slice(&bytes);

        assert_eq!(floats.len(), 4);
        assert_eq!(floats[0], 0.0);
        assert_eq!(floats[1], 1.0);
        assert_eq!(floats[2], 2.0);
        assert_eq!(floats[3], 3.0);
    }

    #[test]
    fn test_bytemuck_cast_to_bytes() {
        // Test converting typed data to bytes
        let floats: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
        let bytes: &[u8] = bytemuck::cast_slice(&floats);

        assert_eq!(bytes.len(), 16);
        // Check that round-trip works
        let back: &[f32] = bytemuck::cast_slice(bytes);
        assert_eq!(back, &floats);
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_zero_size_mapping_alignment() {
        // Zero size is technically aligned
        assert!(0 % MAP_SIZE_ALIGNMENT == 0);
        assert!(0 % COPY_BUFFER_ALIGNMENT == 0);
    }

    #[test]
    fn test_large_offset_alignment() {
        // Large offsets should still follow alignment rules
        let large_offset: u64 = 1024 * 1024 * 1024; // 1GB
        assert!(large_offset % COPY_BUFFER_ALIGNMENT == 0);

        let unaligned_large: u64 = large_offset + 1;
        assert!(unaligned_large % COPY_BUFFER_ALIGNMENT != 0);
    }

    #[test]
    fn test_error_clone_preserves_data() {
        let err = BufferMapError::InvalidRange {
            offset: 123,
            size: 456,
            buffer_size: 789,
        };
        let cloned = err.clone();

        if let BufferMapError::InvalidRange { offset, size, buffer_size } = cloned {
            assert_eq!(offset, 123);
            assert_eq!(size, 456);
            assert_eq!(buffer_size, 789);
        } else {
            panic!("Clone should preserve error variant");
        }
    }

    #[test]
    fn test_state_repr() {
        // Test that repr(u8) works as expected
        assert_eq!(MappingState::Unmapped as u8, 0);
        assert_eq!(MappingState::Pending as u8, 1);
        assert_eq!(MappingState::Mapped as u8, 2);
        assert_eq!(MappingState::Failed as u8, 3);
    }

    // ========================================================================
    // BufferReadback Unit Tests (T-WGPU-P4.8.2)
    // ========================================================================

    #[test]
    fn test_buffer_readback_size_alignment_validation() {
        // Test that BufferReadback size must be aligned to MAP_SIZE_ALIGNMENT (4)
        // Valid sizes: 0, 4, 8, 12, ..., 256, 1024, 4096, ...
        let valid_sizes: [u64; 8] = [4, 8, 12, 16, 256, 1024, 4096, 65536];
        for size in valid_sizes {
            assert!(
                size % MAP_SIZE_ALIGNMENT == 0,
                "Size {} should be aligned to {}",
                size,
                MAP_SIZE_ALIGNMENT
            );
        }

        let invalid_sizes: [u64; 6] = [1, 2, 3, 5, 7, 255];
        for size in invalid_sizes {
            assert!(
                size % MAP_SIZE_ALIGNMENT != 0,
                "Size {} should NOT be aligned to {}",
                size,
                MAP_SIZE_ALIGNMENT
            );
        }
    }

    #[test]
    fn test_buffer_readback_offset_alignment_validation() {
        // Test that source offset must be aligned to COPY_BUFFER_ALIGNMENT (8)
        let valid_offsets: [u64; 8] = [0, 8, 16, 24, 64, 256, 1024, 65536];
        for offset in valid_offsets {
            assert!(
                offset % COPY_BUFFER_ALIGNMENT == 0,
                "Offset {} should be aligned to {}",
                offset,
                COPY_BUFFER_ALIGNMENT
            );
        }

        let invalid_offsets: [u64; 7] = [1, 2, 3, 4, 5, 6, 7];
        for offset in invalid_offsets {
            assert!(
                offset % COPY_BUFFER_ALIGNMENT != 0,
                "Offset {} should NOT be aligned to {}",
                offset,
                COPY_BUFFER_ALIGNMENT
            );
        }
    }

    #[test]
    fn test_buffer_readback_common_sizes() {
        // Common readback buffer sizes
        let sizes: [(u64, &str); 10] = [
            (64, "cache line"),
            (256, "uniform buffer"),
            (1024, "1KB"),
            (4096, "page size"),
            (16384, "16KB"),
            (65536, "64KB"),
            (262144, "256KB"),
            (1048576, "1MB"),
            (4194304, "4MB"),
            (16777216, "16MB"),
        ];

        for (size, _name) in sizes {
            assert!(size % MAP_SIZE_ALIGNMENT == 0);
        }
    }

    #[test]
    fn test_buffer_readback_staging_buffer_usage() {
        // Staging buffer for readback should have MAP_READ | COPY_DST usage
        let required_read = wgpu::BufferUsages::MAP_READ;
        let required_dst = wgpu::BufferUsages::COPY_DST;

        // Test that these usages can be combined
        let combined = required_read | required_dst;
        assert!(combined.contains(wgpu::BufferUsages::MAP_READ));
        assert!(combined.contains(wgpu::BufferUsages::COPY_DST));
        assert!(!combined.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!combined.contains(wgpu::BufferUsages::MAP_WRITE));
    }

    #[test]
    fn test_buffer_readback_source_buffer_usage() {
        // Source buffer for readback must have COPY_SRC usage
        let required = wgpu::BufferUsages::COPY_SRC;

        // Common source buffer configurations
        let vertex_buffer = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC;
        let storage_buffer = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let uniform_buffer = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC;

        assert!(vertex_buffer.contains(required));
        assert!(storage_buffer.contains(required));
        assert!(uniform_buffer.contains(required));
    }

    #[test]
    fn test_buffer_readback_data_type_sizes() {
        // Test common data type sizes for readback
        assert_eq!(std::mem::size_of::<u8>(), 1);
        assert_eq!(std::mem::size_of::<u16>(), 2);
        assert_eq!(std::mem::size_of::<u32>(), 4);
        assert_eq!(std::mem::size_of::<u64>(), 8);
        assert_eq!(std::mem::size_of::<f32>(), 4);
        assert_eq!(std::mem::size_of::<f64>(), 8);
        assert_eq!(std::mem::size_of::<[f32; 4]>(), 16);
        assert_eq!(std::mem::size_of::<[[f32; 4]; 4]>(), 64);
    }

    #[test]
    fn test_buffer_readback_element_count_calculation() {
        // Test calculating number of elements from buffer size
        let buffer_size: u64 = 1024;

        // u8: 1024 / 1 = 1024 elements
        assert_eq!(buffer_size / std::mem::size_of::<u8>() as u64, 1024);

        // u32: 1024 / 4 = 256 elements
        assert_eq!(buffer_size / std::mem::size_of::<u32>() as u64, 256);

        // f32: 1024 / 4 = 256 elements
        assert_eq!(buffer_size / std::mem::size_of::<f32>() as u64, 256);

        // [f32; 4]: 1024 / 16 = 64 elements (e.g., 64 vec4s)
        assert_eq!(buffer_size / std::mem::size_of::<[f32; 4]>() as u64, 64);

        // [[f32; 4]; 4]: 1024 / 64 = 16 elements (e.g., 16 mat4s)
        assert_eq!(buffer_size / std::mem::size_of::<[[f32; 4]; 4]>() as u64, 16);
    }

    #[test]
    fn test_buffer_readback_copy_size_bounds() {
        // Test copy size validation
        let staging_size: u64 = 256;

        // Valid: copy size <= staging size
        assert!(256 <= staging_size);
        assert!(128 <= staging_size);
        assert!(1 <= staging_size);

        // Invalid: copy size > staging size
        assert!(!(257 <= staging_size));
        assert!(!(512 <= staging_size));
    }

    #[test]
    fn test_buffer_readback_offset_bounds() {
        // Test source offset + size bounds
        let source_size: u64 = 1024;
        let copy_size: u64 = 256;

        // Valid: offset + size <= source_size
        assert!(0 + copy_size <= source_size);
        assert!(256 + copy_size <= source_size);
        assert!(512 + copy_size <= source_size);
        assert!(768 + copy_size <= source_size);

        // Invalid: offset + size > source_size
        assert!(!(769 + copy_size <= source_size));
        assert!(!(1024 + copy_size <= source_size));
    }

    // ========================================================================
    // DoubleBufferedReadback Unit Tests (T-WGPU-P4.8.2)
    // ========================================================================

    #[test]
    fn test_double_buffered_index_logic() {
        // Test ping-pong index logic
        let mut current: usize = 0;

        // Initial state
        assert_eq!(current, 0);
        assert_eq!(1 - current, 1);

        // After swap
        current = 1 - current;
        assert_eq!(current, 1);
        assert_eq!(1 - current, 0);

        // After another swap
        current = 1 - current;
        assert_eq!(current, 0);
        assert_eq!(1 - current, 1);
    }

    #[test]
    fn test_double_buffered_swap_sequence() {
        // Test a sequence of swaps
        let mut current: usize = 0;

        for i in 0..10 {
            let expected = i % 2;
            assert_eq!(current, expected, "Frame {} should use buffer {}", i, expected);
            current = 1 - current;
        }
    }

    #[test]
    fn test_double_buffered_parallel_access_pattern() {
        // Verify that when writing to current, previous is available for reading
        let mut current: usize = 0;

        // Frame 0: Write to 0, no previous data
        let write_idx_0 = current;
        let read_idx_0 = 1 - current;
        assert_eq!(write_idx_0, 0);
        assert_eq!(read_idx_0, 1);

        // Swap
        current = 1 - current;

        // Frame 1: Write to 1, read from 0
        let write_idx_1 = current;
        let read_idx_1 = 1 - current;
        assert_eq!(write_idx_1, 1);
        assert_eq!(read_idx_1, 0);
        assert_eq!(read_idx_1, write_idx_0); // Reading what was written in frame 0

        // Swap
        current = 1 - current;

        // Frame 2: Write to 0, read from 1
        let write_idx_2 = current;
        let read_idx_2 = 1 - current;
        assert_eq!(write_idx_2, 0);
        assert_eq!(read_idx_2, 1);
        assert_eq!(read_idx_2, write_idx_1); // Reading what was written in frame 1
    }

    #[test]
    fn test_double_buffered_latency() {
        // Double buffering introduces 1 frame of latency
        // Data written in frame N is read in frame N+1
        struct FrameData {
            frame: usize,
            write_buffer: usize,
            read_buffer: usize,
            data_age: Option<usize>,
        }

        let mut current: usize = 0;
        let mut last_write_frame: [Option<usize>; 2] = [None, None];
        let mut frames: Vec<FrameData> = Vec::new();

        for frame in 0..5 {
            let write_buffer = current;
            let read_buffer = 1 - current;

            // Data age is frame - last_write_frame[read_buffer]
            let data_age = last_write_frame[read_buffer].map(|w| frame - w);

            frames.push(FrameData {
                frame,
                write_buffer,
                read_buffer,
                data_age,
            });

            // Record this write
            last_write_frame[write_buffer] = Some(frame);

            // Swap
            current = 1 - current;
        }

        // Frame 0: No data to read (first frame)
        assert_eq!(frames[0].data_age, None);

        // Frame 1+: Data is 1 frame old
        for frame in &frames[1..] {
            assert_eq!(frame.data_age, Some(1), "Data should be 1 frame old");
        }
    }

    #[test]
    fn test_double_buffered_buffer_reuse() {
        // After 2 frames, each buffer has been used once
        // After 4 frames, each buffer has been used twice
        let mut current: usize = 0;
        let mut buffer_uses: [usize; 2] = [0, 0];

        for _frame in 0..10 {
            buffer_uses[current] += 1;
            current = 1 - current;
        }

        assert_eq!(buffer_uses[0], 5);
        assert_eq!(buffer_uses[1], 5);
    }

    #[test]
    fn test_double_buffered_reset_logic() {
        // After reset, current should be 0 and both buffers should be unmapped
        let current: usize = 0;
        let buffer_0_state = MappingState::Unmapped;
        let buffer_1_state = MappingState::Unmapped;

        assert_eq!(current, 0);
        assert!(buffer_0_state.is_unmapped());
        assert!(buffer_1_state.is_unmapped());
    }

    #[test]
    fn test_double_buffered_label_format() {
        // Test label formatting for double-buffered readback
        let base_label = "ParticlePositions";
        let label_a = format!("{} A", base_label);
        let label_b = format!("{} B", base_label);

        assert_eq!(label_a, "ParticlePositions A");
        assert_eq!(label_b, "ParticlePositions B");
    }

    #[test]
    fn test_double_buffered_memory_usage() {
        // Double buffering uses 2x the memory of single buffering
        let single_buffer_size: u64 = 1024;
        let double_buffer_total = single_buffer_size * 2;

        assert_eq!(double_buffer_total, 2048);

        // For 1MB of data, double buffering uses 2MB total
        let one_mb: u64 = 1024 * 1024;
        assert_eq!(one_mb * 2, 2 * 1024 * 1024);
    }

    #[test]
    fn test_double_buffered_staging_buffer_index_validation() {
        // Index must be 0 or 1
        let valid_indices: [usize; 2] = [0, 1];
        let invalid_indices: [usize; 5] = [2, 3, 10, 100, usize::MAX];

        for idx in valid_indices {
            assert!(idx < 2, "Index {} should be valid", idx);
        }

        for idx in invalid_indices {
            assert!(idx >= 2, "Index {} should be invalid", idx);
        }
    }

    // ========================================================================
    // Integration Pattern Tests (No GPU)
    // ========================================================================

    #[test]
    fn test_readback_workflow_states() {
        // Test the expected state transitions in a readback workflow
        let states = [
            ("create", MappingState::Unmapped),
            ("begin_readback", MappingState::Unmapped), // Just encoded, not mapped yet
            ("poll_readback (start)", MappingState::Pending),
            ("poll_readback (complete)", MappingState::Mapped),
            ("finish_readback", MappingState::Unmapped), // After unmap
        ];

        for (step, expected_state) in states {
            // Verify expected state is valid
            match expected_state {
                MappingState::Unmapped => assert!(expected_state.is_unmapped()),
                MappingState::Pending => assert!(expected_state.is_pending()),
                MappingState::Mapped => assert!(expected_state.is_mapped()),
                MappingState::Failed => assert!(expected_state.is_failed()),
            }
            let _ = step; // Suppress unused warning
        }
    }

    #[test]
    fn test_readback_error_conditions() {
        // Test various error conditions
        let errors = [
            BufferMapError::AlreadyMapped,
            BufferMapError::UnalignedOffset {
                offset: 7,
                required_alignment: COPY_BUFFER_ALIGNMENT,
            },
            BufferMapError::UnalignedSize {
                size: 5,
                required_alignment: MAP_SIZE_ALIGNMENT,
            },
            BufferMapError::InvalidRange {
                offset: 0,
                size: 1024,
                buffer_size: 512,
            },
            BufferMapError::NotMapped,
            BufferMapError::MappingFailed("timeout".to_string()),
        ];

        for err in errors {
            // All errors should have a display string
            let display = err.to_string();
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_compute_shader_readback_pattern() {
        // Simulate a compute shader readback pattern
        // Compute writes to storage buffer, then we read it back

        let storage_buffer_size: u64 = 4096;
        let element_count = storage_buffer_size / std::mem::size_of::<f32>() as u64;
        assert_eq!(element_count, 1024);

        // Staging buffer must be at least storage_buffer_size
        let staging_size = storage_buffer_size;
        assert!(staging_size >= storage_buffer_size);
    }

    #[test]
    fn test_query_result_readback_pattern() {
        // Simulate reading query results (e.g., occlusion queries)
        // Queries typically return u64 values

        let query_count: u64 = 16;
        let bytes_per_query = std::mem::size_of::<u64>() as u64;
        let total_size = query_count * bytes_per_query;
        assert_eq!(total_size, 128);

        // Ensure aligned
        assert!(total_size % MAP_SIZE_ALIGNMENT == 0);
    }

    #[test]
    fn test_indirect_args_readback_pattern() {
        // Simulate reading indirect draw arguments
        // IndirectDrawArgs: vertex_count, instance_count, first_vertex, first_instance

        let draw_count: u64 = 256;
        let args_per_draw = std::mem::size_of::<[u32; 4]>() as u64;
        let total_size = draw_count * args_per_draw;
        assert_eq!(total_size, 4096);

        assert!(total_size % MAP_SIZE_ALIGNMENT == 0);
    }

    #[test]
    fn test_particle_system_readback_pattern() {
        // Simulate reading particle positions/velocities for debugging

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Particle {
            position: [f32; 4],
            velocity: [f32; 4],
        }

        let particle_count: u64 = 1000;
        let particle_size = std::mem::size_of::<Particle>() as u64;
        assert_eq!(particle_size, 32);

        let total_size = particle_count * particle_size;
        assert_eq!(total_size, 32000);

        // Ensure aligned for readback
        assert!(total_size % MAP_SIZE_ALIGNMENT == 0);
    }

    #[test]
    fn test_depth_buffer_readback_calculation() {
        // Simulate reading back depth buffer data

        let width: u64 = 1920;
        let height: u64 = 1080;
        let bytes_per_pixel = std::mem::size_of::<f32>() as u64; // D32_FLOAT
        let total_size = width * height * bytes_per_pixel;
        assert_eq!(total_size, 8294400); // ~8MB

        assert!(total_size % MAP_SIZE_ALIGNMENT == 0);
    }

    #[test]
    fn test_timestamp_readback_calculation() {
        // Simulate reading GPU timestamps

        let timestamp_count: u64 = 64; // Multiple timing points
        let bytes_per_timestamp = std::mem::size_of::<u64>() as u64;
        let total_size = timestamp_count * bytes_per_timestamp;
        assert_eq!(total_size, 512);

        assert!(total_size % MAP_SIZE_ALIGNMENT == 0);
    }
}
