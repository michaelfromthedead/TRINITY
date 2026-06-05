//! Queue management and command submission for TRINITY.
//!
//! This module provides the [`TrinityQueue`] struct, which wraps wgpu's queue
//! with additional tracking for pending submissions and completion callbacks.
//!
//! # Overview
//!
//! In wgpu, the queue is responsible for:
//! - Submitting command buffers to the GPU
//! - Writing data directly to buffers and textures
//! - Tracking when submitted work completes
//!
//! This module enhances wgpu's queue with:
//! - Pending submission tracking via atomic counter
//! - Convenient methods for single and batch command buffer submission
//! - Completion callback support with automatic pending count updates
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::TrinityQueue;
//!
//! # fn example(wgpu_queue: wgpu::Queue, encoder: wgpu::CommandEncoder) {
//! let queue = TrinityQueue::new(wgpu_queue);
//!
//! // Submit a command buffer
//! let command_buffer = encoder.finish();
//! let index = queue.submit_single(command_buffer);
//!
//! // Track pending work
//! println!("Pending submissions: {}", queue.pending_count());
//!
//! // Register completion callback
//! queue.on_submitted_work_done(|| {
//!     println!("GPU work completed!");
//! });
//! # }
//! ```
//!
//! # Thread Safety
//!
//! `TrinityQueue` is `Send + Sync` and can be safely shared across threads.
//! The pending count is managed via atomic operations.

use log::{debug, trace, warn};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ============================================================================
// Constants
// ============================================================================

/// Minimum alignment for buffer write offsets.
///
/// wgpu requires that the offset passed to `write_buffer()` be aligned to 4 bytes.
/// This is the value of `wgpu::COPY_BUFFER_ALIGNMENT`.
pub const COPY_BUFFER_ALIGNMENT: u64 = 4;

/// Minimum alignment for texture row byte counts.
///
/// wgpu requires that `bytes_per_row` in `ImageDataLayout` be aligned to 256 bytes.
/// This is the value of `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`.
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur when validating queue write operations.
///
/// This enum provides detailed information about validation failures when
/// writing to buffers or textures through the queue.
///
/// # Example
///
/// ```
/// use renderer_backend::device::QueueWriteError;
///
/// let error = QueueWriteError::BufferOffsetUnaligned {
///     offset: 3,
///     alignment: 4,
/// };
///
/// assert!(error.to_string().contains("offset 3"));
/// assert!(error.to_string().contains("aligned to 4"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueWriteError {
    /// Buffer write offset is not properly aligned.
    ///
    /// wgpu requires buffer write offsets to be aligned to `COPY_BUFFER_ALIGNMENT`
    /// (4 bytes). This error occurs when an unaligned offset is provided.
    BufferOffsetUnaligned {
        /// The offset that was provided.
        offset: u64,
        /// The required alignment (always 4 bytes currently).
        alignment: u64,
    },

    /// Buffer write would overflow the buffer's capacity.
    ///
    /// This error occurs when `offset + data.len()` exceeds the buffer's size.
    BufferOverflow {
        /// The byte offset into the buffer.
        offset: u64,
        /// The length of data being written.
        data_len: usize,
        /// The total size of the buffer.
        buffer_size: u64,
    },

    /// Texture data layout is invalid.
    ///
    /// This error occurs when the `ImageDataLayout` parameters don't match
    /// the requirements for the texture write operation.
    TextureLayoutInvalid {
        /// Human-readable description of what's wrong.
        reason: String,
    },

    /// Texture write data size doesn't match expected size.
    ///
    /// This error occurs when the provided data slice doesn't contain
    /// enough bytes for the specified texture region.
    TextureDataSizeMismatch {
        /// The number of bytes provided.
        provided: usize,
        /// The minimum number of bytes expected.
        expected: usize,
    },
}

impl fmt::Display for QueueWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueueWriteError::BufferOffsetUnaligned { offset, alignment } => {
                write!(
                    f,
                    "Buffer write offset {} is not aligned to {} bytes",
                    offset, alignment
                )
            }
            QueueWriteError::BufferOverflow {
                offset,
                data_len,
                buffer_size,
            } => {
                write!(
                    f,
                    "Buffer write would overflow: offset {} + data length {} = {} exceeds buffer size {}",
                    offset,
                    data_len,
                    offset + *data_len as u64,
                    buffer_size
                )
            }
            QueueWriteError::TextureLayoutInvalid { reason } => {
                write!(f, "Texture data layout is invalid: {}", reason)
            }
            QueueWriteError::TextureDataSizeMismatch { provided, expected } => {
                write!(
                    f,
                    "Texture data size mismatch: provided {} bytes but expected at least {} bytes",
                    provided, expected
                )
            }
        }
    }
}

impl std::error::Error for QueueWriteError {}

// ============================================================================
// TrinityQueue
// ============================================================================

/// TRINITY's queue wrapper with submission tracking.
///
/// `TrinityQueue` encapsulates a wgpu queue and provides additional functionality
/// for tracking pending GPU work. This is useful for:
///
/// - Monitoring GPU utilization and backpressure
/// - Implementing frame pacing strategies
/// - Coordinating multi-threaded submission
///
/// # Pending Count Semantics
///
/// The pending count tracks the number of submission batches that have been
/// submitted but not yet completed. Each call to `submit()` or `submit_single()`
/// increments the count by 1. When the GPU completes all work up to a submission
/// point (as notified via `on_submitted_work_done`), the count decrements.
///
/// Note: The pending count is a heuristic. Due to the asynchronous nature of
/// GPU execution, the actual number of in-flight operations may differ slightly.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::TrinityQueue;
/// use std::sync::Arc;
///
/// # fn example(wgpu_queue: wgpu::Queue) {
/// let queue = Arc::new(TrinityQueue::new(wgpu_queue));
///
/// // Submit multiple command buffers at once
/// # let buffers: Vec<wgpu::CommandBuffer> = vec![];
/// let index = queue.submit(buffers);
/// assert!(queue.pending_count() >= 1);
///
/// // Get completion notification
/// let queue_clone = Arc::clone(&queue);
/// queue.on_submitted_work_done(move || {
///     println!("Work done! Remaining: {}", queue_clone.pending_count());
/// });
/// # }
/// ```
pub struct TrinityQueue {
    /// The underlying wgpu queue.
    queue: wgpu::Queue,
    /// Counter for pending (not-yet-completed) submissions.
    ///
    /// Incremented on each `submit()` call, decremented when the completion
    /// callback fires. Uses relaxed ordering since exact accuracy is not
    /// required for monitoring purposes.
    pending_submissions: AtomicU64,
}

impl TrinityQueue {
    /// Create a new TrinityQueue wrapping the given wgpu queue.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue to wrap
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(device: &wgpu::Device, queue: wgpu::Queue) {
    /// let trinity_queue = TrinityQueue::new(queue);
    /// # }
    /// ```
    pub fn new(queue: wgpu::Queue) -> Self {
        debug!("TrinityQueue: Created queue wrapper");
        Self {
            queue,
            pending_submissions: AtomicU64::new(0),
        }
    }

    /// Submit command buffers to the GPU.
    ///
    /// This method accepts an iterator of command buffers and submits them
    /// as a single batch to the GPU. The batch will execute in order, but
    /// may overlap with other batches submitted from other threads.
    ///
    /// # Arguments
    ///
    /// * `command_buffers` - An iterator of command buffers to submit
    ///
    /// # Returns
    ///
    /// A [`wgpu::SubmissionIndex`] that can be used to:
    /// - Poll for completion via `device.poll(Maintain::WaitForSubmissionIndex(index))`
    /// - Track which submissions have completed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue, encoder1: wgpu::CommandEncoder, encoder2: wgpu::CommandEncoder) {
    /// let buffers = vec![
    ///     encoder1.finish(),
    ///     encoder2.finish(),
    /// ];
    ///
    /// let index = queue.submit(buffers);
    /// println!("Submitted with index: {:?}", index);
    /// # }
    /// ```
    pub fn submit<I>(&self, command_buffers: I) -> wgpu::SubmissionIndex
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        // Increment pending count before submission
        let prev_count = self.pending_submissions.fetch_add(1, Ordering::Relaxed);
        trace!(
            "TrinityQueue: Submitting command buffers (pending: {} -> {})",
            prev_count,
            prev_count + 1
        );

        // Submit to the underlying queue
        let index = self.queue.submit(command_buffers);

        trace!("TrinityQueue: Submission complete, index: {:?}", index);
        index
    }

    /// Submit a single command buffer to the GPU.
    ///
    /// This is a convenience method equivalent to `submit(std::iter::once(command_buffer))`.
    ///
    /// # Arguments
    ///
    /// * `command_buffer` - The command buffer to submit
    ///
    /// # Returns
    ///
    /// A [`wgpu::SubmissionIndex`] for tracking completion.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue, encoder: wgpu::CommandEncoder) {
    /// let command_buffer = encoder.finish();
    /// let index = queue.submit_single(command_buffer);
    /// # }
    /// ```
    #[inline]
    pub fn submit_single(&self, command_buffer: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        self.submit(std::iter::once(command_buffer))
    }

    /// Submit an empty batch (no command buffers).
    ///
    /// This can be useful for:
    /// - Flushing pending GPU writes
    /// - Creating a sync point for completion callbacks
    /// - Testing submission infrastructure
    ///
    /// # Returns
    ///
    /// A [`wgpu::SubmissionIndex`] that marks this point in the queue.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue) {
    /// // Flush any pending writes and get a sync point
    /// let index = queue.submit_empty();
    /// # }
    /// ```
    #[inline]
    pub fn submit_empty(&self) -> wgpu::SubmissionIndex {
        self.submit(std::iter::empty())
    }

    /// Register a callback to be invoked when all previously submitted work completes.
    ///
    /// The callback will be invoked once all command buffers submitted before this
    /// call have finished executing on the GPU. This is useful for:
    ///
    /// - Releasing temporary resources after GPU usage
    /// - Implementing fence-like synchronization
    /// - Tracking when specific operations complete
    ///
    /// # Arguments
    ///
    /// * `callback` - A closure to invoke on completion. Must be `Send + 'static`
    ///   since it may be invoked from a different thread.
    ///
    /// # Callback Execution
    ///
    /// The callback will be invoked:
    /// - On the next `device.poll()` after the work completes
    /// - Or automatically if using `device.poll(Maintain::Wait)`
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    /// use std::sync::Arc;
    ///
    /// # fn example(queue: &TrinityQueue, encoder: wgpu::CommandEncoder) {
    /// // Submit work
    /// let buffer = encoder.finish();
    /// queue.submit_single(buffer);
    ///
    /// // Get notified when complete
    /// queue.on_submitted_work_done(|| {
    ///     println!("GPU finished the frame!");
    /// });
    /// # }
    /// ```
    pub fn on_submitted_work_done<F>(&self, callback: F)
    where
        F: FnOnce() + Send + 'static,
    {
        trace!("TrinityQueue: Registering completion callback");
        self.queue.on_submitted_work_done(callback);
    }

    /// Register a callback that also decrements the pending count.
    ///
    /// This is the recommended way to track pending submissions. The callback
    /// will first decrement the pending count, then invoke your callback.
    ///
    /// # Arguments
    ///
    /// * `callback` - A closure to invoke on completion
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    /// use std::sync::Arc;
    ///
    /// # fn example(queue: Arc<TrinityQueue>, encoder: wgpu::CommandEncoder) {
    /// // Submit work
    /// let buffer = encoder.finish();
    /// queue.submit_single(buffer);
    ///
    /// // Track completion with automatic pending decrement
    /// let queue_clone = Arc::clone(&queue);
    /// queue.on_submitted_work_done_tracked(move || {
    ///     println!("GPU finished! Pending: {}", queue_clone.pending_count());
    /// });
    /// # }
    /// ```
    pub fn on_submitted_work_done_tracked<F>(self: &Arc<Self>, callback: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let queue_clone = Arc::clone(self);
        self.queue.on_submitted_work_done(move || {
            // Decrement pending count
            let prev = queue_clone
                .pending_submissions
                .fetch_sub(1, Ordering::Relaxed);
            trace!(
                "TrinityQueue: Work completed (pending: {} -> {})",
                prev,
                prev.saturating_sub(1)
            );

            // Invoke user callback
            callback();
        });
    }

    /// Get the current count of pending submissions.
    ///
    /// This returns the number of submission batches that have been submitted
    /// but for which completion callbacks have not yet fired.
    ///
    /// # Returns
    ///
    /// The approximate number of pending submissions. This is a snapshot and
    /// may be stale by the time it's used.
    ///
    /// # Note
    ///
    /// The pending count is only accurate if you use `on_submitted_work_done_tracked`
    /// for your completion callbacks. Using `on_submitted_work_done` directly
    /// will not decrement the count.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue) {
    /// let pending = queue.pending_count();
    /// if pending > 3 {
    ///     println!("GPU backpressure detected! Consider waiting.");
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn pending_count(&self) -> u64 {
        self.pending_submissions.load(Ordering::Relaxed)
    }

    /// Check if there is any pending work.
    ///
    /// # Returns
    ///
    /// `true` if there are pending submissions, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue) {
    /// if queue.has_pending_work() {
    ///     println!("GPU is busy");
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn has_pending_work(&self) -> bool {
        self.pending_count() > 0
    }

    /// Get a reference to the underlying wgpu queue.
    ///
    /// This provides direct access for operations not wrapped by TrinityQueue,
    /// such as `write_buffer` and `write_texture`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue, buffer: &wgpu::Buffer, data: &[u8]) {
    /// // Write data directly to a buffer
    /// queue.inner().write_buffer(buffer, 0, data);
    /// # }
    /// ```
    #[inline]
    pub fn inner(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Write data to a buffer.
    ///
    /// This is a convenience method forwarding to `queue.write_buffer()`.
    /// The write is staged and will be submitted on the next `submit()` call
    /// or when the device polls.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to write to
    /// * `offset` - Byte offset into the buffer
    /// * `data` - The data to write
    ///
    /// # Panics
    ///
    /// - If `offset + data.len()` exceeds the buffer size
    /// - If the buffer doesn't have `COPY_DST` usage
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue, uniform_buffer: &wgpu::Buffer) {
    /// let transform_data: [f32; 16] = [
    ///     1.0, 0.0, 0.0, 0.0,
    ///     0.0, 1.0, 0.0, 0.0,
    ///     0.0, 0.0, 1.0, 0.0,
    ///     0.0, 0.0, 0.0, 1.0,
    /// ];
    /// queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&transform_data));
    /// # }
    /// ```
    #[inline]
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: wgpu::BufferAddress, data: &[u8]) {
        trace!(
            "TrinityQueue: Writing {} bytes to buffer at offset {}",
            data.len(),
            offset
        );
        self.queue.write_buffer(buffer, offset, data);
    }

    /// Write data to a texture.
    ///
    /// This is a convenience method forwarding to `queue.write_texture()`.
    /// The write is staged and will be submitted on the next `submit()` call.
    ///
    /// # Arguments
    ///
    /// * `texture` - The destination texture copy view
    /// * `data` - The data to write
    /// * `data_layout` - How the data is laid out in memory
    /// * `size` - The size of the region to write
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue, texture: &wgpu::Texture) {
    /// let size = wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 };
    /// let data = vec![255u8; 256 * 256 * 4]; // RGBA
    ///
    /// queue.write_texture(
    ///     wgpu::ImageCopyTexture {
    ///         texture,
    ///         mip_level: 0,
    ///         origin: wgpu::Origin3d::ZERO,
    ///         aspect: wgpu::TextureAspect::All,
    ///     },
    ///     &data,
    ///     wgpu::ImageDataLayout {
    ///         offset: 0,
    ///         bytes_per_row: Some(256 * 4),
    ///         rows_per_image: Some(256),
    ///     },
    ///     size,
    /// );
    /// # }
    /// ```
    #[inline]
    pub fn write_texture(
        &self,
        texture: wgpu::ImageCopyTexture<'_>,
        data: &[u8],
        data_layout: wgpu::ImageDataLayout,
        size: wgpu::Extent3d,
    ) {
        trace!(
            "TrinityQueue: Writing {} bytes to texture ({}x{}x{})",
            data.len(),
            size.width,
            size.height,
            size.depth_or_array_layers
        );
        self.queue.write_texture(texture, data, data_layout, size);
    }

    // ========================================================================
    // Validated Write Methods
    // ========================================================================

    /// Write data to a buffer with validation.
    ///
    /// This method validates the write parameters before forwarding to wgpu:
    ///
    /// - **Alignment**: The offset must be aligned to [`COPY_BUFFER_ALIGNMENT`] (4 bytes)
    /// - **Bounds**: `offset + data.len()` must not exceed `buffer_size`
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to write to
    /// * `offset` - Byte offset into the buffer (must be 4-byte aligned)
    /// * `data` - The data to write
    /// * `buffer_size` - The total size of the buffer for bounds checking
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the write was successful
    /// - `Err(QueueWriteError)` if validation failed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue, buffer: &wgpu::Buffer) {
    /// let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    /// let buffer_size = 1024;
    ///
    /// match queue.write_buffer_validated(buffer, 0, bytemuck::cast_slice(&data), buffer_size) {
    ///     Ok(()) => println!("Write successful"),
    ///     Err(e) => eprintln!("Write validation failed: {}", e),
    /// }
    /// # }
    /// ```
    pub fn write_buffer_validated(
        &self,
        buffer: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        data: &[u8],
        buffer_size: u64,
    ) -> Result<(), QueueWriteError> {
        // Validate alignment
        if offset % COPY_BUFFER_ALIGNMENT != 0 {
            warn!(
                "TrinityQueue: Buffer write offset {} is not aligned to {} bytes",
                offset, COPY_BUFFER_ALIGNMENT
            );
            return Err(QueueWriteError::BufferOffsetUnaligned {
                offset,
                alignment: COPY_BUFFER_ALIGNMENT,
            });
        }

        // Validate bounds
        let end_offset = offset.saturating_add(data.len() as u64);
        if end_offset > buffer_size {
            warn!(
                "TrinityQueue: Buffer write would overflow: {} + {} > {}",
                offset,
                data.len(),
                buffer_size
            );
            return Err(QueueWriteError::BufferOverflow {
                offset,
                data_len: data.len(),
                buffer_size,
            });
        }

        // All checks passed, perform the write
        trace!(
            "TrinityQueue: Validated write of {} bytes to buffer at offset {}",
            data.len(),
            offset
        );
        self.queue.write_buffer(buffer, offset, data);
        Ok(())
    }

    /// Write data to a buffer with automatic size detection.
    ///
    /// This is a convenience wrapper around [`write_buffer_validated`] that
    /// automatically retrieves the buffer size via `buffer.size()`.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to write to
    /// * `offset` - Byte offset into the buffer (must be 4-byte aligned)
    /// * `data` - The data to write
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the write was successful
    /// - `Err(QueueWriteError)` if validation failed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue, buffer: &wgpu::Buffer) {
    /// let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    ///
    /// // Buffer size is automatically detected
    /// if let Err(e) = queue.write_buffer_checked(buffer, 0, bytemuck::cast_slice(&data)) {
    ///     eprintln!("Write failed: {}", e);
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn write_buffer_checked(
        &self,
        buffer: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        data: &[u8],
    ) -> Result<(), QueueWriteError> {
        self.write_buffer_validated(buffer, offset, data, buffer.size())
    }

    /// Write data to a texture with validation.
    ///
    /// This method validates the texture write parameters before forwarding to wgpu:
    ///
    /// - **bytes_per_row alignment**: Must be aligned to [`COPY_BYTES_PER_ROW_ALIGNMENT`] (256 bytes)
    ///   when writing multi-row data
    /// - **Data size**: Provided data must be large enough for the specified region
    ///
    /// # Arguments
    ///
    /// * `texture` - The destination texture copy view
    /// * `data` - The data to write
    /// * `data_layout` - How the data is laid out in memory
    /// * `size` - The size of the region to write
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the write was successful
    /// - `Err(QueueWriteError)` if validation failed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityQueue;
    ///
    /// # fn example(queue: &TrinityQueue, texture: &wgpu::Texture) {
    /// let width = 256u32;
    /// let height = 256u32;
    /// let bytes_per_pixel = 4u32;
    ///
    /// // bytes_per_row must be aligned to 256 bytes
    /// let unpadded_bytes_per_row = width * bytes_per_pixel;
    /// let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
    ///
    /// let data_size = (padded_bytes_per_row * height) as usize;
    /// let data = vec![255u8; data_size];
    ///
    /// let result = queue.write_texture_validated(
    ///     wgpu::ImageCopyTexture {
    ///         texture,
    ///         mip_level: 0,
    ///         origin: wgpu::Origin3d::ZERO,
    ///         aspect: wgpu::TextureAspect::All,
    ///     },
    ///     &data,
    ///     wgpu::ImageDataLayout {
    ///         offset: 0,
    ///         bytes_per_row: Some(padded_bytes_per_row),
    ///         rows_per_image: Some(height),
    ///     },
    ///     wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    /// );
    ///
    /// match result {
    ///     Ok(()) => println!("Texture write successful"),
    ///     Err(e) => eprintln!("Texture write validation failed: {}", e),
    /// }
    /// # }
    /// ```
    pub fn write_texture_validated(
        &self,
        texture: wgpu::ImageCopyTexture<'_>,
        data: &[u8],
        data_layout: wgpu::ImageDataLayout,
        size: wgpu::Extent3d,
    ) -> Result<(), QueueWriteError> {
        // Validate bytes_per_row alignment for multi-row writes
        if size.height > 1 || size.depth_or_array_layers > 1 {
            if let Some(bytes_per_row) = data_layout.bytes_per_row {
                if bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT != 0 {
                    warn!(
                        "TrinityQueue: bytes_per_row {} is not aligned to {} bytes",
                        bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT
                    );
                    return Err(QueueWriteError::TextureLayoutInvalid {
                        reason: format!(
                            "bytes_per_row {} must be aligned to {} bytes for multi-row writes",
                            bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT
                        ),
                    });
                }
            } else {
                // bytes_per_row is required for multi-row writes
                return Err(QueueWriteError::TextureLayoutInvalid {
                    reason: "bytes_per_row is required when height > 1 or depth > 1".to_string(),
                });
            }
        }

        // Validate rows_per_image for 3D textures / texture arrays
        if size.depth_or_array_layers > 1 && data_layout.rows_per_image.is_none() {
            return Err(QueueWriteError::TextureLayoutInvalid {
                reason: "rows_per_image is required when depth_or_array_layers > 1".to_string(),
            });
        }

        // Calculate expected data size
        let expected_size = calculate_texture_data_size(&data_layout, &size);
        if data.len() < expected_size {
            warn!(
                "TrinityQueue: Texture data size mismatch: provided {} bytes, expected {} bytes",
                data.len(),
                expected_size
            );
            return Err(QueueWriteError::TextureDataSizeMismatch {
                provided: data.len(),
                expected: expected_size,
            });
        }

        // All checks passed, perform the write
        trace!(
            "TrinityQueue: Validated write of {} bytes to texture ({}x{}x{})",
            data.len(),
            size.width,
            size.height,
            size.depth_or_array_layers
        );
        self.queue.write_texture(texture, data, data_layout, size);
        Ok(())
    }
}

// Implement Debug manually to avoid exposing wgpu::Queue internals
impl std::fmt::Debug for TrinityQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrinityQueue")
            .field("pending_submissions", &self.pending_count())
            .finish_non_exhaustive()
    }
}

// Safety: wgpu::Queue is Send + Sync, and AtomicU64 is Send + Sync
unsafe impl Send for TrinityQueue {}
unsafe impl Sync for TrinityQueue {}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate the expected data size for a texture write operation.
///
/// This function computes the minimum number of bytes required for a texture
/// write operation based on the data layout and extent.
///
/// # Algorithm
///
/// For a texture write, the data is organized as:
/// - Multiple layers (depth_or_array_layers)
/// - Each layer contains multiple rows (determined by rows_per_image or height)
/// - Each row contains `bytes_per_row` bytes
///
/// The total size is:
/// `(layers - 1) * rows_per_image * bytes_per_row + (height - 1) * bytes_per_row + row_data`
///
/// However, for the last row of the last layer, we don't need padding, so we
/// use a conservative estimate that the full buffer must be at least:
/// `offset + total_computed_size`
fn calculate_texture_data_size(layout: &wgpu::ImageDataLayout, size: &wgpu::Extent3d) -> usize {
    // Handle degenerate cases
    if size.width == 0 || size.height == 0 || size.depth_or_array_layers == 0 {
        return 0;
    }

    let bytes_per_row = layout.bytes_per_row.unwrap_or(0) as usize;
    let rows_per_image = layout.rows_per_image.unwrap_or(size.height) as usize;
    let offset = layout.offset as usize;

    // For single-row, single-layer writes, we only need the width's worth of data
    // (after accounting for bytes_per_row which might be larger)
    if size.height == 1 && size.depth_or_array_layers == 1 {
        // For a single row, we need at least one row of data starting at offset
        // The actual row data might be less than bytes_per_row (unpadded)
        if bytes_per_row > 0 {
            return offset + bytes_per_row;
        }
        // If bytes_per_row is 0/None for single row, we can't determine size
        // This will be validated elsewhere - return a minimal size
        return offset;
    }

    // Calculate size for multi-row or multi-layer writes
    let layers = size.depth_or_array_layers as usize;
    let height = size.height as usize;

    // Full layers (all but the last)
    let full_layers_size = if layers > 1 {
        (layers - 1) * rows_per_image * bytes_per_row
    } else {
        0
    };

    // Full rows in the last layer (all but the last row)
    let full_rows_size = if height > 1 {
        (height - 1) * bytes_per_row
    } else {
        0
    };

    // For the last row, we still need at least bytes_per_row
    // (even though the actual pixel data might be smaller)
    let last_row_size = bytes_per_row;

    offset + full_layers_size + full_rows_size + last_row_size
}

/// Compute the aligned bytes_per_row for a given unpadded row size.
///
/// This utility function rounds up to the next multiple of [`COPY_BYTES_PER_ROW_ALIGNMENT`].
///
/// # Example
///
/// ```
/// use renderer_backend::device::align_bytes_per_row;
///
/// assert_eq!(align_bytes_per_row(100), 256);  // Rounds up to 256
/// assert_eq!(align_bytes_per_row(256), 256);  // Already aligned
/// assert_eq!(align_bytes_per_row(257), 512);  // Rounds up to 512
/// ```
#[inline]
pub fn align_bytes_per_row(unpadded: u32) -> u32 {
    let alignment = COPY_BYTES_PER_ROW_ALIGNMENT;
    (unpadded + alignment - 1) / alignment * alignment
}

/// Check if a buffer offset is properly aligned for queue writes.
///
/// # Example
///
/// ```
/// use renderer_backend::device::is_buffer_offset_aligned;
///
/// assert!(is_buffer_offset_aligned(0));
/// assert!(is_buffer_offset_aligned(4));
/// assert!(is_buffer_offset_aligned(1024));
/// assert!(!is_buffer_offset_aligned(1));
/// assert!(!is_buffer_offset_aligned(3));
/// ```
#[inline]
pub fn is_buffer_offset_aligned(offset: u64) -> bool {
    offset % COPY_BUFFER_ALIGNMENT == 0
}

/// Check if a bytes_per_row value is properly aligned for texture writes.
///
/// # Example
///
/// ```
/// use renderer_backend::device::is_bytes_per_row_aligned;
///
/// assert!(is_bytes_per_row_aligned(256));
/// assert!(is_bytes_per_row_aligned(512));
/// assert!(!is_bytes_per_row_aligned(100));
/// assert!(!is_bytes_per_row_aligned(300));
/// ```
#[inline]
pub fn is_bytes_per_row_aligned(bytes_per_row: u32) -> bool {
    bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT == 0
}

// ============================================================================
// SubmissionTracker
// ============================================================================

/// Tracks multiple submissions with individual completion callbacks.
///
/// This is useful when you need to track completion of specific submissions
/// rather than just the overall pending count.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{TrinityQueue, SubmissionTracker};
/// use std::sync::Arc;
///
/// # fn example(queue: Arc<TrinityQueue>) {
/// let tracker = SubmissionTracker::new();
///
/// // Submit and track multiple command buffers
/// # let encoder1: wgpu::CommandEncoder = todo!();
/// # let encoder2: wgpu::CommandEncoder = todo!();
/// let id1 = tracker.track_submission(&queue, encoder1.finish());
/// let id2 = tracker.track_submission(&queue, encoder2.finish());
///
/// // Check if specific submissions completed
/// println!("Submission {} completed: {}", id1, tracker.is_completed(id1));
/// # }
/// ```
pub struct SubmissionTracker {
    /// Next ID to assign
    next_id: AtomicU64,
    /// Set of completed submission IDs
    completed: std::sync::RwLock<std::collections::HashSet<u64>>,
}

impl SubmissionTracker {
    /// Create a new submission tracker.
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(0),
            completed: std::sync::RwLock::new(std::collections::HashSet::new()),
        }
    }

    /// Track a submission and return its ID.
    ///
    /// Submits the command buffer and registers a completion callback
    /// that marks the submission as complete.
    ///
    /// # Arguments
    ///
    /// * `queue` - The queue to submit to
    /// * `command_buffer` - The command buffer to submit
    ///
    /// # Returns
    ///
    /// A unique ID for this submission that can be used with `is_completed()`.
    pub fn track_submission(
        self: &Arc<Self>,
        queue: &Arc<TrinityQueue>,
        command_buffer: wgpu::CommandBuffer,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        // Submit the command buffer
        queue.submit_single(command_buffer);

        // Register completion callback that marks this submission complete
        let tracker = Arc::clone(self);
        queue.on_submitted_work_done(move || {
            tracker.mark_completed(id);
        });

        id
    }

    /// Check if a submission has completed.
    ///
    /// # Arguments
    ///
    /// * `id` - The submission ID returned from `track_submission()`
    ///
    /// # Returns
    ///
    /// `true` if the submission has completed, `false` otherwise.
    pub fn is_completed(&self, id: u64) -> bool {
        self.completed.read().unwrap().contains(&id)
    }

    /// Mark a submission as completed (internal use).
    fn mark_completed(&self, id: u64) {
        self.completed.write().unwrap().insert(id);
    }

    /// Clear completed submissions from tracking.
    ///
    /// Call this periodically to avoid unbounded memory growth.
    pub fn clear_completed(&self) {
        self.completed.write().unwrap().clear();
    }

    /// Get the count of tracked completions.
    pub fn completed_count(&self) -> usize {
        self.completed.read().unwrap().len()
    }
}

impl Default for SubmissionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SubmissionTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubmissionTracker")
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .field("completed_count", &self.completed_count())
            .finish()
    }
}

// ============================================================================
// SubmissionBatcher
// ============================================================================

/// Default threshold for command buffer count before auto-flush.
pub const DEFAULT_BATCH_COUNT_THRESHOLD: usize = 8;

/// Default threshold for time before auto-flush (2ms).
pub const DEFAULT_BATCH_TIME_THRESHOLD_MS: u64 = 2;

/// Configuration for submission batching behavior.
///
/// Controls when the batcher automatically flushes pending command buffers
/// to the GPU queue.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum number of command buffers to accumulate before flushing.
    ///
    /// When the pending buffer count reaches this threshold, the batcher
    /// automatically flushes to the GPU queue.
    ///
    /// Default: 8
    pub count_threshold: usize,

    /// Maximum time to wait before flushing, in milliseconds.
    ///
    /// When this duration has elapsed since the first pending buffer was
    /// added, the batcher automatically flushes on the next operation.
    ///
    /// Default: 2ms
    pub time_threshold_ms: u64,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            count_threshold: DEFAULT_BATCH_COUNT_THRESHOLD,
            time_threshold_ms: DEFAULT_BATCH_TIME_THRESHOLD_MS,
        }
    }
}

impl BatcherConfig {
    /// Create a config with custom thresholds.
    ///
    /// # Arguments
    ///
    /// * `count_threshold` - Max command buffers before flush
    /// * `time_threshold_ms` - Max milliseconds before flush
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::BatcherConfig;
    ///
    /// // More aggressive batching: larger batches, longer wait
    /// let config = BatcherConfig::new(16, 5);
    /// assert_eq!(config.count_threshold, 16);
    /// assert_eq!(config.time_threshold_ms, 5);
    /// ```
    pub fn new(count_threshold: usize, time_threshold_ms: u64) -> Self {
        Self {
            count_threshold,
            time_threshold_ms,
        }
    }

    /// Create a config optimized for low latency (smaller batches, shorter timeout).
    ///
    /// Uses count threshold of 4 and time threshold of 1ms.
    pub fn low_latency() -> Self {
        Self {
            count_threshold: 4,
            time_threshold_ms: 1,
        }
    }

    /// Create a config optimized for throughput (larger batches, longer timeout).
    ///
    /// Uses count threshold of 16 and time threshold of 4ms.
    pub fn high_throughput() -> Self {
        Self {
            count_threshold: 16,
            time_threshold_ms: 4,
        }
    }

    /// Get the time threshold as a Duration.
    #[inline]
    pub fn time_threshold(&self) -> Duration {
        Duration::from_millis(self.time_threshold_ms)
    }
}

/// Metrics tracking batch effectiveness.
///
/// These metrics help monitor how well batching is working and can be used
/// to tune the batcher configuration.
#[derive(Debug, Clone, Default)]
pub struct BatcherMetrics {
    /// Total number of submissions to the GPU queue.
    ///
    /// Each flush increments this by 1, regardless of how many command
    /// buffers were in the batch.
    pub total_submissions: u64,

    /// Total number of command buffers batched.
    ///
    /// This counts every command buffer that was added to a batch,
    /// across all flushes.
    pub total_batched_buffers: u64,

    /// Number of flushes triggered by count threshold.
    pub count_triggered_flushes: u64,

    /// Number of flushes triggered by time threshold.
    pub time_triggered_flushes: u64,

    /// Number of flushes triggered by frame end.
    pub frame_end_flushes: u64,

    /// Number of force flushes (synchronous operations).
    pub force_flushes: u64,

    /// Number of empty flushes (no buffers pending).
    pub empty_flushes: u64,

    /// Largest batch size observed.
    pub max_batch_size: usize,

    /// Smallest non-empty batch size observed.
    pub min_batch_size: usize,
}

impl BatcherMetrics {
    /// Calculate the average batch size.
    ///
    /// Returns 0.0 if no submissions have been made.
    pub fn avg_batch_size(&self) -> f64 {
        let non_empty_submissions = self.total_submissions - self.empty_flushes;
        if non_empty_submissions == 0 {
            0.0
        } else {
            self.total_batched_buffers as f64 / non_empty_submissions as f64
        }
    }

    /// Calculate the batching efficiency ratio.
    ///
    /// This is the ratio of command buffers to submissions. A value of 1.0
    /// means no batching occurred (one buffer per submission). Higher values
    /// indicate more effective batching.
    ///
    /// Returns 0.0 if no buffers have been batched.
    pub fn batching_efficiency(&self) -> f64 {
        if self.total_submissions == 0 {
            0.0
        } else {
            self.total_batched_buffers as f64 / self.total_submissions as f64
        }
    }

    /// Get a summary string suitable for logging.
    pub fn summary(&self) -> String {
        format!(
            "BatcherMetrics {{ submissions: {}, buffers: {}, avg_size: {:.2}, efficiency: {:.2}, \
             triggers: {{ count: {}, time: {}, frame: {}, force: {} }} }}",
            self.total_submissions,
            self.total_batched_buffers,
            self.avg_batch_size(),
            self.batching_efficiency(),
            self.count_triggered_flushes,
            self.time_triggered_flushes,
            self.frame_end_flushes,
            self.force_flushes
        )
    }
}

/// Internal state protected by mutex.
struct BatcherState {
    /// Pending command buffers waiting to be submitted.
    pending_buffers: Vec<wgpu::CommandBuffer>,
    /// Timestamp when the first buffer was added to the current batch.
    batch_start_time: Option<Instant>,
    /// Accumulated metrics.
    metrics: BatcherMetrics,
}

impl BatcherState {
    fn new() -> Self {
        Self {
            pending_buffers: Vec::with_capacity(DEFAULT_BATCH_COUNT_THRESHOLD),
            batch_start_time: None,
            metrics: BatcherMetrics::default(),
        }
    }
}

/// Batches command buffer submissions for improved GPU performance.
///
/// `SubmissionBatcher` accumulates command buffers and submits them as a batch
/// to the GPU queue. This reduces CPU-GPU synchronization overhead and can
/// significantly improve performance when many small submissions would otherwise
/// be made.
///
/// # Batching Strategy
///
/// The batcher flushes pending command buffers when any of these conditions
/// are met:
///
/// 1. **Count threshold**: The number of pending buffers reaches `count_threshold`
/// 2. **Time threshold**: `time_threshold` has elapsed since the first buffer was added
/// 3. **Frame end**: `end_frame()` is called (typically at the end of each render frame)
/// 4. **Force flush**: `force_flush()` is called for synchronous operations
///
/// # Thread Safety
///
/// `SubmissionBatcher` is `Send + Sync` and can be safely shared across threads.
/// Internal state is protected by a mutex.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{SubmissionBatcher, BatcherConfig, TrinityQueue};
/// use std::sync::Arc;
///
/// # fn example(queue: Arc<TrinityQueue>) {
/// let batcher = SubmissionBatcher::new(Arc::clone(&queue), BatcherConfig::default());
///
/// // Add command buffers - they accumulate until a threshold is reached
/// # let encoder1: wgpu::CommandEncoder = todo!();
/// # let encoder2: wgpu::CommandEncoder = todo!();
/// batcher.add(encoder1.finish());
/// batcher.add(encoder2.finish());
///
/// // At frame end, flush any remaining buffers
/// batcher.end_frame();
///
/// // Check batching effectiveness
/// let metrics = batcher.metrics();
/// println!("Average batch size: {:.2}", metrics.avg_batch_size());
/// # }
/// ```
pub struct SubmissionBatcher {
    /// The underlying queue to submit batches to.
    queue: Arc<TrinityQueue>,
    /// Batcher configuration.
    config: BatcherConfig,
    /// Thread-safe state.
    state: Mutex<BatcherState>,
}

impl SubmissionBatcher {
    /// Create a new submission batcher.
    ///
    /// # Arguments
    ///
    /// * `queue` - The queue to submit batched command buffers to
    /// * `config` - Configuration controlling batching behavior
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::{SubmissionBatcher, BatcherConfig, TrinityQueue};
    /// use std::sync::Arc;
    ///
    /// # fn example(queue: Arc<TrinityQueue>) {
    /// // Default configuration
    /// let batcher = SubmissionBatcher::new(Arc::clone(&queue), BatcherConfig::default());
    ///
    /// // Low-latency configuration
    /// let fast_batcher = SubmissionBatcher::new(Arc::clone(&queue), BatcherConfig::low_latency());
    /// # }
    /// ```
    pub fn new(queue: Arc<TrinityQueue>, config: BatcherConfig) -> Self {
        debug!(
            "SubmissionBatcher: Created with count_threshold={}, time_threshold={}ms",
            config.count_threshold, config.time_threshold_ms
        );
        Self {
            queue,
            config,
            state: Mutex::new(BatcherState::new()),
        }
    }

    /// Create a new submission batcher with default configuration.
    ///
    /// Equivalent to `SubmissionBatcher::new(queue, BatcherConfig::default())`.
    pub fn with_defaults(queue: Arc<TrinityQueue>) -> Self {
        Self::new(queue, BatcherConfig::default())
    }

    /// Add a command buffer to the pending batch.
    ///
    /// The buffer may be submitted immediately if thresholds are reached,
    /// or held for later batching.
    ///
    /// # Arguments
    ///
    /// * `command_buffer` - The command buffer to add
    ///
    /// # Returns
    ///
    /// `true` if a flush was triggered (count or time threshold reached),
    /// `false` if the buffer was queued for later.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::SubmissionBatcher;
    ///
    /// # fn example(batcher: &SubmissionBatcher, encoder: wgpu::CommandEncoder) {
    /// let flushed = batcher.add(encoder.finish());
    /// if flushed {
    ///     println!("Batch was submitted to GPU");
    /// }
    /// # }
    /// ```
    pub fn add(&self, command_buffer: wgpu::CommandBuffer) -> bool {
        let mut state = self.state.lock().unwrap();

        // Record batch start time on first buffer
        if state.batch_start_time.is_none() {
            state.batch_start_time = Some(Instant::now());
        }

        state.pending_buffers.push(command_buffer);

        trace!(
            "SubmissionBatcher: Added buffer (pending: {})",
            state.pending_buffers.len()
        );

        // Check count threshold
        if state.pending_buffers.len() >= self.config.count_threshold {
            trace!(
                "SubmissionBatcher: Count threshold reached ({})",
                self.config.count_threshold
            );
            self.flush_internal(&mut state, FlushReason::CountThreshold);
            return true;
        }

        // Check time threshold
        if let Some(start_time) = state.batch_start_time {
            if start_time.elapsed() >= self.config.time_threshold() {
                trace!(
                    "SubmissionBatcher: Time threshold reached ({}ms)",
                    self.config.time_threshold_ms
                );
                self.flush_internal(&mut state, FlushReason::TimeThreshold);
                return true;
            }
        }

        false
    }

    /// Add multiple command buffers to the pending batch.
    ///
    /// This is more efficient than calling `add()` multiple times when you
    /// have several buffers ready.
    ///
    /// # Arguments
    ///
    /// * `command_buffers` - Iterator of command buffers to add
    ///
    /// # Returns
    ///
    /// `true` if a flush was triggered, `false` otherwise.
    pub fn add_many<I>(&self, command_buffers: I) -> bool
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        let mut state = self.state.lock().unwrap();

        // Record batch start time on first buffer
        if state.batch_start_time.is_none() {
            state.batch_start_time = Some(Instant::now());
        }

        let initial_count = state.pending_buffers.len();
        state.pending_buffers.extend(command_buffers);
        let added_count = state.pending_buffers.len() - initial_count;

        trace!(
            "SubmissionBatcher: Added {} buffers (pending: {})",
            added_count,
            state.pending_buffers.len()
        );

        // Check count threshold
        if state.pending_buffers.len() >= self.config.count_threshold {
            trace!(
                "SubmissionBatcher: Count threshold reached ({})",
                self.config.count_threshold
            );
            self.flush_internal(&mut state, FlushReason::CountThreshold);
            return true;
        }

        // Check time threshold
        if let Some(start_time) = state.batch_start_time {
            if start_time.elapsed() >= self.config.time_threshold() {
                trace!(
                    "SubmissionBatcher: Time threshold reached ({}ms)",
                    self.config.time_threshold_ms
                );
                self.flush_internal(&mut state, FlushReason::TimeThreshold);
                return true;
            }
        }

        false
    }

    /// Flush any pending command buffers to the queue.
    ///
    /// This submits all accumulated buffers as a single batch. If no buffers
    /// are pending, this is a no-op.
    ///
    /// # Returns
    ///
    /// The number of command buffers that were flushed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::SubmissionBatcher;
    ///
    /// # fn example(batcher: &SubmissionBatcher) {
    /// let count = batcher.flush();
    /// println!("Flushed {} command buffers", count);
    /// # }
    /// ```
    pub fn flush(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        let count = state.pending_buffers.len();
        self.flush_internal(&mut state, FlushReason::Manual);
        count
    }

    /// Force flush for synchronous operations.
    ///
    /// This is similar to `flush()` but:
    /// - Records the flush as a "force flush" in metrics
    /// - Intended for operations that require all prior work to be submitted
    ///
    /// Use this before operations that need synchronization, such as:
    /// - GPU readback operations
    /// - Presenting to a surface
    /// - Resource destruction
    ///
    /// # Returns
    ///
    /// The number of command buffers that were flushed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::SubmissionBatcher;
    ///
    /// # fn example(batcher: &SubmissionBatcher) {
    /// // Before reading back GPU data
    /// batcher.force_flush();
    /// // ... perform synchronous GPU read ...
    /// # }
    /// ```
    pub fn force_flush(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        let count = state.pending_buffers.len();
        self.flush_internal(&mut state, FlushReason::Force);
        count
    }

    /// Signal the end of a frame and flush pending buffers.
    ///
    /// Call this at the end of each render frame to ensure all command
    /// buffers are submitted before presenting.
    ///
    /// # Returns
    ///
    /// The number of command buffers that were flushed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::SubmissionBatcher;
    ///
    /// # fn example(batcher: &SubmissionBatcher) {
    /// // At frame end
    /// let flushed = batcher.end_frame();
    /// println!("Frame ended, flushed {} buffers", flushed);
    /// # }
    /// ```
    pub fn end_frame(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        let count = state.pending_buffers.len();
        self.flush_internal(&mut state, FlushReason::FrameEnd);
        count
    }

    /// Get the current number of pending command buffers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::SubmissionBatcher;
    ///
    /// # fn example(batcher: &SubmissionBatcher) {
    /// println!("Pending buffers: {}", batcher.pending_count());
    /// # }
    /// ```
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.state.lock().unwrap().pending_buffers.len()
    }

    /// Check if there are any pending command buffers.
    #[inline]
    pub fn has_pending(&self) -> bool {
        self.pending_count() > 0
    }

    /// Get a snapshot of the current metrics.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::SubmissionBatcher;
    ///
    /// # fn example(batcher: &SubmissionBatcher) {
    /// let metrics = batcher.metrics();
    /// println!("Total submissions: {}", metrics.total_submissions);
    /// println!("Average batch size: {:.2}", metrics.avg_batch_size());
    /// println!("Batching efficiency: {:.2}x", metrics.batching_efficiency());
    /// # }
    /// ```
    pub fn metrics(&self) -> BatcherMetrics {
        self.state.lock().unwrap().metrics.clone()
    }

    /// Reset the metrics counters to zero.
    ///
    /// This is useful for per-frame or per-session metrics tracking.
    pub fn reset_metrics(&self) {
        let mut state = self.state.lock().unwrap();
        state.metrics = BatcherMetrics::default();
    }

    /// Get a reference to the underlying queue.
    #[inline]
    pub fn queue(&self) -> &Arc<TrinityQueue> {
        &self.queue
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &BatcherConfig {
        &self.config
    }

    /// Check if the time threshold has elapsed.
    ///
    /// This can be used to determine if a flush should be triggered
    /// without actually performing the flush.
    pub fn time_threshold_elapsed(&self) -> bool {
        let state = self.state.lock().unwrap();
        if let Some(start_time) = state.batch_start_time {
            start_time.elapsed() >= self.config.time_threshold()
        } else {
            false
        }
    }

    /// Poll for time-based flush.
    ///
    /// Call this periodically (e.g., from a render loop) to trigger
    /// time-based flushes even when no new buffers are being added.
    ///
    /// # Returns
    ///
    /// The number of buffers flushed, or 0 if no flush was needed.
    pub fn poll(&self) -> usize {
        let mut state = self.state.lock().unwrap();

        if state.pending_buffers.is_empty() {
            return 0;
        }

        if let Some(start_time) = state.batch_start_time {
            if start_time.elapsed() >= self.config.time_threshold() {
                let count = state.pending_buffers.len();
                self.flush_internal(&mut state, FlushReason::TimeThreshold);
                return count;
            }
        }

        0
    }

    // Internal flush implementation.
    fn flush_internal(&self, state: &mut BatcherState, reason: FlushReason) {
        let buffer_count = state.pending_buffers.len();

        // Update metrics based on reason
        match reason {
            FlushReason::CountThreshold => state.metrics.count_triggered_flushes += 1,
            FlushReason::TimeThreshold => state.metrics.time_triggered_flushes += 1,
            FlushReason::FrameEnd => state.metrics.frame_end_flushes += 1,
            FlushReason::Force => state.metrics.force_flushes += 1,
            FlushReason::Manual => {} // No specific counter for manual flushes
        }

        if buffer_count == 0 {
            state.metrics.empty_flushes += 1;
            trace!("SubmissionBatcher: Empty flush ({:?})", reason);
            return;
        }

        // Update batch size tracking
        state.metrics.total_submissions += 1;
        state.metrics.total_batched_buffers += buffer_count as u64;

        if buffer_count > state.metrics.max_batch_size {
            state.metrics.max_batch_size = buffer_count;
        }
        if state.metrics.min_batch_size == 0 || buffer_count < state.metrics.min_batch_size {
            state.metrics.min_batch_size = buffer_count;
        }

        // Take pending buffers and submit
        let buffers = std::mem::take(&mut state.pending_buffers);

        debug!(
            "SubmissionBatcher: Flushing {} command buffers ({:?})",
            buffer_count, reason
        );

        self.queue.submit(buffers);

        // Reset batch state
        state.batch_start_time = None;
        state.pending_buffers.reserve(self.config.count_threshold);
    }
}

/// Reason for a flush operation (used for metrics).
#[derive(Debug, Clone, Copy)]
enum FlushReason {
    /// Flush triggered by reaching count threshold.
    CountThreshold,
    /// Flush triggered by reaching time threshold.
    TimeThreshold,
    /// Flush triggered by frame end.
    FrameEnd,
    /// Flush triggered by force_flush() call.
    Force,
    /// Flush triggered by manual flush() call.
    Manual,
}

impl std::fmt::Debug for SubmissionBatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("SubmissionBatcher")
            .field("pending_count", &state.pending_buffers.len())
            .field("config", &self.config)
            .field("metrics", &state.metrics)
            .finish()
    }
}

// Safety: All internal state is protected by Mutex, queue is Arc
unsafe impl Send for SubmissionBatcher {}
unsafe impl Sync for SubmissionBatcher {}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Most TrinityQueue tests require actual GPU hardware and are
    // placed in integration tests. These tests cover the non-GPU parts.

    #[test]
    fn test_submission_tracker_new() {
        let tracker = SubmissionTracker::new();
        assert_eq!(tracker.completed_count(), 0);
    }

    #[test]
    fn test_submission_tracker_mark_completed() {
        let tracker = SubmissionTracker::new();

        tracker.mark_completed(0);
        tracker.mark_completed(1);
        tracker.mark_completed(5);

        assert!(tracker.is_completed(0));
        assert!(tracker.is_completed(1));
        assert!(!tracker.is_completed(2));
        assert!(tracker.is_completed(5));
        assert_eq!(tracker.completed_count(), 3);
    }

    #[test]
    fn test_submission_tracker_clear() {
        let tracker = SubmissionTracker::new();

        tracker.mark_completed(0);
        tracker.mark_completed(1);
        assert_eq!(tracker.completed_count(), 2);

        tracker.clear_completed();
        assert_eq!(tracker.completed_count(), 0);
        assert!(!tracker.is_completed(0));
    }

    #[test]
    fn test_submission_tracker_default() {
        let tracker = SubmissionTracker::default();
        assert_eq!(tracker.completed_count(), 0);
    }

    #[test]
    fn test_submission_tracker_debug() {
        let tracker = SubmissionTracker::new();
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("SubmissionTracker"));
        assert!(debug_str.contains("next_id"));
        assert!(debug_str.contains("completed_count"));
    }

    // ========================================================================
    // BatcherConfig Tests
    // ========================================================================

    #[test]
    fn test_batcher_config_default() {
        let config = BatcherConfig::default();
        assert_eq!(config.count_threshold, DEFAULT_BATCH_COUNT_THRESHOLD);
        assert_eq!(config.time_threshold_ms, DEFAULT_BATCH_TIME_THRESHOLD_MS);
    }

    #[test]
    fn test_batcher_config_new() {
        let config = BatcherConfig::new(16, 5);
        assert_eq!(config.count_threshold, 16);
        assert_eq!(config.time_threshold_ms, 5);
    }

    #[test]
    fn test_batcher_config_low_latency() {
        let config = BatcherConfig::low_latency();
        assert_eq!(config.count_threshold, 4);
        assert_eq!(config.time_threshold_ms, 1);
    }

    #[test]
    fn test_batcher_config_high_throughput() {
        let config = BatcherConfig::high_throughput();
        assert_eq!(config.count_threshold, 16);
        assert_eq!(config.time_threshold_ms, 4);
    }

    #[test]
    fn test_batcher_config_time_threshold() {
        let config = BatcherConfig::new(8, 2);
        assert_eq!(config.time_threshold(), Duration::from_millis(2));
    }

    #[test]
    fn test_batcher_config_debug() {
        let config = BatcherConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("BatcherConfig"));
        assert!(debug_str.contains("count_threshold"));
        assert!(debug_str.contains("time_threshold_ms"));
    }

    // ========================================================================
    // BatcherMetrics Tests
    // ========================================================================

    #[test]
    fn test_batcher_metrics_default() {
        let metrics = BatcherMetrics::default();
        assert_eq!(metrics.total_submissions, 0);
        assert_eq!(metrics.total_batched_buffers, 0);
        assert_eq!(metrics.count_triggered_flushes, 0);
        assert_eq!(metrics.time_triggered_flushes, 0);
        assert_eq!(metrics.frame_end_flushes, 0);
        assert_eq!(metrics.force_flushes, 0);
        assert_eq!(metrics.empty_flushes, 0);
        assert_eq!(metrics.max_batch_size, 0);
        assert_eq!(metrics.min_batch_size, 0);
    }

    #[test]
    fn test_batcher_metrics_avg_batch_size_zero() {
        let metrics = BatcherMetrics::default();
        assert_eq!(metrics.avg_batch_size(), 0.0);
    }

    #[test]
    fn test_batcher_metrics_avg_batch_size() {
        let mut metrics = BatcherMetrics::default();
        metrics.total_submissions = 4;
        metrics.total_batched_buffers = 20;
        // avg = 20 / 4 = 5.0
        assert!((metrics.avg_batch_size() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batcher_metrics_avg_batch_size_excludes_empty() {
        let mut metrics = BatcherMetrics::default();
        metrics.total_submissions = 5;
        metrics.total_batched_buffers = 20;
        metrics.empty_flushes = 1;
        // Non-empty submissions = 5 - 1 = 4
        // avg = 20 / 4 = 5.0
        assert!((metrics.avg_batch_size() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batcher_metrics_batching_efficiency_zero() {
        let metrics = BatcherMetrics::default();
        assert_eq!(metrics.batching_efficiency(), 0.0);
    }

    #[test]
    fn test_batcher_metrics_batching_efficiency() {
        let mut metrics = BatcherMetrics::default();
        metrics.total_submissions = 5;
        metrics.total_batched_buffers = 25;
        // efficiency = 25 / 5 = 5.0
        assert!((metrics.batching_efficiency() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batcher_metrics_summary() {
        let mut metrics = BatcherMetrics::default();
        metrics.total_submissions = 10;
        metrics.total_batched_buffers = 50;
        metrics.count_triggered_flushes = 3;
        metrics.time_triggered_flushes = 2;
        metrics.frame_end_flushes = 4;
        metrics.force_flushes = 1;

        let summary = metrics.summary();
        assert!(summary.contains("submissions: 10"));
        assert!(summary.contains("buffers: 50"));
        assert!(summary.contains("count: 3"));
        assert!(summary.contains("time: 2"));
        assert!(summary.contains("frame: 4"));
        assert!(summary.contains("force: 1"));
    }

    #[test]
    fn test_batcher_metrics_debug() {
        let metrics = BatcherMetrics::default();
        let debug_str = format!("{:?}", metrics);
        assert!(debug_str.contains("BatcherMetrics"));
        assert!(debug_str.contains("total_submissions"));
    }
}
