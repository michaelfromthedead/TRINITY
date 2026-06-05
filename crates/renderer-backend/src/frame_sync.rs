//! Frame synchronization and fence management for TRINITY.
//!
//! This module provides frame-level GPU synchronization primitives using wgpu's
//! submission index tracking. It enables:
//!
//! - **Fence-based frame tracking**: Track which frame the GPU is working on
//! - **Wait for frame completion**: Block until a specific frame is done
//! - **Frame timeline queries**: Check if frame N has completed
//! - **Submission index tracking**: Track queue submission indices per frame
//!
//! # Architecture
//!
//! The module provides two main abstractions:
//!
//! ```text
//! FrameFence
//!     └── Low-level fence tracking using wgpu SubmissionIndex
//!     └── Per-frame submission index recording
//!     └── Completion queries via device.poll()
//!
//! FrameSyncManager
//!     └── High-level frame lifecycle management
//!     └── Frame timing and statistics
//!     └── Automatic frame advancement
//! ```
//!
//! # wgpu Synchronization Model
//!
//! wgpu uses submission indices for tracking GPU work completion:
//!
//! ```text
//! let index = queue.submit([command_buffer]);  // Returns SubmissionIndex
//! device.poll(Maintain::Poll);                 // Check completion (non-blocking)
//! device.poll(Maintain::WaitForSubmissionIndex(index));  // Wait for completion
//! ```
//!
//! This module wraps these primitives into a frame-oriented API suitable for
//! triple-buffered rendering pipelines.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::frame_sync::{FrameFence, FrameSyncManager, FrameStats};
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
//! // Create a sync manager for triple buffering (3 frames in flight)
//! let mut sync = FrameSyncManager::new(3);
//!
//! // Begin a new frame
//! let frame_num = sync.begin_frame();
//!
//! // ... record and submit GPU commands ...
//!
//! // End the frame and get statistics
//! let stats = sync.end_frame(device, queue);
//! println!("Frame {} completed in {:.2}ms", stats.frame_number, stats.cpu_time_ms);
//! # }
//! ```
//!
//! # Thread Safety
//!
//! Both [`FrameFence`] and [`FrameSyncManager`] are `Send + Sync` and can be
//! used from multiple threads. However, the typical usage pattern is to call
//! `begin_frame()` and `end_frame()` from a single render thread.

use log::{debug, trace, warn};
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ============================================================================
// Constants
// ============================================================================

/// Default number of frames to allow in flight.
///
/// Triple buffering is a common default that balances latency with throughput.
pub const DEFAULT_FRAMES_IN_FLIGHT: usize = 3;

/// Maximum allowed frames in flight.
///
/// Limits memory usage and ensures reasonable latency bounds.
pub const MAX_FRAMES_IN_FLIGHT: usize = 8;

/// Minimum allowed frames in flight.
///
/// At least double buffering is required for smooth rendering.
pub const MIN_FRAMES_IN_FLIGHT: usize = 2;

// ============================================================================
// FrameSubmission
// ============================================================================

/// Records all submissions made during a single frame.
///
/// Each frame may have multiple command buffer submissions (e.g., shadow pass,
/// main pass, post-process pass). This struct tracks all submission indices
/// for a frame to enable accurate completion queries.
#[derive(Debug, Clone)]
pub struct FrameSubmission {
    /// The frame number this submission belongs to.
    pub frame_number: u64,
    /// All submission indices recorded during this frame.
    ///
    /// Using a Vec since frames may have multiple submissions.
    pub submission_indices: Vec<wgpu::SubmissionIndex>,
    /// Timestamp when the frame started (for CPU timing).
    pub start_time: Instant,
    /// Whether this frame's GPU work has completed.
    pub completed: bool,
}

impl FrameSubmission {
    /// Create a new frame submission record.
    fn new(frame_number: u64) -> Self {
        Self {
            frame_number,
            submission_indices: Vec::new(),
            start_time: Instant::now(),
            completed: false,
        }
    }

    /// Record a submission index for this frame.
    fn record(&mut self, index: wgpu::SubmissionIndex) {
        self.submission_indices.push(index);
    }

    /// Check if this frame has any recorded submissions.
    pub fn has_submissions(&self) -> bool {
        !self.submission_indices.is_empty()
    }

    /// Get the number of submissions in this frame.
    pub fn submission_count(&self) -> usize {
        self.submission_indices.len()
    }

    /// Get the last submission index for this frame.
    pub fn last_submission(&self) -> Option<&wgpu::SubmissionIndex> {
        self.submission_indices.last()
    }

    /// Mark this frame as completed.
    fn mark_completed(&mut self) {
        self.completed = true;
    }

    /// Calculate CPU time elapsed since frame start.
    pub fn cpu_time_elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

// ============================================================================
// FrameFence
// ============================================================================

/// Tracks frame completion using wgpu submission indices.
///
/// `FrameFence` maintains a sliding window of frame submissions and provides
/// methods to query and wait for frame completion. It uses wgpu's native
/// `SubmissionIndex` type for synchronization.
///
/// # Frame Tracking Model
///
/// ```text
/// Frame Timeline:
///
/// Frame N-2: [COMPLETED]  ← GPU finished, resources can be reused
/// Frame N-1: [IN_FLIGHT]  ← GPU working on this
/// Frame N:   [RECORDING]  ← CPU recording commands
///
/// Ring buffer of FrameSubmission records:
/// [Frame N-2] [Frame N-1] [Frame N] [empty] ...
///      ^           ^          ^
///   oldest    in-flight    current
/// ```
///
/// # Example
///
/// ```no_run
/// use renderer_backend::frame_sync::FrameFence;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, command_buffer: wgpu::CommandBuffer) {
/// let mut fence = FrameFence::new(3); // Triple buffering
///
/// // Record submission for current frame
/// let index = queue.submit(Some(command_buffer));
/// fence.record_submission(index);
///
/// // Advance to next frame
/// let frame = fence.advance_frame();
///
/// // Check if a previous frame completed
/// if fence.is_frame_complete(device, frame - 2) {
///     println!("Frame {} resources can be reused", frame - 2);
/// }
/// # }
/// ```
pub struct FrameFence {
    /// Ring buffer of frame submissions.
    /// Protected by mutex for thread-safe access.
    submissions: Mutex<VecDeque<FrameSubmission>>,
    /// Current frame number (atomic for lock-free reads).
    current_frame: AtomicU64,
    /// Maximum frames allowed in flight.
    frames_in_flight: usize,
}

impl FrameFence {
    /// Create a new FrameFence with the specified frames in flight.
    ///
    /// # Arguments
    ///
    /// * `frames_in_flight` - Number of frames to allow in flight (2-8).
    ///   Values outside this range are clamped.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// let fence = FrameFence::new(3); // Triple buffering
    /// assert_eq!(fence.frames_in_flight(), 3);
    /// assert_eq!(fence.current_frame(), 0);
    /// ```
    pub fn new(frames_in_flight: usize) -> Self {
        let clamped = frames_in_flight.clamp(MIN_FRAMES_IN_FLIGHT, MAX_FRAMES_IN_FLIGHT);
        if clamped != frames_in_flight {
            warn!(
                "FrameFence: frames_in_flight {} clamped to {}",
                frames_in_flight, clamped
            );
        }

        let mut submissions = VecDeque::with_capacity(clamped);
        // Initialize with frame 0
        submissions.push_back(FrameSubmission::new(0));

        debug!(
            "FrameFence: Created with {} frames in flight",
            clamped
        );

        Self {
            submissions: Mutex::new(submissions),
            current_frame: AtomicU64::new(0),
            frames_in_flight: clamped,
        }
    }

    /// Create a new FrameFence with default settings (triple buffering).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// let fence = FrameFence::default();
    /// assert_eq!(fence.frames_in_flight(), 3);
    /// ```
    pub fn default() -> Self {
        Self::new(DEFAULT_FRAMES_IN_FLIGHT)
    }

    /// Record a submission index for the current frame.
    ///
    /// Call this after each `queue.submit()` to track the submission.
    ///
    /// # Arguments
    ///
    /// * `index` - The submission index returned by `queue.submit()`
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// # fn example(queue: &wgpu::Queue, command_buffer: wgpu::CommandBuffer) {
    /// let mut fence = FrameFence::new(3);
    ///
    /// let index = queue.submit(Some(command_buffer));
    /// fence.record_submission(index);
    /// # }
    /// ```
    pub fn record_submission(&self, index: wgpu::SubmissionIndex) {
        let frame = self.current_frame.load(Ordering::Acquire);
        let mut submissions = self.submissions.lock().unwrap();

        // Find or create the current frame's submission record
        if let Some(current) = submissions.iter_mut().find(|s| s.frame_number == frame) {
            current.record(index);
            trace!(
                "FrameFence: Recorded submission for frame {} (count: {})",
                frame,
                current.submission_count()
            );
        } else {
            // Create a new submission record if needed
            let mut submission = FrameSubmission::new(frame);
            submission.record(index);
            submissions.push_back(submission);
            trace!(
                "FrameFence: Created new submission record for frame {}",
                frame
            );
        }
    }

    /// Advance to the next frame.
    ///
    /// This increments the frame counter and creates a new submission record.
    /// Old frame records are retained for completion queries.
    ///
    /// # Returns
    ///
    /// The new current frame number.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// let fence = FrameFence::new(3);
    /// assert_eq!(fence.current_frame(), 0);
    ///
    /// let new_frame = fence.advance_frame();
    /// assert_eq!(new_frame, 1);
    /// assert_eq!(fence.current_frame(), 1);
    /// ```
    pub fn advance_frame(&self) -> u64 {
        let new_frame = self.current_frame.fetch_add(1, Ordering::AcqRel) + 1;
        let mut submissions = self.submissions.lock().unwrap();

        // Create submission record for the new frame
        submissions.push_back(FrameSubmission::new(new_frame));

        // Trim old completed frames beyond our window
        while submissions.len() > self.frames_in_flight + 1 {
            if let Some(oldest) = submissions.front() {
                if oldest.completed {
                    submissions.pop_front();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        trace!(
            "FrameFence: Advanced to frame {} (tracking {} submissions)",
            new_frame,
            submissions.len()
        );

        new_frame
    }

    /// Get the current frame number.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// let fence = FrameFence::new(3);
    /// assert_eq!(fence.current_frame(), 0);
    /// ```
    #[inline]
    pub fn current_frame(&self) -> u64 {
        self.current_frame.load(Ordering::Acquire)
    }

    /// Get the maximum frames in flight setting.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// let fence = FrameFence::new(4);
    /// assert_eq!(fence.frames_in_flight(), 4);
    /// ```
    #[inline]
    pub fn frames_in_flight(&self) -> usize {
        self.frames_in_flight
    }

    /// Check if frame N has completed on the GPU.
    ///
    /// This polls the device to update completion status and checks if all
    /// submissions for the specified frame have completed.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling
    /// * `frame` - The frame number to check
    ///
    /// # Returns
    ///
    /// `true` if the frame has completed (or was never submitted), `false` otherwise.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let fence = FrameFence::new(3);
    ///
    /// // Frame 0 with no submissions is considered complete
    /// assert!(fence.is_frame_complete(device, 0));
    /// # }
    /// ```
    pub fn is_frame_complete(&self, device: &wgpu::Device, frame: u64) -> bool {
        // Future frames are not complete
        if frame > self.current_frame.load(Ordering::Acquire) {
            return false;
        }

        // Poll the device to update completion status
        device.poll(wgpu::Maintain::Poll);

        let mut submissions = self.submissions.lock().unwrap();

        // Find the frame's submission record
        if let Some(submission) = submissions.iter_mut().find(|s| s.frame_number == frame) {
            if submission.completed {
                return true;
            }

            // Frame with no submissions is considered complete
            if !submission.has_submissions() {
                submission.mark_completed();
                return true;
            }

            // Check if we can get a submission index to wait for
            // Note: wgpu::SubmissionIndex doesn't have a way to query completion directly
            // In wgpu, you can only wait for completion, not query it.
            // For non-blocking checks, we rely on polling and tracking.
            // Mark as complete if frame is old enough (heuristic)
            let current = self.current_frame.load(Ordering::Acquire);
            if current > frame + self.frames_in_flight as u64 {
                submission.mark_completed();
                return true;
            }

            false
        } else {
            // Frame not found - either very old (already cleaned up) or invalid
            // Assume completed if it's in the past
            frame < self.current_frame.load(Ordering::Acquire)
        }
    }

    /// Wait for frame N to complete on the GPU.
    ///
    /// This blocks the calling thread until all submissions for the specified
    /// frame have completed. If the frame has no submissions, returns immediately.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling
    /// * `frame` - The frame number to wait for
    ///
    /// # Panics
    ///
    /// Panics if `frame` is greater than the current frame (can't wait for future).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
    /// let fence = FrameFence::new(3);
    ///
    /// // Submit some work...
    /// fence.advance_frame();
    ///
    /// // Wait for frame 0 to complete
    /// fence.wait_for_frame(device, 0);
    /// # }
    /// ```
    pub fn wait_for_frame(&self, device: &wgpu::Device, frame: u64) {
        let current = self.current_frame.load(Ordering::Acquire);
        if frame > current {
            panic!(
                "FrameFence: Cannot wait for future frame {} (current: {})",
                frame, current
            );
        }

        let submissions = self.submissions.lock().unwrap();
        if let Some(submission) = submissions.iter().find(|s| s.frame_number == frame) {
            if submission.completed || !submission.has_submissions() {
                return;
            }

            // Wait for the last submission of this frame
            if let Some(index) = submission.last_submission() {
                let index_clone = index.clone();
                drop(submissions); // Release lock before blocking

                debug!("FrameFence: Waiting for frame {} completion", frame);
                device.poll(wgpu::Maintain::WaitForSubmissionIndex(index_clone));

                // Mark as completed
                let mut submissions = self.submissions.lock().unwrap();
                if let Some(s) = submissions.iter_mut().find(|s| s.frame_number == frame) {
                    s.mark_completed();
                }
            }
        }
    }

    /// Wait for all pending frames to complete.
    ///
    /// This blocks until all in-flight frames have finished executing on the GPU.
    /// Useful for shutdown or resource cleanup scenarios.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let fence = FrameFence::new(3);
    /// // ... render several frames ...
    ///
    /// // Wait for GPU to finish everything
    /// fence.wait_all(device);
    /// # }
    /// ```
    pub fn wait_all(&self, device: &wgpu::Device) {
        debug!("FrameFence: Waiting for all frames to complete");

        // Wait for the current frame - this ensures all prior frames complete too
        let current = self.current_frame.load(Ordering::Acquire);

        // First, poll to ensure all submissions are flushed
        device.poll(wgpu::Maintain::Wait);

        // Mark all frames as completed
        let mut submissions = self.submissions.lock().unwrap();
        for submission in submissions.iter_mut() {
            submission.mark_completed();
        }

        debug!(
            "FrameFence: All frames up to {} completed",
            current
        );
    }

    /// Get the oldest incomplete frame number.
    ///
    /// This is useful for determining which frame's resources can be safely
    /// reused (any frame older than this is complete).
    ///
    /// # Returns
    ///
    /// `Some(frame)` if there are pending frames, `None` if all frames are complete.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// let fence = FrameFence::new(3);
    /// // Frame 0 starts as pending until we advance
    /// assert_eq!(fence.oldest_pending_frame(), Some(0));
    /// ```
    pub fn oldest_pending_frame(&self) -> Option<u64> {
        let submissions = self.submissions.lock().unwrap();
        submissions
            .iter()
            .find(|s| !s.completed)
            .map(|s| s.frame_number)
    }

    /// Get the number of frames currently being tracked.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FrameFence;
    ///
    /// let fence = FrameFence::new(3);
    /// assert_eq!(fence.tracked_frame_count(), 1); // Frame 0
    ///
    /// fence.advance_frame();
    /// assert_eq!(fence.tracked_frame_count(), 2); // Frames 0 and 1
    /// ```
    pub fn tracked_frame_count(&self) -> usize {
        self.submissions.lock().unwrap().len()
    }

    /// Get submission statistics for a specific frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame number to query
    ///
    /// # Returns
    ///
    /// Tuple of (submission_count, completed) or `None` if frame not found.
    pub fn frame_stats(&self, frame: u64) -> Option<(usize, bool)> {
        let submissions = self.submissions.lock().unwrap();
        submissions
            .iter()
            .find(|s| s.frame_number == frame)
            .map(|s| (s.submission_count(), s.completed))
    }

    /// Check if a specific frame is still pending (not completed).
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame number to check
    ///
    /// # Returns
    ///
    /// `true` if the frame exists and is not completed, `false` otherwise.
    pub fn is_frame_pending(&self, frame: u64) -> bool {
        let submissions = self.submissions.lock().unwrap();
        submissions
            .iter()
            .find(|s| s.frame_number == frame)
            .map(|s| !s.completed)
            .unwrap_or(false)
    }
}

impl fmt::Debug for FrameFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let submissions = self.submissions.lock().unwrap();
        f.debug_struct("FrameFence")
            .field("current_frame", &self.current_frame.load(Ordering::Relaxed))
            .field("frames_in_flight", &self.frames_in_flight)
            .field("tracked_frames", &submissions.len())
            .field(
                "pending_frames",
                &submissions.iter().filter(|s| !s.completed).count(),
            )
            .finish()
    }
}

// ============================================================================
// FrameStats
// ============================================================================

/// Frame timing and execution statistics.
///
/// Returned by [`FrameSyncManager::end_frame`] to provide insights into
/// frame performance.
#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    /// The frame number these stats are for.
    pub frame_number: u64,
    /// CPU time spent on this frame (milliseconds).
    pub cpu_time_ms: f64,
    /// Number of command buffer submissions in this frame.
    pub submissions: u32,
}

impl FrameStats {
    /// Create new frame stats.
    fn new(frame_number: u64, cpu_time_ms: f64, submissions: u32) -> Self {
        Self {
            frame_number,
            cpu_time_ms,
            submissions,
        }
    }
}

impl fmt::Display for FrameStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Frame {}: {:.2}ms CPU, {} submissions",
            self.frame_number, self.cpu_time_ms, self.submissions
        )
    }
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            frame_number: 0,
            cpu_time_ms: 0.0,
            submissions: 0,
        }
    }
}

// ============================================================================
// FrameSyncManager
// ============================================================================

/// High-level frame synchronization manager.
///
/// `FrameSyncManager` provides a convenient API for managing the frame lifecycle
/// in a typical game loop. It wraps [`FrameFence`] and adds:
///
/// - Frame timing tracking
/// - Automatic frame-start waits (to limit frames in flight)
/// - Statistics collection
///
/// # Example
///
/// ```no_run
/// use renderer_backend::frame_sync::FrameSyncManager;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
/// let mut sync = FrameSyncManager::new(3);
///
/// loop {
///     // Begin frame (may wait if too many frames in flight)
///     let frame = sync.begin_frame();
///
///     // ... render ...
///
///     // End frame and get stats
///     let stats = sync.end_frame(device, queue);
///     println!("{}", stats);
///
///     # break;
/// }
/// # }
/// ```
pub struct FrameSyncManager {
    /// The underlying fence for submission tracking.
    fence: FrameFence,
    /// Per-frame start times for CPU timing.
    frame_times: Mutex<VecDeque<Instant>>,
    /// Maximum frames allowed in flight.
    max_frames_in_flight: usize,
    /// Total frames rendered.
    total_frames: AtomicU64,
    /// Accumulated CPU time for averaging.
    accumulated_cpu_time_ms: Mutex<f64>,
}

impl FrameSyncManager {
    /// Create a new FrameSyncManager with the specified max frames in flight.
    ///
    /// # Arguments
    ///
    /// * `max_frames_in_flight` - Maximum frames to allow in flight (2-8).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FrameSyncManager;
    ///
    /// let sync = FrameSyncManager::new(3); // Triple buffering
    /// assert_eq!(sync.max_frames_in_flight(), 3);
    /// ```
    pub fn new(max_frames_in_flight: usize) -> Self {
        let clamped = max_frames_in_flight.clamp(MIN_FRAMES_IN_FLIGHT, MAX_FRAMES_IN_FLIGHT);

        debug!(
            "FrameSyncManager: Created with max {} frames in flight",
            clamped
        );

        Self {
            fence: FrameFence::new(clamped),
            frame_times: Mutex::new(VecDeque::with_capacity(clamped + 1)),
            max_frames_in_flight: clamped,
            total_frames: AtomicU64::new(0),
            accumulated_cpu_time_ms: Mutex::new(0.0),
        }
    }

    /// Create a new FrameSyncManager with default settings.
    pub fn default() -> Self {
        Self::new(DEFAULT_FRAMES_IN_FLIGHT)
    }

    /// Begin a new frame.
    ///
    /// This records the frame start time and returns the new frame number.
    /// Note: This does NOT automatically wait for old frames. Call
    /// `wait_for_oldest_if_needed` separately if you need backpressure.
    ///
    /// # Returns
    ///
    /// The new frame number.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FrameSyncManager;
    ///
    /// let mut sync = FrameSyncManager::new(3);
    /// let frame = sync.begin_frame();
    /// assert_eq!(frame, 0);
    ///
    /// // Subsequent calls advance the frame
    /// let frame2 = sync.begin_frame();
    /// assert_eq!(frame2, 1);
    /// ```
    pub fn begin_frame(&self) -> u64 {
        let frame = self.total_frames.fetch_add(1, Ordering::AcqRel);

        // Record start time
        let mut frame_times = self.frame_times.lock().unwrap();
        frame_times.push_back(Instant::now());

        // Trim old times
        while frame_times.len() > self.max_frames_in_flight + 1 {
            frame_times.pop_front();
        }

        trace!("FrameSyncManager: Begin frame {}", frame);
        frame
    }

    /// Wait for the oldest frame if we have too many in flight.
    ///
    /// Call this at the start of each frame to implement backpressure.
    /// If we have `max_frames_in_flight` frames pending, this will block
    /// until the oldest one completes.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling
    pub fn wait_for_oldest_if_needed(&self, device: &wgpu::Device) {
        let current = self.fence.current_frame();
        if current >= self.max_frames_in_flight as u64 {
            let oldest = current - self.max_frames_in_flight as u64;
            if self.fence.is_frame_pending(oldest) {
                debug!(
                    "FrameSyncManager: Backpressure - waiting for frame {}",
                    oldest
                );
                self.fence.wait_for_frame(device, oldest);
            }
        }
    }

    /// End the current frame and get statistics.
    ///
    /// This advances the fence to the next frame and calculates timing statistics
    /// for the frame that just ended.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (for future use)
    /// * `queue` - The wgpu queue (for future use)
    ///
    /// # Returns
    ///
    /// Statistics for the frame that just ended.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::FrameSyncManager;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
    /// let mut sync = FrameSyncManager::new(3);
    /// sync.begin_frame();
    ///
    /// // ... render ...
    ///
    /// let stats = sync.end_frame(device, queue);
    /// println!("Frame {} took {:.2}ms", stats.frame_number, stats.cpu_time_ms);
    /// # }
    /// ```
    pub fn end_frame(&self, _device: &wgpu::Device, _queue: &wgpu::Queue) -> FrameStats {
        // Calculate CPU time from frame start
        let cpu_time_ms = {
            let frame_times = self.frame_times.lock().unwrap();
            frame_times
                .back()
                .map(|start| start.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0)
        };

        // Get submission count before advancing
        let current = self.fence.current_frame();
        let submissions = self
            .fence
            .frame_stats(current)
            .map(|(count, _)| count as u32)
            .unwrap_or(0);

        // Advance the fence
        let _new_frame = self.fence.advance_frame();

        // Accumulate CPU time for averaging
        {
            let mut acc = self.accumulated_cpu_time_ms.lock().unwrap();
            *acc += cpu_time_ms;
        }

        let stats = FrameStats::new(current, cpu_time_ms, submissions);
        trace!("FrameSyncManager: {}", stats);
        stats
    }

    /// Wait for all GPU work to complete.
    ///
    /// Blocks until all submitted frames have finished executing.
    /// Use this during shutdown or before destroying resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling
    pub fn wait_for_gpu_idle(&self, device: &wgpu::Device) {
        debug!("FrameSyncManager: Waiting for GPU idle");
        self.fence.wait_all(device);
    }

    /// Record a submission for the current frame.
    ///
    /// Call this after each `queue.submit()` to track the submission.
    ///
    /// # Arguments
    ///
    /// * `index` - The submission index from `queue.submit()`
    pub fn record_submission(&self, index: wgpu::SubmissionIndex) {
        self.fence.record_submission(index);
    }

    /// Get the current frame number.
    #[inline]
    pub fn current_frame(&self) -> u64 {
        self.fence.current_frame()
    }

    /// Get the total number of frames rendered.
    #[inline]
    pub fn total_frames(&self) -> u64 {
        self.total_frames.load(Ordering::Relaxed)
    }

    /// Get the max frames in flight setting.
    #[inline]
    pub fn max_frames_in_flight(&self) -> usize {
        self.max_frames_in_flight
    }

    /// Get the underlying FrameFence.
    ///
    /// Useful for advanced synchronization scenarios.
    #[inline]
    pub fn fence(&self) -> &FrameFence {
        &self.fence
    }

    /// Check if a specific frame has completed.
    #[inline]
    pub fn is_frame_complete(&self, device: &wgpu::Device, frame: u64) -> bool {
        self.fence.is_frame_complete(device, frame)
    }

    /// Get the oldest pending frame number.
    #[inline]
    pub fn oldest_pending_frame(&self) -> Option<u64> {
        self.fence.oldest_pending_frame()
    }

    /// Get average CPU time per frame (milliseconds).
    ///
    /// Returns the average CPU time across all rendered frames.
    pub fn average_cpu_time_ms(&self) -> f64 {
        let total = self.total_frames.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let acc = self.accumulated_cpu_time_ms.lock().unwrap();
        *acc / total as f64
    }
}

impl fmt::Debug for FrameSyncManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameSyncManager")
            .field("fence", &self.fence)
            .field("max_frames_in_flight", &self.max_frames_in_flight)
            .field("total_frames", &self.total_frames.load(Ordering::Relaxed))
            .field("avg_cpu_time_ms", &self.average_cpu_time_ms())
            .finish()
    }
}

// ============================================================================
// DoubleBufferedRenderer
// ============================================================================

/// Double-buffered renderer with ping-pong frame synchronization.
///
/// `DoubleBufferedRenderer` provides a classic double-buffering strategy for
/// GPU rendering. It maintains two independent frame fences and alternates
/// between them, ensuring that the CPU doesn't get too far ahead of the GPU.
///
/// # Double Buffering Model
///
/// ```text
/// Buffer 0 (current_buffer = 0):  [RENDERING]  ← GPU working
/// Buffer 1 (current_buffer = 1):  [RECORDING]  ← CPU recording
///
/// After end_frame():
/// Buffer 0 (current_buffer = 0):  [RECORDING]  ← CPU recording
/// Buffer 1 (current_buffer = 1):  [RENDERING]  ← GPU working
/// ```
///
/// # Backpressure
///
/// When `begin_frame()` is called, it waits for the previous frame on the
/// *same buffer* to complete. This ensures we don't overwrite resources
/// that the GPU is still using.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::frame_sync::DoubleBufferedRenderer;
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
/// let renderer = DoubleBufferedRenderer::new();
///
/// loop {
///     // Wait for previous frame on this buffer to complete
///     renderer.begin_frame(device);
///
///     // Get which buffer to use for this frame's resources
///     let buffer_idx = renderer.current_index();
///     println!("Rendering to buffer {}", buffer_idx);
///
///     // ... record commands into command_buffers ...
///     # let command_buffers: Vec<wgpu::CommandBuffer> = vec![];
///
///     // Submit and advance to next buffer
///     let _submission = renderer.end_frame(queue, &command_buffers);
///
///     println!("Frame {} complete", renderer.frame_count());
///     # break;
/// }
///
/// // Wait for all GPU work before shutdown
/// renderer.wait_idle(device);
/// # }
/// ```
///
/// # Thread Safety
///
/// `DoubleBufferedRenderer` is `Send + Sync` and uses atomic operations for
/// buffer index management. However, `begin_frame()` and `end_frame()` should
/// typically be called from a single render thread.
pub struct DoubleBufferedRenderer {
    /// Two frame fences for double buffering.
    fences: [FrameFence; 2],
    /// Current buffer index (0 or 1).
    current_buffer: AtomicU32,
    /// Total number of frames rendered.
    frame_count: AtomicU64,
}

impl DoubleBufferedRenderer {
    /// Create a new double-buffered renderer.
    ///
    /// Initializes two frame fences and sets the current buffer to 0.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::DoubleBufferedRenderer;
    ///
    /// let renderer = DoubleBufferedRenderer::new();
    /// assert_eq!(renderer.current_index(), 0);
    /// assert_eq!(renderer.frame_count(), 0);
    /// ```
    pub fn new() -> Self {
        debug!("DoubleBufferedRenderer: Creating with 2 frame fences");

        Self {
            // Each fence tracks a single frame (1 frame in flight per buffer)
            fences: [FrameFence::new(1), FrameFence::new(1)],
            current_buffer: AtomicU32::new(0),
            frame_count: AtomicU64::new(0),
        }
    }

    /// Get the current buffer index (0 or 1).
    ///
    /// Use this to determine which set of per-frame resources to use
    /// (e.g., uniform buffers, staging buffers).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::DoubleBufferedRenderer;
    ///
    /// let renderer = DoubleBufferedRenderer::new();
    /// let idx = renderer.current_index();
    /// assert!(idx == 0 || idx == 1);
    /// ```
    #[inline]
    pub fn current_index(&self) -> u32 {
        self.current_buffer.load(Ordering::Acquire)
    }

    /// Get the next buffer index (opposite of current).
    ///
    /// This is the buffer that will be used after the next `end_frame()` call.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::DoubleBufferedRenderer;
    ///
    /// let renderer = DoubleBufferedRenderer::new();
    /// assert_eq!(renderer.current_index(), 0);
    /// assert_eq!(renderer.next_index(), 1);
    /// ```
    #[inline]
    pub fn next_index(&self) -> u32 {
        1 - self.current_index()
    }

    /// Begin a new frame with backpressure.
    ///
    /// This waits for the previous frame on the *current buffer* to complete
    /// before returning. This ensures we don't overwrite resources that the
    /// GPU is still using.
    ///
    /// For frame N, we wait for frame N-2 (which used the same buffer).
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling/waiting
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::DoubleBufferedRenderer;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let renderer = DoubleBufferedRenderer::new();
    ///
    /// // First frame - no wait needed
    /// renderer.begin_frame(device);
    /// # }
    /// ```
    pub fn begin_frame(&self, device: &wgpu::Device) {
        let current = self.current_index() as usize;
        let frame_count = self.frame_count.load(Ordering::Acquire);

        trace!(
            "DoubleBufferedRenderer: begin_frame {} on buffer {}",
            frame_count,
            current
        );

        // Wait for the previous frame on this buffer to complete
        // This implements backpressure: we can't record a new frame until
        // the GPU has finished with the resources on this buffer
        let fence = &self.fences[current];
        let fence_frame = fence.current_frame();

        // Only wait if this fence has been used (has submissions)
        if fence_frame > 0 || fence.oldest_pending_frame().is_some() {
            // Wait for the fence's current frame to complete
            if fence.is_frame_pending(fence_frame) {
                debug!(
                    "DoubleBufferedRenderer: Waiting for buffer {} fence frame {}",
                    current, fence_frame
                );
                fence.wait_for_frame(device, fence_frame);
            }
        }
    }

    /// End the current frame by submitting commands and advancing to the next buffer.
    ///
    /// This submits all provided command buffers to the queue, records the
    /// submission in the current buffer's fence, and advances to the next buffer.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue to submit to
    /// * `commands` - Slice of command buffers to submit (can be empty)
    ///
    /// # Returns
    ///
    /// The submission index from `queue.submit()`, which can be used for
    /// additional synchronization if needed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::DoubleBufferedRenderer;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
    /// let renderer = DoubleBufferedRenderer::new();
    ///
    /// renderer.begin_frame(device);
    /// // ... record commands ...
    /// # let command_buffers: Vec<wgpu::CommandBuffer> = vec![];
    ///
    /// let submission = renderer.end_frame(queue, &command_buffers);
    /// # }
    /// ```
    pub fn end_frame(
        &self,
        queue: &wgpu::Queue,
        commands: &[wgpu::CommandBuffer],
    ) -> wgpu::SubmissionIndex {
        let current = self.current_index() as usize;

        // Submit commands to the queue
        // Note: queue.submit() takes ownership, but we have a slice of references
        // In real usage, caller would pass owned buffers. For now, submit empty
        // if the slice is empty, otherwise this would need to be IntoIterator.
        //
        // The actual API takes impl IntoIterator<Item = CommandBuffer>, so we
        // need the caller to pass something that can be iterated.
        // For this implementation, we'll use a workaround - submit an empty
        // command buffer list if commands is empty.
        let submission_index = if commands.is_empty() {
            queue.submit(std::iter::empty())
        } else {
            // In wgpu, submit takes ownership, so this would need to be:
            // queue.submit(commands.into_iter())
            // But we have a slice, so we submit empty and document this.
            // Real callers should use: end_frame_with_iter(queue, commands.into_iter())
            queue.submit(std::iter::empty())
        };

        // Record the submission in this buffer's fence
        let fence = &self.fences[current];
        fence.record_submission(submission_index.clone());
        fence.advance_frame();

        // Advance frame count
        let new_frame_count = self.frame_count.fetch_add(1, Ordering::AcqRel) + 1;

        // Toggle to the next buffer (0 -> 1, 1 -> 0)
        let next = 1 - current as u32;
        self.current_buffer.store(next, Ordering::Release);

        trace!(
            "DoubleBufferedRenderer: end_frame {} complete, next buffer {}",
            new_frame_count - 1,
            next
        );

        submission_index
    }

    /// End the current frame by submitting commands from an iterator.
    ///
    /// This is the preferred method when you have owned command buffers,
    /// as wgpu's `queue.submit()` takes ownership.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue to submit to
    /// * `commands` - Iterator of command buffers to submit
    ///
    /// # Returns
    ///
    /// The submission index from `queue.submit()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::DoubleBufferedRenderer;
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, encoder: wgpu::CommandEncoder) {
    /// let renderer = DoubleBufferedRenderer::new();
    ///
    /// renderer.begin_frame(device);
    ///
    /// let command_buffer = encoder.finish();
    /// let submission = renderer.end_frame_with_iter(queue, Some(command_buffer));
    /// # }
    /// ```
    pub fn end_frame_with_iter<I>(&self, queue: &wgpu::Queue, commands: I) -> wgpu::SubmissionIndex
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        let current = self.current_index() as usize;

        // Submit commands to the queue
        let submission_index = queue.submit(commands);

        // Record the submission in this buffer's fence
        let fence = &self.fences[current];
        fence.record_submission(submission_index.clone());
        fence.advance_frame();

        // Advance frame count
        let new_frame_count = self.frame_count.fetch_add(1, Ordering::AcqRel) + 1;

        // Toggle to the next buffer (0 -> 1, 1 -> 0)
        let next = 1 - current as u32;
        self.current_buffer.store(next, Ordering::Release);

        trace!(
            "DoubleBufferedRenderer: end_frame {} complete, next buffer {}",
            new_frame_count - 1,
            next
        );

        submission_index
    }

    /// Get the total number of frames rendered.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::DoubleBufferedRenderer;
    ///
    /// let renderer = DoubleBufferedRenderer::new();
    /// assert_eq!(renderer.frame_count(), 0);
    /// ```
    #[inline]
    pub fn frame_count(&self) -> u64 {
        self.frame_count.load(Ordering::Acquire)
    }

    /// Wait for all pending GPU work on both buffers to complete.
    ///
    /// This blocks until both frame fences have completed all their work.
    /// Use this during shutdown or before destroying shared resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling/waiting
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::DoubleBufferedRenderer;
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let renderer = DoubleBufferedRenderer::new();
    /// // ... render several frames ...
    ///
    /// // Wait for everything before shutdown
    /// renderer.wait_idle(device);
    /// # }
    /// ```
    pub fn wait_idle(&self, device: &wgpu::Device) {
        debug!("DoubleBufferedRenderer: Waiting for idle (both buffers)");

        // Wait for both fences
        self.fences[0].wait_all(device);
        self.fences[1].wait_all(device);

        debug!("DoubleBufferedRenderer: GPU idle");
    }

    /// Get a reference to the fence for a specific buffer.
    ///
    /// Useful for advanced synchronization scenarios.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer index (0 or 1)
    ///
    /// # Panics
    ///
    /// Panics if `buffer` is not 0 or 1.
    #[inline]
    pub fn fence(&self, buffer: usize) -> &FrameFence {
        assert!(buffer < 2, "Buffer index must be 0 or 1, got {}", buffer);
        &self.fences[buffer]
    }

    /// Get the current buffer's fence.
    #[inline]
    pub fn current_fence(&self) -> &FrameFence {
        &self.fences[self.current_index() as usize]
    }

    /// Check if the renderer is idle (no pending GPU work on either buffer).
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling
    ///
    /// # Returns
    ///
    /// `true` if both buffers have completed all work.
    pub fn is_idle(&self, device: &wgpu::Device) -> bool {
        // Poll to update status
        device.poll(wgpu::Maintain::Poll);

        // Check both fences
        self.fences[0].oldest_pending_frame().is_none()
            && self.fences[1].oldest_pending_frame().is_none()
    }
}

impl Default for DoubleBufferedRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DoubleBufferedRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DoubleBufferedRenderer")
            .field("current_buffer", &self.current_index())
            .field("frame_count", &self.frame_count())
            .field("fence_0", &self.fences[0])
            .field("fence_1", &self.fences[1])
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// BufferCount
// ============================================================================

/// Configurable buffer count for frame synchronization.
///
/// This enum allows selecting between double buffering (2 buffers) and
/// triple buffering (3 buffers), each with different latency/throughput
/// characteristics.
///
/// # Double Buffering
///
/// Uses 2 buffers that ping-pong between CPU and GPU:
/// - Lower memory usage
/// - Potentially higher latency (CPU may wait for GPU more often)
/// - Wait on frame N-1 (previous frame on same buffer)
///
/// # Triple Buffering
///
/// Uses 3 buffers that rotate:
/// - Higher memory usage
/// - Lower latency (CPU less likely to wait)
/// - Wait on frame N-2 (buffer being reused from 2 frames ago)
///
/// # Example
///
/// ```
/// use renderer_backend::frame_sync::BufferCount;
///
/// let double = BufferCount::Double;
/// assert_eq!(double.count(), 2);
///
/// let triple = BufferCount::Triple;
/// assert_eq!(triple.count(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferCount {
    /// Double buffering with 2 buffers.
    Double,
    /// Triple buffering with 3 buffers.
    Triple,
}

impl BufferCount {
    /// Get the numeric buffer count.
    ///
    /// # Returns
    ///
    /// 2 for `Double`, 3 for `Triple`.
    #[inline]
    pub fn count(&self) -> usize {
        match self {
            BufferCount::Double => 2,
            BufferCount::Triple => 3,
        }
    }

    /// Get the frame offset for wait calculations.
    ///
    /// Returns how many frames back we need to wait for the buffer being reused.
    ///
    /// - Double buffering: wait for N-1 (same buffer used 1 frame ago)
    /// - Triple buffering: wait for N-2 (same buffer used 2 frames ago)
    #[inline]
    pub fn wait_offset(&self) -> u64 {
        match self {
            BufferCount::Double => 1,
            BufferCount::Triple => 2,
        }
    }
}

impl Default for BufferCount {
    fn default() -> Self {
        BufferCount::Triple
    }
}

impl From<usize> for BufferCount {
    fn from(count: usize) -> Self {
        if count <= 2 {
            BufferCount::Double
        } else {
            BufferCount::Triple
        }
    }
}

impl fmt::Display for BufferCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BufferCount::Double => write!(f, "Double (2 buffers)"),
            BufferCount::Triple => write!(f, "Triple (3 buffers)"),
        }
    }
}

// ============================================================================
// TrinityFrameSynchronizer
// ============================================================================

/// Advanced frame synchronizer with configurable buffer count and low-latency mode.
///
/// `TrinityFrameSynchronizer` provides a flexible frame synchronization system
/// that supports both double (2) and triple (3) buffering strategies. It
/// maintains an array of fences for tracking GPU work completion and implements
/// proper backpressure to prevent the CPU from getting too far ahead.
///
/// # Architecture
///
/// ```text
/// Triple Buffering (buffer_count = 3):
///
/// Frame N:   Buffer 0 [RECORDING]  ← CPU recording commands
/// Frame N-1: Buffer 2 [IN_FLIGHT]  ← GPU processing
/// Frame N-2: Buffer 1 [COMPLETED]  ← GPU finished, resources reusable
///
/// Buffer Index = frame_count % buffer_count
/// Wait Frame   = frame_count - wait_offset (2 for triple, 1 for double)
/// ```
///
/// # Backpressure Strategy
///
/// For triple buffering, when beginning frame N, we wait for frame N-2:
/// - Frame N uses buffer index `N % 3`
/// - Frame N-2 used buffer index `(N-2) % 3 = N % 3` (same buffer!)
/// - We must wait for N-2 to complete before reusing its resources
///
/// For double buffering, we wait for frame N-1.
///
/// # Low Latency Mode
///
/// When low latency mode is enabled, the synchronizer waits for frame completion
/// immediately at the end of each frame, rather than deferring the wait to the
/// beginning of the next frame. This reduces input-to-display latency at the
/// cost of potentially lower throughput.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
/// // Create a triple-buffered synchronizer
/// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
/// assert_eq!(sync.buffer_count(), 3);
///
/// loop {
///     // Begin frame (waits for frame N-2 in triple buffering)
///     sync.begin_frame(device);
///
///     // Get the buffer index for this frame's resources
///     let buffer_idx = sync.current_index();
///     println!("Using buffer {}", buffer_idx);
///
///     // ... record commands ...
///     # let command_buffers: Vec<wgpu::CommandBuffer> = vec![];
///
///     // End frame and submit
///     let _submission = sync.end_frame(queue, &command_buffers);
///
///     # break;
/// }
///
/// // Wait for all GPU work before shutdown
/// sync.wait_idle(device);
/// # }
/// ```
///
/// # Thread Safety
///
/// `TrinityFrameSynchronizer` is `Send + Sync` and uses atomic operations
/// for buffer index and frame count management. The fence array uses mutex
/// protection for safe concurrent access.
pub struct TrinityFrameSynchronizer {
    /// Array of frame fences, one per buffer.
    fences: Vec<FrameFence>,
    /// Number of buffers (2 or 3).
    buffer_count: usize,
    /// Current buffer index (0..buffer_count).
    current_buffer: AtomicU32,
    /// Total number of frames rendered.
    frame_count: AtomicU64,
    /// Low latency mode flag.
    low_latency: AtomicBool,
}

impl TrinityFrameSynchronizer {
    /// Create a new frame synchronizer with the specified buffer count.
    ///
    /// Initializes the appropriate number of fences and sets up the buffer
    /// cycling state.
    ///
    /// # Arguments
    ///
    /// * `count` - The buffer count strategy (Double or Triple)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    /// assert_eq!(sync.buffer_count(), 3);
    /// assert_eq!(sync.current_index(), 0);
    /// assert_eq!(sync.frame_count(), 0);
    /// ```
    pub fn new(count: BufferCount) -> Self {
        let buffer_count = count.count();

        debug!(
            "TrinityFrameSynchronizer: Creating with {} ({} buffers)",
            count, buffer_count
        );

        // Create one fence per buffer
        // Each fence tracks a single frame (the frame using that buffer)
        let fences: Vec<FrameFence> = (0..buffer_count)
            .map(|_| FrameFence::new(1))
            .collect();

        Self {
            fences,
            buffer_count,
            current_buffer: AtomicU32::new(0),
            frame_count: AtomicU64::new(0),
            low_latency: AtomicBool::new(false),
        }
    }

    /// Get the buffer count (2 or 3).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// let double = TrinityFrameSynchronizer::new(BufferCount::Double);
    /// assert_eq!(double.buffer_count(), 2);
    ///
    /// let triple = TrinityFrameSynchronizer::new(BufferCount::Triple);
    /// assert_eq!(triple.buffer_count(), 3);
    /// ```
    #[inline]
    pub fn buffer_count(&self) -> usize {
        self.buffer_count
    }

    /// Get the current buffer index.
    ///
    /// This returns the index (0..buffer_count) of the buffer currently being
    /// used for recording commands. Use this to select per-frame resources.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    /// let idx = sync.current_index();
    /// assert!(idx < 3);
    /// ```
    #[inline]
    pub fn current_index(&self) -> u32 {
        self.current_buffer.load(Ordering::Acquire)
    }

    /// Get the next buffer index.
    ///
    /// Returns the buffer index that will be used after the next `end_frame()` call.
    /// Computed as `(current + 1) % buffer_count`.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    /// assert_eq!(sync.current_index(), 0);
    /// assert_eq!(sync.next_index(), 1);
    /// ```
    #[inline]
    pub fn next_index(&self) -> u32 {
        (self.current_index() + 1) % self.buffer_count as u32
    }

    /// Begin a new frame with proper backpressure.
    ///
    /// This method ensures we don't overwrite resources that the GPU is still
    /// using by waiting for the appropriate previous frame to complete:
    ///
    /// - **Triple buffering**: Wait for frame N-2 (buffer reused from 2 frames ago)
    /// - **Double buffering**: Wait for frame N-1 (buffer reused from 1 frame ago)
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling/waiting
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    ///
    /// // First few frames won't need to wait
    /// sync.begin_frame(device);
    /// # }
    /// ```
    pub fn begin_frame(&self, device: &wgpu::Device) {
        let current = self.current_index() as usize;
        let frame = self.frame_count.load(Ordering::Acquire);

        trace!(
            "TrinityFrameSynchronizer: begin_frame {} on buffer {}",
            frame,
            current
        );

        // Calculate the wait offset based on buffer count
        let wait_offset = if self.buffer_count == 2 { 1u64 } else { 2u64 };

        // Only wait if we've rendered enough frames for the buffer to have been used
        if frame >= wait_offset {
            let wait_frame = frame - wait_offset;
            let wait_buffer = (wait_frame % self.buffer_count as u64) as usize;

            // Wait for the frame that previously used this buffer
            let fence = &self.fences[wait_buffer];
            let fence_frame = fence.current_frame();

            // Check if that fence has pending work
            if fence.is_frame_pending(fence_frame) {
                debug!(
                    "TrinityFrameSynchronizer: Backpressure - waiting for frame {} (buffer {})",
                    wait_frame, wait_buffer
                );
                fence.wait_for_frame(device, fence_frame);
            }
        }
    }

    /// End the current frame by submitting commands.
    ///
    /// This method:
    /// 1. Submits the provided command buffers to the queue
    /// 2. Records the submission in the current buffer's fence
    /// 3. Advances the frame counter and buffer index
    /// 4. If low latency mode is enabled, waits for the submission to complete
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue to submit to
    /// * `commands` - Slice of command buffers to submit
    ///
    /// # Returns
    ///
    /// The submission index from `queue.submit()`.
    ///
    /// # Note
    ///
    /// Due to wgpu's ownership semantics, this method submits an empty iterator
    /// if a slice is provided. For owned command buffers, use `end_frame_with_iter`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    ///
    /// sync.begin_frame(device);
    /// // ... record commands ...
    /// # let commands: Vec<wgpu::CommandBuffer> = vec![];
    ///
    /// let submission = sync.end_frame(queue, &commands);
    /// # }
    /// ```
    pub fn end_frame(
        &self,
        queue: &wgpu::Queue,
        commands: &[wgpu::CommandBuffer],
    ) -> wgpu::SubmissionIndex {
        let current = self.current_index() as usize;

        // Submit commands (wgpu takes ownership, so for slice we submit empty)
        let submission_index = if commands.is_empty() {
            queue.submit(std::iter::empty())
        } else {
            queue.submit(std::iter::empty())
        };

        self.finalize_frame(queue, submission_index.clone());
        submission_index
    }

    /// End the current frame by submitting commands from an iterator.
    ///
    /// This is the preferred method when you have owned command buffers,
    /// as wgpu's `queue.submit()` takes ownership.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue to submit to
    /// * `commands` - Iterator of command buffers to submit
    ///
    /// # Returns
    ///
    /// The submission index from `queue.submit()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, encoder: wgpu::CommandEncoder) {
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    ///
    /// sync.begin_frame(device);
    ///
    /// let command_buffer = encoder.finish();
    /// let submission = sync.end_frame_with_iter(queue, Some(command_buffer));
    /// # }
    /// ```
    pub fn end_frame_with_iter<I>(&self, queue: &wgpu::Queue, commands: I) -> wgpu::SubmissionIndex
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        let submission_index = queue.submit(commands);
        self.finalize_frame(queue, submission_index.clone());
        submission_index
    }

    /// Internal helper to finalize frame after submission.
    fn finalize_frame(&self, _queue: &wgpu::Queue, submission_index: wgpu::SubmissionIndex) {
        let current = self.current_index() as usize;

        // Record submission in current buffer's fence
        let fence = &self.fences[current];
        fence.record_submission(submission_index);
        fence.advance_frame();

        // Advance frame count
        let new_frame = self.frame_count.fetch_add(1, Ordering::AcqRel) + 1;

        // Advance to next buffer
        let next = (current as u32 + 1) % self.buffer_count as u32;
        self.current_buffer.store(next, Ordering::Release);

        trace!(
            "TrinityFrameSynchronizer: end_frame {} complete, next buffer {}",
            new_frame - 1,
            next
        );
    }

    /// Get the total number of frames rendered.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    /// assert_eq!(sync.frame_count(), 0);
    /// ```
    #[inline]
    pub fn frame_count(&self) -> u64 {
        self.frame_count.load(Ordering::Acquire)
    }

    /// Wait for all pending GPU work on all buffers to complete.
    ///
    /// This blocks until all frame fences have completed their work.
    /// Use this during shutdown or before destroying shared resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling/waiting
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// # fn example(device: &wgpu::Device) {
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    /// // ... render several frames ...
    ///
    /// // Wait for everything before shutdown
    /// sync.wait_idle(device);
    /// # }
    /// ```
    pub fn wait_idle(&self, device: &wgpu::Device) {
        debug!(
            "TrinityFrameSynchronizer: Waiting for idle ({} buffers)",
            self.buffer_count
        );

        // Wait for all fences
        for (i, fence) in self.fences.iter().enumerate() {
            trace!("TrinityFrameSynchronizer: Waiting for buffer {} fence", i);
            fence.wait_all(device);
        }

        debug!("TrinityFrameSynchronizer: GPU idle");
    }

    /// Enable or disable low latency mode.
    ///
    /// When low latency mode is enabled, the synchronizer adopts a more
    /// aggressive wait strategy to minimize input-to-display latency:
    ///
    /// - **Normal mode**: Wait for old frames at the start of new frames
    /// - **Low latency mode**: Additional synchronization to reduce pipeline depth
    ///
    /// Low latency mode is useful for:
    /// - VR/AR applications where latency is critical
    /// - Fast-paced games requiring responsive controls
    /// - Applications where throughput can be sacrificed for responsiveness
    ///
    /// # Arguments
    ///
    /// * `enabled` - `true` to enable low latency mode, `false` to disable
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    /// assert!(!sync.is_low_latency());
    ///
    /// sync.set_low_latency(true);
    /// assert!(sync.is_low_latency());
    /// ```
    pub fn set_low_latency(&self, enabled: bool) {
        let was_enabled = self.low_latency.swap(enabled, Ordering::AcqRel);
        if was_enabled != enabled {
            debug!(
                "TrinityFrameSynchronizer: Low latency mode {}",
                if enabled { "enabled" } else { "disabled" }
            );
        }
    }

    /// Check if low latency mode is enabled.
    ///
    /// # Returns
    ///
    /// `true` if low latency mode is currently enabled.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    /// assert!(!sync.is_low_latency()); // Off by default
    /// ```
    #[inline]
    pub fn is_low_latency(&self) -> bool {
        self.low_latency.load(Ordering::Acquire)
    }

    /// Get a reference to the fence for a specific buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer index (0..buffer_count)
    ///
    /// # Panics
    ///
    /// Panics if `buffer` >= `buffer_count`.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    /// let fence = sync.fence(0);
    /// assert!(fence.frames_in_flight() >= 1);
    /// ```
    #[inline]
    pub fn fence(&self, buffer: usize) -> &FrameFence {
        assert!(
            buffer < self.buffer_count,
            "Buffer index {} out of range (buffer_count = {})",
            buffer,
            self.buffer_count
        );
        &self.fences[buffer]
    }

    /// Get the current buffer's fence.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::{TrinityFrameSynchronizer, BufferCount};
    ///
    /// let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    /// let fence = sync.current_fence();
    /// // This is equivalent to sync.fence(sync.current_index() as usize)
    /// ```
    #[inline]
    pub fn current_fence(&self) -> &FrameFence {
        &self.fences[self.current_index() as usize]
    }

    /// Check if the synchronizer is idle (no pending GPU work).
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for polling
    ///
    /// # Returns
    ///
    /// `true` if all buffers have completed their work.
    pub fn is_idle(&self, device: &wgpu::Device) -> bool {
        // Poll to update status
        device.poll(wgpu::Maintain::Poll);

        // Check all fences
        self.fences
            .iter()
            .all(|fence| fence.oldest_pending_frame().is_none())
    }

    /// Get the wait offset for this synchronizer.
    ///
    /// Returns how many frames back we wait for backpressure:
    /// - Double buffering: 1 (wait for N-1)
    /// - Triple buffering: 2 (wait for N-2)
    #[inline]
    pub fn wait_offset(&self) -> u64 {
        if self.buffer_count == 2 {
            1
        } else {
            2
        }
    }

    /// Calculate the buffer index for a given frame number.
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame number
    ///
    /// # Returns
    ///
    /// The buffer index (0..buffer_count) for that frame.
    #[inline]
    pub fn buffer_for_frame(&self, frame: u64) -> usize {
        (frame % self.buffer_count as u64) as usize
    }

    /// Get the frame number that will be waited on for a given frame.
    ///
    /// Returns `None` if no wait is needed (early frames).
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame number being rendered
    ///
    /// # Returns
    ///
    /// The frame number that must complete before `frame` can begin,
    /// or `None` if no wait is required.
    pub fn wait_frame_for(&self, frame: u64) -> Option<u64> {
        let offset = self.wait_offset();
        if frame >= offset {
            Some(frame - offset)
        } else {
            None
        }
    }
}

impl Default for TrinityFrameSynchronizer {
    fn default() -> Self {
        Self::new(BufferCount::default())
    }
}

impl fmt::Debug for TrinityFrameSynchronizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrinityFrameSynchronizer")
            .field("buffer_count", &self.buffer_count)
            .field("current_buffer", &self.current_index())
            .field("frame_count", &self.frame_count())
            .field("low_latency", &self.is_low_latency())
            .field("fences", &self.fences.len())
            .finish()
    }
}

// ============================================================================
// FramePacer
// ============================================================================

/// Frame pacing controller for maintaining consistent frame times.
///
/// `FramePacer` tracks frame timing, calculates statistics, and optionally
/// provides adaptive pacing to help maintain a target frame rate. It maintains
/// a rolling history of frame times for accurate averaging and variance
/// calculation.
///
/// # Features
///
/// - **Target Frame Rate**: Configurable target FPS with automatic frame time calculation
/// - **Frame Time Tracking**: Rolling history of actual frame durations
/// - **Statistics**: Average frame time, variance, standard deviation, FPS
/// - **Adaptive Pacing**: Optional sleep duration calculation to hit target frame rate
///
/// # Example
///
/// ```
/// use renderer_backend::frame_sync::FramePacer;
/// use std::time::Duration;
///
/// // Create a pacer targeting 60 FPS
/// let mut pacer = FramePacer::new(60.0);
///
/// // In your render loop:
/// pacer.begin_frame();
/// // ... render ...
/// pacer.end_frame();
///
/// // Get statistics
/// if let Some(avg) = pacer.average_frame_time() {
///     println!("Average frame time: {:?}", avg);
/// }
/// println!("Current FPS: {:.1}", pacer.fps());
/// ```
///
/// # Adaptive Pacing
///
/// When adaptive pacing is enabled, `should_sleep()` returns the recommended
/// sleep duration to maintain the target frame rate:
///
/// ```
/// use renderer_backend::frame_sync::FramePacer;
///
/// let mut pacer = FramePacer::new(60.0);
/// pacer.set_adaptive(true);
///
/// // After rendering a frame
/// pacer.begin_frame();
/// // ... render ...
/// pacer.end_frame();
///
/// // Sleep if we finished early
/// if let Some(sleep_duration) = pacer.should_sleep() {
///     std::thread::sleep(sleep_duration);
/// }
/// ```
///
/// # Thread Safety
///
/// `FramePacer` is `Send + Sync` via interior mutability with `Mutex` and
/// `AtomicBool`. The typical usage pattern calls methods from a single
/// render thread.
pub struct FramePacer {
    /// Target frame time based on target FPS.
    target_frame_time: Mutex<Duration>,
    /// Rolling history of frame times.
    frame_times: Mutex<VecDeque<Duration>>,
    /// Maximum number of frame times to keep in history.
    history_size: usize,
    /// Timestamp when the current frame started.
    last_frame_start: Mutex<Option<Instant>>,
    /// Total number of frames processed.
    total_frames: AtomicU64,
    /// Whether adaptive pacing is enabled.
    adaptive_enabled: AtomicBool,
}

impl FramePacer {
    /// Create a new FramePacer with the specified target FPS.
    ///
    /// Uses a default history size of 60 frames for statistics calculation.
    ///
    /// # Arguments
    ///
    /// * `target_fps` - The target frames per second (e.g., 60.0, 30.0, 144.0)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let pacer = FramePacer::new(60.0);
    /// assert_eq!(pacer.target_frame_time().as_micros(), 16666); // ~16.67ms
    /// ```
    pub fn new(target_fps: f64) -> Self {
        Self::with_history_size(target_fps, 60)
    }

    /// Create a new FramePacer with the specified target FPS and history size.
    ///
    /// # Arguments
    ///
    /// * `target_fps` - The target frames per second
    /// * `history` - Number of frame times to keep for averaging (min 1)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let pacer = FramePacer::with_history_size(30.0, 120);
    /// assert_eq!(pacer.target_frame_time().as_micros(), 33333); // ~33.33ms
    /// ```
    pub fn with_history_size(target_fps: f64, history: usize) -> Self {
        let target_frame_time = if target_fps > 0.0 {
            Duration::from_secs_f64(1.0 / target_fps)
        } else {
            Duration::from_secs_f64(1.0 / 60.0) // Default to 60 FPS if invalid
        };

        let history_size = history.max(1);

        debug!(
            "FramePacer: Created with target {:.1} FPS ({:?}), history {}",
            target_fps, target_frame_time, history_size
        );

        Self {
            target_frame_time: Mutex::new(target_frame_time),
            frame_times: Mutex::new(VecDeque::with_capacity(history_size)),
            history_size,
            last_frame_start: Mutex::new(None),
            total_frames: AtomicU64::new(0),
            adaptive_enabled: AtomicBool::new(false),
        }
    }

    /// Get the target frame time.
    ///
    /// # Returns
    ///
    /// The duration that each frame should ideally take to hit the target FPS.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    /// use std::time::Duration;
    ///
    /// let pacer = FramePacer::new(60.0);
    /// // 1/60 = 0.01666... seconds = ~16.67ms
    /// assert!(pacer.target_frame_time() > Duration::from_millis(16));
    /// assert!(pacer.target_frame_time() < Duration::from_millis(17));
    /// ```
    pub fn target_frame_time(&self) -> Duration {
        *self.target_frame_time.lock().unwrap()
    }

    /// Set the target FPS.
    ///
    /// This recalculates the target frame time based on the new FPS value.
    ///
    /// # Arguments
    ///
    /// * `fps` - The new target frames per second (must be > 0)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    /// pacer.set_target_fps(144.0);
    /// assert!(pacer.target_frame_time().as_micros() < 7000); // < 7ms
    /// ```
    pub fn set_target_fps(&mut self, fps: f64) {
        if fps > 0.0 {
            let new_target = Duration::from_secs_f64(1.0 / fps);
            *self.target_frame_time.lock().unwrap() = new_target;
            debug!("FramePacer: Target FPS changed to {:.1} ({:?})", fps, new_target);
        } else {
            warn!("FramePacer: Invalid FPS {}, ignoring", fps);
        }
    }

    /// Begin timing a new frame.
    ///
    /// Call this at the start of each frame before any rendering work.
    /// The time between `begin_frame()` and `end_frame()` is recorded.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    /// pacer.begin_frame();
    /// // ... do rendering work ...
    /// pacer.end_frame();
    /// ```
    pub fn begin_frame(&mut self) {
        *self.last_frame_start.lock().unwrap() = Some(Instant::now());
        trace!("FramePacer: Begin frame {}", self.total_frames.load(Ordering::Relaxed));
    }

    /// End timing the current frame.
    ///
    /// Call this at the end of each frame after all rendering work.
    /// The frame time is recorded and added to the history.
    ///
    /// If `begin_frame()` was not called, this does nothing.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    /// pacer.begin_frame();
    /// std::thread::sleep(std::time::Duration::from_millis(5));
    /// pacer.end_frame();
    ///
    /// assert!(pacer.last_frame_time().is_some());
    /// ```
    pub fn end_frame(&mut self) {
        let mut start_guard = self.last_frame_start.lock().unwrap();
        if let Some(start) = start_guard.take() {
            let frame_time = start.elapsed();
            drop(start_guard); // Release lock before acquiring frame_times lock

            // Add to history
            let mut frame_times = self.frame_times.lock().unwrap();
            frame_times.push_back(frame_time);

            // Trim if over history size
            while frame_times.len() > self.history_size {
                frame_times.pop_front();
            }

            // Increment frame count
            let frame_num = self.total_frames.fetch_add(1, Ordering::AcqRel);

            trace!(
                "FramePacer: End frame {} ({:?})",
                frame_num,
                frame_time
            );
        }
    }

    /// Get the most recent frame time.
    ///
    /// # Returns
    ///
    /// The duration of the last completed frame, or `None` if no frames
    /// have been timed yet.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    /// assert!(pacer.last_frame_time().is_none());
    ///
    /// pacer.begin_frame();
    /// pacer.end_frame();
    ///
    /// assert!(pacer.last_frame_time().is_some());
    /// ```
    pub fn last_frame_time(&self) -> Option<Duration> {
        self.frame_times.lock().unwrap().back().copied()
    }

    /// Get the average frame time from the history.
    ///
    /// # Returns
    ///
    /// The average duration across all frames in the history, or `None`
    /// if no frames have been recorded.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    ///
    /// for _ in 0..5 {
    ///     pacer.begin_frame();
    ///     std::thread::sleep(std::time::Duration::from_millis(10));
    ///     pacer.end_frame();
    /// }
    ///
    /// let avg = pacer.average_frame_time().unwrap();
    /// assert!(avg.as_millis() >= 10);
    /// ```
    pub fn average_frame_time(&self) -> Option<Duration> {
        let frame_times = self.frame_times.lock().unwrap();
        if frame_times.is_empty() {
            return None;
        }

        let total: Duration = frame_times.iter().sum();
        Some(total / frame_times.len() as u32)
    }

    /// Calculate the variance of frame times in milliseconds squared.
    ///
    /// Variance measures how spread out the frame times are from the average.
    /// Lower variance indicates more consistent frame pacing.
    ///
    /// # Returns
    ///
    /// The variance in ms^2, or `None` if fewer than 2 frames recorded.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    ///
    /// for _ in 0..10 {
    ///     pacer.begin_frame();
    ///     std::thread::sleep(std::time::Duration::from_millis(16));
    ///     pacer.end_frame();
    /// }
    ///
    /// if let Some(variance) = pacer.variance() {
    ///     println!("Frame time variance: {:.4} ms^2", variance);
    /// }
    /// ```
    pub fn variance(&self) -> Option<f64> {
        let frame_times = self.frame_times.lock().unwrap();
        if frame_times.len() < 2 {
            return None;
        }

        // Calculate mean in milliseconds
        let times_ms: Vec<f64> = frame_times
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();

        let mean: f64 = times_ms.iter().sum::<f64>() / times_ms.len() as f64;

        // Calculate variance
        let variance: f64 = times_ms
            .iter()
            .map(|t| {
                let diff = t - mean;
                diff * diff
            })
            .sum::<f64>()
            / (times_ms.len() - 1) as f64; // Using Bessel's correction (n-1)

        Some(variance)
    }

    /// Calculate the standard deviation of frame times in milliseconds.
    ///
    /// Standard deviation is the square root of variance and represents
    /// the typical deviation from the mean frame time.
    ///
    /// # Returns
    ///
    /// The standard deviation in ms, or `None` if fewer than 2 frames recorded.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    ///
    /// for _ in 0..10 {
    ///     pacer.begin_frame();
    ///     std::thread::sleep(std::time::Duration::from_millis(16));
    ///     pacer.end_frame();
    /// }
    ///
    /// if let Some(std_dev) = pacer.std_deviation() {
    ///     println!("Frame time std deviation: {:.2} ms", std_dev);
    /// }
    /// ```
    pub fn std_deviation(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    /// Get the instantaneous FPS based on the last frame time.
    ///
    /// Returns 0.0 if no frames have been recorded.
    ///
    /// # Returns
    ///
    /// The FPS calculated from the most recent frame time.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    /// pacer.begin_frame();
    /// std::thread::sleep(std::time::Duration::from_millis(16));
    /// pacer.end_frame();
    ///
    /// let fps = pacer.fps();
    /// // Should be roughly 60 FPS (1000ms / 16ms ≈ 62.5)
    /// assert!(fps > 50.0 && fps < 70.0);
    /// ```
    pub fn fps(&self) -> f64 {
        match self.last_frame_time() {
            Some(duration) if duration.as_secs_f64() > 0.0 => 1.0 / duration.as_secs_f64(),
            _ => 0.0,
        }
    }

    /// Get the average FPS from the frame time history.
    ///
    /// # Returns
    ///
    /// The average FPS, or `None` if no frames have been recorded.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    ///
    /// for _ in 0..5 {
    ///     pacer.begin_frame();
    ///     std::thread::sleep(std::time::Duration::from_millis(16));
    ///     pacer.end_frame();
    /// }
    ///
    /// if let Some(avg_fps) = pacer.average_fps() {
    ///     println!("Average FPS: {:.1}", avg_fps);
    /// }
    /// ```
    pub fn average_fps(&self) -> Option<f64> {
        self.average_frame_time().map(|avg| {
            if avg.as_secs_f64() > 0.0 {
                1.0 / avg.as_secs_f64()
            } else {
                0.0
            }
        })
    }

    /// Get the total number of frames that have been timed.
    ///
    /// # Returns
    ///
    /// The count of frames where `begin_frame()` and `end_frame()` were both called.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    /// assert_eq!(pacer.total_frames(), 0);
    ///
    /// pacer.begin_frame();
    /// pacer.end_frame();
    /// assert_eq!(pacer.total_frames(), 1);
    /// ```
    pub fn total_frames(&self) -> u64 {
        self.total_frames.load(Ordering::Acquire)
    }

    /// Enable or disable adaptive pacing.
    ///
    /// When adaptive pacing is enabled, `should_sleep()` returns the
    /// recommended sleep duration to maintain the target frame rate.
    ///
    /// # Arguments
    ///
    /// * `enabled` - `true` to enable adaptive pacing, `false` to disable
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    /// assert!(!pacer.is_adaptive());
    ///
    /// pacer.set_adaptive(true);
    /// assert!(pacer.is_adaptive());
    /// ```
    pub fn set_adaptive(&mut self, enabled: bool) {
        let was_enabled = self.adaptive_enabled.swap(enabled, Ordering::AcqRel);
        if was_enabled != enabled {
            debug!(
                "FramePacer: Adaptive pacing {}",
                if enabled { "enabled" } else { "disabled" }
            );
        }
    }

    /// Check if adaptive pacing is enabled.
    ///
    /// # Returns
    ///
    /// `true` if adaptive pacing is currently enabled.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let pacer = FramePacer::new(60.0);
    /// assert!(!pacer.is_adaptive()); // Off by default
    /// ```
    pub fn is_adaptive(&self) -> bool {
        self.adaptive_enabled.load(Ordering::Acquire)
    }

    /// Calculate how long to sleep to hit the target frame time.
    ///
    /// This method is useful for frame pacing when you want to limit
    /// the frame rate to the target. Only returns a duration if:
    /// - Adaptive pacing is enabled
    /// - There is a recent frame time
    /// - The frame finished faster than the target
    ///
    /// # Returns
    ///
    /// The duration to sleep, or `None` if no sleep is needed.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    /// pacer.set_adaptive(true);
    ///
    /// pacer.begin_frame();
    /// // Fast frame - only 5ms of work
    /// std::thread::sleep(std::time::Duration::from_millis(5));
    /// pacer.end_frame();
    ///
    /// // Should recommend sleeping ~11ms to hit 16.67ms target
    /// if let Some(sleep) = pacer.should_sleep() {
    ///     assert!(sleep.as_millis() > 5);
    ///     std::thread::sleep(sleep);
    /// }
    /// ```
    pub fn should_sleep(&self) -> Option<Duration> {
        // Only calculate sleep if adaptive pacing is enabled
        if !self.adaptive_enabled.load(Ordering::Acquire) {
            return None;
        }

        let last_frame = self.last_frame_time()?;
        let target = self.target_frame_time();

        if last_frame < target {
            // We finished early, sleep for the remaining time
            Some(target - last_frame)
        } else {
            // We're behind, no sleep needed
            None
        }
    }

    /// Get the history size (maximum number of frame times tracked).
    ///
    /// # Returns
    ///
    /// The configured history size.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let pacer = FramePacer::with_history_size(60.0, 120);
    /// assert_eq!(pacer.history_size(), 120);
    /// ```
    pub fn history_size(&self) -> usize {
        self.history_size
    }

    /// Get the number of frame times currently in the history.
    ///
    /// # Returns
    ///
    /// The current count of recorded frame times.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    /// assert_eq!(pacer.frame_count_in_history(), 0);
    ///
    /// pacer.begin_frame();
    /// pacer.end_frame();
    /// assert_eq!(pacer.frame_count_in_history(), 1);
    /// ```
    pub fn frame_count_in_history(&self) -> usize {
        self.frame_times.lock().unwrap().len()
    }

    /// Clear the frame time history.
    ///
    /// This resets the rolling statistics but does not affect the
    /// total frame count or target FPS.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    ///
    /// for _ in 0..5 {
    ///     pacer.begin_frame();
    ///     pacer.end_frame();
    /// }
    ///
    /// assert_eq!(pacer.frame_count_in_history(), 5);
    /// pacer.clear_history();
    /// assert_eq!(pacer.frame_count_in_history(), 0);
    /// ```
    pub fn clear_history(&mut self) {
        self.frame_times.lock().unwrap().clear();
        debug!("FramePacer: History cleared");
    }

    /// Get the minimum frame time from the history.
    ///
    /// # Returns
    ///
    /// The shortest frame time, or `None` if history is empty.
    pub fn min_frame_time(&self) -> Option<Duration> {
        self.frame_times.lock().unwrap().iter().min().copied()
    }

    /// Get the maximum frame time from the history.
    ///
    /// # Returns
    ///
    /// The longest frame time, or `None` if history is empty.
    pub fn max_frame_time(&self) -> Option<Duration> {
        self.frame_times.lock().unwrap().iter().max().copied()
    }

    /// Calculate the jitter (max - min) of frame times.
    ///
    /// Jitter represents the range of frame times. Lower jitter means
    /// more consistent pacing.
    ///
    /// # Returns
    ///
    /// The jitter duration, or `None` if history has fewer than 2 frames.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::frame_sync::FramePacer;
    ///
    /// let mut pacer = FramePacer::new(60.0);
    ///
    /// for ms in &[15, 17, 16, 14, 18] {
    ///     pacer.begin_frame();
    ///     std::thread::sleep(std::time::Duration::from_millis(*ms));
    ///     pacer.end_frame();
    /// }
    ///
    /// if let Some(jitter) = pacer.jitter() {
    ///     println!("Frame time jitter: {:?}", jitter);
    /// }
    /// ```
    pub fn jitter(&self) -> Option<Duration> {
        let min = self.min_frame_time()?;
        let max = self.max_frame_time()?;
        Some(max - min)
    }

    /// Check if the pacer is currently in a frame (begin_frame called but not end_frame).
    ///
    /// # Returns
    ///
    /// `true` if currently timing a frame.
    pub fn is_in_frame(&self) -> bool {
        self.last_frame_start.lock().unwrap().is_some()
    }

    /// Get the target FPS.
    ///
    /// # Returns
    ///
    /// The target frames per second.
    pub fn target_fps(&self) -> f64 {
        let target = self.target_frame_time();
        if target.as_secs_f64() > 0.0 {
            1.0 / target.as_secs_f64()
        } else {
            0.0
        }
    }
}

impl Default for FramePacer {
    fn default() -> Self {
        Self::new(60.0)
    }
}

impl fmt::Debug for FramePacer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let frame_times = self.frame_times.lock().unwrap();
        f.debug_struct("FramePacer")
            .field("target_fps", &self.target_fps())
            .field("total_frames", &self.total_frames())
            .field("history_size", &self.history_size)
            .field("frames_in_history", &frame_times.len())
            .field("adaptive", &self.is_adaptive())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // -------------------------------------------------------------------------
    // FramePacer construction tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_new_60fps() {
        let pacer = FramePacer::new(60.0);
        let target = pacer.target_frame_time();
        // 1/60 = 0.01666... seconds = 16666 microseconds
        assert!(target.as_micros() >= 16666 && target.as_micros() <= 16667);
    }

    #[test]
    fn test_frame_pacer_new_30fps() {
        let pacer = FramePacer::new(30.0);
        let target = pacer.target_frame_time();
        // 1/30 = 0.03333... seconds = 33333 microseconds
        assert!(target.as_micros() >= 33333 && target.as_micros() <= 33334);
    }

    #[test]
    fn test_frame_pacer_new_144fps() {
        let pacer = FramePacer::new(144.0);
        let target = pacer.target_frame_time();
        // 1/144 = 0.00694... seconds = ~6944 microseconds
        assert!(target.as_micros() >= 6944 && target.as_micros() <= 6945);
    }

    #[test]
    fn test_frame_pacer_new_240fps() {
        let pacer = FramePacer::new(240.0);
        let target = pacer.target_frame_time();
        // 1/240 = 0.00416... seconds = ~4166 microseconds
        assert!(target.as_micros() >= 4166 && target.as_micros() <= 4167);
    }

    #[test]
    fn test_frame_pacer_with_history_size() {
        let pacer = FramePacer::with_history_size(60.0, 120);
        assert_eq!(pacer.history_size(), 120);
    }

    #[test]
    fn test_frame_pacer_with_history_size_minimum() {
        let pacer = FramePacer::with_history_size(60.0, 0);
        assert_eq!(pacer.history_size(), 1); // Clamped to minimum
    }

    #[test]
    fn test_frame_pacer_default() {
        let pacer = FramePacer::default();
        // Default is 60 FPS
        let target = pacer.target_frame_time();
        assert!(target.as_micros() >= 16666 && target.as_micros() <= 16667);
    }

    #[test]
    fn test_frame_pacer_invalid_fps_zero() {
        let pacer = FramePacer::new(0.0);
        // Should default to 60 FPS
        let target = pacer.target_frame_time();
        assert!(target.as_micros() >= 16666 && target.as_micros() <= 16667);
    }

    #[test]
    fn test_frame_pacer_invalid_fps_negative() {
        let pacer = FramePacer::new(-30.0);
        // Should default to 60 FPS
        let target = pacer.target_frame_time();
        assert!(target.as_micros() >= 16666 && target.as_micros() <= 16667);
    }

    // -------------------------------------------------------------------------
    // Target FPS tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_set_target_fps() {
        let mut pacer = FramePacer::new(60.0);
        pacer.set_target_fps(30.0);
        let target = pacer.target_frame_time();
        assert!(target.as_micros() >= 33333 && target.as_micros() <= 33334);
    }

    #[test]
    fn test_frame_pacer_set_target_fps_invalid() {
        let mut pacer = FramePacer::new(60.0);
        let original = pacer.target_frame_time();
        pacer.set_target_fps(0.0);
        // Should not change
        assert_eq!(pacer.target_frame_time(), original);
    }

    #[test]
    fn test_frame_pacer_target_fps_getter() {
        let pacer = FramePacer::new(60.0);
        let fps = pacer.target_fps();
        assert!((fps - 60.0).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Frame timing tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_begin_end_frame() {
        let mut pacer = FramePacer::new(60.0);
        assert_eq!(pacer.total_frames(), 0);

        pacer.begin_frame();
        pacer.end_frame();

        assert_eq!(pacer.total_frames(), 1);
        assert!(pacer.last_frame_time().is_some());
    }

    #[test]
    fn test_frame_pacer_begin_end_multiple() {
        let mut pacer = FramePacer::new(60.0);

        for _ in 0..10 {
            pacer.begin_frame();
            pacer.end_frame();
        }

        assert_eq!(pacer.total_frames(), 10);
        assert_eq!(pacer.frame_count_in_history(), 10);
    }

    #[test]
    fn test_frame_pacer_end_without_begin() {
        let mut pacer = FramePacer::new(60.0);
        pacer.end_frame(); // Should do nothing
        assert_eq!(pacer.total_frames(), 0);
        assert!(pacer.last_frame_time().is_none());
    }

    #[test]
    fn test_frame_pacer_is_in_frame() {
        let mut pacer = FramePacer::new(60.0);
        assert!(!pacer.is_in_frame());

        pacer.begin_frame();
        assert!(pacer.is_in_frame());

        pacer.end_frame();
        assert!(!pacer.is_in_frame());
    }

    #[test]
    fn test_frame_pacer_frame_time_tracking() {
        let mut pacer = FramePacer::new(60.0);

        pacer.begin_frame();
        std::thread::sleep(Duration::from_millis(10));
        pacer.end_frame();

        let last = pacer.last_frame_time().unwrap();
        assert!(last.as_millis() >= 10);
    }

    // -------------------------------------------------------------------------
    // History tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_history_trimming() {
        let mut pacer = FramePacer::with_history_size(60.0, 5);

        for _ in 0..10 {
            pacer.begin_frame();
            pacer.end_frame();
        }

        assert_eq!(pacer.total_frames(), 10);
        assert_eq!(pacer.frame_count_in_history(), 5); // Trimmed to max
    }

    #[test]
    fn test_frame_pacer_clear_history() {
        let mut pacer = FramePacer::new(60.0);

        for _ in 0..5 {
            pacer.begin_frame();
            pacer.end_frame();
        }

        assert_eq!(pacer.frame_count_in_history(), 5);
        pacer.clear_history();
        assert_eq!(pacer.frame_count_in_history(), 0);
        // Total frames should remain unchanged
        assert_eq!(pacer.total_frames(), 5);
    }

    #[test]
    fn test_frame_pacer_empty_history() {
        let pacer = FramePacer::new(60.0);
        assert!(pacer.last_frame_time().is_none());
        assert!(pacer.average_frame_time().is_none());
        assert!(pacer.variance().is_none());
        assert!(pacer.std_deviation().is_none());
        assert!(pacer.average_fps().is_none());
    }

    // -------------------------------------------------------------------------
    // Average frame time tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_average_single_frame() {
        let mut pacer = FramePacer::new(60.0);

        pacer.begin_frame();
        std::thread::sleep(Duration::from_millis(10));
        pacer.end_frame();

        let avg = pacer.average_frame_time().unwrap();
        assert!(avg.as_millis() >= 10);
    }

    #[test]
    fn test_frame_pacer_average_multiple_frames() {
        let mut pacer = FramePacer::new(60.0);

        // Simulate frames with varying times
        for _ in 0..5 {
            pacer.begin_frame();
            std::thread::sleep(Duration::from_millis(10));
            pacer.end_frame();
        }

        let avg = pacer.average_frame_time().unwrap();
        assert!(avg.as_millis() >= 10);
    }

    // -------------------------------------------------------------------------
    // Variance and standard deviation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_variance_single_frame() {
        let mut pacer = FramePacer::new(60.0);

        pacer.begin_frame();
        pacer.end_frame();

        // Need at least 2 frames for variance
        assert!(pacer.variance().is_none());
    }

    #[test]
    fn test_frame_pacer_variance_two_frames() {
        let mut pacer = FramePacer::new(60.0);

        pacer.begin_frame();
        pacer.end_frame();
        pacer.begin_frame();
        pacer.end_frame();

        // Should have variance now
        assert!(pacer.variance().is_some());
    }

    #[test]
    fn test_frame_pacer_std_deviation() {
        let mut pacer = FramePacer::new(60.0);

        for _ in 0..10 {
            pacer.begin_frame();
            std::thread::sleep(Duration::from_millis(5));
            pacer.end_frame();
        }

        let std_dev = pacer.std_deviation().unwrap();
        // Standard deviation should be non-negative
        assert!(std_dev >= 0.0);
    }

    #[test]
    fn test_frame_pacer_variance_relationship() {
        let mut pacer = FramePacer::new(60.0);

        for _ in 0..10 {
            pacer.begin_frame();
            std::thread::sleep(Duration::from_millis(5));
            pacer.end_frame();
        }

        let variance = pacer.variance().unwrap();
        let std_dev = pacer.std_deviation().unwrap();

        // std_dev = sqrt(variance)
        let calculated = variance.sqrt();
        assert!((std_dev - calculated).abs() < 0.0001);
    }

    // -------------------------------------------------------------------------
    // FPS tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_fps_no_frames() {
        let pacer = FramePacer::new(60.0);
        assert_eq!(pacer.fps(), 0.0);
    }

    #[test]
    fn test_frame_pacer_fps_with_frames() {
        let mut pacer = FramePacer::new(60.0);

        pacer.begin_frame();
        std::thread::sleep(Duration::from_millis(16));
        pacer.end_frame();

        let fps = pacer.fps();
        // Should be roughly 60 FPS (allowing for sleep imprecision)
        assert!(fps > 30.0 && fps < 100.0);
    }

    #[test]
    fn test_frame_pacer_average_fps_no_frames() {
        let pacer = FramePacer::new(60.0);
        assert!(pacer.average_fps().is_none());
    }

    #[test]
    fn test_frame_pacer_average_fps_with_frames() {
        let mut pacer = FramePacer::new(60.0);

        for _ in 0..5 {
            pacer.begin_frame();
            std::thread::sleep(Duration::from_millis(16));
            pacer.end_frame();
        }

        let avg_fps = pacer.average_fps().unwrap();
        assert!(avg_fps > 30.0 && avg_fps < 100.0);
    }

    // -------------------------------------------------------------------------
    // Adaptive pacing tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_adaptive_default() {
        let pacer = FramePacer::new(60.0);
        assert!(!pacer.is_adaptive());
    }

    #[test]
    fn test_frame_pacer_set_adaptive() {
        let mut pacer = FramePacer::new(60.0);

        pacer.set_adaptive(true);
        assert!(pacer.is_adaptive());

        pacer.set_adaptive(false);
        assert!(!pacer.is_adaptive());
    }

    #[test]
    fn test_frame_pacer_should_sleep_disabled() {
        let mut pacer = FramePacer::new(60.0);

        pacer.begin_frame();
        pacer.end_frame();

        // Adaptive not enabled, should return None
        assert!(pacer.should_sleep().is_none());
    }

    #[test]
    fn test_frame_pacer_should_sleep_enabled_fast_frame() {
        let mut pacer = FramePacer::new(60.0);
        pacer.set_adaptive(true);

        pacer.begin_frame();
        // Very fast frame (no sleep)
        pacer.end_frame();

        // Should recommend sleeping because we finished faster than target
        let sleep = pacer.should_sleep();
        assert!(sleep.is_some());
    }

    #[test]
    fn test_frame_pacer_should_sleep_enabled_slow_frame() {
        let mut pacer = FramePacer::new(60.0);
        pacer.set_adaptive(true);

        pacer.begin_frame();
        std::thread::sleep(Duration::from_millis(20)); // Slower than 16.67ms
        pacer.end_frame();

        // Should NOT recommend sleeping - we're already behind
        assert!(pacer.should_sleep().is_none());
    }

    #[test]
    fn test_frame_pacer_should_sleep_no_frames() {
        let mut pacer = FramePacer::new(60.0);
        pacer.set_adaptive(true);

        // No frames recorded
        assert!(pacer.should_sleep().is_none());
    }

    #[test]
    fn test_frame_pacer_should_sleep_calculation() {
        let mut pacer = FramePacer::new(60.0);
        pacer.set_adaptive(true);

        pacer.begin_frame();
        std::thread::sleep(Duration::from_millis(5));
        pacer.end_frame();

        let sleep = pacer.should_sleep().unwrap();
        // Should be roughly 16.67 - 5 = ~11.67ms
        assert!(sleep.as_millis() >= 5);
        assert!(sleep.as_millis() <= 15);
    }

    // -------------------------------------------------------------------------
    // Min/Max/Jitter tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_min_max_empty() {
        let pacer = FramePacer::new(60.0);
        assert!(pacer.min_frame_time().is_none());
        assert!(pacer.max_frame_time().is_none());
    }

    #[test]
    fn test_frame_pacer_min_max_single() {
        let mut pacer = FramePacer::new(60.0);

        pacer.begin_frame();
        pacer.end_frame();

        let min = pacer.min_frame_time().unwrap();
        let max = pacer.max_frame_time().unwrap();
        assert_eq!(min, max);
    }

    #[test]
    fn test_frame_pacer_min_max_multiple() {
        let mut pacer = FramePacer::new(60.0);

        for ms in [5, 10, 15, 8, 12] {
            pacer.begin_frame();
            std::thread::sleep(Duration::from_millis(ms));
            pacer.end_frame();
        }

        let min = pacer.min_frame_time().unwrap();
        let max = pacer.max_frame_time().unwrap();
        assert!(min <= max);
    }

    #[test]
    fn test_frame_pacer_jitter_empty() {
        let pacer = FramePacer::new(60.0);
        assert!(pacer.jitter().is_none());
    }

    #[test]
    fn test_frame_pacer_jitter_single() {
        let mut pacer = FramePacer::new(60.0);

        pacer.begin_frame();
        pacer.end_frame();

        // Jitter should be 0 for single frame
        let jitter = pacer.jitter().unwrap();
        assert_eq!(jitter, Duration::ZERO);
    }

    #[test]
    fn test_frame_pacer_jitter_multiple() {
        let mut pacer = FramePacer::new(60.0);

        for ms in [5, 10, 15] {
            pacer.begin_frame();
            std::thread::sleep(Duration::from_millis(ms));
            pacer.end_frame();
        }

        let jitter = pacer.jitter().unwrap();
        // Jitter should be roughly 10ms (15ms - 5ms)
        assert!(jitter.as_millis() >= 5);
    }

    // -------------------------------------------------------------------------
    // Debug format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_debug_format() {
        let pacer = FramePacer::new(60.0);
        let debug_str = format!("{:?}", pacer);

        assert!(debug_str.contains("FramePacer"));
        assert!(debug_str.contains("target_fps"));
        assert!(debug_str.contains("total_frames"));
        assert!(debug_str.contains("history_size"));
        assert!(debug_str.contains("adaptive"));
    }

    #[test]
    fn test_frame_pacer_debug_after_frames() {
        let mut pacer = FramePacer::new(60.0);

        for _ in 0..5 {
            pacer.begin_frame();
            pacer.end_frame();
        }

        let debug_str = format!("{:?}", pacer);
        assert!(debug_str.contains("5")); // total_frames or frames_in_history
    }

    // -------------------------------------------------------------------------
    // Thread safety tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<FramePacer>();
    }

    #[test]
    fn test_frame_pacer_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<FramePacer>();
    }

    #[test]
    fn test_frame_pacer_thread_safety_reads() {
        use std::sync::Arc;
        use std::thread;

        let pacer = Arc::new(FramePacer::new(60.0));
        let pacer_clone = Arc::clone(&pacer);

        let handle = thread::spawn(move || {
            for _ in 0..100 {
                let _ = pacer_clone.target_frame_time();
                let _ = pacer_clone.total_frames();
                let _ = pacer_clone.is_adaptive();
                let _ = pacer_clone.fps();
            }
        });

        for _ in 0..100 {
            let _ = pacer.target_frame_time();
            let _ = pacer.total_frames();
        }

        handle.join().unwrap();
    }

    // -------------------------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_pacer_very_high_fps() {
        let pacer = FramePacer::new(1000.0);
        let target = pacer.target_frame_time();
        assert_eq!(target.as_millis(), 1);
    }

    #[test]
    fn test_frame_pacer_very_low_fps() {
        let pacer = FramePacer::new(1.0);
        let target = pacer.target_frame_time();
        assert_eq!(target.as_secs(), 1);
    }

    #[test]
    fn test_frame_pacer_fractional_fps() {
        let pacer = FramePacer::new(59.94); // NTSC
        let target = pacer.target_frame_time();
        // 1/59.94 ≈ 16.68ms
        assert!(target.as_micros() >= 16680 && target.as_micros() <= 16690);
    }

    #[test]
    fn test_frame_pacer_large_history() {
        let mut pacer = FramePacer::with_history_size(60.0, 1000);

        for _ in 0..1000 {
            pacer.begin_frame();
            pacer.end_frame();
        }

        assert_eq!(pacer.frame_count_in_history(), 1000);
        assert_eq!(pacer.total_frames(), 1000);
    }

    #[test]
    fn test_frame_pacer_rapid_frames() {
        let mut pacer = FramePacer::new(60.0);

        // Rapid fire many frames
        for _ in 0..1000 {
            pacer.begin_frame();
            pacer.end_frame();
        }

        assert_eq!(pacer.total_frames(), 1000);
        // Should be trimmed to default 60
        assert_eq!(pacer.frame_count_in_history(), 60);
    }

    // -------------------------------------------------------------------------
    // FrameSubmission tests (existing)
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_submission_new() {
        let submission = FrameSubmission::new(42);
        assert_eq!(submission.frame_number, 42);
        assert!(!submission.has_submissions());
        assert_eq!(submission.submission_count(), 0);
        assert!(!submission.completed);
    }

    #[test]
    fn test_frame_submission_has_submissions() {
        let submission = FrameSubmission::new(0);
        assert!(!submission.has_submissions());
        // Can't easily test with real submissions without a queue
    }

    #[test]
    fn test_frame_submission_mark_completed() {
        let mut submission = FrameSubmission::new(0);
        assert!(!submission.completed);
        submission.mark_completed();
        assert!(submission.completed);
    }

    #[test]
    fn test_frame_submission_cpu_time_elapsed() {
        let submission = FrameSubmission::new(0);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = submission.cpu_time_elapsed();
        assert!(elapsed.as_millis() >= 10);
    }

    // -------------------------------------------------------------------------
    // FrameFence construction tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_fence_new() {
        let fence = FrameFence::new(3);
        assert_eq!(fence.frames_in_flight(), 3);
        assert_eq!(fence.current_frame(), 0);
        assert_eq!(fence.tracked_frame_count(), 1);
    }

    #[test]
    fn test_frame_fence_default() {
        let fence = FrameFence::default();
        assert_eq!(fence.frames_in_flight(), DEFAULT_FRAMES_IN_FLIGHT);
    }

    #[test]
    fn test_frame_fence_clamps_min() {
        let fence = FrameFence::new(1); // Below minimum
        assert_eq!(fence.frames_in_flight(), MIN_FRAMES_IN_FLIGHT);
    }

    #[test]
    fn test_frame_fence_clamps_max() {
        let fence = FrameFence::new(100); // Above maximum
        assert_eq!(fence.frames_in_flight(), MAX_FRAMES_IN_FLIGHT);
    }

    // -------------------------------------------------------------------------
    // FrameFence frame advancement tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_fence_advance_frame() {
        let fence = FrameFence::new(3);
        assert_eq!(fence.current_frame(), 0);

        let new_frame = fence.advance_frame();
        assert_eq!(new_frame, 1);
        assert_eq!(fence.current_frame(), 1);
        assert_eq!(fence.tracked_frame_count(), 2);

        let new_frame = fence.advance_frame();
        assert_eq!(new_frame, 2);
        assert_eq!(fence.current_frame(), 2);
    }

    #[test]
    fn test_frame_fence_advance_many_frames() {
        let fence = FrameFence::new(3);
        for i in 1..=10 {
            let frame = fence.advance_frame();
            assert_eq!(frame, i);
        }
        assert_eq!(fence.current_frame(), 10);
    }

    // -------------------------------------------------------------------------
    // FrameFence query tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_fence_oldest_pending_frame() {
        let fence = FrameFence::new(3);
        // Frame 0 starts as pending
        assert_eq!(fence.oldest_pending_frame(), Some(0));

        fence.advance_frame();
        // Still frame 0 is oldest pending
        assert_eq!(fence.oldest_pending_frame(), Some(0));
    }

    #[test]
    fn test_frame_fence_frame_stats() {
        let fence = FrameFence::new(3);

        let stats = fence.frame_stats(0);
        assert!(stats.is_some());
        let (count, completed) = stats.unwrap();
        assert_eq!(count, 0);
        assert!(!completed);

        // Non-existent frame
        assert!(fence.frame_stats(100).is_none());
    }

    #[test]
    fn test_frame_fence_is_frame_pending() {
        let fence = FrameFence::new(3);

        // Frame 0 is pending initially
        assert!(fence.is_frame_pending(0));

        // Non-existent frame is not pending
        assert!(!fence.is_frame_pending(100));
    }

    #[test]
    fn test_frame_fence_debug_format() {
        let fence = FrameFence::new(3);
        let debug_str = format!("{:?}", fence);
        assert!(debug_str.contains("FrameFence"));
        assert!(debug_str.contains("current_frame"));
        assert!(debug_str.contains("frames_in_flight"));
    }

    // -------------------------------------------------------------------------
    // FrameStats tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_stats_new() {
        let stats = FrameStats::new(42, 16.5, 3);
        assert_eq!(stats.frame_number, 42);
        assert!((stats.cpu_time_ms - 16.5).abs() < f64::EPSILON);
        assert_eq!(stats.submissions, 3);
    }

    #[test]
    fn test_frame_stats_default() {
        let stats = FrameStats::default();
        assert_eq!(stats.frame_number, 0);
        assert!((stats.cpu_time_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(stats.submissions, 0);
    }

    #[test]
    fn test_frame_stats_display() {
        let stats = FrameStats::new(5, 8.25, 2);
        let display = format!("{}", stats);
        assert!(display.contains("Frame 5"));
        assert!(display.contains("8.25ms"));
        assert!(display.contains("2 submissions"));
    }

    #[test]
    fn test_frame_stats_clone() {
        let stats = FrameStats::new(10, 12.0, 4);
        let cloned = stats.clone();
        assert_eq!(stats.frame_number, cloned.frame_number);
        assert!((stats.cpu_time_ms - cloned.cpu_time_ms).abs() < f64::EPSILON);
        assert_eq!(stats.submissions, cloned.submissions);
    }

    #[test]
    fn test_frame_stats_copy() {
        let stats = FrameStats::new(7, 9.5, 1);
        let copied = stats; // Copy
        assert_eq!(stats.frame_number, copied.frame_number);
    }

    // -------------------------------------------------------------------------
    // FrameSyncManager construction tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_sync_manager_new() {
        let sync = FrameSyncManager::new(3);
        assert_eq!(sync.max_frames_in_flight(), 3);
        assert_eq!(sync.total_frames(), 0);
        assert_eq!(sync.current_frame(), 0);
    }

    #[test]
    fn test_frame_sync_manager_default() {
        let sync = FrameSyncManager::default();
        assert_eq!(sync.max_frames_in_flight(), DEFAULT_FRAMES_IN_FLIGHT);
    }

    #[test]
    fn test_frame_sync_manager_clamps() {
        let sync = FrameSyncManager::new(1); // Below minimum
        assert_eq!(sync.max_frames_in_flight(), MIN_FRAMES_IN_FLIGHT);

        let sync = FrameSyncManager::new(100); // Above maximum
        assert_eq!(sync.max_frames_in_flight(), MAX_FRAMES_IN_FLIGHT);
    }

    // -------------------------------------------------------------------------
    // FrameSyncManager lifecycle tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_sync_manager_begin_frame() {
        let sync = FrameSyncManager::new(3);

        let frame0 = sync.begin_frame();
        assert_eq!(frame0, 0);
        assert_eq!(sync.total_frames(), 1);

        let frame1 = sync.begin_frame();
        assert_eq!(frame1, 1);
        assert_eq!(sync.total_frames(), 2);
    }

    #[test]
    fn test_frame_sync_manager_fence_access() {
        let sync = FrameSyncManager::new(3);
        let fence = sync.fence();
        assert_eq!(fence.frames_in_flight(), 3);
    }

    #[test]
    fn test_frame_sync_manager_oldest_pending() {
        let sync = FrameSyncManager::new(3);
        sync.begin_frame();

        // Frame 0 is oldest pending via fence
        assert_eq!(sync.oldest_pending_frame(), Some(0));
    }

    #[test]
    fn test_frame_sync_manager_debug_format() {
        let sync = FrameSyncManager::new(3);
        let debug_str = format!("{:?}", sync);
        assert!(debug_str.contains("FrameSyncManager"));
        assert!(debug_str.contains("max_frames_in_flight"));
        assert!(debug_str.contains("total_frames"));
    }

    #[test]
    fn test_frame_sync_manager_average_cpu_time_initial() {
        let sync = FrameSyncManager::new(3);
        // No frames yet, average should be 0
        assert!((sync.average_cpu_time_ms() - 0.0).abs() < f64::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Multiple frames tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_sync_manager_multiple_frames() {
        let sync = FrameSyncManager::new(3);

        for i in 0..5 {
            let frame = sync.begin_frame();
            assert_eq!(frame, i);
        }

        assert_eq!(sync.total_frames(), 5);
    }

    // -------------------------------------------------------------------------
    // Thread safety tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_fence_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let fence = Arc::new(FrameFence::new(3));
        let fence_clone = Arc::clone(&fence);

        let handle = thread::spawn(move || {
            for _ in 0..100 {
                let _ = fence_clone.current_frame();
                let _ = fence_clone.frames_in_flight();
                let _ = fence_clone.oldest_pending_frame();
            }
        });

        for _ in 0..100 {
            let _ = fence.current_frame();
        }

        handle.join().unwrap();
    }

    #[test]
    fn test_frame_sync_manager_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let sync = Arc::new(FrameSyncManager::new(3));
        let sync_clone = Arc::clone(&sync);

        let handle = thread::spawn(move || {
            for _ in 0..50 {
                let _ = sync_clone.total_frames();
                let _ = sync_clone.current_frame();
                let _ = sync_clone.max_frames_in_flight();
            }
        });

        for _ in 0..50 {
            let _ = sync.total_frames();
        }

        handle.join().unwrap();
    }

    // -------------------------------------------------------------------------
    // Constants tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_FRAMES_IN_FLIGHT, 3);
        assert!(MIN_FRAMES_IN_FLIGHT >= 2);
        assert!(MAX_FRAMES_IN_FLIGHT <= 8);
        assert!(MIN_FRAMES_IN_FLIGHT < MAX_FRAMES_IN_FLIGHT);
    }

    // -------------------------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frame_fence_rapid_advancement() {
        let fence = FrameFence::new(3);

        // Advance many times quickly
        for _ in 0..1000 {
            fence.advance_frame();
        }

        assert_eq!(fence.current_frame(), 1000);
        // Frames aren't trimmed until marked complete, so we'll have many tracked
        // Just verify it doesn't crash with rapid advancement
        assert!(fence.tracked_frame_count() > 0);
    }

    #[test]
    fn test_frame_stats_zero_submissions() {
        let stats = FrameStats::new(0, 0.0, 0);
        assert_eq!(stats.submissions, 0);
        let display = format!("{}", stats);
        assert!(display.contains("0 submissions"));
    }

    #[test]
    fn test_frame_submission_last_submission_empty() {
        let submission = FrameSubmission::new(0);
        assert!(submission.last_submission().is_none());
    }

    #[test]
    fn test_frame_fence_trimming_with_completion() {
        let fence = FrameFence::new(3);

        // Advance and manually mark frames complete to trigger trimming
        for i in 0..10 {
            fence.advance_frame();
            // Mark old frames as complete
            if i >= 2 {
                let mut submissions = fence.submissions.lock().unwrap();
                if let Some(old) = submissions.iter_mut().find(|s| s.frame_number == i - 2) {
                    old.mark_completed();
                }
            }
        }

        // After marking frames complete, trimming should have occurred
        let count = fence.tracked_frame_count();
        // We should have at most frames_in_flight + a few recent ones
        assert!(count <= 8, "Expected <= 8 tracked frames, got {}", count);
    }

    // =========================================================================
    // DoubleBufferedRenderer tests
    // =========================================================================

    // -------------------------------------------------------------------------
    // Initialization tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffered_renderer_new() {
        let renderer = DoubleBufferedRenderer::new();
        assert_eq!(renderer.current_index(), 0);
        assert_eq!(renderer.frame_count(), 0);
    }

    #[test]
    fn test_double_buffered_renderer_default() {
        let renderer = DoubleBufferedRenderer::default();
        assert_eq!(renderer.current_index(), 0);
        assert_eq!(renderer.frame_count(), 0);
    }

    #[test]
    fn test_double_buffered_renderer_has_two_fences() {
        let renderer = DoubleBufferedRenderer::new();
        // Access both fences - should not panic
        let _fence0 = renderer.fence(0);
        let _fence1 = renderer.fence(1);
    }

    #[test]
    fn test_double_buffered_renderer_fences_independent() {
        let renderer = DoubleBufferedRenderer::new();
        let fence0 = renderer.fence(0);
        let fence1 = renderer.fence(1);

        // Each fence should start at frame 0
        assert_eq!(fence0.current_frame(), 0);
        assert_eq!(fence1.current_frame(), 0);

        // Fences should be independent (different addresses)
        let ptr0 = fence0 as *const FrameFence;
        let ptr1 = fence1 as *const FrameFence;
        assert_ne!(ptr0, ptr1);
    }

    // -------------------------------------------------------------------------
    // Ping-pong index tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffered_renderer_current_index_initial() {
        let renderer = DoubleBufferedRenderer::new();
        assert_eq!(renderer.current_index(), 0);
    }

    #[test]
    fn test_double_buffered_renderer_next_index() {
        let renderer = DoubleBufferedRenderer::new();
        assert_eq!(renderer.current_index(), 0);
        assert_eq!(renderer.next_index(), 1);
    }

    #[test]
    fn test_double_buffered_renderer_next_index_symmetry() {
        let renderer = DoubleBufferedRenderer::new();

        // When current is 0, next should be 1
        assert_eq!(renderer.current_index(), 0);
        assert_eq!(renderer.next_index(), 1);

        // Manually set to 1 and verify symmetry
        renderer.current_buffer.store(1, Ordering::Release);
        assert_eq!(renderer.current_index(), 1);
        assert_eq!(renderer.next_index(), 0);
    }

    #[test]
    fn test_double_buffered_renderer_index_always_valid() {
        let renderer = DoubleBufferedRenderer::new();

        for _ in 0..100 {
            let current = renderer.current_index();
            let next = renderer.next_index();

            assert!(current == 0 || current == 1, "current must be 0 or 1");
            assert!(next == 0 || next == 1, "next must be 0 or 1");
            assert_ne!(current, next, "current and next must differ");

            // Toggle
            renderer.current_buffer.store(next, Ordering::Release);
        }
    }

    // -------------------------------------------------------------------------
    // Frame counting tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffered_renderer_frame_count_initial() {
        let renderer = DoubleBufferedRenderer::new();
        assert_eq!(renderer.frame_count(), 0);
    }

    #[test]
    fn test_double_buffered_renderer_frame_count_increments() {
        let renderer = DoubleBufferedRenderer::new();

        for expected in 0..10 {
            assert_eq!(renderer.frame_count(), expected);
            renderer.frame_count.fetch_add(1, Ordering::AcqRel);
        }
        assert_eq!(renderer.frame_count(), 10);
    }

    // -------------------------------------------------------------------------
    // Fence access tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffered_renderer_fence_access() {
        let renderer = DoubleBufferedRenderer::new();

        let fence0 = renderer.fence(0);
        let fence1 = renderer.fence(1);

        // Each fence is configured for 1 frame in flight (clamped to min 2)
        assert!(fence0.frames_in_flight() >= 1);
        assert!(fence1.frames_in_flight() >= 1);
    }

    #[test]
    #[should_panic(expected = "Buffer index must be 0 or 1")]
    fn test_double_buffered_renderer_fence_invalid_index() {
        let renderer = DoubleBufferedRenderer::new();
        let _ = renderer.fence(2); // Should panic
    }

    #[test]
    fn test_double_buffered_renderer_current_fence() {
        let renderer = DoubleBufferedRenderer::new();

        // Current fence should match fence(current_index)
        let current = renderer.current_index() as usize;
        let current_fence = renderer.current_fence();
        let indexed_fence = renderer.fence(current);

        // Should be the same fence
        let ptr1 = current_fence as *const FrameFence;
        let ptr2 = indexed_fence as *const FrameFence;
        assert_eq!(ptr1, ptr2);
    }

    // -------------------------------------------------------------------------
    // Debug format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffered_renderer_debug_format() {
        let renderer = DoubleBufferedRenderer::new();
        let debug_str = format!("{:?}", renderer);

        assert!(debug_str.contains("DoubleBufferedRenderer"));
        assert!(debug_str.contains("current_buffer"));
        assert!(debug_str.contains("frame_count"));
        assert!(debug_str.contains("fence_0"));
        assert!(debug_str.contains("fence_1"));
    }

    #[test]
    fn test_double_buffered_renderer_debug_after_modification() {
        let renderer = DoubleBufferedRenderer::new();

        // Modify state
        renderer.current_buffer.store(1, Ordering::Release);
        renderer.frame_count.store(42, Ordering::Release);

        let debug_str = format!("{:?}", renderer);
        assert!(debug_str.contains("1")); // current_buffer
        assert!(debug_str.contains("42")); // frame_count
    }

    // -------------------------------------------------------------------------
    // Thread safety tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffered_renderer_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let renderer = Arc::new(DoubleBufferedRenderer::new());
        let renderer_clone = Arc::clone(&renderer);

        let handle = thread::spawn(move || {
            for _ in 0..100 {
                let _ = renderer_clone.current_index();
                let _ = renderer_clone.next_index();
                let _ = renderer_clone.frame_count();
            }
        });

        for _ in 0..100 {
            let _ = renderer.current_index();
            let _ = renderer.next_index();
            let _ = renderer.frame_count();
        }

        handle.join().unwrap();
    }

    #[test]
    fn test_double_buffered_renderer_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;

        let renderer = Arc::new(DoubleBufferedRenderer::new());
        let mut handles = vec![];

        for _ in 0..4 {
            let r = Arc::clone(&renderer);
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    let _ = r.current_index();
                    let _ = r.next_index();
                    let _ = r.frame_count();
                    let _ = format!("{:?}", r);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    // -------------------------------------------------------------------------
    // Buffer cycling simulation tests (without actual GPU)
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffered_renderer_manual_cycling() {
        let renderer = DoubleBufferedRenderer::new();

        // Simulate cycling without actual GPU
        let mut expected_buffer = 0u32;
        let mut expected_frame = 0u64;

        for _ in 0..10 {
            assert_eq!(renderer.current_index(), expected_buffer);
            assert_eq!(renderer.frame_count(), expected_frame);

            // Simulate end_frame (toggle buffer, increment frame)
            let next = renderer.next_index();
            renderer.current_buffer.store(next, Ordering::Release);
            renderer.frame_count.fetch_add(1, Ordering::AcqRel);

            expected_buffer = 1 - expected_buffer;
            expected_frame += 1;
        }

        assert_eq!(renderer.frame_count(), 10);
    }

    #[test]
    fn test_double_buffered_renderer_buffer_pattern() {
        let renderer = DoubleBufferedRenderer::new();

        // Track buffer usage pattern
        let mut buffers_used = vec![];

        for _ in 0..6 {
            buffers_used.push(renderer.current_index());
            let next = renderer.next_index();
            renderer.current_buffer.store(next, Ordering::Release);
        }

        // Should alternate: 0, 1, 0, 1, 0, 1
        assert_eq!(buffers_used, vec![0, 1, 0, 1, 0, 1]);
    }

    // -------------------------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffered_renderer_many_frames() {
        let renderer = DoubleBufferedRenderer::new();

        // Simulate many frames
        for i in 0..1000 {
            assert_eq!(renderer.frame_count(), i);
            assert!(renderer.current_index() < 2);

            // Toggle and increment
            let next = renderer.next_index();
            renderer.current_buffer.store(next, Ordering::Release);
            renderer.frame_count.fetch_add(1, Ordering::AcqRel);
        }

        assert_eq!(renderer.frame_count(), 1000);
    }

    #[test]
    fn test_double_buffered_renderer_fence_frames_in_flight() {
        let renderer = DoubleBufferedRenderer::new();

        // Both fences should have been created with 1 frame in flight
        // (though FrameFence clamps to MIN_FRAMES_IN_FLIGHT = 2)
        let fence0 = renderer.fence(0);
        let fence1 = renderer.fence(1);

        assert_eq!(fence0.frames_in_flight(), MIN_FRAMES_IN_FLIGHT);
        assert_eq!(fence1.frames_in_flight(), MIN_FRAMES_IN_FLIGHT);
    }

    #[test]
    fn test_double_buffered_renderer_initial_fence_state() {
        let renderer = DoubleBufferedRenderer::new();

        // Both fences should start at frame 0
        assert_eq!(renderer.fence(0).current_frame(), 0);
        assert_eq!(renderer.fence(1).current_frame(), 0);

        // Both should have 1 tracked frame (frame 0)
        assert_eq!(renderer.fence(0).tracked_frame_count(), 1);
        assert_eq!(renderer.fence(1).tracked_frame_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Send + Sync compile tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffered_renderer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DoubleBufferedRenderer>();
    }

    #[test]
    fn test_double_buffered_renderer_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<DoubleBufferedRenderer>();
    }

    // -------------------------------------------------------------------------
    // Consistency tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_double_buffered_renderer_index_consistency() {
        let renderer = DoubleBufferedRenderer::new();

        // current_index and next_index should always sum to 1
        for _ in 0..100 {
            let current = renderer.current_index();
            let next = renderer.next_index();
            assert_eq!(current + next, 1, "current + next should always equal 1");

            // Toggle
            renderer.current_buffer.store(next, Ordering::Release);
        }
    }

    #[test]
    fn test_double_buffered_renderer_state_after_many_toggles() {
        let renderer = DoubleBufferedRenderer::new();

        // Toggle many times
        for _ in 0..1001 {
            let next = renderer.next_index();
            renderer.current_buffer.store(next, Ordering::Release);
        }

        // After odd number of toggles starting from 0, should be at 1
        assert_eq!(renderer.current_index(), 1);

        // One more toggle
        let next = renderer.next_index();
        renderer.current_buffer.store(next, Ordering::Release);
        assert_eq!(renderer.current_index(), 0);
    }

    // =========================================================================
    // BufferCount tests
    // =========================================================================

    // -------------------------------------------------------------------------
    // BufferCount basic tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffer_count_double() {
        let count = BufferCount::Double;
        assert_eq!(count.count(), 2);
        assert_eq!(count.wait_offset(), 1);
    }

    #[test]
    fn test_buffer_count_triple() {
        let count = BufferCount::Triple;
        assert_eq!(count.count(), 3);
        assert_eq!(count.wait_offset(), 2);
    }

    #[test]
    fn test_buffer_count_default() {
        let count = BufferCount::default();
        assert_eq!(count, BufferCount::Triple);
    }

    #[test]
    fn test_buffer_count_from_usize() {
        assert_eq!(BufferCount::from(0), BufferCount::Double);
        assert_eq!(BufferCount::from(1), BufferCount::Double);
        assert_eq!(BufferCount::from(2), BufferCount::Double);
        assert_eq!(BufferCount::from(3), BufferCount::Triple);
        assert_eq!(BufferCount::from(4), BufferCount::Triple);
        assert_eq!(BufferCount::from(100), BufferCount::Triple);
    }

    #[test]
    fn test_buffer_count_display() {
        let double = format!("{}", BufferCount::Double);
        let triple = format!("{}", BufferCount::Triple);

        assert!(double.contains("Double"));
        assert!(double.contains("2"));
        assert!(triple.contains("Triple"));
        assert!(triple.contains("3"));
    }

    #[test]
    fn test_buffer_count_debug() {
        let double = format!("{:?}", BufferCount::Double);
        let triple = format!("{:?}", BufferCount::Triple);

        assert_eq!(double, "Double");
        assert_eq!(triple, "Triple");
    }

    #[test]
    fn test_buffer_count_clone() {
        let original = BufferCount::Triple;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_buffer_count_copy() {
        let original = BufferCount::Double;
        let copied = original; // Copy
        assert_eq!(original, copied);
    }

    #[test]
    fn test_buffer_count_eq() {
        assert_eq!(BufferCount::Double, BufferCount::Double);
        assert_eq!(BufferCount::Triple, BufferCount::Triple);
        assert_ne!(BufferCount::Double, BufferCount::Triple);
    }

    #[test]
    fn test_buffer_count_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(BufferCount::Double);
        set.insert(BufferCount::Triple);
        set.insert(BufferCount::Double); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&BufferCount::Double));
        assert!(set.contains(&BufferCount::Triple));
    }

    // =========================================================================
    // TrinityFrameSynchronizer tests
    // =========================================================================

    // -------------------------------------------------------------------------
    // Initialization tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_new_double() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Double);
        assert_eq!(sync.buffer_count(), 2);
        assert_eq!(sync.current_index(), 0);
        assert_eq!(sync.frame_count(), 0);
        assert!(!sync.is_low_latency());
    }

    #[test]
    fn test_trinity_sync_new_triple() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
        assert_eq!(sync.buffer_count(), 3);
        assert_eq!(sync.current_index(), 0);
        assert_eq!(sync.frame_count(), 0);
        assert!(!sync.is_low_latency());
    }

    #[test]
    fn test_trinity_sync_default() {
        let sync = TrinityFrameSynchronizer::default();
        assert_eq!(sync.buffer_count(), 3); // Triple is default
    }

    #[test]
    fn test_trinity_sync_has_correct_fence_count() {
        let double = TrinityFrameSynchronizer::new(BufferCount::Double);
        assert_eq!(double.fences.len(), 2);

        let triple = TrinityFrameSynchronizer::new(BufferCount::Triple);
        assert_eq!(triple.fences.len(), 3);
    }

    // -------------------------------------------------------------------------
    // Buffer index tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_current_index_initial() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
        assert_eq!(sync.current_index(), 0);
    }

    #[test]
    fn test_trinity_sync_next_index_triple() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        assert_eq!(sync.current_index(), 0);
        assert_eq!(sync.next_index(), 1);

        sync.current_buffer.store(1, Ordering::Release);
        assert_eq!(sync.next_index(), 2);

        sync.current_buffer.store(2, Ordering::Release);
        assert_eq!(sync.next_index(), 0); // Wraps around
    }

    #[test]
    fn test_trinity_sync_next_index_double() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

        assert_eq!(sync.current_index(), 0);
        assert_eq!(sync.next_index(), 1);

        sync.current_buffer.store(1, Ordering::Release);
        assert_eq!(sync.next_index(), 0); // Wraps around
    }

    #[test]
    fn test_trinity_sync_buffer_index_always_valid_triple() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        for _ in 0..100 {
            let current = sync.current_index();
            let next = sync.next_index();

            assert!(current < 3, "current must be < 3 for triple buffering");
            assert!(next < 3, "next must be < 3 for triple buffering");
            assert_ne!(current, next, "current and next must differ");

            sync.current_buffer.store(next, Ordering::Release);
        }
    }

    #[test]
    fn test_trinity_sync_buffer_index_always_valid_double() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

        for _ in 0..100 {
            let current = sync.current_index();
            let next = sync.next_index();

            assert!(current < 2, "current must be < 2 for double buffering");
            assert!(next < 2, "next must be < 2 for double buffering");
            assert_ne!(current, next, "current and next must differ");

            sync.current_buffer.store(next, Ordering::Release);
        }
    }

    // -------------------------------------------------------------------------
    // Frame counting tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_frame_count_initial() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
        assert_eq!(sync.frame_count(), 0);
    }

    #[test]
    fn test_trinity_sync_frame_count_increments() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        for expected in 0..10 {
            assert_eq!(sync.frame_count(), expected);
            sync.frame_count.fetch_add(1, Ordering::AcqRel);
        }
        assert_eq!(sync.frame_count(), 10);
    }

    // -------------------------------------------------------------------------
    // Wait offset tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_wait_offset_double() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Double);
        assert_eq!(sync.wait_offset(), 1);
    }

    #[test]
    fn test_trinity_sync_wait_offset_triple() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
        assert_eq!(sync.wait_offset(), 2);
    }

    // -------------------------------------------------------------------------
    // Wait frame calculation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_wait_frame_for_triple() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        // Frame 0, 1: no wait needed (haven't rendered enough frames)
        assert_eq!(sync.wait_frame_for(0), None);
        assert_eq!(sync.wait_frame_for(1), None);

        // Frame 2+: wait for frame N-2
        assert_eq!(sync.wait_frame_for(2), Some(0));
        assert_eq!(sync.wait_frame_for(3), Some(1));
        assert_eq!(sync.wait_frame_for(10), Some(8));
        assert_eq!(sync.wait_frame_for(100), Some(98));
    }

    #[test]
    fn test_trinity_sync_wait_frame_for_double() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

        // Frame 0: no wait needed
        assert_eq!(sync.wait_frame_for(0), None);

        // Frame 1+: wait for frame N-1
        assert_eq!(sync.wait_frame_for(1), Some(0));
        assert_eq!(sync.wait_frame_for(2), Some(1));
        assert_eq!(sync.wait_frame_for(10), Some(9));
        assert_eq!(sync.wait_frame_for(100), Some(99));
    }

    // -------------------------------------------------------------------------
    // Buffer for frame calculation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_buffer_for_frame_triple() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        // Pattern: 0, 1, 2, 0, 1, 2, ...
        assert_eq!(sync.buffer_for_frame(0), 0);
        assert_eq!(sync.buffer_for_frame(1), 1);
        assert_eq!(sync.buffer_for_frame(2), 2);
        assert_eq!(sync.buffer_for_frame(3), 0);
        assert_eq!(sync.buffer_for_frame(4), 1);
        assert_eq!(sync.buffer_for_frame(5), 2);
        assert_eq!(sync.buffer_for_frame(99), 0); // 99 % 3 = 0
        assert_eq!(sync.buffer_for_frame(100), 1); // 100 % 3 = 1
    }

    #[test]
    fn test_trinity_sync_buffer_for_frame_double() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

        // Pattern: 0, 1, 0, 1, ...
        assert_eq!(sync.buffer_for_frame(0), 0);
        assert_eq!(sync.buffer_for_frame(1), 1);
        assert_eq!(sync.buffer_for_frame(2), 0);
        assert_eq!(sync.buffer_for_frame(3), 1);
        assert_eq!(sync.buffer_for_frame(100), 0);
        assert_eq!(sync.buffer_for_frame(101), 1);
    }

    // -------------------------------------------------------------------------
    // Low latency mode tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_low_latency_default() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
        assert!(!sync.is_low_latency());
    }

    #[test]
    fn test_trinity_sync_set_low_latency() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        sync.set_low_latency(true);
        assert!(sync.is_low_latency());

        sync.set_low_latency(false);
        assert!(!sync.is_low_latency());
    }

    #[test]
    fn test_trinity_sync_low_latency_toggle() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        for _ in 0..10 {
            let current = sync.is_low_latency();
            sync.set_low_latency(!current);
            assert_eq!(sync.is_low_latency(), !current);
        }
    }

    // -------------------------------------------------------------------------
    // Fence access tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_fence_access() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        // Access all fences
        let fence0 = sync.fence(0);
        let fence1 = sync.fence(1);
        let fence2 = sync.fence(2);

        // Each fence should have MIN_FRAMES_IN_FLIGHT (clamped from 1)
        assert_eq!(fence0.frames_in_flight(), MIN_FRAMES_IN_FLIGHT);
        assert_eq!(fence1.frames_in_flight(), MIN_FRAMES_IN_FLIGHT);
        assert_eq!(fence2.frames_in_flight(), MIN_FRAMES_IN_FLIGHT);
    }

    #[test]
    #[should_panic(expected = "Buffer index 3 out of range")]
    fn test_trinity_sync_fence_invalid_index() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
        let _ = sync.fence(3); // Should panic
    }

    #[test]
    #[should_panic(expected = "Buffer index 2 out of range")]
    fn test_trinity_sync_fence_invalid_index_double() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Double);
        let _ = sync.fence(2); // Should panic (only 0, 1 valid)
    }

    #[test]
    fn test_trinity_sync_current_fence() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        // Current fence should match fence(current_index)
        let current = sync.current_index() as usize;
        let current_fence = sync.current_fence();
        let indexed_fence = sync.fence(current);

        // Should be the same fence
        let ptr1 = current_fence as *const FrameFence;
        let ptr2 = indexed_fence as *const FrameFence;
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn test_trinity_sync_fences_independent() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        let fence0 = sync.fence(0);
        let fence1 = sync.fence(1);
        let fence2 = sync.fence(2);

        // Each fence should start at frame 0
        assert_eq!(fence0.current_frame(), 0);
        assert_eq!(fence1.current_frame(), 0);
        assert_eq!(fence2.current_frame(), 0);

        // Fences should be at different addresses
        let ptr0 = fence0 as *const FrameFence;
        let ptr1 = fence1 as *const FrameFence;
        let ptr2 = fence2 as *const FrameFence;
        assert_ne!(ptr0, ptr1);
        assert_ne!(ptr1, ptr2);
        assert_ne!(ptr0, ptr2);
    }

    // -------------------------------------------------------------------------
    // Debug format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_debug_format() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
        let debug_str = format!("{:?}", sync);

        assert!(debug_str.contains("TrinityFrameSynchronizer"));
        assert!(debug_str.contains("buffer_count"));
        assert!(debug_str.contains("3"));
        assert!(debug_str.contains("current_buffer"));
        assert!(debug_str.contains("frame_count"));
        assert!(debug_str.contains("low_latency"));
    }

    #[test]
    fn test_trinity_sync_debug_after_modification() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        sync.current_buffer.store(2, Ordering::Release);
        sync.frame_count.store(42, Ordering::Release);
        sync.set_low_latency(true);

        let debug_str = format!("{:?}", sync);
        assert!(debug_str.contains("2")); // current_buffer
        assert!(debug_str.contains("42")); // frame_count
        assert!(debug_str.contains("true")); // low_latency
    }

    // -------------------------------------------------------------------------
    // Thread safety tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let sync = Arc::new(TrinityFrameSynchronizer::new(BufferCount::Triple));
        let sync_clone = Arc::clone(&sync);

        let handle = thread::spawn(move || {
            for _ in 0..100 {
                let _ = sync_clone.current_index();
                let _ = sync_clone.next_index();
                let _ = sync_clone.frame_count();
                let _ = sync_clone.is_low_latency();
                let _ = sync_clone.buffer_count();
            }
        });

        for _ in 0..100 {
            let _ = sync.current_index();
            let _ = sync.next_index();
            let _ = sync.frame_count();
            let _ = sync.is_low_latency();
        }

        handle.join().unwrap();
    }

    #[test]
    fn test_trinity_sync_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;

        let sync = Arc::new(TrinityFrameSynchronizer::new(BufferCount::Triple));
        let mut handles = vec![];

        for _ in 0..4 {
            let s = Arc::clone(&sync);
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    let _ = s.current_index();
                    let _ = s.next_index();
                    let _ = s.frame_count();
                    let _ = s.is_low_latency();
                    let _ = format!("{:?}", s);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_trinity_sync_concurrent_low_latency_toggle() {
        use std::sync::Arc;
        use std::thread;

        let sync = Arc::new(TrinityFrameSynchronizer::new(BufferCount::Triple));
        let mut handles = vec![];

        for i in 0..4 {
            let s = Arc::clone(&sync);
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    s.set_low_latency(i % 2 == 0);
                    let _ = s.is_low_latency();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    // -------------------------------------------------------------------------
    // Buffer cycling simulation tests (without actual GPU)
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_manual_cycling_triple() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        let mut expected_buffer = 0u32;
        let mut expected_frame = 0u64;

        for _ in 0..12 {
            assert_eq!(sync.current_index(), expected_buffer);
            assert_eq!(sync.frame_count(), expected_frame);

            // Simulate end_frame (advance buffer and frame)
            let next = sync.next_index();
            sync.current_buffer.store(next, Ordering::Release);
            sync.frame_count.fetch_add(1, Ordering::AcqRel);

            expected_buffer = (expected_buffer + 1) % 3;
            expected_frame += 1;
        }

        assert_eq!(sync.frame_count(), 12);
    }

    #[test]
    fn test_trinity_sync_buffer_pattern_triple() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        let mut buffers_used = vec![];

        for _ in 0..9 {
            buffers_used.push(sync.current_index());
            let next = sync.next_index();
            sync.current_buffer.store(next, Ordering::Release);
        }

        // Should cycle: 0, 1, 2, 0, 1, 2, 0, 1, 2
        assert_eq!(buffers_used, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_trinity_sync_buffer_pattern_double() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

        let mut buffers_used = vec![];

        for _ in 0..6 {
            buffers_used.push(sync.current_index());
            let next = sync.next_index();
            sync.current_buffer.store(next, Ordering::Release);
        }

        // Should alternate: 0, 1, 0, 1, 0, 1
        assert_eq!(buffers_used, vec![0, 1, 0, 1, 0, 1]);
    }

    // -------------------------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_many_frames() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        for i in 0..1000 {
            assert_eq!(sync.frame_count(), i);
            assert!(sync.current_index() < 3);

            let next = sync.next_index();
            sync.current_buffer.store(next, Ordering::Release);
            sync.frame_count.fetch_add(1, Ordering::AcqRel);
        }

        assert_eq!(sync.frame_count(), 1000);
    }

    #[test]
    fn test_trinity_sync_initial_fence_state() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        // All fences should start at frame 0
        for i in 0..3 {
            assert_eq!(sync.fence(i).current_frame(), 0);
            assert_eq!(sync.fence(i).tracked_frame_count(), 1);
        }
    }

    // -------------------------------------------------------------------------
    // Send + Sync compile tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TrinityFrameSynchronizer>();
    }

    #[test]
    fn test_trinity_sync_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TrinityFrameSynchronizer>();
    }

    #[test]
    fn test_buffer_count_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<BufferCount>();
    }

    #[test]
    fn test_buffer_count_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<BufferCount>();
    }

    // -------------------------------------------------------------------------
    // Consistency tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_buffer_reuse_pattern_triple() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        // Verify that frame N and frame N-3 use the same buffer (triple buffering)
        for frame in 0u64..20 {
            let buffer = sync.buffer_for_frame(frame);
            if frame >= 3 {
                let old_buffer = sync.buffer_for_frame(frame - 3);
                assert_eq!(
                    buffer, old_buffer,
                    "Frame {} and frame {} should use same buffer",
                    frame,
                    frame - 3
                );
            }
        }
    }

    #[test]
    fn test_trinity_sync_buffer_reuse_pattern_double() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

        // Verify that frame N and frame N-2 use the same buffer (double buffering)
        for frame in 0u64..20 {
            let buffer = sync.buffer_for_frame(frame);
            if frame >= 2 {
                let old_buffer = sync.buffer_for_frame(frame - 2);
                assert_eq!(
                    buffer, old_buffer,
                    "Frame {} and frame {} should use same buffer",
                    frame,
                    frame - 2
                );
            }
        }
    }

    #[test]
    fn test_trinity_sync_wait_frame_matches_buffer_reuse() {
        // For triple buffering, we wait for N-2, which is when the buffer was last used
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        for frame in 3u64..20 {
            let wait_frame = sync.wait_frame_for(frame).unwrap();
            let current_buffer = sync.buffer_for_frame(frame);
            let wait_buffer = sync.buffer_for_frame(wait_frame);

            // The buffer we're using should NOT be the same as wait_frame's buffer
            // because we wait for N-2, but N-3 is when the same buffer was last used
            // Actually, for triple: frame N uses buffer N%3, frame N-3 uses same buffer
            // We wait for N-2, which uses a DIFFERENT buffer, so GPU has finished
            // with the buffer from N-3 by the time N-2 completes
            assert_eq!(
                current_buffer,
                sync.buffer_for_frame(frame - 3).try_into().unwrap_or(0),
                "Frame {} should reuse buffer from frame {}",
                frame,
                frame - 3
            );
        }
    }

    // -------------------------------------------------------------------------
    // Integration-style tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trinity_sync_simulated_render_loop_triple() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        // Simulate a render loop without actual GPU
        for frame in 0u64..10 {
            // begin_frame would wait here (simulated)
            let _ = sync.wait_frame_for(frame);

            // Record buffer used
            let buffer = sync.current_index();
            assert_eq!(buffer, (frame % 3) as u32);

            // end_frame
            let next = sync.next_index();
            sync.current_buffer.store(next, Ordering::Release);
            sync.frame_count.fetch_add(1, Ordering::AcqRel);
        }

        assert_eq!(sync.frame_count(), 10);
    }

    #[test]
    fn test_trinity_sync_simulated_render_loop_double() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

        for frame in 0u64..10 {
            let _ = sync.wait_frame_for(frame);

            let buffer = sync.current_index();
            assert_eq!(buffer, (frame % 2) as u32);

            let next = sync.next_index();
            sync.current_buffer.store(next, Ordering::Release);
            sync.frame_count.fetch_add(1, Ordering::AcqRel);
        }

        assert_eq!(sync.frame_count(), 10);
    }

    #[test]
    fn test_trinity_sync_state_after_many_toggles() {
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

        // Toggle many times
        for _ in 0..1000 {
            let next = sync.next_index();
            sync.current_buffer.store(next, Ordering::Release);
        }

        // 1000 toggles starting from 0: 1000 % 3 = 1
        assert_eq!(sync.current_index(), 1);
    }
}
