//! Frame pacing and timing control for the presentation system.
//!
//! This module provides comprehensive frame pacing infrastructure for consistent
//! frame delivery in the TRINITY renderer:
//!
//! - [`FramePacingMode`] - Configuration for frame timing strategy
//! - [`FrameTimingStats`] - Aggregated frame timing statistics
//! - [`FramePacerV2`] - Advanced frame pacer with mode switching
//! - [`FrameBudget`] - Per-frame budget tracking with checkpoints
//! - [`SmoothedFrameTime`] - Low-pass filtered frame time for smooth display
//!
//! # Architecture
//!
//! The frame pacing system works in conjunction with the presentation pipeline:
//!
//! ```text
//! FramePacerV2
//!     |-- FramePacingMode (VSync, Uncapped, Capped, Adaptive)
//!     |-- frame_history (rolling window of frame times)
//!     `-- FrameTimingStats (computed statistics)
//!
//! FrameBudget
//!     |-- total_budget (target frame time)
//!     |-- checkpoints (named timing points)
//!     `-- elapsed (time since budget start)
//!
//! SmoothedFrameTime
//!     |-- alpha (smoothing factor)
//!     `-- smoothed_ms (filtered frame time)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::presentation::frame_pacing::{
//!     FramePacerV2, FramePacingMode, FrameBudget, SmoothedFrameTime,
//! };
//!
//! // Create a capped frame pacer
//! let mut pacer = FramePacerV2::new(FramePacingMode::Capped(144.0));
//!
//! // In render loop
//! pacer.begin_frame();
//!
//! let mut budget = FrameBudget::new(144.0);
//! // ... render scene ...
//! budget.checkpoint("scene");
//! // ... post-process ...
//! budget.checkpoint("post");
//!
//! let frame_time = pacer.end_frame();
//! pacer.wait_for_target();
//!
//! println!("Stats: {:?}", pacer.stats());
//! println!("Budget used: {:.1}%", budget.used_percentage());
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ============================================================================
// Constants
// ============================================================================

/// Default number of frames to retain in history for statistics.
pub const DEFAULT_PACING_HISTORY_SIZE: usize = 120;

/// Minimum sleep threshold in microseconds.
/// Below this, we spin-wait for better accuracy.
const MIN_SLEEP_THRESHOLD_US: u64 = 500;

// ============================================================================
// FramePacingMode
// ============================================================================

/// Frame pacing mode controlling timing behavior.
///
/// Determines how the frame pacer regulates frame timing:
///
/// - `VSync` - Lock to display refresh rate (handled by hardware)
/// - `Uncapped` - No frame rate limit (maximum performance)
/// - `Capped(f64)` - Software limit to target FPS
/// - `Adaptive` - Dynamic adjustment based on workload
///
/// # Example
///
/// ```ignore
/// use renderer_backend::presentation::frame_pacing::FramePacingMode;
///
/// let mode = FramePacingMode::Capped(60.0);
/// assert!(mode.is_capped());
/// assert_eq!(mode.target_fps(), Some(60.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FramePacingMode {
    /// Lock to display refresh rate (VSync).
    ///
    /// The GPU/display compositor handles timing. Software pacing
    /// is minimal - we rely on present queue blocking.
    VSync,

    /// No frame rate limit.
    ///
    /// Renders as fast as possible. Useful for benchmarking or
    /// when the application has its own timing control.
    Uncapped,

    /// Target a specific FPS (e.g., 60.0, 144.0).
    ///
    /// Software frame limiting via sleep/spin. The pacer will
    /// wait after each frame to maintain the target rate.
    Capped(f64),

    /// Dynamic pacing based on workload.
    ///
    /// Adjusts target frame time based on recent performance,
    /// attempting to find a stable frame rate. Falls back to
    /// VSync behavior when workload is light.
    Adaptive,
}

impl FramePacingMode {
    /// Get the target FPS for this mode.
    ///
    /// Returns `None` for VSync (hardware-determined) and Uncapped (no limit).
    /// Returns `Some(fps)` for Capped mode.
    /// Returns `None` for Adaptive (dynamically determined).
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert_eq!(FramePacingMode::Capped(60.0).target_fps(), Some(60.0));
    /// assert_eq!(FramePacingMode::VSync.target_fps(), None);
    /// ```
    pub fn target_fps(&self) -> Option<f64> {
        match self {
            Self::Capped(fps) => Some(*fps),
            _ => None,
        }
    }

    /// Check if this mode uses VSync.
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert!(FramePacingMode::VSync.is_vsync());
    /// assert!(!FramePacingMode::Uncapped.is_vsync());
    /// ```
    pub fn is_vsync(&self) -> bool {
        matches!(self, Self::VSync)
    }

    /// Check if this mode has a capped frame rate.
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert!(FramePacingMode::Capped(60.0).is_capped());
    /// assert!(!FramePacingMode::VSync.is_capped());
    /// ```
    pub fn is_capped(&self) -> bool {
        matches!(self, Self::Capped(_))
    }

    /// Check if this mode uses adaptive pacing.
    pub fn is_adaptive(&self) -> bool {
        matches!(self, Self::Adaptive)
    }

    /// Check if this mode is uncapped.
    pub fn is_uncapped(&self) -> bool {
        matches!(self, Self::Uncapped)
    }

    /// Get a human-readable name for this mode.
    pub fn name(&self) -> &'static str {
        match self {
            Self::VSync => "VSync",
            Self::Uncapped => "Uncapped",
            Self::Capped(_) => "Capped",
            Self::Adaptive => "Adaptive",
        }
    }

    /// Get a description of this mode.
    pub fn description(&self) -> &'static str {
        match self {
            Self::VSync => "Locked to display refresh rate",
            Self::Uncapped => "No frame rate limit",
            Self::Capped(_) => "Software frame rate limit",
            Self::Adaptive => "Dynamic frame rate based on workload",
        }
    }
}

impl Default for FramePacingMode {
    fn default() -> Self {
        Self::VSync
    }
}

// ============================================================================
// FrameTimingStats
// ============================================================================

/// Aggregated frame timing statistics.
///
/// Provides a snapshot of frame timing performance including frame time,
/// FPS, variance, and budget utilization.
///
/// # Example
///
/// ```ignore
/// let stats = pacer.stats();
/// println!("FPS: {:.1}", stats.fps);
/// println!("Frame time: {:.2}ms", stats.frame_time_ms);
/// println!("Variance: {:.2}ms", stats.frame_variance_ms);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTimingStats {
    /// Average frame time in milliseconds.
    pub frame_time_ms: f64,

    /// Frames per second (1000 / frame_time_ms).
    pub fps: f64,

    /// Frame time variance in milliseconds.
    ///
    /// Lower values indicate more consistent frame pacing.
    pub frame_variance_ms: f64,

    /// Percentage of frame budget used (0-100+).
    ///
    /// Values over 100 indicate frames exceeding their budget.
    pub frame_budget_used_pct: f64,

    /// Total number of frames that exceeded the target frame time.
    pub missed_frames: u64,
}

impl FrameTimingStats {
    /// Create empty statistics.
    pub fn empty() -> Self {
        Self {
            frame_time_ms: 0.0,
            fps: 0.0,
            frame_variance_ms: 0.0,
            frame_budget_used_pct: 0.0,
            missed_frames: 0,
        }
    }

    /// Check if frame pacing is consistent.
    ///
    /// Returns true if variance is below the given threshold (in ms).
    pub fn is_consistent(&self, threshold_ms: f64) -> bool {
        self.frame_variance_ms < threshold_ms
    }

    /// Check if hitting the target frame rate.
    ///
    /// Returns true if budget usage is at or below 100%.
    pub fn is_on_target(&self) -> bool {
        self.frame_budget_used_pct <= 100.0
    }

    /// Get the frame time standard deviation in milliseconds.
    pub fn frame_std_dev_ms(&self) -> f64 {
        self.frame_variance_ms.sqrt()
    }
}

impl Default for FrameTimingStats {
    fn default() -> Self {
        Self::empty()
    }
}

// ============================================================================
// FramePacerV2
// ============================================================================

/// Advanced frame pacer with mode switching and statistics.
///
/// `FramePacerV2` provides comprehensive frame timing control including:
/// - Multiple pacing modes (VSync, Capped, Uncapped, Adaptive)
/// - Frame time history for statistics
/// - Variance and missed frame tracking
/// - Sleep/spin hybrid waiting for accuracy
///
/// # Example
///
/// ```ignore
/// let mut pacer = FramePacerV2::new(FramePacingMode::Capped(60.0));
///
/// loop {
///     pacer.begin_frame();
///
///     // Render scene...
///
///     let frame_time = pacer.end_frame();
///     pacer.wait_for_target();
///
///     if pacer.stats().missed_frames > 100 {
///         // Consider reducing quality
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FramePacerV2 {
    /// Current pacing mode.
    mode: FramePacingMode,

    /// Target frame time based on mode.
    target_frame_time: Duration,

    /// Rolling history of frame times.
    frame_history: VecDeque<Duration>,

    /// Maximum history size.
    history_size: usize,

    /// When the current frame started.
    frame_start: Instant,

    /// Whether we're currently in a frame.
    in_frame: bool,

    /// Total frames rendered.
    frame_count: u64,

    /// Frames that exceeded target time.
    missed_frames: u64,

    /// Last computed frame time.
    last_frame_time: Duration,

    /// Accumulated time error for adaptive correction.
    accumulated_error: i64,

    /// Adaptive mode: current target FPS.
    adaptive_target_fps: f64,

    /// Adaptive mode: refresh rate hint.
    display_refresh_rate: f64,
}

impl FramePacerV2 {
    /// Create a new frame pacer with the specified mode.
    ///
    /// # Arguments
    ///
    /// * `mode` - The pacing mode to use.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pacer = FramePacerV2::new(FramePacingMode::Capped(144.0));
    /// ```
    pub fn new(mode: FramePacingMode) -> Self {
        Self::with_history_size(mode, DEFAULT_PACING_HISTORY_SIZE)
    }

    /// Create a frame pacer with custom history size.
    ///
    /// # Arguments
    ///
    /// * `mode` - The pacing mode to use.
    /// * `size` - Number of frame times to retain for statistics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pacer = FramePacerV2::with_history_size(FramePacingMode::Capped(60.0), 60);
    /// ```
    pub fn with_history_size(mode: FramePacingMode, size: usize) -> Self {
        let target_frame_time = Self::compute_target_frame_time(&mode, 60.0);

        Self {
            mode,
            target_frame_time,
            frame_history: VecDeque::with_capacity(size),
            history_size: size.max(1),
            frame_start: Instant::now(),
            in_frame: false,
            frame_count: 0,
            missed_frames: 0,
            last_frame_time: Duration::ZERO,
            accumulated_error: 0,
            adaptive_target_fps: 60.0,
            display_refresh_rate: 60.0,
        }
    }

    /// Compute target frame time from mode.
    fn compute_target_frame_time(mode: &FramePacingMode, display_hz: f64) -> Duration {
        match mode {
            FramePacingMode::VSync => {
                // Assume display refresh rate
                Duration::from_secs_f64(1.0 / display_hz.max(1.0))
            }
            FramePacingMode::Uncapped => Duration::ZERO,
            FramePacingMode::Capped(fps) => {
                if *fps > 0.0 {
                    Duration::from_secs_f64(1.0 / fps)
                } else {
                    Duration::ZERO
                }
            }
            FramePacingMode::Adaptive => {
                // Start with display refresh rate
                Duration::from_secs_f64(1.0 / display_hz.max(1.0))
            }
        }
    }

    /// Mark the beginning of a new frame.
    ///
    /// Call at the start of each frame loop iteration.
    pub fn begin_frame(&mut self) {
        if self.in_frame {
            // Auto-end previous frame if not ended
            self.end_frame();
        }
        self.frame_start = Instant::now();
        self.in_frame = true;
    }

    /// Mark the end of a frame.
    ///
    /// Call after presenting but before waiting.
    ///
    /// # Returns
    ///
    /// The actual frame time for this frame.
    pub fn end_frame(&mut self) -> Duration {
        if !self.in_frame {
            return Duration::ZERO;
        }

        let frame_time = self.frame_start.elapsed();
        self.last_frame_time = frame_time;
        self.in_frame = false;
        self.frame_count += 1;

        // Track missed frames
        if !self.target_frame_time.is_zero() && frame_time > self.target_frame_time {
            self.missed_frames += 1;
        }

        // Add to history
        if self.frame_history.len() >= self.history_size {
            self.frame_history.pop_front();
        }
        self.frame_history.push_back(frame_time);

        // Adaptive mode: adjust target based on performance
        if self.mode.is_adaptive() {
            self.adapt_target_fps();
        }

        frame_time
    }

    /// Adapt target FPS based on recent performance (for Adaptive mode).
    fn adapt_target_fps(&mut self) {
        if self.frame_history.len() < 10 {
            return;
        }

        // Calculate recent average frame time
        let recent: Vec<_> = self.frame_history.iter().rev().take(10).collect();
        let avg_ns: u128 = recent.iter().map(|d| d.as_nanos()).sum::<u128>() / recent.len() as u128;
        let avg_ms = avg_ns as f64 / 1_000_000.0;

        // Target a frame time that allows 10% headroom
        let target_ms = avg_ms * 1.1;
        let new_fps = (1000.0 / target_ms).clamp(15.0, self.display_refresh_rate);

        // Smooth the transition
        self.adaptive_target_fps = self.adaptive_target_fps * 0.9 + new_fps * 0.1;
        self.target_frame_time = Duration::from_secs_f64(1.0 / self.adaptive_target_fps);
    }

    /// Wait to maintain target frame rate.
    ///
    /// Sleeps or spins as needed to maintain consistent timing.
    /// Does nothing for Uncapped mode.
    pub fn wait_for_target(&mut self) {
        if self.mode.is_uncapped() || self.target_frame_time.is_zero() {
            return;
        }

        // VSync mode relies on present queue blocking
        if self.mode.is_vsync() {
            return;
        }

        let elapsed = self.frame_start.elapsed();
        if elapsed >= self.target_frame_time {
            // Already over budget
            return;
        }

        let wait_time = self.target_frame_time - elapsed;
        let wait_ns = wait_time.as_nanos() as i64 + self.accumulated_error;

        if wait_ns <= 0 {
            // Accumulated debt, skip waiting
            self.accumulated_error = wait_ns.max(-(self.target_frame_time.as_nanos() as i64));
            return;
        }

        let actual_wait = Duration::from_nanos(wait_ns as u64);

        // Hybrid sleep/spin for accuracy
        if actual_wait > Duration::from_micros(MIN_SLEEP_THRESHOLD_US) {
            // Sleep for most of the time
            let sleep_time = actual_wait - Duration::from_micros(MIN_SLEEP_THRESHOLD_US);
            std::thread::sleep(sleep_time);
        }

        // Spin-wait for remaining time
        let target_end = self.frame_start + self.target_frame_time;
        while Instant::now() < target_end {
            std::hint::spin_loop();
        }

        // Track error for next frame
        let actual_elapsed = self.frame_start.elapsed();
        let target_ns = self.target_frame_time.as_nanos() as i64;
        let actual_ns = actual_elapsed.as_nanos() as i64;
        self.accumulated_error = target_ns - actual_ns;
    }

    /// Set the pacing mode.
    ///
    /// # Arguments
    ///
    /// * `mode` - The new pacing mode.
    pub fn set_mode(&mut self, mode: FramePacingMode) {
        self.mode = mode;
        self.target_frame_time = Self::compute_target_frame_time(&mode, self.display_refresh_rate);
        self.accumulated_error = 0;
    }

    /// Get the current pacing mode.
    pub fn mode(&self) -> &FramePacingMode {
        &self.mode
    }

    /// Get the target FPS.
    ///
    /// Returns the effective target FPS based on mode:
    /// - Capped: returns the capped value
    /// - VSync: returns display refresh rate
    /// - Adaptive: returns current adaptive target
    /// - Uncapped: returns 0.0
    pub fn target_fps(&self) -> f64 {
        match &self.mode {
            FramePacingMode::Capped(fps) => *fps,
            FramePacingMode::VSync => self.display_refresh_rate,
            FramePacingMode::Adaptive => self.adaptive_target_fps,
            FramePacingMode::Uncapped => 0.0,
        }
    }

    /// Get the target frame time.
    pub fn target_frame_time(&self) -> Duration {
        self.target_frame_time
    }

    /// Set the display refresh rate hint.
    ///
    /// Used for VSync and Adaptive modes.
    pub fn set_display_refresh_rate(&mut self, hz: f64) {
        self.display_refresh_rate = hz.max(1.0);
        if self.mode.is_vsync() || self.mode.is_adaptive() {
            self.target_frame_time =
                Self::compute_target_frame_time(&self.mode, self.display_refresh_rate);
        }
    }

    /// Get frame timing statistics.
    ///
    /// Computes statistics from the frame history.
    pub fn stats(&self) -> FrameTimingStats {
        if self.frame_history.is_empty() {
            return FrameTimingStats::empty();
        }

        // Calculate average frame time
        let total_ns: u128 = self.frame_history.iter().map(|d| d.as_nanos()).sum();
        let count = self.frame_history.len();
        let avg_ns = total_ns / count as u128;
        let frame_time_ms = avg_ns as f64 / 1_000_000.0;
        let fps = if frame_time_ms > 0.0 {
            1000.0 / frame_time_ms
        } else {
            0.0
        };

        // Calculate variance
        let variance_ns = self.frame_time_variance_ns();
        let frame_variance_ms = (variance_ns as f64 / 1_000_000.0).sqrt();

        // Calculate budget usage
        let frame_budget_used_pct = if !self.target_frame_time.is_zero() {
            let target_ns = self.target_frame_time.as_nanos() as f64;
            (avg_ns as f64 / target_ns) * 100.0
        } else {
            0.0
        };

        FrameTimingStats {
            frame_time_ms,
            fps,
            frame_variance_ms,
            frame_budget_used_pct,
            missed_frames: self.missed_frames,
        }
    }

    /// Get the average frame time from history.
    pub fn average_frame_time(&self) -> Duration {
        if self.frame_history.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.frame_history.iter().sum();
        total / self.frame_history.len() as u32
    }

    /// Get the frame time variance (in seconds squared).
    pub fn frame_time_variance(&self) -> f64 {
        self.frame_time_variance_ns() as f64 / 1_000_000_000_000_000_000.0
    }

    /// Internal: variance in nanoseconds squared.
    fn frame_time_variance_ns(&self) -> u128 {
        if self.frame_history.len() < 2 {
            return 0;
        }

        let mean_ns = self
            .frame_history
            .iter()
            .map(|d| d.as_nanos())
            .sum::<u128>()
            / self.frame_history.len() as u128;

        let variance: u128 = self
            .frame_history
            .iter()
            .map(|d| {
                let diff = if d.as_nanos() > mean_ns {
                    d.as_nanos() - mean_ns
                } else {
                    mean_ns - d.as_nanos()
                };
                diff * diff
            })
            .sum::<u128>()
            / (self.frame_history.len() - 1) as u128;

        variance
    }

    /// Get the total frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the number of missed frames.
    pub fn missed_frame_count(&self) -> u64 {
        self.missed_frames
    }

    /// Get the last frame time.
    pub fn last_frame_time(&self) -> Duration {
        self.last_frame_time
    }

    /// Reset all statistics and history.
    pub fn reset(&mut self) {
        self.frame_history.clear();
        self.frame_count = 0;
        self.missed_frames = 0;
        self.accumulated_error = 0;
        self.last_frame_time = Duration::ZERO;
    }

    /// Get the history size.
    pub fn history_size(&self) -> usize {
        self.history_size
    }

    /// Get current history length.
    pub fn history_len(&self) -> usize {
        self.frame_history.len()
    }
}

impl Default for FramePacerV2 {
    fn default() -> Self {
        Self::new(FramePacingMode::default())
    }
}

// ============================================================================
// FrameBudget
// ============================================================================

/// Per-frame budget tracker with checkpoints.
///
/// Tracks time spent in different parts of the frame and detects
/// when the frame exceeds its budget.
///
/// # Example
///
/// ```ignore
/// let mut budget = FrameBudget::new(60.0);  // 16.67ms budget
///
/// // Simulation
/// update_physics();
/// budget.checkpoint("physics");
///
/// // Rendering
/// render_scene();
/// budget.checkpoint("render");
///
/// // Post-process
/// apply_post_effects();
/// budget.checkpoint("post");
///
/// if budget.is_over_budget() {
///     println!("Frame exceeded budget by {:.2}ms", -budget.remaining().as_secs_f64() * 1000.0);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FrameBudget {
    /// Total budget for the frame.
    total_budget: Duration,

    /// When the budget tracking started.
    start_time: Instant,

    /// Named checkpoints with cumulative elapsed time.
    checkpoints: Vec<(String, Duration)>,
}

impl FrameBudget {
    /// Create a new frame budget for the target FPS.
    ///
    /// # Arguments
    ///
    /// * `target_fps` - Target frames per second (e.g., 60.0 for 16.67ms budget).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let budget = FrameBudget::new(60.0);
    /// assert!((budget.total_budget().as_secs_f64() * 1000.0 - 16.67).abs() < 0.1);
    /// ```
    pub fn new(target_fps: f64) -> Self {
        let total_budget = if target_fps > 0.0 {
            Duration::from_secs_f64(1.0 / target_fps)
        } else {
            Duration::MAX
        };

        Self {
            total_budget,
            start_time: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    /// Create a frame budget from a specific frame time.
    ///
    /// # Arguments
    ///
    /// * `frame_time` - Target frame duration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let budget = FrameBudget::from_frame_time(Duration::from_millis(16));
    /// ```
    pub fn from_frame_time(frame_time: Duration) -> Self {
        Self {
            total_budget: frame_time,
            start_time: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    /// Record a checkpoint with the given name.
    ///
    /// Checkpoints track cumulative time from the budget start.
    ///
    /// # Arguments
    ///
    /// * `name` - Descriptive name for this checkpoint.
    pub fn checkpoint(&mut self, name: &str) {
        let elapsed = self.start_time.elapsed();
        self.checkpoints.push((name.to_string(), elapsed));
    }

    /// Get the total elapsed time since budget start.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get the remaining budget time.
    ///
    /// Returns `Duration::ZERO` if over budget.
    pub fn remaining(&self) -> Duration {
        let elapsed = self.start_time.elapsed();
        self.total_budget.saturating_sub(elapsed)
    }

    /// Get the percentage of budget used (0-100+).
    ///
    /// Values over 100 indicate the frame is over budget.
    pub fn used_percentage(&self) -> f64 {
        if self.total_budget.is_zero() {
            return 0.0;
        }
        let elapsed = self.start_time.elapsed();
        (elapsed.as_secs_f64() / self.total_budget.as_secs_f64()) * 100.0
    }

    /// Check if the frame is over budget.
    pub fn is_over_budget(&self) -> bool {
        self.start_time.elapsed() > self.total_budget
    }

    /// Reset the budget tracker for a new frame.
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.checkpoints.clear();
    }

    /// Get all checkpoints.
    pub fn checkpoints(&self) -> &[(String, Duration)] {
        &self.checkpoints
    }

    /// Get the total budget duration.
    pub fn total_budget(&self) -> Duration {
        self.total_budget
    }

    /// Get time spent between checkpoints.
    ///
    /// Returns a vector of (name, delta_time) pairs showing
    /// how much time was spent in each section.
    pub fn checkpoint_deltas(&self) -> Vec<(&str, Duration)> {
        let mut deltas = Vec::with_capacity(self.checkpoints.len());
        let mut prev_time = Duration::ZERO;

        for (name, time) in &self.checkpoints {
            let delta = time.saturating_sub(prev_time);
            deltas.push((name.as_str(), delta));
            prev_time = *time;
        }

        deltas
    }

    /// Get the name and duration of the most expensive checkpoint section.
    pub fn slowest_section(&self) -> Option<(&str, Duration)> {
        self.checkpoint_deltas()
            .into_iter()
            .max_by_key(|(_, d)| *d)
    }

    /// Set a new total budget.
    pub fn set_budget(&mut self, budget: Duration) {
        self.total_budget = budget;
    }

    /// Set budget from target FPS.
    pub fn set_target_fps(&mut self, fps: f64) {
        self.total_budget = if fps > 0.0 {
            Duration::from_secs_f64(1.0 / fps)
        } else {
            Duration::MAX
        };
    }
}

impl Default for FrameBudget {
    fn default() -> Self {
        Self::new(60.0)
    }
}

// ============================================================================
// SmoothedFrameTime
// ============================================================================

/// Low-pass filtered frame time for smooth display.
///
/// Uses exponential moving average to smooth out frame time jitter
/// for stable FPS display and timing decisions.
///
/// # Example
///
/// ```ignore
/// let mut smooth = SmoothedFrameTime::new(0.1);  // 10% weight to new samples
///
/// for frame_time in frame_times {
///     smooth.update(frame_time);
///     println!("Smoothed FPS: {:.1}", smooth.fps());
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SmoothedFrameTime {
    /// Current smoothed frame time in milliseconds.
    smoothed_ms: f64,

    /// Smoothing factor (0.0-1.0).
    /// Lower values = more smoothing, higher values = more responsive.
    alpha: f64,
}

impl SmoothedFrameTime {
    /// Create a new smoothed frame time filter.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Smoothing factor (0.0-1.0). Lower = more smoothing.
    ///   - 0.05: Very smooth (slow to respond)
    ///   - 0.1: Smooth (good for display)
    ///   - 0.2: Responsive (good for decisions)
    ///   - 0.5: Very responsive (minimal smoothing)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let smooth = SmoothedFrameTime::new(0.1);  // Good for FPS counter
    /// ```
    pub fn new(alpha: f64) -> Self {
        Self {
            smoothed_ms: 16.67, // Default to 60 FPS
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Update with a new frame time sample.
    ///
    /// # Arguments
    ///
    /// * `frame_time_ms` - The new frame time in milliseconds.
    pub fn update(&mut self, frame_time_ms: f64) {
        // Exponential moving average: new = alpha * sample + (1-alpha) * old
        self.smoothed_ms = self.alpha * frame_time_ms + (1.0 - self.alpha) * self.smoothed_ms;
    }

    /// Update with a Duration.
    pub fn update_duration(&mut self, frame_time: Duration) {
        self.update(frame_time.as_secs_f64() * 1000.0);
    }

    /// Get the current smoothed frame time in milliseconds.
    pub fn value(&self) -> f64 {
        self.smoothed_ms
    }

    /// Get the smoothed FPS.
    pub fn fps(&self) -> f64 {
        if self.smoothed_ms > 0.0 {
            1000.0 / self.smoothed_ms
        } else {
            0.0
        }
    }

    /// Get the smoothing factor.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set a new smoothing factor.
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }

    /// Reset to a specific frame time.
    pub fn reset(&mut self, frame_time_ms: f64) {
        self.smoothed_ms = frame_time_ms;
    }

    /// Reset to a specific FPS.
    pub fn reset_to_fps(&mut self, fps: f64) {
        if fps > 0.0 {
            self.smoothed_ms = 1000.0 / fps;
        }
    }
}

impl Default for SmoothedFrameTime {
    fn default() -> Self {
        Self::new(0.1)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- FramePacingMode tests --

    #[test]
    fn test_pacing_mode_vsync() {
        let mode = FramePacingMode::VSync;
        assert!(mode.is_vsync());
        assert!(!mode.is_capped());
        assert!(!mode.is_uncapped());
        assert!(!mode.is_adaptive());
        assert_eq!(mode.target_fps(), None);
        assert_eq!(mode.name(), "VSync");
    }

    #[test]
    fn test_pacing_mode_uncapped() {
        let mode = FramePacingMode::Uncapped;
        assert!(!mode.is_vsync());
        assert!(!mode.is_capped());
        assert!(mode.is_uncapped());
        assert!(!mode.is_adaptive());
        assert_eq!(mode.target_fps(), None);
        assert_eq!(mode.name(), "Uncapped");
    }

    #[test]
    fn test_pacing_mode_capped() {
        let mode = FramePacingMode::Capped(144.0);
        assert!(!mode.is_vsync());
        assert!(mode.is_capped());
        assert!(!mode.is_uncapped());
        assert!(!mode.is_adaptive());
        assert_eq!(mode.target_fps(), Some(144.0));
        assert_eq!(mode.name(), "Capped");
    }

    #[test]
    fn test_pacing_mode_adaptive() {
        let mode = FramePacingMode::Adaptive;
        assert!(!mode.is_vsync());
        assert!(!mode.is_capped());
        assert!(!mode.is_uncapped());
        assert!(mode.is_adaptive());
        assert_eq!(mode.target_fps(), None);
        assert_eq!(mode.name(), "Adaptive");
    }

    #[test]
    fn test_pacing_mode_default() {
        let mode = FramePacingMode::default();
        assert!(mode.is_vsync());
    }

    #[test]
    fn test_pacing_mode_capped_zero_fps() {
        let mode = FramePacingMode::Capped(0.0);
        assert_eq!(mode.target_fps(), Some(0.0));
    }

    #[test]
    fn test_pacing_mode_descriptions() {
        assert!(!FramePacingMode::VSync.description().is_empty());
        assert!(!FramePacingMode::Uncapped.description().is_empty());
        assert!(!FramePacingMode::Capped(60.0).description().is_empty());
        assert!(!FramePacingMode::Adaptive.description().is_empty());
    }

    // -- FramePacerV2 initialization tests --

    #[test]
    fn test_pacer_new_vsync() {
        let pacer = FramePacerV2::new(FramePacingMode::VSync);
        assert!(pacer.mode().is_vsync());
        assert_eq!(pacer.frame_count(), 0);
        assert_eq!(pacer.missed_frame_count(), 0);
    }

    #[test]
    fn test_pacer_new_capped() {
        let pacer = FramePacerV2::new(FramePacingMode::Capped(60.0));
        assert!(pacer.mode().is_capped());
        assert!((pacer.target_fps() - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_pacer_new_uncapped() {
        let pacer = FramePacerV2::new(FramePacingMode::Uncapped);
        assert!(pacer.mode().is_uncapped());
        assert_eq!(pacer.target_fps(), 0.0);
    }

    #[test]
    fn test_pacer_new_adaptive() {
        let pacer = FramePacerV2::new(FramePacingMode::Adaptive);
        assert!(pacer.mode().is_adaptive());
    }

    #[test]
    fn test_pacer_with_history_size() {
        let pacer = FramePacerV2::with_history_size(FramePacingMode::Capped(60.0), 30);
        assert_eq!(pacer.history_size(), 30);
    }

    #[test]
    fn test_pacer_with_history_size_minimum() {
        let pacer = FramePacerV2::with_history_size(FramePacingMode::Capped(60.0), 0);
        assert_eq!(pacer.history_size(), 1); // minimum is 1
    }

    // -- FramePacerV2 frame cycle tests --

    #[test]
    fn test_pacer_begin_end_frame() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Capped(60.0));

        pacer.begin_frame();
        std::thread::sleep(Duration::from_millis(1));
        let frame_time = pacer.end_frame();

        assert!(frame_time >= Duration::from_millis(1));
        assert_eq!(pacer.frame_count(), 1);
        assert_eq!(pacer.history_len(), 1);
    }

    #[test]
    fn test_pacer_multiple_frames() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Uncapped);

        for _ in 0..5 {
            pacer.begin_frame();
            pacer.end_frame();
        }

        assert_eq!(pacer.frame_count(), 5);
        assert_eq!(pacer.history_len(), 5);
    }

    #[test]
    fn test_pacer_auto_end_frame() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Uncapped);

        pacer.begin_frame();
        pacer.begin_frame(); // Should auto-end previous frame
        pacer.end_frame();

        assert_eq!(pacer.frame_count(), 2);
    }

    #[test]
    fn test_pacer_end_frame_without_begin() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Uncapped);
        let frame_time = pacer.end_frame();
        assert_eq!(frame_time, Duration::ZERO);
        assert_eq!(pacer.frame_count(), 0);
    }

    // -- FramePacerV2 wait_for_target tests --

    #[test]
    fn test_pacer_wait_uncapped_no_wait() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Uncapped);
        pacer.begin_frame();
        pacer.end_frame();

        let start = Instant::now();
        pacer.wait_for_target();
        let elapsed = start.elapsed();

        // Should not wait
        assert!(elapsed < Duration::from_millis(1));
    }

    #[test]
    fn test_pacer_wait_vsync_no_software_wait() {
        let mut pacer = FramePacerV2::new(FramePacingMode::VSync);
        pacer.begin_frame();
        pacer.end_frame();

        let start = Instant::now();
        pacer.wait_for_target();
        let elapsed = start.elapsed();

        // VSync relies on hardware, no software wait
        assert!(elapsed < Duration::from_millis(1));
    }

    #[test]
    fn test_pacer_wait_capped() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Capped(1000.0)); // 1ms target
        pacer.begin_frame();
        // Fast frame, should wait
        pacer.end_frame();

        let start = Instant::now();
        pacer.wait_for_target();
        let elapsed = start.elapsed();

        // Should wait close to 1ms (some tolerance for overhead)
        // Note: This test may be flaky on busy systems
        assert!(elapsed >= Duration::from_micros(500));
    }

    // -- FramePacerV2 statistics tests --

    #[test]
    fn test_pacer_stats_empty() {
        let pacer = FramePacerV2::new(FramePacingMode::Uncapped);
        let stats = pacer.stats();
        assert_eq!(stats.frame_time_ms, 0.0);
        assert_eq!(stats.fps, 0.0);
        assert_eq!(stats.missed_frames, 0);
    }

    #[test]
    fn test_pacer_stats_with_frames() {
        let mut pacer = FramePacerV2::with_history_size(FramePacingMode::Uncapped, 10);

        for _ in 0..5 {
            pacer.begin_frame();
            std::thread::sleep(Duration::from_micros(100));
            pacer.end_frame();
        }

        let stats = pacer.stats();
        assert!(stats.frame_time_ms > 0.0);
        assert!(stats.fps > 0.0);
    }

    #[test]
    fn test_pacer_average_frame_time() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Uncapped);

        for _ in 0..3 {
            pacer.begin_frame();
            std::thread::sleep(Duration::from_micros(500));
            pacer.end_frame();
        }

        let avg = pacer.average_frame_time();
        assert!(avg >= Duration::from_micros(500));
    }

    #[test]
    fn test_pacer_frame_time_variance() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Uncapped);

        // All same duration -> low variance
        for _ in 0..5 {
            pacer.begin_frame();
            std::thread::sleep(Duration::from_micros(100));
            pacer.end_frame();
        }

        let variance = pacer.frame_time_variance();
        // Variance exists but should be relatively small for similar frame times
        assert!(variance >= 0.0);
    }

    #[test]
    fn test_pacer_variance_insufficient_samples() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Uncapped);
        pacer.begin_frame();
        pacer.end_frame();

        // Need at least 2 samples for variance
        assert_eq!(pacer.frame_time_variance(), 0.0);
    }

    // -- FrameBudget tests --

    #[test]
    fn test_budget_new_60fps() {
        let budget = FrameBudget::new(60.0);
        let expected_ms = 1000.0 / 60.0;
        let actual_ms = budget.total_budget().as_secs_f64() * 1000.0;
        assert!((actual_ms - expected_ms).abs() < 0.1);
    }

    #[test]
    fn test_budget_from_frame_time() {
        let budget = FrameBudget::from_frame_time(Duration::from_millis(16));
        assert_eq!(budget.total_budget(), Duration::from_millis(16));
    }

    #[test]
    fn test_budget_checkpoint() {
        let mut budget = FrameBudget::new(60.0);
        std::thread::sleep(Duration::from_micros(100));
        budget.checkpoint("test");

        assert_eq!(budget.checkpoints().len(), 1);
        assert_eq!(budget.checkpoints()[0].0, "test");
        assert!(budget.checkpoints()[0].1 >= Duration::from_micros(100));
    }

    #[test]
    fn test_budget_multiple_checkpoints() {
        let mut budget = FrameBudget::new(60.0);
        budget.checkpoint("first");
        std::thread::sleep(Duration::from_micros(100));
        budget.checkpoint("second");

        assert_eq!(budget.checkpoints().len(), 2);
        assert!(budget.checkpoints()[1].1 > budget.checkpoints()[0].1);
    }

    #[test]
    fn test_budget_elapsed() {
        let budget = FrameBudget::new(60.0);
        std::thread::sleep(Duration::from_micros(100));
        assert!(budget.elapsed() >= Duration::from_micros(100));
    }

    #[test]
    fn test_budget_remaining() {
        let budget = FrameBudget::from_frame_time(Duration::from_millis(100));
        let remaining = budget.remaining();
        assert!(remaining <= Duration::from_millis(100));
    }

    #[test]
    fn test_budget_used_percentage() {
        let budget = FrameBudget::from_frame_time(Duration::from_millis(100));
        std::thread::sleep(Duration::from_millis(10));
        let pct = budget.used_percentage();
        assert!(pct >= 10.0); // At least 10%
        assert!(pct < 100.0); // Less than 100%
    }

    #[test]
    fn test_budget_is_over_budget() {
        let budget = FrameBudget::from_frame_time(Duration::from_micros(100));
        assert!(!budget.is_over_budget());

        std::thread::sleep(Duration::from_micros(200));
        assert!(budget.is_over_budget());
    }

    #[test]
    fn test_budget_reset() {
        let mut budget = FrameBudget::new(60.0);
        budget.checkpoint("test");
        std::thread::sleep(Duration::from_micros(100));
        budget.reset();

        assert!(budget.checkpoints().is_empty());
        assert!(budget.elapsed() < Duration::from_micros(100));
    }

    #[test]
    fn test_budget_checkpoint_deltas() {
        let mut budget = FrameBudget::new(60.0);
        budget.checkpoint("first");
        std::thread::sleep(Duration::from_micros(100));
        budget.checkpoint("second");

        let deltas = budget.checkpoint_deltas();
        assert_eq!(deltas.len(), 2);
        assert!(deltas[1].1 >= Duration::from_micros(100));
    }

    #[test]
    fn test_budget_slowest_section() {
        let mut budget = FrameBudget::new(60.0);
        budget.checkpoint("fast");
        std::thread::sleep(Duration::from_micros(200));
        budget.checkpoint("slow");
        budget.checkpoint("fastest");

        let slowest = budget.slowest_section();
        assert!(slowest.is_some());
        assert_eq!(slowest.unwrap().0, "slow");
    }

    #[test]
    fn test_budget_zero_fps() {
        let budget = FrameBudget::new(0.0);
        assert_eq!(budget.total_budget(), Duration::MAX);
        assert!(!budget.is_over_budget());
    }

    #[test]
    fn test_budget_set_budget() {
        let mut budget = FrameBudget::new(60.0);
        budget.set_budget(Duration::from_millis(10));
        assert_eq!(budget.total_budget(), Duration::from_millis(10));
    }

    #[test]
    fn test_budget_set_target_fps() {
        let mut budget = FrameBudget::new(30.0);
        budget.set_target_fps(120.0);
        let expected_ms = 1000.0 / 120.0;
        let actual_ms = budget.total_budget().as_secs_f64() * 1000.0;
        assert!((actual_ms - expected_ms).abs() < 0.1);
    }

    // -- SmoothedFrameTime tests --

    #[test]
    fn test_smoothed_new() {
        let smooth = SmoothedFrameTime::new(0.1);
        assert!((smooth.alpha() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_smoothed_default_value() {
        let smooth = SmoothedFrameTime::new(0.1);
        assert!((smooth.value() - 16.67).abs() < 0.1); // Default ~60 FPS
    }

    #[test]
    fn test_smoothed_update() {
        let mut smooth = SmoothedFrameTime::new(0.5);
        smooth.update(10.0);

        // With alpha=0.5: new = 0.5*10 + 0.5*16.67 = ~13.33
        assert!(smooth.value() < 16.67);
        assert!(smooth.value() > 10.0);
    }

    #[test]
    fn test_smoothed_converges() {
        let mut smooth = SmoothedFrameTime::new(0.5);

        // Update with same value many times
        for _ in 0..20 {
            smooth.update(10.0);
        }

        // Should converge close to 10ms
        assert!((smooth.value() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_smoothed_fps() {
        let mut smooth = SmoothedFrameTime::new(0.1);
        smooth.reset(10.0); // 10ms = 100 FPS

        assert!((smooth.fps() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_smoothed_fps_zero() {
        let mut smooth = SmoothedFrameTime::new(0.1);
        smooth.reset(0.0);
        assert_eq!(smooth.fps(), 0.0);
    }

    #[test]
    fn test_smoothed_update_duration() {
        let mut smooth = SmoothedFrameTime::new(1.0);
        smooth.update_duration(Duration::from_millis(8));

        assert!((smooth.value() - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_smoothed_alpha_clamped() {
        let smooth = SmoothedFrameTime::new(2.0);
        assert_eq!(smooth.alpha(), 1.0);

        let smooth = SmoothedFrameTime::new(-1.0);
        assert_eq!(smooth.alpha(), 0.0);
    }

    #[test]
    fn test_smoothed_set_alpha() {
        let mut smooth = SmoothedFrameTime::new(0.1);
        smooth.set_alpha(0.5);
        assert!((smooth.alpha() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_smoothed_reset() {
        let mut smooth = SmoothedFrameTime::new(0.1);
        smooth.reset(20.0);
        assert!((smooth.value() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_smoothed_reset_to_fps() {
        let mut smooth = SmoothedFrameTime::new(0.1);
        smooth.reset_to_fps(50.0); // 50 FPS = 20ms

        assert!((smooth.value() - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_smoothed_default() {
        let smooth = SmoothedFrameTime::default();
        assert!((smooth.alpha() - 0.1).abs() < 0.001);
    }

    // -- FrameTimingStats tests --

    #[test]
    fn test_timing_stats_empty() {
        let stats = FrameTimingStats::empty();
        assert_eq!(stats.fps, 0.0);
        assert_eq!(stats.frame_time_ms, 0.0);
        assert_eq!(stats.missed_frames, 0);
    }

    #[test]
    fn test_timing_stats_is_consistent() {
        let stats = FrameTimingStats {
            frame_time_ms: 16.67,
            fps: 60.0,
            frame_variance_ms: 0.5,
            frame_budget_used_pct: 90.0,
            missed_frames: 0,
        };

        assert!(stats.is_consistent(1.0));
        assert!(!stats.is_consistent(0.1));
    }

    #[test]
    fn test_timing_stats_is_on_target() {
        let on_target = FrameTimingStats {
            frame_budget_used_pct: 95.0,
            ..Default::default()
        };
        assert!(on_target.is_on_target());

        let over_budget = FrameTimingStats {
            frame_budget_used_pct: 120.0,
            ..Default::default()
        };
        assert!(!over_budget.is_on_target());
    }

    #[test]
    fn test_timing_stats_std_dev() {
        let stats = FrameTimingStats {
            frame_variance_ms: 4.0, // sqrt(4) = 2
            ..Default::default()
        };

        assert!((stats.frame_std_dev_ms() - 2.0).abs() < 0.001);
    }

    // -- Mode switching tests --

    #[test]
    fn test_pacer_set_mode() {
        let mut pacer = FramePacerV2::new(FramePacingMode::VSync);
        pacer.set_mode(FramePacingMode::Capped(120.0));

        assert!(pacer.mode().is_capped());
        assert!((pacer.target_fps() - 120.0).abs() < 0.001);
    }

    #[test]
    fn test_pacer_set_display_refresh_rate() {
        let mut pacer = FramePacerV2::new(FramePacingMode::VSync);
        pacer.set_display_refresh_rate(144.0);

        assert!((pacer.target_fps() - 144.0).abs() < 0.001);
    }

    #[test]
    fn test_pacer_reset() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Uncapped);

        for _ in 0..10 {
            pacer.begin_frame();
            pacer.end_frame();
        }

        pacer.reset();
        assert_eq!(pacer.frame_count(), 0);
        assert_eq!(pacer.history_len(), 0);
    }

    // -- Edge cases --

    #[test]
    fn test_pacer_very_high_fps() {
        let pacer = FramePacerV2::new(FramePacingMode::Capped(10000.0));
        let expected_frame_time_us = 100; // 10000 FPS = 0.1ms = 100us
        let actual_us = pacer.target_frame_time().as_micros();

        assert!((actual_us as i64 - expected_frame_time_us as i64).abs() < 10);
    }

    #[test]
    fn test_pacer_capped_zero_fps() {
        let pacer = FramePacerV2::new(FramePacingMode::Capped(0.0));
        assert_eq!(pacer.target_frame_time(), Duration::ZERO);
    }

    #[test]
    fn test_budget_used_percentage_zero_budget() {
        let budget = FrameBudget::from_frame_time(Duration::ZERO);
        assert_eq!(budget.used_percentage(), 0.0);
    }

    #[test]
    fn test_smoothed_alpha_zero() {
        let mut smooth = SmoothedFrameTime::new(0.0);
        smooth.update(100.0);
        // With alpha=0, no change
        assert!((smooth.value() - 16.67).abs() < 0.1);
    }

    #[test]
    fn test_smoothed_alpha_one() {
        let mut smooth = SmoothedFrameTime::new(1.0);
        smooth.update(100.0);
        // With alpha=1, immediate update
        assert!((smooth.value() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_pacer_last_frame_time() {
        let mut pacer = FramePacerV2::new(FramePacingMode::Uncapped);
        assert_eq!(pacer.last_frame_time(), Duration::ZERO);

        pacer.begin_frame();
        std::thread::sleep(Duration::from_micros(100));
        pacer.end_frame();

        assert!(pacer.last_frame_time() >= Duration::from_micros(100));
    }

    #[test]
    fn test_pacer_history_wraps() {
        let mut pacer = FramePacerV2::with_history_size(FramePacingMode::Uncapped, 5);

        for _ in 0..10 {
            pacer.begin_frame();
            pacer.end_frame();
        }

        assert_eq!(pacer.history_len(), 5); // Capped at history size
        assert_eq!(pacer.frame_count(), 10);
    }
}
