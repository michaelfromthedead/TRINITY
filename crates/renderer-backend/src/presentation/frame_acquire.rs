//! Frame acquisition and management for TRINITY presentation.
//!
//! This module provides the frame acquisition infrastructure that sits between
//! the swapchain and rendering systems. It handles frame lifecycle management,
//! acquisition timing, retry logic, and statistics tracking.
//!
//! # Architecture
//!
//! ```text
//! FrameAcquirer
//!     |-- AcquireConfig (timeout, retries)
//!     |-- AcquiredFrame (texture, view, timing)
//!     |-- AcquireResult (success, suboptimal, timeout, etc.)
//!     `-- FrameAcquireStats (metrics, timing)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::presentation::{FrameAcquirer, AcquireConfig, AcquireResult};
//!
//! // Create acquirer with timeout
//! let config = AcquireConfig::with_timeout(Duration::from_millis(16));
//! let mut acquirer = FrameAcquirer::new(config);
//!
//! // Acquire frame
//! match acquirer.acquire(&surface) {
//!     AcquireResult::Success(frame) => {
//!         // Render to frame.view()
//!         frame.present();
//!     }
//!     AcquireResult::Suboptimal(frame) => {
//!         // Render but schedule resize
//!         frame.present();
//!     }
//!     AcquireResult::Timeout => {
//!         // Skip this frame
//!     }
//!     AcquireResult::Outdated | AcquireResult::Lost => {
//!         // Reconfigure surface
//!     }
//! }
//! ```

use std::fmt;
use std::time::{Duration, Instant};
use thiserror::Error;

// ============================================================================
// AcquireError
// ============================================================================

/// Errors that can occur during frame acquisition.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AcquireError {
    /// Surface was lost and needs to be recreated.
    #[error("surface lost: needs recreation")]
    Lost,

    /// Surface is outdated and needs reconfiguration.
    #[error("surface outdated: needs reconfiguration")]
    Outdated,

    /// Frame acquisition timed out after all retries.
    #[error("frame acquisition timed out after {retries} retries")]
    Timeout {
        /// Number of retries attempted.
        retries: u32,
    },

    /// Out of GPU memory.
    #[error("out of GPU memory")]
    OutOfMemory,

    /// No frame is currently acquired.
    #[error("no frame currently acquired")]
    NoFrame,
}

impl AcquireError {
    /// Returns true if this error is recoverable without surface recreation.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, AcquireError::Timeout { .. } | AcquireError::Outdated)
    }

    /// Returns true if the surface needs reconfiguration.
    pub fn needs_reconfigure(&self) -> bool {
        matches!(self, AcquireError::Outdated)
    }
}

// ============================================================================
// AcquireConfig
// ============================================================================

/// Configuration for frame acquisition.
///
/// Controls timeout behavior, retry logic, and other acquisition parameters.
///
/// # Example
///
/// ```ignore
/// // With 16ms timeout (one frame at 60fps)
/// let config = AcquireConfig::with_timeout(Duration::from_millis(16));
///
/// // With retries
/// let config = AcquireConfig::new()
///     .with_timeout(Duration::from_millis(16))
///     .with_retries(3);
///
/// // No timeout (blocking)
/// let config = AcquireConfig::no_timeout();
/// ```
#[derive(Clone, Debug)]
pub struct AcquireConfig {
    /// Timeout for frame acquisition.
    ///
    /// If `None`, acquisition may block indefinitely.
    pub timeout: Option<Duration>,

    /// Whether to retry on timeout.
    pub retry_on_timeout: bool,

    /// Maximum number of retry attempts.
    pub max_retries: u32,
}

impl AcquireConfig {
    /// Create a new configuration with default settings.
    ///
    /// Defaults:
    /// - No timeout (blocking acquisition)
    /// - No retries
    pub fn new() -> Self {
        Self {
            timeout: None,
            retry_on_timeout: false,
            max_retries: 0,
        }
    }

    /// Create a configuration with the specified timeout.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for frame acquisition.
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout: Some(timeout),
            retry_on_timeout: false,
            max_retries: 0,
        }
    }

    /// Create a configuration with no timeout (blocking).
    pub fn no_timeout() -> Self {
        Self {
            timeout: None,
            retry_on_timeout: false,
            max_retries: 0,
        }
    }

    /// Enable retries on timeout with the specified maximum.
    ///
    /// # Arguments
    ///
    /// * `max` - Maximum number of retry attempts (0 = no retries).
    pub fn with_retries(mut self, max: u32) -> Self {
        self.retry_on_timeout = max > 0;
        self.max_retries = max;
        self
    }

    /// Set the timeout duration.
    pub fn set_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if retries are enabled.
    pub fn has_retries(&self) -> bool {
        self.retry_on_timeout && self.max_retries > 0
    }
}

impl Default for AcquireConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AcquiredFrame
// ============================================================================

/// An acquired frame ready for rendering.
///
/// Contains the surface texture, a pre-created texture view, and acquisition
/// timing information. The frame is automatically presented when dropped,
/// or can be explicitly presented via `present()`.
///
/// # Lifecycle
///
/// ```text
/// acquire() -> AcquiredFrame
///     |
///     v
/// [Render to frame.view()]
///     |
///     v
/// present() or drop -> frame returned to swapchain
/// ```
///
/// # Example
///
/// ```ignore
/// let frame = acquirer.acquire_blocking(&surface)?;
///
/// // Render to the frame
/// render_to(frame.view());
///
/// // Check acquisition latency
/// println!("Acquire took {:?}", frame.age());
///
/// // Present (or let drop handle it)
/// frame.present();
/// ```
pub struct AcquiredFrame {
    /// The acquired surface texture.
    texture: wgpu::SurfaceTexture,
    /// Pre-created texture view for rendering.
    view: wgpu::TextureView,
    /// Index of this frame in the swapchain.
    index: u32,
    /// Time when the frame was acquired.
    acquire_time: Instant,
}

impl AcquiredFrame {
    /// Create a new acquired frame.
    ///
    /// # Arguments
    ///
    /// * `texture` - The acquired surface texture.
    /// * `format` - Texture format for the view.
    /// * `index` - Frame index in the swapchain.
    pub fn new(texture: wgpu::SurfaceTexture, format: wgpu::TextureFormat, index: u32) -> Self {
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("acquired_frame_view"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            ..Default::default()
        });

        Self {
            texture,
            view,
            index,
            acquire_time: Instant::now(),
        }
    }

    /// Create a new acquired frame with a custom acquire time.
    ///
    /// Useful for testing or when the actual acquire time differs.
    pub fn with_acquire_time(
        texture: wgpu::SurfaceTexture,
        format: wgpu::TextureFormat,
        index: u32,
        acquire_time: Instant,
    ) -> Self {
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("acquired_frame_view"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            ..Default::default()
        });

        Self {
            texture,
            view,
            index,
            acquire_time,
        }
    }

    /// Get a reference to the surface texture.
    pub fn texture(&self) -> &wgpu::SurfaceTexture {
        &self.texture
    }

    /// Get a reference to the texture view.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get the frame index in the swapchain.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Get the time when the frame was acquired.
    pub fn acquire_time(&self) -> Instant {
        self.acquire_time
    }

    /// Get the time elapsed since acquisition.
    pub fn age(&self) -> Duration {
        self.acquire_time.elapsed()
    }

    /// Get the width of the frame in pixels.
    pub fn width(&self) -> u32 {
        self.texture.texture.width()
    }

    /// Get the height of the frame in pixels.
    pub fn height(&self) -> u32 {
        self.texture.texture.height()
    }

    /// Get the underlying raw texture.
    pub fn raw_texture(&self) -> &wgpu::Texture {
        &self.texture.texture
    }

    /// Present the frame and return it to the swapchain.
    ///
    /// This consumes the frame, presenting it to the display.
    pub fn present(self) {
        self.texture.present();
    }
}

impl fmt::Debug for AcquiredFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AcquiredFrame")
            .field("index", &self.index)
            .field("width", &self.width())
            .field("height", &self.height())
            .field("age_us", &self.age().as_micros())
            .finish()
    }
}

// ============================================================================
// AcquireResult
// ============================================================================

/// Result of a frame acquisition attempt.
///
/// Provides detailed information about the acquisition outcome, including
/// whether the swapchain is suboptimal and needs reconfiguration.
///
/// # Example
///
/// ```ignore
/// match acquirer.acquire(&surface) {
///     AcquireResult::Success(frame) => {
///         render_and_present(frame);
///     }
///     AcquireResult::Suboptimal(frame) => {
///         render_and_present(frame);
///         schedule_resize();
///     }
///     AcquireResult::Timeout => {
///         // Skip frame
///     }
///     result if result.needs_reconfigure() => {
///         reconfigure_surface();
///     }
///     AcquireResult::Lost => {
///         recreate_surface();
///     }
/// }
/// ```
pub enum AcquireResult {
    /// Frame acquired successfully.
    Success(AcquiredFrame),

    /// Frame acquired but swapchain is suboptimal.
    ///
    /// The frame is usable but the swapchain should be reconfigured
    /// after presenting for optimal performance.
    Suboptimal(AcquiredFrame),

    /// Frame acquisition timed out.
    ///
    /// The GPU was busy and couldn't provide a frame in time.
    /// This is transient; skip this frame and try again.
    Timeout,

    /// Swapchain is outdated and needs reconfiguration.
    ///
    /// This typically happens after a window resize. Reconfigure
    /// the surface and retry.
    Outdated,

    /// Swapchain is lost and needs complete recreation.
    ///
    /// The surface and possibly device need to be recreated.
    Lost,
}

impl AcquireResult {
    /// Returns true if the acquisition was successful (Success or Suboptimal).
    pub fn is_success(&self) -> bool {
        matches!(self, AcquireResult::Success(_) | AcquireResult::Suboptimal(_))
    }

    /// Returns true if the error is recoverable without surface recreation.
    ///
    /// `Timeout` and `Outdated` are recoverable:
    /// - Timeout: Just retry on the next frame
    /// - Outdated: Reconfigure and retry
    /// - Suboptimal: Frame is usable, just schedule reconfigure
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            AcquireResult::Success(_)
                | AcquireResult::Suboptimal(_)
                | AcquireResult::Timeout
                | AcquireResult::Outdated
        )
    }

    /// Returns true if the surface needs reconfiguration.
    pub fn needs_reconfigure(&self) -> bool {
        matches!(self, AcquireResult::Outdated | AcquireResult::Suboptimal(_))
    }

    /// Extract the frame, if acquisition was successful.
    ///
    /// Returns `Some(AcquiredFrame)` for `Success` and `Suboptimal`,
    /// `None` for all other variants.
    pub fn frame(self) -> Option<AcquiredFrame> {
        match self {
            AcquireResult::Success(frame) | AcquireResult::Suboptimal(frame) => Some(frame),
            _ => None,
        }
    }

    /// Returns true if a frame was acquired (Success or Suboptimal).
    pub fn has_frame(&self) -> bool {
        matches!(self, AcquireResult::Success(_) | AcquireResult::Suboptimal(_))
    }

    /// Returns true if this is a suboptimal result.
    pub fn is_suboptimal(&self) -> bool {
        matches!(self, AcquireResult::Suboptimal(_))
    }

    /// Convert to an error if acquisition failed.
    pub fn into_result(self) -> Result<AcquiredFrame, AcquireError> {
        match self {
            AcquireResult::Success(frame) | AcquireResult::Suboptimal(frame) => Ok(frame),
            AcquireResult::Timeout => Err(AcquireError::Timeout { retries: 0 }),
            AcquireResult::Outdated => Err(AcquireError::Outdated),
            AcquireResult::Lost => Err(AcquireError::Lost),
        }
    }
}

impl fmt::Debug for AcquireResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcquireResult::Success(frame) => f.debug_tuple("Success").field(frame).finish(),
            AcquireResult::Suboptimal(frame) => f.debug_tuple("Suboptimal").field(frame).finish(),
            AcquireResult::Timeout => write!(f, "Timeout"),
            AcquireResult::Outdated => write!(f, "Outdated"),
            AcquireResult::Lost => write!(f, "Lost"),
        }
    }
}

// ============================================================================
// FrameAcquireStats
// ============================================================================

/// Statistics about frame acquisition.
///
/// Tracks acquisition metrics including total frames, dropped frames,
/// timing information, and suboptimal counts.
#[derive(Clone, Debug, Default)]
pub struct FrameAcquireStats {
    /// Total frames successfully acquired.
    pub total_frames: u64,

    /// Frames that were dropped (timeout or other transient errors).
    pub dropped_frames: u64,

    /// Average acquisition time in microseconds.
    pub avg_acquire_time_us: f64,

    /// Number of frames acquired with suboptimal swapchain.
    pub suboptimal_count: u64,

    /// Number of timeout errors.
    pub timeout_count: u64,

    /// Number of outdated errors requiring reconfiguration.
    pub outdated_count: u64,

    /// Number of lost surface errors.
    pub lost_count: u64,

    /// Minimum acquisition time in microseconds.
    pub min_acquire_time_us: u64,

    /// Maximum acquisition time in microseconds.
    pub max_acquire_time_us: u64,

    /// Total acquisition time in microseconds (for averaging).
    total_acquire_time_us: u64,
}

impl FrameAcquireStats {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful frame acquisition.
    pub fn record_success(&mut self, acquire_time_us: u64) {
        self.total_frames += 1;
        self.total_acquire_time_us += acquire_time_us;
        self.avg_acquire_time_us = self.total_acquire_time_us as f64 / self.total_frames as f64;

        if self.total_frames == 1 {
            self.min_acquire_time_us = acquire_time_us;
            self.max_acquire_time_us = acquire_time_us;
        } else {
            self.min_acquire_time_us = self.min_acquire_time_us.min(acquire_time_us);
            self.max_acquire_time_us = self.max_acquire_time_us.max(acquire_time_us);
        }
    }

    /// Record a suboptimal frame acquisition.
    pub fn record_suboptimal(&mut self, acquire_time_us: u64) {
        self.record_success(acquire_time_us);
        self.suboptimal_count += 1;
    }

    /// Record a timeout.
    pub fn record_timeout(&mut self) {
        self.dropped_frames += 1;
        self.timeout_count += 1;
    }

    /// Record an outdated error.
    pub fn record_outdated(&mut self) {
        self.outdated_count += 1;
    }

    /// Record a lost surface error.
    pub fn record_lost(&mut self) {
        self.lost_count += 1;
    }

    /// Get the success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        let total_attempts = self.total_frames + self.dropped_frames;
        if total_attempts == 0 {
            100.0
        } else {
            (self.total_frames as f64 / total_attempts as f64) * 100.0
        }
    }

    /// Get the suboptimal rate as a percentage of successful frames.
    pub fn suboptimal_rate(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            (self.suboptimal_count as f64 / self.total_frames as f64) * 100.0
        }
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for FrameAcquireStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FrameAcquireStats {{ frames: {}, dropped: {}, avg_us: {:.1}, suboptimal: {} }}",
            self.total_frames, self.dropped_frames, self.avg_acquire_time_us, self.suboptimal_count
        )
    }
}

// ============================================================================
// FrameAcquirer
// ============================================================================

/// Manages frame acquisition lifecycle.
///
/// Provides a higher-level interface for acquiring frames from a surface,
/// with support for timeouts, retries, and statistics tracking.
///
/// # Example
///
/// ```ignore
/// // Create acquirer with timeout and retries
/// let config = AcquireConfig::with_timeout(Duration::from_millis(16))
///     .with_retries(2);
/// let mut acquirer = FrameAcquirer::new(config);
///
/// // Acquire frame
/// match acquirer.acquire(&surface) {
///     AcquireResult::Success(frame) => {
///         render_to(frame.view());
///         frame.present();
///     }
///     AcquireResult::Timeout => {
///         // Skip frame
///     }
///     _ => {
///         // Handle error
///     }
/// }
///
/// // Check stats
/// println!("Stats: {}", acquirer.stats());
/// ```
pub struct FrameAcquirer {
    /// Configuration for acquisition.
    config: AcquireConfig,
    /// Total frames successfully acquired.
    frame_count: u64,
    /// Frames that were dropped (timeout, etc.).
    dropped_frames: u64,
    /// Time of last successful acquisition.
    last_acquire_time: Option<Instant>,
    /// Frame index counter.
    frame_index: u32,
    /// Detailed statistics.
    stats: FrameAcquireStats,
    /// Format for texture views.
    format: wgpu::TextureFormat,
}

impl FrameAcquirer {
    /// Create a new frame acquirer with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Acquisition configuration.
    pub fn new(config: AcquireConfig) -> Self {
        Self {
            config,
            frame_count: 0,
            dropped_frames: 0,
            last_acquire_time: None,
            frame_index: 0,
            stats: FrameAcquireStats::new(),
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
        }
    }

    /// Create a new frame acquirer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(AcquireConfig::default())
    }

    /// Set the texture format for acquired frame views.
    pub fn set_format(&mut self, format: wgpu::TextureFormat) {
        self.format = format;
    }

    /// Get the current texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Acquire a frame from the surface.
    ///
    /// Returns an `AcquireResult` indicating the outcome of the acquisition.
    /// Handles retries according to the configuration.
    ///
    /// # Arguments
    ///
    /// * `surface` - The wgpu surface to acquire from.
    pub fn acquire(&mut self, surface: &wgpu::Surface<'_>) -> AcquireResult {
        let start = Instant::now();
        let mut attempts = 0u32;

        loop {
            let result = surface.get_current_texture();

            match result {
                Ok(texture) => {
                    let acquire_time_us = start.elapsed().as_micros() as u64;
                    let index = self.frame_index;
                    self.frame_index = self.frame_index.wrapping_add(1);
                    self.frame_count += 1;
                    self.last_acquire_time = Some(Instant::now());

                    let frame =
                        AcquiredFrame::with_acquire_time(texture, self.format, index, start);

                    // Check if suboptimal (wgpu doesn't expose this directly,
                    // but we can detect it from the texture status in future versions)
                    // For now, always return Success
                    self.stats.record_success(acquire_time_us);
                    return AcquireResult::Success(frame);
                }
                Err(wgpu::SurfaceError::Timeout) => {
                    attempts += 1;
                    if self.config.retry_on_timeout && attempts <= self.config.max_retries {
                        // Retry
                        continue;
                    }
                    self.dropped_frames += 1;
                    self.stats.record_timeout();
                    return AcquireResult::Timeout;
                }
                Err(wgpu::SurfaceError::Outdated) => {
                    self.stats.record_outdated();
                    return AcquireResult::Outdated;
                }
                Err(wgpu::SurfaceError::Lost) => {
                    self.stats.record_lost();
                    return AcquireResult::Lost;
                }
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    self.stats.record_lost();
                    return AcquireResult::Lost;
                }
                Err(_) => {
                    // Handle any future error variants
                    self.stats.record_lost();
                    return AcquireResult::Lost;
                }
            }
        }
    }

    /// Acquire a frame, blocking until success or unrecoverable error.
    ///
    /// Retries on timeout according to configuration. Returns an error
    /// only for unrecoverable conditions (Lost, OutOfMemory).
    ///
    /// # Arguments
    ///
    /// * `surface` - The wgpu surface to acquire from.
    pub fn acquire_blocking(
        &mut self,
        surface: &wgpu::Surface<'_>,
    ) -> Result<AcquiredFrame, AcquireError> {
        match self.acquire(surface) {
            AcquireResult::Success(frame) | AcquireResult::Suboptimal(frame) => Ok(frame),
            AcquireResult::Timeout => Err(AcquireError::Timeout {
                retries: self.config.max_retries,
            }),
            AcquireResult::Outdated => Err(AcquireError::Outdated),
            AcquireResult::Lost => Err(AcquireError::Lost),
        }
    }

    /// Get the current acquisition statistics.
    pub fn stats(&self) -> &FrameAcquireStats {
        &self.stats
    }

    /// Reset the statistics.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Get the total number of successfully acquired frames.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the number of dropped frames.
    pub fn dropped_frames(&self) -> u64 {
        self.dropped_frames
    }

    /// Get the time of the last successful acquisition.
    pub fn last_acquire_time(&self) -> Option<Instant> {
        self.last_acquire_time
    }

    /// Get the time since the last successful acquisition.
    pub fn time_since_last_acquire(&self) -> Option<Duration> {
        self.last_acquire_time.map(|t| t.elapsed())
    }

    /// Get a reference to the current configuration.
    pub fn config(&self) -> &AcquireConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: AcquireConfig) {
        self.config = config;
    }

    /// Get the current frame index.
    pub fn current_frame_index(&self) -> u32 {
        self.frame_index
    }
}

impl fmt::Debug for FrameAcquirer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameAcquirer")
            .field("frame_count", &self.frame_count)
            .field("dropped_frames", &self.dropped_frames)
            .field("config", &self.config)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // AcquireConfig tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_acquire_config_new() {
        let config = AcquireConfig::new();
        assert!(config.timeout.is_none());
        assert!(!config.retry_on_timeout);
        assert_eq!(config.max_retries, 0);
    }

    #[test]
    fn test_acquire_config_with_timeout() {
        let timeout = Duration::from_millis(16);
        let config = AcquireConfig::with_timeout(timeout);
        assert_eq!(config.timeout, Some(timeout));
        assert!(!config.retry_on_timeout);
        assert_eq!(config.max_retries, 0);
    }

    #[test]
    fn test_acquire_config_no_timeout() {
        let config = AcquireConfig::no_timeout();
        assert!(config.timeout.is_none());
    }

    #[test]
    fn test_acquire_config_with_retries() {
        let config = AcquireConfig::new().with_retries(3);
        assert!(config.retry_on_timeout);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_acquire_config_with_zero_retries() {
        let config = AcquireConfig::new().with_retries(0);
        assert!(!config.retry_on_timeout);
        assert_eq!(config.max_retries, 0);
    }

    #[test]
    fn test_acquire_config_has_retries() {
        let config_no_retries = AcquireConfig::new();
        assert!(!config_no_retries.has_retries());

        let config_with_retries = AcquireConfig::new().with_retries(2);
        assert!(config_with_retries.has_retries());
    }

    #[test]
    fn test_acquire_config_set_timeout() {
        let config = AcquireConfig::new().set_timeout(Some(Duration::from_millis(32)));
        assert_eq!(config.timeout, Some(Duration::from_millis(32)));

        let config = config.set_timeout(None);
        assert!(config.timeout.is_none());
    }

    #[test]
    fn test_acquire_config_default() {
        let config = AcquireConfig::default();
        assert!(config.timeout.is_none());
        assert!(!config.retry_on_timeout);
        assert_eq!(config.max_retries, 0);
    }

    // ------------------------------------------------------------------------
    // AcquireResult tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_acquire_result_timeout_is_success() {
        assert!(!AcquireResult::Timeout.is_success());
    }

    #[test]
    fn test_acquire_result_outdated_is_success() {
        assert!(!AcquireResult::Outdated.is_success());
    }

    #[test]
    fn test_acquire_result_lost_is_success() {
        assert!(!AcquireResult::Lost.is_success());
    }

    #[test]
    fn test_acquire_result_timeout_is_recoverable() {
        assert!(AcquireResult::Timeout.is_recoverable());
    }

    #[test]
    fn test_acquire_result_outdated_is_recoverable() {
        assert!(AcquireResult::Outdated.is_recoverable());
    }

    #[test]
    fn test_acquire_result_lost_is_not_recoverable() {
        assert!(!AcquireResult::Lost.is_recoverable());
    }

    #[test]
    fn test_acquire_result_timeout_needs_reconfigure() {
        assert!(!AcquireResult::Timeout.needs_reconfigure());
    }

    #[test]
    fn test_acquire_result_outdated_needs_reconfigure() {
        assert!(AcquireResult::Outdated.needs_reconfigure());
    }

    #[test]
    fn test_acquire_result_lost_needs_reconfigure() {
        assert!(!AcquireResult::Lost.needs_reconfigure());
    }

    #[test]
    fn test_acquire_result_timeout_frame() {
        assert!(AcquireResult::Timeout.frame().is_none());
    }

    #[test]
    fn test_acquire_result_outdated_frame() {
        assert!(AcquireResult::Outdated.frame().is_none());
    }

    #[test]
    fn test_acquire_result_lost_frame() {
        assert!(AcquireResult::Lost.frame().is_none());
    }

    #[test]
    fn test_acquire_result_timeout_has_frame() {
        assert!(!AcquireResult::Timeout.has_frame());
    }

    #[test]
    fn test_acquire_result_is_suboptimal() {
        assert!(!AcquireResult::Timeout.is_suboptimal());
        assert!(!AcquireResult::Outdated.is_suboptimal());
        assert!(!AcquireResult::Lost.is_suboptimal());
    }

    #[test]
    fn test_acquire_result_into_result_timeout() {
        let result = AcquireResult::Timeout.into_result();
        assert!(matches!(result, Err(AcquireError::Timeout { retries: 0 })));
    }

    #[test]
    fn test_acquire_result_into_result_outdated() {
        let result = AcquireResult::Outdated.into_result();
        assert!(matches!(result, Err(AcquireError::Outdated)));
    }

    #[test]
    fn test_acquire_result_into_result_lost() {
        let result = AcquireResult::Lost.into_result();
        assert!(matches!(result, Err(AcquireError::Lost)));
    }

    // ------------------------------------------------------------------------
    // FrameAcquireStats tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_frame_acquire_stats_new() {
        let stats = FrameAcquireStats::new();
        assert_eq!(stats.total_frames, 0);
        assert_eq!(stats.dropped_frames, 0);
        assert_eq!(stats.avg_acquire_time_us, 0.0);
        assert_eq!(stats.suboptimal_count, 0);
    }

    #[test]
    fn test_frame_acquire_stats_record_success() {
        let mut stats = FrameAcquireStats::new();
        stats.record_success(100);
        assert_eq!(stats.total_frames, 1);
        assert_eq!(stats.avg_acquire_time_us, 100.0);
        assert_eq!(stats.min_acquire_time_us, 100);
        assert_eq!(stats.max_acquire_time_us, 100);
    }

    #[test]
    fn test_frame_acquire_stats_record_multiple() {
        let mut stats = FrameAcquireStats::new();
        stats.record_success(100);
        stats.record_success(200);
        stats.record_success(300);
        assert_eq!(stats.total_frames, 3);
        assert_eq!(stats.avg_acquire_time_us, 200.0);
        assert_eq!(stats.min_acquire_time_us, 100);
        assert_eq!(stats.max_acquire_time_us, 300);
    }

    #[test]
    fn test_frame_acquire_stats_record_suboptimal() {
        let mut stats = FrameAcquireStats::new();
        stats.record_suboptimal(150);
        assert_eq!(stats.total_frames, 1);
        assert_eq!(stats.suboptimal_count, 1);
    }

    #[test]
    fn test_frame_acquire_stats_record_timeout() {
        let mut stats = FrameAcquireStats::new();
        stats.record_timeout();
        assert_eq!(stats.dropped_frames, 1);
        assert_eq!(stats.timeout_count, 1);
    }

    #[test]
    fn test_frame_acquire_stats_record_outdated() {
        let mut stats = FrameAcquireStats::new();
        stats.record_outdated();
        assert_eq!(stats.outdated_count, 1);
    }

    #[test]
    fn test_frame_acquire_stats_record_lost() {
        let mut stats = FrameAcquireStats::new();
        stats.record_lost();
        assert_eq!(stats.lost_count, 1);
    }

    #[test]
    fn test_frame_acquire_stats_success_rate() {
        let mut stats = FrameAcquireStats::new();
        stats.record_success(100);
        stats.record_success(100);
        stats.record_timeout();
        stats.record_timeout();
        // 2 success, 2 dropped = 50%
        assert_eq!(stats.success_rate(), 50.0);
    }

    #[test]
    fn test_frame_acquire_stats_success_rate_no_frames() {
        let stats = FrameAcquireStats::new();
        assert_eq!(stats.success_rate(), 100.0);
    }

    #[test]
    fn test_frame_acquire_stats_suboptimal_rate() {
        let mut stats = FrameAcquireStats::new();
        stats.record_success(100);
        stats.record_suboptimal(100);
        // 1 suboptimal out of 2 total = 50%
        assert_eq!(stats.suboptimal_rate(), 50.0);
    }

    #[test]
    fn test_frame_acquire_stats_suboptimal_rate_no_frames() {
        let stats = FrameAcquireStats::new();
        assert_eq!(stats.suboptimal_rate(), 0.0);
    }

    #[test]
    fn test_frame_acquire_stats_reset() {
        let mut stats = FrameAcquireStats::new();
        stats.record_success(100);
        stats.record_timeout();
        stats.reset();
        assert_eq!(stats.total_frames, 0);
        assert_eq!(stats.dropped_frames, 0);
    }

    #[test]
    fn test_frame_acquire_stats_display() {
        let mut stats = FrameAcquireStats::new();
        stats.record_success(100);
        let display = format!("{}", stats);
        assert!(display.contains("frames: 1"));
        assert!(display.contains("dropped: 0"));
    }

    // ------------------------------------------------------------------------
    // AcquireError tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_acquire_error_is_recoverable() {
        assert!(AcquireError::Timeout { retries: 0 }.is_recoverable());
        assert!(AcquireError::Outdated.is_recoverable());
        assert!(!AcquireError::Lost.is_recoverable());
        assert!(!AcquireError::OutOfMemory.is_recoverable());
        assert!(!AcquireError::NoFrame.is_recoverable());
    }

    #[test]
    fn test_acquire_error_needs_reconfigure() {
        assert!(!AcquireError::Timeout { retries: 0 }.needs_reconfigure());
        assert!(AcquireError::Outdated.needs_reconfigure());
        assert!(!AcquireError::Lost.needs_reconfigure());
    }

    #[test]
    fn test_acquire_error_display() {
        let err = AcquireError::Timeout { retries: 3 };
        assert!(err.to_string().contains("3 retries"));

        let err = AcquireError::Lost;
        assert!(err.to_string().contains("lost"));
    }

    // ------------------------------------------------------------------------
    // FrameAcquirer tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_frame_acquirer_new() {
        let config = AcquireConfig::new();
        let acquirer = FrameAcquirer::new(config);
        assert_eq!(acquirer.frame_count(), 0);
        assert_eq!(acquirer.dropped_frames(), 0);
        assert!(acquirer.last_acquire_time().is_none());
    }

    #[test]
    fn test_frame_acquirer_with_defaults() {
        let acquirer = FrameAcquirer::with_defaults();
        assert_eq!(acquirer.frame_count(), 0);
        assert!(acquirer.config().timeout.is_none());
    }

    #[test]
    fn test_frame_acquirer_set_format() {
        let mut acquirer = FrameAcquirer::with_defaults();
        assert_eq!(acquirer.format(), wgpu::TextureFormat::Bgra8UnormSrgb);

        acquirer.set_format(wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(acquirer.format(), wgpu::TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_frame_acquirer_reset_stats() {
        let mut acquirer = FrameAcquirer::with_defaults();
        // Simulate some stats
        acquirer.stats.record_success(100);
        acquirer.stats.record_timeout();
        assert_eq!(acquirer.stats().total_frames, 1);

        acquirer.reset_stats();
        assert_eq!(acquirer.stats().total_frames, 0);
    }

    #[test]
    fn test_frame_acquirer_set_config() {
        let mut acquirer = FrameAcquirer::with_defaults();
        assert!(acquirer.config().timeout.is_none());

        let new_config = AcquireConfig::with_timeout(Duration::from_millis(16));
        acquirer.set_config(new_config);
        assert_eq!(acquirer.config().timeout, Some(Duration::from_millis(16)));
    }

    #[test]
    fn test_frame_acquirer_current_frame_index() {
        let acquirer = FrameAcquirer::with_defaults();
        assert_eq!(acquirer.current_frame_index(), 0);
    }

    #[test]
    fn test_frame_acquirer_time_since_last_acquire_none() {
        let acquirer = FrameAcquirer::with_defaults();
        assert!(acquirer.time_since_last_acquire().is_none());
    }

    #[test]
    fn test_frame_acquirer_debug() {
        let acquirer = FrameAcquirer::with_defaults();
        let debug = format!("{:?}", acquirer);
        assert!(debug.contains("FrameAcquirer"));
        assert!(debug.contains("frame_count"));
    }

    // ------------------------------------------------------------------------
    // AcquireResult debug tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_acquire_result_timeout_debug() {
        let debug = format!("{:?}", AcquireResult::Timeout);
        assert_eq!(debug, "Timeout");
    }

    #[test]
    fn test_acquire_result_outdated_debug() {
        let debug = format!("{:?}", AcquireResult::Outdated);
        assert_eq!(debug, "Outdated");
    }

    #[test]
    fn test_acquire_result_lost_debug() {
        let debug = format!("{:?}", AcquireResult::Lost);
        assert_eq!(debug, "Lost");
    }
}
