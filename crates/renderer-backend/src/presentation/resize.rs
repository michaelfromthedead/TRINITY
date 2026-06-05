//! Window resize handling for the TRINITY presentation system.
//!
//! This module provides comprehensive resize handling with debouncing,
//! validation, and various resize strategies for optimal user experience.
//!
//! # Components
//!
//! - [`ResizeEvent`] - Resize event with dimensions, scale factor, and timing
//! - [`ResizeStrategy`] - Strategies for when to apply resize operations
//! - [`ResizeHandler`] - Debounced resize handling with constraints
//! - [`ResizeValidation`] - Validation results for resize dimensions
//! - [`AspectRatioConstraint`] - Aspect ratio enforcement options
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::presentation::resize::{
//!     ResizeEvent, ResizeStrategy, ResizeHandler,
//! };
//! use std::time::Duration;
//!
//! // Create handler with debounced strategy
//! let mut handler = ResizeHandler::new(ResizeStrategy::Debounced(Duration::from_millis(100)))
//!     .with_min_size(320, 240)
//!     .with_max_size(7680, 4320);
//!
//! // Handle resize events
//! let event = ResizeEvent::new(1920, 1080);
//! if handler.handle_resize(event) {
//!     // Apply resize immediately
//!     let (w, h) = handler.consume_pending().map(|e| (e.width, e.height)).unwrap();
//!     swapchain.resize(w, h);
//! }
//! ```

use std::fmt;
use std::time::{Duration, Instant};

// ============================================================================
// ResizeEvent
// ============================================================================

/// A window resize event with dimensions, scale factor, and timing information.
///
/// This type captures all relevant information about a resize event including
/// the new dimensions, display scale factor (for HiDPI), and the timestamp
/// when the event occurred.
///
/// # Scale Factor
///
/// The scale factor represents the ratio between physical pixels and logical
/// pixels. On HiDPI displays, this is typically 2.0 or higher. The physical
/// size is `(width, height)`, while logical size is calculated by dividing
/// by the scale factor.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::presentation::resize::ResizeEvent;
///
/// // Standard display at 1080p
/// let event = ResizeEvent::new(1920, 1080);
/// assert_eq!(event.physical_size(), (1920, 1080));
/// assert!((event.aspect_ratio() - 1.777).abs() < 0.01);
///
/// // HiDPI display at 4K with 2x scaling
/// let event = ResizeEvent::with_scale_factor(3840, 2160, 2.0);
/// assert_eq!(event.physical_size(), (3840, 2160));
/// assert!((event.logical_size().0 - 1920.0).abs() < 0.001);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ResizeEvent {
    /// Width in physical pixels.
    pub width: u32,
    /// Height in physical pixels.
    pub height: u32,
    /// Display scale factor (1.0 for standard, 2.0 for Retina/HiDPI).
    pub scale_factor: f64,
    /// Timestamp when the resize event occurred.
    pub timestamp: Instant,
}

impl ResizeEvent {
    /// Create a new resize event with the given dimensions.
    ///
    /// Uses a scale factor of 1.0 and the current time as timestamp.
    ///
    /// # Arguments
    ///
    /// * `width` - Width in physical pixels.
    /// * `height` - Height in physical pixels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let event = ResizeEvent::new(1920, 1080);
    /// assert_eq!(event.width, 1920);
    /// assert_eq!(event.height, 1080);
    /// assert_eq!(event.scale_factor, 1.0);
    /// ```
    #[inline]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            scale_factor: 1.0,
            timestamp: Instant::now(),
        }
    }

    /// Create a resize event with explicit scale factor.
    ///
    /// The scale factor represents the ratio between physical and logical
    /// pixels. For HiDPI/Retina displays, this is typically 2.0 or higher.
    ///
    /// # Arguments
    ///
    /// * `width` - Width in physical pixels.
    /// * `height` - Height in physical pixels.
    /// * `scale` - Display scale factor (typically 1.0-4.0).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 4K display with 2x scaling (appears as 1920x1080 logical)
    /// let event = ResizeEvent::with_scale_factor(3840, 2160, 2.0);
    /// assert_eq!(event.physical_size(), (3840, 2160));
    /// assert!((event.logical_size().0 - 1920.0).abs() < 0.001);
    /// ```
    #[inline]
    pub fn with_scale_factor(width: u32, height: u32, scale: f64) -> Self {
        Self {
            width,
            height,
            scale_factor: scale.max(0.001), // Prevent division by zero
            timestamp: Instant::now(),
        }
    }

    /// Create a resize event with explicit timestamp.
    ///
    /// Useful for testing or when replaying resize events.
    ///
    /// # Arguments
    ///
    /// * `width` - Width in physical pixels.
    /// * `height` - Height in physical pixels.
    /// * `scale` - Display scale factor.
    /// * `timestamp` - Event timestamp.
    #[inline]
    pub fn with_timestamp(width: u32, height: u32, scale: f64, timestamp: Instant) -> Self {
        Self {
            width,
            height,
            scale_factor: scale.max(0.001),
            timestamp,
        }
    }

    /// Get the physical size (actual pixel dimensions).
    ///
    /// This returns the raw pixel dimensions that should be used for
    /// texture allocation and GPU resources.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let event = ResizeEvent::new(1920, 1080);
    /// let (w, h) = event.physical_size();
    /// assert_eq!((w, h), (1920, 1080));
    /// ```
    #[inline]
    pub fn physical_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the logical size (scaled dimensions).
    ///
    /// This returns the dimensions in logical pixels, which represent the
    /// size as perceived by the user. On HiDPI displays, this is smaller
    /// than the physical size.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let event = ResizeEvent::with_scale_factor(3840, 2160, 2.0);
    /// let (w, h) = event.logical_size();
    /// assert!((w - 1920.0).abs() < 0.001);
    /// assert!((h - 1080.0).abs() < 0.001);
    /// ```
    #[inline]
    pub fn logical_size(&self) -> (f64, f64) {
        (
            self.width as f64 / self.scale_factor,
            self.height as f64 / self.scale_factor,
        )
    }

    /// Calculate the aspect ratio (width / height).
    ///
    /// Returns 1.0 if height is zero to avoid division by zero.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let event = ResizeEvent::new(1920, 1080);
    /// assert!((event.aspect_ratio() - 1.777).abs() < 0.01); // 16:9
    ///
    /// let event = ResizeEvent::new(1920, 1200);
    /// assert!((event.aspect_ratio() - 1.6).abs() < 0.01); // 16:10
    /// ```
    #[inline]
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            1.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    /// Check if this event represents a minimized window.
    ///
    /// A window is considered minimized if either dimension is zero or
    /// both dimensions are 1x1 (platform-dependent behavior).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let event = ResizeEvent::new(0, 0);
    /// assert!(event.is_minimized());
    ///
    /// let event = ResizeEvent::new(1, 1);
    /// assert!(event.is_minimized());
    ///
    /// let event = ResizeEvent::new(1920, 1080);
    /// assert!(!event.is_minimized());
    /// ```
    #[inline]
    pub fn is_minimized(&self) -> bool {
        self.width == 0 || self.height == 0 || (self.width == 1 && self.height == 1)
    }

    /// Check if dimensions are valid for rendering.
    ///
    /// Returns `true` if both dimensions are greater than zero and not
    /// in a minimized state.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.width > 1 && self.height > 1
    }

    /// Calculate the area in pixels.
    ///
    /// Returns the total number of pixels (width * height).
    #[inline]
    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Get the time elapsed since this event was created.
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

impl Default for ResizeEvent {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

impl fmt::Display for ResizeEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if (self.scale_factor - 1.0).abs() < 0.001 {
            write!(f, "{}x{}", self.width, self.height)
        } else {
            write!(f, "{}x{} @{:.1}x", self.width, self.height, self.scale_factor)
        }
    }
}

// ============================================================================
// ResizeStrategy
// ============================================================================

/// Strategy for handling window resize events.
///
/// Different strategies optimize for different use cases:
///
/// - **Immediate**: Best for games where smooth resizing during drag is important.
/// - **Debounced**: Best for editors where resize triggers expensive operations.
/// - **NextFrame**: Best when resize should synchronize with rendering.
/// - **Manual**: Best when application needs full control over resize timing.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::presentation::resize::ResizeStrategy;
/// use std::time::Duration;
///
/// // For smooth game resize
/// let strategy = ResizeStrategy::Immediate;
///
/// // For editor with expensive resize operations
/// let strategy = ResizeStrategy::Debounced(Duration::from_millis(150));
///
/// // Check properties
/// assert!(ResizeStrategy::Immediate.should_apply_immediately());
/// assert_eq!(
///     ResizeStrategy::Debounced(Duration::from_millis(100)).debounce_duration(),
///     Some(Duration::from_millis(100))
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeStrategy {
    /// Apply resize immediately when the event occurs.
    ///
    /// This provides the smoothest resize experience but may cause
    /// performance issues if resize operations are expensive.
    Immediate,

    /// Wait for resize events to settle before applying.
    ///
    /// The duration specifies how long to wait after the last resize
    /// event before applying. This prevents rapid resize operations
    /// during window dragging.
    Debounced(Duration),

    /// Defer resize to the next frame.
    ///
    /// The resize is queued and applied at the start of the next frame.
    /// This ensures resize is synchronized with the render loop.
    NextFrame,

    /// Only resize when explicitly triggered by the application.
    ///
    /// Resize events are queued but not applied until the application
    /// calls `force_resize()` or explicitly consumes the pending event.
    Manual,
}

impl ResizeStrategy {
    /// Check if this strategy applies resizes immediately.
    ///
    /// Returns `true` for [`Immediate`](ResizeStrategy::Immediate), `false` otherwise.
    #[inline]
    pub fn should_apply_immediately(&self) -> bool {
        matches!(self, Self::Immediate)
    }

    /// Get the debounce duration if this is a debounced strategy.
    ///
    /// Returns `Some(duration)` for [`Debounced`](ResizeStrategy::Debounced),
    /// `None` otherwise.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use renderer_backend::presentation::resize::ResizeStrategy;
    /// use std::time::Duration;
    ///
    /// let strategy = ResizeStrategy::Debounced(Duration::from_millis(100));
    /// assert_eq!(strategy.debounce_duration(), Some(Duration::from_millis(100)));
    ///
    /// let strategy = ResizeStrategy::Immediate;
    /// assert_eq!(strategy.debounce_duration(), None);
    /// ```
    #[inline]
    pub fn debounce_duration(&self) -> Option<Duration> {
        match self {
            Self::Debounced(d) => Some(*d),
            _ => None,
        }
    }

    /// Check if this strategy requires debouncing.
    #[inline]
    pub fn is_debounced(&self) -> bool {
        matches!(self, Self::Debounced(_))
    }

    /// Check if this strategy defers to next frame.
    #[inline]
    pub fn is_deferred(&self) -> bool {
        matches!(self, Self::NextFrame | Self::Manual)
    }
}

impl Default for ResizeStrategy {
    fn default() -> Self {
        Self::Immediate
    }
}

impl fmt::Display for ResizeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Immediate => write!(f, "Immediate"),
            Self::Debounced(d) => write!(f, "Debounced({}ms)", d.as_millis()),
            Self::NextFrame => write!(f, "NextFrame"),
            Self::Manual => write!(f, "Manual"),
        }
    }
}

// ============================================================================
// ResizeValidation
// ============================================================================

/// Result of validating resize dimensions against constraints.
///
/// Used by [`ResizeHandler`] to determine if a resize is valid or needs
/// to be rejected or clamped.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::presentation::resize::{ResizeHandler, ResizeValidation};
///
/// let handler = ResizeHandler::new_default()
///     .with_min_size(320, 240)
///     .with_max_size(7680, 4320);
///
/// match handler.validate(1920, 1080) {
///     ResizeValidation::Valid => println!("Size is valid"),
///     ResizeValidation::TooSmall { min_width, min_height } => {
///         println!("Too small, minimum is {}x{}", min_width, min_height);
///     }
///     ResizeValidation::TooLarge { max_width, max_height } => {
///         println!("Too large, maximum is {}x{}", max_width, max_height);
///     }
///     ResizeValidation::ZeroDimension => {
///         println!("Zero dimension not allowed");
///     }
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeValidation {
    /// Dimensions are within all constraints.
    Valid,

    /// Dimensions are below the minimum size.
    TooSmall {
        /// Minimum allowed width.
        min_width: u32,
        /// Minimum allowed height.
        min_height: u32,
    },

    /// Dimensions exceed the maximum size.
    TooLarge {
        /// Maximum allowed width.
        max_width: u32,
        /// Maximum allowed height.
        max_height: u32,
    },

    /// One or both dimensions are zero.
    ///
    /// Zero dimensions are never valid for rendering, though they may
    /// indicate a minimized window state.
    ZeroDimension,
}

impl ResizeValidation {
    /// Check if the validation passed (dimensions are valid).
    #[inline]
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }

    /// Check if the resize should be rejected entirely.
    ///
    /// Returns `true` for `ZeroDimension`, which cannot be clamped.
    #[inline]
    pub fn should_reject(&self) -> bool {
        matches!(self, Self::ZeroDimension)
    }

    /// Check if dimensions can be clamped to valid range.
    ///
    /// Returns `true` for `TooSmall` and `TooLarge`.
    #[inline]
    pub fn can_clamp(&self) -> bool {
        matches!(self, Self::TooSmall { .. } | Self::TooLarge { .. })
    }
}

impl fmt::Display for ResizeValidation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Valid => write!(f, "Valid"),
            Self::TooSmall { min_width, min_height } => {
                write!(f, "Too small (min {}x{})", min_width, min_height)
            }
            Self::TooLarge { max_width, max_height } => {
                write!(f, "Too large (max {}x{})", max_width, max_height)
            }
            Self::ZeroDimension => write!(f, "Zero dimension"),
        }
    }
}

// ============================================================================
// AspectRatioConstraint
// ============================================================================

/// Constraint on aspect ratio for resize operations.
///
/// Some applications need to maintain specific aspect ratios, either
/// exactly (video players) or within a range (games).
///
/// # Example
///
/// ```ignore
/// use renderer_backend::presentation::resize::AspectRatioConstraint;
///
/// // No aspect ratio constraint
/// let constraint = AspectRatioConstraint::None;
///
/// // Fixed 16:9 aspect ratio
/// let constraint = AspectRatioConstraint::Fixed(16.0 / 9.0);
/// assert!(constraint.is_valid(1920, 1080));
/// assert!(!constraint.is_valid(1920, 1200));
///
/// // Allow 16:9 to 16:10
/// let constraint = AspectRatioConstraint::Range(1.6, 1.78);
/// assert!(constraint.is_valid(1920, 1080));
/// assert!(constraint.is_valid(1920, 1200));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AspectRatioConstraint {
    /// No aspect ratio constraint - any ratio is allowed.
    None,

    /// Fixed aspect ratio with small tolerance.
    ///
    /// The ratio must match within 0.1% (0.001).
    Fixed(f32),

    /// Aspect ratio must fall within the given range (inclusive).
    ///
    /// First value is minimum ratio, second is maximum.
    Range(f32, f32),
}

impl AspectRatioConstraint {
    /// Default tolerance for fixed aspect ratio matching.
    const TOLERANCE: f32 = 0.001;

    /// Check if the given dimensions satisfy this constraint.
    ///
    /// # Arguments
    ///
    /// * `width` - Width in pixels.
    /// * `height` - Height in pixels.
    ///
    /// # Returns
    ///
    /// `true` if the aspect ratio satisfies the constraint, `false` otherwise.
    /// Returns `true` for zero height to avoid undefined behavior.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use renderer_backend::presentation::resize::AspectRatioConstraint;
    ///
    /// let fixed = AspectRatioConstraint::Fixed(16.0 / 9.0);
    /// assert!(fixed.is_valid(1920, 1080));
    /// assert!(fixed.is_valid(3840, 2160)); // Same ratio
    /// assert!(!fixed.is_valid(1920, 1200)); // Different ratio
    /// ```
    pub fn is_valid(&self, width: u32, height: u32) -> bool {
        if height == 0 {
            return true; // Avoid division by zero, let other validation catch this
        }

        let ratio = width as f32 / height as f32;

        match self {
            Self::None => true,
            Self::Fixed(target) => (ratio - target).abs() <= Self::TOLERANCE,
            Self::Range(min, max) => ratio >= *min && ratio <= *max,
        }
    }

    /// Calculate constrained dimensions for a given width.
    ///
    /// Given a width, calculates the height that satisfies this constraint.
    /// For `None`, returns the original height. For `Fixed`, returns the
    /// height calculated from the fixed ratio. For `Range`, clamps to the
    /// nearest valid ratio.
    ///
    /// # Arguments
    ///
    /// * `width` - Desired width in pixels.
    /// * `height` - Current/desired height in pixels.
    ///
    /// # Returns
    ///
    /// The constrained height value.
    pub fn constrain_height(&self, width: u32, height: u32) -> u32 {
        match self {
            Self::None => height,
            Self::Fixed(ratio) => {
                if *ratio <= 0.0 {
                    height
                } else {
                    (width as f32 / ratio).round() as u32
                }
            }
            Self::Range(min_ratio, max_ratio) => {
                if height == 0 {
                    return height;
                }
                let current_ratio = width as f32 / height as f32;
                if current_ratio < *min_ratio {
                    // Too tall, calculate height for min ratio
                    (width as f32 / min_ratio).round() as u32
                } else if current_ratio > *max_ratio {
                    // Too wide, calculate height for max ratio
                    (width as f32 / max_ratio).round() as u32
                } else {
                    height
                }
            }
        }
    }

    /// Calculate constrained dimensions for a given height.
    ///
    /// Given a height, calculates the width that satisfies this constraint.
    ///
    /// # Arguments
    ///
    /// * `width` - Current/desired width in pixels.
    /// * `height` - Desired height in pixels.
    ///
    /// # Returns
    ///
    /// The constrained width value.
    pub fn constrain_width(&self, width: u32, height: u32) -> u32 {
        match self {
            Self::None => width,
            Self::Fixed(ratio) => (height as f32 * ratio).round() as u32,
            Self::Range(min_ratio, max_ratio) => {
                if height == 0 {
                    return width;
                }
                let current_ratio = width as f32 / height as f32;
                if current_ratio < *min_ratio {
                    // Too tall, calculate width for min ratio
                    (height as f32 * min_ratio).round() as u32
                } else if current_ratio > *max_ratio {
                    // Too wide, calculate width for max ratio
                    (height as f32 * max_ratio).round() as u32
                } else {
                    width
                }
            }
        }
    }

    /// Check if this constraint is active.
    #[inline]
    pub fn is_constrained(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Get the target ratio for fixed constraints.
    pub fn target_ratio(&self) -> Option<f32> {
        match self {
            Self::Fixed(r) => Some(*r),
            _ => None,
        }
    }

    /// Get the ratio range for range constraints.
    pub fn ratio_range(&self) -> Option<(f32, f32)> {
        match self {
            Self::Range(min, max) => Some((*min, *max)),
            _ => None,
        }
    }
}

impl Default for AspectRatioConstraint {
    fn default() -> Self {
        Self::None
    }
}

impl fmt::Display for AspectRatioConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Fixed(ratio) => write!(f, "Fixed({:.3})", ratio),
            Self::Range(min, max) => write!(f, "Range({:.3}..{:.3})", min, max),
        }
    }
}

// ============================================================================
// ResizeHandler
// ============================================================================

/// Handler for window resize events with debouncing and constraints.
///
/// `ResizeHandler` manages resize events according to a configurable strategy,
/// applies size constraints, and provides debouncing to prevent rapid resize
/// operations during window dragging.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::presentation::resize::{
///     ResizeEvent, ResizeHandler, ResizeStrategy,
/// };
/// use std::time::Duration;
///
/// // Create handler with 100ms debounce and size constraints
/// let mut handler = ResizeHandler::new(ResizeStrategy::Debounced(Duration::from_millis(100)))
///     .with_min_size(320, 240)
///     .with_max_size(7680, 4320);
///
/// // In event loop
/// let event = ResizeEvent::new(1920, 1080);
/// if handler.handle_resize(event) {
///     // Resize should be applied now
///     if let Some(event) = handler.consume_pending() {
///         swapchain.resize(event.width, event.height);
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct ResizeHandler {
    /// The resize strategy to use.
    strategy: ResizeStrategy,
    /// Pending resize event (if any).
    pending_resize: Option<ResizeEvent>,
    /// Timestamp of the last resize operation.
    last_resize: Option<Instant>,
    /// Minimum allowed dimensions.
    min_size: (u32, u32),
    /// Maximum allowed dimensions (if constrained).
    max_size: Option<(u32, u32)>,
    /// Aspect ratio constraint.
    aspect_ratio: AspectRatioConstraint,
    /// Whether resize events are currently suppressed.
    suppressed: bool,
}

impl ResizeHandler {
    /// Default minimum size (320x200 - classic VGA).
    pub const DEFAULT_MIN_WIDTH: u32 = 320;
    pub const DEFAULT_MIN_HEIGHT: u32 = 200;

    /// Create a new resize handler with the specified strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The resize strategy to use.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use renderer_backend::presentation::resize::{ResizeHandler, ResizeStrategy};
    /// use std::time::Duration;
    ///
    /// let handler = ResizeHandler::new(ResizeStrategy::Debounced(Duration::from_millis(100)));
    /// ```
    pub fn new(strategy: ResizeStrategy) -> Self {
        Self {
            strategy,
            pending_resize: None,
            last_resize: None,
            min_size: (Self::DEFAULT_MIN_WIDTH, Self::DEFAULT_MIN_HEIGHT),
            max_size: None,
            aspect_ratio: AspectRatioConstraint::None,
            suppressed: false,
        }
    }

    /// Create a new resize handler with default settings.
    ///
    /// Uses [`ResizeStrategy::Immediate`] and default minimum size.
    pub fn new_default() -> Self {
        Self::new(ResizeStrategy::Immediate)
    }

    /// Set the minimum allowed dimensions.
    ///
    /// Resize events with dimensions below this will be clamped or rejected.
    ///
    /// # Arguments
    ///
    /// * `width` - Minimum width in pixels.
    /// * `height` - Minimum height in pixels.
    #[must_use]
    pub fn with_min_size(mut self, width: u32, height: u32) -> Self {
        self.min_size = (width.max(1), height.max(1));
        self
    }

    /// Set the maximum allowed dimensions.
    ///
    /// Resize events with dimensions above this will be clamped.
    ///
    /// # Arguments
    ///
    /// * `width` - Maximum width in pixels.
    /// * `height` - Maximum height in pixels.
    #[must_use]
    pub fn with_max_size(mut self, width: u32, height: u32) -> Self {
        self.max_size = Some((width, height));
        self
    }

    /// Set the aspect ratio constraint.
    ///
    /// # Arguments
    ///
    /// * `constraint` - The aspect ratio constraint to apply.
    #[must_use]
    pub fn with_aspect_ratio(mut self, constraint: AspectRatioConstraint) -> Self {
        self.aspect_ratio = constraint;
        self
    }

    /// Update the minimum size.
    pub fn set_min_size(&mut self, width: u32, height: u32) {
        self.min_size = (width.max(1), height.max(1));
    }

    /// Update the maximum size.
    pub fn set_max_size(&mut self, width: u32, height: u32) {
        self.max_size = Some((width, height));
    }

    /// Clear the maximum size constraint.
    pub fn clear_max_size(&mut self) {
        self.max_size = None;
    }

    /// Update the aspect ratio constraint.
    pub fn set_aspect_ratio(&mut self, constraint: AspectRatioConstraint) {
        self.aspect_ratio = constraint;
    }

    /// Update the resize strategy.
    pub fn set_strategy(&mut self, strategy: ResizeStrategy) {
        self.strategy = strategy;
    }

    /// Suppress resize events.
    ///
    /// When suppressed, `handle_resize` will return `false` and not queue events.
    pub fn suppress(&mut self) {
        self.suppressed = true;
    }

    /// Resume handling resize events.
    pub fn resume(&mut self) {
        self.suppressed = false;
    }

    /// Check if resize events are currently suppressed.
    pub fn is_suppressed(&self) -> bool {
        self.suppressed
    }

    /// Handle a resize event.
    ///
    /// Returns `true` if the resize should be applied immediately, `false` if
    /// it was queued for later or rejected.
    ///
    /// # Arguments
    ///
    /// * `event` - The resize event to handle.
    ///
    /// # Returns
    ///
    /// - `true` if the resize should be applied now (call `consume_pending`).
    /// - `false` if the resize was queued, rejected, or suppressed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let event = ResizeEvent::new(1920, 1080);
    /// if handler.handle_resize(event) {
    ///     if let Some(event) = handler.consume_pending() {
    ///         apply_resize(event.width, event.height);
    ///     }
    /// }
    /// ```
    pub fn handle_resize(&mut self, event: ResizeEvent) -> bool {
        // Check if suppressed
        if self.suppressed {
            return false;
        }

        // Validate dimensions
        let validation = self.validate(event.width, event.height);
        if validation.should_reject() {
            return false;
        }

        // Clamp dimensions if needed
        let (width, height) = self.clamp_to_limits(event.width, event.height);
        let clamped_event = if width != event.width || height != event.height {
            ResizeEvent {
                width,
                height,
                scale_factor: event.scale_factor,
                timestamp: event.timestamp,
            }
        } else {
            event
        };

        // Store as pending
        self.pending_resize = Some(clamped_event);

        // Determine if we should apply immediately based on strategy
        match &self.strategy {
            ResizeStrategy::Immediate => {
                self.last_resize = Some(Instant::now());
                true
            }
            ResizeStrategy::Debounced(duration) => {
                // Check if enough time has passed since last resize event
                if let Some(pending) = &self.pending_resize {
                    if pending.elapsed() >= *duration {
                        self.last_resize = Some(Instant::now());
                        return true;
                    }
                }
                false
            }
            ResizeStrategy::NextFrame => {
                // Will be consumed on next frame
                false
            }
            ResizeStrategy::Manual => {
                // Never apply automatically
                false
            }
        }
    }

    /// Check if a pending resize is ready to be applied.
    ///
    /// For debounced strategies, returns `true` only if the debounce
    /// duration has elapsed since the last resize event.
    ///
    /// # Returns
    ///
    /// `true` if there is a pending resize ready to apply.
    pub fn is_ready(&self) -> bool {
        if self.pending_resize.is_none() {
            return false;
        }

        match &self.strategy {
            ResizeStrategy::Immediate => true,
            ResizeStrategy::Debounced(duration) => {
                if let Some(pending) = &self.pending_resize {
                    pending.elapsed() >= *duration
                } else {
                    false
                }
            }
            ResizeStrategy::NextFrame => true,
            ResizeStrategy::Manual => false,
        }
    }

    /// Get the pending resize event (if any).
    ///
    /// Does not consume the event - use `consume_pending` to take ownership.
    pub fn pending(&self) -> Option<&ResizeEvent> {
        self.pending_resize.as_ref()
    }

    /// Consume and return the pending resize event.
    ///
    /// Returns `None` if there is no pending resize or if the resize
    /// is not ready (for debounced strategies).
    ///
    /// # Example
    ///
    /// ```ignore
    /// if handler.is_ready() {
    ///     if let Some(event) = handler.consume_pending() {
    ///         swapchain.resize(event.width, event.height);
    ///     }
    /// }
    /// ```
    pub fn consume_pending(&mut self) -> Option<ResizeEvent> {
        if self.is_ready() || matches!(self.strategy, ResizeStrategy::Manual) {
            self.last_resize = Some(Instant::now());
            self.pending_resize.take()
        } else {
            None
        }
    }

    /// Force a resize to specific dimensions.
    ///
    /// Bypasses debouncing and immediately queues a resize event.
    /// The event is marked as ready for consumption.
    ///
    /// # Arguments
    ///
    /// * `width` - Target width in pixels.
    /// * `height` - Target height in pixels.
    pub fn force_resize(&mut self, width: u32, height: u32) {
        let (w, h) = self.clamp_to_limits(width, height);
        self.pending_resize = Some(ResizeEvent::new(w, h));
        self.last_resize = Some(Instant::now());
    }

    /// Force resize with scale factor.
    ///
    /// Like `force_resize` but includes the display scale factor.
    pub fn force_resize_with_scale(&mut self, width: u32, height: u32, scale: f64) {
        let (w, h) = self.clamp_to_limits(width, height);
        self.pending_resize = Some(ResizeEvent::with_scale_factor(w, h, scale));
        self.last_resize = Some(Instant::now());
    }

    /// Clear any pending resize event.
    pub fn clear_pending(&mut self) {
        self.pending_resize = None;
    }

    /// Validate dimensions against constraints.
    ///
    /// # Arguments
    ///
    /// * `width` - Width to validate.
    /// * `height` - Height to validate.
    ///
    /// # Returns
    ///
    /// A [`ResizeValidation`] indicating whether dimensions are valid.
    ///
    /// # Example
    ///
    /// ```ignore
    /// match handler.validate(100, 100) {
    ///     ResizeValidation::Valid => apply_resize(100, 100),
    ///     ResizeValidation::TooSmall { min_width, min_height } => {
    ///         apply_resize(min_width, min_height);
    ///     }
    ///     _ => {}
    /// }
    /// ```
    pub fn validate(&self, width: u32, height: u32) -> ResizeValidation {
        // Check for zero dimensions
        if width == 0 || height == 0 {
            return ResizeValidation::ZeroDimension;
        }

        // Check minimum size
        if width < self.min_size.0 || height < self.min_size.1 {
            return ResizeValidation::TooSmall {
                min_width: self.min_size.0,
                min_height: self.min_size.1,
            };
        }

        // Check maximum size
        if let Some((max_w, max_h)) = self.max_size {
            if width > max_w || height > max_h {
                return ResizeValidation::TooLarge {
                    max_width: max_w,
                    max_height: max_h,
                };
            }
        }

        ResizeValidation::Valid
    }

    /// Clamp dimensions to the configured limits.
    ///
    /// # Arguments
    ///
    /// * `width` - Width to clamp.
    /// * `height` - Height to clamp.
    ///
    /// # Returns
    ///
    /// The clamped (width, height) tuple.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let handler = ResizeHandler::new_default()
    ///     .with_min_size(320, 240)
    ///     .with_max_size(1920, 1080);
    ///
    /// assert_eq!(handler.clamp_to_limits(100, 100), (320, 240));
    /// assert_eq!(handler.clamp_to_limits(3840, 2160), (1920, 1080));
    /// assert_eq!(handler.clamp_to_limits(1280, 720), (1280, 720));
    /// ```
    pub fn clamp_to_limits(&self, width: u32, height: u32) -> (u32, u32) {
        let mut w = width.max(self.min_size.0);
        let mut h = height.max(self.min_size.1);

        if let Some((max_w, max_h)) = self.max_size {
            w = w.min(max_w);
            h = h.min(max_h);
        }

        // Apply aspect ratio constraint
        if self.aspect_ratio.is_constrained() {
            h = self.aspect_ratio.constrain_height(w, h);
            // Re-clamp after aspect ratio adjustment
            h = h.max(self.min_size.1);
            if let Some((_, max_h)) = self.max_size {
                h = h.min(max_h);
            }
        }

        (w, h)
    }

    /// Get the current resize strategy.
    pub fn strategy(&self) -> &ResizeStrategy {
        &self.strategy
    }

    /// Get the minimum size constraint.
    pub fn min_size(&self) -> (u32, u32) {
        self.min_size
    }

    /// Get the maximum size constraint (if any).
    pub fn max_size(&self) -> Option<(u32, u32)> {
        self.max_size
    }

    /// Get the aspect ratio constraint.
    pub fn aspect_ratio(&self) -> &AspectRatioConstraint {
        &self.aspect_ratio
    }

    /// Get the timestamp of the last applied resize.
    pub fn last_resize(&self) -> Option<Instant> {
        self.last_resize
    }

    /// Check if there is a pending resize event.
    pub fn has_pending(&self) -> bool {
        self.pending_resize.is_some()
    }
}

impl Default for ResizeHandler {
    fn default() -> Self {
        Self::new_default()
    }
}

impl fmt::Display for ResizeHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ResizeHandler {{ strategy: {}, min: {}x{}, max: {:?}, pending: {} }}",
            self.strategy,
            self.min_size.0,
            self.min_size.1,
            self.max_size,
            self.pending_resize.is_some()
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // ResizeEvent tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_resize_event_new() {
        let event = ResizeEvent::new(1920, 1080);
        assert_eq!(event.width, 1920);
        assert_eq!(event.height, 1080);
        assert!((event.scale_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_resize_event_with_scale_factor() {
        let event = ResizeEvent::with_scale_factor(3840, 2160, 2.0);
        assert_eq!(event.width, 3840);
        assert_eq!(event.height, 2160);
        assert!((event.scale_factor - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_resize_event_scale_factor_zero_protection() {
        let event = ResizeEvent::with_scale_factor(1920, 1080, 0.0);
        assert!(event.scale_factor >= 0.001);
    }

    #[test]
    fn test_resize_event_physical_size() {
        let event = ResizeEvent::new(1920, 1080);
        assert_eq!(event.physical_size(), (1920, 1080));
    }

    #[test]
    fn test_resize_event_logical_size() {
        let event = ResizeEvent::with_scale_factor(3840, 2160, 2.0);
        let (w, h) = event.logical_size();
        assert!((w - 1920.0).abs() < 0.001);
        assert!((h - 1080.0).abs() < 0.001);
    }

    #[test]
    fn test_resize_event_logical_size_scale_1() {
        let event = ResizeEvent::new(1920, 1080);
        let (w, h) = event.logical_size();
        assert!((w - 1920.0).abs() < 0.001);
        assert!((h - 1080.0).abs() < 0.001);
    }

    #[test]
    fn test_resize_event_aspect_ratio_16_9() {
        let event = ResizeEvent::new(1920, 1080);
        let ratio = event.aspect_ratio();
        assert!((ratio - 1.777).abs() < 0.01);
    }

    #[test]
    fn test_resize_event_aspect_ratio_16_10() {
        let event = ResizeEvent::new(1920, 1200);
        let ratio = event.aspect_ratio();
        assert!((ratio - 1.6).abs() < 0.01);
    }

    #[test]
    fn test_resize_event_aspect_ratio_zero_height() {
        let event = ResizeEvent::new(1920, 0);
        assert_eq!(event.aspect_ratio(), 1.0);
    }

    #[test]
    fn test_resize_event_is_minimized_zero() {
        let event = ResizeEvent::new(0, 0);
        assert!(event.is_minimized());
    }

    #[test]
    fn test_resize_event_is_minimized_one() {
        let event = ResizeEvent::new(1, 1);
        assert!(event.is_minimized());
    }

    #[test]
    fn test_resize_event_is_minimized_width_zero() {
        let event = ResizeEvent::new(0, 1080);
        assert!(event.is_minimized());
    }

    #[test]
    fn test_resize_event_is_minimized_height_zero() {
        let event = ResizeEvent::new(1920, 0);
        assert!(event.is_minimized());
    }

    #[test]
    fn test_resize_event_not_minimized() {
        let event = ResizeEvent::new(1920, 1080);
        assert!(!event.is_minimized());
    }

    #[test]
    fn test_resize_event_is_valid() {
        let event = ResizeEvent::new(1920, 1080);
        assert!(event.is_valid());

        let event = ResizeEvent::new(2, 2);
        assert!(event.is_valid());

        let event = ResizeEvent::new(1, 1);
        assert!(!event.is_valid());
    }

    #[test]
    fn test_resize_event_area() {
        let event = ResizeEvent::new(1920, 1080);
        assert_eq!(event.area(), 2073600);
    }

    #[test]
    fn test_resize_event_display() {
        let event = ResizeEvent::new(1920, 1080);
        assert_eq!(format!("{}", event), "1920x1080");

        let event = ResizeEvent::with_scale_factor(3840, 2160, 2.0);
        assert_eq!(format!("{}", event), "3840x2160 @2.0x");
    }

    // ------------------------------------------------------------------------
    // ResizeStrategy tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_resize_strategy_immediate_apply() {
        let strategy = ResizeStrategy::Immediate;
        assert!(strategy.should_apply_immediately());
    }

    #[test]
    fn test_resize_strategy_debounced_not_immediate() {
        let strategy = ResizeStrategy::Debounced(Duration::from_millis(100));
        assert!(!strategy.should_apply_immediately());
    }

    #[test]
    fn test_resize_strategy_debounce_duration() {
        let strategy = ResizeStrategy::Debounced(Duration::from_millis(100));
        assert_eq!(strategy.debounce_duration(), Some(Duration::from_millis(100)));

        let strategy = ResizeStrategy::Immediate;
        assert_eq!(strategy.debounce_duration(), None);
    }

    #[test]
    fn test_resize_strategy_is_debounced() {
        assert!(ResizeStrategy::Debounced(Duration::from_millis(50)).is_debounced());
        assert!(!ResizeStrategy::Immediate.is_debounced());
        assert!(!ResizeStrategy::NextFrame.is_debounced());
        assert!(!ResizeStrategy::Manual.is_debounced());
    }

    #[test]
    fn test_resize_strategy_is_deferred() {
        assert!(!ResizeStrategy::Immediate.is_deferred());
        assert!(!ResizeStrategy::Debounced(Duration::from_millis(50)).is_deferred());
        assert!(ResizeStrategy::NextFrame.is_deferred());
        assert!(ResizeStrategy::Manual.is_deferred());
    }

    #[test]
    fn test_resize_strategy_default() {
        let strategy = ResizeStrategy::default();
        assert_eq!(strategy, ResizeStrategy::Immediate);
    }

    #[test]
    fn test_resize_strategy_display() {
        assert_eq!(format!("{}", ResizeStrategy::Immediate), "Immediate");
        assert_eq!(format!("{}", ResizeStrategy::Debounced(Duration::from_millis(100))), "Debounced(100ms)");
        assert_eq!(format!("{}", ResizeStrategy::NextFrame), "NextFrame");
        assert_eq!(format!("{}", ResizeStrategy::Manual), "Manual");
    }

    // ------------------------------------------------------------------------
    // ResizeValidation tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_resize_validation_is_valid() {
        assert!(ResizeValidation::Valid.is_valid());
        assert!(!ResizeValidation::TooSmall { min_width: 320, min_height: 240 }.is_valid());
        assert!(!ResizeValidation::TooLarge { max_width: 1920, max_height: 1080 }.is_valid());
        assert!(!ResizeValidation::ZeroDimension.is_valid());
    }

    #[test]
    fn test_resize_validation_should_reject() {
        assert!(!ResizeValidation::Valid.should_reject());
        assert!(!ResizeValidation::TooSmall { min_width: 320, min_height: 240 }.should_reject());
        assert!(!ResizeValidation::TooLarge { max_width: 1920, max_height: 1080 }.should_reject());
        assert!(ResizeValidation::ZeroDimension.should_reject());
    }

    #[test]
    fn test_resize_validation_can_clamp() {
        assert!(!ResizeValidation::Valid.can_clamp());
        assert!(ResizeValidation::TooSmall { min_width: 320, min_height: 240 }.can_clamp());
        assert!(ResizeValidation::TooLarge { max_width: 1920, max_height: 1080 }.can_clamp());
        assert!(!ResizeValidation::ZeroDimension.can_clamp());
    }

    #[test]
    fn test_resize_validation_display() {
        assert_eq!(format!("{}", ResizeValidation::Valid), "Valid");
        assert_eq!(
            format!("{}", ResizeValidation::TooSmall { min_width: 320, min_height: 240 }),
            "Too small (min 320x240)"
        );
        assert_eq!(
            format!("{}", ResizeValidation::TooLarge { max_width: 1920, max_height: 1080 }),
            "Too large (max 1920x1080)"
        );
        assert_eq!(format!("{}", ResizeValidation::ZeroDimension), "Zero dimension");
    }

    // ------------------------------------------------------------------------
    // AspectRatioConstraint tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_aspect_ratio_none_valid() {
        let constraint = AspectRatioConstraint::None;
        assert!(constraint.is_valid(1920, 1080));
        assert!(constraint.is_valid(1920, 1200));
        assert!(constraint.is_valid(800, 600));
    }

    #[test]
    fn test_aspect_ratio_fixed_valid() {
        let constraint = AspectRatioConstraint::Fixed(16.0 / 9.0);
        assert!(constraint.is_valid(1920, 1080));
        assert!(constraint.is_valid(3840, 2160));
        assert!(constraint.is_valid(1280, 720));
    }

    #[test]
    fn test_aspect_ratio_fixed_invalid() {
        let constraint = AspectRatioConstraint::Fixed(16.0 / 9.0);
        assert!(!constraint.is_valid(1920, 1200)); // 16:10
        assert!(!constraint.is_valid(800, 600));    // 4:3
    }

    #[test]
    fn test_aspect_ratio_range_valid() {
        let constraint = AspectRatioConstraint::Range(1.6, 1.78);
        assert!(constraint.is_valid(1920, 1080)); // 16:9 = 1.777
        assert!(constraint.is_valid(1920, 1200)); // 16:10 = 1.6
    }

    #[test]
    fn test_aspect_ratio_range_invalid() {
        let constraint = AspectRatioConstraint::Range(1.6, 1.78);
        assert!(!constraint.is_valid(800, 600)); // 4:3 = 1.333
        assert!(!constraint.is_valid(2560, 1080)); // 21:9 = 2.37
    }

    #[test]
    fn test_aspect_ratio_zero_height() {
        let constraint = AspectRatioConstraint::Fixed(16.0 / 9.0);
        assert!(constraint.is_valid(1920, 0)); // Avoid division by zero
    }

    #[test]
    fn test_aspect_ratio_constrain_height() {
        let constraint = AspectRatioConstraint::Fixed(16.0 / 9.0);
        let height = constraint.constrain_height(1920, 1200);
        assert!((height as f32 - 1080.0).abs() < 1.0);
    }

    #[test]
    fn test_aspect_ratio_constrain_width() {
        let constraint = AspectRatioConstraint::Fixed(16.0 / 9.0);
        let width = constraint.constrain_width(1600, 1080);
        assert!((width as f32 - 1920.0).abs() < 1.0);
    }

    #[test]
    fn test_aspect_ratio_is_constrained() {
        assert!(!AspectRatioConstraint::None.is_constrained());
        assert!(AspectRatioConstraint::Fixed(1.777).is_constrained());
        assert!(AspectRatioConstraint::Range(1.6, 1.78).is_constrained());
    }

    #[test]
    fn test_aspect_ratio_target_ratio() {
        assert_eq!(AspectRatioConstraint::None.target_ratio(), None);
        assert!((AspectRatioConstraint::Fixed(1.777).target_ratio().unwrap() - 1.777).abs() < 0.001);
        assert_eq!(AspectRatioConstraint::Range(1.6, 1.78).target_ratio(), None);
    }

    #[test]
    fn test_aspect_ratio_ratio_range() {
        assert_eq!(AspectRatioConstraint::None.ratio_range(), None);
        assert_eq!(AspectRatioConstraint::Fixed(1.777).ratio_range(), None);
        assert_eq!(AspectRatioConstraint::Range(1.6, 1.78).ratio_range(), Some((1.6, 1.78)));
    }

    #[test]
    fn test_aspect_ratio_display() {
        assert_eq!(format!("{}", AspectRatioConstraint::None), "None");
        assert!(format!("{}", AspectRatioConstraint::Fixed(1.777)).contains("Fixed(1.777"));
        assert!(format!("{}", AspectRatioConstraint::Range(1.6, 1.78)).contains("Range(1.600"));
    }

    // ------------------------------------------------------------------------
    // ResizeHandler tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_resize_handler_new() {
        let handler = ResizeHandler::new(ResizeStrategy::Immediate);
        assert_eq!(*handler.strategy(), ResizeStrategy::Immediate);
        assert_eq!(handler.min_size(), (320, 200));
        assert_eq!(handler.max_size(), None);
        assert!(!handler.has_pending());
    }

    #[test]
    fn test_resize_handler_new_default() {
        let handler = ResizeHandler::new_default();
        assert_eq!(*handler.strategy(), ResizeStrategy::Immediate);
    }

    #[test]
    fn test_resize_handler_with_min_size() {
        let handler = ResizeHandler::new_default().with_min_size(640, 480);
        assert_eq!(handler.min_size(), (640, 480));
    }

    #[test]
    fn test_resize_handler_with_max_size() {
        let handler = ResizeHandler::new_default().with_max_size(1920, 1080);
        assert_eq!(handler.max_size(), Some((1920, 1080)));
    }

    #[test]
    fn test_resize_handler_handle_resize_immediate() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Immediate);
        let event = ResizeEvent::new(1920, 1080);
        assert!(handler.handle_resize(event));
        assert!(handler.has_pending());
    }

    #[test]
    fn test_resize_handler_handle_resize_manual() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Manual);
        let event = ResizeEvent::new(1920, 1080);
        assert!(!handler.handle_resize(event));
        assert!(handler.has_pending());
    }

    #[test]
    fn test_resize_handler_handle_resize_next_frame() {
        let mut handler = ResizeHandler::new(ResizeStrategy::NextFrame);
        let event = ResizeEvent::new(1920, 1080);
        assert!(!handler.handle_resize(event));
        assert!(handler.has_pending());
        assert!(handler.is_ready());
    }

    #[test]
    fn test_resize_handler_consume_pending() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Immediate);
        handler.force_resize(1920, 1080);
        assert!(handler.has_pending());
        let event = handler.consume_pending();
        assert!(event.is_some());
        assert!(!handler.has_pending());
    }

    #[test]
    fn test_resize_handler_pending() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Immediate);
        assert!(handler.pending().is_none());
        handler.force_resize(1920, 1080);
        assert!(handler.pending().is_some());
        assert_eq!(handler.pending().unwrap().width, 1920);
    }

    #[test]
    fn test_resize_handler_force_resize() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Manual);
        handler.force_resize(1920, 1080);
        assert!(handler.has_pending());
        let event = handler.consume_pending();
        assert!(event.is_some());
        assert_eq!(event.unwrap().width, 1920);
    }

    #[test]
    fn test_resize_handler_force_resize_clamped() {
        let mut handler = ResizeHandler::new_default()
            .with_min_size(640, 480)
            .with_max_size(1920, 1080);

        handler.force_resize(100, 100);
        let event = handler.consume_pending().unwrap();
        assert_eq!(event.width, 640);
        assert_eq!(event.height, 480);
    }

    #[test]
    fn test_resize_handler_validate_valid() {
        let handler = ResizeHandler::new_default()
            .with_min_size(320, 240)
            .with_max_size(1920, 1080);
        assert!(handler.validate(1280, 720).is_valid());
    }

    #[test]
    fn test_resize_handler_validate_too_small() {
        let handler = ResizeHandler::new_default().with_min_size(320, 240);
        let result = handler.validate(100, 100);
        assert!(matches!(result, ResizeValidation::TooSmall { .. }));
    }

    #[test]
    fn test_resize_handler_validate_too_large() {
        let handler = ResizeHandler::new_default().with_max_size(1920, 1080);
        let result = handler.validate(3840, 2160);
        assert!(matches!(result, ResizeValidation::TooLarge { .. }));
    }

    #[test]
    fn test_resize_handler_validate_zero() {
        let handler = ResizeHandler::new_default();
        assert_eq!(handler.validate(0, 1080), ResizeValidation::ZeroDimension);
        assert_eq!(handler.validate(1920, 0), ResizeValidation::ZeroDimension);
    }

    #[test]
    fn test_resize_handler_clamp_to_limits() {
        let handler = ResizeHandler::new_default()
            .with_min_size(320, 240)
            .with_max_size(1920, 1080);

        assert_eq!(handler.clamp_to_limits(100, 100), (320, 240));
        assert_eq!(handler.clamp_to_limits(3840, 2160), (1920, 1080));
        assert_eq!(handler.clamp_to_limits(1280, 720), (1280, 720));
    }

    #[test]
    fn test_resize_handler_suppress() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Immediate);
        handler.suppress();
        assert!(handler.is_suppressed());

        let event = ResizeEvent::new(1920, 1080);
        assert!(!handler.handle_resize(event));
        assert!(!handler.has_pending());
    }

    #[test]
    fn test_resize_handler_resume() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Immediate);
        handler.suppress();
        handler.resume();
        assert!(!handler.is_suppressed());

        let event = ResizeEvent::new(1920, 1080);
        assert!(handler.handle_resize(event));
    }

    #[test]
    fn test_resize_handler_clear_pending() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Immediate);
        handler.force_resize(1920, 1080);
        assert!(handler.has_pending());
        handler.clear_pending();
        assert!(!handler.has_pending());
    }

    #[test]
    fn test_resize_handler_set_strategy() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Immediate);
        handler.set_strategy(ResizeStrategy::Manual);
        assert_eq!(*handler.strategy(), ResizeStrategy::Manual);
    }

    #[test]
    fn test_resize_handler_set_min_size() {
        let mut handler = ResizeHandler::new_default();
        handler.set_min_size(640, 480);
        assert_eq!(handler.min_size(), (640, 480));
    }

    #[test]
    fn test_resize_handler_set_max_size() {
        let mut handler = ResizeHandler::new_default();
        handler.set_max_size(1920, 1080);
        assert_eq!(handler.max_size(), Some((1920, 1080)));
    }

    #[test]
    fn test_resize_handler_clear_max_size() {
        let mut handler = ResizeHandler::new_default().with_max_size(1920, 1080);
        handler.clear_max_size();
        assert_eq!(handler.max_size(), None);
    }

    #[test]
    fn test_resize_handler_with_aspect_ratio() {
        let handler = ResizeHandler::new_default()
            .with_aspect_ratio(AspectRatioConstraint::Fixed(16.0 / 9.0));
        assert!(handler.aspect_ratio().is_constrained());
    }

    #[test]
    fn test_resize_handler_is_ready_immediate() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Immediate);
        assert!(!handler.is_ready());
        handler.force_resize(1920, 1080);
        assert!(handler.is_ready());
    }

    #[test]
    fn test_resize_handler_is_ready_manual() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Manual);
        handler.force_resize(1920, 1080);
        assert!(!handler.is_ready()); // Manual never reports ready
    }

    #[test]
    fn test_resize_handler_reject_zero_dimensions() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Immediate);
        let event = ResizeEvent::new(0, 1080);
        assert!(!handler.handle_resize(event));
        assert!(!handler.has_pending());
    }

    #[test]
    fn test_resize_handler_last_resize() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Immediate);
        assert!(handler.last_resize().is_none());
        handler.force_resize(1920, 1080);
        assert!(handler.last_resize().is_some());
    }

    #[test]
    fn test_resize_handler_display() {
        let handler = ResizeHandler::new(ResizeStrategy::Immediate)
            .with_min_size(320, 240)
            .with_max_size(1920, 1080);
        let display = format!("{}", handler);
        assert!(display.contains("Immediate"));
        assert!(display.contains("320x240"));
    }

    #[test]
    fn test_resize_handler_default() {
        let handler = ResizeHandler::default();
        assert_eq!(*handler.strategy(), ResizeStrategy::Immediate);
    }

    #[test]
    fn test_resize_handler_force_resize_with_scale() {
        let mut handler = ResizeHandler::new(ResizeStrategy::Manual);
        handler.force_resize_with_scale(3840, 2160, 2.0);
        let event = handler.consume_pending().unwrap();
        assert_eq!(event.width, 3840);
        assert!((event.scale_factor - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_resize_handler_clamp_with_aspect_ratio() {
        let handler = ResizeHandler::new_default()
            .with_min_size(320, 180)
            .with_max_size(1920, 1080)
            .with_aspect_ratio(AspectRatioConstraint::Fixed(16.0 / 9.0));

        // Width stays, height adjusted to match 16:9
        let (w, h) = handler.clamp_to_limits(1920, 1200);
        assert_eq!(w, 1920);
        assert!((h as f32 - 1080.0).abs() < 1.0);
    }
}
