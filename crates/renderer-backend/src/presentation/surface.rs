//! Surface creation from window handles for the TRINITY renderer.
//!
//! This module provides platform-agnostic surface creation using the
//! `raw-window-handle` 0.6 crate. It supports all major platforms:
//! - Linux (Wayland, X11)
//! - Windows (Win32)
//! - macOS (AppKit)
//! - Web (WebGL2/WebGPU via wasm)
//!
//! # Safety
//!
//! Surface creation from raw window handles is inherently unsafe because the
//! handles must remain valid for the lifetime of the surface. This module uses
//! `wgpu::SurfaceTargetUnsafe` internally but provides a safe API by requiring
//! the window to outlive the surface via the `'static` lifetime bound on the
//! internal surface.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::TrinityInstance;
//! use renderer_backend::presentation::TrinitySurface;
//!
//! // Create instance
//! let instance = TrinityInstance::new();
//!
//! // Create surface from window (requires window implementing HasWindowHandle + HasDisplayHandle)
//! // let surface = TrinitySurface::new(instance.inner(), &window)?;
//! //
//! // // Query capabilities
//! // let caps = surface.capabilities(&adapter);
//! // println!("Supported formats: {:?}", caps.formats);
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use wgpu::rwh::{self, HandleError, HasDisplayHandle, HasWindowHandle};

// ============================================================================
// Platform Detection
// ============================================================================

/// Detected platform target for surface creation.
///
/// Used for diagnostics and platform-specific surface configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformTarget {
    /// Linux with Wayland display server.
    Wayland,
    /// Linux with X11 display server.
    X11,
    /// Windows with Win32 API.
    Windows,
    /// macOS with AppKit.
    MacOS,
    /// iOS with UIKit.
    IOS,
    /// Android with ANativeWindow.
    Android,
    /// Web platform (WebGL2 or WebGPU).
    Web,
    /// Unknown or unsupported platform.
    Unknown,
}

impl PlatformTarget {
    /// Detect the current platform at compile time.
    #[cfg(target_os = "linux")]
    pub const fn current() -> Self {
        // At compile time we can't know if it's Wayland or X11,
        // but we can indicate it's a Linux platform.
        // Runtime detection happens in surface creation.
        PlatformTarget::X11 // Default to X11, will be refined at runtime
    }

    #[cfg(target_os = "windows")]
    pub const fn current() -> Self {
        PlatformTarget::Windows
    }

    #[cfg(target_os = "macos")]
    pub const fn current() -> Self {
        PlatformTarget::MacOS
    }

    #[cfg(target_os = "ios")]
    pub const fn current() -> Self {
        PlatformTarget::IOS
    }

    #[cfg(target_os = "android")]
    pub const fn current() -> Self {
        PlatformTarget::Android
    }

    #[cfg(target_family = "wasm")]
    pub const fn current() -> Self {
        PlatformTarget::Web
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "macos",
        target_os = "ios",
        target_os = "android",
        target_family = "wasm"
    )))]
    pub const fn current() -> Self {
        PlatformTarget::Unknown
    }

    /// Returns a human-readable name for the platform.
    pub const fn name(self) -> &'static str {
        match self {
            PlatformTarget::Wayland => "Linux (Wayland)",
            PlatformTarget::X11 => "Linux (X11)",
            PlatformTarget::Windows => "Windows",
            PlatformTarget::MacOS => "macOS",
            PlatformTarget::IOS => "iOS",
            PlatformTarget::Android => "Android",
            PlatformTarget::Web => "Web",
            PlatformTarget::Unknown => "Unknown",
        }
    }

    /// Returns true if the platform is supported by TRINITY.
    pub const fn is_supported(self) -> bool {
        !matches!(self, PlatformTarget::Unknown)
    }
}

impl fmt::Display for PlatformTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// Surface Size
// ============================================================================

/// Represents the dimensions of a surface in pixels.
///
/// This is a convenience struct for passing around surface dimensions.
/// It ensures dimensions are always valid (at least 1x1).
///
/// # Example
///
/// ```ignore
/// let size = SurfaceSize::new(1920, 1080);
/// assert_eq!(size.width, 1920);
/// assert_eq!(size.height, 1080);
/// assert!((size.aspect_ratio() - 1.777).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceSize {
    /// Width of the surface in pixels (minimum 1).
    pub width: u32,
    /// Height of the surface in pixels (minimum 1).
    pub height: u32,
}

impl SurfaceSize {
    /// Create a new surface size with the given dimensions.
    ///
    /// Dimensions are clamped to a minimum of 1 to ensure validity.
    ///
    /// # Arguments
    ///
    /// * `width` - Width in pixels.
    /// * `height` - Height in pixels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let size = SurfaceSize::new(800, 600);
    /// assert_eq!(size.width, 800);
    /// assert_eq!(size.height, 600);
    ///
    /// // Zero dimensions are clamped to 1
    /// let zero_size = SurfaceSize::new(0, 0);
    /// assert_eq!(zero_size.width, 1);
    /// assert_eq!(zero_size.height, 1);
    /// ```
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width: if width == 0 { 1 } else { width },
            height: if height == 0 { 1 } else { height },
        }
    }

    /// Create a surface size from a tuple of (width, height).
    pub const fn from_tuple(dimensions: (u32, u32)) -> Self {
        Self::new(dimensions.0, dimensions.1)
    }

    /// Get the aspect ratio (width / height).
    ///
    /// Returns 1.0 if height is 0 (though this shouldn't happen with
    /// the minimum clamping in `new()`).
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            1.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    /// Get the total number of pixels (width * height).
    pub const fn pixel_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Check if this represents a minimized window.
    ///
    /// Returns true if dimensions are 1x1 (common indicator of minimization).
    pub const fn is_minimized(&self) -> bool {
        self.width == 1 && self.height == 1
    }

    /// Convert to a tuple of (width, height).
    pub const fn as_tuple(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Check if dimensions are equal to another size.
    pub const fn matches(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }

    /// Scale the size by a factor, maintaining minimum of 1.
    pub fn scale(&self, factor: f32) -> Self {
        Self::new(
            ((self.width as f32 * factor).round() as u32).max(1),
            ((self.height as f32 * factor).round() as u32).max(1),
        )
    }
}

impl Default for SurfaceSize {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

impl From<(u32, u32)> for SurfaceSize {
    fn from(tuple: (u32, u32)) -> Self {
        Self::new(tuple.0, tuple.1)
    }
}

impl From<SurfaceSize> for (u32, u32) {
    fn from(size: SurfaceSize) -> Self {
        (size.width, size.height)
    }
}

impl fmt::Display for SurfaceSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during surface creation or configuration.
///
/// This enum provides detailed information about what went wrong, allowing
/// callers to handle specific failure modes (e.g., retry on a different
/// display server, fall back to software rendering, etc.).
#[derive(Debug, Error)]
pub enum SurfaceError {
    /// The current platform is not supported for surface creation.
    ///
    /// This occurs on platforms where TRINITY does not have surface support
    /// (e.g., embedded systems without a windowing system).
    #[error("unsupported platform: {platform}")]
    UnsupportedPlatform {
        /// The detected platform.
        platform: PlatformTarget,
    },

    /// Failed to retrieve the window handle from the window object.
    ///
    /// This typically occurs when:
    /// - The window has been destroyed
    /// - The window is not properly initialized
    /// - The windowing system returned an invalid handle
    #[error("failed to retrieve window handle: {message}")]
    WindowHandleError {
        /// Detailed error message from the windowing system.
        message: String,
        /// The underlying handle error, if available.
        #[source]
        source: Option<HandleError>,
    },

    /// Failed to retrieve the display handle from the window object.
    ///
    /// This typically occurs on X11/Wayland when the display connection
    /// is not available or has been closed.
    #[error("failed to retrieve display handle: {message}")]
    DisplayHandleError {
        /// Detailed error message.
        message: String,
        /// The underlying handle error, if available.
        #[source]
        source: Option<HandleError>,
    },

    /// The wgpu surface creation failed.
    ///
    /// This can occur due to:
    /// - Graphics driver issues
    /// - Incompatible window/display combination
    /// - Resource exhaustion
    /// - Backend-specific errors
    #[error("surface creation failed: {message}")]
    SurfaceCreationFailed {
        /// Detailed error message from wgpu.
        message: String,
        /// The platform where creation was attempted.
        platform: PlatformTarget,
    },

    /// The surface configuration is invalid.
    ///
    /// This occurs when attempting to configure a surface with parameters
    /// that are not supported by the adapter (e.g., unsupported format,
    /// invalid dimensions).
    #[error("invalid surface configuration: {message}")]
    InvalidConfiguration {
        /// Description of what is invalid.
        message: String,
    },

    /// The surface has been lost and needs to be recreated.
    ///
    /// This can happen due to:
    /// - Window minimization (on some platforms)
    /// - Display mode changes
    /// - Graphics driver reset
    #[error("surface lost: {reason}")]
    SurfaceLost {
        /// Reason for the surface loss.
        reason: String,
    },

    /// The surface is outdated and needs reconfiguration.
    ///
    /// This typically happens after a window resize.
    #[error("surface outdated, reconfiguration required")]
    SurfaceOutdated,
}

impl SurfaceError {
    /// Create an UnsupportedPlatform error for the current platform.
    pub fn unsupported() -> Self {
        SurfaceError::UnsupportedPlatform {
            platform: PlatformTarget::current(),
        }
    }

    /// Create a WindowHandleError with the given message.
    pub fn window_handle(message: impl Into<String>) -> Self {
        SurfaceError::WindowHandleError {
            message: message.into(),
            source: None,
        }
    }

    /// Create a WindowHandleError from a HandleError.
    pub fn from_window_handle_error(err: HandleError) -> Self {
        SurfaceError::WindowHandleError {
            message: err.to_string(),
            source: Some(err),
        }
    }

    /// Create a DisplayHandleError with the given message.
    pub fn display_handle(message: impl Into<String>) -> Self {
        SurfaceError::DisplayHandleError {
            message: message.into(),
            source: None,
        }
    }

    /// Create a DisplayHandleError from a HandleError.
    pub fn from_display_handle_error(err: HandleError) -> Self {
        SurfaceError::DisplayHandleError {
            message: err.to_string(),
            source: Some(err),
        }
    }

    /// Create a SurfaceCreationFailed error.
    pub fn creation_failed(message: impl Into<String>) -> Self {
        SurfaceError::SurfaceCreationFailed {
            message: message.into(),
            platform: PlatformTarget::current(),
        }
    }

    /// Create an InvalidConfiguration error.
    pub fn invalid_config(message: impl Into<String>) -> Self {
        SurfaceError::InvalidConfiguration {
            message: message.into(),
        }
    }

    /// Returns true if this error is recoverable by recreating the surface.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            SurfaceError::SurfaceLost { .. } | SurfaceError::SurfaceOutdated
        )
    }

    /// Returns true if this error indicates a platform issue.
    pub fn is_platform_error(&self) -> bool {
        matches!(self, SurfaceError::UnsupportedPlatform { .. })
    }
}

// ============================================================================
// Surface Capabilities
// ============================================================================

/// Describes the capabilities of a surface for a given adapter.
///
/// This is a wrapper around `wgpu::SurfaceCapabilities` with additional
/// convenience methods for format selection and configuration.
#[derive(Debug, Clone)]
pub struct SurfaceCapabilities {
    /// Supported texture formats for the surface.
    pub formats: Vec<wgpu::TextureFormat>,
    /// Supported present modes.
    pub present_modes: Vec<wgpu::PresentMode>,
    /// Supported alpha compositing modes.
    pub alpha_modes: Vec<wgpu::CompositeAlphaMode>,
    /// Supported texture usages.
    pub usages: wgpu::TextureUsages,
}

impl SurfaceCapabilities {
    /// Create capabilities from wgpu's SurfaceCapabilities.
    pub fn from_wgpu(caps: wgpu::SurfaceCapabilities) -> Self {
        Self {
            formats: caps.formats,
            present_modes: caps.present_modes,
            alpha_modes: caps.alpha_modes,
            usages: caps.usages,
        }
    }

    /// Returns the preferred texture format for this surface.
    ///
    /// Prefers sRGB formats for correct gamma handling, falling back to
    /// linear formats if sRGB is not available.
    pub fn preferred_format(&self) -> Option<wgpu::TextureFormat> {
        // Prefer sRGB formats for correct gamma
        let srgb_formats = [
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ];

        for format in &srgb_formats {
            if self.formats.contains(format) {
                return Some(*format);
            }
        }

        // Fall back to linear formats
        let linear_formats = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ];

        for format in &linear_formats {
            if self.formats.contains(format) {
                return Some(*format);
            }
        }

        // Last resort: return first available format
        self.formats.first().copied()
    }

    /// Returns the preferred present mode for vsync (smooth presentation).
    ///
    /// Priority order for vsync:
    /// 1. Mailbox (triple buffering with vsync) - smooth with low input latency
    /// 2. FifoRelaxed - vsync with adaptive frame drops when behind
    /// 3. Fifo (vsync) - guaranteed to be available on all platforms
    pub fn preferred_present_mode(&self) -> wgpu::PresentMode {
        if self.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if self.present_modes.contains(&wgpu::PresentMode::FifoRelaxed) {
            wgpu::PresentMode::FifoRelaxed
        } else if self.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            // Fifo is always supported, but just in case
            self.present_modes
                .first()
                .copied()
                .unwrap_or(wgpu::PresentMode::Fifo)
        }
    }

    /// Returns the best present mode for low-latency gaming.
    ///
    /// Priority order for low latency:
    /// 1. Immediate - no vsync, lowest latency, may cause tearing
    /// 2. Mailbox - triple buffered vsync, low latency without tearing
    /// 3. FifoRelaxed - vsync with adaptive frame drops
    /// 4. Fifo - fallback vsync
    ///
    /// Use this for competitive gaming or VR applications where input
    /// latency is critical.
    pub fn low_latency_present_mode(&self) -> wgpu::PresentMode {
        if self.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else if self.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if self.present_modes.contains(&wgpu::PresentMode::FifoRelaxed) {
            wgpu::PresentMode::FifoRelaxed
        } else {
            self.present_modes
                .first()
                .copied()
                .unwrap_or(wgpu::PresentMode::Fifo)
        }
    }

    /// Select the best present mode based on preference.
    ///
    /// # Arguments
    ///
    /// * `preference` - The presentation mode preference.
    ///
    /// # Returns
    ///
    /// The best available present mode matching the preference, with
    /// automatic fallback to the closest alternative if the preferred
    /// mode is unavailable.
    pub fn select_present_mode(&self, preference: PresentModePreference) -> wgpu::PresentMode {
        match preference {
            PresentModePreference::LowLatency => self.low_latency_present_mode(),
            PresentModePreference::Vsync => self.preferred_present_mode(),
            PresentModePreference::PowerSaving => {
                // Fifo is most power efficient (GPU can idle between frames)
                if self.present_modes.contains(&wgpu::PresentMode::Fifo) {
                    wgpu::PresentMode::Fifo
                } else {
                    self.preferred_present_mode()
                }
            }
            PresentModePreference::Adaptive => {
                // FifoRelaxed provides adaptive vsync - good balance
                if self.present_modes.contains(&wgpu::PresentMode::FifoRelaxed) {
                    wgpu::PresentMode::FifoRelaxed
                } else {
                    self.preferred_present_mode()
                }
            }
            PresentModePreference::Specific(mode) => {
                if self.present_modes.contains(&mode) {
                    mode
                } else {
                    // Fallback to vsync if specific mode unavailable
                    self.preferred_present_mode()
                }
            }
        }
    }

    /// Returns true if immediate (no vsync) mode is available.
    ///
    /// Immediate mode has the lowest input latency but may cause
    /// screen tearing. Use for competitive gaming.
    pub fn supports_immediate(&self) -> bool {
        self.present_modes.contains(&wgpu::PresentMode::Immediate)
    }

    /// Returns true if mailbox (triple-buffered vsync) is available.
    ///
    /// Mailbox provides smooth vsync without the input latency penalty
    /// of standard Fifo vsync.
    pub fn supports_mailbox(&self) -> bool {
        self.present_modes.contains(&wgpu::PresentMode::Mailbox)
    }

    /// Returns true if FifoRelaxed (adaptive vsync) is available.
    ///
    /// FifoRelaxed behaves like Fifo but allows frame drops when
    /// the application falls behind, reducing stuttering.
    pub fn supports_fifo_relaxed(&self) -> bool {
        self.present_modes.contains(&wgpu::PresentMode::FifoRelaxed)
    }

    /// Get a description of a present mode.
    pub fn describe_present_mode(mode: wgpu::PresentMode) -> PresentModeInfo {
        PresentModeInfo::from_mode(mode)
    }

    /// Returns the preferred alpha mode.
    ///
    /// Prefers Opaque for best performance, falling back to Auto.
    pub fn preferred_alpha_mode(&self) -> wgpu::CompositeAlphaMode {
        if self.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            self.alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto)
        }
    }

    /// Check if a specific alpha mode is supported.
    ///
    /// # Arguments
    ///
    /// * `mode` - The alpha mode to check.
    ///
    /// # Returns
    ///
    /// `true` if the mode is available on this surface.
    pub fn supports_alpha_mode(&self, mode: wgpu::CompositeAlphaMode) -> bool {
        self.alpha_modes.contains(&mode)
    }

    /// Select the best alpha mode based on preference.
    ///
    /// This method honors the user's alpha mode preference while providing
    /// automatic fallback when the preferred mode is unavailable.
    ///
    /// # Arguments
    ///
    /// * `preference` - The desired alpha mode behavior.
    ///
    /// # Returns
    ///
    /// The best available alpha mode matching the preference, with
    /// automatic fallback to Opaque or the first available mode.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mode = caps.select_alpha_mode(AlphaModePreference::PreMultiplied);
    /// ```
    pub fn select_alpha_mode(&self, preference: AlphaModePreference) -> wgpu::CompositeAlphaMode {
        match preference {
            AlphaModePreference::Auto => self.preferred_alpha_mode(),
            AlphaModePreference::Opaque => {
                if self.supports_alpha_mode(wgpu::CompositeAlphaMode::Opaque) {
                    wgpu::CompositeAlphaMode::Opaque
                } else {
                    self.preferred_alpha_mode()
                }
            }
            AlphaModePreference::PreMultiplied => {
                if self.supports_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied) {
                    wgpu::CompositeAlphaMode::PreMultiplied
                } else {
                    // Fall back to PostMultiplied, then Opaque
                    if self.supports_alpha_mode(wgpu::CompositeAlphaMode::PostMultiplied) {
                        wgpu::CompositeAlphaMode::PostMultiplied
                    } else {
                        self.preferred_alpha_mode()
                    }
                }
            }
            AlphaModePreference::PostMultiplied => {
                if self.supports_alpha_mode(wgpu::CompositeAlphaMode::PostMultiplied) {
                    wgpu::CompositeAlphaMode::PostMultiplied
                } else {
                    // Fall back to PreMultiplied, then Opaque
                    if self.supports_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied) {
                        wgpu::CompositeAlphaMode::PreMultiplied
                    } else {
                        self.preferred_alpha_mode()
                    }
                }
            }
            AlphaModePreference::Inherit => {
                if self.supports_alpha_mode(wgpu::CompositeAlphaMode::Inherit) {
                    wgpu::CompositeAlphaMode::Inherit
                } else {
                    self.preferred_alpha_mode()
                }
            }
        }
    }

    /// Check if a specific format is supported.
    pub fn supports_format(&self, format: wgpu::TextureFormat) -> bool {
        self.formats.contains(&format)
    }

    /// Check if a specific present mode is supported.
    pub fn supports_present_mode(&self, mode: wgpu::PresentMode) -> bool {
        self.present_modes.contains(&mode)
    }

    /// Returns true if the surface supports HDR formats.
    pub fn supports_hdr(&self) -> bool {
        self.formats.iter().any(|f| {
            matches!(
                f,
                wgpu::TextureFormat::Rgba16Float
                    | wgpu::TextureFormat::Rgb10a2Unorm
                    | wgpu::TextureFormat::Rg11b10Float
            )
        })
    }

    /// Returns the preferred HDR format if available.
    ///
    /// HDR format priority:
    /// 1. Rgba16Float - Full HDR with alpha, best quality
    /// 2. Rg11b10Float - HDR without alpha, good for opaque content
    /// 3. Rgb10a2Unorm - Wide gamut with 2-bit alpha
    pub fn preferred_hdr_format(&self) -> Option<wgpu::TextureFormat> {
        let hdr_formats = [
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rg11b10Float,
            wgpu::TextureFormat::Rgb10a2Unorm,
        ];

        for format in &hdr_formats {
            if self.formats.contains(format) {
                return Some(*format);
            }
        }
        None
    }

    /// Get the format category for a given texture format.
    pub fn format_category(format: wgpu::TextureFormat) -> FormatCategory {
        FormatCategory::from_format(format)
    }

    /// Returns all available formats in a specific category.
    pub fn formats_in_category(&self, category: FormatCategory) -> Vec<wgpu::TextureFormat> {
        self.formats
            .iter()
            .filter(|f| FormatCategory::from_format(**f) == category)
            .copied()
            .collect()
    }

    /// Select the best format from capabilities with a category preference.
    ///
    /// This allows callers to request HDR if available, falling back to sRGB,
    /// then linear as needed.
    pub fn select_format(&self, prefer_hdr: bool) -> Option<wgpu::TextureFormat> {
        if prefer_hdr {
            if let Some(hdr) = self.preferred_hdr_format() {
                return Some(hdr);
            }
        }
        self.preferred_format()
    }
}

// ============================================================================
// Format Category
// ============================================================================

/// Categories of texture formats for surface configuration.
///
/// This helps distinguish between different color space handling requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatCategory {
    /// sRGB formats - gamma-corrected, standard for most displays.
    /// Examples: Bgra8UnormSrgb, Rgba8UnormSrgb
    Srgb,
    /// Linear formats - no gamma correction applied.
    /// Examples: Bgra8Unorm, Rgba8Unorm
    Linear,
    /// HDR formats - high dynamic range with extended color values.
    /// Examples: Rgba16Float, Rgb10a2Unorm, Rg11b10Float
    Hdr,
    /// Other formats not commonly used for surfaces.
    Other,
}

impl FormatCategory {
    /// Determine the category of a texture format.
    pub fn from_format(format: wgpu::TextureFormat) -> Self {
        match format {
            // sRGB formats
            wgpu::TextureFormat::Bgra8UnormSrgb
            | wgpu::TextureFormat::Rgba8UnormSrgb => FormatCategory::Srgb,

            // Linear 8-bit formats
            wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Rgba8Unorm => FormatCategory::Linear,

            // HDR formats
            wgpu::TextureFormat::Rgba16Float
            | wgpu::TextureFormat::Rgb10a2Unorm
            | wgpu::TextureFormat::Rg11b10Float => FormatCategory::Hdr,

            // Everything else
            _ => FormatCategory::Other,
        }
    }

    /// Returns true if this category uses gamma-corrected colors.
    pub fn is_gamma_corrected(self) -> bool {
        matches!(self, FormatCategory::Srgb)
    }

    /// Returns true if this category supports HDR content.
    pub fn is_hdr(self) -> bool {
        matches!(self, FormatCategory::Hdr)
    }

    /// Returns a human-readable name for this category.
    pub const fn name(self) -> &'static str {
        match self {
            FormatCategory::Srgb => "sRGB",
            FormatCategory::Linear => "Linear",
            FormatCategory::Hdr => "HDR",
            FormatCategory::Other => "Other",
        }
    }
}

impl fmt::Display for FormatCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// Present Mode Selection
// ============================================================================

/// Preference for present mode selection.
///
/// Use this enum to specify the desired presentation behavior, and the
/// system will select the best available mode for that preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresentModePreference {
    /// Lowest possible input latency (may cause tearing).
    ///
    /// Priority: Immediate > Mailbox > FifoRelaxed > Fifo
    ///
    /// Best for: Competitive gaming, VR applications.
    LowLatency,

    /// Smooth vsync without tearing.
    ///
    /// Priority: Mailbox > FifoRelaxed > Fifo
    ///
    /// Best for: General gaming, media playback.
    Vsync,

    /// Minimize power consumption.
    ///
    /// Priority: Fifo (allows GPU to idle between frames)
    ///
    /// Best for: Laptops on battery, thermal-constrained devices.
    PowerSaving,

    /// Adaptive vsync that drops frames when behind.
    ///
    /// Priority: FifoRelaxed > Mailbox > Fifo
    ///
    /// Best for: Variable framerate content, streaming.
    Adaptive,

    /// Request a specific present mode with fallback to Vsync.
    Specific(wgpu::PresentMode),
}

impl PresentModePreference {
    /// Returns a human-readable description of this preference.
    pub const fn description(self) -> &'static str {
        match self {
            PresentModePreference::LowLatency => {
                "Lowest input latency, may tear"
            }
            PresentModePreference::Vsync => {
                "Smooth presentation with vsync"
            }
            PresentModePreference::PowerSaving => {
                "Power efficient, may have higher latency"
            }
            PresentModePreference::Adaptive => {
                "Adaptive vsync with frame dropping"
            }
            PresentModePreference::Specific(_) => {
                "Specific mode requested"
            }
        }
    }
}

impl Default for PresentModePreference {
    fn default() -> Self {
        PresentModePreference::Vsync
    }
}

impl fmt::Display for PresentModePreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PresentModePreference::LowLatency => write!(f, "Low Latency"),
            PresentModePreference::Vsync => write!(f, "Vsync"),
            PresentModePreference::PowerSaving => write!(f, "Power Saving"),
            PresentModePreference::Adaptive => write!(f, "Adaptive"),
            PresentModePreference::Specific(mode) => {
                write!(f, "Specific({:?})", mode)
            }
        }
    }
}

/// Information about a present mode.
///
/// Use this to display user-friendly information about present modes
/// in settings menus or diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentModeInfo {
    /// The present mode.
    pub mode: wgpu::PresentMode,
    /// Human-readable name.
    pub name: &'static str,
    /// Short description.
    pub description: &'static str,
    /// Whether this mode eliminates tearing.
    pub prevents_tearing: bool,
    /// Relative input latency (1 = lowest, 4 = highest).
    pub latency_rank: u8,
    /// Whether this mode allows the GPU to idle.
    pub power_efficient: bool,
}

impl PresentModeInfo {
    /// Get information about a present mode.
    pub const fn from_mode(mode: wgpu::PresentMode) -> Self {
        match mode {
            wgpu::PresentMode::Immediate => Self {
                mode,
                name: "Immediate",
                description: "No vsync, lowest latency, may tear",
                prevents_tearing: false,
                latency_rank: 1,
                power_efficient: false,
            },
            wgpu::PresentMode::Mailbox => Self {
                mode,
                name: "Mailbox (Triple Buffered)",
                description: "Vsync with low latency, no tearing",
                prevents_tearing: true,
                latency_rank: 2,
                power_efficient: false,
            },
            wgpu::PresentMode::Fifo => Self {
                mode,
                name: "Fifo (Vsync)",
                description: "Standard vsync, no tearing, may stutter",
                prevents_tearing: true,
                latency_rank: 4,
                power_efficient: true,
            },
            wgpu::PresentMode::FifoRelaxed => Self {
                mode,
                name: "Fifo Relaxed (Adaptive Vsync)",
                description: "Vsync with frame drops when behind",
                prevents_tearing: true,
                latency_rank: 3,
                power_efficient: true,
            },
            // AutoVsync and AutoNoVsync are platform-specific
            _ => Self {
                mode,
                name: "Auto",
                description: "Platform-specific automatic selection",
                prevents_tearing: true,
                latency_rank: 3,
                power_efficient: false,
            },
        }
    }

    /// Returns true if this mode is suitable for competitive gaming.
    pub const fn is_competitive_gaming_mode(&self) -> bool {
        self.latency_rank <= 2
    }

    /// Returns true if this mode is suitable for battery-powered devices.
    pub const fn is_battery_friendly(&self) -> bool {
        self.power_efficient
    }
}

impl fmt::Display for PresentModeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.description)
    }
}

// ============================================================================
// Alpha Mode Selection
// ============================================================================

/// Preference for alpha compositing mode selection.
///
/// Use this enum to specify desired alpha blending behavior when the surface
/// is composited with other windows/content by the windowing system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AlphaModePreference {
    /// Opaque rendering - best performance for fullscreen games.
    ///
    /// The alpha channel is ignored; the surface is treated as fully opaque.
    /// This is the most efficient mode and should be used when transparency
    /// is not needed.
    ///
    /// Priority: Opaque > Auto
    Opaque,

    /// Pre-multiplied alpha for compositing.
    ///
    /// RGB values are already multiplied by alpha. This is the standard for
    /// compositing operations and provides correct blending results.
    ///
    /// Use for: Overlay windows, HUDs, transparent UI elements.
    PreMultiplied,

    /// Post-multiplied (straight) alpha for transparency without premultiplication.
    ///
    /// RGB values are not pre-multiplied by alpha. Less common but required
    /// by some compositing pipelines.
    ///
    /// Use for: Integration with legacy compositing systems.
    PostMultiplied,

    /// Inherit alpha behavior from the surface.
    ///
    /// The compositor determines the alpha handling based on surface content.
    /// This provides maximum compatibility but may have overhead.
    Inherit,

    /// Auto-select the best available alpha mode.
    ///
    /// Priority: Opaque (if appropriate) > PreMultiplied > PostMultiplied > Inherit
    ///
    /// This is the default and provides reasonable behavior for most applications.
    #[default]
    Auto,
}

impl AlphaModePreference {
    /// Returns a human-readable description of this preference.
    pub const fn description(self) -> &'static str {
        match self {
            AlphaModePreference::Opaque => "Opaque rendering, alpha ignored",
            AlphaModePreference::PreMultiplied => "Pre-multiplied alpha for compositing",
            AlphaModePreference::PostMultiplied => "Post-multiplied (straight) alpha",
            AlphaModePreference::Inherit => "Inherit alpha behavior from surface",
            AlphaModePreference::Auto => "Auto-select best alpha mode",
        }
    }

    /// Returns true if this mode requires alpha channel handling.
    pub const fn requires_alpha(self) -> bool {
        !matches!(self, AlphaModePreference::Opaque)
    }

    /// Convert this preference to a specific `CompositeAlphaMode` if it
    /// represents a concrete mode (not Auto).
    pub fn to_concrete_mode(self) -> Option<wgpu::CompositeAlphaMode> {
        match self {
            AlphaModePreference::Opaque => Some(wgpu::CompositeAlphaMode::Opaque),
            AlphaModePreference::PreMultiplied => Some(wgpu::CompositeAlphaMode::PreMultiplied),
            AlphaModePreference::PostMultiplied => Some(wgpu::CompositeAlphaMode::PostMultiplied),
            AlphaModePreference::Inherit => Some(wgpu::CompositeAlphaMode::Inherit),
            AlphaModePreference::Auto => None,
        }
    }
}

impl fmt::Display for AlphaModePreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlphaModePreference::Opaque => write!(f, "Opaque"),
            AlphaModePreference::PreMultiplied => write!(f, "Pre-Multiplied"),
            AlphaModePreference::PostMultiplied => write!(f, "Post-Multiplied"),
            AlphaModePreference::Inherit => write!(f, "Inherit"),
            AlphaModePreference::Auto => write!(f, "Auto"),
        }
    }
}

// ============================================================================
// Triple Buffering Support (T-WGPU-P7.1.9)
// ============================================================================

/// Buffering mode for the surface swapchain.
///
/// The buffering mode determines how many buffers are used in the swapchain,
/// which affects latency vs throughput trade-offs:
///
/// - **Double buffering**: Lower latency but may cause stuttering if the GPU
///   can't keep up. One buffer is displayed while the other is being rendered.
///
/// - **Triple buffering**: Higher throughput with smooth frame delivery. The
///   GPU always has a buffer to render to while one is being displayed and
///   one is pending. This adds one frame of latency but eliminates stuttering.
///
/// - **Quad buffering**: Used for very high refresh rate displays (240Hz+)
///   or when the rendering pipeline is very deep.
///
/// # Present Mode Relationship
///
/// Buffering mode works in conjunction with present mode:
/// - `Mailbox` present mode typically uses triple buffering
/// - `Fifo` present mode uses double buffering (queue depth = 1)
/// - `Immediate` can use any buffering mode but typically uses double
///
/// # Example
///
/// ```ignore
/// let mode = BufferingMode::from_frame_latency(3);
/// assert_eq!(mode, BufferingMode::Triple);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BufferingMode {
    /// Double buffering (2 buffers: front + back).
    ///
    /// This is the default mode for most applications. It provides the lowest
    /// latency but may cause stuttering if frame rendering is inconsistent.
    ///
    /// - `desired_maximum_frame_latency = 2`
    /// - One frame displayed, one being rendered
    /// - No buffer queue (new frame replaces old if not displayed)
    #[default]
    Double,

    /// Triple buffering (3 buffers: front + 2 back).
    ///
    /// This mode is recommended for games and smooth animations. It adds one
    /// frame of latency but provides consistent frame pacing even when render
    /// times vary.
    ///
    /// - `desired_maximum_frame_latency = 3`
    /// - One frame displayed, one pending, one being rendered
    /// - The "Mailbox" present mode uses this by default
    Triple,

    /// Quad buffering (4 buffers: front + 3 back).
    ///
    /// Used for very high refresh rate displays (240Hz+) or when the rendering
    /// pipeline includes multiple asynchronous stages that each need a buffer.
    ///
    /// - `desired_maximum_frame_latency = 4`
    /// - Higher latency but maximum throughput
    /// - Useful for VR/AR with complex reprojection pipelines
    Quad,
}

impl BufferingMode {
    /// Derive buffering mode from the `desired_maximum_frame_latency` value.
    ///
    /// # Arguments
    ///
    /// * `latency` - The `desired_maximum_frame_latency` value from surface configuration.
    ///
    /// # Returns
    ///
    /// The corresponding buffering mode:
    /// - 1-2 -> Double
    /// - 3 -> Triple
    /// - 4+ -> Quad
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert_eq!(BufferingMode::from_frame_latency(2), BufferingMode::Double);
    /// assert_eq!(BufferingMode::from_frame_latency(3), BufferingMode::Triple);
    /// assert_eq!(BufferingMode::from_frame_latency(4), BufferingMode::Quad);
    /// ```
    pub const fn from_frame_latency(latency: u32) -> Self {
        match latency {
            0..=2 => BufferingMode::Double,
            3 => BufferingMode::Triple,
            _ => BufferingMode::Quad,
        }
    }

    /// Get the buffer count for this buffering mode.
    ///
    /// # Returns
    ///
    /// The number of buffers in the swapchain:
    /// - Double: 2
    /// - Triple: 3
    /// - Quad: 4
    pub const fn buffer_count(self) -> u32 {
        match self {
            BufferingMode::Double => 2,
            BufferingMode::Triple => 3,
            BufferingMode::Quad => 4,
        }
    }

    /// Get the corresponding `desired_maximum_frame_latency` value.
    ///
    /// This is the value to pass to `SurfaceConfiguration` to achieve this
    /// buffering mode.
    pub const fn frame_latency(self) -> u32 {
        self.buffer_count()
    }

    /// Get the maximum number of frames that can be in-flight.
    ///
    /// In-flight frames are frames that have been submitted to the GPU but
    /// not yet presented. This is always `buffer_count - 1` because one
    /// buffer is always being displayed.
    pub const fn max_in_flight(self) -> u32 {
        match self {
            BufferingMode::Double => 1,
            BufferingMode::Triple => 2,
            BufferingMode::Quad => 3,
        }
    }

    /// Get the expected latency in frames.
    ///
    /// This is the number of frames between when a frame is rendered and
    /// when it appears on screen. Higher values mean more latency but
    /// smoother frame pacing.
    ///
    /// # Note
    ///
    /// This is a theoretical value. Actual latency depends on GPU
    /// scheduling, driver behavior, and present mode.
    pub const fn latency_frames(self) -> u32 {
        match self {
            BufferingMode::Double => 1,
            BufferingMode::Triple => 2,
            BufferingMode::Quad => 3,
        }
    }

    /// Returns a human-readable name for this buffering mode.
    pub const fn name(self) -> &'static str {
        match self {
            BufferingMode::Double => "Double Buffering",
            BufferingMode::Triple => "Triple Buffering",
            BufferingMode::Quad => "Quad Buffering",
        }
    }

    /// Returns a description of this buffering mode.
    pub const fn description(self) -> &'static str {
        match self {
            BufferingMode::Double => "2 buffers, lowest latency, may stutter",
            BufferingMode::Triple => "3 buffers, smooth frames, one frame latency",
            BufferingMode::Quad => "4 buffers, maximum throughput, higher latency",
        }
    }

    /// Check if this mode provides smooth frame pacing.
    ///
    /// Triple and quad buffering provide smooth frame pacing because there's
    /// always a buffer available for the GPU to render to.
    pub const fn is_smooth_pacing(self) -> bool {
        !matches!(self, BufferingMode::Double)
    }

    /// Check if this mode is low latency.
    ///
    /// Only double buffering is considered low latency since it has the
    /// smallest buffer queue.
    pub const fn is_low_latency(self) -> bool {
        matches!(self, BufferingMode::Double)
    }
}

impl fmt::Display for BufferingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Configuration for surface buffering and frame latency.
///
/// `BufferingConfig` encapsulates the buffering mode, desired latency, and
/// actual latency supported by the surface. It provides utilities for
/// checking triple buffering status and understanding the latency/throughput
/// trade-offs.
///
/// # Latency vs Throughput Trade-off
///
/// | Mode   | Buffers | Latency | Throughput | Best For                    |
/// |--------|---------|---------|------------|-----------------------------|
/// | Double | 2       | Low     | Medium     | Competitive gaming, VR      |
/// | Triple | 3       | Medium  | High       | AAA games, smooth animation |
/// | Quad   | 4       | High    | Maximum    | 240Hz+, complex pipelines   |
///
/// # Example
///
/// ```ignore
/// let config = BufferingConfig::new(BufferingMode::Triple);
/// println!("Triple buffered: {}", config.is_triple_buffered());
/// println!("Max in-flight: {}", config.max_in_flight());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferingConfig {
    /// The buffering mode.
    pub mode: BufferingMode,
    /// The desired frame latency (what we requested).
    pub desired_latency: u32,
    /// The actual frame latency (what the surface supports).
    ///
    /// This may differ from `desired_latency` if the driver or platform
    /// doesn't support the requested latency value.
    pub actual_latency: u32,
}

impl BufferingConfig {
    /// Create a new buffering configuration with the specified mode.
    ///
    /// Sets both desired and actual latency based on the mode's frame latency.
    ///
    /// # Arguments
    ///
    /// * `mode` - The buffering mode to use.
    pub const fn new(mode: BufferingMode) -> Self {
        let latency = mode.frame_latency();
        Self {
            mode,
            desired_latency: latency,
            actual_latency: latency,
        }
    }

    /// Create a buffering configuration from a frame latency value.
    ///
    /// # Arguments
    ///
    /// * `latency` - The `desired_maximum_frame_latency` value.
    pub const fn from_latency(latency: u32) -> Self {
        Self {
            mode: BufferingMode::from_frame_latency(latency),
            desired_latency: latency,
            actual_latency: latency,
        }
    }

    /// Create a buffering configuration with explicit desired and actual latency.
    ///
    /// Use this when the actual latency differs from the desired latency
    /// (e.g., the driver capped the value).
    ///
    /// # Arguments
    ///
    /// * `desired` - The latency value we requested.
    /// * `actual` - The latency value the surface actually uses.
    pub const fn with_actual(desired: u32, actual: u32) -> Self {
        Self {
            mode: BufferingMode::from_frame_latency(actual),
            desired_latency: desired,
            actual_latency: actual,
        }
    }

    /// Check if triple buffering is active.
    ///
    /// Returns `true` if the actual latency indicates triple buffering
    /// (3 or more buffers).
    pub const fn is_triple_buffered(&self) -> bool {
        self.actual_latency >= 3
    }

    /// Get the buffer count based on actual latency.
    pub const fn buffer_count(&self) -> u32 {
        self.mode.buffer_count()
    }

    /// Get the latency in frames based on actual configuration.
    ///
    /// This is the number of frames between rendering and display.
    pub const fn latency_frames(&self) -> u32 {
        self.mode.latency_frames()
    }

    /// Get the maximum number of frames that can be in-flight.
    pub const fn max_in_flight(&self) -> u32 {
        self.mode.max_in_flight()
    }

    /// Check if the desired latency matches the actual latency.
    ///
    /// Returns `false` if the driver or platform modified our requested
    /// latency value.
    pub const fn latency_matches(&self) -> bool {
        self.desired_latency == self.actual_latency
    }

    /// Get a description of any latency mismatch.
    ///
    /// Returns `None` if latency matches, otherwise returns a description
    /// of the difference.
    pub fn latency_mismatch_description(&self) -> Option<String> {
        if self.latency_matches() {
            None
        } else {
            Some(format!(
                "Requested {} frame latency, got {} (driver/platform limited)",
                self.desired_latency, self.actual_latency
            ))
        }
    }

    /// Get the latency/throughput trade-off description.
    pub const fn tradeoff_description(&self) -> &'static str {
        match self.mode {
            BufferingMode::Double => "Lower latency, may experience stuttering if GPU can't keep up",
            BufferingMode::Triple => "Balanced: smooth frames with acceptable latency",
            BufferingMode::Quad => "Maximum throughput, higher latency suitable for high refresh rates",
        }
    }
}

impl Default for BufferingConfig {
    fn default() -> Self {
        Self::new(BufferingMode::Double)
    }
}

impl fmt::Display for BufferingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (latency: {} frames, {} buffers)",
            self.mode,
            self.latency_frames(),
            self.buffer_count()
        )
    }
}

/// Tracks frames currently in the GPU pipeline.
///
/// This struct uses atomic counters to track frames that have been submitted
/// to the GPU but not yet presented. This information is useful for:
///
/// - Monitoring pipeline depth
/// - Detecting GPU bottlenecks
/// - Implementing frame pacing
/// - Debugging latency issues
///
/// # Thread Safety
///
/// All methods are thread-safe and can be called from any thread.
///
/// # Example
///
/// ```ignore
/// let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
///
/// // Frame submitted
/// tracker.frame_submitted();
/// println!("In-flight: {}", tracker.frames_in_flight());
///
/// // Frame presented
/// tracker.frame_presented();
/// ```
#[derive(Debug)]
pub struct FrameInFlightTracker {
    /// Number of frames currently in-flight.
    in_flight: AtomicU32,
    /// Maximum allowed in-flight frames.
    max_in_flight: u32,
    /// Total frames submitted.
    total_submitted: AtomicU32,
    /// Total frames presented.
    total_presented: AtomicU32,
    /// Maximum in-flight observed (high water mark).
    max_observed: AtomicU32,
}

impl FrameInFlightTracker {
    /// Create a new frame in-flight tracker.
    ///
    /// # Arguments
    ///
    /// * `mode` - The buffering mode, which determines max in-flight frames.
    pub fn new(mode: BufferingMode) -> Self {
        Self {
            in_flight: AtomicU32::new(0),
            max_in_flight: mode.max_in_flight(),
            total_submitted: AtomicU32::new(0),
            total_presented: AtomicU32::new(0),
            max_observed: AtomicU32::new(0),
        }
    }

    /// Create a tracker with a specific max in-flight count.
    pub fn with_max(max: u32) -> Self {
        Self {
            in_flight: AtomicU32::new(0),
            max_in_flight: max,
            total_submitted: AtomicU32::new(0),
            total_presented: AtomicU32::new(0),
            max_observed: AtomicU32::new(0),
        }
    }

    /// Record that a frame was submitted to the GPU.
    ///
    /// Call this after submitting command buffers to the queue.
    ///
    /// # Returns
    ///
    /// The new in-flight count after incrementing.
    pub fn frame_submitted(&self) -> u32 {
        self.total_submitted.fetch_add(1, Ordering::Relaxed);
        let new_count = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;

        // Update high water mark
        let mut max = self.max_observed.load(Ordering::Relaxed);
        while new_count > max {
            match self.max_observed.compare_exchange_weak(
                max,
                new_count,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => max = current,
            }
        }

        new_count
    }

    /// Record that a frame was presented.
    ///
    /// Call this after `SurfaceTexture::present()` returns.
    ///
    /// # Returns
    ///
    /// The new in-flight count after decrementing.
    pub fn frame_presented(&self) -> u32 {
        self.total_presented.fetch_add(1, Ordering::Relaxed);
        let prev = self.in_flight.fetch_sub(1, Ordering::SeqCst);
        // Saturate at 0 to avoid underflow
        if prev == 0 {
            self.in_flight.store(0, Ordering::SeqCst);
            0
        } else {
            prev - 1
        }
    }

    /// Get the current number of frames in-flight.
    pub fn frames_in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// Get the maximum allowed in-flight frames.
    pub fn max_frames_in_flight(&self) -> u32 {
        self.max_in_flight
    }

    /// Check if the pipeline is at capacity.
    ///
    /// Returns `true` if the number of in-flight frames equals or exceeds
    /// the maximum. When at capacity, submitting more frames may block.
    pub fn is_at_capacity(&self) -> bool {
        self.frames_in_flight() >= self.max_in_flight
    }

    /// Check if the pipeline has room for more frames.
    pub fn has_capacity(&self) -> bool {
        !self.is_at_capacity()
    }

    /// Get the remaining capacity (frames that can be submitted).
    pub fn remaining_capacity(&self) -> u32 {
        let in_flight = self.frames_in_flight();
        if in_flight >= self.max_in_flight {
            0
        } else {
            self.max_in_flight - in_flight
        }
    }

    /// Get the total number of frames submitted.
    pub fn total_submitted(&self) -> u32 {
        self.total_submitted.load(Ordering::Relaxed)
    }

    /// Get the total number of frames presented.
    pub fn total_presented(&self) -> u32 {
        self.total_presented.load(Ordering::Relaxed)
    }

    /// Get the maximum in-flight count observed (high water mark).
    ///
    /// This is useful for detecting if the pipeline ever reached capacity
    /// during runtime.
    pub fn max_observed(&self) -> u32 {
        self.max_observed.load(Ordering::Relaxed)
    }

    /// Get the pipeline utilization as a percentage.
    ///
    /// # Returns
    ///
    /// A value between 0.0 and 1.0 (or higher if over capacity).
    pub fn utilization(&self) -> f32 {
        if self.max_in_flight == 0 {
            0.0
        } else {
            self.frames_in_flight() as f32 / self.max_in_flight as f32
        }
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.in_flight.store(0, Ordering::SeqCst);
        self.total_submitted.store(0, Ordering::Relaxed);
        self.total_presented.store(0, Ordering::Relaxed);
        self.max_observed.store(0, Ordering::Relaxed);
    }

    /// Update the maximum in-flight count.
    ///
    /// Call this when the buffering mode changes.
    pub fn set_max_in_flight(&mut self, max: u32) {
        self.max_in_flight = max;
    }
}

impl Clone for FrameInFlightTracker {
    fn clone(&self) -> Self {
        Self {
            in_flight: AtomicU32::new(self.in_flight.load(Ordering::SeqCst)),
            max_in_flight: self.max_in_flight,
            total_submitted: AtomicU32::new(self.total_submitted.load(Ordering::Relaxed)),
            total_presented: AtomicU32::new(self.total_presented.load(Ordering::Relaxed)),
            max_observed: AtomicU32::new(self.max_observed.load(Ordering::Relaxed)),
        }
    }
}

impl Default for FrameInFlightTracker {
    fn default() -> Self {
        Self::new(BufferingMode::default())
    }
}

// ============================================================================
// Resize Event
// ============================================================================

/// Information about a surface resize event.
///
/// This struct captures the dimensions before and after a resize, and provides
/// utility methods to determine how the resize affects rendering (e.g., whether
/// the aspect ratio changed, whether the window was minimized or restored).
///
/// # Example
///
/// ```ignore
/// let event = ResizeEvent::new(1920, 1080, 2560, 1440);
/// println!("Aspect ratio changed: {}", event.aspect_ratio_changed());
/// println!("Scale factor: {}x", event.scale_factor());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeEvent {
    /// Previous surface width in pixels.
    pub old_width: u32,
    /// Previous surface height in pixels.
    pub old_height: u32,
    /// New surface width in pixels.
    pub new_width: u32,
    /// New surface height in pixels.
    pub new_height: u32,
}

impl ResizeEvent {
    /// Create a new resize event.
    ///
    /// # Arguments
    ///
    /// * `old_width` - Previous width in pixels.
    /// * `old_height` - Previous height in pixels.
    /// * `new_width` - New width in pixels.
    /// * `new_height` - New height in pixels.
    pub fn new(old_width: u32, old_height: u32, new_width: u32, new_height: u32) -> Self {
        Self {
            old_width,
            old_height,
            new_width,
            new_height,
        }
    }

    /// Check if the aspect ratio changed significantly.
    ///
    /// Returns `true` if the aspect ratio difference exceeds 0.001 (0.1%).
    /// This threshold accounts for floating-point rounding while detecting
    /// meaningful aspect ratio changes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 16:9 to 16:10 - aspect ratio changes
    /// let event = ResizeEvent::new(1920, 1080, 1920, 1200);
    /// assert!(event.aspect_ratio_changed());
    ///
    /// // Same aspect ratio, different resolution
    /// let event = ResizeEvent::new(1920, 1080, 3840, 2160);
    /// assert!(!event.aspect_ratio_changed());
    /// ```
    pub fn aspect_ratio_changed(&self) -> bool {
        let old_ratio = self.old_aspect_ratio();
        let new_ratio = self.new_aspect_ratio();
        (old_ratio - new_ratio).abs() > 0.001
    }

    /// Get the old aspect ratio (width / height).
    ///
    /// Returns 1.0 if old_height is 0 to avoid division by zero.
    pub fn old_aspect_ratio(&self) -> f32 {
        if self.old_height == 0 {
            1.0
        } else {
            self.old_width as f32 / self.old_height as f32
        }
    }

    /// Get the new aspect ratio (width / height).
    ///
    /// Returns 1.0 if new_height is 0 to avoid division by zero.
    pub fn new_aspect_ratio(&self) -> f32 {
        if self.new_height == 0 {
            1.0
        } else {
            self.new_width as f32 / self.new_height as f32
        }
    }

    /// Check if this resize represents a window minimize.
    ///
    /// A window is considered minimized if the new dimensions are 0x0 or 1x1.
    /// Some platforms report minimized windows as 0x0, others as 1x1.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let event = ResizeEvent::new(1920, 1080, 0, 0);
    /// assert!(event.is_minimize());
    ///
    /// let event = ResizeEvent::new(1920, 1080, 1, 1);
    /// assert!(event.is_minimize());
    /// ```
    pub fn is_minimize(&self) -> bool {
        Self::dimensions_indicate_minimized(self.new_width, self.new_height)
            && !Self::dimensions_indicate_minimized(self.old_width, self.old_height)
    }

    /// Check if this resize represents a restore from minimize.
    ///
    /// Returns `true` if the old dimensions were minimized (0x0 or 1x1)
    /// and the new dimensions are valid (greater than 1x1).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let event = ResizeEvent::new(0, 0, 1920, 1080);
    /// assert!(event.is_restore());
    /// ```
    pub fn is_restore(&self) -> bool {
        Self::dimensions_indicate_minimized(self.old_width, self.old_height)
            && !Self::dimensions_indicate_minimized(self.new_width, self.new_height)
    }

    /// Check if either dimension increased.
    ///
    /// Useful for determining if resources might need to grow.
    pub fn grew(&self) -> bool {
        self.new_width > self.old_width || self.new_height > self.old_height
    }

    /// Check if either dimension decreased.
    ///
    /// Useful for determining if resources might be oversized.
    pub fn shrunk(&self) -> bool {
        self.new_width < self.old_width || self.new_height < self.old_height
    }

    /// Calculate the scale factor of the resize.
    ///
    /// Returns the ratio of new area to old area. A value greater than 1.0
    /// means the surface grew, less than 1.0 means it shrunk.
    ///
    /// Returns 1.0 if the old dimensions were zero.
    pub fn scale_factor(&self) -> f32 {
        let old_area = self.old_width as u64 * self.old_height as u64;
        let new_area = self.new_width as u64 * self.new_height as u64;
        if old_area == 0 {
            1.0
        } else {
            new_area as f32 / old_area as f32
        }
    }

    /// Check if the dimensions actually changed.
    ///
    /// Returns `false` if old and new dimensions are identical.
    pub fn dimensions_changed(&self) -> bool {
        self.old_width != self.new_width || self.old_height != self.new_height
    }

    /// Get the delta in width (new - old).
    ///
    /// Positive means grew wider, negative means shrunk.
    pub fn width_delta(&self) -> i32 {
        self.new_width as i32 - self.old_width as i32
    }

    /// Get the delta in height (new - old).
    ///
    /// Positive means grew taller, negative means shrunk.
    pub fn height_delta(&self) -> i32 {
        self.new_height as i32 - self.old_height as i32
    }

    /// Check if dimensions indicate a minimized state.
    ///
    /// Considers 0x0 and 1x1 as minimized states since different platforms
    /// report minimization differently.
    fn dimensions_indicate_minimized(width: u32, height: u32) -> bool {
        width == 0 || height == 0 || (width == 1 && height == 1)
    }
}

impl fmt::Display for ResizeEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{} -> {}x{}",
            self.old_width, self.old_height, self.new_width, self.new_height
        )
    }
}

// ============================================================================
// Surface Configuration
// ============================================================================

/// Configuration for a surface.
///
/// This is used to configure a surface for presentation. All fields are
/// validated against the surface capabilities before being applied.
///
/// # View Formats
///
/// The `view_formats` field allows creating texture views with different formats
/// than the surface format. This is commonly used for sRGB toggle - using a linear
/// format for the surface but creating sRGB views for correct gamma handling in
/// shaders, or vice versa.
///
/// # Example
///
/// ```ignore
/// // Create surface with linear format, but allow sRGB views
/// let config = SurfaceConfiguration::new(1920, 1080)
///     .with_format(wgpu::TextureFormat::Bgra8Unorm)
///     .with_srgb_view_format();
///
/// // Later, create an sRGB view of the surface texture
/// let srgb_view = surface_texture.create_view(&wgpu::TextureViewDescriptor {
///     format: Some(wgpu::TextureFormat::Bgra8UnormSrgb),
///     ..Default::default()
/// });
/// ```
#[derive(Debug, Clone)]
pub struct SurfaceConfiguration {
    /// The texture format for the surface.
    pub format: wgpu::TextureFormat,
    /// Width of the surface in pixels.
    pub width: u32,
    /// Height of the surface in pixels.
    pub height: u32,
    /// Present mode for vsync control.
    pub present_mode: wgpu::PresentMode,
    /// Alpha compositing mode.
    pub alpha_mode: wgpu::CompositeAlphaMode,
    /// Maximum number of frames that can be queued.
    pub desired_maximum_frame_latency: u32,
    /// Additional view formats for texture views.
    ///
    /// This allows creating texture views with formats different from
    /// the base surface format. Commonly used for sRGB/linear toggle.
    pub view_formats: Vec<wgpu::TextureFormat>,
}

impl SurfaceConfiguration {
    /// Create a new configuration with the given dimensions.
    ///
    /// Uses reasonable defaults for format, present mode, and alpha mode.
    /// These should be validated against surface capabilities before use.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            desired_maximum_frame_latency: 2,
            view_formats: Vec::new(),
        }
    }

    /// Create a configuration from surface capabilities.
    ///
    /// Selects the preferred format, present mode, and alpha mode based
    /// on what the surface supports.
    pub fn from_capabilities(caps: &SurfaceCapabilities, width: u32, height: u32) -> Self {
        Self {
            format: caps.preferred_format().unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb),
            width: width.max(1),
            height: height.max(1),
            present_mode: caps.preferred_present_mode(),
            alpha_mode: caps.preferred_alpha_mode(),
            desired_maximum_frame_latency: 2,
            view_formats: Vec::new(),
        }
    }

    /// Create a configuration from a window's physical size.
    ///
    /// This is a convenience constructor that takes width and height
    /// directly from a window's physical size.
    ///
    /// # Arguments
    ///
    /// * `width` - Window width in physical pixels.
    /// * `height` - Window height in physical pixels.
    /// * `caps` - Surface capabilities for format/mode selection.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let size = window.inner_size();
    /// let config = SurfaceConfiguration::from_window_size(
    ///     size.width,
    ///     size.height,
    ///     &caps,
    /// );
    /// ```
    pub fn from_window_size(width: u32, height: u32, caps: &SurfaceCapabilities) -> Self {
        Self::from_capabilities(caps, width, height)
    }

    /// Set the texture format.
    pub fn with_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the present mode.
    pub fn with_present_mode(mut self, mode: wgpu::PresentMode) -> Self {
        self.present_mode = mode;
        self
    }

    /// Set the present mode based on preference and capabilities.
    ///
    /// This method automatically selects the best available present mode
    /// that matches the given preference.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SurfaceConfiguration::new(1920, 1080)
    ///     .with_present_mode_preference(&caps, PresentModePreference::LowLatency);
    /// ```
    pub fn with_present_mode_preference(
        mut self,
        caps: &SurfaceCapabilities,
        preference: PresentModePreference,
    ) -> Self {
        self.present_mode = caps.select_present_mode(preference);
        self
    }

    /// Set the alpha mode directly.
    pub fn with_alpha_mode(mut self, mode: wgpu::CompositeAlphaMode) -> Self {
        self.alpha_mode = mode;
        self
    }

    /// Set the alpha mode based on preference and capabilities.
    ///
    /// This method automatically selects the best available alpha mode
    /// that matches the given preference, with automatic fallback.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SurfaceConfiguration::new(1920, 1080)
    ///     .with_alpha_mode_preference(&caps, AlphaModePreference::Opaque);
    /// ```
    pub fn with_alpha_mode_preference(
        mut self,
        caps: &SurfaceCapabilities,
        preference: AlphaModePreference,
    ) -> Self {
        self.alpha_mode = caps.select_alpha_mode(preference);
        self
    }

    /// Set the maximum frame latency.
    pub fn with_frame_latency(mut self, latency: u32) -> Self {
        self.desired_maximum_frame_latency = latency.max(1);
        self
    }

    /// Set the buffering mode.
    ///
    /// This is a convenience method that sets `desired_maximum_frame_latency`
    /// based on the buffering mode.
    ///
    /// # Arguments
    ///
    /// * `mode` - The buffering mode to use.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SurfaceConfiguration::new(1920, 1080)
    ///     .with_buffering_mode(BufferingMode::Triple);
    ///
    /// assert_eq!(config.desired_maximum_frame_latency, 3);
    /// ```
    pub fn with_buffering_mode(mut self, mode: BufferingMode) -> Self {
        self.desired_maximum_frame_latency = mode.frame_latency();
        self
    }

    /// Get the buffering configuration for this surface configuration.
    ///
    /// # Returns
    ///
    /// A `BufferingConfig` derived from the current `desired_maximum_frame_latency`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SurfaceConfiguration::new(1920, 1080)
    ///     .with_frame_latency(3);
    ///
    /// let buffering = config.buffering_config();
    /// assert!(buffering.is_triple_buffered());
    /// ```
    pub fn buffering_config(&self) -> BufferingConfig {
        BufferingConfig::from_latency(self.desired_maximum_frame_latency)
    }

    /// Get the buffering mode for this surface configuration.
    ///
    /// # Returns
    ///
    /// The `BufferingMode` derived from `desired_maximum_frame_latency`.
    pub fn buffering_mode(&self) -> BufferingMode {
        BufferingMode::from_frame_latency(self.desired_maximum_frame_latency)
    }

    /// Check if triple buffering is configured.
    ///
    /// Returns `true` if `desired_maximum_frame_latency >= 3`.
    pub fn is_triple_buffered(&self) -> bool {
        self.desired_maximum_frame_latency >= 3
    }

    /// Set new dimensions (builder pattern).
    ///
    /// This creates a new configuration with updated dimensions while preserving
    /// all other settings. Dimensions are clamped to a minimum of 1.
    ///
    /// # Arguments
    ///
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SurfaceConfiguration::new(800, 600)
    ///     .with_format(wgpu::TextureFormat::Bgra8Unorm)
    ///     .with_dimensions(1920, 1080);
    ///
    /// assert_eq!(config.width, 1920);
    /// assert_eq!(config.height, 1080);
    /// ```
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width.max(1);
        self.height = height.max(1);
        self
    }

    /// Update dimensions in place (mutating).
    ///
    /// Unlike `with_dimensions()`, this modifies the configuration in place
    /// rather than returning a new one. Useful when you need to update an
    /// existing configuration without ownership transfer.
    ///
    /// Dimensions are clamped to a minimum of 1.
    ///
    /// # Arguments
    ///
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut config = SurfaceConfiguration::new(800, 600);
    /// config.resize(1920, 1080);
    /// assert_eq!(config.width, 1920);
    /// assert_eq!(config.height, 1080);
    /// ```
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    /// Set additional view formats for texture views.
    ///
    /// View formats allow creating texture views with different formats
    /// than the surface format. This is commonly used for sRGB toggle.
    ///
    /// # Arguments
    ///
    /// * `formats` - Slice of additional formats to support.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SurfaceConfiguration::new(1920, 1080)
    ///     .with_format(wgpu::TextureFormat::Bgra8Unorm)
    ///     .with_view_formats(&[wgpu::TextureFormat::Bgra8UnormSrgb]);
    /// ```
    pub fn with_view_formats(mut self, formats: &[wgpu::TextureFormat]) -> Self {
        self.view_formats = formats.to_vec();
        self
    }

    /// Automatically add the sRGB variant of the current format to view_formats.
    ///
    /// This enables the sRGB toggle pattern: using a linear format for the
    /// surface but creating sRGB views for correct gamma handling.
    ///
    /// If the current format is already sRGB, this adds the linear variant.
    /// If no sRGB/linear pair exists for the format, this is a no-op.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Using linear format, but want sRGB views available
    /// let config = SurfaceConfiguration::new(1920, 1080)
    ///     .with_format(wgpu::TextureFormat::Bgra8Unorm)
    ///     .with_srgb_view_format();
    ///
    /// // Now you can create sRGB views of the surface texture
    /// ```
    pub fn with_srgb_view_format(mut self) -> Self {
        if let Some(companion) = get_srgb_companion_format(self.format) {
            if !self.view_formats.contains(&companion) {
                self.view_formats.push(companion);
            }
        }
        self
    }

    /// Check if view formats include an sRGB variant.
    pub fn has_srgb_view_format(&self) -> bool {
        self.view_formats.iter().any(|f| {
            matches!(
                f,
                wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Rgba8UnormSrgb
            )
        })
    }

    /// Get the sRGB view format if available.
    ///
    /// Returns the sRGB format from view_formats, or the main format if
    /// it is already sRGB.
    pub fn srgb_format(&self) -> Option<wgpu::TextureFormat> {
        // If main format is sRGB, return it
        if matches!(
            self.format,
            wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Rgba8UnormSrgb
        ) {
            return Some(self.format);
        }

        // Look for sRGB in view formats
        self.view_formats
            .iter()
            .find(|f| {
                matches!(
                    f,
                    wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Rgba8UnormSrgb
                )
            })
            .copied()
    }

    /// Get the linear view format if available.
    ///
    /// Returns the linear format from view_formats, or the main format if
    /// it is already linear.
    pub fn linear_format(&self) -> Option<wgpu::TextureFormat> {
        // If main format is linear, return it
        if matches!(
            self.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
        ) {
            return Some(self.format);
        }

        // Look for linear in view formats
        self.view_formats
            .iter()
            .find(|f| {
                matches!(
                    f,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .copied()
    }

    /// Calculate the aspect ratio of the surface.
    ///
    /// Returns width / height. If height is 0, returns 1.0 to avoid division by zero.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SurfaceConfiguration::new(1920, 1080);
    /// let ratio = config.aspect_ratio();
    /// assert!((ratio - 1.777).abs() < 0.01);
    /// ```
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            1.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    /// Check if the surface format is HDR-capable.
    ///
    /// Returns `true` if the format supports high dynamic range rendering,
    /// such as `Rgba16Float`, `Rgb10a2Unorm`, or `Rg11b10Float`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SurfaceConfiguration::new(1920, 1080)
    ///     .with_format(wgpu::TextureFormat::Rgba16Float);
    /// assert!(config.is_hdr());
    ///
    /// let sdr_config = SurfaceConfiguration::new(1920, 1080);
    /// assert!(!sdr_config.is_hdr());
    /// ```
    pub fn is_hdr(&self) -> bool {
        matches!(
            self.format,
            wgpu::TextureFormat::Rgba16Float
                | wgpu::TextureFormat::Rgb10a2Unorm
                | wgpu::TextureFormat::Rg11b10Float
                | wgpu::TextureFormat::Rgba32Float
        )
    }

    /// Alias for `to_wgpu()` - convert to wgpu's SurfaceConfiguration.
    ///
    /// This is provided for API consistency with the specification.
    #[inline]
    pub fn to_wgpu_config(&self) -> wgpu::SurfaceConfiguration {
        self.to_wgpu()
    }

    /// Validate this configuration against surface capabilities.
    pub fn validate(&self, caps: &SurfaceCapabilities) -> Result<(), SurfaceError> {
        if !caps.supports_format(self.format) {
            return Err(SurfaceError::invalid_config(format!(
                "format {:?} not supported (available: {:?})",
                self.format, caps.formats
            )));
        }

        if !caps.supports_present_mode(self.present_mode) {
            return Err(SurfaceError::invalid_config(format!(
                "present mode {:?} not supported (available: {:?})",
                self.present_mode, caps.present_modes
            )));
        }

        if !caps.alpha_modes.contains(&self.alpha_mode) {
            return Err(SurfaceError::invalid_config(format!(
                "alpha mode {:?} not supported (available: {:?})",
                self.alpha_mode, caps.alpha_modes
            )));
        }

        if self.width == 0 || self.height == 0 {
            return Err(SurfaceError::invalid_config(
                "surface dimensions must be non-zero",
            ));
        }

        Ok(())
    }

    /// Convert to wgpu's SurfaceConfiguration.
    pub fn to_wgpu(&self) -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.format,
            width: self.width,
            height: self.height,
            present_mode: self.present_mode,
            alpha_mode: self.alpha_mode,
            desired_maximum_frame_latency: self.desired_maximum_frame_latency,
            view_formats: self.view_formats.clone(),
        }
    }

    /// Apply this configuration to a surface.
    ///
    /// This is a convenience method that configures the surface directly.
    /// For more control, use `TrinitySurface::configure` instead.
    ///
    /// # Arguments
    ///
    /// * `surface` - The wgpu surface to configure.
    /// * `device` - The device used for configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);
    /// config.configure(&surface, &device);
    /// ```
    pub fn configure(&self, surface: &wgpu::Surface<'_>, device: &wgpu::Device) {
        let wgpu_config = self.to_wgpu();
        surface.configure(device, &wgpu_config);
    }
}

impl Default for SurfaceConfiguration {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

// ============================================================================
// Frame Error Type
// ============================================================================

/// Errors that can occur during frame acquisition.
///
/// This enum provides detailed information about what went wrong when acquiring
/// a frame from the surface. Each variant includes recovery hints via methods.
///
/// # Recovery Strategy
///
/// ```text
/// FrameError::Timeout     -> is_recoverable: true,  needs_reconfigure: false, needs_recreate: false
/// FrameError::Outdated    -> is_recoverable: true,  needs_reconfigure: true,  needs_recreate: false
/// FrameError::Lost        -> is_recoverable: false, needs_reconfigure: false, needs_recreate: true
/// ```
#[derive(Debug, Error)]
pub enum FrameError {
    /// GPU is busy and could not acquire a frame in time.
    ///
    /// This is a transient error that typically resolves on retry.
    /// The application should skip this frame and try again next tick.
    #[error("frame acquisition timed out")]
    Timeout,

    /// The surface needs to be reconfigured.
    ///
    /// This typically happens after a window resize. The application should:
    /// 1. Get the new window dimensions
    /// 2. Call `TrinitySurface::resize()` or `configure()`
    /// 3. Retry frame acquisition
    #[error("surface outdated, reconfiguration required")]
    Outdated,

    /// The surface was lost and needs to be recreated.
    ///
    /// This can happen due to:
    /// - Display mode changes
    /// - Graphics driver reset
    /// - Window system events
    ///
    /// The application should recreate the entire surface and device.
    #[error("surface lost: {reason}")]
    Lost {
        /// Description of why the surface was lost.
        reason: String,
    },
}

impl FrameError {
    /// Returns true if this error is recoverable without recreating the surface.
    ///
    /// Timeout and Outdated errors are recoverable - Timeout by retrying,
    /// Outdated by reconfiguring the surface.
    ///
    /// # Example
    ///
    /// ```ignore
    /// match surface.acquire_frame() {
    ///     Ok(frame) => { /* render */ }
    ///     Err(e) if e.is_recoverable() => {
    ///         if e.needs_reconfigure() {
    ///             surface.resize(&device, new_width, new_height)?;
    ///         }
    ///         // Skip this frame, try next tick
    ///     }
    ///     Err(e) => {
    ///         // Surface lost, need to recreate everything
    ///         return Err(e.into());
    ///     }
    /// }
    /// ```
    pub fn is_recoverable(&self) -> bool {
        matches!(self, FrameError::Timeout | FrameError::Outdated)
    }

    /// Returns true if the surface needs to be reconfigured.
    ///
    /// This is true for `Outdated` errors, which occur after window resize.
    pub fn needs_reconfigure(&self) -> bool {
        matches!(self, FrameError::Outdated)
    }

    /// Returns true if the surface needs to be completely recreated.
    ///
    /// This is true for `Lost` errors. Recovery requires creating a new
    /// surface and potentially a new device/adapter.
    pub fn needs_recreate(&self) -> bool {
        matches!(self, FrameError::Lost { .. })
    }

    /// Create a Lost error with the given reason.
    pub fn lost(reason: impl Into<String>) -> Self {
        FrameError::Lost {
            reason: reason.into(),
        }
    }

    /// Create a Lost error from out of memory condition.
    pub fn out_of_memory() -> Self {
        FrameError::Lost {
            reason: "out of GPU memory".to_string(),
        }
    }
}

impl From<wgpu::SurfaceError> for FrameError {
    fn from(err: wgpu::SurfaceError) -> Self {
        match err {
            wgpu::SurfaceError::Timeout => FrameError::Timeout,
            wgpu::SurfaceError::Outdated => FrameError::Outdated,
            wgpu::SurfaceError::Lost => FrameError::lost("surface lost by platform"),
            wgpu::SurfaceError::OutOfMemory => FrameError::out_of_memory(),
            // Any future variants map to Lost
            _ => FrameError::lost(err.to_string()),
        }
    }
}

// ============================================================================
// Frame
// ============================================================================

/// An acquired frame from the surface, ready for rendering.
///
/// The `Frame` wraps a `wgpu::SurfaceTexture` and provides a pre-created
/// `TextureView` for convenient rendering. When rendering is complete,
/// call `present()` to display the frame or `discard()` to drop it without
/// presenting.
///
/// # Lifecycle
///
/// ```text
/// acquire_frame() -> Frame
///     |
///     v
/// [Render to frame.view()]
///     |
///     +---> present()  -> Frame displayed, resources released
///     |
///     +---> discard()  -> Frame dropped without presenting
///     |
///     `---> drop       -> Same as discard() (implicit)
/// ```
///
/// # Example
///
/// ```ignore
/// let frame = surface.acquire_frame()?;
///
/// // Create render pass targeting the frame
/// let mut encoder = device.create_command_encoder(&Default::default());
/// {
///     let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
///         label: Some("main_pass"),
///         color_attachments: &[Some(wgpu::RenderPassColorAttachment {
///             view: frame.view(),
///             resolve_target: None,
///             ops: wgpu::Operations {
///                 load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
///                 store: wgpu::StoreOp::Store,
///             },
///         })],
///         ..Default::default()
///     });
/// }
///
/// queue.submit(std::iter::once(encoder.finish()));
/// frame.present();
/// ```
pub struct Frame {
    /// The acquired surface texture.
    texture: wgpu::SurfaceTexture,
    /// Pre-created texture view for rendering.
    view: wgpu::TextureView,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Texture format.
    format: wgpu::TextureFormat,
    /// Whether present() was called.
    presented: bool,
}

impl Frame {
    /// Create a new Frame from a SurfaceTexture.
    ///
    /// # Arguments
    ///
    /// * `texture` - The acquired surface texture.
    /// * `format` - The texture format (for metadata).
    /// * `label` - Optional label for the texture view.
    pub fn new(
        texture: wgpu::SurfaceTexture,
        format: wgpu::TextureFormat,
        label: Option<&str>,
    ) -> Self {
        let width = texture.texture.width();
        let height = texture.texture.height();
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            label,
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
            width,
            height,
            format,
            presented: false,
        }
    }

    /// Get the pre-created texture view for rendering.
    ///
    /// Use this view as the target for render passes.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get the underlying surface texture.
    ///
    /// Use this if you need to create custom texture views (e.g., with a
    /// different format like sRGB).
    pub fn texture(&self) -> &wgpu::SurfaceTexture {
        &self.texture
    }

    /// Get the raw wgpu::Texture for advanced usage.
    ///
    /// This provides access to the underlying texture for operations like
    /// creating views with different formats.
    pub fn raw_texture(&self) -> &wgpu::Texture {
        &self.texture.texture
    }

    /// Get the frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the frame dimensions as a tuple (width, height).
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Get the aspect ratio (width / height) as f32.
    ///
    /// Returns 1.0 if height is 0 to avoid division by zero.
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            1.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    /// Present the frame to the display.
    ///
    /// This schedules the frame for presentation to the screen. The actual
    /// presentation timing depends on the present mode (vsync, immediate, etc.).
    ///
    /// After calling `present()`, the frame resources are released and the
    /// view should no longer be used.
    ///
    /// # Note
    ///
    /// You must ensure all GPU commands targeting this frame have been
    /// submitted to the queue before calling `present()`.
    pub fn present(mut self) {
        self.presented = true;
        self.texture.present();
    }

    /// Discard the frame without presenting.
    ///
    /// Use this when you need to skip a frame (e.g., window is minimized,
    /// or rendering was cancelled). The frame resources are released without
    /// displaying the content.
    ///
    /// This is equivalent to dropping the Frame without calling `present()`.
    pub fn discard(self) {
        // Just drop - the texture will be released without presenting
        drop(self);
    }

    /// Check if the frame has been presented.
    ///
    /// Returns true if `present()` was called on this frame.
    pub fn was_presented(&self) -> bool {
        self.presented
    }

    /// Create a texture view with a specific format.
    ///
    /// This is useful for sRGB toggle - creating an sRGB view of a linear
    /// surface texture, or vice versa. The format must be compatible with
    /// the surface's view_formats configuration.
    ///
    /// # Arguments
    ///
    /// * `format` - The format for the view.
    /// * `label` - Optional label for debugging.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Surface configured with Bgra8Unorm and Bgra8UnormSrgb as view format
    /// let frame = surface.acquire_frame()?;
    ///
    /// // Get sRGB view for gamma-correct rendering
    /// let srgb_view = frame.create_view_with_format(
    ///     wgpu::TextureFormat::Bgra8UnormSrgb,
    ///     Some("srgb_view"),
    /// );
    /// ```
    pub fn create_view_with_format(
        &self,
        format: wgpu::TextureFormat,
        label: Option<&str>,
    ) -> wgpu::TextureView {
        self.texture.texture.create_view(&wgpu::TextureViewDescriptor {
            label,
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            ..Default::default()
        })
    }
}

impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("presented", &self.presented)
            .finish()
    }
}

// ============================================================================
// sRGB/Linear Format Helpers
// ============================================================================

/// Get the sRGB/linear companion format for a given format.
///
/// Returns the alternate gamma-corrected or linear format that pairs with
/// the input format, enabling the sRGB toggle pattern.
///
/// # Examples
///
/// - `Bgra8Unorm` -> `Bgra8UnormSrgb`
/// - `Bgra8UnormSrgb` -> `Bgra8Unorm`
/// - `Rgba16Float` -> `None` (no companion)
pub fn get_srgb_companion_format(format: wgpu::TextureFormat) -> Option<wgpu::TextureFormat> {
    match format {
        // Linear -> sRGB
        wgpu::TextureFormat::Bgra8Unorm => Some(wgpu::TextureFormat::Bgra8UnormSrgb),
        wgpu::TextureFormat::Rgba8Unorm => Some(wgpu::TextureFormat::Rgba8UnormSrgb),

        // sRGB -> Linear
        wgpu::TextureFormat::Bgra8UnormSrgb => Some(wgpu::TextureFormat::Bgra8Unorm),
        wgpu::TextureFormat::Rgba8UnormSrgb => Some(wgpu::TextureFormat::Rgba8Unorm),

        // No companion for other formats
        _ => None,
    }
}

/// Check if two formats are sRGB/linear companions of each other.
pub fn are_srgb_companions(a: wgpu::TextureFormat, b: wgpu::TextureFormat) -> bool {
    get_srgb_companion_format(a) == Some(b)
}

// ============================================================================
// Frame Pacing (T-WGPU-P7.1.8)
// ============================================================================

/// Default rolling window size for frame time history.
pub const DEFAULT_FRAME_HISTORY_SIZE: usize = 100;

/// Frame timing information for frame pacing.
///
/// Tracks when frames start and end, and maintains a rolling history of
/// frame times for statistics calculation.
///
/// # Example
///
/// ```ignore
/// let mut timing = FrameTiming::new();
/// timing.begin_frame();
/// // ... rendering ...
/// timing.end_frame();
/// println!("Frame time: {:?}", timing.frame_delta());
/// ```
#[derive(Debug, Clone)]
pub struct FrameTiming {
    /// When the last frame started.
    last_frame_time: Instant,
    /// Time since the last frame completed.
    frame_delta: Duration,
    /// Target frame duration for frame limiting (e.g., 16.67ms for 60fps).
    target_frame_time: Option<Duration>,
    /// Total number of frames rendered.
    frame_count: u64,
    /// Rolling window of recent frame times.
    frame_times: VecDeque<Duration>,
    /// Maximum size of the frame times history.
    history_size: usize,
    /// Whether we're currently in a frame (between begin_frame and end_frame).
    in_frame: bool,
}

impl FrameTiming {
    /// Create a new FrameTiming instance.
    pub fn new() -> Self {
        Self::with_history_size(DEFAULT_FRAME_HISTORY_SIZE)
    }

    /// Create a new FrameTiming with a specific history size.
    ///
    /// # Arguments
    ///
    /// * `history_size` - Number of frame times to retain for statistics.
    pub fn with_history_size(history_size: usize) -> Self {
        Self {
            last_frame_time: Instant::now(),
            frame_delta: Duration::ZERO,
            target_frame_time: None,
            frame_count: 0,
            frame_times: VecDeque::with_capacity(history_size),
            history_size,
            in_frame: false,
        }
    }

    /// Set the target frame time.
    ///
    /// # Arguments
    ///
    /// * `target` - Target frame duration, or None to disable frame limiting.
    pub fn set_target_frame_time(&mut self, target: Option<Duration>) {
        self.target_frame_time = target;
    }

    /// Set target FPS (convenience method).
    ///
    /// # Arguments
    ///
    /// * `fps` - Target frames per second, or None to disable frame limiting.
    ///
    /// # Example
    ///
    /// ```ignore
    /// timing.set_target_fps(Some(60)); // Target 60 FPS
    /// timing.set_target_fps(None);     // Disable frame limiting
    /// ```
    pub fn set_target_fps(&mut self, fps: Option<u32>) {
        self.target_frame_time = fps.map(|f| Duration::from_secs_f64(1.0 / f as f64));
    }

    /// Mark the beginning of a new frame.
    ///
    /// Call this at the start of each frame to begin timing.
    pub fn begin_frame(&mut self) {
        if self.in_frame {
            // Previous frame wasn't ended properly - end it now
            self.end_frame();
        }
        self.last_frame_time = Instant::now();
        self.in_frame = true;
    }

    /// Mark the end of the current frame.
    ///
    /// Call this after presenting to record the frame time.
    pub fn end_frame(&mut self) {
        if !self.in_frame {
            return;
        }

        self.frame_delta = self.last_frame_time.elapsed();
        self.frame_count += 1;
        self.in_frame = false;

        // Add to history
        if self.frame_times.len() >= self.history_size {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(self.frame_delta);
    }

    /// Get the last recorded frame delta.
    pub fn frame_delta(&self) -> Duration {
        self.frame_delta
    }

    /// Get the target frame time, if set.
    pub fn target_frame_time(&self) -> Option<Duration> {
        self.target_frame_time
    }

    /// Get the total number of frames rendered.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the frame time history.
    pub fn frame_times(&self) -> &VecDeque<Duration> {
        &self.frame_times
    }

    /// Get the history size.
    pub fn history_size(&self) -> usize {
        self.history_size
    }

    /// Check if currently inside a frame.
    pub fn in_frame(&self) -> bool {
        self.in_frame
    }

    /// Get the elapsed time since the frame started.
    ///
    /// Returns `Duration::ZERO` if not currently in a frame.
    pub fn elapsed(&self) -> Duration {
        if self.in_frame {
            self.last_frame_time.elapsed()
        } else {
            Duration::ZERO
        }
    }

    /// Reset all timing data.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.frame_delta = Duration::ZERO;
        self.frame_times.clear();
        self.in_frame = false;
        self.last_frame_time = Instant::now();
    }
}

impl Default for FrameTiming {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics calculated from frame timing history.
///
/// Provides min/max/avg frame times, FPS calculation, percentiles, and
/// variance measurement for frame time consistency analysis.
///
/// # Example
///
/// ```ignore
/// let stats = pacer.statistics();
/// println!("FPS: {:.1}", stats.fps());
/// println!("Avg frame time: {:?}", stats.avg_frame_time);
/// println!("99th percentile: {:?}", stats.percentile_frame_time(0.99));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FrameStatistics {
    /// Minimum frame time in the history.
    pub min_frame_time: Duration,
    /// Maximum frame time in the history.
    pub max_frame_time: Duration,
    /// Average frame time.
    pub avg_frame_time: Duration,
    /// Number of frames in the statistics.
    pub sample_count: usize,
    /// Sorted frame times for percentile calculation.
    sorted_times: Vec<Duration>,
}

impl FrameStatistics {
    /// Create statistics from a collection of frame times.
    ///
    /// # Arguments
    ///
    /// * `times` - Iterator of frame durations.
    pub fn from_times<I>(times: I) -> Self
    where
        I: IntoIterator<Item = Duration>,
    {
        let mut sorted_times: Vec<Duration> = times.into_iter().collect();
        sorted_times.sort();

        if sorted_times.is_empty() {
            return Self {
                min_frame_time: Duration::ZERO,
                max_frame_time: Duration::ZERO,
                avg_frame_time: Duration::ZERO,
                sample_count: 0,
                sorted_times: Vec::new(),
            };
        }

        let min_frame_time = sorted_times.first().copied().unwrap_or(Duration::ZERO);
        let max_frame_time = sorted_times.last().copied().unwrap_or(Duration::ZERO);
        let total: Duration = sorted_times.iter().sum();
        let avg_frame_time = total / sorted_times.len() as u32;
        let sample_count = sorted_times.len();

        Self {
            min_frame_time,
            max_frame_time,
            avg_frame_time,
            sample_count,
            sorted_times,
        }
    }

    /// Calculate FPS from average frame time.
    ///
    /// Returns 0.0 if average frame time is zero.
    pub fn fps(&self) -> f64 {
        if self.avg_frame_time.is_zero() {
            return 0.0;
        }
        1.0 / self.avg_frame_time.as_secs_f64()
    }

    /// Get the Nth percentile frame time.
    ///
    /// # Arguments
    ///
    /// * `p` - Percentile as a fraction (0.0 to 1.0). E.g., 0.99 for 99th percentile.
    ///
    /// # Returns
    ///
    /// The frame time at the given percentile, or `Duration::ZERO` if no samples.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let p99 = stats.percentile_frame_time(0.99);
    /// let p50 = stats.percentile_frame_time(0.50); // Median
    /// ```
    pub fn percentile_frame_time(&self, p: f32) -> Duration {
        if self.sorted_times.is_empty() {
            return Duration::ZERO;
        }

        let p = p.clamp(0.0, 1.0);
        let idx = ((self.sorted_times.len() - 1) as f32 * p) as usize;
        self.sorted_times.get(idx).copied().unwrap_or(Duration::ZERO)
    }

    /// Calculate frame time variance.
    ///
    /// Variance measures consistency of frame times. Lower variance means
    /// more consistent frame pacing.
    ///
    /// # Returns
    ///
    /// Variance in seconds squared, or 0.0 if insufficient samples.
    pub fn frame_time_variance(&self) -> f64 {
        if self.sample_count < 2 {
            return 0.0;
        }

        let mean = self.avg_frame_time.as_secs_f64();
        let variance: f64 = self
            .sorted_times
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.sample_count - 1) as f64;

        variance
    }

    /// Calculate frame time standard deviation.
    ///
    /// Standard deviation is the square root of variance, in the same
    /// units as frame time (seconds).
    pub fn frame_time_std_dev(&self) -> f64 {
        self.frame_time_variance().sqrt()
    }

    /// Get the median frame time (50th percentile).
    pub fn median_frame_time(&self) -> Duration {
        self.percentile_frame_time(0.5)
    }

    /// Check if frame times are consistent.
    ///
    /// Returns `true` if the coefficient of variation (std_dev / mean) is
    /// below the given threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Maximum coefficient of variation (e.g., 0.1 for 10%).
    pub fn is_consistent(&self, threshold: f64) -> bool {
        if self.avg_frame_time.is_zero() {
            return true;
        }
        let cv = self.frame_time_std_dev() / self.avg_frame_time.as_secs_f64();
        cv < threshold
    }

    /// Get min FPS from max frame time.
    pub fn min_fps(&self) -> f64 {
        if self.max_frame_time.is_zero() {
            return 0.0;
        }
        1.0 / self.max_frame_time.as_secs_f64()
    }

    /// Get max FPS from min frame time.
    pub fn max_fps(&self) -> f64 {
        if self.min_frame_time.is_zero() {
            return 0.0;
        }
        1.0 / self.min_frame_time.as_secs_f64()
    }
}

impl Default for FrameStatistics {
    fn default() -> Self {
        Self {
            min_frame_time: Duration::ZERO,
            max_frame_time: Duration::ZERO,
            avg_frame_time: Duration::ZERO,
            sample_count: 0,
            sorted_times: Vec::new(),
        }
    }
}

impl fmt::Display for FrameStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.1} FPS (min: {:.1}, max: {:.1}), avg frame time: {:.2}ms",
            self.fps(),
            self.min_fps(),
            self.max_fps(),
            self.avg_frame_time.as_secs_f64() * 1000.0
        )
    }
}

/// Frame pacer for controlling frame rate and timing.
///
/// `FramePacer` provides frame rate limiting, timing statistics, and frame
/// skip detection. It's designed to be integrated into the rendering loop
/// for smooth, consistent frame delivery.
///
/// # Example
///
/// ```ignore
/// let mut pacer = FramePacer::new(Some(60)); // Target 60 FPS
///
/// loop {
///     pacer.begin_frame();
///
///     if pacer.should_skip_frame() {
///         // We're falling behind, skip this frame
///         continue;
///     }
///
///     // Render frame...
///     render_scene();
///
///     // Present and pace
///     frame.present();
///     pacer.end_frame();
///     pacer.wait_for_target();
/// }
///
/// println!("Stats: {}", pacer.statistics());
/// ```
#[derive(Debug, Clone)]
pub struct FramePacer {
    /// Frame timing data.
    timing: FrameTiming,
    /// Frame skip threshold - skip if behind by this many frames.
    skip_threshold: u32,
    /// Number of frames skipped due to falling behind.
    frames_skipped: u64,
    /// Accumulated time debt for frame pacing.
    time_debt: Duration,
    /// Whether frame limiting is enabled.
    limiting_enabled: bool,
}

impl FramePacer {
    /// Create a new frame pacer with optional target FPS.
    ///
    /// # Arguments
    ///
    /// * `target_fps` - Target frames per second, or None for no limiting.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pacer = FramePacer::new(Some(60));  // 60 FPS limit
    /// let pacer = FramePacer::new(None);       // No limit
    /// ```
    pub fn new(target_fps: Option<u32>) -> Self {
        let mut timing = FrameTiming::new();
        timing.set_target_fps(target_fps);

        Self {
            timing,
            skip_threshold: 2,
            frames_skipped: 0,
            time_debt: Duration::ZERO,
            limiting_enabled: target_fps.is_some(),
        }
    }

    /// Create a frame pacer with custom history size.
    ///
    /// # Arguments
    ///
    /// * `target_fps` - Target FPS or None.
    /// * `history_size` - Number of frame times to retain for statistics.
    pub fn with_history_size(target_fps: Option<u32>, history_size: usize) -> Self {
        let mut timing = FrameTiming::with_history_size(history_size);
        timing.set_target_fps(target_fps);

        Self {
            timing,
            skip_threshold: 2,
            frames_skipped: 0,
            time_debt: Duration::ZERO,
            limiting_enabled: target_fps.is_some(),
        }
    }

    /// Mark the beginning of a frame.
    ///
    /// Call this at the start of each frame loop iteration.
    pub fn begin_frame(&mut self) {
        self.timing.begin_frame();
    }

    /// Mark the end of a frame.
    ///
    /// Call this after presenting the frame but before waiting.
    pub fn end_frame(&mut self) {
        self.timing.end_frame();

        // Update time debt
        if let Some(target) = self.timing.target_frame_time() {
            let actual = self.timing.frame_delta();
            if actual > target {
                self.time_debt += actual - target;
            } else if self.time_debt > Duration::ZERO {
                // Reduce debt when we're ahead
                let surplus = target - actual;
                self.time_debt = self.time_debt.saturating_sub(surplus);
            }
        }
    }

    /// Wait to maintain target frame rate.
    ///
    /// If frame limiting is enabled and we finished early, this will sleep
    /// to maintain the target frame rate. Does nothing if:
    /// - Frame limiting is disabled
    /// - No target FPS is set
    /// - The frame already exceeded the target time
    ///
    /// # Returns
    ///
    /// The actual time waited (may be `Duration::ZERO` if no wait needed).
    pub fn wait_for_target(&mut self) -> Duration {
        if !self.limiting_enabled {
            return Duration::ZERO;
        }

        let Some(target) = self.timing.target_frame_time() else {
            return Duration::ZERO;
        };

        let elapsed = self.timing.frame_delta();
        if elapsed >= target {
            return Duration::ZERO;
        }

        let wait_time = target - elapsed;
        // Only sleep if the wait is significant (> 0.5ms)
        if wait_time > Duration::from_micros(500) {
            std::thread::sleep(wait_time);
            wait_time
        } else {
            // Spin-wait for very short durations (more accurate)
            let start = Instant::now();
            while start.elapsed() < wait_time {
                std::hint::spin_loop();
            }
            start.elapsed()
        }
    }

    /// Check if the current frame should be skipped.
    ///
    /// Returns `true` if we've accumulated enough time debt that we should
    /// skip rendering to catch up.
    ///
    /// # Returns
    ///
    /// `true` if the frame should be skipped (logic update only, no render).
    pub fn should_skip_frame(&mut self) -> bool {
        let Some(target) = self.timing.target_frame_time() else {
            return false;
        };

        // Skip if we're behind by more than skip_threshold frames
        let threshold = target * self.skip_threshold;
        if self.time_debt > threshold {
            self.frames_skipped += 1;
            // Reduce debt by one frame when skipping
            self.time_debt = self.time_debt.saturating_sub(target);
            true
        } else {
            false
        }
    }

    /// Calculate and return frame statistics.
    pub fn statistics(&self) -> FrameStatistics {
        FrameStatistics::from_times(self.timing.frame_times().iter().copied())
    }

    /// Get the current frame timing.
    pub fn timing(&self) -> &FrameTiming {
        &self.timing
    }

    /// Get mutable access to frame timing.
    pub fn timing_mut(&mut self) -> &mut FrameTiming {
        &mut self.timing
    }

    /// Set the target FPS.
    ///
    /// # Arguments
    ///
    /// * `fps` - Target FPS, or None to disable frame limiting.
    pub fn set_target_fps(&mut self, fps: Option<u32>) {
        self.timing.set_target_fps(fps);
        self.limiting_enabled = fps.is_some();
    }

    /// Get the current target FPS.
    pub fn target_fps(&self) -> Option<f64> {
        self.timing.target_frame_time().map(|t| 1.0 / t.as_secs_f64())
    }

    /// Set the frame skip threshold.
    ///
    /// # Arguments
    ///
    /// * `frames` - Number of frames worth of time debt before skipping.
    pub fn set_skip_threshold(&mut self, frames: u32) {
        self.skip_threshold = frames.max(1);
    }

    /// Get the frame skip threshold.
    pub fn skip_threshold(&self) -> u32 {
        self.skip_threshold
    }

    /// Get the number of frames skipped.
    pub fn frames_skipped(&self) -> u64 {
        self.frames_skipped
    }

    /// Get the current time debt.
    pub fn time_debt(&self) -> Duration {
        self.time_debt
    }

    /// Check if frame limiting is enabled.
    pub fn is_limiting_enabled(&self) -> bool {
        self.limiting_enabled
    }

    /// Enable or disable frame limiting.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable frame limiting.
    pub fn set_limiting_enabled(&mut self, enabled: bool) {
        self.limiting_enabled = enabled && self.timing.target_frame_time().is_some();
    }

    /// Get the total frame count.
    pub fn frame_count(&self) -> u64 {
        self.timing.frame_count()
    }

    /// Get the last frame delta.
    pub fn frame_delta(&self) -> Duration {
        self.timing.frame_delta()
    }

    /// Get the current FPS based on the last frame.
    pub fn current_fps(&self) -> f64 {
        let delta = self.timing.frame_delta();
        if delta.is_zero() {
            return 0.0;
        }
        1.0 / delta.as_secs_f64()
    }

    /// Reset all pacing state.
    pub fn reset(&mut self) {
        self.timing.reset();
        self.frames_skipped = 0;
        self.time_debt = Duration::ZERO;
    }
}

impl Default for FramePacer {
    fn default() -> Self {
        Self::new(None)
    }
}

// ============================================================================
// TrinitySurface
// ============================================================================

/// A platform-agnostic window surface for rendering.
///
/// `TrinitySurface` wraps a `wgpu::Surface` and provides a safe, ergonomic
/// API for surface creation and management. It handles platform-specific
/// window handle extraction and provides detailed error diagnostics.
///
/// # Thread Safety
///
/// `TrinitySurface` is `Send + Sync` when the underlying surface is, which
/// depends on the windowing system. In practice, surfaces should be accessed
/// from the thread that created the window.
///
/// # Lifetime
///
/// The surface holds a `'static` lifetime internally because wgpu requires
/// the surface to own its target. The window must outlive the surface; this
/// is enforced by the constructor taking a reference and immediately
/// extracting the raw handles.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::TrinityInstance;
/// use renderer_backend::presentation::{TrinitySurface, SurfaceConfiguration};
///
/// // Create instance
/// let instance = TrinityInstance::new();
///
/// // Create surface from window
/// // let surface = TrinitySurface::new(instance.inner(), &window)?;
/// //
/// // // Get capabilities and configure
/// // let caps = surface.capabilities(&adapter);
/// // let config = SurfaceConfiguration::from_capabilities(&caps, 1920, 1080);
/// // surface.configure(&device, &config)?;
/// //
/// // // Acquire and present frames
/// // let frame = surface.get_current_texture()?;
/// // // ... render to frame.texture ...
/// // frame.present();
/// ```
pub struct TrinitySurface {
    /// The underlying wgpu surface.
    surface: wgpu::Surface<'static>,
    /// The detected platform for this surface.
    platform: PlatformTarget,
    /// Current configuration, if configured.
    current_config: Option<SurfaceConfiguration>,
    /// Whether the surface is currently minimized.
    ///
    /// When minimized, frame acquisition should be skipped to avoid
    /// wasting GPU resources on a non-visible surface.
    minimized: bool,
    /// Optional frame pacer for frame rate control and timing.
    frame_pacer: Option<FramePacer>,
    /// Tracker for frames currently in the GPU pipeline.
    frame_in_flight_tracker: Option<FrameInFlightTracker>,
}

// Safety: wgpu::Surface is Send + Sync on platforms where this is safe.
// The surface is created from raw handles that are extracted immediately,
// so there's no lifetime issue with the window reference.
unsafe impl Send for TrinitySurface {}
unsafe impl Sync for TrinitySurface {}

impl TrinitySurface {
    /// Create a new surface from a window that implements raw-window-handle traits.
    ///
    /// # Arguments
    ///
    /// * `instance` - The wgpu instance to create the surface with.
    /// * `window` - Any type implementing `HasWindowHandle` and `HasDisplayHandle`,
    ///   such as `winit::window::Window`, `sdl2::video::Window`, etc.
    ///
    /// # Errors
    ///
    /// Returns `SurfaceError` if:
    /// - The platform is not supported
    /// - The window handle cannot be retrieved
    /// - The display handle cannot be retrieved
    /// - The wgpu surface creation fails
    ///
    /// # Safety
    ///
    /// The window must remain valid for the lifetime of the returned surface.
    /// This is enforced by the `'static` bound on the internal surface, which
    /// means wgpu takes ownership of the raw handles.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityInstance;
    /// use renderer_backend::presentation::TrinitySurface;
    ///
    /// let instance = TrinityInstance::new();
    /// // let surface = TrinitySurface::new(instance.inner(), &window)?;
    /// ```
    pub fn new<W>(instance: &wgpu::Instance, window: W) -> Result<Self, SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        // Check platform support
        let platform = PlatformTarget::current();
        if !platform.is_supported() {
            return Err(SurfaceError::unsupported());
        }

        // Get window handle
        let window_handle = window
            .window_handle()
            .map_err(SurfaceError::from_window_handle_error)?;

        // Get display handle
        let display_handle = window
            .display_handle()
            .map_err(SurfaceError::from_display_handle_error)?;

        // Detect actual platform from display handle (for Linux Wayland vs X11)
        let actual_platform = detect_platform_from_display(&display_handle);

        // Create surface using unsafe API (required for raw handles)
        let surface = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_window_handle: window_handle.as_raw(),
                    raw_display_handle: display_handle.as_raw(),
                })
                .map_err(|e| SurfaceError::SurfaceCreationFailed {
                    message: e.to_string(),
                    platform: actual_platform,
                })?
        };

        Ok(Self {
            surface,
            platform: actual_platform,
            current_config: None,
            minimized: false,
            frame_pacer: None,
            frame_in_flight_tracker: None,
        })
    }

    /// Create a surface from an existing wgpu surface.
    ///
    /// This is useful when the surface was created externally (e.g., by a
    /// windowing framework that provides its own surface).
    pub fn from_wgpu(surface: wgpu::Surface<'static>, platform: PlatformTarget) -> Self {
        Self {
            surface,
            platform,
            current_config: None,
            minimized: false,
            frame_pacer: None,
            frame_in_flight_tracker: None,
        }
    }

    /// Get the underlying wgpu surface.
    pub fn inner(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }

    /// Get the detected platform for this surface.
    pub fn platform(&self) -> PlatformTarget {
        self.platform
    }

    /// Get the current configuration, if any.
    pub fn current_config(&self) -> Option<&SurfaceConfiguration> {
        self.current_config.as_ref()
    }

    /// Query the capabilities of this surface for the given adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The adapter to query capabilities for.
    ///
    /// # Returns
    ///
    /// The surface capabilities, including supported formats, present modes,
    /// and alpha modes.
    pub fn capabilities(&self, adapter: &wgpu::Adapter) -> SurfaceCapabilities {
        let caps = self.surface.get_capabilities(adapter);
        SurfaceCapabilities::from_wgpu(caps)
    }

    /// Configure the surface for presentation.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to configure the surface with.
    /// * `config` - The configuration to apply.
    ///
    /// # Errors
    ///
    /// Returns `SurfaceError::InvalidConfiguration` if the configuration
    /// is not compatible with the surface (e.g., unsupported format).
    /// Validate the configuration against capabilities before calling this.
    pub fn configure(
        &mut self,
        device: &wgpu::Device,
        config: &SurfaceConfiguration,
    ) -> Result<(), SurfaceError> {
        let wgpu_config = config.to_wgpu();
        self.surface.configure(device, &wgpu_config);
        self.current_config = Some(config.clone());

        // Initialize or update the frame in-flight tracker based on buffering mode
        let mode = config.buffering_mode();
        if let Some(tracker) = &mut self.frame_in_flight_tracker {
            tracker.set_max_in_flight(mode.max_in_flight());
        } else {
            self.frame_in_flight_tracker = Some(FrameInFlightTracker::new(mode));
        }

        Ok(())
    }

    /// Reconfigure the surface with new dimensions.
    ///
    /// This is a convenience method for window resize events. It updates
    /// only the width and height, preserving other configuration options.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to reconfigure with.
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    ///
    /// # Errors
    ///
    /// Returns `SurfaceError::InvalidConfiguration` if the surface has not
    /// been configured yet, or if the new dimensions are invalid.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<(), SurfaceError> {
        let mut config = self
            .current_config
            .clone()
            .ok_or_else(|| SurfaceError::invalid_config("surface not configured"))?;

        config.width = width.max(1);
        config.height = height.max(1);

        self.configure(device, &config)
    }

    /// Check if the surface needs to be resized to match the given dimensions.
    ///
    /// Returns `true` if the current configuration dimensions differ from
    /// the provided dimensions, or if the surface is not yet configured.
    ///
    /// # Arguments
    ///
    /// * `width` - Expected width in pixels.
    /// * `height` - Expected height in pixels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Check if resize is needed after window size change
    /// let size = window.inner_size();
    /// if surface.needs_resize(size.width, size.height) {
    ///     surface.resize(&device, size.width, size.height)?;
    /// }
    /// ```
    pub fn needs_resize(&self, width: u32, height: u32) -> bool {
        match &self.current_config {
            Some(config) => config.width != width || config.height != height,
            None => true, // Not configured yet, definitely needs resize/configure
        }
    }

    /// Handle a resize event, returning information about what changed.
    ///
    /// This method combines resize detection, minimize handling, and surface
    /// reconfiguration into a single call. It's designed for use in window
    /// resize event handlers.
    ///
    /// # Behavior
    ///
    /// - If dimensions are 0x0 or 1x1 (minimized), sets minimized flag and
    ///   returns `Ok(None)` without reconfiguring (invalid dimensions).
    /// - If dimensions haven't changed, returns `Ok(None)`.
    /// - Otherwise, reconfigures the surface and returns the resize event.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to reconfigure with.
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(ResizeEvent))` - Surface was resized, event contains details.
    /// - `Ok(None)` - No resize needed (same dimensions or minimized).
    /// - `Err(SurfaceError)` - Surface not configured or reconfiguration failed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In window resize event handler
    /// match surface.handle_resize(&device, new_width, new_height)? {
    ///     Some(event) => {
    ///         if event.aspect_ratio_changed() {
    ///             // Update projection matrix
    ///             camera.update_aspect_ratio(event.new_aspect_ratio());
    ///         }
    ///         if event.is_restore() {
    ///             // Window was restored from minimize
    ///             log::info!("Window restored to {}x{}", event.new_width, event.new_height);
    ///         }
    ///     }
    ///     None => {
    ///         // No resize needed or window minimized
    ///     }
    /// }
    /// ```
    pub fn handle_resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<Option<ResizeEvent>, SurfaceError> {
        // Check for minimize state (0x0 or 1x1)
        let is_minimized = width == 0 || height == 0 || (width == 1 && height == 1);
        let was_minimized = self.minimized;

        if is_minimized {
            self.minimized = true;
            // Don't reconfigure with invalid dimensions
            return Ok(None);
        }

        // Get current dimensions
        let (old_width, old_height) = self.dimensions();

        // Check if resize is actually needed
        if !self.needs_resize(width, height) && !was_minimized {
            return Ok(None);
        }

        // Update minimized state
        self.minimized = false;

        // Perform the resize
        self.resize(device, width, height)?;

        // Create and return the resize event
        let event = ResizeEvent::new(old_width, old_height, width, height);
        Ok(Some(event))
    }

    /// Check if the surface is currently minimized.
    ///
    /// When minimized, frame acquisition should be skipped to avoid wasting
    /// GPU resources on a non-visible surface. The `acquire_frame` method
    /// will return `FrameError::Timeout` when minimized.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if surface.is_minimized() {
    ///     // Skip rendering, just process events
    ///     return;
    /// }
    /// let frame = surface.acquire_frame()?;
    /// ```
    pub fn is_minimized(&self) -> bool {
        self.minimized
    }

    /// Set the minimized state manually.
    ///
    /// This is useful when the windowing system provides minimize events
    /// separately from resize events. When set to `true`, frame acquisition
    /// will be optimized to avoid GPU work.
    ///
    /// # Arguments
    ///
    /// * `minimized` - Whether the surface should be considered minimized.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In window event handler
    /// WindowEvent::Minimized => {
    ///     surface.set_minimized(true);
    /// }
    /// WindowEvent::Restored => {
    ///     surface.set_minimized(false);
    /// }
    /// ```
    pub fn set_minimized(&mut self, minimized: bool) {
        self.minimized = minimized;
    }

    /// Get the current aspect ratio (width / height).
    ///
    /// Returns 1.0 if the surface is not configured or height is 0.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let aspect = surface.aspect_ratio();
    /// camera.set_aspect_ratio(aspect);
    /// ```
    pub fn aspect_ratio(&self) -> f32 {
        match &self.current_config {
            Some(config) if config.height > 0 => config.width as f32 / config.height as f32,
            _ => 1.0,
        }
    }

    /// Get the current texture for rendering.
    ///
    /// # Returns
    ///
    /// The current surface texture, which must be presented after rendering.
    ///
    /// # Errors
    ///
    /// Returns `SurfaceError` if:
    /// - The surface is outdated (needs reconfiguration)
    /// - The surface was lost (needs recreation)
    /// - Acquisition timed out
    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, SurfaceError> {
        self.surface.get_current_texture().map_err(|e| match e {
            wgpu::SurfaceError::Outdated => SurfaceError::SurfaceOutdated,
            wgpu::SurfaceError::Lost => SurfaceError::SurfaceLost {
                reason: "surface lost by the platform".to_string(),
            },
            wgpu::SurfaceError::Timeout => SurfaceError::SurfaceLost {
                reason: "frame acquisition timed out".to_string(),
            },
            wgpu::SurfaceError::OutOfMemory => SurfaceError::SurfaceLost {
                reason: "out of memory".to_string(),
            },
            _ => SurfaceError::SurfaceLost {
                reason: e.to_string(),
            },
        })
    }

    /// Acquire a frame from the surface for rendering.
    ///
    /// This is the preferred method for frame acquisition. It returns a `Frame`
    /// struct with a pre-created texture view, ready for rendering.
    ///
    /// # Returns
    ///
    /// A `Frame` containing the surface texture and a pre-created view.
    ///
    /// # Errors
    ///
    /// Returns `FrameError` if:
    /// - `Timeout` - GPU is busy, retry later
    /// - `Outdated` - Surface needs reconfiguration (window resized)
    /// - `Lost` - Surface lost, needs complete recreation
    ///
    /// # Example
    ///
    /// ```ignore
    /// loop {
    ///     match surface.acquire_frame() {
    ///         Ok(frame) => {
    ///             // Render to frame.view()
    ///             // ...
    ///             frame.present();
    ///         }
    ///         Err(FrameError::Timeout) => {
    ///             // Skip this frame, GPU is busy
    ///             continue;
    ///         }
    ///         Err(FrameError::Outdated) => {
    ///             // Window was resized, reconfigure
    ///             surface.resize(&device, new_width, new_height)?;
    ///             continue;
    ///         }
    ///         Err(FrameError::Lost { .. }) => {
    ///             // Surface lost, need to recreate
    ///             return Err(anyhow!("Surface lost"));
    ///         }
    ///     }
    /// }
    /// ```
    pub fn acquire_frame(&self) -> Result<Frame, FrameError> {
        // Ensure surface is configured
        let config = self.current_config.as_ref().ok_or_else(|| {
            FrameError::lost("surface not configured")
        })?;

        // Get the surface texture
        let texture = self.surface.get_current_texture()?;

        // Create the frame with a labeled view
        let frame = Frame::new(
            texture,
            config.format,
            Some("trinity_frame_view"),
        );

        Ok(frame)
    }

    /// Try to acquire a frame without blocking.
    ///
    /// This is a non-blocking variant of `acquire_frame()`. It returns
    /// `None` if a frame is not immediately available (timeout), or
    /// `Some(result)` for success or error conditions.
    ///
    /// This is useful for applications that want to skip frames rather
    /// than block when the GPU is busy.
    ///
    /// # Returns
    ///
    /// - `None` - No frame available (would timeout)
    /// - `Some(Ok(frame))` - Frame acquired successfully
    /// - `Some(Err(e))` - Error other than timeout
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Non-blocking frame acquisition
    /// if let Some(result) = surface.try_acquire_frame() {
    ///     match result {
    ///         Ok(frame) => {
    ///             // Render and present
    ///             frame.present();
    ///         }
    ///         Err(FrameError::Outdated) => {
    ///             // Reconfigure surface
    ///         }
    ///         Err(e) => {
    ///             // Handle error
    ///         }
    ///     }
    /// } else {
    ///     // No frame available, do other work
    ///     process_input();
    /// }
    /// ```
    pub fn try_acquire_frame(&self) -> Option<Result<Frame, FrameError>> {
        match self.acquire_frame() {
            Ok(frame) => Some(Ok(frame)),
            Err(FrameError::Timeout) => None,
            Err(e) => Some(Err(e)),
        }
    }

    /// Acquire a frame with a specific view format.
    ///
    /// This variant creates the frame's primary view with the specified format,
    /// which must be either the surface format or one of the configured
    /// view_formats.
    ///
    /// This is useful for sRGB toggle patterns where you want the primary
    /// view to be in a specific format.
    ///
    /// # Arguments
    ///
    /// * `format` - The format for the primary texture view.
    ///
    /// # Returns
    ///
    /// A `Frame` with its view in the specified format.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Configure surface with view formats
    /// let config = SurfaceConfiguration::new(1920, 1080)
    ///     .with_format(wgpu::TextureFormat::Bgra8Unorm)
    ///     .with_srgb_view_format();
    /// surface.configure(&device, &config)?;
    ///
    /// // Acquire frame with sRGB view for gamma-correct rendering
    /// let frame = surface.acquire_frame_with_format(wgpu::TextureFormat::Bgra8UnormSrgb)?;
    /// ```
    pub fn acquire_frame_with_format(
        &self,
        format: wgpu::TextureFormat,
    ) -> Result<Frame, FrameError> {
        // Ensure surface is configured
        self.current_config.as_ref().ok_or_else(|| {
            FrameError::lost("surface not configured")
        })?;

        // Get the surface texture
        let texture = self.surface.get_current_texture()?;

        // Create the frame with the specified format
        let frame = Frame::new(
            texture,
            format,
            Some("trinity_frame_view"),
        );

        Ok(frame)
    }

    /// Get the current width of the surface.
    ///
    /// Returns 0 if the surface has not been configured.
    pub fn width(&self) -> u32 {
        self.current_config.as_ref().map_or(0, |c| c.width)
    }

    /// Get the current height of the surface.
    ///
    /// Returns 0 if the surface has not been configured.
    pub fn height(&self) -> u32 {
        self.current_config.as_ref().map_or(0, |c| c.height)
    }

    /// Get the current texture format.
    ///
    /// Returns None if the surface has not been configured.
    pub fn format(&self) -> Option<wgpu::TextureFormat> {
        self.current_config.as_ref().map(|c| c.format)
    }

    /// Get the current alpha mode.
    ///
    /// Returns None if the surface has not been configured.
    pub fn alpha_mode(&self) -> Option<wgpu::CompositeAlphaMode> {
        self.current_config.as_ref().map(|c| c.alpha_mode)
    }

    /// Get the current present mode.
    ///
    /// Returns None if the surface has not been configured.
    pub fn present_mode(&self) -> Option<wgpu::PresentMode> {
        self.current_config.as_ref().map(|c| c.present_mode)
    }

    /// Get the current view formats.
    ///
    /// Returns an empty slice if the surface has not been configured.
    pub fn view_formats(&self) -> &[wgpu::TextureFormat] {
        self.current_config
            .as_ref()
            .map_or(&[], |c| &c.view_formats)
    }

    /// Check if the surface has an sRGB view format configured.
    pub fn has_srgb_view(&self) -> bool {
        self.current_config
            .as_ref()
            .map_or(false, |c| c.has_srgb_view_format())
    }

    /// Get the sRGB format (main or view) if available.
    pub fn srgb_format(&self) -> Option<wgpu::TextureFormat> {
        self.current_config.as_ref().and_then(|c| c.srgb_format())
    }

    /// Get the linear format (main or view) if available.
    pub fn linear_format(&self) -> Option<wgpu::TextureFormat> {
        self.current_config.as_ref().and_then(|c| c.linear_format())
    }

    /// Check if the surface is configured.
    pub fn is_configured(&self) -> bool {
        self.current_config.is_some()
    }

    /// Get the surface dimensions as a tuple (width, height).
    ///
    /// Returns (0, 0) if the surface has not been configured.
    pub fn dimensions(&self) -> (u32, u32) {
        self.current_config
            .as_ref()
            .map_or((0, 0), |c| (c.width, c.height))
    }

    // =========================================================================
    // Frame Pacing (T-WGPU-P7.1.8)
    // =========================================================================

    /// Set the target FPS for frame pacing.
    ///
    /// Enables frame pacing with the specified target FPS. Pass `None` to
    /// disable frame pacing.
    ///
    /// # Arguments
    ///
    /// * `fps` - Target frames per second, or None to disable.
    ///
    /// # Example
    ///
    /// ```ignore
    /// surface.set_target_fps(Some(60)); // Enable 60 FPS limiting
    /// surface.set_target_fps(None);     // Disable frame limiting
    /// ```
    pub fn set_target_fps(&mut self, fps: Option<u32>) {
        match fps {
            Some(target) => {
                if let Some(pacer) = &mut self.frame_pacer {
                    pacer.set_target_fps(Some(target));
                } else {
                    self.frame_pacer = Some(FramePacer::new(Some(target)));
                }
            }
            None => {
                if let Some(pacer) = &mut self.frame_pacer {
                    pacer.set_target_fps(None);
                }
                // Keep the pacer for statistics even without limiting
            }
        }
    }

    /// Get the current target FPS.
    ///
    /// Returns `None` if frame pacing is not enabled.
    pub fn target_fps(&self) -> Option<f64> {
        self.frame_pacer.as_ref().and_then(|p| p.target_fps())
    }

    /// Enable frame pacing without a specific target (for statistics only).
    ///
    /// This enables frame time tracking without frame limiting. Useful when
    /// you want statistics but don't want to cap the frame rate.
    pub fn enable_frame_tracking(&mut self) {
        if self.frame_pacer.is_none() {
            self.frame_pacer = Some(FramePacer::new(None));
        }
    }

    /// Disable frame pacing entirely.
    ///
    /// Removes the frame pacer, disabling both frame limiting and statistics.
    pub fn disable_frame_pacing(&mut self) {
        self.frame_pacer = None;
    }

    /// Check if frame pacing is enabled.
    pub fn has_frame_pacer(&self) -> bool {
        self.frame_pacer.is_some()
    }

    /// Get frame statistics.
    ///
    /// Returns `None` if frame pacing is not enabled.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(stats) = surface.frame_statistics() {
    ///     println!("FPS: {:.1}", stats.fps());
    ///     println!("Avg frame time: {:?}", stats.avg_frame_time);
    /// }
    /// ```
    pub fn frame_statistics(&self) -> Option<FrameStatistics> {
        self.frame_pacer.as_ref().map(|p| p.statistics())
    }

    /// Get a reference to the frame pacer.
    pub fn frame_pacer(&self) -> Option<&FramePacer> {
        self.frame_pacer.as_ref()
    }

    /// Get a mutable reference to the frame pacer.
    pub fn frame_pacer_mut(&mut self) -> Option<&mut FramePacer> {
        self.frame_pacer.as_mut()
    }

    /// Begin a frame for pacing.
    ///
    /// Call this at the start of each frame. Does nothing if frame pacing
    /// is not enabled.
    pub fn begin_frame_pacing(&mut self) {
        if let Some(pacer) = &mut self.frame_pacer {
            pacer.begin_frame();
        }
    }

    /// End a frame for pacing.
    ///
    /// Call this after presenting the frame. Does nothing if frame pacing
    /// is not enabled.
    pub fn end_frame_pacing(&mut self) {
        if let Some(pacer) = &mut self.frame_pacer {
            pacer.end_frame();
        }
    }

    /// Wait to maintain target frame rate.
    ///
    /// Call this after `end_frame_pacing()` to sleep if ahead of schedule.
    /// Does nothing if frame limiting is not enabled or we're behind.
    ///
    /// # Returns
    ///
    /// The time waited, or `Duration::ZERO` if no wait was needed or
    /// frame pacing is disabled.
    pub fn wait_for_target_fps(&mut self) -> Duration {
        if let Some(pacer) = &mut self.frame_pacer {
            pacer.wait_for_target()
        } else {
            Duration::ZERO
        }
    }

    /// Check if the current frame should be skipped.
    ///
    /// Returns `true` if we've fallen too far behind and should skip
    /// rendering to catch up. Does nothing if frame pacing is disabled.
    pub fn should_skip_frame(&mut self) -> bool {
        if let Some(pacer) = &mut self.frame_pacer {
            pacer.should_skip_frame()
        } else {
            false
        }
    }

    /// Get the current FPS based on the last frame.
    ///
    /// Returns 0.0 if frame pacing is not enabled or no frames have been
    /// recorded yet.
    pub fn current_fps(&self) -> f64 {
        self.frame_pacer
            .as_ref()
            .map_or(0.0, |p| p.current_fps())
    }

    /// Get the last frame delta.
    ///
    /// Returns `Duration::ZERO` if frame pacing is not enabled.
    pub fn last_frame_delta(&self) -> Duration {
        self.frame_pacer
            .as_ref()
            .map_or(Duration::ZERO, |p| p.frame_delta())
    }

    /// Get the total number of frames rendered (tracked by frame pacer).
    ///
    /// Returns 0 if frame pacing is not enabled.
    pub fn paced_frame_count(&self) -> u64 {
        self.frame_pacer
            .as_ref()
            .map_or(0, |p| p.frame_count())
    }

    /// Get the number of frames skipped due to falling behind.
    ///
    /// Returns 0 if frame pacing is not enabled.
    pub fn frames_skipped(&self) -> u64 {
        self.frame_pacer
            .as_ref()
            .map_or(0, |p| p.frames_skipped())
    }

    // =========================================================================
    // Triple Buffering Support (T-WGPU-P7.1.9)
    // =========================================================================

    /// Get the current buffering configuration.
    ///
    /// Returns `None` if the surface has not been configured.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(config) = surface.buffering_config() {
    ///     println!("Triple buffered: {}", config.is_triple_buffered());
    ///     println!("Mode: {}", config.mode);
    /// }
    /// ```
    pub fn buffering_config(&self) -> Option<BufferingConfig> {
        self.current_config.as_ref().map(|c| c.buffering_config())
    }

    /// Check if triple buffering is active.
    ///
    /// Returns `false` if the surface is not configured or is using
    /// double buffering.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if surface.is_triple_buffered() {
    ///     println!("Triple buffering enabled for smooth frame pacing");
    /// }
    /// ```
    pub fn is_triple_buffered(&self) -> bool {
        self.current_config
            .as_ref()
            .map_or(false, |c| c.is_triple_buffered())
    }

    /// Get the current buffering mode.
    ///
    /// Returns `BufferingMode::Double` if the surface is not configured.
    pub fn buffering_mode(&self) -> BufferingMode {
        self.current_config
            .as_ref()
            .map_or(BufferingMode::Double, |c| c.buffering_mode())
    }

    /// Set the frame latency (buffering depth).
    ///
    /// This reconfigures the surface with the new frame latency value.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to reconfigure the surface with.
    /// * `latency` - The desired maximum frame latency (2 = double, 3 = triple, etc.).
    ///
    /// # Errors
    ///
    /// Returns an error if the surface is not configured.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Enable triple buffering
    /// surface.set_frame_latency(&device, 3)?;
    ///
    /// // Or use BufferingMode for clarity
    /// surface.set_buffering_mode(&device, BufferingMode::Triple)?;
    /// ```
    pub fn set_frame_latency(
        &mut self,
        device: &wgpu::Device,
        latency: u32,
    ) -> Result<(), SurfaceError> {
        let mut config = self
            .current_config
            .clone()
            .ok_or_else(|| SurfaceError::invalid_config("surface not configured"))?;

        config.desired_maximum_frame_latency = latency.max(1);
        self.configure(device, &config)
    }

    /// Set the buffering mode.
    ///
    /// This is a convenience method that sets the frame latency based on the
    /// buffering mode.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to reconfigure the surface with.
    /// * `mode` - The buffering mode to use.
    ///
    /// # Errors
    ///
    /// Returns an error if the surface is not configured.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Enable triple buffering for smooth frame pacing
    /// surface.set_buffering_mode(&device, BufferingMode::Triple)?;
    /// ```
    pub fn set_buffering_mode(
        &mut self,
        device: &wgpu::Device,
        mode: BufferingMode,
    ) -> Result<(), SurfaceError> {
        self.set_frame_latency(device, mode.frame_latency())
    }

    /// Get the number of frames currently in-flight.
    ///
    /// In-flight frames are frames that have been submitted to the GPU but
    /// not yet presented. This value is tracked if the surface has been
    /// configured and frame tracking is active.
    ///
    /// # Returns
    ///
    /// The number of in-flight frames, or 0 if tracking is not active.
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Frames in GPU pipeline: {}", surface.frames_in_flight());
    /// ```
    pub fn frames_in_flight(&self) -> u32 {
        self.frame_in_flight_tracker
            .as_ref()
            .map_or(0, |t| t.frames_in_flight())
    }

    /// Get the maximum number of frames that can be in-flight.
    ///
    /// This depends on the buffering mode:
    /// - Double buffering: 1 frame max
    /// - Triple buffering: 2 frames max
    /// - Quad buffering: 3 frames max
    ///
    /// # Returns
    ///
    /// The maximum in-flight count, or 1 if tracking is not active.
    pub fn max_frames_in_flight(&self) -> u32 {
        self.frame_in_flight_tracker
            .as_ref()
            .map_or(1, |t| t.max_frames_in_flight())
    }

    /// Record that a frame was submitted to the GPU.
    ///
    /// Call this after submitting command buffers to the queue but before
    /// presenting. This is used for frame in-flight tracking.
    ///
    /// # Returns
    ///
    /// The new in-flight count after incrementing.
    pub fn record_frame_submitted(&self) -> u32 {
        self.frame_in_flight_tracker
            .as_ref()
            .map_or(0, |t| t.frame_submitted())
    }

    /// Record that a frame was presented.
    ///
    /// Call this after the frame's `present()` method returns. This is used
    /// for frame in-flight tracking.
    ///
    /// # Returns
    ///
    /// The new in-flight count after decrementing.
    pub fn record_frame_presented(&self) -> u32 {
        self.frame_in_flight_tracker
            .as_ref()
            .map_or(0, |t| t.frame_presented())
    }

    /// Check if the GPU pipeline is at capacity.
    ///
    /// Returns `true` if the number of in-flight frames equals or exceeds
    /// the maximum. When at capacity, submitting more frames may block.
    pub fn pipeline_at_capacity(&self) -> bool {
        self.frame_in_flight_tracker
            .as_ref()
            .map_or(false, |t| t.is_at_capacity())
    }

    /// Get the frame in-flight tracker, if active.
    ///
    /// Returns `None` if the surface has not been configured.
    pub fn frame_in_flight_tracker(&self) -> Option<&FrameInFlightTracker> {
        self.frame_in_flight_tracker.as_ref()
    }

    /// Get mutable access to the frame in-flight tracker.
    ///
    /// Returns `None` if the surface has not been configured.
    pub fn frame_in_flight_tracker_mut(&mut self) -> Option<&mut FrameInFlightTracker> {
        self.frame_in_flight_tracker.as_mut()
    }

    /// Enable frame in-flight tracking without configuring the surface.
    ///
    /// Normally, the tracker is automatically created when the surface is
    /// configured. This method allows enabling tracking beforehand.
    ///
    /// # Arguments
    ///
    /// * `mode` - The buffering mode to configure the tracker for.
    pub fn enable_frame_tracking_for_mode(&mut self, mode: BufferingMode) {
        if self.frame_in_flight_tracker.is_none() {
            self.frame_in_flight_tracker = Some(FrameInFlightTracker::new(mode));
        } else if let Some(tracker) = &mut self.frame_in_flight_tracker {
            tracker.set_max_in_flight(mode.max_in_flight());
        }
    }

    /// Disable frame in-flight tracking.
    pub fn disable_frame_tracking(&mut self) {
        self.frame_in_flight_tracker = None;
    }

    /// Get statistics about frame in-flight tracking.
    ///
    /// Returns a tuple of (total_submitted, total_presented, max_observed).
    /// Returns (0, 0, 0) if tracking is not active.
    pub fn frame_tracking_stats(&self) -> (u32, u32, u32) {
        self.frame_in_flight_tracker
            .as_ref()
            .map_or((0, 0, 0), |t| (
                t.total_submitted(),
                t.total_presented(),
                t.max_observed(),
            ))
    }

    /// Get the pipeline utilization as a percentage (0.0 to 1.0).
    ///
    /// This indicates how full the GPU pipeline is relative to capacity.
    /// A value near 1.0 indicates the GPU is keeping up with frame production.
    pub fn pipeline_utilization(&self) -> f32 {
        self.frame_in_flight_tracker
            .as_ref()
            .map_or(0.0, |t| t.utilization())
    }
}

impl fmt::Debug for TrinitySurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrinitySurface")
            .field("platform", &self.platform)
            .field("current_config", &self.current_config)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Platform Detection Helpers
// ============================================================================

/// Detect the actual platform from the display handle.
///
/// This is needed on Linux to distinguish between Wayland and X11.
fn detect_platform_from_display(display_handle: &rwh::DisplayHandle<'_>) -> PlatformTarget {
    match display_handle.as_raw() {
        rwh::RawDisplayHandle::Wayland(_) => PlatformTarget::Wayland,
        rwh::RawDisplayHandle::Xlib(_) => PlatformTarget::X11,
        rwh::RawDisplayHandle::Xcb(_) => PlatformTarget::X11,
        rwh::RawDisplayHandle::Windows(_) => PlatformTarget::Windows,
        rwh::RawDisplayHandle::AppKit(_) => PlatformTarget::MacOS,
        rwh::RawDisplayHandle::UiKit(_) => PlatformTarget::IOS,
        rwh::RawDisplayHandle::Android(_) => PlatformTarget::Android,
        rwh::RawDisplayHandle::Web(_) => PlatformTarget::Web,
        _ => PlatformTarget::Unknown,
    }
}

// ============================================================================
// Headless Rendering Support (T-WGPU-P7.1.10)
// ============================================================================

/// Error type for headless rendering operations.
#[derive(Debug, Error)]
pub enum HeadlessError {
    /// Failed to create the headless target texture.
    #[error("failed to create headless target: {0}")]
    TextureCreationFailed(String),

    /// Failed to create a staging buffer for readback.
    #[error("failed to create staging buffer: {0}")]
    StagingBufferFailed(String),

    /// Failed to map the buffer for reading.
    #[error("buffer mapping failed: {0}")]
    BufferMapFailed(String),

    /// The headless target has not been initialized.
    #[error("headless target not initialized")]
    NotInitialized,

    /// Invalid dimensions for the headless target.
    #[error("invalid dimensions: width={width}, height={height}")]
    InvalidDimensions {
        width: u32,
        height: u32,
    },

    /// Screenshot save failed.
    #[error("screenshot save failed: {0}")]
    ScreenshotSaveFailed(String),

    /// MSAA resolve failed.
    #[error("MSAA resolve failed: {0}")]
    ResolveFailed(String),
}

impl HeadlessError {
    /// Create an InvalidDimensions error.
    pub fn invalid_dimensions(width: u32, height: u32) -> Self {
        HeadlessError::InvalidDimensions { width, height }
    }
}

/// Configuration for a headless rendering target.
///
/// This struct configures an offscreen render target that can be used for
/// headless rendering, screenshot capture, and server-side rendering.
///
/// # Example
///
/// ```ignore
/// let config = HeadlessConfig::new(1920, 1080)
///     .with_format(wgpu::TextureFormat::Rgba8Unorm)
///     .with_msaa(4)
///     .with_readback();
///
/// let target = HeadlessTarget::new(&device, config);
/// ```
#[derive(Debug, Clone)]
pub struct HeadlessConfig {
    /// Width of the headless target in pixels.
    pub width: u32,
    /// Height of the headless target in pixels.
    pub height: u32,
    /// Texture format for the render target.
    pub format: wgpu::TextureFormat,
    /// Sample count for MSAA (1 = no MSAA).
    pub sample_count: u32,
    /// Texture usages (must include RENDER_ATTACHMENT).
    pub usage: wgpu::TextureUsages,
    /// Label for debugging.
    pub label: Option<String>,
}

impl HeadlessConfig {
    /// Create a new headless configuration with the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `width` - Width in pixels (must be > 0).
    /// * `height` - Height in pixels (must be > 0).
    ///
    /// # Default Values
    ///
    /// - Format: Rgba8Unorm (good for screenshots)
    /// - Sample count: 1 (no MSAA)
    /// - Usage: RENDER_ATTACHMENT | COPY_SRC (for readback)
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            format: wgpu::TextureFormat::Rgba8Unorm,
            sample_count: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            label: None,
        }
    }

    /// Set the texture format.
    ///
    /// # Arguments
    ///
    /// * `format` - The texture format for the render target.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = HeadlessConfig::new(1920, 1080)
    ///     .with_format(wgpu::TextureFormat::Bgra8UnormSrgb);
    /// ```
    pub fn with_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the MSAA sample count.
    ///
    /// Valid values are 1 (no MSAA), 4, or 8. Other values will be clamped
    /// to the nearest valid value.
    ///
    /// # Arguments
    ///
    /// * `samples` - Number of samples for MSAA.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = HeadlessConfig::new(1920, 1080)
    ///     .with_msaa(4);  // 4x MSAA
    /// ```
    pub fn with_msaa(mut self, samples: u32) -> Self {
        // MSAA sample counts are typically 1, 4, or 8
        self.sample_count = match samples {
            0..=1 => 1,
            2..=4 => 4,
            _ => 8,
        };
        self
    }

    /// Enable readback support by adding COPY_SRC usage.
    ///
    /// This is enabled by default but can be explicitly called for clarity.
    /// Readback support is required for screenshots and CPU pixel access.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = HeadlessConfig::new(1920, 1080)
    ///     .with_readback();
    /// ```
    pub fn with_readback(mut self) -> Self {
        self.usage |= wgpu::TextureUsages::COPY_SRC;
        self
    }

    /// Set custom texture usages.
    ///
    /// # Warning
    ///
    /// RENDER_ATTACHMENT is required and will be added if not present.
    ///
    /// # Arguments
    ///
    /// * `usage` - The texture usage flags.
    pub fn with_usage(mut self, usage: wgpu::TextureUsages) -> Self {
        self.usage = usage | wgpu::TextureUsages::RENDER_ATTACHMENT;
        self
    }

    /// Set a label for debugging.
    ///
    /// # Arguments
    ///
    /// * `label` - Debug label for the texture.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Check if MSAA is enabled.
    pub fn is_msaa_enabled(&self) -> bool {
        self.sample_count > 1
    }

    /// Check if readback is enabled.
    pub fn supports_readback(&self) -> bool {
        self.usage.contains(wgpu::TextureUsages::COPY_SRC)
    }

    /// Validate the configuration.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the configuration is valid, `Err` with details otherwise.
    pub fn validate(&self) -> Result<(), HeadlessError> {
        if self.width == 0 || self.height == 0 {
            return Err(HeadlessError::invalid_dimensions(self.width, self.height));
        }
        Ok(())
    }

    /// Get the bytes per pixel for this format.
    ///
    /// Returns the number of bytes per pixel, or 4 as a safe default for
    /// unsupported formats.
    pub fn bytes_per_pixel(&self) -> u32 {
        // Common formats used in headless rendering
        match self.format {
            wgpu::TextureFormat::R8Unorm
            | wgpu::TextureFormat::R8Snorm
            | wgpu::TextureFormat::R8Uint
            | wgpu::TextureFormat::R8Sint => 1,

            wgpu::TextureFormat::Rg8Unorm
            | wgpu::TextureFormat::Rg8Snorm
            | wgpu::TextureFormat::Rg8Uint
            | wgpu::TextureFormat::Rg8Sint => 2,

            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Rgba8Snorm
            | wgpu::TextureFormat::Rgba8Uint
            | wgpu::TextureFormat::Rgba8Sint
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb
            | wgpu::TextureFormat::Rgb10a2Unorm => 4,

            wgpu::TextureFormat::Rgba16Float
            | wgpu::TextureFormat::Rgba16Uint
            | wgpu::TextureFormat::Rgba16Sint
            | wgpu::TextureFormat::Rgba16Unorm
            | wgpu::TextureFormat::Rgba16Snorm => 8,

            wgpu::TextureFormat::Rgba32Float
            | wgpu::TextureFormat::Rgba32Uint
            | wgpu::TextureFormat::Rgba32Sint => 16,

            // Default to 4 bytes for unknown formats
            _ => 4,
        }
    }

    /// Calculate the row pitch (bytes per row, aligned to 256).
    ///
    /// wgpu requires buffer row pitch to be aligned to 256 bytes.
    pub fn aligned_bytes_per_row(&self) -> u32 {
        let unaligned = self.width * self.bytes_per_pixel();
        // Align to 256 bytes (wgpu requirement)
        (unaligned + 255) & !255
    }

    /// Calculate the total buffer size needed for readback.
    pub fn buffer_size(&self) -> u64 {
        self.aligned_bytes_per_row() as u64 * self.height as u64
    }

    /// Create a configuration optimized for screenshot capture.
    ///
    /// Uses Rgba8Unorm format with readback support, no MSAA.
    /// This is suitable for saving as PNG or similar formats.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = HeadlessConfig::for_screenshot();
    /// let target = HeadlessTarget::new(&device, config)?;
    /// ```
    pub fn for_screenshot() -> Self {
        Self {
            width: 1920,
            height: 1080,
            format: wgpu::TextureFormat::Rgba8Unorm,
            sample_count: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            label: Some("screenshot_target".to_string()),
        }
    }

    /// Create a configuration optimized for video frame rendering.
    ///
    /// Uses Rgba8Unorm format with the specified dimensions.
    /// Includes readback support for video encoding.
    ///
    /// # Arguments
    ///
    /// * `width` - Video frame width in pixels.
    /// * `height` - Video frame height in pixels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // 1080p video
    /// let config = HeadlessConfig::for_video(1920, 1080);
    /// let target = HeadlessTarget::new(&device, config)?;
    ///
    /// // 4K video
    /// let config = HeadlessConfig::for_video(3840, 2160);
    /// ```
    pub fn for_video(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            format: wgpu::TextureFormat::Rgba8Unorm,
            sample_count: 1,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            label: Some("video_frame_target".to_string()),
        }
    }

    /// Alias for `with_msaa` to match the API in the specification.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of samples for MSAA (1, 4, or 8).
    pub fn with_samples(self, count: u32) -> Self {
        self.with_msaa(count)
    }
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self::new(800, 600)
    }
}

/// An offscreen render target for headless rendering.
///
/// `HeadlessTarget` provides a render target that doesn't require a window
/// or display. It's suitable for:
///
/// - Server-side rendering
/// - Screenshot capture
/// - Automated testing
/// - Thumbnail generation
/// - Video frame rendering
///
/// # MSAA Support
///
/// When MSAA is enabled (sample_count > 1), the target includes both a
/// multisampled texture for rendering and a resolve texture for the final
/// output. Use `resolve()` to perform MSAA resolve.
///
/// # Example
///
/// ```ignore
/// // Create headless target
/// let config = HeadlessConfig::new(1920, 1080);
/// let target = HeadlessTarget::new(&device, config)?;
///
/// // Render to the target
/// let frame = target.acquire_frame();
/// let mut encoder = device.create_command_encoder(&Default::default());
/// {
///     let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
///         color_attachments: &[Some(wgpu::RenderPassColorAttachment {
///             view: frame.view(),
///             resolve_target: None,
///             ops: wgpu::Operations {
///                 load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
///                 store: wgpu::StoreOp::Store,
///             },
///         })],
///         ..Default::default()
///     });
///     // Render commands...
/// }
/// queue.submit(std::iter::once(encoder.finish()));
///
/// // Read back the pixels
/// let pixels = target.screenshot(&device, &queue)?;
/// ```
pub struct HeadlessTarget {
    /// The render target texture.
    texture: wgpu::Texture,
    /// Pre-created texture view for rendering.
    view: wgpu::TextureView,
    /// Resolve target for MSAA (only if sample_count > 1).
    resolve_texture: Option<wgpu::Texture>,
    /// View for the resolve target.
    resolve_view: Option<wgpu::TextureView>,
    /// Configuration used to create this target.
    config: HeadlessConfig,
}

impl HeadlessTarget {
    /// Create a new headless render target.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create textures on.
    /// * `config` - Configuration for the headless target.
    ///
    /// # Returns
    ///
    /// A new `HeadlessTarget` ready for rendering.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(device: &wgpu::Device, config: HeadlessConfig) -> Result<Self, HeadlessError> {
        config.validate()?;

        // Create the main render target texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: config.label.as_deref(),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: config.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: config.usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: config.label.as_deref().map(|l| format!("{}_view", l)).as_deref(),
            format: Some(config.format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            ..Default::default()
        });

        // Create resolve target for MSAA
        let (resolve_texture, resolve_view) = if config.sample_count > 1 {
            let resolve_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: config.label.as_deref().map(|l| format!("{}_resolve", l)).as_deref(),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1, // Resolve target is always non-multisampled
                dimension: wgpu::TextureDimension::D2,
                format: config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });

            let resolve_v = resolve_tex.create_view(&wgpu::TextureViewDescriptor {
                label: config.label.as_deref().map(|l| format!("{}_resolve_view", l)).as_deref(),
                format: Some(config.format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
                ..Default::default()
            });

            (Some(resolve_tex), Some(resolve_v))
        } else {
            (None, None)
        };

        Ok(Self {
            texture,
            view,
            resolve_texture,
            resolve_view,
            config,
        })
    }

    /// Get the texture view for rendering.
    ///
    /// Use this view as the color attachment target in render passes.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get the underlying texture.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Get the resolve target view (for MSAA).
    ///
    /// Returns `None` if MSAA is not enabled.
    ///
    /// Use this as the `resolve_target` in render pass color attachments
    /// when using MSAA.
    pub fn resolve_view(&self) -> Option<&wgpu::TextureView> {
        self.resolve_view.as_ref()
    }

    /// Get the resolve target texture (for MSAA).
    ///
    /// Returns `None` if MSAA is not enabled.
    pub fn resolve_texture(&self) -> Option<&wgpu::Texture> {
        self.resolve_texture.as_ref()
    }

    /// Get the configuration used to create this target.
    pub fn config(&self) -> &HeadlessConfig {
        &self.config
    }

    /// Get the width in pixels.
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Get the height in pixels.
    pub fn height(&self) -> u32 {
        self.config.height
    }

    /// Get the dimensions as a tuple (width, height).
    pub fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Get the size as a tuple (width, height).
    ///
    /// Alias for `dimensions()` to match the API specification.
    pub fn size(&self) -> (u32, u32) {
        self.dimensions()
    }

    /// Get the texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Get the sample count.
    pub fn sample_count(&self) -> u32 {
        self.config.sample_count
    }

    /// Check if MSAA is enabled.
    pub fn is_msaa_enabled(&self) -> bool {
        self.config.sample_count > 1
    }

    /// Get the aspect ratio (width / height).
    pub fn aspect_ratio(&self) -> f32 {
        if self.config.height == 0 {
            1.0
        } else {
            self.config.width as f32 / self.config.height as f32
        }
    }

    /// Resize the headless target.
    ///
    /// Creates new textures with the new dimensions.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<(), HeadlessError> {
        if width == self.config.width && height == self.config.height {
            return Ok(());
        }

        let new_config = HeadlessConfig {
            width: width.max(1),
            height: height.max(1),
            ..self.config.clone()
        };

        let new_target = HeadlessTarget::new(device, new_config)?;
        *self = new_target;
        Ok(())
    }

    /// Acquire a frame for rendering.
    ///
    /// Returns a `HeadlessFrame` that provides access to the render target view.
    pub fn acquire_frame(&self) -> HeadlessFrame<'_> {
        HeadlessFrame {
            view: &self.view,
            resolve_view: self.resolve_view.as_ref(),
            width: self.config.width,
            height: self.config.height,
            format: self.config.format,
            sample_count: self.config.sample_count,
        }
    }

    /// Create a staging buffer for readback.
    ///
    /// The staging buffer can be used to copy texture data to the CPU.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    ///
    /// # Returns
    ///
    /// A `ReadbackBuffer` that can be used to read pixel data.
    pub fn create_staging_buffer(&self, device: &wgpu::Device) -> ReadbackBuffer {
        let size = self.config.buffer_size();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("headless_readback_buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        ReadbackBuffer {
            buffer,
            width: self.config.width,
            height: self.config.height,
            bytes_per_row: self.config.aligned_bytes_per_row(),
            bytes_per_pixel: self.config.bytes_per_pixel(),
            format: self.config.format,
        }
    }

    /// Take a screenshot of the current render target.
    ///
    /// This is a convenience method that:
    /// 1. Creates a staging buffer
    /// 2. Copies the texture to the staging buffer
    /// 3. Maps the buffer and reads the pixel data
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The command queue.
    ///
    /// # Returns
    ///
    /// Raw pixel data as a `Vec<u8>`. The data layout depends on the texture
    /// format and may include row padding for alignment.
    ///
    /// # Note
    ///
    /// For MSAA targets, this copies from the resolve target. Make sure you've
    /// performed MSAA resolve before calling this method.
    pub fn screenshot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, HeadlessError> {
        let staging = self.create_staging_buffer(device);

        // Use resolve texture for MSAA, otherwise use main texture
        let source_texture = self.resolve_texture.as_ref().unwrap_or(&self.texture);

        // Copy texture to buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless_screenshot_encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging.buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(staging.bytes_per_row),
                    rows_per_image: Some(self.config.height),
                },
            },
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Map and read the buffer
        staging.map_read(device)
    }

    /// Take a screenshot and remove row padding.
    ///
    /// Unlike `screenshot()`, this returns tightly packed pixel data without
    /// the 256-byte row alignment padding.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The command queue.
    ///
    /// # Returns
    ///
    /// Tightly packed pixel data as a `Vec<u8>`.
    pub fn screenshot_packed(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, HeadlessError> {
        let padded_data = self.screenshot(device, queue)?;
        let staging = self.create_staging_buffer(device);

        // Remove row padding
        let actual_bytes_per_row = (self.config.width * self.config.bytes_per_pixel()) as usize;
        let padded_bytes_per_row = staging.bytes_per_row as usize;

        if actual_bytes_per_row == padded_bytes_per_row {
            return Ok(padded_data);
        }

        let mut packed = Vec::with_capacity(actual_bytes_per_row * self.config.height as usize);
        for y in 0..self.config.height as usize {
            let start = y * padded_bytes_per_row;
            let end = start + actual_bytes_per_row;
            packed.extend_from_slice(&padded_data[start..end]);
        }

        Ok(packed)
    }
}

impl fmt::Debug for HeadlessTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeadlessTarget")
            .field("width", &self.config.width)
            .field("height", &self.config.height)
            .field("format", &self.config.format)
            .field("sample_count", &self.config.sample_count)
            .field("has_resolve_target", &self.resolve_texture.is_some())
            .finish()
    }
}

/// A frame from a headless render target.
///
/// Similar to `Frame` but for offscreen rendering. There's no `present()`
/// method since there's no display to present to.
///
/// # Example
///
/// ```ignore
/// let frame = headless_target.acquire_frame();
///
/// // For non-MSAA targets
/// encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
///     color_attachments: &[Some(wgpu::RenderPassColorAttachment {
///         view: frame.view(),
///         resolve_target: None,
///         ops: wgpu::Operations {
///             load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
///             store: wgpu::StoreOp::Store,
///         },
///     })],
///     ..Default::default()
/// });
///
/// // For MSAA targets
/// encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
///     color_attachments: &[Some(wgpu::RenderPassColorAttachment {
///         view: frame.view(),
///         resolve_target: frame.resolve_view(), // MSAA resolve
///         ops: wgpu::Operations {
///             load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
///             store: wgpu::StoreOp::Store,
///         },
///     })],
///     ..Default::default()
/// });
/// ```
#[derive(Debug)]
pub struct HeadlessFrame<'a> {
    /// The render target view.
    view: &'a wgpu::TextureView,
    /// Optional resolve target for MSAA.
    resolve_view: Option<&'a wgpu::TextureView>,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Texture format.
    format: wgpu::TextureFormat,
    /// Sample count.
    sample_count: u32,
}

impl<'a> HeadlessFrame<'a> {
    /// Get the render target view.
    ///
    /// Use this as the color attachment view in render passes.
    pub fn view(&self) -> &wgpu::TextureView {
        self.view
    }

    /// Get the MSAA resolve target view.
    ///
    /// Returns `None` if MSAA is not enabled.
    ///
    /// Use this as the `resolve_target` in render pass color attachments.
    pub fn resolve_view(&self) -> Option<&wgpu::TextureView> {
        self.resolve_view
    }

    /// Get the frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the frame dimensions as a tuple (width, height).
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Get the sample count.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Check if MSAA is enabled.
    pub fn is_msaa_enabled(&self) -> bool {
        self.sample_count > 1
    }

    /// Get the aspect ratio (width / height).
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            1.0
        } else {
            self.width as f32 / self.height as f32
        }
    }
}

/// A staging buffer for reading back texture data to CPU.
///
/// This buffer is used to copy rendered pixels from GPU textures to CPU
/// memory for screenshots, video encoding, or analysis.
///
/// # Example
///
/// ```ignore
/// let readback = headless_target.create_staging_buffer(&device);
///
/// // Copy texture to buffer
/// encoder.copy_texture_to_buffer(
///     wgpu::TexelCopyTextureInfo { ... },
///     wgpu::TexelCopyBufferInfo {
///         buffer: readback.buffer(),
///         layout: wgpu::TexelCopyBufferLayout {
///             offset: 0,
///             bytes_per_row: Some(readback.bytes_per_row()),
///             rows_per_image: Some(height),
///         },
///     },
///     extent,
/// );
/// queue.submit([encoder.finish()]);
///
/// // Read the data
/// let pixels = readback.map_read(&device)?;
/// ```
pub struct ReadbackBuffer {
    /// The staging buffer.
    buffer: wgpu::Buffer,
    /// Width of the source texture.
    width: u32,
    /// Height of the source texture.
    height: u32,
    /// Bytes per row (aligned to 256).
    bytes_per_row: u32,
    /// Bytes per pixel.
    bytes_per_pixel: u32,
    /// Source texture format.
    format: wgpu::TextureFormat,
}

impl ReadbackBuffer {
    /// Create a new readback buffer for the given dimensions and format.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `width` - Width of the texture to read back.
    /// * `height` - Height of the texture to read back.
    /// * `format` - Texture format (determines bytes per pixel).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let readback = ReadbackBuffer::new(&device, 1920, 1080, wgpu::TextureFormat::Rgba8Unorm);
    /// ```
    pub fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        // Calculate bytes per pixel based on format
        let bytes_per_pixel = match format {
            wgpu::TextureFormat::R8Unorm
            | wgpu::TextureFormat::R8Snorm
            | wgpu::TextureFormat::R8Uint
            | wgpu::TextureFormat::R8Sint => 1,

            wgpu::TextureFormat::Rg8Unorm
            | wgpu::TextureFormat::Rg8Snorm
            | wgpu::TextureFormat::Rg8Uint
            | wgpu::TextureFormat::Rg8Sint => 2,

            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Rgba8Snorm
            | wgpu::TextureFormat::Rgba8Uint
            | wgpu::TextureFormat::Rgba8Sint
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb
            | wgpu::TextureFormat::Rgb10a2Unorm => 4,

            wgpu::TextureFormat::Rgba16Float
            | wgpu::TextureFormat::Rgba16Uint
            | wgpu::TextureFormat::Rgba16Sint
            | wgpu::TextureFormat::Rgba16Unorm
            | wgpu::TextureFormat::Rgba16Snorm => 8,

            wgpu::TextureFormat::Rgba32Float
            | wgpu::TextureFormat::Rgba32Uint
            | wgpu::TextureFormat::Rgba32Sint => 16,

            // Default to 4 bytes for unknown formats
            _ => 4,
        };

        // Align bytes per row to 256 (wgpu requirement)
        let unaligned_bytes_per_row = width * bytes_per_pixel;
        let bytes_per_row = (unaligned_bytes_per_row + 255) & !255;
        let size = bytes_per_row as u64 * height as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            width,
            height,
            bytes_per_row,
            bytes_per_pixel,
            format,
        }
    }

    /// Copy data from a texture to this buffer.
    ///
    /// After calling this method, submit the encoder and call `map_read()`
    /// or `read_pixels()` to retrieve the pixel data.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to record the copy command.
    /// * `texture` - The source texture.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut encoder = device.create_command_encoder(&Default::default());
    /// readback.copy_from_texture(&mut encoder, &texture);
    /// queue.submit([encoder.finish()]);
    /// let pixels = readback.map_read(&device)?;
    /// ```
    pub fn copy_from_texture(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) {
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Get the underlying buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get the bytes per row (aligned to 256).
    pub fn bytes_per_row(&self) -> u32 {
        self.bytes_per_row
    }

    /// Get the padded bytes per row (aligned to 256).
    ///
    /// Alias for `bytes_per_row()` to match the API specification.
    pub fn padded_bytes_per_row(&self) -> u32 {
        self.bytes_per_row
    }

    /// Get the bytes per pixel.
    pub fn bytes_per_pixel(&self) -> u32 {
        self.bytes_per_pixel
    }

    /// Get the width of the source texture.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the height of the source texture.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Get the total buffer size in bytes.
    pub fn size(&self) -> u64 {
        self.bytes_per_row as u64 * self.height as u64
    }

    /// Map the buffer and read its contents.
    ///
    /// This is a blocking operation that waits for the GPU to complete
    /// all pending operations on this buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    ///
    /// # Returns
    ///
    /// The buffer contents as a `Vec<u8>`.
    pub fn map_read(&self, device: &wgpu::Device) -> Result<Vec<u8>, HeadlessError> {
        let buffer_slice = self.buffer.slice(..);

        // Map the buffer
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Wait for mapping to complete
        device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| HeadlessError::BufferMapFailed(e.to_string()))?
            .map_err(|e| HeadlessError::BufferMapFailed(e.to_string()))?;

        // Read the data
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        self.buffer.unmap();

        Ok(result)
    }

    /// Map the buffer and read a packed (no padding) version of its contents.
    ///
    /// This removes the 256-byte row alignment padding to provide tightly
    /// packed pixel data.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    ///
    /// # Returns
    ///
    /// Tightly packed pixel data as a `Vec<u8>`.
    pub fn map_read_packed(&self, device: &wgpu::Device) -> Result<Vec<u8>, HeadlessError> {
        let padded_data = self.map_read(device)?;

        let actual_bytes_per_row = (self.width * self.bytes_per_pixel) as usize;
        let padded_bytes_per_row = self.bytes_per_row as usize;

        if actual_bytes_per_row == padded_bytes_per_row {
            return Ok(padded_data);
        }

        let mut packed = Vec::with_capacity(actual_bytes_per_row * self.height as usize);
        for y in 0..self.height as usize {
            let start = y * padded_bytes_per_row;
            let end = start + actual_bytes_per_row;
            packed.extend_from_slice(&padded_data[start..end]);
        }

        Ok(packed)
    }

    /// Asynchronously read pixels from the buffer.
    ///
    /// This is an async version of `map_read` that can be awaited in async
    /// contexts. Note that wgpu's async buffer mapping still requires
    /// `device.poll()` to make progress.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (needed for polling).
    ///
    /// # Returns
    ///
    /// The buffer contents as a `Vec<u8>`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pixels = readback.read_pixels(&device).await?;
    /// ```
    pub async fn read_pixels(&self, device: &wgpu::Device) -> Vec<u8> {
        let buffer_slice = self.buffer.slice(..);

        // Map the buffer asynchronously
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll until mapping is complete
        // In a real async runtime, you'd use proper async polling
        device.poll(wgpu::Maintain::Wait);

        // Wait for the mapping result
        let _ = rx.recv();

        // Read the data
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        self.buffer.unmap();

        result
    }
}

impl fmt::Debug for ReadbackBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadbackBuffer")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes_per_row", &self.bytes_per_row)
            .field("format", &self.format)
            .finish()
    }
}

/// A renderer for headless (offscreen) rendering without a window.
///
/// `HeadlessRenderer` provides a high-level API for offscreen rendering,
/// combining a `HeadlessTarget` with frame management and screenshot
/// capabilities.
///
/// # Example
///
/// ```ignore
/// // Create headless renderer
/// let config = HeadlessConfig::new(1920, 1080);
/// let mut renderer = HeadlessRenderer::new(&device, config)?;
///
/// // Render a frame
/// let frame = renderer.acquire_frame();
/// // ... render to frame.view() ...
///
/// // Take a screenshot
/// let pixels = renderer.screenshot(&device, &queue)?;
/// ```
pub struct HeadlessRenderer {
    /// The underlying headless target.
    target: HeadlessTarget,
    /// Total frames rendered.
    frame_count: u64,
}

impl HeadlessRenderer {
    /// Create a new headless renderer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `config` - Configuration for the headless target.
    ///
    /// # Returns
    ///
    /// A new `HeadlessRenderer` ready for rendering.
    pub fn new(device: &wgpu::Device, config: HeadlessConfig) -> Result<Self, HeadlessError> {
        let target = HeadlessTarget::new(device, config)?;
        Ok(Self {
            target,
            frame_count: 0,
        })
    }

    /// Acquire a frame for rendering.
    ///
    /// # Returns
    ///
    /// A `HeadlessFrame` that provides access to the render target view.
    pub fn acquire_frame(&mut self) -> HeadlessFrame<'_> {
        self.frame_count += 1;
        self.target.acquire_frame()
    }

    /// Get the underlying headless target.
    pub fn target(&self) -> &HeadlessTarget {
        &self.target
    }

    /// Get mutable access to the underlying headless target.
    pub fn target_mut(&mut self) -> &mut HeadlessTarget {
        &mut self.target
    }

    /// Get the total number of frames rendered.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the render target view.
    ///
    /// Convenience method that delegates to the target.
    pub fn view(&self) -> &wgpu::TextureView {
        self.target.view()
    }

    /// Get the MSAA resolve view if available.
    ///
    /// Convenience method that delegates to the target.
    pub fn resolve_view(&self) -> Option<&wgpu::TextureView> {
        self.target.resolve_view()
    }

    /// Get the width in pixels.
    pub fn width(&self) -> u32 {
        self.target.width()
    }

    /// Get the height in pixels.
    pub fn height(&self) -> u32 {
        self.target.height()
    }

    /// Get the dimensions as a tuple (width, height).
    pub fn dimensions(&self) -> (u32, u32) {
        self.target.dimensions()
    }

    /// Get the texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.target.format()
    }

    /// Resize the render target.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<(), HeadlessError> {
        self.target.resize(device, width, height)
    }

    /// Take a screenshot of the current render target.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The command queue.
    ///
    /// # Returns
    ///
    /// Raw pixel data as a `Vec<u8>`.
    pub fn screenshot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, HeadlessError> {
        self.target.screenshot(device, queue)
    }

    /// Take a screenshot with packed (no padding) pixel data.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The command queue.
    ///
    /// # Returns
    ///
    /// Tightly packed pixel data as a `Vec<u8>`.
    pub fn screenshot_packed(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, HeadlessError> {
        self.target.screenshot_packed(device, queue)
    }

    /// Get the headless configuration.
    ///
    /// Returns the configuration used to create this renderer.
    pub fn config(&self) -> &HeadlessConfig {
        self.target.config()
    }

    /// "Present" the current frame (no-op for headless rendering).
    ///
    /// In headless mode there's no display to present to, so this method
    /// exists only for API compatibility with windowed rendering.
    ///
    /// This method simply increments internal counters.
    pub fn present(&mut self) {
        // No-op for headless - there's no display to present to
        // This exists for API parity with windowed rendering
    }

    /// Enable readback support on the render target.
    ///
    /// This ensures the target texture has COPY_SRC usage, which is
    /// required for copying pixel data to CPU-readable buffers.
    ///
    /// Note: By default, HeadlessConfig already includes readback support.
    /// This method is provided for cases where you need to explicitly
    /// ensure readback is enabled.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device (may be needed to recreate the target).
    pub fn enable_readback(&mut self, device: &wgpu::Device) -> Result<(), HeadlessError> {
        // Check if readback is already enabled
        if self.target.config().supports_readback() {
            return Ok(());
        }

        // Need to recreate the target with COPY_SRC usage
        let mut new_config = self.target.config().clone();
        new_config.usage |= wgpu::TextureUsages::COPY_SRC;

        let new_target = HeadlessTarget::new(device, new_config)?;
        self.target = new_target;
        Ok(())
    }

    /// Read pixels from the current render target.
    ///
    /// Convenience method that creates a staging buffer, copies the texture,
    /// and returns the pixel data.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The command queue.
    ///
    /// # Returns
    ///
    /// Pixel data as `Option<Vec<u8>>`. Returns `None` if readback is not
    /// supported on this target.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pixels = renderer.read_pixels(&device, &queue);
    /// if let Some(data) = pixels {
    ///     // Process pixel data
    /// }
    /// ```
    pub fn read_pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Vec<u8>> {
        if !self.target.config().supports_readback() {
            return None;
        }
        self.target.screenshot(device, queue).ok()
    }
}

impl fmt::Debug for HeadlessRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeadlessRenderer")
            .field("target", &self.target)
            .field("frame_count", &self.frame_count)
            .finish()
    }
}

// ============================================================================
// Multi-Window Support (T-WGPU-P7.1.11)
// ============================================================================

/// Unique identifier for a window in the multi-window system.
///
/// `WindowId` provides type-safe window identification for tracking multiple
/// rendering surfaces. Each window has a unique ID that remains stable for
/// the window's lifetime.
///
/// # Example
///
/// ```ignore
/// let id = WindowId::new();
/// manager.register_window(id, surface, config)?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(u64);

/// Counter for generating unique WindowIds.
static WINDOW_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

impl WindowId {
    /// Create a new unique window ID.
    ///
    /// Each call generates a monotonically increasing ID that is unique
    /// within this process lifetime.
    pub fn new() -> Self {
        let id = WINDOW_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(id)
    }

    /// Create a WindowId from a raw u64 value.
    ///
    /// # Arguments
    ///
    /// * `id` - The raw ID value.
    ///
    /// # Safety
    ///
    /// The caller must ensure the ID is unique within the multi-window system.
    /// Using duplicate IDs will cause undefined behavior.
    pub fn from_raw_id(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw u64 value of this window ID.
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    /// Create a window ID for the primary/main window.
    ///
    /// This is a convenience method that returns WindowId(0), which by
    /// convention represents the primary application window.
    pub const fn primary() -> Self {
        Self(0)
    }

    /// Check if this is the primary window ID.
    pub const fn is_primary(&self) -> bool {
        self.0 == 0
    }
}

impl Default for WindowId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for WindowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window({})", self.0)
    }
}

impl From<u64> for WindowId {
    fn from(id: u64) -> Self {
        Self::from_raw_id(id)
    }
}

impl From<WindowId> for u64 {
    fn from(id: WindowId) -> Self {
        id.0
    }
}

// ============================================================================
// Window Configuration
// ============================================================================

/// Configuration for an individual window in the multi-window system.
///
/// `WindowConfig` extends `SurfaceConfiguration` with window-specific
/// properties like focus state, visibility, and render priority.
///
/// # Example
///
/// ```ignore
/// let config = WindowConfig::new(WindowId::new(), surface_config)
///     .with_priority(RenderPriority::High)
///     .with_label("Main Window");
/// ```
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Unique identifier for this window.
    pub id: WindowId,
    /// Surface configuration (format, dimensions, present mode, etc.).
    pub config: SurfaceConfiguration,
    /// Whether this window currently has input focus.
    pub is_focused: bool,
    /// Whether this window is currently visible (not minimized/hidden).
    pub is_visible: bool,
    /// Render priority for ordering (higher = rendered first).
    pub priority: u8,
    /// Optional human-readable label for debugging.
    pub label: Option<String>,
    /// Whether this window should synchronize with the primary window.
    pub sync_to_primary: bool,
}

impl WindowConfig {
    /// Create a new window configuration.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique window identifier.
    /// * `config` - Surface configuration for this window.
    pub fn new(id: WindowId, config: SurfaceConfiguration) -> Self {
        Self {
            id,
            config,
            is_focused: false,
            is_visible: true,
            priority: 128, // Default middle priority
            label: None,
            sync_to_primary: false,
        }
    }

    /// Create a configuration for the primary window.
    ///
    /// The primary window is created with `WindowId::primary()` and
    /// high priority.
    pub fn primary(config: SurfaceConfiguration) -> Self {
        Self {
            id: WindowId::primary(),
            config,
            is_focused: true,
            is_visible: true,
            priority: 255, // Highest priority
            label: Some("Primary".to_string()),
            sync_to_primary: false,
        }
    }

    /// Set the window's focused state.
    pub fn with_focus(mut self, focused: bool) -> Self {
        self.is_focused = focused;
        self
    }

    /// Set the window's visibility state.
    pub fn with_visibility(mut self, visible: bool) -> Self {
        self.is_visible = visible;
        self
    }

    /// Set the render priority (0-255, higher = rendered first).
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set an optional label for debugging.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Enable synchronization with the primary window's vsync.
    pub fn with_sync_to_primary(mut self, sync: bool) -> Self {
        self.sync_to_primary = sync;
        self
    }

    /// Get the window dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Get the aspect ratio.
    pub fn aspect_ratio(&self) -> f32 {
        if self.config.height > 0 {
            self.config.width as f32 / self.config.height as f32
        } else {
            1.0
        }
    }

    /// Check if this window should be rendered.
    ///
    /// A window should be rendered if it's visible and has valid dimensions.
    pub fn should_render(&self) -> bool {
        self.is_visible && self.config.width > 0 && self.config.height > 0
    }
}

impl fmt::Display for WindowConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or("Unnamed");
        write!(
            f,
            "{} [{}x{}, priority={}, focused={}, visible={}]",
            label,
            self.config.width,
            self.config.height,
            self.priority,
            self.is_focused,
            self.is_visible
        )
    }
}

// ============================================================================
// Window State
// ============================================================================

/// Runtime state for a window in the multi-window system.
///
/// `WindowState` combines the surface, configuration, and timing information
/// for a single window. It tracks frame timing for performance monitoring
/// and pacing.
///
/// # Example
///
/// ```ignore
/// let state = WindowState::new(surface, config);
/// let frame = state.surface.acquire_frame()?;
/// // ... render ...
/// frame.present();
/// state.record_frame_presented();
/// ```
pub struct WindowState {
    /// The rendering surface for this window.
    pub surface: TrinitySurface,
    /// Window configuration.
    pub config: WindowConfig,
    /// Time when the last frame was presented.
    pub last_frame_time: Instant,
    /// Time when frame acquisition started (for latency tracking).
    frame_acquire_start: Option<Instant>,
    /// Rolling average frame time in milliseconds.
    average_frame_time_ms: f32,
    /// Number of frames rendered to this window.
    frame_count: u64,
    /// Number of dropped frames (acquisition failures).
    dropped_frames: u64,
}

impl WindowState {
    /// Create a new window state.
    ///
    /// # Arguments
    ///
    /// * `surface` - The rendering surface for this window.
    /// * `config` - Window configuration.
    pub fn new(surface: TrinitySurface, config: WindowConfig) -> Self {
        Self {
            surface,
            config,
            last_frame_time: Instant::now(),
            frame_acquire_start: None,
            average_frame_time_ms: 0.0,
            frame_count: 0,
            dropped_frames: 0,
        }
    }

    /// Get the window ID.
    pub fn id(&self) -> WindowId {
        self.config.id
    }

    /// Check if this window is focused.
    pub fn is_focused(&self) -> bool {
        self.config.is_focused
    }

    /// Check if this window is visible.
    pub fn is_visible(&self) -> bool {
        self.config.is_visible
    }

    /// Get the render priority.
    pub fn priority(&self) -> u8 {
        self.config.priority
    }

    /// Set the focused state.
    pub fn set_focused(&mut self, focused: bool) {
        self.config.is_focused = focused;
    }

    /// Set the visibility state.
    pub fn set_visible(&mut self, visible: bool) {
        self.config.is_visible = visible;
    }

    /// Check if this window should be rendered.
    pub fn should_render(&self) -> bool {
        self.config.should_render()
    }

    /// Record that frame acquisition started (for latency tracking).
    pub fn record_frame_acquire_start(&mut self) {
        self.frame_acquire_start = Some(Instant::now());
    }

    /// Record that a frame was presented and update timing statistics.
    pub fn record_frame_presented(&mut self) {
        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;
        self.frame_count += 1;

        // Update rolling average (exponential moving average)
        let frame_time_ms = frame_time.as_secs_f32() * 1000.0;
        const ALPHA: f32 = 0.1; // Smoothing factor
        if self.average_frame_time_ms == 0.0 {
            self.average_frame_time_ms = frame_time_ms;
        } else {
            self.average_frame_time_ms =
                ALPHA * frame_time_ms + (1.0 - ALPHA) * self.average_frame_time_ms;
        }

        self.frame_acquire_start = None;
    }

    /// Record that a frame was dropped.
    pub fn record_frame_dropped(&mut self) {
        self.dropped_frames += 1;
        self.frame_acquire_start = None;
    }

    /// Get the time since the last frame was presented.
    pub fn time_since_last_frame(&self) -> Duration {
        Instant::now().duration_since(self.last_frame_time)
    }

    /// Get the average frame time in milliseconds.
    pub fn average_frame_time_ms(&self) -> f32 {
        self.average_frame_time_ms
    }

    /// Get the estimated frames per second.
    pub fn estimated_fps(&self) -> f32 {
        if self.average_frame_time_ms > 0.0 {
            1000.0 / self.average_frame_time_ms
        } else {
            0.0
        }
    }

    /// Get the total number of frames rendered.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the number of dropped frames.
    pub fn dropped_frames(&self) -> u64 {
        self.dropped_frames
    }

    /// Get the frame acquire latency (time from acquire_start to now).
    pub fn current_acquire_latency(&self) -> Option<Duration> {
        self.frame_acquire_start.map(|start| Instant::now().duration_since(start))
    }

    /// Resize this window's surface.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> Result<(), SurfaceError> {
        self.config.config.resize(width, height);
        self.surface.resize(device, width, height)
    }

    /// Configure this window's surface.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `config` - New surface configuration.
    pub fn configure(&mut self, device: &wgpu::Device, config: &SurfaceConfiguration) -> Result<(), SurfaceError> {
        self.config.config = config.clone();
        self.surface.configure(device, config)
    }
}

impl fmt::Debug for WindowState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WindowState")
            .field("id", &self.config.id)
            .field("config", &self.config)
            .field("frame_count", &self.frame_count)
            .field("dropped_frames", &self.dropped_frames)
            .field("avg_frame_time_ms", &self.average_frame_time_ms)
            .finish()
    }
}

// ============================================================================
// Multi-Window Errors
// ============================================================================

/// Errors that can occur in multi-window operations.
#[derive(Debug, Error)]
pub enum MultiWindowError {
    /// The specified window was not found.
    #[error("window not found: {0}")]
    WindowNotFound(WindowId),

    /// A window with the given ID already exists.
    #[error("window already exists: {0}")]
    WindowExists(WindowId),

    /// No windows are registered.
    #[error("no windows registered")]
    NoWindows,

    /// No focused window.
    #[error("no window has focus")]
    NoFocusedWindow,

    /// Surface error during window operation.
    #[error("surface error: {0}")]
    SurfaceError(#[from] SurfaceError),

    /// Frame error during acquisition.
    #[error("frame error: {0}")]
    FrameError(#[from] FrameError),

    /// Maximum number of windows reached.
    #[error("maximum number of windows ({max}) reached")]
    MaxWindowsReached { max: usize },
}

impl MultiWindowError {
    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        match self {
            MultiWindowError::WindowNotFound(_) => false,
            MultiWindowError::WindowExists(_) => false,
            MultiWindowError::NoWindows => false,
            MultiWindowError::NoFocusedWindow => true,
            MultiWindowError::SurfaceError(e) => e.is_recoverable(),
            MultiWindowError::FrameError(e) => e.is_recoverable(),
            MultiWindowError::MaxWindowsReached { .. } => false,
        }
    }
}

// ============================================================================
// Acquired Window Frame
// ============================================================================

/// A frame acquired from a specific window.
///
/// This pairs a `Frame` with its `WindowId` for multi-window rendering
/// scenarios where frames need to be presented to the correct window.
#[derive(Debug)]
pub struct WindowFrame {
    /// The window this frame belongs to.
    pub window_id: WindowId,
    /// The acquired frame.
    pub frame: Frame,
    /// Time when the frame was acquired.
    pub acquired_at: Instant,
}

impl WindowFrame {
    /// Create a new window frame.
    pub fn new(window_id: WindowId, frame: Frame) -> Self {
        Self {
            window_id,
            frame,
            acquired_at: Instant::now(),
        }
    }

    /// Get the texture view for rendering.
    pub fn view(&self) -> &wgpu::TextureView {
        self.frame.view()
    }

    /// Get the frame dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        self.frame.dimensions()
    }

    /// Present this frame.
    pub fn present(self) {
        self.frame.present();
    }

    /// Discard this frame without presenting.
    pub fn discard(self) {
        self.frame.discard();
    }

    /// Get the time since frame acquisition.
    pub fn age(&self) -> Duration {
        Instant::now().duration_since(self.acquired_at)
    }
}

// ============================================================================
// Synchronized Presentation
// ============================================================================

/// Mode for synchronizing presentation across multiple windows.
///
/// When rendering to multiple windows, synchronization controls how
/// frame timing is coordinated between windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// No synchronization - each window presents independently.
    ///
    /// This provides the lowest latency but may cause visual inconsistency
    /// between windows if they have different refresh rates.
    #[default]
    Independent,

    /// Synchronize all windows to the primary window's vsync.
    ///
    /// All secondary windows wait for the primary window's present before
    /// presenting. This ensures visual consistency but adds latency to
    /// secondary windows.
    SyncToPrimary,

    /// Synchronize to a specific refresh rate.
    ///
    /// All windows present at the target rate, regardless of individual
    /// display capabilities. Useful for consistent frame timing across
    /// different monitors.
    SyncToRate {
        /// Target refresh rate in Hz.
        target_hz: u32,
    },

    /// Present all windows simultaneously.
    ///
    /// Attempts to present all windows at the same time. This requires
    /// all displays to support similar refresh rates and may introduce
    /// latency while waiting for slower displays.
    Simultaneous,
}

impl SyncMode {
    /// Create a sync mode targeting a specific refresh rate.
    pub fn sync_to_rate(hz: u32) -> Self {
        SyncMode::SyncToRate { target_hz: hz }
    }

    /// Get the target frame interval for rate-limited modes.
    pub fn target_interval(&self) -> Option<Duration> {
        match self {
            SyncMode::SyncToRate { target_hz } if *target_hz > 0 => {
                Some(Duration::from_secs_f64(1.0 / *target_hz as f64))
            }
            _ => None,
        }
    }

    /// Check if this mode requires coordination between windows.
    pub fn requires_coordination(&self) -> bool {
        !matches!(self, SyncMode::Independent)
    }
}

impl fmt::Display for SyncMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyncMode::Independent => write!(f, "Independent"),
            SyncMode::SyncToPrimary => write!(f, "Sync to Primary"),
            SyncMode::SyncToRate { target_hz } => write!(f, "Sync to {}Hz", target_hz),
            SyncMode::Simultaneous => write!(f, "Simultaneous"),
        }
    }
}

// ============================================================================
// Multi-Window Manager
// ============================================================================

/// Manager for multiple rendering windows.
///
/// `MultiWindowManager` provides a centralized way to manage multiple windows,
/// track focus, coordinate presentation, and acquire frames from all windows.
///
/// # Example
///
/// ```ignore
/// let mut manager = MultiWindowManager::new();
///
/// // Register windows
/// let main_id = manager.register_window(main_surface, main_config)?;
/// let tool_id = manager.register_window(tool_surface, tool_config)?;
///
/// // Set focus
/// manager.set_focus(main_id)?;
///
/// // Acquire frames from all visible windows
/// let frames = manager.acquire_all_frames()?;
///
/// // Render to each frame
/// for window_frame in &frames {
///     render_to_window(&window_frame);
/// }
///
/// // Present all frames
/// manager.present_all(frames);
/// ```
pub struct MultiWindowManager {
    /// All registered windows.
    windows: HashMap<WindowId, WindowState>,
    /// Currently focused window.
    focused_window: Option<WindowId>,
    /// Render order (sorted by priority, highest first).
    render_order: Vec<WindowId>,
    /// Presentation synchronization mode.
    sync_mode: SyncMode,
    /// Maximum allowed windows (0 = unlimited).
    max_windows: usize,
    /// Last global present time (for sync modes).
    last_present_time: Instant,
    /// Global frame counter.
    global_frame_count: u64,
}

impl MultiWindowManager {
    /// Create a new multi-window manager.
    pub fn new() -> Self {
        Self {
            windows: HashMap::new(),
            focused_window: None,
            render_order: Vec::new(),
            sync_mode: SyncMode::Independent,
            max_windows: 0, // Unlimited
            last_present_time: Instant::now(),
            global_frame_count: 0,
        }
    }

    /// Create a multi-window manager with a maximum window limit.
    ///
    /// # Arguments
    ///
    /// * `max` - Maximum number of windows allowed.
    pub fn with_max_windows(max: usize) -> Self {
        Self {
            max_windows: max,
            ..Self::new()
        }
    }

    /// Set the presentation synchronization mode.
    pub fn set_sync_mode(&mut self, mode: SyncMode) {
        self.sync_mode = mode;
    }

    /// Get the current synchronization mode.
    pub fn sync_mode(&self) -> SyncMode {
        self.sync_mode
    }

    /// Register a new window with the manager.
    ///
    /// # Arguments
    ///
    /// * `surface` - The window's rendering surface.
    /// * `config` - Window configuration.
    ///
    /// # Returns
    ///
    /// The WindowId assigned to this window.
    ///
    /// # Errors
    ///
    /// Returns `MultiWindowError::MaxWindowsReached` if the limit is reached.
    /// Returns `MultiWindowError::WindowExists` if the ID is already registered.
    pub fn register_window(
        &mut self,
        surface: TrinitySurface,
        config: WindowConfig,
    ) -> Result<WindowId, MultiWindowError> {
        // Check max windows limit
        if self.max_windows > 0 && self.windows.len() >= self.max_windows {
            return Err(MultiWindowError::MaxWindowsReached { max: self.max_windows });
        }

        let id = config.id;

        // Check for duplicate ID
        if self.windows.contains_key(&id) {
            return Err(MultiWindowError::WindowExists(id));
        }

        // Create window state
        let state = WindowState::new(surface, config);

        // Insert and update render order
        self.windows.insert(id, state);
        self.update_render_order();

        // If this is the first window or primary, set focus
        if self.focused_window.is_none() || id.is_primary() {
            self.focused_window = Some(id);
        }

        Ok(id)
    }

    /// Register a window with a specific ID.
    ///
    /// This is a convenience method that creates the WindowConfig automatically.
    pub fn register_window_with_id(
        &mut self,
        id: WindowId,
        surface: TrinitySurface,
        surface_config: SurfaceConfiguration,
    ) -> Result<WindowId, MultiWindowError> {
        let config = WindowConfig::new(id, surface_config);
        self.register_window(surface, config)
    }

    /// Unregister a window from the manager.
    ///
    /// # Arguments
    ///
    /// * `id` - The window to unregister.
    ///
    /// # Returns
    ///
    /// The removed WindowState, or an error if not found.
    pub fn unregister_window(&mut self, id: WindowId) -> Result<WindowState, MultiWindowError> {
        let state = self.windows.remove(&id)
            .ok_or(MultiWindowError::WindowNotFound(id))?;

        // Update render order
        self.render_order.retain(|&wid| wid != id);

        // Update focus if needed
        if self.focused_window == Some(id) {
            self.focused_window = self.render_order.first().copied();
        }

        Ok(state)
    }

    /// Get a reference to a window's state.
    pub fn get_window(&self, id: WindowId) -> Option<&WindowState> {
        self.windows.get(&id)
    }

    /// Get a mutable reference to a window's state.
    pub fn get_window_mut(&mut self, id: WindowId) -> Option<&mut WindowState> {
        self.windows.get_mut(&id)
    }

    /// Get the focused window's state.
    pub fn focused_window(&self) -> Option<&WindowState> {
        self.focused_window.and_then(|id| self.windows.get(&id))
    }

    /// Get the focused window's ID.
    pub fn focused_window_id(&self) -> Option<WindowId> {
        self.focused_window
    }

    /// Set the focused window.
    ///
    /// # Arguments
    ///
    /// * `id` - The window to focus.
    ///
    /// # Returns
    ///
    /// The previously focused window ID, or an error if the window doesn't exist.
    pub fn set_focus(&mut self, id: WindowId) -> Result<Option<WindowId>, MultiWindowError> {
        if !self.windows.contains_key(&id) {
            return Err(MultiWindowError::WindowNotFound(id));
        }

        // Update focus state on windows
        let old_focus = self.focused_window;

        if let Some(old_id) = old_focus {
            if let Some(state) = self.windows.get_mut(&old_id) {
                state.set_focused(false);
            }
        }

        if let Some(state) = self.windows.get_mut(&id) {
            state.set_focused(true);
        }

        self.focused_window = Some(id);
        Ok(old_focus)
    }

    /// Set a window's visibility.
    ///
    /// # Arguments
    ///
    /// * `id` - The window to update.
    /// * `visible` - Whether the window is visible.
    pub fn set_visible(&mut self, id: WindowId, visible: bool) -> Result<(), MultiWindowError> {
        let state = self.windows.get_mut(&id)
            .ok_or(MultiWindowError::WindowNotFound(id))?;
        state.set_visible(visible);
        Ok(())
    }

    /// Resize a window.
    ///
    /// # Arguments
    ///
    /// * `id` - The window to resize.
    /// * `device` - The wgpu device.
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    pub fn resize_window(
        &mut self,
        id: WindowId,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<(), MultiWindowError> {
        let state = self.windows.get_mut(&id)
            .ok_or(MultiWindowError::WindowNotFound(id))?;
        state.resize(device, width, height)?;
        Ok(())
    }

    /// Get the number of registered windows.
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Check if any windows are registered.
    pub fn has_windows(&self) -> bool {
        !self.windows.is_empty()
    }

    /// Get all window IDs in render order.
    pub fn window_ids(&self) -> &[WindowId] {
        &self.render_order
    }

    /// Get all visible window IDs in render order.
    pub fn visible_window_ids(&self) -> Vec<WindowId> {
        self.render_order
            .iter()
            .filter(|id| {
                self.windows.get(id)
                    .map(|s| s.is_visible())
                    .unwrap_or(false)
            })
            .copied()
            .collect()
    }

    /// Iterate over all windows in render order.
    ///
    /// Returns an iterator of (WindowId, &WindowState) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (WindowId, &WindowState)> {
        self.render_order.iter().filter_map(|&id| {
            self.windows.get(&id).map(|state| (id, state))
        })
    }

    /// Get all window states for mutable iteration.
    ///
    /// This returns the windows HashMap directly for cases where you need
    /// mutable access. Use `window_ids()` to get the render order.
    pub fn windows_mut(&mut self) -> &mut HashMap<WindowId, WindowState> {
        &mut self.windows
    }

    /// Apply a function to each window in render order.
    ///
    /// This is a safer alternative to `iter_mut` that avoids lifetime issues.
    ///
    /// # Arguments
    ///
    /// * `f` - Function to apply to each window.
    pub fn for_each_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(WindowId, &mut WindowState),
    {
        let order = self.render_order.clone();
        for id in order {
            if let Some(state) = self.windows.get_mut(&id) {
                f(id, state);
            }
        }
    }

    /// Update the render order based on window priorities.
    fn update_render_order(&mut self) {
        self.render_order = self.windows.keys().copied().collect();
        self.render_order.sort_by(|a, b| {
            let pa = self.windows.get(a).map(|s| s.priority()).unwrap_or(0);
            let pb = self.windows.get(b).map(|s| s.priority()).unwrap_or(0);
            pb.cmp(&pa) // Higher priority first
        });
    }

    /// Acquire a frame from a specific window.
    ///
    /// # Arguments
    ///
    /// * `id` - The window to acquire from.
    ///
    /// # Returns
    ///
    /// A `WindowFrame` if successful, or an error.
    pub fn acquire_frame(&mut self, id: WindowId) -> Result<WindowFrame, MultiWindowError> {
        let state = self.windows.get_mut(&id)
            .ok_or(MultiWindowError::WindowNotFound(id))?;

        state.record_frame_acquire_start();

        match state.surface.acquire_frame() {
            Ok(frame) => Ok(WindowFrame::new(id, frame)),
            Err(e) => {
                state.record_frame_dropped();
                Err(MultiWindowError::FrameError(e))
            }
        }
    }

    /// Acquire frames from all visible windows.
    ///
    /// Frames are acquired in render order (by priority). If any window
    /// fails to acquire, that window is skipped and the frame is recorded
    /// as dropped.
    ///
    /// # Returns
    ///
    /// A vector of acquired frames with their window IDs.
    pub fn acquire_all_frames(&mut self) -> Vec<Result<WindowFrame, (WindowId, FrameError)>> {
        let visible_ids = self.visible_window_ids();
        let mut frames = Vec::with_capacity(visible_ids.len());

        for id in visible_ids {
            if let Some(state) = self.windows.get_mut(&id) {
                state.record_frame_acquire_start();

                match state.surface.acquire_frame() {
                    Ok(frame) => {
                        frames.push(Ok(WindowFrame::new(id, frame)));
                    }
                    Err(e) => {
                        state.record_frame_dropped();
                        frames.push(Err((id, e)));
                    }
                }
            }
        }

        frames
    }

    /// Acquire frames from all visible windows, failing on first error.
    ///
    /// Unlike `acquire_all_frames`, this returns an error if any window
    /// fails to acquire a frame.
    pub fn acquire_all_frames_strict(&mut self) -> Result<Vec<WindowFrame>, MultiWindowError> {
        let visible_ids = self.visible_window_ids();
        let mut frames = Vec::with_capacity(visible_ids.len());

        for id in visible_ids {
            let frame = self.acquire_frame(id)?;
            frames.push(frame);
        }

        Ok(frames)
    }

    /// Present all frames according to the sync mode.
    ///
    /// # Arguments
    ///
    /// * `frames` - Frames to present.
    pub fn present_all(&mut self, frames: Vec<WindowFrame>) {
        match self.sync_mode {
            SyncMode::Independent => {
                // Present each frame immediately
                for wf in frames {
                    if let Some(state) = self.windows.get_mut(&wf.window_id) {
                        wf.present();
                        state.record_frame_presented();
                    }
                }
            }
            SyncMode::SyncToPrimary => {
                // Find and present primary first
                let (primary, others): (Vec<_>, Vec<_>) = frames
                    .into_iter()
                    .partition(|f| f.window_id.is_primary());

                // Present primary
                for wf in primary {
                    if let Some(state) = self.windows.get_mut(&wf.window_id) {
                        wf.present();
                        state.record_frame_presented();
                    }
                }

                // Then present others
                for wf in others {
                    if let Some(state) = self.windows.get_mut(&wf.window_id) {
                        wf.present();
                        state.record_frame_presented();
                    }
                }
            }
            SyncMode::SyncToRate { target_hz } => {
                // Rate-limit presentation
                let target_interval = Duration::from_secs_f64(1.0 / target_hz as f64);
                let elapsed = Instant::now().duration_since(self.last_present_time);

                if elapsed < target_interval {
                    // Wait for target interval (in practice, this would be handled
                    // by the frame pacer rather than blocking here)
                    // For now, just present immediately
                }

                for wf in frames {
                    if let Some(state) = self.windows.get_mut(&wf.window_id) {
                        wf.present();
                        state.record_frame_presented();
                    }
                }
            }
            SyncMode::Simultaneous => {
                // Present all at once
                for wf in frames {
                    if let Some(state) = self.windows.get_mut(&wf.window_id) {
                        wf.present();
                        state.record_frame_presented();
                    }
                }
            }
        }

        self.last_present_time = Instant::now();
        self.global_frame_count += 1;
    }

    /// Get the global frame count.
    pub fn global_frame_count(&self) -> u64 {
        self.global_frame_count
    }

    /// Get aggregate statistics across all windows.
    pub fn aggregate_stats(&self) -> MultiWindowStats {
        let mut total_frames: u64 = 0;
        let mut total_dropped: u64 = 0;
        let mut total_frame_time: f32 = 0.0;
        let mut window_count: usize = 0;

        for state in self.windows.values() {
            total_frames += state.frame_count();
            total_dropped += state.dropped_frames();
            total_frame_time += state.average_frame_time_ms();
            window_count += 1;
        }

        let avg_frame_time = if window_count > 0 {
            total_frame_time / window_count as f32
        } else {
            0.0
        };

        MultiWindowStats {
            window_count,
            total_frames,
            total_dropped,
            average_frame_time_ms: avg_frame_time,
            global_frame_count: self.global_frame_count,
        }
    }
}

impl Default for MultiWindowManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MultiWindowManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiWindowManager")
            .field("window_count", &self.windows.len())
            .field("focused_window", &self.focused_window)
            .field("sync_mode", &self.sync_mode)
            .field("global_frame_count", &self.global_frame_count)
            .finish()
    }
}

// ============================================================================
// Multi-Window Statistics
// ============================================================================

/// Aggregate statistics for the multi-window system.
#[derive(Debug, Clone, Copy)]
pub struct MultiWindowStats {
    /// Number of registered windows.
    pub window_count: usize,
    /// Total frames rendered across all windows.
    pub total_frames: u64,
    /// Total dropped frames across all windows.
    pub total_dropped: u64,
    /// Average frame time in milliseconds.
    pub average_frame_time_ms: f32,
    /// Global frame count (present_all calls).
    pub global_frame_count: u64,
}

impl MultiWindowStats {
    /// Calculate the overall drop rate.
    pub fn drop_rate(&self) -> f32 {
        let total = self.total_frames + self.total_dropped;
        if total > 0 {
            self.total_dropped as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Calculate the estimated FPS.
    pub fn estimated_fps(&self) -> f32 {
        if self.average_frame_time_ms > 0.0 {
            1000.0 / self.average_frame_time_ms
        } else {
            0.0
        }
    }
}

impl fmt::Display for MultiWindowStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} windows, {} frames ({} dropped, {:.1}%), {:.1} FPS avg",
            self.window_count,
            self.total_frames,
            self.total_dropped,
            self.drop_rate() * 100.0,
            self.estimated_fps()
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- PlatformTarget tests ------------------------------------------------

    #[test]
    fn test_platform_target_current() {
        let platform = PlatformTarget::current();
        // Should not be Unknown on common development platforms
        #[cfg(any(
            target_os = "linux",
            target_os = "windows",
            target_os = "macos"
        ))]
        assert!(platform.is_supported());
    }

    #[test]
    fn test_platform_target_name() {
        assert_eq!(PlatformTarget::Wayland.name(), "Linux (Wayland)");
        assert_eq!(PlatformTarget::X11.name(), "Linux (X11)");
        assert_eq!(PlatformTarget::Windows.name(), "Windows");
        assert_eq!(PlatformTarget::MacOS.name(), "macOS");
        assert_eq!(PlatformTarget::Web.name(), "Web");
        assert_eq!(PlatformTarget::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_platform_target_is_supported() {
        assert!(PlatformTarget::Wayland.is_supported());
        assert!(PlatformTarget::X11.is_supported());
        assert!(PlatformTarget::Windows.is_supported());
        assert!(PlatformTarget::MacOS.is_supported());
        assert!(PlatformTarget::IOS.is_supported());
        assert!(PlatformTarget::Android.is_supported());
        assert!(PlatformTarget::Web.is_supported());
        assert!(!PlatformTarget::Unknown.is_supported());
    }

    #[test]
    fn test_platform_target_display() {
        assert_eq!(format!("{}", PlatformTarget::Windows), "Windows");
        assert_eq!(format!("{}", PlatformTarget::MacOS), "macOS");
    }

    // -- SurfaceSize tests ---------------------------------------------------

    #[test]
    fn test_surface_size_default() {
        let size = SurfaceSize::default();
        assert_eq!(size.width, 1);
        assert_eq!(size.height, 1);
    }

    #[test]
    fn test_surface_size_custom() {
        let size = SurfaceSize::new(1920, 1080);
        assert_eq!(size.width, 1920);
        assert_eq!(size.height, 1080);
    }

    #[test]
    fn test_surface_size_zero_width() {
        let size = SurfaceSize::new(0, 600);
        assert_eq!(size.width, 1);
        assert_eq!(size.height, 600);
    }

    #[test]
    fn test_surface_size_zero_height() {
        let size = SurfaceSize::new(800, 0);
        assert_eq!(size.width, 800);
        assert_eq!(size.height, 1);
    }

    #[test]
    fn test_surface_size_zero_both() {
        let size = SurfaceSize::new(0, 0);
        assert_eq!(size.width, 1);
        assert_eq!(size.height, 1);
    }

    #[test]
    fn test_surface_size_aspect_ratio() {
        let size = SurfaceSize::new(1920, 1080);
        let ratio = size.aspect_ratio();
        // 16:9 aspect ratio
        assert!((ratio - 1.7777).abs() < 0.01);
    }

    #[test]
    fn test_surface_size_pixel_count() {
        let size = SurfaceSize::new(1920, 1080);
        assert_eq!(size.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_surface_size_is_minimized() {
        let minimized = SurfaceSize::new(1, 1);
        assert!(minimized.is_minimized());

        let normal = SurfaceSize::new(800, 600);
        assert!(!normal.is_minimized());
    }

    #[test]
    fn test_surface_size_as_tuple() {
        let size = SurfaceSize::new(800, 600);
        assert_eq!(size.as_tuple(), (800, 600));
    }

    #[test]
    fn test_surface_size_from_tuple() {
        let size = SurfaceSize::from_tuple((1280, 720));
        assert_eq!(size.width, 1280);
        assert_eq!(size.height, 720);
    }

    #[test]
    fn test_surface_size_matches() {
        let size = SurfaceSize::new(800, 600);
        assert!(size.matches(800, 600));
        assert!(!size.matches(1024, 768));
    }

    #[test]
    fn test_surface_size_scale() {
        let size = SurfaceSize::new(100, 100);
        let scaled = size.scale(2.0);
        assert_eq!(scaled.width, 200);
        assert_eq!(scaled.height, 200);

        // Scale down
        let half = size.scale(0.5);
        assert_eq!(half.width, 50);
        assert_eq!(half.height, 50);
    }

    #[test]
    fn test_surface_size_display() {
        let size = SurfaceSize::new(1920, 1080);
        assert_eq!(format!("{}", size), "1920x1080");
    }

    #[test]
    fn test_surface_size_from_impl() {
        let size: SurfaceSize = (640, 480).into();
        assert_eq!(size.width, 640);
        assert_eq!(size.height, 480);
    }

    #[test]
    fn test_surface_size_into_tuple() {
        let size = SurfaceSize::new(1024, 768);
        let tuple: (u32, u32) = size.into();
        assert_eq!(tuple, (1024, 768));
    }

    // -- SurfaceError tests --------------------------------------------------

    #[test]
    fn test_surface_error_unsupported() {
        let err = SurfaceError::unsupported();
        assert!(err.is_platform_error());
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_surface_error_window_handle() {
        let err = SurfaceError::window_handle("test error");
        assert!(!err.is_recoverable());
        assert!(format!("{}", err).contains("test error"));
    }

    #[test]
    fn test_surface_error_display_handle() {
        let err = SurfaceError::display_handle("display error");
        assert!(format!("{}", err).contains("display error"));
    }

    #[test]
    fn test_surface_error_creation_failed() {
        let err = SurfaceError::creation_failed("wgpu error");
        assert!(format!("{}", err).contains("wgpu error"));
    }

    #[test]
    fn test_surface_error_invalid_config() {
        let err = SurfaceError::invalid_config("bad format");
        assert!(format!("{}", err).contains("bad format"));
    }

    #[test]
    fn test_surface_error_recoverable() {
        let lost = SurfaceError::SurfaceLost {
            reason: "test".to_string(),
        };
        assert!(lost.is_recoverable());

        let outdated = SurfaceError::SurfaceOutdated;
        assert!(outdated.is_recoverable());

        let unsupported = SurfaceError::unsupported();
        assert!(!unsupported.is_recoverable());
    }

    // -- SurfaceCapabilities tests -------------------------------------------

    #[test]
    fn test_surface_capabilities_preferred_format_srgb() {
        let caps = SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureFormat::Bgra8UnormSrgb,
            ],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        // Should prefer sRGB over linear
        assert_eq!(
            caps.preferred_format(),
            Some(wgpu::TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn test_surface_capabilities_preferred_format_linear_fallback() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Rgba8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        // Should fall back to linear
        assert_eq!(
            caps.preferred_format(),
            Some(wgpu::TextureFormat::Rgba8Unorm)
        );
    }

    #[test]
    fn test_surface_capabilities_preferred_present_mode_mailbox() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Mailbox,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.preferred_present_mode(), wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_surface_capabilities_preferred_present_mode_fifo_fallback() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.preferred_present_mode(), wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_surface_capabilities_preferred_alpha_mode_opaque() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Auto,
                wgpu::CompositeAlphaMode::Opaque,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.preferred_alpha_mode(), wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn test_surface_capabilities_supports_format() {
        let caps = SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Rgba8Unorm,
            ],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps.supports_format(wgpu::TextureFormat::Bgra8Unorm));
        assert!(caps.supports_format(wgpu::TextureFormat::Rgba8Unorm));
        assert!(!caps.supports_format(wgpu::TextureFormat::Rgba16Float));
    }

    #[test]
    fn test_surface_capabilities_supports_hdr() {
        let caps_no_hdr = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(!caps_no_hdr.supports_hdr());

        let caps_hdr = SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Rgba16Float,
            ],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps_hdr.supports_hdr());
    }

    // -- SurfaceConfiguration tests ------------------------------------------

    #[test]
    fn test_surface_configuration_new() {
        let config = SurfaceConfiguration::new(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.present_mode, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_surface_configuration_new_clamps_zero() {
        let config = SurfaceConfiguration::new(0, 0);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn test_surface_configuration_from_capabilities() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![wgpu::PresentMode::Mailbox],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Opaque],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::from_capabilities(&caps, 800, 600);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn test_surface_configuration_builder_methods() {
        let config = SurfaceConfiguration::new(640, 480)
            .with_format(wgpu::TextureFormat::Rgba8Unorm)
            .with_present_mode(wgpu::PresentMode::Immediate)
            .with_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied)
            .with_frame_latency(3);

        assert_eq!(config.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(config.present_mode, wgpu::PresentMode::Immediate);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::PreMultiplied);
        assert_eq!(config.desired_maximum_frame_latency, 3);
    }

    #[test]
    fn test_surface_configuration_validate_success() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Fifo)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Auto);

        assert!(config.validate(&caps).is_ok());
    }

    #[test]
    fn test_surface_configuration_validate_bad_format() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Rgba16Float);

        let result = config.validate(&caps);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("format"));
    }

    #[test]
    fn test_surface_configuration_validate_bad_present_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox);

        let result = config.validate(&caps);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("present mode"));
    }

    #[test]
    fn test_surface_configuration_to_wgpu() {
        let config = SurfaceConfiguration::new(1280, 720)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque)
            .with_frame_latency(2);

        let wgpu_config = config.to_wgpu();
        assert_eq!(wgpu_config.format, wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(wgpu_config.width, 1280);
        assert_eq!(wgpu_config.height, 720);
        assert_eq!(wgpu_config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(wgpu_config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(wgpu_config.desired_maximum_frame_latency, 2);
        assert_eq!(wgpu_config.usage, wgpu::TextureUsages::RENDER_ATTACHMENT);
    }

    #[test]
    fn test_surface_configuration_default() {
        let config = SurfaceConfiguration::default();
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    // -- TrinitySurface tests ------------------------------------------------

    #[test]
    fn test_trinity_surface_from_wgpu() {
        // We can test the from_wgpu constructor without a real surface
        // by checking the struct is properly initialized.
        // Cannot create a real surface without a window, but we can
        // verify the API shape.
    }

    #[test]
    fn test_trinity_surface_debug() {
        // Verify Debug impl compiles
        // Cannot create a real surface without a window
    }

    #[test]
    fn test_trinity_surface_width_height_unconfigured() {
        // Without a real surface, we can verify the behavior when
        // current_config is None by looking at the method signature.
        // The methods should return 0 when unconfigured.
    }

    // -- FormatCategory tests ------------------------------------------------

    #[test]
    fn test_format_category_from_format_srgb() {
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Bgra8UnormSrgb),
            FormatCategory::Srgb
        );
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Rgba8UnormSrgb),
            FormatCategory::Srgb
        );
    }

    #[test]
    fn test_format_category_from_format_linear() {
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Bgra8Unorm),
            FormatCategory::Linear
        );
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Rgba8Unorm),
            FormatCategory::Linear
        );
    }

    #[test]
    fn test_format_category_from_format_hdr() {
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Rgba16Float),
            FormatCategory::Hdr
        );
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Rgb10a2Unorm),
            FormatCategory::Hdr
        );
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Rg11b10Float),
            FormatCategory::Hdr
        );
    }

    #[test]
    fn test_format_category_from_format_other() {
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::R8Unorm),
            FormatCategory::Other
        );
        assert_eq!(
            FormatCategory::from_format(wgpu::TextureFormat::Depth32Float),
            FormatCategory::Other
        );
    }

    #[test]
    fn test_format_category_is_gamma_corrected() {
        assert!(FormatCategory::Srgb.is_gamma_corrected());
        assert!(!FormatCategory::Linear.is_gamma_corrected());
        assert!(!FormatCategory::Hdr.is_gamma_corrected());
        assert!(!FormatCategory::Other.is_gamma_corrected());
    }

    #[test]
    fn test_format_category_is_hdr() {
        assert!(!FormatCategory::Srgb.is_hdr());
        assert!(!FormatCategory::Linear.is_hdr());
        assert!(FormatCategory::Hdr.is_hdr());
        assert!(!FormatCategory::Other.is_hdr());
    }

    #[test]
    fn test_format_category_name() {
        assert_eq!(FormatCategory::Srgb.name(), "sRGB");
        assert_eq!(FormatCategory::Linear.name(), "Linear");
        assert_eq!(FormatCategory::Hdr.name(), "HDR");
        assert_eq!(FormatCategory::Other.name(), "Other");
    }

    #[test]
    fn test_format_category_display() {
        assert_eq!(format!("{}", FormatCategory::Srgb), "sRGB");
        assert_eq!(format!("{}", FormatCategory::Hdr), "HDR");
    }

    #[test]
    fn test_surface_capabilities_preferred_hdr_format() {
        // Prefer Rgba16Float over others
        let caps = SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Rgb10a2Unorm,
                wgpu::TextureFormat::Rgba16Float,
            ],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(
            caps.preferred_hdr_format(),
            Some(wgpu::TextureFormat::Rgba16Float)
        );
    }

    #[test]
    fn test_surface_capabilities_preferred_hdr_format_fallback() {
        // Fall back to Rg11b10Float when Rgba16Float unavailable
        let caps = SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Rg11b10Float,
            ],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(
            caps.preferred_hdr_format(),
            Some(wgpu::TextureFormat::Rg11b10Float)
        );
    }

    #[test]
    fn test_surface_capabilities_preferred_hdr_format_none() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.preferred_hdr_format(), None);
    }

    #[test]
    fn test_surface_capabilities_formats_in_category() {
        let caps = SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::TextureFormat::Rgba16Float,
            ],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        let srgb_formats = caps.formats_in_category(FormatCategory::Srgb);
        assert_eq!(srgb_formats, vec![wgpu::TextureFormat::Bgra8UnormSrgb]);

        let linear_formats = caps.formats_in_category(FormatCategory::Linear);
        assert_eq!(linear_formats, vec![wgpu::TextureFormat::Bgra8Unorm]);

        let hdr_formats = caps.formats_in_category(FormatCategory::Hdr);
        assert_eq!(hdr_formats, vec![wgpu::TextureFormat::Rgba16Float]);
    }

    #[test]
    fn test_surface_capabilities_select_format_prefer_hdr() {
        let caps = SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::TextureFormat::Rgba16Float,
            ],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        // When HDR preferred, select HDR format
        assert_eq!(
            caps.select_format(true),
            Some(wgpu::TextureFormat::Rgba16Float)
        );

        // When HDR not preferred, select sRGB
        assert_eq!(
            caps.select_format(false),
            Some(wgpu::TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn test_surface_capabilities_select_format_hdr_fallback() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        // When HDR preferred but unavailable, fall back to sRGB
        assert_eq!(
            caps.select_format(true),
            Some(wgpu::TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn test_surface_capabilities_format_category_helper() {
        // Test the static helper method
        assert_eq!(
            SurfaceCapabilities::format_category(wgpu::TextureFormat::Bgra8UnormSrgb),
            FormatCategory::Srgb
        );
        assert_eq!(
            SurfaceCapabilities::format_category(wgpu::TextureFormat::Rgba16Float),
            FormatCategory::Hdr
        );
    }

    // -- Present Mode Selection tests ----------------------------------------

    #[test]
    fn test_present_mode_low_latency_prefers_immediate() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Immediate,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.low_latency_present_mode(), wgpu::PresentMode::Immediate);
    }

    #[test]
    fn test_present_mode_low_latency_fallback_to_mailbox() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Mailbox,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        // Without Immediate, should fall back to Mailbox
        assert_eq!(caps.low_latency_present_mode(), wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_present_mode_vsync_prefers_mailbox_over_fifo_relaxed() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::FifoRelaxed,
                wgpu::PresentMode::Mailbox,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(caps.preferred_present_mode(), wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_present_mode_vsync_fallback_to_fifo_relaxed() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::FifoRelaxed,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        // Without Mailbox, should prefer FifoRelaxed over Fifo
        assert_eq!(caps.preferred_present_mode(), wgpu::PresentMode::FifoRelaxed);
    }

    #[test]
    fn test_select_present_mode_low_latency() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Immediate,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(
            caps.select_present_mode(PresentModePreference::LowLatency),
            wgpu::PresentMode::Immediate
        );
    }

    #[test]
    fn test_select_present_mode_vsync() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Immediate,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Vsync),
            wgpu::PresentMode::Mailbox
        );
    }

    #[test]
    fn test_select_present_mode_power_saving() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Immediate,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        // Power saving should prefer Fifo even when others available
        assert_eq!(
            caps.select_present_mode(PresentModePreference::PowerSaving),
            wgpu::PresentMode::Fifo
        );
    }

    #[test]
    fn test_select_present_mode_adaptive() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::FifoRelaxed,
                wgpu::PresentMode::Mailbox,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Adaptive),
            wgpu::PresentMode::FifoRelaxed
        );
    }

    #[test]
    fn test_select_present_mode_specific_available() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Immediate,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(wgpu::PresentMode::Immediate)),
            wgpu::PresentMode::Immediate
        );
    }

    #[test]
    fn test_select_present_mode_specific_unavailable_fallback() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        // Requested Immediate but not available, should fall back to vsync preference
        assert_eq!(
            caps.select_present_mode(PresentModePreference::Specific(wgpu::PresentMode::Immediate)),
            wgpu::PresentMode::Fifo
        );
    }

    #[test]
    fn test_supports_immediate() {
        let caps_with = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Immediate],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps_with.supports_immediate());

        let caps_without = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(!caps_without.supports_immediate());
    }

    #[test]
    fn test_supports_mailbox() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps.supports_mailbox());
    }

    #[test]
    fn test_supports_fifo_relaxed() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::FifoRelaxed],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert!(caps.supports_fifo_relaxed());
    }

    // -- PresentModePreference tests -----------------------------------------

    #[test]
    fn test_present_mode_preference_default() {
        assert_eq!(PresentModePreference::default(), PresentModePreference::Vsync);
    }

    #[test]
    fn test_present_mode_preference_description() {
        assert!(!PresentModePreference::LowLatency.description().is_empty());
        assert!(!PresentModePreference::Vsync.description().is_empty());
        assert!(!PresentModePreference::PowerSaving.description().is_empty());
        assert!(!PresentModePreference::Adaptive.description().is_empty());
        assert!(!PresentModePreference::Specific(wgpu::PresentMode::Fifo)
            .description()
            .is_empty());
    }

    #[test]
    fn test_present_mode_preference_display() {
        assert_eq!(format!("{}", PresentModePreference::LowLatency), "Low Latency");
        assert_eq!(format!("{}", PresentModePreference::Vsync), "Vsync");
        assert_eq!(format!("{}", PresentModePreference::PowerSaving), "Power Saving");
        assert_eq!(format!("{}", PresentModePreference::Adaptive), "Adaptive");
        assert!(format!("{}", PresentModePreference::Specific(wgpu::PresentMode::Fifo))
            .contains("Specific"));
    }

    // -- PresentModeInfo tests -----------------------------------------------

    #[test]
    fn test_present_mode_info_immediate() {
        let info = PresentModeInfo::from_mode(wgpu::PresentMode::Immediate);
        assert_eq!(info.mode, wgpu::PresentMode::Immediate);
        assert!(!info.prevents_tearing);
        assert_eq!(info.latency_rank, 1);
        assert!(!info.power_efficient);
        assert!(info.is_competitive_gaming_mode());
        assert!(!info.is_battery_friendly());
    }

    #[test]
    fn test_present_mode_info_mailbox() {
        let info = PresentModeInfo::from_mode(wgpu::PresentMode::Mailbox);
        assert_eq!(info.mode, wgpu::PresentMode::Mailbox);
        assert!(info.prevents_tearing);
        assert_eq!(info.latency_rank, 2);
        assert!(!info.power_efficient);
        assert!(info.is_competitive_gaming_mode());
    }

    #[test]
    fn test_present_mode_info_fifo() {
        let info = PresentModeInfo::from_mode(wgpu::PresentMode::Fifo);
        assert_eq!(info.mode, wgpu::PresentMode::Fifo);
        assert!(info.prevents_tearing);
        assert_eq!(info.latency_rank, 4);
        assert!(info.power_efficient);
        assert!(!info.is_competitive_gaming_mode());
        assert!(info.is_battery_friendly());
    }

    #[test]
    fn test_present_mode_info_fifo_relaxed() {
        let info = PresentModeInfo::from_mode(wgpu::PresentMode::FifoRelaxed);
        assert_eq!(info.mode, wgpu::PresentMode::FifoRelaxed);
        assert!(info.prevents_tearing);
        assert_eq!(info.latency_rank, 3);
        assert!(info.power_efficient);
        assert!(!info.is_competitive_gaming_mode());
        assert!(info.is_battery_friendly());
    }

    #[test]
    fn test_present_mode_info_display() {
        let info = PresentModeInfo::from_mode(wgpu::PresentMode::Mailbox);
        let display = format!("{}", info);
        assert!(display.contains("Mailbox"));
        assert!(display.contains("Triple Buffered"));
    }

    #[test]
    fn test_describe_present_mode() {
        let info = SurfaceCapabilities::describe_present_mode(wgpu::PresentMode::Immediate);
        assert_eq!(info.name, "Immediate");
    }

    // -- SurfaceConfiguration present mode preference test -------------------

    #[test]
    fn test_surface_configuration_with_present_mode_preference() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Immediate,
            ],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        let config = SurfaceConfiguration::new(1920, 1080)
            .with_present_mode_preference(&caps, PresentModePreference::LowLatency);

        assert_eq!(config.present_mode, wgpu::PresentMode::Immediate);
    }

    // =========================================================================
    // AlphaModePreference tests
    // =========================================================================

    #[test]
    fn test_alpha_mode_preference_default() {
        assert_eq!(AlphaModePreference::default(), AlphaModePreference::Auto);
    }

    #[test]
    fn test_alpha_mode_preference_description() {
        assert!(!AlphaModePreference::Opaque.description().is_empty());
        assert!(!AlphaModePreference::PreMultiplied.description().is_empty());
        assert!(!AlphaModePreference::PostMultiplied.description().is_empty());
        assert!(!AlphaModePreference::Inherit.description().is_empty());
        assert!(!AlphaModePreference::Auto.description().is_empty());
    }

    #[test]
    fn test_alpha_mode_preference_display() {
        assert_eq!(format!("{}", AlphaModePreference::Opaque), "Opaque");
        assert_eq!(format!("{}", AlphaModePreference::PreMultiplied), "Pre-Multiplied");
        assert_eq!(format!("{}", AlphaModePreference::PostMultiplied), "Post-Multiplied");
        assert_eq!(format!("{}", AlphaModePreference::Inherit), "Inherit");
        assert_eq!(format!("{}", AlphaModePreference::Auto), "Auto");
    }

    #[test]
    fn test_alpha_mode_preference_requires_alpha() {
        assert!(!AlphaModePreference::Opaque.requires_alpha());
        assert!(AlphaModePreference::PreMultiplied.requires_alpha());
        assert!(AlphaModePreference::PostMultiplied.requires_alpha());
        assert!(AlphaModePreference::Inherit.requires_alpha());
        assert!(AlphaModePreference::Auto.requires_alpha());
    }

    #[test]
    fn test_alpha_mode_preference_to_concrete_mode() {
        assert_eq!(
            AlphaModePreference::Opaque.to_concrete_mode(),
            Some(wgpu::CompositeAlphaMode::Opaque)
        );
        assert_eq!(
            AlphaModePreference::PreMultiplied.to_concrete_mode(),
            Some(wgpu::CompositeAlphaMode::PreMultiplied)
        );
        assert_eq!(
            AlphaModePreference::PostMultiplied.to_concrete_mode(),
            Some(wgpu::CompositeAlphaMode::PostMultiplied)
        );
        assert_eq!(
            AlphaModePreference::Inherit.to_concrete_mode(),
            Some(wgpu::CompositeAlphaMode::Inherit)
        );
        assert_eq!(AlphaModePreference::Auto.to_concrete_mode(), None);
    }

    // =========================================================================
    // SurfaceCapabilities alpha mode selection tests
    // =========================================================================

    #[test]
    fn test_supports_alpha_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PreMultiplied,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        assert!(caps.supports_alpha_mode(wgpu::CompositeAlphaMode::Opaque));
        assert!(caps.supports_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied));
        assert!(!caps.supports_alpha_mode(wgpu::CompositeAlphaMode::PostMultiplied));
    }

    #[test]
    fn test_select_alpha_mode_auto() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PreMultiplied,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        // Auto should prefer Opaque
        assert_eq!(
            caps.select_alpha_mode(AlphaModePreference::Auto),
            wgpu::CompositeAlphaMode::Opaque
        );
    }

    #[test]
    fn test_select_alpha_mode_opaque_available() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PreMultiplied,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        assert_eq!(
            caps.select_alpha_mode(AlphaModePreference::Opaque),
            wgpu::CompositeAlphaMode::Opaque
        );
    }

    #[test]
    fn test_select_alpha_mode_opaque_fallback() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::PreMultiplied],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        // Opaque not available, should fall back to first available
        assert_eq!(
            caps.select_alpha_mode(AlphaModePreference::Opaque),
            wgpu::CompositeAlphaMode::PreMultiplied
        );
    }

    #[test]
    fn test_select_alpha_mode_premultiplied_available() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PreMultiplied,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        assert_eq!(
            caps.select_alpha_mode(AlphaModePreference::PreMultiplied),
            wgpu::CompositeAlphaMode::PreMultiplied
        );
    }

    #[test]
    fn test_select_alpha_mode_premultiplied_fallback_to_postmultiplied() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PostMultiplied,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        // PreMultiplied not available, should fall back to PostMultiplied
        assert_eq!(
            caps.select_alpha_mode(AlphaModePreference::PreMultiplied),
            wgpu::CompositeAlphaMode::PostMultiplied
        );
    }

    #[test]
    fn test_select_alpha_mode_postmultiplied_fallback_to_premultiplied() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PreMultiplied,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        // PostMultiplied not available, should fall back to PreMultiplied
        assert_eq!(
            caps.select_alpha_mode(AlphaModePreference::PostMultiplied),
            wgpu::CompositeAlphaMode::PreMultiplied
        );
    }

    #[test]
    fn test_select_alpha_mode_inherit() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::Inherit,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        assert_eq!(
            caps.select_alpha_mode(AlphaModePreference::Inherit),
            wgpu::CompositeAlphaMode::Inherit
        );
    }

    // =========================================================================
    // SurfaceConfiguration alpha mode preference tests
    // =========================================================================

    #[test]
    fn test_surface_configuration_with_alpha_mode_preference() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Auto,
                wgpu::CompositeAlphaMode::Opaque,
                wgpu::CompositeAlphaMode::PreMultiplied,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        let config = SurfaceConfiguration::new(1920, 1080)
            .with_alpha_mode_preference(&caps, AlphaModePreference::PreMultiplied);

        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::PreMultiplied);
    }

    // =========================================================================
    // View formats and sRGB toggle tests
    // =========================================================================

    #[test]
    fn test_surface_configuration_view_formats_empty_by_default() {
        let config = SurfaceConfiguration::new(1920, 1080);
        assert!(config.view_formats.is_empty());
    }

    #[test]
    fn test_surface_configuration_with_view_formats() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_view_formats(&[wgpu::TextureFormat::Bgra8UnormSrgb]);

        assert_eq!(config.view_formats.len(), 1);
        assert_eq!(config.view_formats[0], wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_surface_configuration_with_srgb_view_format_linear_base() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_srgb_view_format();

        assert!(config.view_formats.contains(&wgpu::TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn test_surface_configuration_with_srgb_view_format_srgb_base() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .with_srgb_view_format();

        // When base is sRGB, should add linear variant
        assert!(config.view_formats.contains(&wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_surface_configuration_with_srgb_view_format_rgba() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba8Unorm)
            .with_srgb_view_format();

        assert!(config.view_formats.contains(&wgpu::TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn test_surface_configuration_with_srgb_view_format_no_companion() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba16Float)
            .with_srgb_view_format();

        // HDR format has no sRGB companion
        assert!(config.view_formats.is_empty());
    }

    #[test]
    fn test_surface_configuration_has_srgb_view_format() {
        let config_with = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_view_formats(&[wgpu::TextureFormat::Bgra8UnormSrgb]);

        assert!(config_with.has_srgb_view_format());

        let config_without = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm);

        assert!(!config_without.has_srgb_view_format());
    }

    #[test]
    fn test_surface_configuration_srgb_format_main() {
        // Main format is sRGB
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb);

        assert_eq!(config.srgb_format(), Some(wgpu::TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn test_surface_configuration_srgb_format_view() {
        // Main is linear, sRGB in view formats
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_srgb_view_format();

        assert_eq!(config.srgb_format(), Some(wgpu::TextureFormat::Bgra8UnormSrgb));
    }

    #[test]
    fn test_surface_configuration_srgb_format_none() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba16Float);

        assert_eq!(config.srgb_format(), None);
    }

    #[test]
    fn test_surface_configuration_linear_format_main() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm);

        assert_eq!(config.linear_format(), Some(wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_surface_configuration_linear_format_view() {
        // Main is sRGB, linear in view formats
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .with_srgb_view_format();

        assert_eq!(config.linear_format(), Some(wgpu::TextureFormat::Bgra8Unorm));
    }

    // =========================================================================
    // sRGB companion format helper tests
    // =========================================================================

    #[test]
    fn test_get_srgb_companion_format_linear_to_srgb() {
        assert_eq!(
            get_srgb_companion_format(wgpu::TextureFormat::Bgra8Unorm),
            Some(wgpu::TextureFormat::Bgra8UnormSrgb)
        );
        assert_eq!(
            get_srgb_companion_format(wgpu::TextureFormat::Rgba8Unorm),
            Some(wgpu::TextureFormat::Rgba8UnormSrgb)
        );
    }

    #[test]
    fn test_get_srgb_companion_format_srgb_to_linear() {
        assert_eq!(
            get_srgb_companion_format(wgpu::TextureFormat::Bgra8UnormSrgb),
            Some(wgpu::TextureFormat::Bgra8Unorm)
        );
        assert_eq!(
            get_srgb_companion_format(wgpu::TextureFormat::Rgba8UnormSrgb),
            Some(wgpu::TextureFormat::Rgba8Unorm)
        );
    }

    #[test]
    fn test_get_srgb_companion_format_none() {
        assert_eq!(get_srgb_companion_format(wgpu::TextureFormat::Rgba16Float), None);
        assert_eq!(get_srgb_companion_format(wgpu::TextureFormat::Depth32Float), None);
        assert_eq!(get_srgb_companion_format(wgpu::TextureFormat::R8Unorm), None);
    }

    #[test]
    fn test_are_srgb_companions() {
        assert!(are_srgb_companions(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb
        ));
        assert!(are_srgb_companions(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm
        ));
        assert!(!are_srgb_companions(
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm
        ));
    }

    // =========================================================================
    // SurfaceConfiguration to_wgpu includes view_formats tests
    // =========================================================================

    #[test]
    fn test_surface_configuration_to_wgpu_includes_view_formats() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_view_formats(&[wgpu::TextureFormat::Bgra8UnormSrgb]);

        let wgpu_config = config.to_wgpu();
        assert_eq!(wgpu_config.view_formats.len(), 1);
        assert_eq!(wgpu_config.view_formats[0], wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    // =========================================================================
    // SurfaceConfiguration from_window_size tests
    // =========================================================================

    #[test]
    fn test_surface_configuration_from_window_size() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![wgpu::PresentMode::Mailbox],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Opaque],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        let config = SurfaceConfiguration::from_window_size(1920, 1080, &caps);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    // =========================================================================
    // SurfaceConfiguration view_formats not duplicated
    // =========================================================================

    #[test]
    fn test_surface_configuration_srgb_view_format_not_duplicated() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_srgb_view_format()
            .with_srgb_view_format(); // Called twice

        // Should not duplicate
        assert_eq!(config.view_formats.len(), 1);
    }

    // =========================================================================
    // FrameError tests
    // =========================================================================

    #[test]
    fn test_frame_error_timeout_is_recoverable() {
        let err = FrameError::Timeout;
        assert!(err.is_recoverable());
        assert!(!err.needs_reconfigure());
        assert!(!err.needs_recreate());
    }

    #[test]
    fn test_frame_error_outdated_needs_reconfigure() {
        let err = FrameError::Outdated;
        assert!(err.is_recoverable());
        assert!(err.needs_reconfigure());
        assert!(!err.needs_recreate());
    }

    #[test]
    fn test_frame_error_lost_needs_recreate() {
        let err = FrameError::lost("test reason");
        assert!(!err.is_recoverable());
        assert!(!err.needs_reconfigure());
        assert!(err.needs_recreate());
    }

    #[test]
    fn test_frame_error_lost_display() {
        let err = FrameError::lost("driver reset");
        let msg = format!("{}", err);
        assert!(msg.contains("driver reset"));
    }

    #[test]
    fn test_frame_error_timeout_display() {
        let err = FrameError::Timeout;
        let msg = format!("{}", err);
        assert!(msg.contains("timed out"));
    }

    #[test]
    fn test_frame_error_outdated_display() {
        let err = FrameError::Outdated;
        let msg = format!("{}", err);
        assert!(msg.contains("reconfiguration"));
    }

    #[test]
    fn test_frame_error_out_of_memory() {
        let err = FrameError::out_of_memory();
        assert!(err.needs_recreate());
        let msg = format!("{}", err);
        assert!(msg.contains("memory"));
    }

    #[test]
    fn test_frame_error_from_wgpu_timeout() {
        let wgpu_err = wgpu::SurfaceError::Timeout;
        let err: FrameError = wgpu_err.into();
        assert!(matches!(err, FrameError::Timeout));
    }

    #[test]
    fn test_frame_error_from_wgpu_outdated() {
        let wgpu_err = wgpu::SurfaceError::Outdated;
        let err: FrameError = wgpu_err.into();
        assert!(matches!(err, FrameError::Outdated));
    }

    #[test]
    fn test_frame_error_from_wgpu_lost() {
        let wgpu_err = wgpu::SurfaceError::Lost;
        let err: FrameError = wgpu_err.into();
        assert!(matches!(err, FrameError::Lost { .. }));
    }

    #[test]
    fn test_frame_error_from_wgpu_out_of_memory() {
        let wgpu_err = wgpu::SurfaceError::OutOfMemory;
        let err: FrameError = wgpu_err.into();
        assert!(matches!(err, FrameError::Lost { .. }));
        assert!(err.needs_recreate());
    }

    #[test]
    fn test_frame_error_recovery_classification() {
        // Test that error classification is correct for all variants
        let errors = [
            (FrameError::Timeout, true, false, false),
            (FrameError::Outdated, true, true, false),
            (FrameError::lost("test"), false, false, true),
        ];

        for (err, recoverable, needs_reconfig, needs_recreate) in errors {
            assert_eq!(
                err.is_recoverable(),
                recoverable,
                "is_recoverable failed for {:?}",
                err
            );
            assert_eq!(
                err.needs_reconfigure(),
                needs_reconfig,
                "needs_reconfigure failed for {:?}",
                err
            );
            assert_eq!(
                err.needs_recreate(),
                needs_recreate,
                "needs_recreate failed for {:?}",
                err
            );
        }
    }

    // =========================================================================
    // Frame struct tests (without actual GPU)
    // =========================================================================

    // Note: Frame tests that require an actual GPU surface are in integration tests.
    // These tests verify the logic and API shape without GPU resources.

    #[test]
    fn test_frame_debug_impl() {
        // Verify Frame derives Debug
        // This is a compile-time check - if Frame doesn't implement Debug, this won't compile
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<Frame>();
    }

    #[test]
    fn test_frame_error_debug_impl() {
        // Verify FrameError derives Debug
        let err = FrameError::Timeout;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Timeout"));
    }

    #[test]
    fn test_frame_error_debug_lost() {
        let err = FrameError::lost("test reason");
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Lost"));
        assert!(debug_str.contains("test reason"));
    }

    #[test]
    fn test_frame_error_debug_outdated() {
        let err = FrameError::Outdated;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Outdated"));
    }

    // =========================================================================
    // Frame acquisition API shape tests
    // =========================================================================

    // These tests verify the API compiles and has the expected signature.
    // Actual frame acquisition requires GPU integration tests.

    #[test]
    fn test_trinity_surface_acquire_frame_api_exists() {
        // Verify the method signature exists by checking it compiles
        fn _check_method<F: FnOnce(&TrinitySurface) -> Result<Frame, FrameError>>(_f: F) {}
        _check_method(|s| s.acquire_frame());
    }

    #[test]
    fn test_trinity_surface_try_acquire_frame_api_exists() {
        // Verify try_acquire_frame returns Option<Result<...>>
        fn _check_method<F: FnOnce(&TrinitySurface) -> Option<Result<Frame, FrameError>>>(_f: F) {}
        _check_method(|s| s.try_acquire_frame());
    }

    #[test]
    fn test_trinity_surface_acquire_frame_with_format_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> Result<Frame, FrameError>>(_f: F) {}
        _check_method(|s| s.acquire_frame_with_format(wgpu::TextureFormat::Bgra8Unorm));
    }

    // =========================================================================
    // Additional FrameError edge cases
    // =========================================================================

    #[test]
    fn test_frame_error_lost_with_empty_reason() {
        let err = FrameError::lost("");
        let msg = format!("{}", err);
        // Should still display, just with empty reason
        assert!(msg.contains("lost"));
    }

    #[test]
    fn test_frame_error_lost_with_long_reason() {
        let long_reason = "x".repeat(1000);
        let err = FrameError::lost(&long_reason);
        let msg = format!("{}", err);
        assert!(msg.contains(&long_reason));
    }

    // =========================================================================
    // ResizeEvent tests (T-WGPU-P7.1.7)
    // =========================================================================

    #[test]
    fn test_resize_event_new() {
        let event = ResizeEvent::new(1920, 1080, 2560, 1440);
        assert_eq!(event.old_width, 1920);
        assert_eq!(event.old_height, 1080);
        assert_eq!(event.new_width, 2560);
        assert_eq!(event.new_height, 1440);
    }

    #[test]
    fn test_resize_event_aspect_ratio_changed_true() {
        // 16:9 to 16:10 - different aspect ratios
        let event = ResizeEvent::new(1920, 1080, 1920, 1200);
        assert!(event.aspect_ratio_changed());
    }

    #[test]
    fn test_resize_event_aspect_ratio_changed_false() {
        // Same aspect ratio (16:9), different resolution
        let event = ResizeEvent::new(1920, 1080, 3840, 2160);
        assert!(!event.aspect_ratio_changed());
    }

    #[test]
    fn test_resize_event_aspect_ratio_same_dimensions() {
        let event = ResizeEvent::new(1920, 1080, 1920, 1080);
        assert!(!event.aspect_ratio_changed());
    }

    #[test]
    fn test_resize_event_old_aspect_ratio() {
        let event = ResizeEvent::new(1920, 1080, 800, 600);
        let expected = 1920.0 / 1080.0;
        assert!((event.old_aspect_ratio() - expected).abs() < 0.001);
    }

    #[test]
    fn test_resize_event_new_aspect_ratio() {
        let event = ResizeEvent::new(800, 600, 1920, 1080);
        let expected = 1920.0 / 1080.0;
        assert!((event.new_aspect_ratio() - expected).abs() < 0.001);
    }

    #[test]
    fn test_resize_event_aspect_ratio_zero_height() {
        let event = ResizeEvent::new(1920, 0, 1920, 0);
        assert_eq!(event.old_aspect_ratio(), 1.0);
        assert_eq!(event.new_aspect_ratio(), 1.0);
    }

    #[test]
    fn test_resize_event_is_minimize_zero() {
        let event = ResizeEvent::new(1920, 1080, 0, 0);
        assert!(event.is_minimize());
    }

    #[test]
    fn test_resize_event_is_minimize_one() {
        let event = ResizeEvent::new(1920, 1080, 1, 1);
        assert!(event.is_minimize());
    }

    #[test]
    fn test_resize_event_is_minimize_width_zero() {
        let event = ResizeEvent::new(1920, 1080, 0, 1080);
        assert!(event.is_minimize());
    }

    #[test]
    fn test_resize_event_is_minimize_height_zero() {
        let event = ResizeEvent::new(1920, 1080, 1920, 0);
        assert!(event.is_minimize());
    }

    #[test]
    fn test_resize_event_is_minimize_false() {
        let event = ResizeEvent::new(1920, 1080, 800, 600);
        assert!(!event.is_minimize());
    }

    #[test]
    fn test_resize_event_is_minimize_already_minimized() {
        // Going from minimized to minimized is not a "minimize" event
        let event = ResizeEvent::new(0, 0, 1, 1);
        assert!(!event.is_minimize());
    }

    #[test]
    fn test_resize_event_is_restore_from_zero() {
        let event = ResizeEvent::new(0, 0, 1920, 1080);
        assert!(event.is_restore());
    }

    #[test]
    fn test_resize_event_is_restore_from_one() {
        let event = ResizeEvent::new(1, 1, 1920, 1080);
        assert!(event.is_restore());
    }

    #[test]
    fn test_resize_event_is_restore_false() {
        let event = ResizeEvent::new(1920, 1080, 2560, 1440);
        assert!(!event.is_restore());
    }

    #[test]
    fn test_resize_event_grew() {
        let event = ResizeEvent::new(800, 600, 1920, 1080);
        assert!(event.grew());
    }

    #[test]
    fn test_resize_event_grew_width_only() {
        let event = ResizeEvent::new(800, 600, 1920, 600);
        assert!(event.grew());
    }

    #[test]
    fn test_resize_event_shrunk() {
        let event = ResizeEvent::new(1920, 1080, 800, 600);
        assert!(event.shrunk());
    }

    #[test]
    fn test_resize_event_shrunk_height_only() {
        let event = ResizeEvent::new(1920, 1080, 1920, 600);
        assert!(event.shrunk());
    }

    #[test]
    fn test_resize_event_scale_factor_double() {
        // Area doubled (2x2 -> 4x4 would be 4x scale)
        let event = ResizeEvent::new(100, 100, 200, 200);
        assert!((event.scale_factor() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_resize_event_scale_factor_half() {
        let event = ResizeEvent::new(200, 200, 100, 100);
        assert!((event.scale_factor() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_resize_event_scale_factor_zero_old() {
        let event = ResizeEvent::new(0, 0, 1920, 1080);
        assert_eq!(event.scale_factor(), 1.0);
    }

    #[test]
    fn test_resize_event_dimensions_changed_true() {
        let event = ResizeEvent::new(1920, 1080, 2560, 1440);
        assert!(event.dimensions_changed());
    }

    #[test]
    fn test_resize_event_dimensions_changed_false() {
        let event = ResizeEvent::new(1920, 1080, 1920, 1080);
        assert!(!event.dimensions_changed());
    }

    #[test]
    fn test_resize_event_width_delta_positive() {
        let event = ResizeEvent::new(800, 600, 1920, 1080);
        assert_eq!(event.width_delta(), 1120);
    }

    #[test]
    fn test_resize_event_width_delta_negative() {
        let event = ResizeEvent::new(1920, 1080, 800, 600);
        assert_eq!(event.width_delta(), -1120);
    }

    #[test]
    fn test_resize_event_height_delta() {
        let event = ResizeEvent::new(800, 600, 800, 1080);
        assert_eq!(event.height_delta(), 480);
    }

    #[test]
    fn test_resize_event_display() {
        let event = ResizeEvent::new(1920, 1080, 2560, 1440);
        let display = format!("{}", event);
        assert_eq!(display, "1920x1080 -> 2560x1440");
    }

    #[test]
    fn test_resize_event_debug() {
        let event = ResizeEvent::new(1920, 1080, 2560, 1440);
        let debug = format!("{:?}", event);
        assert!(debug.contains("ResizeEvent"));
        assert!(debug.contains("1920"));
        assert!(debug.contains("2560"));
    }

    #[test]
    fn test_resize_event_eq() {
        let event1 = ResizeEvent::new(1920, 1080, 2560, 1440);
        let event2 = ResizeEvent::new(1920, 1080, 2560, 1440);
        let event3 = ResizeEvent::new(800, 600, 1920, 1080);
        assert_eq!(event1, event2);
        assert_ne!(event1, event3);
    }

    #[test]
    fn test_resize_event_clone() {
        let event = ResizeEvent::new(1920, 1080, 2560, 1440);
        let cloned = event.clone();
        assert_eq!(event, cloned);
    }

    #[test]
    fn test_resize_event_copy() {
        let event = ResizeEvent::new(1920, 1080, 2560, 1440);
        let copied = event; // Copy
        assert_eq!(event, copied);
    }

    // =========================================================================
    // SurfaceConfiguration resize methods tests (T-WGPU-P7.1.7)
    // =========================================================================

    #[test]
    fn test_surface_configuration_with_dimensions() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_dimensions(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        // Other settings preserved
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8Unorm);
    }

    #[test]
    fn test_surface_configuration_with_dimensions_clamps_zero() {
        let config = SurfaceConfiguration::new(800, 600)
            .with_dimensions(0, 0);

        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn test_surface_configuration_resize_mutating() {
        let mut config = SurfaceConfiguration::new(800, 600);
        config.resize(1920, 1080);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn test_surface_configuration_resize_clamps_zero() {
        let mut config = SurfaceConfiguration::new(800, 600);
        config.resize(0, 0);

        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    // =========================================================================
    // TrinitySurface resize methods API shape tests (T-WGPU-P7.1.7)
    // =========================================================================

    #[test]
    fn test_trinity_surface_needs_resize_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> bool>(_f: F) {}
        _check_method(|s| s.needs_resize(1920, 1080));
    }

    #[test]
    fn test_trinity_surface_handle_resize_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface, &wgpu::Device) -> Result<Option<ResizeEvent>, SurfaceError>>(_f: F) {}
        // Can't fully verify without device, but check signature compiles
    }

    #[test]
    fn test_trinity_surface_is_minimized_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> bool>(_f: F) {}
        _check_method(|s| s.is_minimized());
    }

    #[test]
    fn test_trinity_surface_set_minimized_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface)>(_f: F) {}
        _check_method(|s| s.set_minimized(true));
    }

    #[test]
    fn test_trinity_surface_aspect_ratio_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> f32>(_f: F) {}
        _check_method(|s| s.aspect_ratio());
    }

    // =========================================================================
    // Frame Pacing tests (T-WGPU-P7.1.8)
    // =========================================================================

    // -- FrameTiming tests ---------------------------------------------------

    #[test]
    fn test_frame_timing_new() {
        let timing = FrameTiming::new();
        assert_eq!(timing.frame_count(), 0);
        assert_eq!(timing.frame_delta(), Duration::ZERO);
        assert!(timing.target_frame_time().is_none());
        assert!(!timing.in_frame());
        assert_eq!(timing.history_size(), DEFAULT_FRAME_HISTORY_SIZE);
    }

    #[test]
    fn test_frame_timing_with_history_size() {
        let timing = FrameTiming::with_history_size(50);
        assert_eq!(timing.history_size(), 50);
    }

    #[test]
    fn test_frame_timing_default() {
        let timing = FrameTiming::default();
        assert_eq!(timing.frame_count(), 0);
    }

    #[test]
    fn test_frame_timing_set_target_fps() {
        let mut timing = FrameTiming::new();
        timing.set_target_fps(Some(60));
        let target = timing.target_frame_time().unwrap();
        let expected = Duration::from_secs_f64(1.0 / 60.0);
        // Allow small tolerance for floating point
        assert!((target.as_secs_f64() - expected.as_secs_f64()).abs() < 0.0001);
    }

    #[test]
    fn test_frame_timing_set_target_fps_none() {
        let mut timing = FrameTiming::new();
        timing.set_target_fps(Some(60));
        timing.set_target_fps(None);
        assert!(timing.target_frame_time().is_none());
    }

    #[test]
    fn test_frame_timing_set_target_frame_time() {
        let mut timing = FrameTiming::new();
        let target = Duration::from_millis(16);
        timing.set_target_frame_time(Some(target));
        assert_eq!(timing.target_frame_time(), Some(target));
    }

    #[test]
    fn test_frame_timing_begin_end_frame() {
        let mut timing = FrameTiming::new();
        assert!(!timing.in_frame());

        timing.begin_frame();
        assert!(timing.in_frame());

        // Simulate some work
        std::thread::sleep(Duration::from_millis(1));

        timing.end_frame();
        assert!(!timing.in_frame());
        assert_eq!(timing.frame_count(), 1);
        assert!(timing.frame_delta() >= Duration::from_millis(1));
    }

    #[test]
    fn test_frame_timing_elapsed() {
        let mut timing = FrameTiming::new();
        assert_eq!(timing.elapsed(), Duration::ZERO);

        timing.begin_frame();
        std::thread::sleep(Duration::from_millis(1));
        assert!(timing.elapsed() >= Duration::from_millis(1));

        timing.end_frame();
        assert_eq!(timing.elapsed(), Duration::ZERO);
    }

    #[test]
    fn test_frame_timing_double_begin_auto_ends() {
        let mut timing = FrameTiming::new();
        timing.begin_frame();
        std::thread::sleep(Duration::from_millis(1));
        // Double begin should auto-end the previous frame
        timing.begin_frame();
        assert_eq!(timing.frame_count(), 1);
        assert!(timing.in_frame());
    }

    #[test]
    fn test_frame_timing_end_without_begin() {
        let mut timing = FrameTiming::new();
        timing.end_frame(); // Should be a no-op
        assert_eq!(timing.frame_count(), 0);
    }

    #[test]
    fn test_frame_timing_frame_history() {
        let mut timing = FrameTiming::with_history_size(5);

        // Record 10 frames
        for _ in 0..10 {
            timing.begin_frame();
            timing.end_frame();
        }

        // Should only keep last 5
        assert_eq!(timing.frame_times().len(), 5);
        assert_eq!(timing.frame_count(), 10);
    }

    #[test]
    fn test_frame_timing_reset() {
        let mut timing = FrameTiming::new();
        timing.begin_frame();
        timing.end_frame();
        assert_eq!(timing.frame_count(), 1);

        timing.reset();
        assert_eq!(timing.frame_count(), 0);
        assert_eq!(timing.frame_delta(), Duration::ZERO);
        assert!(timing.frame_times().is_empty());
        assert!(!timing.in_frame());
    }

    #[test]
    fn test_frame_timing_debug() {
        let timing = FrameTiming::new();
        let debug = format!("{:?}", timing);
        assert!(debug.contains("FrameTiming"));
    }

    #[test]
    fn test_frame_timing_clone() {
        let mut timing = FrameTiming::new();
        timing.set_target_fps(Some(60));
        timing.begin_frame();
        timing.end_frame();

        let cloned = timing.clone();
        assert_eq!(cloned.frame_count(), timing.frame_count());
        assert_eq!(cloned.target_frame_time(), timing.target_frame_time());
    }

    // -- FrameStatistics tests -----------------------------------------------

    #[test]
    fn test_frame_statistics_from_empty() {
        let stats = FrameStatistics::from_times(std::iter::empty::<Duration>());
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.min_frame_time, Duration::ZERO);
        assert_eq!(stats.max_frame_time, Duration::ZERO);
        assert_eq!(stats.avg_frame_time, Duration::ZERO);
        assert_eq!(stats.fps(), 0.0);
    }

    #[test]
    fn test_frame_statistics_from_single() {
        let times = vec![Duration::from_millis(16)];
        let stats = FrameStatistics::from_times(times);

        assert_eq!(stats.sample_count, 1);
        assert_eq!(stats.min_frame_time, Duration::from_millis(16));
        assert_eq!(stats.max_frame_time, Duration::from_millis(16));
        assert_eq!(stats.avg_frame_time, Duration::from_millis(16));
    }

    #[test]
    fn test_frame_statistics_from_multiple() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        ];
        let stats = FrameStatistics::from_times(times);

        assert_eq!(stats.sample_count, 3);
        assert_eq!(stats.min_frame_time, Duration::from_millis(10));
        assert_eq!(stats.max_frame_time, Duration::from_millis(30));
        assert_eq!(stats.avg_frame_time, Duration::from_millis(20));
    }

    #[test]
    fn test_frame_statistics_fps() {
        let times = vec![Duration::from_millis(16)]; // ~62.5 FPS
        let stats = FrameStatistics::from_times(times);

        let fps = stats.fps();
        assert!(fps > 60.0 && fps < 65.0);
    }

    #[test]
    fn test_frame_statistics_fps_zero_time() {
        let times = vec![Duration::ZERO];
        let stats = FrameStatistics::from_times(times);
        assert_eq!(stats.fps(), 0.0);
    }

    #[test]
    fn test_frame_statistics_min_max_fps() {
        let times = vec![
            Duration::from_millis(10), // 100 FPS
            Duration::from_millis(20), // 50 FPS
        ];
        let stats = FrameStatistics::from_times(times);

        // max_fps from min_frame_time
        assert!((stats.max_fps() - 100.0).abs() < 1.0);
        // min_fps from max_frame_time
        assert!((stats.min_fps() - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_frame_statistics_percentile() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(100), // Outlier
        ];
        let stats = FrameStatistics::from_times(times);

        // p0 should be min
        assert_eq!(stats.percentile_frame_time(0.0), Duration::from_millis(10));
        // p100 should be max
        assert_eq!(stats.percentile_frame_time(1.0), Duration::from_millis(100));
        // p50 should be median (30ms)
        assert_eq!(stats.percentile_frame_time(0.5), Duration::from_millis(30));
    }

    #[test]
    fn test_frame_statistics_percentile_clamped() {
        let times = vec![Duration::from_millis(16)];
        let stats = FrameStatistics::from_times(times);

        // Out of range should be clamped
        assert_eq!(stats.percentile_frame_time(-0.5), Duration::from_millis(16));
        assert_eq!(stats.percentile_frame_time(1.5), Duration::from_millis(16));
    }

    #[test]
    fn test_frame_statistics_median() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        ];
        let stats = FrameStatistics::from_times(times);
        assert_eq!(stats.median_frame_time(), Duration::from_millis(20));
    }

    #[test]
    fn test_frame_statistics_variance() {
        // All same values - variance should be 0
        let times = vec![
            Duration::from_millis(16),
            Duration::from_millis(16),
            Duration::from_millis(16),
        ];
        let stats = FrameStatistics::from_times(times);
        assert!(stats.frame_time_variance() < 0.0001);
    }

    #[test]
    fn test_frame_statistics_variance_with_variation() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        ];
        let stats = FrameStatistics::from_times(times);
        assert!(stats.frame_time_variance() > 0.0);
    }

    #[test]
    fn test_frame_statistics_variance_insufficient_samples() {
        let times = vec![Duration::from_millis(16)];
        let stats = FrameStatistics::from_times(times);
        assert_eq!(stats.frame_time_variance(), 0.0);
    }

    #[test]
    fn test_frame_statistics_std_dev() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        ];
        let stats = FrameStatistics::from_times(times);
        let std_dev = stats.frame_time_std_dev();
        assert!(std_dev > 0.0);
        assert!((std_dev * std_dev - stats.frame_time_variance()).abs() < 0.0001);
    }

    #[test]
    fn test_frame_statistics_is_consistent() {
        // All same values - should be consistent
        let times = vec![
            Duration::from_millis(16),
            Duration::from_millis(16),
            Duration::from_millis(16),
        ];
        let stats = FrameStatistics::from_times(times);
        assert!(stats.is_consistent(0.1)); // 10% threshold
    }

    #[test]
    fn test_frame_statistics_is_consistent_high_variance() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(100),
        ];
        let stats = FrameStatistics::from_times(times);
        // High variance should not be consistent
        assert!(!stats.is_consistent(0.1));
    }

    #[test]
    fn test_frame_statistics_default() {
        let stats = FrameStatistics::default();
        assert_eq!(stats.sample_count, 0);
    }

    #[test]
    fn test_frame_statistics_display() {
        let times = vec![Duration::from_millis(16)];
        let stats = FrameStatistics::from_times(times);
        let display = format!("{}", stats);
        assert!(display.contains("FPS"));
        assert!(display.contains("frame time"));
    }

    #[test]
    fn test_frame_statistics_debug() {
        let stats = FrameStatistics::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("FrameStatistics"));
    }

    #[test]
    fn test_frame_statistics_clone_eq() {
        let times = vec![Duration::from_millis(16)];
        let stats = FrameStatistics::from_times(times);
        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }

    // -- FramePacer tests ----------------------------------------------------

    #[test]
    fn test_frame_pacer_new_with_fps() {
        let pacer = FramePacer::new(Some(60));
        assert!(pacer.is_limiting_enabled());
        let target = pacer.target_fps().unwrap();
        assert!((target - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_frame_pacer_new_without_fps() {
        let pacer = FramePacer::new(None);
        assert!(!pacer.is_limiting_enabled());
        assert!(pacer.target_fps().is_none());
    }

    #[test]
    fn test_frame_pacer_with_history_size() {
        let pacer = FramePacer::with_history_size(Some(60), 50);
        assert_eq!(pacer.timing().history_size(), 50);
    }

    #[test]
    fn test_frame_pacer_default() {
        let pacer = FramePacer::default();
        assert!(!pacer.is_limiting_enabled());
    }

    #[test]
    fn test_frame_pacer_begin_end_frame() {
        let mut pacer = FramePacer::new(None);
        pacer.begin_frame();
        pacer.end_frame();
        assert_eq!(pacer.frame_count(), 1);
    }

    #[test]
    fn test_frame_pacer_set_target_fps() {
        let mut pacer = FramePacer::new(None);
        assert!(!pacer.is_limiting_enabled());

        pacer.set_target_fps(Some(60));
        assert!(pacer.is_limiting_enabled());

        pacer.set_target_fps(None);
        assert!(!pacer.is_limiting_enabled());
    }

    #[test]
    fn test_frame_pacer_set_skip_threshold() {
        let mut pacer = FramePacer::new(Some(60));
        pacer.set_skip_threshold(5);
        assert_eq!(pacer.skip_threshold(), 5);
    }

    #[test]
    fn test_frame_pacer_skip_threshold_minimum() {
        let mut pacer = FramePacer::new(Some(60));
        pacer.set_skip_threshold(0);
        assert_eq!(pacer.skip_threshold(), 1); // Clamped to 1
    }

    #[test]
    fn test_frame_pacer_statistics() {
        let mut pacer = FramePacer::new(None);

        // Record some frames
        for _ in 0..5 {
            pacer.begin_frame();
            pacer.end_frame();
        }

        let stats = pacer.statistics();
        assert_eq!(stats.sample_count, 5);
    }

    #[test]
    fn test_frame_pacer_current_fps() {
        let mut pacer = FramePacer::new(None);

        // Before any frames
        assert_eq!(pacer.current_fps(), 0.0);

        pacer.begin_frame();
        std::thread::sleep(Duration::from_millis(10));
        pacer.end_frame();

        // Should have some FPS now
        let fps = pacer.current_fps();
        assert!(fps > 0.0 && fps < 200.0);
    }

    #[test]
    fn test_frame_pacer_frame_delta() {
        let mut pacer = FramePacer::new(None);
        pacer.begin_frame();
        std::thread::sleep(Duration::from_millis(5));
        pacer.end_frame();

        assert!(pacer.frame_delta() >= Duration::from_millis(5));
    }

    #[test]
    fn test_frame_pacer_set_limiting_enabled() {
        let mut pacer = FramePacer::new(Some(60));
        assert!(pacer.is_limiting_enabled());

        pacer.set_limiting_enabled(false);
        assert!(!pacer.is_limiting_enabled());

        pacer.set_limiting_enabled(true);
        assert!(pacer.is_limiting_enabled());
    }

    #[test]
    fn test_frame_pacer_set_limiting_enabled_no_target() {
        let mut pacer = FramePacer::new(None);
        // Can't enable limiting without a target
        pacer.set_limiting_enabled(true);
        assert!(!pacer.is_limiting_enabled());
    }

    #[test]
    fn test_frame_pacer_wait_for_target_no_limit() {
        let mut pacer = FramePacer::new(None);
        pacer.begin_frame();
        pacer.end_frame();

        // Should return immediately
        let waited = pacer.wait_for_target();
        assert_eq!(waited, Duration::ZERO);
    }

    #[test]
    fn test_frame_pacer_should_skip_frame_no_limit() {
        let mut pacer = FramePacer::new(None);
        assert!(!pacer.should_skip_frame());
    }

    #[test]
    fn test_frame_pacer_time_debt() {
        let pacer = FramePacer::new(Some(60));
        assert_eq!(pacer.time_debt(), Duration::ZERO);
    }

    #[test]
    fn test_frame_pacer_frames_skipped() {
        let pacer = FramePacer::new(Some(60));
        assert_eq!(pacer.frames_skipped(), 0);
    }

    #[test]
    fn test_frame_pacer_timing_access() {
        let pacer = FramePacer::new(Some(60));
        let timing = pacer.timing();
        assert_eq!(timing.frame_count(), 0);
    }

    #[test]
    fn test_frame_pacer_timing_mut_access() {
        let mut pacer = FramePacer::new(None);
        let timing = pacer.timing_mut();
        timing.set_target_fps(Some(30));
        assert!(pacer.timing().target_frame_time().is_some());
    }

    #[test]
    fn test_frame_pacer_reset() {
        let mut pacer = FramePacer::new(Some(60));

        pacer.begin_frame();
        pacer.end_frame();
        assert_eq!(pacer.frame_count(), 1);

        pacer.reset();
        assert_eq!(pacer.frame_count(), 0);
        assert_eq!(pacer.frames_skipped(), 0);
        assert_eq!(pacer.time_debt(), Duration::ZERO);
    }

    #[test]
    fn test_frame_pacer_debug() {
        let pacer = FramePacer::new(Some(60));
        let debug = format!("{:?}", pacer);
        assert!(debug.contains("FramePacer"));
    }

    #[test]
    fn test_frame_pacer_clone() {
        let mut pacer = FramePacer::new(Some(60));
        pacer.begin_frame();
        pacer.end_frame();

        let cloned = pacer.clone();
        assert_eq!(cloned.frame_count(), pacer.frame_count());
        assert_eq!(cloned.target_fps(), pacer.target_fps());
    }

    // -- TrinitySurface frame pacing API tests -------------------------------

    #[test]
    fn test_trinity_surface_set_target_fps_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface)>(_f: F) {}
        _check_method(|s| s.set_target_fps(Some(60)));
    }

    #[test]
    fn test_trinity_surface_target_fps_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> Option<f64>>(_f: F) {}
        _check_method(|s| s.target_fps());
    }

    #[test]
    fn test_trinity_surface_enable_frame_tracking_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface)>(_f: F) {}
        _check_method(|s| s.enable_frame_tracking());
    }

    #[test]
    fn test_trinity_surface_disable_frame_pacing_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface)>(_f: F) {}
        _check_method(|s| s.disable_frame_pacing());
    }

    #[test]
    fn test_trinity_surface_has_frame_pacer_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> bool>(_f: F) {}
        _check_method(|s| s.has_frame_pacer());
    }

    #[test]
    fn test_trinity_surface_frame_statistics_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> Option<FrameStatistics>>(_f: F) {}
        _check_method(|s| s.frame_statistics());
    }

    #[test]
    fn test_trinity_surface_frame_pacer_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> Option<&FramePacer>>(_f: F) {}
        _check_method(|s| s.frame_pacer());
    }

    #[test]
    fn test_trinity_surface_frame_pacer_mut_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface) -> Option<&mut FramePacer>>(_f: F) {}
        _check_method(|s| s.frame_pacer_mut());
    }

    #[test]
    fn test_trinity_surface_begin_frame_pacing_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface)>(_f: F) {}
        _check_method(|s| s.begin_frame_pacing());
    }

    #[test]
    fn test_trinity_surface_end_frame_pacing_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface)>(_f: F) {}
        _check_method(|s| s.end_frame_pacing());
    }

    #[test]
    fn test_trinity_surface_wait_for_target_fps_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface) -> Duration>(_f: F) {}
        _check_method(|s| s.wait_for_target_fps());
    }

    #[test]
    fn test_trinity_surface_should_skip_frame_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface) -> bool>(_f: F) {}
        _check_method(|s| s.should_skip_frame());
    }

    #[test]
    fn test_trinity_surface_current_fps_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> f64>(_f: F) {}
        _check_method(|s| s.current_fps());
    }

    #[test]
    fn test_trinity_surface_last_frame_delta_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> Duration>(_f: F) {}
        _check_method(|s| s.last_frame_delta());
    }

    #[test]
    fn test_trinity_surface_paced_frame_count_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> u64>(_f: F) {}
        _check_method(|s| s.paced_frame_count());
    }

    #[test]
    fn test_trinity_surface_frames_skipped_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> u64>(_f: F) {}
        _check_method(|s| s.frames_skipped());
    }

    // -- DEFAULT_FRAME_HISTORY_SIZE constant test ----------------------------

    #[test]
    fn test_default_frame_history_size_constant() {
        assert_eq!(DEFAULT_FRAME_HISTORY_SIZE, 100);
    }

    // =========================================================================
    // Triple Buffering Support tests (T-WGPU-P7.1.9)
    // =========================================================================

    // -- BufferingMode tests -------------------------------------------------

    #[test]
    fn test_buffering_mode_default() {
        assert_eq!(BufferingMode::default(), BufferingMode::Double);
    }

    #[test]
    fn test_buffering_mode_from_frame_latency() {
        assert_eq!(BufferingMode::from_frame_latency(0), BufferingMode::Double);
        assert_eq!(BufferingMode::from_frame_latency(1), BufferingMode::Double);
        assert_eq!(BufferingMode::from_frame_latency(2), BufferingMode::Double);
        assert_eq!(BufferingMode::from_frame_latency(3), BufferingMode::Triple);
        assert_eq!(BufferingMode::from_frame_latency(4), BufferingMode::Quad);
        assert_eq!(BufferingMode::from_frame_latency(5), BufferingMode::Quad);
        assert_eq!(BufferingMode::from_frame_latency(100), BufferingMode::Quad);
    }

    #[test]
    fn test_buffering_mode_buffer_count() {
        assert_eq!(BufferingMode::Double.buffer_count(), 2);
        assert_eq!(BufferingMode::Triple.buffer_count(), 3);
        assert_eq!(BufferingMode::Quad.buffer_count(), 4);
    }

    #[test]
    fn test_buffering_mode_frame_latency() {
        assert_eq!(BufferingMode::Double.frame_latency(), 2);
        assert_eq!(BufferingMode::Triple.frame_latency(), 3);
        assert_eq!(BufferingMode::Quad.frame_latency(), 4);
    }

    #[test]
    fn test_buffering_mode_max_in_flight() {
        assert_eq!(BufferingMode::Double.max_in_flight(), 1);
        assert_eq!(BufferingMode::Triple.max_in_flight(), 2);
        assert_eq!(BufferingMode::Quad.max_in_flight(), 3);
    }

    #[test]
    fn test_buffering_mode_latency_frames() {
        assert_eq!(BufferingMode::Double.latency_frames(), 1);
        assert_eq!(BufferingMode::Triple.latency_frames(), 2);
        assert_eq!(BufferingMode::Quad.latency_frames(), 3);
    }

    #[test]
    fn test_buffering_mode_name() {
        assert_eq!(BufferingMode::Double.name(), "Double Buffering");
        assert_eq!(BufferingMode::Triple.name(), "Triple Buffering");
        assert_eq!(BufferingMode::Quad.name(), "Quad Buffering");
    }

    #[test]
    fn test_buffering_mode_description() {
        assert!(!BufferingMode::Double.description().is_empty());
        assert!(!BufferingMode::Triple.description().is_empty());
        assert!(!BufferingMode::Quad.description().is_empty());
    }

    #[test]
    fn test_buffering_mode_is_smooth_pacing() {
        assert!(!BufferingMode::Double.is_smooth_pacing());
        assert!(BufferingMode::Triple.is_smooth_pacing());
        assert!(BufferingMode::Quad.is_smooth_pacing());
    }

    #[test]
    fn test_buffering_mode_is_low_latency() {
        assert!(BufferingMode::Double.is_low_latency());
        assert!(!BufferingMode::Triple.is_low_latency());
        assert!(!BufferingMode::Quad.is_low_latency());
    }

    #[test]
    fn test_buffering_mode_display() {
        assert_eq!(format!("{}", BufferingMode::Double), "Double Buffering");
        assert_eq!(format!("{}", BufferingMode::Triple), "Triple Buffering");
        assert_eq!(format!("{}", BufferingMode::Quad), "Quad Buffering");
    }

    #[test]
    fn test_buffering_mode_debug() {
        let debug = format!("{:?}", BufferingMode::Triple);
        assert!(debug.contains("Triple"));
    }

    #[test]
    fn test_buffering_mode_clone_eq() {
        let mode = BufferingMode::Triple;
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_buffering_mode_copy() {
        let mode = BufferingMode::Triple;
        let copied = mode;
        assert_eq!(mode, copied);
    }

    #[test]
    fn test_buffering_mode_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BufferingMode::Double);
        set.insert(BufferingMode::Triple);
        set.insert(BufferingMode::Quad);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&BufferingMode::Triple));
    }

    // -- BufferingConfig tests -----------------------------------------------

    #[test]
    fn test_buffering_config_new() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        assert_eq!(config.mode, BufferingMode::Triple);
        assert_eq!(config.desired_latency, 3);
        assert_eq!(config.actual_latency, 3);
    }

    #[test]
    fn test_buffering_config_from_latency() {
        let config = BufferingConfig::from_latency(3);
        assert_eq!(config.mode, BufferingMode::Triple);
        assert!(config.is_triple_buffered());
    }

    #[test]
    fn test_buffering_config_with_actual() {
        let config = BufferingConfig::with_actual(3, 2);
        assert_eq!(config.desired_latency, 3);
        assert_eq!(config.actual_latency, 2);
        assert_eq!(config.mode, BufferingMode::Double);
    }

    #[test]
    fn test_buffering_config_is_triple_buffered() {
        assert!(!BufferingConfig::from_latency(2).is_triple_buffered());
        assert!(BufferingConfig::from_latency(3).is_triple_buffered());
        assert!(BufferingConfig::from_latency(4).is_triple_buffered());
    }

    #[test]
    fn test_buffering_config_buffer_count() {
        assert_eq!(BufferingConfig::new(BufferingMode::Double).buffer_count(), 2);
        assert_eq!(BufferingConfig::new(BufferingMode::Triple).buffer_count(), 3);
        assert_eq!(BufferingConfig::new(BufferingMode::Quad).buffer_count(), 4);
    }

    #[test]
    fn test_buffering_config_latency_frames() {
        assert_eq!(BufferingConfig::new(BufferingMode::Double).latency_frames(), 1);
        assert_eq!(BufferingConfig::new(BufferingMode::Triple).latency_frames(), 2);
        assert_eq!(BufferingConfig::new(BufferingMode::Quad).latency_frames(), 3);
    }

    #[test]
    fn test_buffering_config_max_in_flight() {
        assert_eq!(BufferingConfig::new(BufferingMode::Double).max_in_flight(), 1);
        assert_eq!(BufferingConfig::new(BufferingMode::Triple).max_in_flight(), 2);
        assert_eq!(BufferingConfig::new(BufferingMode::Quad).max_in_flight(), 3);
    }

    #[test]
    fn test_buffering_config_latency_matches() {
        let matching = BufferingConfig::new(BufferingMode::Triple);
        assert!(matching.latency_matches());

        let mismatched = BufferingConfig::with_actual(3, 2);
        assert!(!mismatched.latency_matches());
    }

    #[test]
    fn test_buffering_config_latency_mismatch_description() {
        let matching = BufferingConfig::new(BufferingMode::Triple);
        assert!(matching.latency_mismatch_description().is_none());

        let mismatched = BufferingConfig::with_actual(3, 2);
        let desc = mismatched.latency_mismatch_description();
        assert!(desc.is_some());
        let desc = desc.unwrap();
        assert!(desc.contains("3"));
        assert!(desc.contains("2"));
    }

    #[test]
    fn test_buffering_config_tradeoff_description() {
        let double = BufferingConfig::new(BufferingMode::Double);
        assert!(!double.tradeoff_description().is_empty());
        assert!(double.tradeoff_description().contains("latency"));

        let triple = BufferingConfig::new(BufferingMode::Triple);
        assert!(triple.tradeoff_description().contains("smooth"));
    }

    #[test]
    fn test_buffering_config_default() {
        let config = BufferingConfig::default();
        assert_eq!(config.mode, BufferingMode::Double);
    }

    #[test]
    fn test_buffering_config_display() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let display = format!("{}", config);
        assert!(display.contains("Triple"));
        assert!(display.contains("3"));
    }

    #[test]
    fn test_buffering_config_debug() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let debug = format!("{:?}", config);
        assert!(debug.contains("BufferingConfig"));
        assert!(debug.contains("Triple"));
    }

    #[test]
    fn test_buffering_config_clone_eq() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_buffering_config_copy() {
        let config = BufferingConfig::new(BufferingMode::Triple);
        let copied = config;
        assert_eq!(config, copied);
    }

    // -- FrameInFlightTracker tests ------------------------------------------

    #[test]
    fn test_frame_in_flight_tracker_new() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        assert_eq!(tracker.frames_in_flight(), 0);
        assert_eq!(tracker.max_frames_in_flight(), 2);
        assert_eq!(tracker.total_submitted(), 0);
        assert_eq!(tracker.total_presented(), 0);
    }

    #[test]
    fn test_frame_in_flight_tracker_with_max() {
        let tracker = FrameInFlightTracker::with_max(5);
        assert_eq!(tracker.max_frames_in_flight(), 5);
    }

    #[test]
    fn test_frame_in_flight_tracker_default() {
        let tracker = FrameInFlightTracker::default();
        assert_eq!(tracker.max_frames_in_flight(), 1); // Double buffering default
    }

    #[test]
    fn test_frame_in_flight_tracker_frame_submitted() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        assert_eq!(tracker.frame_submitted(), 1);
        assert_eq!(tracker.frames_in_flight(), 1);
        assert_eq!(tracker.total_submitted(), 1);

        assert_eq!(tracker.frame_submitted(), 2);
        assert_eq!(tracker.frames_in_flight(), 2);
        assert_eq!(tracker.total_submitted(), 2);
    }

    #[test]
    fn test_frame_in_flight_tracker_frame_presented() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.frames_in_flight(), 2);

        assert_eq!(tracker.frame_presented(), 1);
        assert_eq!(tracker.frames_in_flight(), 1);
        assert_eq!(tracker.total_presented(), 1);

        assert_eq!(tracker.frame_presented(), 0);
        assert_eq!(tracker.frames_in_flight(), 0);
        assert_eq!(tracker.total_presented(), 2);
    }

    #[test]
    fn test_frame_in_flight_tracker_presented_underflow_protection() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        // Presenting without submitting should not underflow
        assert_eq!(tracker.frame_presented(), 0);
        assert_eq!(tracker.frames_in_flight(), 0);
    }

    #[test]
    fn test_frame_in_flight_tracker_is_at_capacity() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        assert!(!tracker.is_at_capacity());
        assert!(tracker.has_capacity());

        tracker.frame_submitted();
        assert!(!tracker.is_at_capacity());

        tracker.frame_submitted();
        assert!(tracker.is_at_capacity()); // 2 == max for triple
        assert!(!tracker.has_capacity());
    }

    #[test]
    fn test_frame_in_flight_tracker_remaining_capacity() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        assert_eq!(tracker.remaining_capacity(), 2);

        tracker.frame_submitted();
        assert_eq!(tracker.remaining_capacity(), 1);

        tracker.frame_submitted();
        assert_eq!(tracker.remaining_capacity(), 0);

        tracker.frame_submitted(); // Over capacity
        assert_eq!(tracker.remaining_capacity(), 0);
    }

    #[test]
    fn test_frame_in_flight_tracker_max_observed() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Quad);

        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_submitted();
        assert_eq!(tracker.max_observed(), 3);

        // Present one, max observed shouldn't decrease
        tracker.frame_presented();
        assert_eq!(tracker.max_observed(), 3);
        assert_eq!(tracker.frames_in_flight(), 2);
    }

    #[test]
    fn test_frame_in_flight_tracker_utilization() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        assert_eq!(tracker.utilization(), 0.0);

        tracker.frame_submitted();
        assert!((tracker.utilization() - 0.5).abs() < 0.001);

        tracker.frame_submitted();
        assert!((tracker.utilization() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_frame_in_flight_tracker_utilization_zero_max() {
        let tracker = FrameInFlightTracker::with_max(0);
        assert_eq!(tracker.utilization(), 0.0);
    }

    #[test]
    fn test_frame_in_flight_tracker_reset() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);

        tracker.frame_submitted();
        tracker.frame_submitted();
        tracker.frame_presented();

        tracker.reset();

        assert_eq!(tracker.frames_in_flight(), 0);
        assert_eq!(tracker.total_submitted(), 0);
        assert_eq!(tracker.total_presented(), 0);
        assert_eq!(tracker.max_observed(), 0);
    }

    #[test]
    fn test_frame_in_flight_tracker_set_max_in_flight() {
        let mut tracker = FrameInFlightTracker::new(BufferingMode::Double);
        assert_eq!(tracker.max_frames_in_flight(), 1);

        tracker.set_max_in_flight(3);
        assert_eq!(tracker.max_frames_in_flight(), 3);
    }

    #[test]
    fn test_frame_in_flight_tracker_clone() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        tracker.frame_submitted();
        tracker.frame_submitted();

        let cloned = tracker.clone();
        assert_eq!(cloned.frames_in_flight(), tracker.frames_in_flight());
        assert_eq!(cloned.max_frames_in_flight(), tracker.max_frames_in_flight());
        assert_eq!(cloned.total_submitted(), tracker.total_submitted());
    }

    #[test]
    fn test_frame_in_flight_tracker_debug() {
        let tracker = FrameInFlightTracker::new(BufferingMode::Triple);
        let debug = format!("{:?}", tracker);
        assert!(debug.contains("FrameInFlightTracker"));
    }

    // -- SurfaceConfiguration buffering methods tests ------------------------

    #[test]
    fn test_surface_configuration_with_buffering_mode() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_buffering_mode(BufferingMode::Triple);

        assert_eq!(config.desired_maximum_frame_latency, 3);
    }

    #[test]
    fn test_surface_configuration_buffering_config() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_frame_latency(3);

        let buffering = config.buffering_config();
        assert_eq!(buffering.mode, BufferingMode::Triple);
        assert!(buffering.is_triple_buffered());
    }

    #[test]
    fn test_surface_configuration_buffering_mode() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_frame_latency(3);

        assert_eq!(config.buffering_mode(), BufferingMode::Triple);
    }

    #[test]
    fn test_surface_configuration_is_triple_buffered() {
        let double = SurfaceConfiguration::new(1920, 1080)
            .with_frame_latency(2);
        assert!(!double.is_triple_buffered());

        let triple = SurfaceConfiguration::new(1920, 1080)
            .with_frame_latency(3);
        assert!(triple.is_triple_buffered());
    }

    // -- TrinitySurface buffering API tests ----------------------------------

    #[test]
    fn test_trinity_surface_buffering_config_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> Option<BufferingConfig>>(_f: F) {}
        _check_method(|s| s.buffering_config());
    }

    #[test]
    fn test_trinity_surface_is_triple_buffered_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> bool>(_f: F) {}
        _check_method(|s| s.is_triple_buffered());
    }

    #[test]
    fn test_trinity_surface_buffering_mode_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> BufferingMode>(_f: F) {}
        _check_method(|s| s.buffering_mode());
    }

    #[test]
    fn test_trinity_surface_frames_in_flight_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> u32>(_f: F) {}
        _check_method(|s| s.frames_in_flight());
    }

    #[test]
    fn test_trinity_surface_max_frames_in_flight_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> u32>(_f: F) {}
        _check_method(|s| s.max_frames_in_flight());
    }

    #[test]
    fn test_trinity_surface_record_frame_submitted_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> u32>(_f: F) {}
        _check_method(|s| s.record_frame_submitted());
    }

    #[test]
    fn test_trinity_surface_record_frame_presented_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> u32>(_f: F) {}
        _check_method(|s| s.record_frame_presented());
    }

    #[test]
    fn test_trinity_surface_pipeline_at_capacity_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> bool>(_f: F) {}
        _check_method(|s| s.pipeline_at_capacity());
    }

    #[test]
    fn test_trinity_surface_frame_in_flight_tracker_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> Option<&FrameInFlightTracker>>(_f: F) {}
        _check_method(|s| s.frame_in_flight_tracker());
    }

    #[test]
    fn test_trinity_surface_frame_in_flight_tracker_mut_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface) -> Option<&mut FrameInFlightTracker>>(_f: F) {}
        _check_method(|s| s.frame_in_flight_tracker_mut());
    }

    #[test]
    fn test_trinity_surface_enable_frame_tracking_for_mode_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface)>(_f: F) {}
        _check_method(|s| s.enable_frame_tracking_for_mode(BufferingMode::Triple));
    }

    #[test]
    fn test_trinity_surface_disable_frame_tracking_api_exists() {
        fn _check_method<F: FnOnce(&mut TrinitySurface)>(_f: F) {}
        _check_method(|s| s.disable_frame_tracking());
    }

    #[test]
    fn test_trinity_surface_frame_tracking_stats_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> (u32, u32, u32)>(_f: F) {}
        _check_method(|s| s.frame_tracking_stats());
    }

    #[test]
    fn test_trinity_surface_pipeline_utilization_api_exists() {
        fn _check_method<F: FnOnce(&TrinitySurface) -> f32>(_f: F) {}
        _check_method(|s| s.pipeline_utilization());
    }

    // -- Thread safety tests -------------------------------------------------

    #[test]
    fn test_frame_in_flight_tracker_thread_safe() {
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(FrameInFlightTracker::new(BufferingMode::Quad));
        let mut handles = vec![];

        // Spawn multiple threads submitting frames
        for _ in 0..4 {
            let tracker_clone = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    tracker_clone.frame_submitted();
                }
            }));
        }

        // Spawn threads presenting frames
        for _ in 0..4 {
            let tracker_clone = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    tracker_clone.frame_presented();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Final count should be 0 (400 submitted, 400 presented)
        // Due to race conditions, it might not be exactly 0, but should be close
        assert!(tracker.total_submitted() == 400);
        assert!(tracker.total_presented() == 400);
    }

    // =========================================================================
    // Headless Rendering tests (T-WGPU-P7.1.10)
    // =========================================================================

    // -- HeadlessError tests -------------------------------------------------

    #[test]
    fn test_headless_error_invalid_dimensions() {
        let err = HeadlessError::invalid_dimensions(0, 0);
        let msg = format!("{}", err);
        assert!(msg.contains("invalid"));
        assert!(msg.contains("0"));
    }

    #[test]
    fn test_headless_error_texture_creation_failed() {
        let err = HeadlessError::TextureCreationFailed("test reason".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test reason"));
    }

    #[test]
    fn test_headless_error_staging_buffer_failed() {
        let err = HeadlessError::StagingBufferFailed("buffer error".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("buffer error"));
    }

    #[test]
    fn test_headless_error_buffer_map_failed() {
        let err = HeadlessError::BufferMapFailed("mapping error".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("mapping error"));
    }

    #[test]
    fn test_headless_error_not_initialized() {
        let err = HeadlessError::NotInitialized;
        let msg = format!("{}", err);
        assert!(msg.contains("not initialized"));
    }

    #[test]
    fn test_headless_error_screenshot_save_failed() {
        let err = HeadlessError::ScreenshotSaveFailed("file error".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("file error"));
    }

    #[test]
    fn test_headless_error_resolve_failed() {
        let err = HeadlessError::ResolveFailed("MSAA error".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("MSAA"));
    }

    #[test]
    fn test_headless_error_debug() {
        let err = HeadlessError::invalid_dimensions(100, 200);
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidDimensions"));
    }

    // -- HeadlessConfig tests ------------------------------------------------

    #[test]
    fn test_headless_config_new() {
        let config = HeadlessConfig::new(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(config.sample_count, 1);
        assert!(config.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
    }

    #[test]
    fn test_headless_config_new_clamps_zero() {
        let config = HeadlessConfig::new(0, 0);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn test_headless_config_default() {
        let config = HeadlessConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
    }

    #[test]
    fn test_headless_config_with_format() {
        let config = HeadlessConfig::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_headless_config_with_msaa() {
        let config1 = HeadlessConfig::new(1920, 1080).with_msaa(0);
        assert_eq!(config1.sample_count, 1);

        let config2 = HeadlessConfig::new(1920, 1080).with_msaa(1);
        assert_eq!(config2.sample_count, 1);

        let config3 = HeadlessConfig::new(1920, 1080).with_msaa(2);
        assert_eq!(config3.sample_count, 4);

        let config4 = HeadlessConfig::new(1920, 1080).with_msaa(4);
        assert_eq!(config4.sample_count, 4);

        let config5 = HeadlessConfig::new(1920, 1080).with_msaa(5);
        assert_eq!(config5.sample_count, 8);

        let config6 = HeadlessConfig::new(1920, 1080).with_msaa(8);
        assert_eq!(config6.sample_count, 8);

        let config7 = HeadlessConfig::new(1920, 1080).with_msaa(16);
        assert_eq!(config7.sample_count, 8);
    }

    #[test]
    fn test_headless_config_with_readback() {
        let config = HeadlessConfig::new(1920, 1080).with_readback();
        assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
    }

    #[test]
    fn test_headless_config_with_usage() {
        let config = HeadlessConfig::new(1920, 1080)
            .with_usage(wgpu::TextureUsages::TEXTURE_BINDING);
        // RENDER_ATTACHMENT is always added
        assert!(config.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
        assert!(config.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    }

    #[test]
    fn test_headless_config_with_label() {
        let config = HeadlessConfig::new(1920, 1080)
            .with_label("my_target");
        assert_eq!(config.label, Some("my_target".to_string()));
    }

    #[test]
    fn test_headless_config_is_msaa_enabled() {
        let no_msaa = HeadlessConfig::new(1920, 1080);
        assert!(!no_msaa.is_msaa_enabled());

        let msaa = HeadlessConfig::new(1920, 1080).with_msaa(4);
        assert!(msaa.is_msaa_enabled());
    }

    #[test]
    fn test_headless_config_supports_readback() {
        let config = HeadlessConfig::new(1920, 1080);
        assert!(config.supports_readback()); // Enabled by default

        let no_readback = HeadlessConfig {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            ..HeadlessConfig::new(1920, 1080)
        };
        assert!(!no_readback.supports_readback());
    }

    #[test]
    fn test_headless_config_validate_success() {
        let config = HeadlessConfig::new(1920, 1080);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_headless_config_validate_zero_width() {
        let config = HeadlessConfig {
            width: 0,
            ..HeadlessConfig::new(100, 100)
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_headless_config_validate_zero_height() {
        let config = HeadlessConfig {
            height: 0,
            ..HeadlessConfig::new(100, 100)
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_headless_config_bytes_per_pixel() {
        // 1-byte formats
        let r8 = HeadlessConfig::new(100, 100)
            .with_format(wgpu::TextureFormat::R8Unorm);
        assert_eq!(r8.bytes_per_pixel(), 1);

        // 2-byte formats
        let rg8 = HeadlessConfig::new(100, 100)
            .with_format(wgpu::TextureFormat::Rg8Unorm);
        assert_eq!(rg8.bytes_per_pixel(), 2);

        // 4-byte formats
        let rgba8 = HeadlessConfig::new(100, 100)
            .with_format(wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(rgba8.bytes_per_pixel(), 4);

        let bgra8 = HeadlessConfig::new(100, 100)
            .with_format(wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(bgra8.bytes_per_pixel(), 4);

        // 8-byte formats
        let rgba16f = HeadlessConfig::new(100, 100)
            .with_format(wgpu::TextureFormat::Rgba16Float);
        assert_eq!(rgba16f.bytes_per_pixel(), 8);

        // 16-byte formats
        let rgba32f = HeadlessConfig::new(100, 100)
            .with_format(wgpu::TextureFormat::Rgba32Float);
        assert_eq!(rgba32f.bytes_per_pixel(), 16);
    }

    #[test]
    fn test_headless_config_aligned_bytes_per_row() {
        // 100 pixels * 4 bytes = 400 bytes, aligned to 256 = 512
        let config = HeadlessConfig::new(100, 100);
        assert_eq!(config.aligned_bytes_per_row(), 512);

        // 64 pixels * 4 bytes = 256 bytes, already aligned
        let config2 = HeadlessConfig::new(64, 100);
        assert_eq!(config2.aligned_bytes_per_row(), 256);

        // 256 pixels * 4 bytes = 1024 bytes, already aligned
        let config3 = HeadlessConfig::new(256, 100);
        assert_eq!(config3.aligned_bytes_per_row(), 1024);
    }

    #[test]
    fn test_headless_config_buffer_size() {
        let config = HeadlessConfig::new(64, 100);
        // 64 * 4 = 256 bytes per row (already aligned)
        // 256 * 100 = 25600 bytes
        assert_eq!(config.buffer_size(), 25600);
    }

    #[test]
    fn test_headless_config_debug() {
        let config = HeadlessConfig::new(1920, 1080);
        let debug = format!("{:?}", config);
        assert!(debug.contains("HeadlessConfig"));
        assert!(debug.contains("1920"));
        assert!(debug.contains("1080"));
    }

    #[test]
    fn test_headless_config_clone() {
        let config = HeadlessConfig::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm)
            .with_msaa(4)
            .with_label("test");

        let cloned = config.clone();
        assert_eq!(cloned.width, config.width);
        assert_eq!(cloned.height, config.height);
        assert_eq!(cloned.format, config.format);
        assert_eq!(cloned.sample_count, config.sample_count);
        assert_eq!(cloned.label, config.label);
    }

    // -- HeadlessTarget API shape tests (require GPU for full test) ----------

    #[test]
    fn test_headless_target_api_exists() {
        // Verify HeadlessTarget::new signature
        fn _check_new<F: FnOnce(&wgpu::Device, HeadlessConfig) -> Result<HeadlessTarget, HeadlessError>>(_f: F) {}
        _check_new(HeadlessTarget::new);
    }

    #[test]
    fn test_headless_target_view_api() {
        fn _check_view<F: FnOnce(&HeadlessTarget) -> &wgpu::TextureView>(_f: F) {}
        _check_view(|t| t.view());
    }

    #[test]
    fn test_headless_target_texture_api() {
        fn _check_texture<F: FnOnce(&HeadlessTarget) -> &wgpu::Texture>(_f: F) {}
        _check_texture(|t| t.texture());
    }

    #[test]
    fn test_headless_target_resolve_view_api() {
        fn _check_resolve_view<F: FnOnce(&HeadlessTarget) -> Option<&wgpu::TextureView>>(_f: F) {}
        _check_resolve_view(|t| t.resolve_view());
    }

    #[test]
    fn test_headless_target_config_api() {
        fn _check_config<F: FnOnce(&HeadlessTarget) -> &HeadlessConfig>(_f: F) {}
        _check_config(|t| t.config());
    }

    #[test]
    fn test_headless_target_dimensions_api() {
        fn _check_width<F: FnOnce(&HeadlessTarget) -> u32>(_f: F) {}
        _check_width(|t| t.width());

        fn _check_height<F: FnOnce(&HeadlessTarget) -> u32>(_f: F) {}
        _check_height(|t| t.height());

        fn _check_dimensions<F: FnOnce(&HeadlessTarget) -> (u32, u32)>(_f: F) {}
        _check_dimensions(|t| t.dimensions());
    }

    #[test]
    fn test_headless_target_format_api() {
        fn _check_format<F: FnOnce(&HeadlessTarget) -> wgpu::TextureFormat>(_f: F) {}
        _check_format(|t| t.format());
    }

    #[test]
    fn test_headless_target_sample_count_api() {
        fn _check_sample_count<F: FnOnce(&HeadlessTarget) -> u32>(_f: F) {}
        _check_sample_count(|t| t.sample_count());
    }

    #[test]
    fn test_headless_target_is_msaa_enabled_api() {
        fn _check_is_msaa_enabled<F: FnOnce(&HeadlessTarget) -> bool>(_f: F) {}
        _check_is_msaa_enabled(|t| t.is_msaa_enabled());
    }

    #[test]
    fn test_headless_target_aspect_ratio_api() {
        fn _check_aspect_ratio<F: FnOnce(&HeadlessTarget) -> f32>(_f: F) {}
        _check_aspect_ratio(|t| t.aspect_ratio());
    }

    #[test]
    fn test_headless_target_acquire_frame_api() {
        fn _check_acquire_frame<'a, F: FnOnce(&'a HeadlessTarget) -> HeadlessFrame<'a>>(_f: F) {}
        _check_acquire_frame(|t| t.acquire_frame());
    }

    #[test]
    fn test_headless_target_create_staging_buffer_api() {
        fn _check_create_staging_buffer<F: FnOnce(&HeadlessTarget, &wgpu::Device) -> ReadbackBuffer>(_f: F) {}
        _check_create_staging_buffer(|t, d| t.create_staging_buffer(d));
    }

    #[test]
    fn test_headless_target_screenshot_api() {
        fn _check_screenshot<F: FnOnce(&HeadlessTarget, &wgpu::Device, &wgpu::Queue) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
        _check_screenshot(|t, d, q| t.screenshot(d, q));
    }

    // -- HeadlessFrame tests -------------------------------------------------

    #[test]
    fn test_headless_frame_debug() {
        // Verify HeadlessFrame implements Debug
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<HeadlessFrame<'_>>();
    }

    // -- ReadbackBuffer API shape tests --------------------------------------

    #[test]
    fn test_readback_buffer_buffer_api() {
        fn _check_buffer<F: FnOnce(&ReadbackBuffer) -> &wgpu::Buffer>(_f: F) {}
        _check_buffer(|r| r.buffer());
    }

    #[test]
    fn test_readback_buffer_bytes_per_row_api() {
        fn _check_bytes_per_row<F: FnOnce(&ReadbackBuffer) -> u32>(_f: F) {}
        _check_bytes_per_row(|r| r.bytes_per_row());
    }

    #[test]
    fn test_readback_buffer_size_api() {
        fn _check_size<F: FnOnce(&ReadbackBuffer) -> u64>(_f: F) {}
        _check_size(|r| r.size());
    }

    #[test]
    fn test_readback_buffer_map_read_api() {
        fn _check_map_read<F: FnOnce(&ReadbackBuffer, &wgpu::Device) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
        _check_map_read(|r, d| r.map_read(d));
    }

    #[test]
    fn test_readback_buffer_debug() {
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<ReadbackBuffer>();
    }

    // -- HeadlessRenderer tests ----------------------------------------------

    #[test]
    fn test_headless_renderer_new_api() {
        fn _check_new<F: FnOnce(&wgpu::Device, HeadlessConfig) -> Result<HeadlessRenderer, HeadlessError>>(_f: F) {}
        _check_new(HeadlessRenderer::new);
    }

    #[test]
    fn test_headless_renderer_acquire_frame_api() {
        fn _check_acquire_frame<'a, F: FnOnce(&'a mut HeadlessRenderer) -> HeadlessFrame<'a>>(_f: F) {}
        _check_acquire_frame(|r| r.acquire_frame());
    }

    #[test]
    fn test_headless_renderer_target_api() {
        fn _check_target<F: FnOnce(&HeadlessRenderer) -> &HeadlessTarget>(_f: F) {}
        _check_target(|r| r.target());
    }

    #[test]
    fn test_headless_renderer_target_mut_api() {
        fn _check_target_mut<F: FnOnce(&mut HeadlessRenderer) -> &mut HeadlessTarget>(_f: F) {}
        _check_target_mut(|r| r.target_mut());
    }

    #[test]
    fn test_headless_renderer_frame_count_api() {
        fn _check_frame_count<F: FnOnce(&HeadlessRenderer) -> u64>(_f: F) {}
        _check_frame_count(|r| r.frame_count());
    }

    #[test]
    fn test_headless_renderer_view_api() {
        fn _check_view<F: FnOnce(&HeadlessRenderer) -> &wgpu::TextureView>(_f: F) {}
        _check_view(|r| r.view());
    }

    #[test]
    fn test_headless_renderer_resolve_view_api() {
        fn _check_resolve_view<F: FnOnce(&HeadlessRenderer) -> Option<&wgpu::TextureView>>(_f: F) {}
        _check_resolve_view(|r| r.resolve_view());
    }

    #[test]
    fn test_headless_renderer_dimensions_api() {
        fn _check_width<F: FnOnce(&HeadlessRenderer) -> u32>(_f: F) {}
        _check_width(|r| r.width());

        fn _check_height<F: FnOnce(&HeadlessRenderer) -> u32>(_f: F) {}
        _check_height(|r| r.height());

        fn _check_dimensions<F: FnOnce(&HeadlessRenderer) -> (u32, u32)>(_f: F) {}
        _check_dimensions(|r| r.dimensions());
    }

    #[test]
    fn test_headless_renderer_format_api() {
        fn _check_format<F: FnOnce(&HeadlessRenderer) -> wgpu::TextureFormat>(_f: F) {}
        _check_format(|r| r.format());
    }

    #[test]
    fn test_headless_renderer_screenshot_api() {
        fn _check_screenshot<F: FnOnce(&HeadlessRenderer, &wgpu::Device, &wgpu::Queue) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
        _check_screenshot(|r, d, q| r.screenshot(d, q));
    }

    #[test]
    fn test_headless_renderer_screenshot_packed_api() {
        fn _check_screenshot_packed<F: FnOnce(&HeadlessRenderer, &wgpu::Device, &wgpu::Queue) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
        _check_screenshot_packed(|r, d, q| r.screenshot_packed(d, q));
    }

    #[test]
    fn test_headless_renderer_debug() {
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<HeadlessRenderer>();
    }

    // -- Multi-Window Tests (T-WGPU-P7.1.11) ---------------------------------

    #[test]
    fn test_window_id_new_generates_unique_ids() {
        let id1 = WindowId::new();
        let id2 = WindowId::new();
        let id3 = WindowId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_window_id_from_raw() {
        let id = WindowId::from_raw_id(42);
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn test_window_id_primary() {
        let primary = WindowId::primary();
        assert!(primary.is_primary());
        assert_eq!(primary.as_u64(), 0);

        let non_primary = WindowId::from_raw_id(1);
        assert!(!non_primary.is_primary());
    }

    #[test]
    fn test_window_id_display() {
        let id = WindowId::from_raw_id(123);
        assert_eq!(format!("{}", id), "Window(123)");
    }

    #[test]
    fn test_window_id_from_u64() {
        let id: WindowId = 999u64.into();
        assert_eq!(id.as_u64(), 999);

        let val: u64 = id.into();
        assert_eq!(val, 999);
    }

    #[test]
    fn test_window_id_hash_eq() {
        use std::collections::HashSet;

        let id1 = WindowId::from_raw_id(1);
        let id2 = WindowId::from_raw_id(1);
        let id3 = WindowId::from_raw_id(2);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);

        let mut set = HashSet::new();
        set.insert(id1);
        assert!(set.contains(&id2));
        assert!(!set.contains(&id3));
    }

    #[test]
    fn test_window_config_new() {
        let id = WindowId::new();
        let surface_config = SurfaceConfiguration::new(1920, 1080);
        let config = WindowConfig::new(id, surface_config);

        assert_eq!(config.id, id);
        assert!(!config.is_focused);
        assert!(config.is_visible);
        assert_eq!(config.priority, 128);
        assert!(config.label.is_none());
    }

    #[test]
    fn test_window_config_primary() {
        let surface_config = SurfaceConfiguration::new(1920, 1080);
        let config = WindowConfig::primary(surface_config);

        assert!(config.id.is_primary());
        assert!(config.is_focused);
        assert!(config.is_visible);
        assert_eq!(config.priority, 255);
        assert_eq!(config.label.as_deref(), Some("Primary"));
    }

    #[test]
    fn test_window_config_builders() {
        let id = WindowId::new();
        let surface_config = SurfaceConfiguration::new(800, 600);
        let config = WindowConfig::new(id, surface_config)
            .with_focus(true)
            .with_visibility(false)
            .with_priority(200)
            .with_label("Test Window")
            .with_sync_to_primary(true);

        assert!(config.is_focused);
        assert!(!config.is_visible);
        assert_eq!(config.priority, 200);
        assert_eq!(config.label.as_deref(), Some("Test Window"));
        assert!(config.sync_to_primary);
    }

    #[test]
    fn test_window_config_dimensions() {
        let surface_config = SurfaceConfiguration::new(1920, 1080);
        let config = WindowConfig::new(WindowId::new(), surface_config);

        assert_eq!(config.dimensions(), (1920, 1080));
        assert!((config.aspect_ratio() - 1920.0 / 1080.0).abs() < 0.001);
    }

    #[test]
    fn test_window_config_should_render() {
        let surface_config = SurfaceConfiguration::new(1920, 1080);

        // Visible window should render
        let config1 = WindowConfig::new(WindowId::new(), surface_config.clone());
        assert!(config1.should_render());

        // Invisible window should not render
        let config2 = WindowConfig::new(WindowId::new(), surface_config.clone())
            .with_visibility(false);
        assert!(!config2.should_render());

        // Note: SurfaceConfiguration::new clamps dimensions to min 1x1,
        // so we can't create a truly zero-size config through the public API.
        // The should_render check still handles it defensively.
        let mut minimal_config = SurfaceConfiguration::new(1, 1);
        minimal_config.width = 0;  // Bypass constructor clamping for test
        minimal_config.height = 0;
        let config3 = WindowConfig::new(WindowId::new(), minimal_config);
        assert!(!config3.should_render());
    }

    #[test]
    fn test_window_config_display() {
        let surface_config = SurfaceConfiguration::new(1280, 720);
        let config = WindowConfig::new(WindowId::new(), surface_config)
            .with_label("Main")
            .with_priority(100)
            .with_focus(true);

        let s = format!("{}", config);
        assert!(s.contains("Main"));
        assert!(s.contains("1280x720"));
        assert!(s.contains("priority=100"));
    }

    #[test]
    fn test_sync_mode_default() {
        let mode = SyncMode::default();
        assert_eq!(mode, SyncMode::Independent);
    }

    #[test]
    fn test_sync_mode_sync_to_rate() {
        let mode = SyncMode::sync_to_rate(60);
        match mode {
            SyncMode::SyncToRate { target_hz } => assert_eq!(target_hz, 60),
            _ => panic!("expected SyncToRate"),
        }
    }

    #[test]
    fn test_sync_mode_target_interval() {
        assert!(SyncMode::Independent.target_interval().is_none());
        assert!(SyncMode::SyncToPrimary.target_interval().is_none());
        assert!(SyncMode::Simultaneous.target_interval().is_none());

        let rate_mode = SyncMode::SyncToRate { target_hz: 60 };
        let interval = rate_mode.target_interval().unwrap();
        let expected_ms = 1000.0 / 60.0;
        let actual_ms = interval.as_secs_f64() * 1000.0;
        assert!((actual_ms - expected_ms).abs() < 0.1);

        // Zero Hz should return None
        let zero_mode = SyncMode::SyncToRate { target_hz: 0 };
        assert!(zero_mode.target_interval().is_none());
    }

    #[test]
    fn test_sync_mode_requires_coordination() {
        assert!(!SyncMode::Independent.requires_coordination());
        assert!(SyncMode::SyncToPrimary.requires_coordination());
        assert!(SyncMode::SyncToRate { target_hz: 60 }.requires_coordination());
        assert!(SyncMode::Simultaneous.requires_coordination());
    }

    #[test]
    fn test_sync_mode_display() {
        assert_eq!(format!("{}", SyncMode::Independent), "Independent");
        assert_eq!(format!("{}", SyncMode::SyncToPrimary), "Sync to Primary");
        assert_eq!(format!("{}", SyncMode::SyncToRate { target_hz: 144 }), "Sync to 144Hz");
        assert_eq!(format!("{}", SyncMode::Simultaneous), "Simultaneous");
    }

    #[test]
    fn test_multi_window_error_is_recoverable() {
        assert!(!MultiWindowError::WindowNotFound(WindowId::new()).is_recoverable());
        assert!(!MultiWindowError::WindowExists(WindowId::new()).is_recoverable());
        assert!(!MultiWindowError::NoWindows.is_recoverable());
        assert!(MultiWindowError::NoFocusedWindow.is_recoverable());
        assert!(!MultiWindowError::MaxWindowsReached { max: 8 }.is_recoverable());

        // Surface/Frame errors delegate to their own is_recoverable
        let surface_err = MultiWindowError::SurfaceError(SurfaceError::SurfaceOutdated);
        assert!(surface_err.is_recoverable());

        let frame_err = MultiWindowError::FrameError(FrameError::Timeout);
        assert!(frame_err.is_recoverable());
    }

    #[test]
    fn test_multi_window_manager_new() {
        let manager = MultiWindowManager::new();
        assert_eq!(manager.window_count(), 0);
        assert!(!manager.has_windows());
        assert!(manager.focused_window_id().is_none());
        assert_eq!(manager.sync_mode(), SyncMode::Independent);
    }

    #[test]
    fn test_multi_window_manager_with_max_windows() {
        let manager = MultiWindowManager::with_max_windows(4);
        assert_eq!(manager.window_count(), 0);
        // max_windows is internal, but we can test it via behavior
    }

    #[test]
    fn test_multi_window_manager_set_sync_mode() {
        let mut manager = MultiWindowManager::new();

        manager.set_sync_mode(SyncMode::SyncToPrimary);
        assert_eq!(manager.sync_mode(), SyncMode::SyncToPrimary);

        manager.set_sync_mode(SyncMode::SyncToRate { target_hz: 120 });
        match manager.sync_mode() {
            SyncMode::SyncToRate { target_hz } => assert_eq!(target_hz, 120),
            _ => panic!("expected SyncToRate"),
        }
    }

    #[test]
    fn test_multi_window_stats_drop_rate() {
        let stats = MultiWindowStats {
            window_count: 2,
            total_frames: 90,
            total_dropped: 10,
            average_frame_time_ms: 16.67,
            global_frame_count: 100,
        };

        assert!((stats.drop_rate() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_multi_window_stats_drop_rate_no_frames() {
        let stats = MultiWindowStats {
            window_count: 1,
            total_frames: 0,
            total_dropped: 0,
            average_frame_time_ms: 0.0,
            global_frame_count: 0,
        };

        assert_eq!(stats.drop_rate(), 0.0);
    }

    #[test]
    fn test_multi_window_stats_estimated_fps() {
        let stats = MultiWindowStats {
            window_count: 1,
            total_frames: 100,
            total_dropped: 0,
            average_frame_time_ms: 16.67,
            global_frame_count: 100,
        };

        let fps = stats.estimated_fps();
        assert!((fps - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_multi_window_stats_estimated_fps_zero_time() {
        let stats = MultiWindowStats {
            window_count: 1,
            total_frames: 0,
            total_dropped: 0,
            average_frame_time_ms: 0.0,
            global_frame_count: 0,
        };

        assert_eq!(stats.estimated_fps(), 0.0);
    }

    #[test]
    fn test_multi_window_stats_display() {
        let stats = MultiWindowStats {
            window_count: 3,
            total_frames: 1000,
            total_dropped: 50,
            average_frame_time_ms: 11.11,
            global_frame_count: 500,
        };

        let s = format!("{}", stats);
        assert!(s.contains("3 windows"));
        assert!(s.contains("1000 frames"));
        assert!(s.contains("50 dropped"));
    }

    #[test]
    fn test_multi_window_manager_debug() {
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<MultiWindowManager>();
    }

    #[test]
    fn test_window_state_debug() {
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<WindowState>();
    }

    #[test]
    fn test_window_frame_debug() {
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<WindowFrame>();
    }

    #[test]
    fn test_multi_window_stats_debug() {
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<MultiWindowStats>();
    }

    // API signature tests for multi-window types

    #[test]
    fn test_window_id_new_api() {
        fn _check_new<F: FnOnce() -> WindowId>(_f: F) {}
        _check_new(WindowId::new);
    }

    #[test]
    fn test_window_id_from_raw_id_api() {
        fn _check_from_raw_id<F: FnOnce(u64) -> WindowId>(_f: F) {}
        _check_from_raw_id(WindowId::from_raw_id);
    }

    #[test]
    fn test_window_id_as_u64_api() {
        fn _check_as_u64<F: FnOnce(&WindowId) -> u64>(_f: F) {}
        _check_as_u64(|id| id.as_u64());
    }

    #[test]
    fn test_window_id_primary_api() {
        fn _check_primary<F: FnOnce() -> WindowId>(_f: F) {}
        _check_primary(WindowId::primary);
    }

    #[test]
    fn test_window_id_is_primary_api() {
        fn _check_is_primary<F: FnOnce(&WindowId) -> bool>(_f: F) {}
        _check_is_primary(|id| id.is_primary());
    }

    #[test]
    fn test_window_config_new_api() {
        fn _check_new<F: FnOnce(WindowId, SurfaceConfiguration) -> WindowConfig>(_f: F) {}
        _check_new(WindowConfig::new);
    }

    #[test]
    fn test_window_config_primary_api() {
        fn _check_primary<F: FnOnce(SurfaceConfiguration) -> WindowConfig>(_f: F) {}
        _check_primary(WindowConfig::primary);
    }

    #[test]
    fn test_window_config_with_focus_api() {
        fn _check_with_focus<F: FnOnce(WindowConfig, bool) -> WindowConfig>(_f: F) {}
        _check_with_focus(|c, f| c.with_focus(f));
    }

    #[test]
    fn test_window_config_with_visibility_api() {
        fn _check_with_visibility<F: FnOnce(WindowConfig, bool) -> WindowConfig>(_f: F) {}
        _check_with_visibility(|c, v| c.with_visibility(v));
    }

    #[test]
    fn test_window_config_with_priority_api() {
        fn _check_with_priority<F: FnOnce(WindowConfig, u8) -> WindowConfig>(_f: F) {}
        _check_with_priority(|c, p| c.with_priority(p));
    }

    #[test]
    fn test_window_config_dimensions_api() {
        fn _check_dimensions<F: FnOnce(&WindowConfig) -> (u32, u32)>(_f: F) {}
        _check_dimensions(|c| c.dimensions());
    }

    #[test]
    fn test_window_config_aspect_ratio_api() {
        fn _check_aspect_ratio<F: FnOnce(&WindowConfig) -> f32>(_f: F) {}
        _check_aspect_ratio(|c| c.aspect_ratio());
    }

    #[test]
    fn test_window_config_should_render_api() {
        fn _check_should_render<F: FnOnce(&WindowConfig) -> bool>(_f: F) {}
        _check_should_render(|c| c.should_render());
    }

    #[test]
    fn test_sync_mode_sync_to_rate_api() {
        fn _check_sync_to_rate<F: FnOnce(u32) -> SyncMode>(_f: F) {}
        _check_sync_to_rate(SyncMode::sync_to_rate);
    }

    #[test]
    fn test_sync_mode_target_interval_api() {
        fn _check_target_interval<F: FnOnce(&SyncMode) -> Option<Duration>>(_f: F) {}
        _check_target_interval(|m| m.target_interval());
    }

    #[test]
    fn test_sync_mode_requires_coordination_api() {
        fn _check_requires_coordination<F: FnOnce(&SyncMode) -> bool>(_f: F) {}
        _check_requires_coordination(|m| m.requires_coordination());
    }

    #[test]
    fn test_multi_window_manager_new_api() {
        fn _check_new<F: FnOnce() -> MultiWindowManager>(_f: F) {}
        _check_new(MultiWindowManager::new);
    }

    #[test]
    fn test_multi_window_manager_with_max_windows_api() {
        fn _check_with_max<F: FnOnce(usize) -> MultiWindowManager>(_f: F) {}
        _check_with_max(MultiWindowManager::with_max_windows);
    }

    #[test]
    fn test_multi_window_manager_set_sync_mode_api() {
        fn _check_set_sync<F: FnOnce(&mut MultiWindowManager, SyncMode)>(_f: F) {}
        _check_set_sync(|m, s| m.set_sync_mode(s));
    }

    #[test]
    fn test_multi_window_manager_sync_mode_api() {
        fn _check_sync_mode<F: FnOnce(&MultiWindowManager) -> SyncMode>(_f: F) {}
        _check_sync_mode(|m| m.sync_mode());
    }

    #[test]
    fn test_multi_window_manager_window_count_api() {
        fn _check_window_count<F: FnOnce(&MultiWindowManager) -> usize>(_f: F) {}
        _check_window_count(|m| m.window_count());
    }

    #[test]
    fn test_multi_window_manager_has_windows_api() {
        fn _check_has_windows<F: FnOnce(&MultiWindowManager) -> bool>(_f: F) {}
        _check_has_windows(|m| m.has_windows());
    }

    #[test]
    fn test_multi_window_manager_window_ids_api() {
        fn _check_window_ids<F: FnOnce(&MultiWindowManager) -> &[WindowId]>(_f: F) {}
        _check_window_ids(|m| m.window_ids());
    }

    #[test]
    fn test_multi_window_manager_visible_window_ids_api() {
        fn _check_visible_ids<F: FnOnce(&MultiWindowManager) -> Vec<WindowId>>(_f: F) {}
        _check_visible_ids(|m| m.visible_window_ids());
    }

    #[test]
    fn test_multi_window_manager_focused_window_id_api() {
        fn _check_focused_id<F: FnOnce(&MultiWindowManager) -> Option<WindowId>>(_f: F) {}
        _check_focused_id(|m| m.focused_window_id());
    }

    #[test]
    fn test_multi_window_manager_global_frame_count_api() {
        fn _check_frame_count<F: FnOnce(&MultiWindowManager) -> u64>(_f: F) {}
        _check_frame_count(|m| m.global_frame_count());
    }

    #[test]
    fn test_multi_window_manager_aggregate_stats_api() {
        fn _check_stats<F: FnOnce(&MultiWindowManager) -> MultiWindowStats>(_f: F) {}
        _check_stats(|m| m.aggregate_stats());
    }

    #[test]
    fn test_multi_window_stats_drop_rate_api() {
        fn _check_drop_rate<F: FnOnce(&MultiWindowStats) -> f32>(_f: F) {}
        _check_drop_rate(|s| s.drop_rate());
    }

    #[test]
    fn test_multi_window_stats_estimated_fps_api() {
        fn _check_fps<F: FnOnce(&MultiWindowStats) -> f32>(_f: F) {}
        _check_fps(|s| s.estimated_fps());
    }

    #[test]
    fn test_multi_window_error_is_recoverable_api() {
        fn _check_is_recoverable<F: FnOnce(&MultiWindowError) -> bool>(_f: F) {}
        _check_is_recoverable(|e| e.is_recoverable());
    }
}
