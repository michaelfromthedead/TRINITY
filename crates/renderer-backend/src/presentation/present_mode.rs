//! Present mode selection for the TRINITY renderer.
//!
//! This module provides comprehensive present mode selection logic with
//! preference-based fallback chains, capability queries, and mode-specific
//! information for optimal display synchronization.
//!
//! # Components
//!
//! - [`PresentModePreference`] - User preference with automatic fallback chains
//! - [`LatencyLevel`] - Latency classification for present modes
//! - [`PresentModeInfo`] - Detailed mode information with capability flags
//! - [`PresentModeSelector`] - Hardware capability queries and mode selection
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::presentation::present_mode::{
//!     PresentModePreference, PresentModeSelector, LatencyLevel,
//! };
//!
//! // Create selector from surface capabilities
//! let selector = PresentModeSelector::from_surface_capabilities(&caps);
//!
//! // Select best mode for gaming (low latency)
//! let gaming_mode = selector.select(PresentModePreference::LowLatency);
//!
//! // Check latency level
//! let info = selector.info(gaming_mode);
//! assert!(info.latency == LatencyLevel::Lowest || info.latency == LatencyLevel::Low);
//! ```

use std::fmt;

// ============================================================================
// PresentModePreference
// ============================================================================

/// Present mode preference with automatic fallback chain.
///
/// This enum represents the user's desired presentation behavior. Each preference
/// includes a prioritized fallback chain to handle cases where the preferred mode
/// is not supported by the hardware.
///
/// # Fallback Chains
///
/// - **VSync**: Fifo (always available)
/// - **LowLatency**: Mailbox > Immediate > Fifo
/// - **NoVSync**: Immediate > Mailbox > Fifo
/// - **Adaptive**: FifoRelaxed > Fifo
/// - **PowerSaving**: Fifo (minimize GPU usage)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresentModePreference {
    /// VSync enabled - synchronized to display refresh rate.
    ///
    /// Uses Fifo mode which is always available on all platforms.
    /// Provides no tearing but may have higher input latency.
    ///
    /// **Fallback chain**: Fifo (always available)
    VSync,

    /// Lowest possible latency with smooth presentation.
    ///
    /// Prefers Mailbox (triple buffering) for low latency without tearing,
    /// falls back to Immediate if Mailbox is unavailable.
    ///
    /// **Fallback chain**: Mailbox > Immediate > Fifo
    LowLatency,

    /// No VSync - uncapped framerate, minimal latency.
    ///
    /// Prefers Immediate mode for absolute lowest latency,
    /// even if it may cause visible tearing.
    ///
    /// **Fallback chain**: Immediate > Mailbox > Fifo
    NoVSync,

    /// Adaptive VSync - drops to immediate when behind.
    ///
    /// Uses FifoRelaxed for smooth presentation when above refresh rate,
    /// with automatic fallback to avoid stuttering when below.
    ///
    /// **Fallback chain**: FifoRelaxed > Fifo
    Adaptive,

    /// Power saving mode - minimize GPU usage.
    ///
    /// Uses Fifo mode which allows the GPU to idle between frames,
    /// reducing power consumption and heat generation.
    ///
    /// **Fallback chain**: Fifo (always available)
    PowerSaving,
}

impl PresentModePreference {
    /// Returns the fallback chain for this preference.
    ///
    /// The chain is ordered from most preferred to least preferred.
    /// Fifo is always the final fallback as it's guaranteed to be available.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let chain = PresentModePreference::LowLatency.fallback_chain();
    /// assert_eq!(chain, &[
    ///     wgpu::PresentMode::Mailbox,
    ///     wgpu::PresentMode::Immediate,
    ///     wgpu::PresentMode::Fifo,
    /// ]);
    /// ```
    pub const fn fallback_chain(&self) -> &'static [wgpu::PresentMode] {
        match self {
            PresentModePreference::VSync => &[wgpu::PresentMode::Fifo],
            PresentModePreference::LowLatency => &[
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Immediate,
                wgpu::PresentMode::Fifo,
            ],
            PresentModePreference::NoVSync => &[
                wgpu::PresentMode::Immediate,
                wgpu::PresentMode::Mailbox,
                wgpu::PresentMode::Fifo,
            ],
            PresentModePreference::Adaptive => &[
                wgpu::PresentMode::FifoRelaxed,
                wgpu::PresentMode::Fifo,
            ],
            PresentModePreference::PowerSaving => &[wgpu::PresentMode::Fifo],
        }
    }

    /// Returns the default preference for the current platform.
    ///
    /// Platform-specific defaults:
    /// - **Desktop (Windows, Linux, macOS)**: VSync for smooth presentation
    /// - **Mobile (iOS, Android)**: PowerSaving to preserve battery
    /// - **Web**: VSync for consistent behavior
    ///
    /// # Example
    ///
    /// ```ignore
    /// let default = PresentModePreference::default_for_platform();
    /// // On desktop: VSync
    /// // On mobile: PowerSaving
    /// ```
    pub const fn default_for_platform() -> Self {
        #[cfg(any(target_os = "ios", target_os = "android"))]
        {
            PresentModePreference::PowerSaving
        }
        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        {
            PresentModePreference::VSync
        }
    }

    /// Returns a human-readable name for this preference.
    pub const fn name(self) -> &'static str {
        match self {
            PresentModePreference::VSync => "VSync",
            PresentModePreference::LowLatency => "Low Latency",
            PresentModePreference::NoVSync => "No VSync",
            PresentModePreference::Adaptive => "Adaptive",
            PresentModePreference::PowerSaving => "Power Saving",
        }
    }

    /// Returns a description of this preference.
    pub const fn description(self) -> &'static str {
        match self {
            PresentModePreference::VSync => "Synchronized to display refresh, no tearing",
            PresentModePreference::LowLatency => "Minimal latency with smooth presentation",
            PresentModePreference::NoVSync => "Uncapped framerate, may cause tearing",
            PresentModePreference::Adaptive => "VSync when above refresh rate, drops when below",
            PresentModePreference::PowerSaving => "Power efficient, GPU can idle between frames",
        }
    }

    /// Returns all preference variants in order of latency (lowest first).
    pub const fn latency_order() -> &'static [PresentModePreference] {
        &[
            PresentModePreference::NoVSync,
            PresentModePreference::LowLatency,
            PresentModePreference::Adaptive,
            PresentModePreference::VSync,
            PresentModePreference::PowerSaving,
        ]
    }

    /// Returns all preference variants in order of power efficiency (most efficient first).
    pub const fn power_order() -> &'static [PresentModePreference] {
        &[
            PresentModePreference::PowerSaving,
            PresentModePreference::VSync,
            PresentModePreference::Adaptive,
            PresentModePreference::LowLatency,
            PresentModePreference::NoVSync,
        ]
    }
}

impl Default for PresentModePreference {
    fn default() -> Self {
        PresentModePreference::VSync
    }
}

impl fmt::Display for PresentModePreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// LatencyLevel
// ============================================================================

/// Latency classification for present modes.
///
/// Represents the relative input latency of different present modes,
/// from lowest (Immediate) to highest (Fifo).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LatencyLevel {
    /// Lowest latency - Immediate mode (0-1 frames).
    ///
    /// Frames are presented immediately without waiting for vsync.
    /// May cause visible tearing but provides the lowest input lag.
    Lowest = 0,

    /// Low latency - Mailbox mode (1 frame).
    ///
    /// Triple buffered with the latest frame always ready.
    /// Good balance of low latency and no tearing.
    Low = 1,

    /// Medium latency - FifoRelaxed mode (1-2 frames).
    ///
    /// Similar to Fifo but can skip frames when behind.
    /// Slightly variable latency based on performance.
    Medium = 2,

    /// High latency - Fifo mode (2+ frames).
    ///
    /// Standard vsync with double or triple buffering.
    /// Consistent but highest input latency.
    High = 3,
}

impl LatencyLevel {
    /// Returns the latency level for a present mode.
    pub const fn from_present_mode(mode: wgpu::PresentMode) -> Self {
        match mode {
            wgpu::PresentMode::Immediate => LatencyLevel::Lowest,
            wgpu::PresentMode::Mailbox => LatencyLevel::Low,
            wgpu::PresentMode::FifoRelaxed => LatencyLevel::Medium,
            wgpu::PresentMode::Fifo => LatencyLevel::High,
            _ => LatencyLevel::High, // Unknown modes default to high
        }
    }

    /// Returns the typical frame latency for this level.
    ///
    /// This is an approximation based on typical driver behavior.
    pub const fn typical_frame_latency(self) -> u32 {
        match self {
            LatencyLevel::Lowest => 0,
            LatencyLevel::Low => 1,
            LatencyLevel::Medium => 1,
            LatencyLevel::High => 2,
        }
    }

    /// Returns estimated latency in milliseconds at 60Hz.
    pub const fn estimated_ms_at_60hz(self) -> f32 {
        match self {
            LatencyLevel::Lowest => 0.0,
            LatencyLevel::Low => 16.67,
            LatencyLevel::Medium => 25.0,
            LatencyLevel::High => 33.33,
        }
    }

    /// Returns true if this latency is suitable for competitive gaming.
    pub const fn is_competitive(self) -> bool {
        matches!(self, LatencyLevel::Lowest | LatencyLevel::Low)
    }

    /// Returns a human-readable name for this level.
    pub const fn name(self) -> &'static str {
        match self {
            LatencyLevel::Lowest => "Lowest",
            LatencyLevel::Low => "Low",
            LatencyLevel::Medium => "Medium",
            LatencyLevel::High => "High",
        }
    }
}

impl fmt::Display for LatencyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// PresentModeInfo
// ============================================================================

/// Detailed information about a present mode.
///
/// Provides comprehensive capability information for a present mode,
/// including latency characteristics, tearing behavior, and power efficiency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresentModeInfo {
    /// The underlying wgpu present mode.
    pub mode: wgpu::PresentMode,
    /// Whether this mode may cause screen tearing.
    pub tears: bool,
    /// Latency classification for this mode.
    pub latency: LatencyLevel,
    /// Whether this mode is power efficient (allows GPU idle).
    pub power_efficient: bool,
}

impl PresentModeInfo {
    /// Create info for a present mode.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let info = PresentModeInfo::from_mode(wgpu::PresentMode::Mailbox);
    /// assert!(!info.tears);
    /// assert_eq!(info.latency, LatencyLevel::Low);
    /// ```
    pub const fn from_mode(mode: wgpu::PresentMode) -> Self {
        match mode {
            wgpu::PresentMode::Immediate => Self {
                mode,
                tears: true,
                latency: LatencyLevel::Lowest,
                power_efficient: false,
            },
            wgpu::PresentMode::Mailbox => Self {
                mode,
                tears: false,
                latency: LatencyLevel::Low,
                power_efficient: false,
            },
            wgpu::PresentMode::Fifo => Self {
                mode,
                tears: false,
                latency: LatencyLevel::High,
                power_efficient: true,
            },
            wgpu::PresentMode::FifoRelaxed => Self {
                mode,
                tears: true, // Can tear when below refresh rate
                latency: LatencyLevel::Medium,
                power_efficient: true,
            },
            // Unknown modes - conservative defaults
            _ => Self {
                mode,
                tears: false,
                latency: LatencyLevel::High,
                power_efficient: false,
            },
        }
    }

    /// Returns true if this mode prevents tearing.
    pub const fn prevents_tearing(&self) -> bool {
        !self.tears
    }

    /// Returns true if this mode is suitable for competitive gaming.
    ///
    /// Competitive modes have low latency and minimal frame delays.
    pub const fn is_competitive(&self) -> bool {
        self.latency.is_competitive()
    }

    /// Returns true if this mode is battery-friendly.
    ///
    /// Power efficient modes allow the GPU to idle between frames.
    pub const fn is_battery_friendly(&self) -> bool {
        self.power_efficient
    }

    /// Returns the human-readable name for this mode.
    pub const fn name(&self) -> &'static str {
        match self.mode {
            wgpu::PresentMode::Immediate => "Immediate",
            wgpu::PresentMode::Mailbox => "Mailbox",
            wgpu::PresentMode::Fifo => "Fifo",
            wgpu::PresentMode::FifoRelaxed => "FifoRelaxed",
            _ => "Unknown",
        }
    }

    /// Returns a description of this mode.
    pub const fn description(&self) -> &'static str {
        match self.mode {
            wgpu::PresentMode::Immediate => "No vsync, lowest latency, may tear",
            wgpu::PresentMode::Mailbox => "Triple buffered, low latency, no tearing",
            wgpu::PresentMode::Fifo => "Standard vsync, no tearing, higher latency",
            wgpu::PresentMode::FifoRelaxed => "Adaptive vsync, may tear when slow",
            _ => "Unknown present mode",
        }
    }
}

impl fmt::Display for PresentModeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (latency: {}, {})",
            self.name(),
            self.latency,
            if self.tears { "may tear" } else { "no tearing" }
        )
    }
}

impl Eq for PresentModeInfo {}

// ============================================================================
// PresentModeSelector
// ============================================================================

/// Selects the best present mode from available options.
///
/// `PresentModeSelector` queries hardware capabilities and selects optimal
/// present modes based on user preferences, with automatic fallback handling.
///
/// # Example
///
/// ```ignore
/// let selector = PresentModeSelector::from_surface_capabilities(&caps);
///
/// // Select for gaming
/// let gaming_mode = selector.best_for_gaming();
///
/// // Select based on preference
/// let mode = selector.select(PresentModePreference::LowLatency);
///
/// // Check what's available
/// if selector.supports(wgpu::PresentMode::Mailbox) {
///     println!("Triple buffering available!");
/// }
/// ```
#[derive(Debug, Clone)]
pub struct PresentModeSelector {
    available: Vec<wgpu::PresentMode>,
}

impl PresentModeSelector {
    /// Create a new selector with the given available modes.
    ///
    /// # Arguments
    ///
    /// * `available` - List of present modes supported by the hardware.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let selector = PresentModeSelector::new(vec![
    ///     wgpu::PresentMode::Fifo,
    ///     wgpu::PresentMode::Mailbox,
    /// ]);
    /// ```
    pub fn new(available: Vec<wgpu::PresentMode>) -> Self {
        Self { available }
    }

    /// Create a selector from surface capabilities.
    ///
    /// # Arguments
    ///
    /// * `caps` - Surface capabilities from `surface.get_capabilities(adapter)`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let caps = surface.get_capabilities(&adapter);
    /// let selector = PresentModeSelector::from_surface_capabilities(&caps);
    /// ```
    pub fn from_surface_capabilities(caps: &wgpu::SurfaceCapabilities) -> Self {
        Self::new(caps.present_modes.to_vec())
    }

    /// Select the best present mode for a preference.
    ///
    /// Iterates through the preference's fallback chain and returns
    /// the first available mode. Falls back to Fifo if none match.
    ///
    /// # Arguments
    ///
    /// * `preference` - The desired presentation behavior.
    ///
    /// # Returns
    ///
    /// The best available present mode for the given preference.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mode = selector.select(PresentModePreference::LowLatency);
    /// // Returns Mailbox if available, else Immediate, else Fifo
    /// ```
    pub fn select(&self, preference: PresentModePreference) -> wgpu::PresentMode {
        for &mode in preference.fallback_chain() {
            if self.supports(mode) {
                return mode;
            }
        }
        // Fifo is always available per WGPU spec
        wgpu::PresentMode::Fifo
    }

    /// Select a present mode with explicit fallback.
    ///
    /// Similar to `select`, but uses the provided fallback instead of Fifo
    /// if no modes from the preference chain are available.
    ///
    /// # Arguments
    ///
    /// * `preference` - The desired presentation behavior.
    /// * `fallback` - The fallback mode if preference chain fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mode = selector.select_with_fallback(
    ///     PresentModePreference::Adaptive,
    ///     wgpu::PresentMode::Mailbox,
    /// );
    /// ```
    pub fn select_with_fallback(
        &self,
        preference: PresentModePreference,
        fallback: wgpu::PresentMode,
    ) -> wgpu::PresentMode {
        for &mode in preference.fallback_chain() {
            if self.supports(mode) {
                return mode;
            }
        }
        // Use provided fallback, or Fifo if fallback not available
        if self.supports(fallback) {
            fallback
        } else {
            wgpu::PresentMode::Fifo
        }
    }

    /// Check if a specific present mode is supported.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if selector.supports(wgpu::PresentMode::Mailbox) {
    ///     println!("Triple buffering available!");
    /// }
    /// ```
    pub fn supports(&self, mode: wgpu::PresentMode) -> bool {
        self.available.contains(&mode)
    }

    /// Returns a slice of all available present modes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for mode in selector.available_modes() {
    ///     println!("Supported: {:?}", mode);
    /// }
    /// ```
    pub fn available_modes(&self) -> &[wgpu::PresentMode] {
        &self.available
    }

    /// Get detailed information about a present mode.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let info = selector.info(wgpu::PresentMode::Mailbox);
    /// println!("Tears: {}, Latency: {}", info.tears, info.latency);
    /// ```
    pub fn info(&self, mode: wgpu::PresentMode) -> PresentModeInfo {
        PresentModeInfo::from_mode(mode)
    }

    /// Select the best present mode for gaming.
    ///
    /// Prioritizes low latency while avoiding tearing when possible.
    /// Prefers Mailbox > Immediate > FifoRelaxed > Fifo.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let gaming_mode = selector.best_for_gaming();
    /// ```
    pub fn best_for_gaming(&self) -> wgpu::PresentMode {
        // Gaming prioritizes low latency without tearing
        // Mailbox is ideal (low latency + no tearing)
        // Immediate is acceptable (lowest latency but tears)
        // FifoRelaxed is fallback (medium latency)
        // Fifo is last resort (high latency)
        for &mode in &[
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::FifoRelaxed,
            wgpu::PresentMode::Fifo,
        ] {
            if self.supports(mode) {
                return mode;
            }
        }
        wgpu::PresentMode::Fifo
    }

    /// Select the best present mode for video playback.
    ///
    /// Prioritizes smooth presentation without tearing.
    /// Prefers Fifo > FifoRelaxed > Mailbox > Immediate.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let video_mode = selector.best_for_video();
    /// ```
    pub fn best_for_video(&self) -> wgpu::PresentMode {
        // Video prioritizes smooth presentation without tearing
        // Fifo is ideal (consistent timing)
        // FifoRelaxed is acceptable (mostly no tearing)
        // Mailbox is fallback (may skip frames)
        // Immediate is last resort (will tear)
        for &mode in &[
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::FifoRelaxed,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
        ] {
            if self.supports(mode) {
                return mode;
            }
        }
        wgpu::PresentMode::Fifo
    }

    /// Check if any low-latency mode is available.
    ///
    /// Returns true if Mailbox or Immediate is supported.
    pub fn has_low_latency_mode(&self) -> bool {
        self.supports(wgpu::PresentMode::Mailbox) || self.supports(wgpu::PresentMode::Immediate)
    }

    /// Check if adaptive VSync is available.
    ///
    /// Returns true if FifoRelaxed is supported.
    pub fn has_adaptive_vsync(&self) -> bool {
        self.supports(wgpu::PresentMode::FifoRelaxed)
    }

    /// Check if triple buffering (Mailbox) is available.
    pub fn has_triple_buffering(&self) -> bool {
        self.supports(wgpu::PresentMode::Mailbox)
    }

    /// Returns the number of available modes.
    pub fn mode_count(&self) -> usize {
        self.available.len()
    }

    /// Check if the available modes list is empty.
    ///
    /// Note: This should not happen in practice as Fifo is always supported.
    pub fn is_empty(&self) -> bool {
        self.available.is_empty()
    }

    /// Get info for all available modes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for info in selector.all_info() {
    ///     println!("{}: {}", info.name(), info.description());
    /// }
    /// ```
    pub fn all_info(&self) -> Vec<PresentModeInfo> {
        self.available.iter().map(|&m| PresentModeInfo::from_mode(m)).collect()
    }

    /// Find the lowest latency available mode.
    pub fn lowest_latency(&self) -> Option<wgpu::PresentMode> {
        // Order: Immediate < Mailbox < FifoRelaxed < Fifo
        for &mode in &[
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::FifoRelaxed,
            wgpu::PresentMode::Fifo,
        ] {
            if self.supports(mode) {
                return Some(mode);
            }
        }
        self.available.first().copied()
    }

    /// Find the most power-efficient available mode.
    pub fn most_power_efficient(&self) -> Option<wgpu::PresentMode> {
        // Order: Fifo > FifoRelaxed > Mailbox > Immediate
        for &mode in &[
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::FifoRelaxed,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
        ] {
            if self.supports(mode) {
                return Some(mode);
            }
        }
        self.available.first().copied()
    }
}

impl Default for PresentModeSelector {
    fn default() -> Self {
        // Default to just Fifo (always available)
        Self::new(vec![wgpu::PresentMode::Fifo])
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --------------------------------------------------------------------------
    // PresentModePreference tests
    // --------------------------------------------------------------------------

    #[test]
    fn test_preference_vsync_fallback_chain() {
        let chain = PresentModePreference::VSync.fallback_chain();
        assert_eq!(chain, &[wgpu::PresentMode::Fifo]);
    }

    #[test]
    fn test_preference_low_latency_fallback_chain() {
        let chain = PresentModePreference::LowLatency.fallback_chain();
        assert_eq!(chain, &[
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Fifo,
        ]);
    }

    #[test]
    fn test_preference_no_vsync_fallback_chain() {
        let chain = PresentModePreference::NoVSync.fallback_chain();
        assert_eq!(chain, &[
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Fifo,
        ]);
    }

    #[test]
    fn test_preference_adaptive_fallback_chain() {
        let chain = PresentModePreference::Adaptive.fallback_chain();
        assert_eq!(chain, &[
            wgpu::PresentMode::FifoRelaxed,
            wgpu::PresentMode::Fifo,
        ]);
    }

    #[test]
    fn test_preference_power_saving_fallback_chain() {
        let chain = PresentModePreference::PowerSaving.fallback_chain();
        assert_eq!(chain, &[wgpu::PresentMode::Fifo]);
    }

    #[test]
    fn test_preference_default_is_vsync() {
        assert_eq!(PresentModePreference::default(), PresentModePreference::VSync);
    }

    #[test]
    fn test_preference_names_not_empty() {
        assert!(!PresentModePreference::VSync.name().is_empty());
        assert!(!PresentModePreference::LowLatency.name().is_empty());
        assert!(!PresentModePreference::NoVSync.name().is_empty());
        assert!(!PresentModePreference::Adaptive.name().is_empty());
        assert!(!PresentModePreference::PowerSaving.name().is_empty());
    }

    #[test]
    fn test_preference_descriptions_not_empty() {
        assert!(!PresentModePreference::VSync.description().is_empty());
        assert!(!PresentModePreference::LowLatency.description().is_empty());
        assert!(!PresentModePreference::NoVSync.description().is_empty());
        assert!(!PresentModePreference::Adaptive.description().is_empty());
        assert!(!PresentModePreference::PowerSaving.description().is_empty());
    }

    #[test]
    fn test_preference_display() {
        assert_eq!(format!("{}", PresentModePreference::VSync), "VSync");
        assert_eq!(format!("{}", PresentModePreference::LowLatency), "Low Latency");
        assert_eq!(format!("{}", PresentModePreference::NoVSync), "No VSync");
    }

    #[test]
    fn test_preference_latency_order() {
        let order = PresentModePreference::latency_order();
        assert_eq!(order.len(), 5);
        assert_eq!(order[0], PresentModePreference::NoVSync);
        assert_eq!(order[4], PresentModePreference::PowerSaving);
    }

    #[test]
    fn test_preference_power_order() {
        let order = PresentModePreference::power_order();
        assert_eq!(order.len(), 5);
        assert_eq!(order[0], PresentModePreference::PowerSaving);
        assert_eq!(order[4], PresentModePreference::NoVSync);
    }

    // --------------------------------------------------------------------------
    // LatencyLevel tests
    // --------------------------------------------------------------------------

    #[test]
    fn test_latency_level_ordering() {
        assert!(LatencyLevel::Lowest < LatencyLevel::Low);
        assert!(LatencyLevel::Low < LatencyLevel::Medium);
        assert!(LatencyLevel::Medium < LatencyLevel::High);
    }

    #[test]
    fn test_latency_level_from_immediate() {
        let level = LatencyLevel::from_present_mode(wgpu::PresentMode::Immediate);
        assert_eq!(level, LatencyLevel::Lowest);
    }

    #[test]
    fn test_latency_level_from_mailbox() {
        let level = LatencyLevel::from_present_mode(wgpu::PresentMode::Mailbox);
        assert_eq!(level, LatencyLevel::Low);
    }

    #[test]
    fn test_latency_level_from_fifo_relaxed() {
        let level = LatencyLevel::from_present_mode(wgpu::PresentMode::FifoRelaxed);
        assert_eq!(level, LatencyLevel::Medium);
    }

    #[test]
    fn test_latency_level_from_fifo() {
        let level = LatencyLevel::from_present_mode(wgpu::PresentMode::Fifo);
        assert_eq!(level, LatencyLevel::High);
    }

    #[test]
    fn test_latency_level_typical_frame_latency() {
        assert_eq!(LatencyLevel::Lowest.typical_frame_latency(), 0);
        assert_eq!(LatencyLevel::Low.typical_frame_latency(), 1);
        assert_eq!(LatencyLevel::Medium.typical_frame_latency(), 1);
        assert_eq!(LatencyLevel::High.typical_frame_latency(), 2);
    }

    #[test]
    fn test_latency_level_is_competitive() {
        assert!(LatencyLevel::Lowest.is_competitive());
        assert!(LatencyLevel::Low.is_competitive());
        assert!(!LatencyLevel::Medium.is_competitive());
        assert!(!LatencyLevel::High.is_competitive());
    }

    #[test]
    fn test_latency_level_estimated_ms() {
        assert_eq!(LatencyLevel::Lowest.estimated_ms_at_60hz(), 0.0);
        assert!(LatencyLevel::Low.estimated_ms_at_60hz() > 0.0);
        assert!(LatencyLevel::High.estimated_ms_at_60hz() > LatencyLevel::Low.estimated_ms_at_60hz());
    }

    #[test]
    fn test_latency_level_display() {
        assert_eq!(format!("{}", LatencyLevel::Lowest), "Lowest");
        assert_eq!(format!("{}", LatencyLevel::Low), "Low");
        assert_eq!(format!("{}", LatencyLevel::Medium), "Medium");
        assert_eq!(format!("{}", LatencyLevel::High), "High");
    }

    // --------------------------------------------------------------------------
    // PresentModeInfo tests
    // --------------------------------------------------------------------------

    #[test]
    fn test_info_immediate() {
        let info = PresentModeInfo::from_mode(wgpu::PresentMode::Immediate);
        assert_eq!(info.mode, wgpu::PresentMode::Immediate);
        assert!(info.tears);
        assert_eq!(info.latency, LatencyLevel::Lowest);
        assert!(!info.power_efficient);
    }

    #[test]
    fn test_info_mailbox() {
        let info = PresentModeInfo::from_mode(wgpu::PresentMode::Mailbox);
        assert_eq!(info.mode, wgpu::PresentMode::Mailbox);
        assert!(!info.tears);
        assert_eq!(info.latency, LatencyLevel::Low);
        assert!(!info.power_efficient);
    }

    #[test]
    fn test_info_fifo() {
        let info = PresentModeInfo::from_mode(wgpu::PresentMode::Fifo);
        assert_eq!(info.mode, wgpu::PresentMode::Fifo);
        assert!(!info.tears);
        assert_eq!(info.latency, LatencyLevel::High);
        assert!(info.power_efficient);
    }

    #[test]
    fn test_info_fifo_relaxed() {
        let info = PresentModeInfo::from_mode(wgpu::PresentMode::FifoRelaxed);
        assert_eq!(info.mode, wgpu::PresentMode::FifoRelaxed);
        assert!(info.tears); // Can tear when below refresh rate
        assert_eq!(info.latency, LatencyLevel::Medium);
        assert!(info.power_efficient);
    }

    #[test]
    fn test_info_prevents_tearing() {
        assert!(!PresentModeInfo::from_mode(wgpu::PresentMode::Immediate).prevents_tearing());
        assert!(PresentModeInfo::from_mode(wgpu::PresentMode::Mailbox).prevents_tearing());
        assert!(PresentModeInfo::from_mode(wgpu::PresentMode::Fifo).prevents_tearing());
        assert!(!PresentModeInfo::from_mode(wgpu::PresentMode::FifoRelaxed).prevents_tearing());
    }

    #[test]
    fn test_info_is_competitive() {
        assert!(PresentModeInfo::from_mode(wgpu::PresentMode::Immediate).is_competitive());
        assert!(PresentModeInfo::from_mode(wgpu::PresentMode::Mailbox).is_competitive());
        assert!(!PresentModeInfo::from_mode(wgpu::PresentMode::Fifo).is_competitive());
    }

    #[test]
    fn test_info_is_battery_friendly() {
        assert!(!PresentModeInfo::from_mode(wgpu::PresentMode::Immediate).is_battery_friendly());
        assert!(!PresentModeInfo::from_mode(wgpu::PresentMode::Mailbox).is_battery_friendly());
        assert!(PresentModeInfo::from_mode(wgpu::PresentMode::Fifo).is_battery_friendly());
        assert!(PresentModeInfo::from_mode(wgpu::PresentMode::FifoRelaxed).is_battery_friendly());
    }

    #[test]
    fn test_info_names_not_empty() {
        assert!(!PresentModeInfo::from_mode(wgpu::PresentMode::Immediate).name().is_empty());
        assert!(!PresentModeInfo::from_mode(wgpu::PresentMode::Mailbox).name().is_empty());
        assert!(!PresentModeInfo::from_mode(wgpu::PresentMode::Fifo).name().is_empty());
        assert!(!PresentModeInfo::from_mode(wgpu::PresentMode::FifoRelaxed).name().is_empty());
    }

    #[test]
    fn test_info_display() {
        let info = PresentModeInfo::from_mode(wgpu::PresentMode::Mailbox);
        let display = format!("{}", info);
        assert!(display.contains("Mailbox"));
        assert!(display.contains("Low"));
        assert!(display.contains("no tearing"));
    }

    #[test]
    fn test_info_equality() {
        let info1 = PresentModeInfo::from_mode(wgpu::PresentMode::Fifo);
        let info2 = PresentModeInfo::from_mode(wgpu::PresentMode::Fifo);
        assert_eq!(info1, info2);
    }

    // --------------------------------------------------------------------------
    // PresentModeSelector tests
    // --------------------------------------------------------------------------

    #[test]
    fn test_selector_new_empty() {
        let selector = PresentModeSelector::new(vec![]);
        assert!(selector.is_empty());
        assert_eq!(selector.mode_count(), 0);
    }

    #[test]
    fn test_selector_new_single_mode() {
        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Fifo]);
        assert!(!selector.is_empty());
        assert_eq!(selector.mode_count(), 1);
        assert!(selector.supports(wgpu::PresentMode::Fifo));
    }

    #[test]
    fn test_selector_new_all_modes() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::FifoRelaxed,
        ]);
        assert_eq!(selector.mode_count(), 4);
        assert!(selector.supports(wgpu::PresentMode::Immediate));
        assert!(selector.supports(wgpu::PresentMode::Mailbox));
        assert!(selector.supports(wgpu::PresentMode::Fifo));
        assert!(selector.supports(wgpu::PresentMode::FifoRelaxed));
    }

    #[test]
    fn test_selector_supports() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
        ]);
        assert!(selector.supports(wgpu::PresentMode::Fifo));
        assert!(selector.supports(wgpu::PresentMode::Mailbox));
        assert!(!selector.supports(wgpu::PresentMode::Immediate));
    }

    #[test]
    fn test_selector_available_modes() {
        let modes = vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox];
        let selector = PresentModeSelector::new(modes.clone());
        assert_eq!(selector.available_modes(), &modes);
    }

    #[test]
    fn test_selector_select_vsync() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Fifo,
        ]);
        let mode = selector.select(PresentModePreference::VSync);
        assert_eq!(mode, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_selector_select_low_latency_prefers_mailbox() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
        ]);
        let mode = selector.select(PresentModePreference::LowLatency);
        assert_eq!(mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_selector_select_low_latency_falls_back_to_immediate() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Immediate,
        ]);
        let mode = selector.select(PresentModePreference::LowLatency);
        assert_eq!(mode, wgpu::PresentMode::Immediate);
    }

    #[test]
    fn test_selector_select_no_vsync() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
        ]);
        let mode = selector.select(PresentModePreference::NoVSync);
        assert_eq!(mode, wgpu::PresentMode::Immediate);
    }

    #[test]
    fn test_selector_select_adaptive() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::FifoRelaxed,
        ]);
        let mode = selector.select(PresentModePreference::Adaptive);
        assert_eq!(mode, wgpu::PresentMode::FifoRelaxed);
    }

    #[test]
    fn test_selector_select_adaptive_fallback() {
        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Fifo]);
        let mode = selector.select(PresentModePreference::Adaptive);
        assert_eq!(mode, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_selector_select_power_saving() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Fifo,
        ]);
        let mode = selector.select(PresentModePreference::PowerSaving);
        assert_eq!(mode, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_selector_select_with_fallback() {
        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Mailbox]);
        let mode = selector.select_with_fallback(
            PresentModePreference::Adaptive,
            wgpu::PresentMode::Mailbox,
        );
        assert_eq!(mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_selector_select_with_fallback_uses_fifo() {
        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Fifo]);
        let mode = selector.select_with_fallback(
            PresentModePreference::Adaptive,
            wgpu::PresentMode::Immediate, // Not available
        );
        assert_eq!(mode, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_selector_best_for_gaming() {
        // With Mailbox available
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
        ]);
        assert_eq!(selector.best_for_gaming(), wgpu::PresentMode::Mailbox);

        // Without Mailbox, prefer Immediate
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Immediate,
        ]);
        assert_eq!(selector.best_for_gaming(), wgpu::PresentMode::Immediate);

        // Only Fifo
        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Fifo]);
        assert_eq!(selector.best_for_gaming(), wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_selector_best_for_video() {
        // With Fifo available
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Fifo,
        ]);
        assert_eq!(selector.best_for_video(), wgpu::PresentMode::Fifo);

        // Only Immediate
        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Immediate]);
        assert_eq!(selector.best_for_video(), wgpu::PresentMode::Immediate);
    }

    #[test]
    fn test_selector_info() {
        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Mailbox]);
        let info = selector.info(wgpu::PresentMode::Mailbox);
        assert_eq!(info.mode, wgpu::PresentMode::Mailbox);
        assert!(!info.tears);
    }

    #[test]
    fn test_selector_has_low_latency_mode() {
        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Mailbox]);
        assert!(selector.has_low_latency_mode());

        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Immediate]);
        assert!(selector.has_low_latency_mode());

        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Fifo]);
        assert!(!selector.has_low_latency_mode());
    }

    #[test]
    fn test_selector_has_adaptive_vsync() {
        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::FifoRelaxed]);
        assert!(selector.has_adaptive_vsync());

        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Fifo]);
        assert!(!selector.has_adaptive_vsync());
    }

    #[test]
    fn test_selector_has_triple_buffering() {
        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Mailbox]);
        assert!(selector.has_triple_buffering());

        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Fifo]);
        assert!(!selector.has_triple_buffering());
    }

    #[test]
    fn test_selector_all_info() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
        ]);
        let all_info = selector.all_info();
        assert_eq!(all_info.len(), 2);
    }

    #[test]
    fn test_selector_lowest_latency() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
        ]);
        assert_eq!(selector.lowest_latency(), Some(wgpu::PresentMode::Immediate));

        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Fifo]);
        assert_eq!(selector.lowest_latency(), Some(wgpu::PresentMode::Fifo));
    }

    #[test]
    fn test_selector_most_power_efficient() {
        let selector = PresentModeSelector::new(vec![
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Fifo,
        ]);
        assert_eq!(selector.most_power_efficient(), Some(wgpu::PresentMode::Fifo));

        let selector = PresentModeSelector::new(vec![wgpu::PresentMode::Immediate]);
        assert_eq!(selector.most_power_efficient(), Some(wgpu::PresentMode::Immediate));
    }

    #[test]
    fn test_selector_default() {
        let selector = PresentModeSelector::default();
        assert!(selector.supports(wgpu::PresentMode::Fifo));
        assert_eq!(selector.mode_count(), 1);
    }

    #[test]
    fn test_selector_empty_list_edge_case() {
        let selector = PresentModeSelector::new(vec![]);
        // Select should fallback to Fifo even with empty list
        let mode = selector.select(PresentModePreference::LowLatency);
        assert_eq!(mode, wgpu::PresentMode::Fifo);
    }

    // --------------------------------------------------------------------------
    // Platform-specific tests
    // --------------------------------------------------------------------------

    #[test]
    fn test_platform_default() {
        // On desktop, default should be VSync
        // On mobile, default should be PowerSaving
        let default = PresentModePreference::default_for_platform();
        #[cfg(any(target_os = "ios", target_os = "android"))]
        assert_eq!(default, PresentModePreference::PowerSaving);
        #[cfg(not(any(target_os = "ios", target_os = "android")))]
        assert_eq!(default, PresentModePreference::VSync);
    }
}
