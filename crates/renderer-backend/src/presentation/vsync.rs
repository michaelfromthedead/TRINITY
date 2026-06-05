//! VSync and present mode management for the TRINITY renderer.
//!
//! This module provides VSync control, adaptive frame pacing, and present mode
//! selection for optimal display synchronization across different hardware.
//!
//! # Components
//!
//! - [`VSyncMode`] - User-facing VSync preference
//! - [`VSyncPresentModeInfo`] - Extended present mode information with VSync context
//! - [`VSyncController`] - Adaptive VSync management with frame time tracking
//! - [`PresentModeSelector`] - Query and select optimal present modes
//! - [`VSyncFramePacer`] - Frame timing with VSync-aware pacing
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::presentation::vsync::{VSyncMode, VSyncController, PresentModeSelector};
//!
//! // Create controller with adaptive VSync
//! let mut controller = VSyncController::new();
//! controller.set_mode(VSyncMode::Adaptive);
//! controller.set_target_fps(Some(60.0));
//!
//! // Get best present mode for current settings
//! let available_modes = PresentModeSelector::available_modes(&surface, &adapter);
//! let best_mode = PresentModeSelector::best_for_vsync(&available_modes, controller.mode());
//!
//! // In render loop
//! controller.record_frame_time(frame_time_ms);
//! if controller.should_adapt() {
//!     // Switch present mode based on performance
//! }
//! ```

use std::collections::VecDeque;
use std::fmt;
use std::time::Instant;

// ============================================================================
// VSyncMode
// ============================================================================

/// User-facing VSync mode preference.
///
/// This enum represents the user's desired VSync behavior, which is then
/// mapped to appropriate `wgpu::PresentMode` values based on hardware support.
///
/// # Modes
///
/// - **On**: Traditional VSync, capped at display refresh rate
/// - **Off**: No VSync, uncapped framerate (may cause tearing)
/// - **Adaptive**: VSync on when above refresh rate, off when below
/// - **FastSync**: VSync with frame buffering for low latency
/// - **HalfRate**: VSync at half the display refresh rate (30fps on 60Hz)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum VSyncMode {
    /// VSync enabled - frames are synchronized to display refresh.
    ///
    /// The GPU waits for vertical blank before presenting, eliminating
    /// tearing but potentially increasing input latency. Frame rate is
    /// capped at the display's refresh rate.
    ///
    /// Maps to: `PresentMode::Fifo`
    #[default]
    On,

    /// VSync disabled - frames present immediately.
    ///
    /// No synchronization with display refresh. Provides lowest input
    /// latency and allows frame rates above the refresh rate, but may
    /// cause visible tearing.
    ///
    /// Maps to: `PresentMode::Immediate`
    Off,

    /// Adaptive VSync - dynamically switches based on performance.
    ///
    /// When the GPU can maintain frame rate above the refresh rate,
    /// VSync is enabled to prevent tearing. When frame rate drops
    /// below refresh rate, VSync is disabled to prevent stuttering.
    ///
    /// Maps to: `PresentMode::FifoRelaxed`
    Adaptive,

    /// Fast sync - VSync with frame buffering for low latency.
    ///
    /// Uses triple buffering (mailbox) to maintain low latency while
    /// still synchronizing to the display. The latest completed frame
    /// is always presented, discarding older frames.
    ///
    /// Maps to: `PresentMode::Mailbox`
    FastSync,

    /// Half-rate VSync - synchronize at half the refresh rate.
    ///
    /// Useful for GPU-bound scenarios where maintaining full refresh
    /// rate is impossible. Provides consistent 30fps on 60Hz displays,
    /// avoiding the stutter of fluctuating between full and half rate.
    ///
    /// Maps to: `PresentMode::Fifo` with frame limiting
    HalfRate,
}

impl VSyncMode {
    /// Returns a human-readable name for this mode.
    pub const fn name(self) -> &'static str {
        match self {
            VSyncMode::On => "VSync On",
            VSyncMode::Off => "VSync Off",
            VSyncMode::Adaptive => "Adaptive VSync",
            VSyncMode::FastSync => "Fast Sync",
            VSyncMode::HalfRate => "Half-Rate VSync",
        }
    }

    /// Returns a description of this mode.
    pub const fn description(self) -> &'static str {
        match self {
            VSyncMode::On => "Synced to display refresh, no tearing",
            VSyncMode::Off => "Uncapped framerate, may tear",
            VSyncMode::Adaptive => "VSync when above refresh rate",
            VSyncMode::FastSync => "Low latency triple buffering",
            VSyncMode::HalfRate => "Synced at half refresh rate",
        }
    }

    /// Returns true if this mode prevents screen tearing.
    pub const fn prevents_tearing(self) -> bool {
        match self {
            VSyncMode::On | VSyncMode::FastSync | VSyncMode::HalfRate => true,
            VSyncMode::Off => false,
            VSyncMode::Adaptive => true, // Mostly prevents tearing
        }
    }

    /// Returns true if this mode caps frame rate.
    pub const fn caps_framerate(self) -> bool {
        match self {
            VSyncMode::On | VSyncMode::Adaptive | VSyncMode::FastSync | VSyncMode::HalfRate => true,
            VSyncMode::Off => false,
        }
    }

    /// Returns true if this mode is power efficient.
    ///
    /// Power efficient modes allow the GPU to idle when not needed.
    pub const fn is_power_efficient(self) -> bool {
        match self {
            VSyncMode::On | VSyncMode::Adaptive | VSyncMode::HalfRate => true,
            VSyncMode::Off | VSyncMode::FastSync => false,
        }
    }

    /// Returns the relative latency rank (1 = lowest, 5 = highest).
    pub const fn latency_rank(self) -> u8 {
        match self {
            VSyncMode::Off => 1,       // Lowest latency
            VSyncMode::FastSync => 2,  // Low latency with buffering
            VSyncMode::Adaptive => 3,  // Variable latency
            VSyncMode::On => 4,        // Standard VSync latency
            VSyncMode::HalfRate => 5,  // Highest latency
        }
    }

    /// Convert to the preferred `wgpu::PresentMode`.
    pub const fn to_present_mode(self) -> wgpu::PresentMode {
        match self {
            VSyncMode::On | VSyncMode::HalfRate => wgpu::PresentMode::Fifo,
            VSyncMode::Off => wgpu::PresentMode::Immediate,
            VSyncMode::Adaptive => wgpu::PresentMode::FifoRelaxed,
            VSyncMode::FastSync => wgpu::PresentMode::Mailbox,
        }
    }

    /// Create from a `wgpu::PresentMode`.
    pub const fn from_present_mode(mode: wgpu::PresentMode) -> Self {
        match mode {
            wgpu::PresentMode::Fifo => VSyncMode::On,
            wgpu::PresentMode::FifoRelaxed => VSyncMode::Adaptive,
            wgpu::PresentMode::Immediate => VSyncMode::Off,
            wgpu::PresentMode::Mailbox => VSyncMode::FastSync,
            _ => VSyncMode::On, // Default to VSync on for unknown modes
        }
    }

    /// Returns all available modes in order of preference for competitive gaming.
    pub const fn competitive_order() -> [VSyncMode; 5] {
        [
            VSyncMode::Off,
            VSyncMode::FastSync,
            VSyncMode::Adaptive,
            VSyncMode::On,
            VSyncMode::HalfRate,
        ]
    }

    /// Returns all available modes in order of preference for power saving.
    pub const fn power_saving_order() -> [VSyncMode; 5] {
        [
            VSyncMode::HalfRate,
            VSyncMode::On,
            VSyncMode::Adaptive,
            VSyncMode::FastSync,
            VSyncMode::Off,
        ]
    }
}

impl fmt::Display for VSyncMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// VSyncPresentModeInfo
// ============================================================================

/// Extended present mode information with VSync context.
///
/// This provides detailed information about a present mode, including
/// its relationship to VSync settings, latency characteristics, and
/// power efficiency.
#[derive(Clone, Debug)]
pub struct VSyncPresentModeInfo {
    /// The underlying wgpu present mode.
    pub mode: wgpu::PresentMode,
    /// The VSync mode this represents.
    pub vsync: VSyncMode,
    /// Whether tearing is allowed with this mode.
    pub tear_allowed: bool,
    /// Typical frame latency in number of frames.
    pub latency_frames: u32,
    /// Whether this mode is power efficient.
    pub power_efficient: bool,
}

impl VSyncPresentModeInfo {
    /// Create info for a present mode.
    pub const fn from_present_mode(mode: wgpu::PresentMode) -> Self {
        match mode {
            wgpu::PresentMode::Immediate => Self {
                mode,
                vsync: VSyncMode::Off,
                tear_allowed: true,
                latency_frames: 0,
                power_efficient: false,
            },
            wgpu::PresentMode::Mailbox => Self {
                mode,
                vsync: VSyncMode::FastSync,
                tear_allowed: false,
                latency_frames: 1,
                power_efficient: false,
            },
            wgpu::PresentMode::Fifo => Self {
                mode,
                vsync: VSyncMode::On,
                tear_allowed: false,
                latency_frames: 2,
                power_efficient: true,
            },
            wgpu::PresentMode::FifoRelaxed => Self {
                mode,
                vsync: VSyncMode::Adaptive,
                tear_allowed: true, // Can tear when below refresh
                latency_frames: 1,
                power_efficient: true,
            },
            _ => Self {
                mode,
                vsync: VSyncMode::On,
                tear_allowed: false,
                latency_frames: 2,
                power_efficient: false,
            },
        }
    }

    /// Create info for a VSync mode.
    pub const fn from_vsync_mode(vsync: VSyncMode) -> Self {
        let mode = vsync.to_present_mode();
        match vsync {
            VSyncMode::On => Self {
                mode,
                vsync,
                tear_allowed: false,
                latency_frames: 2,
                power_efficient: true,
            },
            VSyncMode::Off => Self {
                mode,
                vsync,
                tear_allowed: true,
                latency_frames: 0,
                power_efficient: false,
            },
            VSyncMode::Adaptive => Self {
                mode,
                vsync,
                tear_allowed: true,
                latency_frames: 1,
                power_efficient: true,
            },
            VSyncMode::FastSync => Self {
                mode,
                vsync,
                tear_allowed: false,
                latency_frames: 1,
                power_efficient: false,
            },
            VSyncMode::HalfRate => Self {
                mode,
                vsync,
                tear_allowed: false,
                latency_frames: 3,
                power_efficient: true,
            },
        }
    }

    /// Returns true if this mode is suitable for competitive gaming.
    pub const fn is_competitive(&self) -> bool {
        self.latency_frames <= 1
    }

    /// Returns true if this mode is suitable for battery-powered devices.
    pub const fn is_battery_friendly(&self) -> bool {
        self.power_efficient
    }

    /// Returns the estimated input lag in milliseconds at 60Hz.
    pub const fn estimated_lag_ms_at_60hz(&self) -> f64 {
        // Each frame of latency = ~16.67ms at 60Hz
        self.latency_frames as f64 * 16.67
    }
}

impl fmt::Display for VSyncPresentModeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({:?}, {}frame latency, {})",
            self.vsync.name(),
            self.mode,
            self.latency_frames,
            if self.power_efficient { "efficient" } else { "high-perf" }
        )
    }
}

// ============================================================================
// VSyncController
// ============================================================================

/// Default number of frame times to track for adaptive VSync.
const DEFAULT_FRAME_TIME_HISTORY: usize = 60;

/// Controller for VSync mode management with adaptive behavior.
///
/// `VSyncController` tracks frame times and can automatically switch
/// between VSync modes based on performance, implementing adaptive
/// VSync behavior.
///
/// # Example
///
/// ```ignore
/// let mut controller = VSyncController::new();
/// controller.set_mode(VSyncMode::Adaptive);
/// controller.set_target_fps(Some(60.0));
///
/// // In render loop
/// controller.record_frame_time(frame_time_ms);
///
/// if controller.should_adapt() {
///     let new_mode = if controller.average_fps() < 55.0 {
///         VSyncMode::Off
///     } else {
///         VSyncMode::On
///     };
///     controller.set_mode(new_mode);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct VSyncController {
    /// Current VSync mode.
    current_mode: VSyncMode,
    /// Target FPS for frame limiting (if any).
    target_fps: Option<f64>,
    /// Threshold for adaptive switching (fraction of target FPS).
    adaptive_threshold: f64,
    /// History of frame times in milliseconds.
    frame_times: VecDeque<f64>,
    /// Maximum history size.
    history_size: usize,
    /// Last adaptation check time.
    last_adapt_check: Option<Instant>,
    /// Minimum interval between adaptations.
    adapt_cooldown_ms: f64,
    /// Number of mode switches.
    switch_count: u64,
}

impl VSyncController {
    /// Create a new VSync controller with default settings.
    pub fn new() -> Self {
        Self {
            current_mode: VSyncMode::default(),
            target_fps: None,
            adaptive_threshold: 0.95, // Switch at 95% of target
            frame_times: VecDeque::with_capacity(DEFAULT_FRAME_TIME_HISTORY),
            history_size: DEFAULT_FRAME_TIME_HISTORY,
            last_adapt_check: None,
            adapt_cooldown_ms: 500.0, // 500ms between adaptations
            switch_count: 0,
        }
    }

    /// Create a controller with custom history size.
    pub fn with_history_size(size: usize) -> Self {
        Self {
            frame_times: VecDeque::with_capacity(size),
            history_size: size,
            ..Self::new()
        }
    }

    /// Set the current VSync mode.
    pub fn set_mode(&mut self, mode: VSyncMode) {
        if self.current_mode != mode {
            self.current_mode = mode;
            self.switch_count += 1;
        }
    }

    /// Get the current VSync mode.
    pub fn mode(&self) -> VSyncMode {
        self.current_mode
    }

    /// Set the target FPS for frame limiting.
    ///
    /// Pass `None` to disable frame limiting.
    pub fn set_target_fps(&mut self, fps: Option<f64>) {
        self.target_fps = fps.filter(|&f| f > 0.0);
    }

    /// Get the current target FPS.
    pub fn target_fps(&self) -> Option<f64> {
        self.target_fps
    }

    /// Set the adaptive threshold as a fraction of target FPS.
    ///
    /// When the average FPS drops below `target_fps * threshold`,
    /// adaptation is triggered. Default is 0.95 (95%).
    pub fn set_adaptive_threshold(&mut self, threshold: f64) {
        self.adaptive_threshold = threshold.clamp(0.5, 1.0);
    }

    /// Get the adaptive threshold.
    pub fn adaptive_threshold(&self) -> f64 {
        self.adaptive_threshold
    }

    /// Set the cooldown between adaptations in milliseconds.
    pub fn set_adapt_cooldown(&mut self, cooldown_ms: f64) {
        self.adapt_cooldown_ms = cooldown_ms.max(0.0);
    }

    /// Record a frame time in milliseconds.
    pub fn record_frame_time(&mut self, time_ms: f64) {
        if self.frame_times.len() >= self.history_size {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(time_ms);
    }

    /// Get the average FPS based on recorded frame times.
    pub fn average_fps(&self) -> f64 {
        if self.frame_times.is_empty() {
            return 0.0;
        }

        let avg_frame_time_ms: f64 = self.frame_times.iter().sum::<f64>() / self.frame_times.len() as f64;
        if avg_frame_time_ms > 0.0 {
            1000.0 / avg_frame_time_ms
        } else {
            0.0
        }
    }

    /// Get the average frame time in milliseconds.
    pub fn average_frame_time_ms(&self) -> f64 {
        if self.frame_times.is_empty() {
            return 0.0;
        }
        self.frame_times.iter().sum::<f64>() / self.frame_times.len() as f64
    }

    /// Get the minimum frame time in milliseconds.
    pub fn min_frame_time_ms(&self) -> f64 {
        self.frame_times.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Get the maximum frame time in milliseconds.
    pub fn max_frame_time_ms(&self) -> f64 {
        self.frame_times.iter().copied().fold(0.0, f64::max)
    }

    /// Get frame time variance.
    pub fn frame_time_variance(&self) -> f64 {
        if self.frame_times.len() < 2 {
            return 0.0;
        }

        let mean = self.average_frame_time_ms();
        let variance: f64 = self.frame_times
            .iter()
            .map(|&t| (t - mean).powi(2))
            .sum::<f64>() / (self.frame_times.len() - 1) as f64;
        variance
    }

    /// Check if adaptation should occur based on current performance.
    ///
    /// Returns `true` if:
    /// - Mode is Adaptive
    /// - Enough frame time history exists
    /// - Cooldown period has elapsed
    /// - Performance is outside acceptable range
    pub fn should_adapt(&mut self) -> bool {
        // Only adapt in adaptive mode
        if self.current_mode != VSyncMode::Adaptive {
            return false;
        }

        // Need enough history
        if self.frame_times.len() < self.history_size / 2 {
            return false;
        }

        // Check cooldown
        let now = Instant::now();
        if let Some(last_check) = self.last_adapt_check {
            let elapsed_ms = last_check.elapsed().as_secs_f64() * 1000.0;
            if elapsed_ms < self.adapt_cooldown_ms {
                return false;
            }
        }

        // Check if adaptation is needed
        if let Some(target) = self.target_fps {
            let current_fps = self.average_fps();
            let threshold_fps = target * self.adaptive_threshold;

            // Need to adapt if significantly below threshold
            let should_adapt = current_fps < threshold_fps || current_fps > target * 1.1;

            if should_adapt {
                self.last_adapt_check = Some(now);
            }
            should_adapt
        } else {
            false
        }
    }

    /// Get the recommended present mode based on current state.
    pub fn to_present_mode(&self) -> wgpu::PresentMode {
        self.current_mode.to_present_mode()
    }

    /// Get the number of mode switches.
    pub fn switch_count(&self) -> u64 {
        self.switch_count
    }

    /// Get the number of recorded frame times.
    pub fn frame_count(&self) -> usize {
        self.frame_times.len()
    }

    /// Clear the frame time history.
    pub fn clear_history(&mut self) {
        self.frame_times.clear();
        self.last_adapt_check = None;
    }

    /// Reset the controller to default state.
    pub fn reset(&mut self) {
        self.current_mode = VSyncMode::default();
        self.target_fps = None;
        self.frame_times.clear();
        self.last_adapt_check = None;
        self.switch_count = 0;
    }

    /// Get extended info about current mode.
    pub fn mode_info(&self) -> VSyncPresentModeInfo {
        VSyncPresentModeInfo::from_vsync_mode(self.current_mode)
    }
}

impl Default for VSyncController {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PresentModeSelector
// ============================================================================

/// Helper for querying and selecting present modes.
///
/// `PresentModeSelector` provides utilities for determining available
/// present modes and selecting the best one for a given VSync preference.
///
/// # Example
///
/// ```ignore
/// let modes = PresentModeSelector::available_modes(&surface, &adapter);
///
/// if PresentModeSelector::supports_mailbox(&modes) {
///     let mode = PresentModeSelector::best_for_vsync(&modes, VSyncMode::FastSync);
///     surface.configure(&device, &config.with_present_mode(mode));
/// }
/// ```
pub struct PresentModeSelector;

impl PresentModeSelector {
    /// Get all available present modes for a surface.
    ///
    /// Returns the list of present modes supported by the surface
    /// on the given adapter.
    pub fn available_modes(surface: &wgpu::Surface<'_>, adapter: &wgpu::Adapter) -> Vec<wgpu::PresentMode> {
        let caps = surface.get_capabilities(adapter);
        caps.present_modes.to_vec()
    }

    /// Select the best present mode for a VSync preference.
    ///
    /// Returns the best available present mode that matches the requested
    /// VSync mode, falling back to alternatives if the preferred mode
    /// is not available.
    pub fn best_for_vsync(modes: &[wgpu::PresentMode], vsync: VSyncMode) -> wgpu::PresentMode {
        let preferred = vsync.to_present_mode();

        // First try the exact match
        if modes.contains(&preferred) {
            return preferred;
        }

        // Fall back based on VSync preference
        match vsync {
            VSyncMode::On | VSyncMode::HalfRate => {
                // VSync on: prefer Fifo > FifoRelaxed > anything
                Self::first_available(modes, &[
                    wgpu::PresentMode::Fifo,
                    wgpu::PresentMode::FifoRelaxed,
                    wgpu::PresentMode::Mailbox,
                    wgpu::PresentMode::Immediate,
                ])
            }
            VSyncMode::Off => {
                // VSync off: prefer Immediate > Mailbox > anything
                Self::first_available(modes, &[
                    wgpu::PresentMode::Immediate,
                    wgpu::PresentMode::Mailbox,
                    wgpu::PresentMode::FifoRelaxed,
                    wgpu::PresentMode::Fifo,
                ])
            }
            VSyncMode::Adaptive => {
                // Adaptive: prefer FifoRelaxed > Mailbox > Fifo
                Self::first_available(modes, &[
                    wgpu::PresentMode::FifoRelaxed,
                    wgpu::PresentMode::Mailbox,
                    wgpu::PresentMode::Fifo,
                    wgpu::PresentMode::Immediate,
                ])
            }
            VSyncMode::FastSync => {
                // Fast sync: prefer Mailbox > Immediate > FifoRelaxed
                Self::first_available(modes, &[
                    wgpu::PresentMode::Mailbox,
                    wgpu::PresentMode::Immediate,
                    wgpu::PresentMode::FifoRelaxed,
                    wgpu::PresentMode::Fifo,
                ])
            }
        }
    }

    /// Check if the modes support VSync (Fifo or FifoRelaxed).
    pub fn supports_vsync(modes: &[wgpu::PresentMode]) -> bool {
        modes.contains(&wgpu::PresentMode::Fifo) ||
        modes.contains(&wgpu::PresentMode::FifoRelaxed)
    }

    /// Check if the modes support immediate (no VSync) presentation.
    pub fn supports_immediate(modes: &[wgpu::PresentMode]) -> bool {
        modes.contains(&wgpu::PresentMode::Immediate)
    }

    /// Check if the modes support mailbox (triple buffered) presentation.
    pub fn supports_mailbox(modes: &[wgpu::PresentMode]) -> bool {
        modes.contains(&wgpu::PresentMode::Mailbox)
    }

    /// Check if the modes support adaptive VSync.
    pub fn supports_adaptive(modes: &[wgpu::PresentMode]) -> bool {
        modes.contains(&wgpu::PresentMode::FifoRelaxed)
    }

    /// Get information about a present mode.
    pub fn mode_info(mode: wgpu::PresentMode) -> VSyncPresentModeInfo {
        VSyncPresentModeInfo::from_present_mode(mode)
    }

    /// Get information about all available modes.
    pub fn all_mode_info(modes: &[wgpu::PresentMode]) -> Vec<VSyncPresentModeInfo> {
        modes.iter().map(|&m| VSyncPresentModeInfo::from_present_mode(m)).collect()
    }

    /// Find the lowest latency mode from available modes.
    pub fn lowest_latency(modes: &[wgpu::PresentMode]) -> Option<wgpu::PresentMode> {
        // Order: Immediate (0) < Mailbox (1) < FifoRelaxed (1) < Fifo (2)
        Self::first_available(modes, &[
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::FifoRelaxed,
            wgpu::PresentMode::Fifo,
        ]).into()
    }

    /// Find the most power efficient mode from available modes.
    pub fn most_power_efficient(modes: &[wgpu::PresentMode]) -> Option<wgpu::PresentMode> {
        // Order: Fifo > FifoRelaxed > Mailbox > Immediate
        Self::first_available(modes, &[
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::FifoRelaxed,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
        ]).into()
    }

    /// Get the default present mode (Fifo - always supported).
    pub const fn default_mode() -> wgpu::PresentMode {
        wgpu::PresentMode::Fifo
    }

    // Helper to find first available mode from preference list
    fn first_available(modes: &[wgpu::PresentMode], preferences: &[wgpu::PresentMode]) -> wgpu::PresentMode {
        for &pref in preferences {
            if modes.contains(&pref) {
                return pref;
            }
        }
        // Fallback to Fifo (always available per WGPU spec)
        wgpu::PresentMode::Fifo
    }
}

// ============================================================================
// VSyncFramePacer
// ============================================================================

/// Frame pacer with VSync-aware timing.
///
/// `VSyncFramePacer` extends basic frame pacing with VSync mode awareness,
/// providing appropriate timing based on the current VSync setting.
///
/// # Example
///
/// ```ignore
/// let mut pacer = VSyncFramePacer::new(60.0);
/// pacer.set_vsync_mode(VSyncMode::On);
///
/// // In render loop
/// pacer.begin_frame();
/// // ... render ...
/// pacer.end_frame();
/// pacer.wait_for_target();
///
/// println!("Remaining budget: {}ms", pacer.frame_budget_remaining_ms());
/// ```
#[derive(Debug, Clone)]
pub struct VSyncFramePacer {
    /// Target frame time in nanoseconds.
    target_frame_time_ns: u64,
    /// Last frame start time.
    last_frame_time: Instant,
    /// Current frame start time.
    current_frame_start: Instant,
    /// Whether sleeping is enabled for frame limiting.
    sleep_enabled: bool,
    /// Current VSync mode (affects pacing behavior).
    vsync_mode: VSyncMode,
    /// Accumulated time for spin-waiting.
    accumulated_ns: i64,
}

impl VSyncFramePacer {
    /// Create a new frame pacer with target FPS.
    pub fn new(target_fps: f64) -> Self {
        let target_frame_time_ns = if target_fps > 0.0 {
            (1_000_000_000.0 / target_fps) as u64
        } else {
            0
        };

        let now = Instant::now();
        Self {
            target_frame_time_ns,
            last_frame_time: now,
            current_frame_start: now,
            sleep_enabled: true,
            vsync_mode: VSyncMode::default(),
            accumulated_ns: 0,
        }
    }

    /// Create a frame pacer without target FPS (unlimited).
    pub fn unlimited() -> Self {
        let now = Instant::now();
        Self {
            target_frame_time_ns: 0,
            last_frame_time: now,
            current_frame_start: now,
            sleep_enabled: false,
            vsync_mode: VSyncMode::Off,
            accumulated_ns: 0,
        }
    }

    /// Set the target FPS.
    pub fn set_target_fps(&mut self, fps: f64) {
        if fps > 0.0 {
            self.target_frame_time_ns = (1_000_000_000.0 / fps) as u64;
        } else {
            self.target_frame_time_ns = 0;
        }
    }

    /// Get the current target FPS.
    pub fn target_fps(&self) -> f64 {
        if self.target_frame_time_ns > 0 {
            1_000_000_000.0 / self.target_frame_time_ns as f64
        } else {
            0.0
        }
    }

    /// Set the VSync mode (affects pacing behavior).
    pub fn set_vsync_mode(&mut self, mode: VSyncMode) {
        self.vsync_mode = mode;
        // Adjust sleep behavior based on mode
        self.sleep_enabled = mode != VSyncMode::Off;
    }

    /// Get the current VSync mode.
    pub fn vsync_mode(&self) -> VSyncMode {
        self.vsync_mode
    }

    /// Mark the beginning of a frame.
    pub fn begin_frame(&mut self) {
        self.current_frame_start = Instant::now();
    }

    /// Mark the end of a frame.
    pub fn end_frame(&mut self) {
        self.last_frame_time = Instant::now();
    }

    /// Wait to maintain target frame rate.
    ///
    /// If VSync is off, this does nothing (GPU/display handles timing).
    /// Otherwise, waits until the target frame time has elapsed.
    pub fn wait_for_target(&mut self) {
        if self.target_frame_time_ns == 0 || !self.sleep_enabled {
            return;
        }

        // VSync Off doesn't need software pacing
        if self.vsync_mode == VSyncMode::Off {
            return;
        }

        let elapsed_ns = self.current_frame_start.elapsed().as_nanos() as i64;
        let target_ns = self.target_frame_time_ns as i64;
        let remaining_ns = target_ns - elapsed_ns + self.accumulated_ns;

        if remaining_ns > 0 {
            // Sleep for most of the time, then spin for accuracy
            let sleep_threshold_ns = 1_500_000; // 1.5ms
            if remaining_ns > sleep_threshold_ns {
                std::thread::sleep(std::time::Duration::from_nanos(
                    (remaining_ns - sleep_threshold_ns) as u64
                ));
            }

            // Spin-wait for the rest
            let target_end = self.current_frame_start + std::time::Duration::from_nanos(target_ns as u64);
            while Instant::now() < target_end {
                std::hint::spin_loop();
            }

            // Track accumulated error
            let actual_ns = self.current_frame_start.elapsed().as_nanos() as i64;
            self.accumulated_ns = target_ns - actual_ns;
        } else {
            // We're behind - accumulate debt
            self.accumulated_ns = remaining_ns.max(-target_ns);
        }
    }

    /// Enable or disable sleep-based pacing.
    pub fn enable_sleep(&mut self, enabled: bool) {
        self.sleep_enabled = enabled;
    }

    /// Check if sleep is enabled.
    pub fn is_sleep_enabled(&self) -> bool {
        self.sleep_enabled
    }

    /// Get remaining frame budget in milliseconds.
    ///
    /// Returns how much time remains in the current frame's budget.
    /// Negative values indicate the frame is over budget.
    pub fn frame_budget_remaining_ms(&self) -> f64 {
        if self.target_frame_time_ns == 0 {
            return f64::INFINITY;
        }

        let elapsed_ns = self.current_frame_start.elapsed().as_nanos() as i64;
        let remaining_ns = self.target_frame_time_ns as i64 - elapsed_ns;
        remaining_ns as f64 / 1_000_000.0
    }

    /// Get the target frame time in milliseconds.
    pub fn target_frame_time_ms(&self) -> f64 {
        self.target_frame_time_ns as f64 / 1_000_000.0
    }

    /// Get the last frame time in milliseconds.
    pub fn last_frame_time_ms(&self) -> f64 {
        self.current_frame_start
            .duration_since(self.last_frame_time)
            .as_secs_f64() * 1000.0
    }

    /// Reset pacing state.
    pub fn reset(&mut self) {
        let now = Instant::now();
        self.last_frame_time = now;
        self.current_frame_start = now;
        self.accumulated_ns = 0;
    }
}

impl Default for VSyncFramePacer {
    fn default() -> Self {
        Self::new(60.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- VSyncMode tests --

    #[test]
    fn test_vsync_mode_default() {
        let mode = VSyncMode::default();
        assert_eq!(mode, VSyncMode::On);
    }

    #[test]
    fn test_vsync_mode_variants() {
        let modes = [
            VSyncMode::On,
            VSyncMode::Off,
            VSyncMode::Adaptive,
            VSyncMode::FastSync,
            VSyncMode::HalfRate,
        ];
        assert_eq!(modes.len(), 5);
    }

    #[test]
    fn test_vsync_mode_name() {
        assert_eq!(VSyncMode::On.name(), "VSync On");
        assert_eq!(VSyncMode::Off.name(), "VSync Off");
        assert_eq!(VSyncMode::Adaptive.name(), "Adaptive VSync");
        assert_eq!(VSyncMode::FastSync.name(), "Fast Sync");
        assert_eq!(VSyncMode::HalfRate.name(), "Half-Rate VSync");
    }

    #[test]
    fn test_vsync_mode_description() {
        assert!(!VSyncMode::On.description().is_empty());
        assert!(!VSyncMode::Off.description().is_empty());
    }

    #[test]
    fn test_vsync_mode_prevents_tearing() {
        assert!(VSyncMode::On.prevents_tearing());
        assert!(!VSyncMode::Off.prevents_tearing());
        assert!(VSyncMode::Adaptive.prevents_tearing());
        assert!(VSyncMode::FastSync.prevents_tearing());
        assert!(VSyncMode::HalfRate.prevents_tearing());
    }

    #[test]
    fn test_vsync_mode_caps_framerate() {
        assert!(VSyncMode::On.caps_framerate());
        assert!(!VSyncMode::Off.caps_framerate());
    }

    #[test]
    fn test_vsync_mode_power_efficient() {
        assert!(VSyncMode::On.is_power_efficient());
        assert!(!VSyncMode::Off.is_power_efficient());
        assert!(VSyncMode::Adaptive.is_power_efficient());
        assert!(!VSyncMode::FastSync.is_power_efficient());
    }

    #[test]
    fn test_vsync_mode_latency_rank() {
        assert_eq!(VSyncMode::Off.latency_rank(), 1);
        assert_eq!(VSyncMode::FastSync.latency_rank(), 2);
        assert_eq!(VSyncMode::On.latency_rank(), 4);
        assert_eq!(VSyncMode::HalfRate.latency_rank(), 5);
    }

    #[test]
    fn test_vsync_mode_to_present_mode() {
        assert_eq!(VSyncMode::On.to_present_mode(), wgpu::PresentMode::Fifo);
        assert_eq!(VSyncMode::Off.to_present_mode(), wgpu::PresentMode::Immediate);
        assert_eq!(VSyncMode::Adaptive.to_present_mode(), wgpu::PresentMode::FifoRelaxed);
        assert_eq!(VSyncMode::FastSync.to_present_mode(), wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_vsync_mode_from_present_mode() {
        assert_eq!(VSyncMode::from_present_mode(wgpu::PresentMode::Fifo), VSyncMode::On);
        assert_eq!(VSyncMode::from_present_mode(wgpu::PresentMode::Immediate), VSyncMode::Off);
        assert_eq!(VSyncMode::from_present_mode(wgpu::PresentMode::FifoRelaxed), VSyncMode::Adaptive);
        assert_eq!(VSyncMode::from_present_mode(wgpu::PresentMode::Mailbox), VSyncMode::FastSync);
    }

    #[test]
    fn test_vsync_mode_display() {
        let mode = VSyncMode::On;
        assert_eq!(format!("{}", mode), "VSync On");
    }

    #[test]
    fn test_vsync_mode_competitive_order() {
        let order = VSyncMode::competitive_order();
        assert_eq!(order[0], VSyncMode::Off); // Lowest latency first
        assert_eq!(order[4], VSyncMode::HalfRate); // Highest latency last
    }

    #[test]
    fn test_vsync_mode_power_saving_order() {
        let order = VSyncMode::power_saving_order();
        assert_eq!(order[0], VSyncMode::HalfRate); // Most efficient first
        assert_eq!(order[4], VSyncMode::Off); // Least efficient last
    }

    // -- VSyncPresentModeInfo tests --

    #[test]
    fn test_present_mode_info_from_present_mode() {
        let info = VSyncPresentModeInfo::from_present_mode(wgpu::PresentMode::Immediate);
        assert_eq!(info.mode, wgpu::PresentMode::Immediate);
        assert_eq!(info.vsync, VSyncMode::Off);
        assert!(info.tear_allowed);
        assert_eq!(info.latency_frames, 0);
        assert!(!info.power_efficient);
    }

    #[test]
    fn test_present_mode_info_from_vsync_mode() {
        let info = VSyncPresentModeInfo::from_vsync_mode(VSyncMode::FastSync);
        assert_eq!(info.vsync, VSyncMode::FastSync);
        assert_eq!(info.mode, wgpu::PresentMode::Mailbox);
        assert!(!info.tear_allowed);
        assert_eq!(info.latency_frames, 1);
    }

    #[test]
    fn test_present_mode_info_is_competitive() {
        let immediate = VSyncPresentModeInfo::from_present_mode(wgpu::PresentMode::Immediate);
        let fifo = VSyncPresentModeInfo::from_present_mode(wgpu::PresentMode::Fifo);

        assert!(immediate.is_competitive());
        assert!(!fifo.is_competitive());
    }

    #[test]
    fn test_present_mode_info_is_battery_friendly() {
        let fifo = VSyncPresentModeInfo::from_present_mode(wgpu::PresentMode::Fifo);
        let immediate = VSyncPresentModeInfo::from_present_mode(wgpu::PresentMode::Immediate);

        assert!(fifo.is_battery_friendly());
        assert!(!immediate.is_battery_friendly());
    }

    #[test]
    fn test_present_mode_info_estimated_lag() {
        let info = VSyncPresentModeInfo::from_present_mode(wgpu::PresentMode::Fifo);
        let lag = info.estimated_lag_ms_at_60hz();
        // Fifo has 2 frame latency = ~33.34ms
        assert!((lag - 33.34).abs() < 0.1);
    }

    #[test]
    fn test_present_mode_info_display() {
        let info = VSyncPresentModeInfo::from_vsync_mode(VSyncMode::On);
        let display = format!("{}", info);
        assert!(display.contains("VSync On"));
    }

    // -- VSyncController tests --

    #[test]
    fn test_vsync_controller_new() {
        let controller = VSyncController::new();
        assert_eq!(controller.mode(), VSyncMode::On);
        assert_eq!(controller.target_fps(), None);
    }

    #[test]
    fn test_vsync_controller_with_history_size() {
        let controller = VSyncController::with_history_size(120);
        assert_eq!(controller.history_size, 120);
    }

    #[test]
    fn test_vsync_controller_set_mode() {
        let mut controller = VSyncController::new();
        controller.set_mode(VSyncMode::Off);
        assert_eq!(controller.mode(), VSyncMode::Off);
        assert_eq!(controller.switch_count(), 1);
    }

    #[test]
    fn test_vsync_controller_set_target_fps() {
        let mut controller = VSyncController::new();
        controller.set_target_fps(Some(60.0));
        assert_eq!(controller.target_fps(), Some(60.0));

        controller.set_target_fps(None);
        assert_eq!(controller.target_fps(), None);
    }

    #[test]
    fn test_vsync_controller_set_target_fps_zero() {
        let mut controller = VSyncController::new();
        controller.set_target_fps(Some(0.0));
        assert_eq!(controller.target_fps(), None);
    }

    #[test]
    fn test_vsync_controller_record_frame_time() {
        let mut controller = VSyncController::new();
        controller.record_frame_time(16.67);
        controller.record_frame_time(16.5);
        controller.record_frame_time(17.0);

        assert_eq!(controller.frame_count(), 3);
    }

    #[test]
    fn test_vsync_controller_average_fps() {
        let mut controller = VSyncController::new();
        // 60 FPS = 16.67ms per frame
        for _ in 0..10 {
            controller.record_frame_time(16.67);
        }

        let fps = controller.average_fps();
        assert!((fps - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_vsync_controller_average_fps_empty() {
        let controller = VSyncController::new();
        assert_eq!(controller.average_fps(), 0.0);
    }

    #[test]
    fn test_vsync_controller_frame_time_stats() {
        let mut controller = VSyncController::new();
        controller.record_frame_time(10.0);
        controller.record_frame_time(20.0);
        controller.record_frame_time(15.0);

        assert_eq!(controller.min_frame_time_ms(), 10.0);
        assert_eq!(controller.max_frame_time_ms(), 20.0);
        assert_eq!(controller.average_frame_time_ms(), 15.0);
    }

    #[test]
    fn test_vsync_controller_variance() {
        let mut controller = VSyncController::new();
        // Same frame time = 0 variance
        for _ in 0..10 {
            controller.record_frame_time(16.67);
        }

        let variance = controller.frame_time_variance();
        assert!(variance < 0.01);
    }

    #[test]
    fn test_vsync_controller_should_adapt_not_adaptive() {
        let mut controller = VSyncController::new();
        controller.set_mode(VSyncMode::On);
        controller.set_target_fps(Some(60.0));

        for _ in 0..60 {
            controller.record_frame_time(30.0); // 30ms = 33 FPS
        }

        // Should not adapt in non-adaptive mode
        assert!(!controller.should_adapt());
    }

    #[test]
    fn test_vsync_controller_adaptive_threshold() {
        let mut controller = VSyncController::new();
        controller.set_adaptive_threshold(0.8);
        assert_eq!(controller.adaptive_threshold(), 0.8);

        // Clamping
        controller.set_adaptive_threshold(0.3);
        assert_eq!(controller.adaptive_threshold(), 0.5);
        controller.set_adaptive_threshold(1.5);
        assert_eq!(controller.adaptive_threshold(), 1.0);
    }

    #[test]
    fn test_vsync_controller_to_present_mode() {
        let mut controller = VSyncController::new();
        controller.set_mode(VSyncMode::FastSync);
        assert_eq!(controller.to_present_mode(), wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_vsync_controller_clear_history() {
        let mut controller = VSyncController::new();
        controller.record_frame_time(16.67);
        controller.record_frame_time(16.67);

        controller.clear_history();
        assert_eq!(controller.frame_count(), 0);
    }

    #[test]
    fn test_vsync_controller_reset() {
        let mut controller = VSyncController::new();
        controller.set_mode(VSyncMode::Off);
        controller.set_target_fps(Some(144.0));
        controller.record_frame_time(7.0);

        controller.reset();
        assert_eq!(controller.mode(), VSyncMode::On);
        assert_eq!(controller.target_fps(), None);
        assert_eq!(controller.frame_count(), 0);
    }

    #[test]
    fn test_vsync_controller_mode_info() {
        let controller = VSyncController::new();
        let info = controller.mode_info();
        assert_eq!(info.vsync, VSyncMode::On);
    }

    #[test]
    fn test_vsync_controller_default() {
        let controller = VSyncController::default();
        assert_eq!(controller.mode(), VSyncMode::On);
    }

    // -- PresentModeSelector tests --

    #[test]
    fn test_present_mode_selector_supports_vsync() {
        let modes = vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Immediate];
        assert!(PresentModeSelector::supports_vsync(&modes));

        let modes_no_vsync = vec![wgpu::PresentMode::Immediate];
        assert!(!PresentModeSelector::supports_vsync(&modes_no_vsync));
    }

    #[test]
    fn test_present_mode_selector_supports_immediate() {
        let modes = vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Immediate];
        assert!(PresentModeSelector::supports_immediate(&modes));

        let modes_no_immediate = vec![wgpu::PresentMode::Fifo];
        assert!(!PresentModeSelector::supports_immediate(&modes_no_immediate));
    }

    #[test]
    fn test_present_mode_selector_supports_mailbox() {
        let modes = vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox];
        assert!(PresentModeSelector::supports_mailbox(&modes));

        let modes_no_mailbox = vec![wgpu::PresentMode::Fifo];
        assert!(!PresentModeSelector::supports_mailbox(&modes_no_mailbox));
    }

    #[test]
    fn test_present_mode_selector_supports_adaptive() {
        let modes = vec![wgpu::PresentMode::FifoRelaxed];
        assert!(PresentModeSelector::supports_adaptive(&modes));
    }

    #[test]
    fn test_present_mode_selector_best_for_vsync_on() {
        let modes = vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
        ];
        let best = PresentModeSelector::best_for_vsync(&modes, VSyncMode::On);
        assert_eq!(best, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_present_mode_selector_best_for_vsync_off() {
        let modes = vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
        ];
        let best = PresentModeSelector::best_for_vsync(&modes, VSyncMode::Off);
        assert_eq!(best, wgpu::PresentMode::Immediate);
    }

    #[test]
    fn test_present_mode_selector_best_for_vsync_fallback() {
        // Only Fifo available
        let modes = vec![wgpu::PresentMode::Fifo];
        let best = PresentModeSelector::best_for_vsync(&modes, VSyncMode::Off);
        // Falls back to Fifo when Immediate not available
        assert_eq!(best, wgpu::PresentMode::Fifo);
    }

    #[test]
    fn test_present_mode_selector_best_for_fast_sync() {
        let modes = vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
        ];
        let best = PresentModeSelector::best_for_vsync(&modes, VSyncMode::FastSync);
        assert_eq!(best, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_present_mode_selector_best_for_adaptive() {
        let modes = vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::FifoRelaxed,
        ];
        let best = PresentModeSelector::best_for_vsync(&modes, VSyncMode::Adaptive);
        assert_eq!(best, wgpu::PresentMode::FifoRelaxed);
    }

    #[test]
    fn test_present_mode_selector_mode_info() {
        let info = PresentModeSelector::mode_info(wgpu::PresentMode::Mailbox);
        assert_eq!(info.mode, wgpu::PresentMode::Mailbox);
        assert_eq!(info.vsync, VSyncMode::FastSync);
    }

    #[test]
    fn test_present_mode_selector_all_mode_info() {
        let modes = vec![wgpu::PresentMode::Fifo, wgpu::PresentMode::Immediate];
        let infos = PresentModeSelector::all_mode_info(&modes);
        assert_eq!(infos.len(), 2);
    }

    #[test]
    fn test_present_mode_selector_lowest_latency() {
        let modes = vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
        ];
        let lowest = PresentModeSelector::lowest_latency(&modes);
        assert_eq!(lowest, Some(wgpu::PresentMode::Immediate));
    }

    #[test]
    fn test_present_mode_selector_most_power_efficient() {
        let modes = vec![
            wgpu::PresentMode::Fifo,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
        ];
        let efficient = PresentModeSelector::most_power_efficient(&modes);
        assert_eq!(efficient, Some(wgpu::PresentMode::Fifo));
    }

    #[test]
    fn test_present_mode_selector_default_mode() {
        assert_eq!(PresentModeSelector::default_mode(), wgpu::PresentMode::Fifo);
    }

    // -- VSyncFramePacer tests --

    #[test]
    fn test_vsync_frame_pacer_new() {
        let pacer = VSyncFramePacer::new(60.0);
        assert!((pacer.target_fps() - 60.0).abs() < 0.01);
        assert!(pacer.is_sleep_enabled());
    }

    #[test]
    fn test_vsync_frame_pacer_unlimited() {
        let pacer = VSyncFramePacer::unlimited();
        assert_eq!(pacer.target_fps(), 0.0);
        assert!(!pacer.is_sleep_enabled());
        assert_eq!(pacer.vsync_mode(), VSyncMode::Off);
    }

    #[test]
    fn test_vsync_frame_pacer_set_target_fps() {
        let mut pacer = VSyncFramePacer::new(60.0);
        pacer.set_target_fps(144.0);
        assert!((pacer.target_fps() - 144.0).abs() < 0.01);
    }

    #[test]
    fn test_vsync_frame_pacer_set_vsync_mode() {
        let mut pacer = VSyncFramePacer::new(60.0);
        pacer.set_vsync_mode(VSyncMode::Off);
        assert_eq!(pacer.vsync_mode(), VSyncMode::Off);
        assert!(!pacer.is_sleep_enabled());
    }

    #[test]
    fn test_vsync_frame_pacer_begin_end_frame() {
        let mut pacer = VSyncFramePacer::new(60.0);
        pacer.begin_frame();
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(1));
        pacer.end_frame();

        // Frame time should be recorded
        let frame_time = pacer.last_frame_time_ms();
        assert!(frame_time >= 0.0);
    }

    #[test]
    fn test_vsync_frame_pacer_frame_budget_remaining() {
        let mut pacer = VSyncFramePacer::new(60.0);
        pacer.begin_frame();

        // Immediately after begin, should have nearly full budget
        let remaining = pacer.frame_budget_remaining_ms();
        assert!(remaining > 15.0); // Most of 16.67ms remaining
    }

    #[test]
    fn test_vsync_frame_pacer_target_frame_time_ms() {
        let pacer = VSyncFramePacer::new(60.0);
        let target = pacer.target_frame_time_ms();
        // 60 FPS = ~16.67ms
        assert!((target - 16.67).abs() < 0.1);
    }

    #[test]
    fn test_vsync_frame_pacer_enable_sleep() {
        let mut pacer = VSyncFramePacer::new(60.0);
        assert!(pacer.is_sleep_enabled());

        pacer.enable_sleep(false);
        assert!(!pacer.is_sleep_enabled());

        pacer.enable_sleep(true);
        assert!(pacer.is_sleep_enabled());
    }

    #[test]
    fn test_vsync_frame_pacer_reset() {
        let mut pacer = VSyncFramePacer::new(60.0);
        pacer.begin_frame();
        std::thread::sleep(std::time::Duration::from_millis(1));

        pacer.reset();
        // After reset, accumulated_ns should be 0
        assert_eq!(pacer.accumulated_ns, 0);
    }

    #[test]
    fn test_vsync_frame_pacer_default() {
        let pacer = VSyncFramePacer::default();
        assert!((pacer.target_fps() - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_vsync_frame_pacer_unlimited_budget() {
        let mut pacer = VSyncFramePacer::unlimited();
        pacer.begin_frame();

        let remaining = pacer.frame_budget_remaining_ms();
        assert!(remaining.is_infinite());
    }

    // -- API compatibility tests --

    #[test]
    fn test_vsync_controller_api_set_mode() {
        fn _check<F: FnOnce(&mut VSyncController, VSyncMode)>(_f: F) {}
        _check(|c, m| c.set_mode(m));
    }

    #[test]
    fn test_vsync_controller_api_mode() {
        fn _check<F: FnOnce(&VSyncController) -> VSyncMode>(_f: F) {}
        _check(|c| c.mode());
    }

    #[test]
    fn test_vsync_controller_api_set_target_fps() {
        fn _check<F: FnOnce(&mut VSyncController, Option<f64>)>(_f: F) {}
        _check(|c, f| c.set_target_fps(f));
    }

    #[test]
    fn test_vsync_controller_api_target_fps() {
        fn _check<F: FnOnce(&VSyncController) -> Option<f64>>(_f: F) {}
        _check(|c| c.target_fps());
    }

    #[test]
    fn test_vsync_controller_api_record_frame_time() {
        fn _check<F: FnOnce(&mut VSyncController, f64)>(_f: F) {}
        _check(|c, t| c.record_frame_time(t));
    }

    #[test]
    fn test_vsync_controller_api_average_fps() {
        fn _check<F: FnOnce(&VSyncController) -> f64>(_f: F) {}
        _check(|c| c.average_fps());
    }

    #[test]
    fn test_vsync_controller_api_should_adapt() {
        fn _check<F: FnOnce(&mut VSyncController) -> bool>(_f: F) {}
        _check(|c| c.should_adapt());
    }

    #[test]
    fn test_vsync_controller_api_to_present_mode() {
        fn _check<F: FnOnce(&VSyncController) -> wgpu::PresentMode>(_f: F) {}
        _check(|c| c.to_present_mode());
    }

    #[test]
    fn test_vsync_frame_pacer_api_begin_frame() {
        fn _check<F: FnOnce(&mut VSyncFramePacer)>(_f: F) {}
        _check(|p| p.begin_frame());
    }

    #[test]
    fn test_vsync_frame_pacer_api_end_frame() {
        fn _check<F: FnOnce(&mut VSyncFramePacer)>(_f: F) {}
        _check(|p| p.end_frame());
    }

    #[test]
    fn test_vsync_frame_pacer_api_wait_for_target() {
        fn _check<F: FnOnce(&mut VSyncFramePacer)>(_f: F) {}
        _check(|p| p.wait_for_target());
    }

    #[test]
    fn test_vsync_frame_pacer_api_enable_sleep() {
        fn _check<F: FnOnce(&mut VSyncFramePacer, bool)>(_f: F) {}
        _check(|p, e| p.enable_sleep(e));
    }

    #[test]
    fn test_vsync_frame_pacer_api_frame_budget_remaining_ms() {
        fn _check<F: FnOnce(&VSyncFramePacer) -> f64>(_f: F) {}
        _check(|p| p.frame_budget_remaining_ms());
    }
}
