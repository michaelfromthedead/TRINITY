//! Shadow Fallback Chain
//!
//! This module implements a fallback chain for shadow rendering techniques,
//! providing graceful degradation from ray-traced shadows to rasterized
//! alternatives based on device capabilities.
//!
//! # Architecture
//!
//! - `ShadowTechnique`: Available shadow rendering techniques
//! - `ShadowFallbackChain`: Manages technique selection and transitions
//! - `ShadowConfig`: Quality configuration for rasterized shadow techniques
//! - `TransitionState`: Smooth blending between technique changes
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::shadow_fallback::{ShadowFallbackChain, ShadowConfig};
//! use renderer_backend::rt_capability::RTCapability;
//!
//! let mut chain = ShadowFallbackChain::new(RTCapability::RayQueryOnly);
//! let technique = chain.select_technique();
//!
//! // If RT fails, downgrade gracefully
//! if rt_error_occurred {
//!     chain.downgrade();
//! }
//!
//! let config = ShadowConfig::high();
//! ```

use crate::rt_capability::RTCapability;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default transition duration in seconds for technique changes.
const DEFAULT_TRANSITION_DURATION: f32 = 0.5;

/// Minimum CSM cascade count.
const MIN_CSM_CASCADES: u32 = 2;

/// Maximum CSM cascade count.
const MAX_CSM_CASCADES: u32 = 4;

/// Minimum shadow map resolution.
const MIN_SHADOW_RESOLUTION: u32 = 512;

/// Maximum shadow map resolution.
const MAX_SHADOW_RESOLUTION: u32 = 4096;

/// Minimum PCSS sample count.
const MIN_PCSS_SAMPLES: u32 = 0;

/// Maximum PCSS sample count.
const MAX_PCSS_SAMPLES: u32 = 32;

/// Minimum contact shadow ray count.
const MIN_CONTACT_RAYS: u32 = 4;

/// Maximum contact shadow ray count.
const MAX_CONTACT_RAYS: u32 = 16;

/// Default contact shadow maximum distance.
const DEFAULT_CONTACT_SHADOW_MAX_DISTANCE: f32 = 0.5;

// ---------------------------------------------------------------------------
// ShadowTechnique
// ---------------------------------------------------------------------------

/// Shadow rendering technique.
///
/// Ordered by quality level: lower enum discriminant = lower quality.
/// This enables natural comparison: `RayTraced > CascadedShadowMaps > ContactShadows > None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShadowTechnique {
    /// No shadow rendering.
    None,

    /// Contact shadows using screen-space ray marching.
    /// Adds fine detail at contact points, typically combined with CSM.
    ContactShadows,

    /// Cascaded shadow maps for directional lights with PCSS soft shadows.
    /// High-quality rasterized fallback with proper penumbra.
    CascadedShadowMaps,

    /// Ray-traced shadows with pixel-perfect accuracy.
    /// Requires RT capability (ray query or full pipeline).
    RayTraced,
}

impl Default for ShadowTechnique {
    fn default() -> Self {
        ShadowTechnique::CascadedShadowMaps
    }
}

impl ShadowTechnique {
    /// Returns `true` if this technique uses ray tracing.
    pub fn is_ray_traced(&self) -> bool {
        matches!(self, ShadowTechnique::RayTraced)
    }

    /// Returns `true` if this technique uses rasterization.
    pub fn is_rasterized(&self) -> bool {
        matches!(
            self,
            ShadowTechnique::CascadedShadowMaps | ShadowTechnique::ContactShadows
        )
    }

    /// Returns `true` if shadows are enabled.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, ShadowTechnique::None)
    }

    /// Returns `true` if this technique requires RT capability.
    pub fn requires_rt(&self) -> bool {
        matches!(self, ShadowTechnique::RayTraced)
    }

    /// Returns a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            ShadowTechnique::RayTraced => "Ray-traced shadows",
            ShadowTechnique::CascadedShadowMaps => "Cascaded shadow maps with PCSS",
            ShadowTechnique::ContactShadows => "Contact shadows (screen-space)",
            ShadowTechnique::None => "Shadows disabled",
        }
    }

    /// Returns the technique one level below in quality.
    pub fn downgrade(&self) -> Self {
        match self {
            ShadowTechnique::RayTraced => ShadowTechnique::CascadedShadowMaps,
            ShadowTechnique::CascadedShadowMaps => ShadowTechnique::ContactShadows,
            ShadowTechnique::ContactShadows => ShadowTechnique::None,
            ShadowTechnique::None => ShadowTechnique::None,
        }
    }

    /// Returns the technique one level above in quality.
    pub fn upgrade(&self) -> Self {
        match self {
            ShadowTechnique::None => ShadowTechnique::ContactShadows,
            ShadowTechnique::ContactShadows => ShadowTechnique::CascadedShadowMaps,
            ShadowTechnique::CascadedShadowMaps => ShadowTechnique::RayTraced,
            ShadowTechnique::RayTraced => ShadowTechnique::RayTraced,
        }
    }

    /// Returns the quality level as a value from 0 (None) to 3 (RayTraced).
    pub fn quality_level(&self) -> u32 {
        match self {
            ShadowTechnique::None => 0,
            ShadowTechnique::ContactShadows => 1,
            ShadowTechnique::CascadedShadowMaps => 2,
            ShadowTechnique::RayTraced => 3,
        }
    }
}

impl std::fmt::Display for ShadowTechnique {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ---------------------------------------------------------------------------
// TransitionState
// ---------------------------------------------------------------------------

/// State for smooth transitions between shadow techniques.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionState {
    /// The technique we're transitioning from.
    pub from: ShadowTechnique,

    /// The technique we're transitioning to.
    pub to: ShadowTechnique,

    /// Progress of the transition (0.0 = start, 1.0 = complete).
    pub progress: f32,

    /// Duration of the transition in seconds.
    pub duration: f32,
}

impl TransitionState {
    /// Create a new transition state.
    pub fn new(from: ShadowTechnique, to: ShadowTechnique, duration: f32) -> Self {
        Self {
            from,
            to,
            progress: 0.0,
            duration: duration.max(0.001), // Prevent division by zero
        }
    }

    /// Create a completed (instant) transition.
    pub fn instant(technique: ShadowTechnique) -> Self {
        Self {
            from: technique,
            to: technique,
            progress: 1.0,
            duration: 0.0,
        }
    }

    /// Returns `true` if the transition is complete.
    pub fn is_complete(&self) -> bool {
        self.progress >= 1.0
    }

    /// Returns `true` if the transition is in progress.
    pub fn is_transitioning(&self) -> bool {
        self.progress < 1.0 && self.from != self.to
    }

    /// Update the transition progress based on elapsed time.
    ///
    /// Returns `true` if the transition just completed.
    pub fn update(&mut self, delta_time: f32) -> bool {
        if self.is_complete() {
            return false;
        }

        let was_complete = self.is_complete();
        self.progress = (self.progress + delta_time / self.duration).min(1.0);
        !was_complete && self.is_complete()
    }

    /// Returns the current blend factor for the target technique.
    ///
    /// Use this for alpha blending: 0.0 = fully `from`, 1.0 = fully `to`.
    pub fn blend_factor(&self) -> f32 {
        // Smooth step for better visual transitions
        let t = self.progress.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Returns the current effective technique.
    ///
    /// During transition, returns `from` until progress >= 0.5, then `to`.
    pub fn current_technique(&self) -> ShadowTechnique {
        if self.progress >= 0.5 {
            self.to
        } else {
            self.from
        }
    }
}

impl Default for TransitionState {
    fn default() -> Self {
        Self::instant(ShadowTechnique::default())
    }
}

// ---------------------------------------------------------------------------
// ShadowFallbackChain
// ---------------------------------------------------------------------------

/// Manages shadow technique selection and fallback behavior.
///
/// The fallback chain orders techniques from preferred (highest quality)
/// to fallback (lower quality), enabling graceful degradation when
/// RT is unavailable or performance issues occur.
#[derive(Debug, Clone)]
pub struct ShadowFallbackChain {
    /// The preferred technique based on initial capability detection.
    preferred: ShadowTechnique,

    /// Ordered list of fallback techniques (excluding preferred).
    fallbacks: Vec<ShadowTechnique>,

    /// Currently active technique.
    current: ShadowTechnique,

    /// Current transition state.
    transition: TransitionState,

    /// Index into the fallback chain (0 = preferred, 1+ = fallbacks).
    fallback_index: usize,

    /// RT capability that determined the initial chain.
    rt_capability: RTCapability,

    /// Transition duration for technique changes.
    transition_duration: f32,
}

impl ShadowFallbackChain {
    /// Create a new fallback chain based on RT capability.
    ///
    /// The chain is constructed to provide the best possible shadow quality
    /// that the device can support, with graceful fallbacks.
    pub fn new(rt_capability: RTCapability) -> Self {
        let (preferred, fallbacks) = Self::build_chain(rt_capability);
        Self {
            preferred,
            fallbacks,
            current: preferred,
            transition: TransitionState::instant(preferred),
            fallback_index: 0,
            rt_capability,
            transition_duration: DEFAULT_TRANSITION_DURATION,
        }
    }

    /// Build the fallback chain based on capability.
    fn build_chain(capability: RTCapability) -> (ShadowTechnique, Vec<ShadowTechnique>) {
        match capability {
            RTCapability::Full | RTCapability::RayQueryOnly => {
                // RT available: RT -> CSM -> Contact -> None
                (
                    ShadowTechnique::RayTraced,
                    vec![
                        ShadowTechnique::CascadedShadowMaps,
                        ShadowTechnique::ContactShadows,
                        ShadowTechnique::None,
                    ],
                )
            }
            RTCapability::None => {
                // No RT: CSM -> Contact -> None
                (
                    ShadowTechnique::CascadedShadowMaps,
                    vec![ShadowTechnique::ContactShadows, ShadowTechnique::None],
                )
            }
        }
    }

    /// Select and return the current best technique.
    pub fn select_technique(&mut self) -> ShadowTechnique {
        self.current
    }

    /// Move to the next fallback technique.
    ///
    /// Returns `true` if downgrade was possible, `false` if already at lowest.
    pub fn downgrade(&mut self) -> bool {
        if self.fallback_index < self.fallbacks.len() {
            let from = self.current;
            self.fallback_index += 1;

            self.current = if self.fallback_index == 0 {
                self.preferred
            } else {
                self.fallbacks[self.fallback_index - 1]
            };

            self.transition = TransitionState::new(from, self.current, self.transition_duration);
            true
        } else {
            false
        }
    }

    /// Try to restore a higher quality technique.
    ///
    /// Returns `true` if upgrade was possible, `false` if already at highest.
    pub fn upgrade(&mut self) -> bool {
        if self.fallback_index > 0 {
            let from = self.current;
            self.fallback_index -= 1;

            self.current = if self.fallback_index == 0 {
                self.preferred
            } else {
                self.fallbacks[self.fallback_index - 1]
            };

            self.transition = TransitionState::new(from, self.current, self.transition_duration);
            true
        } else {
            false
        }
    }

    /// Get the current technique.
    pub fn current_technique(&self) -> ShadowTechnique {
        self.current
    }

    /// Returns `true` if RT shadows are currently supported.
    pub fn supports_rt(&self) -> bool {
        self.rt_capability.has_ray_query()
    }

    /// Returns `true` if currently using RT shadows.
    pub fn is_using_rt(&self) -> bool {
        self.current.is_ray_traced()
    }

    /// Returns the preferred (highest quality) technique.
    pub fn preferred_technique(&self) -> ShadowTechnique {
        self.preferred
    }

    /// Get the current transition state.
    pub fn transition_state(&self) -> &TransitionState {
        &self.transition
    }

    /// Returns `true` if a transition is in progress.
    pub fn is_transitioning(&self) -> bool {
        self.transition.is_transitioning()
    }

    /// Update transition progress.
    ///
    /// Call this each frame with delta time to animate transitions.
    pub fn update(&mut self, delta_time: f32) -> bool {
        self.transition.update(delta_time)
    }

    /// Get the current blend factor for technique interpolation.
    pub fn blend_factor(&self) -> f32 {
        self.transition.blend_factor()
    }

    /// Set the transition duration for future technique changes.
    pub fn set_transition_duration(&mut self, duration: f32) {
        self.transition_duration = duration.max(0.0);
    }

    /// Get the transition duration.
    pub fn transition_duration(&self) -> f32 {
        self.transition_duration
    }

    /// Get the RT capability level.
    pub fn rt_capability(&self) -> RTCapability {
        self.rt_capability
    }

    /// Reset to the preferred technique.
    pub fn reset(&mut self) {
        let from = self.current;
        self.fallback_index = 0;
        self.current = self.preferred;
        self.transition = TransitionState::new(from, self.current, self.transition_duration);
    }

    /// Force a specific technique, bypassing the chain.
    ///
    /// Returns `false` if the technique requires RT but RT is not available.
    pub fn force_technique(&mut self, technique: ShadowTechnique) -> bool {
        if technique.requires_rt() && !self.supports_rt() {
            return false;
        }

        let from = self.current;
        self.current = technique;

        // Update fallback index to match
        if technique == self.preferred {
            self.fallback_index = 0;
        } else if let Some(idx) = self.fallbacks.iter().position(|&t| t == technique) {
            self.fallback_index = idx + 1;
        }

        self.transition = TransitionState::new(from, self.current, self.transition_duration);
        true
    }

    /// Get the number of available fallback levels (including preferred).
    pub fn chain_length(&self) -> usize {
        1 + self.fallbacks.len()
    }

    /// Get the current fallback index (0 = preferred).
    pub fn current_fallback_index(&self) -> usize {
        self.fallback_index
    }

    /// Returns `true` if we're at the lowest fallback level.
    pub fn is_at_lowest(&self) -> bool {
        self.fallback_index >= self.fallbacks.len()
    }

    /// Returns `true` if we're using the preferred technique.
    pub fn is_at_preferred(&self) -> bool {
        self.fallback_index == 0
    }
}

impl Default for ShadowFallbackChain {
    fn default() -> Self {
        Self::new(RTCapability::default())
    }
}

// ---------------------------------------------------------------------------
// ShadowConfig
// ---------------------------------------------------------------------------

/// Configuration for shadow rendering quality.
///
/// Contains settings for CSM, PCSS, and contact shadows that can be
/// tuned based on performance requirements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowConfig {
    /// Number of CSM cascades (2-4).
    pub csm_cascade_count: u32,

    /// Shadow map resolution per cascade (512-4096).
    pub csm_resolution: u32,

    /// Number of PCSS samples for soft shadows (0-32, 0 = hard shadows).
    pub pcss_sample_count: u32,

    /// Number of rays for contact shadow tracing (4-16).
    pub contact_shadow_ray_count: u32,

    /// Maximum distance for contact shadow tracing in world units.
    pub contact_shadow_max_distance: f32,
}

impl ShadowConfig {
    /// Create a new shadow configuration with custom values.
    ///
    /// Values are clamped to valid ranges.
    pub fn new(
        csm_cascade_count: u32,
        csm_resolution: u32,
        pcss_sample_count: u32,
        contact_shadow_ray_count: u32,
        contact_shadow_max_distance: f32,
    ) -> Self {
        Self {
            csm_cascade_count: csm_cascade_count.clamp(MIN_CSM_CASCADES, MAX_CSM_CASCADES),
            csm_resolution: csm_resolution
                .clamp(MIN_SHADOW_RESOLUTION, MAX_SHADOW_RESOLUTION)
                .next_power_of_two(),
            pcss_sample_count: pcss_sample_count.clamp(MIN_PCSS_SAMPLES, MAX_PCSS_SAMPLES),
            contact_shadow_ray_count: contact_shadow_ray_count
                .clamp(MIN_CONTACT_RAYS, MAX_CONTACT_RAYS),
            contact_shadow_max_distance: contact_shadow_max_distance.max(0.0),
        }
    }

    /// Low quality preset: minimal resource usage.
    ///
    /// - 2 cascades at 1024 resolution
    /// - No PCSS (hard shadows)
    /// - 4 contact shadow rays
    pub fn low() -> Self {
        Self {
            csm_cascade_count: 2,
            csm_resolution: 1024,
            pcss_sample_count: 0,
            contact_shadow_ray_count: 4,
            contact_shadow_max_distance: DEFAULT_CONTACT_SHADOW_MAX_DISTANCE,
        }
    }

    /// Medium quality preset: balanced quality and performance.
    ///
    /// - 3 cascades at 2048 resolution
    /// - 8 PCSS samples
    /// - 8 contact shadow rays
    pub fn medium() -> Self {
        Self {
            csm_cascade_count: 3,
            csm_resolution: 2048,
            pcss_sample_count: 8,
            contact_shadow_ray_count: 8,
            contact_shadow_max_distance: DEFAULT_CONTACT_SHADOW_MAX_DISTANCE,
        }
    }

    /// High quality preset: high quality rasterized shadows.
    ///
    /// - 4 cascades at 2048 resolution
    /// - 16 PCSS samples
    /// - 12 contact shadow rays
    pub fn high() -> Self {
        Self {
            csm_cascade_count: 4,
            csm_resolution: 2048,
            pcss_sample_count: 16,
            contact_shadow_ray_count: 12,
            contact_shadow_max_distance: DEFAULT_CONTACT_SHADOW_MAX_DISTANCE,
        }
    }

    /// Ultra quality preset: RT shadows (config values are fallback settings).
    ///
    /// - 4 cascades at 4096 resolution (for fallback)
    /// - 32 PCSS samples
    /// - 16 contact shadow rays
    ///
    /// Note: When RT shadows are active, these settings are not used.
    /// They serve as high-quality fallback if RT becomes unavailable.
    pub fn ultra() -> Self {
        Self {
            csm_cascade_count: 4,
            csm_resolution: 4096,
            pcss_sample_count: 32,
            contact_shadow_ray_count: 16,
            contact_shadow_max_distance: DEFAULT_CONTACT_SHADOW_MAX_DISTANCE,
        }
    }

    /// Returns `true` if PCSS soft shadows are enabled.
    pub fn has_pcss(&self) -> bool {
        self.pcss_sample_count > 0
    }

    /// Returns the total shadow map memory usage estimate in bytes.
    ///
    /// Assumes 32-bit depth format (4 bytes per texel).
    pub fn estimated_memory_usage(&self) -> usize {
        let texels_per_cascade = (self.csm_resolution * self.csm_resolution) as usize;
        let bytes_per_texel = 4; // D32_FLOAT
        texels_per_cascade * self.csm_cascade_count as usize * bytes_per_texel
    }

    /// Returns the total number of shadow samples per pixel.
    pub fn total_samples_per_pixel(&self) -> u32 {
        // CSM lookup + PCSS samples + contact shadow rays
        1 + self.pcss_sample_count + self.contact_shadow_ray_count
    }

    /// Interpolate between two configs for smooth quality transitions.
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let inv_t = 1.0 - t;

        Self {
            csm_cascade_count: ((a.csm_cascade_count as f32 * inv_t
                + b.csm_cascade_count as f32 * t)
                .round() as u32)
                .clamp(MIN_CSM_CASCADES, MAX_CSM_CASCADES),
            csm_resolution: if t < 0.5 {
                a.csm_resolution
            } else {
                b.csm_resolution
            },
            pcss_sample_count: ((a.pcss_sample_count as f32 * inv_t
                + b.pcss_sample_count as f32 * t)
                .round() as u32)
                .clamp(MIN_PCSS_SAMPLES, MAX_PCSS_SAMPLES),
            contact_shadow_ray_count: ((a.contact_shadow_ray_count as f32 * inv_t
                + b.contact_shadow_ray_count as f32 * t)
                .round() as u32)
                .clamp(MIN_CONTACT_RAYS, MAX_CONTACT_RAYS),
            contact_shadow_max_distance: a.contact_shadow_max_distance * inv_t
                + b.contact_shadow_max_distance * t,
        }
    }

    /// Get a config for the given quality level (0-3).
    pub fn from_quality_level(level: u32) -> Self {
        match level {
            0 => Self::low(),
            1 => Self::medium(),
            2 => Self::high(),
            _ => Self::ultra(),
        }
    }
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self::medium()
    }
}

// ---------------------------------------------------------------------------
// ShadowQualityController
// ---------------------------------------------------------------------------

/// Controls shadow quality based on performance feedback.
///
/// Combines the fallback chain with config management for adaptive
/// shadow quality control.
#[derive(Debug, Clone)]
pub struct ShadowQualityController {
    /// The fallback chain for technique selection.
    chain: ShadowFallbackChain,

    /// Current shadow configuration.
    config: ShadowConfig,

    /// Target frame time in seconds (e.g., 1/60 for 60 FPS).
    target_frame_time: f32,

    /// Frame time threshold above which to consider downgrading.
    downgrade_threshold: f32,

    /// Frame time threshold below which to consider upgrading.
    upgrade_threshold: f32,

    /// Number of consecutive frames above/below threshold before acting.
    stability_frames: u32,

    /// Counter for consecutive frames above threshold.
    frames_above_threshold: u32,

    /// Counter for consecutive frames below threshold.
    frames_below_threshold: u32,
}

impl ShadowQualityController {
    /// Create a new quality controller.
    ///
    /// # Arguments
    ///
    /// * `rt_capability` - Device RT capability
    /// * `target_fps` - Target frame rate (e.g., 60)
    pub fn new(rt_capability: RTCapability, target_fps: f32) -> Self {
        let target_frame_time = 1.0 / target_fps.max(1.0);
        Self {
            chain: ShadowFallbackChain::new(rt_capability),
            config: ShadowConfig::from_quality_level(
                if rt_capability.has_ray_query() { 3 } else { 2 },
            ),
            target_frame_time,
            downgrade_threshold: target_frame_time * 1.2, // 20% over budget
            upgrade_threshold: target_frame_time * 0.8,   // 20% under budget
            stability_frames: 10,
            frames_above_threshold: 0,
            frames_below_threshold: 0,
        }
    }

    /// Update the controller with the current frame time.
    ///
    /// Returns `true` if quality was adjusted.
    pub fn update(&mut self, frame_time: f32, delta_time: f32) -> bool {
        // Update transition
        self.chain.update(delta_time);

        // Check frame time against thresholds
        if frame_time > self.downgrade_threshold {
            self.frames_above_threshold += 1;
            self.frames_below_threshold = 0;

            if self.frames_above_threshold >= self.stability_frames {
                self.frames_above_threshold = 0;
                return self.downgrade();
            }
        } else if frame_time < self.upgrade_threshold {
            self.frames_below_threshold += 1;
            self.frames_above_threshold = 0;

            if self.frames_below_threshold >= self.stability_frames {
                self.frames_below_threshold = 0;
                return self.upgrade();
            }
        } else {
            self.frames_above_threshold = 0;
            self.frames_below_threshold = 0;
        }

        false
    }

    /// Manually trigger a quality downgrade.
    pub fn downgrade(&mut self) -> bool {
        if self.chain.downgrade() {
            self.config = ShadowConfig::from_quality_level(
                self.chain.current_technique().quality_level(),
            );
            true
        } else {
            false
        }
    }

    /// Manually trigger a quality upgrade.
    pub fn upgrade(&mut self) -> bool {
        if self.chain.upgrade() {
            self.config = ShadowConfig::from_quality_level(
                self.chain.current_technique().quality_level(),
            );
            true
        } else {
            false
        }
    }

    /// Get the current shadow technique.
    pub fn current_technique(&self) -> ShadowTechnique {
        self.chain.current_technique()
    }

    /// Get the current shadow config.
    pub fn current_config(&self) -> &ShadowConfig {
        &self.config
    }

    /// Get the fallback chain.
    pub fn chain(&self) -> &ShadowFallbackChain {
        &self.chain
    }

    /// Get mutable access to the fallback chain.
    pub fn chain_mut(&mut self) -> &mut ShadowFallbackChain {
        &mut self.chain
    }

    /// Get the current blend factor for transitions.
    pub fn blend_factor(&self) -> f32 {
        self.chain.blend_factor()
    }

    /// Returns `true` if a transition is in progress.
    pub fn is_transitioning(&self) -> bool {
        self.chain.is_transitioning()
    }

    /// Set the stability frames threshold.
    pub fn set_stability_frames(&mut self, frames: u32) {
        self.stability_frames = frames.max(1);
    }

    /// Reset to preferred quality.
    pub fn reset(&mut self) {
        self.chain.reset();
        self.config = ShadowConfig::from_quality_level(
            self.chain.preferred_technique().quality_level(),
        );
        self.frames_above_threshold = 0;
        self.frames_below_threshold = 0;
    }
}

impl Default for ShadowQualityController {
    fn default() -> Self {
        Self::new(RTCapability::default(), 60.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ShadowTechnique Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shadow_technique_default() {
        assert_eq!(ShadowTechnique::default(), ShadowTechnique::CascadedShadowMaps);
    }

    #[test]
    fn test_shadow_technique_ray_traced() {
        let tech = ShadowTechnique::RayTraced;
        assert!(tech.is_ray_traced());
        assert!(!tech.is_rasterized());
        assert!(tech.is_enabled());
        assert!(tech.requires_rt());
        assert_eq!(tech.quality_level(), 3);
    }

    #[test]
    fn test_shadow_technique_csm() {
        let tech = ShadowTechnique::CascadedShadowMaps;
        assert!(!tech.is_ray_traced());
        assert!(tech.is_rasterized());
        assert!(tech.is_enabled());
        assert!(!tech.requires_rt());
        assert_eq!(tech.quality_level(), 2);
    }

    #[test]
    fn test_shadow_technique_contact() {
        let tech = ShadowTechnique::ContactShadows;
        assert!(!tech.is_ray_traced());
        assert!(tech.is_rasterized());
        assert!(tech.is_enabled());
        assert!(!tech.requires_rt());
        assert_eq!(tech.quality_level(), 1);
    }

    #[test]
    fn test_shadow_technique_none() {
        let tech = ShadowTechnique::None;
        assert!(!tech.is_ray_traced());
        assert!(!tech.is_rasterized());
        assert!(!tech.is_enabled());
        assert!(!tech.requires_rt());
        assert_eq!(tech.quality_level(), 0);
    }

    #[test]
    fn test_shadow_technique_downgrade_chain() {
        assert_eq!(
            ShadowTechnique::RayTraced.downgrade(),
            ShadowTechnique::CascadedShadowMaps
        );
        assert_eq!(
            ShadowTechnique::CascadedShadowMaps.downgrade(),
            ShadowTechnique::ContactShadows
        );
        assert_eq!(
            ShadowTechnique::ContactShadows.downgrade(),
            ShadowTechnique::None
        );
        assert_eq!(ShadowTechnique::None.downgrade(), ShadowTechnique::None);
    }

    #[test]
    fn test_shadow_technique_upgrade_chain() {
        assert_eq!(ShadowTechnique::None.upgrade(), ShadowTechnique::ContactShadows);
        assert_eq!(
            ShadowTechnique::ContactShadows.upgrade(),
            ShadowTechnique::CascadedShadowMaps
        );
        assert_eq!(
            ShadowTechnique::CascadedShadowMaps.upgrade(),
            ShadowTechnique::RayTraced
        );
        assert_eq!(ShadowTechnique::RayTraced.upgrade(), ShadowTechnique::RayTraced);
    }

    #[test]
    fn test_shadow_technique_description() {
        assert!(!ShadowTechnique::RayTraced.description().is_empty());
        assert!(!ShadowTechnique::CascadedShadowMaps.description().is_empty());
        assert!(!ShadowTechnique::ContactShadows.description().is_empty());
        assert!(!ShadowTechnique::None.description().is_empty());
    }

    #[test]
    fn test_shadow_technique_display() {
        let rt = format!("{}", ShadowTechnique::RayTraced);
        assert!(rt.contains("Ray"));
    }

    #[test]
    fn test_shadow_technique_ordering() {
        assert!(ShadowTechnique::None < ShadowTechnique::ContactShadows);
        assert!(ShadowTechnique::ContactShadows < ShadowTechnique::CascadedShadowMaps);
        assert!(ShadowTechnique::CascadedShadowMaps < ShadowTechnique::RayTraced);
    }

    // -------------------------------------------------------------------------
    // TransitionState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transition_state_new() {
        let state = TransitionState::new(
            ShadowTechnique::RayTraced,
            ShadowTechnique::CascadedShadowMaps,
            0.5,
        );
        assert_eq!(state.from, ShadowTechnique::RayTraced);
        assert_eq!(state.to, ShadowTechnique::CascadedShadowMaps);
        assert_eq!(state.progress, 0.0);
        assert_eq!(state.duration, 0.5);
        assert!(!state.is_complete());
        assert!(state.is_transitioning());
    }

    #[test]
    fn test_transition_state_instant() {
        let state = TransitionState::instant(ShadowTechnique::CascadedShadowMaps);
        assert!(state.is_complete());
        assert!(!state.is_transitioning());
        assert_eq!(state.blend_factor(), 1.0);
    }

    #[test]
    fn test_transition_state_update() {
        let mut state = TransitionState::new(
            ShadowTechnique::RayTraced,
            ShadowTechnique::CascadedShadowMaps,
            1.0,
        );

        // Update halfway
        state.update(0.5);
        assert!(!state.is_complete());
        assert!((state.progress - 0.5).abs() < 0.001);

        // Update to completion
        let completed = state.update(0.6);
        assert!(completed);
        assert!(state.is_complete());
        assert_eq!(state.progress, 1.0);
    }

    #[test]
    fn test_transition_state_blend_factor() {
        let mut state = TransitionState::new(
            ShadowTechnique::RayTraced,
            ShadowTechnique::CascadedShadowMaps,
            1.0,
        );

        // At start, blend is 0
        assert_eq!(state.blend_factor(), 0.0);

        // At middle, blend is smoothed 0.5
        state.progress = 0.5;
        let blend = state.blend_factor();
        assert!((blend - 0.5).abs() < 0.001);

        // At end, blend is 1
        state.progress = 1.0;
        assert_eq!(state.blend_factor(), 1.0);
    }

    #[test]
    fn test_transition_state_current_technique() {
        let mut state = TransitionState::new(
            ShadowTechnique::RayTraced,
            ShadowTechnique::CascadedShadowMaps,
            1.0,
        );

        // Before 0.5, use from
        state.progress = 0.4;
        assert_eq!(state.current_technique(), ShadowTechnique::RayTraced);

        // At/after 0.5, use to
        state.progress = 0.5;
        assert_eq!(state.current_technique(), ShadowTechnique::CascadedShadowMaps);
    }

    #[test]
    fn test_transition_state_default() {
        let state = TransitionState::default();
        assert!(state.is_complete());
        assert_eq!(state.current_technique(), ShadowTechnique::CascadedShadowMaps);
    }

    // -------------------------------------------------------------------------
    // ShadowFallbackChain Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fallback_chain_new_full_rt() {
        let chain = ShadowFallbackChain::new(RTCapability::Full);
        assert_eq!(chain.preferred_technique(), ShadowTechnique::RayTraced);
        assert_eq!(chain.current_technique(), ShadowTechnique::RayTraced);
        assert!(chain.supports_rt());
        assert!(chain.is_using_rt());
        assert!(chain.is_at_preferred());
    }

    #[test]
    fn test_fallback_chain_new_ray_query() {
        let chain = ShadowFallbackChain::new(RTCapability::RayQueryOnly);
        assert_eq!(chain.preferred_technique(), ShadowTechnique::RayTraced);
        assert!(chain.supports_rt());
    }

    #[test]
    fn test_fallback_chain_new_no_rt() {
        let chain = ShadowFallbackChain::new(RTCapability::None);
        assert_eq!(chain.preferred_technique(), ShadowTechnique::CascadedShadowMaps);
        assert_eq!(chain.current_technique(), ShadowTechnique::CascadedShadowMaps);
        assert!(!chain.supports_rt());
        assert!(!chain.is_using_rt());
    }

    #[test]
    fn test_fallback_chain_downgrade() {
        let mut chain = ShadowFallbackChain::new(RTCapability::Full);

        assert!(chain.downgrade());
        assert_eq!(chain.current_technique(), ShadowTechnique::CascadedShadowMaps);

        assert!(chain.downgrade());
        assert_eq!(chain.current_technique(), ShadowTechnique::ContactShadows);

        assert!(chain.downgrade());
        assert_eq!(chain.current_technique(), ShadowTechnique::None);

        // Can't go lower
        assert!(!chain.downgrade());
        assert_eq!(chain.current_technique(), ShadowTechnique::None);
        assert!(chain.is_at_lowest());
    }

    #[test]
    fn test_fallback_chain_upgrade() {
        let mut chain = ShadowFallbackChain::new(RTCapability::Full);

        // Downgrade to lowest
        while chain.downgrade() {}

        assert!(chain.upgrade());
        assert_eq!(chain.current_technique(), ShadowTechnique::ContactShadows);

        assert!(chain.upgrade());
        assert_eq!(chain.current_technique(), ShadowTechnique::CascadedShadowMaps);

        assert!(chain.upgrade());
        assert_eq!(chain.current_technique(), ShadowTechnique::RayTraced);

        // Can't go higher
        assert!(!chain.upgrade());
        assert!(chain.is_at_preferred());
    }

    #[test]
    fn test_fallback_chain_select_technique() {
        let mut chain = ShadowFallbackChain::new(RTCapability::Full);
        assert_eq!(chain.select_technique(), ShadowTechnique::RayTraced);
    }

    #[test]
    fn test_fallback_chain_transition() {
        let mut chain = ShadowFallbackChain::new(RTCapability::Full);
        chain.set_transition_duration(1.0);

        chain.downgrade();
        assert!(chain.is_transitioning());

        // Update transition
        chain.update(0.6);
        assert!(chain.is_transitioning());

        chain.update(0.5);
        assert!(!chain.is_transitioning());
    }

    #[test]
    fn test_fallback_chain_blend_factor() {
        let mut chain = ShadowFallbackChain::new(RTCapability::Full);
        chain.set_transition_duration(1.0);

        chain.downgrade();
        assert_eq!(chain.blend_factor(), 0.0);

        chain.update(0.5);
        assert!(chain.blend_factor() > 0.0 && chain.blend_factor() < 1.0);

        chain.update(0.6);
        assert!((chain.blend_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fallback_chain_reset() {
        let mut chain = ShadowFallbackChain::new(RTCapability::Full);

        chain.downgrade();
        chain.downgrade();

        chain.reset();
        assert_eq!(chain.current_technique(), ShadowTechnique::RayTraced);
        assert!(chain.is_at_preferred());
    }

    #[test]
    fn test_fallback_chain_force_technique() {
        let mut chain = ShadowFallbackChain::new(RTCapability::Full);

        assert!(chain.force_technique(ShadowTechnique::ContactShadows));
        assert_eq!(chain.current_technique(), ShadowTechnique::ContactShadows);
    }

    #[test]
    fn test_fallback_chain_force_technique_no_rt() {
        let mut chain = ShadowFallbackChain::new(RTCapability::None);

        // Can't force RT when not available
        assert!(!chain.force_technique(ShadowTechnique::RayTraced));
        assert_eq!(chain.current_technique(), ShadowTechnique::CascadedShadowMaps);
    }

    #[test]
    fn test_fallback_chain_chain_length() {
        let chain_rt = ShadowFallbackChain::new(RTCapability::Full);
        assert_eq!(chain_rt.chain_length(), 4); // RT, CSM, Contact, None

        let chain_no_rt = ShadowFallbackChain::new(RTCapability::None);
        assert_eq!(chain_no_rt.chain_length(), 3); // CSM, Contact, None
    }

    #[test]
    fn test_fallback_chain_default() {
        let chain = ShadowFallbackChain::default();
        assert_eq!(chain.current_technique(), ShadowTechnique::CascadedShadowMaps);
    }

    // -------------------------------------------------------------------------
    // ShadowConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shadow_config_low() {
        let config = ShadowConfig::low();
        assert_eq!(config.csm_cascade_count, 2);
        assert_eq!(config.csm_resolution, 1024);
        assert_eq!(config.pcss_sample_count, 0);
        assert_eq!(config.contact_shadow_ray_count, 4);
        assert!(!config.has_pcss());
    }

    #[test]
    fn test_shadow_config_medium() {
        let config = ShadowConfig::medium();
        assert_eq!(config.csm_cascade_count, 3);
        assert_eq!(config.csm_resolution, 2048);
        assert_eq!(config.pcss_sample_count, 8);
        assert_eq!(config.contact_shadow_ray_count, 8);
        assert!(config.has_pcss());
    }

    #[test]
    fn test_shadow_config_high() {
        let config = ShadowConfig::high();
        assert_eq!(config.csm_cascade_count, 4);
        assert_eq!(config.csm_resolution, 2048);
        assert_eq!(config.pcss_sample_count, 16);
        assert_eq!(config.contact_shadow_ray_count, 12);
    }

    #[test]
    fn test_shadow_config_ultra() {
        let config = ShadowConfig::ultra();
        assert_eq!(config.csm_cascade_count, 4);
        assert_eq!(config.csm_resolution, 4096);
        assert_eq!(config.pcss_sample_count, 32);
        assert_eq!(config.contact_shadow_ray_count, 16);
    }

    #[test]
    fn test_shadow_config_new_clamping() {
        // Test values outside valid ranges are clamped
        let config = ShadowConfig::new(0, 256, 100, 2, -1.0);
        assert_eq!(config.csm_cascade_count, MIN_CSM_CASCADES);
        assert_eq!(config.csm_resolution, MIN_SHADOW_RESOLUTION);
        assert_eq!(config.pcss_sample_count, MAX_PCSS_SAMPLES);
        assert_eq!(config.contact_shadow_ray_count, MIN_CONTACT_RAYS);
        assert_eq!(config.contact_shadow_max_distance, 0.0);
    }

    #[test]
    fn test_shadow_config_resolution_power_of_two() {
        let config = ShadowConfig::new(3, 1500, 8, 8, 0.5);
        assert_eq!(config.csm_resolution, 2048); // Next power of 2
    }

    #[test]
    fn test_shadow_config_estimated_memory() {
        let config = ShadowConfig::low();
        // 2 cascades * 1024^2 * 4 bytes = 8MB
        assert_eq!(config.estimated_memory_usage(), 2 * 1024 * 1024 * 4);
    }

    #[test]
    fn test_shadow_config_total_samples() {
        let config = ShadowConfig::high();
        // 1 + 16 PCSS + 12 contact = 29
        assert_eq!(config.total_samples_per_pixel(), 29);
    }

    #[test]
    fn test_shadow_config_lerp() {
        let low = ShadowConfig::low();
        let high = ShadowConfig::high();

        let mid = ShadowConfig::lerp(&low, &high, 0.5);
        assert_eq!(mid.csm_cascade_count, 3); // (2 + 4) / 2 rounded
        assert_eq!(mid.pcss_sample_count, 8); // (0 + 16) / 2

        // Edges
        let start = ShadowConfig::lerp(&low, &high, 0.0);
        assert_eq!(start.csm_cascade_count, low.csm_cascade_count);

        let end = ShadowConfig::lerp(&low, &high, 1.0);
        assert_eq!(end.csm_cascade_count, high.csm_cascade_count);
    }

    #[test]
    fn test_shadow_config_from_quality_level() {
        assert_eq!(ShadowConfig::from_quality_level(0), ShadowConfig::low());
        assert_eq!(ShadowConfig::from_quality_level(1), ShadowConfig::medium());
        assert_eq!(ShadowConfig::from_quality_level(2), ShadowConfig::high());
        assert_eq!(ShadowConfig::from_quality_level(3), ShadowConfig::ultra());
        assert_eq!(ShadowConfig::from_quality_level(99), ShadowConfig::ultra());
    }

    #[test]
    fn test_shadow_config_default() {
        assert_eq!(ShadowConfig::default(), ShadowConfig::medium());
    }

    // -------------------------------------------------------------------------
    // ShadowQualityController Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_quality_controller_new() {
        let controller = ShadowQualityController::new(RTCapability::Full, 60.0);
        assert_eq!(controller.current_technique(), ShadowTechnique::RayTraced);
        assert_eq!(*controller.current_config(), ShadowConfig::ultra());
    }

    #[test]
    fn test_quality_controller_no_rt() {
        let controller = ShadowQualityController::new(RTCapability::None, 60.0);
        assert_eq!(controller.current_technique(), ShadowTechnique::CascadedShadowMaps);
        assert_eq!(*controller.current_config(), ShadowConfig::high());
    }

    #[test]
    fn test_quality_controller_downgrade() {
        let mut controller = ShadowQualityController::new(RTCapability::Full, 60.0);

        assert!(controller.downgrade());
        assert_eq!(controller.current_technique(), ShadowTechnique::CascadedShadowMaps);
    }

    #[test]
    fn test_quality_controller_upgrade() {
        let mut controller = ShadowQualityController::new(RTCapability::Full, 60.0);

        controller.downgrade();
        assert!(controller.upgrade());
        assert_eq!(controller.current_technique(), ShadowTechnique::RayTraced);
    }

    #[test]
    fn test_quality_controller_reset() {
        let mut controller = ShadowQualityController::new(RTCapability::Full, 60.0);

        controller.downgrade();
        controller.downgrade();

        controller.reset();
        assert_eq!(controller.current_technique(), ShadowTechnique::RayTraced);
    }

    #[test]
    fn test_quality_controller_default() {
        let controller = ShadowQualityController::default();
        assert_eq!(controller.current_technique(), ShadowTechnique::CascadedShadowMaps);
    }

    #[test]
    fn test_quality_controller_chain_access() {
        let mut controller = ShadowQualityController::new(RTCapability::Full, 60.0);
        assert!(controller.chain().supports_rt());
        assert!(controller.chain_mut().downgrade());
    }

    #[test]
    fn test_quality_controller_transition() {
        let mut controller = ShadowQualityController::new(RTCapability::Full, 60.0);
        controller.chain_mut().set_transition_duration(1.0);

        controller.downgrade();
        assert!(controller.is_transitioning());

        controller.update(0.0, 1.1);
        assert!(!controller.is_transitioning());
    }

    #[test]
    fn test_quality_controller_set_stability_frames() {
        let mut controller = ShadowQualityController::new(RTCapability::Full, 60.0);
        controller.set_stability_frames(5);
        // Internal state, just verify no panic
    }
}
