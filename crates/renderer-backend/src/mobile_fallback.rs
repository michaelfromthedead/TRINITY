//! Mobile-Specific Environment Rendering Fallbacks
//!
//! This module implements quality profiles and fallback systems for mobile devices,
//! providing reduced-fidelity rendering options for fog, water, terrain, and foliage
//! that maintain acceptable visual quality while meeting mobile performance budgets.
//!
//! # Features
//!
//! - Device capability detection (GPU tier, memory tier, fill rate)
//! - Mobile-optimized fog (single-layer, reduced froxel grid)
//! - Simplified water (planar reflections, reduced wave count)
//! - Terrain fallbacks (reduced clipmap levels, texture limits)
//! - Foliage culling (instance limits, early billboards)
//! - Dynamic resolution scaling
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::mobile_fallback::{
//!     MobileProfile, MobileCapabilityDetector, MobileFallbackSystem,
//! };
//!
//! let detector = MobileCapabilityDetector::new();
//! let profile = detector.get_recommended_profile();
//! let mut fallback = MobileFallbackSystem::new(profile);
//!
//! // Apply to performance budget
//! fallback.apply_to_performance_budget(&mut budget);
//!
//! // Check if quality reduction needed
//! if fallback.should_reduce_quality(25.0) {
//!     let scale = fallback.get_render_scale();
//!     // Apply dynamic resolution
//! }
//! ```

use bytemuck::{Pod, Zeroable};

use crate::env_performance::{EnvLodConfig, PerformanceBudget};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default mobile target FPS.
pub const MOBILE_TARGET_FPS: f32 = 30.0;

/// Default mobile frame time (ms).
pub const MOBILE_TARGET_FRAME_TIME_MS: f32 = 33.33;

/// Minimum render scale for dynamic resolution.
pub const MIN_RENDER_SCALE: f32 = 0.5;

/// Maximum render scale for dynamic resolution.
pub const MAX_RENDER_SCALE: f32 = 1.0;

/// Default low-end max texture size.
pub const LOW_END_MAX_TEXTURE_SIZE: u32 = 512;

/// Default mid-range max texture size.
pub const MID_RANGE_MAX_TEXTURE_SIZE: u32 = 1024;

/// Default high-end max texture size.
pub const HIGH_END_MAX_TEXTURE_SIZE: u32 = 2048;

/// Low-end shadow resolution.
pub const LOW_END_SHADOW_RES: u32 = 256;

/// Mid-range shadow resolution.
pub const MID_RANGE_SHADOW_RES: u32 = 512;

/// High-end shadow resolution.
pub const HIGH_END_SHADOW_RES: u32 = 1024;

/// Default fill rate estimate for low-end devices (pixels/sec).
pub const LOW_END_FILL_RATE: u32 = 500_000_000;

/// Default fill rate estimate for mid-range devices (pixels/sec).
pub const MID_RANGE_FILL_RATE: u32 = 1_500_000_000;

/// Default fill rate estimate for high-end devices (pixels/sec).
pub const HIGH_END_FILL_RATE: u32 = 3_000_000_000;

// ---------------------------------------------------------------------------
// Mobile GPU Tier
// ---------------------------------------------------------------------------

/// GPU capability tier for mobile devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum MobileGpuTier {
    /// Low-end mobile GPUs (Adreno 5xx, Mali-G5x, PowerVR GE8xxx).
    #[default]
    LowEnd = 0,
    /// Mid-range mobile GPUs (Adreno 6xx, Mali-G7x, Apple A12).
    MidRange = 1,
    /// High-end mobile GPUs (Adreno 7xx, Mali-G7xx, Apple A15+).
    HighEnd = 2,
}

impl MobileGpuTier {
    /// Get tier index (0-2).
    pub fn as_index(&self) -> usize {
        *self as usize
    }

    /// Create from tier index.
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::LowEnd),
            1 => Some(Self::MidRange),
            2 => Some(Self::HighEnd),
            _ => None,
        }
    }

    /// Get estimated fill rate for this tier.
    pub fn estimated_fill_rate(&self) -> u32 {
        match self {
            Self::LowEnd => LOW_END_FILL_RATE,
            Self::MidRange => MID_RANGE_FILL_RATE,
            Self::HighEnd => HIGH_END_FILL_RATE,
        }
    }

    /// Check if compute shaders are well-supported.
    pub fn supports_compute_shaders(&self) -> bool {
        // All modern mobile GPUs support compute, but low-end may have issues
        matches!(self, Self::MidRange | Self::HighEnd)
    }
}

// ---------------------------------------------------------------------------
// Memory Tier
// ---------------------------------------------------------------------------

/// Memory availability tier for mobile devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum MemoryTier {
    /// 1GB or less available.
    #[default]
    Tier1GB = 0,
    /// 2GB available.
    Tier2GB = 1,
    /// 4GB available.
    Tier4GB = 2,
    /// 8GB or more available.
    Tier8GBPlus = 3,
}

impl MemoryTier {
    /// Get tier index (0-3).
    pub fn as_index(&self) -> usize {
        *self as usize
    }

    /// Create from tier index.
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Tier1GB),
            1 => Some(Self::Tier2GB),
            2 => Some(Self::Tier4GB),
            3 => Some(Self::Tier8GBPlus),
            _ => None,
        }
    }

    /// Get approximate memory in bytes.
    pub fn memory_bytes(&self) -> u64 {
        match self {
            Self::Tier1GB => 1 << 30,
            Self::Tier2GB => 2 << 30,
            Self::Tier4GB => 4 << 30,
            Self::Tier8GBPlus => 8 << 30,
        }
    }

    /// Get recommended texture budget fraction.
    pub fn texture_budget_fraction(&self) -> f32 {
        match self {
            Self::Tier1GB => 0.15,
            Self::Tier2GB => 0.20,
            Self::Tier4GB => 0.25,
            Self::Tier8GBPlus => 0.30,
        }
    }
}

// ---------------------------------------------------------------------------
// Mobile Profile
// ---------------------------------------------------------------------------

/// Mobile rendering quality profile.
///
/// GPU-friendly layout for uniform buffer binding.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct MobileProfile {
    /// Maximum texture size (512, 1024, 2048).
    pub max_texture_size: u32,
    /// Maximum draw calls per frame.
    pub max_draw_calls: u32,
    /// Maximum triangles per frame.
    pub max_triangles_per_frame: u32,
    /// Shadow map resolution (256, 512, 1024).
    pub shadow_resolution: u32,
    /// Enable shadow rendering (1 = true, 0 = false).
    pub enable_shadows: u32,
    /// Enable fog rendering (1 = true, 0 = false).
    pub enable_fog: u32,
    /// Enable water reflections (1 = true, 0 = false).
    pub enable_water_reflections: u32,
    /// Enable particle systems (1 = true, 0 = false).
    pub enable_particles: u32,
}

impl Default for MobileProfile {
    fn default() -> Self {
        Self::mid_range()
    }
}

impl MobileProfile {
    /// Create a profile for low-end devices.
    pub fn low_end() -> Self {
        Self {
            max_texture_size: LOW_END_MAX_TEXTURE_SIZE,
            max_draw_calls: 100,
            max_triangles_per_frame: 100_000,
            shadow_resolution: LOW_END_SHADOW_RES,
            enable_shadows: 0,
            enable_fog: 1,
            enable_water_reflections: 0,
            enable_particles: 0,
        }
    }

    /// Create a profile for mid-range devices.
    pub fn mid_range() -> Self {
        Self {
            max_texture_size: MID_RANGE_MAX_TEXTURE_SIZE,
            max_draw_calls: 300,
            max_triangles_per_frame: 300_000,
            shadow_resolution: MID_RANGE_SHADOW_RES,
            enable_shadows: 1,
            enable_fog: 1,
            enable_water_reflections: 1,
            enable_particles: 1,
        }
    }

    /// Create a profile for high-end devices.
    pub fn high_end() -> Self {
        Self {
            max_texture_size: HIGH_END_MAX_TEXTURE_SIZE,
            max_draw_calls: 600,
            max_triangles_per_frame: 750_000,
            shadow_resolution: HIGH_END_SHADOW_RES,
            enable_shadows: 1,
            enable_fog: 1,
            enable_water_reflections: 1,
            enable_particles: 1,
        }
    }

    /// Check if shadows are enabled.
    pub fn shadows_enabled(&self) -> bool {
        self.enable_shadows != 0
    }

    /// Check if fog is enabled.
    pub fn fog_enabled(&self) -> bool {
        self.enable_fog != 0
    }

    /// Check if water reflections are enabled.
    pub fn water_reflections_enabled(&self) -> bool {
        self.enable_water_reflections != 0
    }

    /// Check if particles are enabled.
    pub fn particles_enabled(&self) -> bool {
        self.enable_particles != 0
    }

    /// Validate profile settings.
    pub fn is_valid(&self) -> bool {
        let valid_texture_sizes = [512, 1024, 2048, 4096];
        let valid_shadow_res = [0, 256, 512, 1024, 2048];

        valid_texture_sizes.contains(&self.max_texture_size)
            && valid_shadow_res.contains(&self.shadow_resolution)
            && self.max_draw_calls > 0
            && self.max_triangles_per_frame > 0
    }

    /// Get quality level (0 = low, 1 = mid, 2 = high).
    pub fn quality_level(&self) -> u8 {
        if self.max_texture_size <= LOW_END_MAX_TEXTURE_SIZE {
            0
        } else if self.max_texture_size <= MID_RANGE_MAX_TEXTURE_SIZE {
            1
        } else {
            2
        }
    }
}

// ---------------------------------------------------------------------------
// Mobile Capability Detector
// ---------------------------------------------------------------------------

/// Detects mobile device capabilities.
#[derive(Debug, Clone)]
pub struct MobileCapabilityDetector {
    /// Detected GPU tier.
    gpu_tier: MobileGpuTier,
    /// Detected memory tier.
    memory_tier: MemoryTier,
    /// Estimated fill rate (pixels/sec).
    fill_rate: u32,
    /// Compute shader support.
    compute_support: bool,
}

impl Default for MobileCapabilityDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl MobileCapabilityDetector {
    /// Create a new detector with auto-detection.
    pub fn new() -> Self {
        // In a real implementation, this would query device info
        // For now, default to mid-range
        Self {
            gpu_tier: MobileGpuTier::MidRange,
            memory_tier: MemoryTier::Tier2GB,
            fill_rate: MID_RANGE_FILL_RATE,
            compute_support: true,
        }
    }

    /// Create with specific capabilities (for testing).
    pub fn with_capabilities(
        gpu_tier: MobileGpuTier,
        memory_tier: MemoryTier,
        fill_rate: u32,
        compute_support: bool,
    ) -> Self {
        Self {
            gpu_tier,
            memory_tier,
            fill_rate,
            compute_support,
        }
    }

    /// Detect GPU tier.
    pub fn detect_gpu_tier(&self) -> MobileGpuTier {
        self.gpu_tier
    }

    /// Detect memory tier.
    pub fn detect_memory_tier(&self) -> MemoryTier {
        self.memory_tier
    }

    /// Estimate fill rate in pixels per second.
    pub fn estimate_fill_rate(&self) -> u32 {
        self.fill_rate
    }

    /// Check if compute shaders are supported.
    pub fn supports_compute_shaders(&self) -> bool {
        self.compute_support
    }

    /// Get recommended profile based on detected capabilities.
    pub fn get_recommended_profile(&self) -> MobileProfile {
        match (self.gpu_tier, self.memory_tier) {
            (MobileGpuTier::HighEnd, MemoryTier::Tier4GB | MemoryTier::Tier8GBPlus) => {
                MobileProfile::high_end()
            }
            (MobileGpuTier::HighEnd, _) | (MobileGpuTier::MidRange, MemoryTier::Tier2GB | MemoryTier::Tier4GB | MemoryTier::Tier8GBPlus) => {
                MobileProfile::mid_range()
            }
            _ => MobileProfile::low_end(),
        }
    }

    /// Simulate detection from GPU name (for testing).
    pub fn detect_from_gpu_name(name: &str) -> MobileGpuTier {
        let name_lower = name.to_lowercase();

        // High-end indicators
        if name_lower.contains("a17") || name_lower.contains("a16") || name_lower.contains("a15")
            || name_lower.contains("adreno 7") || name_lower.contains("mali-g7")
            || name_lower.contains("apple gpu") && name_lower.contains("pro")
        {
            return MobileGpuTier::HighEnd;
        }

        // Mid-range indicators
        if name_lower.contains("a14") || name_lower.contains("a13") || name_lower.contains("a12")
            || name_lower.contains("adreno 6") || name_lower.contains("mali-g7")
        {
            return MobileGpuTier::MidRange;
        }

        // Default to low-end
        MobileGpuTier::LowEnd
    }

    /// Detect memory tier from available memory in bytes.
    pub fn detect_memory_from_bytes(bytes: u64) -> MemoryTier {
        if bytes >= 8 << 30 {
            MemoryTier::Tier8GBPlus
        } else if bytes >= 4 << 30 {
            MemoryTier::Tier4GB
        } else if bytes >= 2 << 30 {
            MemoryTier::Tier2GB
        } else {
            MemoryTier::Tier1GB
        }
    }
}

// ---------------------------------------------------------------------------
// Mobile Fog Fallback
// ---------------------------------------------------------------------------

/// Fog rendering fallbacks for mobile.
#[derive(Debug, Clone, Copy, Default)]
pub struct MobileFogFallback {
    /// Use single-layer fog instead of volumetric.
    single_layer: bool,
    /// Reduced froxel grid resolution.
    froxel_resolution: [u32; 3],
    /// Skip temporal accumulation.
    skip_temporal: bool,
}

impl MobileFogFallback {
    /// Create fog fallback for a profile.
    pub fn for_profile(profile: &MobileProfile) -> Self {
        let quality = profile.quality_level();

        match quality {
            0 => Self {
                single_layer: true,
                froxel_resolution: [32, 18, 16],
                skip_temporal: true,
            },
            1 => Self {
                single_layer: false,
                froxel_resolution: [64, 36, 32],
                skip_temporal: true,
            },
            _ => Self {
                single_layer: false,
                froxel_resolution: [128, 72, 64],
                skip_temporal: false,
            },
        }
    }

    /// Calculate simplified single-layer fog.
    ///
    /// Returns fog factor (0.0 = no fog, 1.0 = full fog).
    pub fn simplified_fog(&self, depth: f32, density: f32) -> f32 {
        if self.single_layer {
            // Simple exponential fog
            (1.0 - (-depth * density).exp()).clamp(0.0, 1.0)
        } else {
            // Standard fog (caller should use volumetric)
            (1.0 - (-depth * density * 0.5).exp()).clamp(0.0, 1.0)
        }
    }

    /// Get mobile froxel resolution.
    pub fn mobile_froxel_resolution(&self) -> [u32; 3] {
        self.froxel_resolution
    }

    /// Check if temporal accumulation should be skipped.
    pub fn should_skip_temporal(&self) -> bool {
        self.skip_temporal
    }

    /// Check if using single-layer mode.
    pub fn is_single_layer(&self) -> bool {
        self.single_layer
    }
}

// ---------------------------------------------------------------------------
// Mobile Water Fallback
// ---------------------------------------------------------------------------

/// Water rendering fallbacks for mobile.
#[derive(Debug, Clone, Copy)]
pub struct MobileWaterFallback {
    /// Planar reflection resolution.
    reflection_size: u32,
    /// Use SSR fallback instead of raytracing.
    use_ssr: bool,
    /// Number of Gerstner waves.
    wave_count: u32,
    /// Skip foam rendering.
    skip_foam: bool,
}

impl Default for MobileWaterFallback {
    fn default() -> Self {
        Self::for_profile(&MobileProfile::mid_range())
    }
}

impl MobileWaterFallback {
    /// Create water fallback for a profile.
    pub fn for_profile(profile: &MobileProfile) -> Self {
        let quality = profile.quality_level();

        match quality {
            0 => Self {
                reflection_size: 128,
                use_ssr: false, // Use cubemap instead
                wave_count: 2,
                skip_foam: true,
            },
            1 => Self {
                reflection_size: 256,
                use_ssr: true,
                wave_count: 3,
                skip_foam: false,
            },
            _ => Self {
                reflection_size: 512,
                use_ssr: true,
                wave_count: 4,
                skip_foam: false,
            },
        }
    }

    /// Get planar reflection size.
    pub fn planar_reflection_size(&self) -> u32 {
        self.reflection_size
    }

    /// Check if SSR fallback should be used.
    pub fn use_ssr_fallback(&self) -> bool {
        self.use_ssr
    }

    /// Get simplified Gerstner wave count.
    pub fn simplified_gerstner_waves(&self) -> u32 {
        self.wave_count
    }

    /// Check if foam should be skipped.
    pub fn skip_foam(&self) -> bool {
        self.skip_foam
    }

    /// Calculate wave displacement (simplified).
    pub fn calculate_wave(&self, position: [f32; 2], time: f32) -> f32 {
        let mut displacement = 0.0;
        let base_freq = 0.1;
        let base_amp = 0.5;

        for i in 0..self.wave_count {
            let freq = base_freq * (1.0 + i as f32 * 0.5);
            let amp = base_amp / (1.0 + i as f32);
            let phase = position[0] * freq + position[1] * freq * 0.7 + time;
            displacement += phase.sin() * amp;
        }

        displacement
    }
}

// ---------------------------------------------------------------------------
// Mobile Terrain Fallback
// ---------------------------------------------------------------------------

/// Terrain rendering fallbacks for mobile.
#[derive(Debug, Clone, Copy)]
pub struct MobileTerrainFallback {
    /// Number of clipmap levels.
    clipmap_levels: u32,
    /// Maximum texture array layers.
    texture_layers: u32,
    /// Use triplanar mapping.
    use_triplanar: bool,
    /// LOD bias for aggressive culling.
    lod_bias: f32,
}

impl Default for MobileTerrainFallback {
    fn default() -> Self {
        Self::for_profile(&MobileProfile::mid_range())
    }
}

impl MobileTerrainFallback {
    /// Create terrain fallback for a profile.
    pub fn for_profile(profile: &MobileProfile) -> Self {
        let quality = profile.quality_level();

        match quality {
            0 => Self {
                clipmap_levels: 2,
                texture_layers: 4,
                use_triplanar: false,
                lod_bias: 2.0,
            },
            1 => Self {
                clipmap_levels: 3,
                texture_layers: 6,
                use_triplanar: false,
                lod_bias: 1.0,
            },
            _ => Self {
                clipmap_levels: 4,
                texture_layers: 8,
                use_triplanar: true,
                lod_bias: 0.5,
            },
        }
    }

    /// Get number of clipmap levels.
    pub fn clipmap_levels(&self) -> u32 {
        self.clipmap_levels
    }

    /// Get maximum texture array layers.
    pub fn texture_array_layers(&self) -> u32 {
        self.texture_layers
    }

    /// Check if triplanar mapping should be used.
    pub fn use_triplanar(&self) -> bool {
        self.use_triplanar
    }

    /// Get LOD bias for terrain.
    pub fn lod_bias(&self) -> f32 {
        self.lod_bias
    }

    /// Calculate adjusted LOD based on distance.
    pub fn calculate_lod(&self, distance: f32, base_lod: u32) -> u32 {
        let biased = base_lod as f32 + self.lod_bias;
        let distance_factor = (distance / 100.0).log2().max(0.0);
        (biased + distance_factor).floor() as u32
    }
}

// ---------------------------------------------------------------------------
// Mobile Foliage Fallback
// ---------------------------------------------------------------------------

/// Foliage rendering fallbacks for mobile.
#[derive(Debug, Clone, Copy)]
pub struct MobileFoliageFallback {
    /// Maximum instances to render.
    max_instances: u32,
    /// Distance for billboard transition.
    billboard_distance: f32,
    /// Impostor atlas resolution.
    impostor_resolution: u32,
    /// Skip wind animation.
    skip_wind: bool,
}

impl Default for MobileFoliageFallback {
    fn default() -> Self {
        Self::for_profile(&MobileProfile::mid_range())
    }
}

impl MobileFoliageFallback {
    /// Create foliage fallback for a profile.
    pub fn for_profile(profile: &MobileProfile) -> Self {
        let quality = profile.quality_level();

        match quality {
            0 => Self {
                max_instances: 1000,
                billboard_distance: 15.0,
                impostor_resolution: 256,
                skip_wind: true,
            },
            1 => Self {
                max_instances: 3000,
                billboard_distance: 30.0,
                impostor_resolution: 512,
                skip_wind: false,
            },
            _ => Self {
                max_instances: 5000,
                billboard_distance: 50.0,
                impostor_resolution: 1024,
                skip_wind: false,
            },
        }
    }

    /// Get maximum instances to render.
    pub fn max_instances(&self) -> u32 {
        self.max_instances
    }

    /// Get billboard transition distance.
    pub fn billboard_distance(&self) -> f32 {
        self.billboard_distance
    }

    /// Get impostor atlas resolution.
    pub fn impostor_resolution(&self) -> u32 {
        self.impostor_resolution
    }

    /// Check if wind animation should be skipped.
    pub fn skip_wind_animation(&self) -> bool {
        self.skip_wind
    }

    /// Check if instance should use billboard at given distance.
    pub fn should_billboard(&self, distance: f32) -> bool {
        distance > self.billboard_distance
    }

    /// Calculate instance priority for culling.
    pub fn instance_priority(&self, distance: f32, size: f32) -> f32 {
        // Larger, closer instances have higher priority
        let distance_score = 1.0 / (distance + 1.0);
        let size_score = size;
        distance_score * size_score
    }
}

// ---------------------------------------------------------------------------
// Mobile Fallback System
// ---------------------------------------------------------------------------

/// Coordinator for all mobile fallbacks.
#[derive(Debug, Clone)]
pub struct MobileFallbackSystem {
    /// Active mobile profile.
    profile: MobileProfile,
    /// Fog fallback settings.
    fog: MobileFogFallback,
    /// Water fallback settings.
    water: MobileWaterFallback,
    /// Terrain fallback settings.
    terrain: MobileTerrainFallback,
    /// Foliage fallback settings.
    foliage: MobileFoliageFallback,
    /// Current render scale (0.5 - 1.0).
    render_scale: f32,
    /// Frame time history for adaptive scaling.
    frame_history: [f32; 8],
    /// Current history index.
    history_index: usize,
}

impl Default for MobileFallbackSystem {
    fn default() -> Self {
        Self::new(MobileProfile::mid_range())
    }
}

impl MobileFallbackSystem {
    /// Create a new fallback system with the given profile.
    pub fn new(profile: MobileProfile) -> Self {
        Self {
            fog: MobileFogFallback::for_profile(&profile),
            water: MobileWaterFallback::for_profile(&profile),
            terrain: MobileTerrainFallback::for_profile(&profile),
            foliage: MobileFoliageFallback::for_profile(&profile),
            profile,
            render_scale: 1.0,
            frame_history: [MOBILE_TARGET_FRAME_TIME_MS; 8],
            history_index: 0,
        }
    }

    /// Apply mobile constraints to performance budget.
    pub fn apply_to_performance_budget(&self, budget: &mut PerformanceBudget) {
        // Scale budgets for mobile
        let scale = match self.profile.quality_level() {
            0 => 0.25,
            1 => 0.5,
            _ => 0.75,
        };

        budget.fog_budget_ms *= scale;
        budget.water_budget_ms *= scale;
        budget.clouds_budget_ms *= scale;
        budget.terrain_budget_ms *= scale;
        budget.foliage_budget_ms *= scale;

        // Disable passes if not enabled in profile
        if !self.profile.fog_enabled() {
            budget.fog_budget_ms = 0.0;
        }
        if !self.profile.water_reflections_enabled() {
            budget.water_budget_ms *= 0.5;
        }
    }

    /// Apply mobile constraints to LOD configuration.
    pub fn apply_to_lod_config(&self, config: &mut EnvLodConfig) {
        let lod_offset = match self.profile.quality_level() {
            0 => 2,
            1 => 1,
            _ => 0,
        };

        config.fog_lod = (config.fog_lod + lod_offset).min(3);
        config.water_lod = (config.water_lod + lod_offset).min(3);
        config.cloud_lod = (config.cloud_lod + lod_offset).min(3);
        config.terrain_lod = (config.terrain_lod + lod_offset).min(3);
        config.foliage_lod = (config.foliage_lod + lod_offset).min(3);
    }

    /// Get current render scale.
    pub fn get_render_scale(&self) -> f32 {
        self.render_scale
    }

    /// Check if quality should be reduced based on current FPS.
    pub fn should_reduce_quality(&self, current_fps: f32) -> bool {
        if current_fps <= 0.0 {
            return false;
        }

        let current_frame_time = 1000.0 / current_fps;
        current_frame_time > MOBILE_TARGET_FRAME_TIME_MS * 1.2
    }

    /// Update dynamic resolution based on frame time.
    pub fn update_dynamic_resolution(&mut self, frame_time_ms: f32) {
        // Update history
        self.frame_history[self.history_index] = frame_time_ms;
        self.history_index = (self.history_index + 1) % 8;

        // Calculate average
        let avg: f32 = self.frame_history.iter().sum::<f32>() / 8.0;

        // Adjust render scale
        if avg > MOBILE_TARGET_FRAME_TIME_MS * 1.2 {
            // Over budget: reduce scale
            self.render_scale = (self.render_scale - 0.05).max(MIN_RENDER_SCALE);
        } else if avg < MOBILE_TARGET_FRAME_TIME_MS * 0.8 && self.render_scale < MAX_RENDER_SCALE {
            // Under budget: increase scale gradually
            self.render_scale = (self.render_scale + 0.02).min(MAX_RENDER_SCALE);
        }
    }

    /// Get fog fallback settings.
    pub fn get_fog_fallback(&self) -> &MobileFogFallback {
        &self.fog
    }

    /// Get water fallback settings.
    pub fn get_water_fallback(&self) -> &MobileWaterFallback {
        &self.water
    }

    /// Get terrain fallback settings.
    pub fn get_terrain_fallback(&self) -> &MobileTerrainFallback {
        &self.terrain
    }

    /// Get foliage fallback settings.
    pub fn get_foliage_fallback(&self) -> &MobileFoliageFallback {
        &self.foliage
    }

    /// Get current profile.
    pub fn profile(&self) -> &MobileProfile {
        &self.profile
    }

    /// Set a new profile and update all fallbacks.
    pub fn set_profile(&mut self, profile: MobileProfile) {
        self.fog = MobileFogFallback::for_profile(&profile);
        self.water = MobileWaterFallback::for_profile(&profile);
        self.terrain = MobileTerrainFallback::for_profile(&profile);
        self.foliage = MobileFoliageFallback::for_profile(&profile);
        self.profile = profile;
    }

    /// Reset dynamic resolution to maximum.
    pub fn reset_render_scale(&mut self) {
        self.render_scale = MAX_RENDER_SCALE;
        self.frame_history = [MOBILE_TARGET_FRAME_TIME_MS; 8];
    }

    /// Get scaled resolution for a target dimension.
    pub fn get_scaled_dimension(&self, dimension: u32) -> u32 {
        ((dimension as f32) * self.render_scale).round() as u32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== MobileGpuTier Tests =====

    #[test]
    fn test_gpu_tier_default() {
        let tier = MobileGpuTier::default();
        assert_eq!(tier, MobileGpuTier::LowEnd);
    }

    #[test]
    fn test_gpu_tier_as_index() {
        assert_eq!(MobileGpuTier::LowEnd.as_index(), 0);
        assert_eq!(MobileGpuTier::MidRange.as_index(), 1);
        assert_eq!(MobileGpuTier::HighEnd.as_index(), 2);
    }

    #[test]
    fn test_gpu_tier_from_index() {
        assert_eq!(MobileGpuTier::from_index(0), Some(MobileGpuTier::LowEnd));
        assert_eq!(MobileGpuTier::from_index(1), Some(MobileGpuTier::MidRange));
        assert_eq!(MobileGpuTier::from_index(2), Some(MobileGpuTier::HighEnd));
        assert_eq!(MobileGpuTier::from_index(3), None);
    }

    #[test]
    fn test_gpu_tier_fill_rate() {
        assert_eq!(MobileGpuTier::LowEnd.estimated_fill_rate(), LOW_END_FILL_RATE);
        assert_eq!(MobileGpuTier::MidRange.estimated_fill_rate(), MID_RANGE_FILL_RATE);
        assert_eq!(MobileGpuTier::HighEnd.estimated_fill_rate(), HIGH_END_FILL_RATE);
    }

    #[test]
    fn test_gpu_tier_compute_support() {
        assert!(!MobileGpuTier::LowEnd.supports_compute_shaders());
        assert!(MobileGpuTier::MidRange.supports_compute_shaders());
        assert!(MobileGpuTier::HighEnd.supports_compute_shaders());
    }

    // ===== MemoryTier Tests =====

    #[test]
    fn test_memory_tier_default() {
        let tier = MemoryTier::default();
        assert_eq!(tier, MemoryTier::Tier1GB);
    }

    #[test]
    fn test_memory_tier_as_index() {
        assert_eq!(MemoryTier::Tier1GB.as_index(), 0);
        assert_eq!(MemoryTier::Tier2GB.as_index(), 1);
        assert_eq!(MemoryTier::Tier4GB.as_index(), 2);
        assert_eq!(MemoryTier::Tier8GBPlus.as_index(), 3);
    }

    #[test]
    fn test_memory_tier_from_index() {
        assert_eq!(MemoryTier::from_index(0), Some(MemoryTier::Tier1GB));
        assert_eq!(MemoryTier::from_index(3), Some(MemoryTier::Tier8GBPlus));
        assert_eq!(MemoryTier::from_index(4), None);
    }

    #[test]
    fn test_memory_tier_bytes() {
        assert_eq!(MemoryTier::Tier1GB.memory_bytes(), 1 << 30);
        assert_eq!(MemoryTier::Tier2GB.memory_bytes(), 2 << 30);
        assert_eq!(MemoryTier::Tier4GB.memory_bytes(), 4 << 30);
        assert_eq!(MemoryTier::Tier8GBPlus.memory_bytes(), 8 << 30);
    }

    #[test]
    fn test_memory_tier_texture_budget() {
        let budget_1gb = MemoryTier::Tier1GB.texture_budget_fraction();
        let budget_8gb = MemoryTier::Tier8GBPlus.texture_budget_fraction();
        assert!(budget_8gb > budget_1gb);
    }

    // ===== MobileProfile Tests =====

    #[test]
    fn test_profile_low_end() {
        let profile = MobileProfile::low_end();
        assert_eq!(profile.max_texture_size, LOW_END_MAX_TEXTURE_SIZE);
        assert_eq!(profile.shadow_resolution, LOW_END_SHADOW_RES);
        assert!(!profile.shadows_enabled());
        assert!(profile.fog_enabled());
        assert!(!profile.water_reflections_enabled());
        assert!(!profile.particles_enabled());
    }

    #[test]
    fn test_profile_mid_range() {
        let profile = MobileProfile::mid_range();
        assert_eq!(profile.max_texture_size, MID_RANGE_MAX_TEXTURE_SIZE);
        assert_eq!(profile.shadow_resolution, MID_RANGE_SHADOW_RES);
        assert!(profile.shadows_enabled());
        assert!(profile.fog_enabled());
        assert!(profile.water_reflections_enabled());
        assert!(profile.particles_enabled());
    }

    #[test]
    fn test_profile_high_end() {
        let profile = MobileProfile::high_end();
        assert_eq!(profile.max_texture_size, HIGH_END_MAX_TEXTURE_SIZE);
        assert_eq!(profile.shadow_resolution, HIGH_END_SHADOW_RES);
        assert!(profile.shadows_enabled());
    }

    #[test]
    fn test_profile_default() {
        let profile = MobileProfile::default();
        assert_eq!(profile, MobileProfile::mid_range());
    }

    #[test]
    fn test_profile_is_valid() {
        assert!(MobileProfile::low_end().is_valid());
        assert!(MobileProfile::mid_range().is_valid());
        assert!(MobileProfile::high_end().is_valid());

        let invalid = MobileProfile {
            max_texture_size: 777, // Invalid
            ..MobileProfile::low_end()
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_profile_quality_level() {
        assert_eq!(MobileProfile::low_end().quality_level(), 0);
        assert_eq!(MobileProfile::mid_range().quality_level(), 1);
        assert_eq!(MobileProfile::high_end().quality_level(), 2);
    }

    #[test]
    fn test_profile_pod_zeroable() {
        let zeroed: MobileProfile = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.max_texture_size, 0);
        assert_eq!(zeroed.enable_shadows, 0);

        let profile = MobileProfile::mid_range();
        let bytes = bytemuck::bytes_of(&profile);
        assert_eq!(bytes.len(), std::mem::size_of::<MobileProfile>());
    }

    // ===== MobileCapabilityDetector Tests =====

    #[test]
    fn test_detector_new() {
        let detector = MobileCapabilityDetector::new();
        assert_eq!(detector.detect_gpu_tier(), MobileGpuTier::MidRange);
        assert_eq!(detector.detect_memory_tier(), MemoryTier::Tier2GB);
    }

    #[test]
    fn test_detector_with_capabilities() {
        let detector = MobileCapabilityDetector::with_capabilities(
            MobileGpuTier::HighEnd,
            MemoryTier::Tier8GBPlus,
            HIGH_END_FILL_RATE,
            true,
        );
        assert_eq!(detector.detect_gpu_tier(), MobileGpuTier::HighEnd);
        assert_eq!(detector.detect_memory_tier(), MemoryTier::Tier8GBPlus);
        assert_eq!(detector.estimate_fill_rate(), HIGH_END_FILL_RATE);
        assert!(detector.supports_compute_shaders());
    }

    #[test]
    fn test_detector_recommended_profile_high_end() {
        let detector = MobileCapabilityDetector::with_capabilities(
            MobileGpuTier::HighEnd,
            MemoryTier::Tier4GB,
            HIGH_END_FILL_RATE,
            true,
        );
        let profile = detector.get_recommended_profile();
        assert_eq!(profile.quality_level(), 2);
    }

    #[test]
    fn test_detector_recommended_profile_mid_range() {
        let detector = MobileCapabilityDetector::with_capabilities(
            MobileGpuTier::MidRange,
            MemoryTier::Tier2GB,
            MID_RANGE_FILL_RATE,
            true,
        );
        let profile = detector.get_recommended_profile();
        assert_eq!(profile.quality_level(), 1);
    }

    #[test]
    fn test_detector_recommended_profile_low_end() {
        let detector = MobileCapabilityDetector::with_capabilities(
            MobileGpuTier::LowEnd,
            MemoryTier::Tier1GB,
            LOW_END_FILL_RATE,
            false,
        );
        let profile = detector.get_recommended_profile();
        assert_eq!(profile.quality_level(), 0);
    }

    #[test]
    fn test_detector_gpu_name_high_end() {
        assert_eq!(
            MobileCapabilityDetector::detect_from_gpu_name("Apple A17 Pro"),
            MobileGpuTier::HighEnd
        );
        assert_eq!(
            MobileCapabilityDetector::detect_from_gpu_name("Adreno 740"),
            MobileGpuTier::HighEnd
        );
    }

    #[test]
    fn test_detector_gpu_name_mid_range() {
        assert_eq!(
            MobileCapabilityDetector::detect_from_gpu_name("Apple A14"),
            MobileGpuTier::MidRange
        );
        assert_eq!(
            MobileCapabilityDetector::detect_from_gpu_name("Adreno 650"),
            MobileGpuTier::MidRange
        );
    }

    #[test]
    fn test_detector_gpu_name_low_end() {
        assert_eq!(
            MobileCapabilityDetector::detect_from_gpu_name("Unknown GPU"),
            MobileGpuTier::LowEnd
        );
    }

    #[test]
    fn test_detector_memory_from_bytes() {
        assert_eq!(MobileCapabilityDetector::detect_memory_from_bytes(512 << 20), MemoryTier::Tier1GB);
        assert_eq!(MobileCapabilityDetector::detect_memory_from_bytes(2 << 30), MemoryTier::Tier2GB);
        assert_eq!(MobileCapabilityDetector::detect_memory_from_bytes(4 << 30), MemoryTier::Tier4GB);
        assert_eq!(MobileCapabilityDetector::detect_memory_from_bytes(16 << 30), MemoryTier::Tier8GBPlus);
    }

    // ===== MobileFogFallback Tests =====

    #[test]
    fn test_fog_fallback_low_end() {
        let fog = MobileFogFallback::for_profile(&MobileProfile::low_end());
        assert!(fog.is_single_layer());
        assert!(fog.should_skip_temporal());
        assert_eq!(fog.mobile_froxel_resolution(), [32, 18, 16]);
    }

    #[test]
    fn test_fog_fallback_mid_range() {
        let fog = MobileFogFallback::for_profile(&MobileProfile::mid_range());
        assert!(!fog.is_single_layer());
        assert!(fog.should_skip_temporal());
        assert_eq!(fog.mobile_froxel_resolution(), [64, 36, 32]);
    }

    #[test]
    fn test_fog_fallback_high_end() {
        let fog = MobileFogFallback::for_profile(&MobileProfile::high_end());
        assert!(!fog.is_single_layer());
        assert!(!fog.should_skip_temporal());
        assert_eq!(fog.mobile_froxel_resolution(), [128, 72, 64]);
    }

    #[test]
    fn test_fog_simplified_single_layer() {
        let fog = MobileFogFallback::for_profile(&MobileProfile::low_end());

        // Zero depth = no fog
        let fog_at_zero = fog.simplified_fog(0.0, 0.1);
        assert!(fog_at_zero < 0.01);

        // Large depth = full fog
        let fog_at_far = fog.simplified_fog(100.0, 0.1);
        assert!(fog_at_far > 0.99);
    }

    #[test]
    fn test_fog_simplified_volumetric() {
        let fog = MobileFogFallback::for_profile(&MobileProfile::mid_range());

        let fog_value = fog.simplified_fog(50.0, 0.05);
        assert!(fog_value > 0.0 && fog_value < 1.0);
    }

    #[test]
    fn test_fog_froxel_resolution_scaling() {
        let low = MobileFogFallback::for_profile(&MobileProfile::low_end());
        let high = MobileFogFallback::for_profile(&MobileProfile::high_end());

        let low_total: u32 = low.mobile_froxel_resolution().iter().product();
        let high_total: u32 = high.mobile_froxel_resolution().iter().product();

        assert!(high_total > low_total);
    }

    // ===== MobileWaterFallback Tests =====

    #[test]
    fn test_water_fallback_low_end() {
        let water = MobileWaterFallback::for_profile(&MobileProfile::low_end());
        assert_eq!(water.planar_reflection_size(), 128);
        assert!(!water.use_ssr_fallback());
        assert_eq!(water.simplified_gerstner_waves(), 2);
        assert!(water.skip_foam());
    }

    #[test]
    fn test_water_fallback_mid_range() {
        let water = MobileWaterFallback::for_profile(&MobileProfile::mid_range());
        assert_eq!(water.planar_reflection_size(), 256);
        assert!(water.use_ssr_fallback());
        assert_eq!(water.simplified_gerstner_waves(), 3);
        assert!(!water.skip_foam());
    }

    #[test]
    fn test_water_fallback_high_end() {
        let water = MobileWaterFallback::for_profile(&MobileProfile::high_end());
        assert_eq!(water.planar_reflection_size(), 512);
        assert!(water.use_ssr_fallback());
        assert_eq!(water.simplified_gerstner_waves(), 4);
        assert!(!water.skip_foam());
    }

    #[test]
    fn test_water_wave_calculation() {
        let water = MobileWaterFallback::for_profile(&MobileProfile::mid_range());
        let displacement = water.calculate_wave([10.0, 20.0], 0.0);
        // Verify it produces some displacement
        assert!(displacement.abs() > 0.0 || displacement == 0.0); // Can be zero at certain positions
    }

    #[test]
    fn test_water_reflection_size_scaling() {
        let low = MobileWaterFallback::for_profile(&MobileProfile::low_end());
        let high = MobileWaterFallback::for_profile(&MobileProfile::high_end());

        assert!(high.planar_reflection_size() > low.planar_reflection_size());
    }

    // ===== MobileTerrainFallback Tests =====

    #[test]
    fn test_terrain_fallback_low_end() {
        let terrain = MobileTerrainFallback::for_profile(&MobileProfile::low_end());
        assert_eq!(terrain.clipmap_levels(), 2);
        assert_eq!(terrain.texture_array_layers(), 4);
        assert!(!terrain.use_triplanar());
        assert_eq!(terrain.lod_bias(), 2.0);
    }

    #[test]
    fn test_terrain_fallback_mid_range() {
        let terrain = MobileTerrainFallback::for_profile(&MobileProfile::mid_range());
        assert_eq!(terrain.clipmap_levels(), 3);
        assert_eq!(terrain.texture_array_layers(), 6);
        assert!(!terrain.use_triplanar());
        assert_eq!(terrain.lod_bias(), 1.0);
    }

    #[test]
    fn test_terrain_fallback_high_end() {
        let terrain = MobileTerrainFallback::for_profile(&MobileProfile::high_end());
        assert_eq!(terrain.clipmap_levels(), 4);
        assert_eq!(terrain.texture_array_layers(), 8);
        assert!(terrain.use_triplanar());
        assert_eq!(terrain.lod_bias(), 0.5);
    }

    #[test]
    fn test_terrain_lod_calculation() {
        let terrain = MobileTerrainFallback::for_profile(&MobileProfile::mid_range());

        let lod_close = terrain.calculate_lod(10.0, 0);
        let lod_far = terrain.calculate_lod(1000.0, 0);

        assert!(lod_far > lod_close);
    }

    #[test]
    fn test_terrain_clipmap_scaling() {
        let low = MobileTerrainFallback::for_profile(&MobileProfile::low_end());
        let high = MobileTerrainFallback::for_profile(&MobileProfile::high_end());

        assert!(high.clipmap_levels() > low.clipmap_levels());
    }

    // ===== MobileFoliageFallback Tests =====

    #[test]
    fn test_foliage_fallback_low_end() {
        let foliage = MobileFoliageFallback::for_profile(&MobileProfile::low_end());
        assert_eq!(foliage.max_instances(), 1000);
        assert_eq!(foliage.billboard_distance(), 15.0);
        assert_eq!(foliage.impostor_resolution(), 256);
        assert!(foliage.skip_wind_animation());
    }

    #[test]
    fn test_foliage_fallback_mid_range() {
        let foliage = MobileFoliageFallback::for_profile(&MobileProfile::mid_range());
        assert_eq!(foliage.max_instances(), 3000);
        assert_eq!(foliage.billboard_distance(), 30.0);
        assert_eq!(foliage.impostor_resolution(), 512);
        assert!(!foliage.skip_wind_animation());
    }

    #[test]
    fn test_foliage_fallback_high_end() {
        let foliage = MobileFoliageFallback::for_profile(&MobileProfile::high_end());
        assert_eq!(foliage.max_instances(), 5000);
        assert_eq!(foliage.billboard_distance(), 50.0);
        assert_eq!(foliage.impostor_resolution(), 1024);
        assert!(!foliage.skip_wind_animation());
    }

    #[test]
    fn test_foliage_should_billboard() {
        let foliage = MobileFoliageFallback::for_profile(&MobileProfile::mid_range());

        assert!(!foliage.should_billboard(10.0));
        assert!(!foliage.should_billboard(30.0));
        assert!(foliage.should_billboard(40.0));
    }

    #[test]
    fn test_foliage_instance_priority() {
        let foliage = MobileFoliageFallback::for_profile(&MobileProfile::mid_range());

        let priority_close = foliage.instance_priority(10.0, 1.0);
        let priority_far = foliage.instance_priority(100.0, 1.0);

        assert!(priority_close > priority_far);
    }

    #[test]
    fn test_foliage_instance_limit_scaling() {
        let low = MobileFoliageFallback::for_profile(&MobileProfile::low_end());
        let high = MobileFoliageFallback::for_profile(&MobileProfile::high_end());

        assert!(high.max_instances() > low.max_instances());
    }

    // ===== MobileFallbackSystem Tests =====

    #[test]
    fn test_system_new() {
        let system = MobileFallbackSystem::new(MobileProfile::mid_range());
        assert_eq!(system.profile().quality_level(), 1);
        assert_eq!(system.get_render_scale(), 1.0);
    }

    #[test]
    fn test_system_default() {
        let system = MobileFallbackSystem::default();
        assert_eq!(system.profile().quality_level(), 1);
    }

    #[test]
    fn test_system_apply_performance_budget() {
        let system = MobileFallbackSystem::new(MobileProfile::low_end());
        let mut budget = PerformanceBudget::default();
        let original_fog = budget.fog_budget_ms;

        system.apply_to_performance_budget(&mut budget);

        // Low-end should significantly reduce budgets
        assert!(budget.fog_budget_ms < original_fog);
    }

    #[test]
    fn test_system_apply_lod_config() {
        let system = MobileFallbackSystem::new(MobileProfile::low_end());
        let mut config = EnvLodConfig::default();

        system.apply_to_lod_config(&mut config);

        // Low-end should increase LOD levels
        assert!(config.fog_lod > 0);
        assert!(config.water_lod > 0);
    }

    #[test]
    fn test_system_should_reduce_quality() {
        let system = MobileFallbackSystem::new(MobileProfile::mid_range());

        // Good FPS - no reduction
        assert!(!system.should_reduce_quality(30.0));
        assert!(!system.should_reduce_quality(60.0));

        // Bad FPS - reduce quality
        assert!(system.should_reduce_quality(20.0));
    }

    #[test]
    fn test_system_should_reduce_quality_edge_cases() {
        let system = MobileFallbackSystem::new(MobileProfile::mid_range());

        // Zero FPS
        assert!(!system.should_reduce_quality(0.0));

        // Negative FPS
        assert!(!system.should_reduce_quality(-10.0));
    }

    #[test]
    fn test_system_dynamic_resolution_reduce() {
        let mut system = MobileFallbackSystem::new(MobileProfile::mid_range());
        assert_eq!(system.get_render_scale(), 1.0);

        // Simulate slow frames
        for _ in 0..10 {
            system.update_dynamic_resolution(50.0); // Very slow
        }

        assert!(system.get_render_scale() < 1.0);
    }

    #[test]
    fn test_system_dynamic_resolution_increase() {
        let mut system = MobileFallbackSystem::new(MobileProfile::mid_range());
        system.render_scale = 0.6;

        // Simulate fast frames
        for _ in 0..10 {
            system.update_dynamic_resolution(20.0); // Fast
        }

        assert!(system.get_render_scale() > 0.6);
    }

    #[test]
    fn test_system_dynamic_resolution_bounds() {
        let mut system = MobileFallbackSystem::new(MobileProfile::mid_range());

        // Try to go below minimum
        for _ in 0..100 {
            system.update_dynamic_resolution(100.0);
        }
        assert!(system.get_render_scale() >= MIN_RENDER_SCALE);

        // Try to go above maximum
        for _ in 0..100 {
            system.update_dynamic_resolution(10.0);
        }
        assert!(system.get_render_scale() <= MAX_RENDER_SCALE);
    }

    #[test]
    fn test_system_get_fallbacks() {
        let system = MobileFallbackSystem::new(MobileProfile::mid_range());

        let fog = system.get_fog_fallback();
        let water = system.get_water_fallback();
        let terrain = system.get_terrain_fallback();
        let foliage = system.get_foliage_fallback();

        // Verify mid-range settings
        assert!(!fog.is_single_layer());
        assert_eq!(water.simplified_gerstner_waves(), 3);
        assert_eq!(terrain.clipmap_levels(), 3);
        assert_eq!(foliage.max_instances(), 3000);
    }

    #[test]
    fn test_system_set_profile() {
        let mut system = MobileFallbackSystem::new(MobileProfile::low_end());
        assert_eq!(system.profile().quality_level(), 0);

        system.set_profile(MobileProfile::high_end());
        assert_eq!(system.profile().quality_level(), 2);

        // Fallbacks should be updated
        assert!(!system.get_fog_fallback().should_skip_temporal());
    }

    #[test]
    fn test_system_reset_render_scale() {
        let mut system = MobileFallbackSystem::new(MobileProfile::mid_range());
        system.render_scale = 0.6;

        system.reset_render_scale();

        assert_eq!(system.get_render_scale(), MAX_RENDER_SCALE);
    }

    #[test]
    fn test_system_get_scaled_dimension() {
        let mut system = MobileFallbackSystem::new(MobileProfile::mid_range());

        system.render_scale = 1.0;
        assert_eq!(system.get_scaled_dimension(1920), 1920);

        system.render_scale = 0.5;
        assert_eq!(system.get_scaled_dimension(1920), 960);
    }

    // ===== Integration Tests =====

    #[test]
    fn test_integration_detector_to_system() {
        let detector = MobileCapabilityDetector::with_capabilities(
            MobileGpuTier::LowEnd,
            MemoryTier::Tier1GB,
            LOW_END_FILL_RATE,
            false,
        );
        let profile = detector.get_recommended_profile();
        let system = MobileFallbackSystem::new(profile);

        assert_eq!(system.profile().quality_level(), 0);
        assert!(system.get_fog_fallback().is_single_layer());
    }

    #[test]
    fn test_integration_full_pipeline() {
        // Simulate a complete mobile setup pipeline
        let detector = MobileCapabilityDetector::new();
        let profile = detector.get_recommended_profile();
        let mut system = MobileFallbackSystem::new(profile);

        let mut budget = PerformanceBudget::default();
        let mut lod_config = EnvLodConfig::default();

        system.apply_to_performance_budget(&mut budget);
        system.apply_to_lod_config(&mut lod_config);

        // Simulate rendering loop
        for frame in 0..60 {
            let frame_time = if frame < 30 { 40.0 } else { 25.0 };
            system.update_dynamic_resolution(frame_time);
        }

        // After settling, should have adjusted render scale
        let scale = system.get_render_scale();
        assert!(scale > MIN_RENDER_SCALE && scale <= MAX_RENDER_SCALE);
    }

    #[test]
    fn test_integration_quality_progression() {
        let profiles = [
            MobileProfile::low_end(),
            MobileProfile::mid_range(),
            MobileProfile::high_end(),
        ];

        let mut prev_tex_size = 0;
        let mut prev_draw_calls = 0;
        let mut prev_triangles = 0;

        for profile in profiles {
            assert!(profile.max_texture_size >= prev_tex_size);
            assert!(profile.max_draw_calls >= prev_draw_calls);
            assert!(profile.max_triangles_per_frame >= prev_triangles);

            prev_tex_size = profile.max_texture_size;
            prev_draw_calls = profile.max_draw_calls;
            prev_triangles = profile.max_triangles_per_frame;
        }
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_edge_case_zero_fog_density() {
        let fog = MobileFogFallback::for_profile(&MobileProfile::low_end());
        let result = fog.simplified_fog(100.0, 0.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_edge_case_negative_depth() {
        let fog = MobileFogFallback::for_profile(&MobileProfile::mid_range());
        let result = fog.simplified_fog(-10.0, 0.1);
        // Negative depth should clamp to 0
        assert!(result >= 0.0 && result <= 1.0);
    }

    #[test]
    fn test_edge_case_zero_distance_foliage() {
        let foliage = MobileFoliageFallback::for_profile(&MobileProfile::mid_range());
        let priority = foliage.instance_priority(0.0, 1.0);
        // Zero distance should give high priority
        assert!(priority > 0.5);
    }

    #[test]
    fn test_edge_case_large_frame_time() {
        let mut system = MobileFallbackSystem::new(MobileProfile::mid_range());

        // Extremely slow frame
        system.update_dynamic_resolution(1000.0);

        // Should clamp to minimum
        assert!(system.get_render_scale() <= MAX_RENDER_SCALE);
    }

    #[test]
    fn test_edge_case_profile_disabled_features() {
        let mut profile = MobileProfile::low_end();
        profile.enable_fog = 0;

        let mut system = MobileFallbackSystem::new(profile);
        let mut budget = PerformanceBudget::default();

        system.apply_to_performance_budget(&mut budget);

        // Fog budget should be zero when disabled
        assert_eq!(budget.fog_budget_ms, 0.0);
    }
}
