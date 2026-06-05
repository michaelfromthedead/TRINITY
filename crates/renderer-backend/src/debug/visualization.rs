//! Debug Visualization Modes for GPU Rendering
//!
//! This module provides debug visualization modes for GPU rendering debugging,
//! allowing developers to visualize different aspects of the rendering pipeline.
//!
//! # Overview
//!
//! Debug visualization modes help identify rendering issues by isolating specific
//! aspects of the rendering pipeline:
//!
//! - **Geometry Modes**: Wireframe, normals, tangents, UVs
//! - **Lighting Modes**: Albedo, roughness, metallic, AO, emissive
//! - **Depth/Motion Modes**: Depth, linear depth, motion vectors
//! - **Performance Modes**: Overdraw, mip levels, light complexity, shadow cascades
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::debug::visualization::*;
//!
//! let mut manager = DebugVisualizationManager::new();
//!
//! // Enable wireframe visualization
//! manager.set_mode(DebugVisualization::Wireframe);
//! manager.enable();
//!
//! // Get shader data for binding
//! let shader_data = DebugShaderData::from_config(manager.config());
//! // ... bind to uniform buffer ...
//!
//! // Cycle through modes
//! manager.cycle_next();
//! ```

use std::mem;

/// Debug visualization modes for rendering.
///
/// Each mode visualizes a specific aspect of the rendering pipeline to help
/// identify issues or understand rendering behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum DebugVisualization {
    /// No debug visualization (normal rendering).
    #[default]
    None = 0,

    // Geometry visualization modes
    /// Wireframe rendering showing mesh topology.
    Wireframe = 1,
    /// Visualize vertex/pixel normals as RGB colors.
    Normals = 2,
    /// Visualize tangent vectors as RGB colors.
    Tangents = 3,
    /// Visualize bitangent vectors as RGB colors.
    Bitangents = 4,
    /// Visualize UV coordinates as RG colors.
    UVs = 5,

    // Lighting visualization modes
    /// Show only albedo/diffuse color.
    Albedo = 6,
    /// Visualize roughness values as grayscale.
    Roughness = 7,
    /// Visualize metallic values as grayscale.
    Metallic = 8,
    /// Visualize ambient occlusion values.
    AmbientOcclusion = 9,
    /// Visualize emissive contribution.
    Emissive = 10,

    // Depth and motion visualization modes
    /// Visualize depth buffer (non-linear).
    Depth = 11,
    /// Visualize linearized depth.
    LinearDepth = 12,
    /// Visualize motion/velocity vectors.
    MotionVectors = 13,

    // Performance visualization modes
    /// Visualize overdraw as a heat map.
    Overdraw = 14,
    /// Visualize texture mip levels.
    MipLevel = 15,
    /// Visualize light complexity per pixel.
    LightComplexity = 16,
    /// Visualize shadow cascade boundaries.
    ShadowCascades = 17,
}

impl DebugVisualization {
    /// All visualization modes in order.
    pub const ALL: [DebugVisualization; 18] = [
        DebugVisualization::None,
        DebugVisualization::Wireframe,
        DebugVisualization::Normals,
        DebugVisualization::Tangents,
        DebugVisualization::Bitangents,
        DebugVisualization::UVs,
        DebugVisualization::Albedo,
        DebugVisualization::Roughness,
        DebugVisualization::Metallic,
        DebugVisualization::AmbientOcclusion,
        DebugVisualization::Emissive,
        DebugVisualization::Depth,
        DebugVisualization::LinearDepth,
        DebugVisualization::MotionVectors,
        DebugVisualization::Overdraw,
        DebugVisualization::MipLevel,
        DebugVisualization::LightComplexity,
        DebugVisualization::ShadowCascades,
    ];

    /// Geometry visualization modes.
    pub const GEOMETRY_MODES: [DebugVisualization; 5] = [
        DebugVisualization::Wireframe,
        DebugVisualization::Normals,
        DebugVisualization::Tangents,
        DebugVisualization::Bitangents,
        DebugVisualization::UVs,
    ];

    /// Lighting visualization modes.
    pub const LIGHTING_MODES: [DebugVisualization; 5] = [
        DebugVisualization::Albedo,
        DebugVisualization::Roughness,
        DebugVisualization::Metallic,
        DebugVisualization::AmbientOcclusion,
        DebugVisualization::Emissive,
    ];

    /// Performance visualization modes.
    pub const PERFORMANCE_MODES: [DebugVisualization; 4] = [
        DebugVisualization::Overdraw,
        DebugVisualization::MipLevel,
        DebugVisualization::LightComplexity,
        DebugVisualization::ShadowCascades,
    ];

    /// Get shader define string for this mode.
    ///
    /// Returns a string suitable for shader preprocessing.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mode = DebugVisualization::Wireframe;
    /// assert_eq!(mode.shader_define(), "DEBUG_VIS_WIREFRAME");
    /// ```
    #[must_use]
    pub const fn shader_define(&self) -> &'static str {
        match self {
            DebugVisualization::None => "DEBUG_VIS_NONE",
            DebugVisualization::Wireframe => "DEBUG_VIS_WIREFRAME",
            DebugVisualization::Normals => "DEBUG_VIS_NORMALS",
            DebugVisualization::Tangents => "DEBUG_VIS_TANGENTS",
            DebugVisualization::Bitangents => "DEBUG_VIS_BITANGENTS",
            DebugVisualization::UVs => "DEBUG_VIS_UVS",
            DebugVisualization::Albedo => "DEBUG_VIS_ALBEDO",
            DebugVisualization::Roughness => "DEBUG_VIS_ROUGHNESS",
            DebugVisualization::Metallic => "DEBUG_VIS_METALLIC",
            DebugVisualization::AmbientOcclusion => "DEBUG_VIS_AMBIENT_OCCLUSION",
            DebugVisualization::Emissive => "DEBUG_VIS_EMISSIVE",
            DebugVisualization::Depth => "DEBUG_VIS_DEPTH",
            DebugVisualization::LinearDepth => "DEBUG_VIS_LINEAR_DEPTH",
            DebugVisualization::MotionVectors => "DEBUG_VIS_MOTION_VECTORS",
            DebugVisualization::Overdraw => "DEBUG_VIS_OVERDRAW",
            DebugVisualization::MipLevel => "DEBUG_VIS_MIP_LEVEL",
            DebugVisualization::LightComplexity => "DEBUG_VIS_LIGHT_COMPLEXITY",
            DebugVisualization::ShadowCascades => "DEBUG_VIS_SHADOW_CASCADES",
        }
    }

    /// Get shader define value matching the enum repr.
    #[must_use]
    pub const fn define_value(&self) -> u32 {
        *self as u32
    }

    /// Check if this is a geometry visualization mode.
    #[must_use]
    pub const fn is_geometry(&self) -> bool {
        matches!(
            self,
            DebugVisualization::Wireframe
                | DebugVisualization::Normals
                | DebugVisualization::Tangents
                | DebugVisualization::Bitangents
                | DebugVisualization::UVs
        )
    }

    /// Check if this is a lighting visualization mode.
    #[must_use]
    pub const fn is_lighting(&self) -> bool {
        matches!(
            self,
            DebugVisualization::Albedo
                | DebugVisualization::Roughness
                | DebugVisualization::Metallic
                | DebugVisualization::AmbientOcclusion
                | DebugVisualization::Emissive
        )
    }

    /// Check if this is a performance visualization mode.
    #[must_use]
    pub const fn is_performance(&self) -> bool {
        matches!(
            self,
            DebugVisualization::Overdraw
                | DebugVisualization::MipLevel
                | DebugVisualization::LightComplexity
                | DebugVisualization::ShadowCascades
        )
    }

    /// Check if this is a depth or motion visualization mode.
    #[must_use]
    pub const fn is_depth_motion(&self) -> bool {
        matches!(
            self,
            DebugVisualization::Depth
                | DebugVisualization::LinearDepth
                | DebugVisualization::MotionVectors
        )
    }

    /// Get a human-readable description of this visualization mode.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            DebugVisualization::None => "Normal rendering without debug visualization",
            DebugVisualization::Wireframe => "Wireframe rendering showing mesh topology",
            DebugVisualization::Normals => "Vertex/pixel normals visualized as RGB colors",
            DebugVisualization::Tangents => "Tangent vectors visualized as RGB colors",
            DebugVisualization::Bitangents => "Bitangent vectors visualized as RGB colors",
            DebugVisualization::UVs => "UV coordinates visualized as RG colors",
            DebugVisualization::Albedo => "Albedo/diffuse color only",
            DebugVisualization::Roughness => "Roughness values as grayscale",
            DebugVisualization::Metallic => "Metallic values as grayscale",
            DebugVisualization::AmbientOcclusion => "Ambient occlusion values",
            DebugVisualization::Emissive => "Emissive contribution only",
            DebugVisualization::Depth => "Non-linear depth buffer visualization",
            DebugVisualization::LinearDepth => "Linearized depth visualization",
            DebugVisualization::MotionVectors => "Motion/velocity vectors",
            DebugVisualization::Overdraw => "Overdraw heat map",
            DebugVisualization::MipLevel => "Texture mip level visualization",
            DebugVisualization::LightComplexity => "Light complexity per pixel",
            DebugVisualization::ShadowCascades => "Shadow cascade boundaries",
        }
    }

    /// Get the index of this mode in the ALL array.
    #[must_use]
    pub const fn index(&self) -> usize {
        *self as usize
    }

    /// Create from index, returning None if out of bounds.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        if index < Self::ALL.len() {
            Some(Self::ALL[index])
        } else {
            None
        }
    }
}

/// Channel mask for selecting which color channels to display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelMask {
    /// Enable red channel.
    pub r: bool,
    /// Enable green channel.
    pub g: bool,
    /// Enable blue channel.
    pub b: bool,
    /// Enable alpha channel.
    pub a: bool,
}

impl ChannelMask {
    /// All channels enabled.
    pub const ALL: Self = Self {
        r: true,
        g: true,
        b: true,
        a: true,
    };

    /// Red channel only.
    pub const RED: Self = Self {
        r: true,
        g: false,
        b: false,
        a: false,
    };

    /// Green channel only.
    pub const GREEN: Self = Self {
        r: false,
        g: true,
        b: false,
        a: false,
    };

    /// Blue channel only.
    pub const BLUE: Self = Self {
        r: false,
        g: false,
        b: true,
        a: false,
    };

    /// Alpha channel only.
    pub const ALPHA: Self = Self {
        r: false,
        g: false,
        b: false,
        a: true,
    };

    /// RGB channels (no alpha).
    pub const RGB: Self = Self {
        r: true,
        g: true,
        b: true,
        a: false,
    };

    /// Create a new channel mask.
    #[must_use]
    pub const fn new(r: bool, g: bool, b: bool, a: bool) -> Self {
        Self { r, g, b, a }
    }

    /// Convert to a float array for shader binding [r, g, b, a].
    ///
    /// Each enabled channel is 1.0, disabled is 0.0.
    #[must_use]
    pub const fn as_floats(&self) -> [f32; 4] {
        [
            if self.r { 1.0 } else { 0.0 },
            if self.g { 1.0 } else { 0.0 },
            if self.b { 1.0 } else { 0.0 },
            if self.a { 1.0 } else { 0.0 },
        ]
    }

    /// Check if all channels are enabled.
    #[must_use]
    pub const fn is_all(&self) -> bool {
        self.r && self.g && self.b && self.a
    }

    /// Check if only RGB channels are enabled.
    #[must_use]
    pub const fn is_rgb(&self) -> bool {
        self.r && self.g && self.b && !self.a
    }

    /// Check if no channels are enabled.
    #[must_use]
    pub const fn is_none(&self) -> bool {
        !self.r && !self.g && !self.b && !self.a
    }

    /// Count the number of enabled channels.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.r as u32 + self.g as u32 + self.b as u32 + self.a as u32
    }
}

impl Default for ChannelMask {
    fn default() -> Self {
        Self::ALL
    }
}

/// Configuration for debug visualization.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualizationConfig {
    /// The active visualization mode.
    pub mode: DebugVisualization,
    /// Intensity/scale factor for visualization (0.0 - 2.0 typical).
    pub intensity: f32,
    /// Whether to overlay on top of normal rendering.
    pub overlay: bool,
    /// Which color channels to display.
    pub channel_mask: ChannelMask,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            mode: DebugVisualization::None,
            intensity: 1.0,
            overlay: false,
            channel_mask: ChannelMask::ALL,
        }
    }
}

impl VisualizationConfig {
    /// Create a new config with the specified mode.
    #[must_use]
    pub fn with_mode(mode: DebugVisualization) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set the intensity.
    #[must_use]
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    /// Set overlay mode.
    #[must_use]
    pub fn with_overlay(mut self, overlay: bool) -> Self {
        self.overlay = overlay;
        self
    }

    /// Set the channel mask.
    #[must_use]
    pub fn with_channel_mask(mut self, mask: ChannelMask) -> Self {
        self.channel_mask = mask;
        self
    }
}

/// Manager for debug visualization state.
///
/// Provides a central point for managing visualization modes and cycling
/// through them for debugging purposes.
#[derive(Debug, Clone)]
pub struct DebugVisualizationManager {
    current_mode: DebugVisualization,
    config: VisualizationConfig,
    enabled: bool,
}

impl Default for DebugVisualizationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugVisualizationManager {
    /// Create a new visualization manager with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_mode: DebugVisualization::None,
            config: VisualizationConfig::default(),
            enabled: false,
        }
    }

    /// Set the current visualization mode.
    pub fn set_mode(&mut self, mode: DebugVisualization) {
        self.current_mode = mode;
        self.config.mode = mode;
    }

    /// Get the current visualization mode.
    #[must_use]
    pub fn current_mode(&self) -> DebugVisualization {
        self.current_mode
    }

    /// Toggle visualization on/off.
    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }

    /// Enable visualization.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable visualization.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if visualization is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Cycle to the next visualization mode.
    ///
    /// Wraps around from the last mode to None.
    pub fn cycle_next(&mut self) {
        let next_index = (self.current_mode.index() + 1) % DebugVisualization::ALL.len();
        self.set_mode(DebugVisualization::ALL[next_index]);
    }

    /// Cycle to the previous visualization mode.
    ///
    /// Wraps around from None to the last mode.
    pub fn cycle_prev(&mut self) {
        let prev_index = if self.current_mode.index() == 0 {
            DebugVisualization::ALL.len() - 1
        } else {
            self.current_mode.index() - 1
        };
        self.set_mode(DebugVisualization::ALL[prev_index]);
    }

    /// Set the visualization configuration.
    pub fn set_config(&mut self, config: VisualizationConfig) {
        self.current_mode = config.mode;
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &VisualizationConfig {
        &self.config
    }

    /// Get a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut VisualizationConfig {
        &mut self.config
    }

    /// Get all available visualization modes.
    #[must_use]
    pub fn all_modes() -> &'static [DebugVisualization] {
        &DebugVisualization::ALL
    }

    /// Get the effective mode (None if disabled).
    #[must_use]
    pub fn effective_mode(&self) -> DebugVisualization {
        if self.enabled {
            self.current_mode
        } else {
            DebugVisualization::None
        }
    }

    /// Set intensity.
    pub fn set_intensity(&mut self, intensity: f32) {
        self.config.intensity = intensity;
    }

    /// Get intensity.
    #[must_use]
    pub fn intensity(&self) -> f32 {
        self.config.intensity
    }

    /// Set overlay mode.
    pub fn set_overlay(&mut self, overlay: bool) {
        self.config.overlay = overlay;
    }

    /// Check if overlay mode is enabled.
    #[must_use]
    pub fn is_overlay(&self) -> bool {
        self.config.overlay
    }

    /// Set the channel mask.
    pub fn set_channel_mask(&mut self, mask: ChannelMask) {
        self.config.channel_mask = mask;
    }

    /// Get the channel mask.
    #[must_use]
    pub fn channel_mask(&self) -> ChannelMask {
        self.config.channel_mask
    }
}

/// Shader data structure for debug visualization.
///
/// This struct is designed to be bound to a uniform buffer for shader access.
/// It follows std140 layout rules for GPU compatibility.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebugShaderData {
    /// The visualization mode (matches DebugVisualization repr).
    pub mode: u32,
    /// Intensity/scale factor.
    pub intensity: f32,
    /// Channel mask as floats [r, g, b, a].
    pub channel_mask: [f32; 4],
    /// Padding for std140 alignment (2 floats to reach 32 bytes).
    pub _padding: [f32; 2],
}

impl Default for DebugShaderData {
    fn default() -> Self {
        Self {
            mode: 0,
            intensity: 1.0,
            channel_mask: [1.0, 1.0, 1.0, 1.0],
            _padding: [0.0, 0.0],
        }
    }
}

impl DebugShaderData {
    /// Create shader data from a visualization configuration.
    #[must_use]
    pub fn from_config(config: &VisualizationConfig) -> Self {
        Self {
            mode: config.mode.define_value(),
            intensity: config.intensity,
            channel_mask: config.channel_mask.as_floats(),
            _padding: [0.0, 0.0],
        }
    }

    /// Create shader data from a manager (uses effective mode).
    #[must_use]
    pub fn from_manager(manager: &DebugVisualizationManager) -> Self {
        Self {
            mode: manager.effective_mode().define_value(),
            intensity: manager.intensity(),
            channel_mask: manager.channel_mask().as_floats(),
            _padding: [0.0, 0.0],
        }
    }

    /// Get as a byte slice for GPU binding.
    ///
    /// # Safety
    ///
    /// The returned slice is valid for the lifetime of self.
    /// The struct uses repr(C) to ensure consistent layout.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: DebugShaderData is repr(C) with known layout
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                mem::size_of::<Self>(),
            )
        }
    }

    /// Get the size of this struct in bytes.
    #[must_use]
    pub const fn size() -> usize {
        mem::size_of::<Self>()
    }

    /// Check if this data represents disabled visualization.
    #[must_use]
    pub fn is_disabled(&self) -> bool {
        self.mode == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: DebugVisualization::None
    #[test]
    fn test_debug_visualization_none() {
        let mode = DebugVisualization::None;
        assert_eq!(mode.define_value(), 0);
        assert_eq!(mode.shader_define(), "DEBUG_VIS_NONE");
        assert!(!mode.is_geometry());
        assert!(!mode.is_lighting());
        assert!(!mode.is_performance());
    }

    // Test 2: DebugVisualization::Wireframe
    #[test]
    fn test_debug_visualization_wireframe() {
        let mode = DebugVisualization::Wireframe;
        assert_eq!(mode.define_value(), 1);
        assert_eq!(mode.shader_define(), "DEBUG_VIS_WIREFRAME");
        assert!(mode.is_geometry());
        assert!(!mode.is_lighting());
        assert!(!mode.is_performance());
    }

    // Test 3: All variants exist
    #[test]
    fn test_debug_visualization_all_variants() {
        assert_eq!(DebugVisualization::ALL.len(), 18);
        for (i, mode) in DebugVisualization::ALL.iter().enumerate() {
            assert_eq!(mode.index(), i);
            assert_eq!(mode.define_value() as usize, i);
        }
    }

    // Test 4: Shader define format
    #[test]
    fn test_shader_define_format() {
        for mode in DebugVisualization::ALL.iter() {
            let define = mode.shader_define();
            assert!(define.starts_with("DEBUG_VIS_"), "Define should start with DEBUG_VIS_: {}", define);
            assert!(!define.is_empty());
        }
    }

    // Test 5: Define value matches repr
    #[test]
    fn test_define_value_matches_repr() {
        assert_eq!(DebugVisualization::None.define_value(), 0);
        assert_eq!(DebugVisualization::Wireframe.define_value(), 1);
        assert_eq!(DebugVisualization::Normals.define_value(), 2);
        assert_eq!(DebugVisualization::UVs.define_value(), 5);
        assert_eq!(DebugVisualization::Albedo.define_value(), 6);
        assert_eq!(DebugVisualization::ShadowCascades.define_value(), 17);
    }

    // Test 6: is_geometry modes
    #[test]
    fn test_is_geometry_modes() {
        let geometry_modes = [
            DebugVisualization::Wireframe,
            DebugVisualization::Normals,
            DebugVisualization::Tangents,
            DebugVisualization::Bitangents,
            DebugVisualization::UVs,
        ];
        for mode in geometry_modes.iter() {
            assert!(mode.is_geometry(), "{:?} should be geometry", mode);
        }
        assert!(!DebugVisualization::Albedo.is_geometry());
        assert!(!DebugVisualization::Overdraw.is_geometry());
    }

    // Test 7: is_lighting modes
    #[test]
    fn test_is_lighting_modes() {
        let lighting_modes = [
            DebugVisualization::Albedo,
            DebugVisualization::Roughness,
            DebugVisualization::Metallic,
            DebugVisualization::AmbientOcclusion,
            DebugVisualization::Emissive,
        ];
        for mode in lighting_modes.iter() {
            assert!(mode.is_lighting(), "{:?} should be lighting", mode);
        }
        assert!(!DebugVisualization::Wireframe.is_lighting());
        assert!(!DebugVisualization::Depth.is_lighting());
    }

    // Test 8: is_performance modes
    #[test]
    fn test_is_performance_modes() {
        let perf_modes = [
            DebugVisualization::Overdraw,
            DebugVisualization::MipLevel,
            DebugVisualization::LightComplexity,
            DebugVisualization::ShadowCascades,
        ];
        for mode in perf_modes.iter() {
            assert!(mode.is_performance(), "{:?} should be performance", mode);
        }
        assert!(!DebugVisualization::Normals.is_performance());
        assert!(!DebugVisualization::Albedo.is_performance());
    }

    // Test 9: descriptions are not empty
    #[test]
    fn test_description_not_empty() {
        for mode in DebugVisualization::ALL.iter() {
            let desc = mode.description();
            assert!(!desc.is_empty(), "{:?} should have description", mode);
            assert!(desc.len() > 5, "{:?} description too short: {}", mode, desc);
        }
    }

    // Test 10: ChannelMask::ALL
    #[test]
    fn test_channel_mask_all() {
        let mask = ChannelMask::ALL;
        assert!(mask.r && mask.g && mask.b && mask.a);
        assert!(mask.is_all());
        assert!(!mask.is_rgb());
        assert!(!mask.is_none());
        assert_eq!(mask.count(), 4);
        assert_eq!(mask.as_floats(), [1.0, 1.0, 1.0, 1.0]);
    }

    // Test 11: ChannelMask::RED
    #[test]
    fn test_channel_mask_red() {
        let mask = ChannelMask::RED;
        assert!(mask.r);
        assert!(!mask.g && !mask.b && !mask.a);
        assert!(!mask.is_all());
        assert_eq!(mask.count(), 1);
        assert_eq!(mask.as_floats(), [1.0, 0.0, 0.0, 0.0]);
    }

    // Test 12: ChannelMask::RGB
    #[test]
    fn test_channel_mask_rgb() {
        let mask = ChannelMask::RGB;
        assert!(mask.r && mask.g && mask.b);
        assert!(!mask.a);
        assert!(mask.is_rgb());
        assert!(!mask.is_all());
        assert_eq!(mask.count(), 3);
        assert_eq!(mask.as_floats(), [1.0, 1.0, 1.0, 0.0]);
    }

    // Test 13: VisualizationConfig default
    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        assert_eq!(config.mode, DebugVisualization::None);
        assert_eq!(config.intensity, 1.0);
        assert!(!config.overlay);
        assert!(config.channel_mask.is_all());
    }

    // Test 14: VisualizationConfig with overlay
    #[test]
    fn test_visualization_config_overlay() {
        let config = VisualizationConfig::with_mode(DebugVisualization::Wireframe)
            .with_overlay(true)
            .with_intensity(0.5);

        assert_eq!(config.mode, DebugVisualization::Wireframe);
        assert_eq!(config.intensity, 0.5);
        assert!(config.overlay);
    }

    // Test 15: DebugVisualizationManager::new
    #[test]
    fn test_manager_new() {
        let manager = DebugVisualizationManager::new();
        assert_eq!(manager.current_mode(), DebugVisualization::None);
        assert!(!manager.is_enabled());
        assert_eq!(manager.intensity(), 1.0);
        assert!(!manager.is_overlay());
    }

    // Test 16: DebugVisualizationManager::set_mode
    #[test]
    fn test_manager_set_mode() {
        let mut manager = DebugVisualizationManager::new();
        manager.set_mode(DebugVisualization::Normals);
        assert_eq!(manager.current_mode(), DebugVisualization::Normals);
        assert_eq!(manager.config().mode, DebugVisualization::Normals);
    }

    // Test 17: DebugVisualizationManager::toggle
    #[test]
    fn test_manager_toggle() {
        let mut manager = DebugVisualizationManager::new();
        assert!(!manager.is_enabled());

        manager.toggle();
        assert!(manager.is_enabled());

        manager.toggle();
        assert!(!manager.is_enabled());
    }

    // Test 18: DebugVisualizationManager::cycle_next
    #[test]
    fn test_manager_cycle_next() {
        let mut manager = DebugVisualizationManager::new();
        assert_eq!(manager.current_mode(), DebugVisualization::None);

        manager.cycle_next();
        assert_eq!(manager.current_mode(), DebugVisualization::Wireframe);

        manager.cycle_next();
        assert_eq!(manager.current_mode(), DebugVisualization::Normals);

        // Cycle to last and wrap
        for _ in 0..15 {
            manager.cycle_next();
        }
        assert_eq!(manager.current_mode(), DebugVisualization::ShadowCascades);

        manager.cycle_next();
        assert_eq!(manager.current_mode(), DebugVisualization::None);
    }

    // Test 19: DebugVisualizationManager::cycle_prev
    #[test]
    fn test_manager_cycle_prev() {
        let mut manager = DebugVisualizationManager::new();
        assert_eq!(manager.current_mode(), DebugVisualization::None);

        // Wrap to last
        manager.cycle_prev();
        assert_eq!(manager.current_mode(), DebugVisualization::ShadowCascades);

        manager.cycle_prev();
        assert_eq!(manager.current_mode(), DebugVisualization::LightComplexity);
    }

    // Test 20: DebugVisualizationManager::all_modes
    #[test]
    fn test_manager_all_modes() {
        let modes = DebugVisualizationManager::all_modes();
        assert_eq!(modes.len(), 18);
        assert_eq!(modes[0], DebugVisualization::None);
        assert_eq!(modes[17], DebugVisualization::ShadowCascades);
    }

    // Test 21: DebugShaderData::from_config
    #[test]
    fn test_shader_data_from_config() {
        let config = VisualizationConfig::with_mode(DebugVisualization::Albedo)
            .with_intensity(0.8)
            .with_channel_mask(ChannelMask::RGB);

        let data = DebugShaderData::from_config(&config);
        assert_eq!(data.mode, 6); // Albedo = 6
        assert_eq!(data.intensity, 0.8);
        assert_eq!(data.channel_mask, [1.0, 1.0, 1.0, 0.0]);
    }

    // Test 22: DebugShaderData::as_bytes
    #[test]
    fn test_shader_data_as_bytes() {
        let data = DebugShaderData::default();
        let bytes = data.as_bytes();

        assert_eq!(bytes.len(), DebugShaderData::size());
        assert_eq!(bytes.len(), std::mem::size_of::<DebugShaderData>());

        // Verify it's 32 bytes (u32 + f32 + [f32;4] + [f32;2] = 4 + 4 + 16 + 8 = 32)
        assert_eq!(bytes.len(), 32);
    }

    // Test 23: DebugShaderData::from_manager
    #[test]
    fn test_shader_data_from_manager() {
        let mut manager = DebugVisualizationManager::new();
        manager.set_mode(DebugVisualization::Overdraw);
        manager.enable();

        let data = DebugShaderData::from_manager(&manager);
        assert_eq!(data.mode, 14); // Overdraw = 14

        // When disabled, effective mode is None
        manager.disable();
        let data2 = DebugShaderData::from_manager(&manager);
        assert_eq!(data2.mode, 0);
    }

    // Test 24: ChannelMask other constants
    #[test]
    fn test_channel_mask_constants() {
        assert_eq!(ChannelMask::GREEN.as_floats(), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(ChannelMask::BLUE.as_floats(), [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(ChannelMask::ALPHA.as_floats(), [0.0, 0.0, 0.0, 1.0]);
    }

    // Test 25: from_index
    #[test]
    fn test_from_index() {
        assert_eq!(DebugVisualization::from_index(0), Some(DebugVisualization::None));
        assert_eq!(DebugVisualization::from_index(1), Some(DebugVisualization::Wireframe));
        assert_eq!(DebugVisualization::from_index(17), Some(DebugVisualization::ShadowCascades));
        assert_eq!(DebugVisualization::from_index(18), None);
        assert_eq!(DebugVisualization::from_index(100), None);
    }

    // Test 26: is_depth_motion modes
    #[test]
    fn test_is_depth_motion_modes() {
        assert!(DebugVisualization::Depth.is_depth_motion());
        assert!(DebugVisualization::LinearDepth.is_depth_motion());
        assert!(DebugVisualization::MotionVectors.is_depth_motion());
        assert!(!DebugVisualization::Albedo.is_depth_motion());
        assert!(!DebugVisualization::Wireframe.is_depth_motion());
    }

    // Test 27: effective_mode
    #[test]
    fn test_effective_mode() {
        let mut manager = DebugVisualizationManager::new();
        manager.set_mode(DebugVisualization::Normals);

        // Disabled = None
        assert_eq!(manager.effective_mode(), DebugVisualization::None);

        manager.enable();
        assert_eq!(manager.effective_mode(), DebugVisualization::Normals);
    }

    // Test 28: ChannelMask::new
    #[test]
    fn test_channel_mask_new() {
        let mask = ChannelMask::new(true, false, true, false);
        assert!(mask.r);
        assert!(!mask.g);
        assert!(mask.b);
        assert!(!mask.a);
        assert_eq!(mask.count(), 2);
    }

    // Test 29: DebugShaderData::is_disabled
    #[test]
    fn test_shader_data_is_disabled() {
        let data = DebugShaderData::default();
        assert!(data.is_disabled());

        let data2 = DebugShaderData::from_config(&VisualizationConfig::with_mode(DebugVisualization::Wireframe));
        assert!(!data2.is_disabled());
    }

    // Test 30: Mode category arrays
    #[test]
    fn test_mode_category_arrays() {
        assert_eq!(DebugVisualization::GEOMETRY_MODES.len(), 5);
        assert_eq!(DebugVisualization::LIGHTING_MODES.len(), 5);
        assert_eq!(DebugVisualization::PERFORMANCE_MODES.len(), 4);

        for mode in DebugVisualization::GEOMETRY_MODES.iter() {
            assert!(mode.is_geometry());
        }
        for mode in DebugVisualization::LIGHTING_MODES.iter() {
            assert!(mode.is_lighting());
        }
        for mode in DebugVisualization::PERFORMANCE_MODES.iter() {
            assert!(mode.is_performance());
        }
    }
}
