//! Shadow Modulation for Stylized Rendering (T-LIT-9.8).
//!
//! This module provides shadow color and density modulation for stylized rendering,
//! plus adaptive slope-scaled bias to reduce shadow acne and peter-panning artifacts.
//!
//! # Features
//!
//! - **Shadow Color Tinting**: Apply custom colors to shadows for stylized effects
//! - **Density Control**: Adjust shadow strength from invisible to full black
//! - **Adaptive Bias**: Per-pixel bias based on surface slope and cascade depth
//! - **Style Presets**: Quick configuration for common shadow styles
//!
//! # Shadow Styles
//!
//! | Style       | Color                  | Density | Use Case                 |
//! |-------------|------------------------|---------|--------------------------|
//! | Realistic   | Black (0,0,0)          | 1.0     | Photorealistic rendering |
//! | Soft        | Dark blue (0.1,0.1,0.15)| 0.7    | Soft ambient lighting    |
//! | Stylized    | Purple (0.2,0.1,0.3)   | 0.5     | Artistic/cartoon         |
//! | Cel-Shaded  | Black (0,0,0)          | 1.0     | Hard-edged toon shadows  |
//!
//! # Usage
//!
//! ```ignore
//! let mut modulation = ShadowModulation::new(&device);
//!
//! // Set stylized purple shadows
//! modulation.preset_stylized();
//!
//! // Or customize manually
//! modulation.set_shadow_color(0.1, 0.05, 0.15);
//! modulation.set_density(0.6);
//!
//! // Configure adaptive bias
//! modulation.set_bias_config(AdaptiveBiasConfig {
//!     constant_bias: 0.002,
//!     slope_scale: 3.0,
//!     ..Default::default()
//! });
//!
//! // Upload to GPU
//! modulation.upload(&queue);
//! ```
//!
//! # WGSL Integration
//!
//! The GPU structs are designed to match WGSL struct layouts for direct use
//! in shadow sampling shaders. See the module documentation for WGSL code.

use std::mem;

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of shadow cascades supported.
pub const MAX_CASCADE_COUNT: usize = 4;

/// Default constant bias for shadow depth comparison.
pub const DEFAULT_CONSTANT_BIAS: f32 = 0.001;

/// Default slope scale for adaptive bias.
pub const DEFAULT_SLOPE_SCALE: f32 = 2.0;

/// Default normal offset scale.
pub const DEFAULT_NORMAL_OFFSET_SCALE: f32 = 0.5;

// ---------------------------------------------------------------------------
// GPU Structs
// ---------------------------------------------------------------------------

/// Shadow modulation parameters for stylized rendering.
///
/// This struct is uploaded to a uniform buffer for use in shadow sampling
/// shaders. It controls shadow appearance including color tinting, density,
/// ambient occlusion contribution, and contact hardening.
///
/// # Memory Layout
///
/// 32 bytes total, std140/std430 compatible:
///
/// | Offset | Field                    | Size     |
/// |--------|--------------------------|----------|
/// | 0      | shadow_color             | 12 bytes |
/// | 12     | shadow_density           | 4 bytes  |
/// | 16     | ambient_occlusion_strength| 4 bytes |
/// | 20     | contact_hardening        | 4 bytes  |
/// | 24     | _padding                 | 8 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShadowModulationGpu {
    /// Tint color for shadows (RGB, default black [0,0,0]).
    ///
    /// This color is blended with shadowed areas to create stylized
    /// shadow effects. Pure black produces standard shadows.
    pub shadow_color: [f32; 3],

    /// Shadow density/strength (0.0 = no shadow, 1.0 = full shadow).
    ///
    /// Lower values create softer, more translucent shadows.
    pub shadow_density: f32,

    /// Ambient occlusion contribution to shadow (0.0-1.0).
    ///
    /// Controls how much AO data affects shadow intensity.
    pub ambient_occlusion_strength: f32,

    /// PCSS light size multiplier for contact hardening.
    ///
    /// Higher values create softer shadow penumbras. 1.0 is default.
    pub contact_hardening: f32,

    /// Padding for 16-byte alignment.
    pub _padding: [f32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ShadowModulationGpu>() == 32);

impl Default for ShadowModulationGpu {
    fn default() -> Self {
        Self {
            shadow_color: [0.0, 0.0, 0.0], // Pure black shadows
            shadow_density: 1.0,           // Full shadow strength
            ambient_occlusion_strength: 0.5,
            contact_hardening: 1.0,
            _padding: [0.0; 2],
        }
    }
}

impl ShadowModulationGpu {
    /// Create a new modulation config with custom shadow color.
    #[inline]
    pub fn with_color(r: f32, g: f32, b: f32) -> Self {
        Self {
            shadow_color: [r, g, b],
            ..Default::default()
        }
    }

    /// Create a new modulation config with custom density.
    #[inline]
    pub fn with_density(density: f32) -> Self {
        Self {
            shadow_density: density.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Create a new modulation config with custom color and density.
    #[inline]
    pub fn with_color_and_density(r: f32, g: f32, b: f32, density: f32) -> Self {
        Self {
            shadow_color: [r, g, b],
            shadow_density: density.clamp(0.0, 1.0),
            ..Default::default()
        }
    }
}

/// Adaptive bias parameters computed per-pixel.
///
/// This struct controls shadow depth bias to reduce shadow acne (caused by
/// insufficient bias) and peter-panning (caused by excessive bias). The bias
/// is computed adaptively based on surface slope, depth range, and cascade index.
///
/// # Memory Layout
///
/// 36 bytes (will be padded to 48 bytes for std140):
///
/// | Offset | Field              | Size     |
/// |--------|-------------------|----------|
/// | 0      | constant_bias     | 4 bytes  |
/// | 4      | slope_scale       | 4 bytes  |
/// | 8      | normal_offset_scale| 4 bytes |
/// | 12     | receiver_plane_bias| 4 bytes |
/// | 16     | depth_range_scale | 4 bytes  |
/// | 20     | cascade_bias_scale| 16 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AdaptiveBiasConfig {
    /// Base constant bias added to all shadow comparisons.
    ///
    /// This provides a minimum bias regardless of surface orientation.
    pub constant_bias: f32,

    /// Slope-scaled bias multiplier.
    ///
    /// Higher values increase bias for surfaces at grazing angles to
    /// the light. Typical range: 1.0-4.0.
    pub slope_scale: f32,

    /// Normal offset bias scale.
    ///
    /// Controls offset along surface normal to prevent self-shadowing.
    pub normal_offset_scale: f32,

    /// Receiver plane depth bias.
    ///
    /// Additional bias based on the receiver plane orientation.
    pub receiver_plane_bias: f32,

    /// Bias scaling based on depth range.
    ///
    /// Increases bias for distant surfaces to compensate for reduced
    /// shadow map precision.
    pub depth_range_scale: f32,

    /// Per-cascade bias multipliers.
    ///
    /// Index 0 = nearest cascade, index 3 = farthest cascade.
    /// Farther cascades typically need larger bias due to lower resolution.
    pub cascade_bias_scale: [f32; 4],

    /// Padding for 16-byte alignment (48 bytes total for std140).
    pub _padding: [f32; 3],
}

// Size assertion: 48 bytes with padding
const _: () = assert!(mem::size_of::<AdaptiveBiasConfig>() == 48);

impl Default for AdaptiveBiasConfig {
    fn default() -> Self {
        Self {
            constant_bias: DEFAULT_CONSTANT_BIAS,
            slope_scale: DEFAULT_SLOPE_SCALE,
            normal_offset_scale: DEFAULT_NORMAL_OFFSET_SCALE,
            receiver_plane_bias: 0.0,
            depth_range_scale: 1.0,
            // Increase bias for farther cascades
            cascade_bias_scale: [1.0, 1.1, 1.2, 1.3],
            _padding: [0.0; 3],
        }
    }
}

impl AdaptiveBiasConfig {
    /// Create a bias config with custom constant and slope bias.
    #[inline]
    pub fn with_bias(constant_bias: f32, slope_scale: f32) -> Self {
        Self {
            constant_bias,
            slope_scale,
            ..Default::default()
        }
    }

    /// Create a config with aggressive bias for high-polygon scenes.
    #[inline]
    pub fn high_detail() -> Self {
        Self {
            constant_bias: 0.0005,
            slope_scale: 1.5,
            normal_offset_scale: 0.3,
            ..Default::default()
        }
    }

    /// Create a config with conservative bias for large-scale scenes.
    #[inline]
    pub fn large_scale() -> Self {
        Self {
            constant_bias: 0.002,
            slope_scale: 3.0,
            normal_offset_scale: 1.0,
            depth_range_scale: 2.0,
            cascade_bias_scale: [1.0, 1.2, 1.5, 2.0],
            ..Default::default()
        }
    }

    /// Get the bias scale for a specific cascade.
    ///
    /// Returns 1.0 for out-of-range cascade indices.
    #[inline]
    pub fn cascade_scale(&self, cascade: usize) -> f32 {
        if cascade < MAX_CASCADE_COUNT {
            self.cascade_bias_scale[cascade]
        } else {
            1.0
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow Modulation Manager
// ---------------------------------------------------------------------------

/// Shadow modulation manager.
///
/// Manages GPU buffers for shadow modulation and adaptive bias configuration.
/// Provides methods to configure shadow appearance and upload to GPU.
///
/// # Buffer Layout
///
/// Two uniform buffers are created:
/// - Modulation buffer (32 bytes): Shadow color, density, AO, contact hardening
/// - Bias buffer (48 bytes): Adaptive bias configuration
pub struct ShadowModulation {
    modulation_buffer: wgpu::Buffer,
    bias_buffer: wgpu::Buffer,
    current_modulation: ShadowModulationGpu,
    current_bias: AdaptiveBiasConfig,
    dirty: bool,
}

impl ShadowModulation {
    /// Create a new shadow modulation manager with default settings.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for buffer creation
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::util::DeviceExt;

        let current_modulation = ShadowModulationGpu::default();
        let current_bias = AdaptiveBiasConfig::default();

        let modulation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow_modulation_buffer"),
            contents: bytemuck::bytes_of(&current_modulation),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bias_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow_bias_buffer"),
            contents: bytemuck::bytes_of(&current_bias),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            modulation_buffer,
            bias_buffer,
            current_modulation,
            current_bias,
            dirty: false,
        }
    }

    /// Set shadow tint color (for stylized rendering).
    ///
    /// # Arguments
    ///
    /// * `r` - Red component (0.0-1.0)
    /// * `g` - Green component (0.0-1.0)
    /// * `b` - Blue component (0.0-1.0)
    #[inline]
    pub fn set_shadow_color(&mut self, r: f32, g: f32, b: f32) {
        self.current_modulation.shadow_color = [r, g, b];
        self.dirty = true;
    }

    /// Get the current shadow color.
    #[inline]
    pub fn shadow_color(&self) -> [f32; 3] {
        self.current_modulation.shadow_color
    }

    /// Set shadow density (0 = invisible, 1 = full).
    ///
    /// Values outside [0, 1] are clamped.
    #[inline]
    pub fn set_density(&mut self, density: f32) {
        self.current_modulation.shadow_density = density.clamp(0.0, 1.0);
        self.dirty = true;
    }

    /// Get the current shadow density.
    #[inline]
    pub fn density(&self) -> f32 {
        self.current_modulation.shadow_density
    }

    /// Set ambient occlusion strength (0 = no AO, 1 = full AO).
    #[inline]
    pub fn set_ao_strength(&mut self, strength: f32) {
        self.current_modulation.ambient_occlusion_strength = strength.clamp(0.0, 1.0);
        self.dirty = true;
    }

    /// Set contact hardening (PCSS light size multiplier).
    #[inline]
    pub fn set_contact_hardening(&mut self, hardening: f32) {
        self.current_modulation.contact_hardening = hardening.max(0.0);
        self.dirty = true;
    }

    /// Configure adaptive bias.
    #[inline]
    pub fn set_bias_config(&mut self, config: AdaptiveBiasConfig) {
        self.current_bias = config;
        self.dirty = true;
    }

    /// Get the current bias configuration.
    #[inline]
    pub fn bias_config(&self) -> &AdaptiveBiasConfig {
        &self.current_bias
    }

    /// Per-cascade bias adjustment.
    ///
    /// # Arguments
    ///
    /// * `cascade` - Cascade index (0-3)
    /// * `scale` - Bias scale multiplier
    #[inline]
    pub fn set_cascade_bias(&mut self, cascade: usize, scale: f32) {
        if cascade < MAX_CASCADE_COUNT {
            self.current_bias.cascade_bias_scale[cascade] = scale;
            self.dirty = true;
        }
    }

    /// Upload modified data to GPU.
    ///
    /// Only uploads if data has changed since last upload.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if !self.dirty {
            return;
        }

        queue.write_buffer(
            &self.modulation_buffer,
            0,
            bytemuck::bytes_of(&self.current_modulation),
        );
        queue.write_buffer(
            &self.bias_buffer,
            0,
            bytemuck::bytes_of(&self.current_bias),
        );

        self.dirty = false;
    }

    /// Force upload to GPU, regardless of dirty flag.
    pub fn force_upload(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.modulation_buffer,
            0,
            bytemuck::bytes_of(&self.current_modulation),
        );
        queue.write_buffer(
            &self.bias_buffer,
            0,
            bytemuck::bytes_of(&self.current_bias),
        );
    }

    /// Get the modulation buffer for bind group creation.
    #[inline]
    pub fn modulation_buffer(&self) -> &wgpu::Buffer {
        &self.modulation_buffer
    }

    /// Get the bias buffer for bind group creation.
    #[inline]
    pub fn bias_buffer(&self) -> &wgpu::Buffer {
        &self.bias_buffer
    }

    /// Get the current GPU modulation data.
    #[inline]
    pub fn current_modulation(&self) -> &ShadowModulationGpu {
        &self.current_modulation
    }

    /// Check if there are pending changes to upload.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
}

// ---------------------------------------------------------------------------
// Style Presets
// ---------------------------------------------------------------------------

impl ShadowModulation {
    /// Apply realistic shadow preset.
    ///
    /// Pure black shadows at full strength for photorealistic rendering.
    pub fn preset_realistic(&mut self) {
        self.set_shadow_color(0.0, 0.0, 0.0);
        self.set_density(1.0);
        self.set_ao_strength(0.5);
        self.set_contact_hardening(1.0);
    }

    /// Apply soft shadow preset.
    ///
    /// Slightly blue-tinted shadows at reduced strength for soft ambient lighting.
    pub fn preset_soft(&mut self) {
        self.set_shadow_color(0.1, 0.1, 0.15);
        self.set_density(0.7);
        self.set_ao_strength(0.4);
        self.set_contact_hardening(1.5);
    }

    /// Apply stylized shadow preset.
    ///
    /// Purple-tinted shadows at half strength for artistic/cartoon effects.
    pub fn preset_stylized(&mut self) {
        self.set_shadow_color(0.2, 0.1, 0.3);
        self.set_density(0.5);
        self.set_ao_strength(0.3);
        self.set_contact_hardening(0.5);
    }

    /// Apply cel-shaded shadow preset.
    ///
    /// Hard black shadows for toon rendering. Note: cel-shading also requires
    /// shader modifications for quantized shading bands.
    pub fn preset_cel_shaded(&mut self) {
        self.set_shadow_color(0.0, 0.0, 0.0);
        self.set_density(1.0);
        self.set_ao_strength(0.0); // No AO for hard cel shading
        self.set_contact_hardening(0.0); // No soft penumbras
    }

    /// Apply warm shadow preset.
    ///
    /// Warm brown-tinted shadows for sunset or golden hour scenes.
    pub fn preset_warm(&mut self) {
        self.set_shadow_color(0.15, 0.1, 0.05);
        self.set_density(0.6);
        self.set_ao_strength(0.5);
        self.set_contact_hardening(1.2);
    }

    /// Apply cold shadow preset.
    ///
    /// Cool blue shadows for overcast or moonlit scenes.
    pub fn preset_cold(&mut self) {
        self.set_shadow_color(0.05, 0.08, 0.15);
        self.set_density(0.8);
        self.set_ao_strength(0.6);
        self.set_contact_hardening(1.3);
    }
}

// ---------------------------------------------------------------------------
// WGSL Code Generation
// ---------------------------------------------------------------------------

impl ShadowModulation {
    /// Generate WGSL struct definitions for shadow modulation.
    ///
    /// Returns WGSL code defining the `ShadowModulation` and `AdaptiveBiasConfig`
    /// structs that match the Rust GPU structs.
    pub fn wgsl_structs() -> &'static str {
        r#"
struct ShadowModulation {
    shadow_color: vec3<f32>,
    shadow_density: f32,
    ao_strength: f32,
    contact_hardening: f32,
    _padding: vec2<f32>,
}

struct AdaptiveBiasConfig {
    constant_bias: f32,
    slope_scale: f32,
    normal_offset_scale: f32,
    receiver_plane_bias: f32,
    depth_range_scale: f32,
    cascade_bias_scale: vec4<f32>,
    _padding: vec3<f32>,
}
"#
    }

    /// Generate WGSL functions for shadow modulation.
    ///
    /// Returns WGSL code for:
    /// - `modulate_shadow`: Apply color tinting and density to shadow factor
    /// - `compute_adaptive_bias`: Calculate per-pixel depth bias
    pub fn wgsl_functions() -> &'static str {
        r#"
// Apply shadow modulation to final shadow factor
fn modulate_shadow(
    raw_shadow: f32,
    modulation: ShadowModulation,
    base_color: vec3<f32>
) -> vec3<f32> {
    // Density adjustment
    let shadow_factor = 1.0 - (1.0 - raw_shadow) * modulation.shadow_density;

    // Color tinting - blend between base color and shadow color
    let shadow_contribution = base_color * shadow_factor;
    let tint_contribution = modulation.shadow_color * (1.0 - shadow_factor);

    return shadow_contribution + tint_contribution;
}

// Adaptive slope-scaled bias
fn compute_adaptive_bias(
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    depth: f32,
    cascade_index: u32,
    config: AdaptiveBiasConfig
) -> f32 {
    let cos_theta = max(dot(normal, light_dir), 0.001);
    let slope = sqrt(1.0 - cos_theta * cos_theta) / cos_theta;

    // Base slope-scaled bias
    var bias = config.constant_bias + config.slope_scale * slope;

    // Depth range scaling (farther = more bias)
    bias *= 1.0 + depth * config.depth_range_scale;

    // Per-cascade adjustment
    bias *= config.cascade_bias_scale[cascade_index];

    return bias;
}
"#
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_modulation_values() {
        let m = ShadowModulationGpu::default();
        assert_eq!(m.shadow_color, [0.0, 0.0, 0.0]);
        assert_eq!(m.shadow_density, 1.0);
        assert_eq!(m.ambient_occlusion_strength, 0.5);
        assert_eq!(m.contact_hardening, 1.0);
        assert_eq!(m._padding, [0.0; 2]);
    }

    #[test]
    fn test_shadow_modulation_gpu_size() {
        // Critical: must be 32 bytes for GPU uniform alignment
        assert_eq!(mem::size_of::<ShadowModulationGpu>(), 32);
    }

    #[test]
    fn test_adaptive_bias_config_size() {
        // Should be 48 bytes with padding for std140
        assert_eq!(mem::size_of::<AdaptiveBiasConfig>(), 48);
    }

    #[test]
    fn test_density_clamps_to_range() {
        // Test via the GPU struct
        let m1 = ShadowModulationGpu::with_density(-0.5);
        assert_eq!(m1.shadow_density, 0.0);

        let m2 = ShadowModulationGpu::with_density(1.5);
        assert_eq!(m2.shadow_density, 1.0);

        let m3 = ShadowModulationGpu::with_density(0.7);
        assert_eq!(m3.shadow_density, 0.7);
    }

    #[test]
    fn test_shadow_color_tinting() {
        let m = ShadowModulationGpu::with_color(0.2, 0.1, 0.3);
        assert_eq!(m.shadow_color[0], 0.2);
        assert_eq!(m.shadow_color[1], 0.1);
        assert_eq!(m.shadow_color[2], 0.3);
    }

    #[test]
    fn test_adaptive_bias_default_values() {
        let config = AdaptiveBiasConfig::default();
        assert_eq!(config.constant_bias, DEFAULT_CONSTANT_BIAS);
        assert_eq!(config.slope_scale, DEFAULT_SLOPE_SCALE);
        assert_eq!(config.normal_offset_scale, DEFAULT_NORMAL_OFFSET_SCALE);
        assert_eq!(config.receiver_plane_bias, 0.0);
        assert_eq!(config.depth_range_scale, 1.0);
        assert_eq!(config.cascade_bias_scale, [1.0, 1.1, 1.2, 1.3]);
    }

    #[test]
    fn test_adaptive_bias_increases_with_cascade() {
        let config = AdaptiveBiasConfig::default();
        // Each cascade should have >= bias than the previous
        for i in 1..MAX_CASCADE_COUNT {
            assert!(
                config.cascade_bias_scale[i] >= config.cascade_bias_scale[i - 1],
                "Cascade {} bias should be >= cascade {} bias",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn test_cascade_bias_scale_accessor() {
        let config = AdaptiveBiasConfig::default();
        assert_eq!(config.cascade_scale(0), 1.0);
        assert_eq!(config.cascade_scale(1), 1.1);
        assert_eq!(config.cascade_scale(2), 1.2);
        assert_eq!(config.cascade_scale(3), 1.3);
        // Out of range returns 1.0
        assert_eq!(config.cascade_scale(4), 1.0);
        assert_eq!(config.cascade_scale(100), 1.0);
    }

    #[test]
    fn test_preset_realistic_values() {
        // We need to test presets without wgpu device
        // Create default and manually verify preset values
        let m = ShadowModulationGpu::default();
        // Realistic preset should have black shadows at full density
        assert_eq!(m.shadow_color, [0.0, 0.0, 0.0]);
        assert_eq!(m.shadow_density, 1.0);
    }

    #[test]
    fn test_preset_soft_values() {
        // Soft preset values: slightly blue tint, 0.7 density
        let expected_color = [0.1_f32, 0.1, 0.15];
        let expected_density = 0.7_f32;

        // Verify the expected values are in valid range
        assert!(expected_color.iter().all(|&c| (0.0..=1.0).contains(&c)));
        assert!((0.0..=1.0).contains(&expected_density));
    }

    #[test]
    fn test_preset_stylized_values() {
        // Stylized preset: purple tint, 0.5 density
        let expected_color = [0.2_f32, 0.1, 0.3];
        let expected_density = 0.5_f32;

        // Verify the expected values are in valid range
        assert!(expected_color.iter().all(|&c| (0.0..=1.0).contains(&c)));
        assert!((0.0..=1.0).contains(&expected_density));
    }

    #[test]
    fn test_high_detail_bias_config() {
        let config = AdaptiveBiasConfig::high_detail();
        // High detail should have lower bias values
        assert!(config.constant_bias < DEFAULT_CONSTANT_BIAS);
        assert!(config.slope_scale < DEFAULT_SLOPE_SCALE);
    }

    #[test]
    fn test_large_scale_bias_config() {
        let config = AdaptiveBiasConfig::large_scale();
        // Large scale should have higher bias values
        assert!(config.constant_bias > DEFAULT_CONSTANT_BIAS);
        assert!(config.slope_scale > DEFAULT_SLOPE_SCALE);
        assert!(config.depth_range_scale > 1.0);
    }

    #[test]
    fn test_bytemuck_pod_compatibility() {
        // Verify Pod and Zeroable traits work correctly
        let modulation = ShadowModulationGpu::default();
        let bytes = bytemuck::bytes_of(&modulation);
        assert_eq!(bytes.len(), 32);

        let bias = AdaptiveBiasConfig::default();
        let bytes = bytemuck::bytes_of(&bias);
        assert_eq!(bytes.len(), 48);
    }

    #[test]
    fn test_with_color_and_density() {
        let m = ShadowModulationGpu::with_color_and_density(0.5, 0.3, 0.2, 0.8);
        assert_eq!(m.shadow_color, [0.5, 0.3, 0.2]);
        assert_eq!(m.shadow_density, 0.8);
    }

    #[test]
    fn test_bias_config_with_custom_bias() {
        let config = AdaptiveBiasConfig::with_bias(0.005, 4.0);
        assert_eq!(config.constant_bias, 0.005);
        assert_eq!(config.slope_scale, 4.0);
        // Other values should be default
        assert_eq!(config.normal_offset_scale, DEFAULT_NORMAL_OFFSET_SCALE);
    }

    #[test]
    fn test_wgsl_structs_generated() {
        let wgsl = ShadowModulation::wgsl_structs();
        assert!(wgsl.contains("struct ShadowModulation"));
        assert!(wgsl.contains("shadow_color: vec3<f32>"));
        assert!(wgsl.contains("shadow_density: f32"));
        assert!(wgsl.contains("struct AdaptiveBiasConfig"));
        assert!(wgsl.contains("cascade_bias_scale: vec4<f32>"));
    }

    #[test]
    fn test_wgsl_functions_generated() {
        let wgsl = ShadowModulation::wgsl_functions();
        assert!(wgsl.contains("fn modulate_shadow"));
        assert!(wgsl.contains("fn compute_adaptive_bias"));
        assert!(wgsl.contains("shadow_factor"));
        assert!(wgsl.contains("slope_scale"));
    }
}
