//! Sky Rendering Pass with Bruneton LUT Atmospheric Scattering.
//!
//! This module provides sky rendering infrastructure using precomputed atmospheric
//! scattering LUTs following the Bruneton 2017 approach. It integrates with the
//! Python-side `bruneton_lut.py` which generates the actual LUT textures.
//!
//! # Overview
//!
//! The sky rendering system consists of:
//! - **SkyConfig**: Configurable parameters for sun, exposure, and ground albedo
//! - **SkyUniforms**: GPU-uploadable uniform buffer for shader access
//! - **SkyLutBindings**: Texture bindings for transmittance, sky-view, and aerial perspective LUTs
//! - **SkyRenderer**: Main API for rendering the sky dome
//!
//! # Rendering Pipeline
//!
//! 1. Update sun position based on time of day
//! 2. Bind precomputed LUT textures
//! 3. Update uniform buffer with camera and sun data
//! 4. Render fullscreen triangle with sky shader
//!
//! # Coordinate System
//!
//! - Y-up world space
//! - Sun direction is normalized vector pointing towards the sun
//! - Azimuth: 0 = North, PI/2 = East, PI = South, 3*PI/2 = West
//! - Elevation: 0 = horizon, PI/2 = zenith, -PI/2 = nadir
//!
//! # References
//!
//! - Bruneton, E., & Neyret, F. (2008). Precomputed atmospheric scattering.
//! - https://ebruneton.github.io/precomputed_atmospheric_scattering/

use bytemuck::{Pod, Zeroable};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default sun intensity in lux (clear sky noon).
pub const DEFAULT_SUN_INTENSITY: f32 = 100_000.0;

/// Default sun angular radius in radians (~0.267 degrees).
pub const DEFAULT_SUN_ANGULAR_RADIUS: f32 = 0.00465;

/// Default exposure value for HDR to LDR conversion.
pub const DEFAULT_EXPOSURE: f32 = 10.0;

/// Default ground albedo (neutral gray).
pub const DEFAULT_GROUND_ALBEDO: [f32; 3] = [0.3, 0.3, 0.3];

/// Minimum sun intensity (moonlight level).
pub const MIN_SUN_INTENSITY: f32 = 0.001;

/// Maximum sun intensity (extremely bright conditions).
pub const MAX_SUN_INTENSITY: f32 = 150_000.0;

/// Minimum exposure value.
pub const MIN_EXPOSURE: f32 = 0.001;

/// Maximum exposure value.
pub const MAX_EXPOSURE: f32 = 1000.0;

/// Minimum sun angular radius.
pub const MIN_SUN_ANGULAR_RADIUS: f32 = 0.001;

/// Maximum sun angular radius.
pub const MAX_SUN_ANGULAR_RADIUS: f32 = 0.1;

/// Earth's axial tilt in radians (~23.44 degrees).
pub const EARTH_AXIAL_TILT: f32 = 0.4091;

/// Julian date for J2000.0 epoch.
pub const J2000_EPOCH: f64 = 2451545.0;

/// Earth's mean anomaly rate (radians per day).
pub const EARTH_MEAN_ANOMALY_RATE: f64 = 0.01720197;

// ---------------------------------------------------------------------------
// SkyConfig - Main configuration struct
// ---------------------------------------------------------------------------

/// Configuration for sky rendering.
///
/// This struct holds all configurable parameters for the sky rendering pass.
/// It is designed to be modified at runtime for dynamic time-of-day effects.
///
/// # Example
///
/// ```
/// use renderer_backend::sky_rendering::SkyConfig;
///
/// let mut config = SkyConfig::default();
/// config.update_sun_direction(0.0, 0.7); // Morning sun at 40 degrees elevation
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct SkyConfig {
    /// Normalized direction towards the sun (world space, Y-up).
    pub sun_direction: [f32; 3],

    /// Sun intensity in lux (default: 100000 for clear sky noon).
    pub sun_intensity: f32,

    /// Sun angular radius in radians (default: 0.00465).
    pub sun_angular_radius: f32,

    /// Ground albedo RGB (default: [0.3, 0.3, 0.3]).
    pub ground_albedo: [f32; 3],

    /// Exposure value for tone mapping (default: 10.0).
    pub exposure: f32,

    /// Whether to use aerial perspective for distant objects.
    /// Stored as u32 for GPU compatibility (0 = false, 1 = true).
    pub use_aerial_perspective: u32,

    /// Padding for 16-byte alignment.
    pub _padding: [u32; 2],
}

// Size assertion: 48 bytes (3 vec4s)
const _: () = assert!(std::mem::size_of::<SkyConfig>() == 48);

impl Default for SkyConfig {
    fn default() -> Self {
        Self {
            // Default sun position: slightly above horizon to the south
            sun_direction: [0.0, 0.5, -0.866025], // Azimuth=180, Elevation=30deg
            sun_intensity: DEFAULT_SUN_INTENSITY,
            sun_angular_radius: DEFAULT_SUN_ANGULAR_RADIUS,
            ground_albedo: DEFAULT_GROUND_ALBEDO,
            exposure: DEFAULT_EXPOSURE,
            use_aerial_perspective: 1,
            _padding: [0; 2],
        }
    }
}

impl SkyConfig {
    /// Create a new sky configuration with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration for midday sun (sun at zenith).
    #[inline]
    pub fn midday() -> Self {
        Self {
            sun_direction: [0.0, 1.0, 0.0],
            sun_intensity: DEFAULT_SUN_INTENSITY,
            ..Self::default()
        }
    }

    /// Create a configuration for sunrise/sunset.
    ///
    /// # Arguments
    ///
    /// * `is_sunrise` - If true, sun rises from east; if false, sets in west.
    #[inline]
    pub fn golden_hour(is_sunrise: bool) -> Self {
        let x = if is_sunrise { 1.0 } else { -1.0 };
        Self {
            sun_direction: normalize_vec3([x, 0.1, 0.0]),
            sun_intensity: 40_000.0, // Reduced intensity near horizon
            exposure: 5.0,           // Lower exposure for golden hour
            ..Self::default()
        }
    }

    /// Create a configuration for nighttime (moon).
    #[inline]
    pub fn night() -> Self {
        Self {
            sun_direction: [0.0, -0.5, 0.866025], // Moon below horizon
            sun_intensity: 0.1,                   // Moonlight
            exposure: 1.0,
            use_aerial_perspective: 0,
            ..Self::default()
        }
    }

    /// Update sun direction from azimuth and elevation angles.
    ///
    /// # Arguments
    ///
    /// * `azimuth` - Horizontal angle in radians (0 = North, PI/2 = East).
    /// * `elevation` - Vertical angle in radians (0 = horizon, PI/2 = zenith).
    #[inline]
    pub fn update_sun_direction(&mut self, azimuth: f32, elevation: f32) {
        self.sun_direction = azimuth_elevation_to_direction(azimuth, elevation);
    }

    /// Set sun intensity with clamping.
    #[inline]
    pub fn set_sun_intensity(&mut self, intensity: f32) {
        self.sun_intensity = intensity.clamp(MIN_SUN_INTENSITY, MAX_SUN_INTENSITY);
    }

    /// Set exposure with clamping.
    #[inline]
    pub fn set_exposure(&mut self, exposure: f32) {
        self.exposure = exposure.clamp(MIN_EXPOSURE, MAX_EXPOSURE);
    }

    /// Set sun angular radius with clamping.
    #[inline]
    pub fn set_sun_angular_radius(&mut self, radius: f32) {
        self.sun_angular_radius = radius.clamp(MIN_SUN_ANGULAR_RADIUS, MAX_SUN_ANGULAR_RADIUS);
    }

    /// Set ground albedo with clamping to [0, 1].
    #[inline]
    pub fn set_ground_albedo(&mut self, albedo: [f32; 3]) {
        self.ground_albedo = [
            albedo[0].clamp(0.0, 1.0),
            albedo[1].clamp(0.0, 1.0),
            albedo[2].clamp(0.0, 1.0),
        ];
    }

    /// Enable or disable aerial perspective.
    #[inline]
    pub fn set_aerial_perspective(&mut self, enabled: bool) {
        self.use_aerial_perspective = if enabled { 1 } else { 0 };
    }

    /// Check if aerial perspective is enabled.
    #[inline]
    pub fn aerial_perspective_enabled(&self) -> bool {
        self.use_aerial_perspective != 0
    }

    /// Validate configuration parameters.
    ///
    /// Returns true if all parameters are within valid ranges.
    pub fn validate(&self) -> bool {
        // Check sun direction is normalized
        let len_sq = self.sun_direction[0] * self.sun_direction[0]
            + self.sun_direction[1] * self.sun_direction[1]
            + self.sun_direction[2] * self.sun_direction[2];
        if (len_sq - 1.0).abs() > 0.01 {
            return false;
        }

        // Check intensity
        if self.sun_intensity < MIN_SUN_INTENSITY || self.sun_intensity > MAX_SUN_INTENSITY {
            return false;
        }

        // Check angular radius
        if self.sun_angular_radius < MIN_SUN_ANGULAR_RADIUS
            || self.sun_angular_radius > MAX_SUN_ANGULAR_RADIUS
        {
            return false;
        }

        // Check ground albedo
        for &a in &self.ground_albedo {
            if a < 0.0 || a > 1.0 {
                return false;
            }
        }

        // Check exposure
        if self.exposure < MIN_EXPOSURE || self.exposure > MAX_EXPOSURE {
            return false;
        }

        true
    }
}

// ---------------------------------------------------------------------------
// SkyUniforms - GPU uniform buffer
// ---------------------------------------------------------------------------

/// GPU-uploadable uniform buffer for sky rendering shaders.
///
/// This struct is designed to be uploaded to the GPU as a uniform buffer.
/// The layout matches WGSL struct alignment requirements.
///
/// # Memory Layout (176 bytes)
///
/// | Offset | Field                  | Size     |
/// |--------|------------------------|----------|
/// | 0      | view_projection_inverse| 64 bytes |
/// | 64     | camera_position        | 12 bytes |
/// | 76     | _pad0                  | 4 bytes  |
/// | 80     | sun_direction          | 12 bytes |
/// | 92     | _pad1                  | 4 bytes  |
/// | 96     | sun_disk_params        | 16 bytes |
/// | 112    | ground_albedo          | 12 bytes |
/// | 124    | exposure               | 4 bytes  |
/// | 128    | aerial_perspective     | 4 bytes  |
/// | 132    | time                   | 4 bytes  |
/// | 136    | _padding               | 8 bytes  |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct SkyUniforms {
    /// Inverse view-projection matrix for ray reconstruction.
    pub view_projection_inverse: [[f32; 4]; 4],

    /// Camera world position.
    pub camera_position: [f32; 3],
    pub _pad0: f32,

    /// Normalized sun direction (towards sun).
    pub sun_direction: [f32; 3],
    pub _pad1: f32,

    /// Sun disk parameters:
    /// x = intensity (lux)
    /// y = angular_radius (radians)
    /// z = cos(angular_radius) for limb darkening
    /// w = reserved
    pub sun_disk_params: [f32; 4],

    /// Ground albedo RGB.
    pub ground_albedo: [f32; 3],

    /// Exposure value.
    pub exposure: f32,

    /// Aerial perspective enabled (0 or 1).
    pub aerial_perspective: u32,

    /// Animation time for clouds/atmospheric effects.
    pub time: f32,

    /// Padding for 16-byte alignment.
    pub _padding: [u32; 2],
}

// Size assertion: 144 bytes (9 vec4s)
const _: () = assert!(std::mem::size_of::<SkyUniforms>() == 144);

impl Default for SkyUniforms {
    fn default() -> Self {
        Self {
            view_projection_inverse: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_position: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            sun_direction: [0.0, 0.5, -0.866025],
            _pad1: 0.0,
            sun_disk_params: [DEFAULT_SUN_INTENSITY, DEFAULT_SUN_ANGULAR_RADIUS, 0.99999, 0.0],
            ground_albedo: DEFAULT_GROUND_ALBEDO,
            exposure: DEFAULT_EXPOSURE,
            aerial_perspective: 1,
            time: 0.0,
            _padding: [0; 2],
        }
    }
}

impl SkyUniforms {
    /// Create uniforms from a SkyConfig and camera parameters.
    ///
    /// # Arguments
    ///
    /// * `config` - Sky configuration.
    /// * `view_proj_inv` - Inverse view-projection matrix.
    /// * `camera_pos` - Camera world position.
    /// * `time` - Animation time.
    pub fn from_config(
        config: &SkyConfig,
        view_proj_inv: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        time: f32,
    ) -> Self {
        let cos_angular_radius = config.sun_angular_radius.cos();
        Self {
            view_projection_inverse: view_proj_inv,
            camera_position: camera_pos,
            _pad0: 0.0,
            sun_direction: config.sun_direction,
            _pad1: 0.0,
            sun_disk_params: [
                config.sun_intensity,
                config.sun_angular_radius,
                cos_angular_radius,
                0.0,
            ],
            ground_albedo: config.ground_albedo,
            exposure: config.exposure,
            aerial_perspective: config.use_aerial_perspective,
            time,
            _padding: [0; 2],
        }
    }

    /// Update with new camera transform.
    #[inline]
    pub fn update_camera(&mut self, view_proj_inv: [[f32; 4]; 4], camera_pos: [f32; 3]) {
        self.view_projection_inverse = view_proj_inv;
        self.camera_position = camera_pos;
    }

    /// Update sun direction.
    #[inline]
    pub fn update_sun(&mut self, direction: [f32; 3], intensity: f32, angular_radius: f32) {
        self.sun_direction = direction;
        self.sun_disk_params[0] = intensity;
        self.sun_disk_params[1] = angular_radius;
        self.sun_disk_params[2] = angular_radius.cos();
    }

    /// Update animation time.
    #[inline]
    pub fn update_time(&mut self, time: f32) {
        self.time = time;
    }
}

// ---------------------------------------------------------------------------
// TextureHandle - Placeholder for texture references
// ---------------------------------------------------------------------------

/// Handle to a GPU texture resource.
///
/// This is a lightweight handle that references a texture in the GPU resource
/// system. The actual texture data is managed elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TextureHandle {
    /// Internal texture ID.
    pub id: u32,
    /// Generation counter for validation.
    pub generation: u32,
}

impl TextureHandle {
    /// Create a new texture handle.
    #[inline]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self { id, generation }
    }

    /// Create an invalid/null handle.
    #[inline]
    pub const fn null() -> Self {
        Self {
            id: u32::MAX,
            generation: 0,
        }
    }

    /// Check if this handle is valid (not null).
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.id != u32::MAX
    }
}

// ---------------------------------------------------------------------------
// SkyLutBindings - Texture binding management
// ---------------------------------------------------------------------------

/// Texture bindings for atmospheric scattering LUTs.
///
/// These textures are precomputed by `bruneton_lut.py` and uploaded to the GPU.
/// The sky shader samples these LUTs to compute atmospheric scattering.
///
/// # Texture Formats
///
/// - **Transmittance LUT**: RGBA16F, 256x64
/// - **Sky-View LUT**: RGB16F, 256x512
/// - **Aerial Perspective LUT**: RGBA16F, 32x32x32 (3D)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SkyLutBindings {
    /// Transmittance LUT texture handle.
    ///
    /// Encodes extinction along view rays at various altitudes and zenith angles.
    pub transmittance_lut: TextureHandle,

    /// Sky-view LUT texture handle.
    ///
    /// Single-scattering sky radiance for all view directions at the observer.
    pub sky_view_lut: TextureHandle,

    /// Aerial perspective LUT texture handle (3D texture).
    ///
    /// In-scattering and transmittance for distant objects.
    pub aerial_perspective_lut: TextureHandle,
}

impl SkyLutBindings {
    /// Create new LUT bindings from texture handles.
    #[inline]
    pub const fn new(
        transmittance: TextureHandle,
        sky_view: TextureHandle,
        aerial_perspective: TextureHandle,
    ) -> Self {
        Self {
            transmittance_lut: transmittance,
            sky_view_lut: sky_view,
            aerial_perspective_lut: aerial_perspective,
        }
    }

    /// Check if all required LUTs are bound.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.transmittance_lut.is_valid() && self.sky_view_lut.is_valid()
        // Aerial perspective is optional
    }

    /// Check if aerial perspective LUT is available.
    #[inline]
    pub fn has_aerial_perspective(&self) -> bool {
        self.aerial_perspective_lut.is_valid()
    }
}

// ---------------------------------------------------------------------------
// SkyRenderer - Main rendering API
// ---------------------------------------------------------------------------

/// Main sky renderer using Bruneton LUT atmospheric scattering.
///
/// This struct manages the sky rendering state and provides methods for
/// updating parameters and executing the sky rendering pass.
///
/// # Example
///
/// ```
/// use renderer_backend::sky_rendering::{SkyRenderer, SkyConfig};
///
/// let config = SkyConfig::midday();
/// let renderer = SkyRenderer::new(config);
///
/// // Update sun position for time of day
/// let mut renderer = renderer;
/// renderer.update_sun_position(1.57, 0.8); // East, 45 degrees up
///
/// // Get sky color for environment probes
/// let zenith_color = renderer.sample_sky_color([0.0, 1.0, 0.0], renderer.config().sun_direction);
/// ```
#[derive(Debug, Clone)]
pub struct SkyRenderer {
    /// Sky configuration.
    config: SkyConfig,

    /// Current uniform buffer data.
    uniforms: SkyUniforms,

    /// Bound LUT textures.
    lut_bindings: SkyLutBindings,

    /// Animation time accumulator.
    time: f32,
}

impl Default for SkyRenderer {
    fn default() -> Self {
        Self::new(SkyConfig::default())
    }
}

impl SkyRenderer {
    /// Create a new sky renderer with the given configuration.
    #[inline]
    pub fn new(config: SkyConfig) -> Self {
        let uniforms = SkyUniforms::from_config(
            &config,
            identity_matrix(),
            [0.0, 0.0, 0.0],
            0.0,
        );
        Self {
            config,
            uniforms,
            lut_bindings: SkyLutBindings::default(),
            time: 0.0,
        }
    }

    /// Get the current configuration.
    #[inline]
    pub fn config(&self) -> &SkyConfig {
        &self.config
    }

    /// Get mutable access to the configuration.
    #[inline]
    pub fn config_mut(&mut self) -> &mut SkyConfig {
        &mut self.config
    }

    /// Get the current uniform buffer data.
    #[inline]
    pub fn uniforms(&self) -> &SkyUniforms {
        &self.uniforms
    }

    /// Get the LUT bindings.
    #[inline]
    pub fn lut_bindings(&self) -> &SkyLutBindings {
        &self.lut_bindings
    }

    /// Update sun position from azimuth and elevation angles.
    ///
    /// # Arguments
    ///
    /// * `azimuth` - Horizontal angle in radians (0 = North, PI/2 = East).
    /// * `elevation` - Vertical angle in radians (0 = horizon, PI/2 = zenith).
    #[inline]
    pub fn update_sun_position(&mut self, azimuth: f32, elevation: f32) {
        self.config.update_sun_direction(azimuth, elevation);
        self.uniforms.sun_direction = self.config.sun_direction;
    }

    /// Update uniforms for rendering.
    ///
    /// # Arguments
    ///
    /// * `view_proj_inv` - Inverse view-projection matrix.
    /// * `camera_pos` - Camera world position.
    /// * `delta_time` - Time since last frame for animation.
    pub fn update_uniforms(
        &mut self,
        view_proj_inv: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        delta_time: f32,
    ) {
        self.time += delta_time;
        self.uniforms = SkyUniforms::from_config(&self.config, view_proj_inv, camera_pos, self.time);
    }

    /// Bind LUT textures for sky rendering.
    ///
    /// # Arguments
    ///
    /// * `transmittance` - Handle to transmittance LUT texture.
    /// * `sky_view` - Handle to sky-view LUT texture.
    /// * `aerial_perspective` - Handle to aerial perspective LUT texture.
    #[inline]
    pub fn bind_luts(
        &mut self,
        transmittance: TextureHandle,
        sky_view: TextureHandle,
        aerial_perspective: TextureHandle,
    ) {
        self.lut_bindings = SkyLutBindings::new(transmittance, sky_view, aerial_perspective);
    }

    /// Sample sky color for a view direction (CPU-side, for environment probes).
    ///
    /// This is a simplified approximation of the GPU shader for use in
    /// generating environment probe colors.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized view direction (world space).
    /// * `sun_dir` - Normalized sun direction (world space).
    ///
    /// # Returns
    ///
    /// Linear RGB color value (not exposed).
    pub fn sample_sky_color(&self, view_dir: [f32; 3], sun_dir: [f32; 3]) -> [f32; 3] {
        let view_dir = normalize_vec3(view_dir);
        let sun_dir = normalize_vec3(sun_dir);

        // Get view elevation
        let view_elevation = view_dir[1].asin();

        // Sun elevation
        let sun_elevation = sun_dir[1].asin();

        // Compute colors based on elevation
        let horizon_color = self.get_horizon_color(sun_elevation);
        let zenith_color = self.get_zenith_color(sun_elevation);

        // Interpolate based on view elevation
        let t = ((view_elevation / (PI / 2.0)).clamp(0.0, 1.0)).sqrt();
        let sky_color = lerp_vec3(horizon_color, zenith_color, t);

        // Add sun disk contribution
        let sun_contribution = self.get_sun_disk_contribution(view_dir, sun_dir);
        let sun_color = [
            self.config.sun_intensity * 0.00001,
            self.config.sun_intensity * 0.000009,
            self.config.sun_intensity * 0.000008,
        ];

        [
            sky_color[0] + sun_color[0] * sun_contribution,
            sky_color[1] + sun_color[1] * sun_contribution,
            sky_color[2] + sun_color[2] * sun_contribution,
        ]
    }

    /// Get sun disk contribution for a view direction.
    ///
    /// Returns a value from 0 to 1 indicating how much the view direction
    /// overlaps with the sun disk.
    ///
    /// # Arguments
    ///
    /// * `view_dir` - Normalized view direction.
    /// * `sun_dir` - Normalized sun direction.
    pub fn get_sun_disk_contribution(&self, view_dir: [f32; 3], sun_dir: [f32; 3]) -> f32 {
        let cos_angle = dot_vec3(view_dir, sun_dir);
        let cos_sun_radius = self.config.sun_angular_radius.cos();

        if cos_angle > cos_sun_radius {
            // Inside sun disk - apply limb darkening
            let t = (cos_angle - cos_sun_radius) / (1.0 - cos_sun_radius);
            let limb_darkening = 1.0 - 0.6 * (1.0 - t.sqrt());
            limb_darkening
        } else {
            0.0
        }
    }

    /// Get horizon color based on sun elevation.
    ///
    /// # Arguments
    ///
    /// * `sun_elevation` - Sun elevation in radians.
    fn get_horizon_color(&self, sun_elevation: f32) -> [f32; 3] {
        // Color transitions from blue (day) to orange/red (sunset) to dark blue (night)
        if sun_elevation > 0.1 {
            // Day - blue horizon
            [0.4, 0.6, 0.9]
        } else if sun_elevation > -0.1 {
            // Golden hour - orange/pink
            let t = (sun_elevation + 0.1) / 0.2;
            lerp_vec3([1.0, 0.4, 0.2], [0.4, 0.6, 0.9], t)
        } else {
            // Night - dark blue
            lerp_vec3(
                [0.02, 0.03, 0.08],
                [1.0, 0.4, 0.2],
                ((sun_elevation + 0.3) / 0.2).clamp(0.0, 1.0),
            )
        }
    }

    /// Get zenith color based on sun elevation.
    ///
    /// # Arguments
    ///
    /// * `sun_elevation` - Sun elevation in radians.
    fn get_zenith_color(&self, sun_elevation: f32) -> [f32; 3] {
        // Zenith is deep blue during day, dark at night
        if sun_elevation > 0.1 {
            // Day - deep blue zenith
            [0.1, 0.3, 0.8]
        } else if sun_elevation > -0.1 {
            // Twilight - purple/blue
            let t = (sun_elevation + 0.1) / 0.2;
            lerp_vec3([0.2, 0.1, 0.3], [0.1, 0.3, 0.8], t)
        } else {
            // Night - near black with slight blue
            lerp_vec3(
                [0.01, 0.01, 0.03],
                [0.2, 0.1, 0.3],
                ((sun_elevation + 0.3) / 0.2).clamp(0.0, 1.0),
            )
        }
    }

    /// Render the sky to the current render target.
    ///
    /// This would normally take a render pass or command encoder,
    /// but for CPU-side testing we just validate state.
    ///
    /// # Returns
    ///
    /// True if the sky was rendered successfully.
    pub fn render_sky(&self) -> bool {
        // Validate that we have required LUTs
        if !self.lut_bindings.is_complete() {
            return false;
        }

        // Validate configuration
        if !self.config.validate() {
            return false;
        }

        // In a real implementation, this would:
        // 1. Set the pipeline state
        // 2. Bind the uniform buffer
        // 3. Bind the LUT textures
        // 4. Draw a fullscreen triangle

        true
    }

    /// Get raw bytes for uniform buffer upload.
    #[inline]
    pub fn uniform_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(&self.uniforms)
    }
}

// ---------------------------------------------------------------------------
// Sky Shader Data - WGSL helper functions
// ---------------------------------------------------------------------------

/// Get fullscreen triangle vertex positions.
///
/// Returns 3 vertices that form a fullscreen triangle when rendered.
/// This is more efficient than a fullscreen quad (3 vertices vs 6).
///
/// The triangle extends beyond the clip space to cover the entire screen:
/// ```text
///     v0 (-1, 3)
///      |\
///      | \
///      |  \
///      |   \
///  v1 (-1, -1)----v2 (3, -1)
/// ```
#[inline]
pub fn sky_vertex_positions() -> [[f32; 3]; 3] {
    [
        [-1.0, 3.0, 0.0],  // Top-left, extends above screen
        [-1.0, -1.0, 0.0], // Bottom-left
        [3.0, -1.0, 0.0],  // Bottom-right, extends past screen
    ]
}

/// Encode a direction vector to UV coordinates using octahedral mapping.
///
/// Octahedral mapping projects the unit sphere onto an octahedron,
/// then unfolds it into a square UV space. This provides uniform
/// sampling density and efficient encoding.
///
/// # Arguments
///
/// * `dir` - Normalized direction vector.
///
/// # Returns
///
/// UV coordinates in [0, 1] range.
pub fn encode_direction_to_uv(dir: [f32; 3]) -> [f32; 2] {
    let dir = normalize_vec3(dir);

    // Project onto octahedron
    let inv_l1_norm = 1.0 / (dir[0].abs() + dir[1].abs() + dir[2].abs());
    let mut oct = [dir[0] * inv_l1_norm, dir[2] * inv_l1_norm];

    // Reflect lower hemisphere
    if dir[1] < 0.0 {
        let sign_x = if oct[0] >= 0.0 { 1.0 } else { -1.0 };
        let sign_y = if oct[1] >= 0.0 { 1.0 } else { -1.0 };
        oct = [
            (1.0 - oct[1].abs()) * sign_x,
            (1.0 - oct[0].abs()) * sign_y,
        ];
    }

    // Map from [-1, 1] to [0, 1]
    [oct[0] * 0.5 + 0.5, oct[1] * 0.5 + 0.5]
}

/// Decode UV coordinates back to a direction vector using octahedral mapping.
///
/// # Arguments
///
/// * `uv` - UV coordinates in [0, 1] range.
///
/// # Returns
///
/// Normalized direction vector.
pub fn decode_uv_to_direction(uv: [f32; 2]) -> [f32; 3] {
    // Map from [0, 1] to [-1, 1]
    let oct = [uv[0] * 2.0 - 1.0, uv[1] * 2.0 - 1.0];

    // Reconstruct direction
    let y = 1.0 - oct[0].abs() - oct[1].abs();

    let (x, z) = if y >= 0.0 {
        (oct[0], oct[1])
    } else {
        // Reflect from lower hemisphere
        let sign_x = if oct[0] >= 0.0 { 1.0 } else { -1.0 };
        let sign_z = if oct[1] >= 0.0 { 1.0 } else { -1.0 };
        (
            (1.0 - oct[1].abs()) * sign_x,
            (1.0 - oct[0].abs()) * sign_z,
        )
    };

    normalize_vec3([x, y, z])
}

// ---------------------------------------------------------------------------
// Time of Day Integration
// ---------------------------------------------------------------------------

/// Calculate sun position from geographic location and time.
///
/// Uses a simplified solar position algorithm based on the NOAA solar calculator.
///
/// # Arguments
///
/// * `latitude` - Observer latitude in degrees (-90 to 90).
/// * `longitude` - Observer longitude in degrees (-180 to 180).
/// * `julian_day` - Julian day number (e.g., 2459580 for Jan 1, 2022).
/// * `time_hours` - Time of day in hours (0.0 to 24.0, UTC).
///
/// # Returns
///
/// Normalized direction vector pointing towards the sun.
pub fn sun_position_from_time(
    latitude: f32,
    longitude: f32,
    julian_day: f64,
    time_hours: f32,
) -> [f32; 3] {
    // Convert to radians
    let lat_rad = latitude.to_radians();
    let lon_rad = longitude.to_radians();

    // Days since J2000.0
    let n = julian_day - J2000_EPOCH + (time_hours as f64 / 24.0);

    // Mean solar longitude (degrees)
    let l = (280.460 + 0.9856474 * n) % 360.0;
    let l_rad = (l as f32).to_radians();

    // Mean anomaly (radians)
    let g = ((357.528 + 0.9856003 * n) % 360.0) as f32;
    let g_rad = g.to_radians();

    // Ecliptic longitude (radians)
    let lambda = l_rad + (1.915 * g_rad.sin() + 0.020 * (2.0 * g_rad).sin()).to_radians();

    // Obliquity of ecliptic (radians)
    let epsilon = EARTH_AXIAL_TILT;

    // Sun declination
    let sin_delta = epsilon.sin() * lambda.sin();
    let delta = sin_delta.asin();

    // Equation of time (minutes)
    let b = ((360.0 / 365.0) * (n as f32 - 81.0)).to_radians();
    let eot = 9.87 * (2.0 * b).sin() - 7.53 * b.cos() - 1.5 * b.sin();

    // Solar time (hours)
    let solar_time = time_hours + (eot / 60.0) + (longitude / 15.0);

    // Hour angle (radians, 0 at solar noon)
    let hour_angle = ((solar_time - 12.0) * 15.0).to_radians();

    // Solar elevation angle
    let sin_elevation = lat_rad.sin() * delta.sin() + lat_rad.cos() * delta.cos() * hour_angle.cos();
    let elevation = sin_elevation.asin();

    // Solar azimuth angle (0 = North, clockwise)
    let cos_azimuth = (delta.sin() - lat_rad.sin() * sin_elevation) / (lat_rad.cos() * elevation.cos());
    let cos_azimuth = cos_azimuth.clamp(-1.0, 1.0);
    let mut azimuth = cos_azimuth.acos();

    if hour_angle > 0.0 {
        azimuth = 2.0 * PI - azimuth;
    }

    // Convert to direction vector
    azimuth_elevation_to_direction(azimuth, elevation)
}

/// Get horizon color interpolation factor for a given sun elevation.
///
/// Returns warm colors near sunset/sunrise, cool colors during day.
///
/// # Arguments
///
/// * `sun_elevation` - Sun elevation in radians.
///
/// # Returns
///
/// RGB color for the horizon.
#[inline]
pub fn get_horizon_color(sun_elevation: f32) -> [f32; 3] {
    if sun_elevation > 0.1 {
        // Day
        [0.4, 0.6, 0.9]
    } else if sun_elevation > -0.1 {
        // Golden hour
        let t = (sun_elevation + 0.1) / 0.2;
        lerp_vec3([1.0, 0.4, 0.2], [0.4, 0.6, 0.9], t)
    } else if sun_elevation > -0.3 {
        // Twilight
        let t = (sun_elevation + 0.3) / 0.2;
        lerp_vec3([0.02, 0.03, 0.08], [1.0, 0.4, 0.2], t)
    } else {
        // Night
        [0.02, 0.03, 0.08]
    }
}

/// Get zenith color for a given sun elevation.
///
/// # Arguments
///
/// * `sun_elevation` - Sun elevation in radians.
///
/// # Returns
///
/// RGB color for the zenith.
#[inline]
pub fn get_zenith_color(sun_elevation: f32) -> [f32; 3] {
    if sun_elevation > 0.1 {
        // Day - deep blue
        [0.1, 0.3, 0.8]
    } else if sun_elevation > -0.1 {
        // Twilight - purple
        let t = (sun_elevation + 0.1) / 0.2;
        lerp_vec3([0.2, 0.1, 0.3], [0.1, 0.3, 0.8], t)
    } else if sun_elevation > -0.3 {
        // Late twilight
        let t = (sun_elevation + 0.3) / 0.2;
        lerp_vec3([0.01, 0.01, 0.03], [0.2, 0.1, 0.3], t)
    } else {
        // Night - near black
        [0.01, 0.01, 0.03]
    }
}

// ---------------------------------------------------------------------------
// Aerial Perspective Helpers
// ---------------------------------------------------------------------------

/// Parameters for aerial perspective blending.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct AerialPerspectiveParams {
    /// Maximum distance for aerial perspective effect (meters).
    pub max_distance: f32,

    /// Density multiplier for atmospheric haze.
    pub density: f32,

    /// Scattering coefficient for Rayleigh scattering.
    pub scattering: f32,

    /// Height falloff for density.
    pub height_falloff: f32,
}

impl Default for AerialPerspectiveParams {
    fn default() -> Self {
        Self {
            max_distance: 100_000.0, // 100km
            density: 1.0,
            scattering: 1.0,
            height_falloff: 0.0001, // Density falls off with height
        }
    }
}

impl AerialPerspectiveParams {
    /// Create aerial perspective params with custom max distance.
    #[inline]
    pub fn with_distance(max_distance: f32) -> Self {
        Self {
            max_distance,
            ..Self::default()
        }
    }

    /// Calculate aerial perspective blend factor for a given distance.
    ///
    /// # Arguments
    ///
    /// * `distance` - Distance to the object in meters.
    ///
    /// # Returns
    ///
    /// Blend factor from 0 (no fog) to 1 (full fog).
    #[inline]
    pub fn blend_factor(&self, distance: f32) -> f32 {
        let t = (distance / self.max_distance).clamp(0.0, 1.0);
        1.0 - (-t * self.density * 5.0).exp()
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Convert azimuth and elevation angles to a direction vector.
///
/// # Arguments
///
/// * `azimuth` - Horizontal angle in radians (0 = North, PI/2 = East).
/// * `elevation` - Vertical angle in radians (0 = horizon, PI/2 = zenith).
///
/// # Returns
///
/// Normalized direction vector (Y-up coordinate system).
#[inline]
pub fn azimuth_elevation_to_direction(azimuth: f32, elevation: f32) -> [f32; 3] {
    let cos_elev = elevation.cos();
    normalize_vec3([
        cos_elev * azimuth.sin(),
        elevation.sin(),
        -cos_elev * azimuth.cos(), // Negative Z for North
    ])
}

/// Convert a direction vector to azimuth and elevation angles.
///
/// # Arguments
///
/// * `dir` - Normalized direction vector.
///
/// # Returns
///
/// Tuple of (azimuth, elevation) in radians.
#[inline]
pub fn direction_to_azimuth_elevation(dir: [f32; 3]) -> (f32, f32) {
    let dir = normalize_vec3(dir);
    let elevation = dir[1].asin();
    let azimuth = dir[0].atan2(-dir[2]);
    let azimuth = if azimuth < 0.0 { azimuth + 2.0 * PI } else { azimuth };
    (azimuth, elevation)
}

/// Normalize a 3D vector.
#[inline]
fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-10 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0] // Default to up if zero vector
    }
}

/// Dot product of two 3D vectors.
#[inline]
fn dot_vec3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Linear interpolation between two 3D vectors.
#[inline]
fn lerp_vec3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Create an identity 4x4 matrix.
#[inline]
fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    // =========================================================================
    // SkyConfig Tests
    // =========================================================================

    #[test]
    fn test_sky_config_default() {
        let config = SkyConfig::default();
        assert_eq!(config.sun_intensity, DEFAULT_SUN_INTENSITY);
        assert_eq!(config.sun_angular_radius, DEFAULT_SUN_ANGULAR_RADIUS);
        assert_eq!(config.exposure, DEFAULT_EXPOSURE);
        assert_eq!(config.ground_albedo, DEFAULT_GROUND_ALBEDO);
        assert!(config.validate());
    }

    #[test]
    fn test_sky_config_midday() {
        let config = SkyConfig::midday();
        assert_eq!(config.sun_direction, [0.0, 1.0, 0.0]);
        assert!(config.validate());
    }

    #[test]
    fn test_sky_config_golden_hour_sunrise() {
        let config = SkyConfig::golden_hour(true);
        assert!(config.sun_direction[0] > 0.0); // East
        assert!(config.sun_direction[1] > 0.0); // Above horizon
        assert!(config.validate());
    }

    #[test]
    fn test_sky_config_golden_hour_sunset() {
        let config = SkyConfig::golden_hour(false);
        assert!(config.sun_direction[0] < 0.0); // West
        assert!(config.validate());
    }

    #[test]
    fn test_sky_config_night() {
        let config = SkyConfig::night();
        assert!(config.sun_direction[1] < 0.0); // Below horizon
        assert_eq!(config.use_aerial_perspective, 0);
        assert!(config.validate());
    }

    #[test]
    fn test_sky_config_update_sun_direction() {
        let mut config = SkyConfig::default();

        // Test north, 45 degrees up
        config.update_sun_direction(0.0, FRAC_PI_2 / 2.0);
        assert!(config.sun_direction[1] > 0.0);
        assert!(config.sun_direction[2] < 0.0); // Negative Z for north

        // Test east, horizon
        config.update_sun_direction(FRAC_PI_2, 0.0);
        assert!((config.sun_direction[0] - 1.0).abs() < 0.01);
        assert!(config.sun_direction[1].abs() < 0.01);
    }

    #[test]
    fn test_sky_config_set_sun_intensity() {
        let mut config = SkyConfig::default();

        config.set_sun_intensity(50_000.0);
        assert_eq!(config.sun_intensity, 50_000.0);

        // Test clamping
        config.set_sun_intensity(-100.0);
        assert_eq!(config.sun_intensity, MIN_SUN_INTENSITY);

        config.set_sun_intensity(200_000.0);
        assert_eq!(config.sun_intensity, MAX_SUN_INTENSITY);
    }

    #[test]
    fn test_sky_config_set_exposure() {
        let mut config = SkyConfig::default();

        config.set_exposure(5.0);
        assert_eq!(config.exposure, 5.0);

        config.set_exposure(0.0);
        assert_eq!(config.exposure, MIN_EXPOSURE);

        config.set_exposure(2000.0);
        assert_eq!(config.exposure, MAX_EXPOSURE);
    }

    #[test]
    fn test_sky_config_set_ground_albedo() {
        let mut config = SkyConfig::default();

        config.set_ground_albedo([0.5, 0.6, 0.7]);
        assert_eq!(config.ground_albedo, [0.5, 0.6, 0.7]);

        // Test clamping
        config.set_ground_albedo([-0.1, 1.5, 0.5]);
        assert_eq!(config.ground_albedo, [0.0, 1.0, 0.5]);
    }

    #[test]
    fn test_sky_config_aerial_perspective() {
        let mut config = SkyConfig::default();
        assert!(config.aerial_perspective_enabled());

        config.set_aerial_perspective(false);
        assert!(!config.aerial_perspective_enabled());

        config.set_aerial_perspective(true);
        assert!(config.aerial_perspective_enabled());
    }

    #[test]
    fn test_sky_config_validation_invalid_direction() {
        let mut config = SkyConfig::default();
        config.sun_direction = [0.0, 0.0, 0.0]; // Zero vector
        assert!(!config.validate());

        config.sun_direction = [10.0, 0.0, 0.0]; // Not normalized
        assert!(!config.validate());
    }

    #[test]
    fn test_sky_config_validation_invalid_intensity() {
        let mut config = SkyConfig::default();
        config.sun_intensity = -1.0;
        assert!(!config.validate());
    }

    #[test]
    fn test_sky_config_validation_invalid_albedo() {
        let mut config = SkyConfig::default();
        config.ground_albedo[0] = -0.1;
        assert!(!config.validate());

        config.ground_albedo = [0.5, 1.1, 0.5];
        assert!(!config.validate());
    }

    // =========================================================================
    // SkyUniforms Tests
    // =========================================================================

    #[test]
    fn test_sky_uniforms_default() {
        let uniforms = SkyUniforms::default();
        assert_eq!(uniforms.exposure, DEFAULT_EXPOSURE);
        assert_eq!(uniforms.aerial_perspective, 1);
    }

    #[test]
    fn test_sky_uniforms_from_config() {
        let config = SkyConfig::midday();
        let view_proj = identity_matrix();
        let camera_pos = [100.0, 50.0, 200.0];
        let time = 1.5;

        let uniforms = SkyUniforms::from_config(&config, view_proj, camera_pos, time);

        assert_eq!(uniforms.camera_position, camera_pos);
        assert_eq!(uniforms.sun_direction, config.sun_direction);
        assert_eq!(uniforms.sun_disk_params[0], config.sun_intensity);
        assert_eq!(uniforms.sun_disk_params[1], config.sun_angular_radius);
        assert_eq!(uniforms.time, time);
    }

    #[test]
    fn test_sky_uniforms_update_camera() {
        let mut uniforms = SkyUniforms::default();
        let new_pos = [1.0, 2.0, 3.0];
        let new_view_proj = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        uniforms.update_camera(new_view_proj, new_pos);

        assert_eq!(uniforms.camera_position, new_pos);
        assert_eq!(uniforms.view_projection_inverse[0][0], 2.0);
    }

    #[test]
    fn test_sky_uniforms_update_sun() {
        let mut uniforms = SkyUniforms::default();
        let dir = [0.0, 1.0, 0.0];
        let intensity = 80_000.0;
        let angular_radius = 0.01;

        uniforms.update_sun(dir, intensity, angular_radius);

        assert_eq!(uniforms.sun_direction, dir);
        assert_eq!(uniforms.sun_disk_params[0], intensity);
        assert_eq!(uniforms.sun_disk_params[1], angular_radius);
        assert!((uniforms.sun_disk_params[2] - angular_radius.cos()).abs() < 1e-6);
    }

    #[test]
    fn test_sky_uniforms_size_alignment() {
        // Verify the struct is properly aligned for GPU upload
        assert_eq!(std::mem::size_of::<SkyUniforms>(), 144);
        assert_eq!(std::mem::align_of::<SkyUniforms>(), 4);
    }

    #[test]
    fn test_sky_uniforms_pod_zeroable() {
        // Test bytemuck compatibility
        let uniforms = SkyUniforms::default();
        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 144);

        let zeroed: SkyUniforms = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.exposure, 0.0);
    }

    // =========================================================================
    // TextureHandle Tests
    // =========================================================================

    #[test]
    fn test_texture_handle_new() {
        let handle = TextureHandle::new(42, 1);
        assert_eq!(handle.id, 42);
        assert_eq!(handle.generation, 1);
        assert!(handle.is_valid());
    }

    #[test]
    fn test_texture_handle_null() {
        let handle = TextureHandle::null();
        assert!(!handle.is_valid());
        assert_eq!(handle.id, u32::MAX);
    }

    #[test]
    fn test_texture_handle_equality() {
        let a = TextureHandle::new(1, 1);
        let b = TextureHandle::new(1, 1);
        let c = TextureHandle::new(1, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // =========================================================================
    // SkyLutBindings Tests
    // =========================================================================

    #[test]
    fn test_sky_lut_bindings_new() {
        let t = TextureHandle::new(0, 1);
        let s = TextureHandle::new(1, 1);
        let a = TextureHandle::new(2, 1);

        let bindings = SkyLutBindings::new(t, s, a);
        assert_eq!(bindings.transmittance_lut, t);
        assert_eq!(bindings.sky_view_lut, s);
        assert_eq!(bindings.aerial_perspective_lut, a);
    }

    #[test]
    fn test_sky_lut_bindings_is_complete() {
        // Complete bindings
        let bindings = SkyLutBindings::new(
            TextureHandle::new(0, 1),
            TextureHandle::new(1, 1),
            TextureHandle::null(),
        );
        assert!(bindings.is_complete());

        // Incomplete - missing transmittance
        let bindings = SkyLutBindings::new(
            TextureHandle::null(),
            TextureHandle::new(1, 1),
            TextureHandle::null(),
        );
        assert!(!bindings.is_complete());

        // Incomplete - missing sky view
        let bindings = SkyLutBindings::new(
            TextureHandle::new(0, 1),
            TextureHandle::null(),
            TextureHandle::null(),
        );
        assert!(!bindings.is_complete());
    }

    #[test]
    fn test_sky_lut_bindings_has_aerial_perspective() {
        let bindings = SkyLutBindings::new(
            TextureHandle::new(0, 1),
            TextureHandle::new(1, 1),
            TextureHandle::new(2, 1),
        );
        assert!(bindings.has_aerial_perspective());

        let bindings = SkyLutBindings::new(
            TextureHandle::new(0, 1),
            TextureHandle::new(1, 1),
            TextureHandle::null(),
        );
        assert!(!bindings.has_aerial_perspective());
    }

    // =========================================================================
    // SkyRenderer Tests
    // =========================================================================

    #[test]
    fn test_sky_renderer_new() {
        let config = SkyConfig::midday();
        let renderer = SkyRenderer::new(config);
        assert_eq!(renderer.config().sun_direction, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_sky_renderer_default() {
        let renderer = SkyRenderer::default();
        assert!(renderer.config().validate());
    }

    #[test]
    fn test_sky_renderer_update_sun_position() {
        let mut renderer = SkyRenderer::default();
        renderer.update_sun_position(PI / 2.0, PI / 4.0); // East, 45 degrees

        let dir = renderer.config().sun_direction;
        assert!(dir[0] > 0.5); // Eastward component
        assert!(dir[1] > 0.5); // Upward component
    }

    #[test]
    fn test_sky_renderer_update_uniforms() {
        let mut renderer = SkyRenderer::default();
        let view_proj = identity_matrix();
        let camera_pos = [10.0, 20.0, 30.0];

        renderer.update_uniforms(view_proj, camera_pos, 0.016);

        assert_eq!(renderer.uniforms().camera_position, camera_pos);
        assert!(renderer.time > 0.0);
    }

    #[test]
    fn test_sky_renderer_bind_luts() {
        let mut renderer = SkyRenderer::default();

        renderer.bind_luts(
            TextureHandle::new(0, 1),
            TextureHandle::new(1, 1),
            TextureHandle::new(2, 1),
        );

        assert!(renderer.lut_bindings().is_complete());
        assert!(renderer.lut_bindings().has_aerial_perspective());
    }

    #[test]
    fn test_sky_renderer_sample_sky_color_zenith() {
        let renderer = SkyRenderer::new(SkyConfig::midday());
        let color = renderer.sample_sky_color([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]);

        // Should be blue-ish
        assert!(color[2] > color[0]); // More blue than red
        assert!(color[2] > color[1]); // More blue than green
    }

    #[test]
    fn test_sky_renderer_sample_sky_color_horizon() {
        let renderer = SkyRenderer::new(SkyConfig::midday());
        let color = renderer.sample_sky_color([1.0, 0.0, 0.0], [0.0, 0.5, -0.866]);

        // Should be lighter/desaturated compared to zenith
        assert!(color[0] > 0.0);
        assert!(color[1] > 0.0);
        assert!(color[2] > 0.0);
    }

    #[test]
    fn test_sky_renderer_sun_disk_contribution_inside() {
        let renderer = SkyRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];
        let view_dir = [0.0, 1.0, 0.0]; // Looking directly at sun

        let contribution = renderer.get_sun_disk_contribution(view_dir, sun_dir);
        assert!(contribution > 0.0);
    }

    #[test]
    fn test_sky_renderer_sun_disk_contribution_outside() {
        let renderer = SkyRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];
        let view_dir = [1.0, 0.0, 0.0]; // Looking perpendicular

        let contribution = renderer.get_sun_disk_contribution(view_dir, sun_dir);
        assert_eq!(contribution, 0.0);
    }

    #[test]
    fn test_sky_renderer_render_sky_incomplete_luts() {
        let mut renderer = SkyRenderer::default();
        // Bind only transmittance, missing sky_view
        renderer.bind_luts(
            TextureHandle::new(0, 1),
            TextureHandle::null(),
            TextureHandle::null(),
        );
        assert!(!renderer.render_sky());
    }

    #[test]
    fn test_sky_renderer_render_sky_complete() {
        let mut renderer = SkyRenderer::default();
        renderer.bind_luts(
            TextureHandle::new(0, 1),
            TextureHandle::new(1, 1),
            TextureHandle::null(),
        );
        assert!(renderer.render_sky());
    }

    #[test]
    fn test_sky_renderer_uniform_bytes() {
        let renderer = SkyRenderer::default();
        let bytes = renderer.uniform_bytes();
        assert_eq!(bytes.len(), std::mem::size_of::<SkyUniforms>());
    }

    // =========================================================================
    // Sky Shader Data Tests
    // =========================================================================

    #[test]
    fn test_sky_vertex_positions() {
        let verts = sky_vertex_positions();
        assert_eq!(verts.len(), 3);

        // Verify positions cover screen
        assert_eq!(verts[0], [-1.0, 3.0, 0.0]);
        assert_eq!(verts[1], [-1.0, -1.0, 0.0]);
        assert_eq!(verts[2], [3.0, -1.0, 0.0]);
    }

    #[test]
    fn test_octahedral_encoding_north() {
        let dir = [0.0, 0.0, -1.0]; // North
        let uv = encode_direction_to_uv(dir);

        // Should be in center horizontally
        assert!((uv[0] - 0.5).abs() < 0.1);

        // Decode back
        let decoded = decode_uv_to_direction(uv);
        assert!((decoded[0] - dir[0]).abs() < 0.01);
        assert!((decoded[1] - dir[1]).abs() < 0.01);
        assert!((decoded[2] - dir[2]).abs() < 0.01);
    }

    #[test]
    fn test_octahedral_encoding_up() {
        let dir = [0.0, 1.0, 0.0]; // Up
        let uv = encode_direction_to_uv(dir);

        // Decode back
        let decoded = decode_uv_to_direction(uv);
        assert!((decoded[0] - dir[0]).abs() < 0.01);
        assert!((decoded[1] - dir[1]).abs() < 0.01);
        assert!((decoded[2] - dir[2]).abs() < 0.01);
    }

    #[test]
    fn test_octahedral_encoding_down() {
        let dir = [0.0, -1.0, 0.0]; // Down
        let uv = encode_direction_to_uv(dir);

        let decoded = decode_uv_to_direction(uv);
        assert!((decoded[0] - dir[0]).abs() < 0.01);
        assert!((decoded[1] - dir[1]).abs() < 0.01);
        assert!((decoded[2] - dir[2]).abs() < 0.01);
    }

    #[test]
    fn test_octahedral_encoding_diagonal() {
        let dir = normalize_vec3([1.0, 1.0, 1.0]);
        let uv = encode_direction_to_uv(dir);
        let decoded = decode_uv_to_direction(uv);

        assert!((decoded[0] - dir[0]).abs() < 0.01);
        assert!((decoded[1] - dir[1]).abs() < 0.01);
        assert!((decoded[2] - dir[2]).abs() < 0.01);
    }

    #[test]
    fn test_octahedral_encoding_roundtrip_multiple() {
        // Test many directions
        let directions = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 1.0],
            [-0.5, 0.5, -0.707],
        ];

        for dir in directions {
            let dir = normalize_vec3(dir);
            let uv = encode_direction_to_uv(dir);
            let decoded = decode_uv_to_direction(uv);

            assert!(
                (decoded[0] - dir[0]).abs() < 0.02,
                "X mismatch for {:?}",
                dir
            );
            assert!(
                (decoded[1] - dir[1]).abs() < 0.02,
                "Y mismatch for {:?}",
                dir
            );
            assert!(
                (decoded[2] - dir[2]).abs() < 0.02,
                "Z mismatch for {:?}",
                dir
            );
        }
    }

    // =========================================================================
    // Time of Day Tests
    // =========================================================================

    #[test]
    fn test_sun_position_noon() {
        // Solar noon on equator at equinox
        let pos = sun_position_from_time(0.0, 0.0, 2459270.0, 12.0);

        // Sun should be high
        assert!(pos[1] > 0.7, "Sun not high enough at noon: {:?}", pos);
    }

    #[test]
    fn test_sun_position_sunrise() {
        // Early morning
        let pos = sun_position_from_time(45.0, 0.0, 2459270.0, 6.0);

        // Sun should be near horizon (positive but low)
        assert!(pos[1] > -0.2 && pos[1] < 0.5);
    }

    #[test]
    fn test_sun_position_sunset() {
        // Evening
        let pos = sun_position_from_time(45.0, 0.0, 2459270.0, 18.0);

        // Sun should be near horizon
        assert!(pos[1] > -0.2 && pos[1] < 0.5);
    }

    #[test]
    fn test_sun_position_midnight() {
        // Midnight
        let pos = sun_position_from_time(45.0, 0.0, 2459270.0, 0.0);

        // Sun should be below horizon
        assert!(pos[1] < 0.0);
    }

    #[test]
    fn test_get_horizon_color_day() {
        let color = get_horizon_color(0.5); // High sun
        assert!(color[2] > color[0]); // Blue tint
    }

    #[test]
    fn test_get_horizon_color_sunset() {
        let color = get_horizon_color(0.0); // At horizon
        assert!(color[0] > color[2]); // Orange/red tint
    }

    #[test]
    fn test_get_horizon_color_night() {
        let color = get_horizon_color(-0.5); // Below horizon
        assert!(color[0] < 0.1);
        assert!(color[1] < 0.1);
        assert!(color[2] < 0.1);
    }

    #[test]
    fn test_get_zenith_color_day() {
        let color = get_zenith_color(0.5);
        assert!(color[2] > 0.5); // Deep blue
    }

    #[test]
    fn test_get_zenith_color_twilight() {
        let color = get_zenith_color(0.0);
        // Should be transitioning
        assert!(color[2] > color[0]);
    }

    #[test]
    fn test_get_zenith_color_night() {
        let color = get_zenith_color(-0.5);
        // Should be very dark
        assert!(color[0] < 0.05);
        assert!(color[1] < 0.05);
        assert!(color[2] < 0.05);
    }

    // =========================================================================
    // Aerial Perspective Tests
    // =========================================================================

    #[test]
    fn test_aerial_perspective_params_default() {
        let params = AerialPerspectiveParams::default();
        assert_eq!(params.max_distance, 100_000.0);
        assert_eq!(params.density, 1.0);
    }

    #[test]
    fn test_aerial_perspective_blend_factor_zero() {
        let params = AerialPerspectiveParams::default();
        let blend = params.blend_factor(0.0);
        assert!(blend < 0.01);
    }

    #[test]
    fn test_aerial_perspective_blend_factor_max() {
        let params = AerialPerspectiveParams::default();
        let blend = params.blend_factor(params.max_distance);
        assert!(blend > 0.95);
    }

    #[test]
    fn test_aerial_perspective_blend_factor_half() {
        let params = AerialPerspectiveParams::default();
        let blend = params.blend_factor(params.max_distance / 2.0);
        // At t=0.5, blend = 1 - exp(-0.5 * 5) = 1 - exp(-2.5) ~= 0.918
        assert!(blend > 0.8 && blend < 0.99, "blend was {}", blend);
    }

    #[test]
    fn test_aerial_perspective_with_distance() {
        let params = AerialPerspectiveParams::with_distance(50_000.0);
        assert_eq!(params.max_distance, 50_000.0);
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_azimuth_elevation_to_direction_north() {
        let dir = azimuth_elevation_to_direction(0.0, 0.0);
        assert!((dir[2] - (-1.0)).abs() < 0.01); // Negative Z
        assert!(dir[1].abs() < 0.01); // Horizon
    }

    #[test]
    fn test_azimuth_elevation_to_direction_east() {
        let dir = azimuth_elevation_to_direction(FRAC_PI_2, 0.0);
        assert!((dir[0] - 1.0).abs() < 0.01); // Positive X
        assert!(dir[1].abs() < 0.01); // Horizon
    }

    #[test]
    fn test_azimuth_elevation_to_direction_zenith() {
        let dir = azimuth_elevation_to_direction(0.0, FRAC_PI_2);
        assert!((dir[1] - 1.0).abs() < 0.01); // Up
        assert!(dir[0].abs() < 0.01);
        assert!(dir[2].abs() < 0.01);
    }

    #[test]
    fn test_direction_to_azimuth_elevation_roundtrip() {
        let test_cases = [
            (0.0, 0.0),           // North, horizon
            (FRAC_PI_2, 0.0),     // East, horizon
            (PI, 0.0),            // South, horizon
            (0.0, 0.5),           // North, elevated
            (1.2, 0.7),           // Random direction
        ];

        for (az, el) in test_cases {
            let dir = azimuth_elevation_to_direction(az, el);
            let (az2, el2) = direction_to_azimuth_elevation(dir);

            // Handle azimuth wrap-around
            let az_diff = (az - az2).abs();
            let az_diff = az_diff.min((2.0 * PI - az_diff).abs());

            assert!(az_diff < 0.01, "Azimuth mismatch: {} vs {}", az, az2);
            assert!((el - el2).abs() < 0.01, "Elevation mismatch: {} vs {}", el, el2);
        }
    }

    #[test]
    fn test_normalize_vec3() {
        let v = normalize_vec3([3.0, 4.0, 0.0]);
        assert!((v[0] - 0.6).abs() < 0.01);
        assert!((v[1] - 0.8).abs() < 0.01);
        assert!(v[2].abs() < 0.01);
    }

    #[test]
    fn test_normalize_vec3_zero() {
        let v = normalize_vec3([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 1.0, 0.0]); // Default to up
    }

    #[test]
    fn test_dot_vec3() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert_eq!(dot_vec3(a, b), 0.0); // Perpendicular

        let c = [1.0, 0.0, 0.0];
        assert_eq!(dot_vec3(a, c), 1.0); // Parallel
    }

    #[test]
    fn test_lerp_vec3() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];

        let mid = lerp_vec3(a, b, 0.5);
        assert_eq!(mid, [0.5, 1.0, 1.5]);

        let start = lerp_vec3(a, b, 0.0);
        assert_eq!(start, a);

        let end = lerp_vec3(a, b, 1.0);
        assert_eq!(end, b);
    }

    #[test]
    fn test_identity_matrix() {
        let m = identity_matrix();
        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[1][1], 1.0);
        assert_eq!(m[2][2], 1.0);
        assert_eq!(m[3][3], 1.0);
        assert_eq!(m[0][1], 0.0);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_sun_disk_limb_darkening() {
        let renderer = SkyRenderer::default();
        let sun_dir = [0.0, 1.0, 0.0];

        // Center of sun disk should have higher contribution than edge
        let center_contrib = renderer.get_sun_disk_contribution([0.0, 1.0, 0.0], sun_dir);

        // Slightly off-center
        let offset_dir = normalize_vec3([0.001, 1.0, 0.0]);
        let edge_contrib = renderer.get_sun_disk_contribution(offset_dir, sun_dir);

        assert!(center_contrib >= edge_contrib);
    }

    #[test]
    fn test_sky_config_angular_radius_bounds() {
        let mut config = SkyConfig::default();

        config.set_sun_angular_radius(0.0);
        assert_eq!(config.sun_angular_radius, MIN_SUN_ANGULAR_RADIUS);

        config.set_sun_angular_radius(1.0);
        assert_eq!(config.sun_angular_radius, MAX_SUN_ANGULAR_RADIUS);
    }

    #[test]
    fn test_octahedral_uv_bounds() {
        // Test that all encoded UVs are in [0, 1]
        for i in 0..100 {
            let theta = (i as f32 / 100.0) * PI;
            let phi = (i as f32 / 50.0) * 2.0 * PI;

            let dir = [
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ];

            let uv = encode_direction_to_uv(dir);
            assert!(uv[0] >= 0.0 && uv[0] <= 1.0, "U out of bounds: {}", uv[0]);
            assert!(uv[1] >= 0.0 && uv[1] <= 1.0, "V out of bounds: {}", uv[1]);
        }
    }

    #[test]
    fn test_time_accumulation() {
        let mut renderer = SkyRenderer::default();
        assert_eq!(renderer.time, 0.0);

        renderer.update_uniforms(identity_matrix(), [0.0, 0.0, 0.0], 0.016);
        assert!((renderer.time - 0.016).abs() < 0.001);

        renderer.update_uniforms(identity_matrix(), [0.0, 0.0, 0.0], 0.016);
        assert!((renderer.time - 0.032).abs() < 0.001);
    }

    #[test]
    fn test_pod_compliance() {
        // Ensure all Pod types can be safely cast to bytes
        let config = SkyConfig::default();
        let _ = bytemuck::bytes_of(&config);

        let uniforms = SkyUniforms::default();
        let _ = bytemuck::bytes_of(&uniforms);

        let params = AerialPerspectiveParams::default();
        let _ = bytemuck::bytes_of(&params);
    }

    #[test]
    fn test_config_size_assertion() {
        assert_eq!(std::mem::size_of::<SkyConfig>(), 48);
    }
}
