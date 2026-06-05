//! Cube Shadow Map 2D Array Rendering module.
//!
//! Implements omnidirectional point light shadows using a 2D array texture
//! with 6 layers per light (one for each cube face).
//!
//! # Array Layout
//!
//! For N point lights, the texture array has 6*N layers:
//!
//! ```text
//! Layer Index = light_index * 6 + face_index
//!
//! Light 0: [+X, -X, +Y, -Y, +Z, -Z] -> layers 0-5
//! Light 1: [+X, -X, +Y, -Y, +Z, -Z] -> layers 6-11
//! Light N: [+X, -X, +Y, -Y, +Z, -Z] -> layers 6N to 6N+5
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let config = CubeShadowConfig {
//!     resolution: 1024,
//!     num_lights: 4,
//!     pcf_enabled: true,
//! };
//!
//! let cube_shadows = CubeShadowArray::new(&device, &config);
//!
//! // Get view for rendering light 0, face +X
//! let view = cube_shadows.get_face_view(0, CubeFace::PositiveX);
//! ```

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for cube shadow map array.
///
/// Defines resolution, number of point lights, and PCF settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CubeShadowConfig {
    /// Resolution of each cube face (e.g., 512, 1024).
    /// All faces are square.
    pub resolution: u32,

    /// Number of point lights with shadow casting enabled.
    /// The texture array will have 6 * num_lights layers.
    pub num_lights: u32,

    /// Whether percentage-closer filtering (PCF) is enabled.
    /// When true, uses comparison sampler for hardware-accelerated filtering.
    pub pcf_enabled: bool,
}

impl Default for CubeShadowConfig {
    fn default() -> Self {
        Self {
            resolution: 1024,
            num_lights: 4,
            pcf_enabled: true,
        }
    }
}

impl CubeShadowConfig {
    /// Create a new cube shadow configuration.
    ///
    /// # Parameters
    ///
    /// * `resolution` - Resolution of each cube face (should be power of 2).
    /// * `num_lights` - Number of point lights with shadows.
    /// * `pcf_enabled` - Whether to enable PCF filtering.
    pub fn new(resolution: u32, num_lights: u32, pcf_enabled: bool) -> Self {
        debug_assert!(
            resolution.is_power_of_two(),
            "resolution should be power of two"
        );
        debug_assert!(num_lights > 0, "num_lights must be at least 1");

        Self {
            resolution,
            num_lights,
            pcf_enabled,
        }
    }

    /// Calculate the total number of array layers needed.
    ///
    /// Returns 6 * num_lights (one layer per cube face per light).
    #[inline]
    pub fn total_layers(&self) -> u32 {
        6 * self.num_lights
    }

    /// Validate the configuration.
    ///
    /// Returns `true` if the configuration is valid for GPU resource creation.
    pub fn is_valid(&self) -> bool {
        self.resolution > 0
            && self.resolution <= 4096
            && self.num_lights > 0
            && self.total_layers() <= 2048 // wgpu limit
    }
}

// ---------------------------------------------------------------------------
// Cube Face Enum
// ---------------------------------------------------------------------------

/// Represents a face of a cube map.
///
/// Standard cube map face ordering compatible with OpenGL/Vulkan/WebGPU.
/// Each face corresponds to an axis direction in 3D space.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeFace {
    /// Positive X axis (+X), index 0
    PositiveX = 0,
    /// Negative X axis (-X), index 1
    NegativeX = 1,
    /// Positive Y axis (+Y), index 2
    PositiveY = 2,
    /// Negative Y axis (-Y), index 3
    NegativeY = 3,
    /// Positive Z axis (+Z), index 4
    PositiveZ = 4,
    /// Negative Z axis (-Z), index 5
    NegativeZ = 5,
}

impl CubeFace {
    /// All cube faces in order.
    pub const ALL: [CubeFace; 6] = [
        CubeFace::PositiveX,
        CubeFace::NegativeX,
        CubeFace::PositiveY,
        CubeFace::NegativeY,
        CubeFace::PositiveZ,
        CubeFace::NegativeZ,
    ];

    /// Get the face index (0-5).
    #[inline]
    pub fn index(self) -> u32 {
        self as u32
    }

    /// Create a CubeFace from an index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= 6`.
    pub fn from_index(index: u32) -> Self {
        match index {
            0 => CubeFace::PositiveX,
            1 => CubeFace::NegativeX,
            2 => CubeFace::PositiveY,
            3 => CubeFace::NegativeY,
            4 => CubeFace::PositiveZ,
            5 => CubeFace::NegativeZ,
            _ => panic!("Invalid cube face index: {}", index),
        }
    }

    /// Try to create a CubeFace from an index.
    ///
    /// Returns `None` if `index >= 6`.
    pub fn try_from_index(index: u32) -> Option<Self> {
        match index {
            0 => Some(CubeFace::PositiveX),
            1 => Some(CubeFace::NegativeX),
            2 => Some(CubeFace::PositiveY),
            3 => Some(CubeFace::NegativeY),
            4 => Some(CubeFace::PositiveZ),
            5 => Some(CubeFace::NegativeZ),
            _ => None,
        }
    }

    /// Get the direction vector for this face.
    ///
    /// Returns a unit vector pointing in the face's direction.
    pub fn direction(self) -> [f32; 3] {
        match self {
            CubeFace::PositiveX => [1.0, 0.0, 0.0],
            CubeFace::NegativeX => [-1.0, 0.0, 0.0],
            CubeFace::PositiveY => [0.0, 1.0, 0.0],
            CubeFace::NegativeY => [0.0, -1.0, 0.0],
            CubeFace::PositiveZ => [0.0, 0.0, 1.0],
            CubeFace::NegativeZ => [0.0, 0.0, -1.0],
        }
    }

    /// Get the up vector for this face's view matrix.
    ///
    /// Standard cube map up vectors compatible with standard conventions.
    pub fn up(self) -> [f32; 3] {
        match self {
            CubeFace::PositiveX => [0.0, -1.0, 0.0],
            CubeFace::NegativeX => [0.0, -1.0, 0.0],
            CubeFace::PositiveY => [0.0, 0.0, 1.0],
            CubeFace::NegativeY => [0.0, 0.0, -1.0],
            CubeFace::PositiveZ => [0.0, -1.0, 0.0],
            CubeFace::NegativeZ => [0.0, -1.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// View Matrix Generation
// ---------------------------------------------------------------------------

/// Generate a view matrix for a cube face from a given position.
///
/// Creates a look-at view matrix where the camera is at `position`,
/// looking in the direction of `face`, with the appropriate up vector.
///
/// # Parameters
///
/// * `position` - The position of the point light in world space.
/// * `face` - Which cube face to generate the view matrix for.
///
/// # Returns
///
/// A 4x4 column-major view matrix.
///
/// # Example
///
/// ```ignore
/// let pos = [0.0, 5.0, 0.0];
/// let view = cube_face_view_matrix(pos, CubeFace::PositiveX);
/// ```
pub fn cube_face_view_matrix(position: [f32; 3], face: CubeFace) -> [[f32; 4]; 4] {
    let dir = face.direction();
    let up = face.up();

    // Target = position + direction
    let target = [
        position[0] + dir[0],
        position[1] + dir[1],
        position[2] + dir[2],
    ];

    look_at(position, target, up)
}

/// Compute a look-at view matrix.
///
/// Creates a view matrix that transforms world coordinates to view space
/// where the camera is at `eye`, looking at `target`, with `up` as the
/// world up direction.
///
/// # Parameters
///
/// * `eye` - Camera position in world space.
/// * `target` - Point the camera is looking at.
/// * `up` - World up direction.
///
/// # Returns
///
/// A 4x4 column-major view matrix.
fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    // Forward vector (from target to eye, since we use right-handed coordinates)
    let f = normalize([
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    ]);

    // Right vector
    let r = normalize(cross(f, up));

    // Recompute up to ensure orthogonality
    let u = cross(r, f);

    // Build view matrix (column-major)
    // Note: We negate forward for view space convention
    [
        [r[0], u[0], -f[0], 0.0],
        [r[1], u[1], -f[1], 0.0],
        [r[2], u[2], -f[2], 0.0],
        [-dot(r, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

/// Generate a 90-degree perspective projection matrix for cube face rendering.
///
/// Creates a perspective matrix with a 90-degree FOV (required for cube maps),
/// square aspect ratio (1:1), and the specified near/far planes.
///
/// # Parameters
///
/// * `near` - Near clip plane distance (must be > 0).
/// * `far` - Far clip plane distance (must be > near).
///
/// # Returns
///
/// A 4x4 column-major projection matrix.
pub fn cube_face_projection_matrix(near: f32, far: f32) -> [[f32; 4]; 4] {
    debug_assert!(near > 0.0, "near must be positive");
    debug_assert!(far > near, "far must be greater than near");

    // FOV = 90 degrees, aspect = 1.0
    // tan(45 degrees) = 1.0
    let f = 1.0; // 1.0 / tan(fov / 2)

    // Using reverse-Z for better depth precision
    [
        [f, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, near / (far - near), -1.0],
        [0.0, 0.0, (near * far) / (far - near), 0.0],
    ]
}

/// Generate combined view-projection matrix for a cube face.
///
/// # Parameters
///
/// * `position` - Light position in world space.
/// * `face` - Cube face to render.
/// * `near` - Near clip plane.
/// * `far` - Far clip plane (typically the light's radius).
///
/// # Returns
///
/// A 4x4 column-major view-projection matrix.
pub fn cube_face_view_projection(
    position: [f32; 3],
    face: CubeFace,
    near: f32,
    far: f32,
) -> [[f32; 4]; 4] {
    let view = cube_face_view_matrix(position, face);
    let proj = cube_face_projection_matrix(near, far);
    mat4_mul(proj, view)
}

// ---------------------------------------------------------------------------
// Math Helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-10 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[inline]
fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = a[0][j] * b[i][0]
                + a[1][j] * b[i][1]
                + a[2][j] * b[i][2]
                + a[3][j] * b[i][3];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Cube Shadow Array
// ---------------------------------------------------------------------------

/// 2D array texture for cube shadow maps.
///
/// Stores depth textures for multiple point lights, with 6 layers per light
/// (one for each cube face). Provides per-face texture views for rendering
/// and a comparison sampler for PCF shadow sampling.
///
/// # Layout
///
/// The texture is a 2D array with `6 * num_lights` layers. Each layer
/// corresponds to one cube face of one light:
///
/// ```text
/// layer_index = light_index * 6 + face.index()
/// ```
pub struct CubeShadowArray {
    /// The 2D array depth texture.
    pub texture: wgpu::Texture,

    /// Per-face texture views for render attachment.
    /// views[layer_index] corresponds to light_index * 6 + face_index.
    pub views: Vec<wgpu::TextureView>,

    /// Full array view for shader sampling (all layers).
    pub array_view: wgpu::TextureView,

    /// Comparison sampler for hardware PCF.
    pub sampler: wgpu::Sampler,

    /// Configuration used to create this array.
    config: CubeShadowConfig,
}

impl CubeShadowArray {
    /// Create a new cube shadow array.
    ///
    /// Allocates a 2D array texture with 6 layers per light, creates
    /// per-face texture views for rendering, and a comparison sampler
    /// for PCF shadow sampling.
    ///
    /// # Parameters
    ///
    /// * `device` - The wgpu device.
    /// * `config` - Configuration specifying resolution, light count, etc.
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid.
    pub fn new(device: &wgpu::Device, config: &CubeShadowConfig) -> Self {
        assert!(config.is_valid(), "Invalid CubeShadowConfig");

        let total_layers = config.total_layers();

        // Create the 2D array depth texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cube Shadow Array"),
            size: wgpu::Extent3d {
                width: config.resolution,
                height: config.resolution,
                depth_or_array_layers: total_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create per-face views for render targets
        let views: Vec<wgpu::TextureView> = (0..total_layers)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("Cube Shadow Face View [{}]", layer)),
                    format: Some(wgpu::TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                })
            })
            .collect();

        // Create full array view for shader sampling
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Cube Shadow Array View"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(total_layers),
        });

        // Create comparison sampler for hardware PCF
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Cube Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: if config.pcf_enabled {
                Some(wgpu::CompareFunction::LessEqual)
            } else {
                None
            },
            ..Default::default()
        });

        Self {
            texture,
            views,
            array_view,
            sampler,
            config: *config,
        }
    }

    /// Get the texture view for a specific light and cube face.
    ///
    /// # Parameters
    ///
    /// * `light_index` - Index of the point light (0 to num_lights-1).
    /// * `face` - Which cube face.
    ///
    /// # Returns
    ///
    /// Reference to the texture view for that face.
    ///
    /// # Panics
    ///
    /// Panics if `light_index >= num_lights`.
    pub fn get_face_view(&self, light_index: u32, face: CubeFace) -> &wgpu::TextureView {
        let layer_index = self.layer_index(light_index, face);
        &self.views[layer_index as usize]
    }

    /// Calculate the layer index for a light and face combination.
    ///
    /// # Parameters
    ///
    /// * `light_index` - Index of the point light.
    /// * `face` - Which cube face.
    ///
    /// # Returns
    ///
    /// The layer index in the texture array.
    #[inline]
    pub fn layer_index(&self, light_index: u32, face: CubeFace) -> u32 {
        assert!(
            light_index < self.config.num_lights,
            "light_index {} out of bounds (max {})",
            light_index,
            self.config.num_lights
        );
        light_index * 6 + face.index()
    }

    /// Get all 6 face views for a specific light.
    ///
    /// # Parameters
    ///
    /// * `light_index` - Index of the point light.
    ///
    /// # Returns
    ///
    /// Array of 6 texture view references, one for each cube face.
    pub fn get_light_views(&self, light_index: u32) -> [&wgpu::TextureView; 6] {
        assert!(
            light_index < self.config.num_lights,
            "light_index {} out of bounds",
            light_index
        );

        let base = (light_index * 6) as usize;
        [
            &self.views[base],
            &self.views[base + 1],
            &self.views[base + 2],
            &self.views[base + 3],
            &self.views[base + 4],
            &self.views[base + 5],
        ]
    }

    /// Get the configuration used to create this array.
    #[inline]
    pub fn config(&self) -> &CubeShadowConfig {
        &self.config
    }

    /// Get the resolution of each cube face.
    #[inline]
    pub fn resolution(&self) -> u32 {
        self.config.resolution
    }

    /// Get the number of lights supported.
    #[inline]
    pub fn num_lights(&self) -> u32 {
        self.config.num_lights
    }
}

// ---------------------------------------------------------------------------
// Point Light Shadow Data (GPU uniform)
// ---------------------------------------------------------------------------

/// GPU uniform data for a point light's shadow parameters.
///
/// This structure is designed to be uploaded to a uniform buffer
/// for use in shadow sampling shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointLightShadowData {
    /// Light position in world space, w = light index.
    pub position_index: [f32; 4],

    /// Shadow parameters: near, far, bias, softness.
    pub params: [f32; 4],
}

impl Default for PointLightShadowData {
    fn default() -> Self {
        Self {
            position_index: [0.0, 0.0, 0.0, 0.0],
            params: [0.1, 100.0, 0.001, 1.0], // near, far, bias, softness
        }
    }
}

impl PointLightShadowData {
    /// Create shadow data for a point light.
    ///
    /// # Parameters
    ///
    /// * `position` - Light position in world space.
    /// * `light_index` - Index of the light in the shadow array.
    /// * `near` - Near clip plane.
    /// * `far` - Far clip plane (light radius).
    /// * `bias` - Depth bias for shadow acne prevention.
    pub fn new(
        position: [f32; 3],
        light_index: u32,
        near: f32,
        far: f32,
        bias: f32,
    ) -> Self {
        Self {
            position_index: [position[0], position[1], position[2], light_index as f32],
            params: [near, far, bias, 1.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- CubeShadowConfig tests -----------------------------------------------

    #[test]
    fn test_config_default() {
        let config = CubeShadowConfig::default();
        assert_eq!(config.resolution, 1024);
        assert_eq!(config.num_lights, 4);
        assert!(config.pcf_enabled);
        assert!(config.is_valid());
    }

    #[test]
    fn test_config_total_layers() {
        let config = CubeShadowConfig::new(512, 8, true);
        assert_eq!(config.total_layers(), 48); // 8 lights * 6 faces
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let valid = CubeShadowConfig::new(1024, 10, false);
        assert!(valid.is_valid());

        // Invalid: resolution too large
        let invalid_res = CubeShadowConfig {
            resolution: 8192,
            num_lights: 1,
            pcf_enabled: false,
        };
        assert!(!invalid_res.is_valid());

        // Invalid: zero lights
        let zero_lights = CubeShadowConfig {
            resolution: 512,
            num_lights: 0,
            pcf_enabled: true,
        };
        assert!(!zero_lights.is_valid());

        // Invalid: too many layers (>2048)
        let too_many = CubeShadowConfig {
            resolution: 512,
            num_lights: 500, // 500 * 6 = 3000 > 2048
            pcf_enabled: true,
        };
        assert!(!too_many.is_valid());
    }

    // -- CubeFace tests -------------------------------------------------------

    #[test]
    fn test_cube_face_indices() {
        assert_eq!(CubeFace::PositiveX.index(), 0);
        assert_eq!(CubeFace::NegativeX.index(), 1);
        assert_eq!(CubeFace::PositiveY.index(), 2);
        assert_eq!(CubeFace::NegativeY.index(), 3);
        assert_eq!(CubeFace::PositiveZ.index(), 4);
        assert_eq!(CubeFace::NegativeZ.index(), 5);
    }

    #[test]
    fn test_cube_face_from_index() {
        for i in 0..6 {
            let face = CubeFace::from_index(i);
            assert_eq!(face.index(), i);
        }
    }

    #[test]
    fn test_cube_face_try_from_index() {
        assert!(CubeFace::try_from_index(0).is_some());
        assert!(CubeFace::try_from_index(5).is_some());
        assert!(CubeFace::try_from_index(6).is_none());
        assert!(CubeFace::try_from_index(100).is_none());
    }

    #[test]
    #[should_panic(expected = "Invalid cube face index: 6")]
    fn test_cube_face_from_index_panic() {
        let _ = CubeFace::from_index(6);
    }

    #[test]
    fn test_cube_face_directions() {
        // Check all faces have unit length directions
        for face in CubeFace::ALL {
            let dir = face.direction();
            let len_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
            assert!((len_sq - 1.0).abs() < 1e-6, "Direction not unit length");
        }

        // Check specific directions
        assert_eq!(CubeFace::PositiveX.direction(), [1.0, 0.0, 0.0]);
        assert_eq!(CubeFace::NegativeX.direction(), [-1.0, 0.0, 0.0]);
        assert_eq!(CubeFace::PositiveY.direction(), [0.0, 1.0, 0.0]);
        assert_eq!(CubeFace::NegativeY.direction(), [0.0, -1.0, 0.0]);
        assert_eq!(CubeFace::PositiveZ.direction(), [0.0, 0.0, 1.0]);
        assert_eq!(CubeFace::NegativeZ.direction(), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn test_cube_face_up_vectors() {
        // Check all faces have unit length up vectors
        for face in CubeFace::ALL {
            let up = face.up();
            let len_sq = up[0] * up[0] + up[1] * up[1] + up[2] * up[2];
            assert!((len_sq - 1.0).abs() < 1e-6, "Up vector not unit length");
        }

        // Check up vectors are perpendicular to directions
        for face in CubeFace::ALL {
            let dir = face.direction();
            let up = face.up();
            let dot_product = dot(dir, up);
            assert!(
                dot_product.abs() < 1e-6,
                "Up vector not perpendicular to direction for {:?}",
                face
            );
        }
    }

    #[test]
    fn test_cube_face_all() {
        assert_eq!(CubeFace::ALL.len(), 6);
        for (i, face) in CubeFace::ALL.iter().enumerate() {
            assert_eq!(face.index(), i as u32);
        }
    }

    // -- View Matrix tests ----------------------------------------------------

    #[test]
    fn test_view_matrix_at_origin() {
        let pos = [0.0, 0.0, 0.0];
        let view = cube_face_view_matrix(pos, CubeFace::PositiveX);

        // At origin, translation part should be zero
        assert!((view[3][0]).abs() < 1e-6);
        assert!((view[3][1]).abs() < 1e-6);
        assert!((view[3][2]).abs() < 1e-6);
        assert!((view[3][3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_view_matrix_different_positions() {
        let positions = [
            [0.0, 0.0, 0.0],
            [10.0, 5.0, -3.0],
            [-100.0, 200.0, 50.0],
        ];

        for pos in positions {
            for face in CubeFace::ALL {
                let view = cube_face_view_matrix(pos, face);

                // The last column should be [tx, ty, tz, 1]
                assert!((view[3][3] - 1.0).abs() < 1e-6);

                // First three columns should have w=0
                assert!(view[0][3].abs() < 1e-6);
                assert!(view[1][3].abs() < 1e-6);
                assert!(view[2][3].abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_view_matrix_orthogonality() {
        let pos = [5.0, 10.0, -3.0];

        for face in CubeFace::ALL {
            let view = cube_face_view_matrix(pos, face);

            // Extract the 3x3 rotation part
            let r0 = [view[0][0], view[0][1], view[0][2]];
            let r1 = [view[1][0], view[1][1], view[1][2]];
            let r2 = [view[2][0], view[2][1], view[2][2]];

            // Columns should be orthonormal
            assert!(
                (dot(r0, r1)).abs() < 1e-5,
                "Columns 0 and 1 not orthogonal for {:?}",
                face
            );
            assert!(
                (dot(r0, r2)).abs() < 1e-5,
                "Columns 0 and 2 not orthogonal for {:?}",
                face
            );
            assert!(
                (dot(r1, r2)).abs() < 1e-5,
                "Columns 1 and 2 not orthogonal for {:?}",
                face
            );
        }
    }

    // -- Projection Matrix tests ----------------------------------------------

    #[test]
    fn test_projection_matrix_basic() {
        let proj = cube_face_projection_matrix(0.1, 100.0);

        // Should be symmetric for 90 degree FOV, square aspect
        assert!((proj[0][0] - proj[1][1]).abs() < 1e-6);

        // Last row should be [0, 0, -1, 0] for perspective
        assert!((proj[0][3]).abs() < 1e-6);
        assert!((proj[1][3]).abs() < 1e-6);
        assert!((proj[2][3] - (-1.0)).abs() < 1e-6);
        assert!((proj[3][3]).abs() < 1e-6);
    }

    // -- Layer Index tests ----------------------------------------------------

    #[test]
    fn test_layer_index_calculation() {
        // Light 0, face 0 -> layer 0
        assert_eq!(0 * 6 + CubeFace::PositiveX.index(), 0);

        // Light 0, face 5 -> layer 5
        assert_eq!(0 * 6 + CubeFace::NegativeZ.index(), 5);

        // Light 1, face 0 -> layer 6
        assert_eq!(1 * 6 + CubeFace::PositiveX.index(), 6);

        // Light 2, face 3 -> layer 15
        assert_eq!(2 * 6 + CubeFace::NegativeY.index(), 15);
    }

    // -- PointLightShadowData tests -------------------------------------------

    #[test]
    fn test_point_light_shadow_data_size() {
        // Should be 32 bytes (two vec4s)
        assert_eq!(std::mem::size_of::<PointLightShadowData>(), 32);
    }

    #[test]
    fn test_point_light_shadow_data_new() {
        let data = PointLightShadowData::new([1.0, 2.0, 3.0], 5, 0.1, 50.0, 0.002);

        assert_eq!(data.position_index[0], 1.0);
        assert_eq!(data.position_index[1], 2.0);
        assert_eq!(data.position_index[2], 3.0);
        assert_eq!(data.position_index[3], 5.0); // light index as float

        assert_eq!(data.params[0], 0.1);  // near
        assert_eq!(data.params[1], 50.0); // far
        assert_eq!(data.params[2], 0.002); // bias
    }

    #[test]
    fn test_point_light_shadow_data_default() {
        let data = PointLightShadowData::default();

        assert_eq!(data.position_index, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(data.params[0], 0.1);   // near
        assert_eq!(data.params[1], 100.0); // far
    }

    // -- Integration tests (logic only, no GPU) -------------------------------

    #[test]
    fn test_all_faces_have_unique_views() {
        // Verify that each light's 6 faces map to consecutive layers
        let config = CubeShadowConfig::new(512, 4, true);

        for light in 0..config.num_lights {
            let mut layers: Vec<u32> = Vec::new();
            for face in CubeFace::ALL {
                let layer = light * 6 + face.index();
                layers.push(layer);
            }

            // Should be consecutive
            for i in 0..5 {
                assert_eq!(layers[i + 1], layers[i] + 1);
            }
        }
    }

    #[test]
    fn test_view_projection_composition() {
        let pos = [10.0, 5.0, -3.0];
        let near = 0.1;
        let far = 50.0;

        for face in CubeFace::ALL {
            let vp = cube_face_view_projection(pos, face, near, far);

            // Result should be a valid 4x4 matrix
            // Just verify it's not all zeros
            let sum: f32 = vp.iter().flatten().map(|x| x.abs()).sum();
            assert!(sum > 0.0, "View-projection matrix is all zeros for {:?}", face);
        }
    }
}
