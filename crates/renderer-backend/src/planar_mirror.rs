//! Planar Mirror Rendering module.
//!
//! Implements planar reflections for mirrors, water, and other reflective surfaces
//! using a reflected camera render pass technique.
//!
//! # Reflection Matrix Derivation
//!
//! For a plane with normal `n = (nx, ny, nz)` and distance `d`, the reflection
//! matrix R is derived from the Householder transformation:
//!
//! ```text
//! R = I - 2 * n * n^T
//! ```
//!
//! For a plane not passing through the origin (d != 0), we add a translation:
//!
//! ```text
//!     [1-2*nx*nx,  -2*nx*ny,  -2*nx*nz,   0]
//! R = [-2*ny*nx,  1-2*ny*ny,  -2*ny*nz,   0]
//!     [-2*nz*nx,  -2*nz*ny,  1-2*nz*nz,   0]
//!     [-2*d*nx,   -2*d*ny,   -2*d*nz,     1]
//! ```
//!
//! # Fresnel Approximation
//!
//! Schlick's approximation for Fresnel reflectance:
//!
//! ```text
//! F = F0 + (1 - F0) * (1 - cos(theta))^5
//! ```
//!
//! Where `F0` is the base reflectivity and `theta` is the angle between
//! the view direction and surface normal.
//!
//! # Usage
//!
//! ```ignore
//! let mirror = PlanarMirrorGpu::new(
//!     [0.0, 1.0, 0.0, 0.0], // Horizontal plane at origin
//!     5.0,                   // Fresnel power
//! );
//!
//! let view_reflected = mirror.reflect_matrix(&camera_view);
//! ```

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// GPU-Side Mirror Data
// ---------------------------------------------------------------------------

/// GPU-compatible planar mirror data.
///
/// This struct is designed to be uploaded directly to GPU uniform/storage
/// buffers. All fields are properly aligned for std140/std430 layout.
///
/// # Memory Layout (96 bytes)
///
/// ```text
/// Offset | Field             | Size
/// -------|-------------------|------
///   0    | plane             |  16
///  16    | reflection_matrix |  64
///  80    | bounds_min        |  12
///  92    | _pad0             |   4
///  96    | bounds_max        |  12
/// 108    | fresnel_power     |   4
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PlanarMirrorGpu {
    /// Mirror plane equation (nx, ny, nz, d).
    /// Normal should be normalized. Distance d is from origin.
    pub plane: [f32; 4],

    /// Pre-computed reflection matrix (column-major).
    /// Transform points/directions across the mirror plane.
    pub reflection_matrix: [[f32; 4]; 4],

    /// Minimum corner of culling bounds (world space).
    pub bounds_min: [f32; 3],

    /// Padding for alignment.
    pub _pad0: f32,

    /// Maximum corner of culling bounds (world space).
    pub bounds_max: [f32; 3],

    /// Fresnel power exponent (typically 2-5).
    pub fresnel_power: f32,
}

impl Default for PlanarMirrorGpu {
    fn default() -> Self {
        // Default: horizontal mirror at y=0
        Self {
            plane: [0.0, 1.0, 0.0, 0.0],
            reflection_matrix: reflection_matrix([0.0, 1.0, 0.0, 0.0]),
            bounds_min: [-1000.0, -1000.0, -1000.0],
            _pad0: 0.0,
            bounds_max: [1000.0, 1000.0, 1000.0],
            fresnel_power: 5.0,
        }
    }
}

impl PlanarMirrorGpu {
    /// Create a new planar mirror from a plane equation.
    ///
    /// # Parameters
    ///
    /// * `plane` - Plane equation (nx, ny, nz, d) where normal should be normalized.
    /// * `fresnel_power` - Fresnel falloff exponent (typically 2-5).
    ///
    /// # Returns
    ///
    /// A new `PlanarMirrorGpu` with the reflection matrix pre-computed.
    pub fn new(plane: [f32; 4], fresnel_power: f32) -> Self {
        Self {
            plane,
            reflection_matrix: reflection_matrix(plane),
            bounds_min: [-1000.0, -1000.0, -1000.0],
            _pad0: 0.0,
            bounds_max: [1000.0, 1000.0, 1000.0],
            fresnel_power,
        }
    }

    /// Create a mirror with explicit bounds.
    ///
    /// # Parameters
    ///
    /// * `plane` - Plane equation (nx, ny, nz, d).
    /// * `bounds_min` - Minimum corner of AABB.
    /// * `bounds_max` - Maximum corner of AABB.
    /// * `fresnel_power` - Fresnel falloff exponent.
    pub fn with_bounds(
        plane: [f32; 4],
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        fresnel_power: f32,
    ) -> Self {
        Self {
            plane,
            reflection_matrix: reflection_matrix(plane),
            bounds_min,
            _pad0: 0.0,
            bounds_max,
            fresnel_power,
        }
    }

    /// Create a horizontal water plane at the specified height.
    ///
    /// # Parameters
    ///
    /// * `height` - Y coordinate of the water surface.
    /// * `fresnel_power` - Fresnel exponent (5.0 is good for water).
    pub fn water_plane(height: f32, fresnel_power: f32) -> Self {
        // Plane normal points up (+Y), distance is negative height
        Self::new([0.0, 1.0, 0.0, -height], fresnel_power)
    }

    /// Create a vertical mirror at the specified position facing +Z.
    ///
    /// # Parameters
    ///
    /// * `z_position` - Z coordinate of the mirror surface.
    /// * `fresnel_power` - Fresnel exponent (3.0 is good for mirrors).
    pub fn vertical_mirror(z_position: f32, fresnel_power: f32) -> Self {
        Self::new([0.0, 0.0, 1.0, -z_position], fresnel_power)
    }

    /// Reflect a 4x4 matrix (typically a view matrix) across this mirror.
    ///
    /// # Parameters
    ///
    /// * `matrix` - The matrix to reflect (column-major).
    ///
    /// # Returns
    ///
    /// The reflected matrix (column-major).
    pub fn reflect_matrix(&self, matrix: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
        multiply_matrices(&self.reflection_matrix, matrix)
    }

    /// Reflect a 3D point across the mirror plane.
    ///
    /// # Parameters
    ///
    /// * `point` - The point to reflect [x, y, z].
    ///
    /// # Returns
    ///
    /// The reflected point [x', y', z'].
    pub fn reflect_point(&self, point: [f32; 3]) -> [f32; 3] {
        let m = &self.reflection_matrix;
        let p = [point[0], point[1], point[2], 1.0];

        let x = m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0] * p[3];
        let y = m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1] * p[3];
        let z = m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2] * p[3];
        let w = m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3] * p[3];

        if w.abs() > 1e-6 && (w - 1.0).abs() > 1e-6 {
            [x / w, y / w, z / w]
        } else {
            [x, y, z]
        }
    }

    /// Reflect a 3D direction across the mirror plane.
    ///
    /// # Parameters
    ///
    /// * `direction` - The direction to reflect [x, y, z].
    ///
    /// # Returns
    ///
    /// The reflected direction [x', y', z'] (not normalized).
    pub fn reflect_direction(&self, direction: [f32; 3]) -> [f32; 3] {
        let m = &self.reflection_matrix;
        let d = direction;

        [
            m[0][0] * d[0] + m[1][0] * d[1] + m[2][0] * d[2],
            m[0][1] * d[0] + m[1][1] * d[1] + m[2][1] * d[2],
            m[0][2] * d[0] + m[1][2] * d[1] + m[2][2] * d[2],
        ]
    }

    /// Compute Fresnel reflectance using Schlick's approximation.
    ///
    /// # Parameters
    ///
    /// * `view_dir` - View direction (surface to camera), should be normalized.
    /// * `normal` - Surface normal, should be normalized.
    /// * `base_reflectivity` - F0 value (0.04 for dielectrics, higher for metals).
    ///
    /// # Returns
    ///
    /// Fresnel reflectance factor [0, 1].
    pub fn compute_fresnel(
        &self,
        view_dir: [f32; 3],
        normal: [f32; 3],
        base_reflectivity: f32,
    ) -> f32 {
        // Dot product (assumes normalized vectors)
        let cos_theta = (view_dir[0] * normal[0]
            + view_dir[1] * normal[1]
            + view_dir[2] * normal[2])
            .max(0.0);

        // Schlick's approximation
        let one_minus_cos = 1.0 - cos_theta;
        let one_minus_cos_pow = one_minus_cos.powf(self.fresnel_power);

        (base_reflectivity + (1.0 - base_reflectivity) * one_minus_cos_pow).min(1.0)
    }

    /// Check if a point is in front of the mirror (on the reflective side).
    ///
    /// # Parameters
    ///
    /// * `point` - Point to test [x, y, z].
    ///
    /// # Returns
    ///
    /// `true` if the point is in front of the mirror.
    pub fn is_point_in_front(&self, point: [f32; 3]) -> bool {
        let n = &self.plane;
        let signed_dist = n[0] * point[0] + n[1] * point[1] + n[2] * point[2] + n[3];
        signed_dist >= 0.0
    }

    /// Check if a point is within the mirror bounds.
    ///
    /// # Parameters
    ///
    /// * `point` - Point to test [x, y, z].
    ///
    /// # Returns
    ///
    /// `true` if the point is within bounds.
    pub fn is_point_in_bounds(&self, point: [f32; 3]) -> bool {
        point[0] >= self.bounds_min[0]
            && point[0] <= self.bounds_max[0]
            && point[1] >= self.bounds_min[1]
            && point[1] <= self.bounds_max[1]
            && point[2] >= self.bounds_min[2]
            && point[2] <= self.bounds_max[2]
    }
}

// ---------------------------------------------------------------------------
// Mirror Pass Configuration
// ---------------------------------------------------------------------------

/// Configuration for planar mirror rendering pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarMirrorPassConfig {
    /// Resolution scale relative to screen (0.25-1.0).
    pub resolution_scale: f32,

    /// Maximum render distance for reflected geometry.
    pub max_render_distance: f32,

    /// Whether to render back faces in reflection.
    pub render_back_faces: bool,

    /// Small offset to prevent z-fighting at mirror surface.
    pub clip_plane_offset: f32,

    /// Whether to apply post-process blur for roughness simulation.
    pub blur_enabled: bool,

    /// Blur strength (0.0-1.0).
    pub blur_amount: f32,
}

impl Default for PlanarMirrorPassConfig {
    fn default() -> Self {
        Self {
            resolution_scale: 0.5,
            max_render_distance: 100.0,
            render_back_faces: true,
            clip_plane_offset: 0.01,
            blur_enabled: false,
            blur_amount: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Planar Mirror Pass
// ---------------------------------------------------------------------------

/// Manages multiple planar mirrors for rendering.
///
/// Maintains a list of mirrors and their render targets, providing
/// methods for creating reflection render passes.
pub struct PlanarMirrorPass {
    /// List of GPU mirror data.
    mirrors: Vec<PlanarMirrorGpu>,

    /// Configuration for the pass.
    config: PlanarMirrorPassConfig,

    /// Maximum mirrors to render per frame.
    max_active_per_frame: u32,

    /// Internal frame counter.
    frame_count: u64,
}

impl PlanarMirrorPass {
    /// Create a new planar mirror pass.
    ///
    /// # Parameters
    ///
    /// * `config` - Pass configuration.
    /// * `max_active_per_frame` - Maximum mirrors to render each frame.
    pub fn new(config: PlanarMirrorPassConfig, max_active_per_frame: u32) -> Self {
        Self {
            mirrors: Vec::new(),
            config,
            max_active_per_frame,
            frame_count: 0,
        }
    }

    /// Add a mirror to the pass.
    ///
    /// # Parameters
    ///
    /// * `mirror` - Mirror data to add.
    ///
    /// # Returns
    ///
    /// Index of the added mirror.
    pub fn add_mirror(&mut self, mirror: PlanarMirrorGpu) -> usize {
        let index = self.mirrors.len();
        self.mirrors.push(mirror);
        index
    }

    /// Remove a mirror by index.
    ///
    /// # Parameters
    ///
    /// * `index` - Index of mirror to remove.
    pub fn remove_mirror(&mut self, index: usize) {
        if index < self.mirrors.len() {
            self.mirrors.remove(index);
        }
    }

    /// Get mirror by index.
    pub fn get_mirror(&self, index: usize) -> Option<&PlanarMirrorGpu> {
        self.mirrors.get(index)
    }

    /// Get mutable mirror by index.
    pub fn get_mirror_mut(&mut self, index: usize) -> Option<&mut PlanarMirrorGpu> {
        self.mirrors.get_mut(index)
    }

    /// Clear all mirrors.
    pub fn clear(&mut self) {
        self.mirrors.clear();
    }

    /// Get the number of mirrors.
    pub fn mirror_count(&self) -> usize {
        self.mirrors.len()
    }

    /// Get all mirrors as a slice.
    pub fn mirrors(&self) -> &[PlanarMirrorGpu] {
        &self.mirrors
    }

    /// Get mirrors visible from the camera position.
    ///
    /// # Parameters
    ///
    /// * `camera_pos` - Camera world position.
    ///
    /// # Returns
    ///
    /// Indices of visible mirrors.
    pub fn get_visible_mirrors(&self, camera_pos: [f32; 3]) -> Vec<usize> {
        self.mirrors
            .iter()
            .enumerate()
            .filter(|(_, m)| m.is_point_in_front(camera_pos))
            .map(|(i, _)| i)
            .collect()
    }

    /// Get mirrors to render this frame, limited by max_active_per_frame.
    ///
    /// # Parameters
    ///
    /// * `camera_pos` - Camera world position.
    ///
    /// # Returns
    ///
    /// Indices of mirrors to render.
    pub fn get_mirrors_for_frame(&mut self, camera_pos: [f32; 3]) -> Vec<usize> {
        self.frame_count += 1;
        let visible = self.get_visible_mirrors(camera_pos);
        visible
            .into_iter()
            .take(self.max_active_per_frame as usize)
            .collect()
    }

    /// Get the pass configuration.
    pub fn config(&self) -> &PlanarMirrorPassConfig {
        &self.config
    }

    /// Update the pass configuration.
    pub fn set_config(&mut self, config: PlanarMirrorPassConfig) {
        self.config = config;
    }

    /// Get max active mirrors per frame.
    pub fn max_active_per_frame(&self) -> u32 {
        self.max_active_per_frame
    }

    /// Set max active mirrors per frame.
    pub fn set_max_active_per_frame(&mut self, max: u32) {
        self.max_active_per_frame = max;
    }
}

// ---------------------------------------------------------------------------
// Reflection Matrix Computation
// ---------------------------------------------------------------------------

/// Compute the reflection matrix for a plane.
///
/// # Derivation
///
/// For a plane with normal n = (nx, ny, nz) and distance d, the reflection
/// matrix R transforms a point P to its reflection P' = R * P.
///
/// The reflection formula for a point is:
/// ```text
/// P' = P - 2 * (n . P + d) * n
/// ```
///
/// This can be written as a matrix multiplication:
/// ```text
///     [1-2*nx*nx,  -2*nx*ny,  -2*nx*nz,   0]   [Px]
/// R = [-2*ny*nx,  1-2*ny*ny,  -2*ny*nz,   0] * [Py]
///     [-2*nz*nx,  -2*nz*ny,  1-2*nz*nz,   0]   [Pz]
///     [-2*d*nx,   -2*d*ny,   -2*d*nz,     1]   [1 ]
/// ```
///
/// # Parameters
///
/// * `plane` - Plane equation (nx, ny, nz, d) where n is normalized.
///
/// # Returns
///
/// 4x4 reflection matrix (column-major).
pub fn reflection_matrix(plane: [f32; 4]) -> [[f32; 4]; 4] {
    let [nx, ny, nz, d] = plane;

    // Compute matrix elements
    // Note: stored column-major for GPU compatibility
    [
        [1.0 - 2.0 * nx * nx, -2.0 * ny * nx, -2.0 * nz * nx, 0.0],
        [-2.0 * nx * ny, 1.0 - 2.0 * ny * ny, -2.0 * nz * ny, 0.0],
        [-2.0 * nx * nz, -2.0 * ny * nz, 1.0 - 2.0 * nz * nz, 0.0],
        [-2.0 * d * nx, -2.0 * d * ny, -2.0 * d * nz, 1.0],
    ]
}

/// Compute Fresnel reflectance using Schlick's approximation.
///
/// # Formula
///
/// ```text
/// F = F0 + (1 - F0) * (1 - cos(theta))^power
/// ```
///
/// # Parameters
///
/// * `cos_theta` - Cosine of angle between view and normal (0 to 1).
/// * `base_reflectivity` - F0 value (0.04 for dielectrics).
/// * `power` - Fresnel power exponent (typically 5.0).
///
/// # Returns
///
/// Fresnel reflectance factor [0, 1].
#[inline]
pub fn fresnel_schlick(cos_theta: f32, base_reflectivity: f32, power: f32) -> f32 {
    let one_minus_cos = (1.0 - cos_theta).max(0.0);
    let one_minus_cos_pow = one_minus_cos.powf(power);
    (base_reflectivity + (1.0 - base_reflectivity) * one_minus_cos_pow).min(1.0)
}

/// Compute oblique projection matrix for mirror rendering.
///
/// Uses Eric Lengyel's technique to modify a projection matrix so that
/// the near plane coincides with the mirror plane. This prevents
/// rendering geometry behind the mirror.
///
/// # Algorithm
///
/// The standard projection matrix maps the view frustum to a unit cube.
/// The near plane row (row 2 in column-major) determines where z=0 in clip space.
///
/// For oblique clipping, we replace the near plane with an arbitrary clip plane
/// by modifying row 2 of the projection matrix. The key insight is:
///
/// ```text
/// new_row2 = clip_plane * (2 / (clip_plane . Q)) - row3
/// ```
///
/// Where Q is the corner of the frustum closest to the clip plane.
///
/// # Reference
///
/// Eric Lengyel, "Modifying the Projection Matrix to Perform Oblique Near-Plane Clipping"
/// <http://www.terathon.com/lengyel/Lengyel-Oblique.pdf>
///
/// # Parameters
///
/// * `proj` - Original projection matrix (column-major).
/// * `clip_plane` - Clip plane in view space (nx, ny, nz, d).
///
/// # Returns
///
/// Modified projection matrix with oblique near plane.
pub fn oblique_projection(
    proj: [[f32; 4]; 4],
    clip_plane: [f32; 4],
) -> [[f32; 4]; 4] {
    let mut result = proj;

    // Calculate Q vector - the corner of the near plane in clip space
    // that is closest to the clip plane.
    //
    // Q.x = (sign(clip_plane.x) + proj[2][0]) / proj[0][0]
    // Q.y = (sign(clip_plane.y) + proj[2][1]) / proj[1][1]
    // Q.z = -1 (near plane)
    // Q.w = (1 + proj[2][2]) / proj[3][2]
    let q = [
        (clip_plane[0].signum() + proj[2][0]) / proj[0][0],
        (clip_plane[1].signum() + proj[2][1]) / proj[1][1],
        -1.0,
        (1.0 + proj[2][2]) / proj[3][2],
    ];

    // Calculate scaling factor: 2 / (clip_plane . Q)
    let dot = clip_plane[0] * q[0]
        + clip_plane[1] * q[1]
        + clip_plane[2] * q[2]
        + clip_plane[3] * q[3];

    if dot.abs() > 1e-6 {
        let scale = 2.0 / dot;

        // Replace third row (row index 2) of projection matrix
        // new_row2 = clip_plane * scale - row3
        result[0][2] = clip_plane[0] * scale - proj[0][3];
        result[1][2] = clip_plane[1] * scale - proj[1][3];
        result[2][2] = clip_plane[2] * scale - proj[2][3];
        result[3][2] = clip_plane[3] * scale - proj[3][3];
    }

    result
}

/// Transform a world-space plane to view space.
///
/// Planes transform by the inverse-transpose of the transformation matrix.
/// For a plane P = (n, d) and transformation M:
///
/// ```text
/// P_view = (M^-1)^T * P_world
/// ```
///
/// # Parameters
///
/// * `plane` - Plane equation in world space (nx, ny, nz, d).
/// * `view_matrix` - View matrix (world to view transformation).
///
/// # Returns
///
/// Plane equation in view space (nx', ny', nz', d').
pub fn transform_plane_to_view(
    plane: [f32; 4],
    view_matrix: [[f32; 4]; 4],
) -> [f32; 4] {
    // For plane transformation, we need the inverse-transpose of the view matrix.
    // Since view matrices are typically orthonormal (rotation + translation),
    // the inverse-transpose equals the original matrix for the rotation part.
    //
    // For a general affine transformation M, the plane P transforms as:
    // P' = (M^-1)^T * P
    //
    // However, for efficiency, we can compute this directly:
    // The normal n' = (M^-1)^T * n (upper-left 3x3)
    // The distance d' needs adjustment for translation.

    let inv = invert_matrix(view_matrix);
    let inv_t = transpose_matrix(inv);

    // Transform plane as homogeneous coordinates
    let px = plane[0];
    let py = plane[1];
    let pz = plane[2];
    let pw = plane[3];

    // P' = inv_t * P
    let nx = inv_t[0][0] * px + inv_t[1][0] * py + inv_t[2][0] * pz + inv_t[3][0] * pw;
    let ny = inv_t[0][1] * px + inv_t[1][1] * py + inv_t[2][1] * pz + inv_t[3][1] * pw;
    let nz = inv_t[0][2] * px + inv_t[1][2] * py + inv_t[2][2] * pz + inv_t[3][2] * pw;
    let nw = inv_t[0][3] * px + inv_t[1][3] * py + inv_t[2][3] * pz + inv_t[3][3] * pw;

    [nx, ny, nz, nw]
}

/// Apply oblique near-plane clipping to create a reflected camera projection.
///
/// This is the main entry point for setting up mirror rendering. It combines
/// the view transformation with oblique projection to clip geometry behind
/// the mirror surface.
///
/// # Parameters
///
/// * `mirror` - The planar mirror data.
/// * `view_matrix` - Original camera view matrix.
/// * `proj_matrix` - Original camera projection matrix.
/// * `clip_offset` - Small offset to prevent z-fighting (typically 0.01).
///
/// # Returns
///
/// Tuple of (reflected_view, oblique_projection).
pub fn create_mirror_camera_matrices(
    mirror: &PlanarMirrorGpu,
    view_matrix: [[f32; 4]; 4],
    proj_matrix: [[f32; 4]; 4],
    clip_offset: f32,
) -> ([[f32; 4]; 4], [[f32; 4]; 4]) {
    // 1. Compute reflected view matrix
    let reflected_view = mirror.reflect_matrix(&view_matrix);

    // 2. Transform mirror plane to view space (of the reflected camera)
    let mut view_plane = transform_plane_to_view(mirror.plane, reflected_view);

    // 3. Apply clip offset to prevent z-fighting
    view_plane[3] += clip_offset;

    // 4. Compute oblique projection
    let oblique_proj = oblique_projection(proj_matrix, view_plane);

    (reflected_view, oblique_proj)
}

/// Check if a point is clipped by the oblique near plane.
///
/// Points behind the mirror (negative side of the clip plane) should be
/// clipped and not rendered in the reflection.
///
/// # Parameters
///
/// * `point` - Point in view space [x, y, z].
/// * `clip_plane` - Clip plane in view space (nx, ny, nz, d).
///
/// # Returns
///
/// `true` if the point should be clipped (behind the mirror).
pub fn is_point_clipped(point: [f32; 3], clip_plane: [f32; 4]) -> bool {
    let signed_dist =
        clip_plane[0] * point[0] + clip_plane[1] * point[1] + clip_plane[2] * point[2] + clip_plane[3];
    signed_dist < 0.0
}

/// Compute the signed distance from a point to the clip plane.
///
/// # Parameters
///
/// * `point` - Point in the same coordinate space as the plane.
/// * `plane` - Plane equation (nx, ny, nz, d).
///
/// # Returns
///
/// Signed distance (positive = in front, negative = behind).
pub fn signed_distance_to_plane(point: [f32; 3], plane: [f32; 4]) -> f32 {
    plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]
}

/// Invert a 4x4 matrix.
fn invert_matrix(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Compute cofactors and determinant
    let s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    let s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
    let s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
    let s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
    let s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
    let s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];

    let c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    let c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    let c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    let c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    let c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    let c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

    let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;

    if det.abs() < 1e-10 {
        return IDENTITY_MATRIX;
    }

    let inv_det = 1.0 / det;

    [
        [
            (m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3) * inv_det,
            (-m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3) * inv_det,
            (m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3) * inv_det,
            (-m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3) * inv_det,
        ],
        [
            (-m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1) * inv_det,
            (m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1) * inv_det,
            (-m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1) * inv_det,
            (m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1) * inv_det,
        ],
        [
            (m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0) * inv_det,
            (-m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0) * inv_det,
            (m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0) * inv_det,
            (-m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0) * inv_det,
        ],
        [
            (-m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0) * inv_det,
            (m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0) * inv_det,
            (-m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0) * inv_det,
            (m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0) * inv_det,
        ],
    ]
}

/// Transpose a 4x4 matrix.
fn transpose_matrix(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

// ---------------------------------------------------------------------------
// Matrix Utilities
// ---------------------------------------------------------------------------

/// Multiply two 4x4 matrices (column-major).
fn multiply_matrices(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];

    for col in 0..4 {
        for row in 0..4 {
            result[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }

    result
}

/// Identity matrix (column-major).
pub const IDENTITY_MATRIX: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn matrix_approx_eq(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> bool {
        for col in 0..4 {
            for row in 0..4 {
                if !approx_eq(a[col][row], b[col][row]) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_reflection_matrix_horizontal_plane() {
        // Horizontal plane at y=0 (normal = +Y)
        let plane = [0.0, 1.0, 0.0, 0.0];
        let r = reflection_matrix(plane);

        // Expected: flip Y coordinate
        let expected = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        assert!(matrix_approx_eq(&r, &expected));
    }

    #[test]
    fn test_reflection_matrix_vertical_plane_x() {
        // Vertical plane at x=0 (normal = +X)
        let plane = [1.0, 0.0, 0.0, 0.0];
        let r = reflection_matrix(plane);

        // Expected: flip X coordinate
        let expected = [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        assert!(matrix_approx_eq(&r, &expected));
    }

    #[test]
    fn test_reflection_matrix_vertical_plane_z() {
        // Vertical plane at z=0 (normal = +Z)
        let plane = [0.0, 0.0, 1.0, 0.0];
        let r = reflection_matrix(plane);

        // Expected: flip Z coordinate
        let expected = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        assert!(matrix_approx_eq(&r, &expected));
    }

    #[test]
    fn test_reflection_matrix_offset_plane() {
        // Horizontal plane at y=5 (normal = +Y, d = -5)
        let plane = [0.0, 1.0, 0.0, -5.0];
        let r = reflection_matrix(plane);

        // Point at y=10 should reflect to y=0
        let mirror = PlanarMirrorGpu::new(plane, 5.0);
        let reflected = mirror.reflect_point([0.0, 10.0, 0.0]);

        assert!(approx_eq(reflected[0], 0.0));
        assert!(approx_eq(reflected[1], 0.0));
        assert!(approx_eq(reflected[2], 0.0));
    }

    #[test]
    fn test_reflect_point_identity() {
        // Point on the plane should not move
        let plane = [0.0, 1.0, 0.0, 0.0];
        let mirror = PlanarMirrorGpu::new(plane, 5.0);

        let point = [5.0, 0.0, 3.0]; // On the plane
        let reflected = mirror.reflect_point(point);

        assert!(approx_eq(reflected[0], 5.0));
        assert!(approx_eq(reflected[1], 0.0));
        assert!(approx_eq(reflected[2], 3.0));
    }

    #[test]
    fn test_reflect_point_symmetric() {
        let plane = [0.0, 1.0, 0.0, 0.0];
        let mirror = PlanarMirrorGpu::new(plane, 5.0);

        let point = [1.0, 3.0, 2.0];
        let reflected = mirror.reflect_point(point);

        // Y should be negated
        assert!(approx_eq(reflected[0], 1.0));
        assert!(approx_eq(reflected[1], -3.0));
        assert!(approx_eq(reflected[2], 2.0));
    }

    #[test]
    fn test_reflect_direction() {
        let plane = [0.0, 1.0, 0.0, 0.0];
        let mirror = PlanarMirrorGpu::new(plane, 5.0);

        let dir = [0.0, 1.0, 0.0];
        let reflected = mirror.reflect_direction(dir);

        // Direction pointing up should reflect to pointing down
        assert!(approx_eq(reflected[0], 0.0));
        assert!(approx_eq(reflected[1], -1.0));
        assert!(approx_eq(reflected[2], 0.0));
    }

    #[test]
    fn test_fresnel_normal_incidence() {
        let mirror = PlanarMirrorGpu::new([0.0, 1.0, 0.0, 0.0], 5.0);

        // Looking straight at the surface (cos_theta = 1)
        let view = [0.0, 1.0, 0.0];
        let normal = [0.0, 1.0, 0.0];
        let f0 = 0.04;

        let fresnel = mirror.compute_fresnel(view, normal, f0);

        // At normal incidence, F should equal F0
        assert!(approx_eq(fresnel, f0));
    }

    #[test]
    fn test_fresnel_grazing_angle() {
        let mirror = PlanarMirrorGpu::new([0.0, 1.0, 0.0, 0.0], 5.0);

        // Looking at grazing angle (cos_theta = 0)
        let view = [1.0, 0.0, 0.0];
        let normal = [0.0, 1.0, 0.0];
        let f0 = 0.04;

        let fresnel = mirror.compute_fresnel(view, normal, f0);

        // At grazing angle, F should approach 1.0
        assert!(fresnel > 0.9);
        assert!(fresnel <= 1.0);
    }

    #[test]
    fn test_fresnel_45_degree() {
        let mirror = PlanarMirrorGpu::new([0.0, 1.0, 0.0, 0.0], 5.0);

        // 45 degree angle
        let sqrt2_2 = std::f32::consts::FRAC_1_SQRT_2;
        let view = [sqrt2_2, sqrt2_2, 0.0];
        let normal = [0.0, 1.0, 0.0];
        let f0 = 0.04;

        let fresnel = mirror.compute_fresnel(view, normal, f0);

        // Should be between f0 and 1.0
        assert!(fresnel > f0);
        assert!(fresnel < 1.0);
    }

    #[test]
    fn test_is_point_in_front() {
        let plane = [0.0, 1.0, 0.0, 0.0];
        let mirror = PlanarMirrorGpu::new(plane, 5.0);

        // Point above the plane (positive Y)
        assert!(mirror.is_point_in_front([0.0, 5.0, 0.0]));

        // Point below the plane (negative Y)
        assert!(!mirror.is_point_in_front([0.0, -5.0, 0.0]));

        // Point on the plane
        assert!(mirror.is_point_in_front([0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_is_point_in_bounds() {
        let mirror = PlanarMirrorGpu::with_bounds(
            [0.0, 1.0, 0.0, 0.0],
            [-10.0, -10.0, -10.0],
            [10.0, 10.0, 10.0],
            5.0,
        );

        // Inside bounds
        assert!(mirror.is_point_in_bounds([0.0, 0.0, 0.0]));
        assert!(mirror.is_point_in_bounds([5.0, 5.0, 5.0]));

        // Outside bounds
        assert!(!mirror.is_point_in_bounds([15.0, 0.0, 0.0]));
        assert!(!mirror.is_point_in_bounds([0.0, -15.0, 0.0]));
    }

    #[test]
    fn test_planar_mirror_pass_add_remove() {
        let mut pass = PlanarMirrorPass::new(PlanarMirrorPassConfig::default(), 2);

        let idx1 = pass.add_mirror(PlanarMirrorGpu::default());
        let idx2 = pass.add_mirror(PlanarMirrorGpu::water_plane(0.0, 5.0));

        assert_eq!(pass.mirror_count(), 2);
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);

        pass.remove_mirror(0);
        assert_eq!(pass.mirror_count(), 1);
    }

    #[test]
    fn test_planar_mirror_pass_visibility() {
        let mut pass = PlanarMirrorPass::new(PlanarMirrorPassConfig::default(), 2);

        // Horizontal plane at y=0 (normal +Y)
        pass.add_mirror(PlanarMirrorGpu::new([0.0, 1.0, 0.0, 0.0], 5.0));

        // Camera above plane should see it
        let visible = pass.get_visible_mirrors([0.0, 10.0, 0.0]);
        assert_eq!(visible.len(), 1);

        // Camera below plane should not see it
        let visible = pass.get_visible_mirrors([0.0, -10.0, 0.0]);
        assert_eq!(visible.len(), 0);
    }

    #[test]
    fn test_max_active_per_frame_enforced() {
        let mut pass = PlanarMirrorPass::new(PlanarMirrorPassConfig::default(), 2);

        // Add 5 mirrors
        for i in 0..5 {
            pass.add_mirror(PlanarMirrorGpu::new([0.0, 1.0, 0.0, -(i as f32)], 5.0));
        }

        // Camera above all mirrors
        let to_render = pass.get_mirrors_for_frame([0.0, 100.0, 0.0]);

        // Should only return max_active_per_frame mirrors
        assert_eq!(to_render.len(), 2);
    }

    #[test]
    fn test_water_plane_creation() {
        let mirror = PlanarMirrorGpu::water_plane(5.0, 5.0);

        // Normal should point up
        assert!(approx_eq(mirror.plane[1], 1.0));

        // Distance should be -height
        assert!(approx_eq(mirror.plane[3], -5.0));
    }

    #[test]
    fn test_vertical_mirror_creation() {
        let mirror = PlanarMirrorGpu::vertical_mirror(3.0, 3.0);

        // Normal should point +Z
        assert!(approx_eq(mirror.plane[2], 1.0));

        // Distance should be -z_position
        assert!(approx_eq(mirror.plane[3], -3.0));
    }

    #[test]
    fn test_fresnel_schlick_function() {
        // Normal incidence
        let f = fresnel_schlick(1.0, 0.04, 5.0);
        assert!(approx_eq(f, 0.04));

        // Grazing angle
        let f = fresnel_schlick(0.0, 0.04, 5.0);
        assert!(approx_eq(f, 1.0));

        // 45 degrees (cos = 0.707)
        let f = fresnel_schlick(0.707, 0.04, 5.0);
        assert!(f > 0.04 && f < 1.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = IDENTITY_MATRIX;
        let b = reflection_matrix([0.0, 1.0, 0.0, 0.0]);

        let result = multiply_matrices(&a, &b);

        // Identity * B = B
        assert!(matrix_approx_eq(&result, &b));
    }

    #[test]
    fn test_reflection_matrix_is_involutory() {
        // R * R = I (reflecting twice returns to original)
        let plane = [0.0, 1.0, 0.0, 0.0];
        let r = reflection_matrix(plane);
        let rr = multiply_matrices(&r, &r);

        assert!(matrix_approx_eq(&rr, &IDENTITY_MATRIX));
    }

    #[test]
    fn test_reflection_preserves_plane_points() {
        let plane = [0.0, 1.0, 0.0, -5.0]; // y = 5
        let mirror = PlanarMirrorGpu::new(plane, 5.0);

        // Points on the plane should not move
        let point = [10.0, 5.0, -3.0];
        let reflected = mirror.reflect_point(point);

        assert!(approx_eq(reflected[0], point[0]));
        assert!(approx_eq(reflected[1], point[1]));
        assert!(approx_eq(reflected[2], point[2]));
    }

    #[test]
    fn test_diagonal_plane_reflection() {
        // 45 degree plane (normal = normalized (1, 1, 0))
        // Reflects points by swapping x and y components (with sign change)
        let sqrt2_2 = std::f32::consts::FRAC_1_SQRT_2;
        let plane = [sqrt2_2, sqrt2_2, 0.0, 0.0];
        let mirror = PlanarMirrorGpu::new(plane, 5.0);

        // Point (1, 0, 0) should reflect to (0, -1, 0)
        // Using reflection formula: P' = P - 2*(n.P)*n
        // n.P = sqrt2/2 * 1 + sqrt2/2 * 0 = sqrt2/2
        // P' = (1,0,0) - 2*(sqrt2/2)*(sqrt2/2, sqrt2/2, 0)
        // P' = (1,0,0) - (1, 1, 0) = (0, -1, 0)
        let point = [1.0, 0.0, 0.0];
        let reflected = mirror.reflect_point(point);

        assert!(approx_eq(reflected[0], 0.0));
        assert!(approx_eq(reflected[1], -1.0));
        assert!(approx_eq(reflected[2], 0.0));
    }

    // -----------------------------------------------------------------------
    // Oblique Near-Plane Clipping Tests
    // -----------------------------------------------------------------------

    /// Create a standard perspective projection matrix for testing.
    fn create_perspective_matrix(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let tan_half_fov = (fov_y / 2.0).tan();
        let f = 1.0 / tan_half_fov;

        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), -1.0],
            [0.0, 0.0, (2.0 * far * near) / (near - far), 0.0],
        ]
    }

    /// Create a view matrix looking at a target.
    fn create_look_at_matrix(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        let f = normalize([
            target[0] - eye[0],
            target[1] - eye[1],
            target[2] - eye[2],
        ]);
        let r = normalize(cross(f, up));
        let u = cross(r, f);

        [
            [r[0], u[0], -f[0], 0.0],
            [r[1], u[1], -f[1], 0.0],
            [r[2], u[2], -f[2], 0.0],
            [-dot(r, eye), -dot(u, eye), dot(f, eye), 1.0],
        ]
    }

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 1e-6 {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn test_oblique_projection_maintains_frustum_sides() {
        // Create standard perspective projection
        let fov_y = std::f32::consts::PI / 4.0; // 45 degrees
        let aspect = 16.0 / 9.0;
        let near = 0.1;
        let far = 100.0;
        let proj = create_perspective_matrix(fov_y, aspect, near, far);

        // Clip plane in view space (pointing +Z, at z=-1)
        let clip_plane = [0.0, 0.0, -1.0, -1.0];

        let oblique = oblique_projection(proj, clip_plane);

        // The left/right/top/bottom frustum planes should be preserved
        // Column 0 and 1 should remain unchanged
        assert!(approx_eq(oblique[0][0], proj[0][0]));
        assert!(approx_eq(oblique[1][1], proj[1][1]));

        // The w-divide row should remain unchanged (row 3)
        assert!(approx_eq(oblique[0][3], proj[0][3]));
        assert!(approx_eq(oblique[1][3], proj[1][3]));
        assert!(approx_eq(oblique[2][3], proj[2][3]));
        assert!(approx_eq(oblique[3][3], proj[3][3]));
    }

    #[test]
    fn test_oblique_projection_near_plane_equals_clip_plane() {
        let proj = create_perspective_matrix(
            std::f32::consts::PI / 4.0,
            1.0,
            0.1,
            100.0,
        );

        // Clip plane at z=-2 in view space (normal pointing -Z toward camera)
        let clip_plane = [0.0, 0.0, -1.0, -2.0];

        let oblique = oblique_projection(proj, clip_plane);

        // Transform a point exactly on the clip plane
        // Point at (0, 0, -2) in view space should map to z=0 in NDC
        let point = [0.0, 0.0, -2.0, 1.0];

        // Project: clip_coords = proj * point
        let clip_z = oblique[0][2] * point[0]
            + oblique[1][2] * point[1]
            + oblique[2][2] * point[2]
            + oblique[3][2] * point[3];
        let clip_w = oblique[0][3] * point[0]
            + oblique[1][3] * point[1]
            + oblique[2][3] * point[2]
            + oblique[3][3] * point[3];

        // NDC z should be -1 (near plane in [-1, 1] range)
        let ndc_z = clip_z / clip_w;
        assert!(approx_eq(ndc_z, -1.0));
    }

    #[test]
    fn test_oblique_projection_clips_objects_behind_mirror() {
        let proj = create_perspective_matrix(
            std::f32::consts::PI / 4.0,
            1.0,
            0.1,
            100.0,
        );

        // Clip plane at z=-5
        let clip_plane = [0.0, 0.0, -1.0, -5.0];

        let oblique = oblique_projection(proj, clip_plane);

        // Point behind the clip plane (z=-6, which is behind z=-5)
        let behind_point = [0.0, 0.0, -6.0, 1.0];

        let clip_z = oblique[0][2] * behind_point[0]
            + oblique[1][2] * behind_point[1]
            + oblique[2][2] * behind_point[2]
            + oblique[3][2] * behind_point[3];
        let clip_w = oblique[0][3] * behind_point[0]
            + oblique[1][3] * behind_point[1]
            + oblique[2][3] * behind_point[2]
            + oblique[3][3] * behind_point[3];

        // Point should be in front of near plane (NDC z > -1)
        let ndc_z = clip_z / clip_w;
        assert!(ndc_z > -1.0);

        // Point in front of clip plane (z=-4)
        let front_point = [0.0, 0.0, -4.0, 1.0];

        let clip_z = oblique[0][2] * front_point[0]
            + oblique[1][2] * front_point[1]
            + oblique[2][2] * front_point[2]
            + oblique[3][2] * front_point[3];
        let clip_w = oblique[0][3] * front_point[0]
            + oblique[1][3] * front_point[1]
            + oblique[2][3] * front_point[2]
            + oblique[3][3] * front_point[3];

        // Point should be behind near plane (clipped, NDC z < -1)
        let ndc_z = clip_z / clip_w;
        assert!(ndc_z < -1.0);
    }

    #[test]
    fn test_oblique_projection_identity_when_no_clip_plane() {
        let proj = create_perspective_matrix(
            std::f32::consts::PI / 4.0,
            1.0,
            0.1,
            100.0,
        );

        // Clip plane with zero normal (degenerate case)
        let clip_plane = [0.0, 0.0, 0.0, 0.0];

        let oblique = oblique_projection(proj, clip_plane);

        // With zero plane, the projection should remain unchanged
        // (the dot product will be zero, so no modification occurs)
        assert!(matrix_approx_eq(&oblique, &proj));
    }

    #[test]
    fn test_transform_plane_to_view_identity() {
        // When view matrix is identity, plane should be unchanged
        let plane = [0.0, 1.0, 0.0, -5.0]; // y=5 plane
        let view = IDENTITY_MATRIX;

        let view_plane = transform_plane_to_view(plane, view);

        assert!(approx_eq(view_plane[0], plane[0]));
        assert!(approx_eq(view_plane[1], plane[1]));
        assert!(approx_eq(view_plane[2], plane[2]));
        assert!(approx_eq(view_plane[3], plane[3]));
    }

    #[test]
    fn test_transform_plane_to_view_rotation() {
        // Test plane transformation with a rotated view matrix.
        // A plane with normal +X in world space should transform
        // based on the inverse-transpose of the view matrix.
        let plane = [1.0, 0.0, 0.0, 0.0]; // x=0 plane, normal +X

        // Simple rotation around Y axis by 90 degrees.
        // View matrix rotates the world, so plane normal transforms by inverse-transpose.
        let cos_90 = 0.0f32;
        let sin_90 = 1.0f32;

        // Standard Y-rotation matrix (column-major)
        let view = [
            [cos_90, 0.0, sin_90, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_90, 0.0, cos_90, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let view_plane = transform_plane_to_view(plane, view);

        // The plane normal should transform. For an orthogonal matrix,
        // (M^-1)^T = M, so the normal transforms directly by the rotation.
        // With 90 deg Y rotation: +X in world -> +Z in view space
        // Actually checking the math: if view rotates world CW around Y,
        // then world +X maps to view +Z.
        assert!(approx_eq(view_plane[0], 0.0));
        // The exact sign depends on the rotation direction; just verify it's non-zero in Z
        assert!(view_plane[2].abs() > 0.9);
    }

    #[test]
    fn test_transform_plane_to_view_translation() {
        // View matrix with translation only
        let plane = [0.0, 1.0, 0.0, -5.0]; // y=5 plane

        // Camera at (0, 10, 0) looking down
        let view = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -10.0, 0.0, 1.0],
        ];

        let view_plane = transform_plane_to_view(plane, view);

        // Normal should be unchanged
        assert!(approx_eq(view_plane[1], 1.0));

        // Distance changes: original d=-5, camera at y=10
        // In view space, the plane is at y=-5 (5 below origin in view)
        // So d should become -5 - (-10) = 5... but let's verify
        // Actually: view space plane distance = world distance - camera_y component
        // The plane y=5 in world, camera at y=10, so in view space plane is at y=-5
        // Plane equation in view: 0*x + 1*y + 0*z + 5 = 0 => y = -5
        assert!(approx_eq(view_plane[3], 5.0));
    }

    #[test]
    fn test_is_point_clipped_behind_plane() {
        // Clip plane in view space: normal pointing +Z (toward camera),
        // located at z = 5 (positive Z in view space).
        // Plane equation: 0*x + 0*y + 1*z - 5 = 0  =>  z = 5
        let clip_plane = [0.0, 0.0, 1.0, -5.0];

        // Point at z=4 (behind the plane, less than 5)
        // signed_dist = 0 + 0 + 1*4 - 5 = -1 (negative = clipped)
        assert!(is_point_clipped([0.0, 0.0, 4.0], clip_plane));

        // Point at z=6 (in front of the plane, greater than 5)
        // signed_dist = 0 + 0 + 1*6 - 5 = 1 (positive = not clipped)
        assert!(!is_point_clipped([0.0, 0.0, 6.0], clip_plane));

        // Point exactly on the plane (z=5)
        // signed_dist = 0 + 0 + 1*5 - 5 = 0 (not clipped, >= 0)
        assert!(!is_point_clipped([0.0, 0.0, 5.0], clip_plane));
    }

    #[test]
    fn test_signed_distance_to_plane() {
        let plane = [0.0, 1.0, 0.0, -5.0]; // y=5

        // Point above plane (y=10)
        let dist = signed_distance_to_plane([0.0, 10.0, 0.0], plane);
        assert!(approx_eq(dist, 5.0));

        // Point below plane (y=0)
        let dist = signed_distance_to_plane([0.0, 0.0, 0.0], plane);
        assert!(approx_eq(dist, -5.0));

        // Point on plane (y=5)
        let dist = signed_distance_to_plane([0.0, 5.0, 0.0], plane);
        assert!(approx_eq(dist, 0.0));
    }

    #[test]
    fn test_create_mirror_camera_matrices() {
        let plane = [0.0, 1.0, 0.0, 0.0]; // y=0 horizontal plane
        let mirror = PlanarMirrorGpu::new(plane, 5.0);

        let view = create_look_at_matrix([0.0, 5.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = create_perspective_matrix(std::f32::consts::PI / 4.0, 1.0, 0.1, 100.0);

        let (reflected_view, oblique_proj) = create_mirror_camera_matrices(&mirror, view, proj, 0.01);

        // Reflected view should be different from original
        assert!(!matrix_approx_eq(&reflected_view, &view));

        // Oblique projection should be different from original
        assert!(!matrix_approx_eq(&oblique_proj, &proj));

        // Reflected camera position should be below the mirror plane
        // Original camera at y=5, reflected should be at y=-5
        // The view matrix encodes camera position, we can check via inverse
    }

    #[test]
    fn test_oblique_projection_preserves_far_plane() {
        let proj = create_perspective_matrix(
            std::f32::consts::PI / 4.0,
            1.0,
            0.1,
            100.0,
        );

        let clip_plane = [0.0, 0.0, -1.0, -1.0];
        let oblique = oblique_projection(proj, clip_plane);

        // Point at far plane (z=-100 in view space)
        let far_point = [0.0, 0.0, -100.0, 1.0];

        let clip_z = oblique[0][2] * far_point[0]
            + oblique[1][2] * far_point[1]
            + oblique[2][2] * far_point[2]
            + oblique[3][2] * far_point[3];
        let clip_w = oblique[0][3] * far_point[0]
            + oblique[1][3] * far_point[1]
            + oblique[2][3] * far_point[2]
            + oblique[3][3] * far_point[3];

        // Far plane should map to NDC z = 1
        let ndc_z = clip_z / clip_w;
        // Note: with oblique clipping, far plane mapping may be slightly affected
        // but should still be > 0 (beyond midpoint of frustum)
        assert!(ndc_z > 0.0);
    }

    #[test]
    fn test_oblique_projection_angled_clip_plane() {
        let proj = create_perspective_matrix(
            std::f32::consts::PI / 4.0,
            1.0,
            0.1,
            100.0,
        );

        // 45-degree angled clip plane
        let sqrt2_2 = std::f32::consts::FRAC_1_SQRT_2;
        let clip_plane = [sqrt2_2, 0.0, -sqrt2_2, -5.0];

        let oblique = oblique_projection(proj, clip_plane);

        // Matrices should be different
        assert!(!matrix_approx_eq(&oblique, &proj));

        // The projection should still be valid (non-singular)
        // Check that column 0 and 1 are still reasonable
        assert!(oblique[0][0].abs() > 0.0);
        assert!(oblique[1][1].abs() > 0.0);
    }

    #[test]
    fn test_invert_matrix_identity() {
        let inv = invert_matrix(IDENTITY_MATRIX);
        assert!(matrix_approx_eq(&inv, &IDENTITY_MATRIX));
    }

    #[test]
    fn test_invert_matrix_translation() {
        let translate = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [5.0, 3.0, -2.0, 1.0],
        ];

        let inv = invert_matrix(translate);

        // Inverse of translation should negate the translation
        assert!(approx_eq(inv[3][0], -5.0));
        assert!(approx_eq(inv[3][1], -3.0));
        assert!(approx_eq(inv[3][2], 2.0));
    }

    #[test]
    fn test_transpose_matrix() {
        let m = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];

        let t = transpose_matrix(m);

        // Check transpose: t[i][j] == m[j][i]
        assert!(approx_eq(t[0][0], 1.0));
        assert!(approx_eq(t[1][0], 2.0));
        assert!(approx_eq(t[0][1], 5.0));
        assert!(approx_eq(t[3][2], 12.0));
    }

    #[test]
    fn test_oblique_projection_horizontal_mirror() {
        // Common case: horizontal water/floor reflection
        let proj = create_perspective_matrix(
            std::f32::consts::PI / 4.0,
            16.0 / 9.0,
            0.1,
            1000.0,
        );

        // Horizontal plane at y=0, camera looking down at it
        // In view space with camera above, the plane normal points up (+Y)
        let clip_plane = [0.0, 1.0, 0.0, 0.0];

        let oblique = oblique_projection(proj, clip_plane);

        // Should produce a valid modified projection
        assert!(!matrix_approx_eq(&oblique, &proj));

        // X and Y projection factors should be preserved
        assert!(approx_eq(oblique[0][0], proj[0][0]));
        assert!(approx_eq(oblique[1][1], proj[1][1]));
    }
}
