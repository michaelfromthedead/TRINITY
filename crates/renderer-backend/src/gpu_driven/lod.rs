//! LOD (Level of Detail) Distance Calculation Helpers (T-WGPU-P6.5.1).
//!
//! This module provides helper functions and structs for LOD selection based on
//! camera distance or screen coverage. It supports per-object LOD thresholds
//! and provides sensible defaults for common use cases.
//!
//! # Overview
//!
//! LOD selection reduces rendering cost by displaying lower-detail meshes
//! for distant objects. This module provides two primary methods:
//!
//! 1. **Distance-based LOD**: Select LOD based on raw distance from camera
//! 2. **Screen-size LOD**: Select LOD based on projected screen coverage
//!
//! # Data Layout
//!
//! The `LodDistances` struct is 16 bytes for GPU alignment:
//!
//! | Offset | Field        | Size | Description                    |
//! |--------|--------------|------|--------------------------------|
//! | 0      | thresholds   | 12   | Distance thresholds (3 floats) |
//! | 12     | _pad         | 4    | Padding for alignment          |
//!
//! The `LodParams` struct is 32 bytes for GPU alignment:
//!
//! | Offset | Field           | Size | Description                    |
//! |--------|-----------------|------|--------------------------------|
//! | 0      | camera_position | 12   | Camera world position          |
//! | 12     | _pad0           | 4    | Padding                        |
//! | 16     | screen_width    | 4    | Screen width in pixels         |
//! | 20     | screen_height   | 4    | Screen height in pixels        |
//! | 24     | fov_y           | 4    | Vertical field of view (rads)  |
//! | 28     | _pad1           | 4    | Padding for alignment          |
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::lod::{
//!     LodDistances, LodParams, LodConfig,
//!     distance_to_camera, select_lod_by_distance, select_lod_by_coverage,
//! };
//!
//! // Distance-based LOD selection
//! let camera_pos = [0.0, 0.0, 0.0];
//! let object_center = [15.0, 0.0, 0.0];
//! let distances = LodDistances::default();
//!
//! let dist = distance_to_camera(camera_pos, object_center);
//! let lod = select_lod_by_distance(dist, &distances);
//!
//! // Screen-size based LOD selection
//! let coverage = screen_coverage(
//!     camera_pos,
//!     object_center,
//!     1.0, // object radius
//!     std::f32::consts::FRAC_PI_4, // 45 degree FOV
//!     1080.0, // screen height
//! );
//! let lod = select_lod_by_coverage(coverage);
//! ```

use bytemuck::{Pod, Zeroable};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Size of LodDistances struct in bytes.
pub const LOD_DISTANCES_SIZE: usize = 16;

/// Size of LodParams struct in bytes.
pub const LOD_PARAMS_SIZE: usize = 32;

/// Maximum number of LOD levels (LOD 0, 1, 2, 3).
pub const MAX_LOD_LEVELS: u8 = 4;

/// Default distance threshold for LOD 0 -> 1 transition (meters).
pub const DEFAULT_LOD0_DISTANCE: f32 = 10.0;

/// Default distance threshold for LOD 1 -> 2 transition (meters).
pub const DEFAULT_LOD1_DISTANCE: f32 = 25.0;

/// Default distance threshold for LOD 2 -> 3 transition (meters).
pub const DEFAULT_LOD2_DISTANCE: f32 = 50.0;

/// Screen coverage threshold for LOD 0 (highest detail).
pub const COVERAGE_LOD0: f32 = 0.10; // > 10% of screen

/// Screen coverage threshold for LOD 1.
pub const COVERAGE_LOD1: f32 = 0.04; // > 4% of screen

/// Screen coverage threshold for LOD 2.
pub const COVERAGE_LOD2: f32 = 0.01; // > 1% of screen

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

// =============================================================================
// LOD LEVEL TYPE
// =============================================================================

/// LOD level (0 = highest detail, 3 = lowest).
pub type LodLevel = u8;

// =============================================================================
// LOD DISTANCES
// =============================================================================

/// LOD distance thresholds for an object.
///
/// Contains three distance thresholds for transitioning between LOD levels:
/// - `thresholds[0]`: Distance at which to switch from LOD 0 to LOD 1
/// - `thresholds[1]`: Distance at which to switch from LOD 1 to LOD 2
/// - `thresholds[2]`: Distance at which to switch from LOD 2 to LOD 3
///
/// Beyond `thresholds[2]`, objects remain at LOD 3 (lowest detail).
///
/// # Memory Layout (16 bytes, 16-byte aligned)
///
/// ```text
/// +------------------+--------+--------+----------------------------------+
/// | Field            | Offset | Size   | Description                      |
/// +------------------+--------+--------+----------------------------------+
/// | thresholds       | 0      | 12     | Distance thresholds (3 floats)   |
/// | _pad             | 12     | 4      | Padding for GPU alignment        |
/// +------------------+--------+--------+----------------------------------+
/// | Total            |        | 16     |                                  |
/// +------------------+--------+--------+----------------------------------+
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LodDistances {
    /// Distance thresholds for LOD 0->1, 1->2, 2->3 transitions.
    pub thresholds: [f32; 3],
    /// Padding for GPU alignment.
    pub _pad: f32,
}

impl Default for LodDistances {
    /// Create default LOD distances with sensible thresholds.
    ///
    /// Defaults:
    /// - LOD 0 -> 1: 10 meters
    /// - LOD 1 -> 2: 25 meters
    /// - LOD 2 -> 3: 50 meters
    fn default() -> Self {
        Self {
            thresholds: [DEFAULT_LOD0_DISTANCE, DEFAULT_LOD1_DISTANCE, DEFAULT_LOD2_DISTANCE],
            _pad: 0.0,
        }
    }
}

impl LodDistances {
    /// Size of this struct in bytes.
    pub const SIZE: usize = LOD_DISTANCES_SIZE;

    /// Create new LOD distances with custom thresholds.
    ///
    /// # Arguments
    ///
    /// * `lod0_to_1` - Distance for LOD 0 to LOD 1 transition
    /// * `lod1_to_2` - Distance for LOD 1 to LOD 2 transition
    /// * `lod2_to_3` - Distance for LOD 2 to LOD 3 transition
    ///
    /// # Example
    ///
    /// ```ignore
    /// let distances = LodDistances::new(5.0, 15.0, 30.0);
    /// ```
    #[inline]
    pub const fn new(lod0_to_1: f32, lod1_to_2: f32, lod2_to_3: f32) -> Self {
        Self {
            thresholds: [lod0_to_1, lod1_to_2, lod2_to_3],
            _pad: 0.0,
        }
    }

    /// Create LOD distances from an array of thresholds.
    #[inline]
    pub const fn from_array(thresholds: [f32; 3]) -> Self {
        Self {
            thresholds,
            _pad: 0.0,
        }
    }

    /// Scale all thresholds by a factor.
    ///
    /// Useful for adjusting LOD distances globally based on quality settings.
    #[inline]
    pub fn scaled(&self, factor: f32) -> Self {
        Self {
            thresholds: [
                self.thresholds[0] * factor,
                self.thresholds[1] * factor,
                self.thresholds[2] * factor,
            ],
            _pad: 0.0,
        }
    }

    /// Get the distance threshold for a specific LOD transition.
    ///
    /// # Arguments
    ///
    /// * `level` - The LOD level (0, 1, or 2)
    ///
    /// # Returns
    ///
    /// The distance threshold, or `f32::MAX` if level is out of range.
    #[inline]
    pub fn threshold(&self, level: usize) -> f32 {
        if level < 3 {
            self.thresholds[level]
        } else {
            f32::MAX
        }
    }
}

// =============================================================================
// LOD PARAMS
// =============================================================================

/// LOD selection parameters.
///
/// Contains camera information and screen dimensions for LOD calculations.
/// Used for both distance-based and screen-size-based LOD selection.
///
/// # Memory Layout (32 bytes, 16-byte aligned)
///
/// ```text
/// +------------------+--------+--------+----------------------------------+
/// | Field            | Offset | Size   | Description                      |
/// +------------------+--------+--------+----------------------------------+
/// | camera_position  | 0      | 12     | Camera world position            |
/// | _pad0            | 12     | 4      | Padding                          |
/// | screen_width     | 16     | 4      | Screen width in pixels           |
/// | screen_height    | 20     | 4      | Screen height in pixels          |
/// | fov_y            | 24     | 4      | Vertical FOV (radians)           |
/// | _pad1            | 28     | 4      | Padding for alignment            |
/// +------------------+--------+--------+----------------------------------+
/// | Total            |        | 32     |                                  |
/// +------------------+--------+--------+----------------------------------+
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct LodParams {
    /// Camera position in world space.
    pub camera_position: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad0: f32,
    /// Screen width in pixels.
    pub screen_width: f32,
    /// Screen height in pixels.
    pub screen_height: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Padding for 16-byte alignment.
    pub _pad1: f32,
}

impl Default for LodParams {
    fn default() -> Self {
        Self {
            camera_position: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            screen_width: 1920.0,
            screen_height: 1080.0,
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
            _pad1: 0.0,
        }
    }
}

impl LodParams {
    /// Size of this struct in bytes.
    pub const SIZE: usize = LOD_PARAMS_SIZE;

    /// Create new LOD params.
    ///
    /// # Arguments
    ///
    /// * `camera_position` - Camera world position
    /// * `screen_width` - Screen width in pixels
    /// * `screen_height` - Screen height in pixels
    /// * `fov_y` - Vertical field of view in radians
    #[inline]
    pub const fn new(
        camera_position: [f32; 3],
        screen_width: f32,
        screen_height: f32,
        fov_y: f32,
    ) -> Self {
        Self {
            camera_position,
            _pad0: 0.0,
            screen_width,
            screen_height,
            fov_y,
            _pad1: 0.0,
        }
    }

    /// Create params for standard 1080p display with 45-degree FOV.
    #[inline]
    pub const fn standard_1080p(camera_position: [f32; 3]) -> Self {
        Self::new(
            camera_position,
            1920.0,
            1080.0,
            0.7853981633974483, // PI/4
        )
    }

    /// Create params for standard 4K display with 45-degree FOV.
    #[inline]
    pub const fn standard_4k(camera_position: [f32; 3]) -> Self {
        Self::new(
            camera_position,
            3840.0,
            2160.0,
            0.7853981633974483, // PI/4
        )
    }

    /// Update camera position.
    #[inline]
    pub fn with_camera_position(mut self, position: [f32; 3]) -> Self {
        self.camera_position = position;
        self
    }

    /// Update screen dimensions.
    #[inline]
    pub fn with_screen_size(mut self, width: f32, height: f32) -> Self {
        self.screen_width = width;
        self.screen_height = height;
        self
    }

    /// Update field of view.
    #[inline]
    pub fn with_fov_y(mut self, fov_y: f32) -> Self {
        self.fov_y = fov_y;
        self
    }

    /// Get screen aspect ratio.
    #[inline]
    pub fn aspect_ratio(&self) -> f32 {
        if self.screen_height > EPSILON {
            self.screen_width / self.screen_height
        } else {
            1.0
        }
    }
}

// =============================================================================
// LOD CONFIG
// =============================================================================

/// Per-object LOD configuration.
///
/// Combines LOD distance thresholds with a flag indicating whether to use
/// screen-size-based LOD selection instead of distance-based.
#[derive(Clone, Copy, Debug)]
pub struct LodConfig {
    /// LOD distance thresholds.
    pub distances: LodDistances,
    /// If true, use screen coverage for LOD selection instead of distance.
    pub use_screen_size: bool,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            distances: LodDistances::default(),
            use_screen_size: false,
        }
    }
}

impl LodConfig {
    /// Create a new LOD config with custom distances.
    #[inline]
    pub const fn new(distances: LodDistances, use_screen_size: bool) -> Self {
        Self {
            distances,
            use_screen_size,
        }
    }

    /// Create a distance-based LOD config.
    #[inline]
    pub const fn distance_based(distances: LodDistances) -> Self {
        Self {
            distances,
            use_screen_size: false,
        }
    }

    /// Create a screen-size-based LOD config.
    #[inline]
    pub const fn screen_size_based(distances: LodDistances) -> Self {
        Self {
            distances,
            use_screen_size: true,
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Calculate distance from camera to object center.
///
/// Computes the Euclidean distance between two 3D points.
///
/// # Arguments
///
/// * `camera_pos` - Camera position in world space
/// * `object_center` - Object center in world space
///
/// # Returns
///
/// The distance in world units (meters).
///
/// # Example
///
/// ```ignore
/// let dist = distance_to_camera([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
/// assert_eq!(dist, 10.0);
/// ```
#[inline]
pub fn distance_to_camera(camera_pos: [f32; 3], object_center: [f32; 3]) -> f32 {
    let dx = object_center[0] - camera_pos[0];
    let dy = object_center[1] - camera_pos[1];
    let dz = object_center[2] - camera_pos[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Calculate squared distance from camera to object center.
///
/// More efficient than `distance_to_camera` when comparing distances
/// (avoids the sqrt operation).
///
/// # Arguments
///
/// * `camera_pos` - Camera position in world space
/// * `object_center` - Object center in world space
///
/// # Returns
///
/// The squared distance in world units squared.
#[inline]
pub fn distance_to_camera_squared(camera_pos: [f32; 3], object_center: [f32; 3]) -> f32 {
    let dx = object_center[0] - camera_pos[0];
    let dy = object_center[1] - camera_pos[1];
    let dz = object_center[2] - camera_pos[2];
    dx * dx + dy * dy + dz * dz
}

/// Calculate screen coverage (0.0 to 1.0) for a bounding sphere.
///
/// Estimates what fraction of the screen height the object would cover
/// based on its projected size. This is useful for screen-size-based LOD
/// selection.
///
/// # Arguments
///
/// * `camera_pos` - Camera position in world space
/// * `object_center` - Object center in world space
/// * `object_radius` - Object bounding sphere radius
/// * `fov_y` - Vertical field of view in radians
/// * `screen_height` - Screen height in pixels
///
/// # Returns
///
/// Coverage as a fraction of screen height (0.0 to 1.0+).
/// Values > 1.0 indicate the object fills more than the screen height.
///
/// # Example
///
/// ```ignore
/// let coverage = screen_coverage(
///     [0.0, 0.0, 0.0],
///     [10.0, 0.0, 0.0],
///     1.0,
///     std::f32::consts::FRAC_PI_4,
///     1080.0,
/// );
/// ```
#[inline]
pub fn screen_coverage(
    camera_pos: [f32; 3],
    object_center: [f32; 3],
    object_radius: f32,
    fov_y: f32,
    screen_height: f32,
) -> f32 {
    let distance = distance_to_camera(camera_pos, object_center);

    // Avoid division by zero for very close objects
    if distance < EPSILON {
        return 1.0;
    }

    // Calculate the projected size on screen
    // The projected height of a sphere of radius r at distance d with fov_y is:
    // projected_height_normalized = (2 * r) / (2 * d * tan(fov_y / 2))
    // coverage = projected_height / screen_height

    let half_fov = fov_y * 0.5;
    let tan_half_fov = half_fov.tan();

    // Avoid division by zero for degenerate FOV
    if tan_half_fov < EPSILON {
        return 1.0;
    }

    // The visible world height at the object's distance
    let visible_height = 2.0 * distance * tan_half_fov;

    // Avoid division by zero
    if visible_height < EPSILON {
        return 1.0;
    }

    // Object diameter divided by visible height gives normalized coverage
    let object_diameter = 2.0 * object_radius;
    object_diameter / visible_height
}

/// Select LOD level based on distance.
///
/// Returns the appropriate LOD level (0-3) based on the distance
/// compared to the given thresholds.
///
/// # Arguments
///
/// * `distance` - Distance from camera to object
/// * `thresholds` - LOD distance thresholds
///
/// # Returns
///
/// LOD level from 0 (highest detail) to 3 (lowest detail).
///
/// # Example
///
/// ```ignore
/// let thresholds = LodDistances::new(10.0, 25.0, 50.0);
/// assert_eq!(select_lod_by_distance(5.0, &thresholds), 0);   // Very close
/// assert_eq!(select_lod_by_distance(15.0, &thresholds), 1);  // Medium
/// assert_eq!(select_lod_by_distance(30.0, &thresholds), 2);  // Far
/// assert_eq!(select_lod_by_distance(100.0, &thresholds), 3); // Very far
/// ```
#[inline]
pub fn select_lod_by_distance(distance: f32, thresholds: &LodDistances) -> LodLevel {
    if distance < thresholds.thresholds[0] {
        0
    } else if distance < thresholds.thresholds[1] {
        1
    } else if distance < thresholds.thresholds[2] {
        2
    } else {
        3
    }
}

/// Select LOD level based on squared distance.
///
/// More efficient than `select_lod_by_distance` when distances are already squared.
/// Note: The thresholds must also be squared for correct comparison.
///
/// # Arguments
///
/// * `distance_squared` - Squared distance from camera to object
/// * `thresholds_squared` - Squared LOD distance thresholds
///
/// # Returns
///
/// LOD level from 0 (highest detail) to 3 (lowest detail).
#[inline]
pub fn select_lod_by_distance_squared(
    distance_squared: f32,
    thresholds_squared: &[f32; 3],
) -> LodLevel {
    if distance_squared < thresholds_squared[0] {
        0
    } else if distance_squared < thresholds_squared[1] {
        1
    } else if distance_squared < thresholds_squared[2] {
        2
    } else {
        3
    }
}

/// Select LOD level based on screen coverage.
///
/// Returns the appropriate LOD level (0-3) based on how much of
/// the screen the object covers.
///
/// # Arguments
///
/// * `coverage` - Screen coverage (0.0 to 1.0+)
///
/// # Returns
///
/// LOD level from 0 (highest detail, large coverage) to 3 (lowest detail, small coverage).
///
/// # Thresholds
///
/// - Coverage >= 10%: LOD 0 (highest detail)
/// - Coverage >= 4%:  LOD 1
/// - Coverage >= 1%:  LOD 2
/// - Coverage < 1%:   LOD 3 (lowest detail)
///
/// # Example
///
/// ```ignore
/// assert_eq!(select_lod_by_coverage(0.15), 0); // Large object
/// assert_eq!(select_lod_by_coverage(0.05), 1); // Medium object
/// assert_eq!(select_lod_by_coverage(0.02), 2); // Small object
/// assert_eq!(select_lod_by_coverage(0.005), 3); // Tiny object
/// ```
#[inline]
pub fn select_lod_by_coverage(coverage: f32) -> LodLevel {
    if coverage >= COVERAGE_LOD0 {
        0
    } else if coverage >= COVERAGE_LOD1 {
        1
    } else if coverage >= COVERAGE_LOD2 {
        2
    } else {
        3
    }
}

/// Select LOD level with custom coverage thresholds.
///
/// # Arguments
///
/// * `coverage` - Screen coverage (0.0 to 1.0+)
/// * `coverage_thresholds` - Custom coverage thresholds [lod0, lod1, lod2]
///
/// # Returns
///
/// LOD level from 0 (highest detail) to 3 (lowest detail).
#[inline]
pub fn select_lod_by_coverage_custom(coverage: f32, coverage_thresholds: &[f32; 3]) -> LodLevel {
    if coverage >= coverage_thresholds[0] {
        0
    } else if coverage >= coverage_thresholds[1] {
        1
    } else if coverage >= coverage_thresholds[2] {
        2
    } else {
        3
    }
}

/// Calculate the appropriate LOD for an object using either distance or screen size.
///
/// # Arguments
///
/// * `params` - LOD parameters (camera position, screen info)
/// * `object_center` - Object center in world space
/// * `object_radius` - Object bounding sphere radius
/// * `config` - Object's LOD configuration
///
/// # Returns
///
/// LOD level from 0 (highest detail) to 3 (lowest detail).
#[inline]
pub fn select_lod(
    params: &LodParams,
    object_center: [f32; 3],
    object_radius: f32,
    config: &LodConfig,
) -> LodLevel {
    if config.use_screen_size {
        let coverage = screen_coverage(
            params.camera_position,
            object_center,
            object_radius,
            params.fov_y,
            params.screen_height,
        );
        select_lod_by_coverage(coverage)
    } else {
        let distance = distance_to_camera(params.camera_position, object_center);
        select_lod_by_distance(distance, &config.distances)
    }
}

/// Calculate squared distance thresholds from linear thresholds.
///
/// Useful for efficient GPU comparisons that avoid sqrt.
#[inline]
pub fn squared_thresholds(thresholds: &LodDistances) -> [f32; 3] {
    [
        thresholds.thresholds[0] * thresholds.thresholds[0],
        thresholds.thresholds[1] * thresholds.thresholds[1],
        thresholds.thresholds[2] * thresholds.thresholds[2],
    ]
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CATEGORY 1: STRUCT LAYOUT TESTS
    // =========================================================================

    #[test]
    fn test_lod_distances_size() {
        assert_eq!(
            std::mem::size_of::<LodDistances>(),
            LOD_DISTANCES_SIZE,
            "LodDistances must be {} bytes",
            LOD_DISTANCES_SIZE
        );
        assert_eq!(LodDistances::SIZE, 16);
    }

    #[test]
    fn test_lod_params_size() {
        assert_eq!(
            std::mem::size_of::<LodParams>(),
            LOD_PARAMS_SIZE,
            "LodParams must be {} bytes",
            LOD_PARAMS_SIZE
        );
        assert_eq!(LodParams::SIZE, 32);
    }

    #[test]
    fn test_lod_distances_field_offsets() {
        // Verify GPU alignment: thresholds at offset 0, _pad at offset 12
        let distances = LodDistances::new(1.0, 2.0, 3.0);
        let bytes = bytemuck::bytes_of(&distances);

        // thresholds[0] at offset 0
        let threshold0 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(threshold0, 1.0);

        // thresholds[1] at offset 4
        let threshold1 = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(threshold1, 2.0);

        // thresholds[2] at offset 8
        let threshold2 = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(threshold2, 3.0);

        // _pad at offset 12
        let pad = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(pad, 0.0);
    }

    #[test]
    fn test_lod_params_field_offsets() {
        // Verify GPU alignment per documented layout
        let params = LodParams::new([1.0, 2.0, 3.0], 1920.0, 1080.0, 0.785);
        let bytes = bytemuck::bytes_of(&params);

        // camera_position at offset 0 (12 bytes)
        let cam_x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let cam_y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let cam_z = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(cam_x, 1.0);
        assert_eq!(cam_y, 2.0);
        assert_eq!(cam_z, 3.0);

        // _pad0 at offset 12
        let pad0 = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(pad0, 0.0);

        // screen_width at offset 16
        let width = f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        assert_eq!(width, 1920.0);

        // screen_height at offset 20
        let height = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        assert_eq!(height, 1080.0);

        // fov_y at offset 24
        let fov = f32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        assert!((fov - 0.785).abs() < EPSILON);

        // _pad1 at offset 28
        let pad1 = f32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        assert_eq!(pad1, 0.0);
    }

    #[test]
    fn test_lod_distances_pod_zeroable() {
        // Pod trait: can be safely cast to/from bytes
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let bytes: &[u8] = bytemuck::bytes_of(&distances);
        assert_eq!(bytes.len(), 16);

        // Zeroable trait: can be zeroed
        let zeroed: LodDistances = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.thresholds, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed._pad, 0.0);
    }

    #[test]
    fn test_lod_params_pod_zeroable() {
        let params = LodParams::standard_1080p([1.0, 2.0, 3.0]);
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 32);

        let zeroed: LodParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.camera_position, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed.screen_width, 0.0);
        assert_eq!(zeroed.screen_height, 0.0);
        assert_eq!(zeroed.fov_y, 0.0);
    }

    #[test]
    fn test_constants_values() {
        assert_eq!(LOD_DISTANCES_SIZE, 16);
        assert_eq!(LOD_PARAMS_SIZE, 32);
        assert_eq!(MAX_LOD_LEVELS, 4);
        assert_eq!(DEFAULT_LOD0_DISTANCE, 10.0);
        assert_eq!(DEFAULT_LOD1_DISTANCE, 25.0);
        assert_eq!(DEFAULT_LOD2_DISTANCE, 50.0);
        assert_eq!(COVERAGE_LOD0, 0.10);
        assert_eq!(COVERAGE_LOD1, 0.04);
        assert_eq!(COVERAGE_LOD2, 0.01);
    }

    #[test]
    fn test_max_lod_levels() {
        assert_eq!(MAX_LOD_LEVELS, 4);
    }

    #[test]
    fn test_lod_distances_repr_c() {
        // Ensure repr(C) alignment is correct for GPU interop
        assert_eq!(std::mem::align_of::<LodDistances>(), 4);
        assert_eq!(std::mem::size_of::<LodDistances>(), 16);
    }

    #[test]
    fn test_lod_params_repr_c() {
        // Ensure repr(C) alignment is correct for GPU interop
        assert_eq!(std::mem::align_of::<LodParams>(), 4);
        assert_eq!(std::mem::size_of::<LodParams>(), 32);
    }

    // =========================================================================
    // CATEGORY 2: DISTANCE CALCULATION TESTS
    // =========================================================================

    #[test]
    fn test_distance_to_camera() {
        // Origin to point on X axis
        let dist = distance_to_camera([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        assert!((dist - 10.0).abs() < EPSILON);

        // Origin to point on Y axis
        let dist = distance_to_camera([0.0, 0.0, 0.0], [0.0, 5.0, 0.0]);
        assert!((dist - 5.0).abs() < EPSILON);

        // Origin to point on Z axis
        let dist = distance_to_camera([0.0, 0.0, 0.0], [0.0, 0.0, 3.0]);
        assert!((dist - 3.0).abs() < EPSILON);

        // Diagonal (3-4-5 triangle in 2D, extended to 3D)
        let dist = distance_to_camera([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        assert!((dist - 5.0).abs() < EPSILON);

        // Non-origin camera
        let dist = distance_to_camera([1.0, 2.0, 3.0], [4.0, 6.0, 3.0]);
        // sqrt((4-1)^2 + (6-2)^2 + (3-3)^2) = sqrt(9 + 16) = 5
        assert!((dist - 5.0).abs() < EPSILON);

        // Same point
        let dist = distance_to_camera([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]);
        assert!(dist < EPSILON);
    }

    #[test]
    fn test_distance_to_camera_negative_coords() {
        // Negative X
        let dist = distance_to_camera([0.0, 0.0, 0.0], [-10.0, 0.0, 0.0]);
        assert!((dist - 10.0).abs() < EPSILON);

        // Negative Y
        let dist = distance_to_camera([0.0, 0.0, 0.0], [0.0, -5.0, 0.0]);
        assert!((dist - 5.0).abs() < EPSILON);

        // Negative Z
        let dist = distance_to_camera([0.0, 0.0, 0.0], [0.0, 0.0, -3.0]);
        assert!((dist - 3.0).abs() < EPSILON);

        // All negative
        let dist = distance_to_camera([-1.0, -2.0, -3.0], [-4.0, -6.0, -3.0]);
        assert!((dist - 5.0).abs() < EPSILON);

        // Crossing zero
        let dist = distance_to_camera([-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]);
        assert!((dist - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_distance_to_camera_large_values() {
        // Large coordinate values
        let dist = distance_to_camera([0.0, 0.0, 0.0], [1000000.0, 0.0, 0.0]);
        assert!((dist - 1000000.0).abs() < 1.0); // Allow small error at large scale

        // Very large values (test numerical stability)
        let dist = distance_to_camera([1e6, 1e6, 1e6], [1e6 + 3.0, 1e6 + 4.0, 1e6]);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_to_camera_small_values() {
        // Very small distances
        let dist = distance_to_camera([0.0, 0.0, 0.0], [0.001, 0.0, 0.0]);
        assert!((dist - 0.001).abs() < EPSILON);

        // Tiny values
        let dist = distance_to_camera([0.0, 0.0, 0.0], [1e-5, 0.0, 0.0]);
        assert!((dist - 1e-5).abs() < EPSILON);
    }

    #[test]
    fn test_distance_to_camera_3d_diagonal() {
        // 3D diagonal: sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
        let dist = distance_to_camera([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((dist - 3.0_f32.sqrt()).abs() < EPSILON);

        // 3D diagonal scaled
        let dist = distance_to_camera([0.0, 0.0, 0.0], [3.0, 4.0, 12.0]);
        // sqrt(9 + 16 + 144) = sqrt(169) = 13
        assert!((dist - 13.0).abs() < EPSILON);
    }

    #[test]
    fn test_distance_to_camera_squared() {
        let dist_sq = distance_to_camera_squared([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        assert!((dist_sq - 100.0).abs() < EPSILON);

        let dist_sq = distance_to_camera_squared([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        assert!((dist_sq - 25.0).abs() < EPSILON);
    }

    #[test]
    fn test_distance_to_camera_squared_negative() {
        let dist_sq = distance_to_camera_squared([0.0, 0.0, 0.0], [-10.0, 0.0, 0.0]);
        assert!((dist_sq - 100.0).abs() < EPSILON);

        let dist_sq = distance_to_camera_squared([-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]);
        assert!((dist_sq - 100.0).abs() < EPSILON);
    }

    #[test]
    fn test_distance_to_camera_squared_same_point() {
        let dist_sq = distance_to_camera_squared([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]);
        assert!(dist_sq < EPSILON);
    }

    #[test]
    fn test_distance_squared_vs_regular() {
        // Verify squared distance equals distance^2
        let camera = [1.0, 2.0, 3.0];
        let object = [4.0, 6.0, 8.0];

        let dist = distance_to_camera(camera, object);
        let dist_sq = distance_to_camera_squared(camera, object);

        assert!((dist_sq - dist * dist).abs() < EPSILON);
    }

    // =========================================================================
    // CATEGORY 3: SCREEN COVERAGE TESTS
    // =========================================================================

    #[test]
    fn test_screen_coverage() {
        let camera_pos = [0.0, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4; // 45 degrees
        let screen_height = 1080.0;

        // Object at distance 10 with radius 1
        let coverage = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 1.0, fov_y, screen_height);
        // Should be moderate coverage
        assert!(coverage > 0.0);
        assert!(coverage < 1.0);

        // Closer object should have higher coverage
        let coverage_close = screen_coverage(camera_pos, [5.0, 0.0, 0.0], 1.0, fov_y, screen_height);
        assert!(coverage_close > coverage);

        // Larger object should have higher coverage
        let coverage_large = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 2.0, fov_y, screen_height);
        assert!(coverage_large > coverage);

        // Very close object
        let coverage_very_close = screen_coverage(camera_pos, [0.001, 0.0, 0.0], 1.0, fov_y, screen_height);
        assert!(coverage_very_close > 0.9);
    }

    #[test]
    fn test_screen_coverage_close_objects() {
        let camera_pos = [0.0, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let screen_height = 1080.0;

        // Object at camera position should return 1.0 (clamped)
        let coverage = screen_coverage(camera_pos, [0.0, 0.0, 0.0], 1.0, fov_y, screen_height);
        assert_eq!(coverage, 1.0);

        // Object very close (within EPSILON)
        let coverage = screen_coverage(camera_pos, [1e-7, 0.0, 0.0], 1.0, fov_y, screen_height);
        assert_eq!(coverage, 1.0);
    }

    #[test]
    fn test_screen_coverage_far_objects() {
        let camera_pos = [0.0, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let screen_height = 1080.0;

        // Object very far away
        let coverage = screen_coverage(camera_pos, [1000.0, 0.0, 0.0], 1.0, fov_y, screen_height);
        assert!(coverage < 0.01);
        assert!(coverage > 0.0);

        // Even farther
        let coverage_farther = screen_coverage(camera_pos, [10000.0, 0.0, 0.0], 1.0, fov_y, screen_height);
        assert!(coverage_farther < coverage);
    }

    #[test]
    fn test_screen_coverage_large_radius() {
        let camera_pos = [0.0, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let screen_height = 1080.0;

        // Large object at moderate distance
        let coverage = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 10.0, fov_y, screen_height);
        assert!(coverage > 0.5);

        // Very large object
        let coverage_larger = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 20.0, fov_y, screen_height);
        assert!(coverage_larger > coverage);
    }

    #[test]
    fn test_screen_coverage_small_radius() {
        let camera_pos = [0.0, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let screen_height = 1080.0;

        // Tiny object at moderate distance
        let coverage = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 0.1, fov_y, screen_height);
        assert!(coverage < 0.1);

        // Even smaller
        let coverage_smaller = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 0.01, fov_y, screen_height);
        assert!(coverage_smaller < coverage);
    }

    #[test]
    fn test_screen_coverage_different_fov() {
        let camera_pos = [0.0, 0.0, 0.0];
        let screen_height = 1080.0;
        let object_pos = [10.0, 0.0, 0.0];
        let radius = 1.0;

        // Narrow FOV (30 degrees)
        let fov_narrow = std::f32::consts::PI / 6.0;
        let coverage_narrow = screen_coverage(camera_pos, object_pos, radius, fov_narrow, screen_height);

        // Wide FOV (90 degrees)
        let fov_wide = std::f32::consts::FRAC_PI_2;
        let coverage_wide = screen_coverage(camera_pos, object_pos, radius, fov_wide, screen_height);

        // Narrow FOV should give higher coverage (object appears larger)
        assert!(coverage_narrow > coverage_wide);
    }

    #[test]
    fn test_screen_coverage_degenerate_fov() {
        let camera_pos = [0.0, 0.0, 0.0];
        let screen_height = 1080.0;

        // Very small FOV (degenerate case)
        let coverage = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 1.0, 1e-8, screen_height);
        // Should return 1.0 to avoid division by zero
        assert_eq!(coverage, 1.0);

        // Zero FOV
        let coverage = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 1.0, 0.0, screen_height);
        assert_eq!(coverage, 1.0);
    }

    #[test]
    fn test_screen_coverage_proportional_to_radius() {
        let camera_pos = [0.0, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let screen_height = 1080.0;
        let object_pos = [10.0, 0.0, 0.0];

        let coverage_1 = screen_coverage(camera_pos, object_pos, 1.0, fov_y, screen_height);
        let coverage_2 = screen_coverage(camera_pos, object_pos, 2.0, fov_y, screen_height);

        // Coverage should scale linearly with radius
        assert!((coverage_2 / coverage_1 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_screen_coverage_inverse_to_distance() {
        let camera_pos = [0.0, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let screen_height = 1080.0;

        let coverage_10 = screen_coverage(camera_pos, [10.0, 0.0, 0.0], 1.0, fov_y, screen_height);
        let coverage_20 = screen_coverage(camera_pos, [20.0, 0.0, 0.0], 1.0, fov_y, screen_height);

        // Coverage should be inversely proportional to distance
        assert!((coverage_10 / coverage_20 - 2.0).abs() < 0.01);
    }

    // =========================================================================
    // CATEGORY 4: LOD SELECTION TESTS
    // =========================================================================

    #[test]
    fn test_select_lod_by_distance() {
        let thresholds = LodDistances::new(10.0, 25.0, 50.0);

        // LOD 0: distance < 10
        assert_eq!(select_lod_by_distance(0.0, &thresholds), 0);
        assert_eq!(select_lod_by_distance(5.0, &thresholds), 0);
        assert_eq!(select_lod_by_distance(9.9, &thresholds), 0);

        // LOD 1: 10 <= distance < 25
        assert_eq!(select_lod_by_distance(10.0, &thresholds), 1);
        assert_eq!(select_lod_by_distance(15.0, &thresholds), 1);
        assert_eq!(select_lod_by_distance(24.9, &thresholds), 1);

        // LOD 2: 25 <= distance < 50
        assert_eq!(select_lod_by_distance(25.0, &thresholds), 2);
        assert_eq!(select_lod_by_distance(35.0, &thresholds), 2);
        assert_eq!(select_lod_by_distance(49.9, &thresholds), 2);

        // LOD 3: distance >= 50
        assert_eq!(select_lod_by_distance(50.0, &thresholds), 3);
        assert_eq!(select_lod_by_distance(100.0, &thresholds), 3);
        assert_eq!(select_lod_by_distance(1000.0, &thresholds), 3);
    }

    #[test]
    fn test_select_lod_by_distance_exact_thresholds() {
        let thresholds = LodDistances::new(10.0, 25.0, 50.0);

        // Test exact threshold values (boundary conditions)
        assert_eq!(select_lod_by_distance(10.0, &thresholds), 1); // Exactly at threshold
        assert_eq!(select_lod_by_distance(25.0, &thresholds), 2);
        assert_eq!(select_lod_by_distance(50.0, &thresholds), 3);

        // Just below threshold (use larger offset to avoid float precision issues)
        assert_eq!(select_lod_by_distance(9.99, &thresholds), 0);
        assert_eq!(select_lod_by_distance(24.99, &thresholds), 1);
        assert_eq!(select_lod_by_distance(49.99, &thresholds), 2);
    }

    #[test]
    fn test_select_lod_by_distance_default_thresholds() {
        let thresholds = LodDistances::default();

        // Test with default values (10, 25, 50)
        assert_eq!(select_lod_by_distance(5.0, &thresholds), 0);
        assert_eq!(select_lod_by_distance(15.0, &thresholds), 1);
        assert_eq!(select_lod_by_distance(35.0, &thresholds), 2);
        assert_eq!(select_lod_by_distance(100.0, &thresholds), 3);
    }

    #[test]
    fn test_select_lod_by_distance_custom_thresholds() {
        // Very close thresholds
        let thresholds = LodDistances::new(1.0, 2.0, 3.0);
        assert_eq!(select_lod_by_distance(0.5, &thresholds), 0);
        assert_eq!(select_lod_by_distance(1.5, &thresholds), 1);
        assert_eq!(select_lod_by_distance(2.5, &thresholds), 2);
        assert_eq!(select_lod_by_distance(3.5, &thresholds), 3);

        // Very far thresholds
        let thresholds = LodDistances::new(100.0, 500.0, 1000.0);
        assert_eq!(select_lod_by_distance(50.0, &thresholds), 0);
        assert_eq!(select_lod_by_distance(300.0, &thresholds), 1);
        assert_eq!(select_lod_by_distance(750.0, &thresholds), 2);
        assert_eq!(select_lod_by_distance(1500.0, &thresholds), 3);
    }

    #[test]
    fn test_select_lod_by_distance_negative_distance() {
        let thresholds = LodDistances::new(10.0, 25.0, 50.0);

        // Negative distance should select LOD 0 (closest)
        assert_eq!(select_lod_by_distance(-1.0, &thresholds), 0);
        assert_eq!(select_lod_by_distance(-100.0, &thresholds), 0);
    }

    #[test]
    fn test_select_lod_by_distance_zero() {
        let thresholds = LodDistances::new(10.0, 25.0, 50.0);
        assert_eq!(select_lod_by_distance(0.0, &thresholds), 0);
    }

    #[test]
    fn test_select_lod_by_coverage() {
        // LOD 0: coverage >= 10%
        assert_eq!(select_lod_by_coverage(1.0), 0);
        assert_eq!(select_lod_by_coverage(0.5), 0);
        assert_eq!(select_lod_by_coverage(0.10), 0);

        // LOD 1: 4% <= coverage < 10%
        assert_eq!(select_lod_by_coverage(0.099), 1);
        assert_eq!(select_lod_by_coverage(0.05), 1);
        assert_eq!(select_lod_by_coverage(0.04), 1);

        // LOD 2: 1% <= coverage < 4%
        assert_eq!(select_lod_by_coverage(0.039), 2);
        assert_eq!(select_lod_by_coverage(0.02), 2);
        assert_eq!(select_lod_by_coverage(0.01), 2);

        // LOD 3: coverage < 1%
        assert_eq!(select_lod_by_coverage(0.009), 3);
        assert_eq!(select_lod_by_coverage(0.001), 3);
        assert_eq!(select_lod_by_coverage(0.0), 3);
    }

    #[test]
    fn test_select_lod_by_coverage_exact_thresholds() {
        // Test exact threshold values
        assert_eq!(select_lod_by_coverage(COVERAGE_LOD0), 0); // 0.10
        assert_eq!(select_lod_by_coverage(COVERAGE_LOD1), 1); // 0.04
        assert_eq!(select_lod_by_coverage(COVERAGE_LOD2), 2); // 0.01

        // Just below thresholds
        assert_eq!(select_lod_by_coverage(COVERAGE_LOD0 - EPSILON), 1);
        assert_eq!(select_lod_by_coverage(COVERAGE_LOD1 - EPSILON), 2);
        assert_eq!(select_lod_by_coverage(COVERAGE_LOD2 - EPSILON), 3);
    }

    #[test]
    fn test_select_lod_by_coverage_greater_than_one() {
        // Coverage > 1.0 (object fills more than screen height)
        assert_eq!(select_lod_by_coverage(1.5), 0);
        assert_eq!(select_lod_by_coverage(10.0), 0);
        assert_eq!(select_lod_by_coverage(100.0), 0);
    }

    #[test]
    fn test_select_lod_by_coverage_negative() {
        // Negative coverage (invalid but should handle gracefully)
        assert_eq!(select_lod_by_coverage(-0.1), 3);
        assert_eq!(select_lod_by_coverage(-1.0), 3);
    }

    #[test]
    fn test_select_lod_by_distance_squared() {
        let thresholds_squared = [100.0, 625.0, 2500.0]; // 10^2, 25^2, 50^2

        assert_eq!(select_lod_by_distance_squared(50.0, &thresholds_squared), 0); // sqrt(50) ~= 7
        assert_eq!(select_lod_by_distance_squared(100.0, &thresholds_squared), 1); // sqrt(100) = 10
        assert_eq!(select_lod_by_distance_squared(625.0, &thresholds_squared), 2); // sqrt(625) = 25
        assert_eq!(select_lod_by_distance_squared(2500.0, &thresholds_squared), 3); // sqrt(2500) = 50
    }

    #[test]
    fn test_select_lod_by_distance_squared_exact_thresholds() {
        let thresholds_squared = [100.0, 625.0, 2500.0];

        // Just below threshold
        assert_eq!(select_lod_by_distance_squared(99.9, &thresholds_squared), 0);
        assert_eq!(select_lod_by_distance_squared(624.9, &thresholds_squared), 1);
        assert_eq!(select_lod_by_distance_squared(2499.9, &thresholds_squared), 2);
    }

    #[test]
    fn test_select_lod_by_coverage_custom() {
        let thresholds = [0.20, 0.10, 0.05]; // More aggressive thresholds

        assert_eq!(select_lod_by_coverage_custom(0.25, &thresholds), 0);
        assert_eq!(select_lod_by_coverage_custom(0.15, &thresholds), 1);
        assert_eq!(select_lod_by_coverage_custom(0.07, &thresholds), 2);
        assert_eq!(select_lod_by_coverage_custom(0.02, &thresholds), 3);
    }

    #[test]
    fn test_select_lod_by_coverage_custom_exact_thresholds() {
        let thresholds = [0.20, 0.10, 0.05];

        assert_eq!(select_lod_by_coverage_custom(0.20, &thresholds), 0);
        assert_eq!(select_lod_by_coverage_custom(0.10, &thresholds), 1);
        assert_eq!(select_lod_by_coverage_custom(0.05, &thresholds), 2);
    }

    #[test]
    fn test_select_lod_integration() {
        let params = LodParams::standard_1080p([0.0, 0.0, 0.0]);
        let distances = LodDistances::new(10.0, 25.0, 50.0);

        // Distance-based
        let config = LodConfig::distance_based(distances);
        let lod = select_lod(&params, [5.0, 0.0, 0.0], 1.0, &config);
        assert_eq!(lod, 0); // 5 meters, LOD 0

        let lod = select_lod(&params, [15.0, 0.0, 0.0], 1.0, &config);
        assert_eq!(lod, 1); // 15 meters, LOD 1

        let lod = select_lod(&params, [35.0, 0.0, 0.0], 1.0, &config);
        assert_eq!(lod, 2); // 35 meters, LOD 2

        let lod = select_lod(&params, [100.0, 0.0, 0.0], 1.0, &config);
        assert_eq!(lod, 3); // 100 meters, LOD 3
    }

    #[test]
    fn test_select_lod_screen_size_based() {
        let params = LodParams::standard_1080p([0.0, 0.0, 0.0]);
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let config = LodConfig::screen_size_based(distances);

        // Large object close should be LOD 0
        let lod = select_lod(&params, [2.0, 0.0, 0.0], 2.0, &config);
        assert_eq!(lod, 0);

        // Small object far away should be higher LOD
        let lod_far = select_lod(&params, [100.0, 0.0, 0.0], 0.1, &config);
        assert!(lod_far > 0);
    }

    // =========================================================================
    // CATEGORY 5: HELPER FUNCTION TESTS
    // =========================================================================

    #[test]
    fn test_squared_thresholds() {
        let distances = LodDistances::new(10.0, 20.0, 30.0);
        let squared = squared_thresholds(&distances);

        assert_eq!(squared[0], 100.0);
        assert_eq!(squared[1], 400.0);
        assert_eq!(squared[2], 900.0);
    }

    #[test]
    fn test_squared_thresholds_default() {
        let distances = LodDistances::default();
        let squared = squared_thresholds(&distances);

        assert_eq!(squared[0], DEFAULT_LOD0_DISTANCE * DEFAULT_LOD0_DISTANCE);
        assert_eq!(squared[1], DEFAULT_LOD1_DISTANCE * DEFAULT_LOD1_DISTANCE);
        assert_eq!(squared[2], DEFAULT_LOD2_DISTANCE * DEFAULT_LOD2_DISTANCE);
    }

    #[test]
    fn test_squared_thresholds_small_values() {
        let distances = LodDistances::new(0.1, 0.5, 1.0);
        let squared = squared_thresholds(&distances);

        assert!((squared[0] - 0.01).abs() < EPSILON);
        assert!((squared[1] - 0.25).abs() < EPSILON);
        assert!((squared[2] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_squared_thresholds_large_values() {
        let distances = LodDistances::new(100.0, 200.0, 500.0);
        let squared = squared_thresholds(&distances);

        assert_eq!(squared[0], 10000.0);
        assert_eq!(squared[1], 40000.0);
        assert_eq!(squared[2], 250000.0);
    }

    #[test]
    fn test_lod_config_creation() {
        let config = LodConfig::default();
        assert!(!config.use_screen_size);

        let distances = LodDistances::new(5.0, 15.0, 30.0);

        let config = LodConfig::distance_based(distances);
        assert!(!config.use_screen_size);
        assert_eq!(config.distances.thresholds[0], 5.0);

        let config = LodConfig::screen_size_based(distances);
        assert!(config.use_screen_size);
    }

    #[test]
    fn test_lod_config_new() {
        let distances = LodDistances::new(5.0, 15.0, 30.0);
        let config = LodConfig::new(distances, true);

        assert!(config.use_screen_size);
        assert_eq!(config.distances.thresholds[0], 5.0);
        assert_eq!(config.distances.thresholds[1], 15.0);
        assert_eq!(config.distances.thresholds[2], 30.0);
    }

    #[test]
    fn test_lod_distances_from_array() {
        let thresholds = [5.0, 15.0, 30.0];
        let distances = LodDistances::from_array(thresholds);

        assert_eq!(distances.thresholds, thresholds);
        assert_eq!(distances._pad, 0.0);
    }

    #[test]
    fn test_default_distances() {
        let distances = LodDistances::default();

        assert_eq!(distances.thresholds[0], DEFAULT_LOD0_DISTANCE);
        assert_eq!(distances.thresholds[1], DEFAULT_LOD1_DISTANCE);
        assert_eq!(distances.thresholds[2], DEFAULT_LOD2_DISTANCE);
        assert_eq!(distances._pad, 0.0);

        // Sensible progression
        assert!(distances.thresholds[0] < distances.thresholds[1]);
        assert!(distances.thresholds[1] < distances.thresholds[2]);
    }

    #[test]
    fn test_lod_distances_scaled() {
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let scaled = distances.scaled(2.0);

        assert_eq!(scaled.thresholds[0], 20.0);
        assert_eq!(scaled.thresholds[1], 50.0);
        assert_eq!(scaled.thresholds[2], 100.0);
    }

    #[test]
    fn test_lod_distances_scaled_fractional() {
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let scaled = distances.scaled(0.5);

        assert_eq!(scaled.thresholds[0], 5.0);
        assert_eq!(scaled.thresholds[1], 12.5);
        assert_eq!(scaled.thresholds[2], 25.0);
    }

    #[test]
    fn test_lod_distances_scaled_zero() {
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let scaled = distances.scaled(0.0);

        assert_eq!(scaled.thresholds[0], 0.0);
        assert_eq!(scaled.thresholds[1], 0.0);
        assert_eq!(scaled.thresholds[2], 0.0);
    }

    #[test]
    fn test_lod_distances_scaled_negative() {
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let scaled = distances.scaled(-1.0);

        assert_eq!(scaled.thresholds[0], -10.0);
        assert_eq!(scaled.thresholds[1], -25.0);
        assert_eq!(scaled.thresholds[2], -50.0);
    }

    #[test]
    fn test_lod_distances_threshold_method() {
        let distances = LodDistances::new(10.0, 25.0, 50.0);

        assert_eq!(distances.threshold(0), 10.0);
        assert_eq!(distances.threshold(1), 25.0);
        assert_eq!(distances.threshold(2), 50.0);
        assert_eq!(distances.threshold(3), f32::MAX); // Out of range
        assert_eq!(distances.threshold(100), f32::MAX); // Way out of range
    }

    #[test]
    fn test_lod_params_builders() {
        let params = LodParams::standard_1080p([1.0, 2.0, 3.0]);
        assert_eq!(params.camera_position, [1.0, 2.0, 3.0]);
        assert_eq!(params.screen_width, 1920.0);
        assert_eq!(params.screen_height, 1080.0);

        let params = LodParams::standard_4k([0.0, 0.0, 0.0]);
        assert_eq!(params.screen_width, 3840.0);
        assert_eq!(params.screen_height, 2160.0);

        let params = LodParams::default()
            .with_camera_position([5.0, 10.0, 15.0])
            .with_screen_size(2560.0, 1440.0)
            .with_fov_y(1.0);

        assert_eq!(params.camera_position, [5.0, 10.0, 15.0]);
        assert_eq!(params.screen_width, 2560.0);
        assert_eq!(params.screen_height, 1440.0);
        assert_eq!(params.fov_y, 1.0);
    }

    #[test]
    fn test_lod_params_default() {
        let params = LodParams::default();

        assert_eq!(params.camera_position, [0.0, 0.0, 0.0]);
        assert_eq!(params._pad0, 0.0);
        assert_eq!(params.screen_width, 1920.0);
        assert_eq!(params.screen_height, 1080.0);
        assert_eq!(params.fov_y, std::f32::consts::FRAC_PI_4);
        assert_eq!(params._pad1, 0.0);
    }

    #[test]
    fn test_lod_params_aspect_ratio() {
        let params = LodParams::standard_1080p([0.0, 0.0, 0.0]);
        let aspect = params.aspect_ratio();
        assert!((aspect - 16.0 / 9.0).abs() < 0.01);

        let params = LodParams::new([0.0, 0.0, 0.0], 1920.0, 1920.0, 0.7);
        let aspect = params.aspect_ratio();
        assert!((aspect - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_lod_params_aspect_ratio_4k() {
        let params = LodParams::standard_4k([0.0, 0.0, 0.0]);
        let aspect = params.aspect_ratio();
        assert!((aspect - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_lod_params_aspect_ratio_ultrawide() {
        let params = LodParams::new([0.0, 0.0, 0.0], 3440.0, 1440.0, 0.7);
        let aspect = params.aspect_ratio();
        assert!((aspect - 3440.0 / 1440.0).abs() < 0.01);
    }

    #[test]
    fn test_lod_params_aspect_ratio_zero_height() {
        // Edge case: zero height should return 1.0 to avoid division by zero
        let params = LodParams::new([0.0, 0.0, 0.0], 1920.0, 0.0, 0.7);
        let aspect = params.aspect_ratio();
        assert_eq!(aspect, 1.0);
    }

    #[test]
    fn test_lod_params_aspect_ratio_tiny_height() {
        // Edge case: tiny height should return 1.0
        let params = LodParams::new([0.0, 0.0, 0.0], 1920.0, 1e-8, 0.7);
        let aspect = params.aspect_ratio();
        assert_eq!(aspect, 1.0);
    }

    #[test]
    fn test_lod_config() {
        let config = LodConfig::default();
        assert!(!config.use_screen_size);

        let distances = LodDistances::new(5.0, 15.0, 30.0);

        let config = LodConfig::distance_based(distances);
        assert!(!config.use_screen_size);
        assert_eq!(config.distances.thresholds[0], 5.0);

        let config = LodConfig::screen_size_based(distances);
        assert!(config.use_screen_size);
    }

    #[test]
    fn test_select_lod() {
        let params = LodParams::standard_1080p([0.0, 0.0, 0.0]);
        let distances = LodDistances::new(10.0, 25.0, 50.0);

        // Distance-based
        let config = LodConfig::distance_based(distances);
        let lod = select_lod(&params, [5.0, 0.0, 0.0], 1.0, &config);
        assert_eq!(lod, 0); // 5 meters, LOD 0

        let lod = select_lod(&params, [15.0, 0.0, 0.0], 1.0, &config);
        assert_eq!(lod, 1); // 15 meters, LOD 1

        // Screen-size-based
        let config = LodConfig::screen_size_based(distances);
        // Large object close should be LOD 0
        let lod = select_lod(&params, [2.0, 0.0, 0.0], 2.0, &config);
        assert_eq!(lod, 0);
    }

    #[test]
    fn test_bytemuck_traits() {
        // LodDistances
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let bytes: &[u8] = bytemuck::bytes_of(&distances);
        assert_eq!(bytes.len(), LOD_DISTANCES_SIZE);

        let distances_roundtrip: &LodDistances = bytemuck::from_bytes(bytes);
        assert_eq!(*distances_roundtrip, distances);

        // LodParams
        let params = LodParams::standard_1080p([1.0, 2.0, 3.0]);
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), LOD_PARAMS_SIZE);

        let params_roundtrip: &LodParams = bytemuck::from_bytes(bytes);
        assert_eq!(*params_roundtrip, params);
    }

    #[test]
    fn test_bytemuck_roundtrip_special_values() {
        // Test with special float values
        let distances = LodDistances::new(f32::MIN_POSITIVE, f32::MAX / 2.0, 1.0);
        let bytes: &[u8] = bytemuck::bytes_of(&distances);
        let roundtrip: &LodDistances = bytemuck::from_bytes(bytes);
        assert_eq!(*roundtrip, distances);
    }

    #[test]
    fn test_bytemuck_zeroed() {
        let zeroed_distances: LodDistances = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed_distances.thresholds, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed_distances._pad, 0.0);

        let zeroed_params: LodParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed_params.camera_position, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed_params.screen_width, 0.0);
        assert_eq!(zeroed_params.screen_height, 0.0);
        assert_eq!(zeroed_params.fov_y, 0.0);
    }

    #[test]
    fn test_bytemuck_slice_cast() {
        // Test slice casting for GPU buffer uploads
        let distances_array = [
            LodDistances::new(10.0, 25.0, 50.0),
            LodDistances::new(5.0, 15.0, 30.0),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&distances_array);
        assert_eq!(bytes.len(), LOD_DISTANCES_SIZE * 2);

        // Cast back
        let roundtrip: &[LodDistances] = bytemuck::cast_slice(bytes);
        assert_eq!(roundtrip, &distances_array);
    }

    // =========================================================================
    // EDGE CASE AND STRESS TESTS
    // =========================================================================

    #[test]
    fn test_lod_selection_consistency() {
        // Verify that squared and non-squared selection give same results
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let squared = squared_thresholds(&distances);

        for dist in [0.0, 5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 100.0] {
            let lod_normal = select_lod_by_distance(dist, &distances);
            let lod_squared = select_lod_by_distance_squared(dist * dist, &squared);
            assert_eq!(
                lod_normal, lod_squared,
                "LOD mismatch at distance {}: normal={}, squared={}",
                dist, lod_normal, lod_squared
            );
        }
    }

    #[test]
    fn test_screen_coverage_consistency_with_lod_selection() {
        let camera_pos = [0.0, 0.0, 0.0];
        let fov_y = std::f32::consts::FRAC_PI_4;
        let screen_height = 1080.0;

        // Close object should give LOD 0
        let coverage_close = screen_coverage(camera_pos, [1.0, 0.0, 0.0], 1.0, fov_y, screen_height);
        assert!(coverage_close >= COVERAGE_LOD0);
        assert_eq!(select_lod_by_coverage(coverage_close), 0);

        // Far object should give higher LOD (coverage decreases with distance)
        // At 100m with radius 1 and 45-degree FOV, coverage is ~0.024 (between LOD1 and LOD2 thresholds)
        let coverage_far = screen_coverage(camera_pos, [100.0, 0.0, 0.0], 1.0, fov_y, screen_height);
        assert!(coverage_far < COVERAGE_LOD0); // Less than 10%
        assert!(coverage_far > 0.0);
        let lod_far = select_lod_by_coverage(coverage_far);
        assert!(lod_far > 0, "Far object should not be LOD 0");

        // Very far object should give LOD 3
        let coverage_very_far = screen_coverage(camera_pos, [1000.0, 0.0, 0.0], 1.0, fov_y, screen_height);
        assert!(coverage_very_far < COVERAGE_LOD2); // Less than 1%
        assert_eq!(select_lod_by_coverage(coverage_very_far), 3);
    }

    #[test]
    fn test_lod_level_type() {
        let lod: LodLevel = 3;
        assert_eq!(lod, 3u8);
        assert!(lod < MAX_LOD_LEVELS);
    }

    #[test]
    fn test_all_lod_levels_reachable() {
        let thresholds = LodDistances::new(10.0, 25.0, 50.0);

        // Verify all 4 LOD levels are reachable
        let mut levels_reached = [false; 4];

        for dist in [0.0, 5.0, 15.0, 35.0, 100.0] {
            let lod = select_lod_by_distance(dist, &thresholds);
            levels_reached[lod as usize] = true;
        }

        assert!(levels_reached.iter().all(|&x| x), "Not all LOD levels reachable");
    }

    #[test]
    fn test_lod_params_chain_builders() {
        // Test chaining all builders together
        let params = LodParams::default()
            .with_camera_position([1.0, 2.0, 3.0])
            .with_screen_size(2560.0, 1440.0)
            .with_fov_y(1.2)
            .with_camera_position([4.0, 5.0, 6.0]); // Override

        assert_eq!(params.camera_position, [4.0, 5.0, 6.0]);
        assert_eq!(params.screen_width, 2560.0);
        assert_eq!(params.screen_height, 1440.0);
        assert_eq!(params.fov_y, 1.2);
    }

    #[test]
    fn test_extreme_float_values() {
        // Test with extreme but valid float values
        let distances = LodDistances::new(f32::MIN_POSITIVE, 1e20, f32::MAX / 2.0);
        assert_eq!(distances.threshold(0), f32::MIN_POSITIVE);
        assert_eq!(distances.threshold(1), 1e20);

        // Selection with extreme values
        assert_eq!(select_lod_by_distance(0.0, &distances), 0);
        assert_eq!(select_lod_by_distance(f32::MAX, &distances), 3);
    }

    #[test]
    fn test_lod_distances_debug_display() {
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let debug_str = format!("{:?}", distances);
        assert!(debug_str.contains("LodDistances"));
        assert!(debug_str.contains("10.0"));
    }

    #[test]
    fn test_lod_params_debug_display() {
        let params = LodParams::standard_1080p([1.0, 2.0, 3.0]);
        let debug_str = format!("{:?}", params);
        assert!(debug_str.contains("LodParams"));
        assert!(debug_str.contains("1920.0"));
    }

    #[test]
    fn test_lod_config_debug_display() {
        let config = LodConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("LodConfig"));
    }

    #[test]
    fn test_lod_distances_clone() {
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let cloned = distances.clone();
        assert_eq!(distances, cloned);
    }

    #[test]
    fn test_lod_params_clone() {
        let params = LodParams::standard_1080p([1.0, 2.0, 3.0]);
        let cloned = params.clone();
        assert_eq!(params, cloned);
    }

    #[test]
    fn test_lod_config_clone() {
        let config = LodConfig::screen_size_based(LodDistances::default());
        let cloned = config.clone();
        assert_eq!(config.use_screen_size, cloned.use_screen_size);
        assert_eq!(config.distances, cloned.distances);
    }

    #[test]
    fn test_lod_distances_copy() {
        let distances = LodDistances::new(10.0, 25.0, 50.0);
        let copied: LodDistances = distances; // Copy, not move
        assert_eq!(distances, copied);
    }

    #[test]
    fn test_lod_params_copy() {
        let params = LodParams::standard_1080p([1.0, 2.0, 3.0]);
        let copied: LodParams = params; // Copy, not move
        assert_eq!(params, copied);
    }

    #[test]
    fn test_lod_config_copy() {
        let config = LodConfig::default();
        let copied: LodConfig = config; // Copy, not move
        assert_eq!(config.use_screen_size, copied.use_screen_size);
    }

    #[test]
    fn test_lod_distances_equality() {
        let d1 = LodDistances::new(10.0, 25.0, 50.0);
        let d2 = LodDistances::new(10.0, 25.0, 50.0);
        let d3 = LodDistances::new(10.0, 25.0, 51.0);

        assert_eq!(d1, d2);
        assert_ne!(d1, d3);
    }

    #[test]
    fn test_lod_params_equality() {
        let p1 = LodParams::standard_1080p([1.0, 2.0, 3.0]);
        let p2 = LodParams::standard_1080p([1.0, 2.0, 3.0]);
        let p3 = LodParams::standard_4k([1.0, 2.0, 3.0]);

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }
}
