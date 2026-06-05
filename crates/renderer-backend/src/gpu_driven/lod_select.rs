//! LOD Selection Compute Pipeline (T-WGPU-P6.5.2).
//!
//! This module provides GPU-based Level of Detail (LOD) selection for objects.
//! It supports both distance-based and screen-size-based LOD selection with
//! optional blend factors for smooth LOD transitions.
//!
//! # Overview
//!
//! LOD selection reduces rendering cost by displaying lower-detail meshes
//! for distant or small objects. This compute shader processes objects in
//! parallel to determine the appropriate LOD level (0-3).
//!
//! ```text
//! +------------------+     +-------------------+     +------------------+
//! | LodSelectParams  |---->|                   |     | LodSelectOutput  |
//! | (48 bytes)       |     |  LOD Select       |---->| level: u32       |
//! +------------------+     |  ComputePipeline  |     | blend_factor: f32|
//!                          |  (64 threads/wg)  |     +------------------+
//! +------------------+     |                   |
//! | ObjectLodInput[] |---->|   1. Flag check   |
//! | (48 bytes each)  |     |   2. Distance/cov |
//! +------------------+     |   3. LOD select   |
//!                          |   4. Blend calc   |
//!                          +-------------------+
//! ```
//!
//! # Features
//!
//! - **Distance-based LOD**: Select LOD based on distance from camera
//! - **Screen-size LOD**: Select LOD based on projected screen coverage
//! - **4 LOD levels**: LOD 0 (highest) through LOD 3 (lowest detail)
//! - **Blend factors**: Optional smooth transition factors for LOD blending
//! - **Per-object flags**: Force LOD, always LOD0/LOD3, disable blend
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::lod_select::{
//!     LodSelectPipeline, LodSelectParams, ObjectLodInput, LodSelectOutput,
//!     SelectionMode,
//! };
//!
//! // Create pipeline
//! let pipeline = LodSelectPipeline::new(&device);
//!
//! // Set up parameters
//! let params = LodSelectParams::new(
//!     camera_position,
//!     1920.0, 1080.0,
//!     std::f32::consts::FRAC_PI_4,
//!     SelectionMode::Distance,
//!     object_count,
//! );
//!
//! // Dispatch
//! pipeline.dispatch(&mut encoder, &device, &params_buffer, &input_buffer, &output_buffer, object_count);
//! ```
//!
//! # Performance
//!
//! - Workgroup size: 64 threads per workgroup
//! - One thread per object
//! - Minimal memory bandwidth: 48 bytes in, 8 bytes out per object
//! - Target: < 0.05ms for 100K objects on modern GPU

use bytemuck::{Pod, Zeroable};
use std::mem;

use super::lod::{
    LodDistances, LodParams, DEFAULT_LOD0_DISTANCE, DEFAULT_LOD1_DISTANCE, DEFAULT_LOD2_DISTANCE,
    COVERAGE_LOD0, COVERAGE_LOD1, COVERAGE_LOD2,
};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 64;

/// Size of [`LodSelectParams`] struct in bytes.
pub const LOD_SELECT_PARAMS_SIZE: usize = 48;

/// Size of [`ObjectLodInput`] struct in bytes.
pub const OBJECT_LOD_INPUT_SIZE: usize = 48;

/// Size of [`LodSelectOutput`] struct in bytes.
pub const LOD_SELECT_OUTPUT_SIZE: usize = 8;

/// Maximum number of LOD levels supported.
pub const MAX_LOD_LEVELS: u32 = 4;

/// Default blend range (fraction of distance to threshold).
pub const DEFAULT_BLEND_RANGE: f32 = 0.2;

/// Embedded LOD selection shader source (T-WGPU-P6.5.2).
pub const LOD_SELECT_SHADER: &str = include_str!("../../shaders/lod_select.wgsl");

// =============================================================================
// SELECTION MODE
// =============================================================================

/// LOD selection mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum SelectionMode {
    /// Distance-based LOD selection (default).
    #[default]
    Distance = 0,
    /// Screen-size-based LOD selection.
    ScreenSize = 1,
}

impl SelectionMode {
    /// Convert to u32 for GPU upload.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

// =============================================================================
// OBJECT FLAGS
// =============================================================================

/// Per-object LOD flags.
pub mod object_lod_flags {
    /// Force LOD to the value in `forced_lod` field.
    pub const FORCE_LOD: u32 = 1 << 0;
    /// Always use LOD 0 (highest detail).
    pub const ALWAYS_LOD0: u32 = 1 << 1;
    /// Always use LOD 3 (lowest detail).
    pub const ALWAYS_LOD3: u32 = 1 << 2;
    /// Disable blend factor calculation for this object.
    pub const DISABLE_BLEND: u32 = 1 << 3;
    /// Default flags (none set).
    pub const DEFAULT: u32 = 0;
}

// =============================================================================
// LOD SELECT PARAMS
// =============================================================================

/// LOD selection parameters (uniform buffer).
///
/// Contains camera information and selection configuration.
///
/// # Memory Layout (48 bytes, 16-byte aligned)
///
/// | Offset | Field           | Size | Description                      |
/// |--------|-----------------|------|----------------------------------|
/// | 0      | camera_position | 12   | Camera world position            |
/// | 12     | _pad0           | 4    | Padding for vec4 alignment       |
/// | 16     | screen_width    | 4    | Screen width in pixels           |
/// | 20     | screen_height   | 4    | Screen height in pixels          |
/// | 24     | fov_y           | 4    | Vertical FOV in radians          |
/// | 28     | selection_mode  | 4    | 0=distance, 1=screen-size        |
/// | 32     | object_count    | 4    | Number of objects to process     |
/// | 36     | enable_blend    | 4    | 1=calc blend factor, 0=skip      |
/// | 40     | blend_range     | 4    | Blend distance (% of threshold)  |
/// | 44     | _pad1           | 4    | Padding for 16-byte alignment    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct LodSelectParams {
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
    /// Selection mode (0=distance, 1=screen-size).
    pub selection_mode: u32,
    /// Number of objects to process.
    pub object_count: u32,
    /// Enable blend factor calculation (1=yes, 0=no).
    pub enable_blend: u32,
    /// Blend range as fraction of distance to threshold (0.0-1.0).
    pub blend_range: f32,
    /// Padding for 16-byte alignment.
    pub _pad1: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<LodSelectParams>() == LOD_SELECT_PARAMS_SIZE);

impl LodSelectParams {
    /// Size of this struct in bytes.
    pub const SIZE: usize = LOD_SELECT_PARAMS_SIZE;

    /// Create new LOD selection parameters.
    ///
    /// # Arguments
    ///
    /// * `camera_position` - Camera position in world space.
    /// * `screen_width` - Screen width in pixels.
    /// * `screen_height` - Screen height in pixels.
    /// * `fov_y` - Vertical field of view in radians.
    /// * `mode` - LOD selection mode.
    /// * `object_count` - Number of objects to process.
    pub fn new(
        camera_position: [f32; 3],
        screen_width: f32,
        screen_height: f32,
        fov_y: f32,
        mode: SelectionMode,
        object_count: u32,
    ) -> Self {
        Self {
            camera_position,
            _pad0: 0.0,
            screen_width,
            screen_height,
            fov_y,
            selection_mode: mode.as_u32(),
            object_count,
            enable_blend: 0,
            blend_range: DEFAULT_BLEND_RANGE,
            _pad1: 0.0,
        }
    }

    /// Create parameters from existing [`LodParams`].
    pub fn from_lod_params(params: &LodParams, mode: SelectionMode, object_count: u32) -> Self {
        Self::new(
            params.camera_position,
            params.screen_width,
            params.screen_height,
            params.fov_y,
            mode,
            object_count,
        )
    }

    /// Create distance-based LOD selection parameters.
    pub fn distance_based(
        camera_position: [f32; 3],
        screen_width: f32,
        screen_height: f32,
        fov_y: f32,
        object_count: u32,
    ) -> Self {
        Self::new(
            camera_position,
            screen_width,
            screen_height,
            fov_y,
            SelectionMode::Distance,
            object_count,
        )
    }

    /// Create screen-size-based LOD selection parameters.
    pub fn screen_size_based(
        camera_position: [f32; 3],
        screen_width: f32,
        screen_height: f32,
        fov_y: f32,
        object_count: u32,
    ) -> Self {
        Self::new(
            camera_position,
            screen_width,
            screen_height,
            fov_y,
            SelectionMode::ScreenSize,
            object_count,
        )
    }

    /// Enable blend factor calculation.
    #[inline]
    pub fn with_blend(mut self, blend_range: f32) -> Self {
        self.enable_blend = 1;
        self.blend_range = blend_range.clamp(0.0, 1.0);
        self
    }

    /// Disable blend factor calculation.
    #[inline]
    pub fn without_blend(mut self) -> Self {
        self.enable_blend = 0;
        self
    }

    /// Update object count.
    #[inline]
    pub fn with_object_count(mut self, count: u32) -> Self {
        self.object_count = count;
        self
    }

    /// Update camera position.
    #[inline]
    pub fn with_camera_position(mut self, position: [f32; 3]) -> Self {
        self.camera_position = position;
        self
    }

    /// Check if blending is enabled.
    #[inline]
    pub fn is_blend_enabled(&self) -> bool {
        self.enable_blend != 0
    }

    /// Get selection mode.
    #[inline]
    pub fn selection_mode(&self) -> SelectionMode {
        if self.selection_mode == 1 {
            SelectionMode::ScreenSize
        } else {
            SelectionMode::Distance
        }
    }
}

// =============================================================================
// OBJECT LOD INPUT
// =============================================================================

/// Per-object LOD input data (storage buffer element).
///
/// Contains object position, bounding radius, LOD thresholds, and flags.
///
/// # Memory Layout (48 bytes, 16-byte aligned)
///
/// | Offset | Field          | Size | Description                       |
/// |--------|----------------|------|-----------------------------------|
/// | 0      | world_position | 12   | Object center in world space      |
/// | 12     | bounding_radius| 4    | Bounding sphere radius            |
/// | 16     | thresholds     | 12   | LOD 0->1, 1->2, 2->3 distances    |
/// | 28     | _pad0          | 4    | Padding                           |
/// | 32     | flags          | 4    | Object flags (force LOD, etc.)    |
/// | 36     | forced_lod     | 4    | Forced LOD level (if flag set)    |
/// | 40     | _pad1          | 8    | Padding for 16-byte alignment     |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ObjectLodInput {
    /// Object center position in world space.
    pub world_position: [f32; 3],
    /// Bounding sphere radius.
    pub bounding_radius: f32,
    /// LOD distance thresholds [0->1, 1->2, 2->3].
    pub thresholds: [f32; 3],
    /// Padding for alignment.
    pub _pad0: f32,
    /// Object flags (see `object_lod_flags`).
    pub flags: u32,
    /// Forced LOD level (used when FORCE_LOD flag is set).
    pub forced_lod: u32,
    /// Padding for 16-byte alignment.
    pub _pad1: [f32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ObjectLodInput>() == OBJECT_LOD_INPUT_SIZE);

impl ObjectLodInput {
    /// Size of this struct in bytes.
    pub const SIZE: usize = OBJECT_LOD_INPUT_SIZE;

    /// Create a new object LOD input.
    ///
    /// # Arguments
    ///
    /// * `position` - Object center in world space.
    /// * `radius` - Bounding sphere radius.
    /// * `thresholds` - LOD distance thresholds [0->1, 1->2, 2->3].
    #[inline]
    pub fn new(position: [f32; 3], radius: f32, thresholds: [f32; 3]) -> Self {
        Self {
            world_position: position,
            bounding_radius: radius,
            thresholds,
            _pad0: 0.0,
            flags: object_lod_flags::DEFAULT,
            forced_lod: 0,
            _pad1: [0.0; 2],
        }
    }

    /// Create with default LOD thresholds.
    #[inline]
    pub fn with_default_thresholds(position: [f32; 3], radius: f32) -> Self {
        Self::new(
            position,
            radius,
            [DEFAULT_LOD0_DISTANCE, DEFAULT_LOD1_DISTANCE, DEFAULT_LOD2_DISTANCE],
        )
    }

    /// Create from [`LodDistances`].
    #[inline]
    pub fn from_lod_distances(
        position: [f32; 3],
        radius: f32,
        distances: &LodDistances,
    ) -> Self {
        Self::new(position, radius, distances.thresholds)
    }

    /// Set force LOD flag with a specific LOD level.
    #[inline]
    pub fn with_forced_lod(mut self, lod: u32) -> Self {
        self.flags |= object_lod_flags::FORCE_LOD;
        self.forced_lod = lod.min(MAX_LOD_LEVELS - 1);
        self
    }

    /// Set always LOD 0 flag.
    #[inline]
    pub fn with_always_lod0(mut self) -> Self {
        self.flags |= object_lod_flags::ALWAYS_LOD0;
        self
    }

    /// Set always LOD 3 flag.
    #[inline]
    pub fn with_always_lod3(mut self) -> Self {
        self.flags |= object_lod_flags::ALWAYS_LOD3;
        self
    }

    /// Disable blend factor for this object.
    #[inline]
    pub fn without_blend(mut self) -> Self {
        self.flags |= object_lod_flags::DISABLE_BLEND;
        self
    }

    /// Clear all flags.
    #[inline]
    pub fn with_default_flags(mut self) -> Self {
        self.flags = object_lod_flags::DEFAULT;
        self
    }

    /// Check if force LOD is enabled.
    #[inline]
    pub fn is_forced(&self) -> bool {
        self.flags & object_lod_flags::FORCE_LOD != 0
    }

    /// Update position.
    #[inline]
    pub fn with_position(mut self, position: [f32; 3]) -> Self {
        self.world_position = position;
        self
    }

    /// Update radius.
    #[inline]
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.bounding_radius = radius;
        self
    }
}

// =============================================================================
// LOD SELECT OUTPUT
// =============================================================================

/// LOD selection output per object (storage buffer element).
///
/// Contains the selected LOD level and optional blend factor.
///
/// # Memory Layout (8 bytes)
///
/// | Offset | Field        | Size | Description                        |
/// |--------|--------------|------|------------------------------------|
/// | 0      | level        | 4    | Selected LOD level (0-3)           |
/// | 4      | blend_factor | 4    | Transition blend factor (0.0-1.0)  |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct LodSelectOutput {
    /// Selected LOD level (0 = highest detail, 3 = lowest detail).
    pub level: u32,
    /// Blend factor for smooth LOD transitions (0.0 to 1.0).
    ///
    /// - 0.0: Fully at current LOD
    /// - 1.0: At transition boundary to next LOD
    pub blend_factor: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<LodSelectOutput>() == LOD_SELECT_OUTPUT_SIZE);

impl LodSelectOutput {
    /// Size of this struct in bytes.
    pub const SIZE: usize = LOD_SELECT_OUTPUT_SIZE;

    /// Create a new LOD output.
    #[inline]
    pub const fn new(level: u32, blend_factor: f32) -> Self {
        Self { level, blend_factor }
    }

    /// Create output with just a level (no blend).
    #[inline]
    pub const fn level_only(level: u32) -> Self {
        Self {
            level,
            blend_factor: 0.0,
        }
    }

    /// Check if the object is at the highest detail level.
    #[inline]
    pub fn is_highest_detail(&self) -> bool {
        self.level == 0
    }

    /// Check if the object is at the lowest detail level.
    #[inline]
    pub fn is_lowest_detail(&self) -> bool {
        self.level >= 3
    }

    /// Check if the object is transitioning between LOD levels.
    #[inline]
    pub fn is_transitioning(&self) -> bool {
        self.blend_factor > 0.0 && self.blend_factor < 1.0
    }

    /// Get the next LOD level (for blending).
    #[inline]
    pub fn next_level(&self) -> u32 {
        (self.level + 1).min(3)
    }
}

// =============================================================================
// LOD BUFFER
// =============================================================================

/// Manages LOD output buffer.
///
/// Provides utilities for creating and managing the LOD selection output buffer.
#[derive(Debug)]
pub struct LodBuffer {
    /// Maximum object count this buffer can hold.
    capacity: u32,
}

impl LodBuffer {
    /// Create a new LOD buffer descriptor.
    ///
    /// # Arguments
    ///
    /// * `object_count` - Maximum number of objects.
    pub fn new(object_count: u32) -> Self {
        Self {
            capacity: object_count,
        }
    }

    /// Get the buffer capacity.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Calculate required buffer size in bytes.
    #[inline]
    pub fn buffer_size(&self) -> u64 {
        (self.capacity as u64) * (LOD_SELECT_OUTPUT_SIZE as u64)
    }

    /// Calculate buffer size for a given object count.
    #[inline]
    pub fn size_for_objects(count: u32) -> u64 {
        (count as u64) * (LOD_SELECT_OUTPUT_SIZE as u64)
    }
}

// =============================================================================
// DISPATCH HELPERS
// =============================================================================

/// Calculate the number of workgroups needed for a given object count.
///
/// Each workgroup processes 64 objects.
#[inline]
pub fn workgroups_for_objects(object_count: u32) -> u32 {
    (object_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
}

/// Calculate dispatch parameters for LOD selection.
///
/// Returns (workgroups_x, workgroups_y, workgroups_z).
#[inline]
pub fn calculate_dispatch(object_count: u32) -> (u32, u32, u32) {
    (workgroups_for_objects(object_count), 1, 1)
}

// =============================================================================
// CPU REFERENCE IMPLEMENTATIONS
// =============================================================================

/// Small epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

/// CPU reference: Calculate distance from camera to object.
#[inline]
pub fn cpu_distance_to_camera(camera_pos: [f32; 3], object_pos: [f32; 3]) -> f32 {
    let dx = object_pos[0] - camera_pos[0];
    let dy = object_pos[1] - camera_pos[1];
    let dz = object_pos[2] - camera_pos[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// CPU reference: Calculate screen coverage.
#[inline]
pub fn cpu_screen_coverage(
    camera_pos: [f32; 3],
    object_pos: [f32; 3],
    radius: f32,
    fov_y: f32,
) -> f32 {
    let distance = cpu_distance_to_camera(camera_pos, object_pos);

    if distance < EPSILON {
        return 1.0;
    }

    let half_fov = fov_y * 0.5;
    let tan_half_fov = half_fov.tan();

    if tan_half_fov < EPSILON {
        return 1.0;
    }

    let visible_height = 2.0 * distance * tan_half_fov;
    let diameter = 2.0 * radius;

    diameter / visible_height
}

/// CPU reference: Select LOD by distance.
#[inline]
pub fn cpu_select_lod_by_distance(distance: f32, thresholds: &[f32; 3]) -> u32 {
    if distance < thresholds[0] {
        0
    } else if distance < thresholds[1] {
        1
    } else if distance < thresholds[2] {
        2
    } else {
        3
    }
}

/// CPU reference: Select LOD by screen coverage.
#[inline]
pub fn cpu_select_lod_by_coverage(coverage: f32) -> u32 {
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

/// CPU reference: Process a single object LOD selection.
///
/// Returns the LOD level and blend factor.
pub fn cpu_select_lod(
    params: &LodSelectParams,
    object: &ObjectLodInput,
) -> LodSelectOutput {
    // Check forced LOD flags
    if object.flags & object_lod_flags::FORCE_LOD != 0 {
        return LodSelectOutput::level_only(object.forced_lod.min(3));
    }
    if object.flags & object_lod_flags::ALWAYS_LOD0 != 0 {
        return LodSelectOutput::level_only(0);
    }
    if object.flags & object_lod_flags::ALWAYS_LOD3 != 0 {
        return LodSelectOutput::level_only(3);
    }

    let level = if params.selection_mode == 1 {
        // Screen-size based
        let coverage = cpu_screen_coverage(
            params.camera_position,
            object.world_position,
            object.bounding_radius,
            params.fov_y,
        );
        cpu_select_lod_by_coverage(coverage)
    } else {
        // Distance based
        let distance = cpu_distance_to_camera(params.camera_position, object.world_position);
        cpu_select_lod_by_distance(distance, &object.thresholds)
    };

    // Blend factor calculation not implemented in CPU reference (GPU does this)
    LodSelectOutput::new(level, 0.0)
}

/// CPU reference: Process multiple objects.
pub fn cpu_select_lod_batch(
    params: &LodSelectParams,
    objects: &[ObjectLodInput],
    output: &mut [LodSelectOutput],
) {
    for (i, obj) in objects.iter().enumerate() {
        if i >= output.len() {
            break;
        }
        output[i] = cpu_select_lod(params, obj);
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_4;

    // =========================================================================
    // STRUCT SIZE AND LAYOUT TESTS
    // =========================================================================

    #[test]
    fn test_lod_select_params_size() {
        assert_eq!(
            std::mem::size_of::<LodSelectParams>(),
            LOD_SELECT_PARAMS_SIZE,
            "LodSelectParams must be {} bytes",
            LOD_SELECT_PARAMS_SIZE
        );
        assert_eq!(LodSelectParams::SIZE, 48);
    }

    #[test]
    fn test_object_lod_input_size() {
        assert_eq!(
            std::mem::size_of::<ObjectLodInput>(),
            OBJECT_LOD_INPUT_SIZE,
            "ObjectLodInput must be {} bytes",
            OBJECT_LOD_INPUT_SIZE
        );
        assert_eq!(ObjectLodInput::SIZE, 48);
    }

    #[test]
    fn test_lod_select_output_size() {
        assert_eq!(
            std::mem::size_of::<LodSelectOutput>(),
            LOD_SELECT_OUTPUT_SIZE,
            "LodSelectOutput must be {} bytes",
            LOD_SELECT_OUTPUT_SIZE
        );
        assert_eq!(LodSelectOutput::SIZE, 8);
    }

    #[test]
    fn test_shader_compiles() {
        // Verify shader source is available and non-empty
        assert!(!LOD_SELECT_SHADER.is_empty());
        assert!(LOD_SELECT_SHADER.contains("lod_select_main"));
        assert!(LOD_SELECT_SHADER.contains("@compute"));
        assert!(LOD_SELECT_SHADER.contains("@workgroup_size(64"));
    }

    #[test]
    fn test_lod_select_params_field_offsets() {
        let params = LodSelectParams::new(
            [1.0, 2.0, 3.0],
            1920.0,
            1080.0,
            FRAC_PI_4,
            SelectionMode::Distance,
            100,
        );
        let bytes = bytemuck::bytes_of(&params);

        // camera_position at offset 0 (12 bytes)
        let cam_x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(cam_x, 1.0);

        // screen_width at offset 16
        let width = f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        assert_eq!(width, 1920.0);

        // screen_height at offset 20
        let height = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        assert_eq!(height, 1080.0);

        // object_count at offset 32
        let count = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        assert_eq!(count, 100);
    }

    #[test]
    fn test_lod_select_output_field_offsets() {
        let output = LodSelectOutput::new(2, 0.5);
        let bytes = bytemuck::bytes_of(&output);

        // level at offset 0
        let level = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(level, 2);

        // blend_factor at offset 4
        let blend = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert!((blend - 0.5).abs() < EPSILON);
    }

    // =========================================================================
    // LOD SELECTION TESTS
    // =========================================================================

    #[test]
    fn test_distance_based_selection() {
        let params = LodSelectParams::distance_based(
            [0.0, 0.0, 0.0],
            1920.0,
            1080.0,
            FRAC_PI_4,
            1,
        );

        // LOD 0: distance < 10
        let obj = ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0]);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 0);

        // LOD 1: 10 <= distance < 25
        let obj = ObjectLodInput::new([15.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0]);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 1);

        // LOD 2: 25 <= distance < 50
        let obj = ObjectLodInput::new([35.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0]);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 2);

        // LOD 3: distance >= 50
        let obj = ObjectLodInput::new([100.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0]);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 3);
    }

    #[test]
    fn test_screen_size_selection() {
        let params = LodSelectParams::screen_size_based(
            [0.0, 0.0, 0.0],
            1920.0,
            1080.0,
            FRAC_PI_4,
            1,
        );

        // Very close object (high coverage) -> LOD 0
        let obj = ObjectLodInput::new([2.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0]);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 0);

        // Very far object (low coverage) -> LOD 3
        let obj = ObjectLodInput::new([1000.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0]);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 3);
    }

    #[test]
    fn test_forced_lod() {
        let params = LodSelectParams::distance_based(
            [0.0, 0.0, 0.0],
            1920.0,
            1080.0,
            FRAC_PI_4,
            1,
        );

        // Force LOD 2 regardless of distance
        let obj = ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0])
            .with_forced_lod(2);
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 2);

        // Force LOD 0 flag
        let obj = ObjectLodInput::new([100.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0])
            .with_always_lod0();
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 0);

        // Force LOD 3 flag
        let obj = ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0])
            .with_always_lod3();
        let result = cpu_select_lod(&params, &obj);
        assert_eq!(result.level, 3);
    }

    #[test]
    fn test_blend_factor_calculation() {
        let output = LodSelectOutput::new(1, 0.5);
        assert!(output.is_transitioning());
        assert_eq!(output.next_level(), 2);

        let output = LodSelectOutput::new(0, 0.0);
        assert!(!output.is_transitioning());
        assert!(output.is_highest_detail());

        let output = LodSelectOutput::new(3, 0.0);
        assert!(output.is_lowest_detail());
    }

    // =========================================================================
    // DISPATCH TESTS
    // =========================================================================

    #[test]
    fn test_workgroups_calculation() {
        assert_eq!(workgroups_for_objects(0), 0);
        assert_eq!(workgroups_for_objects(1), 1);
        assert_eq!(workgroups_for_objects(64), 1);
        assert_eq!(workgroups_for_objects(65), 2);
        assert_eq!(workgroups_for_objects(128), 2);
        assert_eq!(workgroups_for_objects(1000), 16);
    }

    #[test]
    fn test_dispatch_calculation() {
        let (x, y, z) = calculate_dispatch(1000);
        assert_eq!(x, 16);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    // =========================================================================
    // LOD BUFFER TESTS
    // =========================================================================

    #[test]
    fn test_lod_buffer_size() {
        let buffer = LodBuffer::new(1000);
        assert_eq!(buffer.capacity(), 1000);
        assert_eq!(buffer.buffer_size(), 8000); // 1000 * 8 bytes

        assert_eq!(LodBuffer::size_for_objects(500), 4000);
    }

    // =========================================================================
    // BYTEMUCK TESTS
    // =========================================================================

    #[test]
    fn test_bytemuck_roundtrip() {
        let params = LodSelectParams::new(
            [1.0, 2.0, 3.0],
            1920.0,
            1080.0,
            FRAC_PI_4,
            SelectionMode::ScreenSize,
            100,
        );
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        let roundtrip: &LodSelectParams = bytemuck::from_bytes(bytes);
        assert_eq!(*roundtrip, params);

        let input = ObjectLodInput::new([10.0, 20.0, 30.0], 5.0, [10.0, 25.0, 50.0]);
        let bytes: &[u8] = bytemuck::bytes_of(&input);
        let roundtrip: &ObjectLodInput = bytemuck::from_bytes(bytes);
        assert_eq!(*roundtrip, input);

        let output = LodSelectOutput::new(2, 0.75);
        let bytes: &[u8] = bytemuck::bytes_of(&output);
        let roundtrip: &LodSelectOutput = bytemuck::from_bytes(bytes);
        assert_eq!(*roundtrip, output);
    }

    #[test]
    fn test_bytemuck_zeroed() {
        let zeroed_params: LodSelectParams = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed_params.camera_position, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed_params.object_count, 0);

        let zeroed_output: LodSelectOutput = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed_output.level, 0);
        assert_eq!(zeroed_output.blend_factor, 0.0);
    }

    // =========================================================================
    // CPU REFERENCE TESTS
    // =========================================================================

    #[test]
    fn test_cpu_distance() {
        let dist = cpu_distance_to_camera([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        assert!((dist - 10.0).abs() < EPSILON);

        let dist = cpu_distance_to_camera([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]);
        assert!((dist - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_cpu_screen_coverage() {
        let coverage = cpu_screen_coverage(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0], // At camera
            1.0,
            FRAC_PI_4,
        );
        assert_eq!(coverage, 1.0);

        // Coverage decreases with distance
        let coverage_close = cpu_screen_coverage(
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            1.0,
            FRAC_PI_4,
        );
        let coverage_far = cpu_screen_coverage(
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            1.0,
            FRAC_PI_4,
        );
        assert!(coverage_close > coverage_far);
    }

    #[test]
    fn test_cpu_batch() {
        let params = LodSelectParams::distance_based(
            [0.0, 0.0, 0.0],
            1920.0,
            1080.0,
            FRAC_PI_4,
            3,
        );

        let objects = [
            ObjectLodInput::new([5.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0]),
            ObjectLodInput::new([15.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0]),
            ObjectLodInput::new([100.0, 0.0, 0.0], 1.0, [10.0, 25.0, 50.0]),
        ];

        let mut output = [LodSelectOutput::default(); 3];
        cpu_select_lod_batch(&params, &objects, &mut output);

        assert_eq!(output[0].level, 0);
        assert_eq!(output[1].level, 1);
        assert_eq!(output[2].level, 3);
    }

    // =========================================================================
    // BUILDER PATTERN TESTS
    // =========================================================================

    #[test]
    fn test_params_builders() {
        let params = LodSelectParams::new(
            [0.0, 0.0, 0.0],
            1920.0,
            1080.0,
            FRAC_PI_4,
            SelectionMode::Distance,
            100,
        )
        .with_blend(0.3)
        .with_object_count(200)
        .with_camera_position([10.0, 20.0, 30.0]);

        assert!(params.is_blend_enabled());
        assert!((params.blend_range - 0.3).abs() < EPSILON);
        assert_eq!(params.object_count, 200);
        assert_eq!(params.camera_position, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_object_input_builders() {
        let obj = ObjectLodInput::with_default_thresholds([10.0, 0.0, 0.0], 5.0)
            .with_forced_lod(2)
            .without_blend();

        assert!(obj.is_forced());
        assert_eq!(obj.forced_lod, 2);
        assert!(obj.flags & object_lod_flags::DISABLE_BLEND != 0);
    }

    // =========================================================================
    // SELECTION MODE TESTS
    // =========================================================================

    #[test]
    fn test_selection_mode() {
        assert_eq!(SelectionMode::Distance.as_u32(), 0);
        assert_eq!(SelectionMode::ScreenSize.as_u32(), 1);

        let params = LodSelectParams::distance_based([0.0, 0.0, 0.0], 1920.0, 1080.0, FRAC_PI_4, 1);
        assert_eq!(params.selection_mode(), SelectionMode::Distance);

        let params = LodSelectParams::screen_size_based([0.0, 0.0, 0.0], 1920.0, 1080.0, FRAC_PI_4, 1);
        assert_eq!(params.selection_mode(), SelectionMode::ScreenSize);
    }

    // =========================================================================
    // CONSTANTS TESTS
    // =========================================================================

    #[test]
    fn test_constants() {
        assert_eq!(WORKGROUP_SIZE, 64);
        assert_eq!(MAX_LOD_LEVELS, 4);
        assert!((DEFAULT_BLEND_RANGE - 0.2).abs() < EPSILON);
    }
}
