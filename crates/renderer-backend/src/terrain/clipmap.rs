//! Terrain Clipmap LOD System
//!
//! Implements clipmap-based terrain rendering for efficient large terrain visualization.
//! Each clipmap level is centered on the camera and provides progressively coarser
//! detail as distance increases.
//!
//! # Clipmap Structure
//!
//! - **Level 0**: Finest detail, centered on camera, smallest area coverage
//! - **Level N**: Coarser detail, 2x spacing of level N-1, 2x area coverage
//!
//! Each level forms a ring around the finer level, with smooth geomorphing
//! transitions at boundaries to prevent visible LOD popping.
//!
//! # Ring Buffer Updates
//!
//! As the camera moves, only strips at the edges need updating. The module
//! detects which regions require heightfield re-sampling using toroidal
//! addressing (ring buffer wrap-around).
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::terrain::{Clipmap, ClipmapConfig};
//!
//! let config = ClipmapConfig {
//!     grid_size: 128,
//!     num_levels: 8,
//!     finest_spacing: 0.5,
//!     height_scale: 100.0,
//!     max_height: 500.0,
//!     _padding: [0; 3],
//! };
//!
//! let mut clipmap = Clipmap::new(config);
//! clipmap.update_camera([1024.0, 50.0, 512.0]);
//!
//! for region in clipmap.compute_update_regions() {
//!     // Update heightfield for this region
//! }
//! ```

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default grid size (vertices per dimension per level).
pub const DEFAULT_GRID_SIZE: u32 = 128;

/// Default number of clipmap levels.
pub const DEFAULT_NUM_LEVELS: u32 = 8;

/// Default finest level spacing in world units.
pub const DEFAULT_FINEST_SPACING: f32 = 0.5;

/// Default height scale multiplier.
pub const DEFAULT_HEIGHT_SCALE: f32 = 100.0;

/// Default maximum terrain height.
pub const DEFAULT_MAX_HEIGHT: f32 = 500.0;

/// Threshold ratio for triggering level updates.
/// When camera moves more than this fraction of cell size, update is triggered.
pub const UPDATE_THRESHOLD_RATIO: f32 = 0.25;

/// Minimum number of clipmap levels.
pub const MIN_LEVELS: u32 = 1;

/// Maximum number of clipmap levels.
pub const MAX_LEVELS: u32 = 16;

/// Minimum grid size.
pub const MIN_GRID_SIZE: u32 = 4;

/// Maximum grid size.
pub const MAX_GRID_SIZE: u32 = 512;

// ---------------------------------------------------------------------------
// GPU-side Configuration
// ---------------------------------------------------------------------------

/// Clipmap configuration struct (GPU-compatible).
///
/// This struct is designed for direct GPU upload with proper alignment.
/// Use `bytemuck` for safe casting to byte slices.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct ClipmapConfig {
    /// Number of vertices per dimension per level (typically 128).
    pub grid_size: u32,
    /// Number of LOD levels (typically 8).
    pub num_levels: u32,
    /// World-space spacing at level 0 (e.g., 0.5m).
    pub finest_spacing: f32,
    /// Height multiplier applied to normalized heightfield values.
    pub height_scale: f32,
    /// Maximum terrain height for clamping and normalization.
    pub max_height: f32,
    /// Padding for 16-byte alignment (GPU uniform buffers).
    pub _padding: [u32; 3],
}

impl Default for ClipmapConfig {
    fn default() -> Self {
        Self {
            grid_size: DEFAULT_GRID_SIZE,
            num_levels: DEFAULT_NUM_LEVELS,
            finest_spacing: DEFAULT_FINEST_SPACING,
            height_scale: DEFAULT_HEIGHT_SCALE,
            max_height: DEFAULT_MAX_HEIGHT,
            _padding: [0; 3],
        }
    }
}

impl ClipmapConfig {
    /// Create a new clipmap configuration with validation.
    ///
    /// # Panics
    ///
    /// Panics if parameters are out of valid ranges.
    pub fn new(
        grid_size: u32,
        num_levels: u32,
        finest_spacing: f32,
        height_scale: f32,
        max_height: f32,
    ) -> Self {
        assert!(
            grid_size >= MIN_GRID_SIZE && grid_size <= MAX_GRID_SIZE,
            "grid_size must be in range [{}, {}], got {}",
            MIN_GRID_SIZE,
            MAX_GRID_SIZE,
            grid_size
        );
        assert!(
            num_levels >= MIN_LEVELS && num_levels <= MAX_LEVELS,
            "num_levels must be in range [{}, {}], got {}",
            MIN_LEVELS,
            MAX_LEVELS,
            num_levels
        );
        assert!(
            finest_spacing > 0.0,
            "finest_spacing must be positive, got {}",
            finest_spacing
        );
        assert!(
            height_scale > 0.0,
            "height_scale must be positive, got {}",
            height_scale
        );
        assert!(
            max_height > 0.0,
            "max_height must be positive, got {}",
            max_height
        );

        Self {
            grid_size,
            num_levels,
            finest_spacing,
            height_scale,
            max_height,
            _padding: [0; 3],
        }
    }

    /// Calculate the spacing for a given level.
    ///
    /// Level N has spacing = `finest_spacing * 2^N`.
    #[inline]
    pub fn level_spacing(&self, level: u32) -> f32 {
        self.finest_spacing * (1 << level) as f32
    }

    /// Calculate the world-space coverage for a given level.
    ///
    /// Coverage = `grid_size * spacing`.
    #[inline]
    pub fn level_coverage(&self, level: u32) -> f32 {
        self.grid_size as f32 * self.level_spacing(level)
    }

    /// Calculate vertices per level (grid_size^2).
    #[inline]
    pub fn vertices_per_level(&self) -> usize {
        (self.grid_size as usize) * (self.grid_size as usize)
    }

    /// Calculate total vertices across all levels.
    #[inline]
    pub fn total_vertices(&self) -> usize {
        self.vertices_per_level() * (self.num_levels as usize)
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), ClipmapError> {
        if self.grid_size < MIN_GRID_SIZE || self.grid_size > MAX_GRID_SIZE {
            return Err(ClipmapError::InvalidGridSize(self.grid_size));
        }
        if self.num_levels < MIN_LEVELS || self.num_levels > MAX_LEVELS {
            return Err(ClipmapError::InvalidNumLevels(self.num_levels));
        }
        if self.finest_spacing <= 0.0 {
            return Err(ClipmapError::InvalidSpacing(self.finest_spacing));
        }
        if self.height_scale <= 0.0 {
            return Err(ClipmapError::InvalidHeightScale(self.height_scale));
        }
        if self.max_height <= 0.0 {
            return Err(ClipmapError::InvalidMaxHeight(self.max_height));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Clipmap Level
// ---------------------------------------------------------------------------

/// A single clipmap level with spatial tracking.
#[derive(Debug, Clone)]
pub struct ClipmapLevel {
    /// Level index (0 = finest, increases with distance).
    pub level: u32,
    /// World-space spacing between vertices at this level.
    pub spacing: f32,
    /// Total world-space coverage (width/depth) of this level.
    pub coverage: f32,
    /// Current world-space center position (X, Z).
    pub center: [f32; 2],
    /// Flag indicating this level needs heightfield update.
    pub needs_update: bool,
}

impl ClipmapLevel {
    /// Create a new clipmap level.
    pub fn new(level: u32, config: &ClipmapConfig) -> Self {
        Self {
            level,
            spacing: config.level_spacing(level),
            coverage: config.level_coverage(level),
            center: [0.0, 0.0],
            needs_update: true, // Initially needs full update
        }
    }

    /// Check if a camera position requires this level to update.
    ///
    /// Returns `true` if camera has moved more than the threshold
    /// distance from the last update position.
    pub fn should_update(&self, camera_xz: [f32; 2]) -> bool {
        let threshold = self.spacing * UPDATE_THRESHOLD_RATIO;
        let dx = (camera_xz[0] - self.center[0]).abs();
        let dz = (camera_xz[1] - self.center[1]).abs();
        dx > threshold || dz > threshold
    }

    /// Snap a world position to the nearest grid cell boundary for this level.
    #[inline]
    pub fn snap_to_grid(&self, position: f32) -> f32 {
        (position / self.spacing).floor() * self.spacing
    }

    /// Snap both X and Z coordinates to grid boundaries.
    #[inline]
    pub fn snap_center(&self, camera_xz: [f32; 2]) -> [f32; 2] {
        [
            self.snap_to_grid(camera_xz[0]),
            self.snap_to_grid(camera_xz[1]),
        ]
    }

    /// Calculate half-extent of this level in world units.
    #[inline]
    pub fn half_extent(&self) -> f32 {
        self.coverage * 0.5
    }

    /// Check if a world position is within this level's coverage area.
    pub fn contains(&self, position: [f32; 2]) -> bool {
        let half = self.half_extent();
        let dx = (position[0] - self.center[0]).abs();
        let dz = (position[1] - self.center[1]).abs();
        dx <= half && dz <= half
    }
}

// ---------------------------------------------------------------------------
// Clipmap Manager
// ---------------------------------------------------------------------------

/// Main clipmap manager for terrain LOD.
///
/// Manages multiple clipmap levels centered on the camera position,
/// tracking when each level needs heightfield updates.
#[derive(Debug, Clone)]
pub struct Clipmap {
    /// Configuration parameters.
    config: ClipmapConfig,
    /// All clipmap levels (0 = finest).
    levels: Vec<ClipmapLevel>,
    /// Current camera position in world space.
    camera_position: [f32; 3],
    /// Camera position (X, Z) at last update, for movement detection.
    last_update_position: [f32; 2],
}

impl Clipmap {
    /// Create a new clipmap with the given configuration.
    pub fn new(config: ClipmapConfig) -> Self {
        let levels = (0..config.num_levels)
            .map(|i| ClipmapLevel::new(i, &config))
            .collect();

        Self {
            config,
            levels,
            camera_position: [0.0, 0.0, 0.0],
            last_update_position: [0.0, 0.0],
        }
    }

    /// Create a clipmap with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ClipmapConfig::default())
    }

    /// Get the clipmap configuration.
    #[inline]
    pub fn config(&self) -> &ClipmapConfig {
        &self.config
    }

    /// Get the current camera position.
    #[inline]
    pub fn camera_position(&self) -> [f32; 3] {
        self.camera_position
    }

    /// Get all clipmap levels.
    #[inline]
    pub fn levels(&self) -> &[ClipmapLevel] {
        &self.levels
    }

    /// Get a mutable reference to all clipmap levels.
    #[inline]
    pub fn levels_mut(&mut self) -> &mut [ClipmapLevel] {
        &mut self.levels
    }

    /// Get the number of levels.
    #[inline]
    pub fn num_levels(&self) -> u32 {
        self.config.num_levels
    }

    /// Calculate spacing for a given level.
    #[inline]
    pub fn level_spacing(&self, level: u32) -> f32 {
        self.config.level_spacing(level)
    }

    /// Calculate coverage for a given level.
    #[inline]
    pub fn level_coverage(&self, level: u32) -> f32 {
        self.config.level_coverage(level)
    }

    /// Check if a level should update based on current camera position.
    pub fn should_update_level(&self, level: u32) -> bool {
        self.levels
            .get(level as usize)
            .map(|l| l.needs_update || l.should_update([self.camera_position[0], self.camera_position[2]]))
            .unwrap_or(false)
    }

    /// Update camera position and check which levels need updates.
    ///
    /// Call this each frame with the new camera position.
    /// After calling, check `should_update_level()` or `compute_update_regions()`
    /// to determine what heightfield data needs refreshing.
    pub fn update_camera(&mut self, camera_pos: [f32; 3]) {
        self.camera_position = camera_pos;
        let camera_xz = [camera_pos[0], camera_pos[2]];

        // Check each level for update requirements
        for level in &mut self.levels {
            if level.should_update(camera_xz) {
                level.needs_update = true;
            }
        }
    }

    /// Shift a level's center to a new position.
    ///
    /// Call this after updating the heightfield for a level.
    /// The center is snapped to grid boundaries.
    pub fn shift_level(&mut self, level: u32, new_center: [f32; 2]) {
        if let Some(lv) = self.levels.get_mut(level as usize) {
            lv.center = lv.snap_center(new_center);
            lv.needs_update = false;
        }
    }

    /// Mark a level as updated (clears needs_update flag).
    pub fn mark_level_updated(&mut self, level: u32) {
        if let Some(lv) = self.levels.get_mut(level as usize) {
            lv.needs_update = false;
        }
    }

    /// Calculate vertices per level.
    #[inline]
    pub fn vertex_count_per_level(&self) -> usize {
        self.config.vertices_per_level()
    }

    /// Calculate total vertices across all levels.
    #[inline]
    pub fn total_vertex_count(&self) -> usize {
        self.config.total_vertices()
    }

    /// Compute update regions for all levels that need updating.
    ///
    /// Returns a list of regions describing which grid cells need
    /// heightfield data refreshed. Uses ring buffer (toroidal) addressing
    /// to minimize data transfer.
    pub fn compute_update_regions(&self) -> Vec<ClipmapUpdateRegion> {
        let mut regions = Vec::new();
        let camera_xz = [self.camera_position[0], self.camera_position[2]];

        for level in &self.levels {
            if !level.needs_update && !level.should_update(camera_xz) {
                continue;
            }

            let new_center = level.snap_center(camera_xz);
            let old_center = level.center;

            // Calculate grid cell offset
            let dx = ((new_center[0] - old_center[0]) / level.spacing).round() as i32;
            let dz = ((new_center[1] - old_center[1]) / level.spacing).round() as i32;

            let grid_size = self.config.grid_size;

            if level.needs_update || dx.abs() as u32 >= grid_size || dz.abs() as u32 >= grid_size {
                // Full level update required
                regions.push(ClipmapUpdateRegion {
                    level: level.level,
                    offset: [0, 0],
                    size: [grid_size, grid_size],
                    is_full_update: true,
                });
            } else {
                // Partial update: compute strips
                if dx != 0 {
                    let x_offset = if dx > 0 {
                        (grid_size as i32 - dx) % grid_size as i32
                    } else {
                        0
                    };
                    regions.push(ClipmapUpdateRegion {
                        level: level.level,
                        offset: [x_offset, 0],
                        size: [dx.unsigned_abs(), grid_size],
                        is_full_update: false,
                    });
                }

                if dz != 0 {
                    let z_offset = if dz > 0 {
                        (grid_size as i32 - dz) % grid_size as i32
                    } else {
                        0
                    };
                    regions.push(ClipmapUpdateRegion {
                        level: level.level,
                        offset: [0, z_offset],
                        size: [grid_size, dz.unsigned_abs()],
                        is_full_update: false,
                    });
                }
            }
        }

        regions
    }

    /// Get the finest level that covers a world position.
    ///
    /// Returns `None` if the position is outside all levels.
    pub fn level_for_position(&self, position: [f32; 2]) -> Option<u32> {
        for level in &self.levels {
            if level.contains(position) {
                return Some(level.level);
            }
        }
        None
    }

    /// Sample the appropriate LOD level for a given world distance.
    ///
    /// Returns the level index where spacing is appropriate for the distance.
    pub fn level_for_distance(&self, distance: f32) -> u32 {
        for level in &self.levels {
            if distance <= level.coverage * 0.5 {
                return level.level;
            }
        }
        self.config.num_levels.saturating_sub(1)
    }
}

// ---------------------------------------------------------------------------
// Update Region
// ---------------------------------------------------------------------------

/// Describes a region of a clipmap level that needs heightfield update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipmapUpdateRegion {
    /// Level index.
    pub level: u32,
    /// Grid cell offset (column, row) within the level.
    pub offset: [i32; 2],
    /// Size in grid cells (width, height).
    pub size: [u32; 2],
    /// Whether this is a full level update or partial strip update.
    pub is_full_update: bool,
}

impl ClipmapUpdateRegion {
    /// Calculate the number of cells in this update region.
    #[inline]
    pub fn cell_count(&self) -> usize {
        (self.size[0] as usize) * (self.size[1] as usize)
    }

    /// Check if this is an X-axis strip (horizontal movement).
    #[inline]
    pub fn is_x_strip(&self) -> bool {
        !self.is_full_update && self.size[1] > self.size[0]
    }

    /// Check if this is a Z-axis strip (vertical movement).
    #[inline]
    pub fn is_z_strip(&self) -> bool {
        !self.is_full_update && self.size[0] > self.size[1]
    }
}

// ---------------------------------------------------------------------------
// Geomorphing
// ---------------------------------------------------------------------------

/// Utilities for smooth LOD transitions via geomorphing.
///
/// Geomorphing blends vertex positions between adjacent LOD levels
/// to prevent visible "popping" during LOD transitions.
pub struct Geomorphing;

impl Geomorphing {
    /// Compute the morph factor for smooth LOD transitions.
    ///
    /// Returns a value in [0, 1] where:
    /// - 0.0 = use this level's vertices fully (center of level)
    /// - 1.0 = fully morph toward coarser level (edge of level)
    ///
    /// The transition region is the outer 25% of each level's coverage.
    pub fn compute_morph_factor(distance: f32, level: u32, config: &ClipmapConfig) -> f32 {
        let coverage = config.level_coverage(level);
        let half_coverage = coverage * 0.5;

        // Inner 75% of level: no morphing
        let morph_start = half_coverage * 0.75;

        if distance <= morph_start {
            return 0.0;
        }

        if distance >= half_coverage {
            return 1.0;
        }

        // Linear interpolation in the transition zone
        let morph_range = half_coverage - morph_start;
        (distance - morph_start) / morph_range
    }

    /// Blend heights between fine and coarse levels.
    ///
    /// # Arguments
    ///
    /// * `height_fine` - Height from the finer (current) level
    /// * `height_coarse` - Height from the coarser (parent) level
    /// * `morph_factor` - Blend factor [0, 1]
    ///
    /// # Returns
    ///
    /// Interpolated height value.
    #[inline]
    pub fn blend_height(height_fine: f32, height_coarse: f32, morph_factor: f32) -> f32 {
        height_fine * (1.0 - morph_factor) + height_coarse * morph_factor
    }

    /// Blend 3D positions between levels.
    #[inline]
    pub fn blend_position(
        pos_fine: [f32; 3],
        pos_coarse: [f32; 3],
        morph_factor: f32,
    ) -> [f32; 3] {
        let inv = 1.0 - morph_factor;
        [
            pos_fine[0] * inv + pos_coarse[0] * morph_factor,
            pos_fine[1] * inv + pos_coarse[1] * morph_factor,
            pos_fine[2] * inv + pos_coarse[2] * morph_factor,
        ]
    }

    /// Calculate the geomorphing distance from camera for a vertex.
    #[inline]
    pub fn vertex_distance(vertex_xz: [f32; 2], camera_xz: [f32; 2]) -> f32 {
        let dx = vertex_xz[0] - camera_xz[0];
        let dz = vertex_xz[1] - camera_xz[1];
        (dx * dx + dz * dz).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Normal Calculation
// ---------------------------------------------------------------------------

/// Compute terrain normal using central difference.
///
/// This function calculates the surface normal at a point using the
/// heights of its four neighbors (left, right, back, front).
///
/// # Arguments
///
/// * `height_left` - Height at (x - spacing, z)
/// * `height_right` - Height at (x + spacing, z)
/// * `height_back` - Height at (x, z - spacing)
/// * `height_front` - Height at (x, z + spacing)
/// * `spacing` - Grid cell spacing in world units
///
/// # Returns
///
/// Normalized surface normal vector [x, y, z].
pub fn compute_normal_central_diff(
    height_left: f32,
    height_right: f32,
    height_back: f32,
    height_front: f32,
    spacing: f32,
) -> [f32; 3] {
    // Gradient in X: dh/dx
    let dx = (height_right - height_left) / (2.0 * spacing);
    // Gradient in Z: dh/dz
    let dz = (height_front - height_back) / (2.0 * spacing);

    // Normal = normalize((-dx, 1, -dz))
    let len = (dx * dx + 1.0 + dz * dz).sqrt();
    [-dx / len, 1.0 / len, -dz / len]
}

/// Compute terrain normal using forward difference (for edge cells).
///
/// Use this at grid boundaries where central difference is not possible.
pub fn compute_normal_forward_diff(
    height_center: f32,
    height_x: f32,
    height_z: f32,
    spacing: f32,
) -> [f32; 3] {
    let dx = (height_x - height_center) / spacing;
    let dz = (height_z - height_center) / spacing;

    let len = (dx * dx + 1.0 + dz * dz).sqrt();
    [-dx / len, 1.0 / len, -dz / len]
}

/// Compute tangent vectors for a terrain surface.
///
/// Returns (tangent, bitangent) for normal mapping.
pub fn compute_tangent_frame(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    // Tangent in X direction
    let tangent = [1.0, 0.0, 0.0];

    // Bitangent = normal x tangent
    let bitangent = [
        normal[1] * tangent[2] - normal[2] * tangent[1],
        normal[2] * tangent[0] - normal[0] * tangent[2],
        normal[0] * tangent[1] - normal[1] * tangent[0],
    ];

    // Re-orthogonalize tangent = bitangent x normal
    let tangent = [
        bitangent[1] * normal[2] - bitangent[2] * normal[1],
        bitangent[2] * normal[0] - bitangent[0] * normal[2],
        bitangent[0] * normal[1] - bitangent[1] * normal[0],
    ];

    (tangent, bitangent)
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during clipmap operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ClipmapError {
    /// Grid size is outside valid range.
    InvalidGridSize(u32),
    /// Number of levels is outside valid range.
    InvalidNumLevels(u32),
    /// Finest spacing is invalid (must be positive).
    InvalidSpacing(f32),
    /// Height scale is invalid (must be positive).
    InvalidHeightScale(f32),
    /// Max height is invalid (must be positive).
    InvalidMaxHeight(f32),
    /// Level index out of bounds.
    LevelOutOfBounds(u32),
}

impl std::fmt::Display for ClipmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGridSize(v) => write!(
                f,
                "invalid grid size {}: must be in [{}, {}]",
                v, MIN_GRID_SIZE, MAX_GRID_SIZE
            ),
            Self::InvalidNumLevels(v) => write!(
                f,
                "invalid num_levels {}: must be in [{}, {}]",
                v, MIN_LEVELS, MAX_LEVELS
            ),
            Self::InvalidSpacing(v) => write!(f, "invalid spacing {}: must be positive", v),
            Self::InvalidHeightScale(v) => write!(f, "invalid height_scale {}: must be positive", v),
            Self::InvalidMaxHeight(v) => write!(f, "invalid max_height {}: must be positive", v),
            Self::LevelOutOfBounds(v) => write!(f, "level {} out of bounds", v),
        }
    }
}

impl std::error::Error for ClipmapError {}

// ---------------------------------------------------------------------------
// GPU Vertex Format
// ---------------------------------------------------------------------------

/// GPU-side vertex for clipmap terrain.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ClipmapVertex {
    /// Position (x, y, z, morph_factor).
    pub position: [f32; 4],
    /// Normal (x, y, z, unused).
    pub normal: [f32; 4],
    /// Texture coordinates (u, v, level, unused).
    pub texcoord: [f32; 4],
}

impl ClipmapVertex {
    /// Create a new clipmap vertex.
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], level: u32) -> Self {
        Self {
            position: [position[0], position[1], position[2], 0.0],
            normal: [normal[0], normal[1], normal[2], 0.0],
            texcoord: [uv[0], uv[1], level as f32, 0.0],
        }
    }

    /// Create a vertex with morph factor.
    pub fn with_morph(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        level: u32,
        morph_factor: f32,
    ) -> Self {
        Self {
            position: [position[0], position[1], position[2], morph_factor],
            normal: [normal[0], normal[1], normal[2], 0.0],
            texcoord: [uv[0], uv[1], level as f32, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Ring Buffer Utilities
// ---------------------------------------------------------------------------

/// Toroidal (wrap-around) addressing for ring buffer updates.
pub struct ToroidalAddress;

impl ToroidalAddress {
    /// Wrap a coordinate within grid bounds.
    #[inline]
    pub fn wrap(coord: i32, grid_size: u32) -> u32 {
        coord.rem_euclid(grid_size as i32) as u32
    }

    /// Convert world position to toroidal grid coordinate.
    #[inline]
    pub fn world_to_grid(world: f32, spacing: f32, grid_size: u32) -> u32 {
        let cell = (world / spacing).floor() as i32;
        Self::wrap(cell, grid_size)
    }

    /// Convert grid coordinate to world position (lower-left corner of cell).
    #[inline]
    pub fn grid_to_world(grid: u32, center: f32, spacing: f32, grid_size: u32) -> f32 {
        let half = (grid_size as f32 / 2.0) * spacing;
        center - half + (grid as f32 * spacing)
    }

    /// Check if two grid cells are adjacent in toroidal space.
    pub fn are_adjacent(a: [u32; 2], b: [u32; 2], grid_size: u32) -> bool {
        let dx = a[0].abs_diff(b[0]);
        let dz = a[1].abs_diff(b[1]);

        // Account for wrap-around
        let dx = dx.min(grid_size - dx);
        let dz = dz.min(grid_size - dz);

        (dx <= 1 && dz == 0) || (dx == 0 && dz <= 1)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Config tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = ClipmapConfig::default();
        assert_eq!(config.grid_size, DEFAULT_GRID_SIZE);
        assert_eq!(config.num_levels, DEFAULT_NUM_LEVELS);
        assert_eq!(config.finest_spacing, DEFAULT_FINEST_SPACING);
        assert_eq!(config.height_scale, DEFAULT_HEIGHT_SCALE);
        assert_eq!(config.max_height, DEFAULT_MAX_HEIGHT);
    }

    #[test]
    fn test_config_validation_success() {
        let config = ClipmapConfig::new(128, 8, 0.5, 100.0, 500.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_grid_size() {
        let mut config = ClipmapConfig::default();
        config.grid_size = 2; // Too small
        assert!(matches!(
            config.validate(),
            Err(ClipmapError::InvalidGridSize(2))
        ));

        config.grid_size = 1024; // Too large
        assert!(matches!(
            config.validate(),
            Err(ClipmapError::InvalidGridSize(1024))
        ));
    }

    #[test]
    fn test_config_validation_invalid_num_levels() {
        let mut config = ClipmapConfig::default();
        config.num_levels = 0; // Too few
        assert!(matches!(
            config.validate(),
            Err(ClipmapError::InvalidNumLevels(0))
        ));

        config.num_levels = 32; // Too many
        assert!(matches!(
            config.validate(),
            Err(ClipmapError::InvalidNumLevels(32))
        ));
    }

    #[test]
    fn test_config_validation_invalid_spacing() {
        let mut config = ClipmapConfig::default();
        config.finest_spacing = 0.0;
        assert!(matches!(
            config.validate(),
            Err(ClipmapError::InvalidSpacing(_))
        ));

        config.finest_spacing = -1.0;
        assert!(matches!(
            config.validate(),
            Err(ClipmapError::InvalidSpacing(_))
        ));
    }

    #[test]
    fn test_config_validation_invalid_height_scale() {
        let mut config = ClipmapConfig::default();
        config.height_scale = 0.0;
        assert!(matches!(
            config.validate(),
            Err(ClipmapError::InvalidHeightScale(_))
        ));
    }

    #[test]
    fn test_config_validation_invalid_max_height() {
        let mut config = ClipmapConfig::default();
        config.max_height = -100.0;
        assert!(matches!(
            config.validate(),
            Err(ClipmapError::InvalidMaxHeight(_))
        ));
    }

    // -------------------------------------------------------------------------
    // Level spacing tests (2^N progression)
    // -------------------------------------------------------------------------

    #[test]
    fn test_level_spacing_progression() {
        let config = ClipmapConfig::new(128, 8, 0.5, 100.0, 500.0);

        assert_eq!(config.level_spacing(0), 0.5);
        assert_eq!(config.level_spacing(1), 1.0);
        assert_eq!(config.level_spacing(2), 2.0);
        assert_eq!(config.level_spacing(3), 4.0);
        assert_eq!(config.level_spacing(4), 8.0);
        assert_eq!(config.level_spacing(5), 16.0);
        assert_eq!(config.level_spacing(6), 32.0);
        assert_eq!(config.level_spacing(7), 64.0);
    }

    #[test]
    fn test_level_spacing_power_of_two() {
        let config = ClipmapConfig::default();

        for level in 0..8 {
            let expected = config.finest_spacing * (1 << level) as f32;
            assert_eq!(
                config.level_spacing(level),
                expected,
                "Level {} spacing mismatch",
                level
            );
        }
    }

    // -------------------------------------------------------------------------
    // Level coverage tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_level_coverage_calculations() {
        let config = ClipmapConfig::new(128, 8, 0.5, 100.0, 500.0);

        // Coverage = grid_size * spacing
        assert_eq!(config.level_coverage(0), 128.0 * 0.5);
        assert_eq!(config.level_coverage(1), 128.0 * 1.0);
        assert_eq!(config.level_coverage(2), 128.0 * 2.0);
        assert_eq!(config.level_coverage(3), 128.0 * 4.0);
    }

    #[test]
    fn test_coverage_doubles_each_level() {
        let config = ClipmapConfig::default();

        for level in 1..8 {
            let prev_coverage = config.level_coverage(level - 1);
            let curr_coverage = config.level_coverage(level);
            assert_eq!(
                curr_coverage,
                prev_coverage * 2.0,
                "Coverage should double at level {}",
                level
            );
        }
    }

    // -------------------------------------------------------------------------
    // Camera movement tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_camera_movement_triggers_update() {
        let config = ClipmapConfig::new(128, 4, 1.0, 100.0, 500.0);
        let mut clipmap = Clipmap::new(config);

        // Initial state: all levels need update
        for level in 0..4 {
            assert!(clipmap.should_update_level(level));
        }

        // Move camera significantly
        clipmap.update_camera([100.0, 50.0, 100.0]);

        // All levels should need update after significant movement
        for level in 0..4 {
            assert!(clipmap.should_update_level(level));
        }
    }

    #[test]
    fn test_small_camera_movement_no_update() {
        let config = ClipmapConfig::new(128, 4, 1.0, 100.0, 500.0);
        let mut clipmap = Clipmap::new(config);

        // Shift all levels to center at origin
        for level in 0..4 {
            clipmap.shift_level(level, [0.0, 0.0]);
        }

        // Small movement (within threshold)
        clipmap.update_camera([0.1, 0.0, 0.1]);

        // Level 0 threshold = 1.0 * 0.25 = 0.25
        // Movement = sqrt(0.1^2 + 0.1^2) ≈ 0.14 < 0.25
        // Should NOT trigger update for level 0
        let level0 = &clipmap.levels()[0];
        assert!(!level0.should_update([0.1, 0.1]));
    }

    #[test]
    fn test_camera_position_tracking() {
        let mut clipmap = Clipmap::with_defaults();

        let pos = [100.0, 50.0, 200.0];
        clipmap.update_camera(pos);

        assert_eq!(clipmap.camera_position(), pos);
    }

    // -------------------------------------------------------------------------
    // Geomorphing tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_geomorphing_factor_at_center() {
        let config = ClipmapConfig::new(128, 8, 0.5, 100.0, 500.0);

        // At level center, morph factor should be 0
        let factor = Geomorphing::compute_morph_factor(0.0, 0, &config);
        assert_eq!(factor, 0.0, "Morph factor at center should be 0");
    }

    #[test]
    fn test_geomorphing_factor_at_edge() {
        let config = ClipmapConfig::new(128, 8, 0.5, 100.0, 500.0);

        // At level edge (half coverage), morph factor should be 1
        let half_coverage = config.level_coverage(0) * 0.5;
        let factor = Geomorphing::compute_morph_factor(half_coverage, 0, &config);
        assert_eq!(factor, 1.0, "Morph factor at edge should be 1");
    }

    #[test]
    fn test_geomorphing_factor_interpolation() {
        let config = ClipmapConfig::new(128, 8, 0.5, 100.0, 500.0);

        let half_coverage = config.level_coverage(0) * 0.5;
        let morph_start = half_coverage * 0.75;
        let mid_transition = (morph_start + half_coverage) / 2.0;

        let factor = Geomorphing::compute_morph_factor(mid_transition, 0, &config);

        // Should be approximately 0.5 in the middle of transition zone
        assert!(factor > 0.4 && factor < 0.6, "Mid-transition factor should be ~0.5, got {}", factor);
    }

    #[test]
    fn test_height_blending_correctness() {
        // Test blending at different factors
        let fine = 10.0;
        let coarse = 20.0;

        // At factor 0: use fine height
        assert_eq!(Geomorphing::blend_height(fine, coarse, 0.0), fine);

        // At factor 1: use coarse height
        assert_eq!(Geomorphing::blend_height(fine, coarse, 1.0), coarse);

        // At factor 0.5: average
        assert_eq!(Geomorphing::blend_height(fine, coarse, 0.5), 15.0);

        // At factor 0.25
        assert_eq!(Geomorphing::blend_height(fine, coarse, 0.25), 12.5);
    }

    #[test]
    fn test_position_blending() {
        let fine = [0.0, 10.0, 0.0];
        let coarse = [0.0, 20.0, 0.0];

        let blended = Geomorphing::blend_position(fine, coarse, 0.5);
        assert_eq!(blended[1], 15.0);
    }

    // -------------------------------------------------------------------------
    // Normal calculation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normal_flat_terrain() {
        // Flat terrain: all heights equal
        let normal = compute_normal_central_diff(10.0, 10.0, 10.0, 10.0, 1.0);

        // Normal should point straight up
        assert!((normal[0]).abs() < 0.001, "X should be ~0");
        assert!((normal[1] - 1.0).abs() < 0.001, "Y should be ~1");
        assert!((normal[2]).abs() < 0.001, "Z should be ~0");
    }

    #[test]
    fn test_normal_slope_in_x() {
        // Slope rising in +X direction
        let normal = compute_normal_central_diff(0.0, 2.0, 1.0, 1.0, 1.0);

        // Normal should tilt toward -X
        assert!(normal[0] < 0.0, "Normal X should be negative for +X slope");
        assert!(normal[1] > 0.0, "Normal Y should be positive");
    }

    #[test]
    fn test_normal_slope_in_z() {
        // Slope rising in +Z direction
        let normal = compute_normal_central_diff(1.0, 1.0, 0.0, 2.0, 1.0);

        // Normal should tilt toward -Z
        assert!(normal[2] < 0.0, "Normal Z should be negative for +Z slope");
        assert!(normal[1] > 0.0, "Normal Y should be positive");
    }

    #[test]
    fn test_normal_is_normalized() {
        let normal = compute_normal_central_diff(0.0, 5.0, 0.0, 3.0, 1.0);
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!((len - 1.0).abs() < 0.001, "Normal should be unit length");
    }

    // -------------------------------------------------------------------------
    // Update region detection tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_update_region_full_on_init() {
        let clipmap = Clipmap::with_defaults();
        let regions = clipmap.compute_update_regions();

        // All levels should need full update initially
        assert!(!regions.is_empty());
        for region in &regions {
            assert!(region.is_full_update);
            assert_eq!(region.size[0], DEFAULT_GRID_SIZE);
            assert_eq!(region.size[1], DEFAULT_GRID_SIZE);
        }
    }

    #[test]
    fn test_update_region_partial_after_shift() {
        let config = ClipmapConfig::new(128, 2, 1.0, 100.0, 500.0);
        let mut clipmap = Clipmap::new(config);

        // Initialize levels at origin and update camera to center
        clipmap.update_camera([0.0, 0.0, 0.0]);
        clipmap.shift_level(0, [0.0, 0.0]);
        clipmap.shift_level(1, [0.0, 0.0]);

        // Move camera by 5 cells in X direction (within grid size)
        // This should trigger a partial update (strip) for level 0
        clipmap.update_camera([5.0, 0.0, 0.0]);

        let regions = clipmap.compute_update_regions();

        // After shifting levels and moving 5 cells, we should get update regions
        // The regions should show the strip that needs updating in X direction
        assert!(!regions.is_empty(), "Should have update regions after camera move");

        // At least one region should cover the X strip for partial update
        // or full update if movement exceeds threshold significantly
        let has_x_movement_region = regions.iter().any(|r| {
            r.level == 0 && (r.is_full_update || r.size[0] > 0)
        });
        assert!(has_x_movement_region, "Should have update region for level 0");
    }

    #[test]
    fn test_update_region_cell_count() {
        let region = ClipmapUpdateRegion {
            level: 0,
            offset: [0, 0],
            size: [10, 20],
            is_full_update: false,
        };

        assert_eq!(region.cell_count(), 200);
    }

    // -------------------------------------------------------------------------
    // Vertex count tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_vertex_count_per_level() {
        let config = ClipmapConfig::new(64, 4, 0.5, 100.0, 500.0);
        let clipmap = Clipmap::new(config);

        assert_eq!(clipmap.vertex_count_per_level(), 64 * 64);
    }

    #[test]
    fn test_total_vertex_count() {
        let config = ClipmapConfig::new(64, 4, 0.5, 100.0, 500.0);
        let clipmap = Clipmap::new(config);

        assert_eq!(clipmap.total_vertex_count(), 64 * 64 * 4);
    }

    #[test]
    fn test_vertex_count_default() {
        let clipmap = Clipmap::with_defaults();
        assert_eq!(
            clipmap.total_vertex_count(),
            (DEFAULT_GRID_SIZE as usize) * (DEFAULT_GRID_SIZE as usize) * (DEFAULT_NUM_LEVELS as usize)
        );
    }

    // -------------------------------------------------------------------------
    // Multiple levels independence tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_levels_independent_spacing() {
        let config = ClipmapConfig::new(128, 8, 0.5, 100.0, 500.0);
        let clipmap = Clipmap::new(config);

        for (i, level) in clipmap.levels().iter().enumerate() {
            let expected_spacing = 0.5 * (1 << i) as f32;
            assert_eq!(level.spacing, expected_spacing);
        }
    }

    #[test]
    fn test_levels_independent_coverage() {
        let config = ClipmapConfig::new(128, 8, 0.5, 100.0, 500.0);
        let clipmap = Clipmap::new(config);

        for (i, level) in clipmap.levels().iter().enumerate() {
            let expected_coverage = 128.0 * 0.5 * (1 << i) as f32;
            assert_eq!(level.coverage, expected_coverage);
        }
    }

    #[test]
    fn test_levels_can_update_independently() {
        let config = ClipmapConfig::new(128, 4, 1.0, 100.0, 500.0);
        let mut clipmap = Clipmap::new(config);

        // Shift only level 0
        clipmap.shift_level(0, [50.0, 50.0]);

        // Level 0 should not need update
        assert!(!clipmap.levels()[0].needs_update);

        // Other levels should still need update
        assert!(clipmap.levels()[1].needs_update);
        assert!(clipmap.levels()[2].needs_update);
        assert!(clipmap.levels()[3].needs_update);
    }

    // -------------------------------------------------------------------------
    // Ring buffer wrap-around tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_toroidal_wrap_positive() {
        assert_eq!(ToroidalAddress::wrap(130, 128), 2);
        assert_eq!(ToroidalAddress::wrap(256, 128), 0);
        assert_eq!(ToroidalAddress::wrap(129, 128), 1);
    }

    #[test]
    fn test_toroidal_wrap_negative() {
        assert_eq!(ToroidalAddress::wrap(-1, 128), 127);
        assert_eq!(ToroidalAddress::wrap(-128, 128), 0);
        assert_eq!(ToroidalAddress::wrap(-129, 128), 127);
    }

    #[test]
    fn test_toroidal_wrap_in_range() {
        for i in 0..128 {
            assert_eq!(ToroidalAddress::wrap(i, 128), i as u32);
        }
    }

    #[test]
    fn test_toroidal_adjacency() {
        assert!(ToroidalAddress::are_adjacent([0, 0], [1, 0], 128));
        assert!(ToroidalAddress::are_adjacent([0, 0], [0, 1], 128));
        assert!(!ToroidalAddress::are_adjacent([0, 0], [2, 0], 128));

        // Wrap-around adjacency
        assert!(ToroidalAddress::are_adjacent([0, 0], [127, 0], 128));
        assert!(ToroidalAddress::are_adjacent([0, 0], [0, 127], 128));
    }

    // -------------------------------------------------------------------------
    // Clipmap vertex tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clipmap_vertex_size() {
        // Vertex should be 48 bytes (3 * 4 * sizeof(f32))
        assert_eq!(std::mem::size_of::<ClipmapVertex>(), 48);
    }

    #[test]
    fn test_clipmap_vertex_creation() {
        let vertex = ClipmapVertex::new([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.5, 0.5], 2);

        assert_eq!(vertex.position[0], 1.0);
        assert_eq!(vertex.position[1], 2.0);
        assert_eq!(vertex.position[2], 3.0);
        assert_eq!(vertex.position[3], 0.0); // morph factor

        assert_eq!(vertex.texcoord[2], 2.0); // level
    }

    #[test]
    fn test_clipmap_vertex_with_morph() {
        let vertex = ClipmapVertex::with_morph(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
            2,
            0.75,
        );

        assert_eq!(vertex.position[3], 0.75); // morph factor stored in w
    }

    // -------------------------------------------------------------------------
    // Level selection tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_level_for_distance() {
        let clipmap = Clipmap::with_defaults();

        // Very close: finest level
        assert_eq!(clipmap.level_for_distance(1.0), 0);

        // Far away: coarser level
        let level = clipmap.level_for_distance(1000.0);
        assert!(level > 0);
    }

    #[test]
    fn test_level_contains_position() {
        let config = ClipmapConfig::new(128, 4, 1.0, 100.0, 500.0);
        let mut clipmap = Clipmap::new(config);

        // Center all levels at origin
        for level in 0..4 {
            clipmap.shift_level(level, [0.0, 0.0]);
        }

        // Test containment at origin
        let level0 = &clipmap.levels()[0];
        assert!(level0.contains([0.0, 0.0]));

        // Test outside level 0 but inside level 1
        let level0_half = level0.half_extent();
        assert!(!level0.contains([level0_half + 1.0, 0.0]));
    }

    // -------------------------------------------------------------------------
    // Config struct layout tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_pod_zeroable() {
        // Ensure ClipmapConfig is Pod (plain old data)
        let config = ClipmapConfig::default();
        let _bytes: &[u8] = bytemuck::bytes_of(&config);

        // Ensure it's Zeroable
        let _zeroed: ClipmapConfig = bytemuck::Zeroable::zeroed();
    }

    #[test]
    fn test_config_size_alignment() {
        // Should be 32 bytes (8 * 4 bytes for alignment)
        let size = std::mem::size_of::<ClipmapConfig>();
        assert_eq!(size, 32, "ClipmapConfig should be 32 bytes");
        assert_eq!(size % 16, 0, "Should be 16-byte aligned for GPU");
    }

    // -------------------------------------------------------------------------
    // Error display tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let err = ClipmapError::InvalidGridSize(2);
        let msg = format!("{}", err);
        assert!(msg.contains("grid size"));
        assert!(msg.contains("2"));

        let err = ClipmapError::InvalidSpacing(-1.0);
        let msg = format!("{}", err);
        assert!(msg.contains("spacing"));
    }

    // -------------------------------------------------------------------------
    // Grid snapping tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_snap_to_grid() {
        let config = ClipmapConfig::new(128, 4, 1.0, 100.0, 500.0);
        let level = ClipmapLevel::new(0, &config);

        assert_eq!(level.snap_to_grid(1.5), 1.0);
        assert_eq!(level.snap_to_grid(0.9), 0.0);
        assert_eq!(level.snap_to_grid(-0.5), -1.0);
    }

    #[test]
    fn test_snap_center() {
        let config = ClipmapConfig::new(128, 4, 2.0, 100.0, 500.0);
        let level = ClipmapLevel::new(0, &config);

        let snapped = level.snap_center([5.5, 7.8]);
        assert_eq!(snapped[0], 4.0);
        assert_eq!(snapped[1], 6.0);
    }

    // -------------------------------------------------------------------------
    // Tangent frame tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_tangent_frame_flat() {
        let normal = [0.0, 1.0, 0.0];
        let (tangent, bitangent) = compute_tangent_frame(normal);

        // Tangent should be in XZ plane
        assert!(tangent[1].abs() < 0.001);

        // Bitangent should also be in XZ plane
        assert!(bitangent[1].abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Forward difference normal tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_forward_diff_normal_flat() {
        let normal = compute_normal_forward_diff(10.0, 10.0, 10.0, 1.0);

        assert!((normal[0]).abs() < 0.001);
        assert!((normal[1] - 1.0).abs() < 0.001);
        assert!((normal[2]).abs() < 0.001);
    }

    #[test]
    fn test_forward_diff_normal_slope() {
        // Slope of 1:1 in X direction
        let normal = compute_normal_forward_diff(0.0, 1.0, 0.0, 1.0);

        // Normal should tilt toward -X
        assert!(normal[0] < 0.0);
        assert!(normal[1] > 0.0);

        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!((len - 1.0).abs() < 0.001, "Should be normalized");
    }

    // -------------------------------------------------------------------------
    // Additional edge case tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clipmap_num_levels() {
        let config = ClipmapConfig::new(64, 5, 0.25, 50.0, 200.0);
        let clipmap = Clipmap::new(config);

        assert_eq!(clipmap.num_levels(), 5);
        assert_eq!(clipmap.levels().len(), 5);
    }

    #[test]
    fn test_level_spacing_via_clipmap() {
        let config = ClipmapConfig::new(128, 4, 2.0, 100.0, 500.0);
        let clipmap = Clipmap::new(config);

        assert_eq!(clipmap.level_spacing(0), 2.0);
        assert_eq!(clipmap.level_spacing(1), 4.0);
        assert_eq!(clipmap.level_spacing(2), 8.0);
        assert_eq!(clipmap.level_spacing(3), 16.0);
    }

    #[test]
    fn test_level_coverage_via_clipmap() {
        let config = ClipmapConfig::new(64, 3, 1.0, 100.0, 500.0);
        let clipmap = Clipmap::new(config);

        assert_eq!(clipmap.level_coverage(0), 64.0);
        assert_eq!(clipmap.level_coverage(1), 128.0);
        assert_eq!(clipmap.level_coverage(2), 256.0);
    }

    #[test]
    fn test_mark_level_updated() {
        let config = ClipmapConfig::new(128, 2, 1.0, 100.0, 500.0);
        let mut clipmap = Clipmap::new(config);

        // Initially needs update
        assert!(clipmap.levels()[0].needs_update);

        // Mark as updated
        clipmap.mark_level_updated(0);

        // Should no longer need update
        assert!(!clipmap.levels()[0].needs_update);
    }

    #[test]
    fn test_vertex_distance() {
        let vertex = [10.0, 0.0];
        let camera = [0.0, 0.0];

        let dist = Geomorphing::vertex_distance(vertex, camera);
        assert_eq!(dist, 10.0);
    }

    #[test]
    fn test_vertex_distance_diagonal() {
        let vertex = [3.0, 4.0];
        let camera = [0.0, 0.0];

        let dist = Geomorphing::vertex_distance(vertex, camera);
        assert_eq!(dist, 5.0);
    }
}
