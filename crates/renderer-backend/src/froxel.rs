//! Froxel (Frustum-Voxel) Volume Management for Volumetric Effects.
//!
//! This module provides infrastructure for managing 3D froxel grids used in
//! volumetric fog, atmospheric scattering, and participating media rendering.
//! Unlike the light culling froxel grid (which maps to screen tiles), this module
//! provides standalone volumetric grid management with configurable quality levels.
//!
//! # Overview
//!
//! Froxels partition the view frustum into a 3D grid where each cell represents
//! a volume of space. The grid dimensions are:
//! - X/Y: Subdivisions across the screen (not tied to tile size)
//! - Z: Depth slices with either linear or logarithmic distribution
//!
//! # Depth Partitioning
//!
//! Two depth partitioning schemes are supported:
//!
//! 1. **Linear**: Uniform depth distribution. Simple but wastes resolution near camera.
//! 2. **Logarithmic**: More slices near camera, fewer far away. Better quality-to-cost ratio.
//!
//! For logarithmic partitioning:
//! ```text
//! depth = near * (far/near)^(slice / (num_slices - 1))
//! slice = (num_slices - 1) * log(depth/near) / log(far/near)
//! ```
//!
//! # Memory Layout
//!
//! Froxel volumes store two 3D textures:
//! - **Radiance** (RGBA16F): Inscattered light accumulated at each froxel
//! - **Extinction** (RGBA16F): Absorption/scattering coefficients
//!
//! # Usage
//!
//! ```ignore
//! let config = FroxelConfig::from_quality(FroxelQuality::Medium);
//! let volume = FroxelVolume::new(config);
//!
//! // Get depth boundaries for shader upload
//! let boundaries = FroxelSlicing::slice_boundaries(&volume.config());
//!
//! // Convert world position to froxel coordinate
//! let froxel = volume.ndc_to_froxel([0.5, 0.5], 0.3);
//! ```

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum grid dimension (width/height/depth).
pub const MIN_GRID_DIM: u32 = 8;

/// Maximum grid dimension (to prevent excessive memory usage).
pub const MAX_GRID_WIDTH: u32 = 256;
pub const MAX_GRID_HEIGHT: u32 = 192;
pub const MAX_GRID_DEPTH: u32 = 128;

/// Bytes per texel for RGBA16F format.
const BYTES_PER_TEXEL_RGBA16F: usize = 8;

// ---------------------------------------------------------------------------
// FroxelQuality — Quality presets
// ---------------------------------------------------------------------------

/// Quality presets for froxel volume resolution.
///
/// Higher quality provides more accurate volumetric effects at the cost
/// of increased GPU memory and compute time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FroxelQuality {
    /// Low quality: 32x24x16 (12,288 cells, ~192KB per 3D texture).
    Low,
    /// Medium quality: 64x48x32 (98,304 cells, ~768KB per 3D texture).
    #[default]
    Medium,
    /// High quality: 128x96x64 (786,432 cells, ~6MB per 3D texture).
    High,
    /// Ultra quality: 256x192x128 (6,291,456 cells, ~48MB per 3D texture).
    Ultra,
}

impl FroxelQuality {
    /// Get the grid dimensions for this quality preset.
    ///
    /// Returns (width, height, depth).
    #[inline]
    pub fn dimensions(self) -> (u32, u32, u32) {
        match self {
            FroxelQuality::Low => (32, 24, 16),
            FroxelQuality::Medium => (64, 48, 32),
            FroxelQuality::High => (128, 96, 64),
            FroxelQuality::Ultra => (256, 192, 128),
        }
    }

    /// Get the cell count for this quality preset.
    #[inline]
    pub fn cell_count(self) -> usize {
        let (w, h, d) = self.dimensions();
        (w as usize) * (h as usize) * (d as usize)
    }

    /// Estimate GPU memory usage for both 3D textures (radiance + extinction).
    ///
    /// Returns memory in bytes.
    #[inline]
    pub fn memory_bytes(self) -> usize {
        // Two RGBA16F 3D textures
        self.cell_count() * BYTES_PER_TEXEL_RGBA16F * 2
    }

    /// Get quality from string name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "low" => Some(FroxelQuality::Low),
            "medium" | "med" => Some(FroxelQuality::Medium),
            "high" => Some(FroxelQuality::High),
            "ultra" => Some(FroxelQuality::Ultra),
            _ => None,
        }
    }

    /// Get the name string for this quality.
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            FroxelQuality::Low => "low",
            FroxelQuality::Medium => "medium",
            FroxelQuality::High => "high",
            FroxelQuality::Ultra => "ultra",
        }
    }
}

// ---------------------------------------------------------------------------
// FroxelConfig — Configuration struct
// ---------------------------------------------------------------------------

/// Configuration for a froxel volume grid.
///
/// This struct is designed to be uploaded to the GPU as a uniform buffer.
/// The layout is `repr(C)` and implements `Pod` for bytemuck compatibility.
///
/// # Memory Layout (24 bytes)
///
/// | Offset | Field        | Size    |
/// |--------|--------------|---------|
/// | 0      | grid_width   | 4 bytes |
/// | 4      | grid_height  | 4 bytes |
/// | 8      | grid_depth   | 4 bytes |
/// | 12     | near_plane   | 4 bytes |
/// | 16     | far_plane    | 4 bytes |
/// | 20     | log_depth    | 4 bytes |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct FroxelConfig {
    /// Grid width (number of froxels along X). Range: 32-256.
    pub grid_width: u32,
    /// Grid height (number of froxels along Y). Range: 24-192.
    pub grid_height: u32,
    /// Grid depth (number of depth slices). Range: 16-128.
    pub grid_depth: u32,
    /// Near plane distance for froxel depth range.
    pub near_plane: f32,
    /// Far plane distance for froxel depth range.
    pub far_plane: f32,
    /// Use logarithmic depth partitioning (1 = true, 0 = false).
    ///
    /// Stored as u32 for GPU compatibility (WGSL doesn't have bool in structs).
    pub log_depth: u32,
}

impl FroxelConfig {
    /// Create a new froxel configuration.
    ///
    /// # Arguments
    ///
    /// * `width` - Grid width (clamped to 8-256).
    /// * `height` - Grid height (clamped to 8-192).
    /// * `depth` - Grid depth (clamped to 8-128).
    /// * `near` - Near plane distance.
    /// * `far` - Far plane distance.
    /// * `log_depth` - Use logarithmic depth partitioning.
    ///
    /// # Panics
    ///
    /// Panics if `near <= 0.0`, `far <= near`, or dimensions are zero.
    pub fn new(
        width: u32,
        height: u32,
        depth: u32,
        near: f32,
        far: f32,
        log_depth: bool,
    ) -> Self {
        assert!(near > 0.0, "near plane must be positive");
        assert!(far > near, "far plane must be greater than near plane");

        Self {
            grid_width: width.clamp(MIN_GRID_DIM, MAX_GRID_WIDTH),
            grid_height: height.clamp(MIN_GRID_DIM, MAX_GRID_HEIGHT),
            grid_depth: depth.clamp(MIN_GRID_DIM, MAX_GRID_DEPTH),
            near_plane: near,
            far_plane: far,
            log_depth: if log_depth { 1 } else { 0 },
        }
    }

    /// Create a configuration from a quality preset.
    ///
    /// # Arguments
    ///
    /// * `quality` - Quality preset.
    /// * `near` - Near plane distance.
    /// * `far` - Far plane distance.
    pub fn from_quality(quality: FroxelQuality, near: f32, far: f32) -> Self {
        let (w, h, d) = quality.dimensions();
        Self::new(w, h, d, near, far, true) // Default to logarithmic
    }

    /// Create a configuration with linear depth partitioning.
    pub fn with_linear_depth(mut self) -> Self {
        self.log_depth = 0;
        self
    }

    /// Create a configuration with logarithmic depth partitioning.
    pub fn with_log_depth(mut self) -> Self {
        self.log_depth = 1;
        self
    }

    /// Check if logarithmic depth partitioning is enabled.
    #[inline]
    pub fn uses_log_depth(&self) -> bool {
        self.log_depth != 0
    }

    /// Total number of froxel cells.
    #[inline]
    pub fn cell_count(&self) -> usize {
        (self.grid_width as usize) * (self.grid_height as usize) * (self.grid_depth as usize)
    }

    /// Validate the configuration.
    ///
    /// Returns `true` if all parameters are within valid ranges.
    pub fn is_valid(&self) -> bool {
        self.grid_width >= MIN_GRID_DIM
            && self.grid_width <= MAX_GRID_WIDTH
            && self.grid_height >= MIN_GRID_DIM
            && self.grid_height <= MAX_GRID_HEIGHT
            && self.grid_depth >= MIN_GRID_DIM
            && self.grid_depth <= MAX_GRID_DEPTH
            && self.near_plane > 0.0
            && self.far_plane > self.near_plane
    }

    /// Get the depth range.
    #[inline]
    pub fn depth_range(&self) -> f32 {
        self.far_plane - self.near_plane
    }

    /// Get the grid dimensions as an array [width, height, depth].
    #[inline]
    pub fn dimensions(&self) -> [u32; 3] {
        [self.grid_width, self.grid_height, self.grid_depth]
    }
}

impl Default for FroxelConfig {
    fn default() -> Self {
        Self::from_quality(FroxelQuality::Medium, 0.1, 100.0)
    }
}

// ---------------------------------------------------------------------------
// FroxelVolume — Volume management
// ---------------------------------------------------------------------------

/// Manages a froxel volume with associated GPU texture handles.
///
/// This struct holds the configuration and placeholder texture handles for
/// the 3D volumetric textures. Actual GPU texture creation is handled
/// elsewhere (typically in the renderer or frame graph).
pub struct FroxelVolume {
    /// Volume configuration.
    config: FroxelConfig,
    /// Handle to the radiance 3D texture (RGBA16F).
    ///
    /// Stores inscattered light accumulated at each froxel.
    radiance_texture: u64,
    /// Handle to the extinction 3D texture (RGBA16F).
    ///
    /// Stores absorption and scattering coefficients.
    extinction_texture: u64,
    /// Frame number when this volume was created.
    created_frame: u64,
}

impl FroxelVolume {
    /// Create a new froxel volume with the given configuration.
    ///
    /// Texture handles are initialized to 0 (invalid). Call `set_textures`
    /// after GPU texture creation.
    pub fn new(config: FroxelConfig) -> Self {
        Self {
            config,
            radiance_texture: 0,
            extinction_texture: 0,
            created_frame: 0,
        }
    }

    /// Create a froxel volume from a quality preset.
    ///
    /// # Arguments
    ///
    /// * `quality` - Quality preset determining grid resolution.
    /// * `near` - Near plane distance.
    /// * `far` - Far plane distance.
    pub fn from_quality(quality: FroxelQuality, near: f32, far: f32) -> Self {
        Self::new(FroxelConfig::from_quality(quality, near, far))
    }

    /// Get the volume configuration.
    #[inline]
    pub fn config(&self) -> &FroxelConfig {
        &self.config
    }

    /// Get the total number of froxel cells.
    #[inline]
    pub fn cell_count(&self) -> usize {
        self.config.cell_count()
    }

    /// Get world-space depth for a given slice index.
    ///
    /// # Arguments
    ///
    /// * `slice` - Slice index (0 to grid_depth - 1).
    ///
    /// # Returns
    ///
    /// World-space depth at the start of the slice.
    #[inline]
    pub fn slice_depth(&self, slice: u32) -> f32 {
        if self.config.uses_log_depth() {
            FroxelSlicing::log_slice_depth(
                slice,
                self.config.grid_depth,
                self.config.near_plane,
                self.config.far_plane,
            )
        } else {
            FroxelSlicing::linear_slice_depth(
                slice,
                self.config.grid_depth,
                self.config.near_plane,
                self.config.far_plane,
            )
        }
    }

    /// Convert froxel grid coordinates to world position.
    ///
    /// # Arguments
    ///
    /// * `x`, `y`, `z` - Froxel grid coordinates.
    /// * `view_proj_inv` - Inverse view-projection matrix (column-major, 16 floats).
    ///
    /// # Returns
    ///
    /// World-space position [x, y, z] at the center of the froxel.
    pub fn world_position(&self, x: u32, y: u32, z: u32, view_proj_inv: &[f32; 16]) -> [f32; 3] {
        // Convert grid coords to NDC
        let ndc_x = (x as f32 + 0.5) / self.config.grid_width as f32 * 2.0 - 1.0;
        let ndc_y = (y as f32 + 0.5) / self.config.grid_height as f32 * 2.0 - 1.0;

        // Get depth at slice center
        let depth = self.slice_depth(z);

        // Convert depth to NDC z (assuming reverse-Z projection)
        // For reverse-Z: ndc_z = near / depth
        // For standard projection, we'd need more info
        let ndc_z = self.config.near_plane / depth;

        // Transform from NDC to world space using inverse view-proj
        Self::transform_point(ndc_x, ndc_y, ndc_z, view_proj_inv)
    }

    /// Transform a point from clip space to world space.
    fn transform_point(x: f32, y: f32, z: f32, inv_matrix: &[f32; 16]) -> [f32; 3] {
        // Column-major matrix multiply: world = inv_matrix * clip
        let w = 1.0; // Assume w=1 for NDC point
        let clip_x = x;
        let clip_y = y;
        let clip_z = z;

        // Extract columns from column-major matrix
        let col0 = [inv_matrix[0], inv_matrix[1], inv_matrix[2], inv_matrix[3]];
        let col1 = [inv_matrix[4], inv_matrix[5], inv_matrix[6], inv_matrix[7]];
        let col2 = [inv_matrix[8], inv_matrix[9], inv_matrix[10], inv_matrix[11]];
        let col3 = [inv_matrix[12], inv_matrix[13], inv_matrix[14], inv_matrix[15]];

        let world_x = col0[0] * clip_x + col1[0] * clip_y + col2[0] * clip_z + col3[0] * w;
        let world_y = col0[1] * clip_x + col1[1] * clip_y + col2[1] * clip_z + col3[1] * w;
        let world_z = col0[2] * clip_x + col1[2] * clip_y + col2[2] * clip_z + col3[2] * w;
        let world_w = col0[3] * clip_x + col1[3] * clip_y + col2[3] * clip_z + col3[3] * w;

        // Perspective divide
        if world_w.abs() > 1e-6 {
            [world_x / world_w, world_y / world_w, world_z / world_w]
        } else {
            [world_x, world_y, world_z]
        }
    }

    /// Convert NDC coordinates and depth to froxel grid coordinates.
    ///
    /// # Arguments
    ///
    /// * `ndc_xy` - NDC x and y coordinates (range -1 to 1).
    /// * `depth` - Linear depth (world-space distance from camera).
    ///
    /// # Returns
    ///
    /// Froxel grid coordinates [x, y, z], or `None` if outside the volume.
    pub fn ndc_to_froxel(&self, ndc_xy: [f32; 2], depth: f32) -> Option<[u32; 3]> {
        // Check depth bounds
        if depth < self.config.near_plane || depth > self.config.far_plane {
            return None;
        }

        // Check NDC bounds
        if ndc_xy[0] < -1.0 || ndc_xy[0] > 1.0 || ndc_xy[1] < -1.0 || ndc_xy[1] > 1.0 {
            return None;
        }

        // Convert NDC to grid coords
        let u = (ndc_xy[0] + 1.0) * 0.5;
        let v = (ndc_xy[1] + 1.0) * 0.5;

        let grid_x = (u * self.config.grid_width as f32).floor() as u32;
        let grid_y = (v * self.config.grid_height as f32).floor() as u32;

        // Clamp to valid range
        let grid_x = grid_x.min(self.config.grid_width - 1);
        let grid_y = grid_y.min(self.config.grid_height - 1);

        // Get depth slice
        let slice = FroxelSlicing::depth_to_slice(
            depth,
            self.config.grid_depth,
            self.config.near_plane,
            self.config.far_plane,
            self.config.uses_log_depth(),
        );

        Some([grid_x, grid_y, slice])
    }

    /// Estimate GPU memory usage for both 3D textures.
    ///
    /// Returns memory in bytes for two RGBA16F 3D textures.
    #[inline]
    pub fn memory_size_bytes(&self) -> usize {
        self.cell_count() * BYTES_PER_TEXEL_RGBA16F * 2
    }

    /// Get the radiance texture handle.
    #[inline]
    pub fn radiance_texture(&self) -> u64 {
        self.radiance_texture
    }

    /// Get the extinction texture handle.
    #[inline]
    pub fn extinction_texture(&self) -> u64 {
        self.extinction_texture
    }

    /// Set the texture handles after GPU texture creation.
    pub fn set_textures(&mut self, radiance: u64, extinction: u64, frame: u64) {
        self.radiance_texture = radiance;
        self.extinction_texture = extinction;
        self.created_frame = frame;
    }

    /// Get the frame number when this volume was created.
    #[inline]
    pub fn created_frame(&self) -> u64 {
        self.created_frame
    }

    /// Check if the volume has valid texture handles.
    #[inline]
    pub fn has_textures(&self) -> bool {
        self.radiance_texture != 0 && self.extinction_texture != 0
    }
}

// ---------------------------------------------------------------------------
// FroxelSlicing — Depth partitioning utilities
// ---------------------------------------------------------------------------

/// Utilities for froxel depth slicing calculations.
///
/// Provides both linear and logarithmic depth partitioning schemes.
pub struct FroxelSlicing;

impl FroxelSlicing {
    /// Calculate depth at a given slice using linear partitioning.
    ///
    /// Linear partitioning divides the depth range uniformly:
    /// ```text
    /// depth = near + (far - near) * (slice / (num_slices - 1))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `slice` - Slice index (0 to num_slices - 1).
    /// * `num_slices` - Total number of depth slices.
    /// * `near` - Near plane distance.
    /// * `far` - Far plane distance.
    #[inline]
    pub fn linear_slice_depth(slice: u32, num_slices: u32, near: f32, far: f32) -> f32 {
        if num_slices <= 1 {
            return near;
        }
        let t = slice as f32 / (num_slices - 1) as f32;
        near + (far - near) * t
    }

    /// Calculate depth at a given slice using logarithmic partitioning.
    ///
    /// Logarithmic partitioning provides more resolution near the camera:
    /// ```text
    /// depth = near * (far / near)^(slice / (num_slices - 1))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `slice` - Slice index (0 to num_slices - 1).
    /// * `num_slices` - Total number of depth slices.
    /// * `near` - Near plane distance.
    /// * `far` - Far plane distance.
    #[inline]
    pub fn log_slice_depth(slice: u32, num_slices: u32, near: f32, far: f32) -> f32 {
        if num_slices <= 1 {
            return near;
        }
        let t = slice as f32 / (num_slices - 1) as f32;
        near * (far / near).powf(t)
    }

    /// Find the slice index for a given depth.
    ///
    /// # Arguments
    ///
    /// * `depth` - Linear depth value.
    /// * `num_slices` - Total number of depth slices.
    /// * `near` - Near plane distance.
    /// * `far` - Far plane distance.
    /// * `log_partitioning` - Use logarithmic partitioning.
    ///
    /// # Returns
    ///
    /// Slice index (clamped to valid range).
    #[inline]
    pub fn depth_to_slice(
        depth: f32,
        num_slices: u32,
        near: f32,
        far: f32,
        log_partitioning: bool,
    ) -> u32 {
        if num_slices <= 1 {
            return 0;
        }

        let t = if log_partitioning {
            // t = log(depth/near) / log(far/near)
            if depth <= near {
                0.0
            } else {
                (depth / near).ln() / (far / near).ln()
            }
        } else {
            // t = (depth - near) / (far - near)
            (depth - near) / (far - near)
        };

        let slice = (t * (num_slices - 1) as f32).round() as i32;
        slice.clamp(0, (num_slices - 1) as i32) as u32
    }

    /// Calculate the center depth of a slice.
    ///
    /// Returns the depth at the center of the slice (midpoint between
    /// this slice's start and the next slice's start).
    #[inline]
    pub fn slice_center_depth(
        slice: u32,
        num_slices: u32,
        near: f32,
        far: f32,
        log_partitioning: bool,
    ) -> f32 {
        if log_partitioning {
            let d0 = Self::log_slice_depth(slice, num_slices, near, far);
            let d1 = if slice + 1 < num_slices {
                Self::log_slice_depth(slice + 1, num_slices, near, far)
            } else {
                far
            };
            (d0 + d1) * 0.5
        } else {
            let d0 = Self::linear_slice_depth(slice, num_slices, near, far);
            let d1 = if slice + 1 < num_slices {
                Self::linear_slice_depth(slice + 1, num_slices, near, far)
            } else {
                far
            };
            (d0 + d1) * 0.5
        }
    }

    /// Get all slice boundary depths.
    ///
    /// Returns a vector with `num_slices` depth values, one for each slice boundary.
    pub fn slice_boundaries(config: &FroxelConfig) -> Vec<f32> {
        let mut boundaries = Vec::with_capacity(config.grid_depth as usize);
        for slice in 0..config.grid_depth {
            let depth = if config.uses_log_depth() {
                Self::log_slice_depth(slice, config.grid_depth, config.near_plane, config.far_plane)
            } else {
                Self::linear_slice_depth(
                    slice,
                    config.grid_depth,
                    config.near_plane,
                    config.far_plane,
                )
            };
            boundaries.push(depth);
        }
        boundaries
    }
}

// ---------------------------------------------------------------------------
// FroxelDebug — Visualization helpers
// ---------------------------------------------------------------------------

/// Debug visualization utilities for froxel volumes.
pub struct FroxelDebug;

impl FroxelDebug {
    /// Get all slice depth boundaries for visualization.
    ///
    /// # Arguments
    ///
    /// * `config` - Froxel configuration.
    ///
    /// # Returns
    ///
    /// Vector of depth values for each slice boundary.
    #[inline]
    pub fn slice_boundaries(config: &FroxelConfig) -> Vec<f32> {
        FroxelSlicing::slice_boundaries(config)
    }

    /// Calculate world-space AABB bounds for a single froxel cell.
    ///
    /// # Arguments
    ///
    /// * `config` - Froxel configuration.
    /// * `x`, `y`, `z` - Froxel grid coordinates.
    /// * `view_proj_inv` - Inverse view-projection matrix (column-major).
    ///
    /// # Returns
    ///
    /// Tuple of (min_corner, max_corner) world-space positions.
    pub fn cell_world_bounds(
        config: &FroxelConfig,
        x: u32,
        y: u32,
        z: u32,
        view_proj_inv: &[f32; 16],
    ) -> ([f32; 3], [f32; 3]) {
        // Get NDC bounds for this cell
        let inv_w = 1.0 / config.grid_width as f32;
        let inv_h = 1.0 / config.grid_height as f32;

        let ndc_x_min = (x as f32 * inv_w) * 2.0 - 1.0;
        let ndc_x_max = ((x + 1) as f32 * inv_w) * 2.0 - 1.0;
        let ndc_y_min = (y as f32 * inv_h) * 2.0 - 1.0;
        let ndc_y_max = ((y + 1) as f32 * inv_h) * 2.0 - 1.0;

        // Get depth bounds
        let depth_near = if config.uses_log_depth() {
            FroxelSlicing::log_slice_depth(z, config.grid_depth, config.near_plane, config.far_plane)
        } else {
            FroxelSlicing::linear_slice_depth(
                z,
                config.grid_depth,
                config.near_plane,
                config.far_plane,
            )
        };
        let depth_far = if z + 1 < config.grid_depth {
            if config.uses_log_depth() {
                FroxelSlicing::log_slice_depth(
                    z + 1,
                    config.grid_depth,
                    config.near_plane,
                    config.far_plane,
                )
            } else {
                FroxelSlicing::linear_slice_depth(
                    z + 1,
                    config.grid_depth,
                    config.near_plane,
                    config.far_plane,
                )
            }
        } else {
            config.far_plane
        };

        // Convert depths to NDC z (reverse-Z assumption)
        let ndc_z_near = config.near_plane / depth_near;
        let ndc_z_far = config.near_plane / depth_far;

        // Transform all 8 corners and find AABB
        let corners = [
            (ndc_x_min, ndc_y_min, ndc_z_near),
            (ndc_x_max, ndc_y_min, ndc_z_near),
            (ndc_x_min, ndc_y_max, ndc_z_near),
            (ndc_x_max, ndc_y_max, ndc_z_near),
            (ndc_x_min, ndc_y_min, ndc_z_far),
            (ndc_x_max, ndc_y_min, ndc_z_far),
            (ndc_x_min, ndc_y_max, ndc_z_far),
            (ndc_x_max, ndc_y_max, ndc_z_far),
        ];

        let mut min = [f32::MAX, f32::MAX, f32::MAX];
        let mut max = [f32::MIN, f32::MIN, f32::MIN];

        for (cx, cy, cz) in corners {
            let world = FroxelVolume::transform_point(cx, cy, cz, view_proj_inv);
            for i in 0..3 {
                min[i] = min[i].min(world[i]);
                max[i] = max[i].max(world[i]);
            }
        }

        (min, max)
    }

    /// Generate wire frame lines for all slice boundaries (for debug rendering).
    ///
    /// Returns pairs of points for line rendering.
    pub fn slice_wireframe_lines(config: &FroxelConfig) -> Vec<([f32; 2], [f32; 2], f32)> {
        let boundaries = FroxelSlicing::slice_boundaries(config);
        let mut lines = Vec::new();

        for depth in boundaries {
            // Create a grid of lines at this depth
            // Horizontal lines
            for row in 0..=4 {
                let y = -1.0 + 0.5 * row as f32;
                lines.push(([-1.0, y], [1.0, y], depth));
            }
            // Vertical lines
            for col in 0..=4 {
                let x = -1.0 + 0.5 * col as f32;
                lines.push(([x, -1.0], [x, 1.0], depth));
            }
        }

        lines
    }

    /// Calculate statistics about froxel distribution.
    pub fn distribution_stats(config: &FroxelConfig) -> FroxelDistributionStats {
        let boundaries = FroxelSlicing::slice_boundaries(config);
        let mut slice_thicknesses = Vec::with_capacity(boundaries.len() - 1);

        for i in 0..boundaries.len() - 1 {
            slice_thicknesses.push(boundaries[i + 1] - boundaries[i]);
        }

        let min_thickness = slice_thicknesses.iter().cloned().fold(f32::MAX, f32::min);
        let max_thickness = slice_thicknesses.iter().cloned().fold(f32::MIN, f32::max);
        let avg_thickness = slice_thicknesses.iter().sum::<f32>() / slice_thicknesses.len() as f32;

        FroxelDistributionStats {
            num_slices: config.grid_depth,
            min_thickness,
            max_thickness,
            avg_thickness,
            near_slice_thickness: slice_thicknesses.first().cloned().unwrap_or(0.0),
            far_slice_thickness: slice_thicknesses.last().cloned().unwrap_or(0.0),
            depth_range: config.far_plane - config.near_plane,
        }
    }
}

/// Statistics about froxel depth distribution.
#[derive(Debug, Clone, Copy)]
pub struct FroxelDistributionStats {
    /// Number of depth slices.
    pub num_slices: u32,
    /// Minimum slice thickness.
    pub min_thickness: f32,
    /// Maximum slice thickness.
    pub max_thickness: f32,
    /// Average slice thickness.
    pub avg_thickness: f32,
    /// Thickness of the nearest slice.
    pub near_slice_thickness: f32,
    /// Thickness of the farthest slice.
    pub far_slice_thickness: f32,
    /// Total depth range (far - near).
    pub depth_range: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // FroxelQuality tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quality_dimensions() {
        assert_eq!(FroxelQuality::Low.dimensions(), (32, 24, 16));
        assert_eq!(FroxelQuality::Medium.dimensions(), (64, 48, 32));
        assert_eq!(FroxelQuality::High.dimensions(), (128, 96, 64));
        assert_eq!(FroxelQuality::Ultra.dimensions(), (256, 192, 128));
    }

    #[test]
    fn test_quality_cell_count() {
        assert_eq!(FroxelQuality::Low.cell_count(), 32 * 24 * 16);
        assert_eq!(FroxelQuality::Medium.cell_count(), 64 * 48 * 32);
        assert_eq!(FroxelQuality::High.cell_count(), 128 * 96 * 64);
        assert_eq!(FroxelQuality::Ultra.cell_count(), 256 * 192 * 128);
    }

    #[test]
    fn test_quality_memory_bytes() {
        // Each quality has 2 RGBA16F textures (8 bytes per texel)
        let low_memory = 32 * 24 * 16 * 8 * 2;
        assert_eq!(FroxelQuality::Low.memory_bytes(), low_memory);

        let ultra_memory = 256 * 192 * 128 * 8 * 2;
        assert_eq!(FroxelQuality::Ultra.memory_bytes(), ultra_memory);
    }

    #[test]
    fn test_quality_from_name() {
        assert_eq!(FroxelQuality::from_name("low"), Some(FroxelQuality::Low));
        assert_eq!(FroxelQuality::from_name("LOW"), Some(FroxelQuality::Low));
        assert_eq!(FroxelQuality::from_name("medium"), Some(FroxelQuality::Medium));
        assert_eq!(FroxelQuality::from_name("med"), Some(FroxelQuality::Medium));
        assert_eq!(FroxelQuality::from_name("high"), Some(FroxelQuality::High));
        assert_eq!(FroxelQuality::from_name("ultra"), Some(FroxelQuality::Ultra));
        assert_eq!(FroxelQuality::from_name("invalid"), None);
    }

    #[test]
    fn test_quality_name() {
        assert_eq!(FroxelQuality::Low.name(), "low");
        assert_eq!(FroxelQuality::Medium.name(), "medium");
        assert_eq!(FroxelQuality::High.name(), "high");
        assert_eq!(FroxelQuality::Ultra.name(), "ultra");
    }

    #[test]
    fn test_quality_default() {
        assert_eq!(FroxelQuality::default(), FroxelQuality::Medium);
    }

    // -----------------------------------------------------------------------
    // FroxelConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_from_quality() {
        let config = FroxelConfig::from_quality(FroxelQuality::Medium, 0.1, 100.0);
        assert_eq!(config.grid_width, 64);
        assert_eq!(config.grid_height, 48);
        assert_eq!(config.grid_depth, 32);
        assert_eq!(config.near_plane, 0.1);
        assert_eq!(config.far_plane, 100.0);
        assert!(config.uses_log_depth()); // Default to log
    }

    #[test]
    fn test_config_new_clamping() {
        // Test clamping to min
        let config = FroxelConfig::new(1, 1, 1, 0.1, 100.0, false);
        assert_eq!(config.grid_width, MIN_GRID_DIM);
        assert_eq!(config.grid_height, MIN_GRID_DIM);
        assert_eq!(config.grid_depth, MIN_GRID_DIM);

        // Test clamping to max
        let config = FroxelConfig::new(1000, 1000, 1000, 0.1, 100.0, false);
        assert_eq!(config.grid_width, MAX_GRID_WIDTH);
        assert_eq!(config.grid_height, MAX_GRID_HEIGHT);
        assert_eq!(config.grid_depth, MAX_GRID_DEPTH);
    }

    #[test]
    fn test_config_cell_count() {
        let config = FroxelConfig::new(10, 10, 10, 0.1, 100.0, false);
        assert_eq!(config.cell_count(), 10 * 10 * 10);
    }

    #[test]
    fn test_config_validation() {
        let valid = FroxelConfig::from_quality(FroxelQuality::Medium, 0.1, 100.0);
        assert!(valid.is_valid());

        // Create an invalid config by manipulating fields
        let mut invalid = valid;
        invalid.near_plane = -1.0;
        assert!(!invalid.is_valid());

        let mut invalid = valid;
        invalid.far_plane = 0.05; // far < near
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_config_depth_range() {
        let config = FroxelConfig::from_quality(FroxelQuality::Medium, 0.1, 100.0);
        assert!((config.depth_range() - 99.9).abs() < 0.001);
    }

    #[test]
    fn test_config_dimensions() {
        let config = FroxelConfig::from_quality(FroxelQuality::High, 0.5, 500.0);
        assert_eq!(config.dimensions(), [128, 96, 64]);
    }

    #[test]
    fn test_config_with_linear_depth() {
        let config = FroxelConfig::from_quality(FroxelQuality::Medium, 0.1, 100.0)
            .with_linear_depth();
        assert!(!config.uses_log_depth());
    }

    #[test]
    fn test_config_with_log_depth() {
        let config = FroxelConfig::new(64, 48, 32, 0.1, 100.0, false)
            .with_log_depth();
        assert!(config.uses_log_depth());
    }

    #[test]
    fn test_config_default() {
        let config = FroxelConfig::default();
        assert_eq!(config.grid_width, 64);
        assert_eq!(config.grid_height, 48);
        assert_eq!(config.grid_depth, 32);
        assert_eq!(config.near_plane, 0.1);
        assert_eq!(config.far_plane, 100.0);
    }

    #[test]
    #[should_panic(expected = "near plane must be positive")]
    fn test_config_invalid_near() {
        FroxelConfig::new(64, 48, 32, 0.0, 100.0, false);
    }

    #[test]
    #[should_panic(expected = "far plane must be greater than near plane")]
    fn test_config_invalid_far() {
        FroxelConfig::new(64, 48, 32, 100.0, 50.0, false);
    }

    // -----------------------------------------------------------------------
    // FroxelSlicing tests — Linear
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_slice_depth_boundaries() {
        let near = 1.0;
        let far = 101.0;
        let num_slices = 11;

        // Slice 0 should be at near
        let d0 = FroxelSlicing::linear_slice_depth(0, num_slices, near, far);
        assert!((d0 - near).abs() < 0.001);

        // Slice num_slices-1 should be at far
        let d_last = FroxelSlicing::linear_slice_depth(num_slices - 1, num_slices, near, far);
        assert!((d_last - far).abs() < 0.001);

        // Middle slice should be at midpoint
        let d_mid = FroxelSlicing::linear_slice_depth(5, num_slices, near, far);
        let expected_mid = near + (far - near) * 0.5;
        assert!((d_mid - expected_mid).abs() < 0.001);
    }

    #[test]
    fn test_linear_uniform_spacing() {
        let near = 0.1;
        let far = 100.1;
        let num_slices = 10;

        // Check uniform spacing
        let mut depths = Vec::new();
        for i in 0..num_slices {
            depths.push(FroxelSlicing::linear_slice_depth(i, num_slices, near, far));
        }

        let expected_spacing = (far - near) / (num_slices - 1) as f32;
        for i in 1..depths.len() {
            let spacing = depths[i] - depths[i - 1];
            assert!((spacing - expected_spacing).abs() < 0.001);
        }
    }

    #[test]
    fn test_linear_single_slice() {
        let depth = FroxelSlicing::linear_slice_depth(0, 1, 0.1, 100.0);
        assert!((depth - 0.1).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // FroxelSlicing tests — Logarithmic
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_slice_depth_boundaries() {
        let near = 0.1;
        let far = 100.0;
        let num_slices = 32;

        // Slice 0 should be at near
        let d0 = FroxelSlicing::log_slice_depth(0, num_slices, near, far);
        assert!((d0 - near).abs() < 0.0001);

        // Slice num_slices-1 should be at far
        let d_last = FroxelSlicing::log_slice_depth(num_slices - 1, num_slices, near, far);
        assert!((d_last - far).abs() < 0.001);
    }

    #[test]
    fn test_log_more_near_resolution() {
        let near = 0.1;
        let far = 100.0;
        let num_slices = 16;

        // Logarithmic should have more slices near camera
        let depths: Vec<f32> = (0..num_slices)
            .map(|i| FroxelSlicing::log_slice_depth(i, num_slices, near, far))
            .collect();

        // First half of slices should cover less than half the depth range
        let midpoint_depth = depths[num_slices as usize / 2];
        let half_range = (far - near) / 2.0 + near;
        assert!(midpoint_depth < half_range);
    }

    #[test]
    fn test_log_exponential_growth() {
        let near = 1.0;
        let far = 1000.0;
        let num_slices = 10;

        // Check that each slice ratio is constant (exponential growth)
        let depths: Vec<f32> = (0..num_slices)
            .map(|i| FroxelSlicing::log_slice_depth(i, num_slices, near, far))
            .collect();

        let expected_ratio = (far / near).powf(1.0 / (num_slices - 1) as f32);
        for i in 1..depths.len() {
            let ratio = depths[i] / depths[i - 1];
            assert!((ratio - expected_ratio).abs() < 0.001);
        }
    }

    #[test]
    fn test_log_single_slice() {
        let depth = FroxelSlicing::log_slice_depth(0, 1, 0.1, 100.0);
        assert!((depth - 0.1).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // FroxelSlicing tests — depth_to_slice
    // -----------------------------------------------------------------------

    #[test]
    fn test_depth_to_slice_linear_boundaries() {
        let near = 1.0;
        let far = 101.0;
        let num_slices = 11;

        // Near should map to slice 0
        let slice = FroxelSlicing::depth_to_slice(near, num_slices, near, far, false);
        assert_eq!(slice, 0);

        // Far should map to last slice
        let slice = FroxelSlicing::depth_to_slice(far, num_slices, near, far, false);
        assert_eq!(slice, num_slices - 1);
    }

    #[test]
    fn test_depth_to_slice_log_boundaries() {
        let near = 0.1;
        let far = 100.0;
        let num_slices = 32;

        // Near should map to slice 0
        let slice = FroxelSlicing::depth_to_slice(near, num_slices, near, far, true);
        assert_eq!(slice, 0);

        // Far should map to last slice
        let slice = FroxelSlicing::depth_to_slice(far, num_slices, near, far, true);
        assert_eq!(slice, num_slices - 1);
    }

    #[test]
    fn test_depth_to_slice_clamping() {
        let near = 1.0;
        let far = 100.0;
        let num_slices = 16;

        // Below near should clamp to 0
        let slice = FroxelSlicing::depth_to_slice(0.1, num_slices, near, far, true);
        assert_eq!(slice, 0);

        // Above far should clamp to last
        let slice = FroxelSlicing::depth_to_slice(1000.0, num_slices, near, far, true);
        assert_eq!(slice, num_slices - 1);
    }

    #[test]
    fn test_depth_to_slice_roundtrip_linear() {
        let near = 0.1;
        let far = 100.0;
        let num_slices = 16;

        for slice in 0..num_slices {
            let depth = FroxelSlicing::linear_slice_depth(slice, num_slices, near, far);
            let recovered = FroxelSlicing::depth_to_slice(depth, num_slices, near, far, false);
            assert_eq!(recovered, slice, "Roundtrip failed for slice {}", slice);
        }
    }

    #[test]
    fn test_depth_to_slice_roundtrip_log() {
        let near = 0.1;
        let far = 100.0;
        let num_slices = 16;

        for slice in 0..num_slices {
            let depth = FroxelSlicing::log_slice_depth(slice, num_slices, near, far);
            let recovered = FroxelSlicing::depth_to_slice(depth, num_slices, near, far, true);
            assert_eq!(recovered, slice, "Roundtrip failed for slice {}", slice);
        }
    }

    #[test]
    fn test_depth_to_slice_single_slice() {
        let slice = FroxelSlicing::depth_to_slice(50.0, 1, 0.1, 100.0, true);
        assert_eq!(slice, 0);
    }

    // -----------------------------------------------------------------------
    // FroxelVolume tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_volume_new() {
        let config = FroxelConfig::from_quality(FroxelQuality::Medium, 0.1, 100.0);
        let volume = FroxelVolume::new(config);

        assert_eq!(volume.cell_count(), 64 * 48 * 32);
        assert_eq!(volume.radiance_texture(), 0);
        assert_eq!(volume.extinction_texture(), 0);
        assert!(!volume.has_textures());
    }

    #[test]
    fn test_volume_from_quality() {
        let volume = FroxelVolume::from_quality(FroxelQuality::High, 0.5, 500.0);
        assert_eq!(volume.config().grid_width, 128);
        assert_eq!(volume.config().grid_height, 96);
        assert_eq!(volume.config().grid_depth, 64);
    }

    #[test]
    fn test_volume_memory_size() {
        let volume = FroxelVolume::from_quality(FroxelQuality::Low, 0.1, 100.0);
        let expected = 32 * 24 * 16 * 8 * 2; // cells * bytes_per_texel * num_textures
        assert_eq!(volume.memory_size_bytes(), expected);
    }

    #[test]
    fn test_volume_slice_depth_linear() {
        let config = FroxelConfig::new(32, 24, 10, 1.0, 101.0, false);
        let volume = FroxelVolume::new(config);

        // First slice at near
        assert!((volume.slice_depth(0) - 1.0).abs() < 0.001);
        // Last slice at far
        assert!((volume.slice_depth(9) - 101.0).abs() < 0.001);
    }

    #[test]
    fn test_volume_slice_depth_log() {
        let config = FroxelConfig::new(32, 24, 10, 0.1, 100.0, true);
        let volume = FroxelVolume::new(config);

        // First slice at near
        assert!((volume.slice_depth(0) - 0.1).abs() < 0.001);
        // Last slice at far
        assert!((volume.slice_depth(9) - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_volume_set_textures() {
        let config = FroxelConfig::from_quality(FroxelQuality::Medium, 0.1, 100.0);
        let mut volume = FroxelVolume::new(config);

        assert!(!volume.has_textures());

        volume.set_textures(123, 456, 42);

        assert!(volume.has_textures());
        assert_eq!(volume.radiance_texture(), 123);
        assert_eq!(volume.extinction_texture(), 456);
        assert_eq!(volume.created_frame(), 42);
    }

    #[test]
    fn test_volume_ndc_to_froxel_valid() {
        let config = FroxelConfig::new(64, 48, 32, 0.1, 100.0, false);
        let volume = FroxelVolume::new(config);

        // Center of screen, mid depth
        let froxel = volume.ndc_to_froxel([0.0, 0.0], 50.0);
        assert!(froxel.is_some());
        let [x, y, _z] = froxel.unwrap();
        assert_eq!(x, 32); // Center of 64
        assert_eq!(y, 24); // Center of 48
    }

    #[test]
    fn test_volume_ndc_to_froxel_out_of_bounds() {
        let config = FroxelConfig::new(64, 48, 32, 0.1, 100.0, false);
        let volume = FroxelVolume::new(config);

        // Too close
        assert!(volume.ndc_to_froxel([0.0, 0.0], 0.05).is_none());

        // Too far
        assert!(volume.ndc_to_froxel([0.0, 0.0], 150.0).is_none());

        // Outside NDC
        assert!(volume.ndc_to_froxel([1.5, 0.0], 50.0).is_none());
        assert!(volume.ndc_to_froxel([0.0, -1.5], 50.0).is_none());
    }

    #[test]
    fn test_volume_ndc_to_froxel_corners() {
        let config = FroxelConfig::new(64, 48, 32, 0.1, 100.0, false);
        let volume = FroxelVolume::new(config);

        // Top-left corner
        let froxel = volume.ndc_to_froxel([-1.0, -1.0], 0.1);
        assert!(froxel.is_some());
        let [x, y, z] = froxel.unwrap();
        assert_eq!(x, 0);
        assert_eq!(y, 0);
        assert_eq!(z, 0);

        // Near bottom-right corner (just inside bounds)
        let froxel = volume.ndc_to_froxel([0.99, 0.99], 99.9);
        assert!(froxel.is_some());
        let [x, y, _z] = froxel.unwrap();
        assert_eq!(x, 63);
        assert_eq!(y, 47);
    }

    #[test]
    fn test_volume_world_position_identity() {
        let config = FroxelConfig::new(32, 24, 16, 0.1, 100.0, false);
        let volume = FroxelVolume::new(config);

        // Identity matrix (no transformation)
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let pos = volume.world_position(16, 12, 8, &identity);
        // With identity matrix, world = clip space
        // NDC coords should be near (0, 0, ~0.5)
        assert!(pos[0].abs() < 0.1); // x near center
        assert!(pos[1].abs() < 0.1); // y near center
    }

    // -----------------------------------------------------------------------
    // FroxelSlicing boundaries tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_slice_boundaries_count() {
        let config = FroxelConfig::new(32, 24, 16, 0.1, 100.0, true);
        let boundaries = FroxelSlicing::slice_boundaries(&config);
        assert_eq!(boundaries.len(), 16);
    }

    #[test]
    fn test_slice_boundaries_ordered() {
        let config = FroxelConfig::new(32, 24, 32, 0.1, 100.0, true);
        let boundaries = FroxelSlicing::slice_boundaries(&config);

        for i in 1..boundaries.len() {
            assert!(boundaries[i] > boundaries[i - 1],
                "Boundaries should be monotonically increasing");
        }
    }

    #[test]
    fn test_slice_center_depth() {
        let near = 0.1;
        let far = 100.0;
        let num_slices = 16;

        for slice in 0..num_slices - 1 {
            let center = FroxelSlicing::slice_center_depth(slice, num_slices, near, far, true);
            let d0 = FroxelSlicing::log_slice_depth(slice, num_slices, near, far);
            let d1 = FroxelSlicing::log_slice_depth(slice + 1, num_slices, near, far);

            assert!(center > d0);
            assert!(center < d1);
        }
    }

    // -----------------------------------------------------------------------
    // FroxelDebug tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_slice_boundaries() {
        let config = FroxelConfig::new(32, 24, 16, 0.1, 100.0, true);
        let boundaries = FroxelDebug::slice_boundaries(&config);
        assert_eq!(boundaries.len(), 16);
    }

    #[test]
    fn test_debug_cell_world_bounds() {
        let config = FroxelConfig::new(32, 24, 16, 0.1, 100.0, false);
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        let (min, max) = FroxelDebug::cell_world_bounds(&config, 0, 0, 0, &identity);

        // Min should be less than max for all axes
        assert!(min[0] < max[0]);
        assert!(min[1] < max[1]);
        assert!(min[2] < max[2]);
    }

    #[test]
    fn test_debug_distribution_stats() {
        let config = FroxelConfig::new(32, 24, 16, 0.1, 100.0, true);
        let stats = FroxelDebug::distribution_stats(&config);

        assert_eq!(stats.num_slices, 16);
        assert!(stats.min_thickness > 0.0);
        assert!(stats.max_thickness > stats.min_thickness);
        assert!(stats.avg_thickness > 0.0);
        assert!((stats.depth_range - 99.9).abs() < 0.001);

        // For log distribution, near slice should be thinner than far
        assert!(stats.near_slice_thickness < stats.far_slice_thickness);
    }

    #[test]
    fn test_debug_distribution_stats_linear() {
        let config = FroxelConfig::new(32, 24, 16, 0.1, 100.0, false);
        let stats = FroxelDebug::distribution_stats(&config);

        // For linear distribution, all slices should have same thickness
        assert!((stats.min_thickness - stats.max_thickness).abs() < 0.001);
    }

    #[test]
    fn test_debug_wireframe_lines() {
        let config = FroxelConfig::new(32, 24, 8, 0.1, 100.0, true);
        let lines = FroxelDebug::slice_wireframe_lines(&config);

        // Should have lines for each slice
        assert!(!lines.is_empty());

        // Each line should have valid depth
        for (_, _, depth) in &lines {
            assert!(*depth >= 0.1);
            assert!(*depth <= 100.0);
        }
    }

    // -----------------------------------------------------------------------
    // Pod/Zeroable tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_pod_zeroable() {
        // Verify FroxelConfig is Pod and Zeroable
        let zeroed: FroxelConfig = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.grid_width, 0);
        assert_eq!(zeroed.grid_height, 0);
        assert_eq!(zeroed.grid_depth, 0);
        assert_eq!(zeroed.near_plane, 0.0);
        assert_eq!(zeroed.far_plane, 0.0);
        assert_eq!(zeroed.log_depth, 0);

        // Test bytemuck cast
        let config = FroxelConfig::new(64, 48, 32, 0.1, 100.0, true);
        let bytes: &[u8] = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), std::mem::size_of::<FroxelConfig>());
    }

    #[test]
    fn test_config_size() {
        // FroxelConfig should be 24 bytes (6 * 4)
        assert_eq!(std::mem::size_of::<FroxelConfig>(), 24);
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extreme_depth_range() {
        // Very small near plane
        let config = FroxelConfig::new(32, 24, 16, 0.001, 10000.0, true);
        assert!(config.is_valid());

        let boundaries = FroxelSlicing::slice_boundaries(&config);
        assert!(boundaries.first().is_some_and(|d| (*d - 0.001).abs() < 0.0001));
        assert!(boundaries.last().is_some_and(|d| (*d - 10000.0).abs() < 1.0));
    }

    #[test]
    fn test_narrow_depth_range() {
        // Very narrow depth range
        let config = FroxelConfig::new(32, 24, 16, 10.0, 10.1, false);
        assert!(config.is_valid());

        let boundaries = FroxelSlicing::slice_boundaries(&config);
        let total_range = boundaries.last().unwrap() - boundaries.first().unwrap();
        assert!((total_range - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_min_grid_dimensions() {
        let config = FroxelConfig::new(MIN_GRID_DIM, MIN_GRID_DIM, MIN_GRID_DIM, 0.1, 100.0, true);
        let volume = FroxelVolume::new(config);

        assert_eq!(volume.cell_count(), (MIN_GRID_DIM * MIN_GRID_DIM * MIN_GRID_DIM) as usize);
    }

    #[test]
    fn test_max_grid_dimensions() {
        let config = FroxelConfig::new(MAX_GRID_WIDTH, MAX_GRID_HEIGHT, MAX_GRID_DEPTH, 0.1, 100.0, true);
        let volume = FroxelVolume::new(config);

        let expected = (MAX_GRID_WIDTH as usize) * (MAX_GRID_HEIGHT as usize) * (MAX_GRID_DEPTH as usize);
        assert_eq!(volume.cell_count(), expected);
    }
}
