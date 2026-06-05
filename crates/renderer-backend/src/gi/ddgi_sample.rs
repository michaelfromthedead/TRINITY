//! DDGI probe sampling pass for indirect irradiance computation.
//!
//! This module provides the GPU dispatch logic for sampling the DDGI probe grid
//! at each shading point. It reads from the G-buffer and writes indirect
//! irradiance to an output texture.
//!
//! # Features
//!
//! - L2 (9-coefficient) spherical harmonics evaluation
//! - Trilinear interpolation between 8 surrounding probes
//! - Visibility-weighted sampling to reduce light leaking
//! - Parallax correction for off-center sampling
//! - Infinite scrolling grid support
//!
//! # Usage
//!
//! ```ignore
//! let pass = DDGISamplePass::new(&device, quality);
//! pass.dispatch(&mut encoder, &resources);
//! ```

use crate::gi::probe_grid::{ProbeGridGpu, ProbeSH};
use bytemuck::{Pod, Zeroable};

// ============================================================================
// Configuration
// ============================================================================

/// DDGI sampling quality preset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DDGISampleQuality {
    /// Fast: trilinear only, no visibility weighting
    Fast = 0,
    /// Standard: trilinear + visibility weighting
    #[default]
    Standard = 1,
    /// High: full visibility + parallax correction
    High = 2,
}

impl DDGISampleQuality {
    /// Get the shader entry point name for this quality level.
    pub const fn entry_point(self) -> &'static str {
        match self {
            DDGISampleQuality::Fast => "ddgi_sample_irradiance_fast",
            DDGISampleQuality::Standard => "ddgi_sample_irradiance",
            DDGISampleQuality::High => "ddgi_sample_irradiance",
        }
    }

    /// Whether this quality level uses visibility weighting.
    pub const fn uses_visibility(self) -> bool {
        match self {
            DDGISampleQuality::Fast => false,
            DDGISampleQuality::Standard => true,
            DDGISampleQuality::High => true,
        }
    }

    /// Whether this quality level uses parallax correction.
    pub const fn uses_parallax(self) -> bool {
        match self {
            DDGISampleQuality::Fast => false,
            DDGISampleQuality::Standard => false,
            DDGISampleQuality::High => true,
        }
    }
}

// ============================================================================
// GPU Structures
// ============================================================================

/// Camera uniforms for DDGI sampling (matches WGSL struct).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct DDGICameraUniforms {
    /// View matrix (world -> view space).
    pub view: [[f32; 4]; 4],
    /// Projection matrix (view -> clip space).
    pub projection: [[f32; 4]; 4],
    /// Inverse projection matrix (clip -> view space).
    pub inv_projection: [[f32; 4]; 4],
    /// Camera world position.
    pub camera_position: [f32; 3],
    /// Padding for alignment.
    pub _pad: f32,
}

impl Default for DDGICameraUniforms {
    fn default() -> Self {
        Self {
            view: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            projection: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            inv_projection: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_position: [0.0, 0.0, 0.0],
            _pad: 0.0,
        }
    }
}

/// DDGI sample pass configuration.
#[derive(Clone, Debug)]
pub struct DDGISampleConfig {
    /// Output texture dimensions.
    pub width: u32,
    pub height: u32,
    /// Sampling quality preset.
    pub quality: DDGISampleQuality,
    /// Enable debug visualization.
    pub debug_mode: DDGIDebugMode,
}

impl Default for DDGISampleConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            quality: DDGISampleQuality::Standard,
            debug_mode: DDGIDebugMode::None,
        }
    }
}

/// Debug visualization modes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DDGIDebugMode {
    /// No debug output
    #[default]
    None = 0,
    /// Visualize interpolation weights
    Weights = 1,
    /// Visualize grid cells
    GridCells = 2,
}

impl DDGIDebugMode {
    /// Get the shader entry point for this debug mode.
    pub const fn entry_point(self) -> Option<&'static str> {
        match self {
            DDGIDebugMode::None => None,
            DDGIDebugMode::Weights => Some("ddgi_debug_weights"),
            DDGIDebugMode::GridCells => Some("ddgi_debug_grid_cell"),
        }
    }
}

// ============================================================================
// DDGI Sample Pass
// ============================================================================

/// DDGI probe sampling pass.
///
/// Samples the probe grid to compute indirect irradiance at each pixel.
#[derive(Debug)]
pub struct DDGISamplePass {
    /// Pass configuration.
    pub config: DDGISampleConfig,
    /// Workgroup size (matches shader).
    workgroup_size: [u32; 2],
}

impl Default for DDGISamplePass {
    fn default() -> Self {
        Self::new(DDGISampleConfig::default())
    }
}

impl DDGISamplePass {
    /// Create a new DDGI sample pass with the given configuration.
    pub fn new(config: DDGISampleConfig) -> Self {
        Self {
            config,
            workgroup_size: [8, 8],
        }
    }

    /// Create with specific dimensions.
    pub fn with_dimensions(width: u32, height: u32) -> Self {
        Self::new(DDGISampleConfig {
            width,
            height,
            ..Default::default()
        })
    }

    /// Calculate dispatch dimensions for the compute shader.
    pub fn dispatch_size(&self) -> [u32; 3] {
        let groups_x = (self.config.width + self.workgroup_size[0] - 1) / self.workgroup_size[0];
        let groups_y = (self.config.height + self.workgroup_size[1] - 1) / self.workgroup_size[1];
        [groups_x, groups_y, 1]
    }

    /// Get the shader entry point based on current mode.
    pub fn entry_point(&self) -> &'static str {
        if let Some(debug_entry) = self.config.debug_mode.entry_point() {
            debug_entry
        } else {
            self.config.quality.entry_point()
        }
    }

    /// Get the shader source path.
    pub const fn shader_path() -> &'static str {
        "shaders/ddgi_probe_sampling.wgsl"
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: DDGISampleConfig) {
        self.config = config;
    }

    /// Set debug mode.
    pub fn set_debug_mode(&mut self, mode: DDGIDebugMode) {
        self.config.debug_mode = mode;
    }

    /// Set quality preset.
    pub fn set_quality(&mut self, quality: DDGISampleQuality) {
        self.config.quality = quality;
    }

    /// Resize the output.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
    }
}

// ============================================================================
// Bind Group Layout
// ============================================================================

/// Binding indices for DDGI sample pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DDGISampleBindings {
    /// Probe grid uniform buffer.
    pub grid_uniform: u32,
    /// Probe SH storage buffer.
    pub probe_storage: u32,
    /// Camera uniforms.
    pub camera_uniform: u32,
    /// World position texture (G-buffer).
    pub world_position: u32,
    /// World normal texture (G-buffer).
    pub world_normal: u32,
    /// Depth texture.
    pub depth: u32,
    /// Irradiance output texture.
    pub irradiance_output: u32,
    /// Probe distance texture (optional).
    pub probe_distance: u32,
    /// Linear sampler.
    pub sampler: u32,
}

impl Default for DDGISampleBindings {
    fn default() -> Self {
        Self {
            grid_uniform: 0,
            probe_storage: 1,
            camera_uniform: 2,
            world_position: 3,
            world_normal: 4,
            depth: 5,
            irradiance_output: 6,
            probe_distance: 7,
            sampler: 8,
        }
    }
}

/// Descriptor for creating the bind group layout.
#[derive(Clone, Debug)]
pub struct DDGISampleBindGroupLayoutDesc {
    /// Binding indices.
    pub bindings: DDGISampleBindings,
    /// Whether probe distance texture is available.
    pub has_distance_texture: bool,
}

impl Default for DDGISampleBindGroupLayoutDesc {
    fn default() -> Self {
        Self {
            bindings: DDGISampleBindings::default(),
            has_distance_texture: false,
        }
    }
}

// ============================================================================
// Frame Graph Integration
// ============================================================================

/// Resources required for DDGI sampling.
#[derive(Clone, Debug)]
pub struct DDGISampleResources {
    /// Probe grid GPU data.
    pub grid: ProbeGridGpu,
    /// Camera uniforms.
    pub camera: DDGICameraUniforms,
    /// Output dimensions.
    pub output_width: u32,
    pub output_height: u32,
}

impl Default for DDGISampleResources {
    fn default() -> Self {
        Self {
            grid: ProbeGridGpu::default(),
            camera: DDGICameraUniforms::default(),
            output_width: 1920,
            output_height: 1080,
        }
    }
}

/// Create a DDGI sample pass for frame graph integration.
///
/// Returns the pass configuration and required resources.
pub fn create_ddgi_sample_pass(
    width: u32,
    height: u32,
    quality: DDGISampleQuality,
) -> (DDGISamplePass, DDGISampleResources) {
    let config = DDGISampleConfig {
        width,
        height,
        quality,
        debug_mode: DDGIDebugMode::None,
    };

    let pass = DDGISamplePass::new(config);
    let resources = DDGISampleResources {
        output_width: width,
        output_height: height,
        ..Default::default()
    };

    (pass, resources)
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute trilinear interpolation weights for 8 surrounding probes.
///
/// Given a world position and probe grid parameters, computes the weights
/// for trilinear interpolation. Returns (probe_indices, weights).
pub fn compute_trilinear_weights(
    world_pos: [f32; 3],
    grid: &ProbeGridGpu,
) -> ([u32; 8], [f32; 8]) {
    // World to grid coordinates
    let grid_pos = [
        (world_pos[0] - grid.origin[0]) / grid.cell_size[0],
        (world_pos[1] - grid.origin[1]) / grid.cell_size[1],
        (world_pos[2] - grid.origin[2]) / grid.cell_size[2],
    ];

    // Base cell index (floor)
    let base = [
        grid_pos[0].floor() as i32,
        grid_pos[1].floor() as i32,
        grid_pos[2].floor() as i32,
    ];

    // Fractional position within cell
    let frac = [
        (grid_pos[0] - base[0] as f32).clamp(0.0, 1.0),
        (grid_pos[1] - base[1] as f32).clamp(0.0, 1.0),
        (grid_pos[2] - base[2] as f32).clamp(0.0, 1.0),
    ];

    let mut indices = [0u32; 8];
    let mut weights = [0.0f32; 8];

    for i in 0..8 {
        let ox = (i & 1) as i32;
        let oy = ((i >> 1) & 1) as i32;
        let oz = ((i >> 2) & 1) as i32;

        // Corner index with clamping
        let cx = (base[0] + ox).clamp(0, grid.dimensions[0] as i32 - 1) as u32;
        let cy = (base[1] + oy).clamp(0, grid.dimensions[1] as i32 - 1) as u32;
        let cz = (base[2] + oz).clamp(0, grid.dimensions[2] as i32 - 1) as u32;

        // Linear index
        indices[i] = cx + cy * grid.dimensions[0] + cz * grid.dimensions[0] * grid.dimensions[1];

        // Trilinear weight
        let wx = if ox == 0 { 1.0 - frac[0] } else { frac[0] };
        let wy = if oy == 0 { 1.0 - frac[1] } else { frac[1] };
        let wz = if oz == 0 { 1.0 - frac[2] } else { frac[2] };
        weights[i] = wx * wy * wz;
    }

    (indices, weights)
}

/// Verify that trilinear weights sum to 1.0.
pub fn verify_trilinear_weights(weights: &[f32; 8]) -> bool {
    let sum: f32 = weights.iter().sum();
    (sum - 1.0).abs() < 1e-5
}

/// Apply scroll offset to grid index with wraparound.
pub fn apply_scroll_offset(
    grid_idx: [u32; 3],
    scroll_offset: [i32; 3],
    dimensions: [u32; 3],
) -> [u32; 3] {
    let wrap = |val: u32, offset: i32, dim: u32| -> u32 {
        let signed = (val as i32) + offset;
        let modded = signed.rem_euclid(dim as i32);
        modded as u32
    };
    [
        wrap(grid_idx[0], scroll_offset[0], dimensions[0]),
        wrap(grid_idx[1], scroll_offset[1], dimensions[1]),
        wrap(grid_idx[2], scroll_offset[2], dimensions[2]),
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // ── Quality preset tests ────────────────────────────────────────────────

    #[test]
    fn test_quality_entry_points() {
        assert_eq!(DDGISampleQuality::Fast.entry_point(), "ddgi_sample_irradiance_fast");
        assert_eq!(DDGISampleQuality::Standard.entry_point(), "ddgi_sample_irradiance");
        assert_eq!(DDGISampleQuality::High.entry_point(), "ddgi_sample_irradiance");
    }

    #[test]
    fn test_quality_uses_visibility() {
        assert!(!DDGISampleQuality::Fast.uses_visibility());
        assert!(DDGISampleQuality::Standard.uses_visibility());
        assert!(DDGISampleQuality::High.uses_visibility());
    }

    #[test]
    fn test_quality_uses_parallax() {
        assert!(!DDGISampleQuality::Fast.uses_parallax());
        assert!(!DDGISampleQuality::Standard.uses_parallax());
        assert!(DDGISampleQuality::High.uses_parallax());
    }

    #[test]
    fn test_quality_default() {
        let q: DDGISampleQuality = Default::default();
        assert_eq!(q, DDGISampleQuality::Standard);
    }

    // ── Debug mode tests ────────────────────────────────────────────────────

    #[test]
    fn test_debug_mode_entry_points() {
        assert_eq!(DDGIDebugMode::None.entry_point(), None);
        assert_eq!(DDGIDebugMode::Weights.entry_point(), Some("ddgi_debug_weights"));
        assert_eq!(DDGIDebugMode::GridCells.entry_point(), Some("ddgi_debug_grid_cell"));
    }

    #[test]
    fn test_debug_mode_default() {
        let m: DDGIDebugMode = Default::default();
        assert_eq!(m, DDGIDebugMode::None);
    }

    // ── Camera uniforms tests ───────────────────────────────────────────────

    #[test]
    fn test_camera_uniforms_size() {
        assert_eq!(std::mem::size_of::<DDGICameraUniforms>(), 208);
    }

    #[test]
    fn test_camera_uniforms_default() {
        let cam = DDGICameraUniforms::default();
        assert_eq!(cam.camera_position, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_camera_uniforms_pod_cast() {
        let cam = DDGICameraUniforms::default();
        let bytes: &[u8] = bytemuck::bytes_of(&cam);
        assert_eq!(bytes.len(), 208);
        let _restored: &DDGICameraUniforms = bytemuck::from_bytes(bytes);
    }

    // ── Config tests ────────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let config = DDGISampleConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.quality, DDGISampleQuality::Standard);
        assert_eq!(config.debug_mode, DDGIDebugMode::None);
    }

    // ── Pass tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_pass_default() {
        let pass = DDGISamplePass::default();
        assert_eq!(pass.config.width, 1920);
        assert_eq!(pass.config.height, 1080);
    }

    #[test]
    fn test_pass_with_dimensions() {
        let pass = DDGISamplePass::with_dimensions(800, 600);
        assert_eq!(pass.config.width, 800);
        assert_eq!(pass.config.height, 600);
    }

    #[test]
    fn test_pass_dispatch_size() {
        let pass = DDGISamplePass::with_dimensions(1920, 1080);
        let dispatch = pass.dispatch_size();
        // 1920/8 = 240, 1080/8 = 135
        assert_eq!(dispatch[0], 240);
        assert_eq!(dispatch[1], 135);
        assert_eq!(dispatch[2], 1);
    }

    #[test]
    fn test_pass_dispatch_size_non_divisible() {
        let pass = DDGISamplePass::with_dimensions(1921, 1081);
        let dispatch = pass.dispatch_size();
        // ceil(1921/8) = 241, ceil(1081/8) = 136
        assert_eq!(dispatch[0], 241);
        assert_eq!(dispatch[1], 136);
    }

    #[test]
    fn test_pass_entry_point_normal() {
        let pass = DDGISamplePass::default();
        assert_eq!(pass.entry_point(), "ddgi_sample_irradiance");
    }

    #[test]
    fn test_pass_entry_point_debug() {
        let mut pass = DDGISamplePass::default();
        pass.set_debug_mode(DDGIDebugMode::Weights);
        assert_eq!(pass.entry_point(), "ddgi_debug_weights");
    }

    #[test]
    fn test_pass_entry_point_fast() {
        let mut pass = DDGISamplePass::default();
        pass.set_quality(DDGISampleQuality::Fast);
        assert_eq!(pass.entry_point(), "ddgi_sample_irradiance_fast");
    }

    #[test]
    fn test_pass_resize() {
        let mut pass = DDGISamplePass::default();
        pass.resize(3840, 2160);
        assert_eq!(pass.config.width, 3840);
        assert_eq!(pass.config.height, 2160);
    }

    #[test]
    fn test_shader_path() {
        assert_eq!(DDGISamplePass::shader_path(), "shaders/ddgi_probe_sampling.wgsl");
    }

    // ── Bindings tests ──────────────────────────────────────────────────────

    #[test]
    fn test_bindings_default() {
        let bindings = DDGISampleBindings::default();
        assert_eq!(bindings.grid_uniform, 0);
        assert_eq!(bindings.probe_storage, 1);
        assert_eq!(bindings.camera_uniform, 2);
        assert_eq!(bindings.irradiance_output, 6);
    }

    // ── Resources tests ─────────────────────────────────────────────────────

    #[test]
    fn test_resources_default() {
        let res = DDGISampleResources::default();
        assert_eq!(res.output_width, 1920);
        assert_eq!(res.output_height, 1080);
    }

    // ── create_ddgi_sample_pass tests ───────────────────────────────────────

    #[test]
    fn test_create_ddgi_sample_pass() {
        let (pass, resources) = create_ddgi_sample_pass(800, 600, DDGISampleQuality::Fast);
        assert_eq!(pass.config.width, 800);
        assert_eq!(pass.config.height, 600);
        assert_eq!(pass.config.quality, DDGISampleQuality::Fast);
        assert_eq!(resources.output_width, 800);
        assert_eq!(resources.output_height, 600);
    }

    // ── Trilinear weights tests ─────────────────────────────────────────────

    #[test]
    fn test_trilinear_weights_at_origin() {
        let grid = ProbeGridGpu::default();
        let (indices, weights) = compute_trilinear_weights([0.0, 0.0, 0.0], &grid);

        // At grid origin, weight should be 1.0 at index 0
        assert_eq!(indices[0], 0);
        assert!(approx_eq(weights[0], 1.0));

        // Other weights should be 0
        for i in 1..8 {
            assert!(approx_eq(weights[i], 0.0));
        }
    }

    #[test]
    fn test_trilinear_weights_sum_to_one() {
        let grid = ProbeGridGpu::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [8, 4, 8]);

        // Test at various positions
        let test_positions = [
            [1.0, 1.0, 1.0],
            [3.5, 2.5, 1.5],
            [7.0, 3.0, 5.0],
            [0.5, 0.5, 0.5],
        ];

        for pos in &test_positions {
            let (_, weights) = compute_trilinear_weights(*pos, &grid);
            assert!(verify_trilinear_weights(&weights), "Weights don't sum to 1.0 at {:?}", pos);
        }
    }

    #[test]
    fn test_trilinear_weights_center_of_cell() {
        let grid = ProbeGridGpu::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [8, 4, 8]);
        let (_, weights) = compute_trilinear_weights([1.0, 1.0, 1.0], &grid);

        // At cell center, all 8 weights should be equal (0.125 each)
        for w in &weights {
            assert!(approx_eq(*w, 0.125), "Weight {} != 0.125", w);
        }
    }

    #[test]
    fn test_trilinear_weights_edge_of_cell() {
        let grid = ProbeGridGpu::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [8, 4, 8]);

        // Position at cell edge (x=2.0, y=1.0, z=1.0) = grid (1.0, 0.5, 0.5)
        let (_, weights) = compute_trilinear_weights([2.0, 1.0, 1.0], &grid);

        // x fraction = 0, y fraction = 0.5, z fraction = 0.5
        // Corners with x offset = 0 should have weight 0.25 each (2 such corners: yz variations)
        // Actually: (1-0)*(0.5)*(0.5) + (1-0)*(0.5)*(0.5) + ... = various

        assert!(verify_trilinear_weights(&weights));
    }

    #[test]
    fn test_trilinear_weights_verify_function() {
        let good_weights = [0.125; 8];
        assert!(verify_trilinear_weights(&good_weights));

        let bad_weights = [0.2; 8]; // Sum = 1.6
        assert!(!verify_trilinear_weights(&bad_weights));
    }

    // ── Scroll offset tests ─────────────────────────────────────────────────

    #[test]
    fn test_scroll_offset_no_offset() {
        let result = apply_scroll_offset([3, 2, 1], [0, 0, 0], [8, 4, 8]);
        assert_eq!(result, [3, 2, 1]);
    }

    #[test]
    fn test_scroll_offset_positive() {
        let result = apply_scroll_offset([3, 2, 1], [2, 1, 3], [8, 4, 8]);
        assert_eq!(result, [5, 3, 4]);
    }

    #[test]
    fn test_scroll_offset_wrap() {
        let result = apply_scroll_offset([7, 3, 7], [2, 2, 2], [8, 4, 8]);
        // (7+2) % 8 = 1, (3+2) % 4 = 1, (7+2) % 8 = 1
        assert_eq!(result, [1, 1, 1]);
    }

    #[test]
    fn test_scroll_offset_negative() {
        let result = apply_scroll_offset([0, 0, 0], [-1, -1, -1], [8, 4, 8]);
        // Euclidean mod: (-1) % 8 = 7, (-1) % 4 = 3
        assert_eq!(result, [7, 3, 7]);
    }

    #[test]
    fn test_scroll_offset_large_negative() {
        let result = apply_scroll_offset([0, 0, 0], [-10, -5, -20], [8, 4, 8]);
        // -10 % 8 = 6, -5 % 4 = 3, -20 % 8 = 4
        assert_eq!(result, [6, 3, 4]);
    }

    // ── Integration tests ───────────────────────────────────────────────────

    #[test]
    fn test_trilinear_indices_unique() {
        let grid = ProbeGridGpu::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [8, 4, 8]);
        let (indices, _) = compute_trilinear_weights([3.0, 3.0, 3.0], &grid);

        // All 8 indices should be unique (for interior positions)
        let mut sorted = indices;
        sorted.sort();
        for i in 0..7 {
            assert_ne!(sorted[i], sorted[i + 1], "Duplicate index found");
        }
    }

    #[test]
    fn test_trilinear_weights_boundary_clamping() {
        let grid = ProbeGridGpu::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [8, 4, 8]);

        // Position outside grid bounds (negative)
        let (_, weights) = compute_trilinear_weights([-5.0, -5.0, -5.0], &grid);
        assert!(verify_trilinear_weights(&weights));

        // Position outside grid bounds (positive)
        let (_, weights) = compute_trilinear_weights([100.0, 100.0, 100.0], &grid);
        assert!(verify_trilinear_weights(&weights));
    }

    #[test]
    fn test_grid_index_calculation() {
        let grid = ProbeGridGpu::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [8, 4, 8]);

        // Test linear index calculation
        // Index at (0,0,0) = 0
        // Index at (1,0,0) = 1
        // Index at (0,1,0) = 8
        // Index at (0,0,1) = 32

        let (indices, _) = compute_trilinear_weights([0.0, 0.0, 0.0], &grid);
        assert_eq!(indices[0], 0); // (0,0,0)

        let (indices, _) = compute_trilinear_weights([2.0, 0.0, 0.0], &grid);
        assert_eq!(indices[0], 1); // (1,0,0)

        let (indices, _) = compute_trilinear_weights([0.0, 2.0, 0.0], &grid);
        assert_eq!(indices[0], 8); // (0,1,0)

        let (indices, _) = compute_trilinear_weights([0.0, 0.0, 2.0], &grid);
        assert_eq!(indices[0], 32); // (0,0,1)
    }
}
