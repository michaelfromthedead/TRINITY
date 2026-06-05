//! SDF Domain Operations for Demoscene Rendering (T-DEMO-1.22 through T-DEMO-1.27)
//!
//! This module provides domain transformation functions that modify the coordinate
//! space before evaluating signed distance functions (SDFs). These operations enable
//! complex visual effects like infinite tiling, symmetry, kaleidoscopic fractals,
//! and non-linear deformations.
//!
//! # Operations
//!
//! | Task       | Function                    | Description                           |
//! |------------|-----------------------------| --------------------------------------|
//! | T-DEMO-1.22| `domain_repeat`             | Infinite tiling via modulo            |
//! | T-DEMO-1.23| `domain_mirror`             | Reflection symmetry via abs()         |
//! | T-DEMO-1.24| `domain_fold_kifs`          | Kaleidoscopic iterated function system|
//! | T-DEMO-1.25| `domain_twist`              | Rotation proportional to height       |
//! | T-DEMO-1.26| `domain_bend`               | Curvature of coordinate axes          |
//! | T-DEMO-1.27| `domain_stretch`            | Anisotropic scaling along axis        |
//!
//! # Usage Pattern
//!
//! Domain operations transform the input position BEFORE evaluating the SDF:
//!
//! ```ignore
//! let p_repeated = domain_repeat(p, 2.0);
//! let d = sdf_sphere(p_repeated, 0.5); // Sphere repeated infinitely
//! ```
//!
//! # WGSL Code Generation
//!
//! Each operation includes a corresponding WGSL code generator for GPU shaders.
//! Use `generate_wgsl_*` functions to emit shader code.
//!
//! # References
//!
//! * Inigo Quilez, "Domain Deformations" - <https://iquilezles.org/articles/domaindeform/>
//! * Kaleidoscopic IFS fractals - <https://www.fractalforums.com/kaleidoscopic-(escape-time-ifs)/>

use std::f32::consts::{PI, TAU};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default epsilon for floating-point comparisons.
pub const EPSILON: f32 = 1e-6;

/// Minimum cell size to avoid division by zero.
pub const MIN_CELL_SIZE: f32 = 1e-8;

/// Minimum radius for bend operations.
pub const MIN_BEND_RADIUS: f32 = 1e-8;

/// Minimum scale factor for stretch operations.
pub const MIN_SCALE: f32 = 1e-8;

/// Default KIFS iterations.
pub const DEFAULT_KIFS_ITERATIONS: u32 = 6;

/// Maximum KIFS iterations (for safety).
pub const MAX_KIFS_ITERATIONS: u32 = 32;

// ---------------------------------------------------------------------------
// Axis Enumeration
// ---------------------------------------------------------------------------

/// Axis enumeration for domain operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis {
    /// Convert axis to unit vector.
    #[inline]
    pub fn to_unit_vector(self) -> [f32; 3] {
        match self {
            Axis::X => [1.0, 0.0, 0.0],
            Axis::Y => [0.0, 1.0, 0.0],
            Axis::Z => [0.0, 0.0, 1.0],
        }
    }

    /// Get axis index (0, 1, or 2).
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Get WGSL component accessor string.
    #[inline]
    pub fn wgsl_component(self) -> &'static str {
        match self {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        }
    }
}

// ---------------------------------------------------------------------------
// T-DEMO-1.22: Domain Repetition
// ---------------------------------------------------------------------------

/// Repeats space infinitely by folding position into a cell of given size.
///
/// This creates an infinite tiling effect where the SDF is evaluated only
/// within a single cell centered at the origin. Points outside the cell
/// are wrapped back using modulo arithmetic.
///
/// # Arguments
///
/// * `p` - Input position `[x, y, z]`
/// * `cell_size` - Size of the repeating cell (same for all axes)
///
/// # Returns
///
/// Position within the cell `[-cell_size/2, cell_size/2)` for each axis.
///
/// # Example
///
/// ```
/// use renderer_backend::sdf_domain_ops::domain_repeat;
///
/// let p = [5.3, 2.1, -1.7];
/// let repeated = domain_repeat(p, 2.0);
/// // repeated is now within [-1.0, 1.0) for each axis
/// assert!(repeated[0].abs() <= 1.0);
/// assert!(repeated[1].abs() <= 1.0);
/// assert!(repeated[2].abs() <= 1.0);
/// ```
#[inline]
pub fn domain_repeat(p: [f32; 3], cell_size: f32) -> [f32; 3] {
    let c = cell_size.max(MIN_CELL_SIZE);
    [
        p[0] - c * (p[0] / c).round(),
        p[1] - c * (p[1] / c).round(),
        p[2] - c * (p[2] / c).round(),
    ]
}

/// Repeats space with different cell sizes per axis.
///
/// # Arguments
///
/// * `p` - Input position
/// * `cell_sizes` - Cell size for each axis `[cx, cy, cz]`
#[inline]
pub fn domain_repeat_aniso(p: [f32; 3], cell_sizes: [f32; 3]) -> [f32; 3] {
    let cx = cell_sizes[0].max(MIN_CELL_SIZE);
    let cy = cell_sizes[1].max(MIN_CELL_SIZE);
    let cz = cell_sizes[2].max(MIN_CELL_SIZE);
    [
        p[0] - cx * (p[0] / cx).round(),
        p[1] - cy * (p[1] / cy).round(),
        p[2] - cz * (p[2] / cz).round(),
    ]
}

/// Returns the integer cell index for a position under tiling.
///
/// Use this for pseudo-random variation between cells (seed a hash with
/// the cell ID before evaluating the SDF).
///
/// # Arguments
///
/// * `p` - Input position
/// * `cell_size` - Size of the repeating cell
///
/// # Returns
///
/// Integer cell indices as `[ix, iy, iz]`.
#[inline]
pub fn domain_cell_id(p: [f32; 3], cell_size: f32) -> [i32; 3] {
    let c = cell_size.max(MIN_CELL_SIZE);
    [
        (p[0] / c).round() as i32,
        (p[1] / c).round() as i32,
        (p[2] / c).round() as i32,
    ]
}

/// Repeats space only within a limited range of cells.
///
/// Useful for limiting repetition to a finite region (e.g., only 5x5 tiles).
///
/// # Arguments
///
/// * `p` - Input position
/// * `cell_size` - Size of the repeating cell
/// * `limit` - Maximum number of cells in each direction (symmetric around origin)
#[inline]
pub fn domain_repeat_limited(p: [f32; 3], cell_size: f32, limit: [f32; 3]) -> [f32; 3] {
    let c = cell_size.max(MIN_CELL_SIZE);
    [
        p[0] - c * (p[0] / c).round().clamp(-limit[0], limit[0]),
        p[1] - c * (p[1] / c).round().clamp(-limit[1], limit[1]),
        p[2] - c * (p[2] / c).round().clamp(-limit[2], limit[2]),
    ]
}

// ---------------------------------------------------------------------------
// T-DEMO-1.23: Domain Mirroring
// ---------------------------------------------------------------------------

/// Mirrors space across a plane perpendicular to the specified axis.
///
/// Points with negative coordinates along the axis are reflected to positive,
/// creating bilateral symmetry. The mirror plane passes through the origin.
///
/// # Arguments
///
/// * `p` - Input position
/// * `axis` - Axis perpendicular to the mirror plane
///
/// # Example
///
/// ```
/// use renderer_backend::sdf_domain_ops::{domain_mirror, Axis};
///
/// let p = [-3.0, 2.0, 1.0];
/// let mirrored = domain_mirror(p, Axis::X);
/// assert_eq!(mirrored, [3.0, 2.0, 1.0]);
/// ```
#[inline]
pub fn domain_mirror(p: [f32; 3], axis: Axis) -> [f32; 3] {
    match axis {
        Axis::X => [p[0].abs(), p[1], p[2]],
        Axis::Y => [p[0], p[1].abs(), p[2]],
        Axis::Z => [p[0], p[1], p[2].abs()],
    }
}

/// Mirrors space across all three coordinate planes (octant symmetry).
///
/// This creates 8-fold symmetry by reflecting all coordinates to positive.
#[inline]
pub fn domain_mirror_all(p: [f32; 3]) -> [f32; 3] {
    [p[0].abs(), p[1].abs(), p[2].abs()]
}

/// Mirrors space across two axes (quadrant symmetry in a plane).
#[inline]
pub fn domain_mirror_two(p: [f32; 3], axis1: Axis, axis2: Axis) -> [f32; 3] {
    let mut result = p;
    result[axis1.index()] = result[axis1.index()].abs();
    result[axis2.index()] = result[axis2.index()].abs();
    result
}

/// Conditional mirror: only mirror if the coordinate is negative.
///
/// Returns the mirrored position and whether a mirror occurred.
#[inline]
pub fn domain_mirror_conditional(p: [f32; 3], axis: Axis) -> ([f32; 3], bool) {
    let idx = axis.index();
    let was_negative = p[idx] < 0.0;
    let mut result = p;
    result[idx] = result[idx].abs();
    (result, was_negative)
}

// ---------------------------------------------------------------------------
// T-DEMO-1.24: Kaleidoscopic Iterated Function System (KIFS)
// ---------------------------------------------------------------------------

/// Kaleidoscopic fold configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KifsConfig {
    /// Number of iterations.
    pub iterations: u32,
    /// Scale factor per iteration (for fractal zooming).
    pub scale: f32,
    /// Offset applied per iteration.
    pub offset: [f32; 3],
    /// Rotation angle per iteration (radians).
    pub rotation: f32,
}

impl Default for KifsConfig {
    fn default() -> Self {
        Self {
            iterations: DEFAULT_KIFS_ITERATIONS,
            scale: 2.0,
            offset: [1.0, 1.0, 1.0],
            rotation: PI / 4.0,
        }
    }
}

/// Applies a kaleidoscopic iterated function system (KIFS) fold.
///
/// KIFS creates complex fractal symmetry by repeatedly folding space,
/// rotating, and scaling. The result is a kaleidoscopic pattern with
/// self-similar structure at different scales.
///
/// # Arguments
///
/// * `p` - Input position
/// * `iterations` - Number of fold iterations (clamped to MAX_KIFS_ITERATIONS)
///
/// # Returns
///
/// Tuple of (transformed position, accumulated scale factor).
/// The scale factor should be used to compensate the SDF distance.
///
/// # Note
///
/// KIFS is NOT an isometry. The returned scale factor should be used to
/// divide the SDF result: `sdf_result / scale_factor`.
///
/// # Example
///
/// ```
/// use renderer_backend::sdf_domain_ops::domain_fold_kifs;
///
/// let p = [1.5, 0.5, 0.3];
/// let (folded, scale) = domain_fold_kifs(p, 6);
/// // Use: sdf_primitive(folded) / scale
/// ```
#[inline]
pub fn domain_fold_kifs(p: [f32; 3], iterations: u32) -> ([f32; 3], f32) {
    let iters = iterations.min(MAX_KIFS_ITERATIONS);
    let mut q = p;
    let mut scale = 1.0_f32;
    let fold_scale = 2.0;
    let offset = [1.0, 1.0, 1.0];

    for _ in 0..iters {
        // Fold: absolute value creates mirror symmetry
        q = [q[0].abs(), q[1].abs(), q[2].abs()];

        // Scale and translate
        q = [
            q[0] * fold_scale - offset[0] * (fold_scale - 1.0),
            q[1] * fold_scale - offset[1] * (fold_scale - 1.0),
            q[2] * fold_scale - offset[2] * (fold_scale - 1.0),
        ];

        scale *= fold_scale;
    }

    (q, scale)
}

/// Applies KIFS with custom configuration.
///
/// # Arguments
///
/// * `p` - Input position
/// * `config` - KIFS configuration parameters
#[inline]
pub fn domain_fold_kifs_config(p: [f32; 3], config: &KifsConfig) -> ([f32; 3], f32) {
    let iters = config.iterations.min(MAX_KIFS_ITERATIONS);
    let mut q = p;
    let mut scale = 1.0_f32;
    let s = config.scale.max(MIN_SCALE);

    for _ in 0..iters {
        // Fold
        q = [q[0].abs(), q[1].abs(), q[2].abs()];

        // Optional rotation around Z axis
        if config.rotation.abs() > EPSILON {
            let c = config.rotation.cos();
            let sin = config.rotation.sin();
            let rx = c * q[0] - sin * q[1];
            let ry = sin * q[0] + c * q[1];
            q[0] = rx;
            q[1] = ry;
        }

        // Scale and translate
        q = [
            q[0] * s - config.offset[0] * (s - 1.0),
            q[1] * s - config.offset[1] * (s - 1.0),
            q[2] * s - config.offset[2] * (s - 1.0),
        ];

        scale *= s;
    }

    (q, scale)
}

/// Hexagonal KIFS fold (6-fold rotational symmetry).
///
/// Creates hexagonal kaleidoscopic patterns by folding with 60-degree angles.
#[inline]
pub fn domain_fold_kifs_hex(p: [f32; 3], iterations: u32) -> ([f32; 3], f32) {
    let iters = iterations.min(MAX_KIFS_ITERATIONS);
    let mut q = p;
    let mut scale = 1.0_f32;
    let angle = TAU / 6.0; // 60 degrees
    let fold_scale = 2.0;

    for _ in 0..iters {
        // Fold across X
        q[0] = q[0].abs();
        q[1] = q[1].abs();

        // Rotate by 60 degrees
        let c = angle.cos();
        let s = angle.sin();
        let rx = c * q[0] - s * q[1];
        let ry = s * q[0] + c * q[1];
        q[0] = rx;
        q[1] = ry;

        // Scale
        q = [q[0] * fold_scale, q[1] * fold_scale, q[2] * fold_scale];
        scale *= fold_scale;
    }

    (q, scale)
}

// ---------------------------------------------------------------------------
// T-DEMO-1.25: Domain Twist
// ---------------------------------------------------------------------------

/// Twists space by rotating around an axis proportional to height along that axis.
///
/// The twist amount specifies how many radians of rotation per unit distance
/// along the twist axis.
///
/// # Arguments
///
/// * `p` - Input position
/// * `axis` - Axis to twist around
/// * `amount` - Twist rate in radians per unit
///
/// # Example
///
/// ```
/// use renderer_backend::sdf_domain_ops::{domain_twist, Axis};
///
/// let p = [1.0, 2.0, 0.0];
/// let twisted = domain_twist(p, Axis::Y, 0.5);
/// // Position is rotated around Y by 0.5 * 2.0 = 1.0 radians
/// ```
#[inline]
pub fn domain_twist(p: [f32; 3], axis: Axis, amount: f32) -> [f32; 3] {
    match axis {
        Axis::X => {
            // Twist around X: rotate YZ plane based on X
            let angle = amount * p[0];
            let c = angle.cos();
            let s = angle.sin();
            [p[0], c * p[1] - s * p[2], s * p[1] + c * p[2]]
        }
        Axis::Y => {
            // Twist around Y: rotate XZ plane based on Y
            let angle = amount * p[1];
            let c = angle.cos();
            let s = angle.sin();
            [c * p[0] - s * p[2], p[1], s * p[0] + c * p[2]]
        }
        Axis::Z => {
            // Twist around Z: rotate XY plane based on Z
            let angle = amount * p[2];
            let c = angle.cos();
            let s = angle.sin();
            [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
        }
    }
}

/// Returns the twist angle at a given position.
///
/// Useful for computing twist amount for gradients or normals.
#[inline]
pub fn domain_twist_angle(p: [f32; 3], axis: Axis, amount: f32) -> f32 {
    amount * p[axis.index()]
}

/// Applies a twist with variable rate based on a falloff function.
///
/// The twist rate diminishes based on distance from the axis.
///
/// # Arguments
///
/// * `p` - Input position
/// * `axis` - Axis to twist around
/// * `amount` - Maximum twist rate
/// * `falloff` - Distance at which twist rate halves
#[inline]
pub fn domain_twist_falloff(p: [f32; 3], axis: Axis, amount: f32, falloff: f32) -> [f32; 3] {
    let f = falloff.max(EPSILON);

    // Compute radial distance from axis
    let radial_dist_sq = match axis {
        Axis::X => p[1] * p[1] + p[2] * p[2],
        Axis::Y => p[0] * p[0] + p[2] * p[2],
        Axis::Z => p[0] * p[0] + p[1] * p[1],
    };

    // Smooth falloff
    let falloff_factor = 1.0 / (1.0 + radial_dist_sq / (f * f));
    let effective_amount = amount * falloff_factor;

    domain_twist(p, axis, effective_amount)
}

// ---------------------------------------------------------------------------
// T-DEMO-1.26: Domain Bend
// ---------------------------------------------------------------------------

/// Bends space along a circular arc.
///
/// The bend transforms straight lines parallel to one axis into circular
/// arcs of the specified radius. Larger radii produce gentler curves.
///
/// # Arguments
///
/// * `p` - Input position
/// * `axis` - Axis along which to bend (the "straight" direction)
/// * `radius` - Bend radius (larger = gentler curve)
///
/// # Example
///
/// ```
/// use renderer_backend::sdf_domain_ops::{domain_bend, Axis};
///
/// let p = [0.0, 1.0, 5.0];
/// let bent = domain_bend(p, Axis::Z, 10.0);
/// // Z-axis is bent into a circular arc of radius 10
/// ```
#[inline]
pub fn domain_bend(p: [f32; 3], axis: Axis, radius: f32) -> [f32; 3] {
    let r = radius.abs().max(MIN_BEND_RADIUS);

    // If radius is effectively infinite, return unchanged
    if radius.abs() < MIN_BEND_RADIUS {
        return p;
    }

    match axis {
        Axis::X => {
            // Bend along X: X becomes arc length, Y is radial
            let theta = p[0] / r;
            let c = theta.cos();
            let s = theta.sin();
            [
                (r + p[1]) * s,
                (r + p[1]) * c - r,
                p[2],
            ]
        }
        Axis::Y => {
            // Bend along Y: Y becomes arc length, X is radial
            let theta = p[1] / r;
            let c = theta.cos();
            let s = theta.sin();
            [
                (r + p[0]) * s,
                (r + p[0]) * c - r,
                p[2],
            ]
        }
        Axis::Z => {
            // Bend along Z: Z becomes arc length, X is radial (most common)
            let theta = p[2] / r;
            let c = theta.cos();
            let s = theta.sin();
            [
                (r + p[0]) * c - r,
                p[1],
                (r + p[0]) * s,
            ]
        }
    }
}

/// Computes the curvature at a point for a bend operation.
///
/// Returns the local curvature (1/radius) which affects SDF compensation.
#[inline]
pub fn domain_bend_curvature(radius: f32) -> f32 {
    let r = radius.abs().max(MIN_BEND_RADIUS);
    1.0 / r
}

/// Applies a bend with smooth transition at the edges.
///
/// # Arguments
///
/// * `p` - Input position
/// * `axis` - Axis along which to bend
/// * `radius` - Bend radius
/// * `start` - Position where bend starts
/// * `end` - Position where bend ends
#[inline]
pub fn domain_bend_clamped(
    p: [f32; 3],
    axis: Axis,
    radius: f32,
    start: f32,
    end: f32,
) -> [f32; 3] {
    let axis_coord = p[axis.index()];

    if axis_coord < start || axis_coord > end {
        // Outside bend region: return unchanged
        return p;
    }

    // Smooth blend factor
    let range = (end - start).max(EPSILON);
    let t = (axis_coord - start) / range;
    let blend = t * t * (3.0 - 2.0 * t); // smoothstep

    let bent = domain_bend(p, axis, radius);

    // Interpolate between original and bent
    [
        p[0] + blend * (bent[0] - p[0]),
        p[1] + blend * (bent[1] - p[1]),
        p[2] + blend * (bent[2] - p[2]),
    ]
}

// ---------------------------------------------------------------------------
// T-DEMO-1.27: Domain Stretch
// ---------------------------------------------------------------------------

/// Stretches space anisotropically along an axis.
///
/// Positive scale > 1 stretches (makes things thinner), scale < 1 compresses.
/// The SDF distance should be compensated by multiplying by the scale.
///
/// # Arguments
///
/// * `p` - Input position
/// * `axis` - Axis along which to stretch
/// * `scale` - Stretch factor (> 1 stretches, < 1 compresses)
///
/// # Example
///
/// ```
/// use renderer_backend::sdf_domain_ops::{domain_stretch, Axis};
///
/// let p = [1.0, 2.0, 3.0];
/// let stretched = domain_stretch(p, Axis::X, 2.0);
/// assert_eq!(stretched, [2.0, 2.0, 3.0]);
/// ```
#[inline]
pub fn domain_stretch(p: [f32; 3], axis: Axis, scale: f32) -> [f32; 3] {
    let s = scale.abs().max(MIN_SCALE);
    match axis {
        Axis::X => [p[0] * s, p[1], p[2]],
        Axis::Y => [p[0], p[1] * s, p[2]],
        Axis::Z => [p[0], p[1], p[2] * s],
    }
}

/// Stretches space with different factors per axis.
///
/// # Arguments
///
/// * `p` - Input position
/// * `scales` - Scale factors `[sx, sy, sz]`
#[inline]
pub fn domain_stretch_aniso(p: [f32; 3], scales: [f32; 3]) -> [f32; 3] {
    [
        p[0] * scales[0].abs().max(MIN_SCALE),
        p[1] * scales[1].abs().max(MIN_SCALE),
        p[2] * scales[2].abs().max(MIN_SCALE),
    ]
}

/// Stretches with volume preservation (determinant = 1).
///
/// Stretching along one axis compresses the others proportionally.
#[inline]
pub fn domain_stretch_volume_preserving(p: [f32; 3], axis: Axis, scale: f32) -> [f32; 3] {
    let s = scale.abs().max(MIN_SCALE);
    let inv_sqrt = 1.0 / s.sqrt();

    match axis {
        Axis::X => [p[0] * s, p[1] * inv_sqrt, p[2] * inv_sqrt],
        Axis::Y => [p[0] * inv_sqrt, p[1] * s, p[2] * inv_sqrt],
        Axis::Z => [p[0] * inv_sqrt, p[1] * inv_sqrt, p[2] * s],
    }
}

/// Computes the SDF scale compensation factor for stretch.
///
/// Due to anisotropic scaling, the SDF distance must be divided by this factor.
#[inline]
pub fn domain_stretch_compensation(scale: f32) -> f32 {
    let s = scale.abs().max(MIN_SCALE);
    s.min(1.0 / s)
}

// ---------------------------------------------------------------------------
// WGSL Code Generation
// ---------------------------------------------------------------------------

/// Generates WGSL code for domain repetition.
pub fn generate_wgsl_repeat() -> String {
    r#"
/// Repeats space infinitely with the given cell size.
fn domain_repeat(p: vec3<f32>, cell_size: f32) -> vec3<f32> {
    let c = max(cell_size, 1e-8);
    return p - c * round(p / c);
}

/// Returns the integer cell index for the given position.
fn domain_cell_id(p: vec3<f32>, cell_size: f32) -> vec3<i32> {
    let c = max(cell_size, 1e-8);
    return vec3<i32>(round(p / c));
}

/// Repeats space with anisotropic cell sizes.
fn domain_repeat_aniso(p: vec3<f32>, cell_sizes: vec3<f32>) -> vec3<f32> {
    let c = max(cell_sizes, vec3<f32>(1e-8));
    return p - c * round(p / c);
}
"#
    .to_string()
}

/// Generates WGSL code for domain mirroring.
pub fn generate_wgsl_mirror() -> String {
    r#"
/// Mirrors space across the YZ plane (x = 0).
fn domain_mirror_x(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(abs(p.x), p.y, p.z);
}

/// Mirrors space across the XZ plane (y = 0).
fn domain_mirror_y(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(p.x, abs(p.y), p.z);
}

/// Mirrors space across the XY plane (z = 0).
fn domain_mirror_z(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(p.x, p.y, abs(p.z));
}

/// Mirrors space across all three coordinate planes.
fn domain_mirror_all(p: vec3<f32>) -> vec3<f32> {
    return abs(p);
}
"#
    .to_string()
}

/// Generates WGSL code for KIFS fold.
pub fn generate_wgsl_kifs(iterations: u32) -> String {
    let iters = iterations.min(MAX_KIFS_ITERATIONS);
    format!(
        r#"
/// Kaleidoscopic IFS fold with {iters} iterations.
/// Returns (transformed position, scale factor).
fn domain_fold_kifs(p: vec3<f32>) -> vec4<f32> {{
    var q = p;
    var scale = 1.0;
    let fold_scale = 2.0;
    let offset = vec3<f32>(1.0, 1.0, 1.0);

    for (var i = 0u; i < {iters}u; i = i + 1u) {{
        q = abs(q);
        q = q * fold_scale - offset * (fold_scale - 1.0);
        scale = scale * fold_scale;
    }}

    return vec4<f32>(q, scale);
}}

/// Hexagonal KIFS fold with {iters} iterations.
fn domain_fold_kifs_hex(p: vec3<f32>) -> vec4<f32> {{
    var q = p;
    var scale = 1.0;
    let angle = 6.283185307179586 / 6.0; // 60 degrees
    let fold_scale = 2.0;
    let ca = cos(angle);
    let sa = sin(angle);

    for (var i = 0u; i < {iters}u; i = i + 1u) {{
        q.x = abs(q.x);
        q.y = abs(q.y);
        let rx = ca * q.x - sa * q.y;
        let ry = sa * q.x + ca * q.y;
        q.x = rx;
        q.y = ry;
        q = q * fold_scale;
        scale = scale * fold_scale;
    }}

    return vec4<f32>(q, scale);
}}
"#,
        iters = iters
    )
}

/// Generates WGSL code for domain twist.
pub fn generate_wgsl_twist() -> String {
    r#"
/// Twists space around the X axis.
fn domain_twist_x(p: vec3<f32>, amount: f32) -> vec3<f32> {
    let angle = amount * p.x;
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

/// Twists space around the Y axis.
fn domain_twist_y(p: vec3<f32>, amount: f32) -> vec3<f32> {
    let angle = amount * p.y;
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

/// Twists space around the Z axis.
fn domain_twist_z(p: vec3<f32>, amount: f32) -> vec3<f32> {
    let angle = amount * p.z;
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}
"#
    .to_string()
}

/// Generates WGSL code for domain bend.
pub fn generate_wgsl_bend() -> String {
    r#"
/// Bends space along the X axis with the given radius.
fn domain_bend_x(p: vec3<f32>, radius: f32) -> vec3<f32> {
    let r = max(abs(radius), 1e-8);
    if (abs(radius) < 1e-8) { return p; }
    let theta = p.x / r;
    let c = cos(theta);
    let s = sin(theta);
    return vec3<f32>((r + p.y) * s, (r + p.y) * c - r, p.z);
}

/// Bends space along the Y axis with the given radius.
fn domain_bend_y(p: vec3<f32>, radius: f32) -> vec3<f32> {
    let r = max(abs(radius), 1e-8);
    if (abs(radius) < 1e-8) { return p; }
    let theta = p.y / r;
    let c = cos(theta);
    let s = sin(theta);
    return vec3<f32>((r + p.x) * s, (r + p.x) * c - r, p.z);
}

/// Bends space along the Z axis with the given radius.
fn domain_bend_z(p: vec3<f32>, radius: f32) -> vec3<f32> {
    let r = max(abs(radius), 1e-8);
    if (abs(radius) < 1e-8) { return p; }
    let theta = p.z / r;
    let c = cos(theta);
    let s = sin(theta);
    return vec3<f32>((r + p.x) * c - r, p.y, (r + p.x) * s);
}
"#
    .to_string()
}

/// Generates WGSL code for domain stretch.
pub fn generate_wgsl_stretch() -> String {
    r#"
/// Stretches space along the X axis.
fn domain_stretch_x(p: vec3<f32>, scale: f32) -> vec3<f32> {
    let s = max(abs(scale), 1e-8);
    return vec3<f32>(p.x * s, p.y, p.z);
}

/// Stretches space along the Y axis.
fn domain_stretch_y(p: vec3<f32>, scale: f32) -> vec3<f32> {
    let s = max(abs(scale), 1e-8);
    return vec3<f32>(p.x, p.y * s, p.z);
}

/// Stretches space along the Z axis.
fn domain_stretch_z(p: vec3<f32>, scale: f32) -> vec3<f32> {
    let s = max(abs(scale), 1e-8);
    return vec3<f32>(p.x, p.y, p.z * s);
}

/// Stretches space with anisotropic scales.
fn domain_stretch_aniso(p: vec3<f32>, scales: vec3<f32>) -> vec3<f32> {
    let s = max(abs(scales), vec3<f32>(1e-8));
    return p * s;
}

/// Volume-preserving stretch along X.
fn domain_stretch_x_volume(p: vec3<f32>, scale: f32) -> vec3<f32> {
    let s = max(abs(scale), 1e-8);
    let inv_sqrt = 1.0 / sqrt(s);
    return vec3<f32>(p.x * s, p.y * inv_sqrt, p.z * inv_sqrt);
}
"#
    .to_string()
}

/// Generates all WGSL domain operation code.
pub fn generate_wgsl_all(kifs_iterations: u32) -> String {
    format!(
        "// SDF Domain Operations - Generated Code\n// T-DEMO-1.22 through T-DEMO-1.27\n{}\n{}\n{}\n{}\n{}\n{}",
        generate_wgsl_repeat(),
        generate_wgsl_mirror(),
        generate_wgsl_kifs(kifs_iterations),
        generate_wgsl_twist(),
        generate_wgsl_bend(),
        generate_wgsl_stretch()
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPSILON
    }

    fn vec_approx_eq(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    // =========================================================================
    // T-DEMO-1.22: Domain Repetition Tests
    // =========================================================================

    #[test]
    fn test_repeat_origin_unchanged() {
        let p = [0.0, 0.0, 0.0];
        let result = domain_repeat(p, 2.0);
        assert!(vec_approx_eq(result, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_repeat_within_cell() {
        let p = [0.5, 0.3, -0.2];
        let result = domain_repeat(p, 2.0);
        assert!(vec_approx_eq(result, p));
    }

    #[test]
    fn test_repeat_wraps_positive() {
        let p = [3.5, 0.0, 0.0];
        let result = domain_repeat(p, 2.0);
        // 3.5 / 2 = 1.75, round = 2, 3.5 - 2*2 = -0.5
        assert!(approx_eq(result[0], -0.5));
    }

    #[test]
    fn test_repeat_wraps_negative() {
        let p = [-3.5, 0.0, 0.0];
        let result = domain_repeat(p, 2.0);
        // -3.5 / 2 = -1.75, round = -2, -3.5 - 2*(-2) = 0.5
        assert!(approx_eq(result[0], 0.5));
    }

    #[test]
    fn test_repeat_period_correctness() {
        let cell_size = 3.0;
        for i in 0..10 {
            let x = i as f32 * cell_size;
            let result = domain_repeat([x, 0.0, 0.0], cell_size);
            // At exact multiples, should be near zero
            assert!(result[0].abs() < cell_size / 2.0 + TEST_EPSILON);
        }
    }

    #[test]
    fn test_repeat_all_axes() {
        let p = [5.3, -2.7, 8.1];
        let result = domain_repeat(p, 2.0);
        // All results should be within [-1, 1)
        assert!(result[0].abs() <= 1.0 + TEST_EPSILON);
        assert!(result[1].abs() <= 1.0 + TEST_EPSILON);
        assert!(result[2].abs() <= 1.0 + TEST_EPSILON);
    }

    #[test]
    fn test_repeat_small_cell() {
        let p = [0.1, 0.2, 0.3];
        let result = domain_repeat(p, 0.5);
        assert!(result[0].abs() <= 0.25 + TEST_EPSILON);
    }

    #[test]
    fn test_repeat_large_cell() {
        let p = [50.0, -30.0, 100.0];
        let result = domain_repeat(p, 1000.0);
        // Point is within cell, should be unchanged
        assert!(vec_approx_eq(result, p));
    }

    #[test]
    fn test_repeat_aniso_different_sizes() {
        let p = [5.0, 3.0, 7.0];
        let result = domain_repeat_aniso(p, [2.0, 1.0, 4.0]);
        assert!(result[0].abs() <= 1.0 + TEST_EPSILON);
        assert!(result[1].abs() <= 0.5 + TEST_EPSILON);
        assert!(result[2].abs() <= 2.0 + TEST_EPSILON);
    }

    #[test]
    fn test_cell_id_origin() {
        let p = [0.0, 0.0, 0.0];
        let id = domain_cell_id(p, 2.0);
        assert_eq!(id, [0, 0, 0]);
    }

    #[test]
    fn test_cell_id_positive() {
        let p = [5.0, 3.0, 7.0];
        let id = domain_cell_id(p, 2.0);
        // round(5.0/2.0)=round(2.5)=3, round(3.0/2.0)=round(1.5)=2, round(7.0/2.0)=round(3.5)=4
        assert_eq!(id, [3, 2, 4]);
    }

    #[test]
    fn test_cell_id_negative() {
        let p = [-5.0, -3.0, -7.0];
        let id = domain_cell_id(p, 2.0);
        // round(-2.5)=-3, round(-1.5)=-2, round(-3.5)=-4
        assert_eq!(id, [-3, -2, -4]);
    }

    #[test]
    fn test_repeat_limited_within_limit() {
        let p = [3.0, 2.0, 1.0];
        let result = domain_repeat_limited(p, 2.0, [5.0, 5.0, 5.0]);
        // Should behave like normal repeat within limits
        let normal = domain_repeat(p, 2.0);
        assert!(vec_approx_eq(result, normal));
    }

    #[test]
    fn test_repeat_limited_at_boundary() {
        let p = [100.0, 0.0, 0.0];
        let result = domain_repeat_limited(p, 2.0, [2.0, 2.0, 2.0]);
        // Should clamp to limit
        let expected_x = 100.0 - 2.0 * 2.0; // limit is 2 cells
        assert!(approx_eq(result[0], expected_x));
    }

    // =========================================================================
    // T-DEMO-1.23: Domain Mirror Tests
    // =========================================================================

    #[test]
    fn test_mirror_x_positive_unchanged() {
        let p = [3.0, 2.0, 1.0];
        let result = domain_mirror(p, Axis::X);
        assert!(vec_approx_eq(result, p));
    }

    #[test]
    fn test_mirror_x_negative_flipped() {
        let p = [-3.0, 2.0, 1.0];
        let result = domain_mirror(p, Axis::X);
        assert!(vec_approx_eq(result, [3.0, 2.0, 1.0]));
    }

    #[test]
    fn test_mirror_y_negative_flipped() {
        let p = [1.0, -5.0, 2.0];
        let result = domain_mirror(p, Axis::Y);
        assert!(vec_approx_eq(result, [1.0, 5.0, 2.0]));
    }

    #[test]
    fn test_mirror_z_negative_flipped() {
        let p = [1.0, 2.0, -7.0];
        let result = domain_mirror(p, Axis::Z);
        assert!(vec_approx_eq(result, [1.0, 2.0, 7.0]));
    }

    #[test]
    fn test_mirror_symmetry() {
        let p1 = [3.0, 2.0, 1.0];
        let p2 = [-3.0, 2.0, 1.0];
        let r1 = domain_mirror(p1, Axis::X);
        let r2 = domain_mirror(p2, Axis::X);
        assert!(vec_approx_eq(r1, r2));
    }

    #[test]
    fn test_mirror_all_octant() {
        let p = [-1.0, -2.0, -3.0];
        let result = domain_mirror_all(p);
        assert!(vec_approx_eq(result, [1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_mirror_all_positive_unchanged() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_mirror_all(p);
        assert!(vec_approx_eq(result, p));
    }

    #[test]
    fn test_mirror_two_axes() {
        let p = [-1.0, -2.0, 3.0];
        let result = domain_mirror_two(p, Axis::X, Axis::Y);
        assert!(vec_approx_eq(result, [1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_mirror_conditional_was_negative() {
        let p = [-5.0, 2.0, 1.0];
        let (result, was_neg) = domain_mirror_conditional(p, Axis::X);
        assert!(vec_approx_eq(result, [5.0, 2.0, 1.0]));
        assert!(was_neg);
    }

    #[test]
    fn test_mirror_conditional_was_positive() {
        let p = [5.0, 2.0, 1.0];
        let (result, was_neg) = domain_mirror_conditional(p, Axis::X);
        assert!(vec_approx_eq(result, p));
        assert!(!was_neg);
    }

    #[test]
    fn test_mirror_at_zero() {
        let p = [0.0, 2.0, 1.0];
        let result = domain_mirror(p, Axis::X);
        assert!(vec_approx_eq(result, p));
    }

    // =========================================================================
    // T-DEMO-1.24: KIFS Fold Tests
    // =========================================================================

    #[test]
    fn test_kifs_zero_iterations() {
        let p = [1.0, 2.0, 3.0];
        let (result, scale) = domain_fold_kifs(p, 0);
        assert!(vec_approx_eq(result, p));
        assert!(approx_eq(scale, 1.0));
    }

    #[test]
    fn test_kifs_one_iteration() {
        let p = [1.0, 2.0, 3.0];
        let (result, scale) = domain_fold_kifs(p, 1);
        // After abs and scale: abs(p) * 2 - offset * (2-1) = p*2 - 1
        let expected = [1.0, 3.0, 5.0];
        assert!(vec_approx_eq(result, expected));
        assert!(approx_eq(scale, 2.0));
    }

    #[test]
    fn test_kifs_scale_doubles_each_iteration() {
        let p = [0.5, 0.5, 0.5];
        for i in 1..=6 {
            let (_, scale) = domain_fold_kifs(p, i);
            assert!(approx_eq(scale, 2.0_f32.powi(i as i32)));
        }
    }

    #[test]
    fn test_kifs_convergence() {
        // After many iterations, scaled result should converge to a pattern
        let p = [0.1, 0.2, 0.3];
        let (result, scale) = domain_fold_kifs(p, 6);
        // Result / scale should remain bounded
        let normalized = [result[0] / scale, result[1] / scale, result[2] / scale];
        assert!(normalized[0].abs() < 10.0);
        assert!(normalized[1].abs() < 10.0);
        assert!(normalized[2].abs() < 10.0);
    }

    #[test]
    fn test_kifs_negative_input_folded() {
        let p = [-1.0, -2.0, -3.0];
        let (result, _) = domain_fold_kifs(p, 1);
        // abs() should make all positive
        assert!(result[0] >= 0.0);
        assert!(result[1] >= 0.0);
        assert!(result[2] >= 0.0);
    }

    #[test]
    fn test_kifs_symmetry() {
        let p1 = [1.0, 2.0, 3.0];
        let p2 = [-1.0, 2.0, 3.0];
        let (r1, _) = domain_fold_kifs(p1, 3);
        let (r2, _) = domain_fold_kifs(p2, 3);
        assert!(vec_approx_eq(r1, r2));
    }

    #[test]
    fn test_kifs_max_iterations_clamped() {
        let p = [1.0, 1.0, 1.0];
        let (_, scale1) = domain_fold_kifs(p, MAX_KIFS_ITERATIONS);
        let (_, scale2) = domain_fold_kifs(p, MAX_KIFS_ITERATIONS + 100);
        assert!(approx_eq(scale1, scale2));
    }

    #[test]
    fn test_kifs_config_custom() {
        let p = [1.0, 1.0, 1.0];
        let config = KifsConfig {
            iterations: 2,
            scale: 3.0,
            offset: [0.5, 0.5, 0.5],
            rotation: 0.0,
        };
        let (_, scale) = domain_fold_kifs_config(p, &config);
        assert!(approx_eq(scale, 9.0)); // 3^2
    }

    #[test]
    fn test_kifs_hex_symmetry() {
        let p = [1.0, 0.5, 0.0];
        let (result, _) = domain_fold_kifs_hex(p, 3);
        // Result should be defined
        assert!(result[0].is_finite());
        assert!(result[1].is_finite());
        assert!(result[2].is_finite());
    }

    #[test]
    fn test_kifs_default_config() {
        let config = KifsConfig::default();
        assert_eq!(config.iterations, DEFAULT_KIFS_ITERATIONS);
        assert!(approx_eq(config.scale, 2.0));
    }

    // =========================================================================
    // T-DEMO-1.25: Domain Twist Tests
    // =========================================================================

    #[test]
    fn test_twist_zero_amount() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_twist(p, Axis::Y, 0.0);
        assert!(vec_approx_eq(result, p));
    }

    #[test]
    fn test_twist_at_origin() {
        let p = [1.0, 0.0, 0.0];
        let result = domain_twist(p, Axis::Y, 1.0);
        // At y=0, no twist occurs
        assert!(vec_approx_eq(result, p));
    }

    #[test]
    fn test_twist_y_90_degrees() {
        let p = [1.0, PI / 2.0, 0.0];
        let result = domain_twist(p, Axis::Y, 1.0);
        // Angle = 1.0 * PI/2 = PI/2
        // cos(PI/2) = 0, sin(PI/2) = 1
        // x' = 0*1 - 1*0 = 0, z' = 1*1 + 0*0 = 1
        assert!(approx_eq(result[0], 0.0));
        assert!(approx_eq(result[2], 1.0));
    }

    #[test]
    fn test_twist_x_axis() {
        let p = [PI / 2.0, 1.0, 0.0];
        let result = domain_twist(p, Axis::X, 1.0);
        // Twist around X based on x coordinate
        assert!(approx_eq(result[1], 0.0));
        assert!(approx_eq(result[2], 1.0));
    }

    #[test]
    fn test_twist_z_axis() {
        let p = [1.0, 0.0, PI / 2.0];
        let result = domain_twist(p, Axis::Z, 1.0);
        // Twist around Z based on z coordinate
        assert!(approx_eq(result[0], 0.0));
        assert!(approx_eq(result[1], 1.0));
    }

    #[test]
    fn test_twist_angle_calculation() {
        let p = [0.0, 3.0, 0.0];
        let angle = domain_twist_angle(p, Axis::Y, 0.5);
        assert!(approx_eq(angle, 1.5));
    }

    #[test]
    fn test_twist_full_rotation() {
        let p = [1.0, TAU, 0.0];
        let result = domain_twist(p, Axis::Y, 1.0);
        // Full rotation should return to original
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[2], 0.0));
    }

    #[test]
    fn test_twist_negative_amount() {
        let p = [1.0, 1.0, 0.0];
        let r_pos = domain_twist(p, Axis::Y, 0.5);
        let r_neg = domain_twist(p, Axis::Y, -0.5);
        // Opposite twist directions
        assert!(approx_eq(r_pos[2], -r_neg[2]));
    }

    #[test]
    fn test_twist_falloff_at_axis() {
        let p = [0.0, 1.0, 0.0]; // On Y axis
        let result = domain_twist_falloff(p, Axis::Y, 1.0, 1.0);
        // Full twist at axis
        assert!(vec_approx_eq(result, domain_twist(p, Axis::Y, 1.0)));
    }

    #[test]
    fn test_twist_falloff_far_from_axis() {
        let p = [10.0, 1.0, 10.0]; // Far from Y axis
        let result = domain_twist_falloff(p, Axis::Y, 1.0, 1.0);
        // Reduced twist far from axis
        let full_twist = domain_twist(p, Axis::Y, 1.0);
        // Should be closer to original than full twist
        let dist_to_orig = (result[0] - p[0]).powi(2) + (result[2] - p[2]).powi(2);
        let dist_to_full = (full_twist[0] - p[0]).powi(2) + (full_twist[2] - p[2]).powi(2);
        assert!(dist_to_orig < dist_to_full);
    }

    // =========================================================================
    // T-DEMO-1.26: Domain Bend Tests
    // =========================================================================

    #[test]
    fn test_bend_zero_radius_identity() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_bend(p, Axis::Z, 0.0);
        assert!(vec_approx_eq(result, p));
    }

    #[test]
    fn test_bend_large_radius_near_identity() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_bend(p, Axis::Z, 1e6);
        // Very large radius = nearly straight
        // With r=1e6, theta=3/1e6=3e-6, cos(theta)~=1, sin(theta)~=theta
        // x' = (r + x) * cos(theta) - r ~= (r + x) - r = x for small theta
        // The bend transformation approaches identity as radius increases
        assert!((result[0] - p[0]).abs() < 0.1);
        assert!((result[2] - p[2]).abs() < 0.1);
    }

    #[test]
    fn test_bend_z_at_origin() {
        let p = [0.0, 0.0, 0.0];
        let result = domain_bend(p, Axis::Z, 10.0);
        // At origin, theta = 0, cos(0) = 1, sin(0) = 0
        // x' = (r + 0) * 1 - r = 0
        assert!(approx_eq(result[0], 0.0));
        assert!(approx_eq(result[2], 0.0));
    }

    #[test]
    fn test_bend_curvature() {
        let curv1 = domain_bend_curvature(10.0);
        let curv2 = domain_bend_curvature(20.0);
        assert!(approx_eq(curv1, 0.1));
        assert!(approx_eq(curv2, 0.05));
        assert!(curv1 > curv2); // Smaller radius = more curvature
    }

    #[test]
    fn test_bend_x_axis() {
        let p = [1.0, 0.0, 0.0];
        let result = domain_bend(p, Axis::X, 5.0);
        assert!(result[0].is_finite());
        assert!(result[1].is_finite());
    }

    #[test]
    fn test_bend_y_axis() {
        let p = [0.0, 1.0, 0.0];
        let result = domain_bend(p, Axis::Y, 5.0);
        assert!(result[0].is_finite());
        assert!(result[1].is_finite());
    }

    #[test]
    fn test_bend_preserves_y() {
        let p = [1.0, 5.0, 2.0];
        let result = domain_bend(p, Axis::Z, 10.0);
        // Y should be unchanged when bending along Z
        assert!(approx_eq(result[1], p[1]));
    }

    #[test]
    fn test_bend_quarter_circle() {
        // When bending along Z with radius r, at z = r*PI/2, we should rotate 90 degrees
        let r = 10.0;
        let p = [0.0, 0.0, r * PI / 2.0];
        let result = domain_bend(p, Axis::Z, r);
        // After 90 degree bend, x should be near -r, z should be near r
        assert!((result[0] - (-r)).abs() < 0.1);
        assert!((result[2] - r).abs() < 0.1);
    }

    #[test]
    fn test_bend_clamped_outside_range() {
        let p = [0.0, 0.0, -5.0]; // Outside bend range
        let result = domain_bend_clamped(p, Axis::Z, 10.0, 0.0, 10.0);
        assert!(vec_approx_eq(result, p));
    }

    #[test]
    fn test_bend_clamped_inside_range() {
        let p = [0.0, 0.0, 5.0]; // Inside bend range
        let result = domain_bend_clamped(p, Axis::Z, 10.0, 0.0, 10.0);
        // Should be blended between original and bent
        let bent = domain_bend(p, Axis::Z, 10.0);
        assert!(result[0] >= p[0].min(bent[0]) - TEST_EPSILON);
        assert!(result[0] <= p[0].max(bent[0]) + TEST_EPSILON);
    }

    // =========================================================================
    // T-DEMO-1.27: Domain Stretch Tests
    // =========================================================================

    #[test]
    fn test_stretch_identity() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_stretch(p, Axis::X, 1.0);
        assert!(vec_approx_eq(result, p));
    }

    #[test]
    fn test_stretch_x_double() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_stretch(p, Axis::X, 2.0);
        assert!(vec_approx_eq(result, [2.0, 2.0, 3.0]));
    }

    #[test]
    fn test_stretch_y_half() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_stretch(p, Axis::Y, 0.5);
        assert!(vec_approx_eq(result, [1.0, 1.0, 3.0]));
    }

    #[test]
    fn test_stretch_z() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_stretch(p, Axis::Z, 3.0);
        assert!(vec_approx_eq(result, [1.0, 2.0, 9.0]));
    }

    #[test]
    fn test_stretch_aniso() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_stretch_aniso(p, [2.0, 3.0, 4.0]);
        assert!(vec_approx_eq(result, [2.0, 6.0, 12.0]));
    }

    #[test]
    fn test_stretch_volume_preserving() {
        let p = [1.0, 1.0, 1.0];
        let result = domain_stretch_volume_preserving(p, Axis::X, 4.0);
        // x * 4, y / 2, z / 2
        assert!(approx_eq(result[0], 4.0));
        assert!(approx_eq(result[1], 0.5));
        assert!(approx_eq(result[2], 0.5));
        // Volume: 4 * 0.5 * 0.5 = 1.0 (preserved)
        assert!(approx_eq(result[0] * result[1] * result[2], 1.0));
    }

    #[test]
    fn test_stretch_compensation_scale_gt_1() {
        let comp = domain_stretch_compensation(2.0);
        assert!(approx_eq(comp, 0.5)); // min(2, 0.5) = 0.5
    }

    #[test]
    fn test_stretch_compensation_scale_lt_1() {
        let comp = domain_stretch_compensation(0.5);
        assert!(approx_eq(comp, 0.5)); // min(0.5, 2) = 0.5
    }

    #[test]
    fn test_stretch_compensation_identity() {
        let comp = domain_stretch_compensation(1.0);
        assert!(approx_eq(comp, 1.0));
    }

    #[test]
    fn test_stretch_negative_scale_uses_abs() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_stretch(p, Axis::X, -2.0);
        assert!(vec_approx_eq(result, [2.0, 2.0, 3.0]));
    }

    #[test]
    fn test_stretch_near_zero_clamped() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_stretch(p, Axis::X, 1e-20);
        // Should use MIN_SCALE instead of zero
        assert!(result[0] > 0.0);
        assert!(result[0].is_finite());
    }

    // =========================================================================
    // WGSL Code Generation Tests
    // =========================================================================

    #[test]
    fn test_wgsl_repeat_contains_function() {
        let code = generate_wgsl_repeat();
        assert!(code.contains("fn domain_repeat"));
        assert!(code.contains("fn domain_cell_id"));
        assert!(code.contains("fn domain_repeat_aniso"));
    }

    #[test]
    fn test_wgsl_mirror_contains_functions() {
        let code = generate_wgsl_mirror();
        assert!(code.contains("fn domain_mirror_x"));
        assert!(code.contains("fn domain_mirror_y"));
        assert!(code.contains("fn domain_mirror_z"));
        assert!(code.contains("fn domain_mirror_all"));
    }

    #[test]
    fn test_wgsl_kifs_contains_iterations() {
        let code = generate_wgsl_kifs(6);
        assert!(code.contains("6u"));
        assert!(code.contains("fn domain_fold_kifs"));
    }

    #[test]
    fn test_wgsl_kifs_different_iterations() {
        let code1 = generate_wgsl_kifs(4);
        let code2 = generate_wgsl_kifs(8);
        assert!(code1.contains("4u"));
        assert!(code2.contains("8u"));
    }

    #[test]
    fn test_wgsl_twist_contains_functions() {
        let code = generate_wgsl_twist();
        assert!(code.contains("fn domain_twist_x"));
        assert!(code.contains("fn domain_twist_y"));
        assert!(code.contains("fn domain_twist_z"));
    }

    #[test]
    fn test_wgsl_bend_contains_functions() {
        let code = generate_wgsl_bend();
        assert!(code.contains("fn domain_bend_x"));
        assert!(code.contains("fn domain_bend_y"));
        assert!(code.contains("fn domain_bend_z"));
    }

    #[test]
    fn test_wgsl_stretch_contains_functions() {
        let code = generate_wgsl_stretch();
        assert!(code.contains("fn domain_stretch_x"));
        assert!(code.contains("fn domain_stretch_y"));
        assert!(code.contains("fn domain_stretch_z"));
        assert!(code.contains("fn domain_stretch_aniso"));
    }

    #[test]
    fn test_wgsl_all_combined() {
        let code = generate_wgsl_all(6);
        assert!(code.contains("domain_repeat"));
        assert!(code.contains("domain_mirror"));
        assert!(code.contains("domain_fold_kifs"));
        assert!(code.contains("domain_twist"));
        assert!(code.contains("domain_bend"));
        assert!(code.contains("domain_stretch"));
    }

    // =========================================================================
    // Axis Enumeration Tests
    // =========================================================================

    #[test]
    fn test_axis_to_unit_vector() {
        assert_eq!(Axis::X.to_unit_vector(), [1.0, 0.0, 0.0]);
        assert_eq!(Axis::Y.to_unit_vector(), [0.0, 1.0, 0.0]);
        assert_eq!(Axis::Z.to_unit_vector(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_axis_index() {
        assert_eq!(Axis::X.index(), 0);
        assert_eq!(Axis::Y.index(), 1);
        assert_eq!(Axis::Z.index(), 2);
    }

    #[test]
    fn test_axis_wgsl_component() {
        assert_eq!(Axis::X.wgsl_component(), "x");
        assert_eq!(Axis::Y.wgsl_component(), "y");
        assert_eq!(Axis::Z.wgsl_component(), "z");
    }

    // =========================================================================
    // Edge Cases and Numerical Stability
    // =========================================================================

    #[test]
    fn test_repeat_large_position() {
        let p = [1e10, 1e10, 1e10];
        let result = domain_repeat(p, 2.0);
        assert!(result[0].is_finite());
        assert!(result[1].is_finite());
        assert!(result[2].is_finite());
    }

    #[test]
    fn test_twist_large_height() {
        let p = [1.0, 1e6, 0.0];
        let result = domain_twist(p, Axis::Y, 0.001);
        assert!(result[0].is_finite());
        assert!(result[2].is_finite());
    }

    #[test]
    fn test_bend_small_radius() {
        let p = [1.0, 2.0, 3.0];
        let result = domain_bend(p, Axis::Z, 0.01);
        assert!(result[0].is_finite());
        assert!(result[2].is_finite());
    }

    #[test]
    fn test_kifs_origin() {
        let p = [0.0, 0.0, 0.0];
        let (result, scale) = domain_fold_kifs(p, 6);
        // Origin stays bounded after folding
        assert!(result[0].is_finite());
        assert!(scale.is_finite());
    }

    #[test]
    fn test_all_operations_with_nan_input() {
        // NaN should propagate (not crash)
        let p = [f32::NAN, 1.0, 1.0];

        let r1 = domain_repeat(p, 2.0);
        assert!(r1[0].is_nan());

        let r2 = domain_mirror(p, Axis::X);
        assert!(r2[0].is_nan());

        let (r3, _) = domain_fold_kifs(p, 1);
        assert!(r3[0].is_nan());
    }

    #[test]
    fn test_all_operations_with_inf_input() {
        let p = [f32::INFINITY, 1.0, 1.0];

        let r1 = domain_stretch(p, Axis::X, 2.0);
        assert!(r1[0].is_infinite());
    }
}
