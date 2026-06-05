//! SDF Combinator Functions (T-DEMO-1.13 through T-DEMO-1.21)
//!
//! This module provides Signed Distance Field combinator functions for combining
//! SDF primitives in TRINITY's demoscene rendering system. Combinators allow
//! building complex shapes from simple primitives through boolean operations.
//!
//! # Operations
//!
//! * **Union**: Surface in either primitive (CSG OR)
//! * **Intersection**: Surface in both primitives (CSG AND)
//! * **Subtraction**: First primitive with second carved out (CSG DIFF)
//! * **Smooth variants**: C1-continuous blends at the junction
//! * **Displacement**: Noise-based surface perturbation
//!
//! # Material ID Propagation
//!
//! SDFs are represented as `vec2<f32>` where:
//! - `x`: signed distance to surface
//! - `y`: material ID (for shading)
//!
//! The combinator functions propagate material IDs correctly based on which
//! primitive's surface is "winning" the combination.
//!
//! # References
//!
//! * Inigo Quilez, "Smooth Minimum" - https://iquilezles.org/articles/smin/
//! * Inigo Quilez, "Distance Functions" - https://iquilezles.org/articles/distfunctions/

use std::fmt::Write;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default blend factor for smooth operations.
pub const DEFAULT_SMOOTH_K: f32 = 0.1;

/// Minimum blend factor to avoid numerical issues.
pub const MIN_SMOOTH_K: f32 = 0.0001;

/// Maximum practical blend factor.
pub const MAX_SMOOTH_K: f32 = 10.0;

/// Epsilon for floating-point comparisons.
const EPSILON: f32 = 1e-7;

// ---------------------------------------------------------------------------
// T-DEMO-1.13: min2 - Vec2 Comparison (distance, material_id)
// ---------------------------------------------------------------------------

/// Compares two SDF results and returns the one with smaller distance.
///
/// This is the fundamental building block for union operations. The input
/// vectors are `(distance, material_id)` pairs, and the function selects
/// the pair with the smaller distance value, preserving the associated
/// material ID.
///
/// # Arguments
///
/// * `a` - First SDF result: `(distance, material_id)`
/// * `b` - Second SDF result: `(distance, material_id)`
///
/// # Returns
///
/// The SDF result with the smaller distance value.
///
/// # Example
///
/// ```
/// use renderer_backend::sdf_combinators::min2;
///
/// let sphere = (1.5, 1.0);  // distance 1.5, material 1
/// let box_sdf = (0.8, 2.0); // distance 0.8, material 2
/// let result = min2(sphere, box_sdf);
/// assert_eq!(result, (0.8, 2.0)); // box wins (closer)
/// ```
#[inline]
pub fn min2(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    if a.0 <= b.0 {
        a
    } else {
        b
    }
}

/// Vectorized version of min2 using arrays.
#[inline]
pub fn min2_arr(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    if a[0] <= b[0] {
        a
    } else {
        b
    }
}

// ---------------------------------------------------------------------------
// T-DEMO-1.14: max2 - Vec2 Intersection
// ---------------------------------------------------------------------------

/// Compares two SDF results for intersection (returns larger distance).
///
/// This is the fundamental building block for intersection operations.
/// The surface exists only where both SDFs are inside (both distances
/// are negative or at the boundary).
///
/// # Arguments
///
/// * `a` - First SDF result: `(distance, material_id)`
/// * `b` - Second SDF result: `(distance, material_id)`
///
/// # Returns
///
/// The SDF result with the larger distance value. The material ID is
/// taken from whichever primitive's surface defines the boundary.
///
/// # Example
///
/// ```
/// use renderer_backend::sdf_combinators::max2;
///
/// let sphere = (-0.5, 1.0);  // inside sphere by 0.5
/// let box_sdf = (0.2, 2.0);  // outside box by 0.2
/// let result = max2(sphere, box_sdf);
/// assert_eq!(result, (0.2, 2.0)); // box surface dominates
/// ```
#[inline]
pub fn max2(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    if a.0 >= b.0 {
        a
    } else {
        b
    }
}

/// Vectorized version of max2 using arrays.
#[inline]
pub fn max2_arr(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    if a[0] >= b[0] {
        a
    } else {
        b
    }
}

// ---------------------------------------------------------------------------
// T-DEMO-1.15: sdf_union - Boolean Union
// ---------------------------------------------------------------------------

/// Computes the union of two SDFs (CSG OR).
///
/// The resulting surface exists wherever either primitive's surface exists.
/// This is equivalent to `min2` but provided with a more semantic name.
///
/// # Arguments
///
/// * `a` - First SDF result: `(distance, material_id)`
/// * `b` - Second SDF result: `(distance, material_id)`
///
/// # Returns
///
/// The union of the two SDFs with correct material propagation.
///
/// # Mathematical Property
///
/// `union(A, B) = min(d_A, d_B)` with material from the closer surface.
#[inline]
pub fn sdf_union(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    min2(a, b)
}

/// Array version of sdf_union.
#[inline]
pub fn sdf_union_arr(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    min2_arr(a, b)
}

// ---------------------------------------------------------------------------
// T-DEMO-1.16: sdf_intersection - Boolean Intersection
// ---------------------------------------------------------------------------

/// Computes the intersection of two SDFs (CSG AND).
///
/// The resulting surface exists only where both primitives' surfaces exist.
/// This is equivalent to `max2` but provided with a more semantic name.
///
/// # Arguments
///
/// * `a` - First SDF result: `(distance, material_id)`
/// * `b` - Second SDF result: `(distance, material_id)`
///
/// # Returns
///
/// The intersection of the two SDFs with correct material propagation.
///
/// # Mathematical Property
///
/// `intersection(A, B) = max(d_A, d_B)` with material from the surface
/// that defines the boundary (the farther one).
#[inline]
pub fn sdf_intersection(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    max2(a, b)
}

/// Array version of sdf_intersection.
#[inline]
pub fn sdf_intersection_arr(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    max2_arr(a, b)
}

// ---------------------------------------------------------------------------
// T-DEMO-1.17: sdf_subtraction - Boolean Subtraction
// ---------------------------------------------------------------------------

/// Computes the subtraction of one SDF from another (CSG DIFF).
///
/// The resulting surface is `a` with `b` carved out. Points inside `b`
/// are excluded from the result, creating a "cut" through `a`.
///
/// # Arguments
///
/// * `a` - Primary SDF to cut from: `(distance, material_id)`
/// * `b` - SDF to subtract (carve out): `(distance, material_id)`
///
/// # Returns
///
/// The difference `a - b` with correct material propagation.
///
/// # Mathematical Property
///
/// `subtraction(A, B) = max(d_A, -d_B)` - we negate B's distance to
/// create a "negative mold" and then intersect.
///
/// # Note
///
/// The order matters: `subtraction(a, b)` is NOT the same as `subtraction(b, a)`.
#[inline]
pub fn sdf_subtraction(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    // Negate b's distance to create the complement
    let neg_b = (-b.0, b.1);
    max2(a, neg_b)
}

/// Array version of sdf_subtraction.
#[inline]
pub fn sdf_subtraction_arr(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let neg_b = [-b[0], b[1]];
    max2_arr(a, neg_b)
}

// ---------------------------------------------------------------------------
// Helper: Polynomial Smooth Min/Max (Quilez's formulation)
// ---------------------------------------------------------------------------

/// Polynomial smooth minimum (C1 continuous).
///
/// This is Inigo Quilez's polynomial smooth-min formulation that provides
/// C1 continuity at the junction between two SDFs. The blend region has
/// width proportional to `k`.
///
/// # Arguments
///
/// * `a` - First distance value
/// * `b` - Second distance value
/// * `k` - Smoothness factor (larger = smoother blend, typical: 0.1-0.5)
///
/// # Returns
///
/// Smooth minimum of `a` and `b` with blend factor `k`.
///
/// # Mathematical Property
///
/// For `k = 0`, this degenerates to `min(a, b)`.
/// The derivative is continuous (C1) everywhere.
#[inline]
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    let k = k.max(EPSILON);
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// Polynomial smooth maximum (C1 continuous).
///
/// The smooth-max counterpart to `smin`. Used for smooth intersection.
///
/// # Arguments
///
/// * `a` - First distance value
/// * `b` - Second distance value
/// * `k` - Smoothness factor
///
/// # Returns
///
/// Smooth maximum of `a` and `b` with blend factor `k`.
#[inline]
pub fn smax(a: f32, b: f32, k: f32) -> f32 {
    let k = k.max(EPSILON);
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.max(b) + h * h * k * 0.25
}

/// Computes the blend factor for smooth operations.
///
/// This returns a value in [0, 1] indicating how much of each primitive
/// contributes to the blended surface. Used for material interpolation.
///
/// # Arguments
///
/// * `a` - First distance value
/// * `b` - Second distance value
/// * `k` - Smoothness factor
///
/// # Returns
///
/// Blend factor `t` where 0 = pure `a`, 1 = pure `b`.
#[inline]
pub fn smooth_blend_factor(a: f32, b: f32, k: f32) -> f32 {
    let k = k.max(EPSILON);
    let h = (k - (a - b).abs()).max(0.0) / k;
    // When a < b, we want t closer to 0 (prefer a)
    // When a > b, we want t closer to 1 (prefer b)
    if a <= b {
        h * h * 0.5
    } else {
        1.0 - h * h * 0.5
    }
}

// ---------------------------------------------------------------------------
// T-DEMO-1.18: sdf_smooth_union - Smooth Boolean Union
// ---------------------------------------------------------------------------

/// Computes a smooth union of two SDFs with C1 continuity.
///
/// Unlike hard union (min), smooth union creates a fillet at the junction
/// between the two surfaces. The fillet radius is controlled by `k`.
///
/// # Arguments
///
/// * `a` - First SDF result: `(distance, material_id)`
/// * `b` - Second SDF result: `(distance, material_id)`
/// * `k` - Smoothness factor (larger = wider fillet, typical: 0.1-0.5)
///
/// # Returns
///
/// The smooth union with interpolated material ID in the blend region.
///
/// # Mathematical Property
///
/// Uses Quilez's polynomial smooth-min:
/// ```text
/// h = max(k - |a - b|, 0) / k
/// result = min(a, b) - h^2 * k * 0.25
/// ```
#[inline]
pub fn sdf_smooth_union(a: (f32, f32), b: (f32, f32), k: f32) -> (f32, f32) {
    let k = k.max(MIN_SMOOTH_K).min(MAX_SMOOTH_K);
    let dist = smin(a.0, b.0, k);
    let t = smooth_blend_factor(a.0, b.0, k);
    let mat = a.1 * (1.0 - t) + b.1 * t;
    (dist, mat)
}

/// Array version of sdf_smooth_union.
#[inline]
pub fn sdf_smooth_union_arr(a: [f32; 2], b: [f32; 2], k: f32) -> [f32; 2] {
    let k = k.max(MIN_SMOOTH_K).min(MAX_SMOOTH_K);
    let dist = smin(a[0], b[0], k);
    let t = smooth_blend_factor(a[0], b[0], k);
    let mat = a[1] * (1.0 - t) + b[1] * t;
    [dist, mat]
}

// ---------------------------------------------------------------------------
// T-DEMO-1.19: sdf_smooth_intersection - Smooth Boolean Intersection
// ---------------------------------------------------------------------------

/// Computes a smooth intersection of two SDFs with C1 continuity.
///
/// Unlike hard intersection (max), smooth intersection creates rounded
/// corners at the junction. The rounding radius is controlled by `k`.
///
/// # Arguments
///
/// * `a` - First SDF result: `(distance, material_id)`
/// * `b` - Second SDF result: `(distance, material_id)`
/// * `k` - Smoothness factor (larger = rounder corners)
///
/// # Returns
///
/// The smooth intersection with interpolated material ID.
#[inline]
pub fn sdf_smooth_intersection(a: (f32, f32), b: (f32, f32), k: f32) -> (f32, f32) {
    let k = k.max(MIN_SMOOTH_K).min(MAX_SMOOTH_K);
    let dist = smax(a.0, b.0, k);
    // For intersection, the "winning" surface is the larger distance
    let t = smooth_blend_factor(-a.0, -b.0, k);
    let mat = a.1 * (1.0 - t) + b.1 * t;
    (dist, mat)
}

/// Array version of sdf_smooth_intersection.
#[inline]
pub fn sdf_smooth_intersection_arr(a: [f32; 2], b: [f32; 2], k: f32) -> [f32; 2] {
    let k = k.max(MIN_SMOOTH_K).min(MAX_SMOOTH_K);
    let dist = smax(a[0], b[0], k);
    let t = smooth_blend_factor(-a[0], -b[0], k);
    let mat = a[1] * (1.0 - t) + b[1] * t;
    [dist, mat]
}

// ---------------------------------------------------------------------------
// T-DEMO-1.20: sdf_smooth_subtraction - Smooth Boolean Subtraction
// ---------------------------------------------------------------------------

/// Computes a smooth subtraction of one SDF from another with C1 continuity.
///
/// Unlike hard subtraction, smooth subtraction creates rounded edges where
/// the cut is made. The rounding radius is controlled by `k`.
///
/// # Arguments
///
/// * `a` - Primary SDF to cut from: `(distance, material_id)`
/// * `b` - SDF to subtract (carve out): `(distance, material_id)`
/// * `k` - Smoothness factor (larger = rounder cut edges)
///
/// # Returns
///
/// The smooth difference `a - b` with interpolated material ID.
///
/// # Note
///
/// Order matters: this carves `b` out of `a`.
#[inline]
pub fn sdf_smooth_subtraction(a: (f32, f32), b: (f32, f32), k: f32) -> (f32, f32) {
    let k = k.max(MIN_SMOOTH_K).min(MAX_SMOOTH_K);
    // Subtraction is intersection with complement: max(a, -b)
    let dist = smax(a.0, -b.0, k);
    // Material from the surface that defines the boundary
    let t = smooth_blend_factor(-a.0, b.0, k);
    let mat = a.1 * (1.0 - t) + b.1 * t;
    (dist, mat)
}

/// Array version of sdf_smooth_subtraction.
#[inline]
pub fn sdf_smooth_subtraction_arr(a: [f32; 2], b: [f32; 2], k: f32) -> [f32; 2] {
    let k = k.max(MIN_SMOOTH_K).min(MAX_SMOOTH_K);
    let dist = smax(a[0], -b[0], k);
    let t = smooth_blend_factor(-a[0], b[0], k);
    let mat = a[1] * (1.0 - t) + b[1] * t;
    [dist, mat]
}

// ---------------------------------------------------------------------------
// T-DEMO-1.21: sdf_displaced - Noise Displacement
// ---------------------------------------------------------------------------

/// Applies noise-based displacement to an SDF distance.
///
/// Displacement perturbs the surface by adding a noise value scaled by
/// an amplitude factor. This creates organic, irregular surfaces from
/// smooth primitives.
///
/// # Arguments
///
/// * `base_dist` - Original SDF distance value
/// * `amplitude` - Displacement amplitude (typical: 0.01-0.5)
/// * `noise` - Noise value at the sample point (typically in [-1, 1])
///
/// # Returns
///
/// The displaced distance value: `base_dist + amplitude * noise`.
///
/// # Note
///
/// Large amplitudes can cause the SDF to become non-Lipschitz (gradient > 1),
/// which may cause ray marching artifacts. Keep `amplitude` small relative
/// to the feature size.
///
/// # Example
///
/// ```
/// use renderer_backend::sdf_combinators::sdf_displaced;
///
/// let sphere_dist = 0.5;
/// let noise = 0.3;
/// let amplitude = 0.1;
/// let displaced = sdf_displaced(sphere_dist, amplitude, noise);
/// assert!((displaced - 0.53).abs() < 1e-6);
/// ```
#[inline]
pub fn sdf_displaced(base_dist: f32, amplitude: f32, noise: f32) -> f32 {
    base_dist + amplitude * noise
}

/// Applies displacement to an SDF with material ID.
///
/// The material ID is preserved; only the distance is modified.
#[inline]
pub fn sdf_displaced_with_mat(sdf: (f32, f32), amplitude: f32, noise: f32) -> (f32, f32) {
    (sdf.0 + amplitude * noise, sdf.1)
}

/// Array version of sdf_displaced_with_mat.
#[inline]
pub fn sdf_displaced_arr(sdf: [f32; 2], amplitude: f32, noise: f32) -> [f32; 2] {
    [sdf[0] + amplitude * noise, sdf[1]]
}

// ---------------------------------------------------------------------------
// WGSL Code Generation
// ---------------------------------------------------------------------------

/// Generates WGSL code for all SDF combinator functions.
///
/// The generated code includes:
/// - `min2`, `max2` for vec2 comparison
/// - `sdf_union`, `sdf_intersection`, `sdf_subtraction` for boolean ops
/// - `smin`, `smax` helper functions for smooth operations
/// - `sdf_smooth_union`, `sdf_smooth_intersection`, `sdf_smooth_subtraction`
/// - `sdf_displaced` for noise displacement
///
/// # Returns
///
/// A String containing valid WGSL function definitions.
pub fn generate_wgsl_combinators() -> String {
    let mut code = String::with_capacity(4096);

    writeln!(
        code,
        "// SPDX-License-Identifier: MIT\n\
         //\n\
         // sdf_combinators.wgsl -- SDF combinator functions for boolean operations.\n\
         //\n\
         // These functions combine SDF primitives through union, intersection,\n\
         // subtraction, and smooth variants. SDFs are represented as vec2<f32>\n\
         // where x = distance, y = material_id.\n\
         //\n\
         // Reference: Inigo Quilez -- Smooth Minimum\n\
         // https://iquilezles.org/articles/smin/\n"
    )
    .unwrap();

    // T-DEMO-1.13: min2
    writeln!(
        code,
        "\n\
         // =============================================================================\n\
         // T-DEMO-1.13: min2 - Vec2 Comparison\n\
         // =============================================================================\n\n\
         /// Selects the SDF result with smaller distance.\n\
         ///   a        -- first SDF (distance, material_id)\n\
         ///   b        -- second SDF (distance, material_id)\n\
         ///   returns  -- SDF with smaller distance\n\
         fn min2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{\n\
             return select(b, a, a.x <= b.x);\n\
         }}\n"
    )
    .unwrap();

    // T-DEMO-1.14: max2
    writeln!(
        code,
        "\n\
         // =============================================================================\n\
         // T-DEMO-1.14: max2 - Vec2 Intersection\n\
         // =============================================================================\n\n\
         /// Selects the SDF result with larger distance.\n\
         ///   a        -- first SDF (distance, material_id)\n\
         ///   b        -- second SDF (distance, material_id)\n\
         ///   returns  -- SDF with larger distance\n\
         fn max2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{\n\
             return select(b, a, a.x >= b.x);\n\
         }}\n"
    )
    .unwrap();

    // T-DEMO-1.15: sdf_union
    writeln!(
        code,
        "\n\
         // =============================================================================\n\
         // T-DEMO-1.15: sdf_union - Boolean Union\n\
         // =============================================================================\n\n\
         /// Computes the union of two SDFs (CSG OR).\n\
         /// The surface exists wherever either primitive exists.\n\
         ///   a        -- first SDF (distance, material_id)\n\
         ///   b        -- second SDF (distance, material_id)\n\
         ///   returns  -- union result\n\
         fn sdf_union(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{\n\
             return min2(a, b);\n\
         }}\n"
    )
    .unwrap();

    // T-DEMO-1.16: sdf_intersection
    writeln!(
        code,
        "\n\
         // =============================================================================\n\
         // T-DEMO-1.16: sdf_intersection - Boolean Intersection\n\
         // =============================================================================\n\n\
         /// Computes the intersection of two SDFs (CSG AND).\n\
         /// The surface exists only where both primitives exist.\n\
         ///   a        -- first SDF (distance, material_id)\n\
         ///   b        -- second SDF (distance, material_id)\n\
         ///   returns  -- intersection result\n\
         fn sdf_intersection(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{\n\
             return max2(a, b);\n\
         }}\n"
    )
    .unwrap();

    // T-DEMO-1.17: sdf_subtraction
    writeln!(
        code,
        "\n\
         // =============================================================================\n\
         // T-DEMO-1.17: sdf_subtraction - Boolean Subtraction\n\
         // =============================================================================\n\n\
         /// Computes the subtraction of b from a (CSG DIFF).\n\
         /// Creates a by carving out b.\n\
         ///   a        -- primary SDF to cut from\n\
         ///   b        -- SDF to subtract\n\
         ///   returns  -- difference result (a - b)\n\
         fn sdf_subtraction(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{\n\
             return max2(a, vec2<f32>(-b.x, b.y));\n\
         }}\n"
    )
    .unwrap();

    // Helper: smin
    writeln!(
        code,
        "\n\
         // =============================================================================\n\
         // Smooth Min/Max Helpers (Quilez polynomial)\n\
         // =============================================================================\n\n\
         /// Polynomial smooth minimum (C1 continuous).\n\
         ///   a        -- first distance\n\
         ///   b        -- second distance\n\
         ///   k        -- smoothness factor (larger = smoother)\n\
         ///   returns  -- smooth minimum\n\
         fn smin(a: f32, b: f32, k: f32) -> f32 {{\n\
             let h = max(k - abs(a - b), 0.0) / k;\n\
             return min(a, b) - h * h * k * 0.25;\n\
         }}\n\n\
         /// Polynomial smooth maximum (C1 continuous).\n\
         ///   a        -- first distance\n\
         ///   b        -- second distance\n\
         ///   k        -- smoothness factor\n\
         ///   returns  -- smooth maximum\n\
         fn smax(a: f32, b: f32, k: f32) -> f32 {{\n\
             let h = max(k - abs(a - b), 0.0) / k;\n\
             return max(a, b) + h * h * k * 0.25;\n\
         }}\n\n\
         /// Computes blend factor for material interpolation.\n\
         ///   a        -- first distance\n\
         ///   b        -- second distance\n\
         ///   k        -- smoothness factor\n\
         ///   returns  -- blend factor [0, 1]\n\
         fn smooth_blend_factor(a: f32, b: f32, k: f32) -> f32 {{\n\
             let h = max(k - abs(a - b), 0.0) / k;\n\
             return select(1.0 - h * h * 0.5, h * h * 0.5, a <= b);\n\
         }}\n"
    )
    .unwrap();

    // T-DEMO-1.18: sdf_smooth_union
    writeln!(
        code,
        "\n\
         // =============================================================================\n\
         // T-DEMO-1.18: sdf_smooth_union - Smooth Boolean Union\n\
         // =============================================================================\n\n\
         /// Computes a smooth union with C1 continuity.\n\
         /// Creates a fillet at the junction between surfaces.\n\
         ///   a        -- first SDF (distance, material_id)\n\
         ///   b        -- second SDF (distance, material_id)\n\
         ///   k        -- smoothness factor (fillet radius)\n\
         ///   returns  -- smooth union result\n\
         fn sdf_smooth_union(a: vec2<f32>, b: vec2<f32>, k: f32) -> vec2<f32> {{\n\
             let dist = smin(a.x, b.x, k);\n\
             let t = smooth_blend_factor(a.x, b.x, k);\n\
             let mat = a.y * (1.0 - t) + b.y * t;\n\
             return vec2<f32>(dist, mat);\n\
         }}\n"
    )
    .unwrap();

    // T-DEMO-1.19: sdf_smooth_intersection
    writeln!(
        code,
        "\n\
         // =============================================================================\n\
         // T-DEMO-1.19: sdf_smooth_intersection - Smooth Boolean Intersection\n\
         // =============================================================================\n\n\
         /// Computes a smooth intersection with C1 continuity.\n\
         /// Creates rounded corners at the junction.\n\
         ///   a        -- first SDF (distance, material_id)\n\
         ///   b        -- second SDF (distance, material_id)\n\
         ///   k        -- smoothness factor (rounding radius)\n\
         ///   returns  -- smooth intersection result\n\
         fn sdf_smooth_intersection(a: vec2<f32>, b: vec2<f32>, k: f32) -> vec2<f32> {{\n\
             let dist = smax(a.x, b.x, k);\n\
             let t = smooth_blend_factor(-a.x, -b.x, k);\n\
             let mat = a.y * (1.0 - t) + b.y * t;\n\
             return vec2<f32>(dist, mat);\n\
         }}\n"
    )
    .unwrap();

    // T-DEMO-1.20: sdf_smooth_subtraction
    writeln!(
        code,
        "\n\
         // =============================================================================\n\
         // T-DEMO-1.20: sdf_smooth_subtraction - Smooth Boolean Subtraction\n\
         // =============================================================================\n\n\
         /// Computes a smooth subtraction with C1 continuity.\n\
         /// Creates rounded edges where the cut is made.\n\
         ///   a        -- primary SDF to cut from\n\
         ///   b        -- SDF to subtract\n\
         ///   k        -- smoothness factor (edge rounding)\n\
         ///   returns  -- smooth difference result\n\
         fn sdf_smooth_subtraction(a: vec2<f32>, b: vec2<f32>, k: f32) -> vec2<f32> {{\n\
             let dist = smax(a.x, -b.x, k);\n\
             let t = smooth_blend_factor(-a.x, b.x, k);\n\
             let mat = a.y * (1.0 - t) + b.y * t;\n\
             return vec2<f32>(dist, mat);\n\
         }}\n"
    )
    .unwrap();

    // T-DEMO-1.21: sdf_displaced
    writeln!(
        code,
        "\n\
         // =============================================================================\n\
         // T-DEMO-1.21: sdf_displaced - Noise Displacement\n\
         // =============================================================================\n\n\
         /// Applies noise-based displacement to an SDF.\n\
         /// Perturbs the surface by adding scaled noise.\n\
         ///   base_dist -- original SDF distance\n\
         ///   amplitude -- displacement amplitude\n\
         ///   noise     -- noise value at sample point\n\
         ///   returns   -- displaced distance\n\
         fn sdf_displaced(base_dist: f32, amplitude: f32, noise: f32) -> f32 {{\n\
             return base_dist + amplitude * noise;\n\
         }}\n\n\
         /// Applies displacement to SDF with material ID.\n\
         ///   sdf       -- (distance, material_id)\n\
         ///   amplitude -- displacement amplitude\n\
         ///   noise     -- noise value at sample point\n\
         ///   returns   -- displaced SDF with preserved material\n\
         fn sdf_displaced_with_mat(sdf: vec2<f32>, amplitude: f32, noise: f32) -> vec2<f32> {{\n\
             return vec2<f32>(sdf.x + amplitude * noise, sdf.y);\n\
         }}\n"
    )
    .unwrap();

    code
}

/// Returns the embedded WGSL combinator code as a static string.
///
/// This is the WGSL code that should be included in shaders that need
/// SDF combinator functionality. This re-exports from the demoscene module.
pub fn wgsl_combinators() -> &'static str {
    crate::demoscene::SDF_COMBINATORS
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPSILON: f32 = 1e-5;

    // =========================================================================
    // T-DEMO-1.13: min2 tests
    // =========================================================================

    #[test]
    fn test_min2_first_smaller() {
        let a = (0.5, 1.0);
        let b = (1.0, 2.0);
        let result = min2(a, b);
        assert_eq!(result, (0.5, 1.0));
    }

    #[test]
    fn test_min2_second_smaller() {
        let a = (1.0, 1.0);
        let b = (0.3, 2.0);
        let result = min2(a, b);
        assert_eq!(result, (0.3, 2.0));
    }

    #[test]
    fn test_min2_equal_prefers_first() {
        let a = (0.5, 1.0);
        let b = (0.5, 2.0);
        let result = min2(a, b);
        assert_eq!(result, (0.5, 1.0));
    }

    #[test]
    fn test_min2_negative_distances() {
        let a = (-0.5, 1.0);
        let b = (-0.2, 2.0);
        let result = min2(a, b);
        assert_eq!(result, (-0.5, 1.0));
    }

    #[test]
    fn test_min2_mixed_signs() {
        let a = (0.5, 1.0);
        let b = (-0.2, 2.0);
        let result = min2(a, b);
        assert_eq!(result, (-0.2, 2.0));
    }

    #[test]
    fn test_min2_arr_basic() {
        let a = [0.5, 1.0];
        let b = [1.0, 2.0];
        let result = min2_arr(a, b);
        assert_eq!(result, [0.5, 1.0]);
    }

    #[test]
    fn test_min2_preserves_material() {
        let sphere = (2.0, 5.0);
        let cube = (0.1, 10.0);
        let result = min2(sphere, cube);
        assert!((result.1 - 10.0).abs() < TEST_EPSILON);
    }

    // =========================================================================
    // T-DEMO-1.14: max2 tests
    // =========================================================================

    #[test]
    fn test_max2_first_larger() {
        let a = (1.0, 1.0);
        let b = (0.5, 2.0);
        let result = max2(a, b);
        assert_eq!(result, (1.0, 1.0));
    }

    #[test]
    fn test_max2_second_larger() {
        let a = (0.3, 1.0);
        let b = (1.0, 2.0);
        let result = max2(a, b);
        assert_eq!(result, (1.0, 2.0));
    }

    #[test]
    fn test_max2_equal_prefers_first() {
        let a = (0.5, 1.0);
        let b = (0.5, 2.0);
        let result = max2(a, b);
        assert_eq!(result, (0.5, 1.0));
    }

    #[test]
    fn test_max2_negative_distances() {
        let a = (-0.5, 1.0);
        let b = (-0.2, 2.0);
        let result = max2(a, b);
        assert_eq!(result, (-0.2, 2.0));
    }

    #[test]
    fn test_max2_mixed_signs() {
        let a = (-0.5, 1.0);
        let b = (0.2, 2.0);
        let result = max2(a, b);
        assert_eq!(result, (0.2, 2.0));
    }

    #[test]
    fn test_max2_arr_basic() {
        let a = [0.5, 1.0];
        let b = [1.0, 2.0];
        let result = max2_arr(a, b);
        assert_eq!(result, [1.0, 2.0]);
    }

    #[test]
    fn test_max2_preserves_material() {
        let sphere = (-1.0, 5.0);  // inside
        let cube = (0.5, 10.0);    // outside
        let result = max2(sphere, cube);
        assert!((result.1 - 10.0).abs() < TEST_EPSILON);
    }

    // =========================================================================
    // T-DEMO-1.15: sdf_union tests
    // =========================================================================

    #[test]
    fn test_union_basic() {
        let a = (1.0, 1.0);
        let b = (0.5, 2.0);
        let result = sdf_union(a, b);
        assert_eq!(result, (0.5, 2.0));
    }

    #[test]
    fn test_union_is_min() {
        let a = (0.3, 1.0);
        let b = (0.8, 2.0);
        let union_result = sdf_union(a, b);
        let min_result = min2(a, b);
        assert_eq!(union_result, min_result);
    }

    #[test]
    fn test_union_commutative_distance() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let ab = sdf_union(a, b);
        let ba = sdf_union(b, a);
        assert!((ab.0 - ba.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_union_inside_both() {
        let a = (-0.5, 1.0);
        let b = (-0.3, 2.0);
        let result = sdf_union(a, b);
        assert!(result.0 < 0.0);
        assert!((result.0 - (-0.5)).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_union_arr_matches_tuple() {
        let a_arr = [0.5, 1.0];
        let b_arr = [0.3, 2.0];
        let a_tup = (0.5, 1.0);
        let b_tup = (0.3, 2.0);
        let arr_result = sdf_union_arr(a_arr, b_arr);
        let tup_result = sdf_union(a_tup, b_tup);
        assert!((arr_result[0] - tup_result.0).abs() < TEST_EPSILON);
        assert!((arr_result[1] - tup_result.1).abs() < TEST_EPSILON);
    }

    // =========================================================================
    // T-DEMO-1.16: sdf_intersection tests
    // =========================================================================

    #[test]
    fn test_intersection_basic() {
        let a = (0.5, 1.0);
        let b = (1.0, 2.0);
        let result = sdf_intersection(a, b);
        assert_eq!(result, (1.0, 2.0));
    }

    #[test]
    fn test_intersection_is_max() {
        let a = (0.3, 1.0);
        let b = (0.8, 2.0);
        let intersection_result = sdf_intersection(a, b);
        let max_result = max2(a, b);
        assert_eq!(intersection_result, max_result);
    }

    #[test]
    fn test_intersection_inside_one() {
        let a = (-0.5, 1.0);  // inside a
        let b = (0.3, 2.0);   // outside b
        let result = sdf_intersection(a, b);
        assert!(result.0 > 0.0); // outside the intersection
    }

    #[test]
    fn test_intersection_inside_both() {
        let a = (-0.5, 1.0);
        let b = (-0.2, 2.0);
        let result = sdf_intersection(a, b);
        assert!(result.0 < 0.0);
        assert!((result.0 - (-0.2)).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_intersection_commutative_distance() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let ab = sdf_intersection(a, b);
        let ba = sdf_intersection(b, a);
        assert!((ab.0 - ba.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_intersection_arr_matches_tuple() {
        let a_arr = [0.5, 1.0];
        let b_arr = [0.3, 2.0];
        let a_tup = (0.5, 1.0);
        let b_tup = (0.3, 2.0);
        let arr_result = sdf_intersection_arr(a_arr, b_arr);
        let tup_result = sdf_intersection(a_tup, b_tup);
        assert!((arr_result[0] - tup_result.0).abs() < TEST_EPSILON);
        assert!((arr_result[1] - tup_result.1).abs() < TEST_EPSILON);
    }

    // =========================================================================
    // T-DEMO-1.17: sdf_subtraction tests
    // =========================================================================

    #[test]
    fn test_subtraction_basic() {
        let a = (0.5, 1.0);   // outside a
        let b = (1.0, 2.0);   // outside b (far from a)
        let result = sdf_subtraction(a, b);
        // max(0.5, -1.0) = 0.5
        assert!((result.0 - 0.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_subtraction_carves_hole() {
        let a = (-0.5, 1.0);  // inside a
        let b = (-0.3, 2.0);  // inside b (will be carved)
        let result = sdf_subtraction(a, b);
        // max(-0.5, 0.3) = 0.3 (point is outside after subtraction)
        assert!(result.0 > 0.0);
    }

    #[test]
    fn test_subtraction_not_commutative() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let ab = sdf_subtraction(a, b);
        let ba = sdf_subtraction(b, a);
        assert!((ab.0 - ba.0).abs() > 0.01);
    }

    #[test]
    fn test_subtraction_no_overlap() {
        let a = (0.5, 1.0);   // outside a
        let b = (2.0, 2.0);   // far outside b
        let result = sdf_subtraction(a, b);
        // When b is far away, subtraction doesn't change a much
        // max(0.5, -2.0) = 0.5
        assert!((result.0 - 0.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_subtraction_arr_matches_tuple() {
        let a_arr = [0.5, 1.0];
        let b_arr = [0.3, 2.0];
        let a_tup = (0.5, 1.0);
        let b_tup = (0.3, 2.0);
        let arr_result = sdf_subtraction_arr(a_arr, b_arr);
        let tup_result = sdf_subtraction(a_tup, b_tup);
        assert!((arr_result[0] - tup_result.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_subtraction_preserves_a_material_when_a_wins() {
        let a = (0.5, 1.0);
        let b = (2.0, 2.0);  // b far away, -b.x = -2.0, a wins
        let result = sdf_subtraction(a, b);
        assert!((result.1 - 1.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_subtraction_uses_b_material_when_b_carved() {
        let a = (-1.0, 1.0);  // deep inside a
        let b = (-0.1, 2.0);  // inside b, -b.x = 0.1
        let result = sdf_subtraction(a, b);
        // max(-1.0, 0.1) = 0.1, material from b
        assert!((result.1 - 2.0).abs() < TEST_EPSILON);
    }

    // =========================================================================
    // smin/smax helper tests
    // =========================================================================

    #[test]
    fn test_smin_k_zero_is_min() {
        let a = 0.5;
        let b = 0.3;
        let result = smin(a, b, 0.0);
        assert!((result - 0.3).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_smin_symmetric() {
        let a = 0.5;
        let b = 0.3;
        let k = 0.2;
        let ab = smin(a, b, k);
        let ba = smin(b, a, k);
        assert!((ab - ba).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_smin_less_than_min() {
        let a = 0.5;
        let b = 0.3;
        let k = 0.5;
        let result = smin(a, b, k);
        assert!(result < a.min(b));
    }

    #[test]
    fn test_smax_k_zero_is_max() {
        let a = 0.5;
        let b = 0.3;
        let result = smax(a, b, 0.0);
        assert!((result - 0.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_smax_symmetric() {
        let a = 0.5;
        let b = 0.3;
        let k = 0.2;
        let ab = smax(a, b, k);
        let ba = smax(b, a, k);
        assert!((ab - ba).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_smax_greater_than_max() {
        let a = 0.5;
        let b = 0.3;
        let k = 0.5;
        let result = smax(a, b, k);
        assert!(result > a.max(b));
    }

    #[test]
    fn test_smooth_blend_factor_at_equal() {
        let t = smooth_blend_factor(0.5, 0.5, 0.2);
        // When equal, blend should be 0.5
        assert!((t - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_smooth_blend_factor_a_smaller() {
        let t = smooth_blend_factor(0.1, 1.0, 0.2);
        // When a << b, prefer a (t near 0)
        assert!(t < 0.3);
    }

    #[test]
    fn test_smooth_blend_factor_b_smaller() {
        let t = smooth_blend_factor(1.0, 0.1, 0.2);
        // When b << a, prefer b (t near 1)
        assert!(t > 0.7);
    }

    // =========================================================================
    // T-DEMO-1.18: sdf_smooth_union tests
    // =========================================================================

    #[test]
    fn test_smooth_union_basic() {
        let a = (0.5, 1.0);
        let b = (0.45, 2.0);  // Closer values to trigger smooth blend
        let result = sdf_smooth_union(a, b, 0.2);
        // Should be less than or equal to hard min (0.45)
        // Smooth min subtracts h*h*k*0.25 where h = max(k - |a-b|, 0) / k
        // For |a-b| = 0.05 and k = 0.2: h = (0.2 - 0.05) / 0.2 = 0.75
        // Subtraction = 0.75 * 0.75 * 0.2 * 0.25 = 0.028125
        assert!(result.0 <= 0.45);
    }

    #[test]
    fn test_smooth_union_symmetric_distance() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let k = 0.2;
        let ab = sdf_smooth_union(a, b, k);
        let ba = sdf_smooth_union(b, a, k);
        assert!((ab.0 - ba.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_smooth_union_k_zero_degenerates() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let result = sdf_smooth_union(a, b, MIN_SMOOTH_K);
        // Should be very close to hard union
        let hard = sdf_union(a, b);
        assert!((result.0 - hard.0).abs() < 0.01);
    }

    #[test]
    fn test_smooth_union_material_interpolation() {
        let a = (0.5, 1.0);
        let b = (0.5, 3.0);  // Same distance, different material
        let result = sdf_smooth_union(a, b, 0.2);
        // Materials should blend to middle
        assert!(result.1 > 1.5 && result.1 < 2.5);
    }

    #[test]
    fn test_smooth_union_large_k() {
        let a = (0.5, 1.0);
        let b = (0.5, 2.0);
        let result = sdf_smooth_union(a, b, 2.0);
        // Larger k = more smoothing = smaller result
        assert!(result.0 < 0.5);
    }

    #[test]
    fn test_smooth_union_arr_matches_tuple() {
        let a_arr = [0.5, 1.0];
        let b_arr = [0.3, 2.0];
        let a_tup = (0.5, 1.0);
        let b_tup = (0.3, 2.0);
        let arr_result = sdf_smooth_union_arr(a_arr, b_arr, 0.2);
        let tup_result = sdf_smooth_union(a_tup, b_tup, 0.2);
        assert!((arr_result[0] - tup_result.0).abs() < TEST_EPSILON);
        assert!((arr_result[1] - tup_result.1).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_smooth_union_c1_continuity() {
        // Test that the derivative doesn't have a discontinuity
        let a = 0.5;
        let k = 0.2;
        let delta = 0.001;

        // Sample around the transition point (where a ~= b)
        let b_minus = a - delta;
        let b_center = a;
        let b_plus = a + delta;

        let d_minus = smin(a, b_minus, k);
        let d_center = smin(a, b_center, k);
        let d_plus = smin(a, b_plus, k);

        // Finite difference derivatives
        let slope_left = (d_center - d_minus) / delta;
        let slope_right = (d_plus - d_center) / delta;

        // C1 continuity means derivatives should be close
        assert!((slope_left - slope_right).abs() < 0.1);
    }

    // =========================================================================
    // T-DEMO-1.19: sdf_smooth_intersection tests
    // =========================================================================

    #[test]
    fn test_smooth_intersection_basic() {
        let a = (0.5, 1.0);
        let b = (0.45, 2.0);  // Closer values to trigger smooth blend
        let result = sdf_smooth_intersection(a, b, 0.2);
        // Should be greater than or equal to hard max (0.5)
        // Smooth max adds h*h*k*0.25 where h = max(k - |a-b|, 0) / k
        // For |a-b| = 0.05 and k = 0.2: h = (0.2 - 0.05) / 0.2 = 0.75
        // Addition = 0.75 * 0.75 * 0.2 * 0.25 = 0.028125
        assert!(result.0 >= 0.5);
    }

    #[test]
    fn test_smooth_intersection_symmetric_distance() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let k = 0.2;
        let ab = sdf_smooth_intersection(a, b, k);
        let ba = sdf_smooth_intersection(b, a, k);
        assert!((ab.0 - ba.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_smooth_intersection_k_zero_degenerates() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let result = sdf_smooth_intersection(a, b, MIN_SMOOTH_K);
        let hard = sdf_intersection(a, b);
        assert!((result.0 - hard.0).abs() < 0.01);
    }

    #[test]
    fn test_smooth_intersection_material_interpolation() {
        let a = (0.5, 1.0);
        let b = (0.5, 3.0);
        let result = sdf_smooth_intersection(a, b, 0.2);
        assert!(result.1 > 1.5 && result.1 < 2.5);
    }

    #[test]
    fn test_smooth_intersection_arr_matches_tuple() {
        let a_arr = [0.5, 1.0];
        let b_arr = [0.3, 2.0];
        let a_tup = (0.5, 1.0);
        let b_tup = (0.3, 2.0);
        let arr_result = sdf_smooth_intersection_arr(a_arr, b_arr, 0.2);
        let tup_result = sdf_smooth_intersection(a_tup, b_tup, 0.2);
        assert!((arr_result[0] - tup_result.0).abs() < TEST_EPSILON);
        assert!((arr_result[1] - tup_result.1).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_smooth_intersection_c1_continuity() {
        let a = 0.5;
        let k = 0.2;
        let delta = 0.001;

        let b_minus = a - delta;
        let b_center = a;
        let b_plus = a + delta;

        let d_minus = smax(a, b_minus, k);
        let d_center = smax(a, b_center, k);
        let d_plus = smax(a, b_plus, k);

        let slope_left = (d_center - d_minus) / delta;
        let slope_right = (d_plus - d_center) / delta;

        assert!((slope_left - slope_right).abs() < 0.1);
    }

    // =========================================================================
    // T-DEMO-1.20: sdf_smooth_subtraction tests
    // =========================================================================

    #[test]
    fn test_smooth_subtraction_basic() {
        let a = (-0.5, 1.0);  // inside a
        let b = (-0.3, 2.0);  // inside b
        let result = sdf_smooth_subtraction(a, b, 0.2);
        // Carving b from a, should push us outside
        assert!(result.0 > 0.0);
    }

    #[test]
    fn test_smooth_subtraction_not_symmetric() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let k = 0.2;
        let ab = sdf_smooth_subtraction(a, b, k);
        let ba = sdf_smooth_subtraction(b, a, k);
        assert!((ab.0 - ba.0).abs() > 0.01);
    }

    #[test]
    fn test_smooth_subtraction_k_zero_degenerates() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let result = sdf_smooth_subtraction(a, b, MIN_SMOOTH_K);
        let hard = sdf_subtraction(a, b);
        assert!((result.0 - hard.0).abs() < 0.02);
    }

    #[test]
    fn test_smooth_subtraction_arr_matches_tuple() {
        let a_arr = [0.5, 1.0];
        let b_arr = [0.3, 2.0];
        let a_tup = (0.5, 1.0);
        let b_tup = (0.3, 2.0);
        let arr_result = sdf_smooth_subtraction_arr(a_arr, b_arr, 0.2);
        let tup_result = sdf_smooth_subtraction(a_tup, b_tup, 0.2);
        assert!((arr_result[0] - tup_result.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_smooth_subtraction_c1_continuity() {
        let a = 0.5;
        let k = 0.2;
        let delta = 0.001;

        // Test around b = 0 (the interesting transition for subtraction)
        let b_minus = -delta;
        let b_center = 0.0;
        let b_plus = delta;

        let d_minus = smax(a, -b_minus, k);
        let d_center = smax(a, -b_center, k);
        let d_plus = smax(a, -b_plus, k);

        let slope_left = (d_center - d_minus) / delta;
        let slope_right = (d_plus - d_center) / delta;

        assert!((slope_left - slope_right).abs() < 0.15);
    }

    // =========================================================================
    // T-DEMO-1.21: sdf_displaced tests
    // =========================================================================

    #[test]
    fn test_displaced_basic() {
        let base_dist = 0.5;
        let amplitude = 0.1;
        let noise = 0.3;
        let result = sdf_displaced(base_dist, amplitude, noise);
        assert!((result - 0.53).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_displaced_zero_amplitude() {
        let base_dist = 0.5;
        let result = sdf_displaced(base_dist, 0.0, 1.0);
        assert!((result - 0.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_displaced_zero_noise() {
        let base_dist = 0.5;
        let result = sdf_displaced(base_dist, 0.1, 0.0);
        assert!((result - 0.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_displaced_negative_noise() {
        let base_dist = 0.5;
        let amplitude = 0.1;
        let noise = -0.5;
        let result = sdf_displaced(base_dist, amplitude, noise);
        assert!((result - 0.45).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_displaced_with_mat_preserves_material() {
        let sdf = (0.5, 7.0);
        let result = sdf_displaced_with_mat(sdf, 0.1, 0.3);
        assert!((result.1 - 7.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_displaced_with_mat_modifies_distance() {
        let sdf = (0.5, 1.0);
        let result = sdf_displaced_with_mat(sdf, 0.1, 0.3);
        assert!((result.0 - 0.53).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_displaced_arr_basic() {
        let sdf = [0.5, 1.0];
        let result = sdf_displaced_arr(sdf, 0.1, 0.3);
        assert!((result[0] - 0.53).abs() < TEST_EPSILON);
        assert!((result[1] - 1.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_displaced_large_amplitude() {
        let base_dist = 0.5;
        let amplitude = 2.0;
        let noise = 1.0;
        let result = sdf_displaced(base_dist, amplitude, noise);
        assert!((result - 2.5).abs() < TEST_EPSILON);
    }

    // =========================================================================
    // WGSL Generation tests
    // =========================================================================

    #[test]
    fn test_wgsl_generation_not_empty() {
        let wgsl = generate_wgsl_combinators();
        assert!(!wgsl.is_empty());
    }

    #[test]
    fn test_wgsl_contains_min2() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn min2("));
    }

    #[test]
    fn test_wgsl_contains_max2() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn max2("));
    }

    #[test]
    fn test_wgsl_contains_sdf_union() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn sdf_union("));
    }

    #[test]
    fn test_wgsl_contains_sdf_intersection() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn sdf_intersection("));
    }

    #[test]
    fn test_wgsl_contains_sdf_subtraction() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn sdf_subtraction("));
    }

    #[test]
    fn test_wgsl_contains_smin() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn smin("));
    }

    #[test]
    fn test_wgsl_contains_smax() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn smax("));
    }

    #[test]
    fn test_wgsl_contains_smooth_union() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn sdf_smooth_union("));
    }

    #[test]
    fn test_wgsl_contains_smooth_intersection() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn sdf_smooth_intersection("));
    }

    #[test]
    fn test_wgsl_contains_smooth_subtraction() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn sdf_smooth_subtraction("));
    }

    #[test]
    fn test_wgsl_contains_displaced() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn sdf_displaced("));
    }

    #[test]
    fn test_wgsl_contains_displaced_with_mat() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("fn sdf_displaced_with_mat("));
    }

    #[test]
    fn test_wgsl_has_task_ids() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("T-DEMO-1.13"));
        assert!(wgsl.contains("T-DEMO-1.14"));
        assert!(wgsl.contains("T-DEMO-1.15"));
        assert!(wgsl.contains("T-DEMO-1.16"));
        assert!(wgsl.contains("T-DEMO-1.17"));
        assert!(wgsl.contains("T-DEMO-1.18"));
        assert!(wgsl.contains("T-DEMO-1.19"));
        assert!(wgsl.contains("T-DEMO-1.20"));
        assert!(wgsl.contains("T-DEMO-1.21"));
    }

    #[test]
    fn test_wgsl_uses_vec2_f32() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("vec2<f32>"));
    }

    #[test]
    fn test_wgsl_uses_select() {
        let wgsl = generate_wgsl_combinators();
        assert!(wgsl.contains("select("));
    }

    // =========================================================================
    // Constants tests
    // =========================================================================

    #[test]
    fn test_default_smooth_k_valid() {
        assert!(DEFAULT_SMOOTH_K > 0.0);
        assert!(DEFAULT_SMOOTH_K <= 1.0);
    }

    #[test]
    fn test_min_smooth_k_positive() {
        assert!(MIN_SMOOTH_K > 0.0);
    }

    #[test]
    fn test_max_smooth_k_reasonable() {
        assert!(MAX_SMOOTH_K >= 1.0);
        assert!(MAX_SMOOTH_K <= 100.0);
    }

    // =========================================================================
    // Edge cases and stress tests
    // =========================================================================

    #[test]
    fn test_min2_very_large_values() {
        let a = (1e10, 1.0);
        let b = (1e10 + 1.0, 2.0);
        let result = min2(a, b);
        assert!((result.0 - 1e10).abs() < 1.0);
    }

    #[test]
    fn test_min2_very_small_values() {
        let a = (1e-10, 1.0);
        let b = (2e-10, 2.0);
        let result = min2(a, b);
        assert!((result.0 - 1e-10).abs() < 1e-11);
    }

    #[test]
    fn test_smooth_union_with_inf() {
        let a = (0.5, 1.0);
        let b = (f32::INFINITY, 2.0);
        let result = sdf_smooth_union(a, b, 0.2);
        assert!(result.0.is_finite());
    }

    #[test]
    fn test_smooth_operations_clamp_k() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        // Very small k should be clamped to MIN_SMOOTH_K
        let result = sdf_smooth_union(a, b, -1.0);
        assert!(result.0.is_finite());
        // Very large k should be clamped
        let result2 = sdf_smooth_union(a, b, 100.0);
        assert!(result2.0.is_finite());
    }

    #[test]
    fn test_displaced_with_large_noise() {
        let result = sdf_displaced(0.5, 0.1, 100.0);
        assert!((result - 10.5).abs() < TEST_EPSILON);
    }

    // =========================================================================
    // Associativity tests (important for CSG)
    // =========================================================================

    #[test]
    fn test_union_associative() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let c = (0.7, 3.0);
        let ab_c = sdf_union(sdf_union(a, b), c);
        let a_bc = sdf_union(a, sdf_union(b, c));
        assert!((ab_c.0 - a_bc.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_intersection_associative() {
        let a = (0.5, 1.0);
        let b = (0.3, 2.0);
        let c = (0.7, 3.0);
        let ab_c = sdf_intersection(sdf_intersection(a, b), c);
        let a_bc = sdf_intersection(a, sdf_intersection(b, c));
        assert!((ab_c.0 - a_bc.0).abs() < TEST_EPSILON);
    }
}
