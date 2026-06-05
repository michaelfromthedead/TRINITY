// SPDX-License-Identifier: MIT
//
// sdf_combinators.wgsl -- SDF combinator functions for boolean operations.
//
// These functions combine SDF primitives through union, intersection,
// subtraction, and smooth variants. SDFs are represented as vec2<f32>
// where x = distance, y = material_id.
//
// Reference: Inigo Quilez -- Smooth Minimum
// https://iquilezles.org/articles/smin/

// =============================================================================
// T-DEMO-1.13: min2 - Vec2 Comparison
// =============================================================================

/// Selects the SDF result with smaller distance.
///   a        -- first SDF (distance, material_id)
///   b        -- second SDF (distance, material_id)
///   returns  -- SDF with smaller distance
fn min2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return select(b, a, a.x <= b.x);
}

// =============================================================================
// T-DEMO-1.14: max2 - Vec2 Intersection
// =============================================================================

/// Selects the SDF result with larger distance.
///   a        -- first SDF (distance, material_id)
///   b        -- second SDF (distance, material_id)
///   returns  -- SDF with larger distance
fn max2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return select(b, a, a.x >= b.x);
}

// =============================================================================
// T-DEMO-1.15: sdf_union - Boolean Union
// =============================================================================

/// Computes the union of two SDFs (CSG OR).
/// The surface exists wherever either primitive exists.
///   a        -- first SDF (distance, material_id)
///   b        -- second SDF (distance, material_id)
///   returns  -- union result
fn sdf_union(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return min2(a, b);
}

// =============================================================================
// T-DEMO-1.16: sdf_intersection - Boolean Intersection
// =============================================================================

/// Computes the intersection of two SDFs (CSG AND).
/// The surface exists only where both primitives exist.
///   a        -- first SDF (distance, material_id)
///   b        -- second SDF (distance, material_id)
///   returns  -- intersection result
fn sdf_intersection(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return max2(a, b);
}

// =============================================================================
// T-DEMO-1.17: sdf_subtraction - Boolean Subtraction
// =============================================================================

/// Computes the subtraction of b from a (CSG DIFF).
/// Creates a by carving out b.
///   a        -- primary SDF to cut from
///   b        -- SDF to subtract
///   returns  -- difference result (a - b)
fn sdf_subtraction(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return max2(a, vec2<f32>(-b.x, b.y));
}

// =============================================================================
// Smooth Min/Max Helpers (Quilez polynomial)
// =============================================================================

/// Polynomial smooth minimum (C1 continuous).
///   a        -- first distance
///   b        -- second distance
///   k        -- smoothness factor (larger = smoother)
///   returns  -- smooth minimum
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

/// Polynomial smooth maximum (C1 continuous).
///   a        -- first distance
///   b        -- second distance
///   k        -- smoothness factor
///   returns  -- smooth maximum
fn smax(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return max(a, b) + h * h * k * 0.25;
}

/// Computes blend factor for material interpolation.
///   a        -- first distance
///   b        -- second distance
///   k        -- smoothness factor
///   returns  -- blend factor [0, 1]
fn smooth_blend_factor(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return select(1.0 - h * h * 0.5, h * h * 0.5, a <= b);
}

// =============================================================================
// T-DEMO-1.18: sdf_smooth_union - Smooth Boolean Union
// =============================================================================

/// Computes a smooth union with C1 continuity.
/// Creates a fillet at the junction between surfaces.
///   a        -- first SDF (distance, material_id)
///   b        -- second SDF (distance, material_id)
///   k        -- smoothness factor (fillet radius)
///   returns  -- smooth union result
fn sdf_smooth_union(a: vec2<f32>, b: vec2<f32>, k: f32) -> vec2<f32> {
    let dist = smin(a.x, b.x, k);
    let t = smooth_blend_factor(a.x, b.x, k);
    let mat = a.y * (1.0 - t) + b.y * t;
    return vec2<f32>(dist, mat);
}

// =============================================================================
// T-DEMO-1.19: sdf_smooth_intersection - Smooth Boolean Intersection
// =============================================================================

/// Computes a smooth intersection with C1 continuity.
/// Creates rounded corners at the junction.
///   a        -- first SDF (distance, material_id)
///   b        -- second SDF (distance, material_id)
///   k        -- smoothness factor (rounding radius)
///   returns  -- smooth intersection result
fn sdf_smooth_intersection(a: vec2<f32>, b: vec2<f32>, k: f32) -> vec2<f32> {
    let dist = smax(a.x, b.x, k);
    let t = smooth_blend_factor(-a.x, -b.x, k);
    let mat = a.y * (1.0 - t) + b.y * t;
    return vec2<f32>(dist, mat);
}

// =============================================================================
// T-DEMO-1.20: sdf_smooth_subtraction - Smooth Boolean Subtraction
// =============================================================================

/// Computes a smooth subtraction with C1 continuity.
/// Creates rounded edges where the cut is made.
///   a        -- primary SDF to cut from
///   b        -- SDF to subtract
///   k        -- smoothness factor (edge rounding)
///   returns  -- smooth difference result
fn sdf_smooth_subtraction(a: vec2<f32>, b: vec2<f32>, k: f32) -> vec2<f32> {
    let dist = smax(a.x, -b.x, k);
    let t = smooth_blend_factor(-a.x, b.x, k);
    let mat = a.y * (1.0 - t) + b.y * t;
    return vec2<f32>(dist, mat);
}

// =============================================================================
// T-DEMO-1.21: sdf_displaced - Noise Displacement
// =============================================================================

/// Applies noise-based displacement to an SDF.
/// Perturbs the surface by adding scaled noise.
///   base_dist -- original SDF distance
///   amplitude -- displacement amplitude
///   noise     -- noise value at sample point
///   returns   -- displaced distance
fn sdf_displaced(base_dist: f32, amplitude: f32, noise: f32) -> f32 {
    return base_dist + amplitude * noise;
}

/// Applies displacement to SDF with material ID.
///   sdf       -- (distance, material_id)
///   amplitude -- displacement amplitude
///   noise     -- noise value at sample point
///   returns   -- displaced SDF with preserved material
fn sdf_displaced_with_mat(sdf: vec2<f32>, amplitude: f32, noise: f32) -> vec2<f32> {
    return vec2<f32>(sdf.x + amplitude * noise, sdf.y);
}
