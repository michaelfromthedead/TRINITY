// SPDX-License-Identifier: MIT
//
// sdf_plane.wgsl -- Signed distance function for an infinite plane.
//
// Computes the signed distance from a point p to an infinite plane defined
// by a normal vector n and an offset d. The plane is the set of points
// satisfying dot(p, n) + d = 0, where n is the plane normal and d is the
// signed distance from the origin to the plane along the normal direction.
//
// Convention: returns a scalar f32 signed distance. Positive values are
// in front of the plane, negative are behind.
//
// Reference: Inigo Quilez -- Signed Distance Functions
// https://iquilezles.org/articles/distfunctions/
//
// Edge cases:
//   - Zero normal: returns d (constant offset) as a finite fallback
//   - Unnormalized normal: automatically normalized for correct distance

// =============================================================================
// T-DEMO-1.6: Plane SDF
// =============================================================================

/// Signed distance from point p to a plane.
///
/// The plane is defined by its outward normal n and signed offset d from the
/// origin (dot(p, n) + d = 0). The normal is normalized internally, so non-unit
/// normals produce correct distances. A zero-length normal yields a constant
/// distance equal to d (degenerate plane).
///
///   p        -- query position
///   n        -- plane normal (does not need to be unit length; zero-safe)
///   d        -- signed distance from origin to the plane along the normal
///   returns  -- signed distance: positive above, zero on, negative below
fn sdPlane(p: vec3<f32>, n: vec3<f32>, d: f32) -> f32 {
    // Squared length of the normal; guard against zero-length
    let len_sq = dot(n, n);

    // When normal is zero, return the constant offset d.
    // Otherwise normalize n and compute signed distance.
    let result = dot(p, n / sqrt(len_sq)) + d;
    return select(result, d, len_sq < 1e-10);
}
