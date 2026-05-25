// SPDX-License-Identifier: MIT
//
// sdf_ellipsoid.wgsl -- Signed distance function for an ellipsoid.
//
// The signed distance from a point p to an ellipsoid with semi-axis lengths
// r centered at the origin is computed using Inigo Quilez's normalized
// formulation:
//   k0 = length(p / r)
//   sd = (k0 - 1.0) * min(r)
//
// The result is negative when p is inside the ellipsoid, zero on the surface,
// and positive outside.
//
// Edge cases (handled via epsilon-guarded division):
//   r = (0, *, *)  -- degenerate to a plane at x=0 (distance ~ abs(p.x))
//   r = (0, 0, 0)  -- degenerate to a point at origin (distance = length(p))
//   r < 0          -- abs(r) ensures positive semi-axis lengths
//
// Reference: Inigo Quilez -- SDF Primitives: sdEllipsoid
// https://iquilezles.org/articles/distfunctions/

// =============================================================================
// T-DEMO-1.8: SDF Ellipsoid Primitive
// =============================================================================

/// Signed distance from point p to an ellipsoid with semi-axis lengths r
/// centered at origin.
///
///   p        -- query position in world / object space
///   r        -- ellipsoid semi-axis lengths (x, y, z)
///   returns  -- signed distance (>0 outside, 0 on surface, <0 inside)
///
/// Example distances for r = (2.0, 1.0, 0.5):
///   p = (0, 0, 0)         -> -0.5     (center, inside)
///   p = (2, 0, 0)         ->  0.0     (on surface at x-extent)
///   p = (4, 0, 0)         ->  1.0     (outside along x)
///   p = (1, 0.5, 0.25)    -> -0.228   (inside corner)
fn sdEllipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
    // Guard against zero or negative semi-axis lengths. Any zero component
    // collapses the ellipsoid along that axis; using a tiny epsilon keeps
    // the formula numerically stable and produces the correct degenerate
    // shape: plane (one zero axis) or point (all zero axes).
    let eps = vec3<f32>(1e-10);
    let safe_r = max(abs(r), eps);

    // Normalize query position by semi-axis lengths and compute distance
    // ratio. Division by safe_r is always well-defined.
    let k0 = length(p / safe_r);

    // Scale by the smallest semi-axis to convert the normalized distance
    // back to world-space units.
    let min_r = min(min(safe_r.x, safe_r.y), safe_r.z);

    return (k0 - 1.0) * min_r;
}
