// SPDX-License-Identifier: MIT
//
// sdf_sphere.wgsl -- Signed distance function for a sphere.
//
// The signed distance from a point p to a sphere of radius r centered at
// the origin is computed as:
//   sd = length(p) - r
//
// The result is negative when p is inside the sphere, zero on the surface,
// and positive outside.
//
// Edge cases:
//   r < 0  -- abs(r) ensures positive radius
//   r = 0  -- degenerate to a point at the origin (sd = length(p))
//
// Reference: Inigo Quilez -- SDF Primitives: sdSphere
// https://iquilezles.org/articles/distfunctions/

// =============================================================================
// T-DEMO-1.1: SDF Sphere Primitive
// =============================================================================

/// Signed distance from point p to a sphere of radius r centered at origin.
///
///   p        -- query position in world / object space
///   r        -- sphere radius
///   returns  -- signed distance (>0 outside, 0 on surface, <0 inside)
///
/// Example distances for r = 2.0:
///   p = (0, 0, 0)     -> -2.0   (center, inside)
///   p = (2, 0, 0)     ->  0.0   (on surface)
///   p = (5, 0, 0)     ->  3.0   (outside)
///   p = (1, 1, 1)     -> -0.268  (inside corner, length = sqrt(3))
fn sdSphere(p: vec3<f32>, r: f32) -> f32 {
    // Guard against negative radius: use abs to ensure non-negative radius
    let safe_r = abs(r);
    return length(p) - safe_r;
}
