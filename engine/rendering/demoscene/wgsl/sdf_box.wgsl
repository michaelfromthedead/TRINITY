// SPDX-License-Identifier: MIT
//
// sdf_box.wgsl -- Signed distance to an axis-aligned box.
//
// The signed distance to a box centered at the origin with half-extents b
// is computed in two parts:
//   1. Outside: distance from p to the closest point on the box surface,
//      measured as length(max(q, 0)) where q = abs(p) - b.
//   2. Inside: the signed distance from p to the interior, measured as the
//      most-negative component of q, which is min(maxComponent(q), 0).
//
// The result is negative when p is inside the box, zero on the surface,
// and positive outside.
//
// Convention: b is the half-extent (half-width, half-height, half-depth).
// A zero-size dimension (b_i = 0) collapses that axis to a point, which
// produces the correct distance to the reduced box (e.g., b = (0,1,1)
// yields a rectangle in the yz-plane centered at the origin).
//
// Reference: Inigo Quilez -- SDF Primitives: sdBox
// https://iquilezles.org/articles/distfunctions/

// =============================================================================
// T-DEMO-1.2: SDF Box Primitive
// =============================================================================

/// Signed distance from point p to an axis-aligned box with half-extents b.
///
///   p        -- query position in world / object space
///   b        -- box half-extents (half-width, half-height, half-depth)
///   returns  -- signed distance (>0 outside, 0 on surface, <0 inside)
///
/// Example distances for b = vec3<f32>(1.0, 1.0, 1.0):
///   p = (2, 2, 2)  ->  sqrt(3)   (outside corner)
///   p = (1, 0, 0)  ->  0         (on face center)
///   p = (0.5, 0.5, 0.5) -> -0.5  (inside corner)
///   p = (0, 0, 0)  -> -1.0       (center)
fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
    // Signed distance components: negative inside, positive outside
    let q = abs(p) - b;

    // Outside: distance to the closest point on the box surface
    // Inside: returns 0 (all components of max(q, 0) are zero)
    let outside_dist = length(max(q, vec3<f32>(0.0, 0.0, 0.0)));

    // Inside: most-negative component of q gives the signed interior distance.
    // The min(..., 0.0) clamp ensures this term contributes nothing when outside.
    let inside_dist = min(max(q.x, max(q.y, q.z)), 0.0);

    return outside_dist + inside_dist;
}
