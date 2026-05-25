// SPDX-License-Identifier: MIT
//
// sdf_rounded_box.wgsl -- Signed distance function for a rounded box.
//
// The signed distance from a point p to a box of half-dimensions b with
// uniform corner radius r centered at the origin is computed using the
// Inigo Quilez reference formula:
//   q = abs(p) - b + r
//   sd = length(max(q, 0)) + min(max(q.x, q.y, q.z), 0) - r
//
// The corner radius r is clamped to [0, min(b)] so the box never vanishes.
// The result is negative when p is inside, zero on the surface, and positive
// outside.
//
// Edge cases:
//   r = 0           -- degenerate to a sharp-edged box (sdBox)
//   r >= min(b)     -- clamped to min(b); the shape approaches a sphere of
//                      radius (min(b) + maxGap)/2 where maxGap is the
//                      difference between largest and smallest half-dim
//
// Reference: Inigo Quilez -- SDF Primitives: sdRoundBox
// https://iquilezles.org/articles/distfunctions/

// =============================================================================
// T-DEMO-1.10: Rounded Box SDF Primitive
// =============================================================================

/// Signed distance from point p to a box of half-dimensions b with uniform
/// corner radius r, centered at origin.
///
///   p        -- query position in world / object space
///   b        -- box half-dimensions (width/2, height/2, depth/2)
///   r        -- corner radius (>= 0, clamped to [0, min(b)])
///   returns  -- signed distance (>0 outside, 0 on surface, <0 inside)
///
/// Example distances for b = vec3(2, 1, 1), r = 0.5:
///   p = (0, 0, 0)      -> -1.0    (center, inside)
///   p = (2, 0, 0)      ->  0.0    (on x-face surface)
///   p = (2.5, 0, 0)    ->  0.5    (outside)
///   p = (3, 0, 0)      ->  1.0    (outside)
///   p = (2, 1, 1)      ->  0.366  (outside, corner cut off)
fn sdRoundedBox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    // Clamp corner radius to valid range: [0, min(b)]
    let safe_r = min(max(r, 0.0), min(b.x, min(b.y, b.z)));

    // Shift query point by safe_r so the rounded corner surface coincides
    // with the original box face at distance b_i from origin.
    let q = abs(p) - b + safe_r;

    // Distance to the box exterior (handles points outside the box)
    let exterior = length(max(q, vec3(0.0)));

    // Distance to the box interior (handles points inside the box)
    let interior = min(max(q.x, max(q.y, q.z)), 0.0);

    return exterior + interior - safe_r;
}
