// SPDX-License-Identifier: MIT
//
// sdf_capsule.wgsl -- Signed distance function for a capsule (line segment
// with spherical end-caps).
//
// A capsule is defined by two endpoints A and B (the medial axis) and a
// radius r.  The surface is the set of points at distance exactly r from the
// line segment AB; the interior is the set of points at distance < r from AB.
// The shape is equivalent to a finite-length cylinder of radius r capped by
// two hemispheres of radius r centered at A and B.
//
// Edge cases (handled naturally by the formula):
//   r = 0      -- collapses to the line segment AB (zero-thickness wire).
//                  The result is the perpendicular distance to the segment.
//   A == B     -- collapses to a sphere of radius r centered at A.
//                  The projection h is degenerate (ba=0, baba=1e-10 guard),
//                  h clamps to 0.0, and the formula reduces to
//                  length(p - A) - r, the sphere SDF.
//   r < 0      -- treated with abs(r) to ensure outward-facing normals.
//                  The abs(r) guard prevents negative radius from inverting
//                  the interior/exterior sense.
//
// Reference: Inigo Quilez -- Signed distance functions
// https://iquilezles.org/articles/distfunctions/

// =============================================================================
// T-DEMO-1.7: Capsule SDF
// =============================================================================

/// Signed distance to a capsule defined by endpoints A, B and radius r.
///
///   p        -- query position
///   a        -- capsule endpoint A (one hemispherical cap center)
///   b        -- capsule endpoint B (other hemispherical cap center)
///   r        -- capsule radius (>= 0; abs(r) used internally)
///   returns  -- signed distance: negative inside, zero on surface,
///               positive outside
fn sdCapsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let baba = max(dot(ba, ba), 1e-10);
    let h = clamp(dot(pa, ba) / baba, 0.0, 1.0);
    return length(pa - ba * h) - abs(r);
}
