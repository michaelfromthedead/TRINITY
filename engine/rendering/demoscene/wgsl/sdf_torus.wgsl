// SPDX-License-Identifier: MIT
//
// sdf_torus.wgsl -- Signed distance function for a torus.
//
// Evaluates the signed distance from a point p to a torus with major
// radius t.x (distance from the ring center to the tube center) and
// minor radius t.y (tube radius).
//
// Edge cases:
//   t.x = 0  -- degenerate to a sphere of radius t.y at the origin
//               (length(p) - t.y)
//   t.y = 0  -- degenerate to an infinitely thin ring of radius t.x
//               in the xz-plane
//
// Reference: Inigo Quilez -- Torus SDF
// https://iquilezles.org/articles/distfunctions/

// =============================================================================
// T-DEMO-1.3: Torus SDF
// =============================================================================

/// Signed distance from point p to a torus.
///
///   p        -- query point
///   t        -- torus radii: t.x = major (ring), t.y = minor (tube)
///   returns  -- signed distance: negative inside the tube, zero on surface,
///               positive outside
fn sdTorus(p: vec3<f32>, t: vec2<f32>) -> f32 {
    // IQ reference: length(vec2(length(p.xz)-t.x, p.y))-t.y
    // Guard against negative radii: take abs to ensure symmetry
    let safe_t = abs(t);
    let q = vec2<f32>(length(p.xz) - safe_t.x, p.y);
    return length(q) - safe_t.y;
}
