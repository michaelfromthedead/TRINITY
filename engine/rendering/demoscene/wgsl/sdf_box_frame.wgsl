// SPDX-License-Identifier: MIT
//
// sdf_box_frame.wgsl -- Signed distance function for a hollow box frame.
//
// The signed distance from a point p to a box frame of dimensions b and
// frame thickness e is computed as:
//   let q = abs(p) - b;
//   sd = length(max(q, vec3(0))) + min(max(q.x, max(q.y, q.z)), 0) - e;
//
// The result is negative inside the frame walls, zero on the surface, and
// positive outside. The interior cavity (hollow region) has positive SDF
// values, creating the frame effect.
//
// Edge cases:
//   e = 0       -- degenerate to solid box (identical to sdBox)
//   e >= min(b) -- frame thickness fills the interior; no hollow region
//
// Reference: Inigo Quilez -- SDF Primitives: sdBoxFrame
// https://iquilezles.org/articles/distfunctions/

// =============================================================================
// T-DEMO-1.9: SDF Box Frame Primitive
// =============================================================================

/// Signed distance from point p to a hollow box frame of dimensions b
/// with frame wall thickness e, centered at the origin.
///
///   p        -- query position in world / object space
///   b        -- half-dimensions of the outer box (x, y, z)
///   e        -- frame wall thickness
///   returns  -- signed distance (>0 outside, 0 on surface, <0 inside walls)
///
/// The interior cavity has positive distance; only the frame walls themselves
/// produce negative (inside) values.
///
/// Example distances for b = vec3(2, 2, 2), e = 0.5:
///   p = (0, 0, 0)     ->  1.5    (center of cavity, outside walls)
///   p = (1.5, 0, 0)   -> -0.5    (inside frame wall)
///   p = (2.0, 0, 0)   ->  0.0    (on outer surface)
///   p = (2.5, 0, 0)   ->  0.5    (outside)
///   p = (1.75, 0, 0)  -> -0.25   (mid-wall)
fn sdBoxFrame(p: vec3<f32>, b: vec3<f32>, e: f32) -> f32 {
    let q = abs(p) - b;
    let d = length(max(q, vec3<f32>(0.0)))
        + min(max(q.x, max(q.y, q.z)), 0.0)
        - e;
    return d;
}
