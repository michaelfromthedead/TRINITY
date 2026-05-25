// SPDX-License-Identifier: MIT
//
// sdf_cone.wgsl -- Signed distance function for a truncated cone (frustum).
//
// The cone is axis-aligned along the y-axis, extending from y=0 to y=h.
// r1 is the radius at the bottom cap (y=0), r2 is the radius at the top
// cap (y=h).  The side surface linearly interpolates between the two radii.
//
// Edge cases (handled naturally by the formula):
//   r1 = r2  -- collapses to a capped cylinder of radius r1 and height h.
//               The side becomes vertical; k2 reduces to (0, h) and the
//               dot-product maths reduces to the cylinder SDF.
//   r1 = 0   -- collapses to a pointed cone with apex at the origin.
//               The bottom-cap radius is zero so the vertex is sharp.
//   r2 = 0   -- collapses to an inverted pointed cone with apex at y=h.
//               The top-cap radius is zero so the vertex is sharp.
//   h = 0    -- special-cased: flat disc of radius max(r1, r2) at y=0.
//               Without the guard, dot(k2,k2) would be zero when r1 == r2,
//               causing division by zero in the side-distance projection.
//               When the radii differ and h=0, max(r1, r2) dominates.
//
// Reference: Inigo Quilez -- Signed distance functions
// https://iquilezles.org/articles/distfunctions/

// =============================================================================
// T-DEMO-1.5: Truncated Cone SDF
// =============================================================================

/// Signed distance to a truncated cone (frustum) along the y-axis.
///
///   p        -- query position
///   h        -- total height (cone extends from y=0 to y=h along y)
///   r1       -- radius at bottom cap (y=0)
///   r2       -- radius at top cap (y=h)
///   returns  -- signed distance: negative inside, zero on surface,
///               positive outside
fn sdCone(p: vec3<f32>, h: f32, r1: f32, r2: f32) -> f32 {
    // Degenerate case: zero height collapses to a flat disc.
    if (h < 1e-8) {
        return max(length(p.xz) - max(r1, r2), abs(p.y));
    }

    // Shift so the cone is centered at y=0 (IQ formulation uses half-height).
    let half_h = h * 0.5;
    let q = vec2<f32>(length(p.xz), p.y - half_h);

    // IQ reference: sdCappedCone
    //   k1  = (r2, half_h)      -- top-right edge of the trapezoid in 2D
    //   k2  = (r2 - r1, h)      -- direction from bottom-right to top-right
    //   ca  = distance to the nearest cap plane (including radial clamping)
    //   cb  = distance to the nearest point on the side line segment
    let k1 = vec2<f32>(r2, half_h);
    let k2 = vec2<f32>(r2 - r1, h);

    // cap-distance term: radial excess (ca.x) and vertical distance to the
    // nearest cap plane (ca.y).  select(r2, r1, q.y < 0) picks the bottom
    // radius when q is below the origin and the top radius otherwise.
    let ca = vec2<f32>(
        q.x - min(q.x, select(r2, r1, q.y < 0.0)),
        abs(q.y) - half_h
    );

    // side-distance term: closest-point projection onto the line segment
    // from the bottom-right to the top-right of the trapezoid.
    let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);

    // Sign: negative (inside) when radially inside the side surface (cb.x < 0)
    // AND vertically between the two cap planes (ca.y < 0).
    let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}
