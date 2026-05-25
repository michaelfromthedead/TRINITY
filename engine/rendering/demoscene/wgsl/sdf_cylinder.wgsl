// SPDX-License-Identifier: MIT
//
// sdf_cylinder.wgsl -- Signed distance function for a capped cylinder.
//
// The cylinder is axis-aligned along the y-axis with its center at the
// origin, extending from -h to +h. Both end-caps (discs) are included
// in the SDF.
//
// Edge cases (handled naturally by the formula -- no special-case guards):
//   r = 0  -- collapses to a line segment of length 2*h along the y-axis.
//             The radial term length(p.xz) - 0 reduces to the perpendicular
//             distance from the y-axis, giving the correct distance to a
//             finite line segment.
//   h = 0  -- collapses to a flat disc of radius r lying on the xz-plane.
//             The vertical term abs(p.y) - 0 gives the absolute y-distance,
//             yielding the correct distance to a disc at the origin.
//
// Reference: Inigo Quilez -- Signed distance functions
// https://iquilezles.org/articles/distfunctions/

// =============================================================================
// T-DEMO-1.4: Capped Cylinder SDF
// =============================================================================

/// Signed distance to a capped cylinder centered at the origin.
///
///   p        -- query position
///   h        -- half-height (cylinder extends from -h to +h along y)
///   r        -- radius in the xz-plane
///   returns  -- signed distance: negative inside, zero on surface, positive outside
fn sdCylinder(p: vec3<f32>, h: f32, r: f32) -> f32 {
    // Distance from the cylinder surface in the radial and axial directions.
    // d.x = radial excess (positive = outside the cylinder wall)
    // d.y = axial excess (positive = above top cap or below bottom cap)
    let d = abs(vec2<f32>(length(p.xz), p.y)) - vec2<f32>(r, h);

    // Inside the cylinder (both d.x <= 0 and d.y <= 0): return the larger
    // (less negative) component -- this is the distance to the nearest surface.
    // Outside (at least one component positive): return the Euclidean distance
    // to the cylinder edge, computed as length(max(d, 0)).
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}
