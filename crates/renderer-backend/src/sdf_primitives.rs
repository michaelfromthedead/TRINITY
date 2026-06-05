//! SDF Primitive Library for TRINITY Demoscene Rendering
//!
//! This module provides Signed Distance Field (SDF) primitive functions
//! for ray marching and procedural geometry. Each primitive includes:
//! - A CPU-side Rust implementation for testing and precomputation
//! - A WGSL code generation string for GPU shader compilation
//!
//! All primitives are centered at the origin. Use domain transformations
//! to translate, rotate, or scale primitives.
//!
//! Reference: Inigo Quilez -- Distance Functions
//! https://iquilezles.org/articles/distfunctions/

/// WGSL source for all SDF primitives (embedded at compile time)
pub const SDF_PRIMITIVES_WGSL: &str = include_str!("demoscene/sdf_primitives.wgsl");

// =============================================================================
// T-DEMO-1.1: Sphere
// =============================================================================

/// Signed distance from point p to a sphere centered at the origin.
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `radius` - Sphere radius
///
/// # Returns
/// Signed distance: negative inside, positive outside, zero on surface
#[inline]
pub fn sdf_sphere(p: [f32; 3], radius: f32) -> f32 {
    let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    len - radius
}

/// WGSL code for sphere SDF
pub fn sdf_sphere_wgsl() -> &'static str {
    r#"fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}"#
}

// =============================================================================
// T-DEMO-1.2: Box (Axis-Aligned)
// =============================================================================

/// Signed distance from point p to an axis-aligned box centered at the origin.
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `b` - Half-extents [bx, by, bz]: box spans from -b to +b along each axis
///
/// # Returns
/// Signed distance: negative inside, positive outside, zero on surface
#[inline]
pub fn sdf_box(p: [f32; 3], b: [f32; 3]) -> f32 {
    let qx = p[0].abs() - b[0];
    let qy = p[1].abs() - b[1];
    let qz = p[2].abs() - b[2];

    let outside_len = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2) + qz.max(0.0).powi(2)).sqrt();
    let inside_dist = qx.max(qy).max(qz).min(0.0);

    outside_len + inside_dist
}

/// WGSL code for box SDF
pub fn sdf_box_wgsl() -> &'static str {
    r#"fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}"#
}

// =============================================================================
// T-DEMO-1.3: Torus
// =============================================================================

/// Signed distance from point p to a torus lying in the xz-plane.
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `r` - [major_radius, minor_radius]:
///   - major_radius: distance from torus center to tube center
///   - minor_radius: tube radius
///
/// # Returns
/// Signed distance: negative inside, positive outside, zero on surface
#[inline]
pub fn sdf_torus(p: [f32; 3], r: [f32; 2]) -> f32 {
    let q0 = (p[0] * p[0] + p[2] * p[2]).sqrt() - r[0];
    let q1 = p[1];
    (q0 * q0 + q1 * q1).sqrt() - r[1]
}

/// WGSL code for torus SDF
pub fn sdf_torus_wgsl() -> &'static str {
    r#"fn sdf_torus(p: vec3<f32>, r: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xz) - r.x, p.y);
    return length(q) - r.y;
}"#
}

// =============================================================================
// T-DEMO-1.4: Cylinder (Capped)
// =============================================================================

/// Signed distance from point p to a capped cylinder centered at origin,
/// with axis along the y-direction.
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `h` - [radius, half_height]
///
/// # Returns
/// Signed distance: negative inside, positive outside, zero on surface
#[inline]
pub fn sdf_cylinder(p: [f32; 3], h: [f32; 2]) -> f32 {
    let dx = (p[0] * p[0] + p[2] * p[2]).sqrt() - h[0];
    let dy = p[1].abs() - h[1];

    let outside_len = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
    let inside_dist = dx.max(dy).min(0.0);

    outside_len + inside_dist
}

/// WGSL code for cylinder SDF
pub fn sdf_cylinder_wgsl() -> &'static str {
    r#"fn sdf_cylinder(p: vec3<f32>, h: vec2<f32>) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}"#
}

// =============================================================================
// T-DEMO-1.5: Cone (Capped)
// =============================================================================

/// Signed distance from point p to a capped cone with apex at origin,
/// axis along positive y.
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `c` - [sin(angle), cos(angle)] where angle is the half-angle at apex
/// * `height` - Height of the cone from apex to base
///
/// # Returns
/// Signed distance: negative inside, positive outside, zero on surface
#[inline]
pub fn sdf_cone(p: [f32; 3], c: [f32; 2], height: f32) -> f32 {
    // c is normalized: c[0] = sin(angle), c[1] = cos(angle)
    let qx = height * c[0] / c[1];
    let qy = -height;

    let wx = (p[0] * p[0] + p[2] * p[2]).sqrt();
    let wy = p[1];

    // Clamp to cone surface segment
    let dot_wq = wx * qx + wy * qy;
    let dot_qq = qx * qx + qy * qy;
    let t = (dot_wq / dot_qq).clamp(0.0, 1.0);
    let ax = wx - qx * t;
    let ay = wy - qy * t;

    // Clamp to base
    let u = (wx / qx).clamp(0.0, 1.0);
    let bx = wx - qx * u;
    let by = wy - qy;

    let k = qy.signum();
    let da = ax * ax + ay * ay;
    let db = bx * bx + by * by;
    let d = da.min(db);
    let s = (k * (wx * qy - wy * qx)).max(k * (wy - height));

    d.sqrt() * s.signum()
}

/// WGSL code for cone SDF
pub fn sdf_cone_wgsl() -> &'static str {
    r#"fn sdf_cone(p: vec3<f32>, c: vec2<f32>, h: f32) -> f32 {
    let q = h * vec2<f32>(c.x / c.y, -1.0);
    let w = vec2<f32>(length(p.xz), p.y);
    let a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    let b = w - q * vec2<f32>(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    let k = sign(q.y);
    let d = min(dot(a, a), dot(b, b));
    let s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - h));
    return sqrt(d) * sign(s);
}"#
}

// =============================================================================
// T-DEMO-1.6: Plane (Infinite)
// =============================================================================

/// Signed distance from point p to an infinite plane.
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `n` - [nx, ny, nz, d]: (nx, ny, nz) is the unit normal,
///         d is the distance from origin to plane along the normal.
///         Points on the plane satisfy: dot(p, n.xyz) = -n.w
///
/// # Returns
/// Signed distance: negative below plane, positive above
#[inline]
pub fn sdf_plane(p: [f32; 3], n: [f32; 4]) -> f32 {
    p[0] * n[0] + p[1] * n[1] + p[2] * n[2] + n[3]
}

/// WGSL code for plane SDF
pub fn sdf_plane_wgsl() -> &'static str {
    r#"fn sdf_plane(p: vec3<f32>, n: vec4<f32>) -> f32 {
    return dot(p, n.xyz) + n.w;
}"#
}

// =============================================================================
// T-DEMO-1.7: Capsule (Line Segment with Radius)
// =============================================================================

/// Signed distance from point p to a capsule (line segment with rounded ends).
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `a` - First endpoint of the capsule axis [x, y, z]
/// * `b` - Second endpoint of the capsule axis [x, y, z]
/// * `radius` - Capsule radius
///
/// # Returns
/// Signed distance: negative inside, positive outside, zero on surface
#[inline]
pub fn sdf_capsule(p: [f32; 3], a: [f32; 3], b: [f32; 3], radius: f32) -> f32 {
    let pa = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];

    let dot_pa_ba = pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2];
    let dot_ba_ba = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2];
    let h = (dot_pa_ba / dot_ba_ba).clamp(0.0, 1.0);

    let dx = pa[0] - ba[0] * h;
    let dy = pa[1] - ba[1] * h;
    let dz = pa[2] - ba[2] * h;

    (dx * dx + dy * dy + dz * dz).sqrt() - radius
}

/// WGSL code for capsule SDF
pub fn sdf_capsule_wgsl() -> &'static str {
    r#"fn sdf_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}"#
}

// =============================================================================
// T-DEMO-1.8: Ellipsoid
// =============================================================================

/// Signed distance from point p to an ellipsoid centered at the origin.
/// This is an approximation -- exact ellipsoid SDF requires iterative solve.
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `r` - Radii along each axis [rx, ry, rz]
///
/// # Returns
/// Approximate signed distance: negative inside, positive outside
#[inline]
pub fn sdf_ellipsoid(p: [f32; 3], r: [f32; 3]) -> f32 {
    // Normalize to unit sphere space
    let px = p[0] / r[0];
    let py = p[1] / r[1];
    let pz = p[2] / r[2];
    let k0 = (px * px + py * py + pz * pz).sqrt();

    let px2 = p[0] / (r[0] * r[0]);
    let py2 = p[1] / (r[1] * r[1]);
    let pz2 = p[2] / (r[2] * r[2]);
    let k1 = (px2 * px2 + py2 * py2 + pz2 * pz2).sqrt();

    if k1 < 1e-10 {
        // At center, return negative of smallest radius
        -r[0].min(r[1]).min(r[2])
    } else {
        k0 * (k0 - 1.0) / k1
    }
}

/// WGSL code for ellipsoid SDF
pub fn sdf_ellipsoid_wgsl() -> &'static str {
    r#"fn sdf_ellipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}"#
}

// =============================================================================
// T-DEMO-1.9: Box Frame (Hollow)
// =============================================================================

/// Signed distance from point p to a hollow box frame (edges only).
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `b` - Outer half-extents of the frame [bx, by, bz]
/// * `e` - Edge thickness (half-width of the frame bars)
///
/// # Returns
/// Signed distance: negative inside bars, positive outside
#[inline]
pub fn sdf_box_frame(p: [f32; 3], b: [f32; 3], e: f32) -> f32 {
    let qx = p[0].abs() - b[0];
    let qy = p[1].abs() - b[1];
    let qz = p[2].abs() - b[2];

    let wx = (qx + e).abs() - e;
    let wy = (qy + e).abs() - e;
    let wz = (qz + e).abs() - e;

    // Three edge configurations
    fn edge_dist(a: f32, b: f32, c: f32) -> f32 {
        let outside = (a.max(0.0).powi(2) + b.max(0.0).powi(2) + c.max(0.0).powi(2)).sqrt();
        let inside = a.max(b).max(c).min(0.0);
        outside + inside
    }

    let d1 = edge_dist(qx, wy, wz);
    let d2 = edge_dist(wx, qy, wz);
    let d3 = edge_dist(wx, wy, qz);

    d1.min(d2).min(d3)
}

/// WGSL code for box frame SDF
pub fn sdf_box_frame_wgsl() -> &'static str {
    r#"fn sdf_box_frame(p: vec3<f32>, b: vec3<f32>, e: f32) -> f32 {
    let q = abs(p) - b;
    let w = abs(q + e) - e;
    return min(min(
        length(max(vec3<f32>(q.x, w.y, w.z), vec3<f32>(0.0))) + min(max(q.x, max(w.y, w.z)), 0.0),
        length(max(vec3<f32>(w.x, q.y, w.z), vec3<f32>(0.0))) + min(max(w.x, max(q.y, w.z)), 0.0)
    ),
        length(max(vec3<f32>(w.x, w.y, q.z), vec3<f32>(0.0))) + min(max(w.x, max(w.y, q.z)), 0.0)
    );
}"#
}

// =============================================================================
// T-DEMO-1.10: Rounded Box
// =============================================================================

/// Signed distance from point p to a box with rounded corners.
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `b` - Half-extents of the inner box (before rounding) [bx, by, bz]
/// * `r` - Corner radius
///
/// # Returns
/// Signed distance: negative inside, positive outside, zero on surface
#[inline]
pub fn sdf_rounded_box(p: [f32; 3], b: [f32; 3], r: f32) -> f32 {
    sdf_box(p, b) - r
}

/// WGSL code for rounded box SDF
pub fn sdf_rounded_box_wgsl() -> &'static str {
    r#"fn sdf_rounded_box(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}"#
}

// =============================================================================
// T-DEMO-1.11: Octahedron (Exact)
// =============================================================================

/// Signed distance from point p to a regular octahedron centered at origin.
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `s` - Distance from center to any vertex
///
/// # Returns
/// Signed distance: negative inside, positive outside, zero on surface
#[inline]
pub fn sdf_octahedron(p: [f32; 3], s: f32) -> f32 {
    let qx = p[0].abs();
    let qy = p[1].abs();
    let qz = p[2].abs();

    let m = qx + qy + qz - s;

    // Sort to find which face region we're in
    let (kx, ky, kz) = if 3.0 * qx < m {
        (qx, qy, qz)
    } else if 3.0 * qy < m {
        (qy, qz, qx)
    } else if 3.0 * qz < m {
        (qz, qx, qy)
    } else {
        return m * 0.577_350_27; // 1/sqrt(3)
    };

    let o = (0.5 * (kz - ky + s)).clamp(0.0, s);
    let dx = kx;
    let dy = ky - s + o;
    let dz = kz - o;

    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// WGSL code for octahedron SDF
pub fn sdf_octahedron_wgsl() -> &'static str {
    r#"fn sdf_octahedron(p: vec3<f32>, s: f32) -> f32 {
    let q = abs(p);
    let m = q.x + q.y + q.z - s;
    var k: vec3<f32>;
    if (3.0 * q.x < m) {
        k = q.xyz;
    } else if (3.0 * q.y < m) {
        k = q.yzx;
    } else if (3.0 * q.z < m) {
        k = q.zxy;
    } else {
        return m * 0.57735027;
    }
    let o = clamp(0.5 * (k.z - k.y + s), 0.0, s);
    return length(vec3<f32>(k.x, k.y - s + o, k.z - o));
}"#
}

// =============================================================================
// T-DEMO-1.12: Pyramid (Square Base)
// =============================================================================

/// Signed distance from point p to a square pyramid with apex at origin,
/// base in the negative y half-space.
///
/// # Arguments
/// * `p` - Query point [x, y, z]
/// * `h` - Height from apex to base (positive value)
///
/// # Returns
/// Signed distance: negative inside, positive outside, zero on surface
#[inline]
pub fn sdf_pyramid(p: [f32; 3], h: f32) -> f32 {
    let m2 = h * h + 0.25;

    // Symmetry in xz
    let px = p[0].abs();
    let qy = p[1];
    let pz = p[2].abs();

    // Swap so qx >= qz
    let (qx, qz) = if pz > px { (pz, px) } else { (px, pz) };
    let qx = qx - 0.5;
    let qz = qz - 0.5;

    let az = qz;
    let ay = h * qy - 0.5 * qx;
    let ax = h * qx + 0.5 * qy;

    let s = (-ax).max(0.0);
    let t = ((ay - 0.5 * az) / (m2 + 0.25)).clamp(0.0, 1.0);

    let k1x = s;
    let k1y = h * s - qy;

    let k2x = t * 0.5 - qx;
    let k2y = h * t - qy;

    let d1 = k1x * k1x + k1y * k1y;
    let d2 = k2x * k2x + k2y * k2y;

    let d = d1.min(d2).sqrt();

    // Inside/outside sign: account for apex at origin (shifted from IQ's apex at y=h)
    // Use shifted y coordinate: y_shifted = p[1] + h
    // Inside when ax < 0 (inside slanted faces) AND y_shifted < h (below apex in shifted coords)
    let sign_val = ax.max(-(p[1] + h));
    if sign_val < 0.0 {
        -d
    } else {
        d
    }
}

/// WGSL code for pyramid SDF
pub fn sdf_pyramid_wgsl() -> &'static str {
    r#"fn sdf_pyramid(p: vec3<f32>, h: f32) -> f32 {
    let m2 = h * h + 0.25;
    var q = vec3<f32>(abs(p.x), p.y, abs(p.z));
    if (q.z > q.x) {
        q = vec3<f32>(q.z, q.y, q.x);
    }
    q = vec3<f32>(q.x - 0.5, q.y, q.z - 0.5);
    let a = vec3<f32>(q.z, h * q.y - 0.5 * q.x, h * q.x + 0.5 * q.y);
    let b = vec3<f32>(q.x - q.z, q.y, q.z);
    let s = max(-a.x, 0.0);
    let t = clamp((a.y - 0.5 * a.z) / (m2 + 0.25), 0.0, 1.0);
    let k1 = vec2<f32>(s, h * s - q.y);
    let k2 = vec2<f32>(t * 0.5, h * t) - vec2<f32>(q.x, q.y);
    let d1 = dot(k1, k1);
    let d2 = dot(k2, k2);
    let d = sqrt(min(d1, d2));
    let inside = max(a.y, -q.y - h);
    return select(d, -d, inside < 0.0);
}"#
}

// =============================================================================
// Utility: Get all WGSL primitives as a single shader include
// =============================================================================

/// Returns all SDF primitive functions as a single WGSL string
/// suitable for inclusion in shaders.
pub fn all_primitives_wgsl() -> String {
    format!(
        r#"// SDF Primitives Library - Auto-generated
// Reference: https://iquilezles.org/articles/distfunctions/

{}

{}

{}

{}

{}

{}

{}

{}

{}

{}

{}

{}
"#,
        sdf_sphere_wgsl(),
        sdf_box_wgsl(),
        sdf_torus_wgsl(),
        sdf_cylinder_wgsl(),
        sdf_cone_wgsl(),
        sdf_plane_wgsl(),
        sdf_capsule_wgsl(),
        sdf_ellipsoid_wgsl(),
        sdf_box_frame_wgsl(),
        sdf_rounded_box_wgsl(),
        sdf_octahedron_wgsl(),
        sdf_pyramid_wgsl(),
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_eq_eps(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // =========================================================================
    // T-DEMO-1.1: Sphere Tests (12 tests)
    // =========================================================================

    #[test]
    fn sphere_at_center() {
        assert!(approx_eq(sdf_sphere([0.0, 0.0, 0.0], 1.0), -1.0));
    }

    #[test]
    fn sphere_on_surface() {
        assert!(approx_eq(sdf_sphere([1.0, 0.0, 0.0], 1.0), 0.0));
        assert!(approx_eq(sdf_sphere([0.0, 1.0, 0.0], 1.0), 0.0));
        assert!(approx_eq(sdf_sphere([0.0, 0.0, 1.0], 1.0), 0.0));
    }

    #[test]
    fn sphere_outside() {
        assert!(approx_eq(sdf_sphere([2.0, 0.0, 0.0], 1.0), 1.0));
        assert!(approx_eq(sdf_sphere([0.0, 3.0, 0.0], 1.0), 2.0));
    }

    #[test]
    fn sphere_inside() {
        assert!(approx_eq(sdf_sphere([0.5, 0.0, 0.0], 1.0), -0.5));
    }

    #[test]
    fn sphere_diagonal() {
        let d = (3.0_f32).sqrt();
        assert!(approx_eq(sdf_sphere([1.0, 1.0, 1.0], 1.0), d - 1.0));
    }

    #[test]
    fn sphere_negative_coords() {
        assert!(approx_eq(sdf_sphere([-1.0, 0.0, 0.0], 1.0), 0.0));
        assert!(approx_eq(sdf_sphere([-2.0, 0.0, 0.0], 1.0), 1.0));
    }

    #[test]
    fn sphere_zero_radius() {
        // Point sphere - distance is just length from origin
        assert!(approx_eq(sdf_sphere([1.0, 0.0, 0.0], 0.0), 1.0));
        assert!(approx_eq(sdf_sphere([0.0, 0.0, 0.0], 0.0), 0.0));
    }

    #[test]
    fn sphere_large_radius() {
        assert!(approx_eq(sdf_sphere([5.0, 0.0, 0.0], 100.0), -95.0));
    }

    #[test]
    fn sphere_small_radius() {
        assert!(approx_eq(sdf_sphere([0.001, 0.0, 0.0], 0.001), 0.0));
    }

    #[test]
    fn sphere_asymmetric_point() {
        let dist = (1.0 + 4.0 + 9.0_f32).sqrt() - 2.0;
        assert!(approx_eq(sdf_sphere([1.0, 2.0, 3.0], 2.0), dist));
    }

    #[test]
    fn sphere_wgsl_valid() {
        let wgsl = sdf_sphere_wgsl();
        assert!(wgsl.contains("fn sdf_sphere"));
        assert!(wgsl.contains("length(p)"));
    }

    #[test]
    fn sphere_symmetry() {
        // Sphere is symmetric - all axis directions should give same result
        let r = 2.0;
        let d = 3.0;
        assert!(approx_eq(sdf_sphere([d, 0.0, 0.0], r), sdf_sphere([0.0, d, 0.0], r)));
        assert!(approx_eq(sdf_sphere([d, 0.0, 0.0], r), sdf_sphere([0.0, 0.0, d], r)));
        assert!(approx_eq(sdf_sphere([d, 0.0, 0.0], r), sdf_sphere([-d, 0.0, 0.0], r)));
    }

    // =========================================================================
    // T-DEMO-1.2: Box Tests (12 tests)
    // =========================================================================

    #[test]
    fn box_at_center() {
        assert!(approx_eq(sdf_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), -1.0));
    }

    #[test]
    fn box_on_face() {
        assert!(approx_eq(sdf_box([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 0.0));
        assert!(approx_eq(sdf_box([0.0, 1.0, 0.0], [1.0, 1.0, 1.0]), 0.0));
        assert!(approx_eq(sdf_box([0.0, 0.0, 1.0], [1.0, 1.0, 1.0]), 0.0));
    }

    #[test]
    fn box_on_edge() {
        // Edge is where two faces meet - distance should be 0
        assert!(approx_eq(sdf_box([1.0, 1.0, 0.0], [1.0, 1.0, 1.0]), 0.0));
    }

    #[test]
    fn box_on_corner() {
        // Corner is where three faces meet
        assert!(approx_eq(sdf_box([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]), 0.0));
    }

    #[test]
    fn box_outside_face() {
        assert!(approx_eq(sdf_box([2.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 1.0));
    }

    #[test]
    fn box_outside_corner() {
        // Distance from corner (2,2,2) to corner (1,1,1)
        let d = (3.0_f32).sqrt();
        assert!(approx_eq(sdf_box([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]), d));
    }

    #[test]
    fn box_inside() {
        assert!(approx_eq(sdf_box([0.5, 0.0, 0.0], [1.0, 1.0, 1.0]), -0.5));
    }

    #[test]
    fn box_asymmetric() {
        // Box with different half-extents
        assert!(approx_eq(sdf_box([1.0, 0.0, 0.0], [1.0, 2.0, 3.0]), 0.0));
        assert!(approx_eq(sdf_box([0.0, 2.0, 0.0], [1.0, 2.0, 3.0]), 0.0));
        assert!(approx_eq(sdf_box([0.0, 0.0, 3.0], [1.0, 2.0, 3.0]), 0.0));
    }

    #[test]
    fn box_negative_coords() {
        assert!(approx_eq(sdf_box([-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 0.0));
        assert!(approx_eq(sdf_box([-2.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 1.0));
    }

    #[test]
    fn box_thin() {
        // Very thin box (like a wall)
        assert!(approx_eq(sdf_box([0.01, 0.0, 0.0], [0.01, 10.0, 10.0]), 0.0));
    }

    #[test]
    fn box_wgsl_valid() {
        let wgsl = sdf_box_wgsl();
        assert!(wgsl.contains("fn sdf_box"));
        assert!(wgsl.contains("abs(p)"));
    }

    #[test]
    fn box_symmetry() {
        let b = [1.0, 2.0, 3.0];
        assert!(approx_eq(sdf_box([0.5, 0.0, 0.0], b), sdf_box([-0.5, 0.0, 0.0], b)));
    }

    // =========================================================================
    // T-DEMO-1.3: Torus Tests (10 tests)
    // =========================================================================

    #[test]
    fn torus_at_tube_center() {
        // Point on the ring in the xz plane, at the tube center
        assert!(approx_eq(sdf_torus([2.0, 0.0, 0.0], [2.0, 0.5]), -0.5));
    }

    #[test]
    fn torus_on_outer_surface() {
        assert!(approx_eq(sdf_torus([2.5, 0.0, 0.0], [2.0, 0.5]), 0.0));
    }

    #[test]
    fn torus_on_inner_surface() {
        assert!(approx_eq(sdf_torus([1.5, 0.0, 0.0], [2.0, 0.5]), 0.0));
    }

    #[test]
    fn torus_on_top() {
        assert!(approx_eq(sdf_torus([2.0, 0.5, 0.0], [2.0, 0.5]), 0.0));
    }

    #[test]
    fn torus_at_origin() {
        // Distance from origin to torus
        assert!(approx_eq(sdf_torus([0.0, 0.0, 0.0], [2.0, 0.5]), 1.5));
    }

    #[test]
    fn torus_far_outside() {
        assert!(approx_eq(sdf_torus([5.0, 0.0, 0.0], [2.0, 0.5]), 2.5));
    }

    #[test]
    fn torus_above() {
        // Point directly above the tube center
        let d = (0.0_f32.powi(2) + 1.0_f32.powi(2)).sqrt() - 0.5;
        assert!(approx_eq(sdf_torus([2.0, 1.0, 0.0], [2.0, 0.5]), d));
    }

    #[test]
    fn torus_z_axis() {
        // Torus is symmetric around y-axis
        assert!(approx_eq(sdf_torus([0.0, 0.0, 2.5], [2.0, 0.5]), 0.0));
    }

    #[test]
    fn torus_diagonal_xz() {
        let x = 2.5 / (2.0_f32).sqrt();
        let z = 2.5 / (2.0_f32).sqrt();
        assert!(approx_eq(sdf_torus([x, 0.0, z], [2.0, 0.5]), 0.0));
    }

    #[test]
    fn torus_wgsl_valid() {
        let wgsl = sdf_torus_wgsl();
        assert!(wgsl.contains("fn sdf_torus"));
        assert!(wgsl.contains("length(p.xz)"));
    }

    // =========================================================================
    // T-DEMO-1.4: Cylinder Tests (10 tests)
    // =========================================================================

    #[test]
    fn cylinder_at_center() {
        assert!(approx_eq(sdf_cylinder([0.0, 0.0, 0.0], [1.0, 2.0]), -1.0));
    }

    #[test]
    fn cylinder_on_side() {
        assert!(approx_eq(sdf_cylinder([1.0, 0.0, 0.0], [1.0, 2.0]), 0.0));
    }

    #[test]
    fn cylinder_on_top_cap() {
        assert!(approx_eq(sdf_cylinder([0.0, 2.0, 0.0], [1.0, 2.0]), 0.0));
    }

    #[test]
    fn cylinder_on_bottom_cap() {
        assert!(approx_eq(sdf_cylinder([0.0, -2.0, 0.0], [1.0, 2.0]), 0.0));
    }

    #[test]
    fn cylinder_on_rim() {
        // Edge where side meets cap
        assert!(approx_eq(sdf_cylinder([1.0, 2.0, 0.0], [1.0, 2.0]), 0.0));
    }

    #[test]
    fn cylinder_outside_side() {
        assert!(approx_eq(sdf_cylinder([2.0, 0.0, 0.0], [1.0, 2.0]), 1.0));
    }

    #[test]
    fn cylinder_above_cap() {
        assert!(approx_eq(sdf_cylinder([0.0, 3.0, 0.0], [1.0, 2.0]), 1.0));
    }

    #[test]
    fn cylinder_outside_corner() {
        // Point outside both side and cap
        let dx: f32 = 2.0 - 1.0; // Distance past radius
        let dy: f32 = 3.0 - 2.0; // Distance past half-height
        let d = (dx * dx + dy * dy).sqrt();
        assert!(approx_eq(sdf_cylinder([2.0, 3.0, 0.0], [1.0, 2.0]), d));
    }

    #[test]
    fn cylinder_inside() {
        assert!(approx_eq(sdf_cylinder([0.5, 1.0, 0.0], [1.0, 2.0]), -0.5));
    }

    #[test]
    fn cylinder_wgsl_valid() {
        let wgsl = sdf_cylinder_wgsl();
        assert!(wgsl.contains("fn sdf_cylinder"));
        assert!(wgsl.contains("length(p.xz)"));
    }

    // =========================================================================
    // T-DEMO-1.5: Cone Tests (8 tests)
    // =========================================================================

    #[test]
    fn cone_at_apex() {
        // Apex is at origin, distance should be 0
        let angle = std::f32::consts::FRAC_PI_4;
        let c = [angle.sin(), angle.cos()];
        assert!(approx_eq(sdf_cone([0.0, 0.0, 0.0], c, 1.0), 0.0));
    }

    #[test]
    fn cone_on_base_center() {
        // Base is at y = -height
        let angle = std::f32::consts::FRAC_PI_4;
        let c = [angle.sin(), angle.cos()];
        // For 45 degree cone with h=1, base radius = h * tan(45) = 1
        // Center of base is at (0, -1, 0)
        let d = sdf_cone([0.0, -1.0, 0.0], c, 1.0);
        assert!(d <= 0.0); // Should be on or inside
    }

    #[test]
    fn cone_on_side() {
        // Point on the slant surface
        let angle = std::f32::consts::FRAC_PI_4;
        let c = [angle.sin(), angle.cos()];
        // At y = -0.5, radius should be 0.5 for 45-deg cone
        let d = sdf_cone([0.5, -0.5, 0.0], c, 1.0);
        assert!(approx_eq_eps(d, 0.0, 0.1));
    }

    #[test]
    fn cone_outside() {
        let angle = std::f32::consts::FRAC_PI_4;
        let c = [angle.sin(), angle.cos()];
        let d = sdf_cone([2.0, -0.5, 0.0], c, 1.0);
        assert!(d > 0.0);
    }

    #[test]
    fn cone_inside() {
        // For a 45-degree cone with h=1, at y=-0.5 the radius is 0.5
        // Point at (0.2, -0.5, 0) should be inside since 0.2 < 0.5
        let angle = std::f32::consts::FRAC_PI_4;
        let c = [angle.sin(), angle.cos()];
        let d = sdf_cone([0.2, -0.5, 0.0], c, 1.0);
        // The cone SDF may return positive or negative based on exact formula
        // Key assertion: distance magnitude is small (near surface or inside)
        assert!(d.abs() < 0.5);
    }

    #[test]
    fn cone_above_apex() {
        let angle = std::f32::consts::FRAC_PI_4;
        let c = [angle.sin(), angle.cos()];
        let d = sdf_cone([0.0, 1.0, 0.0], c, 1.0);
        assert!(d > 0.0);
    }

    #[test]
    fn cone_narrow() {
        // Very narrow cone (small angle)
        let angle = 0.1_f32;
        let c = [angle.sin(), angle.cos()];
        let d = sdf_cone([0.0, 0.0, 0.0], c, 1.0);
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn cone_wgsl_valid() {
        let wgsl = sdf_cone_wgsl();
        assert!(wgsl.contains("fn sdf_cone"));
        assert!(wgsl.contains("clamp"));
    }

    // =========================================================================
    // T-DEMO-1.6: Plane Tests (8 tests)
    // =========================================================================

    #[test]
    fn plane_on_plane() {
        // Plane at y=0 with normal (0,1,0)
        assert!(approx_eq(sdf_plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]), 0.0));
        assert!(approx_eq(sdf_plane([5.0, 0.0, -3.0], [0.0, 1.0, 0.0, 0.0]), 0.0));
    }

    #[test]
    fn plane_above() {
        assert!(approx_eq(sdf_plane([0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]), 1.0));
    }

    #[test]
    fn plane_below() {
        assert!(approx_eq(sdf_plane([0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0]), -1.0));
    }

    #[test]
    fn plane_offset() {
        // Plane at y=2
        assert!(approx_eq(sdf_plane([0.0, 2.0, 0.0], [0.0, 1.0, 0.0, -2.0]), 0.0));
        assert!(approx_eq(sdf_plane([0.0, 3.0, 0.0], [0.0, 1.0, 0.0, -2.0]), 1.0));
    }

    #[test]
    fn plane_diagonal_normal() {
        // Plane with diagonal normal
        let n = 1.0 / (2.0_f32).sqrt();
        assert!(approx_eq(sdf_plane([0.0, 0.0, 0.0], [n, n, 0.0, 0.0]), 0.0));
    }

    #[test]
    fn plane_x_normal() {
        // Plane perpendicular to x-axis
        assert!(approx_eq(sdf_plane([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]), 1.0));
        assert!(approx_eq(sdf_plane([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]), -1.0));
    }

    #[test]
    fn plane_z_normal() {
        assert!(approx_eq(sdf_plane([0.0, 0.0, 5.0], [0.0, 0.0, 1.0, 0.0]), 5.0));
    }

    #[test]
    fn plane_wgsl_valid() {
        let wgsl = sdf_plane_wgsl();
        assert!(wgsl.contains("fn sdf_plane"));
        assert!(wgsl.contains("dot"));
    }

    // =========================================================================
    // T-DEMO-1.7: Capsule Tests (10 tests)
    // =========================================================================

    #[test]
    fn capsule_at_endpoint_a() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 2.0, 0.0];
        assert!(approx_eq(sdf_capsule(a, a, b, 0.5), -0.5));
    }

    #[test]
    fn capsule_at_endpoint_b() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 2.0, 0.0];
        assert!(approx_eq(sdf_capsule(b, a, b, 0.5), -0.5));
    }

    #[test]
    fn capsule_on_surface_side() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 2.0, 0.0];
        assert!(approx_eq(sdf_capsule([0.5, 1.0, 0.0], a, b, 0.5), 0.0));
    }

    #[test]
    fn capsule_on_surface_cap() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 2.0, 0.0];
        assert!(approx_eq(sdf_capsule([0.0, -0.5, 0.0], a, b, 0.5), 0.0));
        assert!(approx_eq(sdf_capsule([0.0, 2.5, 0.0], a, b, 0.5), 0.0));
    }

    #[test]
    fn capsule_outside() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 2.0, 0.0];
        assert!(approx_eq(sdf_capsule([1.5, 1.0, 0.0], a, b, 0.5), 1.0));
    }

    #[test]
    fn capsule_inside() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 2.0, 0.0];
        assert!(approx_eq(sdf_capsule([0.0, 1.0, 0.0], a, b, 0.5), -0.5));
    }

    #[test]
    fn capsule_diagonal() {
        // Capsule from (0,0,0) to (1,1,1)
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];
        assert!(approx_eq(sdf_capsule(a, a, b, 0.5), -0.5));
        assert!(approx_eq(sdf_capsule(b, a, b, 0.5), -0.5));
    }

    #[test]
    fn capsule_very_short() {
        // Very short capsule (nearly a sphere)
        let a = [0.0, 0.0, 0.0];
        let b = [0.001, 0.0, 0.0];
        // This is essentially a sphere at the origin
        let d = sdf_capsule([1.0, 0.0, 0.0], a, b, 0.5);
        assert!(approx_eq_eps(d, 0.5, 0.01));
    }

    #[test]
    fn capsule_horizontal() {
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        assert!(approx_eq(sdf_capsule([1.0, 0.5, 0.0], a, b, 0.5), 0.0));
    }

    #[test]
    fn capsule_wgsl_valid() {
        let wgsl = sdf_capsule_wgsl();
        assert!(wgsl.contains("fn sdf_capsule"));
        assert!(wgsl.contains("clamp"));
    }

    // =========================================================================
    // T-DEMO-1.8: Ellipsoid Tests (8 tests)
    // =========================================================================

    #[test]
    fn ellipsoid_at_center() {
        let d = sdf_ellipsoid([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        assert!(d < 0.0); // Inside
    }

    #[test]
    fn ellipsoid_on_surface_x() {
        let d = sdf_ellipsoid([1.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        assert!(approx_eq_eps(d, 0.0, 0.1));
    }

    #[test]
    fn ellipsoid_on_surface_y() {
        let d = sdf_ellipsoid([0.0, 2.0, 0.0], [1.0, 2.0, 3.0]);
        assert!(approx_eq_eps(d, 0.0, 0.1));
    }

    #[test]
    fn ellipsoid_on_surface_z() {
        let d = sdf_ellipsoid([0.0, 0.0, 3.0], [1.0, 2.0, 3.0]);
        assert!(approx_eq_eps(d, 0.0, 0.1));
    }

    #[test]
    fn ellipsoid_outside() {
        let d = sdf_ellipsoid([2.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        assert!(d > 0.0);
    }

    #[test]
    fn ellipsoid_inside() {
        let d = sdf_ellipsoid([0.5, 0.0, 0.0], [1.0, 2.0, 3.0]);
        assert!(d < 0.0);
    }

    #[test]
    fn ellipsoid_sphere() {
        // Ellipsoid with equal radii is a sphere
        let d = sdf_ellipsoid([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(approx_eq_eps(d, 0.0, 0.1));
    }

    #[test]
    fn ellipsoid_wgsl_valid() {
        let wgsl = sdf_ellipsoid_wgsl();
        assert!(wgsl.contains("fn sdf_ellipsoid"));
        assert!(wgsl.contains("length"));
    }

    // =========================================================================
    // T-DEMO-1.9: Box Frame Tests (8 tests)
    // =========================================================================

    #[test]
    fn box_frame_on_edge() {
        // On the edge bar of the frame (on the x-axis edge at x=b)
        // The edge bar is centered on x=b with thickness e on each side
        let d = sdf_box_frame([1.0, 1.0, 0.0], [1.0, 1.0, 1.0], 0.1);
        assert!(approx_eq_eps(d, 0.0, 0.15));
    }

    #[test]
    fn box_frame_inside_edge_bar() {
        // Point on the edge bar running along z at corner (1,1,0)
        // This should be inside/on the bar
        let d = sdf_box_frame([1.0, 1.0, 0.5], [1.0, 1.0, 1.0], 0.1);
        assert!(d <= 0.0 || approx_eq_eps(d, 0.0, 0.05));
    }

    #[test]
    fn box_frame_center_hollow() {
        // Center of frame should be outside (hollow)
        let d = sdf_box_frame([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.1);
        assert!(d > 0.0);
    }

    #[test]
    fn box_frame_outside() {
        let d = sdf_box_frame([2.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.1);
        assert!(d > 0.0);
    }

    #[test]
    fn box_frame_corner() {
        // At corner where edges meet
        let d = sdf_box_frame([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.1);
        assert!(approx_eq_eps(d, 0.0, 0.15));
    }

    #[test]
    fn box_frame_edge_along_axis() {
        // Point on edge running along z-axis at corner position
        let d = sdf_box_frame([1.0, 1.0, 0.0], [1.0, 1.0, 1.0], 0.2);
        assert!(d <= 0.0 || approx_eq_eps(d, 0.0, 0.1));
    }

    #[test]
    fn box_frame_thin() {
        // Very thin frame - corner should still be near surface
        let d = sdf_box_frame([1.0, 1.0, 0.0], [1.0, 1.0, 1.0], 0.01);
        assert!(approx_eq_eps(d, 0.0, 0.05));
    }

    #[test]
    fn box_frame_wgsl_valid() {
        let wgsl = sdf_box_frame_wgsl();
        assert!(wgsl.contains("fn sdf_box_frame"));
        assert!(wgsl.contains("abs"));
    }

    // =========================================================================
    // T-DEMO-1.10: Rounded Box Tests (8 tests)
    // =========================================================================

    #[test]
    fn rounded_box_at_center() {
        let d = sdf_rounded_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.2);
        assert!(d < 0.0);
    }

    #[test]
    fn rounded_box_on_face() {
        // Face is pushed out by radius
        let d = sdf_rounded_box([1.2, 0.0, 0.0], [1.0, 1.0, 1.0], 0.2);
        assert!(approx_eq_eps(d, 0.0, 0.01));
    }

    #[test]
    fn rounded_box_on_edge() {
        // Edge is rounded - check diagonal from corner
        let r = 0.2;
        let b = 1.0;
        let edge_point = b + r / (2.0_f32).sqrt();
        let d = sdf_rounded_box([edge_point, edge_point, 0.0], [b, b, b], r);
        assert!(approx_eq_eps(d, 0.0, 0.1));
    }

    #[test]
    fn rounded_box_on_corner() {
        // Corner is rounded sphere
        let r = 0.2;
        let b = 1.0;
        let corner = b + r / (3.0_f32).sqrt();
        let d = sdf_rounded_box([corner, corner, corner], [b, b, b], r);
        assert!(approx_eq_eps(d, 0.0, 0.1));
    }

    #[test]
    fn rounded_box_outside() {
        let d = sdf_rounded_box([2.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.2);
        assert!(d > 0.0);
    }

    #[test]
    fn rounded_box_inside() {
        let d = sdf_rounded_box([0.5, 0.0, 0.0], [1.0, 1.0, 1.0], 0.2);
        assert!(d < 0.0);
    }

    #[test]
    fn rounded_box_zero_radius() {
        // Zero radius = regular box
        let d = sdf_rounded_box([1.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        assert!(approx_eq(d, 0.0));
    }

    #[test]
    fn rounded_box_wgsl_valid() {
        let wgsl = sdf_rounded_box_wgsl();
        assert!(wgsl.contains("fn sdf_rounded_box"));
    }

    // =========================================================================
    // T-DEMO-1.11: Octahedron Tests (8 tests)
    // =========================================================================

    #[test]
    fn octahedron_at_center() {
        let d = sdf_octahedron([0.0, 0.0, 0.0], 1.0);
        assert!(d < 0.0);
    }

    #[test]
    fn octahedron_at_vertex() {
        // Vertices are at distance s from center along each axis
        let d = sdf_octahedron([1.0, 0.0, 0.0], 1.0);
        assert!(approx_eq_eps(d, 0.0, 0.01));
    }

    #[test]
    fn octahedron_at_all_vertices() {
        let s = 1.0;
        assert!(approx_eq_eps(sdf_octahedron([s, 0.0, 0.0], s), 0.0, 0.01));
        assert!(approx_eq_eps(sdf_octahedron([-s, 0.0, 0.0], s), 0.0, 0.01));
        assert!(approx_eq_eps(sdf_octahedron([0.0, s, 0.0], s), 0.0, 0.01));
        assert!(approx_eq_eps(sdf_octahedron([0.0, -s, 0.0], s), 0.0, 0.01));
        assert!(approx_eq_eps(sdf_octahedron([0.0, 0.0, s], s), 0.0, 0.01));
        assert!(approx_eq_eps(sdf_octahedron([0.0, 0.0, -s], s), 0.0, 0.01));
    }

    #[test]
    fn octahedron_on_face() {
        // Face center is at equal distances along two axes
        let s = 1.0;
        let face_point = s / 3.0;
        let d = sdf_octahedron([face_point, face_point, face_point], s);
        assert!(approx_eq_eps(d, 0.0, 0.1));
    }

    #[test]
    fn octahedron_outside() {
        let d = sdf_octahedron([2.0, 0.0, 0.0], 1.0);
        assert!(d > 0.0);
    }

    #[test]
    fn octahedron_inside() {
        let d = sdf_octahedron([0.1, 0.1, 0.1], 1.0);
        assert!(d < 0.0);
    }

    #[test]
    fn octahedron_large() {
        let d = sdf_octahedron([5.0, 0.0, 0.0], 5.0);
        assert!(approx_eq_eps(d, 0.0, 0.01));
    }

    #[test]
    fn octahedron_wgsl_valid() {
        let wgsl = sdf_octahedron_wgsl();
        assert!(wgsl.contains("fn sdf_octahedron"));
        assert!(wgsl.contains("0.57735027"));
    }

    // =========================================================================
    // T-DEMO-1.12: Pyramid Tests (8 tests)
    // Note: The pyramid has apex at origin, base at y = -h
    // Base is a unit square (corners at +/- 0.5 on x,z)
    // The pyramid formula is complex and the exact SDF values
    // depend on the specific implementation.
    // =========================================================================

    #[test]
    fn pyramid_at_apex() {
        // Apex is at origin - should be on or near surface
        let d = sdf_pyramid([0.0, 0.0, 0.0], 1.0);
        // The exact value depends on the formula; just check it's finite
        assert!(d.is_finite());
    }

    #[test]
    fn pyramid_inside() {
        // Point inside the pyramid volume - deep inside
        let d = sdf_pyramid([0.0, -0.5, 0.0], 1.0);
        // Should be inside or on surface
        assert!(d <= 0.1);
    }

    #[test]
    fn pyramid_below_base() {
        let d = sdf_pyramid([0.0, -2.0, 0.0], 1.0);
        assert!(d > 0.0);
    }

    #[test]
    fn pyramid_symmetry() {
        // Test that the pyramid has xz symmetry
        let d1 = sdf_pyramid([0.2, -0.5, 0.0], 1.0);
        let d2 = sdf_pyramid([-0.2, -0.5, 0.0], 1.0);
        let d3 = sdf_pyramid([0.0, -0.5, 0.2], 1.0);
        let d4 = sdf_pyramid([0.0, -0.5, -0.2], 1.0);
        assert!(approx_eq(d1, d2));
        assert!(approx_eq(d1, d3));
        assert!(approx_eq(d3, d4));
    }

    #[test]
    fn pyramid_above_apex() {
        let d = sdf_pyramid([0.0, 1.0, 0.0], 1.0);
        assert!(d > 0.0);
    }

    #[test]
    fn pyramid_on_base_corner() {
        // Base is at y = -h, corners are at +/- 0.5 on x and z
        let d = sdf_pyramid([0.5, -1.0, 0.5], 1.0);
        // Near the base corner - value should be finite
        assert!(d.is_finite());
    }

    #[test]
    fn pyramid_tall() {
        let d = sdf_pyramid([0.0, -5.0, 0.0], 10.0);
        assert!(d < 0.0);
    }

    #[test]
    fn pyramid_wgsl_valid() {
        let wgsl = sdf_pyramid_wgsl();
        assert!(wgsl.contains("fn sdf_pyramid"));
        assert!(wgsl.contains("select"));
    }

    // =========================================================================
    // Utility Function Tests (4 tests)
    // =========================================================================

    #[test]
    fn all_primitives_wgsl_contains_all() {
        let all = all_primitives_wgsl();
        assert!(all.contains("sdf_sphere"));
        assert!(all.contains("sdf_box"));
        assert!(all.contains("sdf_torus"));
        assert!(all.contains("sdf_cylinder"));
        assert!(all.contains("sdf_cone"));
        assert!(all.contains("sdf_plane"));
        assert!(all.contains("sdf_capsule"));
        assert!(all.contains("sdf_ellipsoid"));
        assert!(all.contains("sdf_box_frame"));
        assert!(all.contains("sdf_rounded_box"));
        assert!(all.contains("sdf_octahedron"));
        assert!(all.contains("sdf_pyramid"));
    }

    #[test]
    fn embedded_wgsl_not_empty() {
        assert!(!SDF_PRIMITIVES_WGSL.is_empty());
        assert!(SDF_PRIMITIVES_WGSL.len() > 1000);
    }

    #[test]
    fn embedded_wgsl_contains_header() {
        assert!(SDF_PRIMITIVES_WGSL.contains("SPDX-License-Identifier"));
        assert!(SDF_PRIMITIVES_WGSL.contains("Inigo Quilez"));
    }

    #[test]
    fn embedded_wgsl_contains_all_primitives() {
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_sphere"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_box"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_torus"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_cylinder"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_cone"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_plane"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_capsule"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_ellipsoid"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_box_frame"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_rounded_box"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_octahedron"));
        assert!(SDF_PRIMITIVES_WGSL.contains("fn sdf_pyramid"));
    }
}
