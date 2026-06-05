// SPDX-License-Identifier: MIT
//
// sdf_primitives.wgsl -- Signed Distance Field primitive functions.
//
// These functions compute the signed distance from a point to various
// geometric primitives. Negative values indicate interior, positive
// values indicate exterior, and zero is on the surface.
//
// All primitives are centered at the origin unless otherwise noted.
// Use domain transformations (sdf_domain.wgsl) to translate, rotate,
// or scale primitives.
//
// Reference: Inigo Quilez -- Distance Functions
// https://iquilezles.org/articles/distfunctions/

// =============================================================================
// T-DEMO-1.1: Sphere
// =============================================================================

/// Signed distance from point p to a sphere centered at the origin.
///   p  -- query point
///   r  -- sphere radius
///   returns -- signed distance (negative inside, positive outside)
fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

// =============================================================================
// T-DEMO-1.2: Box (Axis-Aligned)
// =============================================================================

/// Signed distance from point p to an axis-aligned box centered at the origin.
///   p  -- query point
///   b  -- half-extents (box spans from -b to +b along each axis)
///   returns -- signed distance (negative inside, positive outside)
fn sdf_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// =============================================================================
// T-DEMO-1.3: Torus
// =============================================================================

/// Signed distance from point p to a torus lying in the xz-plane.
///   p  -- query point
///   r  -- vec2(major_radius, minor_radius)
///         major_radius: distance from torus center to tube center
///         minor_radius: tube radius
///   returns -- signed distance (negative inside, positive outside)
fn sdf_torus(p: vec3<f32>, r: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xz) - r.x, p.y);
    return length(q) - r.y;
}

// =============================================================================
// T-DEMO-1.4: Cylinder (Capped)
// =============================================================================

/// Signed distance from point p to a capped cylinder centered at origin,
/// with axis along the y-direction.
///   p  -- query point
///   h  -- vec2(radius, half_height)
///   returns -- signed distance (negative inside, positive outside)
fn sdf_cylinder(p: vec3<f32>, h: vec2<f32>) -> f32 {
    let d = abs(vec2<f32>(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

// =============================================================================
// T-DEMO-1.5: Cone (Capped)
// =============================================================================

/// Signed distance from point p to a capped cone with apex at origin,
/// axis along positive y.
///   p  -- query point
///   c  -- vec2(sin(angle), cos(angle)) where angle is the half-angle at apex
///   h  -- height of the cone from apex to base
///   returns -- signed distance (negative inside, positive outside)
fn sdf_cone(p: vec3<f32>, c: vec2<f32>, h: f32) -> f32 {
    // c is normalized: c.x = sin(angle), c.y = cos(angle)
    let q = h * vec2<f32>(c.x / c.y, -1.0);
    let w = vec2<f32>(length(p.xz), p.y);
    let a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    let b = w - q * vec2<f32>(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    let k = sign(q.y);
    let d = min(dot(a, a), dot(b, b));
    let s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - h));
    return sqrt(d) * sign(s);
}

// =============================================================================
// T-DEMO-1.6: Plane (Infinite)
// =============================================================================

/// Signed distance from point p to an infinite plane.
///   p  -- query point
///   n  -- vec4(nx, ny, nz, d) where (nx, ny, nz) is the unit normal
///         and d is the distance from origin to plane along the normal.
///         Points on the plane satisfy: dot(p, n.xyz) = n.w
///   returns -- signed distance (negative below plane, positive above)
fn sdf_plane(p: vec3<f32>, n: vec4<f32>) -> f32 {
    // n.xyz must be normalized
    return dot(p, n.xyz) + n.w;
}

// =============================================================================
// T-DEMO-1.7: Capsule (Line Segment with Radius)
// =============================================================================

/// Signed distance from point p to a capsule (line segment with rounded ends).
///   p  -- query point
///   a  -- first endpoint of the capsule axis
///   b  -- second endpoint of the capsule axis
///   r  -- capsule radius
///   returns -- signed distance (negative inside, positive outside)
fn sdf_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// =============================================================================
// T-DEMO-1.8: Ellipsoid
// =============================================================================

/// Signed distance from point p to an ellipsoid centered at the origin.
/// This is an approximation -- exact ellipsoid SDF requires iterative solve.
///   p  -- query point
///   r  -- radii along each axis (x, y, z)
///   returns -- approximate signed distance (negative inside, positive outside)
fn sdf_ellipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
    // Normalize to unit sphere space
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

// =============================================================================
// T-DEMO-1.9: Box Frame (Hollow)
// =============================================================================

/// Signed distance from point p to a hollow box frame (edges only).
///   p  -- query point
///   b  -- outer half-extents of the frame
///   e  -- edge thickness (half-width of the frame bars)
///   returns -- signed distance (negative inside bars, positive outside)
fn sdf_box_frame(p: vec3<f32>, b: vec3<f32>, e: f32) -> f32 {
    let q = abs(p) - b;
    let w = abs(q + e) - e;
    return min(min(
        length(max(vec3<f32>(q.x, w.y, w.z), vec3<f32>(0.0))) + min(max(q.x, max(w.y, w.z)), 0.0),
        length(max(vec3<f32>(w.x, q.y, w.z), vec3<f32>(0.0))) + min(max(w.x, max(q.y, w.z)), 0.0)
    ),
        length(max(vec3<f32>(w.x, w.y, q.z), vec3<f32>(0.0))) + min(max(w.x, max(w.y, q.z)), 0.0)
    );
}

// =============================================================================
// T-DEMO-1.10: Rounded Box
// =============================================================================

/// Signed distance from point p to a box with rounded corners.
///   p  -- query point
///   b  -- half-extents of the inner box (before rounding)
///   r  -- corner radius
///   returns -- signed distance (negative inside, positive outside)
fn sdf_rounded_box(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// =============================================================================
// T-DEMO-1.11: Octahedron (Exact)
// =============================================================================

/// Signed distance from point p to a regular octahedron centered at origin.
///   p  -- query point
///   s  -- distance from center to any vertex
///   returns -- signed distance (negative inside, positive outside)
fn sdf_octahedron(p: vec3<f32>, s: f32) -> f32 {
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
        return m * 0.57735027; // 1/sqrt(3)
    }
    let o = clamp(0.5 * (k.z - k.y + s), 0.0, s);
    return length(vec3<f32>(k.x, k.y - s + o, k.z - o));
}

// =============================================================================
// T-DEMO-1.12: Pyramid (Square Base)
// =============================================================================

/// Signed distance from point p to a square pyramid with apex at origin,
/// base in the negative y half-space.
///   p  -- query point
///   h  -- height from apex to base (positive value)
///   returns -- signed distance (negative inside, positive outside)
fn sdf_pyramid(p: vec3<f32>, h: f32) -> f32 {
    let m2 = h * h + 0.25;

    // Symmetry in xz
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

    // Inside/outside sign
    let inside = max(a.y, -q.y - h);
    return select(d, -d, inside < 0.0);
}
