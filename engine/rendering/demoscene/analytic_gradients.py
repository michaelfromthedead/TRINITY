"""
TRINITY Analytic Gradient Propagation for SDF Primitives (T-DEMO-8.1)

This module provides analytic gradient (normal) computation for SDF primitives,
avoiding the 6-sample central difference method traditionally used for normal
estimation in ray marching.

Benefits:
- 6x fewer SDF evaluations for normal computation
- Exact mathematical derivatives instead of numerical approximations
- Better handling of sharp features and discontinuities
- Winner-ID tracking for combinator gradient propagation

Mathematical Background:
- For an SDF f(p), the gradient grad(f) = (df/dx, df/dy, df/dz)
- The unit surface normal is n = normalize(grad(f))
- At the surface (f=0), the gradient points outward

Primitive Gradients:
- Sphere: grad = p / |p|
- Box: grad based on dominant axis
- Torus: grad from 2D circle in xz-plane extruded
- Cylinder: combination of radial and axial components
- Cone: combination of radial and slope components
- Plane: constant normal
- Capsule: gradient of closest point on line segment
- Ellipsoid: normalized scaled position
- etc.

Combinator Gradient Propagation:
- Union: gradient from primitive with min distance (winner)
- Intersection: gradient from primitive with max distance (winner)
- Subtraction: gradient from winner, negated for subtracted shape
- Smooth combinators: blend gradients based on blend weights

Reference:
- Inigo Quilez, "Normals for an SDF"
- https://iquilezles.org/articles/normalsSDF/
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

from .sdf_ast import Vec3


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    # Gradient result types
    "GradientResult",
    "CombinatorGradientResult",
    # Primitive gradient functions
    "gradient_sphere",
    "gradient_box",
    "gradient_torus",
    "gradient_cylinder",
    "gradient_cone",
    "gradient_plane",
    "gradient_capsule",
    "gradient_ellipsoid",
    "gradient_box_frame",
    "gradient_rounded_box",
    "gradient_octahedron",
    "gradient_pyramid",
    # Combinator gradient functions
    "gradient_union",
    "gradient_intersection",
    "gradient_subtraction",
    "gradient_smooth_union",
    "gradient_smooth_intersection",
    "gradient_smooth_subtraction",
    # Validation
    "validate_gradient",
    "gradient_vs_central_diff",
    # Utilities
    "normalize_gradient",
    "central_difference_gradient",
]

# Numerical constants
EPSILON = 1e-10
GRADIENT_EPSILON = 1e-4  # For validation comparisons
CENTRAL_DIFF_H = 1e-4    # Step size for central differences


# =============================================================================
# Helper Functions
# =============================================================================


def vec3_length(v: Vec3) -> float:
    """Compute vector length."""
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def vec3_length_squared(v: Vec3) -> float:
    """Compute squared vector length."""
    return v.x * v.x + v.y * v.y + v.z * v.z


def vec3_dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two vectors."""
    return a.x * b.x + a.y * b.y + a.z * b.z


def vec3_cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two vectors."""
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    )


def normalize_gradient(g: Vec3) -> Vec3:
    """Normalize a gradient vector to unit length."""
    length = vec3_length(g)
    if length < EPSILON:
        return Vec3(0.0, 1.0, 0.0)  # Default up vector
    inv_len = 1.0 / length
    return Vec3(g.x * inv_len, g.y * inv_len, g.z * inv_len)


def sign(x: float) -> float:
    """Return sign of value: -1, 0, or 1."""
    if x > EPSILON:
        return 1.0
    elif x < -EPSILON:
        return -1.0
    return 0.0


def clamp(x: float, a: float, b: float) -> float:
    """Clamp value to range [a, b]."""
    return max(a, min(b, x))


# =============================================================================
# Gradient Result Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class GradientResult:
    """
    Result of gradient computation for a primitive.

    Attributes:
        gradient: The unnormalized gradient vector (df/dx, df/dy, df/dz)
        distance: The signed distance at the evaluation point
        normal: The normalized surface normal (unit gradient)
    """
    gradient: Vec3
    distance: float

    @property
    def normal(self) -> Vec3:
        """Get normalized surface normal."""
        return normalize_gradient(self.gradient)


@dataclass(frozen=True, slots=True)
class CombinatorGradientResult:
    """
    Result of gradient computation for a combinator.

    Attributes:
        gradient: The combined gradient vector
        distance: The combined signed distance
        winner_id: ID of the primitive whose gradient was selected (0=left, 1=right)
        blend_weight: For smooth combinators, weight of right primitive [0, 1]
    """
    gradient: Vec3
    distance: float
    winner_id: int
    blend_weight: float = 0.0

    @property
    def normal(self) -> Vec3:
        """Get normalized surface normal."""
        return normalize_gradient(self.gradient)


# =============================================================================
# Primitive Gradient Functions (T-DEMO-8.1.1)
# =============================================================================


def gradient_sphere(p: Vec3, radius: float) -> GradientResult:
    """
    Compute analytic gradient for a sphere SDF.

    SDF: f(p) = |p| - r
    Gradient: grad(f) = p / |p|

    At the origin, returns a default up vector.

    Args:
        p: Query point (relative to sphere center)
        radius: Sphere radius

    Returns:
        GradientResult with gradient and distance
    """
    length = vec3_length(p)
    distance = length - radius

    if length < EPSILON:
        # At center, gradient is undefined - use default
        return GradientResult(Vec3(0.0, 1.0, 0.0), distance)

    inv_len = 1.0 / length
    gradient = Vec3(p.x * inv_len, p.y * inv_len, p.z * inv_len)

    return GradientResult(gradient, distance)


def gradient_box(p: Vec3, half_extents: Vec3) -> GradientResult:
    """
    Compute analytic gradient for an axis-aligned box SDF.

    The gradient depends on whether the point is outside, on an edge/corner,
    or inside the box.

    Args:
        p: Query point (relative to box center)
        half_extents: Half-size of box along each axis

    Returns:
        GradientResult with gradient and distance
    """
    # Compute signed distances to each face
    qx = abs(p.x) - half_extents.x
    qy = abs(p.y) - half_extents.y
    qz = abs(p.z) - half_extents.z

    # Signs for gradient direction
    sx = sign(p.x)
    sy = sign(p.y)
    sz = sign(p.z)

    # Outside: gradient points from nearest surface point
    outside_x = max(qx, 0.0)
    outside_y = max(qy, 0.0)
    outside_z = max(qz, 0.0)
    outside_len = math.sqrt(outside_x * outside_x + outside_y * outside_y + outside_z * outside_z)

    # Inside: gradient points toward nearest face
    inside_dist = max(qx, max(qy, qz))

    if outside_len > EPSILON:
        # Outside the box - gradient toward point from surface
        inv_len = 1.0 / outside_len
        gx = (outside_x * sx) * inv_len if qx > 0 else 0.0
        gy = (outside_y * sy) * inv_len if qy > 0 else 0.0
        gz = (outside_z * sz) * inv_len if qz > 0 else 0.0
        distance = outside_len
    else:
        # Inside the box - gradient toward nearest face
        distance = inside_dist
        if qx >= qy and qx >= qz:
            gx, gy, gz = sx, 0.0, 0.0
        elif qy >= qx and qy >= qz:
            gx, gy, gz = 0.0, sy, 0.0
        else:
            gx, gy, gz = 0.0, 0.0, sz

    return GradientResult(Vec3(gx, gy, gz), distance)


def gradient_torus(p: Vec3, major_radius: float, minor_radius: float) -> GradientResult:
    """
    Compute analytic gradient for a torus SDF (lying in xz-plane).

    SDF: f(p) = |vec2(|p.xz| - R, p.y)| - r
    where R = major_radius, r = minor_radius

    Args:
        p: Query point
        major_radius: Distance from torus center to tube center
        minor_radius: Tube radius

    Returns:
        GradientResult with gradient and distance
    """
    # Distance in xz-plane to major circle
    len_xz = math.sqrt(p.x * p.x + p.z * p.z)

    if len_xz < EPSILON:
        # On the y-axis, gradient is radial outward in xz
        # Pick arbitrary direction in xz plane
        q0 = -major_radius
        q1 = p.y
        len_q = math.sqrt(q0 * q0 + q1 * q1)
        distance = len_q - minor_radius

        if len_q < EPSILON:
            return GradientResult(Vec3(1.0, 0.0, 0.0), distance)

        # Gradient points away from torus tube center
        inv_q = 1.0 / len_q
        gx = -q0 * inv_q  # Points outward in +x direction
        gy = q1 * inv_q
        return GradientResult(Vec3(gx, gy, 0.0), distance)

    # Vector in xz-plane from origin to point on major circle
    inv_xz = 1.0 / len_xz
    circle_x = p.x * inv_xz * major_radius
    circle_z = p.z * inv_xz * major_radius

    # Vector from point on major circle to query point (in tube cross-section)
    q0 = len_xz - major_radius  # radial distance from major circle
    q1 = p.y                     # height

    len_q = math.sqrt(q0 * q0 + q1 * q1)
    distance = len_q - minor_radius

    if len_q < EPSILON:
        # On the tube center circle, gradient is undefined
        return GradientResult(Vec3(p.x * inv_xz, 0.0, p.z * inv_xz), distance)

    # Gradient in tube cross-section
    inv_q = 1.0 / len_q
    grad_radial = q0 * inv_q  # component toward/from major circle
    grad_y = q1 * inv_q       # component in y direction

    # Transform radial component to world space
    gx = (p.x * inv_xz) * grad_radial
    gz = (p.z * inv_xz) * grad_radial

    return GradientResult(Vec3(gx, grad_y, gz), distance)


def gradient_cylinder(p: Vec3, radius: float, half_height: float) -> GradientResult:
    """
    Compute analytic gradient for a capped cylinder (axis along y).

    Args:
        p: Query point
        radius: Cylinder radius
        half_height: Half the cylinder height

    Returns:
        GradientResult with gradient and distance
    """
    # Radial distance in xz-plane
    len_xz = math.sqrt(p.x * p.x + p.z * p.z)
    dx = len_xz - radius
    dy = abs(p.y) - half_height

    # Handle the different regions
    if dx > 0 and dy > 0:
        # Outside both caps and sides - corner region
        dist = math.sqrt(dx * dx + dy * dy)
        inv_dist = 1.0 / max(dist, EPSILON)
        # Gradient is combination of radial and axial
        if len_xz > EPSILON:
            inv_xz = 1.0 / len_xz
            gx = (p.x * inv_xz) * dx * inv_dist
            gz = (p.z * inv_xz) * dx * inv_dist
        else:
            gx, gz = 0.0, 0.0
        gy = sign(p.y) * dy * inv_dist
        return GradientResult(Vec3(gx, gy, gz), dist)

    elif dx > dy:
        # Radial surface dominates
        distance = dx
        if len_xz > EPSILON:
            inv_xz = 1.0 / len_xz
            gx = p.x * inv_xz
            gz = p.z * inv_xz
        else:
            gx, gz = 1.0, 0.0
        return GradientResult(Vec3(gx, 0.0, gz), distance)

    else:
        # Cap surface dominates
        distance = dy
        gy = sign(p.y)
        return GradientResult(Vec3(0.0, gy, 0.0), distance)


def gradient_cone(p: Vec3, angle: float, height: float) -> GradientResult:
    """
    Compute analytic gradient for a capped cone (apex at origin, axis along +y).

    Args:
        p: Query point
        angle: Half-angle at apex in radians
        height: Height of cone from apex to base

    Returns:
        GradientResult with gradient and distance
    """
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)

    # q defines the cone surface direction
    qx = height * sin_a / cos_a
    qy = -height

    # w is the point in 2D (radial_dist, y)
    len_xz = math.sqrt(p.x * p.x + p.z * p.z)
    wx = len_xz
    wy = p.y

    # Project onto cone surface segment
    dot_wq = wx * qx + wy * qy
    dot_qq = qx * qx + qy * qy
    t = clamp(dot_wq / dot_qq, 0.0, 1.0)

    # Closest point on cone surface
    ax = wx - qx * t
    ay = wy - qy * t

    # Closest point on base
    u = clamp(wx / qx, 0.0, 1.0) if abs(qx) > EPSILON else 0.0
    bx = wx - qx * u
    by = wy - qy

    # Determine which distance is smaller
    da = ax * ax + ay * ay
    db = bx * bx + by * by

    k = sign(qy)
    s = max(k * (wx * qy - wy * qx), k * (wy - height))
    distance = math.sqrt(min(da, db)) * sign(s)

    # Compute gradient based on which surface is closest
    if da <= db:
        # Cone surface gradient
        # The cone surface normal in 2D is perpendicular to (qx, qy)
        # Normalized: (qy, -qx) / |q| or (-qy, qx) / |q|
        len_q = math.sqrt(dot_qq)
        if len_q < EPSILON:
            return GradientResult(Vec3(0.0, 1.0, 0.0), distance)

        inv_q = 1.0 / len_q
        # Normal pointing outward from cone
        n_radial = qy * inv_q * sign(s)
        n_y = -qx * inv_q * sign(s)

        if len_xz > EPSILON:
            inv_xz = 1.0 / len_xz
            gx = p.x * inv_xz * n_radial
            gz = p.z * inv_xz * n_radial
        else:
            gx, gz = n_radial, 0.0

        return GradientResult(Vec3(gx, n_y, gz), distance)
    else:
        # Base surface gradient (pointing down if inside, up if outside)
        gy = -sign(s) if wy > height else sign(s)
        return GradientResult(Vec3(0.0, gy, 0.0), distance)


def gradient_plane(p: Vec3, normal: Vec3, distance_from_origin: float) -> GradientResult:
    """
    Compute analytic gradient for an infinite plane.

    SDF: f(p) = dot(p, n) + d
    Gradient: grad(f) = n (constant)

    Args:
        p: Query point
        normal: Unit normal of the plane (must be normalized)
        distance_from_origin: Signed distance from origin to plane

    Returns:
        GradientResult with gradient and distance
    """
    dist = vec3_dot(p, normal) + distance_from_origin
    # Gradient is simply the plane normal
    return GradientResult(normal, dist)


def gradient_capsule(
    p: Vec3,
    endpoint_a: Vec3,
    endpoint_b: Vec3,
    radius: float
) -> GradientResult:
    """
    Compute analytic gradient for a capsule (line segment with radius).

    The capsule is essentially a sphere swept along a line segment.

    Args:
        p: Query point
        endpoint_a: First endpoint of capsule axis
        endpoint_b: Second endpoint of capsule axis
        radius: Capsule radius

    Returns:
        GradientResult with gradient and distance
    """
    # Vector from a to p
    pa = Vec3(p.x - endpoint_a.x, p.y - endpoint_a.y, p.z - endpoint_a.z)
    # Vector from a to b
    ba = Vec3(
        endpoint_b.x - endpoint_a.x,
        endpoint_b.y - endpoint_a.y,
        endpoint_b.z - endpoint_a.z
    )

    # Project p onto line ab
    dot_pa_ba = vec3_dot(pa, ba)
    dot_ba_ba = vec3_dot(ba, ba)

    if dot_ba_ba < EPSILON:
        # Degenerate capsule (endpoints coincide) - treat as sphere
        return gradient_sphere(pa, radius)

    h = clamp(dot_pa_ba / dot_ba_ba, 0.0, 1.0)

    # Closest point on line segment to p
    closest = Vec3(
        endpoint_a.x + ba.x * h,
        endpoint_a.y + ba.y * h,
        endpoint_a.z + ba.z * h
    )

    # Vector from closest point to p
    diff = Vec3(p.x - closest.x, p.y - closest.y, p.z - closest.z)
    length = vec3_length(diff)
    distance = length - radius

    if length < EPSILON:
        # At the capsule axis, pick perpendicular direction
        # Use cross product with arbitrary vector
        if abs(ba.y) < 0.9:
            perp = vec3_cross(ba, Vec3(0.0, 1.0, 0.0))
        else:
            perp = vec3_cross(ba, Vec3(1.0, 0.0, 0.0))
        perp = normalize_gradient(perp)
        return GradientResult(perp, distance)

    inv_len = 1.0 / length
    gradient = Vec3(diff.x * inv_len, diff.y * inv_len, diff.z * inv_len)

    return GradientResult(gradient, distance)


def gradient_ellipsoid(p: Vec3, radii: Vec3) -> GradientResult:
    """
    Compute analytic gradient for an ellipsoid.

    The ellipsoid SDF is approximate (exact requires iterative solve).
    SDF: f(p) = k0 * (k0 - 1) / k1
    where k0 = |p/r|, k1 = |p/r^2|

    Gradient is derived from this formula.

    Args:
        p: Query point
        radii: Radii along each axis (rx, ry, rz)

    Returns:
        GradientResult with gradient and distance
    """
    # Normalize to unit sphere space
    px = p.x / radii.x if abs(radii.x) > EPSILON else 0.0
    py = p.y / radii.y if abs(radii.y) > EPSILON else 0.0
    pz = p.z / radii.z if abs(radii.z) > EPSILON else 0.0
    k0 = math.sqrt(px * px + py * py + pz * pz)

    # Second normalization
    px2 = p.x / (radii.x * radii.x) if abs(radii.x) > EPSILON else 0.0
    py2 = p.y / (radii.y * radii.y) if abs(radii.y) > EPSILON else 0.0
    pz2 = p.z / (radii.z * radii.z) if abs(radii.z) > EPSILON else 0.0
    k1 = math.sqrt(px2 * px2 + py2 * py2 + pz2 * pz2)

    if k1 < EPSILON:
        # At center
        min_r = min(radii.x, min(radii.y, radii.z))
        return GradientResult(Vec3(0.0, 1.0, 0.0), -min_r)

    distance = k0 * (k0 - 1.0) / k1

    # Gradient of ellipsoid SDF
    # Using the approximation: grad ~ p / r^2 normalized
    gx = px2
    gy = py2
    gz = pz2
    gradient = normalize_gradient(Vec3(gx, gy, gz))

    return GradientResult(gradient, distance)


def gradient_box_frame(
    p: Vec3,
    half_extents: Vec3,
    edge_thickness: float
) -> GradientResult:
    """
    Compute analytic gradient for a hollow box frame (edges only).

    Args:
        p: Query point
        half_extents: Outer half-extents of the frame
        edge_thickness: Half-width of the frame bars

    Returns:
        GradientResult with gradient and distance
    """
    qx = abs(p.x) - half_extents.x
    qy = abs(p.y) - half_extents.y
    qz = abs(p.z) - half_extents.z

    wx = abs(qx + edge_thickness) - edge_thickness
    wy = abs(qy + edge_thickness) - edge_thickness
    wz = abs(qz + edge_thickness) - edge_thickness

    sx = sign(p.x)
    sy = sign(p.y)
    sz = sign(p.z)

    swx = sign(qx + edge_thickness)
    swy = sign(qy + edge_thickness)
    swz = sign(qz + edge_thickness)

    def edge_dist_grad(a: float, b: float, c: float,
                       sa: float, sb: float, sc: float) -> Tuple[float, Vec3]:
        """Compute edge distance and gradient."""
        outside_a = max(a, 0.0)
        outside_b = max(b, 0.0)
        outside_c = max(c, 0.0)
        outside_len = math.sqrt(outside_a * outside_a + outside_b * outside_b + outside_c * outside_c)
        inside_dist = max(a, max(b, c))

        if outside_len > EPSILON:
            dist = outside_len
            inv_len = 1.0 / outside_len
            ga = (outside_a * sa) * inv_len if a > 0 else 0.0
            gb = (outside_b * sb) * inv_len if b > 0 else 0.0
            gc = (outside_c * sc) * inv_len if c > 0 else 0.0
        else:
            dist = inside_dist
            if a >= b and a >= c:
                ga, gb, gc = sa, 0.0, 0.0
            elif b >= a and b >= c:
                ga, gb, gc = 0.0, sb, 0.0
            else:
                ga, gb, gc = 0.0, 0.0, sc

        return dist, Vec3(ga, gb, gc)

    d1, g1 = edge_dist_grad(qx, wy, wz, sx, sy * swy, sz * swz)
    d2, g2 = edge_dist_grad(wx, qy, wz, sx * swx, sy, sz * swz)
    d3, g3 = edge_dist_grad(wx, wy, qz, sx * swx, sy * swy, sz)

    if d1 <= d2 and d1 <= d3:
        return GradientResult(g1, d1)
    elif d2 <= d1 and d2 <= d3:
        return GradientResult(g2, d2)
    else:
        return GradientResult(g3, d3)


def gradient_rounded_box(
    p: Vec3,
    half_extents: Vec3,
    corner_radius: float
) -> GradientResult:
    """
    Compute analytic gradient for a rounded box.

    A rounded box is just a box with corner radius subtracted from distance.
    The gradient is the same as the box gradient.

    Args:
        p: Query point
        half_extents: Half-extents of the inner box (before rounding)
        corner_radius: Corner rounding radius

    Returns:
        GradientResult with gradient and distance
    """
    box_result = gradient_box(p, half_extents)
    # Just subtract corner radius from distance
    return GradientResult(box_result.gradient, box_result.distance - corner_radius)


def gradient_octahedron(p: Vec3, size: float) -> GradientResult:
    """
    Compute analytic gradient for a regular octahedron.

    Args:
        p: Query point
        size: Distance from center to any vertex

    Returns:
        GradientResult with gradient and distance
    """
    qx = abs(p.x)
    qy = abs(p.y)
    qz = abs(p.z)

    sx = sign(p.x)
    sy = sign(p.y)
    sz = sign(p.z)

    m = qx + qy + qz - size

    # Check which face region
    if 3.0 * qx < m and 3.0 * qy < m and 3.0 * qz < m:
        # Inside the "cut corners" region - use face normal
        # Normal is (1, 1, 1) / sqrt(3) with appropriate signs
        inv_sqrt3 = 0.577350269
        return GradientResult(
            Vec3(sx * inv_sqrt3, sy * inv_sqrt3, sz * inv_sqrt3),
            m * inv_sqrt3
        )

    # Sort to find which face region
    if 3.0 * qx < m:
        kx, ky, kz = qx, qy, qz
        skx, sky, skz = sx, sy, sz
    elif 3.0 * qy < m:
        kx, ky, kz = qy, qz, qx
        skx, sky, skz = sy, sz, sx
    else:  # 3.0 * qz < m
        kx, ky, kz = qz, qx, qy
        skx, sky, skz = sz, sx, sy

    o = clamp(0.5 * (kz - ky + size), 0.0, size)
    dx = kx
    dy = ky - size + o
    dz = kz - o

    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    if dist < EPSILON:
        # On the edge, use face normal
        inv_sqrt3 = 0.577350269
        return GradientResult(
            Vec3(sx * inv_sqrt3, sy * inv_sqrt3, sz * inv_sqrt3),
            dist
        )

    inv_dist = 1.0 / dist

    # Map gradient back to original coordinate system
    if 3.0 * qx < m:
        gx = dx * inv_dist * skx
        gy = dy * inv_dist * sky
        gz = dz * inv_dist * skz
    elif 3.0 * qy < m:
        gy = dx * inv_dist * sky
        gz = dy * inv_dist * skz
        gx = dz * inv_dist * skx
    else:
        gz = dx * inv_dist * skz
        gx = dy * inv_dist * skx
        gy = dz * inv_dist * sky

    return GradientResult(Vec3(gx, gy, gz), dist)


def gradient_pyramid(p: Vec3, height: float) -> GradientResult:
    """
    Compute analytic gradient for a square pyramid (base at y=0, apex at y=height).

    Args:
        p: Query point
        height: Height of pyramid from base to apex

    Returns:
        GradientResult with gradient and distance
    """
    # Pyramid with unit base, height h
    # Fold to first quadrant
    px = abs(p.x)
    pz = abs(p.z)
    sx = sign(p.x)
    sz = sign(p.z)

    # Swap if needed so px >= pz
    if pz > px:
        px, pz = pz, px
        sx, sz = sz, sx

    # Inside or outside base?
    if p.y < 0:
        # Below base - distance to base plane
        dist = -p.y
        return GradientResult(Vec3(0.0, -1.0, 0.0), dist)

    # Face normals for pyramid faces
    # Each triangular face has a normal
    base_half = 0.5  # unit pyramid base goes from -0.5 to 0.5

    # Check if outside the pyramid footprint
    if px > base_half and pz <= base_half:
        # Outside in x direction
        slope = height / base_half
        face_nx = slope / math.sqrt(1.0 + slope * slope)
        face_ny = 1.0 / math.sqrt(1.0 + slope * slope)

        # Distance to sloped face
        face_dist = (px - base_half) * face_nx + (p.y) * face_ny - base_half * face_ny

        if face_dist > 0:
            # Outside sloped face
            return GradientResult(Vec3(sx * face_nx, face_ny, 0.0), face_dist)

    # Check apex region
    if p.y > height:
        # Above apex - distance to apex point
        dx = px
        dy = p.y - height
        dz = pz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < EPSILON:
            return GradientResult(Vec3(0.0, 1.0, 0.0), 0.0)
        inv_dist = 1.0 / dist
        return GradientResult(
            Vec3(sx * dx * inv_dist, dy * inv_dist, sz * dz * inv_dist),
            dist
        )

    # Inside pyramid - distance to nearest face
    # Simplified: compute distance to the 4 sloped faces
    slope = height / base_half
    face_n_len = math.sqrt(1.0 + slope * slope)
    face_nx = slope / face_n_len
    face_ny = 1.0 / face_n_len

    # Current height at this x position on the pyramid
    pyramid_y_at_x = height * (1.0 - px / base_half) if px < base_half else 0.0

    if p.y <= pyramid_y_at_x:
        # Inside - distance to nearest sloped face
        # Use the dominant face based on position
        dist_to_face = (base_half - px) * face_nx + (height - p.y) * face_ny
        dist_to_face = -dist_to_face / face_n_len  # Negate for inside

        return GradientResult(Vec3(sx * face_nx, face_ny, 0.0), dist_to_face)

    # Default - use face gradient
    return GradientResult(Vec3(sx * face_nx, face_ny, 0.0), 0.0)


# =============================================================================
# Combinator Gradient Functions (T-DEMO-8.1.2)
# =============================================================================


def gradient_union(
    grad_a: GradientResult,
    grad_b: GradientResult
) -> CombinatorGradientResult:
    """
    Compute gradient for union of two SDFs.

    Union: min(d_a, d_b)
    Gradient comes from primitive with smaller distance (winner).

    Args:
        grad_a: Gradient result from first primitive
        grad_b: Gradient result from second primitive

    Returns:
        CombinatorGradientResult with winner's gradient
    """
    if grad_a.distance <= grad_b.distance:
        return CombinatorGradientResult(
            gradient=grad_a.gradient,
            distance=grad_a.distance,
            winner_id=0
        )
    else:
        return CombinatorGradientResult(
            gradient=grad_b.gradient,
            distance=grad_b.distance,
            winner_id=1
        )


def gradient_intersection(
    grad_a: GradientResult,
    grad_b: GradientResult
) -> CombinatorGradientResult:
    """
    Compute gradient for intersection of two SDFs.

    Intersection: max(d_a, d_b)
    Gradient comes from primitive with larger distance (winner).

    Args:
        grad_a: Gradient result from first primitive
        grad_b: Gradient result from second primitive

    Returns:
        CombinatorGradientResult with winner's gradient
    """
    if grad_a.distance >= grad_b.distance:
        return CombinatorGradientResult(
            gradient=grad_a.gradient,
            distance=grad_a.distance,
            winner_id=0
        )
    else:
        return CombinatorGradientResult(
            gradient=grad_b.gradient,
            distance=grad_b.distance,
            winner_id=1
        )


def gradient_subtraction(
    grad_a: GradientResult,
    grad_b: GradientResult
) -> CombinatorGradientResult:
    """
    Compute gradient for subtraction (a - b).

    Subtraction: max(d_a, -d_b)
    If -d_b wins, the gradient from b is negated.

    Args:
        grad_a: Gradient result from primary primitive
        grad_b: Gradient result from primitive to subtract

    Returns:
        CombinatorGradientResult with appropriate gradient
    """
    neg_db = -grad_b.distance

    if grad_a.distance >= neg_db:
        return CombinatorGradientResult(
            gradient=grad_a.gradient,
            distance=grad_a.distance,
            winner_id=0
        )
    else:
        # Negate the gradient from b (surface is now "inverted")
        neg_grad = Vec3(-grad_b.gradient.x, -grad_b.gradient.y, -grad_b.gradient.z)
        return CombinatorGradientResult(
            gradient=neg_grad,
            distance=neg_db,
            winner_id=1
        )


def _smooth_blend_factor(a: float, b: float, k: float) -> float:
    """Compute blend factor for smooth operations."""
    k = max(k, EPSILON)
    h = max(k - abs(a - b), 0.0) / k
    if a <= b:
        return h * h * 0.5
    else:
        return 1.0 - h * h * 0.5


def gradient_smooth_union(
    grad_a: GradientResult,
    grad_b: GradientResult,
    k: float
) -> CombinatorGradientResult:
    """
    Compute gradient for smooth union of two SDFs.

    Gradients are blended based on the smooth blend factor.

    Args:
        grad_a: Gradient result from first primitive
        grad_b: Gradient result from second primitive
        k: Smoothness factor

    Returns:
        CombinatorGradientResult with blended gradient
    """
    k = max(k, EPSILON)
    h = max(k - abs(grad_a.distance - grad_b.distance), 0.0) / k

    # Smooth minimum distance
    dist = min(grad_a.distance, grad_b.distance) - h * h * k * 0.25

    # Blend factor: 0 = pure a, 1 = pure b
    t = _smooth_blend_factor(grad_a.distance, grad_b.distance, k)

    # Blend gradients
    ga = grad_a.gradient
    gb = grad_b.gradient
    blended = Vec3(
        ga.x * (1.0 - t) + gb.x * t,
        ga.y * (1.0 - t) + gb.y * t,
        ga.z * (1.0 - t) + gb.z * t
    )

    winner_id = 0 if grad_a.distance <= grad_b.distance else 1

    return CombinatorGradientResult(
        gradient=blended,
        distance=dist,
        winner_id=winner_id,
        blend_weight=t
    )


def gradient_smooth_intersection(
    grad_a: GradientResult,
    grad_b: GradientResult,
    k: float
) -> CombinatorGradientResult:
    """
    Compute gradient for smooth intersection of two SDFs.

    Args:
        grad_a: Gradient result from first primitive
        grad_b: Gradient result from second primitive
        k: Smoothness factor

    Returns:
        CombinatorGradientResult with blended gradient
    """
    k = max(k, EPSILON)
    h = max(k - abs(grad_a.distance - grad_b.distance), 0.0) / k

    # Smooth maximum distance
    dist = max(grad_a.distance, grad_b.distance) + h * h * k * 0.25

    # For intersection, blend factor is based on negated distances
    t = _smooth_blend_factor(-grad_a.distance, -grad_b.distance, k)

    # Blend gradients
    ga = grad_a.gradient
    gb = grad_b.gradient
    blended = Vec3(
        ga.x * (1.0 - t) + gb.x * t,
        ga.y * (1.0 - t) + gb.y * t,
        ga.z * (1.0 - t) + gb.z * t
    )

    winner_id = 0 if grad_a.distance >= grad_b.distance else 1

    return CombinatorGradientResult(
        gradient=blended,
        distance=dist,
        winner_id=winner_id,
        blend_weight=t
    )


def gradient_smooth_subtraction(
    grad_a: GradientResult,
    grad_b: GradientResult,
    k: float
) -> CombinatorGradientResult:
    """
    Compute gradient for smooth subtraction (a - b).

    Args:
        grad_a: Gradient result from primary primitive
        grad_b: Gradient result from primitive to subtract
        k: Smoothness factor

    Returns:
        CombinatorGradientResult with blended gradient
    """
    # Smooth subtraction is smooth intersection with negated b
    neg_b = GradientResult(
        gradient=Vec3(-grad_b.gradient.x, -grad_b.gradient.y, -grad_b.gradient.z),
        distance=-grad_b.distance
    )

    result = gradient_smooth_intersection(grad_a, neg_b, k)

    # Adjust winner_id to refer to original primitives
    return CombinatorGradientResult(
        gradient=result.gradient,
        distance=result.distance,
        winner_id=result.winner_id,
        blend_weight=result.blend_weight
    )


# =============================================================================
# Validation Functions (T-DEMO-8.1.3)
# =============================================================================


def central_difference_gradient(
    sdf_func: Callable[[Vec3], float],
    p: Vec3,
    h: float = CENTRAL_DIFF_H
) -> Vec3:
    """
    Compute gradient using central differences (for validation).

    This is the traditional 6-sample method that analytic gradients aim to replace.

    Args:
        sdf_func: SDF evaluation function
        p: Query point
        h: Step size for finite differences

    Returns:
        Gradient vector computed via central differences
    """
    gx = (sdf_func(Vec3(p.x + h, p.y, p.z)) - sdf_func(Vec3(p.x - h, p.y, p.z))) / (2.0 * h)
    gy = (sdf_func(Vec3(p.x, p.y + h, p.z)) - sdf_func(Vec3(p.x, p.y - h, p.z))) / (2.0 * h)
    gz = (sdf_func(Vec3(p.x, p.y, p.z + h)) - sdf_func(Vec3(p.x, p.y, p.z - h))) / (2.0 * h)

    return Vec3(gx, gy, gz)


def validate_gradient(
    analytic_gradient: Vec3,
    numerical_gradient: Vec3,
    epsilon: float = GRADIENT_EPSILON
) -> Tuple[bool, float]:
    """
    Validate that analytic gradient matches numerical gradient.

    Args:
        analytic_gradient: Gradient from analytic computation
        numerical_gradient: Gradient from central differences
        epsilon: Maximum allowed difference

    Returns:
        Tuple of (is_valid, max_component_difference)
    """
    # Normalize both for comparison
    analytic_norm = normalize_gradient(analytic_gradient)
    numerical_norm = normalize_gradient(numerical_gradient)

    diff_x = abs(analytic_norm.x - numerical_norm.x)
    diff_y = abs(analytic_norm.y - numerical_norm.y)
    diff_z = abs(analytic_norm.z - numerical_norm.z)

    max_diff = max(diff_x, diff_y, diff_z)
    is_valid = max_diff < epsilon

    return is_valid, max_diff


def gradient_vs_central_diff(
    primitive_name: str,
    analytic_func: Callable[[Vec3], GradientResult],
    sdf_func: Callable[[Vec3], float],
    test_points: List[Vec3],
    epsilon: float = GRADIENT_EPSILON
) -> Dict[str, any]:
    """
    Compare analytic gradients to central differences for a set of test points.

    Args:
        primitive_name: Name of the primitive being tested
        analytic_func: Function returning GradientResult
        sdf_func: SDF evaluation function
        test_points: List of points to test
        epsilon: Maximum allowed difference

    Returns:
        Dictionary with validation results
    """
    results = {
        "primitive": primitive_name,
        "total_points": len(test_points),
        "passed": 0,
        "failed": 0,
        "max_error": 0.0,
        "avg_error": 0.0,
        "failures": []
    }

    total_error = 0.0

    for p in test_points:
        analytic_result = analytic_func(p)
        numerical_grad = central_difference_gradient(sdf_func, p)

        is_valid, error = validate_gradient(analytic_result.gradient, numerical_grad, epsilon)

        total_error += error
        results["max_error"] = max(results["max_error"], error)

        if is_valid:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "point": (p.x, p.y, p.z),
                "analytic": (analytic_result.gradient.x, analytic_result.gradient.y, analytic_result.gradient.z),
                "numerical": (numerical_grad.x, numerical_grad.y, numerical_grad.z),
                "error": error
            })

    results["avg_error"] = total_error / len(test_points) if test_points else 0.0

    return results
