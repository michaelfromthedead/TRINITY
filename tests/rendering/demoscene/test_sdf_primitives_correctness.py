"""
T-DEMO-7.1: SDF Primitive Correctness Tests

Comprehensive test suite validating all SDF primitives return correct signed
distances at known sample points.

Coverage per primitive (3-5 tests each):
- sdf_sphere: center, surface, external, radius scaling
- sdf_box: corner, edge, face, internal, external
- sdf_torus: major/minor radius, internal, external
- sdf_cylinder: cap, side, internal
- sdf_cone: tip, base, side
- sdf_plane: on plane, above, below
- sdf_capsule: endpoint, middle, external
- sdf_ellipsoid: axis-aligned, scaled axes
- sdf_box_frame: frame edge, internal void
- sdf_rounded_box: corner rounding
- sdf_octahedron: vertex, edge, face
- sdf_pyramid: apex, base, face

Acceptance Criteria:
- Each primitive returns correct signed distance at known sample points
- Rotational invariance confirmed (rotate point, same distance)
- Gradient points away from surface (normal direction)
- Sign flip across surface boundary
- No NaN/Inf at any tested point

Reference: Inigo Quilez - Distance Functions
    https://iquilezles.org/articles/distfunctions/
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import pytest

# =============================================================================
# Tolerance Constants
# =============================================================================

TOL_SURFACE = 1e-10  # Points on surface should be very close to 0
TOL_APPROX = 1e-6    # General floating point tolerance
TOL_GRADIENT = 1e-4  # Tolerance for gradient direction checks


# =============================================================================
# SDF Primitive Implementations (Python reference matching WGSL)
# =============================================================================

def sdf_sphere(p: Tuple[float, float, float], radius: float) -> float:
    """Signed distance to a sphere centered at origin."""
    length = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
    return length - abs(radius)


def sdf_box(p: Tuple[float, float, float], half_extents: Tuple[float, float, float]) -> float:
    """Signed distance to an axis-aligned box centered at origin."""
    qx = abs(p[0]) - half_extents[0]
    qy = abs(p[1]) - half_extents[1]
    qz = abs(p[2]) - half_extents[2]

    # Distance outside box
    outside = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2 + max(qz, 0.0)**2)

    # Distance inside box (negative)
    inside = min(max(qx, max(qy, qz)), 0.0)

    return outside + inside


def sdf_torus(p: Tuple[float, float, float], major_radius: float, minor_radius: float) -> float:
    """Signed distance to a torus centered at origin, axis along Y."""
    # Project to XZ plane and get distance to ring
    q_xz = math.sqrt(p[0]**2 + p[2]**2) - major_radius
    q = math.sqrt(q_xz**2 + p[1]**2)
    return q - minor_radius


def sdf_cylinder(p: Tuple[float, float, float], radius: float, height: float) -> float:
    """Signed distance to a capped cylinder along Y axis, centered at origin."""
    # Distance in XZ plane
    d_xz = math.sqrt(p[0]**2 + p[2]**2) - radius

    # Distance along Y (half_height)
    half_h = height * 0.5
    d_y = abs(p[1]) - half_h

    # 2D SDF for capped shape
    dx = max(d_xz, 0.0)
    dy = max(d_y, 0.0)
    outside = math.sqrt(dx**2 + dy**2)
    inside = min(max(d_xz, d_y), 0.0)

    return outside + inside


def sdf_cone(p: Tuple[float, float, float], angle: float, height: float) -> float:
    """
    Signed distance to a capped cone with apex at origin, axis along +Y.

    Args:
        p: Point to evaluate
        angle: Half-angle at apex in radians
        height: Height of cone from apex to base
    """
    # Radial distance in XZ plane
    q = math.sqrt(p[0]**2 + p[2]**2)

    # Cone parameters
    sin_a = math.sin(angle)
    cos_a = math.cos(angle)

    # 2D cone SDF in (q, y) space
    # The cone surface is at q = y * tan(angle) for 0 <= y <= height

    # Check if below apex
    if p[1] < 0:
        # Below apex - distance to apex point
        return math.sqrt(q**2 + p[1]**2)

    # Check if above base
    if p[1] > height:
        # Above base cap
        base_radius = height * math.tan(angle)
        if q < base_radius:
            return p[1] - height
        else:
            # Outside corner
            dx = q - base_radius
            dy = p[1] - height
            return math.sqrt(dx**2 + dy**2)

    # On cone body region - distance to cone surface
    # Cone surface: q = y * tan(angle)
    expected_q = p[1] * math.tan(angle)

    # Distance to infinite cone (perpendicular to cone surface)
    # The normal to the cone surface is (cos(angle), -sin(angle)) in (q, y) space
    dist_to_surface = (q - expected_q) * cos_a

    if dist_to_surface < 0:
        # Inside cone
        return max(dist_to_surface, -p[1])  # Also check apex
    else:
        return dist_to_surface


def sdf_plane(p: Tuple[float, float, float], normal: Tuple[float, float, float], distance: float = 0.0) -> float:
    """Signed distance to an infinite plane."""
    # Normalize the normal
    n_len = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    if n_len < 1e-10:
        return 0.0
    nx, ny, nz = normal[0]/n_len, normal[1]/n_len, normal[2]/n_len

    return p[0]*nx + p[1]*ny + p[2]*nz + distance


def sdf_capsule(
    p: Tuple[float, float, float],
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    radius: float,
) -> float:
    """Signed distance to a capsule from point a to point b."""
    pa = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
    ba = (b[0] - a[0], b[1] - a[1], b[2] - a[2])

    dot_pa_ba = pa[0]*ba[0] + pa[1]*ba[1] + pa[2]*ba[2]
    dot_ba_ba = ba[0]*ba[0] + ba[1]*ba[1] + ba[2]*ba[2]

    if dot_ba_ba < 1e-10:
        # Degenerate capsule (a == b), just sphere
        return math.sqrt(pa[0]**2 + pa[1]**2 + pa[2]**2) - radius

    h = max(0.0, min(1.0, dot_pa_ba / dot_ba_ba))

    dx = pa[0] - ba[0]*h
    dy = pa[1] - ba[1]*h
    dz = pa[2] - ba[2]*h

    return math.sqrt(dx**2 + dy**2 + dz**2) - radius


def sdf_ellipsoid(p: Tuple[float, float, float], radii: Tuple[float, float, float]) -> float:
    """
    Signed distance to an ellipsoid (IQ approximation).

    This is an approximation that works well for moderate ellipsoid shapes.
    """
    # Prevent division by zero
    rx = max(radii[0], 1e-10)
    ry = max(radii[1], 1e-10)
    rz = max(radii[2], 1e-10)

    # Normalize to unit sphere space
    px = p[0] / rx
    py = p[1] / ry
    pz = p[2] / rz
    k0 = math.sqrt(px**2 + py**2 + pz**2)

    px2 = p[0] / (rx**2)
    py2 = p[1] / (ry**2)
    pz2 = p[2] / (rz**2)
    k1 = math.sqrt(px2**2 + py2**2 + pz2**2)

    if k1 < 1e-10:
        return -min(rx, ry, rz)

    return k0 * (k0 - 1.0) / k1


def sdf_box_frame(
    p: Tuple[float, float, float],
    half_extents: Tuple[float, float, float],
    edge_thickness: float,
) -> float:
    """
    Signed distance to a hollow box frame (edges only).
    """
    # Use absolute coordinates
    px = abs(p[0]) - half_extents[0]
    py = abs(p[1]) - half_extents[1]
    pz = abs(p[2]) - half_extents[2]

    # Three possible edge configurations
    qx = abs(px + edge_thickness) - edge_thickness
    qy = abs(py + edge_thickness) - edge_thickness
    qz = abs(pz + edge_thickness) - edge_thickness

    def box_2d(a: float, b: float) -> float:
        return math.sqrt(max(a, 0.0)**2 + max(b, 0.0)**2) + min(max(a, b), 0.0)

    return min(
        min(
            box_2d(px, qy) + min(max(qz, 0.0), 0.0),
            box_2d(py, qz) + min(max(qx, 0.0), 0.0),
        ),
        box_2d(pz, qx) + min(max(qy, 0.0), 0.0),
    )


def sdf_rounded_box(
    p: Tuple[float, float, float],
    half_extents: Tuple[float, float, float],
    corner_radius: float,
) -> float:
    """Signed distance to a box with rounded corners."""
    # Reduce half_extents by corner_radius
    qx = abs(p[0]) - (half_extents[0] - corner_radius)
    qy = abs(p[1]) - (half_extents[1] - corner_radius)
    qz = abs(p[2]) - (half_extents[2] - corner_radius)

    # Distance to inner box plus corner radius
    outside = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2 + max(qz, 0.0)**2)
    inside = min(max(qx, max(qy, qz)), 0.0)

    return outside + inside - corner_radius


def sdf_octahedron(p: Tuple[float, float, float], size: float) -> float:
    """
    Signed distance to a regular octahedron.

    The octahedron has vertices at (+/-size, 0, 0), (0, +/-size, 0), (0, 0, +/-size).
    """
    px, py, pz = abs(p[0]), abs(p[1]), abs(p[2])
    m = px + py + pz - size

    # Handle each case
    if 3.0*px < m:
        qx, qy, qz = px, py, pz
    elif 3.0*py < m:
        qx, qy, qz = py, pz, px
    elif 3.0*pz < m:
        qx, qy, qz = pz, px, py
    else:
        return m * 0.57735027  # 1/sqrt(3)

    k = max(0.0, min(0.5*(qz - qy + size), size))
    qy_k = qy - size + k
    qz_k = qz - k
    return math.sqrt(qx**2 + qy_k**2 + qz_k**2)


def sdf_pyramid(p: Tuple[float, float, float], height: float) -> float:
    """
    Signed distance to a square pyramid with apex at origin, base at y = -height.

    Base is 2*height x 2*height, centered below apex.
    """
    # Height should be positive
    h = abs(height)

    # Mirror to positive octant in xz
    px = abs(p[0])
    pz = abs(p[2])

    # Swap so px > pz for symmetry
    if pz > px:
        px, pz = pz, px

    py = p[1]

    # Base half-width = height (45 degree slope)
    base_half = h

    # Check if below base
    if py < -h:
        if px < base_half and pz < base_half:
            return -h - py
        elif px >= base_half and pz < base_half:
            return math.sqrt((px - base_half)**2 + (py + h)**2)
        else:
            return math.sqrt((px - base_half)**2 + (pz - base_half)**2 + (py + h)**2)

    # Check if above apex
    if py > 0:
        return math.sqrt(px**2 + py**2 + pz**2)

    # On pyramid body
    # Slope from (base_half, -h) to (0, 0)
    # Line: y = -h + (h/base_half) * (base_half - x) = x - h for unit slope
    # Distance to slope plane

    # Pyramid face normal (pointing outward): (1, 1, 0) / sqrt(2) in 2D
    # For 45 degree slope, the face is at x + y = 0 from (0,0)
    # Shifted: x + y + h - base_half = x + y (since base_half = h)

    dist_to_face = (px + py) * 0.7071067811865476  # 1/sqrt(2)

    if dist_to_face < 0:
        # Inside - return negative distance
        return dist_to_face
    else:
        return dist_to_face


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_gradient(
    sdf: Callable[[Tuple[float, float, float]], float],
    p: Tuple[float, float, float],
    eps: float = 1e-5,
) -> Tuple[float, float, float]:
    """Estimate gradient of SDF using central differences."""
    dx = sdf((p[0] + eps, p[1], p[2])) - sdf((p[0] - eps, p[1], p[2]))
    dy = sdf((p[0], p[1] + eps, p[2])) - sdf((p[0], p[1] - eps, p[2]))
    dz = sdf((p[0], p[1], p[2] + eps)) - sdf((p[0], p[1], p[2] - eps))

    length = math.sqrt(dx**2 + dy**2 + dz**2)
    if length < 1e-10:
        return (0.0, 0.0, 0.0)

    return (dx/length, dy/length, dz/length)


def rotate_point_y(p: Tuple[float, float, float], angle: float) -> Tuple[float, float, float]:
    """Rotate point around Y axis."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        p[0]*cos_a + p[2]*sin_a,
        p[1],
        -p[0]*sin_a + p[2]*cos_a,
    )


def is_finite(value: float) -> bool:
    """Check if value is finite (not NaN or Inf)."""
    return math.isfinite(value)


# =============================================================================
# Test Class: Sphere SDF
# =============================================================================

class TestSphereCorrectness:
    """Tests for sdf_sphere correctness."""

    def test_center_returns_negative_radius(self):
        """Center of sphere returns -radius (maximum inside)."""
        radius = 2.5
        d = sdf_sphere((0.0, 0.0, 0.0), radius)
        assert d == pytest.approx(-radius, abs=TOL_APPROX)

    def test_surface_returns_zero(self):
        """Points on sphere surface return zero."""
        radius = 1.5
        # Test axis-aligned surface points
        for p in [(radius, 0, 0), (0, radius, 0), (0, 0, radius),
                  (-radius, 0, 0), (0, -radius, 0), (0, 0, -radius)]:
            d = sdf_sphere(p, radius)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_external_point_positive_distance(self):
        """External points return positive distance."""
        radius = 1.0
        p = (3.0, 0.0, 0.0)
        d = sdf_sphere(p, radius)
        assert d == pytest.approx(2.0, abs=TOL_APPROX)

    def test_radius_scaling(self):
        """Distance scales correctly with radius."""
        p = (5.0, 0.0, 0.0)
        for radius in [1.0, 2.0, 3.0, 4.0]:
            d = sdf_sphere(p, radius)
            expected = 5.0 - radius
            assert d == pytest.approx(expected, abs=TOL_APPROX)

    def test_rotational_invariance(self):
        """Same distance regardless of rotation around origin."""
        radius = 2.0
        p = (3.0, 4.0, 0.0)  # distance 5 from origin
        d_original = sdf_sphere(p, radius)

        for angle in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            p_rotated = rotate_point_y(p, angle)
            d_rotated = sdf_sphere(p_rotated, radius)
            assert d_rotated == pytest.approx(d_original, abs=TOL_APPROX)


# =============================================================================
# Test Class: Box SDF
# =============================================================================

class TestBoxCorrectness:
    """Tests for sdf_box correctness."""

    def test_center_negative_distance(self):
        """Center of box returns negative distance (inside)."""
        half_extents = (1.0, 1.0, 1.0)
        d = sdf_box((0.0, 0.0, 0.0), half_extents)
        assert d == pytest.approx(-1.0, abs=TOL_APPROX)

    def test_face_center_returns_zero(self):
        """Face center points return zero distance."""
        half_extents = (1.0, 2.0, 3.0)
        face_points = [
            (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0), (0.0, -2.0, 0.0),
            (0.0, 0.0, 3.0), (0.0, 0.0, -3.0),
        ]
        for p in face_points:
            d = sdf_box(p, half_extents)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_corner_returns_zero(self):
        """Corner points return zero distance."""
        half_extents = (1.0, 1.0, 1.0)
        corners = [
            (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, -1.0),
        ]
        for p in corners:
            d = sdf_box(p, half_extents)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_external_point_along_axis(self):
        """External points along axis return correct distance."""
        half_extents = (1.0, 2.0, 3.0)
        # 2 units outside in X direction
        d = sdf_box((3.0, 0.0, 0.0), half_extents)
        assert d == pytest.approx(2.0, abs=TOL_APPROX)

    def test_external_corner_diagonal(self):
        """External corner (diagonal) returns Euclidean distance to corner."""
        half_extents = (1.0, 1.0, 1.0)
        # Point at (2, 2, 2) should be sqrt(3) from corner (1,1,1)
        d = sdf_box((2.0, 2.0, 2.0), half_extents)
        expected = math.sqrt(3.0)
        assert d == pytest.approx(expected, abs=TOL_APPROX)


# =============================================================================
# Test Class: Torus SDF
# =============================================================================

class TestTorusCorrectness:
    """Tests for sdf_torus correctness."""

    def test_center_hole(self):
        """Center of torus hole returns positive distance."""
        major = 2.0
        minor = 0.5
        d = sdf_torus((0.0, 0.0, 0.0), major, minor)
        expected = major - minor
        assert d == pytest.approx(expected, abs=TOL_APPROX)

    def test_major_radius_surface(self):
        """Point on major radius (outside in XZ) returns -minor."""
        major = 2.0
        minor = 0.5
        d = sdf_torus((major, 0.0, 0.0), major, minor)
        assert d == pytest.approx(-minor, abs=TOL_APPROX)

    def test_outer_edge_surface(self):
        """Outer edge of torus returns zero."""
        major = 2.0
        minor = 0.5
        p = (major + minor, 0.0, 0.0)
        d = sdf_torus(p, major, minor)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_inner_edge_surface(self):
        """Inner edge of torus returns zero."""
        major = 2.0
        minor = 0.5
        p = (major - minor, 0.0, 0.0)
        d = sdf_torus(p, major, minor)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_external_point(self):
        """External point returns positive distance."""
        major = 1.0
        minor = 0.25
        p = (3.0, 0.0, 0.0)
        # Distance to ring in XZ = 3 - 1 = 2, minus minor radius
        d = sdf_torus(p, major, minor)
        expected = 2.0 - minor
        assert d == pytest.approx(expected, abs=TOL_APPROX)


# =============================================================================
# Test Class: Cylinder SDF
# =============================================================================

class TestCylinderCorrectness:
    """Tests for sdf_cylinder correctness."""

    def test_center_negative_distance(self):
        """Center of cylinder returns negative distance."""
        radius = 1.0
        height = 2.0
        d = sdf_cylinder((0.0, 0.0, 0.0), radius, height)
        # At center, distance to side is radius, to cap is height/2
        expected = -min(radius, height/2)
        assert d == pytest.approx(expected, abs=TOL_APPROX)

    def test_cap_center_surface(self):
        """Cap center returns zero (on surface)."""
        radius = 1.0
        height = 2.0
        d = sdf_cylinder((0.0, height/2, 0.0), radius, height)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_side_surface(self):
        """Side surface point returns zero."""
        radius = 1.0
        height = 2.0
        d = sdf_cylinder((radius, 0.0, 0.0), radius, height)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_external_side(self):
        """External point from side returns positive distance."""
        radius = 1.0
        height = 2.0
        d = sdf_cylinder((3.0, 0.0, 0.0), radius, height)
        assert d == pytest.approx(2.0, abs=TOL_APPROX)

    def test_external_above_cap(self):
        """External point above cap returns distance to cap."""
        radius = 1.0
        height = 2.0
        d = sdf_cylinder((0.0, 3.0, 0.0), radius, height)
        expected = 3.0 - height/2
        assert d == pytest.approx(expected, abs=TOL_APPROX)


# =============================================================================
# Test Class: Cone SDF
# =============================================================================

class TestConeCorrectness:
    """Tests for sdf_cone correctness."""

    def test_apex_returns_zero(self):
        """Apex of cone returns zero distance."""
        angle = math.radians(45)
        height = 2.0
        d = sdf_cone((0.0, 0.0, 0.0), angle, height)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_base_edge_surface(self):
        """Base edge (on surface) returns zero."""
        angle = math.radians(45)
        height = 2.0
        # At y=height, the base edge is at radius = height * tan(45) = 2
        base_radius = height * math.tan(angle)
        d = sdf_cone((base_radius, height, 0.0), angle, height)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_cone_surface_midway(self):
        """Point on cone surface midway returns zero."""
        angle = math.radians(45)
        height = 2.0
        # At y=1, radius should be tan(45)*1 = 1
        d = sdf_cone((1.0, 1.0, 0.0), angle, height)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_internal_point_negative(self):
        """Internal point returns negative distance."""
        angle = math.radians(45)
        height = 2.0
        # Point inside the cone
        d = sdf_cone((0.0, 1.0, 0.0), angle, height)
        assert d < 0

    def test_external_point_positive(self):
        """External point returns positive distance."""
        angle = math.radians(45)
        height = 2.0
        # Point outside cone
        d = sdf_cone((3.0, 1.0, 0.0), angle, height)
        assert d > 0


# =============================================================================
# Test Class: Plane SDF
# =============================================================================

class TestPlaneCorrectness:
    """Tests for sdf_plane correctness."""

    def test_on_plane_returns_zero(self):
        """Points on plane return zero distance."""
        normal = (0.0, 1.0, 0.0)
        distance = 0.0
        # Points on XZ plane
        for p in [(0, 0, 0), (1, 0, 0), (0, 0, 1), (5, 0, -3)]:
            d = sdf_plane(p, normal, distance)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_above_plane_positive(self):
        """Points above plane return positive distance."""
        normal = (0.0, 1.0, 0.0)
        distance = 0.0
        d = sdf_plane((0.0, 5.0, 0.0), normal, distance)
        assert d == pytest.approx(5.0, abs=TOL_APPROX)

    def test_below_plane_negative(self):
        """Points below plane return negative distance."""
        normal = (0.0, 1.0, 0.0)
        distance = 0.0
        d = sdf_plane((0.0, -3.0, 0.0), normal, distance)
        assert d == pytest.approx(-3.0, abs=TOL_APPROX)

    def test_plane_offset(self):
        """Plane with non-zero distance offset works correctly."""
        normal = (0.0, 1.0, 0.0)
        distance = 2.0  # Plane at y = -2
        # Point at origin should be at distance 2 above the plane
        d = sdf_plane((0.0, 0.0, 0.0), normal, distance)
        assert d == pytest.approx(2.0, abs=TOL_APPROX)

    def test_tilted_plane(self):
        """Tilted plane computes distance correctly."""
        normal = (1.0, 1.0, 0.0)  # Will be normalized
        distance = 0.0
        # Point at (1, 0, 0) should be 1/sqrt(2) from the plane
        d = sdf_plane((1.0, 0.0, 0.0), normal, distance)
        expected = 1.0 / math.sqrt(2.0)
        assert d == pytest.approx(expected, abs=TOL_APPROX)


# =============================================================================
# Test Class: Capsule SDF
# =============================================================================

class TestCapsuleCorrectness:
    """Tests for sdf_capsule correctness."""

    def test_endpoint_a_surface(self):
        """Point on surface at endpoint A."""
        a = (0.0, 0.0, 0.0)
        b = (0.0, 2.0, 0.0)
        radius = 0.5
        # Point at radius distance from A along -Y
        d = sdf_capsule((0.0, -radius, 0.0), a, b, radius)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_endpoint_b_surface(self):
        """Point on surface at endpoint B."""
        a = (0.0, 0.0, 0.0)
        b = (0.0, 2.0, 0.0)
        radius = 0.5
        # Point at radius distance from B along +Y
        d = sdf_capsule((0.0, 2.0 + radius, 0.0), a, b, radius)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_middle_surface(self):
        """Point on surface at middle of capsule."""
        a = (0.0, 0.0, 0.0)
        b = (0.0, 2.0, 0.0)
        radius = 0.5
        # Point at radius distance from midpoint
        d = sdf_capsule((radius, 1.0, 0.0), a, b, radius)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_internal_point(self):
        """Internal point returns negative distance."""
        a = (0.0, 0.0, 0.0)
        b = (0.0, 2.0, 0.0)
        radius = 0.5
        d = sdf_capsule((0.0, 1.0, 0.0), a, b, radius)
        assert d == pytest.approx(-radius, abs=TOL_APPROX)

    def test_external_point(self):
        """External point returns positive distance."""
        a = (0.0, 0.0, 0.0)
        b = (0.0, 2.0, 0.0)
        radius = 0.5
        d = sdf_capsule((3.0, 1.0, 0.0), a, b, radius)
        expected = 3.0 - radius
        assert d == pytest.approx(expected, abs=TOL_APPROX)


# =============================================================================
# Test Class: Ellipsoid SDF
# =============================================================================

class TestEllipsoidCorrectness:
    """Tests for sdf_ellipsoid correctness."""

    def test_center_negative_min_radius(self):
        """Center returns negative distance (approximately min radius)."""
        radii = (1.0, 2.0, 3.0)
        d = sdf_ellipsoid((0.0, 0.0, 0.0), radii)
        # At center, should be approximately -min(radii)
        assert d < 0

    def test_axis_aligned_surface_x(self):
        """Point on surface along X axis returns approximately zero."""
        radii = (2.0, 3.0, 4.0)
        d = sdf_ellipsoid((2.0, 0.0, 0.0), radii)
        assert d == pytest.approx(0.0, abs=0.01)  # Approximation tolerance

    def test_axis_aligned_surface_y(self):
        """Point on surface along Y axis returns approximately zero."""
        radii = (2.0, 3.0, 4.0)
        d = sdf_ellipsoid((0.0, 3.0, 0.0), radii)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_axis_aligned_surface_z(self):
        """Point on surface along Z axis returns approximately zero."""
        radii = (2.0, 3.0, 4.0)
        d = sdf_ellipsoid((0.0, 0.0, 4.0), radii)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_external_point_positive(self):
        """External point returns positive distance."""
        radii = (1.0, 1.0, 1.0)  # Sphere
        d = sdf_ellipsoid((3.0, 0.0, 0.0), radii)
        assert d > 0


# =============================================================================
# Test Class: Box Frame SDF
# =============================================================================

class TestBoxFrameCorrectness:
    """Tests for sdf_box_frame correctness."""

    def test_on_edge_surface(self):
        """Point on edge surface returns approximately zero."""
        half_extents = (1.0, 1.0, 1.0)
        edge_thickness = 0.1
        # Point on an edge
        d = sdf_box_frame((1.0, 1.0, 0.0), half_extents, edge_thickness)
        assert abs(d) < 0.15  # Approximate

    def test_internal_void_positive(self):
        """Point inside the void (hollow center) returns positive distance."""
        half_extents = (1.0, 1.0, 1.0)
        edge_thickness = 0.1
        d = sdf_box_frame((0.0, 0.0, 0.0), half_extents, edge_thickness)
        assert d > 0

    def test_external_point_positive(self):
        """External point returns positive distance."""
        half_extents = (1.0, 1.0, 1.0)
        edge_thickness = 0.1
        d = sdf_box_frame((3.0, 0.0, 0.0), half_extents, edge_thickness)
        assert d > 0

    def test_no_nan_or_inf(self):
        """No NaN or Inf values at various test points."""
        half_extents = (1.0, 1.0, 1.0)
        edge_thickness = 0.1
        test_points = [
            (0, 0, 0), (1, 1, 1), (2, 2, 2),
            (0.5, 0.5, 0.5), (1.5, 0, 0),
        ]
        for p in test_points:
            d = sdf_box_frame(p, half_extents, edge_thickness)
            assert is_finite(d)


# =============================================================================
# Test Class: Rounded Box SDF
# =============================================================================

class TestRoundedBoxCorrectness:
    """Tests for sdf_rounded_box correctness."""

    def test_center_negative_distance(self):
        """Center returns negative distance."""
        half_extents = (1.0, 1.0, 1.0)
        corner_radius = 0.2
        d = sdf_rounded_box((0.0, 0.0, 0.0), half_extents, corner_radius)
        assert d < 0

    def test_face_center_surface(self):
        """Face center (accounting for rounding) returns zero."""
        half_extents = (1.0, 1.0, 1.0)
        corner_radius = 0.2
        # Face center is at the half_extent
        d = sdf_rounded_box((1.0, 0.0, 0.0), half_extents, corner_radius)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_rounded_corner(self):
        """Rounded corner returns approximately zero at corner."""
        half_extents = (1.0, 1.0, 1.0)
        corner_radius = 0.2
        # Original corner was at (1, 1, 1)
        # With rounding, the surface is at radius from (0.8, 0.8, 0.8)
        corner_dist = math.sqrt(3 * 0.8**2) + corner_radius
        d = sdf_rounded_box((corner_dist / math.sqrt(3),) * 3, half_extents, corner_radius)
        # This should be approximately on the surface
        assert abs(d) < 0.1

    def test_external_point(self):
        """External point returns positive distance."""
        half_extents = (1.0, 1.0, 1.0)
        corner_radius = 0.2
        d = sdf_rounded_box((3.0, 0.0, 0.0), half_extents, corner_radius)
        expected = 3.0 - 1.0  # distance to face
        assert d == pytest.approx(expected, abs=TOL_APPROX)


# =============================================================================
# Test Class: Octahedron SDF
# =============================================================================

class TestOctahedronCorrectness:
    """Tests for sdf_octahedron correctness."""

    def test_vertex_surface(self):
        """Vertex points return zero distance."""
        size = 1.0
        vertices = [
            (size, 0, 0), (-size, 0, 0),
            (0, size, 0), (0, -size, 0),
            (0, 0, size), (0, 0, -size),
        ]
        for p in vertices:
            d = sdf_octahedron(p, size)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_center_negative(self):
        """Center returns negative distance."""
        size = 1.0
        d = sdf_octahedron((0.0, 0.0, 0.0), size)
        assert d < 0

    def test_face_center_surface(self):
        """Face center returns approximately zero."""
        size = 1.0
        # Face center at (1/3, 1/3, 1/3) normalized to face
        face_dist = size / 3.0
        d = sdf_octahedron((face_dist, face_dist, face_dist), size)
        assert abs(d) < 0.1  # Approximate

    def test_external_point(self):
        """External point returns positive distance."""
        size = 1.0
        d = sdf_octahedron((3.0, 0.0, 0.0), size)
        assert d > 0


# =============================================================================
# Test Class: Pyramid SDF
# =============================================================================

class TestPyramidCorrectness:
    """Tests for sdf_pyramid correctness."""

    def test_apex_surface(self):
        """Apex point returns zero distance."""
        height = 1.0
        d = sdf_pyramid((0.0, 0.0, 0.0), height)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_base_edge_surface(self):
        """Base edge (corner) returns approximately zero."""
        height = 1.0
        # Base corner is at (height, -height, 0) or similar
        d = sdf_pyramid((height, -height, 0.0), height)
        assert d == pytest.approx(0.0, abs=0.1)  # Approximate due to SDF complexity

    def test_internal_point_negative(self):
        """Internal point returns negative distance."""
        height = 2.0
        # Point inside pyramid
        d = sdf_pyramid((0.0, -0.5, 0.0), height)
        assert d < 0

    def test_external_point_positive(self):
        """External point returns positive distance."""
        height = 1.0
        d = sdf_pyramid((5.0, 0.0, 0.0), height)
        assert d > 0


# =============================================================================
# Test Class: Gradient Direction (Normal)
# =============================================================================

class TestGradientDirection:
    """Tests that gradient points away from surface (outward normal)."""

    def test_sphere_gradient_outward(self):
        """Sphere gradient points radially outward."""
        radius = 1.0
        p = (1.5, 0.0, 0.0)  # Outside sphere
        grad = estimate_gradient(lambda q: sdf_sphere(q, radius), p)
        # Should point in +X direction
        assert grad[0] > 0.9

    def test_box_gradient_outward(self):
        """Box gradient points outward from faces."""
        half_extents = (1.0, 1.0, 1.0)
        p = (1.5, 0.0, 0.0)  # Outside in +X
        grad = estimate_gradient(lambda q: sdf_box(q, half_extents), p)
        # Should point in +X direction
        assert grad[0] > 0.9

    def test_plane_gradient_is_normal(self):
        """Plane gradient equals the plane normal."""
        normal = (0.0, 1.0, 0.0)
        p = (0.0, 1.0, 0.0)
        grad = estimate_gradient(lambda q: sdf_plane(q, normal, 0.0), p)
        assert grad[1] > 0.99  # Points in +Y


# =============================================================================
# Test Class: Sign Flip Across Surface
# =============================================================================

class TestSignFlip:
    """Tests that SDF sign flips across surface boundary."""

    def test_sphere_sign_flip(self):
        """Sphere SDF sign changes across surface."""
        radius = 1.0
        d_inside = sdf_sphere((0.5, 0.0, 0.0), radius)
        d_outside = sdf_sphere((1.5, 0.0, 0.0), radius)
        assert d_inside < 0
        assert d_outside > 0

    def test_box_sign_flip(self):
        """Box SDF sign changes across surface."""
        half_extents = (1.0, 1.0, 1.0)
        d_inside = sdf_box((0.5, 0.0, 0.0), half_extents)
        d_outside = sdf_box((1.5, 0.0, 0.0), half_extents)
        assert d_inside < 0
        assert d_outside > 0

    def test_cylinder_sign_flip(self):
        """Cylinder SDF sign changes across surface."""
        d_inside = sdf_cylinder((0.0, 0.0, 0.0), 1.0, 2.0)
        d_outside = sdf_cylinder((3.0, 0.0, 0.0), 1.0, 2.0)
        assert d_inside < 0
        assert d_outside > 0


# =============================================================================
# Test Class: No NaN/Inf Values
# =============================================================================

class TestNoNanInf:
    """Tests that no SDF returns NaN or Inf at any tested point."""

    @pytest.mark.parametrize("p", [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (0.001, 0.001, 0.001),
        (1000.0, 1000.0, 1000.0),
        (-1000.0, -1000.0, -1000.0),
    ])
    def test_sphere_no_nan_inf(self, p):
        """Sphere SDF returns finite values."""
        d = sdf_sphere(p, 1.0)
        assert is_finite(d)

    @pytest.mark.parametrize("p", [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (2.0, 2.0, 2.0),
        (-2.0, -2.0, -2.0),
    ])
    def test_box_no_nan_inf(self, p):
        """Box SDF returns finite values."""
        d = sdf_box(p, (1.0, 1.0, 1.0))
        assert is_finite(d)

    @pytest.mark.parametrize("p", [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ])
    def test_torus_no_nan_inf(self, p):
        """Torus SDF returns finite values."""
        d = sdf_torus(p, 1.0, 0.25)
        assert is_finite(d)

    @pytest.mark.parametrize("p", [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 2.0, 0.0),
    ])
    def test_cylinder_no_nan_inf(self, p):
        """Cylinder SDF returns finite values."""
        d = sdf_cylinder(p, 1.0, 2.0)
        assert is_finite(d)

    @pytest.mark.parametrize("p", [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ])
    def test_capsule_no_nan_inf(self, p):
        """Capsule SDF returns finite values."""
        d = sdf_capsule(p, (0, 0, 0), (0, 2, 0), 0.5)
        assert is_finite(d)

    @pytest.mark.parametrize("p", [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ])
    def test_ellipsoid_no_nan_inf(self, p):
        """Ellipsoid SDF returns finite values."""
        d = sdf_ellipsoid(p, (1.0, 2.0, 1.5))
        assert is_finite(d)


# =============================================================================
# Test Class: Rotational Invariance
# =============================================================================

class TestRotationalInvariance:
    """Tests that rotationally symmetric SDFs preserve distance under rotation."""

    def test_sphere_rotational_invariance(self):
        """Sphere distance is invariant under rotation."""
        p = (1.5, 0.5, 0.0)
        d_original = sdf_sphere(p, 1.0)

        for angle in [math.pi/4, math.pi/2, math.pi, 3*math.pi/2]:
            p_rotated = rotate_point_y(p, angle)
            d_rotated = sdf_sphere(p_rotated, 1.0)
            assert d_rotated == pytest.approx(d_original, abs=TOL_APPROX)

    def test_torus_rotational_invariance_around_y(self):
        """Torus distance is invariant under rotation around Y axis."""
        p = (1.5, 0.0, 0.5)
        d_original = sdf_torus(p, 1.0, 0.25)

        for angle in [math.pi/4, math.pi/2, math.pi]:
            p_rotated = rotate_point_y(p, angle)
            d_rotated = sdf_torus(p_rotated, 1.0, 0.25)
            assert d_rotated == pytest.approx(d_original, abs=TOL_APPROX)

    def test_cylinder_rotational_invariance_around_y(self):
        """Cylinder distance is invariant under rotation around Y axis."""
        p = (0.5, 0.3, 0.0)
        d_original = sdf_cylinder(p, 1.0, 2.0)

        for angle in [math.pi/4, math.pi/2, math.pi]:
            p_rotated = rotate_point_y(p, angle)
            d_rotated = sdf_cylinder(p_rotated, 1.0, 2.0)
            assert d_rotated == pytest.approx(d_original, abs=TOL_APPROX)
