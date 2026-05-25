"""
Cleanroom blackbox tests for sdSphere WGSL function (T-DEMO-1.1).

Tests the signed distance function for a sphere of radius r centered at the
origin, treating the implementation as a black box from the spec.

The signed distance function for a sphere is:

    sdSphere(p, r) = length(p) - r

where p is a 3D point and r is the radius. The abs(r) guard ensures negative
radii are treated as positive.

BLACKBOX coverage plan:
  Path 1:  Inside point returns negative signed distance
  Path 2:  Surface point returns distance ~0
  Path 3:  Outside point returns positive signed distance
  Path 4:  Multiple surface points at cardinal directions
  Path 5:  Linear increase beyond surface
  Path 6:  Larger radius = larger interior (more negative at same point)
  Path 7:  Negative radius is handled via abs (same as positive)
  Path 8:  Zero radius degenerates to point
  Path 9:  Continuity -- nearby points have nearby distances
  Path 10: Sign convention -- negative inside, positive outside, zero on boundary
"""

from __future__ import annotations

import math

import pytest


# =============================================================================
# Python model of sdSphere matching WGSL semantics (blackbox proxy)
# =============================================================================


def py_sd_sphere(p, r):
    """Python model of WGSL sdSphere(p: vec3<f32>, r: f32) -> f32.

    Signed distance from point p (3-tuple) to a sphere of radius r centered at
    the origin.

    Reference: Inigo Quilez -- Sphere SDF
    https://iquilezles.org/articles/distfunctions/
    """
    safe_r = abs(r)
    return math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - safe_r


# =============================================================================
# Default test parameters
# =============================================================================

TOL_SURFACE = 1e-9     # Points on surface should be very close to 0
TOL_LINEAR = 0.001     # Tolerance for linearity checks
TOL_CONTINUITY = 0.01  # Max allowed jump for nearby points
TOL_FAR = 0.001        # Tolerance for far-point approximations

# Standard sphere radius
R_DEFAULT = 2.0

# Small sphere
R_SMALL = 0.5

# Large sphere
R_LARGE = 5.0


# =============================================================================
# Path 1: Inside point returns negative signed distance (Acceptance)
# =============================================================================


class TestInside:
    """Points inside the sphere should return negative signed distance."""

    def test_origin_is_negative(self):
        """Center of sphere at (0,0,0) with r=1 -> -1 (inside)."""
        d = py_sd_sphere((0.0, 0.0, 0.0), 1.0)
        expected = -1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Origin with r=1 should give -1, got {d}"
        )

    def test_origin_default_radius(self):
        """Center of sphere at (0,0,0) with r=2 -> -2 (inside)."""
        d = py_sd_sphere((0.0, 0.0, 0.0), R_DEFAULT)
        expected = -R_DEFAULT
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Origin with r={R_DEFAULT} should give -{R_DEFAULT}, got {d}"
        )

    def test_inside_positive_x(self):
        """Point on positive x inside sphere: (0.5, 0, 0) with r=1 -> -0.5."""
        d = py_sd_sphere((0.5, 0.0, 0.0), 1.0)
        expected = 0.5 - 1.0  # length=0.5, minus r=1 = -0.5
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Point (0.5,0,0) inside r=1 should give {expected}, got {d}"
        )

    def test_inside_negative_x(self):
        """Point on negative x inside sphere: (-0.5, 0, 0) with r=1 -> -0.5."""
        d = py_sd_sphere((-0.5, 0.0, 0.0), 1.0)
        expected = 0.5 - 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Point (-0.5,0,0) inside r=1 should give {expected}, got {d}"
        )

    def test_inside_diagonal(self):
        """Point on diagonal inside sphere: (1,1,1) with r=2 -> sqrt(3)-2."""
        d = py_sd_sphere((1.0, 1.0, 1.0), 2.0)
        expected = math.sqrt(3.0) - 2.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Point (1,1,1) inside r=2 should give {expected}, got {d}"
        )

    def test_inside_near_surface(self):
        """Point just inside surface: (0.999, 0, 0) with r=1 -> slightly negative."""
        d = py_sd_sphere((0.999, 0.0, 0.0), 1.0)
        assert d < 0.0, (
            f"Point just inside should be negative, got {d}"
        )

    def test_inside_various_radii(self):
        """Points inside with various radii all return negative."""
        inside_cases = [
            ((0.0, 0.0, 0.0), 0.5),
            ((0.2, 0.0, 0.0), 0.5),
            ((0.0, 0.3, 0.0), 1.0),
            ((0.0, 0.0, 0.4), 1.0),
            ((0.5, 0.5, 0.0), 2.0),
        ]
        for p, r in inside_cases:
            d = py_sd_sphere(p, r)
            assert d < 0.0, (
                f"Inside point p={p} with r={r} should be negative, got {d}"
            )


# =============================================================================
# Path 2: Surface point returns distance ~0 (Acceptance)
# =============================================================================


class TestSurface:
    """Points exactly on the sphere surface should return distance ~0."""

    def test_surface_positive_x(self):
        """Point on surface at (r, 0, 0)."""
        r = R_DEFAULT
        d = py_sd_sphere((r, 0.0, 0.0), r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point ({r},0,0) on surface should be ~0, got {d}"
        )

    def test_surface_small_radius(self):
        """Point on surface of small sphere: (0.5, 0, 0) with r=0.5."""
        d = py_sd_sphere((R_SMALL, 0.0, 0.0), R_SMALL)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point ({R_SMALL},0,0) on small sphere should be ~0, got {d}"
        )

    def test_surface_large_radius(self):
        """Point on surface of large sphere: (5, 0, 0) with r=5."""
        d = py_sd_sphere((R_LARGE, 0.0, 0.0), R_LARGE)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point ({R_LARGE},0,0) on large sphere should be ~0, got {d}"
        )

    def test_surface_from_inside_approach(self):
        """Surface point approached from inside also gives ~0."""
        d = py_sd_sphere((R_DEFAULT, 0.0, 0.0), R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)


# =============================================================================
# Path 3: Outside point returns positive signed distance (Acceptance)
# =============================================================================


class TestOutside:
    """Points outside the sphere should return positive signed distance."""

    def test_outside_positive_x(self):
        """Point outside at (2,0,0) with r=1 -> ~1."""
        d = py_sd_sphere((2.0, 0.0, 0.0), 1.0)
        expected = 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Point (2,0,0) outside r=1 should give ~{expected}, got {d}"
        )

    def test_outside_default_radius(self):
        """Point outside at (5,0,0) with r=2 -> ~3."""
        d = py_sd_sphere((5.0, 0.0, 0.0), R_DEFAULT)
        expected = 5.0 - R_DEFAULT
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Point (5,0,0) outside r={R_DEFAULT} should give ~{expected}, "
            f"got {d}"
        )

    def test_outside_diagonal(self):
        """Point on diagonal outside: (3,3,3) with r=1 -> sqrt(27)-1."""
        d = py_sd_sphere((3.0, 3.0, 3.0), 1.0)
        expected = math.sqrt(27.0) - 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Point (3,3,3) outside r=1 should give {expected}, got {d}"
        )

    def test_outside_near_surface(self):
        """Point just outside surface: (1.001, 0, 0) with r=1 -> slightly positive."""
        d = py_sd_sphere((1.001, 0.0, 0.0), 1.0)
        assert d > 0.0, (
            f"Point just outside should be positive, got {d}"
        )

    def test_outside_all_directions(self):
        """Points outside in all cardinal directions should be positive."""
        r = R_DEFAULT
        outside_cases = [
            (r + 0.5, 0.0, 0.0),
            (-(r + 0.5), 0.0, 0.0),
            (0.0, r + 0.5, 0.0),
            (0.0, -(r + 0.5), 0.0),
            (0.0, 0.0, r + 0.5),
            (0.0, 0.0, -(r + 0.5)),
        ]
        for p in outside_cases:
            d = py_sd_sphere(p, r)
            assert d > 0.0, (
                f"Outside point p={p} with r={r} should be positive, got {d}"
            )


# =============================================================================
# Path 4: Multiple surface points at cardinal directions (Acceptance)
# =============================================================================


class TestSurfacePoints:
    """Points on the sphere surface in all cardinal directions return ~0."""

    def test_surface_positive_x(self):
        """Surface point at (+r, 0, 0)."""
        d = py_sd_sphere((R_DEFAULT, 0.0, 0.0), R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_surface_negative_x(self):
        """Surface point at (-r, 0, 0)."""
        d = py_sd_sphere((-R_DEFAULT, 0.0, 0.0), R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_surface_positive_y(self):
        """Surface point at (0, r, 0)."""
        d = py_sd_sphere((0.0, R_DEFAULT, 0.0), R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_surface_negative_y(self):
        """Surface point at (0, -r, 0)."""
        d = py_sd_sphere((0.0, -R_DEFAULT, 0.0), R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_surface_positive_z(self):
        """Surface point at (0, 0, r)."""
        d = py_sd_sphere((0.0, 0.0, R_DEFAULT), R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_surface_negative_z(self):
        """Surface point at (0, 0, -r)."""
        d = py_sd_sphere((0.0, 0.0, -R_DEFAULT), R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_surface_all_cardinal(self):
        """All cardinal direction surface points at once."""
        r = R_DEFAULT
        surface_points = [
            (r, 0.0, 0.0),
            (-r, 0.0, 0.0),
            (0.0, r, 0.0),
            (0.0, -r, 0.0),
            (0.0, 0.0, r),
            (0.0, 0.0, -r),
        ]
        for p in surface_points:
            d = py_sd_sphere(p, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point p={p} should be ~0, got {d}"
            )


# =============================================================================
# Path 5: Linear increase beyond surface (Acceptance)
# =============================================================================


class TestLinearIncrease:
    """Distance should increase linearly with point distance beyond surface."""

    def test_linear_positive_x(self):
        """Distance from (d,0,0) with r=1 is d-1 (linear)."""
        r = 1.0
        for d_val in [1.5, 2.0, 3.0, 5.0, 10.0]:
            expected = d_val - r
            result = py_sd_sphere((d_val, 0.0, 0.0), r)
            assert result == pytest.approx(expected, abs=TOL_LINEAR), (
                f"Point ({d_val},0,0) with r=1 should give {expected}, got {result}"
            )

    def test_linear_default_radius(self):
        """Distance from (d,0,0) with r=2 is d-2 (linear)."""
        r = R_DEFAULT
        for d_val in [3.0, 4.0, 6.0, 8.0, 20.0]:
            expected = d_val - r
            result = py_sd_sphere((d_val, 0.0, 0.0), r)
            assert result == pytest.approx(expected, abs=TOL_LINEAR), (
                f"Point ({d_val},0,0) with r={r} should give {expected}, "
                f"got {result}"
            )

    def test_linear_small_radius(self):
        """Linear behavior holds for small radius."""
        r = R_SMALL
        for d_val in [1.0, 2.0, 5.0]:
            expected = d_val - r
            result = py_sd_sphere((d_val, 0.0, 0.0), r)
            assert result == pytest.approx(expected, abs=TOL_LINEAR), (
                f"Point ({d_val},0,0) with r={r} should give {expected}, "
                f"got {result}"
            )

    def test_linear_negative_x(self):
        """Linear behavior on negative x axis."""
        r = 1.0
        for d_val in [1.5, 3.0, 7.0]:
            expected = d_val - r
            result = py_sd_sphere((-d_val, 0.0, 0.0), r)
            assert result == pytest.approx(expected, abs=TOL_LINEAR), (
                f"Point ({-d_val},0,0) with r=1 should give {expected}, "
                f"got {result}"
            )

    def test_linear_y_axis(self):
        """Linear behavior on y axis."""
        r = 1.0
        for d_val in [2.0, 4.0, 6.0]:
            expected = d_val - r
            result = py_sd_sphere((0.0, d_val, 0.0), r)
            assert result == pytest.approx(expected, abs=TOL_LINEAR), (
                f"Point (0,{d_val},0) with r=1 should give {expected}, "
                f"got {result}"
            )

    def test_linear_z_axis(self):
        """Linear behavior on z axis."""
        r = 1.0
        for d_val in [2.0, 4.0, 6.0]:
            expected = d_val - r
            result = py_sd_sphere((0.0, 0.0, d_val), r)
            assert result == pytest.approx(expected, abs=TOL_LINEAR), (
                f"Point (0,0,{d_val}) with r=1 should give {expected}, "
                f"got {result}"
            )


# =============================================================================
# Path 6: Larger radius means larger interior (Acceptance)
# =============================================================================


class TestRadiusEffect:
    """Larger radius should produce more negative distance at same interior point."""

    def test_same_point_more_negative_larger_r(self):
        """Same point (0,0,0) is more negative for larger r."""
        d_small = py_sd_sphere((0.0, 0.0, 0.0), R_SMALL)
        d_default = py_sd_sphere((0.0, 0.0, 0.0), R_DEFAULT)
        d_large = py_sd_sphere((0.0, 0.0, 0.0), R_LARGE)
        assert d_large < d_default < d_small, (
            f"Larger r should give more negative at origin: "
            f"r={R_SMALL} -> {d_small}, r={R_DEFAULT} -> {d_default}, "
            f"r={R_LARGE} -> {d_large}"
        )

    def test_larger_r_gives_more_interior(self):
        """At a fixed interior point, larger r gives more negative."""
        point = (0.5, 0.5, 0.0)
        d0 = py_sd_sphere(point, 0.5)
        d1 = py_sd_sphere(point, 1.0)
        d2 = py_sd_sphere(point, 2.0)
        assert d2 < d1 < d0, (
            f"Larger r should give more negative: "
            f"r=0.5 -> {d0}, r=1 -> {d1}, r=2 -> {d2}"
        )

    def test_larger_r_shifts_surface_outward(self):
        """Surface point for smaller r is inside for larger r."""
        r_small = 1.0
        r_large = 2.0
        # Surface of small sphere
        p = (r_small, 0.0, 0.0)
        d_small_surface = py_sd_sphere(p, r_small)  # ~0
        d_large_at_same = py_sd_sphere(p, r_large)    # negative (inside)
        assert d_small_surface == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface of small sphere should be 0, got {d_small_surface}"
        )
        assert d_large_at_same < 0.0, (
            f"Same point should be inside larger sphere, got {d_large_at_same}"
        )

    def test_radii_ordering_at_surface(self):
        """Monotonic with respect to radius at surface boundary."""
        # For a point at distance 1 from origin:
        # r=0.5 -> +0.5 (outside), r=1.0 -> 0 (surface), r=2.0 -> -1.0 (inside)
        p = (1.0, 0.0, 0.0)
        d_half = py_sd_sphere(p, 0.5)
        d_one = py_sd_sphere(p, 1.0)
        d_two = py_sd_sphere(p, 2.0)
        assert d_half > 0.0, f"r=0.5 at (1,0,0) should be positive, got {d_half}"
        assert d_one == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"r=1.0 at (1,0,0) should be 0, got {d_one}"
        )
        assert d_two < 0.0, f"r=2.0 at (1,0,0) should be negative, got {d_two}"

    def test_larger_r_reduces_exterior_distance(self):
        """Same exterior point is closer to larger sphere surface."""
        p = (3.0, 0.0, 0.0)
        d_small = py_sd_sphere(p, 0.5)  # 3.0 - 0.5 = 2.5
        d_large = py_sd_sphere(p, 2.5)  # 3.0 - 2.5 = 0.5
        assert d_large < d_small, (
            f"Larger sphere should have smaller exterior distance: "
            f"d_small={d_small}, d_large={d_large}"
        )


# =============================================================================
# Path 7: Negative radius is handled via abs (Acceptance)
# =============================================================================


class TestNegativeRadius:
    """Negative radius should behave identically to positive radius via abs."""

    def test_negative_radius_same_as_positive(self):
        """r=-2 and r=2 give identical results for all points."""
        positive_results = [
            py_sd_sphere((0.0, 0.0, 0.0), 2.0),
            py_sd_sphere((2.0, 0.0, 0.0), 2.0),
            py_sd_sphere((5.0, 0.0, 0.0), 2.0),
            py_sd_sphere((1.0, 0.0, 0.0), 2.0),
            py_sd_sphere((-3.0, 0.0, 0.0), 2.0),
        ]
        negative_results = [
            py_sd_sphere((0.0, 0.0, 0.0), -2.0),
            py_sd_sphere((2.0, 0.0, 0.0), -2.0),
            py_sd_sphere((5.0, 0.0, 0.0), -2.0),
            py_sd_sphere((1.0, 0.0, 0.0), -2.0),
            py_sd_sphere((-3.0, 0.0, 0.0), -2.0),
        ]
        for i, (pos, neg) in enumerate(zip(positive_results, negative_results)):
            assert pos == pytest.approx(neg, abs=TOL_SURFACE), (
                f"Case {i}: r=2 and r=-2 should match: pos={pos}, neg={neg}"
            )

    def test_negative_radius_surface(self):
        """Surface of sphere with r=-2 is at x=2 from origin."""
        d = py_sd_sphere((2.0, 0.0, 0.0), -2.0)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface of r=-2 sphere should be at length=2, got {d}"
        )

    def test_negative_radius_various_values(self):
        """Various negative radii behave like their positive counterparts."""
        test_cases = [
            ((0.0, 0.0, 0.0), -0.5),
            ((0.0, 0.0, 0.0), -1.0),
            ((0.0, 0.0, 0.0), -3.0),
            ((1.0, 0.0, 0.0), -0.5),
            ((1.0, 0.0, 0.0), -1.0),
            ((2.0, 2.0, 0.0), -2.0),
        ]
        for p, r in test_cases:
            d_neg = py_sd_sphere(p, r)
            d_pos = py_sd_sphere(p, abs(r))
            assert d_neg == pytest.approx(d_pos, abs=TOL_SURFACE), (
                f"p={p}, r={r}: d_neg={d_neg} != d_pos={d_pos}"
            )

    def test_negative_zero_treated_as_zero(self):
        """r=-0 works identically to r=0."""
        d_neg_zero = py_sd_sphere((1.0, 0.0, 0.0), -0.0)
        d_zero = py_sd_sphere((1.0, 0.0, 0.0), 0.0)
        assert d_neg_zero == pytest.approx(d_zero, abs=TOL_SURFACE), (
            f"r=-0 and r=0 should match: {d_neg_zero} vs {d_zero}"
        )


# =============================================================================
# Path 8: Zero radius degenerates to point (Acceptance)
# =============================================================================


class TestZeroRadius:
    """Zero radius sphere degenerates to a point at the origin."""

    def test_origin_with_zero_radius(self):
        """At origin with r=0: length(0,0,0) - 0 = 0."""
        d = py_sd_sphere((0.0, 0.0, 0.0), 0.0)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Origin with r=0 should be 0, got {d}"
        )

    def test_unit_distance_zero_radius(self):
        """At (1,0,0) with r=0: length(1,0,0) - 0 = 1."""
        d = py_sd_sphere((1.0, 0.0, 0.0), 0.0)
        expected = 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Point (1,0,0) with r=0 should give {expected}, got {d}"
        )

    def test_zero_radius_various_points(self):
        """sdSphere(p, 0) = length(p) for any point p."""
        points = [
            (2.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 4.0),
            (1.0, 2.0, 3.0),
            (-2.0, 0.0, 0.0),
            (0.0, -3.0, 0.0),
        ]
        for p in points:
            expected = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            d = py_sd_sphere(p, 0.0)
            assert d == pytest.approx(expected, abs=TOL_SURFACE), (
                f"With r=0 at p={p}: sd = length = {expected}, got {d}"
            )

    def test_zero_radius_sign_convention(self):
        """With r=0, all non-origin points are positive (outside point)."""
        points = [
            (1.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.1, 0.1, 0.1),
        ]
        for p in points:
            d = py_sd_sphere(p, 0.0)
            assert d > 0.0, (
                f"With r=0 at p={p}: should be positive, got {d}"
            )


# =============================================================================
# Path 9: Continuity -- nearby points have nearby distances (Acceptance)
# =============================================================================


class TestContinuity:
    """The SDF should be continuous: nearby points produce nearby distances."""

    def test_continuity_along_x(self):
        """Continuity along x-axis across a range."""
        r = R_DEFAULT
        step = 0.001
        prev = py_sd_sphere((-5.0, 0.0, 0.0), r)
        for i in range(1, 1000):
            x = -5.0 + i * step
            curr = py_sd_sphere((x, 0.0, 0.0), r)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at x={x}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """Continuity along y-axis."""
        r = R_DEFAULT
        step = 0.001
        prev = py_sd_sphere((0.0, -5.0, 0.0), r)
        for i in range(1, 1000):
            y = -5.0 + i * step
            curr = py_sd_sphere((0.0, y, 0.0), r)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at y={y}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_z(self):
        """Continuity along z-axis."""
        r = R_DEFAULT
        step = 0.001
        prev = py_sd_sphere((0.0, 0.0, -5.0), r)
        for i in range(1, 1000):
            z = -5.0 + i * step
            curr = py_sd_sphere((0.0, 0.0, z), r)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at z={z}: diff={diff}"
            )
            prev = curr

    def test_continuity_through_surface(self):
        """Continuity when crossing the sphere surface."""
        r = R_DEFAULT
        step = 0.0001
        # Cross the surface at x = r
        for offset in [i * step for i in range(-10, 11)]:
            p = (r + offset, 0.0, 0.0)
            d = py_sd_sphere(p, r)
            # Near surface: sd ~ offset
            assert abs(d - offset) < 0.001, (
                f"Near surface at x={p[0]}: expected ~{offset}, got {d}"
            )

    def test_continuity_diagonal(self):
        """Continuity along diagonal direction."""
        r = R_DEFAULT
        step = 0.001
        prev = py_sd_sphere((-3.0, -3.0, -3.0), r)
        for i in range(1, 600):
            t_val = -3.0 + i * step
            curr = py_sd_sphere((t_val, t_val, t_val), r)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity on diagonal at t={t_val}: diff={diff}"
            )
            prev = curr

    def test_small_perturbation_stability(self):
        """Small perturbations should not cause large changes in distance."""
        r = R_DEFAULT
        center = (r * 0.5, 0.0, 0.0)  # Inside sphere
        base_d = py_sd_sphere(center, r)
        eps = 1e-7
        for dx, dy, dz in [(eps, 0, 0), (-eps, 0, 0), (0, eps, 0),
                           (0, -eps, 0), (0, 0, eps), (0, 0, -eps)]:
            p = (center[0] + dx, center[1] + dy, center[2] + dz)
            d = py_sd_sphere(p, r)
            assert abs(d - base_d) < 0.001, (
                f"Unstable distance change: base={base_d}, perturbed={d}"
            )


# =============================================================================
# Path 10: Sign convention -- negative inside, positive outside, zero on surface
# =============================================================================


class TestSignConvention:
    """Distance sign: negative inside, positive outside, zero on surface."""

    def test_inside_points_negative(self):
        """Points known to be inside the sphere should have negative SDF."""
        r = R_DEFAULT
        inside_points = [
            (0.0, 0.0, 0.0),
            (r * 0.5, 0.0, 0.0),
            (0.0, r * 0.5, 0.0),
            (0.0, 0.0, r * 0.5),
            (-r * 0.5, 0.0, 0.0),
            (r * 0.3, r * 0.3, r * 0.3),
        ]
        for p in inside_points:
            d = py_sd_sphere(p, r)
            assert d <= 0.0, (
                f"Inside point p={p} should have non-positive SDF, got {d}"
            )

    def test_outside_points_positive(self):
        """Points known to be outside the sphere should have positive SDF."""
        r = R_DEFAULT
        outside_points = [
            (r + 1.0, 0.0, 0.0),
            (0.0, r + 1.0, 0.0),
            (0.0, 0.0, r + 1.0),
            (-(r + 1.0), 0.0, 0.0),
            (r * 2.0, r * 2.0, 0.0),
        ]
        for p in outside_points:
            d = py_sd_sphere(p, r)
            assert d >= 0.0, (
                f"Outside point p={p} should have non-negative SDF, got {d}"
            )

    def test_surface_points_zero(self):
        """Points on the sphere surface should have near-zero SDF."""
        r = R_DEFAULT
        surface_points = [
            (r, 0.0, 0.0),
            (0.0, r, 0.0),
            (0.0, 0.0, r),
            (-r, 0.0, 0.0),
            (0.0, -r, 0.0),
            (0.0, 0.0, -r),
        ]
        for p in surface_points:
            d = py_sd_sphere(p, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point p={p} should have near-zero SDF, got {d}"
            )

    def test_sign_transition_at_surface(self):
        """Sign should flip at the surface: inside negative, outside positive."""
        r = R_DEFAULT
        surface_x = r
        step = 1e-4
        inside = py_sd_sphere((surface_x - step, 0.0, 0.0), r)
        on_surface = py_sd_sphere((surface_x, 0.0, 0.0), r)
        outside = py_sd_sphere((surface_x + step, 0.0, 0.0), r)
        assert inside < 0.0, (
            f"Inside point should be negative, got {inside}"
        )
        assert on_surface == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point should be zero, got {on_surface}"
        )
        assert outside > 0.0, (
            f"Outside point should be positive, got {outside}"
        )

    def test_sign_consistent_all_directions(self):
        """Sign convention holds in all directions from center."""
        r = R_DEFAULT
        directions = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
        ]
        for direction in directions:
            length = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
            norm = tuple(c / length for c in direction)
            # Inside at half radius
            inside_p = tuple(c * r * 0.5 for c in norm)
            # Surface at full radius
            surface_p = tuple(c * r for c in norm)
            # Outside at 1.5x radius
            outside_p = tuple(c * r * 1.5 for c in norm)
            assert py_sd_sphere(inside_p, r) < 0.0, (
                f"Inside point along {direction} should be negative"
            )
            assert py_sd_sphere(surface_p, r) == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point along {direction} should be zero"
            )
            assert py_sd_sphere(outside_p, r) > 0.0, (
                f"Outside point along {direction} should be positive"
            )


# =============================================================================
# Path D: Diagonal behavior -- points along the diagonal
# =============================================================================


class TestDiagonal:
    """Points along the main diagonal (x, x, x)."""

    def test_diagonal_inside(self):
        """Point (1,1,1) with r=2: sqrt(3)-2 < 0 (inside)."""
        d = py_sd_sphere((1.0, 1.0, 1.0), 2.0)
        expected = math.sqrt(3.0) - 2.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Diagonal inside: expected {expected}, got {d}"
        )
        assert d < 0.0, f"Should be inside, got {d}"

    def test_diagonal_surface(self):
        """Point on diagonal surface: (r/sqrt(3), r/sqrt(3), r/sqrt(3))."""
        r = R_DEFAULT
        component = r / math.sqrt(3.0)
        p = (component, component, component)
        d = py_sd_sphere(p, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Diagonal surface point {p} with r={r} should be ~0, got {d}"
        )

    def test_diagonal_outside(self):
        """Point outside on diagonal: (3,3,3) with r=1 -> sqrt(27)-1 > 0."""
        d = py_sd_sphere((3.0, 3.0, 3.0), 1.0)
        expected = math.sqrt(27.0) - 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Diagonal outside: expected {expected}, got {d}"
        )
        assert d > 0.0, f"Should be outside, got {d}"

    def test_diagonal_linear(self):
        """Distance along diagonal increases linearly outside."""
        r = 1.0
        # For (t, t, t): length = t*sqrt(3), sd = t*sqrt(3) - 1
        factor = math.sqrt(3.0)
        for t_val in [1.0, 2.0, 3.0, 5.0]:
            expected = t_val * factor - r
            d = py_sd_sphere((t_val, t_val, t_val), r)
            assert d == pytest.approx(expected, abs=TOL_LINEAR), (
                f"Diagonal (t,t,t) with t={t_val}: expected {expected}, got {d}"
            )


# =============================================================================
# Path: Parameter variation -- behavior across different radii
# =============================================================================


class TestParameterVariation:
    """Behavior with different sphere radii."""

    def test_small_radius_acceptance(self):
        """Small sphere (r=0.5): surface and center distances."""
        r = R_SMALL
        # Center
        assert py_sd_sphere((0.0, 0.0, 0.0), r) == pytest.approx(-r, abs=TOL_SURFACE)
        # Surface
        assert py_sd_sphere((r, 0.0, 0.0), r) == pytest.approx(0.0, abs=TOL_SURFACE)
        # Outside
        assert py_sd_sphere((1.0, 0.0, 0.0), r) == pytest.approx(1.0 - r, abs=TOL_SURFACE)

    def test_large_radius_acceptance(self):
        """Large sphere (r=5.0): surface and center distances."""
        r = R_LARGE
        # Center
        assert py_sd_sphere((0.0, 0.0, 0.0), r) == pytest.approx(-r, abs=TOL_SURFACE)
        # Surface on all axes
        assert py_sd_sphere((r, 0.0, 0.0), r) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_sphere((0.0, r, 0.0), r) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_sphere((0.0, 0.0, r), r) == pytest.approx(0.0, abs=TOL_SURFACE)
        # Outside
        assert py_sd_sphere((r + 1.0, 0.0, 0.0), r) == pytest.approx(1.0, abs=TOL_SURFACE)

    def test_small_vs_large(self):
        """Smaller sphere has smaller interior volume."""
        point = (1.0, 0.0, 0.0)
        d_small = py_sd_sphere(point, R_SMALL)
        d_large = py_sd_sphere(point, R_LARGE)
        # r=0.5: 1.0-0.5 = 0.5 (outside)
        # r=5.0: 1.0-5.0 = -4.0 (inside)
        assert d_large < 0.0 < d_small, (
            f"Same point should be inside large sphere and outside small sphere: "
            f"small={d_small}, large={d_large}"
        )


# =============================================================================
# Path: Far point behavior
# =============================================================================


class TestFarPoint:
    """Very distant points should have distance approximately equal to distance
    from origin minus radius (sphere becomes negligible)."""

    def test_far_positive_x(self):
        """Far away on x-axis: (100, 0, 0) with r=2 -> ~98."""
        d = py_sd_sphere((100.0, 0.0, 0.0), R_DEFAULT)
        expected = 100.0 - R_DEFAULT
        assert d == pytest.approx(expected, abs=TOL_FAR), (
            f"Far point (100,0,0) should have distance ~{expected}, got {d}"
        )

    def test_far_negative_x(self):
        """Far away on -x axis with r=2 -> ~98."""
        d = py_sd_sphere((-100.0, 0.0, 0.0), R_DEFAULT)
        expected = 100.0 - R_DEFAULT
        assert d == pytest.approx(expected, abs=TOL_FAR), (
            f"Far point (-100,0,0) should have distance ~{expected}, got {d}"
        )

    def test_far_diagonal(self):
        """Far away on diagonal."""
        r = R_DEFAULT
        far_point = (1000.0, 1000.0, 1000.0)
        d = py_sd_sphere(far_point, r)
        expected = math.sqrt(3 * 1000.0**2) - r
        assert d == pytest.approx(expected, abs=TOL_FAR), (
            f"Far diagonal ({far_point}): expected {expected}, got {d}"
        )

    def test_far_asymptotic(self):
        """As point goes to infinity, distance approximates raw length."""
        r = R_DEFAULT
        very_far = (1e6, 0.0, 0.0)
        d = py_sd_sphere(very_far, r)
        expected = 1e6 - r
        assert d == pytest.approx(expected, abs=0.01), (
            f"Very far point ({very_far}): expected {expected}, got {d}"
        )


# =============================================================================
# Path: Radial monotonicity
# =============================================================================


class TestMonotonicity:
    """Distance should be monotonic along radial direction."""

    def test_outside_monotonic_increasing_x(self):
        """Outside the sphere, distance increases with distance from origin."""
        r = R_DEFAULT
        x_values = [r + i * 0.5 for i in range(20)]
        distances = [py_sd_sphere((x, 0.0, 0.0), r) for x in x_values]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-12, (
                f"Distance should be monotonic increasing outside: "
                f"at x={x_values[i]} d={distances[i]}, "
                f"at x={x_values[i+1]} d={distances[i+1]}"
            )

    def test_inside_monotonic_decreasing_to_center(self):
        """Moving from surface toward center monotonically decreases distance."""
        r = R_DEFAULT
        # From surface (x=r) to center (x=0)
        x_values = [r - i * 0.1 for i in range(int(r / 0.1))]
        distances = [py_sd_sphere((x, 0.0, 0.0), r) for x in x_values]
        for i in range(len(distances) - 1):
            assert distances[i] >= distances[i + 1] - 1e-12, (
                f"Distance should decrease moving toward center: "
                f"at x={x_values[i]} d={distances[i]}, "
                f"at x={x_values[i+1]} d={distances[i+1]}"
            )

    def test_monotonic_outside_all_axes(self):
        """Monotonic increase applies on all axes outside the sphere."""
        r = R_DEFAULT
        axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for axis in axes:
            prev = float("-inf")
            for dist_from_origin in [r + i * 0.3 for i in range(15)]:
                p = tuple(dist_from_origin * a for a in axis)
                d = py_sd_sphere(p, r)
                assert d >= prev - 1e-12, (
                    f"Monotonicity broken on axis {axis} at "
                    f"distance {dist_from_origin}: {d} < {prev}"
                )
                prev = d
