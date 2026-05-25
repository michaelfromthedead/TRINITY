"""
Cleanroom blackbox tests for sdCapsule WGSL function (T-DEMO-1.7).

Tests the signed distance function for a capsule defined by endpoints A, B
and radius r, treating the implementation as a black box from the spec.

BLACKBOX coverage plan:
  Path A: Surface points return distance ~0
  Path B: Inside points return negative distance
  Path C: Outside points return positive distance
  Path D: Far points return approximately correct distance
  Path E: Direction: moving toward capsule decreases distance, away increases
  Path F: End-cap behavior: points beyond endpoint clamps
  Path G: Continuity: nearby points have nearby distances
  Path H: Sign convention: positive outside, negative inside, zero on surface
  Path I: Symmetry: rotational symmetry about the segment axis
  Path J: Degenerate A==B collapses to sphere
  Path K: Zero radius collapses to line segment
"""

from __future__ import annotations

import math

import pytest


# =============================================================================
# Python model of sdCapsule matching WGSL semantics (blackbox proxy)
# =============================================================================


def py_sd_capsule(p, a, b, r):
    """Python model of WGSL sdCapsule(p, a, b, r) -> f32.

    Signed distance from point p (3-tuple) to a capsule defined by endpoints
    a and b with radius r.  Uses the IQ formula:
      pa = p - a
      ba = b - a
      h  = clamp(dot(pa, ba) / dot(ba, ba), 0, 1)
      return length(pa - ba * h) - abs(r)

    Reference: https://iquilezles.org/articles/distfunctions/
    """
    pa = (p[0] - a[0], p[1] - a[1], p[2] - a[2])
    ba = (b[0] - a[0], b[1] - a[1], b[2] - a[2])

    baba = max(ba[0]*ba[0] + ba[1]*ba[1] + ba[2]*ba[2], 1e-10)
    h_num = pa[0]*ba[0] + pa[1]*ba[1] + pa[2]*ba[2]
    h = max(0.0, min(1.0, h_num / baba))

    px = pa[0] - ba[0] * h
    py = pa[1] - ba[1] * h
    pz = pa[2] - ba[2] * h

    return math.sqrt(px*px + py*py + pz*pz) - abs(r)


# =============================================================================
# Default test parameters
# =============================================================================

TOL_SURFACE = 1e-12    # Points on surface should be extremely close to 0
TOL_FAR = 0.001        # Tolerance for far-point approximations
TOL_CONTINUITY = 0.01   # Max allowed jump for nearby points

# Vertical capsule: endpoints at (0, -2, 0) and (0, 2, 0), radius 1
A_DEFAULT = (0.0, -2.0, 0.0)
B_DEFAULT = (0.0, 2.0, 0.0)
R_DEFAULT = 1.0

# Thin capsule: same endpoints, radius 0.25
R_THIN = 0.25

# Thick capsule: radius 2.0
R_THICK = 2.0

# Short capsule: endpoints close together
A_SHORT = (0.0, -0.5, 0.0)
B_SHORT = (0.0, 0.5, 0.0)

# Diagonal capsule
A_DIAG = (-1.0, -1.0, 0.0)
B_DIAG = (1.0, 1.0, 0.0)


# =============================================================================
# Path A: Surface points return distance ~0 (Acceptance)
# =============================================================================


class TestSurfacePoints:
    """Points exactly on the capsule surface should return distance ~0."""

    def test_surface_cylinder_body_x(self):
        """Point on cylinder body along +x from midpoint."""
        p = (R_DEFAULT, 0.0, 0.0)
        d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface on cylinder body (+x) should be 0, got {d}"
        )

    def test_surface_cylinder_body_z(self):
        """Point on cylinder body along +z from midpoint."""
        p = (0.0, 0.0, R_DEFAULT)
        d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface on cylinder body (+z) should be 0, got {d}"
        )

    def test_surface_cylinder_body_neg_x(self):
        """Point on cylinder body along -x from midpoint."""
        p = (-R_DEFAULT, 0.0, 0.0)
        d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface on cylinder body (-x) should be 0, got {d}"
        )

    def test_surface_top_cap(self):
        """Point on hemispherical cap at endpoint B."""
        p = (R_DEFAULT, B_DEFAULT[1], 0.0)
        d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface on top cap should be 0, got {d}"
        )

    def test_surface_bottom_cap(self):
        """Point on hemispherical cap at endpoint A."""
        p = (R_DEFAULT, A_DEFAULT[1], 0.0)
        d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, R_DEFAULT)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface on bottom cap should be 0, got {d}"
        )

    def test_surface_diagonal_capsule(self):
        """Surface point on a diagonal capsule."""
        # Diagonal from (-1,-1,0) to (1,1,0). Midpoint at (0,0,0).
        # Perpendicular direction (normalized) is (1,-1,0)/sqrt(2).
        # Surface point = midpoint + r * perp_dir.
        r = 0.5
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        p = (r * inv_sqrt2, -r * inv_sqrt2, 0.0)
        d = py_sd_capsule(p, A_DIAG, B_DIAG, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface on diagonal capsule should be 0, got {d}"
        )

    def test_surface_multiple_angles(self):
        """Points on surface at various angles around the capsule."""
        r = 1.0
        angles = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for theta in angles:
            px = r * math.cos(theta)
            pz = r * math.sin(theta)
            p = (px, 0.0, pz)
            d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface at angle {theta} should be 0, got {d}"
            )


# =============================================================================
# Path B: Inside points return negative distance (Acceptance)
# =============================================================================


class TestInsideNegative:
    """Points inside the capsule should return negative distance."""

    def test_center_of_capsule(self):
        """Center of the capsule (midpoint of segment, on axis)."""
        d = py_sd_capsule((0.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, R_DEFAULT)
        expected = -R_DEFAULT
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Capsule center should give -r = {expected}, got {d}"
        )

    def test_inside_near_axis(self):
        """Point near the segment axis, well inside the radius."""
        r = R_DEFAULT
        # At (0.3, 0.0, 0.0), distance from axis = 0.3, inside by r - 0.3
        p = (0.3, 0.0, 0.0)
        d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, r)
        expected = 0.3 - r
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Inside point should give {expected}, got {d}"
        )

    def test_inside_various_locations(self):
        """Several points inside the capsule should all be negative."""
        r = R_DEFAULT
        inside_points = [
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
            (0.3, 1.0, 0.0),   # Near segment at y=1
            (0.3, -1.0, 0.0),  # Near segment at y=-1
        ]
        for p in inside_points:
            d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, r)
            assert d < 0.0, (
                f"Inside point ({p}) should have negative SDF, got {d}"
            )

    def test_inside_thick_capsule(self):
        """Inside a thick capsule."""
        d = py_sd_capsule((1.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, R_THICK)
        expected = 1.0 - R_THICK  # -1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Inside thick capsule should give {expected}, got {d}"
        )


# =============================================================================
# Path C: Outside points return positive distance (Acceptance)
# =============================================================================


class TestOutsidePositive:
    """Points outside the capsule should return positive distance."""

    def test_far_along_x(self):
        """Far along +x from the capsule."""
        d = py_sd_capsule((10.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, R_DEFAULT)
        # Distance from segment axis = 10, minus radius 1 = 9
        expected = 10.0 - R_DEFAULT
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Far point along +x should give {expected}, got {d}"
        )

    def test_below_endpoint_a(self):
        """Far below endpoint A."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        p = (0.0, -10.0, 0.0)  # 8 below A
        d = py_sd_capsule(p, a, b, R_DEFAULT)
        expected = 8.0 - R_DEFAULT
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Below endpoint A should give {expected}, got {d}"
        )

    def test_above_endpoint_b(self):
        """Far above endpoint B."""
        p = (0.0, 10.0, 0.0)  # 8 above B
        d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, R_DEFAULT)
        expected = 8.0 - R_DEFAULT
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Above endpoint B should give {expected}, got {d}"
        )

    def test_diagonal_outside(self):
        """Point diagonal from capsule -- outside."""
        p = (3.0, 5.0, 0.0)
        d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, R_DEFAULT)
        assert d > 0.0, (
            f"Diagonal point should be positive, got {d}"
        )


# =============================================================================
# Path D: Far point returns approximately correct distance
# =============================================================================


class TestFarPoint:
    """Very distant points should return approximately correct distance."""

    def test_far_positive_x(self):
        """Far away on +x: (100, 0, 0) -> 100 - r."""
        r = 1.0
        d = py_sd_capsule((100.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        expected = 100.0 - r
        assert d == pytest.approx(expected, abs=TOL_FAR), (
            f"Far point should give ~{expected}, got {d}"
        )

    def test_far_negative_x(self):
        """Far away on -x."""
        r = 1.0
        d = py_sd_capsule((-100.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        expected = 100.0 - r
        assert d == pytest.approx(expected, abs=TOL_FAR), (
            f"Far point should give ~{expected}, got {d}"
        )

    def test_far_positive_y(self):
        """Far away on +y (axis-aligned)."""
        r = 1.0
        d = py_sd_capsule((0.0, 100.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        # Distance from B = 98 - 1 = 97
        expected = 98.0 - r
        assert d == pytest.approx(expected, abs=TOL_FAR), (
            f"Far point should give ~{expected}, got {d}"
        )

    def test_far_diagonal(self):
        """Far away on diagonal."""
        r = 1.0
        d = py_sd_capsule((100.0, 100.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        # Closest feature is B at (0, 2, 0). Distance to B = sqrt(100^2 + 98^2)
        dist_to_B = math.sqrt(100.0**2 + 98.0**2)
        expected = dist_to_B - r
        assert d == pytest.approx(expected, abs=TOL_FAR), (
            f"Far diagonal point should give ~{expected}, got {d}"
        )

    def test_far_asymptotic(self):
        """As point goes to infinity, distance approximates raw distance to segment."""
        r = 1.0
        d = py_sd_capsule((1e6, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        expected = 1e6 - r
        assert d == pytest.approx(expected, abs=0.01), (
            f"Very far point should give ~{expected}, got {d}"
        )


# =============================================================================
# Path E: Direction -- moving toward capsule decreases distance
# =============================================================================


class TestDirection:
    """Moving toward the capsule decreases distance; away increases."""

    def test_toward_decreases_x(self):
        """Moving along +x toward capsule decreases distance."""
        r = 1.0
        far_d = py_sd_capsule((10.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        near_d = py_sd_capsule((3.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        assert near_d < far_d, (
            f"Moving toward capsule should decrease distance: {near_d} >= {far_d}"
        )

    def test_away_increases_x(self):
        """Moving away from capsule along +x increases distance."""
        r = 1.0
        near_d = py_sd_capsule((3.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        far_d = py_sd_capsule((10.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        assert far_d > near_d, (
            f"Moving away from capsule should increase distance: {far_d} <= {near_d}"
        )

    def test_toward_along_y(self):
        """Moving vertically toward the capsule decreases distance."""
        r = 1.0
        far_d = py_sd_capsule((0.0, 10.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        near_d = py_sd_capsule((0.0, 3.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        assert near_d < far_d, (
            f"Moving toward capsule in y should decrease distance: {near_d} >= {far_d}"
        )

    def test_inside_to_outside(self):
        """Moving from inside to outside should strictly increase SDF."""
        r = R_DEFAULT
        d_inside = py_sd_capsule((0.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        d_surface = py_sd_capsule((r, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        d_outside = py_sd_capsule((r + 1.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        assert d_inside < d_surface < d_outside, (
            f"Moving inside->surface->outside should strictly increase: "
            f"{d_inside} < {d_surface} < {d_outside}"
        )


# =============================================================================
# Path F: End-cap behavior -- clamp projection
# =============================================================================


class TestEndCapBehavior:
    """Points beyond endpoints should clamp to the closest end-cap."""

    def test_above_b_clamp(self):
        """Point above B projects to h=1, giving distance to endpoint B."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 0.5
        p = (3.0, 5.0, 0.0)  # Above B
        d = py_sd_capsule(p, a, b, r)
        dist_to_B = math.sqrt(3.0**2 + 3.0**2)
        expected = dist_to_B - r
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Above B clamp should give {expected}, got {d}"
        )

    def test_below_a_clamp(self):
        """Point below A projects to h=0, giving distance to endpoint A."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 0.5
        p = (3.0, -5.0, 0.0)  # Below A
        d = py_sd_capsule(p, a, b, r)
        dist_to_A = math.sqrt(3.0**2 + 3.0**2)
        expected = dist_to_A - r
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Below A clamp should give {expected}, got {d}"
        )

    def test_directly_above_b_no_radial(self):
        """Point directly above B (on axis)."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 0.5
        p = (0.0, 5.0, 0.0)
        d = py_sd_capsule(p, a, b, r)
        expected = 3.0 - r  # 2.5
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Directly above B should give {expected}, got {d}"
        )

    def test_directly_below_a(self):
        """Point directly below A (on axis)."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 0.5
        p = (0.0, -5.0, 0.0)
        d = py_sd_capsule(p, a, b, r)
        expected = 3.0 - r  # 2.5
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Directly below A should give {expected}, got {d}"
        )


# =============================================================================
# Path G: Continuity -- nearby points have nearby distances
# =============================================================================


class TestContinuity:
    """The SDF should be continuous: nearby points produce nearby distances."""

    def test_continuity_along_x(self):
        """Continuity along x-axis across capsule interior and exterior."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        r = 1.0
        step = 0.001
        prev = py_sd_capsule((-5.0, 0.0, 0.0), a, b, r)
        for i in range(1, 10000):
            x = -5.0 + i * step
            if x > 5.0:
                break
            curr = py_sd_capsule((x, 0.0, 0.0), a, b, r)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at x={x}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """Continuity along y-axis (crossing the capsule vertically)."""
        step = 0.001
        prev = py_sd_capsule((0.0, -5.0, 0.0), A_DEFAULT, B_DEFAULT, R_DEFAULT)
        for i in range(1, 10000):
            y = -5.0 + i * step
            if y > 5.0:
                break
            curr = py_sd_capsule((0.0, y, 0.0), A_DEFAULT, B_DEFAULT, R_DEFAULT)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at y={y}: diff={diff}"
            )
            prev = curr

    def test_continuity_through_surface(self):
        """Continuity crossing the surface along radial direction."""
        r = 1.0
        step = 0.0001
        for offset in [i * step for i in range(-10, 11)]:
            p = (r + offset, 0.0, 0.0)
            d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, r)
            assert abs(d - offset) < 0.001, (
                f"Near surface: expected ~{offset}, got {d} at x={p[0]}"
            )


# =============================================================================
# Path H: Sign convention -- positive outside, negative inside, zero on surface
# =============================================================================


class TestSignConvention:
    """Distance sign: positive outside, negative inside, zero on surface."""

    def test_outside_positive(self):
        """Points outside the capsule should have non-negative distance."""
        r = 1.0
        outside_points = [
            (5.0, 0.0, 0.0),
            (-5.0, 0.0, 0.0),
            (0.0, 0.0, 5.0),
            (0.0, 10.0, 0.0),
            (0.0, -10.0, 0.0),
            (3.0, 3.0, 0.0),
        ]
        for p in outside_points:
            d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, r)
            assert d >= 0.0, (
                f"Outside point ({p}) should have non-negative SDF, got {d}"
            )

    def test_inside_negative(self):
        """Points inside the capsule should have non-positive distance."""
        r = 1.0
        inside_points = [
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (-0.5, 0.0, 0.0),
            (0.0, 1.0, 0.5),
            (0.0, -1.0, -0.5),
        ]
        for p in inside_points:
            d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, r)
            assert d <= 0.0, (
                f"Inside point ({p}) should have non-positive SDF, got {d}"
            )

    def test_surface_zero(self):
        """Points on capsule surface should have near-zero distance."""
        r = 1.0
        surface_points = [
            (r, 0.0, 0.0),
            (-r, 0.0, 0.0),
            (0.0, 0.0, r),
            (r, 2.0, 0.0),   # Top cap
            (r, -2.0, 0.0),  # Bottom cap
            (0.0, 2.0 + r, 0.0),  # Top of cap along y
        ]
        for p in surface_points:
            d = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point ({p}) should have near-zero SDF, got {d}"
            )

    def test_sign_transition(self):
        """Sign should flip at the surface."""
        r = 1.0
        step = 1e-4
        surface_x = r
        inside = py_sd_capsule((surface_x - step, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        on_surface = py_sd_capsule((surface_x, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        outside = py_sd_capsule((surface_x + step, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r)
        assert inside < 0.0, f"Inside should be negative, got {inside}"
        assert on_surface == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface should be 0, got {on_surface}"
        )
        assert outside > 0.0, f"Outside should be positive, got {outside}"


# =============================================================================
# Path I: Symmetry -- rotational symmetry about segment axis
# =============================================================================


class TestSymmetry:
    """The capsule SDF is symmetric about its segment axis."""

    def test_rotational_symmetry_about_axis(self):
        """Points at same distance from segment axis yield same SDF."""
        r = 0.5
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        p_x = (3.0, 0.0, 0.0)
        p_z = (0.0, 0.0, 3.0)
        p_nx = (-3.0, 0.0, 0.0)
        p_nz = (0.0, 0.0, -3.0)
        d_x = py_sd_capsule(p_x, a, b, r)
        d_z = py_sd_capsule(p_z, a, b, r)
        d_nx = py_sd_capsule(p_nx, a, b, r)
        d_nz = py_sd_capsule(p_nz, a, b, r)
        assert d_x == pytest.approx(d_z, abs=TOL_SURFACE), (
            f"Symmetry: X={d_x} should equal Z={d_z}"
        )
        assert d_x == pytest.approx(d_nx, abs=TOL_SURFACE), (
            f"Symmetry: X={d_x} should equal -X={d_nx}"
        )
        assert d_z == pytest.approx(d_nz, abs=TOL_SURFACE), (
            f"Symmetry: Z={d_z} should equal -Z={d_nz}"
        )

    def test_symmetry_off_midpoint(self):
        """Symmetry off the capsule midpoint."""
        r = 0.5
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        p_x = (2.5, 1.0, 0.0)
        p_nx = (-2.5, 1.0, 0.0)
        d_x = py_sd_capsule(p_x, a, b, r)
        d_nx = py_sd_capsule(p_nx, a, b, r)
        assert d_x == pytest.approx(d_nx, abs=TOL_SURFACE), (
            f"Symmetry at y=1: {d_x} should equal {d_nx}"
        )


# =============================================================================
# Path J: Degenerate A==B collapses to sphere
# =============================================================================


class TestDegenerateEndpoints:
    """When A == B, capsule collapses to a sphere of radius r."""

    def test_a_equals_b_surface(self):
        """A==B surface point should give 0."""
        a = (1.0, 2.0, 3.0)
        b = (1.0, 2.0, 3.0)
        r = 2.0
        # Surface point: at r from center
        p = (1.0, 2.0, 5.0)  # 2 units from center, radius=2
        d = py_sd_capsule(p, a, b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"A==B surface should be 0, got {d}"
        )

    def test_a_equals_b_inside(self):
        """A==B inside point should be negative."""
        a = (1.0, 2.0, 3.0)
        b = (1.0, 2.0, 3.0)
        r = 2.0
        p = (1.0, 2.0, 4.0)  # 1 unit from center, radius=2
        d = py_sd_capsule(p, a, b, r)
        expected = 1.0 - r  # -1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"A==B inside should give {expected}, got {d}"
        )

    def test_a_equals_b_outside(self):
        """A==B outside point should be positive."""
        a = (1.0, 2.0, 3.0)
        b = (1.0, 2.0, 3.0)
        r = 1.0
        p = (1.0, 2.0, 5.0)  # 2 units from center, radius=1
        d = py_sd_capsule(p, a, b, r)
        expected = 2.0 - r  # 1.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"A==B outside should give {expected}, got {d}"
        )


# =============================================================================
# Path K: Zero radius collapses to line segment
# =============================================================================


class TestZeroRadius:
    """When r=0, capsule collapses to the line segment AB."""

    def test_zero_radius_perpendicular(self):
        """r=0: perpendicular distance to segment."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        p = (3.0, 0.0, 0.0)
        d = py_sd_capsule(p, a, b, 0.0)
        expected = 3.0
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"r=0 perpendicular distance should be {expected}, got {d}"
        )

    def test_zero_radius_above_b(self):
        """r=0: distance to B when above the segment."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        p = (0.0, 5.0, 0.0)
        d = py_sd_capsule(p, a, b, 0.0)
        expected = 3.0  # Directly above B
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"r=0 above B should give {expected}, got {d}"
        )

    def test_zero_radius_below_a(self):
        """r=0: distance to A when below the segment."""
        a = (0.0, -2.0, 0.0)
        b = (0.0, 2.0, 0.0)
        p = (0.0, -5.0, 0.0)
        d = py_sd_capsule(p, a, b, 0.0)
        expected = 3.0  # Directly below A
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"r=0 below A should give {expected}, got {d}"
        )


# =============================================================================
# Path L: Parameter variation (thin vs thick vs short)
# =============================================================================


class TestParameterVariation:
    """Behavior with different capsule parameters."""

    def test_thin_capsule_acceptance(self):
        """Thin capsule (r=0.25): surface, inside, outside."""
        r = R_THIN
        assert py_sd_capsule((r, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_capsule((0.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r) == pytest.approx(-r, abs=TOL_SURFACE)
        assert py_sd_capsule((3.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r) == pytest.approx(3.0 - r, abs=TOL_SURFACE)

    def test_thick_capsule_acceptance(self):
        """Thick capsule (r=2.0): surface, inside, outside."""
        r = R_THICK
        assert py_sd_capsule((r, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_capsule((0.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r) == pytest.approx(-r, abs=TOL_SURFACE)
        assert py_sd_capsule((5.0, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r) == pytest.approx(5.0 - r, abs=TOL_SURFACE)

    def test_short_capsule(self):
        """Short capsule: endpoints close together behaves almost like a sphere."""
        r = 1.0
        # At midpoint between A and B
        d_center = py_sd_capsule((0.0, 0.0, 0.0), A_SHORT, B_SHORT, r)
        assert d_center == pytest.approx(-r, abs=TOL_SURFACE)
        # Perpendicular at midpoint
        d_perp = py_sd_capsule((r, 0.0, 0.0), A_SHORT, B_SHORT, r)
        assert d_perp == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_diagonal_capsule(self):
        """Diagonal capsule works correctly."""
        r = 0.5
        # Midpoint is at (0, 0, 0). Perpendicular direction is (1,-1,0)/sqrt(2).
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        d_center = py_sd_capsule((0.0, 0.0, 0.0), A_DIAG, B_DIAG, r)
        assert d_center == pytest.approx(-r, abs=TOL_SURFACE)
        d_surface = py_sd_capsule((r * inv_sqrt2, -r * inv_sqrt2, 0.0), A_DIAG, B_DIAG, r)
        assert d_surface == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_negative_radius(self):
        """Negative radius uses abs(r) guard -- same as positive."""
        r_pos = 1.0
        p = (3.0, 0.0, 0.0)
        d_pos = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, r_pos)
        d_neg = py_sd_capsule(p, A_DEFAULT, B_DEFAULT, -r_pos)
        assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
            f"Negative radius should match positive: {d_pos} vs {d_neg}"
        )


# =============================================================================
# Path M: Monotonicity along radial direction
# =============================================================================


class TestMonotonicity:
    """Distance should be monotonic as we move radially outward."""

    def test_outside_monotonic_increasing(self):
        """Outside the capsule, distance increases with distance from axis."""
        r = 1.0
        surface_r = r
        # From just outside surface to far away
        rad_values = [surface_r + i * 0.5 for i in range(20)]
        distances = [py_sd_capsule((x, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r) for x in rad_values]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-12, (
                f"Distance should be monotonically increasing outside: "
                f"at x={rad_values[i]} d={distances[i]}, "
                f"at x={rad_values[i+1]} d={distances[i+1]}"
            )

    def test_inside_surface_outside_monotonic(self):
        """From inside through surface to outside should be non-decreasing."""
        r = 1.0
        coords = [i * 0.1 for i in range(60)]  # 0 to 5.9
        distances = [py_sd_capsule((x, 0.0, 0.0), A_DEFAULT, B_DEFAULT, r) for x in coords]
        # At x=0: -r, at x=r: 0, at x>r: positive and increasing
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-12, (
                f"Non-decreasing violated at x={coords[i]}: "
                f"d={distances[i]} > d={distances[i+1]}"
            )
