"""
Whitebox tests for sdCylinder(p, h, r) WGSL function (T-DEMO-1.4).

Tests a Python model of the WGSL implementation, verifying:
  - Formula decomposition: d = abs(vec2(length(p.xz), p.y)) - vec2(r, h)
  - max(d.x, d.y) gives the larger of radial and axial penetration
  - min(max(d.x,d.y), 0) + length(max(d, 0)) is the full IQ formula
  - Inside the cylinder (negative signed distance)
  - On side surface, top cap, and bottom cap (zero)
  - Outside the cylinder (positive signed distance)
  - r=0 degenerate case (line segment)
  - h=0 degenerate case (flat disc)
  - Symmetry around y-axis and across xz-plane

WHITEBOX coverage plan:
  Path A:  d = abs(vec2(length(p.xz), p.y)) - vec2(r, h) for inside/outside/mixed
  Path B:  max(d.x, d.y) selects radial vs axial dominance
  Path C:  min(max(d.x,d.y), 0) + length(max(d, 0)) = full IQ formula
  Path D:  inside -> both d components negative -> result = max(d.x, d.y) (less negative)
  Path E:  outside wall -> d.x > 0, d.y < 0 -> result = d.x
  Path F:  outside cap -> d.x < 0, d.y > 0 -> result = d.y
  Path G:  outside corner -> both positive -> result = sqrt(d.x^2 + d.y^2)
  Path H:  on side wall -> d.x = 0 -> result = 0
  Path I:  on cap -> d.y = 0 -> result = 0
  Path J:  r=0 -> line segment along y-axis
  Path K:  h=0 -> flat disc on xz-plane
  Path L:  y-axis symmetry: p.x, p.z sign flips yield same result
  Path M:  xz-plane symmetry: p.y sign flip yields same result
"""

import math

import pytest

# =============================================================================
# Python model of WGSL sdCylinder matching GPU semantics
# =============================================================================

TOL = 1e-12


def py_sdCylinder(p, h, r):
    """Model of WGSL sdCylinder: signed distance to capped cylinder.

    The cylinder is axis-aligned along y, centered at origin, radius r,
    half-height h (extends from -h to +h).

    Args:
        p: tuple/list of 3 floats, query position (x, y, z)
        h: float, half-height
        r: float, radius

    Returns:
        float: signed distance (negative inside, zero on surface, positive outside)
    """
    # d = abs(vec2(length(p.xz), p.y)) - vec2(r, h)
    radial = math.sqrt(p[0] * p[0] + p[2] * p[2])  # length(p.xz)
    dx = radial - r
    dy = abs(p[1]) - h
    d = (dx, dy)

    # min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)))
    inside = min(max(d[0], d[1]), 0.0)
    mx = max(d[0], 0.0)
    my = max(d[1], 0.0)
    outside = math.sqrt(mx * mx + my * my)
    return inside + outside


def py_d(p, h, r):
    """Compute d = abs(vec2(length(p.xz), p.y)) - vec2(r, h)."""
    radial = math.sqrt(p[0] * p[0] + p[2] * p[2])
    return (radial - r, abs(p[1]) - h)


def py_max_d(p, h, r):
    """Compute max(d.x, d.y) -- the larger penetration."""
    d = py_d(p, h, r)
    return max(d[0], d[1])


def py_inside_term(p, h, r):
    """Compute min(max(d.x, d.y), 0.0) -- inside contribution."""
    return min(py_max_d(p, h, r), 0.0)


def py_outside_term(p, h, r):
    """Compute length(max(d, vec2(0.0))) -- outside Euclidean distance."""
    d = py_d(p, h, r)
    mx = max(d[0], 0.0)
    my = max(d[1], 0.0)
    return math.sqrt(mx * mx + my * my)


# =============================================================================
# Test: Formula Decomposition -- Path A
# =============================================================================


class TestFormulaDecomposition:
    """Verify d = abs(vec2(length(p.xz), p.y)) - vec2(r, h)."""

    def test_d_inside_both_negative(self):
        """d with p fully inside: both components negative."""
        p = (0.3, 0.4, 0.0)
        h, r = 1.0, 1.0
        d = py_d(p, h, r)
        expected_d = (-0.7, -0.6)
        assert d[0] == pytest.approx(expected_d[0], abs=TOL)
        assert d[1] == pytest.approx(expected_d[1], abs=TOL)

    def test_d_outside_radial(self):
        """d with p outside radially: d.x > 0, d.y < 0."""
        p = (2.0, 0.0, 0.0)
        h, r = 1.0, 1.0
        d = py_d(p, h, r)
        expected_d = (1.0, -1.0)
        assert d[0] == pytest.approx(expected_d[0], abs=TOL)
        assert d[1] == pytest.approx(expected_d[1], abs=TOL)

    def test_d_outside_axial(self):
        """d with p above cap: d.x < 0, d.y > 0."""
        p = (0.0, 2.0, 0.0)
        h, r = 1.0, 1.0
        d = py_d(p, h, r)
        expected_d = (-1.0, 1.0)
        assert d[0] == pytest.approx(expected_d[0], abs=TOL)
        assert d[1] == pytest.approx(expected_d[1], abs=TOL)

    def test_d_outside_both_positive(self):
        """d with p outside both radially and axially: both positive."""
        p = (2.0, 2.0, 0.0)
        h, r = 1.0, 1.0
        d = py_d(p, h, r)
        expected_d = (1.0, 1.0)
        assert d[0] == pytest.approx(expected_d[0], abs=TOL)
        assert d[1] == pytest.approx(expected_d[1], abs=TOL)

    def test_d_on_side_wall(self):
        """d with p on side wall: d.x = 0, d.y < 0."""
        p = (1.0, 0.0, 0.0)
        h, r = 1.0, 1.0
        d = py_d(p, h, r)
        expected_d = (0.0, -1.0)
        assert d[0] == pytest.approx(expected_d[0], abs=TOL)
        assert d[1] == pytest.approx(expected_d[1], abs=TOL)

    def test_d_on_top_cap(self):
        """d with p on top cap: d.x < 0, d.y = 0."""
        p = (0.0, 1.0, 0.0)
        h, r = 1.0, 1.0
        d = py_d(p, h, r)
        expected_d = (-1.0, 0.0)
        assert d[0] == pytest.approx(expected_d[0], abs=TOL)
        assert d[1] == pytest.approx(expected_d[1], abs=TOL)

    def test_d_on_bottom_cap(self):
        """d with p on bottom cap: d.x < 0, d.y = 0 (abs(p.y) = h)."""
        p = (0.0, -1.0, 0.0)
        h, r = 1.0, 1.0
        d = py_d(p, h, r)
        expected_d = (-1.0, 0.0)
        assert d[0] == pytest.approx(expected_d[0], abs=TOL)
        assert d[1] == pytest.approx(expected_d[1], abs=TOL)

    def test_d_length_xz(self):
        """length(p.xz) correctly computes sqrt(x^2 + z^2)."""
        p = (3.0, 0.0, 4.0)
        h, r = 1.0, 1.0
        d = py_d(p, h, r)
        # radial = 5, so d.x = 5 - 1 = 4
        assert d[0] == pytest.approx(4.0, abs=TOL)

    def test_d_abs_py(self):
        """abs(p.y) correctly handles negative y."""
        p = (0.0, -0.5, 0.0)
        h, r = 1.0, 1.0
        d = py_d(p, h, r)
        assert d[1] == pytest.approx(-0.5, abs=TOL)  # abs(-0.5) - 1 = -0.5

    def test_d_on_rim_junction(self):
        """d on rim (wall meets top cap): both components zero."""
        p = (1.0, 1.0, 0.0)
        h, r = 1.0, 1.0
        d = py_d(p, h, r)
        expected_d = (0.0, 0.0)
        assert d[0] == pytest.approx(expected_d[0], abs=TOL)
        assert d[1] == pytest.approx(expected_d[1], abs=TOL)

    def test_d_negative_p_xz(self):
        """abs handles negative x and z: same as positive x and z."""
        p_pos = (1.5, 0.0, 2.0)
        p_neg = (-1.5, 0.0, -2.0)
        h, r = 1.0, 1.0
        d_pos = py_d(p_pos, h, r)
        d_neg = py_d(p_neg, h, r)
        assert d_pos[0] == pytest.approx(d_neg[0], abs=TOL)
        assert d_pos[1] == pytest.approx(d_neg[1], abs=TOL)


# =============================================================================
# Test: max(d.x, d.y) -- Path B
# =============================================================================


class TestMaxD:
    """Verify max(d.x, d.y) gives the larger of radial and axial penetration."""

    def test_radial_dominates_inside(self):
        """Inside: d.x less negative than d.y -> max = d.x."""
        d = (-0.3, -0.7)
        assert max(d[0], d[1]) == -0.3

    def test_axial_dominates_inside(self):
        """Inside: d.y less negative than d.x -> max = d.y."""
        d = (-0.7, -0.3)
        assert max(d[0], d[1]) == -0.3

    def test_radial_dominates_outside(self):
        """Outside radially: d.x positive, d.y negative -> max = d.x."""
        d = (0.5, -1.0)
        assert max(d[0], d[1]) == 0.5

    def test_axial_dominates_outside(self):
        """Outside axially: d.y positive, d.x negative -> max = d.y."""
        d = (-1.0, 0.5)
        assert max(d[0], d[1]) == 0.5

    def test_both_equal_inside(self):
        """Inside with equal radial and axial penetration."""
        d = (-0.5, -0.5)
        assert max(d[0], d[1]) == -0.5

    def test_both_equal_outside(self):
        """Outside corner with equal positive components."""
        d = (0.5, 0.5)
        assert max(d[0], d[1]) == 0.5

    def test_on_side_wall_negative_y(self):
        """On side wall: d.x=0, d.y<0 -> max = 0."""
        d = (0.0, -0.5)
        assert max(d[0], d[1]) == 0.0

    def test_on_cap_positive_y(self):
        """On cap with positive axial: d.x<0, d.y=0 -> max = 0."""
        d = (-0.5, 0.0)
        assert max(d[0], d[1]) == 0.0

    def test_point_on_rim(self):
        """On rim: d.x=0, d.y=0 -> max = 0."""
        d = (0.0, 0.0)
        assert max(d[0], d[1]) == 0.0


# =============================================================================
# Test: IQ Formula -- Path C
# =============================================================================


class TestIQFormula:
    """Verify min(max(d.x,d.y), 0) + length(max(d, 0)) is the complete formula."""

    def test_inside_only_term(self):
        """Inside: outside term is zero, result = inside term = max(d)."""
        p = (0.3, 0.4, 0.0)
        h, r = 1.0, 1.0
        inside = py_inside_term(p, h, r)
        outside = py_outside_term(p, h, r)
        full = py_sdCylinder(p, h, r)
        assert outside == pytest.approx(0.0, abs=TOL)
        assert full == pytest.approx(inside, abs=TOL)

    def test_outside_radial_only_term(self):
        """Outside radially: inside term is zero, result = radial distance."""
        p = (2.0, 0.0, 0.0)
        h, r = 1.0, 1.0
        inside = py_inside_term(p, h, r)
        outside = py_outside_term(p, h, r)
        full = py_sdCylinder(p, h, r)
        assert inside == pytest.approx(0.0, abs=TOL)
        assert full == pytest.approx(outside, abs=TOL)

    def test_outside_axial_only_term(self):
        """Outside axially: inside term is zero, result = axial distance."""
        p = (0.0, 2.0, 0.0)
        h, r = 1.0, 1.0
        inside = py_inside_term(p, h, r)
        outside = py_outside_term(p, h, r)
        full = py_sdCylinder(p, h, r)
        assert inside == pytest.approx(0.0, abs=TOL)
        assert full == pytest.approx(outside, abs=TOL)

    def test_outside_corner_both_terms(self):
        """Outside corner: both inside and outside contribute (inside term = 0)."""
        p = (2.0, 2.0, 0.0)
        h, r = 1.0, 1.0
        inside = py_inside_term(p, h, r)
        outside = py_outside_term(p, h, r)
        full = py_sdCylinder(p, h, r)
        # d = (1, 1), max = 1, min(1, 0) = 0
        # length(max(d, 0)) = length(1, 1) = sqrt(2)
        assert inside == pytest.approx(0.0, abs=TOL)
        assert outside == pytest.approx(math.sqrt(2.0), abs=TOL)
        assert full == pytest.approx(math.sqrt(2.0), abs=TOL)

    def test_on_surface_both_terms_zero(self):
        """On surface: both terms are zero, result = 0."""
        p = (0.0, 1.0, 0.0)
        h, r = 1.0, 1.0
        full = py_sdCylinder(p, h, r)
        assert full == pytest.approx(0.0, abs=TOL)

    def test_sum_equals_full(self):
        """inside_term + outside_term always equals full sdCylinder."""
        points = [
            (0.0, 0.0, 0.0),
            (0.3, 0.4, 0.0),
            (0.5, -0.5, 0.3),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (2.0, 2.0, 0.0),
            (1.5, 1.5, 1.5),
            (-0.5, 0.3, -0.8),
        ]
        h, r = 1.0, 1.0
        for p in points:
            inside = py_inside_term(p, h, r)
            outside = py_outside_term(p, h, r)
            full = py_sdCylinder(p, h, r)
            assert full == pytest.approx(inside + outside, abs=TOL), (
                f"p={p}: inside({inside}) + outside({outside}) != full({full})"
            )


# =============================================================================
# Test: Inside Cylinder -- Path D
# =============================================================================


class TestInside:
    """Inside the cylinder: both d components negative, result = max(d.x, d.y)."""

    def test_center_deep_inside(self):
        """At origin: both axes at max depth, result = -min(r, h)."""
        p = (0.0, 0.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        # radial = 0, dx = -1; dy = 0-1 = -1; max = -1; result = -1
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_inside_off_axis(self):
        """Off-axis inside: p=(0.3, 0, 0), r=1, h=1 -> result=-0.7 (radial limits)."""
        p = (0.3, 0.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        # dx = 0.3-1 = -0.7, dy = 0-1 = -1, max = -0.7, inside = -0.7, outside=0
        assert d == pytest.approx(-0.7, abs=TOL)

    def test_inside_near_side_wall(self):
        """Near side wall: p=(0.9, 0, 0), result=-0.1."""
        p = (0.9, 0.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(-0.1, abs=TOL)

    def test_inside_near_top_cap(self):
        """Near top cap: p=(0, 0.9, 0), result=-0.1."""
        p = (0.0, 0.9, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(-0.1, abs=TOL)

    def test_inside_near_bottom_cap(self):
        """Near bottom cap: p=(0, -0.9, 0), result=-0.1."""
        p = (0.0, -0.9, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(-0.1, abs=TOL)

    def test_inside_tall_cylinder(self):
        """Inside a tall cylinder: h=5, r=1, p=(0.5, 3, 0)."""
        p = (0.5, 3.0, 0.0)
        h, r = 5.0, 1.0
        d = py_sdCylinder(p, h, r)
        # dx = 0.5-1 = -0.5, dy = 3-5 = -2, max = -0.5, result = -0.5
        assert d == pytest.approx(-0.5, abs=TOL)

    def test_inside_wide_cylinder(self):
        """Inside a wide cylinder: h=1, r=5, p=(3, 0.5, 0)."""
        p = (3.0, 0.5, 0.0)
        h, r = 1.0, 5.0
        d = py_sdCylinder(p, h, r)
        # dx = 3-5 = -2, dy = 0.5-1 = -0.5, max = -0.5, result = -0.5
        assert d == pytest.approx(-0.5, abs=TOL)

    def test_inside_deep_multi_dim(self):
        """Deep inside large cylinder: p=(0,0,0), r=10, h=10, result=-10."""
        p = (0.0, 0.0, 0.0)
        h, r = 10.0, 10.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(-10.0, abs=TOL)

    def test_inside_negative_region(self):
        """Inside with negative coordinates: p=(-0.4, -0.3, 0.2), r=1, h=1."""
        p = (-0.4, -0.3, 0.2)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        # radial = sqrt(0.16+0.04) = sqrt(0.2) ≈ 0.447, dx ≈ -0.553
        # dy = 0.3-1 = -0.7, max = -0.553
        radial = math.sqrt(0.16 + 0.04)
        expected = radial - 1.0  # max(-0.553, -0.7) = -0.553
        assert d == pytest.approx(expected, abs=TOL)


# =============================================================================
# Test: On Surface -- Paths H, I
# =============================================================================


class TestOnSurface:
    """On the cylinder surface: signed distance is zero."""

    def test_on_side_wall_pos_x(self):
        """On side wall at +x: p=(1, 0, 0), r=1, h=1 -> 0."""
        p = (1.0, 0.0, 0.0)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_side_wall_neg_x(self):
        """On side wall at -x: p=(-1, 0, 0), r=1, h=1 -> 0."""
        p = (-1.0, 0.0, 0.0)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_side_wall_pos_z(self):
        """On side wall at +z: p=(0, 0, 1), r=1, h=1 -> 0."""
        p = (0.0, 0.0, 1.0)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_side_wall_neg_z(self):
        """On side wall at -z: p=(0, 0, -1), r=1, h=1 -> 0."""
        p = (0.0, 0.0, -1.0)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_side_wall_diagonal(self):
        """On side wall at diagonal: p=(1/sqrt(2), 0, 1/sqrt(2)), r=1, h=1 -> 0."""
        inv_rt2 = 1.0 / math.sqrt(2.0)
        p = (inv_rt2, 0.0, inv_rt2)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_top_cap(self):
        """On top cap: p=(0, 1, 0), r=1, h=1 -> 0."""
        p = (0.0, 1.0, 0.0)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_bottom_cap(self):
        """On bottom cap: p=(0, -1, 0), r=1, h=1 -> 0."""
        p = (0.0, -1.0, 0.0)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_top_cap_off_center(self):
        """On top cap off-center: p=(0.5, 1, 0.3), r=1, h=1 -> 0."""
        p = (0.5, 1.0, 0.3)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_bottom_cap_off_center(self):
        """On bottom cap off-center: p=(-0.3, -1, 0.6), r=1, h=1 -> 0."""
        p = (-0.3, -1.0, 0.6)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_rim_top(self):
        """On top rim (wall meets cap): p=(1, 1, 0), r=1, h=1 -> 0."""
        p = (1.0, 1.0, 0.0)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_rim_bottom(self):
        """On bottom rim: p=(1, -1, 0), r=1, h=1 -> 0."""
        p = (1.0, -1.0, 0.0)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)

    def test_on_rim_top_diagonal(self):
        """On top rim at diagonal: p=(1/sqrt(2), 1, 1/sqrt(2)), r=1, h=1 -> 0."""
        inv_rt2 = 1.0 / math.sqrt(2.0)
        p = (inv_rt2, 1.0, inv_rt2)
        h, r = 1.0, 1.0
        assert py_sdCylinder(p, h, r) == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Test: Outside Cylinder -- Paths E, F, G
# =============================================================================


class TestOutside:
    """Outside the cylinder: positive signed distance."""

    def test_outside_side_wall(self):
        """Outside side wall: p=(2, 0, 0), r=1 -> 1 (wall distance)."""
        p = (2.0, 0.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_outside_side_wall_3_units(self):
        """Outside side wall 3 units: p=(4, 0, 0), r=1 -> 3."""
        p = (4.0, 0.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(3.0, abs=TOL)

    def test_outside_above_cap(self):
        """Above top cap: p=(0, 2, 0), h=1 -> 1."""
        p = (0.0, 2.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_outside_below_cap(self):
        """Below bottom cap: p=(0, -2, 0), h=1 -> 1."""
        p = (0.0, -2.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_outside_above_cap_3_units(self):
        """Above cap 3 units: p=(0, 4, 0), h=1 -> 3."""
        p = (0.0, 4.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(3.0, abs=TOL)

    def test_outside_corner_45_deg(self):
        """Outside corner at 45 deg: p=(2, 2, 0) -> sqrt(2)."""
        p = (2.0, 2.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(math.sqrt(2.0), abs=TOL)

    def test_outside_corner_asymmetric(self):
        """Outside corner with large radial: p=(4, 2, 0) -> sqrt(9+1) = sqrt(10)."""
        p = (4.0, 2.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        # dx = 3, dy = 1, length(max(d,0)) = sqrt(9+1) = sqrt(10)
        assert d == pytest.approx(math.sqrt(10.0), abs=TOL)

    def test_outside_diagonal_3d(self):
        """Outside diagonal 3D: p=(2, 2, 2), r=1, h=1."""
        p = (2.0, 2.0, 2.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        # radial = sqrt(4+4) = sqrt(8) ≈ 2.828, dx = 1.828
        # dy = 2-1 = 1
        # length = sqrt(1.828^2 + 1^2) ≈ sqrt(4.34) ≈ 2.083
        radial = math.sqrt(8.0)
        dx = radial - 1.0
        expected = math.sqrt(dx * dx + 1.0)
        assert d == pytest.approx(expected, abs=TOL)

    def test_outside_far_axial_near_wall(self):
        """Far above but near wall radially: p=(0.9, 3, 0), r=1, h=1."""
        p = (0.9, 3.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        # dx = 0.9-1 = -0.1, dy = 3-1 = 2
        # max = 2, inside = 0
        # length(max(d,0)) = length(0, 2) = 2
        assert d == pytest.approx(2.0, abs=TOL)

    def test_outside_far_radial_near_cap(self):
        """Far radially but near cap: p=(3, 0.9, 0), r=1, h=1."""
        p = (3.0, 0.9, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        # dx = 3-1 = 2, dy = 0.9-1 = -0.1
        # max = 2, inside = 0
        # length = 2
        assert d == pytest.approx(2.0, abs=TOL)

    def test_outside_side_wall_negative_x(self):
        """Outside side wall at negative x: p=(-2, 0, 0) -> 1."""
        p = (-2.0, 0.0, 0.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_outside_z_direction(self):
        """Outside side wall along z: p=(0, 0, 2) -> 1."""
        p = (0.0, 0.0, 2.0)
        h, r = 1.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_outside_monotonic_increase(self):
        """Distance monotonically increases as p moves radially outward from side wall."""
        h, r = 1.0, 1.0
        prev = py_sdCylinder((1.0, 0.0, 0.0), h, r)
        for radial_dist in [1.5, 2.0, 3.0, 5.0]:
            curr = py_sdCylinder((radial_dist, 0.0, 0.0), h, r)
            assert curr >= prev, (
                f"Distance should increase radially"
            )
            prev = curr


# =============================================================================
# Test: r=0 Degenerate Cylinder (Line Segment) -- Path J
# =============================================================================


class TestRadiusZero:
    """r=0: cylinder collapses to a line segment of length 2h along y-axis."""

    def test_on_segment_origin(self):
        """At origin: p=(0,0,0), r=0, h=1 -> on the line, result=0."""
        p = (0.0, 0.0, 0.0)
        h, r = 1.0, 0.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_on_segment_mid(self):
        """On segment midpoint: p=(0, 0.5, 0), r=0, h=1 -> 0."""
        p = (0.0, 0.5, 0.0)
        h, r = 1.0, 0.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_off_segment_radially(self):
        """Off segment radially: p=(0.3, 0, 0), r=0, h=1 -> 0.3."""
        p = (0.3, 0.0, 0.0)
        h, r = 1.0, 0.0
        d = py_sdCylinder(p, h, r)
        # dx = 0.3-0 = 0.3, dy = 0-1 = -1, max=0.3, inside=0, outside=0.3
        assert d == pytest.approx(0.3, abs=TOL)

    def test_above_segment_top(self):
        """Above segment: p=(0, 2, 0), r=0, h=1 -> 1."""
        p = (0.0, 2.0, 0.0)
        h, r = 1.0, 0.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_below_segment_bottom(self):
        """Below segment: p=(0, -2, 0), r=0, h=1 -> 1."""
        p = (0.0, -2.0, 0.0)
        h, r = 1.0, 0.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_diagonal_to_segment(self):
        """Diagonal to segment: p=(0.3, 2, 0), r=0, h=1 -> sqrt(0.09+1)."""
        p = (0.3, 2.0, 0.0)
        h, r = 1.0, 0.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(math.sqrt(0.09 + 1.0), abs=TOL)

    def test_off_segment_end_z(self):
        """Off segment end along z: p=(0, 0, 0.5), r=0, h=1 -> 0.5."""
        p = (0.0, 0.0, 0.5)
        h, r = 1.0, 0.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_off_segment_diagonal_3d(self):
        """Off segment diagonally in 3D: p=(0.3, 0.5, 0.4), r=0, h=1."""
        p = (0.3, 0.5, 0.4)
        h, r = 1.0, 0.0
        d = py_sdCylinder(p, h, r)
        # radial = sqrt(0.09+0.16) = 0.5, dx = 0.5, dy = 0.5-1 = -0.5
        # max(0.5, -0.5) = 0.5, inside = 0, outside = 0.5
        assert d == pytest.approx(0.5, abs=TOL)

    def test_inside_segment_hspan(self):
        """Inside the vertical span but off axis: radial distance."""
        p = (0.0, 0.0, 0.0)  # on axis
        h, r = 5.0, 0.0
        d_on = py_sdCylinder(p, h, r)
        assert d_on == pytest.approx(0.0, abs=TOL)

        p_off = (2.0, 0.0, 0.0)  # off axis
        d_off = py_sdCylinder(p_off, h, r)
        assert d_off == pytest.approx(2.0, abs=TOL)


# =============================================================================
# Test: h=0 Degenerate Cylinder (Flat Disc) -- Path K
# =============================================================================


class TestHeightZero:
    """h=0: cylinder collapses to a flat disc of radius r on xz-plane."""

    def test_on_disc_center(self):
        """At disc center: p=(0,0,0), h=0, r=1 -> on disc, result=0."""
        p = (0.0, 0.0, 0.0)
        h, r = 0.0, 1.0
        d = py_sdCylinder(p, h, r)
        # dx = 0-1 = -1, dy = 0-0 = 0, max = 0, inside = 0, outside = 0
        assert d == pytest.approx(0.0, abs=TOL)

    def test_on_disc_off_center(self):
        """On disc off-center: p=(0.5, 0, 0), h=0, r=1 -> 0."""
        p = (0.5, 0.0, 0.0)
        h, r = 0.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_on_disc_rim(self):
        """On disc rim: p=(1, 0, 0), h=0, r=1 -> 0."""
        p = (1.0, 0.0, 0.0)
        h, r = 0.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_above_disc_center(self):
        """Above disc center: p=(0, 0.5, 0), h=0, r=1 -> 0.5."""
        p = (0.0, 0.5, 0.0)
        h, r = 0.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_below_disc_center(self):
        """Below disc center: p=(0, -0.5, 0), h=0, r=1 -> 0.5."""
        p = (0.0, -0.5, 0.0)
        h, r = 0.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_beyond_disc_rim(self):
        """Beyond disc rim in-plane: p=(2, 0, 0), h=0, r=1 -> 1."""
        p = (2.0, 0.0, 0.0)
        h, r = 0.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_diagonal_to_disc_rim(self):
        """Diagonal beyond rim: p=(2, 0.5, 0), h=0, r=1 -> sqrt(1+0.25)=sqrt(1.25)."""
        p = (2.0, 0.5, 0.0)
        h, r = 0.0, 1.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(math.sqrt(1.25), abs=TOL)

    def test_disc_off_plane_z(self):
        """Disc test with z coordinate: p=(0, 0.3, 0.5), h=0, r=1."""
        p = (0.0, 0.3, 0.5)
        h, r = 0.0, 1.0
        d = py_sdCylinder(p, h, r)
        # radial = 0.5, dx = 0.5-1 = -0.5, dy = 0.3-0 = 0.3
        # max = 0.3, inside = 0, outside = 0.3
        assert d == pytest.approx(0.3, abs=TOL)

    def test_both_zero_r_and_h(self):
        """Both r=0 and h=0: single point at origin."""
        p = (0.0, 0.0, 0.0)
        h, r = 0.0, 0.0
        d = py_sdCylinder(p, h, r)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_both_zero_off_origin(self):
        """Both r=0 and h=0: off origin -> distance to origin."""
        p = (1.0, 2.0, 3.0)
        h, r = 0.0, 0.0
        d = py_sdCylinder(p, h, r)
        # dx = sqrt(1+9) - 0 = sqrt(10), dy = 2-0 = 2
        # length(max(sqrt(10), 2), 0) = sqrt(10+4) = sqrt(14)
        expected = math.sqrt(14.0)
        assert d == pytest.approx(expected, abs=TOL)


# =============================================================================
# Test: Symmetry -- Paths L, M
# =============================================================================


class TestSymmetry:
    """Verify y-axis symmetry and xz-plane symmetry of sdCylinder."""

    def test_y_axis_symmetry_sign_x(self):
        """Sign flip of x yields same distance."""
        p_base = (0.5, 0.3, 0.4)
        h, r = 1.0, 1.0
        base = py_sdCylinder(p_base, h, r)
        signs_x = [
            (1, 1, 1),
            (-1, 1, 1),
            (1, 1, -1),
            (-1, 1, -1),
        ]
        for s in signs_x:
            ps = (p_base[0] * s[0], p_base[1] * s[1], p_base[2] * s[2])
            assert py_sdCylinder(ps, h, r) == pytest.approx(base, abs=TOL), (
                f"y-axis symmetry broken for signs {s}"
            )

    def test_y_axis_symmetry_rotation(self):
        """Rotation around y-axis yields same distance (radial is invariant)."""
        p = (0.5, 0.3, 0.0)
        h, r = 1.0, 1.0
        base = py_sdCylinder(p, h, r)
        # Rotate 90 degrees around y: (0, 0.3, 0.5)
        p_rot = (0.0, 0.3, 0.5)
        assert py_sdCylinder(p_rot, h, r) == pytest.approx(base, abs=TOL)

    def test_y_axis_symmetry_180_rotation(self):
        """180 degree rotation around y: (x,y,z) -> (-x,y,-z)."""
        p = (0.5, 0.3, 0.7)
        h, r = 1.0, 1.0
        base = py_sdCylinder(p, h, r)
        p_rot = (-0.5, 0.3, -0.7)
        assert py_sdCylinder(p_rot, h, r) == pytest.approx(base, abs=TOL)

    def test_xz_plane_symmetry(self):
        """Sign flip of y yields same distance (symmetry across xz-plane)."""
        p = (0.5, 0.7, 0.3)
        h, r = 1.0, 1.0
        base = py_sdCylinder(p, h, r)
        p_flip = (0.5, -0.7, 0.3)
        assert py_sdCylinder(p_flip, h, r) == pytest.approx(base, abs=TOL)

    def test_xz_plane_symmetry_above_below_cap(self):
        """Above and below cap at same distance yield same result."""
        h, r = 1.0, 1.0
        above = py_sdCylinder((0.0, 2.0, 0.0), h, r)
        below = py_sdCylinder((0.0, -2.0, 0.0), h, r)
        assert above == pytest.approx(below, abs=TOL)

    def test_deterministic(self):
        """Same inputs always produce same output."""
        p = (0.7, 0.3, 0.9)
        h, r = 1.0, 2.0
        base = py_sdCylinder(p, h, r)
        for _ in range(20):
            assert py_sdCylinder(p, h, r) == pytest.approx(base, abs=TOL)

    def test_all_quadrants_radial_symmetry(self):
        """All combinations of x,z signs at same magnitude yield same distance."""
        p_mag = (0.5, 0.4, 0.3)
        h, r = 1.5, 1.0
        base = py_sdCylinder(p_mag, h, r)
        xz_signs = [
            (0.5, 0.3),
            (-0.5, 0.3),
            (0.5, -0.3),
            (-0.5, -0.3),
        ]
        for x, z in xz_signs:
            ps = (x, p_mag[1], z)
            assert py_sdCylinder(ps, h, r) == pytest.approx(base, abs=TOL), (
                f"Radial symmetry broken for x={x}, z={z}"
            )

    def test_inside_decreasing_distance(self):
        """Distance becomes less negative as p moves radially outward inside."""
        h, r = 2.0, 2.0
        prev = py_sdCylinder((0.0, 0.0, 0.0), h, r)
        for radial_pos in [0.5, 1.0, 1.5]:
            curr = py_sdCylinder((radial_pos, 0.0, 0.0), h, r)
            assert curr > prev, (
                f"Distance should increase (less negative) from {prev} to {curr}"
            )
            prev = curr
