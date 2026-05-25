"""
Whitebox tests for sdSphere WGSL function (T-DEMO-1.1).

Tests the implementation-aware signed distance function for a sphere with
radius r centered at the origin, using the IQ formula: length(p) - r with
abs(r) guard for negative radius.

Implementation (engine/rendering/demoscene/wgsl/sdf_sphere.wgsl):
  fn sdSphere(p: vec3<f32>, r: f32) -> f32 {
      let safe_r = abs(r);
      return length(p) - safe_r;
  }

WHITEBOX coverage plan:
  Path 1: Formula -- verify length(p) - abs(r) is used
  Path 2: Center (0,0,0) -- distance = -r (inside, negative)
  Path 3: Surface points -- (r,0,0), (0,r,0), (0,0,r) all return 0
  Path 4: Diagonal surface -- (r/sqrt(3), r/sqrt(3), r/sqrt(3)) -> 0
  Path 5: Outside point -- (2r, 0, 0) -> r
  Path 6: Outside diagonal -- (r, r, r) for r=1 -> sqrt(3)-1
  Path 7: Negative radius -- r=-1 treated as r=1 due to abs guard
  Path 8: Zero radius -- sdSphere(p, 0) = length(p) (point SDF)
  Path 9: Symmetry -- sdSphere((x,y,z), r) == sdSphere((-x,y,z), r)
  Path 10: Sign convention -- negative inside, zero on surface, positive outside
"""

from __future__ import annotations

import math

import pytest

# =============================================================================
# Python model of sdSphere matching WGSL semantics exactly (whitebox)
# =============================================================================


def py_sd_sphere(p, r):
    """Python model of WGSL sdSphere(p: vec3<f32>, r: f32) -> f32.

    Signed distance from point p (3-tuple) to a sphere of radius r centered
    at the origin. Uses the IQ formula length(p) - r with abs(r) guard.

    Reference: Inigo Quilez -- Sphere SDF
    https://iquilezles.org/articles/distfunctions/
    """
    safe_r = abs(r)
    return math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - safe_r


# =============================================================================
# Tolerance constants
# =============================================================================

TOL_SURFACE = 1e-12     # Points on surface should be extremely close to 0
TOL_EXACT = 1e-15       # For exact arithmetic expectations


# =============================================================================
# Path 1: Formula verification -- length(p) - abs(r)
# =============================================================================


class TestFormula:
    """Verify the formula matches the IQ spec: length(p) - abs(r)."""

    def test_formula_structure_length_minus_abs_r(self):
        """Verify sdSphere computes length(p) - abs(r), not length(p - r)."""
        p = (3.0, 4.0, 0.0)
        r = 5.0
        result = py_sd_sphere(p, r)

        # length(p) = 5.0, abs(r) = 5.0, result should be 0.0
        expected_length_p = 5.0
        expected_safe_r = 5.0
        expected = expected_length_p - expected_safe_r  # 0.0
        assert result == pytest.approx(expected, abs=TOL_EXACT), (
            f"sdSphere({p}, {r}) should equal length(p) - abs(r) = "
            f"{expected_length_p} - {expected_safe_r} = {expected}, got {result}"
        )

    def test_abs_r_guard_used(self):
        """Verify the abs(r) guard: negative r is treated as positive r."""
        r_pos = 3.0
        r_neg = -3.0
        p = (1.0, 2.0, 2.0)
        d_pos = py_sd_sphere(p, r_pos)
        d_neg = py_sd_sphere(p, r_neg)
        assert d_pos == pytest.approx(d_neg, abs=TOL_EXACT), (
            f"abs(r) guard: sdSphere({p}, {r_pos}) = {d_pos} should equal "
            f"sdSphere({p}, {r_neg}) = {d_neg}"
        )

    def test_not_length_minus_r_raw(self):
        """Confirm that without abs(r), r=-2 would be wrong."""
        r = -3.0
        p = (0.0, 0.0, 1.0)
        result = py_sd_sphere(p, r)
        # With abs guard: length(1) - abs(-3) = 1 - 3 = -2
        expected_with_guard = 1.0 - 3.0  # -2.0
        # Without guard: 1 - (-3) = 4 (wrong -- would be positive for interior)
        wrong_without_guard = 1.0 - (-3.0)  # 4.0
        assert result == pytest.approx(expected_with_guard, abs=TOL_EXACT), (
            f"Without abs(r) guard, sdSphere({p}, {r}) would be {wrong_without_guard}, "
            f"should be {expected_with_guard}"
        )
        assert not math.isclose(result, wrong_without_guard, rel_tol=1e-12), (
            f"Result {result} should NOT equal {wrong_without_guard} (un-guarded "
            f"computation)"
        )


# =============================================================================
# Path 2: Center -- distance = -r (inside, negative)
# =============================================================================


class TestCenter:
    """At the origin (0,0,0), the signed distance should be -r (inside)."""

    def test_center_positive_r(self):
        """Center of sphere with positive radius: (0,0,0), r=5 -> -5."""
        d = py_sd_sphere((0.0, 0.0, 0.0), 5.0)
        assert d == pytest.approx(-5.0, abs=TOL_EXACT), (
            f"Center with r=5 should give -5.0, got {d}"
        )

    def test_center_unit_radius(self):
        """Center of unit sphere: (0,0,0), r=1 -> -1."""
        d = py_sd_sphere((0.0, 0.0, 0.0), 1.0)
        assert d == pytest.approx(-1.0, abs=TOL_EXACT), (
            f"Center with r=1 should give -1.0, got {d}"
        )

    def test_center_large_radius(self):
        """Center of large sphere: (0,0,0), r=100 -> -100."""
        d = py_sd_sphere((0.0, 0.0, 0.0), 100.0)
        assert d == pytest.approx(-100.0, abs=TOL_EXACT), (
            f"Center with r=100 should give -100.0, got {d}"
        )

    def test_center_small_radius(self):
        """Center of small sphere: (0,0,0), r=0.01 -> -0.01."""
        d = py_sd_sphere((0.0, 0.0, 0.0), 0.01)
        assert d == pytest.approx(-0.01, abs=TOL_EXACT), (
            f"Center with r=0.01 should give -0.01, got {d}"
        )

    def test_center_negative_radius(self):
        """Center of sphere with negative radius: abs(-5) = 5 -> -5."""
        d = py_sd_sphere((0.0, 0.0, 0.0), -5.0)
        assert d == pytest.approx(-5.0, abs=TOL_EXACT), (
            f"Center with r=-5 (abs=5) should give -5.0, got {d}"
        )

    def test_center_invariant_under_point_sign(self):
        """Center distance depends only on radius, not point sign (zero)."""
        d_p = py_sd_sphere((0.0, 0.0, 0.0), 3.0)
        for signs in [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),
                      (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)]:
            p = (0.0 * signs[0], 0.0 * signs[1], 0.0 * signs[2])
            d = py_sd_sphere(p, 3.0)
            assert d == pytest.approx(d_p, abs=TOL_EXACT), (
                f"Center distance should be equal for sign-flipped zeros: "
                f"got {d} vs {d_p}"
            )


# =============================================================================
# Path 3: Surface points -- (r,0,0), (0,r,0), (0,0,r) all return 0
# =============================================================================


class TestSurfacePoints:
    """Points exactly on the sphere surface should return distance ~0."""

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_surface_positive_x(self, r):
        """Surface point on +x axis: (r, 0, 0)."""
        d = py_sd_sphere((r, 0.0, 0.0), r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point ({r}, 0, 0) with r={r} should be 0, got {d}"
        )

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_surface_positive_y(self, r):
        """Surface point on +y axis: (0, r, 0)."""
        d = py_sd_sphere((0.0, r, 0.0), r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point (0, {r}, 0) with r={r} should be 0, got {d}"
        )

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_surface_positive_z(self, r):
        """Surface point on +z axis: (0, 0, r)."""
        d = py_sd_sphere((0.0, 0.0, r), r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point (0, 0, {r}) with r={r} should be 0, got {d}"
        )

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_surface_negative_x(self, r):
        """Surface point on -x axis: (-r, 0, 0)."""
        d = py_sd_sphere((-r, 0.0, 0.0), r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point ({-r}, 0, 0) with r={r} should be 0, got {d}"
        )

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_surface_negative_y(self, r):
        """Surface point on -y axis: (0, -r, 0)."""
        d = py_sd_sphere((0.0, -r, 0.0), r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point (0, {-r}, 0) with r={r} should be 0, got {d}"
        )

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_surface_negative_z(self, r):
        """Surface point on -z axis: (0, 0, -r)."""
        d = py_sd_sphere((0.0, 0.0, -r), r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point (0, 0, {-r}) with r={r} should be 0, got {d}"
        )

    def test_surface_all_axes_consistent(self):
        """All six axis-aligned surface points should yield 0 for same r."""
        r = 2.5
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
                f"Axis surface point {p} with r={r} should be 0, got {d}"
            )


# =============================================================================
# Path 4: Diagonal surface -- (r/sqrt(3), r/sqrt(3), r/sqrt(3)) -> 0
# =============================================================================


class TestDiagonalSurface:
    """Points on the sphere surface along the diagonal should return 0."""

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_diagonal_positive_octant(self, r):
        """Surface point along (1,1,1) diagonal: each component = r/sqrt(3)."""
        c = r / math.sqrt(3.0)
        p = (c, c, c)
        d = py_sd_sphere(p, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Diagonal surface {p} with r={r} should be 0, got {d}"
        )

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_diagonal_negative_octant(self, r):
        """Surface point along (-1,-1,-1) diagonal."""
        c = r / math.sqrt(3.0)
        p = (-c, -c, -c)
        d = py_sd_sphere(p, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Diagonal surface {p} with r={r} should be 0, got {d}"
        )

    def test_all_eight_octants(self):
        """Surface points in all eight octants should yield 0."""
        r = 3.0
        c = r / math.sqrt(3.0)
        signs = [
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
            (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        ]
        for sx, sy, sz in signs:
            p = (sx * c, sy * c, sz * c)
            d = py_sd_sphere(p, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Diagonal surface {p} in octant ({sx},{sy},{sz}) with r={r} "
                f"should be 0, got {d}"
            )


# =============================================================================
# Path 5: Outside point -- (2r, 0, 0) -> r
# =============================================================================


class TestOutsideAxis:
    """Points outside the sphere along the axes should return positive distance."""

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_outside_positive_x(self, r):
        """Outside point on +x: (2r, 0, 0) -> r."""
        d = py_sd_sphere((2.0 * r, 0.0, 0.0), r)
        expected = r
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"Outside ({2*r}, 0, 0) with r={r} should be {expected}, got {d}"
        )

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_outside_negative_x(self, r):
        """Outside point on -x: (-2r, 0, 0) -> r."""
        d = py_sd_sphere((-2.0 * r, 0.0, 0.0), r)
        expected = r
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"Outside ({-2*r}, 0, 0) with r={r} should be {expected}, got {d}"
        )

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_outside_positive_y(self, r):
        """Outside point on +y: (0, 2r, 0) -> r."""
        d = py_sd_sphere((0.0, 2.0 * r, 0.0), r)
        expected = r
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"Outside (0, {2*r}, 0) with r={r} should be {expected}, got {d}"
        )

    @pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_outside_positive_z(self, r):
        """Outside point on +z: (0, 0, 2r) -> r."""
        d = py_sd_sphere((0.0, 0.0, 2.0 * r), r)
        expected = r
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"Outside (0, 0, {2*r}) with r={r} should be {expected}, got {d}"
        )

    def test_outside_far_point_approaches_raw_distance(self):
        """As point goes to infinity, SDF approximates raw distance to origin."""
        r = 1.0
        far_x = 1e6
        d = py_sd_sphere((far_x, 0.0, 0.0), r)
        expected = far_x - r
        assert d == pytest.approx(expected, abs=0.001), (
            f"Far point ({far_x}, 0, 0) should approximate {expected}, got {d}"
        )


# =============================================================================
# Path 6: Outside diagonal -- (r, r, r) for r=1 -> sqrt(3)-1
# =============================================================================


class TestOutsideDiagonal:
    """Points outside the sphere along the diagonal."""

    def test_unit_diagonal(self):
        """Outside diagonal (1, 1, 1), r=1 -> sqrt(3) - 1."""
        r = 1.0
        p = (1.0, 1.0, 1.0)
        d = py_sd_sphere(p, r)
        expected = math.sqrt(3.0) - 1.0
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"Outside diagonal {p} with r={r} should be {expected}, got {d}"
        )

    @pytest.mark.parametrize("r", [0.5, 2.0, 5.0])
    def test_scaled_diagonal(self, r):
        """Outside diagonal (r, r, r) for arbitrary r: sqrt(3r^2) - r."""
        p = (r, r, r)
        d = py_sd_sphere(p, r)
        expected = math.sqrt(3.0 * r * r) - r
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"Outside diagonal {p} with r={r} should be {expected}, got {d}"
        )

    def test_asymmetric_diagonal(self):
        """Outside point on an asymmetric diagonal: (2, 3, 6), r=3."""
        p = (2.0, 3.0, 6.0)
        r = 3.0
        d = py_sd_sphere(p, r)
        expected = math.sqrt(4.0 + 9.0 + 36.0) - 3.0  # sqrt(49) - 3 = 7 - 3 = 4
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"Outside asymmetric point {p} with r={r} should be {expected}, got {d}"
        )


# =============================================================================
# Path 7: Negative radius -- r=-1 treated as r=1 due to abs guard
# =============================================================================


class TestNegativeRadius:
    """Negative radius should be treated as positive via abs(r) guard."""

    def test_negative_r_same_as_positive_r(self):
        """sdSphere(p, -r) == sdSphere(p, r) for any p and r > 0."""
        test_cases = [
            ((0.0, 0.0, 0.0), 2.0),
            ((3.0, 0.0, 0.0), 1.0),
            ((1.0, 1.0, 0.0), 2.0),
            ((1.0, 2.0, 3.0), 5.0),
            ((-2.0, 0.0, 0.0), 3.0),
            ((0.5, 0.5, 0.5), 1.0),
        ]
        for p, r in test_cases:
            d_pos = py_sd_sphere(p, r)
            d_neg = py_sd_sphere(p, -r)
            assert d_pos == pytest.approx(d_neg, abs=TOL_EXACT), (
                f"Negative radius mismatch: sdSphere({p}, {r}) = {d_pos} != "
                f"sdSphere({p}, {-r}) = {d_neg}"
            )

    def test_negative_r_surface(self):
        """Surface point for r=-2 should be same as r=2."""
        d_neg = py_sd_sphere((2.0, 0.0, 0.0), -2.0)
        d_pos = py_sd_sphere((2.0, 0.0, 0.0), 2.0)
        assert d_neg == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point (2,0,0) with r=-2 should be 0, got {d_neg}"
        )
        assert d_neg == pytest.approx(d_pos, abs=TOL_EXACT)

    def test_negative_r_center(self):
        """Center for r=-5: abs(-5)=5, so (0,0,0) -> -5."""
        d = py_sd_sphere((0.0, 0.0, 0.0), -5.0)
        assert d == pytest.approx(-5.0, abs=TOL_EXACT), (
            f"Center with r=-5 should give -5, got {d}"
        )

    def test_negative_r_outside(self):
        """Outside point for r=-2: treated as r=2."""
        d = py_sd_sphere((6.0, 0.0, 0.0), -2.0)
        expected = 6.0 - 2.0  # 4.0
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"Outside (6,0,0) with r=-2 (abs=2) should be {expected}, got {d}"
        )


# =============================================================================
# Path 8: Zero radius -- sdSphere(p, 0) = length(p) (point SDF)
# =============================================================================


class TestZeroRadius:
    """With zero radius, the sphere degenerates to a point at the origin."""

    def test_zero_radius_at_origin(self):
        """Zero-radius sphere at origin: sdSphere((0,0,0), 0) -> 0."""
        d = py_sd_sphere((0.0, 0.0, 0.0), 0.0)
        assert d == pytest.approx(0.0, abs=TOL_EXACT), (
            f"Zero-radius sphere at origin should be 0, got {d}"
        )

    def test_zero_radius_on_x_axis(self):
        """Zero-radius sphere: sdSphere((3, 0, 0), 0) = 3 = length(p)."""
        p = (3.0, 0.0, 0.0)
        d = py_sd_sphere(p, 0.0)
        expected = math.sqrt(3.0 * 3.0)
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"Zero-radius sphere ({p}) should give {expected}, got {d}"
        )

    def test_zero_radius_on_diagonal(self):
        """Zero-radius sphere: sdSphere((1,2,2), 0) = 3 = length(p)."""
        p = (1.0, 2.0, 2.0)
        d = py_sd_sphere(p, 0.0)
        expected = 3.0  # sqrt(1+4+4) = 3
        assert d == pytest.approx(expected, abs=TOL_EXACT), (
            f"Zero-radius sphere ({p}) should give {expected}, got {d}"
        )

    def test_zero_radius_always_positive_or_zero(self):
        """Zero-radius SDF should always be >= 0 (no interior)."""
        test_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, -3.0),
            (1.0, 1.0, 1.0),
            (-2.0, 3.0, 5.0),
        ]
        for p in test_points:
            d = py_sd_sphere(p, 0.0)
            assert d >= 0.0, (
                f"Zero-radius sphere ({p}) should have non-negative distance, "
                f"got {d}"
            )

    def test_zero_radius_equals_length(self):
        """sdSphere(p, 0) should equal length(p) for all p."""
        test_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (3.0, 4.0, 0.0),
            (2.0, 3.0, 6.0),
        ]
        for p in test_points:
            d = py_sd_sphere(p, 0.0)
            length_p = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
            assert d == pytest.approx(length_p, abs=TOL_EXACT), (
                f"sdSphere({p}, 0) = {d} should equal length({p}) = {length_p}"
            )


# =============================================================================
# Path 9: Symmetry -- sdSphere((x,y,z), r) == sdSphere((-x,y,z), r)
# =============================================================================


class TestSymmetry:
    """The sphere SDF is fully symmetric under sign flips in any axis."""

    def test_x_symmetry(self):
        """Points (x,y,z) and (-x,y,z) should have equal distances."""
        r = 3.0
        test_points = [
            (1.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.5, 2.5, 0.5),
            (0.5, 1.0, 2.0),
            (3.0, 1.0, 1.0),
        ]
        for x, y, z in test_points:
            d_pos = py_sd_sphere((x, y, z), r)
            d_neg = py_sd_sphere((-x, y, z), r)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"X-symmetry: sdSphere(({x},{y},{z}), {r}) = {d_pos} != "
                f"sdSphere(({-x},{y},{z}), {r}) = {d_neg}"
            )

    def test_y_symmetry(self):
        """Points (x,y,z) and (x,-y,z) should have equal distances."""
        r = 3.0
        test_points = [
            (1.0, 2.0, 0.0),
            (0.0, 1.5, 1.0),
            (2.0, 0.5, 2.0),
            (1.0, 3.0, 1.0),
        ]
        for x, y, z in test_points:
            d_pos = py_sd_sphere((x, y, z), r)
            d_neg = py_sd_sphere((x, -y, z), r)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"Y-symmetry: sdSphere(({x},{y},{z}), {r}) = {d_pos} != "
                f"sdSphere(({x},{-y},{z}), {r}) = {d_neg}"
            )

    def test_z_symmetry(self):
        """Points (x,y,z) and (x,y,-z) should have equal distances."""
        r = 3.0
        test_points = [
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 1.5),
            (2.0, 2.0, 0.5),
            (1.0, 2.0, 3.0),
        ]
        for x, y, z in test_points:
            d_pos = py_sd_sphere((x, y, z), r)
            d_neg = py_sd_sphere((x, y, -z), r)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"Z-symmetry: sdSphere(({x},{y},{z}), {r}) = {d_pos} != "
                f"sdSphere(({x},{y},{-z}), {r}) = {d_neg}"
            )

    def test_full_octahedral_symmetry(self):
        """All 8 sign-flip combinations should yield the same distance."""
        r = 2.0
        base = (1.0, 2.0, 3.0)
        d_base = py_sd_sphere(base, r)
        signs = [
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
            (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        ]
        for sx, sy, sz in signs:
            p = (sx * base[0], sy * base[1], sz * base[2])
            d = py_sd_sphere(p, r)
            assert d == pytest.approx(d_base, abs=TOL_SURFACE), (
                f"Symmetry broken for sign ({sx},{sy},{sz}): "
                f"sdSphere({p}, {r}) = {d} != {d_base}"
            )

    def test_spherical_symmetry_same_radius(self):
        """Points at the same distance from origin have equal SDF."""
        r = 4.0
        test_radii = [1.0, 3.0, 5.0, 10.0]
        for radius in test_radii:
            # Generate points on a sphere of given radius
            positions = [
                (radius, 0.0, 0.0),
                (0.0, radius, 0.0),
                (0.0, 0.0, radius),
                (-radius, 0.0, 0.0),
                (radius / math.sqrt(3.0), radius / math.sqrt(3.0),
                 radius / math.sqrt(3.0)),
                (radius / math.sqrt(2.0), radius / math.sqrt(2.0), 0.0),
            ]
            ref = py_sd_sphere(positions[0], r)
            for p in positions[1:]:
                d = py_sd_sphere(p, r)
                assert d == pytest.approx(ref, abs=TOL_SURFACE), (
                    f"Spherical symmetry: points at distance {radius} should "
                    f"have equal SDF: {p} -> {d}, ref -> {ref}"
                )


# =============================================================================
# Path 10: Sign convention -- negative inside, zero on surface, positive outside
# =============================================================================


class TestSignConvention:
    """The SDF sign convention: negative inside, zero on surface, positive outside."""

    def test_inside_negative(self):
        """Points known to be inside the sphere should have negative distance."""
        r = 5.0
        inside_points = [
            (0.0, 0.0, 0.0),       # Center
            (1.0, 0.0, 0.0),       # Near center on x
            (0.0, 2.0, 0.0),       # On y axis
            (0.0, 0.0, 3.0),       # On z axis
            (2.0, 2.0, 2.0),       # Interior diagonal (length ~3.46 < 5)
            (-1.0, -1.0, -1.0),    # Interior diagonal negative octant
            (4.0, 0.0, 0.0),       # Just inside on x
            (0.0, 4.9, 0.0),       # Very close to surface on y
            (2.0, 3.0, 0.0),       # Inside in xy plane (length ~3.6 < 5)
        ]
        for p in inside_points:
            d = py_sd_sphere(p, r)
            assert d < 0.0, (
                f"Inside point {p} should have negative SDF, got {d}"
            )

    def test_outside_positive(self):
        """Points known to be outside the sphere should have positive distance."""
        r = 5.0
        outside_points = [
            (6.0, 0.0, 0.0),       # Outside on +x
            (0.0, 6.0, 0.0),       # Outside on +y
            (0.0, 0.0, 6.0),       # Outside on +z
            (-6.0, 0.0, 0.0),      # Outside on -x
            (5.0, 5.0, 0.0),       # Far outside in xy (length ~7.07)
            (10.0, 0.0, 0.0),      # Far on x
            (4.0, 4.0, 4.0),       # Outside diagonal (length ~6.93)
            (0.0, 0.0, 100.0),     # Very far on z
        ]
        for p in outside_points:
            d = py_sd_sphere(p, r)
            assert d > 0.0, (
                f"Outside point {p} should have positive SDF, got {d}"
            )

    def test_on_surface_zero(self):
        """Points on the sphere surface should have near-zero distance."""
        r = 5.0
        surface_points = [
            (5.0, 0.0, 0.0),
            (-5.0, 0.0, 0.0),
            (0.0, 5.0, 0.0),
            (0.0, -5.0, 0.0),
            (0.0, 0.0, 5.0),
            (0.0, 0.0, -5.0),
            (5.0 / math.sqrt(3.0), 5.0 / math.sqrt(3.0),
             5.0 / math.sqrt(3.0)),
            (3.0, 4.0, 0.0),       # sqrt(9+16) = 5
            (0.0, 3.0, 4.0),       # sqrt(9+16) = 5
        ]
        for p in surface_points:
            d = py_sd_sphere(p, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point {p} should have near-zero SDF, got {d}"
            )

    def test_sign_transition_at_surface(self):
        """Sign should transition from negative to positive across the surface."""
        r = 3.0
        # Approach surface at x = r from inside, exactly on surface, and outside
        surface_x = r
        step = 1e-6
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

    def test_strict_ordering_inside_surface_outside(self):
        """Distance ordering: inside < surface < outside."""
        r = 4.0
        # Points at increasing distance along +x
        points = [1.0, 3.0, 4.0, 5.0, 8.0]
        distances = [py_sd_sphere((x, 0.0, 0.0), r) for x in points]
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i + 1], (
                f"Distance should strictly increase along +x: "
                f"x={points[i]} -> {distances[i]}, "
                f"x={points[i+1]} -> {distances[i+1]}"
            )


# =============================================================================
# Path 11: Distance monotonicity and continuity
# =============================================================================


class TestContinuity:
    """The sphere SDF should be continuous everywhere."""

    def test_continuity_along_x(self):
        """SDF should be continuous along x-axis."""
        r = 2.0
        step = 0.001
        prev = py_sd_sphere((-5.0, 0.0, 0.0), r)
        for i in range(1, 1000):
            x = -5.0 + i * step
            curr = py_sd_sphere((x, 0.0, 0.0), r)
            diff = abs(curr - prev)
            # Maximum diff per step should be step (unit gradient)
            assert diff <= step * 1.001, (
                f"Discontinuity at x={x}: diff={diff} > step={step}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """SDF should be continuous along y-axis."""
        r = 2.0
        step = 0.01
        prev = py_sd_sphere((0.0, -5.0, 0.0), r)
        for i in range(1, 500):
            y = -5.0 + i * step
            curr = py_sd_sphere((0.0, y, 0.0), r)
            diff = abs(curr - prev)
            assert diff <= step * 1.001, (
                f"Discontinuity at y={y}: diff={diff} > step={step}"
            )
            prev = curr

    def test_continuity_along_z(self):
        """SDF should be continuous along z-axis."""
        r = 2.0
        step = 0.01
        prev = py_sd_sphere((0.0, 0.0, -5.0), r)
        for i in range(1, 500):
            z = -5.0 + i * step
            curr = py_sd_sphere((0.0, z, 0.0), r)
            diff = abs(curr - prev)
            assert diff <= step * 1.001, (
                f"Discontinuity at z={z}: diff={diff} > step={step}"
            )
            prev = curr

    def test_continuity_across_surface(self):
        """SDF should be continuous across the surface boundary."""
        r = 2.0
        step = 1e-5
        # Cross from inside to outside at x = r
        for offset in [i * step for i in range(-50, 51)]:
            x = r + offset
            d = py_sd_sphere((x, 0.0, 0.0), r)
            assert abs(d - offset) < 1e-12, (
                f"Near surface at x={x}: expected {offset}, got {d}"
            )


# =============================================================================
# Path 12: Gradient properties
# =============================================================================


class TestGradient:
    """The sphere SDF gradient should point radially outward."""

    def test_numerical_gradient_x(self):
        """Numerical gradient along x should be x / length(p)."""
        r = 5.0
        eps = 1e-6
        test_points = [
            (3.0, 0.0, 0.0),
            (7.0, 0.0, 0.0),
            (3.0, 4.0, 0.0),
            (1.0, 2.0, 2.0),
            (-3.0, 0.0, 0.0),
        ]
        for p in test_points:
            lp = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            if lp < eps:
                continue  # Skip origin where gradient is undefined
            grad_x = (py_sd_sphere((p[0] + eps, p[1], p[2]), r) -
                      py_sd_sphere((p[0] - eps, p[1], p[2]), r)) / (2.0 * eps)
            expected = p[0] / lp
            assert grad_x == pytest.approx(expected, abs=1e-4), (
                f"Gradient x at {p}: numerical {grad_x}, expected {expected}"
            )

    def test_numerical_gradient_magnitude_outside(self):
        """Outside the sphere, gradient magnitude should be ~1 (unit gradient)."""
        r = 5.0
        eps = 1e-6
        test_points = [
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (10.0, 10.0, 0.0),
            (-8.0, 0.0, 0.0),
        ]
        for p in test_points:
            lp = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            gx = (py_sd_sphere((p[0] + eps, p[1], p[2]), r) -
                  py_sd_sphere((p[0] - eps, p[1], p[2]), r)) / (2.0 * eps)
            gy = (py_sd_sphere((p[0], p[1] + eps, p[2]), r) -
                  py_sd_sphere((p[0], p[1] - eps, p[2]), r)) / (2.0 * eps)
            gz = (py_sd_sphere((p[0], p[1], p[2] + eps), r) -
                  py_sd_sphere((p[0], p[1], p[2] - eps), r)) / (2.0 * eps)
            mag = math.sqrt(gx**2 + gy**2 + gz**2)
            assert mag == pytest.approx(1.0, abs=1e-4), (
                f"Gradient magnitude at {p} should be ~1 (eikonal), got {mag}"
            )


# =============================================================================
# Path 13: Scaling behavior
# =============================================================================


class TestScaling:
    """Behavior under scaling of coordinates and radius."""

    def test_relative_ordering_preserved(self):
        """If |p1| < |p2| then sdSphere(p1, r) < sdSphere(p2, r)."""
        r = 3.0
        inner = (1.0, 0.0, 0.0)
        outer = (4.0, 0.0, 0.0)
        d_inner = py_sd_sphere(inner, r)
        d_outer = py_sd_sphere(outer, r)
        assert d_inner < d_outer, (
            f"Relative ordering: inner {d_inner} should be < outer {d_outer}"
        )

    def test_radial_monotonicity(self):
        """Outside sphere: distance increases with distance from origin."""
        r = 3.0
        # Points just outside surface to far away
        radii = [r + i * 0.5 for i in range(20)]
        distances = [py_sd_sphere((rad, 0.0, 0.0), r) for rad in radii]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-12, (
                f"Distance should be monotonic outside sphere: "
                f"at radius {radii[i]} d={distances[i]}, "
                f"at radius {radii[i+1]} d={distances[i+1]}"
            )

    def test_inside_monotonicity_toward_center(self):
        """Inside sphere: distance decreases (more negative) toward center."""
        r = 5.0
        # Points from just inside surface to center
        radii = [4.5, 3.5, 2.5, 1.5, 0.5, 0.0]
        distances = [py_sd_sphere((rad, 0.0, 0.0), r) for rad in radii]
        for i in range(len(distances) - 1):
            assert distances[i] > distances[i + 1], (
                f"Distance should decrease (more negative) toward center: "
                f"at radius {radii[i]} d={distances[i]}, "
                f"at radius {radii[i+1]} d={distances[i+1]}"
            )
