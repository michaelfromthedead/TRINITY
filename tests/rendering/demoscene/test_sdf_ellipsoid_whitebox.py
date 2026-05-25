"""
Whitebox tests for sdEllipsoid WGSL function (T-DEMO-1.8).

Tests a Python model of the WGSL implementation, verifying:
  - Formula decomposition: k0 = length(p/r), sd = (k0 - 1.0) * min(r)
  - Exact ellipsoid surface points yield distance ~0
  - Points inside/outside produce correct signed distances
  - Anisotropic scaling (r.x != r.y != r.z)
  - Degenerate cases (zero semi-axes -> plane or point)
  - Negative radii handled via abs(r)
  - Epsilon guard for numerical stability

WGSL implementation (589d951d):
    fn sdEllipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
        let eps = vec3<f32>(1e-10);
        let safe_r = max(abs(r), eps);
        let k0 = length(p / safe_r);
        let min_r = min(min(safe_r.x, safe_r.y), safe_r.z);
        return (k0 - 1.0) * min_r;
    }

IQ reference formula: (length(p/r) - 1.0) * min(r)
Note: IQ ellipsoid is NOT a true SDF -- gradient != 1 for anisotropic r.
      Surface points correctly return 0, sign convention holds,
      but off-axis distance values are not exact Euclidean distances.

WHITEBOX coverage plan (90 tests):
  Path A:  k0 = length(p/r) normalized distance computation      (8 tests)
  Path B:  Full IQ formula: (k0 - 1.0) * min(r)                 (10 tests)
  Path C:  Exact ellipsoid surface points                        (10 tests)
  Path D:  Inside points (negative distance)                     (9 tests)
  Path E:  Outside points (positive distance)                    (9 tests)
  Path F:  Anisotropic scaling                                   (9 tests)
  Path G:  Degenerate cases (zero semi-axes)                     (9 tests)
  Path H:  Negative radii via abs(r)                             (8 tests)
  Path I:  Epsilon guard for tiny/near-zero values               (9 tests)
  Path J:  min(r) selection logic                                (9 tests)
"""

from __future__ import annotations

import math

import pytest

# =============================================================================
# Python model matching WGSL semantics
# =============================================================================


def py_sd_ellipsoid(p, r):
    """Model of WGSL sdEllipsoid: signed distance to ellipsoid.

    Formula: (length(p/r) - 1.0) * min(r)
    Uses abs(r) with epsilon guard matching WGSL implementation exactly.
    """
    eps = 1e-10
    safe_rx = max(abs(r[0]), eps)
    safe_ry = max(abs(r[1]), eps)
    safe_rz = max(abs(r[2]), eps)
    k0 = math.sqrt(
        (p[0] / safe_rx) ** 2 + (p[1] / safe_ry) ** 2 + (p[2] / safe_rz) ** 2
    )
    min_r = min(safe_rx, safe_ry, safe_rz)
    return (k0 - 1.0) * min_r


def py_normalized_distance(p, r):
    """Extract the k0 = length(p/r) sub-expression for isolated testing."""
    eps = 1e-10
    safe_rx = max(abs(r[0]), eps)
    safe_ry = max(abs(r[1]), eps)
    safe_rz = max(abs(r[2]), eps)
    return math.sqrt(
        (p[0] / safe_rx) ** 2 + (p[1] / safe_ry) ** 2 + (p[2] / safe_rz) ** 2
    )


TOL = 1e-12
TOL_SURFACE = 1e-9


# =============================================================================
# Path A: k0 = length(p/r) normalized distance computation (8 tests)
# =============================================================================


class TestNormalizedDistance:
    """Tests the sub-expression `length(p / safe_r)` in isolation."""

    def test_on_surface_x_axis(self):
        """When p = (r.x, 0, 0), k0 should be exactly 1.0."""
        r = (2.0, 1.0, 0.5)
        k0 = py_normalized_distance((r[0], 0.0, 0.0), r)
        assert k0 == pytest.approx(1.0, abs=TOL)

    def test_on_surface_y_axis(self):
        """When p = (0, r.y, 0), k0 should be exactly 1.0."""
        r = (2.0, 1.0, 0.5)
        k0 = py_normalized_distance((0.0, r[1], 0.0), r)
        assert k0 == pytest.approx(1.0, abs=TOL)

    def test_on_surface_z_axis(self):
        """When p = (0, 0, r.z), k0 should be exactly 1.0."""
        r = (2.0, 1.0, 0.5)
        k0 = py_normalized_distance((0.0, 0.0, r[2]), r)
        assert k0 == pytest.approx(1.0, abs=TOL)

    def test_at_origin_k0_is_zero(self):
        """At the origin, k0 should be exactly 0."""
        k0 = py_normalized_distance((0.0, 0.0, 0.0), (2.0, 1.0, 0.5))
        assert k0 == pytest.approx(0.0, abs=TOL)

    def test_inside_half_x(self):
        """When p = (r.x/2, 0, 0), k0 = 0.5 (inside)."""
        r = (2.0, 1.0, 0.5)
        k0 = py_normalized_distance((r[0] * 0.5, 0.0, 0.0), r)
        assert k0 == pytest.approx(0.5, abs=TOL)

    def test_outside_double_x(self):
        """When p = (2*r.x, 0, 0), k0 = 2.0 (outside)."""
        r = (2.0, 1.0, 0.5)
        k0 = py_normalized_distance((r[0] * 2.0, 0.0, 0.0), r)
        assert k0 == pytest.approx(2.0, abs=TOL)

    def test_diagonal_surface(self):
        """When p = r (diagonal), k0 = sqrt(3) ~ 1.732."""
        r = (2.0, 1.0, 0.5)
        k0 = py_normalized_distance(r, r)
        assert k0 == pytest.approx(math.sqrt(3.0), abs=TOL)

    def test_k0_scales_with_radial_distance(self):
        """k0 is proportional to radial distance along any axis."""
        r = (3.0, 2.0, 1.0)
        for s in [0.25, 0.5, 1.5, 2.0, 4.0]:
            k0 = py_normalized_distance((s * r[0], 0.0, 0.0), r)
            assert k0 == pytest.approx(s, abs=TOL)


# =============================================================================
# Path B: Full formula (k0 - 1.0) * min(r) (10 tests)
# =============================================================================


class TestFullFormula:
    """Tests the complete IQ ellipsoid SDF: (k0 - 1.0) * min(r)."""

    def test_sphere_surface_zero(self):
        """Sphere case: r=(s,s,s), surface point p=(s,0,0) -> d=0."""
        d = py_sd_ellipsoid((2.0, 0.0, 0.0), (2.0, 2.0, 2.0))
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_sphere_center_negative(self):
        """Sphere center: d(0,0,0) = -s."""
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))
        assert d == pytest.approx(-3.0, abs=TOL)

    def test_ellipsoid_center_negative_min_r(self):
        """Ellipsoid center: d = -min(r)."""
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), (3.0, 2.0, 1.0))
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_outside_positive(self):
        """A point clearly outside produces positive distance."""
        d = py_sd_ellipsoid((10.0, 0.0, 0.0), (2.0, 1.0, 0.5))
        assert d > 0.0

    def test_surface_along_y_axis(self):
        """Surface point along y-axis: p=(0, r.y, 0) -> d=0."""
        d = py_sd_ellipsoid((0.0, 2.0, 0.0), (3.0, 2.0, 1.0))
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_surface_along_z_axis(self):
        """Surface point along z-axis: p=(0, 0, r.z) -> d=0."""
        d = py_sd_ellipsoid((0.0, 0.0, 1.0), (3.0, 2.0, 1.0))
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    @pytest.mark.parametrize(
        "r, p, expected",
        [
            ((2.0, 2.0, 2.0), (4.0, 0.0, 0.0), 2.0),
            ((1.0, 1.0, 1.0), (3.0, 0.0, 0.0), 2.0),
        ],
    )
    def test_sphere_known_values(self, r, p, expected):
        """Sphere case produces exact SDF (true SDF for isotropic)."""
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(expected, abs=TOL)

    @pytest.mark.parametrize(
        "r, p, expected_d",
        [
            ((3.0, 2.0, 1.0), (1.5, 0.0, 0.0), -0.5),
            ((3.0, 2.0, 1.0), (4.5, 0.0, 0.0), 0.5),
        ],
    )
    def test_ellipsoid_known_values(self, r, p, expected_d):
        """Known distance computations for anisotropic ellipsoid."""
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(expected_d, abs=TOL)


# =============================================================================
# Path C: Exact ellipsoid surface points (10 tests)
# =============================================================================


class TestExactSurface:
    """Points on the ellipsoid surface should yield a signed distance of ~0."""

    @pytest.mark.parametrize(
        "r, p",
        [
            ((2.0, 2.0, 2.0), (2.0, 0.0, 0.0)),
            ((2.0, 2.0, 2.0), (0.0, 2.0, 0.0)),
            ((2.0, 2.0, 2.0), (0.0, 0.0, -2.0)),
        ],
    )
    def test_sphere_surface(self, r, p):
        """Sphere surface points should have distance ~0."""
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    @pytest.mark.parametrize(
        "r, p",
        [
            ((3.0, 2.0, 1.0), (3.0, 0.0, 0.0)),
            ((3.0, 2.0, 1.0), (0.0, 2.0, 0.0)),
            ((3.0, 2.0, 1.0), (0.0, 0.0, 1.0)),
            ((3.0, 2.0, 1.0), (-3.0, 0.0, 0.0)),
        ],
    )
    def test_ellipsoid_surface_axes(self, r, p):
        """Ellipsoid surface points along each axis should have distance ~0."""
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    @pytest.mark.parametrize(
        "r, p",
        [
            ((1.0, 0.5, 0.25), (1.0, 0.0, 0.0)),
            ((0.5, 0.1, 4.0), (0.0, 0.0, 4.0)),
            ((4.0, 0.5, 0.1), (4.0, 0.0, 0.0)),
        ],
    )
    def test_various_ellipsoid_surfaces(self, r, p):
        """Surface verification for various ellipsoid parameter sets."""
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)


# =============================================================================
# Path D: Inside points (negative distance) (9 tests)
# =============================================================================


class TestInsidePoints:
    """Points inside the ellipsoid should produce negative signed distance."""

    def test_center_negative_min_r(self):
        """Center offset: d = -min(r)."""
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), (4.0, 2.0, 1.0))
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_halfway_along_x_negative(self):
        """Half-radius along x should be negative."""
        d = py_sd_ellipsoid((2.0, 0.0, 0.0), (4.0, 2.0, 1.0))
        assert d < 0.0

    @pytest.mark.parametrize(
        "r, p, expected",
        [
            ((5.0, 3.0, 1.0), (0.0, 0.0, 0.0), -1.0),
            ((10.0, 5.0, 2.0), (0.0, 0.0, 0.0), -2.0),
            ((1.5, 0.5, 0.25), (0.0, 0.0, 0.0), -0.25),
        ],
    )
    def test_center_various_r(self, r, p, expected):
        """Center distance = -min(r) for various r."""
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(expected, abs=TOL)

    @pytest.mark.parametrize(
        "r, p_frac",
        [
            ((3.0, 2.0, 1.0), 0.25),
            ((3.0, 2.0, 1.0), 0.5),
            ((3.0, 2.0, 1.0), 0.75),
            ((3.0, 2.0, 1.0), 0.9),
        ],
    )
    def test_inside_various_depths(self, r, p_frac):
        """Points at various fractional depths should all be negative."""
        d = py_sd_ellipsoid((r[0] * p_frac, 0.0, 0.0), r)
        assert d < 0.0


# =============================================================================
# Path E: Outside points (positive distance) (9 tests)
# =============================================================================


class TestOutsidePoints:
    """Points outside the ellipsoid should produce positive signed distance."""

    @pytest.mark.parametrize(
        "r, p",
        [
            ((2.0, 1.0, 0.5), (4.0, 0.0, 0.0)),
            ((2.0, 1.0, 0.5), (0.0, 2.0, 0.0)),
            ((2.0, 1.0, 0.5), (0.0, 0.0, 1.0)),
        ],
    )
    def test_double_radius_positive(self, r, p):
        """Points at 2x semi-axis length should be clearly outside."""
        d = py_sd_ellipsoid(p, r)
        assert d > 0.0

    @pytest.mark.parametrize(
        "r, scale",
        [
            ((3.0, 2.0, 1.0), 1.5),
            ((3.0, 2.0, 1.0), 2.0),
            ((3.0, 2.0, 1.0), 5.0),
            ((0.5, 0.3, 0.1), 3.0),
        ],
    )
    def test_outside_various_scales(self, r, scale):
        """Points at various scales outside the ellipsoid."""
        d = py_sd_ellipsoid((r[0] * scale, 0.0, 0.0), r)
        assert d > 0.0

    def test_diagonal_outside_positive(self):
        """Diagonal point outside the ellipsoid should be positive."""
        d = py_sd_ellipsoid((3.0, 2.0, 1.0), (2.0, 1.0, 0.5))
        assert d > 0.0

    def test_just_beyond_surface_positive(self):
        """Point just beyond the surface should be positive."""
        d = py_sd_ellipsoid((2.0 + 1e-4, 0.0, 0.0), (2.0, 1.0, 0.5))
        assert d > 0.0


# =============================================================================
# Path F: Anisotropic scaling (r.x != r.y != r.z) (9 tests)
# =============================================================================


class TestAnisotropicScaling:
    """Behavior when ellipsoid semi-axes are all different lengths."""

    @pytest.mark.parametrize(
        "r, p",
        [
            ((5.0, 1.0, 0.5), (5.0, 0.0, 0.0)),
            ((1.0, 5.0, 0.5), (0.0, 5.0, 0.0)),
        ],
    )
    def test_stretched_axis_surface(self, r, p):
        """Surface verification for ellipsoids stretched along one axis."""
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    @pytest.mark.parametrize(
        "r, p",
        [
            ((5.0, 1.0, 0.5), (2.5, 0.0, 0.0)),
            ((1.0, 0.5, 5.0), (0.0, 0.0, 2.5)),
        ],
    )
    def test_stretched_axis_inside(self, r, p):
        """Points inside stretched ellipsoid are negative."""
        d = py_sd_ellipsoid(p, r)
        assert d < 0.0

    @pytest.mark.parametrize(
        "r, p",
        [
            ((5.0, 1.0, 0.5), (10.0, 0.0, 0.0)),
            ((1.0, 0.5, 5.0), (0.0, 0.0, 10.0)),
        ],
    )
    def test_stretched_axis_outside(self, r, p):
        """Points outside stretched ellipsoid are positive."""
        d = py_sd_ellipsoid(p, r)
        assert d > 0.0

    @pytest.mark.parametrize(
        "r",
        [
            (4.0, 2.0, 0.5),
            (2.0, 4.0, 0.5),
            (0.5, 4.0, 2.0),
        ],
    )
    def test_asymmetric_center_distance(self, r):
        """Center distance always equals -min(r) regardless of asymmetry."""
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
        assert d == pytest.approx(-min(r), abs=TOL)


# =============================================================================
# Path G: Degenerate cases (zero semi-axes) (9 tests)
# =============================================================================


class TestDegenerateCases:
    """When one or more semi-axes are zero, the ellipsoid degenerates."""

    @pytest.mark.parametrize(
        "r, p",
        [
            ((0.0, 2.0, 1.0), (2.0, 0.0, 0.0)),
            ((2.0, 1.0, 0.0), (0.0, 0.0, 2.0)),
        ],
    )
    def test_one_zero_axis_outside(self, r, p):
        """Single zero axis: point along that axis is outside."""
        d = py_sd_ellipsoid(p, r)
        assert d > 0.0

    @pytest.mark.parametrize(
        "r, p, expected_approx",
        [
            ((0.0, 2.0, 1.0), (3.0, 0.0, 0.0), 3.0),
            ((0.0, 2.0, 1.0), (-5.0, 0.0, 0.0), 5.0),
            ((2.0, 1.0, 0.0), (0.0, 0.0, 3.0), 3.0),
        ],
    )
    def test_one_zero_axis_distance(self, r, p, expected_approx):
        """Single zero axis gives plane-like distance along that axis."""
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(expected_approx, abs=TOL_SURFACE)

    @pytest.mark.parametrize(
        "r, p",
        [
            ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
            ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ],
    )
    def test_two_zero_axes_at_origin(self, r, p):
        """Two zero axes at origin: distance = 0."""
        d = py_sd_ellipsoid(p, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_all_zero_at_origin(self):
        """All axes zero at origin: degenerate to point, distance = 0."""
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_all_zero_offset_positive(self):
        """All axes zero offset: degenerate to point, distance = length(p)."""
        d = py_sd_ellipsoid((3.0, 4.0, 0.0), (0.0, 0.0, 0.0))
        assert d == pytest.approx(5.0, abs=TOL_SURFACE)


# =============================================================================
# Path H: Negative radii via abs(r) (8 tests)
# =============================================================================


class TestNegativeRadii:
    """The implementation uses abs(r) to handle negative semi-axes."""

    @pytest.mark.parametrize(
        "r_pos",
        [
            (3.0, 2.0, 1.0),
            (5.0, 1.0, 0.5),
            (0.8, 0.6, 0.4),
        ],
    )
    def test_all_negative(self, r_pos):
        """All components negative equivalent to all positive."""
        r_neg = (-r_pos[0], -r_pos[1], -r_pos[2])
        d_pos = py_sd_ellipsoid((2.0, 1.0, 0.5), r_pos)
        d_neg = py_sd_ellipsoid((2.0, 1.0, 0.5), r_neg)
        assert d_pos == pytest.approx(d_neg, abs=TOL)

    @pytest.mark.parametrize(
        "r_pos, neg_idx",
        [
            ((3.0, 2.0, 1.0), 0),
            ((3.0, 2.0, 1.0), 2),
        ],
    )
    def test_single_negative(self, r_pos, neg_idx):
        """Single negative component equivalent to positive."""
        r_neg = list(r_pos)
        r_neg[neg_idx] = -r_neg[neg_idx]
        d_pos = py_sd_ellipsoid((4.0, 2.0, 1.0), r_pos)
        d_neg = py_sd_ellipsoid((4.0, 2.0, 1.0), tuple(r_neg))
        assert d_pos == pytest.approx(d_neg, abs=TOL)

    @pytest.mark.parametrize(
        "r_pos, r_mixed",
        [
            ((3.0, 2.0, 1.0), (-3.0, 2.0, -1.0)),
            ((3.0, 2.0, 1.0), (-3.0, -2.0, 1.0)),
        ],
    )
    def test_mixed_signs(self, r_pos, r_mixed):
        """Mixed sign components equivalent to all positive (via abs)."""
        d_pos = py_sd_ellipsoid((3.0, 1.5, 0.5), r_pos)
        d_mixed = py_sd_ellipsoid((3.0, 1.5, 0.5), r_mixed)
        assert d_pos == pytest.approx(d_mixed, abs=TOL)

    def test_negative_equivalent_on_surface(self):
        """Surface check: negative r still gives correct surface distance."""
        d = py_sd_ellipsoid((3.0, 0.0, 0.0), (-3.0, -2.0, -1.0))
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)


# =============================================================================
# Path I: Epsilon guard (9 tests)
# =============================================================================


class TestEpsilonGuard:
    """The epsilon guard prevents division by zero for near-zero semi-axes.

    The WGSL implementation uses max(abs(r), vec3<f32>(1e-10)) to ensure
    that division by safe_r is always well-defined.
    """

    @pytest.mark.parametrize(
        "tiny_val",
        [1e-12, 1e-15, 0.0, -1e-12, -1e-15],
    )
    def test_tiny_semi_axis_no_error(self, tiny_val):
        """Tiny or zero semi-axis values do not cause division errors."""
        d = py_sd_ellipsoid((3.0, 4.0, 5.0), (tiny_val, 2.0, 1.0))
        assert isinstance(d, float)
        assert math.isfinite(d)

    def test_tiny_all_axes(self):
        """All axes tiny produces sphere-like behavior (clamped to eps)."""
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), (1e-15, 2e-15, 3e-15))
        assert d == pytest.approx(-1e-10, abs=1e-12)

    def test_epsilon_guard_isolation(self):
        """Only one component is epsilon: behaves as plane."""
        d = py_sd_ellipsoid((5.0, 0.0, 0.0), (1e-10, 3.0, 4.0))
        assert d == pytest.approx(5.0, abs=1e-6)

    def test_mixed_tiny_and_normal(self):
        """Mix of tiny and normal axis lengths, finite result."""
        d = py_sd_ellipsoid((1.0, 2.0, 3.0), (1e-11, 3.0, 4.0))
        assert math.isfinite(d)

    def test_negative_tiny(self):
        """Negative tiny values also guarded (abs + epsilon)."""
        d = py_sd_ellipsoid((1.0, 0.0, 0.0), (-1e-12, 2.0, 3.0))
        assert math.isfinite(d)
        assert d == pytest.approx(1.0, abs=1e-4)


# =============================================================================
# Path J: min(r) selection logic (9 tests)
# =============================================================================


class TestMinRSelection:
    """The formula uses min(r) to scale the normalized distance."""

    @pytest.mark.parametrize(
        "r, expected_min",
        [
            ((3.0, 2.0, 1.0), 1.0),
            ((1.0, 3.0, 2.0), 1.0),
            ((2.0, 1.0, 3.0), 1.0),
            ((5.0, 5.0, 5.0), 5.0),
        ],
    )
    def test_min_r_value(self, r, expected_min):
        """min(r) selects the smallest positive component."""
        eps = 1e-10
        safe_r = (max(abs(r[0]), eps), max(abs(r[1]), eps), max(abs(r[2]), eps))
        assert min(safe_r) == pytest.approx(expected_min, abs=TOL)

    @pytest.mark.parametrize(
        "r",
        [
            (3.0, 2.0, 1.0),
            (1.0, 2.0, 3.0),
            (0.1, 5.0, 10.0),
        ],
    )
    def test_min_determines_center_distance(self, r):
        """Center distance always equals -min(r)."""
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), r)
        assert d == pytest.approx(-min(r), abs=TOL)

    def test_min_r_with_negative_component(self):
        """min(r) after abs for negative component picks correctly."""
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), (-3.0, -2.0, -1.0))
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_min_r_two_equal_smallest(self):
        """When two components tie for smallest, either is correct."""
        d = py_sd_ellipsoid((0.0, 0.0, 0.0), (1.0, 1.0, 3.0))
        assert d == pytest.approx(-1.0, abs=TOL)
