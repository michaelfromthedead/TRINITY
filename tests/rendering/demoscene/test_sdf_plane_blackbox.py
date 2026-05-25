"""
Blackbox tests for sdPlane WGSL signed-distance-to-plane function (T-DEMO-1.6).

Tests a Python reference model of the WGSL sdPlane function, verifying:
  - Correct distance on the plane (zero)
  - Positive distance above the plane
  - Negative distance below the plane
  - Plane offset
  - Axis-aligned and diagonal normals
  - Sign convention (above = positive, below = negative)
  - Continuity across the plane boundary
  - Distance linearity (doubling distance doubles result)
  - Unnormalized normal handling
  - Zero normal edge case

BLACKBOX approach: the reference model is built purely from the mathematical
definition of signed distance to a plane, without reference to any WGSL
implementation. The tests codify the contract that any correct sdPlane
implementation must satisfy.

Reference: Inigo Quilez -- SDF Primitives: sdPlane
https://iquilezles.org/articles/distfunctions/
"""
from __future__ import annotations

import math

import pytest

# =============================================================================
# Python reference model: signed distance to a plane
#
# Mathematically, the signed distance from point p to a plane with normal n
# and offset d is:
#
#   sdPlane(p, n, d) = dot(p, normalize(n)) + d
#
# The plane passes through the point at signed distance -d from the origin
# along the normal n. Points on the side of the plane that n points toward
# return positive distances; points on the opposite side return negative
# distances. Points exactly on the plane return zero.
# =============================================================================


def vec3_length(v) -> float:
    """Euclidean length of a 3D vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_normalize(v):
    """Normalize a 3D vector to unit length. Returns (0,0,0) if length is 0."""
    length = vec3_length(v)
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def vec3_dot(a, b) -> float:
    """Dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def py_sd_plane(p, n, d: float) -> float:
    """Reference Python sdPlane: signed distance from p to plane (n, d).

    Args:
        p: 3D query point (tuple of 3 floats).
        n: Plane normal (tuple of 3 floats); need not be unit length.
        d: Signed plane offset such that dot(p, normalize(n)) + d = 0
           defines the plane.

    Returns:
        Signed distance: positive in direction of n, negative opposite,
        zero on the plane.
    """
    nn = vec3_normalize(n)
    return vec3_dot(p, nn) + d


TOL = 1e-9


# =============================================================================
# Test: T-DEMO-1.6 -- Ground plane (y=0)
# =============================================================================


class TestGroundPlane:
    """Ground plane: n=(0,1,0), d=0. Plane is y=0."""

    def test_point_on_plane_returns_zero(self):
        """A point exactly on the plane should return zero distance."""
        result = py_sd_plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.0)
        assert result == pytest.approx(0.0, abs=TOL)

    def test_point_on_plane_off_origin(self):
        """Points on the plane away from origin should also be zero."""
        result = py_sd_plane((3.0, 0.0, -7.0), (0.0, 1.0, 0.0), 0.0)
        assert result == pytest.approx(0.0, abs=TOL)

    def test_point_above_plane_positive(self):
        """Points above the plane should return positive distance."""
        result = py_sd_plane((0.0, 5.0, 0.0), (0.0, 1.0, 0.0), 0.0)
        assert result == pytest.approx(5.0, abs=TOL)

    def test_point_below_plane_negative(self):
        """Points below the plane should return negative distance."""
        result = py_sd_plane((0.0, -3.0, 0.0), (0.0, 1.0, 0.0), 0.0)
        assert result == pytest.approx(-3.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.6 -- Plane with offset
# =============================================================================


class TestPlaneOffset:
    """Plane n=(0,1,0), d=2. Plane equation: y + 2 = 0, so plane is at y=-2."""

    def test_origin_is_above_offset_plane(self):
        """At origin, distance to plane at y=-2 with upward normal should be 2."""
        result = py_sd_plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.0)
        assert result == pytest.approx(2.0, abs=TOL)

    def test_point_on_offset_plane(self):
        """A point on the plane at y=-2 should return zero."""
        result = py_sd_plane((0.0, -2.0, 0.0), (0.0, 1.0, 0.0), 2.0)
        assert result == pytest.approx(0.0, abs=TOL)

    def test_below_offset_plane(self):
        """A point below the offset plane should return negative distance."""
        result = py_sd_plane((0.0, -5.0, 0.0), (0.0, 1.0, 0.0), 2.0)
        assert result == pytest.approx(-3.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.6 -- Vertical plane
# =============================================================================


class TestVerticalPlane:
    """Vertical plane: n=(1,0,0), d=0. Plane is x=0."""

    def test_positive_x_is_above(self):
        """Points on positive x side should be positive."""
        result = py_sd_plane((3.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0)
        assert result == pytest.approx(3.0, abs=TOL)

    def test_negative_x_is_below(self):
        """Points on negative x side should be negative."""
        result = py_sd_plane((-3.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0)
        assert result == pytest.approx(-3.0, abs=TOL)

    def test_on_plane_zero(self):
        """Points exactly on the vertical plane should be zero."""
        result = py_sd_plane((0.0, 4.0, -2.0), (1.0, 0.0, 0.0), 0.0)
        assert result == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.6 -- Diagonal / depth plane
# =============================================================================


class TestDiagonalPlane:
    """Diagonal plane: n=(0,0,1), d=0. Plane is z=0."""

    def test_positive_z_is_above(self):
        """Points on positive z side should be positive."""
        result = py_sd_plane((0.0, 0.0, 5.0), (0.0, 0.0, 1.0), 0.0)
        assert result == pytest.approx(5.0, abs=TOL)

    def test_negative_z_is_below(self):
        """Points on negative z side should be negative."""
        result = py_sd_plane((0.0, 0.0, -5.0), (0.0, 0.0, 1.0), 0.0)
        assert result == pytest.approx(-5.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.6 -- Sign convention
# =============================================================================


class TestSignConvention:
    """Above the plane (in the direction of n) must be positive everywhere."""

    def test_positive_direction_above_ground(self):
        """Ground plane: positive y is above, should be positive."""
        for y_pos in [0.1, 1.0, 10.0, 100.0]:
            d = py_sd_plane((0.0, y_pos, 0.0), (0.0, 1.0, 0.0), 0.0)
            assert d > 0.0, (
                f"Above ground plane at y={y_pos} should be positive, got {d}"
            )

    def test_negative_direction_below_ground(self):
        """Ground plane: negative y is below, should be negative."""
        for y_neg in [-0.1, -1.0, -10.0, -100.0]:
            d = py_sd_plane((0.0, y_neg, 0.0), (0.0, 1.0, 0.0), 0.0)
            assert d < 0.0, (
                f"Below ground plane at y={y_neg} should be negative, got {d}"
            )

    def test_sign_near_zero(self):
        """Very small positive/negative offsets should preserve sign."""
        eps = 1e-7
        d_above = py_sd_plane((0.0, eps, 0.0), (0.0, 1.0, 0.0), 0.0)
        assert d_above > 0.0, f"Above by {eps} should be positive, got {d_above}"
        d_below = py_sd_plane((0.0, -eps, 0.0), (0.0, 1.0, 0.0), 0.0)
        assert d_below < 0.0, f"Below by {eps} should be negative, got {d_below}"


# =============================================================================
# Test: T-DEMO-1.6 -- Continuity
# =============================================================================


class TestContinuity:
    """sdPlane must be continuous across the plane boundary."""

    def test_continuity_at_zero(self):
        """sdPlane((eps, 0, 0), (1,0,0), 0) should be continuous at eps=0."""
        eps_values = [1e-6, 1e-4, 1e-2, 0.0, -1e-2, -1e-4, -1e-6]
        results = [
            py_sd_plane((eps, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0)
            for eps in eps_values
        ]
        # Result should exactly equal eps (the x-coordinate)
        for eps, r in zip(eps_values, results):
            assert r == pytest.approx(eps, abs=TOL), (
                f"Expected sdPlane(({eps},0,0), (1,0,0), 0) = {eps}, got {r}"
            )

    def test_no_jump_across_plane(self):
        """Small step across the plane should produce smooth transition."""
        step = 1e-7
        above = py_sd_plane((0.0, step, 0.0), (0.0, 1.0, 0.0), 0.0)
        below = py_sd_plane((0.0, -step, 0.0), (0.0, 1.0, 0.0), 0.0)
        # The values should be antisymmetric and continuous
        assert above == pytest.approx(step, abs=TOL)
        assert below == pytest.approx(-step, abs=TOL)
        # The difference should be exactly 2 * step
        assert above - below == pytest.approx(2.0 * step, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.6 -- Distance linearity
# =============================================================================


class TestDistanceScaling:
    """Doubling the distance to the plane should double the signed distance."""

    def test_distance_doubles(self):
        """Moving from distance t to 2t should double the result."""
        n = (0.0, 1.0, 0.0)
        d = 0.0
        dist1 = py_sd_plane((0.0, 1.0, 0.0), n, d)
        dist2 = py_sd_plane((0.0, 2.0, 0.0), n, d)
        assert dist1 == pytest.approx(1.0, abs=TOL)
        assert dist2 == pytest.approx(2.0, abs=TOL)
        assert dist2 == pytest.approx(2.0 * dist1, abs=TOL)

    def test_distance_triples(self):
        """Moving from distance t to 3t should triple the result."""
        n = (0.0, 0.0, 1.0)
        d = 0.0
        dist1 = py_sd_plane((0.0, 0.0, 1.0), n, d)
        dist3 = py_sd_plane((0.0, 0.0, 3.0), n, d)
        assert dist1 == pytest.approx(1.0, abs=TOL)
        assert dist3 == pytest.approx(3.0, abs=TOL)
        assert dist3 == pytest.approx(3.0 * dist1, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.6 -- Unnormalized normal
# =============================================================================


class TestUnnormalizedNormal:
    """sdPlane should handle normals that are not unit-length."""

    def test_doubled_normal(self):
        """n=(0,2,0) is same direction as (0,1,0), sdPlane should normalize.

        Point (0,1,0) should give distance 1 after normalization.
        """
        result = py_sd_plane((0.0, 1.0, 0.0), (0.0, 2.0, 0.0), 0.0)
        assert result == pytest.approx(1.0, abs=TOL)

    def test_scaled_normal_consistency(self):
        """Scaling n by a positive factor should not change the result."""
        p = (3.0, 4.0, 5.0)
        d = -2.0
        result_original = py_sd_plane(p, (1.0, 0.0, 0.0), d)
        result_scaled = py_sd_plane(p, (5.0, 0.0, 0.0), d)
        assert result_original == pytest.approx(result_scaled, abs=TOL)

    def test_large_magnitude_normal(self):
        """Very large normal magnitude should still produce correct distance."""
        p = (0.0, 2.5, 0.0)
        result = py_sd_plane(p, (0.0, 1e6, 0.0), 0.0)
        assert result == pytest.approx(2.5, abs=TOL)

    def test_small_magnitude_normal(self):
        """Very small normal magnitude should still produce correct distance."""
        p = (0.0, 2.5, 0.0)
        result = py_sd_plane(p, (0.0, 1e-6, 0.0), 0.0)
        assert result == pytest.approx(2.5, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.6 -- Zero normal edge case
# =============================================================================


class TestZeroNormal:
    """When n is the zero vector, the concept of 'plane' is degenerate.

    With n=(0,0,0), normalize(n) = (0,0,0), so dot(p, (0,0,0)) + d = d.
    The function returns the offset d for any point p.
    """

    def test_zero_normal_returns_offset(self):
        """n=(0,0,0) should return d for any query point."""
        d = 5.0
        result = py_sd_plane((10.0, -20.0, 30.0), (0.0, 0.0, 0.0), d)
        assert result == pytest.approx(d, abs=TOL)

    def test_zero_normal_at_origin(self):
        """n=(0,0,0) at origin should also return d."""
        d = 5.0
        result = py_sd_plane((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), d)
        assert result == pytest.approx(d, abs=TOL)

    def test_zero_normal_negative_offset(self):
        """n=(0,0,0) with negative d should return negative d."""
        d = -3.0
        result = py_sd_plane((100.0, 200.0, 300.0), (0.0, 0.0, 0.0), d)
        assert result == pytest.approx(d, abs=TOL)

    def test_zero_normal_zero_offset(self):
        """n=(0,0,0) with d=0 should always return 0."""
        result = py_sd_plane((42.0, -17.0, 88.0), (0.0, 0.0, 0.0), 0.0)
        assert result == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.6 -- Arbitrary planes
# =============================================================================


class TestArbitraryPlanes:
    """Planes with arbitrary orientations."""

    def test_diagonal_45_degrees(self):
        """Plane at 45 degrees: n=(1,1,0)/sqrt(2), d=0.

        Point (1,0,0) has signed distance 1/sqrt(2) from this plane.
        """
        n = (1.0, 1.0, 0.0)
        result = py_sd_plane((1.0, 0.0, 0.0), n, 0.0)
        expected = 1.0 / math.sqrt(2.0)
        assert result == pytest.approx(expected, abs=TOL)

    def test_diagonal_45_negative_side(self):
        """Point on negative side of 45-degree plane."""
        n = (1.0, 1.0, 0.0)
        result = py_sd_plane((-1.0, 0.0, 0.0), n, 0.0)
        expected = -1.0 / math.sqrt(2.0)
        assert result == pytest.approx(expected, abs=TOL)

    def test_arbitrary_normal(self):
        """Test with an arbitrary non-axis-aligned normal."""
        n = (2.0, -3.0, 1.0)
        p = (4.0, -5.0, 6.0)
        d = 7.0
        result = py_sd_plane(p, n, d)
        # Expected: dot(p, normalize(n)) + d
        n_len = math.sqrt(2.0 * 2.0 + (-3.0) * (-3.0) + 1.0 * 1.0)
        expected = (4.0 * 2.0 + (-5.0) * (-3.0) + 6.0 * 1.0) / n_len + 7.0
        assert result == pytest.approx(expected, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.6 -- Determinism
# =============================================================================


class TestDeterminism:
    """sdPlane must be deterministic: same inputs produce same result."""

    def test_repeated_calls_identical(self):
        """Multiple calls with same args should produce identical result."""
        p = (1.234, 5.678, 9.012)
        n = (0.5, -0.5, 0.707)
        d = 3.141
        first = py_sd_plane(p, n, d)
        for _ in range(20):
            result = py_sd_plane(p, n, d)
            assert result == pytest.approx(first, abs=TOL)

    def test_all_inputs_used(self):
        """Different inputs should generally produce different results."""
        results = set()
        for i in range(10):
            p = (float(i), float(i * 2), float(i * 3))
            n = (1.0, 0.0, 0.0)
            d = 0.0
            results.add(round(py_sd_plane(p, n, d), 10))
        assert len(results) >= 9, (
            f"Expected at least 9 unique results from varying inputs, "
            f"got {len(results)}"
        )


# =============================================================================
# Test: T-DEMO-1.6 -- Negative d (offset)
# =============================================================================


class TestNegativeOffset:
    """Planes with negative offsets."""

    def test_plane_below_origin(self):
        """n=(0,1,0), d=-5 places the plane at y=5."""
        result = py_sd_plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -5.0)
        assert result == pytest.approx(-5.0, abs=TOL)

    def test_point_at_plane_with_negative_offset(self):
        """Point on the plane at y=5 should return 0."""
        result = py_sd_plane((0.0, 5.0, 0.0), (0.0, 1.0, 0.0), -5.0)
        assert result == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.6 -- Negative normal direction
# =============================================================================


class TestNegativeNormal:
    """Flipping the normal flips the sign convention."""

    def test_negative_normal_flips_sign(self):
        """n=(0,-1,0) should make points with positive y negative."""
        result = py_sd_plane((0.0, 5.0, 0.0), (0.0, -1.0, 0.0), 0.0)
        assert result == pytest.approx(-5.0, abs=TOL)

    def test_negative_normal_above_is_negative(self):
        """With n=(0,-1,0), the 'above' direction is negative y."""
        result = py_sd_plane((0.0, -3.0, 0.0), (0.0, -1.0, 0.0), 0.0)
        assert result == pytest.approx(3.0, abs=TOL)
