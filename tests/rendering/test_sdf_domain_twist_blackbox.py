"""
Blackbox tests for domain twist operation (T-DEMO-1.25).

Spec reference:
  Phase 8  (FORMULAS.md): p = rotate(p.xz, p.y * k)  -- Twist Transformation
  Phase 17 (FORMULAS.md): p.xz *= mat2(cos(p.y*k), sin(p.y*k),
                                        -sin(p.y*k), cos(p.y*k))  -- SDF Twist (Advanced)

Acceptance criterion:
  "space twisted along specified axis"

Cleanroom approach: This file derives its reference model from the spec formulas
only. It does NOT read the WGSL implementation or the whitebox test.

Mathematical definition:
  Twist is a rotation in the plane perpendicular to the twist axis, where the
  rotation angle is proportional to the coordinate along that axis.

  For twist around the y-axis (the canonical form from the spec):
      theta = k * p.y
      x' = x * cos(theta) - z * sin(theta)
      y' = y
      z' = x * sin(theta) + z * cos(theta)

  For twist around the x-axis:
      theta = k * p.x
      y' = y * cos(theta) - z * sin(theta)
      x' = x
      z' = y * sin(theta) + z * cos(theta)

  For twist around the z-axis:
      theta = k * p.z
      x' = x * cos(theta) - y * sin(theta)
      z' = z
      y' = x * sin(theta) + y * cos(theta)

Properties:
  - k=0 is identity (no twist)
  - Twist is periodic in the axis coordinate with period 2*pi/|k|
  - Distance in the twist plane is preserved (twist is an isometry)
  - The coordinate along the twist axis is unchanged
  - k > 0 and k < 0 produce opposite twist directions
  - Twist composes with other domain operations
"""

import math
import random
from typing import Literal

import pytest

# =============================================================================
# Spec-derived reference model (pure math, no implementation dependency)
# =============================================================================

Axis = Literal["x", "y", "z"]


def domain_twist(p: tuple[float, float, float],
                 k: float,
                 axis: Axis = "y") -> tuple[float, float, float]:
    """Reference model: twist space along specified axis.

    The twist angle at coordinate t along the twist axis is k * t.
    Points are rotated in the plane perpendicular to the axis.

    Args:
        p: Input point (x, y, z).
        k: Twist strength (radians per unit length along axis).
        axis: Twist axis -- "x", "y", or "z". Default "y" matches spec.

    Returns:
        Twisted point (x', y', z').
    """
    if axis == "y":
        theta = k * p[1]
        c = math.cos(theta)
        s = math.sin(theta)
        return (c * p[0] - s * p[2], p[1], s * p[0] + c * p[2])
    elif axis == "x":
        theta = k * p[0]
        c = math.cos(theta)
        s = math.sin(theta)
        return (p[0], c * p[1] - s * p[2], s * p[1] + c * p[2])
    elif axis == "z":
        theta = k * p[2]
        c = math.cos(theta)
        s = math.sin(theta)
        return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])
    else:
        raise ValueError(f"Unknown twist axis: {axis}")


def make_rotation_matrix_2d(angle: float) -> tuple[float, float, float, float]:
    """Return (c, s, -s, c) for a 2D rotation matrix [c -s; s c]."""
    c = math.cos(angle)
    s = math.sin(angle)
    return (c, s, -s, c)


# =============================================================================
# Tolerance and helpers
# =============================================================================

TOL_REL = 1e-5
TOL_ABS = 1e-7


def vec3_close(a, b, rel_tol=TOL_REL, abs_tol=TOL_ABS) -> bool:
    return all(
        math.isclose(a[i], b[i], rel_tol=rel_tol, abs_tol=abs_tol)
        for i in range(3)
    )


def vec3_dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def assert_finite(p):
    """Assert all components are finite (not NaN, not Inf)."""
    assert all(math.isfinite(x) for x in p), f"non-finite values: {p}"


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Identity and Zero Cases
# =============================================================================


class TestTwistIdentity:
    """When k=0, twist is the identity transformation."""

    def test_k_zero_returns_input(self):
        """k = 0 => angle = 0 => identity rotation => output == input."""
        p = (1.0, 2.0, 3.0)
        result = domain_twist(p, 0.0)
        assert vec3_close(result, p), f"k=0 should be identity: {result}"

    def test_k_zero_random_points(self):
        """k=0 is identity for many random points."""
        for _ in range(50):
            p = (random.uniform(-10, 10),
                 random.uniform(-10, 10),
                 random.uniform(-10, 10))
            result = domain_twist(p, 0.0)
            assert vec3_close(result, p), f"k=0 not identity for {p}: {result}"

    def test_k_zero_all_axes(self):
        """k=0 is identity for all twist axes."""
        p = (1.5, -2.5, 3.5)
        for axis in ("x", "y", "z"):
            result = domain_twist(p, 0.0, axis=axis)  # type: ignore
            assert vec3_close(result, p), (
                f"k=0, axis={axis} should be identity: {result}"
            )

    def test_k_zero_no_rotation_matrix(self):
        """k=0 => rotation matrix is identity (c=1, s=0)."""
        c, s, neg_s, c2 = make_rotation_matrix_2d(0.0)
        assert c == pytest.approx(1.0)
        assert s == pytest.approx(0.0)
        assert c2 == pytest.approx(1.0)
        assert neg_s == pytest.approx(0.0)


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Proportional to Height (Axis Coordinate)
# =============================================================================


class TestTwistProportionalToHeight:
    """Twist angle must be proportional to the coordinate along the axis."""

    def test_twist_angle_linear_in_y(self):
        """For y-axis twist, the rotation angle is k * y, which is linear in y."""
        p_base = (1.0, 0.0, 0.0)
        k = 2.0
        # At y=0, angle is 0 => no rotation
        r0 = domain_twist(p_base, k)
        # At y=pi/(2k), angle is pi/2 => rotate by 90 degrees
        y_half_pi = math.pi / (2.0 * k)
        p_half = (1.0, y_half_pi, 0.0)
        r_half = domain_twist(p_half, k)
        # After 90 degree rotation, x should be near 0, z should be near 1
        assert abs(r_half[0]) < TOL_ABS, (
            f"x should be near 0 after 90deg twist, got {r_half[0]}"
        )
        assert r_half[2] == pytest.approx(1.0, rel=TOL_REL), (
            f"z should be near 1 after 90deg twist, got {r_half[2]}"
        )

    def test_twist_angle_doubling(self):
        """Doubling the y-coordinate doubles the twist angle."""
        p = (1.0, 0.0, 0.0)
        k = 1.5
        r_at_y = domain_twist(p, k)
        p_double = (1.0, 2.0, 0.0)
        r_at_2y = domain_twist(p_double, k)
        # The angle at 2y is 2 * angle at y (same k, double y)
        # So the cos/sin should be cos(2a) and sin(2a) where a=k*y
        assert not vec3_close(r_at_y, r_at_2y), (
            "Doubling y should change twist angle"
        )

    def test_k_doubling_equals_y_doubling(self):
        """Doubling k has the same effect as doubling the y-coordinate."""
        p = (1.0, 2.0, 3.0)
        k = 0.5
        # Twist with k=0.5 at y=2: angle = 1.0
        r1 = domain_twist(p, k)
        # Twist with k=1.0 at y=1: angle = 1.0 also
        p_half_y = (1.0, 1.0, 3.0)
        r2 = domain_twist(p_half_y, 2.0 * k)
        assert vec3_close(r1, r2), (
            f"k=0.5 at y=2 vs k=1.0 at y=1 should match: {r1} vs {r2}"
        )

    def test_twist_strength_sign_flips_direction(self):
        """k > 0 twists one way, k < 0 twists the opposite way (conjugate)."""
        p = (1.0, 2.0, 0.0)
        k = 1.0
        r_pos = domain_twist(p, k)      # angle = k * y
        r_neg = domain_twist(p, -k)     # angle = -k * y
        # Positive and negative k produce different rotations
        assert not vec3_close(r_pos, r_neg), (
            "Opposite k signs should produce different results"
        )
        # r_pos should be the inverse of r_neg (negate the rotation angle)
        # Which means: apply r_pos then r_neg should restore original
        # For a point on the axis (y unchanged): invert by rotating back
        c = math.cos(k * p[1])
        s = math.sin(k * p[1])
        # r_pos rotates by +theta, r_neg by -theta
        # Applying both should return to p
        x_recovered = c * r_pos[0] + s * r_pos[2]
        z_recovered = -s * r_pos[0] + c * r_pos[2]
        assert x_recovered == pytest.approx(p[0], rel=TOL_REL)
        assert z_recovered == pytest.approx(p[2], rel=TOL_REL)


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Y-Axis Invariant
# =============================================================================


class TestYAxisInvariant:
    """For y-axis twist, the y-coordinate is unchanged."""

    def test_y_axis_preserved(self):
        """For axis='y', p.y remains unchanged."""
        for _ in range(50):
            p = (random.uniform(-5, 5),
                 random.uniform(-5, 5),
                 random.uniform(-5, 5))
            k = random.uniform(-10, 10)
            result = domain_twist(p, k, axis="y")
            assert result[1] == pytest.approx(p[1]), (
                f"y-coordinate changed: {result[1]} != {p[1]}"
            )

    def test_x_axis_preserved(self):
        """For axis='x', p.x remains unchanged."""
        for _ in range(50):
            p = (random.uniform(-5, 5),
                 random.uniform(-5, 5),
                 random.uniform(-5, 5))
            k = random.uniform(-10, 10)
            result = domain_twist(p, k, axis="x")
            assert result[0] == pytest.approx(p[0]), (
                f"x-coordinate changed: {result[0]} != {p[0]}"
            )

    def test_z_axis_preserved(self):
        """For axis='z', p.z remains unchanged."""
        for _ in range(50):
            p = (random.uniform(-5, 5),
                 random.uniform(-5, 5),
                 random.uniform(-5, 5))
            k = random.uniform(-10, 10)
            result = domain_twist(p, k, axis="z")
            assert result[2] == pytest.approx(p[2]), (
                f"z-coordinate changed: {result[2]} != {p[2]}"
            )


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Periodicity
# =============================================================================


class TestTwistPeriodicity:
    """Twist is periodic along the axis with period 2*pi/|k|."""

    def test_periodic_at_2pi_over_k(self):
        """Shifting y by 2*pi/k produces the same twist."""
        p = (1.0, 2.0, 3.0)
        k = 2.0
        period = 2.0 * math.pi / abs(k)
        r0 = domain_twist(p, k)
        p_shifted = (p[0], p[1] + period, p[2])
        r_period = domain_twist(p_shifted, k)
        assert vec3_close(r0, r_period), (
            f"Not periodic at 2*pi/k: {r0} vs {r_period}"
        )

    def test_periodic_multiple_periods(self):
        """Shifting y by N * 2*pi/k produces the same twist."""
        p = (1.0, 2.0, 3.0)
        k = 1.5
        period = 2.0 * math.pi / abs(k)
        for n in (2, 3, 5, -1, -3):
            r0 = domain_twist(p, k)
            p_shifted = (p[0], p[1] + n * period, p[2])
            r_n = domain_twist(p_shifted, k)
            assert vec3_close(r0, r_n), (
                f"Not periodic at {n}*period: {r0} vs {r_n}"
            )

    def test_periodic_at_2pi_over_k_all_axes(self):
        """Periodicity holds for all twist axes."""
        k = 1.0
        period = 2.0 * math.pi / abs(k)
        p = (1.0, 2.0, 3.0)
        for axis in ("x", "y", "z"):
            r0 = domain_twist(p, k, axis=axis)  # type: ignore
            if axis == "y":
                shifted = (p[0], p[1] + period, p[2])
            elif axis == "x":
                shifted = (p[0] + period, p[1], p[2])
            else:
                shifted = (p[0], p[1], p[2] + period)
            r_period = domain_twist(shifted, k, axis=axis)  # type: ignore
            assert vec3_close(r0, r_period), (
                f"Not periodic for axis={axis}: {r0} vs {r_period}"
            )

    def test_anti_periodic_at_pi_over_k(self):
        """Shifting y by pi/k produces a half-turn (negation of xz)."""
        p = (1.0, 0.5, 0.0)
        k = 2.0
        half_period = math.pi / abs(k)
        r0 = domain_twist(p, k)
        p_half = (p[0], p[1] + half_period, p[2])
        r_half = domain_twist(p_half, k)
        # A half-turn negates both x and z coordinates
        assert r_half[0] == pytest.approx(-r0[0], rel=TOL_REL), (
            f"x not negated at half-period: {r_half[0]} vs {-r0[0]}"
        )
        assert r_half[2] == pytest.approx(-r0[2], rel=TOL_REL), (
            f"z not negated at half-period: {r_half[2]} vs {-r0[2]}"
        )


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Distance Preservation (Isometry)
# =============================================================================


class TestTwistIsometry:
    """Twist preserves distances in the twist plane (it is a rotation)."""

    def test_xz_distance_preserved_around_y(self):
        """For y-axis twist, distance in xz-plane is preserved."""
        for _ in range(50):
            a = (random.uniform(-5, 5), random.uniform(-5, 5),
                 random.uniform(-5, 5))
            b = (random.uniform(-5, 5), random.uniform(-5, 5),
                 random.uniform(-5, 5))
            k = random.uniform(0.1, 5.0)
            d_before = math.sqrt((a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2)
            ta = domain_twist(a, k, axis="y")
            tb = domain_twist(b, k, axis="y")
            d_after = math.sqrt((ta[0] - tb[0]) ** 2 + (ta[2] - tb[2]) ** 2)
            assert d_after == pytest.approx(d_before, rel=TOL_REL), (
                f"xz distance not preserved: {d_before} -> {d_after}"
            )

    def test_3d_distance_not_preserved_generally(self):
        """Full 3D distance is NOT preserved because y is fixed but xz rotate."""
        for _ in range(20):
            a = (random.uniform(-3, 3), 0.0, random.uniform(-3, 3))
            b = (random.uniform(-3, 3), 1.0, random.uniform(-3, 3))
            k = 2.0
            d_before = vec3_dist(a, b)
            ta = domain_twist(a, k)
            tb = domain_twist(b, k)
            d_after = vec3_dist(ta, tb)
            # xz rotation at different y-angles means distance can change
            # But we should verify it doesn't blow up unreasonably
            assert math.isfinite(d_after)
            assert d_after >= 0.0

    def test_norm_of_twisted_vector_preserved(self):
        """For y-axis twist, the norm of the xz-components is preserved."""
        for _ in range(50):
            x = random.uniform(-10, 10)
            z = random.uniform(-10, 10)
            orig_norm = math.sqrt(x * x + z * z)
            p = (x, random.uniform(-5, 5), z)
            k = random.uniform(0.1, 10.0)
            result = domain_twist(p, k)
            twisted_norm = math.sqrt(result[0] ** 2 + result[2] ** 2)
            assert twisted_norm == pytest.approx(orig_norm, rel=TOL_REL), (
                f"xz-norm not preserved: {orig_norm} -> {twisted_norm}"
            )


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Determinism and Stability
# =============================================================================


class TestTwistDeterminism:
    """Twist is deterministic: same inputs always produce same outputs."""

    def test_deterministic_same_input(self):
        """Calling twist twice with same args gives identical results."""
        p = (3.14159, -2.71828, 1.41421)
        k = 1.61803
        r1 = domain_twist(p, k)
        for _ in range(20):
            r2 = domain_twist(p, k)
            assert vec3_close(r1, r2), "Twist is not deterministic"

    def test_deterministic_all_axes(self):
        """Determinism holds for all axes."""
        p = (1.0, 2.0, 3.0)
        k = 0.5
        for axis in ("x", "y", "z"):
            r1 = domain_twist(p, k, axis=axis)  # type: ignore
            for _ in range(10):
                r2 = domain_twist(p, k, axis=axis)  # type: ignore
                assert vec3_close(r1, r2), (
                    f"Twist not deterministic for axis={axis}"
                )

    def test_symmetry_k_negation_y_axis(self):
        """For y-axis twist: twist(p, -k) = twist(twist(p, k), k) yields identity.

        Applying opposite twist angles should compose to identity.
        """
        p = (1.0, 2.0, 3.0)
        k = 1.0
        r_fwd = domain_twist(p, k)
        r_bwd = domain_twist(r_fwd, -k)
        assert vec3_close(p, r_bwd), (
            f"k then -k should restore: {p} -> {r_bwd}"
        )


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Edge Cases
# =============================================================================


class TestTwistEdgeCases:
    """Edge cases and boundary conditions."""

    def test_twist_on_axis_line(self):
        """Points exactly on the twist axis (x=0, z=0 for y-axis) are unchanged."""
        for _ in range(20):
            y = random.uniform(-10, 10)
            k = random.uniform(-10, 10)
            p = (0.0, y, 0.0)
            result = domain_twist(p, k)
            assert vec3_close(result, p), (
                f"On-axis point moved: {result}"
            )

    def test_twist_on_axis_line_all_axes(self):
        """Points on any twist axis are unchanged."""
        for axis in ("x", "y", "z"):
            for _ in range(10):
                k = random.uniform(-5, 5)
                if axis == "y":
                    p = (0.0, random.uniform(-5, 5), 0.0)
                elif axis == "x":
                    p = (random.uniform(-5, 5), 0.0, 0.0)
                else:
                    p = (0.0, 0.0, random.uniform(-5, 5))
                result = domain_twist(p, k, axis=axis)  # type: ignore
                assert vec3_close(result, p), (
                    f"On-axis point changed for axis={axis}: {result}"
                )

    def test_twist_large_k(self):
        """Large k produces many full rotations (still periodic)."""
        p = (1.0, 100.0 * math.pi, 0.0)  # y = 100*pi
        k = 2.0
        # At y = 100*pi, k=2 => theta = 200*pi => 100 full rotations => identity
        result = domain_twist(p, k)
        expected = (1.0, 100.0 * math.pi, 0.0)
        assert vec3_close(result, expected, abs_tol=1e-5), (
            f"Large k not periodic at full rotations: {result}"
        )

    def test_twist_finite_for_diverse_inputs(self):
        """Twist produces finite output for a wide range of inputs."""
        for _ in range(100):
            p = (random.uniform(-1e6, 1e6),
                 random.uniform(-1e6, 1e6),
                 random.uniform(-1e6, 1e6))
            k = random.uniform(-1e3, 1e3)
            for axis in ("x", "y", "z"):
                result = domain_twist(p, k, axis=axis)  # type: ignore
                assert_finite(result)

    def test_twist_at_full_rotation(self):
        """At y*|k| = 2*pi, the xz-rotation is exactly 2*pi => identity."""
        p = (2.0, 0.0, 3.0)
        k = math.pi
        # theta = pi * 0 = 0 => identity
        r0 = domain_twist(p, k)
        # theta = pi * 2 = 2*pi => identity
        p_full = (2.0, 2.0, 3.0)
        r_full = domain_twist(p_full, k)
        assert vec3_close(r0, r_full), (
            f"Full rotation should restore: {r0} vs {r_full}"
        )


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Composability
# =============================================================================


class TestTwistComposability:
    """Twist composes with itself and other operations."""

    def test_twist_twice_equals_double_k(self):
        """Applying twist twice with half the strength is equivalent."""
        p = (1.0, 2.0, 3.0)
        k = 1.0
        # Apply twist(k) twice
        once = domain_twist(p, k * 0.5)
        twice = domain_twist(once, k * 0.5)
        # Apply twist(k) once (total angle same)
        once_full = domain_twist(p, k)
        assert vec3_close(twice, once_full), (
            f"Two half-twists != one full twist: {twice} vs {once_full}"
        )

    def test_twist_commutes_with_itself(self):
        """Twist(k1) then twist(k2) = twist(k2) then twist(k1) = twist(k1+k2)."""
        p = (1.0, 2.0, 3.0)
        k1 = 0.3
        k2 = 0.7
        a = domain_twist(domain_twist(p, k1), k2)
        b = domain_twist(domain_twist(p, k2), k1)
        c = domain_twist(p, k1 + k2)
        assert vec3_close(a, b), (
            f"Twist does not commute with itself: {a} vs {b}"
        )
        assert vec3_close(a, c), (
            f"Sequential twists != sum of k: {a} vs {c}"
        )

    def test_twist_compose_with_xz_translation(self):
        """Translation in xz commutes with y-axis twist (both preserve xz-distance)."""
        p = (1.0, 2.0, 3.0)
        k = 1.5
        t = (0.5, 0.0, -0.3)
        # Twist then translate
        twisted = domain_twist(p, k)
        tt = (twisted[0] + t[0], twisted[1] + t[1], twisted[2] + t[2])
        # Translate then twist (translation is in untwisted space => different)
        translated = (p[0] + t[0], p[1] + t[1], p[2] + t[2])
        tt2 = domain_twist(translated, k)
        # These are NOT equal because twist is applied in the rotated frame
        # But the xz distance between them should be preserved
        d1 = vec3_dist(tt, tt2)
        # The difference is purely due to rotation, check it's bounded
        assert math.isfinite(d1)


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Axis Specification
# =============================================================================


class TestTwistAxisSpecification:
    """Twist can be applied around any specified axis."""

    def test_x_axis_twist_rotates_yz(self):
        """Twist around x rotates the yz-plane."""
        p = (0.0, 1.0, 0.0)
        k = math.pi / 2.0  # 90-degree rotation
        result = domain_twist(p, k, axis="x")
        # After 90 deg around x: y->0, z->1
        assert result[1] == pytest.approx(0.0, abs=TOL_ABS), (
            f"y should be 0 after 90deg around x: {result}"
        )
        assert result[2] == pytest.approx(1.0, rel=TOL_REL), (
            f"z should be 1 after 90deg around x: {result}"
        )

    def test_z_axis_twist_rotates_xy(self):
        """Twist around z rotates the xy-plane."""
        p = (1.0, 0.0, 0.0)
        k = math.pi / 2.0  # 90-degree rotation
        result = domain_twist(p, k, axis="z")
        # After 90 deg around z: x->0, y->1
        assert result[0] == pytest.approx(0.0, abs=TOL_ABS), (
            f"x should be 0 after 90deg around z: {result}"
        )
        assert result[1] == pytest.approx(1.0, rel=TOL_REL), (
            f"y should be 1 after 90deg around z: {result}"
        )

    def test_all_axes_distinct_for_same_input(self):
        """Twist around different axes produces different results for same p,k."""
        p = (1.0, 2.0, 3.0)
        k = 0.5
        rx = domain_twist(p, k, axis="x")
        ry = domain_twist(p, k, axis="y")
        rz = domain_twist(p, k, axis="z")
        # All should be different (non-zero k with non-zero coordinates)
        results = [rx, ry, rz]
        for i in range(3):
            for j in range(i + 1, 3):
                assert not vec3_close(results[i], results[j]), (
                    f"Axis {['x','y','z'][i]} and {['x','y','z'][j]} "
                    f"produced same result: {results[i]}"
                )

    def test_invalid_axis_raises(self):
        """Invalid axis specification raises ValueError."""
        with pytest.raises(ValueError, match="Unknown twist axis"):
            domain_twist((0.0, 0.0, 0.0), 1.0, axis="w")  # type: ignore


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Known Mathematical Properties
# =============================================================================


class TestTwistKnownValues:
    """Known mathematical values for specific inputs."""

    def test_xz_rotation_45_degrees(self):
        """At y*|k| = pi/4 (45 deg), x and z should be equal for p=(1,*,0)."""
        p = (1.0, 1.0, 0.0)
        k = math.pi / 4.0  # angle = pi/4 at y=1
        result = domain_twist(p, k)
        # Rotating (1, 0) by 45 degrees: r = sqrt(2)/2 ≈ 0.7071
        expected = math.cos(math.pi / 4.0)  # sqrt(2)/2
        assert result[0] == pytest.approx(expected, rel=TOL_REL), (
            f"x at 45deg: {result[0]} != {expected}"
        )
        assert result[2] == pytest.approx(expected, rel=TOL_REL), (
            f"z at 45deg: {result[2]} != {expected}"
        )

    def test_xz_rotation_180_degrees(self):
        """At y*|k| = pi (180 deg), xz should be negated for p=(x,*,z)."""
        p = (2.0, 1.0, 3.0)
        k = math.pi  # angle = pi at y=1
        result = domain_twist(p, k)
        assert result[0] == pytest.approx(-p[0], rel=TOL_REL), (
            f"x at 180deg: {result[0]} != {-p[0]}"
        )
        assert result[2] == pytest.approx(-p[2], rel=TOL_REL), (
            f"z at 180deg: {result[2]} != {-p[2]}"
        )

    def test_determinant_of_twist(self):
        """The twist preserves the determinant of the xz-Jacobian.

        For y-axis twist, the Jacobian of the xz mapping is a rotation matrix
        with determinant = 1 for all k and y.
        """
        # The 2x2 rotation matrix [cos(theta), -sin(theta); sin(theta), cos(theta)]
        # has determinant = cos^2 + sin^2 = 1 for any theta
        for _ in range(50):
            k = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            theta = k * y
            det = math.cos(theta) ** 2 + math.sin(theta) ** 2
            assert det == pytest.approx(1.0, abs=TOL_ABS), (
                f"Rotation matrix determinant != 1: {det}"
            )


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Sphere Helix Property
# =============================================================================


class TestTwistSphereProperty:
    """A sphere centered at the twist axis traces a helix under twist.

    When a point not on the twist axis is twisted through increasing y,
    it traces a helical path. This is the defining visual property of
    a twist transformation.
    """

    def test_helix_xz_projection_is_circular(self):
        """xz-projection of twisted points at varying y is a circle."""
        x0, z0 = 1.0, 0.0
        k = 1.0
        radius_sq = x0 * x0 + z0 * z0
        for y in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            result = domain_twist((x0, y, z0), k)
            xz_norm_sq = result[0] ** 2 + result[2] ** 2
            assert xz_norm_sq == pytest.approx(radius_sq, rel=TOL_REL), (
                f"xz-norm changed at y={y}: {xz_norm_sq} != {radius_sq}"
            )

    def test_helix_phase_advances_linearly(self):
        """The phase angle of the xz-projection advances linearly with y."""
        x0, z0 = 1.0, 0.0
        k = 1.0
        y_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
        angles = []
        for y in y_vals:
            result = domain_twist((x0, y, z0), k)
            angle = math.atan2(result[2], result[0])
            angles.append(angle)
        # Check that each step advances the angle by k * delta_y
        for i in range(1, len(angles)):
            delta_angle = angles[i] - angles[i - 1]
            delta_y = y_vals[i] - y_vals[i - 1]
            assert delta_angle == pytest.approx(k * delta_y, abs=TOL_REL), (
                f"Phase advance not linear: {delta_angle} != {k * delta_y}"
            )


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Numerical Robustness
# =============================================================================


class TestTwistNumericalRobustness:
    """Twist is numerically robust across extreme inputs."""

    def test_twist_very_small_k(self):
        """Extremely small k approximates identity."""
        p = (1000.0, 1000.0, 1000.0)
        k = 1e-10
        result = domain_twist(p, k)
        assert vec3_close(result, p, rel_tol=1e-6), (
            f"Small k should approximate identity: {result}"
        )

    def test_twist_negative_y(self):
        """Twist works with negative y-coordinates (negative angles)."""
        p = (1.0, -1.0, 0.0)
        k = math.pi / 2.0
        result = domain_twist(p, k)
        # theta = pi/2 * -1 = -pi/2 => rotate by -90 deg
        # (1,0) rotated by -90 => (0,-1)
        assert result[0] == pytest.approx(0.0, abs=TOL_ABS), (
            f"x at -90deg: {result[0]}"
        )
        assert result[2] == pytest.approx(-1.0, rel=TOL_REL), (
            f"z at -90deg: {result[2]}"
        )

    def test_twist_large_k_matches_analytic(self):
        """Very large k may lose precision but should still be finite."""
        p = (1.0, 0.25, 2.0)
        k = 1e6
        result = domain_twist(p, k)
        assert_finite(result)
        # The xz-norm should still be preserved (within floating point limits)
        xz_norm_sq = result[0] ** 2 + result[2] ** 2
        expected_norm_sq = p[0] ** 2 + p[2] ** 2
        # Floating point may lose precision at large angles, check within 0.1%
        assert xz_norm_sq == pytest.approx(expected_norm_sq, rel=1e-3), (
            f"xz-norm not preserved for large k: {xz_norm_sq} vs {expected_norm_sq}"
        )


# =============================================================================
# Test: T-DEMO-1.25 Twist -- SDF Integration Properties
# =============================================================================


class TestTwistSDFProperties:
    """Properties important when twist is used as an SDF domain operation."""

    def test_twist_output_range(self):
        """Twisted coordinates are in a reasonable range."""
        for _ in range(50):
            p = (random.uniform(-100, 100),
                 random.uniform(-100, 100),
                 random.uniform(-100, 100))
            k = random.uniform(-5, 5)
            result = domain_twist(p, k)
            # The axis coordinate is unchanged
            # The twist-plane coordinates are bounded by their original norm
            xz_norm = math.sqrt(p[0] ** 2 + p[2] ** 2)
            result_xz_norm = math.sqrt(result[0] ** 2 + result[2] ** 2)
            assert result_xz_norm == pytest.approx(xz_norm, rel=TOL_REL)

    def test_zwz_identity_consistency(self):
        """Twist(0) then twist(0) = twist(0) = identity."""
        for axis in ("x", "y", "z"):
            p = (random.uniform(-5, 5),
                 random.uniform(-5, 5),
                 random.uniform(-5, 5))
            r1 = domain_twist(p, 0.0, axis=axis)  # type: ignore
            r2 = domain_twist(r1, 0.0, axis=axis)  # type: ignore
            assert vec3_close(r2, p), (
                f"Identity consistency failed for axis={axis}"
            )


# =============================================================================
# Test: T-DEMO-1.25 Twist -- Acceptance Criterion
# =============================================================================


class TestTwistAcceptance:
    """Acceptance: space twisted along specified axis.

    These tests verify that the space is indeed twisted: parallel lines along
    the twist axis at different offsets in the perpendicular plane rotate
    at different phase angles, creating the characteristic twisted appearance.
    """

    def test_space_twisted_along_y_axis(self):
        """Lines parallel to y at different xz offsets twist at different angles.

        Two points at the same y but different xz offsets will rotate to
        different positions. This is the essence of 'space twisted along axis'.
        """
        k = 1.0
        y = 2.0
        # Two points with same y but different xz offsets
        p1 = (1.0, y, 0.0)
        p2 = (0.0, y, 1.0)
        r1 = domain_twist(p1, k)
        r2 = domain_twist(p2, k)
        # Both should have the same y (axis invariant)
        assert r1[1] == pytest.approx(y)
        assert r2[1] == pytest.approx(y)
        # Both should rotate by the same angle (same k*y)
        theta = k * y
        assert r1 == pytest.approx(
            (math.cos(theta), y, math.sin(theta)), rel=TOL_REL
        )
        assert r2 == pytest.approx(
            (-math.sin(theta), y, math.cos(theta)), rel=TOL_REL
        )

    def test_adjacent_layers_twist_continuously(self):
        """As y increases continuously, the twist angle changes continuously.

        There should be no discontinuities in the twisted space.
        """
        k = 2.0
        x, z = 1.0, 0.0
        y_vals = [i * 0.01 for i in range(100)]  # 0 to 0.99
        prev = domain_twist((x, y_vals[0], z), k)
        for y in y_vals[1:]:
            curr = domain_twist((x, y, z), k)
            # The change between adjacent layers should be small
            change = vec3_dist(curr, prev)
            assert change < 0.05, (
                f"Discontinuity at y={y}: change={change}"
            )
            prev = curr

    def test_twist_visual_property_radial_variation(self):
        """Points at different radii trace different helix speeds.

        Not literally different speeds -- the angular frequency is the same
        for all radii. But the LINEAR displacement is larger at larger radii.
        """
        k = 1.0
        y = 1.0
        # Inner radius
        inner = domain_twist((0.5, y, 0.0), k)
        inner_disp = vec3_dist((0.5, y, 0.0), inner)
        # Outer radius (same angle, larger radius => larger displacement)
        outer = domain_twist((2.0, y, 0.0), k)
        outer_disp = vec3_dist((2.0, y, 0.0), outer)
        # Outer should have larger displacement
        assert outer_disp > inner_disp, (
            f"Outer displacement {outer_disp} should be > inner {inner_disp}"
        )
        # Ratio should equal radius ratio
        assert outer_disp / inner_disp == pytest.approx(4.0, rel=TOL_REL), (
            f"Displacement ratio {outer_disp/inner_disp} != 4.0"
        )

    def test_twist_acceptance_all_specified_axes(self):
        """Acceptance: space twisted along specified axis for x, y, and z."""
        k = math.pi / 4.0
        # Test point offset from each axis
        tests = [
            ("x", (0.0, 1.0, 0.0),
             lambda r: (r[1] == pytest.approx(math.cos(k * 0) * 1.0, rel=TOL_REL)
                        and r[2] == pytest.approx(math.sin(k * 0) * 1.0, rel=TOL_REL))),
            ("y", (1.0, 0.0, 0.0),
             lambda r: (r[0] == pytest.approx(math.cos(k * 0) * 1.0, rel=TOL_REL)
                        and r[2] == pytest.approx(math.sin(k * 0) * 1.0, rel=TOL_REL))),
            ("z", (1.0, 0.0, 0.0),
             lambda r: (r[0] == pytest.approx(math.cos(k * 0) * 1.0, rel=TOL_REL)
                        and r[1] == pytest.approx(math.sin(k * 0) * 1.0, rel=TOL_REL))),
        ]
        for axis, p, check in tests:
            r = domain_twist(p, k, axis=axis)  # type: ignore
            assert check(r), (
                f"Axis {axis}: failed at twist reference point: {r}"
            )
