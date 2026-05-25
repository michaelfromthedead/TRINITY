"""
Cleanroom blackbox tests for sdTorus WGSL function (T-DEMO-1.3).

Tests the signed distance function for a torus with major radius t.x and
minor radius t.y, treating the implementation as a black box from the spec.

BLACKBOX coverage plan:
  Path A: Surface points return distance ~0
  Path B: Inside points return negative distance
  Path C: Hole points return positive distance
  Path D: Far point returns approximately correct distance
  Path E: Direction: moving toward center decreases distance, away increases
  Path F: Y-axis behavior: point above/below torus
  Path G: Continuity: nearby points have nearby distances
  Path H: Sign convention: positive outside, negative inside, zero on surface
  Path I: Symmetry: sdTorus((x,0,z), t) == sdTorus((-x,0,z), t)
  Path J: Different torus parameters (thin vs thick)
"""

from __future__ import annotations

import math

import pytest

# =============================================================================
# Python model of sdTorus matching WGSL semantics (blackbox proxy)
# =============================================================================


def py_sd_torus(p, t):
    """Python model of WGSL sdTorus(p: vec3<f32>, t: vec2<f32>) -> f32.

    Signed distance from point p (3-tuple) to a torus with major radius
    t.x (ring) and minor radius t.y (tube).

    Reference: Inigo Quilez -- Torus SDF
    https://iquilezles.org/articles/distfunctions/
    """
    safe_tx = abs(t[0])
    safe_ty = abs(t[1])
    qx = math.sqrt(p[0] * p[0] + p[2] * p[2]) - safe_tx
    qy = p[1]
    return math.sqrt(qx * qx + qy * qy) - safe_ty


# =============================================================================
# Default test parameters
# =============================================================================

TOL_SURFACE = 1e-9    # Points on surface should be very close to 0
TOL_FAR = 0.001       # Tolerance for far-point approximations
TOL_CONTINUITY = 0.01  # Max allowed jump for nearby points

# Standard torus: major=2, minor=0.5
T_DEFAULT = (2.0, 0.5)

# Thin torus: major=2, minor=0.2
T_THIN = (2.0, 0.2)

# Thick torus: major=1, minor=0.8
T_THICK = (1.0, 0.8)


# =============================================================================
# Path A: Surface points return distance ~0 (Acceptance)
# =============================================================================


class TestSurfacePoints:
    """Points exactly on the torus surface should return distance ~0."""

    def test_surface_x_axis(self):
        """Point on surface along x-axis: (t.x + t.y, 0, 0)."""
        t = T_DEFAULT
        p = (t[0] + t[1], 0.0, 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point on x-axis surface ({p}) should be on torus surface, got {d}"
        )

    def test_surface_above_center(self):
        """Point on surface directly above center: (t.x, t.y, 0)."""
        t = T_DEFAULT
        p = (t[0], t[1], 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point above center ({p}) should be on torus surface, got {d}"
        )

    def test_surface_negative_x(self):
        """Point on surface along negative x: (-(t.x + t.y), 0, 0)."""
        t = T_DEFAULT
        p = (-(t[0] + t[1]), 0.0, 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point on -x surface ({p}) should be on torus surface, got {d}"
        )

    def test_surface_along_z(self):
        """Point on surface along z-axis: (0, 0, t.x + t.y)."""
        t = T_DEFAULT
        p = (0.0, 0.0, t[0] + t[1])
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point on z surface ({p}) should be on torus surface, got {d}"
        )

    def test_surface_below_center(self):
        """Point on surface directly below center: (t.x, -(t.y), 0)."""
        t = T_DEFAULT
        p = (t[0], -t[1], 0.0)
        d = py_sd_torus(p, t)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Point below center ({p}) should be on torus surface, got {d}"
        )

    def test_surface_multiple_angles(self):
        """Points on surface at various angles around the tube."""
        t = T_DEFAULT
        angles = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for theta in angles:
            # Point around the tube cross section at x-axis
            px = t[0] + t[1] * math.cos(theta)
            py = t[1] * math.sin(theta)
            p = (px, py, 0.0)
            d = py_sd_torus(p, t)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Point at tube angle {theta} ({p}) should be on surface, "
                f"got {d}"
            )


# =============================================================================
# Path B: Inside points return negative distance (Acceptance)
# =============================================================================


class TestInsideNegative:
    """Points inside the torus tube should return negative distance."""

    def test_major_radius_center(self):
        """Center of the tube on the major radius: (t.x, 0, 0) -> -t.y."""
        t = T_DEFAULT
        p = (t[0], 0.0, 0.0)
        d = py_sd_torus(p, t)
        expected = -t[1]
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Point at tube center ({p}) should give -t.y = {expected}, got {d}"
        )

    def test_inside_tube_various_points(self):
        """Several points known to be inside the tube should give negative."""
        t = T_DEFAULT
        # Point offset slightly but still inside tube
        inside_points = [
            (t[0] + 0.2, 0.0, 0.0),   # Right side, still inside (t.y=0.5)
            (t[0] - 0.2, 0.0, 0.0),   # Left side, still inside
            (t[0], 0.3, 0.0),         # Above center, still inside
            (t[0], -0.3, 0.0),        # Below center, still inside
            (t[0], 0.0, 0.2),         # Along z, still inside
        ]
        for p in inside_points:
            d = py_sd_torus(p, t)
            assert d < 0.0, (
                f"Inside point ({p}) should have negative distance, got {d}"
            )

    def test_inside_thin_torus(self):
        """Points inside a thin torus tube."""
        t = T_THIN
        p = (t[0], 0.0, 0.0)
        expected = -t[1]
        d = py_sd_torus(p, t)
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Inside thin torus center ({p}) should give -{t[1]}, got {d}"
        )


# =============================================================================
# Path C: Hole points return positive distance (Acceptance)
# =============================================================================


class TestHolePositive:
    """Points in the hole (center donut hole) should return positive distance."""

    def test_origin(self):
        """Origin (0, 0, 0) is in the hole for t.x > t.y."""
        t = T_DEFAULT
        d = py_sd_torus((0.0, 0.0, 0.0), t)
        # Distance from origin to surface: t.x - t.y
        expected = t[0] - t[1]
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Origin should be in hole with distance {expected}, got {d}"
        )

    def test_hole_negative_x(self):
        """Point on -x axis inside the hole."""
        t = T_DEFAULT
        p = (-1.0, 0.0, 0.0)
        d = py_sd_torus(p, t)
        assert d > 0.0, (
            f"Hole point ({p}) should have positive distance, got {d}"
        )

    def test_hole_along_z(self):
        """Point on z-axis inside the hole."""
        t = T_DEFAULT
        p = (0.0, 0.0, 0.5)
        d = py_sd_torus(p, t)
        assert d > 0.0, (
            f"Hole point ({p}) should have positive distance, got {d}"
        )

    def test_centerline_is_positive(self):
        """Points whose xz-distance to origin is less than t.x - t.y are in hole."""
        t = T_DEFAULT
        # Point at half the major radius in xz-plane
        p = (t[0] * 0.3, 0.0, 0.0)
        d = py_sd_torus(p, t)
        assert d > 0.0, (
            f"Hole point ({p}) should have positive distance, got {d}"
        )


# =============================================================================
# Path D: Far point returns approximately correct distance
# =============================================================================


class TestFarPoint:
    """Very distant points should return approximately correct distance."""

    def test_far_positive_x(self):
        """Far away on x-axis: (100, 0, 0) -> ~100 - (t.x + t.y)."""
        t = T_DEFAULT
        far_point = (100.0, 0.0, 0.0)
        d = py_sd_torus(far_point, t)
        expected = 100.0 - (t[0] + t[1])
        assert d == pytest.approx(expected, abs=TOL_FAR), (
            f"Far point ({far_point}) should have distance ~{expected}, got {d}"
        )

    def test_far_negative_x(self):
        """Far away on -x axis."""
        t = T_DEFAULT
        far_point = (-100.0, 0.0, 0.0)
        d = py_sd_torus(far_point, t)
        expected = 100.0 - (t[0] + t[1])
        assert d == pytest.approx(expected, abs=TOL_FAR), (
            f"Far point ({far_point}) should have distance ~{expected}, got {d}"
        )

    def test_far_positive_y(self):
        """Far away on y-axis: (0, 100, 0) -> approx distance."""
        t = T_DEFAULT
        far_point = (0.0, 100.0, 0.0)
        d = py_sd_torus(far_point, t)
        # On y-axis: qx = 0 - t.x = -2, qy = 100
        # length = sqrt(4 + 10000) = sqrt(10004) ~ 100.02
        expected = math.sqrt(t[0]**2 + 100.0**2) - t[1]
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Far point ({far_point}) should have distance {expected}, got {d}"
        )

    def test_far_diagonal(self):
        """Far away on diagonal."""
        t = (10.0, 2.0)
        far_point = (1000.0, 1000.0, 0.0)
        d = py_sd_torus(far_point, t)
        qx = math.sqrt(far_point[0]**2 + far_point[2]**2) - t[0]
        expected = math.sqrt(qx*qx + far_point[1]*far_point[1]) - t[1]
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Far diagonal ({far_point}) should have distance {expected}, got {d}"
        )

    def test_far_asymptotic_behavior(self):
        """As point goes to infinity, distance approximates raw distance to origin."""
        t = T_DEFAULT
        very_far = (1e6, 0.0, 0.0)
        d = py_sd_torus(very_far, t)
        expected = 1e6 - (t[0] + t[1])
        assert d == pytest.approx(expected, abs=0.01), (
            f"Very far point ({very_far}) should have distance ~{expected}, got {d}"
        )


# =============================================================================
# Path E: Direction -- moving toward center decreases distance
# =============================================================================


class TestDirection:
    """Moving toward the center of the ring decreases distance; away increases."""

    def test_toward_center_decreases_x(self):
        """Moving along x toward ring center decreases distance."""
        t = T_DEFAULT
        outside = (5.0, 0.0, 0.0)
        closer = (3.0, 0.0, 0.0)
        d_outside = py_sd_torus(outside, t)
        d_closer = py_sd_torus(closer, t)
        assert d_closer < d_outside, (
            f"Moving toward ring center should decrease distance: "
            f"{d_closer} >= {d_outside}"
        )

    def test_away_from_center_increases_x(self):
        """Moving away from ring center along x increases distance."""
        t = T_DEFAULT
        p_near = (3.0, 0.0, 0.0)
        p_far = (5.0, 0.0, 0.0)
        d_near = py_sd_torus(p_near, t)
        d_far = py_sd_torus(p_far, t)
        assert d_far > d_near, (
            f"Moving away from center should increase distance: "
            f"{d_far} <= {d_near}"
        )

    def test_toward_center_along_y(self):
        """Moving in y toward the tube center decreases distance."""
        t = T_DEFAULT
        above = (t[0], 2.0, 0.0)
        nearer = (t[0], 0.5, 0.0)
        d_above = py_sd_torus(above, t)
        d_nearer = py_sd_torus(nearer, t)
        assert d_nearer < d_above, (
            f"Moving toward tube center in y should decrease distance: "
            f"{d_nearer} >= {d_above}"
        )

    def test_away_increases_from_inside(self):
        """Moving from inside the tube outward increases distance."""
        t = T_DEFAULT
        inside = (t[0], 0.0, 0.0)              # Center of tube: -t.y
        on_surface = (t[0] + t[1], 0.0, 0.0)   # On surface: ~0
        outside = (t[0] + t[1] + 1.0, 0.0, 0.0)  # Outside: >0
        d_inside = py_sd_torus(inside, t)
        d_surface = py_sd_torus(on_surface, t)
        d_outside = py_sd_torus(outside, t)
        assert d_inside < d_surface < d_outside, (
            f"Moving from inside to outside should strictly increase: "
            f"{d_inside} < {d_surface} < {d_outside} failed"
        )

    def test_move_through_hole(self):
        """Moving from inside hole to outside on other side."""
        t = T_DEFAULT
        in_hole = (0.0, 0.0, 0.0)
        on_ring = (t[0], 0.0, 0.0)
        far_side = (t[0] * 2 + t[1], 0.0, 0.0)
        d_hole = py_sd_torus(in_hole, t)
        d_ring = py_sd_torus(on_ring, t)
        d_far = py_sd_torus(far_side, t)
        # Goes positive (hole) -> negative (inside tube) -> positive (outside)
        assert d_hole > 0.0, f"Hole point should be positive, got {d_hole}"
        assert d_ring < 0.0, f"Tube center should be negative, got {d_ring}"
        assert d_far > 0.0, f"Far point should be positive, got {d_far}"


# =============================================================================
# Path F: Y-axis behavior -- point above/below torus
# =============================================================================


class TestYAxisBehavior:
    """Behavior of points directly above or below the torus on y-axis."""

    def test_above_torus_positive(self):
        """Point high above the torus on y-axis is outside."""
        t = T_DEFAULT
        p = (t[0], 10.0, 0.0)
        d = py_sd_torus(p, t)
        assert d > 0.0, (
            f"Point above torus ({p}) should have positive distance, got {d}"
        )

    def test_below_torus_positive(self):
        """Point far below the torus on y-axis is outside."""
        t = T_DEFAULT
        p = (t[0], -10.0, 0.0)
        d = py_sd_torus(p, t)
        assert d > 0.0, (
            f"Point below torus ({p}) should have positive distance, got {d}"
        )

    def test_above_vs_below_symmetric(self):
        """Distance above and below should be equal for symmetric y offsets."""
        t = T_DEFAULT
        y_offsets = [0.8, 1.5, 3.0, 5.0]
        for y in y_offsets:
            d_above = py_sd_torus((t[0], y, 0.0), t)
            d_below = py_sd_torus((t[0], -y, 0.0), t)
            assert d_above == pytest.approx(d_below, abs=TOL_SURFACE), (
                f"Distance should be symmetric for y=+/-{y}: "
                f"{d_above} vs {d_below}"
            )

    def test_on_y_axis(self):
        """Point directly on y-axis (0, y, 0) should behave like point in hole."""
        t = T_DEFAULT
        p = (0.0, 2.0, 0.0)
        d = py_sd_torus(p, t)
        expected = math.sqrt(t[0]**2 + 2.0**2) - t[1]
        assert d == pytest.approx(expected, abs=TOL_SURFACE), (
            f"Point on y-axis ({p}) should have correct hole distance, got {d}"
        )


# =============================================================================
# Path G: Continuity -- nearby points have nearby distances
# =============================================================================


class TestContinuity:
    """The SDF should be continuous: nearby points produce nearby distances."""

    def test_continuity_along_x(self):
        """Continuity along x-axis across hole, tube, and beyond."""
        t = T_DEFAULT
        step = 0.001
        # Sweep from -3.0 to 3.0 covering hole, tube, and outside
        x_start = -3.0
        prev = py_sd_torus((x_start, 0.0, 0.0), t)
        for i in range(1, 6000):
            x = x_start + i * step
            if x > 3.0:
                break
            curr = py_sd_torus((x, 0.0, 0.0), t)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at x={x}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """Continuity along y-axis."""
        t = T_DEFAULT
        step = 0.001
        prev = py_sd_torus((t[0], -2.0, 0.0), t)
        for i in range(1, 500):
            p = (t[0], -2.0 + i * step, 0.0)
            curr = py_sd_torus(p, t)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at y={p[1]}: diff={diff}"
            )
            prev = curr

    def test_continuity_along_z(self):
        """Continuity along z-axis."""
        t = T_DEFAULT
        step = 0.001
        prev = py_sd_torus((t[0], 0.0, -2.0), t)
        for i in range(1, 500):
            p = (t[0], 0.0, -2.0 + i * step,)
            curr = py_sd_torus(p, t)
            diff = abs(curr - prev)
            assert diff < TOL_CONTINUITY, (
                f"Discontinuity at z={p[2]}: diff={diff}"
            )
            prev = curr

    def test_continuity_through_surface_x(self):
        """Continuity when crossing the surface along x."""
        t = T_DEFAULT
        step = 0.0001
        # Cross the surface at x = t.x + t.y
        for offset in [i * step for i in range(-10, 11)]:
            p = (t[0] + t[1] + offset, 0.0, 0.0)
            d = py_sd_torus(p, t)
            # Should smoothly transition through zero
            assert abs(d - offset) < 0.001, (
                f"Near surface at x={p[0]}: expected ~{offset}, got {d}"
            )

    def test_gradient_sign_stability(self):
        """Small perturbations should not flip sign unexpectedly (stability)."""
        t = T_DEFAULT
        center = (t[0], 0.0, 0.0)
        base_d = py_sd_torus(center, t)
        eps = 1e-7
        for dx, dy, dz in [(eps, 0, 0), (-eps, 0, 0), (0, eps, 0),
                           (0, -eps, 0), (0, 0, eps), (0, 0, -eps)]:
            p = (center[0] + dx, center[1] + dy, center[2] + dz)
            d = py_sd_torus(p, t)
            # Distance should not change chaotically
            assert abs(d - base_d) < 0.001, (
                f"Unstable distance change at ({p}): base={base_d}, got {d}"
            )


# =============================================================================
# Path H: Sign convention -- positive outside, negative inside, zero on surface
# =============================================================================


class TestSignConvention:
    """Distance sign: positive outside, negative inside, zero on surface."""

    def test_outside_positive(self):
        """Points known to be outside the torus should have positive distance."""
        t = T_DEFAULT
        outside_points = [
            (10.0, 0.0, 0.0),
            (-10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0),
            (2.0, 1.0, 0.0),   # Outside tube in y direction (t.y=0.5, y=1.0)
            (1.0, 0.0, 0.0),   # Inside the hole
        ]
        for p in outside_points:
            d = py_sd_torus(p, t)
            assert d >= 0.0, (
                f"Outside point ({p}) should have non-negative SDF, got {d}"
            )

    def test_inside_negative(self):
        """Points known to be inside the torus should have negative distance."""
        t = T_DEFAULT
        inside_points = [
            (t[0], 0.0, 0.0),         # Center of tube
            (t[0] + 0.3, 0.0, 0.0),   # Inside tube wall
            (t[0] - 0.3, 0.0, 0.0),   # Inside tube wall
            (t[0], 0.2, 0.0),         # Inside y-axis
        ]
        for p in inside_points:
            d = py_sd_torus(p, t)
            assert d <= 0.0, (
                f"Inside point ({p}) should have non-positive SDF, got {d}"
            )

    def test_on_surface_zero(self):
        """Points on the torus surface should have near-zero distance."""
        t = T_DEFAULT
        surface_points = [
            (t[0] + t[1], 0.0, 0.0),
            (t[0], t[1], 0.0),
            (t[0], -t[1], 0.0),
            (-(t[0] + t[1]), 0.0, 0.0),
            (0.0, 0.0, t[0] + t[1]),
        ]
        for p in surface_points:
            d = py_sd_torus(p, t)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point ({p}) should have near-zero SDF, got {d}"
            )

    def test_sign_transition_at_surface(self):
        """Sign should flip at the surface: inside negative, outside positive."""
        t = T_DEFAULT
        surface_x = t[0] + t[1]
        step = 1e-4
        inside = py_sd_torus((surface_x - step, 0.0, 0.0), t)
        on_surface = py_sd_torus((surface_x, 0.0, 0.0), t)
        outside = py_sd_torus((surface_x + step, 0.0, 0.0), t)
        assert inside < 0.0, (
            f"Inside point should be negative, got {inside}"
        )
        assert on_surface == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface point should be zero, got {on_surface}"
        )
        assert outside > 0.0, (
            f"Outside point should be positive, got {outside}"
        )


# =============================================================================
# Path I: Symmetry -- sdTorus((x,0,z), t) == sdTorus((-x,0,z), t)
# =============================================================================


class TestSymmetry:
    """The torus SDF is symmetric in the xz-plane."""

    def test_x_symmetry(self):
        """Points (x, 0, z) and (-x, 0, z) should have equal distances."""
        t = T_DEFAULT
        test_points = [
            (1.0, 0.0, 0.3),
            (2.5, 0.0, 1.0),
            (0.5, 0.0, 2.0),
            (3.0, 0.0, 0.5),
            (1.8, 0.0, 1.8),
        ]
        for x, _, z in test_points:
            p_pos = (x, 0.0, z)
            p_neg = (-x, 0.0, z)
            d_pos = py_sd_torus(p_pos, t)
            d_neg = py_sd_torus(p_neg, t)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"Asymmetry: sdTorus({p_pos})={d_pos} != "
                f"sdTorus({p_neg})={d_neg}"
            )

    def test_z_symmetry(self):
        """Points (x, 0, z) and (x, 0, -z) should have equal distances."""
        t = T_DEFAULT
        test_points = [
            (0.5, 0.0, 1.0),
            (1.0, 0.0, 2.5),
            (2.0, 0.0, 0.5),
            (3.0, 0.0, 1.5),
            (1.8, 0.0, 0.8),
        ]
        for x, _, z in test_points:
            p_pos = (x, 0.0, z)
            p_neg = (x, 0.0, -z)
            d_pos = py_sd_torus(p_pos, t)
            d_neg = py_sd_torus(p_neg, t)
            assert d_pos == pytest.approx(d_neg, abs=TOL_SURFACE), (
                f"Asymmetry: sdTorus({p_pos})={d_pos} != "
                f"sdTorus({p_neg})={d_neg}"
            )

    def test_full_rotational_symmetry_xz(self):
        """Points at same radius in xz-plane have same distance (rotational symmetry)."""
        t = T_DEFAULT
        radius = 2.5
        angles = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
        y_values = [0.0, 0.3, -0.3, 0.8]
        for y in y_values:
            ref = py_sd_torus((radius, y, 0.0), t)
            for theta in angles:
                x = radius * math.cos(theta)
                z_val = radius * math.sin(theta)
                d = py_sd_torus((x, y, z_val), t)
                assert d == pytest.approx(ref, abs=TOL_SURFACE), (
                    f"Rotational asymmetry at angle {theta}, y={y}: "
                    f"reference={ref}, got {d}"
                )


# =============================================================================
# Path J: Different torus parameters (thin vs thick)
# =============================================================================


class TestParameterVariation:
    """Behavior with different torus radii parameters."""

    def test_thin_torus_acceptance(self):
        """Thin torus (t.x=2, t.y=0.2): surface and hole distances."""
        t = T_THIN
        assert py_sd_torus((t[0] + t[1], 0.0, 0.0), t) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_torus((t[0], t[1], 0.0), t) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_torus((t[0], 0.0, 0.0), t) == pytest.approx(-t[1], abs=TOL_SURFACE)
        assert py_sd_torus((0.0, 0.0, 0.0), t) == pytest.approx(t[0] - t[1], abs=TOL_SURFACE)

    def test_thick_torus_acceptance(self):
        """Thick torus (t.x=1, t.y=0.8): surface and hole distances."""
        t = T_THICK
        assert py_sd_torus((t[0] + t[1], 0.0, 0.0), t) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_torus((t[0], t[1], 0.0), t) == pytest.approx(0.0, abs=TOL_SURFACE)
        assert py_sd_torus((t[0], 0.0, 0.0), t) == pytest.approx(-t[1], abs=TOL_SURFACE)
        assert py_sd_torus((0.0, 0.0, 0.0), t) == pytest.approx(t[0] - t[1], abs=TOL_SURFACE)

    def test_thin_vs_thick_hole(self):
        """Thin torus has larger hole (t.x - t.y is larger)."""
        thin_hole = py_sd_torus((0.0, 0.0, 0.0), T_THIN)
        thick_hole = py_sd_torus((0.0, 0.0, 0.0), T_THICK)
        assert thin_hole > thick_hole, (
            f"Thin torus should have larger hole: thin hole {thin_hole} "
            f"vs thick hole {thick_hole}"
        )

    def test_thin_vs_thick_surface(self):
        """Thin torus surface is closer to ring center."""
        offset = 0.4
        thin_surface = py_sd_torus((T_THIN[0] + offset, 0.0, 0.0), T_THIN)
        thick_surface = py_sd_torus((T_THICK[0] + offset, 0.0, 0.0), T_THICK)
        # Both are outside since offset > both minor radii
        assert thin_surface > 0.0, (
            f"Offset {offset} > thin t.y {T_THIN[1]}, should be outside, "
            f"got {thin_surface}"
        )

    def test_negative_radii_handling(self):
        """The function should handle negative radii by taking abs (per WGSL code)."""
        t_neg = (-2.0, 0.5)
        t_pos = (2.0, 0.5)
        test_points = [
            (2.5, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 0.5, 0.0),
        ]
        for p in test_points:
            d_neg = py_sd_torus(p, t_neg)
            d_pos = py_sd_torus(p, t_pos)
            assert d_neg == pytest.approx(d_pos, abs=TOL_SURFACE), (
                f"Sign of t.x should not matter: ({p}) "
                f"neg={d_neg}, pos={d_pos}"
            )

    def test_minor_radius_negative(self):
        """Negative minor radius should work the same as positive."""
        t_neg = (2.0, -0.5)
        t_pos = (2.0, 0.5)
        test_points = [
            (2.5, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 0.5, 0.0),
        ]
        for p in test_points:
            d_neg = py_sd_torus(p, t_neg)
            d_pos = py_sd_torus(p, t_pos)
            assert d_neg == pytest.approx(d_pos, abs=TOL_SURFACE), (
                f"Sign of t.y should not matter: ({p}) "
                f"neg={d_neg}, pos={d_pos}"
            )


# =============================================================================
# Path K: Monotonicity along radial direction
# =============================================================================


class TestMonotonicity:
    """Distance should be monotonic as we move radially outward from the torus."""

    def test_radial_away_from_center(self):
        """Moving radially outward from torus center should be non-decreasing."""
        t = T_DEFAULT
        # Move outward along x from center through hole, tube, and beyond
        x_values = [i * 0.1 for i in range(50)]  # 0 to 4.9
        distances = [py_sd_torus((x, 0.0, 0.0), t) for x in x_values]

        for i in range(len(distances) - 1):
            x1, x2 = x_values[i], x_values[i + 1]
            d1, d2 = distances[i], distances[i + 1]
            # Allow sign change (positive->negative->positive)
            # From x=0 moving right: hole (positive) drops to center (negative)
            # then rises back to positive
            if x1 < t[0] and x2 < t[0] and d1 > 0 and d2 > 0:
                assert d2 <= d1 + 1e-12, (
                    f"Distance should decrease (or stay) moving toward ring "
                    f"center in hole: x from {x1} to {x2}, "
                    f"d from {d1} to {d2}"
                )

    def test_outside_monotonic_increasing(self):
        """Outside the torus, distance should increase with distance from origin."""
        t = T_DEFAULT
        outer_radius = t[0] + t[1]
        # From just outside surface to far away
        x_values = [outer_radius + i * 0.5 for i in range(20)]
        distances = [py_sd_torus((x, 0.0, 0.0), t) for x in x_values]
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1] + 1e-12, (
                f"Distance should be monotonically increasing outside torus: "
                f"at x={x_values[i]} d={distances[i]}, "
                f"at x={x_values[i+1]} d={distances[i+1]}"
            )
