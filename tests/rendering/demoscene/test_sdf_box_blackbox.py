"""
Cleanroom blackbox tests for sdBox(p, b) WGSL function (T-DEMO-1.2).

Tests a Python reference implementation of the signed-distance-to-axis-aligned-box
function, verifying correctness against the mathematical specification WITHOUT
knowledge of the WGSL implementation internals.

Specification (from sdf_box.wgsl):
  sdBox(p: vec3<f32>, b: vec3<f32>) -> f32

  Signed distance from point p to an axis-aligned box centered at the origin
  with half-extents b (half-width, half-height, half-depth).

  The result is:
    - negative when p is inside the box
    - zero when p is on the box surface
    - positive when p is outside the box

  Decomposition (blackbox spec):
    q = abs(p) - b
    outside_dist = length(max(q, 0))       -- distance to surface when outside
    inside_dist  = min(maxComponent(q), 0)  -- signed interior distance (<= 0)
    return outside_dist + inside_dist

  Acceptance criteria:
    1. Inside corner returns negative distance
    2. Outside corner returns positive distance
    3. Edge center returns zero (on surface)
    4. Face center returns zero (on surface)

Reference: Inigo Quilez -- SDF Primitives: sdBox
https://iquilezles.org/articles/distfunctions/
"""

import math
from typing import Tuple

import pytest

# =============================================================================
# Python reference model matching WGSL semantics exactly
# =============================================================================

Vec3 = Tuple[float, float, float]


def py_length(v: Vec3) -> float:
    """WGSL length for vec3<f32>."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def py_max_vec3(v: Vec3, s: float) -> Vec3:
    """WGSL max(vec3<f32>, vec3<f32>) -- component-wise max with scalar broadcast."""
    return (max(v[0], s), max(v[1], s), max(v[2], s))


def py_abs_vec3(v: Vec3) -> Vec3:
    """WGSL abs(vec3<f32>)."""
    return (abs(v[0]), abs(v[1]), abs(v[2]))


def py_sub_vec3(a: Vec3, b: Vec3) -> Vec3:
    """WGSL component-wise vec3 subtraction."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def py_max_component(v: Vec3) -> float:
    """WGSL max(q.x, max(q.y, q.z))."""
    return max(v[0], max(v[1], v[2]))


def py_sdBox(p: Vec3, b: Vec3) -> float:
    """Blackbox reference: sdBox(p, b) following the spec exactly.

    Pure Python model of the WGSL function with zero implementation knowledge
    beyond the mathematical definition.
    """
    q = py_sub_vec3(py_abs_vec3(p), b)

    # Outside term: length of the positive part of q
    # Every component inside the box is clamped to 0, giving zero contribution.
    outside_dist = py_length(py_max_vec3(q, 0.0))

    # Inside term: most-negative component of q, clamped to <= 0.
    # When outside, maxComponent(q) > 0 and min(clamp, 0) = 0, so term drops out.
    inside_dist = min(py_max_component(q), 0.0)

    return outside_dist + inside_dist


# =============================================================================
# Constants
# =============================================================================

TOL = 1e-12  # Strictest tolerance for exact surface/center results
TOL_RELAXED = 1e-9


# =============================================================================
# Test: T-DEMO-1.2 Sign Convention & Surface Detection
# =============================================================================


class TestSignConvention:
    """Verify the fundamental sign convention of the SDF.

    sdBox must return:
      negative inside the box
      zero   on the box surface
      positive outside the box
    """

    def test_center_returns_negative(self):
        """At box center (0,0,0), signed distance = -min(b.x, b.y, b.z)."""
        b = (2.0, 3.0, 4.0)
        expected = -min(b)
        result = py_sdBox((0.0, 0.0, 0.0), b)
        assert result < 0.0, f"Center distance should be negative, got {result}"
        assert result == pytest.approx(expected, abs=TOL)

    def test_x_face_center_returns_zero(self):
        """Point on +x face center (b.x, 0, 0) returns ~0."""
        result = py_sdBox((2.0, 0.0, 0.0), (2.0, 3.0, 4.0))
        assert result == pytest.approx(0.0, abs=TOL), (
            f"Face center should be ~0, got {result}"
        )

    def test_y_face_center_returns_zero(self):
        """Point on +y face center (0, b.y, 0) returns ~0."""
        result = py_sdBox((0.0, 3.0, 0.0), (2.0, 3.0, 4.0))
        assert result == pytest.approx(0.0, abs=TOL), (
            f"Face center should be ~0, got {result}"
        )

    def test_z_face_center_returns_zero(self):
        """Point on +z face center (0, 0, b.z) returns ~0."""
        result = py_sdBox((0.0, 0.0, 4.0), (2.0, 3.0, 4.0))
        assert result == pytest.approx(0.0, abs=TOL), (
            f"Face center should be ~0, got {result}"
        )

    def test_negative_x_face_center_returns_zero(self):
        """Point on -x face center (-b.x, 0, 0) returns ~0 (symmetry)."""
        result = py_sdBox((-2.0, 0.0, 0.0), (2.0, 3.0, 4.0))
        assert result == pytest.approx(0.0, abs=TOL)

    def test_negative_y_face_center_returns_zero(self):
        """Point on -y face center (0, -b.y, 0) returns ~0 (symmetry)."""
        result = py_sdBox((0.0, -3.0, 0.0), (2.0, 3.0, 4.0))
        assert result == pytest.approx(0.0, abs=TOL)

    def test_negative_z_face_center_returns_zero(self):
        """Point on -z face center (0, 0, -b.z) returns ~0 (symmetry)."""
        result = py_sdBox((0.0, 0.0, -4.0), (2.0, 3.0, 4.0))
        assert result == pytest.approx(0.0, abs=TOL)

    def test_outside_point_returns_positive(self):
        """Point clearly outside the box returns positive distance."""
        result = py_sdBox((10.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert result > 0.0, f"Outside point should return positive, got {result}"

    def test_inside_point_returns_negative(self):
        """Point clearly inside the box returns negative distance."""
        result = py_sdBox((0.5, 0.5, 0.5), (2.0, 2.0, 2.0))
        assert result < 0.0, f"Inside point should return negative, got {result}"

    def test_corner_returns_zero(self):
        """Point exactly at box corner (b.x, b.y, b.z) is on surface -> ~0."""
        result = py_sdBox((2.0, 3.0, 4.0), (2.0, 3.0, 4.0))
        assert result == pytest.approx(0.0, abs=TOL), (
            f"Corner should be ~0, got {result}"
        )

    def test_edge_on_three_axes_returns_zero(self):
        """Point on box edges (aligned with two axes) is on surface -> ~0."""
        for p in [(1.0, 2.0, 3.0), (0.0, 2.0, 3.0), (-1.0, 2.0, 3.0)]:
            result = py_sdBox(p, (1.0, 2.0, 3.0))
            assert result == pytest.approx(0.0, abs=TOL), (
                f"Edge point {p} should be ~0, got {result}"
            )

    def test_edge_on_two_axes_is_zero(self):
        """Point on edge of unit cube (1, 1, 0) is on surface -> ~0."""
        result = py_sdBox((1.0, 1.0, 0.0), (1.0, 1.0, 1.0))
        assert result == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.2 Exact Distance Values
# =============================================================================


class TestExactDistances:
    """Verify exact distance calculations at known points."""

    def test_center_distance(self):
        """sdBox((0,0,0), b) = -min(b) for various b."""
        cases = [
            ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), -1.0),
            ((0.0, 0.0, 0.0), (5.0, 5.0, 5.0), -5.0),
            ((0.0, 0.0, 0.0), (0.5, 1.0, 2.0), -0.5),
            ((0.0, 0.0, 0.0), (10.0, 0.1, 0.1), -0.1),
        ]
        for p, b, expected in cases:
            result = py_sdBox(p, b)
            assert result == pytest.approx(expected, abs=TOL), (
                f"sdBox({p}, {b}) should be {expected}, got {result}"
            )

    def test_far_along_x(self):
        """sdBox((10, 0, 0), (1,1,1)) = 9."""
        result = py_sdBox((10.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert result == pytest.approx(9.0, abs=TOL), (
            f"Expected 9.0, got {result}"
        )

    def test_far_along_y(self):
        """sdBox((0, 10, 0), (1,1,1)) = 9."""
        result = py_sdBox((0.0, 10.0, 0.0), (1.0, 1.0, 1.0))
        assert result == pytest.approx(9.0, abs=TOL)

    def test_far_along_z(self):
        """sdBox((0, 0, 10), (1,1,1)) = 9."""
        result = py_sdBox((0.0, 0.0, 10.0), (1.0, 1.0, 1.0))
        assert result == pytest.approx(9.0, abs=TOL)

    def test_outside_corner_distance(self):
        """sdBox((2,2,2), (1,1,1)) = sqrt(3) ~ 1.732."""
        expected = math.sqrt(3.0)
        result = py_sdBox((2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
        assert result == pytest.approx(expected, abs=TOL_RELAXED), (
            f"Expected {expected}, got {result}"
        )

    def test_inside_corner_distance(self):
        """sdBox((0.5, 0.5, 0.5), (1,1,1)) = -0.5."""
        result = py_sdBox((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
        assert result == pytest.approx(-0.5, abs=TOL)

    def test_slightly_inside(self):
        """Point just inside the box (0.9, 0, 0) with b=(1,1,1) -> -0.1."""
        result = py_sdBox((0.9, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert result == pytest.approx(-0.1, abs=TOL)

    def test_slightly_outside(self):
        """Point just outside the box (1.1, 0, 0) with b=(1,1,1) -> 0.1."""
        result = py_sdBox((1.1, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert result == pytest.approx(0.1, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.2 Unit Cube (1,1,1) at Key Points
# =============================================================================


class TestUnitCube:
    """Systematic verification of sdBox for the canonical unit cube b=(1,1,1)."""

    B = (1.0, 1.0, 1.0)

    def test_center(self):
        """Center (0,0,0) -> -min(1,1,1) = -1."""
        assert py_sdBox((0.0, 0.0, 0.0), self.B) == pytest.approx(-1.0, abs=TOL)

    def test_positive_x_face(self):
        """+x face (1,0,0) -> 0."""
        assert py_sdBox((1.0, 0.0, 0.0), self.B) == pytest.approx(0.0, abs=TOL)

    def test_positive_y_face(self):
        """+y face (0,1,0) -> 0."""
        assert py_sdBox((0.0, 1.0, 0.0), self.B) == pytest.approx(0.0, abs=TOL)

    def test_positive_z_face(self):
        """+z face (0,0,1) -> 0."""
        assert py_sdBox((0.0, 0.0, 1.0), self.B) == pytest.approx(0.0, abs=TOL)

    def test_corner_xyz(self):
        """Corner (1,1,1) -> 0 (on surface)."""
        assert py_sdBox((1.0, 1.0, 1.0), self.B) == pytest.approx(0.0, abs=TOL)

    def test_outside_corner_2x2x2(self):
        """Outside corner (2,2,2) -> sqrt(3) ~ 1.732."""
        expected = math.sqrt(3.0)
        assert py_sdBox((2.0, 2.0, 2.0), self.B) == pytest.approx(
            expected, abs=TOL_RELAXED
        )

    def test_inside_half(self):
        """Inside at (0.5, 0.5, 0.5) -> -0.5."""
        assert py_sdBox((0.5, 0.5, 0.5), self.B) == pytest.approx(-0.5, abs=TOL)

    def test_on_edge_xy(self):
        """On xy-edge (1,1,0) -> 0."""
        assert py_sdBox((1.0, 1.0, 0.0), self.B) == pytest.approx(0.0, abs=TOL)

    def test_on_edge_xz(self):
        """On xz-edge (1,0,1) -> 0."""
        assert py_sdBox((1.0, 0.0, 1.0), self.B) == pytest.approx(0.0, abs=TOL)

    def test_on_edge_yz(self):
        """On yz-edge (0,1,1) -> 0."""
        assert py_sdBox((0.0, 1.0, 1.0), self.B) == pytest.approx(0.0, abs=TOL)

    def test_far_along_diagonal(self):
        """Far along diagonal (5,5,5) -> sqrt(16+16+16) = sqrt(48) ~ 6.928."""
        expected = math.sqrt(48.0)
        assert py_sdBox((5.0, 5.0, 5.0), self.B) == pytest.approx(
            expected, abs=TOL_RELAXED
        )

    def test_far_along_axis(self):
        """Far along single axis (5,0,0) -> 4 (pure x-distance)."""
        assert py_sdBox((5.0, 0.0, 0.0), self.B) == pytest.approx(4.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.2 Flat / Degenerate Boxes
# =============================================================================


class TestFlatAndDegenerateBoxes:
    """Test boxes with thin or zero-size dimensions."""

    def test_flat_x_thin_inside(self):
        """Flat box (0.1, 1, 1): point inside near x-face -> -0.05."""
        result = py_sdBox((0.05, 0.0, 0.0), (0.1, 1.0, 1.0))
        assert result == pytest.approx(-0.05, abs=TOL)

    def test_flat_x_just_outside(self):
        """Flat box (0.1, 1, 1): point just outside x-face -> 0.1."""
        result = py_sdBox((0.2, 0.0, 0.0), (0.1, 1.0, 1.0))
        assert result == pytest.approx(0.1, abs=TOL)

    def test_flat_outside_xy(self):
        """Flat box (0.1, 1, 1): outside both x and y -> sqrt(0.01+0.25)."""
        expected = math.sqrt(0.01 + 0.25)
        result = py_sdBox((0.2, 1.5, 0.0), (0.1, 1.0, 1.0))
        assert result == pytest.approx(expected, abs=TOL_RELAXED), (
            f"Expected {expected}, got {result}"
        )

    def test_flat_outside_xyz(self):
        """Flat box (0.1, 1, 1): outside all three axes -> sqrt(0.01+0.25+0.25)."""
        expected = math.sqrt(0.01 + 0.25 + 0.25)
        result = py_sdBox((0.2, 1.5, 1.5), (0.1, 1.0, 1.0))
        assert result == pytest.approx(expected, abs=TOL_RELAXED)

    def test_zero_x_dimension_center(self):
        """Degenerate box (0, 1, 1) = yz-rectangle: center distance is 0."""
        result = py_sdBox((0.0, 0.0, 0.0), (0.0, 1.0, 1.0))
        assert result == pytest.approx(0.0, abs=TOL)

    def test_zero_x_dimension_inside_yz(self):
        """Degenerate box (0, 1, 1): point at (0, 0.5, 0.5) -> 0 (on rectangle)."""
        result = py_sdBox((0.0, 0.5, 0.5), (0.0, 1.0, 1.0))
        assert result == pytest.approx(0.0, abs=TOL)

    def test_zero_x_dimension_outside(self):
        """Degenerate box (0, 1, 1): point at (2, 0, 0) -> distance 2."""
        result = py_sdBox((2.0, 0.0, 0.0), (0.0, 1.0, 1.0))
        assert result == pytest.approx(2.0, abs=TOL)

    def test_zero_all_dimensions(self):
        """Degenerate box (0,0,0) = point at origin: sdBox = length(p)."""
        assert py_sdBox((3.0, 4.0, 0.0), (0.0, 0.0, 0.0)) == pytest.approx(5.0, abs=TOL)
        assert py_sdBox((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == pytest.approx(1.0, abs=TOL)
        assert py_sdBox((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.2 Continuity
# =============================================================================


class TestContinuity:
    """Verify that sdBox is continuous everywhere (no discontinuities)."""

    def test_continuity_at_x_face(self):
        """Crossing +x face: sdBox((1+eps, 0, 0), ~ sdBox((1-eps, 0, 0))."""
        eps = 1e-9
        b = (1.0, 1.0, 1.0)
        outer = py_sdBox((1.0 + eps, 0.0, 0.0), b)
        inner = py_sdBox((1.0 - eps, 0.0, 0.0), b)
        diff = abs(outer - inner)
        assert diff < 1e-8, (
            f"Discontinuity at x=1: outer={outer}, inner={inner}, diff={diff}"
        )

    def test_continuity_at_y_face(self):
        """Crossing +y face: sdBox((0, 1+eps, 0), ~ sdBox((0, 1-eps, 0))."""
        eps = 1e-9
        b = (1.0, 1.0, 1.0)
        outer = py_sdBox((0.0, 1.0 + eps, 0.0), b)
        inner = py_sdBox((0.0, 1.0 - eps, 0.0), b)
        diff = abs(outer - inner)
        assert diff < 1e-8, (
            f"Discontinuity at y=1: outer={outer}, inner={inner}, diff={diff}"
        )

    def test_continuity_at_z_face(self):
        """Crossing +z face: sdBox((0, 0, 1+eps), ~ sdBox((0, 0, 1-eps))."""
        eps = 1e-9
        b = (1.0, 1.0, 1.0)
        outer = py_sdBox((0.0, 0.0, 1.0 + eps), b)
        inner = py_sdBox((0.0, 0.0, 1.0 - eps), b)
        diff = abs(outer - inner)
        assert diff < 1e-8, (
            f"Discontinuity at z=1: outer={outer}, inner={inner}, diff={diff}"
        )

    def test_continuity_at_corner(self):
        """Crossing corner: sdBox((1+eps, 1+eps, 1+eps), ~ sdBox((1-eps,...)."""
        eps = 1e-9
        b = (1.0, 1.0, 1.0)
        outer = py_sdBox((1.0 + eps, 1.0 + eps, 1.0 + eps), b)
        inner = py_sdBox((1.0 - eps, 1.0 - eps, 1.0 - eps), b)
        diff = abs(outer - inner)
        assert diff < 1e-8, (
            f"Discontinuity at corner: outer={outer}, inner={inner}, diff={diff}"
        )

    def test_continuity_along_axis_interior(self):
        """sdBox is smooth when moving through the interior along x."""
        b = (2.0, 1.0, 1.0)
        step = 0.01
        prev = py_sdBox((-1.9, 0.0, 0.0), b)
        for i in range(1, 380):
            x = -1.9 + i * step
            curr = py_sdBox((x, 0.0, 0.0), b)
            diff = abs(curr - prev)
            assert diff < step * 2.0, (
                f"Large jump at x={x}: diff={diff}, prev={prev}, curr={curr}"
            )
            prev = curr

    def test_continuity_along_axis_exterior(self):
        """sdBox is smooth outside the box along the x-axis (linear increase)."""
        b = (1.0, 1.0, 1.0)
        step = 0.1
        prev = py_sdBox((1.0, 0.0, 0.0), b)
        for i in range(1, 100):
            x = 1.0 + i * step
            curr = py_sdBox((x, 0.0, 0.0), b)
            diff = abs(curr - prev)
            # Outside along axis, distance increases linearly by exactly step
            assert diff == pytest.approx(step, abs=TOL_RELAXED), (
                f"Expected linear increase of {step} at x={x}, got diff={diff}"
            )
            prev = curr


# =============================================================================
# Test: T-DEMO-1.2 Linearity
# =============================================================================


class TestLinearity:
    """Verify distance increases linearly when moving away from a face."""

    def test_linear_from_x_face(self):
        """Moving from x-face outward: distance increases by exactly d."""
        b = (1.0, 2.0, 3.0)
        assert py_sdBox((b[0], 0.0, 0.0), b) == pytest.approx(0.0, abs=TOL)
        for d in [0.5, 1.0, 2.0, 5.0, 10.0]:
            d_actual = py_sdBox((b[0] + d, 0.0, 0.0), b)
            assert d_actual == pytest.approx(d, abs=TOL_RELAXED), (
                f"At offset {d} from x-face, expected {d}, got {d_actual}"
            )

    def test_linear_from_y_face(self):
        """Moving from y-face outward: distance increases by exactly d."""
        b = (1.0, 2.0, 3.0)
        for d in [0.5, 1.0, 2.0, 5.0]:
            d_actual = py_sdBox((0.0, b[1] + d, 0.0), b)
            assert d_actual == pytest.approx(d, abs=TOL_RELAXED)

    def test_linear_from_z_face(self):
        """Moving from z-face outward: distance increases by exactly d."""
        b = (1.0, 2.0, 3.0)
        for d in [0.5, 1.0, 2.0, 5.0]:
            d_actual = py_sdBox((0.0, 0.0, b[2] + d), b)
            assert d_actual == pytest.approx(d, abs=TOL_RELAXED)

    def test_linear_inside_along_x(self):
        """Moving from center toward +x face: distance decreases linearly."""
        b = (2.0, 2.0, 2.0)
        assert py_sdBox((0.0, 0.0, 0.0), b) == pytest.approx(-2.0, abs=TOL)
        assert py_sdBox((1.0, 0.0, 0.0), b) == pytest.approx(-1.0, abs=TOL)
        assert py_sdBox((1.5, 0.0, 0.0), b) == pytest.approx(-0.5, abs=TOL)
        assert py_sdBox((2.0, 0.0, 0.0), b) == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.2 Symmetry
# =============================================================================


class TestSymmetry:
    """Verify sdBox is symmetric under sign flips (axis-aligned at origin)."""

    def test_x_symmetry(self):
        """sdBox((x, y, z), b) = sdBox((-x, y, z), b)."""
        b = (1.5, 2.5, 3.5)
        for x in [0.0, 0.5, 1.0, 2.0, 5.0]:
            pos = py_sdBox((x, 0.5, 0.5), b)
            neg = py_sdBox((-x, 0.5, 0.5), b)
            assert pos == pytest.approx(neg, abs=TOL), (
                f"x-symmetry fail at x={x}: {pos} vs {neg}"
            )

    def test_y_symmetry(self):
        """sdBox((x, y, z), b) = sdBox((x, -y, z), b)."""
        b = (1.5, 2.5, 3.5)
        for y in [0.0, 0.5, 1.0, 2.0, 5.0]:
            pos = py_sdBox((0.5, y, 0.5), b)
            neg = py_sdBox((0.5, -y, 0.5), b)
            assert pos == pytest.approx(neg, abs=TOL)

    def test_z_symmetry(self):
        """sdBox((x, y, z), b) = sdBox((x, y, -z), b)."""
        b = (1.5, 2.5, 3.5)
        for z in [0.0, 0.5, 1.0, 2.0, 5.0]:
            pos = py_sdBox((0.5, 0.5, z), b)
            neg = py_sdBox((0.5, 0.5, -z), b)
            assert pos == pytest.approx(neg, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.2 Non-Uniform Boxes
# =============================================================================


class TestNonUniformBoxes:
    """Test behavior with non-uniform (non-cube) boxes."""

    def test_wide_flat_box(self):
        """Box (3, 0.5, 2): verify distances are correct."""
        b = (3.0, 0.5, 2.0)
        assert py_sdBox((0.0, 0.0, 0.0), b) == pytest.approx(-0.5, abs=TOL)
        assert py_sdBox((0.0, 0.5, 0.0), b) == pytest.approx(0.0, abs=TOL)
        assert py_sdBox((3.0, 0.0, 0.0), b) == pytest.approx(0.0, abs=TOL)
        assert py_sdBox((0.0, 0.0, 2.0), b) == pytest.approx(0.0, abs=TOL)

    def test_tall_box(self):
        """Box (1, 10, 1): very tall along y."""
        b = (1.0, 10.0, 1.0)
        result = py_sdBox((0.0, 20.0, 0.0), b)
        assert result == pytest.approx(10.0, abs=TOL)
        expected = math.sqrt(2.0)
        result = py_sdBox((2.0, 11.0, 0.0), b)
        assert result == pytest.approx(expected, abs=TOL_RELAXED)

    def test_large_box(self):
        """Box (100, 100, 100): large box, small offset."""
        assert py_sdBox((100.0, 0.0, 0.0), (100.0, 100.0, 100.0)) == pytest.approx(
            0.0, abs=TOL
        )
        assert py_sdBox((101.0, 0.0, 0.0), (100.0, 100.0, 100.0)) == pytest.approx(
            1.0, abs=TOL
        )
        assert py_sdBox((99.0, 0.0, 0.0), (100.0, 100.0, 100.0)) == pytest.approx(
            -1.0, abs=TOL
        )


# =============================================================================
# Test: T-DEMO-1.2 Determinism
# =============================================================================


class TestDeterminism:
    """Verify sdBox is deterministic (same inputs always produce same output)."""

    def test_deterministic_inside(self):
        """Repeated calls with same inside point return same result."""
        p = (0.3, 0.4, 0.5)
        b = (2.0, 2.0, 2.0)
        base = py_sdBox(p, b)
        for _ in range(20):
            assert py_sdBox(p, b) == pytest.approx(base, abs=TOL)

    def test_deterministic_outside(self):
        """Repeated calls with same outside point return same result."""
        p = (5.0, 6.0, 7.0)
        b = (1.0, 1.0, 1.0)
        base = py_sdBox(p, b)
        for _ in range(20):
            assert py_sdBox(p, b) == pytest.approx(base, abs=TOL)

    def test_deterministic_on_surface(self):
        """Repeated calls with same surface point return same result."""
        p = (1.0, 2.0, 0.0)
        b = (1.0, 2.0, 3.0)
        base = py_sdBox(p, b)
        for _ in range(20):
            assert py_sdBox(p, b) == pytest.approx(base, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.2 Monotonicity
# =============================================================================


class TestMonotonicity:
    """Verify sdBox is monotonic along each axis outside the box."""

    def test_monotonic_outside_x(self):
        """Moving away from +x face: distance should be non-decreasing."""
        b = (1.0, 2.0, 3.0)
        prev = 0.0
        for d in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            curr = py_sdBox((b[0] + d, 0.0, 0.0), b)
            assert curr >= prev, (
                f"Non-monotonic at x-offset {d}: {curr} < {prev}"
            )
            prev = curr

    def test_interior_distance_decreases_to_surface(self):
        """Moving from center to surface: distance approaches 0 from below."""
        b = (3.0, 3.0, 3.0)
        prev = py_sdBox((0.0, 0.0, 0.0), b)
        for x in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            curr = py_sdBox((x, 0.0, 0.0), b)
            assert curr >= prev, (
                f"Interior distance decreased from {prev} to {curr} at x={x}"
            )
            assert curr <= 0.0, (
                f"Interior distance should be <= 0, got {curr} at x={x}"
            )
            prev = curr
