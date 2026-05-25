"""
Whitebox tests for sdRoundedBox WGSL function (T-DEMO-1.10).

Tests the implementation-aware signed distance function for a box with rounded
corners, using the WGSL formula:
    q = abs(p) - b + safe_r
    sd = length(max(q, 0)) + min(maxComponent(q), 0) - safe_r

where safe_r = clamp(r, 0, min(b)).

The rounded box is centered at the origin with half-dimensions b (so the
un-rounded box extends from -b to +b along each axis), and uniform corner
radius r.

Implementation (engine/rendering/demoscene/wgsl/sdf_rounded_box.wgsl):
    fn sdRoundedBox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
        let safe_r = min(max(r, 0.0), min(b.x, min(b.y, b.z)));
        let q = abs(p) - b + vec3(safe_r);
        let exterior = length(max(q, vec3(0.0)));
        let interior = min(max(q.x, max(q.y, q.z)), 0.0);
        return exterior + interior - safe_r;
    }

Key behaviors:
  - r=0 degenerates to a sharp-edged box (sdBox)
  - r clamped to [0, min(b)] -- safe_r never exceeds the smallest dimension
  - Face surface is at b (the original box face) -- safe_r shifts q outward,
    making the surface align with the original box face
  - Interior: exterior term is 0 when fully inside, interior term is the
    least-negative max-component (which is -min(b) at center)

WHITEBOX coverage plan:
  Path 1:  Formula q = abs(p) - b -- per-axis signed distance decomposition
  Path 2:  safe_r = clamp(r, 0, min(b)) -- radius clamping behavior
  Path 3:  Exterior term = length(max(q, 0))  (no safe_r subtraction)
  Path 4:  Interior term = min(max(q.x, q.y, q.z), 0)
  Path 5:  r = 0 -- degenerate to sharp-edged box (sdBox)
  Path 6:  r >= min(b) -- maximally rounded, sphere-like limiting shape
  Path 7:  Corner rounding radius -- transition from face to corner
  Path 8:  Inside the box -- negative signed distance
  Path 9:  On the surface -- zero signed distance
  Path 10: Outside the box -- positive signed distance
  Path 11: Symmetry -- sign independence via abs(p)
  Path 12: Sign convention -- negative inside, zero on surface, positive outside
  Path 13: Continuity and monotonicity
  Path 14: Parameter variation and edge cases
  Path 15: Combined formula -- full exterior + interior integration
  Path 16: Cross-validated edge cases at various positions
"""

from __future__ import annotations

import math

import pytest

# =============================================================================
# Python model of sdRoundedBox matching WGSL semantics exactly (whitebox)
# =============================================================================


def py_sd_rounded_box(p, b, r):
    """Python model of WGSL sdRoundedBox(p, b, r) -> f32.

    Signed distance from point p (3-tuple) to a box of half-dimensions b
    (3-tuple) with uniform corner rounding radius r.

    WGSL formula:
        safe_r = clamp(r, 0, min(b))
        q = abs(p) - b + safe_r
        exterior = length(max(q, 0))
        interior = min(max(q.x, q.y, q.z), 0)
        sd = exterior + interior - safe_r
    """
    safe_r = min(max(r, 0.0), min(b[0], min(b[1], b[2])))
    qx = abs(p[0]) - b[0] + safe_r
    qy = abs(p[1]) - b[1] + safe_r
    qz = abs(p[2]) - b[2] + safe_r

    mx = max(qx, 0.0)
    my = max(qy, 0.0)
    mz = max(qz, 0.0)
    exterior = math.sqrt(mx * mx + my * my + mz * mz)

    interior = min(max(qx, max(qy, qz)), 0.0)

    return exterior + interior - safe_r


def py_q(p, b):
    """Compute q = abs(p) - b (per-axis signed distance to the box faces)."""
    return (abs(p[0]) - b[0], abs(p[1]) - b[1], abs(p[2]) - b[2])


def py_exterior(p, b, r):
    """Compute exterior = length(max(q, 0)) where q = abs(p) - b + safe_r."""
    safe_r = min(max(r, 0.0), min(b[0], min(b[1], b[2])))
    qx = abs(p[0]) - b[0] + safe_r
    qy = abs(p[1]) - b[1] + safe_r
    qz = abs(p[2]) - b[2] + safe_r
    mx = max(qx, 0.0)
    my = max(qy, 0.0)
    mz = max(qz, 0.0)
    return math.sqrt(mx * mx + my * my + mz * mz)


def py_interior(p, b, r=0.0):
    """Compute interior = min(max(q.x, q.y, q.z), 0), with q shifted by safe_r."""
    safe_r = min(max(r, 0.0), min(b[0], min(b[1], b[2])))
    qx = abs(p[0]) - b[0] + safe_r
    qy = abs(p[1]) - b[1] + safe_r
    qz = abs(p[2]) - b[2] + safe_r
    return min(max(qx, max(qy, qz)), 0.0)


def py_safe_r(b, r):
    """Compute safe_r = clamp(r, 0, min(b))."""
    return min(max(r, 0.0), min(b[0], min(b[1], b[2])))


# =============================================================================
# Tolerance constants
# =============================================================================

TOL = 1e-12         # General tolerance for exact arithmetic
TOL_SURFACE = 1e-12  # Surface tolerance


# =============================================================================
# Path 1: Formula q = abs(p) - b
# =============================================================================


class TestFormulaQ:
    """Verify q = abs(p) - b gives the per-axis signed distance to box faces.

    For each axis, q_i = 0 means the point is exactly on that face,
    q_i < 0 means inside the box along that axis,
    q_i > 0 means outside the box along that axis.
    """

    def test_q_inside_all_axes(self):
        """Inside box: all q components negative."""
        p = (1.0, 0.5, 0.3)
        b = (3.0, 2.0, 1.0)
        q = py_q(p, b)
        assert q[0] == pytest.approx(-2.0, abs=TOL)
        assert q[1] == pytest.approx(-1.5, abs=TOL)
        assert q[2] == pytest.approx(-0.7, abs=TOL)

    def test_q_on_x_face(self):
        """On the x-face: q.x = 0, others negative."""
        p = (3.0, 0.0, 0.0)
        b = (3.0, 2.0, 1.0)
        q = py_q(p, b)
        assert q[0] == pytest.approx(0.0, abs=TOL)
        assert q[1] == pytest.approx(-2.0, abs=TOL)
        assert q[2] == pytest.approx(-1.0, abs=TOL)

    def test_q_outside_x_axis(self):
        """Outside on x-axis: q.x positive, others negative."""
        p = (5.0, 0.0, 0.0)
        b = (3.0, 2.0, 1.0)
        q = py_q(p, b)
        assert q[0] == pytest.approx(2.0, abs=TOL)
        assert q[1] == pytest.approx(-2.0, abs=TOL)
        assert q[2] == pytest.approx(-1.0, abs=TOL)

    def test_q_outside_corner_all_positive(self):
        """Outside corner: all q components positive."""
        p = (5.0, 4.0, 3.0)
        b = (3.0, 2.0, 1.0)
        q = py_q(p, b)
        assert q[0] == pytest.approx(2.0, abs=TOL)
        assert q[1] == pytest.approx(2.0, abs=TOL)
        assert q[2] == pytest.approx(2.0, abs=TOL)

    def test_q_abs_handles_negative_p(self):
        """abs(p) ensures negative coordinates are treated like positive."""
        p_neg = (-1.5, -0.5, 0.0)
        p_pos = (1.5, 0.5, 0.0)
        b = (3.0, 2.0, 1.0)
        q_neg = py_q(p_neg, b)
        q_pos = py_q(p_pos, b)
        assert q_neg[0] == pytest.approx(q_pos[0], abs=TOL)
        assert q_neg[1] == pytest.approx(q_pos[1], abs=TOL)
        assert q_neg[2] == pytest.approx(q_pos[2], abs=TOL)

    def test_q_on_y_face_positive(self):
        """On y-face with positive y: q.y = 0."""
        p = (0.0, 2.0, 0.0)
        b = (3.0, 2.0, 1.0)
        q = py_q(p, b)
        assert q[1] == pytest.approx(0.0, abs=TOL)

    def test_q_on_y_face_negative(self):
        """On y-face with negative y: abs(neg) gives q.y = 0."""
        p = (0.0, -2.0, 0.0)
        b = (3.0, 2.0, 1.0)
        q = py_q(p, b)
        assert q[1] == pytest.approx(0.0, abs=TOL)

    def test_q_on_z_face(self):
        """On z-face: q.z = 0."""
        p = (0.0, 0.0, 1.0)
        b = (3.0, 2.0, 1.0)
        q = py_q(p, b)
        assert q[2] == pytest.approx(0.0, abs=TOL)

    def test_q_zero_at_origin(self):
        """At origin, all q components = -b (most negative interior)."""
        b = (3.0, 2.0, 1.0)
        q = py_q((0.0, 0.0, 0.0), b)
        assert q[0] == pytest.approx(-3.0, abs=TOL)
        assert q[1] == pytest.approx(-2.0, abs=TOL)
        assert q[2] == pytest.approx(-1.0, abs=TOL)

    def test_q_mixed_signs(self):
        """Mixed signs: some axes inside, one outside."""
        p = (4.0, 1.0, 0.0)
        b = (3.0, 2.0, 1.0)
        q = py_q(p, b)
        # x: outside (positive), y: inside (negative), z: on face (zero)
        assert q[0] > 0.0
        assert q[1] < 0.0
        assert q[2] == pytest.approx(-1.0, abs=TOL)

    def test_q_with_asymmetric_b(self):
        """Works correctly for asymmetric box dimensions."""
        p = (0.0, 0.0, 0.0)
        b = (5.0, 0.5, 10.0)
        q = py_q(p, b)
        assert q[0] == pytest.approx(-5.0, abs=TOL)
        assert q[1] == pytest.approx(-0.5, abs=TOL)
        assert q[2] == pytest.approx(-10.0, abs=TOL)


# =============================================================================
# Path 2: safe_r = clamp(r, 0, min(b))
# =============================================================================


class TestSafeR:
    """Verify safe_r = clamp(r, 0, min(b)) ensures the corner radius is
    clamped to a valid range: 0 <= safe_r <= min(b.x, b.y, b.z).

    This prevents negative radii and prevents the rounding from exceeding
    the smallest box dimension.
    """

    def test_positive_r_less_than_min_b(self):
        """Positive r < min(b): safe_r = r."""
        b = (3.0, 2.0, 1.0)
        assert py_safe_r(b, 0.5) == pytest.approx(0.5, abs=TOL)

    def test_r_equal_min_b(self):
        """r = min(b): safe_r = r (at upper limit)."""
        b = (3.0, 2.0, 1.0)
        assert py_safe_r(b, 1.0) == pytest.approx(1.0, abs=TOL)

    def test_r_greater_than_min_b(self):
        """r > min(b): safe_r = min(b) (clamped)."""
        b = (3.0, 2.0, 1.0)
        assert py_safe_r(b, 5.0) == pytest.approx(1.0, abs=TOL)
        assert py_safe_r(b, 100.0) == pytest.approx(1.0, abs=TOL)

    def test_r_zero(self):
        """r = 0: safe_r = 0 (no rounding, sharp box)."""
        b = (3.0, 2.0, 1.0)
        assert py_safe_r(b, 0.0) == pytest.approx(0.0, abs=TOL)

    def test_r_negative(self):
        """r < 0: safe_r = 0 (negatives clamped to zero)."""
        b = (3.0, 2.0, 1.0)
        assert py_safe_r(b, -1.0) == pytest.approx(0.0, abs=TOL)
        assert py_safe_r(b, -0.5) == pytest.approx(0.0, abs=TOL)
        assert py_safe_r(b, -100.0) == pytest.approx(0.0, abs=TOL)

    def test_min_b_from_different_dimensions(self):
        """safe_r uses the minimum of all three half-dimensions."""
        # Different dimensions, each one being the min in different cases
        assert py_safe_r((0.5, 3.0, 3.0), 0.3) == pytest.approx(0.3, abs=TOL)
        assert py_safe_r((3.0, 0.5, 3.0), 0.3) == pytest.approx(0.3, abs=TOL)
        assert py_safe_r((3.0, 3.0, 0.5), 0.3) == pytest.approx(0.3, abs=TOL)
        assert py_safe_r((0.5, 3.0, 3.0), 0.8) == pytest.approx(0.5, abs=TOL)

    def test_r_between_zero_and_min_b(self):
        """Various r values between 0 and min(b): safe_r = r."""
        b = (10.0, 5.0, 3.0)
        for r in [0.1, 0.5, 1.0, 2.0, 2.9, 3.0]:
            assert py_safe_r(b, r) == pytest.approx(r, abs=TOL), (
                f"safe_r for r={r} should equal r (since r <= min(b)=3)"
            )

    def test_r_just_above_min_b(self):
        """r just above min(b): safe_r = min(b) (not r)."""
        b = (2.0, 2.0, 2.0)
        assert py_safe_r(b, 2.001) == pytest.approx(2.0, abs=TOL)

    def test_cube_min_all_equal(self):
        """For a cube, min(b) = b.x = b.y = b.z."""
        b = (4.0, 4.0, 4.0)
        assert py_safe_r(b, 3.0) == pytest.approx(3.0, abs=TOL)
        assert py_safe_r(b, 4.0) == pytest.approx(4.0, abs=TOL)
        assert py_safe_r(b, 5.0) == pytest.approx(4.0, abs=TOL)

    def test_safe_r_affects_sd(self):
        """Center sd = -min(b) regardless of r (under WGSL formula)."""
        b = (3.0, 2.0, 1.0)
        p = (0.0, 0.0, 0.0)
        # With WGSL formula q = abs(p) - b + safe_r, center sd = -min(b) for all r
        for r in [0.0, 0.1, 0.5, 0.8, 1.0]:
            d = py_sd_rounded_box(p, b, r)
            assert d == pytest.approx(-1.0, abs=TOL), (
                f"Center sd should be -min(b)=-1.0, got {d} for r={r}"
            )


# =============================================================================
# Path 3: Exterior term = length(max(q, 0)) - safe_r
# =============================================================================


class TestExteriorTerm:
    """Verify exterior = length(max(q, vec3(0))).

    With the WGSL formula, q = abs(p) - b + safe_r. When all q components are
    negative (inside the box), max(q,0) = (0,0,0), so exterior = 0. When some
    q are positive, exterior includes the Euclidean distance to the box corner.
    """

    def test_exterior_all_inside(self):
        """All q negative: exterior = 0 (max(q,0) = (0,0,0))."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        p = (0.0, 0.0, 0.0)
        e = py_exterior(p, b, r)
        # q = (-2.5, -1.5, -0.5), max(q,0) = (0,0,0), length = 0
        # exterior = 0 (no safe_r subtraction at this level)
        assert e == pytest.approx(0.0, abs=TOL)

    def test_exterior_one_axis_outside(self):
        """One q component positive: exterior includes that component (shifted by safe_r)."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        p = (5.0, 0.0, 0.0)  # q = (2.5, -1.5, -0.5)
        e = py_exterior(p, b, r)
        # max(q,0) = (2.5, 0, 0), length = 2.5
        # exterior = 2.5
        assert e == pytest.approx(2.5, abs=TOL)

    def test_exterior_two_axes_outside(self):
        """Two q components positive: exterior = sqrt(qx^2 + qy^2) (shifted by safe_r)."""
        b = (3.0, 2.0, 1.0)
        r = 0.3
        p = (5.0, 4.0, 0.0)  # q = (2.3, 2.3, -0.7)
        e = py_exterior(p, b, r)
        expected = math.sqrt(2.3 * 2.3 + 2.3 * 2.3)  # sqrt(2*2.3^2)
        assert e == pytest.approx(expected, abs=TOL)

    def test_exterior_all_axes_outside(self):
        """All q positive: exterior = length(q) (shifted by safe_r)."""
        b = (3.0, 2.0, 1.0)
        r = 0.4
        p = (5.0, 4.0, 3.0)  # q = (2.4, 2.4, 2.4)
        e = py_exterior(p, b, r)
        expected = math.sqrt(3 * 2.4 * 2.4)  # sqrt(3*2.4^2)
        assert e == pytest.approx(expected, abs=TOL)

    def test_exterior_zero_r(self):
        """With r=0, exterior = length(max(q,0))."""
        b = (3.0, 2.0, 1.0)
        p = (5.0, 0.0, 0.0)
        e = py_exterior(p, b, 0.0)
        assert e == pytest.approx(2.0, abs=TOL)  # exactly q.x = 2

    def test_exterior_on_face_surface(self):
        """On the face surface (b along axis): exterior = safe_r."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        p = (3.0, 0.0, 0.0)  # q = (0.5, -1.5, -0.5)
        e = py_exterior(p, b, r)
        # max(q,0) = (0.5, 0, 0), length = 0.5
        # exterior = 0.5 (no safe_r subtraction)
        assert e == pytest.approx(0.5, abs=TOL)

    def test_exterior_negative_r_clamped(self):
        """Negative r is clamped to 0, so exterior = length(max(q,0))."""
        b = (3.0, 2.0, 1.0)
        p = (5.0, 0.0, 0.0)
        e_neg = py_exterior(p, b, -2.0)
        e_zero = py_exterior(p, b, 0.0)
        assert e_neg == pytest.approx(e_zero, abs=TOL)

    def test_exterior_deep_inside(self):
        """Deep inside: exterior = 0 regardless of position (all q shifted negative)."""
        b = (5.0, 5.0, 5.0)
        r = 1.0
        assert py_exterior((0.0, 0.0, 0.0), b, r) == pytest.approx(0.0, abs=TOL)
        assert py_exterior((1.0, 2.0, 0.0), b, r) == pytest.approx(0.0, abs=TOL)
        assert py_exterior((-3.0, 0.0, 2.0), b, r) == pytest.approx(0.0, abs=TOL)

    def test_exterior_corner_surface(self):
        """On the 3D corner surface: length(q) = r, exterior = r."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        c = r / math.sqrt(3.0)
        p = (b[0] + c - r, b[1] + c - r, b[2] + c - r)
        e = py_exterior(p, b, r)
        # exterior = length(q) = r (safe_r subtracted at sd level)
        assert e == pytest.approx(r, abs=TOL)

    def test_exterior_far_approaches_radial(self):
        """Far from box: exterior approximates qx = x - b.x + safe_r."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        far_x = 1e6
        p = (far_x, 0.0, 0.0)
        e = py_exterior(p, b, r)
        expected = far_x - b[0] + r  # qx = x - b.x + safe_r
        assert e == pytest.approx(expected, abs=0.001)


# =============================================================================
# Path 4: Interior term = min(max(q.x, q.y, q.z), 0)
# =============================================================================


class TestInteriorTerm:
    """Verify interior = min(max(q.x, q.y, q.z), 0).

    The interior term is always <= 0. It represents how far inside the box
    the point is: the maximum of the three q components (least-negative),
    clamped to <= 0. When any q component is positive, interior = 0.
    """

    def test_interior_all_negative(self):
        """All q negative: interior = max(q.x, q.y, q.z) (the least negative)."""
        # q = (-2, -1.5, -0.7), max = -0.7
        p = (1.0, 0.5, 0.3)
        b = (3.0, 2.0, 1.0)
        interior = py_interior(p, b)
        assert interior == pytest.approx(-0.7, abs=TOL)

    def test_interior_at_center(self):
        """At center: q = (-b.x, -b.y, -b.z), max = -min(b)."""
        b = (3.0, 2.0, 1.0)
        interior = py_interior((0.0, 0.0, 0.0), b)
        assert interior == pytest.approx(-1.0, abs=TOL)  # -min(b)

    def test_interior_one_positive(self):
        """One q positive: interior = 0."""
        b = (3.0, 2.0, 1.0)
        p = (5.0, 0.0, 0.0)  # q = (2, -2, -1), max = 2
        interior = py_interior(p, b)
        assert interior == pytest.approx(0.0, abs=TOL)

    def test_interior_two_positive(self):
        """Two q positive: interior = 0."""
        b = (3.0, 2.0, 1.0)
        p = (5.0, 4.0, 0.0)  # q = (2, 2, -1), max = 2
        interior = py_interior(p, b)
        assert interior == pytest.approx(0.0, abs=TOL)

    def test_interior_all_positive(self):
        """All q positive: interior = 0."""
        b = (3.0, 2.0, 1.0)
        p = (5.0, 4.0, 3.0)  # q = (2, 2, 2), max = 2
        interior = py_interior(p, b)
        assert interior == pytest.approx(0.0, abs=TOL)

    def test_interior_on_x_face(self):
        """On x-face: q.x = 0, others negative: max = 0, interior = 0."""
        b = (3.0, 2.0, 1.0)
        p = (3.0, 0.0, 0.0)  # q = (0, -2, -1), max = 0
        interior = py_interior(p, b)
        assert interior == pytest.approx(0.0, abs=TOL)

    def test_interior_on_y_face(self):
        """On y-face: q.y = 0, max = 0, interior = 0."""
        b = (3.0, 2.0, 1.0)
        p = (0.0, 2.0, 0.0)
        interior = py_interior(p, b)
        assert interior == pytest.approx(0.0, abs=TOL)

    def test_interior_on_z_face(self):
        """On z-face: q.z = 0, max = 0, interior = 0."""
        b = (3.0, 2.0, 1.0)
        p = (0.0, 0.0, 1.0)
        interior = py_interior(p, b)
        assert interior == pytest.approx(0.0, abs=TOL)

    def test_interior_x_dominates(self):
        """q.x is least negative: interior = q.x."""
        b = (5.0, 10.0, 10.0)
        p = (4.0, 0.0, 0.0)  # q = (-1, -10, -10), max = -1
        interior = py_interior(p, b)
        assert interior == pytest.approx(-1.0, abs=TOL)

    def test_interior_y_dominates(self):
        """q.y is least negative: interior = q.y."""
        b = (10.0, 5.0, 10.0)
        p = (0.0, 4.0, 0.0)  # q = (-10, -1, -10), max = -1
        interior = py_interior(p, b)
        assert interior == pytest.approx(-1.0, abs=TOL)

    def test_interior_z_dominates(self):
        """q.z is least negative: interior = q.z."""
        b = (10.0, 10.0, 5.0)
        p = (0.0, 0.0, 4.0)  # q = (-10, -10, -1), max = -1
        interior = py_interior(p, b)
        assert interior == pytest.approx(-1.0, abs=TOL)

    def test_interior_always_non_positive(self):
        """Interior term should always be <= 0."""
        b = (3.0, 2.0, 1.0)
        test_points = [
            (0.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (5.0, 4.0, 0.0),
            (-3.0, 2.0, 1.0),
            (1.0, -0.5, 0.3),
            (10.0, 10.0, 10.0),
        ]
        for p in test_points:
            interior = py_interior(p, b)
            assert interior <= 1e-15, (
                f"Interior term should be <= 0 for {p}, got {interior}"
            )


# =============================================================================
# Path 5: r = 0 -- degenerate to sharp-edged box (sdBox)
# =============================================================================


class TestRadiusZero:
    """With r = 0, the rounded box degenerates to a sharp-edged box.

    The formula reduces to:
        sd = length(max(q, 0)) + min(maxComponent(q), 0)
    which is the standard sdBox formula.
    """

    def test_center_at_origin(self):
        """At center of sharp box: sd = -min(b)."""
        b = (3.0, 2.0, 1.0)
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, 0.0)
        # exterior = 0, interior = -1, result = -1
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_center_cube(self):
        """Center of cube b=(s,s,s): sd = -s."""
        b = (2.0, 2.0, 2.0)
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, 0.0)
        assert d == pytest.approx(-2.0, abs=TOL)

    def test_on_x_face(self):
        """On x-face of sharp box: sd = 0."""
        b = (3.0, 2.0, 1.0)
        d = py_sd_rounded_box((3.0, 0.0, 0.0), b, 0.0)
        # q = (0, -2, -1), max(q,0) = (0,0,0), length = 0, exterior = 0
        # maxC = 0, interior = 0, result = 0
        assert d == pytest.approx(0.0, abs=TOL)

    def test_on_y_face(self):
        """On y-face of sharp box: sd = 0."""
        b = (3.0, 2.0, 1.0)
        d = py_sd_rounded_box((0.0, 2.0, 0.0), b, 0.0)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_on_z_face(self):
        """On z-face of sharp box: sd = 0."""
        b = (3.0, 2.0, 1.0)
        d = py_sd_rounded_box((0.0, 0.0, 1.0), b, 0.0)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_on_rim_xy(self):
        """On xy-edge (rim) of sharp box: sd = 0, q.x=0, q.y=0."""
        b = (3.0, 2.0, 1.0)
        d = py_sd_rounded_box((3.0, 2.0, 0.0), b, 0.0)
        # q = (0, 0, -1), max = (0,0,0), length = 0, maxC = 0, result = 0
        assert d == pytest.approx(0.0, abs=TOL)

    def test_on_corner(self):
        """On corner of sharp box: sd = 0."""
        b = (3.0, 2.0, 1.0)
        d = py_sd_rounded_box((3.0, 2.0, 1.0), b, 0.0)
        # q = (0, 0, 0), max = (0,0,0), length = 0, maxC = 0, result = 0
        assert d == pytest.approx(0.0, abs=TOL)

    def test_outside_x_axis(self):
        """Outside on x-axis: sd = x - b.x."""
        b = (3.0, 2.0, 1.0)
        d = py_sd_rounded_box((5.0, 0.0, 0.0), b, 0.0)
        assert d == pytest.approx(2.0, abs=TOL)

    def test_outside_corner(self):
        """Outside corner: sd = Euclidean distance."""
        b = (3.0, 2.0, 1.0)
        d = py_sd_rounded_box((5.0, 4.0, 3.0), b, 0.0)
        # q = (2, 2, 2), length = sqrt(12) = 2*sqrt(3) ≈ 3.464
        expected = math.sqrt(12.0)
        assert d == pytest.approx(expected, abs=TOL)

    def test_negative_r_treated_as_zero(self):
        """Negative r is clamped to 0, same as sharp box."""
        b = (3.0, 2.0, 1.0)
        p = (0.0, 0.0, 0.0)
        d_neg = py_sd_rounded_box(p, b, -1.0)
        d_zero = py_sd_rounded_box(p, b, 0.0)
        assert d_neg == pytest.approx(d_zero, abs=TOL)

    def test_interior_off_center(self):
        """Inside sharp box off center: distance = -min distance to any face."""
        b = (3.0, 2.0, 1.0)
        d = py_sd_rounded_box((1.0, 0.3, 0.2), b, 0.0)
        # q = (-2, -1.7, -0.8), max = -0.8, interior = -0.8
        # exterior = 0 (all q negative)
        assert d == pytest.approx(-0.8, abs=TOL)

    def test_sdBox_diagonal_interior(self):
        """Diagonal interior for sharp box: correct with all axes negative."""
        b = (5.0, 5.0, 5.0)
        # Point near the corner: p = (4, 4, 4), q = (-1, -1, -1)
        d = py_sd_rounded_box((4.0, 4.0, 4.0), b, 0.0)
        # interior = max(-1, -1, -1) = -1, clamped to 0... wait
        # min(-1, 0) = -1
        # exterior = length(max(q,0)) = length(0,0,0) = 0
        # result = 0 + (-1) = -1
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_sdBox_far_field_behavior(self):
        """Sharp box far field: sd approximates radial distance."""
        b = (3.0, 2.0, 1.0)
        far = 1000.0
        d = py_sd_rounded_box((far, 0.0, 0.0), b, 0.0)
        assert d == pytest.approx(far - b[0], abs=0.001)


# =============================================================================
# Path 6: r >= min(b) -- maximally rounded, sphere-like limiting shape
# =============================================================================


class TestRadiusMax:
    """When r >= min(b), safe_r = min(b), giving maximum rounding.

    The rounding radius is clamped to the smallest half-dimension, so the
    shape approaches a sphere-like form in the limit where r = min(b).
    """

    def test_cube_r_equals_side(self):
        """Cube b=(s,s,s) with r=s: q = (0,0,0), sd = -s."""
        b = (2.0, 2.0, 2.0)
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, 2.0)
        # q = (0, 0, 0), exterior = 0, interior = 0, sd = 0 - 2 = -2
        assert d == pytest.approx(-2.0, abs=TOL)

    def test_cube_r_greater_than_side(self):
        """Cube b=(s,s,s) with r > s: safe_r clamped to s."""
        b = (2.0, 2.0, 2.0)
        d_gt = py_sd_rounded_box((0.0, 0.0, 0.0), b, 10.0)
        d_eq = py_sd_rounded_box((0.0, 0.0, 0.0), b, 2.0)
        assert d_gt == pytest.approx(d_eq, abs=TOL)

    def test_max_rounding_x_axis_surface(self):
        """With r=min(b), face surface at x = b.x."""
        b = (3.0, 1.0, 2.0)
        r = 1.0  # min(b) = 1
        # Surface on x-axis at x = b.x = 3
        d = py_sd_rounded_box((3.0, 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_max_rounding_y_axis_surface(self):
        """With r=min(b), face surface at y = b.y."""
        b = (3.0, 1.0, 2.0)
        r = 1.0
        # Surface on y-axis at y = b.y = 1
        d = py_sd_rounded_box((0.0, 1.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_max_rounding_z_axis_surface(self):
        """With r=min(b), face surface at z = b.z."""
        b = (3.0, 1.0, 2.0)
        r = 1.0
        # Surface on z-axis at z = b.z = 2
        d = py_sd_rounded_box((0.0, 0.0, 2.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_max_rounding_diagonal_surface(self):
        """Diagonal surface with max rounding: length(q) = r."""
        b = (3.0, 1.0, 2.0)
        r = 1.0
        c = r / math.sqrt(3.0)
        p = (b[0] + c - r, b[1] + c - r, b[2] + c - r)
        d = py_sd_rounded_box(p, b, r)
        # q = (c, c, c), length = sqrt(3*c^2) = sqrt(3 * r^2/3) = r
        # sd = length(q) + 0 - r = r - r = 0
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_max_rounding_center_distance(self):
        """Center distance with max rounding: -min(b) (q shifted by safe_r)."""
        b = (5.0, 3.0, 4.0)
        r = 3.0  # min(b) = 3, safe_r = 3
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        # q = (-2, 0, -1), maxC = 0, interior = 0, exterior = 0
        # sd = 0 + 0 - 3 = -3
        expected = -3.0  # -min(b)
        assert d == pytest.approx(expected, abs=TOL)

    def test_flat_box_max_rounding(self):
        """Flat box (one small axis) with max rounding: limited by min(b)."""
        b = (5.0, 5.0, 0.5)
        r = 2.0  # min(b) = 0.5, so safe_r = 0.5
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        # q = (-4.5, -4.5, 0), maxC = 0, interior = 0
        # exterior = 0, sd = 0 - 0.5 = -0.5
        assert d == pytest.approx(-0.5, abs=TOL)

    def test_flat_box_max_faces(self):
        """With max rounding, z-face surface at z = b.z."""
        b = (5.0, 5.0, 0.5)
        r = 2.0  # min(b) = 0.5 -> safe_r = 0.5
        # z-face surface at z = b.z = 0.5
        d = py_sd_rounded_box((0.0, 0.0, 0.5), b, r)
        # q = (-4.5, -4.5, 0.5), max(q,0) = (0,0,0.5), length = 0.5
        # exterior = 0.5, maxC = 0.5, interior = 0
        # sd = 0.5 + 0 - 0.5 = 0
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)


# =============================================================================
# Path 7: Corner rounding radius -- transition from face to corner
# =============================================================================


class TestCornerRounding:
    """Verify the corner rounding behavior of the signed distance function.

    The rounding radius r creates a smooth transition between faces.
    Corners are rounded with radius r, and the surface shifts outward
    by r along each face normal.
    """

    def test_corner_rounding_surface_x_axis(self):
        """Face surface along x-axis is at b.x."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((3.0, 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_corner_rounding_surface_y_axis(self):
        """Face surface along y-axis is at b.y."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, 2.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_corner_rounding_surface_z_axis(self):
        """Face surface along z-axis is at b.z."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, 0.0, 1.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_corner_rounding_diagonal(self):
        """Corner surface diagonal: length(q) = r."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        # On the diagonal corner, q components are proportional
        c = r / math.sqrt(3.0)
        p = (b[0] + c - r, b[1] + c - r, b[2] + c - r)
        d = py_sd_rounded_box(p, b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_corner_inside_near_corner(self):
        """Inside near x-face: p=(2.5,0,0) -> q.x=0, sd=-0.5."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        p = (2.5, 0.0, 0.0)
        d = py_sd_rounded_box(p, b, r)
        # q = (0, -1.5, -0.5), max(q,0) = (0,0,0), length=0, exterior = 0
        # maxC = 0, interior = 0, sd = 0 + 0 - 0.5 = -0.5
        assert d == pytest.approx(-0.5, abs=TOL)

    def test_corner_outside_near_edge(self):
        """Outside near a rounded edge: q shifted by safe_r changes values."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        p = (3.6, 2.6, 0.0)
        d = py_sd_rounded_box(p, b, r)
        # q = (1.1, 1.1, -0.5), max(q,0) = (1.1, 1.1, 0)
        # exterior = sqrt(2*1.1^2) = sqrt(2.42)
        # sd = sqrt(2.42) + 0 - 0.5
        expected = math.sqrt(2.42) - 0.5
        assert d == pytest.approx(expected, abs=TOL)

    def test_larger_radius_more_rounded(self):
        """Larger r moves the edge surface inward (less interior at same point)."""
        b = (3.0, 2.0, 1.0)
        p = (2.5, 1.5, 0.0)  # Inside the xy-edge of the rounded box
        d_small_r = py_sd_rounded_box(p, b, 0.2)
        d_large_r = py_sd_rounded_box(p, b, 0.8)
        # Both should be negative (inside), larger r = less interior (q shifted more)
        assert d_small_r < 0.0
        assert d_large_r < 0.0
        # Under WGSL formula, larger r shifts q outward, making sd less negative
        assert d_large_r > d_small_r, (
            f"Larger r should give less negative (closer to surface) at edge: "
            f"r=0.2 -> {d_small_r}, r=0.8 -> {d_large_r}"
        )

    def test_face_center_on_surface(self):
        """Distance at face center (b, 0, 0) is 0 (surface at b)."""
        b = (3.0, 2.0, 1.0)
        # At the geometric face center (3,0,0), the box has sd=0.
        d = py_sd_rounded_box((3.0, 0.0, 0.0), b, 0.5)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_corner_gradient_smoothed(self):
        """Corner SDF transitions smoothly (no discontinuity)."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        step = 0.001
        # Trace along diagonal from inside to outside
        prev = py_sd_rounded_box((1.0, 0.5, 0.0), b, r)
        for i in range(1, 500):
            t = 1.0 + i * 0.01
            p = (min(t, b[0] + r + 1.0), min(t * 0.7, b[1] + r + 1.0), 0.0)
            curr = py_sd_rounded_box(p, b, r)
            diff = abs(curr - prev)
            assert diff <= 0.02, (
                f"Near-discontinuity at t={t}: diff={diff}"
            )
            prev = curr


# =============================================================================
# Path 8: Inside the box -- negative signed distance
# =============================================================================


class TestInside:
    """Inside the rounded box: signed distance should be negative.

    With the WGSL formula (q = abs(p) - b + safe_r), for a point fully inside
    (all q <= 0), the exterior term is 0 (since max(q,0) = (0,0,0)) and the
    interior term is max(q) (the least-negative component, shifted by safe_r).
    The result is maxComponent(q) - safe_r, which is negative.
    """

    def test_center_asymmetric_box(self):
        """Center of asymmetric box with WGSL formula."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        # q = (-2.5, -1.5, -0.5), maxC = -0.5, interior = -0.5
        # exterior = 0, sd = 0 + (-0.5) - 0.5 = -1.0
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_center_cube_with_rounding(self):
        """Center of rounded cube with WGSL formula."""
        b = (2.0, 2.0, 2.0)
        r = 0.3
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        # q = (-1.7, -1.7, -1.7), maxC = -1.7, exterior = 0
        # sd = -1.7 - 0.3 = -2.0
        assert d == pytest.approx(-2.0, abs=TOL)

    def test_inside_near_x_face(self):
        """Inside near x-face with WGSL formula: sd = -0.2."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((2.8, 0.0, 0.0), b, r)
        # q = (0.3, -1.5, -0.5), max(q,0) = (0.3, 0, 0), exterior = 0.3
        # maxC = 0.3, interior = 0, sd = 0.3 + 0 - 0.5 = -0.2
        assert d == pytest.approx(-0.2, abs=TOL)

    def test_inside_near_y_face(self):
        """Inside near y-face with WGSL formula: sd = -0.2."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, 1.8, 0.0), b, r)
        # q = (-2.5, 0.3, -0.5), max(q,0) = (0, 0.3, 0), exterior = 0.3
        # maxC = 0.3, interior = 0, sd = 0.3 - 0.5 = -0.2
        assert d == pytest.approx(-0.2, abs=TOL)

    def test_inside_near_z_face(self):
        """Inside near z-face with WGSL formula: sd = -0.2."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, 0.0, 0.8), b, r)
        # q = (-2.5, -1.5, 0.3), max(q,0) = (0, 0, 0.3), exterior = 0.3
        # maxC = 0.3, interior = 0, sd = 0.3 - 0.5 = -0.2
        assert d == pytest.approx(-0.2, abs=TOL)

    def test_inside_all_negative(self):
        """All q negative with WGSL formula."""
        b = (5.0, 4.0, 3.0)
        r = 0.5
        d = py_sd_rounded_box((2.0, 1.0, 0.5), b, r)
        # q = (-2.5, -2.5, -2.0), maxC = -2.0, interior = -2.0
        # exterior = 0, sd = -2.0 - 0.5 = -2.5
        assert d == pytest.approx(-2.5, abs=TOL)

    def test_inside_x_close_to_y_face(self):
        """Interior term comes from the axis closest to its face."""
        b = (3.0, 3.0, 3.0)
        r = 1.0
        d = py_sd_rounded_box((2.9, 0.0, 0.0), b, r)
        # q = (0.9, -2, -2), max(q,0) = (0.9, 0, 0), exterior = 0.9
        # maxC = 0.9, interior = 0, sd = 0.9 - 1.0 = -0.1
        assert d == pytest.approx(-0.1, abs=TOL)

    def test_inside_negative_r_zero(self):
        """With negative r (clamped to 0): interior dominates."""
        b = (3.0, 2.0, 1.0)
        d = py_sd_rounded_box((1.0, 0.5, 0.3), b, -1.0)
        # q = (-2, -1.5, -0.7), maxC = -0.7, interior = -0.7
        # exterior = 0 (r=0), result = -0.7
        assert d == pytest.approx(-0.7, abs=TOL)

    def test_inside_depth_vs_radius(self):
        """Inside distance increases (more negative) with depth."""
        b = (5.0, 5.0, 5.0)
        r = 1.0
        near = py_sd_rounded_box((4.0, 0.0, 0.0), b, r)  # near face
        deep = py_sd_rounded_box((3.0, 0.0, 0.0), b, r)  # deeper inside
        center = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)  # center
        assert deep < near, (
            f"Deeper point should be more negative: near={near}, deep={deep}"
        )
        assert center < deep, (
            f"Center should be most negative: center={center}, deep={deep}"
        )


# =============================================================================
# Path 9: On the surface -- zero signed distance
# =============================================================================


class TestSurface:
    """Points on the rounded box surface should yield signed distance ~0.

    With the WGSL formula (q = abs(p) - b + safe_r), the effective surface of the
    rounded box is:

    - Along each axis: at b_i (the original box face)
    - At each corner diagonal: where length(q) = safe_r
    - At each edge (rim): where two q components are positive, satisfying
      sqrt(q1^2 + q2^2) = safe_r
    """

    def test_on_x_face_positive(self):
        """On x-face surface: p = (b.x, 0, 0) -> 0."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((b[0], 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_x_face_negative(self):
        """On x-face surface at negative x: p = (-b.x, 0, 0) -> 0."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((-b[0], 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_y_face_positive(self):
        """On y-face surface: p = (0, b.y, 0) -> 0."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, b[1], 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_y_face_negative(self):
        """On y-face surface at negative y: p = (0, -b.y, 0) -> 0."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, -b[1], 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_z_face_positive(self):
        """On z-face surface: p = (0, 0, b.z) -> 0."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, 0.0, b[2]), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_z_face_negative(self):
        """On z-face surface at negative z: p = (0, 0, -b.z) -> 0."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, 0.0, -b[2]), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_edge_xy(self):
        """On the xy-edge (rim): sqrt(q.x^2 + q.y^2) = r, q.z <= 0."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        c = r / math.sqrt(2.0)  # so that sqrt(c^2 + c^2) = r
        p = (b[0] + c - r, b[1] + c - r, 0.0)
        d = py_sd_rounded_box(p, b, r)
        # q = (c, c, -1+r), max(q,0) = (c, c, 0), length = sqrt(2*c^2) = r
        # sd = r + 0 - r = 0
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_edge_xz(self):
        """On the xz-edge: sqrt(q.x^2 + q.z^2) = r, q.y <= 0."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        c = r / math.sqrt(2.0)
        p = (b[0] + c - r, 0.0, b[2] + c - r)
        d = py_sd_rounded_box(p, b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_edge_yz(self):
        """On the yz-edge: sqrt(q.y^2 + q.z^2) = r, q.x <= 0."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        c = r / math.sqrt(2.0)
        p = (0.0, b[1] + c - r, b[2] + c - r)
        d = py_sd_rounded_box(p, b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_corner(self):
        """On the corner surface: length(q) = r, all q > 0."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        c = r / math.sqrt(3.0)  # so length(c,c,c) = r
        p = (b[0] + c - r, b[1] + c - r, b[2] + c - r)
        d = py_sd_rounded_box(p, b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_corner_negative_octant(self):
        """On the corner surface in the negative octant."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        c = r / math.sqrt(3.0)
        p = (-(b[0] + c - r), -(b[1] + c - r), -(b[2] + c - r))
        d = py_sd_rounded_box(p, b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_on_surface_with_zero_r(self):
        """With r=0: surface at b (sharp box)."""
        b = (3.0, 2.0, 1.0)
        assert py_sd_rounded_box((3.0, 0.0, 0.0), b, 0.0) == pytest.approx(
            0.0, abs=TOL_SURFACE
        )
        assert py_sd_rounded_box((0.0, 2.0, 0.0), b, 0.0) == pytest.approx(
            0.0, abs=TOL_SURFACE
        )
        assert py_sd_rounded_box((0.0, 0.0, 1.0), b, 0.0) == pytest.approx(
            0.0, abs=TOL_SURFACE
        )

    @pytest.mark.parametrize(
        "b, r",
        [
            ((2.0, 1.0, 0.5), 0.3),
            ((4.0, 3.0, 2.0), 1.0),
            ((1.0, 1.0, 1.0), 0.0),
            ((5.0, 5.0, 5.0), 2.5),
        ],
    )
    def test_surface_all_face_axes(self, b, r):
        """All six face surfaces should be on the zero isosurface."""
        axes = [
            (b[0], 0.0, 0.0),
            (-b[0], 0.0, 0.0),
            (0.0, b[1], 0.0),
            (0.0, -b[1], 0.0),
            (0.0, 0.0, b[2]),
            (0.0, 0.0, -b[2]),
        ]
        for p in axes:
            d = py_sd_rounded_box(p, b, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point {p} with b={b}, r={r} should have d=0, got {d}"
            )

    @pytest.mark.parametrize(
        "b, r",
        [
            ((3.0, 2.0, 1.0), 0.5),
            ((2.0, 2.0, 2.0), 0.8),
        ],
    )
    def test_surface_diagonal(self, b, r):
        """The rounded box surface includes the diagonal corner region."""
        c = r / math.sqrt(3.0)
        p = (b[0] + c - r, b[1] + c - r, b[2] + c - r)
        d = py_sd_rounded_box(p, b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Corner surface {p} with b={b}, r={r} should have d=0, got {d}"
        )


# =============================================================================
# Path 10: Outside the box -- positive signed distance
# =============================================================================


class TestOutside:
    """Outside the rounded box: signed distance should be positive."""

    def test_outside_x_axis(self):
        """Outside on x-axis: sd = (x - b.x + safe_r) - safe_r = x - b.x."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((5.0, 0.0, 0.0), b, r)
        # q = (2.5, -1.5, -0.5), max(q,0) = (2.5, 0, 0), exterior = 2.5
        # maxC = 2.5, interior = 0, sd = 2.5 - 0.5 = 2.0
        assert d == pytest.approx(2.0, abs=TOL)

    def test_outside_y_axis(self):
        """Outside on y-axis: sd = (y - b.y + safe_r) - safe_r = y - b.y."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, 4.0, 0.0), b, r)
        # q = (-2.5, 2.5, -0.5), max(q,0) = (0, 2.5, 0), exterior = 2.5
        # sd = 2.5 - 0.5 = 2.0
        assert d == pytest.approx(2.0, abs=TOL)

    def test_outside_z_axis(self):
        """Outside on z-axis: sd = (z - b.z + safe_r) - safe_r = z - b.z."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, 0.0, 3.0), b, r)
        # q = (-2.5, -1.5, 2.5), max(q,0) = (0, 0, 2.5), exterior = 2.5
        # sd = 2.5 - 0.5 = 2.0
        assert d == pytest.approx(2.0, abs=TOL)

    def test_outside_corner_asymmetric(self):
        """Outside corner with all axes positive: q shifted by safe_r."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        p = (6.0, 5.0, 4.0)  # q = (3.5, 3.5, 3.5)
        d = py_sd_rounded_box(p, b, r)
        # q = (3.5, 3.5, 3.5), length = sqrt(3*3.5^2) = 3.5*sqrt(3)
        # sd = 3.5*sqrt(3) - 0.5
        expected = math.sqrt(3 * 3.5 * 3.5) - 0.5
        assert d == pytest.approx(expected, abs=TOL)

    def test_outside_diagonal_with_one_inside(self):
        """Outside with q shifted by safe_r."""
        b = (3.0, 2.0, 1.0)
        r = 0.3
        p = (5.0, 3.5, 0.5)  # q = (2.3, 1.8, -0.2)
        d = py_sd_rounded_box(p, b, r)
        # max(q,0) = (2.3, 1.8, 0), length = sqrt(2.3^2 + 1.8^2) = sqrt(8.53)
        # exterior = sqrt(8.53), maxC = 2.3, interior = 0
        # sd = sqrt(8.53) - 0.3
        expected = math.sqrt(2.3 * 2.3 + 1.8 * 1.8) - 0.3
        assert d == pytest.approx(expected, abs=TOL)

    def test_outside_far_field(self):
        """Far from box: sd approximates x - b.x."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        far = 1000.0
        d = py_sd_rounded_box((far, 0.0, 0.0), b, r)
        expected = far - b[0]  # x - b.x (safe_r cancels out)
        assert d == pytest.approx(expected, abs=0.001)

    def test_outside_edge_region(self):
        """Outside near an edge: q shifted by safe_r."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        p = (4.0, 3.0, 0.0)  # q = (1.5, 1.5, -0.5)
        d = py_sd_rounded_box(p, b, r)
        expected = math.sqrt(2 * 1.5 * 1.5) - 0.5  # sqrt(4.5) - 0.5
        assert d == pytest.approx(expected, abs=TOL)

    def test_outside_monotonic_x(self):
        """Distance monotonically increases as point moves outward on x-axis."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        prev = py_sd_rounded_box((b[0], 0.0, 0.0), b, r)
        for x in [b[0] + r + i * 0.5 for i in range(1, 10)]:
            curr = py_sd_rounded_box((x, 0.0, 0.0), b, r)
            assert curr >= prev - 1e-15, (
                f"Distance should increase: at x={x}, curr={curr} < prev={prev}"
            )
            prev = curr

    def test_outside_above_surface_positive(self):
        """Immediately above the surface yields small positive distance."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        eps = 0.001
        d = py_sd_rounded_box((b[0] + eps, 0.0, 0.0), b, r)
        assert d == pytest.approx(eps, abs=TOL_SURFACE)

    def test_outside_below_surface_negative(self):
        """Immediately below the surface yields small negative distance."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        eps = 0.001
        d = py_sd_rounded_box((b[0] - eps, 0.0, 0.0), b, r)
        assert d == pytest.approx(-eps, abs=TOL_SURFACE)


# =============================================================================
# Path 11: Symmetry -- sign independence via abs(p)
# =============================================================================


class TestSymmetry:
    """The rounded box SDF is symmetric under sign flips of any axis.

    The abs(p) operation in q = abs(p) - b ensures that sign flips
    in any coordinate produce the same signed distance.
    """

    def test_x_symmetry(self):
        """Points (x,y,z) and (-x,y,z) should have equal distances."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        test_points = [
            (1.0, 0.5, 0.3),
            (2.0, 1.0, 0.0),
            (4.0, 0.5, 0.0),
            (-1.0, 2.0, 0.5),
        ]
        for x, y, z in test_points:
            d_pos = py_sd_rounded_box((x, y, z), b, r)
            d_neg = py_sd_rounded_box((-x, y, z), b, r)
            assert d_pos == pytest.approx(d_neg, abs=TOL), (
                f"X-symmetry: ({x},{y},{z}) -> {d_pos}, "
                f"({-x},{y},{z}) -> {d_neg}"
            )

    def test_y_symmetry(self):
        """Points (x,y,z) and (x,-y,z) should have equal distances."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        test_points = [
            (0.5, 1.0, 0.3),
            (2.0, 2.0, 0.0),
            (0.0, 3.0, 0.5),
        ]
        for x, y, z in test_points:
            d_pos = py_sd_rounded_box((x, y, z), b, r)
            d_neg = py_sd_rounded_box((x, -y, z), b, r)
            assert d_pos == pytest.approx(d_neg, abs=TOL), (
                f"Y-symmetry: ({x},{y},{z}) -> {d_pos}, "
                f"({x},{-y},{z}) -> {d_neg}"
            )

    def test_z_symmetry(self):
        """Points (x,y,z) and (x,y,-z) should have equal distances."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        test_points = [
            (0.5, 0.3, 1.0),
            (2.0, 0.0, 2.0),
            (0.0, 1.0, 0.5),
        ]
        for x, y, z in test_points:
            d_pos = py_sd_rounded_box((x, y, z), b, r)
            d_neg = py_sd_rounded_box((x, y, -z), b, r)
            assert d_pos == pytest.approx(d_neg, abs=TOL), (
                f"Z-symmetry: ({x},{y},{z}) -> {d_pos}, "
                f"({x},{y},{-z}) -> {d_neg}"
            )

    def test_full_octahedral_symmetry(self):
        """All 8 sign-flip combinations of a point should yield the same distance."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        base = (1.5, 0.8, 0.4)
        d_base = py_sd_rounded_box(base, b, r)
        signs = [
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (-1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
        ]
        for sx, sy, sz in signs:
            p = (sx * base[0], sy * base[1], sz * base[2])
            d = py_sd_rounded_box(p, b, r)
            assert d == pytest.approx(d_base, abs=TOL), (
                f"Symmetry broken for sign ({sx},{sy},{sz}): "
                f"{p} -> {d}, base -> {d_base}"
            )

    def test_abs_ensures_xz_plane_symmetry(self):
        """abs(p.y) ensures xz-plane symmetry."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        p_above = (1.0, 1.5, 0.5)
        p_below = (1.0, -1.5, 0.5)
        d_above = py_sd_rounded_box(p_above, b, r)
        d_below = py_sd_rounded_box(p_below, b, r)
        assert d_above == pytest.approx(d_below, abs=TOL)

    def test_deterministic(self):
        """Same inputs always produce same output."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        p = (1.2, 0.7, 0.3)
        base = py_sd_rounded_box(p, b, r)
        for _ in range(20):
            assert py_sd_rounded_box(p, b, r) == pytest.approx(base, abs=TOL)

    def test_box_symmetry_all_quadrants(self):
        """All combinations of x,z signs at same magnitude yield same distance."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        p_ref = (0.5, 0.4, 0.3)
        d_ref = py_sd_rounded_box(p_ref, b, r)
        xz_combos = [
            (0.5, 0.3),
            (-0.5, 0.3),
            (0.5, -0.3),
            (-0.5, -0.3),
        ]
        for x, z in xz_combos:
            p = (x, 0.4, z)
            d = py_sd_rounded_box(p, b, r)
            assert d == pytest.approx(d_ref, abs=TOL), (
                f"Symmetry broken for (x={x}, z={z}): d={d}, ref={d_ref}"
            )


# =============================================================================
# Path 12: Sign convention -- negative inside, zero on surface, positive outside
# =============================================================================


class TestSignConvention:
    """The SDF sign convention: negative inside, zero on surface, positive outside."""

    def test_inside_negative(self):
        """Points inside the rounded box should have negative signed distance."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        inside_points = [
            (0.0, 0.0, 0.0),         # Center
            (1.0, 0.0, 0.0),         # Inside on x-axis
            (0.0, 0.5, 0.0),         # Inside on y-axis
            (0.0, 0.0, 0.3),         # Inside on z-axis
            (2.0, 1.0, 0.5),         # Interior diagonal
            (-1.0, -0.5, -0.3),      # Negative octant interior
            (0.0, 1.5, 0.0),         # Near y-face
            (1.0, 0.5, 0.3),         # General interior
        ]
        for p in inside_points:
            d = py_sd_rounded_box(p, b, r)
            assert d < 0.0, (
                f"Inside point {p} should have negative SDF, got {d}"
            )

    def test_outside_positive(self):
        """Points outside the rounded box should have positive signed distance."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        outside_points = [
            (5.0, 0.0, 0.0),          # Far on +x
            (0.0, 4.0, 0.0),          # Far on +y
            (0.0, 0.0, 3.0),          # Far on +z
            (-5.0, 0.0, 0.0),         # Far on -x
            (4.0, 3.0, 2.0),          # Diagonal outside
            (10.0, 0.0, 0.0),         # Very far on x
            (b[0] + 1.0, 0.0, 0.0),   # Just outside x-face
        ]
        for p in outside_points:
            d = py_sd_rounded_box(p, b, r)
            assert d > 0.0, (
                f"Outside point {p} should have positive SDF, got {d}"
            )

    def test_on_surface_zero(self):
        """Points on the rounded box surface should have near-zero distance."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        surface_points = [
            (b[0], 0.0, 0.0),
            (-b[0], 0.0, 0.0),
            (0.0, b[1], 0.0),
            (0.0, -b[1], 0.0),
            (0.0, 0.0, b[2]),
            (0.0, 0.0, -b[2]),
        ]
        c3 = r / math.sqrt(3.0)
        surface_points.append((b[0] + c3 - r, b[1] + c3 - r, b[2] + c3 - r))
        for p in surface_points:
            d = py_sd_rounded_box(p, b, r)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point {p} should have near-zero SDF, got {d}"
            )

    def test_sign_transition_at_surface(self):
        """Sign should transition from negative to positive across the surface."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        surface_x = b[0]
        step = 1e-6
        inside = py_sd_rounded_box((surface_x - step, 0.0, 0.0), b, r)
        on_surface = py_sd_rounded_box((surface_x, 0.0, 0.0), b, r)
        outside = py_sd_rounded_box((surface_x + step, 0.0, 0.0), b, r)
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
        """Distance ordering: inside < surface < outside.

        Uses a cube (symmetric b) so that the x-axis is the dominant
        max-component axis, ensuring strict monotonicity.
        """
        b = (2.0, 2.0, 2.0)
        r = 0.5
        points = [0.0, 1.0, 1.5, 2.0, 2.5, 5.0]
        distances = [py_sd_rounded_box((x, 0.0, 0.0), b, r) for x in points]
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i + 1], (
                f"Distance should strictly increase along +x: "
                f"x={points[i]} -> {distances[i]}, "
                f"x={points[i+1]} -> {distances[i+1]}"
            )


# =============================================================================
# Path 13: Continuity and monotonicity
# =============================================================================


class TestContinuity:
    """The rounded box SDF should be continuous everywhere."""

    def test_continuity_along_x(self):
        """SDF should be continuous along the x-axis."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        step = 0.01
        prev = py_sd_rounded_box((-5.0, 0.0, 0.0), b, r)
        for i in range(1, 800):
            x = -5.0 + i * step
            curr = py_sd_rounded_box((x, 0.0, 0.0), b, r)
            diff = abs(curr - prev)
            assert diff <= step * 1.1, (
                f"Near-discontinuity at x={x}: diff={diff} > step={step}"
            )
            prev = curr

    def test_continuity_along_y(self):
        """SDF should be continuous along the y-axis."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        step = 0.01
        prev = py_sd_rounded_box((0.0, -4.0, 0.0), b, r)
        for i in range(1, 600):
            y = -4.0 + i * step
            curr = py_sd_rounded_box((0.0, y, 0.0), b, r)
            diff = abs(curr - prev)
            assert diff <= step * 1.1, (
                f"Near-discontinuity at y={y}: diff={diff} > step={step}"
            )
            prev = curr

    def test_continuity_along_z(self):
        """SDF should be continuous along the z-axis."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        step = 0.01
        prev = py_sd_rounded_box((0.0, 0.0, -3.0), b, r)
        for i in range(1, 400):
            z = -3.0 + i * step
            curr = py_sd_rounded_box((0.0, 0.0, z), b, r)
            diff = abs(curr - prev)
            assert diff <= step * 1.1, (
                f"Near-discontinuity at z={z}: diff={diff} > step={step}"
            )
            prev = curr

    def test_continuity_across_x_face_surface(self):
        """SDF should be continuous across the x-face surface."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        step = 1e-5
        for offset in [i * step for i in range(-20, 21)]:
            x = b[0] + offset
            d = py_sd_rounded_box((x, 0.0, 0.0), b, r)
            assert d == pytest.approx(offset, abs=1e-12), (
                f"Near x-face surface at x={x}: expected {offset}, got {d}"
            )

    def test_monotonicity_inside_to_outside(self):
        """SDF should be monotonically increasing from center outward."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        prev = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        for t in [i * 0.1 for i in range(0, 60)]:
            x = t
            curr = py_sd_rounded_box((x, 0.0, 0.0), b, r)
            assert curr >= prev - 1e-15, (
                f"Non-monotonic at x={x}: curr={curr}, prev={prev}"
            )
            prev = curr


# =============================================================================
# Path 14: Parameter variation and edge cases
# =============================================================================


class TestParameterVariation:
    """Test with varying box dimensions and rounding radii."""

    def test_tall_thin_box(self):
        """Tall thin box: b = (1, 10, 1), r = 0.3."""
        b = (1.0, 10.0, 1.0)
        r = 0.3
        # Center
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        # q = (-0.7, -9.7, -0.7), maxC = -0.7, exterior = 0
        # sd = -0.7 - 0.3 = -1.0
        assert d == pytest.approx(-1.0, abs=TOL)

        # On x-face surface: at x = 1.0 (b.x)
        d = py_sd_rounded_box((1.0, 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_wide_flat_box(self):
        """Wide flat box: b = (10, 0.5, 5), r = 0.2."""
        b = (10.0, 0.5, 5.0)
        r = 0.2  # safe_r = min(0.2, 0.5) = 0.2
        # Center: q = (-9.8, -0.3, -4.8), maxC = -0.3, exterior = 0
        # sd = -0.3 - 0.2 = -0.5
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        assert d == pytest.approx(-0.5, abs=TOL)

        # y-face surface: at y = b.y = 0.5
        d = py_sd_rounded_box((0.0, 0.5, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_large_r_clamped(self):
        """Large r clamped to min(b): behaves same as r = min(b)."""
        b = (3.0, 2.0, 1.0)
        r_clamped = py_sd_rounded_box((0.0, 0.0, 0.0), b, 10.0)
        r_max = py_sd_rounded_box((0.0, 0.0, 0.0), b, 1.0)
        assert r_clamped == pytest.approx(r_max, abs=TOL)

    def test_negative_r_same_as_zero(self):
        """Negative r clamped to 0: same as sharp box."""
        b = (3.0, 2.0, 1.0)
        p = (1.0, 0.5, 0.3)
        d_neg = py_sd_rounded_box(p, b, -2.0)
        d_zero = py_sd_rounded_box(p, b, 0.0)
        assert d_neg == pytest.approx(d_zero, abs=TOL)

    def test_zero_sized_box(self):
        """Box with b = (0, 0, 0): behaves as point SDF with no rounding."""
        b = (0.0, 0.0, 0.0)
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, 0.0)
        assert d == pytest.approx(0.0, abs=TOL)

        d_far = py_sd_rounded_box((3.0, 4.0, 0.0), b, 0.0)
        assert d_far == pytest.approx(5.0, abs=TOL)

    def test_very_small_r(self):
        """Very small rounding: r = 1e-6."""
        b = (3.0, 2.0, 1.0)
        r = 1e-6
        d_center = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        d_sharp = py_sd_rounded_box((0.0, 0.0, 0.0), b, 0.0)
        # With WGSL formula, center sd = -min(b) = -1.0 for any r
        assert d_center == pytest.approx(-1.0, abs=TOL)
        assert d_sharp == pytest.approx(-1.0, abs=TOL)

    def test_large_r_face_surface(self):
        """With r near min(b), face surfaces are at b."""
        b = (5.0, 3.0, 4.0)
        r = 3.0  # min(b) = 3 -> safe_r = 3
        # x-face at x = b.x = 5
        d = py_sd_rounded_box((5.0, 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)


# =============================================================================
# Path 15: Combined formula -- full exterior + interior integration
# =============================================================================


class TestCombinedFormula:
    """Verify that exterior + interior - safe_r correctly reconstructs the full SDF.

    The full SDF always satisfies (WGSL formula):
        sdRoundedBox = exterior_term + interior_term - safe_r
    where:
        q = abs(p) - b + safe_r
        exterior_term = length(max(q, 0))
        interior_term = min(maxComponent(q), 0)
    """

    def test_sum_equals_full(self):
        """exterior_term + interior_term - safe_r always equals full sdRoundedBox."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        test_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.5, 0.3),
            (3.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 1.0),
            (5.0, 0.0, 0.0),
            (4.0, 3.0, 2.0),
            (3.0, 2.0, 1.0),
            (-1.0, -0.5, -0.3),
            (-4.0, 0.0, 0.0),
        ]
        for p in test_points:
            safe_r_val = py_safe_r(b, r)
            exterior = py_exterior(p, b, r)
            interior = py_interior(p, b, r)  # now passes r
            full = py_sd_rounded_box(p, b, r)
            assert full == pytest.approx(exterior + interior - safe_r_val, abs=TOL), (
                f"p={p}: exterior({exterior}) + interior({interior}) - safe_r({safe_r_val})"
                f" = {exterior + interior - safe_r_val} != full({full})"
            )

    def test_exterior_only_when_outside(self):
        """When outside box (any q > 0): interior = 0, sd = exterior - safe_r."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        safe_r_val = py_safe_r(b, r)
        outside_points = [
            (5.0, 0.0, 0.0),
            (0.0, 4.0, 0.0),
            (0.0, 0.0, 3.0),
            (5.0, 4.0, 3.0),
            (3.0, 0.0, 0.0),  # On surface: sd = 0
        ]
        for p in outside_points:
            exterior = py_exterior(p, b, r)
            interior = py_interior(p, b, r)
            full = py_sd_rounded_box(p, b, r)
            assert interior == pytest.approx(0.0, abs=TOL), (
                f"p={p}: interior should be 0 outside, got {interior}"
            )
            assert full == pytest.approx(exterior - safe_r_val, abs=TOL), (
                f"p={p}: full({full}) should equal exterior({exterior}) - safe_r({safe_r_val})"
                f" when outside, got {full} vs {exterior - safe_r_val}"
            )

    def test_interior_exterior_inside(self):
        """When fully inside (all q < 0): exterior = 0, sd = interior - safe_r."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        safe_r_val = py_safe_r(b, r)
        inside_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.5, 0.3),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.5),
            (-1.0, -0.5, 0.0),
        ]
        for p in inside_points:
            exterior = py_exterior(p, b, r)
            interior = py_interior(p, b, r)
            full = py_sd_rounded_box(p, b, r)
            assert exterior == pytest.approx(0.0, abs=TOL), (
                f"p={p}: exterior should be 0 when fully inside, "
                f"got {exterior}"
            )
            assert full == pytest.approx(exterior + interior - safe_r_val, abs=TOL), (
                f"p={p}: full({full}) != exterior({exterior}) + interior({interior}) - safe_r"
            )

    def test_interior_exterior_mixed(self):
        """When some q mixed (some positive, some negative): both terms active."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        safe_r_val = py_safe_r(b, r)
        p = (4.0, 0.5, 0.0)
        exterior = py_exterior(p, b, r)
        interior = py_interior(p, b, r)
        full = py_sd_rounded_box(p, b, r)
        assert full == pytest.approx(exterior + interior - safe_r_val, abs=TOL), (
            f"Mixed point {p}: full({full}) != exterior({exterior}) "
            f"+ interior({interior}) - safe_r({safe_r_val})"
        )

    def test_on_surface_exterior_r(self):
        """On surface: exterior = safe_r (face) or exterior = r (corner)."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        # On x-face surface: q = (0.5, -1.5, -0.5), max(q,0) = (0.5, 0, 0)
        # length = 0.5, exterior = 0.5 (safe_r subtracted at sd level)
        exterior = py_exterior((b[0], 0.0, 0.0), b, r)
        assert exterior == pytest.approx(r, abs=TOL)

    def test_different_radii(self):
        """Sum decomposition holds for various radii."""
        b = (3.0, 2.0, 1.0)
        p = (1.5, 0.5, 0.3)
        for r in [0.0, 0.1, 0.5, 1.0, 5.0]:
            safe_r_val = py_safe_r(b, r)
            exterior = py_exterior(p, b, r)
            interior = py_interior(p, b, r)
            full = py_sd_rounded_box(p, b, r)
            assert full == pytest.approx(exterior + interior - safe_r_val, abs=TOL), (
                f"r={r}: full({full}) != exterior({exterior}) + interior({interior}) - safe_r"
            )

    def test_different_box_sizes(self):
        """Sum decomposition holds for various box sizes."""
        boxes = [
            (1.0, 1.0, 1.0),
            (5.0, 3.0, 2.0),
            (10.0, 0.5, 0.5),
            (2.0, 4.0, 1.0),
        ]
        r = 0.3
        p = (0.5, 0.2, 0.1)
        for b in boxes:
            safe_r_val = py_safe_r(b, r)
            exterior = py_exterior(p, b, r)
            interior = py_interior(p, b, r)
            full = py_sd_rounded_box(p, b, r)
            assert full == pytest.approx(exterior + interior - safe_r_val, abs=TOL), (
                f"b={b}: full({full}) != exterior({exterior}) + interior({interior}) - safe_r"
            )


# =============================================================================
# Path 16: Cross-validated edge cases at various positions
# =============================================================================


class TestCrossValidation:
    """Cross-validate the SDF at specific known-position computations."""

    def test_known_computation_center(self):
        """Known value at center for b=(2,1,1), r=0.5: sd = -1.0."""
        b = (2.0, 1.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
        # q = (-1.5, -0.5, -0.5), maxC = -0.5, interior = -0.5
        # exterior = 0, sd = -0.5 - 0.5 = -1.0
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_known_computation_on_face_x(self):
        """Known value at x-face surface for b=(2,1,1), r=0.5: sd = 0 at x=2."""
        b = (2.0, 1.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((2.0, 0.0, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_known_computation_outside_x(self):
        """Known value outside face for b=(2,1,1), r=0.5: sd = 1.0 at x=3."""
        b = (2.0, 1.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((3.0, 0.0, 0.0), b, r)
        # q = (1.5, -0.5, -0.5), max = (1.5, 0, 0), exterior = 1.5
        # sd = 1.5 - 0.5 = 1.0
        assert d == pytest.approx(1.0, abs=TOL)

    def test_known_computation_at_corner(self):
        """At the geometric corner (b.x,b.y,b.z) for b=(2,1,1), r=0.5: sd = 0.366."""
        b = (2.0, 1.0, 1.0)
        r = 0.5
        d = py_sd_rounded_box((2.0, 1.0, 1.0), b, r)
        # q = (0.5, 0.5, 0.5), length = sqrt(0.75) = 0.866
        # exterior = 0.866, sd = 0.866 - 0.5 = 0.366
        expected = math.sqrt(3 * 0.5 * 0.5) - 0.5
        assert d == pytest.approx(expected, abs=TOL)

    def test_known_computation_corner_surface(self):
        """On the spherical corner for b=(2,1,1), r=0.5."""
        b = (2.0, 1.0, 1.0)
        r = 0.5
        c = r / math.sqrt(3.0)
        p = (b[0] + c - r, b[1] + c - r, b[2] + c - r)
        d = py_sd_rounded_box(p, b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_known_computation_edge_xy(self):
        """On the xy-edge surface for b=(2,1,1), r=0.5."""
        b = (2.0, 1.0, 1.0)
        r = 0.5
        c = r / math.sqrt(2.0)
        d = py_sd_rounded_box((b[0] + c - r, b[1] + c - r, 0.0), b, r)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE)

    def test_cube_various_radii(self):
        """Cube b=(3,3,3) with various radii, center distance = -3 for all r."""
        b = (3.0, 3.0, 3.0)
        for r in [0.0, 0.5, 1.0, 2.0, 3.0]:
            d = py_sd_rounded_box((0.0, 0.0, 0.0), b, r)
            # With WGSL formula, center sd = -b.x = -3 for all r
            assert d == pytest.approx(-3.0, abs=TOL), (
                f"Cube b={b}, r={r}: expected -3.0, got {d}"
            )

    def test_far_field_asymptotic(self):
        """Far field: sdRoundedBox approximates x - b.x."""
        b = (3.0, 2.0, 1.0)
        r = 0.5
        for scale in [100.0, 1000.0, 10000.0]:
            p = (scale, 0.0, 0.0)
            d = py_sd_rounded_box(p, b, r)
            # Far along x: qx = scale - b.x + safe_r >> 0, d ~ (scale - b.x + safe_r) - safe_r
            expected = scale - b[0]
            assert d == pytest.approx(expected, abs=0.001), (
                f"Far field at x={scale}: expected ~{expected}, got {d}"
            )
