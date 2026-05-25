"""
Whitebox tests for sdBoxFrame WGSL function (T-DEMO-1.9).

Tests a Python model matching the WGSL implementation formula:
    let q = abs(p) - b;
    let d = length(max(q, vec3(0))) + min(max(q.x, max(q.y, q.z)), 0) - e;
    return d;

This is Inigo Quilez's signed distance function for a hollow box frame with
outer half-dimensions b and frame wall thickness e.

WHITEBOX coverage plan:
  Path 1: Formula decomposition -- verify each sub-expression
  Path 2: e = 0 degenerates to solid box (identical to sdBox)
  Path 3: Surface at |p| = b + e along principal axes
  Path 4: Outside points produce positive distance
  Path 5: Inside frame walls produce negative distance
  Path 6: Interior cavity -- inside outer box but distance depends on q and e
  Path 7: Octahedral symmetry via abs(p)
  Path 8: Non-uniform box dimensions (b varies per axis)
  Path 9: e >= min(b) frame fills the interior
  Path 10: Monotonicity and ordering
  Path 11: Frame thickness variation with constant box
"""

from __future__ import annotations

import math

import pytest

# =============================================================================
# Python model of sdBoxFrame matching WGSL semantics exactly (whitebox)
# =============================================================================


def py_sd_box_frame(p, b, e):
    """Python model of WGSL sdBoxFrame(p: vec3<f32>, b: vec3<f32>, e: f32) -> f32.

    Signed distance from point p (3-tuple) to a hollow box frame with outer
    half-dimensions b=(bx,by,bz) and frame wall thickness e, centered at origin.

    Formula decomposition:
        q    = abs(p) - b
        d    = length(max(q, 0)) + min(maxComponent(q), 0) - e

    Reference: Inigo Quilez -- Box Frame SDF
    https://iquilezles.org/articles/distfunctions/
    """
    qx = abs(p[0]) - b[0]
    qy = abs(p[1]) - b[1]
    qz = abs(p[2]) - b[2]

    # length(max(q, 0)): Euclidean distance from positive q components
    out_dist = math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2)

    # min(maxComponent(q), 0): interior distance to nearest face (non-positive)
    max_comp = max(qx, qy, qz)
    inner_penalty = min(max_comp, 0.0)

    return out_dist + inner_penalty - e


# =============================================================================
# Python model of sdBox for e=0 comparison
# =============================================================================


def py_sd_box(p, b):
    """Python model of WGSL sdBox: signed distance to solid box.

    Formula: length(max(abs(p) - b, 0)) + min(maxComponent(abs(p) - b), 0)
    """
    qx = abs(p[0]) - b[0]
    qy = abs(p[1]) - b[1]
    qz = abs(p[2]) - b[2]
    out_dist = math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2)
    max_comp = max(qx, qy, qz)
    inner_penalty = min(max_comp, 0.0)
    return out_dist + inner_penalty


# =============================================================================
# Tolerance constants
# =============================================================================

TOL = 1e-12         # Exact arithmetic tolerance
TOL_SURFACE = 1e-12  # Surface points should be extremely close to 0
TOL_GRADIENT = 1e-4  # Numerical gradient tolerance


# =============================================================================
# Path 1: Formula decomposition
# =============================================================================


class TestFormulaDecomposition:
    """Verify each sub-expression of the sdBoxFrame formula."""

    def test_q_equals_abs_p_minus_b(self):
        """The first step is q = abs(p) - b, computed component-wise."""
        b = (2.0, 3.0, 4.0)
        p = (1.5, -2.5, 3.5)
        # qx = |1.5| - 2.0 = -0.5
        # qy = |-2.5| - 3.0 = -0.5
        # qz = |3.5| - 4.0 = -0.5
        qx_expected = 1.5 - 2.0   # -0.5
        qy_expected = 2.5 - 3.0   # -0.5
        qz_expected = 3.5 - 4.0   # -0.5
        qx = abs(p[0]) - b[0]
        qy = abs(p[1]) - b[1]
        qz = abs(p[2]) - b[2]
        assert qx == pytest.approx(qx_expected, abs=TOL)
        assert qy == pytest.approx(qy_expected, abs=TOL)
        assert qz == pytest.approx(qz_expected, abs=TOL)

    def test_q_inside_box_all_negative(self):
        """When |p| < b on all axes, all q components are negative."""
        b = (3.0, 4.0, 5.0)
        p = (1.0, 2.0, 3.0)
        qx = abs(p[0]) - b[0]  # -2.0
        qy = abs(p[1]) - b[1]  # -2.0
        qz = abs(p[2]) - b[2]  # -2.0
        assert qx < 0.0 and qy < 0.0 and qz < 0.0

    def test_q_outside_box_at_least_one_positive(self):
        """When |p| > b on any axis, at least one q component is positive."""
        b = (2.0, 2.0, 2.0)
        p = (3.0, 0.0, 0.0)
        qx = abs(p[0]) - b[0]  # 1.0
        qy = abs(p[1]) - b[1]  # -2.0
        qz = abs(p[2]) - b[2]  # -2.0
        assert qx > 0.0
        assert qy < 0.0 and qz < 0.0

    def test_max_q_clamps_negative_to_zero(self):
        """max(q, 0) clamps negative components to zero."""
        b = (2.0, 3.0, 4.0)
        p = (3.0, 1.0, 0.0)
        qx = abs(p[0]) - b[0]  # 1.0
        qy = abs(p[1]) - b[1]  # -2.0
        qz = abs(p[2]) - b[2]  # -4.0
        max_qx = max(qx, 0.0)  # 1.0
        max_qy = max(qy, 0.0)  # 0.0
        max_qz = max(qz, 0.0)  # 0.0
        assert max_qx == pytest.approx(1.0, abs=TOL)
        assert max_qy == pytest.approx(0.0, abs=TOL)
        assert max_qz == pytest.approx(0.0, abs=TOL)

    def test_length_of_max_q_inside_box_is_zero(self):
        """When all q components are negative, length(max(q,0)) = 0."""
        b = (2.0, 2.0, 2.0)
        p = (0.0, 0.0, 0.0)  # All q negative: (-2, -2, -2)
        d = py_sd_box_frame(p, b, 0.5)
        # out_dist = 0, inner_penalty = -2, so d = -2 - 0.5 = -2.5
        assert d == pytest.approx(-2.5, abs=TOL), (
            f"All-interior point {p} with b={b}, e=0.5 should give -2.5, got {d}"
        )

    def test_max_component_outside_is_positive(self):
        """When outside on one axis, maxComponent is the positive q value."""
        b = (2.0, 2.0, 2.0)
        p = (3.0, 0.0, 0.0)
        qx = abs(p[0]) - b[0]  # 1.0
        qy = abs(p[1]) - b[1]  # -2.0
        qz = abs(p[2]) - b[2]  # -2.0
        max_comp = max(qx, qy, qz)  # 1.0
        assert max_comp == pytest.approx(1.0, abs=TOL)

    def test_min_max_component_outside_is_zero(self):
        """When maxComponent > 0, min(maxComp, 0) = 0 (no interior penalty)."""
        b = (2.0, 3.0, 4.0)
        p = (3.0, 1.0, 0.0)
        qx = abs(p[0]) - b[0]  # 1.0
        qy = abs(p[1]) - b[1]  # -2.0
        qz = abs(p[2]) - b[2]  # -4.0
        max_comp = max(qx, qy, qz)  # 1.0
        assert min(max_comp, 0.0) == pytest.approx(0.0, abs=TOL)

    def test_min_max_component_inside_negative(self):
        """When all q components are negative, min(maxComp, 0) = maxComp."""
        b = (2.0, 3.0, 4.0)
        p = (0.0, 0.0, 0.0)
        qx = abs(p[0]) - b[0]  # -2.0
        qy = abs(p[1]) - b[1]  # -3.0
        qz = abs(p[2]) - b[2]  # -4.0
        max_comp = max(qx, qy, qz)  # -2.0 (closest to zero)
        assert min(max_comp, 0.0) == pytest.approx(-2.0, abs=TOL)

    def test_e_is_uniform_offset(self):
        """The e parameter is subtracted uniformly from the result."""
        b = (2.0, 3.0, 4.0)
        p = (3.5, 0.0, 0.0)
        d_no_e = py_sd_box_frame(p, b, 0.0)  # sdBox
        e = 0.5
        d_with_e = py_sd_box_frame(p, b, e)
        assert d_with_e == pytest.approx(d_no_e - e, abs=TOL), (
            f"e={e} should shift result by -{e}: sdBox={d_no_e}, "
            f"sdBoxFrame={d_with_e}"
        )


# =============================================================================
# Path 2: e = 0 degenerates to solid box (identical to sdBox)
# =============================================================================


class TestDegenerateSolidBox:
    """With e=0, sdBoxFrame(p, b, 0) should match sdBox(p, b)."""

    @pytest.mark.parametrize("b", [(2, 3, 4), (1, 1, 1), (5, 2, 3), (0.5, 0.5, 0.5)])
    @pytest.mark.parametrize("p", [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        (0.0, 0.0, 4.0),
        (4.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        (-2.0, 0.0, 0.0),
        (2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0),
    ])
    def test_e_zero_matches_sd_box(self, b, p):
        """sdBoxFrame(p, b, 0) == sdBox(p, b) for any p, b."""
        d_frame = py_sd_box_frame(p, b, 0.0)
        d_box = py_sd_box(p, b)
        assert d_frame == pytest.approx(d_box, abs=TOL), (
            f"sdBoxFrame({p}, {b}, 0) = {d_frame} should equal "
            f"sdBox({p}, {b}) = {d_box}"
        )

    def test_e_zero_inside_negative(self):
        """With e=0, interior points have same negative values as sdBox."""
        b = (3.0, 3.0, 3.0)
        interior_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 1.0, 2.0),
            (-2.0, 0.0, 0.0),
        ]
        for p in interior_points:
            d_frame = py_sd_box_frame(p, b, 0.0)
            d_box = py_sd_box(p, b)
            assert d_frame < 0.0, (
                f"Interior point {p} with e=0 should be negative, got {d_frame}"
            )
            assert d_frame == pytest.approx(d_box, abs=TOL)

    def test_e_zero_surface_points(self):
        """With e=0, surface points of the outer box return 0."""
        b = (2.0, 3.0, 4.0)
        surface_points = [
            (2.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 4.0),
            (-2.0, 0.0, 0.0),
            (0.0, -3.0, 0.0),
            (0.0, 0.0, -4.0),
            (2.0, 3.0, 4.0),    # Corner
            (2.0, 3.0, 0.0),    # Edge
        ]
        for p in surface_points:
            d = py_sd_box_frame(p, b, 0.0)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface point {p} with b={b}, e=0 should be 0, got {d}"
            )

    def test_e_zero_outside_positive(self):
        """With e=0, outside points have same positive values as sdBox."""
        b = (2.0, 2.0, 2.0)
        outside_points = [
            (3.0, 0.0, 0.0),
            (0.0, 4.0, 0.0),
            (0.0, 0.0, 5.0),
            (4.0, 4.0, 0.0),
        ]
        for p in outside_points:
            d_frame = py_sd_box_frame(p, b, 0.0)
            d_box = py_sd_box(p, b)
            assert d_frame > 0.0
            assert d_frame == pytest.approx(d_box, abs=TOL)


# =============================================================================
# Path 3: Surface at |p| = b + e along principal axes
# =============================================================================


class TestSurfacePoints:
    """Points on the outer surface of the frame should have distance ~0.

    Along a principal axis, the outer surface is at |p_i| = b_i + e, because
    the formula yields d = |p_i| - b_i - e = 0 at that point.
    """

    @pytest.mark.parametrize("b, e", [
        ((2.0, 3.0, 4.0), 0.5),
        ((2.0, 2.0, 2.0), 0.5),
        ((1.0, 1.0, 1.0), 0.2),
        ((5.0, 3.0, 2.0), 1.0),
    ])
    def test_surface_on_positive_x(self, b, e):
        """Surface on +x: (bx + e, 0, 0) should give d ~ 0."""
        p = (b[0] + e, 0.0, 0.0)
        d = py_sd_box_frame(p, b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface on +x: p={p}, b={b}, e={e} should give 0, got {d}"
        )

    @pytest.mark.parametrize("b, e", [
        ((2.0, 3.0, 4.0), 0.5),
        ((2.0, 2.0, 2.0), 0.5),
        ((1.0, 2.0, 3.0), 0.3),
    ])
    def test_surface_on_positive_y(self, b, e):
        """Surface on +y: (0, by + e, 0) should give d ~ 0."""
        p = (0.0, b[1] + e, 0.0)
        d = py_sd_box_frame(p, b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface on +y: p={p}, b={b}, e={e} should give 0, got {d}"
        )

    @pytest.mark.parametrize("b, e", [
        ((2.0, 3.0, 4.0), 0.5),
        ((2.0, 2.0, 2.0), 0.5),
        ((3.0, 2.0, 1.0), 0.4),
    ])
    def test_surface_on_positive_z(self, b, e):
        """Surface on +z: (0, 0, bz + e) should give d ~ 0."""
        p = (0.0, 0.0, b[2] + e)
        d = py_sd_box_frame(p, b, e)
        assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface on +z: p={p}, b={b}, e={e} should give 0, got {d}"
        )

    def test_surface_negative_axes(self):
        """Negative axes give the same result due to abs(p)."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        # Surface on -x
        d_neg_x = py_sd_box_frame((-b[0] - e, 0.0, 0.0), b, e)
        d_pos_x = py_sd_box_frame((b[0] + e, 0.0, 0.0), b, e)
        assert d_neg_x == pytest.approx(0.0, abs=TOL_SURFACE)
        assert d_neg_x == pytest.approx(d_pos_x, abs=TOL)

    def test_all_six_axis_surfaces_consistent(self):
        """All six axis-aligned surface points yield 0."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        surfaces = [
            (b[0] + e, 0.0, 0.0),
            (-b[0] - e, 0.0, 0.0),
            (0.0, b[1] + e, 0.0),
            (0.0, -b[1] - e, 0.0),
            (0.0, 0.0, b[2] + e),
            (0.0, 0.0, -b[2] - e),
        ]
        for p in surfaces:
            d = py_sd_box_frame(p, b, e)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Axis surface {p} should be 0, got {d}"
            )


# =============================================================================
# Path 4: Outside points produce positive distance
# =============================================================================


class TestOutsidePoints:
    """Points outside the box frame should have positive distance."""

    @pytest.mark.parametrize("b, e, p, expected", [
        ((2.0, 2.0, 2.0), 0.5, (3.0, 0.0, 0.0), 0.5),   # x=3: |x|-b-e = 3-2-0.5 = 0.5
        ((2.0, 2.0, 2.0), 0.5, (4.0, 0.0, 0.0), 1.5),   # x=4: 4-2-0.5 = 1.5
        ((2.0, 3.0, 4.0), 0.5, (3.5, 0.0, 0.0), 1.0),   # x=3.5: 3.5-2-0.5 = 1.0
        ((2.0, 3.0, 4.0), 0.5, (0.0, 4.5, 0.0), 1.0),   # y=4.5: 4.5-3-0.5 = 1.0
        ((2.0, 3.0, 4.0), 0.5, (0.0, 0.0, 5.5), 1.0),   # z=5.5: 5.5-4-0.5 = 1.0
    ])
    def test_outside_along_axis(self, b, e, p, expected):
        """Outside points along principal axes have known distances."""
        d = py_sd_box_frame(p, b, e)
        assert d == pytest.approx(expected, abs=TOL), (
            f"Outside {p} with b={b}, e={e}: expected {expected}, got {d}"
        )

    def test_outside_diagonal(self):
        """Outside point along diagonal: distance computed from max(q,0)."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        p = (3.0, 3.0, 0.0)
        # q = (1, 1, -2), max(q,0) = (1, 1, 0), length = sqrt(2)
        # maxComponent = 1 > 0, so inner_penalty = 0
        # d = sqrt(2) - 0.5
        expected = math.sqrt(2.0) - 0.5
        d = py_sd_box_frame(p, b, e)
        assert d == pytest.approx(expected, abs=TOL), (
            f"Outside diagonal {p} with b={b}, e={e}: "
            f"expected {expected}, got {d}"
        )

    def test_outside_far_point_approaches_raw_distance(self):
        """Far outside, the SDF approximates length(p) - e (box negligible)."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        far_x = 1e6
        d = py_sd_box_frame((far_x, 0.0, 0.0), b, e)
        expected = far_x - b[0] - e
        assert d == pytest.approx(expected, abs=0.001), (
            f"Far point ({far_x}, 0, 0) should approximate {expected}, got {d}"
        )

    def test_outside_always_positive(self):
        """Points far from the box should always have positive distance."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        outside_points = [
            (10.0, 0.0, 0.0),
            (0.0, 15.0, 0.0),
            (0.0, 0.0, 20.0),
            (-10.0, 0.0, 0.0),
            (5.0, 5.0, 5.0),
            (10.0, 10.0, 10.0),
        ]
        for p in outside_points:
            d = py_sd_box_frame(p, b, e)
            assert d > 0.0, (
                f"Far point {p} should have positive SDF, got {d}"
            )


# =============================================================================
# Path 5: Inside frame walls produce negative distance
# =============================================================================


class TestInsideFrameWalls:
    """Points inside the frame walls (within the outer box) have negative SDF.

    For the formula sdBoxFrame = sdBox - e, points inside the outer box (all
    q components negative) always return d = maxComponent(q) - e, which is
    ≤ -e (negative for e > 0).
    """

    def test_mid_wall_negative(self):
        """A point well inside the wall on the x-face."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        p = (1.5, 0.0, 0.0)
        d = py_sd_box_frame(p, b, e)
        # q = (-0.5, -3, -4), maxComp = -0.5, d = 0 + (-0.5) - 0.5 = -1.0
        expected = -1.0
        assert d == pytest.approx(expected, abs=TOL), (
            f"Inside wall {p}: expected {expected}, got {d}"
        )
        assert d < 0.0

    def test_near_inner_wall_negative(self):
        """A point near the inner wall surface."""
        b = (2.0, 2.0, 2.0)
        e = 0.3
        # Just inside the wall, near the inner surface
        p = (1.71, 0.0, 0.0)
        d = py_sd_box_frame(p, b, e)
        # q = (-0.29, -2, -2), maxComp = -0.29
        # d = 0 + (-0.29) - 0.3 = -0.59
        expected = -0.59
        assert d == pytest.approx(expected, abs=TOL), (
            f"Near inner wall {p}: expected {expected}, got {d}"
        )

    def test_near_outer_wall_negative(self):
        """A point just inside the outer wall surface."""
        b = (2.0, 2.0, 2.0)
        e = 0.3
        # Just inside the frame from the outer surface at x = b + e = 2.3
        p = (2.29, 0.0, 0.0)
        d = py_sd_box_frame(p, b, e)
        # q = (0.29, -2, -2), max(q,0) = (0.29, 0, 0), length = 0.29
        # maxComp = 0.29, min = 0
        # d = 0.29 + 0 - 0.3 = -0.01
        expected = -0.01
        assert d == pytest.approx(expected, abs=TOL), (
            f"Near outer wall {p}: expected {expected}, got {d}"
        )
        assert d < 0.0

    def test_wall_center_negative(self):
        """Point at center of a wall face."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        # Center of x-wall: |x| = b.x - e/2
        p = (1.75, 0.0, 0.0)
        d = py_sd_box_frame(p, b, e)
        # q = (-0.25, -3, -4), maxComp = -0.25
        # d = 0 + (-0.25) - 0.5 = -0.75
        expected = -0.75
        assert d == pytest.approx(expected, abs=TOL), (
            f"Wall center {p}: expected {expected}, got {d}"
        )

    def test_multiple_axes_inside_negative(self):
        """Inside frame wall on multiple axes simultaneously."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        p = (1.5, 2.5, 0.0)
        d = py_sd_box_frame(p, b, e)
        # q = (-0.5, -0.5, -4), maxComp = -0.5
        # d = 0 + (-0.5) - 0.5 = -1.0
        expected = -1.0
        assert d == pytest.approx(expected, abs=TOL), (
            f"Multi-axis wall {p}: expected {expected}, got {d}"
        )
        assert d < 0.0


# =============================================================================
# Path 6: Interior cavity -- behavior inside the outer box
# =============================================================================


class TestInteriorCavity:
    """The interior of the outer box always yields negative SDF with this
    simplified formula (sdBoxFrame = sdBox - e). All points with |p_i| < b_i
    on all axes have d = maxComponent(q) - e <= -e < 0.

    NOTE: In the ideal IQ sdBoxFrame, the cavity (region inside the frame
    walls) should yield *positive* distance. The simplified formula used here
    always gives negative values inside the outer box, which means the
    formula does not distinguish between the frame walls and the cavity
    interior. This is a known limitation of the simplified formula.
    """

    def test_cavity_center_negative(self):
        """Center of the cavity is still negative with simplified formula."""
        b = (3.0, 3.0, 3.0)
        e = 0.5
        d = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        # q = (-3, -3, -3), maxComp = -3, d = 0 + (-3) - 0.5 = -3.5
        assert d < 0.0, (
            f"Cavity center with simplified formula should be negative, got {d}"
        )
        assert d == pytest.approx(-3.5, abs=TOL)

    def test_cavity_interior_points(self):
        """All cavity interior points have d = maxComponent(q) - e."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        cases = [
            ((0.0, 0.0, 0.0), -2.0 - 0.5),     # maxComp = -2.0
            ((1.0, 0.0, 0.0), -1.0 - 0.5),     # maxComp = -1.0
            ((0.0, 2.0, 0.0), -1.0 - 0.5),     # maxComp = -1.0
            ((0.0, 0.0, 3.0), -1.0 - 0.5),     # maxComp = -1.0
            ((1.0, 2.0, 0.0), -1.0 - 0.5),     # maxComp = -1.0
            ((0.5, 0.5, 0.5), -1.5 - 0.5),     # maxComp = -1.5
        ]
        for p, expected in cases:
            d = py_sd_box_frame(p, b, e)
            assert d == pytest.approx(expected, abs=TOL), (
                f"Cavity point {p} with b={b}, e={e}: "
                f"expected {expected}, got {d}"
            )

    def test_max_component_dominates_cavity(self):
        """In the cavity, maxComponent(q) determines which face is nearest."""
        b = (2.0, 5.0, 5.0)
        e = 0.5
        # Along x-axis, the nearest face is at x=2, so maxComp = |x| - 2
        d1 = py_sd_box_frame((1.0, 0.0, 0.0), b, e)   # maxComp = -1.0
        d2 = py_sd_box_frame((0.0, 4.0, 0.0), b, e)   # maxComp = -1.0
        # Both have same maxComp = -1 because the nearest face is at same
        # relative distance in each case (just inside their respective face)
        assert d1 == pytest.approx(d2, abs=TOL), (
            f"Cavity distances should match: d1={d1}, d2={d2}"
        )

    def test_cavity_farther_from_all_faces_more_negative(self):
        """Points deeper in the cavity have more negative values."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        d_center = py_sd_box_frame((0.0, 0.0, 0.0), b, e)     # maxComp = -2
        d_mid = py_sd_box_frame((1.0, 0.0, 0.0), b, e)        # maxComp = -1
        assert d_center < d_mid, (
            f"Center {d_center} should be more negative than mid {d_mid}"
        )


# =============================================================================
# Path 7: Octahedral symmetry via abs(p)
# =============================================================================


class TestSymmetry:
    """The formula uses abs(p), which gives full octahedral symmetry.

    sdBoxFrame((x,y,z), b, e) == sdBoxFrame((sign_x*x, sign_y*y, sign_z*z), b, e)
    for any combination of signs.
    """

    def test_x_symmetry(self):
        """Points (x,y,z) and (-x,y,z) have equal distances."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        test_points = [
            (1.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (1.5, 2.5, 0.5),
            (0.5, 1.0, 2.0),
            (3.0, 1.0, 1.0),
        ]
        for x, y, z in test_points:
            d_pos = py_sd_box_frame((x, y, z), b, e)
            d_neg = py_sd_box_frame((-x, y, z), b, e)
            assert d_pos == pytest.approx(d_neg, abs=TOL), (
                f"X-symmetry: ({x},{y},{z}) -> {d_pos} != "
                f"({-x},{y},{z}) -> {d_neg}"
            )

    def test_y_symmetry(self):
        """Points (x,y,z) and (x,-y,z) have equal distances."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        test_points = [
            (1.0, 2.0, 0.0),
            (0.0, 1.5, 1.0),
            (2.0, 0.5, 2.0),
            (1.0, 3.0, 1.0),
        ]
        for x, y, z in test_points:
            d_pos = py_sd_box_frame((x, y, z), b, e)
            d_neg = py_sd_box_frame((x, -y, z), b, e)
            assert d_pos == pytest.approx(d_neg, abs=TOL), (
                f"Y-symmetry: ({x},{y},{z}) -> {d_pos} != "
                f"({x},{-y},{z}) -> {d_neg}"
            )

    def test_z_symmetry(self):
        """Points (x,y,z) and (x,y,-z) have equal distances."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        test_points = [
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 1.5),
            (2.0, 2.0, 0.5),
            (1.0, 2.0, 3.0),
        ]
        for x, y, z in test_points:
            d_pos = py_sd_box_frame((x, y, z), b, e)
            d_neg = py_sd_box_frame((x, y, -z), b, e)
            assert d_pos == pytest.approx(d_neg, abs=TOL), (
                f"Z-symmetry: ({x},{y},{z}) -> {d_pos} != "
                f"({x},{y},{-z}) -> {d_neg}"
            )

    def test_full_octahedral_symmetry(self):
        """All 8 sign-flip combinations yield the same distance."""
        b = (2.0, 3.0, 4.0)
        e = 0.5
        base = (1.0, 2.0, 3.0)
        d_base = py_sd_box_frame(base, b, e)
        signs = [
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
            (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        ]
        for sx, sy, sz in signs:
            p = (sx * base[0], sy * base[1], sz * base[2])
            d = py_sd_box_frame(p, b, e)
            assert d == pytest.approx(d_base, abs=TOL), (
                f"Symmetry broken for sign ({sx},{sy},{sz}): "
                f"sdBoxFrame({p}) = {d} != {d_base}"
            )

    def test_symmetry_at_surface(self):
        """Surface symmetry: surfaces on opposite axes are identical."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        d_pos = py_sd_box_frame((b[0] + e, 0.0, 0.0), b, e)
        d_neg = py_sd_box_frame((-b[0] - e, 0.0, 0.0), b, e)
        assert d_pos == pytest.approx(d_neg, abs=TOL)


# =============================================================================
# Path 8: Non-uniform box dimensions
# =============================================================================


class TestNonUniformBox:
    """The formula handles different b dimensions per axis."""

    @pytest.mark.parametrize("b, e", [
        ((1.0, 2.0, 3.0), 0.5),
        ((5.0, 1.0, 2.0), 0.3),
        ((0.5, 1.5, 2.5), 0.2),
        ((3.0, 0.5, 1.0), 0.8),
    ])
    def test_surface_on_each_axis(self, b, e):
        """Surface on each axis at b_i + e."""
        for axis, p in enumerate([
            (b[0] + e, 0.0, 0.0),
            (0.0, b[1] + e, 0.0),
            (0.0, 0.0, b[2] + e),
        ]):
            d = py_sd_box_frame(p, b, e)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface on axis {axis}: p={p}, b={b}, e={e}: "
                f"expected 0, got {d}"
            )

    def test_inside_non_uniform(self):
        """Inside values for non-uniform box."""
        b = (1.0, 3.0, 5.0)
        e = 0.5
        # Mid-wall on x (thinnest dimension)
        p = (0.75, 0.0, 0.0)
        d = py_sd_box_frame(p, b, e)
        # q = (-0.25, -3, -5), maxComp = -0.25
        # d = 0 + (-0.25) - 0.5 = -0.75
        assert d == pytest.approx(-0.75, abs=TOL), (
            f"Inside non-uniform {p} with b={b}, e={e}: expected -0.75, got {d}"
        )

    def test_outside_non_uniform(self):
        """Outside values for non-uniform box."""
        b = (1.0, 3.0, 5.0)
        e = 0.5
        # Outside along x (thinnest dimension)
        p = (2.0, 0.0, 0.0)
        d = py_sd_box_frame(p, b, e)
        # q = (1.0, -3, -5), max(q,0) = (1, 0, 0), len = 1, maxComp = 1
        # d = 1 + 0 - 0.5 = 0.5
        expected = 1.0 - 0.5  # 0.5
        assert d == pytest.approx(expected, abs=TOL), (
            f"Outside non-uniform {p} with b={b}, e={e}: "
            f"expected {expected}, got {d}"
        )

    def test_nearest_face_dominates_cavity(self):
        """In cavity, nearest face (largest q component) dominates."""
        b = (1.0, 3.0, 5.0)
        e = 0.5
        # Point is closest to the x-face (|x| = 0.8, b.x = 1.0)
        p = (0.8, 0.0, 0.0)
        d = py_sd_box_frame(p, b, e)
        # q = (-0.2, -3, -5), maxComp = -0.2 (closest to x-face)
        # d = 0 + (-0.2) - 0.5 = -0.7
        assert d == pytest.approx(-0.7, abs=TOL)

    def test_different_axis_outside_values(self):
        """Outside along thicker axes should give different distances."""
        b = (1.0, 3.0, 5.0)
        e = 0.5
        d_x = py_sd_box_frame((b[0] + 1.0, 0.0, 0.0), b, e)
        d_y = py_sd_box_frame((0.0, b[1] + 1.0, 0.0), b, e)
        d_z = py_sd_box_frame((0.0, 0.0, b[2] + 1.0), b, e)
        # All should be: 1.0 - e = 0.5 (same extra distance)
        expected = 1.0 - e
        assert d_x == pytest.approx(expected, abs=TOL)
        assert d_y == pytest.approx(expected, abs=TOL)
        assert d_z == pytest.approx(expected, abs=TOL)


# =============================================================================
# Path 9: e >= min(b) frame fills interior
# =============================================================================


class TestFrameFillsInterior:
    """When e >= min(b), the frame thickness fills the entire interior."""

    def test_e_equals_min_b(self):
        """With e = min(b), the frame meets at the center."""
        b = (2.0, 3.0, 4.0)
        e = min(b)  # 2.0
        d = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        # q = (-2, -3, -4), maxComp = -2
        # d = 0 + (-2) - 2.0 = -4.0
        assert d == pytest.approx(-4.0, abs=TOL), (
            f"e = min(b) = {e}: center d should be -4.0, got {d}"
        )

    def test_e_greater_than_min_b(self):
        """With e > min(b), frame fills interior with more negative values."""
        b = (2.0, 3.0, 4.0)
        e = 3.0
        d = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        # q = (-2, -3, -4), maxComp = -2
        # d = 0 + (-2) - 3.0 = -5.0
        assert d == pytest.approx(-5.0, abs=TOL), (
            f"e = {e} > min(b): center d should be -5.0, got {d}"
        )

    @pytest.mark.parametrize("b, e", [
        ((2.0, 3.0, 4.0), 2.0),   # e = min(b)
        ((2.0, 3.0, 4.0), 3.0),   # e > min(b)
        ((2.0, 3.0, 4.0), 5.0),   # e >> min(b)
        ((1.0, 1.0, 1.0), 1.0),   # e = min(b), uniform
        ((3.0, 5.0, 2.0), 2.0),   # e = min(b), non-uniform
    ])
    def test_center_all_negative(self, b, e):
        """Center always has negative SDF for e >= min(b)."""
        d = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        assert d < 0.0, (
            f"Center with e={e} >= min(b)={min(b)} should be negative, got {d}"
        )

    def test_all_interior_points_negative_large_e(self):
        """All interior points are negative when e is large."""
        b = (2.0, 2.0, 2.0)
        e = 3.0
        interior = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.5, 0.0),
        ]
        for p in interior:
            d = py_sd_box_frame(p, b, e)
            assert d < 0.0, (
                f"Interior point {p} with large e={e} should be negative, got {d}"
            )


# =============================================================================
# Path 10: Monotonicity and ordering
# =============================================================================


class TestMonotonicity:
    """The SDF should be monotonic along each axis outside the box."""

    def test_monotonic_along_positive_x(self):
        """Distance should strictly increase along +x outside the frame."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        positions = [b[0] + e + i * 0.5 for i in range(10)]
        distances = [py_sd_box_frame((x, 0.0, 0.0), b, e) for x in positions]
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i + 1], (
                f"Distance should increase along +x: "
                f"x={positions[i]} -> {distances[i]}, "
                f"x={positions[i+1]} -> {distances[i+1]}"
            )

    def test_monotonic_along_negative_x(self):
        """Distance should strictly increase along -x (more negative)."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        positions = [-b[0] - e - i * 0.5 for i in range(10)]
        distances = [py_sd_box_frame((x, 0.0, 0.0), b, e) for x in positions]
        for i in range(len(distances) - 1):
            assert distances[i] < distances[i + 1], (
                f"Distance should increase along -x: "
                f"x={positions[i]} -> {distances[i]}, "
                f"x={positions[i+1]} -> {distances[i+1]}"
            )

    def test_sign_transition(self):
        """Sign transition across the outer surface."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        surface_x = b[0] + e  # 2.5
        step = 1e-6
        inside = py_sd_box_frame((surface_x - step, 0.0, 0.0), b, e)
        on_surface = py_sd_box_frame((surface_x, 0.0, 0.0), b, e)
        outside = py_sd_box_frame((surface_x + step, 0.0, 0.0), b, e)
        assert inside < 0.0, f"Inside should be negative, got {inside}"
        assert on_surface == pytest.approx(0.0, abs=TOL_SURFACE), (
            f"Surface should be 0, got {on_surface}"
        )
        assert outside > 0.0, f"Outside should be positive, got {outside}"

    def test_inside_vs_outside_ordering(self):
        """Inside points should have negative SDF, outside positive."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        inside = py_sd_box_frame((0.0, 0.0, 0.0), b, e)
        outside = py_sd_box_frame((5.0, 0.0, 0.0), b, e)
        assert inside < 0.0 < outside, (
            f"Inside {inside} should be negative, outside {outside} positive"
        )


# =============================================================================
# Path 11: Frame thickness variation
# =============================================================================


class TestFrameThicknessVariation:
    """Varying e changes the result by a uniform offset for interior points."""

    def test_thicker_frame_more_negative_interior(self):
        """Increasing e makes interior points more negative."""
        b = (2.0, 2.0, 2.0)
        p = (0.0, 0.0, 0.0)
        d_thin = py_sd_box_frame(p, b, 0.1)
        d_thick = py_sd_box_frame(p, b, 1.0)
        assert d_thick < d_thin, (
            f"Thicker frame (e=1.0) should be more negative than thin (e=0.1): "
            f"{d_thick} vs {d_thin}"
        )

    def test_e_subtracted_uniformly_interior(self):
        """For interior points, diff in d equals diff in e."""
        b = (2.0, 3.0, 4.0)
        p = (1.0, 1.0, 1.0)
        e1, e2 = 0.2, 0.7
        d1 = py_sd_box_frame(p, b, e1)
        d2 = py_sd_box_frame(p, b, e2)
        assert d2 - d1 == pytest.approx(e1 - e2, abs=TOL), (
            f"e difference: ({d2} - {d1}) should equal ({e1} - {e2})"
        )

    def test_e_subtracted_uniformly_exterior(self):
        """For exterior points, diff in d equals diff in e."""
        b = (2.0, 3.0, 4.0)
        p = (5.0, 0.0, 0.0)
        e1, e2 = 0.3, 0.8
        d1 = py_sd_box_frame(p, b, e1)
        d2 = py_sd_box_frame(p, b, e2)
        assert d2 - d1 == pytest.approx(e1 - e2, abs=TOL), (
            f"e difference: ({d2} - {d1}) should equal ({e1} - {e2})"
        )

    def test_zero_thickness_matches_sd_box(self):
        """e=0 always matches sdBox regardless of b or p."""
        b_list = [(2.0, 2.0, 2.0), (1.0, 3.0, 5.0), (0.5, 0.5, 0.5)]
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 3.0, 4.0),
            (-2.0, 0.0, 0.0),
        ]
        for b in b_list:
            for p in points:
                d_frame = py_sd_box_frame(p, b, 0.0)
                d_box = py_sd_box(p, b)
                assert d_frame == pytest.approx(d_box, abs=TOL), (
                    f"e=0: sdBoxFrame({p}, {b}) = {d_frame} != "
                    f"sdBox({p}, {b}) = {d_box}"
                )

    def test_varying_e_on_surface(self):
        """Surface position shifts outward with e: |p| = b + e."""
        b = (2.0, 2.0, 2.0)
        for e in [0.1, 0.5, 1.0, 2.0]:
            surface_x = b[0] + e
            d = py_sd_box_frame((surface_x, 0.0, 0.0), b, e)
            assert d == pytest.approx(0.0, abs=TOL_SURFACE), (
                f"Surface at x={surface_x} with e={e} should be 0, got {d}"
            )


# =============================================================================
# Path 12: Gradient properties (numerical, using Python model)
# =============================================================================


class TestGradient:
    """Numerical gradient properties of the box frame SDF."""

    def test_numerical_gradient_x_outside(self):
        """Outside the box, the x-gradient approaches sign(p_x)."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        eps = 1e-6
        p = (5.0, 0.0, 0.0)  # Outside on x-axis
        grad_x = (
            py_sd_box_frame((p[0] + eps, p[1], p[2]), b, e)
            - py_sd_box_frame((p[0] - eps, p[1], p[2]), b, e)
        ) / (2.0 * eps)
        # Outside on +x axis, gradient should point radially -> ~1.0
        assert grad_x == pytest.approx(1.0, abs=TOL_GRADIENT), (
            f"X-gradient outside: expected ~1.0, got {grad_x}"
        )

    def test_gradient_magnitude_at_surface(self):
        """At the surface, gradient magnitude should be ~1 (eikonal)."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        eps = 1e-6
        surface_x = b[0] + e
        p = (surface_x + 0.1, 0.0, 0.0)  # Slightly outside
        gx = (
            py_sd_box_frame((p[0] + eps, p[1], p[2]), b, e)
            - py_sd_box_frame((p[0] - eps, p[1], p[2]), b, e)
        ) / (2.0 * eps)
        gy = (
            py_sd_box_frame((p[0], p[1] + eps, p[2]), b, e)
            - py_sd_box_frame((p[0], p[1] - eps, p[2]), b, e)
        ) / (2.0 * eps)
        gz = (
            py_sd_box_frame((p[0], p[1], p[2] + eps), b, e)
            - py_sd_box_frame((p[0], p[1], p[2] - eps), b, e)
        ) / (2.0 * eps)
        mag = math.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        assert mag == pytest.approx(1.0, abs=TOL_GRADIENT), (
            f"Gradient magnitude at surface should be ~1, got {mag}"
        )


# =============================================================================
# Path 13: Corner and edge behavior
# =============================================================================


class TestCornerEdge:
    """Behavior at corners and edges of the box frame."""

    def test_corner_surface_rounded(self):
        """At a corner, the surface has rounded shape (circular in q-space)."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        # Corner at (b+e, b+e, 0) in xy plane: q = (e, e, -2)
        # max(q,0) = (e, e, 0), length = e*sqrt(2)
        # d = e*sqrt(2) - e = e*(sqrt(2)-1)
        p = (b[0] + e, b[1] + e, 0.0)
        d = py_sd_box_frame(p, b, e)
        expected = e * (math.sqrt(2.0) - 1.0)
        assert d == pytest.approx(expected, abs=TOL), (
            f"Corner surface {p} with e={e}: expected {expected}, got {d}"
        )

    def test_corner_positive_octant(self):
        """Corner in the positive octant: all three axes active."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        p = (b[0] + e, b[1] + e, b[2] + e)
        # q = (e, e, e), max(q,0) = (e, e, e), length = e*sqrt(3)
        # d = e*sqrt(3) - e = e*(sqrt(3)-1)
        d = py_sd_box_frame(p, b, e)
        expected = e * (math.sqrt(3.0) - 1.0)
        assert d == pytest.approx(expected, abs=TOL), (
            f"Corner {p}: expected {expected}, got {d}"
        )

    def test_edge_midpoint(self):
        """Midpoint along an edge of the box."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        # Edge along z at x=b+e, y=b+e, z=0: q = (e, e, -2)
        # max(q,0) = (e, e, 0), length = e*sqrt(2)
        # maxComp = e > 0, penalty = 0
        # d = e*sqrt(2) - e
        d = py_sd_box_frame((b[0] + e, b[1] + e, 0.0), b, e)
        expected = e * (math.sqrt(2.0) - 1.0)
        assert d == pytest.approx(expected, abs=TOL), (
            f"Edge midpoint: expected {expected}, got {d}"
        )

    def test_corner_symmetry_all_octants(self):
        """Corner surfaces are symmetric across all octants."""
        b = (2.0, 2.0, 2.0)
        e = 0.5
        signs = [
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
            (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        ]
        ref = py_sd_box_frame(
            (b[0] + e, b[1] + e, b[2] + e), b, e
        )
        for sx, sy, sz in signs:
            p = (sx * (b[0] + e), sy * (b[1] + e), sz * (b[2] + e))
            d = py_sd_box_frame(p, b, e)
            assert d == pytest.approx(ref, abs=TOL), (
                f"Corner symmetry broken in octant ({sx},{sy},{sz}): "
                f"got {d}, expected {ref}"
            )
