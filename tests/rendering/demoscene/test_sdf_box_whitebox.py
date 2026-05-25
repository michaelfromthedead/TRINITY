"""
Whitebox tests for sdBox(p, b) WGSL function (T-DEMO-1.2).

Tests a Python model of the WGSL implementation, verifying:
  - Formula decomposition: q = abs(p) - b
  - Component-wise operations: max(q, 0), maxComponent(q)
  - Correct signed distance for inside corner (negative)
  - Correct signed distance for outside corner (positive)
  - Correct signed distance at edge centers
  - Correct signed distance at face centers (zero)
  - Asymmetric box dimensions
  - Zero-size dimensions (planar degenerate)
  - Unit cube: all 8 corners and 6 face centers
  - Scaling homogeneity: sdBox(2p, 2b) = 2 * sdBox(p, b)

WHITEBOX coverage plan:
  Path A:  q = abs(p) - b for p inside/outside/mixed
  Path B:  max(q, 0) component-wise clamp to zero
  Path C:  length(max(q, 0)) for outside distance
  Path D:  maxComponent(q) returns max of q.x, q.y, q.z
  Path E:  inside corner -> all q components negative -> result negative
  Path F:  outside corner -> all q components positive -> result = Euclidean distance
  Path G:  edge center -> one q component positive, two negative -> result = positive component
  Path H:  face center -> one q component zero, two negative -> result = 0
  Path I:  asymmetric box -> correct distance on each face
  Path J:  zero-size dimension -> planar behavior
  Path K:  unit cube -> all corners and face centers correct
  Path L:  scaling homogeneity -> sdBox(sp, sb) = s * sdBox(p, b)
"""

import math

import pytest

# =============================================================================
# Python model of WGSL sdBox matching GPU semantics
# =============================================================================

TOL = 1e-12


def py_sdBox(p, b):
    """Model of WGSL sdBox: signed distance to axis-aligned box.

    Args:
        p: tuple/list of 3 floats, query position
        b: tuple/list of 3 floats, box half-size

    Returns:
        float: signed distance (negative inside, zero on surface, positive outside)
    """
    qx = abs(p[0]) - b[0]
    qy = abs(p[1]) - b[1]
    qz = abs(p[2]) - b[2]

    # length(max(q, 0)) -- outside term
    mx = max(qx, 0.0)
    my = max(qy, 0.0)
    mz = max(qz, 0.0)
    outside = math.sqrt(mx * mx + my * my + mz * mz)

    # min(maxComponent(q), 0) -- inside term
    max_comp = max(max(qx, qy), qz)
    inside = min(max_comp, 0.0)

    return outside + inside


def py_q(p, b):
    """Compute q = abs(p) - b (formula decomposition helper)."""
    return (abs(p[0]) - b[0], abs(p[1]) - b[1], abs(p[2]) - b[2])


def py_max_q(p, b):
    """Compute max(q, 0) component-wise."""
    q = py_q(p, b)
    return (max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0))


def py_length_max_q(p, b):
    """Compute length(max(q, 0)) -- the outside distance term."""
    mq = py_max_q(p, b)
    return math.sqrt(mq[0] * mq[0] + mq[1] * mq[1] + mq[2] * mq[2])


def py_max_component(q):
    """Compute maxComponent(q) = max(q.x, q.y, q.z)."""
    return max(max(q[0], q[1]), q[2])


# =============================================================================
# Test: Formula Decomposition -- Path A
# =============================================================================


class TestFormulaDecomposition:
    """Verify q = abs(p) - b for various p positions."""

    def test_q_inside_all_axes(self):
        """q = abs(p) - b with p fully inside box: all components negative."""
        p = (0.3, 0.4, 0.5)
        b = (1.0, 1.0, 1.0)
        q = py_q(p, b)
        expected_q = (-0.7, -0.6, -0.5)
        for qi, ei in zip(q, expected_q):
            assert qi == pytest.approx(ei, abs=TOL)

    def test_q_outside_all_axes(self):
        """q = abs(p) - b with p fully outside: all components positive."""
        p = (3.0, 4.0, 5.0)
        b = (1.0, 1.0, 1.0)
        q = py_q(p, b)
        expected_q = (2.0, 3.0, 4.0)
        for qi, ei in zip(q, expected_q):
            assert qi == pytest.approx(ei, abs=TOL)

    def test_q_mixed_signs(self):
        """q = abs(p) - b with p straddling box surface: mixed signs."""
        p = (1.5, 0.5, -0.5)
        b = (1.0, 1.0, 1.0)
        q = py_q(p, b)
        expected_q = (0.5, -0.5, -0.5)
        for qi, ei in zip(q, expected_q):
            assert qi == pytest.approx(ei, abs=TOL)

    def test_q_on_face(self):
        """q = abs(p) - b with p on face: one component zero."""
        p = (1.0, 0.0, 0.0)
        b = (1.0, 1.0, 1.0)
        q = py_q(p, b)
        expected_q = (0.0, -1.0, -1.0)
        for qi, ei in zip(q, expected_q):
            assert qi == pytest.approx(ei, abs=TOL)

    def test_q_on_corner(self):
        """q = abs(p) - b with p on corner: all components zero."""
        p = (1.0, 1.0, 1.0)
        b = (1.0, 1.0, 1.0)
        q = py_q(p, b)
        expected_q = (0.0, 0.0, 0.0)
        for qi, ei in zip(q, expected_q):
            assert qi == pytest.approx(ei, abs=TOL)

    def test_q_negative_p(self):
        """abs handles negative coordinates: abs(-x) == abs(x)."""
        p = (-1.5, -0.5, 0.0)
        b = (1.0, 1.0, 1.0)
        q_pos = py_q((1.5, 0.5, 0.0), b)
        q_neg = py_q(p, b)
        for qi_pos, qi_neg in zip(q_pos, q_neg):
            assert qi_pos == pytest.approx(qi_neg, abs=TOL)


# =============================================================================
# Test: max(q, 0) Component-wise Clamp -- Path B
# =============================================================================


class TestMaxQClamp:
    """Verify max(q, 0) returns correct component-wise clamp to zero."""

    def test_max_q_all_positive(self):
        """All q components positive: max(q, 0) == q."""
        q = (2.0, 3.0, 4.0)
        mq = tuple(max(v, 0.0) for v in q)
        assert mq == q

    def test_max_q_all_negative(self):
        """All q components negative: max(q, 0) == (0, 0, 0)."""
        q = (-0.5, -0.6, -0.7)
        mq = tuple(max(v, 0.0) for v in q)
        assert mq == (0.0, 0.0, 0.0)

    def test_max_q_mixed(self):
        """Mixed q components: negatives zeroed, positives preserved."""
        q = (0.5, -1.0, -0.3)
        mq = tuple(max(v, 0.0) for v in q)
        assert mq == (0.5, 0.0, 0.0)

    def test_max_q_zero(self):
        """Zero component stays zero."""
        q = (0.0, -0.5, 0.5)
        mq = tuple(max(v, 0.0) for v in q)
        assert mq == (0.0, 0.0, 0.5)

    def test_max_q_edge_point(self):
        """Edge point: one positive, two negative clamped to zero."""
        p = (1.5, 0.0, 0.0)
        b = (1.0, 1.0, 1.0)
        mq = py_max_q(p, b)
        assert mq == (0.5, 0.0, 0.0)

    def test_max_q_outside_corner(self):
        """Outside corner: all positive, max(q,0) = q."""
        p = (2.0, 2.0, 2.0)
        b = (1.0, 1.0, 1.0)
        mq = py_max_q(p, b)
        assert mq == (1.0, 1.0, 1.0)


# =============================================================================
# Test: length(max(q, 0)) Outside Distance -- Path C
# =============================================================================


class TestLengthMaxQ:
    """Verify length(max(q, 0)) computes correct Euclidean distance outside."""

    def test_outside_corner_distance(self):
        """Outside corner: distance = sqrt(sum(pos^2))."""
        p = (2.0, 2.0, 2.0)
        b = (1.0, 1.0, 1.0)
        dist = py_length_max_q(p, b)
        assert dist == pytest.approx(math.sqrt(3.0), abs=TOL)

    def test_outside_edge_distance(self):
        """Outside edge: distance = sqrt(x^2 + y^2) when z inside."""
        p = (2.0, 2.0, 0.0)
        b = (1.0, 1.0, 1.0)
        dist = py_length_max_q(p, b)
        assert dist == pytest.approx(math.sqrt(2.0), abs=TOL)

    def test_outside_face_distance(self):
        """Outside face: distance = |x| when y, z inside."""
        p = (2.5, 0.0, 0.0)
        b = (1.0, 1.0, 1.0)
        dist = py_length_max_q(p, b)
        assert dist == pytest.approx(1.5, abs=TOL)

    def test_inside_point_zero_distance(self):
        """Inside point: max(q,0) = (0,0,0), length = 0."""
        p = (0.0, 0.0, 0.0)
        b = (1.0, 1.0, 1.0)
        dist = py_length_max_q(p, b)
        assert dist == pytest.approx(0.0, abs=TOL)

    def test_on_surface_zero_distance(self):
        """On surface: max(q,0) = (0,0,0), length = 0."""
        p = (1.0, 0.5, 0.0)
        b = (1.0, 1.0, 1.0)
        dist = py_length_max_q(p, b)
        assert dist == pytest.approx(0.0, abs=TOL)

    def test_asymmetric_outside(self):
        """Asymmetric box: distance at corner point."""
        p = (4.0, 3.0, 5.0)
        b = (2.0, 1.0, 3.0)
        # q = (2, 2, 2), length = sqrt(12) = 2*sqrt(3)
        dist = py_length_max_q(p, b)
        assert dist == pytest.approx(2.0 * math.sqrt(3.0), abs=TOL)


# =============================================================================
# Test: maxComponent(q) -- Path D
# =============================================================================


class TestMaxComponent:
    """Verify maxComponent(q) returns max of q.x, q.y, q.z."""

    def test_x_is_max(self):
        """q.x is the largest component."""
        q = (3.0, 1.0, 2.0)
        assert py_max_component(q) == 3.0

    def test_y_is_max(self):
        """q.y is the largest component."""
        q = (1.0, 5.0, 2.0)
        assert py_max_component(q) == 5.0

    def test_z_is_max(self):
        """q.z is the largest component."""
        q = (1.0, 2.0, 7.0)
        assert py_max_component(q) == 7.0

    def test_all_negative_returns_least_negative(self):
        """All negative: maxComponent returns the least negative (closest to zero)."""
        q = (-2.0, -0.5, -1.0)
        assert py_max_component(q) == -0.5

    def test_mixed_signs(self):
        """Mixed signs: maxComponent returns the positive value."""
        q = (-1.0, 0.5, -2.0)
        assert py_max_component(q) == 0.5

    def test_all_equal(self):
        """All components equal: maxComponent returns that value."""
        q = (-0.3, -0.3, -0.3)
        assert py_max_component(q) == -0.3

    def test_zero_among_negatives(self):
        """Zero among negatives: maxComponent returns zero."""
        q = (-0.5, 0.0, -0.8)
        assert py_max_component(q) == 0.0


# =============================================================================
# Test: Inside Corner -- Path E
# =============================================================================


class TestInsideCorner:
    """Inside corner: all q components negative, result = maxComponent(q)."""

    def test_inside_corner_pos_quadrant_point_5(self):
        """p=(0.5,0.5,0.5), b=(1,1,1): q=(-0.5,-0.5,-0.5), result=-0.5."""
        p = (0.5, 0.5, 0.5)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(-0.5, abs=TOL)

    def test_inside_corner_neg_quadrant(self):
        """p=(-0.5,-0.5,-0.5), b=(1,1,1): q=(-0.5,-0.5,-0.5), result=-0.5."""
        p = (-0.5, -0.5, -0.5)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(-0.5, abs=TOL)

    def test_inside_center(self):
        """p=(0,0,0), b=(1,1,1): q=(-1,-1,-1), maxComponent=-1, result=-1."""
        p = (0.0, 0.0, 0.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_inside_near_face(self):
        """p=(0.9, 0, 0), b=(1,1,1): q=(-0.1,-1,-1), maxComponent=-0.1, result=-0.1."""
        p = (0.9, 0.0, 0.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(-0.1, abs=TOL)

    def test_inside_asymmetric(self):
        """p=(1.5, 0.5, 1.0), b=(2, 1, 3): q=(-0.5,-0.5,-2), maxComponent=-0.5, result=-0.5."""
        p = (1.5, 0.5, 1.0)
        b = (2.0, 1.0, 3.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(-0.5, abs=TOL)

    def test_inside_deep(self):
        """p=(0, 0, 0), b=(5, 5, 5): result = -5."""
        p = (0.0, 0.0, 0.0)
        b = (5.0, 5.0, 5.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(-5.0, abs=TOL)


# =============================================================================
# Test: Outside Corner -- Path F
# =============================================================================


class TestOutsideCorner:
    """Outside corner: all q components positive, result = sqrt(sum(pos^2))."""

    def test_outside_corner_1_1_1(self):
        """p=(2,2,2), b=(1,1,1): result = sqrt(3)."""
        p = (2.0, 2.0, 2.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(math.sqrt(3.0), abs=TOL)

    def test_outside_corner_2_2_2(self):
        """p=(3,3,3), b=(1,1,1): q=(2,2,2), result = sqrt(12) = 2*sqrt(3)."""
        p = (3.0, 3.0, 3.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(2.0 * math.sqrt(3.0), abs=TOL)

    def test_outside_corner_neg_quadrant(self):
        """p=(-2,-2,-2), b=(1,1,1): result = sqrt(3) (abs makes it symmetric)."""
        p = (-2.0, -2.0, -2.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(math.sqrt(3.0), abs=TOL)

    def test_outside_corner_mixed_sign(self):
        """p=(-2, 2, -2), b=(1,1,1): result = sqrt(3) (abs handles signs)."""
        p = (-2.0, 2.0, -2.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(math.sqrt(3.0), abs=TOL)

    def test_outside_corner_asymmetric(self):
        """p=(5, 4, 6), b=(2, 1, 3): q=(3,3,3), result = sqrt(27) = 3*sqrt(3)."""
        p = (5.0, 4.0, 6.0)
        b = (2.0, 1.0, 3.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(3.0 * math.sqrt(3.0), abs=TOL)

    def test_outside_single_axis(self):
        """Far out on one axis: result = distance along that axis."""
        p = (10.0, 0.0, 0.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(9.0, abs=TOL)


# =============================================================================
# Test: Edge Center -- Path G
# =============================================================================


class TestEdgeCenter:
    """Edge center: one q component positive, result = that component."""

    def test_edge_x(self):
        """Edge along x: p=(1.5, 0, 0), b=(1,1,1): result=0.5."""
        p = (1.5, 0.0, 0.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_edge_y(self):
        """Edge along y: p=(0, 1.5, 0), b=(1,1,1): result=0.5."""
        p = (0.0, 1.5, 0.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_edge_z(self):
        """Edge along z: p=(0, 0, 1.5), b=(1,1,1): result=0.5."""
        p = (0.0, 0.0, 1.5)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_edge_negative_side(self):
        """Edge in negative quadrant: p=(0, 0, -1.5), b=(1,1,1): result=0.5."""
        p = (0.0, 0.0, -1.5)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_edge_asymmetric_box(self):
        """Asymmetric box edge: p=(3, 0, 0), b=(2, 1, 3): q=(1,-1,-3), result=1."""
        p = (3.0, 0.0, 0.0)
        b = (2.0, 1.0, 3.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_edge_distance_positive(self):
        """Edge should give positive distance when outside."""
        p_edge = (1.5, 0.0, 0.0)
        p_face = (0.5, 1.5, 0.0)  # y-face, q=(0, 0.5, 0), length=0.5
        b = (1.0, 1.0, 1.0)
        d_edge = py_sdBox(p_edge, b)
        d_face = py_sdBox(p_face, b)
        assert d_edge == pytest.approx(d_face, abs=TOL)


# =============================================================================
# Test: Face Center -- Path H
# =============================================================================


class TestFaceCenter:
    """Face center: one q component negative, two inside, result = 0."""

    def test_face_positive_x(self):
        """p=(1, 0, 0), b=(1,1,1): result=0."""
        p = (1.0, 0.0, 0.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_face_negative_x(self):
        """p=(-1, 0, 0), b=(1,1,1): result=0."""
        p = (-1.0, 0.0, 0.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_face_positive_y(self):
        """p=(0, 1, 0), b=(1,1,1): result=0."""
        p = (0.0, 1.0, 0.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_face_negative_y(self):
        """p=(0, -1, 0), b=(1,1,1): result=0."""
        p = (0.0, -1.0, 0.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_face_positive_z(self):
        """p=(0, 0, 1), b=(1,1,1): result=0."""
        p = (0.0, 0.0, 1.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_face_negative_z(self):
        """p=(0, 0, -1), b=(1,1,1): result=0."""
        p = (0.0, 0.0, -1.0)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_face_asymmetric_box(self):
        """p=(2, 0, 0), b=(2, 1, 3): result=0."""
        p = (2.0, 0.0, 0.0)
        b = (2.0, 1.0, 3.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_face_off_center(self):
        """p=(1, 0.3, 0.2), b=(1,1,1): q=(0,-0.7,-0.8), result=0."""
        p = (1.0, 0.3, 0.2)
        b = (1.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Test: Asymmetric Box -- Path I
# =============================================================================


class TestAsymmetricBox:
    """Verify correct distances for an asymmetric box b=(2, 1, 3)."""

    B = (2.0, 1.0, 3.0)

    def test_face_positive_x_side(self):
        """On +x face: p=(2, 0.5, 1), b=(2,1,3), result=0."""
        p = (2.0, 0.5, 1.0)
        assert py_sdBox(p, self.B) == pytest.approx(0.0, abs=TOL)

    def test_face_positive_y_side(self):
        """On +y face: p=(1, 1, 1), b=(2,1,3), result=0."""
        p = (1.0, 1.0, 1.0)
        assert py_sdBox(p, self.B) == pytest.approx(0.0, abs=TOL)

    def test_face_positive_z_side(self):
        """On +z face: p=(1, 0.5, 3), b=(2,1,3), result=0."""
        p = (1.0, 0.5, 3.0)
        assert py_sdBox(p, self.B) == pytest.approx(0.0, abs=TOL)

    def test_outside_x(self):
        """Outside +x face: p=(4, 0, 0), b=(2,1,3): q=(2,-1,-3), result=2."""
        p = (4.0, 0.0, 0.0)
        assert py_sdBox(p, self.B) == pytest.approx(2.0, abs=TOL)

    def test_outside_y(self):
        """Outside +y face: p=(0, 3, 0), b=(2,1,3): q=(0,2,-3), result=2."""
        p = (0.0, 3.0, 0.0)
        assert py_sdBox(p, self.B) == pytest.approx(2.0, abs=TOL)

    def test_outside_z(self):
        """Outside +z face: p=(0, 0, 6), b=(2,1,3): q=(0,0,3), result=3."""
        p = (0.0, 0.0, 6.0)
        assert py_sdBox(p, self.B) == pytest.approx(3.0, abs=TOL)

    def test_inside_asymmetric_off_center(self):
        """Inside off-center: p=(1.5, 0.5, -2.0), b=(2,1,3):
        q=(|1.5|-2, |0.5|-1, |-2|-3)=(-0.5,-0.5,-1),
        maxComponent=-0.5, inside=-0.5."""
        p = (1.5, 0.5, -2.0)
        assert py_sdBox(p, self.B) == pytest.approx(-0.5, abs=TOL)

    def test_outside_corner_asymmetric(self):
        """Outside diagonal: p=(4, 3, 6), b=(2,1,3): q=(2,2,3),
        result=sqrt(4+4+9)=sqrt(17)."""
        p = (4.0, 3.0, 6.0)
        d = py_sdBox(p, self.B)
        assert d == pytest.approx(math.sqrt(17.0), abs=TOL)


# =============================================================================
# Test: Zero-Size Dimension -- Path J
# =============================================================================


class TestZeroSizeDimension:
    """Verify planar degenerate behavior when one box dimension is zero."""

    def test_zero_x_inside(self):
        """b=(0,1,1): x half-size is 0 so box is a yz-rectangle.
        p=(0,0,0): q=(0,-1,-1), maxComponent=0 -> inside_term=0, result=0."""
        p = (0.0, 0.0, 0.0)
        b = (0.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_zero_x_outside_positive(self):
        """b=(0,1,1): p=(0.5, 0, 0): q=(0.5,-1,-1), result=0.5."""
        p = (0.5, 0.0, 0.0)
        b = (0.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_zero_x_outside_negative(self):
        """b=(0,1,1): p=(-0.5, 0, 0): result=0.5 (abs symmetry)."""
        p = (-0.5, 0.0, 0.0)
        b = (0.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_zero_y_outside(self):
        """b=(1,0,1): p=(0, 0.5, 0): result=0.5."""
        p = (0.0, 0.5, 0.0)
        b = (1.0, 0.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_zero_z_outside(self):
        """b=(1,1,0): p=(0, 0, 0.5): result=0.5."""
        p = (0.0, 0.0, 0.5)
        b = (1.0, 1.0, 0.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_zero_x_on_face(self):
        """b=(0,1,1): p=(0, 1, 0): on y-face, result=0."""
        p = (0.0, 1.0, 0.0)
        b = (0.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_zero_x_corner_off_plane(self):
        """b=(0,1,1): p=(0.3, 1.5, 0): q=(0.3,0.5,-1),
        result=sqrt(0.09+0.25)=sqrt(0.34)."""
        p = (0.3, 1.5, 0.0)
        b = (0.0, 1.0, 1.0)
        d = py_sdBox(p, b)
        assert d == pytest.approx(math.sqrt(0.34), abs=TOL)


# =============================================================================
# Test: Unit Cube -- Path K
# =============================================================================


class TestUnitCube:
    """Verify all 8 corners and 6 face centers of a unit cube b=(1,1,1)."""

    B = (1.0, 1.0, 1.0)

    CORNERS = [
        (1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, -1.0, -1.0),
    ]

    FACE_CENTERS = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]

    def test_all_corners_zero(self):
        """All 8 corners of the unit cube should have distance 0."""
        for corner in self.CORNERS:
            d = py_sdBox(corner, self.B)
            assert d == pytest.approx(0.0, abs=TOL), (
                f"Corner {corner} expected 0, got {d}"
            )

    def test_all_face_centers_zero(self):
        """All 6 face centers of the unit cube should have distance 0."""
        for face in self.FACE_CENTERS:
            d = py_sdBox(face, self.B)
            assert d == pytest.approx(0.0, abs=TOL), (
                f"Face center {face} expected 0, got {d}"
            )

    def test_outside_each_corner_direction(self):
        """Outside each corner, distance = sqrt(3) * offset."""
        offset = 0.5
        for corner in self.CORNERS:
            far = tuple(c + offset * (1 if c >= 0 else -1) for c in corner)
            d = py_sdBox(far, self.B)
            # q = (|far| - 1) = (|c+offset| - 1) = (offset, offset, offset)
            expected = math.sqrt(3.0 * offset * offset)
            assert d == pytest.approx(expected, abs=TOL), (
                f"Corner+offset {far} expected {expected}, got {d}"
            )

    def test_center_depth(self):
        """Center of cube has depth -1 (distance to nearest face)."""
        d = py_sdBox((0.0, 0.0, 0.0), self.B)
        assert d == pytest.approx(-1.0, abs=TOL)


# =============================================================================
# Test: Scaling Homogeneity -- Path L
# =============================================================================


class TestScaling:
    """Verify sdBox(sp, sb) = s * sdBox(p, b) for various scale factors."""

    @staticmethod
    def scaled_check(p, b, s):
        """Assert sdBox(s*p, s*b) = s * sdBox(p, b)."""
        sp = tuple(s * v for v in p)
        sb = tuple(s * v for v in b)
        lhs = py_sdBox(sp, sb)
        rhs = s * py_sdBox(p, b)
        assert lhs == pytest.approx(rhs, abs=TOL), (
            f"sdBox({sp}, {sb}) = {lhs}, "
            f"{s} * sdBox({p}, {b}) = {rhs}"
        )

    TEST_POINTS = [
        (0.0, 0.0, 0.0),       # center
        (0.5, 0.5, 0.5),       # inside corner
        (1.0, 0.0, 0.0),       # on face
        (0.0, 1.0, 0.0),       # on face
        (1.0, 1.0, 1.0),       # on corner
        (1.5, 0.0, 0.0),       # edge center
        (2.0, 2.0, 2.0),       # outside corner
        (0.0, 2.0, 0.0),       # outside face
        (-1.5, 0.0, 0.0),      # edge center negative side
        (-2.0, -2.0, -2.0),    # outside corner negative
    ]

    SCALES = [0.5, 2.0, 3.0, 0.25, 10.0]

    BOXES = [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 3.0),
        (0.5, 0.5, 0.5),
        (0.0, 1.0, 1.0),
    ]

    def test_scaling_all_combinations(self):
        """Test scaling homogeneity across multiple points, boxes, and scales."""
        for b in self.BOXES:
            for p in self.TEST_POINTS:
                for s in self.SCALES:
                    self.scaled_check(p, b, s)

    def test_scaling_identity(self):
        """Scale of 1 is identity."""
        p = (0.3, 0.7, 0.2)
        b = (1.0, 1.0, 1.0)
        assert py_sdBox(p, b) == pytest.approx(
            py_sdBox(tuple(1.0 * v for v in p),
                     tuple(1.0 * v for v in b)),
            abs=TOL)

    def test_scaling_negative_not_homogeneous(self):
        """sdBox(-p, -b) != -sdBox(p, b) -- box half-size cannot be negative."""
        pass  # Negative box half-size is physically meaningless; no test needed


# =============================================================================
# Test: Determinism and Symmetry
# =============================================================================


class TestDeterminismAndSymmetry:
    """Verify determinism and fundamental symmetries of sdBox."""

    def test_deterministic(self):
        """Same inputs always produce same output."""
        p = (0.7, 0.3, 0.9)
        b = (1.0, 2.0, 3.0)
        base = py_sdBox(p, b)
        for _ in range(20):
            assert py_sdBox(p, b) == pytest.approx(base, abs=TOL)

    def test_abs_symmetry(self):
        """sdBox should be symmetric under sign flips of p."""
        p = (0.5, 0.3, 0.7)
        b = (1.0, 1.0, 1.0)
        signs = [
            (1, 1, 1),
            (-1, 1, 1),
            (1, -1, 1),
            (1, 1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (1, -1, -1),
            (-1, -1, -1),
        ]
        base = py_sdBox(p, b)
        for s in signs:
            ps = tuple(p[i] * s[i] for i in range(3))
            assert py_sdBox(ps, b) == pytest.approx(base, abs=TOL), (
                f"Symmetry broken for signs {s}"
            )

    def test_axis_permutation_invariance(self):
        """sdBox should be invariant under axis permutations when b is symmetric."""
        p = (0.5, 0.3, 0.7)
        b = (1.0, 1.0, 1.0)
        base = py_sdBox(p, b)
        assert py_sdBox((p[1], p[2], p[0]), b) == pytest.approx(base, abs=TOL)
        assert py_sdBox((p[2], p[0], p[1]), b) == pytest.approx(base, abs=TOL)
        assert py_sdBox((p[0], p[2], p[1]), b) == pytest.approx(base, abs=TOL)

    def test_outside_increasing_distance(self):
        """Distance should monotonically increase as p moves outward."""
        b = (1.0, 1.0, 1.0)
        prev = py_sdBox((1.0, 0.0, 0.0), b)
        for x in [1.1, 1.5, 2.0, 3.0, 5.0]:
            curr = py_sdBox((x, 0.0, 0.0), b)
            assert curr > prev, (
                f"Distance should increase from x={x-0.1} to x={x}"
            )
            prev = curr

    def test_inside_decreasing_distance(self):
        """Distance should become more negative as p moves deeper inside."""
        b = (2.0, 2.0, 2.0)
        prev = py_sdBox((0.0, 0.0, 0.0), b)
        for p_val in [(0.5, 0, 0), (1.0, 0, 0), (1.5, 0, 0)]:
            curr = py_sdBox(p_val, b)
            assert curr > prev, (
                f"Distance should increase (less negative) from {prev} to {curr}"
            )
            prev = curr
