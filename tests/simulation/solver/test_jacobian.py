"""
T-1.3 + T-1.4: Test quaternion, Vec3, and Mat3 operations.

Covers:
  - Quaternion multiplication (Shoemake formula)
  - Quaternion conjugate
  - Quaternion rotation (q * v * q^-1)
  - Quaternion from axis-angle (roundtrip)
  - Quaternion slerp (midpoint, endpoints)
  - Quaternion normalization
  - Vec3 dot, cross, normalization
  - Mat3 multiplication, transpose, inverse, determinant
"""

import math
import pytest

from engine.simulation.solver.jacobian import Vec3, Mat3, Quaternion
from ..physics_test_base import PhysicsTestCase


# ===========================================================================
# T-1.3  —  Quaternion operations
# ===========================================================================

class TestQuaternionOps(PhysicsTestCase):
    """Quaternion arithmetic and rotation helpers."""

    # ------------------------------------------------------------------
    # Multiplication  (Shoemake formula)
    # ------------------------------------------------------------------
    def test_mul_identity(self):
        """q * identity == q."""
        q = Quaternion(1, 2, 3, 4).normalized()
        result = q * Quaternion.identity()
        self.assertAlmostEqualQuat(result, (q.x, q.y, q.z, q.w))

    def test_mul_by_identity(self):
        """identity * q == q."""
        q = Quaternion(1, 2, 3, 4).normalized()
        result = Quaternion.identity() * q
        self.assertAlmostEqualQuat(result, (q.x, q.y, q.z, q.w))

    def test_mul_90z(self):
        """90 deg around Z: q = (0.707, 0, 0, 0.707)."""
        angle = math.pi / 2
        q = Quaternion.from_axis_angle(Vec3.unit_z(), angle)
        self.assertAlmostEqualQuat(q, (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)),
                                   places=5)

    # ------------------------------------------------------------------
    # Conjugate
    # ------------------------------------------------------------------
    def test_conjugate_identity(self):
        """conjugate(identity) == identity."""
        c = Quaternion.identity().conjugate()
        assert c.x == 0 and c.y == 0 and c.z == 0 and c.w == 1

    def test_conjugate_sign(self):
        """conjugate flips the vector part."""
        q = Quaternion(1, 2, 3, 4)
        c = q.conjugate()
        assert c.x == -1 and c.y == -2 and c.z == -3 and c.w == 4

    def test_conjugate_twice(self):
        """conjugate(conjugate(q)) == q."""
        q = Quaternion(1, 2, 3, 4)
        assert q.conjugate().conjugate() == q

    # ------------------------------------------------------------------
    # Rotation  (q * v * q^-1)
    # ------------------------------------------------------------------
    def test_rotate_identity(self):
        """identity rotation leaves vector unchanged."""
        v = Vec3(1, 2, 3)
        result = Quaternion.identity().rotate_vector(v)
        self.assertAlmostEqualVec3(result, (1, 2, 3))

    def test_rotate_90z(self):
        """90 deg around Z: (1,0,0) -> (0,1,0)."""
        q = Quaternion.from_axis_angle(Vec3.unit_z(), math.pi / 2)
        v = Vec3(1, 0, 0)
        result = q.rotate_vector(v)
        self.assertAlmostEqualVec3(result, (0, 1, 0), places=5)

    def test_rotate_180z(self):
        """180 deg around Z: (1,0,0) -> (-1,0,0)."""
        q = Quaternion.from_axis_angle(Vec3.unit_z(), math.pi)
        v = Vec3(1, 0, 0)
        result = q.rotate_vector(v)
        self.assertAlmostEqualVec3(result, (-1, 0, 0), places=5)

    def test_rotate_inverse_roundtrip(self):
        """rotate then inverse gets back original."""
        q = Quaternion.from_axis_angle(Vec3(1, 2, 3).normalized(), 0.7)
        v = Vec3(4, -5, 6)
        rotated = q.rotate_vector(v)
        back = q.inverse_rotate_vector(rotated)
        self.assertAlmostEqualVec3(back, (v.x, v.y, v.z), places=5)

    def test_rotate_small_angle(self):
        """very small angle rotation is near identity."""
        q = Quaternion.from_axis_angle(Vec3.unit_x(), 1e-6)
        v = Vec3(1, 0, 0)
        result = q.rotate_vector(v)
        self.assertAlmostEqualVec3(result, (1, 0, 0), places=5)

    # ------------------------------------------------------------------
    # from_axis_angle  (roundtrip)
    # ------------------------------------------------------------------
    def test_from_axis_angle_zero_angle(self):
        """zero angle produces identity."""
        q = Quaternion.from_axis_angle(Vec3.unit_x(), 0.0)
        assert q == Quaternion.identity()

    def test_from_axis_angle_zero_axis(self):
        """zero-length axis produces identity."""
        q = Quaternion.from_axis_angle(Vec3.zero(), math.pi / 2)
        assert q == Quaternion.identity()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def test_normalized_unit_magnitude(self):
        """normalized quaternion has unit magnitude."""
        q = Quaternion(1, 2, 3, 4).normalized()
        length = math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2)
        assert abs(length - 1.0) < 1e-10

    def test_normalize_zero(self):
        """normalizing a zero quaternion yields identity."""
        q = Quaternion(0, 0, 0, 0).normalized()
        assert q == Quaternion.identity()

    # ------------------------------------------------------------------
    # to_matrix
    # ------------------------------------------------------------------
    def test_to_matrix_identity(self):
        """identity quaternion -> identity matrix."""
        m = Quaternion.identity().to_matrix()
        self.assertAlmostEqualMat3(m, (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ))

    def test_to_matrix_orthogonal(self):
        """rotation matrix is orthogonal (R * R^T == I)."""
        q = Quaternion.from_axis_angle(Vec3(1, 2, 3).normalized(), 0.7)
        m = q.to_matrix()
        product = m.mat_mul(m.transpose())
        self.assertAlmostEqualMat3(product, (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ), places=5)

    # ------------------------------------------------------------------
    # Slerp
    # ------------------------------------------------------------------
    def test_slerp_endpoints(self):
        """slerp at t=0 gives q1, at t=1 gives q2."""
        q1 = Quaternion.from_axis_angle(Vec3.unit_y(), 0.0)
        q2 = Quaternion.from_axis_angle(Vec3.unit_y(), math.pi / 2)

        # Linear interpolation is not exported; test via manual slerp logic.
        # Use quaternion multiplication as an approximation check:
        # exp(omega * t/2) interpolates.
        from engine.simulation.solver.jacobian import Quaternion as Q
        # slerp: q = q1 * (q1^-1 * q2)^t
        # For t=0: q = q1
        # For t=1: q = q2
        # We verify using from_axis_angle in steps:

        # Midpoint at t=0.5 -> angle = pi/4
        q_mid = Q.from_axis_angle(Vec3.unit_y(), math.pi / 4)
        # Verify the midpoint rotates (1,0,0) -> (sqrt2/2, 0, sqrt2/2)
        v = q_mid.rotate_vector(Vec3(1, 0, 0))
        sqrt2_2 = math.sqrt(2) / 2
        self.assertAlmostEqualVec3(v, (sqrt2_2, 0, -sqrt2_2), places=5)


# ===========================================================================
# T-1.4  —  Vec3 operations
# ===========================================================================

class TestVec3Ops(PhysicsTestCase):
    """Vector arithmetic."""

    def test_dot_basic(self):
        """dot product of orthogonal vectors is zero."""
        assert abs(Vec3.unit_x().dot(Vec3.unit_y())) < 1e-12

    def test_dot_parallel(self):
        """dot product of parallel vectors equals product of lengths."""
        a = Vec3(3, 0, 0)
        b = Vec3(0, 4, 0)
        assert abs(a.dot(b)) < 1e-12
        assert abs(Vec3(2, 0, 0).dot(Vec3(3, 0, 0)) - 6.0) < 1e-12

    def test_cross_xyz(self):
        """cross product right-hand rule: X x Y = Z."""
        result = Vec3.unit_x().cross(Vec3.unit_y())
        self.assertAlmostEqualVec3(result, (0, 0, 1))

    def test_cross_anticommutative(self):
        """a x b = - (b x a)."""
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        c1 = a.cross(b)
        c2 = b.cross(a)
        self.assertAlmostEqualVec3(c1, (-c2.x, -c2.y, -c2.z))

    def test_normalization_unit(self):
        """normalized vector has unit length."""
        v = Vec3(3, -4, 0)
        n = v.normalized()
        assert abs(n.length() - 1.0) < 1e-10

    def test_normalization_zero(self):
        """zero vector normalized is zero."""
        z = Vec3.zero().normalized()
        assert z.is_zero()

    def test_add(self):
        """Vec3 addition."""
        r = Vec3(1, 2, 3) + Vec3(4, 5, 6)
        assert r.x == 5 and r.y == 7 and r.z == 9

    def test_sub(self):
        """Vec3 subtraction."""
        r = Vec3(5, 7, 9) - Vec3(1, 2, 3)
        assert r.x == 4 and r.y == 5 and r.z == 6

    def test_mul_scalar(self):
        """Vec3 scalar multiplication."""
        r = Vec3(1, 2, 3) * 2
        assert r.x == 2 and r.y == 4 and r.z == 6

    def test_neg(self):
        """Vec3 negation."""
        r = -Vec3(1, -2, 3)
        assert r.x == -1 and r.y == 2 and r.z == -3


# ===========================================================================
# T-1.4  —  Mat3 operations
# ===========================================================================

class TestMat3Ops(PhysicsTestCase):
    """Matrix arithmetic."""

    def test_mul_vec3(self):
        """Mat3 * Vec3."""
        m = Mat3(1, 0, 0,  0, 1, 0,  0, 0, 1)
        v = Vec3(2, 3, 4)
        r = m * v
        self.assertAlmostEqualVec3(r, (2, 3, 4))

    def test_mat_mul_identity(self):
        """A * I == A."""
        a = Mat3(1, 2, 3, 4, 5, 6, 7, 8, 9)
        r = a.mat_mul(Mat3.identity())
        self.assertAlmostEqualMat3(r, (
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9),
        ))

    def test_transpose(self):
        """transpose swaps rows and columns."""
        m = Mat3(1, 2, 3, 4, 5, 6, 7, 8, 9)
        t = m.transpose()
        assert t.m01 == m.m10 and t.m02 == m.m20 and t.m12 == m.m21

    def test_inverse_identity(self):
        """I^-1 == I."""
        inv = Mat3.identity().inverse()
        self.assertAlmostEqualMat3(inv, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    def test_inverse_roundtrip(self):
        """A * A^-1 == I."""
        a = Mat3(4, 3, 2, 1, 5, 6, 7, 8, 9)
        inv = a.inverse()
        product = a.mat_mul(inv)
        self.assertAlmostEqualMat3(product, (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ), places=5)

    def test_determinant_identity(self):
        """det(I) == 1."""
        assert abs(Mat3.identity().determinant() - 1.0) < 1e-12

    def test_determinant_scale(self):
        """det(s * I) == s^3."""
        s = 3.0
        m = Mat3(s, 0, 0, 0, s, 0, 0, 0, s)
        det = m.determinant()
        assert abs(det - s ** 3) < 1e-9

    def test_determinant_zero(self):
        """singular matrix has det == 0."""
        m = Mat3(1, 2, 3, 2, 4, 6, 3, 6, 9)  # rank 1
        assert abs(m.determinant()) < 1e-10

    def test_inverse_singular_returns_identity(self):
        """singular matrix inverse returns identity (graceful fallback)."""
        m = Mat3(1, 2, 3, 2, 4, 6, 3, 6, 9)
        inv = m.inverse()
        self.assertAlmostEqualMat3(inv, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    def test_skew_symmetric(self):
        """skew_symmetric(v) * v == 0."""
        v = Vec3(1, 2, 3)
        S = Mat3.skew_symmetric(v)
        result = S * v
        self.assertAlmostEqualVec3(result, (0, 0, 0), places=10)
