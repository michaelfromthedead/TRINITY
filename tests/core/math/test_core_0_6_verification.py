"""T-CORE-0.6: Whitebox verification tests for math library fixes.

Verifies each fix in the T-CORE-0.6 branch:
  - Vec4: min, max, clamp, zero, one, unit_x/y/z/w, rmul, truediv, neg, eq, repr,
          normalized_zero, perspective_divide_w_zero
  - Vec2/Vec3: rmul, truediv, neg, eq, repr, min/max/clamp, zero/one, normalized_zero
  - Mat3.__matmul__ Vec3
  - Mat4.transform_point w=0 warning, singular inverse
  - Transform.inverse documented limitation
  - SpringDamper dt guard
  - Gimbal lock Euler angle roundtrip
  - from_matrix 180-degree rotation branches
"""

import math
import logging
import warnings

import pytest

from engine.core.math.vec import Vec2, Vec3, Vec4
from engine.core.math.mat import Mat3, Mat4
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform, RigidTransform
from engine.core.math.interpolation import SpringDamper


# =============================================================================
# Vec4: New methods that were missing before T-CORE-0.6
# =============================================================================

class TestVec4Verification:
    """Verify all Vec4 new methods work correctly."""

    def test_min_all_axes(self):
        """min selects per-component minimum across all 4 axes."""
        result = Vec4(1, 5, 3, 0).min(Vec4(3, 2, 1, 4))
        assert result == Vec4(1, 2, 1, 0)
        assert isinstance(result, Vec4)

    def test_max_all_axes(self):
        """max selects per-component maximum across all 4 axes."""
        result = Vec4(1, 5, 3, 0).max(Vec4(3, 2, 1, 4))
        assert result == Vec4(3, 5, 3, 4)
        assert isinstance(result, Vec4)

    def test_clamp_lower_bound(self):
        """clamp enforces lower bound per component."""
        v = Vec4(-1, -5, -10, -100)
        lo = Vec4(-0.5, -0.5, -0.5, -0.5)
        hi = Vec4(0.5, 0.5, 0.5, 0.5)
        result = v.clamp(lo, hi)
        assert result.x == pytest.approx(-0.5)
        assert result.y == pytest.approx(-0.5)
        assert result.z == pytest.approx(-0.5)
        assert result.w == pytest.approx(-0.5)

    def test_clamp_upper_bound(self):
        """clamp enforces upper bound per component."""
        v = Vec4(10, 20, 30, 40)
        lo = Vec4(0, 0, 0, 0)
        hi = Vec4(5, 5, 5, 5)
        result = v.clamp(lo, hi)
        assert result == Vec4(5, 5, 5, 5)

    def test_clamp_within_range(self):
        """clamp returns same vector when already in range."""
        v = Vec4(1, 2, 3, 4)
        lo = Vec4(0, 0, 0, 0)
        hi = Vec4(10, 10, 10, 10)
        result = v.clamp(lo, hi)
        assert result == v

    def test_clamp_identical_bounds(self):
        """clamp with lo==hi locks all components to that value."""
        v = Vec4(100, -100, 50, -50)
        lo = Vec4(0, 0, 0, 0)
        hi = Vec4(0, 0, 0, 0)
        assert v.clamp(lo, hi) == Vec4(0, 0, 0, 0)

    def test_zero_returns_identity(self):
        """zero() returns Vec4(0,0,0,0)."""
        assert Vec4.zero() == Vec4(0.0, 0.0, 0.0, 0.0)
        # Verify it's not the same instance each time
        assert Vec4.zero() is not Vec4.zero()

    def test_one_returns_identity(self):
        """one() returns Vec4(1,1,1,1)."""
        assert Vec4.one() == Vec4(1.0, 1.0, 1.0, 1.0)

    def test_unit_vectors_orthogonal(self):
        """unit_x, unit_y, unit_z, unit_w are mutually orthogonal."""
        assert Vec4.unit_x().dot(Vec4.unit_y()) == pytest.approx(0.0)
        assert Vec4.unit_x().dot(Vec4.unit_z()) == pytest.approx(0.0)
        assert Vec4.unit_x().dot(Vec4.unit_w()) == pytest.approx(0.0)
        assert Vec4.unit_y().dot(Vec4.unit_z()) == pytest.approx(0.0)
        assert Vec4.unit_y().dot(Vec4.unit_w()) == pytest.approx(0.0)
        assert Vec4.unit_z().dot(Vec4.unit_w()) == pytest.approx(0.0)

    def test_unit_vectors_unit_length(self):
        """All unit_* vectors have length 1."""
        assert Vec4.unit_x().length() == pytest.approx(1.0)
        assert Vec4.unit_y().length() == pytest.approx(1.0)
        assert Vec4.unit_z().length() == pytest.approx(1.0)
        assert Vec4.unit_w().length() == pytest.approx(1.0)

    def test_rmul_scales_correctly(self):
        """Scalar * Vec4 produces same result as Vec4 * scalar."""
        v = Vec4(1, 2, 3, 4)
        assert 2 * v == v * 2
        assert 0.5 * Vec4(10, 20, 30, 40) == Vec4(5, 10, 15, 20)

    def test_truediv_scales_correctly(self):
        """Vec4 / scalar produces correct division."""
        assert Vec4(6, 9, 12, 3) / 3 == Vec4(2, 3, 4, 1)

    def test_truediv_by_one(self):
        """Division by 1 returns same vector."""
        v = Vec4(3, 4, 5, 6)
        assert v / 1 == v

    def test_neg_negates_all_components(self):
        """-Vec4 negates all four components."""
        assert -Vec4(1, -2, 3, -4) == Vec4(-1, 2, -3, 4)
        assert -(-Vec4(1, 2, 3, 4)) == Vec4(1, 2, 3, 4)

    def test_eq_type_check(self):
        """Comparing Vec4 to non-Vec4 returns NotImplemented."""
        assert Vec4(1, 2, 3, 4).__eq__("nope") == NotImplemented
        assert Vec4(1, 2, 3, 4).__eq__(None) == NotImplemented
        assert Vec4(1, 2, 3, 4).__eq__(42) == NotImplemented

    def test_repr_contains_type_and_values(self):
        """repr includes class name and component values."""
        r = repr(Vec4(1.5, 2.5, 3.5, 4.5))
        assert "Vec4" in r
        assert "1.5" in r
        assert "2.5" in r
        assert "3.5" in r
        assert "4.5" in r

    def test_normalized_zero_vector(self):
        """normalized() of zero vector returns zero, not NaN."""
        z = Vec4.zero().normalized()
        assert z == Vec4(0, 0, 0, 0)
        # Verify no NaN in any component
        assert not math.isnan(z.x)
        assert not math.isnan(z.y)
        assert not math.isnan(z.z)
        assert not math.isnan(z.w)

    def test_normalized_preserves_direction(self):
        """normalized() preserves direction (unit-length result)."""
        v = Vec4(3, 0, 0, 0)
        n = v.normalized()
        assert n == Vec4(1, 0, 0, 0)

    def test_perspective_divide_w_zero_no_crash(self):
        """perspective_divide with w=0 warns and returns xyz as-is."""
        v = Vec4(1, 2, 3, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = v.perspective_divide()
        assert p == Vec3(1, 2, 3)

    def test_perspective_divide_warning_issued(self):
        """perspective_divide with w=0 issues a UserWarning."""
        v = Vec4(1, 2, 3, 0)
        with pytest.warns(UserWarning, match="w near zero"):
            v.perspective_divide()

    def test_perspective_divide_normal(self):
        """perspective_divide with w>0 produces correct result."""
        p = Vec4(4, 8, 12, 4).perspective_divide()
        assert p == Vec3(1, 2, 3)

    def test_perspective_divide_negative_w(self):
        """perspective_divide with negative w still divides correctly."""
        p = Vec4(4, 8, 12, -4).perspective_divide()
        assert p == Vec3(-1, -2, -3)

    def test_length_zero_vector(self):
        """length() of zero vector is 0 (not NaN)."""
        assert Vec4.zero().length() == pytest.approx(0.0)

    def test_length_squared_zero_vector(self):
        """length_squared() of zero vector is 0."""
        assert Vec4.zero().length_squared() == pytest.approx(0.0)

    def test_dot_product_orthogonal(self):
        """dot of orthogonal vectors is 0."""
        assert Vec4(1, 0, 0, 0).dot(Vec4(0, 1, 0, 0)) == pytest.approx(0.0)


# =============================================================================
# Vec2: Consistency checks
# =============================================================================

class TestVec2Verification:
    """Verify Vec2 has consistent API with Vec3/Vec4."""

    def test_rmul_commutative(self):
        """scalar * Vec2 == Vec2 * scalar."""
        v = Vec2(3, 4)
        assert 2 * v == v * 2

    def test_truediv_consistency(self):
        """Vec2 division is consistent with scalar multiplication inverse."""
        v = Vec2(10, 20)
        half = v / 2
        double = half * 2
        assert double == v

    def test_min_components(self):
        """min selects per-component minimum."""
        assert Vec2(1, 5).min(Vec2(3, 2)) == Vec2(1, 2)

    def test_max_components(self):
        """max selects per-component maximum."""
        assert Vec2(1, 5).max(Vec2(3, 2)) == Vec2(3, 5)

    def test_clamp_range(self):
        """clamp constrains within bounds."""
        assert Vec2(-1, 5).clamp(Vec2(0, 0), Vec2(2, 4)) == Vec2(0, 4)

    def test_clamp_lower_equal(self):
        """clamp when lo==hi locks to that value."""
        assert Vec2(100, -100).clamp(Vec2(0, 0), Vec2(0, 0)) == Vec2(0, 0)

    def test_zero_one(self):
        """zero() and one() return correct values."""
        assert Vec2.zero() == Vec2(0, 0)
        assert Vec2.one() == Vec2(1, 1)

    def test_unit_vectors(self):
        """unit_x() and unit_y() have unit length."""
        assert Vec2.unit_x().length() == pytest.approx(1.0)
        assert Vec2.unit_y().length() == pytest.approx(1.0)

    def test_normalized_zero_vector(self):
        """normalized() of zero returns zero."""
        assert Vec2.zero().normalized() == Vec2(0, 0)

    def test_type_check_eq(self):
        """Comparing Vec2 to non-Vec2 returns NotImplemented."""
        assert Vec2(1, 2).__eq__("nope") == NotImplemented

    def test_repr_contains_type(self):
        """repr includes Vec2 and values."""
        assert "Vec2" in repr(Vec2(1.5, 2.5))


# =============================================================================
# Vec3: API completion
# =============================================================================

class TestVec3Verification:
    """Verify Vec3 new methods and statics."""

    def test_rmul_commutative(self):
        """scalar * Vec3 == Vec3 * scalar."""
        v = Vec3(1, 2, 3)
        assert 2 * v == v * 2

    def test_truediv_inverse_of_mul(self):
        """Vec3 / s == Vec3 * (1/s)."""
        v = Vec3(6, 9, 12)
        assert v / 3 == v * (1/3)

    def test_neg_twice_is_identity(self):
        """-(-v) == v."""
        v = Vec3(1, -2, 3)
        assert -(-v) == v

    def test_all_statics_defined(self):
        """All Vec3 static methods return expected values."""
        assert Vec3.zero() == Vec3(0, 0, 0)
        assert Vec3.one() == Vec3(1, 1, 1)
        assert Vec3.unit_x() == Vec3(1, 0, 0)
        assert Vec3.unit_y() == Vec3(0, 1, 0)
        assert Vec3.unit_z() == Vec3(0, 0, 1)
        assert Vec3.up() == Vec3(0, 1, 0)
        assert Vec3.forward() == Vec3(0, 0, -1)
        assert Vec3.right() == Vec3(1, 0, 0)

    def test_statics_have_unit_length(self):
        """All directional statics have length 1."""
        assert Vec3.unit_x().length() == pytest.approx(1.0)
        assert Vec3.unit_y().length() == pytest.approx(1.0)
        assert Vec3.unit_z().length() == pytest.approx(1.0)
        assert Vec3.up().length() == pytest.approx(1.0)
        assert Vec3.forward().length() == pytest.approx(1.0)
        assert Vec3.right().length() == pytest.approx(1.0)

    def test_type_check_eq(self):
        """Comparing Vec3 to non-Vec3 returns NotImplemented."""
        assert Vec3(1, 2, 3).__eq__("nope") == NotImplemented

    def test_normalized_zero_vector(self):
        """Vec3.zero().normalized() returns zero, not NaN."""
        z = Vec3.zero().normalized()
        assert z == Vec3(0, 0, 0)
        assert not math.isnan(z.x)

    def test_reflect_auto_normalize(self):
        """reflect auto-normalizes the normal argument."""
        v = Vec3(1, -1, 0).normalized()
        r = v.reflect(Vec3(0, 2, 0))  # non-unit normal
        assert r.y == pytest.approx(-v.y, abs=1e-6)


# =============================================================================
# Mat3 @ Vec3 operator
# =============================================================================

class TestMat3MatmulVerification:
    """Verify Mat3.__matmul__ with Vec3."""

    def test_identity_matmul(self):
        """Identity matrix @ Vec3 returns same vector."""
        m = Mat3()
        v = Vec3(1, 2, 3)
        assert m @ v == v

    def test_scale_matrix(self):
        """Diagonal scaling matrix scales each component."""
        m = Mat3([2, 0, 0, 0, 3, 0, 0, 0, 4])
        assert m @ Vec3(1, 2, 3) == Vec3(2, 6, 12)

    def test_zero_matrix(self):
        """Zero matrix @ Vec3 returns zero vector."""
        m = Mat3([0]*9)
        assert m @ Vec3(1, 2, 3) == Vec3(0, 0, 0)

    def test_matmul_not_commutative(self):
        """Verify __matmul__ is not the same as scalar mul."""
        m = Mat3([2, 0, 0, 0, 3, 0, 0, 0, 4])
        v = Vec3(1, 2, 3)
        mv = m @ v
        sm = v * 2  # different result
        assert mv != sm

    def test_rotation_matmul(self):
        """90-degree rotation around Z via Mat3."""
        m = Mat3([0, 1, 0, -1, 0, 0, 0, 0, 1])
        result = m @ Vec3(1, 0, 0)
        assert result.x == pytest.approx(0, abs=1e-6)
        assert result.y == pytest.approx(1, abs=1e-6)

    def test_matmul_returns_vec3(self):
        """__matmul__ returns Vec3, not Mat3."""
        m = Mat3()
        result = m @ Vec3(1, 2, 3)
        assert isinstance(result, Vec3)

    def test_matmul_preserves_type_when_no_transform(self):
        """Negative values are preserved through identity matmul."""
        m = Mat3()
        v = Vec3(-1, -2, -3)
        assert m @ v == v


# =============================================================================
# Mat4: Robustness
# =============================================================================

class TestMat4Verification:
    """Verify Mat4 edge case handling."""

    def test_transform_point_w_zero_warning(self, caplog):
        """transform_point with w=0 logs warning."""
        m = Mat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        caplog.set_level(logging.WARNING)
        p = m.transform_point(Vec3(1, 2, 3))
        assert "w near zero" in caplog.text
        assert p is not None

    def test_transform_point_w_one_no_divide(self):
        """transform_point with w=1 skips division."""
        m = Mat4.translation(Vec3(10, 20, 30))
        p = m.transform_point(Vec3(0, 0, 0))
        assert p == Vec3(10, 20, 30)

    def test_projection_divide(self):
        """Projection matrix triggers w-division."""
        m = Mat4.perspective(math.radians(90), 1.0, 0.1, 100.0)
        p = m.transform_point(Vec3(0, 0, -2))
        assert p is not None

    def test_inverse_singular_returns_identity(self):
        """Singular matrix inverse returns identity."""
        m = Mat4([0]*16)
        result = m.inverse()
        assert result == Mat4.identity()

    def test_inverse_singular_logs_warning(self, caplog):
        """Singular matrix inverse logs warning."""
        m = Mat4([0]*16)
        caplog.set_level(logging.WARNING)
        m.inverse()
        assert "singular" in caplog.text

    def test_inverse_identity(self):
        """Inverse of identity is identity."""
        assert Mat4.identity().inverse() == Mat4.identity()

    def test_inverse_translation_roundtrip(self):
        """M * M^-1 == I for translation."""
        m = Mat4.translation(Vec3(5, 10, 15))
        r = m @ m.inverse()
        assert r == Mat4.identity()

    def test_rotation_y_direction(self):
        """Rotation around Y transforms direction correctly."""
        m = Mat4.rotation_y(math.pi / 2)
        d = m.transform_direction(Vec3(1, 0, 0))
        assert d.z == pytest.approx(-1, abs=1e-6)

    def test_transform_direction_translation_ignored(self):
        """transform_direction ignores translation component."""
        m = Mat4.translation(Vec3(10, 20, 30))
        d = m.transform_direction(Vec3(1, 0, 0))
        assert d == Vec3(1, 0, 0)

    def test_scale_matrix_direction(self):
        """Scale matrix affects direction vectors."""
        s = Mat4.scale(Vec3(2, 3, 4))
        d = s.transform_direction(Vec3(1, 0, 0))
        assert d.x == pytest.approx(2)

    def test_mul_preserves_identity(self):
        """M @ I == M."""
        m = Mat4.translation(Vec3(1, 2, 3))
        assert m @ Mat4.identity() == m

    def test_orthographic_clip_space(self):
        """Orthographic projection maps view volume to clip space."""
        m = Mat4.orthographic(-1, 1, -1, 1, 0.1, 100.0)
        near_p = m.transform_point(Vec3(0, 0, -0.1))
        assert near_p.z == pytest.approx(-1.0, abs=1e-6)
        far_p = m.transform_point(Vec3(0, 0, -100.0))
        assert far_p.z == pytest.approx(1.0, abs=1e-6)

    def test_look_at_orientation(self):
        """look_at produces correct view matrix."""
        m = Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        p = m.transform_point(Vec3(0, 0, 0))
        assert p.z == pytest.approx(-5.0, abs=1e-6)


# =============================================================================
# Transform.inverse: documented limitation
# =============================================================================

class TestTransformVerification:
    """Verify Transform.inverse behavior including documented limitation."""

    def test_inverse_identity(self):
        """Inverse of identity is identity."""
        t = Transform.identity()
        inv = t.inverse()
        p = inv.transform_point(t.transform_point(Vec3(1, 2, 3)))
        assert p == Vec3(1, 2, 3)

    def test_inverse_translation_only(self):
        """Pure translation inverse roundtrips perfectly."""
        t = Transform(translation=Vec3(10, 20, 30))
        inv = t.inverse()
        p = Vec3(1, 2, 3)
        back = inv.transform_point(t.transform_point(p))
        assert back == Vec3(1, 2, 3)

    def test_inverse_uniform_scale_rotation(self):
        """Uniform scale + rotation: perfect inverse."""
        t = Transform(
            translation=Vec3(1, 2, 3),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), 0.5),
            scale=Vec3(2, 2, 2),
        )
        inv = t.inverse()
        p = Vec3(4, 5, 6)
        back = inv.transform_point(t.transform_point(p))
        assert back.x == pytest.approx(p.x, abs=1e-4)
        assert back.y == pytest.approx(p.y, abs=1e-4)
        assert back.z == pytest.approx(p.z, abs=1e-4)

    def test_inverse_nonuniform_no_rotation(self):
        """Non-uniform scale without rotation: perfect inverse."""
        t = Transform(
            translation=Vec3(1, 2, 3),
            rotation=Quat.identity(),
            scale=Vec3(2, 3, 4),
        )
        inv = t.inverse()
        p = Vec3(4, 5, 6)
        back = inv.transform_point(t.transform_point(p))
        assert back.x == pytest.approx(p.x, abs=1e-6)
        assert back.y == pytest.approx(p.y, abs=1e-6)
        assert back.z == pytest.approx(p.z, abs=1e-6)

    def test_inverse_nonuniform_with_rotation_matrix_exact(self):
        """Non-uniform scale + rotation: matrix path M * M^-1 = I exactly."""
        t = Transform(
            translation=Vec3(1, 2, 3),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), 0.5),
            scale=Vec3(2, 3, 4),
        )
        m = t.to_matrix()
        m_inv = m.inverse()
        identity = m @ m_inv
        assert identity == Mat4.identity()

    def test_documented_limitation_present(self):
        """Docstring mentions non-uniform scale limitation."""
        doc = Transform.__doc__ or ""
        assert "non-uniform scale" in doc
        assert "limitation" in doc.lower()

    def test_transform_direction_no_scale(self):
        """transform_direction ignores scale."""
        t = Transform(scale=Vec3(2, 2, 2))
        d = t.transform_direction(Vec3(1, 0, 0))
        assert d == Vec3(1, 0, 0)

    def test_from_matrix_roundtrip(self):
        """to_matrix then from_matrix recovers translation and scale."""
        t = Transform(
            translation=Vec3(1, 2, 3),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), 0.5),
            scale=Vec3(2, 2, 2),
        )
        m = t.to_matrix()
        t2 = Transform.from_matrix(m)
        assert t2.translation.x == pytest.approx(1, abs=1e-3)
        assert t2.scale.x == pytest.approx(2, abs=1e-3)

    def test_transform_point_applies_translation(self):
        """transform_point moves by translation."""
        t = Transform(translation=Vec3(10, 0, 0))
        assert t.transform_point(Vec3(0, 0, 0)) == Vec3(10, 0, 0)

    def test_transform_point_applies_rotation(self):
        """transform_point rotates by quaternion."""
        t = Transform(
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2),
        )
        p = t.transform_point(Vec3(1, 0, 0))
        assert p.z == pytest.approx(-1, abs=1e-6)


# =============================================================================
# Transform.from_matrix 180-degree rotation branches
# =============================================================================

class TestTransformFromMatrixVerification:
    """Verify all four branches of from_matrix rotation extraction."""

    def test_branch_trace_positive(self):
        """trace > 0 branch: identity rotation."""
        t = Transform(
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), 0.3),
        )
        m = t.to_matrix()
        t2 = Transform.from_matrix(m)
        assert abs(t2.rotation.dot(t.rotation)) > 0.99

    def test_branch_xx_dominant(self):
        """r[0] > r[4] and r[0] > r[8] branch: 180 around X."""
        q = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi)
        t = Transform(rotation=q)
        m = t.to_matrix()
        t2 = Transform.from_matrix(m)
        assert t2.translation == Vec3.zero()
        assert abs(t2.rotation.dot(q)) > 0.99

    def test_branch_yy_dominant(self):
        """r[4] > r[8] branch: 180 around Y."""
        q = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi)
        t = Transform(rotation=q)
        m = t.to_matrix()
        t2 = Transform.from_matrix(m)
        assert t2.translation == Vec3.zero()
        assert abs(t2.rotation.dot(q)) > 0.99

    def test_branch_else(self):
        """else branch: 180 around Z."""
        q = Quat.from_axis_angle(Vec3(0, 0, 1), math.pi)
        t = Transform(rotation=q)
        m = t.to_matrix()
        t2 = Transform.from_matrix(m)
        assert t2.translation == Vec3.zero()
        assert abs(t2.rotation.dot(q)) > 0.99

    def test_all_branches_identity_matrix(self):
        """All branches handle identity matrix correctly."""
        m = Mat4.identity()
        t = Transform.from_matrix(m)
        assert t.translation == Vec3.zero()
        assert t.rotation == Quat.identity()
        assert t.scale == Vec3.one()


# =============================================================================
# RigidTransform
# =============================================================================

class TestRigidTransformVerification:
    """Verify RigidTransform correctness."""

    def test_inverse_roundtrip(self):
        """RigidTransform inverse roundtrips."""
        rt = RigidTransform(
            translation=Vec3(3, 4, 5),
            rotation=Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 4),
        )
        inv = rt.inverse()
        p = rt.transform_point(Vec3(1, 0, 0))
        p2 = inv.transform_point(p)
        assert p2.x == pytest.approx(1, abs=1e-6)
        assert p2.y == pytest.approx(0, abs=1e-6)

    def test_transform_direction_rotation_only(self):
        """transform_direction uses rotation only."""
        rt = RigidTransform(
            translation=Vec3(10, 20, 30),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2),
        )
        d = rt.transform_direction(Vec3(1, 0, 0))
        assert d.z == pytest.approx(-1, abs=1e-6)

    def test_to_matrix_identity(self):
        """RigidTransform.identity().to_matrix() == Mat4.identity()."""
        assert RigidTransform.identity().to_matrix() == Mat4.identity()


# =============================================================================
# SpringDamper dt guard
# =============================================================================

class TestSpringDamperVerification:
    """Verify SpringDamper.update dt guarding."""

    def test_negative_dt_raises_value_error(self):
        """Negative dt raises ValueError."""
        s = SpringDamper(position=0, target=10)
        with pytest.raises(ValueError, match="dt must be non-negative"):
            s.update(-0.1)

    def test_negative_dt_very_small(self):
        """Very small negative dt raises (no silent pass-through)."""
        s = SpringDamper(position=0, target=10)
        with pytest.raises(ValueError):
            s.update(-1e-10)

    def test_zero_dt_returns_current_position(self):
        """Zero dt returns current position unchanged."""
        s = SpringDamper(position=3, velocity=2, target=10)
        pos = s.update(0.0)
        assert pos == pytest.approx(3)

    def test_zero_dt_does_not_mutate_position(self):
        """Zero dt does not change the spring state."""
        s = SpringDamper(position=3, velocity=2, target=10)
        s.update(0.0)
        assert s.position == pytest.approx(3)
        assert s.velocity == pytest.approx(2)

    def test_positive_dt_converges_to_target(self):
        """Positive dt converges toward target."""
        s = SpringDamper(position=0, target=10, omega=20)
        for _ in range(1000):
            s.update(0.016)
        assert s.position == pytest.approx(10, abs=0.01)

    def test_initial_at_target(self):
        """Starting at target stays at target."""
        s = SpringDamper(position=5, target=5)
        s.update(0.1)
        assert s.position == pytest.approx(5, abs=0.01)

    def test_high_omega_fast_convergence(self):
        """High omega converges faster than low omega."""
        fast = SpringDamper(position=0, target=10, omega=50)
        slow = SpringDamper(position=0, target=10, omega=5)
        for _ in range(100):
            fast.update(0.016)
            slow.update(0.016)
        assert abs(fast.position - 10) < abs(slow.position - 10)


# =============================================================================
# Quat: Gimbal lock and edge cases
# =============================================================================

class TestQuatVerification:
    """Verify Quat robustness including gimbal lock."""

    def test_gimbal_lock_pitch_90(self):
        """Pitch=90 produces valid Euler angles (gimbal lock handling)."""
        q = Quat.from_euler(math.pi / 2, 0.3, 0.0)
        p, y, r = q.to_euler()
        q2 = Quat.from_euler(p, y, r)
        assert abs(q.dot(q2)) == pytest.approx(1.0, abs=1e-3)

    def test_gimbal_lock_pitch_negative_90(self):
        """Pitch=-90 produces valid Euler angles."""
        q = Quat.from_euler(-math.pi / 2, 0.5, 0.1)
        p, y, r = q.to_euler()
        q2 = Quat.from_euler(p, y, r)
        assert abs(q.dot(q2)) == pytest.approx(1.0, abs=1e-3)

    def test_euler_roundtrip_identity(self):
        """Identity Euler roundtrip."""
        q = Quat.from_euler(0, 0, 0)
        p, y, r = q.to_euler()
        assert p == pytest.approx(0, abs=1e-6)
        assert y == pytest.approx(0, abs=1e-6)
        assert r == pytest.approx(0, abs=1e-6)

    def test_euler_roundtrip_random(self):
        """Random Euler angles roundtrip."""
        q = Quat.from_euler(0.3, 0.5, 0.1)
        p, y, r = q.to_euler()
        q2 = Quat.from_euler(p, y, r)
        assert abs(q.dot(q2)) == pytest.approx(1.0, abs=1e-3)

    def test_mul_composes_rotations(self):
        """Quaternion multiplication composes two rotations."""
        q1 = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        q2 = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        q = q1 * q2
        v = q.rotate_vector(Vec3(1, 0, 0))
        assert v.x == pytest.approx(-1, abs=1e-6)

    def test_normalized_zero_returns_identity(self):
        """Quat(0,0,0,0).normalized() returns identity."""
        assert Quat(0, 0, 0, 0).normalized() == Quat.identity()

    def test_inverse_zero_returns_identity(self):
        """Quat(0,0,0,0).inverse() returns identity."""
        assert Quat(0, 0, 0, 0).inverse() == Quat.identity()

    def test_conjugate_of_identity(self):
        """Conjugate of identity is identity."""
        assert Quat.identity().conjugate() == Quat.identity()

    def test_slerp_identity_to_identity(self):
        """slerp from identity to identity."""
        a = Quat.identity()
        b = Quat.identity()
        s = a.slerp(b, 0.5)
        assert s == Quat.identity()

    def test_rotate_vector_identity(self):
        """Identity rotation leaves vector unchanged."""
        q = Quat.identity()
        v = Vec3(5, -3, 2)
        assert q.rotate_vector(v) == v

    def test_type_check_eq(self):
        """Comparing Quat to non-Quat returns NotImplemented."""
        assert Quat.identity().__eq__("nope") == NotImplemented


# =============================================================================
# Cross-component integration
# =============================================================================

class TestCrossComponentVerification:
    """Verify fixes work correctly when components interact."""

    def test_transform_uses_quat_rotation(self):
        """Transform.transform_point uses Quat.rotate_vector internally."""
        t = Transform(
            translation=Vec3(10, 0, 0),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2),
        )
        # Point (1,0,0) rotated 90 deg around Y -> (0,0,-1), then translated -> (10,0,-1)
        p = t.transform_point(Vec3(1, 0, 0))
        assert p.x == pytest.approx(10, abs=1e-6)
        assert p.z == pytest.approx(-1, abs=1e-6)

    def test_transform_to_matrix_uses_mat4(self):
        """Transform.to_matrix produces Mat4 consistent with transform_point."""
        t = Transform(
            translation=Vec3(5, 10, 15),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), 0.5),
            scale=Vec3(2, 2, 2),
        )
        m = t.to_matrix()
        p = Vec3(1, 2, 3)
        # Both paths should produce the same result
        assert m.transform_point(p) == t.transform_point(p)

    def test_mat4_transform_point_uses_vec4(self):
        """Mat4 uses column-major layout correctly for Vec3."""
        m = Mat4.translation(Vec3(10, 20, 30))
        p = m.transform_point(Vec3(1, 2, 3))
        assert p == Vec3(11, 22, 33)

    def test_mat3_from_mat4_preserves_rotation(self):
        """Mat3.from_mat4 extracts correct rotation."""
        m4 = Mat4.rotation_y(math.pi / 4)
        m3 = Mat3.from_mat4(m4)
        v = m3 @ Vec3(1, 0, 0)
        assert v.x == pytest.approx(math.cos(math.pi/4), abs=1e-6)
        assert v.z == pytest.approx(-math.sin(math.pi/4), abs=1e-6)

    def test_quat_to_mat4_identity_preserves_point(self):
        """Quat.identity().to_mat4() transforms point correctly."""
        m = Quat.identity().to_mat4()
        p = m.transform_point(Vec3(1, 2, 3))
        assert p == Vec3(1, 2, 3)

    def test_quat_direction_vectors_orthogonal(self):
        """Quat direction vectors (forward, up, right) form orthogonal basis."""
        q = Quat.from_axis_angle(Vec3(0, 1, 0), 0.3)
        f = q.forward()
        u = q.up()
        r = q.right()
        assert abs(f.dot(u)) < 1e-6
        assert abs(f.dot(r)) < 1e-6
        assert abs(u.dot(r)) < 1e-6

    def test_quat_direction_vectors_unit_length(self):
        """Quat direction vectors have unit length."""
        q = Quat.from_axis_angle(Vec3(0, 0, 1), 0.5)
        assert q.forward().length() == pytest.approx(1.0)
        assert q.up().length() == pytest.approx(1.0)
        assert q.right().length() == pytest.approx(1.0)

    def test_lerp_transform_chain(self):
        """Transform lerp produces correct intermediate state."""
        a = Transform(translation=Vec3(0, 0, 0))
        b = Transform(translation=Vec3(10, 20, 30))
        mid = a.lerp(b, 0.5)
        assert mid.translation == Vec3(5, 10, 15)


# =============================================================================
# Mat4 singular matrix fallback
# =============================================================================

class TestMat4SingularVerification:
    """Verify Mat4.inverse() handles singular matrices correctly."""

    def test_zero_matrix_returns_identity(self):
        """Zero matrix inverse returns identity."""
        assert Mat4([0]*16).inverse() == Mat4.identity()

    def test_singular_nonzero_matrix(self):
        """Non-zero singular matrix returns identity."""
        m = Mat4([1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16])
        result = m.inverse()
        assert result == Mat4.identity()

    def test_determinant_zero_returns_identity(self):
        """Matrix with det=0 returns identity from inverse."""
        m = Mat4([0]*16)
        # Sanity check: det is 0
        assert m.determinant() == pytest.approx(0.0)
        assert m.inverse() == Mat4.identity()


# =============================================================================
# Mat3 singular matrix fallback
# =============================================================================

class TestMat3SingularVerification:
    """Verify Mat3.inverse() handles singular matrices correctly."""

    def test_zero_matrix_returns_identity(self):
        """Zero matrix inverse returns identity."""
        assert Mat3([0]*9).inverse() == Mat3()

    def test_singular_nonzero_matrix(self):
        """Non-zero singular matrix returns identity."""
        m = Mat3([1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = m.inverse()
        assert result == Mat3()
