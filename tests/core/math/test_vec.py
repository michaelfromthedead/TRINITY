"""Tests for Vec2, Vec3, Vec4."""

import math
import pytest

from engine.core.math.vec import Vec2, Vec3, Vec4


class TestVec2:
    def test_add(self):
        assert Vec2(1, 2) + Vec2(3, 4) == Vec2(4, 6)

    def test_sub(self):
        assert Vec2(5, 3) - Vec2(1, 2) == Vec2(4, 1)

    def test_mul(self):
        assert Vec2(2, 3) * 2 == Vec2(4, 6)

    def test_neg(self):
        assert -Vec2(1, -2) == Vec2(-1, 2)

    def test_dot(self):
        assert Vec2(1, 0).dot(Vec2(0, 1)) == pytest.approx(0)

    def test_length(self):
        assert Vec2(3, 4).length() == pytest.approx(5)

    def test_normalized(self):
        n = Vec2(3, 0).normalized()
        assert n.length() == pytest.approx(1)

    def test_lerp(self):
        r = Vec2(0, 0).lerp(Vec2(10, 10), 0.5)
        assert r == Vec2(5, 5)

    def test_distance(self):
        assert Vec2(0, 0).distance(Vec2(3, 4)) == pytest.approx(5)

    def test_rmul(self):
        assert 2 * Vec2(3, 4) == Vec2(6, 8)

    def test_truediv(self):
        assert Vec2(6, 8) / 2 == Vec2(3, 4)

    def test_eq(self):
        assert Vec2(1, 2) == Vec2(1, 2)
        assert Vec2(1, 2) != Vec2(1, 3)
        assert Vec2(1, 2).__eq__("nope") == NotImplemented

    def test_repr(self):
        r = repr(Vec2(1.5, 2.5))
        assert "Vec2" in r and "1.5" in r

    def test_min_max_clamp(self):
        a, b = Vec2(1, 5), Vec2(3, 2)
        assert a.min(b) == Vec2(1, 2)
        assert a.max(b) == Vec2(3, 5)
        lo, hi = Vec2(0, 0), Vec2(2, 4)
        assert Vec2(-1, 5).clamp(lo, hi) == Vec2(0, 4)

    def test_zero_one(self):
        assert Vec2.zero() == Vec2(0, 0)
        assert Vec2.one() == Vec2(1, 1)

    def test_unit_x_y(self):
        assert Vec2.unit_x() == Vec2(1, 0)
        assert Vec2.unit_y() == Vec2(0, 1)

    def test_normalized_zero(self):
        z = Vec2.zero().normalized()
        assert z == Vec2(0, 0)


class TestVec3:
    def test_add_sub(self):
        assert Vec3(1, 2, 3) + Vec3(4, 5, 6) == Vec3(5, 7, 9)
        assert Vec3(5, 5, 5) - Vec3(1, 2, 3) == Vec3(4, 3, 2)

    def test_cross(self):
        c = Vec3(1, 0, 0).cross(Vec3(0, 1, 0))
        assert c == Vec3(0, 0, 1)

    def test_dot(self):
        assert Vec3(1, 0, 0).dot(Vec3(0, 1, 0)) == pytest.approx(0)

    def test_length(self):
        assert Vec3(0, 3, 4).length() == pytest.approx(5)

    def test_normalized(self):
        n = Vec3(0, 0, 5).normalized()
        assert n.length() == pytest.approx(1)
        assert n == Vec3(0, 0, 1)

    def test_reflect(self):
        v = Vec3(1, -1, 0).normalized()
        n = Vec3(0, 1, 0)
        r = v.reflect(n)
        assert r.y == pytest.approx(-v.y, abs=1e-6)

    def test_statics(self):
        assert Vec3.zero() == Vec3(0, 0, 0)
        assert Vec3.one() == Vec3(1, 1, 1)
        assert Vec3.up() == Vec3(0, 1, 0)
        assert Vec3.forward() == Vec3(0, 0, -1)
        assert Vec3.right() == Vec3(1, 0, 0)
        assert Vec3.unit_x() == Vec3(1, 0, 0)
        assert Vec3.unit_y() == Vec3(0, 1, 0)
        assert Vec3.unit_z() == Vec3(0, 0, 1)

    def test_rmul(self):
        assert 2 * Vec3(1, 2, 3) == Vec3(2, 4, 6)

    def test_truediv(self):
        assert Vec3(6, 9, 12) / 3 == Vec3(2, 3, 4)

    def test_neg(self):
        assert -Vec3(1, -2, 3) == Vec3(-1, 2, -3)

    def test_eq(self):
        assert Vec3(1, 2, 3) == Vec3(1, 2, 3)
        assert Vec3(1, 2, 3) != Vec3(1, 2, 4)
        assert Vec3(1, 2, 3).__eq__("nope") == NotImplemented

    def test_repr(self):
        r = repr(Vec3(1.5, 2.5, 3.5))
        assert "Vec3" in r and "1.5" in r

    def test_normalized_zero(self):
        z = Vec3.zero().normalized()
        assert z == Vec3(0, 0, 0)


class TestVec4:
    def test_arithmetic(self):
        a = Vec4(1, 2, 3, 4)
        b = Vec4(4, 3, 2, 1)
        assert (a + b) == Vec4(5, 5, 5, 5)

    def test_xyz(self):
        v = Vec4(1, 2, 3, 4)
        assert v.xyz == Vec3(1, 2, 3)

    def test_perspective_divide(self):
        v = Vec4(4, 8, 12, 4)
        p = v.perspective_divide()
        assert p == Vec3(1, 2, 3)

    def test_dot(self):
        assert Vec4(1, 0, 0, 0).dot(Vec4(0, 1, 0, 0)) == pytest.approx(0)

    def test_lerp(self):
        r = Vec4(0, 0, 0, 0).lerp(Vec4(10, 10, 10, 10), 0.5)
        assert r == Vec4(5, 5, 5, 5)

    def test_min_max_clamp(self):
        a, b = Vec4(1, 5, 3, 0), Vec4(3, 2, 1, 4)
        assert a.min(b) == Vec4(1, 2, 1, 0)
        assert a.max(b) == Vec4(3, 5, 3, 4)
        lo, hi = Vec4(0, 0, 0, 0), Vec4(2, 4, 3, 2)
        assert Vec4(-1, 5, 2, -1).clamp(lo, hi) == Vec4(0, 4, 2, 0)

    def test_zero_one(self):
        assert Vec4.zero() == Vec4(0, 0, 0, 0)
        assert Vec4.one() == Vec4(1, 1, 1, 1)

    def test_unit_xyzw(self):
        assert Vec4.unit_x() == Vec4(1, 0, 0, 0)
        assert Vec4.unit_y() == Vec4(0, 1, 0, 0)
        assert Vec4.unit_z() == Vec4(0, 0, 1, 0)
        assert Vec4.unit_w() == Vec4(0, 0, 0, 1)

    def test_rmul(self):
        assert 2 * Vec4(1, 2, 3, 4) == Vec4(2, 4, 6, 8)

    def test_truediv(self):
        assert Vec4(6, 9, 12, 3) / 3 == Vec4(2, 3, 4, 1)

    def test_neg(self):
        assert -Vec4(1, -2, 3, -4) == Vec4(-1, 2, -3, 4)

    def test_eq(self):
        assert Vec4(1, 2, 3, 4) == Vec4(1, 2, 3, 4)
        assert Vec4(1, 2, 3, 4) != Vec4(1, 2, 3, 5)
        assert Vec4(1, 2, 3, 4).__eq__("nope") == NotImplemented

    def test_repr(self):
        r = repr(Vec4(1.5, 2.5, 3.5, 4.5))
        assert "Vec4" in r and "1.5" in r

    def test_normalized_zero(self):
        z = Vec4.zero().normalized()
        assert z == Vec4(0, 0, 0, 0)

    def test_perspective_divide_w_zero_no_crash(self):
        """w near zero should not crash, returns xyz as-is."""
        v = Vec4(1, 2, 3, 0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = v.perspective_divide()
        assert p == Vec3(1, 2, 3)
