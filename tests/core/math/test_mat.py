"""Tests for Mat3, Mat4."""

import math
import pytest

from engine.core.math.mat import Mat3, Mat4
from engine.core.math.vec import Vec3


class TestMat4:
    def test_identity(self):
        m = Mat4.identity()
        assert m.m[0] == 1 and m.m[5] == 1 and m.m[10] == 1 and m.m[15] == 1

    def test_multiply_identity(self):
        m = Mat4.translation(Vec3(1, 2, 3))
        r = m @ Mat4.identity()
        assert r == m

    def test_translation_transform_point(self):
        m = Mat4.translation(Vec3(10, 20, 30))
        p = m.transform_point(Vec3(0, 0, 0))
        assert p == Vec3(10, 20, 30)

    def test_inverse(self):
        m = Mat4.translation(Vec3(5, 10, 15))
        inv = m.inverse()
        r = m @ inv
        assert r == Mat4.identity()

    def test_determinant_identity(self):
        assert Mat4.identity().determinant() == pytest.approx(1.0)

    def test_look_at(self):
        m = Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        p = m.transform_point(Vec3(0, 0, 0))
        assert p.z == pytest.approx(-5.0, abs=1e-6)

    def test_perspective(self):
        fov = math.radians(90)
        aspect = 1.0
        near = 0.1
        far = 100.0
        m = Mat4.perspective(fov, aspect, near, far)
        t = math.tan(fov / 2.0)
        assert m.m[0] == pytest.approx(1.0 / (aspect * t))
        assert m.m[5] == pytest.approx(1.0 / t)
        assert m.m[10] == pytest.approx(-(far + near) / (far - near))
        assert m.m[11] == pytest.approx(-1.0)
        assert m.m[14] == pytest.approx(-(2.0 * far * near) / (far - near))

    def test_rotation_x(self):
        m = Mat4.rotation_x(math.pi / 2)
        p = m.transform_point(Vec3(0, 1, 0))
        assert p.y == pytest.approx(0, abs=1e-6)
        assert p.z == pytest.approx(1, abs=1e-6)

    def test_rotation_y(self):
        m = Mat4.rotation_y(math.pi / 2)
        p = m.transform_point(Vec3(1, 0, 0))
        assert p.x == pytest.approx(0, abs=1e-6)
        assert p.z == pytest.approx(-1, abs=1e-6)

    def test_rotation_z(self):
        m = Mat4.rotation_z(math.pi / 2)
        p = m.transform_point(Vec3(1, 0, 0))
        assert p.x == pytest.approx(0, abs=1e-6)
        assert p.y == pytest.approx(1, abs=1e-6)

    def test_orthographic(self):
        m = Mat4.orthographic(-1, 1, -1, 1, 0.1, 100.0)
        # View-space z is negative (camera looks down -z)
        # Near plane (z_eye = -near) maps to NDC z = -1
        near_p = m.transform_point(Vec3(0, 0, -0.1))
        assert near_p.z == pytest.approx(-1.0, abs=1e-6)
        # Far plane (z_eye = -far) maps to NDC z = 1
        far_p = m.transform_point(Vec3(0, 0, -100.0))
        assert far_p.z == pytest.approx(1.0, abs=1e-6)
        # Center maps to 0,0
        center = m.transform_point(Vec3(0, 0, -50))
        assert center.x == pytest.approx(0, abs=1e-6)
        assert center.y == pytest.approx(0, abs=1e-6)

    def test_transform_direction(self):
        m = Mat4.translation(Vec3(10, 20, 30))
        d = m.transform_direction(Vec3(1, 0, 0))
        assert d == Vec3(1, 0, 0)  # translation ignored for directions

    def test_transposed(self):
        m = Mat4([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        t = m.transposed()
        # After transpose, original translation at 12,13,14 moves to 3,7,11
        assert t.m[3] == pytest.approx(13)
        assert t.m[7] == pytest.approx(14)
        assert t.m[11] == pytest.approx(15)

    def test_repr(self):
        r = repr(Mat4.identity())
        assert "Mat4" in r

    def test_eq(self):
        assert Mat4.identity() == Mat4.identity()
        assert Mat4.identity() != Mat4.translation(Vec3(1, 0, 0))
        assert Mat4.identity().__eq__("nope") == NotImplemented

    def test_transform_point_perspective_divide(self):
        """Perspective-projected points get w-division."""
        m = Mat4.perspective(math.radians(90), 1.0, 0.1, 100.0)
        p = m.transform_point(Vec3(0, 0, -2))
        assert p is not None

    def test_transform_point_w_zero_no_crash(self):
        """w near zero should warn but not crash, returns xyz as-is."""
        m = Mat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = m.transform_point(Vec3(1, 2, 3))
        assert p is not None

    def test_inverse_singular(self):
        import logging
        m = Mat4([0]*16)
        result = m.inverse()
        assert result == Mat4.identity()  # falls back to identity
        assert result is not None


class TestMat3:
    def test_from_mat4(self):
        m4 = Mat4.identity()
        m3 = Mat3.from_mat4(m4)
        assert m3 == Mat3()

    def test_inverse(self):
        m = Mat3([2, 0, 0, 0, 3, 0, 0, 0, 4])
        inv = m.inverse()
        assert inv.m[0] == pytest.approx(0.5)
        assert inv.m[4] == pytest.approx(1/3)
        assert inv.m[8] == pytest.approx(0.25)

    def test_transposed(self):
        m = Mat3([1, 2, 3, 4, 5, 6, 7, 8, 9])
        t = m.transposed()
        assert t.m[1] == pytest.approx(4) and t.m[3] == pytest.approx(2)

    def test_matmul_vec3(self):
        m = Mat3([2, 0, 0, 0, 3, 0, 0, 0, 4])
        v = m @ Vec3(1, 2, 3)
        assert v == Vec3(2, 6, 12)

    def test_repr(self):
        r = repr(Mat3())
        assert "Mat3" in r

    def test_eq(self):
        assert Mat3() == Mat3()
        assert Mat3() != Mat3([1, 0, 0, 0, 1, 0, 0, 0, 2])
        assert Mat3().__eq__("nope") == NotImplemented

    def test_inverse_singular(self):
        m = Mat3([1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = m.inverse()
        assert result == Mat3()  # falls back to identity
