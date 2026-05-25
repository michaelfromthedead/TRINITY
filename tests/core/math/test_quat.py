"""Tests for Quat."""

import math
import pytest

from engine.core.math.quat import Quat
from engine.core.math.vec import Vec3


class TestQuat:
    def test_identity(self):
        q = Quat.identity()
        assert q.w == pytest.approx(1)
        assert q.x == pytest.approx(0)

    def test_axis_angle(self):
        q = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi)
        assert q.length() == pytest.approx(1)

    def test_rotate_vector(self):
        q = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        v = q.rotate_vector(Vec3(1, 0, 0))
        assert v.x == pytest.approx(0, abs=1e-6)
        assert v.z == pytest.approx(-1, abs=1e-6)

    def test_conjugate(self):
        q = Quat(1, 2, 3, 4)
        c = q.conjugate()
        assert c.x == -1 and c.y == -2 and c.z == -3 and c.w == 4

    def test_slerp_endpoints(self):
        a = Quat.identity()
        b = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        s0 = a.slerp(b, 0.0)
        s1 = a.slerp(b, 1.0)
        assert s0 == a
        assert abs(s1.x - b.x) < 1e-6

    def test_euler_roundtrip(self):
        q = Quat.from_euler(0.3, 0.5, 0.1)
        p, y, r = q.to_euler()
        q2 = Quat.from_euler(p, y, r)
        assert abs(q.dot(q2)) == pytest.approx(1.0, abs=1e-3)

    def test_to_mat4_identity(self):
        from engine.core.math.mat import Mat4
        m = Quat.identity().to_mat4()
        assert m == Mat4.identity()

    def test_direction_vectors(self):
        q = Quat.identity()
        assert q.forward() == Vec3(0, 0, -1)
        assert q.up() == Vec3(0, 1, 0)
        assert q.right() == Vec3(1, 0, 0)

    def test_mul(self):
        """Quaternion multiplication composes rotations."""
        q1 = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        q2 = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        q = q1 * q2
        v = q.rotate_vector(Vec3(1, 0, 0))
        assert v.x == pytest.approx(-1, abs=1e-6)

    def test_normalized_zero(self):
        z = Quat(0, 0, 0, 0).normalized()
        assert z == Quat.identity()

    def test_inverse_zero(self):
        z = Quat(0, 0, 0, 0).inverse()
        assert z == Quat.identity()

    def test_to_euler_gimbal_lock(self):
        """Gimbal lock at pitch = +/- 90 degrees produces valid Euler angles."""
        q = Quat.from_euler(math.pi / 2, 0.3, 0.0)
        p, y, r = q.to_euler()
        q2 = Quat.from_euler(p, y, r)
        assert abs(q.dot(q2)) == pytest.approx(1.0, abs=1e-3)

    def test_repr(self):
        r = repr(Quat(0.5, 0.5, 0.5, 0.5))
        assert "Quat" in r and "0.5" in r

    def test_eq(self):
        assert Quat.identity() == Quat.identity()
        assert Quat.identity() != Quat(1, 0, 0, 0)
        assert Quat.identity().__eq__("nope") == NotImplemented
