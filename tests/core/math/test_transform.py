"""Tests for Transform and RigidTransform."""

import math
import pytest

from engine.core.math.transform import Transform, RigidTransform
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4


class TestTransform:
    def test_identity_matrix(self):
        t = Transform.identity()
        assert t.to_matrix() == Mat4.identity()

    def test_transform_point(self):
        t = Transform(translation=Vec3(10, 0, 0))
        p = t.transform_point(Vec3(0, 0, 0))
        assert p == Vec3(10, 0, 0)

    def test_inverse(self):
        t = Transform(translation=Vec3(5, 10, 15))
        inv = t.inverse()
        p = t.transform_point(Vec3(0, 0, 0))
        p2 = inv.transform_point(p)
        assert p2.x == pytest.approx(0, abs=1e-4)
        assert p2.y == pytest.approx(0, abs=1e-4)
        assert p2.z == pytest.approx(0, abs=1e-4)

    def test_to_matrix_from_matrix_roundtrip(self):
        t = Transform(
            translation=Vec3(1, 2, 3),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), 0.5),
            scale=Vec3(2, 2, 2),
        )
        m = t.to_matrix()
        t2 = Transform.from_matrix(m)
        assert t2.translation.x == pytest.approx(1, abs=1e-3)
        assert t2.scale.x == pytest.approx(2, abs=1e-3)

    def test_lerp(self):
        a = Transform(translation=Vec3(0, 0, 0))
        b = Transform(translation=Vec3(10, 0, 0))
        mid = a.lerp(b, 0.5)
        assert mid.translation.x == pytest.approx(5)

    def test_lerp_endpoints(self):
        """Edge case: lerp at 0 and 1."""
        a = Transform(translation=Vec3(1, 2, 3))
        b = Transform(translation=Vec3(4, 5, 6))
        assert a.lerp(b, 0.0).translation == a.translation
        assert a.lerp(b, 1.0).translation == b.translation

    def test_transform_direction(self):
        t = Transform(scale=Vec3(2, 2, 2))  # scale ignored for directions
        d = t.transform_direction(Vec3(1, 0, 0))
        assert d == Vec3(1, 0, 0)

    def test_transform_direction_rotated(self):
        """Edge case: rotated direction."""
        t = Transform(rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2))
        d = t.transform_direction(Vec3(1, 0, 0))
        assert d.x == pytest.approx(0, abs=1e-6)
        assert d.z == pytest.approx(-1, abs=1e-6)

    def test_inverse_uniform_scale(self):
        """Uniform scale + rotation: perfect inverse roundtrip."""
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

    def test_inverse_nonuniform_scale_no_rotation(self):
        """Non-uniform scale without rotation: perfect inverse roundtrip."""
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

    def test_inverse_nonuniform_scale_matrix_roundtrip(self):
        """Non-uniform scale + rotation: matrix-level M * M^-1 = I."""
        t = Transform(
            translation=Vec3(1, 2, 3),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), 0.5),
            scale=Vec3(2, 3, 4),
        )
        m = t.to_matrix()
        m_inv_exact = m.inverse()
        roundtrip = m @ m_inv_exact
        assert roundtrip == Mat4.identity()

    def test_inverse_zero_scale_x(self):
        """Edge case: inverse with zero scale on X axis (singular matrix falls back to identity)."""
        t = Transform(translation=Vec3(0, 0, 0), scale=Vec3(0, 1, 1))
        inv = t.inverse()
        # Singular matrix → Mat4.inverse returns identity → from_matrix(identity) gives scale (1,1,1)
        assert inv.scale == Vec3(1, 1, 1)

    def test_from_matrix_identity(self):
        """Edge case: from_matrix with identity."""
        t = Transform.from_matrix(Mat4.identity())
        assert t.translation == Vec3(0, 0, 0)
        assert t.rotation == Quat.identity()
        assert t.scale == Vec3(1, 1, 1)

    def test_from_matrix_zero_scale_x(self):
        """Edge case: from_matrix with zero X scale."""
        m = Mat4.rotation_x(0.3)
        m.m[0] = 0
        t = Transform.from_matrix(m)
        assert t.scale.x == pytest.approx(0)

    def test_from_matrix_rotation_180_x(self):
        """from_matrix with 180-degree X rotation triggers alternate quaternion branch."""
        q = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi)
        t = Transform(rotation=q)
        m = t.to_matrix()
        t2 = Transform.from_matrix(m)
        assert t2.translation == Vec3.zero()
        assert abs(t2.rotation.dot(q)) > 0.99

    def test_from_matrix_rotation_180_y(self):
        """from_matrix with 180-degree Y rotation triggers another branch."""
        q = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi)
        t = Transform(rotation=q)
        m = t.to_matrix()
        t2 = Transform.from_matrix(m)
        assert t2.translation == Vec3.zero()
        assert abs(t2.rotation.dot(q)) > 0.99

    def test_from_matrix_rotation_180_z(self):
        """from_matrix with 180-degree Z rotation triggers the else branch."""
        q = Quat.from_axis_angle(Vec3(0, 0, 1), math.pi)
        t = Transform(rotation=q)
        m = t.to_matrix()
        t2 = Transform.from_matrix(m)
        assert t2.translation == Vec3.zero()
        assert abs(t2.rotation.dot(q)) > 0.99

    def test_repr(self):
        t = Transform(translation=Vec3(1, 2, 3))
        r = repr(t)
        assert "Transform" in r and "1" in r


class TestRigidTransform:
    def test_identity(self):
        rt = RigidTransform.identity()
        assert rt.translation == Vec3(0, 0, 0)
        assert rt.rotation == Quat.identity()

    def test_identity_to_matrix(self):
        """Edge case: identity rigid transform matrix."""
        assert RigidTransform.identity().to_matrix() == Mat4.identity()

    def test_to_matrix_with_translation(self):
        rt = RigidTransform(
            translation=Vec3(1, 2, 3),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4),
        )
        m = rt.to_matrix()
        p = m.transform_point(Vec3(0, 0, 0))
        assert p == Vec3(1, 2, 3)

    def test_transform_point(self):
        rt = RigidTransform(translation=Vec3(10, 0, 0))
        p = rt.transform_point(Vec3(0, 0, 0))
        assert p == Vec3(10, 0, 0)

    def test_transform_direction(self):
        rt = RigidTransform(
            translation=Vec3(10, 20, 30),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2),
        )
        d = rt.transform_direction(Vec3(1, 0, 0))
        assert d.z == pytest.approx(-1, abs=1e-6)

    def test_transform_direction_ignores_translation(self):
        """Edge case: direction ignores translation."""
        rt = RigidTransform(translation=Vec3(100, 0, 0))
        d = rt.transform_direction(Vec3(1, 0, 0))
        assert d == Vec3(1, 0, 0)

    def test_inverse_roundtrip(self):
        rt = RigidTransform(
            translation=Vec3(3, 4, 5),
            rotation=Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 4),
        )
        inv = rt.inverse()
        p = rt.transform_point(Vec3(1, 0, 0))
        p2 = inv.transform_point(p)
        assert p2.x == pytest.approx(1, abs=1e-6)
        assert p2.y == pytest.approx(0, abs=1e-6)
        assert p2.z == pytest.approx(0, abs=1e-6)

    def test_inverse_identity(self):
        """Edge case: inverse of identity."""
        rt = RigidTransform.identity()
        inv = rt.inverse()
        assert inv.translation == Vec3(0, 0, 0)
        assert inv.rotation == Quat.identity()

    def test_repr(self):
        rt = RigidTransform(translation=Vec3(1, 2, 3))
        r = repr(rt)
        assert "RigidTransform" in r
