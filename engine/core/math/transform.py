"""Transform and RigidTransform types."""

from __future__ import annotations

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4


class Transform:
    """Translation + rotation + scale.

    Composition order (applied right-to-left): T(t) * R(r) * S(s),
    so f(p) = R * (S * p) + t.

    Note on inverse with non-uniform scale: when both non-uniform scale and
    non-trivial rotation are present, the inverse cannot be exactly
    represented as another TRS decomposition (this is a known mathematical
    limitation shared by all TRS-based engines). In this case the inverse
    stores the matrix-path decomposition, which gives an approximation.
    For exact results, convert to Mat4, invert, and work in matrix space.
    """
    __slots__ = ("translation", "rotation", "scale")

    def __init__(
        self,
        translation: Vec3 | None = None,
        rotation: Quat | None = None,
        scale: Vec3 | None = None,
    ) -> None:
        self.translation = translation or Vec3.zero()
        self.rotation = rotation or Quat.identity()
        self.scale = scale or Vec3.one()

    @staticmethod
    def identity() -> Transform:
        return Transform()

    def to_matrix(self) -> Mat4:
        r = self.rotation.to_mat4()
        s = Mat4.scale(self.scale)
        t = Mat4.translation(self.translation)
        return t @ r @ s

    @staticmethod
    def from_matrix(m: Mat4) -> Transform:
        tx = Vec3(m.m[12], m.m[13], m.m[14])
        import math
        sx = Vec3(m.m[0], m.m[1], m.m[2]).length()
        sy = Vec3(m.m[4], m.m[5], m.m[6]).length()
        sz = Vec3(m.m[8], m.m[9], m.m[10]).length()

        from engine.core.math.mat import Mat3
        rot_m = Mat3([
            m.m[0]/sx if sx else 0, m.m[1]/sx if sx else 0, m.m[2]/sx if sx else 0,
            m.m[4]/sy if sy else 0, m.m[5]/sy if sy else 0, m.m[6]/sy if sy else 0,
            m.m[8]/sz if sz else 0, m.m[9]/sz if sz else 0, m.m[10]/sz if sz else 0,
        ])
        # Extract quaternion from rotation matrix
        r = rot_m.m
        trace = r[0] + r[4] + r[8]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (r[5] - r[7]) * s
            y = (r[6] - r[2]) * s
            z = (r[1] - r[3]) * s
        elif r[0] > r[4] and r[0] > r[8]:
            s = 2.0 * math.sqrt(1.0 + r[0] - r[4] - r[8])
            w = (r[5] - r[7]) / s
            x = 0.25 * s
            y = (r[3] + r[1]) / s
            z = (r[6] + r[2]) / s
        elif r[4] > r[8]:
            s = 2.0 * math.sqrt(1.0 + r[4] - r[0] - r[8])
            w = (r[6] - r[2]) / s
            x = (r[3] + r[1]) / s
            y = 0.25 * s
            z = (r[7] + r[5]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + r[8] - r[0] - r[4])
            w = (r[1] - r[3]) / s
            x = (r[6] + r[2]) / s
            y = (r[7] + r[5]) / s
            z = 0.25 * s

        return Transform(tx, Quat(x, y, z, w).normalized(), Vec3(sx, sy, sz))

    def transform_point(self, p: Vec3) -> Vec3:
        return self.to_matrix().transform_point(p)

    def transform_direction(self, d: Vec3) -> Vec3:
        return self.rotation.rotate_vector(d)

    def inverse(self) -> Transform:
        return Transform.from_matrix(self.to_matrix().inverse())

    def lerp(self, other: Transform, t: float) -> Transform:
        return Transform(
            self.translation.lerp(other.translation, t),
            self.rotation.slerp(other.rotation, t),
            self.scale.lerp(other.scale, t),
        )

    def __repr__(self) -> str:
        return f"Transform(t={self.translation}, r={self.rotation}, s={self.scale})"


class RigidTransform:
    """Translation + rotation only (no scale). Faster inverse."""
    __slots__ = ("translation", "rotation")

    def __init__(self, translation: Vec3 | None = None, rotation: Quat | None = None) -> None:
        self.translation = translation or Vec3.zero()
        self.rotation = rotation or Quat.identity()

    @staticmethod
    def identity() -> RigidTransform:
        return RigidTransform()

    def to_matrix(self) -> Mat4:
        r = self.rotation.to_mat4()
        t = Mat4.translation(self.translation)
        return t @ r

    def transform_point(self, p: Vec3) -> Vec3:
        return self.rotation.rotate_vector(p) + self.translation

    def transform_direction(self, d: Vec3) -> Vec3:
        return self.rotation.rotate_vector(d)

    def inverse(self) -> RigidTransform:
        inv_rot = self.rotation.inverse()
        inv_trans = inv_rot.rotate_vector(-self.translation)
        return RigidTransform(inv_trans, inv_rot)

    def __repr__(self) -> str:
        return f"RigidTransform(t={self.translation}, r={self.rotation})"
