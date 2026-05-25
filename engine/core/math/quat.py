"""Quaternion type for rotations."""

from __future__ import annotations

import math

from engine.core.math.vec import Vec3
from engine.core.math.mat import Mat3, Mat4

from engine.core.constants import MATH_EPSILON as _EPSILON, SLERP_THRESHOLD


class Quat:
    """Unit quaternion (x, y, z, w) for 3D rotations."""
    __slots__ = ("x", "y", "z", "w")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    @staticmethod
    def identity() -> Quat:
        return Quat(0, 0, 0, 1)

    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> Quat:
        half = angle * 0.5
        s = math.sin(half)
        a = axis.normalized()
        return Quat(a.x * s, a.y * s, a.z * s, math.cos(half))

    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float) -> Quat:
        """Create from Euler angles (radians). Order: yaw(Y) * pitch(X) * roll(Z)."""
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return Quat(
            x=cy * sp * cr + sy * cp * sr,
            y=sy * cp * cr - cy * sp * sr,
            z=cy * cp * sr - sy * sp * cr,
            w=cy * cp * cr + sy * sp * sr,
        )

    def __mul__(self, other: Quat) -> Quat:
        return Quat(
            self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w,
            self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
        )

    def rotate_vector(self, v: Vec3) -> Vec3:
        qv = Vec3(self.x, self.y, self.z)
        uv = qv.cross(v)
        uuv = qv.cross(uv)
        return v + (uv * self.w + uuv) * 2.0

    def conjugate(self) -> Quat:
        return Quat(-self.x, -self.y, -self.z, self.w)

    def length_squared(self) -> float:
        return self.x**2 + self.y**2 + self.z**2 + self.w**2

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> Quat:
        ln = self.length()
        if ln < _EPSILON:
            return Quat.identity()
        return Quat(self.x/ln, self.y/ln, self.z/ln, self.w/ln)

    def inverse(self) -> Quat:
        ls = self.length_squared()
        if ls < _EPSILON:
            return Quat.identity()
        return Quat(-self.x/ls, -self.y/ls, -self.z/ls, self.w/ls)

    def dot(self, other: Quat) -> float:
        return self.x*other.x + self.y*other.y + self.z*other.z + self.w*other.w

    def slerp(self, other: Quat, t: float) -> Quat:
        d = self.dot(other)
        o = other
        if d < 0.0:
            o = Quat(-o.x, -o.y, -o.z, -o.w)
            d = -d
        if d > SLERP_THRESHOLD:
            return self.nlerp(o, t)
        theta = math.acos(min(d, 1.0))
        sin_theta = math.sin(theta)
        s0 = math.sin((1.0 - t) * theta) / sin_theta
        s1 = math.sin(t * theta) / sin_theta
        return Quat(
            s0*self.x + s1*o.x,
            s0*self.y + s1*o.y,
            s0*self.z + s1*o.z,
            s0*self.w + s1*o.w,
        )

    def nlerp(self, other: Quat, t: float) -> Quat:
        return Quat(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
            self.w + (other.w - self.w) * t,
        ).normalized()

    def to_mat3(self) -> Mat3:
        x, y, z, w = self.x, self.y, self.z, self.w
        x2, y2, z2 = x+x, y+y, z+z
        xx, xy, xz = x*x2, x*y2, x*z2
        yy, yz, zz = y*y2, y*z2, z*z2
        wx, wy, wz = w*x2, w*y2, w*z2
        return Mat3([
            1-(yy+zz), xy+wz, xz-wy,
            xy-wz, 1-(xx+zz), yz+wx,
            xz+wy, yz-wx, 1-(xx+yy),
        ])

    def to_mat4(self) -> Mat4:
        m3 = self.to_mat3().m
        return Mat4([
            m3[0], m3[1], m3[2], 0,
            m3[3], m3[4], m3[5], 0,
            m3[6], m3[7], m3[8], 0,
            0, 0, 0, 1,
        ])

    def to_euler(self) -> tuple[float, float, float]:
        """Returns (pitch, yaw, roll) in radians."""
        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        pitch = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1.0:
            yaw = math.copysign(math.pi / 2, sinp)
        else:
            yaw = math.asin(sinp)

        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        roll = math.atan2(siny_cosp, cosy_cosp)

        return (pitch, yaw, roll)

    def forward(self) -> Vec3:
        return self.rotate_vector(Vec3(0, 0, -1))

    def up(self) -> Vec3:
        return self.rotate_vector(Vec3(0, 1, 0))

    def right(self) -> Vec3:
        return self.rotate_vector(Vec3(1, 0, 0))

    def __repr__(self) -> str:
        return f"Quat({self.x}, {self.y}, {self.z}, {self.w})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quat):
            return NotImplemented
        return (
            abs(self.x - other.x) < _EPSILON
            and abs(self.y - other.y) < _EPSILON
            and abs(self.z - other.z) < _EPSILON
            and abs(self.w - other.w) < _EPSILON
        )
