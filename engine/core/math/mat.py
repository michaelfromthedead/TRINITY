"""Matrix types: Mat3, Mat4 (column-major)."""

from __future__ import annotations

import logging
import math
from typing import List

from engine.core.constants import MATH_EPSILON, MATH_EPSILON_TIGHT
from engine.core.math.vec import Vec3, Vec4

logger = logging.getLogger(__name__)


class Mat4:
    """4x4 column-major matrix stored as flat list[16].

    Index layout (column-major):
        [0]  [4]  [8]  [12]
        [1]  [5]  [9]  [13]
        [2]  [6]  [10] [14]
        [3]  [7]  [11] [15]
    """
    __slots__ = ("m",)

    def __init__(self, m: List[float] | None = None) -> None:
        if m is not None:
            self.m = list(m)
        else:
            self.m = [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ]

    @staticmethod
    def identity() -> Mat4:
        return Mat4()

    @staticmethod
    def translation(v: Vec3) -> Mat4:
        m = Mat4()
        m.m[12] = v.x
        m.m[13] = v.y
        m.m[14] = v.z
        return m

    @staticmethod
    def rotation_x(radians: float) -> Mat4:
        c, s = math.cos(radians), math.sin(radians)
        m = Mat4()
        m.m[5] = c; m.m[6] = s
        m.m[9] = -s; m.m[10] = c
        return m

    @staticmethod
    def rotation_y(radians: float) -> Mat4:
        c, s = math.cos(radians), math.sin(radians)
        m = Mat4()
        m.m[0] = c; m.m[2] = -s
        m.m[8] = s; m.m[10] = c
        return m

    @staticmethod
    def rotation_z(radians: float) -> Mat4:
        c, s = math.cos(radians), math.sin(radians)
        m = Mat4()
        m.m[0] = c; m.m[1] = s
        m.m[4] = -s; m.m[5] = c
        return m

    @staticmethod
    def scale(v: Vec3) -> Mat4:
        m = Mat4()
        m.m[0] = v.x
        m.m[5] = v.y
        m.m[10] = v.z
        return m

    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
        f = (target - eye).normalized()
        r = f.cross(up).normalized()
        u = r.cross(f)
        m = Mat4()
        m.m[0] = r.x;  m.m[4] = r.y;  m.m[8]  = r.z;  m.m[12] = -r.dot(eye)
        m.m[1] = u.x;  m.m[5] = u.y;  m.m[9]  = u.z;  m.m[13] = -u.dot(eye)
        m.m[2] = -f.x; m.m[6] = -f.y; m.m[10] = -f.z; m.m[14] = f.dot(eye)
        m.m[3] = 0;    m.m[7] = 0;    m.m[11] = 0;    m.m[15] = 1
        return m

    @staticmethod
    def perspective(fov_y: float, aspect: float, near: float, far: float) -> Mat4:
        t = math.tan(fov_y / 2.0)
        m = Mat4([0]*16)
        m.m[0] = 1.0 / (aspect * t)
        m.m[5] = 1.0 / t
        m.m[10] = -(far + near) / (far - near)
        m.m[11] = -1.0
        m.m[14] = -(2.0 * far * near) / (far - near)
        return m

    @staticmethod
    def orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> Mat4:
        m = Mat4([0]*16)
        m.m[0] = 2.0 / (right - left)
        m.m[5] = 2.0 / (top - bottom)
        m.m[10] = -2.0 / (far - near)
        m.m[12] = -(right + left) / (right - left)
        m.m[13] = -(top + bottom) / (top - bottom)
        m.m[14] = -(far + near) / (far - near)
        m.m[15] = 1.0
        return m

    def __matmul__(self, other: Mat4) -> Mat4:
        a, b = self.m, other.m
        r = [0.0] * 16
        for col in range(4):
            for row in range(4):
                s = 0.0
                for k in range(4):
                    s += a[k * 4 + row] * b[col * 4 + k]
                r[col * 4 + row] = s
        return Mat4(r)

    def transform_point(self, v: Vec3) -> Vec3:
        m = self.m
        x = m[0]*v.x + m[4]*v.y + m[8]*v.z + m[12]
        y = m[1]*v.x + m[5]*v.y + m[9]*v.z + m[13]
        z = m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]
        w = m[3]*v.x + m[7]*v.y + m[11]*v.z + m[15]
        if abs(w) < MATH_EPSILON:
            logger.warning("Mat4.transform_point: w near zero (%.2e), result may be degenerate", w)
        if abs(w) > MATH_EPSILON and abs(w - 1.0) > MATH_EPSILON:
            return Vec3(x/w, y/w, z/w)
        return Vec3(x, y, z)

    def transform_direction(self, v: Vec3) -> Vec3:
        m = self.m
        return Vec3(
            m[0]*v.x + m[4]*v.y + m[8]*v.z,
            m[1]*v.x + m[5]*v.y + m[9]*v.z,
            m[2]*v.x + m[6]*v.y + m[10]*v.z,
        )

    def transposed(self) -> Mat4:
        m = self.m
        return Mat4([
            m[0], m[4], m[8],  m[12],
            m[1], m[5], m[9],  m[13],
            m[2], m[6], m[10], m[14],
            m[3], m[7], m[11], m[15],
        ])

    def determinant(self) -> float:
        m = self.m
        a, b, c, d = m[0], m[4], m[8],  m[12]
        e, f, g, h = m[1], m[5], m[9],  m[13]
        i, j, k, l = m[2], m[6], m[10], m[14]
        n, o, p, q = m[3], m[7], m[11], m[15]

        kq_lp = k*q - l*p
        jq_lo = j*q - l*o
        jp_ko = j*p - k*o
        iq_ln = i*q - l*n
        ip_kn = i*p - k*n
        io_jn = i*o - j*n

        return (
            a * (f*kq_lp - g*jq_lo + h*jp_ko)
            - b * (e*kq_lp - g*iq_ln + h*ip_kn)
            + c * (e*jq_lo - f*iq_ln + h*io_jn)
            - d * (e*jp_ko - f*ip_kn + g*io_jn)
        )

    def inverse(self) -> Mat4:
        m = self.m
        inv = [0.0] * 16

        inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10]
        inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10]
        inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9]
        inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9]

        inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10]
        inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10]
        inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9]
        inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9]

        inv[2]  =  m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6]
        inv[6]  = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6]
        inv[10] =  m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5]
        inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5]

        inv[3]  = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6]
        inv[7]  =  m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6]
        inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9]  + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7] + m[8]*m[3]*m[5]
        inv[15] =  m[0]*m[5]*m[10] - m[0]*m[6]*m[9]  - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6] - m[8]*m[2]*m[5]

        det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12]
        if abs(det) < MATH_EPSILON_TIGHT:
            logger.warning("Mat4.inverse(): singular matrix (det=%.2e), returning identity", det)
            return Mat4()  # fallback to identity
        inv_det = 1.0 / det
        return Mat4([x * inv_det for x in inv])

    def __repr__(self) -> str:
        return f"Mat4({self.m})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mat4):
            return NotImplemented
        return all(abs(a - b) < MATH_EPSILON for a, b in zip(self.m, other.m))


class Mat3:
    """3x3 column-major matrix stored as flat list[9]."""
    __slots__ = ("m",)

    def __init__(self, m: List[float] | None = None) -> None:
        if m is not None:
            self.m = list(m)
        else:
            self.m = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    @staticmethod
    def from_mat4(mat: Mat4) -> Mat3:
        m = mat.m
        return Mat3([m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]])

    def transposed(self) -> Mat3:
        m = self.m
        return Mat3([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]])

    def determinant(self) -> float:
        m = self.m
        return (
            m[0] * (m[4]*m[8] - m[5]*m[7])
            - m[3] * (m[1]*m[8] - m[2]*m[7])
            + m[6] * (m[1]*m[5] - m[2]*m[4])
        )

    def inverse(self) -> Mat3:
        m = self.m
        det = self.determinant()
        if abs(det) < MATH_EPSILON_TIGHT:
            logger.warning("Mat3.inverse(): singular matrix (det=%.2e), returning identity", det)
            return Mat3()
        inv_det = 1.0 / det
        return Mat3([
            (m[4]*m[8] - m[5]*m[7]) * inv_det,
            (m[2]*m[7] - m[1]*m[8]) * inv_det,
            (m[1]*m[5] - m[2]*m[4]) * inv_det,
            (m[5]*m[6] - m[3]*m[8]) * inv_det,
            (m[0]*m[8] - m[2]*m[6]) * inv_det,
            (m[2]*m[3] - m[0]*m[5]) * inv_det,
            (m[3]*m[7] - m[4]*m[6]) * inv_det,
            (m[1]*m[6] - m[0]*m[7]) * inv_det,
            (m[0]*m[4] - m[1]*m[3]) * inv_det,
        ])

    def __matmul__(self, other: Vec3) -> Vec3:
        m = self.m
        return Vec3(
            m[0]*other.x + m[3]*other.y + m[6]*other.z,
            m[1]*other.x + m[4]*other.y + m[7]*other.z,
            m[2]*other.x + m[5]*other.y + m[8]*other.z,
        )

    def __repr__(self) -> str:
        return f"Mat3({self.m})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mat3):
            return NotImplemented
        return all(abs(a - b) < MATH_EPSILON for a, b in zip(self.m, other.m))
