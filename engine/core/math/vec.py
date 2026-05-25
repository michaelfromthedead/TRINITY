"""Vector types: Vec2, Vec3, Vec4."""

from __future__ import annotations

import math

from engine.core.constants import MATH_EPSILON as _EPSILON


def nearly_equal(a: float, b: float, eps: float = _EPSILON) -> bool:
    return abs(a - b) <= eps


class Vec2:
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vec2:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vec2:
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return NotImplemented
        return nearly_equal(self.x, other.x) and nearly_equal(self.y, other.y)

    def __repr__(self) -> str:
        return f"Vec2({self.x}, {self.y})"

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> Vec2:
        ln = self.length()
        if ln < _EPSILON:
            return Vec2(0.0, 0.0)
        return self / ln

    def lerp(self, other: Vec2, t: float) -> Vec2:
        return self + (other - self) * t

    def distance(self, other: Vec2) -> float:
        return (self - other).length()

    def min(self, other: Vec2) -> Vec2:
        return Vec2(min(self.x, other.x), min(self.y, other.y))

    def max(self, other: Vec2) -> Vec2:
        return Vec2(max(self.x, other.x), max(self.y, other.y))

    def clamp(self, lo: Vec2, hi: Vec2) -> Vec2:
        return self.max(lo).min(hi)

    @staticmethod
    def zero() -> Vec2:
        return Vec2(0.0, 0.0)

    @staticmethod
    def one() -> Vec2:
        return Vec2(1.0, 1.0)

    @staticmethod
    def unit_x() -> Vec2:
        return Vec2(1.0, 0.0)

    @staticmethod
    def unit_y() -> Vec2:
        return Vec2(0.0, 1.0)


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vec3:
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return (
            nearly_equal(self.x, other.x)
            and nearly_equal(self.y, other.y)
            and nearly_equal(self.z, other.z)
        )

    def __repr__(self) -> str:
        return f"Vec3({self.x}, {self.y}, {self.z})"

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> Vec3:
        ln = self.length()
        if ln < _EPSILON:
            return Vec3(0.0, 0.0, 0.0)
        return self / ln

    def lerp(self, other: Vec3, t: float) -> Vec3:
        return self + (other - self) * t

    def distance(self, other: Vec3) -> float:
        return (self - other).length()

    def reflect(self, normal: Vec3) -> Vec3:
        n = normal.normalized()
        return self - n * (2.0 * self.dot(n))

    @staticmethod
    def zero() -> Vec3:
        return Vec3(0.0, 0.0, 0.0)

    @staticmethod
    def one() -> Vec3:
        return Vec3(1.0, 1.0, 1.0)

    @staticmethod
    def unit_x() -> Vec3:
        return Vec3(1.0, 0.0, 0.0)

    @staticmethod
    def unit_y() -> Vec3:
        return Vec3(0.0, 1.0, 0.0)

    @staticmethod
    def unit_z() -> Vec3:
        return Vec3(0.0, 0.0, 1.0)

    @staticmethod
    def up() -> Vec3:
        return Vec3(0.0, 1.0, 0.0)

    @staticmethod
    def forward() -> Vec3:
        return Vec3(0.0, 0.0, -1.0)

    @staticmethod
    def right() -> Vec3:
        return Vec3(1.0, 0.0, 0.0)


class Vec4:
    __slots__ = ("x", "y", "z", "w")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def __add__(self, other: Vec4) -> Vec4:
        return Vec4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self, other: Vec4) -> Vec4:
        return Vec4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def __mul__(self, scalar: float) -> Vec4:
        return Vec4(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)

    def __rmul__(self, scalar: float) -> Vec4:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vec4:
        return Vec4(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)

    def __neg__(self) -> Vec4:
        return Vec4(-self.x, -self.y, -self.z, -self.w)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec4):
            return NotImplemented
        return (
            nearly_equal(self.x, other.x)
            and nearly_equal(self.y, other.y)
            and nearly_equal(self.z, other.z)
            and nearly_equal(self.w, other.w)
        )

    def __repr__(self) -> str:
        return f"Vec4({self.x}, {self.y}, {self.z}, {self.w})"

    def dot(self, other: Vec4) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> Vec4:
        ln = self.length()
        if ln < _EPSILON:
            return Vec4(0.0, 0.0, 0.0, 0.0)
        return self / ln

    def lerp(self, other: Vec4, t: float) -> Vec4:
        return self + (other - self) * t

    @property
    def xyz(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)

    def perspective_divide(self) -> Vec3:
        import warnings
        if abs(self.w) < _EPSILON:
            warnings.warn("Vec4.perspective_divide: w near zero, division would produce infinity")
            return Vec3(self.x, self.y, self.z)
        return Vec3(self.x / self.w, self.y / self.w, self.z / self.w)

    def min(self, other: Vec4) -> Vec4:
        return Vec4(
            min(self.x, other.x), min(self.y, other.y),
            min(self.z, other.z), min(self.w, other.w),
        )

    def max(self, other: Vec4) -> Vec4:
        return Vec4(
            max(self.x, other.x), max(self.y, other.y),
            max(self.z, other.z), max(self.w, other.w),
        )

    def clamp(self, lo: Vec4, hi: Vec4) -> Vec4:
        return self.max(lo).min(hi)

    @staticmethod
    def zero() -> Vec4:
        return Vec4(0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def one() -> Vec4:
        return Vec4(1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def unit_x() -> Vec4:
        return Vec4(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def unit_y() -> Vec4:
        return Vec4(0.0, 1.0, 0.0, 0.0)

    @staticmethod
    def unit_z() -> Vec4:
        return Vec4(0.0, 0.0, 1.0, 0.0)

    @staticmethod
    def unit_w() -> Vec4:
        return Vec4(0.0, 0.0, 0.0, 1.0)
