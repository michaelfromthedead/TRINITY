"""Geometric primitives: Ray, AABB, Sphere, Plane, Frustum."""

from __future__ import annotations

from engine.core.math.vec import Vec3


class Ray:
    __slots__ = ("origin", "direction")

    def __init__(self, origin: Vec3, direction: Vec3) -> None:
        self.origin = origin
        self.direction = direction.normalized()

    def point_at(self, t: float) -> Vec3:
        return self.origin + self.direction * t

    def __repr__(self) -> str:
        return f"Ray({self.origin}, {self.direction})"


class AABB:
    __slots__ = ("min", "max")

    def __init__(self, min: Vec3, max: Vec3) -> None:
        self.min = min
        self.max = max

    @property
    def center(self) -> Vec3:
        return (self.min + self.max) * 0.5

    @property
    def extents(self) -> Vec3:
        return (self.max - self.min) * 0.5

    def contains(self, point: Vec3) -> bool:
        return (
            self.min.x <= point.x <= self.max.x
            and self.min.y <= point.y <= self.max.y
            and self.min.z <= point.z <= self.max.z
        )

    def intersects(self, other: AABB) -> bool:
        return (
            self.min.x <= other.max.x and self.max.x >= other.min.x
            and self.min.y <= other.max.y and self.max.y >= other.min.y
            and self.min.z <= other.max.z and self.max.z >= other.min.z
        )

    def __repr__(self) -> str:
        return f"AABB({self.min}, {self.max})"


class Sphere:
    __slots__ = ("center", "radius")

    def __init__(self, center: Vec3, radius: float) -> None:
        self.center = center
        self.radius = float(radius)

    def contains(self, point: Vec3) -> bool:
        return self.center.distance(point) <= self.radius

    def intersects(self, other: Sphere) -> bool:
        return self.center.distance(other.center) <= self.radius + other.radius

    def __repr__(self) -> str:
        return f"Sphere({self.center}, {self.radius})"


class Plane:
    __slots__ = ("normal", "distance")

    def __init__(self, normal: Vec3, distance: float) -> None:
        self.normal = normal.normalized()
        self.distance = float(distance)

    def signed_distance(self, point: Vec3) -> float:
        return self.normal.dot(point) + self.distance

    def closest_point(self, point: Vec3) -> Vec3:
        d = self.signed_distance(point)
        return point - self.normal * d

    def __repr__(self) -> str:
        return f"Plane({self.normal}, {self.distance})"


class Frustum:
    __slots__ = ("planes",)

    def __init__(self, planes: list[Plane]) -> None:
        self.planes = planes

    def contains_point(self, point: Vec3) -> bool:
        return all(p.signed_distance(point) >= 0 for p in self.planes)

    def intersects_aabb(self, aabb: AABB) -> bool:
        for plane in self.planes:
            px = aabb.max.x if plane.normal.x >= 0 else aabb.min.x
            py = aabb.max.y if plane.normal.y >= 0 else aabb.min.y
            pz = aabb.max.z if plane.normal.z >= 0 else aabb.min.z
            if plane.signed_distance(Vec3(px, py, pz)) < 0:
                return False
        return True

    def __repr__(self) -> str:
        return f"Frustum({len(self.planes)} planes)"
