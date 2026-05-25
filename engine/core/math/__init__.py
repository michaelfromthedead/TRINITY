"""Math library: vectors, matrices, quaternions, transforms, geometry, interpolation."""

from engine.core.math.vec import Vec2, Vec3, Vec4
from engine.core.math.mat import Mat3, Mat4
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform, RigidTransform
from engine.core.math.geometry import Ray, AABB, Sphere, Plane, Frustum
from engine.core.math.interpolation import (
    lerp, inverse_lerp, remap, clamp, smoothstep, smootherstep,
    in_quad, out_quad, in_out_quad, in_cubic, out_cubic, in_out_cubic,
    SpringDamper,
)

__all__ = [
    "Vec2", "Vec3", "Vec4",
    "Mat3", "Mat4",
    "Quat",
    "Transform", "RigidTransform",
    "Ray", "AABB", "Sphere", "Plane", "Frustum",
    "lerp", "inverse_lerp", "remap", "clamp", "smoothstep", "smootherstep",
    "in_quad", "out_quad", "in_out_quad", "in_cubic", "out_cubic", "in_out_cubic",
    "SpringDamper",
]
