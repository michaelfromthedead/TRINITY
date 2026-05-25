"""Shared math utilities for XR module.

This module provides common mathematical operations for rotation and
quaternion conversion that are used across multiple XR components.
"""

from __future__ import annotations

import math
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.core.math.vec import Vec3
    from engine.core.math.quat import Quat


def rotation_from_direction(forward: "Vec3", up: "Vec3" = None) -> "Quat":
    """Create a quaternion rotation from a forward direction vector.

    Builds an orthonormal basis from the forward direction and converts
    the resulting rotation matrix to a quaternion.

    Args:
        forward: The forward direction vector (will be normalized)
        up: Optional up vector, defaults to world up (0, 1, 0)

    Returns:
        Quaternion representing the rotation to look in that direction
    """
    from engine.core.math.vec import Vec3
    from engine.core.math.quat import Quat

    forward = forward.normalized()

    if up is None:
        up = Vec3.up()

    # Handle case where forward is parallel to up
    if abs(forward.dot(up)) > 0.999:
        up = Vec3.forward() if forward.y > 0 else Vec3(0, 0, 1)

    # Build orthonormal basis
    right = up.cross(forward).normalized()
    up = forward.cross(right).normalized()

    return rotation_from_axes(forward, up, right)


def rotation_from_axes(forward: "Vec3", up: "Vec3", right: "Vec3") -> "Quat":
    """Build a quaternion from orthonormal axes.

    Converts a rotation matrix (represented by its column vectors) to
    a quaternion using the Shepperd method for numerical stability.

    Args:
        forward: Forward direction (-Z in local space)
        up: Up direction (Y in local space)
        right: Right direction (X in local space)

    Returns:
        Normalized rotation quaternion
    """
    from engine.core.math.quat import Quat

    # Build rotation matrix components
    # Note: forward is negated because in OpenGL/typical 3D, forward is -Z
    m00, m01, m02 = right.x, up.x, -forward.x
    m10, m11, m12 = right.y, up.y, -forward.y
    m20, m21, m22 = right.z, up.z, -forward.z

    trace = m00 + m11 + m22

    # Use Shepperd's method for numerical stability
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return Quat(x, y, z, w).normalized()


def multiply_quaternions(
    q1: Tuple[float, float, float, float],
    q2: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Multiply two quaternions represented as tuples.

    For use with tuple-based quaternion representations common in
    rendering code. Prefer using Quat.__mul__ when working with
    Quat objects directly.

    Args:
        q1: First quaternion as (x, y, z, w) tuple
        q2: Second quaternion as (x, y, z, w) tuple

    Returns:
        Result quaternion as (x, y, z, w) tuple
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    )


def quaternion_to_tuple(q: "Quat") -> Tuple[float, float, float, float]:
    """Convert a Quat object to a tuple representation.

    Args:
        q: Quat object

    Returns:
        Tuple of (x, y, z, w)
    """
    return (q.x, q.y, q.z, q.w)


def tuple_to_quaternion(t: Tuple[float, float, float, float]) -> "Quat":
    """Convert a tuple to a Quat object.

    Args:
        t: Tuple of (x, y, z, w)

    Returns:
        Quat object
    """
    from engine.core.math.quat import Quat
    return Quat(t[0], t[1], t[2], t[3])
