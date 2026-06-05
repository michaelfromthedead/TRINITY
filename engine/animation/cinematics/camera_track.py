"""
Cinematic Camera Track System.

Provides camera animation with:
- Position, rotation, FOV keyframe animation
- Spline interpolation (Catmull-Rom, cubic Bezier)
- Blend in/out from gameplay camera
- Look-at targets with constraint solving
- @camera_track decorator for registration

Example usage:
    # Create a camera track with keyframes
    track = CameraTrack(
        id="intro_flyover",
        interpolation=InterpolationMode.CATMULL_ROM,
    )
    track.add_keyframe(CameraKeyframe(
        time=0.0,
        position=(0.0, 10.0, -20.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        fov=60.0,
    ))
    track.add_keyframe(CameraKeyframe(
        time=2.0,
        position=(0.0, 5.0, 0.0),
        rotation=(0.1, 0.0, 0.0, 0.995),
        fov=75.0,
    ))

    # Use decorator
    @camera_track(id="boss_intro", blend_in=0.5, blend_out=0.5)
    class BossIntroCameraTrack:
        keyframes = [...]
"""

from __future__ import annotations

import math
import functools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

# Type aliases
Vec3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (x, y, z, w)

T = TypeVar("T")

# Epsilon for floating point comparisons
_EPSILON: float = 1e-9


class InterpolationMode(Enum):
    """Interpolation modes for camera path."""

    LINEAR = auto()
    CATMULL_ROM = auto()
    CUBIC_BEZIER = auto()


class BlendState(Enum):
    """Camera blend state."""

    INACTIVE = auto()
    BLENDING_IN = auto()
    ACTIVE = auto()
    BLENDING_OUT = auto()
    COMPLETE = auto()


# =============================================================================
# Vector and Quaternion Math
# =============================================================================


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    """Add two vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    """Subtract two vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_scale(v: Vec3, s: float) -> Vec3:
    """Scale a vector."""
    return (v[0] * s, v[1] * s, v[2] * s)


def vec3_length(v: Vec3) -> float:
    """Get vector length."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_normalize(v: Vec3) -> Vec3:
    """Normalize a vector."""
    length = vec3_length(v)
    if length < _EPSILON:
        return (0.0, 0.0, 1.0)
    inv_length = 1.0 / length
    return (v[0] * inv_length, v[1] * inv_length, v[2] * inv_length)


def vec3_dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec3_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    """Linear interpolation between vectors."""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def quat_identity() -> Quaternion:
    """Identity quaternion (no rotation)."""
    return (0.0, 0.0, 0.0, 1.0)


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Multiply two quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_normalize(q: Quaternion) -> Quaternion:
    """Normalize a quaternion."""
    length = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if length < _EPSILON:
        return quat_identity()
    inv_length = 1.0 / length
    return (q[0] * inv_length, q[1] * inv_length, q[2] * inv_length, q[3] * inv_length)


def quat_slerp(a: Quaternion, b: Quaternion, t: float) -> Quaternion:
    """Spherical linear interpolation between quaternions."""
    # Compute cosine of angle
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

    # If negative, negate one quaternion to take shorter path
    b_adj = b
    if dot < 0:
        b_adj = (-b[0], -b[1], -b[2], -b[3])
        dot = -dot

    # Clamp dot product
    dot = min(dot, 1.0)

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = (
            a[0] + (b_adj[0] - a[0]) * t,
            a[1] + (b_adj[1] - a[1]) * t,
            a[2] + (b_adj[2] - a[2]) * t,
            a[3] + (b_adj[3] - a[3]) * t,
        )
        return quat_normalize(result)

    # Compute slerp
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (
        a[0] * s0 + b_adj[0] * s1,
        a[1] * s0 + b_adj[1] * s1,
        a[2] * s0 + b_adj[2] * s1,
        a[3] * s0 + b_adj[3] * s1,
    )


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quaternion:
    """Create quaternion from axis and angle (radians)."""
    axis = vec3_normalize(axis)
    half_angle = angle * 0.5
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)
    return (
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        cos_half,
    )


def quat_look_at(forward: Vec3, up: Vec3 = (0.0, 1.0, 0.0)) -> Quaternion:
    """Create rotation quaternion to look in a direction."""
    forward = vec3_normalize(forward)

    # Handle case where forward is parallel to up
    if abs(vec3_dot(forward, up)) > 0.999:
        up = (1.0, 0.0, 0.0) if abs(forward[1]) > 0.9 else (0.0, 1.0, 0.0)

    right = vec3_normalize(vec3_cross(up, forward))
    up = vec3_cross(forward, right)

    # Build rotation matrix and convert to quaternion
    m00, m01, m02 = right
    m10, m11, m12 = up
    m20, m21, m22 = forward

    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m12 - m21) * s
        y = (m20 - m02) * s
        z = (m01 - m10) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m12 - m21) / s
        x = 0.25 * s
        y = (m10 + m01) / s
        z = (m20 + m02) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m20 - m02) / s
        x = (m10 + m01) / s
        y = 0.25 * s
        z = (m21 + m12) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m01 - m10) / s
        x = (m20 + m02) / s
        y = (m21 + m12) / s
        z = 0.25 * s

    return quat_normalize((x, y, z, w))


def quat_rotate_vector(q: Quaternion, v: Vec3) -> Vec3:
    """Rotate a vector by a quaternion."""
    qx, qy, qz, qw = q
    vx, vy, vz = v

    # q * v * q^-1 computed directly
    ix = qw * vx + qy * vz - qz * vy
    iy = qw * vy + qz * vx - qx * vz
    iz = qw * vz + qx * vy - qy * vx
    iw = -qx * vx - qy * vy - qz * vz

    return (
        ix * qw + iw * (-qx) + iy * (-qz) - iz * (-qy),
        iy * qw + iw * (-qy) + iz * (-qx) - ix * (-qz),
        iz * qw + iw * (-qz) + ix * (-qy) - iy * (-qx),
    )


# =============================================================================
# Spline Interpolation
# =============================================================================


def catmull_rom_interpolate(
    p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float, tension: float = 0.5
) -> Vec3:
    """
    Catmull-Rom spline interpolation.

    Args:
        p0: Control point before start
        p1: Start point
        p2: End point
        p3: Control point after end
        t: Parameter in [0, 1]
        tension: Spline tension (0.5 = standard Catmull-Rom)

    Returns:
        Interpolated position
    """
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom basis functions with tension
    s = (1.0 - tension) / 2.0

    b0 = -s * t3 + 2.0 * s * t2 - s * t
    b1 = (2.0 - s) * t3 + (s - 3.0) * t2 + 1.0
    b2 = (s - 2.0) * t3 + (3.0 - 2.0 * s) * t2 + s * t
    b3 = s * t3 - s * t2

    return (
        p0[0] * b0 + p1[0] * b1 + p2[0] * b2 + p3[0] * b3,
        p0[1] * b0 + p1[1] * b1 + p2[1] * b2 + p3[1] * b3,
        p0[2] * b0 + p1[2] * b1 + p2[2] * b2 + p3[2] * b3,
    )


def catmull_rom_tangent(
    p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float, tension: float = 0.5
) -> Vec3:
    """
    Calculate tangent at point on Catmull-Rom spline.

    Returns the derivative (direction) at parameter t.
    """
    t2 = t * t
    s = (1.0 - tension) / 2.0

    # Derivatives of basis functions
    db0 = -3.0 * s * t2 + 4.0 * s * t - s
    db1 = 3.0 * (2.0 - s) * t2 + 2.0 * (s - 3.0) * t
    db2 = 3.0 * (s - 2.0) * t2 + 2.0 * (3.0 - 2.0 * s) * t + s
    db3 = 3.0 * s * t2 - 2.0 * s * t

    return (
        p0[0] * db0 + p1[0] * db1 + p2[0] * db2 + p3[0] * db3,
        p0[1] * db0 + p1[1] * db1 + p2[1] * db2 + p3[1] * db3,
        p0[2] * db0 + p1[2] * db1 + p2[2] * db2 + p3[2] * db3,
    )


def cubic_bezier_interpolate(
    p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float
) -> Vec3:
    """
    Cubic Bezier spline interpolation.

    Args:
        p0: Start point
        p1: First control point
        p2: Second control point
        p3: End point
        t: Parameter in [0, 1]

    Returns:
        Interpolated position
    """
    u = 1.0 - t
    u2 = u * u
    u3 = u2 * u
    t2 = t * t
    t3 = t2 * t

    return (
        u3 * p0[0] + 3.0 * u2 * t * p1[0] + 3.0 * u * t2 * p2[0] + t3 * p3[0],
        u3 * p0[1] + 3.0 * u2 * t * p1[1] + 3.0 * u * t2 * p2[1] + t3 * p3[1],
        u3 * p0[2] + 3.0 * u2 * t * p1[2] + 3.0 * u * t2 * p2[2] + t3 * p3[2],
    )


def cubic_bezier_tangent(
    p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float
) -> Vec3:
    """
    Calculate tangent at point on cubic Bezier curve.

    Returns the derivative (direction) at parameter t.
    """
    u = 1.0 - t
    u2 = u * u
    t2 = t * t

    # Derivative of cubic Bezier
    return (
        3.0 * u2 * (p1[0] - p0[0]) + 6.0 * u * t * (p2[0] - p1[0]) + 3.0 * t2 * (p3[0] - p2[0]),
        3.0 * u2 * (p1[1] - p0[1]) + 6.0 * u * t * (p2[1] - p1[1]) + 3.0 * t2 * (p3[1] - p2[1]),
        3.0 * u2 * (p1[2] - p0[2]) + 6.0 * u * t * (p2[2] - p1[2]) + 3.0 * t2 * (p3[2] - p2[2]),
    )


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between floats."""
    return a + (b - a) * t


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BezierControlPoint:
    """Control point for cubic Bezier curves."""

    in_tangent: Vec3 = (0.0, 0.0, 0.0)  # Tangent coming into this point
    out_tangent: Vec3 = (0.0, 0.0, 0.0)  # Tangent going out of this point


@dataclass
class CameraKeyframe:
    """
    A single camera keyframe with position, rotation, and FOV.

    Attributes:
        time: Timestamp in seconds
        position: World position (x, y, z)
        rotation: Rotation as quaternion (x, y, z, w)
        fov: Field of view in degrees
        bezier_control: Optional Bezier control points for this keyframe
    """

    time: float
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    fov: float = 60.0
    bezier_control: Optional[BezierControlPoint] = None

    def __post_init__(self) -> None:
        """Validate keyframe data."""
        if self.time < 0:
            raise ValueError("time must be >= 0")
        if not (1.0 <= self.fov <= 179.0):
            raise ValueError("fov must be in [1, 179] degrees")
        # Normalize rotation
        self.rotation = quat_normalize(self.rotation)


@dataclass
class LookAtTarget:
    """
    Target for camera look-at constraint.

    Attributes:
        position: World position to look at (or None if using object tracking)
        object_id: Optional object ID for dynamic tracking
        offset: Offset from target position
        weight: Constraint weight (0-1)
        up_vector: Up vector for constraint
    """

    position: Optional[Vec3] = None
    object_id: Optional[str] = None
    offset: Vec3 = (0.0, 0.0, 0.0)
    weight: float = 1.0
    up_vector: Vec3 = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        """Validate look-at target."""
        if self.position is None and self.object_id is None:
            raise ValueError("Either position or object_id must be provided")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("weight must be in [0, 1]")

    def get_target_position(
        self, object_resolver: Optional[Callable[[str], Vec3]] = None
    ) -> Optional[Vec3]:
        """
        Get the current target position.

        Args:
            object_resolver: Optional callback to resolve object_id to position

        Returns:
            World position or None if cannot resolve
        """
        if self.position is not None:
            return vec3_add(self.position, self.offset)

        if self.object_id is not None and object_resolver is not None:
            pos = object_resolver(self.object_id)
            if pos is not None:
                return vec3_add(pos, self.offset)

        return None


@dataclass
class CameraState:
    """
    Current camera state for interpolation and blending.

    Represents the fully computed camera transform at a point in time.
    """

    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    fov: float = 60.0

    def blend(self, other: "CameraState", weight: float) -> "CameraState":
        """
        Blend between this state and another.

        Args:
            other: Target camera state
            weight: Blend weight (0=self, 1=other)

        Returns:
            Blended camera state
        """
        return CameraState(
            position=vec3_lerp(self.position, other.position, weight),
            rotation=quat_slerp(self.rotation, other.rotation, weight),
            fov=lerp(self.fov, other.fov, weight),
        )


# =============================================================================
# Camera Track
# =============================================================================


class CameraTrack:
    """
    Camera animation track with keyframe interpolation.

    Supports multiple interpolation modes and look-at constraints.
    """

    def __init__(
        self,
        id: str,
        interpolation: InterpolationMode = InterpolationMode.CATMULL_ROM,
        blend_in: float = 0.5,
        blend_out: float = 0.5,
        tension: float = 0.5,
        loop: bool = False,
    ) -> None:
        """
        Create a camera track.

        Args:
            id: Unique identifier for this track
            interpolation: Interpolation mode for position
            blend_in: Blend in duration in seconds
            blend_out: Blend out duration in seconds
            tension: Catmull-Rom tension (only for CATMULL_ROM mode)
            loop: Whether to loop the animation
        """
        if not id:
            raise ValueError("id must be a non-empty string")
        if blend_in < 0:
            raise ValueError("blend_in must be >= 0")
        if blend_out < 0:
            raise ValueError("blend_out must be >= 0")

        self._id = id
        self._interpolation = interpolation
        self._blend_in = blend_in
        self._blend_out = blend_out
        self._tension = tension
        self._loop = loop

        self._keyframes: List[CameraKeyframe] = []
        self._look_at_target: Optional[LookAtTarget] = None
        self._sorted = False

        # Playback state
        self._elapsed: float = 0.0
        self._blend_state = BlendState.INACTIVE
        self._blend_weight: float = 0.0
        self._gameplay_camera: Optional[CameraState] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def id(self) -> str:
        """Track ID."""
        return self._id

    @property
    def interpolation(self) -> InterpolationMode:
        """Interpolation mode."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: InterpolationMode) -> None:
        """Set interpolation mode."""
        self._interpolation = value

    @property
    def blend_in(self) -> float:
        """Blend in duration."""
        return self._blend_in

    @blend_in.setter
    def blend_in(self, value: float) -> None:
        """Set blend in duration."""
        if value < 0:
            raise ValueError("blend_in must be >= 0")
        self._blend_in = value

    @property
    def blend_out(self) -> float:
        """Blend out duration."""
        return self._blend_out

    @blend_out.setter
    def blend_out(self, value: float) -> None:
        """Set blend out duration."""
        if value < 0:
            raise ValueError("blend_out must be >= 0")
        self._blend_out = value

    @property
    def duration(self) -> float:
        """Total track duration in seconds."""
        if not self._keyframes:
            return 0.0
        self._ensure_sorted()
        return self._keyframes[-1].time

    @property
    def keyframe_count(self) -> int:
        """Number of keyframes."""
        return len(self._keyframes)

    @property
    def elapsed(self) -> float:
        """Current elapsed time."""
        return self._elapsed

    @property
    def progress(self) -> float:
        """Current progress (0-1)."""
        dur = self.duration
        if dur < _EPSILON:
            return 0.0
        return min(1.0, self._elapsed / dur)

    @property
    def blend_state(self) -> BlendState:
        """Current blend state."""
        return self._blend_state

    @property
    def blend_weight(self) -> float:
        """Current blend weight (0=gameplay, 1=track)."""
        return self._blend_weight

    @property
    def is_playing(self) -> bool:
        """Whether the track is currently active."""
        return self._blend_state not in (BlendState.INACTIVE, BlendState.COMPLETE)

    @property
    def look_at_target(self) -> Optional[LookAtTarget]:
        """Look-at target."""
        return self._look_at_target

    @look_at_target.setter
    def look_at_target(self, target: Optional[LookAtTarget]) -> None:
        """Set look-at target."""
        self._look_at_target = target

    # =========================================================================
    # Keyframe Management
    # =========================================================================

    def add_keyframe(self, keyframe: CameraKeyframe) -> "CameraTrack":
        """
        Add a keyframe to the track.

        Args:
            keyframe: The keyframe to add

        Returns:
            Self for chaining
        """
        self._keyframes.append(keyframe)
        self._sorted = False
        return self

    def remove_keyframe(self, index: int) -> bool:
        """
        Remove a keyframe by index.

        Args:
            index: Index of keyframe to remove

        Returns:
            True if removed, False if invalid index
        """
        if 0 <= index < len(self._keyframes):
            self._keyframes.pop(index)
            return True
        return False

    def clear_keyframes(self) -> "CameraTrack":
        """
        Remove all keyframes.

        Returns:
            Self for chaining
        """
        self._keyframes.clear()
        self._sorted = False
        return self

    def get_keyframe(self, index: int) -> Optional[CameraKeyframe]:
        """Get keyframe by index."""
        self._ensure_sorted()
        if 0 <= index < len(self._keyframes):
            return self._keyframes[index]
        return None

    def _ensure_sorted(self) -> None:
        """Sort keyframes by time if needed."""
        if not self._sorted:
            self._keyframes.sort(key=lambda k: k.time)
            self._sorted = True

    # =========================================================================
    # Interpolation
    # =========================================================================

    def _find_keyframe_segment(
        self, time: float
    ) -> Tuple[int, int, float]:
        """
        Find the keyframe segment containing the given time.

        Returns:
            (before_index, after_index, segment_t)
        """
        self._ensure_sorted()

        if not self._keyframes:
            return (-1, -1, 0.0)

        if len(self._keyframes) == 1:
            return (0, 0, 0.0)

        # Clamp time to track duration
        if time <= self._keyframes[0].time:
            return (0, 0, 0.0)
        if time >= self._keyframes[-1].time:
            return (len(self._keyframes) - 1, len(self._keyframes) - 1, 0.0)

        # Binary search for segment
        for i in range(len(self._keyframes) - 1):
            k0 = self._keyframes[i]
            k1 = self._keyframes[i + 1]
            if k0.time <= time <= k1.time:
                segment_duration = k1.time - k0.time
                if segment_duration < _EPSILON:
                    return (i, i + 1, 0.0)
                segment_t = (time - k0.time) / segment_duration
                return (i, i + 1, segment_t)

        return (len(self._keyframes) - 1, len(self._keyframes) - 1, 0.0)

    def _get_control_points(
        self, idx0: int, idx1: int
    ) -> Tuple[Vec3, Vec3, Vec3, Vec3]:
        """
        Get four control points for spline interpolation.

        For Catmull-Rom, this includes points before and after the segment.
        """
        n = len(self._keyframes)

        # Get indices for p0, p1, p2, p3
        i_p0 = max(0, idx0 - 1)
        i_p1 = idx0
        i_p2 = idx1
        i_p3 = min(n - 1, idx1 + 1)

        return (
            self._keyframes[i_p0].position,
            self._keyframes[i_p1].position,
            self._keyframes[i_p2].position,
            self._keyframes[i_p3].position,
        )

    def _interpolate_position(self, idx0: int, idx1: int, t: float) -> Vec3:
        """Interpolate position based on interpolation mode."""
        if idx0 == idx1:
            return self._keyframes[idx0].position

        k0 = self._keyframes[idx0]
        k1 = self._keyframes[idx1]

        if self._interpolation == InterpolationMode.LINEAR:
            return vec3_lerp(k0.position, k1.position, t)

        elif self._interpolation == InterpolationMode.CATMULL_ROM:
            p0, p1, p2, p3 = self._get_control_points(idx0, idx1)
            return catmull_rom_interpolate(p0, p1, p2, p3, t, self._tension)

        elif self._interpolation == InterpolationMode.CUBIC_BEZIER:
            # Use Bezier control points if available, otherwise auto-generate
            if k0.bezier_control is not None and k1.bezier_control is not None:
                p0 = k0.position
                p1 = vec3_add(k0.position, k0.bezier_control.out_tangent)
                p2 = vec3_add(k1.position, k1.bezier_control.in_tangent)
                p3 = k1.position
            else:
                # Auto-generate control points as 1/3 distance
                diff = vec3_sub(k1.position, k0.position)
                p0 = k0.position
                p1 = vec3_add(k0.position, vec3_scale(diff, 1.0 / 3.0))
                p2 = vec3_add(k0.position, vec3_scale(diff, 2.0 / 3.0))
                p3 = k1.position
            return cubic_bezier_interpolate(p0, p1, p2, p3, t)

        return k0.position

    def _interpolate_rotation(self, idx0: int, idx1: int, t: float) -> Quaternion:
        """Interpolate rotation using slerp."""
        if idx0 == idx1:
            return self._keyframes[idx0].rotation
        return quat_slerp(
            self._keyframes[idx0].rotation,
            self._keyframes[idx1].rotation,
            t,
        )

    def _interpolate_fov(self, idx0: int, idx1: int, t: float) -> float:
        """Interpolate FOV linearly."""
        if idx0 == idx1:
            return self._keyframes[idx0].fov
        return lerp(self._keyframes[idx0].fov, self._keyframes[idx1].fov, t)

    def sample(
        self,
        time: float,
        object_resolver: Optional[Callable[[str], Vec3]] = None,
    ) -> CameraState:
        """
        Sample the camera state at a specific time.

        Args:
            time: Time in seconds
            object_resolver: Optional callback to resolve object IDs for look-at

        Returns:
            Camera state at the given time
        """
        if not self._keyframes:
            return CameraState()

        idx0, idx1, t = self._find_keyframe_segment(time)

        if idx0 < 0:
            return CameraState()

        position = self._interpolate_position(idx0, idx1, t)
        rotation = self._interpolate_rotation(idx0, idx1, t)
        fov = self._interpolate_fov(idx0, idx1, t)

        # Apply look-at constraint if present
        if self._look_at_target is not None:
            target_pos = self._look_at_target.get_target_position(object_resolver)
            if target_pos is not None:
                look_rotation = self._solve_look_at(position, target_pos)
                rotation = quat_slerp(
                    rotation, look_rotation, self._look_at_target.weight
                )

        return CameraState(position=position, rotation=rotation, fov=fov)

    def _solve_look_at(self, from_pos: Vec3, to_pos: Vec3) -> Quaternion:
        """
        Solve look-at constraint.

        Args:
            from_pos: Camera position
            to_pos: Target position

        Returns:
            Rotation to look at target
        """
        direction = vec3_sub(to_pos, from_pos)
        if vec3_length(direction) < _EPSILON:
            return quat_identity()

        up = (
            self._look_at_target.up_vector
            if self._look_at_target
            else (0.0, 1.0, 0.0)
        )
        return quat_look_at(direction, up)

    # =========================================================================
    # Playback Control
    # =========================================================================

    def start(self, gameplay_camera: Optional[CameraState] = None) -> "CameraTrack":
        """
        Start playback with optional gameplay camera for blending.

        Args:
            gameplay_camera: Current gameplay camera state for blend transition

        Returns:
            Self for chaining
        """
        self._elapsed = 0.0
        self._gameplay_camera = gameplay_camera
        self._blend_weight = 0.0 if self._blend_in > 0 else 1.0
        self._blend_state = BlendState.BLENDING_IN if self._blend_in > 0 else BlendState.ACTIVE
        return self

    def stop(self) -> "CameraTrack":
        """
        Stop playback.

        Returns:
            Self for chaining
        """
        self._blend_state = BlendState.INACTIVE
        self._blend_weight = 0.0
        self._elapsed = 0.0
        return self

    def pause(self) -> "CameraTrack":
        """
        Pause at current position (blend state preserved).

        Returns:
            Self for chaining
        """
        # Simply stop updating elapsed time - state is preserved
        return self

    def seek(self, time: float) -> "CameraTrack":
        """
        Seek to specific time.

        Args:
            time: Time in seconds

        Returns:
            Self for chaining
        """
        self._elapsed = max(0.0, min(time, self.duration))
        return self

    def update(
        self,
        delta_time: float,
        gameplay_camera: Optional[CameraState] = None,
        object_resolver: Optional[Callable[[str], Vec3]] = None,
    ) -> CameraState:
        """
        Update the track and return the blended camera state.

        Args:
            delta_time: Time since last update
            gameplay_camera: Current gameplay camera for blending
            object_resolver: Optional callback to resolve object IDs

        Returns:
            Blended camera state
        """
        if delta_time <= 0:
            return self.sample(self._elapsed, object_resolver)

        if self._blend_state == BlendState.INACTIVE:
            return gameplay_camera or CameraState()

        if self._blend_state == BlendState.COMPLETE:
            return gameplay_camera or CameraState()

        # Update gameplay camera reference
        if gameplay_camera is not None:
            self._gameplay_camera = gameplay_camera

        # Update elapsed time
        self._elapsed += delta_time
        duration = self.duration

        # Handle blend states
        if self._blend_state == BlendState.BLENDING_IN:
            if self._blend_in > 0:
                self._blend_weight = min(1.0, self._elapsed / self._blend_in)
            else:
                self._blend_weight = 1.0

            if self._blend_weight >= 1.0:
                self._blend_state = BlendState.ACTIVE

        elif self._blend_state == BlendState.ACTIVE:
            self._blend_weight = 1.0

            # Check if we should start blending out
            if duration > 0 and self._elapsed >= duration - self._blend_out:
                self._blend_state = BlendState.BLENDING_OUT

        elif self._blend_state == BlendState.BLENDING_OUT:
            if self._blend_out > 0:
                blend_out_elapsed = self._elapsed - (duration - self._blend_out)
                self._blend_weight = max(0.0, 1.0 - blend_out_elapsed / self._blend_out)
            else:
                self._blend_weight = 0.0

            if self._blend_weight <= 0.0:
                if self._loop:
                    self._elapsed = 0.0
                    self._blend_state = BlendState.BLENDING_IN
                    self._blend_weight = 0.0
                else:
                    self._blend_state = BlendState.COMPLETE

        # Sample track state
        track_state = self.sample(self._elapsed, object_resolver)

        # Blend with gameplay camera
        if self._gameplay_camera is not None and self._blend_weight < 1.0:
            return self._gameplay_camera.blend(track_state, self._blend_weight)

        return track_state


# =============================================================================
# Camera Track Manager
# =============================================================================


class CameraTrackManager:
    """
    Manages multiple camera tracks.

    Provides centralized update, lookup, and priority-based track selection.
    """

    def __init__(self) -> None:
        """Create a camera track manager."""
        self._tracks: Dict[str, CameraTrack] = {}
        self._active_track: Optional[str] = None
        self._track_priority: Dict[str, int] = {}

    @property
    def active_track_id(self) -> Optional[str]:
        """ID of the currently active track."""
        return self._active_track

    @property
    def active_track(self) -> Optional[CameraTrack]:
        """The currently active track."""
        if self._active_track:
            return self._tracks.get(self._active_track)
        return None

    def register(
        self, track: CameraTrack, priority: int = 0
    ) -> None:
        """
        Register a track.

        Args:
            track: The camera track to register
            priority: Track priority (higher = plays first when multiple requested)
        """
        self._tracks[track.id] = track
        self._track_priority[track.id] = priority

    def unregister(self, track_id: str) -> bool:
        """
        Unregister a track by ID.

        Returns:
            True if removed, False if not found
        """
        if track_id in self._tracks:
            if self._active_track == track_id:
                self._active_track = None
            del self._tracks[track_id]
            self._track_priority.pop(track_id, None)
            return True
        return False

    def get(self, track_id: str) -> Optional[CameraTrack]:
        """Get a track by ID."""
        return self._tracks.get(track_id)

    def play(
        self,
        track_id: str,
        gameplay_camera: Optional[CameraState] = None,
    ) -> bool:
        """
        Start playing a track.

        Args:
            track_id: ID of the track to play
            gameplay_camera: Current gameplay camera for blending

        Returns:
            True if started, False if track not found
        """
        track = self._tracks.get(track_id)
        if track is None:
            return False

        # Stop current track if different
        if self._active_track and self._active_track != track_id:
            current = self._tracks.get(self._active_track)
            if current:
                current.stop()

        self._active_track = track_id
        track.start(gameplay_camera)
        return True

    def stop(self, track_id: Optional[str] = None) -> bool:
        """
        Stop a track (or the active track if no ID provided).

        Returns:
            True if stopped, False if not found
        """
        target_id = track_id or self._active_track
        if not target_id:
            return False

        track = self._tracks.get(target_id)
        if track is None:
            return False

        track.stop()
        if self._active_track == target_id:
            self._active_track = None
        return True

    def stop_all(self) -> None:
        """Stop all playing tracks."""
        for track in self._tracks.values():
            track.stop()
        self._active_track = None

    def update(
        self,
        delta_time: float,
        gameplay_camera: Optional[CameraState] = None,
        object_resolver: Optional[Callable[[str], Vec3]] = None,
    ) -> CameraState:
        """
        Update active track and return camera state.

        Args:
            delta_time: Time since last update
            gameplay_camera: Current gameplay camera
            object_resolver: Callback to resolve object IDs

        Returns:
            Camera state (from track if active, otherwise gameplay camera)
        """
        if self._active_track:
            track = self._tracks.get(self._active_track)
            if track:
                state = track.update(delta_time, gameplay_camera, object_resolver)

                # Clear active track if complete
                if track.blend_state == BlendState.COMPLETE:
                    self._active_track = None

                return state

        return gameplay_camera or CameraState()


# =============================================================================
# Global Registry
# =============================================================================

_camera_track_registry: Dict[str, type] = {}


def get_camera_track_registry() -> Dict[str, type]:
    """Get the global camera track registry."""
    return _camera_track_registry.copy()


# =============================================================================
# Decorator
# =============================================================================


def camera_track(
    id: Optional[str] = None,
    blend_in: float = 0.5,
    blend_out: float = 0.5,
    interpolation: InterpolationMode = InterpolationMode.CATMULL_ROM,
    tension: float = 0.5,
    loop: bool = False,
) -> Callable[[T], T]:
    """
    Decorator to mark a class as a camera track definition.

    Args:
        id: Track ID (defaults to class name)
        blend_in: Blend in duration in seconds
        blend_out: Blend out duration in seconds
        interpolation: Interpolation mode for position
        tension: Catmull-Rom tension (only for CATMULL_ROM mode)
        loop: Whether to loop the animation

    Returns:
        Decorated class

    Example:
        @camera_track(id="intro_camera", blend_in=1.0, blend_out=1.0)
        class IntroCameraTrack:
            keyframes = [
                CameraKeyframe(0.0, (0, 10, -20), (0, 0, 0, 1), 60.0),
                CameraKeyframe(5.0, (0, 5, 0), (0.1, 0, 0, 0.995), 75.0),
            ]
    """
    if blend_in < 0:
        raise ValueError("blend_in must be >= 0")
    if blend_out < 0:
        raise ValueError("blend_out must be >= 0")

    def decorator(cls: T) -> T:
        track_id = id or cls.__name__

        # Set attributes on the class
        cls._camera_track = True  # type: ignore
        cls._camera_track_id = track_id  # type: ignore
        cls._camera_track_blend_in = blend_in  # type: ignore
        cls._camera_track_blend_out = blend_out  # type: ignore
        cls._camera_track_interpolation = interpolation  # type: ignore
        cls._camera_track_tension = tension  # type: ignore
        cls._camera_track_loop = loop  # type: ignore

        # Track applied decorators
        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = []  # type: ignore
        cls._applied_decorators.append("camera_track")  # type: ignore

        # Track registries
        if not hasattr(cls, "_registries"):
            cls._registries = []  # type: ignore
        cls._registries.append("cinematics")  # type: ignore

        # Track tags
        if not hasattr(cls, "_tags"):
            cls._tags = {}  # type: ignore
        cls._tags["camera_track"] = True  # type: ignore
        cls._tags["camera_track_id"] = track_id  # type: ignore
        cls._tags["camera_track_blend_in"] = blend_in  # type: ignore
        cls._tags["camera_track_blend_out"] = blend_out  # type: ignore

        # Register in global registry
        _camera_track_registry[track_id] = cls  # type: ignore

        return cls

    return decorator


# =============================================================================
# Factory Functions
# =============================================================================


def create_camera_track(
    id: str,
    keyframes: List[CameraKeyframe],
    interpolation: InterpolationMode = InterpolationMode.CATMULL_ROM,
    blend_in: float = 0.5,
    blend_out: float = 0.5,
    tension: float = 0.5,
    loop: bool = False,
    look_at: Optional[LookAtTarget] = None,
) -> CameraTrack:
    """
    Factory function to create a camera track.

    Args:
        id: Track ID
        keyframes: List of keyframes
        interpolation: Interpolation mode
        blend_in: Blend in duration
        blend_out: Blend out duration
        tension: Catmull-Rom tension
        loop: Whether to loop
        look_at: Optional look-at target

    Returns:
        Configured CameraTrack instance
    """
    track = CameraTrack(
        id=id,
        interpolation=interpolation,
        blend_in=blend_in,
        blend_out=blend_out,
        tension=tension,
        loop=loop,
    )

    for kf in keyframes:
        track.add_keyframe(kf)

    if look_at:
        track.look_at_target = look_at

    return track


def create_track_from_class(cls: type) -> CameraTrack:
    """
    Create a CameraTrack instance from a decorated class.

    Args:
        cls: Class decorated with @camera_track

    Returns:
        CameraTrack instance

    Raises:
        ValueError: If class is not a camera track
    """
    if not getattr(cls, "_camera_track", False):
        raise ValueError(f"{cls.__name__} is not a camera track")

    track_id = getattr(cls, "_camera_track_id", cls.__name__)
    blend_in = getattr(cls, "_camera_track_blend_in", 0.5)
    blend_out = getattr(cls, "_camera_track_blend_out", 0.5)
    interpolation = getattr(cls, "_camera_track_interpolation", InterpolationMode.CATMULL_ROM)
    tension = getattr(cls, "_camera_track_tension", 0.5)
    loop = getattr(cls, "_camera_track_loop", False)

    track = CameraTrack(
        id=track_id,
        interpolation=interpolation,
        blend_in=blend_in,
        blend_out=blend_out,
        tension=tension,
        loop=loop,
    )

    # Add keyframes from class attribute
    keyframes = getattr(cls, "keyframes", [])
    for kf in keyframes:
        if isinstance(kf, CameraKeyframe):
            track.add_keyframe(kf)
        elif isinstance(kf, (list, tuple)):
            # Support simple tuple format: (time, position, rotation, fov)
            if len(kf) >= 2:
                kf_time = kf[0]
                kf_pos = kf[1] if len(kf) > 1 else (0.0, 0.0, 0.0)
                kf_rot = kf[2] if len(kf) > 2 else (0.0, 0.0, 0.0, 1.0)
                kf_fov = kf[3] if len(kf) > 3 else 60.0
                track.add_keyframe(CameraKeyframe(
                    time=kf_time,
                    position=kf_pos,
                    rotation=kf_rot,
                    fov=kf_fov,
                ))

    # Look-at target from class attribute
    look_at = getattr(cls, "look_at", None)
    if look_at is not None:
        if isinstance(look_at, LookAtTarget):
            track.look_at_target = look_at
        elif isinstance(look_at, dict):
            track.look_at_target = LookAtTarget(**look_at)

    return track


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums
    "InterpolationMode",
    "BlendState",
    # Data classes
    "BezierControlPoint",
    "CameraKeyframe",
    "CameraState",
    "LookAtTarget",
    # Main classes
    "CameraTrack",
    "CameraTrackManager",
    # Decorator
    "camera_track",
    # Factory functions
    "create_camera_track",
    "create_track_from_class",
    "get_camera_track_registry",
    # Interpolation functions
    "catmull_rom_interpolate",
    "catmull_rom_tangent",
    "cubic_bezier_interpolate",
    "cubic_bezier_tangent",
    # Vector math (exposed for testing/extension)
    "vec3_lerp",
    "quat_slerp",
    "quat_look_at",
]
