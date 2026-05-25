"""
Procedural Look-At Controller.

Provides head, neck, and eye bone rotation to face targets with smooth
interpolation, angle limits, and realistic eye tracking with saccades.

Usage:
    controller = LookAtController(
        head_bone=10,
        neck_bone=9,
        eye_bones=[11, 12]
    )
    modified_pose = controller.update(pose, target_position, dt)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Protocol

# Type aliases
Vec3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (x, y, z, w)


class Pose(Protocol):
    """Protocol for pose data - maps bone indices to transforms."""

    def get_bone_position(self, bone_index: int) -> Vec3:
        """Get world position of a bone."""
        ...

    def set_bone_position(self, bone_index: int, position: Vec3) -> None:
        """Set world position of a bone."""
        ...

    def get_bone_rotation(self, bone_index: int) -> Quaternion:
        """Get world rotation of a bone."""
        ...

    def set_bone_rotation(self, bone_index: int, rotation: Quaternion) -> None:
        """Set world rotation of a bone."""
        ...

    def get_bone_local_rotation(self, bone_index: int) -> Quaternion:
        """Get local rotation of a bone."""
        ...

    def set_bone_local_rotation(self, bone_index: int, rotation: Quaternion) -> None:
        """Set local rotation of a bone."""
        ...

    def copy(self) -> "Pose":
        """Create a copy of this pose."""
        ...


def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    """Subtract two vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    """Add two vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_scale(v: Vec3, s: float) -> Vec3:
    """Scale a vector."""
    return (v[0] * s, v[1] * s, v[2] * s)


def vec3_length(v: Vec3) -> float:
    """Get vector length."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_normalize(v: Vec3) -> Vec3:
    """Normalize a vector."""
    length = vec3_length(v)
    if length < 1e-10:
        return (0.0, 0.0, 1.0)  # Default forward direction
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


def quat_slerp(a: Quaternion, b: Quaternion, t: float) -> Quaternion:
    """Spherical linear interpolation between quaternions."""
    # Compute cosine of angle
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

    # If negative, negate one quaternion to take shorter path
    if dot < 0:
        b = (-b[0], -b[1], -b[2], -b[3])
        dot = -dot

    # Clamp dot product
    dot = min(dot, 1.0)

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        )
        # Normalize
        length = math.sqrt(sum(x * x for x in result))
        return tuple(x / length for x in result)

    # Compute slerp
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (
        a[0] * s0 + b[0] * s1,
        a[1] * s0 + b[1] * s1,
        a[2] * s0 + b[2] * s1,
        a[3] * s0 + b[3] * s1,
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

    # Normalize
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length > 1e-10:
        return (x / length, y / length, z / length, w / length)
    return quat_identity()


def clamp_angle(angle: float, min_angle: float, max_angle: float) -> float:
    """Clamp angle to range."""
    return max(min_angle, min(max_angle, angle))


def angle_between_vectors(a: Vec3, b: Vec3) -> float:
    """Calculate angle between two vectors in radians."""
    a = vec3_normalize(a)
    b = vec3_normalize(b)
    dot = vec3_dot(a, b)
    dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability
    return math.acos(dot)


@dataclass
class InterestPoint:
    """A point of interest that the character can look at."""

    position: Vec3
    priority: float = 1.0  # Higher priority takes precedence
    weight: float = 1.0  # Contribution weight (0-1)
    min_distance: float = 0.1  # Ignore if closer than this
    max_distance: float = 100.0  # Ignore if farther than this

    def __post_init__(self):
        if self.priority < 0:
            raise ValueError("priority must be >= 0")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("weight must be in [0, 1]")
        if self.min_distance < 0:
            raise ValueError("min_distance must be >= 0")
        if self.max_distance <= self.min_distance:
            raise ValueError("max_distance must be > min_distance")

    def is_in_range(self, from_position: Vec3) -> bool:
        """Check if this point is within valid range from a position."""
        distance = vec3_length(vec3_sub(self.position, from_position))
        return self.min_distance <= distance <= self.max_distance


@dataclass
class SaccadeGenerator:
    """
    Generates realistic eye saccades (rapid eye movements).

    Saccades are quick, simultaneous movements of both eyes between
    fixation points.
    """

    min_interval: float = 0.1  # Minimum time between saccades
    max_interval: float = 3.0  # Maximum time between saccades
    max_offset: float = 0.05  # Maximum saccade offset in radians (~3 degrees)
    speed: float = 500.0  # Saccade speed in degrees/second

    _time_until_next: float = field(default=0.0, repr=False)
    _current_offset: Vec3 = field(default=(0.0, 0.0, 0.0), repr=False)
    _target_offset: Vec3 = field(default=(0.0, 0.0, 0.0), repr=False)

    def __post_init__(self):
        if self.min_interval <= 0:
            raise ValueError("min_interval must be > 0")
        if self.max_interval <= self.min_interval:
            raise ValueError("max_interval must be > min_interval")
        if self.max_offset < 0:
            raise ValueError("max_offset must be >= 0")
        if self.speed <= 0:
            raise ValueError("speed must be > 0")

        self._time_until_next = random.uniform(self.min_interval, self.max_interval)

    def update(self, dt: float) -> Vec3:
        """
        Update saccade state and return current eye offset.

        Returns:
            Angular offset for eyes (x=horizontal, y=vertical, z unused)
        """
        self._time_until_next -= dt

        # Generate new saccade target
        if self._time_until_next <= 0:
            self._target_offset = (
                random.uniform(-self.max_offset, self.max_offset),
                random.uniform(-self.max_offset, self.max_offset),
                0.0,
            )
            self._time_until_next = random.uniform(self.min_interval, self.max_interval)

        # Move current offset toward target (fast saccade movement)
        speed_rad = math.radians(self.speed)
        max_delta = speed_rad * dt

        diff = vec3_sub(self._target_offset, self._current_offset)
        distance = vec3_length(diff)

        if distance > max_delta:
            direction = vec3_normalize(diff)
            self._current_offset = vec3_add(
                self._current_offset,
                vec3_scale(direction, max_delta)
            )
        else:
            self._current_offset = self._target_offset

        return self._current_offset

    def reset(self) -> None:
        """Reset saccade state."""
        self._current_offset = (0.0, 0.0, 0.0)
        self._target_offset = (0.0, 0.0, 0.0)
        self._time_until_next = random.uniform(self.min_interval, self.max_interval)


@dataclass
class LookAtController:
    """
    Controller for procedural head, neck, and eye look-at.

    Features:
    - Smooth interpolation with configurable speed
    - Angle limits to prevent unnatural rotation
    - Eye tracking with saccades
    - Multiple interest points with priority
    """

    head_bone: int
    neck_bone: int = -1  # Optional neck bone
    eye_bones: List[int] = field(default_factory=list)

    # Speed and smoothing
    rotation_speed: float = 5.0  # Radians per second
    blend_speed: float = 10.0  # Blend weight change per second

    # Angle limits (radians)
    head_yaw_limit: float = math.radians(80.0)  # Left/right
    head_pitch_limit: float = math.radians(40.0)  # Up/down
    neck_yaw_limit: float = math.radians(30.0)
    neck_pitch_limit: float = math.radians(20.0)
    eye_yaw_limit: float = math.radians(35.0)
    eye_pitch_limit: float = math.radians(25.0)

    # Distribution of rotation
    neck_contribution: float = 0.3  # How much neck contributes (0-1)
    eye_lead_time: float = 0.1  # Eyes lead head by this much

    # Eye tracking
    enable_saccades: bool = True
    saccade_generator: Optional[SaccadeGenerator] = None

    # Internal state
    _current_target: Vec3 = field(default=(0.0, 0.0, 1.0), repr=False)
    _current_weight: float = field(default=0.0, repr=False)
    _current_head_rotation: Quaternion = field(default=None, repr=False)
    _current_neck_rotation: Quaternion = field(default=None, repr=False)
    _current_eye_rotations: List[Quaternion] = field(default_factory=list, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.head_bone < 0:
            raise ValueError("head_bone must be >= 0")
        if self.rotation_speed <= 0:
            raise ValueError("rotation_speed must be > 0")
        if self.blend_speed <= 0:
            raise ValueError("blend_speed must be > 0")
        if not (0.0 <= self.neck_contribution <= 1.0):
            raise ValueError("neck_contribution must be in [0, 1]")

        if self._current_head_rotation is None:
            self._current_head_rotation = quat_identity()
        if self._current_neck_rotation is None:
            self._current_neck_rotation = quat_identity()

        if self.enable_saccades and self.saccade_generator is None:
            self.saccade_generator = SaccadeGenerator()

    def initialize(self, pose: Pose) -> None:
        """Initialize controller from current pose."""
        self._current_head_rotation = pose.get_bone_rotation(self.head_bone)

        if self.neck_bone >= 0:
            self._current_neck_rotation = pose.get_bone_rotation(self.neck_bone)

        self._current_eye_rotations = [
            pose.get_bone_rotation(eye) for eye in self.eye_bones
        ]

        self._initialized = True

    def _calculate_look_rotation(
        self,
        bone_position: Vec3,
        target: Vec3,
        yaw_limit: float,
        pitch_limit: float,
        forward: Vec3 = (0.0, 0.0, 1.0),
    ) -> Quaternion:
        """
        Calculate rotation to look at target with angle limits.

        Handles edge case where target is at bone position (zero-length direction)
        by returning identity rotation.
        """
        to_target = vec3_sub(target, bone_position)

        # Handle target at bone position - return identity (no rotation)
        if vec3_length(to_target) < 1e-6:
            return quat_identity()

        direction = vec3_normalize(to_target)

        # Calculate yaw and pitch angles
        yaw = math.atan2(direction[0], direction[2])
        pitch = math.asin(max(-1.0, min(1.0, direction[1])))

        # Apply limits
        yaw = clamp_angle(yaw, -yaw_limit, yaw_limit)
        pitch = clamp_angle(pitch, -pitch_limit, pitch_limit)

        # Create rotation quaternions
        yaw_quat = quat_from_axis_angle((0.0, 1.0, 0.0), yaw)
        pitch_quat = quat_from_axis_angle((1.0, 0.0, 0.0), -pitch)

        return quat_multiply(yaw_quat, pitch_quat)

    def update(
        self,
        pose: Pose,
        target_position: Vec3,
        dt: float,
        weight: float = 1.0,
    ) -> Pose:
        """
        Update look-at and return modified pose.

        Args:
            pose: Current animation pose
            target_position: World position to look at
            dt: Time step in seconds
            weight: Blend weight for look-at (0=animation, 1=full look-at)

        Returns:
            Modified pose with look-at applied
        """
        if dt <= 0:
            return pose

        if not self._initialized:
            self.initialize(pose)

        result = pose.copy()
        self._current_target = target_position

        # Blend weight interpolation
        target_weight = max(0.0, min(1.0, weight))
        blend_delta = self.blend_speed * dt
        if self._current_weight < target_weight:
            self._current_weight = min(target_weight, self._current_weight + blend_delta)
        else:
            self._current_weight = max(target_weight, self._current_weight - blend_delta)

        if self._current_weight < 0.001:
            return pose

        # Calculate head rotation
        head_position = pose.get_bone_position(self.head_bone)
        target_head_rotation = self._calculate_look_rotation(
            head_position,
            target_position,
            self.head_yaw_limit,
            self.head_pitch_limit,
        )

        # Interpolate head rotation
        rotation_delta = self.rotation_speed * dt
        self._current_head_rotation = quat_slerp(
            self._current_head_rotation,
            target_head_rotation,
            min(1.0, rotation_delta),
        )

        # Blend with animation
        anim_head_rotation = pose.get_bone_rotation(self.head_bone)
        blended_head = quat_slerp(
            anim_head_rotation,
            self._current_head_rotation,
            self._current_weight * (1.0 - self.neck_contribution),
        )
        result.set_bone_rotation(self.head_bone, blended_head)

        # Update neck if present
        if self.neck_bone >= 0:
            neck_position = pose.get_bone_position(self.neck_bone)
            target_neck_rotation = self._calculate_look_rotation(
                neck_position,
                target_position,
                self.neck_yaw_limit,
                self.neck_pitch_limit,
            )

            self._current_neck_rotation = quat_slerp(
                self._current_neck_rotation,
                target_neck_rotation,
                min(1.0, rotation_delta),
            )

            anim_neck_rotation = pose.get_bone_rotation(self.neck_bone)
            blended_neck = quat_slerp(
                anim_neck_rotation,
                self._current_neck_rotation,
                self._current_weight * self.neck_contribution,
            )
            result.set_bone_rotation(self.neck_bone, blended_neck)

        # Update eyes
        saccade_offset = (0.0, 0.0, 0.0)
        if self.enable_saccades and self.saccade_generator:
            saccade_offset = self.saccade_generator.update(dt)

        for i, eye_bone in enumerate(self.eye_bones):
            eye_position = pose.get_bone_position(eye_bone)

            # Eyes can rotate more than head
            target_eye_rotation = self._calculate_look_rotation(
                eye_position,
                target_position,
                self.eye_yaw_limit,
                self.eye_pitch_limit,
            )

            # Apply saccade offset
            if self.enable_saccades:
                saccade_yaw = quat_from_axis_angle((0.0, 1.0, 0.0), saccade_offset[0])
                saccade_pitch = quat_from_axis_angle((1.0, 0.0, 0.0), saccade_offset[1])
                saccade_rotation = quat_multiply(saccade_yaw, saccade_pitch)
                target_eye_rotation = quat_multiply(target_eye_rotation, saccade_rotation)

            # Eyes move faster than head
            eye_rotation_delta = self.rotation_speed * 2.0 * dt

            if i < len(self._current_eye_rotations):
                self._current_eye_rotations[i] = quat_slerp(
                    self._current_eye_rotations[i],
                    target_eye_rotation,
                    min(1.0, eye_rotation_delta),
                )
            else:
                self._current_eye_rotations.append(target_eye_rotation)

            anim_eye_rotation = pose.get_bone_rotation(eye_bone)
            blended_eye = quat_slerp(
                anim_eye_rotation,
                self._current_eye_rotations[i],
                self._current_weight,
            )
            result.set_bone_rotation(eye_bone, blended_eye)

        return result

    def update_with_interest_points(
        self,
        pose: Pose,
        interest_points: List[InterestPoint],
        dt: float,
    ) -> Pose:
        """
        Update look-at based on multiple interest points with priorities.

        The highest priority in-range point is selected.

        Args:
            pose: Current animation pose
            interest_points: List of interest points
            dt: Time step in seconds

        Returns:
            Modified pose with look-at applied
        """
        if not interest_points:
            return self.update(pose, self._current_target, dt, weight=0.0)

        head_position = pose.get_bone_position(self.head_bone)

        # Find highest priority point in range
        best_point: Optional[InterestPoint] = None
        best_priority = -1.0

        for point in interest_points:
            if point.is_in_range(head_position) and point.priority > best_priority:
                best_priority = point.priority
                best_point = point

        if best_point is None:
            return self.update(pose, self._current_target, dt, weight=0.0)

        return self.update(pose, best_point.position, dt, weight=best_point.weight)

    def reset(self, pose: Pose) -> None:
        """Reset controller state."""
        self._initialized = False
        self._current_weight = 0.0
        if self.saccade_generator:
            self.saccade_generator.reset()
        self.initialize(pose)

    def get_current_target(self) -> Vec3:
        """Get current look-at target."""
        return self._current_target

    def get_current_weight(self) -> float:
        """Get current blend weight."""
        return self._current_weight
