"""
Twist Bone Distribution.

Distributes twist rotation across helper bones for better joint deformation.
Common use cases:
- Forearm twist bones for realistic elbow rotation
- Upper arm twist for shoulder rotation
- Thigh twist for hip rotation

Usage:
    twist = TwistBone(
        source_bone=5,  # e.g., wrist
        twist_bones=[6, 7],  # twist helper bones
        distribution=TwistDistribution.LINEAR
    )
    modified_pose = twist.update(pose)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Protocol, Callable

# Type aliases
Vec3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (x, y, z, w)


class Pose(Protocol):
    """Protocol for pose data - maps bone indices to transforms."""

    def get_bone_position(self, bone_index: int) -> Vec3:
        """Get world position of a bone."""
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

    def get_parent_index(self, bone_index: int) -> int:
        """Get parent bone index, -1 for root."""
        ...

    def copy(self) -> "Pose":
        """Create a copy of this pose."""
        ...


class TwistDistribution(Enum):
    """Distribution modes for twist bones."""

    LINEAR = auto()  # Equal linear distribution
    EASE_IN = auto()  # More twist at end (close to source)
    EASE_OUT = auto()  # More twist at start (away from source)
    EASE_IN_OUT = auto()  # Smooth S-curve distribution
    CUSTOM = auto()  # Custom weights provided


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


def quat_conjugate(q: Quaternion) -> Quaternion:
    """Get quaternion conjugate (inverse for unit quaternions)."""
    return (-q[0], -q[1], -q[2], q[3])


def quat_normalize(q: Quaternion) -> Quaternion:
    """Normalize a quaternion."""
    length = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if length < 1e-10:
        return quat_identity()
    inv_length = 1.0 / length
    return (q[0] * inv_length, q[1] * inv_length, q[2] * inv_length, q[3] * inv_length)


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
        return quat_normalize(result)

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


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quaternion:
    """Create quaternion from axis and angle (radians)."""
    length = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    if length < 1e-10:
        return quat_identity()

    inv_length = 1.0 / length
    axis = (axis[0] * inv_length, axis[1] * inv_length, axis[2] * inv_length)

    half_angle = angle * 0.5
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)

    return (
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        cos_half,
    )


def quat_to_axis_angle(q: Quaternion) -> Tuple[Vec3, float]:
    """Convert quaternion to axis-angle representation."""
    # Normalize just in case
    q = quat_normalize(q)

    # Get angle from w component
    angle = 2.0 * math.acos(max(-1.0, min(1.0, q[3])))

    # Get axis from xyz components
    sin_half_angle = math.sqrt(1.0 - q[3] * q[3])

    if sin_half_angle < 1e-10:
        # No rotation, return arbitrary axis
        return ((1.0, 0.0, 0.0), 0.0)

    inv_sin = 1.0 / sin_half_angle
    axis = (q[0] * inv_sin, q[1] * inv_sin, q[2] * inv_sin)

    return (axis, angle)


def extract_twist_rotation(
    rotation: Quaternion,
    twist_axis: Vec3 = (1.0, 0.0, 0.0),
) -> Quaternion:
    """
    Extract the twist component of a rotation around a specific axis.

    This decomposes the rotation into swing and twist components,
    returning only the twist.
    """
    # Project rotation axis onto twist axis
    axis, angle = quat_to_axis_angle(rotation)

    # Dot product gives projection onto twist axis
    dot = axis[0] * twist_axis[0] + axis[1] * twist_axis[1] + axis[2] * twist_axis[2]

    # The twist angle is the angle scaled by the projection
    twist_angle = angle * dot

    # Create twist rotation around the twist axis
    return quat_from_axis_angle(twist_axis, twist_angle)


def ease_in(t: float) -> float:
    """Ease-in function (quadratic)."""
    return t * t


def ease_out(t: float) -> float:
    """Ease-out function (quadratic)."""
    return 1.0 - (1.0 - t) * (1.0 - t)


def ease_in_out(t: float) -> float:
    """Ease-in-out function (smooth S-curve)."""
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - pow(-2.0 * t + 2.0, 2) / 2.0


@dataclass
class TwistBone:
    """
    Distributes twist rotation from a source bone to helper twist bones.

    The source bone's twist (rotation around a specified axis) is extracted
    and distributed across the twist bones according to the distribution mode.
    """

    source_bone: int
    twist_bones: List[int]
    distribution: TwistDistribution = TwistDistribution.LINEAR
    twist_axis: Vec3 = (1.0, 0.0, 0.0)  # Local axis to twist around
    custom_weights: Optional[List[float]] = None  # For CUSTOM distribution
    weight: float = 1.0  # Overall twist weight

    # Reference bone for calculating relative twist
    reference_bone: int = -1  # Parent bone by default

    # Internal state
    _reference_rotation: Quaternion = field(default=None, repr=False)

    def __post_init__(self):
        if self.source_bone < 0:
            raise ValueError("source_bone must be >= 0")
        if not self.twist_bones:
            raise ValueError("twist_bones must not be empty")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("weight must be in [0, 1]")

        if self.distribution == TwistDistribution.CUSTOM:
            if self.custom_weights is None:
                raise ValueError("custom_weights required for CUSTOM distribution")
            if len(self.custom_weights) != len(self.twist_bones):
                raise ValueError(
                    "custom_weights length must match twist_bones length"
                )
            for w in self.custom_weights:
                if not (0.0 <= w <= 1.0):
                    raise ValueError("custom_weights values must be in [0, 1]")

        if self._reference_rotation is None:
            self._reference_rotation = quat_identity()

    def _get_distribution_weight(self, index: int, total: int) -> float:
        """Calculate weight for a bone based on distribution mode."""
        if total <= 1:
            return 1.0

        # Normalized position (0 = first bone, 1 = last bone)
        t = index / (total - 1) if total > 1 else 0.0

        if self.distribution == TwistDistribution.LINEAR:
            return t
        elif self.distribution == TwistDistribution.EASE_IN:
            return ease_in(t)
        elif self.distribution == TwistDistribution.EASE_OUT:
            return ease_out(t)
        elif self.distribution == TwistDistribution.EASE_IN_OUT:
            return ease_in_out(t)
        elif self.distribution == TwistDistribution.CUSTOM:
            return self.custom_weights[index]
        else:
            return t

    def update(self, pose: Pose) -> Pose:
        """
        Update twist bones based on source bone rotation.

        Args:
            pose: Current animation pose

        Returns:
            Modified pose with distributed twist
        """
        result = pose.copy()

        # Get source bone rotation
        source_rotation = pose.get_bone_rotation(self.source_bone)

        # Get reference rotation
        if self.reference_bone >= 0:
            ref_rotation = pose.get_bone_rotation(self.reference_bone)
        else:
            # Try to use parent as reference
            parent_idx = pose.get_parent_index(self.source_bone)
            if parent_idx >= 0:
                ref_rotation = pose.get_bone_rotation(parent_idx)
            else:
                ref_rotation = quat_identity()

        # Calculate relative rotation
        ref_inv = quat_conjugate(ref_rotation)
        relative_rotation = quat_multiply(ref_inv, source_rotation)

        # Extract twist component
        twist_rotation = extract_twist_rotation(relative_rotation, self.twist_axis)

        # Get twist angle
        _, twist_angle = quat_to_axis_angle(twist_rotation)

        # Apply twist to each twist bone based on distribution
        num_bones = len(self.twist_bones)

        for i, bone_idx in enumerate(self.twist_bones):
            # Calculate distribution weight
            dist_weight = self._get_distribution_weight(i, num_bones)

            # Calculate final twist amount
            bone_twist_angle = twist_angle * dist_weight * self.weight

            # Create rotation for this bone
            bone_twist = quat_from_axis_angle(self.twist_axis, bone_twist_angle)

            # Get current bone rotation and apply twist
            current_rotation = pose.get_bone_rotation(bone_idx)
            new_rotation = quat_multiply(current_rotation, bone_twist)
            new_rotation = quat_normalize(new_rotation)

            result.set_bone_rotation(bone_idx, new_rotation)

        return result

    def get_bone_count(self) -> int:
        """Get number of twist bones."""
        return len(self.twist_bones)

    def set_weight(self, weight: float) -> None:
        """Set overall twist weight."""
        if not (0.0 <= weight <= 1.0):
            raise ValueError("weight must be in [0, 1]")
        self.weight = weight

    def set_custom_weights(self, weights: List[float]) -> None:
        """Set custom distribution weights."""
        if len(weights) != len(self.twist_bones):
            raise ValueError("weights length must match twist_bones length")
        for w in weights:
            if not (0.0 <= w <= 1.0):
                raise ValueError("weight values must be in [0, 1]")
        self.custom_weights = weights
        self.distribution = TwistDistribution.CUSTOM


@dataclass
class TwistChain:
    """
    A chain of twist bone setups for common configurations.

    Provides presets for arm and leg twist setups.
    """

    upper_arm_twist: Optional[TwistBone] = None
    forearm_twist: Optional[TwistBone] = None
    thigh_twist: Optional[TwistBone] = None
    calf_twist: Optional[TwistBone] = None

    @classmethod
    def create_arm_setup(
        cls,
        shoulder_bone: int,
        upper_arm_bone: int,
        upper_arm_twist_bones: List[int],
        forearm_bone: int,
        forearm_twist_bones: List[int],
        wrist_bone: int,
    ) -> "TwistChain":
        """
        Create a complete arm twist setup.

        Args:
            shoulder_bone: Shoulder/clavicle bone index
            upper_arm_bone: Upper arm bone index
            upper_arm_twist_bones: Helper bones for upper arm twist
            forearm_bone: Forearm/lower arm bone index
            forearm_twist_bones: Helper bones for forearm twist
            wrist_bone: Wrist/hand bone index

        Returns:
            Configured TwistChain for arm
        """
        upper = None
        if upper_arm_twist_bones:
            upper = TwistBone(
                source_bone=upper_arm_bone,
                twist_bones=upper_arm_twist_bones,
                distribution=TwistDistribution.LINEAR,
                twist_axis=(1.0, 0.0, 0.0),
                reference_bone=shoulder_bone,
            )

        forearm = None
        if forearm_twist_bones:
            forearm = TwistBone(
                source_bone=wrist_bone,
                twist_bones=forearm_twist_bones,
                distribution=TwistDistribution.LINEAR,
                twist_axis=(1.0, 0.0, 0.0),
                reference_bone=forearm_bone,
            )

        return cls(upper_arm_twist=upper, forearm_twist=forearm)

    @classmethod
    def create_leg_setup(
        cls,
        hip_bone: int,
        thigh_bone: int,
        thigh_twist_bones: List[int],
        calf_bone: int,
        calf_twist_bones: List[int],
        foot_bone: int,
    ) -> "TwistChain":
        """
        Create a complete leg twist setup.

        Args:
            hip_bone: Hip bone index
            thigh_bone: Thigh/upper leg bone index
            thigh_twist_bones: Helper bones for thigh twist
            calf_bone: Calf/lower leg bone index
            calf_twist_bones: Helper bones for calf twist
            foot_bone: Foot/ankle bone index

        Returns:
            Configured TwistChain for leg
        """
        thigh = None
        if thigh_twist_bones:
            thigh = TwistBone(
                source_bone=thigh_bone,
                twist_bones=thigh_twist_bones,
                distribution=TwistDistribution.LINEAR,
                twist_axis=(0.0, 1.0, 0.0),  # Y-axis for legs typically
                reference_bone=hip_bone,
            )

        calf = None
        if calf_twist_bones:
            calf = TwistBone(
                source_bone=foot_bone,
                twist_bones=calf_twist_bones,
                distribution=TwistDistribution.LINEAR,
                twist_axis=(0.0, 1.0, 0.0),
                reference_bone=calf_bone,
            )

        return cls(thigh_twist=thigh, calf_twist=calf)

    def update(self, pose: Pose) -> Pose:
        """
        Update all twist bones in the chain.

        Args:
            pose: Current animation pose

        Returns:
            Modified pose with all twists applied
        """
        result = pose

        if self.upper_arm_twist:
            result = self.upper_arm_twist.update(result)
        if self.forearm_twist:
            result = self.forearm_twist.update(result)
        if self.thigh_twist:
            result = self.thigh_twist.update(result)
        if self.calf_twist:
            result = self.calf_twist.update(result)

        return result
