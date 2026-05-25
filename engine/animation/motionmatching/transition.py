"""
Motion Matching Transitions - Smooth motion transition handling.

This module provides transition handling for motion matching:
- TransitionConfig: Configuration for transition behavior
- MotionTransition: State for an active transition
- Inertialization blending (spring-based smooth transitions)
- Velocity matching at transition points
- Foot sliding cleanup

Usage:
    from engine.animation.motionmatching.transition import (
        TransitionConfig, MotionTransition, InertializationBlender
    )

    # Configure transition
    config = TransitionConfig(blend_duration=0.2)

    # Create transition
    transition = MotionTransition(from_entry, to_entry, config)

    # Update and get blended pose
    blended_pose = transition.update(dt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import numpy as np

from engine.animation.motionmatching.database import DatabaseEntry


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================


class BlendMode(Enum):
    """Transition blending modes."""
    LINEAR = auto()          # Simple linear interpolation
    INERTIALIZATION = auto() # Spring-based inertialization
    CROSSFADE = auto()       # Standard crossfade with ease curve


class FootState(Enum):
    """Foot contact state."""
    GROUNDED = auto()
    MOVING = auto()
    UNKNOWN = auto()


# Import centralized config
from engine.animation.motionmatching.config import (
    DEFAULT_TRANSITION_PARAMS,
    DEFAULT_CONTACT_DETECTION,
)

# Default spring parameters for inertialization
DEFAULT_SPRING_HALFLIFE = DEFAULT_TRANSITION_PARAMS.spring_halflife
DEFAULT_BLEND_DURATION = DEFAULT_TRANSITION_PARAMS.default_blend_duration
MIN_BLEND_DURATION = DEFAULT_TRANSITION_PARAMS.min_blend_duration
MIN_SPRING_HALFLIFE = DEFAULT_TRANSITION_PARAMS.min_spring_halflife


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class TransitionConfig:
    """Configuration for motion transitions.

    Attributes:
        blend_duration: Duration of blend in seconds
        blend_mode: Blending method to use
        spring_halflife: Half-life for spring decay (inertialization)
        velocity_matching: Whether to match velocities at transition
        foot_locking: Whether to apply foot locking
        foot_lock_threshold: Maximum foot velocity to consider grounded
    """
    blend_duration: float = DEFAULT_BLEND_DURATION
    blend_mode: BlendMode = BlendMode.INERTIALIZATION
    spring_halflife: float = DEFAULT_SPRING_HALFLIFE
    velocity_matching: bool = True
    foot_locking: bool = True
    foot_lock_threshold: float = 0.1


@dataclass
class BoneTransform:
    """Transform data for a single bone.

    Attributes:
        position: Local position (x, y, z)
        rotation: Local rotation as quaternion (x, y, z, w)
        scale: Local scale (x, y, z)
    """
    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1], dtype=np.float32))

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        self.rotation = np.asarray(self.rotation, dtype=np.float32)
        self.scale = np.asarray(self.scale, dtype=np.float32)

    def copy(self) -> BoneTransform:
        """Create a copy of this transform."""
        return BoneTransform(
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            scale=self.scale.copy(),
        )


@dataclass
class Pose:
    """Complete pose for all bones.

    Attributes:
        bone_transforms: Dictionary mapping bone names to transforms
        root_position: World position of root
        root_rotation: World rotation of root (quaternion)
    """
    bone_transforms: Dict[str, BoneTransform] = field(default_factory=dict)
    root_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    root_rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32))

    def __post_init__(self):
        self.root_position = np.asarray(self.root_position, dtype=np.float32)
        self.root_rotation = np.asarray(self.root_rotation, dtype=np.float32)

    def copy(self) -> Pose:
        """Create a deep copy of this pose."""
        return Pose(
            bone_transforms={name: t.copy() for name, t in self.bone_transforms.items()},
            root_position=self.root_position.copy(),
            root_rotation=self.root_rotation.copy(),
        )

    def get_bone(self, name: str) -> Optional[BoneTransform]:
        """Get bone transform by name."""
        return self.bone_transforms.get(name)

    def set_bone(self, name: str, transform: BoneTransform) -> None:
        """Set bone transform."""
        self.bone_transforms[name] = transform


@dataclass
class InertializationOffset:
    """Offset values for inertialization blending.

    Stores the initial offset and velocity at the moment of transition,
    which are then decayed over time using a spring function.

    Attributes:
        position_offset: Initial position offset
        position_velocity: Initial velocity offset
        rotation_offset: Initial rotation offset (axis-angle or quaternion diff)
        rotation_velocity: Initial angular velocity offset
    """
    position_offset: np.ndarray
    position_velocity: np.ndarray
    rotation_offset: np.ndarray
    rotation_velocity: np.ndarray

    @classmethod
    def zero(cls) -> InertializationOffset:
        """Create zero offset."""
        return cls(
            position_offset=np.zeros(3, dtype=np.float32),
            position_velocity=np.zeros(3, dtype=np.float32),
            rotation_offset=np.zeros(4, dtype=np.float32),  # Quaternion
            rotation_velocity=np.zeros(3, dtype=np.float32),  # Axis-angle
        )


# =============================================================================
# SPRING FUNCTIONS
# =============================================================================


def compute_spring_decay(halflife: float) -> float:
    """Compute spring decay constant from halflife.

    Args:
        halflife: Time to decay to half (seconds)

    Returns:
        Decay constant for exponential decay
    """
    # Use configurable minimum to prevent numerical issues
    return 0.6931472 / max(halflife, MIN_SPRING_HALFLIFE)


def spring_decay_position(
    offset: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    decay: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply spring decay to position offset.

    Uses critical damping for smooth decay without oscillation.

    Args:
        offset: Current position offset
        velocity: Current velocity offset
        dt: Time step
        decay: Decay constant

    Returns:
        Tuple of (new_offset, new_velocity)
    """
    # Exponential decay
    decay_factor = math.exp(-decay * dt)

    # Update offset with velocity contribution
    new_offset = (offset + velocity * dt) * decay_factor

    # Update velocity (decays faster)
    new_velocity = velocity * decay_factor - offset * (decay * decay_factor)

    return new_offset, new_velocity


def spring_decay_rotation(
    offset_quat: np.ndarray,
    angular_velocity: np.ndarray,
    dt: float,
    decay: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply spring decay to rotation offset.

    Args:
        offset_quat: Current rotation offset (quaternion)
        angular_velocity: Current angular velocity
        dt: Time step
        decay: Decay constant

    Returns:
        Tuple of (new_offset_quat, new_angular_velocity)
    """
    # Convert quaternion offset to axis-angle for decay
    axis, angle = quaternion_to_axis_angle(offset_quat)

    # Decay angle
    decay_factor = math.exp(-decay * dt)
    new_angle = angle * decay_factor

    # Convert back to quaternion
    new_offset_quat = axis_angle_to_quaternion(axis, new_angle)

    # Decay angular velocity
    new_angular_velocity = angular_velocity * decay_factor

    return new_offset_quat, new_angular_velocity


# =============================================================================
# QUATERNION UTILITIES
# =============================================================================


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions.

    Args:
        q1: First quaternion (x, y, z, w)
        q2: Second quaternion (x, y, z, w)

    Returns:
        Product quaternion
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (conjugate for unit quaternions).

    Args:
        q: Quaternion (x, y, z, w)

    Returns:
        Inverse quaternion
    """
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quaternion_difference(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute quaternion that rotates from q1 to q2.

    Args:
        q1: Source quaternion
        q2: Target quaternion

    Returns:
        Difference quaternion (q1_inv * q2)
    """
    return quaternion_multiply(quaternion_inverse(q1), q2)


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions.

    Args:
        q1: Start quaternion
        q2: End quaternion
        t: Interpolation factor (0-1)

    Returns:
        Interpolated quaternion
    """
    # Compute dot product
    dot = np.dot(q1, q2)

    # If negative dot, negate one quaternion to take shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # Standard slerp
    theta_0 = math.acos(min(1.0, dot))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q1 + s1 * q2


def quaternion_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert quaternion to axis-angle representation.

    Args:
        q: Quaternion (x, y, z, w)

    Returns:
        Tuple of (axis, angle)
    """
    # Ensure w is positive (shorter rotation)
    if q[3] < 0:
        q = -q

    # Extract angle
    angle = 2 * math.acos(min(1.0, max(-1.0, q[3])))

    # Extract axis
    s = math.sqrt(1 - q[3] * q[3])
    if s < 1e-8:
        axis = np.array([1, 0, 0], dtype=np.float32)
    else:
        axis = q[:3] / s

    return axis, angle


def axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle to quaternion.

    Args:
        axis: Rotation axis (normalized)
        angle: Rotation angle in radians

    Returns:
        Quaternion (x, y, z, w)
    """
    half_angle = angle / 2
    s = math.sin(half_angle)
    c = math.cos(half_angle)

    return np.array([
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        c,
    ], dtype=np.float32)


# =============================================================================
# INERTIALIZATION BLENDER
# =============================================================================


class InertializationBlender:
    """Handles inertialization-based blending for smooth transitions.

    Inertialization computes the difference between current pose and
    target pose at the moment of transition, then decays this offset
    smoothly over time using spring dynamics.

    This produces smooth transitions without the "blending" artifacts
    of standard crossfade.
    """

    def __init__(self, config: TransitionConfig):
        """Initialize blender.

        Args:
            config: Transition configuration
        """
        self.config = config
        self.decay = compute_spring_decay(config.spring_halflife)

        # Per-bone offsets
        self._bone_offsets: Dict[str, InertializationOffset] = {}

        # Root offsets
        self._root_position_offset = np.zeros(3, dtype=np.float32)
        self._root_position_velocity = np.zeros(3, dtype=np.float32)
        self._root_rotation_offset = np.array([0, 0, 0, 1], dtype=np.float32)
        self._root_angular_velocity = np.zeros(3, dtype=np.float32)

        self._elapsed_time = 0.0

    def compute_offsets(
        self,
        from_pose: Pose,
        to_pose: Pose,
        from_velocity: Optional[Dict[str, np.ndarray]] = None,
        to_velocity: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Compute inertialization offsets at transition start.

        Args:
            from_pose: Current pose at transition
            to_pose: Target pose to transition to
            from_velocity: Optional bone velocities of from_pose
            to_velocity: Optional bone velocities of to_pose
        """
        self._elapsed_time = 0.0
        self._bone_offsets.clear()

        # Root position offset
        self._root_position_offset = from_pose.root_position - to_pose.root_position

        # Root velocity offset
        if from_velocity and to_velocity and 'root' in from_velocity and 'root' in to_velocity:
            self._root_position_velocity = from_velocity['root'] - to_velocity['root']
        else:
            self._root_position_velocity = np.zeros(3, dtype=np.float32)

        # Root rotation offset
        self._root_rotation_offset = quaternion_difference(
            to_pose.root_rotation, from_pose.root_rotation
        )
        self._root_angular_velocity = np.zeros(3, dtype=np.float32)

        # Per-bone offsets
        all_bones = set(from_pose.bone_transforms.keys()) | set(to_pose.bone_transforms.keys())

        for bone_name in all_bones:
            from_bone = from_pose.get_bone(bone_name)
            to_bone = to_pose.get_bone(bone_name)

            if from_bone is None or to_bone is None:
                continue

            # Position offset
            pos_offset = from_bone.position - to_bone.position

            # Velocity offset
            if from_velocity and to_velocity:
                from_vel = from_velocity.get(bone_name, np.zeros(3))
                to_vel = to_velocity.get(bone_name, np.zeros(3))
                vel_offset = from_vel - to_vel
            else:
                vel_offset = np.zeros(3, dtype=np.float32)

            # Rotation offset
            rot_offset = quaternion_difference(to_bone.rotation, from_bone.rotation)

            self._bone_offsets[bone_name] = InertializationOffset(
                position_offset=pos_offset,
                position_velocity=vel_offset,
                rotation_offset=rot_offset,
                rotation_velocity=np.zeros(3, dtype=np.float32),
            )

    def update(self, dt: float) -> None:
        """Update offset decay.

        Args:
            dt: Time step in seconds
        """
        self._elapsed_time += dt

        # Decay root offsets
        self._root_position_offset, self._root_position_velocity = spring_decay_position(
            self._root_position_offset,
            self._root_position_velocity,
            dt,
            self.decay,
        )

        self._root_rotation_offset, self._root_angular_velocity = spring_decay_rotation(
            self._root_rotation_offset,
            self._root_angular_velocity,
            dt,
            self.decay,
        )

        # Decay per-bone offsets
        for bone_name, offset in self._bone_offsets.items():
            offset.position_offset, offset.position_velocity = spring_decay_position(
                offset.position_offset,
                offset.position_velocity,
                dt,
                self.decay,
            )

            offset.rotation_offset, offset.rotation_velocity = spring_decay_rotation(
                offset.rotation_offset,
                offset.rotation_velocity,
                dt,
                self.decay,
            )

    def apply(self, target_pose: Pose) -> Pose:
        """Apply offsets to target pose.

        Args:
            target_pose: The base target pose

        Returns:
            Pose with inertialization offsets applied
        """
        result = target_pose.copy()

        # Apply root offsets
        result.root_position = target_pose.root_position + self._root_position_offset
        result.root_rotation = quaternion_multiply(
            target_pose.root_rotation, self._root_rotation_offset
        )

        # Apply per-bone offsets
        for bone_name, offset in self._bone_offsets.items():
            bone = result.get_bone(bone_name)
            if bone is None:
                continue

            bone.position = bone.position + offset.position_offset
            bone.rotation = quaternion_multiply(bone.rotation, offset.rotation_offset)

        return result

    @property
    def is_complete(self) -> bool:
        """Check if inertialization has decayed to negligible levels."""
        # Check if all offsets are small enough
        pos_threshold = 0.001
        rot_threshold = 0.001

        if np.linalg.norm(self._root_position_offset) > pos_threshold:
            return False

        for offset in self._bone_offsets.values():
            if np.linalg.norm(offset.position_offset) > pos_threshold:
                return False
            # Check rotation offset angle
            _, angle = quaternion_to_axis_angle(offset.rotation_offset)
            if abs(angle) > rot_threshold:
                return False

        return True


# =============================================================================
# MOTION TRANSITION
# =============================================================================


class MotionTransition:
    """Manages a transition between motion matching entries.

    Handles the state and progress of transitioning from one
    animation frame to another using inertialization or crossfade.
    """

    def __init__(
        self,
        from_entry: DatabaseEntry,
        to_entry: DatabaseEntry,
        config: Optional[TransitionConfig] = None,
    ):
        """Initialize transition.

        Args:
            from_entry: Source database entry
            to_entry: Target database entry
            config: Transition configuration
        """
        self.from_entry = from_entry
        self.to_entry = to_entry
        self.config = config or TransitionConfig()

        self._progress = 0.0
        self._elapsed_time = 0.0
        self._is_complete = False

        # Inertialization blender (created when poses are provided)
        self._blender: Optional[InertializationBlender] = None

        # Cached poses
        self._from_pose: Optional[Pose] = None
        self._to_pose: Optional[Pose] = None

    @property
    def progress(self) -> float:
        """Transition progress (0-1)."""
        return self._progress

    @property
    def is_complete(self) -> bool:
        """Whether transition is complete."""
        return self._is_complete

    @property
    def elapsed_time(self) -> float:
        """Elapsed time since transition started."""
        return self._elapsed_time

    def initialize(
        self,
        from_pose: Pose,
        to_pose: Pose,
        from_velocity: Optional[Dict[str, np.ndarray]] = None,
        to_velocity: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Initialize transition with actual pose data.

        Args:
            from_pose: Current pose at transition start
            to_pose: Target pose
            from_velocity: Optional velocities of from_pose
            to_velocity: Optional velocities of to_pose
        """
        self._from_pose = from_pose.copy()
        self._to_pose = to_pose.copy()
        self._progress = 0.0
        self._elapsed_time = 0.0
        self._is_complete = False

        if self.config.blend_mode == BlendMode.INERTIALIZATION:
            self._blender = InertializationBlender(self.config)
            self._blender.compute_offsets(from_pose, to_pose, from_velocity, to_velocity)

    def update(self, dt: float, current_target_pose: Optional[Pose] = None) -> Pose:
        """Update transition and get blended pose.

        Args:
            dt: Time step in seconds
            current_target_pose: Current target pose (for inertialization)

        Returns:
            Blended output pose
        """
        self._elapsed_time += dt

        # Use minimum blend duration to prevent instant snapping
        effective_duration = max(self.config.blend_duration, MIN_BLEND_DURATION)
        self._progress = min(1.0, self._elapsed_time / effective_duration)

        # Check completion
        if self._progress >= 1.0:
            self._is_complete = True
            if current_target_pose:
                return current_target_pose.copy()
            return self._to_pose.copy() if self._to_pose else Pose()

        # Blend based on mode
        if self.config.blend_mode == BlendMode.INERTIALIZATION:
            return self._blend_inertialization(dt, current_target_pose)
        elif self.config.blend_mode == BlendMode.CROSSFADE:
            return self._blend_crossfade()
        else:
            return self._blend_linear()

    def _blend_inertialization(
        self, dt: float, current_target_pose: Optional[Pose]
    ) -> Pose:
        """Apply inertialization blending.

        Args:
            dt: Time step
            current_target_pose: Current target pose

        Returns:
            Blended pose
        """
        if self._blender is None:
            return self._to_pose.copy() if self._to_pose else Pose()

        self._blender.update(dt)

        target = current_target_pose or self._to_pose
        if target is None:
            return Pose()

        return self._blender.apply(target)

    def _blend_crossfade(self) -> Pose:
        """Apply crossfade blending with ease curve.

        Returns:
            Blended pose
        """
        if self._from_pose is None or self._to_pose is None:
            return self._to_pose.copy() if self._to_pose else Pose()

        # Use smooth ease curve
        t = self._smooth_step(self._progress)

        return self._interpolate_poses(self._from_pose, self._to_pose, t)

    def _blend_linear(self) -> Pose:
        """Apply simple linear blending.

        Returns:
            Blended pose
        """
        if self._from_pose is None or self._to_pose is None:
            return self._to_pose.copy() if self._to_pose else Pose()

        return self._interpolate_poses(self._from_pose, self._to_pose, self._progress)

    def _interpolate_poses(
        self, pose_a: Pose, pose_b: Pose, t: float
    ) -> Pose:
        """Interpolate between two poses.

        Args:
            pose_a: Start pose
            pose_b: End pose
            t: Interpolation factor (0-1)

        Returns:
            Interpolated pose
        """
        result = Pose()

        # Interpolate root
        result.root_position = (1 - t) * pose_a.root_position + t * pose_b.root_position
        result.root_rotation = quaternion_slerp(pose_a.root_rotation, pose_b.root_rotation, t)

        # Interpolate bones
        all_bones = set(pose_a.bone_transforms.keys()) | set(pose_b.bone_transforms.keys())

        for bone_name in all_bones:
            bone_a = pose_a.get_bone(bone_name)
            bone_b = pose_b.get_bone(bone_name)

            if bone_a is None:
                result.set_bone(bone_name, bone_b.copy())
            elif bone_b is None:
                result.set_bone(bone_name, bone_a.copy())
            else:
                result.set_bone(bone_name, BoneTransform(
                    position=(1 - t) * bone_a.position + t * bone_b.position,
                    rotation=quaternion_slerp(bone_a.rotation, bone_b.rotation, t),
                    scale=(1 - t) * bone_a.scale + t * bone_b.scale,
                ))

        return result

    def _smooth_step(self, t: float) -> float:
        """Smooth step function for ease curve.

        Args:
            t: Linear input (0-1)

        Returns:
            Smoothed output (0-1)
        """
        return t * t * (3 - 2 * t)


# =============================================================================
# FOOT SLIDING CLEANUP
# =============================================================================


class FootSlidingCorrector:
    """Corrects foot sliding artifacts during transitions and playback.

    Uses foot contact information to lock feet in place when they
    should be grounded.
    """

    def __init__(
        self,
        left_foot_bone: str = 'left_foot',
        right_foot_bone: str = 'right_foot',
        velocity_threshold: float = 0.1,
        height_threshold: float = 0.05,
    ):
        """Initialize corrector.

        Args:
            left_foot_bone: Name of left foot bone
            right_foot_bone: Name of right foot bone
            velocity_threshold: Maximum velocity for grounded foot
            height_threshold: Maximum height for grounded foot
        """
        self.left_foot_bone = left_foot_bone
        self.right_foot_bone = right_foot_bone
        self.velocity_threshold = velocity_threshold
        self.height_threshold = height_threshold

        # Locked foot positions
        self._left_lock_position: Optional[np.ndarray] = None
        self._right_lock_position: Optional[np.ndarray] = None

        self._left_locked = False
        self._right_locked = False

    def update_contacts(
        self,
        left_contact: float,
        right_contact: float,
        left_position: np.ndarray,
        right_position: np.ndarray,
    ) -> None:
        """Update foot contact states.

        Args:
            left_contact: Left foot contact (0-1)
            right_contact: Right foot contact (0-1)
            left_position: Current left foot world position
            right_position: Current right foot world position
        """
        # Left foot
        if left_contact > 0.5 and not self._left_locked:
            self._left_lock_position = left_position.copy()
            self._left_locked = True
        elif left_contact < 0.5:
            self._left_locked = False
            self._left_lock_position = None

        # Right foot
        if right_contact > 0.5 and not self._right_locked:
            self._right_lock_position = right_position.copy()
            self._right_locked = True
        elif right_contact < 0.5:
            self._right_locked = False
            self._right_lock_position = None

    def correct_pose(
        self,
        pose: Pose,
        left_world_position: np.ndarray,
        right_world_position: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute corrections for foot positions.

        Args:
            pose: Current pose
            left_world_position: Current left foot world position
            right_world_position: Current right foot world position

        Returns:
            Tuple of (left_correction, right_correction) as position offsets
        """
        left_correction = np.zeros(3, dtype=np.float32)
        right_correction = np.zeros(3, dtype=np.float32)

        if self._left_locked and self._left_lock_position is not None:
            left_correction = self._left_lock_position - left_world_position

        if self._right_locked and self._right_lock_position is not None:
            right_correction = self._right_lock_position - right_world_position

        return left_correction, right_correction

    def reset(self) -> None:
        """Reset all foot locks."""
        self._left_locked = False
        self._right_locked = False
        self._left_lock_position = None
        self._right_lock_position = None


# =============================================================================
# VELOCITY MATCHING
# =============================================================================


def compute_velocity_offsets(
    from_velocity: np.ndarray,
    to_velocity: np.ndarray,
    blend_duration: float,
) -> np.ndarray:
    """Compute initial velocity offset for smooth velocity matching.

    Args:
        from_velocity: Current velocity
        to_velocity: Target velocity
        blend_duration: Blend duration

    Returns:
        Velocity offset to apply
    """
    return from_velocity - to_velocity


def apply_velocity_matching(
    position: np.ndarray,
    velocity: np.ndarray,
    velocity_offset: np.ndarray,
    t: float,
    blend_duration: float,
) -> np.ndarray:
    """Apply velocity matching correction.

    Args:
        position: Current position
        velocity: Current velocity
        velocity_offset: Initial velocity offset
        t: Current progress (0-1)
        blend_duration: Total blend duration

    Returns:
        Corrected position
    """
    # Decay velocity offset over blend duration
    remaining_offset = velocity_offset * (1 - t)

    # Integrate to get position correction
    elapsed = t * blend_duration
    position_correction = remaining_offset * elapsed * (1 - t)

    return position + position_correction
