"""IK goal definitions for inverse kinematics solvers.

This module provides the fundamental data structures for defining IK targets
and goals used by various IK solvers in the animation system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.constants import MATH_EPSILON
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    IK_ROTATION_TOLERANCE,
    LOOK_AT_MAX_ANGLE,
    GOAL_BLENDER_DEFAULT_SPEED,
    FABRIK_DEFAULT_MAX_ITERATIONS,
)


class IKGoalType(Enum):
    """Types of IK goals supported by solvers."""

    POSITION = auto()
    """Goal to move a bone to a specific position."""

    ROTATION = auto()
    """Goal to rotate a bone to a specific orientation."""

    LOOK_AT = auto()
    """Goal to make a bone look at a target point."""

    POSITION_AND_ROTATION = auto()
    """Goal combining both position and rotation targets."""

    POLE_VECTOR = auto()
    """Goal defining the plane for chain bending (e.g., elbow direction)."""

    CENTER_OF_MASS = auto()
    """Goal to maintain balance by positioning center of mass."""


@dataclass
class IKGoal:
    """Base class for IK goals.

    An IK goal represents a desired position, rotation, or constraint
    that an IK solver will try to satisfy.

    Attributes:
        goal_type: The type of goal (position, rotation, look-at, etc.)
        bone_index: Index of the target bone in the skeleton
        weight: Influence of this goal (0.0 to 1.0)
        priority: Higher priority goals are satisfied first
        enabled: Whether this goal is currently active
    """

    bone_index: int
    goal_type: IKGoalType = field(default=IKGoalType.POSITION)
    weight: float = 1.0
    priority: int = 0
    enabled: bool = True

    def validate(self) -> bool:
        """Validate goal parameters.

        Returns:
            True if goal is valid, False otherwise.
        """
        if self.bone_index < 0:
            return False
        if self.weight < 0.0 or self.weight > 1.0:
            return False
        return True


@dataclass
class PositionGoal(IKGoal):
    """Goal for reaching a target position.

    The IK solver will attempt to move the bone's end effector
    to the target position.

    Attributes:
        target_position: World-space position to reach
        tolerance: Distance threshold for considering goal achieved
    """

    target_position: Vec3 = field(default_factory=Vec3.zero)
    tolerance: float = IK_DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        object.__setattr__(self, 'goal_type', IKGoalType.POSITION)

    def distance_to_target(self, current_position: Vec3) -> float:
        """Calculate distance from current position to target.

        Args:
            current_position: Current end effector position

        Returns:
            Distance to target position.
        """
        return current_position.distance(self.target_position)

    def is_achieved(self, current_position: Vec3) -> bool:
        """Check if goal is achieved within tolerance.

        Args:
            current_position: Current end effector position

        Returns:
            True if within tolerance of target.
        """
        return self.distance_to_target(current_position) <= self.tolerance


@dataclass
class RotationGoal(IKGoal):
    """Goal for achieving a target rotation.

    The IK solver will attempt to rotate the bone to match
    the target orientation.

    Attributes:
        target_rotation: Desired rotation as a quaternion
        tolerance: Angular threshold (radians) for goal achieved
    """

    target_rotation: Quat = field(default_factory=Quat.identity)
    tolerance: float = IK_ROTATION_TOLERANCE

    def __post_init__(self) -> None:
        object.__setattr__(self, 'goal_type', IKGoalType.ROTATION)

    def angular_distance(self, current_rotation: Quat) -> float:
        """Calculate angular distance to target rotation.

        Args:
            current_rotation: Current bone rotation

        Returns:
            Angular distance in radians.
        """
        import math
        dot = abs(current_rotation.dot(self.target_rotation))
        dot = min(dot, 1.0)  # Clamp for numerical stability
        return 2.0 * math.acos(dot)

    def is_achieved(self, current_rotation: Quat) -> bool:
        """Check if rotation goal is achieved within tolerance.

        Args:
            current_rotation: Current bone rotation

        Returns:
            True if within angular tolerance of target.
        """
        return self.angular_distance(current_rotation) <= self.tolerance


@dataclass
class LookAtGoal(IKGoal):
    """Goal for making a bone look at a target point.

    Commonly used for head/eye tracking where the bone's
    forward axis should point toward a target.

    Attributes:
        target_point: World-space point to look at
        forward_axis: Local axis of the bone that should point at target
        up_axis: Local up axis for roll control
        max_angle: Maximum rotation angle (radians) from rest pose
    """

    target_point: Vec3 = field(default_factory=Vec3.zero)
    forward_axis: Vec3 = field(default_factory=lambda: Vec3(0, 0, 1))
    up_axis: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    max_angle: float = LOOK_AT_MAX_ANGLE

    def __post_init__(self) -> None:
        object.__setattr__(self, 'goal_type', IKGoalType.LOOK_AT)

    def compute_look_rotation(self, bone_position: Vec3, current_rotation: Quat) -> Quat:
        """Compute rotation to look at target.

        Args:
            bone_position: Current bone world position
            current_rotation: Current bone rotation

        Returns:
            Rotation quaternion to look at target.
        """
        import math
        from engine.core.constants import MATH_EPSILON

        # Direction to target
        to_target = self.target_point - bone_position
        if to_target.length_squared() < MATH_EPSILON:
            return current_rotation

        to_target = to_target.normalized()

        # Current forward direction
        current_forward = current_rotation.rotate_vector(self.forward_axis)

        # Rotation axis and angle
        axis = current_forward.cross(to_target)
        if axis.length_squared() < MATH_EPSILON:
            # Vectors are parallel
            if current_forward.dot(to_target) > 0:
                return current_rotation
            # Opposite directions - rotate 180 around up
            axis = current_rotation.rotate_vector(self.up_axis)
        else:
            axis = axis.normalized()

        dot = max(-1.0, min(1.0, current_forward.dot(to_target)))
        angle = math.acos(dot)

        # Clamp to max angle
        if angle > self.max_angle:
            angle = self.max_angle

        rotation_delta = Quat.from_axis_angle(axis, angle)
        return rotation_delta * current_rotation


@dataclass
class PositionRotationGoal(IKGoal):
    """Combined position and rotation goal.

    Used when both the position and orientation of the end effector
    need to be controlled (e.g., hand placement on a surface).

    Attributes:
        target_position: World-space position target
        target_rotation: Desired rotation as quaternion
        position_weight: Weight for position component
        rotation_weight: Weight for rotation component
        position_tolerance: Distance threshold for position
        rotation_tolerance: Angular threshold for rotation
    """

    target_position: Vec3 = field(default_factory=Vec3.zero)
    target_rotation: Quat = field(default_factory=Quat.identity)
    position_weight: float = 1.0
    rotation_weight: float = 1.0
    position_tolerance: float = IK_DEFAULT_TOLERANCE
    rotation_tolerance: float = IK_ROTATION_TOLERANCE

    def __post_init__(self) -> None:
        object.__setattr__(self, 'goal_type', IKGoalType.POSITION_AND_ROTATION)

    def is_achieved(self, current_position: Vec3, current_rotation: Quat) -> bool:
        """Check if both position and rotation goals are achieved.

        Args:
            current_position: Current end effector position
            current_rotation: Current end effector rotation

        Returns:
            True if both are within their tolerances.
        """
        import math

        # Check position
        pos_dist = current_position.distance(self.target_position)
        if pos_dist > self.position_tolerance:
            return False

        # Check rotation
        dot = abs(current_rotation.dot(self.target_rotation))
        dot = min(dot, 1.0)
        rot_dist = 2.0 * math.acos(dot)
        if rot_dist > self.rotation_tolerance:
            return False

        return True


@dataclass
class PoleVectorGoal(IKGoal):
    """Pole vector goal for controlling chain plane.

    Defines the plane in which a chain should bend, commonly used
    for controlling elbow and knee directions.

    Attributes:
        pole_position: World-space position of the pole target
        twist_offset: Additional twist around the chain axis (radians)
    """

    pole_position: Vec3 = field(default_factory=Vec3.zero)
    twist_offset: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, 'goal_type', IKGoalType.POLE_VECTOR)

    def compute_pole_direction(self, root_position: Vec3, end_position: Vec3) -> Vec3:
        """Compute direction from chain center to pole.

        Args:
            root_position: Position of chain root
            end_position: Position of chain end effector

        Returns:
            Normalized direction toward pole.
        """
        from engine.core.constants import MATH_EPSILON

        # Mid-point of the chain
        mid = root_position.lerp(end_position, 0.5)

        # Direction to pole
        to_pole = self.pole_position - mid
        if to_pole.length_squared() < MATH_EPSILON:
            return Vec3.up()

        return to_pole.normalized()


@dataclass
class ChainGoal:
    """Goal for an entire IK chain.

    Combines multiple goals for a chain of bones, such as
    an arm or leg with position, rotation, and pole vector targets.

    Attributes:
        chain_name: Identifier for this chain
        bone_indices: List of bone indices in the chain (root to tip)
        end_effector_goal: Goal for the end effector (tip of chain)
        pole_goal: Optional pole vector goal
        maintain_length: Whether to maintain bone lengths
        stiffness: Resistance to bending (0 = flexible, 1 = stiff)
    """

    chain_name: str
    bone_indices: list[int]
    end_effector_goal: Optional[IKGoal] = None
    pole_goal: Optional[PoleVectorGoal] = None
    maintain_length: bool = True
    stiffness: float = 0.0

    @property
    def root_index(self) -> int:
        """Get the root bone index of the chain."""
        return self.bone_indices[0] if self.bone_indices else -1

    @property
    def end_index(self) -> int:
        """Get the end effector bone index of the chain."""
        return self.bone_indices[-1] if self.bone_indices else -1

    @property
    def chain_length(self) -> int:
        """Get the number of bones in the chain."""
        return len(self.bone_indices)

    def validate(self) -> bool:
        """Validate chain goal configuration.

        Returns:
            True if configuration is valid.
        """
        if not self.bone_indices:
            return False
        if len(self.bone_indices) < 2:
            return False
        if any(idx < 0 for idx in self.bone_indices):
            return False
        if self.stiffness < 0.0 or self.stiffness > 1.0:
            return False
        return True


@dataclass
class CenterOfMassGoal(IKGoal):
    """Goal for maintaining balance through center of mass positioning.

    Used in full-body IK to keep the character balanced by ensuring
    the center of mass stays within the support polygon.

    Attributes:
        target_com: Target center of mass position
        support_polygon: Vertices of support area (feet positions)
        bone_masses: Dictionary of bone_index -> mass for COM calculation
    """

    target_com: Vec3 = field(default_factory=Vec3.zero)
    support_polygon: list[Vec3] = field(default_factory=list)
    bone_masses: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'goal_type', IKGoalType.CENTER_OF_MASS)

    def is_balanced(self, current_com: Vec3) -> bool:
        """Check if COM is within support polygon.

        Args:
            current_com: Current center of mass position

        Returns:
            True if balanced (COM inside support polygon).
        """
        if len(self.support_polygon) < 3:
            return True  # Can't form polygon

        # Project to ground plane (XZ)
        point = Vec3(current_com.x, 0, current_com.z)

        # Point in polygon test (ray casting)
        n = len(self.support_polygon)
        inside = False

        j = n - 1
        for i in range(n):
            pi = self.support_polygon[i]
            pj = self.support_polygon[j]

            dz = pj.z - pi.z
            if abs(dz) < MATH_EPSILON:
                # Edge is horizontal, skip this edge for the ray cast test
                j = i
                continue
            if ((pi.z > point.z) != (pj.z > point.z)) and \
               (point.x < (pj.x - pi.x) * (point.z - pi.z) / dz + pi.x):
                inside = not inside
            j = i

        return inside


def ik_goal(priority: int = 0, blend_speed: float = GOAL_BLENDER_DEFAULT_SPEED):
    """Decorator for marking components as IK goals.

    This decorator follows the Trinity pattern from ANIMATION_CONTEXT.md
    for registering IK goals with the animation system.

    Args:
        priority: Goal priority (higher = more important)
        blend_speed: Speed to blend toward target

    Returns:
        Decorator function.
    """
    def decorator(cls):
        cls._ik_goal = True
        cls._ik_goal_priority = priority
        cls._ik_goal_blend_speed = blend_speed
        return cls
    return decorator


def ik_chain(solver: str = "fabrik", iterations: int = FABRIK_DEFAULT_MAX_ITERATIONS):
    """Decorator for defining IK chains.

    This decorator follows the Trinity pattern from ANIMATION_CONTEXT.md
    for registering IK chains with the animation system.

    Args:
        solver: Solver algorithm ("fabrik", "ccd", "jacobian", "fullbody")
        iterations: Maximum solver iterations

    Returns:
        Decorator function.
    """
    def decorator(cls):
        cls._ik_chain = True
        cls._ik_solver = solver
        cls._ik_iterations = iterations
        return cls
    return decorator


class IKGoalBlender:
    """Utility for blending between IK goals over time.

    Provides smooth transitions when IK targets change.
    """

    def __init__(self, blend_speed: float = GOAL_BLENDER_DEFAULT_SPEED) -> None:
        """Initialize the goal blender.

        Args:
            blend_speed: Default blend speed (units per second)
        """
        self.blend_speed = blend_speed
        self._current_positions: dict[int, Vec3] = {}
        self._current_rotations: dict[int, Quat] = {}

    def blend_position(
        self,
        goal_id: int,
        target: Vec3,
        dt: float,
        speed: Optional[float] = None
    ) -> Vec3:
        """Blend position toward target.

        Args:
            goal_id: Unique identifier for this goal
            target: Target position
            dt: Delta time
            speed: Optional override for blend speed

        Returns:
            Blended position for this frame.
        """
        speed = speed if speed is not None else self.blend_speed

        if goal_id not in self._current_positions:
            self._current_positions[goal_id] = target
            return target

        current = self._current_positions[goal_id]
        t = min(1.0, speed * dt)
        result = current.lerp(target, t)
        self._current_positions[goal_id] = result
        return result

    def blend_rotation(
        self,
        goal_id: int,
        target: Quat,
        dt: float,
        speed: Optional[float] = None
    ) -> Quat:
        """Blend rotation toward target.

        Args:
            goal_id: Unique identifier for this goal
            target: Target rotation
            dt: Delta time
            speed: Optional override for blend speed

        Returns:
            Blended rotation for this frame.
        """
        speed = speed if speed is not None else self.blend_speed

        if goal_id not in self._current_rotations:
            self._current_rotations[goal_id] = target
            return target

        current = self._current_rotations[goal_id]
        t = min(1.0, speed * dt)
        result = current.slerp(target, t)
        self._current_rotations[goal_id] = result
        return result

    def reset(self, goal_id: Optional[int] = None) -> None:
        """Reset blending state.

        Args:
            goal_id: Specific goal to reset, or None to reset all
        """
        if goal_id is None:
            self._current_positions.clear()
            self._current_rotations.clear()
        else:
            self._current_positions.pop(goal_id, None)
            self._current_rotations.pop(goal_id, None)
