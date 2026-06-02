"""Joint limit constraints for IK solvers.

This module provides abstract and concrete joint limit classes that constrain
rotations to valid ranges. These are used by CCD and other IK solvers to
enforce anatomically plausible joint constraints.

Joint limits work by clamping a quaternion rotation to satisfy constraints:

- **EulerLimit**: Clamp rotation per Euler axis (min/max for X, Y, Z)
- **SwingTwistLimit**: Decompose into swing cone + twist, clamp separately

Example usage:

    from engine.animation.ik.joint_limits import EulerOrder, EulerLimit, SwingTwistLimit

    # Elbow hinge: only bend in X axis
    elbow_limit = EulerLimit(
        min_x=-2.5,  # -143 degrees
        max_x=0.0,   # Can't hyperextend
        min_y=-0.1, max_y=0.1,  # Small twist allowance
        min_z=-0.1, max_z=0.1,
        order=EulerOrder.XYZ
    )

    # Shoulder ball socket: cone swing + limited twist
    shoulder_limit = SwingTwistLimit(
        swing_cone=math.pi / 2,  # 90 degree cone
        twist_min=-math.pi / 4,
        twist_max=math.pi / 4,
        twist_axis=Vec3(0, 1, 0)  # Y-axis twist
    )

    # Apply to rotation
    clamped = elbow_limit.clamp(rotation)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.constants import MATH_EPSILON


class EulerOrder(Enum):
    """Euler angle rotation order.

    Defines the order in which rotations around X, Y, Z axes are applied.
    The order matters because rotation composition is non-commutative.

    Common conventions:
    - XYZ: Often used in robotics (roll-pitch-yaw)
    - ZYX: Common in aerospace (yaw-pitch-roll)
    - YXZ: Common in game engines for characters
    """

    XYZ = auto()
    """X first, then Y, then Z."""

    XZY = auto()
    """X first, then Z, then Y."""

    YXZ = auto()
    """Y first, then X, then Z."""

    YZX = auto()
    """Y first, then Z, then X."""

    ZXY = auto()
    """Z first, then X, then Y."""

    ZYX = auto()
    """Z first, then Y, then X."""


class JointLimit(ABC):
    """Abstract base class for joint rotation limits.

    Joint limits constrain rotations to valid ranges, enforcing
    anatomically plausible or mechanically valid joint configurations.

    Subclasses must implement the clamp() method to enforce their
    specific constraint type.
    """

    @abstractmethod
    def clamp(self, rotation: Quat) -> Quat:
        """Clamp rotation to joint limits.

        Args:
            rotation: Input rotation quaternion (should be normalized)

        Returns:
            Clamped rotation quaternion (normalized)
        """
        pass


def _quat_to_euler_xyz(q: Quat) -> Tuple[float, float, float]:
    """Extract Euler angles with XYZ order from quaternion.

    Args:
        q: Input quaternion

    Returns:
        Tuple of (x, y, z) angles in radians.
    """
    # XYZ order: R = Rz * Ry * Rx
    # Using rotation matrix extraction
    x, y, z, w = q.x, q.y, q.z, q.w

    # Compute rotation matrix elements
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    # Extract XYZ Euler angles
    # Check for gimbal lock
    if abs(r02) >= 1.0 - MATH_EPSILON:
        # Gimbal lock at Y = +/- 90 degrees
        angle_y = math.copysign(math.pi / 2, r02)
        angle_z = 0.0
        angle_x = math.atan2(-r21, r11)
    else:
        angle_y = math.asin(max(-1.0, min(1.0, r02)))
        angle_x = math.atan2(-r12, r22)
        angle_z = math.atan2(-r01, r00)

    return (angle_x, angle_y, angle_z)


def _quat_to_euler_xzy(q: Quat) -> Tuple[float, float, float]:
    """Extract Euler angles with XZY order from quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    if abs(r01) >= 1.0 - MATH_EPSILON:
        angle_z = math.copysign(math.pi / 2, -r01)
        angle_y = 0.0
        angle_x = math.atan2(r20, r22)
    else:
        angle_z = math.asin(max(-1.0, min(1.0, -r01)))
        angle_x = math.atan2(r21, r11)
        angle_y = math.atan2(r02, r00)

    return (angle_x, angle_y, angle_z)


def _quat_to_euler_yxz(q: Quat) -> Tuple[float, float, float]:
    """Extract Euler angles with YXZ order from quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    if abs(r12) >= 1.0 - MATH_EPSILON:
        angle_x = math.copysign(math.pi / 2, -r12)
        angle_z = 0.0
        angle_y = math.atan2(-r20, r00)
    else:
        angle_x = math.asin(max(-1.0, min(1.0, -r12)))
        angle_y = math.atan2(r02, r22)
        angle_z = math.atan2(r10, r11)

    return (angle_x, angle_y, angle_z)


def _quat_to_euler_yzx(q: Quat) -> Tuple[float, float, float]:
    """Extract Euler angles with YZX order from quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    if abs(r10) >= 1.0 - MATH_EPSILON:
        angle_z = math.copysign(math.pi / 2, r10)
        angle_x = 0.0
        angle_y = math.atan2(r02, r22)
    else:
        angle_z = math.asin(max(-1.0, min(1.0, r10)))
        angle_y = math.atan2(-r20, r00)
        angle_x = math.atan2(-r12, r11)

    return (angle_x, angle_y, angle_z)


def _quat_to_euler_zxy(q: Quat) -> Tuple[float, float, float]:
    """Extract Euler angles with ZXY order from quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    if abs(r21) >= 1.0 - MATH_EPSILON:
        angle_x = math.copysign(math.pi / 2, r21)
        angle_y = 0.0
        angle_z = math.atan2(r10, r00)
    else:
        angle_x = math.asin(max(-1.0, min(1.0, r21)))
        angle_z = math.atan2(-r01, r11)
        angle_y = math.atan2(-r20, r22)

    return (angle_x, angle_y, angle_z)


def _quat_to_euler_zyx(q: Quat) -> Tuple[float, float, float]:
    """Extract Euler angles with ZYX order from quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    if abs(r20) >= 1.0 - MATH_EPSILON:
        angle_y = math.copysign(math.pi / 2, -r20)
        angle_x = 0.0
        angle_z = math.atan2(-r01, r11)
    else:
        angle_y = math.asin(max(-1.0, min(1.0, -r20)))
        angle_z = math.atan2(r10, r00)
        angle_x = math.atan2(r21, r22)

    return (angle_x, angle_y, angle_z)


def quat_to_euler(q: Quat, order: EulerOrder) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles with specified order.

    Args:
        q: Input quaternion (should be normalized)
        order: Euler rotation order

    Returns:
        Tuple of (x, y, z) rotation angles in radians.
    """
    q = q.normalized()

    if order == EulerOrder.XYZ:
        return _quat_to_euler_xyz(q)
    elif order == EulerOrder.XZY:
        return _quat_to_euler_xzy(q)
    elif order == EulerOrder.YXZ:
        return _quat_to_euler_yxz(q)
    elif order == EulerOrder.YZX:
        return _quat_to_euler_yzx(q)
    elif order == EulerOrder.ZXY:
        return _quat_to_euler_zxy(q)
    else:  # ZYX
        return _quat_to_euler_zyx(q)


def euler_to_quat(x: float, y: float, z: float, order: EulerOrder) -> Quat:
    """Convert Euler angles to quaternion with specified order.

    Args:
        x: X-axis rotation in radians
        y: Y-axis rotation in radians
        z: Z-axis rotation in radians
        order: Euler rotation order

    Returns:
        Rotation quaternion.
    """
    # Create individual axis rotations
    qx = Quat.from_axis_angle(Vec3(1, 0, 0), x)
    qy = Quat.from_axis_angle(Vec3(0, 1, 0), y)
    qz = Quat.from_axis_angle(Vec3(0, 0, 1), z)

    # Compose in the specified order
    # Note: quaternion multiplication is applied right-to-left
    # So XYZ order means: result = qz * qy * qx (apply X first)
    if order == EulerOrder.XYZ:
        return qz * qy * qx
    elif order == EulerOrder.XZY:
        return qy * qz * qx
    elif order == EulerOrder.YXZ:
        return qz * qx * qy
    elif order == EulerOrder.YZX:
        return qx * qz * qy
    elif order == EulerOrder.ZXY:
        return qy * qx * qz
    else:  # ZYX
        return qx * qy * qz


@dataclass
class EulerLimit(JointLimit):
    """Joint limit using Euler angle constraints.

    Clamps rotation by extracting Euler angles, clamping each axis
    independently, and reconstructing the quaternion.

    This is useful for joints where you want independent control over
    each rotation axis, like hinge joints (one active axis) or
    constrained ball joints.

    Note: Euler angles can suffer from gimbal lock at extreme angles.
    For more robust constraints at full range of motion, consider
    SwingTwistLimit.

    Attributes:
        min_x: Minimum X-axis rotation in radians
        max_x: Maximum X-axis rotation in radians
        min_y: Minimum Y-axis rotation in radians
        max_y: Maximum Y-axis rotation in radians
        min_z: Minimum Z-axis rotation in radians
        max_z: Maximum Z-axis rotation in radians
        order: Euler rotation order
    """

    min_x: float = -math.pi
    max_x: float = math.pi
    min_y: float = -math.pi
    max_y: float = math.pi
    min_z: float = -math.pi
    max_z: float = math.pi
    order: EulerOrder = EulerOrder.XYZ

    def __post_init__(self) -> None:
        """Validate limit ranges."""
        if self.min_x > self.max_x:
            self.min_x, self.max_x = self.max_x, self.min_x
        if self.min_y > self.max_y:
            self.min_y, self.max_y = self.max_y, self.min_y
        if self.min_z > self.max_z:
            self.min_z, self.max_z = self.max_z, self.min_z

    def clamp(self, rotation: Quat) -> Quat:
        """Clamp rotation to Euler angle limits.

        Args:
            rotation: Input rotation quaternion

        Returns:
            Clamped rotation quaternion.
        """
        # Extract Euler angles with specified order
        x, y, z = quat_to_euler(rotation, self.order)

        # Clamp each axis
        x = max(self.min_x, min(x, self.max_x))
        y = max(self.min_y, min(y, self.max_y))
        z = max(self.min_z, min(z, self.max_z))

        # Reconstruct quaternion
        return euler_to_quat(x, y, z, self.order).normalized()

    def is_within_limits(self, rotation: Quat) -> bool:
        """Check if rotation is within limits.

        Args:
            rotation: Rotation to check

        Returns:
            True if rotation is within all axis limits.
        """
        x, y, z = quat_to_euler(rotation, self.order)

        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and self.min_z <= z <= self.max_z
        )


def _decompose_swing_twist(rotation: Quat, twist_axis: Vec3) -> Tuple[Quat, Quat]:
    """Decompose rotation into swing and twist components.

    The decomposition separates a rotation into:
    - Twist: Rotation around the specified axis
    - Swing: Remaining rotation perpendicular to the axis

    Such that: rotation = swing * twist

    Args:
        rotation: Input rotation quaternion
        twist_axis: Axis for twist decomposition (should be normalized)

    Returns:
        Tuple of (swing, twist) quaternions.
    """
    rotation = rotation.normalized()
    twist_axis = twist_axis.normalized()

    # Project quaternion vector part onto twist axis
    # q = (sin(a/2) * axis, cos(a/2))
    quat_vector = Vec3(rotation.x, rotation.y, rotation.z)
    projection = twist_axis * quat_vector.dot(twist_axis)

    # Twist quaternion from projection
    twist = Quat(projection.x, projection.y, projection.z, rotation.w)
    twist_len = twist.length()

    if twist_len < MATH_EPSILON:
        # No twist component
        twist = Quat.identity()
    else:
        twist = Quat(
            twist.x / twist_len,
            twist.y / twist_len,
            twist.z / twist_len,
            twist.w / twist_len,
        )

    # Swing = rotation * twist^-1
    swing = rotation * twist.inverse()
    swing = swing.normalized()

    return (swing, twist)


def _clamp_twist(twist: Quat, twist_axis: Vec3, min_angle: float, max_angle: float) -> Quat:
    """Clamp twist rotation to angle limits.

    Args:
        twist: Twist quaternion
        twist_axis: Axis of twist rotation
        min_angle: Minimum twist angle (radians)
        max_angle: Maximum twist angle (radians)

    Returns:
        Clamped twist quaternion.
    """
    twist = twist.normalized()
    twist_axis = twist_axis.normalized()

    # Extract twist angle
    # For a quaternion representing rotation around axis:
    # q = (sin(a/2) * axis, cos(a/2))
    quat_vector = Vec3(twist.x, twist.y, twist.z)
    sin_half = quat_vector.length()

    if sin_half < MATH_EPSILON:
        # No twist
        return Quat.identity()

    cos_half = twist.w
    angle = 2.0 * math.atan2(sin_half, cos_half)

    # Normalize angle to [-pi, pi]
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi

    # Check sign: if quaternion axis is opposite to twist_axis, negate angle
    quat_axis = quat_vector.normalized()
    if quat_axis.dot(twist_axis) < 0:
        angle = -angle

    # Clamp angle
    clamped_angle = max(min_angle, min(angle, max_angle))

    # Reconstruct twist quaternion
    return Quat.from_axis_angle(twist_axis, clamped_angle)


def _clamp_swing(swing: Quat, cone_angle: float) -> Quat:
    """Clamp swing rotation to cone angle.

    Args:
        swing: Swing quaternion
        cone_angle: Half-angle of swing cone (radians)

    Returns:
        Clamped swing quaternion.
    """
    swing = swing.normalized()

    # Extract swing angle from quaternion
    quat_vector = Vec3(swing.x, swing.y, swing.z)
    sin_half = quat_vector.length()

    if sin_half < MATH_EPSILON:
        # No swing
        return Quat.identity()

    cos_half = swing.w
    angle = 2.0 * math.atan2(sin_half, cos_half)

    # Normalize angle to [0, 2*pi]
    if angle < 0:
        angle += 2.0 * math.pi

    # Clamp to cone
    if angle > cone_angle:
        # Reduce angle while keeping axis
        swing_axis = quat_vector.normalized()
        return Quat.from_axis_angle(swing_axis, cone_angle)

    return swing


@dataclass
class SwingTwistLimit(JointLimit):
    """Joint limit using swing-twist decomposition.

    Decomposes rotation into:
    - Swing: Rotation that moves the twist axis to a new direction
    - Twist: Rotation around the twist axis

    The swing is constrained to a cone, and the twist has min/max limits.
    This provides a more natural constraint for ball-socket joints like
    shoulders and hips.

    Advantages over Euler limits:
    - No gimbal lock issues
    - More intuitive for anatomical joints
    - Better behavior at extreme angles

    Attributes:
        swing_cone: Half-angle of swing cone in radians (0 = locked, pi = free)
        twist_min: Minimum twist angle in radians
        twist_max: Maximum twist angle in radians
        twist_axis: Axis for twist decomposition (default Y-axis)
    """

    swing_cone: float = math.pi
    twist_min: float = -math.pi
    twist_max: float = math.pi
    twist_axis: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))

    def __post_init__(self) -> None:
        """Validate and normalize parameters."""
        # Ensure cone angle is non-negative
        self.swing_cone = abs(self.swing_cone)

        # Swap min/max if needed
        if self.twist_min > self.twist_max:
            self.twist_min, self.twist_max = self.twist_max, self.twist_min

        # Normalize twist axis
        if self.twist_axis.length_squared() < MATH_EPSILON:
            self.twist_axis = Vec3(0, 1, 0)
        else:
            self.twist_axis = self.twist_axis.normalized()

    def clamp(self, rotation: Quat) -> Quat:
        """Clamp rotation using swing-twist decomposition.

        Args:
            rotation: Input rotation quaternion

        Returns:
            Clamped rotation quaternion.
        """
        # Decompose into swing and twist
        swing, twist = _decompose_swing_twist(rotation, self.twist_axis)

        # Clamp swing to cone
        clamped_swing = _clamp_swing(swing, self.swing_cone)

        # Clamp twist to angle limits
        clamped_twist = _clamp_twist(twist, self.twist_axis, self.twist_min, self.twist_max)

        # Recompose: rotation = swing * twist
        result = clamped_swing * clamped_twist
        return result.normalized()

    def is_within_limits(self, rotation: Quat) -> bool:
        """Check if rotation is within limits.

        Args:
            rotation: Rotation to check

        Returns:
            True if rotation is within swing cone and twist limits.
        """
        swing, twist = _decompose_swing_twist(rotation, self.twist_axis)

        # Check swing angle
        swing_vec = Vec3(swing.x, swing.y, swing.z)
        swing_angle = 2.0 * math.atan2(swing_vec.length(), abs(swing.w))
        if swing_angle > self.swing_cone:
            return False

        # Check twist angle
        twist_vec = Vec3(twist.x, twist.y, twist.z)
        twist_sin_half = twist_vec.length()
        if twist_sin_half < MATH_EPSILON:
            twist_angle = 0.0
        else:
            twist_angle = 2.0 * math.atan2(twist_sin_half, twist.w)
            # Normalize to [-pi, pi]
            while twist_angle > math.pi:
                twist_angle -= 2.0 * math.pi
            while twist_angle < -math.pi:
                twist_angle += 2.0 * math.pi

        if twist_angle < self.twist_min or twist_angle > self.twist_max:
            return False

        return True

    def get_swing_angle(self, rotation: Quat) -> float:
        """Get the swing angle of a rotation.

        Args:
            rotation: Input rotation

        Returns:
            Swing angle in radians.
        """
        swing, _ = _decompose_swing_twist(rotation, self.twist_axis)
        swing_vec = Vec3(swing.x, swing.y, swing.z)
        return 2.0 * math.atan2(swing_vec.length(), abs(swing.w))

    def get_twist_angle(self, rotation: Quat) -> float:
        """Get the twist angle of a rotation.

        Args:
            rotation: Input rotation

        Returns:
            Twist angle in radians [-pi, pi].
        """
        _, twist = _decompose_swing_twist(rotation, self.twist_axis)
        twist_vec = Vec3(twist.x, twist.y, twist.z)
        twist_sin_half = twist_vec.length()

        if twist_sin_half < MATH_EPSILON:
            return 0.0

        angle = 2.0 * math.atan2(twist_sin_half, twist.w)

        # Check if twist axis is flipped
        if twist_vec.normalized().dot(self.twist_axis) < 0:
            angle = -angle

        # Normalize to [-pi, pi]
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle


@dataclass
class HingeLimit(JointLimit):
    """Simplified hinge joint limit.

    Constrains rotation to a single axis with min/max angle.
    This is a specialized case of EulerLimit optimized for hinge joints
    like elbows and knees.

    Attributes:
        axis: Hinge axis (rotation happens around this axis)
        min_angle: Minimum rotation angle in radians
        max_angle: Maximum rotation angle in radians
    """

    axis: Vec3 = field(default_factory=lambda: Vec3(1, 0, 0))
    min_angle: float = -math.pi
    max_angle: float = math.pi

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.axis.length_squared() < MATH_EPSILON:
            self.axis = Vec3(1, 0, 0)
        else:
            self.axis = self.axis.normalized()

        if self.min_angle > self.max_angle:
            self.min_angle, self.max_angle = self.max_angle, self.min_angle

    def clamp(self, rotation: Quat) -> Quat:
        """Clamp rotation to hinge axis with angle limits.

        Projects the rotation onto the hinge axis and clamps the angle.

        Args:
            rotation: Input rotation quaternion

        Returns:
            Clamped rotation quaternion.
        """
        rotation = rotation.normalized()

        # Project rotation onto hinge axis
        quat_vec = Vec3(rotation.x, rotation.y, rotation.z)
        projection = self.axis * quat_vec.dot(self.axis)

        # Extract angle
        sin_half = projection.length()
        cos_half = rotation.w

        if sin_half < MATH_EPSILON:
            return Quat.identity()

        angle = 2.0 * math.atan2(sin_half, cos_half)

        # Check sign
        proj_axis = projection.normalized()
        if proj_axis.dot(self.axis) < 0:
            angle = -angle

        # Normalize to [-pi, pi]
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi

        # Clamp
        clamped_angle = max(self.min_angle, min(angle, self.max_angle))

        return Quat.from_axis_angle(self.axis, clamped_angle)


# Utility functions for common joint configurations


def create_elbow_limit(
    min_bend: float = 0.0,
    max_bend: float = 2.5,
    axis: Vec3 = None,
) -> HingeLimit:
    """Create a typical elbow joint limit.

    Elbows are hinge joints that only bend in one direction.

    Args:
        min_bend: Minimum bend angle (0 = straight, negative = hyperextension)
        max_bend: Maximum bend angle in radians
        axis: Bend axis (default: X-axis)

    Returns:
        HingeLimit configured for elbow.
    """
    if axis is None:
        axis = Vec3(1, 0, 0)
    return HingeLimit(axis=axis, min_angle=min_bend, max_angle=max_bend)


def create_knee_limit(
    min_bend: float = 0.0,
    max_bend: float = 2.5,
    axis: Vec3 = None,
) -> HingeLimit:
    """Create a typical knee joint limit.

    Knees bend opposite to elbows (negative angle).

    Args:
        min_bend: Minimum bend angle
        max_bend: Maximum bend angle in radians
        axis: Bend axis (default: X-axis)

    Returns:
        HingeLimit configured for knee.
    """
    if axis is None:
        axis = Vec3(1, 0, 0)
    return HingeLimit(axis=axis, min_angle=-max_bend, max_angle=-min_bend)


def create_shoulder_limit(
    swing_cone: float = math.pi * 0.6,
    twist_range: float = math.pi * 0.5,
) -> SwingTwistLimit:
    """Create a typical shoulder joint limit.

    Shoulders have wide range of motion with limited twist.

    Args:
        swing_cone: Half-angle of swing cone (default ~108 degrees)
        twist_range: Symmetric twist range (default ~90 degrees)

    Returns:
        SwingTwistLimit configured for shoulder.
    """
    return SwingTwistLimit(
        swing_cone=swing_cone,
        twist_min=-twist_range,
        twist_max=twist_range,
        twist_axis=Vec3(0, 1, 0),
    )


def create_hip_limit(
    swing_cone: float = math.pi * 0.4,
    twist_range: float = math.pi * 0.3,
) -> SwingTwistLimit:
    """Create a typical hip joint limit.

    Hips have more restricted range than shoulders.

    Args:
        swing_cone: Half-angle of swing cone (default ~72 degrees)
        twist_range: Symmetric twist range (default ~54 degrees)

    Returns:
        SwingTwistLimit configured for hip.
    """
    return SwingTwistLimit(
        swing_cone=swing_cone,
        twist_min=-twist_range,
        twist_max=twist_range,
        twist_axis=Vec3(0, -1, 0),  # Y-down for legs
    )
