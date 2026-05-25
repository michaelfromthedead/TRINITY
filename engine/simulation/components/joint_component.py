"""
Joint Component.

Provides physics joint/constraint components for connecting rigid bodies
with various joint types including fixed, hinge, ball socket, and more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from ..character.character_controller import Quaternion, Vector3


class JointType(str, Enum):
    """Type of physics joint."""
    FIXED = "fixed"           # No relative motion
    HINGE = "hinge"           # Single axis rotation
    BALL_SOCKET = "ball"      # Spherical joint
    SLIDER = "slider"         # Linear motion along axis
    CONE_TWIST = "cone_twist" # Ball with limits
    SPRING = "spring"         # Spring constraint
    DISTANCE = "distance"     # Fixed distance
    D6 = "d6"                 # 6-DOF configurable


class JointMotion(str, Enum):
    """Motion type for joint axes."""
    LOCKED = "locked"
    LIMITED = "limited"
    FREE = "free"


@dataclass
class JointLimits:
    """
    Limits for joint motion.

    Attributes:
        lower: Lower limit (angle or distance)
        upper: Upper limit
        stiffness: Spring stiffness at limits
        damping: Damping at limits
        restitution: Bounciness at limits
        contact_distance: Distance to start applying limit
    """
    lower: float = 0.0
    upper: float = 0.0
    stiffness: float = 0.0
    damping: float = 0.0
    restitution: float = 0.0
    contact_distance: float = 0.01


@dataclass
class JointDrive:
    """
    Drive configuration for motorized joints.

    Attributes:
        stiffness: Spring stiffness (position)
        damping: Damping coefficient (velocity)
        max_force: Maximum force/torque
        target: Target position/rotation
        target_velocity: Target velocity
        mode: "position", "velocity", or "acceleration"
    """
    stiffness: float = 0.0
    damping: float = 0.0
    max_force: float = float("inf")
    target: float = 0.0
    target_velocity: float = 0.0
    mode: str = "position"


class JointComponent:
    """
    Component for physics joints/constraints.

    Provides:
    - Multiple joint types
    - Configurable limits and drives
    - Breaking force thresholds
    - Event callbacks
    """

    def __init__(
        self,
        entity_id: int,
        joint_type: JointType = JointType.FIXED,
    ):
        self._entity_id = entity_id
        self._joint_type = joint_type

        # Connected bodies
        self._connected_body_a: Optional[int] = None
        self._connected_body_b: Optional[int] = None

        # Anchors
        self._anchor_a = Vector3.zero()
        self._anchor_b = Vector3.zero()
        self._axis = Vector3(1.0, 0.0, 0.0)
        self._secondary_axis = Vector3(0.0, 1.0, 0.0)

        # Limits
        self._limits: dict[str, JointLimits] = {}
        self._motion_x = JointMotion.LOCKED
        self._motion_y = JointMotion.LOCKED
        self._motion_z = JointMotion.LOCKED
        self._angular_motion_x = JointMotion.LOCKED
        self._angular_motion_y = JointMotion.LOCKED
        self._angular_motion_z = JointMotion.LOCKED

        # Drives
        self._drives: dict[str, JointDrive] = {}

        # Breaking
        self._break_force = float("inf")
        self._break_torque = float("inf")
        self._is_broken = False

        # State
        self._joint_id: Optional[int] = None
        self._enabled = True

        # Callbacks
        self._on_break: Optional[Callable[[], None]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def entity_id(self) -> int:
        """Entity this joint belongs to."""
        return self._entity_id

    @property
    def joint_id(self) -> Optional[int]:
        """Physics joint ID."""
        return self._joint_id

    @property
    def joint_type(self) -> JointType:
        """Type of joint."""
        return self._joint_type

    @property
    def connected_body_a(self) -> Optional[int]:
        """First connected body."""
        return self._connected_body_a

    @property
    def connected_body_b(self) -> Optional[int]:
        """Second connected body."""
        return self._connected_body_b

    @property
    def is_broken(self) -> bool:
        """Whether joint is broken."""
        return self._is_broken

    @property
    def enabled(self) -> bool:
        """Whether joint is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def anchor_a(self) -> Vector3:
        """Anchor point on body A (local space)."""
        return self._anchor_a

    @anchor_a.setter
    def anchor_a(self, value: Vector3) -> None:
        self._anchor_a = value

    @property
    def anchor_b(self) -> Vector3:
        """Anchor point on body B (local space)."""
        return self._anchor_b

    @anchor_b.setter
    def anchor_b(self, value: Vector3) -> None:
        self._anchor_b = value

    @property
    def axis(self) -> Vector3:
        """Primary joint axis."""
        return self._axis

    @axis.setter
    def axis(self, value: Vector3) -> None:
        self._axis = value.normalized()

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    def connect(
        self,
        body_a: int,
        body_b: Optional[int] = None,
        anchor_a: Optional[Vector3] = None,
        anchor_b: Optional[Vector3] = None,
    ) -> None:
        """
        Connect bodies to this joint.

        Args:
            body_a: First body ID
            body_b: Second body ID (None for world anchor)
            anchor_a: Anchor on body A (local space)
            anchor_b: Anchor on body B (local space)
        """
        self._connected_body_a = body_a
        self._connected_body_b = body_b

        if anchor_a is not None:
            self._anchor_a = anchor_a
        if anchor_b is not None:
            self._anchor_b = anchor_b

    def disconnect(self) -> None:
        """Disconnect the joint."""
        self._connected_body_a = None
        self._connected_body_b = None

    # -------------------------------------------------------------------------
    # Limits
    # -------------------------------------------------------------------------

    def set_linear_limit(self, axis: str, limits: JointLimits) -> None:
        """
        Set linear motion limits.

        Args:
            axis: "x", "y", or "z"
            limits: Limit configuration
        """
        self._limits[f"linear_{axis}"] = limits

    def set_angular_limit(self, axis: str, limits: JointLimits) -> None:
        """
        Set angular motion limits.

        Args:
            axis: "x", "y", or "z"
            limits: Limit configuration
        """
        self._limits[f"angular_{axis}"] = limits

    def set_motion(
        self,
        linear_x: JointMotion = JointMotion.LOCKED,
        linear_y: JointMotion = JointMotion.LOCKED,
        linear_z: JointMotion = JointMotion.LOCKED,
        angular_x: JointMotion = JointMotion.LOCKED,
        angular_y: JointMotion = JointMotion.LOCKED,
        angular_z: JointMotion = JointMotion.LOCKED,
    ) -> None:
        """Set motion freedom for all axes."""
        self._motion_x = linear_x
        self._motion_y = linear_y
        self._motion_z = linear_z
        self._angular_motion_x = angular_x
        self._angular_motion_y = angular_y
        self._angular_motion_z = angular_z

    def configure_hinge(
        self,
        axis: Vector3,
        lower_angle: float = -3.14159,
        upper_angle: float = 3.14159,
        use_limits: bool = True,
    ) -> None:
        """Configure as hinge joint."""
        self._joint_type = JointType.HINGE
        self._axis = axis.normalized()

        if use_limits:
            self._angular_motion_x = JointMotion.LIMITED
            self.set_angular_limit("x", JointLimits(
                lower=lower_angle,
                upper=upper_angle,
            ))
        else:
            self._angular_motion_x = JointMotion.FREE

    def configure_ball_socket(
        self,
        swing_limit: float = 1.57,
        twist_limit: float = 1.57,
    ) -> None:
        """Configure as ball socket joint."""
        self._joint_type = JointType.BALL_SOCKET

        self._angular_motion_x = JointMotion.LIMITED
        self._angular_motion_y = JointMotion.LIMITED
        self._angular_motion_z = JointMotion.LIMITED

        self.set_angular_limit("x", JointLimits(lower=-twist_limit, upper=twist_limit))
        self.set_angular_limit("y", JointLimits(lower=-swing_limit, upper=swing_limit))
        self.set_angular_limit("z", JointLimits(lower=-swing_limit, upper=swing_limit))

    def configure_slider(
        self,
        axis: Vector3,
        lower_distance: float = 0.0,
        upper_distance: float = 1.0,
    ) -> None:
        """Configure as slider joint."""
        self._joint_type = JointType.SLIDER
        self._axis = axis.normalized()

        self._motion_x = JointMotion.LIMITED
        self.set_linear_limit("x", JointLimits(
            lower=lower_distance,
            upper=upper_distance,
        ))

    def configure_spring(
        self,
        stiffness: float = 100.0,
        damping: float = 10.0,
        rest_length: float = 1.0,
    ) -> None:
        """Configure as spring joint."""
        self._joint_type = JointType.SPRING

        self._motion_x = JointMotion.FREE
        self.set_drive("x", JointDrive(
            stiffness=stiffness,
            damping=damping,
            target=rest_length,
            mode="position",
        ))

    # -------------------------------------------------------------------------
    # Drives
    # -------------------------------------------------------------------------

    def set_drive(self, axis: str, drive: JointDrive) -> None:
        """
        Set drive configuration for an axis.

        Args:
            axis: "x", "y", "z", or "slerp" for rotation
            drive: Drive configuration
        """
        self._drives[axis] = drive

    def set_target_position(self, axis: str, position: float) -> None:
        """Set target position for drive."""
        if axis in self._drives:
            self._drives[axis].target = position

    def set_target_velocity(self, axis: str, velocity: float) -> None:
        """Set target velocity for drive."""
        if axis in self._drives:
            self._drives[axis].target_velocity = velocity

    def set_target_rotation(self, rotation: Quaternion) -> None:
        """Set target rotation for angular drives."""
        # Would convert to per-axis targets
        pass

    # -------------------------------------------------------------------------
    # Breaking
    # -------------------------------------------------------------------------

    def set_break_thresholds(
        self,
        force: float = float("inf"),
        torque: float = float("inf"),
    ) -> None:
        """Set force/torque thresholds for breaking."""
        self._break_force = force
        self._break_torque = torque

    def set_break_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set callback for when joint breaks."""
        self._on_break = callback

    def check_break(self, force: float, torque: float) -> bool:
        """
        Check if joint should break.

        Args:
            force: Current force on joint
            torque: Current torque on joint

        Returns:
            True if joint broke
        """
        if self._is_broken:
            return True

        if force > self._break_force or torque > self._break_torque:
            self._is_broken = True
            self._enabled = False

            if self._on_break:
                self._on_break()

            return True

        return False

    def repair(self) -> None:
        """Repair a broken joint."""
        self._is_broken = False
        self._enabled = True

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_current_force(self) -> Vector3:
        """Get current constraint force."""
        # Would query physics engine
        return Vector3.zero()

    def get_current_torque(self) -> Vector3:
        """Get current constraint torque."""
        # Would query physics engine
        return Vector3.zero()

    def get_position(self) -> float:
        """Get current joint position (for prismatic/hinge)."""
        # Would query physics engine
        return 0.0

    def get_velocity(self) -> float:
        """Get current joint velocity."""
        # Would query physics engine
        return 0.0

    def get_angle(self) -> float:
        """Get current joint angle (for hinge)."""
        # Would query physics engine
        return 0.0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self, joint_id: int) -> None:
        """Initialize with physics joint ID."""
        self._joint_id = joint_id

    def cleanup(self) -> None:
        """Cleanup component."""
        self._joint_id = None

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return {
            "entity_id": self._entity_id,
            "joint_type": self._joint_type.value,
            "connected_body_a": self._connected_body_a,
            "connected_body_b": self._connected_body_b,
            "anchor_a": (self._anchor_a.x, self._anchor_a.y, self._anchor_a.z),
            "anchor_b": (self._anchor_b.x, self._anchor_b.y, self._anchor_b.z),
            "axis": (self._axis.x, self._axis.y, self._axis.z),
            "break_force": self._break_force,
            "break_torque": self._break_torque,
            "is_broken": self._is_broken,
            "enabled": self._enabled,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load from serialized state."""
        self._joint_type = JointType(state.get("joint_type", "fixed"))
        self._connected_body_a = state.get("connected_body_a")
        self._connected_body_b = state.get("connected_body_b")

        anchor = state.get("anchor_a", (0, 0, 0))
        self._anchor_a = Vector3(anchor[0], anchor[1], anchor[2])

        anchor = state.get("anchor_b", (0, 0, 0))
        self._anchor_b = Vector3(anchor[0], anchor[1], anchor[2])

        axis = state.get("axis", (1, 0, 0))
        self._axis = Vector3(axis[0], axis[1], axis[2])

        self._break_force = state.get("break_force", float("inf"))
        self._break_torque = state.get("break_torque", float("inf"))
        self._is_broken = state.get("is_broken", False)
        self._enabled = state.get("enabled", True)


__all__ = [
    "JointType",
    "JointMotion",
    "JointLimits",
    "JointDrive",
    "JointComponent",
]
