"""XR Avatar component with IK targets.

Provides full-body avatar representation for XR with inverse kinematics,
network synchronization, and visibility controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Annotated, Any, Callable, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import RigidTransform
from engine.xr.config import XR_CONFIG

if TYPE_CHECKING:
    from engine.xr.avatars.ik_solver import IKSolver


class AvatarVisibility(Enum):
    """Avatar visibility modes."""
    VISIBLE = auto()       # Visible to all
    HIDDEN = auto()        # Hidden from all
    SELF_HIDDEN = auto()   # Hidden from owner, visible to others
    OTHERS_HIDDEN = auto() # Visible to owner, hidden from others


class DisplayMode(Enum):
    """Hand display modes."""
    CONTROLLER = auto()  # Show controller model
    HAND = auto()        # Show hand model
    TOOL = auto()        # Show held tool


@dataclass(slots=True)
class IKTarget:
    """Inverse kinematics target position and orientation.

    Attributes:
        position: Target position in world space
        rotation: Target rotation as quaternion
        weight: IK influence weight (0-1)
        active: Whether this target is active
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    rotation: Quat = field(default_factory=Quat.identity)
    weight: float = 1.0
    active: bool = True

    def to_rigid_transform(self) -> RigidTransform:
        """Convert to rigid transform."""
        return RigidTransform(self.position, self.rotation)

    @staticmethod
    def from_rigid_transform(transform: RigidTransform, weight: float = 1.0) -> IKTarget:
        """Create from rigid transform."""
        return IKTarget(
            position=transform.translation,
            rotation=transform.rotation,
            weight=weight,
            active=True
        )


@dataclass(slots=True)
class PersonalSpace:
    """Personal space boundary for avatar safety.

    Prevents other avatars from getting too close, maintaining
    user comfort in social XR experiences.

    Attributes:
        radius: Personal space radius in meters
        enabled: Whether personal space is enforced
        push_strength: How strongly to push invaders away (0-1)
        fade_distance: Distance at which fading begins
        show_indicator: Whether to show visual boundary indicator
    """
    radius: float = 0.5
    enabled: bool = True
    push_strength: float = 0.5
    fade_distance: float = 0.3
    show_indicator: bool = True

    def is_invaded(self, other_position: Vec3, my_position: Vec3) -> bool:
        """Check if another avatar is invading personal space."""
        if not self.enabled:
            return False
        distance = my_position.distance(other_position)
        return distance < self.radius

    def get_push_vector(self, other_position: Vec3, my_position: Vec3) -> Vec3:
        """Calculate push vector to move invader out of personal space."""
        if not self.enabled:
            return Vec3.zero()

        direction = my_position - other_position
        distance = direction.length()

        if distance >= self.radius or distance < 0.001:
            return Vec3.zero()

        # Normalize and scale by invasion depth
        direction = direction.normalized()
        invasion_depth = self.radius - distance
        push_magnitude = invasion_depth * self.push_strength

        return direction * push_magnitude

    def get_fade_alpha(self, other_position: Vec3, my_position: Vec3) -> float:
        """Get fade alpha for rendering invading avatars (0 = invisible, 1 = full)."""
        if not self.enabled:
            return 1.0

        distance = my_position.distance(other_position)

        if distance >= self.radius:
            return 1.0

        inner_radius = self.radius - self.fade_distance
        if distance <= inner_radius:
            return 0.0

        # Linear fade in the fade zone
        return (distance - inner_radius) / self.fade_distance


def xr_avatar(
    ik_enabled: bool = True,
    network_sync: bool = True,
    face_tracking: bool = False,
) -> Callable[[type], type]:
    """Decorator to mark a class as an XR avatar component.

    Args:
        ik_enabled: Enable inverse kinematics for body estimation
        network_sync: Enable network synchronization for multiplayer
        face_tracking: Enable face/expression tracking

    Returns:
        Decorated class with XR avatar metadata
    """
    def decorator(cls: type) -> type:
        cls._xr_avatar = True
        cls._xr_avatar_ik_enabled = ik_enabled
        cls._xr_avatar_network_sync = network_sync
        cls._xr_avatar_face_tracking = face_tracking

        # Initialize tags
        if not hasattr(cls, "_tags"):
            cls._tags = {}
        cls._tags["xr_avatar"] = True
        cls._tags["xr_avatar_ik_enabled"] = ik_enabled
        cls._tags["xr_avatar_network_sync"] = network_sync
        cls._tags["xr_avatar_face_tracking"] = face_tracking

        # Track applied decorators
        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = []
        cls._applied_decorators.append("xr_avatar")

        return cls
    return decorator


def xr_ik_target(
    target_type: str = "hand",
    bone_chain: Optional[list[str]] = None,
) -> Callable[[type], type]:
    """Decorator to mark a class as an IK target point.

    Args:
        target_type: Type of IK target (head, hand, foot, etc.)
        bone_chain: List of bones in the IK chain

    Returns:
        Decorated class with IK target metadata
    """
    valid_types = {"head", "hand", "foot", "pelvis", "chest", "elbow", "knee"}
    if target_type not in valid_types:
        raise ValueError(f"Invalid target_type '{target_type}'. Valid: {valid_types}")

    if bone_chain is None:
        bone_chain = []

    def decorator(cls: type) -> type:
        cls._xr_ik_target = True
        cls._xr_ik_target_type = target_type
        cls._xr_ik_target_bone_chain = bone_chain

        if not hasattr(cls, "_tags"):
            cls._tags = {}
        cls._tags["xr_ik_target"] = True
        cls._tags["xr_ik_target_type"] = target_type
        cls._tags["xr_ik_target_bone_chain"] = bone_chain

        if not hasattr(cls, "_applied_decorators"):
            cls._applied_decorators = []
        cls._applied_decorators.append("xr_ik_target")

        return cls
    return decorator


@xr_avatar(ik_enabled=True, network_sync=True)
class XRAvatar:
    """Full-body XR avatar with inverse kinematics.

    The avatar uses IK targets from HMD and controllers to estimate
    body pose. Head position comes from HMD, hands from controllers
    or hand tracking, and body/legs are procedurally estimated.

    Attributes:
        head_target: IK target for head (from HMD)
        left_hand_target: IK target for left hand
        right_hand_target: IK target for right hand
        player_height: Calibrated player height in meters
        arm_span: Calibrated arm span in meters
        floor_level: Y position of floor in world space
        visibility: Avatar visibility mode
        personal_space: Personal space settings
        ik_solver: Optional custom IK solver
    """
    __slots__ = (
        '_head_target', '_left_hand_target', '_right_hand_target',
        '_player_height', '_arm_span', '_floor_level',
        '_visibility', '_personal_space', '_ik_solver',
        '_estimated_pelvis', '_estimated_chest',
        '_left_foot', '_right_foot',
        '_is_calibrated', '_network_id',
        '_mute_indicator', '_name_tag', '_name_tag_visible'
    )

    def __init__(
        self,
        player_height: float = XR_CONFIG.avatar.DEFAULT_AVATAR_HEIGHT_M,
        arm_span: float = XR_CONFIG.avatar.DEFAULT_AVATAR_HEIGHT_M,
    ) -> None:
        """Initialize XR avatar.

        Args:
            player_height: Player height in meters (default 1.7m)
            arm_span: Player arm span in meters (default 1.7m)
        """
        # IK targets
        self._head_target = IKTarget(
            position=Vec3(0.0, player_height, 0.0),
            rotation=Quat.identity(),
        )
        self._left_hand_target = IKTarget(
            position=Vec3(-0.3, player_height * 0.65, 0.0),
            rotation=Quat.identity(),
        )
        self._right_hand_target = IKTarget(
            position=Vec3(0.3, player_height * 0.65, 0.0),
            rotation=Quat.identity(),
        )

        # Calibration data
        self._player_height = player_height
        self._arm_span = arm_span
        self._floor_level = 0.0
        self._is_calibrated = False

        # Visibility and social
        self._visibility = AvatarVisibility.VISIBLE
        self._personal_space = PersonalSpace()
        self._mute_indicator = False
        self._name_tag = ""
        self._name_tag_visible = True

        # Networking
        self._network_id: Optional[int] = None

        # IK solver (lazily initialized)
        self._ik_solver: Optional[IKSolver] = None

        # Estimated body parts (computed by IK)
        self._estimated_pelvis = RigidTransform()
        self._estimated_chest = RigidTransform()
        self._left_foot = RigidTransform()
        self._right_foot = RigidTransform()

    @property
    def head_target(self) -> IKTarget:
        """Get head IK target."""
        return self._head_target

    @head_target.setter
    def head_target(self, value: IKTarget) -> None:
        """Set head IK target."""
        self._head_target = value

    @property
    def left_hand_target(self) -> IKTarget:
        """Get left hand IK target."""
        return self._left_hand_target

    @left_hand_target.setter
    def left_hand_target(self, value: IKTarget) -> None:
        """Set left hand IK target."""
        self._left_hand_target = value

    @property
    def right_hand_target(self) -> IKTarget:
        """Get right hand IK target."""
        return self._right_hand_target

    @right_hand_target.setter
    def right_hand_target(self, value: IKTarget) -> None:
        """Set right hand IK target."""
        self._right_hand_target = value

    @property
    def player_height(self) -> float:
        """Get calibrated player height."""
        return self._player_height

    @property
    def arm_span(self) -> float:
        """Get calibrated arm span."""
        return self._arm_span

    @property
    def floor_level(self) -> float:
        """Get floor Y level."""
        return self._floor_level

    @floor_level.setter
    def floor_level(self, value: float) -> None:
        """Set floor Y level."""
        self._floor_level = value

    @property
    def is_calibrated(self) -> bool:
        """Check if avatar is calibrated."""
        return self._is_calibrated

    @property
    def visibility(self) -> AvatarVisibility:
        """Get avatar visibility mode."""
        return self._visibility

    @visibility.setter
    def visibility(self, value: AvatarVisibility) -> None:
        """Set avatar visibility mode."""
        self._visibility = value

    @property
    def personal_space(self) -> PersonalSpace:
        """Get personal space settings."""
        return self._personal_space

    @property
    def mute_indicator(self) -> bool:
        """Check if mute indicator is shown."""
        return self._mute_indicator

    @mute_indicator.setter
    def mute_indicator(self, value: bool) -> None:
        """Set mute indicator visibility."""
        self._mute_indicator = value

    @property
    def name_tag(self) -> str:
        """Get name tag text."""
        return self._name_tag

    @name_tag.setter
    def name_tag(self, value: str) -> None:
        """Set name tag text."""
        self._name_tag = value

    @property
    def name_tag_visible(self) -> bool:
        """Check if name tag is visible."""
        return self._name_tag_visible

    @name_tag_visible.setter
    def name_tag_visible(self, value: bool) -> None:
        """Set name tag visibility."""
        self._name_tag_visible = value

    @property
    def estimated_pelvis(self) -> RigidTransform:
        """Get estimated pelvis transform (computed from IK)."""
        return self._estimated_pelvis

    @property
    def estimated_chest(self) -> RigidTransform:
        """Get estimated chest transform (computed from IK)."""
        return self._estimated_chest

    def set_ik_solver(self, solver: IKSolver) -> None:
        """Set custom IK solver.

        Args:
            solver: IK solver instance
        """
        self._ik_solver = solver

    def update_from_hmd(
        self,
        hmd_position: Vec3,
        hmd_rotation: Quat,
    ) -> None:
        """Update head target from HMD pose.

        Args:
            hmd_position: HMD world position
            hmd_rotation: HMD world rotation
        """
        self._head_target.position = hmd_position
        self._head_target.rotation = hmd_rotation
        self._head_target.active = True

    def update_from_controllers(
        self,
        left_position: Vec3,
        left_rotation: Quat,
        right_position: Vec3,
        right_rotation: Quat,
    ) -> None:
        """Update hand targets from controller poses.

        Args:
            left_position: Left controller world position
            left_rotation: Left controller world rotation
            right_position: Right controller world position
            right_rotation: Right controller world rotation
        """
        self._left_hand_target.position = left_position
        self._left_hand_target.rotation = left_rotation
        self._left_hand_target.active = True

        self._right_hand_target.position = right_position
        self._right_hand_target.rotation = right_rotation
        self._right_hand_target.active = True

    def calibrate(
        self,
        height: float,
        arm_span: float,
        floor_level: float = 0.0,
    ) -> None:
        """Calibrate avatar to player dimensions.

        Args:
            height: Player height in meters
            arm_span: Player arm span in meters
            floor_level: Floor Y position in world space
        """
        if height <= 0:
            raise ValueError("Height must be positive")
        if arm_span <= 0:
            raise ValueError("Arm span must be positive")

        self._player_height = height
        self._arm_span = arm_span
        self._floor_level = floor_level
        self._is_calibrated = True

    def estimate_body(self) -> None:
        """Estimate body pose from IK targets.

        Uses head and hand positions to estimate pelvis, chest,
        and procedural leg positions. Called each frame after
        updating IK targets.
        """
        if self._ik_solver is not None:
            # Use custom IK solver
            self._ik_solver.solve(
                self._head_target,
                self._left_hand_target,
                self._right_hand_target,
            )
            return

        # Simple body estimation without full IK solver
        head_pos = self._head_target.position
        head_rot = self._head_target.rotation

        # Estimate pelvis position (below head, offset by height ratio)
        pelvis_height = self._floor_level + self._player_height * 0.5
        pelvis_pos = Vec3(head_pos.x, pelvis_height, head_pos.z)

        # Pelvis rotation follows head yaw only
        pitch, yaw, roll = head_rot.to_euler()
        pelvis_rot = Quat.from_euler(0.0, yaw, 0.0)

        self._estimated_pelvis = RigidTransform(pelvis_pos, pelvis_rot)

        # Estimate chest (between head and pelvis)
        chest_height = (head_pos.y + pelvis_height) / 2
        chest_pos = Vec3(head_pos.x, chest_height, head_pos.z)
        chest_rot = pelvis_rot.slerp(head_rot, 0.5)

        self._estimated_chest = RigidTransform(chest_pos, chest_rot)

        # Simple procedural foot placement
        foot_y = self._floor_level
        foot_spread = 0.15  # Half stride width
        forward = pelvis_rot.forward()
        right = pelvis_rot.right()

        self._left_foot = RigidTransform(
            Vec3(
                pelvis_pos.x - right.x * foot_spread,
                foot_y,
                pelvis_pos.z - right.z * foot_spread,
            ),
            pelvis_rot,
        )
        self._right_foot = RigidTransform(
            Vec3(
                pelvis_pos.x + right.x * foot_spread,
                foot_y,
                pelvis_pos.z + right.z * foot_spread,
            ),
            pelvis_rot,
        )

    def is_visible_to(self, viewer_is_self: bool) -> bool:
        """Check if avatar is visible to a specific viewer.

        Args:
            viewer_is_self: True if viewer is the avatar owner

        Returns:
            True if avatar should be rendered for this viewer
        """
        if self._visibility == AvatarVisibility.HIDDEN:
            return False
        elif self._visibility == AvatarVisibility.SELF_HIDDEN:
            return not viewer_is_self
        elif self._visibility == AvatarVisibility.OTHERS_HIDDEN:
            return viewer_is_self
        else:  # VISIBLE
            return True

    def get_network_state(self) -> dict[str, Any]:
        """Get state for network synchronization.

        Returns:
            Dictionary of networked state
        """
        return {
            "head_position": (
                self._head_target.position.x,
                self._head_target.position.y,
                self._head_target.position.z,
            ),
            "head_rotation": (
                self._head_target.rotation.x,
                self._head_target.rotation.y,
                self._head_target.rotation.z,
                self._head_target.rotation.w,
            ),
            "left_hand_position": (
                self._left_hand_target.position.x,
                self._left_hand_target.position.y,
                self._left_hand_target.position.z,
            ),
            "left_hand_rotation": (
                self._left_hand_target.rotation.x,
                self._left_hand_target.rotation.y,
                self._left_hand_target.rotation.z,
                self._left_hand_target.rotation.w,
            ),
            "right_hand_position": (
                self._right_hand_target.position.x,
                self._right_hand_target.position.y,
                self._right_hand_target.position.z,
            ),
            "right_hand_rotation": (
                self._right_hand_target.rotation.x,
                self._right_hand_target.rotation.y,
                self._right_hand_target.rotation.z,
                self._right_hand_target.rotation.w,
            ),
            "mute_indicator": self._mute_indicator,
            "name_tag": self._name_tag,
        }

    def apply_network_state(self, state: dict[str, Any]) -> None:
        """Apply state from network synchronization.

        Args:
            state: Dictionary of networked state
        """
        if "head_position" in state:
            pos = state["head_position"]
            self._head_target.position = Vec3(pos[0], pos[1], pos[2])

        if "head_rotation" in state:
            rot = state["head_rotation"]
            self._head_target.rotation = Quat(rot[0], rot[1], rot[2], rot[3])

        if "left_hand_position" in state:
            pos = state["left_hand_position"]
            self._left_hand_target.position = Vec3(pos[0], pos[1], pos[2])

        if "left_hand_rotation" in state:
            rot = state["left_hand_rotation"]
            self._left_hand_target.rotation = Quat(rot[0], rot[1], rot[2], rot[3])

        if "right_hand_position" in state:
            pos = state["right_hand_position"]
            self._right_hand_target.position = Vec3(pos[0], pos[1], pos[2])

        if "right_hand_rotation" in state:
            rot = state["right_hand_rotation"]
            self._right_hand_target.rotation = Quat(rot[0], rot[1], rot[2], rot[3])

        if "mute_indicator" in state:
            self._mute_indicator = state["mute_indicator"]

        if "name_tag" in state:
            self._name_tag = state["name_tag"]
