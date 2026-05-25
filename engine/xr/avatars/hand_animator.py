"""Hand and finger animation for XR avatars.

Provides hand pose animation from controller input or hand tracking data,
including finger curl, pose library, and physics-based hand interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat


class FingerName(Enum):
    """Finger identifiers."""
    THUMB = auto()
    INDEX = auto()
    MIDDLE = auto()
    RING = auto()
    PINKY = auto()


class HandPoseType(Enum):
    """Predefined hand pose types."""
    OPEN = auto()       # All fingers extended
    FIST = auto()       # All fingers curled
    POINT = auto()      # Index extended, others curled
    PINCH = auto()      # Thumb and index touching
    GRIP = auto()       # Fingers partially curled for gripping
    THUMBS_UP = auto()  # Thumb extended, others curled
    PEACE = auto()      # Index and middle extended
    ROCK = auto()       # Index and pinky extended
    OK = auto()         # Thumb and index form circle


@dataclass(slots=True)
class FingerCurl:
    """Curl values for a single finger.

    Each value is 0.0 (extended) to 1.0 (fully curled).

    Attributes:
        curl: Overall finger curl amount
        spread: Finger spread from center (-1 to 1)
        twist: Finger twist/roll rotation
    """
    curl: float = 0.0
    spread: float = 0.0
    twist: float = 0.0

    def __post_init__(self) -> None:
        self.curl = max(0.0, min(1.0, self.curl))
        self.spread = max(-1.0, min(1.0, self.spread))
        self.twist = max(-1.0, min(1.0, self.twist))

    def lerp(self, other: FingerCurl, t: float) -> FingerCurl:
        """Interpolate to another finger curl state.

        Args:
            other: Target finger curl
            t: Interpolation factor (0-1)

        Returns:
            Interpolated finger curl
        """
        return FingerCurl(
            curl=self.curl + (other.curl - self.curl) * t,
            spread=self.spread + (other.spread - self.spread) * t,
            twist=self.twist + (other.twist - self.twist) * t,
        )


@dataclass(slots=True)
class HandPose:
    """Complete hand pose with all finger curls.

    Attributes:
        thumb: Thumb curl state
        index: Index finger curl state
        middle: Middle finger curl state
        ring: Ring finger curl state
        pinky: Pinky finger curl state
        wrist_position: Wrist position (optional)
        wrist_rotation: Wrist rotation (optional)
    """
    thumb: FingerCurl = field(default_factory=FingerCurl)
    index: FingerCurl = field(default_factory=FingerCurl)
    middle: FingerCurl = field(default_factory=FingerCurl)
    ring: FingerCurl = field(default_factory=FingerCurl)
    pinky: FingerCurl = field(default_factory=FingerCurl)
    wrist_position: Optional[Vec3] = None
    wrist_rotation: Optional[Quat] = None

    def get_finger(self, finger: FingerName) -> FingerCurl:
        """Get curl state for a specific finger.

        Args:
            finger: Finger to get

        Returns:
            Finger curl state
        """
        if finger == FingerName.THUMB:
            return self.thumb
        elif finger == FingerName.INDEX:
            return self.index
        elif finger == FingerName.MIDDLE:
            return self.middle
        elif finger == FingerName.RING:
            return self.ring
        else:
            return self.pinky

    def set_finger(self, finger: FingerName, curl: FingerCurl) -> None:
        """Set curl state for a specific finger.

        Args:
            finger: Finger to set
            curl: New curl state
        """
        if finger == FingerName.THUMB:
            self.thumb = curl
        elif finger == FingerName.INDEX:
            self.index = curl
        elif finger == FingerName.MIDDLE:
            self.middle = curl
        elif finger == FingerName.RING:
            self.ring = curl
        else:
            self.pinky = curl

    def lerp(self, other: HandPose, t: float) -> HandPose:
        """Interpolate to another hand pose.

        Args:
            other: Target hand pose
            t: Interpolation factor (0-1)

        Returns:
            Interpolated hand pose
        """
        result = HandPose(
            thumb=self.thumb.lerp(other.thumb, t),
            index=self.index.lerp(other.index, t),
            middle=self.middle.lerp(other.middle, t),
            ring=self.ring.lerp(other.ring, t),
            pinky=self.pinky.lerp(other.pinky, t),
        )

        # Interpolate wrist if both poses have it
        if self.wrist_position and other.wrist_position:
            result.wrist_position = self.wrist_position.lerp(other.wrist_position, t)
        if self.wrist_rotation and other.wrist_rotation:
            result.wrist_rotation = self.wrist_rotation.slerp(other.wrist_rotation, t)

        return result

    def to_tuple(self) -> tuple[float, ...]:
        """Convert to tuple of curl values.

        Returns:
            Tuple of (thumb, index, middle, ring, pinky) curl values
        """
        return (
            self.thumb.curl,
            self.index.curl,
            self.middle.curl,
            self.ring.curl,
            self.pinky.curl,
        )

    @staticmethod
    def from_tuple(curls: tuple[float, ...]) -> HandPose:
        """Create from tuple of curl values.

        Args:
            curls: Tuple of (thumb, index, middle, ring, pinky) curl values

        Returns:
            Hand pose
        """
        return HandPose(
            thumb=FingerCurl(curl=curls[0] if len(curls) > 0 else 0.0),
            index=FingerCurl(curl=curls[1] if len(curls) > 1 else 0.0),
            middle=FingerCurl(curl=curls[2] if len(curls) > 2 else 0.0),
            ring=FingerCurl(curl=curls[3] if len(curls) > 3 else 0.0),
            pinky=FingerCurl(curl=curls[4] if len(curls) > 4 else 0.0),
        )


class PoseLibrary:
    """Library of predefined hand poses.

    Provides common hand poses and allows registering custom poses.
    """

    _poses: dict[str, HandPose] = {}

    @classmethod
    def initialize_defaults(cls) -> None:
        """Initialize default pose library."""
        cls._poses = {
            "open": HandPose(
                thumb=FingerCurl(curl=0.0, spread=0.3),
                index=FingerCurl(curl=0.0, spread=0.1),
                middle=FingerCurl(curl=0.0, spread=0.0),
                ring=FingerCurl(curl=0.0, spread=-0.1),
                pinky=FingerCurl(curl=0.0, spread=-0.2),
            ),
            "fist": HandPose(
                thumb=FingerCurl(curl=1.0),
                index=FingerCurl(curl=1.0),
                middle=FingerCurl(curl=1.0),
                ring=FingerCurl(curl=1.0),
                pinky=FingerCurl(curl=1.0),
            ),
            "point": HandPose(
                thumb=FingerCurl(curl=0.8),
                index=FingerCurl(curl=0.0),
                middle=FingerCurl(curl=1.0),
                ring=FingerCurl(curl=1.0),
                pinky=FingerCurl(curl=1.0),
            ),
            "pinch": HandPose(
                thumb=FingerCurl(curl=0.5),
                index=FingerCurl(curl=0.5),
                middle=FingerCurl(curl=0.2),
                ring=FingerCurl(curl=0.2),
                pinky=FingerCurl(curl=0.2),
            ),
            "grip": HandPose(
                thumb=FingerCurl(curl=0.6),
                index=FingerCurl(curl=0.7),
                middle=FingerCurl(curl=0.7),
                ring=FingerCurl(curl=0.7),
                pinky=FingerCurl(curl=0.7),
            ),
            "thumbs_up": HandPose(
                thumb=FingerCurl(curl=0.0, spread=0.5),
                index=FingerCurl(curl=1.0),
                middle=FingerCurl(curl=1.0),
                ring=FingerCurl(curl=1.0),
                pinky=FingerCurl(curl=1.0),
            ),
            "peace": HandPose(
                thumb=FingerCurl(curl=0.8),
                index=FingerCurl(curl=0.0, spread=0.2),
                middle=FingerCurl(curl=0.0, spread=-0.2),
                ring=FingerCurl(curl=1.0),
                pinky=FingerCurl(curl=1.0),
            ),
            "rock": HandPose(
                thumb=FingerCurl(curl=0.8),
                index=FingerCurl(curl=0.0, spread=0.2),
                middle=FingerCurl(curl=1.0),
                ring=FingerCurl(curl=1.0),
                pinky=FingerCurl(curl=0.0, spread=-0.2),
            ),
            "ok": HandPose(
                thumb=FingerCurl(curl=0.4),
                index=FingerCurl(curl=0.4),
                middle=FingerCurl(curl=0.0),
                ring=FingerCurl(curl=0.0),
                pinky=FingerCurl(curl=0.0),
            ),
        }

    @classmethod
    def get(cls, name: str) -> Optional[HandPose]:
        """Get a pose by name.

        Args:
            name: Pose name

        Returns:
            Hand pose or None if not found
        """
        if not cls._poses:
            cls.initialize_defaults()
        return cls._poses.get(name.lower())

    @classmethod
    def get_by_type(cls, pose_type: HandPoseType) -> HandPose:
        """Get a pose by type.

        Args:
            pose_type: Pose type enum

        Returns:
            Hand pose
        """
        if not cls._poses:
            cls.initialize_defaults()

        name = pose_type.name.lower()
        pose = cls._poses.get(name)
        if pose is None:
            return HandPose()
        return pose

    @classmethod
    def register(cls, name: str, pose: HandPose) -> None:
        """Register a custom pose.

        Args:
            name: Pose name
            pose: Hand pose to register
        """
        if not cls._poses:
            cls.initialize_defaults()
        cls._poses[name.lower()] = pose

    @classmethod
    def list_poses(cls) -> list[str]:
        """List all registered pose names.

        Returns:
            List of pose names
        """
        if not cls._poses:
            cls.initialize_defaults()
        return list(cls._poses.keys())


class AvatarHand:
    """Avatar hand with finger animation.

    Handles finger curl animation from controller input or hand tracking,
    with support for pose blending and physics interaction.
    """
    __slots__ = (
        '_hand_side', '_current_pose', '_target_pose',
        '_blend_speed', '_display_mode', '_held_tool_id',
        '_physics_enabled', '_collision_enabled',
        '_grip_strength', '_pinch_strength'
    )

    def __init__(
        self,
        hand_side: str = "left",
        blend_speed: float = 10.0,
    ) -> None:
        """Initialize avatar hand.

        Args:
            hand_side: "left" or "right"
            blend_speed: Pose blend speed (poses per second)
        """
        if hand_side not in ("left", "right"):
            raise ValueError("hand_side must be 'left' or 'right'")
        if blend_speed <= 0:
            raise ValueError("blend_speed must be positive")

        self._hand_side = hand_side
        self._current_pose = HandPose()
        self._target_pose = HandPose()
        self._blend_speed = blend_speed
        self._display_mode = "hand"  # hand, controller, tool
        self._held_tool_id: Optional[int] = None
        self._physics_enabled = True
        self._collision_enabled = True
        self._grip_strength = 0.0
        self._pinch_strength = 0.0

    @property
    def hand_side(self) -> str:
        """Get hand side (left/right)."""
        return self._hand_side

    @property
    def current_pose(self) -> HandPose:
        """Get current hand pose."""
        return self._current_pose

    @property
    def target_pose(self) -> HandPose:
        """Get target hand pose."""
        return self._target_pose

    @property
    def display_mode(self) -> str:
        """Get display mode (hand/controller/tool)."""
        return self._display_mode

    @display_mode.setter
    def display_mode(self, value: str) -> None:
        """Set display mode."""
        if value not in ("hand", "controller", "tool"):
            raise ValueError("display_mode must be 'hand', 'controller', or 'tool'")
        self._display_mode = value

    @property
    def held_tool_id(self) -> Optional[int]:
        """Get ID of held tool (if any)."""
        return self._held_tool_id

    @held_tool_id.setter
    def held_tool_id(self, value: Optional[int]) -> None:
        """Set held tool ID."""
        self._held_tool_id = value
        if value is not None:
            self._display_mode = "tool"

    @property
    def grip_strength(self) -> float:
        """Get grip strength (0-1)."""
        return self._grip_strength

    @property
    def pinch_strength(self) -> float:
        """Get pinch strength (0-1)."""
        return self._pinch_strength

    @property
    def thumb_curl(self) -> float:
        """Get thumb curl value."""
        return self._current_pose.thumb.curl

    @property
    def index_curl(self) -> float:
        """Get index finger curl value."""
        return self._current_pose.index.curl

    @property
    def middle_curl(self) -> float:
        """Get middle finger curl value."""
        return self._current_pose.middle.curl

    @property
    def ring_curl(self) -> float:
        """Get ring finger curl value."""
        return self._current_pose.ring.curl

    @property
    def pinky_curl(self) -> float:
        """Get pinky finger curl value."""
        return self._current_pose.pinky.curl

    def set_pose(self, pose: HandPose) -> None:
        """Set target pose immediately.

        Args:
            pose: Target hand pose
        """
        self._target_pose = pose

    def set_pose_by_name(self, name: str) -> bool:
        """Set target pose by name from library.

        Args:
            name: Pose name

        Returns:
            True if pose was found and set
        """
        pose = PoseLibrary.get(name)
        if pose is None:
            return False
        self._target_pose = pose
        return True

    def set_pose_by_type(self, pose_type: HandPoseType) -> None:
        """Set target pose by type.

        Args:
            pose_type: Pose type enum
        """
        self._target_pose = PoseLibrary.get_by_type(pose_type)

    def update_from_controller(
        self,
        trigger_value: float,
        grip_value: float,
        thumbstick_touched: bool = False,
    ) -> None:
        """Update hand pose from controller input.

        Maps trigger and grip values to finger curls.

        Args:
            trigger_value: Trigger value (0-1)
            grip_value: Grip value (0-1)
            thumbstick_touched: Whether thumbstick is being touched
        """
        trigger_value = max(0.0, min(1.0, trigger_value))
        grip_value = max(0.0, min(1.0, grip_value))

        # Update stored strengths
        self._grip_strength = grip_value
        self._pinch_strength = trigger_value * 0.5  # Approximate

        # Thumb curl based on thumbstick touch
        thumb_curl = 0.7 if thumbstick_touched else 0.0

        # Index follows trigger
        index_curl = trigger_value

        # Middle, ring, pinky follow grip
        self._target_pose = HandPose(
            thumb=FingerCurl(curl=thumb_curl),
            index=FingerCurl(curl=index_curl),
            middle=FingerCurl(curl=grip_value),
            ring=FingerCurl(curl=grip_value),
            pinky=FingerCurl(curl=grip_value),
        )

    def update_from_hand_tracking(
        self,
        joint_positions: list[tuple[float, float, float]],
        joint_orientations: list[tuple[float, float, float, float]],
    ) -> None:
        """Update hand pose from hand tracking data.

        Computes finger curls from joint positions.

        Args:
            joint_positions: 26 joint positions
            joint_orientations: 26 joint orientations
        """
        if len(joint_positions) < 26:
            return

        # Calculate curl for each finger based on joint angles
        # This is a simplified calculation - real implementation would
        # compute angles between consecutive joints

        def calculate_finger_curl(
            metacarpal_idx: int,
            proximal_idx: int,
            intermediate_idx: int,
            distal_idx: int,
            tip_idx: int,
        ) -> float:
            """Calculate finger curl from joint positions."""
            meta = Vec3(*joint_positions[metacarpal_idx])
            prox = Vec3(*joint_positions[proximal_idx])
            tip = Vec3(*joint_positions[tip_idx])

            # Simplified: measure how close tip is to metacarpal
            extended_length = meta.distance(prox) * 3.0  # Approximate extended length
            if extended_length < 0.001:
                return 0.0
            actual_distance = meta.distance(tip)

            curl = 1.0 - (actual_distance / extended_length)
            return max(0.0, min(1.0, curl))

        # Joint indices (from HandJoint enum pattern)
        # Wrist=0, then each finger has: metacarpal, proximal, intermediate/distal, tip
        thumb_curl = calculate_finger_curl(1, 2, 3, 3, 4)
        index_curl = calculate_finger_curl(5, 6, 7, 8, 9)
        middle_curl = calculate_finger_curl(10, 11, 12, 13, 14)
        ring_curl = calculate_finger_curl(15, 16, 17, 18, 19)
        pinky_curl = calculate_finger_curl(20, 21, 22, 23, 24)

        self._target_pose = HandPose(
            thumb=FingerCurl(curl=thumb_curl),
            index=FingerCurl(curl=index_curl),
            middle=FingerCurl(curl=middle_curl),
            ring=FingerCurl(curl=ring_curl),
            pinky=FingerCurl(curl=pinky_curl),
        )

        # Update pinch strength from thumb-index distance
        thumb_tip = Vec3(*joint_positions[4])
        index_tip = Vec3(*joint_positions[9])
        pinch_distance = thumb_tip.distance(index_tip)
        self._pinch_strength = max(0.0, 1.0 - pinch_distance / 0.05)  # 5cm = no pinch

        # Update grip strength from average curl
        self._grip_strength = (
            index_curl + middle_curl + ring_curl + pinky_curl
        ) / 4.0

    def update(self, delta_time: float) -> None:
        """Update pose interpolation.

        Call each frame to smoothly blend toward target pose.

        Args:
            delta_time: Time since last update in seconds
        """
        if delta_time <= 0:
            return

        # Calculate blend factor
        t = min(1.0, delta_time * self._blend_speed)

        # Interpolate current toward target
        self._current_pose = self._current_pose.lerp(self._target_pose, t)

    def snap_to_target(self) -> None:
        """Immediately snap current pose to target pose."""
        self._current_pose = HandPose(
            thumb=FingerCurl(
                curl=self._target_pose.thumb.curl,
                spread=self._target_pose.thumb.spread,
                twist=self._target_pose.thumb.twist,
            ),
            index=FingerCurl(
                curl=self._target_pose.index.curl,
                spread=self._target_pose.index.spread,
                twist=self._target_pose.index.twist,
            ),
            middle=FingerCurl(
                curl=self._target_pose.middle.curl,
                spread=self._target_pose.middle.spread,
                twist=self._target_pose.middle.twist,
            ),
            ring=FingerCurl(
                curl=self._target_pose.ring.curl,
                spread=self._target_pose.ring.spread,
                twist=self._target_pose.ring.twist,
            ),
            pinky=FingerCurl(
                curl=self._target_pose.pinky.curl,
                spread=self._target_pose.pinky.spread,
                twist=self._target_pose.pinky.twist,
            ),
        )

    def get_network_state(self) -> dict:
        """Get state for network synchronization.

        Returns:
            Dictionary of networked state
        """
        return {
            "hand_side": self._hand_side,
            "pose": self._current_pose.to_tuple(),
            "display_mode": self._display_mode,
            "held_tool_id": self._held_tool_id,
        }

    def apply_network_state(self, state: dict) -> None:
        """Apply state from network synchronization.

        Args:
            state: Dictionary of networked state
        """
        if "pose" in state:
            self._target_pose = HandPose.from_tuple(state["pose"])
        if "display_mode" in state:
            self._display_mode = state["display_mode"]
        if "held_tool_id" in state:
            self._held_tool_id = state["held_tool_id"]
