"""
Moving Platform Handling.

Provides functionality for attaching characters to moving platforms,
handling platform velocity inheritance, and managing rotating platforms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .character_controller import Quaternion, Transform, Vector3
from .config import (
    MAX_PLATFORM_VELOCITY,
    PLATFORM_DETACH_THRESHOLD,
    PLATFORM_STICK_FORCE,
)


# =============================================================================
# Platform Types
# =============================================================================

class PlatformType(str, Enum):
    """Types of moving platforms."""
    LINEAR = "linear"       # Moves in straight lines
    ROTATING = "rotating"   # Rotates around an axis
    ELEVATOR = "elevator"   # Vertical movement only
    PATH = "path"          # Follows a path
    PHYSICS = "physics"     # Physics-driven (e.g., see-saw)


class AttachmentMode(str, Enum):
    """How the character attaches to platforms."""
    PARENT = "parent"      # Full transform parenting
    VELOCITY = "velocity"  # Inherit velocity only
    HYBRID = "hybrid"      # Parent position, inherit rotation velocity


# =============================================================================
# Platform Data
# =============================================================================

@dataclass
class PlatformData:
    """
    Data for a moving platform.

    Attributes:
        platform_id: Unique identifier
        platform_type: Type of platform movement
        transform: Current platform transform
        velocity: Linear velocity
        angular_velocity: Angular velocity (radians/sec)
        rotation_axis: Axis of rotation
        is_active: Whether platform is currently moving
    """
    platform_id: int = 0
    platform_type: PlatformType = PlatformType.LINEAR
    transform: Transform = field(default_factory=Transform)
    velocity: Vector3 = field(default_factory=Vector3.zero)
    angular_velocity: Vector3 = field(default_factory=Vector3.zero)
    rotation_axis: Vector3 = field(default_factory=Vector3.up)
    is_active: bool = True


@dataclass
class PlatformAttachment:
    """
    Information about character attachment to a platform.

    Attributes:
        platform_id: ID of attached platform
        local_offset: Local position offset on platform
        attachment_mode: How character is attached
        attachment_time: Time when attachment occurred
        previous_platform_transform: Transform from last frame
        inherited_velocity: Velocity inherited from platform
    """
    platform_id: int = 0
    local_offset: Vector3 = field(default_factory=Vector3.zero)
    local_rotation_offset: Quaternion = field(default_factory=Quaternion.identity)
    attachment_mode: AttachmentMode = AttachmentMode.HYBRID
    attachment_time: float = 0.0
    previous_platform_transform: Transform = field(default_factory=Transform)
    inherited_velocity: Vector3 = field(default_factory=Vector3.zero)


# =============================================================================
# Platform Provider Interface
# =============================================================================

class PlatformProvider:
    """Interface for accessing platform data. Override in implementation."""

    def get_platform(self, platform_id: int) -> Optional[PlatformData]:
        """Get platform data by ID."""
        return None

    def get_platform_transform(self, platform_id: int) -> Optional[Transform]:
        """Get current platform transform."""
        return None

    def get_platform_velocity(self, platform_id: int) -> Vector3:
        """Get platform linear velocity."""
        return Vector3.zero()

    def get_platform_angular_velocity(self, platform_id: int) -> Vector3:
        """Get platform angular velocity."""
        return Vector3.zero()


# =============================================================================
# Platform Handler
# =============================================================================

class PlatformHandler:
    """
    Handles character interaction with moving platforms.

    Features:
    - Platform attachment/detachment
    - Velocity inheritance
    - Rotating platform handling
    - Transform synchronization
    """

    def __init__(self, platform_provider: PlatformProvider):
        self._provider = platform_provider
        self._attachment: Optional[PlatformAttachment] = None
        self._stick_force = PLATFORM_STICK_FORCE
        self._detach_threshold = PLATFORM_DETACH_THRESHOLD
        self._max_velocity = MAX_PLATFORM_VELOCITY

        # Callbacks
        self._on_attach: Optional[Callable[[int], None]] = None
        self._on_detach: Optional[Callable[[int, Vector3], None]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_attached(self) -> bool:
        """Whether currently attached to a platform."""
        return self._attachment is not None

    @property
    def attached_platform_id(self) -> Optional[int]:
        """ID of currently attached platform."""
        return self._attachment.platform_id if self._attachment else None

    @property
    def attachment(self) -> Optional[PlatformAttachment]:
        """Current attachment data."""
        return self._attachment

    @property
    def inherited_velocity(self) -> Vector3:
        """Velocity inherited from platform."""
        if self._attachment:
            return self._attachment.inherited_velocity
        return Vector3.zero()

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_attach_callback(self, callback: Optional[Callable[[int], None]]) -> None:
        """Set callback for platform attachment."""
        self._on_attach = callback

    def set_detach_callback(
        self, callback: Optional[Callable[[int, Vector3], None]]
    ) -> None:
        """Set callback for platform detachment (receives platform ID and exit velocity)."""
        self._on_detach = callback

    # -------------------------------------------------------------------------
    # Attachment
    # -------------------------------------------------------------------------

    def attach_to_platform(
        self,
        platform_id: int,
        character_position: Vector3,
        character_rotation: Quaternion,
        mode: AttachmentMode = AttachmentMode.HYBRID,
        time: float = 0.0,
    ) -> bool:
        """
        Attach character to a platform.

        Args:
            platform_id: Platform to attach to
            character_position: Current character world position
            character_rotation: Current character world rotation
            mode: Attachment mode
            time: Current game time

        Returns:
            True if attachment was successful
        """
        platform = self._provider.get_platform(platform_id)
        if platform is None:
            return False

        # Calculate local offset
        local_offset = platform.transform.inverse_transform_point(character_position)

        # Store attachment
        self._attachment = PlatformAttachment(
            platform_id=platform_id,
            local_offset=local_offset,
            local_rotation_offset=character_rotation,  # Simplified
            attachment_mode=mode,
            attachment_time=time,
            previous_platform_transform=platform.transform,
            inherited_velocity=Vector3.zero(),
        )

        if self._on_attach:
            self._on_attach(platform_id)

        return True

    def detach_from_platform(self, preserve_velocity: bool = True) -> Vector3:
        """
        Detach from current platform.

        Args:
            preserve_velocity: Whether to preserve platform velocity

        Returns:
            Exit velocity (platform velocity at time of detachment)
        """
        if self._attachment is None:
            return Vector3.zero()

        exit_velocity = Vector3.zero()
        platform_id = self._attachment.platform_id

        if preserve_velocity:
            exit_velocity = self._get_current_platform_velocity()

        # Fire callback before clearing
        if self._on_detach:
            self._on_detach(platform_id, exit_velocity)

        self._attachment = None

        return exit_velocity

    # -------------------------------------------------------------------------
    # Platform Velocity
    # -------------------------------------------------------------------------

    def get_platform_velocity(self) -> Vector3:
        """
        Get the current velocity of the attached platform.

        Returns:
            Platform velocity or zero if not attached
        """
        if self._attachment is None:
            return Vector3.zero()

        return self._get_current_platform_velocity()

    def _get_current_platform_velocity(self) -> Vector3:
        """Get velocity from platform provider."""
        if self._attachment is None:
            return Vector3.zero()

        velocity = self._provider.get_platform_velocity(self._attachment.platform_id)

        # Clamp to maximum
        if velocity.magnitude() > self._max_velocity:
            velocity = velocity.normalized() * self._max_velocity

        return velocity

    def get_point_velocity(
        self, world_position: Vector3
    ) -> Vector3:
        """
        Get velocity at a specific point on the platform.

        Accounts for rotation causing tangential velocity.

        Args:
            world_position: Point in world space

        Returns:
            Velocity at that point
        """
        if self._attachment is None:
            return Vector3.zero()

        platform = self._provider.get_platform(self._attachment.platform_id)
        if platform is None:
            return Vector3.zero()

        # Linear velocity
        velocity = platform.velocity

        # Add tangential velocity from rotation
        if platform.angular_velocity.magnitude() > 0.001:
            # Vector from platform center to point
            to_point = world_position - platform.transform.position
            # Tangential velocity = angular velocity cross radius
            tangential = platform.angular_velocity.cross(to_point)
            velocity = velocity + tangential

        return velocity

    # -------------------------------------------------------------------------
    # Transform Updates
    # -------------------------------------------------------------------------

    def update(
        self,
        character_position: Vector3,
        character_rotation: Quaternion,
        dt: float,
    ) -> tuple[Vector3, Quaternion, Vector3]:
        """
        Update character transform based on platform movement.

        Args:
            character_position: Current character position
            character_rotation: Current character rotation
            dt: Delta time

        Returns:
            Tuple of (new_position, new_rotation, platform_velocity)
        """
        if self._attachment is None:
            return character_position, character_rotation, Vector3.zero()

        platform = self._provider.get_platform(self._attachment.platform_id)
        if platform is None:
            self.detach_from_platform()
            return character_position, character_rotation, Vector3.zero()

        mode = self._attachment.attachment_mode
        platform_velocity = Vector3.zero()

        if mode == AttachmentMode.PARENT:
            # Full parenting - character moves with platform
            new_position = platform.transform.transform_point(self._attachment.local_offset)

            # Apply rotation delta
            # Simplified: just inherit platform rotation
            new_rotation = character_rotation
            platform_velocity = self.get_point_velocity(new_position)

        elif mode == AttachmentMode.VELOCITY:
            # Velocity inheritance only - ensure full velocity is inherited including vertical
            platform_velocity = self.get_point_velocity(character_position)
            # Update local offset to track platform movement for next frame
            platform = self._provider.get_platform(self._attachment.platform_id)
            if platform is not None:
                self._attachment.local_offset = platform.transform.inverse_transform_point(character_position)
            new_position = character_position + platform_velocity * dt
            new_rotation = character_rotation

        else:  # HYBRID
            # Parent position, but only partial rotation inheritance
            new_position = platform.transform.transform_point(self._attachment.local_offset)
            platform_velocity = self.get_point_velocity(new_position)

            # Apply partial rotation (e.g., yaw only)
            new_rotation = character_rotation

        # Update stored velocity
        self._attachment.inherited_velocity = platform_velocity
        self._attachment.previous_platform_transform = platform.transform

        # Check for detachment conditions
        if self._should_auto_detach(platform_velocity):
            self.detach_from_platform(preserve_velocity=True)
            return character_position, character_rotation, platform_velocity

        return new_position, new_rotation, platform_velocity

    def transform_with_platform(
        self,
        position: Vector3,
        rotation: Quaternion,
    ) -> tuple[Vector3, Quaternion]:
        """
        Transform character with platform without time delta.

        Useful for teleporting or instant updates.

        Args:
            position: Character position
            rotation: Character rotation

        Returns:
            Transformed (position, rotation)
        """
        if self._attachment is None:
            return position, rotation

        platform = self._provider.get_platform(self._attachment.platform_id)
        if platform is None:
            return position, rotation

        new_position = platform.transform.transform_point(self._attachment.local_offset)
        return new_position, rotation

    def _should_auto_detach(self, platform_velocity: Vector3) -> bool:
        """Check if automatic detachment should occur."""
        # Detach if platform velocity is too extreme
        if platform_velocity.magnitude() > self._max_velocity * 1.5:
            return True

        return False

    # -------------------------------------------------------------------------
    # Rotating Platforms
    # -------------------------------------------------------------------------

    def handle_rotating_platform(
        self,
        character_position: Vector3,
        character_forward: Vector3,
        dt: float,
    ) -> tuple[Vector3, float]:
        """
        Handle character position and rotation on a rotating platform.

        Args:
            character_position: Current position
            character_forward: Current forward direction
            dt: Delta time

        Returns:
            Tuple of (new_position, yaw_delta)
        """
        if self._attachment is None:
            return character_position, 0.0

        platform = self._provider.get_platform(self._attachment.platform_id)
        if platform is None or platform.platform_type != PlatformType.ROTATING:
            return character_position, 0.0

        angular_vel = platform.angular_velocity
        if angular_vel.magnitude() < 0.001:
            return character_position, 0.0

        # Calculate rotation for this frame
        rotation_axis = angular_vel.normalized()
        rotation_angle = angular_vel.magnitude() * dt

        # Rotate position around platform center
        center = platform.transform.position
        to_char = character_position - center

        # Apply rotation (using Rodrigues' formula)
        rotated_pos = self._rotate_point_around_axis(
            to_char, rotation_axis, rotation_angle
        )

        new_position = center + rotated_pos

        # Calculate yaw change (assuming Y-up)
        yaw_delta = rotation_angle if rotation_axis.y > 0.5 else (
            -rotation_angle if rotation_axis.y < -0.5 else 0.0
        )

        return new_position, yaw_delta

    def _rotate_point_around_axis(
        self,
        point: Vector3,
        axis: Vector3,
        angle: float,
    ) -> Vector3:
        """Rotate a point around an axis using Rodrigues' formula."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # v' = v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1-cos(a))
        cross = axis.cross(point)
        dot = axis.dot(point)

        return point * cos_a + cross * sin_a + axis * dot * (1.0 - cos_a)

    # -------------------------------------------------------------------------
    # Platform Queries
    # -------------------------------------------------------------------------

    def is_on_moving_platform(self) -> bool:
        """Check if on an actively moving platform."""
        if self._attachment is None:
            return False

        platform = self._provider.get_platform(self._attachment.platform_id)
        if platform is None:
            return False

        return platform.is_active and (
            platform.velocity.magnitude() > 0.01 or
            platform.angular_velocity.magnitude() > 0.01
        )

    def get_platform_type(self) -> Optional[PlatformType]:
        """Get the type of attached platform."""
        if self._attachment is None:
            return None

        platform = self._provider.get_platform(self._attachment.platform_id)
        return platform.platform_type if platform else None

    def get_attachment_duration(self, current_time: float) -> float:
        """Get how long character has been attached."""
        if self._attachment is None:
            return 0.0
        return current_time - self._attachment.attachment_time

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def get_debug_info(self) -> dict[str, Any]:
        """Get debug information."""
        if self._attachment is None:
            return {"attached": False}

        platform = self._provider.get_platform(self._attachment.platform_id)

        return {
            "attached": True,
            "platform_id": self._attachment.platform_id,
            "platform_type": platform.platform_type.value if platform else "unknown",
            "local_offset": (
                self._attachment.local_offset.x,
                self._attachment.local_offset.y,
                self._attachment.local_offset.z,
            ),
            "inherited_velocity": (
                self._attachment.inherited_velocity.x,
                self._attachment.inherited_velocity.y,
                self._attachment.inherited_velocity.z,
            ),
            "mode": self._attachment.attachment_mode.value,
        }
