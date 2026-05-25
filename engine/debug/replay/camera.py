"""Replay camera system.

This module provides camera modes for viewing replays, including
follow, free, POV, and orbit camera modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class Vec3:
    """Simple 3D vector for camera calculations.

    This is a minimal implementation. In production, use the engine's
    native vector type.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self.__mul__(scalar)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> Vec3:
        length = self.length()
        if length < 1e-8:
            return Vec3(0.0, 0.0, 1.0)
        return Vec3(self.x / length, self.y / length, self.z / length)

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def lerp(self, target: Vec3, t: float) -> Vec3:
        """Linear interpolation towards target."""
        return Vec3(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
            self.z + (target.z - self.z) * t,
        )

    def copy(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)


@dataclass
class Mat4:
    """Simple 4x4 matrix for view matrix calculations.

    Stored in column-major order (OpenGL convention).
    """
    data: list[float] = field(default_factory=lambda: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])

    @classmethod
    def identity(cls) -> Mat4:
        """Create an identity matrix."""
        return cls()

    @classmethod
    def look_at(cls, eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
        """Create a view matrix looking from eye to target.

        Args:
            eye: Camera position
            target: Point to look at
            up: Up direction vector

        Returns:
            View matrix
        """
        # Calculate basis vectors
        forward = (target - eye).normalized()
        right = forward.cross(up).normalized()
        new_up = right.cross(forward)

        # Build view matrix (column-major)
        m = cls()
        m.data[0] = right.x
        m.data[1] = new_up.x
        m.data[2] = -forward.x
        m.data[3] = 0.0

        m.data[4] = right.y
        m.data[5] = new_up.y
        m.data[6] = -forward.y
        m.data[7] = 0.0

        m.data[8] = right.z
        m.data[9] = new_up.z
        m.data[10] = -forward.z
        m.data[11] = 0.0

        m.data[12] = -right.dot(eye)
        m.data[13] = -new_up.dot(eye)
        m.data[14] = forward.dot(eye)
        m.data[15] = 1.0

        return m


class ReplayCameraMode(Enum):
    """Camera mode for replay viewing.

    Attributes:
        FOLLOW: Follow a target entity from a fixed offset
        FREE: Free-flying camera controlled by user input
        POV: First-person view from entity's perspective
        ORBIT: Orbit around a target entity
    """
    FOLLOW = auto()
    FREE = auto()
    POV = auto()
    ORBIT = auto()


@runtime_checkable
class EntityProvider(Protocol):
    """Protocol for providing entity transform data.

    Implement this to provide entity positions and rotations
    to the replay camera.
    """

    def get_entity_position(self, entity_id: int) -> Vec3 | None:
        """Get the position of an entity.

        Args:
            entity_id: The entity identifier

        Returns:
            Position vector, or None if entity not found
        """
        ...

    def get_entity_forward(self, entity_id: int) -> Vec3 | None:
        """Get the forward direction of an entity.

        Args:
            entity_id: The entity identifier

        Returns:
            Forward direction vector, or None if entity not found
        """
        ...


@dataclass
class CameraSettings:
    """Settings for replay camera behavior.

    Attributes:
        follow_offset: Offset from target for FOLLOW mode (default: 5 units up, 10 back)
        follow_smooth: Smoothing factor for follow (0-1, lower = smoother)
        orbit_distance: Distance from target for ORBIT mode (units)
        orbit_speed: Rotation speed for ORBIT mode (rad/s)
        pov_offset: Eye offset for POV mode (default: 1.8m eye height)
        free_move_speed: Movement speed for FREE mode (units/s)
        free_look_speed: Look speed for FREE mode (rad/s)
        min_distance: Minimum orbit/follow distance (units)
        max_distance: Maximum orbit/follow distance (units)

    Note:
        These defaults are tuned for a typical third-person game with
        meter-scale units. Adjust based on your game's scale.
    """
    # Camera positioning defaults (in game units, typically meters)
    follow_offset: Vec3 = field(default_factory=lambda: Vec3(0.0, 5.0, -10.0))  # 5m up, 10m behind
    follow_smooth: float = 0.1  # Low value for smooth follow (0.1 = 10% per frame adjustment)
    orbit_distance: float = 10.0  # 10 units from target
    orbit_speed: float = 1.0  # 1 radian/second (full orbit in ~6.28s)
    pov_offset: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.8, 0.0))  # 1.8m = average human eye height
    free_move_speed: float = 10.0  # 10 units/second
    free_look_speed: float = 2.0  # 2 radians/second
    min_distance: float = 1.0  # Minimum 1 unit from target
    max_distance: float = 100.0  # Maximum 100 units from target


class ReplayCamera:
    """Camera controller for replay viewing.

    The ReplayCamera provides different viewing modes for watching
    replay playback. It can follow entities, orbit around them,
    show their POV, or fly freely through the scene.

    Example:
        camera = ReplayCamera()
        camera.set_entity_provider(world)
        camera.set_target(player_entity_id)
        camera.set_mode(ReplayCameraMode.ORBIT)

        # Each frame
        camera.update(dt)
        view_matrix = camera.get_view_matrix()
    """

    def __init__(
        self,
        settings: CameraSettings | None = None,
        entity_provider: EntityProvider | None = None,
    ) -> None:
        """Initialize the replay camera.

        Args:
            settings: Camera behavior settings
            entity_provider: Provider for entity transforms
        """
        self._settings = settings or CameraSettings()
        self._entity_provider = entity_provider

        # Current state
        self._mode = ReplayCameraMode.FREE
        self._target_entity_id: int | None = None
        self._position = Vec3(0.0, 5.0, -10.0)
        self._target_position = Vec3(0.0, 0.0, 0.0)
        self._up = Vec3(0.0, 1.0, 0.0)

        # Orbit state
        self._orbit_yaw = 0.0  # Horizontal angle (radians)
        self._orbit_pitch = math.pi / 6  # Vertical angle (radians)

        # Free camera state
        self._yaw = 0.0
        self._pitch = 0.0

        # Input state for free camera
        self._move_input = Vec3()
        self._look_input = Vec3()  # x=yaw, y=pitch

    @property
    def mode(self) -> ReplayCameraMode:
        """Get the current camera mode."""
        return self._mode

    @property
    def position(self) -> Vec3:
        """Get the current camera position."""
        return self._position.copy()

    @property
    def target_position(self) -> Vec3:
        """Get the current look-at target position."""
        return self._target_position.copy()

    @property
    def target_entity_id(self) -> int | None:
        """Get the current target entity ID."""
        return self._target_entity_id

    def set_mode(self, mode: ReplayCameraMode) -> None:
        """Set the camera mode.

        Args:
            mode: The camera mode to use
        """
        self._mode = mode

    def set_target(self, entity_id: int | None) -> None:
        """Set the target entity for follow/orbit/POV modes.

        Args:
            entity_id: Entity to target, or None to clear
        """
        self._target_entity_id = entity_id

    def set_entity_provider(self, provider: EntityProvider | None) -> None:
        """Set the entity transform provider.

        Args:
            provider: Object implementing EntityProvider protocol
        """
        self._entity_provider = provider

    def set_settings(self, settings: CameraSettings) -> None:
        """Set camera behavior settings.

        Args:
            settings: New settings to use
        """
        self._settings = settings

    @property
    def settings(self) -> CameraSettings:
        """Get the current camera settings."""
        return self._settings

    def set_position(self, position: Vec3) -> None:
        """Set the camera position directly.

        Primarily useful for FREE mode.

        Args:
            position: New camera position
        """
        self._position = position.copy()

    def set_look_at(self, target: Vec3) -> None:
        """Set the point the camera looks at.

        Args:
            target: Point to look at
        """
        self._target_position = target.copy()

    def set_orbit_angles(self, yaw: float, pitch: float) -> None:
        """Set orbit camera angles.

        Args:
            yaw: Horizontal angle in radians
            pitch: Vertical angle in radians (clamped to avoid gimbal lock)
        """
        self._orbit_yaw = yaw
        # Clamp pitch to avoid looking straight up/down
        self._orbit_pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, pitch))

    def set_orbit_distance(self, distance: float) -> None:
        """Set the orbit distance from target.

        Args:
            distance: Distance in units (clamped to min/max)
        """
        self._settings.orbit_distance = max(
            self._settings.min_distance,
            min(distance, self._settings.max_distance),
        )

    def set_free_input(
        self,
        move: Vec3 | None = None,
        look: Vec3 | None = None,
    ) -> None:
        """Set input for free camera movement.

        Args:
            move: Movement input (x=right, y=up, z=forward)
            look: Look input (x=yaw, y=pitch)
        """
        if move is not None:
            self._move_input = move.copy()
        if look is not None:
            self._look_input = look.copy()

    def update(self, dt: float) -> None:
        """Update the camera based on current mode and inputs.

        Args:
            dt: Time delta in seconds
        """
        if self._mode == ReplayCameraMode.FREE:
            self._update_free(dt)
        elif self._mode == ReplayCameraMode.FOLLOW:
            self._update_follow(dt)
        elif self._mode == ReplayCameraMode.ORBIT:
            self._update_orbit(dt)
        elif self._mode == ReplayCameraMode.POV:
            self._update_pov(dt)

    def _get_target_transform(self) -> tuple[Vec3 | None, Vec3 | None]:
        """Get target entity's position and forward vector.

        Returns:
            Tuple of (position, forward), either can be None
        """
        if self._entity_provider is None or self._target_entity_id is None:
            return None, None

        pos = self._entity_provider.get_entity_position(self._target_entity_id)
        fwd = self._entity_provider.get_entity_forward(self._target_entity_id)
        return pos, fwd

    def _update_free(self, dt: float) -> None:
        """Update free camera mode."""
        # Update look direction
        self._yaw += self._look_input.x * self._settings.free_look_speed * dt
        self._pitch += self._look_input.y * self._settings.free_look_speed * dt
        self._pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, self._pitch))

        # Calculate forward and right vectors
        forward = Vec3(
            math.cos(self._pitch) * math.sin(self._yaw),
            math.sin(self._pitch),
            math.cos(self._pitch) * math.cos(self._yaw),
        )
        right = Vec3(
            math.cos(self._yaw),
            0.0,
            -math.sin(self._yaw),
        )

        # Calculate movement
        move = Vec3()
        move = move + forward * self._move_input.z
        move = move + right * self._move_input.x
        move = move + self._up * self._move_input.y

        # Apply movement
        speed = self._settings.free_move_speed * dt
        self._position = self._position + move * speed

        # Update target position (look ahead)
        self._target_position = self._position + forward

    def _update_follow(self, dt: float) -> None:
        """Update follow camera mode."""
        target_pos, target_fwd = self._get_target_transform()

        if target_pos is None:
            return

        # Calculate desired camera position
        offset = self._settings.follow_offset
        if target_fwd is not None:
            # Rotate offset to align with target's forward
            # Simplified: just use world-aligned offset for now
            pass

        desired_pos = target_pos + offset

        # Smooth camera movement
        t = min(1.0, self._settings.follow_smooth * 60.0 * dt)
        self._position = self._position.lerp(desired_pos, t)
        self._target_position = target_pos.copy()

    def _update_orbit(self, dt: float) -> None:
        """Update orbit camera mode."""
        target_pos, _ = self._get_target_transform()

        if target_pos is None:
            return

        self._target_position = target_pos.copy()

        # Calculate camera position on sphere around target
        dist = self._settings.orbit_distance
        x = math.cos(self._orbit_pitch) * math.sin(self._orbit_yaw) * dist
        y = math.sin(self._orbit_pitch) * dist
        z = math.cos(self._orbit_pitch) * math.cos(self._orbit_yaw) * dist

        self._position = target_pos + Vec3(x, y, z)

        # Auto-rotate if enabled (optional)
        # self._orbit_yaw += self._settings.orbit_speed * dt

    def _update_pov(self, dt: float) -> None:
        """Update POV camera mode."""
        target_pos, target_fwd = self._get_target_transform()

        if target_pos is None:
            return

        # Position at entity's eye level
        self._position = target_pos + self._settings.pov_offset

        # Look in entity's forward direction
        if target_fwd is not None:
            self._target_position = self._position + target_fwd
        else:
            self._target_position = self._position + Vec3(0.0, 0.0, 1.0)

    def get_view_matrix(self) -> Mat4:
        """Get the view matrix for the current camera state.

        Returns:
            4x4 view matrix
        """
        return Mat4.look_at(self._position, self._target_position, self._up)

    def cycle_mode(self) -> ReplayCameraMode:
        """Cycle to the next camera mode.

        Returns:
            The new camera mode
        """
        modes = list(ReplayCameraMode)
        current_index = modes.index(self._mode)
        next_index = (current_index + 1) % len(modes)
        self._mode = modes[next_index]
        return self._mode

    def reset(self) -> None:
        """Reset camera to default state."""
        self._position = Vec3(0.0, 5.0, -10.0)
        self._target_position = Vec3(0.0, 0.0, 0.0)
        self._orbit_yaw = 0.0
        self._orbit_pitch = math.pi / 6
        self._yaw = 0.0
        self._pitch = 0.0
        self._move_input = Vec3()
        self._look_input = Vec3()

    def zoom(self, delta: float) -> None:
        """Zoom the camera in ORBIT or FOLLOW mode.

        Args:
            delta: Zoom delta (positive = zoom out, negative = zoom in)
        """
        if self._mode == ReplayCameraMode.ORBIT:
            self._settings.orbit_distance += delta
            self._settings.orbit_distance = max(
                self._settings.min_distance,
                min(self._settings.orbit_distance, self._settings.max_distance),
            )
        elif self._mode == ReplayCameraMode.FOLLOW:
            # Zoom by adjusting follow offset distance
            offset = self._settings.follow_offset
            length = offset.length()
            new_length = max(
                self._settings.min_distance,
                min(length + delta, self._settings.max_distance),
            )
            if length > 0:
                scale = new_length / length
                self._settings.follow_offset = offset * scale
