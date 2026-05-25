"""
Debug Camera - Free cam, orbit, follow, and cycle modes for debugging.

Provides a DebugCamera for viewing the scene from any angle during development.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.core.math.vec import Vec3
    from engine.core.math.mat import Mat4
    from engine.core.math.quat import Quat

logger = logging.getLogger(__name__)


class DebugCameraMode(Enum):
    """Debug camera operating modes."""
    FREE = auto()      # Free-flying camera
    ORBIT = auto()     # Orbit around target
    FOLLOW = auto()    # Follow an entity
    CYCLE = auto()     # Cycle between entities


@dataclass
class CameraConfig:
    """
    Configuration for debug camera.

    All numeric values are configurable to avoid magic numbers.
    Override via config files or runtime configuration.
    """
    # Movement speeds (units per second)
    move_speed: float = 10.0            # Base movement speed
    fast_speed_multiplier: float = 3.0  # Speed multiplier when holding fast key
    slow_speed_multiplier: float = 0.2  # Speed multiplier when holding slow key

    # Rotation speeds (degrees per pixel of mouse movement)
    rotate_speed: float = 0.3           # Mouse look sensitivity
    orbit_speed: float = 1.0            # Orbit rotation sensitivity

    # Orbit mode distances (units)
    min_orbit_distance: float = 1.0     # Minimum distance from orbit center
    max_orbit_distance: float = 100.0   # Maximum distance from orbit center
    default_orbit_distance: float = 10.0  # Starting orbit distance

    # Follow mode offsets (units)
    follow_distance: float = 5.0        # Distance behind followed entity
    follow_height: float = 2.0          # Height above followed entity
    follow_smoothing: float = 5.0       # Follow interpolation factor

    # Pitch limits (degrees, prevents gimbal lock)
    min_pitch: float = -89.0            # Minimum pitch (looking down)
    max_pitch: float = 89.0             # Maximum pitch (looking up)

    # Interpolation smoothing factors (higher = faster)
    position_smoothing: float = 10.0    # Position interpolation factor
    rotation_smoothing: float = 15.0    # Rotation interpolation factor

    # Thresholds
    look_at_min_distance: float = 0.001  # Minimum distance for look_at calculation

    # Build restrictions
    allow_in_shipping: bool = False     # Disable debug camera in shipping builds


@dataclass
class CameraTransform:
    """Camera transform data."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll

    @property
    def pitch(self) -> float:
        return self.rotation[0]

    @property
    def yaw(self) -> float:
        return self.rotation[1]

    @property
    def roll(self) -> float:
        return self.rotation[2]


class DebugCamera:
    """
    Debug camera for viewing scenes from any angle.

    Supports multiple modes:
    - FREE: WASD movement, mouse look
    - ORBIT: Orbit around a target point
    - FOLLOW: Follow an entity with offset
    - CYCLE: Cycle through available entities
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self._config = config or CameraConfig()

        # Check build restrictions
        self._build_allowed = self._check_build_allowed()

        # Transform
        self._position = [0.0, 0.0, 0.0]
        self._target_position = [0.0, 0.0, 0.0]
        self._pitch = 0.0
        self._yaw = 0.0
        self._roll = 0.0
        self._target_pitch = 0.0
        self._target_yaw = 0.0

        # Mode
        self._mode = DebugCameraMode.FREE
        self._previous_mode = DebugCameraMode.FREE

        # Target tracking
        self._target_entity: Optional[Any] = None
        self._orbit_distance = self._config.default_orbit_distance
        self._orbit_center = [0.0, 0.0, 0.0]

        # Entity cycling
        self._entity_list: List[Any] = []
        self._current_entity_index = 0

        # State
        self._enabled = False
        self._fast_mode = False
        self._slow_mode = False

        # Callbacks
        self._mode_callbacks: List[Callable[[DebugCameraMode], None]] = []

    def _check_build_allowed(self) -> bool:
        """Check if debug camera is allowed in this build."""
        import os

        if os.environ.get("GAME_BUILD_TYPE", "").upper() == "SHIPPING":
            if not self._config.allow_in_shipping:
                logger.info("DebugCamera disabled - shipping build")
                return False
        if os.environ.get("SHIPPING") == "1":
            if not self._config.allow_in_shipping:
                return False

        return True

    @property
    def build_allowed(self) -> bool:
        """Check if debug camera is allowed in this build."""
        return self._build_allowed

    @property
    def mode(self) -> DebugCameraMode:
        """Get the current camera mode."""
        return self._mode

    @property
    def enabled(self) -> bool:
        """Check if debug camera is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the debug camera."""
        if value and not self._build_allowed:
            logger.warning("Cannot enable debug camera - not allowed in this build")
            return
        self._enabled = value
        logger.info("Debug camera %s", "enabled" if value else "disabled")

    @property
    def target(self) -> Optional[Any]:
        """Get the current target entity."""
        return self._target_entity

    def set_mode(self, mode: DebugCameraMode) -> None:
        """Set the camera mode."""
        if mode == self._mode:
            return

        self._previous_mode = self._mode
        self._mode = mode

        # Mode-specific initialization
        if mode == DebugCameraMode.ORBIT:
            self._init_orbit_mode()
        elif mode == DebugCameraMode.FOLLOW:
            self._init_follow_mode()
        elif mode == DebugCameraMode.CYCLE:
            self._init_cycle_mode()

        logger.info("Camera mode: %s", mode.name)
        self._notify_mode_callbacks(mode)

    def set_target(self, entity: Optional[Any]) -> None:
        """Set the target entity for orbit/follow modes."""
        self._target_entity = entity

        if entity is not None:
            # Update orbit center to entity position
            position = self._get_entity_position(entity)
            if position:
                self._orbit_center = list(position)

        logger.debug("Camera target: %s", entity)

    def set_entity_list(self, entities: List[Any]) -> None:
        """Set the list of entities for cycle mode."""
        self._entity_list = list(entities)
        self._current_entity_index = 0

        if self._mode == DebugCameraMode.CYCLE and self._entity_list:
            self.set_target(self._entity_list[0])

    def cycle_next(self) -> Optional[Any]:
        """Cycle to the next entity in the list."""
        if not self._entity_list:
            return None

        self._current_entity_index = (
            (self._current_entity_index + 1) % len(self._entity_list)
        )
        entity = self._entity_list[self._current_entity_index]
        self.set_target(entity)
        return entity

    def cycle_previous(self) -> Optional[Any]:
        """Cycle to the previous entity in the list."""
        if not self._entity_list:
            return None

        self._current_entity_index = (
            (self._current_entity_index - 1) % len(self._entity_list)
        )
        entity = self._entity_list[self._current_entity_index]
        self.set_target(entity)
        return entity

    def move(self, direction: Tuple[float, float, float], speed: Optional[float] = None) -> None:
        """
        Move the camera in a direction.

        Args:
            direction: Normalized movement direction (forward, right, up)
            speed: Override movement speed
        """
        if not self._enabled or self._mode not in (DebugCameraMode.FREE,):
            return

        if speed is None:
            speed = self._config.move_speed
            if self._fast_mode:
                speed *= self._config.fast_speed_multiplier
            elif self._slow_mode:
                speed *= self._config.slow_speed_multiplier

        # Calculate forward and right vectors from yaw
        yaw_rad = math.radians(self._yaw)
        pitch_rad = math.radians(self._pitch)

        # Forward vector
        forward = [
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad),
        ]

        # Right vector (perpendicular to forward on XZ plane)
        right = [
            math.cos(yaw_rad),
            0.0,
            -math.sin(yaw_rad),
        ]

        # Up vector
        up = [0.0, 1.0, 0.0]

        # Apply movement
        for i in range(3):
            self._target_position[i] += (
                forward[i] * direction[0] * speed +
                right[i] * direction[1] * speed +
                up[i] * direction[2] * speed
            )

    def rotate(self, yaw: float, pitch: float) -> None:
        """
        Rotate the camera.

        Args:
            yaw: Horizontal rotation (degrees)
            pitch: Vertical rotation (degrees)
        """
        if not self._enabled:
            return

        self._target_yaw += yaw * self._config.rotate_speed
        self._target_pitch += pitch * self._config.rotate_speed

        # Clamp pitch
        self._target_pitch = max(
            self._config.min_pitch,
            min(self._target_pitch, self._config.max_pitch)
        )

        # Normalize yaw
        self._target_yaw = self._target_yaw % 360.0

    def zoom(self, delta: float) -> None:
        """
        Zoom in/out (orbit mode only).

        Args:
            delta: Zoom amount (positive = closer)
        """
        if not self._enabled or self._mode != DebugCameraMode.ORBIT:
            return

        self._orbit_distance = max(
            self._config.min_orbit_distance,
            min(
                self._orbit_distance - delta,
                self._config.max_orbit_distance
            )
        )

    def update(self, dt: float) -> None:
        """
        Update the camera state.

        Args:
            dt: Delta time in seconds
        """
        if not self._enabled:
            return

        if self._mode == DebugCameraMode.FREE:
            self._update_free(dt)
        elif self._mode == DebugCameraMode.ORBIT:
            self._update_orbit(dt)
        elif self._mode == DebugCameraMode.FOLLOW:
            self._update_follow(dt)
        elif self._mode == DebugCameraMode.CYCLE:
            self._update_follow(dt)  # Same as follow

    def _update_free(self, dt: float) -> None:
        """Update free camera mode."""
        # Smooth position
        smoothing = self._config.position_smoothing * dt
        for i in range(3):
            self._position[i] = self._lerp(
                self._position[i],
                self._target_position[i],
                smoothing
            )

        # Smooth rotation
        rot_smoothing = self._config.rotation_smoothing * dt
        self._pitch = self._lerp(self._pitch, self._target_pitch, rot_smoothing)
        self._yaw = self._lerp_angle(self._yaw, self._target_yaw, rot_smoothing)

    def _update_orbit(self, dt: float) -> None:
        """Update orbit camera mode."""
        # Update center position if tracking entity
        if self._target_entity is not None:
            position = self._get_entity_position(self._target_entity)
            if position:
                smoothing = self._config.position_smoothing * dt
                for i in range(3):
                    self._orbit_center[i] = self._lerp(
                        self._orbit_center[i],
                        position[i],
                        smoothing
                    )

        # Smooth rotation
        rot_smoothing = self._config.rotation_smoothing * dt
        self._pitch = self._lerp(self._pitch, self._target_pitch, rot_smoothing)
        self._yaw = self._lerp_angle(self._yaw, self._target_yaw, rot_smoothing)

        # Calculate position on sphere around center
        yaw_rad = math.radians(self._yaw)
        pitch_rad = math.radians(self._pitch)

        self._position[0] = (
            self._orbit_center[0] +
            self._orbit_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        )
        self._position[1] = (
            self._orbit_center[1] +
            self._orbit_distance * math.sin(pitch_rad)
        )
        self._position[2] = (
            self._orbit_center[2] +
            self._orbit_distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        )

    def _update_follow(self, dt: float) -> None:
        """Update follow camera mode."""
        if self._target_entity is None:
            return

        position = self._get_entity_position(self._target_entity)
        if not position:
            return

        # Calculate target position behind and above entity
        yaw_rad = math.radians(self._yaw)

        target_x = position[0] + self._config.follow_distance * math.sin(yaw_rad)
        target_y = position[1] + self._config.follow_height
        target_z = position[2] + self._config.follow_distance * math.cos(yaw_rad)

        # Smooth position
        smoothing = self._config.follow_smoothing * dt
        self._position[0] = self._lerp(self._position[0], target_x, smoothing)
        self._position[1] = self._lerp(self._position[1], target_y, smoothing)
        self._position[2] = self._lerp(self._position[2], target_z, smoothing)

        # Look at target
        dx = position[0] - self._position[0]
        dy = position[1] - self._position[1]
        dz = position[2] - self._position[2]

        distance_xz = math.sqrt(dx * dx + dz * dz)
        if distance_xz > self._config.look_at_min_distance:
            target_yaw = math.degrees(math.atan2(-dx, -dz))
            target_pitch = math.degrees(math.atan2(dy, distance_xz))

            rot_smoothing = self._config.rotation_smoothing * dt
            self._yaw = self._lerp_angle(self._yaw, target_yaw, rot_smoothing)
            self._pitch = self._lerp(self._pitch, -target_pitch, rot_smoothing)

    def _init_orbit_mode(self) -> None:
        """Initialize orbit mode."""
        if self._target_entity is not None:
            position = self._get_entity_position(self._target_entity)
            if position:
                self._orbit_center = list(position)

    def _init_follow_mode(self) -> None:
        """
        Initialize follow mode.

        Currently no special initialization needed beyond target tracking,
        but provides a hook for subclasses to customize follow behavior.
        """
        logger.debug("Follow mode initialized for target: %s", self._target_entity)

    def _init_cycle_mode(self) -> None:
        """Initialize cycle mode."""
        if self._entity_list:
            self.set_target(self._entity_list[self._current_entity_index])

    def _get_entity_position(
        self,
        entity: Any,
    ) -> Optional[Tuple[float, float, float]]:
        """Get entity position. Override for actual implementation."""
        # Try common position attributes
        if hasattr(entity, "position"):
            pos = entity.position
            if hasattr(pos, "x"):
                return (pos.x, pos.y, pos.z)
            elif isinstance(pos, (tuple, list)):
                return tuple(pos[:3])
        if hasattr(entity, "transform"):
            transform = entity.transform
            if hasattr(transform, "position"):
                pos = transform.position
                if hasattr(pos, "x"):
                    return (pos.x, pos.y, pos.z)
        return None

    def get_position(self) -> Tuple[float, float, float]:
        """Get the camera position."""
        return tuple(self._position)

    def get_rotation(self) -> Tuple[float, float, float]:
        """Get the camera rotation (pitch, yaw, roll)."""
        return (self._pitch, self._yaw, self._roll)

    def get_transform(self) -> CameraTransform:
        """Get the camera transform."""
        return CameraTransform(
            position=tuple(self._position),
            rotation=(self._pitch, self._yaw, self._roll),
        )

    def get_view_matrix(self) -> List[List[float]]:
        """
        Get the view matrix.

        Returns a 4x4 matrix as a list of lists.
        """
        # Convert rotation to radians
        pitch_rad = math.radians(self._pitch)
        yaw_rad = math.radians(self._yaw)

        # Calculate view direction
        forward = [
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad),
        ]

        # Up vector
        up = [0.0, 1.0, 0.0]

        # Right vector = forward x up
        right = [
            forward[1] * up[2] - forward[2] * up[1],
            forward[2] * up[0] - forward[0] * up[2],
            forward[0] * up[1] - forward[1] * up[0],
        ]

        # Normalize right
        right_len = math.sqrt(sum(r * r for r in right))
        if right_len > 0:
            right = [r / right_len for r in right]

        # Recalculate up = right x forward
        up = [
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        ]

        # Build view matrix
        pos = self._position

        return [
            [right[0], up[0], -forward[0], 0.0],
            [right[1], up[1], -forward[1], 0.0],
            [right[2], up[2], -forward[2], 0.0],
            [
                -(right[0] * pos[0] + right[1] * pos[1] + right[2] * pos[2]),
                -(up[0] * pos[0] + up[1] * pos[1] + up[2] * pos[2]),
                (forward[0] * pos[0] + forward[1] * pos[1] + forward[2] * pos[2]),
                1.0,
            ],
        ]

    def set_fast_mode(self, enabled: bool) -> None:
        """Enable/disable fast movement mode."""
        self._fast_mode = enabled

    def set_slow_mode(self, enabled: bool) -> None:
        """Enable/disable slow movement mode."""
        self._slow_mode = enabled

    def teleport_to(
        self,
        x: float,
        y: float,
        z: float,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
    ) -> None:
        """Teleport the camera to a position."""
        self._position = [x, y, z]
        self._target_position = [x, y, z]

        if pitch is not None:
            self._pitch = pitch
            self._target_pitch = pitch
        if yaw is not None:
            self._yaw = yaw
            self._target_yaw = yaw

    def look_at(self, x: float, y: float, z: float) -> None:
        """Make the camera look at a point."""
        dx = x - self._position[0]
        dy = y - self._position[1]
        dz = z - self._position[2]

        distance_xz = math.sqrt(dx * dx + dz * dz)
        if distance_xz > self._config.look_at_min_distance:
            self._target_yaw = math.degrees(math.atan2(-dx, -dz))
            self._target_pitch = -math.degrees(math.atan2(dy, distance_xz))

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + (b - a) * min(1.0, t)

    @staticmethod
    def _lerp_angle(a: float, b: float, t: float) -> float:
        """Linear interpolation for angles (handles wrap-around)."""
        diff = ((b - a + 180) % 360) - 180
        return a + diff * min(1.0, t)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def add_mode_callback(
        self,
        callback: Callable[[DebugCameraMode], None],
    ) -> None:
        """Add a callback for mode changes."""
        self._mode_callbacks.append(callback)

    def remove_mode_callback(
        self,
        callback: Callable[[DebugCameraMode], None],
    ) -> bool:
        """Remove a mode callback."""
        try:
            self._mode_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def _notify_mode_callbacks(self, mode: DebugCameraMode) -> None:
        """Notify mode callbacks."""
        for callback in self._mode_callbacks:
            try:
                callback(mode)
            except Exception as e:
                logger.error("Mode callback error: %s", e)


# =============================================================================
# Singleton instance
# =============================================================================

_debug_camera: Optional[DebugCamera] = None


def get_debug_camera() -> DebugCamera:
    """Get the global debug camera instance."""
    global _debug_camera
    if _debug_camera is None:
        _debug_camera = DebugCamera()
    return _debug_camera


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "CameraConfig",
    "CameraTransform",
    "DebugCamera",
    "DebugCameraMode",
    "get_debug_camera",
]
