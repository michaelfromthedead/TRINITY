"""
Debug Camera - Free-fly camera, orbit camera, and debug camera switching.

Provides specialized camera modes for debugging including free movement,
orbital inspection, and smooth transitions between camera states.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, ClassVar, Optional, Any
import math
import threading
import time


class CameraMode(Enum):
    """Camera operation modes."""
    GAME = auto()      # Normal game camera
    FREE_FLY = auto()  # Free-fly debug camera
    ORBIT = auto()     # Orbital debug camera
    FIXED = auto()     # Fixed position camera
    PATH = auto()      # Camera on a path
    FOLLOW = auto()    # Follow target camera


@dataclass(slots=True)
class Vector3:
    """3D vector for camera operations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Vector3":
        length = self.length()
        if length == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / length, self.y / length, self.z / length)

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def lerp(self, other: "Vector3", t: float) -> "Vector3":
        return Vector3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t
        )

    def copy(self) -> "Vector3":
        return Vector3(self.x, self.y, self.z)


@dataclass(slots=True)
class Quaternion:
    """Quaternion for camera rotations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(0, 0, 0, 1)

    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float) -> "Quaternion":
        """Create quaternion from Euler angles (radians)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return Quaternion(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy
        )

    def to_euler(self) -> tuple[float, float, float]:
        """Convert to Euler angles (pitch, yaw, roll in radians)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (pitch, yaw, roll)

    def forward(self) -> Vector3:
        """Get forward direction."""
        return Vector3(
            2 * (self.x * self.z + self.w * self.y),
            2 * (self.y * self.z - self.w * self.x),
            1 - 2 * (self.x * self.x + self.y * self.y)
        )

    def right(self) -> Vector3:
        """Get right direction."""
        return Vector3(
            1 - 2 * (self.y * self.y + self.z * self.z),
            2 * (self.x * self.y + self.w * self.z),
            2 * (self.x * self.z - self.w * self.y)
        )

    def up(self) -> Vector3:
        """Get up direction."""
        return Vector3(
            2 * (self.x * self.y - self.w * self.z),
            1 - 2 * (self.x * self.x + self.z * self.z),
            2 * (self.y * self.z + self.w * self.x)
        )

    def slerp(self, other: "Quaternion", t: float) -> "Quaternion":
        """Spherical linear interpolation."""
        dot = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

        if dot < 0:
            other = Quaternion(-other.x, -other.y, -other.z, -other.w)
            dot = -dot

        if dot > 0.9995:
            result = Quaternion(
                self.x + t * (other.x - self.x),
                self.y + t * (other.y - self.y),
                self.z + t * (other.z - self.z),
                self.w + t * (other.w - self.w)
            )
            length = math.sqrt(result.x**2 + result.y**2 + result.z**2 + result.w**2)
            return Quaternion(
                result.x / length,
                result.y / length,
                result.z / length,
                result.w / length
            )

        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return Quaternion(
            s0 * self.x + s1 * other.x,
            s0 * self.y + s1 * other.y,
            s0 * self.z + s1 * other.z,
            s0 * self.w + s1 * other.w
        )

    def copy(self) -> "Quaternion":
        return Quaternion(self.x, self.y, self.z, self.w)


@dataclass
class CameraState:
    """Represents the state of a camera."""
    position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    fov: float = 60.0
    near_plane: float = 0.1
    far_plane: float = 1000.0
    aspect_ratio: float = 16.0 / 9.0

    def copy(self) -> "CameraState":
        return CameraState(
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            fov=self.fov,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            aspect_ratio=self.aspect_ratio,
        )

    def lerp(self, other: "CameraState", t: float) -> "CameraState":
        """Interpolate between two camera states."""
        return CameraState(
            position=self.position.lerp(other.position, t),
            rotation=self.rotation.slerp(other.rotation, t),
            fov=self.fov + (other.fov - self.fov) * t,
            near_plane=self.near_plane + (other.near_plane - self.near_plane) * t,
            far_plane=self.far_plane + (other.far_plane - self.far_plane) * t,
            aspect_ratio=other.aspect_ratio,
        )


class DebugCamera(ABC):
    """Base class for debug cameras."""

    __slots__ = ('_state', '_enabled', '_speed', '_sensitivity')

    def __init__(
        self,
        position: Optional[Vector3] = None,
        rotation: Optional[Quaternion] = None,
        speed: float = 10.0,
        sensitivity: float = 0.002,
    ):
        self._state = CameraState(
            position=position or Vector3(0, 5, -10),
            rotation=rotation or Quaternion.identity(),
        )
        self._enabled = True
        self._speed = speed
        self._sensitivity = sensitivity

    @property
    def state(self) -> CameraState:
        return self._state

    @property
    def position(self) -> Vector3:
        return self._state.position

    @position.setter
    def position(self, value: Vector3) -> None:
        self._state.position = value

    @property
    def rotation(self) -> Quaternion:
        return self._state.rotation

    @rotation.setter
    def rotation(self, value: Quaternion) -> None:
        self._state.rotation = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        self._speed = max(0.1, value)

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        self._sensitivity = max(0.0001, value)

    def get_forward(self) -> Vector3:
        """Get camera forward direction."""
        return self._state.rotation.forward()

    def get_right(self) -> Vector3:
        """Get camera right direction."""
        return self._state.rotation.right()

    def get_up(self) -> Vector3:
        """Get camera up direction."""
        return self._state.rotation.up()

    @abstractmethod
    def update(self, delta_time: float, input_state: dict[str, Any]) -> None:
        """Update camera based on input."""
        pass

    @abstractmethod
    def get_mode(self) -> CameraMode:
        """Get the camera mode."""
        pass


class FreeFlyCamera(DebugCamera):
    """Free-fly debug camera with WASD movement and mouse look."""

    __slots__ = ('_pitch', '_yaw', '_sprint_multiplier', '_slow_multiplier')

    def __init__(
        self,
        position: Optional[Vector3] = None,
        speed: float = 10.0,
        sensitivity: float = 0.002,
        sprint_multiplier: float = 3.0,
        slow_multiplier: float = 0.25,
    ):
        super().__init__(position=position, speed=speed, sensitivity=sensitivity)
        self._pitch = 0.0
        self._yaw = 0.0
        self._sprint_multiplier = sprint_multiplier
        self._slow_multiplier = slow_multiplier

    def get_mode(self) -> CameraMode:
        return CameraMode.FREE_FLY

    def look_at(self, target: Vector3) -> None:
        """Point camera at target."""
        direction = (target - self._state.position).normalized()
        self._pitch = math.asin(-direction.y)
        self._yaw = math.atan2(direction.x, direction.z)
        self._update_rotation()

    def _update_rotation(self) -> None:
        """Update rotation from pitch and yaw."""
        self._state.rotation = Quaternion.from_euler(self._pitch, self._yaw, 0)

    def update(self, delta_time: float, input_state: dict[str, Any]) -> None:
        """Update free-fly camera."""
        if not self._enabled:
            return

        # Mouse look
        mouse_delta_x = input_state.get("mouse_delta_x", 0.0)
        mouse_delta_y = input_state.get("mouse_delta_y", 0.0)

        self._yaw += mouse_delta_x * self._sensitivity
        self._pitch -= mouse_delta_y * self._sensitivity
        self._pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self._pitch))
        self._update_rotation()

        # Movement
        speed = self._speed
        if input_state.get("sprint", False):
            speed *= self._sprint_multiplier
        elif input_state.get("slow", False):
            speed *= self._slow_multiplier

        move_dir = Vector3(0, 0, 0)
        forward = self.get_forward()
        right = self.get_right()

        if input_state.get("forward", False):
            move_dir = move_dir + forward
        if input_state.get("backward", False):
            move_dir = move_dir - forward
        if input_state.get("right", False):
            move_dir = move_dir + right
        if input_state.get("left", False):
            move_dir = move_dir - right
        if input_state.get("up", False):
            move_dir = move_dir + Vector3(0, 1, 0)
        if input_state.get("down", False):
            move_dir = move_dir - Vector3(0, 1, 0)

        if move_dir.length() > 0:
            move_dir = move_dir.normalized()
            self._state.position = self._state.position + move_dir * speed * delta_time


class OrbitCamera(DebugCamera):
    """Orbital debug camera that rotates around a target."""

    __slots__ = ('_target', '_distance', '_min_distance', '_max_distance', '_pitch', '_yaw', '_zoom_speed')

    def __init__(
        self,
        target: Optional[Vector3] = None,
        distance: float = 10.0,
        min_distance: float = 1.0,
        max_distance: float = 100.0,
        speed: float = 5.0,
        sensitivity: float = 0.005,
        zoom_speed: float = 2.0,
    ):
        super().__init__(speed=speed, sensitivity=sensitivity)
        self._target = target or Vector3(0, 0, 0)
        self._distance = distance
        self._min_distance = min_distance
        self._max_distance = max_distance
        self._pitch = 0.3  # ~17 degrees
        self._yaw = 0.0
        self._zoom_speed = zoom_speed
        self._update_position()

    def get_mode(self) -> CameraMode:
        return CameraMode.ORBIT

    @property
    def target(self) -> Vector3:
        return self._target

    @target.setter
    def target(self, value: Vector3) -> None:
        self._target = value
        self._update_position()

    @property
    def distance(self) -> float:
        return self._distance

    @distance.setter
    def distance(self, value: float) -> None:
        self._distance = max(self._min_distance, min(self._max_distance, value))
        self._update_position()

    def _update_position(self) -> None:
        """Update camera position based on orbit parameters."""
        x = self._distance * math.cos(self._pitch) * math.sin(self._yaw)
        y = self._distance * math.sin(self._pitch)
        z = self._distance * math.cos(self._pitch) * math.cos(self._yaw)

        self._state.position = self._target + Vector3(x, y, z)

        # Look at target
        direction = (self._target - self._state.position).normalized()
        pitch = math.asin(-direction.y)
        yaw = math.atan2(direction.x, direction.z)
        self._state.rotation = Quaternion.from_euler(pitch, yaw, 0)

    def update(self, delta_time: float, input_state: dict[str, Any]) -> None:
        """Update orbit camera."""
        if not self._enabled:
            return

        # Rotation
        if input_state.get("rotate", False):
            mouse_delta_x = input_state.get("mouse_delta_x", 0.0)
            mouse_delta_y = input_state.get("mouse_delta_y", 0.0)

            self._yaw += mouse_delta_x * self._sensitivity
            self._pitch += mouse_delta_y * self._sensitivity
            self._pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, self._pitch))

        # Pan
        if input_state.get("pan", False):
            mouse_delta_x = input_state.get("mouse_delta_x", 0.0)
            mouse_delta_y = input_state.get("mouse_delta_y", 0.0)

            right = self.get_right()
            up = self.get_up()
            pan_speed = self._distance * 0.01

            self._target = self._target - right * mouse_delta_x * pan_speed
            self._target = self._target + up * mouse_delta_y * pan_speed

        # Zoom
        scroll_delta = input_state.get("scroll_delta", 0.0)
        if scroll_delta != 0:
            self._distance -= scroll_delta * self._zoom_speed
            self._distance = max(self._min_distance, min(self._max_distance, self._distance))

        self._update_position()

    def focus_on(self, position: Vector3, distance: Optional[float] = None) -> None:
        """Focus on a specific position."""
        self._target = position.copy()
        if distance is not None:
            self._distance = max(self._min_distance, min(self._max_distance, distance))
        self._update_position()

    def reset(self) -> None:
        """Reset to default orientation."""
        self._pitch = 0.3
        self._yaw = 0.0
        self._distance = 10.0
        self._update_position()


class DebugCameraController:
    """Controls debug camera switching and transitions."""

    _instance: ClassVar[Optional["DebugCameraController"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    __slots__ = (
        '_cameras',
        '_active_camera',
        '_game_camera_state',
        '_transitioning',
        '_transition_start_state',
        '_transition_end_state',
        '_transition_progress',
        '_transition_duration',
        '_on_camera_change',
    )

    def __init__(self):
        self._cameras: dict[CameraMode, DebugCamera] = {}
        self._active_camera: Optional[DebugCamera] = None
        self._game_camera_state: Optional[CameraState] = None
        self._transitioning = False
        self._transition_start_state: Optional[CameraState] = None
        self._transition_end_state: Optional[CameraState] = None
        self._transition_progress = 0.0
        self._transition_duration = 0.5
        self._on_camera_change: list[Callable[[CameraMode], None]] = []

    @classmethod
    def get_instance(cls) -> "DebugCameraController":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def register_camera(self, camera: DebugCamera) -> None:
        """Register a debug camera."""
        self._cameras[camera.get_mode()] = camera

    def unregister_camera(self, mode: CameraMode) -> Optional[DebugCamera]:
        """Unregister a camera."""
        return self._cameras.pop(mode, None)

    def get_camera(self, mode: CameraMode) -> Optional[DebugCamera]:
        """Get a camera by mode."""
        return self._cameras.get(mode)

    @property
    def active_camera(self) -> Optional[DebugCamera]:
        return self._active_camera

    @property
    def active_mode(self) -> Optional[CameraMode]:
        if self._active_camera:
            return self._active_camera.get_mode()
        return None

    def save_game_camera(self, state: CameraState) -> None:
        """Save the game camera state for later restoration."""
        self._game_camera_state = state.copy()

    def restore_game_camera(self) -> Optional[CameraState]:
        """Get the saved game camera state."""
        return self._game_camera_state

    def switch_camera(
        self,
        mode: CameraMode,
        transition_duration: float = 0.5,
        instant: bool = False,
    ) -> bool:
        """Switch to a different camera mode."""
        camera = self._cameras.get(mode)
        if camera is None:
            return False

        if self._active_camera is not None and not instant and transition_duration > 0:
            self._transition_start_state = self._active_camera.state.copy()
            self._transition_end_state = camera.state.copy()
            self._transition_progress = 0.0
            self._transition_duration = transition_duration
            self._transitioning = True

        self._active_camera = camera
        camera.enable()

        for callback in self._on_camera_change:
            callback(mode)

        return True

    def on_camera_change(self, callback: Callable[[CameraMode], None]) -> None:
        """Register a callback for camera changes."""
        self._on_camera_change.append(callback)

    def update(self, delta_time: float, input_state: dict[str, Any]) -> CameraState:
        """Update the active camera and return current state."""
        if self._transitioning:
            self._transition_progress += delta_time / self._transition_duration
            if self._transition_progress >= 1.0:
                self._transition_progress = 1.0
                self._transitioning = False

            if self._transition_start_state and self._transition_end_state:
                t = self._ease_in_out(self._transition_progress)
                return self._transition_start_state.lerp(self._transition_end_state, t)

        if self._active_camera:
            self._active_camera.update(delta_time, input_state)
            return self._active_camera.state

        return CameraState()

    def _ease_in_out(self, t: float) -> float:
        """Smooth ease-in-out interpolation."""
        if t < 0.5:
            return 2 * t * t
        return 1 - pow(-2 * t + 2, 2) / 2

    @property
    def is_transitioning(self) -> bool:
        return self._transitioning

    @property
    def transition_progress(self) -> float:
        return self._transition_progress

    def get_current_state(self) -> CameraState:
        """Get the current camera state (including during transitions)."""
        if self._transitioning and self._transition_start_state and self._transition_end_state:
            t = self._ease_in_out(self._transition_progress)
            return self._transition_start_state.lerp(self._transition_end_state, t)
        if self._active_camera:
            return self._active_camera.state.copy()
        return CameraState()

    def create_free_fly_camera(self, **kwargs) -> FreeFlyCamera:
        """Create and register a free-fly camera."""
        camera = FreeFlyCamera(**kwargs)
        self.register_camera(camera)
        return camera

    def create_orbit_camera(self, **kwargs) -> OrbitCamera:
        """Create and register an orbit camera."""
        camera = OrbitCamera(**kwargs)
        self.register_camera(camera)
        return camera

    @property
    def available_modes(self) -> list[CameraMode]:
        """Get list of available camera modes."""
        return list(self._cameras.keys())

    def cycle_camera(self, transition_duration: float = 0.5) -> Optional[CameraMode]:
        """Cycle to the next available camera."""
        modes = self.available_modes
        if not modes:
            return None

        current_mode = self.active_mode
        if current_mode is None:
            next_mode = modes[0]
        else:
            try:
                idx = modes.index(current_mode)
                next_mode = modes[(idx + 1) % len(modes)]
            except ValueError:
                next_mode = modes[0]

        self.switch_camera(next_mode, transition_duration)
        return next_mode
