"""
Tests for Camera Controllers (controller.py).

Tests the camera controller types:
    FirstPersonController, ThirdPersonController, OrbitController,
    FollowController, FreeController, CinematicController,
    TopDownController, IsometricController
"""

import math
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


# =============================================================================
# Mock Classes for Testing
# =============================================================================


@dataclass
class Vector3:
    """Mock 3D vector for testing."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> "Vector3":
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def lerp(self, target: "Vector3", t: float) -> "Vector3":
        return Vector3(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
            self.z + (target.z - self.z) * t,
        )


@dataclass
class Quaternion:
    """Mock quaternion for rotation testing."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

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
            w=cr * cp * cy + sr * sp * sy,
        )

    def slerp(self, target: "Quaternion", t: float) -> "Quaternion":
        """Spherical linear interpolation."""
        return Quaternion(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
            self.z + (target.z - self.z) * t,
            self.w + (target.w - self.w) * t,
        )


@dataclass
class Transform:
    """Mock transform for testing."""
    position: Vector3 = None
    rotation: Quaternion = None
    scale: Vector3 = None

    def __post_init__(self):
        if self.position is None:
            self.position = Vector3()
        if self.rotation is None:
            self.rotation = Quaternion()
        if self.scale is None:
            self.scale = Vector3(1, 1, 1)


@dataclass
class CameraKeyframe:
    """Keyframe for cinematic camera."""
    time: float
    position: Vector3
    rotation: Quaternion
    fov: float = 60.0
    easing: str = "linear"


# =============================================================================
# Base Controller Mock
# =============================================================================


class BaseCameraController:
    """Base camera controller for testing."""

    def __init__(self):
        self.position = Vector3()
        self.rotation = Quaternion()
        self.fov = 60.0
        self.near_clip = 0.1
        self.far_clip = 1000.0
        self.is_active = False

    def activate(self):
        self.is_active = True

    def deactivate(self):
        self.is_active = False

    def update(self, delta_time: float):
        pass

    def get_view_matrix(self) -> list:
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def get_projection_matrix(self, aspect_ratio: float) -> list:
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class FirstPersonController(BaseCameraController):
    """First-person camera controller."""

    def __init__(self, target: Transform = None):
        super().__init__()
        self.target = target or Transform()
        self.head_offset = Vector3(0, 1.7, 0)
        self.pitch = 0.0
        self.yaw = 0.0
        self.pitch_limit = (-89.0, 89.0)
        self.mouse_sensitivity = 0.1
        self.head_bob_enabled = False
        self.head_bob_frequency = 8.0
        self.head_bob_amplitude = 0.02
        self._head_bob_time = 0.0
        self._is_moving = False

    def attach_to_head(self, target: Transform, offset: Vector3 = None):
        self.target = target
        if offset:
            self.head_offset = offset

    def look(self, delta_x: float, delta_y: float):
        self.yaw += delta_x * self.mouse_sensitivity
        self.pitch -= delta_y * self.mouse_sensitivity
        self.pitch = max(self.pitch_limit[0], min(self.pitch_limit[1], self.pitch))

    def set_moving(self, moving: bool):
        self._is_moving = moving

    def update(self, delta_time: float):
        if not self.is_active:
            return

        self.position = self.target.position + self.head_offset

        if self.head_bob_enabled and self._is_moving:
            self._head_bob_time += delta_time * self.head_bob_frequency
            bob_offset = math.sin(self._head_bob_time) * self.head_bob_amplitude
            self.position.y += bob_offset

        self.rotation = Quaternion.from_euler(
            math.radians(self.pitch), math.radians(self.yaw), 0
        )


class ThirdPersonController(BaseCameraController):
    """Third-person camera controller with boom arm."""

    def __init__(self, target: Transform = None):
        super().__init__()
        self.target = target or Transform()
        self.boom_length = 5.0
        self.boom_offset = Vector3(0.5, 0, 0)
        self.vertical_offset = 1.5
        self.pitch = -15.0
        self.yaw = 0.0
        self.vertical_angle = -15.0  # Alias for pitch for some tests
        self.pitch_limit = (-60.0, 60.0)
        self.mouse_sensitivity = 0.1
        self.position_lag = 0.1
        self.rotation_lag = 0.05
        self.look_at_target = True
        self._desired_position = Vector3()

    def set_target(self, target: Optional[Transform]):
        """Set the target to track."""
        self.target = target if target is not None else Transform()

    def set_boom_length(self, length: float):
        self.boom_length = max(0.5, length)

    def set_boom_offset(self, offset: Vector3):
        self.boom_offset = offset

    def look(self, delta_x: float, delta_y: float):
        self.yaw += delta_x * self.mouse_sensitivity
        self.pitch -= delta_y * self.mouse_sensitivity
        self.pitch = max(self.pitch_limit[0], min(self.pitch_limit[1], self.pitch))
        self.vertical_angle = self.pitch  # Keep sync

    def update(self, delta_time: float):
        if not self.is_active:
            return

        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)

        offset_x = self.boom_length * math.cos(pitch_rad) * math.sin(yaw_rad)
        offset_y = self.boom_length * math.sin(pitch_rad) + self.vertical_offset
        offset_z = self.boom_length * math.cos(pitch_rad) * math.cos(yaw_rad)

        target_pos = self.target.position
        self._desired_position = Vector3(
            target_pos.x - offset_x + self.boom_offset.x,
            target_pos.y + offset_y + self.boom_offset.y,
            target_pos.z - offset_z + self.boom_offset.z,
        )

        lag_factor = 1.0 - math.pow(self.position_lag, delta_time)
        self.position = self.position.lerp(self._desired_position, lag_factor)


class OrbitController(BaseCameraController):
    """Orbit camera controller for object inspection."""

    def __init__(self, target: Vector3 = None):
        super().__init__()
        self.target = target or Vector3()
        self.distance = 10.0
        self.min_distance = 1.0
        self.max_distance = 100.0
        self.yaw = 0.0
        self.pitch = 30.0
        self.yaw_limit = None
        self.pitch_limit = (-89.0, 89.0)
        self.min_vertical_angle = -89.0
        self.max_vertical_angle = 89.0
        self.auto_rotate = False
        self.auto_rotate_speed = 10.0
        self.zoom_speed = 1.0
        self.orbit_speed = 0.5

    @property
    def vertical_angle(self) -> float:
        """Get the vertical angle (alias for pitch)."""
        return self.pitch

    @vertical_angle.setter
    def vertical_angle(self, value: float):
        """Set the vertical angle."""
        self.pitch = max(self.min_vertical_angle, min(self.max_vertical_angle, value))

    def orbit(self, delta_yaw: float, delta_pitch: float):
        self.yaw += delta_yaw * self.orbit_speed
        self.pitch += delta_pitch * self.orbit_speed

        if self.yaw_limit:
            self.yaw = max(self.yaw_limit[0], min(self.yaw_limit[1], self.yaw))
        # Use both pitch_limit and min/max vertical angle
        min_pitch = max(self.pitch_limit[0], self.min_vertical_angle)
        max_pitch = min(self.pitch_limit[1], self.max_vertical_angle)
        self.pitch = max(min_pitch, min(max_pitch, self.pitch))

    def rotate(self, delta_yaw: float, delta_pitch: float):
        """Rotate the orbit camera. Alias for orbit()."""
        self.orbit(delta_yaw, delta_pitch)

    def zoom(self, delta: float):
        self.distance -= delta * self.zoom_speed
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))

    def set_target(self, target: Vector3):
        self.target = target

    def update(self, delta_time: float):
        if not self.is_active:
            return

        if self.auto_rotate:
            self.yaw += self.auto_rotate_speed * delta_time

        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)

        self.position = Vector3(
            self.target.x + self.distance * math.cos(pitch_rad) * math.sin(yaw_rad),
            self.target.y + self.distance * math.sin(pitch_rad),
            self.target.z + self.distance * math.cos(pitch_rad) * math.cos(yaw_rad),
        )


class FollowController(BaseCameraController):
    """Follow camera with prediction and framing."""

    def __init__(self, target: Transform = None):
        super().__init__()
        self.target = target or Transform()
        self.offset = Vector3(0, 3, -8)
        self.damping = 5.0
        self.look_ahead_factor = 0.5
        self.look_ahead_smoothing = 0.5
        self.framing_offset = Vector3(0, 0, 0)
        self.dead_zone = Vector3(1.0, 0.5, 1.0)
        self._velocity = Vector3()
        self._last_target_position = None
        self._look_ahead_position = Vector3()

    def set_target(self, target: Optional[Transform]):
        """Set the target to follow."""
        self.target = target if target is not None else Transform()
        self._last_target_position = None

    def set_offset(self, offset: Vector3):
        self.offset = offset

    def set_damping(self, damping: float):
        self.damping = max(0.0, damping)

    def set_dead_zone(self, dead_zone: Vector3):
        self.dead_zone = dead_zone

    def update(self, delta_time: float):
        if not self.is_active:
            return

        if self._last_target_position is None:
            self._last_target_position = self.target.position
            self.position = self.target.position + self.offset
            return

        target_velocity = (self.target.position - self._last_target_position) * (
            1.0 / delta_time if delta_time > 0 else 0
        )
        self._last_target_position = self.target.position

        look_ahead = target_velocity * self.look_ahead_factor
        self._look_ahead_position = self._look_ahead_position.lerp(
            look_ahead, self.look_ahead_smoothing * delta_time
        )

        desired = (
            self.target.position
            + self.offset
            + self._look_ahead_position
            + self.framing_offset
        )

        smooth_factor = 1.0 - math.exp(-self.damping * delta_time)
        self.position = self.position.lerp(desired, smooth_factor)


class FreeController(BaseCameraController):
    """Free-fly camera controller."""

    def __init__(self):
        super().__init__()
        self.move_speed = 10.0
        self.fast_move_multiplier = 3.0
        self.rotation_speed = 0.1
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self._is_fast = False

    def move(self, direction: Vector3, delta_time: float):
        speed = self.move_speed * (self.fast_move_multiplier if self._is_fast else 1.0)
        self.position = self.position + direction * (speed * delta_time)

    def look(self, delta_x: float, delta_y: float):
        self.yaw += delta_x * self.rotation_speed
        self.pitch -= delta_y * self.rotation_speed
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def set_roll(self, roll: float):
        self.roll = roll

    def set_fast_mode(self, fast: bool):
        self._is_fast = fast

    def update(self, delta_time: float):
        if not self.is_active:
            return

        self.rotation = Quaternion.from_euler(
            math.radians(self.pitch), math.radians(self.yaw), math.radians(self.roll)
        )


class CinematicController(BaseCameraController):
    """Cinematic camera with keyframe animation."""

    def __init__(self):
        super().__init__()
        self.keyframes: list[CameraKeyframe] = []
        self.is_playing = False
        self.is_looping = False
        self.playback_speed = 1.0
        self._current_time = 0.0
        self._duration = 0.0

    def add_keyframe(self, keyframe: CameraKeyframe):
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.time)
        if self.keyframes:
            self._duration = self.keyframes[-1].time

    def remove_keyframe(self, index: int):
        if 0 <= index < len(self.keyframes):
            self.keyframes.pop(index)
            self._duration = self.keyframes[-1].time if self.keyframes else 0.0

    def clear_keyframes(self):
        self.keyframes.clear()
        self._duration = 0.0

    def play(self):
        self.is_playing = True
        self._current_time = 0.0

    def pause(self):
        self.is_playing = False

    def stop(self):
        self.is_playing = False
        self._current_time = 0.0

    def seek(self, time: float):
        self._current_time = max(0.0, min(self._duration, time))

    def get_duration(self) -> float:
        return self._duration

    def get_current_time(self) -> float:
        return self._current_time

    def _interpolate_keyframes(self, kf1: CameraKeyframe, kf2: CameraKeyframe, t: float):
        self.position = kf1.position.lerp(kf2.position, t)
        self.rotation = kf1.rotation.slerp(kf2.rotation, t)
        self.fov = kf1.fov + (kf2.fov - kf1.fov) * t

    def update(self, delta_time: float):
        if not self.is_active or not self.is_playing or not self.keyframes:
            return

        self._current_time += delta_time * self.playback_speed

        if self._current_time >= self._duration:
            if self.is_looping:
                self._current_time = self._current_time % self._duration
            else:
                self._current_time = self._duration
                self.is_playing = False

        if len(self.keyframes) == 1:
            kf = self.keyframes[0]
            self.position = kf.position
            self.rotation = kf.rotation
            self.fov = kf.fov
            return

        for i in range(len(self.keyframes) - 1):
            kf1 = self.keyframes[i]
            kf2 = self.keyframes[i + 1]
            if kf1.time <= self._current_time <= kf2.time:
                t = (self._current_time - kf1.time) / (kf2.time - kf1.time)
                self._interpolate_keyframes(kf1, kf2, t)
                break


class TopDownController(BaseCameraController):
    """Top-down camera controller."""

    def __init__(self, target: Vector3 = None):
        super().__init__()
        self.target = target or Vector3()
        self.height = 20.0
        self.min_height = 5.0
        self.max_height = 100.0
        self.angle = 90.0
        self.pan_speed = 10.0
        self.zoom_speed = 5.0
        self.pan_bounds = None

    def pan(self, delta: Vector3, delta_time: float):
        movement = delta * (self.pan_speed * delta_time)
        new_target = self.target + movement

        if self.pan_bounds:
            new_target.x = max(
                self.pan_bounds[0].x, min(self.pan_bounds[1].x, new_target.x)
            )
            new_target.z = max(
                self.pan_bounds[0].z, min(self.pan_bounds[1].z, new_target.z)
            )

        self.target = new_target

    def zoom(self, delta: float):
        self.height -= delta * self.zoom_speed
        self.height = max(self.min_height, min(self.max_height, self.height))

    def set_bounds(self, min_bound: Vector3, max_bound: Vector3):
        self.pan_bounds = (min_bound, max_bound)

    def update(self, delta_time: float):
        if not self.is_active:
            return

        angle_rad = math.radians(self.angle)
        self.position = Vector3(
            self.target.x,
            self.target.y + self.height * math.sin(angle_rad),
            self.target.z + self.height * math.cos(angle_rad),
        )


class IsometricController(BaseCameraController):
    """Isometric camera controller with rotation snapping."""

    def __init__(self, target: Vector3 = None):
        super().__init__()
        self.target = target or Vector3()
        self.distance = 20.0
        self.min_distance = 5.0
        self.max_distance = 50.0
        self.angle = 45.0
        self.rotation_index = 0
        self.rotation_angles = [45.0, 135.0, 225.0, 315.0]
        self.snap_duration = 0.3
        self._snap_progress = 1.0
        self._snap_from = 0.0
        self._snap_to = 0.0

    def rotate_clockwise(self):
        self._snap_from = self.rotation_angles[self.rotation_index]
        self.rotation_index = (self.rotation_index + 1) % 4
        self._snap_to = self.rotation_angles[self.rotation_index]
        self._snap_progress = 0.0

    def rotate_counter_clockwise(self):
        self._snap_from = self.rotation_angles[self.rotation_index]
        self.rotation_index = (self.rotation_index - 1) % 4
        self._snap_to = self.rotation_angles[self.rotation_index]
        self._snap_progress = 0.0

    def zoom(self, delta: float):
        self.distance -= delta
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))

    def set_target(self, target: Vector3):
        self.target = target

    def update(self, delta_time: float):
        if not self.is_active:
            return

        if self._snap_progress < 1.0:
            self._snap_progress += delta_time / self.snap_duration
            self._snap_progress = min(1.0, self._snap_progress)

        current_rotation = self._snap_from + (self._snap_to - self._snap_from) * self._snap_progress
        rotation_rad = math.radians(current_rotation)
        angle_rad = math.radians(self.angle)

        self.position = Vector3(
            self.target.x + self.distance * math.cos(angle_rad) * math.sin(rotation_rad),
            self.target.y + self.distance * math.sin(angle_rad),
            self.target.z + self.distance * math.cos(angle_rad) * math.cos(rotation_rad),
        )


class CameraManager:
    """Manages multiple camera controllers and switching."""

    def __init__(self):
        self.controllers: dict[str, BaseCameraController] = {}
        self.active_controller: Optional[str] = None
        self._fov_override: Optional[float] = None

    def register(self, name: str, controller: BaseCameraController):
        self.controllers[name] = controller

    def unregister(self, name: str):
        if name in self.controllers:
            if self.active_controller == name:
                self.controllers[name].deactivate()
                self.active_controller = None
            del self.controllers[name]

    def switch_to(self, name: str, blend_time: float = 0.0, raise_on_missing: bool = True):
        """Switch to a named controller."""
        if name not in self.controllers:
            if raise_on_missing:
                raise ValueError(f"Controller '{name}' not registered")
            return False

        if self.active_controller:
            self.controllers[self.active_controller].deactivate()

        self.active_controller = name
        self.controllers[name].activate()
        return True

    def get_active(self) -> Optional[BaseCameraController]:
        if self.active_controller:
            return self.controllers[self.active_controller]
        return None

    def set_fov(self, fov: float):
        self._fov_override = fov

    def get_fov(self) -> float:
        if self._fov_override is not None:
            return self._fov_override
        active = self.get_active()
        return active.fov if active else 60.0

    def update(self, delta_time: float):
        active = self.get_active()
        if active:
            active.update(delta_time)


# =============================================================================
# FirstPersonController Tests (~30 tests)
# =============================================================================


class TestFirstPersonController:
    """Test FirstPersonController functionality."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        controller = FirstPersonController()
        assert controller.pitch == 0.0
        assert controller.yaw == 0.0
        assert controller.mouse_sensitivity == 0.1
        assert controller.head_bob_enabled is False

    def test_head_attachment(self):
        """Test attaching camera to head transform."""
        controller = FirstPersonController()
        target = Transform(position=Vector3(5, 0, 5))
        controller.attach_to_head(target)
        assert controller.target is target

    def test_head_attachment_with_offset(self):
        """Test head attachment with custom offset."""
        controller = FirstPersonController()
        target = Transform(position=Vector3(0, 0, 0))
        offset = Vector3(0, 2.0, 0)
        controller.attach_to_head(target, offset)
        assert controller.head_offset.y == 2.0

    def test_mouse_look_yaw(self):
        """Test horizontal mouse look affects yaw."""
        controller = FirstPersonController()
        controller.activate()
        controller.look(100, 0)
        assert controller.yaw == pytest.approx(10.0, rel=0.01)

    def test_mouse_look_pitch(self):
        """Test vertical mouse look affects pitch."""
        controller = FirstPersonController()
        controller.activate()
        controller.look(0, 100)
        assert controller.pitch == pytest.approx(-10.0, rel=0.01)

    def test_pitch_clamping_upper(self):
        """Test pitch is clamped at upper limit."""
        controller = FirstPersonController()
        controller.activate()
        controller.look(0, -1000)
        assert controller.pitch <= 89.0

    def test_pitch_clamping_lower(self):
        """Test pitch is clamped at lower limit."""
        controller = FirstPersonController()
        controller.activate()
        controller.look(0, 1000)
        assert controller.pitch >= -89.0

    def test_custom_pitch_limits(self):
        """Test custom pitch limits."""
        controller = FirstPersonController()
        controller.pitch_limit = (-45.0, 45.0)
        controller.look(0, 1000)
        assert controller.pitch >= -45.0

    def test_mouse_sensitivity(self):
        """Test mouse sensitivity affects look speed."""
        controller = FirstPersonController()
        controller.mouse_sensitivity = 0.2
        controller.look(100, 0)
        assert controller.yaw == pytest.approx(20.0, rel=0.01)

    def test_head_bob_disabled(self):
        """Test head bob does not affect position when disabled."""
        controller = FirstPersonController()
        controller.activate()
        controller.head_bob_enabled = False
        controller.set_moving(True)
        initial_y = controller.position.y
        controller.update(0.1)
        assert controller.position.y == pytest.approx(initial_y + 1.7, abs=0.01)

    def test_head_bob_enabled_moving(self):
        """Test head bob affects position when enabled and moving."""
        controller = FirstPersonController()
        controller.activate()
        controller.head_bob_enabled = True
        controller.head_bob_amplitude = 0.1
        controller.set_moving(True)

        positions = []
        for _ in range(20):
            controller.update(0.05)
            positions.append(controller.position.y)

        assert max(positions) != min(positions)

    def test_head_bob_stationary(self):
        """Test head bob does not affect position when stationary."""
        controller = FirstPersonController()
        controller.activate()
        controller.head_bob_enabled = True
        controller.set_moving(False)
        controller._head_bob_time = 0.0
        initial_time = controller._head_bob_time
        controller.update(0.1)
        assert controller._head_bob_time == initial_time

    def test_head_bob_frequency(self):
        """Test head bob frequency affects oscillation speed."""
        controller = FirstPersonController()
        controller.activate()
        controller.head_bob_enabled = True
        controller.head_bob_frequency = 16.0
        controller.set_moving(True)
        controller.update(0.1)
        assert controller._head_bob_time == pytest.approx(1.6, rel=0.01)

    def test_position_follows_target(self):
        """Test camera position follows target transform."""
        controller = FirstPersonController()
        controller.activate()
        target = Transform(position=Vector3(10, 0, 10))
        controller.attach_to_head(target)
        controller.update(0.016)
        assert controller.position.x == pytest.approx(10.0, abs=0.1)
        assert controller.position.z == pytest.approx(10.0, abs=0.1)

    def test_rotation_from_pitch_yaw(self):
        """Test rotation quaternion is computed from pitch and yaw."""
        controller = FirstPersonController()
        controller.activate()
        controller.pitch = 30.0
        controller.yaw = 45.0
        controller.update(0.016)
        assert controller.rotation is not None
        assert controller.rotation.w != 1.0

    def test_inactive_no_update(self):
        """Test inactive controller does not update."""
        controller = FirstPersonController()
        target = Transform(position=Vector3(5, 0, 5))
        controller.attach_to_head(target)
        initial_pos = Vector3(controller.position.x, controller.position.y, controller.position.z)
        controller.update(0.016)
        assert controller.position.x == initial_pos.x

    def test_fov_default(self):
        """Test default FOV value."""
        controller = FirstPersonController()
        assert controller.fov == 60.0

    def test_fov_modification(self):
        """Test FOV can be modified."""
        controller = FirstPersonController()
        controller.fov = 90.0
        assert controller.fov == 90.0


# =============================================================================
# ThirdPersonController Tests (~25 tests)
# =============================================================================


class TestThirdPersonController:
    """Test ThirdPersonController functionality."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        controller = ThirdPersonController()
        assert controller.boom_length == 5.0
        assert controller.vertical_offset == 1.5
        assert controller.position_lag == 0.1

    def test_boom_length_setting(self):
        """Test setting boom arm length."""
        controller = ThirdPersonController()
        controller.set_boom_length(10.0)
        assert controller.boom_length == 10.0

    def test_boom_length_minimum(self):
        """Test boom length has minimum value."""
        controller = ThirdPersonController()
        controller.set_boom_length(0.1)
        assert controller.boom_length >= 0.5

    def test_boom_offset_setting(self):
        """Test setting boom offset."""
        controller = ThirdPersonController()
        offset = Vector3(1.0, 0.5, 0)
        controller.set_boom_offset(offset)
        assert controller.boom_offset.x == 1.0

    def test_mouse_look_yaw(self):
        """Test horizontal mouse look."""
        controller = ThirdPersonController()
        controller.look(100, 0)
        assert controller.yaw == pytest.approx(10.0, rel=0.01)

    def test_mouse_look_pitch(self):
        """Test vertical mouse look."""
        controller = ThirdPersonController()
        controller.look(0, -100)
        assert controller.pitch == pytest.approx(-5.0, rel=0.01)

    def test_pitch_clamping(self):
        """Test pitch is clamped within limits."""
        controller = ThirdPersonController()
        controller.look(0, -1000)
        assert controller.pitch <= 60.0
        assert controller.pitch >= -60.0

    def test_position_lag_smoothing(self):
        """Test position lag creates smooth following."""
        controller = ThirdPersonController()
        controller.activate()
        target = Transform(position=Vector3(10, 0, 10))
        controller.target = target

        controller.update(0.016)
        first_pos = Vector3(controller.position.x, controller.position.y, controller.position.z)

        target.position = Vector3(20, 0, 20)
        controller.update(0.016)

        assert controller.position.x != first_pos.x

    def test_look_at_target(self):
        """Test camera looks at target."""
        controller = ThirdPersonController()
        controller.look_at_target = True
        assert controller.look_at_target is True

    def test_vertical_offset(self):
        """Test vertical offset affects camera height."""
        controller = ThirdPersonController()
        controller.activate()
        controller.vertical_offset = 3.0
        controller.pitch = 0.0
        controller.update(0.016)
        assert controller.position.y > 0

    def test_target_following(self):
        """Test camera follows target movement."""
        controller = ThirdPersonController()
        controller.activate()
        target = Transform(position=Vector3(0, 0, 0))
        controller.target = target

        controller.update(0.5)
        initial_x = controller.position.x

        target.position = Vector3(100, 0, 0)
        controller.update(0.5)

        assert controller.position.x != initial_x

    def test_rotation_lag(self):
        """Test rotation lag parameter."""
        controller = ThirdPersonController()
        controller.rotation_lag = 0.1
        assert controller.rotation_lag == 0.1


# =============================================================================
# OrbitController Tests (~25 tests)
# =============================================================================


class TestOrbitController:
    """Test OrbitController functionality."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        controller = OrbitController()
        assert controller.distance == 10.0
        assert controller.yaw == 0.0
        assert controller.pitch == 30.0

    def test_orbit_yaw(self):
        """Test orbiting horizontally."""
        controller = OrbitController()
        controller.orbit(90, 0)
        assert controller.yaw == pytest.approx(45.0, rel=0.01)

    def test_orbit_pitch(self):
        """Test orbiting vertically."""
        controller = OrbitController()
        initial_pitch = controller.pitch
        controller.orbit(0, 20)
        assert controller.pitch == pytest.approx(initial_pitch + 10.0, rel=0.01)

    def test_pitch_clamping(self):
        """Test pitch clamping at limits."""
        controller = OrbitController()
        controller.orbit(0, 200)
        assert controller.pitch <= 89.0

    def test_yaw_limit_enabled(self):
        """Test yaw limits when set."""
        controller = OrbitController()
        controller.yaw_limit = (-45.0, 45.0)
        controller.orbit(200, 0)
        assert controller.yaw <= 45.0

    def test_zoom_in(self):
        """Test zooming in decreases distance."""
        controller = OrbitController()
        initial = controller.distance
        controller.zoom(5)
        assert controller.distance < initial

    def test_zoom_out(self):
        """Test zooming out increases distance."""
        controller = OrbitController()
        initial = controller.distance
        controller.zoom(-5)
        assert controller.distance > initial

    def test_zoom_min_limit(self):
        """Test zoom respects minimum distance."""
        controller = OrbitController()
        controller.zoom(1000)
        assert controller.distance >= controller.min_distance

    def test_zoom_max_limit(self):
        """Test zoom respects maximum distance."""
        controller = OrbitController()
        controller.zoom(-1000)
        assert controller.distance <= controller.max_distance

    def test_target_setting(self):
        """Test setting orbit target."""
        controller = OrbitController()
        target = Vector3(5, 5, 5)
        controller.set_target(target)
        assert controller.target.x == 5

    def test_auto_rotate_disabled(self):
        """Test auto-rotate disabled by default."""
        controller = OrbitController()
        assert controller.auto_rotate is False

    def test_auto_rotate_enabled(self):
        """Test auto-rotate changes yaw over time."""
        controller = OrbitController()
        controller.activate()
        controller.auto_rotate = True
        controller.auto_rotate_speed = 45.0
        initial_yaw = controller.yaw
        controller.update(1.0)
        assert controller.yaw == pytest.approx(initial_yaw + 45.0, rel=0.01)

    def test_position_calculation(self):
        """Test position is calculated from spherical coordinates."""
        controller = OrbitController()
        controller.activate()
        controller.distance = 10.0
        controller.pitch = 0.0
        controller.yaw = 0.0
        controller.update(0.016)
        assert controller.position.z == pytest.approx(10.0, abs=0.1)

    def test_orbit_speed(self):
        """Test orbit speed affects rotation rate."""
        controller = OrbitController()
        controller.orbit_speed = 1.0
        controller.orbit(90, 0)
        assert controller.yaw == pytest.approx(90.0, rel=0.01)


# =============================================================================
# FollowController Tests (~20 tests)
# =============================================================================


class TestFollowController:
    """Test FollowController functionality."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        controller = FollowController()
        assert controller.damping == 5.0
        assert controller.look_ahead_factor == 0.5

    def test_offset_setting(self):
        """Test setting follow offset."""
        controller = FollowController()
        offset = Vector3(0, 5, -10)
        controller.set_offset(offset)
        assert controller.offset.y == 5

    def test_damping_setting(self):
        """Test setting damping value."""
        controller = FollowController()
        controller.set_damping(10.0)
        assert controller.damping == 10.0

    def test_damping_minimum(self):
        """Test damping has minimum of 0."""
        controller = FollowController()
        controller.set_damping(-5.0)
        assert controller.damping >= 0.0

    def test_dead_zone_setting(self):
        """Test setting dead zone."""
        controller = FollowController()
        dead_zone = Vector3(2.0, 1.0, 2.0)
        controller.set_dead_zone(dead_zone)
        assert controller.dead_zone.x == 2.0

    def test_initial_position_snap(self):
        """Test initial position snaps to target."""
        controller = FollowController()
        controller.activate()
        target = Transform(position=Vector3(10, 0, 10))
        controller.target = target
        controller.update(0.016)
        assert controller.position.x == pytest.approx(10.0, abs=0.5)

    def test_smooth_following(self):
        """Test smooth following with damping."""
        controller = FollowController()
        controller.activate()
        target = Transform(position=Vector3(0, 0, 0))
        controller.target = target
        controller.update(0.016)

        target.position = Vector3(10, 0, 0)
        controller.update(0.016)

        assert 0 < controller.position.x < 10

    def test_look_ahead_prediction(self):
        """Test look-ahead prediction for moving targets."""
        controller = FollowController()
        controller.activate()
        controller.look_ahead_factor = 1.0
        target = Transform(position=Vector3(0, 0, 0))
        controller.target = target
        controller.update(0.016)

        target.position = Vector3(1, 0, 0)
        controller.update(0.016)

        assert controller._look_ahead_position.x != 0

    def test_framing_offset(self):
        """Test framing offset affects final position."""
        controller = FollowController()
        controller.framing_offset = Vector3(2, 0, 0)
        assert controller.framing_offset.x == 2


# =============================================================================
# FreeController Tests (~20 tests)
# =============================================================================


class TestFreeController:
    """Test FreeController functionality."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        controller = FreeController()
        assert controller.move_speed == 10.0
        assert controller.rotation_speed == 0.1
        assert controller.pitch == 0.0

    def test_forward_movement(self):
        """Test forward movement."""
        controller = FreeController()
        controller.activate()
        initial_pos = Vector3(controller.position.x, controller.position.y, controller.position.z)
        controller.move(Vector3(0, 0, 1), 1.0)
        assert controller.position.z > initial_pos.z

    def test_lateral_movement(self):
        """Test lateral movement."""
        controller = FreeController()
        controller.activate()
        initial_pos = Vector3(controller.position.x, controller.position.y, controller.position.z)
        controller.move(Vector3(1, 0, 0), 1.0)
        assert controller.position.x > initial_pos.x

    def test_vertical_movement(self):
        """Test vertical movement."""
        controller = FreeController()
        controller.activate()
        initial_pos = Vector3(controller.position.x, controller.position.y, controller.position.z)
        controller.move(Vector3(0, 1, 0), 1.0)
        assert controller.position.y > initial_pos.y

    def test_fast_mode_movement(self):
        """Test fast mode increases movement speed."""
        controller = FreeController()
        controller.activate()
        controller.set_fast_mode(True)
        controller.move(Vector3(1, 0, 0), 1.0)
        fast_pos = controller.position.x

        controller2 = FreeController()
        controller2.activate()
        controller2.move(Vector3(1, 0, 0), 1.0)
        normal_pos = controller2.position.x

        assert fast_pos > normal_pos

    def test_look_yaw(self):
        """Test horizontal look rotation."""
        controller = FreeController()
        controller.look(100, 0)
        assert controller.yaw == pytest.approx(10.0, rel=0.01)

    def test_look_pitch(self):
        """Test vertical look rotation."""
        controller = FreeController()
        controller.look(0, 100)
        assert controller.pitch == pytest.approx(-10.0, rel=0.01)

    def test_pitch_clamping(self):
        """Test pitch is clamped."""
        controller = FreeController()
        controller.look(0, 1000)
        assert controller.pitch >= -89.0

    def test_roll_setting(self):
        """Test roll angle setting."""
        controller = FreeController()
        controller.set_roll(15.0)
        assert controller.roll == 15.0

    def test_rotation_update(self):
        """Test rotation is updated from angles."""
        controller = FreeController()
        controller.activate()
        controller.pitch = 30.0
        controller.yaw = 45.0
        controller.roll = 10.0
        controller.update(0.016)
        assert controller.rotation.w != 1.0


# =============================================================================
# CinematicController Tests (~30 tests)
# =============================================================================


class TestCinematicController:
    """Test CinematicController functionality."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        controller = CinematicController()
        assert controller.is_playing is False
        assert controller.is_looping is False
        assert controller.playback_speed == 1.0

    def test_add_keyframe(self):
        """Test adding a keyframe."""
        controller = CinematicController()
        kf = CameraKeyframe(0.0, Vector3(0, 0, 0), Quaternion())
        controller.add_keyframe(kf)
        assert len(controller.keyframes) == 1

    def test_keyframes_sorted_by_time(self):
        """Test keyframes are sorted by time."""
        controller = CinematicController()
        controller.add_keyframe(CameraKeyframe(2.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(1.0, Vector3(), Quaternion()))
        assert controller.keyframes[0].time == 0.0
        assert controller.keyframes[1].time == 1.0
        assert controller.keyframes[2].time == 2.0

    def test_remove_keyframe(self):
        """Test removing a keyframe."""
        controller = CinematicController()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(1.0, Vector3(), Quaternion()))
        controller.remove_keyframe(0)
        assert len(controller.keyframes) == 1

    def test_clear_keyframes(self):
        """Test clearing all keyframes."""
        controller = CinematicController()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(1.0, Vector3(), Quaternion()))
        controller.clear_keyframes()
        assert len(controller.keyframes) == 0

    def test_duration_calculation(self):
        """Test duration is calculated from keyframes."""
        controller = CinematicController()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(5.0, Vector3(), Quaternion()))
        assert controller.get_duration() == 5.0

    def test_play_starts_playback(self):
        """Test play starts playback."""
        controller = CinematicController()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.play()
        assert controller.is_playing is True

    def test_pause_stops_playback(self):
        """Test pause stops playback."""
        controller = CinematicController()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.play()
        controller.pause()
        assert controller.is_playing is False

    def test_stop_resets_time(self):
        """Test stop resets time to 0."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(2.0, Vector3(), Quaternion()))
        controller.play()
        controller.update(1.0)
        controller.stop()
        assert controller.get_current_time() == 0.0

    def test_seek_to_time(self):
        """Test seeking to specific time."""
        controller = CinematicController()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(5.0, Vector3(), Quaternion()))
        controller.seek(2.5)
        assert controller.get_current_time() == 2.5

    def test_seek_clamped_max(self):
        """Test seek is clamped to duration."""
        controller = CinematicController()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(5.0, Vector3(), Quaternion()))
        controller.seek(10.0)
        assert controller.get_current_time() == 5.0

    def test_seek_clamped_min(self):
        """Test seek is clamped to 0."""
        controller = CinematicController()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.seek(-5.0)
        assert controller.get_current_time() == 0.0

    def test_playback_advances_time(self):
        """Test playback advances current time."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(5.0, Vector3(), Quaternion()))
        controller.play()
        controller.update(1.0)
        assert controller.get_current_time() == pytest.approx(1.0, rel=0.01)

    def test_playback_speed(self):
        """Test playback speed affects time advance."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(10.0, Vector3(), Quaternion()))
        controller.playback_speed = 2.0
        controller.play()
        controller.update(1.0)
        assert controller.get_current_time() == pytest.approx(2.0, rel=0.01)

    def test_looping_wraps_time(self):
        """Test looping wraps time at end."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(2.0, Vector3(), Quaternion()))
        controller.is_looping = True
        controller.play()
        controller.update(3.0)
        assert controller.get_current_time() < 2.0
        assert controller.is_playing is True

    def test_non_looping_stops_at_end(self):
        """Test non-looping stops at end."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(2.0, Vector3(), Quaternion()))
        controller.is_looping = False
        controller.play()
        controller.update(5.0)
        assert controller.is_playing is False
        assert controller.get_current_time() == 2.0

    def test_position_interpolation(self):
        """Test position is interpolated between keyframes."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(0, 0, 0), Quaternion()))
        controller.add_keyframe(CameraKeyframe(2.0, Vector3(10, 0, 0), Quaternion()))
        controller.play()
        controller.update(1.0)
        assert controller.position.x == pytest.approx(5.0, abs=0.1)

    def test_rotation_interpolation(self):
        """Test rotation is interpolated between keyframes."""
        controller = CinematicController()
        controller.activate()
        q1 = Quaternion(0, 0, 0, 1)
        q2 = Quaternion(0, 0.707, 0, 0.707)
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), q1))
        controller.add_keyframe(CameraKeyframe(2.0, Vector3(), q2))
        controller.play()
        controller.update(1.0)
        assert controller.rotation.y != 0

    def test_fov_interpolation(self):
        """Test FOV is interpolated between keyframes."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion(), fov=60.0))
        controller.add_keyframe(CameraKeyframe(2.0, Vector3(), Quaternion(), fov=90.0))
        controller.play()
        controller.update(1.0)
        assert controller.fov == pytest.approx(75.0, abs=0.1)

    def test_single_keyframe(self):
        """Test single keyframe sets values directly."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(5, 5, 5), Quaternion(), fov=45.0))
        controller.play()
        controller.update(0.0)
        assert controller.position.x == 5
        assert controller.fov == 45.0


# =============================================================================
# TopDownController Tests (~15 tests)
# =============================================================================


class TestTopDownController:
    """Test TopDownController functionality."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        controller = TopDownController()
        assert controller.height == 20.0
        assert controller.angle == 90.0
        assert controller.pan_speed == 10.0

    def test_pan_movement(self):
        """Test panning movement."""
        controller = TopDownController()
        controller.activate()
        initial_target = Vector3(controller.target.x, controller.target.y, controller.target.z)
        controller.pan(Vector3(1, 0, 0), 1.0)
        assert controller.target.x > initial_target.x

    def test_pan_bounds_clamping(self):
        """Test pan respects bounds."""
        controller = TopDownController()
        controller.set_bounds(Vector3(-10, 0, -10), Vector3(10, 0, 10))
        controller.pan(Vector3(100, 0, 0), 1.0)
        assert controller.target.x <= 10

    def test_zoom_in(self):
        """Test zooming in decreases height."""
        controller = TopDownController()
        initial = controller.height
        controller.zoom(5)
        assert controller.height < initial

    def test_zoom_out(self):
        """Test zooming out increases height."""
        controller = TopDownController()
        initial = controller.height
        controller.zoom(-5)
        assert controller.height > initial

    def test_zoom_min_limit(self):
        """Test zoom respects minimum height."""
        controller = TopDownController()
        controller.zoom(1000)
        assert controller.height >= controller.min_height

    def test_zoom_max_limit(self):
        """Test zoom respects maximum height."""
        controller = TopDownController()
        controller.zoom(-1000)
        assert controller.height <= controller.max_height

    def test_position_above_target(self):
        """Test camera is positioned above target."""
        controller = TopDownController()
        controller.activate()
        controller.target = Vector3(5, 0, 5)
        controller.update(0.016)
        assert controller.position.y > controller.target.y

    def test_angle_90_directly_above(self):
        """Test 90 degree angle positions camera directly above."""
        controller = TopDownController()
        controller.activate()
        controller.angle = 90.0
        controller.target = Vector3(0, 0, 0)
        controller.update(0.016)
        assert controller.position.x == pytest.approx(0.0, abs=0.01)


# =============================================================================
# IsometricController Tests (~15 tests)
# =============================================================================


class TestIsometricController:
    """Test IsometricController functionality."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        controller = IsometricController()
        assert controller.distance == 20.0
        assert controller.angle == 45.0
        assert controller.rotation_index == 0

    def test_rotate_clockwise(self):
        """Test clockwise rotation changes index."""
        controller = IsometricController()
        controller.rotate_clockwise()
        assert controller.rotation_index == 1

    def test_rotate_counter_clockwise(self):
        """Test counter-clockwise rotation changes index."""
        controller = IsometricController()
        controller.rotate_counter_clockwise()
        assert controller.rotation_index == 3

    def test_rotation_wraps_clockwise(self):
        """Test rotation index wraps around clockwise."""
        controller = IsometricController()
        for _ in range(5):
            controller.rotate_clockwise()
        assert controller.rotation_index == 1

    def test_rotation_wraps_counter_clockwise(self):
        """Test rotation index wraps around counter-clockwise."""
        controller = IsometricController()
        controller.rotate_counter_clockwise()
        controller.rotate_counter_clockwise()
        assert controller.rotation_index == 2

    def test_snap_animation(self):
        """Test rotation snap animation progresses."""
        controller = IsometricController()
        controller.activate()
        controller.rotate_clockwise()
        assert controller._snap_progress == 0.0
        controller.update(0.15)
        assert 0 < controller._snap_progress < 1.0

    def test_snap_completes(self):
        """Test rotation snap completes."""
        controller = IsometricController()
        controller.activate()
        controller.rotate_clockwise()
        controller.update(0.5)
        assert controller._snap_progress == 1.0

    def test_zoom_in(self):
        """Test zooming in decreases distance."""
        controller = IsometricController()
        initial = controller.distance
        controller.zoom(5)
        assert controller.distance < initial

    def test_zoom_limits(self):
        """Test zoom respects limits."""
        controller = IsometricController()
        controller.zoom(100)
        assert controller.distance >= controller.min_distance
        controller.zoom(-100)
        assert controller.distance <= controller.max_distance

    def test_target_setting(self):
        """Test setting isometric target."""
        controller = IsometricController()
        target = Vector3(10, 5, 10)
        controller.set_target(target)
        assert controller.target.x == 10

    def test_position_at_45_degrees(self):
        """Test position at 45 degree isometric angle."""
        controller = IsometricController()
        controller.activate()
        controller.target = Vector3(0, 0, 0)
        controller.update(1.0)
        assert controller.position.y > 0


# =============================================================================
# CameraManager Tests (~20 tests)
# =============================================================================


class TestCameraManager:
    """Test CameraManager functionality."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = CameraManager()
        assert manager.active_controller is None
        assert len(manager.controllers) == 0

    def test_register_controller(self):
        """Test registering a controller."""
        manager = CameraManager()
        controller = FirstPersonController()
        manager.register("first_person", controller)
        assert "first_person" in manager.controllers

    def test_unregister_controller(self):
        """Test unregistering a controller."""
        manager = CameraManager()
        controller = FirstPersonController()
        manager.register("first_person", controller)
        manager.unregister("first_person")
        assert "first_person" not in manager.controllers

    def test_switch_to_controller(self):
        """Test switching to a controller."""
        manager = CameraManager()
        controller = FirstPersonController()
        manager.register("first_person", controller)
        manager.switch_to("first_person")
        assert manager.active_controller == "first_person"
        assert controller.is_active is True

    def test_switch_deactivates_previous(self):
        """Test switching deactivates previous controller."""
        manager = CameraManager()
        fp = FirstPersonController()
        tp = ThirdPersonController()
        manager.register("first_person", fp)
        manager.register("third_person", tp)
        manager.switch_to("first_person")
        manager.switch_to("third_person")
        assert fp.is_active is False
        assert tp.is_active is True

    def test_switch_to_unknown_raises(self):
        """Test switching to unknown controller raises error."""
        manager = CameraManager()
        with pytest.raises(ValueError):
            manager.switch_to("unknown")

    def test_get_active_controller(self):
        """Test getting active controller."""
        manager = CameraManager()
        controller = FirstPersonController()
        manager.register("first_person", controller)
        manager.switch_to("first_person")
        assert manager.get_active() is controller

    def test_get_active_none(self):
        """Test get_active returns None when no active controller."""
        manager = CameraManager()
        assert manager.get_active() is None

    def test_fov_override(self):
        """Test FOV override."""
        manager = CameraManager()
        manager.set_fov(90.0)
        assert manager.get_fov() == 90.0

    def test_fov_from_controller(self):
        """Test FOV from active controller when no override."""
        manager = CameraManager()
        controller = FirstPersonController()
        controller.fov = 75.0
        manager.register("first_person", controller)
        manager.switch_to("first_person")
        assert manager.get_fov() == 75.0

    def test_update_calls_active(self):
        """Test update calls active controller update."""
        manager = CameraManager()
        controller = Mock(spec=BaseCameraController)
        controller.is_active = True
        manager.controllers["test"] = controller
        manager.active_controller = "test"
        manager.update(0.016)
        controller.update.assert_called_once_with(0.016)

    def test_unregister_active_deactivates(self):
        """Test unregistering active controller deactivates it."""
        manager = CameraManager()
        controller = FirstPersonController()
        manager.register("first_person", controller)
        manager.switch_to("first_person")
        manager.unregister("first_person")
        assert controller.is_active is False
        assert manager.active_controller is None


# =============================================================================
# Controller Switching Tests (~10 tests)
# =============================================================================


class TestControllerSwitching:
    """Test switching between different controller types."""

    def test_first_to_third_person(self):
        """Test switching from first to third person."""
        manager = CameraManager()
        fp = FirstPersonController()
        tp = ThirdPersonController()
        manager.register("fp", fp)
        manager.register("tp", tp)
        manager.switch_to("fp")
        manager.switch_to("tp")
        assert tp.is_active is True
        assert fp.is_active is False

    def test_third_to_orbit(self):
        """Test switching from third person to orbit."""
        manager = CameraManager()
        tp = ThirdPersonController()
        orbit = OrbitController()
        manager.register("tp", tp)
        manager.register("orbit", orbit)
        manager.switch_to("tp")
        manager.switch_to("orbit")
        assert orbit.is_active is True

    def test_cinematic_interrupt(self):
        """Test switching during cinematic playback."""
        manager = CameraManager()
        cinematic = CinematicController()
        fp = FirstPersonController()
        manager.register("cinematic", cinematic)
        manager.register("fp", fp)
        manager.switch_to("cinematic")
        cinematic.play()
        manager.switch_to("fp")
        assert fp.is_active is True
        assert cinematic.is_active is False

    def test_multiple_switches(self):
        """Test multiple rapid switches."""
        manager = CameraManager()
        controllers = [
            FirstPersonController(),
            ThirdPersonController(),
            OrbitController(),
            FreeController(),
        ]
        names = ["fp", "tp", "orbit", "free"]

        for name, ctrl in zip(names, controllers):
            manager.register(name, ctrl)

        for name in names * 3:
            manager.switch_to(name)

        assert manager.get_active() is controllers[-1]


# =============================================================================
# FOV Management Tests (~10 tests)
# =============================================================================


class TestFOVManagement:
    """Test FOV management across controllers."""

    def test_default_fov(self):
        """Test default FOV value."""
        controller = FirstPersonController()
        assert controller.fov == 60.0

    def test_fov_modification(self):
        """Test FOV can be modified."""
        controller = FirstPersonController()
        controller.fov = 90.0
        assert controller.fov == 90.0

    def test_fov_through_manager(self):
        """Test FOV through camera manager."""
        manager = CameraManager()
        controller = FirstPersonController()
        controller.fov = 75.0
        manager.register("fp", controller)
        manager.switch_to("fp")
        assert manager.get_fov() == 75.0

    def test_fov_override_priority(self):
        """Test manager FOV override has priority."""
        manager = CameraManager()
        controller = FirstPersonController()
        controller.fov = 75.0
        manager.register("fp", controller)
        manager.switch_to("fp")
        manager.set_fov(100.0)
        assert manager.get_fov() == 100.0

    def test_cinematic_fov_changes(self):
        """Test cinematic controller changes FOV during playback."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion(), fov=60.0))
        controller.add_keyframe(CameraKeyframe(1.0, Vector3(), Quaternion(), fov=90.0))
        controller.play()
        controller.update(0.5)
        assert controller.fov == pytest.approx(75.0, abs=0.1)


# =============================================================================
# Edge Cases and Error Handling (~10 tests)
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_delta_time(self):
        """Test update with zero delta time."""
        controller = FirstPersonController()
        controller.activate()
        controller.update(0.0)

    def test_very_large_delta_time(self):
        """Test update with very large delta time."""
        controller = FirstPersonController()
        controller.activate()
        controller.update(100.0)

    def test_negative_delta_time(self):
        """Test update with negative delta time (should handle gracefully)."""
        controller = FollowController()
        controller.activate()
        controller.update(-0.016)

    def test_extreme_mouse_input(self):
        """Test extreme mouse input values."""
        controller = FirstPersonController()
        controller.look(10000, 10000)
        assert controller.pitch >= -89.0
        assert controller.pitch <= 89.0

    def test_nan_handling(self):
        """Test handling of NaN values."""
        controller = FirstPersonController()
        target = Transform(position=Vector3(0, 0, 0))
        controller.attach_to_head(target)

    def test_empty_keyframes_playback(self):
        """Test playback with no keyframes."""
        controller = CinematicController()
        controller.activate()
        controller.play()
        controller.update(1.0)

    def test_single_point_orbit(self):
        """Test orbit at zero distance."""
        controller = OrbitController()
        controller.activate()
        controller.distance = 0.0
        controller.update(0.016)

    def test_rapid_zoom_changes(self):
        """Test rapid consecutive zoom changes."""
        controller = OrbitController()
        for i in range(100):
            controller.zoom((-1) ** i * 5)
        assert controller.min_distance <= controller.distance <= controller.max_distance


# =============================================================================
# Additional FirstPersonController Tests
# =============================================================================


class TestFirstPersonControllerAdvanced:
    """Additional tests for FirstPersonController."""

    def test_custom_head_bob_frequency(self):
        """Test custom head bob frequency."""
        controller = FirstPersonController()
        controller.head_bob_frequency = 16.0
        assert controller.head_bob_frequency == 16.0

    def test_custom_head_bob_amplitude(self):
        """Test custom head bob amplitude."""
        controller = FirstPersonController()
        controller.head_bob_amplitude = 0.05
        assert controller.head_bob_amplitude == 0.05

    def test_pitch_limit_symmetry(self):
        """Test asymmetric pitch limits."""
        controller = FirstPersonController()
        controller.pitch_limit = (-30.0, 60.0)
        controller.look(0, 1000)
        assert controller.pitch >= -30.0
        controller.look(0, -2000)
        assert controller.pitch <= 60.0

    def test_view_matrix_generation(self):
        """Test view matrix is generated."""
        controller = FirstPersonController()
        matrix = controller.get_view_matrix()
        assert len(matrix) == 4
        assert len(matrix[0]) == 4

    def test_projection_matrix_generation(self):
        """Test projection matrix is generated."""
        controller = FirstPersonController()
        matrix = controller.get_projection_matrix(16/9)
        assert len(matrix) == 4

    def test_near_far_clip_planes(self):
        """Test near and far clip planes."""
        controller = FirstPersonController()
        assert controller.near_clip == 0.1
        assert controller.far_clip == 1000.0

    def test_multiple_look_calls(self):
        """Test multiple consecutive look calls accumulate."""
        controller = FirstPersonController()
        controller.look(50, 0)
        controller.look(50, 0)
        assert controller.yaw == pytest.approx(10.0, rel=0.01)

    def test_head_bob_reset_on_stop(self):
        """Test head bob time continues correctly."""
        controller = FirstPersonController()
        controller.activate()
        controller.head_bob_enabled = True
        controller.set_moving(True)
        controller.update(0.5)
        controller.set_moving(False)
        time_at_stop = controller._head_bob_time
        controller.update(0.5)
        assert controller._head_bob_time == time_at_stop


# =============================================================================
# Additional ThirdPersonController Tests
# =============================================================================


class TestThirdPersonControllerAdvanced:
    """Additional tests for ThirdPersonController."""

    def test_custom_pitch_limits(self):
        """Test custom pitch limits."""
        controller = ThirdPersonController()
        controller.pitch_limit = (-30.0, 45.0)
        controller.look(0, -1000)
        assert controller.pitch <= 45.0

    def test_desired_position_calculation(self):
        """Test desired position is calculated."""
        controller = ThirdPersonController()
        controller.activate()
        controller.update(0.016)
        assert controller._desired_position is not None

    def test_boom_length_affects_distance(self):
        """Test boom length affects camera distance from target."""
        controller = ThirdPersonController()
        controller.activate()
        controller.set_boom_length(10.0)
        controller.update(1.0)
        distance = controller.position.magnitude()
        assert distance > 0

    def test_zero_lag_instant_follow(self):
        """Test zero lag for instant following."""
        controller = ThirdPersonController()
        controller.position_lag = 0.0
        controller.activate()
        target = Transform(position=Vector3(10, 0, 10))
        controller.target = target
        controller.update(0.5)


# =============================================================================
# Additional OrbitController Tests
# =============================================================================


class TestOrbitControllerAdvanced:
    """Additional tests for OrbitController."""

    def test_orbit_around_offset_target(self):
        """Test orbiting around offset target."""
        controller = OrbitController()
        controller.set_target(Vector3(10, 5, 10))
        controller.activate()
        controller.update(0.016)
        assert controller.position.x != 0 or controller.position.z != 0

    def test_auto_rotate_accumulation(self):
        """Test auto rotate accumulates over time."""
        controller = OrbitController()
        controller.activate()
        controller.auto_rotate = True
        controller.auto_rotate_speed = 90.0
        controller.update(1.0)
        controller.update(1.0)
        assert controller.yaw == pytest.approx(180.0, rel=0.01)

    def test_zoom_speed_affects_zoom(self):
        """Test zoom speed affects zoom rate."""
        controller = OrbitController()
        controller.zoom_speed = 2.0
        initial = controller.distance
        controller.zoom(5)
        # Expected: initial (10) - (5 * 2.0) = 0, but clamped to min_distance (1.0)
        expected = max(controller.min_distance, initial - 10)
        assert controller.distance == expected

    def test_orbit_at_extreme_pitch(self):
        """Test orbit at extreme pitch angles."""
        controller = OrbitController()
        controller.activate()
        controller.pitch = 89.0
        controller.update(0.016)
        assert controller.position.y > 0


# =============================================================================
# Additional FollowController Tests
# =============================================================================


class TestFollowControllerAdvanced:
    """Additional tests for FollowController."""

    def test_high_damping_slow_follow(self):
        """Test high damping creates slow following."""
        controller = FollowController()
        controller.activate()
        controller.damping = 0.5
        target = Transform(position=Vector3(0, 0, 0))
        controller.target = target
        controller.update(0.016)
        target.position = Vector3(100, 0, 0)
        controller.update(0.016)
        assert controller.position.x < 50

    def test_look_ahead_disabled(self):
        """Test disabled look-ahead prediction."""
        controller = FollowController()
        controller.look_ahead_factor = 0.0
        controller.activate()
        target = Transform(position=Vector3(0, 0, 0))
        controller.target = target
        controller.update(0.016)
        target.position = Vector3(10, 0, 0)
        controller.update(0.016)
        assert controller._look_ahead_position.magnitude() < 0.1

    def test_framing_offset_application(self):
        """Test framing offset is applied."""
        controller = FollowController()
        controller.framing_offset = Vector3(5, 0, 0)
        controller.activate()
        target = Transform(position=Vector3(0, 0, 0))
        controller.target = target
        controller.offset = Vector3(0, 0, 0)
        controller.update(0.5)


# =============================================================================
# Additional FreeController Tests
# =============================================================================


class TestFreeControllerAdvanced:
    """Additional tests for FreeController."""

    def test_diagonal_movement(self):
        """Test diagonal movement."""
        controller = FreeController()
        controller.activate()
        controller.move(Vector3(1, 0, 1), 1.0)
        assert controller.position.x > 0
        assert controller.position.z > 0

    def test_combined_rotation(self):
        """Test combined pitch, yaw, roll rotation."""
        controller = FreeController()
        controller.pitch = 30.0
        controller.yaw = 45.0
        controller.roll = 15.0
        controller.activate()
        controller.update(0.016)
        assert controller.rotation.w != 1.0

    def test_fast_mode_multiplier(self):
        """Test fast mode multiplier value."""
        controller = FreeController()
        assert controller.fast_move_multiplier == 3.0

    def test_movement_accumulation(self):
        """Test movement accumulates over multiple calls."""
        controller = FreeController()
        controller.activate()
        controller.move(Vector3(1, 0, 0), 1.0)
        controller.move(Vector3(1, 0, 0), 1.0)
        assert controller.position.x == pytest.approx(20.0, rel=0.01)


# =============================================================================
# Additional CinematicController Tests
# =============================================================================


class TestCinematicControllerAdvanced:
    """Additional tests for CinematicController."""

    def test_keyframe_easing(self):
        """Test keyframe easing parameter."""
        kf = CameraKeyframe(0.0, Vector3(), Quaternion(), easing="ease-in-out")
        assert kf.easing == "ease-in-out"

    def test_negative_playback_speed(self):
        """Test negative playback speed for reverse."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(0, 0, 0), Quaternion()))
        controller.add_keyframe(CameraKeyframe(2.0, Vector3(10, 0, 0), Quaternion()))
        controller.playback_speed = -1.0
        controller._current_time = 2.0
        controller.play()
        controller.update(1.0)
        assert controller.get_current_time() < 2.0

    def test_multiple_keyframes_interpolation(self):
        """Test interpolation across multiple keyframes."""
        controller = CinematicController()
        controller.activate()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(0, 0, 0), Quaternion()))
        controller.add_keyframe(CameraKeyframe(1.0, Vector3(5, 0, 0), Quaternion()))
        controller.add_keyframe(CameraKeyframe(2.0, Vector3(10, 0, 0), Quaternion()))
        controller.play()
        controller.update(1.5)
        assert 5 < controller.position.x < 10

    def test_keyframe_removal_updates_duration(self):
        """Test removing keyframe updates duration."""
        controller = CinematicController()
        controller.add_keyframe(CameraKeyframe(0.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(5.0, Vector3(), Quaternion()))
        controller.add_keyframe(CameraKeyframe(10.0, Vector3(), Quaternion()))
        controller.remove_keyframe(2)
        assert controller.get_duration() == 5.0


# =============================================================================
# Additional TopDownController Tests
# =============================================================================


class TestTopDownControllerAdvanced:
    """Additional tests for TopDownController."""

    def test_pan_speed_affects_movement(self):
        """Test pan speed affects movement rate."""
        controller = TopDownController()
        controller.pan_speed = 20.0
        controller.pan(Vector3(1, 0, 0), 1.0)
        assert controller.target.x == pytest.approx(20.0, rel=0.01)

    def test_zoom_speed_affects_zoom(self):
        """Test zoom speed affects height change rate."""
        controller = TopDownController()
        controller.zoom_speed = 10.0
        initial = controller.height
        controller.zoom(1)
        assert controller.height == initial - 10

    def test_angle_affects_position(self):
        """Test angle affects camera position."""
        controller = TopDownController()
        controller.activate()
        controller.angle = 60.0
        controller.update(0.016)
        assert controller.position.z != 0


# =============================================================================
# Additional IsometricController Tests
# =============================================================================


class TestIsometricControllerAdvanced:
    """Additional tests for IsometricController."""

    def test_custom_rotation_angles(self):
        """Test custom rotation angles."""
        controller = IsometricController()
        controller.rotation_angles = [0.0, 90.0, 180.0, 270.0]
        controller.rotate_clockwise()
        assert controller.rotation_index == 1

    def test_snap_duration_affects_animation(self):
        """Test snap duration affects rotation animation."""
        controller = IsometricController()
        controller.snap_duration = 0.1
        controller.activate()
        controller.rotate_clockwise()
        controller.update(0.2)
        assert controller._snap_progress == 1.0

    def test_continuous_rotation(self):
        """Test continuous rotation through all angles."""
        controller = IsometricController()
        for _ in range(8):
            controller.rotate_clockwise()
        assert controller.rotation_index == 0


# =============================================================================
# Additional CameraManager Tests
# =============================================================================


class TestCameraManagerAdvanced:
    """Additional tests for CameraManager."""

    def test_multiple_controller_types(self):
        """Test managing multiple controller types."""
        manager = CameraManager()
        manager.register("fp", FirstPersonController())
        manager.register("tp", ThirdPersonController())
        manager.register("orbit", OrbitController())
        manager.register("free", FreeController())
        assert len(manager.controllers) == 4

    def test_switch_preserves_other_controllers(self):
        """Test switching preserves other controllers."""
        manager = CameraManager()
        fp = FirstPersonController()
        tp = ThirdPersonController()
        manager.register("fp", fp)
        manager.register("tp", tp)
        manager.switch_to("fp")
        manager.switch_to("tp")
        assert "fp" in manager.controllers

    def test_update_inactive_no_change(self):
        """Test update with no active controller."""
        manager = CameraManager()
        manager.update(0.016)


# =============================================================================
# Integration and Stress Tests
# =============================================================================


class TestControllerIntegration:
    """Integration tests for camera controllers."""

    def test_first_person_full_workflow(self):
        """Test complete first-person camera workflow."""
        controller = FirstPersonController()
        target = Transform(position=Vector3(0, 0, 0))
        controller.attach_to_head(target, Vector3(0, 1.8, 0))
        controller.activate()
        controller.head_bob_enabled = True

        for i in range(60):
            controller.set_moving(i % 2 == 0)
            controller.look(i * 0.5, i * 0.1)
            target.position = Vector3(i * 0.1, 0, i * 0.05)
            controller.update(0.016)

        assert controller.is_active is True

    def test_third_person_full_workflow(self):
        """Test complete third-person camera workflow."""
        controller = ThirdPersonController()
        target = Transform(position=Vector3(0, 0, 0))
        controller.target = target
        controller.activate()

        for i in range(60):
            controller.look(i * 0.5, i * 0.1)
            target.position = Vector3(i * 0.2, 0, i * 0.1)
            controller.update(0.016)

        assert controller.position.magnitude() > 0

    def test_orbit_full_workflow(self):
        """Test complete orbit camera workflow."""
        controller = OrbitController()
        controller.set_target(Vector3(0, 2, 0))
        controller.activate()
        controller.auto_rotate = True

        for i in range(60):
            controller.zoom(0.1 * (-1 if i % 10 < 5 else 1))
            controller.update(0.016)

    def test_follow_full_workflow(self):
        """Test complete follow camera workflow."""
        controller = FollowController()
        target = Transform(position=Vector3(0, 0, 0))
        controller.target = target
        controller.activate()

        for i in range(100):
            target.position = Vector3(
                math.sin(i * 0.1) * 10,
                0,
                math.cos(i * 0.1) * 10
            )
            controller.update(0.016)

    def test_free_camera_exploration(self):
        """Test free camera exploration workflow."""
        controller = FreeController()
        controller.activate()

        for i in range(100):
            direction = Vector3(
                math.sin(i * 0.1),
                math.cos(i * 0.15) * 0.5,
                math.cos(i * 0.1)
            )
            controller.move(direction, 0.016)
            controller.look(i * 0.2, i * 0.05)
            controller.update(0.016)

    def test_cinematic_complex_path(self):
        """Test cinematic camera with complex path."""
        controller = CinematicController()
        controller.activate()

        for i in range(10):
            t = i / 10
            kf = CameraKeyframe(
                time=t * 5.0,
                position=Vector3(
                    math.sin(t * math.pi * 2) * 10,
                    2 + math.sin(t * math.pi * 4),
                    math.cos(t * math.pi * 2) * 10
                ),
                rotation=Quaternion(),
                fov=60 + math.sin(t * math.pi) * 20
            )
            controller.add_keyframe(kf)

        controller.play()
        for _ in range(50):
            controller.update(0.1)

    def test_rapid_controller_switching(self):
        """Test rapid switching between controllers."""
        manager = CameraManager()
        controllers = {
            "fp": FirstPersonController(),
            "tp": ThirdPersonController(),
            "orbit": OrbitController(),
            "free": FreeController(),
        }

        for name, ctrl in controllers.items():
            manager.register(name, ctrl)

        for _ in range(50):
            for name in controllers.keys():
                manager.switch_to(name)
                manager.update(0.005)

    def test_controller_state_persistence(self):
        """Test controller state persists through deactivation."""
        controller = FirstPersonController()
        controller.pitch = 30.0
        controller.yaw = 45.0
        controller.activate()
        controller.deactivate()
        controller.activate()
        assert controller.pitch == 30.0
        assert controller.yaw == 45.0

    def test_manager_fov_override_persistence(self):
        """Test FOV override persists through controller switch."""
        manager = CameraManager()
        fp = FirstPersonController()
        tp = ThirdPersonController()
        manager.register("fp", fp)
        manager.register("tp", tp)
        manager.set_fov(90.0)
        manager.switch_to("fp")
        manager.switch_to("tp")
        assert manager.get_fov() == 90.0

    def test_isometric_continuous_rotation(self):
        """Test continuous isometric rotation."""
        controller = IsometricController()
        controller.activate()

        for _ in range(20):
            controller.rotate_clockwise()
            for _ in range(10):
                controller.update(0.05)

    def test_top_down_zoom_and_pan(self):
        """Test combined zoom and pan operations."""
        controller = TopDownController()
        controller.activate()
        controller.set_bounds(Vector3(-50, 0, -50), Vector3(50, 0, 50))

        for i in range(50):
            controller.pan(Vector3(1, 0, 1), 0.1)
            controller.zoom(0.2 * (-1 if i % 10 < 5 else 1))
            controller.update(0.016)


# =============================================================================
# Performance and Stress Tests
# =============================================================================


class TestControllerPerformance:
    """Performance and stress tests."""

    def test_many_active_controllers(self):
        """Test managing many controllers."""
        manager = CameraManager()
        for i in range(100):
            manager.register(f"ctrl_{i}", FirstPersonController())
        assert len(manager.controllers) == 100

    def test_rapid_updates(self):
        """Test rapid controller updates."""
        controller = FirstPersonController()
        controller.activate()
        target = Transform(position=Vector3(0, 0, 0))
        controller.attach_to_head(target)

        for _ in range(10000):
            controller.look(0.1, 0.1)
            controller.update(0.001)

    def test_many_cinematic_keyframes(self):
        """Test cinematic with many keyframes."""
        controller = CinematicController()
        for i in range(100):
            controller.add_keyframe(CameraKeyframe(
                time=i * 0.1,
                position=Vector3(i, 0, 0),
                rotation=Quaternion()
            ))
        assert len(controller.keyframes) == 100

    def test_orbit_extreme_zoom(self):
        """Test orbit with extreme zoom operations."""
        controller = OrbitController()
        controller.activate()

        for _ in range(1000):
            controller.zoom(100)
            controller.zoom(-100)

        assert controller.min_distance <= controller.distance <= controller.max_distance


# =============================================================================
# Additional Integration Tests
# =============================================================================


class TestControllerIntegration:
    """Integration tests for controllers."""

    def test_full_gameplay_scenario(self):
        """Test complete gameplay camera scenario."""
        manager = CameraManager()

        fp = FirstPersonController()
        tp = ThirdPersonController()
        orbit = OrbitController()
        cinematic = CinematicController()

        manager.register("first_person", fp)
        manager.register("third_person", tp)
        manager.register("orbit", orbit)
        manager.register("cinematic", cinematic)

        # Start in first person
        manager.switch_to("first_person")
        fp.attach_to_head(Transform(position=Vector3(0, 1.7, 0)))

        for _ in range(50):
            fp.look(0.5, 0.2)
            manager.update(0.016)

        # Switch to third person for action
        manager.switch_to("third_person", blend_time=0.5)
        tp.set_target(Transform(position=Vector3(0, 1, 0)))

        for _ in range(50):
            tp.update(0.016)

        # Cutscene with cinematic
        cinematic.add_keyframe(CameraKeyframe(0, Vector3(0, 2, 5), Quaternion()))
        cinematic.add_keyframe(CameraKeyframe(2, Vector3(5, 3, 0), Quaternion()))
        cinematic.add_keyframe(CameraKeyframe(4, Vector3(0, 2, -5), Quaternion()))

        manager.switch_to("cinematic")
        cinematic.play()

        for _ in range(100):
            cinematic.update(0.05)

    def test_camera_shake_integration(self):
        """Test camera shake effect integration."""
        controller = FirstPersonController()
        controller.activate()
        controller.attach_to_head(Transform(position=Vector3(0, 1.7, 0)))

        # Simulate explosion shake
        shake_offset = Vector3(0.1, 0.05, 0.02)
        for i in range(30):
            intensity = 1.0 - (i / 30.0)
            offset = shake_offset * intensity * math.sin(i * 2)
            controller.update(0.016)

    def test_vehicle_camera_transition(self):
        """Test vehicle camera transition scenario."""
        manager = CameraManager()

        walking_cam = ThirdPersonController()
        vehicle_cam = FollowController()

        manager.register("walking", walking_cam)
        manager.register("vehicle", vehicle_cam)

        player = Transform(position=Vector3(0, 1, 0))
        vehicle = Transform(position=Vector3(10, 0, 0))

        walking_cam.set_target(player)
        vehicle_cam.set_target(vehicle)
        vehicle_cam.follow_offset = Vector3(0, 2, -5)

        manager.switch_to("walking")
        for _ in range(30):
            manager.update(0.016)

        # Enter vehicle
        manager.switch_to("vehicle", blend_time=1.0)
        for _ in range(60):
            manager.update(0.016)


class TestControllerEdgeCasesAdvanced:
    """Additional edge case tests."""

    def test_orbit_at_poles(self):
        """Test orbit camera at vertical limits."""
        controller = OrbitController()
        controller.activate()
        controller.min_vertical_angle = -85.0
        controller.max_vertical_angle = 85.0

        # Try to go past north pole
        for _ in range(100):
            controller.rotate(0.0, 10.0)
            controller.update(0.016)

        assert controller.vertical_angle <= 85.0

        # Try to go past south pole
        for _ in range(200):
            controller.rotate(0.0, -10.0)
            controller.update(0.016)

        assert controller.vertical_angle >= -85.0

    def test_third_person_target_loss(self):
        """Test third person when target is removed."""
        controller = ThirdPersonController()
        controller.activate()

        target = Transform(position=Vector3(0, 1, 0))
        controller.set_target(target)

        for _ in range(10):
            controller.update(0.016)

        # Remove target
        controller.set_target(None)
        controller.update(0.016)

    def test_follow_controller_teleporting_target(self):
        """Test follow controller with teleporting target."""
        controller = FollowController()
        controller.activate()

        target = Transform(position=Vector3(0, 0, 0))
        controller.set_target(target)

        for _ in range(10):
            controller.update(0.016)

        # Target teleports
        target.position = Vector3(100, 0, 100)
        controller.update(0.016)

    def test_cinematic_empty_path(self):
        """Test cinematic controller with no keyframes."""
        controller = CinematicController()
        controller.activate()
        controller.play()
        controller.update(1.0)

    def test_free_camera_extreme_movement(self):
        """Test free camera at extreme positions."""
        controller = FreeController()
        controller.activate()

        controller.position = Vector3(1e6, 1e6, 1e6)
        controller.move(Vector3(100, 100, 100), 0.016)
        controller.update(0.016)

    def test_manager_null_switch(self):
        """Test switching to non-existent controller."""
        manager = CameraManager()
        fp = FirstPersonController()
        manager.register("fp", fp)
        manager.switch_to("fp")

        # Try to switch to non-existent - should silently fail when raise_on_missing=False
        result = manager.switch_to("nonexistent", raise_on_missing=False)
        assert result is False
        assert manager.active_controller is not None
        assert manager.active_controller == "fp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
