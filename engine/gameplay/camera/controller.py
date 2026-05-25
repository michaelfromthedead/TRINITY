"""Camera Controllers - Various camera behavior implementations.

This module provides different camera controller types including first-person,
third-person, orbit, follow, free, cinematic, top-down, and isometric cameras.
Each controller manages position, rotation, FOV, and target tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import math

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4

from engine.gameplay.camera.constants import (
    DEFAULT_FOV, MIN_FOV, MAX_FOV,
    DEFAULT_CAMERA_DISTANCE, MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE,
    DEFAULT_ORBIT_DISTANCE, MIN_ORBIT_DISTANCE, MAX_ORBIT_DISTANCE,
    DEFAULT_BOOM_ARM_LENGTH,
    DEFAULT_SOCKET_OFFSET_X, DEFAULT_SOCKET_OFFSET_Y, DEFAULT_SOCKET_OFFSET_Z,
    DEFAULT_TARGET_OFFSET_X, DEFAULT_TARGET_OFFSET_Y, DEFAULT_TARGET_OFFSET_Z,
    DEFAULT_CAMERA_LAG_SPEED, DEFAULT_ROTATION_LAG_SPEED,
    MAX_LAG_DISTANCE, LAG_RECOVERY_SPEED,
    DEFAULT_EYE_HEIGHT, CROUCH_EYE_HEIGHT,
    DEFAULT_HEAD_BOB_AMPLITUDE, DEFAULT_HEAD_BOB_FREQUENCY, DEFAULT_HEAD_BOB_SWAY,
    MIN_PITCH_ANGLE, MAX_PITCH_ANGLE,
    MIN_ORBIT_PITCH, MAX_ORBIT_PITCH,
    TOP_DOWN_PITCH, ISOMETRIC_PITCH, ISOMETRIC_ROTATION_SNAP,
    DEFAULT_MOUSE_SENSITIVITY,
    DEFAULT_ORBIT_ROTATION_SPEED, DEFAULT_ZOOM_SPEED,
    DEFAULT_FREE_CAM_SPEED, FREE_CAM_FAST_MULTIPLIER, FREE_CAM_SLOW_MULTIPLIER,
    DEFAULT_NEAR_PLANE, DEFAULT_FAR_PLANE, MIN_NEAR_PLANE,
    DEG_TO_RAD, RAD_TO_DEG,
    CAMERA_EPSILON, MIN_DELTA_TIME, MAX_DELTA_TIME,
    DEFAULT_ASPECT_RATIO,
    HEAD_BOB_DECAY_FACTOR,
    EYE_HEIGHT_INTERP_SPEED,
    BOOM_LENGTH_INTERP_SPEED,
    DISTANCE_INTERP_SPEED,
    DEFAULT_AUTO_ROTATE_SPEED,
    ISOMETRIC_ROTATION_TRANSITION_SPEED,
)

if TYPE_CHECKING:
    from engine.gameplay.components.transform import TransformComponent


class CameraMode(Enum):
    """Available camera mode types."""
    FIRST_PERSON = auto()   # Attached to pawn head, mouse look
    THIRD_PERSON = auto()   # Boom arm behind character with offset
    ORBIT = auto()          # Orbits around a point with zoom
    FOLLOW = auto()         # Damped follow with lead prediction
    FREE = auto()           # Fly camera, WASD + mouse
    CINEMATIC = auto()      # Keyframe-based, timeline-driven
    TOP_DOWN = auto()       # Fixed angle looking down
    ISOMETRIC = auto()      # 45-degree angle with rotation snap


@dataclass(slots=True)
class CameraState:
    """Snapshot of camera state for interpolation and blending."""
    position: Vec3 = field(default_factory=Vec3.zero)
    rotation: Quat = field(default_factory=Quat.identity)
    fov: float = DEFAULT_FOV
    near_plane: float = DEFAULT_NEAR_PLANE
    far_plane: float = DEFAULT_FAR_PLANE
    timestamp: float = 0.0

    def lerp(self, other: CameraState, t: float) -> CameraState:
        """Interpolate between two camera states."""
        t = max(0.0, min(1.0, t))
        return CameraState(
            position=self.position.lerp(other.position, t),
            rotation=self.rotation.slerp(other.rotation, t),
            fov=self.fov + (other.fov - self.fov) * t,
            near_plane=self.near_plane + (other.near_plane - self.near_plane) * t,
            far_plane=self.far_plane + (other.far_plane - self.far_plane) * t,
            timestamp=self.timestamp + (other.timestamp - self.timestamp) * t,
        )


class BaseCameraController(ABC):
    """
    Abstract base class for all camera controllers.

    Provides common functionality for position, rotation, FOV management,
    target tracking, and view matrix calculation.

    Attributes:
        position: Camera world position
        rotation: Camera orientation as quaternion
        fov: Field of view in degrees
        target: Optional target to track
        mode: The camera mode type
    """

    __slots__ = (
        "_position",
        "_rotation",
        "_fov",
        "_target",
        "_target_position",
        "_near_plane",
        "_far_plane",
        "_view_matrix_cache",
        "_view_matrix_dirty",
        "_projection_matrix_cache",
        "_projection_matrix_dirty",
        "_aspect_ratio",
        "_mode",
        "_enabled",
        "_on_state_changed",
    )

    def __init__(
        self,
        position: Optional[Vec3] = None,
        rotation: Optional[Quat] = None,
        fov: float = DEFAULT_FOV,
        mode: CameraMode = CameraMode.FREE,
    ) -> None:
        """
        Initialize the base camera controller.

        Args:
            position: Initial camera position
            rotation: Initial camera rotation
            fov: Initial field of view in degrees
            mode: Camera mode type
        """
        self._position = position if position is not None else Vec3.zero()
        self._rotation = rotation if rotation is not None else Quat.identity()
        self._fov = max(MIN_FOV, min(MAX_FOV, fov))
        self._target: Optional[TransformComponent] = None
        self._target_position = Vec3.zero()
        self._near_plane = DEFAULT_NEAR_PLANE
        self._far_plane = DEFAULT_FAR_PLANE
        self._view_matrix_cache: Optional[Mat4] = None
        self._view_matrix_dirty = True
        self._projection_matrix_cache: Optional[Mat4] = None
        self._projection_matrix_dirty = True
        self._aspect_ratio = DEFAULT_ASPECT_RATIO
        self._mode = mode
        self._enabled = True
        self._on_state_changed: List[Callable[[BaseCameraController], None]] = []

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def position(self) -> Vec3:
        """Get camera world position."""
        return self._position

    @position.setter
    def position(self, value: Vec3) -> None:
        """Set camera world position."""
        self._position = value
        self._view_matrix_dirty = True
        self._notify_state_changed()

    @property
    def rotation(self) -> Quat:
        """Get camera orientation."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Quat) -> None:
        """Set camera orientation."""
        self._rotation = value.normalized()
        self._view_matrix_dirty = True
        self._notify_state_changed()

    @property
    def fov(self) -> float:
        """Get field of view in degrees."""
        return self._fov

    @fov.setter
    def fov(self, value: float) -> None:
        """Set field of view in degrees (clamped)."""
        self._fov = max(MIN_FOV, min(MAX_FOV, value))
        self._projection_matrix_dirty = True
        self._notify_state_changed()

    @property
    def target(self) -> Optional[TransformComponent]:
        """Get the target being tracked."""
        return self._target

    @target.setter
    def target(self, value: Optional[TransformComponent]) -> None:
        """Set the target to track."""
        self._target = value

    @property
    def target_position(self) -> Vec3:
        """Get the target position (from target or manual)."""
        if self._target is not None:
            return self._target.world_position
        return self._target_position

    @target_position.setter
    def target_position(self, value: Vec3) -> None:
        """Set manual target position (used when no target transform)."""
        self._target_position = value

    @property
    def mode(self) -> CameraMode:
        """Get the camera mode."""
        return self._mode

    @property
    def enabled(self) -> bool:
        """Check if camera is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the camera."""
        self._enabled = value

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio."""
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, value: float) -> None:
        """Set aspect ratio."""
        if value > CAMERA_EPSILON:
            self._aspect_ratio = value
            self._projection_matrix_dirty = True

    @property
    def near_plane(self) -> float:
        """Get near clipping plane distance."""
        return self._near_plane

    @near_plane.setter
    def near_plane(self, value: float) -> None:
        """Set near clipping plane distance."""
        self._near_plane = max(MIN_NEAR_PLANE, value)
        self._projection_matrix_dirty = True

    @property
    def far_plane(self) -> float:
        """Get far clipping plane distance."""
        return self._far_plane

    @far_plane.setter
    def far_plane(self, value: float) -> None:
        """Set far clipping plane distance."""
        self._far_plane = max(self._near_plane + 1.0, value)
        self._projection_matrix_dirty = True

    # =========================================================================
    # DIRECTION VECTORS
    # =========================================================================

    @property
    def forward(self) -> Vec3:
        """Get forward direction vector."""
        return self._rotation.rotate_vector(Vec3(0, 0, -1))

    @property
    def up(self) -> Vec3:
        """Get up direction vector."""
        return self._rotation.rotate_vector(Vec3(0, 1, 0))

    @property
    def right(self) -> Vec3:
        """Get right direction vector."""
        return self._rotation.rotate_vector(Vec3(1, 0, 0))

    # =========================================================================
    # MATRICES
    # =========================================================================

    @property
    def view_matrix(self) -> Mat4:
        """Get the view matrix (world to camera space)."""
        if self._view_matrix_dirty or self._view_matrix_cache is None:
            self._view_matrix_cache = Mat4.look_at(
                self._position,
                self._position + self.forward,
                Vec3.up()
            )
            self._view_matrix_dirty = False
        return self._view_matrix_cache

    @property
    def projection_matrix(self) -> Mat4:
        """Get the projection matrix."""
        if self._projection_matrix_dirty or self._projection_matrix_cache is None:
            fov_rad = self._fov * DEG_TO_RAD
            self._projection_matrix_cache = Mat4.perspective(
                fov_rad, self._aspect_ratio, self._near_plane, self._far_plane
            )
            self._projection_matrix_dirty = False
        return self._projection_matrix_cache

    @property
    def view_projection_matrix(self) -> Mat4:
        """Get combined view-projection matrix."""
        return self.projection_matrix @ self.view_matrix

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def get_state(self) -> CameraState:
        """Get current camera state as a snapshot."""
        return CameraState(
            position=Vec3(self._position.x, self._position.y, self._position.z),
            rotation=Quat(self._rotation.x, self._rotation.y, self._rotation.z, self._rotation.w),
            fov=self._fov,
            near_plane=self._near_plane,
            far_plane=self._far_plane,
        )

    def set_state(self, state: CameraState) -> None:
        """Apply a camera state snapshot."""
        self._position = state.position
        self._rotation = state.rotation
        self._fov = state.fov
        self._near_plane = state.near_plane
        self._far_plane = state.far_plane
        self._view_matrix_dirty = True
        self._projection_matrix_dirty = True
        self._notify_state_changed()

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_state_changed(self, callback: Callable[[BaseCameraController], None]) -> None:
        """Register callback for state changes."""
        self._on_state_changed.append(callback)

    def off_state_changed(self, callback: Callable[[BaseCameraController], None]) -> None:
        """Unregister state change callback."""
        if callback in self._on_state_changed:
            self._on_state_changed.remove(callback)

    def _notify_state_changed(self) -> None:
        """Notify listeners of state change."""
        for callback in self._on_state_changed:
            callback(self)

    # =========================================================================
    # LOOK AT
    # =========================================================================

    def look_at(self, target: Vec3, up: Vec3 = Vec3.up()) -> None:
        """Orient camera to look at a target point."""
        direction = (target - self._position).normalized()
        if direction.length_squared() < CAMERA_EPSILON:
            return

        forward = direction
        right = up.cross(forward).normalized()
        if right.length_squared() < CAMERA_EPSILON:
            right = Vec3.right()
        actual_up = forward.cross(right)

        # Convert basis to quaternion
        m00, m01, m02 = right.x, actual_up.x, forward.x
        m10, m11, m12 = right.y, actual_up.y, forward.y
        m20, m21, m22 = right.z, actual_up.z, forward.z

        trace = m00 + m11 + m22
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        self.rotation = Quat(x, y, z, w).normalized()

    # =========================================================================
    # ABSTRACT UPDATE
    # =========================================================================

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """
        Update the camera controller.

        Args:
            delta_time: Time since last update in seconds
        """
        pass

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize camera controller to dictionary."""
        return {
            "position": [self._position.x, self._position.y, self._position.z],
            "rotation": [self._rotation.x, self._rotation.y, self._rotation.z, self._rotation.w],
            "fov": self._fov,
            "near_plane": self._near_plane,
            "far_plane": self._far_plane,
            "mode": self._mode.name,
            "enabled": self._enabled,
        }


class FirstPersonController(BaseCameraController):
    """
    First-person camera controller attached to pawn head.

    Features:
    - Mouse look with pitch/yaw
    - Head bob during movement
    - Eye height adjustment for crouching
    - Weapon sway integration
    """

    __slots__ = (
        "_pitch",
        "_yaw",
        "_eye_height",
        "_target_eye_height",
        "_sensitivity",
        "_head_bob_time",
        "_head_bob_amplitude",
        "_head_bob_frequency",
        "_head_bob_sway",
        "_head_bob_offset",
        "_is_moving",
        "_movement_speed",
        "_invert_y",
    )

    def __init__(
        self,
        position: Optional[Vec3] = None,
        eye_height: float = DEFAULT_EYE_HEIGHT,
        sensitivity: float = DEFAULT_MOUSE_SENSITIVITY,
    ) -> None:
        """
        Initialize first-person camera controller.

        Args:
            position: Initial position
            eye_height: Eye height from ground
            sensitivity: Mouse look sensitivity
        """
        super().__init__(position, mode=CameraMode.FIRST_PERSON)
        self._pitch = 0.0  # Up/down rotation in degrees
        self._yaw = 0.0    # Left/right rotation in degrees
        self._eye_height = eye_height
        self._target_eye_height = eye_height
        self._sensitivity = sensitivity
        self._head_bob_time = 0.0
        self._head_bob_amplitude = DEFAULT_HEAD_BOB_AMPLITUDE
        self._head_bob_frequency = DEFAULT_HEAD_BOB_FREQUENCY
        self._head_bob_sway = DEFAULT_HEAD_BOB_SWAY
        self._head_bob_offset = Vec3.zero()
        self._is_moving = False
        self._movement_speed = 0.0
        self._invert_y = False

    @property
    def pitch(self) -> float:
        """Get pitch angle in degrees."""
        return self._pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        """Set pitch angle (clamped to limits)."""
        self._pitch = max(MIN_PITCH_ANGLE, min(MAX_PITCH_ANGLE, value))
        self._update_rotation_from_angles()

    @property
    def yaw(self) -> float:
        """Get yaw angle in degrees."""
        return self._yaw

    @yaw.setter
    def yaw(self, value: float) -> None:
        """Set yaw angle (wrapped to 0-360)."""
        self._yaw = value % 360.0
        self._update_rotation_from_angles()

    @property
    def eye_height(self) -> float:
        """Get current eye height."""
        return self._eye_height

    @eye_height.setter
    def eye_height(self, value: float) -> None:
        """Set target eye height (will interpolate)."""
        self._target_eye_height = max(0.0, value)

    @property
    def sensitivity(self) -> float:
        """Get mouse sensitivity."""
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        """Set mouse sensitivity."""
        self._sensitivity = max(0.001, value)

    @property
    def invert_y(self) -> bool:
        """Check if Y axis is inverted."""
        return self._invert_y

    @invert_y.setter
    def invert_y(self, value: bool) -> None:
        """Set Y axis inversion."""
        self._invert_y = value

    def set_crouching(self, crouching: bool) -> None:
        """Set crouching state (affects eye height)."""
        self._target_eye_height = CROUCH_EYE_HEIGHT if crouching else DEFAULT_EYE_HEIGHT

    def set_movement(self, is_moving: bool, speed: float = 0.0) -> None:
        """Set movement state for head bob."""
        self._is_moving = is_moving
        self._movement_speed = speed

    def add_input(self, delta_yaw: float, delta_pitch: float) -> None:
        """
        Add mouse input to camera rotation.

        Args:
            delta_yaw: Horizontal mouse movement
            delta_pitch: Vertical mouse movement
        """
        self._yaw += delta_yaw * self._sensitivity
        self._yaw = self._yaw % 360.0

        pitch_delta = delta_pitch * self._sensitivity
        if self._invert_y:
            pitch_delta = -pitch_delta
        self._pitch = max(MIN_PITCH_ANGLE, min(MAX_PITCH_ANGLE, self._pitch - pitch_delta))

        self._update_rotation_from_angles()

    def _update_rotation_from_angles(self) -> None:
        """Update quaternion rotation from pitch/yaw angles."""
        yaw_rad = self._yaw * DEG_TO_RAD
        pitch_rad = self._pitch * DEG_TO_RAD
        self._rotation = Quat.from_euler(pitch_rad, yaw_rad, 0.0)
        self._view_matrix_dirty = True

    def _update_head_bob(self, delta_time: float) -> None:
        """Update head bob based on movement."""
        if self._is_moving and self._movement_speed > 0.1:
            self._head_bob_time += delta_time * self._head_bob_frequency * self._movement_speed
            bob_y = math.sin(self._head_bob_time * math.pi * 2.0) * self._head_bob_amplitude
            bob_x = math.sin(self._head_bob_time * math.pi) * self._head_bob_sway
            self._head_bob_offset = Vec3(bob_x, bob_y, 0.0)
        else:
            # Decay head bob when stopped
            self._head_bob_offset = self._head_bob_offset * HEAD_BOB_DECAY_FACTOR
            if self._head_bob_offset.length() < 0.01:
                self._head_bob_offset = Vec3.zero()
                self._head_bob_time = 0.0

    def update(self, delta_time: float) -> None:
        """Update first-person camera."""
        if not self._enabled:
            return

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Interpolate eye height
        height_diff = self._target_eye_height - self._eye_height
        if abs(height_diff) > 0.01:
            self._eye_height += height_diff * min(1.0, delta_time * EYE_HEIGHT_INTERP_SPEED)

        # Update head bob
        self._update_head_bob(delta_time)

        # Update position from target transform if available
        if self._target is not None:
            base_pos = self._target.world_position
            self._position = Vec3(
                base_pos.x + self._head_bob_offset.x,
                base_pos.y + self._eye_height + self._head_bob_offset.y,
                base_pos.z + self._head_bob_offset.z
            )
            self._view_matrix_dirty = True


class ThirdPersonController(BaseCameraController):
    """
    Third-person camera with boom arm, offset, and lag.

    Features:
    - Boom arm with collision detection support
    - Socket and target offsets
    - Camera lag for smooth following
    - Look-at target tracking
    """

    __slots__ = (
        "_boom_arm_length",
        "_target_boom_length",
        "_socket_offset",
        "_target_offset",
        "_lag_speed",
        "_rotation_lag_speed",
        "_lagged_position",
        "_lagged_rotation",
        "_pitch",
        "_yaw",
        "_sensitivity",
    )

    def __init__(
        self,
        position: Optional[Vec3] = None,
        boom_length: float = DEFAULT_BOOM_ARM_LENGTH,
        socket_offset: Optional[Vec3] = None,
        target_offset: Optional[Vec3] = None,
    ) -> None:
        """
        Initialize third-person camera controller.

        Args:
            position: Initial position
            boom_length: Camera boom arm length
            socket_offset: Offset from character pivot
            target_offset: Offset for look-at point
        """
        super().__init__(position, mode=CameraMode.THIRD_PERSON)
        self._boom_arm_length = boom_length
        self._target_boom_length = boom_length
        self._socket_offset = socket_offset if socket_offset is not None else Vec3(
            DEFAULT_SOCKET_OFFSET_X,
            DEFAULT_SOCKET_OFFSET_Y,
            DEFAULT_SOCKET_OFFSET_Z
        )
        self._target_offset = target_offset if target_offset is not None else Vec3(
            DEFAULT_TARGET_OFFSET_X,
            DEFAULT_TARGET_OFFSET_Y,
            DEFAULT_TARGET_OFFSET_Z
        )
        self._lag_speed = DEFAULT_CAMERA_LAG_SPEED
        self._rotation_lag_speed = DEFAULT_ROTATION_LAG_SPEED
        self._lagged_position = self._position
        self._lagged_rotation = self._rotation
        self._pitch = 0.0
        self._yaw = 0.0
        self._sensitivity = DEFAULT_MOUSE_SENSITIVITY

    @property
    def boom_arm_length(self) -> float:
        """Get boom arm length."""
        return self._boom_arm_length

    @boom_arm_length.setter
    def boom_arm_length(self, value: float) -> None:
        """Set target boom arm length."""
        self._target_boom_length = max(MIN_CAMERA_DISTANCE, min(MAX_CAMERA_DISTANCE, value))

    @property
    def socket_offset(self) -> Vec3:
        """Get socket offset."""
        return self._socket_offset

    @socket_offset.setter
    def socket_offset(self, value: Vec3) -> None:
        """Set socket offset."""
        self._socket_offset = value

    @property
    def target_offset(self) -> Vec3:
        """Get target offset."""
        return self._target_offset

    @target_offset.setter
    def target_offset(self, value: Vec3) -> None:
        """Set target offset."""
        self._target_offset = value

    @property
    def lag_speed(self) -> float:
        """Get camera lag speed."""
        return self._lag_speed

    @lag_speed.setter
    def lag_speed(self, value: float) -> None:
        """Set camera lag speed (higher = less lag)."""
        self._lag_speed = max(0.1, value)

    def add_input(self, delta_yaw: float, delta_pitch: float) -> None:
        """Add mouse input to orbit rotation."""
        self._yaw += delta_yaw * self._sensitivity
        self._yaw = self._yaw % 360.0
        self._pitch = max(MIN_PITCH_ANGLE, min(MAX_PITCH_ANGLE, self._pitch - delta_pitch * self._sensitivity))

    def get_desired_position(self) -> Vec3:
        """Calculate desired camera position based on current state."""
        if self._target is None:
            return self._position

        # Get target position with socket offset
        target_world_pos = self._target.world_position
        socket_pos = target_world_pos + self._socket_offset

        # Calculate boom direction from pitch/yaw
        yaw_rad = self._yaw * DEG_TO_RAD
        pitch_rad = self._pitch * DEG_TO_RAD

        # Direction from target to camera
        cos_pitch = math.cos(pitch_rad)
        boom_dir = Vec3(
            math.sin(yaw_rad) * cos_pitch,
            math.sin(pitch_rad),
            math.cos(yaw_rad) * cos_pitch
        )

        # Camera position is socket + boom direction * length
        return socket_pos + boom_dir * self._boom_arm_length

    def update(self, delta_time: float) -> None:
        """Update third-person camera."""
        if not self._enabled:
            return

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Interpolate boom arm length
        length_diff = self._target_boom_length - self._boom_arm_length
        if abs(length_diff) > 0.01:
            self._boom_arm_length += length_diff * min(1.0, delta_time * BOOM_LENGTH_INTERP_SPEED)

        # Get desired position
        desired_pos = self.get_desired_position()

        # Apply camera lag
        lag_factor = 1.0 - math.exp(-self._lag_speed * delta_time)
        self._lagged_position = self._lagged_position.lerp(desired_pos, lag_factor)

        # Check if lag distance is too great
        lag_distance = (self._lagged_position - desired_pos).length()
        if lag_distance > MAX_LAG_DISTANCE:
            recover_dir = (desired_pos - self._lagged_position).normalized()
            self._lagged_position = self._lagged_position + recover_dir * LAG_RECOVERY_SPEED * delta_time

        self._position = self._lagged_position
        self._view_matrix_dirty = True

        # Look at target
        if self._target is not None:
            look_target = self._target.world_position + self._target_offset
            self.look_at(look_target)


class OrbitController(BaseCameraController):
    """
    Orbit camera that circles around a target point.

    Features:
    - Distance (zoom) control
    - Yaw and pitch rotation
    - Configurable pitch limits
    - Smooth zoom transitions
    """

    __slots__ = (
        "_distance",
        "_target_distance",
        "_yaw",
        "_pitch",
        "_min_distance",
        "_max_distance",
        "_min_pitch",
        "_max_pitch",
        "_rotation_speed",
        "_zoom_speed",
        "_auto_rotate",
        "_auto_rotate_speed",
    )

    def __init__(
        self,
        target_position: Optional[Vec3] = None,
        distance: float = DEFAULT_ORBIT_DISTANCE,
        yaw: float = 0.0,
        pitch: float = -30.0,
    ) -> None:
        """
        Initialize orbit camera controller.

        Args:
            target_position: Point to orbit around
            distance: Initial orbit distance
            yaw: Initial yaw angle in degrees
            pitch: Initial pitch angle in degrees
        """
        super().__init__(mode=CameraMode.ORBIT)
        if target_position is not None:
            self._target_position = target_position
        self._distance = distance
        self._target_distance = distance
        self._yaw = yaw
        self._pitch = pitch
        self._min_distance = MIN_ORBIT_DISTANCE
        self._max_distance = MAX_ORBIT_DISTANCE
        self._min_pitch = MIN_ORBIT_PITCH
        self._max_pitch = MAX_ORBIT_PITCH
        self._rotation_speed = DEFAULT_ORBIT_ROTATION_SPEED
        self._zoom_speed = DEFAULT_ZOOM_SPEED
        self._auto_rotate = False
        self._auto_rotate_speed = DEFAULT_AUTO_ROTATE_SPEED

    @property
    def distance(self) -> float:
        """Get current orbit distance."""
        return self._distance

    @distance.setter
    def distance(self, value: float) -> None:
        """Set target orbit distance (will interpolate)."""
        self._target_distance = max(self._min_distance, min(self._max_distance, value))

    @property
    def yaw(self) -> float:
        """Get yaw angle in degrees."""
        return self._yaw

    @yaw.setter
    def yaw(self, value: float) -> None:
        """Set yaw angle."""
        self._yaw = value % 360.0

    @property
    def pitch(self) -> float:
        """Get pitch angle in degrees."""
        return self._pitch

    @pitch.setter
    def pitch(self, value: float) -> None:
        """Set pitch angle (clamped)."""
        self._pitch = max(self._min_pitch, min(self._max_pitch, value))

    def set_pitch_limits(self, min_pitch: float, max_pitch: float) -> None:
        """Set pitch angle limits."""
        self._min_pitch = min_pitch
        self._max_pitch = max_pitch
        self._pitch = max(min_pitch, min(max_pitch, self._pitch))

    def set_distance_limits(self, min_dist: float, max_dist: float) -> None:
        """Set zoom distance limits."""
        self._min_distance = max(1.0, min_dist)
        self._max_distance = max(self._min_distance, max_dist)

    def zoom(self, delta: float) -> None:
        """Adjust zoom by delta amount."""
        self.distance = self._target_distance - delta * self._zoom_speed * 0.01

    def rotate(self, delta_yaw: float, delta_pitch: float) -> None:
        """Rotate orbit by delta angles."""
        self._yaw = (self._yaw + delta_yaw) % 360.0
        self._pitch = max(self._min_pitch, min(self._max_pitch, self._pitch + delta_pitch))

    def set_auto_rotate(self, enabled: bool, speed: float = 10.0) -> None:
        """Enable/disable automatic rotation."""
        self._auto_rotate = enabled
        self._auto_rotate_speed = speed

    def update(self, delta_time: float) -> None:
        """Update orbit camera."""
        if not self._enabled:
            return

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Auto rotate
        if self._auto_rotate:
            self._yaw = (self._yaw + self._auto_rotate_speed * delta_time) % 360.0

        # Interpolate distance
        dist_diff = self._target_distance - self._distance
        if abs(dist_diff) > 0.01:
            self._distance += dist_diff * min(1.0, delta_time * DISTANCE_INTERP_SPEED)

        # Calculate camera position
        yaw_rad = self._yaw * DEG_TO_RAD
        pitch_rad = self._pitch * DEG_TO_RAD

        cos_pitch = math.cos(pitch_rad)
        offset = Vec3(
            math.sin(yaw_rad) * cos_pitch * self._distance,
            math.sin(pitch_rad) * self._distance,
            math.cos(yaw_rad) * cos_pitch * self._distance
        )

        self._position = self.target_position + offset
        self._view_matrix_dirty = True

        # Look at target
        self.look_at(self.target_position)


class FollowController(BaseCameraController):
    """
    Damped follow camera with lead prediction and framing.

    Features:
    - Smooth damped following
    - Lead prediction based on target velocity
    - Smart framing to keep target in view
    - Configurable follow parameters
    """

    __slots__ = (
        "_offset",
        "_follow_speed",
        "_rotation_speed",
        "_lead_amount",
        "_lead_speed",
        "_current_lead",
        "_last_target_pos",
        "_target_velocity",
        "_frame_offset",
    )

    def __init__(
        self,
        offset: Optional[Vec3] = None,
        follow_speed: float = DEFAULT_CAMERA_LAG_SPEED,
    ) -> None:
        """
        Initialize follow camera controller.

        Args:
            offset: Camera offset from target
            follow_speed: Following interpolation speed
        """
        super().__init__(mode=CameraMode.FOLLOW)
        self._offset = offset if offset is not None else Vec3(0, 200, 400)
        self._follow_speed = follow_speed
        self._rotation_speed = DEFAULT_ROTATION_LAG_SPEED
        self._lead_amount = 50.0
        self._lead_speed = 5.0
        self._current_lead = Vec3.zero()
        self._last_target_pos = Vec3.zero()
        self._target_velocity = Vec3.zero()
        self._frame_offset = Vec3.zero()

    @property
    def offset(self) -> Vec3:
        """Get camera offset from target."""
        return self._offset

    @offset.setter
    def offset(self, value: Vec3) -> None:
        """Set camera offset from target."""
        self._offset = value

    @property
    def follow_speed(self) -> float:
        """Get follow interpolation speed."""
        return self._follow_speed

    @follow_speed.setter
    def follow_speed(self, value: float) -> None:
        """Set follow interpolation speed."""
        self._follow_speed = max(0.1, value)

    @property
    def lead_amount(self) -> float:
        """Get lead prediction amount."""
        return self._lead_amount

    @lead_amount.setter
    def lead_amount(self, value: float) -> None:
        """Set lead prediction amount."""
        self._lead_amount = max(0.0, value)

    def set_frame_offset(self, offset: Vec3) -> None:
        """Set additional frame offset (for screen-space adjustments)."""
        self._frame_offset = offset

    def _estimate_velocity(self, delta_time: float) -> None:
        """Estimate target velocity from position change."""
        if self._target is not None:
            current_pos = self._target.world_position
            if delta_time > CAMERA_EPSILON:
                self._target_velocity = (current_pos - self._last_target_pos) / delta_time
            self._last_target_pos = current_pos

    def _calculate_lead(self, delta_time: float) -> None:
        """Calculate lead offset based on velocity."""
        # Only apply horizontal lead
        horizontal_vel = Vec3(self._target_velocity.x, 0, self._target_velocity.z)
        target_lead = horizontal_vel.normalized() * min(horizontal_vel.length(), 1.0) * self._lead_amount

        # Smoothly interpolate lead
        lead_factor = 1.0 - math.exp(-self._lead_speed * delta_time)
        self._current_lead = self._current_lead.lerp(target_lead, lead_factor)

    def update(self, delta_time: float) -> None:
        """Update follow camera."""
        if not self._enabled:
            return

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Estimate target velocity
        self._estimate_velocity(delta_time)

        # Calculate lead
        self._calculate_lead(delta_time)

        # Calculate desired position
        target_pos = self.target_position
        desired_pos = target_pos + self._offset + self._current_lead + self._frame_offset

        # Smooth follow
        follow_factor = 1.0 - math.exp(-self._follow_speed * delta_time)
        self._position = self._position.lerp(desired_pos, follow_factor)
        self._view_matrix_dirty = True

        # Smooth look-at
        self.look_at(target_pos + self._current_lead)


class FreeController(BaseCameraController):
    """
    Free fly camera with WASD movement and mouse look.

    Features:
    - WASD movement in camera space
    - Mouse look with pitch/yaw
    - Speed modifiers (fast/slow)
    - Smooth movement
    """

    __slots__ = (
        "_pitch",
        "_yaw",
        "_move_speed",
        "_fast_multiplier",
        "_slow_multiplier",
        "_sensitivity",
        "_velocity",
        "_input_direction",
        "_is_fast",
        "_is_slow",
    )

    def __init__(
        self,
        position: Optional[Vec3] = None,
        move_speed: float = DEFAULT_FREE_CAM_SPEED,
    ) -> None:
        """
        Initialize free camera controller.

        Args:
            position: Initial position
            move_speed: Movement speed
        """
        super().__init__(position, mode=CameraMode.FREE)
        self._pitch = 0.0
        self._yaw = 0.0
        self._move_speed = move_speed
        self._fast_multiplier = FREE_CAM_FAST_MULTIPLIER
        self._slow_multiplier = FREE_CAM_SLOW_MULTIPLIER
        self._sensitivity = DEFAULT_MOUSE_SENSITIVITY
        self._velocity = Vec3.zero()
        self._input_direction = Vec3.zero()
        self._is_fast = False
        self._is_slow = False

    @property
    def move_speed(self) -> float:
        """Get base movement speed."""
        return self._move_speed

    @move_speed.setter
    def move_speed(self, value: float) -> None:
        """Set base movement speed."""
        self._move_speed = max(1.0, value)

    def set_speed_modifiers(self, is_fast: bool = False, is_slow: bool = False) -> None:
        """Set speed modifier states."""
        self._is_fast = is_fast
        self._is_slow = is_slow

    def set_input(self, forward: float, right: float, up: float) -> None:
        """
        Set movement input direction.

        Args:
            forward: Forward/backward input (-1 to 1)
            right: Right/left input (-1 to 1)
            up: Up/down input (-1 to 1)
        """
        self._input_direction = Vec3(right, up, -forward)

    def add_look_input(self, delta_yaw: float, delta_pitch: float) -> None:
        """Add mouse look input."""
        self._yaw = (self._yaw + delta_yaw * self._sensitivity) % 360.0
        self._pitch = max(MIN_PITCH_ANGLE, min(MAX_PITCH_ANGLE, self._pitch - delta_pitch * self._sensitivity))
        self._update_rotation_from_angles()

    def _update_rotation_from_angles(self) -> None:
        """Update rotation from pitch/yaw angles."""
        yaw_rad = self._yaw * DEG_TO_RAD
        pitch_rad = self._pitch * DEG_TO_RAD
        self._rotation = Quat.from_euler(pitch_rad, yaw_rad, 0.0)
        self._view_matrix_dirty = True

    def update(self, delta_time: float) -> None:
        """Update free camera."""
        if not self._enabled:
            return

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Calculate effective speed
        speed = self._move_speed
        if self._is_fast:
            speed *= self._fast_multiplier
        elif self._is_slow:
            speed *= self._slow_multiplier

        # Transform input to world space using camera orientation
        if self._input_direction.length_squared() > CAMERA_EPSILON:
            # Normalize input
            move_dir = self._input_direction.normalized()

            # Apply camera rotation to movement direction
            world_move = self._rotation.rotate_vector(move_dir)

            # Apply movement
            self._position = self._position + world_move * speed * delta_time
            self._view_matrix_dirty = True


class CinematicController(BaseCameraController):
    """
    Cinematic camera with keyframe-based animation and timeline control.

    Features:
    - Keyframe-based position and rotation
    - Timeline with play/pause/seek
    - Multiple easing curves
    - Event callbacks at specific times
    """

    __slots__ = (
        "_keyframes",
        "_timeline_duration",
        "_current_time",
        "_playback_speed",
        "_is_playing",
        "_loop",
        "_events",
    )

    def __init__(self) -> None:
        """Initialize cinematic camera controller."""
        super().__init__(mode=CameraMode.CINEMATIC)
        self._keyframes: List[CinematicKeyframe] = []
        self._timeline_duration = 0.0
        self._current_time = 0.0
        self._playback_speed = 1.0
        self._is_playing = False
        self._loop = False
        self._events: Dict[float, List[Callable[[], None]]] = {}

    @property
    def duration(self) -> float:
        """Get timeline duration."""
        return self._timeline_duration

    @property
    def current_time(self) -> float:
        """Get current playback time."""
        return self._current_time

    @property
    def is_playing(self) -> bool:
        """Check if playing."""
        return self._is_playing

    @property
    def playback_speed(self) -> float:
        """Get playback speed multiplier."""
        return self._playback_speed

    @playback_speed.setter
    def playback_speed(self, value: float) -> None:
        """Set playback speed (can be negative for reverse)."""
        self._playback_speed = value

    def add_keyframe(
        self,
        time: float,
        position: Vec3,
        rotation: Quat,
        fov: float = DEFAULT_FOV,
        easing: str = "linear",
    ) -> None:
        """Add a keyframe at the specified time."""
        keyframe = CinematicKeyframe(
            time=time,
            position=position,
            rotation=rotation,
            fov=fov,
            easing=easing,
        )
        self._keyframes.append(keyframe)
        self._keyframes.sort(key=lambda k: k.time)

        # Update duration
        if time > self._timeline_duration:
            self._timeline_duration = time

    def clear_keyframes(self) -> None:
        """Remove all keyframes."""
        self._keyframes.clear()
        self._timeline_duration = 0.0

    def add_event(self, time: float, callback: Callable[[], None]) -> None:
        """Add an event callback at a specific time."""
        if time not in self._events:
            self._events[time] = []
        self._events[time].append(callback)

    def play(self) -> None:
        """Start playback."""
        self._is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        self._is_playing = False

    def stop(self) -> None:
        """Stop playback and reset to start."""
        self._is_playing = False
        self._current_time = 0.0

    def seek(self, time: float) -> None:
        """Seek to specific time."""
        self._current_time = max(0.0, min(self._timeline_duration, time))
        self._evaluate_at_time(self._current_time)

    def _find_keyframes(self, time: float) -> tuple[Optional[CinematicKeyframe], Optional[CinematicKeyframe]]:
        """Find surrounding keyframes for interpolation."""
        if not self._keyframes:
            return None, None

        prev_kf = None
        next_kf = None

        for kf in self._keyframes:
            if kf.time <= time:
                prev_kf = kf
            elif next_kf is None:
                next_kf = kf
                break

        return prev_kf, next_kf

    def _evaluate_at_time(self, time: float) -> None:
        """Evaluate camera state at given time."""
        prev_kf, next_kf = self._find_keyframes(time)

        if prev_kf is None and next_kf is None:
            return

        if prev_kf is None:
            # Before first keyframe
            self._position = next_kf.position
            self._rotation = next_kf.rotation
            self._fov = next_kf.fov
        elif next_kf is None:
            # After last keyframe
            self._position = prev_kf.position
            self._rotation = prev_kf.rotation
            self._fov = prev_kf.fov
        else:
            # Interpolate between keyframes
            t = (time - prev_kf.time) / (next_kf.time - prev_kf.time) if next_kf.time != prev_kf.time else 0.0
            t = _apply_easing(t, next_kf.easing)

            self._position = prev_kf.position.lerp(next_kf.position, t)
            self._rotation = prev_kf.rotation.slerp(next_kf.rotation, t)
            self._fov = prev_kf.fov + (next_kf.fov - prev_kf.fov) * t

        self._view_matrix_dirty = True
        self._projection_matrix_dirty = True

    def _check_events(self, old_time: float, new_time: float) -> None:
        """Check and fire events between old and new time."""
        for event_time, callbacks in self._events.items():
            if old_time < event_time <= new_time or new_time < event_time <= old_time:
                for callback in callbacks:
                    callback()

    def update(self, delta_time: float) -> None:
        """Update cinematic camera."""
        if not self._enabled or not self._is_playing:
            return

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        old_time = self._current_time
        self._current_time += delta_time * self._playback_speed

        # Handle looping or stopping
        if self._current_time >= self._timeline_duration:
            if self._loop:
                self._current_time = self._current_time % self._timeline_duration
            else:
                self._current_time = self._timeline_duration
                self._is_playing = False
        elif self._current_time < 0:
            if self._loop:
                self._current_time = self._timeline_duration + (self._current_time % self._timeline_duration)
            else:
                self._current_time = 0.0
                self._is_playing = False

        # Check events
        self._check_events(old_time, self._current_time)

        # Evaluate position
        self._evaluate_at_time(self._current_time)


@dataclass(slots=True)
class CinematicKeyframe:
    """Keyframe data for cinematic camera."""
    time: float
    position: Vec3
    rotation: Quat
    fov: float = DEFAULT_FOV
    easing: str = "linear"


class TopDownController(BaseCameraController):
    """
    Top-down camera with fixed angle, zoom, and pan limits.

    Features:
    - Fixed downward angle
    - Zoom control
    - Pan limits for bounded areas
    - Smooth following
    """

    __slots__ = (
        "_height",
        "_target_height",
        "_min_height",
        "_max_height",
        "_pan_limits_min",
        "_pan_limits_max",
        "_follow_speed",
        "_use_pan_limits",
    )

    def __init__(
        self,
        height: float = 500.0,
        target_position: Optional[Vec3] = None,
    ) -> None:
        """
        Initialize top-down camera controller.

        Args:
            height: Camera height above target
            target_position: Initial target position
        """
        super().__init__(mode=CameraMode.TOP_DOWN)
        self._height = height
        self._target_height = height
        self._min_height = 100.0
        self._max_height = 2000.0
        self._pan_limits_min = Vec3(-1000, 0, -1000)
        self._pan_limits_max = Vec3(1000, 0, 1000)
        self._follow_speed = DEFAULT_CAMERA_LAG_SPEED
        self._use_pan_limits = False

        if target_position is not None:
            self._target_position = target_position

        # Set fixed rotation (looking straight down)
        pitch_rad = TOP_DOWN_PITCH * DEG_TO_RAD
        self._rotation = Quat.from_euler(pitch_rad, 0.0, 0.0)

    @property
    def height(self) -> float:
        """Get camera height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set target height (will interpolate)."""
        self._target_height = max(self._min_height, min(self._max_height, value))

    def set_height_limits(self, min_height: float, max_height: float) -> None:
        """Set zoom height limits."""
        self._min_height = max(10.0, min_height)
        self._max_height = max(self._min_height, max_height)

    def set_pan_limits(self, min_pos: Vec3, max_pos: Vec3) -> None:
        """Set pan limits."""
        self._pan_limits_min = min_pos
        self._pan_limits_max = max_pos
        self._use_pan_limits = True

    def disable_pan_limits(self) -> None:
        """Disable pan limits."""
        self._use_pan_limits = False

    def zoom(self, delta: float) -> None:
        """Adjust zoom (height) by delta."""
        self.height = self._target_height - delta * 10.0

    def update(self, delta_time: float) -> None:
        """Update top-down camera."""
        if not self._enabled:
            return

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Interpolate height
        height_diff = self._target_height - self._height
        if abs(height_diff) > 0.01:
            self._height += height_diff * min(1.0, delta_time * DISTANCE_INTERP_SPEED)

        # Get target position
        target_pos = self.target_position

        # Apply pan limits
        if self._use_pan_limits:
            target_pos = Vec3(
                max(self._pan_limits_min.x, min(self._pan_limits_max.x, target_pos.x)),
                target_pos.y,
                max(self._pan_limits_min.z, min(self._pan_limits_max.z, target_pos.z)),
            )

        # Calculate desired position
        desired_pos = Vec3(target_pos.x, target_pos.y + self._height, target_pos.z)

        # Smooth follow
        follow_factor = 1.0 - math.exp(-self._follow_speed * delta_time)
        self._position = self._position.lerp(desired_pos, follow_factor)
        self._view_matrix_dirty = True


class IsometricController(BaseCameraController):
    """
    Isometric camera with 45-degree angle and rotation snap.

    Features:
    - Fixed isometric angle
    - 45-degree rotation snapping
    - Zoom control
    - Smooth transitions
    """

    __slots__ = (
        "_distance",
        "_target_distance",
        "_rotation_index",  # 0-7 for 8 possible 45-degree angles
        "_target_rotation_index",
        "_follow_speed",
        "_rotation_transition",
    )

    def __init__(
        self,
        distance: float = 500.0,
        target_position: Optional[Vec3] = None,
        rotation_index: int = 0,
    ) -> None:
        """
        Initialize isometric camera controller.

        Args:
            distance: Camera distance from target
            target_position: Initial target position
            rotation_index: Initial rotation (0-7 for 8 angles)
        """
        super().__init__(mode=CameraMode.ISOMETRIC)
        self._distance = distance
        self._target_distance = distance
        self._rotation_index = rotation_index % 8
        self._target_rotation_index = self._rotation_index
        self._follow_speed = DEFAULT_CAMERA_LAG_SPEED
        self._rotation_transition = 0.0

        if target_position is not None:
            self._target_position = target_position

    @property
    def distance(self) -> float:
        """Get camera distance."""
        return self._distance

    @distance.setter
    def distance(self, value: float) -> None:
        """Set target distance."""
        self._target_distance = max(MIN_ORBIT_DISTANCE, min(MAX_ORBIT_DISTANCE, value))

    @property
    def rotation_index(self) -> int:
        """Get current rotation index (0-7)."""
        return self._rotation_index

    def rotate_clockwise(self) -> None:
        """Rotate camera 45 degrees clockwise."""
        self._target_rotation_index = (self._rotation_index + 1) % 8

    def rotate_counter_clockwise(self) -> None:
        """Rotate camera 45 degrees counter-clockwise."""
        self._target_rotation_index = (self._rotation_index - 1) % 8

    def set_rotation_index(self, index: int) -> None:
        """Set specific rotation index (0-7)."""
        self._target_rotation_index = index % 8

    def zoom(self, delta: float) -> None:
        """Adjust zoom by delta."""
        self.distance = self._target_distance - delta * 10.0

    def _get_rotation_for_index(self, index: int) -> float:
        """Get yaw angle for rotation index."""
        return index * ISOMETRIC_ROTATION_SNAP

    def update(self, delta_time: float) -> None:
        """Update isometric camera."""
        if not self._enabled:
            return

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Interpolate distance
        dist_diff = self._target_distance - self._distance
        if abs(dist_diff) > 0.01:
            self._distance += dist_diff * min(1.0, delta_time * DISTANCE_INTERP_SPEED)

        # Handle rotation transition
        if self._rotation_index != self._target_rotation_index:
            self._rotation_transition += delta_time * ISOMETRIC_ROTATION_TRANSITION_SPEED
            if self._rotation_transition >= 1.0:
                self._rotation_index = self._target_rotation_index
                self._rotation_transition = 0.0

        # Calculate yaw angle
        current_yaw = self._get_rotation_for_index(self._rotation_index)
        if self._rotation_index != self._target_rotation_index:
            target_yaw = self._get_rotation_for_index(self._target_rotation_index)
            # Handle wrap-around
            if abs(target_yaw - current_yaw) > 180:
                if target_yaw > current_yaw:
                    current_yaw += 360
                else:
                    target_yaw += 360
            current_yaw = current_yaw + (target_yaw - current_yaw) * self._rotation_transition

        # Calculate position
        yaw_rad = current_yaw * DEG_TO_RAD
        pitch_rad = ISOMETRIC_PITCH * DEG_TO_RAD

        cos_pitch = math.cos(pitch_rad)
        offset = Vec3(
            math.sin(yaw_rad) * cos_pitch * self._distance,
            -math.sin(pitch_rad) * self._distance,
            math.cos(yaw_rad) * cos_pitch * self._distance
        )

        target_pos = self.target_position
        desired_pos = target_pos + offset

        # Smooth follow
        follow_factor = 1.0 - math.exp(-self._follow_speed * delta_time)
        self._position = self._position.lerp(desired_pos, follow_factor)
        self._view_matrix_dirty = True

        # Update rotation to look at target
        self._rotation = Quat.from_euler(pitch_rad, yaw_rad, 0.0)


def _apply_easing(t: float, easing: str) -> float:
    """Apply easing function to interpolation parameter."""
    if easing == "linear":
        return t
    elif easing == "ease_in":
        return t * t
    elif easing == "ease_out":
        return t * (2.0 - t)
    elif easing == "ease_in_out":
        if t < 0.5:
            return 2.0 * t * t
        return -1.0 + (4.0 - 2.0 * t) * t
    elif easing == "cubic_in":
        return t * t * t
    elif easing == "cubic_out":
        u = t - 1.0
        return u * u * u + 1.0
    elif easing == "cubic_in_out":
        if t < 0.5:
            return 4.0 * t * t * t
        u = 2.0 * t - 2.0
        return 0.5 * u * u * u + 1.0
    else:
        return t


__all__ = [
    "CameraMode",
    "CameraState",
    "BaseCameraController",
    "FirstPersonController",
    "ThirdPersonController",
    "OrbitController",
    "FollowController",
    "FreeController",
    "CinematicController",
    "CinematicKeyframe",
    "TopDownController",
    "IsometricController",
]
