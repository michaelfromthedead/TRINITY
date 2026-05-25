"""
Eye Animation Module.

Provides eye movement, tracking, blinking, and pupil control
for realistic facial animation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Tuple


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (x, y, z, w)


# =============================================================================
# Eye Configuration
# =============================================================================


@dataclass
class EyeLimits:
    """
    Rotation limits for eye movement.

    Attributes:
        max_yaw: Maximum horizontal rotation (degrees)
        max_pitch_up: Maximum upward rotation (degrees)
        max_pitch_down: Maximum downward rotation (degrees)
        max_vergence: Maximum convergence angle for near objects (degrees)
    """
    max_yaw: float = 35.0
    max_pitch_up: float = 25.0
    max_pitch_down: float = 30.0
    max_vergence: float = 15.0

    def clamp_rotation(self, yaw: float, pitch: float) -> Tuple[float, float]:
        """
        Clamp rotation to limits.

        Args:
            yaw: Horizontal rotation (degrees)
            pitch: Vertical rotation (degrees)

        Returns:
            Clamped (yaw, pitch)
        """
        clamped_yaw = max(-self.max_yaw, min(self.max_yaw, yaw))
        if pitch > 0:
            clamped_pitch = min(self.max_pitch_up, pitch)
        else:
            clamped_pitch = max(-self.max_pitch_down, pitch)
        return (clamped_yaw, clamped_pitch)


@dataclass
class BlinkSettings:
    """
    Settings for automatic blinking.

    Attributes:
        min_interval: Minimum time between blinks (seconds)
        max_interval: Maximum time between blinks (seconds)
        blink_duration: Duration of a single blink (seconds)
        half_blink_chance: Chance of a half-blink (0-1)
        double_blink_chance: Chance of double-blink (0-1)
    """
    min_interval: float = 2.0
    max_interval: float = 6.0
    blink_duration: float = 0.15
    half_blink_chance: float = 0.1
    double_blink_chance: float = 0.15


@dataclass
class SaccadeSettings:
    """
    Settings for saccadic eye movements.

    Attributes:
        micro_saccade_interval: Time between micro-saccades (seconds)
        micro_saccade_magnitude: Magnitude of micro-saccades (degrees)
        saccade_speed: Speed of saccadic movements (degrees/second)
        fixation_duration: How long to fixate before saccade (seconds)
    """
    micro_saccade_interval: float = 0.5
    micro_saccade_magnitude: float = 0.5
    saccade_speed: float = 500.0  # Very fast
    fixation_duration: float = 0.2


@dataclass
class PupilSettings:
    """
    Settings for pupil dilation.

    Attributes:
        base_size: Base pupil size (0-1 normalized)
        min_size: Minimum pupil size
        max_size: Maximum pupil size
        dilation_speed: Speed of dilation changes
        light_response: How much light affects pupil (0-1)
        emotional_response: How much emotion affects pupil (0-1)
    """
    base_size: float = 0.5
    min_size: float = 0.2
    max_size: float = 0.9
    dilation_speed: float = 2.0
    light_response: float = 0.5
    emotional_response: float = 0.3


# =============================================================================
# Eye State
# =============================================================================


class EyeState(Enum):
    """Current state of eye behavior."""
    IDLE = auto()
    TRACKING = auto()
    SACCADE = auto()
    BLINKING = auto()
    FIXATING = auto()


@dataclass
class EyeTransform:
    """
    Transform state for a single eye.

    Attributes:
        yaw: Horizontal rotation (degrees, positive = right)
        pitch: Vertical rotation (degrees, positive = up)
        vergence: Convergence adjustment (degrees)
        blink_weight: Blink weight (0 = open, 1 = closed)
        pupil_size: Pupil dilation (0-1)
    """
    yaw: float = 0.0
    pitch: float = 0.0
    vergence: float = 0.0
    blink_weight: float = 0.0
    pupil_size: float = 0.5

    def to_euler(self) -> Vector3:
        """
        Convert to Euler angles (radians).

        Returns:
            (pitch, yaw, roll) in radians
        """
        return (
            math.radians(self.pitch),
            math.radians(self.yaw + self.vergence),
            0.0,
        )

    def to_quaternion(self) -> Quaternion:
        """
        Convert to quaternion rotation.

        Returns:
            Quaternion (x, y, z, w)
        """
        # Convert degrees to radians
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw + self.vergence)

        # Create quaternion from euler (ZYX order)
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)

        return (
            cy * sp,  # x
            sy * cp,  # y
            -sy * sp,  # z
            cy * cp,  # w
        )


# =============================================================================
# Blink Controller
# =============================================================================


class BlinkController:
    """
    Controller for realistic eye blinking.

    Handles periodic blinks with natural variation.
    """

    def __init__(
        self,
        settings: Optional[BlinkSettings] = None,
        on_blink: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Initialize the blink controller.

        Args:
            settings: Blink settings
            on_blink: Callback when blink occurs (receives blink weight)
        """
        self._settings = settings or BlinkSettings()
        self._on_blink = on_blink

        self._time_to_next_blink: float = self._random_interval()
        self._blink_time: float = 0.0
        self._is_blinking: bool = False
        self._blink_intensity: float = 1.0
        self._double_blink_pending: bool = False
        self._current_weight: float = 0.0

        # External triggers
        self._force_blink: bool = False

    @property
    def current_weight(self) -> float:
        """Get current blink weight (0 = open, 1 = closed)."""
        return self._current_weight

    @property
    def is_blinking(self) -> bool:
        """Check if currently blinking."""
        return self._is_blinking

    def trigger_blink(self, intensity: float = 1.0) -> None:
        """
        Trigger a manual blink.

        Args:
            intensity: Blink intensity (0-1, 1 = full close)
        """
        self._force_blink = True
        self._blink_intensity = max(0.0, min(1.0, intensity))

    def reset(self) -> None:
        """Reset blink state."""
        self._time_to_next_blink = self._random_interval()
        self._blink_time = 0.0
        self._is_blinking = False
        self._current_weight = 0.0
        self._double_blink_pending = False
        self._force_blink = False

    def update(self, dt: float) -> float:
        """
        Update blink state.

        Args:
            dt: Delta time in seconds

        Returns:
            Current blink weight (0 = open, 1 = closed)
        """
        if self._is_blinking:
            # Continue current blink
            self._blink_time += dt
            progress = self._blink_time / self._settings.blink_duration

            if progress >= 1.0:
                # Blink complete
                self._is_blinking = False
                self._blink_time = 0.0
                self._current_weight = 0.0

                if self._double_blink_pending:
                    self._double_blink_pending = False
                    self._is_blinking = True
                else:
                    self._time_to_next_blink = self._random_interval()
            else:
                # Blink curve (quick close, slower open)
                if progress < 0.3:
                    # Closing phase
                    self._current_weight = (progress / 0.3) * self._blink_intensity
                else:
                    # Opening phase
                    self._current_weight = (1.0 - (progress - 0.3) / 0.7) * self._blink_intensity
        else:
            # Check for blink trigger
            if self._force_blink:
                self._force_blink = False
                self._start_blink()
            else:
                self._time_to_next_blink -= dt
                if self._time_to_next_blink <= 0:
                    self._start_blink()

        if self._on_blink and self._is_blinking:
            self._on_blink(self._current_weight)

        return self._current_weight

    def _start_blink(self) -> None:
        """Start a new blink."""
        self._is_blinking = True
        self._blink_time = 0.0

        # Determine blink type
        if random.random() < self._settings.half_blink_chance:
            self._blink_intensity = 0.5 + random.random() * 0.3
        else:
            self._blink_intensity = 0.9 + random.random() * 0.1

        # Check for double blink
        if random.random() < self._settings.double_blink_chance:
            self._double_blink_pending = True

    def _random_interval(self) -> float:
        """Get random interval until next blink."""
        return self._settings.min_interval + random.random() * (
            self._settings.max_interval - self._settings.min_interval
        )


# =============================================================================
# Eye Controller
# =============================================================================


class EyeController:
    """
    Controller for eye movement and tracking.

    Handles look-at targeting, saccades, vergence, and blinking.
    """

    def __init__(
        self,
        eye_limits: Optional[EyeLimits] = None,
        blink_settings: Optional[BlinkSettings] = None,
        saccade_settings: Optional[SaccadeSettings] = None,
        pupil_settings: Optional[PupilSettings] = None,
        eye_separation: float = 0.065,  # ~6.5cm between eyes
        on_state_changed: Optional[Callable[[EyeState], None]] = None,
    ) -> None:
        """
        Initialize the eye controller.

        Args:
            eye_limits: Rotation limits
            blink_settings: Blink behavior settings
            saccade_settings: Saccade behavior settings
            pupil_settings: Pupil dilation settings
            eye_separation: Distance between eyes (meters)
            on_state_changed: Callback when state changes
        """
        self._limits = eye_limits or EyeLimits()
        self._saccade_settings = saccade_settings or SaccadeSettings()
        self._pupil_settings = pupil_settings or PupilSettings()
        self._eye_separation = eye_separation
        self._on_state_changed = on_state_changed

        # Blink controller
        self._blink_controller = BlinkController(blink_settings)

        # Eye positions (local space)
        self._head_position: Vector3 = (0.0, 0.0, 0.0)
        self._head_forward: Vector3 = (0.0, 0.0, 1.0)

        # Current transforms
        self._left_eye = EyeTransform()
        self._right_eye = EyeTransform()

        # Target tracking
        self._look_at_target: Optional[Vector3] = None
        self._look_at_weight: float = 1.0
        self._smooth_speed: float = 10.0

        # Target rotation (what we're blending toward)
        self._target_yaw: float = 0.0
        self._target_pitch: float = 0.0

        # State
        self._state = EyeState.IDLE
        self._time_in_state: float = 0.0

        # Saccades
        self._micro_saccade_timer: float = 0.0
        self._saccade_offset_yaw: float = 0.0
        self._saccade_offset_pitch: float = 0.0

        # Pupil state
        self._target_pupil_size: float = self._pupil_settings.base_size
        self._light_level: float = 0.5  # 0 = dark, 1 = bright
        self._emotional_arousal: float = 0.0  # -1 to 1

        self._dirty = False

    @property
    def left_eye(self) -> EyeTransform:
        """Get left eye transform."""
        return self._left_eye

    @property
    def right_eye(self) -> EyeTransform:
        """Get right eye transform."""
        return self._right_eye

    @property
    def state(self) -> EyeState:
        """Get current eye state."""
        return self._state

    @property
    def look_at_target(self) -> Optional[Vector3]:
        """Get current look-at target."""
        return self._look_at_target

    @property
    def dirty(self) -> bool:
        """Check if state has changed."""
        return self._dirty

    def set_head_transform(
        self,
        position: Vector3,
        forward: Vector3,
    ) -> None:
        """
        Set head position and orientation.

        Args:
            position: Head position in world space
            forward: Head forward direction (normalized)
        """
        self._head_position = position
        self._head_forward = forward

    def look_at(
        self,
        target: Vector3,
        weight: float = 1.0,
        smooth_speed: Optional[float] = None,
    ) -> None:
        """
        Set look-at target.

        Args:
            target: Target position in world space
            weight: Influence weight (0-1)
            smooth_speed: Blend speed (optional)
        """
        self._look_at_target = target
        self._look_at_weight = max(0.0, min(1.0, weight))
        if smooth_speed is not None:
            self._smooth_speed = max(0.1, smooth_speed)
        self._set_state(EyeState.TRACKING)

    def clear_target(self) -> None:
        """Clear look-at target."""
        self._look_at_target = None
        self._set_state(EyeState.IDLE)

    def blink(self, intensity: float = 1.0) -> None:
        """
        Trigger a blink.

        Args:
            intensity: Blink intensity (0-1)
        """
        self._blink_controller.trigger_blink(intensity)

    def set_light_level(self, level: float) -> None:
        """
        Set ambient light level for pupil response.

        Args:
            level: Light level (0 = dark, 1 = bright)
        """
        self._light_level = max(0.0, min(1.0, level))
        self._update_target_pupil_size()

    def set_emotional_arousal(self, arousal: float) -> None:
        """
        Set emotional arousal for pupil response.

        Args:
            arousal: Arousal level (-1 to 1, positive = dilated)
        """
        self._emotional_arousal = max(-1.0, min(1.0, arousal))
        self._update_target_pupil_size()

    def update(self, dt: float) -> Tuple[EyeTransform, EyeTransform]:
        """
        Update eye animation.

        Args:
            dt: Delta time in seconds

        Returns:
            (left_eye, right_eye) transforms
        """
        self._time_in_state += dt

        # Update blink
        blink_weight = self._blink_controller.update(dt)
        self._left_eye.blink_weight = blink_weight
        self._right_eye.blink_weight = blink_weight

        # Update gaze
        if self._look_at_target is not None and not self._blink_controller.is_blinking:
            self._update_tracking(dt)
        else:
            self._update_idle(dt)

        # Update micro-saccades
        self._update_micro_saccades(dt)

        # Apply saccade offsets
        left_yaw = self._target_yaw + self._saccade_offset_yaw
        left_pitch = self._target_pitch + self._saccade_offset_pitch
        right_yaw = self._target_yaw + self._saccade_offset_yaw
        right_pitch = self._target_pitch + self._saccade_offset_pitch

        # Apply limits
        left_yaw, left_pitch = self._limits.clamp_rotation(left_yaw, left_pitch)
        right_yaw, right_pitch = self._limits.clamp_rotation(right_yaw, right_pitch)

        # Smooth blend to target
        blend = min(1.0, self._smooth_speed * dt)
        self._left_eye.yaw += (left_yaw - self._left_eye.yaw) * blend
        self._left_eye.pitch += (left_pitch - self._left_eye.pitch) * blend
        self._right_eye.yaw += (right_yaw - self._right_eye.yaw) * blend
        self._right_eye.pitch += (right_pitch - self._right_eye.pitch) * blend

        # Update vergence
        self._update_vergence()

        # Update pupil
        self._update_pupil(dt)

        self._dirty = True
        return (self._left_eye, self._right_eye)

    def _update_tracking(self, dt: float) -> None:
        """Update gaze when tracking a target."""
        if self._look_at_target is None:
            return

        # Calculate direction to target
        dx = self._look_at_target[0] - self._head_position[0]
        dy = self._look_at_target[1] - self._head_position[1]
        dz = self._look_at_target[2] - self._head_position[2]

        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance < 0.001:
            return

        # Calculate angles relative to head forward
        # Assuming head forward is (0, 0, 1) in local space
        yaw = math.degrees(math.atan2(dx, dz))
        pitch = math.degrees(math.atan2(dy, math.sqrt(dx * dx + dz * dz)))

        # Apply weight
        self._target_yaw = yaw * self._look_at_weight
        self._target_pitch = pitch * self._look_at_weight

    def _update_idle(self, dt: float) -> None:
        """Update gaze when idle (no target)."""
        # Gradually return to center
        self._target_yaw *= (1.0 - dt * 0.5)
        self._target_pitch *= (1.0 - dt * 0.5)

    def _update_micro_saccades(self, dt: float) -> None:
        """Update micro-saccade movements."""
        if self._blink_controller.is_blinking:
            return

        self._micro_saccade_timer += dt

        if self._micro_saccade_timer >= self._saccade_settings.micro_saccade_interval:
            self._micro_saccade_timer = 0.0

            # Random micro-saccade
            magnitude = self._saccade_settings.micro_saccade_magnitude
            self._saccade_offset_yaw = (random.random() - 0.5) * 2.0 * magnitude
            self._saccade_offset_pitch = (random.random() - 0.5) * 2.0 * magnitude

        # Decay saccade offsets
        decay = min(1.0, dt * 5.0)
        self._saccade_offset_yaw *= (1.0 - decay)
        self._saccade_offset_pitch *= (1.0 - decay)

    def _update_vergence(self) -> None:
        """Update eye vergence for distance."""
        if self._look_at_target is None:
            self._left_eye.vergence = 0.0
            self._right_eye.vergence = 0.0
            return

        # Calculate distance to target
        dx = self._look_at_target[0] - self._head_position[0]
        dy = self._look_at_target[1] - self._head_position[1]
        dz = self._look_at_target[2] - self._head_position[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance < 0.1:
            distance = 0.1

        # Calculate vergence angle (eyes converge for near objects)
        # Using small angle approximation
        vergence_angle = math.degrees(math.atan2(self._eye_separation * 0.5, distance))
        vergence_angle = min(vergence_angle, self._limits.max_vergence)

        # Left eye turns right (positive), right eye turns left (negative)
        self._left_eye.vergence = vergence_angle
        self._right_eye.vergence = -vergence_angle

    def _update_pupil(self, dt: float) -> None:
        """Update pupil dilation."""
        # Blend toward target
        blend = min(1.0, self._pupil_settings.dilation_speed * dt)
        current = self._left_eye.pupil_size
        target = self._target_pupil_size

        new_size = current + (target - current) * blend
        new_size = max(self._pupil_settings.min_size, min(self._pupil_settings.max_size, new_size))

        self._left_eye.pupil_size = new_size
        self._right_eye.pupil_size = new_size

    def _update_target_pupil_size(self) -> None:
        """Calculate target pupil size from inputs."""
        base = self._pupil_settings.base_size

        # Light response (constrict in bright light)
        light_effect = (0.5 - self._light_level) * self._pupil_settings.light_response

        # Emotional response (dilate with arousal)
        emotion_effect = self._emotional_arousal * self._pupil_settings.emotional_response

        self._target_pupil_size = base + light_effect + emotion_effect
        self._target_pupil_size = max(
            self._pupil_settings.min_size,
            min(self._pupil_settings.max_size, self._target_pupil_size)
        )

    def _set_state(self, state: EyeState) -> None:
        """Set eye state with callback."""
        if self._state != state:
            self._state = state
            self._time_in_state = 0.0
            if self._on_state_changed:
                self._on_state_changed(state)

    def get_blend_shape_weights(self) -> dict[str, float]:
        """
        Get blend shape weights for eye animation.

        Returns:
            Dictionary of blend shape names to weights
        """
        weights = {}

        # Blink weights
        weights["eyeBlinkLeft"] = self._left_eye.blink_weight
        weights["eyeBlinkRight"] = self._right_eye.blink_weight

        # Gaze weights (normalized to blend shape range)
        # Convert degrees to 0-1 range based on limits
        left_look_right = max(0.0, self._left_eye.yaw / self._limits.max_yaw)
        left_look_left = max(0.0, -self._left_eye.yaw / self._limits.max_yaw)
        left_look_up = max(0.0, self._left_eye.pitch / self._limits.max_pitch_up)
        left_look_down = max(0.0, -self._left_eye.pitch / self._limits.max_pitch_down)

        right_look_right = max(0.0, self._right_eye.yaw / self._limits.max_yaw)
        right_look_left = max(0.0, -self._right_eye.yaw / self._limits.max_yaw)
        right_look_up = max(0.0, self._right_eye.pitch / self._limits.max_pitch_up)
        right_look_down = max(0.0, -self._right_eye.pitch / self._limits.max_pitch_down)

        # ARKit-style blend shapes
        weights["eyeLookInLeft"] = left_look_right  # Left eye looking toward nose
        weights["eyeLookOutLeft"] = left_look_left  # Left eye looking away from nose
        weights["eyeLookUpLeft"] = left_look_up
        weights["eyeLookDownLeft"] = left_look_down

        weights["eyeLookInRight"] = right_look_left  # Right eye looking toward nose
        weights["eyeLookOutRight"] = right_look_right  # Right eye looking away from nose
        weights["eyeLookUpRight"] = right_look_up
        weights["eyeLookDownRight"] = right_look_down

        return weights

    def clear_dirty(self) -> None:
        """Clear the dirty flag."""
        self._dirty = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "state": self._state.name,
            "left_eye": {
                "yaw": self._left_eye.yaw,
                "pitch": self._left_eye.pitch,
                "vergence": self._left_eye.vergence,
                "blink_weight": self._left_eye.blink_weight,
                "pupil_size": self._left_eye.pupil_size,
            },
            "right_eye": {
                "yaw": self._right_eye.yaw,
                "pitch": self._right_eye.pitch,
                "vergence": self._right_eye.vergence,
                "blink_weight": self._right_eye.blink_weight,
                "pupil_size": self._right_eye.pupil_size,
            },
            "look_at_target": self._look_at_target,
        }
