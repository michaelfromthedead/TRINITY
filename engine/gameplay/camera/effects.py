"""Camera Effects - Visual effects including shake, DOF, motion blur, and vignette.

This module provides various camera effects for gameplay feedback and visual polish,
including camera shake, FOV effects, depth of field, motion blur, and vignette.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import random

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4

from engine.gameplay.camera.constants import (
    SHAKE_DECAY_RATE,
    SHAKE_TRAUMA_EXPONENT,
    MAX_TRAUMA,
    DEFAULT_SHAKE_FREQUENCY,
    DEFAULT_SHAKE_AMPLITUDE_TRANSLATION,
    DEFAULT_SHAKE_AMPLITUDE_ROTATION,
    SHAKE_NOISE_OCTAVES,
    SHAKE_NOISE_PERSISTENCE,
    EXPLOSION_SHAKE_TRAUMA,
    DAMAGE_SHAKE_TRAUMA,
    FOOTSTEP_SHAKE_TRAUMA,
    DEFAULT_FOV, MIN_FOV, MAX_FOV,
    FOV_TRANSITION_SPEED,
    DEFAULT_FOCUS_DISTANCE, DEFAULT_APERTURE, DEFAULT_FOCAL_LENGTH,
    MIN_FOCUS_DISTANCE, MAX_FOCUS_DISTANCE, AUTO_FOCUS_SPEED,
    DEFAULT_MOTION_BLUR_INTENSITY, MAX_MOTION_BLUR,
    MOTION_BLUR_VELOCITY_THRESHOLD, MOTION_BLUR_SAMPLES,
    DEFAULT_VIGNETTE_INTENSITY, DEFAULT_VIGNETTE_FEATHER,
    DAMAGE_VIGNETTE_INTENSITY, LOW_HEALTH_VIGNETTE_INTENSITY,
    VIGNETTE_TRANSITION_SPEED,
    DEG_TO_RAD, RAD_TO_DEG,
    CAMERA_EPSILON,
    MIN_DELTA_TIME, MAX_DELTA_TIME,
    # Shake effect constants
    EXPLOSION_SHAKE_DURATION,
    DAMAGE_SHAKE_DURATION,
    FOOTSTEP_SHAKE_DURATION,
    FOOTSTEP_SHAKE_FREQUENCY,
    SHAKE_ROTATION_AMPLITUDE_MULTIPLIER,
    EXPLOSION_SHAKE_BASE_FREQUENCY,
    EXPLOSION_SHAKE_FREQUENCY_VARIATION,
    EXPLOSION_SHAKE_DECAY_RATE,
    EXPLOSION_SHAKE_Z_MULTIPLIER,
    IMPACT_SHAKE_ROTATION_MULTIPLIER,
    CONTINUOUS_SHAKE_FREQ_X, CONTINUOUS_SHAKE_FREQ_Y, CONTINUOUS_SHAKE_FREQ_Z,
    CONTINUOUS_SHAKE_AMP_X, CONTINUOUS_SHAKE_AMP_Y, CONTINUOUS_SHAKE_AMP_Z,
    CONTINUOUS_SHAKE_ROT_FREQ_X, CONTINUOUS_SHAKE_ROT_FREQ_Y, CONTINUOUS_SHAKE_ROT_FREQ_Z,
    CONTINUOUS_SHAKE_ROT_AMP_XY, CONTINUOUS_SHAKE_ROT_AMP_Z,
    # FOV effect constants
    DEFAULT_PUNCH_DECAY,
    FOV_PUNCH_THRESHOLD,
    # Tilt effect constants
    DEFAULT_MAX_TILT,
    DEFAULT_TILT_TRANSITION_SPEED,
    DEFAULT_AUTO_LEVEL_SPEED,
    # DOF constants
    DEFAULT_BOKEH_BLADES,
    MIN_APERTURE,
    MAX_APERTURE,
    # Motion blur constants
    MOTION_BLUR_VELOCITY_NORMALIZATION,
)


class ShakeType(Enum):
    """Types of camera shake patterns."""
    PERLIN = auto()       # Smooth Perlin noise-based shake
    SINE = auto()         # Sinusoidal oscillation
    RANDOM = auto()       # Random noise
    DIRECTIONAL = auto()  # Shake in a specific direction
    EXPLOSION = auto()    # Explosive burst with decay
    IMPACT = auto()       # Single direction impact
    CONTINUOUS = auto()   # Continuous low-level shake


@dataclass(slots=True)
class ShakeSettings:
    """Configuration for camera shake."""
    shake_type: ShakeType = ShakeType.PERLIN
    trauma: float = 0.0
    decay_rate: float = SHAKE_DECAY_RATE
    frequency: float = DEFAULT_SHAKE_FREQUENCY
    amplitude_translation: float = DEFAULT_SHAKE_AMPLITUDE_TRANSLATION
    amplitude_rotation: float = DEFAULT_SHAKE_AMPLITUDE_ROTATION
    direction: Vec3 = field(default_factory=Vec3.zero)  # For directional shake
    max_trauma: float = MAX_TRAUMA
    trauma_exponent: float = SHAKE_TRAUMA_EXPONENT


class CameraShake:
    """
    Camera shake effect with trauma-based intensity.

    Features:
    - Multiple shake patterns (Perlin, sine, random, directional)
    - Trauma accumulation and decay
    - Non-linear intensity curve
    - Position and rotation offsets
    """

    __slots__ = (
        "_settings",
        "_trauma",
        "_time",
        "_noise_offset",
        "_last_offset_pos",
        "_last_offset_rot",
        "_shake_instances",
    )

    def __init__(self, settings: Optional[ShakeSettings] = None) -> None:
        """
        Initialize camera shake.

        Args:
            settings: Shake configuration
        """
        self._settings = settings if settings is not None else ShakeSettings()
        self._trauma = 0.0
        self._time = 0.0
        self._noise_offset = random.uniform(0, 1000)
        self._last_offset_pos = Vec3.zero()
        self._last_offset_rot = Vec3.zero()
        self._shake_instances: List[ShakeInstance] = []

    @property
    def trauma(self) -> float:
        """Get current trauma level (0.0 to 1.0)."""
        return self._trauma

    @property
    def intensity(self) -> float:
        """Get shake intensity (trauma with exponent applied)."""
        return math.pow(self._trauma, self._settings.trauma_exponent)

    @property
    def is_active(self) -> bool:
        """Check if shake is currently active."""
        return self._trauma > CAMERA_EPSILON or len(self._shake_instances) > 0

    @property
    def position_offset(self) -> Vec3:
        """Get current position offset from shake."""
        return self._last_offset_pos

    @property
    def rotation_offset(self) -> Vec3:
        """Get current rotation offset (pitch, yaw, roll in degrees)."""
        return self._last_offset_rot

    def add_trauma(self, amount: float) -> None:
        """
        Add trauma to shake system.

        Args:
            amount: Trauma amount to add (0.0 to 1.0)
        """
        self._trauma = min(self._settings.max_trauma, self._trauma + amount)

    def add_shake(
        self,
        trauma: float,
        shake_type: ShakeType = ShakeType.PERLIN,
        duration: Optional[float] = None,
        frequency: Optional[float] = None,
        amplitude: Optional[float] = None,
        direction: Optional[Vec3] = None,
    ) -> None:
        """
        Add a specific shake instance.

        Args:
            trauma: Initial trauma amount
            shake_type: Type of shake pattern
            duration: Optional duration (uses decay if None)
            frequency: Optional frequency override
            amplitude: Optional amplitude override
            direction: Optional direction for directional shake
        """
        instance = ShakeInstance(
            trauma=min(MAX_TRAUMA, trauma),
            shake_type=shake_type,
            duration=duration,
            frequency=frequency if frequency is not None else self._settings.frequency,
            amplitude=amplitude if amplitude is not None else self._settings.amplitude_translation,
            direction=direction if direction is not None else Vec3.zero(),
            time=0.0,
        )
        self._shake_instances.append(instance)

    def add_explosion_shake(self, intensity: float = 1.0) -> None:
        """Add explosion-style shake."""
        self.add_shake(
            EXPLOSION_SHAKE_TRAUMA * intensity,
            ShakeType.EXPLOSION,
            duration=EXPLOSION_SHAKE_DURATION,
        )

    def add_damage_shake(self, intensity: float = 1.0) -> None:
        """Add damage feedback shake."""
        self.add_shake(
            DAMAGE_SHAKE_TRAUMA * intensity,
            ShakeType.IMPACT,
            duration=DAMAGE_SHAKE_DURATION,
        )

    def add_footstep_shake(self, intensity: float = 1.0) -> None:
        """Add subtle footstep shake."""
        self.add_shake(
            FOOTSTEP_SHAKE_TRAUMA * intensity,
            ShakeType.SINE,
            duration=FOOTSTEP_SHAKE_DURATION,
            frequency=FOOTSTEP_SHAKE_FREQUENCY,
        )

    def clear(self) -> None:
        """Clear all shake."""
        self._trauma = 0.0
        self._shake_instances.clear()
        self._last_offset_pos = Vec3.zero()
        self._last_offset_rot = Vec3.zero()

    def get_offset(self) -> Tuple[Vec3, Vec3]:
        """
        Get current shake offsets.

        Returns:
            Tuple of (position_offset, rotation_offset)
        """
        return self._last_offset_pos, self._last_offset_rot

    def update(self, delta_time: float) -> Tuple[Vec3, Vec3]:
        """
        Update shake and return offsets.

        Args:
            delta_time: Time since last update

        Returns:
            Tuple of (position_offset, rotation_offset)
        """
        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))
        self._time += delta_time

        # Decay trauma
        if self._trauma > 0:
            decay = self._settings.decay_rate * delta_time
            self._trauma = max(0.0, self._trauma - decay)

        # Update shake instances
        self._update_shake_instances(delta_time)

        # Calculate combined offset
        total_pos = Vec3.zero()
        total_rot = Vec3.zero()

        # Base trauma shake
        if self._trauma > CAMERA_EPSILON:
            pos, rot = self._calculate_shake_offset(
                self._settings.shake_type,
                self.intensity,
                self._settings.frequency,
                self._settings.amplitude_translation,
                self._settings.amplitude_rotation,
                self._settings.direction,
            )
            total_pos = total_pos + pos
            total_rot = total_rot + rot

        # Instance shakes
        for instance in self._shake_instances:
            pos, rot = self._calculate_shake_offset(
                instance.shake_type,
                instance.trauma,
                instance.frequency,
                instance.amplitude,
                instance.amplitude * SHAKE_ROTATION_AMPLITUDE_MULTIPLIER,  # Less rotation for instances
                instance.direction,
            )
            total_pos = total_pos + pos
            total_rot = total_rot + rot

        self._last_offset_pos = total_pos
        self._last_offset_rot = total_rot

        return total_pos, total_rot

    def _update_shake_instances(self, delta_time: float) -> None:
        """Update and remove finished shake instances."""
        to_remove = []
        for instance in self._shake_instances:
            instance.time += delta_time

            if instance.duration is not None:
                # Decay over duration
                progress = instance.time / instance.duration
                instance.trauma = instance.initial_trauma * max(0.0, 1.0 - progress)

                if progress >= 1.0:
                    to_remove.append(instance)
            else:
                # Use standard decay
                instance.trauma = max(0.0, instance.trauma - SHAKE_DECAY_RATE * delta_time)
                if instance.trauma < CAMERA_EPSILON:
                    to_remove.append(instance)

        for instance in to_remove:
            self._shake_instances.remove(instance)

    def _calculate_shake_offset(
        self,
        shake_type: ShakeType,
        intensity: float,
        frequency: float,
        amplitude_pos: float,
        amplitude_rot: float,
        direction: Vec3,
    ) -> Tuple[Vec3, Vec3]:
        """Calculate shake offset for given parameters."""
        if intensity < CAMERA_EPSILON:
            return Vec3.zero(), Vec3.zero()

        t = self._time * frequency

        if shake_type == ShakeType.PERLIN:
            return self._perlin_shake(t, intensity, amplitude_pos, amplitude_rot)
        elif shake_type == ShakeType.SINE:
            return self._sine_shake(t, intensity, amplitude_pos, amplitude_rot)
        elif shake_type == ShakeType.RANDOM:
            return self._random_shake(intensity, amplitude_pos, amplitude_rot)
        elif shake_type == ShakeType.DIRECTIONAL:
            return self._directional_shake(t, intensity, amplitude_pos, direction)
        elif shake_type == ShakeType.EXPLOSION:
            return self._explosion_shake(t, intensity, amplitude_pos, amplitude_rot)
        elif shake_type == ShakeType.IMPACT:
            return self._impact_shake(intensity, amplitude_pos, direction)
        elif shake_type == ShakeType.CONTINUOUS:
            return self._continuous_shake(t, intensity, amplitude_pos, amplitude_rot)

        return Vec3.zero(), Vec3.zero()

    def _perlin_shake(
        self,
        t: float,
        intensity: float,
        amp_pos: float,
        amp_rot: float,
    ) -> Tuple[Vec3, Vec3]:
        """Generate Perlin-like noise shake."""
        # Approximate Perlin with layered sine waves
        pos_x = 0.0
        pos_y = 0.0
        pos_z = 0.0
        rot_x = 0.0
        rot_y = 0.0
        rot_z = 0.0

        amp = 1.0
        freq = 1.0
        for _ in range(SHAKE_NOISE_OCTAVES):
            offset = self._noise_offset
            pos_x += math.sin(t * freq + offset) * amp
            pos_y += math.sin(t * freq * 1.1 + offset * 2) * amp
            pos_z += math.sin(t * freq * 0.9 + offset * 3) * amp
            rot_x += math.sin(t * freq * 1.2 + offset * 4) * amp
            rot_y += math.sin(t * freq * 0.8 + offset * 5) * amp
            rot_z += math.sin(t * freq * 1.3 + offset * 6) * amp
            amp *= SHAKE_NOISE_PERSISTENCE
            freq *= 2.0

        pos = Vec3(pos_x, pos_y, pos_z) * intensity * amp_pos
        rot = Vec3(rot_x, rot_y, rot_z) * intensity * amp_rot

        return pos, rot

    def _sine_shake(
        self,
        t: float,
        intensity: float,
        amp_pos: float,
        amp_rot: float,
    ) -> Tuple[Vec3, Vec3]:
        """Generate sinusoidal shake."""
        pos = Vec3(
            math.sin(t * 1.0) * intensity * amp_pos,
            math.sin(t * 1.3 + 0.5) * intensity * amp_pos,
            math.sin(t * 0.7 + 1.0) * intensity * amp_pos * 0.5,
        )
        rot = Vec3(
            math.sin(t * 1.1) * intensity * amp_rot,
            math.sin(t * 0.9 + 0.3) * intensity * amp_rot,
            math.sin(t * 1.2 + 0.7) * intensity * amp_rot * 0.5,
        )
        return pos, rot

    def _random_shake(
        self,
        intensity: float,
        amp_pos: float,
        amp_rot: float,
    ) -> Tuple[Vec3, Vec3]:
        """Generate random shake."""
        pos = Vec3(
            (random.random() * 2 - 1) * intensity * amp_pos,
            (random.random() * 2 - 1) * intensity * amp_pos,
            (random.random() * 2 - 1) * intensity * amp_pos * 0.5,
        )
        rot = Vec3(
            (random.random() * 2 - 1) * intensity * amp_rot,
            (random.random() * 2 - 1) * intensity * amp_rot,
            (random.random() * 2 - 1) * intensity * amp_rot * 0.5,
        )
        return pos, rot

    def _directional_shake(
        self,
        t: float,
        intensity: float,
        amp_pos: float,
        direction: Vec3,
    ) -> Tuple[Vec3, Vec3]:
        """Generate directional shake."""
        if direction.length_squared() < CAMERA_EPSILON:
            direction = Vec3.up()
        direction = direction.normalized()

        oscillation = math.sin(t) * intensity * amp_pos
        pos = direction * oscillation

        # Subtle rotation perpendicular to direction
        rot = Vec3(
            direction.z * math.sin(t * 1.1) * intensity * 0.5,
            0,
            -direction.x * math.sin(t * 0.9) * intensity * 0.5,
        )

        return pos, rot

    def _explosion_shake(
        self,
        t: float,
        intensity: float,
        amp_pos: float,
        amp_rot: float,
    ) -> Tuple[Vec3, Vec3]:
        """Generate explosion-style shake with fast decay."""
        # High frequency, decaying shake
        # Using seeded offset for more consistent behavior
        freq = EXPLOSION_SHAKE_BASE_FREQUENCY + (self._noise_offset % 1.0) * EXPLOSION_SHAKE_FREQUENCY_VARIATION
        decay = math.exp(-t * EXPLOSION_SHAKE_DECAY_RATE)

        pos = Vec3(
            math.sin(t * freq) * intensity * amp_pos * decay,
            math.cos(t * freq * 1.1) * intensity * amp_pos * decay,
            math.sin(t * freq * 0.9) * intensity * amp_pos * EXPLOSION_SHAKE_Z_MULTIPLIER * decay,
        )
        rot = Vec3(
            math.sin(t * freq * 1.2) * intensity * amp_rot * decay,
            math.cos(t * freq * 0.8) * intensity * amp_rot * decay,
            math.sin(t * freq * 1.1) * intensity * amp_rot * SHAKE_ROTATION_AMPLITUDE_MULTIPLIER * decay,
        )

        return pos, rot

    def _impact_shake(
        self,
        intensity: float,
        amp_pos: float,
        direction: Vec3,
    ) -> Tuple[Vec3, Vec3]:
        """Generate single-direction impact shake."""
        if direction.length_squared() < CAMERA_EPSILON:
            direction = Vec3(0, -1, 0)  # Default downward
        direction = direction.normalized()

        pos = direction * intensity * amp_pos
        rot = Vec3(
            direction.z * intensity * IMPACT_SHAKE_ROTATION_MULTIPLIER,
            0,
            -direction.x * intensity * IMPACT_SHAKE_ROTATION_MULTIPLIER,
        )

        return pos, rot

    def _continuous_shake(
        self,
        t: float,
        intensity: float,
        amp_pos: float,
        amp_rot: float,
    ) -> Tuple[Vec3, Vec3]:
        """Generate low-level continuous shake (vehicle, running, etc.)."""
        # Low amplitude, medium frequency
        pos = Vec3(
            math.sin(t * CONTINUOUS_SHAKE_FREQ_X) * intensity * amp_pos * CONTINUOUS_SHAKE_AMP_X,
            math.sin(t * CONTINUOUS_SHAKE_FREQ_Y) * intensity * amp_pos * CONTINUOUS_SHAKE_AMP_Y,
            math.sin(t * CONTINUOUS_SHAKE_FREQ_Z) * intensity * amp_pos * CONTINUOUS_SHAKE_AMP_Z,
        )
        rot = Vec3(
            math.sin(t * CONTINUOUS_SHAKE_ROT_FREQ_X) * intensity * amp_rot * CONTINUOUS_SHAKE_ROT_AMP_XY,
            math.sin(t * CONTINUOUS_SHAKE_ROT_FREQ_Y) * intensity * amp_rot * CONTINUOUS_SHAKE_ROT_AMP_XY,
            math.sin(t * CONTINUOUS_SHAKE_ROT_FREQ_Z) * intensity * amp_rot * CONTINUOUS_SHAKE_ROT_AMP_Z,
        )

        return pos, rot


@dataclass
class ShakeInstance:
    """Individual shake instance."""
    trauma: float
    shake_type: ShakeType
    duration: Optional[float]
    frequency: float
    amplitude: float
    direction: Vec3
    time: float

    def __post_init__(self):
        self.initial_trauma = self.trauma


class ScreenShake:
    """
    Applies camera shake to view matrix.

    Wrapper for CameraShake that applies offsets to camera transform.
    """

    __slots__ = (
        "_shake",
        "_position_scale",
        "_rotation_scale",
    )

    def __init__(
        self,
        shake: Optional[CameraShake] = None,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
    ) -> None:
        """
        Initialize screen shake.

        Args:
            shake: CameraShake instance
            position_scale: Scale for position offset
            rotation_scale: Scale for rotation offset
        """
        self._shake = shake if shake is not None else CameraShake()
        self._position_scale = position_scale
        self._rotation_scale = rotation_scale

    @property
    def shake(self) -> CameraShake:
        """Get underlying CameraShake."""
        return self._shake

    def apply_to_matrix(self, view_matrix: Mat4, delta_time: float) -> Mat4:
        """
        Apply shake to view matrix.

        Args:
            view_matrix: Original view matrix
            delta_time: Time since last update

        Returns:
            Modified view matrix with shake applied
        """
        pos_offset, rot_offset = self._shake.update(delta_time)

        if pos_offset.length_squared() < CAMERA_EPSILON and rot_offset.length_squared() < CAMERA_EPSILON:
            return view_matrix

        # Scale offsets
        pos_offset = pos_offset * self._position_scale
        rot_offset = rot_offset * self._rotation_scale

        # Create shake transformation
        shake_rot = Quat.from_euler(
            rot_offset.x * DEG_TO_RAD,
            rot_offset.y * DEG_TO_RAD,
            rot_offset.z * DEG_TO_RAD,
        )

        shake_mat = shake_rot.to_mat4()
        shake_mat.m[12] = pos_offset.x
        shake_mat.m[13] = pos_offset.y
        shake_mat.m[14] = pos_offset.z

        return shake_mat @ view_matrix


class FOVEffect:
    """
    Field of view effects including punch, zoom, and transitions.

    Features:
    - FOV punch (brief increase/decrease)
    - Smooth zoom transitions
    - Sprint FOV increase
    - ADS FOV decrease
    """

    __slots__ = (
        "_base_fov",
        "_current_fov",
        "_target_fov",
        "_transition_speed",
        "_punch_amount",
        "_punch_decay",
        "_modifier_stack",
    )

    def __init__(
        self,
        base_fov: float = DEFAULT_FOV,
        transition_speed: float = FOV_TRANSITION_SPEED,
    ) -> None:
        """
        Initialize FOV effect.

        Args:
            base_fov: Base field of view
            transition_speed: Degrees per second for transitions
        """
        self._base_fov = max(MIN_FOV, min(MAX_FOV, base_fov))
        self._current_fov = self._base_fov
        self._target_fov = self._base_fov
        self._transition_speed = transition_speed
        self._punch_amount = 0.0
        self._punch_decay = DEFAULT_PUNCH_DECAY
        self._modifier_stack: Dict[str, float] = {}  # name -> offset

    @property
    def base_fov(self) -> float:
        """Get base FOV."""
        return self._base_fov

    @base_fov.setter
    def base_fov(self, value: float) -> None:
        """Set base FOV."""
        self._base_fov = max(MIN_FOV, min(MAX_FOV, value))
        self._recalculate_target()

    @property
    def current_fov(self) -> float:
        """Get current FOV after all effects."""
        return self._current_fov + self._punch_amount

    @property
    def target_fov(self) -> float:
        """Get target FOV."""
        return self._target_fov

    def add_modifier(self, name: str, offset: float) -> None:
        """
        Add FOV modifier (e.g., sprint, ADS).

        Args:
            name: Modifier identifier
            offset: FOV offset (positive = wider, negative = narrower)
        """
        self._modifier_stack[name] = offset
        self._recalculate_target()

    def remove_modifier(self, name: str) -> None:
        """Remove FOV modifier."""
        if name in self._modifier_stack:
            del self._modifier_stack[name]
            self._recalculate_target()

    def clear_modifiers(self) -> None:
        """Remove all FOV modifiers."""
        self._modifier_stack.clear()
        self._recalculate_target()

    def punch(self, amount: float, decay: float = 5.0) -> None:
        """
        Apply FOV punch effect.

        Args:
            amount: Punch amount in degrees
            decay: Decay rate
        """
        self._punch_amount += amount
        self._punch_decay = decay

    def zoom_to(self, fov: float, speed: Optional[float] = None) -> None:
        """
        Smoothly zoom to specific FOV.

        Args:
            fov: Target FOV
            speed: Optional transition speed override
        """
        self._target_fov = max(MIN_FOV, min(MAX_FOV, fov))
        if speed is not None:
            self._transition_speed = speed

    def reset(self) -> None:
        """Reset to base FOV."""
        self._target_fov = self._base_fov
        self._punch_amount = 0.0

    def _recalculate_target(self) -> None:
        """Recalculate target FOV from modifiers."""
        total_offset = sum(self._modifier_stack.values())
        self._target_fov = max(MIN_FOV, min(MAX_FOV, self._base_fov + total_offset))

    def update(self, delta_time: float) -> float:
        """
        Update FOV effect.

        Args:
            delta_time: Time since last update

        Returns:
            Current FOV value
        """
        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Update punch decay
        if abs(self._punch_amount) > CAMERA_EPSILON:
            self._punch_amount *= math.exp(-self._punch_decay * delta_time)
            if abs(self._punch_amount) < FOV_PUNCH_THRESHOLD:
                self._punch_amount = 0.0

        # Interpolate to target
        fov_diff = self._target_fov - self._current_fov
        if abs(fov_diff) > CAMERA_EPSILON:
            max_change = self._transition_speed * delta_time
            if abs(fov_diff) <= max_change:
                self._current_fov = self._target_fov
            else:
                self._current_fov += math.copysign(max_change, fov_diff)

        return self.current_fov


class TiltEffect:
    """
    Camera tilt/roll effects (dutch angle).

    Features:
    - Smooth tilt transitions
    - Auto-level correction
    - Tilt limits
    """

    __slots__ = (
        "_current_tilt",
        "_target_tilt",
        "_max_tilt",
        "_transition_speed",
        "_auto_level",
        "_auto_level_speed",
    )

    def __init__(
        self,
        max_tilt: float = DEFAULT_MAX_TILT,
        transition_speed: float = DEFAULT_TILT_TRANSITION_SPEED,
    ) -> None:
        """
        Initialize tilt effect.

        Args:
            max_tilt: Maximum tilt angle in degrees
            transition_speed: Degrees per second
        """
        self._current_tilt = 0.0
        self._target_tilt = 0.0
        self._max_tilt = max_tilt
        self._transition_speed = transition_speed
        self._auto_level = True
        self._auto_level_speed = DEFAULT_AUTO_LEVEL_SPEED

    @property
    def current_tilt(self) -> float:
        """Get current tilt angle in degrees."""
        return self._current_tilt

    @property
    def auto_level(self) -> bool:
        """Check if auto-level is enabled."""
        return self._auto_level

    @auto_level.setter
    def auto_level(self, value: bool) -> None:
        """Enable/disable auto-level."""
        self._auto_level = value

    def set_tilt(self, angle: float) -> None:
        """Set target tilt angle."""
        self._target_tilt = max(-self._max_tilt, min(self._max_tilt, angle))
        self._auto_level = False

    def add_tilt(self, delta: float) -> None:
        """Add to current tilt."""
        self.set_tilt(self._target_tilt + delta)

    def reset(self) -> None:
        """Reset tilt to zero."""
        self._target_tilt = 0.0
        self._auto_level = True

    def update(self, delta_time: float) -> float:
        """
        Update tilt effect.

        Args:
            delta_time: Time since last update

        Returns:
            Current tilt angle in degrees
        """
        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Auto-level
        if self._auto_level and abs(self._target_tilt) > CAMERA_EPSILON:
            level_amount = self._auto_level_speed * delta_time
            if abs(self._target_tilt) <= level_amount:
                self._target_tilt = 0.0
            else:
                self._target_tilt -= math.copysign(level_amount, self._target_tilt)

        # Interpolate
        tilt_diff = self._target_tilt - self._current_tilt
        if abs(tilt_diff) > CAMERA_EPSILON:
            max_change = self._transition_speed * delta_time
            if abs(tilt_diff) <= max_change:
                self._current_tilt = self._target_tilt
            else:
                self._current_tilt += math.copysign(max_change, tilt_diff)

        return self._current_tilt


@dataclass(slots=True)
class DOFSettings:
    """Depth of field settings."""
    enabled: bool = False
    focus_distance: float = DEFAULT_FOCUS_DISTANCE
    aperture: float = DEFAULT_APERTURE
    focal_length: float = DEFAULT_FOCAL_LENGTH
    auto_focus: bool = False
    auto_focus_speed: float = AUTO_FOCUS_SPEED
    near_blur_size: float = 1.0
    far_blur_size: float = 1.0
    bokeh_shape: int = DEFAULT_BOKEH_BLADES  # Number of aperture blades


class DOFEffect:
    """
    Depth of field effect.

    Features:
    - Configurable focus distance and aperture
    - Auto-focus with raycast
    - Smooth focus transitions
    - Bokeh simulation parameters
    """

    __slots__ = (
        "_settings",
        "_current_focus",
        "_target_focus",
        "_auto_focus_target",
    )

    def __init__(self, settings: Optional[DOFSettings] = None) -> None:
        """
        Initialize DOF effect.

        Args:
            settings: DOF configuration
        """
        self._settings = settings if settings is not None else DOFSettings()
        self._current_focus = self._settings.focus_distance
        self._target_focus = self._settings.focus_distance
        self._auto_focus_target: Optional[Vec3] = None

    @property
    def settings(self) -> DOFSettings:
        """Get DOF settings."""
        return self._settings

    @property
    def focus_distance(self) -> float:
        """Get current focus distance."""
        return self._current_focus

    @property
    def aperture(self) -> float:
        """Get aperture (f-stop)."""
        return self._settings.aperture

    @aperture.setter
    def aperture(self, value: float) -> None:
        """Set aperture."""
        self._settings.aperture = max(MIN_APERTURE, min(MAX_APERTURE, value))

    def set_focus_distance(self, distance: float) -> None:
        """Set target focus distance."""
        self._target_focus = max(MIN_FOCUS_DISTANCE, min(MAX_FOCUS_DISTANCE, distance))

    def set_auto_focus_target(self, position: Optional[Vec3]) -> None:
        """Set auto-focus target position."""
        self._auto_focus_target = position

    def focus_on_point(self, camera_pos: Vec3, focus_point: Vec3) -> None:
        """Focus on a specific world point."""
        distance = (focus_point - camera_pos).length()
        self.set_focus_distance(distance)

    def update(
        self,
        delta_time: float,
        camera_pos: Optional[Vec3] = None,
    ) -> DOFSettings:
        """
        Update DOF effect.

        Args:
            delta_time: Time since last update
            camera_pos: Optional camera position for auto-focus

        Returns:
            Updated DOF settings
        """
        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Auto-focus
        if self._settings.auto_focus and self._auto_focus_target is not None and camera_pos is not None:
            target_distance = (self._auto_focus_target - camera_pos).length()
            self._target_focus = target_distance

        # Interpolate focus
        focus_diff = self._target_focus - self._current_focus
        if abs(focus_diff) > CAMERA_EPSILON:
            change = self._settings.auto_focus_speed * delta_time
            if abs(focus_diff) <= change:
                self._current_focus = self._target_focus
            else:
                self._current_focus += math.copysign(change, focus_diff) * (abs(focus_diff) / 100.0)

        self._settings.focus_distance = self._current_focus
        return self._settings


@dataclass(slots=True)
class MotionBlurSettings:
    """Motion blur settings."""
    enabled: bool = False
    intensity: float = DEFAULT_MOTION_BLUR_INTENSITY
    samples: int = MOTION_BLUR_SAMPLES
    velocity_threshold: float = MOTION_BLUR_VELOCITY_THRESHOLD
    max_blur: float = MAX_MOTION_BLUR
    object_blur: bool = True
    camera_blur: bool = True


class MotionBlur:
    """
    Motion blur effect based on camera/object velocity.

    Features:
    - Velocity-based blur intensity
    - Configurable sample count
    - Separate camera and object blur
    """

    __slots__ = (
        "_settings",
        "_last_position",
        "_last_rotation",
        "_current_velocity",
        "_current_angular_velocity",
        "_blur_amount",
    )

    def __init__(self, settings: Optional[MotionBlurSettings] = None) -> None:
        """
        Initialize motion blur.

        Args:
            settings: Motion blur configuration
        """
        self._settings = settings if settings is not None else MotionBlurSettings()
        self._last_position = Vec3.zero()
        self._last_rotation = Quat.identity()
        self._current_velocity = Vec3.zero()
        self._current_angular_velocity = 0.0
        self._blur_amount = 0.0

    @property
    def settings(self) -> MotionBlurSettings:
        """Get motion blur settings."""
        return self._settings

    @property
    def blur_amount(self) -> float:
        """Get current blur amount (0.0 to 1.0)."""
        return self._blur_amount

    @property
    def velocity(self) -> Vec3:
        """Get current velocity."""
        return self._current_velocity

    def update(
        self,
        delta_time: float,
        position: Vec3,
        rotation: Quat,
    ) -> MotionBlurSettings:
        """
        Update motion blur based on camera movement.

        Args:
            delta_time: Time since last update
            position: Current camera position
            rotation: Current camera rotation

        Returns:
            Updated motion blur settings
        """
        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Calculate velocity
        self._current_velocity = (position - self._last_position) / delta_time
        self._last_position = position

        # Calculate angular velocity (simplified)
        rot_diff = self._last_rotation.inverse() * rotation
        angle = 2.0 * math.acos(min(1.0, abs(rot_diff.w)))
        self._current_angular_velocity = angle / delta_time
        self._last_rotation = rotation

        # Calculate blur amount
        linear_speed = self._current_velocity.length()
        angular_speed = self._current_angular_velocity * RAD_TO_DEG

        # Combine velocities
        combined_speed = linear_speed + angular_speed * 0.1

        if combined_speed > self._settings.velocity_threshold:
            raw_blur = (combined_speed - self._settings.velocity_threshold) / MOTION_BLUR_VELOCITY_NORMALIZATION
            self._blur_amount = min(
                self._settings.max_blur,
                raw_blur * self._settings.intensity
            )
        else:
            self._blur_amount = 0.0

        return self._settings


@dataclass(slots=True)
class VignetteSettings:
    """Vignette effect settings."""
    enabled: bool = False
    intensity: float = DEFAULT_VIGNETTE_INTENSITY
    feather: float = DEFAULT_VIGNETTE_FEATHER
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # RGB
    roundness: float = 1.0  # 1.0 = circular, 0.0 = rectangular


class VignetteEffect:
    """
    Vignette effect with color and intensity control.

    Features:
    - Configurable intensity and feather
    - Color tinting
    - Smooth transitions
    - Presets for damage/low health
    """

    __slots__ = (
        "_settings",
        "_current_intensity",
        "_target_intensity",
        "_current_color",
        "_target_color",
        "_transition_speed",
    )

    def __init__(self, settings: Optional[VignetteSettings] = None) -> None:
        """
        Initialize vignette effect.

        Args:
            settings: Vignette configuration
        """
        self._settings = settings if settings is not None else VignetteSettings()
        self._current_intensity = self._settings.intensity
        self._target_intensity = self._settings.intensity
        self._current_color = self._settings.color
        self._target_color = self._settings.color
        self._transition_speed = VIGNETTE_TRANSITION_SPEED

    @property
    def settings(self) -> VignetteSettings:
        """Get vignette settings."""
        return self._settings

    @property
    def intensity(self) -> float:
        """Get current intensity."""
        return self._current_intensity

    @property
    def color(self) -> Tuple[float, float, float]:
        """Get current color."""
        return self._current_color

    def set_intensity(self, intensity: float, immediate: bool = False) -> None:
        """Set target intensity."""
        self._target_intensity = max(0.0, min(1.0, intensity))
        if immediate:
            self._current_intensity = self._target_intensity

    def set_color(
        self,
        color: Tuple[float, float, float],
        immediate: bool = False,
    ) -> None:
        """Set target color."""
        self._target_color = color
        if immediate:
            self._current_color = color

    def apply_damage_vignette(self) -> None:
        """Apply damage feedback vignette."""
        self._target_intensity = DAMAGE_VIGNETTE_INTENSITY
        self._target_color = (0.5, 0.0, 0.0)  # Red tint

    def apply_low_health_vignette(self) -> None:
        """Apply low health warning vignette."""
        self._target_intensity = LOW_HEALTH_VIGNETTE_INTENSITY
        self._target_color = (0.3, 0.0, 0.0)

    def clear(self) -> None:
        """Clear vignette effect."""
        self._target_intensity = 0.0
        self._target_color = (0.0, 0.0, 0.0)

    def update(self, delta_time: float) -> VignetteSettings:
        """
        Update vignette effect.

        Args:
            delta_time: Time since last update

        Returns:
            Updated vignette settings
        """
        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Interpolate intensity
        intensity_diff = self._target_intensity - self._current_intensity
        if abs(intensity_diff) > CAMERA_EPSILON:
            change = self._transition_speed * delta_time
            if abs(intensity_diff) <= change:
                self._current_intensity = self._target_intensity
            else:
                self._current_intensity += math.copysign(change, intensity_diff)

        # Interpolate color
        color_factor = min(1.0, self._transition_speed * delta_time)
        self._current_color = (
            self._current_color[0] + (self._target_color[0] - self._current_color[0]) * color_factor,
            self._current_color[1] + (self._target_color[1] - self._current_color[1]) * color_factor,
            self._current_color[2] + (self._target_color[2] - self._current_color[2]) * color_factor,
        )

        # Update settings
        self._settings.intensity = self._current_intensity
        self._settings.color = self._current_color

        return self._settings


class CameraEffectsManager:
    """
    Manages all camera effects in one place.

    Provides unified interface for shake, FOV, DOF, motion blur, and vignette.
    """

    __slots__ = (
        "_shake",
        "_screen_shake",
        "_fov_effect",
        "_tilt_effect",
        "_dof_effect",
        "_motion_blur",
        "_vignette",
        "_enabled",
    )

    def __init__(self) -> None:
        """Initialize camera effects manager."""
        self._shake = CameraShake()
        self._screen_shake = ScreenShake(self._shake)
        self._fov_effect = FOVEffect()
        self._tilt_effect = TiltEffect()
        self._dof_effect = DOFEffect()
        self._motion_blur = MotionBlur()
        self._vignette = VignetteEffect()
        self._enabled = True

    @property
    def shake(self) -> CameraShake:
        """Get camera shake."""
        return self._shake

    @property
    def screen_shake(self) -> ScreenShake:
        """Get screen shake."""
        return self._screen_shake

    @property
    def fov(self) -> FOVEffect:
        """Get FOV effect."""
        return self._fov_effect

    @property
    def tilt(self) -> TiltEffect:
        """Get tilt effect."""
        return self._tilt_effect

    @property
    def dof(self) -> DOFEffect:
        """Get DOF effect."""
        return self._dof_effect

    @property
    def motion_blur(self) -> MotionBlur:
        """Get motion blur."""
        return self._motion_blur

    @property
    def vignette(self) -> VignetteEffect:
        """Get vignette effect."""
        return self._vignette

    @property
    def enabled(self) -> bool:
        """Check if effects are enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable/disable all effects."""
        self._enabled = value

    def update(
        self,
        delta_time: float,
        camera_pos: Vec3,
        camera_rot: Quat,
    ) -> None:
        """
        Update all effects.

        Args:
            delta_time: Time since last update
            camera_pos: Current camera position
            camera_rot: Current camera rotation
        """
        if not self._enabled:
            return

        self._shake.update(delta_time)
        self._fov_effect.update(delta_time)
        self._tilt_effect.update(delta_time)
        self._dof_effect.update(delta_time, camera_pos)
        self._motion_blur.update(delta_time, camera_pos, camera_rot)
        self._vignette.update(delta_time)

    def reset_all(self) -> None:
        """Reset all effects to defaults."""
        self._shake.clear()
        self._fov_effect.reset()
        self._tilt_effect.reset()
        self._vignette.clear()


__all__ = [
    "ShakeType",
    "ShakeSettings",
    "CameraShake",
    "ShakeInstance",
    "ScreenShake",
    "FOVEffect",
    "TiltEffect",
    "DOFSettings",
    "DOFEffect",
    "MotionBlurSettings",
    "MotionBlur",
    "VignetteSettings",
    "VignetteEffect",
    "CameraEffectsManager",
]
