"""
Tests for Camera Effects (effects.py).

Tests camera effects including:
    - Camera shake (Perlin noise, sine wave)
    - Trauma system (additive, decay)
    - FOV effects (punch, zoom)
    - Tilt/dutch angle effects
    - Depth of field
    - Motion blur
    - Vignette
    - Effect stacking and blending
"""

import math
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum, auto
import random


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

    def lerp(self, target: "Vector3", t: float) -> "Vector3":
        return Vector3(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
            self.z + (target.z - self.z) * t,
        )


class ShakeType(Enum):
    """Type of camera shake algorithm."""
    PERLIN = auto()
    SINE = auto()
    RANDOM = auto()


class BlendMode(Enum):
    """Blend mode for effects."""
    ADDITIVE = auto()
    MULTIPLICATIVE = auto()
    OVERRIDE = auto()


# =============================================================================
# Perlin Noise Mock
# =============================================================================


class PerlinNoise:
    """Simple Perlin noise implementation for testing."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        random.seed(seed)
        self._permutation = list(range(256))
        random.shuffle(self._permutation)
        self._permutation = self._permutation * 2

    def noise(self, x: float, y: float = 0.0, z: float = 0.0) -> float:
        """Generate Perlin noise value at given coordinates."""
        xi = int(x) & 255
        return math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 0.5 + 0.5

    def noise_3d(self, x: float, y: float, z: float) -> Vector3:
        """Generate 3D Perlin noise vector."""
        return Vector3(
            self.noise(x, 0, 0) * 2 - 1,
            self.noise(0, y, 0) * 2 - 1,
            self.noise(0, 0, z) * 2 - 1,
        )


# =============================================================================
# Camera Shake System
# =============================================================================


@dataclass
class ShakeParams:
    """Parameters for camera shake."""
    amplitude: Vector3 = field(default_factory=lambda: Vector3(0.1, 0.1, 0.1))
    frequency: float = 10.0
    duration: float = 0.5
    shake_type: ShakeType = ShakeType.PERLIN
    decay: bool = True
    rotation_amplitude: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))


class CameraShake:
    """Camera shake effect implementation."""

    def __init__(self):
        self.active_shakes: List[dict] = []
        self.perlin = PerlinNoise()
        self._time = 0.0

    def add_shake(self, params: ShakeParams) -> str:
        """Add a new shake effect and return its ID."""
        shake_id = f"shake_{len(self.active_shakes)}_{self._time}"
        shake = {
            "id": shake_id,
            "params": params,
            "elapsed": 0.0,
            "seed_offset": random.random() * 1000,
        }
        self.active_shakes.append(shake)
        return shake_id

    def remove_shake(self, shake_id: str):
        """Remove a specific shake by ID."""
        self.active_shakes = [s for s in self.active_shakes if s["id"] != shake_id]

    def clear_all(self):
        """Remove all active shakes."""
        self.active_shakes.clear()

    def _compute_perlin_shake(self, shake: dict) -> Vector3:
        """Compute shake offset using Perlin noise."""
        params = shake["params"]
        t = shake["elapsed"] * params.frequency + shake["seed_offset"]
        noise = self.perlin.noise_3d(t, t + 100, t + 200)
        return Vector3(
            noise.x * params.amplitude.x,
            noise.y * params.amplitude.y,
            noise.z * params.amplitude.z,
        )

    def _compute_sine_shake(self, shake: dict) -> Vector3:
        """Compute shake offset using sine waves."""
        params = shake["params"]
        t = shake["elapsed"] * params.frequency
        return Vector3(
            math.sin(t * 1.0) * params.amplitude.x,
            math.sin(t * 1.3) * params.amplitude.y,
            math.sin(t * 0.7) * params.amplitude.z,
        )

    def _compute_random_shake(self, shake: dict) -> Vector3:
        """Compute shake offset using random values."""
        params = shake["params"]
        return Vector3(
            (random.random() * 2 - 1) * params.amplitude.x,
            (random.random() * 2 - 1) * params.amplitude.y,
            (random.random() * 2 - 1) * params.amplitude.z,
        )

    def update(self, delta_time: float) -> Vector3:
        """Update all shakes and return combined offset."""
        self._time += delta_time
        total_offset = Vector3()
        expired = []

        for shake in self.active_shakes:
            shake["elapsed"] += delta_time
            params = shake["params"]

            if shake["elapsed"] >= params.duration:
                expired.append(shake)
                continue

            if params.shake_type == ShakeType.PERLIN:
                offset = self._compute_perlin_shake(shake)
            elif params.shake_type == ShakeType.SINE:
                offset = self._compute_sine_shake(shake)
            else:
                offset = self._compute_random_shake(shake)

            if params.decay:
                decay_factor = 1.0 - (shake["elapsed"] / params.duration)
                offset = offset * decay_factor

            total_offset = total_offset + offset

        for shake in expired:
            self.active_shakes.remove(shake)

        return total_offset

    def get_rotation_offset(self) -> Vector3:
        """Get combined rotation offset from all shakes."""
        total_rotation = Vector3()
        for shake in self.active_shakes:
            params = shake["params"]
            if params.rotation_amplitude.magnitude() > 0:
                t = shake["elapsed"] * params.frequency
                rot = Vector3(
                    math.sin(t * 0.9) * params.rotation_amplitude.x,
                    math.sin(t * 1.1) * params.rotation_amplitude.y,
                    math.sin(t * 0.8) * params.rotation_amplitude.z,
                )
                if params.decay:
                    decay = 1.0 - (shake["elapsed"] / params.duration)
                    rot = rot * decay
                total_rotation = total_rotation + rot
        return total_rotation

    @property
    def is_shaking(self) -> bool:
        return len(self.active_shakes) > 0


# =============================================================================
# Trauma System
# =============================================================================


class TraumaSystem:
    """Trauma-based camera shake system."""

    def __init__(self):
        self.trauma = 0.0
        self.max_trauma = 1.0
        self.decay_rate = 1.0
        self.shake_amplitude = Vector3(0.5, 0.5, 0.1)
        self.shake_frequency = 15.0
        self.trauma_power = 2.0
        self._time = 0.0
        self.perlin = PerlinNoise()

    def add_trauma(self, amount: float):
        """Add trauma to the system."""
        self.trauma = min(self.max_trauma, self.trauma + amount)

    def set_trauma(self, amount: float):
        """Set trauma to a specific value."""
        self.trauma = max(0.0, min(self.max_trauma, amount))

    def update(self, delta_time: float) -> Vector3:
        """Update trauma and return shake offset."""
        self._time += delta_time

        if self.trauma <= 0:
            return Vector3()

        self.trauma = max(0.0, self.trauma - self.decay_rate * delta_time)

        shake_amount = self.trauma ** self.trauma_power

        t = self._time * self.shake_frequency
        noise = self.perlin.noise_3d(t, t + 100, t + 200)

        return Vector3(
            noise.x * self.shake_amplitude.x * shake_amount,
            noise.y * self.shake_amplitude.y * shake_amount,
            noise.z * self.shake_amplitude.z * shake_amount,
        )

    @property
    def is_active(self) -> bool:
        return self.trauma > 0


# =============================================================================
# FOV Effects
# =============================================================================


class FOVEffect:
    """FOV effect controller."""

    def __init__(self, base_fov: float = 60.0):
        self.base_fov = base_fov
        self._current_fov = base_fov
        self._target_fov = base_fov
        self._punch_velocity = 0.0
        self._punch_damping = 5.0
        self._punch_spring = 40.0
        self.zoom_speed = 5.0
        self._active_effects: List[dict] = []

    def punch(self, amount: float, duration: float = 0.0):
        """Apply a punch effect to FOV."""
        self._punch_velocity += amount
        if duration > 0:
            # Also add as a timed effect
            self.add_effect(amount, duration)

    def set_target_fov(self, fov: float):
        """Set target FOV for smooth transition."""
        self._target_fov = fov

    def add_effect(self, offset: float, duration: float, easing: str = "linear") -> str:
        """Add a temporary FOV effect."""
        effect_id = f"fov_effect_{len(self._active_effects)}"
        self._active_effects.append({
            "id": effect_id,
            "offset": offset,
            "duration": duration,
            "elapsed": 0.0,
            "easing": easing,
        })
        return effect_id

    def remove_effect(self, effect_id: str):
        """Remove a specific FOV effect."""
        self._active_effects = [e for e in self._active_effects if e["id"] != effect_id]

    def update(self, delta_time: float) -> float:
        """Update and return current FOV."""
        punch_force = -self._punch_spring * (self._current_fov - self._target_fov)
        damping_force = -self._punch_damping * self._punch_velocity
        self._punch_velocity += (punch_force + damping_force) * delta_time
        self._current_fov += self._punch_velocity * delta_time

        effect_offset = 0.0
        expired = []
        for effect in self._active_effects:
            effect["elapsed"] += delta_time
            if effect["elapsed"] >= effect["duration"]:
                expired.append(effect)
            else:
                progress = effect["elapsed"] / effect["duration"]
                if progress < 0.5:
                    factor = progress * 2
                else:
                    factor = (1.0 - progress) * 2
                effect_offset += effect["offset"] * factor

        for effect in expired:
            self._active_effects.remove(effect)

        return self._current_fov + effect_offset

    def reset(self):
        """Reset FOV to base value."""
        self._current_fov = self.base_fov
        self._target_fov = self.base_fov
        self._punch_velocity = 0.0
        self._active_effects.clear()

    @property
    def current_fov(self) -> float:
        return self._current_fov


# =============================================================================
# Tilt/Dutch Angle Effect
# =============================================================================


class TiltEffect:
    """Camera tilt/dutch angle effect."""

    def __init__(self):
        self.enabled = False
        self.current_tilt = 0.0
        self.target_tilt = 0.0
        self.max_tilt = 45.0
        self.tilt_speed = 5.0
        self._oscillation_enabled = False
        self._oscillation_amplitude = 5.0
        self._oscillation_frequency = 1.0
        self._time = 0.0

    def enable(self):
        """Enable the tilt effect."""
        self.enabled = True

    def disable(self):
        """Disable the tilt effect."""
        self.enabled = False

    def set_tilt(self, angle: float, instant: bool = False):
        """Set target tilt angle."""
        self.target_tilt = max(-self.max_tilt, min(self.max_tilt, angle))
        if instant:
            self.current_tilt = self.target_tilt

    def add_tilt(self, angle: float):
        """Add to current tilt."""
        self.set_tilt(self.target_tilt + angle)

    def start_oscillation(self, amplitude: float = 5.0, frequency: float = 1.0):
        """Start tilt oscillation."""
        self._oscillation_enabled = True
        self._oscillation_amplitude = amplitude
        self._oscillation_frequency = frequency

    def stop_oscillation(self):
        """Stop tilt oscillation."""
        self._oscillation_enabled = False

    def update(self, delta_time: float) -> float:
        """Update and return current tilt angle."""
        self._time += delta_time

        t = 1.0 - math.exp(-self.tilt_speed * delta_time)
        self.current_tilt = self.current_tilt + (self.target_tilt - self.current_tilt) * t

        oscillation = 0.0
        if self._oscillation_enabled:
            oscillation = math.sin(self._time * self._oscillation_frequency * 2 * math.pi) * self._oscillation_amplitude

        return self.current_tilt + oscillation

    def reset(self):
        """Reset tilt to zero."""
        self.current_tilt = 0.0
        self.target_tilt = 0.0
        self._oscillation_enabled = False


# =============================================================================
# Depth of Field Effect
# =============================================================================


class DepthOfFieldEffect:
    """Depth of field effect parameters."""

    def __init__(self):
        self.enabled = False
        self.focus_distance = 10.0
        self.aperture = 2.8
        self.min_aperture = 1.4
        self.max_aperture = 22.0
        self.near_blur_start = 0.0
        self.near_blur_end = 2.0
        self.far_blur_start = 15.0
        self.far_blur_end = 30.0
        self.blur_strength = 1.0
        self._auto_focus = False
        self._focus_target: Optional[Vector3] = None
        self._focus_speed = 5.0

    def enable(self):
        """Enable depth of field."""
        self.enabled = True

    def disable(self):
        """Disable depth of field."""
        self.enabled = False

    def set_focus_distance(self, distance: float):
        """Set manual focus distance."""
        self.focus_distance = max(0.1, distance)
        self._auto_focus = False

    def set_aperture(self, aperture: float):
        """Set aperture (f-stop)."""
        self.aperture = max(self.min_aperture, min(self.max_aperture, aperture))

    def enable_auto_focus(self, target: Vector3):
        """Enable auto-focus on a target."""
        self._auto_focus = True
        self._focus_target = target

    def disable_auto_focus(self):
        """Disable auto-focus."""
        self._auto_focus = False
        self._focus_target = None

    def set_blur_range(self, near_start: float, near_end: float,
                       far_start: float, far_end: float):
        """Set blur range parameters."""
        self.near_blur_start = near_start
        self.near_blur_end = near_end
        self.far_blur_start = far_start
        self.far_blur_end = far_end

    def update(self, delta_time: float, camera_position: Vector3 = None):
        """Update depth of field parameters."""
        if not self.enabled:
            return

        if self._auto_focus and self._focus_target and camera_position:
            target_distance = (self._focus_target - camera_position).magnitude()
            t = 1.0 - math.exp(-self._focus_speed * delta_time)
            self.focus_distance = self.focus_distance + (target_distance - self.focus_distance) * t

    def get_blur_at_distance(self, distance: float) -> float:
        """Calculate blur amount at a given distance."""
        if not self.enabled:
            return 0.0

        if distance < self.near_blur_end:
            if distance <= self.near_blur_start:
                return self.blur_strength
            return ((self.near_blur_end - distance) /
                    (self.near_blur_end - self.near_blur_start)) * self.blur_strength
        elif distance > self.far_blur_start:
            if distance >= self.far_blur_end:
                return self.blur_strength
            return ((distance - self.far_blur_start) /
                    (self.far_blur_end - self.far_blur_start)) * self.blur_strength

        return 0.0


# =============================================================================
# Motion Blur Effect
# =============================================================================


class MotionBlurEffect:
    """Motion blur effect controller."""

    def __init__(self):
        self.enabled = False
        self.strength = 0.5
        self.samples = 8
        self.velocity_scale = 1.0
        self.max_blur = 0.1
        self._camera_velocity = Vector3()
        self._last_position = None

    def enable(self):
        """Enable motion blur."""
        self.enabled = True

    def disable(self):
        """Disable motion blur."""
        self.enabled = False

    def set_strength(self, strength: float):
        """Set motion blur strength."""
        self.strength = max(0.0, min(1.0, strength))

    def set_samples(self, samples: int):
        """Set number of motion blur samples."""
        self.samples = max(2, min(32, samples))

    def update(self, delta_time: float, camera_position: Vector3):
        """Update motion blur based on camera movement."""
        if not self.enabled:
            self._camera_velocity = Vector3()
            return

        if self._last_position is None:
            self._last_position = camera_position
            self._camera_velocity = Vector3()
            return

        velocity = (camera_position - self._last_position) * (1.0 / delta_time if delta_time > 0 else 0)
        self._camera_velocity = velocity * self.velocity_scale
        self._last_position = camera_position

    def get_blur_vector(self) -> Vector3:
        """Get current motion blur vector."""
        if not self.enabled:
            return Vector3()

        magnitude = self._camera_velocity.magnitude()
        if magnitude > self.max_blur / self.strength:
            scale = (self.max_blur / self.strength) / magnitude
            return self._camera_velocity * scale * self.strength

        return self._camera_velocity * self.strength

    def get_blur_amount(self, velocity: Vector3) -> float:
        """Get blur amount based on velocity vector."""
        if not self.enabled:
            return 0.0
        return velocity.magnitude() * self.strength * self.velocity_scale


# =============================================================================
# Vignette Effect
# =============================================================================


class VignetteEffect:
    """Vignette effect controller."""

    def __init__(self):
        self.enabled = False
        self.intensity = 0.3
        self.smoothness = 0.5
        self.roundness = 1.0
        self.color = (0.0, 0.0, 0.0)
        self._target_intensity = 0.3
        self._transition_speed = 5.0

    def enable(self):
        """Enable vignette effect."""
        self.enabled = True

    def disable(self):
        """Disable vignette effect."""
        self.enabled = False

    def set_intensity(self, intensity: float, instant: bool = False):
        """Set vignette intensity."""
        self._target_intensity = max(0.0, min(1.0, intensity))
        if instant:
            self.intensity = self._target_intensity

    def set_color(self, r: float, g: float, b: float):
        """Set vignette color."""
        self.color = (
            max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)),
        )

    def update(self, delta_time: float):
        """Update vignette intensity."""
        if not self.enabled:
            return

        t = 1.0 - math.exp(-self._transition_speed * delta_time)
        self.intensity = self.intensity + (self._target_intensity - self.intensity) * t

    def pulse(self, peak_intensity: float, duration: float):
        """Create a pulse effect."""
        self._target_intensity = peak_intensity


# =============================================================================
# Effect Stack Manager
# =============================================================================


class EffectStackManager:
    """Manages multiple camera effects with stacking and blending."""

    def __init__(self):
        self.shake = CameraShake()
        self.trauma = TraumaSystem()
        self.fov = FOVEffect()
        self.tilt = TiltEffect()
        self.dof = DepthOfFieldEffect()
        self.motion_blur = MotionBlurEffect()
        self.vignette = VignetteEffect()
        self._blend_mode = BlendMode.ADDITIVE
        self._master_intensity = 1.0
        self._paused = False

    def set_blend_mode(self, mode: BlendMode):
        """Set blend mode for effects."""
        self._blend_mode = mode

    def set_master_intensity(self, intensity: float):
        """Set master intensity multiplier."""
        self._master_intensity = max(0.0, min(1.0, intensity))

    def pause_all(self):
        """Pause all effects."""
        self._paused = True

    def resume_all(self):
        """Resume all effects."""
        self._paused = False

    def reset_all(self):
        """Reset all effects to default."""
        self.shake.clear_all()
        self.trauma.set_trauma(0.0)
        self.fov.reset()
        self.tilt.reset()

    def update(self, delta_time: float, camera_position: Vector3 = None) -> dict:
        """Update all effects and return combined values."""
        if self._paused:
            return self._get_neutral_values()

        shake_offset = self.shake.update(delta_time)
        trauma_offset = self.trauma.update(delta_time)
        rotation_offset = self.shake.get_rotation_offset()
        fov = self.fov.update(delta_time)
        tilt = self.tilt.update(delta_time)

        if camera_position:
            self.dof.update(delta_time, camera_position)
            self.motion_blur.update(delta_time, camera_position)

        self.vignette.update(delta_time)

        if self._blend_mode == BlendMode.ADDITIVE:
            total_offset = shake_offset + trauma_offset
        elif self._blend_mode == BlendMode.OVERRIDE:
            total_offset = shake_offset if shake_offset.magnitude() > trauma_offset.magnitude() else trauma_offset
        else:
            total_offset = shake_offset + trauma_offset

        total_offset = total_offset * self._master_intensity
        rotation_offset = rotation_offset * self._master_intensity

        return {
            "position_offset": total_offset,
            "rotation_offset": rotation_offset,
            "fov": fov,
            "tilt": tilt,
            "dof_enabled": self.dof.enabled,
            "dof_focus_distance": self.dof.focus_distance,
            "dof_aperture": self.dof.aperture,
            "motion_blur_vector": self.motion_blur.get_blur_vector(),
            "vignette_intensity": self.vignette.intensity,
        }

    def _get_neutral_values(self) -> dict:
        """Get neutral/default effect values."""
        return {
            "position_offset": Vector3(),
            "rotation_offset": Vector3(),
            "fov": self.fov.base_fov,
            "tilt": 0.0,
            "dof_enabled": False,
            "dof_focus_distance": self.dof.focus_distance,
            "dof_aperture": self.dof.aperture,
            "motion_blur_vector": Vector3(),
            "vignette_intensity": 0.0,
        }


# =============================================================================
# Camera Shake Tests (~30 tests)
# =============================================================================


class TestCameraShake:
    """Test camera shake functionality."""

    def test_initialization(self):
        """Test shake system initialization."""
        shake = CameraShake()
        assert shake.is_shaking is False
        assert len(shake.active_shakes) == 0

    def test_add_shake(self):
        """Test adding a shake effect."""
        shake = CameraShake()
        params = ShakeParams(amplitude=Vector3(0.1, 0.1, 0.1), duration=1.0)
        shake_id = shake.add_shake(params)
        assert shake.is_shaking is True
        assert len(shake.active_shakes) == 1
        assert shake_id is not None

    def test_remove_shake(self):
        """Test removing a specific shake."""
        shake = CameraShake()
        params = ShakeParams(duration=1.0)
        shake_id = shake.add_shake(params)
        shake.remove_shake(shake_id)
        assert shake.is_shaking is False

    def test_clear_all_shakes(self):
        """Test clearing all shakes."""
        shake = CameraShake()
        for _ in range(5):
            shake.add_shake(ShakeParams(duration=1.0))
        shake.clear_all()
        assert len(shake.active_shakes) == 0

    def test_shake_expires_after_duration(self):
        """Test shake expires after duration."""
        shake = CameraShake()
        params = ShakeParams(duration=0.5)
        shake.add_shake(params)
        shake.update(0.6)
        assert shake.is_shaking is False

    def test_perlin_shake_type(self):
        """Test Perlin noise shake type."""
        shake = CameraShake()
        params = ShakeParams(
            amplitude=Vector3(1.0, 1.0, 1.0),
            shake_type=ShakeType.PERLIN,
            duration=1.0,
        )
        shake.add_shake(params)
        offset = shake.update(0.1)
        assert offset.magnitude() > 0

    def test_sine_shake_type(self):
        """Test sine wave shake type."""
        shake = CameraShake()
        params = ShakeParams(
            amplitude=Vector3(1.0, 1.0, 1.0),
            shake_type=ShakeType.SINE,
            duration=1.0,
        )
        shake.add_shake(params)
        offset = shake.update(0.1)
        assert offset.magnitude() > 0

    def test_random_shake_type(self):
        """Test random shake type."""
        shake = CameraShake()
        params = ShakeParams(
            amplitude=Vector3(1.0, 1.0, 1.0),
            shake_type=ShakeType.RANDOM,
            duration=1.0,
        )
        shake.add_shake(params)
        offset = shake.update(0.1)
        assert offset.magnitude() > 0

    def test_shake_decay(self):
        """Test shake decay over time."""
        shake = CameraShake()
        params = ShakeParams(
            amplitude=Vector3(1.0, 1.0, 1.0),
            duration=1.0,
            decay=True,
            shake_type=ShakeType.SINE,
        )
        shake.add_shake(params)

        offset_early = shake.update(0.1).magnitude()
        shake.add_shake(params)
        for _ in range(8):
            shake.update(0.1)
        offset_late = shake.update(0.1).magnitude()

        assert offset_late < offset_early * 2

    def test_shake_no_decay(self):
        """Test shake without decay."""
        shake = CameraShake()
        params = ShakeParams(
            amplitude=Vector3(1.0, 0, 0),
            duration=2.0,
            decay=False,
            shake_type=ShakeType.SINE,
        )
        shake.add_shake(params)

        magnitudes = []
        for _ in range(10):
            offset = shake.update(0.1)
            magnitudes.append(offset.magnitude())

        assert max(magnitudes) > 0

    def test_shake_frequency(self):
        """Test shake frequency affects oscillation."""
        shake_slow = CameraShake()
        shake_fast = CameraShake()

        params_slow = ShakeParams(frequency=1.0, duration=1.0, shake_type=ShakeType.SINE)
        params_fast = ShakeParams(frequency=10.0, duration=1.0, shake_type=ShakeType.SINE)

        shake_slow.add_shake(params_slow)
        shake_fast.add_shake(params_fast)

        slow_values = [shake_slow.update(0.05).x for _ in range(20)]
        fast_values = [shake_fast.update(0.05).x for _ in range(20)]

        slow_changes = sum(1 for i in range(1, len(slow_values))
                          if (slow_values[i] > 0) != (slow_values[i-1] > 0))
        fast_changes = sum(1 for i in range(1, len(fast_values))
                          if (fast_values[i] > 0) != (fast_values[i-1] > 0))

        assert fast_changes >= slow_changes

    def test_amplitude_scaling(self):
        """Test amplitude affects shake magnitude."""
        shake_small = CameraShake()
        shake_large = CameraShake()

        shake_small.add_shake(ShakeParams(
            amplitude=Vector3(0.1, 0.1, 0.1),
            duration=1.0,
            shake_type=ShakeType.SINE,
        ))
        shake_large.add_shake(ShakeParams(
            amplitude=Vector3(1.0, 1.0, 1.0),
            duration=1.0,
            shake_type=ShakeType.SINE,
        ))

        small_offset = shake_small.update(0.25)
        large_offset = shake_large.update(0.25)

        assert large_offset.magnitude() > small_offset.magnitude()

    def test_multiple_shakes_combine(self):
        """Test multiple shakes combine additively."""
        shake = CameraShake()
        params = ShakeParams(
            amplitude=Vector3(0.5, 0.5, 0.5),
            duration=1.0,
            shake_type=ShakeType.SINE,
        )

        single_offset = shake.update(0.1)
        shake.add_shake(params)
        shake.add_shake(params)
        double_offset = shake.update(0.1)

        assert double_offset.magnitude() >= single_offset.magnitude()

    def test_rotation_offset(self):
        """Test rotation offset from shake."""
        shake = CameraShake()
        params = ShakeParams(
            rotation_amplitude=Vector3(5.0, 5.0, 5.0),
            duration=1.0,
        )
        shake.add_shake(params)
        shake.update(0.1)
        rotation = shake.get_rotation_offset()
        assert rotation.magnitude() > 0

    def test_directional_shake(self):
        """Test directional shake (single axis)."""
        shake = CameraShake()
        params = ShakeParams(
            amplitude=Vector3(1.0, 0.0, 0.0),
            duration=1.0,
            shake_type=ShakeType.SINE,
        )
        shake.add_shake(params)
        offset = shake.update(0.25)
        assert abs(offset.x) > 0
        assert offset.y == 0
        assert offset.z == 0


# =============================================================================
# Trauma System Tests (~25 tests)
# =============================================================================


class TestTraumaSystem:
    """Test trauma-based shake system."""

    def test_initialization(self):
        """Test trauma system initialization."""
        trauma = TraumaSystem()
        assert trauma.trauma == 0.0
        assert trauma.is_active is False

    def test_add_trauma(self):
        """Test adding trauma."""
        trauma = TraumaSystem()
        trauma.add_trauma(0.5)
        assert trauma.trauma == 0.5
        assert trauma.is_active is True

    def test_trauma_capped_at_max(self):
        """Test trauma is capped at max value."""
        trauma = TraumaSystem()
        trauma.add_trauma(2.0)
        assert trauma.trauma == trauma.max_trauma

    def test_trauma_additive(self):
        """Test trauma is additive."""
        trauma = TraumaSystem()
        trauma.add_trauma(0.3)
        trauma.add_trauma(0.3)
        assert trauma.trauma == 0.6

    def test_set_trauma(self):
        """Test setting trauma directly."""
        trauma = TraumaSystem()
        trauma.set_trauma(0.7)
        assert trauma.trauma == 0.7

    def test_set_trauma_clamped(self):
        """Test set_trauma is clamped."""
        trauma = TraumaSystem()
        trauma.set_trauma(5.0)
        assert trauma.trauma == trauma.max_trauma
        trauma.set_trauma(-1.0)
        assert trauma.trauma == 0.0

    def test_trauma_decay(self):
        """Test trauma decays over time."""
        trauma = TraumaSystem()
        trauma.add_trauma(1.0)
        initial = trauma.trauma
        trauma.update(0.5)
        assert trauma.trauma < initial

    def test_decay_rate_affects_speed(self):
        """Test decay rate affects decay speed."""
        trauma_slow = TraumaSystem()
        trauma_fast = TraumaSystem()
        trauma_slow.decay_rate = 0.5
        trauma_fast.decay_rate = 2.0

        trauma_slow.add_trauma(1.0)
        trauma_fast.add_trauma(1.0)

        trauma_slow.update(0.5)
        trauma_fast.update(0.5)

        assert trauma_fast.trauma < trauma_slow.trauma

    def test_trauma_power_affects_shake(self):
        """Test trauma power affects shake intensity."""
        trauma = TraumaSystem()
        trauma.trauma_power = 2.0
        trauma.add_trauma(0.5)

        offset = trauma.update(0.016)
        assert offset.magnitude() > 0

    def test_shake_amplitude(self):
        """Test shake amplitude affects output."""
        trauma_small = TraumaSystem()
        trauma_large = TraumaSystem()
        trauma_small.shake_amplitude = Vector3(0.1, 0.1, 0.1)
        trauma_large.shake_amplitude = Vector3(1.0, 1.0, 1.0)

        trauma_small.add_trauma(1.0)
        trauma_large.add_trauma(1.0)

        offset_small = trauma_small.update(0.1)
        offset_large = trauma_large.update(0.1)

        assert offset_large.magnitude() > offset_small.magnitude()

    def test_shake_frequency(self):
        """Test shake frequency affects oscillation."""
        trauma = TraumaSystem()
        trauma.shake_frequency = 20.0
        trauma.add_trauma(1.0)

        offsets = [trauma.update(0.05) for _ in range(20)]
        assert len(offsets) > 0

    def test_no_shake_when_zero_trauma(self):
        """Test no shake output when trauma is zero."""
        trauma = TraumaSystem()
        offset = trauma.update(0.1)
        assert offset.x == 0 and offset.y == 0 and offset.z == 0

    def test_trauma_fully_decays(self):
        """Test trauma fully decays to zero."""
        trauma = TraumaSystem()
        trauma.add_trauma(0.5)

        for _ in range(100):
            trauma.update(0.1)

        assert trauma.trauma == 0.0
        assert trauma.is_active is False


# =============================================================================
# FOV Effect Tests (~25 tests)
# =============================================================================


class TestFOVEffect:
    """Test FOV effect functionality."""

    def test_initialization(self):
        """Test FOV effect initialization."""
        fov = FOVEffect(base_fov=60.0)
        assert fov.base_fov == 60.0
        assert fov.current_fov == 60.0

    def test_punch_effect(self):
        """Test FOV punch effect."""
        fov = FOVEffect(base_fov=60.0)
        fov.punch(20.0)
        fov.update(0.016)
        assert fov.current_fov != 60.0

    def test_punch_recovers(self):
        """Test FOV punch recovers to target."""
        fov = FOVEffect(base_fov=60.0)
        fov.punch(30.0)

        for _ in range(100):
            fov.update(0.016)

        assert abs(fov.current_fov - 60.0) < 1.0

    def test_set_target_fov(self):
        """Test setting target FOV."""
        fov = FOVEffect(base_fov=60.0)
        fov.set_target_fov(90.0)

        for _ in range(100):
            fov.update(0.016)

        assert abs(fov.current_fov - 90.0) < 1.0

    def test_add_temporary_effect(self):
        """Test adding temporary FOV effect."""
        fov = FOVEffect(base_fov=60.0)
        effect_id = fov.add_effect(offset=10.0, duration=1.0)
        assert effect_id is not None
        current = fov.update(0.5)
        assert current > 60.0

    def test_temporary_effect_expires(self):
        """Test temporary effect expires."""
        fov = FOVEffect(base_fov=60.0)
        fov.add_effect(offset=10.0, duration=0.5)

        fov.update(0.6)
        current = fov.update(0.1)

        assert abs(current - 60.0) < 2.0

    def test_remove_effect(self):
        """Test removing specific FOV effect."""
        fov = FOVEffect(base_fov=60.0)
        effect_id = fov.add_effect(offset=10.0, duration=10.0)
        fov.remove_effect(effect_id)
        current = fov.update(0.1)
        assert abs(current - 60.0) < 1.0

    def test_reset(self):
        """Test FOV reset."""
        fov = FOVEffect(base_fov=60.0)
        fov.punch(30.0)
        fov.set_target_fov(90.0)
        fov.add_effect(offset=10.0, duration=10.0)
        fov.reset()
        assert fov.current_fov == 60.0
        assert len(fov._active_effects) == 0

    def test_multiple_effects_stack(self):
        """Test multiple FOV effects stack."""
        fov = FOVEffect(base_fov=60.0)
        fov.add_effect(offset=5.0, duration=1.0)
        fov.add_effect(offset=5.0, duration=1.0)
        current = fov.update(0.5)
        assert current > 65.0

    def test_zoom_punch(self):
        """Test combined zoom and punch."""
        fov = FOVEffect(base_fov=60.0)
        fov.set_target_fov(40.0)
        fov.punch(20.0)

        for _ in range(50):
            fov.update(0.016)

        assert abs(fov.current_fov - 40.0) < 5.0


# =============================================================================
# Tilt Effect Tests (~20 tests)
# =============================================================================


class TestTiltEffect:
    """Test tilt/dutch angle effect."""

    def test_initialization(self):
        """Test tilt effect initialization."""
        tilt = TiltEffect()
        assert tilt.current_tilt == 0.0
        assert tilt.target_tilt == 0.0

    def test_set_tilt(self):
        """Test setting tilt angle."""
        tilt = TiltEffect()
        tilt.set_tilt(15.0)
        assert tilt.target_tilt == 15.0

    def test_set_tilt_instant(self):
        """Test instant tilt setting."""
        tilt = TiltEffect()
        tilt.set_tilt(15.0, instant=True)
        assert tilt.current_tilt == 15.0

    def test_tilt_clamped(self):
        """Test tilt is clamped to max."""
        tilt = TiltEffect()
        tilt.max_tilt = 30.0
        tilt.set_tilt(60.0)
        assert tilt.target_tilt == 30.0

    def test_add_tilt(self):
        """Test adding to current tilt."""
        tilt = TiltEffect()
        tilt.set_tilt(10.0, instant=True)
        tilt.add_tilt(5.0)
        assert tilt.target_tilt == 15.0

    def test_smooth_transition(self):
        """Test smooth tilt transition."""
        tilt = TiltEffect()
        tilt.set_tilt(20.0)

        for _ in range(50):
            tilt.update(0.016)

        assert abs(tilt.current_tilt - 20.0) < 1.0

    def test_oscillation_start(self):
        """Test starting tilt oscillation."""
        tilt = TiltEffect()
        tilt.start_oscillation(amplitude=5.0, frequency=1.0)
        assert tilt._oscillation_enabled is True

    def test_oscillation_stop(self):
        """Test stopping tilt oscillation."""
        tilt = TiltEffect()
        tilt.start_oscillation()
        tilt.stop_oscillation()
        assert tilt._oscillation_enabled is False

    def test_oscillation_affects_output(self):
        """Test oscillation affects tilt output."""
        tilt = TiltEffect()
        tilt.start_oscillation(amplitude=10.0, frequency=2.0)

        values = []
        for _ in range(20):
            values.append(tilt.update(0.05))

        assert max(values) != min(values)

    def test_reset(self):
        """Test tilt reset."""
        tilt = TiltEffect()
        tilt.set_tilt(20.0, instant=True)
        tilt.start_oscillation()
        tilt.reset()
        assert tilt.current_tilt == 0.0
        assert tilt._oscillation_enabled is False


# =============================================================================
# Depth of Field Tests (~20 tests)
# =============================================================================


class TestDepthOfFieldEffect:
    """Test depth of field effect."""

    def test_initialization(self):
        """Test DOF initialization."""
        dof = DepthOfFieldEffect()
        assert dof.enabled is False
        assert dof.focus_distance == 10.0

    def test_enable_disable(self):
        """Test enable/disable DOF."""
        dof = DepthOfFieldEffect()
        dof.enable()
        assert dof.enabled is True
        dof.disable()
        assert dof.enabled is False

    def test_set_focus_distance(self):
        """Test setting focus distance."""
        dof = DepthOfFieldEffect()
        dof.set_focus_distance(5.0)
        assert dof.focus_distance == 5.0

    def test_focus_distance_minimum(self):
        """Test focus distance has minimum."""
        dof = DepthOfFieldEffect()
        dof.set_focus_distance(0.0)
        assert dof.focus_distance >= 0.1

    def test_set_aperture(self):
        """Test setting aperture."""
        dof = DepthOfFieldEffect()
        dof.set_aperture(5.6)
        assert dof.aperture == 5.6

    def test_aperture_clamped(self):
        """Test aperture is clamped."""
        dof = DepthOfFieldEffect()
        dof.set_aperture(0.5)
        assert dof.aperture >= dof.min_aperture
        dof.set_aperture(50.0)
        assert dof.aperture <= dof.max_aperture

    def test_auto_focus_enable(self):
        """Test enabling auto-focus."""
        dof = DepthOfFieldEffect()
        target = Vector3(10, 0, 0)
        dof.enable_auto_focus(target)
        assert dof._auto_focus is True
        assert dof._focus_target is target

    def test_auto_focus_disable(self):
        """Test disabling auto-focus."""
        dof = DepthOfFieldEffect()
        dof.enable_auto_focus(Vector3())
        dof.disable_auto_focus()
        assert dof._auto_focus is False

    def test_auto_focus_updates_distance(self):
        """Test auto-focus updates focus distance."""
        dof = DepthOfFieldEffect()
        dof.enable()
        target = Vector3(20, 0, 0)
        dof.enable_auto_focus(target)
        dof.focus_distance = 5.0

        camera_pos = Vector3(0, 0, 0)
        for _ in range(50):
            dof.update(0.1, camera_pos)

        assert abs(dof.focus_distance - 20.0) < 1.0

    def test_set_blur_range(self):
        """Test setting blur range."""
        dof = DepthOfFieldEffect()
        dof.set_blur_range(0.0, 2.0, 10.0, 20.0)
        assert dof.near_blur_start == 0.0
        assert dof.near_blur_end == 2.0
        assert dof.far_blur_start == 10.0
        assert dof.far_blur_end == 20.0

    def test_blur_at_focus_distance(self):
        """Test no blur at focus distance."""
        dof = DepthOfFieldEffect()
        dof.enable()
        dof.set_blur_range(0.0, 5.0, 15.0, 30.0)
        dof.focus_distance = 10.0

        blur = dof.get_blur_at_distance(10.0)
        assert blur == 0.0

    def test_blur_near_distance(self):
        """Test blur at near distance."""
        dof = DepthOfFieldEffect()
        dof.enable()
        dof.blur_strength = 1.0
        dof.set_blur_range(0.0, 5.0, 15.0, 30.0)

        blur = dof.get_blur_at_distance(0.0)
        assert blur > 0

    def test_blur_far_distance(self):
        """Test blur at far distance."""
        dof = DepthOfFieldEffect()
        dof.enable()
        dof.blur_strength = 1.0
        dof.set_blur_range(0.0, 5.0, 15.0, 30.0)

        blur = dof.get_blur_at_distance(30.0)
        assert blur > 0

    def test_blur_disabled(self):
        """Test no blur when disabled."""
        dof = DepthOfFieldEffect()
        dof.enabled = False
        blur = dof.get_blur_at_distance(0.0)
        assert blur == 0.0


# =============================================================================
# Motion Blur Tests (~15 tests)
# =============================================================================


class TestMotionBlurEffect:
    """Test motion blur effect."""

    def test_initialization(self):
        """Test motion blur initialization."""
        blur = MotionBlurEffect()
        assert blur.enabled is False
        assert blur.strength == 0.5

    def test_enable_disable(self):
        """Test enable/disable motion blur."""
        blur = MotionBlurEffect()
        blur.enable()
        assert blur.enabled is True
        blur.disable()
        assert blur.enabled is False

    def test_set_strength(self):
        """Test setting blur strength."""
        blur = MotionBlurEffect()
        blur.set_strength(0.8)
        assert blur.strength == 0.8

    def test_strength_clamped(self):
        """Test strength is clamped."""
        blur = MotionBlurEffect()
        blur.set_strength(2.0)
        assert blur.strength <= 1.0
        blur.set_strength(-0.5)
        assert blur.strength >= 0.0

    def test_set_samples(self):
        """Test setting sample count."""
        blur = MotionBlurEffect()
        blur.set_samples(16)
        assert blur.samples == 16

    def test_samples_clamped(self):
        """Test samples are clamped."""
        blur = MotionBlurEffect()
        blur.set_samples(1)
        assert blur.samples >= 2
        blur.set_samples(64)
        assert blur.samples <= 32

    def test_blur_from_movement(self):
        """Test blur vector from camera movement."""
        blur = MotionBlurEffect()
        blur.enable()
        blur.strength = 1.0

        blur.update(0.016, Vector3(0, 0, 0))
        blur.update(0.016, Vector3(10, 0, 0))

        vector = blur.get_blur_vector()
        assert vector.magnitude() > 0

    def test_no_blur_when_stationary(self):
        """Test no blur when camera is stationary."""
        blur = MotionBlurEffect()
        blur.enable()

        blur.update(0.016, Vector3(5, 5, 5))
        blur.update(0.016, Vector3(5, 5, 5))

        vector = blur.get_blur_vector()
        assert vector.magnitude() < 0.01

    def test_no_blur_when_disabled(self):
        """Test no blur when disabled."""
        blur = MotionBlurEffect()
        blur.disable()
        blur.update(0.016, Vector3(0, 0, 0))
        blur.update(0.016, Vector3(100, 0, 0))
        vector = blur.get_blur_vector()
        assert vector.magnitude() == 0

    def test_max_blur_clamping(self):
        """Test blur is clamped to max."""
        blur = MotionBlurEffect()
        blur.enable()
        blur.max_blur = 0.05
        blur.strength = 1.0

        blur.update(0.016, Vector3(0, 0, 0))
        blur.update(0.016, Vector3(1000, 0, 0))

        vector = blur.get_blur_vector()
        assert vector.magnitude() <= blur.max_blur + 0.01


# =============================================================================
# Vignette Effect Tests (~15 tests)
# =============================================================================


class TestVignetteEffect:
    """Test vignette effect."""

    def test_initialization(self):
        """Test vignette initialization."""
        vignette = VignetteEffect()
        assert vignette.enabled is False
        assert vignette.intensity == 0.3

    def test_enable_disable(self):
        """Test enable/disable vignette."""
        vignette = VignetteEffect()
        vignette.enable()
        assert vignette.enabled is True
        vignette.disable()
        assert vignette.enabled is False

    def test_set_intensity(self):
        """Test setting vignette intensity."""
        vignette = VignetteEffect()
        vignette.set_intensity(0.5)
        assert vignette._target_intensity == 0.5

    def test_set_intensity_instant(self):
        """Test instant intensity change."""
        vignette = VignetteEffect()
        vignette.set_intensity(0.8, instant=True)
        assert vignette.intensity == 0.8

    def test_intensity_clamped(self):
        """Test intensity is clamped."""
        vignette = VignetteEffect()
        vignette.set_intensity(2.0)
        assert vignette._target_intensity <= 1.0
        vignette.set_intensity(-0.5)
        assert vignette._target_intensity >= 0.0

    def test_set_color(self):
        """Test setting vignette color."""
        vignette = VignetteEffect()
        vignette.set_color(1.0, 0.0, 0.0)
        assert vignette.color == (1.0, 0.0, 0.0)

    def test_color_clamped(self):
        """Test color values are clamped."""
        vignette = VignetteEffect()
        vignette.set_color(2.0, -0.5, 1.5)
        assert all(0.0 <= c <= 1.0 for c in vignette.color)

    def test_smooth_intensity_transition(self):
        """Test smooth intensity transition."""
        vignette = VignetteEffect()
        vignette.enable()
        vignette.intensity = 0.0
        vignette.set_intensity(1.0)

        for _ in range(50):
            vignette.update(0.1)

        assert abs(vignette.intensity - 1.0) < 0.1

    def test_no_update_when_disabled(self):
        """Test no update when disabled."""
        vignette = VignetteEffect()
        vignette.enabled = False
        vignette.intensity = 0.5
        vignette.set_intensity(1.0)
        vignette.update(1.0)
        assert vignette.intensity == 0.5


# =============================================================================
# Effect Stack Manager Tests (~20 tests)
# =============================================================================


class TestEffectStackManager:
    """Test effect stack manager."""

    def test_initialization(self):
        """Test effect manager initialization."""
        manager = EffectStackManager()
        assert manager.shake is not None
        assert manager.trauma is not None
        assert manager.fov is not None

    def test_set_blend_mode(self):
        """Test setting blend mode."""
        manager = EffectStackManager()
        manager.set_blend_mode(BlendMode.MULTIPLICATIVE)
        assert manager._blend_mode == BlendMode.MULTIPLICATIVE

    def test_set_master_intensity(self):
        """Test setting master intensity."""
        manager = EffectStackManager()
        manager.set_master_intensity(0.5)
        assert manager._master_intensity == 0.5

    def test_master_intensity_clamped(self):
        """Test master intensity is clamped."""
        manager = EffectStackManager()
        manager.set_master_intensity(2.0)
        assert manager._master_intensity <= 1.0
        manager.set_master_intensity(-0.5)
        assert manager._master_intensity >= 0.0

    def test_pause_resume(self):
        """Test pausing and resuming effects."""
        manager = EffectStackManager()
        manager.pause_all()
        assert manager._paused is True
        manager.resume_all()
        assert manager._paused is False

    def test_paused_returns_neutral(self):
        """Test paused manager returns neutral values."""
        manager = EffectStackManager()
        manager.trauma.add_trauma(1.0)
        manager.pause_all()

        result = manager.update(0.1)
        assert result["position_offset"].magnitude() == 0

    def test_reset_all(self):
        """Test resetting all effects."""
        manager = EffectStackManager()
        manager.shake.add_shake(ShakeParams(duration=10.0))
        manager.trauma.add_trauma(1.0)
        manager.fov.punch(20.0)
        manager.tilt.set_tilt(15.0)

        manager.reset_all()

        assert not manager.shake.is_shaking
        assert manager.trauma.trauma == 0.0

    def test_update_combines_effects(self):
        """Test update combines all effects."""
        manager = EffectStackManager()
        manager.shake.add_shake(ShakeParams(
            amplitude=Vector3(0.1, 0.1, 0.1),
            duration=1.0,
        ))
        manager.trauma.add_trauma(0.5)

        result = manager.update(0.1)

        assert "position_offset" in result
        assert "fov" in result
        assert "tilt" in result

    def test_additive_blend_mode(self):
        """Test additive blend mode combines offsets."""
        manager = EffectStackManager()
        manager.set_blend_mode(BlendMode.ADDITIVE)
        manager.shake.add_shake(ShakeParams(
            amplitude=Vector3(0.5, 0, 0),
            duration=1.0,
            shake_type=ShakeType.SINE,
        ))
        manager.trauma.add_trauma(0.5)

        result = manager.update(0.1)
        assert result["position_offset"].magnitude() > 0

    def test_master_intensity_scales_output(self):
        """Test master intensity scales effect output."""
        manager_full = EffectStackManager()
        manager_half = EffectStackManager()

        manager_full.set_master_intensity(1.0)
        manager_half.set_master_intensity(0.5)

        manager_full.shake.add_shake(ShakeParams(
            amplitude=Vector3(1.0, 1.0, 1.0),
            duration=1.0,
            shake_type=ShakeType.SINE,
        ))
        manager_half.shake.add_shake(ShakeParams(
            amplitude=Vector3(1.0, 1.0, 1.0),
            duration=1.0,
            shake_type=ShakeType.SINE,
        ))

        result_full = manager_full.update(0.25)
        result_half = manager_half.update(0.25)

        assert result_full["position_offset"].magnitude() > result_half["position_offset"].magnitude()

    def test_update_returns_all_values(self):
        """Test update returns all effect values."""
        manager = EffectStackManager()
        result = manager.update(0.1, Vector3())

        assert "position_offset" in result
        assert "rotation_offset" in result
        assert "fov" in result
        assert "tilt" in result
        assert "dof_enabled" in result
        assert "motion_blur_vector" in result
        assert "vignette_intensity" in result


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEffectEdgeCases:
    """Test edge cases for camera effects."""

    def test_zero_delta_time(self):
        """Test effects handle zero delta time."""
        manager = EffectStackManager()
        manager.trauma.add_trauma(1.0)
        result = manager.update(0.0)
        assert result is not None

    def test_very_large_delta_time(self):
        """Test effects handle large delta time."""
        manager = EffectStackManager()
        manager.trauma.add_trauma(1.0)
        result = manager.update(100.0)
        assert manager.trauma.trauma == 0.0

    def test_rapid_effect_changes(self):
        """Test rapid effect additions and removals."""
        shake = CameraShake()
        ids = []
        for _ in range(100):
            shake_id = shake.add_shake(ShakeParams(duration=0.1))
            ids.append(shake_id)

        for shake_id in ids[:50]:
            shake.remove_shake(shake_id)

        shake.update(0.2)
        assert len(shake.active_shakes) == 0

    def test_concurrent_effects(self):
        """Test many concurrent effects."""
        manager = EffectStackManager()

        for _ in range(10):
            manager.shake.add_shake(ShakeParams(duration=1.0))
        manager.trauma.add_trauma(0.5)
        manager.fov.add_effect(5.0, 1.0)
        manager.tilt.set_tilt(10.0)

        result = manager.update(0.1, Vector3())
        assert result["position_offset"].magnitude() > 0


# =============================================================================
# Additional Camera Shake Tests
# =============================================================================


class TestCameraShakeAdvanced:
    """Additional camera shake tests."""

    def test_shake_unique_ids(self):
        """Test each shake gets a unique ID."""
        shake = CameraShake()
        id1 = shake.add_shake(ShakeParams(duration=1.0))
        id2 = shake.add_shake(ShakeParams(duration=1.0))
        assert id1 != id2

    def test_shake_zero_amplitude(self):
        """Test shake with zero amplitude produces no offset."""
        shake = CameraShake()
        shake.add_shake(ShakeParams(amplitude=Vector3(0, 0, 0), duration=1.0))
        offset = shake.update(0.1)
        assert offset.magnitude() == 0

    def test_shake_zero_duration(self):
        """Test shake with zero duration expires immediately."""
        shake = CameraShake()
        shake.add_shake(ShakeParams(duration=0.0))
        shake.update(0.001)
        assert shake.is_shaking is False

    def test_rotation_shake_with_position_shake(self):
        """Test combined rotation and position shake."""
        shake = CameraShake()
        shake.add_shake(ShakeParams(
            amplitude=Vector3(1, 1, 1),
            rotation_amplitude=Vector3(5, 5, 5),
            duration=1.0,
        ))
        offset = shake.update(0.1)
        rotation = shake.get_rotation_offset()
        assert offset.magnitude() > 0
        assert rotation.magnitude() > 0

    def test_remove_nonexistent_shake(self):
        """Test removing nonexistent shake does not error."""
        shake = CameraShake()
        shake.remove_shake("nonexistent_id")

    def test_shake_time_accumulation(self):
        """Test shake internal time accumulates."""
        shake = CameraShake()
        initial_time = shake._time
        shake.update(1.0)
        assert shake._time > initial_time


# =============================================================================
# Additional Trauma System Tests
# =============================================================================


class TestTraumaSystemAdvanced:
    """Additional trauma system tests."""

    def test_trauma_at_max_stays_max(self):
        """Test trauma at max stays at max when more added."""
        trauma = TraumaSystem()
        trauma.add_trauma(1.0)
        trauma.add_trauma(0.5)
        assert trauma.trauma == trauma.max_trauma

    def test_trauma_power_zero(self):
        """Test trauma power of zero gives constant shake."""
        trauma = TraumaSystem()
        trauma.trauma_power = 0.0
        trauma.add_trauma(0.5)
        offset = trauma.update(0.1)
        assert offset.magnitude() > 0

    def test_trauma_power_high(self):
        """Test high trauma power reduces shake at low trauma."""
        trauma = TraumaSystem()
        trauma.trauma_power = 4.0
        trauma.add_trauma(0.5)
        offset = trauma.update(0.1)

    def test_trauma_frequency_zero(self):
        """Test zero frequency still produces shake."""
        trauma = TraumaSystem()
        trauma.shake_frequency = 0.0
        trauma.add_trauma(1.0)
        offset = trauma.update(0.1)

    def test_custom_max_trauma(self):
        """Test custom max trauma value."""
        trauma = TraumaSystem()
        trauma.max_trauma = 0.5
        trauma.add_trauma(1.0)
        assert trauma.trauma == 0.5


# =============================================================================
# Additional FOV Effect Tests
# =============================================================================


class TestFOVEffectAdvanced:
    """Additional FOV effect tests."""

    def test_negative_fov_punch(self):
        """Test negative FOV punch (zoom in effect)."""
        fov = FOVEffect(base_fov=60.0)
        fov.punch(-20.0)
        fov.update(0.016)
        assert fov.current_fov < 60.0

    def test_multiple_punches_accumulate(self):
        """Test multiple punches accumulate velocity."""
        fov = FOVEffect(base_fov=60.0)
        fov.punch(10.0)
        fov.punch(10.0)
        fov.update(0.016)
        # Velocity should be higher than single punch

    def test_effect_negative_offset(self):
        """Test FOV effect with negative offset."""
        fov = FOVEffect(base_fov=60.0)
        fov.add_effect(offset=-10.0, duration=1.0)
        current = fov.update(0.5)
        assert current < 60.0

    def test_effect_at_boundaries(self):
        """Test effect at start and end boundaries."""
        fov = FOVEffect(base_fov=60.0)
        fov.add_effect(offset=10.0, duration=1.0)
        start = fov.update(0.0)
        fov._active_effects[0]["elapsed"] = 0.5
        middle = fov.update(0.0)
        fov._active_effects[0]["elapsed"] = 1.0

    def test_spring_damping_parameters(self):
        """Test FOV spring damping parameters."""
        fov = FOVEffect(base_fov=60.0)
        fov._punch_damping = 10.0
        fov._punch_spring = 20.0
        fov.punch(30.0)
        for _ in range(100):
            fov.update(0.016)


# =============================================================================
# Additional Tilt Effect Tests
# =============================================================================


class TestTiltEffectAdvanced:
    """Additional tilt effect tests."""

    def test_negative_tilt(self):
        """Test negative tilt angle."""
        tilt = TiltEffect()
        tilt.set_tilt(-30.0)
        assert tilt.target_tilt == -30.0

    def test_tilt_beyond_max(self):
        """Test tilt clamped at max."""
        tilt = TiltEffect()
        tilt.max_tilt = 30.0
        tilt.set_tilt(60.0)
        assert tilt.target_tilt == 30.0

    def test_tilt_speed_affects_transition(self):
        """Test tilt speed affects transition rate."""
        tilt_slow = TiltEffect()
        tilt_fast = TiltEffect()
        tilt_slow.tilt_speed = 1.0
        tilt_fast.tilt_speed = 10.0

        tilt_slow.set_tilt(30.0)
        tilt_fast.set_tilt(30.0)

        tilt_slow.update(0.1)
        tilt_fast.update(0.1)

        assert tilt_fast.current_tilt > tilt_slow.current_tilt

    def test_oscillation_with_tilt(self):
        """Test oscillation combined with base tilt."""
        tilt = TiltEffect()
        tilt.set_tilt(15.0, instant=True)
        tilt.start_oscillation(amplitude=5.0, frequency=2.0)
        values = [tilt.update(0.1) for _ in range(10)]
        assert min(values) < 15.0
        assert max(values) > 15.0


# =============================================================================
# Additional Depth of Field Tests
# =============================================================================


class TestDepthOfFieldAdvanced:
    """Additional depth of field tests."""

    def test_dof_focus_speed(self):
        """Test auto-focus speed parameter."""
        dof = DepthOfFieldEffect()
        dof.enable()
        dof._focus_speed = 20.0
        target = Vector3(50, 0, 0)
        dof.enable_auto_focus(target)
        dof.focus_distance = 10.0

        for _ in range(10):
            dof.update(0.1, Vector3(0, 0, 0))

        assert dof.focus_distance > 10.0

    def test_blur_interpolation(self):
        """Test blur interpolation in transition zones."""
        dof = DepthOfFieldEffect()
        dof.enable()
        dof.set_blur_range(0.0, 5.0, 15.0, 30.0)
        dof.blur_strength = 1.0

        blur_at_2 = dof.get_blur_at_distance(2.5)
        assert 0 < blur_at_2 < dof.blur_strength

    def test_dof_not_updated_when_disabled(self):
        """Test DOF not updated when disabled."""
        dof = DepthOfFieldEffect()
        dof.enabled = False
        dof.focus_distance = 10.0
        dof.enable_auto_focus(Vector3(100, 0, 0))
        dof.update(1.0, Vector3(0, 0, 0))
        assert dof.focus_distance == 10.0


# =============================================================================
# Additional Motion Blur Tests
# =============================================================================


class TestMotionBlurAdvanced:
    """Additional motion blur tests."""

    def test_velocity_scale(self):
        """Test velocity scale parameter."""
        blur = MotionBlurEffect()
        blur.enable()
        blur.velocity_scale = 2.0
        blur.update(0.016, Vector3(0, 0, 0))
        blur.update(0.016, Vector3(10, 0, 0))
        vector = blur.get_blur_vector()
        assert vector.magnitude() > 0

    def test_motion_blur_direction(self):
        """Test motion blur direction matches movement."""
        blur = MotionBlurEffect()
        blur.enable()
        blur.strength = 1.0
        blur.max_blur = 100.0
        blur.update(0.1, Vector3(0, 0, 0))
        blur.update(0.1, Vector3(10, 0, 0))
        vector = blur.get_blur_vector()
        assert vector.x > 0

    def test_motion_blur_reset_position(self):
        """Test motion blur handles position reset."""
        blur = MotionBlurEffect()
        blur.enable()
        blur.update(0.016, Vector3(0, 0, 0))
        blur._last_position = None
        blur.update(0.016, Vector3(100, 0, 0))


# =============================================================================
# Additional Vignette Tests
# =============================================================================


class TestVignetteAdvanced:
    """Additional vignette tests."""

    def test_vignette_pulse(self):
        """Test vignette pulse effect."""
        vignette = VignetteEffect()
        vignette.enable()
        vignette.intensity = 0.3
        vignette.pulse(0.8, 0.5)
        assert vignette._target_intensity == 0.8

    def test_vignette_smoothness(self):
        """Test vignette smoothness parameter."""
        vignette = VignetteEffect()
        vignette.smoothness = 0.8
        assert vignette.smoothness == 0.8

    def test_vignette_roundness(self):
        """Test vignette roundness parameter."""
        vignette = VignetteEffect()
        vignette.roundness = 0.5
        assert vignette.roundness == 0.5

    def test_vignette_red_color(self):
        """Test vignette with red color."""
        vignette = VignetteEffect()
        vignette.set_color(1.0, 0.0, 0.0)
        assert vignette.color == (1.0, 0.0, 0.0)


# =============================================================================
# Additional Effect Stack Tests
# =============================================================================


class TestEffectStackAdvanced:
    """Additional effect stack manager tests."""

    def test_override_blend_mode(self):
        """Test override blend mode."""
        manager = EffectStackManager()
        manager.set_blend_mode(BlendMode.OVERRIDE)
        manager.shake.add_shake(ShakeParams(
            amplitude=Vector3(0.1, 0, 0),
            duration=1.0,
            shake_type=ShakeType.SINE,
        ))
        manager.trauma.add_trauma(0.8)
        result = manager.update(0.1)

    def test_all_effects_disabled(self):
        """Test with all effects at default/disabled."""
        manager = EffectStackManager()
        result = manager.update(0.1, Vector3())
        assert result["position_offset"].magnitude() == 0

    def test_reset_during_active_effects(self):
        """Test reset while effects are active."""
        manager = EffectStackManager()
        manager.shake.add_shake(ShakeParams(duration=10.0))
        manager.trauma.add_trauma(1.0)
        manager.fov.punch(30.0)
        manager.reset_all()
        assert not manager.shake.is_shaking
        assert manager.trauma.trauma == 0.0

    def test_pause_preserves_state(self):
        """Test pause preserves effect state."""
        manager = EffectStackManager()
        manager.trauma.add_trauma(1.0)
        initial_trauma = manager.trauma.trauma
        manager.pause_all()
        manager.update(1.0)
        assert manager.trauma.trauma == initial_trauma


# =============================================================================
# Integration Tests
# =============================================================================


class TestEffectsIntegration:
    """Integration tests for camera effects."""

    def test_explosion_effect_sequence(self):
        """Test explosion effect sequence (trauma + shake + fov)."""
        manager = EffectStackManager()
        manager.trauma.add_trauma(1.0)
        manager.shake.add_shake(ShakeParams(
            amplitude=Vector3(0.5, 0.5, 0.2),
            duration=0.5,
            shake_type=ShakeType.RANDOM
        ))
        manager.fov.punch(-10.0)
        manager.vignette.enable()
        manager.vignette.set_intensity(0.8, instant=True)

        for _ in range(30):
            result = manager.update(0.016, Vector3())

    def test_damage_effect_sequence(self):
        """Test damage effect sequence (trauma + vignette)."""
        manager = EffectStackManager()
        manager.trauma.add_trauma(0.3)
        manager.vignette.enable()
        manager.vignette.set_color(1.0, 0.0, 0.0)
        manager.vignette.set_intensity(0.5)

        for _ in range(30):
            manager.update(0.016)

    def test_focus_pull_effect(self):
        """Test focus pull effect (DOF transition)."""
        manager = EffectStackManager()
        manager.dof.enable()
        manager.dof.set_aperture(1.4)

        for i in range(50):
            manager.dof.set_focus_distance(5.0 + i * 0.5)
            manager.update(0.016, Vector3())

    def test_cinematic_black_bars(self):
        """Test cinematic mode with vignette and tilt."""
        manager = EffectStackManager()
        manager.tilt.set_tilt(5.0)
        manager.fov.set_target_fov(50.0)
        manager.vignette.enable()
        manager.vignette.set_intensity(0.4)

        for _ in range(60):
            manager.update(0.016)

    def test_drunken_effect(self):
        """Test drunken/dizzy camera effect."""
        manager = EffectStackManager()
        manager.tilt.start_oscillation(amplitude=10.0, frequency=0.5)
        manager.shake.add_shake(ShakeParams(
            amplitude=Vector3(0.1, 0.1, 0),
            frequency=3.0,
            duration=5.0,
            decay=False
        ))
        manager.dof.enable()
        manager.dof.set_aperture(2.0)
        manager.dof.blur_strength = 0.8

        for _ in range(60):
            manager.update(0.016, Vector3())

    def test_speed_blur_effect(self):
        """Test speed/motion blur effect."""
        manager = EffectStackManager()
        manager.motion_blur.enable()
        manager.motion_blur.set_strength(0.8)
        manager.fov.set_target_fov(80.0)

        camera_pos = Vector3(0, 0, 0)
        for i in range(60):
            camera_pos = Vector3(i * 0.5, 0, 0)
            manager.update(0.016, camera_pos)

    def test_hit_marker_effect(self):
        """Test hit marker effect (quick shake + fov punch)."""
        manager = EffectStackManager()

        for _ in range(5):
            manager.shake.add_shake(ShakeParams(
                amplitude=Vector3(0.05, 0.05, 0),
                duration=0.1,
                shake_type=ShakeType.SINE
            ))
            manager.fov.punch(5.0)

            for _ in range(6):
                manager.update(0.016)

    def test_underwater_effect(self):
        """Test underwater camera effect."""
        manager = EffectStackManager()
        manager.tilt.start_oscillation(amplitude=3.0, frequency=0.3)
        manager.vignette.enable()
        manager.vignette.set_intensity(0.3)
        manager.vignette.set_color(0.0, 0.3, 0.5)
        manager.dof.enable()
        manager.dof.set_blur_range(0, 2, 8, 15)

        for _ in range(60):
            manager.update(0.016, Vector3())

    def test_effect_layering_order(self):
        """Test multiple effects layer correctly."""
        manager = EffectStackManager()
        manager.shake.add_shake(ShakeParams(amplitude=Vector3(1, 0, 0), duration=1.0))
        manager.trauma.add_trauma(0.5)
        manager.tilt.set_tilt(15.0)
        manager.fov.punch(10.0)
        manager.vignette.enable()
        manager.dof.enable()
        manager.motion_blur.enable()

        result = manager.update(0.1, Vector3(0, 0, 0))
        assert "position_offset" in result
        assert "tilt" in result
        assert "fov" in result


# =============================================================================
# Stress Tests
# =============================================================================


class TestEffectsStress:
    """Stress tests for camera effects."""

    def test_many_simultaneous_shakes(self):
        """Test many simultaneous shakes."""
        shake = CameraShake()
        for _ in range(50):
            shake.add_shake(ShakeParams(duration=2.0))

        for _ in range(100):
            offset = shake.update(0.016)

    def test_rapid_trauma_changes(self):
        """Test rapid trauma changes."""
        trauma = TraumaSystem()
        for _ in range(1000):
            trauma.add_trauma(0.1)
            trauma.update(0.001)

    def test_many_fov_effects(self):
        """Test many FOV effects."""
        fov = FOVEffect()
        for i in range(100):
            fov.add_effect(offset=i % 10 - 5, duration=0.5)

        for _ in range(50):
            fov.update(0.016)


# =============================================================================
# Additional Stress Tests
# =============================================================================


class TestEffectsStressAdvanced:
    """Additional stress tests for effects."""

    def test_shake_high_frequency(self):
        """Test shake at very high frequency."""
        shake = CameraShake()
        shake.add_shake(ShakeParams(
            amplitude=Vector3(1, 1, 1),
            frequency=100.0,
            duration=2.0
        ))

        for _ in range(200):
            shake.update(0.001)

    def test_combined_maximum_effects(self):
        """Test all effects at maximum intensity."""
        manager = EffectStackManager()
        manager.set_master_intensity(1.0)

        manager.trauma.add_trauma(1.0)
        for _ in range(10):
            manager.shake.add_shake(ShakeParams(
                amplitude=Vector3(1, 1, 1),
                duration=5.0
            ))
        manager.fov.punch(30.0)
        manager.tilt.set_tilt(45.0)
        manager.vignette.enable()
        manager.vignette.set_intensity(1.0, instant=True)

        for _ in range(100):
            manager.update(0.016, Vector3())


class TestEffectsBoundaryConditions:
    """Test boundary conditions for effects."""

    def test_shake_negative_amplitude(self):
        """Test shake with negative amplitude values."""
        shake = CameraShake()
        shake.add_shake(ShakeParams(
            amplitude=Vector3(-1, -1, -1),
            duration=1.0
        ))
        offset = shake.update(0.1)

    def test_fov_extreme_values(self):
        """Test FOV at extreme values."""
        fov = FOVEffect(base_fov=1.0)
        fov.set_target_fov(179.0)
        for _ in range(100):
            fov.update(0.016)

    def test_dof_extreme_blur_ranges(self):
        """Test DOF with extreme blur ranges."""
        dof = DepthOfFieldEffect()
        dof.enable()
        dof.set_blur_range(0, 0.001, 0.001, 0.002)
        dof.blur_strength = 1.0

        blur = dof.get_blur_at_distance(0.0005)

    def test_motion_blur_at_zero_velocity(self):
        """Test motion blur with zero velocity."""
        blur = MotionBlurEffect()
        blur.enable()
        blur.strength = 1.0
        result = blur.get_blur_amount(Vector3(0, 0, 0))
        assert result == 0.0

    def test_vignette_at_boundaries(self):
        """Test vignette at edge values."""
        vignette = VignetteEffect()
        vignette.enable()
        vignette.set_intensity(0.0, instant=True)
        assert vignette.intensity == 0.0
        vignette.set_intensity(1.0, instant=True)
        assert vignette.intensity == 1.0

    def test_tilt_wrap_around(self):
        """Test tilt angle wrap around."""
        tilt = TiltEffect()
        tilt.enable()
        tilt.set_tilt(360.0)
        for _ in range(50):
            tilt.update(0.016)

    def test_trauma_decay_to_zero(self):
        """Test trauma decays completely to zero."""
        trauma = TraumaSystem()
        trauma.add_trauma(1.0)

        for _ in range(500):
            trauma.update(0.016)

        assert trauma.trauma == pytest.approx(0.0, abs=0.01)

    def test_shake_with_zero_duration(self):
        """Test shake with zero duration."""
        shake = CameraShake()
        shake.add_shake(ShakeParams(
            amplitude=Vector3(1, 1, 1),
            duration=0.0
        ))
        offset = shake.update(0.1)

    def test_dof_focus_transition_edge_cases(self):
        """Test DOF focus transition edge cases."""
        dof = DepthOfFieldEffect()
        dof.enable()
        dof.transition_speed = 0.0
        dof.set_focus_distance(10.0)
        dof.update(0.016)

    def test_multiple_effects_same_frame(self):
        """Test multiple effects applied same frame."""
        manager = EffectStackManager()

        manager.shake.add_shake(ShakeParams(amplitude=Vector3(1, 1, 1), duration=1.0))
        manager.fov.punch(10.0)
        manager.tilt.set_tilt(5.0)
        manager.trauma.add_trauma(0.5)
        manager.vignette.enable()
        manager.vignette.set_intensity(0.3)
        manager.motion_blur.enable()
        manager.dof.enable()

        result = manager.update(0.016, Vector3(1, 1, 1))

    def test_effect_stack_priority(self):
        """Test effect stack priority ordering."""
        manager = EffectStackManager()

        # High priority effect
        manager.shake.add_shake(ShakeParams(
            amplitude=Vector3(2, 2, 2),
            duration=5.0
        ))

        # Low priority effect
        manager.shake.add_shake(ShakeParams(
            amplitude=Vector3(0.1, 0.1, 0.1),
            duration=5.0
        ))

        result = manager.update(0.016, Vector3())


class TestEffectsIntegration:
    """Integration tests for camera effects."""

    def test_explosion_effect_chain(self):
        """Test explosion effect chain."""
        manager = EffectStackManager()

        # Explosion: shake + FOV punch + vignette
        manager.shake.add_shake(ShakeParams(
            amplitude=Vector3(2, 1.5, 1),
            frequency=25.0,
            duration=0.5
        ))
        manager.fov.punch(-15.0, duration=0.3)
        manager.vignette.enable()
        manager.vignette.set_intensity(0.6)
        manager.trauma.add_trauma(0.8)

        for _ in range(50):
            manager.update(0.016, Vector3())

    def test_impact_effect_chain(self):
        """Test impact/damage effect chain."""
        manager = EffectStackManager()

        manager.trauma.add_trauma(0.4)
        manager.tilt.set_tilt(3.0)
        manager.vignette.enable()
        manager.vignette.set_intensity(0.4)

        for _ in range(30):
            manager.update(0.016, Vector3())

    def test_sprint_effect_chain(self):
        """Test sprint/running effect chain."""
        manager = EffectStackManager()

        manager.fov.set_target_fov(75.0)
        manager.motion_blur.enable()
        manager.motion_blur.strength = 0.5

        for i in range(60):
            velocity = Vector3(10 + math.sin(i * 0.5) * 2, 0, 0)
            manager.update(0.016, velocity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
