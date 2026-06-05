"""Material Animation System: Time-based parameter modulation.

T-MAT-5.5: Material Animation System

This module provides:
    - TimeUniforms: GPU uniform buffer for time data
    - TimeContext: Python accessor for time in material DSL
    - AnimationCurve: Procedural animation functions
    - MaterialAnimator: Animation controller for materials

The animation system integrates with the material DSL to provide time-based
modulation of material parameters like UV coordinates, colors, and emission.

Example usage::

    from trinity.materials import Material, MaterialMeta, surface
    from trinity.materials import SurfaceContext, SurfaceOutput
    from trinity.materials.animation import AnimationCurve

    class PulsingMaterial(Material, metaclass=MaterialMeta):
        @surface
        def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
            t = ctx.time()
            pulse = AnimationCurve.sin_wave_01(t, frequency=2.0)
            out.emissive = Vec3(1.0, 0.5, 0.0) * pulse
            out.base_color = Vec3(0.8, 0.3, 0.1)
            out.roughness = 0.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple, TypeVar

__all__ = [
    # Core types
    "TimeUniforms",
    "TimeContext",
    # Animation curves
    "AnimationCurve",
    "WaveType",
    "EasingType",
    # Material animator
    "MaterialAnimator",
    "AnimationState",
    "AnimatedParameter",
    # WGSL access
    "get_time_wgsl",
    "TIME_WGSL",
    # Reference values
    "ANIMATION_REFERENCE_VALUES",
]

# Mathematical constants
TAU = 2.0 * math.pi
PI = math.pi


# =============================================================================
# WGSL SOURCE ACCESS
# =============================================================================


def get_time_wgsl() -> str:
    """Load the time animation WGSL source code.

    Returns:
        WGSL source code for time uniforms and animation functions.
    """
    wgsl_path = Path(__file__).parent / "wgsl" / "time.wgsl"
    return wgsl_path.read_text()


# Lazy load for module-level constant
_time_wgsl_cache: Optional[str] = None


def _get_cached_time_wgsl() -> str:
    """Get cached time WGSL source."""
    global _time_wgsl_cache
    if _time_wgsl_cache is None:
        _time_wgsl_cache = get_time_wgsl()
    return _time_wgsl_cache


# Module-level constant (lazy property simulation)
class _TimeWGSLProperty:
    """Lazy loader for TIME_WGSL constant."""

    def __str__(self) -> str:
        return _get_cached_time_wgsl()

    def __repr__(self) -> str:
        return f"<TIME_WGSL: {len(_get_cached_time_wgsl())} chars>"


TIME_WGSL = _TimeWGSLProperty()


# =============================================================================
# TIME UNIFORMS
# =============================================================================


@dataclass
class TimeUniforms:
    """Time-related uniform values for GPU shaders.

    This struct mirrors the WGSL TimeUniforms struct and is updated
    each frame by the renderer.

    Attributes:
        elapsed_seconds: Total elapsed time since animation start
        delta_time: Time since last frame in seconds
        frame_count: Frame counter (wraps at 2^32)

    The GPU buffer layout (16-byte aligned):
        offset 0:  elapsed_seconds (f32)
        offset 4:  delta_time (f32)
        offset 8:  frame_count (u32)
        offset 12: padding (u32)
    """

    elapsed_seconds: float = 0.0
    delta_time: float = 0.0
    frame_count: int = 0

    def update(self, dt: float) -> None:
        """Update time values for next frame.

        Args:
            dt: Delta time since last frame in seconds
        """
        self.delta_time = dt
        self.elapsed_seconds += dt
        self.frame_count = (self.frame_count + 1) % (2**32)

    def reset(self) -> None:
        """Reset time values to initial state."""
        self.elapsed_seconds = 0.0
        self.delta_time = 0.0
        self.frame_count = 0

    def to_bytes(self) -> bytes:
        """Serialize to GPU buffer bytes (16-byte aligned).

        Returns:
            16-byte buffer for GPU uniform upload.
        """
        import struct

        return struct.pack(
            "<ffII",  # Little-endian: 2 floats, 2 unsigned ints
            self.elapsed_seconds,
            self.delta_time,
            self.frame_count,
            0,  # Padding
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "TimeUniforms":
        """Deserialize from GPU buffer bytes.

        Args:
            data: 16-byte buffer from GPU.

        Returns:
            TimeUniforms instance.
        """
        import struct

        elapsed, delta, frame, _ = struct.unpack("<ffII", data)
        return cls(elapsed_seconds=elapsed, delta_time=delta, frame_count=frame)


# =============================================================================
# TIME CONTEXT
# =============================================================================


@dataclass
class TimeContext:
    """Context for accessing time values in material shaders.

    This class provides the Python-side interface for time access in
    material surface shaders. Methods map to WGSL uniform reads.

    Example::

        def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
            t = ctx.time()  # Maps to uniforms.elapsed_seconds
            dt = ctx.delta()  # Maps to uniforms.delta_time
    """

    _uniforms: TimeUniforms = field(default_factory=TimeUniforms)

    def time(self) -> float:
        """Get elapsed time in seconds.

        Maps to WGSL: uniforms.elapsed_seconds
        """
        return self._uniforms.elapsed_seconds

    def delta(self) -> float:
        """Get delta time since last frame.

        Maps to WGSL: uniforms.delta_time
        """
        return self._uniforms.delta_time

    def frame(self) -> int:
        """Get current frame count.

        Maps to WGSL: uniforms.frame_count
        """
        return self._uniforms.frame_count

    def seconds_mod(self, period: float) -> float:
        """Get elapsed time modulo a period.

        Useful for looping animations.

        Args:
            period: Loop period in seconds

        Returns:
            Time modulo period
        """
        if period <= 0:
            return 0.0
        return self._uniforms.elapsed_seconds % period

    def phase(self, period: float) -> float:
        """Get animation phase in [0, 1] range.

        Args:
            period: Loop period in seconds

        Returns:
            Phase in [0, 1]
        """
        if period <= 0:
            return 0.0
        return (self._uniforms.elapsed_seconds % period) / period


# =============================================================================
# WAVE TYPES
# =============================================================================


class WaveType(Enum):
    """Types of wave functions for animation."""

    SINE = "sine"
    COSINE = "cosine"
    SAWTOOTH = "sawtooth"
    SAWTOOTH_REVERSE = "sawtooth_reverse"
    TRIANGLE = "triangle"
    SQUARE = "square"
    PULSE = "pulse"


class EasingType(Enum):
    """Types of easing functions for interpolation."""

    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    ELASTIC = "elastic"
    BOUNCE = "bounce"
    BACK = "back"


# =============================================================================
# ANIMATION CURVE
# =============================================================================


class AnimationCurve:
    """Static methods for procedural animation curves.

    These functions generate time-varying values that can be used
    to animate material parameters. Each function has a corresponding
    WGSL implementation in time.wgsl.

    All wave functions that produce [-1, 1] output have _01 variants
    that produce [0, 1] output for convenience.
    """

    # =========================================================================
    # Basic Wave Functions
    # =========================================================================

    @staticmethod
    def sin_wave(t: float, frequency: float = 1.0, phase: float = 0.0) -> float:
        """Sine wave oscillation in [-1, 1].

        Args:
            t: Time in seconds
            frequency: Oscillation frequency in Hz
            phase: Phase offset in radians

        Returns:
            Sine wave value in [-1, 1]
        """
        return math.sin(t * frequency * TAU + phase)

    @staticmethod
    def sin_wave_01(t: float, frequency: float = 1.0, phase: float = 0.0) -> float:
        """Normalized sine wave in [0, 1].

        Args:
            t: Time in seconds
            frequency: Oscillation frequency in Hz
            phase: Phase offset in radians

        Returns:
            Sine wave value in [0, 1]
        """
        return AnimationCurve.sin_wave(t, frequency, phase) * 0.5 + 0.5

    @staticmethod
    def cos_wave(t: float, frequency: float = 1.0, phase: float = 0.0) -> float:
        """Cosine wave oscillation in [-1, 1].

        Args:
            t: Time in seconds
            frequency: Oscillation frequency in Hz
            phase: Phase offset in radians

        Returns:
            Cosine wave value in [-1, 1]
        """
        return math.cos(t * frequency * TAU + phase)

    @staticmethod
    def cos_wave_01(t: float, frequency: float = 1.0, phase: float = 0.0) -> float:
        """Normalized cosine wave in [0, 1].

        Args:
            t: Time in seconds
            frequency: Oscillation frequency in Hz
            phase: Phase offset in radians

        Returns:
            Cosine wave value in [0, 1]
        """
        return AnimationCurve.cos_wave(t, frequency, phase) * 0.5 + 0.5

    @staticmethod
    def sawtooth(t: float, period: float = 1.0) -> float:
        """Sawtooth wave (linear ramp) in [0, 1].

        Args:
            t: Time in seconds
            period: Period of the wave in seconds

        Returns:
            Sawtooth value in [0, 1]
        """
        if period <= 0:
            return 0.0
        return (t / period) % 1.0

    @staticmethod
    def sawtooth_reverse(t: float, period: float = 1.0) -> float:
        """Reverse sawtooth wave in [0, 1].

        Args:
            t: Time in seconds
            period: Period of the wave in seconds

        Returns:
            Reverse sawtooth value in [0, 1]
        """
        return 1.0 - AnimationCurve.sawtooth(t, period)

    @staticmethod
    def triangle_wave(t: float, period: float = 1.0) -> float:
        """Triangle wave in [0, 1].

        Args:
            t: Time in seconds
            period: Period of the wave in seconds

        Returns:
            Triangle wave value in [0, 1]
        """
        saw = AnimationCurve.sawtooth(t, period)
        return 1.0 - abs(saw * 2.0 - 1.0)

    @staticmethod
    def pulse(t: float, period: float = 1.0, duty: float = 0.5) -> float:
        """Pulse/square wave with controllable duty cycle.

        Args:
            t: Time in seconds
            period: Period of the wave in seconds
            duty: Duty cycle in [0, 1]

        Returns:
            0.0 or 1.0
        """
        if period <= 0:
            return 0.0
        phase = (t / period) % 1.0
        return 1.0 if phase < duty else 0.0

    @staticmethod
    def smooth_pulse(
        t: float, period: float = 1.0, attack: float = 0.1, release: float = 0.1
    ) -> float:
        """Smooth pulse with attack and release.

        Args:
            t: Time in seconds
            period: Period of the wave in seconds
            attack: Attack time as fraction of period [0, 0.5]
            release: Release time as fraction of period [0, 0.5]

        Returns:
            Smooth pulse value in [0, 1]
        """
        if period <= 0:
            return 0.0
        phase = (t / period) % 1.0
        hold_end = 1.0 - release

        if phase < attack:
            # Attack phase
            return AnimationCurve._smoothstep(0.0, attack, phase)
        elif phase < hold_end:
            # Hold phase
            return 1.0
        else:
            # Release phase
            return 1.0 - AnimationCurve._smoothstep(hold_end, 1.0, phase)

    # =========================================================================
    # Noise-Based Animation
    # =========================================================================

    @staticmethod
    def _hash(p: float) -> float:
        """Simple hash function for procedural noise."""
        x = (p * 0.1031) % 1.0
        x = x * (x + 33.33)
        return ((x + x) * x) % 1.0

    @staticmethod
    def noise_anim(t: float, frequency: float = 1.0, octaves: int = 1) -> float:
        """Noise-based animation value.

        Args:
            t: Time in seconds
            frequency: Base frequency for noise
            octaves: Number of noise octaves (1-4)

        Returns:
            Noise value in [0, 1]
        """
        value = 0.0
        amplitude = 1.0
        total_amplitude = 0.0
        freq = frequency

        for _ in range(min(octaves, 4)):
            floor_t = math.floor(t * freq)
            fract_t = (t * freq) % 1.0
            # Smooth interpolation
            smooth_t = fract_t * fract_t * (3.0 - 2.0 * fract_t)
            a = AnimationCurve._hash(floor_t)
            b = AnimationCurve._hash(floor_t + 1.0)
            value += (a + (b - a) * smooth_t) * amplitude
            total_amplitude += amplitude
            amplitude *= 0.5
            freq *= 2.0

        return value / total_amplitude if total_amplitude > 0 else 0.0

    @staticmethod
    def flicker(t: float, intensity: float = 0.5, speed: float = 1.0) -> float:
        """Flicker animation using high-frequency noise.

        Args:
            t: Time in seconds
            intensity: Flicker intensity [0, 1]
            speed: Flicker speed multiplier

        Returns:
            Flicker value in [0, 1]
        """
        noise = AnimationCurve.noise_anim(t, speed * 10.0, 2)
        return 1.0 - intensity + noise * intensity

    # =========================================================================
    # Easing Functions
    # =========================================================================

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, x: float) -> float:
        """Smoothstep interpolation."""
        if edge1 == edge0:
            return 0.0 if x < edge0 else 1.0
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def ease_in(t: float) -> float:
        """Quadratic ease-in.

        Args:
            t: Linear time [0, 1]

        Returns:
            Eased value [0, 1]
        """
        return t * t

    @staticmethod
    def ease_out(t: float) -> float:
        """Quadratic ease-out.

        Args:
            t: Linear time [0, 1]

        Returns:
            Eased value [0, 1]
        """
        return t * (2.0 - t)

    @staticmethod
    def ease_in_out(t: float) -> float:
        """Smoothstep ease-in-out.

        Args:
            t: Linear time [0, 1]

        Returns:
            Eased value [0, 1]
        """
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def ease_elastic(t: float, amplitude: float = 1.0, period: float = 0.3) -> float:
        """Elastic easing with overshoot.

        Args:
            t: Linear time [0, 1]
            amplitude: Overshoot amplitude
            period: Oscillation period

        Returns:
            Elastic value (may exceed [0, 1])
        """
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0

        s = period / TAU * math.asin(1.0 / amplitude) if amplitude >= 1.0 else period / 4.0
        return amplitude * math.pow(2.0, -10.0 * t) * math.sin((t - s) * TAU / period) + 1.0

    @staticmethod
    def ease_bounce(t: float) -> float:
        """Bounce easing.

        Args:
            t: Linear time [0, 1]

        Returns:
            Bouncing value [0, 1+]
        """
        if t < 1.0 / 2.75:
            return 7.5625 * t * t
        elif t < 2.0 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375

    # =========================================================================
    # UV Animation
    # =========================================================================

    @staticmethod
    def animate_uv_scroll(
        uv: Tuple[float, float], speed: Tuple[float, float], t: float
    ) -> Tuple[float, float]:
        """Animate UV with linear scrolling.

        Args:
            uv: Original UV coordinates
            speed: Scroll speed per axis
            t: Time in seconds

        Returns:
            Animated UV coordinates (wrapped to [0, 1])
        """
        return ((uv[0] + speed[0] * t) % 1.0, (uv[1] + speed[1] * t) % 1.0)

    @staticmethod
    def animate_uv_rotate(
        uv: Tuple[float, float], speed: float, t: float
    ) -> Tuple[float, float]:
        """Animate UV with rotation around center.

        Args:
            uv: Original UV coordinates
            speed: Rotation speed in radians per second
            t: Time in seconds

        Returns:
            Rotated UV coordinates
        """
        centered = (uv[0] - 0.5, uv[1] - 0.5)
        angle = speed * t
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated = (
            centered[0] * cos_a - centered[1] * sin_a,
            centered[0] * sin_a + centered[1] * cos_a,
        )
        return (rotated[0] + 0.5, rotated[1] + 0.5)

    @staticmethod
    def animate_uv_oscillate(
        uv: Tuple[float, float],
        amplitude: Tuple[float, float],
        frequency: float,
        t: float,
    ) -> Tuple[float, float]:
        """Animate UV with oscillating offset.

        Args:
            uv: Original UV coordinates
            amplitude: Max offset per axis
            frequency: Oscillation frequency in Hz
            t: Time in seconds

        Returns:
            Oscillating UV coordinates
        """
        offset_x = amplitude[0] * math.sin(t * frequency * TAU)
        offset_y = amplitude[1] * math.sin(t * frequency * TAU)
        return (uv[0] + offset_x, uv[1] + offset_y)

    # =========================================================================
    # Color Animation
    # =========================================================================

    @staticmethod
    def animate_color_pulse(
        color: Tuple[float, float, float],
        frequency: float,
        intensity: float,
        t: float,
    ) -> Tuple[float, float, float]:
        """Animate color with intensity pulse.

        Args:
            color: Base RGB color
            frequency: Pulse frequency in Hz
            intensity: Pulse intensity [0, 1]
            t: Time in seconds

        Returns:
            Pulsing RGB color
        """
        factor = AnimationCurve.sin_wave_01(t, frequency) * intensity + (1.0 - intensity)
        return (color[0] * factor, color[1] * factor, color[2] * factor)

    @staticmethod
    def animate_color_flicker(
        color: Tuple[float, float, float], intensity: float, speed: float, t: float
    ) -> Tuple[float, float, float]:
        """Animate color with random flicker.

        Args:
            color: Base RGB color
            intensity: Flicker intensity [0, 1]
            speed: Flicker speed multiplier
            t: Time in seconds

        Returns:
            Flickering RGB color
        """
        factor = AnimationCurve.flicker(t, intensity, speed)
        return (color[0] * factor, color[1] * factor, color[2] * factor)

    # =========================================================================
    # Emission Animation
    # =========================================================================

    @staticmethod
    def animate_emission_pulse(
        base_emission: Tuple[float, float, float],
        frequency: float,
        min_intensity: float,
        max_intensity: float,
        t: float,
    ) -> Tuple[float, float, float]:
        """Animate emission with pulsing effect.

        Args:
            base_emission: Base emission RGB
            frequency: Pulse frequency in Hz
            min_intensity: Minimum intensity
            max_intensity: Maximum intensity
            t: Time in seconds

        Returns:
            Pulsing emission RGB
        """
        factor = min_intensity + (max_intensity - min_intensity) * AnimationCurve.sin_wave_01(
            t, frequency
        )
        return (
            base_emission[0] * factor,
            base_emission[1] * factor,
            base_emission[2] * factor,
        )

    @staticmethod
    def animate_emission_flicker(
        base_emission: Tuple[float, float, float],
        frequency: float,
        intensity: float,
        t: float,
    ) -> Tuple[float, float, float]:
        """Animate emission with flickering effect.

        Args:
            base_emission: Base emission RGB
            frequency: Base flicker frequency
            intensity: Flicker intensity [0, 1]
            t: Time in seconds

        Returns:
            Flickering emission RGB
        """
        # Multi-frequency flicker for natural look
        f1 = AnimationCurve.sin_wave_01(t, frequency)
        f2 = AnimationCurve.sin_wave_01(t, frequency * 1.7, 0.5)
        f3 = AnimationCurve.noise_anim(t, frequency * 3.0, 2)

        combined = f1 * 0.4 + f2 * 0.3 + f3 * 0.3
        factor = 1.0 - intensity + combined * intensity

        return (
            base_emission[0] * factor,
            base_emission[1] * factor,
            base_emission[2] * factor,
        )


# =============================================================================
# ANIMATED PARAMETER
# =============================================================================


@dataclass
class AnimatedParameter:
    """Configuration for an animated material parameter.

    Describes how a single parameter should be animated over time.

    Attributes:
        name: Parameter name (matches SurfaceOutput attribute)
        wave_type: Type of wave function
        frequency: Animation frequency in Hz
        amplitude: Animation amplitude (parameter-specific)
        offset: Value offset (center of oscillation)
        phase: Phase offset in radians
    """

    name: str
    wave_type: WaveType = WaveType.SINE
    frequency: float = 1.0
    amplitude: float = 1.0
    offset: float = 0.0
    phase: float = 0.0

    def evaluate(self, t: float) -> float:
        """Evaluate the animated parameter at time t.

        Args:
            t: Time in seconds

        Returns:
            Animated parameter value
        """
        wave_value: float
        if self.wave_type == WaveType.SINE:
            wave_value = AnimationCurve.sin_wave(t, self.frequency, self.phase)
        elif self.wave_type == WaveType.COSINE:
            wave_value = AnimationCurve.cos_wave(t, self.frequency, self.phase)
        elif self.wave_type == WaveType.SAWTOOTH:
            wave_value = AnimationCurve.sawtooth(t, 1.0 / self.frequency) * 2.0 - 1.0
        elif self.wave_type == WaveType.SAWTOOTH_REVERSE:
            wave_value = AnimationCurve.sawtooth_reverse(t, 1.0 / self.frequency) * 2.0 - 1.0
        elif self.wave_type == WaveType.TRIANGLE:
            wave_value = AnimationCurve.triangle_wave(t, 1.0 / self.frequency) * 2.0 - 1.0
        elif self.wave_type == WaveType.SQUARE:
            wave_value = AnimationCurve.pulse(t, 1.0 / self.frequency, 0.5) * 2.0 - 1.0
        elif self.wave_type == WaveType.PULSE:
            wave_value = AnimationCurve.pulse(t, 1.0 / self.frequency, 0.25) * 2.0 - 1.0
        else:
            wave_value = 0.0

        return self.offset + wave_value * self.amplitude


# =============================================================================
# ANIMATION STATE
# =============================================================================


@dataclass
class AnimationState:
    """State for a material animation.

    Tracks the current state of an animation including time,
    playing status, and loop configuration.

    Attributes:
        time_uniforms: Current time uniform values
        playing: Whether the animation is currently playing
        loop: Whether the animation should loop
        speed: Playback speed multiplier
        duration: Total animation duration (0 = infinite)
    """

    time_uniforms: TimeUniforms = field(default_factory=TimeUniforms)
    playing: bool = True
    loop: bool = True
    speed: float = 1.0
    duration: float = 0.0  # 0 = infinite

    def update(self, dt: float) -> bool:
        """Update animation state.

        Args:
            dt: Delta time since last update

        Returns:
            True if animation is still active, False if ended
        """
        if not self.playing:
            return True

        effective_dt = dt * self.speed
        self.time_uniforms.update(effective_dt)

        # Check duration
        if self.duration > 0 and self.time_uniforms.elapsed_seconds >= self.duration:
            if self.loop:
                self.time_uniforms.elapsed_seconds %= self.duration
            else:
                self.playing = False
                return False

        return True

    def reset(self) -> None:
        """Reset animation to initial state."""
        self.time_uniforms.reset()
        self.playing = True

    def seek(self, t: float) -> None:
        """Seek to a specific time.

        Args:
            t: Time to seek to in seconds
        """
        self.time_uniforms.elapsed_seconds = t

    @property
    def time(self) -> float:
        """Get current animation time."""
        return self.time_uniforms.elapsed_seconds


# =============================================================================
# MATERIAL ANIMATOR
# =============================================================================


class MaterialAnimator:
    """Controller for material animations.

    Manages multiple animated parameters and provides a unified
    interface for updating and querying animation state.

    Example::

        animator = MaterialAnimator()
        animator.add_parameter(AnimatedParameter(
            name="emissive",
            wave_type=WaveType.SINE,
            frequency=2.0,
            amplitude=0.5,
            offset=0.5,
        ))

        # In render loop:
        animator.update(delta_time)
        emissive_value = animator.get_value("emissive")
    """

    def __init__(self) -> None:
        """Initialize the material animator."""
        self._state = AnimationState()
        self._parameters: dict[str, AnimatedParameter] = {}
        self._callbacks: list[Callable[[float], None]] = []

    @property
    def state(self) -> AnimationState:
        """Get the animation state."""
        return self._state

    @property
    def time(self) -> float:
        """Get current animation time."""
        return self._state.time

    @property
    def time_uniforms(self) -> TimeUniforms:
        """Get time uniform values for GPU upload."""
        return self._state.time_uniforms

    def add_parameter(self, param: AnimatedParameter) -> None:
        """Add an animated parameter.

        Args:
            param: Parameter configuration
        """
        self._parameters[param.name] = param

    def remove_parameter(self, name: str) -> None:
        """Remove an animated parameter.

        Args:
            name: Parameter name to remove
        """
        self._parameters.pop(name, None)

    def get_parameter(self, name: str) -> Optional[AnimatedParameter]:
        """Get an animated parameter by name.

        Args:
            name: Parameter name

        Returns:
            Parameter configuration or None
        """
        return self._parameters.get(name)

    def get_value(self, name: str) -> Optional[float]:
        """Get the current value of an animated parameter.

        Args:
            name: Parameter name

        Returns:
            Current value or None if parameter not found
        """
        param = self._parameters.get(name)
        if param is None:
            return None
        return param.evaluate(self._state.time)

    def get_all_values(self) -> dict[str, float]:
        """Get all animated parameter values.

        Returns:
            Dict mapping parameter names to current values
        """
        return {name: param.evaluate(self._state.time) for name, param in self._parameters.items()}

    def add_callback(self, callback: Callable[[float], None]) -> None:
        """Add a callback to be called on each update.

        Args:
            callback: Function taking current time as argument
        """
        self._callbacks.append(callback)

    def update(self, dt: float) -> bool:
        """Update the animation.

        Args:
            dt: Delta time since last update

        Returns:
            True if animation is active, False if ended
        """
        active = self._state.update(dt)

        # Call callbacks
        for callback in self._callbacks:
            callback(self._state.time)

        return active

    def play(self) -> None:
        """Start or resume playback."""
        self._state.playing = True

    def pause(self) -> None:
        """Pause playback."""
        self._state.playing = False

    def stop(self) -> None:
        """Stop playback and reset."""
        self._state.playing = False
        self._state.reset()

    def reset(self) -> None:
        """Reset animation to start."""
        self._state.reset()

    def seek(self, t: float) -> None:
        """Seek to a specific time.

        Args:
            t: Time in seconds
        """
        self._state.seek(t)

    def set_speed(self, speed: float) -> None:
        """Set playback speed.

        Args:
            speed: Speed multiplier (1.0 = normal)
        """
        self._state.speed = speed

    def set_loop(self, loop: bool) -> None:
        """Set looping mode.

        Args:
            loop: Whether to loop
        """
        self._state.loop = loop

    def set_duration(self, duration: float) -> None:
        """Set animation duration.

        Args:
            duration: Duration in seconds (0 = infinite)
        """
        self._state.duration = duration


# =============================================================================
# REFERENCE VALUES FOR TESTING
# =============================================================================

ANIMATION_REFERENCE_VALUES = {
    "sin_wave": [
        {"t": 0.0, "frequency": 1.0, "phase": 0.0, "expected": 0.0, "tolerance": 1e-6},
        {"t": 0.25, "frequency": 1.0, "phase": 0.0, "expected": 1.0, "tolerance": 1e-6},
        {"t": 0.5, "frequency": 1.0, "phase": 0.0, "expected": 0.0, "tolerance": 1e-6},
        {"t": 0.75, "frequency": 1.0, "phase": 0.0, "expected": -1.0, "tolerance": 1e-6},
        {"t": 0.0, "frequency": 2.0, "phase": 0.0, "expected": 0.0, "tolerance": 1e-6},
        {"t": 0.125, "frequency": 2.0, "phase": 0.0, "expected": 1.0, "tolerance": 1e-6},
    ],
    "sin_wave_01": [
        {"t": 0.0, "frequency": 1.0, "phase": 0.0, "expected": 0.5, "tolerance": 1e-6},
        {"t": 0.25, "frequency": 1.0, "phase": 0.0, "expected": 1.0, "tolerance": 1e-6},
        {"t": 0.5, "frequency": 1.0, "phase": 0.0, "expected": 0.5, "tolerance": 1e-6},
        {"t": 0.75, "frequency": 1.0, "phase": 0.0, "expected": 0.0, "tolerance": 1e-6},
    ],
    "sawtooth": [
        {"t": 0.0, "period": 1.0, "expected": 0.0, "tolerance": 1e-6},
        {"t": 0.25, "period": 1.0, "expected": 0.25, "tolerance": 1e-6},
        {"t": 0.5, "period": 1.0, "expected": 0.5, "tolerance": 1e-6},
        {"t": 0.75, "period": 1.0, "expected": 0.75, "tolerance": 1e-6},
        {"t": 1.0, "period": 1.0, "expected": 0.0, "tolerance": 1e-6},
        {"t": 0.5, "period": 2.0, "expected": 0.25, "tolerance": 1e-6},
    ],
    "triangle_wave": [
        {"t": 0.0, "period": 1.0, "expected": 0.0, "tolerance": 1e-6},
        {"t": 0.25, "period": 1.0, "expected": 0.5, "tolerance": 1e-6},
        {"t": 0.5, "period": 1.0, "expected": 1.0, "tolerance": 1e-6},
        {"t": 0.75, "period": 1.0, "expected": 0.5, "tolerance": 1e-6},
    ],
    "pulse": [
        {"t": 0.0, "period": 1.0, "duty": 0.5, "expected": 1.0, "tolerance": 1e-6},
        {"t": 0.25, "period": 1.0, "duty": 0.5, "expected": 1.0, "tolerance": 1e-6},
        {"t": 0.5, "period": 1.0, "duty": 0.5, "expected": 0.0, "tolerance": 1e-6},
        {"t": 0.75, "period": 1.0, "duty": 0.5, "expected": 0.0, "tolerance": 1e-6},
    ],
    "ease_in_out": [
        {"t": 0.0, "expected": 0.0, "tolerance": 1e-6},
        {"t": 0.5, "expected": 0.5, "tolerance": 1e-6},
        {"t": 1.0, "expected": 1.0, "tolerance": 1e-6},
    ],
    "uv_scroll": [
        {
            "uv": (0.0, 0.0),
            "speed": (0.5, 0.0),
            "t": 1.0,
            "expected": (0.5, 0.0),
            "tolerance": 1e-6,
        },
        {
            "uv": (0.5, 0.5),
            "speed": (1.0, 1.0),
            "t": 0.5,
            "expected": (0.0, 0.0),
            "tolerance": 1e-6,
        },
    ],
}
