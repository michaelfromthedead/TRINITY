"""Doppler Effect Calculation for Spatial Audio.

Implements the Doppler effect with:
- Velocity-based pitch shift
- Configurable exaggeration factor
- Smoothing to prevent artifacts
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from engine.audio.spatial.config import (
    DOPPLER_FACTOR,
    DOPPLER_SMOOTHING_TIME,
    DOPPLER_VELOCITY_THRESHOLD,
    MAX_DOPPLER_SHIFT,
    MIN_DOPPLER_SHIFT,
    SPEED_OF_SOUND,
)
from engine.core.math.vec import Vec3


def calculate_doppler_shift(
    source_pos: Vec3,
    source_velocity: Vec3,
    listener_pos: Vec3,
    listener_velocity: Vec3,
    doppler_factor: float = DOPPLER_FACTOR,
    speed_of_sound: float = SPEED_OF_SOUND
) -> float:
    """Calculate Doppler pitch shift.

    Uses the classical Doppler formula:
    f' = f * (c + v_listener) / (c + v_source)

    where velocities are the components along the source-to-listener direction.
    Positive velocity = moving towards listener, negative = moving away.

    Args:
        source_pos: Source position.
        source_velocity: Source velocity vector.
        listener_pos: Listener position.
        listener_velocity: Listener velocity vector.
        doppler_factor: Exaggeration factor (1.0 = realistic).
        speed_of_sound: Speed of sound in units/second.

    Returns:
        Pitch multiplier (1.0 = no change, >1 = higher pitch, <1 = lower pitch).
    """
    # Direction from source to listener
    direction = listener_pos - source_pos
    distance = direction.length()

    if distance < 0.0001:
        return 1.0

    direction = direction / distance

    # Project velocities onto the direction axis
    # Positive = approaching, negative = receding
    source_approach = -source_velocity.dot(direction)  # Negative because moving towards listener
    listener_approach = listener_velocity.dot(direction)

    # Check velocity threshold
    relative_velocity = abs(source_approach - listener_approach)
    if relative_velocity < DOPPLER_VELOCITY_THRESHOLD:
        return 1.0

    # Apply Doppler formula with factor
    # Scale velocities by Doppler factor for exaggeration
    source_v = source_approach * doppler_factor
    listener_v = listener_approach * doppler_factor

    # Prevent division by zero or negative speed of sound
    effective_source_speed = speed_of_sound + source_v
    if effective_source_speed <= 0.0:
        effective_source_speed = 0.01 * speed_of_sound

    effective_listener_speed = speed_of_sound + listener_v

    # Calculate pitch shift
    shift = effective_listener_speed / effective_source_speed

    # Clamp to prevent extreme values
    return max(MIN_DOPPLER_SHIFT, min(MAX_DOPPLER_SHIFT, shift))


@dataclass
class DopplerState:
    """State for smoothed Doppler processing on a single source."""

    source_id: int = 0
    """Identifier for the source."""

    current_shift: float = 1.0
    """Current pitch shift value."""

    target_shift: float = 1.0
    """Target pitch shift value."""

    last_source_pos: Vec3 = field(default_factory=Vec3.zero)
    """Last known source position."""

    last_listener_pos: Vec3 = field(default_factory=Vec3.zero)
    """Last known listener position."""

    last_source_velocity: Vec3 = field(default_factory=Vec3.zero)
    """Last calculated source velocity."""

    last_listener_velocity: Vec3 = field(default_factory=Vec3.zero)
    """Last calculated listener velocity."""

    smoothing_time: float = DOPPLER_SMOOTHING_TIME
    """Time constant for smoothing (seconds)."""

    first_update: bool = True
    """Whether this is the first update (no previous position)."""

    def reset(self) -> None:
        """Reset state."""
        self.current_shift = 1.0
        self.target_shift = 1.0
        self.last_source_velocity = Vec3.zero()
        self.last_listener_velocity = Vec3.zero()
        self.first_update = True


class DopplerProcessor:
    """Processor for Doppler effect with smoothing.

    Handles velocity estimation from position changes and
    smooth interpolation to prevent audio artifacts.
    """

    def __init__(
        self,
        doppler_factor: float = DOPPLER_FACTOR,
        speed_of_sound: float = SPEED_OF_SOUND,
        smoothing_time: float = DOPPLER_SMOOTHING_TIME
    ) -> None:
        self._doppler_factor = doppler_factor
        self._speed_of_sound = speed_of_sound
        self._smoothing_time = smoothing_time
        self._states: dict[int, DopplerState] = {}

    @property
    def doppler_factor(self) -> float:
        """Get Doppler exaggeration factor."""
        return self._doppler_factor

    @doppler_factor.setter
    def doppler_factor(self, value: float) -> None:
        self._doppler_factor = max(0.0, value)

    @property
    def speed_of_sound(self) -> float:
        """Get speed of sound."""
        return self._speed_of_sound

    @speed_of_sound.setter
    def speed_of_sound(self, value: float) -> None:
        self._speed_of_sound = max(1.0, value)

    @property
    def smoothing_time(self) -> float:
        """Get smoothing time constant."""
        return self._smoothing_time

    @smoothing_time.setter
    def smoothing_time(self, value: float) -> None:
        self._smoothing_time = max(0.0, value)

    def get_or_create_state(self, source_id: int) -> DopplerState:
        """Get or create state for a source."""
        if source_id not in self._states:
            self._states[source_id] = DopplerState(source_id=source_id, smoothing_time=self._smoothing_time)
        return self._states[source_id]

    def remove_state(self, source_id: int) -> None:
        """Remove state for a source."""
        self._states.pop(source_id, None)

    def clear_states(self) -> None:
        """Clear all states."""
        self._states.clear()

    def update(
        self,
        source_id: int,
        source_pos: Vec3,
        listener_pos: Vec3,
        dt: float,
        source_velocity: Optional[Vec3] = None,
        listener_velocity: Optional[Vec3] = None
    ) -> float:
        """Update Doppler state and return current pitch shift.

        Args:
            source_id: Source identifier.
            source_pos: Current source position.
            listener_pos: Current listener position.
            dt: Time delta since last update (seconds).
            source_velocity: Optional explicit source velocity.
            listener_velocity: Optional explicit listener velocity.

        Returns:
            Current pitch shift multiplier.
        """
        state = self.get_or_create_state(source_id)

        if dt <= 0.0:
            return state.current_shift

        # Calculate or use provided velocities
        if source_velocity is not None:
            src_vel = source_velocity
        elif state.first_update:
            src_vel = Vec3.zero()
        else:
            # Estimate velocity from position change
            src_vel = (source_pos - state.last_source_pos) / dt

        if listener_velocity is not None:
            lst_vel = listener_velocity
        elif state.first_update:
            lst_vel = Vec3.zero()
        else:
            lst_vel = (listener_pos - state.last_listener_pos) / dt

        # Calculate target Doppler shift
        state.target_shift = calculate_doppler_shift(
            source_pos, src_vel,
            listener_pos, lst_vel,
            self._doppler_factor,
            self._speed_of_sound
        )

        # Smooth towards target
        if self._smoothing_time > 0.0:
            # Exponential smoothing
            alpha = 1.0 - math.exp(-dt / self._smoothing_time)
            state.current_shift += alpha * (state.target_shift - state.current_shift)
        else:
            state.current_shift = state.target_shift

        # Update state for next frame
        state.last_source_pos = source_pos
        state.last_listener_pos = listener_pos
        state.last_source_velocity = src_vel
        state.last_listener_velocity = lst_vel
        state.first_update = False

        return state.current_shift

    def get_current_shift(self, source_id: int) -> float:
        """Get current pitch shift for a source without updating."""
        state = self._states.get(source_id)
        return state.current_shift if state else 1.0


def estimate_arrival_time(
    source_pos: Vec3,
    source_velocity: Vec3,
    listener_pos: Vec3,
    listener_velocity: Vec3,
    speed_of_sound: float = SPEED_OF_SOUND
) -> Optional[float]:
    """Estimate time for sound to arrive at listener.

    Accounts for the finite speed of sound and relative motion.

    Args:
        source_pos: Source position at emission time.
        source_velocity: Source velocity.
        listener_pos: Listener position.
        listener_velocity: Listener velocity.
        speed_of_sound: Speed of sound.

    Returns:
        Estimated arrival time in seconds, or None if sound will never arrive.
    """
    # Current distance
    relative_pos = listener_pos - source_pos
    distance = relative_pos.length()

    if distance < 0.0001:
        return 0.0

    # Direction from source to listener
    direction = relative_pos / distance

    # Relative velocity along the direction
    relative_velocity = listener_velocity - source_velocity
    approach_speed = -relative_velocity.dot(direction)

    # Effective speed of sound propagation
    effective_speed = speed_of_sound - approach_speed

    if effective_speed <= 0.0:
        # Sound will never arrive (supersonic separation)
        return None

    return distance / effective_speed


@dataclass
class DopplerConfig:
    """Configuration for Doppler effect processing."""

    enabled: bool = True
    """Whether Doppler effect is enabled."""

    factor: float = DOPPLER_FACTOR
    """Exaggeration factor (1.0 = realistic)."""

    speed_of_sound: float = SPEED_OF_SOUND
    """Speed of sound in units/second."""

    smoothing_time: float = DOPPLER_SMOOTHING_TIME
    """Smoothing time constant to prevent artifacts."""

    min_shift: float = MIN_DOPPLER_SHIFT
    """Minimum allowed pitch shift."""

    max_shift: float = MAX_DOPPLER_SHIFT
    """Maximum allowed pitch shift."""

    velocity_threshold: float = DOPPLER_VELOCITY_THRESHOLD
    """Minimum velocity to apply Doppler."""

    def create_processor(self) -> DopplerProcessor:
        """Create a processor with this configuration."""
        return DopplerProcessor(
            doppler_factor=self.factor,
            speed_of_sound=self.speed_of_sound,
            smoothing_time=self.smoothing_time
        )


# Preset configurations
DOPPLER_PRESETS = {
    "realistic": DopplerConfig(factor=1.0, smoothing_time=0.05),
    "exaggerated": DopplerConfig(factor=2.0, smoothing_time=0.03),
    "subtle": DopplerConfig(factor=0.5, smoothing_time=0.1),
    "arcade": DopplerConfig(factor=3.0, smoothing_time=0.01),
    "disabled": DopplerConfig(enabled=False),
    "underwater": DopplerConfig(factor=1.0, speed_of_sound=1500.0, smoothing_time=0.1),
}


def get_doppler_preset(name: str) -> Optional[DopplerConfig]:
    """Get a Doppler configuration preset by name."""
    return DOPPLER_PRESETS.get(name.lower())
