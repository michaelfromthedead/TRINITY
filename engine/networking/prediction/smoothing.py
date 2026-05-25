"""
Smoothing system for hiding network correction artifacts.

When reconciliation detects a misprediction, we need to correct the
player's position. This module provides methods to apply corrections
smoothly to avoid jarring visual teleports.

Smoothing methods:
- SNAP: Immediate teleport (for large errors)
- INTERPOLATE: Gradual blend over time
- THRESHOLD: Automatic selection based on error magnitude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
import math

from engine.networking.config import (
    DEFAULT_BLEND_TIME,
    DEFAULT_SMOOTHING_SNAP_THRESHOLD,
    DEFAULT_EXPONENTIAL_FACTOR,
    DEFAULT_MIN_BLEND_SPEED,
    DEFAULT_MAX_BLEND_SPEED,
    DEFAULT_ROTATION_SNAP_THRESHOLD,
    EXPONENTIAL_CONVERGENCE_THRESHOLD,
    QUATERNION_LERP_THRESHOLD,
)


# Type aliases
Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


class SmoothingMethod(Enum):
    """Method for applying corrections."""

    SNAP = auto()
    """Immediately teleport to target position."""

    INTERPOLATE = auto()
    """Smoothly blend from current to target."""

    THRESHOLD = auto()
    """Automatically choose snap or interpolate based on error."""

    EXPONENTIAL = auto()
    """Exponential smoothing - faster at start, slower as approaching target."""


@dataclass
class SmoothingConfig:
    """Configuration for the smoothing system."""

    blend_time: float = DEFAULT_BLEND_TIME
    """Time to blend from current to target (seconds)."""

    snap_threshold: float = DEFAULT_SMOOTHING_SNAP_THRESHOLD
    """Error threshold above which to snap instead of interpolate (units)."""

    exponential_factor: float = DEFAULT_EXPONENTIAL_FACTOR
    """Exponential smoothing factor (higher = faster)."""

    min_blend_speed: float = DEFAULT_MIN_BLEND_SPEED
    """Minimum blend speed (units/second)."""

    max_blend_speed: float = DEFAULT_MAX_BLEND_SPEED
    """Maximum blend speed (units/second)."""

    rotation_snap_threshold: float = DEFAULT_ROTATION_SNAP_THRESHOLD
    """Rotation snap threshold (radians)."""


def smooth_position(
    from_pos: Vector3,
    to_pos: Vector3,
    alpha: float,
) -> Vector3:
    """
    Smoothly interpolate between two positions.

    Args:
        from_pos: Current position.
        to_pos: Target position.
        alpha: Blend factor (0.0 = from, 1.0 = to).

    Returns:
        Blended position.
    """
    alpha = max(0.0, min(1.0, alpha))
    return (
        from_pos[0] + (to_pos[0] - from_pos[0]) * alpha,
        from_pos[1] + (to_pos[1] - from_pos[1]) * alpha,
        from_pos[2] + (to_pos[2] - from_pos[2]) * alpha,
    )


def smooth_rotation(
    from_rot: Quaternion,
    to_rot: Quaternion,
    alpha: float,
) -> Quaternion:
    """
    Smoothly interpolate between two rotations using slerp.

    Args:
        from_rot: Current rotation quaternion (x, y, z, w).
        to_rot: Target rotation quaternion (x, y, z, w).
        alpha: Blend factor (0.0 = from, 1.0 = to).

    Returns:
        Blended rotation quaternion.
    """
    alpha = max(0.0, min(1.0, alpha))

    # Compute dot product
    dot = (
        from_rot[0] * to_rot[0] +
        from_rot[1] * to_rot[1] +
        from_rot[2] * to_rot[2] +
        from_rot[3] * to_rot[3]
    )

    # Take shorter path
    if dot < 0.0:
        to_rot = (-to_rot[0], -to_rot[1], -to_rot[2], -to_rot[3])
        dot = -dot

    # Use linear interpolation for very close quaternions
    if dot > QUATERNION_LERP_THRESHOLD:
        result = (
            from_rot[0] + (to_rot[0] - from_rot[0]) * alpha,
            from_rot[1] + (to_rot[1] - from_rot[1]) * alpha,
            from_rot[2] + (to_rot[2] - from_rot[2]) * alpha,
            from_rot[3] + (to_rot[3] - from_rot[3]) * alpha,
        )
        # Normalize
        length = math.sqrt(sum(x*x for x in result))
        if length > 0:
            return tuple(x / length for x in result)
        return result

    # Slerp
    theta_0 = math.acos(dot)
    theta = theta_0 * alpha

    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    # Prevent division by zero
    if abs(sin_theta_0) < 1e-10:
        # Fall back to linear interpolation
        result = (
            from_rot[0] + (to_rot[0] - from_rot[0]) * alpha,
            from_rot[1] + (to_rot[1] - from_rot[1]) * alpha,
            from_rot[2] + (to_rot[2] - from_rot[2]) * alpha,
            from_rot[3] + (to_rot[3] - from_rot[3]) * alpha,
        )
        length = math.sqrt(sum(x*x for x in result))
        if length > 0:
            return tuple(x / length for x in result)
        return result

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (
        from_rot[0] * s0 + to_rot[0] * s1,
        from_rot[1] * s0 + to_rot[1] * s1,
        from_rot[2] * s0 + to_rot[2] * s1,
        from_rot[3] * s0 + to_rot[3] * s1,
    )


def exponential_smooth(
    current: float,
    target: float,
    factor: float,
    delta_time: float,
) -> float:
    """
    Exponential smoothing - fast approach that slows near target.

    Args:
        current: Current value.
        target: Target value.
        factor: Smoothing factor (higher = faster).
        delta_time: Time step.

    Returns:
        Smoothed value.
    """
    # Exponential interpolation: x' = x + (target - x) * (1 - e^(-factor * dt))
    blend = 1.0 - math.exp(-factor * delta_time)
    return current + (target - current) * blend


def exponential_smooth_vector(
    current: Vector3,
    target: Vector3,
    factor: float,
    delta_time: float,
) -> Vector3:
    """
    Exponential smoothing for 3D vectors.

    Args:
        current: Current position.
        target: Target position.
        factor: Smoothing factor.
        delta_time: Time step.

    Returns:
        Smoothed position.
    """
    blend = 1.0 - math.exp(-factor * delta_time)
    return (
        current[0] + (target[0] - current[0]) * blend,
        current[1] + (target[1] - current[1]) * blend,
        current[2] + (target[2] - current[2]) * blend,
    )


@dataclass
class CorrectionState:
    """Tracks an ongoing correction blend."""

    start_position: Vector3
    """Position when correction started."""

    target_position: Vector3
    """Target corrected position."""

    start_rotation: Optional[Quaternion] = None
    """Rotation when correction started."""

    target_rotation: Optional[Quaternion] = None
    """Target corrected rotation."""

    elapsed_time: float = 0.0
    """Time elapsed since correction started."""

    blend_duration: float = 0.1
    """Total blend duration."""

    is_complete: bool = False
    """Whether correction is complete."""

    @property
    def progress(self) -> float:
        """Get blend progress (0.0 to 1.0)."""
        if self.blend_duration <= 0:
            return 1.0
        return min(1.0, self.elapsed_time / self.blend_duration)


class CorrectionSmoother:
    """
    Manages smooth application of network corrections.

    When a misprediction is detected, this class handles blending
    from the incorrect position to the corrected position over time.

    Example:
        smoother = CorrectionSmoother()

        # When correction needed:
        smoother.apply_correction(current_pos, target_pos, SmoothingMethod.THRESHOLD)

        # Each frame:
        new_pos = smoother.update(delta_time)
        if smoother.is_correcting:
            entity.position = new_pos
    """

    def __init__(
        self,
        config: Optional[SmoothingConfig] = None,
    ) -> None:
        """
        Initialize the smoother.

        Args:
            config: Smoothing configuration.
        """
        self._config = config or SmoothingConfig()
        self._correction: Optional[CorrectionState] = None
        self._current_position: Vector3 = (0.0, 0.0, 0.0)
        self._current_rotation: Optional[Quaternion] = None

    @property
    def blend_time(self) -> float:
        """Get the blend time."""
        return self._config.blend_time

    @blend_time.setter
    def blend_time(self, value: float) -> None:
        """Set the blend time."""
        self._config.blend_time = max(0.001, value)

    @property
    def snap_threshold(self) -> float:
        """Get the snap threshold."""
        return self._config.snap_threshold

    @snap_threshold.setter
    def snap_threshold(self, value: float) -> None:
        """Set the snap threshold."""
        self._config.snap_threshold = max(0.0, value)

    @property
    def is_correcting(self) -> bool:
        """Check if a correction is in progress."""
        return self._correction is not None and not self._correction.is_complete

    @property
    def correction_progress(self) -> float:
        """Get current correction progress (0-1)."""
        if self._correction:
            return self._correction.progress
        return 1.0

    def apply_correction(
        self,
        current: Vector3,
        target: Vector3,
        method: SmoothingMethod,
        current_rotation: Optional[Quaternion] = None,
        target_rotation: Optional[Quaternion] = None,
    ) -> Vector3:
        """
        Apply a position correction using the specified method.

        Args:
            current: Current (incorrect) position.
            target: Target (correct) position.
            method: Smoothing method to use.
            current_rotation: Optional current rotation.
            target_rotation: Optional target rotation.

        Returns:
            The resulting position (may be snapped or start of blend).
        """
        error = self._calculate_error(current, target)

        # Determine actual method
        if method == SmoothingMethod.THRESHOLD:
            if error >= self._config.snap_threshold:
                method = SmoothingMethod.SNAP
            else:
                method = SmoothingMethod.INTERPOLATE

        # Handle snap
        if method == SmoothingMethod.SNAP:
            self._current_position = target
            self._current_rotation = target_rotation
            self._correction = None
            return target

        # Start interpolation
        blend_time = self._calculate_blend_time(error)

        self._correction = CorrectionState(
            start_position=current,
            target_position=target,
            start_rotation=current_rotation,
            target_rotation=target_rotation,
            elapsed_time=0.0,
            blend_duration=blend_time,
            is_complete=False,
        )

        self._current_position = current
        self._current_rotation = current_rotation

        return current

    def _calculate_error(self, current: Vector3, target: Vector3) -> float:
        """Calculate position error magnitude."""
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        dz = target[2] - current[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _calculate_blend_time(self, error: float) -> float:
        """Calculate blend time based on error magnitude."""
        # Scale blend time with error, but clamp
        speed = error / self._config.blend_time
        speed = max(self._config.min_blend_speed, min(self._config.max_blend_speed, speed))
        return error / speed

    def update(
        self,
        delta_time: float,
        method: SmoothingMethod = SmoothingMethod.INTERPOLATE,
    ) -> Tuple[Vector3, Optional[Quaternion]]:
        """
        Update the smoothing and return current smoothed state.

        Args:
            delta_time: Time since last update.
            method: Smoothing method (INTERPOLATE or EXPONENTIAL).

        Returns:
            Tuple of (smoothed_position, smoothed_rotation).
        """
        if self._correction is None or self._correction.is_complete:
            return self._current_position, self._current_rotation

        self._correction.elapsed_time += delta_time

        if method == SmoothingMethod.EXPONENTIAL:
            # Exponential smoothing
            self._current_position = exponential_smooth_vector(
                self._current_position,
                self._correction.target_position,
                self._config.exponential_factor,
                delta_time,
            )

            # Check if close enough to target
            error = self._calculate_error(
                self._current_position,
                self._correction.target_position,
            )
            if error < EXPONENTIAL_CONVERGENCE_THRESHOLD:
                self._current_position = self._correction.target_position
                self._correction.is_complete = True
        else:
            # Linear blend
            t = self._correction.progress
            self._current_position = smooth_position(
                self._correction.start_position,
                self._correction.target_position,
                t,
            )

            if t >= 1.0:
                self._current_position = self._correction.target_position
                self._correction.is_complete = True

        # Handle rotation smoothing
        if (self._correction.start_rotation is not None and
                self._correction.target_rotation is not None):
            t = self._correction.progress
            self._current_rotation = smooth_rotation(
                self._correction.start_rotation,
                self._correction.target_rotation,
                t,
            )

            if self._correction.is_complete:
                self._current_rotation = self._correction.target_rotation

        return self._current_position, self._current_rotation

    def get_position(self) -> Vector3:
        """Get current smoothed position."""
        return self._current_position

    def get_rotation(self) -> Optional[Quaternion]:
        """Get current smoothed rotation."""
        return self._current_rotation

    def cancel_correction(self) -> None:
        """Cancel any in-progress correction."""
        self._correction = None

    def snap_to_target(self) -> None:
        """Immediately complete any in-progress correction."""
        if self._correction:
            self._current_position = self._correction.target_position
            self._current_rotation = self._correction.target_rotation
            self._correction.is_complete = True


class VisualSmoother:
    """
    High-level visual smoothing for entity rendering.

    Separates visual position from simulation position, allowing
    smooth rendering while simulation uses authoritative state.
    """

    def __init__(
        self,
        config: Optional[SmoothingConfig] = None,
    ) -> None:
        """Initialize the visual smoother."""
        self._config = config or SmoothingConfig()
        self._visual_position: Vector3 = (0.0, 0.0, 0.0)
        self._simulation_position: Vector3 = (0.0, 0.0, 0.0)
        self._visual_rotation: Optional[Quaternion] = None
        self._simulation_rotation: Optional[Quaternion] = None

    def set_simulation_state(
        self,
        position: Vector3,
        rotation: Optional[Quaternion] = None,
    ) -> None:
        """
        Set the authoritative simulation state.

        Args:
            position: Simulation position.
            rotation: Simulation rotation.
        """
        self._simulation_position = position
        self._simulation_rotation = rotation

    def update(self, delta_time: float) -> None:
        """
        Update visual state towards simulation state.

        Args:
            delta_time: Time since last update.
        """
        # Exponential smoothing towards simulation position
        self._visual_position = exponential_smooth_vector(
            self._visual_position,
            self._simulation_position,
            self._config.exponential_factor,
            delta_time,
        )

        # Rotation smoothing
        if (self._visual_rotation is not None and
                self._simulation_rotation is not None):
            blend = 1.0 - math.exp(-self._config.exponential_factor * delta_time)
            self._visual_rotation = smooth_rotation(
                self._visual_rotation,
                self._simulation_rotation,
                blend,
            )
        elif self._simulation_rotation is not None:
            self._visual_rotation = self._simulation_rotation

    def get_visual_position(self) -> Vector3:
        """Get position for rendering."""
        return self._visual_position

    def get_visual_rotation(self) -> Optional[Quaternion]:
        """Get rotation for rendering."""
        return self._visual_rotation

    def snap_to_simulation(self) -> None:
        """Immediately sync visual to simulation."""
        self._visual_position = self._simulation_position
        self._visual_rotation = self._simulation_rotation
