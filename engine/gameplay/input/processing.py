"""Raw input processing for the gameplay input system.

This module provides input processing functions including dead zone handling,
response curves, smoothing, and value inversion/scaling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from math import copysign, pow as math_pow, tanh
from typing import Callable, List, Optional, Tuple

from .constants import (
    DEFAULT_DEAD_ZONE,
    DEFAULT_RADIAL_DEAD_ZONE,
    DEFAULT_OUTER_DEAD_ZONE,
    DEFAULT_RESPONSE_EXPONENT,
    DEFAULT_SCURVE_MIDPOINT,
    DEFAULT_SCURVE_STEEPNESS,
    DEFAULT_SMOOTHING_FACTOR,
    DEFAULT_SMOOTHING_WINDOW,
    DEFAULT_SMOOTHING_ALPHA,
    MAX_RESPONSE_EXPONENT,
    MAX_SMOOTHING_FACTOR,
    MIN_DELTA_TIME,
)


# =============================================================================
# Dead Zone Processing
# =============================================================================

class DeadZoneType(Enum):
    """Types of dead zone processing."""
    NONE = auto()
    AXIAL = auto()
    RADIAL = auto()
    CROSS = auto()


def apply_dead_zone(
    value: float,
    dead_zone: float = DEFAULT_DEAD_ZONE,
    outer_zone: float = DEFAULT_OUTER_DEAD_ZONE
) -> float:
    """Apply axial dead zone to a single value.

    Args:
        value: Input value (-1.0 to 1.0)
        dead_zone: Inner dead zone threshold
        outer_zone: Outer dead zone threshold

    Returns:
        Processed value with dead zone applied
    """
    abs_value = abs(value)

    if abs_value < dead_zone:
        return 0.0

    if abs_value > outer_zone:
        return copysign(1.0, value)

    # Rescale the value to maintain smooth transition
    rescaled = (abs_value - dead_zone) / (outer_zone - dead_zone)
    return copysign(rescaled, value)


def apply_radial_dead_zone(
    x: float,
    y: float,
    dead_zone: float = DEFAULT_RADIAL_DEAD_ZONE,
    outer_zone: float = DEFAULT_OUTER_DEAD_ZONE
) -> Tuple[float, float]:
    """Apply radial dead zone to a 2D input.

    Args:
        x: X input value (-1.0 to 1.0)
        y: Y input value (-1.0 to 1.0)
        dead_zone: Inner dead zone radius
        outer_zone: Outer dead zone radius

    Returns:
        Tuple of (x, y) with dead zone applied
    """
    magnitude = (x * x + y * y) ** 0.5

    if magnitude < dead_zone:
        return (0.0, 0.0)

    if magnitude > outer_zone:
        # Normalize to unit circle
        return (x / magnitude, y / magnitude)

    # Rescale to maintain smooth transition
    rescaled_magnitude = (magnitude - dead_zone) / (outer_zone - dead_zone)
    scale = rescaled_magnitude / magnitude

    return (x * scale, y * scale)


def apply_cross_dead_zone(
    x: float,
    y: float,
    dead_zone: float = DEFAULT_DEAD_ZONE
) -> Tuple[float, float]:
    """Apply cross/axis-aligned dead zone.

    This creates a cross-shaped dead zone that prevents
    small inputs near the axes.

    Args:
        x: X input value
        y: Y input value
        dead_zone: Dead zone threshold

    Returns:
        Tuple of (x, y) with cross dead zone applied
    """
    processed_x = apply_dead_zone(x, dead_zone)
    processed_y = apply_dead_zone(y, dead_zone)

    # If either is in dead zone, reduce the other
    if abs(x) < dead_zone:
        processed_y *= (1.0 - dead_zone + abs(x)) / (1.0 - dead_zone) if dead_zone < 1.0 else 0.0
    if abs(y) < dead_zone:
        processed_x *= (1.0 - dead_zone + abs(y)) / (1.0 - dead_zone) if dead_zone < 1.0 else 0.0

    return (processed_x, processed_y)


# =============================================================================
# Response Curves
# =============================================================================

class ResponseCurveType(Enum):
    """Types of response curves for input processing."""
    LINEAR = auto()
    POWER = auto()
    EXPONENTIAL = auto()
    SCURVE = auto()
    STEP = auto()
    CUSTOM = auto()


def apply_linear_curve(value: float) -> float:
    """Apply linear response (identity function).

    Args:
        value: Input value

    Returns:
        Same value unchanged
    """
    return value


def apply_power_curve(
    value: float,
    exponent: float = DEFAULT_RESPONSE_EXPONENT
) -> float:
    """Apply power curve response.

    Args:
        value: Input value (-1.0 to 1.0)
        exponent: Power exponent (1.0 = linear, 2.0 = quadratic)

    Returns:
        Value with power curve applied
    """
    exponent = min(exponent, MAX_RESPONSE_EXPONENT)
    return copysign(math_pow(abs(value), exponent), value)


def apply_exponential_curve(
    value: float,
    base: float = 2.0
) -> float:
    """Apply exponential response curve.

    Args:
        value: Input value (-1.0 to 1.0)
        base: Exponential base

    Returns:
        Value with exponential curve applied
    """
    abs_val = abs(value)
    result = (math_pow(base, abs_val) - 1.0) / (base - 1.0)
    return copysign(min(1.0, result), value)


def apply_scurve(
    value: float,
    midpoint: float = DEFAULT_SCURVE_MIDPOINT,
    steepness: float = DEFAULT_SCURVE_STEEPNESS
) -> float:
    """Apply S-curve (sigmoid-like) response.

    Args:
        value: Input value (-1.0 to 1.0)
        midpoint: Inflection point (0.0 to 1.0)
        steepness: Curve steepness

    Returns:
        Value with S-curve applied
    """
    abs_val = abs(value)

    # Use tanh for smooth S-curve
    # Map input to -steepness to steepness range centered at midpoint
    mapped = (abs_val - midpoint) * steepness
    result = (tanh(mapped) + 1.0) / 2.0

    # Rescale to ensure 0 maps to 0 and 1 maps to 1
    at_zero = (tanh(-midpoint * steepness) + 1.0) / 2.0
    at_one = (tanh((1.0 - midpoint) * steepness) + 1.0) / 2.0
    result = (result - at_zero) / (at_one - at_zero) if at_one != at_zero else result

    return copysign(max(0.0, min(1.0, result)), value)


def apply_step_curve(
    value: float,
    steps: int = 4
) -> float:
    """Apply stepped/quantized response.

    Args:
        value: Input value (-1.0 to 1.0)
        steps: Number of discrete steps

    Returns:
        Quantized value
    """
    if steps <= 0:
        return 0.0

    abs_val = abs(value)
    step_size = 1.0 / steps
    stepped = round(abs_val / step_size) * step_size
    return copysign(min(1.0, stepped), value)


# =============================================================================
# Input Smoothing
# =============================================================================

class SmoothingType(Enum):
    """Types of input smoothing."""
    NONE = auto()
    MOVING_AVERAGE = auto()
    EXPONENTIAL = auto()
    DOUBLE_EXPONENTIAL = auto()


class InputSmoother:
    """Smooths input values over time to reduce jitter."""
    __slots__ = (
        '_smoothing_type', '_alpha', '_window_size',
        '_history', '_smoothed_value', '_velocity'
    )

    def __init__(
        self,
        smoothing_type: SmoothingType = SmoothingType.EXPONENTIAL,
        alpha: float = DEFAULT_SMOOTHING_ALPHA,
        window_size: int = DEFAULT_SMOOTHING_WINDOW
    ):
        """Initialize the input smoother.

        Args:
            smoothing_type: Type of smoothing to apply
            alpha: Smoothing factor for exponential smoothing
            window_size: Window size for moving average
        """
        self._smoothing_type = smoothing_type
        self._alpha = max(0.0, min(1.0, alpha))
        self._window_size = max(1, window_size)
        self._history: List[float] = []
        self._smoothed_value: float = 0.0
        self._velocity: float = 0.0

    @property
    def smoothed_value(self) -> float:
        """Get the current smoothed value."""
        return self._smoothed_value

    @property
    def velocity(self) -> float:
        """Get the estimated velocity (rate of change)."""
        return self._velocity

    def update(self, value: float) -> float:
        """Update with a new value and return smoothed result.

        Args:
            value: New input value

        Returns:
            Smoothed value
        """
        if self._smoothing_type == SmoothingType.NONE:
            self._smoothed_value = value
            return value

        if self._smoothing_type == SmoothingType.MOVING_AVERAGE:
            self._history.append(value)
            if len(self._history) > self._window_size:
                self._history.pop(0)
            self._smoothed_value = sum(self._history) / len(self._history)

        elif self._smoothing_type == SmoothingType.EXPONENTIAL:
            if not self._history:
                self._smoothed_value = value
            else:
                self._smoothed_value = (
                    self._alpha * value +
                    (1.0 - self._alpha) * self._smoothed_value
                )
            self._history = [self._smoothed_value]

        elif self._smoothing_type == SmoothingType.DOUBLE_EXPONENTIAL:
            if not self._history:
                self._smoothed_value = value
                self._velocity = 0.0
            else:
                prev_smoothed = self._smoothed_value
                self._smoothed_value = (
                    self._alpha * value +
                    (1.0 - self._alpha) * (self._smoothed_value + self._velocity)
                )
                self._velocity = (
                    self._alpha * (self._smoothed_value - prev_smoothed) +
                    (1.0 - self._alpha) * self._velocity
                )
            self._history = [self._smoothed_value]

        return self._smoothed_value

    def reset(self) -> None:
        """Reset the smoother state."""
        self._history.clear()
        self._smoothed_value = 0.0
        self._velocity = 0.0


class Vector2Smoother:
    """Smooths 2D vector input."""
    __slots__ = ('_x_smoother', '_y_smoother')

    def __init__(
        self,
        smoothing_type: SmoothingType = SmoothingType.EXPONENTIAL,
        alpha: float = DEFAULT_SMOOTHING_ALPHA,
        window_size: int = DEFAULT_SMOOTHING_WINDOW
    ):
        """Initialize the 2D smoother.

        Args:
            smoothing_type: Type of smoothing
            alpha: Smoothing factor
            window_size: Window size for moving average
        """
        self._x_smoother = InputSmoother(smoothing_type, alpha, window_size)
        self._y_smoother = InputSmoother(smoothing_type, alpha, window_size)

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Update with new values.

        Args:
            x: X value
            y: Y value

        Returns:
            Smoothed (x, y) tuple
        """
        return (
            self._x_smoother.update(x),
            self._y_smoother.update(y)
        )

    @property
    def smoothed_value(self) -> Tuple[float, float]:
        """Get current smoothed values."""
        return (
            self._x_smoother.smoothed_value,
            self._y_smoother.smoothed_value
        )

    def reset(self) -> None:
        """Reset the smoother."""
        self._x_smoother.reset()
        self._y_smoother.reset()


# =============================================================================
# Input Modifiers
# =============================================================================

class InputModifierType(Enum):
    """Types of input modifiers."""
    NEGATE = auto()
    SCALE = auto()
    CLAMP = auto()
    SWIZZLE = auto()
    DEAD_ZONE = auto()
    RESPONSE_CURVE = auto()
    SMOOTH = auto()
    INVERT_Y = auto()


@dataclass
class InputModifier:
    """Describes a modifier to apply to input values."""
    modifier_type: InputModifierType
    params: dict = field(default_factory=dict)


class InputModifierChain:
    """Chain of modifiers to apply to input values."""
    __slots__ = ('_modifiers', '_smoothers')

    def __init__(self, modifiers: Optional[List[InputModifier]] = None):
        """Initialize the modifier chain.

        Args:
            modifiers: List of modifiers to apply
        """
        self._modifiers = modifiers or []
        self._smoothers: dict = {}

    def add_modifier(self, modifier: InputModifier) -> None:
        """Add a modifier to the chain.

        Args:
            modifier: Modifier to add
        """
        self._modifiers.append(modifier)

    def remove_modifier(self, index: int) -> bool:
        """Remove a modifier at the given index.

        Args:
            index: Index of modifier to remove

        Returns:
            True if removed
        """
        if 0 <= index < len(self._modifiers):
            self._modifiers.pop(index)
            return True
        return False

    def clear(self) -> None:
        """Clear all modifiers."""
        self._modifiers.clear()
        self._smoothers.clear()

    def process(self, value: float, modifier_id: str = "default") -> float:
        """Process a value through the modifier chain.

        Args:
            value: Input value
            modifier_id: Identifier for stateful modifiers

        Returns:
            Processed value
        """
        result = value

        for modifier in self._modifiers:
            result = self._apply_modifier(result, modifier, modifier_id)

        return result

    def process_2d(
        self,
        x: float,
        y: float,
        modifier_id: str = "default"
    ) -> Tuple[float, float]:
        """Process 2D values through the modifier chain.

        Args:
            x: X value
            y: Y value
            modifier_id: Identifier for stateful modifiers

        Returns:
            Processed (x, y) tuple
        """
        result_x = x
        result_y = y

        for modifier in self._modifiers:
            if modifier.modifier_type == InputModifierType.SWIZZLE:
                result_x, result_y = result_y, result_x
            elif modifier.modifier_type == InputModifierType.INVERT_Y:
                result_y = -result_y
            elif modifier.modifier_type == InputModifierType.DEAD_ZONE:
                dead_zone = modifier.params.get("dead_zone", DEFAULT_DEAD_ZONE)
                dz_type = modifier.params.get("type", DeadZoneType.RADIAL)
                if dz_type == DeadZoneType.RADIAL:
                    result_x, result_y = apply_radial_dead_zone(result_x, result_y, dead_zone)
                elif dz_type == DeadZoneType.CROSS:
                    result_x, result_y = apply_cross_dead_zone(result_x, result_y, dead_zone)
                else:
                    result_x = apply_dead_zone(result_x, dead_zone)
                    result_y = apply_dead_zone(result_y, dead_zone)
            else:
                result_x = self._apply_modifier(result_x, modifier, f"{modifier_id}_x")
                result_y = self._apply_modifier(result_y, modifier, f"{modifier_id}_y")

        return (result_x, result_y)

    def _apply_modifier(
        self,
        value: float,
        modifier: InputModifier,
        modifier_id: str
    ) -> float:
        """Apply a single modifier.

        Args:
            value: Input value
            modifier: Modifier to apply
            modifier_id: Identifier for stateful modifiers

        Returns:
            Modified value
        """
        if modifier.modifier_type == InputModifierType.NEGATE:
            return -value

        elif modifier.modifier_type == InputModifierType.SCALE:
            scale = modifier.params.get("scale", 1.0)
            return value * scale

        elif modifier.modifier_type == InputModifierType.CLAMP:
            min_val = modifier.params.get("min", -1.0)
            max_val = modifier.params.get("max", 1.0)
            return max(min_val, min(max_val, value))

        elif modifier.modifier_type == InputModifierType.DEAD_ZONE:
            dead_zone = modifier.params.get("dead_zone", DEFAULT_DEAD_ZONE)
            outer_zone = modifier.params.get("outer_zone", DEFAULT_OUTER_DEAD_ZONE)
            return apply_dead_zone(value, dead_zone, outer_zone)

        elif modifier.modifier_type == InputModifierType.RESPONSE_CURVE:
            curve_type = modifier.params.get("curve_type", ResponseCurveType.LINEAR)
            if curve_type == ResponseCurveType.LINEAR:
                return apply_linear_curve(value)
            elif curve_type == ResponseCurveType.POWER:
                exponent = modifier.params.get("exponent", DEFAULT_RESPONSE_EXPONENT)
                return apply_power_curve(value, exponent)
            elif curve_type == ResponseCurveType.EXPONENTIAL:
                base = modifier.params.get("base", 2.0)
                return apply_exponential_curve(value, base)
            elif curve_type == ResponseCurveType.SCURVE:
                midpoint = modifier.params.get("midpoint", DEFAULT_SCURVE_MIDPOINT)
                steepness = modifier.params.get("steepness", DEFAULT_SCURVE_STEEPNESS)
                return apply_scurve(value, midpoint, steepness)
            elif curve_type == ResponseCurveType.STEP:
                steps = modifier.params.get("steps", 4)
                return apply_step_curve(value, steps)

        elif modifier.modifier_type == InputModifierType.SMOOTH:
            smoother_key = f"smooth_{modifier_id}"
            if smoother_key not in self._smoothers:
                smoothing_type = modifier.params.get(
                    "type", SmoothingType.EXPONENTIAL
                )
                alpha = modifier.params.get("alpha", DEFAULT_SMOOTHING_ALPHA)
                window = modifier.params.get("window", DEFAULT_SMOOTHING_WINDOW)
                self._smoothers[smoother_key] = InputSmoother(
                    smoothing_type, alpha, window
                )
            return self._smoothers[smoother_key].update(value)

        return value


# =============================================================================
# Input Processor
# =============================================================================

@dataclass
class ProcessingSettings:
    """Settings for input processing."""
    dead_zone_type: DeadZoneType = DeadZoneType.RADIAL
    dead_zone: float = DEFAULT_DEAD_ZONE
    outer_zone: float = DEFAULT_OUTER_DEAD_ZONE
    response_curve: ResponseCurveType = ResponseCurveType.LINEAR
    response_exponent: float = DEFAULT_RESPONSE_EXPONENT
    smoothing_type: SmoothingType = SmoothingType.NONE
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA
    sensitivity: float = 1.0
    invert_x: bool = False
    invert_y: bool = False


class InputProcessor:
    """Processes raw input values according to settings."""
    __slots__ = ('_settings', '_modifier_chain', '_smoothers')

    def __init__(self, settings: Optional[ProcessingSettings] = None):
        """Initialize the input processor.

        Args:
            settings: Processing settings
        """
        self._settings = settings or ProcessingSettings()
        self._modifier_chain = InputModifierChain()
        self._smoothers: dict[str, InputSmoother] = {}
        self._build_modifier_chain()

    @property
    def settings(self) -> ProcessingSettings:
        """Get current settings."""
        return self._settings

    @settings.setter
    def settings(self, value: ProcessingSettings) -> None:
        """Set new settings and rebuild modifier chain."""
        self._settings = value
        self._build_modifier_chain()

    def _build_modifier_chain(self) -> None:
        """Build the modifier chain from settings."""
        self._modifier_chain.clear()

        # Dead zone
        if self._settings.dead_zone_type != DeadZoneType.NONE:
            self._modifier_chain.add_modifier(InputModifier(
                modifier_type=InputModifierType.DEAD_ZONE,
                params={
                    "dead_zone": self._settings.dead_zone,
                    "outer_zone": self._settings.outer_zone,
                    "type": self._settings.dead_zone_type,
                }
            ))

        # Response curve
        if self._settings.response_curve != ResponseCurveType.LINEAR:
            self._modifier_chain.add_modifier(InputModifier(
                modifier_type=InputModifierType.RESPONSE_CURVE,
                params={
                    "curve_type": self._settings.response_curve,
                    "exponent": self._settings.response_exponent,
                }
            ))

        # Smoothing
        if self._settings.smoothing_type != SmoothingType.NONE:
            self._modifier_chain.add_modifier(InputModifier(
                modifier_type=InputModifierType.SMOOTH,
                params={
                    "type": self._settings.smoothing_type,
                    "alpha": self._settings.smoothing_alpha,
                }
            ))

        # Sensitivity
        if self._settings.sensitivity != 1.0:
            self._modifier_chain.add_modifier(InputModifier(
                modifier_type=InputModifierType.SCALE,
                params={"scale": self._settings.sensitivity}
            ))

    def process_1d(self, value: float, input_id: str = "default") -> float:
        """Process a 1D input value.

        Args:
            value: Raw input value
            input_id: Identifier for stateful processing

        Returns:
            Processed value
        """
        result = self._modifier_chain.process(value, input_id)

        # Apply invert
        if self._settings.invert_x:
            result = -result

        return max(-1.0, min(1.0, result))

    def process_2d(
        self,
        x: float,
        y: float,
        input_id: str = "default"
    ) -> Tuple[float, float]:
        """Process a 2D input value.

        Args:
            x: Raw X value
            y: Raw Y value
            input_id: Identifier for stateful processing

        Returns:
            Processed (x, y) tuple
        """
        result_x, result_y = self._modifier_chain.process_2d(x, y, input_id)

        # Apply invert
        if self._settings.invert_x:
            result_x = -result_x
        if self._settings.invert_y:
            result_y = -result_y

        return (
            max(-1.0, min(1.0, result_x)),
            max(-1.0, min(1.0, result_y))
        )

    def process_trigger(
        self,
        value: float,
        input_id: str = "default"
    ) -> float:
        """Process a trigger input (0 to 1).

        Args:
            value: Raw trigger value (0.0 to 1.0)
            input_id: Identifier for stateful processing

        Returns:
            Processed trigger value
        """
        # Clamp to valid range first
        value = max(0.0, min(1.0, value))

        # Process with modifier chain
        result = self._modifier_chain.process(value, input_id)

        # Clamp result to 0-1 range
        return max(0.0, min(1.0, result))

    def reset(self) -> None:
        """Reset all stateful processing."""
        self._modifier_chain.clear()
        self._smoothers.clear()
        self._build_modifier_chain()
