"""
Easing functions for UI animations.

Provides a comprehensive set of easing functions for smooth animations,
including standard easing curves (quad, cubic, quart, quint, sine, expo, circ)
and special effects (elastic, back, bounce). Also supports custom bezier curves.

All easing functions take a normalized time value t in [0, 1] and return
a normalized progress value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

# Type alias for easing functions
EasingFunction = Callable[[float], float]


class EasingType(Enum):
    """Enumeration of all built-in easing types."""

    LINEAR = auto()
    # Quad
    QUAD_IN = auto()
    QUAD_OUT = auto()
    QUAD_IN_OUT = auto()
    # Cubic
    CUBIC_IN = auto()
    CUBIC_OUT = auto()
    CUBIC_IN_OUT = auto()
    # Quart
    QUART_IN = auto()
    QUART_OUT = auto()
    QUART_IN_OUT = auto()
    # Quint
    QUINT_IN = auto()
    QUINT_OUT = auto()
    QUINT_IN_OUT = auto()
    # Sine
    SINE_IN = auto()
    SINE_OUT = auto()
    SINE_IN_OUT = auto()
    # Expo
    EXPO_IN = auto()
    EXPO_OUT = auto()
    EXPO_IN_OUT = auto()
    # Circ
    CIRC_IN = auto()
    CIRC_OUT = auto()
    CIRC_IN_OUT = auto()
    # Elastic
    ELASTIC_IN = auto()
    ELASTIC_OUT = auto()
    ELASTIC_IN_OUT = auto()
    # Back
    BACK_IN = auto()
    BACK_OUT = auto()
    BACK_IN_OUT = auto()
    # Bounce
    BOUNCE_IN = auto()
    BOUNCE_OUT = auto()
    BOUNCE_IN_OUT = auto()


# =============================================================================
# LINEAR
# =============================================================================


def linear(t: float) -> float:
    """Linear easing - constant speed."""
    return t


# =============================================================================
# QUAD (power of 2)
# =============================================================================


def quad_in(t: float) -> float:
    """Quadratic ease-in - accelerating from zero velocity."""
    return t * t


def quad_out(t: float) -> float:
    """Quadratic ease-out - decelerating to zero velocity."""
    return t * (2 - t)


def quad_in_out(t: float) -> float:
    """Quadratic ease-in-out - acceleration until halfway, then deceleration."""
    if t < 0.5:
        return 2 * t * t
    return -1 + (4 - 2 * t) * t


# =============================================================================
# CUBIC (power of 3)
# =============================================================================


def cubic_in(t: float) -> float:
    """Cubic ease-in - accelerating from zero velocity."""
    return t * t * t


def cubic_out(t: float) -> float:
    """Cubic ease-out - decelerating to zero velocity."""
    t1 = t - 1
    return t1 * t1 * t1 + 1


def cubic_in_out(t: float) -> float:
    """Cubic ease-in-out - acceleration until halfway, then deceleration."""
    if t < 0.5:
        return 4 * t * t * t
    t1 = 2 * t - 2
    return 0.5 * t1 * t1 * t1 + 1


# =============================================================================
# QUART (power of 4)
# =============================================================================


def quart_in(t: float) -> float:
    """Quartic ease-in - accelerating from zero velocity."""
    return t * t * t * t


def quart_out(t: float) -> float:
    """Quartic ease-out - decelerating to zero velocity."""
    t1 = t - 1
    return 1 - t1 * t1 * t1 * t1


def quart_in_out(t: float) -> float:
    """Quartic ease-in-out - acceleration until halfway, then deceleration."""
    if t < 0.5:
        return 8 * t * t * t * t
    t1 = t - 1
    return 1 - 8 * t1 * t1 * t1 * t1


# =============================================================================
# QUINT (power of 5)
# =============================================================================


def quint_in(t: float) -> float:
    """Quintic ease-in - accelerating from zero velocity."""
    return t * t * t * t * t


def quint_out(t: float) -> float:
    """Quintic ease-out - decelerating to zero velocity."""
    t1 = t - 1
    return 1 + t1 * t1 * t1 * t1 * t1


def quint_in_out(t: float) -> float:
    """Quintic ease-in-out - acceleration until halfway, then deceleration."""
    if t < 0.5:
        return 16 * t * t * t * t * t
    t1 = 2 * t - 2
    return 0.5 * t1 * t1 * t1 * t1 * t1 + 1


# =============================================================================
# SINE
# =============================================================================


def sine_in(t: float) -> float:
    """Sinusoidal ease-in - accelerating from zero velocity."""
    return 1 - math.cos(t * math.pi / 2)


def sine_out(t: float) -> float:
    """Sinusoidal ease-out - decelerating to zero velocity."""
    return math.sin(t * math.pi / 2)


def sine_in_out(t: float) -> float:
    """Sinusoidal ease-in-out - acceleration until halfway, then deceleration."""
    return 0.5 * (1 - math.cos(math.pi * t))


# =============================================================================
# EXPO (exponential)
# =============================================================================


def expo_in(t: float) -> float:
    """Exponential ease-in - accelerating from zero velocity."""
    if t == 0:
        return 0
    return math.pow(2, 10 * (t - 1))


def expo_out(t: float) -> float:
    """Exponential ease-out - decelerating to zero velocity."""
    if t == 1:
        return 1
    return 1 - math.pow(2, -10 * t)


def expo_in_out(t: float) -> float:
    """Exponential ease-in-out - acceleration until halfway, then deceleration."""
    if t == 0:
        return 0
    if t == 1:
        return 1
    if t < 0.5:
        return 0.5 * math.pow(2, 20 * t - 10)
    return 1 - 0.5 * math.pow(2, -20 * t + 10)


# =============================================================================
# CIRC (circular)
# =============================================================================


def circ_in(t: float) -> float:
    """Circular ease-in - accelerating from zero velocity."""
    return 1 - math.sqrt(1 - t * t)


def circ_out(t: float) -> float:
    """Circular ease-out - decelerating to zero velocity."""
    t1 = t - 1
    return math.sqrt(1 - t1 * t1)


def circ_in_out(t: float) -> float:
    """Circular ease-in-out - acceleration until halfway, then deceleration."""
    if t < 0.5:
        return 0.5 * (1 - math.sqrt(1 - 4 * t * t))
    t1 = 2 * t - 2
    return 0.5 * (math.sqrt(1 - t1 * t1) + 1)


# =============================================================================
# ELASTIC
# =============================================================================


_ELASTIC_P = 0.3
_ELASTIC_S = _ELASTIC_P / 4


def elastic_in(t: float) -> float:
    """Elastic ease-in - elastic snap at start."""
    if t == 0:
        return 0
    if t == 1:
        return 1
    t1 = t - 1
    return -math.pow(2, 10 * t1) * math.sin((t1 - _ELASTIC_S) * (2 * math.pi) / _ELASTIC_P)


def elastic_out(t: float) -> float:
    """Elastic ease-out - elastic snap at end."""
    if t == 0:
        return 0
    if t == 1:
        return 1
    return math.pow(2, -10 * t) * math.sin((t - _ELASTIC_S) * (2 * math.pi) / _ELASTIC_P) + 1


def elastic_in_out(t: float) -> float:
    """Elastic ease-in-out - elastic snap at both ends."""
    if t == 0:
        return 0
    if t == 1:
        return 1
    t2 = t * 2
    if t2 < 1:
        t1 = t2 - 1
        return -0.5 * math.pow(2, 10 * t1) * math.sin((t1 - _ELASTIC_S) * (2 * math.pi) / _ELASTIC_P)
    t1 = t2 - 1
    return math.pow(2, -10 * t1) * math.sin((t1 - _ELASTIC_S) * (2 * math.pi) / _ELASTIC_P) * 0.5 + 1


# =============================================================================
# BACK (overshoot)
# =============================================================================


_BACK_S = 1.70158


def back_in(t: float) -> float:
    """Back ease-in - overshooting at start."""
    return t * t * ((_BACK_S + 1) * t - _BACK_S)


def back_out(t: float) -> float:
    """Back ease-out - overshooting at end."""
    t1 = t - 1
    return t1 * t1 * ((_BACK_S + 1) * t1 + _BACK_S) + 1


def back_in_out(t: float) -> float:
    """Back ease-in-out - overshooting at both ends."""
    s = _BACK_S * 1.525
    t2 = t * 2
    if t2 < 1:
        return 0.5 * (t2 * t2 * ((s + 1) * t2 - s))
    t2 = t2 - 2
    return 0.5 * (t2 * t2 * ((s + 1) * t2 + s) + 2)


# =============================================================================
# BOUNCE
# =============================================================================

# Bounce constants - derived from physics simulation of bouncing ball
_BOUNCE_DIVISOR = 2.75  # Time divisor for bounce segments
_BOUNCE_MULTIPLIER = 7.5625  # = (1 / (1 / 2.75))^2, ensures f(1) = 1
_BOUNCE_T1 = 1.0 / _BOUNCE_DIVISOR  # ~0.3636 - first bounce threshold
_BOUNCE_T2 = 2.0 / _BOUNCE_DIVISOR  # ~0.7273 - second bounce threshold
_BOUNCE_T3 = 2.5 / _BOUNCE_DIVISOR  # ~0.9091 - third bounce threshold
_BOUNCE_OFFSET1 = 1.5 / _BOUNCE_DIVISOR  # First segment offset
_BOUNCE_OFFSET2 = 2.25 / _BOUNCE_DIVISOR  # Second segment offset
_BOUNCE_OFFSET3 = 2.625 / _BOUNCE_DIVISOR  # Third segment offset
_BOUNCE_HEIGHT1 = 0.75  # Height after first bounce
_BOUNCE_HEIGHT2 = 0.9375  # Height after second bounce
_BOUNCE_HEIGHT3 = 0.984375  # Height after third bounce


def bounce_out(t: float) -> float:
    """Bounce ease-out - bouncing at end."""
    if t < _BOUNCE_T1:
        return _BOUNCE_MULTIPLIER * t * t
    elif t < _BOUNCE_T2:
        t1 = t - _BOUNCE_OFFSET1
        return _BOUNCE_MULTIPLIER * t1 * t1 + _BOUNCE_HEIGHT1
    elif t < _BOUNCE_T3:
        t1 = t - _BOUNCE_OFFSET2
        return _BOUNCE_MULTIPLIER * t1 * t1 + _BOUNCE_HEIGHT2
    else:
        t1 = t - _BOUNCE_OFFSET3
        return _BOUNCE_MULTIPLIER * t1 * t1 + _BOUNCE_HEIGHT3


def bounce_in(t: float) -> float:
    """Bounce ease-in - bouncing at start."""
    return 1 - bounce_out(1 - t)


def bounce_in_out(t: float) -> float:
    """Bounce ease-in-out - bouncing at both ends."""
    if t < 0.5:
        return bounce_in(t * 2) * 0.5
    return bounce_out(t * 2 - 1) * 0.5 + 0.5


# =============================================================================
# CUBIC BEZIER
# =============================================================================


@dataclass
class CubicBezier:
    """
    Custom cubic bezier easing curve.

    Control points define the curve shape:
    - P0 = (0, 0) - start point (implicit)
    - P1 = (x1, y1) - first control point
    - P2 = (x2, y2) - second control point
    - P3 = (1, 1) - end point (implicit)

    Args:
        x1: X coordinate of first control point (0-1)
        y1: Y coordinate of first control point (can be outside 0-1 for overshoot)
        x2: X coordinate of second control point (0-1)
        y2: Y coordinate of second control point (can be outside 0-1 for overshoot)
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        """Validate control point coordinates."""
        if not (0 <= self.x1 <= 1):
            raise ValueError(f"x1 must be in [0, 1], got {self.x1}")
        if not (0 <= self.x2 <= 1):
            raise ValueError(f"x2 must be in [0, 1], got {self.x2}")

    def __call__(self, t: float) -> float:
        """
        Evaluate the bezier curve at time t.

        Uses Newton-Raphson iteration to find the parameter value
        that corresponds to the given x (time) value, then evaluates y.
        """
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0

        # Newton-Raphson to find parameter for given x
        guess = t
        for _ in range(8):
            x_at_guess = self._sample_x(guess)
            slope = self._slope_x(guess)
            if abs(slope) < 1e-10:
                break
            guess = guess - (x_at_guess - t) / slope

        return self._sample_y(guess)

    def _sample_x(self, t: float) -> float:
        """Sample the X value of the bezier at parameter t."""
        # Bezier formula: (1-t)^3*0 + 3*(1-t)^2*t*x1 + 3*(1-t)*t^2*x2 + t^3*1
        return 3 * (1 - t) * (1 - t) * t * self.x1 + 3 * (1 - t) * t * t * self.x2 + t * t * t

    def _sample_y(self, t: float) -> float:
        """Sample the Y value of the bezier at parameter t."""
        return 3 * (1 - t) * (1 - t) * t * self.y1 + 3 * (1 - t) * t * t * self.y2 + t * t * t

    def _slope_x(self, t: float) -> float:
        """Derivative of X with respect to parameter t."""
        # d/dt of bezier: 3*(1-t)^2*x1 + 6*(1-t)*t*(x2-x1) + 3*t^2*(1-x2)
        return 3 * (1 - t) * (1 - t) * self.x1 + 6 * (1 - t) * t * (self.x2 - self.x1) + 3 * t * t * (1 - self.x2)


# Pre-defined bezier curves (CSS standard curves)
EASE = CubicBezier(0.25, 0.1, 0.25, 1.0)
EASE_IN = CubicBezier(0.42, 0.0, 1.0, 1.0)
EASE_OUT = CubicBezier(0.0, 0.0, 0.58, 1.0)
EASE_IN_OUT = CubicBezier(0.42, 0.0, 0.58, 1.0)


# =============================================================================
# EASING REGISTRY
# =============================================================================


# Mapping from EasingType to function
_EASING_FUNCTIONS: dict[EasingType, EasingFunction] = {
    EasingType.LINEAR: linear,
    EasingType.QUAD_IN: quad_in,
    EasingType.QUAD_OUT: quad_out,
    EasingType.QUAD_IN_OUT: quad_in_out,
    EasingType.CUBIC_IN: cubic_in,
    EasingType.CUBIC_OUT: cubic_out,
    EasingType.CUBIC_IN_OUT: cubic_in_out,
    EasingType.QUART_IN: quart_in,
    EasingType.QUART_OUT: quart_out,
    EasingType.QUART_IN_OUT: quart_in_out,
    EasingType.QUINT_IN: quint_in,
    EasingType.QUINT_OUT: quint_out,
    EasingType.QUINT_IN_OUT: quint_in_out,
    EasingType.SINE_IN: sine_in,
    EasingType.SINE_OUT: sine_out,
    EasingType.SINE_IN_OUT: sine_in_out,
    EasingType.EXPO_IN: expo_in,
    EasingType.EXPO_OUT: expo_out,
    EasingType.EXPO_IN_OUT: expo_in_out,
    EasingType.CIRC_IN: circ_in,
    EasingType.CIRC_OUT: circ_out,
    EasingType.CIRC_IN_OUT: circ_in_out,
    EasingType.ELASTIC_IN: elastic_in,
    EasingType.ELASTIC_OUT: elastic_out,
    EasingType.ELASTIC_IN_OUT: elastic_in_out,
    EasingType.BACK_IN: back_in,
    EasingType.BACK_OUT: back_out,
    EasingType.BACK_IN_OUT: back_in_out,
    EasingType.BOUNCE_IN: bounce_in,
    EasingType.BOUNCE_OUT: bounce_out,
    EasingType.BOUNCE_IN_OUT: bounce_in_out,
}

# String name mapping for convenience
_EASING_BY_NAME: dict[str, EasingFunction] = {
    "linear": linear,
    "quad_in": quad_in,
    "quad_out": quad_out,
    "quad_in_out": quad_in_out,
    "cubic_in": cubic_in,
    "cubic_out": cubic_out,
    "cubic_in_out": cubic_in_out,
    "quart_in": quart_in,
    "quart_out": quart_out,
    "quart_in_out": quart_in_out,
    "quint_in": quint_in,
    "quint_out": quint_out,
    "quint_in_out": quint_in_out,
    "sine_in": sine_in,
    "sine_out": sine_out,
    "sine_in_out": sine_in_out,
    "expo_in": expo_in,
    "expo_out": expo_out,
    "expo_in_out": expo_in_out,
    "circ_in": circ_in,
    "circ_out": circ_out,
    "circ_in_out": circ_in_out,
    "elastic_in": elastic_in,
    "elastic_out": elastic_out,
    "elastic_in_out": elastic_in_out,
    "back_in": back_in,
    "back_out": back_out,
    "back_in_out": back_in_out,
    "bounce_in": bounce_in,
    "bounce_out": bounce_out,
    "bounce_in_out": bounce_in_out,
    # CSS-style aliases
    "ease": EASE,
    "ease_in": EASE_IN,
    "ease_out": EASE_OUT,
    "ease_in_out": EASE_IN_OUT,
}


def get_easing(name: str | EasingType) -> EasingFunction:
    """
    Get an easing function by name or type.

    Args:
        name: Easing name (string) or EasingType enum

    Returns:
        The easing function

    Raises:
        ValueError: If the easing name is not found
    """
    if isinstance(name, EasingType):
        return _EASING_FUNCTIONS[name]

    name_lower = name.lower()
    if name_lower in _EASING_BY_NAME:
        return _EASING_BY_NAME[name_lower]

    raise ValueError(f"Unknown easing function: {name}")


def create_bezier(x1: float, y1: float, x2: float, y2: float) -> CubicBezier:
    """
    Create a custom cubic bezier easing function.

    Args:
        x1: X coordinate of first control point (0-1)
        y1: Y coordinate of first control point
        x2: X coordinate of second control point (0-1)
        y2: Y coordinate of second control point

    Returns:
        A CubicBezier easing function
    """
    return CubicBezier(x1, y1, x2, y2)


def clamp(t: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a value to the given range."""
    return max(minimum, min(maximum, t))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


__all__ = [
    # Enum
    "EasingType",
    # Linear
    "linear",
    # Quad
    "quad_in",
    "quad_out",
    "quad_in_out",
    # Cubic
    "cubic_in",
    "cubic_out",
    "cubic_in_out",
    # Quart
    "quart_in",
    "quart_out",
    "quart_in_out",
    # Quint
    "quint_in",
    "quint_out",
    "quint_in_out",
    # Sine
    "sine_in",
    "sine_out",
    "sine_in_out",
    # Expo
    "expo_in",
    "expo_out",
    "expo_in_out",
    # Circ
    "circ_in",
    "circ_out",
    "circ_in_out",
    # Elastic
    "elastic_in",
    "elastic_out",
    "elastic_in_out",
    # Back
    "back_in",
    "back_out",
    "back_in_out",
    # Bounce
    "bounce_in",
    "bounce_out",
    "bounce_in_out",
    # Bezier
    "CubicBezier",
    "create_bezier",
    "EASE",
    "EASE_IN",
    "EASE_OUT",
    "EASE_IN_OUT",
    # Registry
    "get_easing",
    "EasingFunction",
    # Utils
    "clamp",
    "lerp",
]
