"""
Screen Transitions.

Implements screen transition effects including fade, slide, zoom,
and custom transitions with configurable duration and easing.

References:
- UI_CONTEXT.md Section: Screen Transitions
- ARCHITECTURE_UI.md Section: Screen Transitions / Page Transitions
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .screen import Screen


# =============================================================================
# TRANSITION DIRECTION
# =============================================================================


class TransitionDirection(Enum):
    """Direction for directional transitions like slide."""
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()


# =============================================================================
# EASING FUNCTIONS
# =============================================================================


class Easing(Enum):
    """Common easing function types."""
    LINEAR = auto()
    EASE_IN = auto()
    EASE_OUT = auto()
    EASE_IN_OUT = auto()
    EASE_IN_QUAD = auto()
    EASE_OUT_QUAD = auto()
    EASE_IN_OUT_QUAD = auto()
    EASE_IN_CUBIC = auto()
    EASE_OUT_CUBIC = auto()
    EASE_IN_OUT_CUBIC = auto()
    EASE_IN_EXPO = auto()
    EASE_OUT_EXPO = auto()
    EASE_IN_OUT_EXPO = auto()
    EASE_IN_BACK = auto()
    EASE_OUT_BACK = auto()
    EASE_IN_OUT_BACK = auto()
    EASE_IN_ELASTIC = auto()
    EASE_OUT_ELASTIC = auto()
    EASE_IN_OUT_ELASTIC = auto()
    EASE_IN_BOUNCE = auto()
    EASE_OUT_BOUNCE = auto()
    EASE_IN_OUT_BOUNCE = auto()


EasingFunction = Callable[[float], float]


def get_easing_function(easing: Easing) -> EasingFunction:
    """
    Get the easing function for a given easing type.

    Args:
        easing: Easing type

    Returns:
        Easing function that takes t in [0,1] and returns eased value
    """
    return _EASING_FUNCTIONS.get(easing, _linear)


def _linear(t: float) -> float:
    """Linear easing (no easing)."""
    return t


def _ease_in(t: float) -> float:
    """Ease in (start slow)."""
    return t * t


def _ease_out(t: float) -> float:
    """Ease out (end slow)."""
    return 1.0 - (1.0 - t) ** 2


def _ease_in_out(t: float) -> float:
    """Ease in-out (both slow)."""
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def _ease_in_quad(t: float) -> float:
    """Quadratic ease in."""
    return t * t


def _ease_out_quad(t: float) -> float:
    """Quadratic ease out."""
    return 1.0 - (1.0 - t) * (1.0 - t)


def _ease_in_out_quad(t: float) -> float:
    """Quadratic ease in-out."""
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def _ease_in_cubic(t: float) -> float:
    """Cubic ease in."""
    return t * t * t


def _ease_out_cubic(t: float) -> float:
    """Cubic ease out."""
    return 1.0 - (1.0 - t) ** 3


def _ease_in_out_cubic(t: float) -> float:
    """Cubic ease in-out."""
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def _ease_in_expo(t: float) -> float:
    """Exponential ease in."""
    if t == 0:
        return 0.0
    return 2.0 ** (10.0 * t - 10.0)


def _ease_out_expo(t: float) -> float:
    """Exponential ease out."""
    if t == 1.0:
        return 1.0
    return 1.0 - 2.0 ** (-10.0 * t)


def _ease_in_out_expo(t: float) -> float:
    """Exponential ease in-out."""
    if t == 0:
        return 0.0
    if t == 1.0:
        return 1.0
    if t < 0.5:
        return 2.0 ** (20.0 * t - 10.0) / 2.0
    return (2.0 - 2.0 ** (-20.0 * t + 10.0)) / 2.0


# Back easing overshoot coefficient (standard value for ~10% overshoot)
BACK_OVERSHOOT = 1.70158
BACK_OVERSHOOT_SCALED = BACK_OVERSHOOT * 1.525  # Adjusted for in-out easing


def _ease_in_back(t: float) -> float:
    """Back ease in (overshoot at start)."""
    c3 = BACK_OVERSHOOT + 1.0
    return c3 * t * t * t - BACK_OVERSHOOT * t * t


def _ease_out_back(t: float) -> float:
    """Back ease out (overshoot at end)."""
    c3 = BACK_OVERSHOOT + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + BACK_OVERSHOOT * (t - 1.0) ** 2


def _ease_in_out_back(t: float) -> float:
    """Back ease in-out."""
    c2 = BACK_OVERSHOOT_SCALED
    if t < 0.5:
        return ((2.0 * t) ** 2 * ((c2 + 1.0) * 2.0 * t - c2)) / 2.0
    return ((2.0 * t - 2.0) ** 2 * ((c2 + 1.0) * (t * 2.0 - 2.0) + c2) + 2.0) / 2.0


def _ease_in_elastic(t: float) -> float:
    """Elastic ease in."""
    if t == 0:
        return 0.0
    if t == 1.0:
        return 1.0
    c4 = (2.0 * math.pi) / 3.0
    return -(2.0 ** (10.0 * t - 10.0)) * math.sin((t * 10.0 - 10.75) * c4)


def _ease_out_elastic(t: float) -> float:
    """Elastic ease out."""
    if t == 0:
        return 0.0
    if t == 1.0:
        return 1.0
    c4 = (2.0 * math.pi) / 3.0
    return 2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0


def _ease_in_out_elastic(t: float) -> float:
    """Elastic ease in-out."""
    if t == 0:
        return 0.0
    if t == 1.0:
        return 1.0
    c5 = (2.0 * math.pi) / 4.5
    if t < 0.5:
        return -(2.0 ** (20.0 * t - 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0
    return (2.0 ** (-20.0 * t + 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0 + 1.0


# Bounce easing coefficients (standard values for natural bounce physics)
BOUNCE_AMPLITUDE = 7.5625  # Controls bounce height
BOUNCE_DIVISOR = 2.75     # Controls bounce timing divisions


def _ease_out_bounce(t: float) -> float:
    """Bounce ease out."""
    n1 = BOUNCE_AMPLITUDE
    d1 = BOUNCE_DIVISOR
    if t < 1.0 / d1:
        return n1 * t * t
    elif t < 2.0 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


def _ease_in_bounce(t: float) -> float:
    """Bounce ease in."""
    return 1.0 - _ease_out_bounce(1.0 - t)


def _ease_in_out_bounce(t: float) -> float:
    """Bounce ease in-out."""
    if t < 0.5:
        return (1.0 - _ease_out_bounce(1.0 - 2.0 * t)) / 2.0
    return (1.0 + _ease_out_bounce(2.0 * t - 1.0)) / 2.0


_EASING_FUNCTIONS: Dict[Easing, EasingFunction] = {
    Easing.LINEAR: _linear,
    Easing.EASE_IN: _ease_in,
    Easing.EASE_OUT: _ease_out,
    Easing.EASE_IN_OUT: _ease_in_out,
    Easing.EASE_IN_QUAD: _ease_in_quad,
    Easing.EASE_OUT_QUAD: _ease_out_quad,
    Easing.EASE_IN_OUT_QUAD: _ease_in_out_quad,
    Easing.EASE_IN_CUBIC: _ease_in_cubic,
    Easing.EASE_OUT_CUBIC: _ease_out_cubic,
    Easing.EASE_IN_OUT_CUBIC: _ease_in_out_cubic,
    Easing.EASE_IN_EXPO: _ease_in_expo,
    Easing.EASE_OUT_EXPO: _ease_out_expo,
    Easing.EASE_IN_OUT_EXPO: _ease_in_out_expo,
    Easing.EASE_IN_BACK: _ease_in_back,
    Easing.EASE_OUT_BACK: _ease_out_back,
    Easing.EASE_IN_OUT_BACK: _ease_in_out_back,
    Easing.EASE_IN_ELASTIC: _ease_in_elastic,
    Easing.EASE_OUT_ELASTIC: _ease_out_elastic,
    Easing.EASE_IN_OUT_ELASTIC: _ease_in_out_elastic,
    Easing.EASE_IN_BOUNCE: _ease_in_bounce,
    Easing.EASE_OUT_BOUNCE: _ease_out_bounce,
    Easing.EASE_IN_OUT_BOUNCE: _ease_in_out_bounce,
}


# =============================================================================
# TRANSITION INTERFACE
# =============================================================================


class ITransition(ABC):
    """
    Interface for screen transitions.

    Transitions animate the visual change between two screens.
    """

    @property
    @abstractmethod
    def duration(self) -> float:
        """Get the transition duration in seconds."""
        pass

    @property
    @abstractmethod
    def progress(self) -> float:
        """Get the current transition progress (0.0 to 1.0)."""
        pass

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        """Check if the transition is complete."""
        pass

    @property
    @abstractmethod
    def is_started(self) -> bool:
        """Check if the transition has started."""
        pass

    @abstractmethod
    def start(
        self,
        exiting_screen: Optional["Screen"],
        entering_screen: Optional["Screen"],
    ) -> None:
        """
        Start the transition.

        Args:
            exiting_screen: Screen that is leaving (may be None)
            entering_screen: Screen that is entering (may be None)
        """
        pass

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """
        Update the transition.

        Args:
            delta_time: Time since last update in seconds
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the transition to initial state."""
        pass

    @abstractmethod
    def get_exiting_transform(self) -> Dict[str, Any]:
        """
        Get the transform to apply to the exiting screen.

        Returns:
            Dictionary with transform properties (alpha, position, scale, etc.)
        """
        pass

    @abstractmethod
    def get_entering_transform(self) -> Dict[str, Any]:
        """
        Get the transform to apply to the entering screen.

        Returns:
            Dictionary with transform properties (alpha, position, scale, etc.)
        """
        pass


# =============================================================================
# BASE TRANSITION
# =============================================================================


class BaseTransition(ITransition):
    """
    Base class for transitions with common functionality.
    """

    def __init__(
        self,
        duration: float = 0.3,
        easing: Easing = Easing.EASE_IN_OUT,
    ) -> None:
        """
        Initialize the transition.

        Args:
            duration: Transition duration in seconds
            easing: Easing function to use
        """
        if duration < 0:
            raise ValueError("duration must be non-negative")

        self._duration = duration
        self._easing = easing
        self._easing_func = get_easing_function(easing)
        self._elapsed = 0.0
        self._started = False
        self._exiting_screen: Optional["Screen"] = None
        self._entering_screen: Optional["Screen"] = None

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        if value < 0:
            raise ValueError("duration must be non-negative")
        self._duration = value

    @property
    def easing(self) -> Easing:
        return self._easing

    @easing.setter
    def easing(self, value: Easing) -> None:
        self._easing = value
        self._easing_func = get_easing_function(value)

    @property
    def progress(self) -> float:
        if self._duration == 0:
            return 1.0
        return min(1.0, self._elapsed / self._duration)

    @property
    def eased_progress(self) -> float:
        """Get the eased progress value."""
        return self._easing_func(self.progress)

    @property
    def is_complete(self) -> bool:
        return self._elapsed >= self._duration

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def exiting_screen(self) -> Optional["Screen"]:
        return self._exiting_screen

    @property
    def entering_screen(self) -> Optional["Screen"]:
        return self._entering_screen

    def start(
        self,
        exiting_screen: Optional["Screen"],
        entering_screen: Optional["Screen"],
    ) -> None:
        self._exiting_screen = exiting_screen
        self._entering_screen = entering_screen
        self._elapsed = 0.0
        self._started = True

    def update(self, delta_time: float) -> None:
        if not self._started:
            return

        self._elapsed += delta_time

    def reset(self) -> None:
        self._elapsed = 0.0
        self._started = False
        self._exiting_screen = None
        self._entering_screen = None

    def get_exiting_transform(self) -> Dict[str, Any]:
        return {"alpha": 1.0, "x": 0.0, "y": 0.0, "scale": 1.0}

    def get_entering_transform(self) -> Dict[str, Any]:
        return {"alpha": 1.0, "x": 0.0, "y": 0.0, "scale": 1.0}


# =============================================================================
# FADE TRANSITION
# =============================================================================


class FadeTransition(BaseTransition):
    """
    Fade transition (alpha blend between screens).

    The exiting screen fades out while the entering screen fades in.
    """

    def __init__(
        self,
        duration: float = 0.3,
        easing: Easing = Easing.EASE_IN_OUT,
        crossfade: bool = True,
    ) -> None:
        """
        Initialize the fade transition.

        Args:
            duration: Transition duration in seconds
            easing: Easing function to use
            crossfade: If True, screens crossfade; if False, fade out then in
        """
        super().__init__(duration, easing)
        self._crossfade = crossfade

    @property
    def crossfade(self) -> bool:
        return self._crossfade

    @crossfade.setter
    def crossfade(self, value: bool) -> None:
        self._crossfade = value

    def get_exiting_transform(self) -> Dict[str, Any]:
        t = self.eased_progress

        if self._crossfade:
            # Fade out throughout transition
            alpha = 1.0 - t
        else:
            # Fade out in first half
            alpha = 1.0 - min(1.0, t * 2.0)

        return {"alpha": alpha, "x": 0.0, "y": 0.0, "scale": 1.0}

    def get_entering_transform(self) -> Dict[str, Any]:
        t = self.eased_progress

        if self._crossfade:
            # Fade in throughout transition
            alpha = t
        else:
            # Fade in in second half
            alpha = max(0.0, (t - 0.5) * 2.0)

        return {"alpha": alpha, "x": 0.0, "y": 0.0, "scale": 1.0}


# =============================================================================
# SLIDE TRANSITION
# =============================================================================


# Default screen dimensions (can be overridden via set_screen_size)
DEFAULT_SCREEN_WIDTH = 1920.0
DEFAULT_SCREEN_HEIGHT = 1080.0


class SlideTransition(BaseTransition):
    """
    Slide transition (screens slide in/out from a direction).
    """

    def __init__(
        self,
        duration: float = 0.3,
        easing: Easing = Easing.EASE_OUT,
        direction: TransitionDirection = TransitionDirection.LEFT,
        push: bool = True,
        screen_width: float = DEFAULT_SCREEN_WIDTH,
        screen_height: float = DEFAULT_SCREEN_HEIGHT,
    ) -> None:
        """
        Initialize the slide transition.

        Args:
            duration: Transition duration in seconds
            easing: Easing function to use
            direction: Direction to slide from/to
            push: If True, both screens move; if False, only entering screen moves
            screen_width: Screen width for slide calculations
            screen_height: Screen height for slide calculations
        """
        super().__init__(duration, easing)
        self._direction = direction
        self._push = push
        self._screen_width = screen_width
        self._screen_height = screen_height

    @property
    def direction(self) -> TransitionDirection:
        return self._direction

    @direction.setter
    def direction(self, value: TransitionDirection) -> None:
        self._direction = value

    @property
    def push(self) -> bool:
        return self._push

    @push.setter
    def push(self, value: bool) -> None:
        self._push = value

    def set_screen_size(self, width: float, height: float) -> None:
        """Set the screen dimensions for slide calculations."""
        self._screen_width = width
        self._screen_height = height

    def _get_direction_offset(self, progress: float) -> Tuple[float, float]:
        """Get the x, y offset for the slide direction."""
        if self._direction == TransitionDirection.LEFT:
            return (-self._screen_width * progress, 0.0)
        elif self._direction == TransitionDirection.RIGHT:
            return (self._screen_width * progress, 0.0)
        elif self._direction == TransitionDirection.UP:
            return (0.0, -self._screen_height * progress)
        else:  # DOWN
            return (0.0, self._screen_height * progress)

    def get_exiting_transform(self) -> Dict[str, Any]:
        t = self.eased_progress

        if self._push:
            x, y = self._get_direction_offset(t)
        else:
            x, y = 0.0, 0.0

        return {"alpha": 1.0, "x": x, "y": y, "scale": 1.0}

    def get_entering_transform(self) -> Dict[str, Any]:
        t = self.eased_progress

        # Entering screen comes from the opposite direction
        if self._direction == TransitionDirection.LEFT:
            x = self._screen_width * (1.0 - t)
            y = 0.0
        elif self._direction == TransitionDirection.RIGHT:
            x = -self._screen_width * (1.0 - t)
            y = 0.0
        elif self._direction == TransitionDirection.UP:
            x = 0.0
            y = self._screen_height * (1.0 - t)
        else:  # DOWN
            x = 0.0
            y = -self._screen_height * (1.0 - t)

        return {"alpha": 1.0, "x": x, "y": y, "scale": 1.0}


# =============================================================================
# ZOOM TRANSITION
# =============================================================================


class ZoomTransition(BaseTransition):
    """
    Zoom transition (scale in/out effect).
    """

    def __init__(
        self,
        duration: float = 0.3,
        easing: Easing = Easing.EASE_IN_OUT,
        zoom_in: bool = True,
        min_scale: float = 0.8,
        max_scale: float = 1.2,
        fade: bool = True,
    ) -> None:
        """
        Initialize the zoom transition.

        Args:
            duration: Transition duration in seconds
            easing: Easing function to use
            zoom_in: If True, entering screen zooms in; if False, zooms out
            min_scale: Minimum scale value
            max_scale: Maximum scale value
            fade: Whether to also apply a fade effect
        """
        super().__init__(duration, easing)
        self._zoom_in = zoom_in
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._fade = fade

    @property
    def zoom_in(self) -> bool:
        return self._zoom_in

    @zoom_in.setter
    def zoom_in(self, value: bool) -> None:
        self._zoom_in = value

    @property
    def min_scale(self) -> float:
        return self._min_scale

    @min_scale.setter
    def min_scale(self, value: float) -> None:
        self._min_scale = max(0.0, value)

    @property
    def max_scale(self) -> float:
        return self._max_scale

    @max_scale.setter
    def max_scale(self, value: float) -> None:
        self._max_scale = max(self._min_scale, value)

    @property
    def fade(self) -> bool:
        return self._fade

    @fade.setter
    def fade(self, value: bool) -> None:
        self._fade = value

    def get_exiting_transform(self) -> Dict[str, Any]:
        t = self.eased_progress

        if self._zoom_in:
            # Exiting screen zooms out
            scale = 1.0 + (self._max_scale - 1.0) * t
        else:
            # Exiting screen shrinks
            scale = 1.0 - (1.0 - self._min_scale) * t

        alpha = 1.0 - t if self._fade else 1.0

        return {"alpha": alpha, "x": 0.0, "y": 0.0, "scale": scale}

    def get_entering_transform(self) -> Dict[str, Any]:
        t = self.eased_progress

        if self._zoom_in:
            # Entering screen starts small and grows
            scale = self._min_scale + (1.0 - self._min_scale) * t
        else:
            # Entering screen starts large and shrinks to normal
            scale = self._max_scale - (self._max_scale - 1.0) * t

        alpha = t if self._fade else 1.0

        return {"alpha": alpha, "x": 0.0, "y": 0.0, "scale": scale}


# =============================================================================
# INSTANT TRANSITION
# =============================================================================


class InstantTransition(ITransition):
    """
    Instant transition (no animation).
    """

    def __init__(self) -> None:
        self._started = False
        self._complete = False

    @property
    def duration(self) -> float:
        return 0.0

    @property
    def progress(self) -> float:
        return 1.0 if self._complete else 0.0

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def is_started(self) -> bool:
        return self._started

    def start(
        self,
        exiting_screen: Optional["Screen"],
        entering_screen: Optional["Screen"],
    ) -> None:
        self._started = True
        self._complete = True

    def update(self, delta_time: float) -> None:
        pass

    def reset(self) -> None:
        self._started = False
        self._complete = False

    def get_exiting_transform(self) -> Dict[str, Any]:
        return {"alpha": 0.0, "x": 0.0, "y": 0.0, "scale": 1.0}

    def get_entering_transform(self) -> Dict[str, Any]:
        return {"alpha": 1.0, "x": 0.0, "y": 0.0, "scale": 1.0}


# =============================================================================
# COMPOSITE TRANSITION
# =============================================================================


class CompositeTransition(BaseTransition):
    """
    Composite transition combining multiple effects.
    """

    def __init__(
        self,
        duration: float = 0.3,
        easing: Easing = Easing.EASE_IN_OUT,
    ) -> None:
        """
        Initialize the composite transition.

        Args:
            duration: Transition duration in seconds
            easing: Easing function to use
        """
        super().__init__(duration, easing)
        self._effects: List[ITransition] = []

    def add_effect(self, effect: ITransition) -> "CompositeTransition":
        """Add an effect to the composite."""
        self._effects.append(effect)
        return self

    def clear_effects(self) -> None:
        """Clear all effects."""
        self._effects.clear()

    def start(
        self,
        exiting_screen: Optional["Screen"],
        entering_screen: Optional["Screen"],
    ) -> None:
        super().start(exiting_screen, entering_screen)
        for effect in self._effects:
            effect.start(exiting_screen, entering_screen)

    def update(self, delta_time: float) -> None:
        super().update(delta_time)
        for effect in self._effects:
            effect.update(delta_time)

    def reset(self) -> None:
        super().reset()
        for effect in self._effects:
            effect.reset()

    def get_exiting_transform(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"alpha": 1.0, "x": 0.0, "y": 0.0, "scale": 1.0}

        for effect in self._effects:
            transform = effect.get_exiting_transform()
            # Combine transforms
            result["alpha"] *= transform.get("alpha", 1.0)
            result["x"] += transform.get("x", 0.0)
            result["y"] += transform.get("y", 0.0)
            result["scale"] *= transform.get("scale", 1.0)

        return result

    def get_entering_transform(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"alpha": 1.0, "x": 0.0, "y": 0.0, "scale": 1.0}

        for effect in self._effects:
            transform = effect.get_entering_transform()
            # Combine transforms
            result["alpha"] *= transform.get("alpha", 1.0)
            result["x"] += transform.get("x", 0.0)
            result["y"] += transform.get("y", 0.0)
            result["scale"] *= transform.get("scale", 1.0)

        return result


# =============================================================================
# CUSTOM TRANSITION
# =============================================================================


TransformFunction = Callable[[float], Dict[str, Any]]


class CustomTransition(BaseTransition):
    """
    Custom transition with user-provided transform functions.
    """

    def __init__(
        self,
        duration: float = 0.3,
        easing: Easing = Easing.EASE_IN_OUT,
        exiting_transform: Optional[TransformFunction] = None,
        entering_transform: Optional[TransformFunction] = None,
    ) -> None:
        """
        Initialize the custom transition.

        Args:
            duration: Transition duration in seconds
            easing: Easing function to use
            exiting_transform: Function that returns transform for exiting screen
            entering_transform: Function that returns transform for entering screen
        """
        super().__init__(duration, easing)
        self._exiting_transform_func = exiting_transform
        self._entering_transform_func = entering_transform

    def set_exiting_transform(self, func: TransformFunction) -> None:
        """Set the exiting screen transform function."""
        self._exiting_transform_func = func

    def set_entering_transform(self, func: TransformFunction) -> None:
        """Set the entering screen transform function."""
        self._entering_transform_func = func

    def get_exiting_transform(self) -> Dict[str, Any]:
        if self._exiting_transform_func:
            return self._exiting_transform_func(self.eased_progress)
        return super().get_exiting_transform()

    def get_entering_transform(self) -> Dict[str, Any]:
        if self._entering_transform_func:
            return self._entering_transform_func(self.eased_progress)
        return super().get_entering_transform()


# =============================================================================
# TRANSITION FACTORY
# =============================================================================


class TransitionFactory:
    """
    Factory for creating common transitions.
    """

    @staticmethod
    def fade(duration: float = 0.3, crossfade: bool = True) -> FadeTransition:
        """Create a fade transition."""
        return FadeTransition(duration=duration, crossfade=crossfade)

    @staticmethod
    def slide_left(duration: float = 0.3, push: bool = True) -> SlideTransition:
        """Create a slide left transition."""
        return SlideTransition(
            duration=duration,
            direction=TransitionDirection.LEFT,
            push=push,
        )

    @staticmethod
    def slide_right(duration: float = 0.3, push: bool = True) -> SlideTransition:
        """Create a slide right transition."""
        return SlideTransition(
            duration=duration,
            direction=TransitionDirection.RIGHT,
            push=push,
        )

    @staticmethod
    def slide_up(duration: float = 0.3, push: bool = True) -> SlideTransition:
        """Create a slide up transition."""
        return SlideTransition(
            duration=duration,
            direction=TransitionDirection.UP,
            push=push,
        )

    @staticmethod
    def slide_down(duration: float = 0.3, push: bool = True) -> SlideTransition:
        """Create a slide down transition."""
        return SlideTransition(
            duration=duration,
            direction=TransitionDirection.DOWN,
            push=push,
        )

    @staticmethod
    def zoom_in(duration: float = 0.3, fade: bool = True) -> ZoomTransition:
        """Create a zoom in transition."""
        return ZoomTransition(duration=duration, zoom_in=True, fade=fade)

    @staticmethod
    def zoom_out(duration: float = 0.3, fade: bool = True) -> ZoomTransition:
        """Create a zoom out transition."""
        return ZoomTransition(duration=duration, zoom_in=False, fade=fade)

    @staticmethod
    def instant() -> InstantTransition:
        """Create an instant (no animation) transition."""
        return InstantTransition()


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    # Constants
    "DEFAULT_SCREEN_WIDTH",
    "DEFAULT_SCREEN_HEIGHT",
    "BACK_OVERSHOOT",
    "BOUNCE_AMPLITUDE",
    "BOUNCE_DIVISOR",
    # Enums
    "TransitionDirection",
    "Easing",
    # Types
    "EasingFunction",
    "TransformFunction",
    # Functions
    "get_easing_function",
    # Interface
    "ITransition",
    # Base class
    "BaseTransition",
    # Transitions
    "FadeTransition",
    "SlideTransition",
    "ZoomTransition",
    "InstantTransition",
    "CompositeTransition",
    "CustomTransition",
    # Factory
    "TransitionFactory",
]
