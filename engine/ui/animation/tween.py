"""
Tween animation system for UI animations.

Provides a powerful tweening system for animating properties over time,
including support for sequences, groups, delays, repeating, and yoyo effects.

Example usage:
    # Simple property tween
    tween = Tween(widget, "opacity", 1.0, 0.0, duration=0.5)
    tween.start()

    # Tween with callbacks
    tween = (
        Tween(widget, "x", 0, 100, duration=1.0)
        .set_easing("ease_out")
        .on_complete(lambda: print("Done!"))
    )

    # Sequence of tweens
    seq = TweenSequence([
        Tween(widget, "x", 0, 100, duration=0.5),
        Tween(widget, "y", 0, 100, duration=0.5),
    ])

    # Parallel tweens
    group = TweenGroup([
        Tween(widget, "x", 0, 100, duration=1.0),
        Tween(widget, "y", 0, 50, duration=1.0),
    ])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, Optional, TypeVar

from engine.ui.animation.easing import EasingFunction, EasingType, get_easing, lerp

T = TypeVar("T")


class TweenState(Enum):
    """State of a tween animation."""

    IDLE = auto()
    PLAYING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class LoopMode(Enum):
    """How a tween should loop."""

    NONE = auto()
    RESTART = auto()
    YOYO = auto()


@dataclass
class TweenConfig:
    """Configuration for a tween animation."""

    duration: float = 1.0
    delay: float = 0.0
    easing: str | EasingType | EasingFunction = "linear"
    repeat: int = 0  # 0 = no repeat, -1 = infinite
    yoyo: bool = False  # Reverse on repeat
    auto_start: bool = False


# Callback types
TweenCallback = Callable[[], None]
TweenUpdateCallback = Callable[[float], None]
TweenValueCallback = Callable[[Any], None]


class Tween(Generic[T]):
    """
    A tween animation that interpolates a property value over time.

    Type parameter T represents the value type being animated (typically float).
    """

    def __init__(
        self,
        target: Any,
        property_name: str,
        from_value: T,
        to_value: T,
        duration: float = 1.0,
        delay: float = 0.0,
        easing: str | EasingType | EasingFunction = "linear",
        repeat: int = 0,
        yoyo: bool = False,
        auto_start: bool = False,
    ) -> None:
        """
        Create a new tween animation.

        Args:
            target: The object whose property will be animated
            property_name: Name of the property to animate
            from_value: Starting value
            to_value: Ending value
            duration: Animation duration in seconds
            delay: Delay before starting in seconds
            easing: Easing function name, type, or callable
            repeat: Number of times to repeat (0=none, -1=infinite)
            yoyo: If True, reverse direction on each repeat
            auto_start: If True, start immediately on creation
        """
        self._target = target
        self._property_name = property_name
        self._from_value = from_value
        self._to_value = to_value
        self._duration = max(0.001, duration)  # Prevent division by zero
        self._delay = max(0.0, delay)
        self._repeat = repeat
        self._yoyo = yoyo

        # Resolve easing function
        if callable(easing) and not isinstance(easing, EasingType):
            self._easing_func = easing
        else:
            self._easing_func = get_easing(easing)

        # State
        self._state = TweenState.IDLE
        self._elapsed_time: float = 0.0
        self._delay_elapsed: float = 0.0
        self._current_repeat: int = 0
        self._is_reversed: bool = False
        self._current_value: T = from_value

        # Callbacks
        self._on_start: Optional[TweenCallback] = None
        self._on_update: Optional[TweenUpdateCallback] = None
        self._on_value_change: Optional[TweenValueCallback] = None
        self._on_complete: Optional[TweenCallback] = None
        self._on_repeat: Optional[TweenCallback] = None
        self._on_cancel: Optional[TweenCallback] = None

        if auto_start:
            self.start()

    @property
    def state(self) -> TweenState:
        """Current state of the tween."""
        return self._state

    @property
    def is_playing(self) -> bool:
        """Whether the tween is currently playing."""
        return self._state == TweenState.PLAYING

    @property
    def is_complete(self) -> bool:
        """Whether the tween has completed."""
        return self._state == TweenState.COMPLETED

    # Epsilon for floating point comparisons to avoid precision issues
    _EPSILON: float = 1e-9

    @property
    def progress(self) -> float:
        """Current normalized progress (0 to 1)."""
        if self._duration <= self._EPSILON:
            return 1.0
        return min(1.0, self._elapsed_time / self._duration)

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since start."""
        return self._elapsed_time

    @property
    def current_value(self) -> T:
        """Current interpolated value."""
        return self._current_value

    @property
    def target(self) -> Any:
        """The target object being animated."""
        return self._target

    @property
    def property_name(self) -> str:
        """The property being animated."""
        return self._property_name

    @property
    def from_value(self) -> T:
        """Starting value."""
        return self._from_value

    @property
    def to_value(self) -> T:
        """Ending value."""
        return self._to_value

    @property
    def duration(self) -> float:
        """Animation duration in seconds."""
        return self._duration

    # =========================================================================
    # FLUENT CONFIGURATION
    # =========================================================================

    def set_easing(self, easing: str | EasingType | EasingFunction) -> Tween[T]:
        """Set the easing function. Returns self for chaining."""
        if callable(easing) and not isinstance(easing, EasingType):
            self._easing_func = easing
        else:
            self._easing_func = get_easing(easing)
        return self

    def set_delay(self, delay: float) -> Tween[T]:
        """Set the delay before starting. Returns self for chaining."""
        self._delay = max(0.0, delay)
        return self

    def set_repeat(self, count: int, yoyo: bool = False) -> Tween[T]:
        """Set repeat count and yoyo mode. Returns self for chaining."""
        self._repeat = count
        self._yoyo = yoyo
        return self

    def on_start(self, callback: TweenCallback) -> Tween[T]:
        """Set callback for when tween starts. Returns self for chaining."""
        self._on_start = callback
        return self

    def on_update(self, callback: TweenUpdateCallback) -> Tween[T]:
        """Set callback for each update (receives progress). Returns self for chaining."""
        self._on_update = callback
        return self

    def on_value_change(self, callback: TweenValueCallback) -> Tween[T]:
        """Set callback for value changes. Returns self for chaining."""
        self._on_value_change = callback
        return self

    def on_complete(self, callback: TweenCallback) -> Tween[T]:
        """Set callback for when tween completes. Returns self for chaining."""
        self._on_complete = callback
        return self

    def on_repeat(self, callback: TweenCallback) -> Tween[T]:
        """Set callback for each repeat. Returns self for chaining."""
        self._on_repeat = callback
        return self

    def on_cancel(self, callback: TweenCallback) -> Tween[T]:
        """Set callback for when tween is cancelled. Returns self for chaining."""
        self._on_cancel = callback
        return self

    # =========================================================================
    # CONTROL METHODS
    # =========================================================================

    def start(self) -> Tween[T]:
        """Start or restart the tween from the beginning."""
        self._elapsed_time = 0.0
        self._delay_elapsed = 0.0
        self._current_repeat = 0
        self._is_reversed = False
        self._state = TweenState.PLAYING
        self._current_value = self._from_value
        self._apply_value(self._current_value)
        return self

    def pause(self) -> Tween[T]:
        """Pause the tween."""
        if self._state == TweenState.PLAYING:
            self._state = TweenState.PAUSED
        return self

    def resume(self) -> Tween[T]:
        """Resume a paused tween."""
        if self._state == TweenState.PAUSED:
            self._state = TweenState.PLAYING
        return self

    def stop(self, clear_callbacks: bool = False) -> Tween[T]:
        """
        Stop the tween and reset to initial state.

        Args:
            clear_callbacks: If True, also clear callback references to prevent memory leaks
        """
        self._state = TweenState.IDLE
        self._elapsed_time = 0.0
        self._delay_elapsed = 0.0
        self._current_repeat = 0
        self._is_reversed = False
        if clear_callbacks:
            self._on_start = None
            self._on_update = None
            self._on_value_change = None
            self._on_complete = None
            self._on_repeat = None
            self._on_cancel = None
        return self

    def cancel(self) -> Tween[T]:
        """Cancel the tween without completing."""
        if self._state in (TweenState.PLAYING, TweenState.PAUSED):
            self._state = TweenState.CANCELLED
            if self._on_cancel:
                self._on_cancel()
        return self

    def complete(self) -> Tween[T]:
        """Immediately complete the tween."""
        if self._state in (TweenState.PLAYING, TweenState.PAUSED, TweenState.IDLE):
            self._elapsed_time = self._duration
            self._current_value = self._to_value
            self._apply_value(self._current_value)
            self._state = TweenState.COMPLETED
            if self._on_complete:
                self._on_complete()
        return self

    def reverse(self) -> Tween[T]:
        """Reverse the tween direction."""
        self._from_value, self._to_value = self._to_value, self._from_value
        self._elapsed_time = self._duration - self._elapsed_time
        return self

    def update(self, delta_time: float) -> bool:
        """
        Update the tween by the given time delta.

        Args:
            delta_time: Time passed since last update in seconds

        Returns:
            True if the tween is still active, False if completed
        """
        if self._state != TweenState.PLAYING:
            return self._state not in (TweenState.COMPLETED, TweenState.CANCELLED)

        # Handle delay
        if self._delay_elapsed < self._delay:
            self._delay_elapsed += delta_time
            if self._delay_elapsed >= self._delay:
                # Delay just completed, fire start callback
                if self._on_start:
                    self._on_start()
                delta_time = self._delay_elapsed - self._delay
            else:
                return True

        # Update elapsed time
        self._elapsed_time += delta_time

        # Calculate progress and apply easing
        raw_progress = min(1.0, self._elapsed_time / self._duration)

        # Handle yoyo direction
        if self._is_reversed:
            raw_progress = 1.0 - raw_progress

        eased_progress = self._easing_func(raw_progress)

        # Interpolate value
        self._current_value = self._interpolate(
            self._from_value, self._to_value, eased_progress
        )

        # Apply to target
        self._apply_value(self._current_value)

        # Fire update callbacks
        if self._on_update:
            self._on_update(raw_progress)
        if self._on_value_change:
            self._on_value_change(self._current_value)

        # Check for completion (with epsilon to handle floating point precision)
        if self._elapsed_time >= self._duration - self._EPSILON:
            return self._handle_cycle_complete()

        return True

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _interpolate(self, from_val: T, to_val: T, t: float) -> T:
        """Interpolate between two values."""
        # Handle numeric types
        if isinstance(from_val, (int, float)) and isinstance(to_val, (int, float)):
            result = lerp(float(from_val), float(to_val), t)
            if isinstance(from_val, int) and isinstance(to_val, int):
                return round(result)  # type: ignore
            return result  # type: ignore

        # Handle tuple/list (for colors, vectors, etc.)
        if isinstance(from_val, (tuple, list)) and isinstance(to_val, (tuple, list)):
            interpolated = [
                self._interpolate(f, t_val, t)
                for f, t_val in zip(from_val, to_val)
            ]
            return type(from_val)(interpolated)  # type: ignore

        # Handle dict
        if isinstance(from_val, dict) and isinstance(to_val, dict):
            result = {}
            for key in from_val:
                if key in to_val:
                    result[key] = self._interpolate(from_val[key], to_val[key], t)
            return result  # type: ignore

        # Fallback: snap at halfway
        return to_val if t >= 0.5 else from_val

    def _apply_value(self, value: T) -> None:
        """Apply the interpolated value to the target."""
        if hasattr(self._target, self._property_name):
            setattr(self._target, self._property_name, value)

    def _handle_cycle_complete(self) -> bool:
        """Handle completion of one animation cycle. Returns True if continuing."""
        # Check for infinite or remaining repeats
        if self._repeat == -1 or self._current_repeat < self._repeat:
            self._current_repeat += 1

            # Fire repeat callback
            if self._on_repeat:
                self._on_repeat()

            # Handle yoyo
            if self._yoyo:
                self._is_reversed = not self._is_reversed

            # Reset for next cycle
            self._elapsed_time = 0.0
            return True

        # Animation complete
        self._state = TweenState.COMPLETED
        if self._on_complete:
            self._on_complete()
        return False


@dataclass
class TweenSequence:
    """
    A sequence of tweens that play one after another.

    Tweens in a sequence play in order - each tween starts
    only after the previous one completes.
    """

    tweens: list[Tween] = field(default_factory=list)
    _current_index: int = field(default=0, init=False)
    _state: TweenState = field(default=TweenState.IDLE, init=False)
    _on_complete: Optional[TweenCallback] = field(default=None, init=False)

    @property
    def state(self) -> TweenState:
        """Current state of the sequence."""
        return self._state

    @property
    def is_playing(self) -> bool:
        """Whether the sequence is currently playing."""
        return self._state == TweenState.PLAYING

    @property
    def is_complete(self) -> bool:
        """Whether the sequence has completed."""
        return self._state == TweenState.COMPLETED

    @property
    def current_tween(self) -> Optional[Tween]:
        """The currently active tween."""
        if 0 <= self._current_index < len(self.tweens):
            return self.tweens[self._current_index]
        return None

    @property
    def progress(self) -> float:
        """Overall progress of the sequence (0 to 1)."""
        if not self.tweens:
            return 1.0
        completed = self._current_index
        current_progress = 0.0
        if self.current_tween:
            current_progress = self.current_tween.progress
        return (completed + current_progress) / len(self.tweens)

    def add(self, tween: Tween) -> TweenSequence:
        """Add a tween to the sequence. Returns self for chaining."""
        self.tweens.append(tween)
        return self

    def on_complete(self, callback: TweenCallback) -> TweenSequence:
        """Set callback for when sequence completes. Returns self for chaining."""
        self._on_complete = callback
        return self

    def start(self) -> TweenSequence:
        """Start the sequence from the beginning."""
        self._current_index = 0
        self._state = TweenState.PLAYING
        if self.tweens:
            self.tweens[0].start()
        return self

    def pause(self) -> TweenSequence:
        """Pause the sequence."""
        if self._state == TweenState.PLAYING:
            self._state = TweenState.PAUSED
            if self.current_tween:
                self.current_tween.pause()
        return self

    def resume(self) -> TweenSequence:
        """Resume the sequence."""
        if self._state == TweenState.PAUSED:
            self._state = TweenState.PLAYING
            if self.current_tween:
                self.current_tween.resume()
        return self

    def stop(self) -> TweenSequence:
        """Stop the sequence."""
        self._state = TweenState.IDLE
        self._current_index = 0
        for tween in self.tweens:
            tween.stop()
        return self

    def cancel(self) -> TweenSequence:
        """Cancel the sequence."""
        if self._state in (TweenState.PLAYING, TweenState.PAUSED):
            self._state = TweenState.CANCELLED
            for tween in self.tweens:
                tween.cancel()
        return self

    def update(self, delta_time: float) -> bool:
        """
        Update the sequence.

        Args:
            delta_time: Time passed since last update

        Returns:
            True if still active, False if completed
        """
        if self._state != TweenState.PLAYING:
            return self._state not in (TweenState.COMPLETED, TweenState.CANCELLED)

        if not self.tweens:
            self._state = TweenState.COMPLETED
            if self._on_complete:
                self._on_complete()
            return False

        # Update current tween
        current = self.current_tween
        if current:
            if not current.update(delta_time):
                # Current tween completed, move to next
                self._current_index += 1
                if self._current_index < len(self.tweens):
                    self.tweens[self._current_index].start()
                else:
                    self._state = TweenState.COMPLETED
                    if self._on_complete:
                        self._on_complete()
                    return False

        return True


@dataclass
class TweenGroup:
    """
    A group of tweens that play simultaneously.

    All tweens in a group start together and play in parallel.
    The group completes when all tweens have completed.
    """

    tweens: list[Tween] = field(default_factory=list)
    _state: TweenState = field(default=TweenState.IDLE, init=False)
    _on_complete: Optional[TweenCallback] = field(default=None, init=False)

    @property
    def state(self) -> TweenState:
        """Current state of the group."""
        return self._state

    @property
    def is_playing(self) -> bool:
        """Whether the group is currently playing."""
        return self._state == TweenState.PLAYING

    @property
    def is_complete(self) -> bool:
        """Whether the group has completed."""
        return self._state == TweenState.COMPLETED

    @property
    def progress(self) -> float:
        """Overall progress of the group (0 to 1)."""
        if not self.tweens:
            return 1.0
        return sum(t.progress for t in self.tweens) / len(self.tweens)

    def add(self, tween: Tween) -> TweenGroup:
        """Add a tween to the group. Returns self for chaining."""
        self.tweens.append(tween)
        return self

    def on_complete(self, callback: TweenCallback) -> TweenGroup:
        """Set callback for when group completes. Returns self for chaining."""
        self._on_complete = callback
        return self

    def start(self) -> TweenGroup:
        """Start all tweens in the group."""
        self._state = TweenState.PLAYING
        for tween in self.tweens:
            tween.start()
        return self

    def pause(self) -> TweenGroup:
        """Pause all tweens in the group."""
        if self._state == TweenState.PLAYING:
            self._state = TweenState.PAUSED
            for tween in self.tweens:
                tween.pause()
        return self

    def resume(self) -> TweenGroup:
        """Resume all tweens in the group."""
        if self._state == TweenState.PAUSED:
            self._state = TweenState.PLAYING
            for tween in self.tweens:
                tween.resume()
        return self

    def stop(self) -> TweenGroup:
        """Stop all tweens in the group."""
        self._state = TweenState.IDLE
        for tween in self.tweens:
            tween.stop()
        return self

    def cancel(self) -> TweenGroup:
        """Cancel all tweens in the group."""
        if self._state in (TweenState.PLAYING, TweenState.PAUSED):
            self._state = TweenState.CANCELLED
            for tween in self.tweens:
                tween.cancel()
        return self

    def update(self, delta_time: float) -> bool:
        """
        Update all tweens in the group.

        Args:
            delta_time: Time passed since last update

        Returns:
            True if still active, False if all completed
        """
        if self._state != TweenState.PLAYING:
            return self._state not in (TweenState.COMPLETED, TweenState.CANCELLED)

        if not self.tweens:
            self._state = TweenState.COMPLETED
            if self._on_complete:
                self._on_complete()
            return False

        # Update all tweens
        all_complete = True
        for tween in self.tweens:
            if tween.update(delta_time):
                all_complete = False

        if all_complete:
            self._state = TweenState.COMPLETED
            if self._on_complete:
                self._on_complete()
            return False

        return True


class TweenManager:
    """
    Manages active tweens and provides factory methods for creating them.

    This is a convenience class for managing multiple tweens.
    Call update() each frame to advance all active tweens.
    """

    def __init__(self) -> None:
        self._tweens: list[Tween | TweenSequence | TweenGroup] = []
        self._to_remove: list[Tween | TweenSequence | TweenGroup] = []

    @property
    def active_count(self) -> int:
        """Number of active tweens/sequences/groups."""
        return len(self._tweens)

    def create(
        self,
        target: Any,
        property_name: str,
        to_value: Any,
        duration: float = 1.0,
        **kwargs: Any,
    ) -> Tween:
        """
        Create and register a new tween.

        The from_value is automatically read from the target.

        Args:
            target: Object to animate
            property_name: Property to animate
            to_value: Target value
            duration: Animation duration
            **kwargs: Additional Tween arguments

        Returns:
            The created Tween
        """
        from_value = getattr(target, property_name)
        tween = Tween(
            target, property_name, from_value, to_value, duration, **kwargs
        )
        self._tweens.append(tween)
        return tween

    def add(self, item: Tween | TweenSequence | TweenGroup) -> None:
        """Add a tween, sequence, or group to be managed."""
        self._tweens.append(item)

    def remove(self, item: Tween | TweenSequence | TweenGroup) -> None:
        """Remove a tween, sequence, or group from management."""
        self._to_remove.append(item)

    def clear(self) -> None:
        """Remove all managed tweens."""
        for tween in self._tweens:
            if hasattr(tween, "cancel"):
                tween.cancel()
        self._tweens.clear()
        self._to_remove.clear()

    def update(self, delta_time: float) -> None:
        """
        Update all managed tweens.

        Call this once per frame with the time since last frame.

        Args:
            delta_time: Time passed since last update
        """
        # Process removals
        for item in self._to_remove:
            if item in self._tweens:
                self._tweens.remove(item)
        self._to_remove.clear()

        # Update all tweens
        completed = []
        for tween in self._tweens:
            if not tween.update(delta_time):
                completed.append(tween)

        # Remove completed tweens and clear their callbacks to prevent memory leaks
        for tween in completed:
            self._tweens.remove(tween)
            # Clear callbacks on completed tweens to release references
            if hasattr(tween, 'stop'):
                tween.stop(clear_callbacks=True)


# Factory functions for convenience
def tween_to(
    target: Any,
    property_name: str,
    to_value: Any,
    duration: float = 1.0,
    easing: str = "linear",
    **kwargs: Any,
) -> Tween:
    """
    Create a tween that animates from the current value to a target value.

    Args:
        target: Object to animate
        property_name: Property to animate
        to_value: Target value
        duration: Animation duration
        easing: Easing function name
        **kwargs: Additional Tween arguments

    Returns:
        The created Tween
    """
    from_value = getattr(target, property_name)
    return Tween(
        target, property_name, from_value, to_value, duration, easing=easing, **kwargs
    )


def tween_from(
    target: Any,
    property_name: str,
    from_value: Any,
    duration: float = 1.0,
    easing: str = "linear",
    **kwargs: Any,
) -> Tween:
    """
    Create a tween that animates from a value to the current value.

    Args:
        target: Object to animate
        property_name: Property to animate
        from_value: Starting value
        duration: Animation duration
        easing: Easing function name
        **kwargs: Additional Tween arguments

    Returns:
        The created Tween
    """
    to_value = getattr(target, property_name)
    return Tween(
        target, property_name, from_value, to_value, duration, easing=easing, **kwargs
    )


def tween_by(
    target: Any,
    property_name: str,
    delta: Any,
    duration: float = 1.0,
    easing: str = "linear",
    **kwargs: Any,
) -> Tween:
    """
    Create a tween that animates by a relative amount.

    Args:
        target: Object to animate
        property_name: Property to animate
        delta: Amount to change by
        duration: Animation duration
        easing: Easing function name
        **kwargs: Additional Tween arguments

    Returns:
        The created Tween
    """
    from_value = getattr(target, property_name)
    to_value = from_value + delta
    return Tween(
        target, property_name, from_value, to_value, duration, easing=easing, **kwargs
    )


__all__ = [
    # Core
    "Tween",
    "TweenState",
    "TweenConfig",
    "LoopMode",
    # Sequences/Groups
    "TweenSequence",
    "TweenGroup",
    # Manager
    "TweenManager",
    # Factory functions
    "tween_to",
    "tween_from",
    "tween_by",
    # Callback types
    "TweenCallback",
    "TweenUpdateCallback",
    "TweenValueCallback",
]
