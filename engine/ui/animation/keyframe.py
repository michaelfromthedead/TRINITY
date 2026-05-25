"""
Keyframe animation system for UI animations.

Provides keyframe-based animations with:
- Keyframes with time, value, and easing
- Tracks for animating individual properties
- Multi-track animations
- Loop modes (once, loop, ping_pong)
- Interpolation between keyframes

Example usage:
    # Create a fade-in animation
    animation = KeyframeAnimation("fade_in", duration=1.0)
    animation.add_track(KeyframeTrack(
        "opacity",
        keyframes=[
            Keyframe(0.0, 0.0),
            Keyframe(0.5, 0.8, "ease_out"),
            Keyframe(1.0, 1.0, "ease_in_out"),
        ]
    ))

    # Create a complex animation
    bounce_in = KeyframeAnimation("bounce_in")
    bounce_in.add_track(create_property_track(
        "y", [(0.0, 100), (0.5, -10), (0.7, 5), (1.0, 0)]
    ))
    bounce_in.add_track(create_property_track(
        "scale", [(0.0, 0.5), (0.6, 1.1), (1.0, 1.0)]
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from engine.ui.animation.easing import EasingFunction, EasingType, get_easing, lerp

T = TypeVar("T")

# Epsilon for floating point comparisons to avoid precision issues
_EPSILON: float = 1e-9


class LoopMode(Enum):
    """How a keyframe animation should loop."""

    ONCE = auto()       # Play once and stop
    LOOP = auto()       # Loop back to start
    PING_PONG = auto()  # Reverse direction at ends


class AnimationState(Enum):
    """State of a keyframe animation."""

    IDLE = auto()
    PLAYING = auto()
    PAUSED = auto()
    COMPLETED = auto()


# Callback types
AnimationCallback = Callable[[], None]
KeyframeCallback = Callable[[int], None]  # Receives keyframe index


@dataclass
class Keyframe(Generic[T]):
    """
    A single keyframe in an animation track.

    Represents a value at a specific time with optional easing
    for interpolation to the next keyframe.
    """

    time: float  # Normalized time (0 to 1)
    value: T
    easing: str | EasingType | EasingFunction = "linear"

    # Resolved easing function (set during track compilation)
    _easing_func: Optional[EasingFunction] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Resolve the easing function."""
        self._resolve_easing()

    def _resolve_easing(self) -> None:
        """Resolve the easing to a function."""
        if callable(self.easing) and not isinstance(self.easing, EasingType):
            self._easing_func = self.easing
        else:
            self._easing_func = get_easing(self.easing)

    def get_eased_progress(self, t: float) -> float:
        """
        Apply easing to a progress value.

        Args:
            t: Linear progress (0 to 1)

        Returns:
            Eased progress
        """
        if self._easing_func:
            return self._easing_func(t)
        return t


@dataclass
class KeyframeTrack(Generic[T]):
    """
    A track containing keyframes for a single property.

    Tracks manage keyframe interpolation for one property,
    handling sorting, interpolation, and value calculation.
    """

    property_name: str
    keyframes: List[Keyframe[T]] = field(default_factory=list)

    # Callbacks
    on_keyframe_reached: Optional[KeyframeCallback] = None

    # Runtime state
    _sorted: bool = field(default=False, init=False)
    _last_keyframe_index: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        """Sort keyframes on creation."""
        self._sort_keyframes()

    def _sort_keyframes(self) -> None:
        """Sort keyframes by time."""
        self.keyframes.sort(key=lambda k: k.time)
        self._sorted = True

    def add_keyframe(
        self,
        time: float,
        value: T,
        easing: str | EasingType | EasingFunction = "linear",
    ) -> KeyframeTrack[T]:
        """
        Add a keyframe to the track.

        Args:
            time: Normalized time (0 to 1)
            value: Value at this keyframe
            easing: Easing to use when interpolating to next keyframe

        Returns:
            Self for chaining
        """
        keyframe = Keyframe(time=time, value=value, easing=easing)
        self.keyframes.append(keyframe)
        self._sorted = False
        return self

    def remove_keyframe(self, index: int) -> bool:
        """
        Remove a keyframe by index.

        Args:
            index: Index of the keyframe to remove

        Returns:
            True if removed, False if invalid index
        """
        if 0 <= index < len(self.keyframes):
            self.keyframes.pop(index)
            return True
        return False

    def clear(self) -> KeyframeTrack[T]:
        """
        Remove all keyframes.

        Returns:
            Self for chaining
        """
        self.keyframes.clear()
        self._last_keyframe_index = -1
        return self

    def get_value_at(self, time: float) -> Optional[T]:
        """
        Get the interpolated value at a specific time.

        Args:
            time: Normalized time (0 to 1)

        Returns:
            The interpolated value, or None if no keyframes
        """
        if not self.keyframes:
            return None

        if not self._sorted:
            self._sort_keyframes()

        # Find surrounding keyframes
        before: Optional[Keyframe[T]] = None
        after: Optional[Keyframe[T]] = None
        before_index: int = -1

        for i, keyframe in enumerate(self.keyframes):
            if keyframe.time <= time:
                before = keyframe
                before_index = i
            if keyframe.time >= time:
                after = keyframe
                break

        # Fire keyframe reached callback if we crossed a keyframe
        if before_index != self._last_keyframe_index and before_index >= 0:
            self._last_keyframe_index = before_index
            if self.on_keyframe_reached:
                self.on_keyframe_reached(before_index)

        # Handle edge cases
        if before is None:
            return self.keyframes[0].value
        if after is None:
            return self.keyframes[-1].value
        if before is after:
            return before.value

        # Interpolate between keyframes
        segment_duration = after.time - before.time
        if segment_duration <= _EPSILON:
            return before.value

        segment_progress = (time - before.time) / segment_duration
        eased_progress = before.get_eased_progress(segment_progress)

        return self._interpolate(before.value, after.value, eased_progress)

    def _interpolate(self, from_val: T, to_val: T, t: float) -> T:
        """
        Interpolate between two values.

        Args:
            from_val: Starting value
            to_val: Ending value
            t: Progress (0 to 1)

        Returns:
            Interpolated value
        """
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

    def reset(self) -> None:
        """Reset track state for replay."""
        self._last_keyframe_index = -1


class KeyframeAnimation:
    """
    A complete keyframe animation with multiple tracks.

    Animations can contain multiple tracks affecting different properties,
    all synchronized to the same timeline.
    """

    def __init__(
        self,
        name: str,
        duration: float = 1.0,
        loop_mode: LoopMode = LoopMode.ONCE,
        speed: float = 1.0,
    ) -> None:
        """
        Create a keyframe animation.

        Args:
            name: Unique name for the animation
            duration: Animation duration in seconds
            loop_mode: How the animation should loop
            speed: Playback speed multiplier
        """
        self._name = name
        self._duration = max(0.001, duration)
        self._loop_mode = loop_mode
        self._speed = speed

        self._tracks: Dict[str, KeyframeTrack] = {}
        self._state = AnimationState.IDLE
        self._elapsed: float = 0.0
        self._direction: int = 1  # 1 = forward, -1 = backward
        self._loop_count: int = 0

        # Target
        self._target: Optional[Any] = None

        # Callbacks
        self._on_start: Optional[AnimationCallback] = None
        self._on_complete: Optional[AnimationCallback] = None
        self._on_loop: Optional[AnimationCallback] = None
        self._on_update: Optional[Callable[[float], None]] = None

    @property
    def name(self) -> str:
        """Animation name."""
        return self._name

    @property
    def duration(self) -> float:
        """Animation duration in seconds."""
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        """Set animation duration."""
        self._duration = max(0.001, value)

    @property
    def loop_mode(self) -> LoopMode:
        """Loop mode."""
        return self._loop_mode

    @loop_mode.setter
    def loop_mode(self, value: LoopMode) -> None:
        """Set loop mode."""
        self._loop_mode = value

    @property
    def speed(self) -> float:
        """Playback speed."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Set playback speed."""
        self._speed = value

    @property
    def state(self) -> AnimationState:
        """Current animation state."""
        return self._state

    @property
    def is_playing(self) -> bool:
        """Whether the animation is playing."""
        return self._state == AnimationState.PLAYING

    @property
    def is_complete(self) -> bool:
        """Whether the animation has completed."""
        return self._state == AnimationState.COMPLETED

    @property
    def progress(self) -> float:
        """Current progress (0 to 1)."""
        return min(1.0, self._elapsed / self._duration)

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return self._elapsed

    @property
    def loop_count(self) -> int:
        """Number of times the animation has looped."""
        return self._loop_count

    @property
    def tracks(self) -> Dict[str, KeyframeTrack]:
        """All animation tracks."""
        return self._tracks.copy()

    @property
    def target(self) -> Optional[Any]:
        """The target object being animated."""
        return self._target

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def set_target(self, target: Any) -> KeyframeAnimation:
        """
        Set the target object to animate.

        Args:
            target: Object whose properties will be animated

        Returns:
            Self for chaining
        """
        self._target = target
        return self

    def add_track(self, track: KeyframeTrack) -> KeyframeAnimation:
        """
        Add a track to the animation.

        Args:
            track: The KeyframeTrack to add

        Returns:
            Self for chaining
        """
        self._tracks[track.property_name] = track
        return self

    def remove_track(self, property_name: str) -> bool:
        """
        Remove a track by property name.

        Args:
            property_name: Name of the property track to remove

        Returns:
            True if removed, False if not found
        """
        if property_name in self._tracks:
            del self._tracks[property_name]
            return True
        return False

    def get_track(self, property_name: str) -> Optional[KeyframeTrack]:
        """
        Get a track by property name.

        Args:
            property_name: Name of the property

        Returns:
            The KeyframeTrack or None
        """
        return self._tracks.get(property_name)

    def clear_tracks(self) -> KeyframeAnimation:
        """
        Remove all tracks.

        Returns:
            Self for chaining
        """
        self._tracks.clear()
        return self

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_start(self, callback: AnimationCallback) -> KeyframeAnimation:
        """Set callback for when animation starts."""
        self._on_start = callback
        return self

    def on_complete(self, callback: AnimationCallback) -> KeyframeAnimation:
        """Set callback for when animation completes."""
        self._on_complete = callback
        return self

    def on_loop(self, callback: AnimationCallback) -> KeyframeAnimation:
        """Set callback for when animation loops."""
        self._on_loop = callback
        return self

    def on_update(self, callback: Callable[[float], None]) -> KeyframeAnimation:
        """Set callback for each update (receives progress)."""
        self._on_update = callback
        return self

    # =========================================================================
    # CONTROL
    # =========================================================================

    def start(self) -> KeyframeAnimation:
        """
        Start the animation from the beginning.

        Returns:
            Self for chaining
        """
        self._elapsed = 0.0
        self._direction = 1
        self._loop_count = 0
        self._state = AnimationState.PLAYING

        # Reset all tracks
        for track in self._tracks.values():
            track.reset()

        if self._on_start:
            self._on_start()

        # Apply initial values
        self._apply_values(0.0)

        return self

    def pause(self) -> KeyframeAnimation:
        """
        Pause the animation.

        Returns:
            Self for chaining
        """
        if self._state == AnimationState.PLAYING:
            self._state = AnimationState.PAUSED
        return self

    def resume(self) -> KeyframeAnimation:
        """
        Resume a paused animation.

        Returns:
            Self for chaining
        """
        if self._state == AnimationState.PAUSED:
            self._state = AnimationState.PLAYING
        return self

    def stop(self, clear_callbacks: bool = False) -> KeyframeAnimation:
        """
        Stop the animation and reset.

        Args:
            clear_callbacks: If True, also clear callback references to prevent memory leaks

        Returns:
            Self for chaining
        """
        self._state = AnimationState.IDLE
        self._elapsed = 0.0
        self._direction = 1
        self._loop_count = 0

        for track in self._tracks.values():
            track.reset()

        if clear_callbacks:
            self._on_start = None
            self._on_complete = None
            self._on_loop = None
            self._on_update = None

        return self

    def complete(self) -> KeyframeAnimation:
        """
        Jump to the end of the animation.

        Returns:
            Self for chaining
        """
        self._elapsed = self._duration
        self._apply_values(1.0)
        self._state = AnimationState.COMPLETED

        if self._on_complete:
            self._on_complete()

        return self

    def seek(self, time: float) -> KeyframeAnimation:
        """
        Seek to a specific time.

        Args:
            time: Time to seek to (in seconds)

        Returns:
            Self for chaining
        """
        self._elapsed = max(0.0, min(self._duration, time))
        self._apply_values(self.progress)
        return self

    def seek_normalized(self, progress: float) -> KeyframeAnimation:
        """
        Seek to a specific progress.

        Args:
            progress: Progress to seek to (0 to 1)

        Returns:
            Self for chaining
        """
        self._elapsed = max(0.0, min(1.0, progress)) * self._duration
        self._apply_values(progress)
        return self

    def update(self, delta_time: float) -> bool:
        """
        Update the animation.

        Args:
            delta_time: Time since last update

        Returns:
            True if still playing, False if complete
        """
        if self._state != AnimationState.PLAYING:
            return self._state != AnimationState.COMPLETED

        # Update elapsed time
        self._elapsed += delta_time * self._speed * self._direction

        # Calculate normalized time
        normalized_time: float

        if self._direction > 0:
            if self._elapsed >= self._duration:
                normalized_time = 1.0
            else:
                normalized_time = self._elapsed / self._duration
        else:
            if self._elapsed <= 0:
                normalized_time = 0.0
            else:
                normalized_time = self._elapsed / self._duration

        # Apply values
        self._apply_values(normalized_time)

        # Fire update callback
        if self._on_update:
            self._on_update(normalized_time)

        # Check for completion/loop (with epsilon to handle floating point precision)
        if self._direction > 0 and self._elapsed >= self._duration - _EPSILON:
            return self._handle_end()
        elif self._direction < 0 and self._elapsed <= _EPSILON:
            return self._handle_end()

        return True

    def _handle_end(self) -> bool:
        """Handle reaching the end of the animation."""
        if self._loop_mode == LoopMode.ONCE:
            self._state = AnimationState.COMPLETED
            if self._on_complete:
                self._on_complete()
            return False

        elif self._loop_mode == LoopMode.LOOP:
            self._elapsed = 0.0
            self._loop_count += 1
            for track in self._tracks.values():
                track.reset()
            if self._on_loop:
                self._on_loop()
            return True

        elif self._loop_mode == LoopMode.PING_PONG:
            self._direction *= -1
            if self._direction > 0:
                self._elapsed = 0.0
                self._loop_count += 1
            else:
                self._elapsed = self._duration
            for track in self._tracks.values():
                track.reset()
            if self._on_loop:
                self._on_loop()
            return True

        return False

    def _apply_values(self, normalized_time: float) -> None:
        """Apply track values to the target."""
        if self._target is None:
            return

        for property_name, track in self._tracks.items():
            value = track.get_value_at(normalized_time)
            if value is not None and hasattr(self._target, property_name):
                setattr(self._target, property_name, value)


class KeyframeAnimationManager:
    """
    Manages multiple keyframe animations.

    Provides centralized update and lookup for animations.
    """

    def __init__(self) -> None:
        """Create an animation manager."""
        self._animations: Dict[str, KeyframeAnimation] = {}
        self._active: List[KeyframeAnimation] = []

    @property
    def count(self) -> int:
        """Number of registered animations."""
        return len(self._animations)

    @property
    def active_count(self) -> int:
        """Number of currently playing animations."""
        return len(self._active)

    def register(self, animation: KeyframeAnimation) -> None:
        """Register an animation by name."""
        self._animations[animation.name] = animation

    def unregister(self, name: str) -> bool:
        """Unregister an animation by name."""
        if name in self._animations:
            anim = self._animations.pop(name)
            if anim in self._active:
                self._active.remove(anim)
            return True
        return False

    def get(self, name: str) -> Optional[KeyframeAnimation]:
        """Get an animation by name."""
        return self._animations.get(name)

    def play(self, name: str, target: Optional[Any] = None) -> bool:
        """
        Play an animation by name.

        Args:
            name: Animation name
            target: Optional target to animate

        Returns:
            True if started, False if not found
        """
        animation = self._animations.get(name)
        if animation:
            if target:
                animation.set_target(target)
            animation.start()
            if animation not in self._active:
                self._active.append(animation)
            return True
        return False

    def stop(self, name: str) -> bool:
        """Stop an animation by name."""
        animation = self._animations.get(name)
        if animation:
            animation.stop()
            if animation in self._active:
                self._active.remove(animation)
            return True
        return False

    def stop_all(self, clear_callbacks: bool = False) -> None:
        """
        Stop all playing animations.

        Args:
            clear_callbacks: If True, also clear callback references to prevent memory leaks
        """
        for animation in self._active:
            animation.stop(clear_callbacks=clear_callbacks)
        self._active.clear()

    def update(self, delta_time: float) -> None:
        """Update all active animations."""
        completed = []
        for animation in self._active:
            if not animation.update(delta_time):
                completed.append(animation)

        # Remove completed animations and optionally clear their callbacks
        for animation in completed:
            self._active.remove(animation)
            # Clear callbacks on completed animations to release references
            animation.stop(clear_callbacks=True)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_keyframe_animation(
    name: str,
    duration: float = 1.0,
    loop_mode: LoopMode = LoopMode.ONCE,
    **tracks: List[Tuple[float, Any]],
) -> KeyframeAnimation:
    """
    Create a keyframe animation with tracks.

    Args:
        name: Animation name
        duration: Animation duration
        loop_mode: How to loop
        **tracks: Property names mapped to [(time, value), ...] lists

    Returns:
        Configured KeyframeAnimation

    Example:
        animation = create_keyframe_animation(
            "fade_scale",
            duration=0.5,
            opacity=[(0.0, 0.0), (1.0, 1.0)],
            scale=[(0.0, 0.8), (0.5, 1.1), (1.0, 1.0)],
        )
    """
    animation = KeyframeAnimation(name, duration, loop_mode)

    for property_name, keyframe_data in tracks.items():
        track = KeyframeTrack(property_name)
        for time, value in keyframe_data:
            track.add_keyframe(time, value)
        animation.add_track(track)

    return animation


def create_property_track(
    property_name: str,
    keyframes: List[Tuple[float, Any, str] | Tuple[float, Any]],
) -> KeyframeTrack:
    """
    Create a keyframe track from a list of tuples.

    Args:
        property_name: Property to animate
        keyframes: List of (time, value) or (time, value, easing) tuples

    Returns:
        Configured KeyframeTrack

    Example:
        track = create_property_track("opacity", [
            (0.0, 0.0),
            (0.5, 0.8, "ease_out"),
            (1.0, 1.0, "ease_in_out"),
        ])
    """
    track: KeyframeTrack = KeyframeTrack(property_name)

    for keyframe_data in keyframes:
        if len(keyframe_data) == 2:
            time, value = keyframe_data
            easing = "linear"
        else:
            time, value, easing = keyframe_data

        track.add_keyframe(time, value, easing)

    return track


__all__ = [
    # Core types
    "Keyframe",
    "LoopMode",
    "AnimationState",
    # Track
    "KeyframeTrack",
    # Animation
    "KeyframeAnimation",
    # Manager
    "KeyframeAnimationManager",
    # Factory functions
    "create_keyframe_animation",
    "create_property_track",
    # Callback types
    "AnimationCallback",
    "KeyframeCallback",
]
