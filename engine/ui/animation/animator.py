"""
Animator component for UI widget animations.

Provides a state machine-based animation system for widgets with:
- Named animation states (idle, hover, pressed, etc.)
- Transitions between states with configurable duration/easing
- Animation layers for blending multiple animations
- Automatic state machine management

Example usage:
    # Create animator for a button
    animator = Animator(button)

    # Add animation states
    animator.add_state("idle", idle_animation)
    animator.add_state("hover", hover_animation)
    animator.add_state("pressed", pressed_animation)

    # Add transitions
    animator.add_transition("idle", "hover", duration=0.2, easing="ease_out")
    animator.add_transition("hover", "pressed", duration=0.1)
    animator.add_transition("*", "idle", duration=0.3)  # Any state to idle

    # Control
    animator.transition_to("hover")
    animator.update(delta_time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from weakref import ref, ReferenceType

from engine.ui.animation.easing import EasingFunction, EasingType, get_easing, lerp
from engine.ui.animation.tween import TweenState

# Default animation timing constants
DEFAULT_TRANSITION_DURATION: float = 0.3  # Default duration for state transitions in seconds
DEFAULT_ANIMATION_SPEED: float = 1.0  # Default playback speed multiplier


class AnimatorState(Enum):
    """State of the animator component."""

    IDLE = auto()
    PLAYING = auto()
    TRANSITIONING = auto()
    PAUSED = auto()


class LayerBlendMode(Enum):
    """How animation layers blend together."""

    OVERRIDE = auto()    # Later layers override earlier
    ADDITIVE = auto()    # Values are added together
    MULTIPLY = auto()    # Values are multiplied
    AVERAGE = auto()     # Values are averaged


# Callback types
AnimatorCallback = Callable[[], None]
StateCallback = Callable[[str], None]
TransitionCallback = Callable[[str, str], None]


@dataclass
class AnimationState:
    """
    Represents a named animation state.

    An animation state contains the animations to play while in that state,
    along with configuration for looping, speed, and callbacks.
    """

    name: str
    animation: Any = None  # Can be Tween, KeyframeAnimation, or custom
    loop: bool = False
    speed: float = 1.0
    enter_callback: Optional[AnimatorCallback] = None
    exit_callback: Optional[AnimatorCallback] = None
    update_callback: Optional[Callable[[float], None]] = None

    # Runtime state
    _elapsed: float = field(default=0.0, init=False)
    _is_playing: bool = field(default=False, init=False)

    def reset(self) -> None:
        """Reset the state to initial conditions."""
        self._elapsed = 0.0
        self._is_playing = False
        if self.animation and hasattr(self.animation, "stop"):
            self.animation.stop()

    def start(self) -> None:
        """Start playing this animation state."""
        self._elapsed = 0.0
        self._is_playing = True
        if self.animation and hasattr(self.animation, "start"):
            self.animation.start()
        if self.enter_callback:
            self.enter_callback()

    def stop(self) -> None:
        """Stop this animation state."""
        self._is_playing = False
        if self.animation and hasattr(self.animation, "stop"):
            self.animation.stop()
        if self.exit_callback:
            self.exit_callback()

    def update(self, delta_time: float) -> bool:
        """
        Update the animation state.

        Args:
            delta_time: Time since last update

        Returns:
            True if still playing, False if complete
        """
        if not self._is_playing:
            return False

        self._elapsed += delta_time * self.speed

        # Update the animation
        if self.animation and hasattr(self.animation, "update"):
            still_active = self.animation.update(delta_time * self.speed)
            if not still_active and self.loop:
                if hasattr(self.animation, "start"):
                    self.animation.start()
            elif not still_active and not self.loop:
                self._is_playing = False

        if self.update_callback:
            self.update_callback(self._elapsed)

        return self._is_playing


@dataclass
class TransitionCondition:
    """Condition that must be met for a transition to occur."""

    name: str
    check: Callable[[], bool]


@dataclass
class AnimationTransition:
    """
    Defines a transition between two animation states.

    Transitions control how the animator moves from one state to another,
    including duration, easing, and conditions.
    """

    from_state: str  # Use "*" for any state
    to_state: str
    duration: float = DEFAULT_TRANSITION_DURATION
    easing: str | EasingType | EasingFunction = "linear"
    conditions: List[TransitionCondition] = field(default_factory=list)
    on_start: Optional[TransitionCallback] = None
    on_complete: Optional[TransitionCallback] = None

    # Resolved easing function
    _easing_func: Optional[EasingFunction] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Resolve the easing function."""
        if callable(self.easing) and not isinstance(self.easing, EasingType):
            self._easing_func = self.easing
        else:
            self._easing_func = get_easing(self.easing)

    def can_transition(self, current_state: str) -> bool:
        """
        Check if this transition can be taken from the current state.

        Args:
            current_state: The current animator state name

        Returns:
            True if all conditions are met
        """
        # Check source state match
        if self.from_state != "*" and self.from_state != current_state:
            return False

        # Check all conditions
        for condition in self.conditions:
            if not condition.check():
                return False

        return True

    def get_progress(self, elapsed: float) -> float:
        """
        Get the eased progress value for the given elapsed time.

        Args:
            elapsed: Time elapsed since transition started

        Returns:
            Progress value between 0 and 1
        """
        if self.duration <= 0:
            return 1.0

        raw_progress = min(1.0, elapsed / self.duration)
        if self._easing_func:
            return self._easing_func(raw_progress)
        return raw_progress


@dataclass
class AnimationLayer:
    """
    A layer for blending multiple animations.

    Layers allow complex animation setups where multiple animations
    can play simultaneously and blend together.
    """

    name: str
    weight: float = 1.0
    blend_mode: LayerBlendMode = LayerBlendMode.OVERRIDE
    mask: Optional[Set[str]] = None  # Property names to affect (None = all)
    enabled: bool = True

    # Current animation on this layer
    _current_animation: Optional[Any] = field(default=None, init=False)
    _property_values: Dict[str, Any] = field(default_factory=dict, init=False)

    def set_animation(self, animation: Any) -> None:
        """Set the animation for this layer."""
        self._current_animation = animation

    def clear(self) -> None:
        """Clear the animation and property values."""
        self._current_animation = None
        self._property_values.clear()

    def update(self, delta_time: float) -> None:
        """Update the layer's animation."""
        if not self.enabled or self._current_animation is None:
            return

        if hasattr(self._current_animation, "update"):
            self._current_animation.update(delta_time)

    def get_value(self, property_name: str) -> Optional[Any]:
        """Get the current value for a property from this layer."""
        if not self.enabled:
            return None
        if self.mask is not None and property_name not in self.mask:
            return None
        return self._property_values.get(property_name)

    def set_value(self, property_name: str, value: Any) -> None:
        """Set a property value on this layer."""
        self._property_values[property_name] = value


class Animator:
    """
    Component that manages animation states and transitions for a widget.

    The Animator provides a state machine for animations, allowing
    smooth transitions between different animation states.
    """

    def __init__(
        self,
        target: Any,
        initial_state: str = "idle",
    ) -> None:
        """
        Create an animator for a target widget.

        Args:
            target: The widget to animate
            initial_state: Name of the initial animation state
        """
        self._target: ReferenceType[Any] = ref(target)
        self._initial_state = initial_state
        self._current_state_name: str = initial_state
        self._state = AnimatorState.IDLE

        # State machine
        self._states: Dict[str, AnimationState] = {}
        self._transitions: List[AnimationTransition] = []

        # Layers
        self._layers: List[AnimationLayer] = []
        self._default_layer = AnimationLayer("default")
        self._layers.append(self._default_layer)

        # Transition state
        self._active_transition: Optional[AnimationTransition] = None
        self._transition_elapsed: float = 0.0
        self._transition_from_state: Optional[AnimationState] = None
        self._transition_to_state: Optional[AnimationState] = None

        # Callbacks
        self._on_state_enter: Optional[StateCallback] = None
        self._on_state_exit: Optional[StateCallback] = None
        self._on_transition_start: Optional[TransitionCallback] = None
        self._on_transition_complete: Optional[TransitionCallback] = None

        # Parameters for condition evaluation
        self._parameters: Dict[str, Any] = {}

    @property
    def target(self) -> Optional[Any]:
        """The target widget being animated."""
        return self._target()

    @property
    def state(self) -> AnimatorState:
        """Current animator state."""
        return self._state

    @property
    def current_state_name(self) -> str:
        """Name of the current animation state."""
        return self._current_state_name

    @property
    def current_state(self) -> Optional[AnimationState]:
        """The current animation state object."""
        return self._states.get(self._current_state_name)

    @property
    def is_transitioning(self) -> bool:
        """Whether the animator is currently transitioning."""
        return self._state == AnimatorState.TRANSITIONING

    @property
    def layers(self) -> List[AnimationLayer]:
        """The animation layers."""
        return self._layers.copy()

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def add_state(
        self,
        name: str,
        animation: Any = None,
        loop: bool = False,
        speed: float = 1.0,
    ) -> AnimationState:
        """
        Add an animation state.

        Args:
            name: Unique name for the state
            animation: Animation to play (Tween, KeyframeAnimation, etc.)
            loop: Whether the animation should loop
            speed: Playback speed multiplier

        Returns:
            The created AnimationState
        """
        state = AnimationState(
            name=name,
            animation=animation,
            loop=loop,
            speed=speed,
        )
        self._states[name] = state
        return state

    def remove_state(self, name: str) -> bool:
        """
        Remove an animation state.

        Args:
            name: Name of the state to remove

        Returns:
            True if removed, False if not found
        """
        if name in self._states:
            del self._states[name]
            return True
        return False

    def get_state(self, name: str) -> Optional[AnimationState]:
        """
        Get an animation state by name.

        Args:
            name: Name of the state

        Returns:
            The AnimationState or None if not found
        """
        return self._states.get(name)

    def has_state(self, name: str) -> bool:
        """
        Check if a state exists.

        Args:
            name: Name of the state

        Returns:
            True if the state exists
        """
        return name in self._states

    # =========================================================================
    # TRANSITION MANAGEMENT
    # =========================================================================

    def add_transition(
        self,
        from_state: str,
        to_state: str,
        duration: float = DEFAULT_TRANSITION_DURATION,
        easing: str | EasingType | EasingFunction = "linear",
    ) -> AnimationTransition:
        """
        Add a transition between states.

        Args:
            from_state: Source state name (use "*" for any)
            to_state: Destination state name
            duration: Transition duration in seconds
            easing: Easing function for the transition

        Returns:
            The created AnimationTransition
        """
        transition = AnimationTransition(
            from_state=from_state,
            to_state=to_state,
            duration=duration,
            easing=easing,
        )
        self._transitions.append(transition)
        return transition

    def remove_transition(self, from_state: str, to_state: str) -> bool:
        """
        Remove a transition.

        Args:
            from_state: Source state name
            to_state: Destination state name

        Returns:
            True if removed, False if not found
        """
        for i, transition in enumerate(self._transitions):
            if transition.from_state == from_state and transition.to_state == to_state:
                self._transitions.pop(i)
                return True
        return False

    def get_transition(
        self,
        from_state: str,
        to_state: str,
    ) -> Optional[AnimationTransition]:
        """
        Get a transition between states.

        Args:
            from_state: Source state name
            to_state: Destination state name

        Returns:
            The AnimationTransition or None
        """
        # First try exact match
        for transition in self._transitions:
            if transition.from_state == from_state and transition.to_state == to_state:
                return transition

        # Then try wildcard match
        for transition in self._transitions:
            if transition.from_state == "*" and transition.to_state == to_state:
                return transition

        return None

    # =========================================================================
    # LAYER MANAGEMENT
    # =========================================================================

    def add_layer(
        self,
        name: str,
        weight: float = 1.0,
        blend_mode: LayerBlendMode = LayerBlendMode.OVERRIDE,
    ) -> AnimationLayer:
        """
        Add an animation layer.

        Args:
            name: Unique name for the layer
            weight: Blend weight (0-1)
            blend_mode: How to blend with other layers

        Returns:
            The created AnimationLayer
        """
        layer = AnimationLayer(
            name=name,
            weight=weight,
            blend_mode=blend_mode,
        )
        self._layers.append(layer)
        return layer

    def remove_layer(self, name: str) -> bool:
        """
        Remove an animation layer.

        Args:
            name: Name of the layer to remove

        Returns:
            True if removed, False if not found
        """
        for i, layer in enumerate(self._layers):
            if layer.name == name and layer != self._default_layer:
                self._layers.pop(i)
                return True
        return False

    def get_layer(self, name: str) -> Optional[AnimationLayer]:
        """
        Get a layer by name.

        Args:
            name: Name of the layer

        Returns:
            The AnimationLayer or None
        """
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None

    # =========================================================================
    # PARAMETER MANAGEMENT
    # =========================================================================

    def set_parameter(self, name: str, value: Any) -> Animator:
        """
        Set an animator parameter.

        Parameters can be used in transition conditions.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            Self for chaining
        """
        self._parameters[name] = value
        return self

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get an animator parameter.

        Args:
            name: Parameter name
            default: Default value if not found

        Returns:
            The parameter value
        """
        return self._parameters.get(name, default)

    def clear_parameters(self) -> Animator:
        """
        Clear all parameters.

        Returns:
            Self for chaining
        """
        self._parameters.clear()
        return self

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_state_enter(self, callback: StateCallback) -> Animator:
        """Set callback for entering any state."""
        self._on_state_enter = callback
        return self

    def on_state_exit(self, callback: StateCallback) -> Animator:
        """Set callback for exiting any state."""
        self._on_state_exit = callback
        return self

    def on_transition_start(self, callback: TransitionCallback) -> Animator:
        """Set callback for starting any transition."""
        self._on_transition_start = callback
        return self

    def on_transition_complete(self, callback: TransitionCallback) -> Animator:
        """Set callback for completing any transition."""
        self._on_transition_complete = callback
        return self

    # =========================================================================
    # CONTROL METHODS
    # =========================================================================

    def play(self, state_name: Optional[str] = None) -> Animator:
        """
        Start playing the current or specified state.

        Args:
            state_name: Optional state to switch to and play

        Returns:
            Self for chaining
        """
        if state_name and state_name != self._current_state_name:
            self._switch_state(state_name)

        current = self.current_state
        if current:
            current.start()
            self._state = AnimatorState.PLAYING

        return self

    def pause(self) -> Animator:
        """
        Pause the animator.

        Returns:
            Self for chaining
        """
        if self._state in (AnimatorState.PLAYING, AnimatorState.TRANSITIONING):
            self._state = AnimatorState.PAUSED
        return self

    def resume(self) -> Animator:
        """
        Resume the animator.

        Returns:
            Self for chaining
        """
        if self._state == AnimatorState.PAUSED:
            if self._active_transition:
                self._state = AnimatorState.TRANSITIONING
            else:
                self._state = AnimatorState.PLAYING
        return self

    def stop(self) -> Animator:
        """
        Stop the animator and reset to initial state.

        Returns:
            Self for chaining
        """
        self._state = AnimatorState.IDLE
        self._active_transition = None
        self._transition_elapsed = 0.0

        for state in self._states.values():
            state.reset()

        self._current_state_name = self._initial_state
        return self

    def transition_to(
        self,
        state_name: str,
        immediate: bool = False,
    ) -> bool:
        """
        Transition to a new state.

        Args:
            state_name: Name of the target state
            immediate: If True, skip transition animation

        Returns:
            True if transition started, False if not possible
        """
        if state_name not in self._states:
            return False

        if state_name == self._current_state_name:
            return False

        if immediate:
            self._switch_state(state_name)
            return True

        # Find a valid transition
        transition = self.get_transition(self._current_state_name, state_name)

        if transition and not transition.can_transition(self._current_state_name):
            transition = None

        if transition:
            self._start_transition(transition, state_name)
            return True
        else:
            # No transition defined, switch immediately
            self._switch_state(state_name)
            return True

    def _switch_state(self, state_name: str) -> None:
        """Switch to a new state immediately."""
        # Exit current state
        current = self.current_state
        if current:
            current.stop()
            if self._on_state_exit:
                self._on_state_exit(self._current_state_name)

        # Enter new state
        self._current_state_name = state_name
        new_state = self.current_state
        if new_state:
            new_state.start()
            if self._on_state_enter:
                self._on_state_enter(state_name)

        self._state = AnimatorState.PLAYING

    def _start_transition(
        self,
        transition: AnimationTransition,
        to_state_name: str,
    ) -> None:
        """Start a transition to a new state."""
        self._active_transition = transition
        self._transition_elapsed = 0.0
        self._transition_from_state = self.current_state
        self._transition_to_state = self._states.get(to_state_name)

        self._state = AnimatorState.TRANSITIONING

        # Fire callbacks
        if transition.on_start:
            transition.on_start(self._current_state_name, to_state_name)
        if self._on_transition_start:
            self._on_transition_start(self._current_state_name, to_state_name)

        # Start the target state animation (it will be blended)
        if self._transition_to_state:
            self._transition_to_state.start()

    def _complete_transition(self) -> None:
        """Complete the current transition."""
        if not self._active_transition or not self._transition_to_state:
            return

        from_name = self._current_state_name
        to_name = self._transition_to_state.name

        # Stop the old state
        if self._transition_from_state:
            self._transition_from_state.stop()
            if self._on_state_exit:
                self._on_state_exit(from_name)

        # Switch to the new state
        self._current_state_name = to_name
        if self._on_state_enter:
            self._on_state_enter(to_name)

        # Fire callbacks
        if self._active_transition.on_complete:
            self._active_transition.on_complete(from_name, to_name)
        if self._on_transition_complete:
            self._on_transition_complete(from_name, to_name)

        # Clean up transition state
        self._active_transition = None
        self._transition_elapsed = 0.0
        self._transition_from_state = None
        self._transition_to_state = None

        self._state = AnimatorState.PLAYING

    def update(self, delta_time: float) -> None:
        """
        Update the animator.

        Call this every frame with the time since last frame.

        Args:
            delta_time: Time since last update in seconds
        """
        if self._state == AnimatorState.IDLE or self._state == AnimatorState.PAUSED:
            return

        # Handle transition
        if self._state == AnimatorState.TRANSITIONING and self._active_transition:
            self._transition_elapsed += delta_time

            progress = self._active_transition.get_progress(self._transition_elapsed)

            # Update both states during transition
            if self._transition_from_state:
                self._transition_from_state.update(delta_time * (1 - progress))
            if self._transition_to_state:
                self._transition_to_state.update(delta_time * progress)

            # Check for completion
            if self._transition_elapsed >= self._active_transition.duration:
                self._complete_transition()
        else:
            # Normal playback
            current = self.current_state
            if current:
                current.update(delta_time)

        # Update all layers
        for layer in self._layers:
            layer.update(delta_time)


class AnimatorManager:
    """
    Manages multiple animators and provides global update.

    Use this to centralize animator updates in your game loop.
    """

    def __init__(self) -> None:
        """Create an animator manager."""
        self._animators: List[Animator] = []

    @property
    def count(self) -> int:
        """Number of managed animators."""
        return len(self._animators)

    def add(self, animator: Animator) -> None:
        """Add an animator to be managed."""
        if animator not in self._animators:
            self._animators.append(animator)

    def remove(self, animator: Animator) -> None:
        """Remove an animator from management."""
        if animator in self._animators:
            self._animators.remove(animator)

    def clear(self, release_resources: bool = True) -> None:
        """
        Remove all animators.

        Args:
            release_resources: If True, also clear callbacks to prevent memory leaks
        """
        for animator in self._animators:
            animator.stop()
            if release_resources:
                # Clear callbacks to release references
                animator._on_state_enter = None
                animator._on_state_exit = None
                animator._on_transition_start = None
                animator._on_transition_complete = None
        self._animators.clear()

    def update(self, delta_time: float) -> None:
        """
        Update all managed animators.

        Args:
            delta_time: Time since last update
        """
        for animator in self._animators:
            animator.update(delta_time)

    def get_by_target(self, target: Any) -> Optional[Animator]:
        """
        Find an animator by its target widget.

        Args:
            target: The target widget

        Returns:
            The Animator or None if not found
        """
        for animator in self._animators:
            if animator.target is target:
                return animator
        return None


__all__ = [
    # Core
    "Animator",
    "AnimatorState",
    # Animation state
    "AnimationState",
    # Transition
    "AnimationTransition",
    "TransitionCondition",
    # Layer
    "AnimationLayer",
    "LayerBlendMode",
    # Manager
    "AnimatorManager",
    # Callback types
    "AnimatorCallback",
    "StateCallback",
    "TransitionCallback",
    # Constants
    "DEFAULT_TRANSITION_DURATION",
    "DEFAULT_ANIMATION_SPEED",
]
