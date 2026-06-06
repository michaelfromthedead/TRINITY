"""
Animation state machine implementation.

Provides finite state machine (FSM) functionality for animation:
- AnimationState: Individual states with associated animation nodes
- StateTransition: Transitions between states with conditions and blending
- TransitionCondition: Parameter-based conditions for transitions
- StateMachine: The complete FSM managing states and transitions

State machines allow for organized, predictable animation flow with
explicit control over transitions and blending.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from .animation_graph import (
    AnimationNode,
    AnimationGraph,
    GraphContext,
    GraphParameter,
    ParameterType,
    Pose,
    Transform,
)
from .config import get_config


# =============================================================================
# MOTION MODE
# =============================================================================


class MotionMode(Enum):
    """Motion modes for animation state playback."""

    LOOP = auto()  # Repeat animation continuously
    ONCE = auto()  # Play animation once, then stop
    PING_PONG = auto()  # Play forward, then backward, repeat


# Forward reference for AnimationClip (defined in blend_node.py to avoid circular imports)
# Will be resolved at runtime via TYPE_CHECKING or string annotation
AnimationClip = Any  # Type alias for forward reference


# =============================================================================
# BLEND CURVES
# =============================================================================


class BlendCurve(Enum):
    """Easing curves for transition blending."""

    LINEAR = auto()
    EASE_IN = auto()
    EASE_OUT = auto()
    EASE_IN_OUT = auto()
    SMOOTH_STEP = auto()
    SMOOTHER_STEP = auto()


def evaluate_blend_curve(curve: BlendCurve, t: float) -> float:
    """Evaluate a blend curve at time t (0-1)."""
    t = max(0.0, min(1.0, t))

    if curve == BlendCurve.LINEAR:
        return t
    elif curve == BlendCurve.EASE_IN:
        return t * t
    elif curve == BlendCurve.EASE_OUT:
        return t * (2.0 - t)
    elif curve == BlendCurve.EASE_IN_OUT:
        return t * t * (3.0 - 2.0 * t)
    elif curve == BlendCurve.SMOOTH_STEP:
        return t * t * (3.0 - 2.0 * t)
    elif curve == BlendCurve.SMOOTHER_STEP:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    return t


# =============================================================================
# TRANSITION CONDITIONS
# =============================================================================


class ConditionOperator(Enum):
    """
    Operators for transition conditions.

    Supports comparison operators and logical operators for compound conditions.
    String values match common animation system conventions.
    """

    # Comparison operators
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "ge"
    LESS_EQUAL = "le"

    # Logical operators for compound conditions
    AND = "and"
    OR = "or"


# Legacy alias for backwards compatibility
ComparisonOp = ConditionOperator


class ParameterTypeError(TypeError):
    """Raised when a parameter type doesn't match the expected type for an operation."""

    def __init__(self, parameter: str, expected_type: str, actual_type: str) -> None:
        self.parameter = parameter
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(
            f"Parameter '{parameter}' type mismatch: expected {expected_type}, got {actual_type}"
        )


@dataclass
class TransitionCondition:
    """
    A condition that must be met for a transition to occur.

    Supports:
    - Parameter-based comparisons (equals, greater than, etc.)
    - Trigger parameters (one-shot, auto-reset after evaluation)
    - Exit time conditions (time-based transitions)
    - Compound conditions (AND/OR with sub-conditions)
    - Parameter type checking for safe comparisons

    Attributes:
        parameter: Name of the parameter to check (or None for compound/exit_time only)
        operator: The comparison or logical operator to use
        value: The value to compare against
        is_trigger: If True, parameter is auto-reset after successful evaluation
        exit_time: Optional normalized time (0-1) when condition becomes valid
        sub_conditions: List of sub-conditions for AND/OR operators
        expected_type: Optional type for parameter validation
    """

    parameter: str = ""
    operator: ConditionOperator = ConditionOperator.EQUALS
    value: Any = None
    is_trigger: bool = False
    exit_time: Optional[float] = None
    sub_conditions: List["TransitionCondition"] = field(default_factory=list)
    expected_type: Optional[Type] = None

    # Legacy field aliases
    @property
    def parameter_name(self) -> str:
        """Legacy alias for parameter field."""
        return self.parameter

    @parameter_name.setter
    def parameter_name(self, value: str) -> None:
        """Legacy alias for parameter field."""
        self.parameter = value

    @property
    def comparison(self) -> ConditionOperator:
        """Legacy alias for operator field."""
        return self.operator

    @comparison.setter
    def comparison(self, value: ConditionOperator) -> None:
        """Legacy alias for operator field."""
        self.operator = value

    def _check_type(self, param_value: Any) -> bool:
        """
        Validate parameter type if expected_type is set.

        Returns True if type is valid, raises ParameterTypeError if not.
        """
        if self.expected_type is None:
            return True

        if not isinstance(param_value, self.expected_type):
            raise ParameterTypeError(
                self.parameter,
                self.expected_type.__name__,
                type(param_value).__name__
            )
        return True

    def _evaluate_comparison(self, param_value: Any) -> bool:
        """Evaluate a comparison operation between param_value and self.value."""
        # Type checking
        try:
            self._check_type(param_value)
        except ParameterTypeError:
            return False

        op = self.operator

        if op == ConditionOperator.EQUALS:
            return param_value == self.value
        elif op == ConditionOperator.NOT_EQUALS:
            return param_value != self.value
        elif op == ConditionOperator.GREATER_THAN:
            try:
                return param_value > self.value
            except TypeError:
                return False
        elif op == ConditionOperator.LESS_THAN:
            try:
                return param_value < self.value
            except TypeError:
                return False
        elif op == ConditionOperator.GREATER_EQUAL:
            try:
                return param_value >= self.value
            except TypeError:
                return False
        elif op == ConditionOperator.LESS_EQUAL:
            try:
                return param_value <= self.value
            except TypeError:
                return False

        return False

    def _evaluate_compound(self, context: GraphContext) -> bool:
        """Evaluate compound AND/OR conditions."""
        if not self.sub_conditions:
            return True

        if self.operator == ConditionOperator.AND:
            return all(cond.evaluate(context) for cond in self.sub_conditions)
        elif self.operator == ConditionOperator.OR:
            return any(cond.evaluate(context) for cond in self.sub_conditions)

        return False

    def evaluate(self, context: GraphContext, state_normalized_time: float = 0.0) -> bool:
        """
        Evaluate this condition against the context.

        Args:
            context: The graph context containing parameters
            state_normalized_time: Current normalized time in the state (0-1),
                                   used for exit_time conditions

        Returns:
            True if the condition is met, False otherwise
        """
        # Check exit_time condition first
        if self.exit_time is not None:
            if state_normalized_time < self.exit_time:
                return False
            # If only exit_time is set (no parameter), return True
            if not self.parameter and self.operator not in (ConditionOperator.AND, ConditionOperator.OR):
                return True

        # Handle compound conditions (AND/OR)
        if self.operator in (ConditionOperator.AND, ConditionOperator.OR):
            return self._evaluate_compound(context)

        # Get parameter value
        if not self.parameter:
            return True  # No parameter to check

        param_value = context.get_parameter(self.parameter)

        if param_value is None:
            return False

        # Evaluate the comparison
        result = self._evaluate_comparison(param_value)

        # Handle trigger parameters (one-shot, auto-reset)
        if result and self.is_trigger:
            # Reset the trigger parameter after successful evaluation
            context.set_parameter(self.parameter, False)

        return result

    # ==========================================================================
    # Factory methods for creating conditions
    # ==========================================================================

    @classmethod
    def equals(cls, param: str, value: Any, *,
               is_trigger: bool = False,
               expected_type: Optional[Type] = None) -> "TransitionCondition":
        """Create an equals condition."""
        return cls(param, ConditionOperator.EQUALS, value,
                   is_trigger=is_trigger, expected_type=expected_type)

    @classmethod
    def not_equals(cls, param: str, value: Any, *,
                   is_trigger: bool = False,
                   expected_type: Optional[Type] = None) -> "TransitionCondition":
        """Create a not equals condition."""
        return cls(param, ConditionOperator.NOT_EQUALS, value,
                   is_trigger=is_trigger, expected_type=expected_type)

    @classmethod
    def greater_than(cls, param: str, value: float, *,
                     is_trigger: bool = False) -> "TransitionCondition":
        """Create a greater than condition."""
        return cls(param, ConditionOperator.GREATER_THAN, value,
                   is_trigger=is_trigger, expected_type=(int, float))

    @classmethod
    def greater_or_equal(cls, param: str, value: float, *,
                         is_trigger: bool = False) -> "TransitionCondition":
        """Create a greater or equal condition."""
        return cls(param, ConditionOperator.GREATER_EQUAL, value,
                   is_trigger=is_trigger, expected_type=(int, float))

    @classmethod
    def less_than(cls, param: str, value: float, *,
                  is_trigger: bool = False) -> "TransitionCondition":
        """Create a less than condition."""
        return cls(param, ConditionOperator.LESS_THAN, value,
                   is_trigger=is_trigger, expected_type=(int, float))

    @classmethod
    def less_or_equal(cls, param: str, value: float, *,
                      is_trigger: bool = False) -> "TransitionCondition":
        """Create a less or equal condition."""
        return cls(param, ConditionOperator.LESS_EQUAL, value,
                   is_trigger=is_trigger, expected_type=(int, float))

    @classmethod
    def is_true(cls, param: str, *, is_trigger: bool = False) -> "TransitionCondition":
        """Create a boolean true condition."""
        return cls(param, ConditionOperator.EQUALS, True,
                   is_trigger=is_trigger, expected_type=bool)

    @classmethod
    def is_false(cls, param: str, *, is_trigger: bool = False) -> "TransitionCondition":
        """Create a boolean false condition."""
        return cls(param, ConditionOperator.EQUALS, False,
                   is_trigger=is_trigger, expected_type=bool)

    @classmethod
    def trigger(cls, param: str) -> "TransitionCondition":
        """
        Create a trigger condition (one-shot parameter).

        Trigger parameters are automatically reset to False after the condition
        evaluates to True, ensuring the transition only fires once per trigger.
        """
        return cls(param, ConditionOperator.EQUALS, True,
                   is_trigger=True, expected_type=bool)

    @classmethod
    def at_exit_time(cls, exit_time: float) -> "TransitionCondition":
        """
        Create an exit time condition.

        Args:
            exit_time: Normalized time (0-1) when the condition becomes valid

        Returns:
            A condition that is True when state normalized time >= exit_time
        """
        return cls(exit_time=exit_time)

    @classmethod
    def and_conditions(cls, *conditions: "TransitionCondition") -> "TransitionCondition":
        """
        Create a compound AND condition.

        All sub-conditions must be True for this condition to be True.
        """
        return cls(operator=ConditionOperator.AND, sub_conditions=list(conditions))

    @classmethod
    def or_conditions(cls, *conditions: "TransitionCondition") -> "TransitionCondition":
        """
        Create a compound OR condition.

        At least one sub-condition must be True for this condition to be True.
        """
        return cls(operator=ConditionOperator.OR, sub_conditions=list(conditions))

    # ==========================================================================
    # Legacy compatibility aliases
    # ==========================================================================

    # These maintain backwards compatibility with code using IS_TRUE/IS_FALSE
    @classmethod
    def _legacy_is_true(cls, param: str) -> "TransitionCondition":
        """Legacy alias - use is_true() instead."""
        return cls.is_true(param)

    @classmethod
    def _legacy_is_false(cls, param: str) -> "TransitionCondition":
        """Legacy alias - use is_false() instead."""
        return cls.is_false(param)


# =============================================================================
# ANIMATION STATE
# =============================================================================


@dataclass
class AnimationState:
    """
    A state in the animation state machine.

    Each state has an associated animation source (clip, graph, or node) that
    produces the pose when the state is active. States can have enter/exit
    events for triggering side effects.

    Supports three motion modes:
    - LOOP: Animation repeats continuously
    - ONCE: Animation plays once and stops at the end
    - PING_PONG: Animation alternates between forward and backward playback

    Attributes:
        name: Unique identifier for this state
        clip: Direct reference to an animation clip (optional)
        graph: Reference to an animation subgraph (optional)
        animation_node: Generic animation node (optional, fallback)
        motion_mode: How the animation should loop/repeat
        speed: Playback speed multiplier (1.0 = normal, 2.0 = double speed)
        current_time: Current playback position in seconds
        on_enter: Callback invoked when entering this state
        on_exit: Callback invoked when exiting this state
    """

    name: str

    # Animation sources (use one of these)
    clip: Optional["AnimationClip"] = None  # Direct animation clip reference
    graph: Optional["AnimationGraph"] = None  # Animation subgraph reference
    animation_node: Optional[AnimationNode] = None  # Generic node (fallback)

    # Motion handling
    motion_mode: MotionMode = MotionMode.LOOP
    speed: float = 1.0  # Speed multiplier (1.0 = normal, 2.0 = double speed)

    # Optional callbacks
    on_enter: Optional[Callable[["AnimationState", GraphContext], None]] = None
    on_exit: Optional[Callable[["AnimationState", GraphContext], None]] = None
    on_update: Optional[Callable[["AnimationState", GraphContext, float], None]] = None

    # State metadata
    can_interrupt: bool = True  # Whether this state can be interrupted mid-animation

    # Runtime state - current playback position
    current_time: float = field(default=0.0)
    _normalized_time: float = field(default=0.0, repr=False)
    _playback_direction: int = field(default=1, repr=False)  # 1 = forward, -1 = backward
    _animation_finished: bool = field(default=False, repr=False)

    # Legacy compatibility
    @property
    def speed_multiplier(self) -> float:
        """Legacy alias for speed."""
        return self.speed

    @speed_multiplier.setter
    def speed_multiplier(self, value: float) -> None:
        """Legacy alias for speed."""
        self.speed = value

    @property
    def loop(self) -> bool:
        """Legacy compatibility: True if motion_mode is LOOP."""
        return self.motion_mode == MotionMode.LOOP

    @loop.setter
    def loop(self, value: bool) -> None:
        """Legacy compatibility: Set motion_mode based on bool."""
        self.motion_mode = MotionMode.LOOP if value else MotionMode.ONCE

    def enter(self, context: GraphContext) -> None:
        """Called when entering this state."""
        self.current_time = 0.0
        self._normalized_time = 0.0
        self._playback_direction = 1
        self._animation_finished = False
        if self.on_enter:
            self.on_enter(self, context)

    def exit(self, context: GraphContext) -> None:
        """Called when exiting this state."""
        if self.on_exit:
            self.on_exit(self, context)

    def _get_duration(self) -> float:
        """Get the duration of the animation source."""
        if self.clip is not None and hasattr(self.clip, "duration"):
            return self.clip.duration
        # For graphs or nodes, assume normalized time (0-1 range)
        return 1.0

    def update(self, dt: float, context: Optional[GraphContext] = None) -> None:
        """
        Update the state's playback time.

        Args:
            dt: Delta time in seconds
            context: Optional graph context for callback
        """
        if self._animation_finished:
            return

        duration = self._get_duration()
        time_delta = dt * self.speed * self._playback_direction

        self.current_time += time_delta

        # Handle motion modes
        if self.motion_mode == MotionMode.LOOP:
            if duration > 0:
                self.current_time = self.current_time % duration

        elif self.motion_mode == MotionMode.ONCE:
            if self.current_time >= duration:
                self.current_time = duration
                self._animation_finished = True
            elif self.current_time < 0:
                self.current_time = 0
                self._animation_finished = True

        elif self.motion_mode == MotionMode.PING_PONG:
            if duration > 0:
                if self.current_time >= duration:
                    self.current_time = duration - (self.current_time - duration)
                    self._playback_direction = -1
                elif self.current_time < 0:
                    self.current_time = -self.current_time
                    self._playback_direction = 1

        # Update normalized time
        if duration > 0:
            self._normalized_time = self.current_time / duration
        else:
            self._normalized_time = 0.0

        # Call update callback (support both old and new signatures)
        if self.on_update:
            if context is not None:
                self.on_update(self, context, dt)

    def evaluate(self, context: GraphContext) -> Pose:
        """
        Evaluate this state's animation source and return the current pose.

        Checks for animation sources in order: clip, graph, animation_node.
        """
        # Create context with current state's normalized time
        state_context = GraphContext(
            parameters=context.parameters,
            dt=context.dt,
            skeleton=context.skeleton,
            bone_masks=context.bone_masks,
            normalized_time=self._normalized_time,
            sync_group=context.sync_group,
            layer_weight=context.layer_weight,
        )

        # Evaluate clip directly if available
        if self.clip is not None and hasattr(self.clip, "sample"):
            bone_count = len(context.skeleton.bones) if context.skeleton else 0
            return self.clip.sample(self.current_time, bone_count)

        # Evaluate subgraph if available
        if self.graph is not None:
            return self.graph.evaluate(state_context)

        # Fall back to animation node
        if self.animation_node:
            return self.animation_node.evaluate(state_context)

        return Pose()

    @property
    def time_in_state(self) -> float:
        """Get the time spent in this state (alias for current_time)."""
        return self.current_time

    @property
    def normalized_time(self) -> float:
        """Get the normalized time (0-1) in this state."""
        return self._normalized_time

    @normalized_time.setter
    def normalized_time(self, value: float) -> None:
        """Set the normalized time and update current_time accordingly."""
        self._normalized_time = value
        duration = self._get_duration()
        self.current_time = value * duration

    @property
    def is_finished(self) -> bool:
        """Check if a ONCE animation has finished playing."""
        return self._animation_finished

    def reset(self) -> None:
        """Reset the state's playback to the beginning."""
        self.current_time = 0.0
        self._normalized_time = 0.0
        self._playback_direction = 1
        self._animation_finished = False


# =============================================================================
# STATE TRANSITION
# =============================================================================


class InterruptMode(Enum):
    """
    Interrupt modes for state transitions.

    Controls whether and how a transition can be interrupted by other transitions.
    """

    NONE = "none"  # Cannot be interrupted once started
    ANY = "any"  # Can be interrupted by any valid transition
    HIGHER_PRIORITY = "higher_priority"  # Only interrupted by higher priority transitions


class TransitionSyncMode(Enum):
    """How to sync animation time during transitions."""

    NONE = auto()  # Don't sync
    NORMALIZED = auto()  # Match normalized time
    PROPORTIONAL = auto()  # Proportional to remaining time
    MARKER = auto()  # Sync to markers


@dataclass
class StateTransition:
    """
    A transition between two states in the state machine.

    Transitions define when and how to move from one state to another,
    including conditions, blend duration, and blend curve.

    Attributes:
        source: Source state name (use "*" for any-state transitions)
        target: Target state name
        conditions: List of conditions that must all pass for transition to occur
        duration: Blend duration in seconds (or percentage if duration_mode="percentage")
        duration_mode: How to interpret duration - "fixed" (seconds) or "percentage" (of source anim)
        blend_curve: Easing curve for the blend weight
        sync_mode: How to synchronize animation time during transition
        exit_time: Optional normalized time (0-1) when transition becomes valid
        offset: Start offset in target state (normalized time)
        priority: Higher values = higher priority (checked first)
        interrupt_mode: Controls whether/how this transition can be interrupted
        can_interrupt_self: Legacy - whether this can interrupt itself (deprecated, use interrupt_mode)
        can_be_interrupted: Legacy - whether this can be interrupted (deprecated, use interrupt_mode)
    """

    source: str  # Source state name, or "*" for any-state
    target: str  # Target state name
    conditions: List[TransitionCondition] = field(default_factory=list)

    # Blend settings
    duration: float = 0.25  # Blend duration in seconds (or percentage)
    duration_mode: str = "fixed"  # "fixed" (seconds) or "percentage" (of source animation)
    blend_curve: BlendCurve = BlendCurve.SMOOTH_STEP
    sync_mode: TransitionSyncMode = TransitionSyncMode.NONE

    # Timing settings
    exit_time: Optional[float] = None  # Normalized time to exit (0-1)
    offset: float = 0.0  # Start offset in target state

    # Priority and interruption
    priority: int = 0  # Higher priority transitions are checked first
    interrupt_mode: InterruptMode = InterruptMode.HIGHER_PRIORITY

    # Legacy fields for backward compatibility (deprecated)
    can_interrupt_self: bool = False  # Deprecated: use interrupt_mode
    can_be_interrupted: bool = True  # Deprecated: use interrupt_mode

    def __post_init__(self) -> None:
        """Post-initialization to sync legacy fields with interrupt_mode."""
        # If legacy fields were explicitly set to non-default values, update interrupt_mode
        # This maintains backward compatibility
        if not self.can_be_interrupted and self.interrupt_mode == InterruptMode.HIGHER_PRIORITY:
            self.interrupt_mode = InterruptMode.NONE

    @property
    def fixed_duration(self) -> bool:
        """Legacy property: True if duration_mode is 'fixed'."""
        return self.duration_mode == "fixed"

    @fixed_duration.setter
    def fixed_duration(self, value: bool) -> None:
        """Legacy property setter for fixed_duration."""
        self.duration_mode = "fixed" if value else "percentage"

    def get_effective_duration(self, source_animation_duration: float = 1.0) -> float:
        """
        Get the effective duration in seconds.

        Args:
            source_animation_duration: Duration of source animation (for percentage mode)

        Returns:
            Duration in seconds
        """
        if self.duration_mode == "percentage":
            return self.duration * source_animation_duration
        return self.duration

    def allows_interruption_by(self, other_priority: int) -> bool:
        """
        Check if this transition can be interrupted by a transition with the given priority.

        Args:
            other_priority: Priority of the interrupting transition

        Returns:
            True if interruption is allowed, False otherwise
        """
        if self.interrupt_mode == InterruptMode.NONE:
            return False
        if self.interrupt_mode == InterruptMode.ANY:
            return True
        if self.interrupt_mode == InterruptMode.HIGHER_PRIORITY:
            return other_priority > self.priority
        return False

    def can_transition(self, current_state: AnimationState,
                       context: GraphContext) -> bool:
        """
        Check if this transition can occur.

        All conditions must pass for the transition to be valid.

        Args:
            current_state: The current animation state
            context: Graph context containing parameters

        Returns:
            True if all conditions are met, False otherwise
        """
        # Check exit time at transition level
        if self.exit_time is not None:
            if current_state.normalized_time < self.exit_time:
                return False

        # Check all conditions, passing state normalized time for exit_time conditions
        state_time = current_state.normalized_time
        return all(cond.evaluate(context, state_time) for cond in self.conditions)

    def is_any_state(self) -> bool:
        """Check if this is an any-state transition (source == '*')."""
        return self.source == "*"


# =============================================================================
# ACTIVE TRANSITION
# =============================================================================


@dataclass
class ActiveTransition:
    """Represents an in-progress transition between states."""

    transition: StateTransition
    source_state: AnimationState
    target_state: AnimationState
    progress: float = 0.0
    source_pose: Optional[Pose] = None
    duration: float = 0.0

    def __post_init__(self) -> None:
        # Compute effective duration based on duration_mode
        # For percentage mode, use source animation duration; for fixed, use raw value
        source_animation_duration = self.source_state._get_duration()
        self.duration = self.transition.get_effective_duration(source_animation_duration)

    @property
    def blend_weight(self) -> float:
        """Get the current blend weight (0 = source, 1 = target)."""
        if self.duration <= 0:
            return 1.0
        t = self.progress / self.duration
        return evaluate_blend_curve(self.transition.blend_curve, t)

    @property
    def is_complete(self) -> bool:
        """Check if the transition is complete."""
        return self.progress >= self.duration

    def update(self, dt: float) -> None:
        """Update the transition progress."""
        self.progress += dt


# =============================================================================
# STATE MACHINE
# =============================================================================


class StateMachine(AnimationNode):
    """
    A finite state machine for controlling animation flow.

    The state machine manages a set of states with transitions between them.
    It evaluates to a pose by blending between the current state and any
    active transition target.
    """

    _abstract = False

    def __init__(self, node_id: str, initial_state: Optional[str] = None) -> None:
        super().__init__(node_id)

        self.states: Dict[str, AnimationState] = {}
        self.transitions: List[StateTransition] = []
        self._initial_state: Optional[str] = initial_state

        # Runtime state
        self._current_state: Optional[AnimationState] = None
        self._active_transition: Optional[ActiveTransition] = None
        self._pending_transition: Optional[ActiveTransition] = None
        self._transition_this_frame: bool = False  # Prevent infinite transition loops

        # Any-state transitions (can occur from any state)
        self._any_state_transitions: List[StateTransition] = []

    @property
    def current_state(self) -> Optional[AnimationState]:
        """Get the current state."""
        return self._current_state

    @property
    def current_state_name(self) -> Optional[str]:
        """Get the current state name."""
        return self._current_state.name if self._current_state else None

    @property
    def active_transition(self) -> Optional[ActiveTransition]:
        """Get the active transition, if any."""
        return self._active_transition

    @property
    def is_transitioning(self) -> bool:
        """Check if a transition is in progress."""
        return self._active_transition is not None

    def add_state(self, state: AnimationState) -> None:
        """Add a state to the state machine."""
        self.states[state.name] = state
        if self._initial_state is None:
            self._initial_state = state.name

    def remove_state(self, name: str) -> bool:
        """Remove a state from the state machine."""
        if name in self.states:
            # Remove transitions involving this state
            self.transitions = [
                t for t in self.transitions
                if t.source != name and t.target != name
            ]
            del self.states[name]
            return True
        return False

    def get_state(self, name: str) -> Optional[AnimationState]:
        """Get a state by name."""
        return self.states.get(name)

    def add_transition(self, transition: StateTransition) -> None:
        """Add a transition to the state machine."""
        if transition.is_any_state():
            self._any_state_transitions.append(transition)
        else:
            self.transitions.append(transition)

        # Sort by priority (highest first)
        self.transitions.sort(key=lambda t: t.priority, reverse=True)
        self._any_state_transitions.sort(key=lambda t: t.priority, reverse=True)

    def remove_transition(self, source: str, target: str) -> bool:
        """Remove a transition between two states."""
        for i, t in enumerate(self.transitions):
            if t.source == source and t.target == target:
                self.transitions.pop(i)
                return True
        for i, t in enumerate(self._any_state_transitions):
            if t.source == source and t.target == target:
                self._any_state_transitions.pop(i)
                return True
        return False

    def get_transitions_from(self, state_name: str) -> List[StateTransition]:
        """Get all transitions from a given state."""
        return [t for t in self.transitions if t.source == state_name]

    def get_transitions_to(self, state_name: str) -> List[StateTransition]:
        """Get all transitions to a given state."""
        return [t for t in self.transitions if t.target == state_name]

    def set_initial_state(self, state_name: str) -> bool:
        """Set the initial state."""
        if state_name in self.states:
            self._initial_state = state_name
            return True
        return False

    def start(self, context: GraphContext) -> None:
        """Start the state machine from the initial state."""
        if self._initial_state and self._initial_state in self.states:
            self._current_state = self.states[self._initial_state]
            self._current_state.enter(context)

    def force_state(self, state_name: str, context: GraphContext,
                    immediate: bool = False) -> bool:
        """Force transition to a specific state."""
        if state_name not in self.states:
            return False

        target_state = self.states[state_name]

        if immediate:
            # Immediate transition without blend
            if self._current_state:
                self._current_state.exit(context)
            self._current_state = target_state
            self._current_state.enter(context)
            self._active_transition = None
        else:
            # Create a transition
            config = get_config()
            transition = StateTransition(
                source=self._current_state.name if self._current_state else "",
                target=state_name,
                duration=config.transition.FORCED_TRANSITION_DURATION,
            )
            self._start_transition(transition, context)

        return True

    def _start_transition(self, transition: StateTransition,
                          context: GraphContext) -> None:
        """Start a transition to a new state."""
        if not self._current_state:
            return

        target_state = self.states.get(transition.target)
        if not target_state:
            return

        # Mark that we've started a transition this frame (infinite loop prevention)
        self._transition_this_frame = True

        # Store source pose for blending
        source_pose = self._current_state.evaluate(context)

        # Create active transition
        self._active_transition = ActiveTransition(
            transition=transition,
            source_state=self._current_state,
            target_state=target_state,
        )
        self._active_transition.source_pose = source_pose

        # Enter target state first (resets time)
        target_state.enter(context)

        # Apply sync mode to synchronize target state time with source
        self._apply_sync_mode(transition, self._current_state, target_state)

        # Apply explicit offset if specified (overrides sync mode)
        if transition.offset > 0:
            target_state.normalized_time = transition.offset

    def _apply_sync_mode(
        self,
        transition: StateTransition,
        source_state: AnimationState,
        target_state: AnimationState,
    ) -> None:
        """
        Apply sync mode to synchronize target state time with source.

        Args:
            transition: The transition being started
            source_state: The source state being transitioned from
            target_state: The target state being transitioned to
        """
        sync_mode = transition.sync_mode

        if sync_mode == TransitionSyncMode.NONE:
            # No synchronization - target starts from beginning (or offset)
            return

        elif sync_mode == TransitionSyncMode.NORMALIZED:
            # Match normalized time directly (0-1 maps to 0-1)
            target_state.normalized_time = source_state.normalized_time

        elif sync_mode == TransitionSyncMode.PROPORTIONAL:
            # Proportional sync: scale by duration ratio
            # If source is 50% through a 2s animation, and target is 1s,
            # target starts at 50% (0.5s)
            source_duration = source_state._get_duration()
            target_duration = target_state._get_duration()

            if source_duration > 0 and target_duration > 0:
                # Calculate how much time remains in source
                remaining_ratio = 1.0 - source_state.normalized_time
                # Scale target to have same proportional time remaining
                target_state.normalized_time = 1.0 - remaining_ratio
            else:
                # Fallback to normalized sync if durations are invalid
                target_state.normalized_time = source_state.normalized_time

        elif sync_mode == TransitionSyncMode.MARKER:
            # Marker-based sync would require animation markers/events
            # For now, fall back to normalized sync
            # TODO: Implement marker-based synchronization when marker system is available
            target_state.normalized_time = source_state.normalized_time

    def update(self, dt: float, context: GraphContext) -> None:
        """Update the state machine."""
        # Reset per-frame transition flag
        self._transition_this_frame = False

        # Initialize if needed
        if self._current_state is None and self._initial_state:
            self.start(context)

        if not self._current_state:
            return

        # Update active transition
        if self._active_transition:
            self._active_transition.update(dt)
            self._active_transition.target_state.update(dt, context)

            # Also update source state (for synchronized transitions)
            self._active_transition.source_state.update(dt, context)

            # Check for transition completion
            if self._active_transition.is_complete:
                self._complete_transition(context)
        else:
            # Update current state
            self._current_state.update(dt, context)

            # Check for new transitions
            self._check_transitions(context)

    def _check_transitions(self, context: GraphContext) -> None:
        """Check if any transition conditions are met."""
        if not self._current_state:
            return

        # Prevent multiple transitions in the same frame (infinite loop prevention)
        if self._transition_this_frame:
            return

        # Check any-state transitions first (highest priority)
        for transition in self._any_state_transitions:
            # Don't transition to current state from any-state
            if transition.target == self._current_state.name:
                continue
            if transition.can_transition(self._current_state, context):
                self._start_transition(transition, context)
                return

        # Check regular transitions from current state
        for transition in self.transitions:
            if transition.source != self._current_state.name:
                continue
            if transition.can_transition(self._current_state, context):
                self._start_transition(transition, context)
                return

    def _check_transition_interruption(self, context: GraphContext) -> bool:
        """
        Check if current transition should be interrupted.

        Uses the InterruptMode of the active transition to determine if
        interruption is allowed by candidate transitions.

        Returns:
            True if the transition was interrupted, False otherwise
        """
        if not self._active_transition:
            return False

        active_transition = self._active_transition.transition

        # Check any-state transitions for potential interruption
        for transition in self._any_state_transitions:
            # Skip if targeting the same state we're already transitioning to
            if transition.target == self._active_transition.target_state.name:
                continue

            # Check if interruption is allowed based on interrupt_mode
            if not active_transition.allows_interruption_by(transition.priority):
                continue

            target_state = self.states.get(transition.target)
            if target_state and transition.can_transition(
                self._active_transition.target_state, context
            ):
                # Interrupt current transition
                self._current_state = self._active_transition.target_state
                self._active_transition = None
                self._start_transition(transition, context)
                return True

        # Check regular transitions from the target state (for chained interruptions)
        for transition in self.transitions:
            if transition.source != self._active_transition.target_state.name:
                continue

            # Check if interruption is allowed
            if not active_transition.allows_interruption_by(transition.priority):
                continue

            target_state = self.states.get(transition.target)
            if target_state and transition.can_transition(
                self._active_transition.target_state, context
            ):
                # Interrupt current transition
                self._current_state = self._active_transition.target_state
                self._active_transition = None
                self._start_transition(transition, context)
                return True

        return False

    def _complete_transition(self, context: GraphContext) -> None:
        """Complete the current transition."""
        if not self._active_transition:
            return

        # Exit source state
        self._active_transition.source_state.exit(context)

        # Set new current state
        self._current_state = self._active_transition.target_state
        self._active_transition = None

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate the state machine and return the current pose."""
        # Initialize if needed
        if self._current_state is None and self._initial_state:
            self.start(context)

        if not self._current_state:
            return Pose()

        if self._active_transition:
            # Blend between source and target poses
            target_pose = self._active_transition.target_state.evaluate(context)
            weight = self._active_transition.blend_weight

            if self._active_transition.source_pose:
                return self._active_transition.source_pose.lerp(target_pose, weight)
            else:
                source_pose = self._active_transition.source_state.evaluate(context)
                return source_pose.lerp(target_pose, weight)
        else:
            # Return current state's pose
            return self._current_state.evaluate(context)

    def reset(self, context: GraphContext) -> None:
        """Reset the state machine to initial state."""
        if self._current_state:
            self._current_state.exit(context)

        self._current_state = None
        self._active_transition = None

        if self._initial_state:
            self.start(context)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the state machine."""
        info = super().get_debug_info()
        info.update({
            "current_state": self._current_state.name if self._current_state else None,
            "is_transitioning": self.is_transitioning,
            "states": list(self.states.keys()),
            "transition_count": len(self.transitions) + len(self._any_state_transitions),
        })
        if self._active_transition:
            info["transition"] = {
                "source": self._active_transition.source_state.name,
                "target": self._active_transition.target_state.name,
                "progress": self._active_transition.progress,
                "duration": self._active_transition.duration,
                "weight": self._active_transition.blend_weight,
            }
        return info


# =============================================================================
# STATE MACHINE DECORATOR
# =============================================================================


def state_machine(
    initial: str,
    states: Optional[Set[str]] = None,
    transitions: Optional[Dict[str, List[str]]] = None,
) -> Callable[[Type], Type]:
    """
    Decorator to define a state machine class.

    Usage:
        @state_machine(
            initial="idle",
            states={"idle", "walk", "run"},
            transitions={"idle": ["walk"], "walk": ["idle", "run"], "run": ["walk"]}
        )
        class LocomotionStateMachine:
            pass
    """
    def decorator(cls: Type) -> Type:
        cls._state_machine = True
        cls._sm_initial = initial
        cls._sm_states = frozenset(states) if states else frozenset()
        cls._sm_transitions = dict(transitions) if transitions else {}
        return cls
    return decorator


# =============================================================================
# STATE MACHINE BUILDER
# =============================================================================


class StateMachineBuilderError(ValueError):
    """Raised when StateMachineBuilder validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class StateMachineBuilder:
    """
    Fluent builder for creating state machines.

    Provides a clean, chainable API for constructing state machines with
    validation to catch configuration errors early.

    Example:
        sm = (
            StateMachineBuilder("locomotion")
            .add_state("idle", idle_clip)
            .add_state("walk", walk_clip, speed=1.2)
            .add_state("run", run_graph)
            .set_initial("idle")
            .add_transition("idle", "walk", TransitionCondition.is_true("moving"))
            .add_transition("walk", "run", TransitionCondition.greater_than("speed", 5.0))
            .add_transition("walk", "idle", TransitionCondition.is_false("moving"))
            .add_transition("run", "walk", TransitionCondition.less_than("speed", 5.0))
            .add_any_state_transition("hurt", TransitionCondition.trigger("take_damage"))
            .build()
        )
    """

    def __init__(self, name: str = "state_machine") -> None:
        """
        Initialize a new state machine builder.

        Args:
            name: Node ID for the state machine (default: "state_machine")
        """
        self._name = name
        self._states: Dict[str, AnimationState] = {}
        self._transitions: List[StateTransition] = []
        self._initial_state: Optional[str] = None

    def add_state(
        self,
        name: str,
        source: Any = None,
        *,
        clip: Optional["AnimationClip"] = None,
        graph: Optional["AnimationGraph"] = None,
        animation_node: Optional[AnimationNode] = None,
        motion_mode: MotionMode = MotionMode.LOOP,
        speed: float = 1.0,
        loop: Optional[bool] = None,  # Legacy parameter
        on_enter: Optional[Callable[[AnimationState, GraphContext], None]] = None,
        on_exit: Optional[Callable[[AnimationState, GraphContext], None]] = None,
        can_interrupt: bool = True,
    ) -> "StateMachineBuilder":
        """
        Add a state to the state machine.

        The animation source can be specified in multiple ways:
        - As the `source` positional argument (auto-detected type)
        - As explicit keyword arguments: `clip`, `graph`, or `animation_node`

        Args:
            name: Unique name for the state
            source: Animation source (clip, graph, or node) - auto-detected type
            clip: Explicit AnimationClip source
            graph: Explicit AnimationGraph source
            animation_node: Explicit AnimationNode source
            motion_mode: How the animation should loop/repeat
            speed: Playback speed multiplier (1.0 = normal)
            loop: Legacy parameter - use motion_mode instead
            on_enter: Callback invoked when entering this state
            on_exit: Callback invoked when exiting this state
            can_interrupt: Whether this state can be interrupted mid-animation

        Returns:
            Self for method chaining

        Raises:
            StateMachineBuilderError: If state name is empty or already exists
        """
        # Validate name
        if not name or not name.strip():
            raise StateMachineBuilderError(
                "State name cannot be empty",
                {"provided_name": repr(name)}
            )

        if name in self._states:
            raise StateMachineBuilderError(
                f"State '{name}' already exists in the state machine",
                {"existing_states": list(self._states.keys())}
            )

        # Auto-detect source type if provided as positional argument
        detected_clip = clip
        detected_graph = graph
        detected_node = animation_node

        if source is not None:
            # Detect type based on attributes/class
            if hasattr(source, "sample") and hasattr(source, "duration"):
                # Looks like an AnimationClip
                detected_clip = source
            elif isinstance(source, AnimationGraph):
                detected_graph = source
            elif isinstance(source, AnimationNode):
                detected_node = source
            else:
                # Assume it's a generic animation node
                detected_node = source

        # Handle legacy 'loop' parameter
        if loop is not None:
            motion_mode = MotionMode.LOOP if loop else MotionMode.ONCE

        state = AnimationState(
            name=name,
            clip=detected_clip,
            graph=detected_graph,
            animation_node=detected_node,
            motion_mode=motion_mode,
            speed=speed,
            on_enter=on_enter,
            on_exit=on_exit,
            can_interrupt=can_interrupt,
        )
        self._states[name] = state

        # Auto-set initial state to first added state
        if self._initial_state is None:
            self._initial_state = name

        return self

    def set_initial(self, state_name: str) -> "StateMachineBuilder":
        """
        Set the initial/default state.

        Args:
            state_name: Name of the state to set as initial

        Returns:
            Self for method chaining

        Note:
            Validation that the state exists is performed in build().
        """
        self._initial_state = state_name
        return self

    def add_transition(
        self,
        source: str,
        target: str,
        condition: Optional[TransitionCondition] = None,
        *,
        conditions: Optional[List[TransitionCondition]] = None,
        duration: float = 0.25,
        duration_mode: str = "fixed",
        blend_curve: BlendCurve = BlendCurve.SMOOTH_STEP,
        sync_mode: TransitionSyncMode = TransitionSyncMode.NONE,
        exit_time: Optional[float] = None,
        offset: float = 0.0,
        priority: int = 0,
        interrupt_mode: InterruptMode = InterruptMode.HIGHER_PRIORITY,
        # Legacy parameter
        curve: Optional[BlendCurve] = None,
    ) -> "StateMachineBuilder":
        """
        Add a transition between states.

        Args:
            source: Source state name
            target: Target state name
            condition: Single transition condition (convenience parameter)
            conditions: List of conditions (all must pass for transition)
            duration: Blend duration in seconds (or percentage if duration_mode="percentage")
            duration_mode: "fixed" (seconds) or "percentage" (of source animation)
            blend_curve: Easing curve for the blend weight
            sync_mode: How to synchronize animation time during transition
            exit_time: Normalized time (0-1) when transition becomes valid
            offset: Start offset in target state (normalized time)
            priority: Higher values = higher priority (checked first)
            interrupt_mode: Controls whether/how this transition can be interrupted
            curve: Legacy alias for blend_curve

        Returns:
            Self for method chaining
        """
        # Build conditions list
        all_conditions: List[TransitionCondition] = []
        if condition is not None:
            all_conditions.append(condition)
        if conditions:
            all_conditions.extend(conditions)

        # Handle legacy 'curve' parameter
        if curve is not None:
            blend_curve = curve

        transition = StateTransition(
            source=source,
            target=target,
            conditions=all_conditions,
            duration=duration,
            duration_mode=duration_mode,
            blend_curve=blend_curve,
            sync_mode=sync_mode,
            exit_time=exit_time,
            offset=offset,
            priority=priority,
            interrupt_mode=interrupt_mode,
        )
        self._transitions.append(transition)
        return self

    def add_any_state_transition(
        self,
        target: str,
        condition: Optional[TransitionCondition] = None,
        *,
        conditions: Optional[List[TransitionCondition]] = None,
        duration: float = 0.2,
        blend_curve: BlendCurve = BlendCurve.SMOOTH_STEP,
        priority: Optional[int] = None,
        # Legacy parameter
        curve: Optional[BlendCurve] = None,
    ) -> "StateMachineBuilder":
        """
        Add an any-state transition (source='*').

        Any-state transitions can occur from any state and are typically used
        for global events like damage reactions or death states.

        Args:
            target: Target state name
            condition: Single transition condition (convenience parameter)
            conditions: List of conditions (all must pass for transition)
            duration: Blend duration in seconds
            blend_curve: Easing curve for the blend weight
            priority: Transition priority (default: high priority from config)
            curve: Legacy alias for blend_curve

        Returns:
            Self for method chaining
        """
        if priority is None:
            config = get_config()
            priority = config.transition.ANY_STATE_PRIORITY

        # Handle legacy 'curve' parameter
        if curve is not None:
            blend_curve = curve

        return self.add_transition(
            source="*",
            target=target,
            condition=condition,
            conditions=conditions,
            duration=duration,
            blend_curve=blend_curve,
            priority=priority,
        )

    def _validate(self) -> List[str]:
        """
        Validate the state machine configuration.

        Returns:
            List of error messages (empty if valid)
        """
        errors: List[str] = []

        # Check for at least one state
        if not self._states:
            errors.append("State machine must have at least one state")

        # Validate initial state
        if self._initial_state is not None and self._initial_state not in self._states:
            errors.append(
                f"Initial state '{self._initial_state}' does not exist. "
                f"Available states: {list(self._states.keys())}"
            )

        # Validate transitions reference existing states
        all_state_names = set(self._states.keys())
        for i, transition in enumerate(self._transitions):
            # Validate source (skip '*' for any-state)
            if transition.source != "*" and transition.source not in all_state_names:
                errors.append(
                    f"Transition {i + 1}: source state '{transition.source}' does not exist. "
                    f"Available states: {list(all_state_names)}"
                )

            # Validate target
            if transition.target not in all_state_names:
                errors.append(
                    f"Transition {i + 1}: target state '{transition.target}' does not exist. "
                    f"Available states: {list(all_state_names)}"
                )

            # Warn about self-transitions (usually a mistake)
            if transition.source == transition.target and transition.source != "*":
                # Not an error, but could be intentional for animation restarts
                pass

        # Check for orphaned states (no incoming transitions except initial)
        states_with_incoming: Set[str] = set()
        for transition in self._transitions:
            states_with_incoming.add(transition.target)

        if self._initial_state:
            states_with_incoming.add(self._initial_state)

        orphaned = all_state_names - states_with_incoming
        # Note: Orphaned states are not errors, just potential issues
        # Users might want states only reachable via force_state()

        return errors

    def build(self) -> StateMachine:
        """
        Build and validate the state machine.

        Returns:
            A fully configured StateMachine instance

        Raises:
            StateMachineBuilderError: If validation fails with details about
                all configuration errors found
        """
        # Validate configuration
        errors = self._validate()
        if errors:
            error_list = "\n  - ".join(errors)
            raise StateMachineBuilderError(
                f"Invalid state machine configuration:\n  - {error_list}",
                {
                    "errors": errors,
                    "state_count": len(self._states),
                    "transition_count": len(self._transitions),
                    "initial_state": self._initial_state,
                }
            )

        # Build the state machine
        sm = StateMachine(self._name, self._initial_state)

        for state in self._states.values():
            sm.add_state(state)

        for transition in self._transitions:
            sm.add_transition(transition)

        return sm

    def __repr__(self) -> str:
        return (
            f"StateMachineBuilder(name={self._name!r}, "
            f"states={len(self._states)}, "
            f"transitions={len(self._transitions)}, "
            f"initial={self._initial_state!r})"
        )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Blend curves
    "BlendCurve",
    "evaluate_blend_curve",
    # Motion mode
    "MotionMode",
    # Conditions
    "ConditionOperator",
    "ComparisonOp",  # Legacy alias for ConditionOperator
    "TransitionCondition",
    "ParameterTypeError",
    # State
    "AnimationState",
    # Transition
    "InterruptMode",
    "TransitionSyncMode",
    "StateTransition",
    "ActiveTransition",
    # State machine
    "StateMachine",
    "StateMachineBuilder",
    "StateMachineBuilderError",
    # Decorator
    "state_machine",
]
