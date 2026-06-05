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
    GraphContext,
    GraphParameter,
    ParameterType,
    Pose,
    Transform,
)
from .config import get_config


# =============================================================================
# EXCEPTIONS
# =============================================================================


class StateMachineBuilderError(Exception):
    """Raised when state machine builder encounters invalid configuration."""
    pass


class ParameterTypeError(TypeError):
    """Raised when parameter type doesn't match expected type."""
    pass


# =============================================================================
# INTERRUPT MODE
# =============================================================================


class InterruptMode(Enum):
    """How transitions can be interrupted."""

    NONE = auto()  # Cannot be interrupted
    HIGHER_PRIORITY = auto()  # Can be interrupted by higher priority transitions
    ANY = auto()  # Can be interrupted by any transition
    SAME_PRIORITY = auto()  # Can be interrupted by same or higher priority


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
# MOTION MODE
# =============================================================================


class MotionMode(Enum):
    """Mode for handling root motion in animation states.

    ``NONE``
        No root motion is applied.
    ``IN_PLACE``
        Animation plays in place, root motion is discarded.
    ``EXTRACT``
        Root motion is extracted and applied to the character.
    ``BLEND``
        Root motion is blended with gameplay-driven movement.
    """
    NONE = auto()
    IN_PLACE = auto()
    EXTRACT = auto()
    BLEND = auto()


# =============================================================================
# TRANSITION CONDITIONS
# =============================================================================


class ComparisonOp(Enum):
    """Comparison operators for conditions."""

    EQUALS = auto()
    NOT_EQUALS = auto()
    GREATER = auto()
    GREATER_THAN = GREATER  # Alias
    GREATER_EQUALS = auto()
    GREATER_OR_EQUAL = GREATER_EQUALS  # Alias
    LESS = auto()
    LESS_THAN = LESS  # Alias
    LESS_EQUALS = auto()
    LESS_OR_EQUAL = LESS_EQUALS  # Alias
    IS_TRUE = auto()
    IS_FALSE = auto()


# Alias for backwards compatibility
ConditionOperator = ComparisonOp


@dataclass
class TransitionCondition:
    """A condition that must be met for a transition to occur."""

    parameter_name: str
    comparison: ComparisonOp
    value: Any = None

    def evaluate(self, context: GraphContext) -> bool:
        """Evaluate this condition against the context."""
        param_value = context.get_parameter(self.parameter_name)

        if param_value is None:
            return False

        if self.comparison == ComparisonOp.EQUALS:
            return param_value == self.value
        elif self.comparison == ComparisonOp.NOT_EQUALS:
            return param_value != self.value
        elif self.comparison == ComparisonOp.GREATER:
            return param_value > self.value
        elif self.comparison == ComparisonOp.GREATER_EQUALS:
            return param_value >= self.value
        elif self.comparison == ComparisonOp.LESS:
            return param_value < self.value
        elif self.comparison == ComparisonOp.LESS_EQUALS:
            return param_value <= self.value
        elif self.comparison == ComparisonOp.IS_TRUE:
            return bool(param_value)
        elif self.comparison == ComparisonOp.IS_FALSE:
            return not bool(param_value)

        return False

    @classmethod
    def equals(cls, param: str, value: Any) -> "TransitionCondition":
        """Create an equals condition."""
        return cls(param, ComparisonOp.EQUALS, value)

    @classmethod
    def not_equals(cls, param: str, value: Any) -> "TransitionCondition":
        """Create a not equals condition."""
        return cls(param, ComparisonOp.NOT_EQUALS, value)

    @classmethod
    def greater_than(cls, param: str, value: float) -> "TransitionCondition":
        """Create a greater than condition."""
        return cls(param, ComparisonOp.GREATER, value)

    @classmethod
    def greater_or_equal(cls, param: str, value: float) -> "TransitionCondition":
        """Create a greater or equal condition."""
        return cls(param, ComparisonOp.GREATER_EQUALS, value)

    @classmethod
    def less_than(cls, param: str, value: float) -> "TransitionCondition":
        """Create a less than condition."""
        return cls(param, ComparisonOp.LESS, value)

    @classmethod
    def less_or_equal(cls, param: str, value: float) -> "TransitionCondition":
        """Create a less or equal condition."""
        return cls(param, ComparisonOp.LESS_EQUALS, value)

    @classmethod
    def is_true(cls, param: str) -> "TransitionCondition":
        """Create a boolean true condition."""
        return cls(param, ComparisonOp.IS_TRUE)

    @classmethod
    def is_false(cls, param: str) -> "TransitionCondition":
        """Create a boolean false condition."""
        return cls(param, ComparisonOp.IS_FALSE)

    @classmethod
    def trigger(cls, param: str) -> "TransitionCondition":
        """Create a trigger condition (fires once when triggered)."""
        return cls(param, ComparisonOp.IS_TRUE)


# =============================================================================
# ANIMATION STATE
# =============================================================================


@dataclass
class AnimationState:
    """
    A state in the animation state machine.

    Each state has an associated animation node that produces the pose
    when the state is active. States can have enter/exit events for
    triggering side effects.
    """

    name: str
    animation_node: Optional[AnimationNode] = None

    # Optional events
    on_enter: Optional[Callable[["AnimationState", GraphContext], None]] = None
    on_exit: Optional[Callable[["AnimationState", GraphContext], None]] = None
    on_update: Optional[Callable[["AnimationState", GraphContext, float], None]] = None

    # State metadata
    speed_multiplier: float = 1.0
    loop: bool = True
    can_interrupt: bool = True  # Whether this state can be interrupted mid-animation

    # Runtime state
    _time_in_state: float = field(default=0.0, repr=False)
    _normalized_time: float = field(default=0.0, repr=False)

    def enter(self, context: GraphContext) -> None:
        """Called when entering this state."""
        self._time_in_state = 0.0
        self._normalized_time = 0.0
        if self.on_enter:
            self.on_enter(self, context)

    def exit(self, context: GraphContext) -> None:
        """Called when exiting this state."""
        if self.on_exit:
            self.on_exit(self, context)

    def update(self, context: GraphContext, dt: float) -> None:
        """Update the state."""
        self._time_in_state += dt * self.speed_multiplier
        if self.on_update:
            self.on_update(self, context, dt)

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate this state's animation node."""
        if self.animation_node:
            # Create a context with the state's normalized time
            state_context = GraphContext(
                parameters=context.parameters,
                dt=context.dt,
                skeleton=context.skeleton,
                bone_masks=context.bone_masks,
                normalized_time=self._normalized_time,
                sync_group=context.sync_group,
                layer_weight=context.layer_weight,
            )
            return self.animation_node.evaluate(state_context)
        return Pose()

    @property
    def time_in_state(self) -> float:
        """Get the time spent in this state."""
        return self._time_in_state

    @property
    def normalized_time(self) -> float:
        """Get the normalized time (0-1) in this state."""
        return self._normalized_time

    @normalized_time.setter
    def normalized_time(self, value: float) -> None:
        """Set the normalized time."""
        self._normalized_time = value


# =============================================================================
# STATE TRANSITION
# =============================================================================


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
    """

    source: str  # Source state name, or "*" for any-state
    target: str  # Target state name
    conditions: List[TransitionCondition] = field(default_factory=list)

    # Blend settings
    duration: float = 0.2  # Blend duration in seconds
    blend_curve: BlendCurve = BlendCurve.SMOOTH_STEP
    sync_mode: TransitionSyncMode = TransitionSyncMode.NONE

    # Timing settings
    exit_time: Optional[float] = None  # Normalized time to exit (0-1)
    fixed_duration: bool = True  # Use fixed duration vs percentage
    offset: float = 0.0  # Start offset in target state

    # Priority
    priority: int = 0  # Higher priority transitions are checked first
    can_interrupt_self: bool = False  # Can this transition interrupt an active transition
    can_be_interrupted: bool = True  # Can this transition be interrupted
    interrupt_mode: InterruptMode = InterruptMode.HIGHER_PRIORITY  # Interruption behavior

    def can_transition(self, current_state: AnimationState,
                       context: GraphContext) -> bool:
        """Check if this transition can occur."""
        # Check exit time
        if self.exit_time is not None:
            if current_state.normalized_time < self.exit_time:
                return False

        # Check all conditions
        return all(cond.evaluate(context) for cond in self.conditions)

    def is_any_state(self) -> bool:
        """Check if this is an any-state transition."""
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
        self.duration = self.transition.duration

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

        # Apply offset to target state
        if transition.offset > 0:
            target_state.normalized_time = transition.offset

        # Enter target state
        target_state.enter(context)

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
            self._active_transition.target_state.update(context, dt)

            # Also update source state (for synchronized transitions)
            self._active_transition.source_state.update(context, dt)

            # Check for transition completion
            if self._active_transition.is_complete:
                self._complete_transition(context)
        else:
            # Update current state
            self._current_state.update(context, dt)

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
        """Check if current transition should be interrupted."""
        if not self._active_transition:
            return False

        if not self._active_transition.transition.can_be_interrupted:
            return False

        # Check for higher priority transitions
        current_priority = self._active_transition.transition.priority

        # Check any-state transitions
        for transition in self._any_state_transitions:
            if transition.priority <= current_priority:
                break  # Sorted by priority, so we can stop early
            if transition.target == self._active_transition.target_state.name:
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


class StateMachineBuilder:
    """Fluent builder for creating state machines."""

    def __init__(self, node_id: str) -> None:
        self._node_id = node_id
        self._states: Dict[str, AnimationState] = {}
        self._transitions: List[StateTransition] = []
        self._initial_state: Optional[str] = None

    def add_state(
        self,
        name: str,
        animation_node: Optional[AnimationNode] = None,
        speed: float = 1.0,
        loop: bool = True,
    ) -> "StateMachineBuilder":
        """Add a state to the builder."""
        state = AnimationState(
            name=name,
            animation_node=animation_node,
            speed_multiplier=speed,
            loop=loop,
        )
        self._states[name] = state
        if self._initial_state is None:
            self._initial_state = name
        return self

    def set_initial(self, state_name: str) -> "StateMachineBuilder":
        """Set the initial state."""
        self._initial_state = state_name
        return self

    def add_transition(
        self,
        source: str,
        target: str,
        conditions: Optional[List[TransitionCondition]] = None,
        duration: float = 0.2,
        curve: BlendCurve = BlendCurve.SMOOTH_STEP,
        priority: int = 0,
        interrupt_mode: InterruptMode = InterruptMode.HIGHER_PRIORITY,
    ) -> "StateMachineBuilder":
        """Add a transition between states."""
        transition = StateTransition(
            source=source,
            target=target,
            conditions=conditions or [],
            duration=duration,
            blend_curve=curve,
            priority=priority,
            interrupt_mode=interrupt_mode,
        )
        self._transitions.append(transition)
        return self

    def add_any_state_transition(
        self,
        target: str,
        conditions: Optional[List[TransitionCondition]] = None,
        duration: float = 0.2,
        curve: BlendCurve = BlendCurve.SMOOTH_STEP,
        priority: Optional[int] = None,  # High priority by default (from config)
    ) -> "StateMachineBuilder":
        """Add an any-state transition."""
        if priority is None:
            config = get_config()
            priority = config.transition.ANY_STATE_PRIORITY
        return self.add_transition(
            source="*",
            target=target,
            conditions=conditions,
            duration=duration,
            curve=curve,
            priority=priority,
        )

    def build(self) -> StateMachine:
        """Build the state machine."""
        sm = StateMachine(self._node_id, self._initial_state)

        for state in self._states.values():
            sm.add_state(state)

        for transition in self._transitions:
            sm.add_transition(transition)

        return sm


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
    "ComparisonOp",
    "TransitionCondition",
    # State
    "AnimationState",
    # Transition
    "TransitionSyncMode",
    "StateTransition",
    "ActiveTransition",
    # State machine
    "StateMachine",
    "StateMachineBuilder",
    # Decorator
    "state_machine",
]
