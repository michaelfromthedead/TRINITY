"""ECS system for animation graphs.

Processes entities with animation graph components, evaluating state machines
and blend trees to produce poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import ANIMATION_SYSTEM_CONFIG


class ParameterType(Enum):
    """Animation graph parameter type."""
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    TRIGGER = auto()


@dataclass
class GraphParameter:
    """Animation graph parameter definition.

    Attributes:
        name: Parameter name
        param_type: Parameter type
        value: Current value
        default_value: Default value
        min_value: Minimum value (for float/int)
        max_value: Maximum value (for float/int)
    """
    name: str = ""
    param_type: ParameterType = ParameterType.FLOAT
    value: Any = 0.0
    default_value: Any = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def set_value(self, new_value: Any) -> None:
        """Set parameter value with type checking."""
        if self.param_type == ParameterType.FLOAT:
            val = float(new_value)
            if self.min_value is not None:
                val = max(self.min_value, val)
            if self.max_value is not None:
                val = min(self.max_value, val)
            self.value = val
        elif self.param_type == ParameterType.INT:
            val = int(new_value)
            if self.min_value is not None:
                val = max(int(self.min_value), val)
            if self.max_value is not None:
                val = min(int(self.max_value), val)
            self.value = val
        elif self.param_type == ParameterType.BOOL:
            self.value = bool(new_value)
        elif self.param_type == ParameterType.TRIGGER:
            self.value = bool(new_value)

    def reset(self) -> None:
        """Reset to default value."""
        self.value = self.default_value

    def consume_trigger(self) -> bool:
        """Consume trigger if set, returns previous value."""
        if self.param_type == ParameterType.TRIGGER:
            was_set = bool(self.value)
            self.value = False
            return was_set
        return False


@dataclass
class AnimationState:
    """State in animation state machine."""
    name: str = ""
    animation_clip: str = ""
    speed: float = 1.0
    loop: bool = True
    blend_time: float = ANIMATION_SYSTEM_CONFIG.DEFAULT_GRAPH_TRANSITION


@dataclass
class StateTransition:
    """Transition between animation states."""
    from_state: str = ""
    to_state: str = ""
    condition: Callable[[dict[str, GraphParameter]], bool] | None = None
    duration: float = ANIMATION_SYSTEM_CONFIG.DEFAULT_GRAPH_TRANSITION
    has_exit_time: bool = False
    exit_time: float = 1.0


@dataclass
class AnimationGraphInstance:
    """Runtime instance of an animation graph.

    Attributes:
        current_state: Current state name
        states: Available states
        transitions: State transitions
        parameters: Graph parameters
        state_time: Time in current state
        transition_time: Current transition time
        transitioning: Whether currently transitioning
        transition_target: Target state during transition
    """
    current_state: str = ""
    states: dict[str, AnimationState] = field(default_factory=dict)
    transitions: list[StateTransition] = field(default_factory=list)
    parameters: dict[str, GraphParameter] = field(default_factory=dict)
    state_time: float = 0.0
    transition_time: float = 0.0
    transitioning: bool = False
    transition_target: str = ""
    transition_duration: float = 0.0

    def add_state(self, state: AnimationState) -> None:
        """Add a state to the graph."""
        self.states[state.name] = state
        if not self.current_state:
            self.current_state = state.name

    def add_transition(self, transition: StateTransition) -> None:
        """Add a transition to the graph."""
        self.transitions.append(transition)

    def add_parameter(self, param: GraphParameter) -> None:
        """Add a parameter to the graph."""
        self.parameters[param.name] = param

    def set_parameter(self, name: str, value: Any) -> bool:
        """Set parameter value.

        Returns:
            True if parameter exists
        """
        param = self.parameters.get(name)
        if param:
            param.set_value(value)
            return True
        return False

    def get_parameter(self, name: str) -> GraphParameter | None:
        """Get parameter by name."""
        return self.parameters.get(name)

    def get_current_animation(self) -> str:
        """Get current animation clip name."""
        state = self.states.get(self.current_state)
        return state.animation_clip if state else ""

    def get_blend_weight(self) -> float:
        """Get current transition blend weight (0 = from, 1 = to)."""
        if not self.transitioning:
            return 1.0
        if self.transition_duration <= 0:
            return 1.0
        return min(1.0, self.transition_time / self.transition_duration)


@dataclass
class AnimationPose:
    """Output pose from animation evaluation."""
    bone_transforms: dict[int, Transform] = field(default_factory=dict)
    root_motion: Transform = field(default_factory=Transform.identity)

    def get_bone_transform(self, bone_index: int) -> Transform:
        """Get transform for bone, identity if not present."""
        return self.bone_transforms.get(bone_index, Transform.identity())

    def blend_with(self, other: AnimationPose, weight: float) -> AnimationPose:
        """Blend this pose with another."""
        result = AnimationPose()

        all_bones = set(self.bone_transforms.keys()) | set(other.bone_transforms.keys())
        for bone_idx in all_bones:
            t1 = self.get_bone_transform(bone_idx)
            t2 = other.get_bone_transform(bone_idx)
            result.bone_transforms[bone_idx] = t1.lerp(t2, weight)

        result.root_motion = self.root_motion.lerp(other.root_motion, weight)
        return result


@dataclass
class AnimationGraphComponent:
    """Component for entities with animation graphs.

    Attributes:
        graph: Animation graph instance
        output_pose: Evaluated pose output
        enabled: Whether animation is enabled
        time_scale: Animation time scale
    """
    graph: AnimationGraphInstance = field(default_factory=AnimationGraphInstance)
    output_pose: AnimationPose = field(default_factory=AnimationPose)
    enabled: bool = True
    time_scale: float = 1.0

    # For syncing with gameplay
    parameter_bindings: dict[str, str] = field(default_factory=dict)  # param_name -> gameplay_property


class AnimationGraphSystem:
    """ECS system for evaluating animation graphs.

    Processes entities with AnimationGraphComponent, evaluates state machines,
    and produces poses.
    """

    def __init__(self):
        self._pose_cache: dict[str, AnimationPose] = {}
        self._animation_provider: Callable[[str, float], AnimationPose] | None = None

    def set_animation_provider(
        self,
        provider: Callable[[str, float], AnimationPose]
    ) -> None:
        """Set function that provides animation poses.

        Args:
            provider: Function(clip_name, time) -> AnimationPose
        """
        self._animation_provider = provider

    def update(
        self,
        world: World,
        dt: float,
        entity_components: list[tuple[Entity, AnimationGraphComponent]]
    ) -> None:
        """Update all animation graph components.

        Args:
            world: ECS world
            dt: Delta time in seconds
            entity_components: List of (entity, component) tuples
        """
        for entity, component in entity_components:
            if not component.enabled:
                continue

            self._update_graph(component, dt * component.time_scale)

    def _update_graph(self, component: AnimationGraphComponent, dt: float) -> None:
        """Update single animation graph."""
        graph = component.graph

        # Update state time
        graph.state_time += dt

        # Check for transitions
        if graph.transitioning:
            self._update_transition(graph, dt)
        else:
            self._check_transitions(graph)

        # Evaluate pose
        component.output_pose = self._evaluate_pose(graph)

    def _check_transitions(self, graph: AnimationGraphInstance) -> None:
        """Check if any transition conditions are met."""
        current_state = graph.states.get(graph.current_state)
        if not current_state:
            return

        for transition in graph.transitions:
            if transition.from_state != graph.current_state:
                continue

            # Check exit time
            if transition.has_exit_time:
                # Assume normalized time for looping
                if graph.state_time < transition.exit_time:
                    continue

            # Check condition
            if transition.condition is not None:
                if not transition.condition(graph.parameters):
                    continue

            # Start transition
            self._start_transition(graph, transition)
            break

    def _start_transition(
        self,
        graph: AnimationGraphInstance,
        transition: StateTransition
    ) -> None:
        """Start a state transition."""
        graph.transitioning = True
        graph.transition_target = transition.to_state
        graph.transition_time = 0.0
        graph.transition_duration = transition.duration

    def _update_transition(self, graph: AnimationGraphInstance, dt: float) -> None:
        """Update ongoing transition."""
        graph.transition_time += dt

        if graph.transition_time >= graph.transition_duration:
            # Complete transition
            graph.current_state = graph.transition_target
            graph.transitioning = False
            graph.state_time = 0.0
            graph.transition_target = ""

            # Consume any triggers
            for param in graph.parameters.values():
                param.consume_trigger()

    def _evaluate_pose(self, graph: AnimationGraphInstance) -> AnimationPose:
        """Evaluate current pose from graph state."""
        current_state = graph.states.get(graph.current_state)
        if not current_state:
            return AnimationPose()

        # Get current animation pose
        current_pose = self._sample_animation(
            current_state.animation_clip,
            graph.state_time * current_state.speed,
            current_state.loop,
        )

        if not graph.transitioning:
            return current_pose

        # Blend with target state
        target_state = graph.states.get(graph.transition_target)
        if not target_state:
            return current_pose

        target_pose = self._sample_animation(
            target_state.animation_clip,
            graph.transition_time * target_state.speed,
            target_state.loop,
        )

        blend_weight = graph.get_blend_weight()
        return current_pose.blend_with(target_pose, blend_weight)

    def _sample_animation(
        self,
        clip_name: str,
        time: float,
        loop: bool = True
    ) -> AnimationPose:
        """Sample animation at given time."""
        if self._animation_provider:
            return self._animation_provider(clip_name, time)

        # Default empty pose
        return AnimationPose()

    def sync_parameters_from_gameplay(
        self,
        component: AnimationGraphComponent,
        gameplay_data: dict[str, Any]
    ) -> None:
        """Sync graph parameters from gameplay data.

        Args:
            component: Animation graph component
            gameplay_data: Dictionary of gameplay properties
        """
        for param_name, gameplay_prop in component.parameter_bindings.items():
            if gameplay_prop in gameplay_data:
                component.graph.set_parameter(param_name, gameplay_data[gameplay_prop])

    def trigger_transition(
        self,
        component: AnimationGraphComponent,
        target_state: str
    ) -> bool:
        """Manually trigger transition to target state.

        Args:
            component: Animation graph component
            target_state: Target state name

        Returns:
            True if transition started
        """
        graph = component.graph
        if target_state not in graph.states:
            return False

        if graph.transitioning:
            return False

        # Find matching transition
        for transition in graph.transitions:
            if transition.from_state == graph.current_state and transition.to_state == target_state:
                self._start_transition(graph, transition)
                return True

        # Force transition if no explicit transition exists
        graph.transitioning = True
        graph.transition_target = target_state
        graph.transition_time = 0.0
        graph.transition_duration = ANIMATION_SYSTEM_CONFIG.DEFAULT_GRAPH_TRANSITION
        return True
