"""AI subsystem: Behavior trees, blackboards, utility AI, GOAP, perception, and combat AI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from heapq import heappush, heappop
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from engine.gameplay.constants import (
    BTNodeStatus,
    UtilityCurveType,
    PerceptionSense,
    AI_UPDATE_RATE_DEFAULT,
    GOAP_MAX_PLAN_DEPTH,
    GOAP_MAX_ITERATIONS,
    PERCEPTION_DEFAULT_SIGHT_RANGE,
    PERCEPTION_DEFAULT_HEARING_RANGE,
    PERCEPTION_DEFAULT_FOV,
    UTILITY_SCORE_MIN,
    UTILITY_SCORE_MAX,
    UTILITY_LOGISTIC_CENTER,
    UTILITY_LOGISTIC_STEEPNESS,
)

if TYPE_CHECKING:
    from engine.gameplay.entity import Actor


# === Blackboard System ===

class BlackboardKey:
    """Typed blackboard key."""

    def __init__(self, name: str, value_type: type) -> None:
        self._name = name
        self._value_type = value_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def value_type(self) -> type:
        return self._value_type


class Blackboard:
    """AI knowledge storage with observers and scoping."""

    def __init__(self, parent: Optional[Blackboard] = None) -> None:
        self._data: Dict[str, Any] = {}
        self._observers: Dict[str, List[Callable[[str, Any, Any], None]]] = {}
        self._parent: Optional[Blackboard] = parent

    def set(self, key: str, value: Any) -> None:
        """Set blackboard value, notifying observers."""
        old_value = self._data.get(key)
        self._data[key] = value
        self._notify_observers(key, old_value, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get blackboard value, checking parent if not found."""
        if key in self._data:
            return self._data[key]
        if self._parent:
            return self._parent.get(key, default)
        return default

    def has(self, key: str) -> bool:
        """Check if key exists in blackboard or parent."""
        if key in self._data:
            return True
        if self._parent:
            return self._parent.has(key)
        return False

    def remove(self, key: str) -> bool:
        """Remove key from blackboard."""
        if key in self._data:
            old_value = self._data.pop(key)
            self._notify_observers(key, old_value, None)
            return True
        return False

    def clear(self) -> None:
        """Clear all blackboard data."""
        for key in list(self._data.keys()):
            self.remove(key)

    def add_observer(
        self,
        key: str,
        callback: Callable[[str, Any, Any], None],
    ) -> None:
        """Add observer for key changes."""
        if key not in self._observers:
            self._observers[key] = []
        self._observers[key].append(callback)

    def remove_observer(
        self,
        key: str,
        callback: Callable[[str, Any, Any], None],
    ) -> None:
        """Remove observer for key."""
        if key in self._observers:
            if callback in self._observers[key]:
                self._observers[key].remove(callback)

    def _notify_observers(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify observers of value change."""
        if key in self._observers:
            for callback in self._observers[key]:
                callback(key, old_value, new_value)

    def get_keys(self) -> List[str]:
        """Get all keys in blackboard."""
        keys = set(self._data.keys())
        if self._parent:
            keys.update(self._parent.get_keys())
        return list(keys)


# === Behavior Tree ===

class BTNode(ABC):
    """Base behavior tree node."""

    def __init__(self, name: str = "") -> None:
        self._name = name or self.__class__.__name__
        self._status: BTNodeStatus = BTNodeStatus.RUNNING
        self._blackboard: Optional[Blackboard] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> BTNodeStatus:
        return self._status

    def set_blackboard(self, blackboard: Blackboard) -> None:
        """Set blackboard for this node."""
        self._blackboard = blackboard

    @abstractmethod
    def tick(self, delta_time: float) -> BTNodeStatus:
        """Execute node logic and return status."""
        pass

    def reset(self) -> None:
        """Reset node state."""
        self._status = BTNodeStatus.RUNNING

    def abort(self) -> None:
        """Abort node execution."""
        self._status = BTNodeStatus.FAILURE


class BTComposite(BTNode):
    """Composite node with children."""

    def __init__(self, name: str = "", children: Optional[List[BTNode]] = None) -> None:
        super().__init__(name)
        self._children: List[BTNode] = children or []
        self._current_child: int = 0

    def add_child(self, child: BTNode) -> None:
        """Add child node."""
        self._children.append(child)
        if self._blackboard:
            child.set_blackboard(self._blackboard)

    def set_blackboard(self, blackboard: Blackboard) -> None:
        super().set_blackboard(blackboard)
        for child in self._children:
            child.set_blackboard(blackboard)

    def reset(self) -> None:
        super().reset()
        self._current_child = 0
        for child in self._children:
            child.reset()


class BTSelector(BTComposite):
    """Selector: succeeds if any child succeeds (OR)."""

    def tick(self, delta_time: float) -> BTNodeStatus:
        while self._current_child < len(self._children):
            child = self._children[self._current_child]
            status = child.tick(delta_time)

            if status == BTNodeStatus.RUNNING:
                self._status = BTNodeStatus.RUNNING
                return self._status

            if status == BTNodeStatus.SUCCESS:
                self._status = BTNodeStatus.SUCCESS
                return self._status

            self._current_child += 1

        self._status = BTNodeStatus.FAILURE
        return self._status


class BTSequence(BTComposite):
    """Sequence: succeeds if all children succeed (AND)."""

    def tick(self, delta_time: float) -> BTNodeStatus:
        while self._current_child < len(self._children):
            child = self._children[self._current_child]
            status = child.tick(delta_time)

            if status == BTNodeStatus.RUNNING:
                self._status = BTNodeStatus.RUNNING
                return self._status

            if status == BTNodeStatus.FAILURE:
                self._status = BTNodeStatus.FAILURE
                return self._status

            self._current_child += 1

        self._status = BTNodeStatus.SUCCESS
        return self._status


class BTParallel(BTComposite):
    """Parallel: runs all children simultaneously."""

    def __init__(
        self,
        name: str = "",
        children: Optional[List[BTNode]] = None,
        success_threshold: int = 1,
        failure_threshold: int = 1,
    ) -> None:
        super().__init__(name, children)
        self._success_threshold = success_threshold
        self._failure_threshold = failure_threshold

    def tick(self, delta_time: float) -> BTNodeStatus:
        success_count = 0
        failure_count = 0

        for child in self._children:
            if child.status == BTNodeStatus.RUNNING:
                status = child.tick(delta_time)
            else:
                status = child.status

            if status == BTNodeStatus.SUCCESS:
                success_count += 1
            elif status == BTNodeStatus.FAILURE:
                failure_count += 1

        if success_count >= self._success_threshold:
            self._status = BTNodeStatus.SUCCESS
        elif failure_count >= self._failure_threshold:
            self._status = BTNodeStatus.FAILURE
        else:
            self._status = BTNodeStatus.RUNNING

        return self._status


class BTDecorator(BTNode):
    """Decorator: modifies child behavior."""

    def __init__(self, name: str = "", child: Optional[BTNode] = None) -> None:
        super().__init__(name)
        self._child: Optional[BTNode] = child

    def set_child(self, child: BTNode) -> None:
        """Set child node."""
        self._child = child
        if self._blackboard:
            child.set_blackboard(self._blackboard)

    def set_blackboard(self, blackboard: Blackboard) -> None:
        super().set_blackboard(blackboard)
        if self._child:
            self._child.set_blackboard(blackboard)


class BTInverter(BTDecorator):
    """Inverts child result."""

    def tick(self, delta_time: float) -> BTNodeStatus:
        if not self._child:
            return BTNodeStatus.FAILURE

        status = self._child.tick(delta_time)

        if status == BTNodeStatus.SUCCESS:
            self._status = BTNodeStatus.FAILURE
        elif status == BTNodeStatus.FAILURE:
            self._status = BTNodeStatus.SUCCESS
        else:
            self._status = BTNodeStatus.RUNNING

        return self._status


class BTRepeater(BTDecorator):
    """Repeats child execution."""

    def __init__(
        self,
        name: str = "",
        child: Optional[BTNode] = None,
        repeat_count: int = -1,  # -1 = infinite
    ) -> None:
        super().__init__(name, child)
        self._repeat_count = repeat_count
        self._current_count = 0

    def tick(self, delta_time: float) -> BTNodeStatus:
        if not self._child:
            return BTNodeStatus.FAILURE

        status = self._child.tick(delta_time)

        if status == BTNodeStatus.RUNNING:
            self._status = BTNodeStatus.RUNNING
            return self._status

        self._current_count += 1
        self._child.reset()

        if self._repeat_count > 0 and self._current_count >= self._repeat_count:
            self._status = BTNodeStatus.SUCCESS
        else:
            self._status = BTNodeStatus.RUNNING

        return self._status

    def reset(self) -> None:
        super().reset()
        self._current_count = 0
        if self._child:
            self._child.reset()


class BTCooldown(BTDecorator):
    """Prevents child execution during cooldown."""

    def __init__(
        self,
        name: str = "",
        child: Optional[BTNode] = None,
        cooldown_time: float = 1.0,
    ) -> None:
        super().__init__(name, child)
        self._cooldown_time = cooldown_time
        self._time_remaining: float = 0.0

    def tick(self, delta_time: float) -> BTNodeStatus:
        if self._time_remaining > 0:
            self._time_remaining -= delta_time
            self._status = BTNodeStatus.FAILURE
            return self._status

        if not self._child:
            return BTNodeStatus.FAILURE

        status = self._child.tick(delta_time)

        if status != BTNodeStatus.RUNNING:
            self._time_remaining = self._cooldown_time

        self._status = status
        return self._status


class BTAction(BTNode):
    """Leaf node that performs an action."""

    def __init__(
        self,
        name: str = "",
        action: Optional[Callable[[float], BTNodeStatus]] = None,
    ) -> None:
        super().__init__(name)
        self._action = action

    def tick(self, delta_time: float) -> BTNodeStatus:
        if self._action:
            self._status = self._action(delta_time)
        else:
            self._status = BTNodeStatus.SUCCESS
        return self._status


class BTCondition(BTNode):
    """Leaf node that checks a condition."""

    def __init__(
        self,
        name: str = "",
        condition: Optional[Callable[[], bool]] = None,
    ) -> None:
        super().__init__(name)
        self._condition = condition

    def tick(self, delta_time: float) -> BTNodeStatus:
        if self._condition and self._condition():
            self._status = BTNodeStatus.SUCCESS
        else:
            self._status = BTNodeStatus.FAILURE
        return self._status


class BehaviorTree:
    """Complete behavior tree with root node and blackboard."""

    def __init__(
        self,
        tree_id: str,
        root: Optional[BTNode] = None,
    ) -> None:
        self._tree_id = tree_id
        self._root = root
        self._blackboard = Blackboard()
        self._is_running = False

        if root:
            root.set_blackboard(self._blackboard)

    @property
    def tree_id(self) -> str:
        return self._tree_id

    @property
    def blackboard(self) -> Blackboard:
        return self._blackboard

    @property
    def is_running(self) -> bool:
        return self._is_running

    def set_root(self, root: BTNode) -> None:
        """Set root node."""
        self._root = root
        root.set_blackboard(self._blackboard)

    def tick(self, delta_time: float) -> BTNodeStatus:
        """Execute one tick of the behavior tree."""
        if not self._root:
            return BTNodeStatus.FAILURE

        self._is_running = True
        status = self._root.tick(delta_time)

        if status != BTNodeStatus.RUNNING:
            self._root.reset()

        return status

    def abort(self) -> None:
        """Abort tree execution."""
        if self._root:
            self._root.abort()
        self._is_running = False

    def reset(self) -> None:
        """Reset tree state."""
        if self._root:
            self._root.reset()
        self._is_running = False


# === Utility AI ===

@dataclass
class ConsiderationCurve:
    """Response curve for utility consideration."""

    curve_type: UtilityCurveType = UtilityCurveType.LINEAR
    slope: float = 1.0
    exponent: float = 2.0
    x_shift: float = 0.0
    y_shift: float = 0.0

    def evaluate(self, x: float) -> float:
        """Evaluate curve at x."""
        x = max(0.0, min(1.0, x + self.x_shift))

        if self.curve_type == UtilityCurveType.LINEAR:
            y = self.slope * x
        elif self.curve_type == UtilityCurveType.QUADRATIC:
            y = self.slope * (x ** self.exponent)
        elif self.curve_type == UtilityCurveType.LOGISTIC:
            import math
            # Logistic sigmoid curve using module-level constants
            y = 1.0 / (1.0 + math.exp(-self.slope * (x - UTILITY_LOGISTIC_CENTER) * UTILITY_LOGISTIC_STEEPNESS))
        elif self.curve_type == UtilityCurveType.EXPONENTIAL:
            import math
            y = (math.exp(self.slope * x) - 1) / (math.exp(self.slope) - 1)
        else:
            y = x

        return max(UTILITY_SCORE_MIN, min(UTILITY_SCORE_MAX, y + self.y_shift))


class Consideration:
    """Single consideration for utility evaluation."""

    def __init__(
        self,
        name: str,
        input_func: Callable[[], float],
        curve: Optional[ConsiderationCurve] = None,
    ) -> None:
        self._name = name
        self._input_func = input_func
        self._curve = curve or ConsiderationCurve()

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self) -> float:
        """Evaluate consideration score."""
        input_value = self._input_func()
        return self._curve.evaluate(input_value)


class UtilityAction:
    """Action with utility considerations."""

    def __init__(
        self,
        name: str,
        action: Callable[[], None],
        considerations: Optional[List[Consideration]] = None,
    ) -> None:
        self._name = name
        self._action = action
        self._considerations: List[Consideration] = considerations or []
        self._weight: float = 1.0

    @property
    def name(self) -> str:
        return self._name

    def add_consideration(self, consideration: Consideration) -> None:
        """Add consideration to action."""
        self._considerations.append(consideration)

    def calculate_utility(self) -> float:
        """Calculate combined utility score."""
        if not self._considerations:
            return 0.0

        # Use geometric mean for consideration combination
        score = 1.0
        for consideration in self._considerations:
            score *= consideration.evaluate()

        # Apply compensation factor for number of considerations
        n = len(self._considerations)
        modification = 1.0 - (1.0 / n)
        make_up = (1.0 - score) * modification
        final_score = score + (make_up * score)

        return final_score * self._weight

    def execute(self) -> None:
        """Execute the action."""
        self._action()


class UtilityAI:
    """Utility-based AI decision system."""

    def __init__(
        self,
        utility_id: str,
        update_rate: float = AI_UPDATE_RATE_DEFAULT,
    ) -> None:
        self._utility_id = utility_id
        self._update_rate = update_rate
        self._actions: List[UtilityAction] = []
        self._time_since_update: float = 0.0
        self._current_action: Optional[UtilityAction] = None

    @property
    def utility_id(self) -> str:
        return self._utility_id

    @property
    def current_action(self) -> Optional[UtilityAction]:
        return self._current_action

    def add_action(self, action: UtilityAction) -> None:
        """Add action to AI."""
        self._actions.append(action)

    def select_action(self) -> Optional[UtilityAction]:
        """Select highest utility action."""
        if not self._actions:
            return None

        best_action = None
        best_score = -1.0

        for action in self._actions:
            score = action.calculate_utility()
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def tick(self, delta_time: float) -> None:
        """Update utility AI."""
        self._time_since_update += delta_time

        if self._time_since_update >= self._update_rate:
            self._time_since_update = 0.0
            self._current_action = self.select_action()

            if self._current_action:
                self._current_action.execute()


# === GOAP (Goal-Oriented Action Planning) ===

@dataclass
class WorldState:
    """World state representation for GOAP."""

    facts: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def copy(self) -> WorldState:
        return WorldState(facts=dict(self.facts))

    def satisfies(self, goal: WorldState) -> bool:
        """Check if this state satisfies goal state."""
        for key, value in goal.facts.items():
            if self.facts.get(key) != value:
                return False
        return True

    def difference(self, other: WorldState) -> int:
        """Count differences between states."""
        diff = 0
        for key, value in other.facts.items():
            if self.facts.get(key) != value:
                diff += 1
        return diff


@dataclass
class GOAPAction:
    """Action for GOAP planning."""

    name: str
    cost: float = 1.0
    preconditions: WorldState = field(default_factory=WorldState)
    effects: WorldState = field(default_factory=WorldState)
    action_func: Optional[Callable[[], bool]] = None

    def can_execute(self, state: WorldState) -> bool:
        """Check if action can be executed in given state."""
        return state.satisfies(self.preconditions)

    def apply(self, state: WorldState) -> WorldState:
        """Apply action effects to state."""
        new_state = state.copy()
        for key, value in self.effects.facts.items():
            new_state.set(key, value)
        return new_state

    def execute(self) -> bool:
        """Execute the action."""
        if self.action_func:
            return self.action_func()
        return True


@dataclass
class GOAPNode:
    """Node in GOAP search graph."""

    state: WorldState
    action: Optional[GOAPAction]
    parent: Optional[GOAPNode]
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost

    def __lt__(self, other: GOAPNode) -> bool:
        return self.f_cost < other.f_cost


class GOAPPlanner:
    """Goal-Oriented Action Planner using A*."""

    def __init__(self) -> None:
        self._actions: List[GOAPAction] = []

    def add_action(self, action: GOAPAction) -> None:
        """Add available action."""
        self._actions.append(action)

    def plan(
        self,
        current_state: WorldState,
        goal_state: WorldState,
    ) -> Optional[List[GOAPAction]]:
        """Find action sequence from current to goal state."""
        if current_state.satisfies(goal_state):
            return []

        open_set: List[GOAPNode] = []
        closed_set: Set[str] = set()

        # Start node
        start_node = GOAPNode(
            state=current_state,
            action=None,
            parent=None,
            g_cost=0.0,
            h_cost=float(current_state.difference(goal_state)),
        )
        heappush(open_set, start_node)

        iterations = 0
        while open_set and iterations < GOAP_MAX_ITERATIONS:
            iterations += 1

            current = heappop(open_set)
            state_key = str(sorted(current.state.facts.items()))

            if state_key in closed_set:
                continue
            closed_set.add(state_key)

            # Check if goal reached
            if current.state.satisfies(goal_state):
                return self._reconstruct_plan(current)

            # Check depth limit
            if current.g_cost > GOAP_MAX_PLAN_DEPTH:
                continue

            # Expand node
            for action in self._actions:
                if action.can_execute(current.state):
                    new_state = action.apply(current.state)
                    new_state_key = str(sorted(new_state.facts.items()))

                    if new_state_key not in closed_set:
                        neighbor = GOAPNode(
                            state=new_state,
                            action=action,
                            parent=current,
                            g_cost=current.g_cost + action.cost,
                            h_cost=float(new_state.difference(goal_state)),
                        )
                        heappush(open_set, neighbor)

        return None  # No plan found

    def _reconstruct_plan(self, node: GOAPNode) -> List[GOAPAction]:
        """Reconstruct action sequence from goal node."""
        plan = []
        current = node
        while current.parent:
            if current.action:
                plan.append(current.action)
            current = current.parent
        plan.reverse()
        return plan


class GOAP:
    """Complete GOAP system."""

    def __init__(self, goap_id: str) -> None:
        self._goap_id = goap_id
        self._planner = GOAPPlanner()
        self._current_state = WorldState()
        self._current_goal: Optional[WorldState] = None
        self._current_plan: List[GOAPAction] = []
        self._current_action_index: int = 0

    @property
    def goap_id(self) -> str:
        return self._goap_id

    @property
    def has_plan(self) -> bool:
        return len(self._current_plan) > 0

    def add_action(self, action: GOAPAction) -> None:
        """Add available action."""
        self._planner.add_action(action)

    def set_state(self, key: str, value: Any) -> None:
        """Set current world state."""
        self._current_state.set(key, value)

    def set_goal(self, goal: WorldState) -> bool:
        """Set goal and create plan."""
        self._current_goal = goal
        plan = self._planner.plan(self._current_state, goal)

        if plan is not None:
            self._current_plan = plan
            self._current_action_index = 0
            return True

        self._current_plan = []
        return False

    def tick(self) -> bool:
        """Execute current action. Returns True if plan complete."""
        if self._current_action_index >= len(self._current_plan):
            return True

        action = self._current_plan[self._current_action_index]
        success = action.execute()

        if success:
            # Apply effects
            for key, value in action.effects.facts.items():
                self._current_state.set(key, value)
            self._current_action_index += 1
        else:
            # Replan
            if self._current_goal:
                self.set_goal(self._current_goal)

        return self._current_action_index >= len(self._current_plan)


# === Perception System ===

@dataclass
class Stimulus:
    """Perceived stimulus from the world."""

    source: Optional[Actor]
    sense: PerceptionSense
    position: Tuple[float, float, float]
    strength: float = 1.0
    timestamp: float = 0.0
    age: float = 0.0


class PerceptionComponent:
    """Perception configuration for an AI agent."""

    def __init__(
        self,
        sight_range: float = PERCEPTION_DEFAULT_SIGHT_RANGE,
        hearing_range: float = PERCEPTION_DEFAULT_HEARING_RANGE,
        fov: float = PERCEPTION_DEFAULT_FOV,
    ) -> None:
        self._sight_range = sight_range
        self._hearing_range = hearing_range
        self._fov = fov
        self._senses: Set[PerceptionSense] = {
            PerceptionSense.SIGHT,
            PerceptionSense.HEARING,
        }

    @property
    def sight_range(self) -> float:
        return self._sight_range

    @property
    def hearing_range(self) -> float:
        return self._hearing_range

    @property
    def fov(self) -> float:
        return self._fov

    def has_sense(self, sense: PerceptionSense) -> bool:
        return sense in self._senses

    def add_sense(self, sense: PerceptionSense) -> None:
        self._senses.add(sense)

    def remove_sense(self, sense: PerceptionSense) -> None:
        self._senses.discard(sense)


class Perception:
    """AI perception system for detecting stimuli."""

    # Known targets persist longer in memory than raw stimuli
    KNOWN_TARGET_PERSISTENCE_MULTIPLIER = 3

    def __init__(self, config: Optional[PerceptionComponent] = None) -> None:
        self._config = config or PerceptionComponent()
        self._stimuli: List[Stimulus] = []
        self._known_targets: Dict[int, Stimulus] = {}  # actor_id -> last known
        self._decay_rate: float = 1.0  # seconds until stimulus forgotten

    @property
    def config(self) -> PerceptionComponent:
        return self._config

    @property
    def stimuli(self) -> List[Stimulus]:
        return list(self._stimuli)

    @property
    def known_targets(self) -> Dict[int, Stimulus]:
        return dict(self._known_targets)

    def add_stimulus(self, stimulus: Stimulus) -> None:
        """Add perceived stimulus."""
        self._stimuli.append(stimulus)
        if stimulus.source:
            actor_id = stimulus.source.actor_id
            self._known_targets[actor_id] = stimulus

    def update(self, delta_time: float) -> None:
        """Update perception, aging stimuli."""
        # Age stimuli
        for stimulus in self._stimuli:
            stimulus.age += delta_time

        # Remove old stimuli
        self._stimuli = [s for s in self._stimuli if s.age < self._decay_rate]

        # Age known targets
        expired_targets = []
        for actor_id, stimulus in self._known_targets.items():
            stimulus.age += delta_time
            # Known targets persist longer in memory
            if stimulus.age >= self._decay_rate * self.KNOWN_TARGET_PERSISTENCE_MULTIPLIER:
                expired_targets.append(actor_id)

        for actor_id in expired_targets:
            del self._known_targets[actor_id]

    def clear(self) -> None:
        """Clear all perception data."""
        self._stimuli.clear()
        self._known_targets.clear()

    def get_nearest_target(self, position: Tuple[float, float, float]) -> Optional[Stimulus]:
        """Get nearest known target."""
        if not self._known_targets:
            return None

        def distance_squared(stim: Stimulus) -> float:
            dx = stim.position[0] - position[0]
            dy = stim.position[1] - position[1]
            dz = stim.position[2] - position[2]
            return dx * dx + dy * dy + dz * dz

        return min(self._known_targets.values(), key=distance_squared)


# === Knowledge System ===

class Knowledge:
    """AI knowledge representation combining blackboard and world facts."""

    def __init__(self) -> None:
        self._blackboard = Blackboard()
        self._facts = WorldState()
        self._beliefs: Dict[str, float] = {}  # Confidence in facts

    @property
    def blackboard(self) -> Blackboard:
        return self._blackboard

    @property
    def facts(self) -> WorldState:
        return self._facts

    def set_fact(self, key: str, value: Any, confidence: float = 1.0) -> None:
        """Set a fact with confidence level."""
        self._facts.set(key, value)
        self._beliefs[key] = max(0.0, min(1.0, confidence))

    def get_fact(self, key: str, default: Any = None) -> Any:
        """Get a fact value."""
        return self._facts.get(key, default)

    def get_confidence(self, key: str) -> float:
        """Get confidence in a fact."""
        return self._beliefs.get(key, 0.0)

    def has_fact(self, key: str, min_confidence: float = 0.0) -> bool:
        """Check if fact exists with minimum confidence."""
        return key in self._facts.facts and self.get_confidence(key) >= min_confidence

    def decay_beliefs(self, amount: float) -> None:
        """Decay confidence in all beliefs."""
        for key in self._beliefs:
            self._beliefs[key] = max(0.0, self._beliefs[key] - amount)

    def forget_weak_beliefs(self, threshold: float = 0.1) -> None:
        """Remove facts with confidence below threshold."""
        to_remove = [k for k, v in self._beliefs.items() if v < threshold]
        for key in to_remove:
            if key in self._facts.facts:
                del self._facts.facts[key]
            del self._beliefs[key]


# === Combat AI ===

class CombatBehavior(IntEnum):
    """Combat AI behavior types."""
    ATTACK = auto()
    DEFEND = auto()
    FLANK = auto()
    RETREAT = auto()
    SUPPORT = auto()
    PATROL = auto()


@dataclass
class ThreatAssessment:
    """Assessment of a potential threat."""

    target: Actor
    threat_level: float = 0.0
    distance: float = 0.0
    is_visible: bool = False
    last_seen_position: Optional[Tuple[float, float, float]] = None
    last_seen_time: float = 0.0


class CombatAI:
    """Combat-focused AI decision making."""

    # Combat AI default configuration
    DEFAULT_AGGRESSION = 0.5  # 0 = defensive, 1 = aggressive
    DEFAULT_HEALTH_THRESHOLD = 0.3  # Retreat below this
    DEFAULT_ATTACK_RANGE = 10.0
    DEFAULT_RETREAT_DISTANCE = 20.0
    # Aggression thresholds for behavior selection
    AGGRESSION_ATTACK_THRESHOLD = 0.5  # Above this, attack when in range
    AGGRESSION_FLANK_THRESHOLD = 0.7   # Above this, flank when out of range

    def __init__(self) -> None:
        self._current_behavior: CombatBehavior = CombatBehavior.PATROL
        self._current_target: Optional[Actor] = None
        self._threats: Dict[int, ThreatAssessment] = {}
        self._aggression: float = self.DEFAULT_AGGRESSION
        self._health_threshold: float = self.DEFAULT_HEALTH_THRESHOLD
        self._attack_range: float = self.DEFAULT_ATTACK_RANGE
        self._retreat_distance: float = self.DEFAULT_RETREAT_DISTANCE

    @property
    def current_behavior(self) -> CombatBehavior:
        return self._current_behavior

    @property
    def current_target(self) -> Optional[Actor]:
        return self._current_target

    @property
    def aggression(self) -> float:
        return self._aggression

    def add_threat(self, assessment: ThreatAssessment) -> None:
        """Add or update threat assessment."""
        self._threats[assessment.target.actor_id] = assessment

    def remove_threat(self, target: Actor) -> None:
        """Remove threat from tracking."""
        if target.actor_id in self._threats:
            del self._threats[target.actor_id]

    def get_highest_threat(self) -> Optional[ThreatAssessment]:
        """Get highest priority threat."""
        if not self._threats:
            return None
        return max(self._threats.values(), key=lambda t: t.threat_level)

    def select_behavior(self, health_percent: float) -> CombatBehavior:
        """Select appropriate combat behavior."""
        # Check for retreat
        if health_percent < self._health_threshold:
            self._current_behavior = CombatBehavior.RETREAT
            return self._current_behavior

        highest_threat = self.get_highest_threat()

        if not highest_threat:
            self._current_behavior = CombatBehavior.PATROL
            self._current_target = None
            return self._current_behavior

        self._current_target = highest_threat.target

        # Behavior based on aggression and situation
        if highest_threat.distance < self._attack_range:
            if self._aggression > self.AGGRESSION_ATTACK_THRESHOLD:
                self._current_behavior = CombatBehavior.ATTACK
            else:
                self._current_behavior = CombatBehavior.DEFEND
        else:
            if self._aggression > self.AGGRESSION_FLANK_THRESHOLD:
                self._current_behavior = CombatBehavior.FLANK
            else:
                self._current_behavior = CombatBehavior.ATTACK

        return self._current_behavior

    def clear_threats(self) -> None:
        """Clear all threat assessments."""
        self._threats.clear()
        self._current_target = None


__all__ = [
    # Blackboard
    "BlackboardKey",
    "Blackboard",
    # Behavior Tree
    "BTNode",
    "BTNodeStatus",
    "BTComposite",
    "BTSelector",
    "BTSequence",
    "BTParallel",
    "BTDecorator",
    "BTInverter",
    "BTRepeater",
    "BTCooldown",
    "BTAction",
    "BTCondition",
    "BehaviorTree",
    # Utility AI
    "ConsiderationCurve",
    "Consideration",
    "UtilityAction",
    "UtilityAI",
    # GOAP
    "WorldState",
    "GOAPAction",
    "GOAPNode",
    "GOAPPlanner",
    "GOAP",
    # Perception
    "Stimulus",
    "PerceptionComponent",
    "Perception",
    # Knowledge
    "Knowledge",
    # Combat AI
    "CombatBehavior",
    "ThreatAssessment",
    "CombatAI",
]
