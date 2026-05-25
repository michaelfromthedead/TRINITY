"""
Goal-Oriented Action Planning (GOAP) - A* based planning through action space.

Provides a GOAP implementation with:
- World state representation
- Goals with preconditions
- Actions with preconditions, effects, and costs
- A* search through action space
- Plan caching and validation
"""

from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from .constants import (
    GOAP_MAX_ITERATIONS,
    GOAP_MAX_PLAN_LENGTH,
    GOAP_DEFAULT_ACTION_COST,
    GOAP_HEURISTIC_WEIGHT,
    GOAP_PLAN_CACHE_SIZE,
    GOAP_PLAN_CACHE_TTL,
)

T = TypeVar("T")


# =============================================================================
# World State
# =============================================================================


class WorldState:
    """
    Represents the state of the world for GOAP planning.

    A world state is a dictionary of key-value pairs representing
    facts about the world.
    """

    def __init__(self, state: Optional[Dict[str, Any]] = None) -> None:
        self._state: Dict[str, Any] = dict(state) if state else {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the world state."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> "WorldState":
        """Set a value in the world state (returns new state)."""
        new_state = WorldState(self._state)
        new_state._state[key] = value
        return new_state

    def has(self, key: str) -> bool:
        """Check if a key exists in the world state."""
        return key in self._state

    def remove(self, key: str) -> "WorldState":
        """Remove a key from the world state (returns new state)."""
        new_state = WorldState(self._state)
        new_state._state.pop(key, None)
        return new_state

    def copy(self) -> "WorldState":
        """Create a copy of this world state."""
        return WorldState(self._state)

    def apply(self, effects: Dict[str, Any]) -> "WorldState":
        """Apply effects to create a new world state."""
        new_state = WorldState(self._state)
        new_state._state.update(effects)
        return new_state

    def satisfies(self, conditions: Dict[str, Any]) -> bool:
        """Check if this state satisfies the given conditions."""
        for key, expected_value in conditions.items():
            if key not in self._state:
                return False
            if self._state[key] != expected_value:
                return False
        return True

    def difference(self, target: "WorldState") -> Dict[str, Any]:
        """Get the differences between this state and a target state."""
        diff = {}
        for key, value in target._state.items():
            if key not in self._state or self._state[key] != value:
                diff[key] = value
        return diff

    def count_unsatisfied(self, conditions: Dict[str, Any]) -> int:
        """Count how many conditions are not satisfied."""
        count = 0
        for key, expected_value in conditions.items():
            if key not in self._state or self._state[key] != expected_value:
                count += 1
        return count

    def to_hashable(self) -> FrozenSet[Tuple[str, Any]]:
        """Convert to a hashable representation."""
        return frozenset(
            (k, v if not isinstance(v, (list, dict, set)) else str(v))
            for k, v in sorted(self._state.items())
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WorldState):
            return self._state == other._state
        return False

    def __hash__(self) -> int:
        return hash(self.to_hashable())

    def __repr__(self) -> str:
        return f"WorldState({self._state})"

    def __len__(self) -> int:
        return len(self._state)

    def items(self) -> List[Tuple[str, Any]]:
        """Get all key-value pairs."""
        return list(self._state.items())

    def keys(self) -> List[str]:
        """Get all keys."""
        return list(self._state.keys())


# =============================================================================
# Goals
# =============================================================================


@dataclass
class Goal:
    """
    A goal that the GOAP planner tries to achieve.

    A goal is defined by a set of conditions that must be true
    in the target world state.
    """
    name: str
    conditions: Dict[str, Any]
    priority: float = 1.0
    insistence: float = 0.0  # How strongly the agent wants this goal

    def is_satisfied(self, state: WorldState) -> bool:
        """Check if this goal is satisfied by the given state."""
        return state.satisfies(self.conditions)

    def get_unsatisfied_count(self, state: WorldState) -> int:
        """Get the number of unsatisfied conditions."""
        return state.count_unsatisfied(self.conditions)

    def __hash__(self) -> int:
        return hash((self.name, frozenset(self.conditions.items())))


# =============================================================================
# Actions
# =============================================================================


class GOAPAction(ABC):
    """
    An action that can be used in GOAP planning.

    Each action has:
    - Preconditions: conditions that must be true before executing
    - Effects: changes to the world state after execution
    - Cost: how expensive this action is
    """

    def __init__(
        self,
        name: str,
        preconditions: Optional[Dict[str, Any]] = None,
        effects: Optional[Dict[str, Any]] = None,
        cost: float = GOAP_DEFAULT_ACTION_COST,
    ) -> None:
        self.name = name
        self.preconditions: Dict[str, Any] = preconditions or {}
        self.effects: Dict[str, Any] = effects or {}
        self._base_cost = cost

    def get_cost(self, state: WorldState, context: Any = None) -> float:
        """
        Get the cost of this action.

        Can be overridden to provide dynamic costs based on state.
        """
        return self._base_cost

    def check_procedural_preconditions(
        self, state: WorldState, context: Any = None
    ) -> bool:
        """
        Check additional procedural preconditions.

        Override this to add runtime checks that can't be expressed
        as simple key-value conditions.
        """
        return True

    def can_execute(self, state: WorldState, context: Any = None) -> bool:
        """Check if this action can be executed in the given state."""
        if not state.satisfies(self.preconditions):
            return False
        return self.check_procedural_preconditions(state, context)

    def apply_effects(self, state: WorldState) -> WorldState:
        """Apply this action's effects to create a new state."""
        return state.apply(self.effects)

    @abstractmethod
    def execute(self, context: Any = None) -> bool:
        """
        Execute this action.

        Returns True if successful.
        """
        pass

    def on_start(self, context: Any = None) -> None:
        """Called when execution starts."""
        pass

    def on_end(self, context: Any = None) -> None:
        """Called when execution ends."""
        pass

    def __repr__(self) -> str:
        return f"GOAPAction({self.name})"

    def __hash__(self) -> int:
        return hash(self.name)


class FunctionGOAPAction(GOAPAction):
    """A GOAP action that executes a function."""

    def __init__(
        self,
        name: str,
        func: Callable[[Any], bool],
        preconditions: Optional[Dict[str, Any]] = None,
        effects: Optional[Dict[str, Any]] = None,
        cost: float = GOAP_DEFAULT_ACTION_COST,
    ) -> None:
        super().__init__(name, preconditions, effects, cost)
        self._func = func

    def execute(self, context: Any = None) -> bool:
        return self._func(context)


# =============================================================================
# Planner
# =============================================================================


@dataclass
class PlanNode:
    """A node in the planning search."""
    state: WorldState
    action: Optional[GOAPAction]
    parent: Optional["PlanNode"]
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    depth: int

    @property
    def f_cost(self) -> float:
        """Total cost (g + h)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: "PlanNode") -> bool:
        return self.f_cost < other.f_cost


@dataclass
class Plan:
    """A plan produced by the GOAP planner."""
    actions: List[GOAPAction]
    goal: Goal
    total_cost: float
    start_state: WorldState
    final_state: WorldState
    creation_time: float = field(default_factory=time.time)

    def is_valid(self, current_state: WorldState) -> bool:
        """Check if this plan is still valid for the given state."""
        # Check if start state matches
        state = current_state.copy()
        for action in self.actions:
            if not action.can_execute(state):
                return False
            state = action.apply_effects(state)
        return self.goal.is_satisfied(state)

    def is_expired(self, current_time: float, ttl: float = GOAP_PLAN_CACHE_TTL) -> bool:
        """Check if this plan has expired."""
        return (current_time - self.creation_time) > ttl

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)


class GOAPPlanner:
    """
    A* based GOAP planner.

    Searches through the space of actions to find a plan
    that achieves the given goal from the current state.
    """

    def __init__(
        self,
        actions: Optional[List[GOAPAction]] = None,
        max_iterations: int = GOAP_MAX_ITERATIONS,
        max_plan_length: int = GOAP_MAX_PLAN_LENGTH,
        heuristic_weight: float = GOAP_HEURISTIC_WEIGHT,
    ) -> None:
        self._actions: List[GOAPAction] = actions or []
        self.max_iterations = max_iterations
        self.max_plan_length = max_plan_length
        self.heuristic_weight = heuristic_weight
        self._plan_cache: Dict[Tuple[FrozenSet, str], Plan] = {}

    @property
    def actions(self) -> List[GOAPAction]:
        """Get all registered actions."""
        return self._actions

    def add_action(self, action: GOAPAction) -> "GOAPPlanner":
        """Add an action to the planner."""
        self._actions.append(action)
        return self

    def remove_action(self, action: GOAPAction) -> bool:
        """Remove an action from the planner."""
        if action in self._actions:
            self._actions.remove(action)
            return True
        return False

    def _heuristic(self, state: WorldState, goal: Goal) -> float:
        """
        Heuristic function for A* search.

        Estimates the cost to reach the goal from the given state.
        """
        return goal.get_unsatisfied_count(state) * self.heuristic_weight

    def _get_applicable_actions(
        self, state: WorldState, context: Any = None
    ) -> List[GOAPAction]:
        """Get all actions that can be executed in the given state."""
        return [
            action for action in self._actions
            if action.can_execute(state, context)
        ]

    def _reconstruct_plan(
        self,
        node: PlanNode,
        goal: Goal,
        start_state: WorldState,
    ) -> Plan:
        """Reconstruct the plan from a goal node."""
        actions: List[GOAPAction] = []
        current = node

        while current.action is not None:
            actions.append(current.action)
            if current.parent is not None:
                current = current.parent
            else:
                break

        actions.reverse()

        return Plan(
            actions=actions,
            goal=goal,
            total_cost=node.g_cost,
            start_state=start_state,
            final_state=node.state,
        )

    def plan(
        self,
        current_state: WorldState,
        goal: Goal,
        context: Any = None,
        use_cache: bool = True,
    ) -> Optional[Plan]:
        """
        Create a plan to achieve the goal from the current state.

        Returns a Plan if successful, None if no plan could be found.
        """
        # Check cache first
        if use_cache:
            cache_key = (current_state.to_hashable(), goal.name)
            if cache_key in self._plan_cache:
                cached_plan = self._plan_cache[cache_key]
                if (
                    not cached_plan.is_expired(time.time())
                    and cached_plan.is_valid(current_state)
                ):
                    return cached_plan

        # Check if goal is already satisfied
        if goal.is_satisfied(current_state):
            return Plan(
                actions=[],
                goal=goal,
                total_cost=0.0,
                start_state=current_state,
                final_state=current_state,
            )

        # Initialize search
        start_node = PlanNode(
            state=current_state,
            action=None,
            parent=None,
            g_cost=0.0,
            h_cost=self._heuristic(current_state, goal),
            depth=0,
        )

        open_set: List[PlanNode] = [start_node]
        closed_set: Set[FrozenSet] = set()
        iterations = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1

            # Get node with lowest f_cost
            current = heapq.heappop(open_set)
            current_state_hash = current.state.to_hashable()

            # Skip if already visited
            if current_state_hash in closed_set:
                continue

            closed_set.add(current_state_hash)

            # Check if goal is satisfied
            if goal.is_satisfied(current.state):
                plan = self._reconstruct_plan(current, goal, current_state)

                # Cache the plan
                if use_cache and len(self._plan_cache) < GOAP_PLAN_CACHE_SIZE:
                    cache_key = (current_state.to_hashable(), goal.name)
                    self._plan_cache[cache_key] = plan

                return plan

            # Check depth limit
            if current.depth >= self.max_plan_length:
                continue

            # Expand neighbors
            for action in self._get_applicable_actions(current.state, context):
                new_state = action.apply_effects(current.state)
                new_state_hash = new_state.to_hashable()

                if new_state_hash in closed_set:
                    continue

                g_cost = current.g_cost + action.get_cost(current.state, context)
                h_cost = self._heuristic(new_state, goal)

                neighbor = PlanNode(
                    state=new_state,
                    action=action,
                    parent=current,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    depth=current.depth + 1,
                )

                heapq.heappush(open_set, neighbor)

        return None

    def find_best_goal(
        self,
        current_state: WorldState,
        goals: List[Goal],
        context: Any = None,
    ) -> Tuple[Optional[Goal], Optional[Plan]]:
        """
        Find the best achievable goal and its plan.

        Goals are evaluated by priority * (1 / plan_cost).
        """
        best_goal: Optional[Goal] = None
        best_plan: Optional[Plan] = None
        best_score = -1.0

        for goal in goals:
            # Skip satisfied goals
            if goal.is_satisfied(current_state):
                continue

            plan = self.plan(current_state, goal, context)
            if plan is None:
                continue

            # Calculate goal score
            cost_factor = 1.0 / (plan.total_cost + 1.0)
            score = goal.priority * cost_factor + goal.insistence

            if score > best_score:
                best_score = score
                best_goal = goal
                best_plan = plan

        return best_goal, best_plan

    def clear_cache(self) -> None:
        """Clear the plan cache."""
        self._plan_cache.clear()


# =============================================================================
# GOAP Agent
# =============================================================================


@dataclass
class GOAPAgentState:
    """Current state of a GOAP agent."""
    current_goal: Optional[Goal] = None
    current_plan: Optional[Plan] = None
    current_action_index: int = 0
    is_executing: bool = False


class GOAPAgent:
    """
    An agent that uses GOAP for decision making.

    Manages goals, planning, and plan execution.
    """

    def __init__(
        self,
        planner: Optional[GOAPPlanner] = None,
        goals: Optional[List[Goal]] = None,
    ) -> None:
        self.planner = planner or GOAPPlanner()
        self._goals: List[Goal] = goals or []
        self._state = GOAPAgentState()
        self._world_state = WorldState()

    @property
    def goals(self) -> List[Goal]:
        """Get all registered goals."""
        return self._goals

    @property
    def world_state(self) -> WorldState:
        """Get the current world state."""
        return self._world_state

    @world_state.setter
    def world_state(self, state: WorldState) -> None:
        """Set the world state."""
        self._world_state = state

    @property
    def agent_state(self) -> GOAPAgentState:
        """Get the current agent state."""
        return self._state

    def add_goal(self, goal: Goal) -> "GOAPAgent":
        """Add a goal to the agent."""
        self._goals.append(goal)
        return self

    def remove_goal(self, goal: Goal) -> bool:
        """Remove a goal from the agent."""
        if goal in self._goals:
            self._goals.remove(goal)
            return True
        return False

    def set_goal_insistence(self, goal_name: str, insistence: float) -> bool:
        """Set the insistence of a goal by name."""
        for goal in self._goals:
            if goal.name == goal_name:
                goal.insistence = insistence
                return True
        return False

    def replan(self, context: Any = None) -> bool:
        """
        Create a new plan for the highest priority goal.

        Returns True if a plan was found.
        """
        goal, plan = self.planner.find_best_goal(
            self._world_state, self._goals, context
        )

        if goal is None or plan is None:
            self._state.current_goal = None
            self._state.current_plan = None
            self._state.current_action_index = 0
            self._state.is_executing = False
            return False

        self._state.current_goal = goal
        self._state.current_plan = plan
        self._state.current_action_index = 0
        self._state.is_executing = True

        return True

    def update(self, context: Any = None) -> bool:
        """
        Update the agent: execute the current plan step.

        Returns True if an action was executed.
        """
        # Check if we need to replan
        if self._state.current_plan is None:
            if not self.replan(context):
                return False

        plan = self._state.current_plan
        if plan is None:
            return False

        # Check if plan is still valid
        if not plan.is_valid(self._world_state):
            if not self.replan(context):
                return False
            plan = self._state.current_plan
            if plan is None:
                return False

        # Check if plan is complete
        if self._state.current_action_index >= len(plan.actions):
            self._state.is_executing = False
            return False

        # Execute current action
        action = plan.actions[self._state.current_action_index]

        if not action.can_execute(self._world_state, context):
            # Action no longer valid, replan
            if not self.replan(context):
                return False
            return self.update(context)

        action.on_start(context)
        success = action.execute(context)
        action.on_end(context)

        if success:
            # Apply effects to world state
            self._world_state = action.apply_effects(self._world_state)
            self._state.current_action_index += 1
        else:
            # Action failed, replan
            if not self.replan(context):
                return False

        return True

    def abort(self) -> None:
        """Abort the current plan."""
        self._state.current_goal = None
        self._state.current_plan = None
        self._state.current_action_index = 0
        self._state.is_executing = False

    def reset(self) -> None:
        """Reset the agent state."""
        self.abort()
        self._world_state = WorldState()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # World State
    "WorldState",
    # Goals
    "Goal",
    # Actions
    "GOAPAction",
    "FunctionGOAPAction",
    # Planner
    "PlanNode",
    "Plan",
    "GOAPPlanner",
    # Agent
    "GOAPAgentState",
    "GOAPAgent",
]
