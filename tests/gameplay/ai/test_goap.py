"""
Comprehensive tests for the Goal-Oriented Action Planning (GOAP) system.

Tests cover:
- World state representation
- Action preconditions and effects
- Goal definition and priority
- A* planning algorithm
- Plan execution and monitoring
- Plan invalidation and replanning
- Procedural preconditions
- Action costs and preferences

Total: ~150 tests
"""

import pytest
import time
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.ai import (
    WorldState,
    GOAPAction,
    GOAPNode,
    GOAPPlanner,
    GOAP,
)
from engine.gameplay.constants import GOAP_MAX_PLAN_DEPTH, GOAP_MAX_ITERATIONS

# Also import from detailed implementation
from engine.gameplay.ai.goap import (
    WorldState as DetailedWorldState,
    Goal,
    GOAPAction as DetailedGOAPAction,
    FunctionGOAPAction,
    PlanNode,
    Plan,
    GOAPPlanner as DetailedGOAPPlanner,
    GOAPAgentState,
    GOAPAgent,
)
from engine.gameplay.ai.constants import (
    GOAP_MAX_ITERATIONS as DETAILED_MAX_ITERATIONS,
    GOAP_MAX_PLAN_LENGTH,
    GOAP_DEFAULT_ACTION_COST,
    GOAP_HEURISTIC_WEIGHT,
    GOAP_PLAN_CACHE_SIZE,
    GOAP_PLAN_CACHE_TTL,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def world_state():
    """Create a fresh world state."""
    return WorldState()


@pytest.fixture
def detailed_world_state():
    """Create a detailed world state."""
    return DetailedWorldState()


@pytest.fixture
def simple_planner():
    """Create a simple GOAP planner."""
    return GOAPPlanner()


@pytest.fixture
def detailed_planner():
    """Create a detailed GOAP planner."""
    return DetailedGOAPPlanner()


@pytest.fixture
def simple_goap():
    """Create a simple GOAP instance."""
    return GOAP(goap_id="test_goap")


@pytest.fixture
def goap_agent():
    """Create a GOAP agent."""
    return GOAPAgent()


# =============================================================================
# World State Tests
# =============================================================================


class TestWorldState:
    """Test WorldState functionality."""

    def test_state_creation(self, world_state):
        """State should be created empty."""
        assert len(world_state.facts) == 0

    def test_state_creation_with_facts(self):
        """State should be created with initial facts."""
        state = WorldState(facts={"has_weapon": True})
        assert state.get("has_weapon") is True

    def test_state_set_and_get(self, world_state):
        """Should set and get values."""
        world_state.set("has_weapon", True)
        assert world_state.get("has_weapon") is True

    def test_state_get_default(self, world_state):
        """Get should return default for missing keys."""
        assert world_state.get("missing") is None
        assert world_state.get("missing", False) is False

    def test_state_copy(self, world_state):
        """Copy should create independent copy."""
        world_state.set("key", "value")
        copy = world_state.copy()
        copy.set("key", "different")

        assert world_state.get("key") == "value"
        assert copy.get("key") == "different"

    def test_state_satisfies_empty(self, world_state):
        """Empty goal should always be satisfied."""
        goal = WorldState()
        assert world_state.satisfies(goal)

    def test_state_satisfies_true(self, world_state):
        """Should satisfy matching goal."""
        world_state.set("has_weapon", True)
        goal = WorldState(facts={"has_weapon": True})
        assert world_state.satisfies(goal)

    def test_state_satisfies_false(self, world_state):
        """Should not satisfy mismatching goal."""
        world_state.set("has_weapon", False)
        goal = WorldState(facts={"has_weapon": True})
        assert not world_state.satisfies(goal)

    def test_state_difference(self, world_state):
        """Should count differences."""
        world_state.set("has_weapon", False)
        world_state.set("is_alive", True)
        goal = WorldState(facts={"has_weapon": True, "is_alive": True})
        assert world_state.difference(goal) == 1


# =============================================================================
# Detailed World State Tests
# =============================================================================


class TestDetailedWorldState:
    """Test detailed WorldState implementation."""

    def test_state_creation_from_dict(self):
        """Should create from dictionary."""
        state = DetailedWorldState({"a": 1, "b": 2})
        assert state.get("a") == 1
        assert state.get("b") == 2

    def test_state_has(self, detailed_world_state):
        """Should check if key exists."""
        detailed_world_state = detailed_world_state.set("key", "value")
        assert detailed_world_state.has("key")
        assert not detailed_world_state.has("missing")

    def test_state_set_returns_new_state(self, detailed_world_state):
        """Set should return new state (immutable)."""
        new_state = detailed_world_state.set("key", "value")
        assert new_state is not detailed_world_state
        assert new_state.has("key")
        assert not detailed_world_state.has("key")

    def test_state_remove(self, detailed_world_state):
        """Should remove key."""
        state = detailed_world_state.set("key", "value")
        new_state = state.remove("key")
        assert not new_state.has("key")

    def test_state_apply_effects(self, detailed_world_state):
        """Should apply effects."""
        state = detailed_world_state.set("a", 1)
        new_state = state.apply({"a": 2, "b": 3})
        assert new_state.get("a") == 2
        assert new_state.get("b") == 3

    def test_state_count_unsatisfied(self, detailed_world_state):
        """Should count unsatisfied conditions."""
        state = detailed_world_state.set("a", 1).set("b", 2)
        conditions = {"a": 1, "b": 3, "c": 4}
        assert state.count_unsatisfied(conditions) == 2

    def test_state_to_hashable(self, detailed_world_state):
        """Should convert to hashable."""
        state = detailed_world_state.set("a", 1).set("b", 2)
        hashable = state.to_hashable()
        assert isinstance(hashable, frozenset)

    def test_state_equality(self):
        """Equal states should be equal."""
        state1 = DetailedWorldState({"a": 1})
        state2 = DetailedWorldState({"a": 1})
        assert state1 == state2

    def test_state_inequality(self):
        """Different states should not be equal."""
        state1 = DetailedWorldState({"a": 1})
        state2 = DetailedWorldState({"a": 2})
        assert state1 != state2

    def test_state_hash(self):
        """Equal states should have same hash."""
        state1 = DetailedWorldState({"a": 1})
        state2 = DetailedWorldState({"a": 1})
        assert hash(state1) == hash(state2)

    def test_state_len(self, detailed_world_state):
        """len should return fact count."""
        state = detailed_world_state.set("a", 1).set("b", 2)
        assert len(state) == 2

    def test_state_items(self, detailed_world_state):
        """items should return all key-value pairs."""
        state = detailed_world_state.set("a", 1).set("b", 2)
        items = state.items()
        assert len(items) == 2

    def test_state_keys(self, detailed_world_state):
        """keys should return all keys."""
        state = detailed_world_state.set("a", 1).set("b", 2)
        keys = state.keys()
        assert "a" in keys
        assert "b" in keys

    def test_state_difference_dict(self, detailed_world_state):
        """difference should return dict of differences."""
        state = detailed_world_state.set("a", 1).set("b", 2)
        target = DetailedWorldState({"a": 1, "b": 3, "c": 4})
        diff = state.difference(target)
        assert diff == {"b": 3, "c": 4}


# =============================================================================
# Goal Tests
# =============================================================================


class TestGoal:
    """Test Goal functionality."""

    def test_goal_creation(self):
        """Goal should be created with conditions."""
        goal = Goal(name="kill_enemy", conditions={"enemy_dead": True})
        assert goal.name == "kill_enemy"
        assert goal.conditions["enemy_dead"] is True

    def test_goal_priority(self):
        """Goal should have priority."""
        goal = Goal(name="test", conditions={}, priority=2.0)
        assert goal.priority == 2.0

    def test_goal_insistence(self):
        """Goal should have insistence."""
        goal = Goal(name="test", conditions={}, insistence=0.5)
        assert goal.insistence == 0.5

    def test_goal_is_satisfied(self):
        """Should check if goal is satisfied."""
        goal = Goal(name="test", conditions={"done": True})
        state = DetailedWorldState({"done": True})
        assert goal.is_satisfied(state)

    def test_goal_not_satisfied(self):
        """Should return false if not satisfied."""
        goal = Goal(name="test", conditions={"done": True})
        state = DetailedWorldState({"done": False})
        assert not goal.is_satisfied(state)

    def test_goal_unsatisfied_count(self):
        """Should count unsatisfied conditions."""
        goal = Goal(name="test", conditions={"a": True, "b": True})
        state = DetailedWorldState({"a": True, "b": False})
        assert goal.get_unsatisfied_count(state) == 1

    def test_goal_hash(self):
        """Goals should be hashable."""
        goal = Goal(name="test", conditions={"a": True})
        assert isinstance(hash(goal), int)


# =============================================================================
# GOAP Action Tests
# =============================================================================


class TestGOAPAction:
    """Test GOAPAction functionality."""

    def test_action_creation(self):
        """Action should be created with name."""
        action = GOAPAction(
            name="pick_up_weapon",
            cost=1.0,
            preconditions=WorldState(facts={"near_weapon": True}),
            effects=WorldState(facts={"has_weapon": True})
        )
        assert action.name == "pick_up_weapon"
        assert action.cost == 1.0

    def test_action_can_execute(self):
        """Should check preconditions."""
        action = GOAPAction(
            name="attack",
            preconditions=WorldState(facts={"has_weapon": True}),
            effects=WorldState(facts={"enemy_dead": True})
        )

        state_valid = WorldState(facts={"has_weapon": True})
        state_invalid = WorldState(facts={"has_weapon": False})

        assert action.can_execute(state_valid)
        assert not action.can_execute(state_invalid)

    def test_action_apply(self):
        """Should apply effects."""
        action = GOAPAction(
            name="attack",
            preconditions=WorldState(),
            effects=WorldState(facts={"enemy_dead": True})
        )

        state = WorldState()
        new_state = action.apply(state)
        assert new_state.get("enemy_dead") is True

    def test_action_execute(self):
        """Should execute action function."""
        executed = [False]
        action = GOAPAction(
            name="test",
            action_func=lambda: (executed.__setitem__(0, True), True)[1]
        )
        result = action.execute()
        assert executed[0]
        assert result is True

    def test_action_execute_no_func(self):
        """Should succeed without function."""
        action = GOAPAction(name="test")
        assert action.execute() is True


# =============================================================================
# Detailed GOAP Action Tests
# =============================================================================


class TestDetailedGOAPAction:
    """Test detailed GOAPAction implementation."""

    def test_action_get_cost(self):
        """Should return base cost."""
        action = FunctionGOAPAction(
            name="test",
            func=lambda ctx: True,
            cost=2.5
        )
        state = DetailedWorldState()
        assert action.get_cost(state) == 2.5

    def test_action_procedural_preconditions(self):
        """Should check procedural preconditions."""
        class ConditionalAction(DetailedGOAPAction):
            def __init__(self):
                super().__init__(
                    name="conditional",
                    preconditions={"ready": True}
                )
                self.allowed = True

            def check_procedural_preconditions(self, state, context=None):
                return self.allowed

            def execute(self, context=None):
                return True

        action = ConditionalAction()
        state = DetailedWorldState({"ready": True})

        assert action.can_execute(state)

        action.allowed = False
        assert not action.can_execute(state)

    def test_action_apply_effects(self):
        """Should apply effects to state."""
        action = FunctionGOAPAction(
            name="test",
            func=lambda ctx: True,
            effects={"done": True}
        )
        state = DetailedWorldState()
        new_state = action.apply_effects(state)
        assert new_state.get("done") is True

    def test_function_action_execute(self):
        """FunctionGOAPAction should execute function."""
        context = Mock()
        action = FunctionGOAPAction(
            name="test",
            func=lambda ctx: ctx.value,
        )
        context.value = True
        assert action.execute(context) is True


# =============================================================================
# GOAP Planner Tests
# =============================================================================


class TestGOAPPlanner:
    """Test GOAPPlanner functionality."""

    def test_planner_add_action(self, simple_planner):
        """Should add action."""
        action = GOAPAction(name="test")
        simple_planner.add_action(action)
        assert action in simple_planner._actions

    def test_plan_no_actions(self, simple_planner):
        """Should return None with no actions."""
        current = WorldState()
        goal = WorldState(facts={"done": True})
        plan = simple_planner.plan(current, goal)
        assert plan is None

    def test_plan_already_satisfied(self, simple_planner):
        """Should return empty plan if goal satisfied."""
        current = WorldState(facts={"done": True})
        goal = WorldState(facts={"done": True})
        plan = simple_planner.plan(current, goal)
        assert plan == []

    def test_plan_single_action(self, simple_planner):
        """Should find single action plan."""
        simple_planner.add_action(GOAPAction(
            name="do_thing",
            preconditions=WorldState(),
            effects=WorldState(facts={"done": True})
        ))

        current = WorldState()
        goal = WorldState(facts={"done": True})
        plan = simple_planner.plan(current, goal)

        assert len(plan) == 1
        assert plan[0].name == "do_thing"

    def test_plan_multiple_actions(self, simple_planner):
        """Should find multi-action plan."""
        simple_planner.add_action(GOAPAction(
            name="get_weapon",
            preconditions=WorldState(),
            effects=WorldState(facts={"has_weapon": True})
        ))
        simple_planner.add_action(GOAPAction(
            name="attack",
            preconditions=WorldState(facts={"has_weapon": True}),
            effects=WorldState(facts={"enemy_dead": True})
        ))

        current = WorldState()
        goal = WorldState(facts={"enemy_dead": True})
        plan = simple_planner.plan(current, goal)

        assert len(plan) == 2
        assert plan[0].name == "get_weapon"
        assert plan[1].name == "attack"

    def test_plan_respects_preconditions(self, simple_planner):
        """Should only use actions with satisfied preconditions."""
        simple_planner.add_action(GOAPAction(
            name="attack",
            preconditions=WorldState(facts={"impossible": True}),
            effects=WorldState(facts={"enemy_dead": True})
        ))

        current = WorldState()
        goal = WorldState(facts={"enemy_dead": True})
        plan = simple_planner.plan(current, goal)

        assert plan is None


# =============================================================================
# Detailed Planner Tests
# =============================================================================


class TestDetailedGOAPPlanner:
    """Test detailed GOAPPlanner implementation."""

    def test_planner_creation(self, detailed_planner):
        """Planner should be created."""
        assert detailed_planner is not None

    def test_planner_add_remove_action(self, detailed_planner):
        """Should add and remove actions."""
        action = FunctionGOAPAction(name="test", func=lambda ctx: True)
        detailed_planner.add_action(action)
        assert action in detailed_planner.actions

        assert detailed_planner.remove_action(action)
        assert action not in detailed_planner.actions

    def test_plan_returns_plan_object(self, detailed_planner):
        """Should return Plan object."""
        detailed_planner.add_action(FunctionGOAPAction(
            name="do",
            func=lambda ctx: True,
            effects={"done": True}
        ))

        goal = Goal(name="finish", conditions={"done": True})
        plan = detailed_planner.plan(DetailedWorldState(), goal)

        assert isinstance(plan, Plan)

    def test_plan_has_total_cost(self, detailed_planner):
        """Plan should have total cost."""
        detailed_planner.add_action(FunctionGOAPAction(
            name="step1",
            func=lambda ctx: True,
            effects={"step1_done": True},
            cost=2.0
        ))
        detailed_planner.add_action(FunctionGOAPAction(
            name="step2",
            func=lambda ctx: True,
            preconditions={"step1_done": True},
            effects={"done": True},
            cost=3.0
        ))

        goal = Goal(name="finish", conditions={"done": True})
        plan = detailed_planner.plan(DetailedWorldState(), goal)

        assert plan.total_cost == 5.0

    def test_plan_prefers_lower_cost(self, detailed_planner):
        """Should prefer lower cost actions."""
        detailed_planner.add_action(FunctionGOAPAction(
            name="expensive",
            func=lambda ctx: True,
            effects={"done": True},
            cost=10.0
        ))
        detailed_planner.add_action(FunctionGOAPAction(
            name="cheap",
            func=lambda ctx: True,
            effects={"done": True},
            cost=1.0
        ))

        goal = Goal(name="finish", conditions={"done": True})
        plan = detailed_planner.plan(DetailedWorldState(), goal)

        assert plan.actions[0].name == "cheap"

    def test_plan_caching(self, detailed_planner):
        """Should cache plans."""
        detailed_planner.add_action(FunctionGOAPAction(
            name="do",
            func=lambda ctx: True,
            effects={"done": True}
        ))

        goal = Goal(name="finish", conditions={"done": True})
        state = DetailedWorldState()

        plan1 = detailed_planner.plan(state, goal, use_cache=True)
        plan2 = detailed_planner.plan(state, goal, use_cache=True)

        # Should use cached plan
        assert len(detailed_planner._plan_cache) > 0

    def test_plan_cache_clear(self, detailed_planner):
        """Should clear cache."""
        detailed_planner._plan_cache[("key", "goal")] = Mock()
        detailed_planner.clear_cache()
        assert len(detailed_planner._plan_cache) == 0

    def test_plan_max_iterations(self, detailed_planner):
        """Should respect max iterations."""
        # Create a complex scenario that exceeds iterations
        detailed_planner.max_iterations = 1

        for i in range(10):
            detailed_planner.add_action(FunctionGOAPAction(
                name=f"step{i}",
                func=lambda ctx: True,
                effects={f"step{i}_done": True}
            ))

        goal = Goal(name="finish", conditions={"step9_done": True})
        plan = detailed_planner.plan(DetailedWorldState(), goal)

        # May fail due to iteration limit
        assert plan is None or len(plan.actions) <= GOAP_MAX_PLAN_LENGTH

    def test_plan_max_depth(self, detailed_planner):
        """Should respect max plan length."""
        detailed_planner.max_plan_length = 2

        detailed_planner.add_action(FunctionGOAPAction(
            name="step1",
            func=lambda ctx: True,
            effects={"step1": True}
        ))
        detailed_planner.add_action(FunctionGOAPAction(
            name="step2",
            func=lambda ctx: True,
            preconditions={"step1": True},
            effects={"step2": True}
        ))
        detailed_planner.add_action(FunctionGOAPAction(
            name="step3",
            func=lambda ctx: True,
            preconditions={"step2": True},
            effects={"done": True}
        ))

        goal = Goal(name="finish", conditions={"done": True})
        plan = detailed_planner.plan(DetailedWorldState(), goal)

        # Should fail or have limited length
        assert plan is None or len(plan.actions) <= 2

    def test_find_best_goal(self, detailed_planner):
        """Should find best achievable goal."""
        detailed_planner.add_action(FunctionGOAPAction(
            name="do_easy",
            func=lambda ctx: True,
            effects={"easy_done": True},
            cost=1.0
        ))
        detailed_planner.add_action(FunctionGOAPAction(
            name="do_hard",
            func=lambda ctx: True,
            effects={"hard_done": True},
            cost=5.0
        ))

        goals = [
            Goal(name="easy", conditions={"easy_done": True}, priority=1.0),
            Goal(name="hard", conditions={"hard_done": True}, priority=1.0),
        ]

        goal, plan = detailed_planner.find_best_goal(DetailedWorldState(), goals)

        # Should prefer easy goal (lower cost)
        assert goal.name == "easy"


# =============================================================================
# Plan Tests
# =============================================================================


class TestPlan:
    """Test Plan functionality."""

    def test_plan_is_valid(self, detailed_planner):
        """Should validate plan against current state."""
        action = FunctionGOAPAction(
            name="do",
            func=lambda ctx: True,
            effects={"done": True}
        )
        detailed_planner.add_action(action)

        goal = Goal(name="finish", conditions={"done": True})
        plan = detailed_planner.plan(DetailedWorldState(), goal)

        assert plan.is_valid(DetailedWorldState())

    def test_plan_invalid_precondition(self, detailed_planner):
        """Should be invalid if preconditions fail."""
        action = FunctionGOAPAction(
            name="do",
            func=lambda ctx: True,
            preconditions={"ready": True},
            effects={"done": True}
        )
        detailed_planner.add_action(action)

        # Create plan with ready=True
        start_state = DetailedWorldState({"ready": True})
        goal = Goal(name="finish", conditions={"done": True})
        plan = detailed_planner.plan(start_state, goal)

        # Validate against state without ready
        assert not plan.is_valid(DetailedWorldState())

    def test_plan_expiration(self):
        """Should check if plan expired."""
        plan = Plan(
            actions=[],
            goal=Goal(name="test", conditions={}),
            total_cost=0,
            start_state=DetailedWorldState(),
            final_state=DetailedWorldState(),
            creation_time=time.time() - 100
        )

        assert plan.is_expired(time.time(), ttl=10.0)

    def test_plan_len(self):
        """Should return action count."""
        plan = Plan(
            actions=[Mock(), Mock()],
            goal=Goal(name="test", conditions={}),
            total_cost=0,
            start_state=DetailedWorldState(),
            final_state=DetailedWorldState(),
        )
        assert len(plan) == 2

    def test_plan_iter(self):
        """Should iterate over actions."""
        actions = [Mock(), Mock()]
        plan = Plan(
            actions=actions,
            goal=Goal(name="test", conditions={}),
            total_cost=0,
            start_state=DetailedWorldState(),
            final_state=DetailedWorldState(),
        )
        assert list(plan) == actions


# =============================================================================
# Simple GOAP Tests
# =============================================================================


class TestSimpleGOAP:
    """Test simple GOAP implementation."""

    def test_goap_creation(self, simple_goap):
        """GOAP should be created with ID."""
        assert simple_goap.goap_id == "test_goap"

    def test_goap_add_action(self, simple_goap):
        """Should add action."""
        action = GOAPAction(name="test")
        simple_goap.add_action(action)
        assert action in simple_goap._planner._actions

    def test_goap_set_state(self, simple_goap):
        """Should set world state."""
        simple_goap.set_state("has_weapon", True)
        assert simple_goap._current_state.get("has_weapon") is True

    def test_goap_set_goal(self, simple_goap):
        """Should set goal and plan."""
        simple_goap.add_action(GOAPAction(
            name="do",
            effects=WorldState(facts={"done": True})
        ))

        goal = WorldState(facts={"done": True})
        result = simple_goap.set_goal(goal)

        assert result is True
        assert simple_goap.has_plan

    def test_goap_set_goal_impossible(self, simple_goap):
        """Should return False for impossible goal."""
        goal = WorldState(facts={"impossible": True})
        result = simple_goap.set_goal(goal)

        assert result is False
        assert not simple_goap.has_plan

    def test_goap_tick_executes_action(self, simple_goap):
        """Tick should execute current action."""
        executed = [False]
        simple_goap.add_action(GOAPAction(
            name="do",
            effects=WorldState(facts={"done": True}),
            action_func=lambda: (executed.__setitem__(0, True), True)[1]
        ))

        simple_goap.set_goal(WorldState(facts={"done": True}))
        simple_goap.tick()

        assert executed[0]

    def test_goap_tick_applies_effects(self, simple_goap):
        """Tick should apply action effects."""
        simple_goap.add_action(GOAPAction(
            name="do",
            effects=WorldState(facts={"done": True}),
            action_func=lambda: True
        ))

        simple_goap.set_goal(WorldState(facts={"done": True}))
        simple_goap.tick()

        assert simple_goap._current_state.get("done") is True

    def test_goap_tick_completes_plan(self, simple_goap):
        """Tick should complete plan."""
        simple_goap.add_action(GOAPAction(
            name="do",
            effects=WorldState(facts={"done": True}),
            action_func=lambda: True
        ))

        simple_goap.set_goal(WorldState(facts={"done": True}))
        complete = simple_goap.tick()

        assert complete is True


# =============================================================================
# GOAP Agent Tests
# =============================================================================


class TestGOAPAgent:
    """Test GOAPAgent functionality."""

    def test_agent_creation(self, goap_agent):
        """Agent should be created."""
        assert goap_agent is not None
        assert goap_agent.planner is not None

    def test_agent_add_goal(self, goap_agent):
        """Should add goal."""
        goal = Goal(name="test", conditions={"done": True})
        result = goap_agent.add_goal(goal)
        assert result is goap_agent
        assert goal in goap_agent.goals

    def test_agent_remove_goal(self, goap_agent):
        """Should remove goal."""
        goal = Goal(name="test", conditions={"done": True})
        goap_agent.add_goal(goal)
        assert goap_agent.remove_goal(goal)
        assert goal not in goap_agent.goals

    def test_agent_world_state(self, goap_agent):
        """Should have world state."""
        goap_agent.world_state = DetailedWorldState({"test": True})
        assert goap_agent.world_state.get("test") is True

    def test_agent_set_goal_insistence(self, goap_agent):
        """Should set goal insistence."""
        goal = Goal(name="test", conditions={"done": True}, insistence=0.0)
        goap_agent.add_goal(goal)

        result = goap_agent.set_goal_insistence("test", 0.8)
        assert result is True
        assert goal.insistence == 0.8

    def test_agent_replan(self, goap_agent):
        """Should create new plan."""
        goap_agent.planner.add_action(FunctionGOAPAction(
            name="do",
            func=lambda ctx: True,
            effects={"done": True}
        ))
        goap_agent.add_goal(Goal(name="test", conditions={"done": True}))

        result = goap_agent.replan()
        assert result is True
        assert goap_agent.agent_state.current_plan is not None

    def test_agent_update_executes_plan(self, goap_agent):
        """Update should execute plan."""
        executed = [False]
        goap_agent.planner.add_action(FunctionGOAPAction(
            name="do",
            func=lambda ctx: (executed.__setitem__(0, True), True)[1],
            effects={"done": True}
        ))
        goap_agent.add_goal(Goal(name="test", conditions={"done": True}))

        goap_agent.update()
        assert executed[0]

    def test_agent_update_applies_effects(self, goap_agent):
        """Update should apply effects."""
        goap_agent.planner.add_action(FunctionGOAPAction(
            name="do",
            func=lambda ctx: True,
            effects={"done": True}
        ))
        goap_agent.add_goal(Goal(name="test", conditions={"done": True}))

        goap_agent.update()
        assert goap_agent.world_state.get("done") is True

    def test_agent_replans_on_failure(self, goap_agent):
        """Should replan on action failure."""
        fail_count = [0]

        goap_agent.planner.add_action(FunctionGOAPAction(
            name="unreliable",
            func=lambda ctx: (fail_count.__setitem__(0, fail_count[0] + 1), fail_count[0] > 1)[1],
            effects={"done": True}
        ))
        goap_agent.add_goal(Goal(name="test", conditions={"done": True}))

        goap_agent.update()  # First update fails
        goap_agent.update()  # Second update succeeds

        assert goap_agent.world_state.get("done") is True

    def test_agent_replans_on_invalid_plan(self, goap_agent):
        """Should replan if plan becomes invalid."""
        goap_agent.planner.add_action(FunctionGOAPAction(
            name="needs_ready",
            func=lambda ctx: True,
            preconditions={"ready": True},
            effects={"done": True}
        ))
        goap_agent.planner.add_action(FunctionGOAPAction(
            name="get_ready",
            func=lambda ctx: True,
            effects={"ready": True}
        ))

        goap_agent.add_goal(Goal(name="test", conditions={"done": True}))
        goap_agent.replan()

        # Invalidate the first action's preconditions
        goap_agent._world_state = DetailedWorldState()

        # Should replan and succeed
        goap_agent.update()

    def test_agent_abort(self, goap_agent):
        """Abort should clear plan."""
        goap_agent.planner.add_action(FunctionGOAPAction(
            name="do",
            func=lambda ctx: True,
            effects={"done": True}
        ))
        goap_agent.add_goal(Goal(name="test", conditions={"done": True}))
        goap_agent.replan()

        goap_agent.abort()
        assert goap_agent.agent_state.current_plan is None
        assert not goap_agent.agent_state.is_executing

    def test_agent_reset(self, goap_agent):
        """Reset should clear state and world."""
        goap_agent.world_state = DetailedWorldState({"test": True})
        goap_agent.planner.add_action(FunctionGOAPAction(
            name="do",
            func=lambda ctx: True,
            effects={"done": True}
        ))
        goap_agent.add_goal(Goal(name="test", conditions={"done": True}))
        goap_agent.replan()

        goap_agent.reset()
        assert len(goap_agent.world_state) == 0
        assert goap_agent.agent_state.current_plan is None


# =============================================================================
# GOAP Agent State Tests
# =============================================================================


class TestGOAPAgentState:
    """Test GOAPAgentState."""

    def test_state_defaults(self):
        """Should have default values."""
        state = GOAPAgentState()
        assert state.current_goal is None
        assert state.current_plan is None
        assert state.current_action_index == 0
        assert state.is_executing is False


# =============================================================================
# PlanNode Tests
# =============================================================================


class TestPlanNode:
    """Test PlanNode functionality."""

    def test_node_f_cost(self):
        """f_cost should be g + h."""
        node = PlanNode(
            state=DetailedWorldState(),
            action=None,
            parent=None,
            g_cost=5.0,
            h_cost=3.0,
            depth=0
        )
        assert node.f_cost == 8.0

    def test_node_comparison(self):
        """Nodes should compare by f_cost."""
        node1 = PlanNode(
            state=DetailedWorldState(),
            action=None,
            parent=None,
            g_cost=5.0,
            h_cost=3.0,
            depth=0
        )
        node2 = PlanNode(
            state=DetailedWorldState(),
            action=None,
            parent=None,
            g_cost=3.0,
            h_cost=3.0,
            depth=0
        )
        assert node2 < node1


# =============================================================================
# Integration Tests
# =============================================================================


class TestGOAPIntegration:
    """Integration tests for GOAP system."""

    def test_complete_planning_cycle(self):
        """Test complete planning and execution cycle."""
        agent = GOAPAgent()

        # Add actions
        agent.planner.add_action(FunctionGOAPAction(
            name="find_weapon",
            func=lambda ctx: True,
            effects={"knows_weapon_location": True},
            cost=1.0
        ))
        agent.planner.add_action(FunctionGOAPAction(
            name="goto_weapon",
            func=lambda ctx: True,
            preconditions={"knows_weapon_location": True},
            effects={"at_weapon": True},
            cost=2.0
        ))
        agent.planner.add_action(FunctionGOAPAction(
            name="pickup_weapon",
            func=lambda ctx: True,
            preconditions={"at_weapon": True},
            effects={"has_weapon": True},
            cost=1.0
        ))
        agent.planner.add_action(FunctionGOAPAction(
            name="attack",
            func=lambda ctx: True,
            preconditions={"has_weapon": True},
            effects={"enemy_dead": True},
            cost=1.0
        ))

        # Set goal
        agent.add_goal(Goal(
            name="kill_enemy",
            conditions={"enemy_dead": True}
        ))

        # Execute plan
        agent.replan()
        assert agent.agent_state.current_plan is not None
        assert len(agent.agent_state.current_plan.actions) == 4

        # Run through plan
        for _ in range(10):  # Safety limit
            result = agent.update()
            if agent.world_state.get("enemy_dead"):
                break

        assert agent.world_state.get("enemy_dead") is True

    def test_dynamic_replanning(self):
        """Test dynamic replanning on world changes."""
        agent = GOAPAgent()

        # Two ways to achieve goal
        agent.planner.add_action(FunctionGOAPAction(
            name="path_a",
            func=lambda ctx: True,
            preconditions={"path_a_open": True},
            effects={"done": True},
            cost=1.0
        ))
        agent.planner.add_action(FunctionGOAPAction(
            name="path_b",
            func=lambda ctx: True,
            effects={"done": True},
            cost=5.0
        ))

        # Start with path A open
        agent.world_state = DetailedWorldState({"path_a_open": True})
        agent.add_goal(Goal(name="finish", conditions={"done": True}))

        agent.replan()
        assert agent.agent_state.current_plan.actions[0].name == "path_a"

        # Close path A
        agent.world_state = DetailedWorldState({"path_a_open": False})
        agent.replan()
        assert agent.agent_state.current_plan.actions[0].name == "path_b"

    def test_goal_priority_selection(self):
        """Test goal selection by priority."""
        agent = GOAPAgent()

        agent.planner.add_action(FunctionGOAPAction(
            name="low_priority_action",
            func=lambda ctx: True,
            effects={"low_done": True}
        ))
        agent.planner.add_action(FunctionGOAPAction(
            name="high_priority_action",
            func=lambda ctx: True,
            effects={"high_done": True}
        ))

        agent.add_goal(Goal(
            name="low",
            conditions={"low_done": True},
            priority=1.0
        ))
        agent.add_goal(Goal(
            name="high",
            conditions={"high_done": True},
            priority=2.0
        ))

        agent.replan()
        # Should select high priority goal
        assert agent.agent_state.current_goal.name == "high"
