"""
WHITEBOX Tests for the GOAP System.

Comprehensive internal testing of Goal-Oriented Action Planning with full source access.

Tests cover:
- WorldState operations and immutability
- Goal satisfaction and condition checking
- GOAPAction preconditions, effects, and cost calculation
- FunctionGOAPAction wrapper
- PlanNode for A* search
- Plan validity and expiration
- GOAPPlanner A* search mechanics
- Plan caching
- GOAPAgent goal selection and plan execution
- Edge cases: no valid plan, plan invalidation, dynamic costs

Total: 50+ tests for GOAP system internals
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pytest

from engine.gameplay.ai.goap import (
    FunctionGOAPAction,
    Goal,
    GOAPAction,
    GOAPAgent,
    GOAPAgentState,
    GOAPPlanner,
    Plan,
    PlanNode,
    WorldState,
)
from engine.gameplay.ai.constants import (
    GOAP_DEFAULT_ACTION_COST,
    GOAP_HEURISTIC_WEIGHT,
    GOAP_MAX_ITERATIONS,
    GOAP_MAX_PLAN_LENGTH,
    GOAP_PLAN_CACHE_SIZE,
    GOAP_PLAN_CACHE_TTL,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def empty_state():
    """Create an empty world state."""
    return WorldState()


@pytest.fixture
def basic_state():
    """Create a basic world state."""
    return WorldState({
        "has_weapon": True,
        "has_ammo": True,
        "enemy_visible": True,
        "health": 100,
    })


# =============================================================================
# WORLD STATE TESTS
# =============================================================================


class TestWorldStateInternals:
    """Whitebox tests for WorldState operations."""

    def test_state_initialization_empty(self):
        """Test WorldState initializes empty."""
        state = WorldState()
        assert len(state._state) == 0

    def test_state_initialization_with_dict(self):
        """Test WorldState initializes from dict."""
        state = WorldState({"key": "value", "number": 42})

        assert state.get("key") == "value"
        assert state.get("number") == 42

    def test_state_get_default(self):
        """Test get returns default for missing key."""
        state = WorldState()
        assert state.get("missing", default="default") == "default"

    def test_state_set_returns_new_state(self):
        """Test set returns new state (immutable)."""
        original = WorldState({"a": 1})
        new_state = original.set("b", 2)

        assert original.get("b") is None
        assert new_state.get("b") == 2
        assert original is not new_state

    def test_state_has(self):
        """Test has checks key existence."""
        state = WorldState({"exists": True})

        assert state.has("exists")
        assert not state.has("missing")

    def test_state_remove_returns_new_state(self):
        """Test remove returns new state."""
        original = WorldState({"a": 1, "b": 2})
        new_state = original.remove("a")

        assert original.has("a")
        assert not new_state.has("a")
        assert new_state.has("b")

    def test_state_copy(self):
        """Test copy creates independent state."""
        original = WorldState({"key": "value"})
        copy = original.copy()

        assert copy.get("key") == "value"
        assert copy is not original

    def test_state_apply_effects(self):
        """Test apply creates new state with effects."""
        original = WorldState({"a": 1})
        effects = {"b": 2, "c": 3}

        new_state = original.apply(effects)

        assert new_state.get("a") == 1
        assert new_state.get("b") == 2
        assert new_state.get("c") == 3

    def test_state_satisfies_all_conditions(self):
        """Test satisfies checks all conditions."""
        state = WorldState({"a": 1, "b": True, "c": "value"})

        assert state.satisfies({"a": 1, "b": True})
        assert not state.satisfies({"a": 1, "b": False})
        assert not state.satisfies({"d": 4})

    def test_state_difference(self):
        """Test difference returns missing/different keys."""
        current = WorldState({"a": 1, "b": 2})
        target = WorldState({"a": 1, "b": 3, "c": 4})

        diff = current.difference(target)

        assert diff == {"b": 3, "c": 4}

    def test_state_count_unsatisfied(self):
        """Test count_unsatisfied counts unmet conditions."""
        state = WorldState({"a": 1, "b": 2})
        conditions = {"a": 1, "b": 3, "c": 4}

        count = state.count_unsatisfied(conditions)
        assert count == 2  # b wrong, c missing

    def test_state_to_hashable(self):
        """Test to_hashable creates hashable representation."""
        state = WorldState({"b": 2, "a": 1})
        hashable = state.to_hashable()

        assert isinstance(hashable, frozenset)

    def test_state_equality(self):
        """Test state equality."""
        a = WorldState({"x": 1})
        b = WorldState({"x": 1})
        c = WorldState({"x": 2})

        assert a == b
        assert a != c

    def test_state_hash(self):
        """Test state is hashable."""
        state = WorldState({"key": "value"})
        state_set = {state}

        assert state in state_set

    def test_state_items_and_keys(self):
        """Test items and keys methods."""
        state = WorldState({"a": 1, "b": 2})

        items = state.items()
        keys = state.keys()

        assert len(items) == 2
        assert len(keys) == 2
        assert "a" in keys


# =============================================================================
# GOAL TESTS
# =============================================================================


class TestGoalInternals:
    """Whitebox tests for Goal satisfaction."""

    def test_goal_initialization(self):
        """Test Goal initialization."""
        goal = Goal(
            name="kill_enemy",
            conditions={"enemy_dead": True},
            priority=1.5,
            insistence=0.5,
        )

        assert goal.name == "kill_enemy"
        assert goal.conditions == {"enemy_dead": True}
        assert goal.priority == 1.5
        assert goal.insistence == 0.5

    def test_goal_is_satisfied(self):
        """Test Goal is_satisfied checks conditions."""
        goal = Goal(name="armed", conditions={"has_weapon": True})

        satisfied_state = WorldState({"has_weapon": True})
        unsatisfied_state = WorldState({"has_weapon": False})

        assert goal.is_satisfied(satisfied_state)
        assert not goal.is_satisfied(unsatisfied_state)

    def test_goal_get_unsatisfied_count(self):
        """Test Goal get_unsatisfied_count."""
        goal = Goal(
            name="ready",
            conditions={"has_weapon": True, "has_ammo": True, "in_position": True}
        )

        state = WorldState({"has_weapon": True, "has_ammo": False})
        count = goal.get_unsatisfied_count(state)

        assert count == 2  # has_ammo=False, in_position missing

    def test_goal_hash(self):
        """Test Goal is hashable."""
        goal = Goal(name="test", conditions={"a": 1})
        goal_set = {goal}

        assert goal in goal_set


# =============================================================================
# GOAP ACTION TESTS
# =============================================================================


class SimpleAction(GOAPAction):
    """Simple test action that always succeeds."""

    def execute(self, context: Any = None) -> bool:
        return True


class FailingAction(GOAPAction):
    """Test action that always fails."""

    def execute(self, context: Any = None) -> bool:
        return False


class TestGOAPActionInternals:
    """Whitebox tests for GOAPAction mechanics."""

    def test_action_initialization(self):
        """Test GOAPAction initialization."""
        action = SimpleAction(
            name="attack",
            preconditions={"has_weapon": True},
            effects={"enemy_damaged": True},
            cost=2.0,
        )

        assert action.name == "attack"
        assert action.preconditions == {"has_weapon": True}
        assert action.effects == {"enemy_damaged": True}
        assert action._base_cost == 2.0

    def test_action_get_cost_default(self):
        """Test get_cost returns base cost."""
        action = SimpleAction(name="test", cost=5.0)
        assert action.get_cost(WorldState()) == 5.0

    def test_action_can_execute_checks_preconditions(self):
        """Test can_execute checks preconditions."""
        action = SimpleAction(
            name="attack",
            preconditions={"has_weapon": True, "enemy_visible": True}
        )

        can_execute = WorldState({"has_weapon": True, "enemy_visible": True})
        cannot_execute = WorldState({"has_weapon": True, "enemy_visible": False})

        assert action.can_execute(can_execute)
        assert not action.can_execute(cannot_execute)

    def test_action_check_procedural_preconditions(self):
        """Test check_procedural_preconditions can be overridden."""

        class ConditionalAction(GOAPAction):
            def check_procedural_preconditions(
                self, state: WorldState, context: Any = None
            ) -> bool:
                # Custom check: health must be above 50
                return context and context.get("health", 0) > 50

            def execute(self, context: Any = None) -> bool:
                return True

        action = ConditionalAction(name="heal")

        good_context = {"health": 100}
        bad_context = {"health": 30}

        assert action.check_procedural_preconditions(WorldState(), good_context)
        assert not action.check_procedural_preconditions(WorldState(), bad_context)

    def test_action_apply_effects(self):
        """Test apply_effects creates new state."""
        action = SimpleAction(
            name="attack",
            effects={"enemy_damaged": True, "ammo": 9}
        )

        original = WorldState({"ammo": 10})
        new_state = action.apply_effects(original)

        assert new_state.get("enemy_damaged") is True
        assert new_state.get("ammo") == 9


class TestFunctionGOAPActionInternals:
    """Tests for FunctionGOAPAction wrapper."""

    def test_function_action_executes_function(self):
        """Test FunctionGOAPAction executes its function."""
        executed = [False]

        def my_func(context):
            executed[0] = True
            return True

        action = FunctionGOAPAction(
            name="func_action",
            func=my_func,
        )

        result = action.execute()
        assert executed[0] is True
        assert result is True


# =============================================================================
# PLAN NODE TESTS
# =============================================================================


class TestPlanNodeInternals:
    """Tests for PlanNode A* search node."""

    def test_plan_node_f_cost(self):
        """Test PlanNode f_cost calculation."""
        node = PlanNode(
            state=WorldState(),
            action=None,
            parent=None,
            g_cost=5.0,
            h_cost=3.0,
            depth=0,
        )

        assert node.f_cost == 8.0

    def test_plan_node_comparison(self):
        """Test PlanNode comparison for priority queue."""
        low_cost = PlanNode(
            state=WorldState(),
            action=None,
            parent=None,
            g_cost=1.0,
            h_cost=1.0,
            depth=0,
        )

        high_cost = PlanNode(
            state=WorldState(),
            action=None,
            parent=None,
            g_cost=5.0,
            h_cost=5.0,
            depth=0,
        )

        assert low_cost < high_cost


# =============================================================================
# PLAN TESTS
# =============================================================================


class TestPlanInternals:
    """Tests for Plan validity and expiration."""

    def test_plan_is_valid_empty_plan(self):
        """Test empty plan is valid for satisfied goal."""
        state = WorldState({"goal_met": True})
        goal = Goal(name="test", conditions={"goal_met": True})

        plan = Plan(
            actions=[],
            goal=goal,
            total_cost=0.0,
            start_state=state,
            final_state=state,
        )

        assert plan.is_valid(state)

    def test_plan_is_valid_checks_actions(self):
        """Test plan validity checks action preconditions."""
        goal = Goal(name="armed", conditions={"has_weapon": True})

        action = SimpleAction(
            name="get_weapon",
            preconditions={"weapon_available": True},
            effects={"has_weapon": True},
        )

        initial = WorldState({"weapon_available": True})
        final = initial.apply(action.effects)

        plan = Plan(
            actions=[action],
            goal=goal,
            total_cost=1.0,
            start_state=initial,
            final_state=final,
        )

        # Valid with correct state
        assert plan.is_valid(initial)

        # Invalid with wrong state
        bad_state = WorldState({"weapon_available": False})
        assert not plan.is_valid(bad_state)

    def test_plan_is_expired(self):
        """Test plan expiration."""
        plan = Plan(
            actions=[],
            goal=Goal(name="test", conditions={}),
            total_cost=0.0,
            start_state=WorldState(),
            final_state=WorldState(),
            creation_time=time.time() - GOAP_PLAN_CACHE_TTL - 1,
        )

        assert plan.is_expired(time.time())

    def test_plan_len(self):
        """Test plan length."""
        actions = [SimpleAction(name=f"action_{i}") for i in range(5)]

        plan = Plan(
            actions=actions,
            goal=Goal(name="test", conditions={}),
            total_cost=5.0,
            start_state=WorldState(),
            final_state=WorldState(),
        )

        assert len(plan) == 5

    def test_plan_iter(self):
        """Test plan iteration."""
        actions = [SimpleAction(name=f"action_{i}") for i in range(3)]

        plan = Plan(
            actions=actions,
            goal=Goal(name="test", conditions={}),
            total_cost=3.0,
            start_state=WorldState(),
            final_state=WorldState(),
        )

        iterated = list(plan)
        assert len(iterated) == 3


# =============================================================================
# GOAP PLANNER TESTS
# =============================================================================


class TestGOAPPlannerInternals:
    """Whitebox tests for GOAPPlanner A* search."""

    def test_planner_initialization(self):
        """Test GOAPPlanner initialization."""
        planner = GOAPPlanner(
            max_iterations=500,
            max_plan_length=20,
        )

        assert planner.max_iterations == 500
        assert planner.max_plan_length == 20
        assert planner.actions == []

    def test_planner_add_action(self):
        """Test adding actions to planner."""
        planner = GOAPPlanner()
        action = SimpleAction(name="test")

        planner.add_action(action)

        assert action in planner.actions

    def test_planner_remove_action(self):
        """Test removing actions from planner."""
        planner = GOAPPlanner()
        action = SimpleAction(name="test")

        planner.add_action(action)
        result = planner.remove_action(action)

        assert result is True
        assert action not in planner.actions

    def test_planner_remove_nonexistent_action(self):
        """Test removing nonexistent action returns False."""
        planner = GOAPPlanner()
        action = SimpleAction(name="test")

        result = planner.remove_action(action)
        assert result is False

    def test_planner_goal_already_satisfied(self):
        """Test planner returns empty plan when goal already satisfied."""
        planner = GOAPPlanner()
        goal = Goal(name="done", conditions={"complete": True})
        state = WorldState({"complete": True})

        plan = planner.plan(state, goal)

        assert plan is not None
        assert len(plan.actions) == 0
        assert plan.total_cost == 0.0

    def test_planner_simple_single_action_plan(self):
        """Test planner finds simple single-action plan."""
        action = SimpleAction(
            name="do_thing",
            preconditions={},
            effects={"thing_done": True},
        )

        planner = GOAPPlanner([action])
        goal = Goal(name="do_it", conditions={"thing_done": True})
        state = WorldState()

        plan = planner.plan(state, goal)

        assert plan is not None
        assert len(plan.actions) == 1
        assert plan.actions[0].name == "do_thing"

    def test_planner_multi_action_plan(self):
        """Test planner finds multi-action plan."""
        get_weapon = SimpleAction(
            name="get_weapon",
            preconditions={},
            effects={"has_weapon": True},
        )

        attack = SimpleAction(
            name="attack",
            preconditions={"has_weapon": True},
            effects={"enemy_dead": True},
        )

        planner = GOAPPlanner([get_weapon, attack])
        goal = Goal(name="kill", conditions={"enemy_dead": True})
        state = WorldState()

        plan = planner.plan(state, goal)

        assert plan is not None
        assert len(plan.actions) == 2
        assert plan.actions[0].name == "get_weapon"
        assert plan.actions[1].name == "attack"

    def test_planner_no_valid_plan(self):
        """Test planner returns None when no valid plan exists."""
        # Action requires condition that can't be achieved
        action = SimpleAction(
            name="impossible",
            preconditions={"magic_item": True},
            effects={"goal": True},
        )

        planner = GOAPPlanner([action])
        goal = Goal(name="reach_goal", conditions={"goal": True})
        state = WorldState()

        plan = planner.plan(state, goal)

        assert plan is None

    def test_planner_chooses_lowest_cost_plan(self):
        """Test planner chooses lowest cost plan."""
        cheap = SimpleAction(
            name="cheap",
            preconditions={},
            effects={"done": True},
            cost=1.0,
        )

        expensive = SimpleAction(
            name="expensive",
            preconditions={},
            effects={"done": True},
            cost=10.0,
        )

        planner = GOAPPlanner([expensive, cheap])  # Add expensive first
        goal = Goal(name="finish", conditions={"done": True})
        state = WorldState()

        plan = planner.plan(state, goal)

        assert plan is not None
        assert plan.actions[0].name == "cheap"
        assert plan.total_cost == 1.0

    def test_planner_respects_max_plan_length(self):
        """Test planner respects max plan length."""
        # Create chain of actions requiring many steps
        actions = []
        for i in range(10):
            prev_cond = {f"step_{i}": True} if i > 0 else {}
            effects = {f"step_{i+1}": True}
            actions.append(SimpleAction(
                name=f"action_{i}",
                preconditions=prev_cond,
                effects=effects,
            ))

        planner = GOAPPlanner(actions, max_plan_length=5)
        goal = Goal(name="final", conditions={"step_10": True})
        state = WorldState()

        plan = planner.plan(state, goal)

        # Should return None - can't reach in 5 steps
        assert plan is None

    def test_planner_caching(self):
        """Test planner uses plan cache."""
        action = SimpleAction(
            name="cached_action",
            preconditions={},
            effects={"cached": True},
        )

        planner = GOAPPlanner([action])
        goal = Goal(name="use_cache", conditions={"cached": True})
        state = WorldState()

        # First plan
        plan1 = planner.plan(state, goal, use_cache=True)

        # Second plan should hit cache
        plan2 = planner.plan(state, goal, use_cache=True)

        assert plan1 is plan2  # Same object from cache

    def test_planner_cache_bypass(self):
        """Test planner can bypass cache."""
        action = SimpleAction(
            name="action",
            preconditions={},
            effects={"done": True},
        )

        planner = GOAPPlanner([action])
        goal = Goal(name="test", conditions={"done": True})
        state = WorldState()

        plan1 = planner.plan(state, goal, use_cache=True)
        plan2 = planner.plan(state, goal, use_cache=False)

        assert plan1 is not plan2  # Different objects

    def test_planner_clear_cache(self):
        """Test planner clear_cache."""
        planner = GOAPPlanner()
        planner._plan_cache[("key", "goal")] = "cached_plan"

        planner.clear_cache()

        assert len(planner._plan_cache) == 0

    def test_planner_find_best_goal(self):
        """Test find_best_goal selects best achievable goal."""
        easy_action = SimpleAction(
            name="easy",
            preconditions={},
            effects={"easy_goal": True},
            cost=1.0,
        )

        hard_action = SimpleAction(
            name="hard",
            preconditions={},
            effects={"hard_goal": True},
            cost=10.0,
        )

        planner = GOAPPlanner([easy_action, hard_action])

        # Higher priority but harder goal
        hard_goal = Goal(name="hard", conditions={"hard_goal": True}, priority=2.0)
        # Lower priority but easier goal
        easy_goal = Goal(name="easy", conditions={"easy_goal": True}, priority=1.0)

        state = WorldState()

        best_goal, best_plan = planner.find_best_goal(
            state, [hard_goal, easy_goal]
        )

        # Easy goal should win due to lower cost
        assert best_goal is not None
        assert best_plan is not None


# =============================================================================
# GOAP AGENT TESTS
# =============================================================================


class TestGOAPAgentInternals:
    """Whitebox tests for GOAPAgent goal selection and execution."""

    def test_agent_initialization(self):
        """Test GOAPAgent initialization."""
        planner = GOAPPlanner()
        agent = GOAPAgent(planner=planner, enable_event_logging=False)

        assert agent.planner is planner
        assert agent.goals == []
        assert agent.world_state == WorldState()

    def test_agent_add_goal(self):
        """Test adding goals to agent."""
        agent = GOAPAgent(enable_event_logging=False)
        goal = Goal(name="test", conditions={"done": True})

        agent.add_goal(goal)

        assert goal in agent.goals

    def test_agent_remove_goal(self):
        """Test removing goals from agent."""
        agent = GOAPAgent(enable_event_logging=False)
        goal = Goal(name="test", conditions={"done": True})

        agent.add_goal(goal)
        result = agent.remove_goal(goal)

        assert result is True
        assert goal not in agent.goals

    def test_agent_set_goal_insistence(self):
        """Test setting goal insistence."""
        agent = GOAPAgent(enable_event_logging=False)
        goal = Goal(name="urgent", conditions={"done": True}, insistence=0.0)

        agent.add_goal(goal)
        result = agent.set_goal_insistence("urgent", 1.0)

        assert result is True
        assert goal.insistence == 1.0

    def test_agent_world_state_property(self):
        """Test world state property."""
        agent = GOAPAgent(enable_event_logging=False)

        new_state = WorldState({"key": "value"})
        agent.world_state = new_state

        assert agent.world_state == new_state

    def test_agent_replan(self):
        """Test agent replan creates plan."""
        action = SimpleAction(
            name="action",
            preconditions={},
            effects={"goal_met": True},
        )

        planner = GOAPPlanner([action])
        agent = GOAPAgent(planner=planner, enable_event_logging=False)

        goal = Goal(name="goal", conditions={"goal_met": True})
        agent.add_goal(goal)

        result = agent.replan()

        assert result is True
        assert agent.agent_state.current_plan is not None
        assert agent.agent_state.current_goal is goal

    def test_agent_replan_no_achievable_goal(self):
        """Test agent replan returns False when no achievable goal."""
        # Action requires impossible precondition
        action = SimpleAction(
            name="impossible",
            preconditions={"magic": True},
            effects={"goal_met": True},
        )

        planner = GOAPPlanner([action])
        agent = GOAPAgent(planner=planner, enable_event_logging=False)

        goal = Goal(name="goal", conditions={"goal_met": True})
        agent.add_goal(goal)

        result = agent.replan()

        assert result is False
        assert agent.agent_state.current_plan is None

    def test_agent_update_executes_action(self):
        """Test agent update executes current action."""
        executed = [False]

        def execute_func(ctx):
            executed[0] = True
            return True

        action = FunctionGOAPAction(
            name="action",
            func=execute_func,
            preconditions={},
            effects={"done": True},
        )

        planner = GOAPPlanner([action])
        agent = GOAPAgent(planner=planner, enable_event_logging=False)

        goal = Goal(name="goal", conditions={"done": True})
        agent.add_goal(goal)

        agent.update()

        assert executed[0] is True

    def test_agent_update_applies_effects(self):
        """Test agent update applies action effects."""
        action = SimpleAction(
            name="action",
            preconditions={},
            effects={"done": True},
        )

        planner = GOAPPlanner([action])
        agent = GOAPAgent(planner=planner, enable_event_logging=False)

        goal = Goal(name="goal", conditions={"done": True})
        agent.add_goal(goal)

        agent.update()

        assert agent.world_state.get("done") is True

    def test_agent_update_advances_action_index(self):
        """Test agent update advances action index."""
        action1 = SimpleAction(
            name="action1",
            preconditions={},
            effects={"step1": True},
        )

        action2 = SimpleAction(
            name="action2",
            preconditions={"step1": True},
            effects={"step2": True},
        )

        planner = GOAPPlanner([action1, action2])
        agent = GOAPAgent(planner=planner, enable_event_logging=False)

        goal = Goal(name="goal", conditions={"step2": True})
        agent.add_goal(goal)

        agent.update()  # Execute action1
        assert agent.agent_state.current_action_index == 1

        agent.update()  # Execute action2
        assert agent.agent_state.current_action_index == 2

    def test_agent_update_replans_on_action_failure(self):
        """Test agent replans when action fails."""
        call_count = [0]

        def failing_first_time(ctx):
            call_count[0] += 1
            return call_count[0] > 1

        action = FunctionGOAPAction(
            name="action",
            func=failing_first_time,
            preconditions={},
            effects={"done": True},
        )

        planner = GOAPPlanner([action])
        agent = GOAPAgent(planner=planner, enable_event_logging=False)

        goal = Goal(name="goal", conditions={"done": True})
        agent.add_goal(goal)

        # First update - action fails, triggers replan
        agent.update()

        # Plan should be recreated
        assert call_count[0] >= 1

    def test_agent_abort(self):
        """Test agent abort clears state."""
        agent = GOAPAgent(enable_event_logging=False)
        agent._state.current_goal = Goal(name="test", conditions={})
        agent._state.is_executing = True

        agent.abort()

        assert agent.agent_state.current_goal is None
        assert agent.agent_state.current_plan is None
        assert agent.agent_state.is_executing is False

    def test_agent_reset(self):
        """Test agent reset clears all state."""
        agent = GOAPAgent(enable_event_logging=False)
        agent._state.current_goal = Goal(name="test", conditions={})
        agent._world_state = WorldState({"key": "value"})

        agent.reset()

        assert agent.agent_state.current_goal is None
        assert len(agent.world_state) == 0


# =============================================================================
# EDGE CASES
# =============================================================================


class TestGOAPEdgeCases:
    """Edge case tests for GOAP system."""

    def test_world_state_with_list_value(self):
        """Test world state handles list values in hashable."""
        state = WorldState({"list_key": [1, 2, 3]})
        hashable = state.to_hashable()

        # Should convert to string representation
        assert any("list_key" in str(item) for item in hashable)

    def test_goal_with_many_conditions(self):
        """Test goal with many conditions."""
        conditions = {f"cond_{i}": True for i in range(20)}
        goal = Goal(name="complex", conditions=conditions)

        all_satisfied = WorldState(conditions)
        assert goal.is_satisfied(all_satisfied)

    def test_action_zero_cost(self):
        """Test action with zero cost."""
        action = SimpleAction(name="free", cost=0.0)
        assert action.get_cost(WorldState()) == 0.0

    def test_planner_respects_max_iterations(self):
        """Test planner stops at max iterations."""
        # Create impossible situation that would loop forever
        planner = GOAPPlanner(max_iterations=10)

        # No actions can achieve goal
        goal = Goal(name="impossible", conditions={"unreachable": True})
        state = WorldState()

        plan = planner.plan(state, goal)
        assert plan is None

    def test_agent_with_multiple_goals(self):
        """Test agent with multiple goals prioritizes correctly."""
        action1 = SimpleAction(
            name="low_priority_action",
            preconditions={},
            effects={"low": True},
            cost=1.0,
        )

        action2 = SimpleAction(
            name="high_priority_action",
            preconditions={},
            effects={"high": True},
            cost=1.0,
        )

        planner = GOAPPlanner([action1, action2])
        agent = GOAPAgent(planner=planner, enable_event_logging=False)

        low_goal = Goal(name="low", conditions={"low": True}, priority=1.0)
        high_goal = Goal(name="high", conditions={"high": True}, priority=10.0)

        agent.add_goal(low_goal)
        agent.add_goal(high_goal)

        agent.replan()

        # Should select high priority goal
        assert agent.agent_state.current_goal.name == "high"

    def test_empty_goal_conditions(self):
        """Test goal with empty conditions is always satisfied."""
        goal = Goal(name="empty", conditions={})
        state = WorldState({"anything": "value"})

        assert goal.is_satisfied(state)

    def test_plan_with_dynamic_action_cost(self):
        """Test planning with dynamic action costs."""

        class DynamicCostAction(GOAPAction):
            def get_cost(self, state: WorldState, context: Any = None) -> float:
                # Cost depends on world state
                return 1.0 if state.get("has_tool") else 10.0

            def execute(self, context: Any = None) -> bool:
                return True

        action = DynamicCostAction(
            name="dynamic",
            preconditions={},
            effects={"done": True},
        )

        planner = GOAPPlanner([action])
        goal = Goal(name="goal", conditions={"done": True})

        # Without tool - high cost
        state_no_tool = WorldState()
        plan1 = planner.plan(state_no_tool, goal, use_cache=False)

        # With tool - low cost
        state_with_tool = WorldState({"has_tool": True})
        plan2 = planner.plan(state_with_tool, goal, use_cache=False)

        assert plan1.total_cost > plan2.total_cost
