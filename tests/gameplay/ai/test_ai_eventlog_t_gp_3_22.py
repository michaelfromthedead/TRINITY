"""
T-GP-3.22: Tests for AI EventLog Integration.

Tests Foundation EventLog integration for AI decision logging and debugging.

Tests cover:
- Behavior Tree events (BTNodeEntered, BTNodeExited)
- GOAP events (GOAPPlanCreated, GOAPActionExecuted)
- Utility AI events (UtilityScoreComputed, UtilityActionSelected)
- Blackboard events (BlackboardValueChanged)
- Causal chain tracking
- Event query by entity
- Decision replay from events
- Performance benchmarks

Total: 50+ tests
"""

import time
import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock, patch

from foundation import EventLog, Event, get_event_log, clear_event_log, set_current_tick

from engine.gameplay.ai.ai_events import (
    AIEvent,
    BTNodeEntered,
    BTNodeExited,
    GOAPPlanCreated,
    GOAPActionExecuted,
    UtilityScoreComputed,
    UtilityActionSelected,
    BlackboardValueChanged,
    CausalChain,
    AIEventLogger,
    get_ai_event_logger,
    set_ai_event_logger,
    AIDecisionReplay,
)
from engine.gameplay.ai.behavior_tree import (
    BTContext,
    BTNode,
    BTStatus,
    Sequence,
    Selector,
    Parallel,
    Action,
    Condition,
    Wait,
    BehaviorTree,
)
from engine.gameplay.ai.blackboard import Blackboard
from engine.gameplay.ai.goap import (
    GOAPAgent,
    GOAPPlanner,
    GOAPAction,
    FunctionGOAPAction,
    Goal,
    WorldState,
)
from engine.gameplay.ai.utility_ai import (
    UtilityAI,
    UtilityAction,
    FunctionAction,
    Consideration,
    FunctionConsideration,
    ConsiderationContext,
)
from engine.gameplay.ai.constants import BTNodeType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_events():
    """Clear event log before and after each test."""
    clear_event_log()
    set_current_tick(0)
    yield
    clear_event_log()


@pytest.fixture
def event_logger():
    """Create a fresh AI event logger."""
    logger = AIEventLogger()
    set_ai_event_logger(logger)
    return logger


@pytest.fixture
def blackboard():
    """Create a blackboard for testing."""
    return Blackboard(name="test", entity_id=1)


@pytest.fixture
def mock_entity():
    """Create a mock entity with an ID."""
    entity = Mock()
    entity.id = 42
    return entity


# =============================================================================
# BTNodeEntered Event Tests
# =============================================================================


class TestBTNodeEnteredEvent:
    """Tests for BTNodeEntered event."""

    def test_bt_node_entered_fires_on_node_entry(self, event_logger, mock_entity):
        """BTNodeEntered fires when node begins execution."""
        event_logger.log_bt_node_entered(
            entity_id=mock_entity.id,
            bt_name="TestBT",
            node_name="TestSequence",
            node_type="SEQUENCE",
        )

        events = get_event_log().events_for_operation("BT.NodeEntered")
        assert len(events) == 1
        assert events[0].entity == mock_entity.id
        assert events[0].operation_args["node_name"] == "TestSequence"
        assert events[0].operation_args["node_type"] == "SEQUENCE"

    def test_bt_node_entered_captures_bt_name(self, event_logger, mock_entity):
        """BTNodeEntered captures the behavior tree name."""
        event_logger.log_bt_node_entered(
            entity_id=mock_entity.id,
            bt_name="PatrolBehavior",
            node_name="RootSelector",
            node_type="SELECTOR",
        )

        events = get_event_log().events_for_operation("BT.NodeEntered")
        assert events[0].operation_args["bt_name"] == "PatrolBehavior"

    def test_bt_node_entered_has_timestamp(self, event_logger, mock_entity):
        """BTNodeEntered includes a timestamp."""
        before = time.time()
        event_logger.log_bt_node_entered(
            entity_id=mock_entity.id,
            bt_name="TestBT",
            node_name="TestNode",
            node_type="ACTION",
        )
        after = time.time()

        events = get_event_log().events_for_operation("BT.NodeEntered")
        timestamp = events[0].operation_args["timestamp"]
        assert before <= timestamp <= after

    def test_bt_node_entered_fires_for_each_node_type(self, event_logger, mock_entity):
        """BTNodeEntered fires for all node types."""
        node_types = ["SEQUENCE", "SELECTOR", "PARALLEL", "ACTION", "CONDITION"]

        for i, node_type in enumerate(node_types):
            set_current_tick(i)
            event_logger.log_bt_node_entered(
                entity_id=mock_entity.id,
                bt_name="TestBT",
                node_name=f"Node_{node_type}",
                node_type=node_type,
            )

        events = get_event_log().events_for_operation("BT.NodeEntered")
        assert len(events) == len(node_types)

    def test_bt_node_entered_in_behavior_tree_tick(self, event_logger, mock_entity):
        """BTNodeEntered fires during behavior tree tick."""
        action = Action(lambda ctx: BTStatus.SUCCESS, name="TestAction")
        tree = BehaviorTree(root=action, name="TestBT")

        tree.tick(entity=mock_entity)

        events = get_event_log().events_for_operation("BT.NodeEntered")
        assert len(events) >= 1
        assert any(e.operation_args["node_name"] == "TestAction" for e in events)


# =============================================================================
# BTNodeExited Event Tests
# =============================================================================


class TestBTNodeExitedEvent:
    """Tests for BTNodeExited event."""

    def test_bt_node_exited_fires_with_correct_result(self, event_logger, mock_entity):
        """BTNodeExited fires with correct result status."""
        event_logger.log_bt_node_exited(
            entity_id=mock_entity.id,
            bt_name="TestBT",
            node_name="TestAction",
            result="SUCCESS",
        )

        events = get_event_log().events_for_operation("BT.NodeExited")
        assert len(events) == 1
        assert events[0].operation_args["result"] == "SUCCESS"

    def test_bt_node_exited_captures_failure(self, event_logger, mock_entity):
        """BTNodeExited captures FAILURE result."""
        event_logger.log_bt_node_exited(
            entity_id=mock_entity.id,
            bt_name="TestBT",
            node_name="TestCondition",
            result="FAILURE",
        )

        events = get_event_log().events_for_operation("BT.NodeExited")
        assert events[0].operation_args["result"] == "FAILURE"

    def test_bt_node_exited_captures_running(self, event_logger, mock_entity):
        """BTNodeExited captures RUNNING result."""
        event_logger.log_bt_node_exited(
            entity_id=mock_entity.id,
            bt_name="TestBT",
            node_name="WaitNode",
            result="RUNNING",
        )

        events = get_event_log().events_for_operation("BT.NodeExited")
        assert events[0].operation_args["result"] == "RUNNING"

    def test_bt_node_exited_pairs_with_entered(self, event_logger, mock_entity):
        """Each BTNodeEntered should have a corresponding BTNodeExited."""
        event_logger.log_bt_node_entered(
            entity_id=mock_entity.id,
            bt_name="TestBT",
            node_name="PairedNode",
            node_type="ACTION",
        )
        event_logger.log_bt_node_exited(
            entity_id=mock_entity.id,
            bt_name="TestBT",
            node_name="PairedNode",
            result="SUCCESS",
        )

        entered = get_event_log().events_for_operation("BT.NodeEntered")
        exited = get_event_log().events_for_operation("BT.NodeExited")
        assert len(entered) == len(exited)

    def test_bt_node_exited_in_sequence_execution(self, event_logger, mock_entity):
        """BTNodeExited fires for each node in sequence."""
        action1 = Action(lambda ctx: BTStatus.SUCCESS, name="Action1")
        action2 = Action(lambda ctx: BTStatus.SUCCESS, name="Action2")
        seq = Sequence(children=[action1, action2], name="TestSequence")
        tree = BehaviorTree(root=seq, name="TestBT")

        tree.tick(entity=mock_entity)

        events = get_event_log().events_for_operation("BT.NodeExited")
        # Should have exited events for both actions and the sequence
        assert len(events) >= 2


# =============================================================================
# GOAPPlanCreated Event Tests
# =============================================================================


class TestGOAPPlanCreatedEvent:
    """Tests for GOAPPlanCreated event."""

    def test_goap_plan_created_captures_full_plan(self, event_logger):
        """GOAPPlanCreated captures the complete plan."""
        event_logger.log_goap_plan_created(
            entity_id=1,
            goal="CollectResources",
            actions=["FindResource", "MoveToResource", "Harvest"],
            cost=5.0,
        )

        events = get_event_log().events_for_operation("GOAP.PlanCreated")
        assert len(events) == 1
        assert events[0].operation_args["goal"] == "CollectResources"
        assert events[0].operation_args["actions"] == ["FindResource", "MoveToResource", "Harvest"]
        assert events[0].operation_args["cost"] == 5.0

    def test_goap_plan_created_fires_on_replan(self):
        """GOAPPlanCreated fires when agent replans."""
        # Create a simple GOAP setup
        action = FunctionGOAPAction(
            name="TestAction",
            func=lambda ctx: True,
            preconditions={},
            effects={"goal_achieved": True},
            cost=1.0,
        )

        planner = GOAPPlanner(actions=[action])
        agent = GOAPAgent(planner=planner, entity_id=10)
        agent.add_goal(Goal(name="TestGoal", conditions={"goal_achieved": True}))

        agent.replan()

        events = get_event_log().events_for_operation("GOAP.PlanCreated")
        assert len(events) == 1
        assert events[0].entity == 10
        assert events[0].operation_args["goal"] == "TestGoal"

    def test_goap_plan_created_captures_empty_plan(self, event_logger):
        """GOAPPlanCreated handles empty action list."""
        event_logger.log_goap_plan_created(
            entity_id=1,
            goal="AlreadySatisfied",
            actions=[],
            cost=0.0,
        )

        events = get_event_log().events_for_operation("GOAP.PlanCreated")
        assert events[0].operation_args["actions"] == []
        assert events[0].operation_args["cost"] == 0.0

    def test_goap_plan_created_includes_cost(self, event_logger):
        """GOAPPlanCreated includes the total plan cost."""
        event_logger.log_goap_plan_created(
            entity_id=1,
            goal="ExpensiveGoal",
            actions=["A", "B", "C"],
            cost=15.5,
        )

        events = get_event_log().events_for_operation("GOAP.PlanCreated")
        assert events[0].operation_args["cost"] == 15.5


# =============================================================================
# GOAPActionExecuted Event Tests
# =============================================================================


class TestGOAPActionExecutedEvent:
    """Tests for GOAPActionExecuted event."""

    def test_goap_action_executed_fires_per_action(self, event_logger):
        """GOAPActionExecuted fires for each action execution."""
        event_logger.log_goap_action_executed(
            entity_id=1,
            action="MoveToTarget",
            success=True,
        )

        events = get_event_log().events_for_operation("GOAP.ActionExecuted")
        assert len(events) == 1
        assert events[0].operation_args["action"] == "MoveToTarget"
        assert events[0].operation_args["success"] is True

    def test_goap_action_executed_captures_failure(self, event_logger):
        """GOAPActionExecuted captures action failures."""
        event_logger.log_goap_action_executed(
            entity_id=1,
            action="FailedAction",
            success=False,
        )

        events = get_event_log().events_for_operation("GOAP.ActionExecuted")
        assert events[0].operation_args["success"] is False

    def test_goap_action_executed_during_agent_update(self):
        """GOAPActionExecuted fires during GOAPAgent.update()."""
        action = FunctionGOAPAction(
            name="ExecuteAction",
            func=lambda ctx: True,
            preconditions={},
            effects={"done": True},
            cost=1.0,
        )

        planner = GOAPPlanner(actions=[action])
        agent = GOAPAgent(planner=planner, entity_id=5)
        agent.add_goal(Goal(name="CompleteTask", conditions={"done": True}))

        agent.replan()
        agent.update()

        events = get_event_log().events_for_operation("GOAP.ActionExecuted")
        assert len(events) == 1
        assert events[0].entity == 5
        assert events[0].operation_args["action"] == "ExecuteAction"


# =============================================================================
# UtilityScoreComputed Event Tests
# =============================================================================


class TestUtilityScoreComputedEvent:
    """Tests for UtilityScoreComputed event."""

    def test_utility_score_computed_fires_for_each_consideration(self, event_logger):
        """UtilityScoreComputed fires for each consideration."""
        event_logger.log_utility_score_computed(
            entity_id=1,
            consideration="HealthFactor",
            score=0.75,
        )
        event_logger.log_utility_score_computed(
            entity_id=1,
            consideration="DistanceFactor",
            score=0.5,
        )

        events = get_event_log().events_for_operation("Utility.ScoreComputed")
        assert len(events) == 2
        considerations = [e.operation_args["consideration"] for e in events]
        assert "HealthFactor" in considerations
        assert "DistanceFactor" in considerations

    def test_utility_score_computed_captures_score_value(self, event_logger):
        """UtilityScoreComputed captures the exact score."""
        event_logger.log_utility_score_computed(
            entity_id=1,
            consideration="Urgency",
            score=0.923,
        )

        events = get_event_log().events_for_operation("Utility.ScoreComputed")
        assert abs(events[0].operation_args["score"] - 0.923) < 0.001

    def test_utility_score_computed_during_evaluation(self):
        """UtilityScoreComputed fires during UtilityAI.evaluate()."""
        consideration = FunctionConsideration(
            name="TestFactor",
            func=lambda ctx: 0.8,
        )
        action = FunctionAction(
            name="TestAction",
            func=lambda ctx: True,
            considerations=[consideration],
        )
        ai = UtilityAI(name="TestAI", entity_id=7)
        ai.add_action(action)

        ai.evaluate()

        events = get_event_log().events_for_operation("Utility.ScoreComputed")
        assert len(events) >= 1


# =============================================================================
# UtilityActionSelected Event Tests
# =============================================================================


class TestUtilityActionSelectedEvent:
    """Tests for UtilityActionSelected event."""

    def test_utility_action_selected_fires_on_selection(self, event_logger):
        """UtilityActionSelected fires when action is selected."""
        event_logger.log_utility_action_selected(
            entity_id=1,
            action="AttackEnemy",
            score=0.9,
            all_scores={"AttackEnemy": 0.9, "Retreat": 0.3, "Heal": 0.5},
        )

        events = get_event_log().events_for_operation("Utility.ActionSelected")
        assert len(events) == 1
        assert events[0].operation_args["action"] == "AttackEnemy"
        assert events[0].operation_args["score"] == 0.9

    def test_utility_action_selected_includes_all_scores(self, event_logger):
        """UtilityActionSelected includes scores for all evaluated actions."""
        event_logger.log_utility_action_selected(
            entity_id=1,
            action="BestAction",
            score=0.95,
            all_scores={"BestAction": 0.95, "SecondBest": 0.7, "Worst": 0.1},
        )

        events = get_event_log().events_for_operation("Utility.ActionSelected")
        all_scores = events[0].operation_args["all_scores"]
        assert len(all_scores) == 3
        assert all_scores["BestAction"] == 0.95

    def test_utility_action_selected_during_select_action(self):
        """UtilityActionSelected fires during UtilityAI.select_action()."""
        action = FunctionAction(
            name="SelectedAction",
            func=lambda ctx: True,
            base_score=0.5,
        )
        ai = UtilityAI(name="TestAI", entity_id=9)
        ai.add_action(action)

        ai.select_action()

        events = get_event_log().events_for_operation("Utility.ActionSelected")
        assert len(events) == 1
        assert events[0].entity == 9


# =============================================================================
# BlackboardValueChanged Event Tests
# =============================================================================


class TestBlackboardValueChangedEvent:
    """Tests for BlackboardValueChanged event."""

    def test_blackboard_value_changed_tracks_mutations(self):
        """BlackboardValueChanged fires when values change."""
        bb = Blackboard(name="test", entity_id=3)
        bb.set("target", "enemy1")

        events = get_event_log().events_for_operation("Blackboard.ValueChanged")
        assert len(events) == 1
        assert events[0].entity == 3
        assert events[0].operation_args["key"] == "global.target"

    def test_blackboard_value_changed_captures_old_and_new_values(self):
        """BlackboardValueChanged captures both old and new values."""
        bb = Blackboard(name="test", entity_id=4)
        bb.set("health", 100)
        bb.set("health", 75)

        events = get_event_log().events_for_operation("Blackboard.ValueChanged")
        # First set: old=None, new=100
        # Second set: old=100, new=75
        assert len(events) == 2
        assert "75" in events[1].operation_args["new_value"]

    def test_blackboard_value_changed_uses_entity_id(self):
        """BlackboardValueChanged uses the blackboard's entity_id."""
        bb = Blackboard(name="test", entity_id=42)
        bb.set("key", "value")

        events = get_event_log().events_for_operation("Blackboard.ValueChanged")
        assert events[0].entity == 42

    def test_blackboard_value_changed_not_fired_for_same_value(self):
        """BlackboardValueChanged doesn't fire if value is unchanged."""
        bb = Blackboard(name="test", entity_id=5)
        bb.set("key", "same_value")
        bb.set("key", "same_value")  # Same value, should not fire

        events = get_event_log().events_for_operation("Blackboard.ValueChanged")
        assert len(events) == 1  # Only the first set


# =============================================================================
# Causal Chain Tests
# =============================================================================


class TestCausalChain:
    """Tests for causal chain tracking."""

    def test_causal_chain_links_parent_to_children(self, event_logger):
        """Causal chain correctly links parent nodes to children."""
        chain = event_logger.causal_chain

        chain.push_parent("ParentNode")
        event_logger.log_bt_node_entered(
            entity_id=1,
            bt_name="TestBT",
            node_name="ChildNode",
            node_type="ACTION",
        )

        events = get_event_log().events_for_operation("BT.NodeEntered")
        # The parent should be set on the event
        assert len(events) == 1

    def test_causal_chain_nested_operations(self, event_logger):
        """Causal chain handles nested operations."""
        chain = event_logger.causal_chain

        chain.push_parent("Level1")
        chain.push_parent("Level2")
        assert chain.current_parent() == "Level2"

        chain.pop_parent()
        assert chain.current_parent() == "Level1"

        chain.pop_parent()
        assert chain.current_parent() is None

    def test_causal_chain_clear(self, event_logger):
        """CausalChain.clear() resets the chain."""
        chain = event_logger.causal_chain
        chain.push_parent("Node1")
        chain.push_parent("Node2")

        chain.clear()

        assert chain.current_parent() is None

    def test_behavior_tree_causal_chain(self, mock_entity):
        """Behavior tree execution builds causal chain."""
        action1 = Action(lambda ctx: BTStatus.SUCCESS, name="Child1")
        action2 = Action(lambda ctx: BTStatus.SUCCESS, name="Child2")
        seq = Sequence(children=[action1, action2], name="ParentSeq")
        tree = BehaviorTree(root=seq, name="CausalBT")

        tree.tick(entity=mock_entity)

        # Get all events
        all_events = get_event_log().all_events()
        # There should be entry/exit pairs with parent relationships
        assert len(all_events) >= 4  # At least 2 entries and 2 exits


# =============================================================================
# Event Query Tests
# =============================================================================


class TestEventQuery:
    """Tests for event querying functionality."""

    def test_event_query_by_entity_id(self, event_logger):
        """Events can be queried by entity_id."""
        event_logger.log_bt_node_entered(entity_id=1, bt_name="BT1", node_name="N1", node_type="ACTION")
        event_logger.log_bt_node_entered(entity_id=2, bt_name="BT2", node_name="N2", node_type="ACTION")
        event_logger.log_bt_node_entered(entity_id=1, bt_name="BT1", node_name="N3", node_type="ACTION")

        events_for_1 = event_logger.get_events_for_entity(1)
        assert len(events_for_1) == 2
        assert all(e.entity == 1 for e in events_for_1)

    def test_event_query_by_operation_type(self, event_logger):
        """Events can be queried by operation type."""
        event_logger.log_bt_node_entered(entity_id=1, bt_name="BT", node_name="N", node_type="ACTION")
        event_logger.log_goap_plan_created(entity_id=1, goal="G", actions=[], cost=0)

        bt_events = event_logger.get_bt_events()
        goap_events = event_logger.get_goap_events()

        assert len(bt_events) == 1
        assert len(goap_events) == 1

    def test_get_utility_events(self, event_logger):
        """get_utility_events returns utility-related events."""
        event_logger.log_utility_score_computed(entity_id=1, consideration="C", score=0.5)
        event_logger.log_utility_action_selected(entity_id=1, action="A", score=0.5)

        events = event_logger.get_utility_events()
        assert len(events) == 2

    def test_get_blackboard_events(self, event_logger):
        """get_blackboard_events returns blackboard events."""
        event_logger.log_blackboard_value_changed(entity_id=1, key="k1", old_value=None, new_value=1)
        event_logger.log_blackboard_value_changed(entity_id=1, key="k2", old_value=None, new_value=2)

        events = event_logger.get_blackboard_events()
        assert len(events) == 2


# =============================================================================
# Decision Replay Tests
# =============================================================================


class TestDecisionReplay:
    """Tests for AI decision replay functionality."""

    def test_decision_replay_from_events(self, event_logger):
        """Decisions can be replayed from recorded events."""
        # Record a sequence of events
        set_current_tick(1)
        event_logger.log_bt_node_entered(entity_id=1, bt_name="BT", node_name="Root", node_type="SEQUENCE")
        set_current_tick(2)
        event_logger.log_bt_node_entered(entity_id=1, bt_name="BT", node_name="Action1", node_type="ACTION")
        set_current_tick(3)
        event_logger.log_bt_node_exited(entity_id=1, bt_name="BT", node_name="Action1", result="SUCCESS")
        set_current_tick(4)
        event_logger.log_bt_node_exited(entity_id=1, bt_name="BT", node_name="Root", result="SUCCESS")

        replay = AIDecisionReplay()
        path = replay.reconstruct_bt_execution(entity_id=1)

        assert len(path) == 4
        assert path[0]["node_name"] == "Root"
        assert path[1]["node_name"] == "Action1"
        assert path[2]["result"] == "SUCCESS"

    def test_decision_replay_goap_execution(self, event_logger):
        """GOAP plan execution can be reconstructed."""
        set_current_tick(1)
        event_logger.log_goap_plan_created(
            entity_id=1, goal="Goal", actions=["A", "B"], cost=2.0
        )
        set_current_tick(2)
        event_logger.log_goap_action_executed(entity_id=1, action="A", success=True)
        set_current_tick(3)
        event_logger.log_goap_action_executed(entity_id=1, action="B", success=True)

        replay = AIDecisionReplay()
        sequence = replay.reconstruct_goap_plan_execution(entity_id=1)

        assert len(sequence) == 3
        assert sequence[0]["operation"] == "GOAP.PlanCreated"
        assert sequence[1]["operation"] == "GOAP.ActionExecuted"
        assert sequence[1]["action"] == "A"

    def test_decision_replay_utility_decision(self, event_logger):
        """Utility AI decision can be reconstructed at a tick."""
        set_current_tick(5)
        event_logger.log_utility_score_computed(entity_id=1, consideration="Health", score=0.8)
        event_logger.log_utility_score_computed(entity_id=1, consideration="Distance", score=0.6)
        event_logger.log_utility_action_selected(entity_id=1, action="Attack", score=0.9)

        replay = AIDecisionReplay()
        decision = replay.reconstruct_utility_decision(entity_id=1, tick=5)

        assert decision["tick"] == 5
        assert decision["selected_action"] == "Attack"
        assert "Health" in decision["scores"]
        assert "Distance" in decision["scores"]

    def test_decision_replay_blackboard_history(self, event_logger):
        """Blackboard change history can be reconstructed."""
        set_current_tick(1)
        event_logger.log_blackboard_value_changed(entity_id=1, key="target", old_value=None, new_value="enemy1")
        set_current_tick(2)
        event_logger.log_blackboard_value_changed(entity_id=1, key="target", old_value="enemy1", new_value="enemy2")

        replay = AIDecisionReplay()
        history = replay.get_blackboard_history(entity_id=1, key="target")

        assert len(history) == 2
        assert history[0]["tick"] == 1
        assert history[1]["tick"] == 2

    def test_decision_replay_with_tick_range(self, event_logger):
        """Decision replay can filter by tick range."""
        for tick in range(10):
            set_current_tick(tick)
            event_logger.log_bt_node_entered(
                entity_id=1, bt_name="BT", node_name=f"Node{tick}", node_type="ACTION"
            )

        replay = AIDecisionReplay()
        path = replay.reconstruct_bt_execution(entity_id=1, start_tick=3, end_tick=6)

        assert len(path) == 4  # Ticks 3, 4, 5, 6


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance benchmarks for AI event logging."""

    def test_1000_ai_decisions_under_200ms(self, event_logger):
        """1000 AI decisions should be logged in under 200ms."""
        start_time = time.time()

        for i in range(1000):
            set_current_tick(i)
            # Simulate a full decision cycle
            event_logger.log_bt_node_entered(
                entity_id=1, bt_name="BT", node_name=f"Node{i}", node_type="ACTION"
            )
            event_logger.log_bt_node_exited(
                entity_id=1, bt_name="BT", node_name=f"Node{i}", result="SUCCESS"
            )

        elapsed_ms = (time.time() - start_time) * 1000

        assert elapsed_ms < 200, f"Took {elapsed_ms:.2f}ms, expected < 200ms"

    def test_event_query_performance(self, event_logger):
        """Event queries should be efficient with many events."""
        # Log many events
        for i in range(1000):
            set_current_tick(i)
            event_logger.log_bt_node_entered(
                entity_id=i % 10,  # 10 different entities
                bt_name="BT",
                node_name=f"Node{i}",
                node_type="ACTION",
            )

        start_time = time.time()
        events = event_logger.get_events_for_entity(5)
        elapsed_ms = (time.time() - start_time) * 1000

        assert len(events) == 100  # 1000 / 10 entities
        assert elapsed_ms < 50, f"Query took {elapsed_ms:.2f}ms, expected < 50ms"

    def test_replay_performance(self, event_logger):
        """Decision replay should be efficient."""
        # Log a complex decision tree
        for i in range(500):
            set_current_tick(i)
            event_logger.log_bt_node_entered(
                entity_id=1, bt_name="BT", node_name=f"Node{i}", node_type="ACTION"
            )
            event_logger.log_bt_node_exited(
                entity_id=1, bt_name="BT", node_name=f"Node{i}", result="SUCCESS"
            )

        replay = AIDecisionReplay()
        start_time = time.time()
        path = replay.reconstruct_bt_execution(entity_id=1)
        elapsed_ms = (time.time() - start_time) * 1000

        assert len(path) == 1000
        assert elapsed_ms < 100, f"Replay took {elapsed_ms:.2f}ms, expected < 100ms"


# =============================================================================
# Event Logger Configuration Tests
# =============================================================================


class TestEventLoggerConfiguration:
    """Tests for AIEventLogger configuration."""

    def test_event_logging_can_be_disabled(self):
        """Event logging can be disabled."""
        logger = AIEventLogger()
        logger.enabled = False

        logger.log_bt_node_entered(entity_id=1, bt_name="BT", node_name="N", node_type="ACTION")

        events = get_event_log().events_for_operation("BT.NodeEntered")
        assert len(events) == 0

    def test_event_logging_can_be_enabled(self):
        """Event logging can be re-enabled."""
        logger = AIEventLogger()
        logger.enabled = False
        logger.enabled = True

        logger.log_bt_node_entered(entity_id=1, bt_name="BT", node_name="N", node_type="ACTION")

        events = get_event_log().events_for_operation("BT.NodeEntered")
        assert len(events) == 1

    def test_global_logger_singleton(self):
        """get_ai_event_logger returns the same instance."""
        logger1 = get_ai_event_logger()
        logger2 = get_ai_event_logger()
        assert logger1 is logger2

    def test_set_ai_event_logger(self):
        """set_ai_event_logger replaces the global logger."""
        original = get_ai_event_logger()
        new_logger = AIEventLogger()
        set_ai_event_logger(new_logger)

        assert get_ai_event_logger() is new_logger
        assert get_ai_event_logger() is not original

        # Restore original
        set_ai_event_logger(original)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete AI scenarios."""

    def test_full_behavior_tree_execution_logging(self, mock_entity):
        """Complete behavior tree execution is fully logged."""
        # Build a complex tree
        action1 = Action(lambda ctx: BTStatus.SUCCESS, name="GatherResources")
        action2 = Action(lambda ctx: BTStatus.SUCCESS, name="ReturnToBase")
        condition = Condition(lambda ctx: True, name="HasResources")
        seq = Sequence(children=[action1, condition, action2], name="GatherSequence")
        tree = BehaviorTree(root=seq, name="GatherBT")

        # Execute
        tree.tick(entity=mock_entity)

        # Verify all events were logged
        entered = get_event_log().events_for_operation("BT.NodeEntered")
        exited = get_event_log().events_for_operation("BT.NodeExited")

        assert len(entered) >= 4  # Sequence + 3 children
        assert len(exited) >= 4
        assert all(e.entity == mock_entity.id for e in entered)

    def test_goap_agent_complete_cycle(self):
        """Complete GOAP agent cycle is logged."""
        action1 = FunctionGOAPAction(
            name="Step1",
            func=lambda ctx: True,
            effects={"step1_done": True},
        )
        action2 = FunctionGOAPAction(
            name="Step2",
            func=lambda ctx: True,
            preconditions={"step1_done": True},
            effects={"goal_done": True},
        )

        planner = GOAPPlanner(actions=[action1, action2])
        agent = GOAPAgent(planner=planner, entity_id=100)
        agent.add_goal(Goal(name="CompleteSteps", conditions={"goal_done": True}))

        # Execute complete cycle
        agent.replan()
        agent.update()  # Step1
        agent.update()  # Step2

        # Verify events
        plan_events = get_event_log().events_for_operation("GOAP.PlanCreated")
        action_events = get_event_log().events_for_operation("GOAP.ActionExecuted")

        assert len(plan_events) == 1
        assert len(action_events) == 2

    def test_utility_ai_complete_evaluation(self):
        """Complete utility AI evaluation is logged."""
        consideration1 = FunctionConsideration(name="Factor1", func=lambda ctx: 0.9)
        consideration2 = FunctionConsideration(name="Factor2", func=lambda ctx: 0.7)

        action = FunctionAction(
            name="OptimalAction",
            func=lambda ctx: True,
            considerations=[consideration1, consideration2],
        )

        ai = UtilityAI(name="TestAI", entity_id=50)
        ai.add_action(action)
        ai.select_action()

        # Verify events
        score_events = get_event_log().events_for_operation("Utility.ScoreComputed")
        select_events = get_event_log().events_for_operation("Utility.ActionSelected")

        assert len(score_events) >= 2  # At least 2 considerations
        assert len(select_events) == 1

    def test_mixed_ai_systems_logging(self, mock_entity):
        """Multiple AI systems can be logged simultaneously."""
        # BT
        action = Action(lambda ctx: BTStatus.SUCCESS, name="BTAction")
        tree = BehaviorTree(root=action, name="MixedBT")
        tree.tick(entity=mock_entity)

        # GOAP
        goap_action = FunctionGOAPAction(
            name="GOAPAction",
            func=lambda ctx: True,
            effects={"done": True},
        )
        planner = GOAPPlanner(actions=[goap_action])
        agent = GOAPAgent(planner=planner, entity_id=mock_entity.id)
        agent.add_goal(Goal(name="Goal", conditions={"done": True}))
        agent.replan()
        agent.update()

        # Utility
        util_action = FunctionAction(name="UtilAction", func=lambda ctx: True, base_score=0.5)
        util_ai = UtilityAI(entity_id=mock_entity.id)
        util_ai.add_action(util_action)
        util_ai.select_action()

        # Blackboard
        bb = Blackboard(entity_id=mock_entity.id)
        bb.set("key", "value")

        # Verify all systems logged events
        all_events = get_event_log().all_events()
        operations = {e.operation for e in all_events}

        assert "BT.NodeEntered" in operations
        assert "GOAP.PlanCreated" in operations
        assert "Utility.ActionSelected" in operations
        assert "Blackboard.ValueChanged" in operations
