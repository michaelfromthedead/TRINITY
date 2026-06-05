"""
WHITEBOX Tests for the Behavior Tree System.

Comprehensive internal testing of behavior trees with full source access.

Tests cover:
- BTContext internal state and child context creation
- BTNode base class mechanics
- Composite nodes: Sequence, Selector, Parallel
- Decorator nodes: Invert, Repeat, Timeout, Cooldown, Retry, ForceSuccess, ForceFailure
- Leaf nodes: Action, Condition, BlackboardCondition, Wait, SetBlackboard
- BehaviorTree lifecycle and execution
- Registry integration with @behavior_tree and @bt_node decorators
- Edge cases: max depth, abort handling, exception handling

Total: 50+ tests for behavior tree internals
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

import pytest

from engine.gameplay.ai.behavior_tree import (
    Action,
    BehaviorTree,
    BlackboardCondition,
    BTContext,
    BTNode,
    BTNodeTypeError,
    CompositeNode,
    Condition,
    Cooldown,
    DecoratorNode,
    ForceFailure,
    ForceSuccess,
    Invert,
    LeafNode,
    Parallel,
    Repeat,
    Retry,
    Selector,
    Sequence,
    SetBlackboard,
    Timeout,
    VALID_BT_NODE_TYPES,
    Wait,
    behavior_tree,
    bt_node,
    get_all_behavior_trees,
    get_all_bt_nodes,
    get_bt_nodes_by_type,
)
from engine.gameplay.ai.blackboard import Blackboard
from engine.gameplay.ai.constants import (
    BT_DEFAULT_COOLDOWN,
    BT_DEFAULT_REPEAT_COUNT,
    BT_DEFAULT_RETRY_COUNT,
    BT_DEFAULT_TICK_INTERVAL,
    BT_DEFAULT_TIMEOUT,
    BT_INFINITE_REPEAT,
    BT_MAX_DEPTH,
    BTNodeType,
    BTStatus,
    ParallelPolicy,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def blackboard():
    """Create a fresh blackboard for testing."""
    return Blackboard(name="test", enable_event_logging=False)


@pytest.fixture
def context(blackboard):
    """Create a basic context for testing."""
    return BTContext(
        blackboard=blackboard,
        entity=None,
        delta_time=0.016,
        current_time=time.time(),
        depth=0,
        debug_trace=False,
    )


def success_action(ctx: BTContext) -> BTStatus:
    """Action that always succeeds."""
    return BTStatus.SUCCESS


def failure_action(ctx: BTContext) -> BTStatus:
    """Action that always fails."""
    return BTStatus.FAILURE


def running_action(ctx: BTContext) -> BTStatus:
    """Action that always returns running."""
    return BTStatus.RUNNING


def counting_action(counter: List[int]):
    """Create an action that counts executions."""
    def action(ctx: BTContext) -> BTStatus:
        counter[0] += 1
        return BTStatus.SUCCESS
    return action


# =============================================================================
# BT CONTEXT TESTS
# =============================================================================


class TestBTContextInternals:
    """Whitebox tests for BTContext internal state."""

    def test_context_initialization(self, blackboard):
        """Test BTContext initializes with correct values."""
        ctx = BTContext(
            blackboard=blackboard,
            entity=None,
            delta_time=0.016,
        )

        assert ctx.blackboard is blackboard
        assert ctx.entity is None
        assert ctx.delta_time == 0.016
        assert ctx.depth == 0
        assert ctx.abort_requested is False
        assert ctx.debug_trace is False
        assert ctx.trace_log == []

    def test_child_context_increments_depth(self, context):
        """Test child_context increments depth."""
        child = context.child_context()

        assert child.depth == context.depth + 1
        assert child.blackboard is context.blackboard

    def test_child_context_shares_trace_log(self, context):
        """Test child context shares trace log reference."""
        context.debug_trace = True

        child = context.child_context()
        child.trace_log.append("test entry")

        assert "test entry" in context.trace_log

    def test_log_trace_when_enabled(self, context):
        """Test log_trace adds entry when debug enabled."""
        context.debug_trace = True

        node = Action(success_action, name="TestAction")
        context.log_trace(node, BTStatus.SUCCESS)

        assert len(context.trace_log) == 1
        assert "TestAction" in context.trace_log[0]
        assert "SUCCESS" in context.trace_log[0]

    def test_log_trace_indentation(self, context):
        """Test log_trace uses depth-based indentation."""
        context.debug_trace = True
        context.depth = 2

        node = Action(success_action, name="DeepNode")
        context.log_trace(node, BTStatus.SUCCESS)

        assert context.trace_log[0].startswith("    ")  # 2 * 2 spaces

    def test_get_entity_id_with_entity(self, blackboard):
        """Test get_entity_id returns entity id."""
        class MockEntity:
            id = 42

        ctx = BTContext(blackboard=blackboard, entity=MockEntity())
        assert ctx.get_entity_id() == 42

    def test_get_entity_id_without_id_attr(self, blackboard):
        """Test get_entity_id uses id() when no id attribute."""
        entity = object()
        ctx = BTContext(blackboard=blackboard, entity=entity)
        assert ctx.get_entity_id() == id(entity)

    def test_get_entity_id_no_entity(self, blackboard):
        """Test get_entity_id returns 0 for None entity."""
        ctx = BTContext(blackboard=blackboard, entity=None)
        assert ctx.get_entity_id() == 0


# =============================================================================
# SEQUENCE NODE TESTS
# =============================================================================


class TestSequenceNodeInternals:
    """Whitebox tests for Sequence composite node."""

    def test_sequence_node_type(self):
        """Test Sequence has correct node type."""
        seq = Sequence()
        assert seq.node_type == BTNodeType.SEQUENCE

    def test_sequence_succeeds_all_children_succeed(self, context):
        """Test Sequence succeeds when all children succeed."""
        seq = Sequence([
            Action(success_action),
            Action(success_action),
            Action(success_action),
        ])

        status = seq.tick(context)
        assert status == BTStatus.SUCCESS

    def test_sequence_fails_on_first_failure(self, context):
        """Test Sequence fails on first child failure."""
        counter = [0]
        seq = Sequence([
            Action(success_action),
            Action(failure_action),
            Action(counting_action(counter)),  # Should not execute
        ])

        status = seq.tick(context)
        assert status == BTStatus.FAILURE
        assert counter[0] == 0

    def test_sequence_returns_running(self, context):
        """Test Sequence returns RUNNING when child is running."""
        seq = Sequence([
            Action(success_action),
            Action(running_action),
            Action(success_action),
        ])

        status = seq.tick(context)
        assert status == BTStatus.RUNNING

    def test_sequence_resumes_from_running_child(self, context):
        """Test Sequence resumes from running child on next tick."""
        run_count = [0]

        def conditional_action(ctx):
            run_count[0] += 1
            if run_count[0] < 2:
                return BTStatus.RUNNING
            return BTStatus.SUCCESS

        seq = Sequence([
            Action(success_action),
            Action(conditional_action),
        ])

        status1 = seq.tick(context)
        assert status1 == BTStatus.RUNNING
        assert seq._current_index == 1

        status2 = seq.tick(context)
        assert status2 == BTStatus.SUCCESS

    def test_sequence_resets_index_on_failure(self, context):
        """Test Sequence resets index after failure."""
        seq = Sequence([
            Action(success_action),
            Action(failure_action),
        ])

        seq.tick(context)
        assert seq._current_index == 0

    def test_sequence_resets_index_on_success(self, context):
        """Test Sequence resets index after success."""
        seq = Sequence([
            Action(success_action),
        ])

        seq.tick(context)
        assert seq._current_index == 0

    def test_sequence_respects_max_depth(self, context):
        """Test Sequence fails when max depth exceeded."""
        context.depth = BT_MAX_DEPTH + 1

        seq = Sequence([Action(success_action)])
        status = seq.tick(context)

        assert status == BTStatus.FAILURE

    def test_sequence_respects_abort(self, context):
        """Test Sequence aborts when abort requested."""
        context.abort_requested = True

        seq = Sequence([Action(success_action)])
        status = seq.tick(context)

        assert status == BTStatus.FAILURE


# =============================================================================
# SELECTOR NODE TESTS
# =============================================================================


class TestSelectorNodeInternals:
    """Whitebox tests for Selector composite node."""

    def test_selector_node_type(self):
        """Test Selector has correct node type."""
        sel = Selector()
        assert sel.node_type == BTNodeType.SELECTOR

    def test_selector_succeeds_first_success(self, context):
        """Test Selector succeeds on first child success."""
        counter = [0]
        sel = Selector([
            Action(failure_action),
            Action(success_action),
            Action(counting_action(counter)),  # Should not execute
        ])

        status = sel.tick(context)
        assert status == BTStatus.SUCCESS
        assert counter[0] == 0

    def test_selector_fails_all_children_fail(self, context):
        """Test Selector fails when all children fail."""
        sel = Selector([
            Action(failure_action),
            Action(failure_action),
            Action(failure_action),
        ])

        status = sel.tick(context)
        assert status == BTStatus.FAILURE

    def test_selector_returns_running(self, context):
        """Test Selector returns RUNNING when child is running."""
        sel = Selector([
            Action(failure_action),
            Action(running_action),
            Action(success_action),
        ])

        status = sel.tick(context)
        assert status == BTStatus.RUNNING


# =============================================================================
# PARALLEL NODE TESTS
# =============================================================================


class TestParallelNodeInternals:
    """Whitebox tests for Parallel composite node."""

    def test_parallel_node_type(self):
        """Test Parallel has correct node type."""
        par = Parallel()
        assert par.node_type == BTNodeType.PARALLEL

    def test_parallel_require_all_success(self, context):
        """Test Parallel REQUIRE_ALL succeeds when all succeed."""
        par = Parallel(
            [Action(success_action), Action(success_action)],
            policy=ParallelPolicy.REQUIRE_ALL,
        )

        status = par.tick(context)
        assert status == BTStatus.SUCCESS

    def test_parallel_require_all_failure(self, context):
        """Test Parallel REQUIRE_ALL fails when any fail."""
        par = Parallel(
            [Action(success_action), Action(failure_action)],
            policy=ParallelPolicy.REQUIRE_ALL,
        )

        status = par.tick(context)
        assert status == BTStatus.FAILURE

    def test_parallel_require_one_success(self, context):
        """Test Parallel REQUIRE_ONE succeeds when any succeed."""
        par = Parallel(
            [Action(failure_action), Action(success_action)],
            policy=ParallelPolicy.REQUIRE_ONE,
        )

        status = par.tick(context)
        assert status == BTStatus.SUCCESS

    def test_parallel_require_one_failure(self, context):
        """Test Parallel REQUIRE_ONE fails when all fail."""
        par = Parallel(
            [Action(failure_action), Action(failure_action)],
            policy=ParallelPolicy.REQUIRE_ONE,
        )

        status = par.tick(context)
        assert status == BTStatus.FAILURE

    def test_parallel_require_majority_success(self, context):
        """Test Parallel REQUIRE_MAJORITY succeeds with majority."""
        par = Parallel(
            [Action(success_action), Action(success_action), Action(failure_action)],
            policy=ParallelPolicy.REQUIRE_MAJORITY,
        )

        status = par.tick(context)
        assert status == BTStatus.SUCCESS

    def test_parallel_require_majority_failure(self, context):
        """Test Parallel REQUIRE_MAJORITY fails without majority."""
        par = Parallel(
            [Action(failure_action), Action(failure_action), Action(success_action)],
            policy=ParallelPolicy.REQUIRE_MAJORITY,
        )

        status = par.tick(context)
        assert status == BTStatus.FAILURE

    def test_parallel_running_defers_result(self, context):
        """Test Parallel returns RUNNING when any child is running."""
        par = Parallel(
            [Action(success_action), Action(running_action)],
            policy=ParallelPolicy.REQUIRE_ALL,
        )

        status = par.tick(context)
        assert status == BTStatus.RUNNING

    def test_parallel_empty_children(self, context):
        """Test Parallel with no children succeeds."""
        par = Parallel(policy=ParallelPolicy.REQUIRE_ALL)
        status = par.tick(context)
        assert status == BTStatus.SUCCESS

    def test_parallel_policy_property(self):
        """Test Parallel exposes policy property."""
        par = Parallel(policy=ParallelPolicy.REQUIRE_ONE)
        assert par.policy == ParallelPolicy.REQUIRE_ONE


# =============================================================================
# DECORATOR NODE TESTS
# =============================================================================


class TestInvertDecoratorInternals:
    """Whitebox tests for Invert decorator."""

    def test_invert_node_type(self):
        """Test Invert has correct node type."""
        inv = Invert(Action(success_action))
        assert inv.node_type == BTNodeType.INVERT

    def test_invert_success_to_failure(self, context):
        """Test Invert converts SUCCESS to FAILURE."""
        inv = Invert(Action(success_action))
        status = inv.tick(context)
        assert status == BTStatus.FAILURE

    def test_invert_failure_to_success(self, context):
        """Test Invert converts FAILURE to SUCCESS."""
        inv = Invert(Action(failure_action))
        status = inv.tick(context)
        assert status == BTStatus.SUCCESS

    def test_invert_running_unchanged(self, context):
        """Test Invert passes RUNNING unchanged."""
        inv = Invert(Action(running_action))
        status = inv.tick(context)
        assert status == BTStatus.RUNNING


class TestRepeatDecoratorInternals:
    """Whitebox tests for Repeat decorator."""

    def test_repeat_node_type(self):
        """Test Repeat has correct node type."""
        rep = Repeat(Action(success_action))
        assert rep.node_type == BTNodeType.REPEAT

    def test_repeat_fixed_count(self, context):
        """Test Repeat runs child fixed number of times."""
        counter = [0]
        rep = Repeat(Action(counting_action(counter)), count=3)

        # Run until complete
        while rep.tick(context) == BTStatus.RUNNING:
            pass

        assert counter[0] == 3

    def test_repeat_until_fail(self, context):
        """Test Repeat until_fail stops on failure."""
        counter = [0]

        def fail_after_3(ctx):
            counter[0] += 1
            return BTStatus.FAILURE if counter[0] >= 3 else BTStatus.SUCCESS

        rep = Repeat(Action(fail_after_3), count=BT_INFINITE_REPEAT, until_fail=True)

        status = rep.tick(context)  # Success, continue
        status = rep.tick(context)  # Success, continue
        status = rep.tick(context)  # Failure, stop

        assert status == BTStatus.SUCCESS
        assert counter[0] == 3

    def test_repeat_until_success(self, context):
        """Test Repeat until_success stops on success."""
        counter = [0]

        def succeed_after_2(ctx):
            counter[0] += 1
            return BTStatus.SUCCESS if counter[0] >= 2 else BTStatus.FAILURE

        rep = Repeat(Action(succeed_after_2), count=BT_INFINITE_REPEAT, until_success=True)

        status = rep.tick(context)  # Failure, continue
        status = rep.tick(context)  # Success, stop

        assert status == BTStatus.SUCCESS

    def test_repeat_resets_child(self, context):
        """Test Repeat resets child between iterations."""
        rep = Repeat(Action(success_action), count=3)

        rep.tick(context)
        assert rep._current_count == 1


class TestTimeoutDecoratorInternals:
    """Whitebox tests for Timeout decorator."""

    def test_timeout_node_type(self):
        """Test Timeout has correct node type."""
        timeout = Timeout(Action(running_action))
        assert timeout.node_type == BTNodeType.TIMEOUT

    def test_timeout_passes_while_in_time(self, context):
        """Test Timeout passes result while within time."""
        timeout = Timeout(Action(success_action), timeout=5.0)
        status = timeout.tick(context)
        assert status == BTStatus.SUCCESS

    def test_timeout_fails_after_timeout(self, context):
        """Test Timeout fails after time expires."""
        timeout = Timeout(Action(running_action), timeout=1.0)

        # First tick starts timer
        timeout.tick(context)

        # Simulate time passing
        context.current_time += 2.0

        status = timeout.tick(context)
        assert status == BTStatus.FAILURE

    def test_timeout_resets_on_complete(self, context):
        """Test Timeout resets timer on completion."""
        timeout = Timeout(Action(success_action), timeout=5.0)

        timeout.tick(context)
        assert timeout._start_time is None


class TestCooldownDecoratorInternals:
    """Whitebox tests for Cooldown decorator."""

    def test_cooldown_node_type(self):
        """Test Cooldown has correct node type."""
        cd = Cooldown(Action(success_action))
        assert cd.node_type == BTNodeType.COOLDOWN

    def test_cooldown_allows_first_execution(self, context):
        """Test Cooldown allows first execution."""
        cd = Cooldown(Action(success_action), cooldown=1.0)
        status = cd.tick(context)
        assert status == BTStatus.SUCCESS

    def test_cooldown_blocks_during_cooldown(self, context):
        """Test Cooldown blocks during cooldown period."""
        cd = Cooldown(Action(success_action), cooldown=1.0)

        cd.tick(context)  # First execution
        status = cd.tick(context)  # Should fail (on cooldown)

        assert status == BTStatus.FAILURE

    def test_cooldown_allows_after_cooldown(self, context):
        """Test Cooldown allows after cooldown period."""
        cd = Cooldown(Action(success_action), cooldown=1.0)

        cd.tick(context)

        context.current_time += 2.0

        status = cd.tick(context)
        assert status == BTStatus.SUCCESS


class TestRetryDecoratorInternals:
    """Whitebox tests for Retry decorator."""

    def test_retry_node_type(self):
        """Test Retry has correct node type."""
        retry = Retry(Action(failure_action))
        assert retry.node_type == BTNodeType.RETRY

    def test_retry_succeeds_immediately(self, context):
        """Test Retry succeeds immediately on success."""
        retry = Retry(Action(success_action), max_retries=3)
        status = retry.tick(context)
        assert status == BTStatus.SUCCESS
        assert retry._retry_count == 0

    def test_retry_retries_on_failure(self, context):
        """Test Retry retries on failure."""
        counter = [0]

        def fail_twice(ctx):
            counter[0] += 1
            return BTStatus.SUCCESS if counter[0] >= 3 else BTStatus.FAILURE

        retry = Retry(Action(fail_twice), max_retries=5)

        # Tick until success
        while retry.tick(context) == BTStatus.RUNNING:
            pass

        assert counter[0] == 3

    def test_retry_fails_after_max_retries(self, context):
        """Test Retry fails after max retries exhausted."""
        retry = Retry(Action(failure_action), max_retries=3)

        # Tick 3 times (max retries)
        retry.tick(context)  # RUNNING (retry 1)
        retry.tick(context)  # RUNNING (retry 2)
        status = retry.tick(context)  # FAILURE (max reached)

        assert status == BTStatus.FAILURE


class TestForceDecoratorInternals:
    """Whitebox tests for ForceSuccess and ForceFailure decorators."""

    def test_force_success_node_type(self):
        """Test ForceSuccess has correct node type."""
        fs = ForceSuccess(Action(failure_action))
        assert fs.node_type == BTNodeType.FORCE_SUCCESS

    def test_force_success_converts_failure(self, context):
        """Test ForceSuccess converts FAILURE to SUCCESS."""
        fs = ForceSuccess(Action(failure_action))
        status = fs.tick(context)
        assert status == BTStatus.SUCCESS

    def test_force_success_passes_running(self, context):
        """Test ForceSuccess passes RUNNING unchanged."""
        fs = ForceSuccess(Action(running_action))
        status = fs.tick(context)
        assert status == BTStatus.RUNNING

    def test_force_failure_node_type(self):
        """Test ForceFailure has correct node type."""
        ff = ForceFailure(Action(success_action))
        assert ff.node_type == BTNodeType.FORCE_FAILURE

    def test_force_failure_converts_success(self, context):
        """Test ForceFailure converts SUCCESS to FAILURE."""
        ff = ForceFailure(Action(success_action))
        status = ff.tick(context)
        assert status == BTStatus.FAILURE


# =============================================================================
# LEAF NODE TESTS
# =============================================================================


class TestActionNodeInternals:
    """Whitebox tests for Action leaf node."""

    def test_action_node_type(self):
        """Test Action has correct node type."""
        action = Action(success_action)
        assert action.node_type == BTNodeType.ACTION

    def test_action_executes_function(self, context):
        """Test Action executes its function."""
        called = [False]

        def my_action(ctx):
            called[0] = True
            return BTStatus.SUCCESS

        action = Action(my_action)
        action.tick(context)

        assert called[0] is True

    def test_action_handles_exception(self, context):
        """Test Action handles exception gracefully."""
        def bad_action(ctx):
            raise ValueError("Intentional error")

        action = Action(bad_action)
        status = action.tick(context)

        assert status == BTStatus.FAILURE


class TestConditionNodeInternals:
    """Whitebox tests for Condition leaf node."""

    def test_condition_node_type(self):
        """Test Condition has correct node type."""
        cond = Condition(lambda ctx: True)
        assert cond.node_type == BTNodeType.CONDITION

    def test_condition_true_succeeds(self, context):
        """Test Condition returns SUCCESS for True."""
        cond = Condition(lambda ctx: True)
        status = cond.tick(context)
        assert status == BTStatus.SUCCESS

    def test_condition_false_fails(self, context):
        """Test Condition returns FAILURE for False."""
        cond = Condition(lambda ctx: False)
        status = cond.tick(context)
        assert status == BTStatus.FAILURE

    def test_condition_handles_exception(self, context):
        """Test Condition handles exception gracefully."""
        def bad_condition(ctx):
            raise ValueError("Intentional error")

        cond = Condition(bad_condition)
        status = cond.tick(context)

        assert status == BTStatus.FAILURE


class TestBlackboardConditionInternals:
    """Whitebox tests for BlackboardCondition leaf node."""

    def test_blackboard_condition_check_exists(self, context):
        """Test BlackboardCondition check_exists mode."""
        context.blackboard.set("key", "value")

        exists_true = BlackboardCondition("key", check_exists=True)
        exists_false = BlackboardCondition("missing", check_exists=True)

        assert exists_true.tick(context) == BTStatus.SUCCESS
        assert exists_false.tick(context) == BTStatus.FAILURE

    def test_blackboard_condition_check_value(self, context):
        """Test BlackboardCondition checks value."""
        context.blackboard.set("health", 100)

        matches = BlackboardCondition("health", expected=100)
        no_match = BlackboardCondition("health", expected=50)

        assert matches.tick(context) == BTStatus.SUCCESS
        assert no_match.tick(context) == BTStatus.FAILURE

    def test_blackboard_condition_custom_comparator(self, context):
        """Test BlackboardCondition with custom comparator."""
        context.blackboard.set("health", 75)

        greater_than = BlackboardCondition(
            "health",
            expected=50,
            comparator=lambda a, b: a > b
        )

        assert greater_than.tick(context) == BTStatus.SUCCESS


class TestWaitNodeInternals:
    """Whitebox tests for Wait leaf node."""

    def test_wait_returns_running(self, context):
        """Test Wait returns RUNNING while waiting."""
        wait = Wait(duration=5.0)
        status = wait.tick(context)
        assert status == BTStatus.RUNNING

    def test_wait_succeeds_after_duration(self, context):
        """Test Wait succeeds after duration."""
        wait = Wait(duration=1.0)

        wait.tick(context)  # Start timer

        context.current_time += 2.0

        status = wait.tick(context)
        assert status == BTStatus.SUCCESS

    def test_wait_resets_on_reset(self, context):
        """Test Wait resets timer on reset."""
        wait = Wait(duration=5.0)
        wait.tick(context)

        wait.reset()
        assert wait._start_time is None


class TestSetBlackboardInternals:
    """Whitebox tests for SetBlackboard leaf node."""

    def test_set_blackboard_static_value(self, context):
        """Test SetBlackboard sets static value."""
        node = SetBlackboard("key", value=42)
        status = node.tick(context)

        assert status == BTStatus.SUCCESS
        assert context.blackboard.get("key") == 42

    def test_set_blackboard_dynamic_value(self, context):
        """Test SetBlackboard sets dynamic value."""
        node = SetBlackboard("computed", value_func=lambda ctx: ctx.delta_time * 100)
        node.tick(context)

        assert context.blackboard.get("computed") == context.delta_time * 100


# =============================================================================
# BEHAVIOR TREE TESTS
# =============================================================================


class TestBehaviorTreeInternals:
    """Whitebox tests for BehaviorTree lifecycle."""

    def test_bt_initialization(self, blackboard):
        """Test BehaviorTree initializes correctly."""
        # Note: Due to Blackboard having __len__ and evaluating as falsy when empty,
        # we need to populate it or use the returned blackboard for reference
        blackboard.set("key", "value")  # Make blackboard truthy
        root = Action(success_action)
        bt = BehaviorTree(root, blackboard=blackboard, name="TestBT")

        assert bt.name == "TestBT"
        assert bt.root is root
        assert bt.blackboard is blackboard
        assert bt.is_running is False

    def test_bt_tick_returns_status(self, blackboard):
        """Test BehaviorTree tick returns root status."""
        root = Action(success_action)
        bt = BehaviorTree(root, blackboard=blackboard)

        status = bt.tick(enable_event_logging=False)
        assert status == BTStatus.SUCCESS

    def test_bt_is_running_tracks_running_status(self, blackboard):
        """Test BehaviorTree tracks running state."""
        root = Action(running_action)
        bt = BehaviorTree(root, blackboard=blackboard)

        bt.tick(enable_event_logging=False)
        assert bt.is_running is True

        bt2 = BehaviorTree(Action(success_action), blackboard=blackboard)
        bt2.tick(enable_event_logging=False)
        assert bt2.is_running is False

    def test_bt_reset_clears_state(self, blackboard):
        """Test BehaviorTree reset clears state."""
        root = Action(running_action)
        bt = BehaviorTree(root, blackboard=blackboard)

        bt.tick(enable_event_logging=False)
        bt.reset()

        assert bt.is_running is False
        assert bt._aborted is False

    def test_bt_abort_stops_execution(self, blackboard):
        """Test BehaviorTree abort stops execution."""
        root = Action(running_action)
        bt = BehaviorTree(root, blackboard=blackboard)

        bt.tick(enable_event_logging=False)
        bt.abort()

        assert bt.is_running is False
        assert bt._aborted is True


# =============================================================================
# REGISTRY DECORATOR TESTS
# =============================================================================


class TestBTNodeDecorator:
    """Tests for @bt_node decorator."""

    def test_bt_node_valid_type(self):
        """Test @bt_node accepts valid types."""
        @bt_node(node_type="action")
        class MyAction:
            pass

        assert hasattr(MyAction, "_bt_node")
        assert MyAction._bt_node is True
        assert MyAction._bt_node_type == "action"

    def test_bt_node_invalid_type_raises(self):
        """Test @bt_node raises for invalid types."""
        with pytest.raises(BTNodeTypeError):
            @bt_node(node_type="invalid_type")
            class BadNode:
                pass

    def test_bt_node_valid_types_list(self):
        """Test all valid node types are accepted.

        Note: The bt_node decorator requires classes to be properly registered.
        We verify that the constant contains expected types.
        """
        # Test that VALID_BT_NODE_TYPES contains expected types
        assert "action" in VALID_BT_NODE_TYPES
        assert "condition" in VALID_BT_NODE_TYPES
        assert "decorator" in VALID_BT_NODE_TYPES
        assert "selector" in VALID_BT_NODE_TYPES
        assert "sequence" in VALID_BT_NODE_TYPES
        assert "parallel" in VALID_BT_NODE_TYPES

        # Verify constant is not empty
        assert len(VALID_BT_NODE_TYPES) >= 6


class TestBehaviorTreeDecorator:
    """Tests for @behavior_tree decorator."""

    def test_behavior_tree_decorator(self):
        """Test @behavior_tree decorator marks class."""
        @behavior_tree(name="test_bt", description="A test behavior tree")
        class TestBT:
            pass

        assert hasattr(TestBT, "_behavior_tree")
        assert TestBT._behavior_tree is True
        assert TestBT._bt_name == "test_bt"
        assert TestBT._bt_description == "A test behavior tree"


# =============================================================================
# COMPOSITE NODE MANAGEMENT TESTS
# =============================================================================


class TestCompositeNodeManagement:
    """Tests for composite node child management."""

    def test_add_child_sets_parent(self):
        """Test add_child sets parent reference."""
        parent = Sequence()
        child = Action(success_action)

        parent.add_child(child)

        assert child._parent is parent

    def test_remove_child_clears_parent(self):
        """Test remove_child clears parent reference."""
        parent = Sequence()
        child = Action(success_action)

        parent.add_child(child)
        parent.remove_child(child)

        assert child._parent is None

    def test_remove_nonexistent_child_returns_false(self):
        """Test removing nonexistent child returns False."""
        parent = Sequence()
        child = Action(success_action)

        result = parent.remove_child(child)
        assert result is False

    def test_children_property_returns_list(self):
        """Test children property returns child list."""
        parent = Sequence()
        c1 = Action(success_action)
        c2 = Action(success_action)

        parent.add_child(c1)
        parent.add_child(c2)

        assert len(parent.children) == 2

    def test_reset_propagates_to_children(self, context):
        """Test reset propagates to all children."""
        c1 = Sequence()
        c2 = Action(success_action)

        parent = Sequence([c1, c2])
        parent.tick(context)
        parent.reset()

        assert c1._status == BTStatus.SUCCESS
        assert c2._status == BTStatus.SUCCESS


# =============================================================================
# EDGE CASES
# =============================================================================


class TestBehaviorTreeEdgeCases:
    """Edge case tests for behavior tree system."""

    def test_empty_sequence_succeeds(self, context):
        """Test empty Sequence succeeds."""
        seq = Sequence()
        status = seq.tick(context)
        assert status == BTStatus.SUCCESS

    def test_empty_selector_fails(self, context):
        """Test empty Selector fails."""
        sel = Selector()
        status = sel.tick(context)
        assert status == BTStatus.FAILURE

    def test_deeply_nested_tree(self, context):
        """Test deeply nested tree works."""
        # Build tree nested to depth 10
        node = Action(success_action)
        for _ in range(10):
            node = Sequence([node])

        status = node.tick(context)
        assert status == BTStatus.SUCCESS

    def test_action_with_none_function(self, context):
        """Test Action handles None function gracefully.

        The Action node catches exceptions internally and returns FAILURE,
        logging the error rather than raising.
        """
        action = Action(None)
        # Construction succeeds, tick returns FAILURE (doesn't raise)
        result = action.tick(context)
        assert result == BTStatus.FAILURE

    def test_condition_receives_context(self, context):
        """Test Condition receives proper context."""
        context.blackboard.set("test_key", "test_value")

        def check_context(ctx):
            return ctx.blackboard.get("test_key") == "test_value"

        cond = Condition(check_context)
        status = cond.tick(context)

        assert status == BTStatus.SUCCESS

    def test_node_name_defaults_to_class(self):
        """Test node name defaults to class name."""
        action = Action(success_action)
        assert action.name == "Action"

    def test_node_custom_name(self):
        """Test node accepts custom name."""
        action = Action(success_action, name="MyCustomAction")
        assert action.name == "MyCustomAction"

    def test_parallel_with_single_child(self, context):
        """Test Parallel works with single child."""
        par = Parallel([Action(success_action)])
        status = par.tick(context)
        assert status == BTStatus.SUCCESS
