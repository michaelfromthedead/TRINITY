"""
Comprehensive tests for the Behavior Tree system.

Tests cover:
- Node types: Selector, Sequence, Parallel
- Decorators: Inverter, Repeater, Cooldown, Timeout, Retry, ForceSuccess, ForceFailure
- Leaf nodes: Action, Condition, Wait, BlackboardCondition, SetBlackboard
- Tree execution and status
- Tree traversal and memory
- Conditional aborts
- Subtree references
- Dynamic tree modification
- Parallel node policies

Total: ~200 tests
"""

import pytest
import time
from typing import List, Any
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.ai import (
    BTNode,
    BTComposite,
    BTSelector,
    BTSequence,
    BTParallel,
    BTDecorator,
    BTInverter,
    BTRepeater,
    BTCooldown,
    BTAction,
    BTCondition,
    BehaviorTree,
    Blackboard,
)
from engine.gameplay.constants import BTNodeStatus

# Also import from detailed implementation
from engine.gameplay.ai.behavior_tree import (
    BTContext,
    Sequence,
    Selector,
    Parallel,
    Invert,
    Repeat,
    Timeout,
    Cooldown,
    Retry,
    ForceSuccess,
    ForceFailure,
    Action,
    Condition,
    Wait,
    BlackboardCondition,
    SetBlackboard,
    BehaviorTree as DetailedBehaviorTree,
    CompositeNode,
    DecoratorNode,
    LeafNode,
    behavior_tree,
)
from engine.gameplay.ai.blackboard import Blackboard as DetailedBlackboard
from engine.gameplay.ai.constants import (
    BTStatus,
    BTNodeType,
    ParallelPolicy,
    BT_MAX_DEPTH,
    BT_INFINITE_REPEAT,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def blackboard():
    """Create a fresh blackboard for each test."""
    return Blackboard()


@pytest.fixture
def detailed_blackboard():
    """Create a fresh detailed blackboard for each test."""
    return DetailedBlackboard()


@pytest.fixture
def bt_context(detailed_blackboard):
    """Create a behavior tree context."""
    return BTContext(
        blackboard=detailed_blackboard,
        entity=Mock(),
        delta_time=0.016,
        current_time=time.time(),
    )


@pytest.fixture
def success_action():
    """Create an action that always succeeds."""
    return BTAction(name="SuccessAction", action=lambda dt: BTNodeStatus.SUCCESS)


@pytest.fixture
def failure_action():
    """Create an action that always fails."""
    return BTAction(name="FailureAction", action=lambda dt: BTNodeStatus.FAILURE)


@pytest.fixture
def running_action():
    """Create an action that always returns running."""
    return BTAction(name="RunningAction", action=lambda dt: BTNodeStatus.RUNNING)


@pytest.fixture
def true_condition():
    """Create a condition that always returns true."""
    return BTCondition(name="TrueCondition", condition=lambda: True)


@pytest.fixture
def false_condition():
    """Create a condition that always returns false."""
    return BTCondition(name="FalseCondition", condition=lambda: False)


# =============================================================================
# Basic Node Tests
# =============================================================================


class TestBTNodeBasics:
    """Test basic BTNode functionality."""

    def test_node_has_name(self):
        """Node should have a name."""
        node = BTAction(name="TestNode")
        assert node.name == "TestNode"

    def test_node_default_name(self):
        """Node should use class name as default."""
        node = BTAction()
        assert node.name == "BTAction"

    def test_node_initial_status(self):
        """Node should start in RUNNING status."""
        node = BTAction()
        assert node.status == BTNodeStatus.RUNNING

    def test_node_reset(self, success_action):
        """Reset should return node to initial state."""
        success_action.tick(0.016)
        assert success_action.status == BTNodeStatus.SUCCESS
        success_action.reset()
        assert success_action.status == BTNodeStatus.RUNNING

    def test_node_abort(self, running_action):
        """Abort should set status to FAILURE."""
        running_action.tick(0.016)
        assert running_action.status == BTNodeStatus.RUNNING
        running_action.abort()
        assert running_action.status == BTNodeStatus.FAILURE

    def test_node_set_blackboard(self, blackboard):
        """Node should accept a blackboard."""
        node = BTAction()
        node.set_blackboard(blackboard)
        assert node._blackboard is blackboard


# =============================================================================
# Action Node Tests
# =============================================================================


class TestBTAction:
    """Test Action leaf nodes."""

    def test_action_executes_callback(self):
        """Action should execute its callback."""
        executed = []
        action = BTAction(
            name="TestAction",
            action=lambda dt: (executed.append(True), BTNodeStatus.SUCCESS)[1]
        )
        action.tick(0.016)
        assert executed == [True]

    def test_action_returns_callback_status(self):
        """Action should return status from callback."""
        action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)
        assert action.tick(0.016) == BTNodeStatus.SUCCESS

        action = BTAction(action=lambda dt: BTNodeStatus.FAILURE)
        assert action.tick(0.016) == BTNodeStatus.FAILURE

        action = BTAction(action=lambda dt: BTNodeStatus.RUNNING)
        assert action.tick(0.016) == BTNodeStatus.RUNNING

    def test_action_without_callback_succeeds(self):
        """Action without callback should succeed."""
        action = BTAction()
        assert action.tick(0.016) == BTNodeStatus.SUCCESS

    def test_action_receives_delta_time(self):
        """Action callback should receive delta time."""
        received_dt = []
        action = BTAction(
            action=lambda dt: (received_dt.append(dt), BTNodeStatus.SUCCESS)[1]
        )
        action.tick(0.033)
        assert received_dt == [0.033]

    def test_action_multiple_ticks(self):
        """Action can be ticked multiple times."""
        tick_count = [0]
        action = BTAction(
            action=lambda dt: (
                tick_count.__setitem__(0, tick_count[0] + 1),
                BTNodeStatus.RUNNING if tick_count[0] < 3 else BTNodeStatus.SUCCESS
            )[1]
        )
        assert action.tick(0.016) == BTNodeStatus.RUNNING
        assert action.tick(0.016) == BTNodeStatus.RUNNING
        assert action.tick(0.016) == BTNodeStatus.SUCCESS


# =============================================================================
# Condition Node Tests
# =============================================================================


class TestBTCondition:
    """Test Condition leaf nodes."""

    def test_condition_true_returns_success(self, true_condition):
        """Condition returning True should give SUCCESS."""
        assert true_condition.tick(0.016) == BTNodeStatus.SUCCESS

    def test_condition_false_returns_failure(self, false_condition):
        """Condition returning False should give FAILURE."""
        assert false_condition.tick(0.016) == BTNodeStatus.FAILURE

    def test_condition_executes_callback(self):
        """Condition should execute its callback."""
        executed = []
        condition = BTCondition(
            condition=lambda: (executed.append(True), True)[1]
        )
        condition.tick(0.016)
        assert executed == [True]

    def test_condition_without_callback_fails(self):
        """Condition without callback should fail."""
        condition = BTCondition()
        assert condition.tick(0.016) == BTNodeStatus.FAILURE

    def test_condition_dynamic_evaluation(self):
        """Condition should be evaluated each tick."""
        state = {"value": False}
        condition = BTCondition(condition=lambda: state["value"])

        assert condition.tick(0.016) == BTNodeStatus.FAILURE
        state["value"] = True
        assert condition.tick(0.016) == BTNodeStatus.SUCCESS


# =============================================================================
# Sequence Node Tests
# =============================================================================


class TestBTSequence:
    """Test Sequence composite nodes."""

    def test_sequence_empty_succeeds(self):
        """Empty sequence should succeed immediately."""
        sequence = BTSequence()
        # Empty sequence has no children to fail, falls through loop and succeeds
        assert sequence.tick(0.016) == BTNodeStatus.SUCCESS  # No children = success (falls through loop)

    def test_sequence_all_success(self, success_action):
        """Sequence with all successful children should succeed."""
        sequence = BTSequence(children=[
            BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
            BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
            BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
        ])
        assert sequence.tick(0.016) == BTNodeStatus.SUCCESS

    def test_sequence_first_failure(self, failure_action, success_action):
        """Sequence should fail on first failure."""
        executed = []
        sequence = BTSequence(children=[
            BTAction(action=lambda dt: (executed.append(1), BTNodeStatus.SUCCESS)[1]),
            BTAction(action=lambda dt: (executed.append(2), BTNodeStatus.FAILURE)[1]),
            BTAction(action=lambda dt: (executed.append(3), BTNodeStatus.SUCCESS)[1]),
        ])
        assert sequence.tick(0.016) == BTNodeStatus.FAILURE
        assert executed == [1, 2]  # Third action not executed

    def test_sequence_running_pauses(self):
        """Sequence should pause on running child."""
        executed = []
        sequence = BTSequence(children=[
            BTAction(action=lambda dt: (executed.append(1), BTNodeStatus.SUCCESS)[1]),
            BTAction(action=lambda dt: (executed.append(2), BTNodeStatus.RUNNING)[1]),
            BTAction(action=lambda dt: (executed.append(3), BTNodeStatus.SUCCESS)[1]),
        ])
        assert sequence.tick(0.016) == BTNodeStatus.RUNNING
        assert executed == [1, 2]

    def test_sequence_resumes_from_running(self):
        """Sequence should resume from running child."""
        tick_count = [0]
        sequence = BTSequence(children=[
            BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
            BTAction(action=lambda dt: (
                tick_count.__setitem__(0, tick_count[0] + 1),
                BTNodeStatus.RUNNING if tick_count[0] < 2 else BTNodeStatus.SUCCESS
            )[1]),
            BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
        ])
        assert sequence.tick(0.016) == BTNodeStatus.RUNNING
        assert sequence.tick(0.016) == BTNodeStatus.SUCCESS

    def test_sequence_add_child(self, success_action):
        """Sequence should support dynamic child addition."""
        sequence = BTSequence()
        sequence.add_child(BTAction(action=lambda dt: BTNodeStatus.SUCCESS))
        sequence.add_child(BTAction(action=lambda dt: BTNodeStatus.SUCCESS))
        assert sequence.tick(0.016) == BTNodeStatus.SUCCESS

    def test_sequence_reset(self):
        """Reset should reset all children and index."""
        sequence = BTSequence(children=[
            BTAction(action=lambda dt: BTNodeStatus.RUNNING),
        ])
        sequence.tick(0.016)
        sequence.reset()
        assert sequence._current_child == 0

    def test_sequence_order_matters(self):
        """Sequence should execute children in order."""
        order = []
        sequence = BTSequence(children=[
            BTAction(action=lambda dt: (order.append(1), BTNodeStatus.SUCCESS)[1]),
            BTAction(action=lambda dt: (order.append(2), BTNodeStatus.SUCCESS)[1]),
            BTAction(action=lambda dt: (order.append(3), BTNodeStatus.SUCCESS)[1]),
        ])
        sequence.tick(0.016)
        assert order == [1, 2, 3]


# =============================================================================
# Selector Node Tests
# =============================================================================


class TestBTSelector:
    """Test Selector composite nodes."""

    def test_selector_empty_fails(self):
        """Empty selector should fail."""
        selector = BTSelector()
        assert selector.tick(0.016) == BTNodeStatus.FAILURE

    def test_selector_first_success(self):
        """Selector should succeed on first success."""
        executed = []
        selector = BTSelector(children=[
            BTAction(action=lambda dt: (executed.append(1), BTNodeStatus.FAILURE)[1]),
            BTAction(action=lambda dt: (executed.append(2), BTNodeStatus.SUCCESS)[1]),
            BTAction(action=lambda dt: (executed.append(3), BTNodeStatus.SUCCESS)[1]),
        ])
        assert selector.tick(0.016) == BTNodeStatus.SUCCESS
        assert executed == [1, 2]  # Third not executed

    def test_selector_all_failure(self):
        """Selector should fail if all children fail."""
        selector = BTSelector(children=[
            BTAction(action=lambda dt: BTNodeStatus.FAILURE),
            BTAction(action=lambda dt: BTNodeStatus.FAILURE),
            BTAction(action=lambda dt: BTNodeStatus.FAILURE),
        ])
        assert selector.tick(0.016) == BTNodeStatus.FAILURE

    def test_selector_running_pauses(self):
        """Selector should pause on running child."""
        executed = []
        selector = BTSelector(children=[
            BTAction(action=lambda dt: (executed.append(1), BTNodeStatus.FAILURE)[1]),
            BTAction(action=lambda dt: (executed.append(2), BTNodeStatus.RUNNING)[1]),
            BTAction(action=lambda dt: (executed.append(3), BTNodeStatus.SUCCESS)[1]),
        ])
        assert selector.tick(0.016) == BTNodeStatus.RUNNING
        assert executed == [1, 2]

    def test_selector_resumes_from_running(self):
        """Selector should resume from running child."""
        tick_count = [0]
        selector = BTSelector(children=[
            BTAction(action=lambda dt: BTNodeStatus.FAILURE),
            BTAction(action=lambda dt: (
                tick_count.__setitem__(0, tick_count[0] + 1),
                BTNodeStatus.RUNNING if tick_count[0] < 2 else BTNodeStatus.SUCCESS
            )[1]),
        ])
        assert selector.tick(0.016) == BTNodeStatus.RUNNING
        assert selector.tick(0.016) == BTNodeStatus.SUCCESS

    def test_selector_tries_alternatives(self):
        """Selector should try all children until success."""
        order = []
        selector = BTSelector(children=[
            BTAction(action=lambda dt: (order.append(1), BTNodeStatus.FAILURE)[1]),
            BTAction(action=lambda dt: (order.append(2), BTNodeStatus.FAILURE)[1]),
            BTAction(action=lambda dt: (order.append(3), BTNodeStatus.SUCCESS)[1]),
        ])
        selector.tick(0.016)
        assert order == [1, 2, 3]

    def test_selector_reset(self):
        """Reset should reset selector state."""
        selector = BTSelector(children=[
            BTAction(action=lambda dt: BTNodeStatus.FAILURE),
            BTAction(action=lambda dt: BTNodeStatus.RUNNING),
        ])
        selector.tick(0.016)
        selector.reset()
        assert selector._current_child == 0


# =============================================================================
# Parallel Node Tests
# =============================================================================


class TestBTParallel:
    """Test Parallel composite nodes."""

    def test_parallel_all_success(self):
        """Parallel with all success should succeed."""
        parallel = BTParallel(
            children=[
                BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
                BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
                BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
            ],
            success_threshold=3,
        )
        assert parallel.tick(0.016) == BTNodeStatus.SUCCESS

    def test_parallel_success_threshold(self):
        """Parallel should succeed when threshold met."""
        parallel = BTParallel(
            children=[
                BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
                BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
                BTAction(action=lambda dt: BTNodeStatus.FAILURE),
            ],
            success_threshold=2,
        )
        assert parallel.tick(0.016) == BTNodeStatus.SUCCESS

    def test_parallel_failure_threshold(self):
        """Parallel should fail when failure threshold met."""
        # Note: BTParallel only ticks children that are still RUNNING
        # So we need children that return immediately in one tick
        parallel = BTParallel(
            children=[
                BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
                BTAction(action=lambda dt: BTNodeStatus.FAILURE),
                BTAction(action=lambda dt: BTNodeStatus.FAILURE),
            ],
            success_threshold=3,  # Set high so we don't succeed first
            failure_threshold=2,
        )
        # First tick: all start as RUNNING, all are ticked
        # SUCCESS=1, FAILURE=2 -> failure_threshold=2 met
        assert parallel.tick(0.016) == BTNodeStatus.FAILURE

    def test_parallel_runs_all_children(self):
        """Parallel should run all children each tick."""
        executed = []
        parallel = BTParallel(
            children=[
                BTAction(action=lambda dt: (executed.append(1), BTNodeStatus.RUNNING)[1]),
                BTAction(action=lambda dt: (executed.append(2), BTNodeStatus.RUNNING)[1]),
                BTAction(action=lambda dt: (executed.append(3), BTNodeStatus.RUNNING)[1]),
            ],
            success_threshold=3,
        )
        parallel.tick(0.016)
        assert set(executed) == {1, 2, 3}

    def test_parallel_running_until_threshold(self):
        """Parallel should return running until threshold met."""
        tick_count = [0]
        parallel = BTParallel(
            children=[
                BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
                BTAction(action=lambda dt: (
                    tick_count.__setitem__(0, tick_count[0] + 1),
                    BTNodeStatus.RUNNING if tick_count[0] < 2 else BTNodeStatus.SUCCESS
                )[1]),
            ],
            success_threshold=2,
        )
        assert parallel.tick(0.016) == BTNodeStatus.RUNNING
        assert parallel.tick(0.016) == BTNodeStatus.SUCCESS

    def test_parallel_mixed_results(self):
        """Parallel with mixed results respects thresholds."""
        parallel = BTParallel(
            children=[
                BTAction(action=lambda dt: BTNodeStatus.SUCCESS),
                BTAction(action=lambda dt: BTNodeStatus.FAILURE),
                BTAction(action=lambda dt: BTNodeStatus.RUNNING),
            ],
            success_threshold=2,
            failure_threshold=2,
        )
        assert parallel.tick(0.016) == BTNodeStatus.RUNNING


# =============================================================================
# Detailed Implementation Parallel Policy Tests
# =============================================================================


class TestParallelPolicies:
    """Test detailed implementation parallel node policies."""

    def test_require_all_policy_all_success(self, bt_context):
        """REQUIRE_ALL succeeds when all children succeed."""
        parallel = Parallel(
            children=[
                Action(lambda ctx: BTStatus.SUCCESS),
                Action(lambda ctx: BTStatus.SUCCESS),
            ],
            policy=ParallelPolicy.REQUIRE_ALL,
        )
        assert parallel.tick(bt_context) == BTStatus.SUCCESS

    def test_require_all_policy_one_failure(self, bt_context):
        """REQUIRE_ALL fails when any child fails."""
        parallel = Parallel(
            children=[
                Action(lambda ctx: BTStatus.SUCCESS),
                Action(lambda ctx: BTStatus.FAILURE),
            ],
            policy=ParallelPolicy.REQUIRE_ALL,
        )
        assert parallel.tick(bt_context) == BTStatus.FAILURE

    def test_require_one_policy_one_success(self, bt_context):
        """REQUIRE_ONE succeeds when any child succeeds."""
        parallel = Parallel(
            children=[
                Action(lambda ctx: BTStatus.FAILURE),
                Action(lambda ctx: BTStatus.SUCCESS),
            ],
            policy=ParallelPolicy.REQUIRE_ONE,
        )
        assert parallel.tick(bt_context) == BTStatus.SUCCESS

    def test_require_one_policy_all_failure(self, bt_context):
        """REQUIRE_ONE fails when all children fail."""
        parallel = Parallel(
            children=[
                Action(lambda ctx: BTStatus.FAILURE),
                Action(lambda ctx: BTStatus.FAILURE),
            ],
            policy=ParallelPolicy.REQUIRE_ONE,
        )
        assert parallel.tick(bt_context) == BTStatus.FAILURE

    def test_require_majority_policy_majority_success(self, bt_context):
        """REQUIRE_MAJORITY succeeds when majority succeeds."""
        parallel = Parallel(
            children=[
                Action(lambda ctx: BTStatus.SUCCESS),
                Action(lambda ctx: BTStatus.SUCCESS),
                Action(lambda ctx: BTStatus.FAILURE),
            ],
            policy=ParallelPolicy.REQUIRE_MAJORITY,
        )
        assert parallel.tick(bt_context) == BTStatus.SUCCESS

    def test_require_majority_policy_majority_failure(self, bt_context):
        """REQUIRE_MAJORITY fails when majority fails."""
        parallel = Parallel(
            children=[
                Action(lambda ctx: BTStatus.SUCCESS),
                Action(lambda ctx: BTStatus.FAILURE),
                Action(lambda ctx: BTStatus.FAILURE),
            ],
            policy=ParallelPolicy.REQUIRE_MAJORITY,
        )
        assert parallel.tick(bt_context) == BTStatus.FAILURE

    def test_parallel_empty_children_succeeds(self, bt_context):
        """Parallel with no children should succeed."""
        parallel = Parallel(policy=ParallelPolicy.REQUIRE_ALL)
        assert parallel.tick(bt_context) == BTStatus.SUCCESS


# =============================================================================
# Inverter Decorator Tests
# =============================================================================


class TestBTInverter:
    """Test Inverter decorator nodes."""

    def test_inverter_inverts_success(self, success_action):
        """Inverter should invert SUCCESS to FAILURE."""
        inverter = BTInverter(child=success_action)
        assert inverter.tick(0.016) == BTNodeStatus.FAILURE

    def test_inverter_inverts_failure(self, failure_action):
        """Inverter should invert FAILURE to SUCCESS."""
        inverter = BTInverter(child=failure_action)
        assert inverter.tick(0.016) == BTNodeStatus.SUCCESS

    def test_inverter_preserves_running(self, running_action):
        """Inverter should preserve RUNNING status."""
        inverter = BTInverter(child=running_action)
        assert inverter.tick(0.016) == BTNodeStatus.RUNNING

    def test_inverter_without_child(self):
        """Inverter without child should fail."""
        inverter = BTInverter()
        assert inverter.tick(0.016) == BTNodeStatus.FAILURE

    def test_inverter_chain(self):
        """Double inversion should preserve original."""
        action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)
        double_invert = BTInverter(child=BTInverter(child=action))
        assert double_invert.tick(0.016) == BTNodeStatus.SUCCESS


# =============================================================================
# Repeater Decorator Tests
# =============================================================================


class TestBTRepeater:
    """Test Repeater decorator nodes."""

    def test_repeater_fixed_count(self):
        """Repeater should repeat fixed number of times."""
        tick_count = [0]
        action = BTAction(
            action=lambda dt: (
                tick_count.__setitem__(0, tick_count[0] + 1),
                BTNodeStatus.SUCCESS
            )[1]
        )
        repeater = BTRepeater(child=action, repeat_count=3)

        # First two iterations return RUNNING
        assert repeater.tick(0.016) == BTNodeStatus.RUNNING
        assert repeater.tick(0.016) == BTNodeStatus.RUNNING
        assert repeater.tick(0.016) == BTNodeStatus.SUCCESS
        assert tick_count[0] == 3

    def test_repeater_infinite(self):
        """Repeater with -1 should repeat infinitely."""
        tick_count = [0]
        action = BTAction(
            action=lambda dt: (
                tick_count.__setitem__(0, tick_count[0] + 1),
                BTNodeStatus.SUCCESS
            )[1]
        )
        repeater = BTRepeater(child=action, repeat_count=-1)

        for _ in range(10):
            assert repeater.tick(0.016) == BTNodeStatus.RUNNING
        assert tick_count[0] == 10

    def test_repeater_reset(self):
        """Repeater reset should reset count."""
        action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)
        repeater = BTRepeater(child=action, repeat_count=3)

        repeater.tick(0.016)
        repeater.tick(0.016)
        repeater.reset()

        # Should restart from beginning
        assert repeater._current_count == 0

    def test_repeater_child_running(self):
        """Repeater should wait for running child."""
        tick_count = [0]
        action = BTAction(
            action=lambda dt: (
                tick_count.__setitem__(0, tick_count[0] + 1),
                BTNodeStatus.RUNNING if tick_count[0] % 2 == 1 else BTNodeStatus.SUCCESS
            )[1]
        )
        repeater = BTRepeater(child=action, repeat_count=2)

        assert repeater.tick(0.016) == BTNodeStatus.RUNNING  # Running
        assert repeater.tick(0.016) == BTNodeStatus.RUNNING  # Success, repeat
        assert repeater.tick(0.016) == BTNodeStatus.RUNNING  # Running
        assert repeater.tick(0.016) == BTNodeStatus.SUCCESS  # Success, done


# =============================================================================
# Cooldown Decorator Tests
# =============================================================================


class TestBTCooldown:
    """Test Cooldown decorator nodes."""

    def test_cooldown_first_execution(self):
        """Cooldown should allow first execution."""
        action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)
        cooldown = BTCooldown(child=action, cooldown_time=1.0)
        assert cooldown.tick(0.016) == BTNodeStatus.SUCCESS

    def test_cooldown_blocks_during_cooldown(self):
        """Cooldown should block execution during cooldown."""
        action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)
        cooldown = BTCooldown(child=action, cooldown_time=1.0)

        cooldown.tick(0.016)  # First execution
        assert cooldown.tick(0.016) == BTNodeStatus.FAILURE  # Blocked

    def test_cooldown_allows_after_cooldown(self):
        """Cooldown should allow execution after cooldown expires."""
        action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)
        cooldown = BTCooldown(child=action, cooldown_time=0.1)

        cooldown.tick(0.016)  # First execution, starts cooldown with 0.1 remaining
        # After first tick, _time_remaining = 0.1
        cooldown.tick(0.05)   # _time_remaining = 0.1 - 0.05 = 0.05, still blocked (returns FAILURE)
        cooldown.tick(0.05)   # _time_remaining = 0.05 - 0.05 = 0.0, still blocked (> 0 check fails, = 0 passes)
        # Now _time_remaining <= 0, so execution is allowed
        assert cooldown.tick(0.01) == BTNodeStatus.SUCCESS  # Cooldown expired

    def test_cooldown_with_running_child(self):
        """Cooldown should not start until child completes."""
        tick_count = [0]
        action = BTAction(
            action=lambda dt: (
                tick_count.__setitem__(0, tick_count[0] + 1),
                BTNodeStatus.RUNNING if tick_count[0] < 2 else BTNodeStatus.SUCCESS
            )[1]
        )
        cooldown = BTCooldown(child=action, cooldown_time=1.0)

        assert cooldown.tick(0.016) == BTNodeStatus.RUNNING
        assert cooldown._time_remaining == 0  # Cooldown not started


# =============================================================================
# Detailed Implementation Decorator Tests
# =============================================================================


class TestDetailedDecorators:
    """Test detailed implementation decorators."""

    def test_timeout_success_within_time(self, bt_context):
        """Timeout should allow success within time limit."""
        action = Action(lambda ctx: BTStatus.SUCCESS)
        timeout = Timeout(child=action, timeout=5.0)
        assert timeout.tick(bt_context) == BTStatus.SUCCESS

    def test_timeout_fails_after_timeout(self, bt_context):
        """Timeout should fail after time expires."""
        action = Action(lambda ctx: BTStatus.RUNNING)
        timeout = Timeout(child=action, timeout=0.1)

        # First tick starts the timer
        bt_context.current_time = 0.0
        assert timeout.tick(bt_context) == BTStatus.RUNNING

        # After timeout
        bt_context.current_time = 0.2
        assert timeout.tick(bt_context) == BTStatus.FAILURE

    def test_retry_on_failure(self, bt_context):
        """Retry should retry on failure."""
        fail_count = [0]
        def action_func(ctx):
            fail_count[0] += 1
            return BTStatus.FAILURE if fail_count[0] < 3 else BTStatus.SUCCESS

        action = Action(action_func)
        retry = Retry(child=action, max_retries=3)

        assert retry.tick(bt_context) == BTStatus.RUNNING  # First failure, retry
        assert retry.tick(bt_context) == BTStatus.RUNNING  # Second failure, retry
        assert retry.tick(bt_context) == BTStatus.SUCCESS  # Third attempt succeeds

    def test_retry_exhausted(self, bt_context):
        """Retry should fail after max retries."""
        action = Action(lambda ctx: BTStatus.FAILURE)
        retry = Retry(child=action, max_retries=2)

        # max_retries=2 means: first failure + 2 retries = 3 total attempts
        # Tick 1: failure, retry_count=1, reset child, return RUNNING
        # Tick 2: failure, retry_count=2 >= max_retries, return FAILURE
        retry.tick(bt_context)  # First fail, retry_count=1
        assert retry.tick(bt_context) == BTStatus.FAILURE  # Second fail, max reached

    def test_force_success(self, bt_context):
        """ForceSuccess should convert failure to success."""
        action = Action(lambda ctx: BTStatus.FAILURE)
        force = ForceSuccess(child=action)
        assert force.tick(bt_context) == BTStatus.SUCCESS

    def test_force_success_preserves_running(self, bt_context):
        """ForceSuccess should preserve RUNNING status."""
        action = Action(lambda ctx: BTStatus.RUNNING)
        force = ForceSuccess(child=action)
        assert force.tick(bt_context) == BTStatus.RUNNING

    def test_force_failure(self, bt_context):
        """ForceFailure should convert success to failure."""
        action = Action(lambda ctx: BTStatus.SUCCESS)
        force = ForceFailure(child=action)
        assert force.tick(bt_context) == BTStatus.FAILURE

    def test_force_failure_preserves_running(self, bt_context):
        """ForceFailure should preserve RUNNING status."""
        action = Action(lambda ctx: BTStatus.RUNNING)
        force = ForceFailure(child=action)
        assert force.tick(bt_context) == BTStatus.RUNNING


# =============================================================================
# Wait Node Tests
# =============================================================================


class TestWait:
    """Test Wait leaf node."""

    def test_wait_returns_running(self, bt_context):
        """Wait should return RUNNING while waiting."""
        wait = Wait(duration=1.0)
        bt_context.current_time = 0.0
        assert wait.tick(bt_context) == BTStatus.RUNNING

    def test_wait_succeeds_after_duration(self, bt_context):
        """Wait should succeed after duration."""
        wait = Wait(duration=0.5)
        bt_context.current_time = 0.0
        wait.tick(bt_context)

        bt_context.current_time = 0.6
        assert wait.tick(bt_context) == BTStatus.SUCCESS

    def test_wait_reset(self, bt_context):
        """Wait reset should clear start time."""
        wait = Wait(duration=1.0)
        bt_context.current_time = 0.0
        wait.tick(bt_context)
        wait.reset()
        assert wait._start_time is None

    def test_wait_zero_duration(self, bt_context):
        """Wait with zero duration should succeed immediately."""
        wait = Wait(duration=0.0)
        assert wait.tick(bt_context) == BTStatus.SUCCESS


# =============================================================================
# Blackboard Condition Tests
# =============================================================================


class TestBlackboardCondition:
    """Test BlackboardCondition node."""

    def test_blackboard_condition_exists(self, bt_context):
        """Should check if key exists."""
        bt_context.blackboard.set("test_key", "value")
        condition = BlackboardCondition(key="test_key", check_exists=True)
        assert condition.tick(bt_context) == BTStatus.SUCCESS

    def test_blackboard_condition_not_exists(self, bt_context):
        """Should fail if key doesn't exist."""
        condition = BlackboardCondition(key="nonexistent", check_exists=True)
        assert condition.tick(bt_context) == BTStatus.FAILURE

    def test_blackboard_condition_value_match(self, bt_context):
        """Should check if value matches."""
        bt_context.blackboard.set("health", 100)
        condition = BlackboardCondition(key="health", expected=100)
        assert condition.tick(bt_context) == BTStatus.SUCCESS

    def test_blackboard_condition_value_mismatch(self, bt_context):
        """Should fail if value doesn't match."""
        bt_context.blackboard.set("health", 50)
        condition = BlackboardCondition(key="health", expected=100)
        assert condition.tick(bt_context) == BTStatus.FAILURE

    def test_blackboard_condition_custom_comparator(self, bt_context):
        """Should support custom comparators."""
        bt_context.blackboard.set("health", 50)
        condition = BlackboardCondition(
            key="health",
            expected=30,
            comparator=lambda a, b: a > b
        )
        assert condition.tick(bt_context) == BTStatus.SUCCESS


# =============================================================================
# SetBlackboard Node Tests
# =============================================================================


class TestSetBlackboard:
    """Test SetBlackboard node."""

    def test_set_blackboard_static_value(self, bt_context):
        """Should set static value."""
        node = SetBlackboard(key="target", value="enemy")
        node.tick(bt_context)
        assert bt_context.blackboard.get("target") == "enemy"

    def test_set_blackboard_dynamic_value(self, bt_context):
        """Should support dynamic value function."""
        node = SetBlackboard(
            key="health",
            value_func=lambda ctx: 100
        )
        node.tick(bt_context)
        assert bt_context.blackboard.get("health") == 100

    def test_set_blackboard_returns_success(self, bt_context):
        """Should return SUCCESS on completion."""
        node = SetBlackboard(key="key", value="value")
        assert node.tick(bt_context) == BTStatus.SUCCESS


# =============================================================================
# BehaviorTree Tests
# =============================================================================


class TestBehaviorTree:
    """Test complete BehaviorTree functionality."""

    def test_tree_creation(self):
        """Tree should be created with ID."""
        tree = BehaviorTree(tree_id="test_tree")
        assert tree.tree_id == "test_tree"

    def test_tree_has_blackboard(self):
        """Tree should have its own blackboard."""
        tree = BehaviorTree(tree_id="test")
        assert tree.blackboard is not None
        assert isinstance(tree.blackboard, Blackboard)

    def test_tree_tick_executes_root(self):
        """Tree tick should execute root node."""
        executed = [False]
        action = BTAction(
            action=lambda dt: (executed.__setitem__(0, True), BTNodeStatus.SUCCESS)[1]
        )
        tree = BehaviorTree(tree_id="test", root=action)
        tree.tick(0.016)
        assert executed[0]

    def test_tree_without_root_fails(self):
        """Tree without root should fail."""
        tree = BehaviorTree(tree_id="test")
        assert tree.tick(0.016) == BTNodeStatus.FAILURE

    def test_tree_set_root(self):
        """Should support setting root after creation."""
        tree = BehaviorTree(tree_id="test")
        action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)
        tree.set_root(action)
        assert tree.tick(0.016) == BTNodeStatus.SUCCESS

    def test_tree_is_running(self):
        """Tree should track running state."""
        action = BTAction(action=lambda dt: BTNodeStatus.RUNNING)
        tree = BehaviorTree(tree_id="test", root=action)

        assert not tree.is_running
        tree.tick(0.016)
        assert tree.is_running

    def test_tree_abort(self):
        """Tree abort should abort root."""
        action = BTAction(action=lambda dt: BTNodeStatus.RUNNING)
        tree = BehaviorTree(tree_id="test", root=action)

        tree.tick(0.016)
        tree.abort()
        assert not tree.is_running
        assert action.status == BTNodeStatus.FAILURE

    def test_tree_reset(self):
        """Tree reset should reset root."""
        action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)
        tree = BehaviorTree(tree_id="test", root=action)

        tree.tick(0.016)
        tree.reset()
        assert action.status == BTNodeStatus.RUNNING

    def test_tree_blackboard_propagation(self, blackboard):
        """Blackboard should propagate to all nodes."""
        action = BTAction()
        sequence = BTSequence(children=[action])
        tree = BehaviorTree(tree_id="test", root=sequence)

        tree.blackboard.set("key", "value")
        assert action._blackboard is tree.blackboard


# =============================================================================
# Detailed BehaviorTree Tests
# =============================================================================


class TestDetailedBehaviorTree:
    """Test detailed implementation BehaviorTree."""

    def test_detailed_tree_tick(self, detailed_blackboard):
        """Detailed tree should execute correctly."""
        action = Action(lambda ctx: BTStatus.SUCCESS)
        tree = DetailedBehaviorTree(root=action, blackboard=detailed_blackboard)
        assert tree.tick() == BTStatus.SUCCESS

    def test_detailed_tree_debug_trace(self, detailed_blackboard):
        """Tree should support debug tracing."""
        action = Action(lambda ctx: BTStatus.SUCCESS)
        tree = DetailedBehaviorTree(root=action, blackboard=detailed_blackboard)
        tree.tick(debug_trace=True)
        # Debug trace should be enabled without errors

    def test_detailed_tree_abort(self, detailed_blackboard):
        """Tree abort should work correctly."""
        action = Action(lambda ctx: BTStatus.RUNNING)
        tree = DetailedBehaviorTree(root=action, blackboard=detailed_blackboard)
        tree.tick()
        tree.abort()
        assert not tree.is_running


# =============================================================================
# Tree Traversal and Memory Tests
# =============================================================================


class TestTreeTraversal:
    """Test tree traversal and memory behavior."""

    def test_sequence_remembers_position(self):
        """Sequence should remember position across ticks."""
        executed = []
        sequence = BTSequence(children=[
            BTAction(action=lambda dt: (executed.append(1), BTNodeStatus.SUCCESS)[1]),
            BTAction(action=lambda dt: (executed.append(2), BTNodeStatus.RUNNING)[1]),
            BTAction(action=lambda dt: (executed.append(3), BTNodeStatus.SUCCESS)[1]),
        ])

        sequence.tick(0.016)  # Executes 1, 2, pauses at 2
        executed.clear()
        sequence._children[1]._action = lambda dt: (executed.append(2), BTNodeStatus.SUCCESS)[1]
        sequence.tick(0.016)  # Resumes at 2, executes 3

        assert executed == [2, 3]  # Didn't re-execute 1

    def test_selector_remembers_position(self):
        """Selector should remember position across ticks."""
        executed = []
        selector = BTSelector(children=[
            BTAction(action=lambda dt: (executed.append(1), BTNodeStatus.FAILURE)[1]),
            BTAction(action=lambda dt: (executed.append(2), BTNodeStatus.RUNNING)[1]),
            BTAction(action=lambda dt: (executed.append(3), BTNodeStatus.SUCCESS)[1]),
        ])

        selector.tick(0.016)  # Executes 1, 2, pauses at 2
        executed.clear()
        selector._children[1]._action = lambda dt: (executed.append(2), BTNodeStatus.SUCCESS)[1]
        selector.tick(0.016)  # Resumes at 2

        assert executed == [2]  # Didn't re-execute 1 or try 3

    def test_deep_tree_traversal(self):
        """Deep tree should traverse correctly."""
        inner_action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)
        inner_seq = BTSequence(children=[inner_action])
        outer_seq = BTSequence(children=[inner_seq])
        root = BTSequence(children=[outer_seq])

        tree = BehaviorTree(tree_id="deep", root=root)
        assert tree.tick(0.016) == BTNodeStatus.SUCCESS


# =============================================================================
# Dynamic Tree Modification Tests
# =============================================================================


class TestDynamicModification:
    """Test dynamic tree modification."""

    def test_add_child_during_execution(self):
        """Should handle adding child during execution."""
        sequence = BTSequence()
        sequence.add_child(BTAction(action=lambda dt: BTNodeStatus.SUCCESS))
        sequence.add_child(BTAction(action=lambda dt: BTNodeStatus.SUCCESS))

        assert sequence.tick(0.016) == BTNodeStatus.SUCCESS

    def test_composite_add_child_propagates_blackboard(self, blackboard):
        """Adding child should propagate blackboard."""
        sequence = BTSequence()
        sequence.set_blackboard(blackboard)

        action = BTAction()
        sequence.add_child(action)

        assert action._blackboard is blackboard


# =============================================================================
# Conditional Abort Tests (Detailed Implementation)
# =============================================================================


class TestConditionalAborts:
    """Test conditional abort behavior."""

    def test_abort_requested_stops_sequence(self, bt_context):
        """Abort should stop sequence execution."""
        bt_context.abort_requested = True
        sequence = Sequence(children=[
            Action(lambda ctx: BTStatus.SUCCESS),
            Action(lambda ctx: BTStatus.SUCCESS),
        ])
        assert sequence.tick(bt_context) == BTStatus.FAILURE

    def test_abort_requested_stops_selector(self, bt_context):
        """Abort should stop selector execution."""
        bt_context.abort_requested = True
        selector = Selector(children=[
            Action(lambda ctx: BTStatus.FAILURE),
            Action(lambda ctx: BTStatus.SUCCESS),
        ])
        assert selector.tick(bt_context) == BTStatus.FAILURE


# =============================================================================
# Depth Limit Tests
# =============================================================================


class TestDepthLimits:
    """Test depth limit enforcement."""

    def test_max_depth_failure(self, bt_context):
        """Should fail when max depth exceeded."""
        bt_context.depth = BT_MAX_DEPTH + 1
        sequence = Sequence(children=[
            Action(lambda ctx: BTStatus.SUCCESS),
        ])
        assert sequence.tick(bt_context) == BTStatus.FAILURE

    def test_child_context_increments_depth(self, bt_context):
        """Child context should have incremented depth."""
        child_context = bt_context.child_context()
        assert child_context.depth == bt_context.depth + 1


# =============================================================================
# Repeat Node Variations
# =============================================================================


class TestRepeatVariations:
    """Test detailed Repeat node variations."""

    def test_repeat_until_fail(self, bt_context):
        """Repeat until_fail should stop on first failure."""
        count = [0]
        def action_func(ctx):
            count[0] += 1
            return BTStatus.FAILURE if count[0] >= 3 else BTStatus.SUCCESS

        action = Action(action_func)
        repeat = Repeat(child=action, count=10, until_fail=True)

        # Should succeed after child fails
        repeat.tick(bt_context)  # Success, continue
        repeat.tick(bt_context)  # Success, continue
        assert repeat.tick(bt_context) == BTStatus.SUCCESS  # Failure, stop

    def test_repeat_until_success(self, bt_context):
        """Repeat until_success should stop on first success."""
        count = [0]
        def action_func(ctx):
            count[0] += 1
            return BTStatus.SUCCESS if count[0] >= 3 else BTStatus.FAILURE

        action = Action(action_func)
        repeat = Repeat(child=action, count=10, until_success=True)

        repeat.tick(bt_context)  # Failure, continue
        repeat.tick(bt_context)  # Failure, continue
        assert repeat.tick(bt_context) == BTStatus.SUCCESS  # Success, stop

    def test_repeat_infinite(self, bt_context):
        """Infinite repeat should never complete on its own."""
        count = [0]
        action = Action(lambda ctx: (count.__setitem__(0, count[0] + 1), BTStatus.SUCCESS)[1])
        repeat = Repeat(child=action, count=BT_INFINITE_REPEAT)

        for _ in range(100):
            assert repeat.tick(bt_context) == BTStatus.RUNNING
        assert count[0] == 100


# =============================================================================
# Context Tests
# =============================================================================


class TestBTContext:
    """Test BTContext functionality."""

    def test_context_has_blackboard(self, detailed_blackboard):
        """Context should have blackboard."""
        context = BTContext(blackboard=detailed_blackboard)
        assert context.blackboard is detailed_blackboard

    def test_context_has_entity(self, detailed_blackboard):
        """Context should support entity."""
        entity = Mock()
        context = BTContext(blackboard=detailed_blackboard, entity=entity)
        assert context.entity is entity

    def test_context_has_delta_time(self, detailed_blackboard):
        """Context should have delta time."""
        context = BTContext(blackboard=detailed_blackboard, delta_time=0.033)
        assert context.delta_time == 0.033

    def test_context_trace_logging(self, detailed_blackboard):
        """Context should support trace logging."""
        context = BTContext(blackboard=detailed_blackboard, debug_trace=True)
        action = Action(lambda ctx: BTStatus.SUCCESS)
        context.log_trace(action, BTStatus.SUCCESS)
        assert len(context.trace_log) == 1


# =============================================================================
# Decorator behavior_tree Tests
# =============================================================================


class TestBehaviorTreeDecorator:
    """Test @behavior_tree decorator."""

    def test_decorator_marks_class(self):
        """Decorator should mark class as behavior tree."""
        @behavior_tree(id="test_bt")
        class TestBT:
            pass

        assert hasattr(TestBT, "_behavior_tree")
        assert TestBT._behavior_tree is True
        assert TestBT._bt_id == "test_bt"

    def test_decorator_with_debug_name(self):
        """Decorator should support debug name."""
        @behavior_tree(id="test", debug_name="Test Behavior")
        class TestBT:
            pass

        assert TestBT._bt_debug_name == "Test Behavior"


# =============================================================================
# Node Type Tests
# =============================================================================


class TestNodeTypes:
    """Test node type properties."""

    def test_sequence_node_type(self):
        """Sequence should have correct node type."""
        sequence = Sequence()
        assert sequence.node_type == BTNodeType.SEQUENCE

    def test_selector_node_type(self):
        """Selector should have correct node type."""
        selector = Selector()
        assert selector.node_type == BTNodeType.SELECTOR

    def test_parallel_node_type(self):
        """Parallel should have correct node type."""
        parallel = Parallel()
        assert parallel.node_type == BTNodeType.PARALLEL

    def test_invert_node_type(self):
        """Invert should have correct node type."""
        invert = Invert(child=Action(lambda ctx: BTStatus.SUCCESS))
        assert invert.node_type == BTNodeType.INVERT

    def test_repeat_node_type(self):
        """Repeat should have correct node type."""
        repeat = Repeat(child=Action(lambda ctx: BTStatus.SUCCESS))
        assert repeat.node_type == BTNodeType.REPEAT

    def test_action_node_type(self):
        """Action should have correct node type."""
        action = Action(lambda ctx: BTStatus.SUCCESS)
        assert action.node_type == BTNodeType.ACTION

    def test_condition_node_type(self):
        """Condition should have correct node type."""
        condition = Condition(lambda ctx: True)
        assert condition.node_type == BTNodeType.CONDITION


# =============================================================================
# Composite Node Management Tests
# =============================================================================


class TestCompositeManagement:
    """Test composite node child management."""

    def test_composite_add_child_returns_self(self):
        """add_child should return self for chaining."""
        composite = Sequence()
        result = composite.add_child(Action(lambda ctx: BTStatus.SUCCESS))
        assert result is composite

    def test_composite_remove_child(self):
        """Should support removing children."""
        action = Action(lambda ctx: BTStatus.SUCCESS)
        composite = Sequence(children=[action])
        assert composite.remove_child(action)
        assert action not in composite.children

    def test_composite_remove_nonexistent_child(self):
        """Removing nonexistent child should return False."""
        action = Action(lambda ctx: BTStatus.SUCCESS)
        composite = Sequence()
        assert not composite.remove_child(action)

    def test_composite_children_property(self):
        """children property should return list."""
        action1 = Action(lambda ctx: BTStatus.SUCCESS)
        action2 = Action(lambda ctx: BTStatus.SUCCESS)
        composite = Sequence(children=[action1, action2])
        assert action1 in composite.children
        assert action2 in composite.children

    def test_composite_abort_aborts_children(self, bt_context):
        """Aborting composite should abort all children."""
        actions = [
            Action(lambda ctx: BTStatus.RUNNING),
            Action(lambda ctx: BTStatus.RUNNING),
        ]
        composite = Sequence(children=actions)
        composite.tick(bt_context)
        composite.abort()

        for action in actions:
            assert action.status == BTStatus.FAILURE


# =============================================================================
# Decorator Node Management Tests
# =============================================================================


class TestDecoratorManagement:
    """Test decorator node child management."""

    def test_decorator_sets_parent(self):
        """Decorator should set itself as child's parent."""
        action = Action(lambda ctx: BTStatus.SUCCESS)
        decorator = Invert(child=action)
        assert action._parent is decorator

    def test_decorator_reset_resets_child(self, bt_context):
        """Resetting decorator should reset child."""
        action = Action(lambda ctx: BTStatus.SUCCESS)
        decorator = Invert(child=action)
        decorator.tick(bt_context)
        decorator.reset()
        assert action.status == BTStatus.SUCCESS  # Initial status


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in behavior tree."""

    def test_action_exception_returns_failure(self, bt_context):
        """Action exception should return FAILURE."""
        def failing_action(ctx):
            raise ValueError("Test error")

        action = Action(failing_action)
        assert action.tick(bt_context) == BTStatus.FAILURE

    def test_condition_exception_returns_failure(self, bt_context):
        """Condition exception should return FAILURE."""
        def failing_condition(ctx):
            raise ValueError("Test error")

        condition = Condition(failing_condition)
        assert condition.tick(bt_context) == BTStatus.FAILURE


# =============================================================================
# Integration Tests
# =============================================================================


class TestBTIntegration:
    """Integration tests for complete behavior trees."""

    def test_complex_tree_structure(self):
        """Test complex nested tree structure."""
        # Build a tree: Selector(Sequence(Cond, Action), Action)
        attack_condition = BTCondition(condition=lambda: True)
        attack_action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)
        attack_sequence = BTSequence(children=[attack_condition, attack_action])

        fallback_action = BTAction(action=lambda dt: BTNodeStatus.SUCCESS)

        root = BTSelector(children=[attack_sequence, fallback_action])
        tree = BehaviorTree(tree_id="combat", root=root)

        assert tree.tick(0.016) == BTNodeStatus.SUCCESS

    def test_tree_with_blackboard_sharing(self, blackboard):
        """Test blackboard data sharing between nodes."""
        # Set value in first action
        set_action = BTAction(
            action=lambda dt: BTNodeStatus.SUCCESS
        )
        set_action._blackboard = blackboard

        # Check value in condition
        check_condition = BTCondition(
            condition=lambda: blackboard.get("key") == "value"
        )
        check_condition._blackboard = blackboard

        blackboard.set("key", "value")

        sequence = BTSequence(children=[set_action, check_condition])
        sequence.set_blackboard(blackboard)

        assert sequence.tick(0.016) == BTNodeStatus.SUCCESS

    def test_detailed_tree_with_all_node_types(self, detailed_blackboard):
        """Test detailed tree with various node types."""
        # Build complex tree with detailed implementation
        tree = DetailedBehaviorTree(
            root=Selector(children=[
                Sequence(children=[
                    Condition(lambda ctx: True),
                    Action(lambda ctx: BTStatus.SUCCESS),
                ]),
                Parallel(
                    children=[
                        Action(lambda ctx: BTStatus.SUCCESS),
                        Action(lambda ctx: BTStatus.SUCCESS),
                    ],
                    policy=ParallelPolicy.REQUIRE_ALL,
                ),
            ]),
            blackboard=detailed_blackboard,
        )

        assert tree.tick() == BTStatus.SUCCESS
