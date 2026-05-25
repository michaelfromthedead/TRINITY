"""
Behavior Tree Runtime - Full BT implementation with all node types.

Supports:
- Composite nodes: Sequence, Selector, Parallel
- Decorator nodes: Invert, Repeat, Timeout, Cooldown, Retry, ForceSuccess, ForceFailure
- Leaf nodes: Action, Condition
- Tick-based execution with abort on condition change
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .blackboard import Blackboard
from .constants import (
    BTStatus,
    BTNodeType,
    ParallelPolicy,
    BT_DEFAULT_TICK_INTERVAL,
    BT_MAX_DEPTH,
    BT_DEFAULT_TIMEOUT,
    BT_DEFAULT_COOLDOWN,
    BT_DEFAULT_REPEAT_COUNT,
    BT_DEFAULT_RETRY_COUNT,
    BT_INFINITE_REPEAT,
)

if TYPE_CHECKING:
    from .blackboard import BlackboardKey


@dataclass
class BTContext:
    """Context passed through the behavior tree during execution."""
    blackboard: Blackboard
    entity: Any = None
    delta_time: float = 0.0
    current_time: float = field(default_factory=time.time)
    depth: int = 0
    abort_requested: bool = False
    debug_trace: bool = False
    trace_log: List[str] = field(default_factory=list)

    def child_context(self) -> "BTContext":
        """Create a child context with incremented depth."""
        return BTContext(
            blackboard=self.blackboard,
            entity=self.entity,
            delta_time=self.delta_time,
            current_time=self.current_time,
            depth=self.depth + 1,
            abort_requested=self.abort_requested,
            debug_trace=self.debug_trace,
            trace_log=self.trace_log,
        )

    def log_trace(self, node: "BTNode", status: BTStatus) -> None:
        """Log a trace entry if debug tracing is enabled."""
        if self.debug_trace:
            indent = "  " * self.depth
            self.trace_log.append(f"{indent}{node.name}: {status.name}")


class BTNode(ABC):
    """Base class for all behavior tree nodes."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self._status = BTStatus.SUCCESS
        self._parent: Optional[BTNode] = None

    @property
    def status(self) -> BTStatus:
        """Get the current status of this node."""
        return self._status

    @property
    @abstractmethod
    def node_type(self) -> BTNodeType:
        """Get the type of this node."""
        pass

    @abstractmethod
    def tick(self, context: BTContext) -> BTStatus:
        """Execute one tick of this node."""
        pass

    def reset(self) -> None:
        """Reset this node to its initial state."""
        self._status = BTStatus.SUCCESS

    def abort(self) -> None:
        """Abort this node's execution."""
        self._status = BTStatus.FAILURE


# =============================================================================
# Composite Nodes
# =============================================================================


class CompositeNode(BTNode):
    """Base class for composite nodes that have children."""

    def __init__(
        self,
        children: Optional[List[BTNode]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self._children: List[BTNode] = []
        if children:
            for child in children:
                self.add_child(child)

    @property
    def children(self) -> List[BTNode]:
        """Get the children of this node."""
        return self._children

    def add_child(self, child: BTNode) -> "CompositeNode":
        """Add a child node."""
        child._parent = self
        self._children.append(child)
        return self

    def remove_child(self, child: BTNode) -> bool:
        """Remove a child node."""
        if child in self._children:
            child._parent = None
            self._children.remove(child)
            return True
        return False

    def reset(self) -> None:
        """Reset this node and all children."""
        super().reset()
        for child in self._children:
            child.reset()

    def abort(self) -> None:
        """Abort this node and all children."""
        super().abort()
        for child in self._children:
            child.abort()


class Sequence(CompositeNode):
    """
    Executes children in order until one fails.

    - Returns SUCCESS if all children succeed
    - Returns FAILURE if any child fails
    - Returns RUNNING if a child is still running
    """

    def __init__(
        self,
        children: Optional[List[BTNode]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(children, name)
        self._current_index = 0

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.SEQUENCE

    def tick(self, context: BTContext) -> BTStatus:
        if context.depth > BT_MAX_DEPTH:
            self._status = BTStatus.FAILURE
            return self._status

        if context.abort_requested:
            self.abort()
            return self._status

        child_context = context.child_context()

        while self._current_index < len(self._children):
            child = self._children[self._current_index]
            status = child.tick(child_context)
            context.log_trace(child, status)

            if status == BTStatus.RUNNING:
                self._status = BTStatus.RUNNING
                return self._status
            elif status == BTStatus.FAILURE:
                self._current_index = 0
                self._status = BTStatus.FAILURE
                return self._status

            self._current_index += 1

        self._current_index = 0
        self._status = BTStatus.SUCCESS
        return self._status

    def reset(self) -> None:
        super().reset()
        self._current_index = 0


class Selector(CompositeNode):
    """
    Executes children in order until one succeeds.

    - Returns SUCCESS if any child succeeds
    - Returns FAILURE if all children fail
    - Returns RUNNING if a child is still running
    """

    def __init__(
        self,
        children: Optional[List[BTNode]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(children, name)
        self._current_index = 0

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.SELECTOR

    def tick(self, context: BTContext) -> BTStatus:
        if context.depth > BT_MAX_DEPTH:
            self._status = BTStatus.FAILURE
            return self._status

        if context.abort_requested:
            self.abort()
            return self._status

        child_context = context.child_context()

        while self._current_index < len(self._children):
            child = self._children[self._current_index]
            status = child.tick(child_context)
            context.log_trace(child, status)

            if status == BTStatus.RUNNING:
                self._status = BTStatus.RUNNING
                return self._status
            elif status == BTStatus.SUCCESS:
                self._current_index = 0
                self._status = BTStatus.SUCCESS
                return self._status

            self._current_index += 1

        self._current_index = 0
        self._status = BTStatus.FAILURE
        return self._status

    def reset(self) -> None:
        super().reset()
        self._current_index = 0


class Parallel(CompositeNode):
    """
    Executes all children simultaneously.

    Success/failure is determined by the policy:
    - REQUIRE_ALL: All children must succeed
    - REQUIRE_ONE: Any one child succeeding is enough
    - REQUIRE_MAJORITY: Majority must succeed
    """

    def __init__(
        self,
        children: Optional[List[BTNode]] = None,
        policy: ParallelPolicy = ParallelPolicy.REQUIRE_ALL,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(children, name)
        self._policy = policy

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.PARALLEL

    @property
    def policy(self) -> ParallelPolicy:
        return self._policy

    def tick(self, context: BTContext) -> BTStatus:
        if context.depth > BT_MAX_DEPTH:
            self._status = BTStatus.FAILURE
            return self._status

        if context.abort_requested:
            self.abort()
            return self._status

        if not self._children:
            self._status = BTStatus.SUCCESS
            return self._status

        child_context = context.child_context()
        success_count = 0
        failure_count = 0
        running_count = 0

        for child in self._children:
            status = child.tick(child_context)
            context.log_trace(child, status)

            if status == BTStatus.SUCCESS:
                success_count += 1
            elif status == BTStatus.FAILURE:
                failure_count += 1
            else:
                running_count += 1

        total = len(self._children)

        if self._policy == ParallelPolicy.REQUIRE_ALL:
            if failure_count > 0:
                self._status = BTStatus.FAILURE
            elif running_count > 0:
                self._status = BTStatus.RUNNING
            else:
                self._status = BTStatus.SUCCESS
        elif self._policy == ParallelPolicy.REQUIRE_ONE:
            if success_count > 0:
                self._status = BTStatus.SUCCESS
            elif running_count > 0:
                self._status = BTStatus.RUNNING
            else:
                self._status = BTStatus.FAILURE
        else:  # REQUIRE_MAJORITY
            majority = total // 2 + 1
            if success_count >= majority:
                self._status = BTStatus.SUCCESS
            elif failure_count >= majority:
                self._status = BTStatus.FAILURE
            elif running_count > 0:
                self._status = BTStatus.RUNNING
            else:
                self._status = BTStatus.FAILURE

        return self._status


# =============================================================================
# Decorator Nodes
# =============================================================================


class DecoratorNode(BTNode):
    """Base class for decorator nodes that wrap a single child."""

    def __init__(self, child: BTNode, name: Optional[str] = None) -> None:
        super().__init__(name)
        self._child = child
        self._child._parent = self

    @property
    def child(self) -> BTNode:
        """Get the child node."""
        return self._child

    def reset(self) -> None:
        super().reset()
        self._child.reset()

    def abort(self) -> None:
        super().abort()
        self._child.abort()


class Invert(DecoratorNode):
    """Inverts the result of its child node."""

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.INVERT

    def tick(self, context: BTContext) -> BTStatus:
        child_context = context.child_context()
        status = self._child.tick(child_context)
        context.log_trace(self._child, status)

        if status == BTStatus.SUCCESS:
            self._status = BTStatus.FAILURE
        elif status == BTStatus.FAILURE:
            self._status = BTStatus.SUCCESS
        else:
            self._status = BTStatus.RUNNING

        return self._status


class Repeat(DecoratorNode):
    """
    Repeats its child a specified number of times.

    - count: Number of times to repeat (-1 for infinite)
    - until_fail: If True, stops when child fails
    - until_success: If True, stops when child succeeds
    """

    def __init__(
        self,
        child: BTNode,
        count: int = BT_DEFAULT_REPEAT_COUNT,
        until_fail: bool = False,
        until_success: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(child, name)
        self._count = count
        self._until_fail = until_fail
        self._until_success = until_success
        self._current_count = 0

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.REPEAT

    def tick(self, context: BTContext) -> BTStatus:
        child_context = context.child_context()
        status = self._child.tick(child_context)
        context.log_trace(self._child, status)

        if status == BTStatus.RUNNING:
            self._status = BTStatus.RUNNING
            return self._status

        if status == BTStatus.FAILURE and self._until_fail:
            self._current_count = 0
            self._child.reset()
            self._status = BTStatus.SUCCESS
            return self._status

        if status == BTStatus.SUCCESS and self._until_success:
            self._current_count = 0
            self._child.reset()
            self._status = BTStatus.SUCCESS
            return self._status

        self._current_count += 1

        if self._count != BT_INFINITE_REPEAT and self._current_count >= self._count:
            self._current_count = 0
            self._child.reset()
            self._status = status
            return self._status

        self._child.reset()
        self._status = BTStatus.RUNNING
        return self._status

    def reset(self) -> None:
        super().reset()
        self._current_count = 0


class Timeout(DecoratorNode):
    """
    Fails if the child doesn't complete within the timeout.
    """

    def __init__(
        self,
        child: BTNode,
        timeout: float = BT_DEFAULT_TIMEOUT,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(child, name)
        self._timeout = timeout
        self._start_time: Optional[float] = None

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.TIMEOUT

    def tick(self, context: BTContext) -> BTStatus:
        if self._start_time is None:
            self._start_time = context.current_time

        elapsed = context.current_time - self._start_time

        if elapsed >= self._timeout:
            self._start_time = None
            self._child.abort()
            self._status = BTStatus.FAILURE
            return self._status

        child_context = context.child_context()
        status = self._child.tick(child_context)
        context.log_trace(self._child, status)

        if status != BTStatus.RUNNING:
            self._start_time = None

        self._status = status
        return self._status

    def reset(self) -> None:
        super().reset()
        self._start_time = None


class Cooldown(DecoratorNode):
    """
    Prevents the child from running more often than the cooldown allows.
    """

    def __init__(
        self,
        child: BTNode,
        cooldown: float = BT_DEFAULT_COOLDOWN,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(child, name)
        self._cooldown = cooldown
        self._last_run_time: Optional[float] = None

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.COOLDOWN

    def tick(self, context: BTContext) -> BTStatus:
        if self._last_run_time is not None:
            elapsed = context.current_time - self._last_run_time
            if elapsed < self._cooldown:
                self._status = BTStatus.FAILURE
                return self._status

        child_context = context.child_context()
        status = self._child.tick(child_context)
        context.log_trace(self._child, status)

        if status != BTStatus.RUNNING:
            self._last_run_time = context.current_time

        self._status = status
        return self._status

    def reset(self) -> None:
        super().reset()
        self._last_run_time = None


class Retry(DecoratorNode):
    """
    Retries the child a specified number of times on failure.
    """

    def __init__(
        self,
        child: BTNode,
        max_retries: int = BT_DEFAULT_RETRY_COUNT,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(child, name)
        self._max_retries = max_retries
        self._retry_count = 0

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.RETRY

    def tick(self, context: BTContext) -> BTStatus:
        child_context = context.child_context()
        status = self._child.tick(child_context)
        context.log_trace(self._child, status)

        if status == BTStatus.SUCCESS:
            self._retry_count = 0
            self._status = BTStatus.SUCCESS
            return self._status

        if status == BTStatus.RUNNING:
            self._status = BTStatus.RUNNING
            return self._status

        # Child failed
        self._retry_count += 1
        if self._retry_count >= self._max_retries:
            self._retry_count = 0
            self._status = BTStatus.FAILURE
            return self._status

        self._child.reset()
        self._status = BTStatus.RUNNING
        return self._status

    def reset(self) -> None:
        super().reset()
        self._retry_count = 0


class ForceSuccess(DecoratorNode):
    """Forces the child result to SUCCESS (unless RUNNING)."""

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.FORCE_SUCCESS

    def tick(self, context: BTContext) -> BTStatus:
        child_context = context.child_context()
        status = self._child.tick(child_context)
        context.log_trace(self._child, status)

        if status == BTStatus.RUNNING:
            self._status = BTStatus.RUNNING
        else:
            self._status = BTStatus.SUCCESS

        return self._status


class ForceFailure(DecoratorNode):
    """Forces the child result to FAILURE (unless RUNNING)."""

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.FORCE_FAILURE

    def tick(self, context: BTContext) -> BTStatus:
        child_context = context.child_context()
        status = self._child.tick(child_context)
        context.log_trace(self._child, status)

        if status == BTStatus.RUNNING:
            self._status = BTStatus.RUNNING
        else:
            self._status = BTStatus.FAILURE

        return self._status


# =============================================================================
# Leaf Nodes
# =============================================================================


class LeafNode(BTNode):
    """Base class for leaf nodes with no children."""

    pass


class Action(LeafNode):
    """
    A leaf node that executes an action function.

    The action function receives the context and should return a BTStatus.
    """

    def __init__(
        self,
        action: Callable[[BTContext], BTStatus],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self._action = action

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.ACTION

    def tick(self, context: BTContext) -> BTStatus:
        try:
            self._status = self._action(context)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Action node '%s' raised exception: %s", self.name, e
            )
            self._status = BTStatus.FAILURE
        return self._status


class Condition(LeafNode):
    """
    A leaf node that checks a condition.

    The condition function receives the context and should return a bool.
    """

    def __init__(
        self,
        condition: Callable[[BTContext], bool],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self._condition = condition

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.CONDITION

    def tick(self, context: BTContext) -> BTStatus:
        try:
            result = self._condition(context)
            self._status = BTStatus.SUCCESS if result else BTStatus.FAILURE
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Condition node '%s' raised exception: %s", self.name, e
            )
            self._status = BTStatus.FAILURE
        return self._status


class BlackboardCondition(Condition):
    """
    A condition that checks a blackboard value.
    """

    def __init__(
        self,
        key: str,
        expected: Any = None,
        check_exists: bool = False,
        comparator: Optional[Callable[[Any, Any], bool]] = None,
        name: Optional[str] = None,
    ) -> None:
        self._key = key
        self._expected = expected
        self._check_exists = check_exists
        self._comparator = comparator or (lambda a, b: a == b)

        def check_condition(context: BTContext) -> bool:
            if self._check_exists:
                return context.blackboard.has(self._key)
            value = context.blackboard.get(self._key)
            return self._comparator(value, self._expected)

        super().__init__(check_condition, name)


class Wait(LeafNode):
    """
    A leaf node that waits for a specified duration.
    """

    def __init__(self, duration: float, name: Optional[str] = None) -> None:
        super().__init__(name)
        self._duration = duration
        self._start_time: Optional[float] = None

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.ACTION

    def tick(self, context: BTContext) -> BTStatus:
        if self._start_time is None:
            self._start_time = context.current_time

        elapsed = context.current_time - self._start_time

        if elapsed >= self._duration:
            self._start_time = None
            self._status = BTStatus.SUCCESS
        else:
            self._status = BTStatus.RUNNING

        return self._status

    def reset(self) -> None:
        super().reset()
        self._start_time = None


class SetBlackboard(LeafNode):
    """
    A leaf node that sets a blackboard value.
    """

    def __init__(
        self,
        key: str,
        value: Any = None,
        value_func: Optional[Callable[[BTContext], Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self._key = key
        self._value = value
        self._value_func = value_func

    @property
    def node_type(self) -> BTNodeType:
        return BTNodeType.ACTION

    def tick(self, context: BTContext) -> BTStatus:
        try:
            if self._value_func:
                value = self._value_func(context)
            else:
                value = self._value
            context.blackboard.set(self._key, value)
            self._status = BTStatus.SUCCESS
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "SetBlackboard node '%s' raised exception: %s", self.name, e
            )
            self._status = BTStatus.FAILURE
        return self._status


# =============================================================================
# Behavior Tree
# =============================================================================


class BehaviorTree:
    """
    A complete behavior tree with a root node.

    Manages the tick loop and provides utilities for tree management.
    """

    def __init__(
        self,
        root: BTNode,
        blackboard: Optional[Blackboard] = None,
        tick_interval: float = BT_DEFAULT_TICK_INTERVAL,
        name: Optional[str] = None,
    ) -> None:
        self.name = name or "BehaviorTree"
        self._root = root
        self._blackboard = blackboard or Blackboard()
        self._tick_interval = tick_interval
        self._last_tick_time = 0.0
        self._running = False
        self._aborted = False

    @property
    def root(self) -> BTNode:
        """Get the root node."""
        return self._root

    @property
    def blackboard(self) -> Blackboard:
        """Get the blackboard."""
        return self._blackboard

    @property
    def is_running(self) -> bool:
        """Check if the tree is currently running."""
        return self._running

    def tick(
        self,
        entity: Any = None,
        delta_time: float = 0.0,
        debug_trace: bool = False,
    ) -> BTStatus:
        """Execute one tick of the behavior tree."""
        current_time = time.time()

        context = BTContext(
            blackboard=self._blackboard,
            entity=entity,
            delta_time=delta_time,
            current_time=current_time,
            abort_requested=self._aborted,
            debug_trace=debug_trace,
        )

        status = self._root.tick(context)
        self._running = status == BTStatus.RUNNING
        self._last_tick_time = current_time

        if debug_trace:
            context.log_trace(self._root, status)

        return status

    def reset(self) -> None:
        """Reset the behavior tree."""
        self._root.reset()
        self._running = False
        self._aborted = False

    def abort(self) -> None:
        """Abort the behavior tree execution."""
        self._aborted = True
        self._root.abort()
        self._running = False


def behavior_tree(
    id: str,
    debug_name: Optional[str] = None,
) -> Callable[[type], type]:
    """
    Decorator to mark a class as a behavior tree.

    Usage:
        @behavior_tree(id="patrol", debug_name="Patrol AI")
        class PatrolBehavior:
            pass
    """
    def decorator(cls: type) -> type:
        cls._behavior_tree = True
        cls._bt_id = id
        cls._bt_debug_name = debug_name
        return cls
    return decorator


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Context
    "BTContext",
    # Base
    "BTNode",
    # Composite
    "CompositeNode",
    "Sequence",
    "Selector",
    "Parallel",
    # Decorator
    "DecoratorNode",
    "Invert",
    "Repeat",
    "Timeout",
    "Cooldown",
    "Retry",
    "ForceSuccess",
    "ForceFailure",
    # Leaf
    "LeafNode",
    "Action",
    "Condition",
    "BlackboardCondition",
    "Wait",
    "SetBlackboard",
    # Tree
    "BehaviorTree",
    # Decorator function
    "behavior_tree",
]
