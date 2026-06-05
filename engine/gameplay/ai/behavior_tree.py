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

# Import Foundation registry
try:
    from foundation import registry
    FOUNDATION_AVAILABLE = True
except ImportError:
    registry = None
    FOUNDATION_AVAILABLE = False


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
    # Event logging callback: (entity_id, bt_name, node, is_enter, status) -> None
    event_callback: Optional[Callable[..., None]] = None
    bt_name: str = ""

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
            event_callback=self.event_callback,
            bt_name=self.bt_name,
        )

    def log_trace(self, node: "BTNode", status: BTStatus) -> None:
        """Log a trace entry if debug tracing is enabled."""
        if self.debug_trace:
            indent = "  " * self.depth
            self.trace_log.append(f"{indent}{node.name}: {status.name}")

    def get_entity_id(self) -> int:
        """Get the entity ID for event logging.

        Returns:
            The entity's id attribute if present, id(entity) if not,
            or 0 if no entity is set.
        """
        if self.entity is None:
            return 0
        if hasattr(self.entity, 'id'):
            return self.entity.id
        return id(self.entity)

    def log_node_enter(self, node: "BTNode") -> None:
        """Log node entry event if callback is set."""
        if self.event_callback is not None:
            self.event_callback(
                self.get_entity_id(),
                self.bt_name,
                node.name,
                node.node_type.name if hasattr(node.node_type, 'name') else str(node.node_type),
                True,  # is_enter
                None,  # status (not yet known)
            )

    def log_node_exit(self, node: "BTNode", status: BTStatus) -> None:
        """Log node exit event if callback is set."""
        if self.event_callback is not None:
            self.event_callback(
                self.get_entity_id(),
                self.bt_name,
                node.name,
                node.node_type.name if hasattr(node.node_type, 'name') else str(node.node_type),
                False,  # is_enter
                status.name if hasattr(status, 'name') else str(status),
            )


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
            child_context.log_node_enter(child)
            status = child.tick(child_context)
            child_context.log_node_exit(child, status)
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
            child_context.log_node_enter(child)
            status = child.tick(child_context)
            child_context.log_node_exit(child, status)
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
            child_context.log_node_enter(child)
            status = child.tick(child_context)
            child_context.log_node_exit(child, status)
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
        enable_event_logging: bool = True,
    ) -> BTStatus:
        """Execute one tick of the behavior tree."""
        current_time = time.time()

        # Set up event callback if logging is enabled
        event_callback = None
        if enable_event_logging and entity is not None:
            event_callback = self._create_event_callback()

        context = BTContext(
            blackboard=self._blackboard,
            entity=entity,
            delta_time=delta_time,
            current_time=current_time,
            abort_requested=self._aborted,
            debug_trace=debug_trace,
            event_callback=event_callback,
            bt_name=self.name,
        )

        # Log root entry
        context.log_node_enter(self._root)

        # Execute the tree
        status = self._root.tick(context)

        # Log root exit
        context.log_node_exit(self._root, status)

        self._running = status == BTStatus.RUNNING
        self._last_tick_time = current_time

        if debug_trace:
            context.log_trace(self._root, status)

        return status

    def _create_event_callback(self) -> Callable[..., None]:
        """Create the event callback for logging."""
        try:
            from .ai_events import get_ai_event_logger
            logger = get_ai_event_logger()

            def callback(
                entity_id: int,
                bt_name: str,
                node_name: str,
                node_type: str,
                is_enter: bool,
                status: Optional[str],
            ) -> None:
                if is_enter:
                    logger.log_bt_node_entered(
                        entity_id=entity_id,
                        bt_name=bt_name,
                        node_name=node_name,
                        node_type=node_type,
                    )
                else:
                    logger.log_bt_node_exited(
                        entity_id=entity_id,
                        bt_name=bt_name,
                        node_name=node_name,
                        result=status or "UNKNOWN",
                    )

            return callback
        except ImportError:
            return lambda *args, **kwargs: None

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

    @classmethod
    def from_registry(
        cls,
        name: str,
        blackboard: Optional[Blackboard] = None,
        **kwargs: Any,
    ) -> "BehaviorTree":
        """
        Create a BehaviorTree instance from a registered behavior tree class.

        Args:
            name: The registered name of the behavior tree class.
            blackboard: Optional blackboard to use.
            **kwargs: Additional keyword arguments passed to create_root/build method.

        Returns:
            A new BehaviorTree instance.

        Raises:
            KeyError: If the behavior tree name is not found in registry.
            TypeError: If the registered class doesn't have create_root or build method.
        """
        if not FOUNDATION_AVAILABLE or registry is None:
            raise KeyError(f"Behavior tree '{name}' not found (Foundation not available)")

        # Look up in registry
        registry_name = f"bt.{name}"
        bt_class = registry.get(registry_name)

        if bt_class is None:
            raise KeyError(f"Behavior tree '{name}' not found in registry")

        # Create from the registered class
        if hasattr(bt_class, "create_root"):
            # create_root returns a root node - pass kwargs to it
            root = bt_class.create_root(**kwargs)
            return cls(root=root, blackboard=blackboard, name=name)
        elif hasattr(bt_class, "build"):
            # build may return a BehaviorTree instance or a root node
            result = bt_class.build(blackboard=blackboard, **kwargs)
            if isinstance(result, cls):
                return result
            # Assume it's a root node
            return cls(root=result, blackboard=blackboard, name=name)
        else:
            raise TypeError(
                f"Registered behavior tree '{name}' must have 'create_root()' or 'build()' method"
            )


# =============================================================================
# BT Registry Integration with Foundation
# =============================================================================

# Valid node types for bt_node decorator
VALID_BT_NODE_TYPES = frozenset({
    "action", "condition", "sequence", "selector", "parallel",
    "decorator", "invert", "repeat", "timeout", "cooldown",
    "retry", "force_success", "force_failure", "wait", "set_blackboard",
})


class BTNodeTypeError(Exception):
    """Raised when an invalid BT node type is specified."""
    pass


# Tag constants for BT types
TAG_BEHAVIOR_TREE = "behavior_tree"
TAG_BT_NODE = "bt_node"


def behavior_tree(
    name: Optional[str] = None,
    *,
    id: Optional[str] = None,
    description: Optional[str] = None,
    debug_name: Optional[str] = None,
    track_instances: bool = False,
) -> Callable[[type], type]:
    """
    Decorator to register a class as a behavior tree with Foundation Registry.

    Args:
        name: Unique name for the behavior tree (can also use 'id' for backwards compat)
        id: Alias for name (backwards compatibility)
        description: Human-readable description of what this BT does
        debug_name: Alias for description (backwards compatibility)
        track_instances: If True, track all instances via WeakSet

    Usage:
        @behavior_tree(name="patrol", description="Patrol AI")
        class PatrolBehavior:
            pass
    """
    # Handle backwards compatibility
    bt_name = name or id
    bt_desc = description or debug_name

    if not bt_name:
        raise ValueError("behavior_tree requires a 'name' or 'id' argument")

    def decorator(cls: type) -> type:
        # Set class attributes
        cls._behavior_tree = True
        cls._bt_id = bt_name
        cls._bt_name = bt_name
        cls._bt_debug_name = bt_desc
        cls._bt_description = bt_desc or ""

        # Count nodes if the class has a method to do so
        node_count = 0
        if hasattr(cls, "_count_nodes"):
            try:
                node_count = cls._count_nodes()
            except Exception:
                pass

        # Register with Foundation Registry if available
        if FOUNDATION_AVAILABLE and registry is not None:
            registry_name = f"bt.{bt_name}"
            try:
                registry.register(cls, name=registry_name, track_instances=track_instances)
            except ValueError:
                # Already registered - fine in reload scenarios
                pass

            # Add tag for query-based discovery
            registry.add_tag(cls, TAG_BEHAVIOR_TREE)

            # Store metadata
            registry.set_metadata(cls, "bt_name", bt_name)
            registry.set_metadata(cls, "description", bt_desc or "")
            registry.set_metadata(cls, "node_count", node_count)

        return cls
    return decorator


def bt_node(
    node_type: str,
    *,
    id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    track_instances: bool = False,
) -> Callable[[type], type]:
    """
    Decorator to register a BT node class with Foundation Registry.

    Args:
        node_type: The type of BT node (action, condition, sequence, etc.)
        id: Optional custom registry id
        name: Alias for id
        description: Optional description of the node's purpose
        track_instances: If True, track all instances via WeakSet

    Usage:
        @bt_node(node_type="action", id="patrol_move")
        class PatrolMoveAction(Action):
            pass
    """
    # Normalize node_type: strip whitespace and convert to lowercase
    node_type = node_type.strip().lower()

    if node_type not in VALID_BT_NODE_TYPES:
        raise BTNodeTypeError(
            f"Invalid node type '{node_type}'. Must be one of: {VALID_BT_NODE_TYPES}"
        )

    def decorator(cls: type) -> type:
        node_id = id or name or cls.__name__

        # Set class attributes
        cls._bt_node = True
        cls._bt_node_type = node_type
        cls._bt_node_id = node_id
        cls._bt_node_description = description or ""

        # Register with Foundation Registry if available
        if FOUNDATION_AVAILABLE and registry is not None:
            registry_name = f"bt_node.{node_type}.{node_id}"
            try:
                registry.register(cls, name=registry_name, track_instances=track_instances)
            except ValueError:
                # Already registered - fine in reload scenarios
                pass

            # Add tags
            registry.add_tag(cls, TAG_BT_NODE)
            registry.add_tag(cls, f"bt_node_{node_type}")

            # Store metadata
            registry.set_metadata(cls, "node_type", node_type)
            registry.set_metadata(cls, "node_id", node_id)
            if description:
                registry.set_metadata(cls, "description", description)

        return cls
    return decorator


def get_all_behavior_trees() -> List[type]:
    """Get all registered behavior tree classes."""
    if FOUNDATION_AVAILABLE and registry is not None:
        return registry.query(tag=TAG_BEHAVIOR_TREE)
    return []


def get_all_bt_nodes() -> List[type]:
    """Get all registered BT node classes."""
    if FOUNDATION_AVAILABLE and registry is not None:
        return registry.query(tag=TAG_BT_NODE)
    return []


def get_bt_nodes_by_type(node_type: str) -> List[type]:
    """Get all registered BT nodes of a specific type."""
    if node_type not in VALID_BT_NODE_TYPES:
        raise BTNodeTypeError(
            f"Invalid node type '{node_type}'. Must be one of: {VALID_BT_NODE_TYPES}"
        )
    if FOUNDATION_AVAILABLE and registry is not None:
        return registry.query(tag=TAG_BT_NODE, node_type=node_type)
    return []


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Context
    "BTContext",
    # Base
    "BTNode",
    "BTNodeTypeError",
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
    # Decorator functions
    "behavior_tree",
    "bt_node",
    # Registry functions
    "get_all_behavior_trees",
    "get_all_bt_nodes",
    "get_bt_nodes_by_type",
    # Constants
    "VALID_BT_NODE_TYPES",
]
