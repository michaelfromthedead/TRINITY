"""
Behavior Tree Runtime - Full BT implementation with all node types.

Supports:
- Composite nodes: Sequence, Selector, Parallel
- Decorator nodes: Invert, Repeat, Timeout, Cooldown, Retry, ForceSuccess, ForceFailure
- Leaf nodes: Action, Condition
- Tick-based execution with abort on condition change
- Foundation Registry integration for runtime discovery
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING, Type

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

# Import Foundation Registry and EventLog
from foundation import registry, Registry

# Import AI events for EventLog integration
from .ai_events import get_ai_event_logger, AIEventLogger

if TYPE_CHECKING:
    from .blackboard import BlackboardKey


# =============================================================================
# Valid BT Node Types for @bt_node decorator
# =============================================================================

VALID_BT_NODE_TYPES: Set[str] = {
    "selector",
    "sequence",
    "parallel",
    "action",
    "condition",
    "decorator",
}


# =============================================================================
# Registry-backed Factory Storage
# =============================================================================

class BTNodeTypeError(ValueError):
    """Raised when an invalid BT node type is specified."""
    pass


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
    bt_name: str = ""  # Name of the behavior tree for event logging
    event_logger: Optional[AIEventLogger] = None  # Event logger for AI decisions

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
            bt_name=self.bt_name,
            event_logger=self.event_logger,
        )

    def log_trace(self, node: "BTNode", status: BTStatus) -> None:
        """Log a trace entry if debug tracing is enabled."""
        if self.debug_trace:
            indent = "  " * self.depth
            self.trace_log.append(f"{indent}{node.name}: {status.name}")

    def get_entity_id(self) -> int:
        """Get entity ID for event logging."""
        if self.entity is None:
            return 0
        return getattr(self.entity, 'id', id(self.entity))


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
    def _execute(self, context: BTContext) -> BTStatus:
        """Internal execution method to be implemented by subclasses."""
        pass

    def tick(self, context: BTContext) -> BTStatus:
        """Execute one tick of this node with event logging."""
        # Log node entry
        if context.event_logger is not None:
            context.event_logger.log_bt_node_entered(
                entity_id=context.get_entity_id(),
                bt_name=context.bt_name,
                node_name=self.name,
                node_type=self.node_type.name,
            )

        # Execute the node
        result = self._execute(context)

        # Log node exit
        if context.event_logger is not None:
            context.event_logger.log_bt_node_exited(
                entity_id=context.get_entity_id(),
                bt_name=context.bt_name,
                node_name=self.name,
                result=result.name,
            )

        return result

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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

    def _execute(self, context: BTContext) -> BTStatus:
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

        # Get event logger for AI decision tracking
        event_logger = get_ai_event_logger() if enable_event_logging else None

        context = BTContext(
            blackboard=self._blackboard,
            entity=entity,
            delta_time=delta_time,
            current_time=current_time,
            abort_requested=self._aborted,
            debug_trace=debug_trace,
            bt_name=self.name,
            event_logger=event_logger,
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

    @classmethod
    def from_registry(
        cls,
        name: str,
        blackboard: Optional[Blackboard] = None,
        tick_interval: float = BT_DEFAULT_TICK_INTERVAL,
        **kwargs: Any,
    ) -> "BehaviorTree":
        """
        Factory method to instantiate a behavior tree from the Foundation Registry.

        This method looks up a registered behavior tree class by name and
        instantiates it. The registered class must either:
        1. Be a subclass of BehaviorTree that can be instantiated directly
        2. Have a `build()` class method that returns a BehaviorTree instance
        3. Have a `create_root()` class method that returns a BTNode

        Usage:
            @behavior_tree(name="patrol", description="Patrol AI")
            class PatrolBehavior:
                @classmethod
                def create_root(cls) -> BTNode:
                    return Sequence([
                        MoveToWaypoint(),
                        Wait(1.0),
                    ])

            # Later, instantiate from registry:
            bt = BehaviorTree.from_registry("patrol")

        Args:
            name: The registered name of the behavior tree
            blackboard: Optional blackboard to use
            tick_interval: How often to tick the tree
            **kwargs: Additional arguments passed to the BT class

        Returns:
            An instantiated BehaviorTree

        Raises:
            KeyError: If no behavior tree with the given name is registered
            TypeError: If the registered class cannot be instantiated as a BehaviorTree
        """
        # Look up in registry
        registry_name = f"bt.{name}"
        bt_class = registry.get(registry_name)

        if bt_class is None:
            # Try finding by querying
            matches = registry.query(tag="behavior_tree", bt_name=name)
            if matches:
                bt_class = matches[0]

        if bt_class is None:
            raise KeyError(
                f"No behavior tree registered with name '{name}'. "
                f"Use @behavior_tree(name='{name}') to register one."
            )

        # Case 1: Class has a build() method that returns a BehaviorTree
        if hasattr(bt_class, "build") and callable(getattr(bt_class, "build")):
            return bt_class.build(blackboard=blackboard, tick_interval=tick_interval, **kwargs)

        # Case 2: Class has a create_root() method that returns a BTNode
        if hasattr(bt_class, "create_root") and callable(getattr(bt_class, "create_root")):
            root = bt_class.create_root(**kwargs)
            return cls(
                root=root,
                blackboard=blackboard,
                tick_interval=tick_interval,
                name=name,
            )

        # Case 3: Class is a subclass of BehaviorTree
        if isinstance(bt_class, type) and issubclass(bt_class, cls):
            return bt_class(
                blackboard=blackboard,
                tick_interval=tick_interval,
                **kwargs,
            )

        # Case 4: Try direct instantiation if it has a root attribute after init
        try:
            instance = bt_class(**kwargs)
            if hasattr(instance, "root") or hasattr(instance, "_root"):
                root = getattr(instance, "root", None) or getattr(instance, "_root", None)
                if root is not None:
                    return cls(
                        root=root,
                        blackboard=blackboard,
                        tick_interval=tick_interval,
                        name=name,
                    )
        except Exception:
            pass

        raise TypeError(
            f"Registered class '{bt_class.__name__}' cannot be instantiated as a BehaviorTree. "
            f"It must either subclass BehaviorTree, have a build() method, or have a create_root() method."
        )


def get_all_behavior_trees() -> List[Type]:
    """
    Get all registered behavior tree classes from the Foundation Registry.

    Returns:
        List of all classes registered with the @behavior_tree decorator
    """
    return registry.query(tag="behavior_tree")


def get_bt_nodes_by_type(node_type: str) -> List[Type]:
    """
    Get all registered BT node classes of a specific type.

    Args:
        node_type: The node type to filter by (selector, sequence, parallel, action, condition, decorator)

    Returns:
        List of all node classes matching the given type
    """
    normalized_type = node_type.lower().strip()
    return registry.query(tag="bt_node", node_type=normalized_type)


def get_all_bt_nodes() -> List[Type]:
    """
    Get all registered BT node classes from the Foundation Registry.

    Returns:
        List of all classes registered with the @bt_node decorator
    """
    return registry.query(tag="bt_node")


def behavior_tree(
    name: str,
    description: Optional[str] = None,
    track_instances: bool = False,
) -> Callable[[type], type]:
    """
    Decorator to register a class as a behavior tree definition with Foundation Registry.

    This decorator:
    1. Registers the BT class with the Foundation Registry
    2. Tags it as "behavior_tree"
    3. Stores metadata (name, description, node_count)

    Usage:
        @behavior_tree(name="patrol", description="Patrol AI behavior")
        class PatrolBehavior:
            pass

        # Query all behavior trees:
        Registry.query(tag="behavior_tree")

    Args:
        name: Unique name for the behavior tree
        description: Human-readable description of what this BT does
        track_instances: If True, track all instances via WeakSet

    Returns:
        Decorated class registered with Foundation Registry
    """
    def decorator(cls: type) -> type:
        # Mark class attributes for BT identification
        cls._behavior_tree = True
        cls._bt_name = name
        cls._bt_description = description or ""

        # Count nodes if the class has a root or build method
        node_count = 0
        if hasattr(cls, "_count_nodes"):
            try:
                node_count = cls._count_nodes()
            except Exception:
                pass

        # Register with Foundation Registry
        registry_name = f"bt.{name}"
        try:
            registry.register(cls, name=registry_name, track_instances=track_instances)
        except ValueError:
            # Already registered - that's fine in reload scenarios
            pass

        # Add tag for query-based discovery
        registry.add_tag(cls, "behavior_tree")

        # Store metadata
        registry.set_metadata(cls, "bt_name", name)
        registry.set_metadata(cls, "description", description or "")
        registry.set_metadata(cls, "node_count", node_count)

        return cls

    return decorator


def bt_node(
    node_type: str,
    description: Optional[str] = None,
    track_instances: bool = False,
) -> Callable[[type], type]:
    """
    Decorator to register a class as a behavior tree node with Foundation Registry.

    This decorator:
    1. Validates the node type against allowed types
    2. Registers the node class with the Foundation Registry
    3. Tags it as "bt_node"
    4. Stores metadata (node_type, description)

    Valid node types:
    - "selector": Runs children until one succeeds
    - "sequence": Runs children until one fails
    - "parallel": Runs all children simultaneously
    - "action": Leaf node that performs work
    - "condition": Leaf node that checks a condition
    - "decorator": Wraps and modifies child behavior

    Usage:
        @bt_node(type="action", description="Move to target position")
        class MoveToAction:
            pass

        # Query all action nodes:
        Registry.query(tag="bt_node", node_type="action")

    Args:
        node_type: Type of BT node (selector, sequence, parallel, action, condition, decorator)
        description: Human-readable description of what this node does
        track_instances: If True, track all instances via WeakSet

    Returns:
        Decorated class registered with Foundation Registry

    Raises:
        BTNodeTypeError: If node_type is not one of the valid types
    """
    # Normalize node type
    normalized_type = node_type.lower().strip()

    if normalized_type not in VALID_BT_NODE_TYPES:
        raise BTNodeTypeError(
            f"Invalid BT node type '{node_type}'. "
            f"Valid types are: {', '.join(sorted(VALID_BT_NODE_TYPES))}"
        )

    def decorator(cls: type) -> type:
        # Mark class attributes for BT node identification
        cls._bt_node = True
        cls._bt_node_type = normalized_type
        cls._bt_node_description = description or ""

        # Register with Foundation Registry using unique name
        registry_name = f"bt_node.{cls.__module__}.{cls.__name__}"
        try:
            registry.register(cls, name=registry_name, track_instances=track_instances)
        except ValueError:
            # Already registered - that's fine in reload scenarios
            pass

        # Add tags for query-based discovery
        registry.add_tag(cls, "bt_node")
        registry.add_tag(cls, f"bt_node_{normalized_type}")

        # Store metadata
        registry.set_metadata(cls, "node_type", normalized_type)
        registry.set_metadata(cls, "description", description or "")

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
    # Registry Integration
    "behavior_tree",
    "bt_node",
    "get_all_behavior_trees",
    "get_bt_nodes_by_type",
    "get_all_bt_nodes",
    "BTNodeTypeError",
    "VALID_BT_NODE_TYPES",
]
