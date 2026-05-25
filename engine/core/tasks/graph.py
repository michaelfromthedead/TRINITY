"""Task graph — DAG of tasks with dependency edges.

Provides :class:`TaskGraph` for explicit DAG construction and
:class:`TaskGraphBuilder` with a fluent API.
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from engine.core.tasks.scheduler import TaskHandle, TaskScheduler

logger = logging.getLogger(__name__)

TaskNodeId = int


class TaskState(enum.Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class TaskNode:
    id: TaskNodeId
    name: str
    func: Optional[Callable]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    dependencies: Set[TaskNodeId] = field(default_factory=set)
    dependents: Set[TaskNodeId] = field(default_factory=set)
    state: TaskState = TaskState.PENDING
    result: Any = None
    is_fence: bool = False


class TaskGraph:
    """Directed acyclic graph of tasks with dependency tracking."""

    def __init__(self) -> None:
        self._nodes: Dict[TaskNodeId, TaskNode] = {}
        self._next_id: int = 0
        self._compiled: bool = False
        self._sorted: List[TaskNodeId] = []
        self._complete = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_task(self, name: str, func: Callable, *args: Any, **kwargs: Any) -> TaskNodeId:
        nid = self._next_id
        self._next_id += 1
        self._nodes[nid] = TaskNode(id=nid, name=name, func=func, args=args, kwargs=kwargs)
        self._compiled = False
        return nid

    def add_dependency(self, from_id: TaskNodeId, to_id: TaskNodeId) -> None:
        """*from_id* depends on *to_id* (to must finish before from starts)."""
        self._nodes[from_id].dependencies.add(to_id)
        self._nodes[to_id].dependents.add(from_id)
        self._compiled = False

    def add_fence(self, name: str) -> TaskNodeId:
        """Add a barrier node (no-op function) that others can depend on."""
        nid = self._next_id
        self._next_id += 1
        self._nodes[nid] = TaskNode(id=nid, name=name, func=None, is_fence=True)
        self._compiled = False
        return nid

    # ------------------------------------------------------------------
    # Compilation (topological sort + cycle detection)
    # ------------------------------------------------------------------

    def compile(self) -> List[TaskNodeId]:
        """Topological sort via Kahn's algorithm. Raises on cycle."""
        in_degree: Dict[TaskNodeId, int] = {
            nid: len(node.dependencies) for nid, node in self._nodes.items()
        }
        queue: deque[TaskNodeId] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        order: List[TaskNodeId] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for dep_id in self._nodes[nid].dependents:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        if len(order) != len(self._nodes):
            raise ValueError("TaskGraph contains a cycle")

        self._sorted = order
        self._compiled = True
        return order

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, executor: TaskScheduler) -> None:
        """Run all tasks respecting dependency order using *executor*."""
        if not self._compiled:
            self.compile()

        self._complete = False
        handles: Dict[TaskNodeId, TaskHandle] = {}

        for nid in self._sorted:
            node = self._nodes[nid]
            deps = [handles[d] for d in node.dependencies if d in handles]

            if node.is_fence or node.func is None:
                # Fence: just wait for deps, produce no-op
                if deps:
                    fn = lambda _deps=deps: [executor.wait(d) for d in _deps]  # noqa: E731
                else:
                    fn = lambda: None  # noqa: E731
                handle = executor.submit(fn)
            else:
                if deps:
                    handle = executor.submit_after(node.func, deps, *node.args, **node.kwargs)
                else:
                    handle = executor.submit(node.func, *node.args, **node.kwargs)

            handles[nid] = handle

        # Wait for all
        for h in handles.values():
            try:
                executor.wait(h)
            except Exception as exc:
                logger.error("Task failed during graph execution: %s", exc)

        # Mark states
        for nid, h in handles.items():
            node = self._nodes[nid]
            try:
                node.result = h.result(timeout=0)
                node.state = TaskState.COMPLETE
            except Exception:
                node.state = TaskState.FAILED

        self._complete = True

    def has_failures(self) -> bool:
        """Return True if any task in the graph has failed."""
        return any(node.state == TaskState.FAILED for node in self._nodes.values())

    def failed_nodes(self) -> List[TaskNodeId]:
        """Return list of node IDs that are in FAILED state."""
        return [nid for nid, node in self._nodes.items() if node.state == TaskState.FAILED]

    def is_complete(self) -> bool:
        return self._complete

    @property
    def nodes(self) -> Dict[TaskNodeId, TaskNode]:
        return self._nodes


# ---------------------------------------------------------------------------
# Fluent builder
# ---------------------------------------------------------------------------

class _TaskNodeRef:
    """Fluent helper returned by :meth:`TaskGraphBuilder.task`."""

    def __init__(self, builder: TaskGraphBuilder, node_id: TaskNodeId, name: str) -> None:
        self._builder = builder
        self._id = node_id
        self._name = name

    @property
    def id(self) -> TaskNodeId:
        return self._id

    def depends_on(self, *refs: _TaskNodeRef | str) -> _TaskNodeRef:
        for r in refs:
            if isinstance(r, str):
                dep_id = self._builder._name_to_id[r]
            else:
                dep_id = r._id
            self._builder._graph.add_dependency(self._id, dep_id)
        return self


class TaskGraphBuilder:
    """Fluent API for building a :class:`TaskGraph`."""

    def __init__(self) -> None:
        self._graph = TaskGraph()
        self._name_to_id: Dict[str, TaskNodeId] = {}

    def task(self, name: str, func: Callable, *args: Any, **kwargs: Any) -> _TaskNodeRef:
        nid = self._graph.add_task(name, func, *args, **kwargs)
        self._name_to_id[name] = nid
        return _TaskNodeRef(self, nid, name)

    def fence(self, name: str) -> _TaskNodeRef:
        nid = self._graph.add_fence(name)
        self._name_to_id[name] = nid
        return _TaskNodeRef(self, nid, name)

    def build(self) -> TaskGraph:
        self._graph.compile()
        return self._graph
