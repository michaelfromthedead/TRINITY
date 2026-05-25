"""Task/Job Scheduler subsystem.

Public API
----------
TaskScheduler   — Parallel job execution with thread pool
TaskGraph       — DAG of tasks with dependencies
TaskGraphBuilder — Fluent API for building task graphs
TaskHandle      — Opaque handle for submitted tasks
TaskPriority    — Priority levels (CRITICAL..IDLE)
TaskAffinity    — Thread affinity (ANY, MAIN, WORKER, IO)
TaskCounter     — Atomic counter with wait-until-zero
Future / Promise — Async result pair
Latch           — One-shot count-down barrier
Barrier         — Reusable N-party barrier
WorkerPool      — Work-stealing thread pool
Fiber           — Cooperative coroutine wrapper
FiberScheduler  — asyncio-backed fiber runner
"""

from engine.core.tasks.scheduler import TaskScheduler, TaskHandle
from engine.core.tasks.graph import TaskGraph, TaskGraphBuilder, TaskNode, TaskNodeId, TaskState
from engine.core.tasks.worker import TaskPriority, TaskAffinity, WorkerPool, WorkItem
from engine.core.tasks.sync import TaskCounter, Future, Promise, Latch, Barrier
from engine.core.tasks.fiber import Fiber, FiberScheduler

__all__ = [
    "TaskScheduler",
    "TaskHandle",
    "TaskGraph",
    "TaskGraphBuilder",
    "TaskNode",
    "TaskNodeId",
    "TaskState",
    "TaskPriority",
    "TaskAffinity",
    "WorkerPool",
    "WorkItem",
    "TaskCounter",
    "Future",
    "Promise",
    "Latch",
    "Barrier",
    "Fiber",
    "FiberScheduler",
]
