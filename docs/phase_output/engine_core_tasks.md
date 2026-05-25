# Investigation: engine/core/tasks/

**Date**: 2026-05-22  
**Total Lines**: 967  
**Classification**: REAL (100%)

## Summary

The tasks subsystem is a comprehensive job/task scheduling system with multiple execution models: thread pool, work-stealing, task graphs, fibers, and synchronization primitives. All six files contain complete, production-ready implementations.

## File Analysis

### scheduler.py (198 lines) - REAL

**Purpose**: TaskScheduler for parallel job execution with thread pool.

**Key Components**:
- `TaskHandle` class: Opaque handle wrapping concurrent.futures.Future
- `TaskScheduler` class:
  - `initialize()`: Start thread pool (auto-detect worker count)
  - `shutdown()`: Stop workers
  - `submit()`: Submit callable for execution
  - `submit_after()`: Submit with dependencies
  - `wait()` / `wait_all()`: Blocking result retrieval
  - `is_complete()`: Non-blocking status check
  - `parallel_for()`: Chunked parallel iteration

**Evidence of Real Implementation**:
- Full ThreadPoolExecutor lifecycle management
- Dependency-aware submission via blocking wrapper
- Proper chunking algorithm for parallel_for
- Priority support via TaskPriority (though not used in scheduling)

### graph.py (219 lines) - REAL

**Purpose**: Task DAG with dependency tracking and execution.

**Key Components**:
- `TaskState` enum: PENDING, READY, RUNNING, COMPLETE, FAILED
- `TaskNode` dataclass: Task metadata with dependencies/dependents tracking
- `TaskGraph` class:
  - `add_task()`: Add callable with args/kwargs
  - `add_dependency()`: Establish execution order
  - `add_fence()`: Barrier nodes
  - `compile()`: Topological sort with cycle detection
  - `execute()`: Run graph on TaskScheduler
  - `has_failures()` / `failed_nodes()`: Error inspection
- `TaskGraphBuilder` class: Fluent API with method chaining

**Evidence of Real Implementation**:
- Full Kahn's algorithm for topological sort
- Fence/barrier support for synchronization points
- Complete execution with proper dependency waiting
- Fluent builder with `depends_on()` chaining

### worker.py (193 lines) - REAL

**Purpose**: Work-stealing thread pool with per-worker deques.

**Key Components**:
- `TaskPriority` IntEnum: CRITICAL, HIGH, NORMAL, LOW, IDLE
- `TaskAffinity` enum: ANY, MAIN, WORKER, IO
- `WorkItem` dataclass: Unit of work with priority, affinity, future
- `WorkerThread` class:
  - Local deque with LIFO pop (owner) and FIFO steal (thief)
  - Lock-protected queue operations
  - Idle polling with configurable interval
- `WorkerPool` class:
  - N-worker management
  - Round-robin initial distribution
  - Random steal from peers

**Evidence of Real Implementation**:
- Classic work-stealing algorithm (Chase-Lev style deque semantics)
- Proper shutdown signaling via events
- Future integration for async results
- Daemon threads for clean process exit

### sync.py (184 lines) - REAL

**Purpose**: Synchronization primitives for task system.

**Key Components**:
- `TaskCounter`: Atomic counter with wait-until-zero
- `Future[T]` / `Promise[T]`: Async result pair (read/write sides)
- `Latch`: One-shot count-down barrier
- `Barrier`: Reusable N-party synchronization

**Evidence of Real Implementation**:
- All primitives built on threading.Event/Condition
- Generic typing for Future/Promise
- Proper timeout support on all wait operations
- Barrier uses generation counting for reusability

### fiber.py (133 lines) - REAL

**Purpose**: Cooperative coroutine/fiber support via asyncio.

**Key Components**:
- `Fiber` class:
  - Wraps asyncio coroutine
  - `yield_()` / `resume()`: Suspend/resume semantics via Event
  - Result tracking
- `FiberScheduler` class:
  - Background thread running asyncio event loop
  - `spawn()`: Schedule fiber for execution
  - `run_sync()`: Block until coroutine completes

**Evidence of Real Implementation**:
- Proper asyncio event loop in background thread
- Thread-safe submission via run_coroutine_threadsafe
- Done callback for result capture
- Clean shutdown with loop.stop() and thread.join()

### __init__.py (46 lines) - REAL

**Purpose**: Module exports.

**Exports**: TaskScheduler, TaskHandle, TaskGraph, TaskGraphBuilder, TaskNode, TaskNodeId, TaskState, TaskPriority, TaskAffinity, WorkerPool, WorkItem, TaskCounter, Future, Promise, Latch, Barrier, Fiber, FiberScheduler

## Architecture Quality

| Aspect | Rating | Notes |
|--------|--------|-------|
| Completeness | Very High | Multiple execution models, all working |
| Flexibility | Very High | Thread pool, work-stealing, DAG, fibers |
| Thread Safety | High | Proper locking, events, conditions |
| Error Handling | High | Exception propagation through futures |
| Documentation | High | Comprehensive docstrings and __init__ summary |

## Integration Points

- Consumes: `ENGINE_CORE_CONSTANTS` (FIBER_JOIN_TIMEOUT, WORKER_IDLE_POLL_INTERVAL)
- Produces: Parallel task execution, sync primitives, fiber scheduling
- External Dependencies: threading, concurrent.futures, asyncio

## Execution Models Comparison

| Model | Use Case | Scheduling |
|-------|----------|------------|
| TaskScheduler | General parallelism | ThreadPoolExecutor |
| TaskGraph | Dependency DAGs | Topological order |
| WorkerPool | High-throughput | Work-stealing |
| FiberScheduler | Cooperative multitasking | asyncio event loop |

## Gaps / Concerns

1. **Priority not enforced**: TaskPriority exists but TaskScheduler doesn't use it for ordering
2. **Affinity not enforced**: TaskAffinity exists but WorkerPool ignores it
3. **No cancellation**: Tasks cannot be cancelled once submitted
4. **Work-stealing granularity**: Single-item stealing, no batch stealing
5. **Fiber-to-thread bridging**: FiberScheduler is isolated, no direct integration with TaskScheduler
6. **No task introspection**: WorkerPool provides no way to query pending work count
