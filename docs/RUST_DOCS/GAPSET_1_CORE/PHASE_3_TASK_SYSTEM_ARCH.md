# PHASE 3: Task/Job System

**Scope:** Implement a work-stealing thread pool with job graph dependency tracking and parallel_for iteration.
**Depends on:** Phase 0 (omega crate for base types, crossbeam dependency declared in Cargo.toml)
**Produces:** Planned: ThreadPool (work-stealing), JobGraph (dependency DAG), parallel_for (range splitting)
**Status:** NOT IMPLEMENTED (0/4 tasks DONE)

## 1. Overview

Phase 3 is the planned task/parallelism system for the TRINITY engine core. It does not yet exist in any form -- no Rust ThreadPool, no JobGraph, no parallel_for implementation. The `crossbeam` crate is already a declared dependency in `renderer-backend/Cargo.toml` (v0.8) and `omega/Cargo.toml`, providing the deque and synchronization primitives needed for work-stealing, but no task infrastructure has been built with it.

The architecture (from PHASE_N_TODO.md acceptance criteria) follows a three-layer design:

- **ThreadPool (work-stealing):** N worker threads, each with a local deque of tasks. Idle workers steal tasks from sibling deques. Priority-aware scheduling with 6 levels. Auto-detect worker count (0 = num CPUs).
- **JobGraph (dependency DAG):** A compiled DAG where nodes are tasks and edges are dependencies (Task A must complete before Task B starts). `JobGraphBuilder` constructs the graph with cycle detection. `TaskHandle` provides non-blocking completion checks.
- **parallel_for (range splitting):** Splits a range `[start, end)` into chunks distributed via the JobGraph. Configurable chunk size; `chunk_size=0` auto-sizes to worker count. Returns when all chunks complete.

The system maps to the GAP 3 activation model: a standalone crate with optional PyO3 bindings, fallible at import time (Python falls back to sequential execution), and progressive Rust acceleration.

## 2. Architectural decisions (planned)

- **crossbeam-deque for work-stealing.** crossbeam 0.8 provides `Worker<T>` (local deque, push/pop) and `Stealer<T>` (steal from sibling). This is the standard Rust work-stealing substrate, used by rayon.
- **JobGraph compiled before execution.** `JobGraphBuilder` validates the DAG (cycle detection) and produces a compiled `JobGraph` with topological ordering. The execution engine dispatches ready tasks to the thread pool, tracking completion counts per node.
- **Priority-aware scheduling with 6 levels.** HIGHEST, HIGH, NORMAL, LOW, LOWEST, BACKGROUND. Higher-priority tasks execute before lower-priority ones within the same worker's deque. Priority inversion prevented by design (no locks between priority levels).
- **parallel_for builds a JobGraph internally.** Rather than a standalone function, `parallel_for(range, chunk_size, f)` constructs a mini JobGraph with N chunks as independent tasks, submits to the thread pool, and blocks until completion.
- **Python fallback via try/import.** Following the established pattern (see `_HAVE_OMEGA` in world.py), the scheduler bridge should import `_omega.scheduler_submit_job_graph` with a fallback to sequential execution when the Rust module isn't available.
- **A standalone crate or module within renderer-backend.** The thread pool could live in `crates/renderer-backend/src/scheduler/` or as a separate `trinity-scheduler` crate. Given the existing dependency structure, a module in renderer-backend is simpler and avoids adding another workspace member.

## 3. Constraints specific to this phase

- Workers must not panic on task failure; task errors must be returned or propagated to the caller.
- Work-stealing must be lock-free on the fast path (local push/pop) with minimal synchronization on steal.
- JobGraph cycle detection must reject circular dependencies at build time, not at execution time.
- parallel_for must invoke the closure exactly once for each index in the range, with no duplicate or skipped indices.
- Shutdown must drain all submitted tasks before joining threads.
- Single-threaded mode (`num_workers = 0`) must produce deterministic execution order for reproducible debugging.

## 4. Component breakdown (planned -- not yet implemented)

| File/Component | Role | Status |
|----------------|------|--------|
| (planned) `crates/renderer-backend/src/scheduler/mod.rs` | ThreadPool with work-stealing deque, worker threads, priority scheduling, shutdown. | NOT IMPLEMENTED |
| (planned) `crates/renderer-backend/src/scheduler/job_graph.rs` | JobGraph, JobGraphBuilder (add_task, depends_on, finalize), TaskHandle (is_complete). | NOT IMPLEMENTED |
| (planned) `crates/renderer-backend/src/scheduler/parallel.rs` | parallel_for(range, chunk_size, function) constructing internal JobGraph. | NOT IMPLEMENTED |
| (planned) `omega/src/bridge.rs` addition | scheduler_submit_job_graph PyO3 function for Python→Rust job dispatch. | NOT IMPLEMENTED |
| `engine/platform/os/threading.py` | Python threading utilities (not the GAP 1 spec). | EXISTS (not related) |

**Dependency chain:** `parallel_for` depends on `JobGraph`, which depends on `ThreadPool`, which depends only on `crossbeam`.

## 5. Testing strategy (planned)

- ThreadPool: 4-worker pool runs 4 tasks concurrently; work-stealing balances load; shutdown drains and joins.
- JobGraph: diamond dependency (A->B, A->C, B->C) executes correctly; cycle detection rejects circular deps; 1000-node graph builds and runs.
- parallel_for: each index 0..999 called exactly once; chunk_size=0 auto-sizes; speedup vs sequential >3x on 4-core.
- Priority inversion test: HIGH tasks always complete before LOW tasks.
- Single-threaded mode: deterministic execution order.
- Throughput: 10k tasks at >50k tasks/sec on 4-core.

## 6. Open questions

- **Standalone crate vs. module in renderer-backend?** A module in renderer-backend (`crates/renderer-backend/src/scheduler/`) is the simplest path -- it already has crossbeam as a dependency. However, if the thread pool should be independently usable (e.g., by a physics crate), a separate `trinity-scheduler` crate would be cleaner.
- **Should parallel_for use rayon instead of a custom JobGraph?** Rayon provides parallel iteration, but doesn't give the explicit JobGraph dependency model needed for the scheduler bridge (T-CORE-5.5). A custom JobGraph over crossbeam-deque allows frame-phase ordering that rayon can't express.
- **PyO3 bridge for job submission:** The scheduler bridge (T-CORE-5.5) needs Python→Rust job graph submission. Should job graphs be constructed in Python and serialized to Rust, or should Python submit individual tasks to a Rust-managed graph?
- **Integration with existing Python threading:** The existing `engine/platform/os/threading.py` provides Python-level threading utilities. Should the Rust ThreadPool replace these, or should they coexist with Python for non-performance-critical paths?

## 7. References

- GAP_1_SUMMARY.md -- Investigation for T-CORE-3.1 through T-CORE-3.4 (all ABSENT)
- PHASE_N_TODO.md -- Detailed acceptance criteria for each planned component
- CLARIFICATION.md -- Rationale for deferring Phase 3 (lowest priority, not blocking any gap set)
- crossbeam 0.8 crate -- Provides `crossbeam-deque` for work-stealing and `crossbeam-channel` for inter-thread communication (already a dependency)
- `engine/platform/os/threading.py` -- Existing Python threading utilities (not a replacement)
