# Investigation: engine/core/scheduler/

**Date**: 2026-05-22  
**Total Lines**: 390  
**Classification**: REAL (100%)

## Summary

The scheduler subsystem is a fully implemented, production-ready phase-based system scheduler with dependency ordering and parallel dispatch. All five files contain complete, working implementations with no placeholder code.

## File Analysis

### scheduler.py (137 lines) - REAL

**Purpose**: Core SystemScheduler class that orchestrates phase-based execution.

**Key Components**:
- `_SystemEntry` dataclass: Internal record for registered systems (id, system, phase, run_if, access)
- `SystemScheduler` class with:
  - `register_system()`: Register callable/object with phase, conditional run, and access declarations
  - `add_dependency()`: Establish execution order between systems
  - `run()`: Execute all phases in order
  - `run_phase()`: Execute single phase with optional parallel dispatch
  - Internal topological sort cache for sorted system lists per phase

**Evidence of Real Implementation**:
- Complete integration with `SystemGraph` for dependency tracking
- `ParallelDispatcher` integration for concurrent execution
- Conditional execution via `run_if` callbacks
- Proper cache invalidation on registration/dependency changes

### parallel.py (106 lines) - REAL

**Purpose**: Parallel dispatch with conflict detection based on component access declarations.

**Key Components**:
- `SystemAccess` dataclass: Declares read/write component types per system
- `can_run_parallel()`: Conflict detection (read-read OK, read-write and write-write conflict)
- `compute_parallel_groups()`: Partitions systems into concurrency-safe groups
- `ParallelDispatcher` class: Executes groups via ThreadPoolExecutor

**Evidence of Real Implementation**:
- Complete conflict detection algorithm
- Proper exception handling with future cancellation
- Reusable executor with shutdown method

### graph.py (95 lines) - REAL

**Purpose**: Dependency graph with topological sort and cycle detection.

**Key Components**:
- `CycleDetectedError` exception
- `SystemGraph` dataclass with:
  - Node/edge management with reverse edge tracking
  - `topological_sort()`: Kahn's algorithm implementation
  - `detect_cycles()`: Validation wrapper
  - `get_parallel_groups()`: Level-based parallel grouping

**Evidence of Real Implementation**:
- Full Kahn's algorithm with sorted deterministic ordering
- Cycle detection with informative error messages
- Dual-use: topological sort and parallel group computation

### phases.py (34 lines) - REAL

**Purpose**: Phase definitions for frame execution.

**Key Components**:
- `Phase` IntEnum: PRE_UPDATE, UPDATE, POST_UPDATE, PRE_RENDER, RENDER, POST_RENDER
- `DEFAULT_PHASE_ORDER`: Standard phase sequence
- `PhaseGroup` dataclass: Ordered collection of system IDs within a phase

**Evidence of Real Implementation**:
- Complete phase enumeration matching standard game engine patterns
- PhaseGroup with add/remove operations

### __init__.py (18 lines) - REAL

**Purpose**: Module exports.

**Exports**: SystemScheduler, Phase, PhaseGroup, DEFAULT_PHASE_ORDER, SystemGraph, CycleDetectedError, SystemAccess, ParallelDispatcher, can_run_parallel

## Architecture Quality

| Aspect | Rating | Notes |
|--------|--------|-------|
| Completeness | High | All advertised functionality implemented |
| Separation of Concerns | High | Graph, parallel, phases, scheduler cleanly separated |
| Thread Safety | Medium | Executor-based parallelism, but cache invalidation not thread-safe |
| Error Handling | High | Cycle detection, exception propagation in parallel dispatch |
| Documentation | Medium | Good docstrings but sparse inline comments |

## Integration Points

- Consumes: `world` object, `delta_time` float
- Produces: Phased, dependency-ordered system execution
- External Dependencies: `concurrent.futures.ThreadPoolExecutor`

## Gaps / Concerns

1. **Cache thread safety**: `_sorted_cache` invalidation is not thread-safe if systems are registered during execution
2. **No priority support**: Systems within a phase/group execute in dependency order only, no priority scheduling
3. **Single executor**: ParallelDispatcher creates one executor at init time with no resize capability
