# PHASE 4 ARCHITECTURE: Parallel Solving

---

## Phase Overview

Implement actual parallel constraint solving for the `ParallelIslandSolver.solve_parallel()` method, which the investigation identified as a sequential placeholder. Enable multi-core utilization for physics-heavy scenes.

---

## Architectural Decisions

### AD-4.1: Island-Level Parallelism

**Decision**: Parallelize at island granularity, not constraint granularity.

**Rationale**:
- Islands are independent by definition (no contacts between them)
- No synchronization needed between island solvers
- Constraint-level parallelism requires complex ordering (Jacobi vs Gauss-Seidel trade-offs)
- Island-level is embarrassingly parallel

**Trade-off**: Large single islands cannot parallelize internally. This is acceptable because:
- Most scenes have multiple islands
- Single large island = everything connected = realistic to solve sequentially

### AD-4.2: ThreadPoolExecutor for Parallelism

**Decision**: Use `concurrent.futures.ThreadPoolExecutor` for parallel execution.

**Rationale**:
- Standard library, no external dependencies
- Simple map/submit API
- GIL concern mitigated: solver is compute-bound, releases GIL in numeric operations
- Alternative (multiprocessing) has serialization overhead for body state

**GIL Mitigation**:
- Pure Python math releases GIL during long computations
- Future: consider native extension for solver hot path

### AD-4.3: Deterministic Results

**Decision**: Enforce deterministic results regardless of thread count.

**Rationale**:
- Physics must be reproducible for debugging, replays, networking
- Non-determinism from race conditions is unacceptable

**Implementation**:
- Sort islands by consistent key (e.g., body ID of root)
- Process islands in sorted order (within parallel batch)
- Accumulate results in pre-allocated array by island index

### AD-4.4: Work Stealing for Load Balancing

**Decision**: Rely on ThreadPoolExecutor's built-in work distribution (not explicit work stealing).

**Rationale**:
- ThreadPoolExecutor submits tasks to a queue; threads pull work as available
- Sufficient for coarse-grained island tasks
- Custom work stealing adds complexity without proven benefit at this scale

---

## Component Boundaries

### Modified Components

| Component | Change |
|-----------|--------|
| `island_manager.py` | `ParallelIslandSolver.solve_parallel()` becomes truly parallel |
| `island_manager.py` | Add island sorting for determinism |
| `constraint_solver.py` | No change (islands solved independently) |

### New Components

| Component | Responsibility |
|-----------|----------------|
| `SolverConfig` | Thread count, determinism settings |

---

## Interfaces

### SolverConfig

```python
@dataclass
class SolverConfig:
    velocity_iterations: int = 10
    position_iterations: int = 4
    parallel: bool = True
    thread_count: int = 0  # 0 = auto (cpu_count)
    deterministic: bool = True
```

### ParallelIslandSolver (modified)

```python
class ParallelIslandSolver:
    def __init__(self, config: SolverConfig):
        self._config = config
        self._executor: Optional[ThreadPoolExecutor] = None
        
    def solve_parallel(self, islands: List[Island], dt: float) -> None:
        """
        Solve all islands in parallel.
        
        Deterministic: Islands sorted by root body ID before processing.
        Results written back to body state arrays in island order.
        """
        
    def _solve_island(self, island: Island, dt: float) -> IslandResult:
        """Solve single island, return velocity/position deltas."""
        
    def _apply_results(self, results: List[IslandResult]) -> None:
        """Apply all results to body state."""
```

### Island (no change)

```python
@dataclass
class Island:
    body_indices: List[int]
    contact_indices: List[int]
    joint_indices: List[int]
```

---

## Data Flow

```
ParallelIslandSolver.solve_parallel(islands)
    |
    +-- Sort islands by root body ID (determinism)
    |
    +-- Submit _solve_island() to ThreadPoolExecutor for each island
    |       |
    |       +-- (Thread 1) solve island 0, return deltas
    |       +-- (Thread 2) solve island 1, return deltas
    |       +-- (Thread N) solve island M, return deltas
    |
    +-- Gather all futures in island order
    |
    +-- Apply results sequentially to avoid races
```

---

## Thread Safety Analysis

### Read-Only During Solve

- Body mass, inertia tensors
- Constraint definitions (contacts, joints)
- Solver parameters

### Written Per-Island (no conflicts)

- Body velocities (linear, angular)
- Body positions (center, orientation)
- Constraint cached data (warm start lambdas)

### Why No Races

Each island has disjoint set of body indices. Bodies in different islands are, by definition, not in contact. Therefore:
- Island A writes to bodies {0, 1, 5}
- Island B writes to bodies {2, 3, 4}
- No overlap, no race

---

## Determinism Strategy

### Problem

Thread scheduling is non-deterministic. If results depend on execution order, simulation diverges.

### Solution

1. **Sort islands** by consistent key before submission
2. **Gather results** in submission order (not completion order)
3. **Apply results** sequentially in sorted order

```python
islands_sorted = sorted(islands, key=lambda i: min(i.body_indices))
futures = [executor.submit(self._solve_island, island, dt) for island in islands_sorted]
results = [f.result() for f in futures]  # Blocks in order
for result in results:
    self._apply_results(result)
```

Result: Same islands, same order, same floating-point operations, same output.

---

## Performance Model

### Assumptions

- N = total bodies
- K = number of islands
- Average island size = N/K
- Solver iterations = 10

### Sequential Time

```
T_seq = K * (N/K) * iterations * cost_per_body
      = N * iterations * cost_per_body
```

### Parallel Time (P threads)

```
T_par = (K/P) * (N/K) * iterations * cost_per_body
      = (N/P) * iterations * cost_per_body
```

Ideal speedup = P (linear with thread count).

### Reality

- GIL contention reduces speedup for pure Python
- Amdahl's law: serial portions (sorting, result application) limit speedup
- Overhead: thread creation, future management

Expected speedup: 2-4x on 8-core system for typical scenes.

---

## Dependencies

### Internal

- `engine/simulation/solver/island_manager.py`
- `engine/simulation/solver/constraint_solver.py`

### External

- `concurrent.futures` (stdlib)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GIL limits parallelism | HIGH | Profile actual speedup; consider Cython for hot path |
| Non-deterministic results | HIGH | Strict sort-gather-apply order |
| Thread pool overhead > benefit for small scenes | MEDIUM | Fall back to sequential for K < 4 islands |
| Deadlock in thread pool | LOW | Use executor context manager, no nested parallelism |
