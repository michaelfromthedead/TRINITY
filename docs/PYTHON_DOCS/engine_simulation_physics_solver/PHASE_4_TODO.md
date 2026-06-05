# PHASE 4 TODO: Parallel Solving

---

## T-4.1: Add SolverConfig Dataclass

**File**: `engine/simulation/solver/config.py`

**Tasks**:
- [ ] Create `SolverConfig` dataclass
- [ ] Add `velocity_iterations: int = 10`
- [ ] Add `position_iterations: int = 4`
- [ ] Add `parallel: bool = True`
- [ ] Add `thread_count: int = 0` (0 = auto-detect cpu_count)
- [ ] Add `deterministic: bool = True`
- [ ] Add `parallel_threshold: int = 4` (min islands for parallel)

**Acceptance Criteria**:
- [ ] Default config produces same results as current sequential solver
- [ ] Config is immutable (frozen dataclass)

---

## T-4.2: Create IslandResult Return Type

**File**: `engine/simulation/solver/island_manager.py`

**Tasks**:
- [ ] Create `IslandResult` dataclass
- [ ] Include `body_velocities: Dict[int, Tuple[Vec3, Vec3]]` (linear, angular)
- [ ] Include `body_positions: Dict[int, Tuple[Vec3, Quaternion]]` (center, orientation)
- [ ] Include `warm_start_lambdas: Dict[int, float]` (per constraint)
- [ ] Include `island_id: int` for result ordering

**Acceptance Criteria**:
- [ ] Result fully captures island solve output
- [ ] Result can be applied without re-reading island structure

---

## T-4.3: Implement Island Sorting for Determinism

**File**: `engine/simulation/solver/island_manager.py`

**Tasks**:
- [ ] Add `Island.sort_key` property (min body index)
- [ ] Sort islands before parallel submission
- [ ] Ensure sort is stable (no randomness from hash collisions)

**Sort Key**:
```python
@property
def sort_key(self) -> int:
    return min(self.body_indices) if self.body_indices else 0
```

**Acceptance Criteria**:
- [ ] Same islands always sort to same order
- [ ] Sort is O(K log K) where K = island count

---

## T-4.4: Implement _solve_island Method

**File**: `engine/simulation/solver/island_manager.py`

**Tasks**:
- [ ] Extract single-island solving from current solve() method
- [ ] Return `IslandResult` instead of modifying bodies in place
- [ ] Copy body state at start, compute deltas, return deltas
- [ ] Ensure no writes to shared state (pure function)

**Interface**:
```python
def _solve_island(self, island: Island, dt: float) -> IslandResult:
    """Solve single island, return result without side effects."""
```

**Acceptance Criteria**:
- [ ] Method is thread-safe (no shared mutable state)
- [ ] Result contains all information needed to apply changes
- [ ] Sequential solve of all islands produces same result as current code

---

## T-4.5: Implement Parallel solve_parallel Method

**File**: `engine/simulation/solver/island_manager.py`

**Tasks**:
- [ ] Create `ThreadPoolExecutor` with configured thread count
- [ ] Sort islands by sort_key
- [ ] Submit `_solve_island` for each island
- [ ] Gather results in submission order (not completion order)
- [ ] Apply results sequentially

**Implementation**:
```python
def solve_parallel(self, islands: List[Island], dt: float) -> None:
    if len(islands) < self._config.parallel_threshold:
        # Fall back to sequential
        for island in islands:
            self._solve_sequential(island, dt)
        return
    
    # Sort for determinism
    sorted_islands = sorted(islands, key=lambda i: i.sort_key)
    
    # Parallel solve
    with ThreadPoolExecutor(max_workers=self._thread_count) as executor:
        futures = [executor.submit(self._solve_island, island, dt) 
                   for island in sorted_islands]
        results = [f.result() for f in futures]
    
    # Apply in order
    for result in results:
        self._apply_result(result)
```

**Acceptance Criteria**:
- [ ] Uses ThreadPoolExecutor (stdlib, no external deps)
- [ ] Gathers in submission order for determinism
- [ ] Falls back to sequential for small island counts

---

## T-4.6: Implement _apply_result Method

**File**: `engine/simulation/solver/island_manager.py`

**Tasks**:
- [ ] Apply velocity deltas to bodies
- [ ] Apply position/orientation deltas to bodies
- [ ] Update warm start lambdas for constraints
- [ ] Handle missing bodies gracefully (defensive)

**Implementation**:
```python
def _apply_result(self, result: IslandResult) -> None:
    for body_id, (linear_vel, angular_vel) in result.body_velocities.items():
        body = self._bodies[body_id]
        body.linear_velocity = linear_vel
        body.angular_velocity = angular_vel
    
    for body_id, (position, orientation) in result.body_positions.items():
        body = self._bodies[body_id]
        body.position = position
        body.orientation = orientation
```

**Acceptance Criteria**:
- [ ] All result data applied to bodies
- [ ] No race conditions (called sequentially)
- [ ] No exceptions if body was removed mid-solve (defensive)

---

## T-4.7: Add Thread Pool Lifecycle Management

**File**: `engine/simulation/solver/island_manager.py`

**Tasks**:
- [ ] Create thread pool once, reuse across frames
- [ ] Add shutdown method for clean exit
- [ ] Handle executor in context manager for safety
- [ ] Auto-detect thread count from `os.cpu_count()`

**Lifecycle**:
```python
class ParallelIslandSolver:
    def __init__(self, config: SolverConfig):
        self._config = config
        count = config.thread_count if config.thread_count > 0 else os.cpu_count()
        self._executor = ThreadPoolExecutor(max_workers=count)
    
    def shutdown(self):
        self._executor.shutdown(wait=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
```

**Acceptance Criteria**:
- [ ] Thread pool not recreated every frame
- [ ] Clean shutdown on program exit
- [ ] Configurable thread count

---

## T-4.8: Add Determinism Verification Tests

**File**: `tests/simulation/solver/test_parallel_determinism.py`

**Tasks**:
- [ ] Run same scene 100 times with parallel solver
- [ ] Verify all runs produce identical body positions
- [ ] Test with varying thread counts (1, 2, 4, 8)
- [ ] Test with random island counts and sizes

**Test Approach**:
```python
def test_determinism():
    scene = create_random_scene(bodies=100, islands=10)
    results = [run_simulation(scene, steps=60) for _ in range(100)]
    # All results must be bit-identical
    for r in results[1:]:
        assert r == results[0]
```

**Acceptance Criteria**:
- [ ] 100% determinism across 100 runs
- [ ] Determinism holds for all thread counts
- [ ] No floating-point ordering issues

---

## T-4.9: Add Parallel Performance Tests

**File**: `tests/simulation/benchmarks.py` (extend)

**Tasks**:
- [ ] Benchmark parallel vs sequential for K = 4, 8, 16, 32 islands
- [ ] Measure wall-clock time, not CPU time
- [ ] Compute speedup factor
- [ ] Profile GIL contention if speedup is low

**Expected Results**:
```
K=4 islands:  sequential=10ms, parallel=6ms,  speedup=1.7x
K=8 islands:  sequential=20ms, parallel=8ms,  speedup=2.5x
K=16 islands: sequential=40ms, parallel=12ms, speedup=3.3x
K=32 islands: sequential=80ms, parallel=20ms, speedup=4.0x
```

**Acceptance Criteria**:
- [ ] Speedup > 1.5x for K >= 4 on 4+ core machine
- [ ] No slowdown for K < 4 (sequential fallback works)
- [ ] Results documented in benchmarks report

---

## T-4.10: Integrate with PhysicsWorld

**File**: `engine/simulation/physics/physics_world.py`

**Tasks**:
- [ ] Add `solver_config` parameter to PhysicsWorld constructor
- [ ] Pass config to island solver
- [ ] Add `set_parallel(enabled: bool)` method for runtime toggle
- [ ] Default to parallel=True

**Acceptance Criteria**:
- [ ] PhysicsWorld uses parallel solver by default
- [ ] Can disable parallel at runtime for debugging
- [ ] No API breakage for existing code

---

## T-4.11: Document Thread Safety Guarantees

**File**: `engine/simulation/solver/island_manager.py` (docstrings)

**Tasks**:
- [ ] Document which methods are thread-safe
- [ ] Document which data structures are read-only during solve
- [ ] Document determinism guarantees and requirements
- [ ] Add example usage in module docstring

**Docstring Content**:
```python
"""
Parallel Island Solver

Thread Safety:
- solve_parallel() is safe to call from main thread only
- _solve_island() is thread-safe (no shared mutable state)
- Body state is read-only during parallel phase, written sequentially after

Determinism:
- Islands sorted by minimum body index before parallel dispatch
- Results gathered in submission order
- Floating-point results are bit-identical across runs

Usage:
    config = SolverConfig(parallel=True, thread_count=4)
    with ParallelIslandSolver(config) as solver:
        solver.solve_parallel(islands, dt)
"""
```

**Acceptance Criteria**:
- [ ] All thread safety rules documented
- [ ] Determinism contract explicit
- [ ] Usage example is copy-pasteable
