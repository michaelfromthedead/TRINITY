# PHASE 3 TODO: Scheduling and Data Flow Decorators

## Summary
Implement scheduling decorators (fixed timestep, async, ordering) and data flow decorators (serialization, snapshots).

---

## T-DEC-3.1: Implement @fixed Decorator

**File**: `trinity/decorators/scheduling.py`

**Task**: Fixed timestep scheduling decorator.

**Acceptance Criteria**:
- [ ] `@fixed(hz=60)` syntax
- [ ] Attaches `_fixed = True`
- [ ] Attaches `_fixed_hz = hz`
- [ ] Attaches `_fixed_delta = 1.0 / hz`
- [ ] Default hz from `DEFAULT_PHYSICS_HZ` constant
- [ ] Validates hz > 0

---

## T-DEC-3.2: Implement @async_system Decorator

**File**: `trinity/decorators/scheduling.py`

**Task**: Mark and detect async systems.

**Acceptance Criteria**:
- [ ] Attaches `_async_system = True`
- [ ] Detects via `asyncio.iscoroutinefunction(target)`
- [ ] Attaches `_is_coroutine = True/False`
- [ ] Works on both functions and methods

---

## T-DEC-3.3: Implement @parallel and @exclusive Decorators

**File**: `trinity/decorators/scheduling.py`

**Task**: Parallelism control decorators.

**Acceptance Criteria**:
- [ ] `@parallel` attaches `_parallel = True`
- [ ] `@exclusive` attaches `_exclusive = True`
- [ ] Validation: parallel and exclusive are mutually exclusive
- [ ] Error raised if both applied

---

## T-DEC-3.4: Implement @before/@after Ordering Decorators

**File**: `trinity/decorators/scheduling.py`

**Task**: System ordering decorators.

**Acceptance Criteria**:
- [ ] `@before("SystemX")` attaches `_before = ["SystemX"]`
- [ ] `@after("SystemY")` attaches `_after = ["SystemY"]`
- [ ] Multiple targets: `@before("A", "B")`
- [ ] Validation: no self-reference
- [ ] Cycle detection at registration time

---

## T-DEC-3.5: Implement @stage, @throttle, @chain Decorators

**File**: `trinity/decorators/scheduling.py`

**Task**: Additional scheduling decorators.

**Acceptance Criteria**:
- [ ] `@stage(2)` attaches `_stage = 2`
- [ ] `@throttle(hz=10)` attaches `_throttle_hz = 10`
- [ ] `@chain("A", "B", "C")` attaches `_chain = ["A", "B", "C"]`
- [ ] Validates stage >= 0
- [ ] Validates throttle hz > 0

---

## T-DEC-3.6: Implement Remaining Scheduling Decorators

**File**: `trinity/decorators/scheduling.py`

**Task**: Complete all 12 scheduling decorators.

**Acceptance Criteria**:
- [ ] @startup - run once at startup
- [ ] @shutdown - run once at shutdown
- [ ] @on_event(event) - event-triggered
- [ ] @lazy - deferred execution
- [ ] Each with appropriate metadata

---

## T-DEC-3.7: Implement @serializable Decorator

**File**: `trinity/decorators/data_flow.py`

**Task**: Generate serialize/deserialize methods.

**Acceptance Criteria**:
- [ ] `@serializable(version=1, fields=["x", "y"])`
- [ ] Generates `serialize(cls, obj) -> dict` classmethod
- [ ] Generates `deserialize(cls, data) -> obj` classmethod
- [ ] Output includes `__version__` and `__type__`
- [ ] Only specified fields serialized
- [ ] Auto-detect fields if not specified (from `__annotations__`)

---

## T-DEC-3.8: Implement @snapshot Decorator

**File**: `trinity/decorators/data_flow.py`

**Task**: State history with ring buffer.

**Acceptance Criteria**:
- [ ] `@snapshot(frames=60)` syntax
- [ ] Attaches `_snapshot_history: list = []`
- [ ] Attaches `_snapshot_history_frames = 60`
- [ ] Generates `snapshot_save(self) -> int` returning frame index
- [ ] Generates `snapshot_restore(self, frame_idx)` method
- [ ] Ring buffer behavior: oldest evicted when full
- [ ] Validates frames > 0

---

## T-DEC-3.9: Implement @networked Decorator

**File**: `trinity/decorators/data_flow.py`

**Task**: Network synchronization configuration.

**Acceptance Criteria**:
- [ ] `@networked(priority=1, reliable=True, interpolate=True)`
- [ ] Attaches `_networked_config` dict with all params
- [ ] Attaches `_networked = True`
- [ ] Validates priority >= 0

---

## T-DEC-3.10: Implement @delta Decorator

**File**: `trinity/decorators/data_flow.py`

**Task**: Delta compression for network/serialization.

**Acceptance Criteria**:
- [ ] `@delta(fields=["position", "rotation"])`
- [ ] Attaches `_delta_fields = ["position", "rotation"]`
- [ ] Generates `compute_delta(self, previous) -> dict` method
- [ ] Generates `apply_delta(self, delta)` method

---

## Dependencies

```
PHASE 1 ──> T-DEC-3.1 ──┐
            T-DEC-3.2 ──┼──> T-DEC-3.3 ──> T-DEC-3.4 ──> T-DEC-3.5 ──> T-DEC-3.6
                        │
PHASE 1 ──> T-DEC-3.7 ──┼──> T-DEC-3.8 ──> T-DEC-3.9 ──> T-DEC-3.10
```

## Estimated Effort

| Task | Lines | Complexity |
|------|-------|------------|
| T-DEC-3.1 | ~40 | Low |
| T-DEC-3.2 | ~30 | Low |
| T-DEC-3.3 | ~40 | Low |
| T-DEC-3.4 | ~50 | Medium |
| T-DEC-3.5 | ~60 | Low |
| T-DEC-3.6 | ~80 | Low |
| T-DEC-3.7 | ~100 | Medium |
| T-DEC-3.8 | ~80 | Medium |
| T-DEC-3.9 | ~40 | Low |
| T-DEC-3.10 | ~60 | Medium |
