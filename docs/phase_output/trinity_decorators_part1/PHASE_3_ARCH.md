# PHASE 3 ARCHITECTURE: Scheduling and Data Flow Decorators

## Phase Scope

Runtime coordination decorators: `scheduling.py` (Tier 3), `data_flow.py` (Tier 4)

## Architecture Decisions

### ADR-DEC-009: Fixed Timestep Scheduling

**Context**: Physics simulation requires deterministic time steps regardless of frame rate.

**Decision**: `@fixed` decorator with Hz configuration:
```python
@fixed(hz=60)
def physics_update(dt: float):
    ...
```

Attaches:
- `_fixed = True`
- `_fixed_hz = 60`
- `_fixed_delta = 1.0 / 60  # ~0.0167`

**Consequences**:
- Physics runs at constant rate
- Multiple physics updates per frame if needed
- Interpolation required for rendering

### ADR-DEC-010: Async Coroutine Detection

**Context**: Some systems are async, need different execution path.

**Decision**: `@async_system` detects coroutine functions:
```python
target._async_system = True
target._is_coroutine = asyncio.iscoroutinefunction(target)
```

**Consequences**:
- Scheduler knows to await async systems
- Sync systems called directly
- Mixed sync/async execution supported

### ADR-DEC-011: Serialization with Version Tracking

**Context**: Save/load needs version migration support.

**Decision**: `@serializable` generates `serialize()`/`deserialize()` methods:
```python
@classmethod
def serialize(cls, obj: Any) -> dict:
    return {
        "__version__": cls._serializable_version,
        "__type__": cls.__name__,
        **{f: getattr(obj, f) for f in cls._serializable_fields}
    }
```

**Consequences**:
- Version field enables migration
- Type field enables polymorphic deserialization
- Only marked fields serialized

### ADR-DEC-012: Snapshot History Ring Buffer

**Context**: Need history for rollback/interpolation.

**Decision**: Fixed-size ring buffer:
```python
if len(self._snapshot_history) >= self._snapshot_history_frames:
    self._snapshot_history.pop(0)
self._snapshot_history.append(state)
```

**Consequences**:
- Bounded memory usage
- O(1) append
- Oldest snapshot lost when full

## Component Diagram

```
+-----------------+
|  scheduling.py  |  @fixed, @parallel, @exclusive, @async_system
+--------+--------+
         |
         +-- _after_fixed() --> Hz/delta calculation
         |
         +-- _after_async_system() --> Coroutine detection
         |
         +-- Chain linking for dependent systems

+-----------------+
|  data_flow.py   |  @serializable, @networked, @snapshot
+--------+--------+
         |
         +-- serialize()/deserialize() generation
         |
         +-- snapshot_save()/snapshot_restore()
         |
         +-- Ring buffer history
```

## Scheduling Decorator Matrix

| Decorator | Purpose | Key Metadata |
|-----------|---------|--------------|
| @fixed | Fixed timestep | `_fixed_hz`, `_fixed_delta` |
| @parallel | Can run in parallel | `_parallel=True` |
| @exclusive | Must run alone | `_exclusive=True` |
| @async_system | Async execution | `_is_coroutine` |
| @before(X) | Run before system X | `_before=["X"]` |
| @after(X) | Run after system X | `_after=["X"]` |
| @stage(N) | Run in stage N | `_stage=N` |
| @throttle(hz) | Max execution rate | `_throttle_hz` |
| @chain(X, Y) | Link systems | `_chain=["X", "Y"]` |

## Data Flow Decorator Matrix

| Decorator | Purpose | Key Metadata |
|-----------|---------|--------------|
| @serializable | Save/load | `_serializable_fields`, `_serializable_version` |
| @networked | Network sync | `_networked_config` |
| @snapshot | State history | `_snapshot_history`, `_snapshot_history_frames` |
| @delta | Delta compression | `_delta_fields` |

## Scheduling Constraints

```
parallel ∧ exclusive = ERROR
fixed ∧ throttle = WARNING (fixed takes priority)
before(X) ∧ after(X) = ERROR (cycle)
```

## Serialization Type Support

| Type | Serialization | Notes |
|------|---------------|-------|
| int, float, str, bool | Direct | JSON-safe |
| list, dict | Recursive | Nested serialization |
| Entity | ID reference | Requires registry lookup |
| Component | Full serialize | If @serializable |
| Custom class | `__getstate__` | Fallback |
