# PHASE 4 TODO: ECS and Dev Decorators

## Summary
Implement ECS integration decorators and developer tool decorators.

---

## T-DEC-4.1: Implement @component Decorator

**File**: `trinity/decorators/ecs_core.py`

**Task**: Register class as ECS component with metaclass integration.

**Acceptance Criteria**:
- [ ] `@component` syntax (no parameters)
- [ ] Calls `ComponentMeta._process_fields(target)`
- [ ] Calls `ComponentMeta._install_descriptors(target)`
- [ ] Assigns unique `_component_id` under lock
- [ ] Registers in `ComponentMeta._registry`
- [ ] Idempotent: no-op if already registered

---

## T-DEC-4.2: Implement @system Decorator

**File**: `trinity/decorators/ecs_core.py`

**Task**: Register function as ECS system.

**Acceptance Criteria**:
- [ ] `@system` marks function as system
- [ ] Attaches `_system = True`
- [ ] Extracts queries from type hints via `_extract_queries()`
- [ ] Attaches `_system_queries = [...]`
- [ ] Works on both functions and methods

---

## T-DEC-4.3: Implement Query Extraction

**File**: `trinity/decorators/ecs_core.py`

**Task**: Extract Query types from function type hints.

**Acceptance Criteria**:
- [ ] `_extract_queries(fn) -> list[Query]`
- [ ] Uses `typing.get_type_hints()`
- [ ] Identifies `Query[...]` via `get_origin()`
- [ ] Returns list of Query type parameters
- [ ] Handles `Optional`, `Union` wrappers

---

## T-DEC-4.4: Implement @resource, @event, @command Decorators

**File**: `trinity/decorators/ecs_core.py`

**Task**: ECS primitive decorators.

**Acceptance Criteria**:
- [ ] `@resource` attaches `_resource = True`
- [ ] `@event` attaches `_event = True`
- [ ] `@command` attaches `_command = True`
- [ ] Each marks class for appropriate registry

---

## T-DEC-4.5: Implement @query, @bundle, @archetype Decorators

**File**: `trinity/decorators/ecs_core.py`

**Task**: Advanced ECS decorators.

**Acceptance Criteria**:
- [ ] `@query(with_=[A, B], without=[C])` - query configuration
- [ ] `@bundle(components=[A, B, C])` - component grouping
- [ ] `@archetype(components=[A, B])` - archetype hint
- [ ] Validates component types exist

---

## T-DEC-4.6: Implement @world Decorator

**File**: `trinity/decorators/ecs_core.py`

**Task**: World configuration decorator.

**Acceptance Criteria**:
- [ ] `@world(max_entities=10000)` syntax
- [ ] Attaches `_world_config` dict
- [ ] Validates max_entities > 0

---

## T-DEC-4.7: Implement @profile Decorator

**File**: `trinity/decorators/dev.py`

**Task**: Timing statistics decorator.

**Acceptance Criteria**:
- [ ] Wraps function with timing
- [ ] Uses `time.perf_counter()` for precision
- [ ] Tracks: call_count, total_ms, min_ms, max_ms
- [ ] Stats accessible via `target._profile_stats`
- [ ] Uses `functools.wraps` to preserve metadata

---

## T-DEC-4.8: Implement @trace Decorator

**File**: `trinity/decorators/dev.py`

**Task**: Debug logging decorator.

**Acceptance Criteria**:
- [ ] Logs function entry with arguments
- [ ] Logs function exit with return value
- [ ] Uses `logging.debug()` level
- [ ] Attaches `_trace = True`
- [ ] Configurable log format

---

## T-DEC-4.9: Implement @deprecated Decorator

**File**: `trinity/decorators/dev.py`

**Task**: Deprecation warning decorator.

**Acceptance Criteria**:
- [ ] `@deprecated("Use X instead")` syntax
- [ ] Emits `DeprecationWarning` on call
- [ ] Warning includes reason
- [ ] `stacklevel=2` for correct caller location
- [ ] Attaches `_deprecated_reason`

---

## T-DEC-4.10: Implement Remaining Dev Decorators

**File**: `trinity/decorators/dev.py`

**Task**: Complete all 9 dev decorators.

**Acceptance Criteria**:
- [ ] @debug_only - only runs in debug build
- [ ] @todo(msg) - marks incomplete code
- [ ] @experimental - experimental API warning
- [ ] @internal - internal API marker
- [ ] @benchmark - marks benchmark targets
- [ ] @hotpath - marks performance-critical code
- [ ] Each with appropriate metadata

---

## Dependencies

```
PHASE 1 ──> T-DEC-4.1 ──> T-DEC-4.2 ──> T-DEC-4.3
                │
                v
            T-DEC-4.4 ──> T-DEC-4.5 ──> T-DEC-4.6

PHASE 1 ──> T-DEC-4.7 ──> T-DEC-4.8 ──> T-DEC-4.9 ──> T-DEC-4.10
```

Note: T-DEC-4.1 through T-DEC-4.6 depend on `trinity.metaclasses.ComponentMeta`.

## Estimated Effort

| Task | Lines | Complexity |
|------|-------|------------|
| T-DEC-4.1 | ~50 | Medium |
| T-DEC-4.2 | ~40 | Medium |
| T-DEC-4.3 | ~60 | Medium |
| T-DEC-4.4 | ~40 | Low |
| T-DEC-4.5 | ~80 | Medium |
| T-DEC-4.6 | ~30 | Low |
| T-DEC-4.7 | ~60 | Medium |
| T-DEC-4.8 | ~50 | Low |
| T-DEC-4.9 | ~40 | Low |
| T-DEC-4.10 | ~100 | Low |
