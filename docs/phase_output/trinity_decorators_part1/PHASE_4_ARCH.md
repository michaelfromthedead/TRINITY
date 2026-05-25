# PHASE 4 ARCHITECTURE: ECS and Dev Decorators

## Phase Scope

ECS integration: `ecs_core.py` (Tier 1), Developer tools: `dev.py` (Tier 6)

## Architecture Decisions

### ADR-DEC-013: ComponentMeta Integration

**Context**: `@component` must integrate with existing metaclass system.

**Decision**: Direct metaclass method invocation:
```python
def _after_component(target: Any, params: dict[str, Any]) -> Any:
    if not hasattr(target, "_component_id"):
        from trinity.metaclasses import ComponentMeta
        ComponentMeta._process_fields(target)
        ComponentMeta._install_descriptors(target)
        with ComponentMeta._lock:
            target._component_id = ComponentMeta._next_id
            ComponentMeta._next_id += 1
            ComponentMeta._registry[target._component_id] = target
```

**Consequences**:
- Decorator and metaclass share registry
- Thread-safe ID assignment
- Field descriptors installed for component access

### ADR-DEC-014: Query Extraction from Type Hints

**Context**: System functions declare component queries via type hints.

**Decision**: Extract Query types from function annotations:
```python
def _extract_queries(fn: Callable[..., Any]) -> list[Any]:
    hints = get_type_hints(fn)
    for param_name, param_type in hints.items():
        origin = get_origin(param_type)
        if origin.__name__ == "Query":
            queries.append(param_type)
```

**Consequences**:
- Type hints are source of truth
- Query[Position, Velocity] parsed correctly
- Runtime type checking possible

### ADR-DEC-015: Profile Timing Statistics

**Context**: Need real timing data for performance analysis.

**Decision**: Wrapper with statistics dictionary:
```python
stats = {
    "call_count": 0,
    "total_ms": 0.0,
    "min_ms": float("inf"),
    "max_ms": 0.0,
}

@functools.wraps(target)
def wrapper(*args, **kwargs):
    start = time.perf_counter()
    result = target(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    stats["call_count"] += 1
    stats["total_ms"] += elapsed
    stats["min_ms"] = min(stats["min_ms"], elapsed)
    stats["max_ms"] = max(stats["max_ms"], elapsed)
    return result
```

**Consequences**:
- Per-function statistics available
- Overhead minimal (perf_counter is fast)
- Stats accessible via `target._profile_stats`

### ADR-DEC-016: Deprecation Warnings

**Context**: Need graceful deprecation path for API changes.

**Decision**: `@deprecated` emits warnings.warn on call:
```python
@functools.wraps(target)
def wrapper(*args, **kwargs):
    warnings.warn(
        f"{target.__name__} is deprecated: {reason}",
        DeprecationWarning,
        stacklevel=2
    )
    return target(*args, **kwargs)
```

**Consequences**:
- Warning shows caller's location
- Can be filtered via warnings module
- Message includes reason

## Component Diagram

```
+----------------+
|  ecs_core.py   |  @component, @system, @query, @resource
+-------+--------+
        |
        +-- ComponentMeta._process_fields()
        |
        +-- ComponentMeta._install_descriptors()
        |
        +-- _extract_queries() from type hints

+----------------+
|    dev.py      |  @profile, @trace, @deprecated, @debug_only
+-------+--------+
        |
        +-- time.perf_counter() timing
        |
        +-- warnings.warn() deprecation
        |
        +-- logging.debug() traces
```

## ECS Decorator Matrix

| Decorator | Purpose | Key Metadata |
|-----------|---------|--------------|
| @component | Register component type | `_component_id` |
| @system | Register system function | `_system=True`, queries |
| @query | Query configuration | `_query_config` |
| @resource | Singleton resource | `_resource=True` |
| @event | Event type | `_event=True` |
| @command | Command buffer | `_command=True` |
| @world | World configuration | `_world_config` |
| @bundle | Component bundle | `_bundle_components` |
| @archetype | Archetype hint | `_archetype_components` |

## Dev Decorator Matrix

| Decorator | Purpose | Key Metadata |
|-----------|---------|--------------|
| @profile | Timing stats | `_profile_stats` |
| @trace | Debug logging | `_trace=True` |
| @deprecated(reason) | Deprecation warning | `_deprecated_reason` |
| @debug_only | Debug build only | `_debug_only=True` |
| @todo(msg) | TODO marker | `_todo_message` |
| @experimental | Experimental API | `_experimental=True` |
| @internal | Internal API | `_internal=True` |
| @benchmark | Benchmark target | `_benchmark=True` |
| @hotpath | Hot code marker | `_hotpath=True` |

## Query Type Parsing

```python
Query[Position, Velocity]  # -> components = [Position, Velocity]
Query[Position, !Velocity]  # -> include = [Position], exclude = [Velocity]
Query[Position, Optional[Velocity]]  # -> required = [Position], optional = [Velocity]
```

## Timing Accuracy

| Counter | Resolution | Overhead |
|---------|------------|----------|
| time.perf_counter() | ~100ns | ~100ns |
| time.time() | ~1us | ~1us |
| time.monotonic() | ~1us | ~100ns |

Using `perf_counter()` for best resolution.
