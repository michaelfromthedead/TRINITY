# Investigation: trinity/metaclasses

## Summary
The `trinity/metaclasses/` directory contains 8 fully-implemented metaclasses that form the heart of the Trinity Pattern. These are not stubs -- they are production-quality implementations with thread-safe registries, validation, step recording, pooling, dependency analysis, and real integration points for both Python Foundation and Rust _omega backend.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 38 | Complete | Exports all 8 metaclasses with hierarchy docstring |
| `engine_meta.py` | 119 | Complete | Base metaclass, global registry, `_metaclass_steps` init |
| `component_meta.py` | 761 | Complete | Most complex: field processing, descriptors, Rust registration, pooling, budgets |
| `system_meta.py` | 544 | Complete | Dependency analysis, topological sort, hot-reload, parallel groups |
| `resource_meta.py` | 364 | Complete | Singleton enforcement, priority-ordered init, lazy loading |
| `event_meta.py` | 440 | Complete | Event pooling, serialization/deserialization, inheritance tracking |
| `asset_meta.py` | 427 | Complete | Extension mapping, async load queue, hot-reload watcher |
| `protocol_meta.py` | 366 | Complete | Version negotiation, message registration, migration paths |
| `state_meta.py` | 491 | Complete | Hierarchical states, transition validation, history tracking |

**Total: ~3,550 lines of metaclass implementation**

## The 8 Metaclasses
| Metaclass | Purpose | Lines | Implementation Status |
|-----------|---------|-------|----------------------|
| `EngineMeta` | Base for all engine types, global registry, `_metaclass_steps` | 119 | COMPLETE |
| `ComponentMeta` | ECS components, field descriptors, Rust integration | 761 | COMPLETE |
| `SystemMeta` | ECS systems, dependency graphs, parallel scheduling | 544 | COMPLETE |
| `ResourceMeta` | Global singletons, priority init, lazy loading | 364 | COMPLETE |
| `EventMeta` | Event types, pooling, serialization | 440 | COMPLETE |
| `AssetMeta` | Asset handles, async loading, hot-reload | 427 | COMPLETE |
| `ProtocolMeta` | Network protocols, versioning, message types | 366 | COMPLETE |
| `StateMeta` | State machine states, transitions, hierarchy | 491 | COMPLETE |

## ComponentMeta Deep Dive

### `__new__` Steps (7 phases + 2 optional)
1. **GENERATE UNIQUE ID** - `_component_id`, `_component_name`, TAG steps recorded
2. **PROCESS FIELDS** - `_process_fields()` extracts type hints, handles `Annotated[]`, builds `_field_types`, `_field_offsets`, `_field_defaults`
3. **INSTALL DESCRIPTORS** - `_install_descriptors()` creates descriptor chains (Storage + Validated + Tracked + Networked)
4. **VALIDATE COMPONENT** - `_validate_component()` warns about methods on data-only components
5. **REGISTER** - Add to `_registry` and `_name_to_id` dictionaries
6. **FOUNDATION INTEGRATION** - `_register_with_foundation()` calls `foundation.registry.register()`
6b. **RUST TYPE REGISTRATION** - `_build_rust_layout()` + `_omega.type_register()` call
7. **INITIALIZE POOL AND BUDGET** - If `_pooled_config` or `_budget_config` set, initialize counters

### Rust Integration (Step 6b)
```python
# Lines 121-127 of component_meta.py
fields, total_size = mcs._build_rust_layout(cls)
try:
    from _omega import type_register
    type_register(cls._component_id, cls._component_name, total_size, json.dumps(fields))
except (ImportError, AttributeError):
    pass  # Graceful fallback if Rust backend not available
```

`_build_rust_layout()` maps Python types to Rust type codes:
- `int` -> `("i32", 4)`
- `float` -> `("f32", 4)`
- `bool` -> `("u8", 1)`
- `str` -> `("string", -1)` (variable width)

Returns `(fields_list, total_byte_size)` for SoA memory layout.

### _omega Imports
Three distinct integration points with the `_omega` Rust module:

1. **ComponentMeta (component_meta.py:124-125)**:
   ```python
   from _omega import type_register
   type_register(cls._component_id, cls._component_name, total_size, json.dumps(fields))
   ```

2. **RustStorageDescriptor (rust_storage.py:15-19)**:
   ```python
   from _omega import component_read, component_write, component_delete
   ```

3. **Fallback behavior**: All `_omega` imports are wrapped in try/except. When Rust backend unavailable:
   - ComponentMeta skips `type_register` silently
   - RustStorageDescriptor falls back to `obj.__dict__` storage

### Foundation Registration
```python
# Lines 152-162 of component_meta.py
@classmethod
def _register_with_foundation(mcs, cls: type) -> None:
    try:
        from foundation import registry
        if not registry.is_registered(cls):
            registry.register(cls, name=cls._component_name, track_instances=True)
    except ImportError:
        pass  # Foundation not available
```

Foundation's `registry.py` (200 lines) provides:
- Unified type lookup across all metaclasses
- Optional instance tracking via WeakSet
- Metadata storage per type
- Query methods: `subclasses()`, `types_with_decorator()`, `types_where()`

## Connections

### To foundation/registry
- Every component is registered with `registry.register(cls, name=cls._component_name, track_instances=True)`
- Foundation wraps `__init__` to track instances in WeakSet
- Enables `registry.instances(cls)`, `registry.instance_count(cls)`

### To descriptors
Descriptors installed per-field based on component configuration:
- `StorageDescriptor` / `RustStorageDescriptor` - innermost, actual value storage
- `ValidatedDescriptor` - if `_validation_rules` set on field
- `TrackedDescriptor` - if `_track_changes = True` on component
- `NetworkedDescriptor` - if `_network_config` set
- `SerializableDescriptor` - if `HOOK(on_serialize)` in steps

`DescriptorComposer.compose()` builds the chain with proper nesting.

### To Rust (_omega)
1. **Type registration**: `type_register(component_id, name, size, fields_json)`
2. **Field access**: `component_read(entity_id, component_id, offset, type_code)`
3. **Field mutation**: `component_write(entity_id, component_id, offset, value)`
4. **Field deletion**: `component_delete(entity_id, component_id, offset)`

## Verdict
**REAL IMPLEMENTATION**

This is production-quality code, not stubs. Evidence:
- 3,550+ lines of carefully structured metaclass logic
- Thread-safe registries with proper locking
- Complete Op/Step recording for introspection
- Graceful degradation when Rust backend unavailable
- Full descriptor chain composition system
- Complex features: pooling, budgets, hot-reload, async loading, version negotiation, hierarchical states

## Evidence

### ComponentMeta field processing (lines 214-282):
```python
@classmethod
def _process_fields(mcs, cls: type) -> None:
    try:
        annotations = get_type_hints(cls, include_extras=True)
    except Exception:
        annotations = getattr(cls, "__annotations__", {})

    cls._field_types = {}
    cls._field_offsets = {}
    cls._field_defaults = {}
    cls._field_descriptors = {}
    
    # ... processes Annotated[T, descriptors], extracts base types,
    # computes byte offsets, validates types, installs descriptors
```

### SystemMeta topological sort (lines 280-334):
```python
@classmethod
def get_phase_order(mcs, phase: SystemPhase) -> list[type]:
    # Kahn's algorithm for topological sort
    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    queue.sort(key=lambda sid: mcs._registry[sid]._priority)
    result = []
    while queue:
        current = queue.pop(0)
        result.append(mcs._registry[current])
        # ... update in-degrees, detect cycles
```

### ResourceMeta singleton enforcement (lines 88-113):
```python
def __call__(cls, *args: Any, **kwargs: Any) -> Any:
    with ResourceMeta._initialization_lock:
        if cls._resource_id in ResourceMeta._instances:
            if args or kwargs:
                raise TypeError(f"{cls.__name__} is a singleton...")
            return ResourceMeta._instances[cls._resource_id]
        instance = super().__call__(*args, **kwargs)
        ResourceMeta._instances[cls._resource_id] = instance
        return instance
```

### StateMeta cycle detection (lines 294-336):
```python
@classmethod
def _would_create_cycle(mcs, parent_state: type, child_state: type) -> bool:
    visited = set()
    current = parent_state
    while current is not None:
        if current is child_state:
            return True
        # ... traverse parent chain, check for visited
```
