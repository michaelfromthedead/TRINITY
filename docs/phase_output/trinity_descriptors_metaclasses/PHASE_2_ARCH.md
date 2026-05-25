# PHASE 2 ARCHITECTURE: Metaclass System

## Phase Scope

The metaclass system (`trinity/metaclasses/`) implementing ECS type hierarchy with 8 metaclasses.

## Components

### 2.1 EngineMeta (engine_meta.py - 118 lines)

Base metaclass for all engine types:

- Global type registry for debugging (`_engine_types`)
- `_metaclass_steps` recording for introspection
- Thread-safe registration via `threading.Lock`
- Clean `__repr__` for engine types

All other metaclasses inherit from EngineMeta.

### 2.2 ComponentMeta (component_meta.py - 760 lines)

ECS component type management:

| Feature | Description |
|---------|-------------|
| Unique IDs | `_component_id` generation and registration |
| Field Processing | Type hints including `Annotated` parsing |
| Descriptor Installation | Automatic based on field markers |
| Mutable Default Detection | Rejected at class definition time |
| Pool Management | `return_to_pool`, `pool_stats` |
| Budget Enforcement | `_instance_count`, `max_instances` |
| Layout Optimization | SoA/AoS via `get_layout_arrays` |
| Rust Registration | `_omega.type_register` integration |
| Foundation Registry | Central component tracking |

### 2.3 SystemMeta (system_meta.py - 543 lines)

ECS system organization:

| Feature | Description |
|---------|-------------|
| Phase Organization | `SystemPhase` enum (INIT, UPDATE, RENDER, etc.) |
| Dependency Analysis | `@reads`/`@writes` declarations |
| Parallelization | `_can_parallelize` detection |
| Execution Order | Topological sort via `get_phase_order` |
| Parallel Groups | `get_parallel_groups` for concurrent execution |
| Hot Reload | `hot_reload`, `reload_system` support |
| Resource Conflicts | Automatic detection |

### 2.4 StateMeta (state_meta.py - 490 lines)

Finite state machine types:

| Feature | Description |
|---------|-------------|
| State Registry | Per-machine state tracking |
| Transition Validation | `can_transition`, `validate_transitions` |
| Hierarchical States | `register_substate`, `get_substates` |
| Cycle Detection | Hierarchy cycle prevention |
| History | `record_transition`, `get_previous_state` |
| Hooks | Enter/exit callbacks |

### 2.5 EventMeta (event_meta.py - 439 lines)

Event type management:

| Feature | Description |
|---------|-------------|
| Data-Only Validation | No methods except `__init__`, `__repr__`, etc. |
| Inheritance Tracking | `_event_parent_ids` |
| Channel Filtering | Event routing by channel |
| Event Pooling | `acquire`, `release`, `pool_stats` |
| Serialization | Encode/decode support |

### 2.6 AssetMeta (asset_meta.py - 426 lines)

Asset pipeline types:

| Feature | Description |
|---------|-------------|
| Extension Mapping | Type-to-extension registration |
| Conflict Detection | Duplicate extension warnings |
| Priority Loading | Async queue with priorities |
| Hot Reload | File watching via `watch`, `check_changes` |
| Dependency Order | `get_load_order` |
| Cycle Detection | Circular dependency prevention |

### 2.7 ProtocolMeta (protocol_meta.py - 365 lines)

Network protocol types:

| Feature | Description |
|---------|-------------|
| Version Validation | Compatibility checking |
| Message Registration | Type-to-protocol mapping |
| Version Decoders | Per-version deserialization |
| Negotiation | `negotiate_version` |
| Migration | Path generation between versions |

### 2.8 ResourceMeta (resource_meta.py - 363 lines)

Global singleton types:

| Feature | Description |
|---------|-------------|
| Singleton Pattern | Enforced single instance |
| Dependency Order | `initialize_all` with dependencies |
| Lazy Resources | Deferred initialization |
| Shutdown | Error-handled cleanup |

## Architecture Decisions

### AD-2.1: EngineMeta as Universal Base

All engine metaclasses inherit from EngineMeta.

Rationale: Common infrastructure (registry, step recording, repr, thread safety) is shared without duplication.

### AD-2.2: Step Recording

Every metaclass action appends to `cls._metaclass_steps`:

```python
cls._metaclass_steps.append(Step(Op.REGISTER, {"registry": "component_registry"}))
cls._metaclass_steps.append(Step(Op.TAG, {"key": "component_id", "value": 42}))
```

Rationale: Enables tools (step_trace, lint, coverage) to introspect what happened during class creation.

### AD-2.3: Mutable Default Rejection

ComponentMeta scans field defaults at class creation:

```python
class Bad(Component):
    items: list = []  # REJECTED
```

Rationale: Mutable defaults shared across instances are a Python footgun. Early detection prevents subtle bugs.

### AD-2.4: Declarative Dependencies

SystemMeta uses `@reads`/`@writes` decorators rather than runtime analysis:

```python
@reads(Position, Velocity)
@writes(Position)
class MovementSystem(System):
    ...
```

Rationale: Static analysis enables topological sorting and parallelization detection without running the system.

### AD-2.5: Pool Management at Metaclass Level

ComponentMeta and EventMeta manage object pools:

- Reduces allocation pressure in hot paths
- Transparent to user code
- Configurable pool sizes

### AD-2.6: clear_registry() for Testing

All metaclasses provide `clear_registry()`:

Rationale: Tests need isolation. Registries must be clearable between test cases.

## Metaclass Inheritance Hierarchy

```
type
  |
  v
EngineMeta (base)
  |
  +-- ComponentMeta (ECS data)
  |
  +-- SystemMeta (ECS logic)
  |
  +-- StateMeta (FSM)
  |
  +-- EventMeta (messages)
  |
  +-- AssetMeta (resources)
  |
  +-- ProtocolMeta (network)
  |
  +-- ResourceMeta (singletons)
```

## Class Creation Flow

```
class MyComponent(Component):
    health: int = 100

          |
          v
    [EngineMeta.__new__]
          |
          +-- Register in _engine_types
          |
          v
    [ComponentMeta.__new__]
          |
          +-- Generate _component_id
          +-- Process field type hints
          +-- Detect mutable defaults (reject if found)
          +-- Install descriptors from markers
          +-- Register with Foundation (if available)
          +-- Register with _omega (if available)
          +-- Record steps
          |
          v
    MyComponent class object
```

## Registry Pattern

Each metaclass maintains its own registry:

| Metaclass | Registry | Key |
|-----------|----------|-----|
| EngineMeta | `_engine_types` | class |
| ComponentMeta | `_component_registry` | component_id |
| SystemMeta | `_system_registry` | phase -> [systems] |
| StateMeta | `_state_registry` | machine -> [states] |
| EventMeta | `_event_registry` | event_id |
| AssetMeta | `_asset_registry` | extension -> type |
| ProtocolMeta | `_protocol_registry` | version -> protocol |
| ResourceMeta | `_resource_registry` | resource_id |
