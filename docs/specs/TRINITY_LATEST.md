# Trinity Pattern: Complete Specification & Reference

A comprehensive specification for the three-layer Python metaprogramming architecture and six-system runtime infrastructure powering the AI Game Engine.

**Implementation Status: COMPLETE** — 989 tests passing (1 xfail known limitation). All 10 specification phases implemented.

---

## Table of Contents

- [Part I: Context & Philosophy](#part-i-context--philosophy)
- [Part II: The Trinity Pattern (Definition-Time)](#part-ii-the-trinity-pattern-definition-time)
  - [Overview](#overview)
  - [The Three Layers](#the-three-layers)
    - [Layer 1: Metaclasses — Class Creation](#layer-1-metaclasses--class-creation)
    - [Layer 2: Descriptors — Attribute Access](#layer-2-descriptors--attribute-access)
    - [Layer 3: Decorators — Behavior Modification](#layer-3-decorators--behavior-modification)
  - [Ops-First Architecture](#ops-first-architecture)
  - [Stacks & Composite Stacks](#stacks--composite-stacks)
  - [Execution Timeline](#execution-timeline)
  - [Unified decompose() — Cross-Layer Introspection](#unified-decompose--cross-layer-introspection)
  - [Composition Rules (Implemented)](#composition-rules-implemented)
- [Part III: Core Foundation (Runtime)](#part-iii-core-foundation-runtime)
- [Part IV: Implementation Guidelines](#part-iv-implementation-guidelines)
- [Part V: Integration & Determinism](#part-v-integration--determinism)
- [Part VI: Annotated Field Syntax](#part-vi-annotated-field-syntax)
- [Part VII: Metaclass Auto-Install of Descriptors](#part-vii-metaclass-auto-install-of-descriptors)
- [Part VIII: Tooling](#part-viii-tooling)
- [Appendices](#appendices)
  - [Appendix A: Complete Metaclass Step Reference](#appendix-a-complete-metaclass-step-reference)
  - [Appendix B: Complete Descriptor Step Reference](#appendix-b-complete-descriptor-step-reference)
  - [Appendix C: Complete Decorator Decomposition Reference](#appendix-c-complete-decorator-decomposition-reference)
  - [Appendix D: Implementation History](#appendix-d-implementation-history)

---

# Part I: Context & Philosophy

## Core Philosophy

### Design Principles

1. **Data-Oriented Design** - Prioritize cache-efficient data layouts (SoA over AoS) and ECS architecture for gameplay systems
2. **Separation of Concerns** - Clear boundaries between simulation, presentation, and tooling layers
3. **Platform Agnosticism** - Abstract OS and graphics APIs (Vulkan, D3D12, Metal) behind clean interfaces
4. **Scalability as First-Class** - Quality tiers, dynamic resolution, and budget management throughout
5. **Determinism as Foundation** - Fixed-point math and command-based mutation enable replay, rollback netcode, and reproducible bugs

## Engine Architecture

```
+------------------------------------------------------------------+
|                         TOOLING & XR                              |
+------------------------------------------------------------------+
|                      GAMEPLAY SYSTEMS                             |
|            (Entity, AI, Input, Camera, Abilities)                 |
+------------------------------------------------------------------+
|     WORLD      |     SIMULATION      |      ANIMATION            |
+------------------------------------------------------------------+
|                         RENDERING                                 |
|       (Frame Graph, GPU-Driven, Lighting, Post-Process)           |
+------------------------------------------------------------------+
|            RESOURCE LAYER           |         AUDIO               |
+------------------------------------------------------------------+
|                       CORE SYSTEMS                                |
|              (Memory, Math, Task System, ECS)                     |
+------------------------------------------------------------------+
|                      PLATFORM LAYER                               |
|           (OS, Graphics API, Audio API, Input API)                |
+------------------------------------------------------------------+
```

Dependencies flow **downward only**.

## The Two Pillars

The engine infrastructure rests on two complementary pillars:

| | Pillar 1: Trinity Pattern | Pillar 2: Core Foundation |
|---|---|---|
| **When** | Definition-time (class creation, import) | Runtime (game loop, interaction) |
| **What** | How classes are built | How objects are observed |
| **Layers** | Decorators, Metaclasses, Descriptors | Mirror, Serializer, Registry, Tracker, EventLog, Inspector, Shell |
| **Count** | ~275 decorators, ~30 descriptor families, 5-10 metaclasses | 6+ systems, ~2500 lines total |

### How They Connect

| Trinity Layer | Foundation System | Connection |
|---|---|---|
| ComponentMeta | Registry | Metaclass registers types |
| TrackedDescriptor | Tracker/EventLog | Descriptor notifies changes |
| TrinityMirror | Mirror | Same system / Trinity wraps |
| @serializable | Serializer | Decorator configures format |
| TrinityDebugger | Inspector + Shell | Interactive tools |

---

# Part II: The Trinity Pattern (Definition-Time)

## Overview

The Trinity Pattern separates game engine infrastructure into three distinct layers, each operating at a different phase of Python's execution model.

**The key insight:** Users write decorators. Decorators orchestrate metaclasses and descriptors behind the scenes.

```
Decorators  ->  configure markers on  ->  Classes built by  ->  Metaclasses
(Layer 3)                                                       (Layer 1)
                                           | which install
                                        Descriptors
                                         (Layer 2)
```

The flow: decorator sets `_track_changes = True` on the class namespace -> `ComponentMeta.__new__` reads that marker -> installs `TrackedDescriptor` on each field.

## The Three Layers

---

### Layer 1: Metaclasses -- Class Creation

Metaclasses control **how classes are created**. They run once, at class definition time.

```python
class ComponentMeta(type):
    """Runs when you write: class Player(Component): ..."""
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        ComponentRegistry.register(cls)
        return cls
```

**Responsibilities:** Validate class structure, register with engine systems, generate unique IDs/type codes, set up class-level metadata, enforce inheritance rules.

**Count:** 8 metaclasses (fixed set -- unlike descriptors/decorators, you rarely add new metaclasses).

#### What Are Metaclasses?

Metaclasses are Python's class factories. They control how classes themselves are created -- they run once at `class Foo:` definition time, not at instantiation time. In the Trinity Pattern, they are **Layer 1** -- the foundational layer that builds the classes that descriptors and decorators operate on.

When you write `@component class Player: health: float`, a metaclass (`ComponentMeta`) intercepts that class creation, assigns it a unique ID, processes its fields, installs descriptors, validates it, and registers it with the engine.

They run once. They must be correct. They are the foundation everything else stands on.

#### Metaclass Hierarchy

All metaclasses inherit from `EngineMeta`, which inherits from Python's `type`. This avoids metaclass conflicts -- Python only allows one metaclass per class, and a single inheritance chain satisfies that.

```
type (Python built-in)
+-- EngineMeta
    +-- ComponentMeta
    +-- SystemMeta
    +-- ResourceMeta
    +-- EventMeta
    +-- AssetMeta
    +-- ProtocolMeta
    +-- StateMeta
```

#### Template for New Metaclasses

```python
class MyMeta(EngineMeta):
    _registry: ClassVar[dict[int, type]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        if name == "MyBaseClass":
            return cls  # Skip the abstract base

        with mcs._lock:
            # 1. Generate unique ID
            cls._my_id = mcs._next_id
            mcs._next_id += 1
            cls._my_name = f"{cls.__module__}.{name}"

            # 2. Set defaults for optional markers
            if not hasattr(cls, "_some_config"):
                cls._some_config = None

            # 3. Validate
            # ... raise TypeError for invalid definitions

            # 4. Register
            mcs._registry[cls._my_id] = cls

        return cls

    @classmethod
    def clear_registry(mcs):
        with mcs._lock:
            mcs._registry.clear()
            mcs._next_id = 1
        super().clear_registry()
```

Key rules:
- Always inherit from `EngineMeta` (never directly from `type`).
- Thread-safe: use `threading.Lock` around all registry mutations.
- Skip the abstract base class by name (e.g., `if name == "Component": return cls`).
- Generate a unique `_*_id` integer for every concrete type.
- Register in a `ClassVar[dict[int, type]]` registry.
- Provide `clear_registry()` for testing.
- Validate at class creation time -- fail loud and early with `TypeError`.

#### Metaclass Purposes

| Metaclass | Decorator | What It Does |
|-----------|-----------|-------------|
| `EngineMeta` | (none, base) | Global debug registry, common `__repr__`, `get_all_types()` |
| `ComponentMeta` | `@component` | Field processing, descriptor installation, component registry, Foundation integration |
| `SystemMeta` | `@system` | Dependency analysis, phase ordering, parallelization grouping, topological sort |
| `ResourceMeta` | `@resource` | Singleton enforcement, dependency-ordered initialization, shutdown lifecycle |
| `EventMeta` | `@event` | Data-only validation, inheritance tracking, channel routing |
| `AssetMeta` | `@asset` | Extension mapping, loader pipeline, hot-reload support |
| `ProtocolMeta` | `@protocol` | Version negotiation, message registration, compatibility checking |
| `StateMeta` | `@state` | Transition validation, enter/exit hooks, machine-scoped registration |

Only `ComponentMeta` installs descriptors -- it is the only metaclass that manages data fields needing per-access behavior.

#### Implemented Metaclasses

All 8 metaclasses are implemented. Located in `trinity/metaclasses/`.

| Metaclass | Module | ID Attr | Registry Methods | Status |
|-----------|--------|---------|-----------------|--------|
| `EngineMeta` | `engine_meta.py` | -- | `get_all_types()`, `get_types_by_metaclass()`, `clear_registry()` | Complete |
| `ComponentMeta` | `component_meta.py` | `_component_id` | `get_by_id()`, `get_by_name()`, `all_components()`, `component_count()`, `clear_registry()` | Complete |
| `SystemMeta` | `system_meta.py` | `_system_id` | `get_by_id()`, `get_by_name()`, `all_systems()`, `get_phase_systems()`, `get_phase_order()`, `get_parallel_groups()`, `clear_registry()` | Complete |
| `ResourceMeta` | `resource_meta.py` | `_resource_id` | `get_by_id()`, `get_by_name()`, `all_resources()`, `get_instance()`, `has_instance()`, `initialize_all()`, `shutdown_all()`, `reset_instance()`, `clear_registry()` | Complete |
| `EventMeta` | `event_meta.py` | `_event_id` | `get_by_id()`, `get_by_name()`, `all_events()`, `is_subtype()`, `get_subtypes()`, `get_by_channel()`, `clear_registry()` | Complete |
| `AssetMeta` | `asset_meta.py` | `_asset_id` | `get_by_id()`, `get_by_name()`, `all_assets()`, `get_for_extension()`, `get_for_path()`, `get_loader()`, `get_supported_extensions()`, `get_hot_reloadable()`, `clear_registry()` | Complete |
| `ProtocolMeta` | `protocol_meta.py` | `_protocol_id` | `get_by_id()`, `get_by_name()`, `all_protocols()`, `is_compatible()`, `negotiate_version()`, `register_message()`, `get_message_type()`, `clear_registry()` | Complete |
| `StateMeta` | `state_meta.py` | `_state_id` | `get_by_id()`, `get_by_name()`, `all_states()`, `get_machine_states()`, `can_transition()`, `validate_transitions()`, `register_with_machine()`, `get_enter_hook()`, `get_exit_hook()`, `clear_registry()` | Complete |

#### Op-Aware Metaclasses (Implemented)

Every metaclass `__new__` now records its operations as Steps via `_metaclass_steps`. This makes the entire metaclass layer visible to `decompose()` and the introspection API. Each metaclass records TAG, DESCRIBE, VALIDATE, REGISTER, and HOOK Steps as appropriate.

**ComponentMeta.__new__ -- 6 recorded step groups:**
1. Generate `_component_id` -- emits `TAG(component_id)`, `TAG(component_name)`
2. `_process_fields()`: extract annotations, build `_field_types`/`_field_offsets`/`_field_defaults` -- emits `DESCRIBE(field, type)` per field
3. `_install_descriptors()`: build descriptor chain via `DescriptorComposer.compose()` -- emits `INTERCEPT(field, descriptor)` per installed descriptor
4. `_validate_component()`: warn about methods on components -- emits `VALIDATE(component_rules)`
5. Register in `_registry` and `_name_to_id` -- emits `REGISTER(component_registry)`
6. `_register_with_foundation()` -- emits `REGISTER(foundation)`
7. Optional pool/budget init -- emits `TAG(pooled)`, `HOOK(on_create, pool_allocate)`, `TAG(budgeted)`, `VALIDATE(budget_limit)` when configured

**SystemMeta.__new__ -- 5 recorded step groups:**
1. Generate `_system_id` -- emits `TAG(system_id)`, `TAG(system_name)`
2. Set defaults -- emits `TAG(system_phase)`, `TAG(reads)`, `TAG(writes)`, `TAG(exclusive)`, `TAG(priority)`
3. `_validate_declarations()` -- emits `VALIDATE(system_declarations)`
4. `_analyze_dependencies()` -- emits `DESCRIBE(dependencies, can_parallelize)`
5. Register -- emits `REGISTER(system_registry)`

**ResourceMeta.__new__ -- 3 recorded step groups:**
1. Generate `_resource_id` -- emits `TAG(resource_id)`, `TAG(resource_name)`, `TAG(resource_priority)`, `TAG(resource_lazy)`
2. Register -- emits `REGISTER(resource_registry)`
3. Singleton enforcement -- emits `HOOK(on_create, singleton_enforce)`

**EventMeta.__new__ -- 6 recorded step groups:**
1. Generate `_event_id` -- emits `TAG(event_id)`, `TAG(event_name)`
2. `_collect_fields()` -- emits `DESCRIBE(field, type)` per field
3. `_collect_parent_ids()` -- emits `TAG(event_parents)`
4. Set defaults -- emits `TAG(event_priority)`, `TAG(event_channels)`, `TAG(event_pooled)`
5. `_validate_event()` -- emits `VALIDATE(event_data_only)`
6. Register -- emits `REGISTER(event_registry)`
7. If pooled -- emits `HOOK(on_create, event_pool_acquire)`, `HOOK(on_destroy, event_pool_release)`

**StateMeta.__new__ -- 4 recorded step groups:**
1. Generate `_state_id` -- emits `TAG(state_id)`, `TAG(state_name)`
2. Set defaults -- emits `TAG(state_transitions)`
3. If enter/exit hooks -- emits `HOOK(on_enter)`, `HOOK(on_exit)`
4. Register globally -- emits `REGISTER(state_global)`, optionally `REGISTER(state_machine:<name>)`

**AssetMeta.__new__ -- 6 recorded step groups:**
1. Generate `_asset_id` and `_asset_type_code` -- emits `TAG(asset_id)`, `TAG(asset_type_code)`
2. Validate extensions -- emits `VALIDATE(asset_extensions_required)`, `TAG(extensions)`
3. Check extension conflicts -- emits `VALIDATE(extension_uniqueness)`
4. Set defaults -- emits `TAG(cache_policy)`, `TAG(hot_reload)`, `TAG(asset_priority)`
5. Register extensions -- emits `REGISTER(asset_extension_map)` per extension
6. Register -- emits `REGISTER(asset_registry)`

**ProtocolMeta.__new__ -- 4 recorded step groups:**
1. Generate `_protocol_id` -- emits `TAG(protocol_id)`, `TAG(protocol_name)`
2. Validate version -- emits `VALIDATE(protocol_version_valid)`, `TAG(protocol_version)`, `TAG(protocol_min_version)`
3. Validate min_version <= version -- emits `VALIDATE(min_version_lte_version)`
4. Register -- emits `REGISTER(protocol_registry)`

**EngineMeta.__new__ -- 1 recorded step:**
1. Register in `_all_engine_types` -- emits `REGISTER(engine_types)`

#### Metaclass Enhancement Status

The following enhancements are documented as open items on the existing implementations:

**ComponentMeta:**
- SoA/AoS layout optimization -- `_packed_layout` marker is read but not acted on. Actual memory layout transformation for cache-friendly iteration is pending.
- Pool integration -- `_pooled_config` marker exists but pool allocation is not wired into `__call__`.
- Budget enforcement -- `_budget_config` marker exists but is not enforced at runtime.

**SystemMeta:**
- Parallel group conflict detection needs resource access conflicts (`_resources` declarations).
- System hot-reload for dev iteration.

**ResourceMeta:**
- Lazy initialization -- `initialize_all()` is eager; lazy resources that instantiate on first access are pending.

**EventMeta:**
- Event pooling -- `_event_pooled` marker is set but pool allocation is not implemented.
- Event serialization for networked dispatch.

**AssetMeta:**
- Async loading pipeline (load queues, priority, callbacks).
- Hot-reload file watcher connection.
- Dependency-ordered loading.

**ProtocolMeta:**
- Protocol evolution / migration path for version bumps.

**StateMeta:**
- Hierarchical states (substates / nested state machines).
- State history for "return to last state" transitions.

**Cross-Cutting:**
- Registry persistence for editor tooling and hot-reload.

---

### Layer 2: Descriptors -- Attribute Access

Descriptors control **how attributes are read and written**. They run on every attribute access.

```python
class TrackedDescriptor:
    """Runs when you access obj.health or obj.health = 50"""
    def __init__(self, name, field_type):
        self.name = name
        self.field_type = field_type
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        old_value = obj.__dict__.get(self.name)
        if old_value != value:
            obj.__dict__[self.name] = value
            obj._dirty_fields.add(self.name)
```

They are the hottest code path in the engine. They must be fast.

**Count:** ~30 descriptor families.

#### How Descriptors Fit

```
Decorators  ->  install  ->  Descriptors  ->  on classes built by  ->  Metaclasses
(Layer 3)                    (Layer 2)                                  (Layer 1)
```

- **Metaclasses** build classes, wire up registries, assign field offsets.
- **Descriptors** control per-field read/write behavior (validation, tracking, networking, caching, observation).
- **Decorators** are the user-facing API. They decompose into Ops, which install descriptors onto fields.

A decorator like `@networked` does not do networking itself -- it installs a `NetworkedDescriptor` (wrapping a `TrackedDescriptor`, wrapping a `StorageDescriptor`) onto each annotated field.

#### Base Class: BaseDescriptor[T]

All descriptors inherit from `BaseDescriptor` in `trinity/descriptors/base.py`. It provides:

- **Lifecycle hooks**: `pre_get`, `post_get`, `pre_set`, `post_set` -- subclasses override these instead of `__get__`/`__set__` directly.
- **Inner delegation**: Every descriptor has an optional `_inner` descriptor it wraps. Reads/writes delegate down the chain.
- **Storage fallback**: The innermost descriptor stores values in `obj.__dict__`.
- **Introspection**: `get_chain()`, `get_metadata()`, `descriptor_id`.
- **Composition rules**: `accepts_inner`, `accepts_outer`, `excludes` -- class-level tuples that the `DescriptorComposer` validates.
- **Op-awareness**: `descriptor_steps` property returning the Steps this descriptor performs (implemented).

#### Protocol: TrinityDescriptor[T]

A `typing.Protocol` that defines the full contract: identity (`name`, `field_type`, `descriptor_id`), composition (`inner`, `accepts_inner`, `accepts_outer`, `excludes`), core methods, lifecycle hooks, introspection, and `descriptor_steps`.

#### Template for New Descriptors

```python
class MyDescriptor(BaseDescriptor[T]):
    __slots__ = ("_my_config",)

    descriptor_id = "my-id"
    accepts_inner = ("storage", "validated", "range")  # what I can wrap
    accepts_outer = ("networked", "observable")          # what can wrap me
    excludes = ()                                        # conflicts

    def __init__(self, field_type=object, inner=None, my_config=None, **config):
        super().__init__(field_type=field_type, inner=inner, **config)
        self._my_config = my_config

    def pre_set(self, obj, value):
        # validate or transform value BEFORE storing
        return value

    def post_set(self, obj, value, old_value):
        # side effects AFTER storing (dirty flags, events, queues)
        pass

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.INTERCEPT, {"get": "my_get", "set": "my_set"})]
```

Key rules:
- Use `__slots__` for all instance attributes (performance).
- Override `pre_set`/`post_set` for write interception, `pre_get`/`post_get` for read interception.
- Never access `obj.__dict__` directly -- delegate to `self._inner` or call `super()`.
- Declare composition rules honestly. The `DescriptorComposer` enforces them.
- Always implement `descriptor_steps` to declare what Ops the descriptor performs.

#### Descriptor Composition

Descriptors chain via wrapping. The `DescriptorComposer.compose()` static method validates and links them:

```python
chain = DescriptorComposer.compose(
    NetworkedDescriptor(field_type=float),
    TrackedDescriptor(field_type=float),
    RangeDescriptor(field_type=float, min_val=0, max_val=100),
    StorageDescriptor(field_type=float, default=100),
)
# Result: Networked -> Tracked -> Range -> Storage
```

Outermost runs first on write, innermost stores the value. On read, innermost retrieves, outermost returns.

`DescriptorComposer.collect_steps(descriptor)` aggregates `descriptor_steps` from the entire chain.

#### Descriptor Purposes

| Purpose | What It Does | Example |
|---------|-------------|---------|
| **Validation** | Reject or clamp bad values on write | `health = 150` -> stored as `100` |
| **Tracking** | Mark fields dirty on change (bit flags or sets) | Only serialize/network changed fields |
| **Networking** | Queue changed fields for replication | Server->client state sync |
| **Caching** | Cache computed values with optional TTL | Derived stats recomputed only when deps change |
| **Observation** | Notify callbacks on change | UI bindings, reactive systems |
| **Storage** | Provide defaults, factory initialization | `StorageDescriptor(default=0)` |
| **Composition** | Validate and chain descriptors safely | `DescriptorComposer.compose(...)` |

#### Implemented Descriptors

Located in `trinity/descriptors/`. Each is a separate module. All descriptors implement `descriptor_steps`.

**Core Descriptors:**

| Descriptor | Module | `descriptor_id` | Purpose | Steps |
|-----------|--------|-----------------|---------|-------|
| `BaseDescriptor` | `base.py` | `"base"` | Abstract base with lifecycle hooks | `[]` |
| `StorageDescriptor` | `storage.py` | `"storage"` | Default values, `__dict__` storage | `[]` (passive) |
| `ValidatedDescriptor` | `validation.py` | `"validated"` | Custom validator functions | `[VALIDATE(custom)]` |
| `RangeDescriptor` | `validation.py` | `"range"` | Numeric min/max clamping | `[VALIDATE(range, min, max, clamp)]` |
| `TrackedDescriptor` | `tracking.py` | `"tracked"` | Dirty flags (bitmask + set), Foundation/EventLog integration | `[TRACK(field)]` + optional `TAG(track_bitmask)` |
| `NetworkedDescriptor` | `networking.py` | `"networked"` | Network queue on change, authority rules | `[TAG(networked), TAG(authority), TAG(interpolated), INTERCEPT(set=network_queue)]` |
| `ObservableDescriptor` | `observable.py` | `"observable"` | Observer callbacks on change | `[HOOK(on_change, observer_dispatch)]` |
| `CachedDescriptor` | `caching.py` | `"cached"` | TTL-based caching over computed | `[INTERCEPT(get=cache_check), TAG(ttl)]` |
| `ComputedDescriptor` | `caching.py` | `"computed"` | Read-only derived values from function | `[INTERCEPT(get=compute, set=deny, delete=deny), TAG(computed), TAG(transient)]` |

**Persistence Descriptors:**

| Descriptor | Module | `descriptor_id` | Purpose | Steps |
|-----------|--------|-----------------|---------|-------|
| `SerializableDescriptor` | `persistence.py` | `"serializable"` | Custom encode/decode for serialization | `[HOOK(on_serialize, encode), HOOK(on_deserialize, decode), TAG(serialization_format)]` |
| `TransientDescriptor` | `persistence.py` | `"transient"` | Mark field as non-serializable | `[TAG(transient=True)]` |
| `MigratedDescriptor` | `persistence.py` | `"migrated"` | Handle field renames across save versions | `[TAG(migrated_from), TAG(version_added)]` |
| `EncryptedDescriptor` | `persistence.py` | `"encrypted"` | Encrypt field value at rest | `[INTERCEPT(get=decrypt, set=encrypt), TAG(encrypted)]` |

**Debug Descriptors (stripped in release):**

| Descriptor | Module | `descriptor_id` | Purpose | Steps |
|-----------|--------|-----------------|---------|-------|
| `ProfiledDescriptor` | `debug.py` | `"profiled"` | Time every get/set, feed to profiler | `[INTERCEPT(get=profile_get, set=profile_set), TAG(profiled)]` |
| `LoggedDescriptor` | `debug.py` | `"logged"` | Log all field accesses | `[INTERCEPT(get=log_get, set=log_set), TAG(logged)]` |
| `WatchedDescriptor` | `debug.py` | `"watched"` | Conditional breakpoint | `[INTERCEPT(set=watch_condition), TAG(watched)]` |

**Async Descriptors:**

| Descriptor | Module | `descriptor_id` | Purpose | Steps |
|-----------|--------|-----------------|---------|-------|
| `LazyDescriptor` | `async_descriptors.py` | `"lazy"` | Defer initialization until first access | `[INTERCEPT(get=lazy_init), TAG(lazy), TAG(init_mode)]` |
| `AsyncLoadDescriptor` | `async_descriptors.py` | `"async_load"` | Load value asynchronously, return fallback | `[INTERCEPT(get=async_load), TAG(async_load)]` |

**Extended Descriptors (built in Phase 7):**

| Descriptor | Module | `descriptor_id` | Purpose | Steps |
|-----------|--------|-----------------|---------|-------|
| `ImmutableDescriptor` | `immutable.py` | `"immutable"` | Deny writes after init | `[INTERCEPT(set=deny_after_init)]` |
| `VersionedDescriptor` | `versioning.py` | `"versioned"` | Per-field version counter + history | `[TRACK(field), TAG(versioned)]` |
| `IndexedDescriptor` | `indexing.py` | `"indexed"` | Value-based index for fast lookup | `[INTERCEPT(set=index_update), REGISTER(index:<field>)]` |
| `AtomicDescriptor` | `atomic.py` | `"atomic"` | Thread-safe get/set with CAS | `[INTERCEPT(get=atomic_get, set=atomic_set)]` |
| `InterpolatedDescriptor` | `interpolation.py` | `"interpolated"` | Smooth between values (linear, slerp, ease) | `[INTERCEPT(get=interpolate, set=set_target), TRACK(field)]` |
| `SparseDescriptor` | `sparse.py` | `"sparse"` | Only store non-default values | `[INTERCEPT(get=sparse_get, set=sparse_set)]` |
| `RateLimitedDescriptor` | `rate_limiting.py` | `"rate_limited"` | Cap write frequency | `[INTERCEPT(set=rate_check), VALIDATE(rate_limit)]` |
| `ConditionalDescriptor` | `conditional.py` | `"conditional"` | Predicate-gated writes | `[INTERCEPT(set=condition_check)]` |
| `TransformDescriptor` | `transform.py` | `"transform"` | Read/write value transforms | `[INTERCEPT(get=read_transform, set=write_transform)]` |
| `ExpiringDescriptor` | `expiring.py` | `"expiring"` | TTL-based value expiration | `[INTERCEPT(get=ttl_check), TAG(ttl)]` |
| `AuditDescriptor` | `audit.py` | `"audit"` | Full access audit trail | `[INTERCEPT(get=audit_read, set=audit_write), TAG(audited)]` |
| `PooledDescriptor` | `pooled_field.py` | `"pooled_field"` | Object pool for field values | `[INTERCEPT(set=pool_return, delete=pool_return), TAG(pooled_field)]` |

**Additional descriptors specified but lower priority:**

| Descriptor | Purpose | Composition Rules |
|-----------|---------|-------------------|
| `PriorityDescriptor` | Higher-priority writes override lower | `accepts_inner: (storage)`, `accepts_outer: (tracked)` |
| `MirrorDescriptor` | Propagate value to mirror target | `accepts_inner: (tracked, storage)`, `accepts_outer: (observable, networked)` |
| `EventSourcedDescriptor` | Append-only event log, derive current value | `accepts_inner: (storage)`, `excludes: (versioned)` |
| `BatchedDescriptor` | Buffer writes, flush on threshold | `accepts_inner: (storage)`, `accepts_outer: (tracked, validated)` |
| `BroadcastDescriptor` | Topic-based observer routing | `accepts_inner: (tracked, observable, storage)`, `accepts_outer: (networked)` |
| `CompressedDescriptor` | Transparent compression codec | `accepts_inner: (storage)`, `accepts_outer: (serializable, networked)` |
| `DiffDescriptor` | Compute and store diffs between values | `accepts_inner: (versioned, storage)`, `accepts_outer: (networked, observable)` |
| `SchemaDescriptor` | JSON-Schema-like validation | `accepts_inner: (storage)`, `accepts_outer: (tracked, observable)` |
| `ProxyDescriptor` | Load from external source on access | `accepts_inner: (storage)`, `accepts_outer: (cached, tracked)` |

Helper functions also implemented: `is_dirty`, `get_dirty_fields`, `clear_dirty`, `get_network_queue`, `pop_network_updates`, `add_observer`, `remove_observer`, `clear_observers`.

---

### Layer 3: Decorators -- Behavior Modification

Decorators are **the user-facing API**. They configure metaclasses and install descriptors.

```python
def track_changes(cls):
    cls._dirty_fields = set()
    for name, annotation in cls.__annotations__.items():
        if not name.startswith('_'):
            setattr(cls, name, TrackedDescriptor(name, annotation))
    cls.get_dirty_fields = lambda self: self._dirty_fields.copy()
    cls.clear_dirty = lambda self: self._dirty_fields.clear()
    return cls
```

**Count:** ~275 decorators organized into tiers.

| Tier | Purpose | Examples |
|------|---------|----------|
| Tier 0 | Compilation | `@native`, `@ffi`, `@target` |
| Tier 1 | ECS Core | `@component`, `@system`, `@resource`, `@event` |
| Tier 2 | Memory | `@pooled`, `@packed`, `@aligned`, `@arena` |
| Tier 3 | Scheduling | `@phase`, `@parallel`, `@exclusive`, `@throttle` |

#### The Seven Primitives

All ~275 decorators decompose into exactly **7 primitive operations**:

| Primitive | Signature | Effects |
|-----------|-----------|---------|
| `REGISTER` | `(class, registry)` | Adds class to global registry, enables lookup by name |
| `DESCRIBE` | `(class)` | Extracts schema from `__annotations__`, makes queryable via `mirror()` |
| `TRACK` | `(field)` | Wraps field in descriptor, notifies tracker on change |
| `VALIDATE` | `(field, constraint)` | Checks constraint on set, raises on violation |
| `INTERCEPT` | `(field, get, set, delete)` | Wraps field access with custom logic |
| `HOOK` | `(event, callback)` | Attaches callback to lifecycle event |
| `TAG` | `(target, key, value)` | Attaches metadata, queryable via `mirror()` |

**Implementation:** `trinity/decorators/primitives.py` (700 lines, 47 decorator compositions, 40 tests). Also `trinity/decorators/ops.py` (Op enum, Step dataclass, run_* functions, make_decorator factory).

**Key Insight:** 265 decorators = 265 different CONFIGURATIONS of 7 primitives.

#### Primitive Definitions

**REGISTER**

```
REGISTER(class, registry) -> class

Effects:
  - Adds class to a named registry
  - Enables lookup by name
  - Enables instance tracking (optional)

Requires: nothing
Provides: registry membership

Implementation: registry.register() called by metaclass __new__
```

**DESCRIBE**

```
DESCRIBE(class) -> class

Effects:
  - Extracts schema from __annotations__
  - Records field names, types, defaults
  - Makes schema queryable via mirror()

Requires: nothing
Provides: schema

Implementation: Mirror schema extraction (Foundation Layer 0)
```

**TRACK**

```
TRACK(field) -> field

Effects:
  - Wraps field in TrackedDescriptor
  - Notifies tracker on change
  - Records old/new value
  - Sets dirty flag (bitmask + set)

Requires: nothing
Provides: change notification

Implementation: TrackedDescriptor + tracker.mark_dirty() (Foundation Layer 2)
```

**VALIDATE**

```
VALIDATE(field, constraint) -> field

Effects:
  - Checks constraint on set
  - Raises or clamps on violation
  - Constraint types: Range, Type, Choices, Custom

Requires: nothing
Provides: constraint enforcement

Implementation: ValidatedDescriptor, RangeDescriptor (trinity/descriptors/)
```

**INTERCEPT**

```
INTERCEPT(field, get=fn, set=fn, delete=fn) -> field

Effects:
  - Wraps field access with custom logic
  - Can transform, deny, redirect on get/set/delete

Requires: nothing
Provides: access control

Implementation: BaseDescriptor pre/post hooks (trinity/descriptors/base.py)
```

**HOOK**

```
HOOK(event, callback) -> attachment

Events (class-level):
  on_create, on_destroy, on_serialize, on_deserialize,
  on_attach, on_detach, on_enter, on_exit, on_tick,
  on_damage, on_collision, on_compile
  ... 30+ event types

Events (field-level):
  on_change, on_access

Requires: depends on event (HOOK(on_change) requires TRACK)
Provides: lifecycle integration

Implementation: Lifecycle callbacks in metaclass (trinity/metaclasses/)
```

**TAG**

```
TAG(target, category, key=value, ...) -> target

Effects:
  - Attaches metadata
  - Queryable via mirror()
  - No behavioral effect

Categories: network, memory, scheduling, persistence, graphics,
            physics, ai, audio, compilation, destruction, ...
            50+ categories

Requires: nothing
Provides: metadata

Implementation: FieldInfo.metadata in Mirror (Foundation Layer 0)
```

#### Composition Rules

**Dependencies:**

```
RULE                                          WHY
HOOK(on_change) requires TRACK                Can't hook what isn't tracked
VALIDATE benefits from DESCRIBE               Schema enables richer validation
REGISTER should be last                       Finalize after all transforms
TAG(network) requires TAG(serialization)      Networked data must be serializable
```

**Conflicts:**

```
CONFLICT                                      RESOLUTION
INTERCEPT(set=deny) + TRACK                   INTERCEPT wins, no tracking
INTERCEPT(set=deny) + VALIDATE                INTERCEPT wins, no validation
Multiple INTERCEPT on same field              Chain in order (outer wraps inner)
Multiple VALIDATE on same field               All must pass
Multiple HOOK on same event                   All fire in order
Multiple TAG with same key                    Last wins
```

**Ordering (canonical):**

```
INNERMOST -> OUTERMOST (application order)

1. TAG           (metadata first, affects nothing)
2. VALIDATE      (constraints before tracking)
3. TRACK         (tracking before hooks)
4. INTERCEPT     (interception wraps everything)
5. HOOK          (hooks after field setup)
6. DESCRIBE      (schema after fields configured)
7. REGISTER      (registration last)
```

**Validation Messages:**

```
ERROR: HOOK(on_change) requires TRACK
FIX:   Add @tracked to field 'health'

ERROR: INTERCEPT(set=deny) conflicts with TRACK
FIX:   Remove @tracked or @readonly (can't track readonly field)

ERROR: INTERCEPT(set=deny) conflicts with VALIDATE
FIX:   Remove @validated or @readonly (can't validate readonly field)

ERROR: REGISTER should be last
FIX:   Move @component to top of decorator stack

ERROR: TAG(network) requires TAG(serialization)
FIX:   Add @serializable before @networked

WARNING: @networked without @tracked fields
FIX:     Add @tracked to fields that should sync

WARNING: VALIDATE without DESCRIBE
FIX:     Add @component for full validation support
```

All rules are implemented in `ops.py` RULES list and enforced by `validate_steps()` and `validate_ordering()`.

#### How Primitives Multiply Into 265 Decorators

The 7 primitives generate the full decorator space through parameterization:

**REGISTER** x 20+ registries = 20+ decorator families:
- ComponentRegistry (`@component`), SystemRegistry (`@system`), ResourceRegistry (`@resource`), EventRegistry (`@event`), AssetRegistry (`@asset`), NetworkRegistry (`@networked`), AIRegistry (`@behavior_tree`, `@utility_ai`), AudioRegistry (`@sound`), UIRegistry (`@widget`), DestructionSystem (`@destructible`), etc.

**HOOK** x 30+ events = 30+ hook decorators:
- on_create (`@on_spawn`, `@pooled`), on_destroy (`@on_despawn`), on_change (`@on_change`, `@observable`), on_serialize (`@serializable`, `@networked`), on_tick (`@behavior_tree`, `@system`), on_damage (`@destructible`), on_collision (`@physics`), on_enter/on_exit (`@state_machine`), etc.

**TAG** x 50+ categories = 50+ tag decorators:
- network (`@networked`, `@rpc`, `@interest`), memory (`@pooled`, `@packed`, `@aligned`, `@budget`), scheduling (`@phase`, `@parallel`, `@throttle`), persistence (`@serializable`, `@versioned`, `@transient`), graphics (`@lod`, `@shadow_caster`), physics (`@physics_material`, `@buoyancy`), ai (`@behavior_tree`, `@perception`), audio (`@sound`, `@spatial_audio`), etc.

**VALIDATE** x 10+ constraint types:
- Range, Type, Choices, Pattern, Custom, etc.

Total possibility space: 7 x (20 + 30 + 50 + 10) = **770 possible leaf decorators**. 265 implemented = ~35% coverage.

#### Composite Definitions

Standard decorator compositions. Each defined by its primitive decomposition.

**@component**

```
@component = DESCRIBE + REGISTER(ComponentRegistry)

Metaclass: ComponentMeta
Primitives: [DESCRIBE, REGISTER]

Usage:
  @component
  class Health:
      current: float = 100
      max: float = 100
```

**@system**

```
@system(phase) = REGISTER(SystemRegistry) + TAG(scheduling={phase}) + DESCRIBE

Metaclass: SystemMeta
Primitives: [REGISTER, TAG, DESCRIBE]
```

**@tracked**

```
@tracked = TRACK

Primitives: [TRACK]

Usage:
  @component
  class Health:
      current: float = tracked(100)
```

**@validated**

```
@validated(constraint) = VALIDATE(constraint) + TAG(constraint=constraint)

Primitives: [VALIDATE, TAG]

Usage:
  current: Annotated[float, Range(0, 100)] = 100
```

**@networked**

```
@networked(authority) = TAG(network={authority, relevance, delta})
                      + HOOK(on_serialize -> network_serialize)
                      + HOOK(on_deserialize -> network_deserialize)
                      + REGISTER(NetworkSystem)
                      + requires TRACK on fields

Primitives: [TAG, HOOK, HOOK, REGISTER]
Requires: [TRACK]
```

**@serialized**

```
@serialized = DESCRIBE + HOOK(on_serialize) + HOOK(on_deserialize)

Primitives: [DESCRIBE, HOOK, HOOK]
```

**@transient**

```
@transient = TAG(transient=True)

Primitives: [TAG]
```

**@readonly**

```
@readonly = INTERCEPT(set=deny)

Primitives: [INTERCEPT]
```

**@computed**

```
@computed(fn) = INTERCEPT(get=fn) + TAG(transient=True, computed=True)

Primitives: [INTERCEPT, TAG]
```

**@pooled**

```
@pooled(initial_size) = TAG(pool={initial_size, max_size})
                       + HOOK(on_create -> pool_allocate)
                       + HOOK(on_destroy -> pool_return)
                       + REGISTER(PoolManager)

Primitives: [TAG, HOOK, HOOK, REGISTER]
```

**@behavior_tree**

```
@behavior_tree(id) = TAG(ai={tree_id})
                    + REGISTER(AIRegistry)
                    + HOOK(on_tick -> tree_evaluate)

Primitives: [TAG, REGISTER, HOOK]
```

**@destructible**

```
@destructible(health, fracture_depth) = TAG(destruction={health, fracture_depth})
                                       + HOOK(on_damage -> apply_damage)
                                       + HOOK(on_destroy -> fracture)
                                       + REGISTER(DestructionSystem)

Primitives: [TAG, HOOK, HOOK, REGISTER]
```

**@native**

```
@native(backend, nogil) = TAG(compile={backend, nogil})
                         + HOOK(on_compile -> cython_transform)

Primitives: [TAG, HOOK]
```

**@serializable**

```
@serializable(format) = DESCRIBE
                       + TAG(serialization={format})
                       + HOOK(on_serialize -> serialize_fn)
                       + HOOK(on_deserialize -> deserialize_fn)

Primitives: [DESCRIBE, TAG, HOOK, HOOK]
```

**@parallel**

```
@parallel(chunk_size) = TAG(execution={parallel, chunk_size})

Primitives: [TAG]
```

**The Universal Pattern:**

Every decorator follows this shape:

```
EVERY DECORATOR IS:
  1. Zero or more TAGs (metadata)
  2. Zero or more HOOKs (lifecycle callbacks)
  3. Zero or one REGISTER (to some registry)
  4. Zero or one DESCRIBE (schema extraction)
  5. Zero or more TRACK/VALIDATE/INTERCEPT (field wrappers)
```

265 decorators = 265 different configurations of this pattern.

#### Decorator Tier Map

```
TIER 0-9:    ~55 decorators    FOUNDATION
TIER 10-21:  ~35 decorators    ENGINE SYSTEMS
TIER 22-41:  ~85 decorators    EXTENDED CORE
TIER 42-52:  ~90 decorators    DOMAIN-SPECIFIC

Most are COMPOSITES or DOMAIN SUGAR over the 7 primitives.
```

**Proof By Decomposition (Spot Checks):**

| Tier | Decorator | Decomposition |
|------|-----------|---------------|
| 0 | `@native(backend, nogil)` | TAG(compile) + HOOK(on_compile) |
| 1 | `@system(phase)` | REGISTER(SystemRegistry) + TAG(scheduling) + DESCRIBE |
| 3 | `@parallel(chunk_size)` | TAG(execution) |
| 4 | `@serializable(format)` | DESCRIBE + TAG(serialization) + HOOK(serialize) + HOOK(deserialize) |
| 11 | `@track_changes` | TRACK on all fields |
| 31 | `@validated(rules)` | VALIDATE(rules) |
| 36 | `@behavior_tree(id)` | TAG(ai) + REGISTER(AIRegistry) + HOOK(on_tick) |
| 43 | `@destructible(health, fracture)` | TAG(destruction) + HOOK(on_damage) + HOOK(on_destroy) + REGISTER(DestructionSystem) |

#### Field Declaration Syntax

Three equivalent forms. All compile to the same primitives.

**Form 1: Annotated (implemented -- see Part VI)**

```python
@component
class Health:
    current: Annotated[float, Tracked, Range(0, 100)] = 100
    max: Annotated[float, Tracked] = 100
```

**Form 2: Function**

```python
@component
class Health:
    current: float = tracked(validated(Range(0, 100), 100))
    max: float = tracked(100)
```

**Form 3: Decorator**

```python
@component
class Health:
    @tracked
    @validated(Range(0, 100))
    current: float = 100

    @tracked
    max: float = 100
```

#### Formal Grammar

```
decoration  := class_dec | field_dec

class_dec   := REGISTER(registry)
             | DESCRIBE
             | HOOK(class_event, callback)
             | TAG(key=value, ...)
             | class_dec class_dec        # composition

field_dec   := TRACK
             | VALIDATE(constraint)
             | INTERCEPT(get=fn, set=fn)
             | HOOK(field_event, callback)
             | TAG(key=value, ...)
             | field_dec field_dec        # composition

class_event := on_create | on_destroy | on_serialize | on_deserialize
             | on_attach | on_detach | on_enter | on_exit
             | on_tick | on_damage | on_collision | on_compile

field_event := on_change | on_access

constraint  := Range(min, max)
             | Type(T)
             | Choices([...])
             | Pattern(regex)
             | Custom(fn)
             | constraint & constraint    # all must pass
             | constraint | constraint    # any must pass

composite   := @component       = DESCRIBE + REGISTER(ComponentRegistry)
             | @system(phase)    = REGISTER(SystemRegistry) + TAG(scheduling) + DESCRIBE
             | @tracked          = TRACK
             | @validated(c)     = VALIDATE(c) + TAG(constraint=c)
             | @networked(a)     = TAG(network) + HOOK(serialize) + HOOK(deserialize) + REGISTER(NetworkSystem)
             | @serialized       = DESCRIBE + HOOK(on_serialize) + HOOK(on_deserialize)
             | @transient        = TAG(transient=True)
             | @readonly         = INTERCEPT(set=deny)
             | @computed(fn)     = INTERCEPT(get=fn) + TAG(transient=True)
             | @pooled(size)     = TAG(pool) + HOOK(on_create) + HOOK(on_destroy) + REGISTER(PoolManager)
             | @behavior_tree(id)= TAG(ai) + REGISTER(AIRegistry) + HOOK(on_tick)
             | @destructible(h,f)= TAG(destruction) + HOOK(on_damage) + HOOK(on_destroy) + REGISTER(DestructionSystem)
             | @native(b,n)      = TAG(compile) + HOOK(on_compile)
             | @serializable(f)  = DESCRIBE + TAG(serialization) + HOOK(serialize) + HOOK(deserialize)
             | @parallel(c)      = TAG(execution)
```

#### Introspection API (Implemented)

File: `trinity/decorators/introspection.py`

```python
# What primitives are on this class?
primitives(Health)
# -> [DESCRIBE, REGISTER(ComponentRegistry)]

# What primitives are on this field?
primitives(Health, 'current')
# -> [TAG(range=(0,100)), VALIDATE(Range(0,100)), TRACK]

# What composites were applied?
composites(Health)
# -> [component]

composites(Health, 'current')
# -> [validated(Range(0,100)), tracked]

# Full descriptor chain on a field
chain(Health, 'current')
# -> TAG -> VALIDATE -> TRACK

# Expand a composite to its primitives
expand(component)
# -> DESCRIBE + REGISTER(ComponentRegistry)

expand(networked)
# -> TAG(network) + HOOK(on_serialize) + HOOK(on_deserialize) + REGISTER(NetworkSystem)

# Find decorators using a given primitive
find_decorators(primitive=HOOK, event="on_change")
# -> [@on_change, @observable, @track_changes, @reactive, ...]

# Compose primitives directly
@compose(
    register(ComponentRegistry),
    describe,
    tag("network", authority="server"),
    hook("on_serialize", my_serialize)
)
class CustomNetworked: ...

# Validate a combination
validate_combination([TRACK, HOOK(on_change)])
# -> {valid: True}

validate_combination([HOOK(on_change)])
# -> {valid: False, error: "HOOK(on_change) requires TRACK"}

# List all rules
all_rules()
# -> [
#     Rule("HOOK(on_change) requires TRACK"),
#     Rule("REGISTER should be last"),
#     Rule("INTERCEPT(set=deny) conflicts with TRACK"),
#     Rule("INTERCEPT(set=deny) conflicts with VALIDATE"),
#     Rule("TAG(network) requires TAG(serialization)"),
#   ]
```

#### Primitive-Level Validation Rules (Implemented)

```python
PRIMITIVE_RULES = [
    # HOOK constraints
    Rule("HOOK(on_change) requires TRACK",
         when=has(HOOK, event="on_change"),
         requires=has(TRACK)),

    # INTERCEPT constraints
    Rule("INTERCEPT(set=deny) conflicts with TRACK",
         when=has(INTERCEPT, set="deny"),
         conflicts=has(TRACK)),

    Rule("INTERCEPT(set=deny) conflicts with VALIDATE",
         when=has(INTERCEPT, set="deny"),
         conflicts=has(VALIDATE)),

    # REGISTER constraints
    Rule("REGISTER should be applied last",
         when=has(REGISTER),
         order="last"),

    # TAG constraints
    Rule("TAG(network) requires TAG(serialization)",
         when=has(TAG, category="network"),
         requires=has(TAG, category="serialization")),
]

# These rules apply to ALL decorators automatically
# because decorators are composed of primitives.
```

#### Canonical Patterns

One right way for each use case.

**Basic Component:**

```python
@component
class Position:
    x: float = 0
    y: float = 0
    z: float = 0
```

**Tracked Component:**

```python
@component
class Health:
    current: Annotated[float, Tracked] = 100
    max: Annotated[float, Tracked] = 100
```

**Validated Component:**

```python
@component
class Health:
    current: Annotated[float, Tracked, Range(0, 100)] = 100
    max: Annotated[float, Tracked, Range(1, 1000)] = 100
```

**Networked Component:**

```python
@component
@networked(authority="server")
class Health:
    current: Annotated[float, Tracked, Range(0, 100)] = 100
    max: Annotated[float, Tracked] = 100
```

**Component with Transient:**

```python
@component
class Renderer:
    mesh_id: Annotated[int, Tracked] = 0
    _cached_mesh: Annotated[Mesh, Transient] = None
```

**Component with Computed:**

```python
@component
class Health:
    current: Annotated[float, Tracked] = 100
    max: Annotated[float, Tracked] = 100

    @computed
    def percent(self) -> float:
        return self.current / self.max
```

**Full Featured Component:**

```python
@component
@networked(authority="server")
class Player:
    # Identity (readonly after creation)
    id: Annotated[int, Readonly] = 0
    name: Annotated[str, Tracked] = "Unknown"

    # Stats (validated, tracked, networked)
    health: Annotated[float, Tracked, Range(0, 100)] = 100
    mana: Annotated[float, Tracked, Range(0, 100)] = 100

    # Position (tracked, networked, interpolated)
    position: Annotated[Vec3, Tracked, Interpolated] = Vec3(0, 0, 0)

    # Runtime only (not serialized)
    _controller: Annotated[Controller, Transient] = None

    # Derived (computed on access)
    @computed
    def is_alive(self) -> bool:
        return self.health > 0
```

#### Decorator Implementation Status

All 7 primitives are implemented both implicitly (in metaclasses and descriptors) and explicitly (as the Op enum, Step dataclass, and run_* functions in `ops.py`).

| Primitive | Implementation | Where | Status |
|-----------|---------------|-------|--------|
| REGISTER | `registry.register()` in metaclass `__new__` + `run_register()` | Foundation Layer 1 + ops.py | Complete |
| DESCRIBE | Mirror `__annotations__` parsing + `run_describe()` | Foundation Layer 0 + ops.py | Complete |
| TRACK | `TrackedDescriptor` + `tracker.mark_dirty()` + `run_track()` | Foundation Layer 2 + ops.py | Complete |
| VALIDATE | `ValidatedDescriptor`, `RangeDescriptor` + `run_validate()` | trinity/descriptors/ + ops.py | Complete |
| INTERCEPT | `BaseDescriptor` pre/post hooks + `run_intercept()` | trinity/descriptors/base.py + ops.py | Complete |
| HOOK | Lifecycle callbacks in metaclass + `run_hook()` | trinity/metaclasses/ + ops.py | Complete |
| TAG | `FieldInfo.metadata` in Mirror + `run_tag()` | Foundation Layer 0 + ops.py | Complete |

**Decorator Modules (60+ files):**

Located in `trinity/decorators/`:

```
accessibility.py    cinematics.py      error_handling.py  narrative.py         rpc.py
achievements.py     compilation.py     game_ai.py         network_extended.py  save_system.py
ai_generation.py    composition.py     gameplay.py        ops.py               scheduling.py
analytics.py        crafting.py        gpu.py             particles_vfx.py     security.py
animation.py        data_flow.py       ik_procedural.py   physics_sim.py       social.py
assets.py           debug_cheat.py     input.py           platform_specifics.py spatial.py
audio_extended.py   debug_extended.py  lifecycle.py       prefabs.py           stacks.py
audio.py            debug_safety.py    localization.py    procedural.py        state_machine.py
base.py             destruction.py     lod_streaming.py   registry.py          time.py
bridges_caching.py  dev.py             memory.py          rendering.py         transactions.py
build_deploy.py     economy.py         modding.py         replay.py            ui.py
builtin_stacks/     ecs_core.py                                                world_building.py
```

Plus `builtin_stacks/` with meta.py, platform.py, development.py.

---

## Ops-First Architecture

*Design principles learned the hard way.*

### The Problem

We built decorators first. Each one was self-contained -- it manually set attributes, added methods, tracked itself. Then we tried to add primitives on top as metadata annotations. The result: two parallel systems doing the same work. Nobody read from the primitive layer.

### The Insight

The build order was wrong. **Ops come first.** Everything else is composition.

A decorator is not a self-contained block of code that manually sets `cls._pooled = True`. A decorator is **a named list of steps**. Each step is one of 7 operations. The operations do the real work. The decorator is just configuration.

### The 7 Ops

| Op | What it does |
|----|-------------|
| TAG | Attach queryable metadata to target |
| HOOK | Wire a lifecycle callback |
| REGISTER | Add target to a named registry |
| DESCRIBE | Extract schema from annotations |
| TRACK | Enable change monitoring / dirty flags |
| VALIDATE | Enforce a constraint |
| INTERCEPT | Wrap get/set/delete on fields |

### A Decorator is a List of Steps

```python
# This IS @pooled. Nothing else.
pooled_steps = [
    Step(Op.TAG, {"key": "pool", "value": {...}}),
    Step(Op.HOOK, {"event": "on_create", "callback": _pool_acquire}),
    Step(Op.HOOK, {"event": "on_destroy", "callback": _pool_release}),
    Step(Op.REGISTER, {"registry": "PoolManager"}),
]
```

No manual `cls._pooled = True`. No manual `_track_decorator()`. The op functions set the canonical attributes. Downstream code reads from `cls._tags["pool"]`, `cls._hooks["on_create"]`, `cls._registries`.

### Why This Matters

1. **Lock-step.** Change how TAG works? Every decorator that uses TAG changes automatically. One system, not 80 independent implementations.
2. **AI composability.** An AI can read a decorator's step list and know exactly what it does. It can compose new decorators by combining steps.
3. **Introspection for free.** `decompose(pooled)` returns the step list. No separate metadata layer needed -- the steps ARE the metadata AND the implementation.
4. **Stacks just work.** A Stack is a list of decorators. A decorator is a list of steps. So a Stack is a flat list of steps. Composition is concatenation.
5. **No duplication.** There is no "primitive layer" and "decorator layer" doing the same thing. One layer: steps executed by op functions.

### The Layers (corrected)

```
Layer 0: Ops (7)              -- The real implementation. Do the work.
Layer 1: Decorators (~80)     -- Named step lists. Configuration only.
Layer 2: Stacks (~25)         -- Named decorator lists. Composition only.
Layer 3: Composite Stacks     -- Named stack lists. Project archetypes.
```

### What a Decorator File Looks Like (target state)

```python
pooled = make_decorator("pooled",
    tier=Tier.MEMORY,
    requires=("component",),
    params={"initial_size": 1024, "grow_factor": 2.0, "max_size": None},
    steps=lambda p: [
        Step(Op.TAG, {"key": "pool", "value": {
            "initial_size": p["initial_size"],
            "grow_factor": p["grow_factor"],
            "max_size": p["max_size"],
        }}),
        Step(Op.HOOK, {"event": "on_create", "callback": _pool_acquire}),
        Step(Op.HOOK, {"event": "on_destroy", "callback": _pool_release}),
        Step(Op.REGISTER, {"registry": "PoolManager"}),
    ],
    doc="Pre-allocate and reuse memory for component instances.",
)
```

Every decorator in every tier follows this exact pattern. Lock-step.

---

## Stacks & Composite Stacks

Stacks are **composable decorator groups** that act as a single decorator.

### Stack Infrastructure

```python
class Stack:
    def __init__(self, *decorators, name: str = None):
        self._decorators = decorators
        self._name = name or f"Stack({len(decorators)} decorators)"
    
    def __call__(self, cls):
        for decorator in reversed(self._decorators):
            cls = decorator(cls)
        return cls
    
    def __add__(self, other: 'Stack') -> 'Stack':
        return Stack(*self._decorators, *other._decorators,
                     name=f"{self._name} + {other._name}")
    
    def expand(self) -> list[str]:
        return [d.__name__ if hasattr(d, '__name__') else str(d)
                for d in self._decorators]

def parameterized_stack(fn):
    """Decorator for creating parameterized stacks."""
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Stack:
        result = fn(*args, **kwargs)
        if not isinstance(result, Stack):
            raise TypeError(f"{fn.__name__} must return a Stack")
        return result
    wrapper._is_parameterized_stack = True
    return wrapper
```

### Built-in Stacks

| Stack | Description |
|-------|-------------|
| `production_component(pool_size, layout, category)` | Production-ready component with memory optimization |
| `safe_system(phase, read, write)` | System with explicit read/write declarations |
| `saveable_data(version, format, migrations)` | Data with save/load and version migration |
| `networked_entity(authority, relevance, priority, pool_size)` | Entity with network replication |
| `bandwidth_efficient(radius, max_updates_per_second, priority)` | Bandwidth-optimized networking |
| `predicted_entity(history_frames, max_reconcile_frames, snap_threshold)` | Client-predicted entity with server reconciliation |
| `secure_multiplayer(rate_limit)` | Security-hardened multiplayer |
| `versioned_saveable(version, migrations)` | Saveable data with version migration |
| `replay_ready(history_frames, keyframe_interval)` | Component ready for replay recording |
| `deterministic_data()` | Data for deterministic simulation |
| `streaming_chunk(chunk_size, overlap, min_age)` | Streamable world chunk |
| `lod_scalable(levels, distances)` | LOD-scalable content |
| `complete_ai(behavior_tree_id, sense, range, fov, states)` | Complete AI entity with BT and perception |
| `profiled_dev(name, warn_ms)` | Development instrumentation (stripped in release) |

### Composite Stacks

| Composite Stack | Description |
|----------------|-------------|
| `multiplayer_character(pool_size, history_frames, version)` | Predicted, saveable, secure player character |
| `competitive_entity(pool_size, history_frames)` | Deterministic, replayable competitive entity |
| `open_world_entity(pool_size, chunk_size)` | Streamed, LOD-scaled, saveable open world entity |
| `mmo_entity(pool_size, relevance_radius)` | Bandwidth efficient, secure, streamed MMO entity |
| `moddable_content(namespace, version)` | JSON serialized, versioned, extensible mod content |

### Migration: Before and After

```python
# BEFORE: 12 decorators
@component
@packed(layout="soa")
@pooled(initial_size=64)
@budget(category="players")
@networked(authority="server", predicted=True, interpolated="hermite")
@snapshot(history_frames=30)
@server_reconcile(max_reconcile_frames=10, snap_threshold=0.5)
@diff(strategy="shallow")
@serializable(format="binary")
@versioned(version=2, migrations={1: migrate_v1})
@server_authoritative
@validated(rules=[position_bounds])
class PlayerCharacter:
    position: Vec3
    velocity: Vec3
    health: float

# AFTER: 1 stack
@multiplayer_character(pool_size=64, history_frames=30, version=2)
class PlayerCharacter:
    position: Vec3
    velocity: Vec3
    health: float
```

### Stack Selection Guide

| You Want | Use This Stack |
|----------|----------------|
| Basic component | `@production_component()` |
| Networked entity | `@networked_entity()` |
| Player character | `@multiplayer_character()` |
| Competitive game entity | `@competitive_entity()` |
| Open world entity | `@open_world_entity()` |
| MMO entity | `@mmo_entity()` |
| AI enemy | `@complete_ai("behavior_id")` |
| Moddable content | `@moddable_content("namespace")` |
| Dev instrumentation | `@profiled_dev("Category/Name")` |

Use individual decorators when no built-in stack matches, you need fine-grained control, or you are building a custom stack.

---

## Execution Timeline

| Layer | When | Frequency | Cost Tolerance |
|-------|------|-----------|----------------|
| Metaclass | Import/class definition | Once per class | High (startup only) |
| Decorator | Import/class definition | Once per decorated item | High (startup only) |
| Descriptor | Every attribute access | Potentially every frame | **Must be fast** |

### How Decorators Orchestrate Everything

```
@component                   -> Sets __metaclass__ = ComponentMeta
@networked                   -> Adds _network_config to class
@track_changes               -> Installs TrackedDescriptor on fields
class Player:                -> Installs NetworkedDescriptor (wrapping Tracked)
    health: float            -> ComponentMeta.__new__ runs: validates, registers, generates ID
```

### Descriptor Chain Example

`@networked + @track_changes + @range(0, 100)` on `health: float` creates:

1. NetworkedDescriptor (outermost)
2. TrackedDescriptor (middle)
3. RangeDescriptor (innermost)

Execution for `player.health = 150`:
1. NetworkedDescriptor.__set__ called with value=150
2. TrackedDescriptor.__set__ called with value=150
3. RangeDescriptor.__set__ clamps to 100, stores in obj.__dict__
4. TrackedDescriptor marks 'health' as dirty
5. NetworkedDescriptor queues network update

### Data Flow

**Write path** `obj.health = 150`: Outermost descriptor delegates inward, innermost validates/stores, then side effects fire outward (dirty flags, network queue, event log).

**Read path** `x = obj.health`: Outer to inner delegation, innermost reads from `__dict__`, provenance tracking if in computation context.

---

## Unified decompose() -- Cross-Layer Introspection

**Status: Implemented (Phase 4)**

The `decompose()` function in `ops.py` provides cross-layer introspection, collecting Steps from all three Trinity layers.

### API

```python
def decompose(target, include_metaclass=True, include_descriptors=True) -> list[Step]:
    """
    Returns Steps from all three layers for a class, or from _steps for a decorator.
    
    For classes, collects:
      - _applied_steps (decorator Steps already run)
      - _metaclass_steps (if include_metaclass and attr exists)
      - Per-field descriptor_steps (if include_descriptors)
    
    For decorators (has _steps attr), returns _steps (backward compatible).
    """
```

### decompose_layered()

```python
def decompose_layered(cls: type) -> dict[str, list[Step]]:
    """Return Steps grouped by layer."""
    return {
        "decorators": getattr(cls, "_applied_steps", []),
        "metaclass": getattr(cls, "_metaclass_steps", []),
        "descriptors": _collect_descriptor_steps(cls),
    }
```

### expand()

For classes, `expand()` calls `decompose_layered()` and formats as:

```
[Decorators]  TAG(component=True) + TAG(component_name=Health) + REGISTER(ecs_core)
[Metaclass]   DESCRIBE(field=current, type=float) + INTERCEPT(field=current, descriptor=tracked) + REGISTER(component_registry, id=1)
[Descriptors] TRACK(field=current) + VALIDATE(constraint=range, min=0, max=100)
```

For decorators, returns flat expansion as before.

---

## Composition Rules (Implemented)

**Status: Implemented (Phase 5)**

All composition rules from LANG_DEC.md are now enforced in code via `validate_steps()` and `validate_ordering()` in `ops.py`.

### Implemented Rules

| Rule | Description |
|------|-------------|
| HOOK(on_change) requires TRACK | Cannot hook what is not tracked |
| INTERCEPT(set=deny) conflicts with TRACK | Cannot track if cannot write |
| INTERCEPT(set=deny) conflicts with VALIDATE | Cannot validate if cannot write |
| REGISTER should be last | Finalize after all transforms |
| TAG(network) requires TAG(serialization) | Networked data must be serializable |

### validate_ordering()

Checks that Steps follow the canonical order: TAG -> VALIDATE -> TRACK -> INTERCEPT -> HOOK -> DESCRIBE -> REGISTER.

```python
_OP_ORDER = {
    Op.TAG: 0, Op.VALIDATE: 1, Op.TRACK: 2, Op.INTERCEPT: 3,
    Op.HOOK: 4, Op.DESCRIBE: 5, Op.REGISTER: 6
}

def validate_ordering(steps: list[Step]) -> dict[str, Any]:
    """Check if Steps follow canonical ordering."""
    ...
```

---

# Part III: Core Foundation (Runtime)

While the Trinity Pattern defines **how classes are built**, Core Foundation provides **how objects are observed and manipulated** at runtime.

## The Six Systems

```
+------------------------------------------------------------------------------+
|                       CORE FOUNDATION                                        |
+------------------------------------------------------------------------------+
|  LAYER 3: INTERACTIVE    Inspector, Shell                                    |
|  LAYER 2: REACTIVE       Tracker, EventLog                                  |
|  LAYER 1: STRUCTURAL     Registry                                           |
|  LAYER 0: ESSENTIAL      Mirror, Serializer                                 |
+------------------------------------------------------------------------------+
```

## Layer 0: Essential

### System 1: Mirror

**Purpose:** Answer questions about any object at runtime. **Implementation:** `foundation/mirror.py` (174 lines, 44 tests)

```python
m = mirror(obj)           # ObjectMirror
m.type_name               # -> str
m.fields                  # -> dict[str, FieldInfo]
m.methods                 # -> dict[str, MethodInfo]
m.get(name)               # -> Any
m.set(name, value)        # -> None
m.to_dict()               # -> dict
m.describe()              # -> str (human-readable)
m.get_path(path)          # -> Any (dotted path access)
m.set_path(path, value)   # -> None (dotted path mutation)

schema_hash(Player)       # -> "a3f2c1e8" (hash of canonical schema)
```

### System 2: Serializer

**Purpose:** Convert any object to portable format and reconstruct it. **Implementation:** `foundation/serializer.py` (232 lines, 69 tests)

```python
to_dict(obj, include_schema_hash=True)  # -> dict
from_dict(data)                          # -> object (with auto-migration)
to_file(obj, path) / from_file(path)
deep_copy(obj)
diff(obj_a, obj_b) / patch(obj, delta)
```

### System 2b: Migration Registry

**Implementation:** `foundation/migrations.py` (160 lines, 26 tests)

```python
migrations.register(from_hash="a3f2c1e8", to_hash="b4e3d2f9",
    migrate=lambda old: {**old, "mana": 50})
# Migrations chain automatically via BFS: a->b + b->c = a->c
```

## Layer 1: Structural

### System 3: Registry

**Purpose:** Track all known types. **Implementation:** `foundation/registry.py` (189 lines, 36 tests)

```python
registry.get("Player")                         # -> type | None
registry.all_types()                            # -> list[type]
registry.subclasses(Entity)                     # -> list[type]
registry.types_with_decorator("networked")      # -> list[type]
registry.instances(Player)                      # -> iterator[object]
```

## Layer 2: Reactive

### System 4: Tracker

**Purpose:** Know when objects change. **Implementation:** `foundation/tracker.py` (195 lines, 45 tests)

```python
tracker.is_dirty(obj) / tracker.dirty_fields(obj)
tracker.mark_dirty(obj, field, old_value, new_value) / tracker.mark_clean(obj)
tracker.subscribe(obj, callback) / tracker.on_change(callback)
tracker.begin_transaction("Move Player") / tracker.commit_transaction() / tracker.rollback_transaction()
tracker.undo() / tracker.redo()
```

### System 4b: EventLog

**Purpose:** Unified log answering "what happened?" and "why?" with entity-centric causal chains. **Implementation:** `foundation/eventlog.py` (330 lines, 34 tests)

```python
@dataclass
class Event:
    tick: int
    operation: str                    # "Player.take_damage"
    operation_args: dict[str, Any]
    entity: int | None
    changes: list[Change]
    immediate_parent: str | None      # Causal chain
    root_cause: str | None            # First entity-bound operation
    depth: int
```

**Key Insight: Systems are pass-through, entities are actors.**

```
GameLoop.update()           -- NOT root cause (system)
  AISystem.process()        -- NOT root cause (system)
    Monster_G.think()       -- ROOT CAUSE (first entity-bound operation)
      Monster_G.command_attack(Monster_B)
        Monster_B.attack(player)
          Player.take_damage()
            Change(health, 100, 70)
```

Use `@traced` on methods to enable automatic causal chain tracking.

```python
event_log.events_where(entity=player.id)                          # What happened to player?
event_log.changes_where(entity=player.id, field="health", new_value=0)  # Who killed me?
event_log.events_where(root_cause_entity=monster_g.id, tick=5000)       # What chaos did Monster_G cause?
```

## Layer 3: Interactive

### System 5: Inspector

**Purpose:** Visualize any object. Edit values live. Navigate object graphs. **Implementation:** `foundation/inspector.py` (281 lines, 55 tests)

Built-in views: Fields, Raw, JSON, Collection, History, Causality, Provenance.

### System 6: Shell

**Purpose:** Execute code live. Enable AI collaboration. **Implementation:** `foundation/shell.py` (150 lines, 63 tests)

Default namespace includes: `mirror`, `registry`, `tracker`, `serializer`, `inspector`, `shell`. Convenience: `inspect(obj)`, `save(obj, path)`, `load(path)`, `undo()`, `redo()`. `_` = last result.

## ShellLang: The Shell Language

**Implementation:** `foundation/shelllang/` (1200 lines total, 76 tests)

Three layers: Human Sugar, AI Interface, Core.

**Human Sugar:**
```python
Enemy.near(player, 10).where(lambda e: e.health.current < 50)
e.health.current = 100        # -> Health.current: 45 -> 100
Enemy.all.set(health__current=0)
mark("before_fight") / rewind("before_fight")
```

**AI Structured Interface:**
```python
ai.execute({'op': 'query', 'components': ['Enemy', 'Health'],
    'where': {'health.current': {'<': 50}}, 'near': {'entity': 'player', 'distance': 10}})
ai.dry_run({'op': 'set', 'entity': 7, 'component': 'Health', 'field': 'current', 'value': 100})
```

## VIPER Extensions

### Path Utilities (`foundation/paths.py`, 47 tests)

`parse_path("inventory.items[0].damage")`, `get_path(obj, path)`, `set_path(obj, path, value)`

### First-Class Queries (`foundation/query.py`, 71 tests)

```python
q = Query(Enemy, Health).near(player, 10).where(health__lt=50)
q.hash()            # stable, content-based
q.subscribe(on_add=..., on_remove=...)
q1 & q2 / q1 | q2 / q1 - q2    # query algebra
```

### ContentStore (`foundation/content_store.py`, 64 tests)

Content-addressable storage. Identical content = identical hash. Tree storage with structural sharing. Efficient diff: O(differences), not O(size).

### Delta Sync (`foundation/delta_sync.py`, 42 tests)

Compute minimal delta between states, apply delta to target.

### Computed Provenance (`foundation/provenance.py`, 46 tests)

Answer "why is this computed value X?" with full derivation trees. Use `@computed @track_provenance` and query with `provenance(obj, field)` or `derivation_tree(obj, field)`.

### QueryCacheMirror (`foundation/query_cache_mirror.py`, 23 tests)

Introspect query cache: registered queries, hit/miss rates, cached counts.

### Capability-Based Security (`foundation/capabilities.py` + `foundation/secure_shell.py`, 58 tests)

```python
class Capability(Flag):
    READ = auto(); WRITE = auto(); CREATE = auto(); DELETE = auto()
    EXECUTE = auto(); SPAWN = auto(); NETWORK = auto(); FILESYSTEM = auto()
    FULL = READ | WRITE | CREATE | DELETE | EXECUTE | SPAWN | NETWORK | FILESYSTEM

# Frozen, composable capability sets
readonly_shell = create_readonly_shell()
sandbox_shell = create_sandbox_shell()
```

## Session Object (The "Image")

```python
class Session:
    world: World; settings: Settings; editor_layout: EditorLayout
    selection: list[EntityRef]; camera_position: Vec3
    open_inspectors: list[InspectorState]; shell_history: list[str]
    undo_stack: list[Transaction]; redo_stack: list[Transaction]

serializer.to_file(session, "project.session")
```

---

# Part IV: Implementation Guidelines

## Composition Rules

- **Metaclass:** One metaclass per class hierarchy. Use a unified `EngineMeta` base.
- **Descriptor:** Descriptors wrap in decorator application order (last applied = outermost).
- **Decorator:** Order matters. Specific before general (`@server_authoritative` before `@networked` before `@component`).

## Design Principles

1. **User sees only decorators** -- metaclasses and descriptors are hidden.
2. **Fail fast at definition time** -- metaclass `__new__` validates structure.
3. **Descriptors must be fast** -- use `__slots__`, direct `__dict__` access.
4. **Metadata flows down** -- decorator sets class attributes, metaclass reads config and installs descriptors, descriptors implement behavior.
5. **Composition over inheritance** -- compose via decorators, avoid deep class hierarchies.

## Common Patterns

- **Decorator that requires metaclass:** If class is not already using the right metaclass, wrap it.
- **Descriptor that wraps another:** Store `self.inner`, delegate `__get__`/`__set__`.
- **Metaclass that auto-installs descriptors:** Check class attributes set by decorators, install matching descriptors.

## Anti-Patterns

- **Logic in descriptor `__get__`:** Expensive computation on every access. Cache results instead.
- **Metaclass for single-use behavior:** Use a decorator instead.
- **Deep descriptor chains:** 7+ layers = 7 function calls per access. Combine related behaviors into one descriptor.

---

# Part V: Integration & Determinism

## Trinity + Foundation Connection

`@component` triggers `ComponentMeta.__new__()` which calls `registry.register(cls)` and installs `TrackedDescriptor`, which on `__set__` calls `tracker.mark_dirty(...)` and `event_log.record(Change)`.

**Implementation:** `foundation/bridge.py` (230 lines, 19 integration tests)

## Deterministic Simulation Architecture

Determinism is a **first-class architectural principle**. Given identical inputs, the simulation produces bit-identical outputs across all platforms.

**What we get:** Replay for free, lockstep multiplayer, rollback netcode, time-travel debugging, reproducible bugs, parallel simulation, desync detection.

**What it costs:** Fixed-point math, explicit ordering, command-based mutation, controlled randomness, discipline at the simulation boundary.

### The Simulation Boundary

The boundary cuts horizontally through the engine. It is not a new layer -- it is a property.

| System | Simulation | Presentation | Notes |
|--------|:----------:|:------------:|-------|
| Entity State | X | | Health, position, inventory |
| Gameplay Logic | X | | Abilities, effects, rules |
| AI Decisions | X | | Behavior tree outputs |
| Physics (Gameplay) | X | | Collision, movement |
| State Machines | X | | Unified, feeds both sides |
| Input Processing | X | | Buffered, ordered |
| Animation State | X | | State machine, sync points |
| Animation Playback | | X | Bone transforms, blending |
| Rendering | | X | Meshes, materials, lights |
| Audio | | X | Playback, mixing |
| Particles/VFX | | X | Visual effects |
| Camera | | X | Smoothing, shake |
| UI | | X | Display, input feedback |

**Rules:** Presentation cannot write to simulation. Simulation cannot read from presentation. All simulation mutation goes through Command Queue. Snapshots are immutable once produced.

### Fixed-Point Mathematics

No bare floats in simulation. The type system enforces this.

| Type | Description |
|------|-------------|
| `Fixed16` (Q8.8) | 16-bit, range +/-127.996, precision 0.00390625 |
| `Fixed32` (Q16.16) | 32-bit, range +/-32767.9999, precision 0.0000152588 |
| `FVec2/3/4` | Fixed-point vectors with add, sub, mul, div, dot, cross, length_squared, normalize |
| `FQuat` | Fixed-point quaternion, renormalize periodically, slerp via approx |
| `FAngle` | Trig via lookup tables or CORDIC |
| `FTransform` | position (FVec3) + rotation (FQuat) + scale (FVec3) |
| `Tick` | u64 frame number (not time) |
| `EntityID` | u64 with deterministic assignment |
| `SimRNG` | u64/u128 state, xorshift or PCG |

**Avoiding Transcendentals:**

| Function | Deterministic Alternative |
|----------|--------------------------|
| sin/cos | Lookup table (e.g., 4096 entries) |
| sqrt | Newton-Raphson iteration (fixed count) |
| atan2 | CORDIC algorithm |
| normalize | Iterative or lookup-based inverse sqrt |
| length | Return length_squared when possible |

### The Simulation Kernel

The simulation world is conceptually a **pure function**: `(State, Inputs) -> State`

**System Execution Order:**

1. **INPUT PROCESSING:** Receive input buffer, validate inputs, convert to commands
2. **COMMAND EXECUTION:** Sort command queue by (tick, entity_id, command_type, sequence), execute spawn, then mutation, then despawn commands
3. **SIMULATION SYSTEMS (fixed order):**
   1. InputSystem, 2. AISystem, 3. AbilitySystem, 4. EffectSystem, 5. MovementSystem, 6. PhysicsSystem, 7. TransformSystem, 8. StateMachineSystem, 9. AnimationStateSystem, 10. TriggerSystem, 11. DamageSystem, 12. DeathSystem, 13. CleanupSystem
4. **SNAPSHOT:** Generate snapshot, compute checksum, push to snapshot buffer, advance tick counter

### Command Architecture

All simulation mutations go through commands. No direct writes.

Commands are sorted by `(tick, entity_id, command_type, sequence)`. This guarantees: all commands for tick N execute before tick N+1, commands for same entity execute in predictable order, same command type groups together, sequence breaks ties from same source.

### Snapshot System

**Accordion snapshot strategy:**
- **Dense region (recent):** Snapshot every tick for last N ticks (~10-30). Enables frame-perfect rollback.
- **Sparse region (older):** Snapshot every f(distance) ticks, f(n) = n^1.5 or similar growth. Old snapshots pruned as new ones arrive.
- **Keyframe anchors:** Full snapshot every K ticks (~300-600, i.e. 5-10 seconds at 60Hz). Never pruned.

Activity-based density: more snapshots during entity spawn/despawn, significant state changes, player input, combat, checkpoint crossings.

### Hierarchical Checksums

**Finding divergence (O(log N)):**
1. Compare world checksums (XOR of chunk checksums)
2. Find bad chunk
3. Find bad archetype
4. Find bad entity
5. Diff entity's components to find exact field

XOR is commutative (order does not matter), fast (single CPU instruction), and incremental (XOR out old, XOR in new).

### Deterministic RNG

Requirements: Same seed = same sequence always. Same sequence across all platforms. State is part of simulation (snapshot/restore).

**Recommended:** PCG (Permuted Congruential Generator). Per-entity RNG: fork using entity_id as additional seed.

**Anti-patterns:** `random.random()` (system RNG), `time.time()` as seed (wall clock varies), `hash(object)` as seed (object ID may vary).

### Entity Identity

EntityID: u64 = Index (48 bits) + Generation (16 bits, reuse counter).

IDs are assigned by the Command Queue at tick boundary, not by the spawn call. Same commands, same order, same tick = same IDs. Always. Archetype maintains sorted order by EntityID for deterministic iteration.

### Deterministic Physics

Most physics engines (PhysX, Bullet) are NOT deterministic. We build our own for gameplay physics. Presentation can use PhysX for ragdolls, debris, cloth.

**Requirements:** All math in fixed-point. Fixed iteration counts. Deterministic contact ordering. Fixed substep count per tick.

**Collision shapes:** FCircle, FSphere, FAABB, FCapsule, FConvex. Broad phase: spatial hash grid, deterministic iteration. Narrow phase: GJK/EPA with capped iterations. Solver: sequential impulse with fixed iteration count.

### Unified State Machines

State machines live in simulation but output to both sides. Simulation gets: damage windows, abilities, movement speed. Presentation gets: animation state, audio triggers, VFX. Transitions evaluated in definition order. First match wins. Sync points defined by tick, not animation time.

### Tick Rate Architecture

| Game Type | Tick Rate | Physics Substeps |
|-----------|-----------|------------------|
| Fighting | 60 | 2 |
| RTS | 15-30 | 1 |
| FPS | 60-128 | 4 |
| Turn-based | 1-10 | 1 |

Tick = simulation step (fixed rate, deterministic). Frame = render frame (variable rate). Accumulator pattern handles mismatch.

### Trinity Integration for Determinism

**Metaclass layer:** `SimulationMeta` validates all fields are determinism-safe types, registers with checksum system, assigns deterministic type ID, rejects forbidden API access. `PresentationMeta` has no type restrictions, cannot be referenced by SimulationMeta classes, can read sim via snapshot.

**Descriptor layer:** `SimulationDescriptor` uses fixed-point storage only, mutation via command queue, contributes to checksum, tracks dirty for replication. `PresentationDescriptor` allows any storage, direct mutation, no checksum, reads from snapshots.

**Decorator layer:** `@simulation` forces SimulationMeta, validates types. `@presentation` forces PresentationMeta. `@command` makes methods return Commands. `@deterministic_rng` injects RNG through world.rng.

**New tier:** TIER 0.5: DETERMINISM (`@simulation`, `@presentation`) between TIER 0 (Foundation) and TIER 1 (Identity).

### Network Determinism

Because simulation is deterministic, networking synchronizes INPUTS, not STATE.

**Model 1: Lockstep** -- All clients wait for all inputs. Simple, guaranteed sync. Latency = slowest player. Use: RTS, turn-based, co-op.

**Model 2: Rollback (GGPO-style)** -- Simulate ahead, rollback on misprediction. Responsive, low perceived latency. CPU cost of resimulation. Use: Fighting games, action games.

**Model 3: Server Authoritative + Prediction** -- Server runs "real" simulation, clients predict. Cheat-resistant, scalable. Correction artifacts. Use: MMO, competitive FPS.

**Desync recovery:** Authority sends full snapshot, desynced client restores, replays inputs. Hierarchical checksums diagnose which entity diverged.

### Replay System

Replay file = header (version, sim_config, seed, initial_snapshot, metadata) + inputs per tick + checksums.

**File size:** Input-based ~2-5 MB for 10 minutes (compressed). State-based ~7 GB for same duration. Deterministic replay wins by orders of magnitude.

Playback modes: Normal, fast-forward, rewind (restore snapshot + replay), seek (nearest snapshot + replay), slow-motion.

### Debugging Determinism

**The golden test:** Run same replay twice. Compare checksums every tick. Any difference = determinism bug.

| Symptom | Likely Cause |
|---------|-------------|
| Random divergence | Unseeded RNG |
| Diverge after N ticks | Float accumulation |
| Diverge on specific entity | Iteration order |
| Diverge on collision | Physics non-determinism |
| Diverge cross-platform | Transcendental functions |
| Diverge after load | Uninitialized memory |

Time-travel debugging: `debugger.step_back()`, `debugger.goto_tick(N)`, `debugger.watch(entity, field, condition)`.

### Mod Sandboxing

Mods interact only through: registered decorators, descriptors, commands, and exposed APIs.

**Mods CAN access:** `entity.get(Component)`, `world.query(...)`, `world.spawn/despawn` (via commands), `world.rng.next_*()`, `world.current_tick`, `Fixed`, `FVec3`, etc.

**Mods CANNOT access:** `time.time()`, `random.random()`, `float` in sim, `dict/set` iteration (must sort), any IO.

**Enforcement:** Metaclass validation (reject float fields), descriptor interception (validate type on write), API surface restriction (modules not importable), command validation (reject invalid data).

### Deterministic Frame Structure

```python
while running:
    delta_time = get_delta_time()
    accumulator += delta_time

    # SIMULATION PHASE (deterministic, may run 0-N times)
    while accumulator >= tick_duration:
        gather_inputs()
        simulation_tick()       # sort commands, execute, run systems, snapshot, checksum
        network_sync()          # if multiplayer: send inputs, compare checksums
        accumulator -= tick_duration

    # PRESENTATION PHASE (non-deterministic, once per frame)
    alpha = accumulator / tick_duration
    interpolate(prev_snapshot, current_snapshot, alpha)
    update_animation()
    render()
    update_audio()
    update_ui()
    present()
```

### Determinism Anti-Patterns

| Anti-Pattern | Why It Breaks | Fix |
|--------------|---------------|-----|
| `float` in sim | Platform differences | Use `Fixed` |
| `time.time()` | Wall clock varies | Use tick counter |
| `random.random()` | System RNG | Use `world.rng` |
| `dict` iteration | Order varies | Sort by key |
| `while error > thresh` | Variable iterations | Fixed count |
| `math.sin/cos/sqrt` | Platform differences | Lookup tables |
| Reading presentation from sim | Non-deterministic | Use snapshot |
| Direct mutation | No replay/network | Use commands |
| Async spawn order | Order varies | Collect, sort, spawn |

---

# Part VI: Annotated Field Syntax

**Status: Implemented (Phase 8)**

`ComponentMeta._process_fields()` supports the `Annotated` type hint syntax for declaring field descriptors inline.

### How It Works

1. During `_process_fields()`, if a field's type annotation has `get_origin(field_type) is Annotated`, the metaclass extracts the base type and metadata via `get_args(field_type)`.
2. Each metadata item is checked: if it is a descriptor class (e.g., `Tracked`), it is instantiated with `field_type=base_type`. If it is a descriptor instance (e.g., `Range(0, 100)`), it is used directly.
3. The descriptor chain is built from metadata items using `DescriptorComposer.compose()`.
4. The composed descriptor is installed on the field, and the unwrapped `base_type` is stored in `_field_types[field_name]`.

### Syntax

```python
from typing import Annotated

@component
class Health:
    # Tracked + validated in one annotation
    current: Annotated[float, Tracked, Range(0, 100)] = 100
    max: Annotated[float, Tracked] = 100
    
    # Transient field
    _cache: Annotated[dict, Transient] = None
```

This is equivalent to manually composing and installing descriptors, but more concise and declarative.

---

# Part VII: Metaclass Auto-Install of Descriptors

**Status: Implemented (Phase 9)**

`ComponentMeta._install_descriptors()` now reads both decorator-set markers AND Op Steps from `_applied_steps` to decide what descriptors to install.

### Behavior

1. After building the descriptor chain from decorator markers, the metaclass checks `_applied_steps` on the class.
2. If `Step(Op.TRACK)` is present but no `TrackedDescriptor` in chain: auto-adds `TrackedDescriptor`.
3. If `Step(Op.VALIDATE)` is present but no `ValidatedDescriptor` in chain: auto-adds `ValidatedDescriptor` with constraint from Step args.
4. If `Step(Op.INTERCEPT)` is present but no corresponding descriptor: logs a warning.
5. If `Step(Op.HOOK, event=on_serialize)` is present: auto-adds `SerializableDescriptor` if not present.
6. Each auto-installed descriptor is recorded in `_metaclass_steps`.

This ensures that the Op-based decorator system and the marker-based legacy system produce identical results.

---

# Part VIII: Tooling

**Status: Implemented (Phase 10)**

Located in `trinity/tools/`. Each tool is a standalone function importable from `trinity.tools`.

### trinity/tools/doctor.py

Iterates all registered classes from all metaclass registries, calls `validate_steps()` on `decompose(cls)`, and reports errors. Use to validate the entire codebase at import time or as a CI check.

### trinity/tools/step_trace.py

`trace(cls)` returns a formatted string showing all Steps grouped by layer with source file:line information. Useful for debugging decorator/metaclass/descriptor interactions.

### trinity/tools/op_coverage.py

For each Op, counts how many classes use it, finds classes with zero Steps, and flags gaps. Useful for ensuring all registered classes are properly wired into the Op system.

### trinity/tools/lint.py

Import-time validator that hooks into `EngineMeta.__new__()` to call `validate_steps()` on every class created. Catches composition rule violations at definition time rather than runtime.

---

# Appendices

## Appendix A: Complete Metaclass Step Reference

Every metaclass `__new__` records its operations as Steps in `cls._metaclass_steps`. Here is the complete reference for all 8 metaclasses.

### EngineMeta

| # | Step | Details |
|---|------|---------|
| 1 | `REGISTER(engine_types)` | Registers in `_all_engine_types` with qualified name |

### ComponentMeta

| # | Step | Details |
|---|------|---------|
| 1 | `TAG(component_id=<id>)` | After generating `_component_id` |
| 2 | `TAG(component_name=<name>)` | Qualified module.name |
| 3 | `DESCRIBE(field=<name>, type=<type>)` | Per field, from `_process_fields()` |
| 4 | `INTERCEPT(field=<name>, descriptor=<id>)` | Per installed descriptor, from `_install_descriptors()` |
| 5 | `VALIDATE(component_rules)` | After `_validate_component()` |
| 6 | `REGISTER(component_registry, id=<id>)` | After registry insertion |
| 7 | `REGISTER(foundation, name=<name>)` | After `_register_with_foundation()` |
| 8 | `TAG(pooled=True)` | If pool configured |
| 9 | `HOOK(on_create, pool_allocate)` | If pool configured |
| 10 | `TAG(budgeted=True)` | If budget configured |
| 11 | `VALIDATE(budget_limit)` | If budget configured |

### SystemMeta

| # | Step | Details |
|---|------|---------|
| 1 | `TAG(system_id=<id>)` | After generating `_system_id` |
| 2 | `TAG(system_name=<name>)` | Qualified name |
| 3 | `TAG(system_phase=<phase>)` | Execution phase |
| 4 | `TAG(reads=<tuple>)` | Read component declarations |
| 5 | `TAG(writes=<tuple>)` | Write component declarations |
| 6 | `TAG(exclusive=<bool>)` | Exclusive execution flag |
| 7 | `TAG(priority=<int>)` | Execution priority |
| 8 | `VALIDATE(system_declarations)` | After `_validate_declarations()` |
| 9 | `DESCRIBE(dependencies=<list>, can_parallelize=<bool>)` | After `_analyze_dependencies()` |
| 10 | `REGISTER(system_registry, id=<id>, phase=<phase>)` | After registry insertion |

### ResourceMeta

| # | Step | Details |
|---|------|---------|
| 1 | `TAG(resource_id=<id>)` | After generating `_resource_id` |
| 2 | `TAG(resource_name=<name>)` | Qualified name |
| 3 | `TAG(resource_priority=<int>)` | Init priority |
| 4 | `TAG(resource_lazy=<bool>)` | Lazy init flag |
| 5 | `REGISTER(resource_registry, id=<id>)` | After registry insertion |
| 6 | `HOOK(on_create, singleton_enforce)` | Singleton `__call__` enforcement |

### EventMeta

| # | Step | Details |
|---|------|---------|
| 1 | `TAG(event_id=<id>)` | After generating `_event_id` |
| 2 | `TAG(event_name=<name>)` | Qualified name |
| 3 | `DESCRIBE(field=<name>, type=<type>)` | Per field, from `_collect_fields()` |
| 4 | `TAG(event_parents=<ids>)` | Inheritance chain |
| 5 | `TAG(event_priority=<int>)` | Dispatch priority |
| 6 | `TAG(event_channels=<list>)` | Channel routing |
| 7 | `TAG(event_pooled=<bool>)` | Pool flag |
| 8 | `VALIDATE(event_data_only)` | After `_validate_event()` |
| 9 | `REGISTER(event_registry, id=<id>)` | After registry insertion |
| 10 | `HOOK(on_create, event_pool_acquire)` | If pooled |
| 11 | `HOOK(on_destroy, event_pool_release)` | If pooled |

### StateMeta

| # | Step | Details |
|---|------|---------|
| 1 | `TAG(state_id=<id>)` | After generating `_state_id` |
| 2 | `TAG(state_name=<name>)` | Qualified name |
| 3 | `TAG(state_transitions=<list>)` | Allowed transitions |
| 4 | `HOOK(on_enter, <callback>)` | If `_state_on_enter` set |
| 5 | `HOOK(on_exit, <callback>)` | If `_state_on_exit` set |
| 6 | `REGISTER(state_global, id=<id>)` | Global registration |
| 7 | `REGISTER(state_machine:<name>, state=<name>)` | If registered with machine |

### AssetMeta

| # | Step | Details |
|---|------|---------|
| 1 | `TAG(asset_id=<id>)` | After generating `_asset_id` |
| 2 | `TAG(asset_type_code=<code>)` | Type code |
| 3 | `VALIDATE(asset_extensions_required)` | Extensions must exist |
| 4 | `TAG(extensions=<list>)` | Supported extensions |
| 5 | `VALIDATE(extension_uniqueness)` | No conflicts |
| 6 | `TAG(cache_policy=<policy>)` | Cache policy |
| 7 | `TAG(hot_reload=<bool>)` | Hot reload flag |
| 8 | `TAG(asset_priority=<int>)` | Load priority |
| 9 | `REGISTER(asset_extension_map, extension=<ext>)` | Per extension |
| 10 | `REGISTER(asset_registry, id=<id>)` | After registry insertion |

### ProtocolMeta

| # | Step | Details |
|---|------|---------|
| 1 | `TAG(protocol_id=<id>)` | After generating `_protocol_id` |
| 2 | `TAG(protocol_name=<name>)` | Qualified name |
| 3 | `VALIDATE(protocol_version_valid)` | Version is positive int |
| 4 | `TAG(protocol_version=<int>)` | Current version |
| 5 | `TAG(protocol_min_version=<int>)` | Minimum supported version |
| 6 | `VALIDATE(min_version_lte_version)` | min_version <= version |
| 7 | `REGISTER(protocol_registry, id=<id>)` | After registry insertion |

---

## Appendix B: Complete Descriptor Step Reference

Every descriptor implements `descriptor_steps` returning the Ops it performs. Here is the complete reference.

| Descriptor | Steps |
|-----------|-------|
| `BaseDescriptor` | `[]` |
| `StorageDescriptor` | `[]` (passive storage) |
| `TrackedDescriptor` | `[TRACK(field=<name>)]` + optional `[TAG(track_bitmask=True)]` |
| `ValidatedDescriptor` | `[VALIDATE(constraint=custom, validator_count=<n>)]` |
| `RangeDescriptor` | `[VALIDATE(constraint=range, min=<min>, max=<max>, clamp=<bool>)]` |
| `ObservableDescriptor` | `[HOOK(on_change, observer_dispatch)]` |
| `NetworkedDescriptor` | `[TAG(networked=True), TAG(authority=<auth>), TAG(interpolated=<mode>), INTERCEPT(set=network_queue)]` |
| `SerializableDescriptor` | `[HOOK(on_serialize, encode), HOOK(on_deserialize, decode), TAG(serialization_format=<fmt>)]` |
| `TransientDescriptor` | `[TAG(transient=True)]` |
| `MigratedDescriptor` | `[TAG(migrated_from=<name>), TAG(version_added=<ver>)]` |
| `EncryptedDescriptor` | `[INTERCEPT(get=decrypt, set=encrypt), TAG(encrypted=True)]` |
| `CachedDescriptor` | `[INTERCEPT(get=cache_check), TAG(ttl=<seconds>)]` |
| `ComputedDescriptor` | `[INTERCEPT(get=compute, set=deny, delete=deny), TAG(computed=True), TAG(transient=True)]` |
| `ProfiledDescriptor` | `[INTERCEPT(get=profile_get, set=profile_set), TAG(profiled=True)]` |
| `LoggedDescriptor` | `[INTERCEPT(get=log_get, set=log_set), TAG(logged=True)]` |
| `WatchedDescriptor` | `[INTERCEPT(set=watch_condition), TAG(watched=True)]` |
| `LazyDescriptor` | `[INTERCEPT(get=lazy_init), TAG(lazy=True), TAG(init_mode=<mode>)]` |
| `AsyncLoadDescriptor` | `[INTERCEPT(get=async_load), TAG(async_load=True)]` |
| `ImmutableDescriptor` | `[INTERCEPT(set=deny_after_init)]` |
| `VersionedDescriptor` | `[TRACK(field=<name>), TAG(versioned=True)]` |
| `IndexedDescriptor` | `[INTERCEPT(set=index_update), REGISTER(index:<field>)]` |
| `AtomicDescriptor` | `[INTERCEPT(get=atomic_get, set=atomic_set)]` |
| `InterpolatedDescriptor` | `[INTERCEPT(get=interpolate, set=set_target), TRACK(field=<name>)]` |
| `SparseDescriptor` | `[INTERCEPT(get=sparse_get, set=sparse_set)]` |
| `RateLimitedDescriptor` | `[INTERCEPT(set=rate_check), VALIDATE(rate_limit)]` |
| `ConditionalDescriptor` | `[INTERCEPT(set=condition_check)]` |
| `TransformDescriptor` | `[INTERCEPT(get=read_transform, set=write_transform)]` |
| `ExpiringDescriptor` | `[INTERCEPT(get=ttl_check), TAG(ttl=<seconds>)]` |
| `AuditDescriptor` | `[INTERCEPT(get=audit_read, set=audit_write), TAG(audited=True)]` |
| `PooledDescriptor` | `[INTERCEPT(set=pool_return, delete=pool_return), TAG(pooled_field=True)]` |

`DescriptorComposer.collect_steps(descriptor)` aggregates steps from the entire descriptor chain.

---

## Appendix C: Complete Decorator Decomposition Reference

All composite decorators with their Step lists.

| Decorator | Steps |
|-----------|-------|
| `@component` | `DESCRIBE + REGISTER(ComponentRegistry)` |
| `@system(phase)` | `REGISTER(SystemRegistry) + TAG(scheduling={phase}) + DESCRIBE` |
| `@tracked` | `TRACK` |
| `@validated(c)` | `VALIDATE(c) + TAG(constraint=c)` |
| `@networked(a)` | `TAG(network={authority:a}) + HOOK(on_serialize) + HOOK(on_deserialize) + REGISTER(NetworkSystem)` |
| `@serializable(f)` | `DESCRIBE + TAG(serialization={format:f}) + HOOK(on_serialize) + HOOK(on_deserialize)` |
| `@serialized` | `DESCRIBE + HOOK(on_serialize) + HOOK(on_deserialize)` |
| `@transient` | `TAG(transient=True)` |
| `@readonly` | `INTERCEPT(set=deny)` |
| `@computed(fn)` | `INTERCEPT(get=fn) + TAG(transient=True, computed=True)` |
| `@pooled(n)` | `TAG(pool={size:n}) + HOOK(on_create) + HOOK(on_destroy) + REGISTER(PoolManager)` |
| `@behavior_tree(id)` | `TAG(ai={tree_id}) + REGISTER(AIRegistry) + HOOK(on_tick)` |
| `@destructible(h,f)` | `TAG(destruction={h,f}) + HOOK(on_damage) + HOOK(on_destroy) + REGISTER(DestructionSystem)` |
| `@native(b,n)` | `TAG(compile={backend:b, nogil:n}) + HOOK(on_compile)` |
| `@parallel(c)` | `TAG(execution={parallel, chunk_size:c})` |

---

## Appendix D: Implementation History

Phases 1-10 from the original specification roadmap are complete. Here is a summary of what each phase accomplished.

| Phase | Title | What It Accomplished |
|-------|-------|---------------------|
| **1** | Clean Dead Code from `base.py` | Deprecated `make_marker_decorator()` and `make_configurable_decorator()` (superseded by `make_decorator()` from `ops.py`). Removed from exports. Fixed `merge_attributes()` silent data loss, added runtime `DeprecationWarning`, cleaned imports, extracted magic sentinels. 70 tests. |
| **2** | Add `_steps` to Every Descriptor | Added `descriptor_steps` property to `TrinityDescriptor` protocol and all 17+ descriptor implementations. Each descriptor now declares what Ops it performs as Steps. Added `DescriptorComposer.collect_steps()` for chain aggregation. Exported missing persistence/debug/async descriptors. 45 tests. |
| **3** | Add `_metaclass_steps` to Every Metaclass | All 8 metaclasses now record their `__new__` operations as Steps in `cls._metaclass_steps`. Fixed critical double-init bug where 4 metaclasses were wiping parent steps. Normalized registry names. 86 tests. |
| **4** | Unified `decompose()` -- Cross-Layer Introspection | `decompose()` now accepts classes (not just decorators), collecting Steps from all three layers. Added `decompose_layered()` for structured output. Updated `expand()` for layered display. 17 tests. |
| **5** | Implement Missing Composition Rules | All documented rules from LANG_DEC.md now in code: REGISTER-last, TAG(network) requires TAG(serialization), INTERCEPT(set=deny) conflicts with VALIDATE. Added `validate_ordering()` for canonical Step order enforcement. 27 tests. |
| **6** | Implement Introspection API | Created `trinity/decorators/introspection.py` with all functions from LANG_DEC.md: `primitives()`, `composites()`, `chain()`, `find_decorators()`, `compose()`, `validate_combination()`, `all_rules()`. Added `get_definitions()` public accessor in ops.py. 32 tests. |
| **7** | Build Missing Descriptors | Created 12 new descriptor classes: `ImmutableDescriptor`, `VersionedDescriptor`, `IndexedDescriptor`, `AtomicDescriptor`, `InterpolatedDescriptor`, `SparseDescriptor`, `RateLimitedDescriptor`, `ConditionalDescriptor`, `TransformDescriptor`, `ExpiringDescriptor`, `AuditDescriptor`, `PooledDescriptor`. Each follows the BaseDescriptor pattern with `descriptor_steps`. |
| **8** | `Annotated` Field Syntax | `ComponentMeta._process_fields()` now detects `Annotated` type hints, extracts base type and metadata, instantiates descriptor classes/instances, and builds chains via `DescriptorComposer.compose()`. Supports `current: Annotated[float, Tracked, Range(0, 100)] = 100`. |
| **9** | Metaclass Auto-Install of Descriptors | `ComponentMeta._install_descriptors()` reads `_applied_steps` to auto-install missing descriptors (TrackedDescriptor, ValidatedDescriptor, SerializableDescriptor) when Op Steps indicate they should be present. |
| **10** | Tooling | Created `trinity/tools/` with 4 tools: `doctor.py` (validation), `step_trace.py` (trace formatting), `op_coverage.py` (coverage analysis), `lint.py` (import-time validation). All importable from `trinity.tools`. |

**Total: ~160 atomic tasks across 10 phases, all complete.**

Phase dependency graph: Phases 1, 2, 5 ran in parallel. Phase 3 depended on 2. Phase 4 depended on 2+3. Phase 6 depended on 4. Phase 7 started after 2. Phase 8 depended on 7. Phase 9 depended on 2+3. Phase 10 depended on 4+5.
