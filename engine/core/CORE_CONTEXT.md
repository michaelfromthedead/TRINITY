# CORE_CONTEXT.md — Engine Core Layer

> **Purpose:** Collect ALL Trinity decorators, metaclasses, descriptors, Foundation
> integration points, architecture specs, decorator stacks, and canonical usage examples
> needed to implement `engine/core/`. This document is the single source of truth —
> implementation should require zero external references.

**Implementation Status:**
| Layer | Status | Notes |
|-------|--------|-------|
| Python | ✅ COMPLETE | All algorithms implemented |
| Rust | ⚠️ 49% | GAPSET_1_CORE — ThreadPool, JobGraph, Scheduler absent |
| Wired | ❌ | Blocked on Rust completion |

*See `docs/STATUS.md` for current progress. See `docs/gap_sets/GAPSET_1_CORE/` for tasks.*

---

## 1. Architecture Summary

The **Core** layer (`engine/core/`) is the **runtime systems** layer — the active
machinery that powers the engine. It sits alongside `engine/common/` (shared types,
containers, utilities) and together they form **Layer 3: Core Systems** in the
engine stack.

**Core vs Common:**
- `engine/common/` = shared vocabulary (Vec3, Entity, HashMap, Clock, constants)
- `engine/core/` = active machinery (World, SystemScheduler, Allocators, TaskScheduler, Engine loop, Session)

**Core provides:**
- **Engine Bootstrap** — Engine class, game loop, frame phases, initialization
- **System Scheduler** — Topological sort, phase execution, parallel dispatch
- **ECS World** — Entity container, archetype storage, queries, command buffers, hierarchy
- **Memory Allocators** — Linear, Stack, Pool, Ring, Slab, TLSF runtime allocators
- **Task Scheduler** — Job graph, work-stealing, parallel patterns, fibers
- **Session** — Save/load entire engine state, crash recovery

**Layer Stack:**
```
┌─────────────────────────────────────────────────────────────────────┐
│  GAMEPLAY, UI, AUDIO, RENDERING, ANIMATION, SIMULATION             │
├─────────────────────────────────────────────────────────────────────┤
│  CORE SYSTEMS                                                       │
│  ┌──────────────────────────┐  ┌────────────────────────────────┐  │
│  │ engine/core/ (runtime)   │  │ engine/common/ (shared types)  │  │
│  │ Engine, World, Scheduler │  │ Vec3, Entity, HashMap, Clock   │  │
│  │ Allocators, TaskSched    │  │ Constants, Containers, Utils   │  │
│  │ Session, Bootstrap       │  │ Math, Reflection               │  │
│  └──────────────────────────┘  └────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  PLATFORM LAYER                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  FOUNDATION + TRINITY (Runtime + Definition Time)                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Dependency Rule:** Core depends on Common, Foundation, Trinity. Upper layers depend on Core. Core NEVER depends on upper layers.

**Exports to Upper Layers:**
```python
# engine/core/ public API
Engine          # Main engine class, game loop
World           # Central entity container
SystemScheduler # Phase-based system execution
TaskScheduler   # Parallel job execution
FrameAllocator  # Per-frame allocation
Session         # Save/load engine state
```

---

## 2. Trinity Decorators of Interest

### 2.1 ECS Core Decorators (Tier: FOUNDATION)

These define the types that Core's runtime systems manage.

| Decorator | Parameters | Op Types | Purpose |
|-----------|-----------|----------|---------|
| `@component` | `name: Optional[str]` | TAG, REGISTER | Mark class as ECS component — World stores these |
| `@tag` | — | TAG, REGISTER | Zero-sized component — archetype filtering only |
| `@resource` | `name: Optional[str]` | TAG, REGISTER | Singleton resource — ResourceMeta enforces single instance |
| `@event` | — | TAG, REGISTER | Event type — EventMeta pools these |
| `@system` | `phase: str = "update"` | TAG, REGISTER | Mark as ECS system — SystemScheduler executes these |
| `@query` | `components=(), with_=(), without=(), maybe=()` | TAG, REGISTER | Declarative query — World.query() resolves these |
| `@bundle` | — | TAG, DESCRIBE, REGISTER | Spawn template — World.spawn_bundle() uses these |
| `@relation` | `kind="one_to_many", exclusive=False` | TAG, REGISTER | Entity relationships — hierarchy system |
| `@derived` | `from_components=(), cache=True` | TAG, REGISTER | Computed components — auto-recomputed on dependency change |

### 2.2 Scheduling Decorators (Tier: SCHEDULING) — 12 decorators

These directly control how SystemScheduler dispatches systems.

| Decorator | Parameters | Purpose |
|-----------|-----------|---------|
| `@phase` | `name, after=(), before=()` | Assign system to named phase with ordering |
| `@parallel` | `chunk_size=64, min_batch=256` | Mark system as parallelizable; excludes @exclusive |
| `@exclusive` | — | System requires sole access; excludes @parallel |
| `@after` | `*systems` | Run after specified systems |
| `@before` | `*systems` | Run before specified systems |
| `@run_if` | `condition: callable` | Conditionally skip system execution |
| `@fixed` | `hz=60` | Fixed-timestep execution; excludes @throttle |
| `@job` | `priority=0, affinity="any", stack_size=65536` | Task system job configuration |
| `@async_system` | — | Mark as async/coroutine system |
| `@throttle` | `max_hz=None, max_ms=None` | Rate-limit execution; excludes @fixed |
| `@deferred` | — | Deferred execution (command buffer pattern) |
| `@chain` | `*systems` | Chain systems into sequential pipeline |

**Scheduling decorator details:**

**`@parallel(chunk_size=64, min_batch=256)`**
- Op Types: TAG (3x), REGISTER
- Unique: True
- Excludes: `("exclusive",)`
- Sets: `_parallel=True`, `_parallel_chunk_size`, `_parallel_min_batch`
- Core use: TaskScheduler splits query iteration across worker threads

**`@exclusive`**
- Op Types: TAG, REGISTER
- Unique: True
- Excludes: `("parallel",)`
- Sets: `_exclusive=True`
- Core use: SystemScheduler runs this system alone (no concurrent systems)

**`@fixed(hz=60)`**
- Op Types: TAG (3x), REGISTER
- Unique: True
- Excludes: `("throttle",)`
- Sets: `_fixed=True`, `_fixed_hz`, `_fixed_delta=1.0/hz`, `_fixed_accumulator=0.0`
- Core use: Engine loop accumulates time, runs system at fixed rate

**`@job(priority=0, affinity="any", stack_size=65536)`**
- Op Types: TAG (4x), REGISTER
- Unique: True
- Sets: `_job=True`, `_job_priority`, `_job_affinity`, `_job_stack_size`
- Core use: TaskScheduler assigns to worker with matching affinity
- Valid affinities: "any", "main", "render", "worker", "io"

**`@after(*systems)` / `@before(*systems)`**
- Op Types: TAG (2x), REGISTER
- Unique: False (accumulates)
- Sets: `_after=systems`, `_after_names` / `_before=systems`, `_before_names`
- Core use: SystemScheduler topological sort respects these edges

**`@run_if(condition)`**
- Op Types: TAG (2x), REGISTER
- Unique: False
- Sets: `_run_if=condition`, `_run_if_name`
- Core use: SystemScheduler checks condition before dispatching

**`@chain(*systems)`**
- Op Types: TAG (3x), REGISTER
- Unique: True
- Sets: `_chain=True`, `_chain_systems`, `_chain_names`
- Also sets on each chained system: `_chain_member`, `_chain_index`, `_chain_class`, `_after`
- Core use: SystemScheduler treats chain as atomic sequential pipeline

**`@deferred`**
- Op Types: TAG, REGISTER
- Unique: True
- Sets: `_deferred=True`
- Core use: System writes go to CommandBuffer, flushed after phase

**`@async_system`**
- Op Types: TAG, REGISTER
- Unique: True
- Sets: `_async_system=True`, `_is_coroutine`
- Core use: TaskScheduler wraps in fiber/coroutine execution

### 2.3 Time/Determinism Decorators (Tier: TIME) — 4 decorators

| Decorator | Parameters | Purpose |
|-----------|-----------|---------|
| `@deterministic` | — | Mark as deterministic (no floats in sim) |
| `@time_scale` | `layer="gameplay", min_scale=0.0, max_scale=10.0` | Per-layer time scaling |
| `@pausable` | `pause_layers={"gameplay"}` | System pauses when layer paused |
| `@rewindable` | `history_seconds=5.0, interpolation="linear"` | Enables time rewind |

**`@deterministic`**
- Sets: `_deterministic=True`
- Core use: Engine loop enforces fixed-point math, deterministic ordering

**`@time_scale(layer="gameplay")`**
- Sets: `_time_scale=True`, `_time_layer`, `_time_min_scale`, `_time_max_scale`
- Core use: FrameTimer provides per-layer delta times

**`@pausable(pause_layers={"gameplay"})`**
- Sets: `_pausable=True`, `_pause_layers`
- Core use: SystemScheduler skips system when its layer is paused

**`@rewindable(history_seconds=5.0, interpolation="linear")`**
- Sets: `_rewindable=True`, `_rewind_history`, `_rewind_interpolation`
- Valid interpolation: "linear", "cubic", "hermite"
- Core use: Session stores snapshots, rewind replays from checkpoint

### 2.4 Memory Decorators (Tier: MEMORY) — 12 decorators

These configure how Core's allocators manage component storage.

| Decorator | Parameters | Purpose |
|-----------|-----------|---------|
| `@pooled` | `initial_size, grow_factor, max_size` | Pool allocator for components |
| `@packed` | `layout="soa"` | Memory layout (aos/soa/hybrid) |
| `@aligned` | `bytes=64` | Cache-line alignment |
| `@arena` | `name="default"` | Arena allocator scope |
| `@flyweight` | — | Shared instance caching |
| `@intern` | — | String interning |
| `@generations` | — | Generational handle validation |
| `@copy_on_write` | — | CoW semantics |
| `@inline_array` | `size` | Fixed-size inline storage |
| `@budget` | `category, max_bytes, warn_at=0.8` | Memory budget tracking |
| `@allocator` | `type, size, thread_safe=False` | Custom allocator assignment |
| `@atomic` | — | Atomic load/store/CAS |

### 2.5 Data Flow Decorators (Tier: DATA_FLOW)

| Decorator | Parameters | Purpose |
|-----------|-----------|---------|
| `@serializable` | `format="binary", version=1` | Session save/load |
| `@networked` | `relevance, authority, priority, ...` | Network replication |
| `@snapshot` | `history_frames=60` | State snapshots for rewind |
| `@versioned` | `version=1, migrations={}` | Schema versioning |

### 2.6 Lifecycle Decorators (Tier: LIFECYCLE)

| Decorator | Parameters | Purpose |
|-----------|-----------|---------|
| `@on_add` | `component` | Hook when component added to entity |
| `@on_remove` | `component` | Hook when component removed from entity |
| `@on_change` | `component` | Hook when component field changes |
| `@on_spawn` | — | Hook on entity spawn |
| `@on_despawn` | — | Hook on entity despawn |

Core use: World fires these hooks during entity/component lifecycle operations.

### 2.7 Debug/Safety Decorators (Tier: DEBUG_SAFETY)

| Decorator | Parameters | Purpose |
|-----------|-----------|---------|
| `@reads` | `*components` | Declare read access for scheduler |
| `@writes` | `*components` | Declare write access for scheduler |
| `@track_changes` | `fields: Optional[list]` | Enable dirty flags |
| `@trace_stack` | `depth=3, show_decorator_chain=True` | Debug call tracing |

Core use: SystemScheduler uses @reads/@writes to determine parallelism safety.

### 2.8 Dev/Optimization Decorators (Tier: DEV) — 9 decorators

| Decorator | Parameters | Purpose |
|-----------|-----------|---------|
| `@profile` | `name` | CPU profiling with timing stats |
| `@gpu_profile` | `category, include_memory=False` | GPU profiling |
| `@trace` | `level="debug"` | Execution tracing |
| `@reloadable` | `enabled, preserve, reinitialize, validate` | Hot reload support |
| `@editor` | `category="General", hidden=False` | Editor integration metadata |
| `@test` | `cases, fuzz=False, property_based=False` | Test case declaration |
| `@bench` | `iterations=1000, warmup=100` | Benchmark declaration |
| `@invariant` | `check: callable, when="debug"` | Runtime invariant checking |
| `@deprecated` | `since, replacement=None, remove_in=None` | Deprecation warnings |

Core use: @profile on systems for perf tracking, @invariant for World consistency checks, @reloadable for hot-reload of system logic.

---

## 3. Metaclasses of Interest

### 3.1 EngineMeta (Base)
- Global debug registry, common `__repr__`
- All metaclasses inherit from this
- Core use: Engine class itself uses EngineMeta

### 3.2 ComponentMeta — **Critical for Core**
- **Registry:** `_registry`, `_components` (ClassVar dicts)
- **Processes:** Field types, descriptor installation, pool config, network config
- **Sets:** `_component_id`, `_component_name`, `_field_types`, `_field_descriptors`, `_field_offsets`, `_field_defaults`, `_track_changes`, `_network_config`, `_serialization_config`
- **Core integration:** World reads `_registry` at startup to initialize archetype storage. `_field_offsets` drives SoA layout. `_pool_config` drives pool allocator sizing.

### 3.3 SystemMeta — **Critical for Core**
- **Registry:** `_systems`, `_dependencies` (ClassVar dicts)
- **Processes:** Dependency analysis, phase ordering, parallelization grouping
- **Sets:** `_system_id`, `_system_name`, `_dependencies`, `_can_parallelize`, `_reads`, `_writes`, `_resources`, `_exclusive`, `_priority`
- **Core integration:** SystemScheduler reads `_systems` registry to discover all systems. `_dependencies` drives topological sort. `_reads`/`_writes` determine parallel safety. `_system_phase` assigns to frame phase.

### 3.4 ResourceMeta — **Critical for Core**
- **Registry:** `_resources`, `_instances` (ClassVar dicts)
- **Enforces:** Singleton pattern, dependency-ordered initialization
- **Sets:** `_resource_id`, `_resource_name`, `_resource_priority`, `_resource_dependencies`
- **Core integration:** Engine bootstrap initializes resources in priority order. World provides resource access to systems.

### 3.5 EventMeta — **Critical for Core**
- **Registry:** `_events`, `_pools` (ClassVar dicts)
- **Processes:** Event pooling, serialization
- **Sets:** `_event_id`, `_event_name`, `_event_fields`, `_event_parent_ids`, `_event_priority`, `_event_channels`, `_event_pooled`
- **Core integration:** World event bus dispatches events. Pool reduces allocation pressure.

### 3.6 AssetMeta
- **Registry:** `_assets` (ClassVar dict)
- **Sets:** `_asset_id`, `_asset_name`, `_asset_type_code`, `_asset_extensions`
- **Core integration:** Asset system (upper layer) uses this; Core provides handle storage.

### 3.7 StateMeta
- **Registry:** `_states` (ClassVar dict)
- **Sets:** `_state_id`, `_state_name`, `_state_transitions`, `_state_machine_cls`
- **Core integration:** State machine systems reference state registry.

### 3.8 ProtocolMeta
- **Registry:** `_protocols` (ClassVar dict)
- **Sets:** `_protocol_id`, `_protocol_version`, `_protocol_messages`
- **Core integration:** Networking layer uses this; Core provides transport.

---

## 4. Descriptors of Interest

### 4.1 Core Descriptors (Used by World, Scheduler, Allocators)

| Descriptor | ID | Runtime Behavior | Core Use |
|-----------|-----|-----------------|----------|
| `TrackedDescriptor` | `"tracked"` | Notifies Tracker on set | Query cache invalidation via dirty flags |
| `ValidatedDescriptor` | `"validated"` | Validates before set | Component field constraints |
| `RangeDescriptor` | `"validated"` | Clamps to min/max | Bounded component values |
| `SerializableDescriptor` | `"serializable"` | Controls serialization | Session save/load |
| `TransientDescriptor` | `"transient"` | Excluded from serialization | Runtime-only fields |
| `ObservableDescriptor` | `"observable"` | Notifies subscribers | Reactive component updates |
| `ProfiledDescriptor` | `"profiled"` | Records timing | System/allocator profiling |

### 4.2 Phase 7 Descriptors (Advanced Runtime)

| Descriptor | ID | Runtime Behavior | Core Use |
|-----------|-----|-----------------|----------|
| `ImmutableDescriptor` | `"immutable"` | Prevents modification after first set; excludes tracked/observable/networked | Archetype IDs, entity handles |
| `VersionedDescriptor` | `"versioned"` | Per-field version counter; increments on change | Cache invalidation, optimistic concurrency |
| `IndexedDescriptor` | `"indexed"` | Class-level index mapping values→objects; optional uniqueness | Fast entity lookup by component value |
| `AtomicDescriptor` | `"atomic"` | Thread-safe get/set with Lock; compare_and_swap support | Concurrent system access, command buffer |
| `InterpolatedDescriptor` | `"interpolated"` | Smooths between snapshots; linear/hermite modes | Network entity rendering |
| `SparseDescriptor` | `"sparse"` | Stores only non-default values; memory optimization | Large component arrays with defaults |
| `RateLimitedDescriptor` | `"rate_limited"` | Throttles writes per second; raise/drop policies | Network update frequency |
| `ConditionalDescriptor` | `"conditional"` | Gates writes with predicate function | Authority checks, validation |
| `TransformDescriptor` | `"transform"` | Bidirectional transforms on read/write | Coordinate system conversion |
| `ExpiringDescriptor` | `"expiring"` | TTL-based value expiry; returns default when expired | Cache entries, temp buffs |
| `AuditDescriptor` | `"audit"` | Append-only audit log of accesses | Debug, replay analysis |
| `PooledDescriptor` | `"pooled_field"` | Object pool for field values; acquire/release | Reduce allocations |

### 4.3 Descriptor Composition (DescriptorComposer)

Descriptors chain via `Annotated` syntax:
```python
from typing import Annotated
from trinity.descriptors import Tracked, Range, Atomic, Versioned

class Health(Component):
    # Outer → Inner: Atomic wraps Versioned wraps Tracked wraps Range wraps storage
    hp: Annotated[float, Atomic(), Versioned(), Tracked(), Range(0, 100)]
```

Composition order: Outermost descriptor's `__set__` fires first, delegates inward.

---

## 5. Foundation Integration Points

### 5.1 Registry — System Discovery & Type Management
- **API:** `register(cls, name, track_instances)`, `get(name)`, `all_types()`, `subclasses(base)`, `types_with_decorator(name)`, `types_where(predicate)`, `instances(cls)`, `instance_count(cls)`, `set_metadata()`, `get_metadata()`, `describe(cls)`
- **Core use:**
  - Engine bootstrap: `registry.subclasses(System)` → discover all systems
  - Engine bootstrap: `registry.subclasses(Resource)` → initialize resources in priority order
  - World init: `registry.subclasses(Component)` → pre-allocate archetype storage
  - SystemScheduler: `registry.types_with_decorator("parallel")` → identify parallelizable systems

### 5.2 Tracker — Change Detection & Transactions
- **API:** `mark_dirty(obj, field, old, new)`, `is_dirty(obj)`, `dirty_fields(obj)`, `mark_clean(obj)`, `all_dirty()`, `on_change(target, callback)`, `off_change(callback)`, `begin_transaction(name)`, `commit_transaction()`, `rollback_transaction()`, `undo()`, `redo()`
- **Core use:**
  - World.query(Changed<T>): uses `is_dirty()` to filter modified components
  - End of frame: `mark_clean()` on all processed entities
  - Session: `begin_transaction("frame_N")` wraps each frame for rollback
  - Editor: `undo()` / `redo()` for editor operations

### 5.3 EventLog — Operation Recording & Causal Chains
- **API:** `record(event)`, `events_at(tick)`, `events_for_entity(id)`, `events_for_operation(op)`, `events_caused_by(entity)`, `events_where(**kwargs)`, `@traced` decorator
- **Core use:**
  - System execution: `@traced` on system.execute() records timing and entity changes
  - Entity lifecycle: record create/destroy events with tick
  - Deterministic replay: replay events_at(tick) to reconstruct state
  - Debug: `events_caused_by(entity)` traces causality

### 5.4 Mirror — Runtime Introspection
- **API:** `mirror(obj_or_cls)`, `get(name)`, `set(name, value)`, `has(name)`, `to_dict()`, `get_path(dotted)`, `set_path(dotted, value)`, `fields`, `methods`, `describe()`, `schema_hash(cls)`
- **Core use:**
  - Inspector: enumerate component fields at runtime
  - Serializer: `schema_hash()` for migration detection
  - ShellLang: REPL access to live entities

### 5.5 Serializer — Persistence
- **API:** `to_dict(obj)`, `from_dict(data)`, `to_bytes(obj)`, `from_bytes(data)`, `to_file(obj, path)`, `from_file(path)`, `deep_copy(obj)`, `diff(a, b)` → Delta, `patch(obj, delta)`
- **Core use:**
  - Session.save(): serialize entire World state
  - Session.load(): deserialize and reconstruct World
  - Prefab: `deep_copy()` for template instantiation
  - Network: `diff()` for delta compression

### 5.6 Bridge — Trinity ↔ ShellLang World
- **API:** `get_trinity_registry()`, `create_world_from_trinity()`, `create_ai_interface()`, `create_shell()`, `TrinityWorldAdapter`
- **Core use:**
  - World initialization: `create_world_from_trinity()` pre-populates component registry
  - AI agents: `create_ai_interface()` for AI-driven entity manipulation
  - Debug shell: `create_shell()` for live REPL
  - Runtime sync: `TrinityWorldAdapter.sync_from_foundation_registry()`

---

## 6. Architecture Spec Details

> **Reference:** `DIAGRAMS/ARCHITECTURE_CORE.md`, `DIAGRAMS/ARCHITECTURE.md`

### 6.1 Engine Bootstrap & Game Loop

**Engine Class (using EngineMeta):**
```python
class Engine(metaclass=EngineMeta):
    world: World
    scheduler: SystemScheduler
    task_scheduler: TaskScheduler
    frame_timer: FrameTimer
    session: Session
    
    def initialize(self):
        # 1. Initialize platform (window, input, GPU)
        # 2. Initialize resources (priority order from ResourceMeta)
        # 3. Initialize World (from Bridge.create_world_from_trinity())
        # 4. Discover and register all Systems (from SystemMeta._systems)
        # 5. Build system dependency graph
        # 6. Initialize allocators
        # 7. Restore session (if crash recovery)
    
    def run(self):
        while self.running:
            self.frame_timer.begin_frame()
            self.update()
            self.frame_timer.end_frame()
    
    def update(self):
        # Fixed timestep loop
        accumulator += delta_time
        while accumulator >= fixed_dt:
            self.fixed_update(fixed_dt)  # Simulation
            accumulator -= fixed_dt
        self.variable_update(delta_time)  # Rendering
```

**Frame Phases (execution order):**
```
Input → PrePhysics → Physics → PostPhysics → PreUpdate → Update → PostUpdate → PreRender → Render → PostRender → Audio → Cleanup
```

Maps to `SystemPhase` enum:
- PRE_PHYSICS = 0
- PHYSICS = 1
- POST_PHYSICS = 2
- PRE_UPDATE = 3
- UPDATE = 4
- POST_UPDATE = 5
- PRE_RENDER = 6
- RENDER = 7

Plus engine-defined phases: Input, Audio, Cleanup.

### 6.2 System Scheduler

**Responsibilities:**
1. Discover systems from `SystemMeta._systems` registry
2. Assign to phases from `@system(phase=...)` and `@phase(name=...)`
3. Build dependency graph from `@after`, `@before`, `SystemMeta._dependencies`
4. Topological sort within each phase
5. Identify parallel groups from `@reads`/`@writes` conflict analysis
6. Execute: sequential for `@exclusive`, parallel for `@parallel`
7. Handle `@fixed` accumulator, `@throttle` rate limiting, `@run_if` conditions
8. Process `@chain` as atomic sequential pipelines
9. Flush `@deferred` command buffers after each phase

**System Lifecycle:**
```
Definition Time:
  SystemMeta.__init_subclass__ → registers system, extracts reads/writes/phase

Engine Init:
  SystemScheduler.build_graph() → topological sort, parallel group detection

Each Frame:
  For each phase:
    For each system group (sequential or parallel):
      If @run_if condition passes:
        If @fixed: accumulate time, run N times at fixed_dt
        If @throttle: check rate limit
        If @parallel: TaskScheduler.parallel_for_each()
        If @exclusive: run alone
        If @deferred: flush CommandBuffer after
```

### 6.3 ECS World

**Entity:** `{ index: u32, generation: u32 }` — generational index for safe reuse.

**Archetype Storage:**
- Each unique combination of component types = one Archetype
- Components stored in SoA (Structure of Arrays) for cache efficiency
- ArchetypeGraph: tracks transitions (add component → new archetype, remove → new archetype)
- ComponentArray: `{ id, element_size, data: void*, capacity, count }`

**World Operations:**
```python
# Entity lifecycle
entity = world.create()
world.add(entity, Transform(x=10))
world.add(entity, Velocity(vx=1))
world.has(entity, Transform)  # True
world.get(entity, Transform)  # Transform instance
world.remove(entity, Transform)
world.destroy(entity)
world.is_alive(entity)  # False (generation incremented)

# Queries
world.for_each(Transform, Velocity, fn=lambda t, v: ...)
world.par_for_each(Transform, Velocity, fn=lambda t, v: ...)  # parallel
view = world.view(Transform, Velocity)  # iterator

# With filters
world.for_each(Transform, With=[PlayerTag], Without=[StaticTag], fn=...)
world.for_each(Transform, Changed=True, fn=...)  # only dirty

# Command buffer (thread-safe deferred)
cmd = CommandBuffer()
cmd.spawn(Transform(), Velocity(), PlayerTag())
cmd.despawn(entity)
cmd.insert(entity, Health(100))
cmd.remove(entity, Velocity)
cmd.flush(world)  # apply all at once

# Hierarchy
world.set_parent(child, parent)
world.get_children(parent)
world.destroy_hierarchy(parent)  # destroys parent + all descendants

# Serialization
data = world.serialize()  # full snapshot
delta = world.serialize_delta(since_tick=100)  # incremental
world.deserialize(data)

# Prefabs
prefab = Prefab.from_entity(world, template_entity)
new_entity = prefab.instantiate(world)
new_entity = prefab.instantiate_at(world, position=Vec3(10, 0, 5))
```

**Bundles:**
```python
@bundle
class PlayerBundle:
    transform: Transform
    velocity: Velocity
    health: Health
    player: PlayerTag

# Spawns all 4 components atomically
entity = world.spawn_bundle(PlayerBundle(
    transform=Transform(x=0, y=0),
    velocity=Velocity(vx=1),
    health=Health(hp=100),
    player=PlayerTag()
))
```

### 6.4 Memory Allocators

**Allocator Hierarchy:**

| Allocator | Pattern | O(alloc) | O(free) | Use Case |
|-----------|---------|----------|---------|----------|
| `LinearAllocator` | Bump pointer | O(1) | reset only | Per-frame scratch |
| `StackAllocator` | LIFO stack | O(1) | O(1) LIFO | Scoped allocations |
| `PoolAllocator` | Fixed-size blocks | O(1) | O(1) | Component storage (@pooled) |
| `RingAllocator` | Circular buffer | O(1) | implicit | Streaming data |
| `SlabAllocator` | Size classes | O(1) | O(1) | Mixed-size allocs |
| `TLSFAllocator` | Two-Level Segregated Fit | O(1) | O(1) | General purpose, real-time |

**FrameAllocator (Special):**
```python
class FrameAllocator:
    def begin_frame(self): ...   # Reset to start
    def end_frame(self): ...     # Validate all freed
    def allocate(self, size): ...  # Bump pointer
```

**DoubleBufferAllocator:**
```python
class DoubleBufferAllocator:
    def allocate_current(self, size): ...
    def allocate_previous(self, size): ...
    def swap(self): ...  # Current becomes previous
```

**Memory Tracking:**
- `MemoryTag` enum: Unknown, Core, Rendering, Physics, Animation, Audio, Gameplay, UI, Network
- `AllocationInfo`: address, size, tag, file, line, frame
- `MemoryTracker`: track_allocation, track_free, get_stats, dump_leaks, guard_bytes

**Data Layout:**
- AoS: `[{x,y,z}, {x,y,z}, ...]` — simple, poor cache
- SoA: `{[x,x,...], [y,y,...], [z,z,...]}` — excellent cache for iteration
- AoSoA: chunks of 256 elements in SoA — compromise
- Controlled by `@packed(layout="soa")` decorator

### 6.5 Task Scheduler

**Task:** `{ function, user_data, priority, affinity }`
**Priority:** Critical > High > Normal > Low > Idle
**Affinity:** Any, Main, Render, Worker, IO

**TaskScheduler:**
```python
class TaskScheduler:
    def initialize(self, worker_count): ...
    def shutdown(self): ...
    def submit(self, task) -> Future: ...
    def submit_after(self, task, dependency) -> Future: ...
    def wait(self, future): ...
    def wait_all(self, futures): ...
    def parallel_for(self, range, fn, chunk_size=64): ...
    def parallel_for_each(self, items, fn): ...
    def worker_count(self) -> int: ...
    def is_main_thread(self) -> bool: ...
```

**TaskGraph:**
```python
builder = TaskGraphBuilder()
load_mesh = builder.task("load_mesh", load_mesh_fn)
load_tex = builder.task("load_texture", load_texture_fn)
create_mat = builder.task("create_material", create_material_fn)
create_mat.depends_on(load_mesh, load_tex)
finalize = builder.task("finalize", finalize_fn)
finalize.depends_on(create_mat)
graph = builder.build()
graph.execute(task_scheduler)
```

**Sync Primitives:**
- `TaskCounter`: increment/decrement, wait_until_zero
- `Future<T>`: get, is_ready, wait
- `Promise<T>`: set_value, set_exception, get_future
- `Latch`: count_down, wait, try_wait
- `Barrier`: arrive_and_wait, arrive_and_drop

### 6.6 Session

**Session Object:**
```python
class Session:
    world: World              # All entities and components
    settings: Settings        # Engine configuration
    editor_layout: EditorLayout
    selection: list[EntityRef]
    camera_position: Vec3
    open_inspectors: list[InspectorState]
    shell_history: list[str]
    undo_stack: list[Transaction]
    redo_stack: list[Transaction]
```

**Session Operations:**
```python
# Save/load
session.save(path)       # Foundation Serializer → file
session.load(path)       # Foundation Serializer ← file
session.auto_save()      # ContentStore with structural sharing

# Crash recovery
session.save_checkpoint() # Periodic snapshot
session.recover()         # Restore from latest checkpoint

# Delta sync
delta = session.diff(previous_session)  # Foundation DeltaSync
session.apply_delta(delta)              # Incremental restore
```

### 6.7 Deterministic Simulation

**Simulation Boundary:**
```
DETERMINISTIC (fixed-point, no floats):      PRESENTATION (floats OK):
  Input → Physics → Gameplay → State    →    Animation → Rendering → Audio
  Fixed16/Fixed32 types                       float types
  @deterministic systems                      Variable-rate systems
  EventLog records every op                   Interpolation from sim state
```

**Foundation Support:**
| System | Determinism Role |
|--------|-----------------|
| EventLog | Records all ops with causal chains → exact replay |
| Tracker | Transaction system → rollback to any checkpoint |
| Mirror | `schema_hash()` → detect desync across clients |
| Registry | Instance tracking → snapshot all live entities |
| ContentStore | Content-addressable snapshots → efficient diff |
| DeltaSync | Minimal delta patches → efficient netcode |
| Provenance | Derivation trees → debug "why did sim diverge?" |

---

## 7. Decorator Stacks of Interest

### 7.1 Core Stacks (from `builtin_stacks/core.py`)

**`production_component(pool_size, layout="soa", category="gameplay")`**
Composes: `@track_changes` + `@budget(category)` + `@pooled(initial_size=pool_size)` + `@packed(layout)` + `@component`
→ Core allocator uses pool_size for PoolAllocator, layout for SoA storage.

**`safe_system(phase="update", read=(), write=())`**
Composes: `@system(phase)` + `@reads(*read)` + `@writes(*write)`
→ SystemScheduler uses reads/writes for parallel safety analysis.

**`saveable_data(version=1, format="binary", migrations=None)`**
Composes: `@track_changes` + `@versioned(version, migrations)` + `@serializable(format)`
→ Session save/load with schema migration.

### 7.2 Persistence Stacks (from `builtin_stacks/persistence.py`)

**`versioned_saveable(version=1, migrations=None)`**
Composes: `@serializable(format="binary")` + `@versioned(version, migrations)` + `@track_changes`

**`deterministic_data()`**
Composes: `@component` + `@deterministic` + `@serializable(format="binary")` + `@track_changes`
→ For simulation-safe components.

**`replay_ready(history_frames=600, keyframe_interval=5.0)`**
Composes: `@recorded` + `@replay_authority` + `@serializable` + `@track_changes` + `@snapshot` + `@keyframe` + `@diff`
→ For rewindable/replayable components.

### 7.3 Development Stacks (from `builtin_stacks/development.py`)

**`profiled_dev(name, warn_ms=2.0)`**
Composes: `@profile(name, warn_ms)` + `@trace(level="debug")` + `@build_only({"debug", "development"})`
→ Dev-only system profiling.

### 7.4 Proposed Core-Specific Stacks

```python
# Fixed-timestep physics system with profiling
physics_system = stack(
    system(phase="physics"),
    fixed(hz=60),
    parallel(chunk_size=128),
    profile(name="physics"),
    name="physics_system"
)

# Exclusive system that needs sole access
exclusive_system = stack(
    system(phase="update"),
    exclusive,
    deferred,
    name="exclusive_system"
)

# Deterministic simulation component
sim_component = deterministic_data()  # Already exists

# Core resource singleton
core_resource = stack(resource, track_changes, name="core_resource")

# Engine event
engine_event = stack(event, name="engine_event")
```

---

## 8. TODO Checklist

> From `docs/GAME_ENGINE_INTEGRATION_TODO.md` sections 1 and 3

### 8.1 Engine Bootstrap & Game Loop
- [ ] Implement `Engine` class using `EngineMeta` metaclass
- [ ] Implement fixed-timestep game loop (fixed update + variable render)
- [ ] Implement frame phases: Input → Simulation → Animation → Rendering → Audio → Cleanup
- [ ] Wire phase ordering to `SystemMeta` phase assignments
- [ ] Integrate Foundation EventLog — record frame boundaries as Events
- [ ] Integrate Foundation Tracker — flush dirty flags per frame

### 8.2 System Scheduler
- [ ] Implement topological sort of Systems based on `SystemMeta._dependencies`
- [ ] Implement parallel system execution within phases (non-conflicting queries)
- [ ] Implement `@exclusive` systems that require sole access
- [ ] Wire `@phase` decorator to scheduler phase assignment
- [ ] Wire `@parallel` decorator to mark parallelizable systems
- [ ] Wire `@fixed` decorator to fixed-timestep accumulator
- [ ] Wire `@throttle` decorator to rate-limited execution
- [ ] Wire `@run_if` decorator to conditional execution
- [ ] Wire `@after`/`@before` to dependency edges
- [ ] Wire `@chain` to sequential pipeline execution
- [ ] Wire `@deferred` to command buffer flush after phase
- [ ] Wire `@async_system` to fiber/coroutine execution
- [ ] Integrate Foundation Registry — discover all registered Systems at startup

### 8.3 World Management (ECS Runtime)
- [ ] Implement World as the central entity container
- [ ] Implement entity lifecycle: create → attach → update → destroy
- [ ] Implement generational entity IDs (index + generation)
- [ ] Implement archetype-based component storage (SoA)
- [ ] Implement ArchetypeGraph (add/remove transitions)
- [ ] Implement query system (Query, With, Without, Optional, Changed)
- [ ] Implement par_for_each for parallel query iteration
- [ ] Implement CommandBuffer (thread-safe deferred modifications)
- [ ] Implement entity hierarchy (Parent, Children, transform propagation)
- [ ] Implement WorldSerializer (serialize, deserialize, delta)
- [ ] Implement Prefab system (instantiate, from_entity)
- [ ] Wire ALL ComponentMeta-registered types to archetype storage
- [ ] Wire `@query` decorator → runtime query execution
- [ ] Wire `@relation` decorator → entity relationship tracking
- [ ] Wire `@derived` decorator → computed component generation
- [ ] Wire `@bundle` decorator → entity spawn templates
- [ ] Wire lifecycle hooks (@on_add, @on_remove, @on_change, @on_spawn, @on_despawn)
- [ ] Wire Foundation Registry → archetype storage initialization
- [ ] Wire Foundation Tracker → dirty flag integration with query cache invalidation
- [ ] Wire Foundation Bridge → TrinityWorldAdapter syncs engine World ↔ ShellLang World
- [ ] Wire Foundation EventLog → entity lifecycle event recording

### 8.4 Memory Allocators
- [ ] Implement LinearAllocator (bump pointer, reset, mark/restore)
- [ ] Implement StackAllocator (LIFO alloc/free, markers)
- [ ] Implement PoolAllocator (fixed-size blocks, O(1) alloc/free)
- [ ] Implement RingAllocator (circular buffer)
- [ ] Implement SlabAllocator (size classes 8-1024+)
- [ ] Implement TLSFAllocator (two-level segregated fit)
- [ ] Implement FrameAllocator (per-frame linear, begin/end_frame)
- [ ] Implement DoubleBufferAllocator (current/previous swap)
- [ ] Implement ScopedAllocator (RAII lifetime)
- [ ] Implement MemoryTracker (stats, leak detection, guard bytes)
- [ ] Wire `@pooled` decorator → PoolAllocator for component storage
- [ ] Wire `@packed` decorator → SoA/AoS/AoSoA layout selection
- [ ] Wire `@aligned` decorator → alignment requirements
- [ ] Wire `@arena` decorator → arena allocator scope
- [ ] Wire `@budget` decorator → per-tag memory budget tracking
- [ ] Wire `@allocator` decorator → custom allocator assignment
- [ ] Integrate Foundation Mirror — expose allocator stats for inspection
- [ ] Integrate Foundation Tracker — track allocation/deallocation events

### 8.5 Task Scheduler
- [ ] Implement TaskScheduler (submit, wait, parallel_for, parallel_for_each)
- [ ] Implement work-stealing queue per worker thread
- [ ] Implement TaskGraph + TaskGraphBuilder
- [ ] Implement parallel patterns (reduce, transform, sort, filter)
- [ ] Implement sync primitives (TaskCounter, Future/Promise, Latch, Barrier)
- [ ] Implement fiber-based coroutines (or async equivalent)
- [ ] Wire `@parallel` decorator → task graph parallelism
- [ ] Wire `@exclusive` decorator → exclusive access requirements
- [ ] Wire `@job` decorator → job priority and affinity
- [ ] Wire SystemMeta phase/dependency info → task graph construction
- [ ] Integrate Foundation EventLog — profile task execution timing

### 8.6 Session
- [ ] Implement Session class (world, settings, editor state, undo/redo stacks)
- [ ] Implement Session.save() / Session.load() using Foundation Serializer
- [ ] Implement auto-save with Foundation ContentStore (structural sharing)
- [ ] Implement crash recovery from Session checkpoints
- [ ] Implement delta sync using Foundation DeltaSync
- [ ] Wire Foundation Tracker undo/redo stacks into Session

---

## 9. Directory Structure

```
engine/core/
├── __init__.py              # Public API: Engine, World, SystemScheduler, TaskScheduler, Session
├── CORE_CONTEXT.md          # This file
│
├── engine.py                # Engine class (EngineMeta), game loop, initialization
├── frame.py                 # Frame phases, fixed timestep logic
│
├── ecs/
│   ├── __init__.py          # Re-export: World, Entity, Query, CommandBuffer
│   ├── world.py             # World (entity container, create/destroy/query)
│   ├── archetype.py         # Archetype, ArchetypeGraph, ComponentArray (SoA)
│   ├── query.py             # Query, With, Without, Optional, Changed, View
│   ├── command_buffer.py    # CommandBuffer (thread-safe deferred ops)
│   ├── hierarchy.py         # Parent, Children, transform propagation
│   ├── prefab.py            # Prefab, WorldSerializer
│   ├── bundle.py            # Bundle instantiation from @bundle
│   ├── event_bus.py         # Event dispatch, subscriber management
│   └── lifecycle.py         # Lifecycle hook dispatch (@on_add, @on_remove, etc.)
│
├── scheduler/
│   ├── __init__.py          # Re-export: SystemScheduler
│   ├── scheduler.py         # SystemScheduler (phase execution, parallel dispatch)
│   ├── graph.py             # System dependency graph, topological sort
│   ├── phases.py            # Phase definitions, fixed timestep, throttle
│   └── parallel.py          # Parallel group detection from @reads/@writes
│
├── memory/
│   ├── __init__.py          # Re-export all allocators
│   ├── linear.py            # LinearAllocator, StackAllocator
│   ├── pool.py              # PoolAllocator (for @pooled components)
│   ├── ring.py              # RingAllocator
│   ├── slab.py              # SlabAllocator (size classes)
│   ├── tlsf.py              # TLSFAllocator (general purpose real-time)
│   ├── frame.py             # FrameAllocator, DoubleBufferAllocator, ScopedAllocator
│   ├── tracker.py           # MemoryTracker, AllocationInfo, MemoryTag
│   └── layout.py            # SoA/AoS/AoSoA helpers, cache alignment
│
├── tasks/
│   ├── __init__.py          # Re-export: TaskScheduler, TaskGraph
│   ├── scheduler.py         # TaskScheduler (submit, parallel_for, work-stealing)
│   ├── graph.py             # TaskGraph, TaskGraphBuilder
│   ├── sync.py              # TaskCounter, Future, Promise, Latch, Barrier
│   ├── worker.py            # Worker threads, work-stealing queues
│   └── fiber.py             # Fiber/coroutine support
│
└── session/
    ├── __init__.py          # Re-export: Session
    ├── session.py           # Session class (save/load/recover)
    ├── checkpoint.py        # Auto-save, crash recovery
    └── delta.py             # Delta sync, incremental saves
```

---

## 10. Canonical Usage Examples

### 10.1 Engine Bootstrap

```python
from engine.core import Engine

class MyGame(Engine):
    def on_init(self):
        # Resources auto-initialized by ResourceMeta priority order
        # Systems auto-discovered from SystemMeta._systems
        # World pre-populated via Bridge.create_world_from_trinity()
        pass
    
    def on_shutdown(self):
        self.session.save("autosave.session")

game = MyGame()
game.run()
```

### 10.2 Defining Systems with Scheduling

```python
from trinity.decorators.ecs_core import system
from trinity.decorators.scheduling import parallel, fixed, after, run_if, exclusive, chain
from trinity.decorators.debug_safety import reads, writes
from trinity.base import System

@system(phase="physics")
@fixed(hz=60)
@parallel(chunk_size=128)
@reads(Transform)
@writes(Velocity)
class GravitySystem(System):
    def execute(self, world, dt):
        world.par_for_each(Velocity, fn=lambda v: setattr(v, 'vy', v.vy - 9.8 * dt))

@system(phase="physics")
@fixed(hz=60)
@after(GravitySystem)
@reads(Velocity)
@writes(Transform)
class MovementSystem(System):
    def execute(self, world, dt):
        world.for_each(Transform, Velocity, fn=lambda t, v: (
            setattr(t, 'x', t.x + v.vx * dt),
            setattr(t, 'y', t.y + v.vy * dt),
        ))

@system(phase="update")
@exclusive  # Needs sole access — structural changes
class SpawnSystem(System):
    def execute(self, world, dt):
        cmd = CommandBuffer()
        # ... spawn/despawn logic ...
        cmd.flush(world)

@system(phase="update")
@run_if(lambda: game_state.is_playing)
class GameplaySystem(System):
    def execute(self, world, dt):
        ...
```

### 10.3 Deterministic Simulation Component

```python
from trinity.decorators.builtin_stacks.persistence import deterministic_data
from trinity.types import Fixed32

@deterministic_data()
class SimPosition(Component):
    x: Fixed32 = Fixed32(0)
    y: Fixed32 = Fixed32(0)
    z: Fixed32 = Fixed32(0)
```

### 10.4 World Usage with Queries

```python
from engine.core.ecs import World, CommandBuffer

world = World()

# Create entities
player = world.create()
world.add(player, Transform(x=0, y=0))
world.add(player, Velocity(vx=1, vy=0))
world.add(player, Health(hp=100))
world.add(player, PlayerTag())

# Query with filters
for transform in world.query(Transform, with_=[PlayerTag], without=[StaticTag]):
    print(f"Player at {transform.x}, {transform.y}")

# Changed-only query (uses Foundation Tracker dirty flags)
for health in world.query(Health, changed=True):
    if health.hp <= 0:
        world.defer_destroy(health.entity)

# Parallel iteration
world.par_for_each(Transform, Velocity, fn=update_position)
```

### 10.5 Memory Allocator Usage

```python
from engine.core.memory import FrameAllocator, PoolAllocator, MemoryTracker

# Frame allocator — reset every frame
frame_alloc = FrameAllocator(size=1024 * 1024)  # 1MB
frame_alloc.begin_frame()
temp_data = frame_alloc.allocate(256)
# ... use temp_data ...
frame_alloc.end_frame()  # All freed

# Pool allocator — for @pooled components
pool = PoolAllocator(element_size=64, initial_count=1024)
ptr = pool.allocate()
pool.free(ptr)

# Memory tracking
tracker = MemoryTracker()
tracker.track_allocation(ptr, size=64, tag=MemoryTag.Gameplay)
stats = tracker.get_stats(MemoryTag.Gameplay)
tracker.dump_leaks()
```

### 10.6 Task Graph

```python
from engine.core.tasks import TaskScheduler, TaskGraphBuilder

scheduler = TaskScheduler(worker_count=4)

builder = TaskGraphBuilder()
t1 = builder.task("load_mesh", load_mesh_fn)
t2 = builder.task("load_texture", load_texture_fn)
t3 = builder.task("create_material", create_material_fn)
t3.depends_on(t1, t2)
t4 = builder.task("finalize", finalize_fn)
t4.depends_on(t3)

graph = builder.build()
graph.execute(scheduler)
graph.wait()  # Block until all complete
```

### 10.7 Session Save/Load

```python
from engine.core.session import Session

session = Session(world=world, settings=settings)

# Save
session.save("savegame.bin")

# Load
session = Session.load("savegame.bin")
world = session.world  # Fully reconstructed

# Auto-save with structural sharing
session.auto_save()  # Uses Foundation ContentStore

# Crash recovery
if Session.has_checkpoint():
    session = Session.recover()
```

### 10.8 Foundation Integration in Core

```python
from foundation.registry import registry
from foundation.tracker import tracker
from foundation.eventlog import get_event_log, traced, set_current_tick
from foundation.bridge import create_world_from_trinity

# Engine init: create world from Trinity registry
world = create_world_from_trinity()

# System scheduler: discover all systems
all_systems = registry.subclasses(System)
parallel_systems = registry.types_with_decorator("parallel")

# Frame execution with event logging
set_current_tick(frame_number)
tracker.begin_transaction(f"frame_{frame_number}")

for system in scheduled_systems:
    system.execute(world, dt)

tracker.commit_transaction()

# End of frame: clear dirty flags
for obj in tracker.all_dirty():
    tracker.mark_clean(obj)
```

---

## 11. Key Integration Patterns

### Pattern 1: SystemMeta → SystemScheduler
```
Definition Time:
  @system(phase="physics") + @parallel + @reads(T) + @writes(V)
    → SystemMeta stores: _system_phase, _can_parallelize, _reads, _writes

Engine Init:
  SystemScheduler.build_graph():
    → For each system in SystemMeta._systems:
      → Assign to phase from _system_phase
      → Add edges from _after/_before
      → Detect parallel groups: systems with non-overlapping _reads/_writes
    → Topological sort per phase

Each Frame:
  SystemScheduler.run_phase("physics"):
    → For parallel group: TaskScheduler.parallel_for_each()
    → For exclusive system: run alone
    → For deferred system: CommandBuffer.flush() after
```

### Pattern 2: ComponentMeta → Archetype Storage
```
Definition Time:
  @component + @pooled(1024) + @packed(layout="soa")
    → ComponentMeta stores: _pool_config, _packed_layout, _field_offsets

Engine Init:
  World.initialize():
    → For each in ComponentMeta._registry:
      → Create initial Archetype with SoA storage
      → Initialize PoolAllocator(initial_size=_pool_config.initial_size)

Runtime (add component):
  world.add(entity, Transform()):
    → Find current archetype for entity
    → Find target archetype (current + Transform)
    → Migrate entity data to new archetype
    → Fire @on_add lifecycle hook
```

### Pattern 3: @fixed → Accumulator Pattern
```
@fixed(hz=60) on system:
  → SystemScheduler stores: fixed_dt = 1/60, accumulator = 0

Each Frame:
  accumulator += frame_delta_time
  while accumulator >= fixed_dt:
    system.execute(world, fixed_dt)
    accumulator -= fixed_dt
  
  // Remaining accumulator used for interpolation
  alpha = accumulator / fixed_dt
  // Upper layers interpolate between prev_state and current_state
```

### Pattern 4: Tracker → Changed<T> Query
```
Component field write:
  TrackedDescriptor.__set__(obj, value)
    → tracker.mark_dirty(obj, field, old, new)

Query with Changed filter:
  world.for_each(Health, changed=True):
    → For each entity with Health:
      → if tracker.is_dirty(entity.health): yield

End of frame:
  for obj in tracker.all_dirty():
    tracker.mark_clean(obj)
```

### Pattern 5: Session → Foundation Serializer
```
Session.save(path):
  → Foundation Serializer.to_file(session, path, binary=True)
    → For each entity in world:
      → to_dict(component) for each component
      → schema_hash(component_type) embedded for migration
    → Undo/redo stacks serialized
    → Editor state serialized

Session.load(path):
  → Foundation Serializer.from_file(path, binary=True)
    → schema_hash check → auto-migrate if mismatch
    → Reconstruct World with all entities/components
    → Restore undo/redo stacks
```

### Pattern 6: EventLog → Deterministic Replay
```
Recording:
  Each frame:
    set_current_tick(frame_number)
    For each system with @traced:
      → EventLog records: tick, operation, entity, changes, result

Replay:
  For each tick:
    events = event_log.events_at(tick)
    For each event:
      → Re-execute operation with recorded args
      → Verify changes match recorded changes (desync detection)
```

---

## 12. Quick Reference Tables

### Decorator Quick Reference (Core-Specific)

| Decorator | Tier | Requires | Excludes | Core System |
|-----------|------|----------|----------|-------------|
| `@system` | Foundation | — | — | SystemScheduler |
| `@query` | Foundation | — | — | World.query() |
| `@bundle` | Foundation | — | — | World.spawn_bundle() |
| `@relation` | Foundation | @component | — | World hierarchy |
| `@derived` | Foundation | @component | — | World computed components |
| `@parallel` | Scheduling | — | @exclusive | TaskScheduler dispatch |
| `@exclusive` | Scheduling | — | @parallel | SystemScheduler sole access |
| `@fixed` | Scheduling | — | @throttle | Engine accumulator loop |
| `@throttle` | Scheduling | — | @fixed | SystemScheduler rate limit |
| `@after/@before` | Scheduling | — | — | SystemScheduler topo sort |
| `@run_if` | Scheduling | — | — | SystemScheduler conditional |
| `@job` | Scheduling | — | — | TaskScheduler affinity |
| `@chain` | Scheduling | — | — | SystemScheduler pipeline |
| `@deferred` | Scheduling | — | — | CommandBuffer flush |
| `@async_system` | Scheduling | — | — | Fiber/coroutine execution |
| `@deterministic` | Time | — | — | Fixed-point simulation |
| `@time_scale` | Time | — | — | FrameTimer per-layer dt |
| `@pausable` | Time | — | — | SystemScheduler skip |
| `@rewindable` | Time | — | — | Session snapshots |
| `@pooled` | Memory | @component | — | PoolAllocator |
| `@packed` | Memory | — | — | SoA/AoS layout |
| `@aligned` | Memory | — | — | Cache alignment |
| `@budget` | Memory | — | — | MemoryTracker budgets |
| `@profile` | Dev | — | — | System timing |
| `@invariant` | Dev | — | — | World consistency |

### Metaclass → Core System Mapping

| Metaclass | Registry | Core System |
|-----------|---------|-------------|
| ComponentMeta | `_components` | World archetype storage |
| SystemMeta | `_systems` | SystemScheduler |
| ResourceMeta | `_resources` | Engine bootstrap init |
| EventMeta | `_events` | World event bus |
| AssetMeta | `_assets` | (Upper layer) |
| StateMeta | `_states` | (Upper layer) |
| ProtocolMeta | `_protocols` | (Networking layer) |

### Phase 7 Descriptor → Core Use

| Descriptor | Core System | Use Case |
|-----------|------------|----------|
| Immutable | World | Entity handles, archetype IDs |
| Versioned | World | Query cache invalidation |
| Indexed | World | Fast lookup by component value |
| Atomic | TaskScheduler | Concurrent system field access |
| Interpolated | Session | Network snapshot smoothing |
| Sparse | World | Memory optimization for defaults |
| RateLimited | SystemScheduler | Network update frequency |
| Conditional | World | Authority-gated writes |
| Transform | World | Coordinate conversion |
| Expiring | World | TTL-based cache, temp buffs |
| Audit | Session | Debug trail for replay |
| Pooled | MemoryAllocator | Field value pooling |

### Foundation Constant Limits

| Constant | Value | Core Use |
|----------|-------|----------|
| `MAX_UNDO_STACK_SIZE` | 1000 | Session undo cap |
| `MAX_REDO_STACK_SIZE` | 1000 | Session redo cap |
| `DEFAULT_POOL_SIZE` | 1024 | PoolAllocator default |
| `MAX_POOL_SIZE` | 1_000_000 | PoolAllocator ceiling |
| `MAX_TRANSACTION_CHANGES` | 10_000 | Per-frame change limit |
| `MAX_DIRTY_OBJECTS` | 100_000 | Tracker ceiling |
| `MAX_CAUSAL_DEPTH` | 100 | EventLog chain depth |

### Trinity Scheduling Constants

| Constant | Value | Core Use |
|----------|-------|----------|
| `DEFAULT_PHYSICS_HZ` | 60 | @fixed default rate |
| `DEFAULT_CHUNK_SIZE` | 64 | @parallel chunk default |
| `DEFAULT_MIN_BATCH` | 256 | @parallel min batch |
| `DEFAULT_STACK_SIZE` | 65536 | @job stack size |
| `DEFAULT_SYSTEM_PRIORITY` | 0 | System ordering default |
