# Trinity <-> Foundation Integration Reference

A comprehensive guide to how Trinity (definition-time metaprogramming) and
Foundation (runtime infrastructure) connect at every level of the AI Game Engine.

**Implementation Status:** All 10 Trinity specification phases are complete (989 tests passing). This document reflects the fully implemented system.

---

## Table of Contents

- [Overview](#overview)
- [1. The Two-Pillar Architecture](#1-the-two-pillar-architecture)
- [2. Foundation's Six Core Systems + VIPER Extensions](#2-foundations-six-core-systems--viper-extensions)
  - [2.1 Registry](#21-registry-foundationregistrypy)
  - [2.2 Tracker](#22-tracker-foundationtrackerpy)
  - [2.3 EventLog](#23-eventlog-foundationeventlogpy)
  - [2.4 Mirror](#24-mirror-foundationmirrorpy)
  - [2.5 Bridge](#25-bridge-foundationbridgepy)
  - [2.6 ShellLang](#26-shelllang-foundationshelllang)
  - [2.7 Path Utilities](#27-path-utilities-foundationpathspy)
  - [2.8 First-Class Queries](#28-first-class-queries-foundationquerypy)
  - [2.9 ContentStore](#29-contentstore-foundationcontent_storepy)
  - [2.10 DeltaSync](#210-deltasync-foundationdelta_syncpy)
  - [2.11 Computed Provenance](#211-computed-provenance-foundationprovenancepy)
  - [2.12 QueryCacheMirror](#212-querycachemirror-foundationquery_cache_mirrorpy)
  - [2.13 Capability-Based Security](#213-capability-based-security-foundationcapabilitiespy--foundationsecure_shellpy)
  - [2.14 Session Object](#214-session-object-the-image)
- [3. Integration Points: How Trinity Feeds Foundation](#3-integration-points-how-trinity-feeds-foundation)
  - [3.1 Metaclass to Registry (Definition Time)](#31-metaclass---registry-definition-time)
  - [3.1b Metaclass Step Recording](#31b-metaclass-step-recording-_metaclass_steps)
  - [3.1c Metaclass Auto-Install of Descriptors](#31c-metaclass-auto-install-of-descriptors-phase-9)
  - [3.2 Descriptor to Tracker (Runtime)](#32-descriptor---tracker-runtime)
  - [3.3 Decorator to EventLog (Runtime)](#33-decorator---eventlog-runtime)
  - [3.4 Any Object to Mirror (Anytime)](#34-any-object---mirror-anytime)
  - [3.5 Cross-Layer Introspection](#35-cross-layer-introspection-decompose-and-decompose_layered)
  - [3.6 Composition Rule Enforcement](#36-composition-rule-enforcement)
  - [3.7 Programmatic Introspection API](#37-programmatic-introspection-api)
  - [3.8 Decorator Stacks and Foundation](#38-decorator-stacks-and-foundation)
- [4. How Engine Layers Connect Through Foundation](#4-how-engine-layers-connect-through-foundation)
- [5. FlowForge Integration Path](#5-flowforge-integration-path)
- [6. ShellLang and AI Interface](#6-shelllang-and-ai-interface)
- [7. End-to-End Examples](#7-end-to-end-examples)
- [8. Dependency and Import Rules](#8-dependency-and-import-rules)
- [9. Extension Points](#9-extension-points)
- [10. Glossary](#10-glossary)

---

## Overview

The AI Game Engine rests on two metaprogramming pillars:

| Pillar | When | What it owns | Key modules |
|--------|------|-------------|-------------|
| **Trinity** | Definition time | Class creation, field behavior, decorator composition | `trinity/metaclasses/`, `trinity/descriptors/`, `trinity/decorators/` |
| **Foundation** | Runtime | Observation, tracking, queries, interactive access | `foundation/registry.py`, `foundation/tracker.py`, `foundation/eventlog.py`, `foundation/mirror.py`, `foundation/bridge.py`, `foundation/shelllang/` |

The two pillars are coupled through well-defined integration points:

1. **Metaclasses register types** into Foundation's Registry at class creation time.
2. **Descriptors notify** Foundation's Tracker and EventLog when fields change at runtime.
3. **Decorators record** operations into Foundation's EventLog for causal tracing.
4. **Mirror reflects** on Trinity-installed metadata for introspection at any time.
5. **Bridge** is the sole module that joins both pillars for live interop.

A critical architectural rule: **FlowForge accesses Trinity ONLY through Foundation
(never directly)**. This ensures security (Capabilities), consistency (Tracker), and
observation (EventLog) are never bypassed.

---

## 1. The Two-Pillar Architecture

```
                         METAPROGRAMMING LAYER
    =====================================================================

    TRINITY (Definition Time)             FOUNDATION (Runtime)
    =========================             ======================

    +-----------------------+             +-----------------------+
    | Metaclasses (8)       |             | Registry              |
    | - EngineMeta          |             | - Unified type lookup |
    | - ComponentMeta       |  ---------> | - Instance tracking   |
    | - SystemMeta          |  registers  +-----------------------+
    | - ResourceMeta        |
    | - EventMeta           |             +-----------------------+
    | - AssetMeta           |             | Tracker               |
    | - ProtocolMeta        |             | - Dirty flags         |
    | - StateMeta           |             | - Undo/redo           |
    +-----------------------+             | - Change subscriptions|
                                          +-----------------------+
    +-----------------------+
    | Descriptors           |  ---------> +-----------------------+
    | - StorageDescriptor   |  notifies   | EventLog              |
    | - TrackedDescriptor   |             | - Causal chains       |
    | - ValidatedDescriptor |             | - Operation history   |
    | - NetworkedDescriptor |             +-----------------------+
    | - ObservableDescriptor|
    | - SerializableDesc.   |             +-----------------------+
    | - ProfiledDescriptor  |             | Mirror                |
    | - LoggedDescriptor    |             | - Deep introspection  |
    | + 12 Phase 7 descs    |             | - Schema hashing      |
    +-----------------------+             +-----------------------+

    +-----------------------+             +-----------------------+
    | Decorators            |             | Bridge                |
    | - @component          |             | - Trinity <-> Shell   |
    | - @system             |  <--------> | - Bidirectional sync  |
    | - @resource           |  sole link  +-----------------------+
    | - @event              |
    | - @traced             |             +-----------------------+
    | - @tracked            |             | ShellLang             |
    | - @networked          |             | - 5 primitives        |
    | - @pooled, @budgeted  |             | - AI Interface        |
    +-----------------------+             | - Interactive Shell   |
                                          +-----------------------+
```

**Separation principle.** Trinity owns *class creation and field behavior*. It
decides what fields a component has, how they are validated, whether they are
tracked, and how they are stored. Foundation owns *runtime observation, tracking,
queries, and interactive access*. It knows which types exist, which instances are
alive, which fields are dirty, what operations happened, and how to reflect on
any object.

**Ops-first architecture.** Trinity uses an Ops-first architecture where every
decorator decomposes into 7 primitive Ops: TAG, HOOK, REGISTER, DESCRIBE, TRACK,
VALIDATE, and INTERCEPT. These 7 Ops are the common language between all three
Trinity layers (decorators, metaclasses, descriptors) AND Foundation. A decorator
is not self-contained code -- it is a named list of Steps, each being one of these
7 Ops. The Ops do the real work; decorators are just configuration. This means
`decompose(Health)` returns every operation at every layer, making the entire
system transparent through a single API.

Neither pillar depends on the other at import time. Trinity metaclasses wrap
Foundation calls in `try/except ImportError` blocks so that Trinity can run
standalone for testing, code generation, or static analysis. Foundation's core
modules (`registry.py`, `tracker.py`, `eventlog.py`, `mirror.py`) never import
from `trinity/`. The sole coupling point is `foundation/bridge.py`, which imports
from both pillars to provide live integration.

---

## 2. Foundation's Six Core Systems + VIPER Extensions

Foundation has grown beyond the original 6 core systems. The core 6 (Registry,
Tracker, EventLog, Mirror, Bridge, ShellLang) remain the backbone, but VIPER
extensions (Sections 2.7-2.13) add path utilities, first-class queries,
content-addressable storage, delta synchronization, computed provenance, query
cache introspection, and capability-based security. Section 2.14 covers the
Session object that wraps the entire engine state.

### 2.1 Registry (`foundation/registry.py`)

The Registry is the central type directory -- a singleton `registry` that provides
unified access to every type the engine knows about.

**API surface:**

| Method | Signature | Purpose |
|--------|-----------|---------|
| `register()` | `(cls, name=None, track_instances=False)` | Add a type to the registry |
| `unregister()` | `(cls)` | Remove a type |
| `get()` | `(name) -> type` | Look up a type by its registered name |
| `get_name()` | `(cls) -> str` | Get the registered name for a type |
| `is_registered()` | `(cls) -> bool` | Check if a type is known |
| `all_types()` | `() -> list[type]` | Return every registered type |
| `subclasses()` | `(base) -> list[type]` | Return all registered subclasses of a base |
| `types_with_decorator()` | `(decorator_name) -> list[type]` | Find types with a specific decorator |
| `types_where()` | `(predicate) -> list[type]` | General predicate-based lookup |
| `instances()` | `(cls) -> Iterator[object]` | Iterate live instances (if `track_instances=True`) |
| `instance_count()` | `(cls) -> int` | Count live instances |
| `set_metadata()` | `(cls, key, value)` | Attach arbitrary metadata to a type |
| `get_metadata()` | `(cls, key) -> Any` | Retrieve metadata |
| `describe()` | `(cls) -> str` | Human-readable description using Mirror |

**What gets registered.** Every class created by any of the 8 metaclasses calls
`registry.register()` during `__new__`. For ComponentMeta this happens explicitly
in step 6 of `__new__`:

```python
# ComponentMeta.__new__, step 6 (from trinity/metaclasses/component_meta.py)
@classmethod
def _register_with_foundation(mcs, cls: type) -> None:
    try:
        from foundation import registry
        if not registry.is_registered(cls):
            registry.register(cls, name=cls._component_name, track_instances=True)
    except ImportError:
        pass  # Foundation not available
```

**Instance tracking.** When `track_instances=True`, the Registry wraps `__init__`
to add each new instance to a `WeakSet`. This lets Foundation answer questions like
"how many Health components are alive?" without Trinity needing to maintain that
bookkeeping. Instances are held weakly so garbage collection still works.

```python
# How the Registry wraps __init__ (from foundation/registry.py)
def _wrap_init(self, cls: type) -> None:
    original_init = cls.__init__
    weak_set = self._instances[cls]

    @functools.wraps(original_init)
    def tracking_init(self_obj, *args, **kwargs):
        original_init(self_obj, *args, **kwargs)
        weak_set.add(self_obj)

    cls.__init__ = tracking_init
```

**How it differs from Trinity's internal registries.** Each metaclass maintains its
own `_registry` dict keyed by integer ID. These are per-metaclass and use integer
keys for fast lookup during gameplay. Foundation's Registry is *unified* across all
metaclass families, uses string names for human/tool access, and adds metadata and
instance tracking that Trinity does not provide.

| Feature | Trinity per-metaclass | Foundation Registry |
|---------|----------------------|---------------------|
| Scope | One metaclass family | All engine types |
| Key type | Integer ID | String name |
| Instance tracking | No | Yes (WeakSet) |
| Metadata slots | No | Yes (arbitrary dict) |
| Thread safety | Per-metaclass lock | Single RLock |
| Mirror integration | No | Yes (`describe()`) |

---

### 2.2 Tracker (`foundation/tracker.py`)

The Tracker is a centralized change-tracking system -- a singleton `tracker` that
records every field mutation, provides dirty flags, fires subscriptions, and
supports transactional undo/redo.

**API surface:**

| Method | Purpose |
|--------|---------|
| `mark_dirty(obj, field, old, new)` | Record a field change |
| `is_dirty(obj) -> bool` | Check if any field is dirty |
| `dirty_fields(obj) -> set[str]` | Get dirty field names |
| `all_dirty() -> list[Any]` | Get all objects with dirty fields |
| `mark_clean(obj)` | Clear dirty flags |
| `on_change(target, callback)` | Subscribe to changes |
| `off_change(callback)` | Unsubscribe |
| `begin_transaction(name)` | Start an atomic group |
| `commit_transaction()` | Commit the group |
| `rollback_transaction()` | Revert all changes in the group |
| `undo() -> bool` | Undo the last transaction |
| `redo() -> bool` | Redo the last undone transaction |

**Fed by TrackedDescriptor.** The Tracker does not poll -- it is *pushed* data by
Trinity's `TrackedDescriptor.post_set()`, which calls `tracker.mark_dirty()` on
every value change. This is the primary runtime integration seam between Trinity
and Foundation.

**Three subscription scopes:**

```python
from foundation.tracker import tracker

# Global: fires for ANY field change on ANY object
tracker.on_change(callback=lambda obj, fld, old, new: print(f"Changed: {fld}"))

# Per-object: fires only for changes on `player`
tracker.on_change(player, callback=lambda obj, fld, old, new: ...)

# Per-type: fires for changes on ANY instance of the Health class
tracker.on_change(Health, callback=lambda obj, fld, old, new: ...)
```

**Transaction support.** The Tracker supports atomic grouping of changes. Within
a transaction, changes to the same field on the same object are coalesced (only
the first old value and last new value are kept). Rollback reverts every change
in reverse order.

```python
tracker.begin_transaction("apply_damage")
player.health = 80   # mark_dirty called
player.shield = 0    # mark_dirty called
tracker.commit_transaction()

# Later:
tracker.undo()   # reverts both health and shield atomically
tracker.redo()   # re-applies both
```

**Consumers:**

| Consumer | How it uses Tracker |
|----------|-------------------|
| Networking | `all_dirty()` to find objects needing replication |
| Serialization | `dirty_fields()` to save only changed data |
| UI data binding | `on_change(obj, cb)` to re-render when model changes |
| Editor undo/redo | `undo()` / `redo()` for user-facing undo |
| Inspector | `on_change(type, cb)` to refresh property panels |

---

### 2.3 EventLog (`foundation/eventlog.py`)

The EventLog records the history of *operations* (not just field changes) with
full causal chain tracking. While the Tracker knows *what* changed, the EventLog
knows *why* it changed, *who* caused it, and *what sequence* of operations led
to the current state.

**Core data structures:**

```python
@dataclass
class Change:
    entity: int       # Entity ID
    field: str        # Field name
    old_value: Any
    new_value: Any

@dataclass
class Event:
    tick: int                         # Game tick
    operation: str                    # e.g. "DamageSystem.update"
    operation_args: dict[str, Any]    # Arguments passed
    entity: Optional[int]            # Entity this was called on
    changes: list[Change]            # Field changes caused by this op
    result: Any                       # Return value
    error: Optional[Exception]       # Exception if failed
    immediate_parent: Optional[str]  # Calling operation
    immediate_parent_entity: Optional[int]
    root_cause: Optional[str]        # First entity-bound operation
    root_cause_entity: Optional[int]
    depth: int                        # Nesting depth
```

**Indexes.** Events are indexed four ways for efficient querying:

| Index | Lookup |
|-------|--------|
| `_by_tick` | All events at a specific game tick |
| `_by_entity` | All events for a specific entity |
| `_by_operation` | All events of a specific operation type |
| `_by_root_cause` | All events caused by a specific entity |

**Causal chain tracking.** The EventLog uses Python `ContextVar` objects to
maintain a call stack:

```
DamageSystem.update()           depth=0, root_cause=None
  -> Player.take_damage()       depth=1, root_cause="Player.take_damage", entity=42
    -> HealthComponent.apply()  depth=2, root_cause="Player.take_damage", entity=42
      -> field write (hp=80)    Change added to current Event
```

The first entity-bound operation in a call chain becomes the `root_cause`. This
means a system's `update()` (which has no entity) is pass-through -- the causal
root is the entity method it calls. This is critical for AI agent reasoning:
"Why did the player die?" -> trace `root_cause_entity` back through the chain.

**Connection to Trinity:** Two paths:

1. **@traced decorator** wraps system `update()` methods to create Event records.
2. **TrackedDescriptor.post_set()** calls `add_change_to_current_event()` to
   attach field Changes to the currently executing Event.

---

### 2.4 Mirror (`foundation/mirror.py`)

Mirror provides uniform reflection for any Python object, inspired by Smalltalk's
mirror-based architecture. It reads Trinity-installed metadata without importing
Trinity.

**Entry point:**

```python
from foundation.mirror import mirror, schema_hash

m = mirror(player)         # Returns ObjectMirror
m = mirror(Health)         # Returns ClassMirror
h = schema_hash(Health)    # Returns 16-char hex hash
```

**ObjectMirror API:**

| Property/Method | Returns |
|----------------|---------|
| `type_name` | Class name as string |
| `type_class` | The actual class object |
| `fields` | `dict[str, FieldInfo]` |
| `methods` | `dict[str, MethodInfo]` |
| `get(name)` | Field value via `getattr` |
| `set(name, value)` | Field value via `setattr` |
| `has(name)` | `hasattr` check |
| `to_dict()` | All fields as a plain dict |
| `get_path(path)` | Dotted path access: `"inventory.items[0].damage"` |
| `set_path(path, value)` | Dotted path mutation |
| `describe()` | Human-readable string |

**ClassMirror** has the same interface but operates on the type rather than an
instance, showing field types and defaults rather than live values.

**FieldInfo dataclass:**

```python
@dataclass(frozen=True)
class FieldInfo:
    name: str
    type: Optional[type]        # Base type (unwrapped from Annotated)
    has_default: bool
    default: Any
    metadata: dict[str, Any]    # Extracted from Annotated[T, {...}]
```

**Schema hashing.** `schema_hash(cls)` generates a stable SHA-256 hash (truncated
to 16 hex characters) of a class's schema. This is used for:

- Hot reload detection (hash changed = migration needed)
- Versioned save files (schema hash embedded in header)
- Network protocol compatibility (both sides compare hashes)

**How Mirror reads Trinity metadata.** Mirror does not import Trinity. It reads
standard Python attributes that Trinity's metaclasses install:

- `__annotations__` for field names and types
- `__dict__` for default values
- `__slots__` for slot-based fields
- `get_type_hints(cls, include_extras=True)` for `Annotated` metadata

Because Trinity metaclasses set these standard Python attributes, Mirror can
reflect on Trinity objects without any coupling.

---

### 2.5 Bridge (`foundation/bridge.py`)

The Bridge is the **sole module that imports from both Trinity and Foundation**.
It provides bidirectional mapping between Trinity component instances and
ShellLang entities.

**Factory functions:**

```python
from foundation.bridge import (
    get_trinity_registry,       # -> dict[str, Type]
    create_world_from_trinity,  # -> World (pre-populated)
    create_ai_interface,        # -> AIInterface (connected)
    create_shell,               # -> Shell (connected)
)
```

**get_trinity_registry()** imports `ComponentMeta` and returns all registered
components. This is the only function in Foundation that directly imports from
`trinity.metaclasses`:

```python
def get_trinity_registry() -> Dict[str, Type]:
    try:
        from trinity.metaclasses.component_meta import ComponentMeta
        return {cls.__name__: cls for cls in ComponentMeta.all_components()}
    except ImportError:
        return {}
```

**TrinityWorldAdapter** provides bidirectional sync:

```python
adapter = TrinityWorldAdapter()  # Creates World from Trinity registry

# Trinity instance -> ShellLang entity
entity = adapter.add_instance(health_component)

# ShellLang entity -> Trinity instance
instance = adapter.get_instance(entity, Health)

# Sync all tracked instances from Foundation Registry
adapter.sync_from_foundation_registry()
```

The adapter maintains two mapping dicts:

| Mapping | Direction | Key | Value |
|---------|-----------|-----|-------|
| `_instance_to_entity` | Trinity -> Shell | `id(instance)` | `Entity` |
| `_entity_to_instances` | Shell -> Trinity | `entity.id` | `{comp_name: instance}` |

---

### 2.6 ShellLang (`foundation/shelllang/`)

ShellLang is a minimal ECS language built on 5 semantic primitives. It is
Foundation's interactive and programmatic access layer.

**The 5 primitives:**

| Primitive | Meaning | Implementation |
|-----------|---------|---------------|
| **ENTITY** | `uint64` identifier for game objects | `Entity(id: int)` |
| **COMPONENT** | Typed data attached to an entity | Any Python class |
| **QUERY** | Entity predicate -> `[Entity]` | `World.query(*Cs)` |
| **MUTATE** | `(entity, field, value)` -> tracked change | `World.set(e, C, field, value)` |
| **SNAPSHOT** | Frozen world state | `World.snap()` / `World.restore()` |

**Module structure:**

| Module | Purpose |
|--------|---------|
| `core.py` | `Entity`, `World`, `Change`, `Snapshot` -- the primitives |
| `sugar.py` | `EntityProxy`, `QueryResult`, `TypeQuery`, `TimeManager` -- ergonomic API |
| `ai.py` | `AIInterface` -- structured JSON commands for AI agents |
| `repl.py` | `Shell`, `Feedback` -- interactive REPL for humans |

**sugar.py** provides chainable queries:

```python
# Find all enemies with low health near the player
Enemys.all.where(lambda e: e.health.current < 20).near(player, 10.0)

# Bulk mutation
Enemys.all.where(lambda e: e.health.current <= 0).destroy()
```

**ai.py** provides structured commands:

```python
ai.execute({
    "op": "query",
    "components": ["Health"],
    "where": {"Health.hp": {"<": 20}},
    "limit": 50
})

ai.execute({
    "op": "set",
    "entity": 42,
    "component": "Health",
    "field": "hp",
    "value": 100
})
```

**How ShellLang connects to Trinity:** Bridge populates the World with
Trinity-registered types. MUTATE operations flow through `World.set()`, which
calls `setattr()` on the component. If that component has a `TrackedDescriptor`
chain, the write triggers `post_set()` -> `tracker.mark_dirty()` -> Foundation
Tracker picks it up. The data flows:

```
ShellLang MUTATE -> World.set() -> setattr(component, field, value)
                                       |
                                  TrackedDescriptor.__set__()
                                       |
                                  tracker.mark_dirty(obj, field, old, new)
                                       |
                              Foundation Tracker fires subscriptions
```

---

### 2.7 Path Utilities (`foundation/paths.py`)

Path utilities provide dotted-path access to nested object fields.

**API surface:**

| Function | Purpose |
|----------|---------|
| `parse_path(path_str)` | Parse `"entities[0].health.hp"` into path segments |
| `get_path(obj, path)` | Read a value at a dotted path |
| `set_path(obj, path, value)` | Write a value at a dotted path |

Supports array indices: `"inventory.items[0].damage"` resolves through dicts,
lists, and object attributes.

**Used by:** `Mirror.get_path()` / `Mirror.set_path()`, `Inspector.edit()`,
Shell commands for nested access.

---

### 2.8 First-Class Queries (`foundation/query.py`)

Query objects with content-based identity hashing, automatic invalidation, and
algebraic composition.

**Key features:**

| Feature | Description |
|---------|-------------|
| Content-based identity | `Query.hash()` produces a stable hash from query structure |
| QueryCache | Automatic invalidation via Tracker subscriptions |
| Query algebra | Union (`\|`), intersection (`&`), difference (`-`) |
| TrackedQueryCache | Auto-invalidates when tracked fields change |
| Reactive subscriptions | `query.subscribe(on_add=..., on_remove=...)` |

```python
q = Query(Enemy, Health).near(player, 10).where(health__lt=50)
q.hash()            # stable, content-based
q1 & q2             # intersection
q1 | q2             # union
q1 - q2             # difference
```

---

### 2.9 ContentStore (`foundation/content_store.py`)

Content-addressable storage with structural sharing.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `ContentStore` | Store and retrieve content by hash |
| `MemoryBackend` | In-memory storage |
| `FileBackend` | Filesystem-backed storage |
| `ContentDiffer` | Hash-based diffing -- O(differences) not O(size) |

**Used for:** Snapshot storage, asset deduplication, undo history. Identical
content stored once regardless of how many references point to it.

---

### 2.10 DeltaSync (`foundation/delta_sync.py`)

Minimal change patch computation between states.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `DeltaPatch` | Represents a minimal diff between two states |
| `DeltaSync` | Computes and applies patches |

Performs recursive dict diffing to produce the smallest possible patch.

**Used for:** Network replication (send only what changed), save file compression,
undo deltas.

---

### 2.11 Computed Provenance (`foundation/provenance.py`)

Answer "why is this computed value X?" with full derivation trees.

**Key features:**

| Feature | Description |
|---------|-------------|
| `@track_provenance` | Decorator for automatic dependency tracking |
| `derivation_tree(obj, field)` | Full tree with cycle detection |
| Input summary capture | Records what inputs fed a computation |
| Descriptor integration | Reads tracked by descriptors feed the provenance graph |

**Used for:** Computed fields, reactive recalculation, debug "why did this
change?" queries.

---

### 2.12 QueryCacheMirror (`foundation/query_cache_mirror.py`)

Introspection into query cache state.

**Features:** Registered query enumeration, cache hit/miss statistics,
invalidation tracking. Useful for performance tuning of query-heavy systems.

---

### 2.13 Capability-Based Security (`foundation/capabilities.py` + `foundation/secure_shell.py`)

Fine-grained permission control for engine access.

**Key features:**

| Feature | Description |
|---------|-------------|
| Capability flags | `READ`, `WRITE`, `EXECUTE`, `ADMIN` (and more) |
| `CapabilitySet` | Immutable, composable permission sets |
| `SecureContext` | Context manager for scoped permissions |
| `@require_capability` | Decorator that checks permissions before execution |
| Factory functions | `create_readonly_shell()`, `create_sandbox_shell()`, `create_full_shell()` |

```python
from foundation.capabilities import Capability, CapabilitySet
from foundation.secure_shell import create_readonly_shell, create_sandbox_shell

readonly = create_readonly_shell()    # Can read, cannot write
sandbox = create_sandbox_shell()      # Limited permissions for mods
full = create_full_shell()            # All permissions
```

**Used for:** FlowForge sandboxing, mod safety, AI agent permission control.

---

### 2.14 Session Object ("The Image")

The Session wraps the entire engine state into a single serializable object:

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

Session enables: save/load entire editor state, crash recovery, collaborative
editing, replay from any point. Foundation's Serializer handles persistence;
ContentStore provides structural sharing for efficient storage.

---

## 3. Integration Points: How Trinity Feeds Foundation

### 3.1 Metaclass -> Registry (Definition Time)

Every metaclass registers its newly-created class with Foundation's Registry.
This happens during `__new__`, which runs once when the class statement is
executed by Python.

**ComponentMeta flow (the most detailed):**

```
class Health(Component):     # Python triggers ComponentMeta.__new__
    hp: float = 100.0        #
    max_hp: float = 100.0    #
                              #   Step 1: Generate unique _component_id
                              #   Step 2: Process fields -> _field_types, _field_defaults
                              #   Step 3: Install descriptors -> TrackedDescriptor chains
                              #   Step 4: Validate component rules
                              #   Step 5: Register in ComponentMeta._registry (int key)
                              #   Step 6: _register_with_foundation(cls)
                              #           -> registry.register(cls,
                              #                name="mymodule.Health",
                              #                track_instances=True)
                              #   Step 7: Initialize pool/budget if configured
```

**What each metaclass passes to Foundation:**

| Metaclass | Registered Name | track_instances | Additional Metadata |
|-----------|----------------|-----------------|-------------------|
| `ComponentMeta` | `module.ClassName` | `True` | Component ID, field types, pool config |
| `SystemMeta` | `module.ClassName` | `False` | Phase, dependencies, reads/writes |
| `ResourceMeta` | `module.ClassName` | `False` | Priority, dependencies, singleton flag |
| `EventMeta` | `module.ClassName` | `False` | Event fields, channels, priority |
| `AssetMeta` | `module.ClassName` | `False` | Asset type, loader info, format |
| `ProtocolMeta` | `module.ClassName` | `False` | Network metadata, serialization |
| `StateMeta` | `module.ClassName` | `False` | Valid transitions, initial state |

All metaclasses inherit from `EngineMeta`, which maintains its own
`_all_engine_types` dict for debug introspection. Foundation's Registry
supplements this with instance tracking, metadata slots, and unified lookup.

**The try/except pattern** ensures Trinity works standalone:

```python
# Every metaclass uses this pattern:
try:
    from foundation import registry
    if not registry.is_registered(cls):
        registry.register(cls, name=cls._component_name, track_instances=True)
except ImportError:
    pass  # Foundation not available -- skip integration
```

---

### 3.1b Metaclass Step Recording (`_metaclass_steps`)

All 8 metaclasses now record their `__new__` operations as Step objects in
`cls._metaclass_steps`. This makes the entire metaclass layer visible to
introspection:

```python
# After ComponentMeta.__new__ completes, cls._metaclass_steps contains:
[
    Step(Op.TAG, {"component_id": 1, "component_name": "Health"}),
    Step(Op.DESCRIBE, {"field": "hp", "type": "float", "default": 100.0}),
    Step(Op.DESCRIBE, {"field": "max_hp", "type": "float", "default": 100.0}),
    Step(Op.INTERCEPT, {"field": "hp", "descriptor": "TrackedDescriptor"}),
    Step(Op.VALIDATE, {"component": "Health", "rule": "no_methods"}),
    Step(Op.REGISTER, {"registry": "foundation", "track_instances": True}),
]
```

**What each metaclass records:**

| Metaclass | Steps Recorded |
|-----------|---------------|
| EngineMeta | TAG(engine_type) |
| ComponentMeta | TAG(id), DESCRIBE(fields), INTERCEPT(descriptors), VALIDATE, REGISTER |
| SystemMeta | TAG(id, phase), DESCRIBE(dependencies), TAG(exclusions), VALIDATE, REGISTER |
| ResourceMeta | TAG(singleton), DESCRIBE(fields), REGISTER |
| EventMeta | TAG(event_id, priority), DESCRIBE(payload_fields), VALIDATE(payload), HOOK(handlers), TAG(bubbles/cancellable), REGISTER |
| StateMeta | TAG(state_id), DESCRIBE(valid_transitions), VALIDATE(transitions), REGISTER |
| AssetMeta | TAG(asset_id, extensions), DESCRIBE(fields), TAG(loader), VALIDATE(extensions), HOOK(on_load/on_unload), REGISTER |
| ProtocolMeta | TAG(protocol_id, version), DESCRIBE(messages), VALIDATE(messages), REGISTER |

These steps feed into `decompose()` for cross-layer introspection (see
Section 3.5).

---

### 3.1c Metaclass Auto-Install of Descriptors (Phase 9)

`ComponentMeta._install_descriptors()` reads `_applied_steps` from decorators
and **automatically installs the corresponding descriptors**:

```
@tracked decorator    -> Step(Op.TRACK)             -> ComponentMeta reads it -> installs TrackedDescriptor
@validated decorator  -> Step(Op.VALIDATE)           -> ComponentMeta reads it -> installs ValidatedDescriptor
@networked decorator  -> Step(Op.INTERCEPT, network=True) -> installs NetworkedDescriptor
```

This means users never manually create descriptors. The flow is:

1. User applies `@tracked` decorator
2. Decorator records `Step(Op.TRACK, ...)`
3. `ComponentMeta.__new__` reads all Steps
4. For each TRACK step -> installs TrackedDescriptor on the relevant fields
5. For each VALIDATE step -> installs ValidatedDescriptor
6. Descriptor chain is built via DescriptorComposer

---

### 3.2 Descriptor -> Tracker (Runtime)

This is the most frequently-exercised integration point. Every time a tracked
field is written, the descriptor chain fires and Foundation's Tracker is notified.

**Complete flow for `player.health = 50`:**

```
  1. Python calls TrackedDescriptor.__set__(player, 50)
          |
  2. BaseDescriptor.__set__ calls self.pre_set(player, 50)
     (validation chain runs -- ValidatedDescriptor checks range, type, etc.)
          |
  3. Value stored via inner descriptor:
     inner.__set__(player, 50)
       -> StorageDescriptor stores in player.__dict__["health"] = 50
          |
  4. BaseDescriptor.__set__ calls self.post_set(player, 50, old_value=100)
          |
  5. TrackedDescriptor.post_set():
     a) Adds "health" to player._dirty_fields set
     b) Calls _notify_foundation_tracker(player, 100, 50):
          |
  6.     from foundation import tracker
         tracker.mark_dirty(player, "health", 100, 50)
          |
  7. Tracker.mark_dirty():
     a) Adds (weakref(player), {"health"}) to _dirty dict
     b) Creates Change record; appends to _undo stack (or active transaction)
     c) Calls _notify(player, "health", 100, 50):
          |
  8. Tracker._notify() fires callbacks:
     - Global callbacks (all subscribers)
     - Per-object callbacks (subscribers to this player)
     - Per-type callbacks (subscribers to Health class)
          |
  9. TrackedDescriptor.post_set() also calls _notify_eventlog():
     a) Creates eventlog.Change(entity=player.id, field="health", old=100, new=50)
     b) Calls add_change_to_current_event(change)
     c) If inside a @traced context, the Change is appended to the current Event
```

**Which descriptors feed which Foundation systems:**

**Original Descriptors -> Foundation:**

| Descriptor | Foundation System | What Flows |
|-----------|------------------|------------|
| StorageDescriptor | (internal) | Raw field storage |
| TrackedDescriptor | Tracker | mark_dirty(obj, field, old, new) |
| ValidatedDescriptor | (internal) | Constraint enforcement |
| RangeDescriptor | (internal) | Numeric clamping |
| ObservableDescriptor | Tracker | on_change callbacks |
| NetworkedDescriptor | Tracker | Replication dirty flags |
| SerializableDescriptor | Mirror | Schema info for serialization |
| TransientDescriptor | Mirror | Excluded from serialization |
| MigratedDescriptor | Mirror + Migrations | Schema version + migration path |
| EncryptedDescriptor | (internal) | Encrypt/decrypt on get/set |
| CachedDescriptor | (internal) | TTL-based value caching |
| ComputedDescriptor | Provenance | Dependency-tracked computation |
| ProfiledDescriptor | EventLog | Access timing metrics |
| LoggedDescriptor | EventLog | Access audit trail |
| WatchedDescriptor | Inspector | Breakpoint on access |
| LazyDescriptor | (internal) | Deferred initialization |
| AsyncLoadDescriptor | (internal) | Async value loading |

**Phase 7 Descriptors -> Foundation:**

| Descriptor | Foundation System | What Flows |
|-----------|------------------|------------|
| ImmutableDescriptor | (internal) | Deny writes after __init__ |
| VersionedDescriptor | Tracker + EventLog | Version counter + history per field |
| IndexedDescriptor | Registry | Value-based index for fast lookup |
| AtomicDescriptor | (internal) | Thread-safe CAS operations |
| InterpolatedDescriptor | Tracker | Smooth interpolation between values |
| SparseDescriptor | (internal) | Only store non-default values |
| RateLimitedDescriptor | (internal) | Cap write frequency |
| ConditionalDescriptor | (internal) | Predicate-gated writes |
| TransformDescriptor | (internal) | Read/write value transforms |
| ExpiringDescriptor | Tracker | TTL-based value expiration with notifications |
| AuditDescriptor | EventLog | Full access audit trail with timestamps |
| PooledDescriptor | Registry | Object pool for field values |

**descriptor_steps():** Every descriptor implements a `descriptor_steps` property
returning the Ops it performs:

```python
class TrackedDescriptor(BaseDescriptor[T]):
    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.TRACK, {"field": self.name, "dirty_mask": True}),
            Step(Op.HOOK, {"event": "on_change", "field": self.name}),
        ]
```

These steps feed into `decompose()` for cross-layer introspection. When you call
`decompose(Health)`, it collects steps from decorators, metaclass, AND every
descriptor in every field's chain.

**Annotated Field Syntax (Preferred):**

The recommended way to declare descriptor-enhanced fields:

```python
from typing import Annotated

@component
class Health(Component):
    # Annotated syntax -- descriptors declared inline
    current: Annotated[float, Tracked(), Range(0, 100), Networked(interpolate=True)] = 100.0
    max_hp: Annotated[float, Tracked(), Validated(gt=0)] = 100.0

    # Equivalent to manual installation, but metaclass handles it automatically
```

`ComponentMeta._process_fields()` reads `Annotated` metadata and installs
descriptors via `DescriptorComposer.compose()`. This replaces manual descriptor
chain construction.

**Important:** NetworkedDescriptor and ObservableDescriptor do not themselves call
`tracker.mark_dirty()`. They rely on TrackedDescriptor being present *inner* in the
chain. The standard chain for a networked tracked component is:

```
NetworkedDescriptor  (outermost -- queues network updates)
  -> TrackedDescriptor  (marks dirty, notifies Foundation)
    -> ValidatedDescriptor  (validates value)
      -> StorageDescriptor  (innermost -- stores in __dict__)
```

This chain is composed by `ComponentMeta._install_descriptors()`:

```python
# From trinity/metaclasses/component_meta.py, simplified
descriptors = []
descriptors.append(StorageDescriptor(field_type=ft, default=default))

if field_name in validation_rules:
    descriptors.append(ValidatedDescriptor(...))

if track_changes:
    descriptors.append(TrackedDescriptor(field_type=ft, field_offset=offset))

if network_config is not None:
    descriptors.append(NetworkedDescriptor(field_type=ft, authority=...))

# Compose from innermost to outermost
descriptor = DescriptorComposer.compose(*reversed(descriptors))
```

---

### 3.3 Decorator -> EventLog (Runtime)

The `@traced` decorator wraps any callable to automatically record Events in the
EventLog. It is typically applied to system `update()` methods and entity methods.

**How @traced works:**

```python
@traced
def take_damage(self, amount: int) -> None:
    self.health -= amount
```

When `take_damage` is called:

```
  1. @traced wrapper extracts entity_id from self.id (if present)
  2. Gets operation name from fn.__qualname__ ("Player.take_damage")
  3. Reads ContextVars for current causal chain:
     - _root_cause: first entity-bound op in the chain
     - _immediate_parent: calling operation
     - _depth: nesting level
  4. Creates Event with all context
  5. Sets ContextVars for nested calls:
     - _current_event = this Event (so TrackedDescriptor can attach Changes)
     - _root_cause = this op (if first entity-bound op)
     - _immediate_parent = this op
     - _depth += 1
  6. Calls the actual function
  7. Records result or exception on Event
  8. Calls _event_log.record(event) to index by tick/entity/operation/root_cause
  9. Resets ContextVars to previous state
```

**Causal chain example:**

```python
@system(phase="gameplay")
@traced
class DamageSystem(System):
    def update(self, dt):              # depth=0, entity=None, root=None
        for entity in self.query(Health):
            if should_take_damage(entity):
                entity.take_damage(10)  # depth=1, entity=42, root="take_damage"
                                        #   -> self.health -= 10
                                        #     TrackedDescriptor.post_set
                                        #       -> Change added to current Event
```

The Event for `take_damage` will have:
- `operation`: `"Player.take_damage"`
- `entity`: `42`
- `immediate_parent`: `"DamageSystem.update"`
- `root_cause`: `"Player.take_damage"` (first entity-bound op)
- `changes`: `[Change(entity=42, field="health", old=100, new=90)]`

---

### 3.4 Any Object -> Mirror (Anytime)

Mirror provides reflection without requiring any integration setup. It works
on any Python object by reading standard attributes.

**How `mirror(player)` works with Trinity objects:**

```
  1. mirror(player) creates ObjectMirror(player)
  2. ObjectMirror.fields property calls _collect_fields(type(player), player)
  3. _collect_fields:
     a) Calls get_type_hints(cls, include_extras=True)
        -> reads __annotations__ set by ComponentMeta._process_fields
     b) For dataclass-based classes: reads dataclasses.fields()
     c) For annotation-based classes: iterates __annotations__
     d) For each field, extracts:
        - Type (unwrapped from Annotated[T, ...])
        - Default value (from __dict__ or class attrs)
        - Metadata (from Annotated[T, {"range": (0, 100)}])
     e) Also reads __slots__ entries
     f) Also reads instance __dict__ for dynamic attributes
  4. Returns dict[str, FieldInfo] with full provenance
```

**Mirror does NOT read descriptor chains directly.** It reads the standard Python
metadata that Trinity metaclasses install. This is intentional -- Mirror should
work on any Python object, not just Trinity objects. To get descriptor chain
information, use the descriptor's own `get_chain()` and `get_metadata()` methods.

**Used by:**

| Consumer | How it uses Mirror |
|----------|-------------------|
| Inspector (`foundation/inspector.py`) | `FieldsView` renders editable fields via Mirror |
| Registry | `describe(cls)` uses Mirror for type description |
| Serializer | Reads field types and defaults for serialization schema |
| FlowForge | Reflects on types to render property editors |
| Schema hashing | `schema_hash(cls)` generates migration-detection hashes |

---

### 3.5 Cross-Layer Introspection: decompose() and decompose_layered()

The unified introspection API collects Steps from ALL three Trinity layers:

```python
from trinity.decorators.ops import decompose, decompose_layered

# Flat list of every Step that built this class
steps = decompose(Health)
# Returns: [TAG(component=True), REGISTER(ecs_core), TAG(component_id=1),
#           DESCRIBE(field=hp), TRACK(field=hp), VALIDATE(range=0..100), ...]

# Grouped by layer
layered = decompose_layered(Health)
# {
#   "decorator_steps": [Step(Op.TAG, {component: True}), Step(Op.REGISTER, {ecs_core})],
#   "metaclass_steps": [Step(Op.TAG, {component_id: 1}), Step(Op.DESCRIBE, {field: "hp"}), ...],
#   "descriptor_steps": {
#       "hp": [Step(Op.TRACK, {dirty_mask: True}), Step(Op.VALIDATE, {range: [0, 100]})],
#       "max_hp": [Step(Op.TRACK, {dirty_mask: True})],
#   }
# }
```

`expand(Health)` prints a human-readable layered trace:

```
Health
+-- Decorators:
|   +-- TAG(component=True)
|   +-- REGISTER(target=ecs_core)
+-- Metaclass (ComponentMeta):
|   +-- TAG(component_id=1, component_name="Health")
|   +-- DESCRIBE(field="hp", type=float, default=100.0)
|   +-- DESCRIBE(field="max_hp", type=float, default=100.0)
|   +-- INTERCEPT(field="hp", descriptor=TrackedDescriptor)
|   +-- VALIDATE(component="Health")
|   +-- REGISTER(registry="foundation")
+-- Descriptors:
    +-- hp:
    |   +-- TRACK(dirty_mask=True)
    |   +-- VALIDATE(range=[0, 100])
    |   +-- HOOK(event="on_change")
    +-- max_hp:
        +-- TRACK(dirty_mask=True)
```

This is what makes the entire system transparent. Every operation at every layer
is visible through a single API.

---

### 3.6 Composition Rule Enforcement

All composition rules documented in LANG_DEC.md are now enforced in code:

**Dependencies** (A requires B):
- `HOOK(on_change)` requires `TRACK`
- `TAG(network)` requires `TAG(serialization)`

**Conflicts** (A excludes B):
- `INTERCEPT(set=deny)` conflicts with `TRACK`
- `INTERCEPT(set=deny)` conflicts with `VALIDATE`

**Canonical Ordering:**

```
TAG -> VALIDATE -> TRACK -> INTERCEPT -> HOOK -> DESCRIBE -> REGISTER
```

**Enforcement API:**

```python
from trinity.decorators.ops import validate_steps, validate_ordering

# Check if a combination of steps is valid
errors = validate_steps([Step(Op.HOOK, {"event": "on_change"})])
# Returns: ["HOOK(on_change) requires TRACK -- add @tracked"]

# Check ordering
errors = validate_ordering([Step(Op.REGISTER, ...), Step(Op.TAG, ...)])
# Returns: ["REGISTER must come after TAG"]
```

Invalid combinations are caught at class definition time --
`ComponentMeta.__new__` calls `validate_steps()` and raises `CompositionError`
if rules are violated.

---

### 3.7 Programmatic Introspection API

Beyond `decompose()`, Trinity provides a full introspection API:

```python
from trinity.decorators.introspection import (
    primitives, composites, chain, find_decorators,
    compose, validate_combination, all_rules
)

# What primitive Ops are on this class?
primitives(Health)           # {Op.TAG, Op.TRACK, Op.VALIDATE, Op.REGISTER, ...}
primitives(Health, "hp")     # {Op.TRACK, Op.VALIDATE} -- field-specific

# What composite decorators were applied?
composites(Health)           # ["component", "tracked", "networked"]

# Full descriptor chain for a field
chain(Health, "hp")          # [NetworkedDescriptor -> TrackedDescriptor -> ValidatedDescriptor -> StorageDescriptor]

# Find all decorators using a specific primitive
find_decorators(primitive=Op.HOOK, event="on_change")  # ["tracked", "observable", ...]

# Build a decorator from primitives
my_decorator = compose(Op.TAG, Op.TRACK, Op.VALIDATE)

# Check if a combination is valid before applying
validate_combination([Op.TRACK, Op.INTERCEPT])  # True/False + error messages

# List all composition rules
all_rules()  # Returns list of Rule objects
```

FlowForge uses this API to: discover available decorators, validate user-created
combinations, display what a class is composed of.

---

### 3.8 Decorator Stacks and Foundation

Stacks are pre-composed decorator bundles that register multiple Trinity layers
at once:

```python
from trinity.decorators.builtin_stacks import multiplayer_character

@multiplayer_character(authority="server")
class Player(Component):
    health: float = 100.0
    position: Vec3 = Vec3.zero()
```

`multiplayer_character` expands to `@component + @tracked + @networked +
@serializable + @validated`, which means Foundation receives:

- **Registry**: Player type registered with instance tracking
- **Tracker**: All fields tracked with dirty flags + network replication flags
- **EventLog**: Changes logged for replay
- **Mirror**: Full schema for serialization

Stack algebra allows combining stacks:

```python
debug_multiplayer = multiplayer_character + debug_overlay  # Combines all Steps
```

14 built-in stacks are available: `production_component`, `networked_entity`,
`multiplayer_character`, `persistent_entity`, `debug_component`, etc.

---

## 4. How Engine Layers Connect Through Foundation

This section explains how each of the engine layers uses the
Trinity + Foundation integration.

### 4.1 Platform Layer

Platform backends are registered as Resources via `@resource` decorator, which
triggers `ResourceMeta.__new__` -> Foundation Registry. GPU device abstractions
(RHI) are Resources tracked in Foundation, making them discoverable via
`registry.subclasses(GPUDevice)`.

### 4.2 Core: ECS

This IS Trinity and Foundation working together directly:

- **ComponentMeta** defines data types, installs descriptor chains
- **SystemMeta** defines logic, schedules into phases
- **World** is ShellLang's `World`, populated by Bridge from Trinity-registered types
- **Queries** go through ShellLang's `World.query()`, with dirty-flag optimization
  backed by Tracker

### 4.3 Core: Memory

The `@pooled` decorator configures `ComponentMeta` with a pool allocator. When
`ComponentMeta.__call__` fires, it checks the pool before allocating. Pool stats
are visible through Foundation Mirror (`pool_stats()` method on the metaclass).
Memory profiling data from `ProfiledDescriptor` can be read through the Inspector.

### 4.4 Core: Task System

The `@system(phase="gameplay")` decorator with `@parallel` configures SystemMeta
to schedule systems into phases with parallelism hints. The task graph
(which systems run when, their dependencies) is visible through Foundation Mirror
by reflecting on `_system_phase`, `_dependencies`, and `_can_parallelize`
attributes. Execution profiling uses `@traced` to record timing in EventLog.

### 4.5 Resource Layer

`AssetMeta` registers asset types (meshes, textures, audio) with Foundation
Registry. Asset loading and streaming operations are tracked by EventLog when
wrapped with `@traced`. Virtual texturing and virtual geometry are modeled as
Resources in Foundation Registry, with their streaming state managed via
StateMeta transitions.

### 4.6 Rendering Layer

Frame graph nodes are Systems registered via `SystemMeta`. Render passes are
registered in Foundation. Material parameters are Components with
`ValidatedDescriptor` chains ensuring values stay in valid ranges. GPU buffer
contents are tracked for hot-reload: when an artist changes a texture, the
`TrackedDescriptor` chain fires, Tracker marks the buffer dirty, and the
rendering system picks it up on the next frame.

### 4.7 Simulation (Physics)

Physics bodies are Components (position, velocity, mass). Collision events are
Events defined via `EventMeta` and dispatched through the event bus. Constraint
types (hinge, ball, slider) are Protocols registered via `ProtocolMeta`.
Simulation state changes are tracked via EventLog, enabling deterministic
replay: replay the Event sequence to reproduce the exact simulation.

### 4.8 Animation

Animation states (idle, walk, run, jump) are defined via `StateMeta` with valid
transitions. Blend tree parameters are tracked components with
`ValidatedDescriptor` ensuring blend weights stay in [0, 1]. Motion matching
databases are Assets registered via `AssetMeta`.

### 4.9 Audio

Audio sources and listeners are Components. Mix bus hierarchies are Resources
(singleton per bus configuration). Audio events (play, stop, fade) are dispatched
through `EventMeta`. Spatial audio parameters (position, attenuation) are
tracked fields that the audio system reads each frame.

### 4.10 Gameplay

Behavior trees, ability systems, and quest systems are all defined with Trinity
decorators. AI decisions are logged in EventLog via `@traced`, enabling post-hoc
debugging: "Why did the AI attack the player?" -> query EventLog by entity. Game
state is queryable through ShellLang: `Enemys.all.where(lambda e: e.ai.state == "aggressive")`.

### 4.11 UI

UI widgets are Components with data binding through `TrackedDescriptor` ->
`on_change` subscriptions. When a game model value changes, the Tracker fires
the UI subscription, which re-renders the widget. Style properties use
`ValidatedDescriptor` to ensure valid CSS-like values. Screen management
(push/pop/replace) uses `StateMeta` for screen state machines.

### 4.12 Networking

This is where the full descriptor chain shines:

```
NetworkedDescriptor   -> queues field update for replication
  TrackedDescriptor   -> marks dirty in Foundation Tracker
    ValidatedDescriptor -> validates new value
      StorageDescriptor -> stores in __dict__
```

RPCs are defined through `ProtocolMeta`. Prediction/reconciliation uses EventLog:
client records predicted events, server sends authoritative events, client rolls
back by replaying EventLog. `SerializableDescriptor` provides Mirror with schema
info for network packet serialization.

### 4.13 World

World partition cells are Resources (one per cell, with streaming state). Cell
streaming state uses `StateMeta` (unloaded -> loading -> loaded -> activated).
Terrain and foliage instances are Components tracked in Foundation Registry.
PCG (procedural content generation) rules are decorated systems that generate
content at load time.

### 4.14 Tooling and Debug

This layer is the heaviest consumer of Foundation:

- **Inspector** uses Mirror for object visualization and field editing
- **Shell** uses Bridge for live manipulation of Trinity objects
- **Profiler** reads Tracker (dirty counts) and EventLog (operation timings)
- **FlowForge** uses `trinity_adapter` which goes through Foundation
- **Replay system** replays EventLog to reproduce exact game state

### 4.15 Deterministic Simulation Support

Foundation provides the runtime infrastructure for deterministic simulation:

| Foundation System | Determinism Role |
|------------------|-----------------|
| **EventLog** | Records all operations with causal chains -- enables exact replay |
| **Tracker** | Transaction system -- enables rollback to any checkpoint |
| **Mirror** | `schema_hash()` -- detects state desynchronization across clients |
| **Registry** | Instance tracking -- enables snapshot of all live entities |
| **ContentStore** | Content-addressable snapshots -- efficient diff between states |
| **DeltaSync** | Minimal delta patches -- efficient state transfer for netcode |
| **Provenance** | Derivation trees -- debug "why did simulation diverge?" |

The deterministic simulation kernel uses fixed-point math (no floats in
simulation), command-based input (deterministic ordering), and hierarchical
checksums (desync detection). Foundation's EventLog records every command and
state change, enabling frame-perfect replay.

---

## 5. FlowForge Integration Path

### 5.1 The Rule: FlowForge -> Foundation -> Trinity

FlowForge NEVER imports from `trinity/` directly. All access goes through
Foundation, specifically through a `trinity_adapter` module that wraps
Foundation calls.

```
  FlowForge UI
      |
      v
  trinity_adapter (FlowForge backend)
      |
      v
  Foundation (Registry, Mirror, Tracker, Bridge, Inspector)
      |
      v
  Trinity (metaclasses, descriptors, decorators)
```

**Why this matters:**

1. **Security.** Foundation's Capabilities system can restrict what FlowForge
   can access. Direct Trinity access would bypass this.
2. **Consistency.** All mutations go through Tracker, so undo/redo and change
   subscriptions work correctly. Direct field writes would bypass tracking.
3. **Observation.** All operations are logged in EventLog. Direct calls would
   be invisible to debugging and replay.

### 5.2 FlowForge API Surface

The trinity_adapter provides these functions to FlowForge:

| Function | Returns | Foundation System Used |
|----------|---------|----------------------|
| `check_trinity_status()` | `TrinityStatus` (available, type count) | Registry |
| `list_registered_types()` | `list[RegisteredType]` | Registry.all_types() |
| `get_type_info(name)` | `dict` (fields, defaults, metadata) | Registry + Mirror |
| `query_active_instances()` | `list[ActiveInstance]` | Registry.instances() |
| `get_instance_mirror(id)` | `ObjectMirror` | Mirror |
| `get_recent_events(limit)` | `list[RecentEvent]` | EventLog |
| `list_decorators()` | `list[str]` | Registry.types_with_decorator() |

### 5.3 Data Flow: Visual Editor -> Runtime

Here is the complete flow when a user edits a component field in FlowForge:

```
  1. User drags a Health component onto the canvas in FlowForge
  2. FlowForge calls trinity_adapter.get_type_info("Health")
  3. trinity_adapter calls Foundation Registry:
     registry.get("mymodule.Health") -> Health class
  4. trinity_adapter calls Foundation Mirror:
     mirror(Health) -> ClassMirror with fields, types, defaults
  5. FlowForge renders editable fields based on FieldInfo:
     - hp: float, default=100.0
     - max_hp: float, default=100.0
  6. User changes hp to 50 in the property editor
  7. FlowForge calls trinity_adapter.set_field(instance_id, "hp", 50)
  8. trinity_adapter locates the instance via Registry.instances()
  9. trinity_adapter calls Inspector (or setattr directly):
     inspector -> Mirror.set("hp", 50)
       -> setattr(instance, "hp", 50)
         -> TrackedDescriptor.__set__
           -> tracker.mark_dirty(instance, "hp", 100, 50)
 10. Tracker fires subscriptions:
     - FlowForge's own subscription updates the UI
     - Networking marks the field for replication
     - EventLog records the change (if in traced context)
```

---

## 6. ShellLang and AI Interface

### 6.1 The 5 Primitives and Their Trinity+Foundation Mapping

| Primitive | ShellLang | Trinity Source | Foundation Backing |
|-----------|-----------|---------------|-------------------|
| **ENTITY** | `Entity(id)` | -- | World manages entity lifecycle |
| **COMPONENT** | Any class | ComponentMeta creates it | Registry tracks it |
| **QUERY** | `World.query(*Cs)` | SystemMeta defines queries | Tracker dirty flags optimize |
| **MUTATE** | `World.set(e, C, f, v)` | Descriptors intercept write | Tracker records change |
| **SNAPSHOT** | `World.snap()` | -- | World deep-copies state |

### 6.2 AI Agent Interaction

AI agents interact through the `AIInterface`, which accepts structured dict
commands and returns structured responses.

**Example session:**

```python
from foundation.bridge import create_ai_interface
ai = create_ai_interface()

# Check what types exist
result = ai.execute({"op": "list_types"})
# {"types": ["Health", "Position", "Velocity", ...], "count": 42}

# Spawn an entity with a Health component
result = ai.execute({
    "op": "spawn",
    "component": "Health",
    "fields": {"hp": 50, "max_hp": 100}
})
# {"entity": 1, "component": "Health", "fields": {"hp": 50, "max_hp": 100}}

# Query for low-health entities
result = ai.execute({
    "op": "query",
    "components": ["Health"],
    "where": {"Health.hp": {"<": 20}},
    "limit": 10
})
# {"entities": [{"id": 1, "components": ["Health"]}], "count": 1}

# Validate before executing
result = ai.validate({
    "op": "set",
    "entity": 1,
    "component": "Health",
    "field": "hp",
    "value": 100
})
# {"valid": True}

# Preview without executing
result = ai.dry_run({
    "op": "set",
    "entity": 1,
    "component": "Health",
    "field": "hp",
    "value": 100
})
# {"would_change": {"entity": 1, "field": "Health.hp", "from": 50, "to": 100}}
```

**Validation flow:**

```
  AI sends: {"op": "set", "entity": 42, "component": "Health", "field": "hp", "value": 100}
      |
  AIInterface.validate():
    1. Checks "op" is in VALID_OPERATIONS
    2. Checks required fields present (entity, component, field, value)
    3. Checks component type exists in registry
    4. Checks entity exists in World
      |
  AIInterface.execute():
    1. Gets component type from registry
    2. Gets component instance from World.get(entity, Type)
    3. Reads old_value via getattr
    4. Calls World.set(entity, Type, field, value)
       -> setattr(component, "hp", 100)
         -> TrackedDescriptor chain fires
           -> Foundation Tracker notified
           -> EventLog records Change
    5. Returns {"entity": 42, "component": "Health", "field": "hp", "old": 50, "new": 100}
```

### 6.3 Live Debugging via Shell

The interactive Shell provides a Python-like REPL with pre-configured namespace:

```python
from foundation.bridge import create_shell
shell = create_shell()

# The namespace includes all registered component types and sugar functions
shell.execute("e = create()")                       # Create entity
shell.execute("world.attach(e, Health(hp=80))")     # Attach component
shell.execute("e_proxy = EntityProxy(e)")           # Wrap for dot access
shell.execute("e_proxy.health.hp")                  # -> 80
shell.execute("e_proxy.health.hp = 50")             # -> "Health.hp: 80 -> 50"

# Time travel
shell.execute("mark('before_fight')")
shell.execute("e_proxy.health.hp = 10")
shell.execute("rewind('before_fight')")             # -> "Rewound to 'before_fight'"
shell.execute("e_proxy.health.hp")                  # -> 80 (restored)

# Chainable queries
shell.execute("Healths.all.where(lambda e: e.health.hp < 20).count()")
```

The Shell uses Foundation's sugar layer (`EntityProxy`, `QueryResult`, `TypeQuery`)
which route all mutations through `World.set()` -> descriptor chain -> Tracker.

---

## 7. End-to-End Examples

### 7.1 Defining and Using a Networked Tracked Component

This example traces the COMPLETE lifecycle of a component from definition through
every Foundation system.

**Step 1: Definition (decorator application)**

```python
# Modern syntax (Phase 8 -- preferred)
@component
@tracked
@networked(authority="server", interpolate=True)
class Health(Component):
    hp: Annotated[float, Range(0, 100)] = 100.0
    max_hp: Annotated[float, Validated(gt=0)] = 100.0
```

What happens at class definition time:

```
  1. @component decorator:
     - calls make_decorator("component", ...) which returns a decorator function
     - the decorator applies Steps: TAG(component=True), REGISTER(ecs_core)
     - _after_component() runs:
       a) Sets cls._component = True, cls._component_name = "Health"
       b) Calls ComponentMeta._process_fields(cls):
          -> _field_types = {"hp": float, "max_hp": float}
          -> _field_defaults = {"hp": 100.0, "max_hp": 100.0}
          -> _field_offsets = {"hp": 0, "max_hp": 1}
       c) Calls ComponentMeta._install_descriptors(cls):
          -> For each field, builds descriptor chain based on decorator markers
          -> Reads Annotated metadata for Range/Validated descriptors
       d) Registers in ComponentMeta._registry

  2. @tracked decorator:
     - Sets cls._track_changes = True
     - Records Step(Op.TRACK)
     - This marker causes _install_descriptors to include TrackedDescriptor

  3. @networked(authority="server", interpolate=True) decorator:
     - Sets cls._network_config = NetworkConfig(authority="server", interpolated=True)
     - Records Step(Op.INTERCEPT, network=True)
     - This marker causes _install_descriptors to include NetworkedDescriptor

  4. ComponentMeta._install_descriptors builds this chain for each field:
     NetworkedDescriptor(authority="server", interpolated=True)
       -> TrackedDescriptor(field_offset=0)
         -> ValidatedDescriptor(range=(0, 100))  [from Annotated metadata]
           -> StorageDescriptor(default=100.0)

  5. ComponentMeta._register_with_foundation(cls):
     -> from foundation import registry
     -> registry.register(Health, name="mymodule.Health", track_instances=True)
     -> Registry wraps Health.__init__ to track instances in WeakSet
```

**Step 2: Instance creation**

```python
player_health = Health()
```

```
  1. ComponentMeta.__call__ (metaclass __call__):
     a) Checks budget (if @budgeted): not over limit
     b) Checks pool (if @pooled): pool empty, fall through
     c) Calls type.__call__ -> Health.__init__
  2. Registry's wrapped __init__ runs:
     -> original __init__ sets hp=100.0, max_hp=100.0
     -> WeakSet.add(player_health)
  3. player_health is now tracked in Foundation Registry
```

**Step 3: Field write**

```python
player_health.hp = 50
```

```
  1. Python calls NetworkedDescriptor.__set__(player_health, 50)
  2. BaseDescriptor.__set__:
     a) pre_set(player_health, 50) -> no transform (no validation on this chain)
     b) old_value = _get_stored_safe(player_health) -> 100.0
     c) inner.__set__(player_health, 50) -> TrackedDescriptor.__set__
        i)  TrackedDescriptor delegates to inner: StorageDescriptor.__set__
            -> player_health.__dict__["hp"] = 50
        ii) TrackedDescriptor.post_set(player_health, 50, 100.0):
            - Adds "hp" to player_health._dirty_fields
            - _notify_foundation_tracker:
              tracker.mark_dirty(player_health, "hp", 100.0, 50)
            - _notify_eventlog:
              Creates Change(entity=player_health.id, field="hp", old=100.0, new=50)
              add_change_to_current_event(change)
     d) NetworkedDescriptor.post_set(player_health, 50, 100.0):
        - Appends to player_health._network_queue:
          {"field": "hp", "value": 50, "old_value": 100.0, "priority": 0}

  FOUNDATION TRACKER fires:
    - Global callbacks see (player_health, "hp", 100.0, 50)
    - Per-object callbacks for player_health fire
    - Per-type callbacks for Health fire

  NETWORKING system (on next tick):
    - Reads player_health._network_queue -> sends replication packet
    - Calls tracker.mark_clean(player_health) after successful send

  EVENT LOG (if inside @traced context):
    - Change is attached to current Event
    - Event has causal chain back to root cause
```

**Step 4: Mirror introspection**

```python
from foundation.mirror import mirror, schema_hash
m = mirror(player_health)
print(m.fields)        # {"hp": FieldInfo(name="hp", type=float, default=100.0, ...),
                        #  "max_hp": FieldInfo(...)}
print(m.to_dict())     # {"hp": 50, "max_hp": 100.0}
print(schema_hash(Health))  # "a3f7b2c1e9d04f82"
```

**Step 5: FlowForge visualization**

```python
# FlowForge backend calls:
type_info = trinity_adapter.get_type_info("Health")
# -> Uses Registry.get("mymodule.Health") then mirror(Health)
# -> Returns {"name": "Health", "fields": {"hp": {"type": "float", "default": 100.0}, ...}}

# FlowForge renders a property panel with:
# [hp    ] [50.0    ]  <- editable float field
# [max_hp] [100.0   ]  <- editable float field
```

**Step 6: ShellLang query**

```python
shell.execute("Healths.all.where(lambda e: e.health.hp < 60).count()")
# -> 1 (our player_health has hp=50)
```

**Step 7: AI agent command**

```python
ai.execute({
    "op": "set",
    "entity": player_health.id,
    "component": "Health",
    "field": "hp",
    "value": 100
})
# -> setattr triggers the full descriptor chain again
# -> Tracker records the change, EventLog records it
```

---

### 7.2 A System Processing Tracked Components

```python
@system(phase="gameplay")
@traced
class DamageSystem(System):
    def update(self, dt):
        for entity in self.query(Health):
            if entity.health.hp <= 0:
                self.emit(EntityDied(entity=entity.id))
```

**What happens when `DamageSystem.update(dt=0.016)` is called:**

```
  1. SystemMeta.__new__ (at definition time):
     a) Registers DamageSystem with SystemMeta._registry
     b) Sets _system_phase = SystemPhase.UPDATE (from @system(phase="gameplay"))
     c) Extracts query types from type hints
     d) Registers with Foundation Registry

  2. @traced wrapper activates (at call time):
     a) entity_id = None (DamageSystem has no .id -> pass-through)
     b) operation = "DamageSystem.update"
     c) Creates Event(tick=current_tick, operation="DamageSystem.update", ...)
     d) Sets ContextVars: _current_event, _depth=0

  3. self.query(Health) executes:
     a) Goes through Foundation Query system
     b) Returns entities that have Health component

  4. For each entity, entity.health.hp is READ:
     a) TrackedDescriptor.__get__ fires
     b) BaseDescriptor.__get__ calls read-tracking (for incremental computation)
     c) Provenance recording (if foundation.provenance is available)
     d) Returns stored value

  5. If hp <= 0, self.emit(EntityDied(...)):
     a) EntityDied is an @event type (EventMeta registered)
     b) Event bus dispatches to subscribers
     c) @traced records this as a nested operation:
        Event(operation="DamageSystem.emit", depth=1,
              immediate_parent="DamageSystem.update")

  6. @traced wrapper completes:
     a) Records result on Event
     b) _event_log.record(event) indexes by tick, entity, operation
     c) Resets ContextVars
```

---

### 7.3 Inspecting at Runtime

All three access methods (Inspector, Shell, AI) go through Foundation:

**Inspector (GUI tool):**

```python
from foundation.inspector import inspector
panel = inspector.inspect(player_health)
output = panel.render()
# === Health ===
#   [hp]: 50.0
#   [max_hp]: 100.0
```

The Inspector uses Mirror to enumerate fields, then renders appropriate widgets
(float input for `float` fields, checkbox for `bool`, etc.). Edits go through
`mirror.set(name, value)` -> `setattr` -> descriptor chain -> Tracker.

**Shell (REPL):**

```python
shell.execute("mirror(player_health).to_dict()")
# {"hp": 50.0, "max_hp": 100.0}

shell.execute("mirror(Health).describe()")
# class Health:
#   hp: float = 100.0
#   max_hp: float = 100.0
```

**AI Interface (structured commands):**

```python
ai.execute({"op": "inspect", "entity": player_health_entity_id})
# {"entity": 1, "components": {"Health": {"hp": 50.0, "max_hp": 100.0}}}

ai.execute({"op": "schema", "type": "Health"})
# {"name": "Health", "fields": {"hp": {"type": "float", "default": 100.0}, ...}}
```

**Cross-layer introspection:**

```python
# See every Step that built this class across all three Trinity layers
from trinity.decorators.ops import expand
expand(Health)  # Prints full layered trace of all Steps
```

All three access methods converge on the same Foundation systems: Mirror for
reflection, Tracker for change notification, EventLog for history.

---

## 8. Dependency and Import Rules

### 8.1 Import Graph

```
                         +-----------+
                         | FlowForge |
                         +-----+-----+
                               |
                               | imports from
                               v
             +-------------------------------------+
             |          foundation/                 |
             |  +----------+  +--------+  +------+ |
             |  | registry |  | mirror |  |tracker| |
             |  +----------+  +--------+  +------+ |
             |  +----------+  +---------+          |
             |  | eventlog |  |inspector|          |
             |  +----------+  +---------+          |
             |                                     |
             |  +--------------------------------+ |
             |  |  bridge.py  (SOLE COUPLING)    | |
             |  |  imports from both trinity/    | |
             |  |  and foundation/shelllang/     | |
             |  +--------------------------------+ |
             |                                     |
             |  +--------------------------------+ |
             |  |  shelllang/                     | |
             |  |  core.py | sugar.py | ai.py    | |
             |  |  repl.py                       | |
             |  +--------------------------------+ |
             +------------------+------------------+
                                |
                  bridge.py imports from (try/except)
                                |
                                v
             +-------------------------------------+
             |           trinity/                   |
             |  +-------------------+              |
             |  | metaclasses/      |              |
             |  |  component_meta   |  registers   |
             |  |  system_meta      |  into        |
             |  |  resource_meta    |  Foundation   |
             |  |  event_meta       |  Registry     |
             |  |  asset_meta       |  (try/except) |
             |  |  protocol_meta    |              |
             |  |  state_meta       |              |
             |  +-------------------+              |
             |  +-------------------+              |
             |  | descriptors/      |  notifies    |
             |  |  tracking.py      |  Foundation   |
             |  |  networking.py    |  Tracker      |
             |  |  observable.py    |  (try/except) |
             |  +-------------------+              |
             |  +-------------------+              |
             |  | decorators/       |              |
             |  |  ecs_core.py      |              |
             |  |  scheduling.py    |              |
             |  |  data_flow.py     |              |
             |  |  ...              |              |
             |  +-------------------+              |
             +-------------------------------------+
```

**Key import rules:**

| Source | Can import from | Cannot import from |
|--------|----------------|-------------------|
| `trinity/` | Standard library only (+ `foundation` via try/except) | `flowforge/` |
| `foundation/` core modules | Standard library, other `foundation/` modules | `trinity/`, `flowforge/` |
| `foundation/bridge.py` | `trinity.metaclasses`, `foundation.shelllang` | `flowforge/` |
| `flowforge/` | `foundation/` (via trinity_adapter) | `trinity/` directly |

### 8.2 Why This Matters

**Trinity can run standalone.** Without Foundation installed, Trinity metaclasses
still create classes, install descriptors, and maintain their internal registries.
The `try/except ImportError` pattern means all Foundation calls gracefully degrade.
This enables:

- Unit testing Trinity in isolation
- Code generation from Trinity type definitions
- Static analysis of component schemas
- Using Trinity in non-game contexts

**Foundation can run standalone.** With manually registered types (no metaclasses),
Foundation's Registry, Tracker, EventLog, Mirror, and ShellLang all work. This
enables:

- Testing Foundation systems independently
- Using Foundation with non-Trinity types
- Building alternative metaprogramming layers

**Bridge is optional.** It is only needed when you want Trinity objects accessible
through ShellLang's World. In a headless simulation or a test, you might use
Trinity directly without Bridge.

**FlowForge is fully decoupled from Trinity internals.** If Trinity's metaclass
implementation changes (e.g., field processing is rewritten), FlowForge does not
need to change as long as Foundation's API is stable. This is the classic
adapter/facade pattern applied at the architecture level.

---

## 9. Extension Points

### 9.1 Adding a New Metaclass

To add a new metaclass (e.g., `WidgetMeta` for UI widgets):

1. **Inherit from EngineMeta** so it participates in the global type registry:

```python
class WidgetMeta(EngineMeta):
    _registry: ClassVar[dict[int, type]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if name == "Widget":
            return cls

        with mcs._lock:
            cls._widget_id = mcs._next_id
            mcs._next_id += 1
            mcs._registry[cls._widget_id] = cls

            # Foundation integration
            try:
                from foundation import registry
                if not registry.is_registered(cls):
                    registry.register(cls, name=f"{cls.__module__}.{name}",
                                     track_instances=True)
                    registry.set_metadata(cls, "widget_id", cls._widget_id)
            except ImportError:
                pass

        return cls
```

2. **Set up Tracker subscriptions** if your metaclass installs descriptors that
   should be tracked.

3. **Define Mirror schema** by ensuring your metaclass sets `__annotations__`
   and standard field attributes that Mirror can read.

### 9.2 Adding a New Descriptor

To add a new descriptor (e.g., `AuditedDescriptor` that logs to an external
audit service):

1. **Inherit from BaseDescriptor** and set `descriptor_id`:

```python
class AuditedDescriptor(BaseDescriptor[T]):
    descriptor_id = "audited"
    accepts_inner = ("tracked", "storage", "validated")
    accepts_outer = ("networked",)
    excludes = ()

    def post_set(self, obj, value, old_value):
        if value != old_value:
            # Foundation integration: mark dirty
            try:
                from foundation import tracker
                tracker.mark_dirty(obj, self._name, old_value, value)
            except ImportError:
                pass

            # Custom: send to audit service
            self._send_audit(obj, old_value, value)

    def get_metadata(self):
        meta = super().get_metadata()
        meta["audited"] = True
        return meta

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.INTERCEPT, {"get": "audit_read", "set": "audit_write"}),
            Step(Op.TAG, {"audited": True}),
        ]
```

2. **Add `mark_dirty()` calls** if this descriptor replaces TrackedDescriptor
   or is the outermost tracked layer.

3. **Add EventLog events** if the descriptor should contribute to causal tracing:
   call `add_change_to_current_event()` from `post_set`.

4. **Add Mirror metadata** by returning it from `get_metadata()`. Mirror does
   not read this directly, but Inspector and other tools can query the descriptor
   chain and aggregate metadata.

### 9.3 Adding a New Foundation System

To add a new Foundation system (e.g., `Capabilities` for security):

1. **Create the module** in `foundation/` with no imports from `trinity/`:

```python
# foundation/capabilities.py
class Capabilities:
    def can_read(self, cls, field): ...
    def can_write(self, cls, field): ...
```

2. **Wire into Bridge** by adding accessor functions:

```python
# In foundation/bridge.py, add:
def create_secure_ai_interface():
    from foundation.capabilities import capabilities
    ai = create_ai_interface()
    # Wrap with capability checks
    return SecureAIInterface(ai, capabilities)
```

3. **Expose through ShellLang** by adding it to the Shell namespace:

```python
# In Shell._setup_namespace():
self._namespace["capabilities"] = capabilities
```

4. **Make available to FlowForge** through the trinity_adapter:

```python
# In flowforge trinity_adapter:
def check_permission(type_name, field_name, operation):
    from foundation.capabilities import capabilities
    cls = registry.get(type_name)
    return capabilities.can_write(cls, field_name) if operation == "write" else ...
```

### 9.4 Trinity Validation Tooling (Phase 10)

Four tools in `trinity/tools/`:

| Tool | Purpose |
|------|---------|
| `doctor.py` | Validates all registered classes -- checks descriptor chains, composition rules, Foundation registration |
| `step_trace.py` | Formatted trace of all Steps with source file locations |
| `op_coverage.py` | Coverage analysis -- which Ops are used by which classes |
| `lint.py` | Import-time validation -- catches composition errors before runtime |

```python
# Run doctor on all registered components
from trinity.tools.doctor import doctor
report = doctor()  # Returns validation report for every registered class

# Trace a specific class
from trinity.tools.step_trace import trace
trace(Health)  # Prints Steps with file:line locations

# Check Op coverage
from trinity.tools.op_coverage import coverage
coverage()  # Shows which Ops are used/unused across codebase
```

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **Trinity** | The definition-time metaprogramming pillar: metaclasses, descriptors, and decorators that control class creation and field behavior. |
| **Foundation** | The runtime infrastructure pillar: Registry, Tracker, EventLog, Mirror, Bridge, and ShellLang. |
| **Bridge** | `foundation/bridge.py` -- the sole module that imports from both Trinity and Foundation to provide live integration. |
| **Registry** | Foundation's unified type directory. Singleton `registry` that tracks all registered types and their instances. |
| **Tracker** | Foundation's change-tracking system. Singleton `tracker` that maintains dirty flags, fires change subscriptions, and supports undo/redo. |
| **EventLog** | Foundation's operation history. Records Events with causal chains linking operations to their causes and effects. |
| **Mirror** | Foundation's reflection system. `mirror(obj)` returns an ObjectMirror or ClassMirror for uniform introspection. |
| **ShellLang** | Foundation's 5-primitive ECS language (Entity, Component, Query, Mutate, Snapshot) with interactive Shell and AI Interface. |
| **FlowForge** | The visual node editor that accesses Trinity only through Foundation. |
| **Descriptor Chain** | A linked list of descriptors (outermost to innermost) that intercept field reads and writes. Example: `Networked -> Tracked -> Validated -> Storage`. |
| **Metaclass Steps** | The numbered operations in a metaclass's `__new__` method: generate ID, process fields, install descriptors, validate, register, Foundation integration. |
| **Op** | An enumerated operation type in the decorator system (`TAG`, `REGISTER`, `DESCRIBE`, etc.). One of the 7 primitive operations. |
| **Step** | A single Op with arguments -- the atomic unit of Trinity's introspection system. Represented as `(Op, params)` pair. |
| **descriptor_steps** | Property on every descriptor returning the Steps it performs. Fed into `decompose()` for cross-layer introspection. |
| **_metaclass_steps** | Class attribute recording Steps performed during metaclass `__new__`. Makes the metaclass layer visible to introspection. |
| **decompose()** | Function that collects Steps from all three Trinity layers (decorators, metaclasses, descriptors) for a class. |
| **decompose_layered()** | Variant of `decompose()` that groups Steps by layer (decorator/metaclass/descriptor). |
| **Composition Rule** | Constraint on which Ops can coexist -- includes dependencies (A requires B), conflicts (A excludes B), and canonical ordering. |
| **Stack** | Pre-composed bundle of decorators applied as a unit. Supports algebra (`+` to combine). 14 built-in stacks available. |
| **Session** | Serializable snapshot of entire engine state (world + editor + undo history). The "image" of the running engine. |
| **VIPER Extensions** | Foundation systems beyond the core 6: Paths, Query, ContentStore, DeltaSync, Provenance, QueryCacheMirror, Capabilities. |
| **Annotated Syntax** | `Annotated[type, Descriptor1(), Descriptor2()]` field declaration form. The preferred way to declare descriptor-enhanced fields (Phase 8). |
| **Primitive** | One of ShellLang's 5 semantic primitives: Entity, Component, Query, Mutate, Snapshot. |
| **Causal Chain** | The linked sequence of Events in the EventLog from root cause to final effect. Tracked via `root_cause`, `immediate_parent`, and `depth` fields. |
| **Dirty Flag** | A marker indicating a field has been modified since last clean. Maintained by TrackedDescriptor -> Tracker.mark_dirty(). |
| **WeakSet** | Python's `weakref.WeakSet` used by Registry for instance tracking. Instances are garbage-collected normally. |
| **try/except ImportError** | The pattern used by Trinity to optionally integrate with Foundation. Ensures Trinity works standalone. |
| **FieldInfo** | Mirror's `dataclass` describing a field: name, type, has_default, default, metadata. |
| **Event** | EventLog's `dataclass` describing an operation: tick, operation, entity, changes, causal chain. |
| **Change** | A record of a single field mutation: entity, field, old_value, new_value. Used by both Tracker and EventLog (separate classes with similar shape). |
| **Transaction** | A group of Tracker Changes that can be committed or rolled back atomically. |
| **EntityProxy** | ShellLang sugar that provides dot-access to components: `proxy.health.hp`. |
| **QueryResult** | ShellLang sugar that provides chainable filtering: `.where(...).near(...).count()`. |
| **AIInterface** | ShellLang's structured command API for AI agents: `execute()`, `validate()`, `dry_run()`. |
| **TrinityWorldAdapter** | Bridge class that maintains bidirectional mapping between Trinity instances and ShellLang entities. |
| **ComponentMeta** | The most complex metaclass. Creates component types with unique IDs, field processing, descriptor installation, and Foundation registration. |
| **TrackedDescriptor** | The primary runtime integration seam. Its `post_set()` calls `tracker.mark_dirty()` and `add_change_to_current_event()`. |
| **@traced** | Decorator that wraps operations to record Events in the EventLog with full causal chain tracking. |
| **schema_hash** | Mirror function that generates a stable hash of a class's schema for migration detection and protocol compatibility. |
