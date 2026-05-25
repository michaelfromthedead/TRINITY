# PROJECT: Foundation Runtime Infrastructure

## Scope

The Foundation package (8,487+ lines) provides Trinity's runtime infrastructure layer bridging the component system with runtime capabilities including reflection, serialization, change tracking, querying, debugging, and security.

### In Scope

- Layer 0 (Essential): mirror, serializer, paths, eventlog, provenance, content_store, delta_sync, migrations, constants
- Layer 1 (Structural): registry with type and instance tracking
- Layer 2 (Reactive): tracker (change detection, undo/redo), query system with caching
- Layer 3 (Interactive): inspector, inspector_views, shell, secure_shell, capabilities
- Layer 4 (Integration): bridge.py (Trinity adapter), shelllang/ (ECS DSL + AI interface)
- ShellLang subsystem: core (5 primitives), sugar (fluent DSL), ai (JSON commands), repl

### Out of Scope

- Trinity component system itself (separate package)
- Renderer/GPU systems
- Audio systems
- External networking beyond delta_sync primitives

## Goals

1. Provide uniform reflection for any Python object (ObjectMirror/ClassMirror)
2. Implement full serialize/deserialize with schema versioning and circular reference handling
3. Enable content-addressable storage with structural diffing (Git-style)
4. Track computed value lineage via context variables (provenance)
5. Implement causal chain event logging with @traced decorator
6. Provide change detection with dirty flags and undo/redo transactions
7. Implement first-class query objects with caching and reactive subscriptions
8. Deliver capability-based security (immutable sets, context managers)
9. Bridge Trinity component system to Foundation infrastructure
10. Provide dual-interface ECS access: human-friendly DSL and AI-friendly JSON commands

## Constraints

- Thread safety required: use RLock in Tracker, Registry, QueryCache
- Memory safety: use weak references in instance tracking and query cache
- Context isolation: use contextvars for thread-safe state (_current_event, _current_tick, _current_capabilities)
- Layered dependencies: each layer may only depend on lower layers
- Schema versioning: migrations must support BFS path finding between versions
- Capability immutability: CapabilitySet must be immutable, grant/revoke return new sets
- Python 3.13 target: all code must be compatible with statically-linked 3.13 interpreter

## Acceptance Criteria

### Layer 0 (Essential)

- [ ] ObjectMirror can reflect any Python object including slots, properties, methods
- [ ] ClassMirror provides class-level reflection with inheritance traversal
- [ ] Serializer handles circular references without infinite recursion
- [ ] Schema versioning tracks version hashes
- [ ] Paths module supports get_path/set_path for dotted navigation
- [ ] EventLog tracks causal chains with immediate_parent, root_cause, depth
- [ ] Provenance captures computation inputs via @track_provenance
- [ ] ContentStore implements SHA-256 content hashing
- [ ] ContentStore supports MemoryBackend and FileBackend protocols
- [ ] ContentDiffer computes structural diffs by comparing hashes
- [ ] DeltaSync produces minimal change patches
- [ ] Migrations registry finds BFS paths between schema versions

### Layer 1 (Structural)

- [ ] Registry tracks types and instances with thread-safe operations
- [ ] Instance tracking uses weak references to avoid memory leaks
- [ ] Registry integrates with Trinity's ComponentMeta.all_components()

### Layer 2 (Reactive)

- [ ] Tracker detects changes and maintains dirty flags
- [ ] Undo/redo transactions work correctly in nested scenarios
- [ ] Query objects support filter algebra: union, intersection, difference
- [ ] Query filters: WhereFilter, NearFilter, HasComponentFilter, AndFilter, OrFilter, NotFilter
- [ ] TrackedQueryCache auto-invalidates on object changes
- [ ] Query subscriptions fire on_add/on_remove/on_change callbacks
- [ ] Query hashing is content-based for stable identity

### Layer 3 (Interactive)

- [ ] Inspector visualizes objects with pluggable views
- [ ] History/Causality/Provenance views render correctly
- [ ] Shell executes live Python code safely
- [ ] SecureShell enforces capabilities before code execution
- [ ] Capabilities: READ, WRITE, CREATE, DELETE, EXECUTE, SPAWN, NETWORK, FILESYSTEM
- [ ] @require_capability decorator enforces capability checks
- [ ] SecureContext supports nested capability restriction

### Layer 4 (Integration)

- [ ] Bridge pulls ComponentMeta.all_components() into ShellLang
- [ ] TrinityWorldAdapter syncs bidirectionally between Trinity instances and ShellLang entities
- [ ] create_ai_interface() returns working AI command executor

### ShellLang

- [ ] Entity primitive: uint64 identifier, hashable, auto-incrementing
- [ ] Component primitive: any Python class attaches to entity
- [ ] Query primitive: predicate returns matching entities
- [ ] Mutate primitive: tracked change with entity, field, value
- [ ] Snapshot primitive: frozen world state, supports save/restore/diff
- [ ] EntityProxy provides dot access to components
- [ ] QueryResult supports chainable fluent interface
- [ ] TimeManager supports named snapshots and undo/redo
- [ ] AIInterface supports: query, set, spawn, destroy, snap, restore, inspect, schema, list_types, count
- [ ] AIInterface.validate() checks command validity without executing
- [ ] AIInterface.dry_run() previews effects without mutating
- [ ] REPL runs interactive loop with eval/exec

## Dependencies

- Python 3.13 (not 3.14)
- contextvars (stdlib)
- hashlib for SHA-256 (stdlib)
- typing with TypeVar for generics (stdlib)
- Trinity component system (external, for bridge.py)
