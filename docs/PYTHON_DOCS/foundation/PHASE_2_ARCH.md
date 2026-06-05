# PHASE 2 ARCHITECTURE: Layers 1-2 Structural and Reactive

## Overview

Phase 2 builds upon Layer 0 primitives to provide structural organization (registry) and reactive systems (tracker, query). These modules depend on Layer 0 and enable higher-level features.

## Layer 1: Structural

### registry.py (199 lines)

Unified type registry with instance tracking.

**Classes**:
- `TypeRegistry`: Singleton managing type registrations
  - `register(cls)`: Add type to registry
  - `unregister(cls)`: Remove type
  - `get(name)`: Retrieve type by name
  - `all_types()`: List all registered types

- `InstanceTracker`: Tracks live instances per type
  - Uses weak references to avoid memory leaks
  - Automatic cleanup on garbage collection
  - `instances(cls)`: Returns list of live instances

**Thread Safety**: RLock protects all registry operations.

**Integration Point**: `get_trinity_registry()` in bridge.py pulls ComponentMeta.all_components() into this registry.

**Design Decisions**:
- Weak references prevent registry from keeping objects alive
- Type names are unique (no namespace support at this layer)
- Instance tracking is opt-in per type

## Layer 2: Reactive

### tracker.py (218 lines)

Change detection with dirty flags, undo/redo transactions.

**Classes**:
- `ChangeTracker`: Singleton managing tracked objects
  - `track(obj)`: Begin tracking object
  - `untrack(obj)`: Stop tracking object
  - `is_dirty(obj)`: Check if object modified
  - `clear_dirty(obj)`: Reset dirty flag

- `Transaction`: Context manager for atomic changes
  - `begin()`: Start transaction
  - `commit()`: Apply changes
  - `rollback()`: Revert changes
  - Supports nesting (inner rollback doesn't affect outer)

- `UndoStack`: Manages undo/redo history
  - `push(change)`: Record change
  - `undo()`: Revert last change
  - `redo()`: Reapply reverted change
  - `clear()`: Reset history

**Change Detection Mechanism**:
1. Tracked objects get proxy wrapper
2. Proxy intercepts __setattr__
3. Old value recorded before modification
4. Dirty flag set on modification

**Thread Safety**: RLock protects tracker state.

### query.py (1,042 lines)

First-class query objects with filters, algebra, and caching.

**Filter Classes**:
- `Filter`: Base protocol for all filters
- `WhereFilter`: Field predicate (e.g., health__lt=50)
- `NearFilter`: Spatial proximity (entity, radius)
- `HasComponentFilter`: Component type check
- `AndFilter`: Logical AND of filters
- `OrFilter`: Logical OR of filters
- `NotFilter`: Logical negation

**Query Class**:
- Constructor: `Query(*ComponentTypes)`
- Filter chaining: `.where()`, `.near()`, `.has()`
- Execution: `query(world)` returns matching entities
- Algebra operators:
  - `q1 | q2`: Union (OR)
  - `q1 & q2`: Intersection (AND)
  - `q1 - q2`: Difference (EXCEPT)

**Reactive Subscriptions**:
- `query.subscribe(on_add=fn, on_remove=fn, on_change=fn)`
- Callbacks fire when query results change
- Subscriptions are weak references (auto-cleanup)

**TrackedQueryCache**:
- Caches query results by query hash
- Auto-invalidates when tracked objects change
- Uses content-based hashing for stable keys
- Thread-safe via RLock

**Content-Based Hashing**:
- Query hash = hash(component_types, filters, sort_order)
- Same query structure = same hash
- Enables cache deduplication

### query_cache_mirror.py (178 lines)

Meta-inspection of query cache state.

**Classes**:
- `QueryCacheMirror`: Inspects TrackedQueryCache
  - `cache_size()`: Number of cached queries
  - `hit_rate()`: Cache hit percentage
  - `cached_queries()`: List of cached query hashes
  - `inspect_query(hash)`: Details for specific query
  - `invalidation_count()`: Total invalidations

**Purpose**: Debugging and performance tuning of query system.

## Data Flow

```
Object Registration:
  Type -> TypeRegistry -> InstanceTracker (weak refs)

Change Tracking:
  Object -> ChangeTracker.track() -> Proxy Wrapper
  Mutation -> Proxy intercept -> Record Change -> Set Dirty Flag

Query Execution:
  Query(*types) -> Filter Chain -> Execute on World -> Results
       |
       v
  TrackedQueryCache (by content hash)
       |
       v
  Invalidation on tracked object change

Reactive Subscriptions:
  Query.subscribe() -> Subscription Registry (weak refs)
  Object Change -> Cache Invalidation -> Re-execute Query -> Diff -> Fire Callbacks
```

## Dependencies

Layer 1 depends on:
- Layer 0: None directly (pure structural)
- stdlib: weakref, threading

Layer 2 depends on:
- Layer 1: registry (for type lookup in queries)
- Layer 0: content_store (for query hashing), eventlog (for change recording)
- stdlib: weakref, threading, hashlib

## Thread Safety Model

All Layer 1-2 modules use the same locking pattern:

```python
class ThreadSafeComponent:
    def __init__(self):
        self._lock = threading.RLock()
    
    def operation(self):
        with self._lock:
            # Critical section
```

RLock chosen over Lock because:
1. Same thread can re-enter (nested calls)
2. Prevents deadlock in recursive scenarios
3. Slightly higher overhead but safer

## Testing Strategy

### Registry Tests
- Register/unregister types
- Type name uniqueness enforcement
- Instance tracking with weak references
- Garbage collection cleanup verification
- Thread safety under concurrent access

### Tracker Tests
- Dirty flag detection on modification
- Transaction commit applies changes
- Transaction rollback reverts changes
- Nested transaction semantics
- Undo/redo correctness
- Thread safety under concurrent modifications

### Query Tests
- Filter correctness (each filter type)
- Filter algebra (AND, OR, NOT, combinations)
- Query algebra (union, intersection, difference)
- Cache hit/miss behavior
- Invalidation on object change
- Subscription callbacks fire correctly
- Content-based hashing produces stable keys
- Thread safety under concurrent queries

### QueryCacheMirror Tests
- Accurate cache statistics
- Query inspection returns correct details
- Invalidation counting accuracy
