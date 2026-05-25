# PHASE 2 TODO: Layers 1-2 Structural and Reactive

## T-FND-2.1: Type Registry

### Description
Implement unified type registry for tracking all registered types.

### Tasks
- [ ] Implement TypeRegistry singleton class
- [ ] Implement register(cls) method
- [ ] Implement unregister(cls) method
- [ ] Implement get(name) method for type lookup
- [ ] Implement all_types() method
- [ ] Enforce type name uniqueness
- [ ] Add RLock for thread safety
- [ ] Create module-level registry instance

### Acceptance Criteria
- Types registered by name are retrievable by name
- Duplicate registration raises RegistrationError
- unregister removes type from registry
- all_types returns list of all registered type names
- Thread-safe under concurrent registration
- Singleton pattern enforced

---

## T-FND-2.2: Instance Tracker

### Description
Track live instances per registered type using weak references.

### Tasks
- [ ] Implement InstanceTracker class
- [ ] Use WeakSet for instance storage
- [ ] Implement track_instance(obj) method
- [ ] Implement untrack_instance(obj) method
- [ ] Implement instances(cls) method returning live instances
- [ ] Implement instance_count(cls) method
- [ ] Automatic cleanup on garbage collection
- [ ] Integration with TypeRegistry

### Acceptance Criteria
- Tracked instances retrievable via instances(cls)
- Garbage-collected objects automatically removed from tracking
- instance_count returns accurate count
- Weak references don't prevent garbage collection
- Thread-safe under concurrent tracking

---

## T-FND-2.3: Change Tracker

### Description
Implement change detection with dirty flags and proxy wrappers.

### Tasks
- [ ] Implement ChangeTracker singleton class
- [ ] Implement track(obj) method that wraps object
- [ ] Implement untrack(obj) method
- [ ] Implement is_dirty(obj) method
- [ ] Implement clear_dirty(obj) method
- [ ] Create proxy wrapper that intercepts __setattr__
- [ ] Record old values before modification
- [ ] Add RLock for thread safety

### Acceptance Criteria
- Tracked objects report dirty after modification
- clear_dirty resets flag
- Untracked objects no longer monitored
- Old values recorded for undo support
- Multiple fields tracked independently
- Thread-safe under concurrent modifications

---

## T-FND-2.4: Transactions

### Description
Implement atomic transaction support with commit/rollback.

### Tasks
- [ ] Implement Transaction class as context manager
- [ ] Implement begin() method
- [ ] Implement commit() method
- [ ] Implement rollback() method
- [ ] Support nested transactions
- [ ] Inner rollback doesn't affect outer transaction
- [ ] Track changes per transaction level
- [ ] Integrate with ChangeTracker

### Acceptance Criteria
- Changes visible only after commit
- rollback reverts all changes in transaction
- Nested transactions work correctly
- Inner rollback allows outer to continue
- Exception triggers automatic rollback
- with Transaction() as t: syntax works

---

## T-FND-2.5: Undo/Redo Stack

### Description
Implement undo/redo history for tracked changes.

### Tasks
- [ ] Implement UndoStack class
- [ ] Implement push(change) method
- [ ] Implement undo() method
- [ ] Implement redo() method
- [ ] Implement clear() method
- [ ] Implement can_undo() and can_redo() methods
- [ ] Limit stack size (configurable)
- [ ] Clear redo stack on new change

### Acceptance Criteria
- undo reverts last change
- redo reapplies reverted change
- Multiple undo/redo steps work
- New change clears redo stack
- Stack respects size limit
- can_undo/can_redo report correctly

---

## T-FND-2.6: Query Filters

### Description
Implement filter classes for query predicates.

### Tasks
- [ ] Define Filter protocol with matches(entity, world) method
- [ ] Implement WhereFilter for field predicates
- [ ] Support operators: eq, ne, lt, le, gt, ge, in, contains
- [ ] Implement NearFilter for spatial proximity
- [ ] Implement HasComponentFilter for type checks
- [ ] Implement AndFilter combining multiple filters
- [ ] Implement OrFilter combining multiple filters
- [ ] Implement NotFilter for negation

### Acceptance Criteria
- WhereFilter matches correct entities
- All comparison operators work
- NearFilter uses correct distance calculation
- HasComponentFilter checks component presence
- AndFilter requires all filters match
- OrFilter requires any filter match
- NotFilter inverts result
- Filters are composable

---

## T-FND-2.7: Query Class

### Description
Implement first-class query objects with filter chaining.

### Tasks
- [ ] Implement Query class constructor taking ComponentTypes
- [ ] Implement where() method for field predicates
- [ ] Implement near() method for spatial queries
- [ ] Implement has() method for component checks
- [ ] Implement __call__(world) for execution
- [ ] Implement __or__ for query union
- [ ] Implement __and__ for query intersection
- [ ] Implement __sub__ for query difference
- [ ] Implement content-based __hash__

### Acceptance Criteria
- Query execution returns matching entities
- Filter chaining works: query.where().near().has()
- Union combines results from both queries
- Intersection returns common results
- Difference excludes second query results
- Same query structure produces same hash

---

## T-FND-2.8: Query Subscriptions

### Description
Implement reactive subscriptions for query result changes.

### Tasks
- [ ] Implement subscribe(on_add, on_remove, on_change) method
- [ ] Implement unsubscribe(subscription_id) method
- [ ] Store subscriptions as weak references
- [ ] Fire on_add when new entity matches query
- [ ] Fire on_remove when entity no longer matches
- [ ] Fire on_change when matched entity changes
- [ ] Automatic cleanup of dead subscriptions

### Acceptance Criteria
- on_add fires for new matches
- on_remove fires when entity no longer matches
- on_change fires when matched entity modified
- Dead callbacks automatically cleaned up
- Subscription returns ID for unsubscribe
- Multiple subscriptions per query supported

---

## T-FND-2.9: Query Cache

### Description
Implement TrackedQueryCache with auto-invalidation.

### Tasks
- [ ] Implement TrackedQueryCache class
- [ ] Cache results by query content hash
- [ ] Implement get(query) method
- [ ] Implement put(query, results) method
- [ ] Implement invalidate(query) method
- [ ] Auto-invalidate when tracked objects change
- [ ] Add RLock for thread safety
- [ ] Track cache statistics (hits, misses)

### Acceptance Criteria
- Cache hit returns stored results
- Cache miss returns None
- Object change invalidates relevant queries
- Statistics accurately track hits/misses
- Thread-safe under concurrent access
- Cache respects size limits

---

## T-FND-2.10: Query Cache Mirror

### Description
Implement meta-inspection of query cache state.

### Tasks
- [ ] Implement QueryCacheMirror class
- [ ] Implement cache_size() method
- [ ] Implement hit_rate() method
- [ ] Implement cached_queries() method
- [ ] Implement inspect_query(hash) method
- [ ] Implement invalidation_count() method
- [ ] Expose cache performance metrics

### Acceptance Criteria
- cache_size returns correct count
- hit_rate calculates correctly
- cached_queries lists all cached query hashes
- inspect_query returns query details
- invalidation_count tracks total invalidations
- All metrics update in real-time
