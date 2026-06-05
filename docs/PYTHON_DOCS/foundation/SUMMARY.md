# SUMMARY: foundation/

## Metrics

| Metric | Value |
|--------|-------|
| Total Lines | 8,487+ |
| Core Files | 20 |
| ShellLang Files | 4 |
| Total Files | 24 |
| Classes | ~45 |
| Functions | ~200+ |
| Decorators | 5 (@traced, @track_provenance, @require_capability, @provides_view, @register_migration) |

---

## File Breakdown

### foundation/ Core (6,700+ lines)

| File | Lines | Classes | Key Functions |
|------|-------|---------|---------------|
| query.py | 1,042 | 10 | filter(), where(), near(), subscribe() |
| content_store.py | 655 | 6 | store(), fetch(), tree_store(), diff() |
| provenance.py | 441 | 3 | record_read(), derivation_tree(), clear_context() |
| inspector_views.py | 440 | 5 | render(), export() |
| __init__.py | 418 | 0 | Layer exports |
| eventlog.py | 401 | 2 | traced(), log_event(), get_causal_chain() |
| capabilities.py | 363 | 3 | require_capability(), grant(), revoke() |
| serializer.py | 297 | 2 | serialize(), deserialize(), register_type() |
| inspector.py | 282 | 3 | inspect(), register_view() |
| mirror.py | 272 | 2 | reflect(), get_attribute(), invoke() |
| secure_shell.py | 249 | 2 | execute(), validate() |
| bridge.py | 241 | 2 | get_trinity_registry(), sync() |
| tracker.py | 218 | 2 | track(), undo(), redo(), checkpoint() |
| paths.py | 210 | 1 | get_path(), set_path(), resolve() |
| shell.py | 203 | 1 | execute(), repl() |
| delta_sync.py | 203 | 2 | compute_delta(), apply_patch() |
| registry.py | 199 | 1 | register(), lookup(), instances() |
| query_cache_mirror.py | 178 | 1 | stats(), entries(), invalidate() |
| migrations.py | 173 | 2 | register_migration(), migrate(), find_path() |
| constants.py | 106 | 0 | Configuration constants |

### foundation/shelllang/ (1,787 lines)

| File | Lines | Classes | Key Functions |
|------|-------|---------|---------------|
| sugar.py | 541 | 4 | fluent API methods |
| ai.py | 515 | 2 | execute(), parse_command() |
| core.py | 395 | 5 | create(), get(), set(), query() |
| repl.py | 274 | 1 | run(), feedback() |

---

## Algorithm Inventory

| Algorithm | File | Status | Complexity |
|-----------|------|--------|------------|
| Query Filter Algebra | query.py | COMPLETE | O(n) per filter |
| Reactive Subscriptions | query.py | COMPLETE | O(1) subscribe |
| Content Hashing (SHA-256) | content_store.py | COMPLETE | O(n) data size |
| Tree Storage | content_store.py | COMPLETE | O(log n) depth |
| Structural Diffing | content_store.py | COMPLETE | O(nodes) |
| Provenance Capture | provenance.py | COMPLETE | O(1) per read |
| Derivation Tree | provenance.py | COMPLETE | O(edges) |
| Causal Chain | eventlog.py | COMPLETE | O(1) per event |
| Multi-Index Events | eventlog.py | COMPLETE | O(1) insert |
| Capability Grant/Revoke | capabilities.py | COMPLETE | O(1) |
| Context Restriction | capabilities.py | COMPLETE | O(1) |
| BFS Migration Path | migrations.py | COMPLETE | O(V+E) |
| Delta Computation | delta_sync.py | COMPLETE | O(n) |
| Object Reflection | mirror.py | COMPLETE | O(attrs) |
| Undo/Redo Stack | tracker.py | COMPLETE | O(1) push/pop |
| Path Navigation | paths.py | COMPLETE | O(depth) |

---

## Key Evidence Snippets

### Query Algebra (query.py)
- Filter Classes: WhereFilter, NearFilter, HasComponentFilter, AndFilter, OrFilter, NotFilter
- Query Algebra: Union, Intersection, Difference via __or__, __and__, __sub__
- Reactive Subscriptions: on_add/on_remove/on_change callbacks
- TrackedQueryCache: Auto-invalidation on object changes

### Content-Addressable Storage (content_store.py)
- ContentHash: Immutable SHA-256 hash wrapper
- StorageBackend Protocol: MemoryBackend, FileBackend
- Tree Storage: Recursive structural sharing for nested objects
- ContentDiffer: Efficient structural diffing by comparing hashes

### Provenance Tracking (provenance.py)
- @track_provenance: Decorator captures computation inputs
- record_read(): Called by descriptors during field access
- derivation_tree(): Builds full data lineage tree with cycle detection

### Capability Security (capabilities.py)
- Capability: Flag enum (READ, WRITE, CREATE, DELETE, EXECUTE, SPAWN, NETWORK, FILESYSTEM)
- CapabilitySet: Immutable, supports grant/revoke returning new sets
- SecureContext: Context manager with nested restriction
- @require_capability: Decorator for capability enforcement

### ShellLang (shelllang/)
- 5 Primitives: Entity, Component, Query, Mutate, Snapshot
- Sugar Layer: EntityProxy, ComponentProxy, QueryResult with fluent API
- AI Interface: Structured JSON commands (query, set, spawn, destroy, snap, restore, inspect, schema)
- Time Manager: Named snapshots, undo/redo, history
