# CLARIFICATION: Foundation Design Rationale

## Philosophical Framing

### Why Layered Architecture?

Foundation explicitly organizes modules into layers (0-4) with strict dependency rules. This addresses:

1. **Dependency Hell Prevention**: Lower layers cannot accidentally depend on higher layers
2. **Testability**: Each layer can be tested in isolation with mocked upper layers
3. **Bootstrapping Order**: System initialization follows layer order naturally
4. **Cognitive Load**: Developers know which modules are "safe" to use at each level

The layer numbers (0-4) are not arbitrary. Layer 0 modules have zero internal dependencies and can be used anywhere. Layer 4 (Integration) depends on Trinity itself and represents the boundary between foundation and the component system.

### Why Content-Addressable Storage?

The content_store module implements Git-style SHA-256 hashing because:

1. **Immutability Guarantees**: Content hash proves data hasn't changed
2. **Structural Sharing**: Identical subtrees share storage automatically
3. **Efficient Diffing**: Compare hashes before comparing content
4. **Network Sync**: Send only changed content (hash as identity)
5. **Audit Trail**: Every state version is recoverable by hash

This enables time-travel debugging where any historical state can be reconstructed from its content hash.

### Why Capability-Based Security?

Traditional permission systems (user/group/other) don't map well to game engine runtime:

1. **Granular Control**: Each operation type (READ, WRITE, EXECUTE, SPAWN, NETWORK, FILESYSTEM) is a separate capability
2. **Explicit Delegation**: Capabilities are passed explicitly, not inherited implicitly
3. **Immutable Sets**: CapabilitySet.grant() returns a new set, preventing mutation bugs
4. **Context Scoping**: SecureContext allows temporary capability restriction
5. **Auditability**: Capability requirements are documented via @require_capability

This prevents privilege escalation in user-provided scripts and AI-generated code.

### Why Context Variables?

The foundation uses Python's contextvars extensively for:

- _current_event (eventlog)
- _current_tick (provenance)
- _current_capabilities (security)

Reasons:

1. **Thread Safety Without Locking**: Each thread/async task gets its own copy
2. **Implicit Context Propagation**: Nested calls inherit context automatically
3. **No Global State Pollution**: Context is scoped, not global
4. **Decorator Integration**: @traced and @track_provenance work via context

### Why Dual Interface (Human DSL + AI JSON)?

ShellLang provides two interfaces to the same ECS operations:

**Human Interface (sugar.py)**:
```python
enemies.where(lambda e: e.health.current < 50).near(player, 10).set(health__current=0)
```

**AI Interface (ai.py)**:
```json
{"op": "set", "entity": 42, "component": "Health", "field": "current", "value": 0}
```

Design rationale:

1. **Same Semantics**: Both interfaces execute identical operations
2. **Validation Parity**: AI commands go through same validation as DSL
3. **Structured Errors**: AI gets machine-readable error responses
4. **Dry Run**: AI can preview effects without committing
5. **Schema Discovery**: AI can query available types and fields

This enables AI agents to manipulate game state safely while humans use ergonomic syntax.

## Design Decisions

### Singleton Instances

Foundation creates singleton instances for:
- `registry`: Type registry
- `tracker`: Change tracker
- `inspector`: Object visualizer
- `shell`: Interactive REPL
- `_event_log`: Causal chain logger
- `_migration_registry`: Schema migrations

Rationale: These are coordination points that must be globally consistent. Multiple trackers would miss cross-tracker changes. Multiple registries would have inconsistent type views.

### Weak References in Instance Tracking

The registry uses weak references to track instances because:

1. **No Memory Leaks**: Tracked objects can be garbage collected normally
2. **Automatic Cleanup**: Dead references are pruned on access
3. **Query Correctness**: Queries never return dead objects

### Content-Based Query Hashing

Query objects hash by their content (filters, component types) not by identity:

1. **Cache Key Stability**: Same query structure = same cache key
2. **Deduplication**: Identical queries share cache entries
3. **Deterministic**: Hash is reproducible across sessions

### Protocol/ABC for Extensibility

Foundation uses Protocol and ABC for extension points:

- `View` protocol for inspector plugins
- `UIContext` protocol for rendering
- `Filter` protocol for query filters
- `StorageBackend` protocol for content store backends

This allows users to extend foundation without modifying core code.

## Trade-offs

### Thread Safety vs Performance

Foundation uses RLock in critical sections (Tracker, Registry, QueryCache). This adds overhead but prevents data races. The trade-off favors correctness over raw speed because:

1. Race conditions in game state cause non-deterministic bugs
2. Most operations are fast (in-memory), so lock overhead is small
3. Coarse-grained locks are simpler than fine-grained (fewer deadlocks)

### Immutability vs Convenience

CapabilitySet is immutable (grant/revoke return new sets). This adds allocation overhead but prevents:

1. Capability escalation via mutation
2. Shared-state bugs across contexts
3. Audit log inconsistency

### Complete Snapshots vs Delta Snapshots

Snapshot captures full world state, not deltas. This trades space for simplicity:

1. Any snapshot is independently restorable
2. No dependency chains between snapshots
3. Diff computed on-demand via world.diff(a, b)

## Open Questions

### Q: Should query cache invalidation be lazy or eager?

Current: Eager invalidation on any tracked change.
Alternative: Lazy invalidation with version numbers.
Trade-off: Eager guarantees correctness but may over-invalidate.

### Q: Should ShellLang support transactions?

Current: Individual operations with undo/redo.
Alternative: Explicit transaction blocks with atomic commit.
Trade-off: Transactions add complexity but enable batch rollback.

### Q: Should provenance track function source code?

Current: Tracks input values and call graph.
Alternative: Include function bytecode hash.
Trade-off: Bytecode changes more than semantics (reformatting).
