# PHASE 1 TODO: Layer 0 Essential Primitives

## T-FND-1.1: Mirror Reflection System

### Description
Implement ObjectMirror and ClassMirror for uniform reflection of any Python object.

### Tasks
- [ ] Implement ObjectMirror class with slots inspection
- [ ] Implement ObjectMirror property enumeration
- [ ] Implement ObjectMirror method listing
- [ ] Implement ObjectMirror attribute access (handles __dict__ and __slots__)
- [ ] Implement ClassMirror with inheritance traversal
- [ ] Implement ClassMirror method resolution order inspection
- [ ] Handle descriptor protocol in reflection
- [ ] Add type annotations throughout

### Acceptance Criteria
- ObjectMirror reflects __slots__ classes correctly
- ObjectMirror reflects __dict__ classes correctly
- ObjectMirror handles mixed __slots__ + __dict__ classes
- ClassMirror traverses full MRO
- All public methods have docstrings
- 100% of edge cases covered by tests

---

## T-FND-1.2: Serializer with Schema Versioning

### Description
Implement full serialize/deserialize with circular reference handling and schema versioning.

### Tasks
- [ ] Implement serialize() function with object identity tracking
- [ ] Implement deserialize() function with reference resolution
- [ ] Implement circular reference detection and handling
- [ ] Implement schema version hash computation
- [ ] Implement custom serializer registration API
- [ ] Support dataclasses, namedtuples, enums
- [ ] Support nested collections (list, dict, set)
- [ ] Add __getstate__/__setstate__ protocol support

### Acceptance Criteria
- Circular references serialize without infinite recursion
- Deserialized objects match original structure
- Schema version changes when class definition changes
- Custom serializers override default behavior
- Round-trip tests pass for all supported types

---

## T-FND-1.3: Path Navigation Utilities

### Description
Implement dotted path navigation for nested attribute access.

### Tasks
- [ ] Implement get_path(obj, "a.b.c") function
- [ ] Implement set_path(obj, "a.b.c", value) function
- [ ] Implement has_path(obj, "a.b.c") function
- [ ] Implement delete_path(obj, "a.b.c") function
- [ ] Handle None intermediates gracefully
- [ ] Handle missing attributes with clear errors
- [ ] Support list indices in paths (e.g., "a.0.b")
- [ ] Support dict keys in paths

### Acceptance Criteria
- get_path returns correct nested values
- set_path creates intermediate objects if needed
- has_path returns False for missing paths (no exception)
- delete_path removes attribute and cleans up empty containers
- List index syntax works: "items.0.name"
- Dict key syntax works: "config.settings.debug"

---

## T-FND-1.4: Event Log with Causal Chains

### Description
Implement causal chain tracking with @traced decorator and multiple indexes.

### Tasks
- [ ] Implement Event dataclass with timestamp, entity, operation fields
- [ ] Implement CausalChain with immediate_parent, root_cause, depth
- [ ] Implement @traced decorator for automatic event recording
- [ ] Implement EventLog singleton with add/query methods
- [ ] Implement index by entity ID
- [ ] Implement index by tick number
- [ ] Implement index by operation type
- [ ] Implement index by root cause
- [ ] Use _current_event context variable for nesting

### Acceptance Criteria
- @traced functions auto-record events
- Nested @traced calls create correct causal chain
- All four indexes return correct results
- Context variable isolates concurrent threads
- Event timestamps are monotonically increasing
- Root cause correctly identifies first event in chain

---

## T-FND-1.5: Provenance Tracking

### Description
Track computed value lineage using context variables.

### Tasks
- [ ] Implement @track_provenance decorator
- [ ] Implement record_read(source, field) function
- [ ] Implement derivation_tree(value) function
- [ ] Implement ProvenanceNode dataclass
- [ ] Add cycle detection in derivation graphs
- [ ] Handle transitive dependencies
- [ ] Use _current_tick context variable for temporal ordering
- [ ] Support multiple computation chains in same tick

### Acceptance Criteria
- @track_provenance captures all input reads
- derivation_tree builds complete lineage
- Cycles in derivation graph detected and handled
- Transitive dependencies included in tree
- Thread safety via context variables
- Nested computations create nested provenance

---

## T-FND-1.6: Content-Addressable Storage

### Description
Implement Git-style SHA-256 content storage with structural diffing.

### Tasks
- [ ] Implement ContentHash immutable wrapper
- [ ] Define StorageBackend protocol
- [ ] Implement MemoryBackend for testing
- [ ] Implement FileBackend with hash-based directory structure
- [ ] Implement ContentStore main API (store, retrieve, exists)
- [ ] Implement tree storage with structural sharing
- [ ] Implement ContentDiffer for structural diffs
- [ ] Handle large objects with streaming hashing

### Acceptance Criteria
- Same content always produces same hash
- Different content produces different hash (collision-free for practical use)
- MemoryBackend passes all ContentStore tests
- FileBackend creates correct directory structure
- Identical subtrees share storage (structural sharing)
- ContentDiffer finds minimal diff between trees
- Large objects (>1MB) don't exhaust memory

---

## T-FND-1.7: Delta Sync

### Description
Implement minimal change patches for networking.

### Tasks
- [ ] Implement compute_delta(old_state, new_state) function
- [ ] Implement apply_delta(state, delta) function
- [ ] Implement merge_deltas(delta1, delta2) function
- [ ] Define patch operations: set, delete, append, remove
- [ ] Ensure patches are JSON-serializable
- [ ] Handle nested object changes
- [ ] Optimize for common cases (single field change)

### Acceptance Criteria
- apply_delta(old, compute_delta(old, new)) == new
- Patches are minimal (no redundant operations)
- merge_deltas combines non-conflicting changes
- All patch operations JSON-serializable
- Nested changes produce nested patches
- Empty diff produces empty patch

---

## T-FND-1.8: Schema Migrations

### Description
Implement schema migration path finding with BFS.

### Tasks
- [ ] Implement MigrationRegistry singleton
- [ ] Implement Migration dataclass (from_version, to_version, transform)
- [ ] Implement register_migration() function
- [ ] Implement find_path(from_version, to_version) with BFS
- [ ] Implement migrate(data, from_version, to_version) function
- [ ] Handle missing migration paths with clear error
- [ ] Support migration chains (A -> B -> C)

### Acceptance Criteria
- BFS finds shortest migration path
- Migration chains execute in correct order
- Missing path raises MigrationNotFoundError
- Circular migration definitions handled
- Version numbers can be strings or integers
- migrate() transforms data correctly through chain

---

## T-FND-1.9: Constants Module

### Description
Centralize system-wide configuration constants.

### Tasks
- [ ] Define version constants (FOUNDATION_VERSION)
- [ ] Define default timeouts (DEFAULT_TIMEOUT_MS)
- [ ] Define buffer sizes (DEFAULT_BUFFER_SIZE)
- [ ] Define feature flags (ENABLE_PROVENANCE_TRACKING)
- [ ] Define path conventions (DEFAULT_STORAGE_PATH)
- [ ] Add docstrings explaining each constant's purpose
- [ ] Group constants by category

### Acceptance Criteria
- No magic numbers in other foundation modules
- All constants have descriptive names
- All constants have docstrings
- Constants are importable: from foundation.constants import X
- Reasonable default values for all constants
