# PHASE 1 ARCHITECTURE: Layer 0 Essential Primitives

## Overview

Phase 1 establishes the zero-dependency foundation modules that all higher layers build upon. These modules have no internal dependencies and can be used anywhere in the system.

## Components

### mirror.py (272 lines)

Uniform reflection system for any Python object.

**Classes**:
- `ObjectMirror`: Inspects individual object instances
  - slots, properties, methods, attributes
  - handles __slots__, __dict__, descriptors
- `ClassMirror`: Inspects class definitions
  - inheritance traversal
  - method resolution order
  - class-level attributes

**Design**: Mirrors wrap objects without modifying them. All inspection is read-only.

### serializer.py (297 lines)

Full serialize/deserialize with schema versioning.

**Features**:
- Circular reference handling via object identity tracking
- Schema version hashing for migration compatibility
- Support for custom serializers via registration
- Handles nested objects, collections, dataclasses

**Protocol**: Objects implement `__getstate__`/`__setstate__` or provide registered serializers.

### paths.py (210 lines)

Dotted path navigation utilities.

**Functions**:
- `get_path(obj, "a.b.c")`: Navigate nested attributes
- `set_path(obj, "a.b.c", value)`: Set nested attributes
- `has_path(obj, "a.b.c")`: Check path existence
- `delete_path(obj, "a.b.c")`: Remove attribute

**Edge Cases**: Handles None intermediates, missing attributes, list indices in paths.

### eventlog.py (401 lines)

Causal chain tracking with @traced decorator.

**Components**:
- `Event`: Recorded operation with timestamp, entity, operation type
- `CausalChain`: Links events via immediate_parent, root_cause, depth
- `@traced`: Decorator auto-records function calls as events
- `EventLog`: Singleton storage with multiple indexes

**Indexes**:
- By entity ID
- By tick number
- By operation type
- By root cause

**Context**: Uses `_current_event` context variable for thread-safe nesting.

### provenance.py (441 lines)

Computed value lineage tracking via context variables.

**Mechanism**:
- `@track_provenance`: Decorator captures computation inputs
- `record_read(source, field)`: Called by descriptors during field access
- `derivation_tree(value)`: Builds full lineage tree

**Features**:
- Cycle detection in derivation graphs
- Handles transitive dependencies
- Context variable `_current_tick` for temporal ordering

### content_store.py (655 lines)

Content-addressable storage with structural diffing.

**Classes**:
- `ContentHash`: Immutable SHA-256 hash wrapper
- `StorageBackend`: Protocol for storage implementations
- `MemoryBackend`: In-memory storage for testing
- `FileBackend`: Filesystem storage with hash-based paths
- `ContentStore`: Main API for storing/retrieving content
- `ContentDiffer`: Computes diffs between trees by hash comparison

**Tree Storage**: Recursive structural sharing for nested objects. Identical subtrees share storage.

### delta_sync.py (203 lines)

Minimal change patches for networking.

**Operations**:
- `compute_delta(old_state, new_state)`: Generate patch
- `apply_delta(state, delta)`: Apply patch
- `merge_deltas(delta1, delta2)`: Combine patches

**Patch Format**: JSON-serializable operations (set, delete, append, remove).

### migrations.py (173 lines)

Schema migration path finding.

**Components**:
- `MigrationRegistry`: Stores registered migrations
- `Migration`: Single version-to-version transform
- `find_path(from_version, to_version)`: BFS path finding
- `migrate(data, from_version, to_version)`: Apply migration chain

**Algorithm**: BFS ensures shortest migration path.

### constants.py (106 lines)

System-wide configuration constants.

**Categories**:
- Version constants
- Default timeouts
- Buffer sizes
- Feature flags
- Path conventions

**Purpose**: Centralized configuration avoids magic numbers throughout codebase.

## Data Flow

```
Object -> Mirror -> Serializer -> ContentStore
                        |
                        v
                   ContentHash
                        |
                        v
                   StorageBackend
```

Events flow:
```
@traced function -> EventLog -> CausalChain
                        |
                        v
                   Multiple Indexes
```

Provenance flow:
```
Computation -> @track_provenance -> record_read() -> DerivationTree
```

## Dependencies

Layer 0 modules depend only on Python stdlib:
- typing
- hashlib
- contextvars
- weakref
- dataclasses
- json
- pathlib

No internal foundation dependencies.

## Testing Strategy

Each module testable in isolation:
- mirror: Test reflection on various object types (slots, properties, dynamic)
- serializer: Round-trip tests with circular references
- paths: Navigation tests with edge cases (None, missing, indices)
- eventlog: Causal chain verification, index correctness
- provenance: Lineage accuracy, cycle detection
- content_store: Hash uniqueness, structural sharing verification
- delta_sync: Patch correctness, idempotence
- migrations: Path finding correctness, migration ordering
