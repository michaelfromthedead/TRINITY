# ENGINE_WORLD - Phase 1 Architecture: Foundation Validation

## Phase Overview

Phase 1 establishes confidence that the existing 29,000+ lines of world subsystem code are correct, consistent, and ready for integration work. This is audit and cleanup, not new development.

## Architecture Decisions

### ADR-W1-001: Constants Audit

**Context**: Each subsystem has a `constants.py` file, but magic numbers may still exist in implementation files.

**Decision**: Audit all files, extract any remaining magic numbers to `constants.py`, ensure all constants have documentation.

**Rationale**: Centralized constants enable tuning without code archaeology. Documentation prevents "what does 0.5 mean?" questions.

### ADR-W1-002: Type Hint Completeness

**Context**: Public APIs have type hints, but internal functions may lack them.

**Decision**: Add type hints to all functions, not just public APIs.

**Rationale**: Type hints enable static analysis, improve IDE experience, and serve as documentation. Cost is low for existing code.

### ADR-W1-003: Protocol Verification

**Context**: Several `Protocol` classes define contracts but implementations may drift.

**Decision**: Add `runtime_checkable` decorator to all Protocols, add validation at integration boundaries.

**Rationale**: Protocols are only useful if implementations actually satisfy them. Runtime checks catch drift early.

### ADR-W1-004: Stub Removal

**Context**: Investigation found no stubs, but "demonstration" and "simplified" comments exist.

**Decision**: Audit all such comments, document known limitations, mark as technical debt if functionality is incomplete.

**Rationale**: Honest documentation prevents surprise failures. Known limitations can be scheduled for improvement.

## Component Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    engine/world/                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    queries    ┌─────────┐                     │
│  │ terrain │ <──────────── │ queries │                     │
│  └────┬────┘               └────┬────┘                     │
│       │                         │                          │
│       │ height_at               │ terrain_raycast          │
│       │ normal_at               │ path_query               │
│       │                         │                          │
│       ▼                         ▼                          │
│  ┌─────────┐               ┌─────────┐                     │
│  │ foliage │               │partition│                     │
│  └────┬────┘               └────┬────┘                     │
│       │                         │                          │
│       │ placement               │ cell state               │
│       │                         │                          │
│       ▼                         ▼                          │
│  ┌─────────┐               ┌─────────┐                     │
│  │   pcg   │               │  hlod   │                     │
│  └─────────┘               └─────────┘                     │
│                                                             │
│  ┌─────────────────────────────────────┐                   │
│  │           environment               │                   │
│  │  (sky, weather, time, lighting)     │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## File Organization

Each subsystem follows the same structure:
```
engine/world/<subsystem>/
├── __init__.py      # Module exports, docstring
├── constants.py     # All magic numbers
├── <core>.py        # Main implementation
└── <auxiliary>.py   # Supporting types
```

## Validation Strategy

### Unit Test Requirements

Each subsystem needs tests covering:
1. **Construction**: Objects create with valid defaults
2. **Configuration**: Invalid configs raise appropriate errors
3. **Core Operations**: Primary use cases work correctly
4. **Edge Cases**: Boundary conditions handled

### Test File Locations

```
tests/unit/world/
├── test_terrain.py
├── test_environment.py
├── test_foliage.py
├── test_hlod.py
├── test_partition.py
├── test_pcg.py
└── test_queries.py
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Type hint coverage | 100% of functions |
| Constants extracted | 100% of magic numbers |
| Protocol decorators | All Protocols `@runtime_checkable` |
| Unit test files | 7 (one per subsystem) |
| Known limitations documented | All "simplified"/"demonstration" comments addressed |
