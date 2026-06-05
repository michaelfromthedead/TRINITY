# GAPSET_20_CROSS_CUTTING — Phase Architecture Documents

This file contains the architecture plan for each implementation phase.
Each phase corresponds to filling one or more identified cross-cutting gaps.

---

## Phase 0: Common Utilities

| Field | Value |
|-------|-------|
| **Task ID** | T-CC-0.x → T-CC-4.x (corrected) |
| **Priority** | CRITICAL |
| **Current State** | MISSING — `engine/common/` has empty directories |
| **Target** | Shared utility functions and common type definitions |

### Architecture

```
engine/common/
├── __init__.py          # Re-export all
├── constants/
│   ├── __init__.py      # Version info, build flags, platform IDs
│   ├── paths.py         # Standardized path constants
│   └── limits.py        # Shared limit/max constants
├── types/
│   ├── __init__.py      # Re-export
│   ├── result.py        # Result<T, E> pattern (Rust-inspired)
│   ├── maybe.py         # Maybe/Optional pattern
│   └── identifiers.py   # UUID, EntityID, AssetID types
└── utils/
    ├── __init__.py
    ├── math.py           # Shared math helpers
    ├── hashing.py        # Consistent hash utilities
    ├── time.py           # Time conversion utilities
    └── serialization.py  # Shared serialization primitives
```

### Key Decisions
- Use Python `dataclasses` for types (consistent with trinity pattern)
- Implement `Result` type similar to Rust's `Result<T, E>` for error propagation
- All utilities must be pure Python (no external dependencies)
- Test coverage target: 95%+

### Integration Points
- All engine modules will import from `engine.common`
- GAPSET_3_BRIDGE: Rust bridge types will mirror these

---

## Phase 1: Serialization Runtime

| Field | Value |
|-------|-------|
| **Task ID** | T-CC-1.x |
| **Priority** | HIGH |
| **Current State** | PARTIAL — descriptors exist, runtime missing |
| **Target** | Full binary serialization format with schema validation |

### Architecture

```
engine/
└── serialization/
    ├── __init__.py
    ├── format.py            # Binary format specification
    ├── encoder.py           # Encode Python objects → binary
    ├── decoder.py           # Decode binary → Python objects
    ├── schema.py            # Schema validation engine
    ├── registry.py          # Type registry for serializable classes
    ├── migrations.py        # Schema migration support
    └── bridges/
        ├── __init__.py
        ├── json_bridge.py   # JSON format compat
        ├── protobuf_bridge.py # Protobuf integration (future)
        └── flatbuffers_bridge.py # FlatBuffers integration (future)
```

### Key Decisions
- Compact binary format with schema headers
- Versioned schemas with forward/backward compatibility
- `@serializable` decorator in trinity should wire into this runtime
- Rust bridge: protobuf or flatbuffers for cross-language data interchange

### Dependencies
- Phase 0: Result type, hashing utilities
- Tests: `tests/serialization/`

---

## Phase 2: Plugin/Mod Runtime

| Field | Value |
|-------|-------|
| **Task ID** | T-CC-2.x |
| **Priority** | HIGH |
| **Current State** | STUB — 9 modding decorators, no runtime |
| **Target** | WASM-based plugin sandbox with mod loading, isolation, and lifecycle |

### Architecture

```
engine/
└── plugins/
    ├── __init__.py
    ├── loader.py            # Module loader (importlib-based)
    ├── sandbox.py           # WASM sandbox via wasmtime-py
    ├── isolation.py         # Filesystem/network isolation
    ├── lifecycle.py         # Plugin lifecycle (init, run, shutdown)
    ├── registry.py          # Plugin registry
    ├── dependencies.py      # Dependency resolution (DAG)
    └── events.py            # Plugin event hooks
```

### Key Decisions
- `@mod` decorator in trinity should register with this runtime
- Load ordering from `@load_order` decorator feeds into DAG resolution
- Sandbox uses WASM for untrusted plugin code
- Python-native plugins use restricted import via `isolation.py`
- GAPSET_3_BRIDGE: WASM runtime runs in Rust via wasmtime

### Dependencies
- Phase 0: Result type
- Phase 1: Serialization for plugin config/save data
- Tests: `tests/plugins/`

---

## Phase 3: Hot-Reload Integration

| Field | Value |
|-------|-------|
| **Task ID** | T-CC-3.x |
| **Priority** | MEDIUM |
| **Current State** | PARTIAL — file watcher works, not integrated |
| **Target** | Wire file_watcher → engine runtime for code/asset hot-reload |

### Architecture

```
engine/
└── hotreload/
    ├── __init__.py
    ├── watcher.py           # Wraps engine/platform/os/file_watcher.py
    ├── code_reloader.py     # Python module reload logic
    ├── asset_reloader.py    # Asset reload (textures, shaders, etc.)
    ├── state_preserver.py   # State preservation across reload
    └── scheduler.py         # Debounced reload scheduling
```

### Key Decisions
- `@reloadable` decorator wires into this runtime
- Use `importlib.reload()` for Python code hot-reload
- Asset reload is subsystem-specific (each asset type needs its own handler)
- State preservation: mark fields with `@preserve` decorator
- Debounce file events to avoid rapid-reload thrashing

### Dependencies
- Phase 0: Result type
- Existing: `engine/platform/os/file_watcher.py`
- Tests: `tests/hotreload/`

---

## Phase 4: Security Runtime

| Field | Value |
|-------|-------|
| **Task ID** | T-CC-4.x |
| **Priority** | HIGH |
| **Current State** | STUB — 4 decorators, no enforcement |
| **Target** | Authorization engine, rate limiter, input validation, encryption |

### Architecture

```
engine/
└── security/
    ├── __init__.py
    ├── authz.py             # Authorization engine
    ├── rate_limiter.py      # Token bucket rate limiter
    ├── validator.py         # Input validation runtime
    ├── encryption.py        # Encryption/decryption utilities
    ├── audit.py             # Security audit log
    └── middleware.py         # Integration hooks for engine pipeline
```

### Key Decisions
- `@server_authoritative` → authz check in engine/system pipeline
- `@validated` → input validation rules evaluated at runtime
- `@rate_limited` → token bucket algorithm
- `@encrypted` → field-level encryption with key management
- Use Python `cryptography` library for encryption
- All security checks log to `engine/debug/logging/`

### Dependencies
- Phase 0: Result type
- Existing: `engine/debug/logging/`
- Tests: `tests/security/`

---

## Phase 5: Save System Runtime

| Field | Value |
|-------|-------|
| **Task ID** | T-CC-5.x |
| **Priority** | MEDIUM |
| **Current State** | STUB — 4 decorators, no implementation |
| **Target** | Full save/load system with slots, atomic writes, cloud sync |

### Architecture

```
engine/
└── saves/
    ├── __init__.py
    ├── manager.py           # Save slot manager
    ├── atomic.py            # Atomic write-then-rename
    ├── migration.py         # Save file migration engine
    ├── cloud_sync.py        # Cloud synchronization
    ├── compression.py       # Optional compression
    └── format.py            # Save file format spec
```

### Key Decisions
- `@save_slot` → metadata for save manager
- `@atomic_save` → write to temp file, rename on success
- `@save_migration` → version-checked migration pipeline
- `@cloud_sync` → platform-specific cloud storage adapter
- Save format uses Phase 1 serialization

### Dependencies
- Phase 0: Result type
- Phase 1: Serialization
- Tests: `tests/saves/`

---

## Phase 6: Error Handling Runtime

| Field | Value |
|-------|-------|
| **Task ID** | T-CC-6.x |
| **Priority** | MEDIUM |
| **Current State** | PARTIAL — 4 decorators with runtime attributes only |
| **Target** | Retry logic, fallback execution, circuit-breaker |

### Architecture

```
engine/
└── errors/
    ├── __init__.py
    ├── retry.py             # Retry with backoff (exponential, jitter)
    ├── fallback.py          # Fallback execution chain
    ├── circuit_breaker.py   # Circuit breaker pattern
    ├── recovery.py          # State recovery engine
    ├── boundary.py          # Error isolation boundary
    └── reporting.py         # Error reporting (wraps crash reporter)
```

### Key Decisions
- `@crash_safe` recovery strategies: retry, skip, fallback, crash
- `@recoverable` checkpoint-based recovery
- `@error_boundary` scope-based isolation (system, entity, global)
- `@bug_report` auto-generate diagnostic data
- Integrate with `engine/debug/crash/` for fatal errors
- Integrate with `engine/debug/logging/` for error logging

### Dependencies
- Phase 0: Result type
- Existing: `engine/debug/crash/`, `engine/debug/logging/`
- Tests: `tests/errors/`

---

## Phase 7: Build System Unification

| Field | Value |
|-------|-------|
| **Task ID** | T-CC-7.x |
| **Priority** | HIGH |
| **Current State** | PARTIAL — Rust workspace exists, no unified build |
| **Target** | Single command build, CI/CD pipeline, cross-language test harness |

### Architecture

```
ci/
├── build.sh                # Unified build script
├── test.sh                 # Unified test runner
├── lint.sh                 # Cross-language linting
├── Dockerfile              # CI container definition
└── github/
    ├── build.yml           # GitHub Actions: build
    ├── test.yml            # GitHub Actions: test
    ├── lint.yml            # GitHub Actions: lint
    └── release.yml         # GitHub Actions: release
```

### Key Decisions
- Single `make build` / `make test` commands for both Python and Rust
- Python: `pytest` with coverage, linting via `ruff`
- Rust: `cargo test`, lint via `clippy`
- CI: GitHub Actions with matrix builds (Linux, macOS, Windows)
- Test results aggregated into unified report
- Coverage: Python `coverage.py` + Rust `tarpaulin`
- Rust bridge: workspace members build before Python tests

### Dependencies
- All phases
- Tests: Infrastructure testing (CI pipeline tests)

---

## Dependency Graph Across Phases

```
Phase 0 (Common Utilities)
  ├── Phase 1 (Serialization) ──────┐
  │   ├── Phase 2 (Plugin Runtime) ──┤
  │   └── Phase 5 (Save System) ─────┤
  ├── Phase 3 (Hot-Reload) ──────────┤
  ├── Phase 4 (Security) ────────────┤
  └── Phase 6 (Error Handling) ──────┤
                                      │
                                      ▼
                              Phase 7 (Build System)
```

Phases 1-6 can be built in parallel once Phase 0 is complete.
Phase 7 (Build System) depends on all phases being complete for test integration.
