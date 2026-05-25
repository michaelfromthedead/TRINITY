# GAPSET_20_CROSS_CUTTING — Project Overview

**Owner:** RDC Worker
**Status:** MIXED (see per-domain breakdown)
**Total Lines of Code:** ~13,000+ (across all cross-cutting domains)
**RDC run:** 2026-05-22

---

## 1. Goal

Implement and integrate all cross-cutting concerns that span every layer of the engine:
error handling, logging, profiling, serialization, plugin system, hot-reloading,
localization, accessibility, platform abstraction, security, save system,
configuration, common utilities, and build system integration.

## 2. Implementation Status

### REAL (fully implemented with tests)
- [x] **Logging** — `engine/debug/logging/` (Logger, Sinks, Filters, StructuredLog, Rotation) — ~1,800 lines, full test coverage
- [x] **Profiling** — `engine/debug/profiling/` (CPU, GPU, Memory, Network, Stats) — ~2,500 lines, full test coverage
- [x] **Crash Handling** — `engine/debug/crash/` (assertions, handler, minidump, reporter) — ~1,500 lines, full test coverage
- [x] **Platform Abstraction** — `engine/platform/` (os, input, window, gpu, rhi, audio, services) — ~5,000+ lines, partial test coverage
- [x] **Configuration** — `engine/core/constants.py` + `trinity/constants.py` + `engine/platform/constants.py` — ~250 lines, centralized and consistent

### PARTIAL (infrastructure exists, runtime missing)
- [~] **Error Handling** — `trinity/decorators/error_handling.py`: 4 decorators defined (`@crash_safe`, `@recoverable`, `@error_boundary`, `@bug_report`) — all tag-only (Op-based steps). No actual retry logic, fallback execution, or circuit-breaker runtime.
- [~] **Serialization** — `trinity/descriptors/persistence.py`: SerializableDescriptor with custom encode/decode. SchemaDescriptor validates types. Format is "binary" with no implementation. Partial test coverage.
- [~] **Hot-Reloading** — `engine/platform/os/file_watcher.py`: polling-based watcher works. `trinity/decorators/dev.py`: `@reloadable` is tag-only. `engine/tooling/hotreload/`: tool-level hot reload exists but not integrated across engine.
- [~] **Build System** — Rust/Cargo workspace defined in `/Cargo.toml`. Python projects in `omega/`, `flowforge/`. No unified build, no CI/CD pipeline, no automated test harness across languages.

### STUB (tag/decorator metadata only)
- [-] **Plugin System** — `trinity/decorators/modding.py`: 9 decorators (`@mod`, `@requires`, `@conflicts`, `@provides`, `@replaces`, `@mod_extends`, `@patch`, `@load_order`, `@moddable`). All tag-only. No mod loader, WASM sandbox, or plugin isolation.
- [-] **Security** — `trinity/decorators/security.py`: 4 decorators (`@server_authoritative`, `@validated`, `@rate_limited`, `@encrypted`). All tag-only. No runtime enforcement. No tests.
- [-] **Save System** — `trinity/decorators/save_system.py`: 4 decorators (`@save_slot`, `@atomic_save`, `@cloud_sync`, `@save_migration`). All tag-only. No runtime save/load. No tests.
- [-] **Localization (decorator)** — `trinity/decorators/localization.py`: 4 decorators (`@localized`, `@plural`, `@rtl_aware`, `@text_overflow`). Tag-only. NOTE: engine layer (`engine/ui/text/localization.py`) IS real.
- [-] **Accessibility (decorator)** — `trinity/decorators/accessibility.py`: 1 decorator (`@accessible`). Tag-only. NOTE: engine layer (`engine/ui/accessibility/`) IS real.

### MISSING
- [-] **Common Utilities** — `engine/common/constants/`, `engine/common/types/`, `engine/common/utils/`: All directories exist but are EMPTY (zero code). No shared utility functions, no common type definitions.

## 3. Key Observations

### Architecture Pattern
All cross-cutting concerns follow a dual-layer architecture:
1. **Trinity decorator layer** (`trinity/decorators/`): Adds metadata tags via `make_decorator()` + Op-based steps
2. **Engine runtime layer** (`engine/`): Actual implementation

In many cases the engine layer is complete while the decorator layer is stub (localization, accessibility). In other cases only the decorator layer exists (security, save system). Common Utilities has neither.

### Dependency Order

```
engine/debug/profiling/ ──┐
engine/debug/logging/ ────┤
engine/debug/crash/ ──────┤
engine/core/constants/ ────┼──→ engine/core/engine.py
engine/common/ (MISSING) ──┘
         │
         └──→ engine/platform/ ──→ engine/debug/testing/
                        │
                        └──→ engine/ui/ (accessibility, text/localization)
```

## 4. Integration Points

### Upstream Dependencies
- `engine/core/constants.py` — all debug subsystems import from here
- `engine/debug/logging/` — used by all engine modules

### Downstream Consumers
- `engine/core/engine.py` — consumes logging, profiling, crash handling
- `engine/debug/testing/` — consumes profiling, logging
- `engine/tooling/` — consumes all debug subsystems

### GAPSET_3_BRIDGE Connections
- **Serialization**: protobuf/flatbuffers bridge needed for Rust↔Python data interchange
- **Error Handling**: Rust Result types need Python bridge with structured error propagation
- **Platform Abstraction**: Rust OS API bindings for file system, threading, memory management
- **Build System**: Unified cargo+pip build pipeline

## 5. GRANDPHASE2 Recommendations

### High Priority
1. Implement `engine/common/` — shared utilities, types, constants
2. Serialization runtime — binary format, schema engine, Rust bridge
3. Plugin runtime — WASM sandbox, mod loader
4. Security runtime — authorization checks, rate limiting, encryption

### Medium Priority
5. Error Handling runtime — retry with backoff, circuit-breaker pattern
6. Save System runtime — atomic save, slot management, migration
7. Build system unification — CI/CD pipeline, cross-language test harness

### Low Priority
8. Hot-Reload full integration — wire file_watcher → runtime reload
9. Decorator ↔ Engine wiring for localization and accessibility
