# GAPSET_20_CROSS_CUTTING — Comprehensive Summary

**Date:** 2026-05-22
**RDC Worker:** Single-agent investigation (full codebase walk)
**Scope:** All cross-cutting concerns spanning engine layers

---

## Cross-Cutting Dimensions Inventory (15 domains)

| # | Domain | Location | Lines | Status | Tests |
|---|--------|----------|-------|--------|-------|
| 1 | Error Handling | `trinity/decorators/error_handling.py` (Tier 40) | ~217 | PARTIAL/STUB | `tests/trinity/test_error_handling.py` |
| 2 | Logging | `engine/debug/logging/` | ~1,800 | REAL | `tests/debug/logging/`, `tests/tooling/logging/` |
| 3 | Profiling | `engine/debug/profiling/` + `trinity/decorators/dev.py` | ~2,500 | REAL | `tests/debug/profiling/`, `tests/tooling/profiling/` |
| 4 | Crash Handling | `engine/debug/crash/` | ~1,500 | REAL | `tests/debug/crash/`, `tests/tooling/crash/` |
| 5 | Serialization | `trinity/descriptors/persistence.py`, `trinity/descriptors/schema.py` | ~300 | PARTIAL | `tests/trinity/test_persistence_descriptors.py` |
| 6 | Plugin System | `trinity/decorators/modding.py` (Tier 30) | ~422 | STUB | `tests/tooling/editor/test_plugins.py` |
| 7 | Hot-Reloading | `engine/platform/os/file_watcher.py`, `trinity/decorators/dev.py`, `engine/tooling/hotreload/` | ~600 | PARTIAL | `tests/tooling/hotreload/`, `tests/resource/asset/test_hot_reload.py` |
| 8 | Localization | `engine/ui/text/localization.py`, `trinity/decorators/localization.py` | ~800 | REAL (UI) / STUB (decorator) | `tests/ui/text/`, `tests/trinity/test_localization.py`, `tests/tooling/localization/` |
| 9 | Accessibility | `engine/ui/accessibility/`, `trinity/decorators/accessibility.py` | ~1,300 | REAL (UI) / STUB (decorator) | `tests/ui/accessibility/`, `tests/trinity/test_accessibility.py` |
| 10 | Platform Abstraction | `engine/platform/` (os, input, window, gpu, rhi, audio, services) | ~5,000+ | REAL | `tests/platform/` |
| 11 | Build System Integration | `/Cargo.toml`, `crates/`, `omega/`, `flowforge/` | ~20 | PARTIAL | `npm test` (via CLAUDE.md) |
| 12 | Configuration | `engine/core/constants.py`, `trinity/constants.py`, `engine/platform/constants.py` | ~250 | REAL | Implicit (direct imports) |
| 13 | Security | `trinity/decorators/security.py` | ~197 | STUB | None found |
| 14 | Save System | `trinity/decorators/save_system.py` | ~237 | STUB | None found |
| 15 | Common Utilities | `engine/common/` | ~0 | MISSING | None found |

---

## Overall Status

- **REAL (fully implemented):** Logging, Profiling, Crash Handling, Platform Abstraction, Configuration
- **PARTIAL (infrastructure exists, runtime missing):** Serialization, Hot-Reloading, Build System, Error Handling
- **STUB (tag/decorator metadata only):** Plugin System, Localization (decorator), Accessibility (decorator), Security, Save System
- **MISSING:** Common Utilities (`engine/common/` is empty)

## Key Architecture Pattern

All cross-cutting concerns in the Trinity pattern follow the same architecture:
- **Decorator layer** (`trinity/decorators/`) — metadata tagging via the Op-based `make_decorator()` system
- **Engine layer** (`engine/`) — actual runtime implementation
- The decorator layer is often STUB while the engine layer is REAL (logging, profiling, accessibility, localization)
- Some have neither layer implemented (Common Utilities)
- Some have only the decorator layer with no runtime (Security, Save System)

## GAPSET_3_BRIDGE Connections

| Domain | Rust Bridge Potential | Priority |
|--------|----------------------|----------|
| Error Handling | High — structured error types, Result<T,E> bridge | Medium |
| Logging | Medium — structured log ingestion from Rust | Low |
| Serialization | High — protobuf/flatbuffers serde bridge | High |
| Plugin System | High — WASM-based plugin sandbox | Medium |
| Hot-Reloading | Medium — watch-based asset recompile | Low |
| Platform Abstraction | High — native OS APIs via Rust | High |
| Build System | High — unified cargo/pip build | High |

## Corrected Task Breakdown

The original 3 tasks (T-CC-0.1, T-CC-0.2, T-CC-0.3) are insufficient. The corrected breakdown should cover at minimum:

T-CC-0.x: Common Utilities (`engine/common/` is empty — needs implementation)
T-CC-1.x: Serialization runtime (binary format, schema, protobuf/flatbuffers)
T-CC-2.x: Plugin/Mod runtime (mod loader, WASM sandbox, isolation)
T-CC-3.x: Hot-Reload integration (wire file_watcher to runtime)
T-CC-4.x: Security runtime enforcement (authz, rate limiting, encryption)
T-CC-5.x: Save/Restore runtime (save slots, atomic writes, migration)
T-CC-6.x: Error Handling runtime (retry, fallback, circuit breaker)
T-CC-7.x: Build system unification (CI/CD, cargo+python integration)
