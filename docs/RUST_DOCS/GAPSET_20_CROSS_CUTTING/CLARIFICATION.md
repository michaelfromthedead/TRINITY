# GAPSET_20_CROSS_CUTTING — Clarification Document

**Date:** 2026-05-22
**RDC Investigator:** Single-agent

---

## Why Cross-Cutting Concerns Exist

Cross-cutting concerns are functionalities that cannot be cleanly encapsulated
within any single engine layer (rendering, physics, audio, UI, etc.). They
permeate all subsystems and require consistent, centralized implementation:

- **Logging**: Every subsystem needs diagnostic output
- **Profiling**: Every subsystem needs performance measurement
- **Error Handling**: Every subsystem needs failure recovery
- **Platform Abstraction**: Every subsystem depends on OS capabilities
- **Serialization**: Every subsystem needs data persistence
- **Security**: Every subsystem needs access control

The Trinity pattern addresses this through a **two-layer architecture**:
decorator metadata at the Python class/function level, plus runtime engine
implementations.

---

## Architectural Decisions Discovered

### Decision 1: Op-Based Decorator System
All cross-cutting concerns use the 7-Op system (`trinity/decorators/ops.py`):
TAG, HOOK, REGISTER, DESCRIBE, TRACK, VALIDATE, INTERCEPT. Decorators
become pure metadata — the runtime behavior lives in engine modules.

**Consequence**: Many decorator layers look complete (steps defined, validated,
registered) but produce NO runtime behavior. The actual implementation must
be written separately in the engine layer.

### Decision 2: Trinity ↔ Engine Separation
The `trinity/` package handles class-level metadata (what IS this?).
The `engine/` package handles runtime behavior (what DOES this?).
This separation is clean architecturally but leads to incomplete implementations
when only one layer is built.

### Decision 3: Constants Layer
Configuration is split across three files:
- `engine/core/constants.py` — core engine constants (timing, entities, memory, logging defaults)
- `trinity/constants.py` — Trinity pattern constants (pool sizes, fixed-point, serialization)
- `engine/platform/constants.py` — platform constants (polling intervals, thread timeouts)

This is consistent and well-organized.

### Decision 4: Platform Abstraction via Singleton Bootstrap
`engine/platform/__init__.py` uses a `bootstrap_platform()` function with
singleton pattern (thread-safe, double-checked locking). Initializes platform
detection, app lifecycle, graphics device (null backend), and low latency features.

### Decision 5: Engine Debug as Primary Runtime
The `engine/debug/` directory is the runtime home for logging, profiling,
crash handling, console, visual debug, replay, and testing — rather than
being scattered across individual subsystems. This is a good architectural choice
but creates dependency: all subsystems must import from `engine.debug.*`.

---

## GRANDPHASE1 vs GRANDPHASE2 Relationship

### GRANDPHASE1 (current — Python implementation)
- All Trinity decorators implemented with the 7-Op system
- Engine runtime mostly implemented (logging, profiling, crash handling, platform abstraction)
- Tests exist for most REAL subsystems
- Build system is minimal (Cargo workspace only)

### GRANDPHASE2 (planned — Rust bridge)
- Serialization to Rust native types (protobuf, flatbuffers, or custom binary format)
- Error propagation from Rust results to Python exceptions
- WASM-based plugin sandbox (mod loader in Rust)
- OS-level platform bindings in Rust (file system, threading, memory)
- CI/CD pipeline for cross-language building and testing
- Performance-critical profiling infrastructure in Rust

---

## What Remains

### Critical Gaps
1. **`engine/common/` is empty** — no shared utilities, types, or helper functions
2. **No serialization runtime** — only descriptor metadata, no actual binary/JSON serialization
3. **No plugin/mod loader** — 9 modding decorators defined, zero execution capability
4. **No security enforcement** — 4 security decorators defined, zero runtime authorization

### Architectural Debt
5. **Decorator ↔ Engine disconnect** — e.g., `@accessible` adds a tag but the actual accessibility engine (`engine/ui/accessibility/`) does not read it
6. **No unified build** — Rust and Python build independently
7. **Hot-reload not integrated** — file watcher exists but no engine integration
8. **Save system not implemented** — only metadata decorators exist

### Quality Gaps
9. **Security tests missing** (0 test files for security decorators)
10. **Save system tests missing** (0 test files for save system)
11. **Common utilities tests missing** (empty directory)
12. **Build system has no test infrastructure** (CI/CD not configured)
