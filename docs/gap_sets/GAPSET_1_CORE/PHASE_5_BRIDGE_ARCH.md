# PHASE 5: Python-Side Bridge Wiring

**Scope:** Wire Python layer to Rust via 3-channel PyO3 bridge protocol (Type, Data, Command) with RustStorageDescriptor, ComponentMeta integration, World/Entity classes, and scheduler bridge.
**Depends on:** Phase 0 (omega crate with pyo3), Phase 2 (TypeRegistry, ComponentStore), Phase 2.4 (CommandBuffer), Phase 3 (JobGraph for scheduler -- deferred)
**Produces:** 14 PyO3 functions in `_omega` module, 3-channel protocol (LIVE), RustStorageDescriptor (LIVE), ComponentMeta auto-registration (LIVE), World/Entity dual-write (LIVE)
**Status:** MOSTLY COMPLETE (5/6 tasks DONE, 1 ABSENT)

## 1. Overview

Phase 5 delivers the bridge that connects Python's ergonomic ECS API to Rust's high-performance ComponentStore. It is the "wiring" layer that makes the dual-language architecture work: Python classes call Rust functions transparently, with automatic fallback when the Rust module (`_omega`) is not available.

The bridge implements a **3-channel protocol** established in GAPSET_3_BRIDGE:

1. **Type Channel** (registration): When a Python component class is defined (`class Position(Component): x: float, y: float`), `ComponentMeta.__new__()` automatically calls `_omega.type_register()` with the component's computed layout (fields, offsets, total size). This populates the Rust TypeRegistry at definition time.
2. **Data Channel** (per-field access): `RustStorageDescriptor` intercepts reads/writes on component instances. Reading `instance.position` calls `_omega.component_read()` which fetches raw bytes from the Rust ComponentStore SoA column and decodes them to Python values. Writing calls `_omega.component_write()` which encodes the Python value to bytes and writes to the column.
3. **Command Channel** (deferred mutations): `renderer_*` and `frame_graph_execute()` functions in `_omega` accept commands from Python (renderer init, resize, shader recompilation, screenshot, shutdown, material compilation) and dispatch them to the Rust side.

The bridge is **already live** with 14 PyO3 functions compiled into `_omega.so` (~70MB, ABI3 forward-compatible). The only missing piece is T-CORE-5.5 (Scheduler Bridge), which depends on Phase 3 (ThreadPool/JobGraph) and Phase 2.5 (Checksum/SystemPhase), neither of which exist yet.

## 2. Architectural decisions

- **PyO3 as the bridge mechanism.** omega crate's `bridge.rs` defines `#[pyfunction]` functions exported as the `_omega` Python module. Python imports `_omega` at its own risk: `try: from _omega import component_read` with `except ImportError` fallback to dict storage.
- **3-channel protocol separation.** Type channel runs at class definition time (import); Data channel runs per-field per-frame (hot path); Command channel runs at frame boundaries and explicit engine calls. Each channel has distinct latency requirements and error handling.
- **RustStorageDescriptor as the innermost descriptor.** The descriptor chain model (from GAP 3) places `RustStorageDescriptor` at the innermost position -- it directly reads/writes the Rust ComponentStore. Non-component objects fall back to `__dict__` storage transparently.
- **ComponentMeta auto-registration (Step 6b).** When `ComponentMeta.__new__()` creates a new component class, it calls `_build_rust_layout()` to compute the byte layout (Python float -> f32, int -> i32, bool -> u8, str -> String). The layout is JSON-encoded and sent to `_omega.type_register()`. This happens automatically at class definition time.
- **Dual-write World pattern.** `World.spawn()` dual-writes to both the Python ArchetypeGraph and the Rust ComponentStore when `_HAVE_OMEGA` is true. `World.spawn_rust()` writes exclusively to Rust (no Python dict overhead). This allows gradual migration: existing Python ECS code works unchanged, while Rust-accelerated paths activate transparently.
- **Lazy import with silent fallback.** The `_HAVE_OMEGA` boolean gates all Rust bridge calls. If `_omega.so` isn't available (no PyO3 build), the engine runs entirely in Python with dict-backed storage. This is critical for development environments without Rust compilation.
- **GIL release for long-running operations.** Bridge functions release the Python GIL during ECS queries and render operations via `Python::allow_threads()` or equivalent, allowing Python threads to run concurrently with Rust operations.

## 3. Constraints specific to this phase

- Bridge functions must handle Python type conversion correctly: float->f32, int->i32, bool->u8, str->String with proper encoding.
- Errors on the Rust side must propagate as Python exceptions (`PyErr`, `PyValueError`, `RuntimeError`).
- Type registration must be idempotent (duplicate definition returns same type ID).
- Data channel must be fast -- per-field read/write on the hot path must not allocate where possible.
- Command channel functions must be safe to call from any thread (Arc<RwLock<...>> on the Rust side).
- The scheduler bridge (T-CORE-5.5) is deferred indefinitely -- it depends on Phase 3 and Phase 2.5.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `omega/src/bridge.rs` | 14 PyO3 functions: `type_register`, `type_list`, `initialize_component_store`, `component_read`, `component_write`, `component_delete`, `renderer_init`, `renderer_resize`, `renderer_screenshot`, `renderer_recompile_materials`, `renderer_shutdown`, `material_compile`, `editor_list_entities`, `frame_graph_execute`. Type conversion (f32/i32/u8/string). RuntimeError fallbacks. | DONE |
| `trinity/descriptors/rust_storage.py` | `RustStorageDescriptor`: routes `_get_stored()`/_set_stored()` to _omega bridge. Falls back to `__dict__` when `_HAVE_OMEGA` is false. Type code mapping (float->f32, int->i32, bool->u8, str->string). | DONE |
| `engine/core/ecs/world.py` | `World` class: spawn, spawn_bundle, spawn_rust, destroy, add/remove/get/has_component, query, for_each, command_buffer, flush_commands. Dual-writes to Rust ComponentStore when _HAVE_OMEGA. | DONE |
| `engine/core/ecs/entity.py` | `Entity` + `EntityAllocator` with index/generation packing, free-list, generation bump. | DONE |
| `engine/core/ecs/command_buffer.py` | Python CommandBuffer with SpawnCommand, DespawnCommand, InsertComponentCommand, RemoveComponentCommand, flush(). | DONE |
| (planned) `trinity/omega/scheduler.py` | Frame loop integration: dispatch system phases as job graphs to Rust ThreadPool. Phase transition triggers CommandBuffer flush. Frame start -> LinearAllocator reset. Frame end -> HierarchicalChecksum verification. | NOT IMPLEMENTED (depends on Phase 3 + Phase 2.5) |

**14 PyO3 bridge functions (from `omega/src/bridge.rs`):**

| Function | Channel | Description |
|----------|---------|-------------|
| `type_register` | Type | Register component layout with Rust TypeRegistry |
| `type_list` | Type | List all registered component types as (id, name, size) tuples |
| `initialize_component_store` | Data | One-time init of global ComponentStore singleton |
| `component_read` | Data | Read one field from Rust SoA, decode to Python value |
| `component_write` | Data | Encode Python value to bytes, write to Rust SoA |
| `component_delete` | Data | Delete a component field from Rust SoA |
| `renderer_init` | Command | Initialize wgpu renderer instance/adapter/device/surface |
| `renderer_resize` | Command | Handle window resize, reconfigure swapchain |
| `renderer_screenshot` | Command | Capture current frame to screenshot |
| `renderer_recompile_materials` | Command | Hot-reload material shaders |
| `renderer_shutdown` | Command | Graceful renderer teardown |
| `material_compile` | Command | Compile material from Python descriptor |
| `editor_list_entities` | Command | List entities for editor UI |
| `frame_graph_execute` | Command | Execute compiled frame graph for current frame |

## 5. Testing strategy

- Bridge tests validate type registration round-trip (Python -> Rust type_register -> type_list -> Python).
- RustStorageDescriptor tests validate per-field read/write round-trip via component_read/component_write.
- World tests validate spawn/despawn/query through the dual-write path.
- 1M field read stress test (target: <100ms total) validates data channel performance.
- Import fallback tests validate that the engine works without _omega.so.
- **Missing:** Scheduler bridge integration tests (T-CORE-5.5 doesn't exist).

## 6. Open questions

- **Scheduler bridge priority:** T-CORE-5.5 (frame loop -> ThreadPool dispatch) depends on Phase 3 + Phase 2.5, both absent. Should this be re-scoped to use Python threading instead of a Rust thread pool, or remain deferred until those phases are implemented?
- **Error propagation design:** Some bridge functions silently ignore errors (`component_write` returns `Ok(())` even when the entity doesn't exist). Should these be noisy in debug mode with a configurable error threshold?
- **String field handling:** Strings in SoA columns use C-string convention (null-terminated). This limits field length and requires careful handling. Should strings be stored as fixed-size byte arrays with a configurable max length?

## 7. References

- `omega/src/bridge.rs` -- All 14 PyO3 bridge functions
- `omega/src/lib.rs` -- `_omega` PyO3 module definition with all function exports
- `trinity/descriptors/rust_storage.py` -- RustStorageDescriptor (innermost descriptor)
- `engine/core/ecs/world.py` -- World class with dual-write to Rust
- `engine/core/ecs/entity.py` -- Entity + EntityAllocator
- `engine/core/ecs/command_buffer.py` -- Python CommandBuffer
- GAP_1_SUMMARY.md -- Investigation for T-CORE-5.1 through T-CORE-5.6
- GAPSET_3_BRIDGE docs -- Original bridge implementation with 3-channel protocol
