# PHASE 2: Data Channel -- Component Store

**Scope:** Route ECS component field reads and writes through Rust's SoA (Struct of Arrays) storage, with Python-world accessors that transparently degrade to dict storage when Rust is unavailable.
**Depends on:** Phase 1 (TypeRegistry must know component layouts before storage can interpret fields)
**Produces:** Rust ComponentStore (SoA archetypes), RustStorageDescriptor (Python descriptor wiring), PyO3 component_read/write/delete functions
**Status:** PARTIALLY IMPLEMENTED -- Python ECS is fully functional (World/Entity/ArchetypeGraph/Query). RustStorageDescriptor exists with _omega imports wired. No Rust ComponentStore exists. The PyO3 bridge functions (_omega.component_read, _omega.component_write, _omega.component_delete) are imported but always fail.

## 1. Overview

The Data Channel is the middle layer of the bridge: it carries live entity component data between Python and Rust. The Python ECS (`engine/core/ecs/`) manages entity lifecycle, archetypes, and queries -- and is fully working. `RustStorageDescriptor` (`trinity/descriptors/rust_storage.py`) sits as the innermost descriptor in the ComponentMeta descriptor chain, attempting to route each field read/write through Rust. When the `_omega` PyO3 module is absent (always, in the current state), it falls back to Python's `__dict__`. The design is correct and well-factored; only the PyO3 bindings are missing.

## 2. Architectural decisions

- **Descriptor chain model**: ComponentMeta installs a stack of descriptors per field (validation, serialization, storage). RustStorageDescriptor sits at the innermost position -- it is the final storage backend. Everything above it (type conversion, bounds checking, serialization) remains in Python regardless of storage backend.
- **Python ECS as primary, Rust as accelerator**: The Python ECS (World, ArchetypeGraph, Query) is not replaced by Rust -- it is augmented. Entity lifecycle and archetype management stay in Python. Only the per-component per-field data storage migrates to Rust SoA columns.
- **Three PyO3 functions needed**: `component_read(entity_id, component_id, offset, field_type)` returns the field value; `component_write(entity_id, component_id, offset, value)` stores it; `component_delete(entity_id, component_id, offset)` removes it. Each has a pure-Python `RuntimeError` fallback in the descriptor.
- **Python World and Entity are real, working, and complete**: `world.py` (153 lines) provides spawn, spawn_bundle, destroy, add_component, remove_component, get_component, has_component, query, for_each, command_buffer. `entity.py` provides EntityAllocator with allocate/deallocate/is_alive. These are not stubs.

## 3. Constraints specific to this phase

- RustStorageDescriptor must be the innermost descriptor -- nothing wraps below it. `accepts_inner = ()` enforces this.
- Each field has a `_rust_offset` set by ComponentMeta._make_storage_descriptor() at class creation time.
- The `_HAVE_OMEGA` flag is read once at import time (module level in rust_storage.py), not dynamically. Adding a new Rust component at runtime would require a reload.
- Field type must be round-trippable through Rust and back -- the `field_type` argument to `component_read()` tells Rust how to interpret the raw bytes.

## 4. Component breakdown

| File/Function | Role | Status |
|---------------|------|--------|
| `engine/core/ecs/world.py` | World class (153 lines) | EXISTS -- spawn, destroy, query, for_each, command_buffer |
| `engine/core/ecs/entity.py` | Entity + EntityAllocator | EXISTS |
| `engine/core/ecs/archetype.py` | ArchetypeGraph | EXISTS |
| `engine/core/ecs/component.py` | ComponentId, ComponentMask | EXISTS |
| `engine/core/ecs/query.py` | Query, QueryDescriptor, QueryResult | EXISTS |
| `engine/core/ecs/command_buffer.py` | CommandBuffer for deferred operations | EXISTS |
| `trinity/descriptors/rust_storage.py` | RustStorageDescriptor (132 lines) | EXISTS -- wired but inactive |
| `component_meta.py:189-207` | _make_storage_descriptor() | EXISTS -- prefers Rust, falls back to Python |
| `component_meta.py:122-127` | Step 6b: _build_rust_layout + type_register | EXISTS -- feeds type info to Rust |
| `bridge.rs` | PyO3 component_read/write/delete | STUB -- no implementation |
| `type_registry.rs` | TypeRegistry for layout queries | EXISTS -- structs only, no runtime store |

## 5. Testing strategy

- Unit: RustStorageDescriptor._dict_get/_dict_set tests (Python-only mode).
- Unit: World.spawn/destroy/query round-trip with pure Python storage.
- Unit: Entity lifecycle (allocate -> deallocate -> is_alive).
- Integration: Add PyO3 component_read/write/delete stubs, verify Python code routes through Rust path.
- Integration: End-to-end: create component via ComponentMeta, spawn entity via World, read field via RustStorageDescriptor (Rust mode).

## 6. Open questions

- The `_omega` import in rust_storage.py expects `component_read(entity_id, component_id, offset, field_type)`. Should the Rust ComponentStore be keyed by (entity_id, component_id) or by (archetype_id, row_index)? The current Python ECS uses entity IDs, but Rust SoA storage is more natural with archetype rows.
- Should the ComponentStore live in omega (alongside math types) or in renderer-backend? It has no GPU dependency, so omega is a candidate, but it is conceptually part of the bridge, not the math library.

## 7. References

- Phase 1 (Type Channel) feeds layout metadata into the storage layer.
- Phase 7 (Frame Graph) consumes component data for rendering -- requires working Data Channel.
- GAP_3_SUMMARY.md section "Phase 2: Component Store" (corrected status, 13 real / 7 partial / 11 absent).
