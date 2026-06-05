# PHASE 2: Archetype ECS Runtime

**Scope:** Implement the archetype-based ECS storage with SoA columns, component type registry, command buffer, hierarchical checksum, and system phase infrastructure.
**Depends on:** Phase 0 (omega math types for component data), Phase 1 (EntityId, allocators)
**Produces:** TypeRegistry, ComponentStore (SoA) in Rust; CommandBuffer in Python; HierarchicalChecksum and SystemPhase do not yet exist
**Status:** PARTIAL (3/8 tasks DONE, 2 PARTIAL, 3 ABSENT)

## 1. Overview

Phase 2 delivers the ComponentStore, the Rust-accelerated ECS storage backend that the Python ECS layer calls via PyO3. It is NOT a full ECS -- entity lifecycle (World, ArchetypeGraph, Query) stays in Python (`engine/core/ecs/`). The Rust side stores raw bytes in SoA columns and provides read/write/query primitives that the Python layer calls through the bridge.

The architecture follows a data-oriented design:

- **TypeRegistry** -- A global `RwLock<HashMap<u32, ComponentTypeInfo>>` that maps component type IDs to their layout metadata (field names, types, offsets, total size). `ArchetypeId` is derived deterministically from a set of component IDs (sort + hash + XOR-fold).
- **ComponentStore (Archetype + SoA columns)** -- Each Archetype groups entities sharing the same component type signature. Data is stored column-major: one contiguous `Vec<u8>` per component type, with row `i` at byte `i * stride`. Swap-remove removal preserves density. A free list reuses freed rows.
- **CommandBuffer** -- Python-side deferred mutation commands (spawn, despawn, insert, remove) that flush atomically to the World. No Rust CommandBuffer exists.
- **HierarchicalChecksum** (absent) -- Planned per-entity rolling checksum for deterministic state verification. Not implemented.
- **SystemPhase/SystemContext** (absent) -- Planned system ordering and execution context. Not implemented.

## 2. Architectural decisions

- **Storage accelerator, not full ECS.** The Rust ComponentStore is a storage accelerator called from Python. The Python World/AchetypeGraph/Query remain the primary ECS API. This avoids a Rust ECS rewrite while getting data-oriented SoA performance.
- **SoA layout in raw Vec<u8> columns.** Each component type in an archetype has its own `Vec<u8>`. The stride is the component's total byte size from TypeRegistry. Field offsets are interpreted by Python callers via the bridge. This layout is GPU-direct: `column_slice(component_id)` returns a contiguous `&[u8]` ready for wgpu buffer upload.
- **Deterministic ArchetypeId.** Derived from sorted component IDs via DefaultHasher + XOR-fold to 32 bits. Order-independent; same component set always produces same archetype ID.
- **Swap-remove for removal.** When an entity is despawned, its row is swapped with the last alive row and the row index is added to the free list. This keeps storage dense and insertion O(1) amortized.
- **Global singleton via OnceLock.** `COMPONENT_STORE: OnceLock<Arc<RwLock<ComponentStore>>>` is initialized once by `initialize_component_store()` from Python. PyO3 bridge calls read/write/delete on this singleton.
- **No Rust CommandBuffer.** The Python CommandBuffer (`engine/core/ecs/command_buffer.py`) records and flushes commands. A Rust CommandBuffer would be faster but requires implementing the deferred mutation pattern in Rust, which hasn't been done.

## 3. Constraints specific to this phase

- All component data stored as raw bytes; the store is type-agnostic beyond stride (byte size) from TypeRegistry.
- Field offset interpretation is the caller's responsibility (Python bridge or Rust read_field/write_field).
- Read/write must be bounds-checked: archetype+row+offset must be within valid range.
- Query uses superset matching: requesting [Transform, Velocity] returns entities with at least both components.
- Global singleton must be thread-safe (Arc<RwLock<...>>) for concurrent read access from multiple threads.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `crates/renderer-backend/src/type_registry.rs` | `FieldLayout`, `FieldType`, `ComponentTypeInfo` (id, name, size, fields, flags, archetype_id), `ArchetypeId` derivation, `TypeRegistry` (RwLock<HashMap<u32, ComponentTypeInfo>>, 9 methods). 28 tests. | DONE |
| `crates/renderer-backend/src/component_store.rs` | `Archetype` (id, component_ids, columns: Vec<Vec<u8>>, entities, free_rows, row_count), `ComponentStore` (spawn, despawn, read_field, write_field, query, column_slice, global singleton). 35 tests. | DONE |
| `engine/core/ecs/command_buffer.py` | Python `CommandBuffer` with `Command` ABC, `SpawnCommand`, `DespawnCommand`, `InsertComponentCommand`, `RemoveComponentCommand`, `flush()`. World has `command_buffer` property + `flush_commands()`. | PARTIAL -- Python exists; no Rust CommandBuffer |
| (does not exist) | `HierarchicalChecksum` -- per-entity rolling checksum, world-level verification, xxhash-like fast hash. | ABSENT |
| (does not exist) | `SystemPhase` -- ordered collection of systems with `SystemContext` (delta_time, command_buffer, world_checksum). | ABSENT |
| (does not exist) | Rust CommandBuffer -- deferred spawn/despawn/add/remove commands applied atomically to ComponentStore. | ABSENT |

## 5. Testing strategy

- **35 tests in `component_store.rs`** covering: spawn, despawn (idempotent, free-list reuse), read_field/write_field (bounds-checked), query (superset match, freed-row exclusion, empty results), column_slice (contiguous, GPU-ready).
- **28 tests in `type_registry.rs`** covering: register_type, lookup_type, ArchetypeId determinism, field layout storage.
- **Python ECS tests** cover CommandBuffer recording + flush, entity lifecycle.
- **Missing tests:** No Rust CommandBuffer tests, no HierarchicalChecksum tests (component absent), no SystemPhase tests (component absent).

## 6. Open questions

- **Rust CommandBuffer priority:** The Python CommandBuffer works but requires Python GIL for flushing. A Rust CommandBuffer would enable lock-free command recording from worker threads. Should this be implemented before or after Phase 3 (thread pool)?
- **HierarchicalChecksum algorithm:** Should use a fast hash (xxhash-like) rather than cryptographic. What hash function should the Rust crate use -- `xxhash-rust` or `twox-hash`?
- **System ordering model:** Should SystemPhase define ordering via explicit `before/after` annotations or a DAG-based approach similar to JobGraph (Phase 3)? Using JobGraph would unify the scheduling model.

## 7. References

- `crates/renderer-backend/src/type_registry.rs` -- TypeRegistry, ArchetypeId, FieldLayout
- `crates/renderer-backend/src/component_store.rs` -- ComponentStore, Archetype, SoA columns
- `engine/core/ecs/command_buffer.py` -- Python CommandBuffer
- `engine/core/ecs/world.py` -- World class with dual-write (Python + Rust) paths
- `omega/src/bridge.rs` -- PyO3 functions wrapping ComponentStore calls
- GAP_1_SUMMARY.md -- Investigation for T-CORE-2.1 through T-CORE-2.6
- PHASE_N_TODO.md -- Corrected task list with detailed acceptance criteria
