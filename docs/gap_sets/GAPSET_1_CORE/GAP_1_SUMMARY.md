# GAP_1_SUMMARY — GAPSET_1_CORE Source-Code Investigation

**Investigation date:** 2026-05-22
**Method:** Source-code inspection of every claimed file/function/struct in the TRINITY workspace
**Scope:** All tasks across 6 phases (T-CORE-0.x through T-CORE-5.x)
**RDC context:** The original PHASE_N_TODO.md was the initial plan with all `[ ]` checkmarks. 
Significant work was completed via GAPSET_3_BRIDGE (the bridge implementation) which crossed over with many GAP 1 deliverables. This investigation determines what actually exists.

---

## Phase 0: Deterministic Math Library

### T-CORE-0.1: Crate Scaffolding
- **Status: [x] DONE** — The `omega/` crate exists at workspace root with its own `Cargo.toml`
- Dependencies present: bytemuck, parking_lot, crossbeam, slotmap, serde, pyo3
- wgpu is NOT a direct dep of omega (by design — omega has no GPU dependency; GPU types in renderer-backend use omega via bytemuck casts)
- `cargo check --workspace` succeeds with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`
- **Verdict:** GREEN_LIGHT. Crate scaffolding is complete.

### T-CORE-0.2: Fixed16 and Fixed32
- **Status: [x] DONE** — `omega/src/fixed.rs` contains:
  - `Fixed16` (Q8.8 via i16) with add, sub, mul, div, PartialEq, PartialOrd, From/Into f32
  - `Fixed32` (Q16.16 via i32) with same operators + bytemuck Pod/Zeroable
  - Conversion to/from f32, serialize/deserialize
  - 240 inline + external tests pass
- **Verdict:** GREEN_LIGHT. All acceptance criteria met.

### T-CORE-0.3: FVec2, FVec3, FVec4
- **Status: [x] DONE** — `omega/src/vec.rs` contains:
  - `FVec2`, `FVec3`, `FVec4` with component access, arithmetic, dot, cross (FVec3), length, normalize, lerp
  - All bytemuck Pod+Zeroable
  - 28 inline tests pass
- **Verdict:** GREEN_LIGHT.

### T-CORE-0.4: FQuat and M64
- **Status: [x] DONE** — `omega/src/quat.rs` + `omega/src/mat.rs` contain:
  - `FQuat` (w,x,y,z Fixed32) with identity, mul, conjugate, inverse, rotate_vector, slerp
  - `M64` (4x4 column-major Fixed32) with identity, mul, inverse, transpose, look_at, perspective
  - All bytemuck Pod+Zeroable
  - slerp threshold at 0.9995 (verified in source)
- **Verdict:** GREEN_LIGHT.

### T-CORE-0.5a: Standard Float Types (Vec2/3/4, Mat3/4, Quat)
- **Status: [x] DONE** — Same source files as fixed-point:
  - `Vec2`, `Vec3`, `Vec4` (f32) — full API surface, bytemuck Pod+Zeroable
  - `Mat3`, `Mat4` (f32) — identity, mul, inverse, transpose, translate, rotate, scale, perspective, look_at
  - `Quat` (f32) — full operations including slerp, from_axis_angle, from_euler
- **Verdict:** GREEN_LIGHT.

### T-CORE-0.5b: SimRng and TrigLUT
- **Status: [x] DONE** — `omega/src/rng.rs` + `omega/src/trig.rs` contain:
  - `SimRng` (splitmix64 deterministic PRNG) — seed-based, platform-independent output
  - `TrigLUT` — precomputed sin/cos/tan at 4096 intervals with linear interpolation
  - Tests verify determinism and accuracy bounds
- **Verdict:** GREEN_LIGHT.

### T-CORE-0.6: Math Library Tests
- **Status: [x] DONE** — 240 tests in `omega/tests/math_tests.rs` + ~76 inline tests across modules
- 316 omega tests pass total
- Covers: fixed-point arithmetic, vector ops, quaternion ops, matrix ops, TrigLUT accuracy, SimRng determinism
- **Verdict:** GREEN_LIGHT.

---

## Phase 1: Memory Management + Entity System

### T-CORE-1.1: LinearAllocator (Frame-Scoped Bump Allocator)
- **Status: [x] DONE (as FrameAllocator)** — `crates/renderer-backend/src/memory.rs`:
  - `FrameAllocator` is the bump-pointer allocator — allocate(size, alignment), reset() per frame, high-water mark tracking
  - Alignment: conceptually CACHE_LINE (64) but uses the requested alignment
  - Per-frame reset model matches the spec
- **Notable divergence:** Named `FrameAllocator` instead of `LinearAllocator`. Same semantics.
- **Verdict:** GREEN_LIGHT.

### T-CORE-1.2: PoolAllocator
- **Status: [x] DONE** — `memory.rs`:
  - `PoolAllocator` with fixed-size block classes (64KB, 256KB, 1MB, 4MB)
  - acquire() / release() pattern via free-list tracking
  - CACHE_LINE aligned allocations
- **Notable divergence:** Uses block-size classes rather than uniform slots. Different from spec's "fixed-size slots" but serves the same purpose.
- **Verdict:** GREEN_LIGHT.

### T-CORE-1.3: RingBuffer (Staging Allocator)
- **Status: [x] DONE (as StackAllocator)** — `memory.rs`:
  - `StackAllocator` provides LIFO allocation for nested staging
  - LIFO semantics differ from spec's circular buffer with head/tail cursors
  - The triple-buffered `BufferRegistry` in `gpu_driven/buffers.rs` (777 lines) covers the GPU staging need
- **Notable divergence:** No circular ring buffer. StackAllocator (LIFO) + BufferRegistry (triple-buffered) cover the staging use case differently.
- **Verdict:** [~] PARTIAL — staging covered, but not via RingBuffer as specified.

### T-CORE-1.4: EntityId (Generational Index)
- **Status: [~] PARTIAL** — No standalone `EntityId` type in Rust with 24-bit index + 8-bit generation.
  - `engine/core/ecs/entity.py` has Python `Entity` + `EntityAllocator` with allocation/deallocation
  - `omega/src/bridge.rs` uses `entity_id: u64` (plain integer, no generation bits)
  - The 24+8 bit packing with generation checking for stale IDs does NOT exist in Rust
- **Verdict:** [~] PARTIAL — Entity lifecycle in Python works, but the generational-index Rust type doesn't exist.

### T-CORE-1.5: Memory and Entity Tests
- **Status: [x] DONE** — `memory.rs` has 15 inline tests covering all allocators
- Python ECS tests cover entity lifecycle
- **Verdict:** GREEN_LIGHT for allocators; EntityId tests are in Python realm.

---

## Phase 2: Archetype ECS Runtime

### T-CORE-2.1: ComponentTypeInfo and TypeRegistry
- **Status: [x] DONE** — `crates/renderer-backend/src/type_registry.rs`:
  - `ComponentTypeInfo` — id, name, size, fields (Vec<FieldLayout>), flags: u32, archetype_id
  - `TypeRegistry` — parking_lot::RwLock<HashMap<u32, ComponentTypeInfo>>, 9 methods
  - `ArchetypeId` — deterministic derivation from component ID set (sort + hash + XOR-fold)
  - PyO3: `type_register()`, `type_list()` via `_omega.so`
- **Verdict:** GREEN_LIGHT.

### T-CORE-2.2: Archetype and SoA Columns
- **Status: [x] DONE** — `crates/renderer-backend/src/component_store.rs`:
  - `Archetype` — id, component_ids, columns (Vec<Vec<u8>>), entities (Vec<usize>), row_count
  - Swap-remove removal for dense storage
  - Column access by component_id
- **Verdict:** GREEN_LIGHT.

### T-CORE-2.3: ComponentStore
- **Status: [x] DONE** — `component_store.rs`:
  - spawn(entity_id, component_values) — with free-list reuse
  - despawn(entity_id) — marks for reuse
  - read_field/write_field — archetype+row+offset, bounds-checked
  - query(component_ids) — superset match with freed-row exclusion
  - column_slice(component_id) — contiguous &[u8] for GPU upload
  - Global singleton via OnceLock<Arc<RwLock<ComponentStore>>>
  - 35 tests pass
- **Verdict:** GREEN_LIGHT.

### T-CORE-2.4: CommandBuffer
- **Status: [~] PARTIAL** — `engine/core/ecs/command_buffer.py` exists in Python with CommandBuffer + flush.
  - No Rust CommandBuffer implementation
  - The Python World class has `command_buffer` property and `flush_commands()` method
- **Verdict:** [~] PARTIAL — Python implementation exists; Rust port not done.

### T-CORE-2.5a: HierarchicalChecksum
- **Status: [-] ABSENT** — No checksum system in Rust or Python ECS.
- **Verdict:** [-] Does not exist.

### T-CORE-2.5b: SystemPhase and SystemContext
- **Status: [-] ABSENT** — No system phase or system context infrastructure.
- **Verdict:** [-] Does not exist.

### T-CORE-2.6: ECS Tests
- **Status: [~] PARTIAL** — `component_store.rs` has 35 tests (spawn, despawn, read, write, query, column_slice).
  - Python ECS has its own test suite
  - No Rust-level command buffer, checksum, or system phase tests (those components don't exist)
- **Verdict:** [~] PARTIAL — core storage tests exist; structural change tests absent.

---

## Phase 3: Task/Job System

### T-CORE-3.1: ThreadPool with Work-Stealing
- **Status: [-] ABSENT** — No Rust ThreadPool implementation.
  - `engine/platform/os/threading.py` exists (Python threading utilities)
  - crossbeam is a declared dependency but no thread pool is built with it
- **Verdict:** [-] Does not exist.

### T-CORE-3.2: JobGraph and Dependencies
- **Status: [-] ABSENT** — No job graph dependency DAG in Rust.
- **Verdict:** [-] Does not exist.

### T-CORE-3.3: parallel_for
- **Status: [-] ABSENT** — No parallel_for implementation.
- **Verdict:** [-] Does not exist.

### T-CORE-3.4: Task System Tests
- **Status: [-] ABSENT** — No tests (substrate doesn't exist).
- **Verdict:** [-] Does not exist.

---

## Phase 4: RHI wgpu Mapping

### T-CORE-4.1 through T-CORE-4.7: RHI Device/Buffer/Pipeline/Command/Swapchain/BindGroup layers
- **Status: [~] PARTIAL** — Two-track implementation:
  1. **Python RHI ABCs exist:** `engine/platform/rhi/` (7 files — device.py, resources.py, pipeline.py, commands.py, swapchain.py, sync.py, raytracing.py)
  2. **Rust wgpu renderer exists:** `crates/renderer-backend/src/renderer.rs` (438+ lines) with Instance/Adapter/Device/Queue/Surface, triangle pipeline, render loop. `pipeline.rs` (705 lines) with PipelineTable+ShaderCache.
  3. **No formal Python→Rust RHI mapping layer.** The two sides exist independently but aren't bridged.
- **Verdict:** [~] PARTIAL — Both Python RHI and Rust wgpu backends exist independently. The mapping layer (Python RHI ABC → wgpu calls via PyO3) doesn't exist.

---

## Phase 5: Compute Shader Infrastructure
(From the tail of the TODO — verification of remaining phases)

### T-CORE-5.x tasks
- **Status: PARTIALLY COVERED** — The `gpu_driven/` module (buffers.rs, texture_table.rs, material_table.rs, mesh_table.rs) covers GPU-driven infrastructure. 8 WGSL shaders in `crates/renderer-backend/shaders/` cover PBR, culling, shadows, DDGI, and particles.
- **Verdict:** [~] PARTIAL — GPU compute infrastructure exists but wasn't architected per the original GAP 1 plan.

---

## Quantitative Summary

| Phase | Tasks | [x] Done | [~] Partial | [-] Absent |
|-------|-------|----------|-------------|------------|
| 0: Math Library | 7 | 7 | 0 | 0 |
| 1: Memory/Entity | 5 | 3 | 2 | 0 |
| 2: Archetype ECS | 8 | 3 | 2 | 3 |
| 3: Task/Job System | 4 | 0 | 0 | 4 |
| 4: RHI wgpu | 7 | 0 | 7 | 0 |
| 5+: Compute/Other | ~6 | 0 | 6 | 0 |
| **Total** | **~37** | **13** | **17** | **7** |

### What was built (via GAPSET_3_BRIDGE crossover)

1. **omega math crate** — All Phase 0 deliverables (Fixed types, vectors, matrices, quaternions, TrigLUT, SimRng). 316 tests.
2. **renderer-backend crate** — TypeRegistry, ComponentStore (SoA), FrameAllocator, PoolAllocator, StackAllocator, wgpu Renderer, PipelineTable, FrameGraph compiler, PostProcess, Particles, DDGI, Editor.
3. **_omega.so** — 14 PyO3 functions bridging Python↔Rust.
4. **8 WGSL shaders** — PBR, forward+ culling, CSM shadows, DDGI probes, particles.
5. **Python ECS** — World, Entity, ArchetypeGraph, Query, CommandBuffer, RustStorageDescriptor.

### What was NOT built (still planned)

1. **Thread pool / job system** (Phase 3 entire) — No work-stealing, JobGraph, or parallel_for.
2. **HierarchicalChecksum** (T-CORE-2.5a) — No deterministic state verification.
3. **SystemPhase** (T-CORE-2.5b) — No system ordering/runtime infrastructure.
4. **RHI mapping layer** — Python RHI ABCs and Rust wgpu backends exist but aren't bridged.
5. **RingBuffer** — StackAllocator + BufferRegistry cover staging differently.

### Key Architectural Divergence

The original GAP 1 plan imagined a monolithic Rust core with its own ECS, thread pool, and RHI abstraction. The actual implementation split into:
- **Python side:** ECS (World/Entity/ArchetypeGraph), RHI ABCs, engine modules
- **Rust side:** omega math, ComponentStore (SoA storage accelerator), wgpu renderer, FrameGraph compiler, WGSL shaders
- **Bridge:** 14 PyO3 functions connecting the two via _omega.so

This is architecturally valid — the descriptor chain model from GAP 3 allows progressive Rust activation without a monolithic rewrite. The remaining GAP 1 items (thread pool, checksum, system phases) can be implemented as independent Rust crates and wired in later.
