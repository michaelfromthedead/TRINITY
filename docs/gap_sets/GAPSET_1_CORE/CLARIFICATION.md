# CLARIFICATION — GAPSET_1_CORE

**Purpose:** Conceptual framing, decision rationales, architectural divergence analysis, and pedagogical context for the engine core.

**Relationship to other docs:**
- `PROJECT.md` — the what (project overview and scope)
- `PHASE_N_TODO.md` — the do (corrected task list with source-code verification)
- `GAP_1_SUMMARY.md` — the was (deep codebase investigation results)
- `PHASE_<N>_*_ARCH.md` — the how per phase (when produced)
- This document — the "why it looks this way"

---

## 1. Why a Rust core for a Python engine

TRINITY's architecture follows a "Python for ergonomics, Rust for performance" model. The engine core (GAP 1) is the foundation: it provides types, storage, and execution primitives that every other layer consumes. Python handles editor tooling, asset pipelines, ECS orchestration, and configuration — everything that benefits from rapid iteration. Rust handles deterministic numerics, GPU-uploadable data, memory management, and parallel execution — everything that needs performance or safety guarantees.

The split is not arbitrary. Deterministic simulation (fixed-point math) cannot tolerate floating-point non-determinism across platforms. GPU data upload requires precise byte layout (bytemuck Pod/Zeroable). Memory allocation patterns in a game engine are predictable (frame-scoped, pool-based) and benefit from Rust's ownership model. A Python-only core would fail all three constraints; a Rust-only core would lose the iteration speed that makes TRINITY practical for development.

---

## 2. Why omega is math-only (no GPU dependency)

The `omega` crate provides deterministic math types with zero GPU dependencies. This was intentional:

1. **CI without a GPU**: Omega compiles and tests on any machine — no wgpu, no Vulkan drivers. This means 316 math tests run in CI on every commit.
2. **Shared across crates**: renderer-backend, potential future physics/networking crates all consume omega types. A shared dependency prevents type duplication.
3. **bytemuck as the bridge**: Omega types implement Pod/Zeroable, allowing any crate to cast them to byte slices for GPU upload. Omega itself never touches wgpu — the consumer does the upload.

This separation proved correct during GAP 3 development. The bridge crate (renderer-backend) imports omega for math types and wgpu for rendering, composing the two concerns without omega knowing about either.

---

## 3. Why the codebase diverged from the original GAP 1 plan

The original PHASE_N_TODO.md described a monolithic Rust core: all math, memory, ECS, thread pool, and RHI layers built as a single cohesive system. The actual implementation took a different path:

### What was built (via GAPSET_3_BRIDGE crossover)

The bridge project needed math types, component storage, and GPU rendering to function. These were implemented directly in the omega and renderer-backend crates:

- **Phase 0 (Math)**: Fully implemented. omega crate with all types (Fixed, Vec, Mat, Quat, TrigLUT, SimRng). 316 tests.
- **Phase 1 (Memory)**: Allocators implemented as FrameAllocator, PoolAllocator, StackAllocator in renderer-backend/src/memory.rs. Entity lifecycle in Python.
- **Phase 2 (ECS)**: TypeRegistry + ComponentStore (SoA) implemented in renderer-backend. CommandBuffer in Python. HierarchicalChecksum and SystemPhase not built.
- **Phase 4 (RHI)**: Two independent tracks: Python RHI ABCs (engine/platform/rhi/, 7 files) + Rust wgpu backend (renderer.rs, pipeline.rs). No formal mapping layer.

### What was NOT built

- **Phase 3 (Thread pool)**: Not implemented at all. The work-stealing thread pool, JobGraph, and parallel_for remain in the plan.
- **Phase 2.5 (Checksum + SystemPhase)**: Neither exists. Deterministic state verification and system ordering were deferred.
- **Phase 4 mapping layer**: The Python RHI → Rust wgpu bridge doesn't exist. Both sides work independently.

### Why this divergence is valid

The GAP 3 bridge took priority because it unblocked GPU rendering — the highest-value path. Math types, component storage, and wgpu integration were prerequisites for the bridge and were built in service of that goal. The thread pool, checksum, and formal RHI mapping are lower-priority items that can be implemented as independent crates later without blocking any gap set.

The architectural pattern proven in GAP 3 (descriptor chain activation, Python fallback, progressive Rust acceleration) applies equally to the remaining GAP 1 items. A thread pool can be a standalone crate with a PyO3 binding. SystemPhase can be wired into the existing World class. The RHI mapping can follow the same pattern as the ComponentStore bridge.

---

## 4. What "deterministic" means in this codebase

Determinism in the TRINITY engine core means:

1. **Fixed-point math is bit-exact**: Fixed32(1.5) * Fixed32(2.0) == Fixed32(3.0) on x86, ARM, and any future platform. No floating-point nondeterminism.
2. **SimRng is seed-reproducible**: Same seed → same sequence, regardless of platform or thread scheduling.
3. **TrigLUT is table-driven**: sin/cos/tan from a 4096-entry precomputed table with linear interpolation, guaranteeing identical results across all hardware.

Floating-point types (Vec2/3/4, Mat3/4, Quat as f32) are explicitly non-deterministic — they exist for the rendering path where slight variations don't matter. The fixed-point path is for simulation state that must replicate exactly (networked physics, replay systems, lockstep multiplayer).

---

## 5. The ECS storage model: two worlds, one API

The storage model has two layers:

```
Python World (engine/core/ecs/world.py)
  ├─ Python path: ArchetypeGraph → dict storage (always works)
  └─ Rust path: ComponentStore (SoA) → bytemuck-cast columns (when _omega available)

Per-field access via descriptor chain:
  RustStorageDescriptor → _omega.component_read/write/delete() → ComponentStore
  StorageDescriptor → obj.__dict__ (fallback)
```

Entities spawned via `World.spawn_rust()` exist purely in the Rust ComponentStore — no Python dict overhead. Entities spawned via `World.spawn()` dual-write to both stores when `_HAVE_OMEGA` is true. This allows gradual migration: existing Python ECS code works unchanged while Rust-accelerated paths activate transparently.

---

## 6. What remains and why it's deferred

| Item | Why deferred | Priority |
|------|-------------|----------|
| Thread pool (Phase 3) | Python threading + GIL-free Rust workers are not blocking any gap set | Low |
| HierarchicalChecksum | Needed for lockstep networking verification, which isn't the current focus | Medium |
| SystemPhase | System ordering exists implicitly in Python; formal Rust SystemPhase is optimization | Low |
| RHI mapping layer | Python RHI ABCs + Rust wgpu work independently; mapping is nice-to-have | Medium |
| RingBuffer | StackAllocator + BufferRegistry cover GPU staging differently | Low |

## 7. References

- GAPSET_3_BRIDGE docs — Where most GAP 1 implementation was actually delivered
- `omega/src/` — Math library (fixed.rs, vec.rs, mat.rs, quat.rs, trig.rs, rng.rs, spatial.rs)
- `crates/renderer-backend/src/` — ComponentStore, TypeRegistry, memory allocators, wgpu renderer
- `engine/core/ecs/` — Python ECS (World, Entity, ArchetypeGraph, Query, CommandBuffer)
- `engine/platform/rhi/` — Python RHI ABCs (device, resources, pipeline, commands, swapchain, sync)
