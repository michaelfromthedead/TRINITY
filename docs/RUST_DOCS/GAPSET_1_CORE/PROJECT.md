# GAPSET_1_CORE — Deterministic Engine Core

**Owner:** Michael
**Status:** PARTIALLY IMPLEMENTED (18/37 tasks DONE, 11 PARTIAL, 8 ABSENT)
**RDC run:** 2026-05-22
**Crossover:** Significant work completed via GAPSET_3_BRIDGE

## 1. Goal

Build the deterministic engine core for the TRINITY engine: a Rust math library with fixed-point and floating-point types for deterministic simulation, GPU-compatible SoA (Struct of Arrays) component storage, memory allocators for frame-scoped and pooled GPU resources, a work-stealing thread pool for parallel task execution, and an RHI (Render Hardware Interface) abstraction layer mapping Python ABCs to wgpu.

The core is the foundation that all other gap sets build upon — GAP 2 (Frame Graph), GAP 3 (Bridge), GAP 4 (Materials), and beyond all consume omega math types, the ComponentStore, and the RHI layer.

## 2. Why

TRINITY needs deterministic simulation for networking (lockstep/replay), GPU compute for rendering, and a Python-accessible Rust core for performance-critical paths. The engine core provides:

- **Bit-exact determinism** (Fixed32/Fixed16) for cross-platform reproducible simulation
- **GPU-ready data layout** (bytemuck Pod/Zeroable, SoA columns) for direct buffer upload
- **Frame-scoped memory** (bump/pool allocators) for predictable per-frame allocation patterns
- **Parallel execution** (work-stealing thread pool) for multi-core utilization
- **RHI abstraction** (Python ABCs → wgpu) for backend-independent rendering

The split design (Python for ergonomics, Rust for performance) mirrors the bridge architecture proven in GAP 3.

## 3. Hardware / Environment constraints

- Target: GPU-accelerated desktop (wgpu abstracts Vulkan/Metal/DX12/WebGPU)
- Rust edition 2021
- Python 3.12+
- bytemuck Pod/Zeroable required for all GPU-uploadable types
- Deterministic math must produce identical results on x86 and ARM
- Cross-platform: Windows/Linux/macOS

## 4. Non-goals

- Not a standalone engine — always part of TRINITY
- Not a general-purpose ECS — specifically designed for TRINITY's component model
- Not a cryptographic library — checksums/PRNGs are for determinism, not security
- Not a replacement for the Python ECS — Rust ComponentStore accelerates, Python ECS orchestrates
- Not a full RHI replacement — the Python RHI ABCs remain the public API

## 5. Phase overview

| Phase | Name | Status | Key Deliverable |
|-------|------|--------|-----------------|
| 0 | Deterministic Math Library | ✅ COMPLETE | omega crate: Fixed, Vec, Mat, Quat, TrigLUT, SimRng — 316 tests |
| 1 | Memory Management + Entity System | ✅ MOSTLY COMPLETE | FrameAllocator, PoolAllocator, StackAllocator; Python Entity |
| 2 | Archetype ECS Runtime | [~] PARTIAL | TypeRegistry, ComponentStore (SoA) complete; CommandBuffer (Python), Checksum/SystemPhase absent |
| 3 | Task/Job System | [-] NOT IMPLEMENTED | No work-stealing thread pool, JobGraph, or parallel_for |
| 4 | RHI wgpu Mapping | [~] PARTIAL | Python RHI ABCs + Rust wgpu backend exist; no mapping layer |
| 5 | Python-Side Bridge Wiring | ✅ MOSTLY COMPLETE | 14 PyO3 functions, 3-channel protocol LIVE, scheduler deferred |

## 6. Key reference documents

- [GAP_1_SUMMARY.md](GAP_1_SUMMARY.md) — Deep source-code investigation with per-item verification (2026-05-22)
- [PHASE_N_TODO.md](PHASE_N_TODO.md) — Corrected task list with checkmarks and Reality: annotations
- [CLARIFICATION.md](CLARIFICATION.md) — Architectural philosophy, design decisions, divergence analysis
- GAPSET_3_BRIDGE docs — Crossover implementation (omega, ComponentStore, wgpu, WGSL shaders)
