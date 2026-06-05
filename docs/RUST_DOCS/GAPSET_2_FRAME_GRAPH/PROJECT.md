# GAPSET_2_FRAME_GRAPH -- Project Overview

## Scope

GAPSET_2_FRAME_GRAPH covers the **frame graph compiler** subsystem for the TRINITY rendering engine. This project implements the render pass declaration, dependency scheduling, automatic barrier insertion, resource aliasing, async compute scheduling, and dead pass elimination pipeline that forms the core compilation workflow for GPU frame rendering.

### Boundaries

- **In scope**: IR types and conversion (Phase 1), dependency DAG construction and topological sort (Phase 2), resource lifetime computation and memory aliasing (Phase 3), automatic GPU barrier insertion (Phase 4), async compute scheduling (Phase 5), dead pass elimination (Phase 6), Python-Rust bridge protocol (Phase 7).
- **Out of scope**: Actual wgpu device/queue/command buffer management, shader compilation, material systems, render passes outside the frame graph (e.g., presentation swap chain management), user-facing rendering effects.

## Architecture Overview

The frame graph subsystem has **two independent implementations** connected by a JSON serialization bridge:

```
Python (engine/rendering/framegraph/)     Rust (renderer-backend/src/frame_graph/)
    |                                              |
    |-- Pass declaration                           |-- IR types (IrPass, IrResource, IrEdge)
    |-- Resource management                        |-- Phase 2: DAG builder
    |-- Barrier management                         |-- Phase 2b: Topological sort (Kahn's)
    |-- Async scheduling                           |-- Phase 3: Resource lifetimes
    |-- Dead pass culling                          |-- Phase 4: Barrier insertion
    |-- serialize() ----> JSON ----> deserialize_from_json()
    |                                              |-- Phase 5: Async scheduling
    |                                              |-- Phase 6: Dead pass elimination
    |                                              |-- execute() -> JSON statistics
```

### Current Architecture Strength
- Rust compiler (GAP 3): 3,156 lines, all 6 phases, production-quality DAG/topological sort
- Python frame graph: 7 modules, all 7 phases, strong resource/barrier/scheduler management

### Current Architecture Gap
- No PyO3 FFI: Python and Rust operate independently
- JSON bridge exists (`serialize()` in Python, `deserialize_from_json()` in Rust) but is not wired into any automated pipeline
- No wgpu command generation -- barrier records exist but cannot drive actual GPU commands

## Goals

### Primary
1. **Complete the frame graph compiler** -- verify and complete all 47 tasks across 7 phases
2. **Unify Python and Rust paths** -- connect Python pass declarations through the Rust compiler via the existing JSON bridge (or proper PyO3 FFI)
3. **Enable wgpu execution** -- generate actual GPU commands from compiled frame graphs

### Secondary
4. Achieve 40%+ memory savings through resource aliasing
5. Support async compute for parallel GPU execution
6. Provide comprehensive diagnostics and debug tooling

## Phase Overview

### Phase 1: Compiler Foundation + IR (44% complete)
Core data structures for the entire compiler. Rust `IrPass`, `IrResource`, `IrEdge` are fully defined. Python->Rust conversion exists for passes but not resources. `View` trait entirely missing.

### Phase 2: Dependency DAG (64% complete)
Strongest phase. DAG builder classifies RAW/WAR/WAW edges. Kahn's algorithm with cycle detection. Missing: topological depth, parallel region identification.

### Phase 3: Resource Aliasing (39% complete)
Lifetime intervals computed. External resource import works. Missing: wgpu resource allocator, allocation table, proper graph coloring.

### Phase 4: Barrier Insertion (56% complete)
State tracking, barrier record generation, and batching work. Missing: wgpu command generation, redundant elimination.

### Phase 5: Async Compute Scheduling (31% complete)
Candidate identification works. Missing: secondary timeline builder, sync point wiring, feature gating, serial fallback, all tests.

### Phase 6: Dead Pass Elimination (50% complete)
Reverse reachability and pass removal work. Missing: formal live output set, culling statistics, dynamic toggle.

### Phase 7: Bridge + Emit (25% complete)
CompiledFrameGraph struct exists. CompilationResult exists (partial). All 3 bridge channels (Type, Data, Command) are absent. No ArcSwap, no CLI debug flags.

## Implementation Status (All Gaps)

| Gap ID | Severity | Status | Key Missing Work |
|--------|----------|--------|-----------------|
| S1-G1 | CRITICAL | Partial | PyResourceDesc conversion, PyO3 compile function, mock constructors |
| S1-G2 | CRITICAL | Absent | View trait with bind(), CameraView, Box<dyn View> on passes |
| S1-G3 | CRITICAL | Most absent | All 3 bridge channels, ArcSwap, CLI debug flags |
| S1-G4 | HIGH | Strong | Depth assignment, parallel regions |
| S1-G5 | HIGH | Weak | wgpu resource allocator, allocation table, graph coloring |
| S1-G6 | HIGH | Moderate | wgpu command generation, redundant barrier elimination |
| S1-G7 | MEDIUM | Weak | Feature gating, serial fallback, sync point wiring, tests |
| S1-G8 | MEDIUM | Moderate | Live output set, statistics, dynamic toggle |

**Overall completion: 43% weighted (16 [x], 17 [~], 24 [-] out of 57 tasks)**
