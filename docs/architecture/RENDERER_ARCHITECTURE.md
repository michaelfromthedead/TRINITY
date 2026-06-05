# TRINITY Renderer Architecture

**Last Updated:** 2026-05-24  
**Classification:** AAA+ / SOTA  
**Role:** Implementation Guide & Status Tracker

---

## Document Hierarchy

> **This document is a GUIDE, not the source of truth.**
>
> The authoritative specification lives in `engine/rendering/RENDERING_CONTEXT.md` (1,360 lines).
> This document tracks implementation status, design decisions, and the path forward.

```
LAYER 1: CONTEXT.md (14 files, 16,922 lines) — THE SPEC
    │   Defines what the engine SHOULD do. "Phantom" files are BUILD TARGETS.
    │
    ├── RENDERING_CONTEXT.md  (1,360 lines) — Graphics spec
    ├── PLATFORM_CONTEXT.md   (1,256 lines) — RHI/wgpu spec  
    ├── TOOLING_CONTEXT.md    (1,968 lines) — FlowForge/Blueprint
    └── ... 11 more CONTEXT files

LAYER 2: PROJECT.md (55 files) — IMPLEMENTATION TRACKING
    │   Task breakdown, progress status, dependencies.
    │
    ├── gap_sets/ (20 gapsets)
    └── phase_output/ (35 subsystem plans)

LAYER 3: This Document — SUMMARY & DECISIONS
    │   How to navigate the spec. Why we made certain choices.
    │
    └── RENDERER_ARCHITECTURE.md (this file)
```

---

## 1. Authoritative Sources

### Primary Specs (READ FIRST)

| Document | Lines | Purpose |
|----------|-------|---------|
| **RENDERING_CONTEXT.md** | 1,360 | Complete rendering spec: 6 subsystems, all decorators, ALL GI techniques (DDGI, Lumen, Voxel, Path Tracing), visibility buffer, frame graph |
| **PLATFORM_CONTEXT.md** | 1,256 | RHI spec: wgpu/Vulkan/Metal/DX12, device model, command recording, synchronization |
| **TOOLING_CONTEXT.md** | 1,968 | FlowForge (Blueprint analog), material editor, profiler |
| **GAMEPLAY_CONTEXT.md** | 1,191 | GAS-style abilities, behavior trees, utility AI |

### Implementation Tracking

| Gapset | Scope | Status |
|--------|-------|--------|
| GAPSET_1_CORE | ECS, memory, threading | Partial |
| GAPSET_2_FRAME_GRAPH | IR, compiler, async | **Rust complete** |
| GAPSET_3_BRIDGE | Python↔Rust binding | In progress |
| GAPSET_4_MATERIALS | Node graph, variants | Python complete |
| GAPSET_5_LIGHTING | Froxel, 7 light types | Python complete |
| GAPSET_6_GI_REFLECTIONS | DDGI/SSR/Voxel/RT | 8/44 partial |
| GAPSET_7_POST_PROCESS | Tonemap, bloom, DOF | Python complete |
| GAPSET_8_GPU_COMPUTE | Compute pipelines | Partial |
| GAPSET_9_RAY_TRACING | BLAS/TLAS, RT passes | 0/35 built |

---

## 2. Spec Coverage Analysis

### What RENDERING_CONTEXT.md Specifies vs What Exists

**Directories:**
| Spec Directory | Exists | Files Implemented |
|----------------|--------|-------------------|
| framegraph/ | ✓ | 6/6 (100%) |
| gpu_driven/ | ✓ | 6/6 (100%) |
| materials/ | ✓ | 7/7 (100%) |
| lighting/ | ✓ | 6/10 (60%) — missing: virtual_shadow_maps, gi_lumen, reflections, contact_shadows |
| postprocess/ | ✓ | 11/11 (100%) |
| particles/ | ✓ | 7/7 (100%) |
| atmosphere/ | ❌ | 0/5 — BUILD TARGET |
| terrain/ | ❌ | 0/4 — BUILD TARGET |
| water/ | ❌ | 0/4 — BUILD TARGET |
| raytracing/ | ❌ | 0/6 — BUILD TARGET |
| texturing/ | ❌ | 0/3 — BUILD TARGET |
| geometry/ | ❌ | 0/4 — BUILD TARGET |

**Summary:** 7/13 directories exist. 43/67 files implemented (64%). Python algorithms REAL, GPU dispatch in progress.

### GI Technique Support (Per Spec)

RENDERING_CONTEXT.md lines 645-654 specify ALL techniques as capabilities:

| Technique | Spec Status | Implementation |
|-----------|-------------|----------------|
| Baked Lightmaps | Specified | gi_probes.py (partial) |
| Light Probes (SH) | Specified | gi_probes.py (779 lines) ✓ |
| **DDGI** | Specified | gi_ddgi.py (844 lines) ✓, ddgi.wgsl (240 lines) ✓ |
| Voxel GI | Specified | BUILD TARGET |
| Screen-Space GI | Specified | BUILD TARGET |
| **Lumen** | Specified | BUILD TARGET (gi_lumen.py) |
| Path Tracing | Specified | BUILD TARGET |

> **Note:** The spec supports ALL techniques. Design decisions below explain implementation order.

---

## 3. Implementation Status

### Python Algorithms (600K+ lines REAL)

| Subsystem | Lines | Status | Key Content |
|-----------|-------|--------|-------------|
| gpu_driven/ | 4,859 | ✓ REAL | Visibility buffer, meshlets, culling, bindless |
| lighting/ | 4,470 | ✓ REAL | DDGI, froxel, shadows (CSM/cube/spot) |
| materials/ | 5,976 | ✓ REAL | Node graph, 22 shader functions, PBR |
| framegraph/ | 3,524 | ✓ REAL | Pass declaration, dependency analysis |
| particles/ | 5,982 | ✓ REAL | VFX Graph, emitters, GPU simulation |
| postprocess/ | 8,861 | ✓ REAL | Tonemap, bloom, DOF, TAA, color grading |

### Rust Backend (wgpu)

| Component | Status | Location |
|-----------|--------|----------|
| Frame Graph IR | ✓ COMPLETE | `crates/renderer-backend/src/frame_graph/` (108KB) |
| Frame Graph Compiler | ✓ COMPLETE | Topo sort, barriers, async scheduling |
| Frame Graph Executor | ✓ COMPLETE | `executor.rs` — resource alloc, pass execution |
| Headless Rendering | ✓ COMPLETE | `headless.rs` — offscreen render + readback |
| RHI Device | ✓ COMPLETE | `rhi_device.rs` — wgpu abstraction |
| DDGI Passes | ✓ COMPLETE | `ddgi.rs` (303 lines, 20 tests) |
| Material System | ○ NOT WIRED | Phase 3 target |
| Mesh System | ○ NOT WIRED | Phase 4 target |

### WGSL Shaders

| Shader | Lines | Wired to Executor |
|--------|-------|-------------------|
| pbr.frag.wgsl | 377 | ❌ Not yet |
| light_culling.wgsl | 229 | ❌ Not yet |
| ddgi.wgsl | 240 | ❌ Not yet |
| shadow_csm.wgsl | 161 | ❌ Not yet |

---

## 4. wgpu Execution Plan

### RHI Architecture (Per PLATFORM_CONTEXT.md lines 566-605)

```
Adapter → Device → Queues (Graphics, Compute, Transfer)
                    │
                    ├── Resources: Buffer, Texture, Sampler, View
                    ├── Pipeline: Shader, PSO, Root Signature
                    ├── Commands: Command List, Queue, Indirect
                    ├── Binding: Descriptor Heap, Bindless, Push Constants
                    └── Sync: Fence, Semaphore, Barrier
```

### 5-Phase Implementation

| Phase | Goal | Status |
|-------|------|--------|
| **1. Headless Rendering** | Render to texture, CI-testable | ✓ COMPLETE |
| **2. Frame Graph Executor** | Execute CompiledFrameGraph as GPU commands | ✓ COMPLETE |
| **3. Material Pipeline** | Wire MaterialTable to PBR shader | ○ NEXT |
| **4. Mesh Rendering** | Wire MeshTable to draw calls | ○ Pending |
| **5. Python Integration** | Python triggers real frame rendering | ○ Pending |

### Phase 3 Deliverables (Current Focus)

```rust
// MaterialSystem — wire to pbr.frag.wgsl
pub struct MaterialSystem {
    material_buffer: wgpu::Buffer,      // MaterialTableEntry[]
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

// Tasks:
// [ ] Material buffer upload from Python MaterialTable
// [ ] PBR pipeline creation with pbr.frag.wgsl
// [ ] Bind group for material buffer
// [ ] Test: Render colored cube with PBR material
```

See `WGPU_EXECUTION_PLAN.md` for detailed implementation code.

---

## 5. Design Decisions

> These are implementation choices for THIS project, not spec limitations.
> The spec (RENDERING_CONTEXT.md) supports all techniques listed.

### Why DDGI First (before Lumen)?

The spec supports both. We implement DDGI first because:
- More portable (no mesh card/SDF preprocessing)
- Lower memory footprint
- Simpler integration path
- Ray tracing can enhance DDGI, not replace it
- **Lumen remains a BUILD TARGET** per gi_lumen.py spec

### Why Visibility Buffer (not thick G-Buffer)?

- Single geometry pass for unlimited materials
- Smaller bandwidth than 5-texture G-buffer
- Natural fit for meshlet/Nanite-style culling
- Enables material-sorted shading

### Why Python Reference + Rust Backend?

- Python for rapid algorithm iteration (600K lines)
- Rust for GPU dispatch performance
- Frame graph IR decouples them cleanly
- Shared WGSL shaders

### Why wgpu?

- Cross-platform: Vulkan, Metal, DX12, WebGPU
- Rust-native with safe abstractions
- Ray tracing support (experimental)
- Active development, WebGPU convergence

---

## 6. Rendering Stack (Current)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRINITY RENDERING STACK                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  PYTHON ENGINE LAYER (600K+ lines REAL)                                     │
│  ├── engine/rendering/gpu_driven/      Visibility buffer, meshlets, culling │
│  ├── engine/rendering/lighting/        DDGI, froxel, shadows                │
│  ├── engine/rendering/materials/       Node graph, PBR                      │
│  ├── engine/rendering/framegraph/      Dependency analysis                  │
│  ├── engine/rendering/particles/       VFX Graph                            │
│  └── engine/rendering/postprocess/     Tonemap, DOF, bloom                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  WGSL SHADER LAYER (partial — not wired to executor yet)                    │
│  ├── pbr.frag.wgsl              Cook-Torrance BRDF                          │
│  ├── light_culling.wgsl         Froxel grid                                 │
│  ├── ddgi.wgsl                  Probe update + sample                       │
│  └── shadow_csm.wgsl            Cascade select, PCF                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  RUST BACKEND LAYER (wgpu-based)                                            │
│  ├── frame_graph/               ✓ IR, compiler, barriers, async             │
│  ├── executor.rs                ✓ Pass execution, resource alloc            │
│  ├── headless.rs                ✓ Offscreen rendering, readback             │
│  ├── ddgi.rs                    ✓ Pass builders, 20 unit tests              │
│  └── rhi_device.rs              ✓ wgpu device abstraction                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  wgpu (Vulkan / Metal / DX12 / WebGPU)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Canonical Pass Order

From RENDERING_CONTEXT.md — the standard frame structure:

| Order | Pass | Type | Notes |
|-------|------|------|-------|
| 1 | Shadow Atlas | Graphics | CSM + point/spot shadows |
| 2 | G-Buffer / Visibility | Graphics | Visibility buffer preferred |
| 3 | SSAO | Compute | Depth + normal → AO |
| 4 | Light Culling | Compute | Build froxel light lists |
| 5 | Lighting | Compute | G-Buffer + shadows + AO → HDR |
| 6 | Transparent | Graphics | Forward pass, OIT optional |
| 7 | Post-Process | Compute | Bloom, DOF, TAA, tonemap |
| 8 | UI | Graphics | Overlay |
| 9 | Present | — | Swap chain |

---

## 8. Quick Reference

### Find Technique Details

| Topic | Location |
|-------|----------|
| Visibility Buffer | RENDERING_CONTEXT.md §6.2, gpu_driven/visibility_buffer.py |
| Meshlet/Nanite | RENDERING_CONTEXT.md §6.7, gpu_driven/meshlet.py |
| DDGI | RENDERING_CONTEXT.md §6.4, lighting/gi_ddgi.py |
| Lumen (spec) | RENDERING_CONTEXT.md lines 662-666 |
| Froxel Lighting | RENDERING_CONTEXT.md §6.3, lighting/light_culling.py |
| PBR BRDF | materials/pbr_model.py, pbr.frag.wgsl |
| Frame Graph | GAPSET_2_FRAME_GRAPH/, crates/renderer-backend/src/frame_graph/ |
| RHI | PLATFORM_CONTEXT.md §6.3 |

### Unreal Engine 5 Parallels

| UE5 Feature | TRINITY Location | Status |
|-------------|------------------|--------|
| Nanite | gpu_driven/visibility_buffer.py, meshlet.py | Python REAL |
| Lumen | lighting/gi_lumen.py | BUILD TARGET |
| Niagara | particles/vfx_graph.py | Python REAL |
| Blueprint | flowforge/ (TOOLING_CONTEXT.md) | Spec only |
| GAS | gameplay/abilities/ | Python REAL |
| Virtual Shadow Maps | lighting/virtual_shadow_maps.py | BUILD TARGET |

---

## 9. Gap Analysis Summary

### Implemented (Python algorithms REAL)
- Visibility buffer rendering
- Meshlet pipeline
- GPU culling (frustum, HZB, distance)
- Bindless resources
- DDGI global illumination
- Froxel clustered lighting
- CSM/cube/spot shadows
- PBR materials with node graph
- VFX particle system
- Full post-processing stack

### In Progress (Rust/wgpu wiring)
- Frame graph executor ✓
- Material system binding
- Mesh system binding
- Shader wiring

### Build Targets (Spec defined, not yet built)
- Lumen GI (gi_lumen.py)
- Virtual shadow maps
- RT shadows/reflections/GI
- Voxel GI
- Screen-space GI
- Path tracing
- atmosphere/, terrain/, water/, raytracing/ directories

---

*Last Updated: 2026-05-24*  
*Authoritative Spec: engine/rendering/RENDERING_CONTEXT.md*  
*Implementation: 64% of spec files, 600K+ lines Python*
