# PROJECT.md — GAPSET_9_RAY_TRACING

> **Purpose**: Ray tracing support through hardware RT via wgpu: acceleration structures, ray queries, ray tracing pipelines, denoising, and fallback chains.
> **Gap Set**: S10 — Ray Tracing
> **Phases**: 3 (P1: Inline ray queries, P2: Full RT pipeline, P3: Path tracing + neural denoising)
> **Total Tasks**: 35 (P1: 15, P2: 14, P3: 6)
> **Status**: Foundation laid (IR, RHI stubs, frame graph). Backend implementation [~] started.
> **RDC**: 2026-05-22

---

## 1. Project Overview

GAPSET_9 implements hardware-accelerated ray tracing for the TRINITY engine via the wgpu graphics API. It provides:

- **Acceleration Structures**: Bottom-level (BLAS) and top-level (TLAS) building blocks for ray tracing.
- **Ray Queries**: Inline ray queries for shadow rays (Phase 1, available now).
- **Ray Tracing Pipelines**: Full raygen/hit/miss pipeline model for reflections and GI (Phase 2, gated).
- **Denoising**: Spatial, temporal, bilateral, and neural denoising.
- **Fallback Chains**: Graceful degradation when RT is unavailable.
- **Adaptive Quality**: Frame-time feedback loop for dynamic quality adjustment.

## 2. Architecture

### 2.1 Layer Structure

```
Python Engine Layer
  engine/rendering/framegraph/
    pass_node.py        RayTracingPass class [x]
    frame_graph.py      add_raytracing_pass() [x]
  engine/platform/rhi/
    raytracing.py       BLASDesc, TLASDesc, AccelerationStructure [x]
    pipeline.py         RaytracingPipelineDesc [x]
    device.py           FeatureSupport.ray_tracing [x]
    __init__.py         Re-exports RT types [x]

Rust Backend Layer
  crates/renderer-backend/src/frame_graph/
    mod.rs              PassType::RayTracing, IrPass::ray_tracing() [x]
                        ResourceState::AccelerationStructure [x]
                        ViewType::AccelerationStructure [x]
                        JSON deserialize "RayTracing" passes [x]
    python.rs           PyPassType::RayTracing bridge [x]

Future Rust RT Backend (planned)
  crates/renderer-backend/src/rt/     BLAS/TLAS/SBT/pipeline [-]

Shader Layer (planned)
  shaders/rt_*.wgsl                  Ray query, RT pipeline, denoisers [-]
```

### 2.2 Frame Graph Integration

The `RayTracingPass` node declares:
- **dispatch_width/height/depth**: Ray dispatch dimensions (typically screen size).
- **tlas**: Top-level acceleration structure handle.
- **shader_binding_table**: SBT handle for hit/miss shader dispatch.
- **max_recursion_depth**: Maximum ray bounce count.
- **reads/writes**: Standard resource dependency tracking.

### 2.3 Fallback Chain Design

```
RT Shadows       —> CSM + PCSS + Contact Shadows
RT Reflections   —> SSR (HiZ) + Reflection Probes
RT GI            —> DDGI + SSGI
```

## 3. Current Implementation Status

### 3.1 What Works (Verified on Disk)

| Component | File | Lines | Details |
|-----------|------|-------|---------|
| `PassType::RayTracing` enum | `frame_graph/mod.rs` | 94 | IR pass type variant |
| `IrPass::ray_tracing()` | `frame_graph/mod.rs` | 869-892 | Constructor with dispatch |
| `ResourceState::AccelerationStructure` | `frame_graph/mod.rs` | 998 | State tracking for AS |
| `ViewType::AccelerationStructure` | `frame_graph/mod.rs` | 541 | Binding type for AS |
| JSON deser "RayTracing" | `frame_graph/mod.rs` | 1863 | Pass_type parsing |
| `PyPassType::RayTracing` | `python.rs` | 113 | Python bridge |
| `RayTracingPass` class | `pass_node.py` | 581-696 | Full Python pass node |
| `FrameGraph.add_raytracing_pass()` | `frame_graph.py` | 357-366 | Pass registration |
| `RaytracingPipelineDesc` | `pipeline.py` | 161-166 | Pipeline descriptor stub |
| `PipelineType.RAYTRACING` | `pipeline.py` | 173 | Pipeline type enum |
| `BLASDesc` dataclass | `raytracing.py` | 20-28 | BLAS descriptor |
| `TLASDesc` dataclass | `raytracing.py` | 30-36 | TLAS descriptor |
| `AccelerationStructure` ABC | `raytracing.py` | 38-63 | Abstract base |
| `NullAccelerationStructure` | `raytracing.py` | 65-98 | Null implementation |
| `BuildFlags` enum | `raytracing.py` | 12-16 | Flag values |
| `FeatureSupport.ray_tracing` | `device.py` | - | Feature detection bool |
| Rust unit test (RT pass) | `frame_graph/mod.rs` | 2403-2416 | RT pass constructor |
| Python test (RT pass) | `tests/test_frame_graph.py` | 66-74 | RT pass creation |
| Python test (RT feature) | `tests/test_device.py` | 38,47 | Feature detection |
| `GPU_ADDRESS_START` | `constants.py` | 71 | GPU address base |
| `ACCELERATION_STRUCTURE_ALIGNMENT` | `constants.py` | 72 | 64KB alignment |

### 3.2 What Needs Implementation (Phase 1)

| Task | Effort | Priority | Dependencies |
|------|--------|----------|-------------|
| BLASManager/TLASManager/BLASPool | 5 | High | T-RT-P1.1 |
| Rust BLAS build/refit/compact | 8 | High | T-RT-P1.2 |
| Rust TLAS build | 5 | High | T-RT-P1.3 |
| RTCapability enum + getter | 3 | High | T-RT-P1.4 |
| RT shadow ray query WGSL | 8 | High | T-RT-P1.5 |
| Any-hit alpha test (inline) | 5 | Medium | T-RT-P1.6 |
| Python RT shadow dispatch | 5 | High | T-RT-P1.7 |
| Shadow fallback chain | 8 | Medium | T-RT-P1.8 |
| A Trous spatial denoiser WGSL | 8 | Medium | T-RT-P1.9 |
| Python denoiser dispatch | 3 | Medium | T-RT-P1.10 |
| BLASPool ref counting | 5 | High | T-RT-P1.11 |
| Instance buffer management | 5 | High | T-RT-P1.12 |
| Ray budget management | 3 | Medium | T-RT-P1.13 |
| Static mesh BLAS on load | 5 | High | T-RT-P1.14 |
| Dynamic mesh BLAS refit | 5 | Medium | T-RT-P1.15 |

## 4. Files Reference

### Source Files (Rust)

| Path | Status | Purpose |
|------|--------|---------|
| `crates/renderer-backend/src/frame_graph/mod.rs` | [x] | IR types, PassType::RayTracing, IrPass::ray_tracing() |
| `crates/renderer-backend/src/frame_graph/python.rs` | [x] | PyPassType bridge, JSON deserialization |

### Source Files (Python)

| Path | Status | Purpose |
|------|--------|---------|
| `engine/platform/rhi/raytracing.py` | [x] | BLASDesc, TLASDesc, AccelerationStructure, NullAccelerationStructure |
| `engine/platform/rhi/pipeline.py` | [x] | RaytracingPipelineDesc, PipelineType.RAYTRACING |
| `engine/platform/rhi/device.py` | [x] | FeatureSupport.ray_tracing |
| `engine/platform/rhi/__init__.py` | [x] | Module re-exports |
| `engine/platform/constants.py` | [x] | GPU_ADDRESS_START, ACCELERATION_STRUCTURE_ALIGNMENT |
| `engine/rendering/framegraph/pass_node.py` | [x] | RayTracingPass class |
| `engine/rendering/framegraph/frame_graph.py` | [x] | add_raytracing_pass() |
| `engine/rendering/__init__.py` | [x] | RayTracingPass re-export |

### Test Files

| Path | Status | Purpose |
|------|--------|---------|
| `crates/renderer-backend/src/frame_graph/mod.rs` | [x] | test_ir_pass_ray_tracing_constructor |
| `tests/rendering/framegraph/test_frame_graph.py` | [x] | test_add_raytracing_pass |
| `tests/platform/rhi/test_device.py` | [x] | Ray tracing feature tests |
| `tests/trinity/test_compilation.py` | [x] | capability(requires={"ray_tracing"}) |

## 5. Dependencies

### Internal Dependencies

- **S1 (Frame Graph)**: RayTracingPass node depends on the frame graph IR.
- **S12 (RHI)**: Ray tracing types depend on the RHI layer (Buffer, Device).
- **S6 (Global Illumination)**: RT GI fallback chain depends on DDGI/SSGI.
- **S7 (Reflections)**: RT reflection fallback chain depends on SSR/probes.
- **S16 (Asset Pipeline)**: Static mesh BLAS on load integration.

### External Dependencies

- **wgpu**: The entire RT implementation depends on wgpu's ray tracing features.
  - Phase 1: `ray_query` + `acceleration_structure` — available now.
  - Phase 2: `ray_tracing_pipeline` + `shader_binding_table` — experimental.
  - Phase 3: NPU/tensor core extensions — on wgpu roadmap.
- **Naga**: WGSL shader compilation pipeline for RT shaders.
