# GAP_9_SUMMARY.md — GAPSET_9_RAY_TRACING

> **Generated**: 2026-05-22 — RDC (Review-Document-Correct) pass
> **Methodology**: Read `PHASE_N_TODO.md` + source-code investigation at key code locations.
> **Key Locations**: `crates/renderer-backend/src/frame_graph/mod.rs`, `engine/platform/rhi/raytracing.py`, `engine/rendering/framegraph/pass_node.py`, `engine/rendering/framegraph/frame_graph.py`, `engine/platform/rhi/pipeline.py`, `engine/platform/rhi/device.py`, `engine/platform/constants.py`, `tests/`

---

## Summary

GAPSET_9_RAY_TRACING covers ray tracing support through hardware RT via wgpu: acceleration structures, ray queries, ray tracing pipelines, denoising, and fallback chains. It is split into 3 phases spanning 35 tasks.

## Reality Assessment — What Exists vs What Is Planned

### Already Implemented (Frame Graph / IR / RHI stubs / Tests)

| Component | Location | Status |
|-----------|----------|--------|
| `PassType::RayTracing` enum variant | `frame_graph/mod.rs:94` | [x] Complete |
| `IrPass::ray_tracing()` constructor | `frame_graph/mod.rs:869-892` | [x] Complete |
| `ResourceState::AccelerationStructure` variant | `frame_graph/mod.rs:998` | [x] Complete |
| `ViewType::AccelerationStructure` variant | `frame_graph/mod.rs:541` | [x] Complete |
| JSON deserialize of "RayTracing" pass_type | `frame_graph/mod.rs:1863` | [x] Complete |
| Python bridge `PyPassType::RayTracing` | `frame_graph/python.rs:113` | [x] Complete |
| `RayTracingPass` Python class | `pass_node.py:581-696` | [x] Complete |
| `FrameGraph.add_raytracing_pass()` | `frame_graph.py:357-366` | [x] Complete |
| `RaytracingPipelineDesc` dataclass | `pipeline.py:161-166` | [x] Complete (stub) |
| `PipelineType.RAYTRACING` enum | `pipeline.py:173` | [x] Complete |
| `BLASDesc`, `TLASDesc` dataclasses | `raytracing.py:20-36` | [x] Complete |
| `AccelerationStructure` ABC | `raytracing.py:38-63` | [x] Complete |
| `NullAccelerationStructure` | `raytracing.py:65-98` | [x] Complete |
| `BuildFlags` enum | `raytracing.py:12-16` | [x] Complete |
| `GPU_ADDRESS_START`, `ACCELERATION_STRUCTURE_ALIGNMENT` | `constants.py:71-72` | [x] Complete |
| `FeatureSupport.ray_tracing` bool | `device.py` | [x] Complete |
| Frame graph async scheduling RayTracing exclusion | `frame_graph/mod.rs:1611` | [x] Complete |
| Rust unit tests for RayTracing pass | `frame_graph/mod.rs:2403-2416` | [x] Complete |
| Python unit test: `test_add_raytracing_pass` | `tests/test_frame_graph.py:66-74` | [x] Complete |
| Python unit test: `capability(requires={"ray_tracing"})` | `tests/test_compilation.py:316` | [x] Complete |
| Python unit test: device ray_tracing feature | `tests/test_device.py:38,47` | [x] Complete |

### Not Yet Implemented (Planned for Phase 1-3)

| Component | Phase | Location Planned | Status |
|-----------|-------|-----------------|--------|
| Rust BLAS build/refit/compact dispatch | P1 | crates/renderer-backend/src/rt/ | [-] Not started |
| Rust TLAS build dispatch | P1 | crates/renderer-backend/src/rt/ | [-] Not started |
| WGSL ray query shadow shader | P1 | shaders/rt_shadow.comp.wgsl | [-] Not started |
| WGSL any-hit alpha test shader | P1 | shaders/ (inline in ray query) | [-] Not started |
| Python RT shadow dispatch | P1 | engine/rendering/rt/ | [-] Not started |
| A Trous spatial denoiser | P1 | shaders/denoiser_spatial.comp.wgsl | [-] Not started |
| BLASPool with reference counting | P1 | engine/platform/rhi/raytracing.py | [-] Not started |
| Instance buffer management | P1 | engine/rendering/rt/ | [-] Not started |
| Ray budget management | P1 | engine/rendering/rt/ | [-] Not started |
| Fallback: RT shadows -> CSM+PCSS | P1 | engine/rendering/lighting/ | [-] Not started |
| Rust RT pipeline creation | P2 | crates/renderer-backend/src/rt/ | [-] Not started (gated) |
| SBT builder | P2 | crates/renderer-backend/src/rt/ | [-] Not started (gated) |
| RT reflection pipeline shaders | P2 | shaders/rt_reflection.*.wgsl | [-] Not started (gated) |
| RT GI pipeline shaders | P2 | shaders/rt_gi.*.wgsl | [-] Not started (gated) |
| Python RT reflection dispatch | P2 | engine/rendering/rt/ | [-] Not started (gated) |
| Python RT GI dispatch | P2 | engine/rendering/rt/ | [-] Not started (gated) |
| Temporal denoising shader | P2 | shaders/denoiser_temporal.comp.wgsl | [-] Not started (gated) |
| Joint bilateral filter shader | P2 | shaders/denoiser_bilateral.comp.wgsl | [-] Not started (gated) |
| Three-stage denoiser pipeline | P2 | engine/rendering/rt/ | [-] Not started (gated) |
| Bindless material table | P2 | engine/rendering/gpu_driven/ | [-] Not started (gated) |
| Reflection fallback chain | P2 | engine/rendering/lighting/ | [-] Not started (gated) |
| GI fallback chain | P2 | engine/rendering/lighting/ | [-] Not started (gated) |
| Adaptive quality system | P2 | engine/rendering/rt/ | [-] Not started (gated) |
| S1 frame graph RT integration | P2 | engine/rendering/framegraph/ | [-] Not started (gated) |
| Full path tracing shader | P3 | shaders/rt_pathtrace.comp.wgsl | [-] Not started |
| Path tracing temporal accumulation | P3 | engine/rendering/rt/ | [-] Not started |
| Neural denoising U-Net shader | P3 | shaders/denoiser_neural.comp.wgsl | [-] Not started |
| Denoising model architecture survey | P3 | docs/research/ | [-] Not started |
| wgpu OMM/DMM timeline research | P3 | docs/research/ | [-] Not started |
| Adaptive quality 2.0 ML prediction | P3 | engine/rendering/rt/ | [-] Not started |

## Key Findings

1. **Frame graph IR is well-prepared**: `PassType::RayTracing`, `ViewType::AccelerationStructure`, and `ResourceState::AccelerationStructure` are all first-class variants in the IR with matching serialization and tests.

2. **Python RHI stubs are in place**: `BLASDesc`, `TLASDesc`, `AccelerationStructure` ABC, `NullAccelerationStructure`, `BuildFlags`, `RaytracingPipelineDesc`, and `PipelineType.RAYTRACING` all exist as stubs.

3. **No RT backend code exists**: There is no `wgpu::RayTracingAccelerationStructure` usage, no BLAS/TLAS Rust build functions, no ray query dispatch, no WGSL shaders, and no ray tracing pipeline creation.

4. **Phase 1 is unblocked**: The wgpu `ray_query` and `acceleration_structure` features are available, so Phase 1 (inline ray queries for shadow rays) could theoretically be implemented now.

5. **Phase 2 is gated**: 4 tasks (T-RT-P2.1, T-RT-P2.2, T-RT-P2.3, T-RT-P2.6) are correctly flagged as platform-gated on wgpu `ray_tracing_pipeline` stability, which is still experimental.

6. **Phase 3 is future work**: Full path tracing and neural denoising depend on Phase 2 completion plus wgpu NPU/tensor core extensions.

## Task Count

- **Total Tasks**: 35 (Phase 1: 15, Phase 2: 14, Phase 3: 6)
- **Already Implemented (foundation)**: 19 items (IR types, RHI stubs, tests)
- **Not Yet Implemented**: 30 planned code items
- **Research Tasks**: 2 (T-RT-P3.4 denoising survey, T-RT-P3.5 OMM/DMM timeline)
- **Platform-Gated Tasks**: 4 (T-RT-P2.1, T-RT-P2.2, T-RT-P2.3, T-RT-P2.6)

## Cross-Phase Dependencies (Verified)

```
Phase 1 (foundation exists, backend work needed):
  T-RT-P1.4 (capability detect) → T-RT-P1.8 (fallback chain)
  T-RT-P1.1 (AS stubs [x]) → T-RT-P1.2 (BLAS Rust [-])
  T-RT-P1.2 → T-RT-P1.3 (TLAS Rust [-])
  T-RT-P1.3 → T-RT-P1.5 (shadow shader [-])
  T-RT-P1.9 (spatial denoise [-]) → T-RT-P1.10 (denoiser dispatch [-])

Phase 2 (gated):
  T-RT-P2.1 (RT pipeline) → T-RT-P2.2 (SBT) → T-RT-P2.3 (reflection shaders)
  T-RT-P2.1 → T-RT-P2.4 (GI shaders) → T-RT-P2.6 (GI dispatch)

Phase 3:
  T-RT-P3.4 (research) → T-RT-P3.3 (neural denoising)
  T-RT-P2.4 → T-RT-P3.1 (path tracing)
```

## Reality Corrections to PHASE_N_TODO.md

The following corrections must be applied to the TODO document:

1. **T-RT-P1.1 Reality**: `BLASDesc` (raytracing.py:20-28), `TLASDesc` (raytracing.py:30-36) exist as stubs. `BLASPool`, `BLASManager`, `TLASManager` do NOT exist. Change description to "Extend existing BLASDesc/TLASDesc stubs with manager classes".

2. **T-RT-P1.4 Reality**: `FeatureSupport.ray_tracing` exists in device.py but `RTCapability` enum and `get_rt_capability()` function do not exist. The feature detection is a simple bool, not a 3-level enum.

3. **T-RT-P1.7 Reality**: Python `RayTracingPass` class exists (pass_node.py:581-696) with dispatch dimensions, TLAS, SBT, output. `RTShadows` class does NOT exist.

4. **T-RT-P2.14 Reality**: The `RayTracingPass` node exists in the frame graph Python code, but no RT effects are wired into the S1 pipeline. Integration is partial — the scaffolding exists, no effects are registered.

5. **No WGSL shaders found at all**: A search for `.wgsl`, `.frag`, `.vert`, `.comp` files found nothing in the repository. All shader tasks are unimplemented.
