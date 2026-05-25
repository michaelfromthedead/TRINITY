# GAP SET 6: Global Illumination & Reflections -- Reality Summary

## Overview

GAPSET_6 covers global illumination techniques (DDGI, SSGI, Voxel GI) and reflections (SSR, RT reflections, planar reflections, reflection probes). It spans 44 tasks across 11 phases. The plan describes a comprehensive GI/reflection pipeline; the codebase has foundational elements for DDGI in both Python and WGSL/Rust, but most higher-level GI and reflection techniques remain unimplemented.

## Reality Assessment

| Phase | Tasks | Status | Key Findings |
|-------|-------|--------|-------------|
| P1 Foundation | 5 | 3 [~], 2 [-] | SH exists at L0+L1 (WGSL) and L2 (Python). Probe GPU structs exist but at different paths. No budget monitor or reflection buffer. |
| P2 DDGI Core | 9 | 4 [~], 5 [-] | WGSL update/sample shaders exist (simplified). Python has full DDGI reference. No hardware RT, no radiance cache, no lightmap baker. |
| P3 SSGI | 2 | 0 [x], 2 [-] | Nothing built. |
| P4 SSR Core | 5 | 0 [x], 5 [-] | Nothing built. No HiZ buffer, no SSR shaders. |
| P5 Reflection Probes | 6 | 1 [~], 5 [-] | ReflectionProbe class exists with parallax correction (placeholder implementation). No atlas, no pre-filter, no realtime capture. |
| P6 Planar Reflections | 2 | 0 [x], 2 [-] | Nothing built. |
| P7 Voxel GI | 3 | 0 [x], 3 [-] | Nothing built. |
| P8 RT Reflections | 5 | 0 [x], 5 [-] | PassType::RayTracing exists as frame graph infrastructure only. No RT shaders. |
| P9 Denoising | 3 | 0 [x], 3 [-] | Nothing built. |
| P10 Visualization | 1 | 0 [x], 1 [-] | Nothing built. |
| P11 Research | 3 | 0 [x], 3 [-] | No research documents exist. |

**Totals: 0 [x] fully built, 8 [~] partially built, 36 [-] not built**

## Key Source Files

| File | Lines | Reality | Notes |
|------|-------|---------|-------|
| `crates/renderer-backend/src/ddgi.rs` | 303 | REAL | DDGIProbeVolume struct, update+sample pass builders, 20 unit tests |
| `crates/renderer-backend/shaders/ddgi.wgsl` | 240 | REAL | ddgi_update_probes + ddgi_sample_probes compute shaders, L0+L1 SH |
| `engine/rendering/lighting/gi_ddgi.py` | 844 | REAL | Full DDGI Python reference: DDGIProbe, DDGIProbeGrid, DDGIUpdatePass, DDGILookup |
| `engine/rendering/lighting/gi_probes.py` | 779 | REAL | SphericalHarmonics (L2), LightProbe, ProbeGrid, IrradianceVolume, ReflectionProbe |
| `engine/rendering/lighting/constants.py` | 175 | REAL | GIProbeConstants, DDGIConstants classes |
| `engine/rendering/lighting/light_types.py` | ~650 | REAL | GIImportance enum, gi_contributor decorator, SkyLight with cubemap |

## Critical Findings

1. **WGSL SH is L0+L1 only (4 coefficients per channel)**: The plan claims 3rd-order SH (9 coefficients). The actual WGSL shader only implements L0 and L1, which is sufficient for diffuse irradiance but incorrect for the plan's spec. Python reference implements full L2 (9 coefficients, 27 total).

2. **DDGI pass builders exist but are disconnected**: `ddgi.rs` creates IrPass instances for `ddgi_update` and `ddgi_sample`, but they are never connected to an actual frame graph execution. The dispatch counts assume simple N-probe dispatch and 1x1x1 for sampling, which would not work at production resolutions.

3. **DDGI WGSL shader uses placeholder scene data**: The update shader does not trace against an actual depth buffer or geometry. It generates procedural sky/ground colors. The sample shader correctly reads G-buffer world position/normal and performs trilinear interpolation.

4. **All reflection techniques are absent**: No SSR shaders, no HiZ buffer generation, no RT reflection shaders (S10 TLAS/SBT infrastructure doesn't exist), no planar mirror rendering, no voxel cone tracing.

5. **Denoising infrastructure is absent**: No A-trous wavelet, temporal denoiser, or SVGF code exists anywhere in the codebase.

6. **Python reference is disconnected from runtime**: The 1,623 lines of Python GI/DDGI/probe code implement a full reference design but none of it is wired to the WGSL/Rust runtime.

7. **Frame graph supports RayTracing pass type**: `PassType::RayTracing` exists in the IR. `IrPass::ray_tracing()` constructor exists. `ViewType::AccelerationStructure` exists. These are infrastructure-only; no RT shaders or TLAS management code exists.
