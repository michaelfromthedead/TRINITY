# GAPSET 6: Global Illumination & Reflections -- Project Overview

## Scope

Implement a comprehensive global illumination and reflection pipeline for the Trinity renderer, covering diffuse indirect lighting (DDGI, SSGI, Voxel GI) and specular reflections (SSR, RT reflections, planar reflections, reflection probes). The pipeline spans 44 tasks across 11 phases, with a dependency chain from foundation infrastructure through advanced research.

## Architecture Overview

```
G-Buffer (diffuse albedo, world position, world normal, roughness, metallic)
    |
    +---> Phase 2: DDGI (probe grid update + sample) ---> Indirect diffuse
    |         |
    |         +---> Phase 7: Voxel GI (cone tracing) ---> Indirect diffuse + specular
    |
    +---> Phase 3: SSGI (screen-space ray march) ---> Indirect diffuse
    |
    +---> Phase 4: SSR (HiZ ray march + temporal) ---> Screen-space reflections
    |
    +---> Phase 5: Reflection Probes (cubemap capture + blend) ---> Environment reflections
    |
    +---> Phase 6: Planar Reflections (mirror render) ---> Planar reflections
    |
    +---> Phase 8: RT Reflections (hardware ray tracing) ---> Hardware-accelerated reflections
    |
    +---> Phase 9: Denoising (A-trous + temporal + SVGF) ---> Denoised output
    |
    +---> Phase 10: Visualization (debug overlay) ---> Debug views
```

## Phase Dependency Graph

```
Phase 1 (Foundation)
  |
  v
Phase 2 (DDGI Core) ---> Phase 3 (SSGI) ---> Phase 10 (Visualization)
  |
  |                         v
  |                      Phase 4 (SSR Core)
  |                         |
  |                         v
  |                      Phase 5 (Reflection Probes)
  |                         |
  |                         v
  |                      Phase 6 (Planar Reflections)
  |
  +---> Phase 7 (Voxel GI) ---> Phase 11.2 (SVO Research)
  |
  +---> Phase 8 (RT Reflections) <--- Phase 9 (Denoising)
  |
  +---> Phase 11.1 (Adaptive DDGI Research)

Phase 9 (Denoising) feeds into DDGI temporal, SSGI temporal, SSR temporal, RT denoising
```

## File Layout

### Existing Source Files (Foundational)

| File | Lines | Purpose |
|------|-------|---------|
| `crates/renderer-backend/src/ddgi.rs` | 303 | DDGIProbeVolume struct, update+sample pass builders, 20 unit tests |
| `crates/renderer-backend/shaders/ddgi.wgsl` | 240 | ddgi_update_probes + ddgi_sample_probes compute shaders (L0+L1 SH) |
| `engine/rendering/lighting/gi_ddgi.py` | 844 | Full DDGI Python reference: DDGIProbe, DDGIProbeGrid, DDGIUpdatePass, DDGILookup |
| `engine/rendering/lighting/gi_probes.py` | 779 | SphericalHarmonics (L2), LightProbe, ProbeGrid, IrradianceVolume, ReflectionProbe |
| `engine/rendering/lighting/constants.py` | 175 | GIProbeConstants, DDGIConstants |
| `engine/rendering/lighting/light_types.py` | ~650 | GIImportance enum, gi_contributor decorator, SkyLight |

### Needed Source Files (Not Yet Built)

| File | Phase | Purpose |
|------|-------|---------|
| `crates/renderer-backend/shaders/ssgi_trace.comp.wgsl` | P3 | SSGI ray marching |
| `crates/renderer-backend/shaders/ssgi_temporal.comp.wgsl` | P3 | SSGI temporal accumulation |
| `crates/renderer-backend/shaders/hiz_generate.comp.wgsl` | P4 | HiZ buffer generation |
| `crates/renderer-backend/shaders/ssr_ray_march.comp.wgsl` | P4 | SSR HiZ ray marching |
| `crates/renderer-backend/shaders/ssr_ray_march_linear.comp.wgsl` | P4 | SSR linear fallback |
| `crates/renderer-backend/shaders/ssr_temporal.comp.wgsl` | P4 | SSR temporal reprojection |
| `crates/renderer-backend/src/reflection_probes.rs` | P5 | Reflection probe runtime |
| `crates/renderer-backend/shaders/probe_parallax_correction.wgsl` | P5 | Parallax correction GPU |
| `crates/renderer-backend/shaders/probe_prefilter.comp.wgsl` | P5 | Pre-filtered cubemaps |
| `crates/renderer-backend/shaders/probe_blend.comp.wgsl` | P5 | Probe blending |
| `engine/rendering/lighting/probe_atlas.py` | P5 | Probe atlas packing |
| `crates/renderer-backend/src/planar_reflections.rs` | P6 | Planar mirror rendering |
| `crates/renderer-backend/shaders/voxelize.comp.wgsl` | P7 | Scene voxelisation |
| `crates/renderer-backend/shaders/voxel_downsample.comp.wgsl` | P7 | Voxel mip chain |
| `crates/renderer-backend/shaders/voxel_cone_trace.comp.wgsl` | P7 | Voxel cone tracing |
| `crates/renderer-backend/shaders/rt_reflections.rgen` | P8 | RT reflection ray gen |
| `crates/renderer-backend/shaders/rt_reflections_denoise.comp.wgsl` | P8 | RT denoising |
| `crates/renderer-backend/shaders/reflection_fallback_chain.comp.wgsl` | P8 | Reflection fallback |
| `crates/renderer-backend/shaders/denoise_atrous.comp.wgsl` | P9 | A-trous wavelet denoiser |
| `crates/renderer-backend/shaders/denoise_temporal.comp.wgsl` | P9 | Temporal denoiser |
| `crates/renderer-backend/src/denoise_state.rs` | P9 | Denoiser state management |
| `crates/renderer-backend/src/gi_visualization.rs` | P10 | Debug overlay |

## Build & Test

```bash
# Build the renderer backend
cargo build -p renderer-backend

# Run DDGI unit tests
cargo test -p renderer-backend -- ddgi

# Run all renderer backend tests
cargo test -p renderer-backend

# Validate WGSL shaders (when naga is available)
cargo run -p renderer-backend -- validate-shaders
```

## Key Integration Points

1. **GAPSET_3_BRIDGE**: Shared WGSL infrastructure (buffer bindings, camera uniforms, G-buffer textures) must be wired into DDGI passes. Currently no passes are connected to the runtime frame graph.

2. **S8 (Velocity Buffer)**: SSR temporal, SSGI temporal, DDGI temporal, and RT denoising all require velocity buffer infrastructure from GAPSET_8, which is not yet built.

3. **S10 (TLAS/SBT)**: RT reflections require hardware ray tracing acceleration structures and shader binding tables. S10 infrastructure is absent.

4. **S16 (Asset Pipeline)**: Lightmap baker output, baked cubemap storage, and voxel mesh data depend on the asset pipeline.

5. **Python-to-Runtime Bridge**: 1,623 lines of Python GI reference code (gi_ddgi.py + gi_probes.py) need to be translated to WGSL/Rust and wired into the production pipeline.

## Completion Criteria

- Phase 1-2: Foundation and DDGI core (partial WGSL exists, needs production hardening)
- Phase 3-10: No implementation exists (SSGI, SSR, probes full pipeline, planar, voxel, RT, denoising, visualization)
- Phase 11: No research documents exist
- **Current: 0/44 tasks fully built, 8/44 partially built, 36/44 not built**
