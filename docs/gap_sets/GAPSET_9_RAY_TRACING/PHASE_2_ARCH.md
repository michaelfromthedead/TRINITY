# PHASE_2_ARCH.md — Full RT Pipeline for Reflections and GI

> **Phase**: 2 of 3
> **Status**: Gated — requires wgpu `ray_tracing_pipeline` stability
> **Tasks**: 14 (all [-])
> **Gaps Covered**: S10-G1, S10-G4, S10-G5, S10-G7, S10-G9
> **Platform Gate**: wgpu `ray_tracing_pipeline` + `shader_binding_table` (experimental)
> **Reality (2026-05-22)**: Zero implementation. All 14 tasks are [-] not started.

---

## 1. Architecture Overview

Phase 2 uses the full ray tracing pipeline model (raygen + hit + miss shaders) for reflections and global illumination, alongside a three-stage denoiser pipeline.

```
                     Ray Tracing Pipeline Model
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  Ray Generation          Hit Groups         Miss        │
  │  Shaders                 (Closest-Hit       Shaders     │
  │                          + Any-Hit)                     │
  │  ┌──────────────┐      ┌──────────────┐   ┌──────────┐ │
  │  │rt_reflection │      │ surface/     │   │ sky/env  │ │
  │  │.rgen         │─────▶│ opaque.rchit │──▶│ .rmiss   │ │
  │  └──────────────┘      └──────────────┘   └──────────┘ │
  │                         ┌──────────────┐                │
  │  ┌──────────────┐      │ surface/     │                │
  │  │rt_gi_indirect│─────▶│ masked.rchit │                │
  │  │.rgen         │      └──────────────┘                │
  │  └──────────────┘      ┌──────────────┐                │
  │                        │ surface/     │                │
  │                        │ translucent  │                │
  │                        │.rchit        │                │
  │                        └──────────────┘                │
  │                        ┌──────────────┐                │
  │                        │ volume.rchit │                │
  │                        └──────────────┘                │
  └─────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │                  Denoiser Pipeline                      │
  │                                                         │
  │  RT Output ──▶ Spatial ──▶ Temporal ──▶ Bilateral ──▶   │
  │               (Phase 1)   (P2.7)       (P2.8)   Final  │
  └─────────────────────────────────────────────────────────┘
```

## 2. Component Details

### 2.1 RT Pipeline Creation [-] (T-RT-P2.1)

**Gate**: wgpu `ray_tracing_pipeline` feature stable.

**Required implementation**:
```rust
fn create_rt_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    raygen: &wgpu::ShaderModule,
    hit_groups: &[wgpu::RayTracingHitGroup],
    miss: &wgpu::ShaderModule,
) -> wgpu::RayTracingPipeline {
    device.create_ray_tracing_pipeline(&wgpu::RayTracingPipelineDescriptor {
        label: Some("rt_reflection"),
        layout: Some(layout),
        raygen: raygen.get_entry_point("main"),
        hit_groups: hit_groups,
        miss: miss.get_entry_point("main"),
        max_recursion_depth: 1,
    })
}
```

Separate pipelines needed for:
- **Shadows** (Phase 1 fallback to ray query, or RT pipeline if preferred).
- **Reflections** (BRDF importance sampling, 1-4 rays/pixel).
- **GI** (Cosine-weighted hemisphere sampling, single indirect bounce).

### 2.2 SBTBuilder [-] (T-RT-P2.2)

**Gate**: wgpu `shader_binding_table` feature stable.

The Shader Binding Table maps material domains to hit group indices:

| Material Domain | Hit Group | SBT Offset |
|----------------|-----------|------------|
| Surface/Opaque | 0 | 0 |
| Surface/Masked | 1 | 1 |
| Surface/Translucent | 2 | 2 |
| Volume | 3 | 3 |

Instance buffer stores `instance_custom_index` → material domain → SBT offset.

### 2.3 RT Reflection Shaders [-] (T-RT-P2.3)

**Required WGSL shader modules**:
- `rt_reflection.rgen.wgsl`: BRDF importance sampling, trace reflection ray.
- `rt_reflection.rchit.wgsl`: PBR evaluation (albedo, normal, roughness from bindless table).
- `rt_reflection.rmiss.wgsl`: Environment map sampling.

### 2.4 RT GI Shaders [-] (T-RT-P2.4)

**Required WGSL shader modules**:
- `rt_gi_indirect.rgen.wgsl`: Cosine-weighted hemisphere sampling.
- `rt_gi.rchit.wgsl`: Incident radiance evaluation.
- `rt_gi.rmiss.wgsl`: Environment lighting.

### 2.5 Denoiser Pipeline [-] (P2.7, P2.8, P2.9)

Three-stage pipeline:

```
Stage 1: Spatial (Phase 1, P1.9)
  A Trous wavelet, 3-4 iterations, edge-stopping weights

Stage 2: Temporal (P2.7)
  Reprojection + accumulation + clamping + disocclusion detection
  History buffer management

Stage 3: Joint Bilateral (P2.8)
  5x5/7x7 kernel, spatial + range Gaussian weighting
  Configurable sigma parameters
```

Quality tier mapping:
- **Low**: Spatial only
- **Medium/High**: Spatial + Temporal
- **Ultra**: Full three-stage

### 2.6 Bindless Material Table [-] (T-RT-P2.10)

**Current state**: `BindlessManager` exists at `engine/rendering/gpu_driven/bindless.py` for general bindless resources.

**Required RT extension**: Storage buffer of `MaterialData` structs:
```rust
struct MaterialData {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emissive: vec4<f32>,
    albedo_texture_index: i32,
    normal_texture_index: i32,
    roughness_texture_index: i32,
    alpha_cutoff: f32,
    pad: f32,
};
```

Instance `instance_custom_index` references material index in this table.

### 2.7 Fallback Chains [-] (P2.11, P2.12)

```
RT Reflections  ──> SSR (HiZ ray march) ──> Reflection Probes ──> None
    [P2.5]           [S7 spec]                 [S7 spec]

RT GI ──> DDGI (Dynamic Diffuse GI) ──> SSGI ──> None
 [P2.6]    [S6 spec]                       [S6 spec]
```

### 2.8 Adaptive Quality [-] (T-RT-P2.13)

Frame-time feedback loop:
```
Measure RT pass time
  → Compare to target
    → Above target: reduce rays -> half-res -> disable GI -> disable reflections
    → Below target: restore in reverse order
  → Hysteresis to prevent oscillation
```

### 2.9 Frame Graph Integration [~] (T-RT-P2.14)

**Current state**: `RayTracingPass` node and `FrameGraph.add_raytracing_pass()` exist. Rust IR supports PassType::RayTracing.

**Required**: Wire actual RT effects into S1 pipeline:
```
AS Build ──▶ RT Shadows ──▶ RT Reflections ──▶ RT GI ──▶ Denoise ──▶ Composite
```

## 3. File Map

| Task | New Files Required |
|------|-------------------|
| P2.1 | `crates/renderer-backend/src/rt/pipeline.rs` |
| P2.2 | `crates/renderer-backend/src/rt/sbt.rs` |
| P2.3 | `shaders/rt_reflection.rgen.wgsl`, `.rchit.wgsl`, `.rmiss.wgsl` |
| P2.4 | `shaders/rt_gi_indirect.rgen.wgsl`, `rt_gi.rchit.wgsl`, `rt_gi.rmiss.wgsl` |
| P2.5 | `engine/rendering/rt/reflections.py` |
| P2.6 | `engine/rendering/rt/gi.py` |
| P2.7 | `shaders/denoiser_temporal.comp.wgsl` |
| P2.8 | `shaders/denoiser_bilateral.comp.wgsl` |
| P2.9 | `engine/rendering/rt/denoiser_pipeline.py` |
| P2.10 | `engine/rendering/gpu_driven/bindless_rt.py` |
| P2.11 | `engine/rendering/lighting/reflection_fallback.py` |
| P2.12 | `engine/rendering/lighting/gi_fallback.py` |
| P2.13 | `engine/rendering/rt/adaptive_quality.py` |
| P2.14 | `engine/rendering/framegraph/rt_passes.py` |

## 4. Data Flow (Frame)

```
Frame Start
  │
  ├─ AS Build Phase
  │   ├─ BLAS refit/rebuild (P1.2)
  │   └─ TLAS build (P1.3)
  │
  ├─ G-Buffer Pass
  │
  ├─ RT Shadows (dispatch via pipeline or ray query)
  │   └─ Output: shadow_mask
  │
  ├─ RT Reflections (raygen -> hit -> miss)
  │   └─ Output: reflection_buffer
  │
  ├─ RT GI (raygen -> hit -> miss)
  │   └─ Output: indirect_radiance
  │
  ├─ Denoiser Pipeline
  │   ├─ Spatial (P1.9)
  │   ├─ Temporal reprojection (P2.7)
  │   └─ Joint bilateral (P2.8)
  │
  ├─ Adaptive Quality Feedback (P2.13)
  │   └─ Adjust next frame's ray budget
  │
  └─ Composite / Lighting
```

## 5. Gating Assessment (2026-05-22)

| Feature | wgpu Status | Impact |
|---------|-------------|--------|
| `ray_tracing_pipeline` | Experimental | Blocks P2.1, P2.2, P2.3, P2.6 |
| `shader_binding_table` | Experimental | Blocks P2.2 |
| Ray generation shaders | Experimental | Blocks P2.3, P2.4 |
| Hit group shaders | Experimental | Blocks P2.3, P2.4 |
| Miss shaders | Experimental | Blocks P2.3, P2.4 |

The 6-12 month gating estimate in the TODO appears reasonable based on wgpu's current development trajectory.
