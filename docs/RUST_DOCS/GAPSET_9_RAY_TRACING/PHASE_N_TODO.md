# PHASE_N_TODO.md — GAPSET_9_RAY_TRACING

> **Cluster**: GAPSET_9_RAY_TRACING (S10 — Ray Tracing)
> **Discovered Phases**: 3 (Phase 1 = current wgpu, Phase 2 = RT pipeline stable, Phase 3 = future)
> **TASK_ID Format**: T-RT-{PHASE}.{N}
> **Total Tasks**: 35 (Phase 1: 15, Phase 2: 14, Phase 3: 6)
> **Research Tasks**: 2 (T-RT-P3.4, T-RT-P3.5)
> **Platform-Gated Tasks**: 4 (T-RT-P2.1, T-RT-P2.2, T-RT-P2.3, T-RT-P2.6 — gated on wgpu RT pipeline stability)
> **Effort Scale**: 1 = < 1 day, 2 = 1-3 days, 3 = 3-7 days, 5 = 1-2 weeks, 8 = 2-4 weeks, 13 = 1-2 months

---

## Phase 1: Inline Ray Queries for Shadow Rays

**Status**: Ready to implement (wgpu ray_query + acceleration_structure features are available)
**Minimum wgpu version**: Current
**Gaps covered**: S10-G2, S10-G3, S10-G6, S10-G8
**Platform gate**: None

### T-RT-P1.1 — Define Python ABI stubs for BLAS/TLAS management

- **Description**: Implement the Python-side interface for BLAS and TLAS lifecycle: `BLASDesc`, `TLASDesc`, `BLASManager`, `TLASManager`, `BLASPool` classes with Trinity Pattern decorators.
- **Acceptance Criteria**:
  - `BLASDesc` dataclass with vertex_buffer, vertex_count, vertex_stride, index_buffer, index_count, build_flags
  - `TLASDesc` dataclass with instance_count, instances, build_flags
  - `BLASManager.build_static()` and `BLASManager.build_dynamic()` stubs
  - `BLASManager.refit()` and `BLASManager.compact()` stubs
  - `TLASManager.build_frame()` stub
  - `BLASPool` with mesh-asset-ID-keyed reference counting
  - All classes decorated with appropriate Trinity decorators
  - All stubs pass Python type checking
- **Dependencies**: None
- **Effort**: 5

### T-RT-P1.2 — Implement Rust BLAS build/refit/compact dispatch

- **Description**: Implement the Rust/wgpu backend for BLAS construction: upload vertex buffer, create BLAS with `wgpu::RayTracingAccelerationStructure`, compaction query, compacted copy, and refit with `ALLOW_UPDATE`.
- **Acceptance Criteria**:
  - `create_blas()` takes vertex buffer + build flags, returns BLAS handle
  - `refit_blas()` updates BLAS with new vertex positions (ALLOW_UPDATE only)
  - `compact_blas()` queries compacted size, allocates, copies, discards original
  - Builder pattern for BLASDesc translation from Python to wgpu
  - Scratch buffer allocation and pooling
  - Unit tests: BLAS creation, compact size < original size, refit preserves trace results
- **Dependencies**: T-RT-P1.1
- **Effort**: 8

### T-RT-P1.3 — Implement Rust TLAS build dispatch

- **Description**: Implement the Rust/wgpu backend for TLAS construction: instance buffer creation, TLAS build with `PREFER_FAST_BUILD`, per-frame rebuild.
- **Acceptance Criteria**:
  - `create_tlas()` takes instance buffer + build flags, returns TLAS handle
  - Instance buffer layout matches `RayTracingInstance` struct
  - Per-frame rebuild with updated instance transforms
  - Scratch buffer sharing with BLAS build pipeline
  - Unit tests: TLAS creation, instance transform change reflected, TLAS with 1000 instances
- **Dependencies**: T-RT-P1.2
- **Effort**: 5

### T-RT-P1.4 — Implement RTCapability Detection

- **Description**: Implement device feature query for ray tracing capabilities. Route all RT effects through a capability gate that selects the appropriate rendering path.
- **Acceptance Criteria**:
  - `RTCapability` enum: NONE, RAY_QUERY_ONLY, FULL
  - `get_rt_capability(device)` queries wgpu `device.has_feature()` for acceleration_structure, ray_query, ray_tracing_pipeline
  - Capability gate routes each effect to RT or fallback
  - Graceful transition when capability changes between sessions
  - Unit tests: each capability level returns expected effect configuration
- **Dependencies**: T-RT-P1.1
- **Effort**: 3

### T-RT-P1.5 — Write RT shadow ray query compute shader

- **Description**: Write the WGSL compute shader for inline ray query shadow rays. Takes G-Buffer depth+normal, reconstructs world position, traces shadow rays toward each light.
- **Acceptance Criteria**:
  - Compute shader using `RayQueryInitialize`, `RayQueryProceed`, `RayQueryGetIntersectionType`
  - Read G-Buffer depth + normal
  - Reconstruct world-space position
  - Trace shadow ray for each light in visible set
  - `RAY_FLAG_ACCEPT_FIRST_AND_END` for binary occlusion
  - Output: shadow factor texture (one float per light per pixel)
  - Handle directional, point, spot, and area light ray directions
- **Dependencies**: T-RT-P1.3 (for TLAS)
- **Effort**: 8

### T-RT-P1.6 — Write any-hit shader for alpha-tested geometry

- **Description**: Write the WGSL any-hit shader for shadow rays that handles alpha-tested geometry (foliage, fences, wire mesh).
- **Acceptance Criteria**:
  - Inline any-hit logic within ray query (ray queries use inline, not separate shaders)
  - Load alpha texture at hit UV coordinates
  - Compare against alpha cutoff threshold
  - Ignore hit if alpha < threshold (continue traversal)
  - Accept hit if alpha >= threshold (shadow)
  - Skip any-hit for instances marked OPAQUE
- **Dependencies**: T-RT-P1.5
- **Effort**: 5

### T-RT-P1.7 — Implement Python shadow ray dispatch

- **Description**: Implement the Python-side dispatch logic for RT shadows: G-Buffer binding, TLAS binding, ray query compute shader dispatch, output shadow factor texture.
- **Acceptance Criteria**:
  - `RTShadows` class with Trinity decorators
  - `dispatch_raytraced_shadows(g_buffer, tlas, lights, output_texture)` method
  - Per-light shadow ray generation
  - Quality tier parameterization (1/2/4/8 rays per pixel)
  - Read G-Buffer, write shadow factor to output texture
  - Integration with S1 `RayTracingPass` pass node
- **Dependencies**: T-RT-P1.5, T-RT-P1.6
- **Effort**: 5

### T-RT-P1.8 — Implement fallback chain: RT shadows -> CSM + PCSS + Contact Shadows

- **Description**: Implement the full fallback chain for shadow rendering. When RT is unavailable, fall through to cascaded shadow maps + PCSS filtering + contact shadows in the correct priority order.
- **Acceptance Criteria**:
  - Fallback decision at capability gate
  - CSM with PSSM (parallel split shadow maps)
  - PCSS filtering for soft shadows
  - Contact shadows for small-scale detail
  - Smooth transition between RT shadows and fallback
  - Unit tests: fallback activation at each capability level
- **Dependencies**: T-RT-P1.4
- **Effort**: 8

### T-RT-P1.9 — Write A Trous spatial denoiser compute shader

- **Description**: Write the WGSL compute shader for A Trous wavelet spatial denoising. 3-4 iterations with edge-aware filtering using G-Buffer features.
- **Acceptance Criteria**:
  - 3-4 iterative passes with increasing step sizes (1, 2, 4, 8)
  - Edge-stopping weights based on depth, normal, luminance
  - Separable passes: horizontal + vertical
  - Configurable sigma (filter strength) parameter
  - Output: denoised shadow factor texture
  - Performance: < 0.5ms for 3 iterations at 1080p
- **Dependencies**: T-RT-P1.7
- **Effort**: 8

### T-RT-P1.10 — Implement Python denoiser dispatch

- **Description**: Implement Python-side denoiser management. Spatial denoiser dispatch with quality tier parameterization.
- **Acceptance Criteria**:
  - `Denoiser` class with Trinity decorators
  - `spatial_denoise(noisy_input, g_buffer, output)` method
  - Quality tier: iterations (2 Low, 3 Medium, 4 High/Ultra)
  - Integration with shadow, reflection, and GI outputs
  - Support for single-effect and batched denoising
- **Dependencies**: T-RT-P1.9
- **Effort**: 3

### T-RT-P1.11 — Implement BLAS pool with reference counting

- **Description**: Implement the BLAS pool that tracks BLAS resources by mesh asset ID with reference counting.
- **Acceptance Criteria**:
  - `BLASPool` class with dictionary of mesh asset ID to BLAS entry
  - Reference counting for shared meshes (multiple instances sharing one BLAS)
  - `acquire(mesh_id) -> BLASHandle`
  - `release(blas_handle)` (decrements ref count)
  - Automatic cleanup when ref count reaches zero
  - Pending build/compact queues for batched submission
- **Dependencies**: T-RT-P1.2
- **Effort**: 5

### T-RT-P1.12 — Implement instance buffer management

- **Description**: Implement CPU-to-GPU instance buffer upload for TLAS construction each frame.
- **Acceptance Criteria**:
  - Collect instance transforms from scene graph
  - Build `RayTracingInstance` array (transform, blas_address, instance_id, mask, hit_group_offset, flags)
  - Upload to GPU instance buffer
  - Two-buffer scheme (ping-pong) to avoid GPU read stalls
  - Support for instance mask filtering
  - Performance: < 0.1ms CPU overhead for 1000 instances
- **Dependencies**: T-RT-P1.3, T-RT-P1.11
- **Effort**: 5

### T-RT-P1.13 — Implement ray budget management

- **Description**: Implement the per-frame ray budget system that prevents frame-time spikes by limiting total rays per frame.
- **Acceptance Criteria**:
  - `RayBudget` class with configurable `max_rays_per_frame`
  - `allocate(effect, rays) -> bool` allocation method
  - Automatic degradation when budget exceeded (reduce samples, half-res)
  - Reset at start of each frame
  - Integration with quality tier system
  - Unit tests: budget limits enforced, degradation cascades correctly
- **Dependencies**: None
- **Effort**: 3

### T-RT-P1.14 — Implement static mesh BLAS build on load

- **Description**: Integrate BLAS construction into the mesh asset pipeline. Build BLAS when a static mesh is loaded.
- **Acceptance Criteria**:
  - Mesh load pipeline triggers BLAS build for static geometry
  - Compaction runs after initial build
  - Compacted BLAS stored in mesh resource
  - BLAS handle available for TLAS instance creation
  - No BLAS rebuild on subsequent frames
  - Memory budget tracked (static BLAS total)
- **Dependencies**: T-RT-P1.11, S16 (Asset Pipeline)
- **Effort**: 5

### T-RT-P1.15 — Implement dynamic mesh BLAS refit per frame

- **Description**: Integrate dynamic BLAS refit into the frame loop. Refit or rebuild BLAS for dynamic/skinned meshes each frame.
- **Acceptance Criteria**:
  - Rigid dynamic objects: refit BLAS each frame (ALLOW_UPDATE)
  - Skinned meshes: rebuild BLAS each frame (FAST_BUILD)
  - BLAS update queue submitted before TLAS build
  - Fallback: if refit quality degrades (detected by bounds growth), trigger full rebuild
  - Performance: refit < 0.1ms per mesh, rebuild < 1ms per mesh
  - Unit tests: refit preserves trace results after vertex movement
- **Dependencies**: T-RT-P1.2, T-RT-P1.14
- **Effort**: 5

---

## Phase 2: Full RT Pipeline for Reflections and GI

**Status**: Gated on wgpu `ray_tracing_pipeline` stability (estimated 6-12 months)
**Minimum wgpu version**: TBD (requires stable ray_tracing_pipeline + shader_binding_table)
**Gaps covered**: S10-G1, S10-G4, S10-G5, S10-G7, S10-G9
**Platform-gated tasks**: T-RT-P2.1, T-RT-P2.2, T-RT-P2.3, T-RT-P2.6

### T-RT-P2.1 [PLATFORM-GATED] — Implement Rust RT pipeline creation

- **Description**: Implement the Rust/wgpu backend for `RayTracingPipeline` creation. Pipeline layout with ray generation, hit groups, and miss shaders.
- **Gate**: wgpu `ray_tracing_pipeline` feature must be stable (non-experimental)
- **Acceptance Criteria**:
  - `create_rt_pipeline()` takes shader modules (raygen, hit, miss), returns pipeline handle
  - Pipeline layout with bind groups for TLAS, output textures, material data
  - Maximum recursion depth configurable (1-4)
  - Separate pipelines for shadows, reflections, GI
  - Pipeline cache for faster creation on subsequent loads
  - Dynamic state: dispatch dimensions configurable per frame
- **Dependencies**: T-RT-P1.3 (TLAS), S12 (RHI)
- **Effort**: 8

### T-RT-P2.2 [PLATFORM-GATED] — Implement SBTBuilder

- **Description**: Implement the Shader Binding Table builder that maps material domains to hit group indices. Resolves S10-G9.
- **Gate**: wgpu `shader_binding_table` feature must be stable
- **Acceptance Criteria**:
  - `SBTBuilder` class with material domain registration
  - `build_sbt(material_layout) -> ShaderBindingTable` method
  - Hit groups: surface/opaque, surface/masked, surface/translucent, volume
  - SBT record generation (ray gen, hit groups, miss groups)
  - SBT stride configuration
  - SBT rebuilt only on material layout change (not per frame)
  - Unit tests: correct shader group assignment, SBT offsets match instance data
- **Dependencies**: T-RT-P2.1
- **Effort**: 8

### T-RT-P2.3 [PLATFORM-GATED] — Write RT reflection pipeline shaders

- **Description**: Write the WGSL shaders for the ray tracing reflection pipeline: ray generation with BRDF importance sampling, closest-hit PBR shading, miss environment sampling.
- **Gate**: wgpu `ray_tracing_pipeline` feature must be stable (uses raygen/hit/miss shader model)
- **Acceptance Criteria**:
  - `rt_reflection.rgen.wgsl`: Read G-Buffer, BRDF importance sample, trace reflection ray
  - `rt_reflection.rchit.wgsl`: Evaluate PBR at hit point (albedo, normal, roughness from bindless material table)
  - `rt_reflection.rmiss.wgsl`: Sample sky/environment map
  - Roughness-dependent sample count (1-4 rays per pixel)
  - GGX importance sampling for rough reflections
  - Maximum recursion depth 1 (no secondary reflections)
  - Half-resolution dispatch option
- **Dependencies**: T-RT-P2.1, T-RT-P2.2
- **Effort**: 13

### T-RT-P2.4 [PLATFORM-GATED] — Write RT GI pipeline shaders

- **Description**: Write the WGSL shaders for ray-traced global illumination: single indirect bounce with hemisphere sampling, closest-hit radiance evaluation.
- **Gate**: wgpu `ray_tracing_pipeline` feature must be stable
- **Acceptance Criteria**:
  - `rt_gi_indirect.rgen.wgsl`: Single indirect bounce from G-Buffer pixels
  - `rt_gi.rchit.wgsl`: Evaluate incident radiance at hit point (light samples + emissive)
  - `rt_gi.rmiss.wgsl`: Environment lighting
  - Cosine-weighted hemisphere sampling
  - Half-resolution tracing option
  - Output: indirect radiance buffer
  - Performance: < 4ms indirect only at 1080p half-res
- **Dependencies**: T-RT-P2.1, T-RT-P2.2
- **Effort**: 13

### T-RT-P2.5 — Implement Python RT reflection dispatch

- **Description**: Implement the Python-side dispatch logic for RT reflections: G-Buffer binding, TLAS binding, pipeline dispatch, output reflection buffer.
- **Acceptance Criteria**:
  - `RTReflections` class with Trinity decorators
  - `dispatch_raytraced_reflections(g_buffer, tlas, sbt, output)` method
  - Quality tier parameterization (Off/Low/Mid/High/Ultra)
  - Half-resolution dispatch for Medium/High, full resolution for Ultra
  - BRDF sampling based on roughness
  - Integration with S1 `RayTracingPass`
- **Dependencies**: T-RT-P2.3
- **Effort**: 5

### T-RT-P2.6 [PLATFORM-GATED] — Implement Python RT GI dispatch

- **Description**: Implement the Python-side dispatch logic for RT GI: G-Buffer binding, hemisphere sampling, pipeline dispatch, output indirect radiance buffer.
- **Gate**: wgpu `ray_tracing_pipeline` must be stable
- **Acceptance Criteria**:
  - `RTGI` class with Trinity decorators
  - `dispatch_raytraced_gi(g_buffer, tlas, sbt, light_list, output)` method
  - Quality tier parameterization (Off/High/Ultra)
  - Indirect light only mode (primary target)
  - Full path tracing mode (ground truth, Ultra only)
  - Temporal accumulation across frames
  - Integration with S1 `RayTracingPass`
- **Dependencies**: T-RT-P2.4
- **Effort**: 5

### T-RT-P2.7 — Write temporal denoising compute shader

- **Description**: Write the WGSL compute shader for temporal denoising: reprojection, accumulation, clamping, and disocclusion detection.
- **Acceptance Criteria**:
  - Reproject current pixel to previous frame using motion vectors
  - Accumulate: `accumulated = lerp(history, current, alpha)`
  - Clamp history to current frame's neighborhood variance (3x3)
  - Disocclusion detection: depth difference threshold + normal cone threshold
  - Reset accumulator on disocclusion (alpha = 1.0)
  - Motion-adaptive alpha (clamp to 0.05-0.2 on movement)
  - Output: temporally accumulated, denoised result
  - Performance: < 0.5ms at 1080p
- **Dependencies**: T-RT-P1.9 (spatial denoiser)
- **Effort**: 8

### T-RT-P2.8 — Write joint bilateral filter compute shader

- **Description**: Write the WGSL compute shader for the final joint bilateral filter in the denoising pipeline.
- **Acceptance Criteria**:
  - Spatial Gaussian weighting based on pixel distance
  - Range Gaussian weighting based on depth, normal, luminance differences
  - Configurable sigma_spatial and sigma_range parameters
  - 5x5 or 7x7 kernel with adaptive radius
  - Output: final denoised result
  - Performance: < 0.3ms at 1080p
- **Dependencies**: T-RT-P2.7
- **Effort**: 5

### T-RT-P2.9 — Implement full three-stage denoiser pipeline

- **Description**: Chain the spatial, temporal, and bilateral filters into a complete denoising pipeline with history buffer management.
- **Acceptance Criteria**:
  - `DenoiserPipeline` class chaining spatial -> temporal -> bilateral
  - History texture management (allocate, update, clear on cut)
  - Accumulation count tracking (per-pixel or global)
  - Disocclusion mask texture (1 byte per pixel)
  - Reprojection matrix storage (previous frame view-projection)
  - Quality tier: Low = spatial only, Medium/High = spatial+temporal, Ultra = full three-stage
  - Performance: < 2ms full pipeline at 1080p
- **Dependencies**: T-RT-P1.10, T-RT-P2.7, T-RT-P2.8
- **Effort**: 8

### T-RT-P2.10 — Implement bindless material table for hit shaders

- **Description**: Implement the GPU-side bindless material table that provides material data to RT hit shaders.
- **Acceptance Criteria**:
  - Bindless storage buffer of `MaterialData` structs
  - Material data uploaded on material load (not per frame)
  - Instance buffer references material index via `instance_custom_index`
  - Material data includes: base_color, metallic, roughness, emissive, texture indices, alpha_cutoff
  - Texture arrays for bindless texture sampling in hit shaders
  - Fallback: if bindless not supported, use per-pipeline material constants
- **Dependencies**: T-RT-P2.1
- **Effort**: 5

### T-RT-P2.11 — Implement reflection fallback chain (RT -> SSR -> Probes)

- **Description**: Implement the cascading fallback chain for reflections: RT reflections -> SSR -> reflection probes -> none.
- **Acceptance Criteria**:
  - Capability-gated fallback decision
  - SSR with HiZ ray marching
  - Reflection probe blending (existing S7 spec)
  - Roughness-threshold hybrid mode (RT for smooth, SSR for rough)
  - Smooth visual transition between modes
- **Dependencies**: T-RT-P2.5, S7 (Reflections)
- **Effort**: 8

### T-RT-P2.12 — Implement GI fallback chain (RT GI -> DDGI -> SSGI)

- **Description**: Implement the cascading fallback chain for global illumination: RT GI -> DDGI (dynamic diffuse GI) -> SSGI (screen-space GI) -> none.
- **Acceptance Criteria**:
  - Capability-gated fallback decision
  - DDGI probe update and blending (existing S6 spec)
  - SSGI ray marching compute shader
  - Distance-based hybrid mode (RT GI near camera, DDGI far)
  - Smooth visual transition between modes
- **Dependencies**: T-RT-P2.6, S6 (Global Illumination)
- **Effort**: 8

### T-RT-P2.13 — Implement adaptive quality system

- **Description**: Implement frame-time feedback loop that dynamically adjusts ray budget, resolution, and feature enablement.
- **Acceptance Criteria**:
  - Frame time measurement after each RT pass
  - Comparison against target frame time per quality tier
  - Degradation steps: reduce rays -> half-res -> disable GI -> disable reflections -> reduce denoiser iterations
  - Recovery steps: when frame time is below target, restore features in reverse order
  - Hysteresis to prevent oscillation
  - Per-frame logging of quality adjustments
  - Unit tests: feedback loop converges to target frame time
- **Dependencies**: T-RT-P1.13
- **Effort**: 8

### T-RT-P2.14 — Implement S1 frame graph integration for RT passes

- **Description**: Integrate all RT passes into the existing `RayTracingPass` node in the S1 frame graph. Handle resource tracking, barriers, and pass ordering.
- **Acceptance Criteria**:
  - RT shadow pass reads G-Buffer + TLAS, writes shadow mask
  - RT reflection pass reads G-Buffer + TLAS, writes reflection buffer
  - RT GI pass reads G-Buffer + TLAS + light list, writes indirect radiance
  - Denoiser passes read noisy RT output + G-Buffer, write denoised output
  - Resource state tracking for AS (build -> read)
  - Pass ordering: AS build -> RT passes -> denoise -> composite
  - Barriers between passes correct (wgpu synchronization)
- **Dependencies**: T-RT-P1.7, T-RT-P2.5, T-RT-P2.6, T-RT-P2.9
- **Effort**: 8

---

## Phase 3: Full Path Tracing and Neural Denoising

**Status**: Future work. Requires stable RT pipeline + neural denoising extensions
**Minimum wgpu version**: Stable RT pipeline + NPU/tensor core extensions (WINN or equivalent)
**Gaps covered**: S10-G7 (neural denoising), S10-G1 (full API stability)

### T-RT-P3.1 — Write full path tracing compute shader

- **Description**: Write the WGSL path tracing shader with multi-bounce sampling, Russian roulette termination, and temporal accumulation.
- **Acceptance Criteria**:
  - Multi-bounce path tracing (up to 4 bounces)
  - Russian roulette termination (probability = max(albedo * (1 - metallic), 0.2))
  - Direct + indirect illumination accumulation
  - 1 sample/pixel/frame with temporal accumulation
  - Reset on camera cut or large scene change
  - Clamp accumulator at 256 frames maximum
  - Bias toward recent frames on movement
- **Dependencies**: T-RT-P2.4 (GI shaders)
- **Effort**: 13

### T-RT-P3.2 — Implement path tracing temporal accumulation

- **Description**: Implement the temporal accumulation system for path tracing: per-pixel accumulation count, frame-to-frame blending, reset detection.
- **Acceptance Criteria**:
  - Per-pixel accumulation count buffer
  - `accumulated = lerp(previous_accumulated, current_sample, 1.0 / frame_count)`
  - Camera cut detection (threshold-based)
  - Large scene change detection (instance count change, transform threshold)
  - Motion-based bias adjustment
  - Unit tests: accumulation convergence, reset on cut, bias on movement
- **Dependencies**: T-RT-P3.1
- **Effort**: 5

### T-RT-P3.3 — Write neural denoising compute shader (U-Net)

- **Description**: Write the WGSL compute shader for a lightweight U-Net neural denoiser. 2-3 encoder/decoder stages with skip connections.
- **Acceptance Criteria**:
  - U-Net with 2 encoder stages and 2 decoder stages
  - Input: 7 channels (noisy RGB + depth, normal, roughness, albedo)
  - Output: 3 channels (denoised RGB)
  - Convolution kernel weights loaded from storage buffer (ONNX model)
  - ReLU activations between stages
  - Skip connections from encoder to decoder
  - Fallback to spatial+temporal when neural path unavailable
  - Performance: < 3ms at 1080p
- **Dependencies**: T-RT-P2.9 (three-stage denoiser as fallback)
- **Effort**: 13

### T-RT-P3.4 [RESEARCH] — Neural denoising model architecture survey

- **Description**: Survey the literature for real-time neural denoising architectures suitable for compute shader implementation. Evaluate U-Net vs. KPCN vs. NFN vs. other architectures.
- **Acceptance Criteria**:
  - Survey 5+ candidate architectures
  - Quality comparison on standard test scenes (PSNR, SSIM, inference time)
  - Memory footprint analysis per architecture
  - Training data requirements and generation pipeline
  - ONNX export compatibility analysis
  - Recommendation with rationale for selected architecture
  - Published as research note in `docs/research/denoising_survey.md`
- **Dependencies**: None
- **Effort**: 5

### T-RT-P3.5 [RESEARCH] — wgpu Opaque Opacity Micromap support timeline

- **Description**: Research the wgpu roadmap for Opaque Opacity Micromaps (OMM) and Displacement Mesh (DMM) support in acceleration structures.
- **Acceptance Criteria**:
  - Survey wgpu GitHub issues, PRs, and milestone tracking
  - Identify driver support requirements (NVIDIA SER, AMD, Intel)
  - Timeline estimate for availability
  - Impact analysis on any-hit shader performance
  - Integration points in BLAS build pipeline
  - Published as research note in `docs/research/rt_extensions_timeline.md`
- **Dependencies**: None
- **Effort**: 3

### T-RT-P3.6 — Implement adaptive quality 2.0 with ML prediction

- **Description**: Enhance the adaptive quality system with machine learning prediction of frame time based on scene complexity features.
- **Acceptance Criteria**:
  - Scene complexity feature extraction (instance count, ray count, material count, resolution)
  - Lightweight ML model (linear regression or small neural network) predicting frame time
  - Training data collection from previous frames
  - Proactive quality adjustment (before rendering, not reactive)
  - Fallback to reactive system when prediction confidence is low
- **Dependencies**: T-RT-P2.13
- **Effort**: 8

---

## Task Summary

| Phase | Tasks | Est. Effort (person-weeks) | Dependencies | Gated |
|-------|-------|---------------------------|-------------|-------|
| P1 | 15 | ~8-12 | S1, S12 | None |
| P2 | 14 | ~16-24 | P1, S6, S7 | wgpu RT pipeline stable |
| P3 | 6 | ~8-12 | P2, wgpu stable | wgpu + NPU extensions |
| **Total** | **35** | **~32-48** | | |

---

## Key Cross-Phase Dependencies

```
Phase 1 (now):
  T-RT-P1.4 (capability detect) ─> T-RT-P1.8 (fallback chain)
  T-RT-P1.1 (AS stubs) ─> T-RT-P1.2 (BLAS Rust) ─> T-RT-P1.11 (BLAS pool)
  T-RT-P1.2 ─> T-RT-P1.3 (TLAS Rust) ─> T-RT-P1.12 (instance buffer)
  T-RT-P1.3 ─> T-RT-P1.5 (shadow shader) ─> T-RT-P1.7 (shadow dispatch)
  T-RT-P1.9 (spatial denoise) ─> T-RT-P1.10 (denoiser dispatch)

Phase 2 (gated on wgpu RT pipeline stability):
  T-RT-P2.1 (RT pipeline Rust) ─> T-RT-P2.2 (SBT) ─> T-RT-P2.3 (reflection shaders)
  T-RT-P2.1 ─> T-RT-P2.4 (GI shaders) ─> T-RT-P2.6 (GI dispatch)
  T-RT-P2.3 ─> T-RT-P2.5 (reflection dispatch)
  T-RT-P1.9 ─> T-RT-P2.7 (temporal denoise) ─> T-RT-P2.8 (bilateral) ─> T-RT-P2.9 (full pipeline)
  T-RT-P2.5 + T-RT-P2.6 + T-RT-P2.9 ─> T-RT-P2.14 (frame graph integration)

Phase 3:
  T-RT-P3.4 (research: denoiser arch) ─> T-RT-P3.3 (neural denoising shader)
  T-RT-P2.4 ─> T-RT-P3.1 (path tracing shader)
  T-RT-P2.13 ─> T-RT-P3.6 (adaptive quality 2.0)
```

---

## Test Plan Summary

| Phase | Test Type | Count | Coverage |
|-------|-----------|-------|----------|
| P1 | Unit | ~16 | BLAS/TLAS creation, compaction, refit; RTCapability transitions; ray budget enforcement; denoiser edge detection |
| P1 | Integration | ~4 | Frame graph with RT shadows, RT shadows + G-Buffer, denoiser pipeline, fallback chain |
| P1 | Visual | ~2 | RT shadow quality vs path-traced reference, fallback visual parity |
| P1 | Performance | ~4 | BLAS build time, TLAS build time, RT shadow frame time, denoiser frame time |
| P2 | Unit | ~12 | SBT creation, hit group mapping, temporal reprojection, disocclusion detection, bindless material table |
| P2 | Integration | ~6 | Frame graph with all RT passes, denoiser pipeline, material hit group mapping, fallback chain |
| P2 | Visual | ~4 | RT reflection quality, RT GI quality, denoiser quality, hybrid mode visual parity |
| P2 | Performance | ~6 | RT reflection frame time, RT GI frame time, full denoiser frame time, adaptive quality convergence |
| P3 | Unit | ~6 | Path tracing accumulation, neural denoiser weights, adaptive quality ML prediction |
| P3 | Visual | ~3 | Path tracing quality vs reference, neural denoiser quality vs reference, temporal convergence |
| P3 | Performance | ~3 | Path tracing frame time, neural denoiser inference time, adaptive quality prediction accuracy |
