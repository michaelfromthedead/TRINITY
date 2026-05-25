# GAPSET 6: Global Illumination & Reflections -- Clarification Document

## Plan vs. Reality Discrepancies

This document catalogs the differences between the plan described in PHASE_N_TODO.md and the actual source code found on disk.

---

## Critical Discrepancies

### 1. WGSL SH Order: L0+L1 vs. Claimed L2

- **Plan claims**: 3rd-order spherical harmonics (9 coefficients per channel, 27 total).
- **Reality**: WGSL `ddgi.wgsl` implements only L0 and L1 (4 coefficients per channel, 12 total). Constants `SH_L0` and `SH_L1` are correct per Ramamoorthi & Hanrahan. `eval_sh()`, `eval_sh_rgb()`, and `project_sh()` only handle 4-element vec4.
- **Python reference**: `SphericalHarmonics` in `gi_probes.py` has full L2 (9 coefficients per channel, 27 total) with `evaluate()` and `add_sample()`.
- **Impact**: L0+L1 is sufficient for low-frequency diffuse irradiance but cannot represent the detail the plan specifies. L0+L1 corresponds to ~5% RMS error for typical irradiance maps (Ramamoorthi & Hanrahan 2001). Full L2 reduces this to ~1%.
- **Action**: Either accept L0+L1 as sufficient for DDGI (common industry practice) or port Python's L2 SH to WGSL.

### 2. DDGI Probe Volume Location: ddgi.rs vs. gi/probe_grid.rs

- **Plan states**: Probe storage is at `gi/probe_grid.rs`.
- **Reality**: `DDGIProbeVolume` struct is at `crates/renderer-backend/src/ddgi.rs`. No `gi/` directory exists under the renderer-backend source.
- **Impact**: Minimal -- the struct exists and is functional. Documentation needs updating.

### 3. DDGI Pass Builders Are Disconnected

- **Plan assumes**: Passes are wired into frame graph execution.
- **Reality**: `ddgi.rs` creates `IrPass` instances for `ddgi_update` and `ddgi_sample` but these are never registered with a frame graph builder or scheduled for execution. The dispatch counts assume simple N-probe dispatch and 1x1x1 for sampling, which would not work at production resolutions.
- **Impact**: DDGI exists in isolation -- no runtime integration, no trigger mechanism.

### 4. DDGI Update Shader Uses Placeholder Scene Data

- **Plan assumes**: DDGI update traces against actual scene depth/geometry.
- **Reality**: `ddgi_update_probes` generates procedural sky/ground colors (`sky_color = vec3(0.05, 0.05, 0.1)`, `ground_color = vec3(0.02, 0.03, 0.01)`). No depth buffer tracing, no geometry intersection.
- **Impact**: The update shader is a structural skeleton, not a production implementation.

### 5. All Reflection Techniques Are Absent

- **Plan describes**: SSR (HiZ ray march + temporal + blur), RT reflections (importance sampling + denoising + fallback chain), planar reflections (mirror + oblique clip), reflection probes (capture + blend + prefilter + atlas).
- **Reality**: Zero reflection shaders exist. No HiZ buffer generation. No RT shaders. No planar mirror code. `ReflectionProbe` has Python-level parallax correction only.
- **Impact**: All of Phases 4, 6, and 8 are entirely unbuilt. Phase 5 has one partial implementation.

### 6. reflection_probe Decorator: 3 of 11 Parameters Implemented

- **Plan specifies**: 11 decorator parameters (capture_mode, resolution, update_rate, importance, box_extents, inner_radius, outer_radius, roughness_levels, capture_lod_bias, include_layers, exclude_actors).
- **Reality**: Only 3 exist (capture_mode, resolution, update_rate). `GIImportance` enum exists in `light_types.py` but is not wired to the probe system.
- **Impact**: 8 parameters need to be added. Small effort per parameter but requires understanding the capture pipeline.

### 7. Python Reference Is Disconnected from Runtime

- **Plan does not address**: The relationship between Python reference code and WGSL/Rust runtime.
- **Reality**: 1,623 lines of Python code (gi_ddgi.py + gi_probes.py) implement full DDGI probes, probe grids, update passes, lookup, spherical harmonics, irradiance volumes, and reflection probes. None of this is wired to the WGSL/Rust runtime.
- **Action**: Decide whether Python is a design reference to translate or a runtime component to bridge.

### 8. Frame Graph Has RayTracing Infrastructure But No RT Shaders

- **Plan assumes**: RT pipeline has pass infrastructure.
- **Reality**: `PassType::RayTracing` and `ViewType::AccelerationStructure` exist in the frame graph IR. `IrPass::ray_tracing()` constructor exists. But no TLAS management, no SBT construction, no RT entry points.
- **Impact**: Frame graph is ready for RT passes -- the scaffolding is in place but the actual work (S10 TLAS/SBT) remains.

---

## Minor Discrepancies

### 9. DDGI Constant Naming

- **Plan may expect**: `gi_config.py` with all DDGI/SSR/RT constants.
- **Reality**: Constants are split across `constants.py` (GIProbeConstants, DDGIConstants) and `light_types.py` (GIImportance). No `gi_config.py` exists.

### 10. Probe Grid Spacing Configuration

- **Plan specifies**: Configurable spacing tiers (4-8m) with fixed GPU allocations.
- **Reality**: `DDGIProbeVolume` has single `probe_spacing` field. No tier system exists.

### 11. Irradiance Volume Manager

- **Plan may expect**: Multi-volume manager with cross-fade.
- **Reality**: `IrradianceVolume` wraps a single `ProbeGrid` with blend distance and falloff mode. No `IrradianceVolumeManager`.

### 12. Baked Lightmap Infrastructure

- **Plan specifies**: Editor tool, .ktx2 output, offline trace pipeline.
- **Reality**: `LightProbe.bake()` does CPU-side SH sampling. `BakedLightmap` exists with bilinear sampling. No editor tool, no .ktx2 output, no offline trace pipeline.

---

## Summary of Action Items

| # | Issue | Severity | Effort | Fix |
|---|-------|----------|--------|-----|
| 1 | WGSL SH only L0+L1 | Medium | Medium | Port Python L2 SH to WGSL or accept L0+L1 |
| 2 | Wrong file path in plan | Low | Trivial | Update documentation |
| 3 | DDGI passes disconnected | High | Medium | Wire passes into frame graph execution |
| 4 | DDGI update uses placeholders | High | Medium | Implement scene tracing in update shader |
| 5 | No reflection techniques exist | High | Large | Build all SSR/RT/planar/probe pipeline |
| 6 | Decorator has 3/11 params | Medium | Small | Add 8 missing parameters |
| 7 | Python reference disconnected | Medium | Large | Bridge or translate Python to Rust/WGSL |
| 8 | RT infra without RT shaders | High | Large | Build S10 TLAS/SBT, write RT shaders |
| 9 | No gi_config.py | Low | Small | Consolidate constants |
| 10 | No probe spacing tiers | Low | Medium | Add tier configuration |
| 11 | No irradiance volume manager | Low | Medium | Add multi-volume support |
| 12 | No baked lightmap pipeline | Medium | Large | Build editor + .ktx2 export |
