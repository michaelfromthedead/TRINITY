# Phase 2: DDGI Core -- Uniform Grid -- Architecture

## Overview

Phase 2 implements the core Dynamic Diffuse Global Illumination pipeline: probe placement, ray tracing (hardware and rasterized fallback), probe update, probe sampling at shading points, infinite scrolling volumes, radiance cache, irradiance volumes, and lightmap baking.

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P2.1 | [~] | Implement DDGI probe placement system |
| T-GIR-P2.2 | [-] | Implement DDGI probe ray tracing -- hardware RT path |
| T-GIR-P2.3 | [-] | Implement DDGI probe ray tracing -- rasterised fallback |
| T-GIR-P2.4 | [~] | Implement DDGI probe update |
| T-GIR-P2.5 | [~] | Implement DDGI probe sampling at shading point |
| T-GIR-P2.6 | [~] | Implement DDGI infinite scrolling volumes |
| T-GIR-P2.7 | [-] | Implement radiance cache |
| T-GIR-P2.8 | [~] | Implement irradiance volume system |
| T-GIR-P2.9 | [-] | Implement light probe lightmap baker |

## Existing Architecture

### DDGI Probe Placement (T-GIR-P2.1)

**Python (`gi_ddgi.py`)**:
- `DDGIProbeGrid(dataset, bounds, resolution)` -- builds grid from AABB + probe count
- `world_to_grid(position)`, `get_probe_index(ix, iy, iz)`, `scroll_grid(camera_pos)`
- `get_probes_for_update()` -- returns N probes per frame for distributed update

**Rust (`ddgi.rs`)**:
- `DDGIProbeVolume { origin, extents, probe_count, probe_spacing }`
- No camera-relative placement, no configurable spacing tiers

### DDGI Probe Update (T-GIR-P2.4)

**WGSL (`ddgi.wgsl`)**:
```
@compute @workgroup_size(8, 8, 1) ddgi_update_probes
```
- Per-frame subset: `NUM_FRAMES_PER_UPDATE = 8`, updates `total_probes / 8` per frame
- Accumulates irradiance from `num_rays` ray directions into SH coefficients
- Temporal blending: `mix(old, new, 0.3)` (fixed alpha = 0.3)
- **Issue**: Uses procedural sky/ground colors -- no scene tracing

**Python (`gi_ddgi.py`)**:
- `DDGIUpdatePass`: Fibonacci spiral ray directions, rotation jitter (golden angle), octahedral accumulation, configurable hysteresis

### DDGI Probe Sampling (T-GIR-P2.5)

**WGSL (`ddgi.wgsl`)**:
```
@compute @workgroup_size(8, 8, 1) ddgi_sample_probes
```
- Reads G-buffer: world_position_texture, world_normal_texture
- 8-probe trilinear interpolation with fractional grid coordinates
- SH evaluation in direction of surface normal
- Writes indirect irradiance to storage texture (rgba16float)
- **Missing**: visibility modulation, parallax correction, wall-normal weighting

**Python (`gi_ddgi.py`)**:
- `DDGILookup`: Chebyshev visibility weighting, normal bias, view bias, backface rejection

### Infinite Scrolling (T-GIR-P2.6)

**Python (`gi_ddgi.py`)**:
- `DDGIProbeGrid.scroll_grid(camera_pos)`: detects axis overflow, shifts probe indices
- `_scroll(axis, direction)`: shifts data, marks edge probes as `NEXT_PLACED`
- No GPU scroll shader (`ddgi_grid_shift.comp.wgsl`)

### Irradiance Volumes (T-GIR-P2.8)

**Python (`gi_probes.py`)**:
- `IrradianceVolume(dataset, probe_grid, blend_distance, falloff_mode)`
- `sample(world_position, world_normal)`: trilinear interpolation with edge falloff
- `falloff_mode`: none, linear, smooth
- No `IrradianceVolumeManager` for multi-volume cross-fade

## Architecture Gaps

| Component | Status | Notes |
|-----------|--------|-------|
| Probe placement | Partial | Python grid, no camera-relative GPU placement |
| RT probe tracing | Not built | Requires S10 TLAS |
| Rasterized fallback | Not built | Requires 6 face maps |
| Probe update | Partial | WGSL skeleton, placeholder data |
| Probe sampling | Partial | Trilinear works, no visibility/parallax |
| Infinite scrolling | Partial | Python only, no WGSL shift shader |
| Radiance cache | Not built | 64x64x32 3D texture |
| Irradiance volumes | Partial | Single volume, no multi-volume manager |
| Lightmap baker | Not built | No editor tool, no .ktx2 |

## Build Dependencies

- T-GIR-P2.1: T-GIR-P1.2 (probe storage)
- T-GIR-P2.2: T-GIR-P2.1, S10 (TLAS)
- T-GIR-P2.3: T-GIR-P2.1
- T-GIR-P2.4: T-GIR-P2.2, T-GIR-P2.3
- T-GIR-P2.5: T-GIR-P1.1, T-GIR-P2.1
- T-GIR-P2.6: T-GIR-P2.1
- T-GIR-P2.7: T-GIR-P2.4
- T-GIR-P2.8: T-GIR-P2.1
- T-GIR-P2.9: T-GIR-P1.1

## Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| DDGI update shader traces against scene depth | Failing -- uses placeholder colors |
| Trilinear interpolation works | Passing -- in both WGSL and Python |
| Infinite scrolling shifts probe grid | Partial -- Python only |
| Radiance cache updates in real time | Failing -- not built |
| Multi-volume blending works | Failing -- not built |
| Lightmap baker produces .ktx2 output | Failing -- not built |
