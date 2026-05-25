# Phase 5: Reflection Probe System -- Architecture

## Overview

Phase 5 implements the reflection probe pipeline: baked cubemap capture, realtime probe capture, probe blending, parallax correction, pre-filtered cubemaps (GGX), and probe atlas management.

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P5.1 | [-] | Implement baked probe capture |
| T-GIR-P5.2 | [-] | Implement realtime probe capture |
| T-GIR-P5.3 | [-] | Implement probe blending |
| T-GIR-P5.4 | [~] | Implement parallax correction |
| T-GIR-P5.5 | [-] | Implement pre-filtered cubemaps |
| T-GIR-P5.6 | [-] | Implement probe atlas |

## Existing Architecture

### ReflectionProbe (Python: `gi_probes.py`)

```python
@dataclass
class ReflectionProbeConfig:
    capture_mode: str = "static"     # static, realtime, baked
    resolution: int = 256
    update_rate: float = 0.0

class ReflectionProbe:
    config: ReflectionProbeConfig
    bounds: BoundingBox  # AABB in world space
    blend_distance: float
    
    def needs_update(self) -> bool: ...
    def get_blend_factor(self, world_position) -> float: ...
    def sample(self, world_position, world_normal, roughness, lod_bias) -> Color: ...
```

### Decorator: 3 of 11 Parameters

```python
def reflection_probe(capture_mode="static", resolution=256, update_rate=0.0):
```

Missing 8 parameters: importance, box_extents, inner_radius, outer_radius, roughness_levels, capture_lod_bias, include_layers, exclude_actors.

`GIImportance` enum (CRITICAL, HIGH, MEDIUM, LOW, OFF) exists in `light_types.py` but not wired.

### Parallax Correction (T-GIR-P5.4) -- Existing

**Python (`gi_probes.py`)**: `ReflectionProbe._parallax_correct(position, direction)`

Implements ray-box intersection for UE4/Lagarde box projection:
```
1. Compute intersection of reflected ray with probe AABB
2. t_near = entry point, t_far = exit point
3. Use far intersection point as corrected reflection direction
4. Return direction from corrected position to intersection point
```

This is a correct CPU implementation but no WGSL equivalent exists.

## Architecture Gaps

### Baked Probe Capture (T-GIR-P5.1)
- `LightProbe.bake()` exists and samples SH via Fibonacci spiral
- No cubemap rendering pipeline
- No BC6H compression
- No .ktx2 storage or mip chain generation

### Realtime Probe Capture (T-GIR-P5.2)
- `ReflectionProbe.needs_update` returns True for realtime probes
- No face_to_render scheduler
- No actual cubemap rendering pipeline
- No CPU scheduler module (`reflection_probes.py`)

### Probe Blending (T-GIR-P5.3)
- `ReflectionProbe.get_blend_factor()` computes per-probe blend factor from distance to probe bounds
- No per-pixel multi-probe blending
- No WGSL blend shader (`probe_blend.comp.wgsl`)

### Pre-filtered Cubemaps (T-GIR-P5.5)
- `ReflectionProbe.sample()` has placeholder comment about mip level
- No GGX distribution pre-filter
- No WGSL pre-filter shader (`probe_prefilter.comp.wgsl`)

### Probe Atlas (T-GIR-P5.6)
- No `probe_atlas.py` module
- No atlas packing algorithm
- No atlas update scheduling

## Dependencies

- T-GIR-P5.1: S16 (asset pipeline for .ktx2)
- T-GIR-P5.2: T-GIR-P5.1
- T-GIR-P5.3: T-GIR-P5.2
- T-GIR-P5.4: T-GIR-P5.2
- T-GIR-P5.5: T-GIR-P5.2
- T-GIR-P5.6: T-GIR-P5.2

## Files to Create

| File | Purpose |
|------|---------|
| `crates/renderer-backend/shaders/probe_prefilter.comp.wgsl` | GGX cubemap pre-filter |
| `crates/renderer-backend/shaders/probe_blend.comp.wgsl` | Multi-probe blending |
| `crates/renderer-backend/shaders/probe_parallax_correction.wgsl` | GPU parallax correction |
| `crates/renderer-backend/src/reflection_probes.rs` | Probe runtime management |
| `engine/rendering/lighting/probe_atlas.py` | Atlas packing |
| `engine/rendering/lighting/reflection_probes_cpu.py` | CPU scheduler |

## Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| Baked probe produces BC6H cubemap | Failing -- not built |
| Realtime probe captures cubemap each frame | Failing -- not built |
| Multi-probe blending produces smooth transitions | Failing -- not built |
| Parallax correction works on GPU | Failing -- Python only |
| Pre-filtered cubemap minimizes specular aliasing | Failing -- not built |
| Probe atlas supports 256+ probes | Failing -- not built |
