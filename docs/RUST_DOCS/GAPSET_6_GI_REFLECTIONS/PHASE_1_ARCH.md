# Phase 1: Foundation Infrastructure -- Architecture

## Overview

Phase 1 provides the mathematical and structural foundation for all GI and reflection techniques: spherical harmonics math, probe GPU storage, decorator extensions, performance budget monitoring, and reflection buffer format specification.

## Tasks

| ID | Status | Description |
|----|--------|-------------|
| T-GIR-P1.1 | [~] | Implement spherical harmonics math library |
| T-GIR-P1.2 | [~] | Define probe GPU storage buffers |
| T-GIR-P1.3 | [~] | Extend @reflection_probe decorator with 8 missing parameters |
| T-GIR-P1.4 | [-] | Define GI performance budget table |
| T-GIR-P1.5 | [-] | Define reflection buffer format |

## Existing Architecture

### Spherical Harmonics (T-GIR-P1.1)

**WGSL (`ddgi.wgsl`)**:
- L0+L1 only: `SH_L0 = 0.28209479177387814`, `SH_L1 = 0.4886025119029199`
- Three functions: `eval_sh(channel_coeffs, dir)`, `eval_sh_rgb(probe, dir)`, `project_sh(dir, irradiance)`
- Coefficients stored per channel as `vec4<f32>`: [L0, L1.x, L1.y, L1.z]
- No `sh_rotate`, `sh_convolve_irradiance`

**Python (`gi_probes.py`)**:
- Full L2: `SphericalHarmonics` class with 27 coefficients (9 per channel)
- Methods: `evaluate(direction)`, `add_sample(direction, color)`, `scale(factor)`, `add(other)`, `lerp(a, b, t)`
- Order-2 basis: Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22

### Probe GPU Storage (T-GIR-P1.2)

**Rust (`ddgi.rs`)**:
```rust
pub struct DDGIProbeVolume {
    pub origin: [f32; 3],
    pub extents: [f32; 3],
    pub probe_count: [u32; 3],
    pub probe_spacing: f32,
}
```

**WGSL (`ddgi.wgsl`)**:
- `DDGIProbe`: `sh_r: vec4<f32>`, `sh_g: vec4<f32>`, `sh_b: vec4<f32>` -- `vec4<f32>` stride is 16 bytes (std430 aligned)
- `ProbeVolume`: origin, extents, spacing, num_rays, max_ray_distance, energy_preservation, num_irradiance_texels, num_depth_texels
- Storage layout: `array<DDGIProbe, MAX_PROBES>` where MAX_PROBES = 4096

### Decorator (T-GIR-P1.3)

**Python (`gi_probes.py`)**:
```python
def reflection_probe(
    capture_mode: str = "static",
    resolution: int = 256,
    update_rate: float = 0.0,
)
```
Parameters implemented: 3 of 11 planned. `GIImportance` enum (CRITICAL, HIGH, MEDIUM, LOW, OFF) exists in `light_types.py` but not wired to probe system.

## Architecture Gaps

### Missing: Performance Budget Table (T-GIR-P1.4)
- No `gi_config.py` module
- `DDGIConstants` in `constants.py` has ray count, bias, hysteresis defaults only
- No GPU timestamp instrumentation
- No fallback logic
- Design target: budget table with per-technique cost caps and automatic downscaling

### Missing: Reflection Buffer Format (T-GIR-P1.5)
- No `ReflectionBuffer` struct
- No bilateral upscale specification
- Design target: half-resolution RGBA16F buffer with bilateral upscale to full resolution

## Build Dependencies

- T-GIR-P1.1: None
- T-GIR-P1.2: T-GIR-P1.1 (SH structs used by probe storage)
- T-GIR-P1.3: None
- T-GIR-P1.4: T-GIR-P1.2 (budget depends on probe storage layout)
- T-GIR-P1.5: None

## Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| SH library has L2 validation | Partial -- Python correct, WGSL L0+L1 only |
| Probe GPU buffers compile with correct alignment | Verified -- `vec4<f32>` stride 16 bytes |
| Decorator supports 11 parameters | Failing -- only 3 of 11 |
| Budget table exists | Failing -- not built |
| Reflection buffer format specified | Failing -- not built |
