# Phase 1: GPU Light Data Infrastructure Architecture

## Current Status: ABSENT (0/5 tasks real)

No Rust light type definitions exist. `pbr.frag.wgsl` has 3 inline light structs.

## Architecture

### Light Type GPU Layout

All 7 light types need repr(C) structs in a tagged union:

```
struct DirectionalLight { direction: vec3<f32>, _pad: f32, color: vec3<f32>, intensity: f32 }
struct PointLight { position: vec3<f32>, radius: f32, color: vec3<f32>, intensity: f32 }
struct SpotLight { position: vec3<f32>, radius: f32, direction: vec3<f32>, cos_outer: f32, cos_inner: f32, _pad0: f32, _pad1: f32, color: vec3<f32>, intensity: f32 }
struct RectAreaLight { position: vec3<f32>, _pad0: f32, direction: vec3<f32>, _pad1: f32, up: vec3<f32>, width: f32, height: f32, two_sided: u32, _pad2: f32, _pad3: f32, color: vec3<f32>, intensity: f32 }
struct DiskAreaLight { position: vec3<f32>, _pad0: f32, direction: vec3<f32>, _pad1: f32, radius: f32, two_sided: u32, _pad2: f32, _pad3: f32, color: vec3<f32>, intensity: f32 }
struct IESLight { position: vec3<f32>, radius: f32, direction: vec3<f32>, _pad0: f32, color: vec3<f32>, intensity: f32, profile_index: u32, _pad1: f32, _pad2: f32 }
struct SkyLight { ... } // cubemap reference, no GPU struct needed
```

These MUST match the Python reference in `light_types.py` for field semantics.

### File Structure

```
crates/renderer-backend/src/lighting/
  mod.rs              -- Public API, re-exports
  light_types.rs      -- LightTypeGPU enum, LightUnion, 7 repr(C) structs, static_assert size checks
  light_buffer.rs     -- SoA buffer builder, CPU-side layout computation
  light_system.rs     -- Orchestrator (dirty tracking, upload scheduling) -- Rust counterpart to lighting_system.py
```

### Dependencies

- `gpu_driven/buffers.rs` for staging buffer allocation and upload
- `pbr.frag.wgsl` struct definitions replaced by shared `light_types.wgsl` include
- Python `light_types.py` as reference for field semantics and byte layout

### Key Design Decisions

1. **AoS vs SoA**: WGSL uses AoS (`array<PointLight>`) for simplicity. The Rust buffer builder can store as SoA for GPU efficiency if needed, but must expose AoS WGSL bindings.
2. **LightUnion overhead**: A tagged union per light adds 4 bytes overhead. For 1024 lights this is trivial. Use `LightType` enum + `array<u32>` for light type dispatch rather than a full union if performance is critical.
3. **Dirty tracking**: Each light needs a frame-number dirty stamp. The orchestrator polls at the start of each frame and rebuilds only changed lights.
