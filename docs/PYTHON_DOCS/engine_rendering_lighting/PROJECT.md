# PROJECT: engine/rendering/lighting GPU Integration

## Scope

Complete GPU integration for TRINITY's lighting subsystem. The CPU-side implementation (4,470 lines) contains production-quality mathematics and data structures for shadow mapping, clustered lighting, and global illumination. All GPU execution paths are stubbed with placeholder integers and hardcoded return values.

## Current State

- **6 core files** with complete algorithms
- **7 light types** with proper attenuation math
- **4 shadow map types** (CSM, cube, spot, atlas)
- **5 shadow filter types** (PCF, PCSS, VSM, ESM, contact)
- **Clustered lighting** with froxel grid and culling
- **Global illumination** via SH probes, DDGI, lightmaps, reflection probes

All math is implemented. Zero GPU code exists.

## Goals

1. Create actual GPU resources (textures, buffers, handles) for all lighting components
2. Generate WGSL shaders for shadow sampling, light evaluation, and GI lookup
3. Implement render passes for shadow map generation
4. Wire froxel culling output to GPU-uploadable light lists
5. Implement DDGI ray tracing pass (or provide adapter for existing RT system)
6. Enable reflection probe cubemap sampling

## Constraints

- Must preserve existing CPU-side API surface
- GPU resources must integrate with renderer-backend crate
- Shader generation must follow existing codegen patterns
- DDGI ray tracing must work with or without hardware RT
- Memory budget for shadow atlas: configurable, default 4096x4096

## Acceptance Criteria

### Phase 1: Shadow Infrastructure
- [ ] Shadow textures created via renderer-backend GPU resource API
- [ ] Shadow atlas allocator returns real texture regions
- [ ] CSM depth passes render actual geometry

### Phase 2: Shadow Filtering
- [ ] WGSL shaders for PCF, PCSS, VSM, ESM
- [ ] Contact shadow ray march in screen space
- [ ] Shadow sampling integrated with main lighting shader

### Phase 3: Clustered Lighting
- [ ] Froxel grid uploaded to GPU buffer
- [ ] Per-froxel light lists accessible in shaders
- [ ] Light evaluation shader with proper attenuation

### Phase 4: Global Illumination
- [ ] SH probe coefficients uploaded and sampled
- [ ] Reflection probe cubemaps created and sampled
- [ ] DDGI octahedral textures created
- [ ] DDGI update pass (ray trace + accumulate)
- [ ] DDGI lookup in lighting shader

## Files in Scope

| File | Lines | Integration Need |
|------|-------|------------------|
| `light_types.py` | 650 | Shader uniform generation |
| `shadows.py` | 784 | GPU texture creation, render passes |
| `shadow_filtering.py` | 796 | WGSL filter shaders |
| `light_culling.py` | 618 | GPU buffer upload |
| `gi_probes.py` | 779 | SH textures, cubemap creation |
| `gi_ddgi.py` | 843 | RT pass, octahedral textures |

## Dependencies

- `renderer-backend` crate for GPU resource creation
- Existing shader codegen infrastructure
- Frame graph for render pass scheduling
- Ray tracing system (software or hardware) for DDGI

## Risks

1. **DDGI RT performance** - May need compute fallback for non-RT GPUs
2. **Shadow atlas fragmentation** - Defrag algorithm exists but untested at scale
3. **SH precision** - L2 may be insufficient for high-frequency lighting
4. **Memory pressure** - DDGI + reflection probes + shadow atlas could exceed budget

## Success Metrics

- All shadow filtering modes produce correct soft shadows
- Clustered lighting handles 1000+ lights without frame drops
- DDGI provides real-time indirect lighting updates
- Reflection probes show parallax-corrected reflections
