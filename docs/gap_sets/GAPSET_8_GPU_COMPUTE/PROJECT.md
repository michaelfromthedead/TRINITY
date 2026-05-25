# PROJECT: GAPSET 8 -- GPU Compute Infrastructure

> **Codebase**: TRINITY Graphics Engine
> **Root**: `/home/user/dev/USER/PROJECTS_VOID/TRINITY`
> **RDC Date**: 2026-05-22

---

## Overview

GAPSET_8_GPU_COMPUTE covers the GPU compute infrastructure: compute shader dispatch, buffer management, GPU-driven rendering (bindless tables, culling, indirect draw), and GPU particle systems. It spans 7 phases, 32 tasks, and crosses Rust (renderer-backend), Python (decorators, engine), and WGSL (shaders).

## Architecture

```
TRINITY Graphics Engine -- GPU Compute Subsystem
==================================================

Python Layer (engine/rendering/gpu_driven/)
  culling.py           -- CPU-side frustum, distance, occlusion culling pipeline
  indirect_draw.py     -- Indirect draw command structures (CPU)
  bindless.py          -- Bindless resource management (CPU)
  meshlet.py           -- Meshlet generation and culling (CPU)
  visibility_buffer.py -- Visibility buffer processing (CPU)
  instancing.py        -- Instance management (CPU)

Python Layer (engine/rendering/particles/)
  gpu_particles.py     -- GPU particle simulation orchestrator
  particle_system.py   -- CPU particle system (ParticleEmitter, simulation)
  particle_modules.py  -- Reusable particle behavior modules
  trail_renderer.py    -- CPU trail/ribbon rendering
  decal_system.py      -- Deferred decal system
  vfx_graph.py         -- VFX graph authoring and compilation
  constants.py         -- Particle system constants

Decorator Layer (trinity/decorators/)
  gpu.py               -- @gpu_buffer, @gpu_struct, @gpu_kernel, @bind_group, etc.
  rendering.py         -- @render_layer, @shadow_caster, @gi_contributor, etc.
  lod_streaming.py     -- @lod, @streamable, @chunk, etc.
  particles_vfx.py     -- @gpu_particle, @vfx_event decorators

Rust Layer (crates/renderer-backend/)
  src/gpu_driven/
    mod.rs             -- Module exports
    buffers.rs         -- BufferRegistry with triple-buffered staging (777 lines)
    material_table.rs  -- Bindless Material Table (1302 lines)
    material_table.wgsl-- WGSL material entry struct + helpers
    mesh_table.rs      -- Bindless Mesh Table (1803 lines)
    mesh_table.wgsl    -- WGSL mesh entry struct + helpers
    texture_table.rs   -- Bindless Texture Table with free-list (271 lines)
  src/particles.rs     -- Particle system pass builders (431 lines)
  src/memory.rs        -- FrameAllocator, PoolAllocator, StackAllocator, GpuBudget
  src/ddgi.rs          -- DDGI pass (likely similar structure)
  src/frame_graph/     -- Frame graph core (mod.rs, python.rs)

WGSL Shaders (crates/renderer-backend/shaders/)
  particles.wgsl       -- Spawn/Update/Render/Compact compute shaders
  light_culling.wgsl   -- Forward+ clustered light culling
  ddgi.wgsl            -- Dynamic Diffuse GI (update + sample probes)
  pbr.vert/frag.wgsl   -- PBR vertex/fragment shaders
  shadow.vert/frag.wgsl-- Shadow map shaders
  shadow_csm.wgsl      -- Cascaded shadow map
```

## Data Flow

```
CPU writes → BufferRegistry (triple-buffered staging)
              ├── MeshTable (array<MeshTableEntry>)
              ├── MaterialTable (array<MaterialTableEntry>, dirty-flag tracking)
              ├── TextureTable (free-list, MAX_BINDLESS_TEXTURES=4096)
              └── Instance data
                    ↓ (GPU reads from front-buffer slot)
GPU compute shaders
  ├── Frustum cull instances    [NOT IMPL]
  ├── HZB occlusion cull        [NOT IMPL]
  ├── LOD/distance cull         [NOT IMPL]
  ├── Generate indirect draw    [NOT IMPL]
  ├── Particle spawn/update/render/compact [IMPL in particles.wgsl]
  └── Light culling             [IMPL in light_culling.wgsl]
       ↓ (indirect draw commands)
Graphics pipeline
  ├── Opaque layer
  ├── Transparent layer (sorted)
  └── Overlay layer
```

## Key Design Decisions

1. **Triple-buffered staging** (buffers.rs): CPU writes frame N while GPU reads frame N-1; no sync stalls in common case. Back-pressure only when all 3 slots occupied.

2. **Bindless tables**: Mesh/material/texture tables are `array<T>` in GPU storage buffers, accessed by u32 index. No per-object bind groups.

3. **Material dirty-flag tracking**: Bit 31 of flags marks modified entries; staging pipeline uploads entire table (no range tracking yet).

4. **Free-list slot reuse**: TextureTable recycles slots via free-list; material_table zeroes removed slots (index remains valid as sentinel).

5. **Swap-based particle compaction**: Simplified single-pass compaction (racy but converging) rather than prefix-sum + scatter. Acceptable for VFX particles per design.

6. **Compute-based billboard rendering**: Particle quads generated via compute shader (vertex buffer fill) rather than vertex shader with instance data pull.

## Gap Coverage

| Gap ID | Description | Status |
|--------|-------------|--------|
| S2-G1 | GPU culling (frustum, HZB, occlusion, distance, LOD) | **[-]** Not implemented |
| S2-G2 | GPU struct layout (alignment WGSL compat) | **[x]** Implemented |
| S2-G3 | Decorator integration (layer dispatch, LOD, shadow) | **[~]** Decorators exist, GPU integration missing |
| S2-G4 | Indirect draw, multi-draw, fallback tiers | **[~]** Python side exists, GPU/fallback missing |
| S2-G5 | Bindless tables (mesh, material, texture) | **[x]** All three implemented |
| S2-G6 | GPU radix sort | **[-]** Not implemented |
| S2-G7 | Meshlet pipeline | **[~]** Python side exists, GPU missing |
| S9-G1 | GPU particle spawn | **[x]** Implemented |
| S9-G2 | GPU particle update | **[x]** Implemented |
| S9-G3 | GPU particle compact | **[x]** Implemented (swap-based) |
| S9-G4 | GPU particle sort | **[-]** Not implemented |
| S9-G5 | VFX graph | **[x]** Implemented |
| S9-G6 | Trail rendering | **[x]** Implemented |
| S9-G7 | Deferred decals | **[x]** Implemented |
| S9-G8 | Frame graph registration (culling) | **[~]** Particle passes registered |
| S9-G9 | Frame graph registration (particle) | **[x]** Particle passes created |
| S9-G10 | Frame graph dependency ordering | **[~]** Partial |
