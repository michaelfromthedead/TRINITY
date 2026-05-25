# PHASE 6 ARCHITECTURE: Particle/VFX Rendering

> **Phase**: 6/7 | **Status**: [x] 70% (3 implemented, 1 partial, 1 absent)
> **Tasks**: T-GPU-6.1 through T-GPU-6.5 (5 tasks)
> **Gaps**: S9-G5, S9-G6, S9-G7

---

## Files Implemented

| File | Lines | Role |
|------|-------|------|
| `crates/.../shaders/particles.wgsl` (particle_render) | 37 lines (195-227) | Compute-based billboard generation |
| `crates/.../src/particles.rs` | 431 | Render pass factory |
| `engine/rendering/particles/trail_renderer.py` | 815 | Trail/ribbon rendering |
| `engine/rendering/particles/decal_system.py` | 946 | Deferred decal system |
| `engine/rendering/particles/vfx_graph.py` | 946 | VFX graph authoring |
| `engine/rendering/particles/gpu_particles.py` | 776 | GPU particle orchestrator |
| `engine/rendering/particles/particle_modules.py` | 1060 | Reusable particle modules |

## Reality by Task

### T-GPU-6.1: Billboard rendering [~] COMPUTE-BASED ALTERNATIVE
- No `billboard.vert.wgsl` or `billboard.frag.wgsl` files exist
- Instead: `particles.wgsl` `particle_render()` is a compute shader that generates 6 quad vertices per particle:
  ```
  for each thread gid.x < alive_count:
      right = camera_right * particle.size
      up = camera_up * particle.size
      // Write 6 vertices (2 triangles) to vertex_buffer
      vertex_buffer[base_idx + 0] = position + right + up     // top-right
      vertex_buffer[base_idx + 1] = position - right - up     // bottom-left
      vertex_buffer[base_idx + 2] = position - right + up     // top-left
      vertex_buffer[base_idx + 3] = position + right + up     // top-right
      vertex_buffer[base_idx + 4] = position + right - up     // bottom-right
      vertex_buffer[base_idx + 5] = position - right - up     // bottom-left
  ```
- No texture/bindless texture support in current implementation (single color via basic rendering)
- No additive blend support in current pass builder
- Camera uniforms read via binding(4)

### T-GPU-6.2: Mesh particle rendering [ - ] NOT IMPLEMENTED
- No `mesh_particle.vert.wgsl` file
- `gpu_particles.py` defines `DrawMode.MESH` enum value but no GPU implementation
- No instanced indirect draw using mesh table reference

### T-GPU-6.3: Trail rendering [x] IMPLEMENTED
`trail_renderer.py` (815 lines) full implementation:
- `TrailBuffer` — ring buffer with head/tail wrapping
- `TrailVertex` — position, normal, tangent, uv, color
- `TrailConfig` — cap mode (ROUND, FLAT, ARROW), UV mode (STRETCH, TILE)
- Catmull-Rom spline interpolation for smooth curves
- Per-vertex color alpha fade along trail length
- `TrailRenderer.render()` — generates CPU geometry for frame graph

### T-GPU-6.4: Deferred decal system [x] IMPLEMENTED
`decal_system.py` (946 lines) full implementation:
- `DecalInstance` — world-space position, rotation, scale, material
- `DecalAtlas` — atlas packing with occupancy tracking
- `DecalRenderer` — deferred decal rendering with:
  - Volume projection (box/sphere)
  - Per-channel blend modes (alpha, additive, multiply, overlay)
  - Material property blending
- No dedicated GPU compute shader (decal is CPU-driven)

### T-GPU-6.5: VFX graph [x] IMPLEMENTED
`vfx_graph.py` (946 lines) full implementation:
- `VFXGraph` class with `compile()` method (line 830) producing `ParticleEmitter`
- `VFXModule` — 13 base module types + composable pipeline
- `VFXContextType` — SPAWN, UPDATE, RENDER, EVENT, GLOBAL
- `VFXParameterType` — FLOAT, FLOAT2-4, INT, BOOL, COLOR, CURVE, GRADIENT, TEXTURE, MESH
- `VFXNodeType` — 26 node types (SpawnRate, Burst, Shape, Color, Size, Velocity, Noise, etc.)
- `VFXEventTrigger` — SPAWN, DEATH, COLLISION, CUSTOM event wiring
- JSON serialization support
- Integration with `particle_modules.py` (BillboardRenderer, BurstEmitter, ColorOverLife, etc.)

## Particle System Module Graph (particle_modules.py)

```
ParticleModule (ABC)
  ├── Spawn Modules
  │   ├── RateEmitter     (continuous spawn)
  │   ├── BurstEmitter    (burst spawn)
  │   └── ShapeEmitter    (spawn within shape volume)
  ├── Update Modules
  │   ├── LifetimeModule  (age tracking)
  │   ├── GravityModule   (force)
  │   ├── VelocityModule  (speed control)
  │   ├── DragModule      (air resistance)
  │   ├── ColorOverLifeModule   (gradient color)
  │   ├── SizeOverLifeModule    (size curve)
  │   ├── RotationModule  (angular velocity)
  │   ├── NoiseModule     (turbulence)
  │   ├── VortexModule    (vortex force)
  │   └── AttractorModule (point/line attract)
  └── Render Modules
      ├── BillboardRenderer  (camera-facing)
      └── MeshRenderer       (mesh instancing)
```

## Recommended Implementation Path for T-GPU-6.2 (mesh particles)
```
MeshParticleRenderer:
  1. Bind MeshTable at group(N) binding(M)
  2. In vertex shader, read particle data from storage buffer
  3. For each instance, look up mesh table entry by particle.mesh_id
  4. Apply particle position/rotation/scale as instance transform
  5. Output: instanced indirect draw with per-instance data
```
Requires MeshTable bindless access from vertex shader context.
