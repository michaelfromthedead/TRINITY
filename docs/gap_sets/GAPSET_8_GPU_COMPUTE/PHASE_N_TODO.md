# Phase TODO -- GAPSET_8_GPU_COMPUTE

> **Task ID format**: T-GPU-{PHASE}.{N}
> **Total tasks**: 32
> **Gaps covered**: S2-G1 through S2-G7, S9-G1 through S9-G10

---

## Phase 1: Foundation -- Buffer Management & Bindless Infrastructure

**Dependencies**: S14 (RHI), S3 (Materials table specification)
**Gaps**: S2-G2 (BLOCKING), S2-G4 (partial), S2-G5
**Effort**: 3-4 weeks

### Tasks

- [ ] **T-GPU-1.1**: Extend `@gpu_struct` to support `Vec2`, `Vec3`, `Vec4`, `Mat4`, `f32[N]`, and nested struct types with proper WGSL `@size`/`@align` alignment
  - Acceptance: MeshTableEntry, MaterialTableEntry, GPUInstance can be defined with @gpu_struct and produce correct byte-accurate layouts
  - Gap: S2-G2 (BLOCKING -- all subsequent phases require this)
  - Files: `trinity/decorators/gpu.py`

- [ ] **T-GPU-1.2**: Implement `BufferRegistry` with triple-buffered staging for instance data uploads
  - Acceptance: CPU can write frame N while GPU reads frame N-1; no sync stalls
  - Gap: S2-G4 (partial)
  - Files: `crates/renderer-backend/src/gpu_driven/buffers.rs`

- [ ] **T-GPU-1.3**: Create bindless Mesh Table GPU buffer (`array<MeshTableEntry>`) with CPU-side manager for mesh load-time population
  - Acceptance: Meshes loaded via S16 Asset Pipeline automatically append to MeshTable; instances reference by u32 index
  - Gap: S2-G5 (partial)
  - Files: `crates/renderer-backend/src/gpu_driven/bindless.rs`

- [ ] **T-GPU-1.4**: Create bindless Material Table GPU buffer (`array<MaterialTableEntry>`) with CPU-side manager for dirty flag updates
  - Acceptance: Material parameter changes via @track_changes propagate to GPU table within 1 frame
  - Gap: S2-G5 (partial)
  - Files: `crates/renderer-backend/src/gpu_driven/bindless.rs`, `crates/renderer-backend/src/materials/material_table.rs`

- [ ] **T-GPU-1.5**: Create bindless Texture Table (`texture_2d_array<f32>`, MAX_BINDLESS_TEXTURES=4096 slots) with free-list management
  - Acceptance: 4096 textures can be bound; slot reuse without bind group invalidation
  - Gap: S2-G5 (partial)
  - Files: `crates/renderer-backend/src/gpu_driven/bindless.rs`

- [ ] **T-GPU-1.6**: Wire `@gpu_buffer` decorator to wgpu buffer allocation with proper usage flags
  - Acceptance: @gpu_buffer(usage={"storage", "indirect"}) creates wgpu buffer with STORAGE | INDIRECT | COPY_DST usage
  - Gap: S2-G4 (partial), S2-G5 (partial)
  - Files: `trinity/decorators/gpu.py`

---

## Phase 2: GPU Compute Core -- Radix Sort & Compaction

**Dependencies**: T-GPU-1.1, T-GPU-1.2
**Gaps**: S2-G6, S2-G1 (compaction), S9-G3 (compact), S9-G4 (sort)
**Effort**: 2-3 weeks

### Tasks

- [ ] **T-GPU-2.1**: Implement GPU radix sort compute shader (32-bit key, 4-bit radix, 8 passes)
  - Acceptance: Correctly sorts 100K 32-bit keys in < 0.5ms; matches CPU reference sort
  - Gap: S2-G6
  - Files: `shaders/gpu_driven/gpu_sort.comp.wgsl`, `crates/renderer-backend/src/gpu_driven/sort.rs`

- [ ] **T-GPU-2.2**: Implement visibility/particle buffer compaction compute shader (prefix sum + scatter)
  - Acceptance: Compacts [alive, dead, alive, alive] to [0, 2, 3] with correct alive count
  - Gap: S2-G1 (compaction), S9-G3
  - Files: `shaders/gpu_driven/gpu_compact.comp.wgsl`

- [ ] **T-GPU-2.3**: Extract shared prefix-sum utility function to `shaders/common/` for reuse across compaction and sort passes
  - Acceptance: Both compaction and sort histogram passes use the same prefix-sum WGSL code
  - Gap: S2-G1 (partial), S9-G3
  - Files: `shaders/common/prefix_sum.wgsl`

- [ ] **T-GPU-2.4**: Implement indirect draw command buffer structures with CPU-side manager
  - Acceptance: IndirectDrawIndexedArgs struct matches wgpu expected layout; buffer allocated with INDIRECT | STORAGE usage
  - Gap: S2-G4
  - Files: `crates/renderer-backend/src/gpu_driven/indirect_draw.rs`

---

## Phase 3: Core Culling Pipeline (S2)

**Dependencies**: T-GPU-2.2 (compaction), T-GPU-2.4 (indirect buffer)
**Gaps**: S2-G1 (frustum, distance/LOD culling), S2-G4 (draw-arg generation)
**Effort**: 2-3 weeks

### Tasks

- [ ] **T-GPU-3.1**: Implement frustum culling compute shader (sphere test then AABB test, 1 thread per instance)
  - Acceptance: Correctly culls instances outside all 6 frustum planes; handles camera-inside and zero-radius edge cases
  - Gap: S2-G1 (partial)
  - Files: `shaders/gpu_driven/gpu_cull_frustum.comp.wgsl`, `engine/rendering/gpu_driven/culling.py`

- [ ] **T-GPU-3.2**: Implement distance/LOD culling compute shader (LOD selection + max-distance cull)
  - Acceptance: Correct LOD selected per distance band; instances beyond max_draw_distance culled; @lod(bias) applied correctly
  - Gap: S2-G1 (partial)
  - Files: `shaders/gpu_driven/gpu_cull_distance.comp.wgsl`

- [ ] **T-GPU-3.3**: Implement draw-arg generation compute shader (batch detection + IndirectDrawIndexedArgs write)
  - Acceptance: Sorted visibility buffer produces correct batch boundaries; each IndirectDrawIndexedArgs has correct index_count and instance_count
  - Gap: S2-G4
  - Files: `shaders/gpu_driven/gpu_gen_draw_args.comp.wgsl`

- [ ] **T-GPU-3.4**: Implement multi-draw indirect execution with Tier 1/2/3 fallback paths
  - Acceptance: Tier 1 uses single draw_indexed_indirect_count; Tier 2 CPU readback of count; Tier 3 CPU batching
  - Gap: S2-G4
  - Files: `engine/rendering/gpu_driven/indirect_draw.py`, `crates/renderer-backend/src/gpu_driven/indirect_draw.rs`

---

## Phase 4: Occlusion Culling & Meshlet Pipeline (S2)

**Dependencies**: T-GPU-3.1 (frustum cull provides survivors for occlusion test)
**Gaps**: S2-G1 (HZB, occlusion, meshlet culling), S2-G7 (meshlet generation)
**Effort**: 3-4 weeks

### Tasks

- [ ] **T-GPU-4.1**: Implement HZB construction compute shader (4x max reduction mip chain)
  - Acceptance: Produces correct mip chain from depth buffer; each level is 4x reduction with max operator; runs in < 0.3ms at 4K
  - Gap: S2-G1 (partial)
  - Files: `shaders/gpu_driven/hzb_build.comp.wgsl`, `crates/renderer-backend/src/gpu_driven/hzb.rs`

- [ ] **T-GPU-4.2**: Implement HZB occlusion culling compute shader (project, mip-select, sample, compare)
  - Acceptance: Correctly culls instances behind known occluders within bias tolerance; conservative variant tests 8 corners
  - Gap: S2-G1 (partial)
  - Files: `shaders/gpu_driven/gpu_cull_occlusion.comp.wgsl`

- [ ] **T-GPU-4.3**: Implement meshlet generation (offline or load-time, Rust side)
  - Acceptance: Meshes partitioned into 64-vert / ~124-triangle meshlets with bounding spheres and normal cones
  - Gap: S2-G7
  - Files: `crates/renderer-backend/src/gpu_driven/meshlet_culling.rs`

- [ ] **T-GPU-4.4**: Implement meshlet culling compute shader (workgroup-per-mesh, frustum + normal cone + optional HZB)
  - Acceptance: Correctly culls meshlets outside frustum or with normals facing away; optional HZB test at meshlet granularity
  - Gap: S2-G7
  - Files: `shaders/gpu_driven/gpu_cull_meshlet.comp.wgsl`

- [ ] **T-GPU-4.5**: Implement triangle culling compute shader (backface, zero-area, sub-pixel) -- stretch goal, LOD 0 only
  - Acceptance: Correctly culls backface, zero-area, and sub-pixel triangles; only active at LOD 0
  - Gap: S2-G1 (partial)
  - Files: `shaders/gpu_driven/gpu_cull_triangle.comp.wgsl`

---

## Phase 5: GPU Particle Compute Passes (S9)

**Dependencies**: T-GPU-1.2 (buffer management), T-GPU-2.2 (compaction), T-GPU-2.1 (sort)
**Gaps**: S9-G1, S9-G2, S9-G3, S9-G4
**Effort**: 2-3 weeks

### Tasks

- [ ] **T-GPU-5.1**: Implement GPU particle spawn compute shader with indirect spawn counter
  - Acceptance: Spawns up to `spawn_count` new particles per frame; initializes position/velocity/color/lifetime from emitter config
  - Gap: S9-G1
  - Files: `shaders/particles/gpu_particle_spawn.comp.wgsl`, `engine/rendering/particles/gpu_particles.py`

- [ ] **T-GPU-5.2**: Implement GPU particle update compute shader with ping-pong SoA buffers
  - Acceptance: Applies gravity, wind, turbulence, vortex, attraction forces; advances age; marks dead particles; ping-pong buffer swap each frame
  - Gap: S9-G2
  - Files: `shaders/particles/gpu_particle_update.comp.wgsl`

- [ ] **T-GPU-5.3**: Implement GPU particle compact compute shader (prefix-sum over alive flags, scatter, write alive count to indirect buffer)
  - Acceptance: Dead particles removed each frame; alive particles contiguous in output buffer; alive_count written to indirect draw counter
  - Gap: S9-G3
  - Files: `shaders/particles/gpu_particle_compact.comp.wgsl`

- [ ] **T-GPU-5.4**: Implement GPU particle sort compute shader (depth computation + radix sort + indirection array)
  - Acceptance: Translucent particles sorted back-to-front; sort key is quantized float depth; indirection array for ordered rendering
  - Gap: S9-G4
  - Files: `shaders/particles/gpu_particle_sort.comp.wgsl`

---

## Phase 6: Particle/VFX Rendering (S9)

**Dependencies**: T-GPU-5.1-5.4 (compacted + sorted particle buffer)
**Gaps**: S9-G5 (VFX graph), S9-G6 (trails), S9-G7 (decals)
**Effort**: 2-3 weeks

### Tasks

- [ ] **T-GPU-6.1**: Implement billboard particle rendering (vertex shader with SoA attribute pull, fragment shader with bindless texture + alpha/additive blend)
  - Acceptance: VIEW/VELOCITY/CUSTOM alignment modes correct; velocity stretch blends correctly; additive blend for fire
  - Gap: S9 (indirect draw consumption), shared with S2 indirect draw
  - Files: `shaders/particles/billboard.vert.wgsl`, `shaders/particles/billboard.frag.wgsl`

- [ ] **T-GPU-6.2**: Implement mesh particle rendering (instanced indirect draw, bindless mesh table reference)
  - Acceptance: Mesh particles render at correct position/rotation/scale; scale_from_size controls mapping
  - Gap: S9 (mesh particle render gap)
  - Files: `shaders/particles/mesh_particle.vert.wgsl`

- [ ] **T-GPU-6.3**: Implement trail rendering (CPU ribbon geometry generation, integration with frame graph transparent pass)
  - Acceptance: TrailBuffer ring buffer correctly wraps; Catmull-Rom tangents correct; ROUND/FLAT/ARROW caps render; STRETCH/TILE UV modes work
  - Gap: S9-G6
  - Files: `engine/rendering/particles/trail_renderer.py`

- [ ] **T-GPU-6.4**: Implement deferred decal system (full-screen pass, volume projection, per-channel blend, atlas packing)
  - Acceptance: Decals project correctly onto geometry within bounding volume; per-channel blend modes work; atlas occupancy tracks correctly
  - Gap: S9-G7
  - Files: `engine/rendering/particles/decal_system.py`

- [ ] **T-GPU-6.5**: Implement VFX graph runtime execution engine (node compilation, event wiring, serialization)
  - Acceptance: VFXGraph.compile() produces correctly configured ParticleEmitter; JSON round-trip preserves all parameters; 26 node types functional
  - Gap: S9-G5
  - Files: `engine/rendering/particles/vfx_graph.py`

---

## Phase 7: Integration & Decorators

**Dependencies**: All prior phases
**Gaps**: S2-G3, S9-G8, S9-G9, S9-G10
**Effort**: 2-3 weeks

### Tasks

- [ ] **T-GPU-7.1**: Wire `@render_layer` decorator to layer dispatch (opaque/transparent/shadow culling streams with separate indirect draw buffers)
  - Acceptance: @render_layer("opaque") instances go to opaque cull+draw; @render_layer("transparent") uses no occlusion + sort; layer ordering correct (sky, opaque, masked, transparent, overlay)
  - Gap: S2-G1 (partial -- layer dispatch), S2-G3
  - Files: `trinity/decorators/rendering.py`, `engine/rendering/gpu_driven/culling.py`

- [ ] **T-GPU-7.2**: Wire `@lod` decorator to GPU LOD distance upload + streaming feedback readback loop
  - Acceptance: LOD distances from decorator packed into GPUInstance; GPU LOD selection read back async; streaming priority updated
  - Gap: S2-G3
  - Files: `trinity/decorators/lod_streaming.py`, `crates/renderer-backend/src/gpu_driven/lod_feedback.rs`

- [ ] **T-GPU-7.3**: Wire `@shadow_caster` decorator to shadow culling stream (second culling pass with shadow frustum)
  - Acceptance: @shadow_caster(mode="dynamic") instances re-culled each frame with shadow frustum; mode="static" cached; mode="none" excluded
  - Gap: S2-G3
  - Files: `trinity/decorators/rendering.py`, `engine/rendering/gpu_driven/shadow_culling.py`

- [ ] **T-GPU-7.4**: Wire `@bind_group` decorator to bindless descriptor layout (group 0 = bindless tables, group 1 = frame constants, group 2 = pass-specific)
  - Acceptance: @bind_group(index=0) produces correct wgpu bind group layout with texture arrays; mesh/material tables bound at correct slots
  - Gap: S2-G5 (bind_group integration)
  - Files: `trinity/decorators/gpu.py`

- [ ] **T-GPU-7.5**: Implement `@gpu_driven_mesh` composite decorator (combines @component + @lod + @render_layer + @gpu_buffer + @streamable)
  - Acceptance: Single decorator replaces 5 separate decorators; all metadata correctly merged on target class
  - Gap: S2-G3
  - Files: `trinity/decorators/gpu.py`

- [ ] **T-GPU-7.6**: Implement frame graph pass registration for culling pipeline and particle passes (S1 interop)
  - Acceptance: Culling compute pass registered as S1 node; dependency analysis correctly orders HZB build -> cull -> draw; particle transparent pass at layer order 6
  - Gap: S9-G8, S9-G9, S9-G10
  - Files: `engine/rendering/gpu_driven/culling.py`, `engine/rendering/particles/gpu_particles.py`

- [ ] **T-GPU-7.7**: Implement fallback paths (no indirect count, no descriptor indexing, no mesh shaders, HZB unavailable)
  - Acceptance: Graceful degradation on limited hardware; all fallback paths tested on Vulkan 1.0 class hardware
  - Gap: S2-G4 (fallback tier)
  - Files: `crates/renderer-backend/src/gpu_driven/fallback.rs`

---

## Task Summary

| Phase | Task IDs | Count | Primary Gaps |
|-------|----------|-------|--------------|
| 1: Foundation | T-GPU-1.1 -- T-GPU-1.6 | 6 | S2-G2, S2-G4 (partial), S2-G5 |
| 2: GPU Compute Core | T-GPU-2.1 -- T-GPU-2.4 | 4 | S2-G6, S2-G1 (compaction), S9-G3, S9-G4 |
| 3: Core Culling | T-GPU-3.1 -- T-GPU-3.4 | 4 | S2-G1 (frustum, distance), S2-G4 |
| 4: Occlusion + Meshlet | T-GPU-4.1 -- T-GPU-4.5 | 5 | S2-G1 (HZB, occlusion, meshlet), S2-G7 |
| 5: Particle Compute | T-GPU-5.1 -- T-GPU-5.4 | 4 | S9-G1, S9-G2, S9-G3, S9-G4 |
| 6: Particle/VFX Render | T-GPU-6.1 -- T-GPU-6.5 | 5 | S9-G5, S9-G6, S9-G7 |
| 7: Integration | T-GPU-7.1 -- T-GPU-7.7 | 7 | S2-G3, S9-G8, S9-G9, S9-G10 |
| **Total** | | **35** | **All 17 gaps** |
