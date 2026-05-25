# CLARIFICATION: GAPSET 8 GPU Compute -- Source Code vs TODO Discrepancies

> **RDC Date**: 2026-05-22

---

## 1. Checkbox Errors (8 tasks implemented but marked [ ])

### T-GPU-1.4 -- Bindless Material Table
- TODO says: `[ ]`, Files: `bindless.rs`, `materials/material_table.rs`
- Reality: `crates/renderer-backend/src/gpu_driven/material_table.rs` (1302 lines) is **fully implemented** with:
  - MaterialTableEntry (80 bytes, repr(C), align(16))
  - add(), update(), remove(), mark_dirty(), any_dirty()
  - Dirty flag tracking (bit 31 = MATERIAL_FLAG_DIRTY)
  - stage(), stage_and_submit() via BufferRegistry
  - WGSL counterpart at `material_table.wgsl`
- **Action**: Change to [x]; correct file paths

### T-GPU-1.5 -- Bindless Texture Table
- TODO says: `[ ]`, Files: `bindless.rs`
- Reality: `crates/renderer-backend/src/gpu_driven/texture_table.rs` (271 lines) is **fully implemented** with:
  - TextureTableEntry (24 bytes, 6 x u32)
  - add(), insert_at(), update(), remove(), clear() with free-list management
  - stage(), stage_and_submit() via BufferRegistry
  - MAX_BINDLESS_TEXTURES = 4096 slots
- **Action**: Change to [x]; correct file paths

### T-GPU-5.1 -- GPU Particle Spawn Compute Shader
- TODO says: `[ ]`, Files: `gpu_particle_spawn.comp.wgsl`, `gpu_particles.py`
- Reality: `particles.wgsl` has `fn particle_spawn()` (line 88-151) with:
  - Atomic pool allocation from alive_count
  - Random position within emitter sphere
  - Random velocity (hemisphere up), lifetime, size initialization
  - `particles.rs` has `create_particle_spawn_pass()` factory
- **Action**: Change to [x]; note actual file is particles.wgsl

### T-GPU-5.2 -- GPU Particle Update Compute Shader
- TODO says: `[ ]`, Files: `gpu_particle_update.comp.wgsl`
- Reality: `particles.wgsl` has `fn particle_update()` (line 157-188) with:
  - Gravity, drag integration
  - Age advance and lifetime culling
  - Age-based size reduction
  - `particles.rs` has `create_particle_update_pass()`
- **Note**: No ping-pong SoA buffer swap; uses single storage buffer read-modify-write
- **Action**: Change to [x]; note architecture difference

### T-GPU-5.3 -- GPU Particle Compact Compute Shader
- TODO says: `[ ]`, Files: `gpu_particle_compact.comp.wgsl`
- Reality: `particles.wgsl` has `fn particle_compact()` (line 235-265) with:
  - Simplified single-pass swap-based compaction
  - Racy but converging atomic decrement
  - `particles.rs` has `create_particle_compact_pass()`
- **Note**: NOT a prefix-sum + scatter implementation as specified. The existing implementation is a simplified alternative that may produce transient visual artifacts (documented in WGSL comments: "one-frame visual glitch at worst").
- **Action**: Change to [~]; mark as partial/alternative implementation

### T-GPU-6.3 -- Trail Rendering
- TODO says: `[ ]`, Files: `trail_renderer.py`
- Reality: `engine/rendering/particles/trail_renderer.py` (815 lines) is **fully implemented** with:
  - TrailBuffer with ring buffer wrapping
  - Catmull-Rom spline interpolation
  - ROUND/FLAT/ARROW cap modes
  - STRETCH/TILE UV modes
- **Action**: Change to [x]

### T-GPU-6.4 -- Deferred Decal System
- TODO says: `[ ]`, Files: `decal_system.py`
- Reality: `engine/rendering/particles/decal_system.py` (946 lines) is **fully implemented** with:
  - DecalInstance, DecalAtlas, DecalRenderer classes
  - Volume projection, per-channel blend modes
  - Atlas packing and occupancy tracking
- **Action**: Change to [x]

### T-GPU-6.5 -- VFX Graph Runtime
- TODO says: `[ ]`, Files: `vfx_graph.py`
- Reality: `engine/rendering/particles/vfx_graph.py` (946 lines) is **fully implemented** with:
  - VFXGraph class with compile() method (line 830)
  - 26 VFX node types (VFXNodeType enum)
  - Event wiring, module composition
- **Action**: Change to [x]

---

## 2. File Path Corrections

| TODO Path | Actual Path |
|-----------|-------------|
| `gpu_driven/bindless.rs` | `gpu_driven/mesh_table.rs`, `material_table.rs`, `texture_table.rs` |
| `materials/material_table.rs` | `gpu_driven/material_table.rs` |
| `gpu_driven/sort.rs` | **Does not exist** |
| `gpu_driven/indirect_draw.rs` | **Only at** `engine/rendering/gpu_driven/indirect_draw.py` |
| `gpu_driven/hzb.rs` | **Does not exist** |
| `gpu_driven/meshlet_culling.rs` | **Does not exist** (Python at `engine/rendering/gpu_driven/meshlet.py`) |
| `gpu_driven/lod_feedback.rs` | **Does not exist** |
| `gpu_driven/fallback.rs` | **Does not exist** |
| `shaders/gpu_driven/*` | **Directory does not exist** (all WGSL is at `crates/renderer-backend/shaders/`) |
| `shaders/common/*` | **Directory does not exist** |
| `shaders/particles/*` | All particle WGSL is in `shaders/particles.wgsl` (single file) |
| `trinity/decorators/gpu.py` for @gpu_driven_mesh | **@gpu_driven_mesh does not exist** in any file |

---

## 3. Architecture Deviations

### Particle render (T-GPU-6.1): compute-based vs vertex+fragment shaders
- Spec: billboard.vert.wgsl + billboard.frag.wgsl with vertex attribute pull
- Reality: `particles.wgsl` `particle_render()` compute shader generates 6 quad vertices into vertex_buffer, then a single indirect draw call renders the full buffer
- Not necessarily better/worse but different architecture

### Particle compact (T-GPU-5.3): swap-based vs prefix-sum
- Spec: prefix-sum + scatter with indirect draw count update
- Reality: single-pass swap compaction with racy atomic decrement
- The existing implementation trades correctness for simplicity. Full prefix-sum would require an additional pass (as noted in the WGSL source comments)

### Particle update (T-GPU-5.2): no ping-pong buffers
- Spec: ping-pong SoA buffers (read from one, write to other, swap each frame)
- Reality: single storage buffer with read-modify-write

No particle sort (T-GPU-5.4): depth-sorted translucent particle rendering is unimplemented. The WGSL Particle struct includes no depth field.

---

## 4. Missing Decorators

### @gpu_driven_mesh (T-GPU-7.5)
The composite decorator that should combine @component + @lod + @render_layer + @gpu_buffer + @streamable does not exist in any `trinity/decorators/` file. No search result found for "gpu_driven_mesh" in the entire codebase.

---

## 5. Count Discrepancies

- Body lists 32 task IDs (T-GPU-1.1 through T-GPU-7.7)
- Summary table at bottom says 35 tasks, 17 gaps
- Actual unique task IDs: 32 (1.1-1.6=6, 2.1-2.4=4, 3.1-3.4=4, 4.1-4.5=5, 5.1-5.4=4, 6.1-6.5=5, 7.1-7.7=7 = 35... wait, 6+4+4+5+4+5+7 = 35. So 35 is correct. The body header says "Total tasks: 32" which is wrong -- it should be 35.
- The body header says 32 but the summary correctly says 35.
