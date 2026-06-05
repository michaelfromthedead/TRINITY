# PHASE 7 ARCHITECTURE: Integration & Decorators

> **Phase**: 7/7 | **Status**: [~] 28% (5 partial, 2 absent)
> **Tasks**: T-GPU-7.1 through T-GPU-7.7 (7 tasks)
> **Gaps**: S2-G3, S9-G8, S9-G9, S9-G10

---

## Files Implemented

| File | Lines | Role |
|------|-------|------|
| `trinity/decorators/rendering.py` | 443 | @render_layer, @shadow_caster, etc. |
| `trinity/decorators/lod_streaming.py` | 292 | @lod, @streamable, @chunk, etc. |
| `trinity/decorators/gpu.py` | 887 | @bind_group decorator |
| `crates/.../src/particles.rs` | 431 | Particle pass registration |
| `crates/.../frame_graph/mod.rs` | N/A | Frame graph core |
| `trinity/decorators/particles_vfx.py` | N/A | @gpu_particle decorator |

## Missing Files

| File | Status |
|------|--------|
| `trinity/decorators/gpu.py` (@gpu_driven_mesh) | [-] Does not exist |
| `crates/.../gpu_driven/lod_feedback.rs` | [-] Does not exist |
| `engine/rendering/gpu_driven/shadow_culling.py` | [-] Does not exist |
| `crates/.../gpu_driven/fallback.rs` | [-] Does not exist |
| `engine/rendering/gpu_driven/culling.py` (pass registration) | [-] No frame graph node for culling |

## Reality by Task

### T-GPU-7.1: @render_layer [~] DECORATOR EXISTS, NO GPU DISPATCH
- `rendering.py` defines `render_layer = make_decorator(...)` with:
  - Valid layers: opaque, masked, translucent, additive, modulate
  - RenderLayerConfig(layer, order)
  - Sets `obj._render_layer`, `_render_layer_name`, `_render_layer_order`
- Missing: layer dispatch (opaque/transparent/shadow culling streams), separate indirect draw buffers per layer, layer ordering

### T-GPU-7.2: @lod to GPU LOD feedback [~] DECORATOR EXISTS, NO GPU LOOP
- `lod_streaming.py` defines `lod = make_decorator(...)` with:
  - `levels`, `distances`, `bias` parameters
  - Validation: distances must be strictly ascending, levels > 0
- Missing: GPUInstance LOD distance packing, GPU LOD selection readback, streaming priority feedback, `lod_feedback.rs`

### T-GPU-7.3: @shadow_caster [~] DECORATOR EXISTS, NO GPU CULLING
- `rendering.py` defines `shadow_caster = make_decorator(...)` with:
  - `mode` (static, dynamic, none), `resolution_scale`, `cascade_bias`
  - Sets `obj._shadow_caster = True`
- Missing: shadow culling stream, second GPU culling pass with shadow frustum, `shadow_culling.py`

### T-GPU-7.4: @bind_group [~] DECORATOR EXISTS, NO BIND GROUP LAYOUT
- `gpu.py` defines `bind_group = make_decorator(...)` with:
  - `index` validation (must be >= 0)
  - Sets `obj._bind_group_index`
- Missing: wgpu bind group layout generation, descriptor layout wiring (group 0 = bindless tables, group 1 = frame constants, group 2 = pass-specific)

### T-GPU-7.5: @gpu_driven_mesh [ - ] NOT IMPLEMENTED
- Composite decorator combining @component + @lod + @render_layer + @gpu_buffer + @streamable
- Does NOT exist anywhere in the codebase (searched gpu.py, rendering.py, lod_streaming.py, particles_vfx.py)

### T-GPU-7.6: Frame graph pass registration [~] PARTICLE ONLY
- `particles.rs` creates IrPass instances for spawn/update/render/compact with proper AccessSet
- Frame graph exists at `crates/renderer-backend/src/frame_graph/`
- Missing: culling compute pass registration as frame graph node, dependency ordering (HZB build -> cull -> draw), particle transparent pass at layer order 6

### T-GPU-7.7: Fallback paths [ - ] NOT IMPLEMENTED
- No `fallback.rs`
- No graceful degradation for: no indirect count, no descriptor indexing, no mesh shaders, HZB unavailable
- No Tier 2 (CPU readback) or Tier 3 (CPU batching) fallback paths

## Decorator Architecture

```
trinity/decorators/rendering.py:
  @render_layer(layer="opaque", order=0)
    → tags target with render_layer_config
    → target._render_layer_name = "opaque"
  
  @shadow_caster(mode="dynamic", resolution_scale=1.0, cascade_bias=0.005)
    → tags target with shadow_caster_config
    → target._shadow_caster = True

  @gi_contributor(importance="high", emissive=True)
    → tags for GI contribution

trinity/decorators/lod_streaming.py:
  @lod(levels=4, distances=[10, 50, 200, 500], bias=0.0)
    → validates distances ascending, levels > 0
    → tags target with lod_config

  @streamable(priority="normal", keep_loaded=False)
    → validates priority in {critical, high, normal, low}

trinity/decorators/gpu.py:
  @gpu_buffer(usage={"storage", "indirect"})
    → resolves wgpu usage flags (auto-adds COPY_DST)
    → computes buffer size from annotations

  @gpu_struct
    → computes WGSL-compatible struct layout

  @bind_group(index=0)
    → tags with bind_group_index

trinity/decorators/particles_vfx.py:
  @gpu_particle(attributes=[...], compute_shader="shader.wgsl")
    → tags with attribute list and shader path

  @vfx_event(trigger="spawn", modules=[...])
    → wires event handlers
```

## Required Integration Points

For full Phase 7 completion:
1. **Render Layer dispatch**: Culling pipeline must filter instances by layer, produce separate indirect draw buffers per layer, and order layers (sky > opaque > masked > transparent > overlay)
2. **LOD feedback loop**: GPU selects LOD per instance → CPU reads back selection → streaming system prioritizes high-LOD meshes → new mesh loaded to MeshTable
3. **Shadow culling**: Second culling pass with directional/spot light shadow frustums, static caching, dynamic per-frame cull
4. **Bind group layout**: Group 0 = bindless tables (texture array, mesh table, material table), Group 1 = frame uniforms, Group 2 = pass-specific uniforms
5. **Frame graph integration**: Culling pipeline as S1 frame graph node, with proper dependency edges between HZB build, culling, and draw passes
