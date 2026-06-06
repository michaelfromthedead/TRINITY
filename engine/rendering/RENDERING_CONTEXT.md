# RENDERING_CONTEXT.md — Rendering Layer

> **Purpose**: Complete implementation reference for the engine/rendering/ layer.  
> Read this file and ONLY this file when implementing rendering systems.

**Implementation Status:**
| Subsystem | Python | Rust | Gapset |
|-----------|--------|------|--------|
| Frame Graph | ✅ | ⚠️ 28% | GAPSET_2 |
| Materials | ✅ | ❌ 6% | GAPSET_4 |
| Lighting | ✅ | ❌ 3% | GAPSET_5 |
| GI/Reflections | ✅ | ❌ 0% | GAPSET_6 |
| Post-Process | ✅ | ⚠️ 29% | GAPSET_7 |
| Ray Tracing | BUILD TARGET | ❌ 9% | GAPSET_9 |

*Python algorithms COMPLETE (35K lines). Rust rendering subsystems BLOCKED on frame graph wiring.*
*See `docs/STATUS.md` for current progress.*

---

## 1. Architecture Summary

The rendering layer transforms scene data into visual frames. It sits between gameplay/simulation (above) and the Render Hardware Interface (below, in engine/platform/rhi/). It is GPU-driven, data-driven, scalable from mobile to ultra, and supports hybrid raster + ray tracing.

**Core Subsystems (6):**
1. **Frame Graph** — Render pass declaration, resource aliasing, automatic barrier insertion, dependency scheduling, async compute scheduling, unused pass culling
2. **GPU-Driven Rendering** — Indirect draw generation, GPU culling (frustum/occlusion/distance), instance batching, meshlet culling, bindless resources, multi-draw indirect
3. **Materials & Shading** — PBR metallic-roughness, material instances, shader variant compilation, material graphs, material domains/blend modes
4. **Lighting & GI** — Direct lights (point/spot/directional/area), shadows (CSM/cube/spot/virtual/contact), GI (probes/DDGI/Lumen/voxel/screen-space/path tracing), reflections (probes/SSR/RT)
5. **Post-Processing** — Tonemapping, color grading, bloom, DOF, motion blur, AO (SSAO/HBAO/GTAO), TAA/TSR, upscaling (DLSS/FSR/XeSS), frame generation
6. **Particles & VFX** — GPU particle system, VFX graph, trails, decals, emitters, particle modules (spawn/update/render)

**Pipeline Models:**
- **Forward** — Single pass, Forward+, Clustered Forward (transparency)
- **Deferred** — G-Buffer → Lighting Pass, Tiled/Clustered Deferred
- **Visibility Buffer** — Primitive ID → Deferred texturing, Nanite integration
- **Hybrid** — Opaque deferred + transparent forward + decals modify G-Buffer

**Frame Graph Pass Types:**
- Graphics Pass (rasterization)
- Compute Pass (dispatch)
- Copy Pass (transfer)
- Ray Tracing Pass

**Rendering Phase in Engine Loop:**
```
SystemPhase.PRE_RENDER (6) — Interpolation, LOD selection, culling prep
SystemPhase.RENDER (7)     — Draw calls, GPU submission, present
```

**Dependency Chain:**
```
Platform (RHI/Window/Display/GPUDevice)
    → Core (Memory/Math/ECS/Task)
        → Resource (Assets: textures, meshes, shaders)
            → Simulation (Transform data from physics)
                → Animation (Skeletal, blend shapes)
                    → RENDERING ← THIS LAYER
```

---

## 2. Decorators

### 2.1 Rendering Decorators (Tier 42 — rendering.py)

#### @gi_contributor
```python
@gi_contributor(
    importance: str = "medium",  # "low" | "medium" | "high" | "critical"
    emissive: bool = False,
)
```
- **Config:** `GIContributorConfig(importance, emissive)`
- **Steps:** TAG(gi_contributor=True), TAG(gi_config), REGISTER(rendering)
- **After-Apply:** `_gi_contributor`, `_gi_importance`, `_gi_emissive`, `_gi_config`

#### @shadow_caster
```python
@shadow_caster(
    mode: str = "dynamic",          # "static" | "dynamic" | "none"
    resolution_scale: float = 1.0,  # > 0
    cascade_bias: float = 0.0,
)
```
- **Config:** `ShadowCasterConfig(mode, resolution_scale, cascade_bias)`
- **Steps:** TAG(shadow_caster=True), TAG(shadow_config), REGISTER(rendering)
- **After-Apply:** `_shadow_caster`, `_shadow_mode`, `_shadow_resolution_scale`, `_shadow_cascade_bias`, `_shadow_config`

#### @reflection_probe
```python
@reflection_probe(
    capture_mode: str = "baked",  # "baked" | "realtime" | "mixed"
    resolution: int = 256,        # > 0
    update_rate: float = 0.0,
)
```
- **Config:** `ReflectionProbeConfig(capture_mode, resolution, update_rate)`
- **Steps:** TAG(reflection_probe=True), TAG(reflection_config), REGISTER(rendering)
- **After-Apply:** `_reflection_probe`, `_reflection_capture_mode`, `_reflection_resolution`, `_reflection_update_rate`, `_reflection_config`

#### @material_domain
```python
@material_domain(
    domain: str,  # REQUIRED: "surface" | "deferred_decal" | "volume" | "post_process" | "ui"
)
```
- **Config:** `MaterialDomainConfig(domain)`
- **Steps:** TAG(material_domain=True), TAG(material_domain_config), REGISTER(rendering)
- **After-Apply:** `_material_domain`, `_material_domain_type`, `_material_domain_config`

#### @material_blend
```python
@material_blend(
    mode: str,  # REQUIRED: "opaque" | "masked" | "translucent" | "additive" | "modulate"
)
```
- **Config:** `MaterialBlendConfig(mode)`
- **Steps:** TAG(material_blend=True), TAG(material_blend_config), REGISTER(rendering)
- **After-Apply:** `_material_blend`, `_material_blend_mode`, `_material_blend_config`

#### @render_layer
```python
@render_layer(
    layer: str,     # REQUIRED, non-empty
    order: int = 0, # rendering order within layer
)
```
- **Config:** `RenderLayerConfig(layer, order)`
- **Steps:** TAG(render_layer=True), TAG(render_layer_config), REGISTER(rendering)
- **After-Apply:** `_render_layer`, `_render_layer_name`, `_render_layer_order`, `_render_layer_config`

### 2.2 GPU Decorators (Tier 40 — gpu.py)

#### @gpu_buffer
```python
@gpu_buffer(
    usage: set = {"storage"},  # subset of {"vertex", "index", "uniform", "storage", "indirect"}
    mapped: bool = False,
)
```
- **Config:** `GpuBufferConfig(usage: frozenset, mapped: bool)`
- **Steps:** TAG(gpu_buffer=True), TAG(gpu_buffer_config), REGISTER(gpu), DESCRIBE()
- **After-Apply:** `_gpu_buffer`, `_gpu_usage`, `_gpu_mapped`, `_gpu_buffer_fields`

#### @gpu_kernel
```python
@gpu_kernel(
    workgroup_size: tuple[int, int, int] = (64, 1, 1),
    backend: str = "wgpu",  # "wgpu" | "cuda" | "metal"
)
```
- **Config:** `GpuKernelConfig(workgroup_size, backend)`
- **Steps:** TAG(gpu_kernel=True), TAG(gpu_kernel_config), REGISTER(gpu)
- **After-Apply:** `_gpu_kernel`, `_workgroup_size`, `_gpu_backend`

#### @gpu_struct
```python
@gpu_struct
```
- **Steps:** TAG(gpu_struct=True), REGISTER(gpu), DESCRIBE()
- **After-Apply:** `_gpu_struct`, `_gpu_struct_size` (computed), `_gpu_struct_alignment=4`

#### @bind_group
```python
@bind_group(index: int = 0)  # >= 0
```
- **Steps:** TAG(bind_group=True), TAG(bind_group_index), REGISTER(gpu)
- **After-Apply:** `_bind_group`, `_bind_group_index`

#### @dispatch
```python
@dispatch(indirect: bool = False)
```
- **Steps:** TAG(dispatch=True), TAG(dispatch_indirect), REGISTER(gpu)
- **After-Apply:** `_dispatch`, `_dispatch_indirect`
- **Applies to:** functions or classes

#### @shader
```python
@shader(
    stage: str = "compute",  # "vertex" | "fragment" | "compute"
    entry: str = "main",
)
```
- **Config:** `ShaderConfig(stage, entry)`
- **Steps:** TAG(shader=True), TAG(shader_config), REGISTER(gpu)
- **After-Apply:** `_shader`, `_shader_stage`, `_shader_entry`

#### @render_pass
```python
@render_pass(
    color_attachments: int = 1,  # >= 1
    depth: bool = True,
    msaa: int = 1,               # 1 | 2 | 4 | 8 | 16
)
```
- **Config:** `RenderPassConfig(color_attachments, depth, msaa)`
- **Steps:** TAG(render_pass=True), TAG(render_pass_config), REGISTER(gpu)
- **After-Apply:** `_render_pass`, `_render_pass_colors`, `_render_pass_depth`, `_render_pass_msaa`

#### @async_compute
```python
@async_compute
```
- **Steps:** TAG(async_compute=True), REGISTER(gpu)
- **After-Apply:** `_async_compute`

### 2.3 LOD & Streaming Decorators (Tier 31 — lod_streaming.py)

#### @lod
```python
@lod(
    levels: int = 4,                    # > 0
    distances: list[float] = None,      # len == levels, strictly ascending, all > 0
    bias: float = 0.0,
)
```
- **Steps:** TAG(lod=True), TAG(lod_levels), TAG(lod_distances), TAG(lod_bias), REGISTER(lod_streaming)
- **After-Apply:** `_lod`, `_lod_levels`, `_lod_distances`, `_lod_bias`

#### @streamable
```python
@streamable(
    priority: str = "normal",  # "critical" | "high" | "normal" | "low"
    keep_loaded: bool = False,
)
```
- **Steps:** TAG(streamable=True), TAG(stream_priority), TAG(stream_keep_loaded), REGISTER(lod_streaming)
- **After-Apply:** `_streamable`, `_stream_priority`, `_stream_keep_loaded`

#### @chunk
```python
@chunk(
    size: tuple[float, float, float],  # REQUIRED, all > 0
    overlap: float = 0.0,              # >= 0
)
```
- **Steps:** TAG(chunk=True), TAG(chunk_size), TAG(chunk_overlap), REGISTER(lod_streaming)
- **After-Apply:** `_chunk`, `_chunk_size`, `_chunk_overlap`

#### @loading_priority
```python
@loading_priority(
    visibility_weight: float = 1.0,        # >= 0
    player_velocity_weight: float = 1.0,   # >= 0
)
```
- **Steps:** TAG(loading_priority=True), TAG weights, REGISTER(lod_streaming)
- **After-Apply:** `_loading_priority`, `_loading_priority_visibility_weight`, `_loading_priority_player_velocity_weight`

#### @unloadable
```python
@unloadable(
    min_age: float = 60.0,    # > 0 seconds
    save_state: bool = True,
)
```
- **Steps:** TAG(unloadable=True), TAG(unloadable_min_age), TAG(unloadable_save_state), REGISTER(lod_streaming)
- **After-Apply:** `_unloadable`, `_unloadable_min_age`, `_unloadable_save_state`

### 2.4 Particles & VFX Decorators (Tier 45 — particles_vfx.py)

#### @particle_emitter
```python
@particle_emitter(
    max_particles: int = 1000,       # > 0
    simulation: str = "auto",        # "cpu" | "gpu" | "auto"
    budget_category: str = None,
)
```
- **Steps:** TAG(particle_emitter=True), TAG(max_particles), TAG(simulation), TAG(budget_category), REGISTER(particles_vfx)
- **After-Apply:** `_particle_emitter`, `_particle_max_particles`, `_particle_simulation`, `_particle_budget_category`

#### @particle_module
```python
@particle_module(
    stage: str,                          # REQUIRED: "spawn" | "update" | "render"
    lod_range: tuple[int, int] = (0, 3), # min_lod <= max_lod
)
```
- **Steps:** TAG(particle_module=True), TAG(stage), TAG(lod_range), REGISTER(particles_vfx)
- **After-Apply:** `_particle_module`, `_particle_module_stage`, `_particle_module_lod_range`

#### @vfx_event
```python
@vfx_event(
    trigger: str,  # REQUIRED: "spawn" | "death" | "collision" | "custom"
)
```
- **Steps:** TAG(vfx_event=True), TAG(vfx_event_trigger), REGISTER(particles_vfx)
- **After-Apply:** `_vfx_event`, `_vfx_event_trigger`

#### @gpu_particle
```python
@gpu_particle(
    attributes: list[str],       # REQUIRED, non-empty
    compute_shader: str = None,
)
```
- **Steps:** TAG(gpu_particle=True), TAG(attributes), TAG(compute_shader), REGISTER(particles_vfx)
- **After-Apply:** `_gpu_particle`, `_gpu_particle_attributes`, `_gpu_particle_compute_shader`

#### @trail
```python
@trail(
    width: float = 0.1,              # > 0
    fade_time: float = 1.0,          # > 0
    texture_mode: str = "stretch",   # "stretch" | "tile"
)
```
- **Steps:** TAG(trail=True), TAG(width), TAG(fade_time), TAG(texture_mode), REGISTER(particles_vfx)
- **After-Apply:** `_trail`, `_trail_width`, `_trail_fade_time`, `_trail_texture_mode`

#### @decal
```python
@decal(
    lifetime: float = None,    # > 0 or None (infinite)
    fade_time: float = 1.0,   # >= 0
    channel: int = 0,          # >= 0
)
```
- **Steps:** TAG(decal=True), TAG(lifetime), TAG(fade_time), TAG(channel), REGISTER(particles_vfx)
- **After-Apply:** `_decal`, `_decal_lifetime`, `_decal_fade_time`, `_decal_channel`

### 2.5 Asset Decorators (Tier 8 — assets.py)

#### @asset
```python
@asset(extensions: tuple[str, ...], loader: Callable = None)
```
- **After-Apply:** `_asset`, `_asset_extensions`, `_asset_loader`
- **Platform usage:** Texture, Mesh, Shader, Material asset types

#### @residency
```python
@residency(priority: str = "normal", min_mip: int = 0)
```
- **After-Apply:** `_residency`, `_residency_priority`, `_residency_min_mip`
- **Platform usage:** GPU memory residency for textures

#### @cook
```python
@cook(platform: str = None, compression: str = "lz4", strip_debug: bool = True)
```
- **After-Apply:** `_cook`, `_cook_platform`, `_cook_compression`, `_cook_strip_debug`
- **Platform usage:** Platform-specific asset cooking

### 2.6 Supporting Decorators (from other modules)

| Decorator | Module | Rendering Role |
|-----------|--------|---------------|
| @system(phase="render") | ecs_core | Register rendering systems |
| @system(phase="pre_render") | ecs_core | LOD selection, culling prep |
| @component | ecs_core | Material, MeshRenderer, Camera, Light components |
| @resource | ecs_core | Frame graph, material system, render pipeline Resources |
| @query | ecs_core | Query visible entities with rendering components |
| @parallel | scheduling | Parallel culling, mesh processing |
| @exclusive | scheduling | Swap chain present (main thread) |
| @job | scheduling | Shader compilation jobs |
| @pooled | memory | Pooled render objects (draw calls, commands) |
| @packed(layout="soa") | memory | SoA for cache-friendly batch processing |
| @aligned(bytes=256) | memory | GPU buffer alignment |
| @budget(category="gpu") | memory | GPU memory budgeting |
| @profile | dev | Per-system CPU timing |
| @gpu_profile | dev | GPU timestamp queries, pipeline stats |
| @trace | dev | EventLog render operations |
| @reloadable | dev | Hot-reload shaders |
| @serializable | data_flow | Material/light config persistence |
| @track_changes | debug_safety | Dirty flags for material/transform/camera |
| @reads / @writes | debug_safety | Rendering system access declarations |

---

## 3. Metaclasses

### AssetMeta (PRIMARY for render assets)
```python
class AssetMeta(EngineMeta):
    _registry: dict[int, type]
    _extension_map: dict[str, type]
```
- **__new__:** Assigns `_asset_id`, `_asset_name`, `_asset_type_code` (8 chars), validates `_asset_extensions` (REQUIRED)
- **Key Methods:**
  - `get_by_id(id)` → type
  - `get_by_name(name)` → type
  - `get_for_extension(ext)` → type (e.g., ".png" → TextureAsset)
  - `get_for_path(path)` → type (extracts extension, looks up)
  - `get_loader(asset_type)` → loader class
  - `get_supported_extensions()` → list[str]
  - `queue_load(asset_cls, path, priority, callback)` → async load
  - `process_queue(max_items=10)` → process N queued loads
  - `watch(asset_cls, path)` → file watcher for hot-reload
  - `check_changes()` → list[(type, path)] of changed assets
- **Rendering usage:** Texture, Mesh, Shader, Material, Animation assets

### ComponentMeta (for render components)
- **Rendering usage:** Material, MeshRenderer, Camera, Light, ParticleEmitter components
- Auto-processes `Annotated[]` fields, installs descriptor chains

### ResourceMeta (for render resources)
- **Rendering usage:** FrameGraph, MaterialSystem, LightingSystem, PostProcessStack as singleton Resources

### SystemMeta (for render systems)
- **Rendering usage:** All @system(phase="render") and @system(phase="pre_render") systems
- `get_phase_order(RENDER)` → topologically sorted render systems
- `get_parallel_groups(RENDER)` → parallelizable render system groups

---

## 4. Descriptors

### TrackedDescriptor (tracking.py) — Material/Transform Dirty Flags
```python
TrackedDescriptor(field_offset: int = 0, use_bitmask: bool = False)
```
- **post_set():** Marks field dirty → Foundation Tracker notified
- **Rendering usage:** Every material parameter, transform, camera, light property uses this
- Camera FOV changes → dirty → projection matrix rebuild
- Material roughness changes → dirty → shader re-bind

### ValidatedDescriptor (validation.py) — Material Parameter Bounds
```python
ValidatedDescriptor(validators: list)
```
- **pre_set():** Runs validators before write
- **Rendering usage:** Validate material colors (0-1 range), light intensity (>0), shadow bias

### RangeDescriptor (validation.py) — Clamped Render Values
```python
RangeDescriptor(min_val, max_val, clamp=True)
```
- **pre_set():** Clamp or reject out-of-range
- **Rendering usage:** FOV (10-170), roughness (0-1), metallic (0-1), near/far planes

### ObservableDescriptor (observable.py) — Render State Subscriptions
```python
ObservableDescriptor()
```
- **post_set():** Notifies subscribers
- **Rendering usage:** Material parameter change → re-compile shader variant; Light change → re-sort light list

### ProfiledDescriptor — Render Access Timing
- **Rendering usage:** Track how often transform/material fields are accessed per frame → optimization hints

---

## 5. Foundation Integration Points

### Foundation Registry → Rendering Discovery
```
registry.subclasses(GPUDevice)         # Discover GPU backends (Vulkan/D3D12/Metal)
registry.instances(GPUDevice)          # Get active GPU device
registry.get_all("rendering")         # All @system(phase="render") systems
AssetMeta.get_for_extension(".png")   # Texture asset loader
AssetMeta.get_for_extension(".glb")   # Mesh asset loader
```

### Foundation Tracker → Rendering Dirty Flags
```
material.roughness = 0.8
    → TrackedDescriptor.post_set()
    → Tracker.mark_dirty(material, "roughness")
    → Per-frame: Tracker.all_dirty()
    → Rendering system collects dirty materials
    → Re-binds shader parameters for dirty materials
```

### Foundation EventLog → Render Operation Audit
```
@system(phase="render")
@trace
class RenderSystem(System):
    def execute(self, dt):
        # EventLog.record("RenderSystem.execute", tick=N)
        for entity in query(Transform, MeshRenderer):
            self.draw(entity)
            # EventLog.record("draw_entity", root_cause="RenderSystem.execute")
```

### Foundation Mirror → Render Introspection
```python
m = mirror(gpu_device)
m.get("memory_allocated")            # GPU memory usage
m.get("supports_raytracing")         # Feature query

m = mirror(material_instance)
m.get("base_color")                  # Read parameter
m.set("metallic", 0.8)              # Live edit → triggers recompile
m.to_dict()                          # Export all parameters
```

### Foundation Bridge → ShellLang Render Debugging
```
QUERY MeshRenderer WHERE visible == True
QUERY Camera WHERE active == True
MUTATE entity=42 COMPONENT=Material SET roughness=0.2
QUERY Light WHERE type == "directional"
```

### Foundation ContentStore → Render Asset Deduplication
```
content_store.store(texture_data)    # Hash-based, same texture stored once
session.save()                       # Assets referenced, not duplicated
```

### Foundation Capabilities → GPU Mod Sandboxing
```
Capabilities.has(mod, "gpu_access")  # Can mod use GPU?
Capabilities.has(mod, "shader_load") # Can mod load custom shaders?
```

---

## 6. Architecture Spec Details

### 6.1 Frame Graph

**Resource Management:**
- Transient resources — allocated per-frame, aliased across passes
- History resources — persisted across frames (TAA history, GI accumulators)
- External resources — swap chain backbuffer, imported textures
- Automatic barrier insertion between passes

**Pass Scheduling:**
```
Declare passes → Build dependency graph → Cull unused passes
    → Schedule async compute → Insert barriers → Execute
```

**Frame Graph API (conceptual):**
```python
fg = FrameGraph()
gbuffer = fg.add_pass("GBuffer", type="graphics")
gbuffer.write(albedo_rt, normal_rt, depth_rt)

lighting = fg.add_pass("Lighting", type="compute")
lighting.read(albedo_rt, normal_rt, depth_rt)
lighting.write(hdr_target)

postprocess = fg.add_pass("PostProcess", type="graphics")
postprocess.read(hdr_target)
postprocess.write(backbuffer)

fg.compile()  # Dependency analysis, barrier insertion, resource aliasing
fg.execute()  # Run all passes in order
```

### 6.2 GPU-Driven Rendering Pipeline

**Culling Pipeline (GPU):**
```
All Instances → Frustum Cull (GPU) → Occlusion Cull (HZB) → Distance Cull
    → Meshlet Cull (per-cluster) → Triangle Cull (backface, small)
        → Indirect Draw Buffer → Multi-Draw Indirect
```

**Visibility Buffer Pipeline (Nanite-style):**
```
1. GPU generates visibility buffer (triangle ID + instance ID + depth)
2. Material sorting pass (group pixels by material)
3. Deferred shading pass (fetch vertex data, shade per-pixel)
```

**Instancing & Batching:**
- Same mesh + same material → GPU instanced draw
- Different meshes → Multi-draw indirect
- Bindless resources → no per-draw binding changes

### 6.3 Materials & Shading

**PBR Parameters (Metallic-Roughness):**
| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| base_color | Vec4 | [0,1] per channel | (1,1,1,1) |
| metallic | float | [0,1] | 0.0 |
| roughness | float | [0,1] | 0.5 |
| normal | Vec3 | [-1,1] | (0,0,1) |
| ao | float | [0,1] | 1.0 |
| emissive | Vec3 | [0,∞) | (0,0,0) |

**Advanced Shading Models:**
- Subsurface scattering (SSS)
- Clear coat
- Anisotropy
- Sheen
- Iridescence
- Transmission

**Material System:**
```
Material Template (defines parameters, shader)
    → Material Instance (overrides parameters)
        → Material Function (reusable shader snippets)
            → Material Layer (composable material stacking)
```

**Shader Variant System:**
```
Base Shader Source → Permutation Keys (defines/toggles)
    → Compiled Variant → PSO Cache
    → Hot-reload via @reloadable + AssetMeta.check_changes()
```

**Material Domains:**
| Domain | Pipeline | Usage |
|--------|----------|-------|
| surface | Deferred/Forward | Standard 3D surfaces |
| deferred_decal | G-Buffer modify | Bullet holes, blood, graffiti |
| volume | Compute | Fog, clouds, volumetrics |
| post_process | Post-process | Screen-space effects |
| ui | Forward/2D | UI elements |

**Blend Modes:**
| Mode | Usage |
|------|-------|
| opaque | Solid geometry (default) |
| masked | Alpha-tested (foliage, fences) |
| translucent | Glass, water, particles |
| additive | Fire, glow, energy |
| modulate | Multiply blend |

### 6.4 Lighting

**Light Types:**
| Type | Shadow | Params |
|------|--------|--------|
| Directional | CSM (cascaded) | direction, color, intensity |
| Point | Cube shadow map | position, radius, color, intensity |
| Spot | Spot shadow map | position, direction, inner/outer angle, radius |
| Rect Area | — | position, size, color, intensity |
| Disk Area | — | position, radius, color, intensity |
| IES Profile | Per-light-type | IES data, light type |
| Sky Light | — | cubemap, intensity |

**Light Culling:**
```
Clustered Light Culling:
    Screen → 3D froxels (X × Y × Z)
    Each froxel stores light list
    Per-pixel: fetch froxel → iterate lights
```

**Shadow Systems:**
| Shadow Type | Technique | Use Case |
|-------------|-----------|----------|
| Cascaded (CSM) | Multi-split frustum | Directional (sun) |
| Cube Shadow | 6-face cubemap | Point lights |
| Spot Shadow | Single frustum | Spot lights |
| Contact | Screen-space | Fine detail |
| Virtual Shadow Maps | Page pool, clipmap | All (scalable) |

**Shadow Filtering:**
- PCF (Percentage-Closer Filtering)
- PCSS (Percentage-Closer Soft Shadows)
- VSM (Variance Shadow Maps)

**Global Illumination Techniques:**
| Technique | Type | Quality | Cost |
|-----------|------|---------|------|
| Baked Lightmaps | Static | High | Offline |
| Light Probes (SH) | Static + dynamic | Medium | Low |
| DDGI | Dynamic | High | Medium-High |
| Voxel GI | Dynamic | Medium | Medium |
| Screen-Space GI | Dynamic | Low | Low |
| Lumen | Dynamic | Very High | High |
| Path Tracing | Reference | Perfect | Very High |

**DDGI (Dynamic Diffuse GI):**
```
Probe grid → Ray trace per probe → Update irradiance/visibility
    → Trilinear interpolation at shading point
```

**Lumen:**
```
Mesh cards (simplified scene) → Screen probes → Radiance cache
    → Software ray tracing → Final gather → Temporal accumulation
```

### 6.5 Reflections

**Techniques (increasing quality/cost):**
1. Reflection Probes (baked/realtime cubemaps, parallax corrected)
2. Screen-Space Reflections (SSR — ray march in screen space, HZB acceleration)
3. Ray-Traced Reflections (per-pixel rays, half-res upscale, denoised)

### 6.6 Post-Processing

**Post-Process Stack (ordered):**
```
HDR Scene → Exposure → Bloom → Depth of Field → Motion Blur
    → AO → Tone Mapping → Color Grading → TAA → Upscaling → Output
```

**Anti-Aliasing:**
| Method | Quality | Cost | Temporal |
|--------|---------|------|----------|
| MSAA | Good geometry | High | No |
| FXAA | Low | Low | No |
| SMAA | Medium | Medium | No |
| TAA | High | Medium | Yes |

**Upscaling / Super Resolution:**
| Method | Type | Vendor |
|--------|------|--------|
| FSR 1.0 | Spatial | AMD |
| FSR 2.0+ | Temporal | AMD |
| DLSS | Temporal + AI | NVIDIA |
| XeSS | Temporal + AI | Intel |
| Frame Gen | Interpolation | NVIDIA/AMD |

**AO Methods:**
- SSAO (Screen-Space Ambient Occlusion)
- HBAO (Horizon-Based AO)
- GTAO (Ground-Truth AO)

### 6.7 Geometry Systems

**Mesh Representation:**
- Position, Normal, Tangent, UV0, UV1, Color, BoneWeights, BoneIndices
- Vertex layouts: interleaved, split streams, compressed
- Cache optimization: vertex cache ordering, overdraw ordering

**Meshlet / Cluster System:**
- ~64 vertices, ~124 triangles per meshlet
- Bounding sphere + normal cone for culling
- Enables mesh shader pipeline

**LOD Systems:**
| System | Approach | Transition |
|--------|----------|------------|
| Discrete LOD | Pre-authored meshes at distance thresholds | Pop/cross-fade |
| Continuous LOD | Geomorphing between levels | Smooth |
| Hierarchical LOD | Pre-merged cluster groups | Smooth |
| Nanite | DAG cut at runtime, software raster | Seamless |

**Compression:**
- Quantization (position, normal, UV)
- Delta encoding
- Meshopt compression

### 6.8 Texturing

**Virtual Texturing:**
```
Page Table → Indirection Texture → Physical Page Pool
Feedback Buffer (GPU) → CPU reads → Stream pages → Upload
Eviction: LRU when pool full
```

**Block Compression:**
| Format | Channels | Use |
|--------|----------|-----|
| BC1 | RGB + 1-bit A | Albedo (opaque) |
| BC3 | RGBA | Albedo (transparent) |
| BC4 | R | Roughness, AO |
| BC5 | RG | Normal maps |
| BC6H | RGB HDR | HDR environment maps |
| BC7 | RGBA | High-quality albedo |
| ASTC | Configurable | Mobile |
| ETC2 | RGB/RGBA | Mobile |

### 6.9 Atmospheric & Volumetric

**Sky:** Preetham/Hosek-Wilkie/Bruneton models, sun disk, moon, stars
**Fog:** Distance fog, height fog, local volumes, volumetric fog (froxel grid)
**Clouds:** Shape noise + detail noise, Beer-Powder lighting, multi-scattering
**Volumetric Lighting:** Sun shafts, froxel-based scatter

### 6.10 Water & Terrain

**Water:** Gerstner waves, FFT spectrum, multi-cascade, Fresnel, refraction, caustics, foam
**Terrain:** Clipmap/CDLOD/quadtree LOD, splat maps, height blending, virtual texture terrain

### 6.11 Ray Tracing

**Architecture:**
```
BLAS (per-mesh) → TLAS (scene) → Shader Binding Table
    → Ray Generation → Traversal → Hit/Miss → Shading
```

**BVH Management:** Build vs refit, dynamic updates, compaction
**Applications:** RT shadows, RT reflections, RT GI, RT AO
**Denoising:** Spatial filter, temporal accumulation, neural denoisers (NRD/OptiX)

---

## 7. Decorator Stacks

### Builtin Streaming Stacks (streaming.py)

#### @streaming_chunk — World Streaming
```python
from trinity.decorators.builtin_stacks.streaming import streaming_chunk

@streaming_chunk(chunk_size=(100, 100, 100), overlap=10, min_age=60.0)
class TerrainChunk(Component): ...
```
**Combines:** @chunk + @streamable + @loading_priority + @unloadable + @serializable

#### @lod_scalable — LOD + Streaming
```python
from trinity.decorators.builtin_stacks.streaming import lod_scalable

@lod_scalable(levels=4, distances=[10, 50, 200, 1000])
class ScalableMesh(Component): ...
```
**Combines:** @lod + @streamable + @residency

### Builtin Composite Stacks (composite.py)

#### @open_world_entity — Full Open World
```python
from trinity.decorators.builtin_stacks.composite import open_world_entity

@open_world_entity(pool_size=10000, chunk_size=(100, 100, 100))
class WorldObject(Component): ...
```
**Combines:** production_component + streaming_chunk + lod_scalable + versioned_saveable

### Proposed Rendering Stacks

#### render_component — Standard Render Data
```python
# Proposed:
@component
@packed(layout="soa")
@pooled(initial_size=4096)
@budget(category="gpu")
@track_changes
class RenderComponentBase(Component): ...
```
**Combines:** @component + @packed + @pooled + @budget + @track_changes

#### gpu_driven_mesh — GPU-Driven Meshlet Mesh
```python
# Proposed:
@component
@packed(layout="soa")
@pooled(initial_size=8192)
@lod(levels=4, distances=[10, 50, 200, 1000])
@streamable(priority="normal")
@budget(category="gpu")
class GPUDrivenMeshBase(Component): ...
```
**Combines:** @component + @packed + @pooled + @lod + @streamable + @budget

#### material_instance — PBR Material
```python
# Proposed:
@component
@material_domain(domain="surface")
@material_blend(mode="opaque")
@serializable(format="binary")
@track_changes
class MaterialBase(Component):
    base_color: Annotated[Vec4, TrackedDescriptor(), RangeDescriptor(0, 1)]
    metallic: Annotated[float, TrackedDescriptor(), RangeDescriptor(0, 1)]
    roughness: Annotated[float, TrackedDescriptor(), RangeDescriptor(0, 1)]
```
**Combines:** @component + @material_domain + @material_blend + @serializable + @track_changes

#### particle_system — Full Particle System
```python
# Proposed:
@component
@particle_emitter(max_particles=10000, simulation="gpu")
@gpu_particle(attributes=["position", "velocity", "color", "life"])
@budget(category="particles")
@lod(levels=3)
class ParticleBase(Component): ...
```
**Combines:** @component + @particle_emitter + @gpu_particle + @budget + @lod

---

## 8. TODO Checklist (from GAME_ENGINE_INTEGRATION_TODO.md S5)

### 5.1 Frame Graph
- [ ] Implement frame graph (render pass declaration, resource aliasing)
- [ ] Implement automatic resource barrier insertion
- [ ] Implement render pass dependency analysis and scheduling
- [ ] Wire rendering Systems (`@system(phase="render")`) as frame graph nodes
- [ ] Implement async compute pass scheduling
- [ ] Implement transient resource allocation and aliasing
- [ ] Implement unused pass culling

### 5.2 GPU-Driven Rendering
- [ ] Implement indirect draw call generation
- [ ] Implement GPU culling (frustum, occlusion via HZB, distance)
- [ ] Implement instance batching and merging
- [ ] Implement meshlet/cluster culling
- [ ] Implement triangle culling (backface, small feature)
- [ ] Implement multi-draw indirect
- [ ] Implement bindless resource binding
- [ ] Wire @render_layer decorator → render layer assignment
- [ ] Wire @shadow_caster decorator → shadow pass inclusion
- [ ] Wire @lod decorator → LOD distance selection
- [ ] Wire @gpu_buffer decorator → GPU buffer allocation
- [ ] Wire @gpu_struct decorator → GPU struct layout

### 5.3 Materials & Shading
- [ ] Implement PBR material model (metallic-roughness)
- [ ] Implement material instance system (parameter overrides)
- [ ] Implement shader variant compilation and PSO caching
- [ ] Implement material graph / node-based authoring
- [ ] Implement advanced shading models (SSS, clear coat, anisotropy, sheen, transmission)
- [ ] Wire @material_domain decorator → material type classification
- [ ] Wire @material_blend decorator → blend mode selection
- [ ] Wire @shader decorator → shader stage registration
- [ ] Wire @reloadable decorator → shader hot-reload
- [ ] Wire ValidatedDescriptor → material parameter validation
- [ ] Wire TrackedDescriptor → material dirty flag → re-bind

### 5.4 Lighting & GI
- [ ] Implement direct lighting (point, spot, directional, rect area, disk area)
- [ ] Implement IES light profiles
- [ ] Implement shadow mapping (CSM, cube, spot)
- [ ] Implement shadow filtering (PCF, PCSS, VSM)
- [ ] Implement virtual shadow maps (page pool, clipmap)
- [ ] Implement contact shadows
- [ ] Implement clustered light culling (3D froxels)
- [ ] Implement global illumination (choose: Lumen-style, DDGI, or probe-based)
- [ ] Wire @gi_contributor decorator → GI participation flags
- [ ] Wire @reflection_probe decorator → reflection probe placement
- [ ] Wire @shadow_caster decorator → shadow caster config

### 5.5 Post-Processing
- [ ] Implement post-process stack (ordered pass chain)
- [ ] Implement exposure / auto-exposure
- [ ] Implement bloom
- [ ] Implement tone mapping (ACES, AgX)
- [ ] Implement color grading (LUT-based)
- [ ] Implement depth of field
- [ ] Implement motion blur
- [ ] Implement AO (SSAO / HBAO / GTAO)
- [ ] Implement TAA (temporal anti-aliasing)
- [ ] Implement TSR / upscaling integration (DLSS, FSR, XeSS)
- [ ] Wire post-process Systems to frame graph

### 5.6 Particles & VFX
- [ ] Implement GPU particle system
- [ ] Implement particle emitter lifecycle (spawn, update, render)
- [ ] Implement particle modules (velocity, color, size, noise, collision)
- [ ] Implement VFX graph (node-based effect authoring)
- [ ] Implement trail renderer
- [ ] Implement decal system
- [ ] Wire @particle_emitter decorator → emitter config
- [ ] Wire @particle_module decorator → module stage registration
- [ ] Wire @gpu_particle decorator → GPU particle attributes
- [ ] Wire @vfx_event decorator → event trigger registration
- [ ] Wire @trail decorator → trail config
- [ ] Wire @decal decorator → decal config
- [ ] Wire @budget(category="particles") → particle memory budget

### 5.7 Additional Rendering Systems
- [ ] Implement atmosphere / sky rendering
- [ ] Implement volumetric fog (froxel-based)
- [ ] Implement cloud rendering
- [ ] Implement water rendering (Gerstner/FFT)
- [ ] Implement terrain rendering (clipmap/CDLOD LOD)
- [ ] Implement foliage rendering (GPU instancing, wind)
- [ ] Implement virtual texturing (page table, streaming, feedback)
- [ ] Implement ray tracing pipeline (BLAS/TLAS, RT shadows/reflections/GI)
- [ ] Implement denoising (spatial + temporal)

---

## 9. Directory Structure

```
engine/rendering/
├── __init__.py                       # Public API exports
├── framegraph/
│   ├── __init__.py
│   ├── frame_graph.py                # Pass declaration, dependency graph, compilation
│   ├── pass_node.py                  # Graphics/Compute/Copy/RT pass types
│   ├── resource_manager.py           # Transient/history/external resource allocation
│   ├── barrier_manager.py            # Automatic barrier insertion
│   └── async_scheduler.py            # Async compute scheduling
├── gpu_driven/
│   ├── __init__.py
│   ├── culling.py                    # GPU frustum/occlusion/distance culling
│   ├── meshlet.py                    # Meshlet/cluster system, cone culling
│   ├── indirect_draw.py              # Indirect draw buffer generation
│   ├── instancing.py                 # Instance batching, multi-draw indirect
│   ├── visibility_buffer.py          # Nanite-style visibility buffer
│   └── bindless.py                   # Bindless resource management
├── materials/
│   ├── __init__.py
│   ├── material_system.py            # Material template + instance management
│   ├── pbr_model.py                  # PBR metallic-roughness model
│   ├── shader_compiler.py            # Shader variant compilation, PSO cache
│   ├── material_graph.py             # Node-based material authoring
│   ├── material_functions.py         # Reusable shader snippets
│   └── advanced_models.py            # SSS, clear coat, anisotropy, sheen, transmission
├── lighting/
│   ├── __init__.py
│   ├── light_types.py                # Directional, point, spot, area, IES, sky
│   ├── light_culling.py              # Clustered light culling (3D froxels)
│   ├── shadows.py                    # CSM, cube, spot shadow mapping
│   ├── shadow_filtering.py           # PCF, PCSS, VSM
│   ├── virtual_shadow_maps.py        # Page pool, clipmap, per-page culling
│   ├── gi_probes.py                  # Light probes (SH), baked lightmaps
│   ├── gi_ddgi.py                    # DDGI (probe grid, ray-traced updates)
│   ├── gi_lumen.py                   # Lumen (mesh cards, screen probes, radiance cache)
│   ├── reflections.py                # Reflection probes, SSR, RT reflections
│   └── contact_shadows.py            # Screen-space contact shadows
├── postprocess/
│   ├── __init__.py
│   ├── postprocess_stack.py          # Ordered pass chain, frame graph integration
│   ├── exposure.py                   # Auto-exposure, histogram
│   ├── bloom.py                      # Bloom (downsample, blur, upsample)
│   ├── tonemapping.py                # ACES, AgX, Reinhard
│   ├── color_grading.py              # LUT-based color grading
│   ├── dof.py                        # Depth of field (bokeh, CoC)
│   ├── motion_blur.py                # Per-object and camera motion blur
│   ├── ambient_occlusion.py          # SSAO, HBAO, GTAO
│   ├── antialiasing.py               # TAA, FXAA, SMAA
│   └── upscaling.py                  # FSR, DLSS, XeSS integration
├── particles/
│   ├── __init__.py
│   ├── particle_system.py            # Emitter lifecycle, CPU/GPU dispatch
│   ├── gpu_particles.py              # GPU compute particle simulation
│   ├── particle_modules.py           # Spawn/update/render modules
│   ├── vfx_graph.py                  # Node-based VFX authoring
│   ├── trail_renderer.py             # Trail rendering
│   └── decal_system.py               # Projected decals
├── atmosphere/
│   ├── __init__.py
│   ├── sky.py                        # Atmospheric scattering models
│   ├── volumetric_fog.py             # Froxel-based volumetric fog
│   ├── clouds.py                     # Volumetric clouds
│   └── volumetric_lighting.py        # Sun shafts, light scattering
├── terrain/
│   ├── __init__.py
│   ├── terrain_lod.py                # Clipmap/CDLOD/quadtree LOD
│   ├── terrain_material.py           # Splat maps, height blend, virtual texture
│   └── foliage.py                    # GPU instanced foliage, wind, interaction
├── water/
│   ├── __init__.py
│   ├── water_simulation.py           # Gerstner/FFT waves
│   ├── water_shading.py              # Fresnel, refraction, caustics
│   └── foam.py                       # Wave/shore/simulated foam
├── raytracing/
│   ├── __init__.py
│   ├── acceleration_structure.py     # BLAS/TLAS build, refit, compact
│   ├── rt_shadows.py                 # Ray-traced shadows
│   ├── rt_reflections.py             # Ray-traced reflections
│   ├── rt_gi.py                      # Ray-traced GI
│   └── denoiser.py                   # Spatial/temporal/neural denoising
├── texturing/
│   ├── __init__.py
│   ├── virtual_texturing.py          # Page table, feedback, streaming
│   └── texture_compression.py        # BC1-7, ASTC, ETC2 handling
└── geometry/
    ├── __init__.py
    ├── mesh_representation.py         # Vertex formats, layouts
    ├── mesh_compression.py            # Quantization, meshopt
    └── lod_system.py                  # Discrete/continuous/hierarchical/Nanite LOD
```

---

## 10. Canonical Usage Examples

### Example 1: PBR Material Component
```python
from typing import Annotated
from trinity.base import Component
from trinity.types import Vec3, Vec4
from trinity.descriptors.tracking import TrackedDescriptor
from trinity.descriptors.validation import RangeDescriptor, ValidatedDescriptor
from trinity.decorators.rendering import material_domain, material_blend

@component
@material_domain(domain="surface")
@material_blend(mode="opaque")
@serializable(format="binary")
@track_changes
class PBRMaterial(Component):
    """Standard PBR material with metallic-roughness workflow."""
    base_color: Annotated[Vec4,
        TrackedDescriptor(),
        RangeDescriptor(min_val=0.0, max_val=1.0, clamp=True),
    ] = Vec4(1, 1, 1, 1)

    metallic: Annotated[float,
        TrackedDescriptor(),
        RangeDescriptor(min_val=0.0, max_val=1.0, clamp=True),
    ] = 0.0

    roughness: Annotated[float,
        TrackedDescriptor(),
        RangeDescriptor(min_val=0.0, max_val=1.0, clamp=True),
    ] = 0.5

    normal_scale: Annotated[float,
        TrackedDescriptor(),
        RangeDescriptor(min_val=0.0, max_val=2.0, clamp=True),
    ] = 1.0

    ao: Annotated[float,
        TrackedDescriptor(),
        RangeDescriptor(min_val=0.0, max_val=1.0, clamp=True),
    ] = 1.0

    emissive: Annotated[Vec3,
        TrackedDescriptor(),
    ] = Vec3(0, 0, 0)
```

### Example 2: Light Component with Shadow Config
```python
@component
@shadow_caster(mode="dynamic", resolution_scale=1.0, cascade_bias=0.001)
@gi_contributor(importance="high", emissive=False)
@track_changes
class DirectionalLight(Component):
    """Sun/moon directional light with CSM shadows."""
    direction: Annotated[Vec3, TrackedDescriptor()]
    color: Annotated[Vec3, TrackedDescriptor(), RangeDescriptor(0, 1)]
    intensity: Annotated[float, TrackedDescriptor(), RangeDescriptor(0, 100)]
    cascade_count: Annotated[int, TrackedDescriptor(), RangeDescriptor(1, 4)] = 4
```

### Example 3: GPU-Driven Render System
```python
@system(phase="render")
@parallel(chunk_size=512)
@profile(name="GPUCullingSystem")
@gpu_profile(category="culling")
@reads(Transform, MeshRenderer)
class GPUCullingSystem(System):
    """GPU frustum + occlusion culling, generates indirect draw buffer."""

    def execute(self, dt: float):
        # 1. Upload instance data (transforms + bounds) to GPU buffer
        # 2. Dispatch frustum cull compute shader
        # 3. Dispatch occlusion cull (read HZB from previous frame)
        # 4. Compact surviving instances into indirect draw buffer
        # 5. Output: IndirectDrawBuffer ready for multi-draw indirect
        pass
```

### Example 4: Frame Graph Setup
```python
@resource
class RenderPipeline(Resource):
    """Central render pipeline — builds and executes frame graph each frame."""
    _resource_priority = 50
    _resource_dependencies = (GPUDevice, Window)

    def build_frame_graph(self, scene, camera):
        fg = FrameGraph()

        # G-Buffer pass
        gbuffer = fg.add_pass("GBuffer", type="graphics")
        gbuffer.set_render_pass(color_attachments=4, depth=True)
        # writes: albedo, normal, metallic_roughness, emissive, depth

        # Shadow pass
        shadows = fg.add_pass("Shadows", type="graphics")
        # writes: shadow_atlas

        # Lighting pass
        lighting = fg.add_pass("Lighting", type="compute")
        # reads: gbuffer outputs + shadow_atlas
        # writes: hdr_target

        # Post-process
        postprocess = fg.add_pass("PostProcess", type="graphics")
        # reads: hdr_target
        # writes: backbuffer

        fg.compile()
        return fg
```

### Example 5: GPU Particle System
```python
@component
@particle_emitter(max_particles=50000, simulation="gpu")
@gpu_particle(
    attributes=["position", "velocity", "color", "life", "size"],
    compute_shader="particles/fire.comp",
)
@lod(levels=3, distances=[50, 200, 500])
@budget(category="particles")
class FireEmitter(Component):
    """GPU-simulated fire particle system with LOD."""
    spawn_rate: Annotated[float, TrackedDescriptor(), RangeDescriptor(0, 10000)] = 500.0
    lifetime: Annotated[float, TrackedDescriptor(), RangeDescriptor(0.1, 10.0)] = 2.0
    initial_velocity: Annotated[Vec3, TrackedDescriptor()] = Vec3(0, 5, 0)
    color_start: Annotated[Vec4, TrackedDescriptor()] = Vec4(1, 0.8, 0, 1)
    color_end: Annotated[Vec4, TrackedDescriptor()] = Vec4(0.5, 0, 0, 0)
```

### Example 6: Reflection Probe
```python
@component
@reflection_probe(capture_mode="realtime", resolution=512, update_rate=30.0)
class RealtimeReflectionProbe(Component):
    """Realtime cubemap reflection probe updated at 30 Hz."""
    position: Annotated[Vec3, TrackedDescriptor()]
    radius: Annotated[float, TrackedDescriptor(), RangeDescriptor(1, 10000)] = 100.0
    blend_distance: Annotated[float, TrackedDescriptor(), RangeDescriptor(0, 100)] = 10.0
```

---

## 11. Integration Patterns

### Pattern 1: Material Dirty Flag → Shader Rebind
```python
# Material parameter changes flow:
material.roughness = 0.8
# 1. TrackedDescriptor.post_set() fires
# 2. Tracker.mark_dirty(material, "roughness")
# 3. Per-frame render system:
#    dirty_materials = Tracker.all_dirty(type=PBRMaterial)
#    for mat in dirty_materials:
#        upload_material_params(mat)  # Re-upload to GPU constant buffer
#    Tracker.clear_dirty(type=PBRMaterial)
```

### Pattern 2: Frame Graph → Barrier Management
```python
# Automatic barrier insertion:
# Pass A writes texture as RTV (render target)
# Pass B reads same texture as SRV (shader resource)
# Frame graph automatically inserts:
#   Barrier(texture, RENDER_TARGET → SHADER_RESOURCE) between A and B
```

### Pattern 3: Asset Hot-Reload → Shader Recompile
```python
# Shader file changes:
# 1. FileWatcher detects change → AssetMeta.check_changes()
# 2. Returns [(ShaderAsset, "shaders/pbr.hlsl")]
# 3. Shader compiler re-compiles variants
# 4. PSO cache invalidated for affected pipelines
# 5. @reloadable preserves material instances, only rebinds PSOs
```

### Pattern 4: GPU Culling → Indirect Draw
```python
# Per-frame GPU-driven pipeline:
# 1. Upload all instance transforms + bounds to GPU buffer
# 2. Dispatch frustum cull kernel:
#    @gpu_kernel(workgroup_size=(64, 1, 1))
#    → Writes surviving instance IDs to output buffer
# 3. Dispatch occlusion cull kernel:
#    → Reads HZB (hierarchical-Z) from previous frame
#    → Further filters instances
# 4. Generate indirect draw args:
#    IndirectDrawArgs { vertex_count, instance_count, first_vertex, first_instance }
# 5. Execute multi-draw indirect: one draw call renders thousands of instances
```

### Pattern 5: LOD Selection → Streaming
```python
# LOD + streaming integration:
# 1. Camera position known
# 2. For each @lod entity: compute distance → select LOD level
# 3. For each @streamable entity:
#    If LOD needs higher detail → bump stream priority
#    If LOD at lowest → allow unload (@unloadable)
# 4. @loading_priority weights by visibility + player velocity
# 5. @residency.min_mip ensures minimum quality always loaded
```

### Pattern 6: Foundation Bridge → Live Render Debug
```python
# ShellLang debugging:
# > QUERY MeshRenderer WHERE visible == True AND lod_level > 2
#   → "Found 342 entities at LOD 3+"
# > QUERY Light WHERE type == "point" AND intensity > 10
#   → Shows all bright point lights
# > MUTATE entity=42 COMPONENT=PBRMaterial SET roughness=0.0
#   → Live material edit
# > QUERY ParticleEmitter WHERE particle_count > 10000
#   → Find expensive particle systems
```

---

## 12. Quick Reference Tables

### Rendering Decorators Summary
| Decorator | Module | Tier | Key Params | Purpose |
|-----------|--------|------|------------|---------|
| @gi_contributor | rendering | 42 | importance, emissive | GI participation |
| @shadow_caster | rendering | 42 | mode, resolution_scale, cascade_bias | Shadow casting |
| @reflection_probe | rendering | 42 | capture_mode, resolution, update_rate | Reflection probes |
| @material_domain | rendering | 42 | domain | Material pipeline type |
| @material_blend | rendering | 42 | mode | Blend mode |
| @render_layer | rendering | 42 | layer, order | Render ordering |
| @gpu_buffer | gpu | 40 | usage, mapped | GPU buffer allocation |
| @gpu_kernel | gpu | 40 | workgroup_size, backend | Compute kernel |
| @gpu_struct | gpu | 40 | — | GPU struct layout |
| @bind_group | gpu | 40 | index | Shader binding |
| @dispatch | gpu | 40 | indirect | Compute dispatch |
| @shader | gpu | 40 | stage, entry | Shader stage |
| @render_pass | gpu | 40 | color_attachments, depth, msaa | Render pass config |
| @async_compute | gpu | 40 | — | Async compute |
| @lod | lod_streaming | 31 | levels, distances, bias | Level of detail |
| @streamable | lod_streaming | 31 | priority, keep_loaded | Asset streaming |
| @chunk | lod_streaming | 31 | size, overlap | World chunks |
| @loading_priority | lod_streaming | 31 | visibility_weight, velocity_weight | Smart loading |
| @unloadable | lod_streaming | 31 | min_age, save_state | Unload policy |
| @particle_emitter | particles_vfx | 45 | max_particles, simulation | Particle emitter |
| @particle_module | particles_vfx | 45 | stage, lod_range | Particle module |
| @vfx_event | particles_vfx | 45 | trigger | VFX event trigger |
| @gpu_particle | particles_vfx | 45 | attributes, compute_shader | GPU particles |
| @trail | particles_vfx | 45 | width, fade_time, texture_mode | Trail renderer |
| @decal | particles_vfx | 45 | lifetime, fade_time, channel | Projected decal |
| @asset | assets | 8 | extensions, loader | Asset handler |
| @residency | assets | 8 | priority, min_mip | GPU residency |
| @cook | assets | 8 | platform, compression | Asset cooking |

### PBR Material Parameters
| Parameter | Type | Range | Default | Descriptor |
|-----------|------|-------|---------|------------|
| base_color | Vec4 | [0,1] | (1,1,1,1) | Tracked + Range |
| metallic | float | [0,1] | 0.0 | Tracked + Range |
| roughness | float | [0,1] | 0.5 | Tracked + Range |
| normal_scale | float | [0,2] | 1.0 | Tracked + Range |
| ao | float | [0,1] | 1.0 | Tracked + Range |
| emissive | Vec3 | [0,∞) | (0,0,0) | Tracked |

### SystemPhase for Rendering
| Phase | Value | Rendering Systems |
|-------|-------|-------------------|
| PRE_RENDER | 6 | LOD selection, culling prep, interpolation |
| RENDER | 7 | GPU culling, draw submission, present |

### Frame Graph Pass Order (typical)
| Order | Pass | Type | Reads | Writes |
|-------|------|------|-------|--------|
| 1 | Shadow Atlas | Graphics | Scene | Shadow maps |
| 2 | G-Buffer | Graphics | Scene | Albedo, Normal, Metal/Rough, Depth |
| 3 | SSAO | Compute | Depth, Normal | AO buffer |
| 4 | Light Culling | Compute | Depth | Light lists (froxels) |
| 5 | Lighting | Compute | G-Buffer, Shadows, AO, Lights | HDR target |
| 6 | Transparent | Graphics | Depth | HDR target (blend) |
| 7 | Post-Process | Graphics/Compute | HDR target | LDR backbuffer |
| 8 | UI | Graphics | — | Backbuffer (overlay) |
| 9 | Present | — | Backbuffer | Swap chain |

### Descriptor Stacking for Render Fields
```
StorageDescriptor           ← innermost (raw storage)
  → RangeDescriptor         ← value bounds (roughness 0-1)
    → ValidatedDescriptor   ← custom validators
      → TrackedDescriptor   ← dirty flags → Foundation Tracker → render system
        → ObservableDescriptor ← subscriber notifications (optional)
```

### GI Technique Selection Guide
| Technique | Static Geo | Dynamic Geo | Indoor | Outdoor | Cost |
|-----------|-----------|-------------|--------|---------|------|
| Baked Lightmaps | Excellent | No | Good | Good | Offline |
| Light Probes | Good | Good | Good | Good | Low |
| DDGI | Good | Good | Excellent | Good | Medium |
| Lumen | Good | Excellent | Excellent | Excellent | High |
| Path Tracing | Perfect | Perfect | Perfect | Perfect | Very High |

---

*End of RENDERING_CONTEXT.md — This file is the sole reference for implementing engine/rendering/.*
