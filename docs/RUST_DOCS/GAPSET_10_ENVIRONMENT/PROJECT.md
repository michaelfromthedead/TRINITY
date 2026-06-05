# GAPSET_10_ENVIRONMENT -- Project Overview

## Purpose

Gapset 10 covers the Environment rendering subsystem: transforming the existing world-layer environment data into GPU-driven visual output. This is the largest gapset by task count (38 tasks across 3 phases) and spans the boundary between Python world data and Rust/WGSL GPU rendering.

## Scope

### What Already Exists (World Layer -- `engine/world/`)

The world/logic layer is well-implemented with production-quality Python code:

| Module | Lines | Key Classes |
|--------|-------|-------------|
| `engine/world/environment/sky.py` | 777 | `ProceduralSky`, `HDRISky`, `StaticSky`, `CelestialBody`, `StarField`, `SkyManager` |
| `engine/world/environment/weather.py` | 823 | `WeatherType` (9 enums), `WeatherParameters` (16 fields), `WeatherStateMachine`, `WeatherPreset` (8 presets), `RegionalWeather` |
| `engine/world/environment/time_of_day.py` | 761 | `SunPosition`, `TODLighting` (12 params), `TODCurve`, `TimeOfDayController`, `TimeOfDayPreset` |
| `engine/world/environment/lighting.py` | 813 | `DirectionalLight`, `SunLight`, `MoonLight`, `EnvironmentLighting`, `LightProbe`, `LightProbeGrid` |
| `engine/world/environment/volumes.py` | 1297 | 18 volume types including `FogVolume`, `WaterVolume`, `VolumeManager` |
| `engine/world/environment/constants.py` | 393 | ~150 centralized constants |
| `engine/world/terrain/` | 8 files | `heightfield.py`, `lod.py`, `materials.py`, `patch.py`, `features.py`, `sculpting.py`, `component.py`, `constants.py` |
| `engine/world/foliage/` | 6 files | `grass.py`, `instances.py`, `placement.py`, `types.py`, `constants.py` |
| `engine/world/partition/` | 5 files | `cell.py`, `grid.py`, `streaming.py`, `data_layer.py`, `constants.py` |
| `engine/simulation/fluid/shallow_water.py` | 478 | CPU-based 2D shallow water equations solver |

### What Is Missing (Rendering Layer -- `engine/rendering/` + `crates/renderer-backend/shaders/`)

**Zero environment-specific rendering passes or shaders.** The `engine/rendering/` directory has no `environment/`, `terrain/`, `water/`, or `texturing/` subdirectories. The shader directory has only `ddgi.wgsl` as an environment-adjacent shader. All 38 tasks in GAPSET_10 require rendering-layer implementation.

### Key Architecture Principles

1. **World layer owns data**, rendering layer owns GPU output. The world-layer classes listed above provide parameters and state that the rendering layer reads each frame.
2. **Rendering passes** are declared in the frame graph (`engine/rendering/framegraph/`) and orchestrated by the render pipeline.
3. **WGSL shaders** live in `crates/renderer-backend/shaders/` and are loaded by the Rust renderer backend.
4. **Python rendering code** (`engine/rendering/`) bridges world data to shader uniforms/buffers and manages pass dispatch.
5. **Decorators** tie ECS components to rendering features. New decorators in Tier 48 (world building) will tag entities with terrain, foliage, biome, and environment-volume metadata.

## Task Summary

| Phase | Tasks | Focus | Effort |
|-------|-------|-------|--------|
| Phase 1 | 13 | Atmosphere foundation + water/terrain core | ~60-90 person-days |
| Phase 2 | 13 | Clouds, FFT ocean, virtual texturing | ~60-90 person-days |
| Phase 3 | 12 | Integration, polish, mobile fallback | ~45-70 person-days |
| **Total** | **38** | | **~165-250 person-days** |

### Phase 1: Atmosphere Foundation & Water/Terrain Core

Establishes the base rendering infrastructure for environment effects:

- **Atmosphere:** Bruneton LUT precomputation, sky rendering pass, sun/moon/stars GPU rendering
- **Volumetrics:** Froxel volume management, density field + light scattering, froxel compositing
- **Water:** Gerstner wave compute shader, water shading pass (Fresnel, reflection, refraction)
- **Terrain:** Clipmap compute shader (8-level nested grid), material blending (splat maps, 8 layers)
- **Foliage:** GPU instancing (1M instances, 3 LODs, frustum culling)
- **Foundation:** 3 missing rendering directories, 9 missing world-building decorators

### Phase 2: Clouds, FFT Ocean & Virtual Texturing

Adds advanced atmospheric and water effects plus terrain texturing infrastructure:

- **Clouds:** Noise textures (Worley/Perlin-Worley FBM), ray marching (Beer's law, powder effect, multi-scattering), cloud shadows on terrain, god rays
- **Temporal:** Temporal reprojection for fog and clouds (velocity buffer, depth rejection)
- **LUT Pipeline:** LUT cooking (S16 integration, .trinity_lut format)
- **Ocean:** FFT ocean simulation (Phillips spectrum, IFFT), foam generation (crest + shore)
- **Virtual Texturing:** Page table (1024x1024 RGBA16Uint), physical atlas (16Kx16K), feedback pass, streaming system

### Phase 3: Integration, Polish & Mobile Fallback

Completes the environment system with full integration, performance optimization, and mobile support:

- **Weather Maps:** 2D weather map texture (512x512), GPU-integrated cloud coverage/type blending
- **Integration:** Aerial perspective on terrain, layered fog system (ground/mid/high), S4 light integration for froxels
- **Optimization:** Performance budget system (per-feature GPU time targets), quality tiers (Low/Medium/High/Ultra), adaptive froxel resolution
- **Mobile:** Mobile fallback profile (Low everywhere, simplified techniques)
- **Advanced Water:** FFT multi-cascade ocean, underwater post-process (caustics, absorption), shoreline interaction
- **Polish:** Foliage wind animation, advanced advection-based foam

## Dependencies

Tasks depend on these other gapsets and systems:

| Dependency | Gapset | Details |
|-----------|--------|---------|
| Frame Graph | S1 | Pass declarations, resource aliasing, barrier insertion |
| GPU-Driven Rendering | S2 | Indirect draw generation, GPU culling infrastructure |
| Clustered Lighting | S4 | Clustered light list for froxel light scattering |
| Reflections | S7 | SSR, planar reflections for water shading |
| Render Hardware Interface | S14 | Bindless textures, storage buffers for virtual texturing |
| Math Library | S15 | Vec3, Mat4, complex numbers for FFT ocean |
| Asset Pipeline | S16 | LUT storage, async I/O for virtual texturing streaming |
| Trinity Decorator System | Tier system | Base classes for 9 new environment decorators |

## Testing Strategy

- **Unit tests** for world-layer data classes (7 existing test files, all passing)
- **Shader tests** for each WGSL compute shader (LUT correctness, wave validation, FFT round-trip)
- **Integration tests** for rendering passes through frame graph
- **Visual regression** tests for atmosphere, water, terrain
- **Performance benchmarks** with GPU timing queries per pass

## Key Design Decisions

1. **Bruneton scattering** over Preetham/Hosek-Wilkie for LUT-based atmosphere (higher quality, consistent with modern AAA)
2. **Gerstner + optional FFT** for water (Gerstner for near-surface detail, FFT for distant ocean)
3. **Froxel-based volumetrics** over ray-marched only (reuse existing S4 froxel infrastructure)
4. **Clipmap terrain** over CDLOD/quadtree (constant vertex count per level, simpler GPU implementation)
5. **GPU compute for foliage culling** rather than CPU (handles 1M+ instances efficiently)
6. **Virtual texturing** for terrain materials (splat maps at 2K resolution, streamed from disk)
7. **CPU LUT precomputation** for Bruneton (portable, under 500ms target, deterministic)

## Directory Structure Targets

After full implementation, the rendering layer will gain these new directories:

```
engine/rendering/
  atmosphere/
    __init__.py
    bruneton_luts.py            # CPU LUT precomputation
    sky_pass.py                  # Full-screen sky rendering
    volumetric_fog.py            # Froxel-based volumetric fog
    clouds.py                    # Cloud noise + ray marching
    god_rays.py                  # Volumetric light shafts
  
  terrain/
    __init__.py
    terrain_clipmap.py           # Clipmap compute shader wrapper
    terrain_material.py          # Splat map material blending
    foliage_instancing.py        # GPU foliage instancing + culling
  
  water/
    __init__.py
    gerstner_waves.py            # Gerstner wave compute
    fft_ocean.py                 # FFT ocean simulation
    water_shading.py             # Fresnel/reflection/refraction
    foam.py                      # Crest + shore + advection foam
    underwater.py                # Underwater post-process
  
  texturing/
    __init__.py
    virtual_texturing.py         # Page table + feedback + streaming
    texture_compression.py       # BC/ASTC/ETC2 handling
```

And these new shader files:

```
crates/renderer-backend/shaders/
  atmosphere/
    bruneton_transmittance.wgsl  # Transmittance LUT compute
    bruneton_sky_view.wgsl       # Sky-View LUT compute  
    bruneton_aerial.wgsl         # Aerial Perspective LUT compute
    sky.wgsl                     # Full-screen sky pass
    sun_moon.wgsl                # Sun disk + moon rendering
    stars.wgsl                   # Star field rendering
  
  clouds/
    cloud_noise.wgsl             # Noise texture generation
    cloud_march.wgsl             # Ray marching compute
    cloud_shadow.wgsl            # Cloud shadow on terrain
    god_ray.wgsl                 # Volumetric light shafts
  
  water/
    gerstner.wgsl                # Gerstner wave displacement compute
    fft_ocean.wgsl               # FFT ocean compute
    water_shading.wgsl           # Water surface shading
    foam.wgsl                    # Foam generation
  
  terrain/
    clipmap.wgsl                 # Clipmap vertex generation
    terrain_material.wgsl        # Splat map blending
    foliage_cull.wgsl            # GPU foliage culling
    foliage_wind.wgsl            # Foliage wind animation
  
  texturing/
    vt_feedback.wgsl             # Virtual texture feedback pass
    vt_sample.wgsl               # Virtual texture sampling
```
