# PHASE_1_ARCH.md -- Atmosphere Foundation & Water/Terrain Core

## Overview

Phase 1 establishes the core rendering infrastructure for environment effects. It converts existing world-layer data into GPU-executable rendering passes. No new world-layer data is required -- all 13 tasks consume data from already-implemented `engine/world/environment/`, `engine/world/terrain/`, and `engine/world/foliage/` modules.

**Total effort:** ~60-90 person-days
**Files to create:** ~25 new Python files + ~12 new WGSL shaders + 9 new decorator definitions

---

## T-ENV-1.1: Bruneton LUT Precomputation

### World Layer Inputs

From `engine/world/environment/sky.py`:
- `AtmosphereSettings.rayleigh_coefficient` -- Rayleigh scattering coefficient at sea level
- `AtmosphereSettings.mie_coefficient` -- Mie scattering coefficient at sea level
- `AtmosphereSettings.rayleigh_scale_height` -- 8.4km default
- `AtmosphereSettings.mie_scale_height` -- 1.2km default
- `AtmosphereSettings.mie_asymmetry` -- Henyey-Greenstein g parameter
- `AtmosphereSettings.solar_irradiance` -- Sun brightness
- `AtmosphereSettings.zenith_angle` -- Sun position from TOD
- Ozone layer: Chappuis band absorption (needs new constants)

### Implementation

Create `engine/rendering/atmosphere/bruneton_luts.py`:

```python
class TransmittanceLUT:
    """256x64 RGBA16F, maps (view_zenith_cos, view_azimuth) -> transmittance RGB."""
    resolution = (256, 64)
    format = "rgba16f"
    def compute(self, atmosphere: AtmosphereSettings) -> np.ndarray: ...
    def save(self, path: str): ...
    def load(self, path: str) -> LUT: ...

class SkyViewLUT:
    """256x512 RGB16F, maps (view_zenith_cos, sun_zenith_cos) -> sky radiance."""
    resolution = (256, 512)
    format = "rgb16f"
    def compute(self, atmosphere: AtmosphereSettings) -> np.ndarray: ...

class AerialPerspectiveLUT:
    """32x32x32 RGBA16F, maps (view_zenith_cos, sun_zenith_cos, distance) -> inscatter + transmittance."""
    resolution = (32, 32, 32)
    format = "rgba16f"
    def compute(self, atmosphere: AtmosphereSettings) -> np.ndarray: ...
```

**Algorithm details:**
- Transmittance LUT: Integrate extinction along view ray from ground to space. Rayleigh extinction = beta_R * exp(-h/HR), Mie extinction = beta_M * exp(-h/HM). Result: exp(-integral_extinction).
- Sky-View LUT: Single scattering integral along view ray. At each sample point, compute sun transmittance, scattering coefficient, phase function.
- Aerial Perspective LUT: Same as Sky-View but parameterized by distance instead of full integral.

**Testing:** LUT plausibility -- non-zero at all angles, max >0.5 near sun direction, deterministic output for same inputs. Precompute under 500ms target with numpy.

---

## T-ENV-1.2: Sky Rendering Pass

### World Layer Inputs

- `SkyManager.atmosphere` -- AtmosphereSettings with all scattering params
- `TimeOfDayController.sun_position.direction` -- Sun direction for LUT sampling
- `TODLighting.sky_tint` -- Color tint from TOD curve

### Implementation

Create `engine/rendering/atmosphere/sky_pass.py`:

```python
class SkyPass:
    """Full-screen triangle pass. Reads Sky-View LUT + Aerial Perspective LUT."""
    def build(self, fg: FrameGraph) -> PassNode:
        # Create HDR render target (RGB16F)
        # Declare pass with depth_test = greater_equal
        # Bind LUTs as textures, sky params as uniforms
        # Return pass node
```

Create `crates/renderer-backend/shaders/atmosphere/sky.wgsl`:

```wgsl
struct SkyUniforms {
    sun_direction: vec3<f32>,
    sun_zenith_cos: f32,
    view_zenith_cos: f32,
    // ... remaining params
}

// Full-screen triangle VS: passes through, computes view direction
// PS: samples Sky-View LUT by (view_zenith_cos, view_azimuth relative to sun)
// Output: HDR color to RGB16F render target
// Depth: greater_equal test, write sky only where no geometry
```

**Pass schedule:** After G-Buffer, before transparent objects (or as a background clear equivalent).

**Integration:** The pass must handle all sun angles (noon, sunset, twilight, night) and avoid color banding via 16-bit LUT precision.

---

## T-ENV-1.3: Sun Disk, Moon & Star Field

### World Layer Inputs

From `sky.py`:
- `CelestialBody` -- position, angular_radius, brightness for sun and moon
- `StarField.stars` -- list of 5000 stars with (direction, magnitude, twinkle_freq, twinkle_phase)
- `MoonLight.phase` -- Lunar phase fraction from sun-moon-earth angle
- `MoonLight.surface_albedo` -- Default 0.12

### Implementation

Extend sky pass or create separate overlay pass.

**Sun disk:** Simple angular disk rendered at sun position. Brightness modulated by atmospheric transmittance along sun direction. Additive bloom-friendly radial gradient beyond disk.

**Moon:** Sphere rendering (or textured quad) at moon position. Lit by sun direction using diffuse reflectance model. Surface uses either procedural pattern or albedo texture. Phase computed from angle between sun, moon, and camera.

**Stars:** Point sprite or small quad per star. Magnitude maps to brightness/size. Twinkle uses sin(time * frequency + phase). Milky Way rendered as procedural noise band.

Create `crates/renderer-backend/shaders/atmosphere/sun_moon.wgsl` and `crates/renderer-backend/shaders/atmosphere/stars.wgsl`.

**Testing:** Visual verification at different times of day, different moon phases, different latitudes.

---

## T-ENV-1.4: Froxel Volume Management

### World Layer Inputs

- `FroxelConstants` (from S4 lighting) -- grid dimensions, depth partitions
- `Volumes.FogVolume` settings for local fog volumes (used in T-ENV-1.5)

### Implementation

Create `engine/rendering/atmosphere/volumetric_fog.py`:

```python
class FroxelConfig:
    """Froxel grid configuration."""
    grid_dimensions: tuple[int, int, int] = (64, 48, 32)
    near_plane: float = 0.1
    far_plane: float = 1000.0
    quality_presets = {
        "Low": (32, 24, 16),
        "Medium": (64, 48, 32),
        "High": (128, 96, 64),
        "Ultra": (256, 192, 128),
    }

class FroxelVolume:
    """3D texture pair: radiance (RGBA16F) + extinction (RGBA16F)."""
    def allocate(self, device, config: FroxelConfig): ...
    def bind_as_uav(self, pass_node, slot): ...
    def bind_as_srv(self, pass_node, slot): ...
```

**Logarithmic depth partitioning:** Depth slice Z = log2(depth/near) / log2(far/near) * num_slices. This provides better precision near the camera.

---

## T-ENV-1.5: Froxel Density Field & Light Scattering

### World Layer Inputs

- `FogVolume.density`, `FogVolume.color`, `FogVolume.height_falloff` from `volumes.py`
- `WeatherParameters.fog_density`, `WeatherParameters.fog_height` from `weather.py`
- `SunLight.direction`, `SunLight.intensity` from `lighting.py`
- `AtmosphereSettings.mie_asymmetry` for Henyey-Greenstein phase function

### Implementation

Create `crates/renderer-backend/shaders/atmosphere/froxel_scatter.wgsl`:

```wgsl
// Dispatch: one thread per froxel
// For each froxel at world-space coordinate:
//   1. Compute uniform fog density with height falloff
//   2. Accumulate local fog volume contributions (box/sphere intersection)
//   3. Ray march toward sun: accumulate density -> sun transmittance
//   4. Compute scattering: sun_radiance * transmittance_sun * HG_phase(view, sun)
//   5. Store: radiance = scattered_light, extinction = total_density
// Output: froxel radiance texture + froxel extinction texture
```

**Phase function:** `HG(g, cos_theta) = (1 - g^2) / (4 * PI * (1 + g^2 - 2*g*cos_theta)^1.5)`

**Energy conservation:** Verify incident = transmitted + absorbed within 0.01 tolerance.

---

## T-ENV-1.6: Froxel Compositing

### Implementation

Create `crates/renderer-backend/shaders/atmosphere/froxel_composite.wgsl`:

```wgsl
// Full-screen pass: one pixel = one froxel lookup
// Sample froxel volume at (x, y, depth) using bilinear XY + nearest Z
// Output: final_color = scene_color * transmittance + radiance
```

Two approaches:
- **Approach A (high quality):** Ray march through froxel slices along view ray
- **Approach B (standard):** Single froxel lookup per pixel based on depth

**Quality tier:** Low/Medium/High select approach and bilinear vs nearest sampling.

---

## T-ENV-1.7: Gerstner Wave Compute Shader

### World Layer Inputs

- `ShallowWater` parameters (domain size, resolution) -- but Gerstner is fundamentally different from SWE
- Wave parameters are new: amplitude, wavelength, direction, steepness, speed

### Implementation

Create `engine/rendering/water/gerstner_waves.py`:

```python
class GerstnerWaveSet:
    """Configurable wave ensemble."""
    num_waves: int = 16  # 8-32 depending on quality tier
    waves: list[GerstnerWave]  # amplitude, wavelength, direction_2d, steepness, phase_speed
    def to_uniform_buffer(self) -> bytes: ...
    def generate_wave_distribution(self, seed: int):
        """1-2 large swells, 4-8 medium chop, 4-8 small detail."""
```

Create `crates/renderer-backend/shaders/water/gerstner.wgsl`:

```wgsl
struct GerstnerUniforms {
    num_waves: u32,
    waves: array<WaveParams, 32>,  // padded to 32
}

// Compute: dispatch over water vertex grid
// Each thread: evaluate N waves for one vertex
// Vertical: SUM(A_i * sin(w_i * dot(D_i, (x,z)) + t * phase_i))
// Horizontal (choppy): SUM(Q_i * A_i * D_i * cos(...))
// Steepness: Q_i = 1 / (w_i * A_i * num_waves)  -- prevents loops
// Normal: analytic gradient of displacement
// Output: vertex buffer (position, normal)
```

**Dispersion relation:** `w^2 = g * k` where `g = 9.81`, `k = 2PI / wavelength`

**Quality tiers:** 8/16/24/32 waves for Low/Medium/High/Ultra. Performance target: <0.1ms for 256x256 grid with 16 waves.

---

## T-ENV-1.8: Water Shading Pass

### Dependencies

- T-ENV-1.7 provides vertex displacement + normals
- S4 lighting for sun direction
- S7 reflections for SSR + planar reflections

### Implementation

Create `engine/rendering/water/water_shading.py`:

```python
class WaterShadingConfig:
    fresnel_f0: float = 0.02  # IOR ~1.33
    refraction_strength: float = 0.05
    shallow_color: vec3 = (0.2, 0.4, 0.2)
    deep_color: vec3 = (0.0, 0.05, 0.1)
    specular_roughness: float = 0.1
    anisotropic_stretch: float = 0.3
```

Create `crates/renderer-backend/shaders/water/water_shading.wgsl`:

```wgsl
// For each water surface pixel:
//   1. Fresnel (Schlick): F = F0 + (1 - F0) * (1 - cos_theta)^5
//   2. Reflection: blend SSR (near) + probe (far) + planar (local)
//   3. Refraction: scene behind water, offset by normal distortion
//   4. Specular: GGX with anisotropic stretch along wave direction
//   5. Shallow/deep color interpolation by depth
//   6. Subsurface scatter: depth-dependent color blend
//   7. Final: reflection * Fresnel + refraction * (1 - Fresnel) + specular
```

**Performance target:** <0.3ms at 1080p.

---

## T-ENV-1.9: Terrain Clipmap Compute Shader

### World Layer Inputs

- `engine/world/terrain/heightfield.py` -- Heightfield data (resolution, height range)
- `engine/world/terrain/lod.py` -- LOD level definitions, distance thresholds
- `engine/world/terrain/patch.py` -- Terrain patch management

### Implementation

Create `engine/rendering/terrain/terrain_clipmap.py`:

```python
class ClipmapConfig:
    grid_size: int = 128  # vertices per level
    num_levels: int = 8
    finest_spacing: float = 0.5  # meters

class ClipmapRing:
    """One level of the clipmap hierarchy."""
    def compute_snapped_origin(self, camera_pos: vec3) -> vec3: ...
    def generate_indirect_args(self) -> IndirectDrawArgs: ...
    def update_height_texture(self, region: HeightRegion): ...

class TerrainClipmap:
    """All 8 clipmap levels, ring buffer management."""
    levels: list[ClipmapRing]
    def update(self, camera_pos: vec3, heightfield: Texture): ...
    def draw(self, encoder: CommandEncoder): ...
```

Create `crates/renderer-backend/shaders/terrain/clipmap.wgsl`:

```wgsl
// Compute shader: heightfield texture -> vertex buffer
// Dispatch per clipmap level
// Each thread: read height at (x, y) -> compute world position + normal
// Geomorphing: lerp(height_N, height_N+1, morph_alpha) between levels
// Normal: central differencing on heightfield
// Output: position + normal vertex buffer
// Indirect draw: generate draw args per level
```

**Ring buffer:** When camera moves more than 1 cell, shift grid and upload new strip of height data. Height data upload per strip: <0.05ms. Total clipmap overhead: <0.4ms for all 8 levels. Geomorphing prevents visible popping.

---

## T-ENV-1.10: Terrain Material Blending

### World Layer Inputs

- `engine/world/terrain/materials.py` -- TerrainMaterial definitions (albedo, normal, mask textures)

### Implementation

Create `engine/rendering/terrain/terrain_material.py`:

```python
class TerrainMaterialConfig:
    splat_resolution: int = 2048
    max_layers: int = 8

class TerrainLayerDef:
    albedo_texture: Texture
    normal_texture: Texture
    mask_texture: Texture
    uv_scale: float

class SplatMap:
    """RGBA splat maps: 4 layers per texture, 2 textures for 8 total."""
    textures: list[Texture]  # 2 textures for 8 layers
    def blend_weights(self, world_pos: vec3) -> vec4: ...
```

Create `crates/renderer-backend/shaders/terrain/terrain_material.wgsl`:

```wgsl
// Per-pixel:
//   1. Sample splat map at world_pos.uv 
//   2. Normalize weights
//   3. For each layer: sample albedo/normal/mask, blend by weight
//   4. Apply slope auto-blend (smoothstep on slope angle)
//   5. Apply height auto-blend (smoothstep on world_pos.y)
//   6. Apply curvature blending (convex/concave detection)
//   7. Stochastic sampling: random UV offset per pixel
// Output: albedo, normal, roughness, metallic, AO
```

---

## T-ENV-1.11: Foliage GPU Instancing

### World Layer Inputs

- `engine/world/foliage/instances.py` -- Instance data (positions, scales, rotations, colors)
- `engine/world/foliage/types.py` -- Foliage type definitions (density, cull distance)
- `engine/world/foliage/grass.py` -- Grass-specific settings
- `engine/world/foliage/placement.py` -- Placement rules

### Implementation

Create `engine/rendering/terrain/foliage_instancing.py`:

```python
class FoliageInstanceBuffer:
    max_instances: int = 1_000_000
    stride: int = 32  # bytes per instance
    # Instance data packed as: pos(float3), scale(float), rotation(quat packed), color(RGBA8)

class FoliageLOD:
    levels: list[LODLevel]
    # LOD 0: full mesh (near), LOD 1: cross-plane (mid), LOD 2: billboard (far)

class FoliageCulling:
    def dispatch_cull(self, view_frustum: Frustum, instance_buffer: Buffer) -> IndirectDrawBuffer:
        """GPU frustum culling + distance/LOD selection + compaction."""
```

Create `crates/renderer-backend/shaders/terrain/foliage_cull.wgsl`:

```wgsl
// Compute: one thread per instance
// For each instance:
//   1. AABB frustum test (6 frustum planes)
//   2. Distance-based LOD selection (mesh/cross-plane/billboard)
//   3. Write survivor to compacted list via atomic counter
// Output: indirect draw buffer for multi-draw-indirect
```

**LOD counts:** Grass 500K (no shadows), Shrubs 100K, Trees 50K (dynamic shadows). GPU culling time: <0.3ms for 100K instances. LOD cross-fade over 10m transition zone.

---

## T-ENV-1.12: Create Missing Rendering Directories

### Current Reality

The directories listed in the task do not account for existing world-layer implementations. The actual work needed:

| Task Says | Reality | Action |
|-----------|---------|--------|
| Create `engine/rendering/terrain/` | Does not exist | CREATE with 3 stubs |
| Create `engine/rendering/water/` | Does not exist | CREATE with 3 stubs |
| Create `engine/rendering/texturing/` | Does not exist | CREATE with 2 stubs |
| Create `engine/world/partition/` | Already exists (5 files) | NO ACTION needed |
| Create `engine/world/terrain/` | Already exists (8 files) | NO ACTION needed |
| Create `engine/world/foliage/` | Already exists (6 files) | NO ACTION needed |

Stubs should contain `@component` class definitions matching spec and import from Trinity base classes. They will be expanded by their respective tasks (T-ENV-1.7 through T-ENV-1.11 for rendering stubs).

---

## T-ENV-1.13: Create 9 Missing Trinity Decorators

### Current Decorator Landscape

Existing rendering decorators (Tier 42): `@shadow_caster`, `@gi_contributor`, `@reflection_probe`, `@material_domain`, `@material_blend`, `@render_layer`
Existing GPU decorators (Tier 40): `@gpu_buffer`, `@gpu_kernel`, `@gpu_struct`, `@bind_group`, `@dispatch`, `@shader`, `@render_pass`, `@async_compute`
Existing LOD/streaming decorators (Tier 31): `@lod`, `@streamable`, `@chunk`, `@loading_priority`, `@unloadable`

### New Decorators (Tier 48 -- World Building)

Each follows the existing Trinity decorator pattern:

```python
# Template for each new decorator
def decorator_name(param1: type = default1, ...):
    def wrapper(cls):
        # 1. TAG(decorator_name=True)
        # 2. TAG(params)
        # 3. REGISTER(world_building)  # Tier 48
        setattr(cls, f"_{decorator_name}", True)
        setattr(cls, f"_{decorator_name}_params", {...})
        return cls
    return wrapper
```

The 9 decorators:

| Decorator | Params | Purpose |
|-----------|--------|---------|
| `@terrain_patch` | size, overlap, height_data | Tag terrain patch components |
| `@heightfield` | resolution, height_scale, height_bias | Tag heightfield data components |
| `@terrain_layer` | index, name, blend_mode | Tag terrain material layers |
| `@grass_type` | density, blade_height, color_variation | Tag grass foliage types |
| `@scatter_rule` | noise_type, density, slope_range, height_range | Define placement rules |
| `@biome` | climate_zone, vegetation_set, terrain_materials | Define biomes |
| `@weather_zone` | coverage_range, wind_range, fog_range | Define weather zones |
| `@hlod_layer` | level, cell_size, merge_threshold | Hierarchical LOD layers |
| `@environment_volume` | shape, blend_radius, priority | Environment volume tags |

**File location:** Create `engine/decorators/world_building.py` (Tier 48).

---

## Phase 1 Pass Schedule (Frame Graph Integration)

```
Pre-Render (LOD selection, culling prep)
│
├─ 1. Terrain Clipmap Update (compute)
│     - Heightfield -> vertex buffers
│     - Geomorphing between levels
│     - Indirect draw arg generation
│
├─ 2. Foliage Culling (compute)
│     - Frustum cull, LOD select, compact
│     - Generate indirect draw buffer
│
├─ 3. Gerstner Wave Compute
│     - Water vertex displacement + normals
│
Render
│
├─ 4. Shadow Passes (from S4)
│     - CSM for sun, cube shadows for water/terrain objects
│
├─ 5. G-Buffer Pass (from S1)
│     - Terrain material blending, opaque geometry
│
├─ 6. Froxel Scattering (compute)
│     - Density field, light scattering, sun transmittance
│
├─ 7. Lighting Pass (from S4)
│     - Clustered deferred shading
│
├─ 8. Sky Rendering Pass
│     - Full-screen triangle, sample LUTs
│     - Sun disk, moon, stars overlay
│     - Depth test: greater_equal, sky behind geometry
│
├─ 9. Water Shading Pass
│     - Fresnel, reflection, refraction, specular
│
├─ 10. Froxel Compositing
│      - Apply volumetric fog to scene
│
├─ 11. Foliage Draw (indirect)
│      - Multi-draw-indirect for all foliage
│
├─ 12. Transparent Pass
│      - Water transparency, glass effects
│
└─ 13. Post-Process (from S5)
```

---

## Key Technical Decisions for Phase 1

1. **Bruneton LUTs computed on CPU, not GPU.** CPU precomputation (target <500ms) is portable and deterministic. GPU LUT generation is a future optimization (could be done during loading screen).

2. **Froxel compositing uses Approach B by default** (single lookup per pixel). Approach A (ray marching) is optional for High/Ultra quality tiers.

3. **Gerstner over FFT for Phase 1.** Gerstner provides detail near the camera. FFT ocean (Phase 2) adds distant wave simulation.

4. **Clipmap over CDLOD/Quadtree.** Clipmap's constant vertex count per level simplifies GPU implementation. Ring buffer pattern avoids full rebuild each frame.

5. **8-layer splat maps** (2x RGBA textures). The splat resolution is 2048x2048, stored as a virtual texture (Phase 2) or direct texture.

6. **3 LOD levels for foliage:** Full mesh (0-50m), cross-plane quads (50-150m), billboard (150-500m). LOD cross-fade over 10m using alpha blending.

7. **New decorators at Tier 48** (world building), following the same TAG/REGISTER pattern as existing Tier 42 rendering decorators. They do NOT duplicate existing decorator functionality.
