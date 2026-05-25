# PHASE_3_ARCH.md -- Integration, Polish & Mobile Fallback

## Overview

Phase 3 completes the environment rendering system by integrating all components, adding performance optimization, mobile support, and advanced water interactions. Unlike Phases 1 and 2 which focus on individual feature implementation, Phase 3 focuses on cross-system integration, quality-of-life features, and scalability.

**Total effort:** ~45-70 person-days
**Files to create/modify:** ~15 new Python files + ~6 new WGSL shaders + modifications to Phase 1+2 files

---

## T-ENV-3.1: Weather Map System

### World Layer Inputs

`weather.py` already provides:
- `WeatherStateMachine` with states: CLEAR, CLOUDY, RAIN, STORM, FOG, SNOW, HAIL, SANDSTORM, OVERCAST
- `WeatherParameters` with: cloud_density, precipitation, wind_direction, wind_speed, fog_density, temperature, humidity, pressure
- `WeatherZone`, `RegionalWeather` -- spatial weather zones with blending

### Gap

The weather system has no **2D weather map texture** for GPU consumption. Cloud rendering (T-ENV-2.2) has a temporary uniform float for coverage. The weather map provides spatial variation across the world.

### Implementation

Create weather map generation tied to existing `WeatherStateMachine`:

```python
class WeatherMapGenerator:
    """Generates 2D weather map texture (512x512 RG8) from weather state."""
    def generate(self, weather_system: WeatherSystem, world_bounds: AABB) -> Texture:
        """RG channels: coverage (R), cloud_type (G)."""
        # Sample weather zone at world coordinates
        # Apply procedural noise for detail
        # Blend between weather zones over transition regions
        return weather_map_texture

class WeatherMapRenderer:
    """Bridges weather state to GPU cloud parameters."""
    def update_uniforms(self, weather: WeatherStateMachine, cloud_pass: CloudMarchPass):
        # Upload current weather parameters to cloud shader uniforms
        # Bind weather map texture for per-pixel coverage variation
        cloud_pass.uniforms.wind_direction = weather.current.wind_direction
        cloud_pass.uniforms.wind_speed = weather.current.wind_speed
        cloud_pass.uniforms.cloud_coverage = weather.current.cloud_density
```

**Weather map channels:**
- R: Cloud coverage (0 = clear, 1 = overcast)
- G: Cloud type (0 = cumulus, 1 = stratus, interpolated)

**Weather state transitions:** Blend weather maps over 30-120 seconds (matches existing `WeatherStateMachine.transition_duration`).

**Wind:** `WeatherParameters.wind_direction` + `wind_speed` drive cloud animation speed and direction. The cloud shader scrolls noise texture UVs by `wind_speed * dt` in `wind_direction`.

---

## T-ENV-3.2: Aerial Perspective Integration

### World Layer Inputs

`sky.py` has `ProceduralSky.get_aerial_perspective(origin, view_dir, distance)` -- a simplified CPU implementation.

### Rendering Implementation

```python
class AerialPerspectivePass:
    """Applies aerial perspective LUT (from T-ENV-1.1) to scene."""
    def apply_to_terrain(self, encoder, terrain_color, depth, view_dir):
        """Sample Aerial Perspective LUT at (view_zenith_cos, sun_zenith_cos, distance).
           far_color = terrain_albedo * transmittance + inscatter"""
```

Create `crates/renderer-backend/shaders/atmosphere/aerial_perspective.wgsl`:

```wgsl
// For each pixel:
//   1. Compute view distance from depth buffer
//   2. Compute view_zenith_cos = dot(normalize(view_dir), vec3(0, 1, 0))
//   3. Sample aerial perspective LUT at (view_zenith_cos, sun_zenith_cos, distance)
//   4. LUT returns (inscatter.rgb, transmittance.rgb)
//   5. Output: scene_color * transmittance + inscatter
```

**Distance thresholds:** Match terrain LOD levels for consistent blending. Horizon: farthest terrain LOD blends into sky horizon color with no visible seam.

---

## T-ENV-3.3: Layered Fog System

### World Layer Inputs

`volumes.py` has `FogVolume` with:
- `density` -- base fog density
- `color` -- fog color
- `height_falloff` -- exponential falloff rate
- `inscattering` -- light scattering coefficient

### Implementation

Extend the froxel density field (T-ENV-1.5) with multi-layer support:

```python
class FogLayer:
    """One layer in the layered fog system."""
    density: float
    color: vec3
    height: float       # center altitude
    falloff: float      # exponential falloff rate above/below center
    quality_tier: str   # "always" | "high_only" | "ultra_only"

class LayeredFogConfig:
    layers: list[FogLayer] = [
        FogLayer("ground", density=0.1, color=(0.8, 0.85, 0.9), height=0, falloff=0.5),
        FogLayer("mid_haze", density=0.03, color=(0.7, 0.75, 0.8), height=500, falloff=0.2),
        FogLayer("high_haze", density=0.01, color=(0.6, 0.65, 0.7), height=2000, falloff=0.1),
    ]
    def accumulate_into_froxel(self, froxel_volume: FroxelVolume):
        """Add each layer's contribution to the froxel radiance/extinction field."""
```

**Ground fog:** Dense, low altitude, sharp falloff (mist, morning fog).
**Mid haze:** Subtle, mid altitude, gradual falloff (distant mountains).
**High haze:** Thin, high altitude (upper atmosphere blue haze).

Each layer contributes additively to the froxel density field. Per-layer quality toggle allows disabling ground fog on mobile.

---

## T-ENV-3.4: Full S4 Light Integration for Froxels

### World Layer Inputs

S4 lighting (`engine/rendering/lighting/`) provides:
- Clustered light culling (`FroxelGrid`, `ClusteredLightCuller`) -- partitions view frustum into 3D froxels
- Point lights, spot lights, directional lights with shadow maps
- Light lists per cluster

### Implementation

Connect S4 clustered lights with the froxel volumetric system:

```python
class FroxelLightIntegration:
    """Evaluates S4 lights through the froxel volume."""
    def scatter_lights(self, froxel_volume: FroxelVolume, 
                       light_list: LightList, shadow_maps: ShadowAtlas):
        """For each froxel, evaluate N nearest lights from S4 clustered list."""
```

Create `crates/renderer-backend/shaders/atmosphere/froxel_lights.wgsl`:

```wgsl
// Extends froxel scatter shader (T-ENV-1.5):
// For each froxel:
//   1. Fetch S4 clustered light list for this froxel's (x, y, z) cell
//   2. For each light:
//      a. Compute light contribution: light_intensity * phase_function
//      b. Check shadow map for directional shadow
//      c. Accumulate radiance += light_scatter * transmittance
//   3. Add sun scattering (existing T-ENV-1.5 code)
```

**Phase function per light type:** Point/spot lights use isotropic fog scattering. Directional light (sun) uses Henyey-Greenstein phase function for forward-scattering mist effect.

**Performance:** Bounded by max lights per froxel from S4 cluster culling (typically 8-16 lights max).

---

## T-ENV-3.5: Performance Budget & LOD System

### World Layer (New)

```python
class EnvironmentQualityTier(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class EnvironmentQualityProfile:
    """Per-feature quality settings for each tier."""
    tier: EnvironmentQualityTier
    
    # Atmosphere
    sky_technique: str  # "full_bruneton" | "aerial_only" | "static"
    atmosphere_enabled: bool
    
    # Fog
    froxel_resolution: tuple  # (64,48,32) for High, (32,24,16) for Low
    froxel_compositing: str  # "ray_march" | "single_lookup"
    fog_layers: list[str]  # which layers are enabled
    
    # Clouds
    cloud_steps: int  # 32/64/128/256
    cloud_half_res: bool
    cloud_shadows: bool
    god_rays: bool
    
    # Water
    water_technique: str  # "gerstner_fft" | "gerstner_only" | "simplified"
    foam_enabled: bool
    
    # Terrain
    clipmap_levels: int  # 4/6/8
    clipmap_finest_spacing: float  # 0.5/1.0/2.0
    splat_resolution: int  # 512/1024/2048
    virtual_texturing: bool
    
    # Foliage
    max_foliage_instances: int  # 100K/250K/500K/1M
    foliage_lod_count: int  # 1/2/3
    foliage_wind: bool
    
    # Temporal
    temporal_reprojection: bool
    velocity_buffer: bool
    
    # Budget
    total_gpu_budget_ms: float  # 1.0/2.0/4.0/8.0

    @classmethod
    def for_tier(cls, tier: EnvironmentQualityTier) -> 'EnvironmentQualityProfile':
        ...
```

**Budget allocation (1080p Medium):**
- Sky + atmosphere: <0.3ms
- Fog (froxels): <0.5ms
- Clouds: <0.5ms
- Water: <0.3ms
- Terrain (clipmap + material): <0.4ms
- Foliage (culling + draw): <0.3ms
- Virtual texturing (feedback + upload): <0.2ms
- Temporal reprojection: <0.2ms
- **Total environment GPU budget:** <2.0ms

---

## T-ENV-3.6: Mobile Fallback Quality Profile

### Implementation

```python
mobile_profile = EnvironmentQualityProfile.for_tier(EnvironmentQualityTier.LOW)

# Explicit mobile-only overrides:
mobile_profile.sky_technique = "aerial_only"  # No full Bruneton sky
mobile_profile.froxel_resolution = (32, 24, 16)
mobile_profile.froxel_compositing = "single_lookup"
mobile_profile.cloud_steps = 32
mobile_profile.cloud_half_res = True
mobile_profile.cloud_shadows = False
mobile_profile.god_rays = False
mobile_profile.water_technique = "gerstner_only"  # No FFT
mobile_profile.foam_enabled = False
mobile_profile.clipmap_levels = 6
mobile_profile.clipmap_finest_spacing = 2.0  # 2m spacing
mobile_profile.splat_resolution = 512
mobile_profile.virtual_texturing = False  # Standard textures instead
mobile_profile.max_foliage_instances = 100_000
mobile_profile.foliage_lod_count = 1  # Billboard only
mobile_profile.foliage_wind = False
mobile_profile.temporal_reprojection = False
mobile_profile.total_gpu_budget_ms = 3.0  # Total GPU budget for mobile
```

**Key decisions for mobile:**
- No Bruneton sky rendering -- use aerial perspective LUT only (single texture lookup)
- 32x24x16 froxels (vs 64x48x32 on desktop) -- reduces compute by 8x
- 32 cloud steps at half resolution -- reduces cost by ~8x vs Ultra (256 steps full res)
- Gerstner only (no FFT) -- saves the expensive IFFT passes
- 6 clipmap levels at 2m finest spacing -- reduces vertex count by ~8x
- Billboard-only foliage -- lowest LOD only, saves geometry and culling cost
- No virtual texturing -- standard bindless texture arrays instead

---

## T-ENV-3.7: World Partition Streaming

### World Layer Inputs (Already Exist)

`engine/world/partition/` has:
- `cell.py` -- Cell definition with size, overlap
- `grid.py` -- Spatial grid structure
- `streaming.py` -- Streaming priority, state machine
- `data_layer.py` -- Data layer management
- `constants.py` -- Partition constants

### Rendering Integration

Create `engine/rendering/world_partition_streaming.py`:

```python
class WorldPartitionRenderer:
    """Bridges world partition streaming to GPU rendering."""
    def on_cell_activated(self, cell: Cell):
        """When a cell is activated:
           1. Trigger clipmap heightfield update for this region
           2. Merge foliage instances into GPU buffer
           3. Load virtual texture tiles for this cell
           """
    def on_cell_deactivated(self, cell: Cell):
        """When a cell is deactivated:
           1. Remove heightfield region from clipmap
           2. Remove foliage instances from GPU buffer
           """
    def update_streaming_priority(self, camera_pos: vec3, camera_velocity: vec3):
        """Update cell streaming priorities based on camera position + velocity."""
```

The existing world partition states (UNLOADED -> LOADING -> LOADED -> ACTIVATED) trigger GPU operations when transitioning:
- LOADING: Begin async height data load
- LOADED: Upload height data to GPU clipmap texture
- ACTIVATED: Merge foliage instances, enable VT tiles

---

## T-ENV-3.8: FFT Multi-Cascade Ocean

### Implementation

Extend T-ENV-2.8 (single cascade FFT) to three cascades:

```python
class FFTMultiCascadeOcean:
    cascades: list[FFTCascade] = [
        FFTCascade(name="near",  fft_size=512, patch_size=100,  coverage=200),
        FFTCascade(name="mid",   fft_size=256, patch_size=500,  coverage=1000),
        FFTCascade(name="far",   fft_size=128, patch_size=2000, coverage=5000),
    ]
    def render(self, encoder, camera_pos, wind_params):
        """Execute all three FFT cascades, blend transitions."""
```

**Cascade blending:** Cross-fade between cascades in the overlap region (200m and 1000m boundaries). Smoothstep function ensures no visible seam.

**Total FFT cost:** <1.0ms for all three cascades (512+256+128). Near cascade at 512x512 provides fine detail for shoreline and reflections. Far cascade at 128x128 provides distant wave motion at low cost.

---

## T-ENV-3.9: Underwater Post-Process

### Implementation

Create `engine/rendering/water/underwater.py`:

Create `crates/renderer-backend/shaders/water/underwater.wgsl`:

```wgsl
// Activated when camera is below water surface
// For each pixel:
//   1. Caustics: project sunlight through water heightfield onto scene
//      - Screen-space approach: sample water height at screen UV + offset
//      - Compute caustic intensity from water surface curvature
//   2. Absorption: wavelength-dependent transmittance
//      - Red: 0.02/m, Green: 0.05/m, Blue: 0.15/m (clear water)
//      - transmittance = exp(-absorption_coefficient * distance)
//   3. Blue-green color shift
//      - final_color = scene_color * water_color * transmittance
//   4. Fog: exponential distance fade based on water turbidity
//   5. Distortion: screen-space offset from surface normal
//      - Sample water surface normal at screen UV
//      - Offset scene UV by normal.xy * refraction_strength
```

**Performance:** <0.2ms full-screen pass. Uses one additional texture read for water surface height/normal.

---

## T-ENV-3.10: Shoreline Interaction

### Implementation

Create `engine/rendering/water/shoreline.py`:

```python
class ShorelineConfig:
    morph_distance: float = 10.0      # meters: water transitions to terrain
    wave_absorption_distance: float = 50.0  # meters: waves dampen near shore

class ShorelineRenderer:
    """Handles water-terrain boundary interactions."""
    def clamp_to_terrain(self, water_vertices, terrain_heightfield):
        """Snap water surface to terrain height at shoreline."""
    def shoaling(self, wave_amplitudes, water_depth):
        """Decrease wave amplitude as depth decreases."""
```

Create `crates/renderer-backend/shaders/water/shoreline.wgsl`:

```wgsl
// In water vertex shader:
// For each water vertex:
//   1. Sample terrain height at this position
//   2. Compute water depth: water_height - terrain_height
//   3. If depth < morph_distance:
//      a. Morph water height toward terrain height
//      b. Vertex morph alpha = 1 - depth / morph_distance
//      c. final_height = lerp(water_height, terrain_height, morph_alpha)
//   4. Wave shoaling: scale amplitude by clamp(depth / absorption_distance, 0, 1)
//   5. Normal adjustment: blend water normal with terrain normal at shore
```

**Seam prevention:** Water vertices overlap terrain slightly (vertices extend 1-2m beyond shore). No visible seam at water/land boundary.

---

## T-ENV-3.11: Foliage Wind Animation

### World Layer Inputs

`weather.py` already has `WeatherParameters.wind_speed` and `WeatherParameters.wind_direction`.

### Implementation

Create `crates/renderer-backend/shaders/terrain/foliage_wind.wgsl`:

```wgsl
struct WindUniforms {
    direction: vec2<f32>,  // wind direction (normalized)
    strength: f32,         // 0-10 m/s
    frequency: f32,        // gust frequency
    time: f32,             // global time
}

// In foliage vertex shader:
// For each vertex:
//   1. Compute primary sway:
//      sway = sin(time * frequency + dot(world_pos.xz, direction) * 0.5)
//      displacement = direction * strength * sway * vertex_height
//   2. Compute turbulence:
//      noise = sample_noise(world_pos.xz + time * direction * strength)
//      turbulence = noise * 0.3 * strength
//   3. Total displacement = primary_sway + turbulence
//   4. Apply rotation matrix to vertex position + normal
```

**Per-type response:** Configurable per foliage type via `@foliage_type` decorator (`wind_response` parameter):
- Grass: full response (bends a lot)
- Shrubs: 0.7x response
- Trees: 0.3x response (stiffer trunk, flexible branches)
- Wind response param: `FoliageType.wind_response` from `world/foliage/types.py`

**Weather system integration:** Wind params come directly from `WeatherParameters` each frame.

---

## T-ENV-3.12: Advanced Foam Simulation (Advection-Based)

### Implementation

Create `engine/rendering/water/foam_advection.py`:

```python
class AdvectionFoamConfig:
    enabled: bool = False  # Off by default (performance cost)
    advection_speed: float = 1.0
    dissipation_rate: float = 0.5  # per second
    accumulation_strength: float = 0.3

class FoamAdvectionPass:
    """Advection-based foam texture evolve."""
    def execute(self, encoder, velocity_field, prev_foam_texture, dt):
        """Advect foam texture by wave velocity field.
           Accumulate in lee of obstacles (rocks, shore).
           Dissipate at rate per frame.
           Output: updated foam texture."""
```

Create `crates/renderer-backend/shaders/water/foam_advection.wgsl`:

```wgsl
// Compute dispatch over foam texture (full-res or half-res):
// For each texel:
//   1. Sample wave velocity field at this position
//   2. Trace backward: prev_uv = uv - velocity * dt
//   3. Sample previous foam at prev_uv
//   4. Apply dissipation: foam *= exp(-dissipation_rate * dt)
//   5. Accumulate in lee: if velocity_divergence < 0:
//      foam += accumulation_strength * -divergence * dt
//   6. Add new crest foam from T-ENV-2.9
//   7. Clamp foam to [0, 1]
```

**Particle foam:** For boat wakes and impacts, spawn GPU particles that leave foam trails. Integrate with existing particle system from `engine/rendering/particles/`.

**Performance:** Advection pass adds ~0.15ms at half resolution.

---

## Phase 3 Pass Schedule (Full Frame Graph)

```
Pre-Render
│
├─ Cell Streaming Update
│   └─ Priority computation, cell state transitions
│
├─ Weather Map Update
│   └─ Generate/update weather map from weather state machine
│   └─ Update wind params for clouds + foliage
│
├─ LOD Selection (all systems)
│   └─ Terrain clipmap level culling
│   └─ Foliage LOD distance compute
│   └─ Water cascade LOD (Gerstner vs FFT mix)
│   └─ Cloud quality tier selection
│
Render
│
├─ 1. Shadow Passes (S4)
├─ 2. Virtual Texturing Feedback (if VT enabled)
├─ 3. G-Buffer
│   ├─ Terrain (clipmap + splat material)
│   ├─ Opaque geometry
│   └─ Underwater objects (pre-pass)
│
├─ 4. FFT Ocean Cascade 1 (near)
├─ 5. FFT Ocean Cascade 2 (mid)
├─ 6. FFT Ocean Cascade 3 (far)
├─ 7. Gerstner Wave Compute (near-surface detail)
├─ 8. Foam Generation (crest + advection)
│
├─ 9. Froxel Scattering
│   ├─ Layered fog (3 layers)
│   ├─ Sun scattering (Heney-Greenstein)
│   ├─ S4 light integration (point/spot lights)
│   └─ Cloud shadow integration
│
├─ 10. Lighting Pass (S4)
├─ 11. Sky Rendering (LUT-based)
│   ├─ Aerial perspective integration
│   └─ Sun/moon/stars overlay
│
├─ 12. Cloud Ray Marching
│   ├─ Weather map sampling per pixel
│   ├─ Cumulus/stratus based on cloud type channel
│   └─ Temporal reprojection (blend with previous frame)
│
├─ 13. God Rays (volumetric + screen-space blend)
│
├─ 14. Water Shading
│   ├─ Fresnel (Schlick, F0=0.02)
│   ├─ Reflection (SSR + probe blend)
│   ├─ Refraction (normal-offset scene color)
│   ├─ Subsurface scatter (depth-based color)
│   ├─ Foam overlay (additive)
│   └─ Shoreline morph + shoaling
│
├─ 15. Froxel Compositing
├─ 16. Cloud Shadows on Terrain
├─ 17. Foliage Draw (indirect, wind-animated)
├─ 18. Foliage Wind Animation (vertex shader)
│
├─ 19. Underwater Post-Process (if underwater)
│   ├─ Caustics
│   ├─ Absorption
│   └─ Distortion
│
├─ 20. Transparent Pass
└─ 21. Post-Process (S5)
    └─ Temporal reprojection (full scene)
```

---

## Phase 3 Quality Tier Matrix

| Feature | Ultra | High | Medium | Low (Mobile) |
|---------|-------|------|--------|--------------|
| Sky | Full Bruneton | Full Bruneton | Aerial LUT only | Aerial LUT only |
| Froxel Res | 256x192x128 | 128x96x64 | 64x48x32 | 32x24x16 |
| Froxel Comp | Ray march | Ray march | Single lookup | Single lookup |
| Cloud Steps | 256 | 128 | 64 | 32 |
| Cloud Res | Full | Full | Full | Half |
| Cloud Shadows | Yes | Yes | No | No |
| God Rays | Yes | Yes | No | No |
| Water | Gerstner + FFT | Gerstner + FFT | Gerstner only | Gerstner only |
| FFT Cascades | 3 | 2 | 0 | 0 |
| Foam | Advection | Crest+Shore | Crest+Shore | None |
| Clipmap Levels | 8 | 8 | 6 | 6 |
| Finest Spacing | 0.25m | 0.5m | 1.0m | 2.0m |
| Splat Res | 2048 | 2048 | 1024 | 512 |
| Virtual Texturing | Yes | Yes | Yes | No |
| Foliage Instances | 1M | 500K | 250K | 100K |
| Foliage LODs | 3 | 3 | 2 | 1 (billboard) |
| Foliage Wind | Yes | Yes | Yes | No |
| Temporal Repro | Full | Full | Froxel only | No |
| Mobile Profile | -- | -- | -- | All Low |
| **Total GPU Budget** | **<8ms** | **<4ms** | **<2ms** | **<3ms** |

---

## Key Technical Decisions for Phase 3

1. **Weather maps are generated procedurally**, not artist-painted. This matches the existing procedural world-building philosophy in `engine/world/` and avoids needing a custom weather editor.

2. **Aerial perspective applied as a post-effect on terrain**, not as a separate pass. Sampled at full resolution using depth buffer, integrated into the sky pass or as a compositing step.

3. **Layered fog accumulates into the froxel density field**, not as a separate pass. This reuses the froxel scattering and compositing infrastructure from Phase 1.

4. **Mobile profile trades quality for budget across every feature** rather than disabling whole systems. Even clouds render (at 32 steps, half res) rather than being completely disabled, providing visual continuity.

5. **World partition streaming triggers GPU uploads** as a side effect of cell activation, not as a polling loop. The world partition already has a state machine; rendering hooks into it.

6. **FFT multi-cascade uses independent dispatches**, not a single large FFT. Cascade blending provides seamless transitions and optimal resolution distribution.

7. **Underwater post-process is a single full-screen pass** that reads the water surface height/normal texture. It activates automatically when camera dips below water surface.

8. **Foliage wind is a vertex shader transform** at near LOD and a texture scroll at billboard LOD. Per-type response coefficients match existing `foliage/types.py` data.

9. **Advection foam is off by default** (performance cost ~0.15ms). Crest + shore foam (T-ENV-2.9) is the primary foam system for all tiers.
