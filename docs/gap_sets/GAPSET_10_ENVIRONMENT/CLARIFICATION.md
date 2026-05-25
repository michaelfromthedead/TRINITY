# CLARIFICATION -- World Layer vs. Rendering Layer Architectural Split

## The Core Issue

GAPSET_10_ENVIRONMENT tasks span two distinct architectural layers that are **already separated** in the codebase but where the separation is not immediately obvious from the task descriptions in PHASE_N_TODO.md. Understanding this split is critical for correct implementation.

## The Two Layers

### Layer 1: World Layer (`engine/world/`)

**Purpose:** Data, logic, simulation, and authoring for environment elements.

**What exists:** Comprehensive Python implementations for sky, weather, time-of-day, lighting, volumes, terrain heightfields/LOD/materials, foliage placement/types, and world partition streaming.

**Characteristics:**
- Runs on CPU in the game/simulation loop
- Owns the authoritative state (sun position, weather state, terrain height data)
- Provides params to the rendering layer each frame
- Fully testable without GPU (7 passing test files)
- Uses Python ECS components (@component decorators)
- Follows Trinity patterns for data-driven design

### Layer 2: Rendering Layer (`engine/rendering/` + `crates/renderer-backend/shaders/`)

**Purpose:** GPU execution -- rendering passes, shaders, GPU memory management.

**What exists:** Frame graph framework, GPU-driven culling infrastructure, lighting system (with froxel grid), PBR materials, post-processing, particles. **No environment-specific rendering code at all.**

**Characteristics:**
- Runs on GPU via WGSL compute/fragment/vertex shaders
- Reads world-layer data as uniforms/buffers each frame
- Owns GPU resources (render targets, buffers, textures)
- Managed by Python rendering classes that bridge world data to shaders
- Rust backend (`crates/renderer-backend/`) handles shader compilation and dispatch

## Architectural Data Flow

```
                   Simulation Loop (CPU)
                        |
    ┌───────────────────┴───────────────────┐
    │                                       │
    ▼                                       ▼
World Layer                          Rendering Layer
(engine/world/)                      (engine/rendering/)
    │                                       │
    │  weather_state,                        │  Pass A (shader bind)
    │  sky_params,                           │  Pass B (compute dispatch)
    │  terrain_heightfield,                  │  Pass C (framebuffer blend)
    │  foliage_instance_data                 │
    │  water_params                          │
    │                                       │
    └────────────→ Uniforms/Buffers ←───────┘
                            │
                            ▼
                    GPU Shaders
          (crates/renderer-backend/shaders/)
                            │
                            ▼
                     Frame Output

```

## Why Tasks Overlap Layers

The 38 tasks in PHASE_N_TODO.md were written before the world-layer code existed. Many tasks describe features that have partial implementation in the world layer but complete absence in the rendering layer. This creates three patterns:

### Pattern A: World Data Exists + Rendering Layer Needed

The world layer has the data/models, but the rendering pass and shader are entirely missing.

**Examples:**
- **T-ENV-1.2 (Sky Rendering Pass):** `sky.py` has `ProceduralSky.compute_sky_color()`, sun direction, and atmosphere params. Missing: full-screen quad pass in frame graph, sky WGSL shader, LUT sampling.
- **T-ENV-1.9 (Terrain Clipmap):** `world/terrain/lod.py` has LOD data structures. Missing: GPU clipmap compute shader, ring buffer management, indirect draw generation.
- **T-ENV-1.10 (Terrain Material):** `world/terrain/materials.py` has material definitions. Missing: GPU splat map blending shader, bindless texture arrays.
- **T-ENV-1.11 (Foliage Instancing):** `world/foliage/instances.py` and `types.py` have instance data. Missing: GPU frustum culling compute shader, multi-draw-indirect, LOD selection shader.
- **T-ENV-3.1 (Weather Maps):** `weather.py` has `WeatherStateMachine` with all states and 16-field `WeatherParameters`. Missing: 2D weather map texture generation, GPU integration for clouds.
- **T-ENV-3.7 (World Partition Streaming):** `world/partition/streaming.py` has streaming logic. Missing: GPU async loading bridge, cell activation triggering clipmap updates.

**Implementation approach for these:** Read the world-layer data structures, then build rendering passes and shaders that consume that data via uniforms/buffers. The world layer is NOT modified -- it already produces the correct data. The rendering layer reads it.

### Pattern B: No World Data + Full New Implementation Needed

Neither the world layer nor rendering layer has anything. Both must be built (or the world layer first, then the rendering layer).

**Examples:**
- **T-ENV-1.1 (Bruneton LUTs):** No CPU or GPU LUT precomputation exists. `sky.py` uses simplified non-LUT scattering.
- **T-ENV-1.4, 1.5, 1.6 (Froxel System):** No froxel volume, density field, or compositing exists. The S4 lighting module has `FroxelGrid` for light culling only (not for volumetric fog).
- **T-ENV-1.7, 1.8 (Gerstner + Water Shading):** No Gerstner wave compute or water shading exists. `shallow_water.py` is a CPU SWE solver, not GPU waves.
- **T-ENV-2.1-2.13 (Clouds, FFT, Virtual Texturing):** Entirely new subsystems.
- **T-ENV-3.2-3.6, 3.8-3.12 (Integration, Mobile, Advanced Water):** Mixed -- some world data exists, some doesn't.

**Implementation approach:** Build world-layer data components first (if needed), then build rendering passes and shaders. New world components should follow existing patterns in `engine/world/environment/`.

### Pattern C: Partially Done -- Directory/Structure Exists but Location Differs

Some files exist but in a different location than the task expects, or at a different depth.

**Examples:**
- **T-ENV-1.12 (Missing Directories):** The task says `engine/world/terrain/`, `engine/world/foliage/`, `engine/world/partition/` are missing directories that need stubs. In reality, these directories exist with **full implementations** -- not stubs. The directories that are actually missing are `engine/rendering/terrain/`, `engine/rendering/water/`, and `engine/rendering/texturing/`.
- **T-ENV-1.13 (Decorators):** The task describes 9 decorators for world building (`@terrain_patch`, `@heightfield`, `@grass_type`, `@biome`, etc.). None exist. However, the Trinity decorator system does exist at the framework level (Tier 42 rendering decorators, Tier 40 GPU decorators, Tier 31 LOD/streaming decorators). The 9 new decorators should follow existing patterns but use Tier 48 for world building.

## Impact on Implementation

When implementing any GAPSET_10 task, the developer must:

1. **Check world layer first:** Read `engine/world/environment/`, `engine/world/terrain/`, `engine/world/foliage/`, or `engine/world/partition/` to see if relevant data structures exist.
2. **Map world data to rendering inputs:** Create a Python rendering class that reads world data and packs it into GPU uniforms/buffers.
3. **Write the rendering pass:** Declare the pass in the frame graph, write the WGSL shader.
4. **Do NOT duplicate world data:** The world layer is the single source of truth. The rendering layer reads, not owns.

## Concrete Example: Sky Rendering

What the world layer provides (`sky.py`):
```python
class ProceduralSky:
    def compute_sky_color(self, view_dir, sun_dir, t):  # CPU path
    def get_aerial_perspective(self, origin, view_dir, distance):  # CPU
    @property
    def sun_direction(self): ...
    @property
    def rayleigh_coefficient(self): ...
    @property
    def mie_coefficient(self): ...

class CelestialBody:
    position, angular_radius, brightness

class StarField:
    stars: list of (direction, magnitude, twinkle_freq, twinkle_phase)
```

What the rendering layer must create (missing):
```python
# engine/rendering/atmosphere/sky_pass.py
class SkyPass:
    """Full-screen sky pass consuming world-layer atmosphere data."""
    def build(self, fg: FrameGraph) -> PassNode:
        # Declare pass, bind sky params as uniforms
        # Dispatch full-screen triangle with sky WGSL shader

# crates/renderer-backend/shaders/atmosphere/sky.wgsl
# Sample Sky-View LUT by view direction
# Apply aerial perspective integration
# Output HDR color to render target
```
