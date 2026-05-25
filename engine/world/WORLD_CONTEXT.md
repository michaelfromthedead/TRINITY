# World Layer Implementation Context

Complete context for implementing `engine/world/` using the Trinity Pattern + Foundation runtime.

---

## Table of Contents

- [Overview](#overview)
- [Part I: Trinity Recommendations for World](#part-i-trinity-recommendations-for-world)
  - [Metaclasses](#metaclasses)
  - [Descriptors](#descriptors)
  - [Decorators](#decorators)
- [Part II: Decorator Stacks for World](#part-ii-decorator-stacks-for-world)
- [Part III: World-Specific Patterns](#part-iii-world-specific-patterns)
- [Part IV: Engine/World Directory Structure](#part-iv-engineworld-directory-structure)
- [Part V: Implementation Checklist](#part-v-implementation-checklist)

---

## Overview

The World Layer (`engine/world/`) manages all aspects of the game world including level structure, terrain, foliage, environment systems, and world streaming. It provides the foundation for building and running game levels of any scale.

### Architecture Reference

From `DIAGRAMS/ARCHITECTURE_WORLD.md`:

```
WORLD LAYER
+-- Level Architecture (persistent, streaming, sub-levels, instances)
+-- World Partition (cells, grid, data layers, HLOD)
+-- Terrain System (heightfield, sculpting, materials, LOD)
+-- Foliage System (grass, shrubs, trees, instancing)
+-- Placement Systems (PCG, scatter, spline-based)
+-- Environment Volumes (physics, gameplay, visual, audio, navigation)
+-- Lighting Environment (sky, time-of-day, weather, GI)
+-- Water Bodies (ocean, lake, river, interaction)
+-- Environmental Effects (particles, decals, destruction, hazards)
+-- World Queries (spatial, actor, terrain queries)
+-- World Runtime (update, persistence, spawning, events)
```

### Integration Point

From `GAME_ENGINE_INTEGRATION.md`:

> World streaming uses `@streamable` + `@chunk` decorators with `TrackedDescriptor` for cell state. Terrain components use `@lod` for distance-based quality. Foliage uses `@procedural_placement` with seeded randomness. Environment volumes use `@trigger_volume` with enter/exit hooks. All world data flows through Foundation's Registry for lookup and Tracker for change detection.

---

## Part I: Trinity Recommendations for World

### Metaclasses

**Use `ComponentMeta` for world entities.** Terrain patches, foliage instances, volumes, and water bodies are all ECS components. This gives you:
- Unique IDs for fast lookup
- Field processing with type annotations
- Automatic descriptor installation
- Foundation Registry integration

**Use `AssetMeta` for world data assets.** Heightmaps, foliage types, material layers, and weather presets are assets. AssetMeta provides:
- Asset path management
- Loading/unloading lifecycle
- Reference tracking
- Hot-reload support

**Use `ResourceMeta` for world-global state.** Time-of-day, weather state, and streaming budget are singleton resources:
- Single instance per world
- Global access pattern
- Automatic lifecycle

**Use `StateMeta` for weather/TOD state machines.** Weather transitions and day/night cycles are state machines:
- Valid transition validation
- Enter/exit hooks
- Blending support

### Descriptors

#### Core Descriptors for World

| Descriptor | World Purpose | Example |
|-----------|--------------|---------|
| `TrackedDescriptor` | Dirty flags for streaming state | Cell load state changes trigger streaming |
| `SparseDescriptor` | Large world data with gaps | Terrain heightfield with empty chunks |
| `IndexedDescriptor` | Fast lookup by coordinates | Cell lookup by grid position |
| `CompressedDescriptor` | Memory-efficient terrain data | Heightmap compression |
| `LazyDescriptor` | Deferred loading | LOD data loaded on demand |
| `AsyncLoadDescriptor` | Async streaming | Chunk data streaming |
| `TransientDescriptor` | Non-serializable runtime state | Cached LOD meshes, render data |
| `ValidatedDescriptor` | Parameter constraints | Density, height ranges |
| `RangeDescriptor` | Numeric bounds | Height range, slope limits |
| `ObservableDescriptor` | Change notifications | Weather state -> visual updates |
| `InterpolatedDescriptor` | Smooth transitions | TOD lighting, weather blending |
| `VersionedDescriptor` | Save game compatibility | Terrain modifications |
| `BatchedDescriptor` | Bulk updates | Mass foliage placement |

#### Descriptor Chains for World Fields

**Streaming Cell State:**
```
TrackedDescriptor -> IndexedDescriptor -> StorageDescriptor
```
Cell state changes trigger streaming logic, indexed by grid position.

**Terrain Height Data:**
```
CompressedDescriptor -> SparseDescriptor -> StorageDescriptor
```
Memory-efficient storage with compression for large heightfields.

**Foliage Instance Data:**
```
BatchedDescriptor -> SparseDescriptor -> StorageDescriptor
```
Bulk operations on sparse foliage data.

**Time-of-Day Values:**
```
InterpolatedDescriptor -> TrackedDescriptor -> RangeDescriptor -> StorageDescriptor
```
Smooth interpolation, tracked changes, clamped to valid range.

**Weather Parameters:**
```
ObservableDescriptor -> InterpolatedDescriptor -> TrackedDescriptor -> StorageDescriptor
```
Observers notified of interpolated weather changes.

**Async Chunk Data:**
```
AsyncLoadDescriptor -> LazyDescriptor -> TransientDescriptor -> StorageDescriptor
```
Async loading of lazy-initialized transient chunk data.

#### Annotated Field Syntax (Preferred)

```python
from typing import Annotated
from trinity.descriptors import (
    Tracked, Sparse, Indexed, Compressed, Lazy, Async,
    Transient, Range, Observable, Interpolated, Versioned, Batched
)

@component
class TerrainPatch:
    # Identity (indexed for fast lookup)
    grid_x: Annotated[int, Indexed("terrain_grid")] = 0
    grid_y: Annotated[int, Indexed("terrain_grid")] = 0
    
    # Height data (compressed, sparse for empty areas)
    heights: Annotated[bytes, Compressed, Sparse] = b""
    
    # Material weights (sparse, versioned for saves)
    layer_weights: Annotated[dict, Sparse, Versioned(1)] = field(default_factory=dict)
    
    # LOD state (tracked for streaming)
    current_lod: Annotated[int, Tracked, Range(0, 8)] = 0
    
    # Runtime cache (transient, not saved)
    _cached_mesh: Annotated[Any, Transient] = None
    _collision_data: Annotated[Any, Transient] = None

@component  
class StreamingCell:
    # Grid position (indexed)
    cell_x: Annotated[int, Indexed("world_grid")] = 0
    cell_y: Annotated[int, Indexed("world_grid")] = 0
    
    # Load state (tracked for streaming system)
    state: Annotated[str, Tracked] = "unloaded"  # unloaded, loading, loaded, activated
    
    # Priority (tracked, range-clamped)
    load_priority: Annotated[float, Tracked, Range(0, 100)] = 0.0
    
    # Contents (async loaded)
    actors: Annotated[list, Async, Lazy] = field(default_factory=list)

@resource
class TimeOfDay:
    # Current time (interpolated for smooth transitions)
    time_hours: Annotated[float, Interpolated, Tracked, Range(0, 24)] = 12.0
    
    # Sun position (computed from time, tracked for lighting)
    sun_angle: Annotated[float, Tracked] = 0.0
    sun_color: Annotated[tuple, Tracked] = (1.0, 1.0, 1.0)
    
    # Ambient (interpolated)
    ambient_intensity: Annotated[float, Interpolated, Range(0, 2)] = 0.5
    
@resource
class WeatherState:
    # Current weather (observable for visual systems)
    weather_type: Annotated[str, Observable, Tracked] = "clear"
    
    # Intensity (interpolated for smooth transitions)
    precipitation: Annotated[float, Interpolated, Range(0, 1)] = 0.0
    wind_speed: Annotated[float, Interpolated, Range(0, 100)] = 0.0
    fog_density: Annotated[float, Interpolated, Range(0, 1)] = 0.0
    
    # Transition state
    blend_factor: Annotated[float, Tracked, Range(0, 1)] = 1.0
```

### Decorators

#### Existing World Decorators

From `trinity/decorators/world_building.py`:

| Decorator | Purpose | Parameters | Steps |
|-----------|---------|------------|-------|
| `@foliage_type` | Define foliage category | `density`, `cull_distance`, `collision`, `wind_response` | `TAG(foliage_type), TAG(density), TAG(cull_distance), TAG(collision), TAG(wind_response), REGISTER(world)` |
| `@procedural_placement` | Configure procedural scatter | `density`, `noise`, `slope_range`, `height_range` | `TAG(procedural_placement), TAG(density), TAG(noise_params), TAG(slope_range), TAG(height_range), REGISTER(world)` |
| `@level_instance` | Level streaming config | `always_loaded`, `load_on_proximity`, `proximity_radius` | `TAG(level_instance), TAG(always_loaded), TAG(load_on_proximity), TAG(proximity_radius), REGISTER(world)` |
| `@water_body` | Water type definition | `type`, `wave_source` | `TAG(water_body), TAG(water_type), TAG(wave_source), REGISTER(world)` |
| `@navmesh_modifier` | Navigation modification | `area_class`, `modifier` | `TAG(navmesh_modifier), TAG(area_class), TAG(modifier), REGISTER(world)` |
| `@trigger_volume` | Volume triggers | `events`, `filter_tags` | `TAG(trigger_volume), TAG(trigger_events), TAG(filter_tags), REGISTER(world), HOOK(on_enter), HOOK(on_exit)` |

From `trinity/decorators/lod_streaming.py`:

| Decorator | Purpose | Parameters | Steps |
|-----------|---------|------------|-------|
| `@lod` | LOD configuration | `levels`, `distances`, `bias` | `TAG(lod), TAG(lod_levels), TAG(lod_distances), TAG(lod_bias), REGISTER(lod)` |
| `@streamable` | Streaming support | `priority`, `keep_loaded` | `TAG(streamable), TAG(stream_priority), TAG(keep_loaded), REGISTER(streaming)` |
| `@chunk` | World chunk definition | `size`, `overlap` | `TAG(chunk), TAG(chunk_size), TAG(chunk_overlap), REGISTER(streaming)` |
| `@loading_priority` | Load order control | `visibility_weight`, `player_velocity_weight` | `TAG(loading_priority), TAG(visibility_weight), TAG(velocity_weight), REGISTER(streaming)` |
| `@unloadable` | Unload behavior | `min_age`, `save_state` | `TAG(unloadable), TAG(min_age), TAG(save_state), REGISTER(streaming)` |

From `trinity/decorators/spatial.py`:

| Decorator | Purpose | Parameters | Steps |
|-----------|---------|------------|-------|
| `@spatial` | Spatial indexing | `structure`, `cell_size` | `TAG(spatial), TAG(spatial_structure), TAG(spatial_cell_size), REGISTER(spatial)` |
| `@partitioned` | World partition | `dimensions`, `max_entities` | `TAG(partitioned), TAG(partition_dimensions), TAG(max_entities), REGISTER(spatial)` |

From `trinity/decorators/procedural.py`:

| Decorator | Purpose | Parameters | Steps |
|-----------|---------|------------|-------|
| `@seeded` | Deterministic generation | `seed_source` | `TAG(seeded), TAG(seed_source), REGISTER(procedural)` |
| `@procedural` | Procedural generation | `cache`, `validate` | `TAG(procedural), TAG(procedural_cache), TAG(procedural_validate), REGISTER(procedural)` |
| `@constraint` | Generation constraints | `rules` | `TAG(constraint), TAG(constraint_rules), REGISTER(procedural)` |

From `trinity/decorators/rendering.py` (relevant for world):

| Decorator | Purpose | Parameters | Steps |
|-----------|---------|------------|-------|
| `@gi_contributor` | GI contribution | `importance`, `emissive` | `TAG(gi_contributor), TAG(gi_config), REGISTER(rendering)` |
| `@shadow_caster` | Shadow casting | `mode`, `resolution_scale`, `cascade_bias` | `TAG(shadow_caster), TAG(shadow_config), REGISTER(rendering)` |
| `@reflection_probe` | Reflection capture | `capture_mode`, `resolution`, `update_rate` | `TAG(reflection_probe), TAG(reflection_config), REGISTER(rendering)` |

#### Related Decorators for World

| Decorator | Module | Purpose |
|-----------|--------|---------|
| `@on_add`, `@on_remove` | `lifecycle.py` | Cell load/unload hooks |
| `@on_spawn`, `@on_despawn` | `lifecycle.py` | Actor spawn hooks |
| `@serializable` | `data_flow.py` | Save/load world state |
| `@versioned` | `data_flow.py` | Save compatibility |
| `@networked` | `data_flow.py` | Multiplayer world sync |
| `@cached` | `bridges_caching.py` | Query result caching |
| `@async_load` | `bridges_caching.py` | Async streaming |
| `@lazy` | `bridges_caching.py` | Deferred initialization |
| `@batch` | `bridges_caching.py` | Bulk operations |

#### New World Decorators to Create

| Decorator | Purpose | Steps |
|-----------|---------|-------|
| `@terrain_patch` | Mark as terrain component | `TAG(terrain_patch), TAG(patch_size), REGISTER(terrain)` |
| `@terrain_layer` | Material layer definition | `TAG(terrain_layer), TAG(layer_index), TAG(blend_mode)` |
| `@heightfield` | Heightfield configuration | `TAG(heightfield), TAG(resolution), TAG(height_range)` |
| `@hlod_layer` | HLOD level config | `TAG(hlod_layer), TAG(hlod_level), TAG(simplification)` |
| `@weather_zone` | Weather region | `TAG(weather_zone), TAG(climate), TAG(blend_radius)` |
| `@biome` | Biome definition | `TAG(biome), TAG(biome_type), TAG(foliage_rules)` |
| `@environment_volume` | Generic env volume | `TAG(env_volume), TAG(volume_type), TAG(priority)` |
| `@grass_type` | Grass-specific foliage | `TAG(grass_type), TAG(density_scale), TAG(wind_sway)` |
| `@scatter_rule` | Scatter placement rule | `TAG(scatter_rule), TAG(pattern), TAG(filters)` |
| `@spline_actor` | Spline-based placement | `TAG(spline_actor), TAG(spline_type), TAG(repeat_mode)` |

---

## Part II: Decorator Stacks for World

### Existing Stacks (from builtin_stacks)

From `trinity/decorators/builtin_stacks/streaming.py`:

```python
@parameterized_stack
def streaming_chunk(
    chunk_size: tuple = (100, 100, 100),
    overlap: int = 10,
    min_age: float = 60.0,
) -> Stack:
    """Open world chunk streaming."""
    return stack(
        chunk(size=chunk_size, overlap=overlap),
        streamable(priority="normal"),
        loading_priority(visibility_weight=3.0, player_velocity_weight=1.5),
        unloadable(min_age=min_age, save_state=True),
        serializable(format="binary"),
        track_changes,
        async_load(priority=0, fallback=None),
        lazy(init_on="first_access"),
    )

@parameterized_stack
def lod_scalable(
    levels: int = 4,
    distances: list = None,
) -> Stack:
    """Scalable quality rendering."""
    distances = distances or [10, 50, 200, 1000]
    return stack(
        lod(levels=levels, distances=distances),
        streamable(priority="normal"),
        residency(priority="normal", min_mip=2),
    )
```

From `trinity/decorators/builtin_stacks/composite.py`:

```python
@parameterized_stack
def open_world_entity(
    pool_size: int = 10000,
    chunk_size: Tuple[int, int, int] = (100, 100, 100),
) -> Stack:
    """Open world entity with streaming, LOD, and persistence."""
    return (
        production_component(pool_size=pool_size)
        + streaming_chunk(chunk_size=chunk_size)
        + lod_scalable()
        + versioned_saveable()
    )
```

### Recommended New Stacks

Create in `trinity/decorators/builtin_stacks/world.py`:

```python
@parameterized_stack
def terrain_component(
    patch_size: int = 64,
    lod_levels: int = 6,
    distances: list = None,
) -> Stack:
    """Terrain patch with LOD and streaming."""
    distances = distances or [50, 100, 250, 500, 1000, 2000]
    return stack(
        terrain_patch(size=patch_size),
        lod(levels=lod_levels, distances=distances),
        streamable(priority="high"),
        gi_contributor(importance="high"),
        shadow_caster(mode="static"),
        track_changes,
        component,
    )

@parameterized_stack
def foliage_instance(
    foliage_type: str = "shrub",
    cull_distance: float = 1000.0,
    wind: bool = True,
) -> Stack:
    """Foliage instance with culling and wind."""
    return stack(
        foliage_type(
            density=1.0, 
            cull_distance=cull_distance, 
            collision=False, 
            wind_response=wind
        ),
        lod(levels=3, distances=[50, 200, cull_distance]),
        streamable(priority="low"),
        shadow_caster(mode="dynamic" if foliage_type == "tree" else "none"),
        component,
    )

@parameterized_stack
def procedural_foliage(
    density: float = 1.0,
    slope_range: tuple = (0, 45),
    height_range: tuple = None,
    seed: str = "world",
) -> Stack:
    """Procedurally placed foliage."""
    return stack(
        foliage_type(density=density, cull_distance=500.0, collision=False, wind_response=True),
        procedural_placement(
            density=density,
            noise={"type": "perlin", "scale": 10.0},
            slope_range=slope_range,
            height_range=height_range,
        ),
        seeded(seed_source=seed),
        streamable(priority="low"),
        component,
    )

@parameterized_stack
def world_cell(
    cell_size: tuple = (128, 128, 128),
    hlod_enabled: bool = True,
) -> Stack:
    """World partition cell."""
    return stack(
        chunk(size=cell_size, overlap=8),
        streamable(priority="normal"),
        loading_priority(visibility_weight=2.0, player_velocity_weight=1.0),
        unloadable(min_age=30.0, save_state=True),
        spatial(structure="grid", cell_size=cell_size[0]),
        hlod_layer(level=0, simplification=0.5) if hlod_enabled else _noop,
        serializable(format="binary"),
        track_changes,
        component,
    )

@parameterized_stack
def environment_trigger(
    events: list = None,
    filter_tags: list = None,
) -> Stack:
    """Environment trigger volume."""
    events = events or ["on_enter", "on_exit"]
    return stack(
        trigger_volume(events=events, filter_tags=filter_tags or []),
        track_changes,
        component,
    )

@parameterized_stack
def water_surface(
    water_type: str = "lake",
    wave_enabled: bool = True,
    buoyancy: bool = True,
) -> Stack:
    """Water body with physics."""
    return stack(
        water_body(type=water_type, wave_source="wind" if wave_enabled else "none"),
        reflection_probe(capture_mode="realtime" if water_type == "ocean" else "baked", resolution=512),
        track_changes,
        component,
    )

@parameterized_stack
def weather_system(
    initial_weather: str = "clear",
    transition_speed: float = 0.1,
) -> Stack:
    """Weather state with smooth transitions."""
    return stack(
        track_changes,
        # Weather-specific tags would go here
        resource,
    )

@parameterized_stack
def time_of_day_system(
    day_length_minutes: float = 24.0,
    start_hour: float = 12.0,
) -> Stack:
    """Time of day with lighting updates."""
    return stack(
        track_changes,
        resource,
    )

@parameterized_stack
def pcg_scatter(
    pattern: str = "poisson",
    density: float = 1.0,
    seed: str = "world",
) -> Stack:
    """PCG scatter system for placement."""
    return stack(
        seeded(seed_source=seed),
        procedural(cache=True),
        track_changes,
        component,
    )

@parameterized_stack
def navigation_modifier(
    area_class: str = "default",
    modifier: str = "include",
) -> Stack:
    """NavMesh modification volume."""
    return stack(
        navmesh_modifier(area_class=area_class, modifier=modifier),
        track_changes,
        component,
    )
```

### Composite Stacks for Common World Patterns

```python
@parameterized_stack
def open_world_terrain(
    patch_size: int = 64,
    cell_size: tuple = (256, 256, 256),
) -> Stack:
    """Complete open world terrain setup."""
    return (
        terrain_component(patch_size=patch_size)
        + world_cell(cell_size=cell_size)
        + versioned_saveable()
    )

@parameterized_stack
def dense_forest_zone(
    tree_density: float = 0.3,
    shrub_density: float = 0.6,
    grass_density: float = 1.0,
) -> Stack:
    """Dense forest biome with all vegetation layers."""
    return stack(
        biome(biome_type="forest"),
        procedural_foliage(density=tree_density, slope_range=(0, 30)),
        streamable(priority="normal"),
        component,
    )

@parameterized_stack
def streaming_level_instance(
    always_loaded: bool = False,
    proximity_radius: float = 5000.0,
) -> Stack:
    """Streaming sub-level instance."""
    return stack(
        level_instance(
            always_loaded=always_loaded,
            load_on_proximity=not always_loaded,
            proximity_radius=proximity_radius,
        ),
        streamable(priority="high" if always_loaded else "normal"),
        serializable(format="binary"),
        track_changes,
        component,
    )
```

---

## Part III: World-Specific Patterns

### World Partition Pattern

Cells divide the world into streamable units.

```python
@spatial(structure="grid", cell_size=256)
@partitioned(dimensions=2, max_entities=1000)
@component
class WorldGrid:
    """Top-level world partition grid."""
    
    # Grid dimensions
    width: Annotated[int, Tracked] = 100
    height: Annotated[int, Tracked] = 100
    cell_size: Annotated[float, Tracked] = 256.0
    
    # Active cells (indexed for fast lookup)
    cells: Annotated[dict[tuple, "StreamingCell"], Indexed("grid_cells")] = field(default_factory=dict)
    
    # Streaming state
    load_center: Annotated[tuple, Tracked] = (0, 0)
    load_radius: Annotated[float, Tracked] = 5.0  # In cells

@chunk(size=(256, 256, 256), overlap=8)
@streamable(priority="normal")
@loading_priority(visibility_weight=2.0, player_velocity_weight=1.5)
@unloadable(min_age=30.0, save_state=True)
@component
class StreamingCell:
    """Individual streaming cell."""
    
    # Grid position
    grid_x: Annotated[int, Indexed("world_grid")] = 0
    grid_y: Annotated[int, Indexed("world_grid")] = 0
    
    # Bounds
    min_bounds: Annotated[tuple, Tracked] = (0, 0, 0)
    max_bounds: Annotated[tuple, Tracked] = (256, 256, 256)
    
    # State machine
    state: Annotated[str, Tracked] = "unloaded"
    load_progress: Annotated[float, Tracked, Range(0, 1)] = 0.0
    
    # Data layers (async loaded)
    actors: Annotated[list, Async, Lazy] = field(default_factory=list)
    foliage: Annotated[list, Async, Lazy] = field(default_factory=list)
    
    # HLOD reference
    hlod_proxy: Annotated[Any, Transient] = None
```

### Terrain System Pattern

Terrain uses heightfields with LOD and material layers.

```python
@terrain_patch(size=64)
@lod(levels=6, distances=[50, 100, 250, 500, 1000, 2000])
@streamable(priority="high")
@gi_contributor(importance="high")
@shadow_caster(mode="static")
@component
class TerrainPatch:
    """Single terrain patch with LOD."""
    
    # Grid identity
    patch_x: Annotated[int, Indexed("terrain_grid")] = 0
    patch_y: Annotated[int, Indexed("terrain_grid")] = 0
    
    # Height data (compressed for memory)
    resolution: Annotated[int, Tracked] = 65  # 64+1 for stitching
    heights: Annotated[bytes, Compressed] = b""
    height_range: Annotated[tuple, Tracked] = (-500.0, 500.0)
    
    # Material layers
    layer_count: Annotated[int, Tracked, Range(1, 16)] = 4
    layer_weights: Annotated[bytes, Compressed] = b""  # RGBA weight maps
    
    # LOD state
    current_lod: Annotated[int, Tracked, Range(0, 8)] = 0
    
    # Runtime (transient)
    _mesh: Annotated[Any, Transient] = None
    _collision: Annotated[Any, Transient] = None
    _neighbors: Annotated[dict, Transient] = field(default_factory=dict)

@terrain_layer(layer_index=0, blend_mode="height")
@component
class TerrainMaterialLayer:
    """Material layer for terrain blending."""
    
    layer_index: Annotated[int, Tracked, Range(0, 15)] = 0
    material_id: Annotated[str, Tracked] = ""
    
    # Blending
    blend_mode: Annotated[str, Tracked] = "height"  # height, linear
    height_scale: Annotated[float, Tracked] = 1.0
    
    # Auto-placement rules
    slope_range: Annotated[tuple, Tracked] = (0.0, 90.0)
    height_range: Annotated[tuple, Tracked] = None  # None = no restriction
    
    # Tiling
    uv_scale: Annotated[float, Tracked] = 1.0
    stochastic: Annotated[bool, Tracked] = True
```

### Foliage System Pattern

Foliage uses hierarchical instanced meshes with procedural placement.

```python
@foliage_type(density=1.0, cull_distance=2000.0, collision=True, wind_response=True)
@lod(levels=4, distances=[50, 150, 500, 2000])
@shadow_caster(mode="dynamic")
@component
class TreeType:
    """Tree foliage type definition."""
    
    # Identity
    type_id: Annotated[str, Tracked] = ""
    mesh_id: Annotated[str, Tracked] = ""
    
    # LOD meshes
    lod_meshes: Annotated[list[str], Tracked] = field(default_factory=list)
    impostor_mesh: Annotated[str, Tracked] = ""  # Billboard at max distance
    
    # Variation
    scale_range: Annotated[tuple, Tracked] = (0.8, 1.2)
    rotation_random: Annotated[bool, Tracked] = True
    color_variation: Annotated[float, Tracked, Range(0, 1)] = 0.1
    
    # Physics
    collision_type: Annotated[str, Tracked] = "capsule"
    destructible: Annotated[bool, Tracked] = False
    
    # Wind
    wind_weight: Annotated[float, Tracked, Range(0, 2)] = 1.0

@procedural_placement(
    density=0.5,
    noise={"type": "perlin", "scale": 20.0, "octaves": 3},
    slope_range=(0, 35),
    height_range=(0, 500),
)
@seeded(seed_source="chunk")
@component
class FoliagePlacement:
    """Procedural foliage placement configuration."""
    
    # Type reference
    foliage_type_id: Annotated[str, Tracked] = ""
    
    # Placement
    density: Annotated[float, Tracked, Range(0, 10)] = 1.0
    min_spacing: Annotated[float, Tracked] = 2.0
    
    # Filters
    slope_range: Annotated[tuple, Tracked] = (0, 35)
    height_range: Annotated[tuple, Tracked] = None
    terrain_layers: Annotated[list[int], Tracked] = field(default_factory=list)
    
    # Noise modulation
    noise_type: Annotated[str, Tracked] = "perlin"
    noise_scale: Annotated[float, Tracked] = 20.0
    noise_threshold: Annotated[float, Tracked, Range(0, 1)] = 0.5
```

### Environment Volume Pattern

Volumes define regions with specific behavior.

```python
@trigger_volume(events=["on_enter", "on_exit", "on_overlap"], filter_tags=["player", "vehicle"])
@component
class GameplayVolume:
    """Base gameplay trigger volume."""
    
    # Bounds
    bounds_type: Annotated[str, Tracked] = "box"  # box, sphere, capsule
    bounds_size: Annotated[tuple, Tracked] = (100, 100, 100)
    
    # State
    is_active: Annotated[bool, Tracked] = True
    
    # Events
    @on_enter
    def handle_enter(self, actor):
        pass
    
    @on_exit
    def handle_exit(self, actor):
        pass

@environment_volume(volume_type="post_process", priority=10)
@component
class PostProcessVolume:
    """Post-process effect region."""
    
    bounds: Annotated[tuple, Tracked] = (100, 100, 100)
    priority: Annotated[int, Tracked] = 0
    blend_radius: Annotated[float, Tracked] = 100.0
    
    # Settings
    exposure: Annotated[float, Tracked, Range(-5, 5)] = 0.0
    saturation: Annotated[float, Tracked, Range(0, 2)] = 1.0
    contrast: Annotated[float, Tracked, Range(0, 2)] = 1.0
    bloom_intensity: Annotated[float, Tracked, Range(0, 5)] = 1.0

@weather_zone(climate="temperate", blend_radius=500.0)
@component
class WeatherZone:
    """Regional weather configuration."""
    
    bounds: Annotated[tuple, Tracked] = (1000, 1000, 500)
    priority: Annotated[int, Tracked] = 0
    
    # Climate
    climate_type: Annotated[str, Tracked] = "temperate"
    base_temperature: Annotated[float, Tracked] = 20.0
    humidity: Annotated[float, Tracked, Range(0, 1)] = 0.5
    
    # Weather probabilities
    clear_chance: Annotated[float, Tracked, Range(0, 1)] = 0.5
    rain_chance: Annotated[float, Tracked, Range(0, 1)] = 0.3
    storm_chance: Annotated[float, Tracked, Range(0, 1)] = 0.1
```

### Water Body Pattern

```python
@water_body(type="ocean", wave_source="wind")
@reflection_probe(capture_mode="realtime", resolution=1024, update_rate=0.033)
@component
class OceanBody:
    """Ocean water body with waves."""
    
    # Bounds
    surface_height: Annotated[float, Tracked] = 0.0
    depth: Annotated[float, Tracked] = 100.0
    
    # Waves
    wave_amplitude: Annotated[float, Tracked, Range(0, 10)] = 1.0
    wave_frequency: Annotated[float, Tracked] = 0.5
    wave_direction: Annotated[tuple, Tracked] = (1, 0)
    
    # Visual
    water_color: Annotated[tuple, Tracked] = (0.1, 0.3, 0.5)
    scatter_color: Annotated[tuple, Tracked] = (0.2, 0.5, 0.4)
    opacity: Annotated[float, Tracked, Range(0, 1)] = 0.8
    
    # Physics
    buoyancy_density: Annotated[float, Tracked] = 1.0
    drag_coefficient: Annotated[float, Tracked] = 0.5

@water_body(type="river", wave_source="flow")
@component
class RiverBody:
    """River water body with flow."""
    
    # Spline path
    spline_points: Annotated[list[tuple], Tracked] = field(default_factory=list)
    width: Annotated[float, Tracked] = 10.0
    depth: Annotated[float, Tracked] = 2.0
    
    # Flow
    flow_speed: Annotated[float, Tracked] = 2.0
    flow_direction: Annotated[int, Tracked] = 1  # Along spline
    
    # Terrain interaction
    carve_terrain: Annotated[bool, Tracked] = True
    shore_blend: Annotated[float, Tracked] = 5.0
```

### Time of Day / Weather Pattern

```python
@state_machine(
    initial="clear",
    states={"clear", "cloudy", "rain", "storm", "fog", "snow"},
    transitions={
        "clear": {"cloudy", "fog"},
        "cloudy": {"clear", "rain", "storm"},
        "rain": {"cloudy", "storm"},
        "storm": {"rain", "cloudy"},
        "fog": {"clear", "cloudy"},
        "snow": {"cloudy", "clear"},
    }
)
@resource
class WeatherStateMachine:
    """Global weather state machine."""
    
    current: Annotated[str, Observable, Tracked] = "clear"
    target: Annotated[str, Tracked] = "clear"
    blend_factor: Annotated[float, Interpolated, Range(0, 1)] = 1.0
    
    # Current weather parameters (interpolated during transitions)
    precipitation: Annotated[float, Interpolated, Range(0, 1)] = 0.0
    wind_speed: Annotated[float, Interpolated, Range(0, 50)] = 5.0
    cloud_density: Annotated[float, Interpolated, Range(0, 1)] = 0.0
    fog_density: Annotated[float, Interpolated, Range(0, 1)] = 0.0
    
    @on_enter(state="rain")
    def start_rain(self):
        # Enable rain particles, wet surfaces
        pass
    
    @on_exit(state="rain")
    def stop_rain(self):
        # Disable rain, start drying
        pass

@resource
class TimeOfDayController:
    """Global time of day controller."""
    
    # Time
    time_hours: Annotated[float, Interpolated, Tracked, Range(0, 24)] = 12.0
    day_count: Annotated[int, Tracked] = 0
    time_scale: Annotated[float, Tracked] = 1.0  # 1 = real-time, 60 = 1 hour/minute
    
    # Computed sun position
    sun_azimuth: Annotated[float, Tracked] = 0.0
    sun_elevation: Annotated[float, Tracked] = 45.0
    
    # Lighting (interpolated)
    sun_color: Annotated[tuple, Interpolated, Tracked] = (1.0, 0.95, 0.9)
    sun_intensity: Annotated[float, Interpolated, Range(0, 10)] = 1.0
    ambient_color: Annotated[tuple, Interpolated, Tracked] = (0.5, 0.6, 0.7)
    ambient_intensity: Annotated[float, Interpolated, Range(0, 2)] = 0.3
    
    # Sky
    sky_color_zenith: Annotated[tuple, Interpolated, Tracked] = (0.3, 0.5, 0.9)
    sky_color_horizon: Annotated[tuple, Interpolated, Tracked] = (0.7, 0.8, 0.9)
```

### World Query Pattern

```python
from trinity.decorators.bridges_caching import cached

@cached(ttl=0.1, scope="frame")  # Cache for one frame
@component
class SpatialQuery:
    """Cached spatial query results."""
    
    query_type: Annotated[str, Tracked] = "overlap"
    bounds: Annotated[tuple, Tracked] = (0, 0, 0, 10, 10, 10)
    filter_tags: Annotated[list, Tracked] = field(default_factory=list)
    
    # Results (transient, computed)
    results: Annotated[list, Transient] = field(default_factory=list)
    
    def execute(self, spatial_index) -> list:
        """Execute query against spatial index."""
        return spatial_index.query(self.bounds, self.filter_tags)
```

---

## Part IV: Engine/World Directory Structure

```
engine/world/
+-- __init__.py              # Public API exports
+-- partition/
|   +-- __init__.py
|   +-- grid.py              # WorldGrid component
|   +-- cell.py              # StreamingCell component
|   +-- streaming.py         # Streaming manager system
|   +-- hlod.py              # HLOD generation and management
+-- terrain/
|   +-- __init__.py
|   +-- patch.py             # TerrainPatch component
|   +-- layer.py             # TerrainMaterialLayer component
|   +-- heightfield.py       # Heightfield data structures
|   +-- sculpting.py         # Terrain modification operations
|   +-- lod.py               # Terrain LOD system
+-- foliage/
|   +-- __init__.py
|   +-- types.py             # Foliage type definitions (tree, shrub, grass)
|   +-- placement.py         # Procedural placement component
|   +-- instances.py         # Instance management (HISM)
|   +-- grass.py             # Grass-specific system
+-- environment/
|   +-- __init__.py
|   +-- volumes.py           # Environment volume components
|   +-- weather.py           # Weather state machine
|   +-- time_of_day.py       # TOD controller
|   +-- sky.py               # Sky atmosphere system
|   +-- lighting.py          # Environment lighting
+-- pcg/
|   +-- __init__.py
|   +-- scatter.py           # Scatter placement system
|   +-- rules.py             # Placement rules and filters
|   +-- noise.py             # Noise generation utilities
|   +-- seeds.py             # Seed management
+-- queries/
|   +-- __init__.py
|   +-- spatial.py           # Spatial queries (raycast, overlap)
|   +-- terrain.py           # Terrain queries (height, normal, layer)
|   +-- navigation.py        # Navigation queries
+-- hlod/
|   +-- __init__.py
|   +-- generator.py         # HLOD mesh generation
|   +-- layers.py            # HLOD layer management
|   +-- transitions.py       # LOD transition handling
```

---

## Part V: Implementation Checklist

### Phase 1: World Partition Foundation

- [ ] `partition/grid.py` - WorldGrid component with spatial indexing
- [ ] `partition/cell.py` - StreamingCell with state machine
- [ ] `partition/streaming.py` - Streaming manager (load/unload logic)
- [ ] `partition/hlod.py` - HLOD proxy management

### Phase 2: Terrain System

- [ ] `terrain/heightfield.py` - Heightfield data structure with compression
- [ ] `terrain/patch.py` - TerrainPatch component with LOD
- [ ] `terrain/layer.py` - Material layer blending
- [ ] `terrain/lod.py` - Terrain LOD selection and stitching
- [ ] `terrain/sculpting.py` - Height modification operations

### Phase 3: Foliage System

- [ ] `foliage/types.py` - TreeType, ShrubType, GrassType components
- [ ] `foliage/placement.py` - FoliagePlacement with procedural rules
- [ ] `foliage/instances.py` - Hierarchical instanced mesh management
- [ ] `foliage/grass.py` - Procedural grass generation

### Phase 4: Environment Volumes

- [ ] `environment/volumes.py` - GameplayVolume, PostProcessVolume, etc.
- [ ] `environment/weather.py` - WeatherStateMachine resource
- [ ] `environment/time_of_day.py` - TimeOfDayController resource
- [ ] `environment/sky.py` - Procedural sky atmosphere
- [ ] `environment/lighting.py` - Sun, moon, ambient lighting

### Phase 5: Procedural Generation

- [ ] `pcg/noise.py` - Perlin, simplex, worley noise generators
- [ ] `pcg/scatter.py` - Scatter placement (poisson disk, etc.)
- [ ] `pcg/rules.py` - Placement rules (slope, height, layer filters)
- [ ] `pcg/seeds.py` - Deterministic seed management

### Phase 6: World Queries

- [ ] `queries/spatial.py` - Raycast, sweep, overlap queries
- [ ] `queries/terrain.py` - Height, normal, layer weight queries
- [ ] `queries/navigation.py` - NavMesh queries

### Phase 7: HLOD System

- [ ] `hlod/generator.py` - Mesh merging and simplification
- [ ] `hlod/layers.py` - Multi-level HLOD management
- [ ] `hlod/transitions.py` - Distance-based transitions

### Phase 8: Integration

- [ ] Wire TrackedDescriptor changes to streaming system
- [ ] Wire IndexedDescriptor for fast cell/patch lookup
- [ ] Wire AsyncLoadDescriptor for async chunk loading
- [ ] Wire InterpolatedDescriptor for TOD/weather blending
- [ ] Wire ObservableDescriptor for weather -> visual updates
- [ ] Wire Foundation Tracker for undo/redo in editor
- [ ] Wire Foundation Mirror for world inspector

---

## Quick Reference

### Descriptor Choice Guide

| Need | Descriptor | Example |
|------|-----------|---------|
| Trigger streaming on change | `Tracked` | Cell state, load priority |
| Fast coordinate lookup | `Indexed` | Grid position, cell lookup |
| Large sparse data | `Sparse` | Heightfield with empty areas |
| Memory-efficient storage | `Compressed` | Terrain heights, weight maps |
| Deferred loading | `Lazy` | LOD data, chunk contents |
| Async streaming | `Async` | Cell actors, foliage instances |
| Non-serializable runtime | `Transient` | Cached meshes, render data |
| Numeric constraints | `Range` | Height range, density limits |
| Change notifications | `Observable` | Weather -> visuals |
| Smooth transitions | `Interpolated` | TOD values, weather blend |
| Save compatibility | `Versioned` | Terrain modifications |
| Bulk operations | `Batched` | Mass foliage placement |

### Decorator Choice Guide

| Need | Decorator | Module |
|------|-----------|--------|
| Terrain patch | `@terrain_patch` | (new in world.py) |
| Foliage type | `@foliage_type` | `world_building.py` |
| Procedural placement | `@procedural_placement` | `world_building.py` |
| Level streaming | `@level_instance` | `world_building.py` |
| Water body | `@water_body` | `world_building.py` |
| NavMesh mod | `@navmesh_modifier` | `world_building.py` |
| Trigger volume | `@trigger_volume` | `world_building.py` |
| LOD config | `@lod` | `lod_streaming.py` |
| Streaming support | `@streamable` | `lod_streaming.py` |
| Chunk definition | `@chunk` | `lod_streaming.py` |
| Load priority | `@loading_priority` | `lod_streaming.py` |
| Unload behavior | `@unloadable` | `lod_streaming.py` |
| Spatial indexing | `@spatial` | `spatial.py` |
| World partition | `@partitioned` | `spatial.py` |
| Deterministic gen | `@seeded` | `procedural.py` |
| Procedural gen | `@procedural` | `procedural.py` |
| Gen constraints | `@constraint` | `procedural.py` |
| GI contribution | `@gi_contributor` | `rendering.py` |
| Shadow casting | `@shadow_caster` | `rendering.py` |
| Reflection probe | `@reflection_probe` | `rendering.py` |
| Spawn/despawn hooks | `@on_spawn`, `@on_despawn` | `lifecycle.py` |
| State machine | `@state_machine` | `state_machine.py` |
| State hooks | `@on_enter`, `@on_exit` | `state_machine.py` |

### Foundation Integration Points

| System | World Use |
|--------|-----------|
| Registry | Cell/patch type lookup, instance tracking |
| Tracker | Dirty flags, streaming triggers, undo/redo |
| EventLog | Load/unload events, weather transitions |
| Mirror | World introspection in editor |
| Bridge | ShellLang access to world for debugging |

### Stack Choice Guide

| Scenario | Stack | Purpose |
|----------|-------|---------|
| Basic terrain | `terrain_component()` | Terrain patch with LOD |
| World cell | `world_cell()` | Streaming partition cell |
| Tree foliage | `foliage_instance(foliage_type="tree")` | LOD tree instance |
| Procedural vegetation | `procedural_foliage()` | Auto-placed plants |
| Trigger zone | `environment_trigger()` | Enter/exit events |
| Water surface | `water_surface()` | Ocean/lake/river |
| Open world entity | `open_world_entity()` | Full streaming support |
| Streaming level | `streaming_level_instance()` | Sub-level loading |

---

## References

- `docs/TRINITY_LATEST.md` - Full Trinity Pattern specification
- `docs/GAME_ENGINE_INTEGRATION.md` - Trinity <-> Foundation integration
- `docs/GAME_ENGINE_INTEGRATION_TODO.md` - Section 9 (World Layer)
- `DIAGRAMS/ARCHITECTURE_WORLD.md` - World layer architecture
- `trinity/decorators/world_building.py` - World building decorators
- `trinity/decorators/lod_streaming.py` - LOD and streaming decorators
- `trinity/decorators/spatial.py` - Spatial indexing decorators
- `trinity/decorators/procedural.py` - Procedural generation decorators
- `trinity/decorators/rendering.py` - Rendering decorators
- `trinity/decorators/builtin_stacks/streaming.py` - Streaming stacks
- `trinity/decorators/builtin_stacks/composite.py` - Composite stacks
- `trinity/descriptors/` - All descriptor implementations
