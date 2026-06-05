# Investigation: engine/world/foliage

## Summary

The foliage system contains substantial, production-ready implementations for grass rendering, hierarchical instanced static meshes (HISM), procedural placement, and foliage type definitions. This is a REAL IMPLEMENTATION with working algorithms for terrain-based placement, frustum culling, LOD management, chunk streaming, and wind animation support. The code includes proper validation, spatial organization, and GPU buffer generation.

## Files

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 131 | Complete | Full module exports with docstrings |
| `constants.py` | 248 | Complete | Comprehensive constants for LOD, culling, density, wind |
| `grass.py` | 739 | Full Implementation | ProceduralGrass, LandscapeGrass, GrassRenderer with chunk streaming |
| `instances.py` | 960 | Full Implementation | HISM pattern with frustum culling, LOD, spatial clustering |
| `placement.py` | 735 | Full Implementation | ProceduralPlacer with noise, slope/height/layer filtering |
| `types.py` | 528 | Full Implementation | FoliageType hierarchy: Tree, Shrub, Grass, Rock, Debris |

**Total: 3,341 lines of implementation code**

## Foliage Components

### Type System (`types.py`)
- `FoliageCategory`: GRASS, SHRUB, TREE, ROCK, DEBRIS
- `FoliageType`: Base class with LOD meshes, cull distance, wind, density
- `TreeType`: Trunk/canopy meshes, canopy sway, branch detail distance
- `ShrubType`: Berry/flower support with mesh IDs
- `GrassType`: Blade width/height/curve/bend, color gradients, blades per instance
- `RockType`: Moss coverage, weathering, embed depth
- `DebrisType`: Decay rate, physics, scatter
- `FoliageTypeRegistry`: Type storage with category queries
- `@foliage_type` decorator: Trinity Pattern for declarative definitions

### Placement System (`placement.py`)
- `PlacementRule`: Slope/height/terrain layer filtering, noise threshold
- `PlacementResult`: Position, rotation, scale with 4x4 transform matrix calculation
- `TerrainInterface`: Protocol for height, normal, layer, water, road queries
- `Bounds`: AABB with intersection, containment tests
- `NoiseGenerator`: Hash-based deterministic noise with bilinear interpolation
- `ProceduralPlacer`: Grid-based placement with jitter and rule evaluation
- `FoliagePlacement`: Configuration combining type, rules, density
- `ManualPlacement`: Instance CRUD for editor-placed foliage
- `@procedural_placement` decorator: Rule-based placement definitions

### Instance Management (`instances.py`)
- `FoliageInstance`: Transform, visibility, LOD level per instance
- `Frustum`: 6-plane frustum with point/sphere/AABB containment tests
- `FoliageCluster`: Spatial grouping with per-cluster culling
- `HierarchicalInstancedMesh`: HISM pattern with cluster-based spatial organization
- `FoliageManager`: Multi-type HISM coordination
- `BatchedDescriptor`: Bulk add/remove operations

### Grass System (`grass.py`)
- `GrassSettings`: Density scale, wind sway, alpha cutoff, cull/fade distance
- `GrassInstance`: Blade position, rotation, height, width, bend, color blend
- `GrassChunk`: Streaming unit with bounds, visibility, instance buffer
- `ProceduralGrass`: Terrain-aware blade generation with layer/slope/water checks
- `LandscapeGrass`: Chunk-based streaming with camera-driven loading/unloading
- `GrassRenderer`: Wind time, shader parameters, render data preparation

## Implementation

- **Real grass system?** YES - Complete procedural grass with terrain queries, blade variation, chunk streaming, density control, wind animation support
- **Real tree placement?** YES - TreeType with trunk/canopy meshes, collision, LOD, canopy sway parameters
- **Real instancing?** YES - Full HISM pattern with spatial clustering, per-cluster frustum culling, LOD transitions, instance buffers for GPU
- **Real wind simulation?** PARTIAL - Wind time accumulation and shader parameter preparation; actual WGSL shaders not in this directory

## Verdict

**REAL IMPLEMENTATION**

This is a substantial, production-quality foliage system with:
1. Complete type hierarchy with category-specific attributes
2. Working procedural placement with terrain integration
3. Full HISM pattern for efficient instanced rendering
4. Chunk-based grass streaming with camera-driven updates
5. Frustum culling at cluster and instance level
6. LOD system with distance-based mesh selection
7. GPU buffer generation for rendering pipeline
8. Manual placement support for editor workflows

## Evidence

### Procedural Grass Generation (grass.py:247-317)
```python
def generate_for_chunk(
    self,
    terrain: TerrainInterface,
    chunk_bounds: Bounds,
    grass_type: GrassType,
) -> List[GrassInstance]:
    instances = []
    density = grass_type.density * self._settings.density_scale
    spacing = 1.0 / math.sqrt(density)
    
    x = chunk_bounds.min_x
    while x <= chunk_bounds.max_x:
        z = chunk_bounds.min_z
        while z <= chunk_bounds.max_z:
            jitter_x = (self._hash_position(x, z, 0) - 0.5) * spacing
            jitter_z = (self._hash_position(x, z, 1) - 0.5) * spacing
            px = x + jitter_x
            pz = z + jitter_z
            
            if self.should_grow_grass(terrain, px, pz):
                py = terrain.get_height_at(px, pz)
                rotation = self._hash_position(px, pz, 2) * math.pi * 2
                instances.append(GrassInstance(
                    position=(px, py, pz),
                    rotation=rotation,
                    height=grass_type.blade_height * height_var,
                    ...
                ))
            z += spacing
        x += spacing
    return instances
```

### Frustum Culling (instances.py:361-386)
```python
def cull(self, frustum: Frustum) -> int:
    # First check if entire cluster is visible
    if not frustum.contains_bounds(self._bounds, self._min_y, self._max_y):
        for inst in self._instances:
            inst.visible = False
        self._visible_count = 0
        return 0
    
    # Check individual instances
    self._visible_count = 0
    for inst in self._instances:
        inst.visible = frustum.contains_point(inst.position)
        if inst.visible:
            self._visible_count += 1
    return self._visible_count
```

### LOD Distance Selection (types.py:111-124)
```python
def get_lod_level(self, distance: float) -> int:
    for i, lod_dist in enumerate(self.lod_distances):
        if distance < lod_dist:
            return i
    return len(self.lod_distances)
```

### Terrain-Aware Placement (placement.py:296-345)
```python
def evaluate_position(self, terrain, x, z, rule) -> bool:
    slope = self.get_slope_at(terrain, x, z)
    if slope < rule.slope_range[0] or slope > rule.slope_range[1]:
        return False
    if rule.height_range is not None:
        height = terrain.get_height_at(x, z)
        if height < rule.height_range[0] or height > rule.height_range[1]:
            return False
    if rule.terrain_layers:
        layer = terrain.get_layer_at(x, z)
        if layer not in rule.terrain_layers:
            return False
    if rule.exclude_water and terrain.is_water_at(x, z):
        return False
    noise = self._noise.sample(x, z, rule.noise_scale)
    if noise < rule.noise_threshold:
        return False
    return True
```
