# ENGINE_WORLD - Clarification and Design Rationale

## Philosophical Framing

The world subsystem embodies the principle that **open-world engines are streaming problems first, rendering problems second**. Every design decision flows from this understanding:

1. **Spatial locality is king**: Data organized by world position, not by type
2. **Budget-driven execution**: Memory and frame time budgets are hard constraints, not suggestions
3. **Determinism enables debugging**: Every procedural result must be reproducible given the same seed

## Why Seven Subsystems?

The subsystem boundaries reflect natural data ownership:

| Subsystem | Owns | Queries |
|-----------|------|---------|
| Terrain | Heightfield, materials, holes | Height at point, normal, slope |
| Environment | Sky, weather, time, volumes | Current weather, sun position, volume at point |
| Foliage | Instance data, type definitions | Instances in bounds |
| HLOD | Proxy meshes, cluster hierarchy | LOD level at distance |
| Partition | Cell state, streaming queues | Cells in radius, load priority |
| PCG | Noise state, placement rules | Noise at point, scatter positions |
| Queries | Query execution, result caching | Raycast, pathfinding, overlap |

Each subsystem is a bounded context with clear input/output contracts. Cross-subsystem communication happens through queries, not direct data access.

## Key Design Decisions

### 1. Protocol-Based Abstractions

All subsystems define `Protocol` classes for their external interfaces:
- `TerrainInterface`: Height/normal/layer queries
- `SpatialIndex`: Raycast/overlap registration
- `NavMesh`: Pathfinding operations

This allows mock implementations for testing and future Rust implementations to satisfy the same contracts.

### 2. State Machine Cells (Partition)

Streaming cells use explicit state machines:
```
UNLOADED -> LOADING -> LOADED -> ACTIVATED
                  \              /
                   <- UNLOADING <-
```

This prevents race conditions from async load/unload operations and makes debugging streaming issues tractable.

### 3. Priority-Based Streaming

The streaming system uses a priority queue, not FIFO:
- Closer cells load first
- High-priority sources (player) override camera-only sources
- Hysteresis prevents thrashing at cell boundaries

### 4. Quadric Error Metrics (HLOD)

Mesh simplification uses QEM rather than simpler algorithms because:
- Preserves visual silhouettes better than uniform decimation
- Handles non-manifold meshes gracefully
- Well-understood performance characteristics

### 5. Deterministic PCG

All procedural algorithms are seeded:
- `ChunkSeed`: Position-based, deterministic per chunk
- `LayerSeed`: Named seeds within chunks
- `InstanceSeed`: Per-object seeds within layers

This hierarchy means adding/removing foliage in one chunk doesn't affect another.

### 6. Query Caching (Navigation)

A* pathfinding results are cached:
- FIFO eviction (simple, predictable)
- Key: (start, end) tuple
- Size: 100 entries default

Future: Consider LRU or access-frequency tracking for better hit rates.

## Why Pure Python?

The world subsystem is implemented in pure Python for:
1. **Portability**: No compilation required, works on any platform with Python 3.13
2. **Debuggability**: Standard Python tooling works without special configuration
3. **Iteration Speed**: Changes don't require recompilation
4. **Bridge Flexibility**: Can migrate hot paths to Rust incrementally

Performance-critical paths (noise generation, mesh simplification, A* pathfinding) are candidates for Rust bridges, but the Python implementation serves as both reference and fallback.

## Integration Philosophy

The world subsystem integrates with rendering and physics through **data generation, not direct coupling**:

- Foliage generates instance buffers; rendering consumes them
- HLOD generates proxy meshes; rendering uploads them
- Terrain generates collision shapes; physics registers them
- Partition manages load state; rendering queries visibility

This separation means the world subsystem can be tested without GPU/physics dependencies.

## What This Subsystem Does NOT Handle

- **Rendering**: No draw calls, no shaders, no GPU resource management
- **Physics Simulation**: No rigid body dynamics, no constraint solving
- **Audio Playback**: No sound mixing, no 3D positioning (volumes only define zones)
- **Gameplay Logic**: No game rules, no entity behavior

These are handled by sibling subsystems (`engine/rendering`, `engine/physics`, `engine/audio`, etc.).

## Terminology Alignment

| TRINITY Term | Unreal Equivalent | Unity Equivalent |
|--------------|-------------------|------------------|
| StreamingCell | World Partition Cell | Addressable Asset |
| HLODCluster | HLOD Cluster | LOD Group |
| TerrainPatch | Landscape Component | Terrain Tile |
| DataLayer | Data Layer | Scene Layer |
| ProceduralPlacer | PCG Graph | - |
| WeatherStateMachine | - | - |
| TimeOfDayController | Sky Atmosphere | - |

## Open Questions (To Be Resolved)

1. **Async Streaming**: Current implementation is synchronous. Should we add async I/O via `asyncio`?
2. **GPU Impostor Capture**: Current HLOD impostor generation is CPU-based. When do we add GPU capture?
3. **Hierarchical Pathfinding**: Current A* works on flat navmesh. Large worlds may need HPA* or similar.
4. **L-System / WFC**: PCG module mentions these but they're not implemented. Priority?
