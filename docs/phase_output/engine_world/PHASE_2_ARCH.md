# ENGINE_WORLD - Phase 2 Architecture: Subsystem Integration

## Phase Overview

Phase 2 validates that the seven world subsystems integrate correctly. Each subsystem was designed to query others through Protocol interfaces; this phase verifies those integrations work in practice.

## Architecture Decisions

### ADR-W2-001: Terrain-Foliage Integration

**Context**: Foliage placement queries terrain for height, normal, slope, and layer information via `TerrainInterface` protocol.

**Decision**: Create integration test that places foliage using real terrain data, validates placement respects slope/height/layer rules.

**Implementation**:
```
Terrain (heightfield) ──┐
                        │ get_height_at()
                        │ get_normal_at()
                        │ get_layer_at()
                        ▼
               ProceduralPlacer
                        │
                        ▼
               FoliageInstances
```

**Validation**:
- Grass does not appear on slopes > configured max
- Trees do not appear below water level
- Layer-restricted foliage only appears on matching layers

### ADR-W2-002: HLOD-Partition Integration

**Context**: HLOD cells must align with partition streaming cells. HLOD proxy meshes must be generated when cells reach LOADED state.

**Decision**: Create integration test that loads partition cells, triggers HLOD generation, validates proxy meshes exist for loaded cells only.

**Implementation**:
```
StreamingCell ──┐
(state=LOADED)  │ on_loaded callback
                ▼
         HLODGenerator
                │
                ▼
         HLODCell (proxy mesh)
                │
                ▼
         HLODLayerManager (tracks all cells)
```

**Validation**:
- HLOD cells exist only for LOADED/ACTIVATED streaming cells
- Unloading a cell removes its HLOD data
- HLOD cluster hierarchy updates when cells load/unload

### ADR-W2-003: Environment-Terrain Integration

**Context**: Environment lighting affects terrain rendering. Time-of-day sun position determines directional light for terrain shadows. Weather affects ambient lighting.

**Decision**: Integration validates that terrain material system receives correct lighting parameters from environment.

**Implementation**:
```
TimeOfDayController ──┐
                      │ sun_direction, sun_color
                      ▼
              SunLight (updates from TOD)
                      │
                      ▼
              TerrainMaterial (receives light params)
                      │
                      ▼
              Shader Parameters (for GPU)
```

**Validation**:
- Sun direction changes over simulated day
- Terrain receives consistent sun direction/color
- Weather cloud coverage dims sun intensity

### ADR-W2-004: Query-Subsystem Integration

**Context**: Query systems must work against real spatial data from terrain, foliage, and navigation.

**Decision**: Create integration tests for each query type against real subsystem data.

**Integrations**:
| Query Type | Data Source | Protocol |
|------------|-------------|----------|
| Terrain Raycast | Heightfield | `TerrainSystem` |
| Spatial Overlap | Foliage instances | `SpatialIndex` |
| Path Query | NavMesh from terrain | `NavMesh` |

### ADR-W2-005: PCG-Terrain Integration

**Context**: PCG noise queries are used for terrain material auto-rules and foliage density variation.

**Decision**: Validate that PCG noise generators can be used in terrain material auto-layer rules and foliage placement noise filters.

**Implementation**:
```
NoiseGenerator (PCG) ──┐
                       │ sample()
                       ▼
              AutoLayerRule (terrain)
                       │
                       ▼
              WeightMap update
                       
NoiseGenerator (PCG) ──┐
                       │ sample()
                       ▼
              NoiseFilter (foliage placement)
                       │
                       ▼
              Placement decision
```

## Component Interaction Matrix

```
           ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
           │ Terrain │  Env    │ Foliage │  HLOD   │Partition│   PCG   │ Queries │
┌──────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ Terrain  │    -    │ receives│ provides│ provides│    -    │ uses    │ provides│
│          │         │ lighting│ height  │ meshes  │         │ noise   │ height  │
├──────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ Env      │ provides│    -    │    -    │    -    │    -    │    -    │    -    │
│          │ lighting│         │         │         │         │         │         │
├──────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ Foliage  │ queries │    -    │    -    │ provides│ responds│ uses    │ provides│
│          │ terrain │         │         │instances│ to load │ noise   │ spatial │
├──────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ HLOD     │ consumes│    -    │consumes │    -    │ responds│    -    │    -    │
│          │ meshes  │         │instances│         │ to load │         │         │
├──────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│Partition │ triggers│    -    │ triggers│ triggers│    -    │    -    │    -    │
│          │ load    │         │ load    │ gen     │         │         │         │
├──────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ PCG      │    -    │    -    │    -    │    -    │    -    │    -    │    -    │
│          │         │         │         │         │         │         │         │
├──────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ Queries  │ queries │    -    │ queries │    -    │    -    │    -    │    -    │
│          │ terrain │         │ spatial │         │         │         │         │
└──────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

## Integration Test Organization

```
tests/integration/world/
├── test_terrain_foliage.py     # T-W2-001
├── test_hlod_partition.py      # T-W2-002
├── test_environment_terrain.py # T-W2-003
├── test_queries_integration.py # T-W2-004 (all query types)
└── test_pcg_integration.py     # T-W2-005
```

## Success Metrics

| Integration | Validation | Metric |
|-------------|------------|--------|
| Terrain-Foliage | Placement respects rules | 100% of placed instances pass validation |
| HLOD-Partition | Lifecycle sync | HLOD cells match streaming cell count |
| Environment-Terrain | Lighting consistency | Sun direction matches across systems |
| Query-Subsystem | Query accuracy | Raycast hits match expected positions |
| PCG-Subsystems | Determinism | Same seed produces same results |
