# ENGINE_WORLD - Phase 2 TODO: Subsystem Integration

## Task List

### T-W2-001: Terrain-Foliage Integration Test
**Description**: Validate foliage placement correctly queries terrain data
**File**: `tests/integration/world/test_terrain_foliage.py`
**Dependencies**: Phase 1 unit tests passing
**Acceptance Criteria**:
- [ ] Create test terrain with known heightfield (simple gradient)
- [ ] Create test terrain with multiple material layers
- [ ] Place grass using ProceduralPlacer with slope filter
- [ ] Verify grass positions have slope within filter range
- [ ] Place trees using height filter
- [ ] Verify tree positions have height within filter range
- [ ] Place shrubs using layer filter
- [ ] Verify shrubs only appear on matching material layer
- [ ] Test noise filter with deterministic seed
- [ ] Verify noise filter produces reproducible placement

### T-W2-002: HLOD-Partition Lifecycle Integration
**Description**: Validate HLOD generation syncs with partition cell lifecycle
**File**: `tests/integration/world/test_hlod_partition.py`
**Dependencies**: Phase 1 unit tests passing
**Acceptance Criteria**:
- [ ] Create WorldGrid with 4x4 cells
- [ ] Create HLODLayerManager linked to partition
- [ ] Load cell (0,0) - verify HLOD cell created
- [ ] Load cells (0,1), (1,0), (1,1) - verify 4 HLOD cells
- [ ] Verify HLOD cluster contains all 4 cells
- [ ] Unload cell (0,0) - verify HLOD cell removed
- [ ] Verify HLOD cluster updated to 3 cells
- [ ] Reload cell (0,0) - verify HLOD cell recreated
- [ ] Verify no orphan HLOD cells (all map to streaming cells)

### T-W2-003: HLOD Mesh Simplification Integration
**Description**: Validate HLOD mesh simplification works with real geometry
**File**: `tests/integration/world/test_hlod_partition.py` (same file)
**Dependencies**: T-W2-002
**Acceptance Criteria**:
- [ ] Create test meshes with known triangle counts
- [ ] Add meshes to streaming cell
- [ ] Load cell and trigger HLOD generation
- [ ] Verify simplified mesh has fewer triangles
- [ ] Verify simplification ratio matches configuration
- [ ] Verify simplified mesh maintains bounding box (roughly)

### T-W2-004: Environment-Terrain Lighting Integration
**Description**: Validate environment lighting parameters reach terrain
**File**: `tests/integration/world/test_environment_terrain.py`
**Dependencies**: Phase 1 unit tests passing
**Acceptance Criteria**:
- [ ] Create TimeOfDayController
- [ ] Create SunLight linked to TOD
- [ ] Create TerrainMaterial system
- [ ] Set TOD to noon - verify sun elevation ~90 degrees
- [ ] Verify terrain receives correct sun direction
- [ ] Set TOD to sunset - verify sun elevation ~0 degrees
- [ ] Verify terrain receives updated sun direction
- [ ] Set weather to cloudy - verify sun intensity reduced
- [ ] Verify terrain ambient boost matches overcast setting

### T-W2-005: Weather-Environment Integration
**Description**: Validate weather state machine drives environment parameters
**File**: `tests/integration/world/test_environment_terrain.py` (same file)
**Dependencies**: T-W2-004
**Acceptance Criteria**:
- [ ] Create WeatherStateMachine
- [ ] Set weather to CLEAR
- [ ] Verify sky parameters match clear preset
- [ ] Transition to CLOUDY
- [ ] Verify transition interpolates parameters over time
- [ ] Verify cloud coverage increases during transition
- [ ] Verify fog parameters update correctly
- [ ] Test invalid transition (CLEAR -> STORM directly)
- [ ] Verify invalid transition rejected

### T-W2-006: Spatial Query Integration
**Description**: Validate spatial queries work against foliage instances
**File**: `tests/integration/world/test_queries_integration.py`
**Dependencies**: Phase 1 unit tests passing
**Acceptance Criteria**:
- [ ] Create HISM with known instance positions
- [ ] Register HISM with SpatialIndex (mock or real)
- [ ] Query overlap at known instance position
- [ ] Verify instance returned in overlap results
- [ ] Query overlap at empty position
- [ ] Verify no instances returned
- [ ] Query with tag filter
- [ ] Verify only tagged instances returned

### T-W2-007: Terrain Query Integration
**Description**: Validate terrain queries work against real heightfield
**File**: `tests/integration/world/test_queries_integration.py` (same file)
**Dependencies**: Phase 1 unit tests passing
**Acceptance Criteria**:
- [ ] Create terrain with known heightfield (e.g., y = x * 0.1)
- [ ] Raycast from above terrain
- [ ] Verify hit position matches expected height
- [ ] Raycast from below terrain
- [ ] Verify no hit (ray starts inside terrain)
- [ ] Raycast parallel to terrain
- [ ] Verify no hit or correct edge case handling
- [ ] Query area height statistics
- [ ] Verify min/max/avg match expected values

### T-W2-008: Navigation Query Integration
**Description**: Validate pathfinding works with stub navmesh
**File**: `tests/integration/world/test_queries_integration.py` (same file)
**Dependencies**: Phase 1 unit tests passing
**Acceptance Criteria**:
- [ ] Create StubNavMesh with 10x10 grid
- [ ] Add blocked cells creating a maze
- [ ] Query path from (0,0) to (9,9)
- [ ] Verify path avoids blocked cells
- [ ] Verify path length is optimal (or near-optimal)
- [ ] Query path to unreachable goal
- [ ] Verify partial path returned (closest reachable)
- [ ] Test path caching
- [ ] Verify second query returns cached result

### T-W2-009: PCG-Terrain Noise Integration
**Description**: Validate PCG noise used in terrain auto-layer rules
**File**: `tests/integration/world/test_pcg_integration.py`
**Dependencies**: Phase 1 unit tests passing
**Acceptance Criteria**:
- [ ] Create NoiseGenerator with known seed
- [ ] Create AutoLayerRule using noise threshold
- [ ] Apply rule to terrain weightmap
- [ ] Sample noise at multiple positions
- [ ] Verify layer applied where noise > threshold
- [ ] Verify layer not applied where noise < threshold
- [ ] Change seed and reapply
- [ ] Verify different result with different seed

### T-W2-010: PCG-Foliage Noise Integration
**Description**: Validate PCG noise used in foliage placement filtering
**File**: `tests/integration/world/test_pcg_integration.py` (same file)
**Dependencies**: Phase 1 unit tests passing
**Acceptance Criteria**:
- [ ] Create NoiseGenerator with known seed
- [ ] Create NoiseFilter for placement
- [ ] Generate placement positions with filter
- [ ] Verify positions cluster in high-noise regions
- [ ] Verify low-noise regions have fewer placements
- [ ] Reset generator and regenerate
- [ ] Verify identical positions (determinism)

### T-W2-011: Volume-Environment Integration
**Description**: Validate volume system integrates with environment
**File**: `tests/integration/world/test_environment_terrain.py` (extend)
**Dependencies**: Phase 1 unit tests passing
**Acceptance Criteria**:
- [ ] Create PostProcessVolume with settings
- [ ] Query volume at point inside
- [ ] Verify settings returned
- [ ] Query volume at point outside
- [ ] Verify no settings (or fallback)
- [ ] Create overlapping volumes with different priorities
- [ ] Verify higher priority volume wins
- [ ] Test blending at volume boundaries

---

## Integration Test Matrix

| Test | Subsystems | Protocol Used |
|------|------------|---------------|
| T-W2-001 | Terrain, Foliage, PCG | TerrainInterface |
| T-W2-002 | HLOD, Partition | Cell lifecycle callbacks |
| T-W2-003 | HLOD, Partition | Mesh simplification |
| T-W2-004 | Environment, Terrain | Lighting parameters |
| T-W2-005 | Environment (internal) | Weather state machine |
| T-W2-006 | Queries, Foliage | SpatialIndex |
| T-W2-007 | Queries, Terrain | TerrainSystem |
| T-W2-008 | Queries (Navigation) | NavMesh |
| T-W2-009 | PCG, Terrain | Noise sampling |
| T-W2-010 | PCG, Foliage | Noise filtering |
| T-W2-011 | Environment (Volumes) | VolumeManager |

---

## Summary

| Category | Tasks | Estimated Effort |
|----------|-------|------------------|
| Terrain-Foliage | 1 | High |
| HLOD-Partition | 2 | High |
| Environment-Terrain | 2 | Medium |
| Query Integration | 3 | High |
| PCG Integration | 2 | Medium |
| Volume Integration | 1 | Low |
| **Total** | **11** | |
