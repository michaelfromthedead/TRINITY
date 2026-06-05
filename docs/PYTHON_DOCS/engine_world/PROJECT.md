# ENGINE_WORLD - Project Definition

## Scope

The `engine/world/` subsystem encompasses all spatial, environmental, and procedural systems required for open-world game engine functionality. This includes:

- **Terrain**: Heightfield management, LOD systems, material splatmapping, sculpting tools
- **Environment**: Sky rendering, weather state machines, time-of-day, lighting, volumes
- **Foliage**: Grass rendering, HISM instancing, procedural placement, type hierarchies
- **HLOD**: Hierarchical LOD with mesh simplification (QEM), impostor generation, cluster management
- **World Partition**: Streaming cells, data layers, spatial grid management, memory budgeting
- **PCG**: Noise generation (Perlin, Simplex, Worley), scatter algorithms, placement rules, seed management
- **Queries**: Spatial queries (raycast, sweep, overlap), terrain queries, navigation/pathfinding

## Goals

1. **Complete Integration Testing**: Validate all subsystems work together (terrain + foliage + HLOD + streaming)
2. **Performance Optimization**: Ensure GPU-driven rendering paths are efficient for large worlds
3. **API Stabilization**: Finalize public interfaces for Trinity Pattern decorator compatibility
4. **Rust Bridge Readiness**: Prepare subsystems for FFI exposure via PyO3 bindings

## Constraints

- **Python 3.13**: All code must be compatible with statically-linked Python 3.13 interpreter
- **No NumPy in Hot Paths**: Pure Python implementations preferred for cross-compilation
- **Determinism Required**: All procedural systems must be fully seeded and reproducible
- **Memory Budget**: Streaming system must respect configurable memory limits (default 256MB)
- **Frame Time Budget**: Per-frame processing limited to configurable ms (default 16ms)

## Code Statistics

| Subsystem | Lines | Files | Status |
|-----------|-------|-------|--------|
| Environment | 5,063 | 7 | REAL |
| Terrain | 6,563 | 9 | REAL |
| PCG | 4,232 | 6 | REAL |
| HLOD | 3,725 | 5 | REAL |
| Queries | 3,530 | 5 | REAL |
| Foliage | 3,341 | 6 | REAL |
| Partition | 2,569 | 6 | REAL |
| **TOTAL** | **29,023** | **44** | **REAL** |

## Acceptance Criteria

### Phase 1: Foundation
- [ ] All subsystem unit tests pass
- [ ] No stub implementations remain
- [ ] Constants centralized in each module's `constants.py`
- [ ] Type hints complete on all public APIs

### Phase 2: Integration
- [ ] Terrain-foliage integration validated (placement queries work)
- [ ] HLOD-partition integration validated (cells generate HLOD correctly)
- [ ] Environment-terrain integration validated (lighting affects terrain materials)
- [ ] Query systems work against all spatial data

### Phase 3: Optimization
- [ ] Streaming respects memory budget under stress test
- [ ] LOD transitions are smooth (no popping artifacts)
- [ ] Frustum culling operates at cluster granularity
- [ ] Procedural generation is deterministic across runs

### Phase 4: Bridge
- [ ] Critical hot paths identified for Rust migration
- [ ] PyO3 type stubs generated for bridged functions
- [ ] Performance benchmarks established for bridge targets

## Dependencies

### Internal
- `engine/core/math`: Vec3, AABB, Frustum primitives
- `engine/rendering`: GPU buffer generation, shader parameter setup
- `engine/physics`: Collision queries, physics volumes

### External
- `zlib`: Heightfield compression (via Python stdlib)
- None other - pure Python implementation

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Pure Python noise too slow | Batch operations, consider WASM fallback |
| HLOD impostor capture CPU-bound | Stub GPU capture, prioritize mesh simplification |
| Streaming thrash on fast movement | Hysteresis already implemented, tune parameters |
| Navmesh A* too slow for large maps | Path caching already implemented, add hierarchical pathfinding |
