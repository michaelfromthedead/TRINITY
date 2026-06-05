# ENGINE_WORLD - Phase 3 TODO: Performance Optimization

## Task List

### T-W3-001: Streaming Budget Stress Test
**Description**: Create stress test for partition streaming under rapid movement
**File**: `tests/performance/world/stress_streaming.py`
**Dependencies**: Phase 2 integration tests passing
**Acceptance Criteria**:
- [ ] Test simulates camera moving at 100 units/second
- [ ] Test runs for 60 simulated seconds
- [ ] Memory usage stays within budget (256MB default)
- [ ] Frame time budget respected (no tick exceeds 16ms)
- [ ] No cell state machine errors
- [ ] No orphan cells (unloaded but still referenced)

### T-W3-002: Streaming Budget Enforcement Fix
**Description**: Implement hard limits if stress test reveals issues
**File**: `engine/world/partition/streaming.py`
**Dependencies**: T-W3-001 identifies issues
**Acceptance Criteria**:
- [ ] Maximum concurrent loads enforced
- [ ] Queue overflow protection (oldest low-priority items dropped)
- [ ] Load pause when memory budget reached
- [ ] Resume when memory freed
- [ ] Stress test passes after fix

### T-W3-003: LOD Transition Validation
**Description**: Validate LOD transitions are smooth
**File**: `tests/performance/world/bench_hlod.py`
**Dependencies**: Phase 2 tests passing
**Acceptance Criteria**:
- [ ] Create test with camera moving toward HLOD object
- [ ] Capture LOD level at each distance
- [ ] Verify hysteresis prevents rapid switching
- [ ] Verify transition blend factor interpolates smoothly
- [ ] Document any remaining pop issues

### T-W3-004: Dither Pattern Validation
**Description**: Validate Bayer dither pattern produces smooth transitions
**File**: `tests/performance/world/bench_hlod.py` (extend)
**Dependencies**: T-W3-003
**Acceptance Criteria**:
- [ ] Render test scene with dither transition
- [ ] Verify pattern is 4x4 Bayer matrix
- [ ] Verify threshold progression is linear
- [ ] Verify no visible banding at slow transition

### T-W3-005: Frustum Culling Benchmark
**Description**: Benchmark cluster culling at different cluster sizes
**File**: `tests/performance/world/bench_foliage.py`
**Dependencies**: Phase 2 tests passing
**Acceptance Criteria**:
- [ ] Generate 100,000 foliage instances
- [ ] Test cluster sizes: 100, 250, 500, 1000
- [ ] Measure CPU time for full frustum cull
- [ ] Measure memory overhead per cluster
- [ ] Document optimal cluster size range

### T-W3-006: Instance Buffer Generation Benchmark
**Description**: Benchmark GPU buffer generation for large instance counts
**File**: `tests/performance/world/bench_foliage.py` (extend)
**Dependencies**: T-W3-005
**Acceptance Criteria**:
- [ ] Generate instance buffer for 10,000 instances
- [ ] Measure time for buffer creation
- [ ] Measure memory allocation
- [ ] Compare to target (< 10ms)
- [ ] Document if Rust bridge needed

### T-W3-007: Noise Batch Interface Implementation
**Description**: Add batch sampling to noise generators
**Files**: `engine/world/pcg/noise.py`
**Dependencies**: Phase 1 tests passing
**Acceptance Criteria**:
- [ ] Add `sample_batch(positions)` to NoiseGenerator base
- [ ] Implement for Perlin, Simplex, Worley, Value
- [ ] Batch implementation uses local variable caching
- [ ] Unit test validates batch == individual results
- [ ] Benchmark shows improvement over individual calls

### T-W3-008: Noise Batch Performance Benchmark
**Description**: Benchmark batch vs individual noise sampling
**File**: `tests/performance/world/bench_pcg.py`
**Dependencies**: T-W3-007
**Acceptance Criteria**:
- [ ] Benchmark 10,000 individual samples
- [ ] Benchmark 10,000 batch samples
- [ ] Measure speedup factor
- [ ] Target: batch at least 2x faster
- [ ] Document actual speedup

### T-W3-009: PCG Determinism Test Suite
**Description**: Validate determinism across all PCG algorithms
**File**: `tests/unit/world/test_pcg_determinism.py`
**Dependencies**: Phase 1 tests passing
**Acceptance Criteria**:
- [ ] Test all noise types (Perlin, Simplex, Worley, Value, White, Fractal)
- [ ] Test all scatter algorithms (Poisson, Grid, Jittered, Clustered, Organic)
- [ ] Generate output, hash, regenerate, compare
- [ ] All hashes match across 10 runs
- [ ] Document any non-deterministic components

### T-W3-010: Path Cache LRU Implementation
**Description**: Replace FIFO cache with LRU for pathfinding
**File**: `engine/world/queries/navigation.py`
**Dependencies**: Phase 2 tests passing
**Acceptance Criteria**:
- [ ] Replace cache dict with OrderedDict
- [ ] Move accessed keys to end (LRU behavior)
- [ ] Evict from front when full
- [ ] Unit test validates LRU behavior
- [ ] Benchmark shows improved hit rate

### T-W3-011: Path Cache Benchmark
**Description**: Benchmark pathfinding cache hit rates
**File**: `tests/performance/world/bench_queries.py`
**Dependencies**: T-W3-010
**Acceptance Criteria**:
- [ ] Create navigation scenario with repeated paths
- [ ] Measure hit rate with FIFO (baseline)
- [ ] Measure hit rate with LRU
- [ ] LRU should show higher hit rate
- [ ] Document improvement percentage

### T-W3-012: Terrain Query Benchmark
**Description**: Benchmark terrain height/normal queries
**File**: `tests/performance/world/bench_terrain.py`
**Dependencies**: Phase 1 tests passing
**Acceptance Criteria**:
- [ ] Benchmark 100,000 height queries
- [ ] Benchmark 100,000 normal queries
- [ ] Measure per-query time
- [ ] Target: height < 1us, normal < 2us
- [ ] Document actual performance

### T-W3-013: Raycast Benchmark
**Description**: Benchmark terrain raycast performance
**File**: `tests/performance/world/bench_queries.py`
**Dependencies**: Phase 1 tests passing
**Acceptance Criteria**:
- [ ] Benchmark 10,000 terrain raycasts
- [ ] Measure average time per raycast
- [ ] Measure worst-case (long distance, many iterations)
- [ ] Target: average < 100us
- [ ] Document binary search iteration impact

### T-W3-014: A* Pathfinding Benchmark
**Description**: Benchmark pathfinding for various path lengths
**File**: `tests/performance/world/bench_queries.py` (extend)
**Dependencies**: Phase 1 tests passing
**Acceptance Criteria**:
- [ ] Benchmark paths of 100, 500, 1000, 2000 nodes
- [ ] Measure time per path
- [ ] Target: 1000 nodes < 5ms
- [ ] Document scaling behavior
- [ ] Identify if hierarchical pathfinding needed

### T-W3-015: QEM Simplification Benchmark
**Description**: Benchmark mesh simplification performance
**File**: `tests/performance/world/bench_hlod.py`
**Dependencies**: Phase 1 tests passing
**Acceptance Criteria**:
- [ ] Benchmark simplification of 10K, 50K, 100K triangle meshes
- [ ] Measure time per simplification
- [ ] Measure quality (error metric)
- [ ] Document scaling behavior
- [ ] Identify if Rust bridge needed

### T-W3-016: Performance Regression Suite
**Description**: Create automated performance regression suite
**File**: `tests/performance/world/regression.py`
**Dependencies**: All benchmarks complete
**Acceptance Criteria**:
- [ ] Suite runs all benchmarks
- [ ] Results saved to JSON
- [ ] Comparison with baseline thresholds
- [ ] CI-friendly exit codes (0=pass, 1=regression)
- [ ] Baseline thresholds documented

---

## Performance Target Summary

| Subsystem | Operation | Target | Benchmark Task |
|-----------|-----------|--------|----------------|
| Terrain | Height query | < 1us | T-W3-012 |
| Terrain | Normal query | < 2us | T-W3-012 |
| Terrain | Raycast | < 100us | T-W3-013 |
| Foliage | Cluster cull | < 100us/100 clusters | T-W3-005 |
| Foliage | Buffer gen | < 10ms/10K instances | T-W3-006 |
| HLOD | Simplify 10K | < 500ms | T-W3-015 |
| Partition | Stress test | Budget enforced | T-W3-001 |
| PCG | Batch 10K | < 10ms | T-W3-008 |
| Queries | A* 1K nodes | < 5ms | T-W3-014 |

---

## Summary

| Category | Tasks | Estimated Effort |
|----------|-------|------------------|
| Streaming | 2 | Medium |
| LOD Transitions | 2 | Low |
| Culling | 2 | Medium |
| PCG Optimization | 3 | Medium |
| Query Optimization | 4 | Medium |
| HLOD Benchmark | 1 | Low |
| Regression Suite | 1 | Medium |
| **Total** | **16** | |
