# ENGINE_WORLD - Phase 3 Architecture: Performance Optimization

## Phase Overview

Phase 3 addresses performance concerns identified during investigation. The world subsystem is implemented in pure Python for portability, but hot paths need optimization for production use. This phase identifies bottlenecks and implements optimizations while maintaining the Python-first architecture.

## Architecture Decisions

### ADR-W3-001: Streaming Budget Enforcement

**Context**: Partition streaming has memory and frame time budgets, but enforcement may not handle edge cases (rapid movement, many cells loading simultaneously).

**Decision**: Add stress testing and implement budget enforcement fixes:
1. Hard-cap concurrent load operations
2. Add queue overflow protection
3. Implement load prioritization under pressure

**Implementation**:
```python
class WorldStreaming:
    def tick(self, delta_time: float) -> None:
        # Budget check BEFORE starting new loads
        if self._budget.memory_remaining < CELL_MEMORY_ESTIMATE:
            self._pause_loads = True
            return
        
        # Frame time budget
        frame_start = time.perf_counter()
        while self._load_queue and not self._pause_loads:
            if time.perf_counter() - frame_start > self._budget.frame_ms / 1000:
                break
            self._process_next_load()
```

**Validation**: Stress test with rapid camera movement across many cells.

### ADR-W3-002: LOD Transition Smoothing

**Context**: LOD transitions may pop if distance thresholds are crossed rapidly. Current implementation has hysteresis but may need tuning.

**Decision**: Implement and validate three transition methods:
1. **Dither**: Bayer pattern fade (already in HLOD)
2. **Crossfade**: Render both LODs with alpha blend
3. **Morph**: Vertex position interpolation (requires shader support)

**Implementation**: Focus on dither (CPU-side) and crossfade (requires dual render). Morph documented as GPU-dependent future work.

### ADR-W3-003: Frustum Culling Granularity

**Context**: Foliage HISM does per-cluster culling, but cluster size may not be optimal. Large clusters waste GPU; small clusters waste CPU.

**Decision**: Benchmark different cluster sizes, document optimal range for target instance counts.

**Benchmark Matrix**:
| Instances | Cluster Size | Clusters | CPU Time | GPU Time |
|-----------|--------------|----------|----------|----------|
| 100,000   | 100          | 1000     | TBD      | TBD      |
| 100,000   | 500          | 200      | TBD      | TBD      |
| 100,000   | 1000         | 100      | TBD      | TBD      |

### ADR-W3-004: PCG Noise Batching

**Context**: Pure Python noise generation is slow for large maps. Individual `sample()` calls have function call overhead.

**Decision**: Implement batch sampling interface:
```python
def sample_batch(self, positions: List[Tuple[float, float]]) -> List[float]:
    """Sample noise at multiple positions in single call."""
```

**Rationale**: Reduces Python function call overhead. Future Rust bridge can optimize the batch implementation further.

### ADR-W3-005: Determinism Validation

**Context**: All PCG algorithms claim determinism, but this must be validated across runs.

**Decision**: Create determinism test suite:
1. Generate output with fixed seed
2. Save output hash
3. Regenerate with same seed
4. Compare hashes

**Scope**: All noise types, all scatter algorithms, all placement rules.

### ADR-W3-006: Path Caching Improvement

**Context**: Navigation uses simple FIFO cache. Under heavy pathfinding load, frequently-used paths may be evicted.

**Decision**: Evaluate LRU cache as alternative. Benchmark shows:
- FIFO: O(1) eviction, poor hit rate under load
- LRU: O(1) eviction (with OrderedDict), better hit rate

**Implementation**: Replace FIFO with `OrderedDict`-based LRU if benchmarks show improvement.

## Performance Metrics

### Target Metrics

| Subsystem | Operation | Target | Notes |
|-----------|-----------|--------|-------|
| Terrain | Height query | < 1us | Bilinear interpolation |
| Terrain | Normal query | < 2us | Central difference |
| Foliage | Cluster cull | < 100us | Per 100 clusters |
| HLOD | LOD select | < 50us | Per 100 cells |
| Partition | Cell load | < 16ms | Per cell (async) |
| PCG | Noise batch | < 10ms | Per 10,000 samples |
| Queries | Raycast | < 100us | Single ray |
| Queries | A* path | < 5ms | 1000 node path |

### Profiling Strategy

1. **Micro-benchmarks**: Isolated function timing with `timeit`
2. **Macro-benchmarks**: Full system simulation with `cProfile`
3. **Memory profiling**: `tracemalloc` for allocation tracking
4. **Stress tests**: High-load scenarios (many cells, many instances)

## Optimization Techniques

### CPU Optimizations (Pure Python)

1. **Local variable caching**: `sqrt = math.sqrt` in tight loops
2. **List comprehensions**: Replace explicit loops where possible
3. **Generator expressions**: Avoid materializing large lists
4. **`__slots__`**: Reduce memory per instance
5. **Batch operations**: Single call for multiple items

### Future Optimizations (Post-Bridge)

These require Rust implementation:
1. **SIMD noise**: Vectorized noise sampling
2. **Parallel mesh simplification**: Multi-threaded QEM
3. **GPU frustum culling**: Compute shader culling
4. **Async I/O**: Non-blocking cell loading

## Test Organization

```
tests/performance/world/
├── bench_terrain.py    # Height/normal query benchmarks
├── bench_foliage.py    # Culling, instance generation
├── bench_hlod.py       # Simplification, LOD selection
├── bench_partition.py  # Streaming stress test
├── bench_pcg.py        # Noise generation
├── bench_queries.py    # Raycast, pathfinding
└── stress_streaming.py # Full system stress test
```

## Success Criteria

- [ ] All target metrics met or documented as bridge candidates
- [ ] Streaming budget enforced under stress test
- [ ] LOD transitions smooth (no visible popping)
- [ ] Determinism validated for all PCG algorithms
- [ ] Performance regression suite established
