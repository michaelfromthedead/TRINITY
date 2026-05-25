# ENGINE_WORLD - Phase 4 TODO: Rust Bridge Preparation

## Task List

### T-W4-001: Bridge Candidate Analysis
**Description**: Analyze Phase 3 benchmarks to finalize bridge candidates
**Dependencies**: Phase 3 complete
**Acceptance Criteria**:
- [ ] Review all Phase 3 benchmark results
- [ ] List operations exceeding targets
- [ ] Evaluate each against bridge criteria (CPU-bound, hot path, self-contained)
- [ ] Produce ranked list of bridge candidates
- [ ] Document expected speedup for each

### T-W4-006: Noise Bridge Integration
**Description**: Create Python wrapper with fallback
**Files**: `engine/world/bridge.py`, `engine/world/_bridge.pyi`
**Dependencies**: T-W4-003, T-W4-004, T-W4-005
**Acceptance Criteria**:
- [ ] Create `bridge.py` with import fallback
- [ ] Define `HAS_RUST_BRIDGE` flag
- [ ] Implement wrapper functions with threshold check
- [ ] Create type stub file
- [ ] Integration test verifies fallback works

### T-W4-007: Noise Crossover Benchmark
**Description**: Determine crossover point for noise bridge
**File**: `tests/performance/world/bench_bridge.py`
**Dependencies**: T-W4-006
**Acceptance Criteria**:
- [ ] Benchmark Python vs Rust at various batch sizes (10, 100, 1000, 10000)
- [ ] Find crossover point where Rust becomes faster
- [ ] Document threshold in bridge.py
- [ ] Document overhead analysis

### T-W4-014: CI Integration
**Description**: Add bridge build and test to CI
**Files**: CI configuration
**Dependencies**: All bridge tasks complete
**Acceptance Criteria**:
- [ ] CI builds Rust crate
- [ ] CI runs Rust tests
- [ ] CI runs property tests
- [ ] CI benchmarks (optional, track regressions)
- [ ] CI produces Python wheel

---

## Bridge Function Summary

| Function | Input | Output | Python Fallback |
|----------|-------|--------|-----------------|
| `perlin_noise_batch` | `[(x,z), ...]`, seed | `[float, ...]` | `PerlinNoise.sample()` loop |
| `simplex_noise_batch` | `[(x,z), ...]`, seed | `[float, ...]` | `SimplexNoise.sample()` loop |
| `worley_noise_batch` | `[(x,z), ...]`, seed, metric | `[float, ...]` | `WorleyNoise.sample()` loop |
| `simplify_mesh_qem` | vertices, indices, ratio | vertices, indices | `MeshSimplifier.simplify()` |
| `astar_pathfind` | start, goal, grid, cell_size | `[(x,z), ...]` | `NavigationQuerySystem.find_path()` |
| `frustum_cull_aabbs` | planes, aabbs | `[bool, ...]` | `Frustum.contains_bounds()` loop |

---

## Summary

| Category | Tasks | Estimated Effort |
|----------|-------|------------------|
| Analysis | 1 | Low |
| Setup | 1 | Low |
| Noise Bridge | 4 | Medium |
| Mesh Bridge | 1 | High |
| Pathfinding Bridge | 1 | Medium |
| Culling Bridge | 1 | Medium |
| Infrastructure | 3 | Medium |
| CI | 1 | Low |
| **Total** | **14** | |

---

## Phase Completion Criteria

Phase 4 is complete when:
1. At least one bridge function is production-ready (noise recommended first)
2. Property tests validate correctness
3. Benchmarks show expected speedup
4. Python fallback works when Rust unavailable
5. CI builds and tests the bridge
