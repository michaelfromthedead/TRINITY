# PHASE 4 TODO: Cross-Cutting Enhancements

**Scope**: All 16 files, ~10,973 lines  
**Classification**: Enhancement Opportunities

---

## Status: Enhancement Phase

All three subsystems are production-ready. This phase addresses:
1. GPU acceleration
2. Multi-threading
3. SIMD optimization
4. Testing infrastructure
5. Common module extraction

---

## T-CROSS-4.1: GPU Acceleration Infrastructure

### Tasks

- [ ] **T-CROSS-4.1.1**: Design unified GPU compute interface
  - Acceptance: Abstract dispatch class supporting Vulkan/Metal/WGPU
  - File: `engine/simulation/common/gpu_interface.py`

- [ ] **T-CROSS-4.1.2**: Implement Vulkan compute backend
  - Acceptance: Dispatch methods for grid build, kernel eval, integration

- [ ] **T-CROSS-4.1.3**: Implement Metal compute backend (macOS)
  - Acceptance: Dispatch methods matching Vulkan backend

- [ ] **T-CROSS-4.1.4**: Implement WGPU compute backend (WebGPU)
  - Acceptance: Dispatch methods matching Vulkan backend

- [ ] **T-CROSS-4.1.5**: CPU fallback for all GPU operations
  - Acceptance: Reference implementation passes all tests

### Acceptance Criteria
- Unified interface covers all GPU-accelerable operations
- At least one GPU backend implemented
- CPU fallback available on all platforms
- GPU results match CPU within tolerance

---

## T-CROSS-4.2: Multi-Threading Support

### Tasks

- [ ] **T-CROSS-4.2.1**: Add thread pool infrastructure
  - Acceptance: Configurable worker count, task submission API

- [ ] **T-CROSS-4.2.2**: Parallelize hair strand updates
  - Acceptance: Per-strand FTL solve runs in parallel, linear scaling

- [ ] **T-CROSS-4.2.3**: Parallelize fluid particle force computation
  - Acceptance: Per-particle force accumulation in parallel

- [ ] **T-CROSS-4.2.4**: Parallelize destruction triangle clipping
  - Acceptance: Per-triangle clip operations in parallel

- [ ] **T-CROSS-4.2.5**: Add synchronization barriers where needed
  - Acceptance: Spatial hash rebuild barrier, output aggregation barrier

### Acceptance Criteria
- Thread pool reusable across subsystems
- No data races (verified via thread sanitizer)
- Linear scaling demonstrated for candidate operations
- Synchronization overhead minimal

---

## T-CROSS-4.3: SIMD Optimization

### Tasks

- [ ] **T-CROSS-4.3.1**: Profile hot paths in each subsystem
  - Acceptance: Flame graph identifying >5% time functions

- [ ] **T-CROSS-4.3.2**: Vectorize SPH kernel evaluation (Poly6, Spiky)
  - Acceptance: Batch evaluate 8 kernel values with AVX

- [ ] **T-CROSS-4.3.3**: Vectorize triangle-plane tests
  - Acceptance: Batch test 4 triangles against plane

- [ ] **T-CROSS-4.3.4**: Vectorize vec3 operations via numba
  - Acceptance: JIT-compiled vec3 add/sub/mul/dot/cross

### Acceptance Criteria
- Hot paths identified and documented
- SIMD implementations match scalar results
- Performance improvement measured

---

## T-CROSS-4.4: Testing Infrastructure

### Tasks

- [ ] **T-CROSS-4.4.1**: Create edge case test suite
  - Acceptance: Tests for all edge cases listed in PHASE_4_ARCH.md
  - Files: `tests/destruction/test_edge_cases.py`, `tests/fluid/test_edge_cases.py`, `tests/hair/test_edge_cases.py`

- [ ] **T-CROSS-4.4.2**: Create numerical stability test suite
  - Acceptance: Tests for epsilon inputs, large values, accumulated error

- [ ] **T-CROSS-4.4.3**: Create performance regression test suite
  - Acceptance: Baseline timings, memory tracking, scaling curves

- [ ] **T-CROSS-4.4.4**: Create integration test suite
  - Acceptance: Cross-system scenarios, LOD transitions, callback sequences

- [ ] **T-CROSS-4.4.5**: Add fuzzing infrastructure
  - Acceptance: Random input generation for robustness testing

### Acceptance Criteria
- All edge cases covered
- Numerical stability verified
- Performance baselines established
- Integration scenarios tested
- Fuzzing finds no new crashes

---

## T-CROSS-4.5: Common Module Extraction

### Tasks

- [ ] **T-CROSS-4.5.1**: Extract shared spatial hash implementation
  - Acceptance: Single class used by fluid and hair collision
  - File: `engine/simulation/common/spatial_hash.py`

- [ ] **T-CROSS-4.5.2**: Extract shared object pool implementation
  - Acceptance: Generic pool class parameterized by object type
  - File: `engine/simulation/common/object_pool.py`

- [ ] **T-CROSS-4.5.3**: Extract shared LOD framework
  - Acceptance: Hysteresis logic shared between debris and hair
  - File: `engine/simulation/common/lod.py`

- [ ] **T-CROSS-4.5.4**: Extract numerical utilities
  - Acceptance: Epsilon guards, clamp, safe_divide in one module
  - File: `engine/simulation/common/numerical.py`

- [ ] **T-CROSS-4.5.5**: Standardize Vec3 type alias across systems
  - Acceptance: Single Vec3 definition imported everywhere

### Acceptance Criteria
- No code duplication for shared patterns
- Existing tests still pass after extraction
- Import paths updated consistently

---

## T-CROSS-4.6: Documentation Verification

### Tasks

- [ ] **T-CROSS-4.6.1**: Verify all config.py parameters are documented
  - Acceptance: Each parameter has docstring with units and typical range

- [ ] **T-CROSS-4.6.2**: Verify algorithm comments match implementation
  - Acceptance: No stale comments, all formulas cited

- [ ] **T-CROSS-4.6.3**: Generate API documentation
  - Acceptance: Sphinx docs for public interfaces

- [ ] **T-CROSS-4.6.4**: Document cross-system integration patterns
  - Acceptance: Examples for debris+fluid, hair+collision

### Acceptance Criteria
- All parameters documented
- Comments accurate
- API docs generated
- Integration examples provided

---

## T-CROSS-4.7: Memory Layout Optimization

### Tasks

- [ ] **T-CROSS-4.7.1**: Profile memory access patterns
  - Acceptance: Cachegrind analysis of hot paths

- [ ] **T-CROSS-4.7.2**: Convert fluid particles to SoA layout
  - Acceptance: Separate arrays for position, velocity, density

- [ ] **T-CROSS-4.7.3**: Convert hair particles to SoA layout
  - Acceptance: Separate arrays for position, velocity, mass

- [ ] **T-CROSS-4.7.4**: Benchmark AoS vs SoA for each subsystem
  - Acceptance: Cache miss reduction documented

### Acceptance Criteria
- Memory access patterns profiled
- SoA conversion complete for hot paths
- Cache efficiency improved

---

## Dependencies

| Task | Depends On | Blocking |
|------|------------|----------|
| T-CROSS-4.1 | None | T-CROSS-4.2 (can share thread pool) |
| T-CROSS-4.2 | None | None |
| T-CROSS-4.3 | T-CROSS-4.4.3 (need baseline) | None |
| T-CROSS-4.4 | Phases 1-3 complete | T-CROSS-4.3 |
| T-CROSS-4.5 | Phases 1-3 complete | None |
| T-CROSS-4.6 | Phases 1-3 complete | None |
| T-CROSS-4.7 | T-CROSS-4.4.3 | None |

---

## Estimated Effort

| Task Group | Effort | Priority |
|------------|--------|----------|
| GPU Acceleration (T-CROSS-4.1) | High | Medium |
| Multi-Threading (T-CROSS-4.2) | Medium | Medium |
| SIMD (T-CROSS-4.3) | Medium | Low |
| Testing (T-CROSS-4.4) | Medium | High |
| Common Modules (T-CROSS-4.5) | Low | Medium |
| Documentation (T-CROSS-4.6) | Low | Low |
| Memory Layout (T-CROSS-4.7) | Medium | Low |

---

## Phase 4 Summary

This phase is entirely optional enhancement work. The simulation systems are production-ready without these changes. Priorities:

1. **HIGH**: Testing infrastructure (T-CROSS-4.4) - validates existing code
2. **MEDIUM**: GPU/Threading (T-CROSS-4.1, 4.2) - performance scaling
3. **LOW**: SIMD/Memory (T-CROSS-4.3, 4.7) - micro-optimization

**Recommendation**: Complete T-CROSS-4.4 first to establish baselines before optimization work.
