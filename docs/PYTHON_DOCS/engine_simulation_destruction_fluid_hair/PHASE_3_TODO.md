# PHASE 3 TODO: Hair System

**Scope**: ~2,600 lines across 4 files  
**Classification**: REAL (Production-Ready)

---

## Status: Production-Ready

The hair system is fully implemented. Tasks below focus on:
1. Documentation verification
2. Testing coverage
3. GPU acceleration opportunities
4. Performance optimization

---

## T-HAIR-3.1: Verify Hair Simulation Core

**File**: `engine/simulation/hair/hair_simulation.py` (662 lines)

### Tasks

- [ ] **T-HAIR-3.1.1**: Verify FTL constraint solving correctness
  - Acceptance: Unit test showing exact length preservation after solve

- [ ] **T-HAIR-3.1.2**: Test Verlet integration stability
  - Acceptance: No energy drift over extended simulation (1000+ frames)

- [ ] **T-HAIR-3.1.3**: Verify inertia transfer from head motion
  - Acceptance: Visual test showing correct hair following during head rotation

- [ ] **T-HAIR-3.1.4**: Test guide hair interpolation accuracy
  - Acceptance: Interpolated hair positions within tolerance of full simulation

### Acceptance Criteria
- FTL produces exact lengths
- Verlet stable long-term
- Inertia transfer produces natural following
- Guide interpolation visually acceptable

---

## T-HAIR-3.2: Verify Collision System

**File**: `engine/simulation/hair/hair_collision.py` (564 lines)

### Tasks

- [ ] **T-HAIR-3.2.1**: Test point-capsule collision correctness
  - Acceptance: Unit test for all capsule regions (cylinder, caps)

- [ ] **T-HAIR-3.2.2**: Test point-sphere collision correctness
  - Acceptance: Unit test showing correct surface push-out

- [ ] **T-HAIR-3.2.3**: Verify SDF collision with signed distance field
  - Acceptance: Test with analytical SDF (sphere), verify correct gradient direction

- [ ] **T-HAIR-3.2.4**: Test density-field self-collision
  - Acceptance: No hair-hair interpenetration, grid resolution adequate

- [ ] **T-HAIR-3.2.5**: Verify friction application on all collision types
  - Acceptance: Tangential velocity reduced proportionally to friction coefficient

### Acceptance Criteria
- All collision types push particles to correct positions
- SDF gradient direction correct
- Self-collision prevents interpenetration
- Friction reduces sliding velocity

---

## T-HAIR-3.3: Verify Constraint System

**File**: `engine/simulation/hair/hair_constraints.py` (542 lines)

### Tasks

- [ ] **T-HAIR-3.3.1**: Verify length constraint enforcement
  - Acceptance: Segment lengths within 0.1% of rest length after solve

- [ ] **T-HAIR-3.3.2**: Test global shape matching polar decomposition
  - Acceptance: Rotation matrix is pure rotation (det = 1, orthonormal)

- [ ] **T-HAIR-3.3.3**: Verify local shape constraint (angle preservation)
  - Acceptance: Inter-segment angles converge toward rest angles

- [ ] **T-HAIR-3.3.4**: Test Rodrigues rotation formula correctness
  - Acceptance: Compare against quaternion-based rotation for same axis/angle

- [ ] **T-HAIR-3.3.5**: Test constraint stacking (length + shape + collision)
  - Acceptance: All constraints satisfied simultaneously within tolerance

### Acceptance Criteria
- Length preserved exactly
- Shape matching uses valid rotation
- Angle constraints restore curvature
- Rodrigues matches quaternion reference
- Multiple constraints compose correctly

---

## T-HAIR-3.4: Verify LOD System

**File**: `engine/simulation/hair/hair_lod.py` (470 lines)

### Tasks

- [ ] **T-HAIR-3.4.1**: Test LOD level selection based on distance
  - Acceptance: Correct level selected for each distance range

- [ ] **T-HAIR-3.4.2**: Verify hysteresis prevents LOD thrashing
  - Acceptance: No rapid level switching when camera oscillates at boundary

- [ ] **T-HAIR-3.4.3**: Test guide hair selection algorithms
  - Acceptance: Guides uniformly distributed over scalp

- [ ] **T-HAIR-3.4.4**: Test inverse-distance weighted interpolation
  - Acceptance: Weights sum to 1, closer guides have higher weight

- [ ] **T-HAIR-3.4.5**: Verify shell rendering data preparation
  - Acceptance: Shell layers, textures, alpha correctly generated

### Acceptance Criteria
- LOD selection correct for all distances
- Hysteresis prevents thrashing
- Guide distribution uniform
- Interpolation weights correct
- Shell data renderable

---

## T-HAIR-3.5: GPU Acceleration Opportunities

### Identified Candidates (from investigation)

- [ ] **T-HAIR-3.5.1**: GPU compute for FTL constraint solving
  - Acceptance: Parallel per-strand solve, match CPU results

- [ ] **T-HAIR-3.5.2**: GPU compute for density-field self-collision
  - Acceptance: 3D grid build and lookup on GPU

- [ ] **T-HAIR-3.5.3**: GPU compute for guide hair interpolation
  - Acceptance: Parallel interpolation of all render hairs

- [ ] **T-HAIR-3.5.4**: Benchmark GPU vs CPU for 10K, 50K, 100K strands
  - Acceptance: Performance comparison table

### Acceptance Criteria
- GPU implementations match CPU within tolerance
- Performance improvement documented
- Memory usage bounded

---

## T-HAIR-3.6: Performance Optimization Opportunities

### Identified Candidates (from investigation)

- [ ] **T-HAIR-3.6.1**: SIMD optimization for constraint solving
  - Acceptance: Batch process multiple segments with SIMD

- [ ] **T-HAIR-3.6.2**: Multi-threaded strand update (embarrassingly parallel)
  - Acceptance: Linear scaling with core count for FTL

- [ ] **T-HAIR-3.6.3**: Spatial hash for collision pair culling
  - Acceptance: Reduce collision checks via broad-phase

- [ ] **T-HAIR-3.6.4**: Cache-friendly memory layout (SoA)
  - Acceptance: Position array, velocity array vs interleaved

### Acceptance Criteria
- Optimizations benchmarked against baseline
- No behavioral changes
- Thread safety verified

---

## T-HAIR-3.7: Edge Case Testing

### Numerical Stability Tests

- [ ] **T-HAIR-3.7.1**: Test zero-length segment edge case
  - Acceptance: Graceful handling, no NaN/Inf

- [ ] **T-HAIR-3.7.2**: Test collinear segments (degenerate rotation axis)
  - Acceptance: Rodrigues formula handles gracefully

- [ ] **T-HAIR-3.7.3**: Test extreme head velocity (whiplash scenario)
  - Acceptance: Constraints prevent strand breakage/stretching

- [ ] **T-HAIR-3.7.4**: Test single-segment hair
  - Acceptance: Degenerate case handled correctly

- [ ] **T-HAIR-3.7.5**: Test hair inside collision geometry at start
  - Acceptance: Particles pushed out without explosion

### Acceptance Criteria
- All edge cases handled gracefully
- No NaN/Inf output
- No strand stretching under stress

---

## Dependencies

| Task | Depends On | Blocking |
|------|------------|----------|
| T-HAIR-3.1 | None | T-HAIR-3.3, T-HAIR-3.4 |
| T-HAIR-3.2 | None | None |
| T-HAIR-3.3 | T-HAIR-3.1 | None |
| T-HAIR-3.4 | T-HAIR-3.1 | None |
| T-HAIR-3.5 | All above | None |
| T-HAIR-3.6 | All above | None |
| T-HAIR-3.7 | All above | None |

---

## Estimated Effort

| Task Group | Effort | Priority |
|------------|--------|----------|
| Verification (T-HAIR-3.1 to T-HAIR-3.4) | Medium | High |
| GPU Acceleration (T-HAIR-3.5) | High | Medium |
| Performance (T-HAIR-3.6) | Medium | Low |
| Edge Cases (T-HAIR-3.7) | Low | Medium |

**Total**: No implementation work required. Focus is verification, testing, and GPU acceleration.
