# PHASE 2 TODO: Fluid System

**Scope**: ~3,504 lines across 6 files  
**Classification**: REAL (Production-Ready), GPU interface partial

---

## Status: Production-Ready (GPU Backends TBD)

The fluid system is fully implemented for CPU. Tasks below focus on:
1. Documentation verification
2. Testing coverage
3. GPU backend implementation
4. Performance optimization

---

## T-FLUID-2.1: Verify SPH Implementation

**File**: `engine/simulation/fluid/sph.py` (729 lines)

### Tasks

- [ ] **T-FLUID-2.1.1**: Verify kernel function normalization constants
  - Acceptance: Unit tests confirming integral of each kernel equals 1

- [ ] **T-FLUID-2.1.2**: Test spatial hash grid correctness
  - Acceptance: Unit test comparing neighbor results against brute-force O(n^2)

- [ ] **T-FLUID-2.1.3**: Verify Tait equation pressure stability
  - Acceptance: No negative pressures, stable compression behavior

- [ ] **T-FLUID-2.1.4**: Test surface tension via color field method
  - Acceptance: Visual test showing correct surface curvature response

### Acceptance Criteria
- Kernel integrals verified
- Spatial hash produces correct neighbors
- Pressure always non-negative
- Surface tension produces droplet formation

---

## T-FLUID-2.2: Verify FLIP/PIC Implementation

**File**: `engine/simulation/fluid/flip_pic.py` (678 lines)

### Tasks

- [ ] **T-FLUID-2.2.1**: Verify MAC grid staggering correctness
  - Acceptance: Unit test showing correct velocity sampling at cell centers/faces

- [ ] **T-FLUID-2.2.2**: Test Jacobi iteration convergence
  - Acceptance: Residual below threshold after max iterations

- [ ] **T-FLUID-2.2.3**: Test FLIP/PIC ratio effect on behavior
  - Acceptance: Visual comparison at alpha=0, 0.05, 0.5, 1.0

- [ ] **T-FLUID-2.2.4**: Verify trilinear interpolation boundary handling
  - Acceptance: No out-of-bounds access at grid edges

### Acceptance Criteria
- MAC grid correctly staggered
- Jacobi converges to target residual
- FLIP/PIC ratio produces expected stability/detail trade-off
- Boundary handling robust

---

## T-FLUID-2.3: Verify Position Based Fluids Implementation

**File**: `engine/simulation/fluid/pbf.py` (572 lines)

### Tasks

- [ ] **T-FLUID-2.3.1**: Verify Lagrange multiplier computation stability
  - Acceptance: No division by zero, clamped lambda values

- [ ] **T-FLUID-2.3.2**: Test tensile instability correction
  - Acceptance: No particle clustering under compression

- [ ] **T-FLUID-2.3.3**: Verify vorticity confinement energy restoration
  - Acceptance: Swirling motion maintained over extended simulation

- [ ] **T-FLUID-2.3.4**: Test XSPH viscosity coherence
  - Acceptance: Smooth velocity field without excessive damping

### Acceptance Criteria
- Lambda computation numerically stable
- No particle clustering
- Vorticity preserved
- Velocity field coherent

---

## T-FLUID-2.4: Implement GPU Backends

**File**: `engine/simulation/fluid/gpu_fluid.py` (560 lines)

### Tasks

- [ ] **T-FLUID-2.4.1**: Implement Vulkan compute shader dispatch
  - Acceptance: All abstract methods implemented, passes CPU fallback tests

- [ ] **T-FLUID-2.4.2**: Implement Metal compute shader dispatch (macOS)
  - Acceptance: All abstract methods implemented, passes CPU fallback tests

- [ ] **T-FLUID-2.4.3**: Implement WGPU compute shader dispatch (WebGPU)
  - Acceptance: All abstract methods implemented, passes CPU fallback tests

- [ ] **T-FLUID-2.4.4**: Benchmark GPU vs CPU performance
  - Acceptance: Performance comparison table for 10K, 100K, 1M particles

### Acceptance Criteria
- At least one GPU backend implemented
- GPU results match CPU reference within tolerance
- Performance improvement documented

---

## T-FLUID-2.5: Verify Eulerian Solver Implementation

**File**: `engine/simulation/fluid/eulerian.py` (488 lines)

### Tasks

- [ ] **T-FLUID-2.5.1**: Verify semi-Lagrangian advection stability
  - Acceptance: No oscillations, energy-preserving for inviscid case

- [ ] **T-FLUID-2.5.2**: Test cell type marking (SOLID/FLUID/AIR)
  - Acceptance: Correct boundary behavior at each cell type

- [ ] **T-FLUID-2.5.3**: Verify CFL condition enforcement
  - Acceptance: Adaptive timestep or sub-stepping kicks in at high velocities

- [ ] **T-FLUID-2.5.4**: Test Poisson pressure solve convergence
  - Acceptance: Divergence-free velocity field after projection

### Acceptance Criteria
- Advection unconditionally stable
- Boundary conditions correct
- CFL enforced
- Velocity field divergence-free

---

## T-FLUID-2.6: Verify Shallow Water Implementation

**File**: `engine/simulation/fluid/shallow_water.py` (477 lines)

### Tasks

- [ ] **T-FLUID-2.6.1**: Verify Saint-Venant equation implementation
  - Acceptance: Mass conservation test (total water volume constant)

- [ ] **T-FLUID-2.6.2**: Test Strang splitting accuracy
  - Acceptance: Compare against un-split reference at fine timestep

- [ ] **T-FLUID-2.6.3**: Test terrain boundary variants (bowl, slope)
  - Acceptance: Correct pooling/flowing behavior

- [ ] **T-FLUID-2.6.4**: Test dam break scenario
  - Acceptance: Wave propagation matches analytical solution

### Acceptance Criteria
- Mass conserved
- Strang splitting produces correct 2D behavior
- Terrain boundaries work correctly
- Dam break matches theory

---

## T-FLUID-2.7: Performance Optimization Opportunities

### Identified Candidates (from investigation)

- [ ] **T-FLUID-2.7.1**: SIMD optimization for kernel evaluation
  - Acceptance: Benchmark showing improvement for Poly6/Spiky batched evaluation

- [ ] **T-FLUID-2.7.2**: Multi-threaded particle update loop
  - Acceptance: Linear scaling with core count for neighbor search and force computation

- [ ] **T-FLUID-2.7.3**: Incremental spatial hash update
  - Acceptance: Reduce rebuild overhead for slowly-moving fluids

- [ ] **T-FLUID-2.7.4**: Memory layout optimization (SoA vs AoS)
  - Acceptance: Cache efficiency comparison

### Acceptance Criteria
- Optimizations benchmarked against baseline
- No behavioral changes
- Thread safety verified

---

## T-FLUID-2.8: Edge Case Testing

### Numerical Stability Tests

- [ ] **T-FLUID-2.8.1**: Test zero-particle edge case
  - Acceptance: No crash, graceful no-op

- [ ] **T-FLUID-2.8.2**: Test single-particle edge case
  - Acceptance: No NaN/Inf, correct gravity-only behavior

- [ ] **T-FLUID-2.8.3**: Test extreme velocities (CFL violation risk)
  - Acceptance: Sub-stepping or clamping prevents instability

- [ ] **T-FLUID-2.8.4**: Test extreme particle counts (stress test)
  - Acceptance: Performance degrades gracefully, no OOM

### Acceptance Criteria
- All edge cases handled gracefully
- No NaN/Inf output
- Memory bounded

---

## Dependencies

| Task | Depends On | Blocking |
|------|------------|----------|
| T-FLUID-2.1 | None | T-FLUID-2.3 (uses SPH kernels) |
| T-FLUID-2.2 | None | None |
| T-FLUID-2.3 | T-FLUID-2.1 | None |
| T-FLUID-2.4 | T-FLUID-2.1, T-FLUID-2.3 | None |
| T-FLUID-2.5 | None | None |
| T-FLUID-2.6 | None | None |
| T-FLUID-2.7 | All above | None |
| T-FLUID-2.8 | All above | None |

---

## Estimated Effort

| Task Group | Effort | Priority |
|------------|--------|----------|
| Verification (T-FLUID-2.1 to T-FLUID-2.3, 2.5, 2.6) | Medium | High |
| GPU Backends (T-FLUID-2.4) | High | Medium |
| Performance (T-FLUID-2.7) | Medium | Low |
| Edge Cases (T-FLUID-2.8) | Low | Medium |

**Total**: CPU implementations complete. GPU backends are primary implementation work.
