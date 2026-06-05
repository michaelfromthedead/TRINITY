# PROJECT: engine/simulation (components, constraints, softbody, vehicles)

**Date**: 2026-05-23  
**Source**: Archaeological Investigation 2026-05-22  
**Files**: 23 real implementations (~14,944 lines)

---

## Scope

This project encompasses the core physics simulation subsystems within the TRINITY engine:

1. **Components** (6 files, ~3,406 lines) — Simulation components for cloth, colliders, fluid, destruction, characters, and vehicles
2. **Constraints** (6 files, ~3,311 lines) — Joint and contact constraint solvers
3. **Softbody** (5 files, ~3,283 lines) — Deformable body simulation via PBD, FEM, and shape matching
4. **Vehicles** (6 files, ~4,681 lines) — Complete vehicle dynamics including wheeled, aircraft, and watercraft

---

## Goals

### Primary Goals

1. **Maintain Production Quality** — All 23 files are verified real implementations; preserve mathematical correctness
2. **Algorithm Consistency** — Ensure all physics formulas match published references (Pacejka 2012, Hill 1938, Baumgarte 1972)
3. **Integration Readiness** — All subsystems must expose stepping, force application, and state query interfaces
4. **Performance Baseline** — Establish benchmarks for constraint solving, soft body iteration, and vehicle simulation

### Secondary Goals

1. **Test Coverage** — Each algorithm must have numerical validation tests
2. **Documentation** — Document physical units and dimensional consistency requirements
3. **GPU Acceleration Path** — Identify parallelization opportunities in constraint solvers and soft body

---

## Constraints

### Technical Constraints

- **Python 3.13** — All code must be compatible with statically-linked Python 3.13 interpreter
- **NumPy Dependency** — Linear algebra operations assume NumPy ndarray semantics
- **Real-time Budget** — Physics substep must complete within 2ms for 60 FPS target
- **Memory Footprint** — Soft body simulations limited to 100K particles per scene

### Design Constraints

- **No Stubs** — All implementations must be complete; no pass/NotImplementedError patterns
- **Unit Correctness** — All physical quantities must use SI units (N, m, kg, rad/s)
- **Dimensional Consistency** — Force calculations must balance units across all terms

---

## Acceptance Criteria

### Phase 1: Components
- [ ] All 6 component files have unit tests covering core algorithms
- [ ] Inertia tensor computations validated against analytical solutions
- [ ] PBD cloth simulation converges within 10 iterations

### Phase 2: Constraints
- [ ] Joint constraint warm starting reduces iteration count by 30%+
- [ ] Contact friction matches Coulomb cone approximation
- [ ] Soft constraint CFM/ERP parameters documented with derivation

### Phase 3: Softbody
- [ ] FEM solver handles Neo-Hookean, corotational, and St. Venant-Kirchhoff materials
- [ ] Volume preservation error < 1% after 1000 substeps
- [ ] Shape matching SVD decomposition numerically stable

### Phase 4: Vehicles
- [ ] Pacejka Magic Formula matches published tire data within 5%
- [ ] Differential types (Open, LSD, Torsen, Locked) produce correct torque splits
- [ ] Aircraft stall behavior exhibits smooth tanh transition

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Numerical instability in FEM solver | High | Polar decomposition with SVD fallback |
| Constraint solver divergence | Medium | Warm starting and Baumgarte stabilization |
| Performance regression in soft body | Medium | Spatial hashing for collision; SIMD vectorization |
| Pacejka parameter tuning | Low | Use validated datasets from tire manufacturers |

---

## Success Metrics

1. **Test Pass Rate**: 100% of physics algorithm tests pass
2. **Numerical Accuracy**: Constraint error < 1e-6 after convergence
3. **Performance**: Full vehicle simulation < 0.5ms per frame
4. **Coverage**: All 23 files have at least 80% line coverage
