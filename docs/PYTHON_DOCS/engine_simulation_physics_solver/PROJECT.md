# PROJECT: engine_simulation_physics_solver

**Investigation Source:** docs/investigation/engine_simulation_physics_solver.md  
**Classification:** REAL IMPLEMENTATION  
**Total Lines Analyzed:** ~9,792 lines (5,805 physics + 3,987 solver)

---

## Scope

Complete production-ready rigid body physics simulation system consisting of two tightly coupled modules:

- **engine/simulation/physics** (5,805 lines): Collision shapes, rigid bodies, physics world, queries, sleeping, materials
- **engine/simulation/solver** (3,987 lines): Sequential Impulse, TGS, XPBD constraint solvers, island management

---

## Goals

### Primary Goals

1. Maintain and extend existing real physics simulation implementation
2. Address identified algorithmic gaps without breaking working code
3. Implement missing spatial acceleration structures (broadphase BVH/octree)
4. Complete stubbed features (convex hull computation, mesh BVH, parallel solving)

### Secondary Goals

1. Establish test coverage for mathematical correctness
2. Document numerical stability guarantees
3. Profile and optimize critical paths

---

## Constraints

### Technical Constraints

- **Python 3.13 only**: Must use `uv run python` (not system 3.14)
- **Zero external dependencies**: Python standard library only (math, dataclasses, typing, enum, uuid, time)
- **No numpy/scipy**: All math is handwritten

### Numerical Constraints

- Division-by-zero guards required throughout
- Epsilon comparisons for floating-point equality
- Velocity clamping: linear 500 m/s, angular 100 rad/s
- Contact slop: 1mm penetration tolerance
- Baumgarte stabilization factor: 0.2

### Architectural Constraints

- Existing simulation step order must be preserved (10-step pipeline)
- Island-based architecture must remain (Union-Find with path compression)
- Three solver architectures (SI, TGS, XPBD) must coexist

---

## Acceptance Criteria

### Phase 1: Gap Analysis and Test Infrastructure

- [ ] All existing mathematical formulas verified against textbook references
- [ ] Test suite covering inertia tensors for all 10 shape types
- [ ] Test suite covering quaternion operations
- [ ] Baseline performance benchmarks established

### Phase 2: Broadphase Acceleration

- [ ] BVH or octree implementation replacing O(n^2) baseline
- [ ] Performance improvement measurable for N > 100 bodies
- [ ] No regression in collision detection accuracy

### Phase 3: Geometry Completeness

- [ ] Convex hull computation producing actual hull (not passthrough)
- [ ] Mesh BVH producing spatial tree (not single AABB)
- [ ] True SAT for OBB overlap (not conservative AABB approximation)

### Phase 4: Parallel Solving

- [ ] `ParallelIslandSolver.solve_parallel()` actually parallelizes
- [ ] Deterministic results regardless of thread count
- [ ] Performance scaling demonstrated on multi-core

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mathematical regressions | HIGH | Comprehensive test suite before any changes |
| Numerical instability | HIGH | Preserve all existing epsilon guards and clamps |
| Performance degradation | MEDIUM | Benchmark before/after each phase |
| Breaking solver interop | MEDIUM | Integration tests for all three solver types |

---

## File Inventory

### engine/simulation/physics

| File | Lines | Status |
|------|-------|--------|
| `collision_shapes.py` | 1,624 | REAL - 10 shape types |
| `rigid_body.py` | 1,061 | REAL - Complete body dynamics |
| `physics_world.py` | 1,010 | REAL - World + simulation step |
| `queries.py` | 1,008 | REAL - Ray/overlap/sweep |
| `sleeping.py` | 660 | REAL - Island sleep management |
| `physics_material.py` | 442 | REAL - 15 material presets |

### engine/simulation/solver

| File | Lines | Status |
|------|-------|--------|
| `jacobian.py` | 840 | REAL - Math primitives + Jacobians |
| `xpbd_solver.py` | 791 | REAL - Position-based dynamics |
| `tgs_solver.py` | 750 | REAL - Temporal Gauss-Seidel |
| `constraint_solver.py` | 725 | REAL - Sequential Impulse base |
| `island_manager.py` | 591 | REAL - Union-Find islands |

---

## Academic References

- Erin Catto, GDC 2005: Sequential Impulse
- Dirk Gregorius, GDC 2013: Temporal Gauss-Seidel
- Macklin et al., 2016: XPBD (Extended Position-Based Dynamics)
