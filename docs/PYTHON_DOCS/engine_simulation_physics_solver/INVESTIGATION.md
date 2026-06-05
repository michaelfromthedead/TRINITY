# Archaeological Investigation: engine/simulation/physics + engine/simulation/solver

**Investigation Date:** 2026-05-22  
**Total Lines Analyzed:** ~9,792 lines (5,805 physics + 3,987 solver)  
**Classification:** REAL IMPLEMENTATION

---

## Executive Summary

Both `engine/simulation/physics` and `engine/simulation/solver` modules contain **production-ready, real implementations** of a complete rigid body physics simulation system. This is not stub code or placeholder infrastructure. The implementation demonstrates deep understanding of physics simulation algorithms including Sequential Impulse (SI), Temporal Gauss-Seidel (TGS), and Extended Position-Based Dynamics (XPBD) constraint solvers.

---

## Classification: REAL (Not Stub)

### Evidence of Real Implementation

1. **Complete Mathematical Formulas**: Correct inertia tensor calculations for spheres (`(2/5) * m * r^2`), boxes (`(1/12) * m * (a^2 + b^2)`), capsules (cylinder + hemispheres with parallel axis theorem)

2. **Working Algorithms**: Full GJK support point calculations, ray-shape intersection tests (quadratic formula for sphere, slab method for box), contact manifold generation

3. **Multiple Solver Implementations**: Three complete constraint solver architectures with academic references (Erin Catto GDC 2005, Dirk Gregorius GDC 2013, Macklin XPBD 2016)

4. **Numerical Stability Handling**: Division-by-zero guards, epsilon comparisons, velocity clamping, Baumgarte stabilization, penetration slop

5. **Performance Optimizations**: Island-based sleeping, warm starting, Union-Find for connectivity, cached AABB computation

---

## File-by-File Analysis

### engine/simulation/physics (5,805 lines)

| File | Lines | Classification | Purpose |
|------|-------|----------------|---------|
| `collision_shapes.py` | 1,624 | REAL | 10 shape types: sphere, box, capsule, cylinder, cone, convex hull, mesh, compound, plane, heightfield |
| `rigid_body.py` | 1,061 | REAL | Complete rigid body with mass properties, inertia tensors, force/impulse application, state interpolation |
| `physics_world.py` | 1,010 | REAL | World container with broadphase (O(n^2) baseline), narrowphase, collision callbacks, query interface |
| `queries.py` | 1,008 | REAL | Raycasting, overlap tests, sweep tests with collision filtering |
| `sleeping.py` | 660 | REAL | Island-based sleep manager with Union-Find connectivity |
| `physics_material.py` | 442 | REAL | 15 material presets, friction/restitution combine modes |

### engine/simulation/solver (3,987 lines)

| File | Lines | Classification | Purpose |
|------|-------|----------------|---------|
| `jacobian.py` | 840 | REAL | Vec3, Mat3, Quaternion math; Jacobian computation for constraints |
| `xpbd_solver.py` | 791 | REAL | Extended Position-Based Dynamics with compliance, damping, Lagrange multipliers |
| `tgs_solver.py` | 750 | REAL | Temporal Gauss-Seidel with split impulse, regularization, mass scaling for extreme ratios |
| `constraint_solver.py` | 725 | REAL | Base Sequential Impulse solver, RigidBody class, constraint protocol |
| `island_manager.py` | 591 | REAL | Union-Find islands, parallel solving support, sleep state transitions |

---

## Key Algorithms Found

### Collision Detection

```
Broadphase: AABB overlap test (O(n^2) baseline)
Narrowphase: GJK support points, SAT-style AABB overlap
Ray-Sphere: Quadratic formula intersection
Ray-Box: Slab method (axis-aligned)
Ray-Capsule: Segment closest point + sphere test
```

### Constraint Solving

```
Sequential Impulse (SI):
  - Jacobian: J = [linear_a, angular_a, linear_b, angular_b]
  - Effective mass: K = J * M^-1 * J^T
  - Impulse: lambda = -K^-1 * (Jv + bias)
  - Warm starting: apply previous lambda * factor

Temporal Gauss-Seidel (TGS):
  - Regularization: gamma = compliance / dt
  - Mass scaling for extreme ratios (< 0.01)
  - Split impulse for position correction
  - Substep integration

XPBD:
  - Position-based with compliance parameter
  - Delta lambda = (-C - alpha * lambda) / (w + alpha)
  - Constraint types: distance, bending, volume, collision
```

### Sleep System

```
Island Detection: Union-Find with path compression + rank
Sleep Criteria: |v| < threshold AND |omega| < threshold for T seconds
Wake Propagation: Entire island wakes on contact
```

---

## Physics Constants (config.py)

```python
DEFAULT_GRAVITY = (0.0, -9.81, 0.0)
DEFAULT_TIMESTEP = 1/60 (60 Hz)
SOLVER_ITERATIONS = 10
POSITION_ITERATIONS = 4
BAUMGARTE_FACTOR = 0.2
CONTACT_SLOP = 0.001  # 1mm
WARM_START_FACTOR = 0.8
MAX_LINEAR_VELOCITY = 500.0 m/s
MAX_ANGULAR_VELOCITY = 100.0 rad/s
```

---

## Mathematical Correctness

### Inertia Tensors (Verified Correct)

| Shape | Formula | Code Location |
|-------|---------|---------------|
| Sphere | `I = (2/5) m r^2` | collision_shapes.py:469 |
| Box | `Ixx = (1/12) m (sy^2 + sz^2)` | collision_shapes.py:620 |
| Capsule | Cylinder + hemispheres + parallel axis | collision_shapes.py:786-811 |
| Cylinder | `Ixx = (1/12) m (3r^2 + h^2)` | collision_shapes.py:973 |

### Quaternion Operations (Verified Correct)

```python
# Rotation: q * v * q^-1 (optimized form)
tx = 2.0 * (qy * vz - qz * vy)
ty = 2.0 * (qz * vx - qx * vz)
tz = 2.0 * (qx * vy - qy * vx)
result = (vx + qw * tx + qy * tz - qz * ty, ...)
```

---

## Integration Architecture

### Simulation Step Order (physics_world.py:412-463)

1. Save previous states (interpolation)
2. Broadphase collision detection
3. Narrowphase contact generation
4. Build islands (Union-Find)
5. Integrate velocities (forces -> velocities)
6. Solve velocity constraints
7. Integrate positions (velocities -> positions)
8. Solve position constraints
9. Update sleep states
10. Fire collision callbacks

---

## Identified Gaps / TODOs

1. **Broadphase**: O(n^2) baseline only; BVH/octree mentioned in enum but not implemented
2. **Convex Hull**: `_compute_hull()` returns input points, no actual hull computation
3. **Mesh BVH**: `_build_bvh()` computes overall AABB only, no spatial tree
4. **OBB Tests**: Box overlap uses conservative AABB approximation, not true SAT
5. **Friction Model**: Simplified Coulomb cone approximation
6. **Parallel Solving**: `ParallelIslandSolver.solve_parallel()` is sequential placeholder

---

## Dependencies

### Internal

- `engine.simulation.physics.config` (shared between physics/solver via import)
- Cross-references between rigid_body, collision_shapes, physics_material, sleeping, queries

### External

- Python standard library only (math, dataclasses, typing, enum, uuid, time)
- No numpy/scipy/external physics libraries

---

## Quality Assessment

| Aspect | Score | Notes |
|--------|-------|-------|
| Algorithmic Correctness | 9/10 | Correct formulas, proper numerical guards |
| Code Organization | 9/10 | Clean separation of concerns, well-documented |
| Completeness | 7/10 | Core features complete, advanced features stubbed |
| Performance | 6/10 | No spatial acceleration structures implemented |
| Test Coverage | Unknown | No tests in analyzed files |

---

## Conclusion

The engine/simulation/physics and engine/simulation/solver modules represent **genuine, working physics simulation code** implementing industry-standard algorithms. The mathematical foundations are correct, numerical stability is properly handled, and multiple solver architectures demonstrate expertise. The code is production-grade for the features implemented, with clear placeholders for unimplemented optimizations (spatial trees, parallel solving).

This is **real implementation work**, not scaffolding or stubs.
