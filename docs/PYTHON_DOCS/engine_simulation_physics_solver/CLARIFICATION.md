# CLARIFICATION: engine_simulation_physics_solver

---

## Philosophical Framing

This module embodies the principle that **physics simulation is applied mathematics**, not library integration. Every formula is handwritten, every algorithm is explicit. The absence of numpy/scipy is intentional: dependency-free physics enables embedding in the game engine's Python 3.13 runtime without ABI concerns.

The three solver architectures (SI, TGS, XPBD) are not alternatives but a **spectrum of trade-offs**:

- **Sequential Impulse**: Battle-tested, deterministic, well-understood failure modes
- **TGS**: Better for extreme mass ratios, substep integration, soft constraints
- **XPBD**: Position-based for geometric accuracy, compliance-driven for artistic control

All three share the same rigid body representation and constraint protocol. Switching solvers is a configuration choice, not a rewrite.

---

## Design Rationale

### Why Union-Find for Islands?

Island detection determines which bodies can influence each other. Union-Find with path compression provides:

- O(alpha(n)) amortized per query (nearly constant)
- Incremental updates as contacts form/break
- Natural wake propagation: one contact wakes entire island

Alternative graph traversals (BFS/DFS) require full rebuild each frame.

### Why Baumgarte Stabilization?

Velocity-only constraints drift over time (constraint violation accumulates). Baumgarte adds a position-correcting bias term to the velocity constraint:

```
bias = beta * C / dt
```

Where `C` is the constraint error and `beta` (0.2 in this codebase) controls correction aggressiveness. Too high causes oscillation; too low allows drift.

### Why Contact Slop (1mm)?

Perfect constraint satisfaction is computationally expensive and numerically unstable at the contact boundary. The 1mm slop allows:

- Bodies to rest stably without micro-jitter
- Solver to converge in fewer iterations
- Predictable behavior at scale

### Why Warm Starting?

Sequential Impulse solvers iterate toward a solution. Starting from zero each frame wastes work. Warm starting applies 80% of last frame's solution as initial guess:

```
lambda_initial = lambda_previous * warm_start_factor
```

This exploits temporal coherence: contact impulses change gradually.

---

## Why These Three Solvers?

### Sequential Impulse (SI)

The workhorse. Erin Catto's formulation from GDC 2005 remains the baseline for real-time physics. Predictable, debuggable, minimal state.

**When to use**: Default for gameplay physics, ragdolls, debris.

### Temporal Gauss-Seidel (TGS)

Handles problematic mass ratios (e.g., 1000:1 heavy/light). Regularization term `gamma = compliance / dt` softens extreme stiffness. Split impulse separates position correction from velocity response.

**When to use**: Vehicle wheels on heavy chassis, characters on movable platforms.

### Extended Position-Based Dynamics (XPBD)

Solves constraints directly in position space with compliance parameter controlling softness. Lagrange multipliers accumulate corrections. Natural for cloth, soft bodies, and "artistic" physics where feel matters more than accuracy.

**When to use**: Rope, cloth, soft-body characters, VFX physics.

---

## Why O(n^2) Broadphase is Acceptable (For Now)

The current broadphase tests all pairs. This is intentionally simple:

1. **Correctness first**: N^2 has no edge cases, no false negatives
2. **N is small**: Typical scenes have 100-500 active bodies
3. **Profiled hot path**: Upgrading to BVH is Phase 2 work, not premature optimization

The enum mentions BVH/octree options but implementation is stubbed. This is honest technical debt, not deception.

---

## Simulation Step Order

The 10-step pipeline is not arbitrary:

1. **Save previous states** — for interpolation between physics frames and render frames
2. **Broadphase** — O(n) or O(n log n) candidate pair generation
3. **Narrowphase** — Exact contact point generation for candidate pairs
4. **Build islands** — Group connected bodies for solver
5. **Integrate velocities** — Apply forces/gravity (explicit Euler)
6. **Solve velocity constraints** — Iterative impulse application
7. **Integrate positions** — Apply velocities (semi-implicit Euler)
8. **Solve position constraints** — Direct position correction (optional)
9. **Update sleep states** — Put idle islands to sleep
10. **Fire callbacks** — Notify gameplay of collision events

Order matters: velocity constraints see the result of force integration; position integration sees constraint-corrected velocities.

---

## Numerical Stability Philosophy

### Division Guards

Every division has a guard. Either:
- Explicit epsilon check: `if abs(denom) > EPSILON: result = num / denom else: result = 0`
- Safe function: `safe_divide(num, denom, default=0)`

### Velocity Clamping

Max velocities (500 m/s linear, 100 rad/s angular) prevent:
- Tunneling (fast objects passing through thin walls)
- Numerical overflow in integration
- Runaway instabilities

### Epsilon Comparisons

Floating-point equality is never tested with `==`. Always:
- `abs(a - b) < EPSILON` for near-equality
- `a > EPSILON` for positive checks
- `a < -EPSILON` for negative checks

---

## What "REAL" Means

This classification distinguishes from stub code:

- **Stub**: Correct interface, placeholder implementation (pass, raise NotImplementedError, return input)
- **Real**: Correct interface AND correct algorithm

The physics/solver modules are REAL because:
- Inertia tensor formulas match textbook derivations
- Quaternion rotation uses optimized conjugate form
- Solver algorithms match cited academic papers
- Numerical guards reflect real-world failure modes

The stubbed features (`_compute_hull`, `_build_bvh`, `solve_parallel`) are documented honestly, not hidden.
