# Investigation: engine/simulation/solver

## Summary

The solver module contains a fully-implemented physics constraint solver suite with three solver types: Sequential Impulse (SI), Temporal Gauss-Seidel (TGS), and Extended Position Based Dynamics (XPBD). The implementation includes comprehensive Jacobian computation, island-based optimization with Union-Find connectivity detection, warm starting, mass scaling for extreme mass ratios, and split impulse for stability. This is production-grade physics code following industry-standard algorithms from Erin Catto (Box2D) and the XPBD paper (Macklin et al., 2016).

## Files

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 62 | COMPLETE | Exports all solvers, constraints, jacobian utils, island manager |
| `config.py` | 291 | COMPLETE | Full configuration with 30+ tunable parameters, preset profiles |
| `constraint_solver.py` | 726 | COMPLETE | Sequential Impulse solver with RigidBody, BaseConstraint, PointConstraint, AxisConstraint, AngularConstraint |
| `island_manager.py` | 592 | COMPLETE | Island detection with Union-Find, sleep management, parallel solving support |
| `jacobian.py` | 841 | COMPLETE | Vec3, Mat3, Quaternion, Jacobian computation, effective mass, impulse application |
| `tgs_solver.py` | 751 | COMPLETE | TGS solver with mass scaling, regularization, split impulse, stale data detection |
| `xpbd_solver.py` | 792 | COMPLETE | XPBD solver with distance, bending, volume, collision constraints |

**Total: ~4,055 lines of real solver code**

## Solver Components

1. **Sequential Impulse Solver (constraint_solver.py)**
   - Velocity constraint iterations with Gauss-Seidel
   - Position constraint iterations for drift correction
   - Warm starting with cached impulse accumulation
   - Baumgarte stabilization for position correction

2. **TGS Solver (tgs_solver.py)**
   - Substep integration for improved stability
   - Mass scaling for extreme mass ratios (handles 100:1 ratios)
   - Regularization term (gamma) for soft constraints
   - Split impulse to prevent energy injection
   - Stale data detection for warm starting
   - Specialized TGSContactSolver with friction clamping

3. **XPBD Solver (xpbd_solver.py)**
   - Compliance parameter (inverse stiffness)
   - Lagrange multiplier accumulation
   - Distance, bending (angle), volume (tetrahedron), collision constraints
   - Damping support
   - Velocity derivation from position change

4. **Island Manager (island_manager.py)**
   - Union-Find (Disjoint Set Union) with path compression and union by rank
   - Sleep detection based on velocity thresholds
   - Island wake/sleep lifecycle management
   - Parallel group partitioning for multi-threaded solving
   - Event listener interface for island state changes

5. **Jacobian Utilities (jacobian.py)**
   - Vec3, Mat3, Quaternion math primitives
   - Jacobian computation for: point, axis, angular, limit, contact, friction, distance constraints
   - Effective mass matrix computation (including block solver support)
   - Impulse application with inertia tensor transformation
   - Friction tangent basis computation

## Implementation

- Real iterative solver? **YES** - Full Gauss-Seidel iteration with configurable velocity/position iterations (default 8/3)
- Real island splitting? **YES** - Union-Find algorithm with path compression, proper merge/split operations, parallel group computation
- Real PGS/XPBD? **YES** - Complete implementations:
  - PGS: Projected Gauss-Seidel with inequality clamping for contacts (solve_velocity with lambda clamping)
  - TGS: Temporal variant with mass scaling, regularization, split impulse
  - XPBD: Full position-based dynamics with compliance, Lagrange multipliers

## Verdict

**REAL IMPLEMENTATION** - This is a complete, production-ready physics constraint solver module implementing industry-standard algorithms. The code quality is high, with proper abstractions, extensive documentation with academic references, and realistic numerical handling (tolerance checks, clamping, edge cases).

## Evidence

### Sequential Impulse Iteration (constraint_solver.py:335-346)
```python
# Velocity iterations
for i in range(self.config.velocity_iterations):
    self._iteration_count = i
    error = self._solve_velocity_iteration()
    if error < 1e-6:
        break

# Position iterations
for i in range(self.config.position_iterations):
    error = self._solve_position_iteration()
    self._last_error = error
    if error < self.config.slop:
        break
```

### Union-Find Island Detection (island_manager.py:108-126)
```python
def find(self, x: int) -> int:
    if x not in self._parent:
        self.make_set(x)
        return x

    if self._parent[x] != x:
        self._parent[x] = self.find(self._parent[x])  # Path compression
    return self._parent[x]

def union(self, x: int, y: int) -> None:
    root_x = self.find(x)
    root_y = self.find(y)

    if root_x == root_y:
        return

    # Union by rank
    if self._rank[root_x] < self._rank[root_y]:
        self._parent[root_x] = root_y
    elif self._rank[root_x] > self._rank[root_y]:
        self._parent[root_y] = root_x
    else:
        self._parent[root_y] = root_x
        self._rank[root_x] += 1
```

### XPBD Lagrange Multiplier (xpbd_solver.py:218-232)
```python
# XPBD: alpha = compliance / dt^2
alpha = self.compliance / (dt * dt)

# Compute Lagrange multiplier update
# delta_lambda = (-C - alpha * lambda) / (w + alpha)
delta_lambda = (-c - alpha * self._lambda) / (w + alpha)

# Apply damping
if self.damping > 0:
    rel_vel = (self.particle_b.velocity - self.particle_a.velocity).dot(direction)
    damping_force = self.damping * rel_vel / (w + alpha)
    delta_lambda -= damping_force * dt

self._lambda += delta_lambda
```

### TGS Mass Scaling for Extreme Ratios (tgs_solver.py:234-269)
```python
def _compute_mass_scales(self, constraint: Constraint) -> Tuple[float, float]:
    # Compute mass ratio
    ratio = min(mass_a, mass_b) / max(mass_a, mass_b)

    # Apply scaling for extreme ratios (< 0.01)
    if ratio < 0.01:
        # Scale down the lighter body's response
        if mass_a < mass_b:
            scale_factor = math.sqrt(ratio)
            return scale_factor, 1.0
        else:
            scale_factor = math.sqrt(ratio)
            return 1.0, scale_factor

    return 1.0, 1.0
```

### Effective Mass with Inertia (jacobian.py:501-549)
```python
def compute_effective_mass(
    jacobian: Jacobian,
    inv_mass_a: float,
    inv_inertia_a: Mat3,
    inv_mass_b: float,
    inv_inertia_b: Mat3
) -> float:
    # K = J_la . (1/m_a) . J_la + J_aa . I_a^-1 . J_aa
    #   + J_lb . (1/m_b) . J_lb + J_ab . I_b^-1 . J_ab

    k = 0.0

    # Body A contribution
    if inv_mass_a > 0:
        k += inv_mass_a * jacobian.linear_a.dot(jacobian.linear_a)

    ang_a_contrib = inv_inertia_a * jacobian.angular_a
    k += jacobian.angular_a.dot(ang_a_contrib)

    # Body B contribution
    if inv_mass_b > 0:
        k += inv_mass_b * jacobian.linear_b.dot(jacobian.linear_b)

    ang_b_contrib = inv_inertia_b * jacobian.angular_b
    k += jacobian.angular_b.dot(ang_b_contrib)

    if k < 1e-10:
        return 0.0

    return 1.0 / k
```
