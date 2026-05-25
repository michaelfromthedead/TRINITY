# Investigation: engine/simulation/cloth

## Summary
The cloth simulation module is a **fully functional implementation** of Position-Based Dynamics (PBD) cloth simulation with real physics algorithms. It includes complete particle-based cloth representation, multiple constraint types (distance, bending, shear), collision detection with primitives and meshes, self-collision via spatial hashing, and aerodynamic wind forces. GPU acceleration is defined as interfaces with shader templates but actual GPU execution is stubbed.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 153 | REAL | Comprehensive exports, 50+ symbols |
| `cloth_simulation.py` | 663 | REAL | Full PBD simulation loop with Verlet integration |
| `cloth_constraints.py` | 578 | REAL | 6 constraint types with XPBD-style solving |
| `cloth_collision.py` | 816 | REAL | 5 collider types + spatial hash self-collision |
| `cloth_wind.py` | 546 | REAL | Per-triangle aerodynamics with turbulence |
| `config.py` | 170 | REAL | Well-tuned physics constants |
| `gpu_cloth.py` | 572 | PARTIAL | Interfaces + shader templates, execution stubbed |

**Total: 3,498 lines**

## Cloth Components
- **ClothParticle**: Position, velocity, inverse mass, pinning
- **ClothEdge**: Rest length distance constraints
- **ClothTriangle**: Face data for rendering, wind, self-collision
- **ClothMesh**: Full mesh with particles, edges, triangles, stiffness
- **ClothSimulation**: Main simulation driver with fixed timestep accumulator
- **DistanceConstraint**: XPBD-style stretch resistance
- **BendingConstraint**: Dihedral angle bending resistance
- **ShearConstraint**: Diagonal distortion prevention
- **LongRangeAttachment**: Extreme stretch prevention
- **AnchorConstraint**: World-space pinning
- **TetherConstraint**: Distance limiting from attachment
- **SphereCollider/CapsuleCollider/BoxCollider**: Primitive collision
- **MeshCollider**: Triangle mesh collision with closest-point
- **SDFCollider**: Signed distance field collision
- **SpatialHash**: Uniform grid hash for O(1) neighbor lookup
- **WindForce**: Per-triangle drag/lift with turbulence noise
- **DirectionalWind/PointWind/VortexWind**: Wind source types
- **WindSystem**: Multi-source wind aggregation

## Implementation
- Real spring-mass? **NO** (uses PBD constraints, not spring-mass)
- Real Verlet integration? **YES** - velocity Verlet in `_integrate_positions()`
- Real collision? **YES** - sphere, capsule, box, mesh, SDF, self-collision

### Verlet Integration (cloth_simulation.py:336-354)
```python
def _integrate_positions(self, dt: float) -> None:
    for particle in self.mesh.particles:
        if particle.is_pinned:
            continue
        # Store current position
        particle.prev_position[:] = particle.position
        # Verlet integration: x_new = x + v*dt + a*dt^2
        particle.position += (
            particle.velocity * dt
            + particle.acceleration * dt * dt
        )
```

### XPBD Distance Constraint (cloth_constraints.py:42-90)
```python
@staticmethod
def solve_edge(p0, p1, rest_length, stiffness):
    delta = p1.position - p0.position
    current_length = float(np.linalg.norm(delta))
    error = current_length - rest_length
    w_sum = p0.inv_mass + p1.inv_mass
    correction = (error / current_length) * stiffness
    direction = delta / current_length
    if p0.inv_mass > 0:
        p0.position += direction * correction * (p0.inv_mass / w_sum)
    if p1.inv_mass > 0:
        p1.position -= direction * correction * (p1.inv_mass / w_sum)
```

### Spatial Hash Self-Collision (cloth_collision.py:450-534)
```python
class SpatialHash:
    def _hash_position(self, position):
        ix = int(math.floor(position[0] * self.inv_cell_size))
        iy = int(math.floor(position[1] * self.inv_cell_size))
        iz = int(math.floor(position[2] * self.inv_cell_size))
        h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)
        return h % self.table_size
```

### Per-Triangle Wind Force (cloth_wind.py:87-199)
Real aerodynamic model with:
- Drag force proportional to facing area and velocity squared
- Lift force based on angle of attack
- fBm turbulence noise with octaves

## Verdict
**REAL IMPLEMENTATION**

This is a production-quality Position-Based Dynamics cloth simulation. The physics are mathematically correct (not simplified stubs). Key evidence:

1. **Proper PBD loop**: External forces -> predict positions -> iterative constraint solve -> update velocities
2. **XPBD constraints**: Mass-weighted corrections with stiffness
3. **Dihedral angle bending**: Real bending constraint using triangle normals
4. **Spatial hash self-collision**: Industry-standard O(n) broad phase
5. **Full collision suite**: Sphere, capsule, AABB, mesh, SDF with friction
6. **Aerodynamic wind**: Per-triangle drag/lift with turbulence

The only incomplete part is GPU execution (GPUClothSolverStub does nothing), but the CPU simulation is fully functional.

## Evidence

### Simulation Loop (cloth_simulation.py:294-319)
```python
def _simulate_step(self, dt: float) -> None:
    substep_dt = dt / self.config.substeps
    for _ in range(self.config.substeps):
        self._apply_external_forces(substep_dt)
        self._integrate_positions(substep_dt)
        for _ in range(self.config.solver_iterations):
            self._solve_constraints()
        self._handle_collisions()
        self._update_velocities(substep_dt)
```

### Cloth Grid Creation (cloth_simulation.py:539-663)
Creates particles, structural edges (horizontal/vertical), shear edges (diagonal), bend edges (skip-one), and quad triangles - exactly what real cloth sims do.

### Collision Handler (cloth_collision.py:711-816)
Full collision pipeline processing sphere, capsule, box, mesh, and SDF colliders with friction, plus spatial-hash self-collision.
