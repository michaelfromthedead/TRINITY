# Investigation: engine/simulation/hair

## Summary
The hair simulation module is a fully implemented Position-Based Dynamics (PBD) and Follow-The-Leader (FTL) hair simulation system. It includes complete strand simulation with physics, collision detection against primitives and SDFs, self-collision via density fields, guide hair interpolation for rendering, and a multi-level LOD system with smooth transitions.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 143 | REAL | Clean exports of all components |
| `config.py` | 219 | REAL | Comprehensive physics/LOD constants |
| `hair_simulation.py` | 662 | REAL | Full PBD simulation with verlet integration |
| `hair_collision.py` | 564 | REAL | Capsule, sphere, SDF collision + density field |
| `hair_constraints.py` | 542 | REAL | Length, shape, root, local constraints |
| `hair_lod.py` | 470 | REAL | 4-level LOD system with shell fallback |
| **Total** | **2600** | | |

## Hair Components
- **HairStrand**: Base strand with control points, rest lengths, thickness
- **GuideHair**: Simulated strands (max 1000) that drive motion
- **InterpolatedHair**: Derived strands for rendering (up to 10K)
- **HairControlPoint**: Position, velocity, inverse mass (0 = fixed root)
- **HairSimulation**: Main simulation loop with fixed timestep
- **HairCollisionSystem**: Multi-primitive collision manager
- **HairDensityField**: 3D grid for self-collision
- **HairLODSystem**: Distance-based quality reduction
- **Constraint classes**: Length, GlobalShape, LocalShape, Root, Collision

## Implementation
- Real hair strands? **YES** - Full strand representation with control points, rest poses, verlet integration
- Real follow curves? **YES** - FTL (Follow-The-Leader) constraint solving, global/local shape matching
- Real collision? **YES** - Capsule, sphere, SDF collision with friction; density-field self-collision

## Verdict
**REAL IMPLEMENTATION**

This is a production-quality hair simulation system implementing industry-standard techniques:
- Position-Based Dynamics (PBD) for stability
- Follow-The-Leader (FTL) for efficient constraint solving
- Inertia transfer from head motion
- Wind forces with configurable strength
- Multi-collider support (capsules for body approximation, SDFs for detail)
- Self-collision via density field grid
- Guide/interpolated hair architecture for performance
- 4-level LOD with hysteresis and shell rendering fallback
- Quality presets (Ultra/High/Medium/Low/Mobile)

## Evidence

### Verlet Integration + FTL Constraint Solving
```python
def _simulate_step(self, dt: float) -> None:
    # Apply inertia from head motion
    self._apply_inertia_from_head_motion(dt)

    # Apply external forces (gravity, wind)
    self._apply_external_forces(dt)

    # Integrate positions
    self._integrate_positions(dt)

    # Solve constraints
    for _ in range(self.config.solver_iterations):
        self._solve_constraints()

    # Handle collisions
    if self.config.enable_collision:
        self._handle_collisions()

    # Update velocities
    self._update_velocities(dt)
```

### Capsule Collision with Friction
```python
def collide_point_with_capsule(
    point: "HairControlPoint",
    capsule_a: NDArray[np.float32],
    capsule_b: NDArray[np.float32],
    capsule_radius: float,
    margin: float = HAIR_COLLISION_MARGIN,
    friction: float = 0.3,
) -> HairCollisionResult:
    # Find closest point on capsule axis
    t = np.dot(point.position - capsule_a, axis) / axis_len_sq
    t = float(np.clip(t, 0.0, 1.0))
    closest = capsule_a + t * axis
    # ... compute penetration and apply correction with friction
```

### FTL Length Constraint
```python
def solve_length_constraint(cp0, cp1, rest_length, stiffness=1.0):
    """FTL style: only the child point moves."""
    delta = cp1.position - cp0.position
    current_length = float(np.linalg.norm(delta))
    error = current_length - rest_length
    if cp1.inv_mass > 0:
        direction = delta / current_length
        correction = error * stiffness
        cp1.position -= direction * correction
```

### Self-Collision via Density Field
```python
if self._enable_self_collision and self._density_field:
    density = self._density_field.sample_density(cp.position)
    if density > SELF_COLLISION_DENSITY_THRESHOLD:
        gradient = self._density_field.sample_gradient(cp.position)
        excess_density = density - SELF_COLLISION_DENSITY_THRESHOLD
        push = gradient / grad_len * excess_density * SELF_COLLISION_PUSH_STRENGTH
        cp.position += push
```

### LOD System with Shell Fallback
```python
class HairLODLevel(Enum):
    HIGH = auto()    # Full quality
    MEDIUM = auto()  # Reduced guides (50%)
    LOW = auto()     # Minimal guides (25%)
    SHELL = auto()   # Shell-based fallback (0% - no strands)
```
