# PHASE 1 ARCHITECTURE: Simulation Components

**Scope**: engine/simulation/components (6 files, ~3,406 lines)  
**Focus**: Cloth, Colliders, Fluid, Destruction, Character, Vehicle components

---

## Subsystem Overview

The components subsystem provides simulation behaviors that attach to game entities. Each component encapsulates a specific physics domain:

| Component | Lines | Domain | Key Algorithm |
|-----------|-------|--------|---------------|
| cloth_component.py | 607 | Fabric simulation | PBD distance constraints |
| collider_components.py | 587 | Collision shapes | Inertia tensors, support mapping |
| fluid_component.py | 581 | Buoyancy and drag | Archimedes principle |
| destruction_component.py | 559 | Fracture simulation | Voronoi tessellation |
| character_component.py | 542 | Character physics | Ground detection, slope handling |
| vehicle_component.py | 530 | Vehicle basics | Suspension, engine torque |

---

## Architecture Decisions

### ADR-SIM-001: PBD for Cloth Simulation

**Context**: Cloth simulation requires stable, real-time deformation with wind and collision response.

**Decision**: Use Position-Based Dynamics with distance constraints.

**Rationale**:
- Unconditionally stable regardless of timestep
- Inverse mass weighting handles pinned vertices naturally
- Iterative solver converges predictably
- Self-collision via spatial hashing is straightforward

**Implementation Details**:
```
correction = (current_length - rest_length) / current_length * delta
p1 += w1 / (w1 + w2) * correction * stiffness
```

### ADR-SIM-002: Analytical Inertia Tensors

**Context**: Collision shapes need inertia tensors for rigid body dynamics.

**Decision**: Compute inertia tensors analytically per primitive type.

**Rationale**:
- Sphere, box, capsule have closed-form solutions
- Avoids Monte Carlo sampling overhead
- Exact values for primitive shapes
- Mesh colliders fall back to convex hull approximation

**Formulas**:
- Sphere: `I = 2/5 * m * r^2 * Identity`
- Box: `I_xx = m/12 * (h_y^2 + h_z^2)`, etc.
- Capsule: Cylinder + hemisphere contributions

### ADR-SIM-003: Distributed Buoyancy Sampling

**Context**: Fluid interaction requires computing submerged volume for buoyancy.

**Decision**: Use multiple sample points distributed across object volume.

**Rationale**:
- Handles partial submersion correctly
- Produces realistic bobbing and rolling
- Linear approximation sufficient for game physics
- Configurable sample density for quality/performance tradeoff

### ADR-SIM-004: Voronoi-Based Fracture

**Context**: Destructible objects need convincing fracture patterns.

**Decision**: Generate fragments via Voronoi tessellation from impact point.

**Rationale**:
- Voronoi produces natural-looking shards
- Seed distribution controls fragment count
- Pre-fractured meshes can be cached
- Connectivity graph enables structural integrity simulation

---

## Data Flow

```
Entity System
     |
     v
+--------------------+
| Simulation Manager |
+--------------------+
     |
     +---> ClothComponent
     |          |
     |          +--> PBD Solver (distance, volume constraints)
     |          +--> Wind Force Application
     |          +--> Self-Collision (spatial hash)
     |
     +---> ColliderComponent
     |          |
     |          +--> Shape Definition (sphere, box, capsule, mesh)
     |          +--> Inertia Tensor Computation
     |          +--> Support Mapping for GJK
     |
     +---> FluidComponent
     |          |
     |          +--> Sample Point Distribution
     |          +--> Buoyancy Force (Archimedes)
     |          +--> Drag Force (quadratic law)
     |
     +---> DestructionComponent
     |          |
     |          +--> Damage Accumulation
     |          +--> Voronoi Seed Distribution
     |          +--> Fragment Generation
     |          +--> Connectivity Graph Update
     |
     +---> CharacterComponent
     |          |
     |          +--> Ground Detection (sphere sweep)
     |          +--> Slope Handling
     |          +--> Movement State Machine
     |
     +---> VehicleComponent
              |
              +--> Wheel Raycasting
              +--> Suspension Spring/Damper
              +--> Engine Torque Curves
```

---

## Interface Contracts

### ClothComponent

```python
class ClothComponent:
    def step(self, dt: float) -> None:
        """Advance simulation by timestep."""
    
    def apply_wind(self, direction: np.ndarray, strength: float) -> None:
        """Apply wind force with direction and magnitude."""
    
    def pin_vertex(self, index: int) -> None:
        """Pin vertex (set inverse mass to zero)."""
    
    def tear_at(self, position: np.ndarray) -> None:
        """Initiate tear from position if strain exceeds threshold."""
    
    def get_positions(self) -> np.ndarray:
        """Return current vertex positions (Nx3)."""
```

### ColliderComponent

```python
class ColliderComponent:
    def compute_inertia(self) -> np.ndarray:
        """Return 3x3 inertia tensor matrix."""
    
    def support(self, direction: np.ndarray) -> np.ndarray:
        """Return furthest point in given direction (for GJK)."""
    
    def get_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return axis-aligned bounding box (min, max)."""
```

### FluidComponent

```python
class FluidComponent:
    def compute_forces(self, transform: Transform) -> Tuple[np.ndarray, np.ndarray]:
        """Return (buoyancy_force, torque) for given transform."""
    
    def get_water_height(self, x: float, z: float) -> float:
        """Return water surface height at world position."""
```

### DestructionComponent

```python
class DestructionComponent:
    def apply_damage(self, point: np.ndarray, impulse: float) -> None:
        """Accumulate damage at impact point."""
    
    def is_destroyed(self) -> bool:
        """Return True if damage exceeds threshold."""
    
    def generate_fragments(self) -> List[Mesh]:
        """Generate fragment meshes via Voronoi."""
```

---

## Dependencies

### Internal
- engine/math: Vector3, Matrix3, Quaternion, Transform
- engine/collision: Spatial hash, AABB queries

### External
- NumPy: Linear algebra operations
- (Optional) SciPy: Voronoi tessellation for destruction

---

## Performance Considerations

### Cloth Simulation
- Iteration count scales with mesh connectivity
- Self-collision dominates for dense meshes
- Spatial hashing reduces collision checks from O(n^2) to O(n)

### Inertia Tensors
- Computed once at initialization
- Cached until shape parameters change

### Buoyancy Sampling
- Sample count configurable (default: 8-16 points)
- Wave height evaluated per sample

### Fracture Generation
- Pre-fracture meshes during loading for deterministic patterns
- Runtime fracture limited to low-fragment-count objects
