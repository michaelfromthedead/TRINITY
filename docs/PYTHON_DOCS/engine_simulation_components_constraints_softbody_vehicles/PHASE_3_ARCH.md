# PHASE 3 ARCHITECTURE: Soft Body Simulation

**Scope**: engine/simulation/softbody (5 files, ~3,283 lines)  
**Focus**: PBD, FEM, shape matching, muscle simulation, mesh embedding

---

## Subsystem Overview

The softbody subsystem provides three distinct simulation approaches plus support infrastructure:

| File | Lines | Approach | Use Case |
|------|-------|----------|----------|
| soft_body_pbd.py | 766 | Position-Based Dynamics | Cloth, jelly, ropes |
| fem_solver.py | 724 | Finite Element Method | Elastic solids, tissue |
| shape_matching.py | 621 | Clustered Shape Matching | Character flesh, impact |
| muscle.py | 590 | Hill-type muscle model | Biomechanics, procedural |
| deformable_mesh.py | 582 | Mesh embedding | Surface-to-volume mapping |

---

## Architecture Decisions

### ADR-SOFT-001: Multiple Solver Strategies

**Context**: Different deformable objects have different fidelity/performance requirements.

**Decision**: Provide three distinct soft body solvers.

**Rationale**:

| Solver | Fidelity | Performance | Best For |
|--------|----------|-------------|----------|
| PBD | Low-Medium | Fast | Cloth, simple deformables |
| FEM | High | Slow | Accurate materials, tissue |
| Shape Matching | Medium | Fast | Visual deformation, flesh |

Developers choose based on object requirements.

### ADR-SOFT-002: PBD with Compliance-Based Constraints

**Context**: PBD stiffness is tied to iteration count, making tuning difficult.

**Decision**: Use Extended PBD (XPBD) with compliance parameter.

**Rationale**:
- Compliance = 1/stiffness separates stiffness from iteration count
- More predictable material behavior
- Matches modern PBD literature (Macklin 2016)

**Implementation**:
```
lambda = -C / (sum(w_i * grad_i^2) + compliance / dt^2)
```

### ADR-SOFT-003: FEM with Multiple Material Models

**Context**: Different elastic materials require different constitutive models.

**Decision**: Support Neo-Hookean, Corotational, and St. Venant-Kirchhoff materials.

**Rationale**:

| Model | Behavior | Use Case |
|-------|----------|----------|
| Neo-Hookean | Handles large deformation | Rubber, flesh |
| Corotational | Fast, linear in rotation frame | General purpose |
| St. Venant-Kirchhoff | Simple, can invert | Small deformation |

**Deformation Gradient**:
```
F = Ds * Dm_inv
```
where Ds is deformed shape matrix, Dm_inv is inverse rest shape matrix.

### ADR-SOFT-004: SVD-Based Polar Decomposition

**Context**: Corotational FEM needs rotation extraction from deformation gradient.

**Decision**: Use SVD for polar decomposition: F = R * S.

**Rationale**:
- Numerically stable
- Handles inverted elements gracefully
- R = U * V^T gives orthogonal rotation matrix
- S = V * Sigma * V^T gives symmetric stretch

### ADR-SOFT-005: Clustered Shape Matching

**Context**: Shape matching per-vertex is expensive; need spatial coherence.

**Decision**: Use clustered shape matching with skinning blending.

**Rationale**:
- Clusters reduce matrix computations
- Overlapping clusters enable smooth deformation
- Skinning weights blend cluster contributions
- O(n_clusters) instead of O(n_vertices) for matching step

**Implementation**:
```
Apq = sum(m * outer(p - cm, q - rest_cm))  # per cluster
R = polar_decomposition(Apq @ Aqq_inv)
```

### ADR-SOFT-006: Hill-Type Muscle Model

**Context**: Biomechanical simulation needs physiologically accurate muscles.

**Decision**: Implement Hill-type muscle with force-length and force-velocity relationships.

**Rationale**:
- Phenomenological model captures muscle behavior
- Closed-form equations (no PDE solving)
- Parameters map to measurable properties
- Standard in biomechanics (Hill 1938)

**Force Equation**:
```
F_active = activation * F_max * f_l(length) * f_v(velocity)
F_total = F_active + F_passive + F_series_elastic
```

---

## Data Flow

```
Game State
     |
     v (positions, velocities)
+----------------------+
| Soft Body Manager    |
+----------------------+
     |
     +---> SoftBodyPBD
     |          |
     |          +--> Predict positions
     |          +--> Solve distance constraints
     |          +--> Solve volume constraint
     |          +--> Apply strain limiting
     |          +--> Update velocities
     |
     +---> FEMSolver
     |          |
     |          +--> Compute deformation gradients
     |          +--> Evaluate strain energy
     |          +--> Compute stress (P = dW/dF)
     |          +--> Assemble forces
     |          +--> Integrate (Newton iteration)
     |
     +---> ShapeMatching
     |          |
     |          +--> Cluster center of mass
     |          +--> Build Apq, Aqq matrices
     |          +--> SVD polar decomposition
     |          +--> Blend cluster rotations
     |          +--> Update positions
     |
     +---> Muscle
     |          |
     |          +--> Get activation signal
     |          +--> Compute fiber length/velocity
     |          +--> Force-length relationship
     |          +--> Force-velocity relationship
     |          +--> Apply muscle force
     |
     +---> DeformableMesh
              |
              +--> Embed surface in tet mesh
              +--> Interpolate surface positions
              +--> Recompute normals
              +--> Generate collision proxy
```

---

## Interface Contracts

### SoftBodyPBD

```python
class SoftBodyPBD:
    def step(self, dt: float) -> None:
        """Advance simulation: predict, solve, update."""
    
    def add_distance_constraint(self, i: int, j: int, rest_length: float, compliance: float) -> None:
        """Add distance constraint between particles i and j."""
    
    def set_external_force(self, forces: np.ndarray) -> None:
        """Set external forces (gravity, wind) for all particles."""
    
    def pin_particle(self, index: int) -> None:
        """Pin particle (infinite mass)."""
```

### FEMSolver

```python
class FEMSolver:
    def step(self, dt: float) -> None:
        """One timestep of FEM solve (Newton iterations)."""
    
    def set_material(self, material: MaterialModel) -> None:
        """Set constitutive model (NeoHookean, Corotational, StVK)."""
    
    def get_stress(self, tet_index: int) -> np.ndarray:
        """Return 3x3 stress tensor for tetrahedron."""
    
    def get_strain_energy(self) -> float:
        """Return total elastic potential energy."""
```

### ShapeMatching

```python
class ShapeMatching:
    def step(self, dt: float) -> None:
        """Update positions via shape matching."""
    
    def create_clusters(self, radius: float, overlap: float) -> None:
        """Generate clusters with given radius and overlap."""
    
    def set_stiffness(self, alpha: float) -> None:
        """Set shape matching stiffness (0-1)."""
```

### Muscle

```python
class Muscle:
    def set_activation(self, a: float) -> None:
        """Set muscle activation level (0-1)."""
    
    def compute_force(self) -> float:
        """Return current muscle force based on length and velocity."""
    
    def get_fiber_length(self) -> float:
        """Return current fiber length."""
```

### DeformableMesh

```python
class DeformableMesh:
    def embed(self, surface_mesh: Mesh, tet_mesh: TetMesh) -> None:
        """Embed surface vertices in tetrahedral mesh."""
    
    def update(self) -> None:
        """Update surface positions from tet mesh deformation."""
    
    def get_positions(self) -> np.ndarray:
        """Return current surface vertex positions."""
    
    def get_normals(self) -> np.ndarray:
        """Return recomputed surface normals."""
```

---

## Material Models

### Neo-Hookean

```python
# Strain energy density
W = mu/2 * (I_C - 3) - mu * log(J) + lambda/2 * log(J)^2

# First Piola-Kirchhoff stress
P = mu * (F - F_inv_T) + lambda * log(J) * F_inv_T
```

### Corotational

```python
# Extract rotation via polar decomposition
F = R * S

# Strain in rotation frame
epsilon = S - I

# Stress (linear in rotation frame)
P = R * (2*mu*epsilon + lambda*tr(epsilon)*I)
```

### St. Venant-Kirchhoff

```python
# Green strain
E = 0.5 * (F^T * F - I)

# Second Piola-Kirchhoff stress
S = 2*mu*E + lambda*tr(E)*I

# First Piola-Kirchhoff
P = F * S
```

---

## Dependencies

### Internal
- engine/math: Vector3, Matrix3, Quaternion
- engine/mesh: TetMesh, SurfaceMesh

### External
- NumPy: Linear algebra
- (Optional) SciPy: Sparse matrix solvers for implicit FEM

---

## Performance Considerations

### PBD Iteration Count
- Distance constraints: 10-20 iterations typical
- Volume constraint: 1 iteration per PBD iteration
- Parallel constraint projection for GPU

### FEM Newton Convergence
- 3-5 Newton iterations typical
- Polar decomposition is bottleneck
- Batch SVD for GPU acceleration

### Shape Matching
- O(clusters) per frame, not O(vertices)
- Cluster radius 10-20% of object size
- 50% overlap for smooth blending
