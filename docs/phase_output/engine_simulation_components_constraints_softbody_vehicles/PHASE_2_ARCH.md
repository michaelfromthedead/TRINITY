# PHASE 2 ARCHITECTURE: Constraint Solvers

**Scope**: engine/simulation/constraints (6 files, ~3,311 lines)  
**Focus**: Joint constraints, contact manifolds, constraint solving infrastructure

---

## Subsystem Overview

The constraints subsystem provides a unified framework for all physics constraints using Jacobian-based formulation:

| File | Lines | Constraint Type | Key Feature |
|------|-------|-----------------|-------------|
| joint_base.py | 546 | Abstract base | Effective mass, warm starting |
| joint_d6.py | 657 | 6-DOF configurable | Per-axis motion modes |
| joint_hinge.py | 486 | Revolute (hinge) | Single axis rotation, limits |
| joint_slider.py | 518 | Prismatic (slider) | Single axis translation |
| joint_spring.py | 499 | Spring-damper | Soft constraint CFM/ERP |
| contact_constraint.py | 605 | Contact manifold | Normal + friction, Baumgarte |

---

## Architecture Decisions

### ADR-CON-001: Jacobian-Based Constraint Formulation

**Context**: Need unified treatment for all constraint types (joints, contacts, limits).

**Decision**: Use Jacobian matrices mapping constraint space to body space.

**Rationale**:
- Single solver handles all constraint types
- Effective mass computation is generic
- Warm starting accelerates convergence
- Standard approach used by Bullet, PhysX, Box2D

**Implementation**:
```
J = constraint jacobian (maps velocity to constraint velocity)
M_eff = 1 / (J * M_inv * J^T)
lambda = M_eff * (bias - J * v)
```

### ADR-CON-002: Baumgarte Stabilization

**Context**: Position errors accumulate over time, causing constraint drift.

**Decision**: Use velocity-level Baumgarte stabilization.

**Rationale**:
- Adds bias term to velocity constraint
- Single-pass correction (no separate position solver)
- Tunable beta parameter (typically 0.1-0.3)
- Simpler than post-stabilization or pseudo-velocity methods

**Formula**:
```
bias = (beta / dt) * position_error
```

### ADR-CON-003: Warm Starting

**Context**: Iterative constraint solver converges slowly from zero impulse.

**Decision**: Cache and reuse impulses from previous frame.

**Rationale**:
- Reduces iteration count by 50%+
- Impulses change slowly between frames
- Minimal memory overhead (one float per constraint row)
- Standard practice in all modern physics engines

**Implementation**:
```
# At start of solve
apply_impulse(cached_lambda)

# After solve
cached_lambda = accumulated_lambda
```

### ADR-CON-004: CFM/ERP for Soft Constraints

**Context**: Springs and dampers need soft constraint behavior.

**Decision**: Use Constraint Force Mixing (CFM) and Error Reduction Parameter (ERP).

**Rationale**:
- Converts spring-damper to constraint solver terms
- gamma and beta coefficients derived from k and c
- Works within existing impulse-based solver
- ODE physics engine validated this approach

**Derivation**:
```
gamma = 1.0 / (h * (c + h * k))
beta = h * k * gamma
effective_mass = 1.0 / (J * M_inv * J^T + gamma)
```

### ADR-CON-005: Tangent Basis Construction (Duff Method)

**Context**: Contact friction requires two perpendicular tangent vectors.

**Decision**: Use Duff method for tangent basis construction.

**Rationale**:
- Avoids singularity at axis-aligned normals
- Single branch based on dominant normal component
- Numerically stable
- Fast (two cross products)

**Implementation**:
```python
if abs(normal[0]) > 0.9:
    t1 = cross(normal, [0, 1, 0])
else:
    t1 = cross(normal, [1, 0, 0])
t1 = normalize(t1)
t2 = cross(normal, t1)
```

---

## Data Flow

```
Collision Detection
     |
     v (contact manifolds)
+----------------------+
| Constraint Manager   |
+----------------------+
     |
     +---> JointBase (abstract)
     |          |
     |          +--> Jacobian computation
     |          +--> Effective mass
     |          +--> Warm starting
     |          +--> Break detection
     |
     +---> JointD6
     |          |
     |          +--> Per-axis motion config
     |          +--> Positional motors
     |          +--> Angular motors
     |
     +---> JointHinge
     |          |
     |          +--> Single rotation axis
     |          +--> Angular limits
     |          +--> Motor with velocity target
     |
     +---> JointSlider
     |          |
     |          +--> Single translation axis
     |          +--> Linear limits
     |          +--> Motor force
     |
     +---> JointSpring
     |          |
     |          +--> CFM/ERP computation
     |          +--> Soft constraint solve
     |          +--> Rest length with hysteresis
     |
     +---> ContactConstraint
              |
              +--> Normal constraint (Baumgarte)
              +--> Tangent basis (Duff)
              +--> Coulomb friction cone
              +--> Contact persistence
```

---

## Interface Contracts

### JointBase

```python
class JointBase:
    def prepare(self, dt: float) -> None:
        """Compute Jacobians and effective mass for this timestep."""
    
    def warm_start(self) -> None:
        """Apply cached impulses from previous frame."""
    
    def solve_velocity(self) -> None:
        """One iteration of velocity constraint solve."""
    
    def solve_position(self) -> float:
        """Return position error after correction attempt."""
    
    def is_broken(self) -> bool:
        """Return True if joint force/torque exceeds break threshold."""
```

### JointD6

```python
class JointD6(JointBase):
    def set_motion(self, axis: int, mode: MotionType) -> None:
        """Set motion type (LOCKED, LIMITED, FREE) for axis 0-5."""
    
    def set_limit(self, axis: int, lower: float, upper: float) -> None:
        """Set motion limits for given axis."""
    
    def set_drive(self, axis: int, stiffness: float, damping: float, target: float) -> None:
        """Configure motor drive for axis."""
```

### ContactConstraint

```python
class ContactConstraint:
    def set_manifold(self, contacts: List[ContactPoint]) -> None:
        """Set contact points for this frame."""
    
    def warm_start(self) -> None:
        """Apply cached normal and tangent impulses."""
    
    def solve_velocity(self) -> None:
        """Solve normal and friction constraints."""
    
    def get_separating_velocity(self) -> float:
        """Return relative velocity along contact normal."""
```

### JointSpring

```python
class JointSpring(JointBase):
    def set_stiffness(self, k: float) -> None:
        """Set spring stiffness coefficient."""
    
    def set_damping(self, c: float) -> None:
        """Set damping coefficient."""
    
    def set_rest_length(self, length: float) -> None:
        """Set spring rest length."""
```

---

## Solver Configuration

### Iteration Counts

| Constraint Type | Velocity Iterations | Position Iterations |
|-----------------|---------------------|---------------------|
| Contact | 4-8 | 2-4 |
| Hinge | 4-6 | 2 |
| D6 | 6-10 | 2-4 |
| Spring | 4 | 0 (soft) |

### Solver Order

1. Warm start all constraints
2. Velocity iterations (Gauss-Seidel)
3. Position iterations (Baumgarte or pseudo-velocity)
4. Cache impulses for next frame

---

## Dependencies

### Internal
- engine/math: Vector3, Matrix3, Quaternion
- engine/collision: Contact manifolds

### External
- NumPy: Matrix operations, linear algebra

---

## Performance Considerations

### Warm Starting Impact
- 30-50% reduction in iterations needed
- Memory: 1-6 floats per constraint row
- Cache locality: store impulses adjacent to constraint data

### Effective Mass Computation
- Computed once per frame during prepare()
- Matrix inversion via Cholesky for SPD matrices
- 3x3 inversion for most joints

### Contact Persistence
- ID matching reduces impulse lookup overhead
- Persistent contacts reuse previous frame data
- New contacts interpolate from neighbors
