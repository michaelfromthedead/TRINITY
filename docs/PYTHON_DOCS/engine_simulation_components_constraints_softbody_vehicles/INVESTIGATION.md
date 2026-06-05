# Archaeological Investigation: engine/simulation

**Date**: 2026-05-22  
**Scope**: components, constraints, softbody, vehicles  
**Files Examined**: 23  
**Total Lines**: ~14,944

---

## Classification Summary

| Directory | Files | Classification | Evidence Level |
|-----------|-------|----------------|----------------|
| components | 6 | ALL REAL | High |
| constraints | 6 | ALL REAL | High |
| softbody | 5 | ALL REAL | High |
| vehicles | 6 | ALL REAL | High |

**Overall Verdict**: ALL 23 FILES ARE REAL IMPLEMENTATIONS

---

## engine/simulation/components (6 files, ~3,406 lines)

### cloth_component.py (607 lines) - REAL

**Purpose**: Cloth simulation using Position-Based Dynamics

**Key Algorithms**:
- PBD distance constraints with inverse mass weighting
- Wind force with normal-based pressure and friction components
- Tearing via strain threshold monitoring
- Self-collision with spatial hashing

**Evidence**:
```python
# Distance constraint projection
def _solve_distance_constraint(self, p1_idx: int, p2_idx: int, rest_length: float):
    delta = p2 - p1
    current_length = np.linalg.norm(delta)
    correction = (current_length - rest_length) / current_length * delta
    # Inverse mass weighting
    p1 += w1 / (w1 + w2) * correction * self.stiffness
```

### collider_components.py (587 lines) - REAL

**Purpose**: Physics collision shapes with inertia tensor computation

**Key Algorithms**:
- Sphere, Box, Capsule, Mesh collider implementations
- Analytical inertia tensor computation per shape type
- Support mapping for GJK/EPA collision detection
- Convex hull construction for mesh colliders

**Evidence**:
```python
# Box inertia tensor
def compute_inertia(self) -> np.ndarray:
    m = self.mass
    h = self.half_extents
    return np.diag([
        m/12 * (h[1]**2 + h[2]**2),
        m/12 * (h[0]**2 + h[2]**2),
        m/12 * (h[0]**2 + h[1]**2)
    ])
```

### fluid_component.py (581 lines) - REAL

**Purpose**: Fluid interaction with buoyancy and drag

**Key Algorithms**:
- Archimedes principle with distributed sample points
- Quadratic drag law (Cd * rho * A * v^2 / 2)
- Wave height function with multiple frequency components
- Surface tension approximation

**Evidence**:
```python
# Archimedes buoyancy
submerged_volume = self._compute_submerged_volume(transform)
buoyancy_force = self.fluid_density * gravity * submerged_volume
```

### destruction_component.py (559 lines) - REAL

**Purpose**: Destructible objects with fracture simulation

**Key Algorithms**:
- Voronoi-based fracture pattern generation
- Damage propagation with stress tensor analysis
- Fragment mass/inertia computation
- Connectivity graph for structural integrity

**Evidence**:
```python
# Voronoi fracture
def _generate_fracture_pattern(self, impact_point: np.ndarray):
    seeds = self._distribute_voronoi_seeds(impact_point, self.fragment_count)
    fragments = voronoi_tessellation(self.mesh, seeds)
```

### character_component.py (542 lines) - REAL

**Purpose**: Character controller with physics-based movement

**Key Algorithms**:
- Ground detection with sphere/capsule sweep
- Slope handling with angle-based movement scaling
- Step climbing with height threshold
- Movement state machine (grounded, falling, sliding)

**Evidence**:
```python
# Ground detection
hit = self.physics_world.sweep_sphere(
    self.position, self.position - Vector3(0, self.ground_probe_distance, 0),
    self.capsule_radius
)
self.grounded = hit and hit.normal.dot(Vector3.UP) > self.slope_limit_cos
```

### vehicle_component.py (530 lines) - REAL

**Purpose**: Vehicle physics with wheels, engine, transmission

**Key Algorithms**:
- Wheel ray casting for suspension
- Engine torque curves with RPM mapping
- Gear ratio transmission
- Ackermann steering geometry

**Evidence**:
```python
# Suspension force
compression = (self.rest_length - hit.distance) / self.rest_length
spring_force = compression * self.spring_stiffness
damper_force = -self.damping * compression_velocity
```

---

## engine/simulation/constraints (6 files, ~3,311 lines)

### joint_d6.py (657 lines) - REAL

**Purpose**: 6-DOF configurable joint

**Key Algorithms**:
- Per-axis motion configuration (locked, limited, free)
- Positional motors with target position/velocity
- Angular motors with drive stiffness/damping
- Limit enforcement with soft contacts

**Evidence**:
```python
# Per-axis limit enforcement
for axis in range(3):
    if self.motion[axis] == MotionType.LIMITED:
        violation = position[axis] - self.limits[axis].clamp(position[axis])
        if abs(violation) > 0:
            jacobian[axis] = axis_direction
            self._apply_position_correction(violation, jacobian)
```

### contact_constraint.py (605 lines) - REAL

**Purpose**: Contact manifold with friction constraints

**Key Algorithms**:
- Normal constraint with Baumgarte stabilization
- Tangent basis construction (Duff method)
- Coulomb friction cone approximation
- Contact point persistence with ID matching

**Evidence**:
```python
# Tangent basis (Duff method)
def _compute_tangent_basis(self, normal: np.ndarray):
    if abs(normal[0]) > 0.9:
        t1 = np.cross(normal, [0, 1, 0])
    else:
        t1 = np.cross(normal, [1, 0, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    return t1, t2
```

### joint_base.py (546 lines) - REAL

**Purpose**: Abstract joint with constraint solving infrastructure

**Key Algorithms**:
- Effective mass matrix computation
- Warm starting with accumulated impulses
- Break detection with force/torque thresholds
- Jacobian computation for position/velocity

**Evidence**:
```python
# Effective mass
def _compute_effective_mass(self, jacobian: np.ndarray):
    jm_inv_jt = (
        jacobian @ self.body_a.inv_mass_matrix @ jacobian.T +
        jacobian @ self.body_b.inv_mass_matrix @ jacobian.T
    )
    return np.linalg.inv(jm_inv_jt)
```

### joint_slider.py (518 lines) - REAL

**Purpose**: Prismatic (slider) joint

**Key Algorithms**:
- Slide axis constraint with 5-DOF lock
- Linear limits with restitution
- Motor force with max force clamping
- Perpendicular axis constraints

**Evidence**:
```python
# 5-DOF constraint (lock 2 translations + 3 rotations)
def _build_jacobian(self):
    # Perpendicular translations
    j_perp1 = np.hstack([perp1, np.cross(r_a, perp1), -perp1, -np.cross(r_b, perp1)])
    j_perp2 = np.hstack([perp2, np.cross(r_a, perp2), -perp2, -np.cross(r_b, perp2)])
    # All 3 rotations
    j_rot = self._build_rotation_jacobians()
```

### joint_spring.py (499 lines) - REAL

**Purpose**: Spring-damper joint with soft constraint

**Key Algorithms**:
- Soft constraint with gamma/beta coefficients
- CFM (Constraint Force Mixing) implementation
- ERP (Error Reduction Parameter) for stability
- Rest length with hysteresis

**Evidence**:
```python
# Soft constraint coefficients
gamma = 1.0 / (h * (damping + h * stiffness))
beta = h * stiffness * gamma
effective_mass = 1.0 / (jm_inv_jt + gamma)
bias = beta * position_error
```

### joint_hinge.py (486 lines) - REAL

**Purpose**: Revolute (hinge) joint with limits

**Key Algorithms**:
- Single rotation axis constraint
- Angular limits with restitution
- Ackermann steering for multi-wheel setups
- Motor with velocity target

**Evidence**:
```python
# Angle extraction
def _get_hinge_angle(self) -> float:
    relative_rotation = self.body_b.orientation * self.body_a.orientation.conjugate()
    axis_component = relative_rotation.axis.dot(self.hinge_axis)
    return 2.0 * math.atan2(axis_component * relative_rotation.w, 1.0)
```

---

## engine/simulation/softbody (5 files, ~3,283 lines)

### soft_body_pbd.py (766 lines) - REAL

**Purpose**: Position-Based Dynamics soft body solver

**Key Algorithms**:
- Distance constraints with compliance
- Volume preservation constraint (global)
- Strain limiting to prevent over-extension
- Gauss-Seidel iterative solver

**Evidence**:
```python
# Volume constraint
def _solve_volume_constraint(self):
    current_volume = self._compute_volume()
    volume_error = current_volume - self.rest_volume
    gradients = self._compute_volume_gradients()
    lambda_ = -volume_error / (sum(w * g.dot(g) for w, g in zip(self.inv_masses, gradients)) + self.compliance)
    for i, grad in enumerate(gradients):
        self.positions[i] += self.inv_masses[i] * lambda_ * grad
```

### fem_solver.py (724 lines) - REAL

**Purpose**: Finite Element Method solver for elastic materials

**Key Algorithms**:
- Tetrahedral mesh with shape functions
- Deformation gradient computation (F = Ds * Dm_inv)
- Neo-Hookean, corotational, St. Venant-Kirchhoff materials
- Polar decomposition for corotational

**Evidence**:
```python
# Neo-Hookean strain energy density
def _neo_hookean_stress(self, F: np.ndarray):
    J = np.linalg.det(F)
    F_inv_T = np.linalg.inv(F).T
    P = self.mu * (F - F_inv_T) + self.lambda_ * np.log(J) * F_inv_T
    return P
```

### shape_matching.py (621 lines) - REAL

**Purpose**: Clustered shape matching for soft bodies

**Key Algorithms**:
- Apq/Aqq matrix construction
- SVD polar decomposition for rotation extraction
- Cluster-based region deformation
- Skinning weight blending between clusters

**Evidence**:
```python
# Shape matching step
def _match_shape(self, cluster: Cluster):
    Apq = sum(m * np.outer(p - cluster.cm, q - cluster.rest_cm) for m, p, q in zip(masses, positions, rest_positions))
    Aqq = sum(m * np.outer(q - cluster.rest_cm, q - cluster.rest_cm) for m, q in zip(masses, rest_positions))
    A = Apq @ np.linalg.inv(Aqq)
    R, _ = polar_decomposition(A)  # SVD-based
    return R
```

### muscle.py (590 lines) - REAL

**Purpose**: Hill-type muscle model for biomechanics

**Key Algorithms**:
- Force-length relationship (Gaussian curve)
- Force-velocity relationship (Hill equation)
- Pennation angle for fiber orientation
- Series elastic element

**Evidence**:
```python
# Hill muscle model
def _compute_active_force(self, activation: float, fiber_length: float, fiber_velocity: float):
    # Force-length (Gaussian)
    fl = math.exp(-((fiber_length / self.optimal_length - 1.0) / self.width) ** 2)
    # Force-velocity (Hill)
    if fiber_velocity < 0:  # Concentric
        fv = (self.max_velocity + fiber_velocity) / (self.max_velocity - fiber_velocity / self.a_rel)
    else:  # Eccentric
        fv = (self.max_velocity - fiber_velocity / self.eccentric_factor) / (self.max_velocity + fiber_velocity)
    return activation * self.max_force * fl * fv
```

### deformable_mesh.py (582 lines) - REAL

**Purpose**: Surface mesh embedded in tetrahedral simulation mesh

**Key Algorithms**:
- Barycentric coordinate embedding
- Surface-to-volume mapping
- Normal recomputation after deformation
- Collision proxy generation

**Evidence**:
```python
# Barycentric embedding
def _embed_vertex(self, surface_pos: np.ndarray, tet_idx: int) -> np.ndarray:
    v0, v1, v2, v3 = self.tet_mesh.get_tet_vertices(tet_idx)
    mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
    bary = np.linalg.solve(mat, surface_pos - v0)
    return np.array([1 - bary.sum(), bary[0], bary[1], bary[2]])
```

---

## engine/simulation/vehicles (6 files, ~4,681 lines)

### drivetrain.py (982 lines) - REAL

**Purpose**: Complete vehicle drivetrain simulation

**Key Algorithms**:
- Engine torque curves with polynomial interpolation
- Clutch slip with engagement threshold
- Automatic transmission with shift maps
- Differential types: Open, Limited Slip, Torsen, Locked

**Evidence**:
```python
# Torsen differential
def _torsen_split(self, input_torque: float, speed_left: float, speed_right: float):
    bias_ratio = self.torsen_bias_ratio
    speed_diff = abs(speed_left - speed_right)
    if speed_diff < 0.01:
        return 0.5, 0.5
    faster = 0 if speed_left > speed_right else 1
    bias = min(bias_ratio, 1 + speed_diff * self.torsen_sensitivity)
    split_fast = 1.0 / (1.0 + bias)
    return (split_fast, 1 - split_fast) if faster == 0 else (1 - split_fast, split_fast)
```

### tire_model.py (831 lines) - REAL

**Purpose**: Pacejka Magic Formula tire physics

**Key Algorithms**:
- Pacejka Magic Formula: F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
- Combined slip with weighting functions
- Load sensitivity with vertical force scaling
- Temperature and wear modeling

**Evidence**:
```python
# Pacejka Magic Formula
def _magic_formula(self, slip: float, B: float, C: float, D: float, E: float) -> float:
    Bx = B * slip
    return D * math.sin(C * math.atan(Bx - E * (Bx - math.atan(Bx))))

def compute_lateral_force(self, slip_angle: float, vertical_load: float):
    D = self.peak_lateral * vertical_load / self.nominal_load
    return self._magic_formula(slip_angle, self.B_lat, self.C_lat, D, self.E_lat)
```

### wheeled_vehicle.py (762 lines) - REAL

**Purpose**: Complete wheeled vehicle dynamics

**Key Algorithms**:
- Ackermann steering geometry
- Anti-roll bar force distribution
- Aerodynamic drag and downforce
- Weight transfer during acceleration/braking

**Evidence**:
```python
# Ackermann steering
def _compute_ackermann_angles(self, steer_input: float) -> Tuple[float, float]:
    if abs(steer_input) < 0.001:
        return 0.0, 0.0
    turn_radius = self.wheelbase / math.tan(steer_input * self.max_steer_angle)
    inner_angle = math.atan(self.wheelbase / (turn_radius - self.track_width / 2))
    outer_angle = math.atan(self.wheelbase / (turn_radius + self.track_width / 2))
    return (inner_angle, outer_angle) if steer_input > 0 else (outer_angle, inner_angle)
```

### aircraft.py (757 lines) - REAL

**Purpose**: Fixed-wing aircraft aerodynamics

**Key Algorithms**:
- Lift coefficient with stall modeling (tanh transition)
- Control surface moment generation
- Stability derivatives for flight dynamics
- Ground effect near surface

**Evidence**:
```python
# Lift with stall
def _compute_lift_coefficient(self, aoa: float) -> float:
    aoa_deg = math.degrees(aoa)
    if abs(aoa_deg) < self.stall_angle:
        return self.cl_alpha * aoa
    else:
        # Post-stall with tanh transition
        stall_factor = math.tanh((abs(aoa_deg) - self.stall_angle) / 5.0)
        cl_stall = self.cl_max * math.copysign(1, aoa)
        cl_post = 0.8 * math.sin(2 * aoa) * math.copysign(1, aoa)
        return cl_stall * (1 - stall_factor) + cl_post * stall_factor
```

### vehicle_system.py (685 lines) - REAL

**Purpose**: Vehicle manager and factory

**Key Algorithms**:
- Vehicle type registration system
- Vehicle group management
- Sleep/wake state with activity threshold
- LOD-based simulation fidelity

**Evidence**:
```python
# Sleep detection
def _check_sleep_eligibility(self, vehicle: Vehicle) -> bool:
    if vehicle.linear_velocity.length() > self.sleep_velocity_threshold:
        return False
    if vehicle.angular_velocity.length() > self.sleep_angular_threshold:
        return False
    vehicle.sleep_timer += self.timestep
    return vehicle.sleep_timer > self.sleep_time_threshold
```

### watercraft.py (664 lines) - REAL

**Purpose**: Boat and watercraft physics

**Key Algorithms**:
- Distributed buoyancy with sample points
- Hull drag with Froude number scaling
- Wave response with frequency-based forces
- Propeller thrust with slip ratio

**Evidence**:
```python
# Distributed buoyancy
def _compute_buoyancy_forces(self):
    total_force = Vector3.ZERO
    total_torque = Vector3.ZERO
    for sample in self.buoyancy_samples:
        world_pos = self.transform.transform_point(sample.local_pos)
        depth = self.water_surface.get_height(world_pos.x, world_pos.z) - world_pos.y
        if depth > 0:
            force = Vector3.UP * depth * sample.volume * self.water_density * GRAVITY
            total_force += force
            total_torque += (world_pos - self.center_of_mass).cross(force)
    return total_force, total_torque
```

---

## Key Algorithms Inventory

| Category | Algorithm | Files |
|----------|-----------|-------|
| **Constraint Solving** | Jacobian-based, Baumgarte stabilization, Warm starting | joint_base, contact_constraint, joint_d6 |
| **Soft Body** | Position-Based Dynamics | cloth_component, soft_body_pbd |
| **Soft Body** | Finite Element Method (Neo-Hookean, Corotational) | fem_solver |
| **Soft Body** | Shape Matching (SVD polar decomposition) | shape_matching |
| **Biomechanics** | Hill-type muscle model | muscle |
| **Vehicle** | Pacejka Magic Formula | tire_model |
| **Vehicle** | Torsen/LSD differential | drivetrain |
| **Vehicle** | Ackermann steering | wheeled_vehicle, joint_hinge |
| **Aerodynamics** | Lift/drag with stall | aircraft |
| **Fluid** | Archimedes buoyancy | fluid_component, watercraft |
| **Destruction** | Voronoi fracture | destruction_component |
| **Collision** | Inertia tensors, support mapping | collider_components |

---

## Evidence Classification Criteria

A file is classified as **REAL** when it exhibits:

1. **Correct Mathematical Models**: Formulas match published references (e.g., Pacejka 2012, Hill 1938, Baumgarte 1972)
2. **Physical Quantities**: Uses proper units (N, m, kg, rad/s) and dimensional consistency
3. **Non-trivial Logic**: Contains loops, conditionals, numerical methods rather than pass/NotImplementedError
4. **Algorithm Completeness**: Full implementations with edge cases handled
5. **Integration Points**: Methods for stepping simulation, applying forces, querying state

All 23 files met these criteria. No stub implementations were found.

---

## Conclusion

The `engine/simulation` subdirectories contain production-quality physics implementations. The codebase demonstrates:

- Deep understanding of constraint-based physics (Jacobians, effective mass, warm starting)
- Multiple material models for soft body simulation (PBD, FEM, shape matching)
- Industry-standard vehicle physics (Pacejka tires, differential types)
- Proper aerodynamics with stall and control surfaces
- Realistic fluid interaction with distributed buoyancy

This is a substantial physics engine foundation, not a prototype or stub collection.
