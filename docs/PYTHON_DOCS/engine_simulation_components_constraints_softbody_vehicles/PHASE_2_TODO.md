# PHASE 2 TODO: Constraint Solvers

**Scope**: engine/simulation/constraints (6 files, ~3,311 lines)  
**Priority**: High — Core physics infrastructure

---

## joint_base.py (546 lines)

### T-CON-B1.1: Validate Effective Mass Computation
- [ ] Test symmetric positive definite matrix result
- [ ] Verify formula: `M_eff = inv(J * M_inv * J^T)`
- [ ] Test with unit mass and identity inertia
- [ ] Acceptance: Effective mass correct for simple cases

### T-CON-B1.2: Test Warm Starting Mechanism
- [ ] Verify cached impulses applied at frame start
- [ ] Test iteration count reduction (30%+ expected)
- [ ] Test cache persistence across frames
- [ ] Acceptance: Warm starting measurably reduces iterations

### T-CON-B1.3: Validate Break Detection
- [ ] Test force threshold detection
- [ ] Test torque threshold detection
- [ ] Verify joint removal on break
- [ ] Acceptance: Joints break at configured thresholds

### T-CON-B1.4: Test Jacobian Computation
- [ ] Verify linear velocity Jacobian rows
- [ ] Verify angular velocity Jacobian rows
- [ ] Test body A and body B contributions
- [ ] Acceptance: Jacobian maps velocities correctly

---

## joint_d6.py (657 lines)

### T-CON-D6.1: Test Per-Axis Motion Configuration
- [ ] Test LOCKED mode: zero motion allowed
- [ ] Test LIMITED mode: motion within bounds
- [ ] Test FREE mode: unrestricted motion
- [ ] Acceptance: Each mode behaves correctly per axis

### T-CON-D6.2: Validate Positional Motors
- [ ] Test target position tracking
- [ ] Test drive stiffness and damping
- [ ] Test motor force limits
- [ ] Acceptance: Motor drives to target with configured response

### T-CON-D6.3: Test Angular Motors
- [ ] Test target orientation tracking
- [ ] Test angular drive stiffness
- [ ] Test angular velocity limits
- [ ] Acceptance: Angular motor drives correctly

### T-CON-D6.4: Validate Limit Enforcement
- [ ] Test soft contact at limits
- [ ] Test restitution at limit contacts
- [ ] Test limit range configuration
- [ ] Acceptance: Bodies stop at configured limits

---

## joint_hinge.py (486 lines)

### T-CON-H1.1: Test Single Axis Rotation Constraint
- [ ] Verify 5-DOF locked (2 translations + 3 rotations - 1)
- [ ] Test rotation only around hinge axis
- [ ] Test with arbitrary hinge axis orientation
- [ ] Acceptance: Only hinge axis rotation permitted

### T-CON-H1.2: Validate Angular Limit Enforcement
- [ ] Test lower and upper angle limits
- [ ] Test restitution at limits
- [ ] Test soft limit behavior
- [ ] Acceptance: Rotation constrained to limit range

### T-CON-H1.3: Test Hinge Angle Extraction
- [ ] Verify angle extraction from relative quaternion
- [ ] Test full 360-degree range
- [ ] Test angle continuity (no wrap-around jumps)
- [ ] Acceptance: Angle matches expected geometry

### T-CON-H1.4: Test Motor with Velocity Target
- [ ] Test velocity mode motor
- [ ] Test motor torque limits
- [ ] Test acceleration smoothing
- [ ] Acceptance: Motor achieves target velocity

---

## joint_slider.py (518 lines)

### T-CON-S1.1: Test Single Axis Translation Constraint
- [ ] Verify 5-DOF locked (2 translations + 3 rotations)
- [ ] Test translation only along slide axis
- [ ] Test with arbitrary axis orientation
- [ ] Acceptance: Only slide axis translation permitted

### T-CON-S1.2: Validate Linear Limit Enforcement
- [ ] Test lower and upper position limits
- [ ] Test restitution at limits
- [ ] Test limit force magnitude
- [ ] Acceptance: Translation constrained to limit range

### T-CON-S1.3: Test Perpendicular Axis Constraints
- [ ] Verify perpendicular translation locked
- [ ] Test all three rotation axes locked
- [ ] Test constraint stability
- [ ] Acceptance: Perpendicular motion prevented

### T-CON-S1.4: Test Motor Force with Clamping
- [ ] Test force mode motor
- [ ] Test max force clamping
- [ ] Test bidirectional motor force
- [ ] Acceptance: Motor applies clamped force

---

## joint_spring.py (499 lines)

### T-CON-SP1.1: Validate CFM/ERP Coefficients
- [ ] Derive gamma from formula: `gamma = 1/(h*(c + h*k))`
- [ ] Derive beta from formula: `beta = h*k*gamma`
- [ ] Test coefficient values for known k, c
- [ ] Acceptance: Coefficients match derived formulas

### T-CON-SP1.2: Test Soft Constraint Behavior
- [ ] Verify spring-like oscillation
- [ ] Test damping effect on oscillation decay
- [ ] Compare to analytical spring-damper solution
- [ ] Acceptance: Motion matches expected spring-damper behavior

### T-CON-SP1.3: Validate Rest Length with Hysteresis
- [ ] Test rest length equilibrium
- [ ] Test hysteresis prevents oscillation
- [ ] Test rest length change handling
- [ ] Acceptance: Stable equilibrium at rest length

### T-CON-SP1.4: Test Effective Mass with CFM
- [ ] Verify formula: `M_eff = 1/(J*M_inv*J^T + gamma)`
- [ ] Test CFM contribution to effective mass
- [ ] Test numerical stability with high stiffness
- [ ] Acceptance: Effective mass includes CFM correctly

---

## contact_constraint.py (605 lines)

### T-CON-C1.1: Validate Normal Constraint with Baumgarte
- [ ] Test penetration correction via bias term
- [ ] Test beta parameter effect (0.1-0.3 range)
- [ ] Test separating contacts (no correction)
- [ ] Acceptance: Penetration reduced over iterations

### T-CON-C1.2: Test Tangent Basis Construction
- [ ] Verify Duff method avoids singularity
- [ ] Test tangents perpendicular to normal
- [ ] Test tangents perpendicular to each other
- [ ] Acceptance: Orthonormal basis constructed correctly

### T-CON-C1.3: Validate Coulomb Friction Cone
- [ ] Test friction force clamping to cone
- [ ] Test friction coefficient application
- [ ] Test static vs. kinetic friction
- [ ] Acceptance: Friction bounded by mu * normal_force

### T-CON-C1.4: Test Contact Point Persistence
- [ ] Test ID matching across frames
- [ ] Test impulse cache lookup by ID
- [ ] Test new contact initialization
- [ ] Acceptance: Persistent contacts reuse cached impulses

---

## Integration Tasks

### T-CON-INT1: Solver Pipeline Integration
- [ ] Test prepare -> warm_start -> solve_velocity -> cache sequence
- [ ] Verify constraint order independence (Gauss-Seidel)
- [ ] Test mixed constraint types in single solve
- [ ] Acceptance: All constraints solve correctly together

### T-CON-INT2: Collision System Integration
- [ ] Verify contact manifold consumption
- [ ] Test contact creation and destruction
- [ ] Test manifold update each frame
- [ ] Acceptance: Contacts flow from collision to solver

### T-CON-INT3: Performance Benchmark
- [ ] Benchmark solve time for 100 joints
- [ ] Benchmark solve time for 500 contacts
- [ ] Profile hot spots
- [ ] Acceptance: <1ms for 100 joints + 500 contacts

---

## Completion Criteria

Phase 2 is complete when:
1. All T-CON-*.* tasks pass acceptance criteria
2. All 6 constraint files have >80% test coverage
3. Warm starting reduces iterations by 30%+
4. Constraint error < 1e-6 after convergence
5. Performance: <1ms for typical constraint workload
