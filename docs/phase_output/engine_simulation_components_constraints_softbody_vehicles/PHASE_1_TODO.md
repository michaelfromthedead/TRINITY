# PHASE 1 TODO: Simulation Components

**Scope**: engine/simulation/components (6 files, ~3,406 lines)  
**Priority**: High — Foundation for entity-based simulation

---

## cloth_component.py (607 lines)

### T-SIM-C1.1: Validate PBD Distance Constraint Solver
- [ ] Test that constraint projection converges within 10 iterations
- [ ] Verify inverse mass weighting distributes correction correctly
- [ ] Test with pinned vertices (w=0) — no displacement expected
- [ ] Acceptance: Constraint error < 1e-4 after 10 iterations

### T-SIM-C1.2: Test Wind Force Application
- [ ] Verify wind force scales with surface normal alignment
- [ ] Test pressure and friction components independently
- [ ] Confirm no force on backfacing triangles
- [ ] Acceptance: Force direction matches expected physics

### T-SIM-C1.3: Validate Tearing Mechanism
- [ ] Test strain threshold detection
- [ ] Verify constraint removal updates mesh topology
- [ ] Test tear propagation along high-strain edges
- [ ] Acceptance: Tears initiate at correct strain threshold

### T-SIM-C1.4: Test Self-Collision with Spatial Hashing
- [ ] Verify spatial hash cell size appropriate for vertex spacing
- [ ] Test collision detection catches close vertices
- [ ] Confirm collision response prevents interpenetration
- [ ] Acceptance: No cloth self-intersection after 1000 frames

---

## collider_components.py (587 lines)

### T-SIM-C2.1: Validate Sphere Inertia Tensor
- [ ] Test: `I = 2/5 * m * r^2 * Identity` for solid sphere
- [ ] Verify diagonal matrix with equal elements
- [ ] Acceptance: Match analytical formula within 1e-10

### T-SIM-C2.2: Validate Box Inertia Tensor
- [ ] Test: `I_xx = m/12 * (h_y^2 + h_z^2)` etc.
- [ ] Verify non-uniform dimensions produce non-uniform tensor
- [ ] Acceptance: Match analytical formula within 1e-10

### T-SIM-C2.3: Validate Capsule Inertia Tensor
- [ ] Test cylinder contribution
- [ ] Test hemisphere contributions
- [ ] Verify composite tensor combines correctly
- [ ] Acceptance: Match analytical formula within 1e-10

### T-SIM-C2.4: Test Support Mapping for GJK
- [ ] Verify support returns furthest point in given direction
- [ ] Test all primitive types: sphere, box, capsule
- [ ] Test mesh collider convex hull support
- [ ] Acceptance: Support point lies on shape boundary in direction

---

## fluid_component.py (581 lines)

### T-SIM-C3.1: Validate Archimedes Buoyancy Calculation
- [ ] Test fully submerged object: F = rho * V * g
- [ ] Test partially submerged: force proportional to submerged volume
- [ ] Test object above water: zero buoyancy force
- [ ] Acceptance: Force magnitude matches expected within 1%

### T-SIM-C3.2: Test Quadratic Drag Law
- [ ] Verify drag scales with velocity squared
- [ ] Test drag coefficient application
- [ ] Test cross-sectional area calculation
- [ ] Acceptance: F_drag = 0.5 * Cd * rho * A * v^2

### T-SIM-C3.3: Validate Wave Height Function
- [ ] Test single frequency component
- [ ] Test multiple frequency superposition
- [ ] Verify time-varying behavior
- [ ] Acceptance: Wave heights match expected sinusoidal pattern

### T-SIM-C3.4: Test Distributed Sample Point Buoyancy
- [ ] Verify torque generated for off-center submersion
- [ ] Test asymmetric object produces rotation
- [ ] Acceptance: Object naturally finds stable floating orientation

---

## destruction_component.py (559 lines)

### T-SIM-C4.1: Test Damage Accumulation
- [ ] Verify damage adds from multiple impacts
- [ ] Test damage threshold detection
- [ ] Test damage localization at impact points
- [ ] Acceptance: Destruction triggers at correct threshold

### T-SIM-C4.2: Validate Voronoi Seed Distribution
- [ ] Test seed placement relative to impact point
- [ ] Verify seed count matches fragment_count parameter
- [ ] Test radial falloff from impact
- [ ] Acceptance: Seeds distributed with expected pattern

### T-SIM-C4.3: Test Fragment Generation
- [ ] Verify fragments cover original mesh volume
- [ ] Test fragment mesh validity (closed, no holes)
- [ ] Verify fragment mass sums to original mass
- [ ] Acceptance: All fragments are valid meshes

### T-SIM-C4.4: Test Connectivity Graph
- [ ] Verify structural integrity detection
- [ ] Test isolated fragments fall away
- [ ] Test cascading destruction
- [ ] Acceptance: Graph correctly identifies connected regions

---

## character_component.py (542 lines)

### T-SIM-C5.1: Test Ground Detection
- [ ] Verify sphere sweep detects ground contact
- [ ] Test ground normal extraction
- [ ] Test probe distance parameter
- [ ] Acceptance: Grounded state correct for all test cases

### T-SIM-C5.2: Validate Slope Handling
- [ ] Test movement scaling on slopes
- [ ] Test slope limit angle (grounded vs. sliding)
- [ ] Test perpendicular-to-slope movement
- [ ] Acceptance: Movement scales correctly with slope angle

### T-SIM-C5.3: Test Step Climbing
- [ ] Verify step detection within height threshold
- [ ] Test step climb animation/smoothing
- [ ] Test step too high rejection
- [ ] Acceptance: Steps below threshold climbed smoothly

### T-SIM-C5.4: Test Movement State Machine
- [ ] Verify transitions: grounded -> falling -> grounded
- [ ] Test sliding state on steep slopes
- [ ] Test state persistence (no oscillation)
- [ ] Acceptance: State machine transitions correctly

---

## vehicle_component.py (530 lines)

### T-SIM-C6.1: Test Wheel Raycasting Suspension
- [ ] Verify raycast detects ground below wheel
- [ ] Test suspension compression calculation
- [ ] Test wheel at full extension vs. full compression
- [ ] Acceptance: Suspension force proportional to compression

### T-SIM-C6.2: Validate Spring-Damper Force
- [ ] Test spring force: F = k * compression
- [ ] Test damper force: F = -c * velocity
- [ ] Test combined spring-damper response
- [ ] Acceptance: Force matches spring-damper formula

### T-SIM-C6.3: Test Engine Torque Curves
- [ ] Verify torque interpolation across RPM range
- [ ] Test idle and redline boundaries
- [ ] Test throttle scaling
- [ ] Acceptance: Torque matches defined curve within 1%

### T-SIM-C6.4: Validate Ackermann Steering Geometry
- [ ] Test inner wheel turns more than outer
- [ ] Verify zero steer input produces parallel wheels
- [ ] Test steering angle limits
- [ ] Acceptance: Geometry matches Ackermann formula

---

## Integration Tasks

### T-SIM-C-INT1: Component Attachment to Entities
- [ ] Verify component registration with entity system
- [ ] Test component initialization on attach
- [ ] Test component cleanup on detach
- [ ] Acceptance: Components integrate with entity lifecycle

### T-SIM-C-INT2: Simulation Step Integration
- [ ] Verify all components stepped each frame
- [ ] Test step order (dependencies respected)
- [ ] Test variable timestep handling
- [ ] Acceptance: All components advance correctly per frame

---

## Completion Criteria

Phase 1 is complete when:
1. All T-SIM-C*.* tasks pass acceptance criteria
2. All 6 component files have >80% test coverage
3. No regression in existing engine integration tests
4. Performance benchmark: <0.5ms total for 100 components
