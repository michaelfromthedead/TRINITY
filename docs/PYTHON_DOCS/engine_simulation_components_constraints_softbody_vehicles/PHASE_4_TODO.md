# PHASE 4 TODO: Vehicle Dynamics

**Scope**: engine/simulation/vehicles (6 files, ~4,681 lines)  
**Priority**: Medium — Complete vehicle simulation stack

---

## drivetrain.py (982 lines)

### T-VEH-DT1.1: Validate Engine Torque Curves
- [ ] Test polynomial interpolation across RPM range
- [ ] Verify peak torque at specified RPM
- [ ] Test idle and redline boundaries
- [ ] Test throttle scaling
- [ ] Acceptance: Torque matches defined curve within 1%

### T-VEH-DT1.2: Test Clutch Slip Model
- [ ] Test engagement threshold
- [ ] Test slip torque transmission
- [ ] Test full engagement behavior
- [ ] Test launch control scenario
- [ ] Acceptance: Smooth torque transfer through clutch

### T-VEH-DT1.3: Validate Automatic Transmission
- [ ] Test upshift points on shift map
- [ ] Test downshift points (kickdown)
- [ ] Test gear ratio application
- [ ] Test shift time and torque interruption
- [ ] Acceptance: Shift points match defined map

### T-VEH-DT1.4: Test Open Differential
- [ ] Verify equal torque split
- [ ] Test independent wheel speeds
- [ ] Test zero output with lifted wheel
- [ ] Acceptance: 50/50 torque always

### T-VEH-DT1.5: Test Limited Slip Differential
- [ ] Test preload torque
- [ ] Test bias under speed difference
- [ ] Test ramp factor effect
- [ ] Acceptance: Torque bias increases with speed difference

### T-VEH-DT1.6: Test Torsen Differential
- [ ] Verify bias ratio application
- [ ] Test sensitivity parameter
- [ ] Test torque split at various speed differences
- [ ] Acceptance: Split matches Torsen formula

### T-VEH-DT1.7: Test Locked Differential
- [ ] Verify 50/50 torque split
- [ ] Test wheel speed constraint
- [ ] Test tire scrub in turns
- [ ] Acceptance: Wheels spin at equal speed

---

## tire_model.py (831 lines)

### T-VEH-TM1.1: Validate Pacejka Magic Formula
- [ ] Test formula: `F = D * sin(C * atan(Bx - E*(Bx - atan(Bx))))`
- [ ] Plot force vs. slip curve
- [ ] Verify peak location and magnitude
- [ ] Acceptance: Curve matches expected Pacejka shape

### T-VEH-TM1.2: Test Lateral Force (Slip Angle)
- [ ] Test small slip angle (linear region)
- [ ] Test peak slip angle
- [ ] Test post-peak saturation
- [ ] Acceptance: Lateral force follows Pacejka curve

### T-VEH-TM1.3: Test Longitudinal Force (Slip Ratio)
- [ ] Test small slip ratio (linear region)
- [ ] Test peak slip ratio
- [ ] Test post-peak saturation
- [ ] Acceptance: Longitudinal force follows Pacejka curve

### T-VEH-TM1.4: Validate Combined Slip
- [ ] Test friction circle concept
- [ ] Test combined lateral + longitudinal
- [ ] Verify force reduction at combined slip
- [ ] Acceptance: Combined forces follow friction ellipse

### T-VEH-TM1.5: Test Load Sensitivity
- [ ] Verify D scales with vertical load
- [ ] Test load sensitivity exponent
- [ ] Test at multiple load levels
- [ ] Acceptance: Force scales correctly with load

### T-VEH-TM1.6: Test Temperature Model
- [ ] Test temperature rise with slip energy
- [ ] Test cooling rate
- [ ] Test grip variation with temperature
- [ ] Acceptance: Temperature dynamics plausible

### T-VEH-TM1.7: Test Wear Model
- [ ] Test wear accumulation
- [ ] Test grip degradation with wear
- [ ] Acceptance: Wear affects grip progressively

---

## wheeled_vehicle.py (762 lines)

### T-VEH-WV1.1: Validate Ackermann Steering
- [ ] Test inner wheel angle > outer wheel angle
- [ ] Test turn radius calculation
- [ ] Test zero steer input (parallel wheels)
- [ ] Test steering limits
- [ ] Acceptance: Geometry matches Ackermann formula

### T-VEH-WV1.2: Test Anti-Roll Bar Forces
- [ ] Test force distribution in roll
- [ ] Test front vs. rear anti-roll balance
- [ ] Test straight-line behavior (no roll force)
- [ ] Acceptance: Roll stiffness matches anti-roll bar rate

### T-VEH-WV1.3: Validate Aerodynamic Forces
- [ ] Test drag force: `F = 0.5 * Cd * A * rho * v^2`
- [ ] Test downforce: `F = 0.5 * Cl * A * rho * v^2`
- [ ] Test lift-to-drag ratio
- [ ] Acceptance: Aero forces match expected formulas

### T-VEH-WV1.4: Test Weight Transfer
- [ ] Test forward weight transfer on braking
- [ ] Test rear weight transfer on acceleration
- [ ] Test lateral weight transfer in turns
- [ ] Acceptance: Weight transfer follows vehicle dynamics

### T-VEH-WV1.5: Test Suspension Raycast
- [ ] Verify raycast detects ground
- [ ] Test suspension compression calculation
- [ ] Test spring/damper force application
- [ ] Acceptance: Suspension responds to terrain

### T-VEH-WV1.6: Benchmark Vehicle Performance
- [ ] Time full vehicle step for 1, 10, 100 vehicles
- [ ] Profile subsystem contributions
- [ ] Acceptance: <0.5ms per vehicle

---

## aircraft.py (757 lines)

### T-VEH-AC1.1: Validate Lift Coefficient Pre-Stall
- [ ] Test cl = cl_alpha * aoa relationship
- [ ] Verify linear range
- [ ] Test cl at zero aoa
- [ ] Acceptance: Lift matches pre-stall theory

### T-VEH-AC1.2: Test Stall Transition
- [ ] Verify tanh transition smoothness
- [ ] Test stall angle threshold
- [ ] Test transition width parameter
- [ ] Acceptance: Smooth, continuous stall transition

### T-VEH-AC1.3: Validate Post-Stall Behavior
- [ ] Test cl = 0.8 * sin(2*aoa) post-stall
- [ ] Verify stall recovery
- [ ] Test deep stall characteristics
- [ ] Acceptance: Post-stall lift follows expected curve

### T-VEH-AC1.4: Test Control Surface Moments
- [ ] Test elevator pitch moment
- [ ] Test aileron roll moment
- [ ] Test rudder yaw moment
- [ ] Test deflection limits
- [ ] Acceptance: Control moments proportional to deflection

### T-VEH-AC1.5: Validate Stability Derivatives
- [ ] Test pitch damping (Cmq)
- [ ] Test roll damping (Clp)
- [ ] Test yaw damping (Cnr)
- [ ] Acceptance: Stability derivatives produce damped response

### T-VEH-AC1.6: Test Ground Effect
- [ ] Verify lift increase near ground
- [ ] Test altitude threshold for ground effect
- [ ] Test ground effect fade
- [ ] Acceptance: Ground effect matches expected altitude curve

---

## vehicle_system.py (685 lines)

### T-VEH-VS1.1: Test Vehicle Registration
- [ ] Test vehicle type registration
- [ ] Test vehicle spawning from type
- [ ] Test type lookup
- [ ] Acceptance: Registration and spawning work correctly

### T-VEH-VS1.2: Validate Sleep Detection
- [ ] Test velocity threshold detection
- [ ] Test angular velocity threshold
- [ ] Test sleep timer accumulation
- [ ] Test wake on input
- [ ] Acceptance: Vehicles sleep/wake correctly

### T-VEH-VS1.3: Test LOD-Based Simulation
- [ ] Test LOD level assignment
- [ ] Test simulation fidelity per LOD
- [ ] Test LOD transition smoothness
- [ ] Acceptance: LOD reduces CPU for distant vehicles

### T-VEH-VS1.4: Test Vehicle Group Management
- [ ] Test group creation
- [ ] Test vehicle assignment to groups
- [ ] Test group-level operations
- [ ] Acceptance: Group operations work correctly

---

## watercraft.py (664 lines)

### T-VEH-WC1.1: Validate Distributed Buoyancy
- [ ] Test per-sample buoyancy calculation
- [ ] Test total force summation
- [ ] Test torque from off-center samples
- [ ] Acceptance: Buoyancy force matches Archimedes

### T-VEH-WC1.2: Test Hull Drag with Froude Scaling
- [ ] Test low-speed (displacement) drag
- [ ] Test high-speed (planing) drag
- [ ] Test Froude number calculation
- [ ] Test transition between regimes
- [ ] Acceptance: Drag follows Froude scaling

### T-VEH-WC1.3: Validate Wave Response
- [ ] Test wave height evaluation per sample
- [ ] Test pitch/roll from wave forces
- [ ] Test multiple wave frequency interaction
- [ ] Acceptance: Realistic wave response

### T-VEH-WC1.4: Test Propeller Thrust
- [ ] Test thrust vs. throttle relationship
- [ ] Test slip ratio effect
- [ ] Test reverse thrust
- [ ] Acceptance: Propeller produces expected thrust

---

## Integration Tasks

### T-VEH-INT1: Vehicle-Terrain Integration
- [ ] Test vehicle on various terrain types
- [ ] Test tire grip variation with surface
- [ ] Test suspension on rough terrain
- [ ] Acceptance: Vehicles respond realistically to terrain

### T-VEH-INT2: Vehicle-Vehicle Collision
- [ ] Test collision detection between vehicles
- [ ] Test damage application
- [ ] Test post-collision dynamics
- [ ] Acceptance: Collisions handled correctly

### T-VEH-INT3: Vehicle Input System Integration
- [ ] Test input mapping to vehicle controls
- [ ] Test input smoothing/filtering
- [ ] Test multiple input devices
- [ ] Acceptance: Inputs drive vehicles correctly

---

## Completion Criteria

Phase 4 is complete when:
1. All T-VEH-*.* tasks pass acceptance criteria
2. All 6 vehicle files have >80% test coverage
3. Pacejka forces match published tire data within 5%
4. Differential types produce correct torque splits
5. Aircraft stall exhibits smooth tanh transition
6. Watercraft floats at correct waterline
7. Performance: <0.5ms per vehicle simulation step
