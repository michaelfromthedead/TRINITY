# PHASE 4 ARCHITECTURE: Vehicle Dynamics

**Scope**: engine/simulation/vehicles (6 files, ~4,681 lines)  
**Focus**: Drivetrain, tires, wheeled vehicles, aircraft, watercraft

---

## Subsystem Overview

The vehicles subsystem provides complete dynamics simulation for multiple vehicle types:

| File | Lines | Domain | Key Algorithm |
|------|-------|--------|---------------|
| drivetrain.py | 982 | Powertrain | Torque curves, differentials |
| tire_model.py | 831 | Tire physics | Pacejka Magic Formula |
| wheeled_vehicle.py | 762 | Ground vehicles | Ackermann, anti-roll, aero |
| aircraft.py | 757 | Fixed-wing flight | Lift/drag, stall, stability |
| vehicle_system.py | 685 | Management | Sleep/wake, LOD, factory |
| watercraft.py | 664 | Boats | Buoyancy, hull drag, waves |

---

## Architecture Decisions

### ADR-VEH-001: Pacejka Magic Formula for Tires

**Context**: Realistic tire behavior is critical for vehicle handling.

**Decision**: Implement full Pacejka Magic Formula with combined slip.

**Rationale**:
- Industry standard for vehicle simulation
- Coefficients fitted to real tire test data
- Handles combined braking and cornering
- Load sensitivity correctly modeled

**Formula**:
```
F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))

where:
  B = stiffness factor
  C = shape factor
  D = peak value (load-scaled)
  E = curvature factor
  x = slip (longitudinal or lateral)
```

### ADR-VEH-002: Multiple Differential Types

**Context**: Different driving feel requires different torque distribution.

**Decision**: Implement Open, Limited Slip, Torsen, and Locked differentials.

**Rationale**:

| Type | Torque Split | Feel |
|------|--------------|------|
| Open | Equal torque | Neutral, understeer on accel |
| LSD | Friction-biased | Tighter cornering |
| Torsen | Gear-biased | Progressive, predictable |
| Locked | 50/50 always | Maximum traction, tight feel |

### ADR-VEH-003: Ackermann Steering Geometry

**Context**: All four wheels must track correctly through turns.

**Decision**: Implement Ackermann steering angle calculation.

**Rationale**:
- Inner wheel turns more than outer
- Both wheels follow correct turn radius
- Eliminates tire scrub in turns
- Industry standard for wheeled vehicles

**Formula**:
```
turn_radius = wheelbase / tan(steer_angle)
inner_angle = atan(wheelbase / (turn_radius - track_width/2))
outer_angle = atan(wheelbase / (turn_radius + track_width/2))
```

### ADR-VEH-004: Stall Modeling with tanh Transition

**Context**: Aircraft stall must be smooth for playability but realistic.

**Decision**: Use tanh transition between attached and stalled lift.

**Rationale**:
- Smooth transition (no discontinuity)
- Tunable transition sharpness
- Post-stall lift follows sin(2*alpha)
- Predictable recovery behavior

**Implementation**:
```python
if abs(aoa_deg) < stall_angle:
    cl = cl_alpha * aoa
else:
    stall_factor = tanh((abs(aoa_deg) - stall_angle) / 5.0)
    cl = cl_stall * (1 - stall_factor) + cl_post * stall_factor
```

### ADR-VEH-005: Distributed Buoyancy for Watercraft

**Context**: Boats need realistic floating and wave response.

**Decision**: Use multiple sample points for buoyancy calculation.

**Rationale**:
- Partial submersion handled correctly
- Roll/pitch response from off-center submersion
- Wave forces from per-sample height differences
- Configurable sample density

### ADR-VEH-006: Sleep/Wake with Activity Thresholds

**Context**: Parked vehicles waste CPU if simulated.

**Decision**: Implement sleep detection with velocity and time thresholds.

**Rationale**:
- Stationary vehicles sleep after timeout
- Any significant velocity wakes vehicle
- User input immediately wakes
- Reduces CPU for parked vehicle populations

---

## Data Flow

```
Vehicle Input (steering, throttle, brake)
     |
     v
+----------------------+
| Vehicle System       | (manager, factory, LOD)
+----------------------+
     |
     +---> WheeledVehicle
     |          |
     |          +--> Ackermann Steering
     |          +--> Wheel Raycasting (suspension)
     |          +--> TireModel (per wheel)
     |          |        |
     |          |        +--> Pacejka lateral force
     |          |        +--> Pacejka longitudinal force
     |          |        +--> Combined slip weighting
     |          |
     |          +--> Drivetrain
     |          |        |
     |          |        +--> Engine torque curves
     |          |        +--> Clutch slip
     |          |        +--> Transmission (gear ratios)
     |          |        +--> Differential (type-specific split)
     |          |
     |          +--> Anti-roll bar forces
     |          +--> Weight transfer
     |          +--> Aerodynamic drag/downforce
     |
     +---> Aircraft
     |          |
     |          +--> Angle of attack calculation
     |          +--> Lift coefficient (with stall)
     |          +--> Drag coefficient
     |          +--> Control surface moments
     |          +--> Stability derivatives
     |          +--> Ground effect
     |
     +---> Watercraft
              |
              +--> Distributed buoyancy samples
              +--> Hull drag (Froude scaling)
              +--> Wave response forces
              +--> Propeller thrust
```

---

## Interface Contracts

### TireModel

```python
class TireModel:
    def compute_forces(self, slip_angle: float, slip_ratio: float, 
                       vertical_load: float) -> Tuple[float, float]:
        """Return (lateral_force, longitudinal_force)."""
    
    def set_pacejka_coefficients(self, B: float, C: float, D: float, E: float) -> None:
        """Set Magic Formula coefficients."""
    
    def update_temperature(self, dt: float, slip_power: float) -> None:
        """Update tire temperature based on slip energy dissipation."""
```

### Drivetrain

```python
class Drivetrain:
    def compute_wheel_torques(self, throttle: float, brake: float, 
                               wheel_speeds: List[float]) -> List[float]:
        """Return torque for each wheel."""
    
    def get_engine_rpm(self) -> float:
        """Return current engine RPM."""
    
    def shift_gear(self, gear: int) -> None:
        """Shift to specified gear (auto or manual)."""
    
    def set_differential_type(self, diff_type: DifferentialType) -> None:
        """Set differential behavior."""
```

### WheeledVehicle

```python
class WheeledVehicle:
    def step(self, dt: float, input: VehicleInput) -> None:
        """Advance vehicle simulation."""
    
    def get_wheel_states(self) -> List[WheelState]:
        """Return wheel positions, rotations, contact info."""
    
    def get_speed(self) -> float:
        """Return forward speed in m/s."""
    
    def apply_damage(self, damage: VehicleDamage) -> None:
        """Apply damage to vehicle components."""
```

### Aircraft

```python
class Aircraft:
    def step(self, dt: float, input: FlightInput) -> None:
        """Advance flight simulation."""
    
    def get_airspeed(self) -> float:
        """Return indicated airspeed."""
    
    def get_aoa(self) -> float:
        """Return current angle of attack in radians."""
    
    def is_stalled(self) -> bool:
        """Return True if wing is stalled."""
```

### Watercraft

```python
class Watercraft:
    def step(self, dt: float, input: BoatInput) -> None:
        """Advance watercraft simulation."""
    
    def get_hull_speed(self) -> float:
        """Return forward speed through water."""
    
    def get_submersion_ratio(self) -> float:
        """Return fraction of hull volume submerged."""
```

---

## Tire Model Details

### Pacejka Coefficients

| Coefficient | Meaning | Typical Range |
|-------------|---------|---------------|
| B | Stiffness factor | 8-15 |
| C | Shape factor | 1.0-2.0 |
| D | Peak factor | ~1.0 (load-normalized) |
| E | Curvature factor | -2 to +1 |

### Combined Slip

```python
# Combined slip reduces peak force
slip_total = sqrt(slip_angle^2 + slip_ratio^2)
F_lat = F_lat_pure * slip_angle / slip_total
F_long = F_long_pure * slip_ratio / slip_total
```

### Load Sensitivity

```python
D_scaled = D_nominal * (load / nominal_load)^load_sensitivity
```

---

## Differential Models

### Open Differential

```python
# Equal torque, independent speed
torque_left = input_torque / 2
torque_right = input_torque / 2
```

### Limited Slip (Clutch-Type)

```python
# Friction bias based on speed difference
speed_diff = abs(left_speed - right_speed)
bias = min(preload + speed_diff * ramp_factor, max_bias)
# Apply bias toward slower wheel
```

### Torsen

```python
# Gear-based bias with configurable ratio
bias_ratio = min(torsen_bias_ratio, 1 + speed_diff * sensitivity)
split_fast = 1 / (1 + bias_ratio)
split_slow = 1 - split_fast
```

### Locked

```python
# Rigid connection
torque_left = input_torque / 2
torque_right = input_torque / 2
# Wheels forced to same speed (constraint)
```

---

## Dependencies

### Internal
- engine/simulation/components: VehicleComponent base
- engine/simulation/constraints: JointHinge for steering, suspension
- engine/collision: Wheel raycasting

### External
- NumPy: Vector/matrix math

---

## Performance Considerations

### Tire Model
- Magic Formula evaluates in < 1us per wheel
- Combined slip adds ~50% overhead
- Temperature/wear optional for performance

### Raycasting
- 4-8 rays per vehicle (wheels + suspension)
- Batch raycasting for vehicle populations

### Sleep Detection
- Check velocity every N frames
- Sleep timer accumulates only when below threshold
- Immediate wake on input or collision
