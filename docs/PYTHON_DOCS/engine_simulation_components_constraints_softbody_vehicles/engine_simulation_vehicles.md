# Investigation: engine/simulation/vehicles

## Summary
The vehicles module is a comprehensive, production-quality vehicle physics simulation totaling 7,004 lines across 11 Python files. It implements multiple tire models (Pacejka Magic Formula, Linear, Brush), realistic spring-damper suspension with anti-roll bars, full drivetrain simulation (engine, clutch, transmission, differentials), plus specialized vehicle types (tracked, hover, aircraft, watercraft).

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| drivetrain.py | 982 | REAL | Engine torque curves, clutch slip, transmission shifting, open/LSD/Torsen differentials |
| tire_model.py | 831 | REAL | Pacejka Magic Formula, Linear, Brush models with slip ratio/angle calculations |
| wheeled_vehicle.py | 762 | REAL | 4-wheel vehicle with Ackermann steering, suspension, aerodynamics, integration |
| aircraft.py | 757 | REAL | Aerodynamic surfaces, lift/drag, control surfaces, flight phases |
| vehicle_system.py | 685 | REAL | Base classes, Vector3, Transform, VehicleState, collision info |
| watercraft.py | 664 | REAL | Buoyancy, propellers, rudders, wave interaction |
| hover_vehicle.py | 581 | REAL | Air cushion physics, lift fans, thrust vectors, skirt state |
| suspension.py | 546 | REAL | Spring-damper, bump stops, progressive rate, anti-roll bars, geometry (camber/caster/toe) |
| tracked_vehicle.py | 514 | REAL | Track physics, road wheels, differential steering |
| config.py | 369 | REAL | Physics constants (Pacejka coefficients, gear ratios, presets) |
| __init__.py | 313 | REAL | Comprehensive exports for all components |

## Vehicle Components
- **Wheels**: Position, radius, mass, inertia, steering flag, driven flag
- **Suspension**: SuspensionType enum (SPRING_DAMPER, DOUBLE_WISHBONE, MACPHERSON, TRAILING_ARM, MULTI_LINK, SOLID_AXLE, TORSION_BEAM), spring/damper forces, bump stops, geometry with camber gain
- **Tires**: TireSurface (9 types), slip ratio/angle calculation, friction circle, rolling resistance, temperature/wear tracking
- **Drivetrain**: Engine with torque curve interpolation, Clutch with slip modeling, Transmission with shift timing, Differential (OPEN, LIMITED_SLIP, LOCKED, TORSEN, ELECTRONIC)
- **Anti-roll bars**: Torsional stiffness, asymmetric support
- **Aerodynamics**: Drag coefficient, frontal area, lift/downforce

## Implementation
- Real wheel physics? **YES** - Angular velocity integration, inertia, drive/brake torque balance, wheel spin-down
- Real suspension? **YES** - Spring force F=kx, asymmetric compression/rebound damping, progressive springs, bump stops with smooth engagement
- Real drivetrain? **YES** - Engine torque curves with RPM interpolation, rev limiter, idle governor, clutch slip calculation, gear ratio multiplication, multiple differential types with torque biasing algorithms

## Verdict
**REAL IMPLEMENTATION**

This is a complete, simulation-grade vehicle physics system suitable for realistic driving games or engineering simulations. All core physics are mathematically correct implementations, not approximations or stubs.

## Evidence

### Pacejka Magic Formula (tire_model.py:452-474)
```python
def _magic_formula(
    self,
    x: float,
    b: float,
    c: float,
    d: float,
    e: float,
) -> float:
    """
    Evaluate Magic Formula.
    F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
    """
    bx = b * x
    return d * math.sin(c * math.atan(bx - e * (bx - math.atan(bx))))
```

### Spring-Damper Suspension (suspension.py:254-288)
```python
def spring_force(self, compression: float) -> float:
    # Linear spring
    force = self._spring_strength * compression
    # Add progressive rate
    if compression > 0 and self._progressive_rate > 0:
        force += self._progressive_rate * compression * compression
    return force

def damper_force(self, velocity: float) -> float:
    if velocity > 0:
        # Compressing - use compression damping
        return self._damper_compression * velocity
    else:
        # Extending - use rebound damping
        return self._damper_rebound * velocity
```

### Limited Slip Differential (drivetrain.py:719-758)
```python
def _lsd_split(
    self,
    input_torque: float,
    left_speed: float,
    right_speed: float,
) -> Tuple[float, float]:
    speed_diff = abs(left_speed - right_speed)
    is_power = input_torque > 0
    lock_ratio = self._power_ratio if is_power else self._coast_ratio
    lock_torque = self._preload + abs(input_torque) * lock_ratio
    # Transfer torque from fast wheel to slow wheel
    if left_speed > right_speed:
        transfer = min(lock_torque, abs(input_torque) / 2)
        if speed_diff > 0.1:
            left_torque = input_torque / 2 - transfer * 0.5
            right_torque = input_torque / 2 + transfer * 0.5
        # ...
```

### Engine Torque Curve (drivetrain.py:216-244)
```python
def compute_torque(self, throttle: float, rpm: float) -> float:
    throttle = max(0.0, min(1.0, throttle))
    multiplier = self.get_torque_multiplier(rpm)  # Interpolates torque curve
    torque = self._max_torque * multiplier * throttle
    # Engine braking at closed throttle
    if throttle < 0.1:
        braking_factor = (0.1 - throttle) / 0.1
        torque -= self._max_torque * 0.1 * braking_factor
    # Friction loss
    friction_torque = self._friction * rpm
    return torque - friction_torque
```

### Ackermann Steering (wheeled_vehicle.py:396-424)
```python
if abs(steer_rad) > 0.001:
    turn_radius = self._wheelbase / math.tan(abs(steer_rad))
    inner_radius = turn_radius - self._track_width_front / 2
    outer_radius = turn_radius + self._track_width_front / 2
    inner_angle = math.atan(self._wheelbase / inner_radius)
    outer_angle = math.atan(self._wheelbase / outer_radius)
    # Blend based on Ackermann ratio
    inner_final = base_angle + (inner_angle - base_angle) * self._ackermann_ratio
    outer_final = base_angle + (outer_angle - base_angle) * self._ackermann_ratio
```
