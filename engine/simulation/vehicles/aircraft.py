"""
Aircraft simulation.

This module provides aircraft physics including aerodynamic forces,
control surfaces, engine thrust, and flight dynamics.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from .config import (
    AIR_DENSITY,
    AIRCRAFT_LIFT_SLOPE,
    AIRCRAFT_STALL_ANGLE,
    AIRCRAFT_PARASITE_DRAG,
    AIRCRAFT_OSWALD_EFFICIENCY,
    AIRCRAFT_SCALE_HEIGHT,
    AIRCRAFT_CONTROL_EFFECTIVENESS,
    AIRCRAFT_TAIL_MOMENT_ARM,
    AIRCRAFT_PITCH_STABILITY,
    AIRCRAFT_YAW_STABILITY,
    AIRCRAFT_ROLL_DAMPING,
    AIRCRAFT_GROUND_FRICTION,
    GRAVITY,
)
from .vehicle_system import (
    Vector3,
    Transform,
    VehicleType,
    VehicleState,
    generate_vehicle_id,
)


class AircraftType(Enum):
    """Types of aircraft."""

    FIXED_WING = auto()
    HELICOPTER = auto()
    VTOL = auto()        # Vertical takeoff and landing
    GLIDER = auto()


class FlightPhase(Enum):
    """Aircraft flight phases."""

    GROUNDED = auto()
    TAKEOFF = auto()
    CLIMB = auto()
    CRUISE = auto()
    DESCENT = auto()
    APPROACH = auto()
    LANDING = auto()
    HOVERING = auto()    # For helicopters/VTOL


@dataclass
class AerodynamicSurface:
    """
    An aerodynamic surface (wing, tail, control surface).

    Generates lift and drag based on angle of attack.
    """

    name: str = "surface"
    local_position: Vector3 = field(default_factory=Vector3.zero)

    # Geometry
    area: float = 10.0               # Surface area (m^2)
    span: float = 10.0               # Wingspan (m)
    chord: float = 1.0               # Mean chord length (m)
    aspect_ratio: float = 10.0       # Span^2 / Area

    # Aerodynamic coefficients
    lift_coeff_slope: float = AIRCRAFT_LIFT_SLOPE  # dCL/d_alpha
    zero_lift_aoa: float = -2.0      # AoA for zero lift (degrees)
    max_lift_coeff: float = 1.5      # Maximum CL
    min_lift_coeff: float = -1.0     # Minimum CL (inverted)

    drag_coeff_zero: float = 0.01    # Zero-lift drag
    induced_drag_factor: float = 0.04  # K for induced drag

    # Stall characteristics
    stall_angle: float = AIRCRAFT_STALL_ANGLE  # degrees
    post_stall_lift_factor: float = 0.5  # Lift reduction after stall

    # Control
    deflection: float = 0.0          # Current deflection (degrees)
    max_deflection: float = 30.0     # Maximum deflection (degrees)
    control_effectiveness: float = 0.05  # CL change per degree

    def __post_init__(self):
        """Calculate aspect ratio from span and area."""
        if self.area > 0:
            self.aspect_ratio = self.span ** 2 / self.area

    def compute_lift_coefficient(self, aoa: float) -> float:
        """
        Compute lift coefficient at given angle of attack.

        Args:
            aoa: Angle of attack in degrees.

        Returns:
            Lift coefficient (CL).
        """
        # Effective AoA including control deflection
        effective_aoa = aoa - self.zero_lift_aoa + self.deflection * self.control_effectiveness

        # Pre-stall (linear region)
        if abs(effective_aoa) < self.stall_angle:
            cl = self.lift_coeff_slope * math.radians(effective_aoa)
            return max(self.min_lift_coeff, min(self.max_lift_coeff, cl))

        # Post-stall behavior
        stall_cl = self.lift_coeff_slope * math.radians(
            math.copysign(self.stall_angle, effective_aoa)
        )

        # Gradual reduction after stall
        excess_angle = abs(effective_aoa) - self.stall_angle
        reduction = 1.0 - (excess_angle / 45.0) * (1 - self.post_stall_lift_factor)
        reduction = max(self.post_stall_lift_factor, reduction)

        cl = stall_cl * reduction
        return max(self.min_lift_coeff, min(self.max_lift_coeff, cl))

    def compute_drag_coefficient(self, cl: float, aoa: float) -> float:
        """
        Compute drag coefficient.

        Args:
            cl: Current lift coefficient.
            aoa: Angle of attack in degrees.

        Returns:
            Drag coefficient (CD).
        """
        # Parasite drag (zero-lift drag)
        cd0 = self.drag_coeff_zero

        # Induced drag: CDi = CL^2 / (pi * e * AR)
        if self.aspect_ratio > 0:
            cdi = cl ** 2 / (math.pi * AIRCRAFT_OSWALD_EFFICIENCY * self.aspect_ratio)
        else:
            cdi = self.induced_drag_factor * cl ** 2

        # Profile drag increase with deflection
        deflection_drag = 0.0001 * abs(self.deflection) ** 2

        # Post-stall drag increase
        if abs(aoa) > self.stall_angle:
            excess = abs(aoa) - self.stall_angle
            stall_drag = 0.01 * excess ** 2
        else:
            stall_drag = 0.0

        return cd0 + cdi + deflection_drag + stall_drag


@dataclass
class ControlSurface:
    """
    Aircraft control surface configuration.
    """

    # Control deflections (-1 to 1 input)
    aileron: float = 0.0     # Roll control
    elevator: float = 0.0    # Pitch control
    rudder: float = 0.0      # Yaw control
    flaps: float = 0.0       # High-lift devices (0-1)
    spoilers: float = 0.0    # Drag devices (0-1)

    # Limits
    max_aileron: float = 25.0    # degrees
    max_elevator: float = 20.0
    max_rudder: float = 25.0
    max_flaps: float = 40.0


@dataclass
class AircraftEngine:
    """
    Aircraft engine/motor.
    """

    local_position: Vector3 = field(default_factory=Vector3.zero)
    thrust_direction: Vector3 = field(default_factory=lambda: Vector3(0, 0, 1))
    max_thrust: float = 20000.0      # Maximum thrust (N)
    current_throttle: float = 0.0    # 0-1

    # For propeller aircraft
    is_propeller: bool = True
    propeller_efficiency: float = 0.85
    propeller_diameter: float = 2.0

    # State
    current_thrust: float = 0.0
    is_running: bool = True

    def compute_thrust(self, airspeed: float, altitude: float = 0.0) -> float:
        """
        Compute thrust at current conditions.

        Args:
            airspeed: Aircraft airspeed (m/s).
            altitude: Altitude (m).

        Returns:
            Thrust force (N).
        """
        if not self.is_running:
            return 0.0

        # Base thrust
        thrust = self.max_thrust * self.current_throttle

        # Altitude effect (air density)
        density_ratio = math.exp(-altitude / AIRCRAFT_SCALE_HEIGHT)
        thrust *= density_ratio

        # Propeller efficiency varies with airspeed
        if self.is_propeller and airspeed > 0:
            # Advance ratio effect (simplified)
            optimal_speed = 50.0  # m/s
            speed_ratio = airspeed / optimal_speed
            if speed_ratio < 1.0:
                efficiency = self.propeller_efficiency * (0.7 + 0.3 * speed_ratio)
            else:
                efficiency = self.propeller_efficiency * (1.0 - 0.1 * (speed_ratio - 1.0))
            efficiency = max(0.3, min(self.propeller_efficiency, efficiency))
            thrust *= efficiency

        return thrust


class Aircraft:
    """
    Fixed-wing aircraft simulation.

    Models aerodynamic lift, drag, thrust, and control surfaces.
    """

    def __init__(
        self,
        vehicle_id: Optional[str] = None,
        aircraft_type: AircraftType = AircraftType.FIXED_WING,
        mass: float = 1000.0,
        wing_area: float = 16.0,
        wing_span: float = 12.0,
        max_thrust: float = 5000.0,
    ):
        """
        Initialize aircraft.

        Args:
            vehicle_id: Unique ID (generated if None).
            aircraft_type: Type of aircraft.
            mass: Aircraft mass (kg).
            wing_area: Wing area (m^2).
            wing_span: Wing span (m).
            max_thrust: Maximum engine thrust (N).
        """
        self.vehicle_id = vehicle_id or generate_vehicle_id()
        self.vehicle_type = VehicleType.AIRCRAFT
        self.state = VehicleState.ACTIVE

        self._aircraft_type = aircraft_type
        self._mass = mass
        self._wing_area = wing_area
        self._wing_span = wing_span

        # Inertia tensor (simplified)
        self._inertia = Vector3(
            mass * wing_span ** 2 / 12,        # Roll
            mass * (wing_span ** 2 + 5.0) / 12,  # Yaw
            mass * 5.0 / 12,                   # Pitch
        )

        # Transform and motion
        self.transform = Transform()
        self.velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()

        # Main wing
        self._main_wing = AerodynamicSurface(
            name="main_wing",
            area=wing_area,
            span=wing_span,
            chord=wing_area / wing_span,
        )

        # Horizontal tail
        self._h_tail = AerodynamicSurface(
            name="h_tail",
            local_position=Vector3(0, 0, -4.0),
            area=wing_area * 0.2,
            span=wing_span * 0.35,
        )

        # Vertical tail
        self._v_tail = AerodynamicSurface(
            name="v_tail",
            local_position=Vector3(0, 0.5, -4.0),
            area=wing_area * 0.1,
            span=2.0,
        )

        # Engine(s)
        self._engines: List[AircraftEngine] = [
            AircraftEngine(max_thrust=max_thrust)
        ]

        # Control surfaces
        self._controls = ControlSurface()

        # Flight state
        self._flight_phase = FlightPhase.GROUNDED
        self._altitude = 0.0
        self._airspeed = 0.0
        self._angle_of_attack = 0.0
        self._sideslip = 0.0

        # Accumulated forces
        self._accumulated_force = Vector3.zero()
        self._accumulated_torque = Vector3.zero()

        # Ground contact
        self._is_grounded = True
        self._ground_height = 0.0

    @property
    def mass(self) -> float:
        """Aircraft mass."""
        return self._mass

    @property
    def aircraft_type(self) -> AircraftType:
        """Aircraft type."""
        return self._aircraft_type

    @property
    def airspeed(self) -> float:
        """True airspeed (m/s)."""
        return self._airspeed

    @property
    def altitude(self) -> float:
        """Altitude above sea level (m)."""
        return self._altitude

    @property
    def angle_of_attack(self) -> float:
        """Angle of attack (degrees)."""
        return self._angle_of_attack

    @property
    def flight_phase(self) -> FlightPhase:
        """Current flight phase."""
        return self._flight_phase

    @property
    def controls(self) -> ControlSurface:
        """Control surfaces."""
        return self._controls

    @property
    def is_stalled(self) -> bool:
        """Whether aircraft is in stall condition."""
        return abs(self._angle_of_attack) > self._main_wing.stall_angle

    def set_throttle(self, throttle: float) -> None:
        """Set engine throttle (0-1)."""
        throttle = max(0.0, min(1.0, throttle))
        for engine in self._engines:
            engine.current_throttle = throttle

    def set_control_inputs(
        self,
        pitch: float = 0.0,
        roll: float = 0.0,
        yaw: float = 0.0,
        flaps: float = 0.0,
    ) -> None:
        """
        Set control surface inputs.

        Args:
            pitch: Elevator input (-1 to 1).
            roll: Aileron input (-1 to 1).
            yaw: Rudder input (-1 to 1).
            flaps: Flap setting (0 to 1).
        """
        self._controls.elevator = max(-1.0, min(1.0, pitch))
        self._controls.aileron = max(-1.0, min(1.0, roll))
        self._controls.rudder = max(-1.0, min(1.0, yaw))
        self._controls.flaps = max(0.0, min(1.0, flaps))

    def compute_lift(self, dynamic_pressure: float) -> float:
        """
        Compute total lift force.

        Args:
            dynamic_pressure: Dynamic pressure (Pa).

        Returns:
            Lift force (N).
        """
        cl = self._main_wing.compute_lift_coefficient(self._angle_of_attack)

        # Flaps effect
        flap_cl_bonus = self._controls.flaps * 0.8
        cl += flap_cl_bonus

        lift = dynamic_pressure * self._wing_area * cl
        return lift

    def compute_drag(self, dynamic_pressure: float, lift: float) -> float:
        """
        Compute total drag force.

        Args:
            dynamic_pressure: Dynamic pressure (Pa).
            lift: Current lift force (N).

        Returns:
            Drag force (N).
        """
        # Guard against zero dynamic pressure
        if dynamic_pressure <= 0 or self._wing_area <= 0:
            return 0.0

        # Lift coefficient from lift force
        cl = lift / (dynamic_pressure * self._wing_area)

        cd = self._main_wing.compute_drag_coefficient(cl, self._angle_of_attack)

        # Add parasite drag from fuselage etc.
        cd += AIRCRAFT_PARASITE_DRAG

        # Flaps drag
        cd += self._controls.flaps * 0.05

        # Spoilers
        cd += self._controls.spoilers * 0.1

        drag = dynamic_pressure * self._wing_area * cd
        return drag

    def _update_flight_state(self) -> None:
        """Update flight state variables."""
        # Altitude
        self._altitude = self.transform.position.y - self._ground_height

        # Airspeed (simplified - no wind)
        self._airspeed = self.velocity.magnitude()

        # Calculate angle of attack and sideslip
        if self._airspeed > 1.0:
            # Transform velocity to body frame
            yaw = math.radians(self.transform.rotation.y)
            pitch = math.radians(self.transform.rotation.z)

            # Body-axis velocity components (simplified)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            vx = self.velocity.x * cos_yaw + self.velocity.z * sin_yaw
            vy = self.velocity.y
            vz = -self.velocity.x * sin_yaw + self.velocity.z * cos_yaw

            # Angle of attack
            horizontal_speed = math.sqrt(vx ** 2 + vz ** 2)
            if horizontal_speed > 0.1:
                self._angle_of_attack = math.degrees(math.atan2(-vy, vz))
            else:
                self._angle_of_attack = 0.0

            # Sideslip angle
            if abs(vz) > 0.1:
                self._sideslip = math.degrees(math.atan2(vx, abs(vz)))
            else:
                self._sideslip = 0.0
        else:
            self._angle_of_attack = 0.0
            self._sideslip = 0.0

        # Update flight phase
        if self._is_grounded:
            if self._airspeed > 5.0 and self._engines[0].current_throttle > 0.5:
                self._flight_phase = FlightPhase.TAKEOFF
            else:
                self._flight_phase = FlightPhase.GROUNDED
        else:
            if self.velocity.y > 2.0:
                self._flight_phase = FlightPhase.CLIMB
            elif self.velocity.y < -2.0:
                self._flight_phase = FlightPhase.DESCENT
            else:
                self._flight_phase = FlightPhase.CRUISE

    def update_aerodynamics(self, dt: float) -> None:
        """
        Update aerodynamic forces.

        Args:
            dt: Delta time.
        """
        # Dynamic pressure
        air_density = AIR_DENSITY * math.exp(-self._altitude / AIRCRAFT_SCALE_HEIGHT)
        q = 0.5 * air_density * self._airspeed ** 2

        if q < 1.0:
            # Not enough speed for meaningful aero
            return

        # Lift and drag
        lift = self.compute_lift(q)
        drag = self.compute_drag(q, lift)

        # Lift direction (perpendicular to velocity, in vertical plane)
        if self._airspeed > 1.0:
            vel_dir = self.velocity.normalized()

            # Lift perpendicular to velocity in vertical plane
            lift_dir = Vector3(
                -vel_dir.x * vel_dir.y,
                1 - vel_dir.y ** 2,
                -vel_dir.z * vel_dir.y,
            ).normalized()

            if lift_dir.magnitude() < 0.1:
                lift_dir = Vector3.up()

            # Drag opposite to velocity
            drag_dir = vel_dir * -1

            # Apply forces
            self.apply_force(lift_dir * lift)
            self.apply_force(drag_dir * drag)

    def update_control_moments(self, dt: float) -> None:
        """
        Update moments from control surfaces.

        Args:
            dt: Delta time.
        """
        air_density = AIR_DENSITY * math.exp(-self._altitude / AIRCRAFT_SCALE_HEIGHT)
        q = 0.5 * air_density * self._airspeed ** 2

        # Minimum effectiveness from propwash when engines running
        min_q = 0.0
        for engine in self._engines:
            if engine.is_running and engine.current_throttle > 0.1:
                # Propwash provides some airflow over control surfaces
                min_q = max(min_q, 50.0 * engine.current_throttle)

        q = max(q, min_q)

        if q < 10.0:
            return

        # Control effectiveness scales with dynamic pressure
        control_power = q * self._wing_area * AIRCRAFT_CONTROL_EFFECTIVENESS

        # Roll moment from ailerons
        roll_moment = (
            self._controls.aileron *
            self._controls.max_aileron *
            control_power *
            self._wing_span / 2
        )

        # Pitch moment from elevator
        pitch_moment = (
            self._controls.elevator *
            self._controls.max_elevator *
            control_power *
            AIRCRAFT_TAIL_MOMENT_ARM
        )

        # Yaw moment from rudder
        yaw_moment = (
            self._controls.rudder *
            self._controls.max_rudder *
            control_power *
            AIRCRAFT_TAIL_MOMENT_ARM
        )

        self.apply_torque(Vector3(roll_moment, yaw_moment, pitch_moment))

        # Stability derivatives (simplified)
        # Pitch stability
        pitch_stability = -self._angle_of_attack * q * self._wing_area * AIRCRAFT_PITCH_STABILITY
        self.apply_torque(Vector3(0, 0, pitch_stability))

        # Yaw stability (weathervane)
        yaw_stability = -self._sideslip * q * self._wing_area * AIRCRAFT_YAW_STABILITY
        self.apply_torque(Vector3(0, yaw_stability, 0))

        # Roll damping
        roll_damping = -self.angular_velocity.x * q * self._wing_area * AIRCRAFT_ROLL_DAMPING * self._wing_span
        self.apply_torque(Vector3(roll_damping, 0, 0))

    def update_thrust(self, dt: float) -> None:
        """
        Update engine thrust.

        Args:
            dt: Delta time.
        """
        total_thrust = Vector3.zero()

        for engine in self._engines:
            thrust = engine.compute_thrust(self._airspeed, self._altitude)
            engine.current_thrust = thrust

            # Thrust direction in world space
            yaw = math.radians(self.transform.rotation.y)
            pitch = math.radians(self.transform.rotation.z)

            # Simplified - thrust along aircraft forward axis
            thrust_dir = Vector3(
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            )

            thrust_vec = thrust_dir * thrust
            self.apply_force(thrust_vec, engine.local_position)

    def apply_force(
        self,
        force: Vector3,
        position: Optional[Vector3] = None,
    ) -> None:
        """
        Apply force to aircraft.

        Args:
            force: Force vector.
            position: Application point (local). CG if None.
        """
        self._accumulated_force = self._accumulated_force + force

        if position is not None:
            torque = position.cross(force)
            self._accumulated_torque = self._accumulated_torque + torque

    def apply_torque(self, torque: Vector3) -> None:
        """
        Apply torque to aircraft.

        Args:
            torque: Torque vector.
        """
        self._accumulated_torque = self._accumulated_torque + torque

    def update(self, dt: float) -> None:
        """
        Update aircraft for one physics step.

        Args:
            dt: Delta time.
        """
        if dt <= 0:
            return

        # Reset accumulated forces
        self._accumulated_force = Vector3.zero()
        self._accumulated_torque = Vector3.zero()

        # Update flight state
        self._update_flight_state()

        # Gravity
        gravity = Vector3(0, -GRAVITY * self._mass, 0)
        self.apply_force(gravity)

        # Aerodynamic forces
        self.update_aerodynamics(dt)

        # Control moments
        self.update_control_moments(dt)

        # Thrust
        self.update_thrust(dt)

        # Ground contact
        if self._is_grounded:
            # Simple ground reaction
            if self._accumulated_force.y < 0:
                ground_reaction = -self._accumulated_force.y
                self.apply_force(Vector3(0, ground_reaction, 0))

                # Ground friction
                friction = AIRCRAFT_GROUND_FRICTION * self._mass * GRAVITY
                if self._airspeed > 0.1:
                    friction_dir = self.velocity.normalized() * -1
                    self.apply_force(friction_dir * friction)

        # Integrate motion
        self._integrate(dt)

    def _integrate(self, dt: float) -> None:
        """
        Integrate equations of motion.

        Args:
            dt: Delta time.
        """
        # Linear acceleration
        accel = self._accumulated_force / self._mass

        # Angular acceleration
        angular_accel = Vector3(
            self._accumulated_torque.x / self._inertia.x if self._inertia.x > 0 else 0,
            self._accumulated_torque.y / self._inertia.y if self._inertia.y > 0 else 0,
            self._accumulated_torque.z / self._inertia.z if self._inertia.z > 0 else 0,
        )

        # Update velocities
        self.velocity = self.velocity + accel * dt
        self.angular_velocity = self.angular_velocity + angular_accel * dt

        # Update position
        self.transform.position = self.transform.position + self.velocity * dt

        # Ground clamp
        if self.transform.position.y < self._ground_height:
            self.transform.position.y = self._ground_height
            if self.velocity.y < 0:
                self.velocity.y = 0
            self._is_grounded = True
        else:
            self._is_grounded = self.transform.position.y < self._ground_height + 1.0

        # Update rotation
        angular_deg = self.angular_velocity * (180.0 / math.pi) * dt
        self.transform.rotation = self.transform.rotation + angular_deg

    def reset(self) -> None:
        """Reset aircraft to initial state."""
        self.transform = Transform()
        self.velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()
        self._controls = ControlSurface()
        self._flight_phase = FlightPhase.GROUNDED
        self._is_grounded = True
        for engine in self._engines:
            engine.current_throttle = 0.0
