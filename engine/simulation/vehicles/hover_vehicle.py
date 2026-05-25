"""
Hover vehicle simulation (hovercraft, hovercars).

This module provides hover vehicle physics including air cushion,
skirt dynamics, and thrust vectoring.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from .config import (
    DEFAULT_HOVER_HEIGHT,
    HOVER_SPRING_CONSTANT,
    HOVER_DAMPING,
    SKIRT_DRAG_COEFFICIENT,
    AIR_DENSITY,
    GRAVITY,
    HOVER_CUSHION_AREA_RATIO,
    HOVER_AIR_DRAG_COEFFICIENT,
    HOVER_CUSHION_DRAG_COEFFICIENT,
    HOVER_LIFT_MARGIN,
    HOVER_MAX_PRESSURE_RATIO,
    HOVER_YAW_DAMPING,
)
from .vehicle_system import (
    Vector3,
    Transform,
    VehicleType,
    VehicleState,
    generate_vehicle_id,
)


class HoverMode(Enum):
    """Hover vehicle operating modes."""

    GROUND_EFFECT = auto()   # Standard hovercraft (air cushion)
    REPULSOR = auto()        # Sci-fi repulsor lift
    MAGNETIC = auto()        # Maglev-style


@dataclass
class LiftFan:
    """
    Lift fan generating hover cushion pressure.
    """

    local_position: Vector3 = field(default_factory=Vector3.zero)
    max_thrust: float = 50000.0      # Maximum lift force (N)
    efficiency: float = 0.8          # Efficiency factor
    current_power: float = 1.0       # Power setting (0-1)

    # State
    current_thrust: float = 0.0
    is_active: bool = True


@dataclass
class ThrustVector:
    """
    Thrust source for propulsion and steering.
    """

    local_position: Vector3 = field(default_factory=Vector3.zero)
    direction: Vector3 = field(default_factory=lambda: Vector3(0, 0, 1))  # Forward
    max_thrust: float = 10000.0      # Maximum thrust (N)
    gimbal_range: float = 30.0       # Maximum gimbal angle (degrees)

    # State
    current_thrust: float = 0.0
    gimbal_angle: float = 0.0        # Current gimbal (degrees)
    is_active: bool = True

    def get_thrust_vector(self) -> Vector3:
        """Get current thrust direction accounting for gimbal."""
        gimbal_rad = math.radians(self.gimbal_angle)
        cos_g = math.cos(gimbal_rad)
        sin_g = math.sin(gimbal_rad)

        # Rotate direction around vertical axis
        return Vector3(
            self.direction.x * cos_g - self.direction.z * sin_g,
            self.direction.y,
            self.direction.x * sin_g + self.direction.z * cos_g,
        )


@dataclass
class SkirtState:
    """
    State of the hover skirt (flexible seal).
    """

    compression: float = 0.0         # How much skirt is compressed
    pressure: float = 0.0            # Air cushion pressure
    leakage: float = 0.0             # Air leakage rate
    contact_points: int = 0          # Number of ground contact points


class HoverVehicle:
    """
    Hover vehicle simulation.

    Models air cushion vehicles (hovercraft) with:
    - Lift fan air cushion generation
    - Flexible skirt ground interaction
    - Thrust vectoring for propulsion and steering
    - Low-friction ground effect physics
    """

    def __init__(
        self,
        vehicle_id: Optional[str] = None,
        mass: float = 2000.0,
        length: float = 5.0,
        width: float = 3.0,
        hover_height: float = DEFAULT_HOVER_HEIGHT,
        hover_mode: HoverMode = HoverMode.GROUND_EFFECT,
        skirt_depth: float = 0.3,
        num_lift_fans: int = 1,
        num_thrust_vectors: int = 1,
    ):
        """
        Initialize hover vehicle.

        Args:
            vehicle_id: Unique ID (generated if None).
            mass: Vehicle mass (kg).
            length: Vehicle length (m).
            width: Vehicle width (m).
            hover_height: Target hover height (m).
            hover_mode: Type of hover system.
            skirt_depth: Skirt depth below hull (m).
            num_lift_fans: Number of lift fans.
            num_thrust_vectors: Number of thrust sources.
        """
        self.vehicle_id = vehicle_id or generate_vehicle_id()
        self.vehicle_type = VehicleType.HOVER
        self.state = VehicleState.ACTIVE

        # Physical properties
        self._mass = mass
        self._length = length
        self._width = width
        self._hover_height = hover_height
        self._hover_mode = hover_mode
        self._skirt_depth = skirt_depth

        # Cushion area
        self._cushion_area = length * width * HOVER_CUSHION_AREA_RATIO

        # Inertia
        self._inertia = Vector3(
            mass * (length ** 2 + 1.0) / 12,   # Roll
            mass * (length ** 2 + width ** 2) / 12,  # Yaw
            mass * (width ** 2 + 1.0) / 12,   # Pitch
        )

        # Transform and motion
        self.transform = Transform()
        self.velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()

        # Lift system
        self._lift_fans: List[LiftFan] = []
        self._setup_lift_fans(num_lift_fans)

        # Thrust system
        self._thrust_vectors: List[ThrustVector] = []
        self._setup_thrust_vectors(num_thrust_vectors)

        # Skirt
        self._skirt_state = SkirtState()
        self._skirt_spring = HOVER_SPRING_CONSTANT
        self._skirt_damping = HOVER_DAMPING

        # Drag coefficients
        self._air_drag = HOVER_AIR_DRAG_COEFFICIENT
        self._cushion_drag = HOVER_CUSHION_DRAG_COEFFICIENT
        self._skirt_drag = SKIRT_DRAG_COEFFICIENT

        # Control inputs
        self._throttle = 0.0           # 0 to 1
        self._lift_power = 1.0         # 0 to 1
        self._rudder = 0.0             # -1 to 1

        # Accumulated forces
        self._accumulated_force = Vector3.zero()
        self._accumulated_torque = Vector3.zero()

        # Height above ground (from raycast)
        self._ground_height = 0.0
        self._ground_normal = Vector3.up()

    @property
    def mass(self) -> float:
        """Vehicle mass."""
        return self._mass

    @property
    def hover_height(self) -> float:
        """Target hover height."""
        return self._hover_height

    @hover_height.setter
    def hover_height(self, value: float) -> None:
        """Set target hover height."""
        self._hover_height = max(0.1, value)

    @property
    def hover_force(self) -> float:
        """Current total lift force."""
        return sum(fan.current_thrust for fan in self._lift_fans)

    @property
    def speed(self) -> float:
        """Current speed (m/s)."""
        return self.velocity.magnitude()

    @property
    def throttle(self) -> float:
        """Current throttle input."""
        return self._throttle

    @throttle.setter
    def throttle(self, value: float) -> None:
        """Set throttle input."""
        self._throttle = max(0.0, min(1.0, value))

    @property
    def lift_power(self) -> float:
        """Current lift power setting."""
        return self._lift_power

    @lift_power.setter
    def lift_power(self, value: float) -> None:
        """Set lift power."""
        self._lift_power = max(0.0, min(1.0, value))

    @property
    def rudder(self) -> float:
        """Current rudder input."""
        return self._rudder

    @rudder.setter
    def rudder(self, value: float) -> None:
        """Set rudder input."""
        self._rudder = max(-1.0, min(1.0, value))

    def _setup_lift_fans(self, num_fans: int) -> None:
        """Create lift fans."""
        required_lift = self._mass * GRAVITY * HOVER_LIFT_MARGIN
        lift_per_fan = required_lift / max(1, num_fans)  # Guard against division by zero

        if num_fans == 1:
            # Single central fan
            self._lift_fans.append(LiftFan(
                local_position=Vector3.zero(),
                max_thrust=lift_per_fan,
            ))
        else:
            # Distributed fans
            spacing_x = self._width * 0.3
            spacing_z = self._length * 0.3

            positions = [
                Vector3(-spacing_x, 0, spacing_z),
                Vector3(spacing_x, 0, spacing_z),
                Vector3(-spacing_x, 0, -spacing_z),
                Vector3(spacing_x, 0, -spacing_z),
            ]

            for i in range(min(num_fans, len(positions))):
                self._lift_fans.append(LiftFan(
                    local_position=positions[i],
                    max_thrust=lift_per_fan,
                ))

    def _setup_thrust_vectors(self, num_thrusters: int) -> None:
        """Create thrust vectors."""
        base_thrust = self._mass * 5.0  # Acceleration capability

        if num_thrusters == 1:
            # Single rear thruster
            self._thrust_vectors.append(ThrustVector(
                local_position=Vector3(0, 0, -self._length / 2),
                direction=Vector3(0, 0, 1),
                max_thrust=base_thrust,
                gimbal_range=30.0,
            ))
        else:
            # Dual rear thrusters
            spacing = self._width * 0.3
            thrust_per = base_thrust / num_thrusters

            self._thrust_vectors.append(ThrustVector(
                local_position=Vector3(-spacing, 0, -self._length / 2),
                direction=Vector3(0, 0, 1),
                max_thrust=thrust_per,
            ))
            self._thrust_vectors.append(ThrustVector(
                local_position=Vector3(spacing, 0, -self._length / 2),
                direction=Vector3(0, 0, 1),
                max_thrust=thrust_per,
            ))

    def calculate_cushion_pressure(self, height: float) -> float:
        """
        Calculate air cushion pressure based on height.

        Args:
            height: Current height above ground.

        Returns:
            Cushion pressure (Pa).
        """
        if height >= self._hover_height + self._skirt_depth:
            # Too high - no cushion
            return 0.0

        if height <= 0:
            # On ground - maximum pressure
            height = 0.01

        # Pressure inversely related to height
        # As gap decreases, pressure increases
        target_pressure = (self._mass * 9.81) / self._cushion_area

        # Height ratio affects pressure
        height_ratio = self._hover_height / height
        pressure = target_pressure * min(height_ratio, HOVER_MAX_PRESSURE_RATIO)

        return pressure * self._lift_power

    def update_lift(self, dt: float) -> None:
        """
        Update lift fan forces.

        Args:
            dt: Delta time.
        """
        height = self.transform.position.y - self._ground_height

        # Calculate cushion pressure
        pressure = self.calculate_cushion_pressure(height)
        self._skirt_state.pressure = pressure

        # Lift force from pressure
        lift_force = pressure * self._cushion_area

        # Distribute among fans based on power
        total_power = sum(
            fan.current_power for fan in self._lift_fans if fan.is_active
        )

        for fan in self._lift_fans:
            if fan.is_active and total_power > 0:
                fan_ratio = fan.current_power / total_power
                fan.current_thrust = lift_force * fan_ratio * fan.efficiency
            else:
                fan.current_thrust = 0.0

        # Apply lift force
        self.apply_force(Vector3(0, lift_force, 0))

        # Update skirt contact
        skirt_bottom = height - self._skirt_depth
        if skirt_bottom < 0:
            # Skirt touching ground
            self._skirt_state.compression = -skirt_bottom
            self._skirt_state.contact_points = 4  # Simplified

            # Skirt spring force
            skirt_force = self._skirt_spring * self._skirt_state.compression
            skirt_force += self._skirt_damping * -self.velocity.y

            self.apply_force(Vector3(0, skirt_force, 0))
        else:
            self._skirt_state.compression = 0.0
            self._skirt_state.contact_points = 0

    def update_thrust(self, dt: float) -> None:
        """
        Update propulsion thrust.

        Args:
            dt: Delta time.
        """
        # Get vehicle forward direction
        yaw = math.radians(self.transform.rotation.y)

        for thruster in self._thrust_vectors:
            if not thruster.is_active:
                thruster.current_thrust = 0.0
                continue

            # Calculate thrust
            thrust_mag = thruster.max_thrust * self._throttle
            thruster.current_thrust = thrust_mag

            # Apply rudder to gimbal
            thruster.gimbal_angle = self._rudder * thruster.gimbal_range

            # Get thrust direction in world space
            local_dir = thruster.get_thrust_vector()

            # Transform to world space
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            world_dir = Vector3(
                local_dir.x * cos_yaw - local_dir.z * sin_yaw,
                local_dir.y,
                local_dir.x * sin_yaw + local_dir.z * cos_yaw,
            )

            # Apply thrust force
            thrust_force = world_dir * thrust_mag
            self.apply_force(thrust_force, thruster.local_position)

    def update_drag(self, dt: float) -> None:
        """
        Update drag forces.

        Args:
            dt: Delta time.
        """
        speed = self.velocity.magnitude()
        if speed < 0.1:
            return

        velocity_dir = self.velocity.normalized()

        # Aerodynamic drag
        aero_drag = 0.5 * AIR_DENSITY * speed ** 2 * self._air_drag * (
            self._length * 1.5  # Approximate frontal area
        )

        # Cushion drag (very low)
        height = self.transform.position.y - self._ground_height
        if height < self._hover_height + self._skirt_depth:
            cushion_drag = self._cushion_drag * self._mass * 9.81
        else:
            cushion_drag = 0.0

        # Skirt drag (if contacting)
        if self._skirt_state.contact_points > 0:
            skirt_drag = self._skirt_drag * self._mass * 9.81
        else:
            skirt_drag = 0.0

        total_drag = aero_drag + cushion_drag + skirt_drag

        # Apply drag force
        self.apply_force(velocity_dir * -total_drag)

        # Yaw damping
        yaw_damping = self.angular_velocity.y * -HOVER_YAW_DAMPING
        self.apply_torque(Vector3(0, yaw_damping, 0))

    def apply_force(
        self,
        force: Vector3,
        position: Optional[Vector3] = None,
    ) -> None:
        """
        Apply force to vehicle.

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
        Apply torque to vehicle.

        Args:
            torque: Torque vector.
        """
        self._accumulated_torque = self._accumulated_torque + torque

    def update(self, dt: float) -> None:
        """
        Update vehicle for one physics step.

        Args:
            dt: Delta time.
        """
        if dt <= 0:
            return

        # Reset accumulated forces
        self._accumulated_force = Vector3.zero()
        self._accumulated_torque = Vector3.zero()

        # Update lift system
        self.update_lift(dt)

        # Update thrust
        self.update_thrust(dt)

        # Update drag
        self.update_drag(dt)

        # Gravity
        gravity = Vector3(0, -GRAVITY * self._mass, 0)
        self.apply_force(gravity)

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

        # Clamp height to ground
        if self.transform.position.y < self._ground_height:
            self.transform.position.y = self._ground_height
            self.velocity.y = max(0, self.velocity.y)

        # Update rotation
        angular_deg = self.angular_velocity * (180.0 / math.pi) * dt
        self.transform.rotation = self.transform.rotation + angular_deg

    def reset(self) -> None:
        """Reset vehicle to initial state."""
        self.transform = Transform()
        self.transform.position.y = self._hover_height
        self.velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()
        self._throttle = 0.0
        self._lift_power = 1.0
        self._rudder = 0.0
        self._skirt_state = SkirtState()

    def set_ground_data(
        self,
        height: float,
        normal: Vector3 = None,
    ) -> None:
        """
        Set ground data from raycast.

        Args:
            height: Ground height at vehicle position.
            normal: Ground surface normal.
        """
        self._ground_height = height
        if normal is not None:
            self._ground_normal = normal
