"""
Watercraft simulation (boats, ships).

This module provides watercraft physics including buoyancy,
hull hydrodynamics, propulsion, and wave interaction.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from .config import (
    WATER_DENSITY,
    HULL_DRAG_COEFFICIENT,
    WAVE_FREQUENCY,
    WAVE_AMPLITUDE,
    PROPELLER_EFFICIENCY,
    GRAVITY,
)
from .vehicle_system import (
    Vector3,
    Transform,
    VehicleType,
    VehicleState,
    generate_vehicle_id,
)


class HullType(Enum):
    """Types of boat hulls."""

    DISPLACEMENT = auto()    # Traditional heavy hull
    PLANING = auto()         # High-speed planing hull
    CATAMARAN = auto()       # Twin hull
    HYDROFOIL = auto()       # Foil-assisted


class WatercraftType(Enum):
    """Types of watercraft."""

    SPEEDBOAT = auto()
    SAILBOAT = auto()
    YACHT = auto()
    CARGO_SHIP = auto()
    SUBMARINE = auto()


@dataclass
class BuoyancySamplePoint:
    """
    Sample point for buoyancy calculation.

    Multiple points distributed across hull provide
    realistic pitch and roll behavior.
    """

    local_position: Vector3 = field(default_factory=Vector3.zero)
    volume: float = 1.0              # Displaced volume when submerged (m^3)

    # Runtime state
    submerged_ratio: float = 0.0     # 0-1, how much is underwater
    buoyancy_force: float = 0.0      # Current force
    water_height: float = 0.0        # Water surface at this point


@dataclass
class Propeller:
    """
    Propulsion propeller.
    """

    local_position: Vector3 = field(default_factory=Vector3.zero)
    thrust_direction: Vector3 = field(default_factory=lambda: Vector3(0, 0, 1))
    max_thrust: float = 10000.0      # Maximum thrust (N)
    diameter: float = 0.5            # Propeller diameter (m)
    efficiency: float = PROPELLER_EFFICIENCY

    # State
    current_throttle: float = 0.0    # -1 to 1 (negative = reverse)
    current_thrust: float = 0.0
    rpm: float = 0.0


@dataclass
class Rudder:
    """
    Steering rudder.
    """

    local_position: Vector3 = field(default_factory=lambda: Vector3(0, 0, -3.0))
    area: float = 0.5                # Rudder area (m^2)
    max_angle: float = 35.0          # Maximum deflection (degrees)

    # State
    current_angle: float = 0.0       # Current angle (degrees)


@dataclass
class WaveState:
    """
    Current wave conditions.
    """

    amplitude: float = WAVE_AMPLITUDE
    frequency: float = WAVE_FREQUENCY
    direction: Vector3 = field(default_factory=lambda: Vector3(1, 0, 0))
    phase: float = 0.0               # Current phase


class Watercraft:
    """
    Watercraft simulation (boats, ships).

    Models buoyancy, hull drag, propulsion, and wave interaction.
    """

    def __init__(
        self,
        vehicle_id: Optional[str] = None,
        watercraft_type: WatercraftType = WatercraftType.SPEEDBOAT,
        hull_type: HullType = HullType.DISPLACEMENT,
        mass: float = 1000.0,
        length: float = 8.0,
        beam: float = 2.5,           # Width
        draft: float = 0.8,          # Depth below waterline
        displaced_volume: float = None,  # Auto-calculated if None
    ):
        """
        Initialize watercraft.

        Args:
            vehicle_id: Unique ID (generated if None).
            watercraft_type: Type of watercraft.
            hull_type: Hull configuration.
            mass: Boat mass (kg).
            length: Length overall (m).
            beam: Maximum width (m).
            draft: Design draft (m).
            displaced_volume: Displaced volume for buoyancy.
        """
        self.vehicle_id = vehicle_id or generate_vehicle_id()
        self.vehicle_type = VehicleType.WATERCRAFT
        self.state = VehicleState.ACTIVE

        self._watercraft_type = watercraft_type
        self._hull_type = hull_type
        self._mass = mass
        self._length = length
        self._beam = beam
        self._draft = draft

        # Calculate displaced volume for neutral buoyancy
        if displaced_volume is None:
            # Volume = mass / water_density
            self._displaced_volume = mass / WATER_DENSITY
        else:
            self._displaced_volume = displaced_volume

        # Guard against zero or negative volume
        if self._displaced_volume <= 0:
            raise ValueError("Displaced volume must be positive")

        # Inertia tensor
        self._inertia = Vector3(
            mass * (beam ** 2 + draft ** 2) / 12,      # Roll
            mass * (length ** 2 + beam ** 2) / 12,    # Yaw
            mass * (length ** 2 + draft ** 2) / 12,   # Pitch
        )

        # Transform and motion
        self.transform = Transform()
        self.velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()

        # Buoyancy sample points
        self._buoyancy_points: List[BuoyancySamplePoint] = []
        self._setup_buoyancy_points()

        # Propulsion
        self._propellers: List[Propeller] = [
            Propeller(
                local_position=Vector3(0, -draft / 2, -length / 3),
                max_thrust=mass * 3.0,  # Can accelerate at ~3 m/s^2
            )
        ]

        # Steering
        self._rudder = Rudder(
            local_position=Vector3(0, -draft / 2, -length / 2 + 0.5)
        )

        # Hydrodynamic coefficients
        self._hull_drag = HULL_DRAG_COEFFICIENT
        self._wetted_area = length * beam * 0.7 + 2 * length * draft
        self._lateral_drag = 2.0  # Higher resistance to sideways motion

        # Wave state
        self._waves = WaveState()
        self._time = 0.0

        # Controls
        self._throttle = 0.0         # -1 to 1
        self._steering = 0.0         # -1 to 1

        # Accumulated forces
        self._accumulated_force = Vector3.zero()
        self._accumulated_torque = Vector3.zero()

        # Water surface height (can vary with waves)
        self._water_height = 0.0

    @property
    def mass(self) -> float:
        """Boat mass."""
        return self._mass

    @property
    def speed(self) -> float:
        """Current speed (m/s)."""
        return self.velocity.magnitude()

    @property
    def speed_knots(self) -> float:
        """Current speed in knots."""
        return self.speed * 1.944

    @property
    def throttle(self) -> float:
        """Current throttle setting."""
        return self._throttle

    @throttle.setter
    def throttle(self, value: float) -> None:
        """Set throttle (-1 to 1)."""
        self._throttle = max(-1.0, min(1.0, value))

    @property
    def steering(self) -> float:
        """Current steering input."""
        return self._steering

    @steering.setter
    def steering(self, value: float) -> None:
        """Set steering (-1 to 1)."""
        self._steering = max(-1.0, min(1.0, value))

    def _setup_buoyancy_points(self) -> None:
        """Create buoyancy sample points across hull."""
        # Distribute points for stable calculation
        # More points = more accurate but slower

        # Volume per point
        num_points = 8
        volume_per_point = self._displaced_volume / num_points

        # Create points in a grid pattern
        half_length = self._length / 2 * 0.8
        half_beam = self._beam / 2 * 0.8
        depth = self._draft / 2

        positions = [
            Vector3(-half_beam, -depth, half_length),     # Front left
            Vector3(half_beam, -depth, half_length),      # Front right
            Vector3(-half_beam, -depth, 0),               # Mid left
            Vector3(half_beam, -depth, 0),                # Mid right
            Vector3(-half_beam, -depth, -half_length),    # Rear left
            Vector3(half_beam, -depth, -half_length),     # Rear right
            Vector3(0, -depth, half_length * 0.5),        # Center front
            Vector3(0, -depth, -half_length * 0.5),       # Center rear
        ]

        for pos in positions:
            self._buoyancy_points.append(BuoyancySamplePoint(
                local_position=pos,
                volume=volume_per_point,
            ))

    def get_water_height_at(self, world_pos: Vector3) -> float:
        """
        Get water surface height at a world position.

        Includes wave effects.

        Args:
            world_pos: World position.

        Returns:
            Water height.
        """
        # Base water level
        height = self._water_height

        # Add wave effect
        if self._waves.amplitude > 0:
            wave_phase = (
                world_pos.x * self._waves.direction.x +
                world_pos.z * self._waves.direction.z
            ) * self._waves.frequency * 2 * math.pi

            wave_offset = self._waves.amplitude * math.sin(
                wave_phase + self._waves.phase
            )
            height += wave_offset

        return height

    def calculate_buoyancy(self) -> Tuple[Vector3, Vector3]:
        """
        Calculate buoyancy force and torque.

        Returns:
            Tuple of (force, torque).
        """
        total_force = Vector3.zero()
        total_torque = Vector3.zero()

        yaw = math.radians(self.transform.rotation.y)
        pitch = math.radians(self.transform.rotation.z)
        roll = math.radians(self.transform.rotation.x)

        for point in self._buoyancy_points:
            # Transform point to world space (simplified rotation)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            cos_pitch = math.cos(pitch)
            sin_pitch = math.sin(pitch)
            cos_roll = math.cos(roll)
            sin_roll = math.sin(roll)

            local = point.local_position

            # Apply rotations (simplified Euler)
            world_x = (
                local.x * cos_yaw +
                local.z * sin_yaw
            ) + self.transform.position.x

            world_y = (
                local.y * cos_pitch * cos_roll -
                local.x * sin_roll +
                local.z * sin_pitch
            ) + self.transform.position.y

            world_z = (
                -local.x * sin_yaw +
                local.z * cos_yaw
            ) + self.transform.position.z

            world_pos = Vector3(world_x, world_y, world_z)

            # Get water height at this point
            water_height = self.get_water_height_at(world_pos)
            point.water_height = water_height

            # Calculate submersion
            submersion_depth = water_height - world_y

            if submersion_depth > 0:
                # Point is underwater
                # Ratio based on approximate point "height"
                point_height = self._draft  # Simplified
                point.submerged_ratio = min(1.0, submersion_depth / point_height)

                # Buoyancy force: F = rho * g * V_submerged
                submerged_volume = point.volume * point.submerged_ratio
                buoyancy = WATER_DENSITY * GRAVITY * submerged_volume
                point.buoyancy_force = buoyancy

                # Apply force upward
                force = Vector3(0, buoyancy, 0)
                total_force = total_force + force

                # Torque from off-center buoyancy
                moment_arm = Vector3(
                    world_pos.x - self.transform.position.x,
                    0,
                    world_pos.z - self.transform.position.z,
                )
                torque = moment_arm.cross(force)
                total_torque = total_torque + torque

            else:
                # Point above water
                point.submerged_ratio = 0.0
                point.buoyancy_force = 0.0

        return (total_force, total_torque)

    def calculate_hull_drag(self) -> Tuple[Vector3, Vector3]:
        """
        Calculate hull hydrodynamic drag.

        Returns:
            Tuple of (drag_force, drag_torque).
        """
        speed = self.velocity.magnitude()
        if speed < 0.01:
            return (Vector3.zero(), Vector3.zero())

        # Get velocity in local frame
        yaw = math.radians(self.transform.rotation.y)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        local_vx = self.velocity.x * cos_yaw + self.velocity.z * sin_yaw
        local_vz = -self.velocity.x * sin_yaw + self.velocity.z * cos_yaw

        # Forward drag (hull resistance)
        forward_speed = local_vz
        forward_drag_coeff = self._hull_drag

        # Planing hulls have reduced drag at speed
        if self._hull_type == HullType.PLANING and abs(forward_speed) > 10:
            forward_drag_coeff *= 0.5

        forward_drag = (
            0.5 * WATER_DENSITY * forward_speed ** 2 *
            forward_drag_coeff * self._wetted_area
        )
        forward_drag = math.copysign(forward_drag, -forward_speed)

        # Lateral drag (much higher - boats don't slide sideways easily)
        lateral_speed = local_vx
        lateral_drag = (
            0.5 * WATER_DENSITY * lateral_speed ** 2 *
            self._lateral_drag * self._length * self._draft
        )
        lateral_drag = math.copysign(lateral_drag, -lateral_speed)

        # Transform back to world space
        world_drag_x = forward_drag * sin_yaw + lateral_drag * cos_yaw
        world_drag_z = forward_drag * cos_yaw - lateral_drag * sin_yaw

        drag_force = Vector3(world_drag_x, 0, world_drag_z)

        # Yaw damping (resistance to turning)
        yaw_damping = (
            -self.angular_velocity.y *
            0.5 * WATER_DENSITY *
            self._length ** 3 * self._draft *
            0.5  # Yaw drag coefficient
        )
        drag_torque = Vector3(0, yaw_damping, 0)

        # Roll damping
        roll_damping = -self.angular_velocity.x * self._mass * 0.5
        drag_torque.x = roll_damping

        return (drag_force, drag_torque)

    def update_propulsion(self, dt: float) -> None:
        """
        Update propeller thrust.

        Args:
            dt: Delta time.
        """
        yaw = math.radians(self.transform.rotation.y)

        for prop in self._propellers:
            # Calculate thrust
            thrust_mag = prop.max_thrust * self._throttle * prop.efficiency

            # Reduce efficiency at high speed (cavitation, slip)
            speed = self.velocity.magnitude()
            speed_factor = 1.0 - min(0.5, speed / 30.0)
            thrust_mag *= speed_factor

            prop.current_thrust = thrust_mag

            # Thrust direction in world space
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            thrust_dir = Vector3(
                prop.thrust_direction.z * sin_yaw,
                prop.thrust_direction.y,
                prop.thrust_direction.z * cos_yaw,
            )

            thrust_force = thrust_dir * thrust_mag
            self.apply_force(thrust_force, prop.local_position)

    def update_steering(self, dt: float) -> None:
        """
        Update rudder steering.

        Args:
            dt: Delta time.
        """
        # Update rudder angle
        target_angle = self._steering * self._rudder.max_angle
        self._rudder.current_angle = target_angle  # Instant for simplicity

        # Rudder only effective when moving
        speed = self.velocity.magnitude()
        if speed < 0.5:
            return

        # Rudder force
        # F = 0.5 * rho * v^2 * A * CL
        # CL approximated as sin(angle) for flat plate
        angle_rad = math.radians(self._rudder.current_angle)
        cl = math.sin(2 * angle_rad)  # Approximation

        rudder_force = (
            0.5 * WATER_DENSITY * speed ** 2 *
            self._rudder.area * cl
        )

        # Force is lateral (creates yaw moment)
        yaw = math.radians(self.transform.rotation.y)
        lateral_dir = Vector3(math.cos(yaw), 0, -math.sin(yaw))

        force = lateral_dir * rudder_force
        self.apply_force(force, self._rudder.local_position)

    def update_waves(self, dt: float) -> None:
        """
        Update wave interaction.

        Args:
            dt: Delta time.
        """
        # Advance wave phase
        self._waves.phase += self._waves.frequency * 2 * math.pi * dt

        # Wave forces are handled through buoyancy calculation
        # as water height varies across the hull

    def apply_force(
        self,
        force: Vector3,
        position: Optional[Vector3] = None,
    ) -> None:
        """
        Apply force to watercraft.

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
        Apply torque to watercraft.

        Args:
            torque: Torque vector.
        """
        self._accumulated_torque = self._accumulated_torque + torque

    def update(self, dt: float) -> None:
        """
        Update watercraft for one physics step.

        Args:
            dt: Delta time.
        """
        if dt <= 0:
            return

        self._time += dt

        # Reset accumulated forces
        self._accumulated_force = Vector3.zero()
        self._accumulated_torque = Vector3.zero()

        # Gravity
        gravity = Vector3(0, -GRAVITY * self._mass, 0)
        self.apply_force(gravity)

        # Update waves
        self.update_waves(dt)

        # Buoyancy
        buoyancy_force, buoyancy_torque = self.calculate_buoyancy()
        self.apply_force(buoyancy_force)
        self.apply_torque(buoyancy_torque)

        # Hull drag
        drag_force, drag_torque = self.calculate_hull_drag()
        self.apply_force(drag_force)
        self.apply_torque(drag_torque)

        # Propulsion
        self.update_propulsion(dt)

        # Steering
        self.update_steering(dt)

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

        # Update rotation
        angular_deg = self.angular_velocity * (180.0 / math.pi) * dt
        self.transform.rotation = self.transform.rotation + angular_deg

    def reset(self) -> None:
        """Reset watercraft to initial state."""
        self.transform = Transform()
        # Start at water level
        self.transform.position.y = self._water_height + self._draft * 0.5
        self.velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()
        self._throttle = 0.0
        self._steering = 0.0
        self._rudder.current_angle = 0.0

    def set_wave_conditions(
        self,
        amplitude: float = 0.5,
        frequency: float = 0.5,
        direction: Vector3 = None,
    ) -> None:
        """
        Set wave conditions.

        Args:
            amplitude: Wave amplitude (m).
            frequency: Wave frequency (Hz).
            direction: Wave direction vector.
        """
        self._waves.amplitude = max(0.0, amplitude)
        self._waves.frequency = max(0.01, frequency)
        if direction is not None:
            self._waves.direction = direction.normalized()

    def set_water_height(self, height: float) -> None:
        """Set base water surface height."""
        self._water_height = height
