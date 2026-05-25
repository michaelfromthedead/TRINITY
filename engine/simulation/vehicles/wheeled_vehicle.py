"""
Wheeled vehicle simulation.

This module provides a complete wheeled vehicle implementation including
wheel physics, suspension, tire forces, and steering.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from .config import (
    DEFAULT_WHEEL_RADIUS,
    DEFAULT_WHEEL_MASS,
    DEFAULT_WHEEL_INERTIA,
    DEFAULT_BRAKE_TORQUE,
    BRAKE_BIAS_FRONT,
    HANDBRAKE_TORQUE,
    MAX_STEER_ANGLE,
    STEERING_RATE,
    STEERING_RETURN_RATE,
    ACKERMANN_RATIO,
    DEFAULT_DRAG_COEFFICIENT,
    DEFAULT_FRONTAL_AREA,
    AIR_DENSITY,
    GRAVITY,
)
from .vehicle_system import (
    Vector3,
    Transform,
    VehicleType,
    VehicleState,
    generate_vehicle_id,
)
from .suspension import Suspension, SuspensionType, AntiRollBar, SuspensionSystem
from .tire_model import TireModel, PacejkaTire, TireForces
from .drivetrain import Drivetrain, DrivetrainLayout


class WheelPosition(Enum):
    """Wheel positions on vehicle."""

    FRONT_LEFT = auto()
    FRONT_RIGHT = auto()
    REAR_LEFT = auto()
    REAR_RIGHT = auto()


@dataclass
class WheelState:
    """Current state of a wheel."""

    angular_velocity: float = 0.0   # rad/s
    steer_angle: float = 0.0        # radians
    camber: float = 0.0             # radians
    brake_torque: float = 0.0       # N*m
    drive_torque: float = 0.0       # N*m

    # Raycast results
    contact_point: Vector3 = field(default_factory=Vector3.zero)
    contact_normal: Vector3 = field(default_factory=Vector3.up)
    contact_distance: float = 0.0
    is_grounded: bool = False

    # Forces
    suspension_force: float = 0.0
    tire_forces: TireForces = field(default_factory=TireForces)


@dataclass
class Wheel:
    """
    Wheel configuration and state.

    Combines wheel geometry, suspension, and tire model.
    """

    # Position and configuration
    position: WheelPosition = WheelPosition.FRONT_LEFT
    local_position: Vector3 = field(default_factory=Vector3.zero)
    radius: float = DEFAULT_WHEEL_RADIUS
    width: float = 0.225
    mass: float = DEFAULT_WHEEL_MASS
    inertia: float = DEFAULT_WHEEL_INERTIA

    # Steering
    is_steerable: bool = False
    max_steer_angle: float = MAX_STEER_ANGLE  # degrees

    # Driven
    is_driven: bool = False

    # Braking
    has_handbrake: bool = False

    # Components (set after creation)
    suspension: Optional[Suspension] = None
    tire_model: Optional[TireModel] = None

    # State
    state: WheelState = field(default_factory=WheelState)

    def __post_init__(self):
        """Initialize components if not provided."""
        if self.suspension is None:
            self.suspension = Suspension()
        if self.tire_model is None:
            self.tire_model = PacejkaTire()


class WheeledVehicle:
    """
    Complete wheeled vehicle simulation.

    Handles multi-wheel physics including suspension, tires,
    drivetrain, steering, and braking.
    """

    def __init__(
        self,
        vehicle_id: Optional[str] = None,
        mass: float = 1500.0,
        wheelbase: float = 2.8,
        track_width_front: float = 1.6,
        track_width_rear: float = 1.6,
        cg_height: float = 0.5,
        drivetrain: Optional[Drivetrain] = None,
    ):
        """
        Initialize wheeled vehicle.

        Args:
            vehicle_id: Unique ID (generated if None).
            mass: Vehicle mass (kg).
            wheelbase: Distance between front and rear axles (m).
            track_width_front: Front track width (m).
            track_width_rear: Rear track width (m).
            cg_height: Center of gravity height (m).
            drivetrain: Drivetrain (creates default RWD if None).
        """
        self.vehicle_id = vehicle_id or generate_vehicle_id()
        self.vehicle_type = VehicleType.WHEELED
        self.state = VehicleState.ACTIVE

        # Physical properties
        self._mass = mass
        self._wheelbase = wheelbase
        self._track_width_front = track_width_front
        self._track_width_rear = track_width_rear
        self._cg_height = cg_height

        # Inertia tensor (simplified box approximation)
        self._inertia = Vector3(
            mass * (wheelbase ** 2 + cg_height ** 2) / 12,  # Roll (X)
            mass * (wheelbase ** 2 + track_width_front ** 2) / 12,  # Yaw (Y)
            mass * (cg_height ** 2 + track_width_front ** 2) / 12,  # Pitch (Z)
        )

        # Transform and motion
        self.transform = Transform()
        self.velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()

        # Local velocities (for slip calculations)
        self._local_velocity = Vector3.zero()
        self._local_angular_velocity = Vector3.zero()

        # Drivetrain
        self._drivetrain = drivetrain or Drivetrain(layout=DrivetrainLayout.RWD)

        # Initialize wheels
        self._wheels: List[Wheel] = []
        self._setup_wheels()

        # Anti-roll bars
        self._front_arb: Optional[AntiRollBar] = None
        self._rear_arb: Optional[AntiRollBar] = None

        # Steering
        self._steer_input = 0.0        # -1 to 1
        self._current_steer = 0.0      # Current steer angle (degrees)
        self._ackermann_ratio = ACKERMANN_RATIO

        # Throttle/brake
        self._throttle_input = 0.0     # 0 to 1
        self._brake_input = 0.0        # 0 to 1
        self._handbrake_input = 0.0    # 0 to 1

        # Brake configuration
        self._max_brake_torque = DEFAULT_BRAKE_TORQUE
        self._brake_bias_front = BRAKE_BIAS_FRONT
        self._handbrake_torque = HANDBRAKE_TORQUE

        # Aerodynamics
        self._drag_coefficient = DEFAULT_DRAG_COEFFICIENT
        self._frontal_area = DEFAULT_FRONTAL_AREA
        self._lift_coefficient = 0.0

        # Accumulated forces for this frame
        self._accumulated_force = Vector3.zero()
        self._accumulated_torque = Vector3.zero()

    @property
    def mass(self) -> float:
        """Vehicle mass."""
        return self._mass

    @property
    def wheelbase(self) -> float:
        """Wheelbase length."""
        return self._wheelbase

    @property
    def drivetrain(self) -> Drivetrain:
        """Drivetrain."""
        return self._drivetrain

    @property
    def steering_input(self) -> float:
        """Current steering input (-1 to 1)."""
        return self._steer_input

    @steering_input.setter
    def steering_input(self, value: float) -> None:
        """Set steering input."""
        self._steer_input = max(-1.0, min(1.0, value))

    @property
    def throttle_input(self) -> float:
        """Current throttle input (0 to 1)."""
        return self._throttle_input

    @throttle_input.setter
    def throttle_input(self, value: float) -> None:
        """Set throttle input."""
        self._throttle_input = max(0.0, min(1.0, value))

    @property
    def brake_input(self) -> float:
        """Current brake input (0 to 1)."""
        return self._brake_input

    @brake_input.setter
    def brake_input(self, value: float) -> None:
        """Set brake input."""
        self._brake_input = max(0.0, min(1.0, value))

    @property
    def handbrake_input(self) -> float:
        """Current handbrake input (0 to 1)."""
        return self._handbrake_input

    @handbrake_input.setter
    def handbrake_input(self, value: float) -> None:
        """Set handbrake input."""
        self._handbrake_input = max(0.0, min(1.0, value))

    @property
    def speed(self) -> float:
        """Current speed (m/s)."""
        return self.velocity.magnitude()

    @property
    def speed_kmh(self) -> float:
        """Current speed (km/h)."""
        return self.speed * 3.6

    @property
    def wheels(self) -> List[Wheel]:
        """List of wheels."""
        return self._wheels

    def _setup_wheels(self) -> None:
        """Create and position wheels."""
        # Front left
        fl = Wheel(
            position=WheelPosition.FRONT_LEFT,
            local_position=Vector3(
                -self._track_width_front / 2,
                0,
                self._wheelbase / 2,
            ),
            is_steerable=True,
            is_driven=(self._drivetrain.layout in (
                DrivetrainLayout.FWD,
                DrivetrainLayout.AWD,
                DrivetrainLayout.FOURWD,
            )),
        )

        # Front right
        fr = Wheel(
            position=WheelPosition.FRONT_RIGHT,
            local_position=Vector3(
                self._track_width_front / 2,
                0,
                self._wheelbase / 2,
            ),
            is_steerable=True,
            is_driven=(self._drivetrain.layout in (
                DrivetrainLayout.FWD,
                DrivetrainLayout.AWD,
                DrivetrainLayout.FOURWD,
            )),
        )

        # Rear left
        rl = Wheel(
            position=WheelPosition.REAR_LEFT,
            local_position=Vector3(
                -self._track_width_rear / 2,
                0,
                -self._wheelbase / 2,
            ),
            is_steerable=False,
            is_driven=(self._drivetrain.layout in (
                DrivetrainLayout.RWD,
                DrivetrainLayout.AWD,
                DrivetrainLayout.FOURWD,
            )),
            has_handbrake=True,
        )

        # Rear right
        rr = Wheel(
            position=WheelPosition.REAR_RIGHT,
            local_position=Vector3(
                self._track_width_rear / 2,
                0,
                -self._wheelbase / 2,
            ),
            is_steerable=False,
            is_driven=(self._drivetrain.layout in (
                DrivetrainLayout.RWD,
                DrivetrainLayout.AWD,
                DrivetrainLayout.FOURWD,
            )),
            has_handbrake=True,
        )

        self._wheels = [fl, fr, rl, rr]

    def get_wheel(self, position: WheelPosition) -> Wheel:
        """Get wheel by position."""
        for wheel in self._wheels:
            if wheel.position == position:
                return wheel
        raise ValueError(f"Wheel not found: {position}")

    def set_anti_roll_bars(
        self,
        front_stiffness: float = 5000.0,
        rear_stiffness: float = 3000.0,
    ) -> None:
        """
        Configure anti-roll bars.

        Args:
            front_stiffness: Front ARB stiffness.
            rear_stiffness: Rear ARB stiffness.
        """
        self._front_arb = AntiRollBar(stiffness=front_stiffness)
        self._rear_arb = AntiRollBar(stiffness=rear_stiffness)

    def update_steering(self, dt: float) -> None:
        """
        Update steering angles with Ackermann geometry.

        Args:
            dt: Delta time.
        """
        # Smoothly interpolate steering
        target_steer = self._steer_input * MAX_STEER_ANGLE

        if abs(self._steer_input) > 0.01:
            # Moving toward target
            rate = STEERING_RATE
        else:
            # Auto-centering
            rate = STEERING_RETURN_RATE
            target_steer = 0.0

        steer_diff = target_steer - self._current_steer
        max_change = rate * dt * 60.0  # Rate is per unit input

        if abs(steer_diff) <= max_change:
            self._current_steer = target_steer
        else:
            self._current_steer += math.copysign(max_change, steer_diff)

        # Apply Ackermann steering geometry
        steer_rad = math.radians(self._current_steer)

        if abs(steer_rad) > 0.001:
            # Calculate turn radius
            turn_radius = self._wheelbase / math.tan(abs(steer_rad))

            # Inner and outer wheel angles
            inner_radius = turn_radius - self._track_width_front / 2
            outer_radius = turn_radius + self._track_width_front / 2

            inner_angle = math.atan(self._wheelbase / inner_radius)
            outer_angle = math.atan(self._wheelbase / outer_radius)

            # Blend based on Ackermann ratio
            base_angle = abs(steer_rad)
            inner_final = base_angle + (inner_angle - base_angle) * self._ackermann_ratio
            outer_final = base_angle + (outer_angle - base_angle) * self._ackermann_ratio

            # Apply to wheels
            if self._current_steer > 0:  # Turning right
                self._wheels[0].state.steer_angle = inner_final  # FL (inner)
                self._wheels[1].state.steer_angle = outer_final  # FR (outer)
            else:  # Turning left
                self._wheels[0].state.steer_angle = -outer_final  # FL (outer)
                self._wheels[1].state.steer_angle = -inner_final  # FR (inner)
        else:
            # Straight
            self._wheels[0].state.steer_angle = 0.0
            self._wheels[1].state.steer_angle = 0.0

    def update_wheels(self, dt: float) -> None:
        """
        Update wheel physics (suspension and contact).

        This method should be called after raycast/sweep results
        are available for each wheel.

        Args:
            dt: Delta time.
        """
        for wheel in self._wheels:
            if wheel.state.is_grounded:
                # Update suspension
                suspension_length = wheel.state.contact_distance
                suspension_force = wheel.suspension.update(suspension_length, dt)
                wheel.state.suspension_force = suspension_force

                # Get velocities at wheel contact point
                # (simplified - using vehicle velocity)
                forward_vel, lateral_vel = self._get_wheel_velocities(wheel)

                # Update tire model and get forces
                tire_forces = wheel.tire_model.update(
                    wheel.state.angular_velocity,
                    wheel.radius,
                    forward_vel,
                    lateral_vel,
                    suspension_force,
                    wheel.state.camber,
                )
                wheel.state.tire_forces = tire_forces
            else:
                # Wheel in air
                wheel.state.suspension_force = 0.0
                wheel.state.tire_forces = TireForces()

                # Wheel spins down due to drag
                wheel.state.angular_velocity *= 0.99

    def _get_wheel_velocities(self, wheel: Wheel) -> Tuple[float, float]:
        """
        Get forward and lateral velocities at wheel contact.

        Args:
            wheel: The wheel.

        Returns:
            Tuple of (forward_velocity, lateral_velocity).
        """
        # Transform velocity to wheel local space
        # Simplified: using vehicle forward direction
        steer_angle = wheel.state.steer_angle

        # Rotate velocity by steering angle
        cos_steer = math.cos(steer_angle)
        sin_steer = math.sin(steer_angle)

        # Vehicle local velocity
        local_vx = self._local_velocity.x  # Lateral
        local_vz = self._local_velocity.z  # Forward

        # Account for angular velocity contribution at wheel position
        # v_wheel = v_body + omega x r
        omega_y = self._local_angular_velocity.y  # Yaw rate
        wheel_x = wheel.local_position.x
        wheel_z = wheel.local_position.z

        local_vx += -omega_y * wheel_z
        local_vz += omega_y * wheel_x

        # Rotate to wheel frame
        forward_vel = local_vz * cos_steer - local_vx * sin_steer
        lateral_vel = local_vz * sin_steer + local_vx * cos_steer

        return (forward_vel, lateral_vel)

    def apply_tire_forces(self) -> None:
        """Apply tire forces to vehicle body."""
        for wheel in self._wheels:
            if not wheel.state.is_grounded:
                continue

            forces = wheel.state.tire_forces

            # Transform forces from wheel to vehicle space
            steer_angle = wheel.state.steer_angle
            cos_steer = math.cos(steer_angle)
            sin_steer = math.sin(steer_angle)

            # Tire forces in vehicle space
            fx_vehicle = forces.longitudinal * cos_steer - forces.lateral * sin_steer
            fy_vehicle = forces.longitudinal * sin_steer + forces.lateral * cos_steer

            # Apply at wheel position
            force = Vector3(fy_vehicle, forces.vertical, fx_vehicle)
            self.apply_force(force, wheel.local_position)

    def apply_brakes(self) -> None:
        """Apply brake torque to wheels."""
        # Regular brakes
        total_brake = self._max_brake_torque * self._brake_input
        front_brake = total_brake * self._brake_bias_front / 2
        rear_brake = total_brake * (1 - self._brake_bias_front) / 2

        self._wheels[0].state.brake_torque = front_brake
        self._wheels[1].state.brake_torque = front_brake
        self._wheels[2].state.brake_torque = rear_brake
        self._wheels[3].state.brake_torque = rear_brake

        # Handbrake (rear wheels only)
        if self._handbrake_input > 0:
            handbrake = self._handbrake_torque * self._handbrake_input / 2
            self._wheels[2].state.brake_torque += handbrake
            self._wheels[3].state.brake_torque += handbrake

    def update_wheel_rotation(self, dt: float) -> None:
        """
        Update wheel angular velocities.

        Args:
            dt: Delta time.
        """
        # Get drive torques from drivetrain
        wheel_speeds = (
            self._wheels[0].state.angular_velocity,
            self._wheels[1].state.angular_velocity,
            self._wheels[2].state.angular_velocity,
            self._wheels[3].state.angular_velocity,
        )

        drive_torques = self._drivetrain.update(
            self._throttle_input,
            wheel_speeds,
            dt,
        )

        # Update each wheel
        for i, wheel in enumerate(self._wheels):
            drive_torque = drive_torques[i] if wheel.is_driven else 0.0
            wheel.state.drive_torque = drive_torque

            # Net torque = drive - brake - tire_reaction
            tire_reaction = wheel.state.tire_forces.longitudinal * wheel.radius
            brake_direction = -1 if wheel.state.angular_velocity > 0 else 1
            brake = wheel.state.brake_torque * brake_direction

            net_torque = drive_torque + brake - tire_reaction

            # Angular acceleration
            angular_accel = net_torque / wheel.inertia

            # Update angular velocity
            wheel.state.angular_velocity += angular_accel * dt

            # Prevent reversal due to braking
            if wheel.state.brake_torque > 0:
                if (
                    abs(wheel.state.angular_velocity) < 0.1 and
                    abs(drive_torque) < wheel.state.brake_torque
                ):
                    wheel.state.angular_velocity = 0.0

    def apply_aerodynamics(self) -> None:
        """Apply aerodynamic forces."""
        speed = self.velocity.magnitude()
        if speed < 1.0:
            return

        # Dynamic pressure
        q = 0.5 * AIR_DENSITY * speed ** 2

        # Drag force (opposes velocity)
        drag_force = q * self._drag_coefficient * self._frontal_area
        drag_direction = self.velocity.normalized() * -1
        drag = drag_direction * drag_force

        # Lift force (negative for downforce)
        lift_force = q * self._lift_coefficient * self._frontal_area
        lift = Vector3(0, -lift_force, 0)

        self.apply_force(drag)
        self.apply_force(lift)

    def apply_force(
        self,
        force: Vector3,
        position: Optional[Vector3] = None,
    ) -> None:
        """
        Apply force to vehicle.

        Args:
            force: Force vector in world space.
            position: Application point (local space). CG if None.
        """
        self._accumulated_force = self._accumulated_force + force

        if position is not None:
            # Torque = r x F
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

        # Update local velocity (for slip calculations)
        self._update_local_velocity()

        # Update steering
        self.update_steering(dt)

        # Update brakes
        self.apply_brakes()

        # Update wheels (requires raycast data)
        self.update_wheels(dt)

        # Apply tire forces
        self.apply_tire_forces()

        # Update wheel rotation
        self.update_wheel_rotation(dt)

        # Apply aerodynamic forces
        self.apply_aerodynamics()

        # Apply gravity (vehicle manages its own gravity for proper suspension interaction)
        gravity_force = Vector3(0, -GRAVITY * self._mass, 0)
        self.apply_force(gravity_force)

        # Integrate motion
        self._integrate(dt)

    def _update_local_velocity(self) -> None:
        """Transform velocity to local space."""
        # Simplified rotation (assuming Y-up, yaw rotation only)
        yaw = math.radians(self.transform.rotation.y)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        self._local_velocity = Vector3(
            self.velocity.x * cos_yaw + self.velocity.z * sin_yaw,
            self.velocity.y,
            -self.velocity.x * sin_yaw + self.velocity.z * cos_yaw,
        )

        self._local_angular_velocity = self.angular_velocity.copy()

    def _integrate(self, dt: float) -> None:
        """
        Integrate equations of motion.

        Args:
            dt: Delta time.
        """
        # Linear acceleration
        accel = self._accumulated_force / self._mass

        # Angular acceleration (simplified, ignoring cross-product terms)
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

        # Update rotation (Euler integration)
        angular_deg = self.angular_velocity * (180.0 / math.pi) * dt
        self.transform.rotation = self.transform.rotation + angular_deg

    def reset(self) -> None:
        """Reset vehicle to initial state."""
        self.transform = Transform()
        self.velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()
        self._local_velocity = Vector3.zero()
        self._steer_input = 0.0
        self._current_steer = 0.0
        self._throttle_input = 0.0
        self._brake_input = 0.0
        self._handbrake_input = 0.0

        for wheel in self._wheels:
            wheel.state = WheelState()
            wheel.suspension.reset()

    def set_raycast_result(
        self,
        wheel_position: WheelPosition,
        hit: bool,
        contact_point: Vector3,
        contact_normal: Vector3,
        distance: float,
    ) -> None:
        """
        Set raycast result for a wheel.

        Called by physics system after wheel raycasts.

        Args:
            wheel_position: Which wheel.
            hit: Whether ray hit ground.
            contact_point: World space contact point.
            contact_normal: Surface normal.
            distance: Distance from wheel center.
        """
        wheel = self.get_wheel(wheel_position)
        wheel.state.is_grounded = hit
        wheel.state.contact_point = contact_point
        wheel.state.contact_normal = contact_normal
        wheel.state.contact_distance = distance
