"""
Tracked vehicle simulation (tanks, excavators, etc.).

This module provides tracked vehicle physics including track simulation,
differential steering, and ground interaction.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from .config import (
    TRACK_FRICTION,
    TRACK_ROLLING_RESISTANCE,
    TRACK_WIDTH,
    TRACK_LENGTH,
    GRAVITY,
)
from .vehicle_system import (
    Vector3,
    Transform,
    VehicleType,
    VehicleState,
    generate_vehicle_id,
)
from .suspension import Suspension


class TrackType(Enum):
    """Types of track systems."""

    STANDARD = auto()       # Standard tank track
    RUBBER = auto()         # Rubber track (lighter vehicles)
    SEGMENTED = auto()      # Segmented metal track


@dataclass
class RoadWheel:
    """
    Road wheel on a tracked vehicle.

    Road wheels support the track and distribute weight.
    """

    local_position: Vector3 = field(default_factory=Vector3.zero)
    radius: float = 0.3
    suspension: Optional[Suspension] = None

    # State
    angular_velocity: float = 0.0
    contact_force: float = 0.0
    is_grounded: bool = False
    ground_height: float = 0.0


@dataclass
class TrackState:
    """Current state of a track."""

    velocity: float = 0.0           # Track linear velocity (m/s)
    tension: float = 1000.0         # Track tension (N)
    slip: float = 0.0               # Longitudinal slip
    lateral_slip: float = 0.0       # Lateral slip
    longitudinal_force: float = 0.0
    lateral_force: float = 0.0
    ground_contact_length: float = 0.0  # Length of track on ground


@dataclass
class Track:
    """
    Track assembly for one side of vehicle.

    Includes multiple road wheels and track physics.
    """

    side: str = "left"              # "left" or "right"
    track_type: TrackType = TrackType.STANDARD

    # Geometry
    width: float = TRACK_WIDTH
    contact_length: float = TRACK_LENGTH
    sprocket_radius: float = 0.4    # Drive sprocket radius

    # Road wheels
    road_wheels: List[RoadWheel] = field(default_factory=list)

    # Physics properties
    mass: float = 500.0             # Track mass
    friction_coefficient: float = TRACK_FRICTION
    rolling_resistance: float = TRACK_ROLLING_RESISTANCE

    # State
    state: TrackState = field(default_factory=TrackState)

    # Throttle input for this track (differential steering)
    throttle: float = 0.0           # -1 to 1

    def __post_init__(self):
        """Initialize default road wheels if none provided."""
        if not self.road_wheels:
            # Create default road wheels along track length
            num_wheels = 5
            spacing = self.contact_length / (num_wheels + 1)
            for i in range(num_wheels):
                z_pos = (i + 1) * spacing - self.contact_length / 2
                wheel = RoadWheel(
                    local_position=Vector3(0, 0, z_pos),
                    radius=0.25,
                    suspension=Suspension(
                        spring_strength=80000.0,
                        damper_compression=8000.0,
                        damper_rebound=6000.0,
                    ),
                )
                self.road_wheels.append(wheel)


class TrackedVehicle:
    """
    Tracked vehicle simulation (tanks, bulldozers, etc.).

    Uses differential steering where left and right tracks
    can be controlled independently.
    """

    def __init__(
        self,
        vehicle_id: Optional[str] = None,
        mass: float = 40000.0,
        length: float = 7.0,
        width: float = 3.5,
        height: float = 2.5,
        track_separation: float = 2.8,
        max_engine_torque: float = 4000.0,
        max_speed: float = 15.0,  # m/s (~54 km/h)
    ):
        """
        Initialize tracked vehicle.

        Args:
            vehicle_id: Unique ID (generated if None).
            mass: Vehicle mass (kg).
            length: Vehicle length (m).
            width: Vehicle width (m).
            height: Vehicle height (m).
            track_separation: Distance between track centers (m).
            max_engine_torque: Maximum engine torque (N*m).
            max_speed: Maximum forward speed (m/s).
        """
        self.vehicle_id = vehicle_id or generate_vehicle_id()
        self.vehicle_type = VehicleType.TRACKED
        self.state = VehicleState.ACTIVE

        # Dimensions
        self._mass = mass
        self._length = length
        self._width = width
        self._height = height
        self._track_separation = track_separation

        # Inertia (box approximation)
        self._inertia = Vector3(
            mass * (height ** 2 + length ** 2) / 12,   # Roll
            mass * (width ** 2 + length ** 2) / 12,    # Yaw
            mass * (width ** 2 + height ** 2) / 12,    # Pitch
        )

        # Transform and motion
        self.transform = Transform()
        self.velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()

        # Engine
        self._max_engine_torque = max_engine_torque
        self._max_speed = max_speed
        self._engine_rpm = 0.0

        # Create tracks
        self._left_track = Track(
            side="left",
            contact_length=length * 0.8,
        )
        self._right_track = Track(
            side="right",
            contact_length=length * 0.8,
        )

        # Input
        self._left_track_throttle = 0.0   # -1 to 1
        self._right_track_throttle = 0.0  # -1 to 1
        self._brake_input = 0.0           # 0 to 1

        # Accumulated forces
        self._accumulated_force = Vector3.zero()
        self._accumulated_torque = Vector3.zero()

        # Steering mode
        self._steer_input = 0.0           # -1 to 1 (combined with throttle)
        self._combined_steering = True     # Use combined throttle/steer

    @property
    def mass(self) -> float:
        """Vehicle mass."""
        return self._mass

    @property
    def left_track(self) -> Track:
        """Left track."""
        return self._left_track

    @property
    def right_track(self) -> Track:
        """Right track."""
        return self._right_track

    @property
    def left_track_throttle(self) -> float:
        """Left track throttle."""
        return self._left_track_throttle

    @left_track_throttle.setter
    def left_track_throttle(self, value: float) -> None:
        """Set left track throttle."""
        self._left_track_throttle = max(-1.0, min(1.0, value))

    @property
    def right_track_throttle(self) -> float:
        """Right track throttle."""
        return self._right_track_throttle

    @right_track_throttle.setter
    def right_track_throttle(self, value: float) -> None:
        """Set right track throttle."""
        self._right_track_throttle = max(-1.0, min(1.0, value))

    @property
    def speed(self) -> float:
        """Current forward speed (m/s)."""
        # Project velocity onto forward direction
        yaw = math.radians(self.transform.rotation.y)
        forward = Vector3(math.sin(yaw), 0, math.cos(yaw))
        return self.velocity.dot(forward)

    def set_throttle_steer(self, throttle: float, steer: float) -> None:
        """
        Set combined throttle and steering.

        Converts to differential track throttles.

        Args:
            throttle: Forward/back throttle (-1 to 1).
            steer: Steering input (-1 to 1, negative = left).
        """
        throttle = max(-1.0, min(1.0, throttle))
        steer = max(-1.0, min(1.0, steer))

        # Differential steering
        # Positive steer turns right (left track faster)
        self._left_track_throttle = throttle + steer
        self._right_track_throttle = throttle - steer

        # Clamp and normalize
        max_val = max(abs(self._left_track_throttle), abs(self._right_track_throttle), 1.0)
        self._left_track_throttle /= max_val
        self._right_track_throttle /= max_val

    def _update_track(
        self,
        track: Track,
        throttle: float,
        weight_on_track: float,
        forward_velocity: float,
        lateral_velocity: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Update track physics.

        Args:
            track: The track to update.
            throttle: Track throttle (-1 to 1).
            weight_on_track: Normal force on track.
            forward_velocity: Forward velocity at track.
            lateral_velocity: Lateral velocity at track.
            dt: Delta time.

        Returns:
            Tuple of (longitudinal_force, lateral_force).
        """
        # Maximum force from friction
        max_friction_force = track.friction_coefficient * weight_on_track

        # Drive force from engine
        drive_force = throttle * self._max_engine_torque / track.sprocket_radius

        # Speed limiting (simple model)
        speed_ratio = abs(forward_velocity) / self._max_speed if self._max_speed > 0 else 0
        if speed_ratio > 1.0:
            drive_force *= 0.0  # No more acceleration at max speed
        elif speed_ratio > 0.8:
            drive_force *= (1.0 - speed_ratio) / 0.2  # Fade power near top speed

        # Clamp drive force to friction limit
        longitudinal_force = max(-max_friction_force, min(max_friction_force, drive_force))

        # Rolling resistance
        if abs(forward_velocity) > 0.1:
            rolling_force = track.rolling_resistance * weight_on_track
            rolling_force *= -1 if forward_velocity > 0 else 1
            longitudinal_force += rolling_force

        # Calculate slip
        if weight_on_track > 0:
            track.state.slip = abs(drive_force) / max_friction_force if max_friction_force > 0 else 0
        else:
            track.state.slip = 0

        # Lateral force (track resists sliding)
        # Tracks have very high lateral friction
        lateral_friction = track.friction_coefficient * 1.5  # Tracks grip better laterally
        max_lateral = lateral_friction * weight_on_track

        # Lateral force opposes lateral velocity
        if abs(lateral_velocity) > 0.1:
            lateral_force = -lateral_velocity * 10000.0  # High lateral stiffness
            lateral_force = max(-max_lateral, min(max_lateral, lateral_force))
        else:
            lateral_force = 0.0

        # Update track state
        track.state.velocity = forward_velocity
        track.state.longitudinal_force = longitudinal_force
        track.state.lateral_force = lateral_force
        track.state.lateral_slip = abs(lateral_velocity) / max(abs(forward_velocity), 1.0)

        return (longitudinal_force, lateral_force)

    def update_tracks(self, dt: float) -> None:
        """
        Update both tracks and compute forces.

        Args:
            dt: Delta time.
        """
        # Weight distribution (assuming level ground)
        total_weight = self._mass * GRAVITY
        weight_per_track = total_weight / 2

        # Get velocities at each track position
        yaw = math.radians(self.transform.rotation.y)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # Transform velocity to local space
        local_vx = self.velocity.x * cos_yaw + self.velocity.z * sin_yaw
        local_vz = -self.velocity.x * sin_yaw + self.velocity.z * cos_yaw

        # Yaw rate contribution to track velocities
        yaw_rate = self.angular_velocity.y
        half_separation = self._track_separation / 2

        left_forward_vel = local_vz - yaw_rate * half_separation
        right_forward_vel = local_vz + yaw_rate * half_separation

        # Update tracks
        left_long, left_lat = self._update_track(
            self._left_track,
            self._left_track_throttle,
            weight_per_track,
            left_forward_vel,
            local_vx,
            dt,
        )

        right_long, right_lat = self._update_track(
            self._right_track,
            self._right_track_throttle,
            weight_per_track,
            right_forward_vel,
            local_vx,
            dt,
        )

        # Apply brakes
        if self._brake_input > 0:
            brake_force = self._brake_input * total_weight * 0.8  # 80% of weight as brake force
            if left_forward_vel > 0:
                left_long -= brake_force / 2
            elif left_forward_vel < 0:
                left_long += brake_force / 2

            if right_forward_vel > 0:
                right_long -= brake_force / 2
            elif right_forward_vel < 0:
                right_long += brake_force / 2

        # Transform forces to world space
        total_long = left_long + right_long
        total_lat = left_lat + right_lat

        # World space force
        world_fx = total_long * sin_yaw + total_lat * cos_yaw
        world_fz = total_long * cos_yaw - total_lat * sin_yaw

        self.apply_force(Vector3(world_fx, 0, world_fz))

        # Yaw torque from differential thrust
        yaw_torque = (right_long - left_long) * half_separation
        self.apply_torque(Vector3(0, yaw_torque, 0))

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

        # Update tracks
        self.update_tracks(dt)

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

        # Damping (tracks have high resistance to rotation)
        self.angular_velocity = self.angular_velocity * 0.98

        # Update position
        self.transform.position = self.transform.position + self.velocity * dt

        # Update rotation
        angular_deg = self.angular_velocity * (180.0 / math.pi) * dt
        self.transform.rotation = self.transform.rotation + angular_deg

    def reset(self) -> None:
        """Reset vehicle to initial state."""
        self.transform = Transform()
        self.velocity = Vector3.zero()
        self.angular_velocity = Vector3.zero()
        self._left_track_throttle = 0.0
        self._right_track_throttle = 0.0
        self._brake_input = 0.0
        self._left_track.state = TrackState()
        self._right_track.state = TrackState()

    def pivot_turn(self, direction: float) -> None:
        """
        Execute a pivot turn (neutral steering).

        Args:
            direction: Turn direction (-1 = left, 1 = right).
        """
        direction = max(-1.0, min(1.0, direction))
        self._left_track_throttle = direction
        self._right_track_throttle = -direction
