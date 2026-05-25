"""
Suspension system simulation.

This module provides suspension physics including spring-damper calculations,
various suspension types, anti-roll bar, and travel limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from .config import (
    DEFAULT_SUSPENSION_REST,
    DEFAULT_SPRING_STRENGTH,
    DEFAULT_DAMPER_COMPRESSION,
    DEFAULT_DAMPER_REBOUND,
    DEFAULT_SUSPENSION_TRAVEL,
    DEFAULT_ANTI_ROLL_STRENGTH,
)
from .vehicle_system import Vector3


class SuspensionType(Enum):
    """Types of suspension geometries."""

    SPRING_DAMPER = auto()      # Simple spring-damper (strut)
    DOUBLE_WISHBONE = auto()    # Double A-arm
    MACPHERSON = auto()         # MacPherson strut
    TRAILING_ARM = auto()       # Trailing arm (rear)
    MULTI_LINK = auto()         # Multi-link (complex)
    SOLID_AXLE = auto()         # Live axle
    TORSION_BEAM = auto()       # Twist beam (semi-independent)


@dataclass
class SuspensionState:
    """Current state of a suspension unit."""

    compression: float = 0.0            # Current compression (0 = rest)
    velocity: float = 0.0               # Compression velocity (positive = compressing)
    force: float = 0.0                  # Current force output
    contact_point: Vector3 = field(default_factory=Vector3.zero)
    contact_normal: Vector3 = field(default_factory=Vector3.up)
    is_grounded: bool = False


@dataclass
class SuspensionGeometry:
    """Geometry parameters for suspension calculations."""

    # Attachment points (in vehicle local space)
    upper_mount: Vector3 = field(default_factory=Vector3.zero)
    lower_mount: Vector3 = field(default_factory=Vector3.zero)
    steering_axis: Vector3 = field(default_factory=Vector3.up)

    # Geometry angles
    camber: float = 0.0         # Static camber angle (degrees)
    caster: float = 0.0         # Caster angle (degrees)
    toe: float = 0.0            # Toe angle (degrees, positive = toe-in)

    # Kingpin inclination
    kpi: float = 0.0            # Kingpin inclination (degrees)
    scrub_radius: float = 0.0   # Scrub radius (meters)

    def get_camber_gain(self, compression: float) -> float:
        """
        Calculate camber change from compression.

        Args:
            compression: Current compression amount.

        Returns:
            Additional camber in degrees.
        """
        # Simple linear approximation
        # Double wishbone typically has negative gain
        return -compression * 2.0  # degrees per meter

    def get_toe_change(self, compression: float) -> float:
        """
        Calculate toe change from compression (bump steer).

        Args:
            compression: Current compression amount.

        Returns:
            Toe change in degrees.
        """
        # Ideally zero for good geometry
        return compression * 0.5  # Small bump steer


class Suspension:
    """
    Suspension physics simulation.

    Handles spring force, damper force, travel limits, and geometry.
    """

    def __init__(
        self,
        suspension_type: SuspensionType = SuspensionType.SPRING_DAMPER,
        rest_length: float = DEFAULT_SUSPENSION_REST,
        spring_strength: float = DEFAULT_SPRING_STRENGTH,
        damper_compression: float = DEFAULT_DAMPER_COMPRESSION,
        damper_rebound: float = DEFAULT_DAMPER_REBOUND,
        travel: float = DEFAULT_SUSPENSION_TRAVEL,
        min_length: Optional[float] = None,
        max_length: Optional[float] = None,
    ):
        """
        Initialize suspension.

        Args:
            suspension_type: Type of suspension geometry.
            rest_length: Rest length of suspension.
            spring_strength: Spring stiffness (N/m).
            damper_compression: Compression damping (N*s/m).
            damper_rebound: Rebound damping (N*s/m).
            travel: Total suspension travel.
            min_length: Minimum length (fully compressed). Computed if None.
            max_length: Maximum length (fully extended). Computed if None.
        """
        self._type = suspension_type
        self._rest_length = rest_length
        self._spring_strength = spring_strength
        self._damper_compression = damper_compression
        self._damper_rebound = damper_rebound
        self._travel = travel

        # Compute travel limits if not specified
        self._min_length = min_length if min_length is not None else rest_length - travel / 2
        self._max_length = max_length if max_length is not None else rest_length + travel / 2

        # Current state
        self._state = SuspensionState()
        self._prev_length = rest_length
        self._current_length = rest_length

        # Geometry (optional)
        self._geometry: Optional[SuspensionGeometry] = None

        # Progressive spring (optional)
        self._progressive_rate: float = 0.0  # Additional stiffness per meter

        # Bump stops
        self._bump_stop_stiffness: float = 100000.0  # Very stiff at limits
        self._bump_stop_threshold: float = 0.02  # Start bump stop 2cm from limit

    @property
    def suspension_type(self) -> SuspensionType:
        """Suspension type."""
        return self._type

    @property
    def rest_length(self) -> float:
        """Rest length."""
        return self._rest_length

    @property
    def spring_strength(self) -> float:
        """Spring stiffness."""
        return self._spring_strength

    @spring_strength.setter
    def spring_strength(self, value: float) -> None:
        """Set spring stiffness."""
        if value < 0:
            raise ValueError("Spring strength must be non-negative")
        self._spring_strength = value

    @property
    def damper_compression(self) -> float:
        """Compression damping."""
        return self._damper_compression

    @damper_compression.setter
    def damper_compression(self, value: float) -> None:
        """Set compression damping."""
        if value < 0:
            raise ValueError("Damper compression must be non-negative")
        self._damper_compression = value

    @property
    def damper_rebound(self) -> float:
        """Rebound damping."""
        return self._damper_rebound

    @damper_rebound.setter
    def damper_rebound(self, value: float) -> None:
        """Set rebound damping."""
        if value < 0:
            raise ValueError("Damper rebound must be non-negative")
        self._damper_rebound = value

    @property
    def min_length(self) -> float:
        """Minimum length (fully compressed)."""
        return self._min_length

    @property
    def max_length(self) -> float:
        """Maximum length (fully extended)."""
        return self._max_length

    @property
    def travel(self) -> float:
        """Total suspension travel."""
        return self._travel

    @property
    def compression(self) -> float:
        """Current compression (positive = compressed)."""
        return self._state.compression

    @property
    def velocity(self) -> float:
        """Current compression velocity."""
        return self._state.velocity

    @property
    def state(self) -> SuspensionState:
        """Current suspension state."""
        return self._state

    @property
    def compression_ratio(self) -> float:
        """
        Compression as ratio of total travel.

        Returns:
            0.0 (fully extended) to 1.0 (fully compressed).
        """
        if self._travel <= 0:
            return 0.0
        compression = self._rest_length - self._current_length
        return max(0.0, min(1.0, (compression + self._travel / 2) / self._travel))

    def set_geometry(self, geometry: SuspensionGeometry) -> None:
        """Set suspension geometry."""
        self._geometry = geometry

    def set_progressive_rate(self, rate: float) -> None:
        """
        Set progressive spring rate.

        Args:
            rate: Additional stiffness per meter of compression.
        """
        self._progressive_rate = rate

    def spring_force(self, compression: float) -> float:
        """
        Calculate spring force for given compression.

        Args:
            compression: Compression amount (positive = compressed).

        Returns:
            Spring force (positive = extension force).
        """
        # Linear spring
        force = self._spring_strength * compression

        # Add progressive rate
        if compression > 0 and self._progressive_rate > 0:
            force += self._progressive_rate * compression * compression

        return force

    def damper_force(self, velocity: float) -> float:
        """
        Calculate damper force for given velocity.

        Args:
            velocity: Compression velocity (positive = compressing).

        Returns:
            Damper force (opposes motion).
        """
        if velocity > 0:
            # Compressing - use compression damping
            return self._damper_compression * velocity
        else:
            # Extending - use rebound damping
            return self._damper_rebound * velocity

    def bump_stop_force(self, length: float) -> float:
        """
        Calculate bump stop force at travel limits.

        Uses progressive stiffness to avoid sudden force spikes.

        Args:
            length: Current suspension length.

        Returns:
            Bump stop force (adds to spring force).
        """
        force = 0.0

        # Compression bump stop
        distance_to_min = length - self._min_length
        if distance_to_min < self._bump_stop_threshold:
            penetration = self._bump_stop_threshold - distance_to_min
            # Use squared penetration for progressive force (smoother engagement)
            normalized_penetration = penetration / self._bump_stop_threshold
            force += self._bump_stop_stiffness * penetration * (1.0 + normalized_penetration)

        # Extension bump stop (droop)
        distance_to_max = self._max_length - length
        if distance_to_max < self._bump_stop_threshold:
            penetration = self._bump_stop_threshold - distance_to_max
            # Use squared penetration for progressive force
            normalized_penetration = penetration / self._bump_stop_threshold
            force -= self._bump_stop_stiffness * penetration * (1.0 + normalized_penetration)

        return force

    def update(self, length: float, dt: float) -> float:
        """
        Update suspension state and calculate total force.

        Args:
            length: Current suspension length (from raycast).
            dt: Delta time.

        Returns:
            Total suspension force.
        """
        # Clamp to travel limits
        clamped_length = max(self._min_length, min(self._max_length, length))

        # Calculate compression and velocity
        compression = self._rest_length - clamped_length
        velocity = (self._prev_length - clamped_length) / dt if dt > 0 else 0.0

        # Calculate forces
        spring = self.spring_force(compression)
        damper = self.damper_force(velocity)
        bump_stop = self.bump_stop_force(clamped_length)

        total_force = spring + damper + bump_stop

        # Update state
        self._state.compression = compression
        self._state.velocity = velocity
        self._state.force = total_force
        self._state.is_grounded = length < self._max_length

        # Store for next frame
        self._prev_length = clamped_length
        self._current_length = clamped_length

        return total_force

    def reset(self) -> None:
        """Reset suspension to rest position."""
        self._state = SuspensionState()
        self._prev_length = self._rest_length
        self._current_length = self._rest_length


class AntiRollBar:
    """
    Anti-roll bar (sway bar) simulation.

    Connects left and right suspension to reduce body roll.
    """

    def __init__(
        self,
        stiffness: float = DEFAULT_ANTI_ROLL_STRENGTH,
        asymmetric: bool = False,
        asymmetric_ratio: float = 1.0,
    ):
        """
        Initialize anti-roll bar.

        Args:
            stiffness: Torsional stiffness (N*m/rad).
            asymmetric: Whether bar is asymmetric.
            asymmetric_ratio: Ratio for asymmetric bars.
        """
        self._stiffness = stiffness
        self._asymmetric = asymmetric
        self._asymmetric_ratio = asymmetric_ratio

    @property
    def stiffness(self) -> float:
        """Bar stiffness."""
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value: float) -> None:
        """Set bar stiffness."""
        if value < 0:
            raise ValueError("Stiffness must be non-negative")
        self._stiffness = value

    def calculate_force(
        self,
        left_compression: float,
        right_compression: float,
        wheel_base_half: float,
    ) -> Tuple[float, float]:
        """
        Calculate anti-roll bar forces.

        Args:
            left_compression: Left suspension compression.
            right_compression: Right suspension compression.
            wheel_base_half: Half of track width.

        Returns:
            Tuple of (left_force, right_force).
        """
        # Difference in compression creates twist
        compression_diff = left_compression - right_compression

        # Approximate angle from compression difference
        # angle = atan(diff / track_width) ~= diff / track_width for small angles
        angle = compression_diff / (2 * wheel_base_half) if wheel_base_half > 0 else 0

        # Torque from bar
        torque = self._stiffness * angle

        # Convert to forces at wheels
        force_magnitude = torque / wheel_base_half if wheel_base_half > 0 else 0

        left_force = -force_magnitude
        right_force = force_magnitude

        if self._asymmetric:
            left_force *= self._asymmetric_ratio

        return (left_force, right_force)


class SuspensionSystem:
    """
    Complete suspension system for one axle.

    Manages left/right suspension units and anti-roll bar.
    """

    def __init__(
        self,
        suspension_type: SuspensionType = SuspensionType.DOUBLE_WISHBONE,
        track_width: float = 1.6,
        **suspension_kwargs,
    ):
        """
        Initialize suspension system for an axle.

        Args:
            suspension_type: Type of suspension.
            track_width: Distance between left and right wheels.
            **suspension_kwargs: Arguments passed to Suspension.
        """
        self._track_width = track_width

        # Create left and right suspensions
        self._left = Suspension(
            suspension_type=suspension_type,
            **suspension_kwargs,
        )
        self._right = Suspension(
            suspension_type=suspension_type,
            **suspension_kwargs,
        )

        # Anti-roll bar (optional)
        self._anti_roll_bar: Optional[AntiRollBar] = None

    @property
    def left(self) -> Suspension:
        """Left suspension unit."""
        return self._left

    @property
    def right(self) -> Suspension:
        """Right suspension unit."""
        return self._right

    @property
    def track_width(self) -> float:
        """Track width."""
        return self._track_width

    def set_anti_roll_bar(self, anti_roll_bar: Optional[AntiRollBar]) -> None:
        """Set or remove anti-roll bar."""
        self._anti_roll_bar = anti_roll_bar

    def update(
        self,
        left_length: float,
        right_length: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Update suspension system.

        Args:
            left_length: Left suspension length from raycast.
            right_length: Right suspension length from raycast.
            dt: Delta time.

        Returns:
            Tuple of (left_force, right_force).
        """
        # Update individual suspensions
        left_force = self._left.update(left_length, dt)
        right_force = self._right.update(right_length, dt)

        # Add anti-roll bar forces
        if self._anti_roll_bar is not None:
            arb_left, arb_right = self._anti_roll_bar.calculate_force(
                self._left.compression,
                self._right.compression,
                self._track_width / 2,
            )
            left_force += arb_left
            right_force += arb_right

        return (left_force, right_force)

    def get_roll_angle(self) -> float:
        """
        Estimate body roll angle from suspension compressions.

        Returns:
            Roll angle in radians (positive = roll right).
        """
        import math
        compression_diff = self._left.compression - self._right.compression
        if self._track_width <= 0:
            return 0.0
        return math.atan2(compression_diff, self._track_width)

    def reset(self) -> None:
        """Reset suspension system."""
        self._left.reset()
        self._right.reset()
