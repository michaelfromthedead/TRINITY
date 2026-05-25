"""
Tire physics models.

This module provides tire force calculation using various models including
the Pacejka Magic Formula and a simplified linear model.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

from .config import (
    PACEJKA_B_LONGITUDINAL,
    PACEJKA_C_LONGITUDINAL,
    PACEJKA_D_LONGITUDINAL,
    PACEJKA_E_LONGITUDINAL,
    PACEJKA_B_LATERAL,
    PACEJKA_C_LATERAL,
    PACEJKA_D_LATERAL,
    PACEJKA_E_LATERAL,
    TIRE_FRICTION_COEFFICIENT,
    TIRE_LOAD_SENSITIVITY,
    TIRE_ROLLING_RESISTANCE,
)
from .vehicle_system import Vector3


class TireSurface(Enum):
    """Surface types affecting tire grip."""

    ASPHALT_DRY = auto()
    ASPHALT_WET = auto()
    CONCRETE = auto()
    GRAVEL = auto()
    DIRT = auto()
    GRASS = auto()
    SAND = auto()
    SNOW = auto()
    ICE = auto()


# Surface friction multipliers
SURFACE_FRICTION: dict[TireSurface, float] = {
    TireSurface.ASPHALT_DRY: 1.0,
    TireSurface.ASPHALT_WET: 0.7,
    TireSurface.CONCRETE: 0.95,
    TireSurface.GRAVEL: 0.6,
    TireSurface.DIRT: 0.55,
    TireSurface.GRASS: 0.4,
    TireSurface.SAND: 0.3,
    TireSurface.SNOW: 0.25,
    TireSurface.ICE: 0.15,
}


@dataclass
class TireState:
    """Current state of a tire."""

    # Slip values
    slip_ratio: float = 0.0       # Longitudinal slip (-1 to 1+)
    slip_angle: float = 0.0       # Lateral slip angle (radians)

    # Forces (in tire local space)
    longitudinal_force: float = 0.0  # Forward/backward force
    lateral_force: float = 0.0       # Side force
    aligning_moment: float = 0.0     # Self-aligning torque

    # Load
    normal_load: float = 0.0      # Vertical load on tire

    # Velocities
    wheel_velocity: float = 0.0   # Wheel surface velocity
    ground_velocity: float = 0.0  # Ground contact velocity

    # Temperature/wear (for advanced simulation)
    temperature: float = 25.0     # Celsius
    wear: float = 0.0             # 0 to 1 (worn out)

    # Contact
    contact_point: Vector3 = field(default_factory=Vector3.zero)
    is_grounded: bool = False


@dataclass
class TireForces:
    """Computed tire forces in world space."""

    longitudinal: float = 0.0  # Forward force
    lateral: float = 0.0       # Side force
    vertical: float = 0.0      # Normal force
    aligning: float = 0.0      # Aligning torque

    def to_vector(self) -> Vector3:
        """Convert to force vector."""
        return Vector3(self.lateral, self.vertical, self.longitudinal)


class TireModel(ABC):
    """
    Abstract base class for tire models.

    Provides interface for computing tire forces from slip values.
    """

    def __init__(
        self,
        friction: float = TIRE_FRICTION_COEFFICIENT,
        load_sensitivity: float = TIRE_LOAD_SENSITIVITY,
        rolling_resistance: float = TIRE_ROLLING_RESISTANCE,
    ):
        """
        Initialize tire model.

        Args:
            friction: Base friction coefficient.
            load_sensitivity: Load sensitivity factor.
            rolling_resistance: Rolling resistance coefficient.
        """
        self._friction = friction
        self._load_sensitivity = load_sensitivity
        self._rolling_resistance = rolling_resistance
        self._surface = TireSurface.ASPHALT_DRY
        self._state = TireState()

    @property
    def friction(self) -> float:
        """Base friction coefficient."""
        return self._friction

    @friction.setter
    def friction(self, value: float) -> None:
        """Set base friction."""
        if value < 0:
            raise ValueError("Friction must be non-negative")
        self._friction = value

    @property
    def surface(self) -> TireSurface:
        """Current surface type."""
        return self._surface

    @surface.setter
    def surface(self, value: TireSurface) -> None:
        """Set surface type."""
        self._surface = value

    @property
    def state(self) -> TireState:
        """Current tire state."""
        return self._state

    def get_effective_friction(self, load: float) -> float:
        """
        Get friction adjusted for load and surface.

        Args:
            load: Normal load on tire.

        Returns:
            Effective friction coefficient.
        """
        surface_mult = SURFACE_FRICTION.get(self._surface, 1.0)
        # Load sensitivity: friction decreases slightly with load
        load_factor = 1.0 - self._load_sensitivity * max(0, load)
        return self._friction * surface_mult * max(0.1, load_factor)

    def compute_slip_ratio(
        self,
        wheel_angular_velocity: float,
        wheel_radius: float,
        ground_velocity: float,
    ) -> float:
        """
        Compute longitudinal slip ratio.

        Args:
            wheel_angular_velocity: Angular velocity of wheel (rad/s).
            wheel_radius: Wheel radius (m).
            ground_velocity: Forward velocity at contact (m/s).

        Returns:
            Slip ratio (-1 to 1+, negative = braking).
        """
        wheel_velocity = wheel_angular_velocity * wheel_radius

        # Minimum velocity threshold to avoid division by zero
        MIN_VELOCITY = 0.1

        # Use the larger of wheel or ground velocity as reference
        # This is the standard SAE slip ratio definition
        reference_velocity = max(abs(wheel_velocity), abs(ground_velocity))

        if reference_velocity < MIN_VELOCITY:
            # Near standstill - use small fixed denominator
            slip = (wheel_velocity - ground_velocity) / MIN_VELOCITY
        else:
            # Standard slip ratio calculation
            # Positive slip = acceleration (wheel faster than ground)
            # Negative slip = braking (wheel slower than ground)
            slip = (wheel_velocity - ground_velocity) / reference_velocity

        # Clamp to reasonable range
        return max(-1.5, min(1.5, slip))

    def compute_slip_angle(
        self,
        velocity_x: float,
        velocity_y: float,
    ) -> float:
        """
        Compute lateral slip angle.

        Args:
            velocity_x: Forward velocity at contact.
            velocity_y: Lateral velocity at contact.

        Returns:
            Slip angle in radians.
        """
        # Avoid division by zero
        if abs(velocity_x) < 0.1:
            if abs(velocity_y) < 0.1:
                return 0.0
            return math.copysign(math.pi / 2, velocity_y)

        return math.atan2(velocity_y, abs(velocity_x))

    @abstractmethod
    def compute_longitudinal_force(
        self,
        slip_ratio: float,
        normal_load: float,
    ) -> float:
        """
        Compute longitudinal (drive/brake) force.

        Args:
            slip_ratio: Longitudinal slip ratio.
            normal_load: Vertical load on tire.

        Returns:
            Longitudinal force (positive = forward).
        """
        ...

    @abstractmethod
    def compute_lateral_force(
        self,
        slip_angle: float,
        normal_load: float,
        camber: float = 0.0,
    ) -> float:
        """
        Compute lateral (cornering) force.

        Args:
            slip_angle: Slip angle in radians.
            normal_load: Vertical load on tire.
            camber: Camber angle in radians (optional).

        Returns:
            Lateral force (positive = right).
        """
        ...

    def compute_rolling_resistance(self, normal_load: float) -> float:
        """
        Compute rolling resistance force.

        Args:
            normal_load: Vertical load on tire.

        Returns:
            Rolling resistance force (always opposes motion).
        """
        return self._rolling_resistance * normal_load

    def compute_aligning_moment(
        self,
        slip_angle: float,
        lateral_force: float,
        pneumatic_trail: float = 0.01,
    ) -> float:
        """
        Compute self-aligning torque.

        Args:
            slip_angle: Slip angle in radians.
            lateral_force: Computed lateral force.
            pneumatic_trail: Pneumatic trail length.

        Returns:
            Aligning moment.
        """
        # Simple model: moment = lateral_force * trail * reduction_factor
        reduction = math.cos(slip_angle) ** 2
        return lateral_force * pneumatic_trail * reduction

    def update(
        self,
        wheel_angular_velocity: float,
        wheel_radius: float,
        ground_velocity_forward: float,
        ground_velocity_lateral: float,
        normal_load: float,
        camber: float = 0.0,
    ) -> TireForces:
        """
        Update tire state and compute forces.

        Args:
            wheel_angular_velocity: Wheel angular velocity (rad/s).
            wheel_radius: Wheel radius (m).
            ground_velocity_forward: Forward velocity at contact.
            ground_velocity_lateral: Lateral velocity at contact.
            normal_load: Vertical load on tire.
            camber: Camber angle in radians.

        Returns:
            Computed tire forces.
        """
        # Compute slip values
        slip_ratio = self.compute_slip_ratio(
            wheel_angular_velocity,
            wheel_radius,
            ground_velocity_forward,
        )
        slip_angle = self.compute_slip_angle(
            ground_velocity_forward,
            ground_velocity_lateral,
        )

        # Update state
        self._state.slip_ratio = slip_ratio
        self._state.slip_angle = slip_angle
        self._state.normal_load = normal_load
        self._state.wheel_velocity = wheel_angular_velocity * wheel_radius
        self._state.ground_velocity = ground_velocity_forward
        self._state.is_grounded = normal_load > 0

        if normal_load <= 0:
            # Tire not in contact
            self._state.longitudinal_force = 0.0
            self._state.lateral_force = 0.0
            self._state.aligning_moment = 0.0
            return TireForces()

        # Compute forces
        fx = self.compute_longitudinal_force(slip_ratio, normal_load)
        fy = self.compute_lateral_force(slip_angle, normal_load, camber)

        # Combined slip reduction (friction circle)
        total_force = math.sqrt(fx ** 2 + fy ** 2)
        max_force = self.get_effective_friction(normal_load) * normal_load

        if total_force > max_force:
            scale = max_force / total_force
            fx *= scale
            fy *= scale

        # Rolling resistance
        fr = self.compute_rolling_resistance(normal_load)

        # Aligning moment
        mz = self.compute_aligning_moment(slip_angle, fy)

        # Update state
        self._state.longitudinal_force = fx - fr
        self._state.lateral_force = fy
        self._state.aligning_moment = mz

        return TireForces(
            longitudinal=fx - fr,
            lateral=fy,
            vertical=normal_load,
            aligning=mz,
        )


class PacejkaTire(TireModel):
    """
    Pacejka Magic Formula tire model.

    Implements the well-known "Magic Formula" for accurate tire behavior.
    F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))

    Where:
    - B = stiffness factor
    - C = shape factor
    - D = peak value
    - E = curvature factor
    """

    def __init__(
        self,
        # Longitudinal coefficients
        b_long: float = PACEJKA_B_LONGITUDINAL,
        c_long: float = PACEJKA_C_LONGITUDINAL,
        d_long: float = PACEJKA_D_LONGITUDINAL,
        e_long: float = PACEJKA_E_LONGITUDINAL,
        # Lateral coefficients
        b_lat: float = PACEJKA_B_LATERAL,
        c_lat: float = PACEJKA_C_LATERAL,
        d_lat: float = PACEJKA_D_LATERAL,
        e_lat: float = PACEJKA_E_LATERAL,
        # Camber coefficients
        camber_stiffness: float = 0.5,
        # Base parameters
        friction: float = TIRE_FRICTION_COEFFICIENT,
        load_sensitivity: float = TIRE_LOAD_SENSITIVITY,
        rolling_resistance: float = TIRE_ROLLING_RESISTANCE,
    ):
        """
        Initialize Pacejka tire model.

        Args:
            b_long: Longitudinal stiffness factor.
            c_long: Longitudinal shape factor.
            d_long: Longitudinal peak factor.
            e_long: Longitudinal curvature factor.
            b_lat: Lateral stiffness factor.
            c_lat: Lateral shape factor.
            d_lat: Lateral peak factor.
            e_lat: Lateral curvature factor.
            camber_stiffness: Camber effect coefficient.
            friction: Base friction coefficient.
            load_sensitivity: Load sensitivity.
            rolling_resistance: Rolling resistance coefficient.
        """
        super().__init__(friction, load_sensitivity, rolling_resistance)

        # Longitudinal coefficients
        self._b_long = b_long
        self._c_long = c_long
        self._d_long = d_long
        self._e_long = e_long

        # Lateral coefficients
        self._b_lat = b_lat
        self._c_lat = c_lat
        self._d_lat = d_lat
        self._e_lat = e_lat

        # Camber
        self._camber_stiffness = camber_stiffness

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

        Args:
            x: Input (slip ratio or slip angle).
            b: Stiffness factor.
            c: Shape factor.
            d: Peak value.
            e: Curvature factor.

        Returns:
            Normalized force coefficient.
        """
        bx = b * x
        return d * math.sin(c * math.atan(bx - e * (bx - math.atan(bx))))

    def compute_longitudinal_force(
        self,
        slip_ratio: float,
        normal_load: float,
    ) -> float:
        """
        Compute longitudinal force using Magic Formula.

        Args:
            slip_ratio: Longitudinal slip ratio.
            normal_load: Vertical load on tire.

        Returns:
            Longitudinal force.
        """
        # D coefficient scales with load and friction
        effective_friction = self.get_effective_friction(normal_load)
        d = self._d_long * effective_friction * normal_load

        # Evaluate Magic Formula
        force_coefficient = self._magic_formula(
            slip_ratio,
            self._b_long,
            self._c_long,
            1.0,  # Normalized
            self._e_long,
        )

        return force_coefficient * d

    def compute_lateral_force(
        self,
        slip_angle: float,
        normal_load: float,
        camber: float = 0.0,
    ) -> float:
        """
        Compute lateral force using Magic Formula.

        Args:
            slip_angle: Slip angle in radians.
            normal_load: Vertical load on tire.
            camber: Camber angle in radians.

        Returns:
            Lateral force.
        """
        # D coefficient scales with load and friction
        effective_friction = self.get_effective_friction(normal_load)
        d = self._d_lat * effective_friction * normal_load

        # Evaluate Magic Formula
        force_coefficient = self._magic_formula(
            slip_angle,
            self._b_lat,
            self._c_lat,
            1.0,  # Normalized
            self._e_lat,
        )

        lateral_force = force_coefficient * d

        # Add camber thrust
        if camber != 0:
            camber_thrust = self._camber_stiffness * camber * normal_load
            lateral_force += camber_thrust

        return lateral_force

    def get_peak_slip_ratio(self) -> float:
        """Get slip ratio at peak longitudinal force."""
        # Approximate: peak occurs around atan(1/B) / B
        return math.atan(1.0 / self._b_long) / self._b_long

    def get_peak_slip_angle(self) -> float:
        """Get slip angle at peak lateral force."""
        return math.atan(1.0 / self._b_lat) / self._b_lat


class LinearTire(TireModel):
    """
    Simple linear tire model.

    Provides computationally cheap approximation suitable for
    arcade-style or lower-fidelity simulations.
    """

    def __init__(
        self,
        longitudinal_stiffness: float = 10000.0,  # N per unit slip
        lateral_stiffness: float = 8000.0,        # N per radian
        saturation_slip_ratio: float = 0.15,
        saturation_slip_angle: float = 0.2,       # radians (~11 degrees)
        friction: float = TIRE_FRICTION_COEFFICIENT,
        load_sensitivity: float = TIRE_LOAD_SENSITIVITY,
        rolling_resistance: float = TIRE_ROLLING_RESISTANCE,
    ):
        """
        Initialize linear tire model.

        Args:
            longitudinal_stiffness: Stiffness for longitudinal force.
            lateral_stiffness: Cornering stiffness.
            saturation_slip_ratio: Slip ratio at which force saturates.
            saturation_slip_angle: Slip angle at which force saturates.
            friction: Base friction coefficient.
            load_sensitivity: Load sensitivity.
            rolling_resistance: Rolling resistance.
        """
        super().__init__(friction, load_sensitivity, rolling_resistance)

        self._long_stiffness = longitudinal_stiffness
        self._lat_stiffness = lateral_stiffness
        self._sat_slip_ratio = saturation_slip_ratio
        self._sat_slip_angle = saturation_slip_angle

    @property
    def longitudinal_stiffness(self) -> float:
        """Longitudinal stiffness."""
        return self._long_stiffness

    @longitudinal_stiffness.setter
    def longitudinal_stiffness(self, value: float) -> None:
        """Set longitudinal stiffness."""
        if value < 0:
            raise ValueError("Stiffness must be non-negative")
        self._long_stiffness = value

    @property
    def lateral_stiffness(self) -> float:
        """Lateral (cornering) stiffness."""
        return self._lat_stiffness

    @lateral_stiffness.setter
    def lateral_stiffness(self, value: float) -> None:
        """Set lateral stiffness."""
        if value < 0:
            raise ValueError("Stiffness must be non-negative")
        self._lat_stiffness = value

    def _saturate(self, x: float, saturation: float) -> float:
        """
        Apply saturation to keep force within friction limits.

        Uses smooth saturation: x / sqrt(1 + (x/sat)^2)
        """
        if saturation <= 0:
            return 0.0
        normalized = x / saturation
        return x / math.sqrt(1 + normalized ** 2)

    def compute_longitudinal_force(
        self,
        slip_ratio: float,
        normal_load: float,
    ) -> float:
        """
        Compute longitudinal force (linear with saturation).

        Args:
            slip_ratio: Longitudinal slip ratio.
            normal_load: Vertical load on tire.

        Returns:
            Longitudinal force.
        """
        # Linear force
        raw_force = self._long_stiffness * slip_ratio

        # Apply saturation
        saturated = self._saturate(raw_force, self._long_stiffness * self._sat_slip_ratio)

        # Scale by friction
        effective_friction = self.get_effective_friction(normal_load)
        max_force = effective_friction * normal_load

        # Clamp to friction limit
        return max(-max_force, min(max_force, saturated))

    def compute_lateral_force(
        self,
        slip_angle: float,
        normal_load: float,
        camber: float = 0.0,
    ) -> float:
        """
        Compute lateral force (linear with saturation).

        Args:
            slip_angle: Slip angle in radians.
            normal_load: Vertical load on tire.
            camber: Camber angle in radians (ignored in simple model).

        Returns:
            Lateral force.
        """
        # Linear force
        raw_force = self._lat_stiffness * slip_angle

        # Apply saturation
        saturated = self._saturate(raw_force, self._lat_stiffness * self._sat_slip_angle)

        # Scale by friction
        effective_friction = self.get_effective_friction(normal_load)
        max_force = effective_friction * normal_load

        # Clamp to friction limit
        return max(-max_force, min(max_force, saturated))


class BrushTire(TireModel):
    """
    Brush model tire.

    Physics-based model treating the contact patch as deformable bristles.
    More accurate than linear, simpler than Pacejka.
    """

    def __init__(
        self,
        contact_length: float = 0.15,  # Contact patch length (m)
        tread_stiffness: float = 200000.0,  # N/m^2 per bristle
        friction: float = TIRE_FRICTION_COEFFICIENT,
        load_sensitivity: float = TIRE_LOAD_SENSITIVITY,
        rolling_resistance: float = TIRE_ROLLING_RESISTANCE,
    ):
        """
        Initialize brush tire model.

        Args:
            contact_length: Length of contact patch.
            tread_stiffness: Stiffness of tread elements.
            friction: Base friction coefficient.
            load_sensitivity: Load sensitivity.
            rolling_resistance: Rolling resistance.
        """
        super().__init__(friction, load_sensitivity, rolling_resistance)

        self._contact_length = contact_length
        self._tread_stiffness = tread_stiffness

    def compute_longitudinal_force(
        self,
        slip_ratio: float,
        normal_load: float,
    ) -> float:
        """
        Compute longitudinal force using brush model.

        Args:
            slip_ratio: Longitudinal slip ratio.
            normal_load: Vertical load on tire.

        Returns:
            Longitudinal force.
        """
        effective_friction = self.get_effective_friction(normal_load)

        # Critical slip where sliding begins
        if normal_load <= 0:
            return 0.0

        sigma_critical = (3 * effective_friction * normal_load) / (
            self._tread_stiffness * self._contact_length ** 2
        )

        sigma = abs(slip_ratio)

        if sigma < sigma_critical:
            # Adhesion region (parabolic)
            force = (
                self._tread_stiffness * self._contact_length ** 2 * sigma
                - (self._tread_stiffness ** 2 * self._contact_length ** 4 * sigma ** 2)
                / (3 * effective_friction * normal_load)
            )
        else:
            # Full sliding
            force = effective_friction * normal_load

        return math.copysign(force, slip_ratio)

    def compute_lateral_force(
        self,
        slip_angle: float,
        normal_load: float,
        camber: float = 0.0,
    ) -> float:
        """
        Compute lateral force using brush model.

        Args:
            slip_angle: Slip angle in radians.
            normal_load: Vertical load on tire.
            camber: Camber angle (ignored in basic brush model).

        Returns:
            Lateral force.
        """
        effective_friction = self.get_effective_friction(normal_load)

        if normal_load <= 0:
            return 0.0

        # Use tan(slip_angle) for small angles
        tan_alpha = math.tan(slip_angle)

        # Critical angle
        alpha_critical = math.atan(
            (3 * effective_friction * normal_load)
            / (self._tread_stiffness * self._contact_length ** 2)
        )

        if abs(slip_angle) < alpha_critical:
            # Adhesion region
            force = (
                self._tread_stiffness * self._contact_length ** 2 * tan_alpha
                - (self._tread_stiffness ** 2 * self._contact_length ** 4 * tan_alpha ** 2)
                / (3 * effective_friction * normal_load)
                + (self._tread_stiffness ** 3 * self._contact_length ** 6 * tan_alpha ** 3)
                / (27 * (effective_friction * normal_load) ** 2)
            )
        else:
            # Full sliding
            force = effective_friction * normal_load

        return math.copysign(force, slip_angle)


def create_tire_model(
    model_type: str = "pacejka",
    **kwargs,
) -> TireModel:
    """
    Factory function to create tire models.

    Args:
        model_type: Type of model ("pacejka", "linear", "brush").
        **kwargs: Arguments passed to model constructor.

    Returns:
        Tire model instance.

    Raises:
        ValueError: If unknown model type.
    """
    models = {
        "pacejka": PacejkaTire,
        "linear": LinearTire,
        "brush": BrushTire,
    }

    model_class = models.get(model_type.lower())
    if model_class is None:
        raise ValueError(f"Unknown tire model type: {model_type}")

    return model_class(**kwargs)
