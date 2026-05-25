"""
Vehicle simulation configuration constants.

This module defines default values for vehicle physics simulation including
wheel parameters, suspension settings, engine characteristics, and physics
tuning values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Final


# =============================================================================
# Wheel Configuration
# =============================================================================

DEFAULT_WHEEL_RADIUS: Final[float] = 0.35  # meters
DEFAULT_WHEEL_WIDTH: Final[float] = 0.225  # meters
DEFAULT_WHEEL_MASS: Final[float] = 15.0  # kg
DEFAULT_WHEEL_INERTIA: Final[float] = 1.5  # kg*m^2


# =============================================================================
# Suspension Configuration
# =============================================================================

DEFAULT_SUSPENSION_REST: Final[float] = 0.5  # meters (rest length)
DEFAULT_SPRING_STRENGTH: Final[float] = 35000.0  # N/m
DEFAULT_DAMPER_COMPRESSION: Final[float] = 4500.0  # N*s/m
DEFAULT_DAMPER_REBOUND: Final[float] = 4000.0  # N*s/m
DEFAULT_SUSPENSION_TRAVEL: Final[float] = 0.2  # meters (total travel)
DEFAULT_ANTI_ROLL_STRENGTH: Final[float] = 5000.0  # N*m/rad


# =============================================================================
# Steering Configuration
# =============================================================================

MAX_STEER_ANGLE: Final[float] = 35.0  # degrees
ACKERMANN_RATIO: Final[float] = 0.8  # 0 = parallel, 1 = full Ackermann
STEERING_RATE: Final[float] = 2.5  # degrees per second per input unit
STEERING_RETURN_RATE: Final[float] = 4.0  # auto-center rate


# =============================================================================
# Engine Configuration
# =============================================================================

ENGINE_IDLE_RPM: Final[int] = 1000
ENGINE_MAX_RPM: Final[int] = 7000
ENGINE_REDLINE_RPM: Final[int] = 6500
ENGINE_INERTIA: Final[float] = 0.15  # kg*m^2
ENGINE_FRICTION: Final[float] = 0.02  # N*m / RPM
DEFAULT_MAX_TORQUE: Final[float] = 450.0  # N*m
DEFAULT_MAX_POWER: Final[float] = 250000.0  # Watts (250 kW)


# =============================================================================
# Transmission Configuration
# =============================================================================

DEFAULT_GEAR_RATIOS: Final[tuple] = (
    -3.5,   # Reverse
    0.0,    # Neutral
    3.8,    # 1st
    2.4,    # 2nd
    1.65,   # 3rd
    1.2,    # 4th
    0.95,   # 5th
    0.75,   # 6th
)

DEFAULT_FINAL_DRIVE: Final[float] = 3.7
SHIFT_TIME: Final[float] = 0.2  # seconds
CLUTCH_ENGAGEMENT_RATE: Final[float] = 5.0  # 1/s


# =============================================================================
# Brake Configuration
# =============================================================================

DEFAULT_BRAKE_TORQUE: Final[float] = 3000.0  # N*m (total)
BRAKE_BIAS_FRONT: Final[float] = 0.6  # 60% front, 40% rear
HANDBRAKE_TORQUE: Final[float] = 2000.0  # N*m (rear only)
ABS_SLIP_TARGET: Final[float] = 0.12  # optimal slip ratio for braking


# =============================================================================
# Tire Configuration
# =============================================================================

# Pacejka Magic Formula defaults (based on typical passenger car tires)
PACEJKA_B_LONGITUDINAL: Final[float] = 10.0  # Stiffness factor
PACEJKA_C_LONGITUDINAL: Final[float] = 1.9   # Shape factor
PACEJKA_D_LONGITUDINAL: Final[float] = 1.0   # Peak factor (normalized)
PACEJKA_E_LONGITUDINAL: Final[float] = 0.97  # Curvature factor

PACEJKA_B_LATERAL: Final[float] = 8.0        # Stiffness factor
PACEJKA_C_LATERAL: Final[float] = 1.4        # Shape factor
PACEJKA_D_LATERAL: Final[float] = 1.0        # Peak factor (normalized)
PACEJKA_E_LATERAL: Final[float] = -0.5       # Curvature factor (must be in range [-1, 1])

TIRE_FRICTION_COEFFICIENT: Final[float] = 1.0  # Base friction
TIRE_LOAD_SENSITIVITY: Final[float] = 0.0001  # Load sensitivity
TIRE_ROLLING_RESISTANCE: Final[float] = 0.015  # Rolling resistance coefficient


# =============================================================================
# Differential Configuration
# =============================================================================

LSD_PRELOAD: Final[float] = 100.0  # N*m
LSD_POWER_RATIO: Final[float] = 0.6  # Power locking ratio
LSD_COAST_RATIO: Final[float] = 0.4  # Coast locking ratio


# =============================================================================
# Aerodynamics Configuration
# =============================================================================

DEFAULT_DRAG_COEFFICIENT: Final[float] = 0.35
DEFAULT_FRONTAL_AREA: Final[float] = 2.2  # m^2
DEFAULT_LIFT_COEFFICIENT: Final[float] = 0.1
AIR_DENSITY: Final[float] = 1.225  # kg/m^3 at sea level


# =============================================================================
# Tracked Vehicle Configuration
# =============================================================================

TRACK_FRICTION: Final[float] = 0.8
TRACK_ROLLING_RESISTANCE: Final[float] = 0.05
TRACK_WIDTH: Final[float] = 0.5  # meters
TRACK_LENGTH: Final[float] = 4.0  # meters (contact patch)


# =============================================================================
# Hover Vehicle Configuration
# =============================================================================

DEFAULT_HOVER_HEIGHT: Final[float] = 0.5  # meters
HOVER_SPRING_CONSTANT: Final[float] = 50000.0  # N/m
HOVER_DAMPING: Final[float] = 3000.0  # N*s/m
SKIRT_DRAG_COEFFICIENT: Final[float] = 0.1
HOVER_CUSHION_AREA_RATIO: Final[float] = 0.9  # 90% of footprint
HOVER_AIR_DRAG_COEFFICIENT: Final[float] = 0.4
HOVER_CUSHION_DRAG_COEFFICIENT: Final[float] = 0.01
HOVER_LIFT_MARGIN: Final[float] = 1.5  # 50% lift margin
HOVER_MAX_PRESSURE_RATIO: Final[float] = 3.0  # Max pressure multiplier
HOVER_YAW_DAMPING: Final[float] = 1000.0  # Yaw damping coefficient


# =============================================================================
# Aircraft Configuration
# =============================================================================

AIRCRAFT_LIFT_SLOPE: Final[float] = 2.0 * 3.14159  # rad^-1 (2*pi for thin airfoil)
AIRCRAFT_STALL_ANGLE: Final[float] = 15.0  # degrees
AIRCRAFT_PARASITE_DRAG: Final[float] = 0.025  # CD0
AIRCRAFT_OSWALD_EFFICIENCY: Final[float] = 0.8  # e
AIRCRAFT_SCALE_HEIGHT: Final[float] = 8500.0  # Atmospheric scale height (m)
AIRCRAFT_CONTROL_EFFECTIVENESS: Final[float] = 0.1  # Control surface effectiveness
AIRCRAFT_TAIL_MOMENT_ARM: Final[float] = 4.0  # Distance from CG to tail (m)
AIRCRAFT_PITCH_STABILITY: Final[float] = 0.01  # Pitch stability coefficient
AIRCRAFT_YAW_STABILITY: Final[float] = 0.02  # Yaw stability coefficient
AIRCRAFT_ROLL_DAMPING: Final[float] = 0.1  # Roll damping coefficient
AIRCRAFT_GROUND_FRICTION: Final[float] = 0.05  # Ground rolling friction


# =============================================================================
# Watercraft Configuration
# =============================================================================

WATER_DENSITY: Final[float] = 1025.0  # kg/m^3 (seawater)
HULL_DRAG_COEFFICIENT: Final[float] = 0.3
WAVE_FREQUENCY: Final[float] = 0.5  # Hz
WAVE_AMPLITUDE: Final[float] = 0.5  # meters
PROPELLER_EFFICIENCY: Final[float] = 0.7


# =============================================================================
# Physics Simulation
# =============================================================================

PHYSICS_SUBSTEPS: Final[int] = 4
VELOCITY_SLEEP_THRESHOLD: Final[float] = 0.1  # m/s
ANGULAR_SLEEP_THRESHOLD: Final[float] = 0.1  # rad/s
GRAVITY: Final[float] = 9.81  # m/s^2


# =============================================================================
# Helper Data Classes
# =============================================================================

@dataclass
class WheelConfig:
    """Configuration for a single wheel."""

    radius: float = DEFAULT_WHEEL_RADIUS
    width: float = DEFAULT_WHEEL_WIDTH
    mass: float = DEFAULT_WHEEL_MASS
    inertia: float = DEFAULT_WHEEL_INERTIA

    def validate(self) -> bool:
        """Validate wheel configuration."""
        return (
            self.radius > 0 and
            self.width > 0 and
            self.mass > 0 and
            self.inertia > 0
        )


@dataclass
class SuspensionConfig:
    """Configuration for suspension system."""

    rest_length: float = DEFAULT_SUSPENSION_REST
    spring_strength: float = DEFAULT_SPRING_STRENGTH
    damper_compression: float = DEFAULT_DAMPER_COMPRESSION
    damper_rebound: float = DEFAULT_DAMPER_REBOUND
    travel: float = DEFAULT_SUSPENSION_TRAVEL
    anti_roll_strength: float = DEFAULT_ANTI_ROLL_STRENGTH

    @property
    def min_length(self) -> float:
        """Minimum suspension length (fully compressed)."""
        return self.rest_length - self.travel / 2

    @property
    def max_length(self) -> float:
        """Maximum suspension length (fully extended)."""
        return self.rest_length + self.travel / 2

    def validate(self) -> bool:
        """Validate suspension configuration."""
        return (
            self.rest_length > 0 and
            self.spring_strength > 0 and
            self.damper_compression >= 0 and
            self.damper_rebound >= 0 and
            self.travel > 0 and
            self.min_length > 0
        )


@dataclass
class EngineConfig:
    """Configuration for engine."""

    idle_rpm: int = ENGINE_IDLE_RPM
    max_rpm: int = ENGINE_MAX_RPM
    redline_rpm: int = ENGINE_REDLINE_RPM
    max_torque: float = DEFAULT_MAX_TORQUE
    inertia: float = ENGINE_INERTIA
    friction: float = ENGINE_FRICTION

    def validate(self) -> bool:
        """Validate engine configuration."""
        return (
            0 < self.idle_rpm < self.redline_rpm < self.max_rpm and
            self.max_torque > 0 and
            self.inertia > 0
        )


@dataclass
class TransmissionConfig:
    """Configuration for transmission."""

    gear_ratios: tuple = DEFAULT_GEAR_RATIOS
    final_drive: float = DEFAULT_FINAL_DRIVE
    shift_time: float = SHIFT_TIME

    def validate(self) -> bool:
        """Validate transmission configuration."""
        return (
            len(self.gear_ratios) >= 3 and  # At least R, N, 1st
            self.final_drive > 0 and
            self.shift_time >= 0
        )


@dataclass
class TireConfig:
    """Configuration for tire model."""

    # Pacejka coefficients
    b_long: float = PACEJKA_B_LONGITUDINAL
    c_long: float = PACEJKA_C_LONGITUDINAL
    d_long: float = PACEJKA_D_LONGITUDINAL
    e_long: float = PACEJKA_E_LONGITUDINAL

    b_lat: float = PACEJKA_B_LATERAL
    c_lat: float = PACEJKA_C_LATERAL
    d_lat: float = PACEJKA_D_LATERAL
    e_lat: float = PACEJKA_E_LATERAL

    friction: float = TIRE_FRICTION_COEFFICIENT
    load_sensitivity: float = TIRE_LOAD_SENSITIVITY
    rolling_resistance: float = TIRE_ROLLING_RESISTANCE

    def validate(self) -> bool:
        """Validate tire configuration."""
        return (
            self.friction > 0 and
            self.load_sensitivity >= 0 and
            self.rolling_resistance >= 0
        )


class VehiclePreset(Enum):
    """Preset vehicle configurations."""

    SEDAN = auto()
    SPORTS_CAR = auto()
    SUV = auto()
    TRUCK = auto()
    MOTORCYCLE = auto()
    FORMULA_CAR = auto()
    TANK = auto()
    HOVERCRAFT = auto()
    LIGHT_AIRCRAFT = auto()
    HELICOPTER = auto()
    SPEEDBOAT = auto()
    CARGO_SHIP = auto()


# Preset configurations mapped to parameter sets
VEHICLE_PRESETS: Dict[VehiclePreset, dict] = {
    VehiclePreset.SEDAN: {
        "mass": 1500.0,
        "max_power": 150000.0,
        "max_torque": 300.0,
        "drag_coefficient": 0.32,
        "frontal_area": 2.2,
    },
    VehiclePreset.SPORTS_CAR: {
        "mass": 1400.0,
        "max_power": 350000.0,
        "max_torque": 500.0,
        "drag_coefficient": 0.28,
        "frontal_area": 1.9,
    },
    VehiclePreset.SUV: {
        "mass": 2200.0,
        "max_power": 200000.0,
        "max_torque": 400.0,
        "drag_coefficient": 0.40,
        "frontal_area": 3.0,
    },
    VehiclePreset.TRUCK: {
        "mass": 3500.0,
        "max_power": 250000.0,
        "max_torque": 700.0,
        "drag_coefficient": 0.45,
        "frontal_area": 3.5,
    },
    VehiclePreset.TANK: {
        "mass": 40000.0,
        "max_power": 1100000.0,
        "max_torque": 4000.0,
        "drag_coefficient": 0.8,
        "frontal_area": 8.0,
    },
}
