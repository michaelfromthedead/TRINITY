"""
Vehicle Simulation Module.

This module provides comprehensive vehicle physics simulation including:
- Wheeled vehicles with realistic tire and suspension models
- Tracked vehicles (tanks, excavators) with differential steering
- Hover vehicles (hovercraft) with air cushion physics
- Aircraft with aerodynamic forces and control surfaces
- Watercraft with buoyancy and hydrodynamics

Example Usage:
    from engine.simulation.vehicles import (
        VehicleSystem,
        VehicleType,
        WheeledVehicle,
        TrackedVehicle,
        HoverVehicle,
        Aircraft,
        Watercraft,
    )

    # Create vehicle system
    vehicle_system = VehicleSystem()

    # Create a wheeled vehicle
    car = WheeledVehicle(mass=1500.0, wheelbase=2.8)
    vehicle_system.register_vehicle(car)

    # Update simulation
    vehicle_system.update(dt=0.016)

"""

# Configuration
from .config import (
    # Wheel
    DEFAULT_WHEEL_RADIUS,
    DEFAULT_WHEEL_WIDTH,
    DEFAULT_WHEEL_MASS,
    DEFAULT_WHEEL_INERTIA,
    # Suspension
    DEFAULT_SUSPENSION_REST,
    DEFAULT_SPRING_STRENGTH,
    DEFAULT_DAMPER_COMPRESSION,
    DEFAULT_DAMPER_REBOUND,
    DEFAULT_SUSPENSION_TRAVEL,
    DEFAULT_ANTI_ROLL_STRENGTH,
    # Steering
    MAX_STEER_ANGLE,
    ACKERMANN_RATIO,
    STEERING_RATE,
    STEERING_RETURN_RATE,
    # Engine
    ENGINE_IDLE_RPM,
    ENGINE_MAX_RPM,
    ENGINE_REDLINE_RPM,
    ENGINE_INERTIA,
    ENGINE_FRICTION,
    DEFAULT_MAX_TORQUE,
    DEFAULT_MAX_POWER,
    # Transmission
    DEFAULT_GEAR_RATIOS,
    DEFAULT_FINAL_DRIVE,
    SHIFT_TIME,
    # Brakes
    DEFAULT_BRAKE_TORQUE,
    BRAKE_BIAS_FRONT,
    HANDBRAKE_TORQUE,
    # Tires
    PACEJKA_B_LONGITUDINAL,
    PACEJKA_C_LONGITUDINAL,
    PACEJKA_D_LONGITUDINAL,
    PACEJKA_E_LONGITUDINAL,
    PACEJKA_B_LATERAL,
    PACEJKA_C_LATERAL,
    PACEJKA_D_LATERAL,
    PACEJKA_E_LATERAL,
    TIRE_FRICTION_COEFFICIENT,
    # Aerodynamics
    DEFAULT_DRAG_COEFFICIENT,
    DEFAULT_FRONTAL_AREA,
    AIR_DENSITY,
    # Environment
    WATER_DENSITY,
    GRAVITY,
    # Config classes
    WheelConfig,
    SuspensionConfig,
    EngineConfig,
    TransmissionConfig,
    TireConfig,
    VehiclePreset,
    VEHICLE_PRESETS,
)

# Vehicle system
from .vehicle_system import (
    VehicleType,
    VehicleState,
    Vector3,
    Transform,
    VehicleGroup,
    CollisionInfo,
    VehicleSystem,
    VehicleBase,
    generate_vehicle_id,
)

# Suspension
from .suspension import (
    SuspensionType,
    SuspensionState,
    SuspensionGeometry,
    Suspension,
    AntiRollBar,
    SuspensionSystem,
)

# Tire models
from .tire_model import (
    TireSurface,
    SURFACE_FRICTION,
    TireState,
    TireForces,
    TireModel,
    PacejkaTire,
    LinearTire,
    BrushTire,
    create_tire_model,
)

# Drivetrain
from .drivetrain import (
    DiffType,
    DrivetrainLayout,
    EngineState,
    Engine,
    TransmissionState,
    Transmission,
    ClutchState,
    Clutch,
    Differential,
    Drivetrain,
)

# Wheeled vehicles
from .wheeled_vehicle import (
    WheelPosition,
    WheelState,
    Wheel,
    WheeledVehicle,
)

# Tracked vehicles
from .tracked_vehicle import (
    TrackType,
    RoadWheel,
    TrackState,
    Track,
    TrackedVehicle,
)

# Hover vehicles
from .hover_vehicle import (
    HoverMode,
    LiftFan,
    ThrustVector,
    SkirtState,
    HoverVehicle,
)

# Aircraft
from .aircraft import (
    AircraftType,
    FlightPhase,
    AerodynamicSurface,
    ControlSurface,
    AircraftEngine,
    Aircraft,
)

# Watercraft
from .watercraft import (
    HullType,
    WatercraftType,
    BuoyancySamplePoint,
    Propeller,
    Rudder,
    WaveState,
    Watercraft,
)

__all__ = [
    # Configuration
    "DEFAULT_WHEEL_RADIUS",
    "DEFAULT_WHEEL_WIDTH",
    "DEFAULT_WHEEL_MASS",
    "DEFAULT_WHEEL_INERTIA",
    "DEFAULT_SUSPENSION_REST",
    "DEFAULT_SPRING_STRENGTH",
    "DEFAULT_DAMPER_COMPRESSION",
    "DEFAULT_DAMPER_REBOUND",
    "DEFAULT_SUSPENSION_TRAVEL",
    "DEFAULT_ANTI_ROLL_STRENGTH",
    "MAX_STEER_ANGLE",
    "ACKERMANN_RATIO",
    "STEERING_RATE",
    "STEERING_RETURN_RATE",
    "ENGINE_IDLE_RPM",
    "ENGINE_MAX_RPM",
    "ENGINE_REDLINE_RPM",
    "ENGINE_INERTIA",
    "ENGINE_FRICTION",
    "DEFAULT_MAX_TORQUE",
    "DEFAULT_MAX_POWER",
    "DEFAULT_GEAR_RATIOS",
    "DEFAULT_FINAL_DRIVE",
    "SHIFT_TIME",
    "DEFAULT_BRAKE_TORQUE",
    "BRAKE_BIAS_FRONT",
    "HANDBRAKE_TORQUE",
    "PACEJKA_B_LONGITUDINAL",
    "PACEJKA_C_LONGITUDINAL",
    "PACEJKA_D_LONGITUDINAL",
    "PACEJKA_E_LONGITUDINAL",
    "PACEJKA_B_LATERAL",
    "PACEJKA_C_LATERAL",
    "PACEJKA_D_LATERAL",
    "PACEJKA_E_LATERAL",
    "TIRE_FRICTION_COEFFICIENT",
    "DEFAULT_DRAG_COEFFICIENT",
    "DEFAULT_FRONTAL_AREA",
    "AIR_DENSITY",
    "WATER_DENSITY",
    "GRAVITY",
    "WheelConfig",
    "SuspensionConfig",
    "EngineConfig",
    "TransmissionConfig",
    "TireConfig",
    "VehiclePreset",
    "VEHICLE_PRESETS",
    # Vehicle system
    "VehicleType",
    "VehicleState",
    "Vector3",
    "Transform",
    "VehicleGroup",
    "CollisionInfo",
    "VehicleSystem",
    "VehicleBase",
    "generate_vehicle_id",
    # Suspension
    "SuspensionType",
    "SuspensionState",
    "SuspensionGeometry",
    "Suspension",
    "AntiRollBar",
    "SuspensionSystem",
    # Tire models
    "TireSurface",
    "SURFACE_FRICTION",
    "TireState",
    "TireForces",
    "TireModel",
    "PacejkaTire",
    "LinearTire",
    "BrushTire",
    "create_tire_model",
    # Drivetrain
    "DiffType",
    "DrivetrainLayout",
    "EngineState",
    "Engine",
    "TransmissionState",
    "Transmission",
    "ClutchState",
    "Clutch",
    "Differential",
    "Drivetrain",
    # Wheeled vehicles
    "WheelPosition",
    "WheelState",
    "Wheel",
    "WheeledVehicle",
    # Tracked vehicles
    "TrackType",
    "RoadWheel",
    "TrackState",
    "Track",
    "TrackedVehicle",
    # Hover vehicles
    "HoverMode",
    "LiftFan",
    "ThrustVector",
    "SkirtState",
    "HoverVehicle",
    # Aircraft
    "AircraftType",
    "FlightPhase",
    "AerodynamicSurface",
    "ControlSurface",
    "AircraftEngine",
    "Aircraft",
    # Watercraft
    "HullType",
    "WatercraftType",
    "BuoyancySamplePoint",
    "Propeller",
    "Rudder",
    "WaveState",
    "Watercraft",
]
