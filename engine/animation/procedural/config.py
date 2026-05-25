"""
Configuration Constants for Procedural Animation.

Centralizes all magic numbers and tunable parameters for procedural animation.
This allows easy adjustment without modifying implementation code.

Usage:
    from engine.animation.procedural.config import ProceduralConfig

    spring = SpringBone(
        bone_index=5,
        stiffness=ProceduralConfig.SPRING_DEFAULT_STIFFNESS,
        damping=ProceduralConfig.SPRING_DEFAULT_DAMPING
    )
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SpringPhysicsConfig:
    """Configuration for spring bone physics."""

    # Default spring parameters
    DEFAULT_STIFFNESS: float = 50.0
    DEFAULT_DAMPING: float = 0.3
    DEFAULT_MASS: float = 1.0

    # Physics limits
    MIN_STIFFNESS: float = 0.0
    MAX_STIFFNESS: float = 1000.0
    MIN_DAMPING: float = 0.0
    MAX_DAMPING: float = 1.0
    MIN_MASS: float = 0.001

    # Numerical stability
    MAX_DT: float = 1.0 / 30.0  # Maximum timestep for stability (0.033s)
    MIN_DT: float = 1e-6  # Minimum meaningful timestep
    EPSILON: float = 1e-10  # Near-zero threshold for divisions

    # Default gravity
    DEFAULT_GRAVITY: Tuple[float, float, float] = (0.0, -9.81, 0.0)

    # Constraint solver
    DEFAULT_CONSTRAINT_ITERATIONS: int = 3
    MIN_CONSTRAINT_ITERATIONS: int = 1
    MAX_CONSTRAINT_ITERATIONS: int = 20


@dataclass(frozen=True)
class WindForceConfig:
    """Configuration for wind forces."""

    DEFAULT_STRENGTH: float = 1.0
    DEFAULT_TURBULENCE: float = 0.0
    DEFAULT_FREQUENCY: float = 1.0
    MIN_TURBULENCE: float = 0.0
    MAX_TURBULENCE: float = 1.0


@dataclass(frozen=True)
class LookAtConfig:
    """Configuration for look-at controller."""

    # Speed and smoothing
    DEFAULT_ROTATION_SPEED: float = 5.0  # Radians per second
    DEFAULT_BLEND_SPEED: float = 10.0  # Blend weight change per second

    # Angle limits (radians)
    DEFAULT_HEAD_YAW_LIMIT: float = math.radians(80.0)
    DEFAULT_HEAD_PITCH_LIMIT: float = math.radians(40.0)
    DEFAULT_NECK_YAW_LIMIT: float = math.radians(30.0)
    DEFAULT_NECK_PITCH_LIMIT: float = math.radians(20.0)
    DEFAULT_EYE_YAW_LIMIT: float = math.radians(35.0)
    DEFAULT_EYE_PITCH_LIMIT: float = math.radians(25.0)

    # Distribution
    DEFAULT_NECK_CONTRIBUTION: float = 0.3
    DEFAULT_EYE_LEAD_TIME: float = 0.1

    # Eye movement multiplier (eyes move faster than head)
    EYE_SPEED_MULTIPLIER: float = 2.0


@dataclass(frozen=True)
class SaccadeConfig:
    """Configuration for eye saccades."""

    DEFAULT_MIN_INTERVAL: float = 0.1  # Minimum time between saccades (seconds)
    DEFAULT_MAX_INTERVAL: float = 3.0  # Maximum time between saccades (seconds)
    DEFAULT_MAX_OFFSET: float = 0.05  # Maximum saccade offset (radians, ~3 degrees)
    DEFAULT_SPEED: float = 500.0  # Saccade speed (degrees/second)


@dataclass(frozen=True)
class TwistConfig:
    """Configuration for twist bone distribution."""

    DEFAULT_WEIGHT: float = 1.0

    # Default twist axes
    ARM_TWIST_AXIS: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # X-axis
    LEG_TWIST_AXIS: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # Y-axis


@dataclass(frozen=True)
class RagdollConfig:
    """Configuration for ragdoll physics."""

    # Physics parameters
    DEFAULT_LINEAR_DAMPING: float = 0.1
    DEFAULT_ANGULAR_DAMPING: float = 0.1
    DEFAULT_BLEND_DURATION: float = 0.3

    # Joint motor defaults
    DEFAULT_MOTOR_MAX_TORQUE: float = 100.0
    DEFAULT_MOTOR_STIFFNESS: float = 1000.0
    DEFAULT_MOTOR_DAMPING: float = 100.0

    # Joint limit defaults (radians)
    DEFAULT_TWIST_LIMIT: float = math.radians(45.0)
    DEFAULT_SWING_LIMIT: float = math.radians(45.0)
    DEFAULT_CONTACT_DISTANCE: float = math.radians(5.0)

    # Humanoid body mass distribution (kg)
    MASS_HIPS: float = 15.0
    MASS_SPINE: float = 10.0
    MASS_CHEST: float = 10.0
    MASS_HEAD: float = 5.0
    MASS_UPPER_ARM: float = 3.0
    MASS_LOWER_ARM: float = 2.0
    MASS_THIGH: float = 8.0
    MASS_CALF: float = 5.0

    # Humanoid body dimensions (meters)
    RADIUS_HIPS: float = 0.15
    RADIUS_SPINE: float = 0.12
    RADIUS_CHEST: float = 0.15
    RADIUS_HEAD: float = 0.12
    RADIUS_UPPER_ARM: float = 0.05
    RADIUS_LOWER_ARM: float = 0.04
    RADIUS_THIGH: float = 0.08
    RADIUS_CALF: float = 0.06

    HEIGHT_HIPS: float = 0.2
    HEIGHT_SPINE: float = 0.15
    HEIGHT_CHEST: float = 0.2
    HEIGHT_UPPER_ARM: float = 0.25
    HEIGHT_LOWER_ARM: float = 0.22
    HEIGHT_THIGH: float = 0.4
    HEIGHT_CALF: float = 0.35


@dataclass(frozen=True)
class LocomotionConfig:
    """Configuration for procedural locomotion."""

    # Gait defaults
    DEFAULT_STEP_HEIGHT: float = 0.15
    DEFAULT_STEP_LENGTH: float = 0.6
    DEFAULT_CYCLE_DURATION: float = 0.5

    # Foot trajectory
    DEFAULT_HEEL_STRIKE_ANGLE: float = math.radians(-15.0)
    DEFAULT_TOE_OFF_ANGLE: float = math.radians(25.0)
    DEFAULT_STANCE_RATIO: float = 0.6
    DEFAULT_ARC_EXPONENT: float = 2.0

    # Body dynamics
    DEFAULT_BOB_AMPLITUDE: float = 0.03
    DEFAULT_BOB_FREQUENCY: float = 2.0
    DEFAULT_SWAY_AMPLITUDE: float = 0.02
    DEFAULT_LEAN_ANGLE: float = math.radians(5.0)
    DEFAULT_SPEED_LEAN_FACTOR: float = 0.05
    DEFAULT_HIP_ROTATION_AMPLITUDE: float = math.radians(10.0)
    DEFAULT_SPINE_TWIST_AMPLITUDE: float = math.radians(5.0)

    # Speed thresholds
    DEFAULT_MIN_SPEED: float = 0.0
    DEFAULT_MAX_SPEED: float = 5.0
    DEFAULT_WALK_TO_RUN_SPEED: float = 2.0

    # Run gait adjustments
    RUN_STEP_HEIGHT: float = 0.2
    RUN_STEP_LENGTH: float = 1.0
    RUN_CYCLE_DURATION: float = 0.35
    RUN_STANCE_RATIO: float = 0.4
    RUN_BOB_AMPLITUDE: float = 0.05
    RUN_LEAN_ANGLE: float = math.radians(10.0)

    # Arm swing
    DEFAULT_ARM_SWING_AMPLITUDE: float = math.radians(30.0)


@dataclass(frozen=True)
class BreathingConfig:
    """Configuration for breathing animation."""

    # Breathing rates (Hz = breaths per second)
    RATE_RELAXED: float = 0.2  # 12 breaths/minute
    RATE_NORMAL: float = 0.25  # 15 breaths/minute
    RATE_ACTIVE: float = 0.4  # 24 breaths/minute
    RATE_HEAVY: float = 0.6  # 36 breaths/minute
    RATE_EXHAUSTED: float = 0.8  # 48 breaths/minute

    # Motion amplitudes
    DEFAULT_CHEST_EXPANSION: float = 0.025
    DEFAULT_SHOULDER_RISE: float = 0.008
    DEFAULT_SPINE_CURVE: float = math.radians(3.0)
    DEFAULT_NECK_EXTENSION: float = math.radians(1.0)

    # Timing
    DEFAULT_INHALE_RATIO: float = 0.4
    DEFAULT_HOLD_RATIO: float = 0.05


@dataclass(frozen=True)
class SecondaryMotionConfig:
    """Configuration for secondary motion effects."""

    # Delayed motion
    DEFAULT_DELAY: float = 0.1

    # Oscillating motion
    DEFAULT_OSCILLATION_FREQUENCY: float = 1.0

    # Noise motion
    DEFAULT_NOISE_FREQUENCY: float = 1.0
    DEFAULT_NOISE_OCTAVES: int = 2
    DEFAULT_NOISE_PERSISTENCE: float = 0.5

    # Impulse response
    DEFAULT_IMPULSE_STIFFNESS: float = 50.0
    DEFAULT_IMPULSE_DAMPING: float = 0.7
    DEFAULT_IMPULSE_THRESHOLD: float = 0.5
    DEFAULT_IMPULSE_MAX_RESPONSE: float = 0.1


class ProceduralConfig:
    """
    Master configuration class for all procedural animation constants.

    Usage:
        from engine.animation.procedural.config import ProceduralConfig

        stiffness = ProceduralConfig.Spring.DEFAULT_STIFFNESS
        look_speed = ProceduralConfig.LookAt.DEFAULT_ROTATION_SPEED
    """

    Spring = SpringPhysicsConfig()
    Wind = WindForceConfig()
    LookAt = LookAtConfig()
    Saccade = SaccadeConfig()
    Twist = TwistConfig()
    Ragdoll = RagdollConfig()
    Locomotion = LocomotionConfig()
    Breathing = BreathingConfig()
    SecondaryMotion = SecondaryMotionConfig()

    # Global numerical constants
    EPSILON: float = 1e-10
    MAX_PHYSICS_DT: float = 1.0 / 30.0


# Convenience constants for backward compatibility
SPRING_DEFAULT_STIFFNESS = ProceduralConfig.Spring.DEFAULT_STIFFNESS
SPRING_DEFAULT_DAMPING = ProceduralConfig.Spring.DEFAULT_DAMPING
SPRING_DEFAULT_GRAVITY = ProceduralConfig.Spring.DEFAULT_GRAVITY
MAX_PHYSICS_DT = ProceduralConfig.MAX_PHYSICS_DT
EPSILON = ProceduralConfig.EPSILON
