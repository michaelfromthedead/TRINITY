"""
Constants for Gameplay Components.

This module centralizes all magic numbers and configuration values used
across the gameplay component system to improve maintainability and clarity.
"""

from typing import Tuple


# =============================================================================
# HEALTH COMPONENT CONSTANTS
# =============================================================================

class HealthConstants:
    """Constants for the HealthComponent."""

    # Default values
    DEFAULT_MAX_HEALTH: float = 100.0
    DEFAULT_REGEN_RATE: float = 0.0

    # Resistance system
    MAX_RESISTANCE_CAP: float = 0.99  # Maximum damage reduction (99%)
    MIN_RESISTANCE: float = -1.0  # Maximum damage amplification

    # Revival system
    DEFAULT_REVIVE_HEALTH_PERCENTAGE: float = 0.5  # 50% of max health
    MIN_REVIVE_HEALTH_PERCENTAGE: float = 0.01  # Minimum 1% health on revive

    # History tracking
    DEFAULT_HISTORY_LIMIT: int = 10


# =============================================================================
# MOVEMENT COMPONENT CONSTANTS
# =============================================================================

class MovementConstants:
    """Constants for the MovementComponent."""

    # Default movement values
    DEFAULT_MAX_SPEED: float = 5.0
    DEFAULT_MAX_JUMPS: int = 1

    # Jump timing (in seconds)
    DEFAULT_COYOTE_TIME: float = 0.15  # Grace period after leaving ground
    DEFAULT_JUMP_BUFFER_TIME: float = 0.1  # Pre-landing jump buffer

    # Cancel jump reduction factor
    CANCEL_JUMP_VELOCITY_FACTOR: float = 0.5

    # Speed thresholds for state detection
    IS_MOVING_SPEED_THRESHOLD: float = 0.1  # Minimum speed to be "moving"
    IDLE_SPEED_THRESHOLD: float = 0.1  # Maximum speed to be "idle"

    # Input thresholds
    HAS_INPUT_THRESHOLD: float = 0.01  # Minimum input magnitude
    FACING_DIRECTION_THRESHOLD: float = 0.01  # Minimum direction magnitude

    # Velocity calculation thresholds
    VELOCITY_DIFF_THRESHOLD: float = 0.001  # Minimum diff for acceleration

    # Default mode settings
    class WalkingMode:
        MAX_SPEED: float = 4.0
        ACCELERATION: float = 15.0

    class RunningMode:
        MAX_SPEED: float = 7.0
        ACCELERATION: float = 20.0

    class SprintingMode:
        MAX_SPEED: float = 10.0
        ACCELERATION: float = 25.0
        TURN_RATE: float = 180.0

    class CrouchingMode:
        MAX_SPEED: float = 2.0
        ACCELERATION: float = 10.0
        HEIGHT_SCALE: float = 0.5

    class SwimmingMode:
        MAX_SPEED: float = 3.0
        ACCELERATION: float = 8.0
        GRAVITY_SCALE: float = 0.1
        JUMP_VELOCITY: float = 4.0

    class FlyingMode:
        MAX_SPEED: float = 8.0
        ACCELERATION: float = 12.0
        GRAVITY_SCALE: float = 0.0
        AIR_CONTROL: float = 1.0

    class FallingMode:
        MAX_SPEED: float = 50.0
        ACCELERATION: float = 0.0
        AIR_CONTROL: float = 0.2

    class ClimbingMode:
        MAX_SPEED: float = 2.0
        ACCELERATION: float = 10.0
        GRAVITY_SCALE: float = 0.0

    class SlidingMode:
        MAX_SPEED: float = 12.0
        ACCELERATION: float = 5.0
        DECELERATION: float = 3.0


# =============================================================================
# TRANSFORM COMPONENT CONSTANTS
# =============================================================================

class TransformConstants:
    """Constants for the TransformComponent."""

    # Look-at calculation thresholds
    LOOK_AT_DIRECTION_EPSILON: float = 0.0001  # Minimum direction magnitude
    LOOK_AT_RIGHT_EPSILON: float = 0.0001  # Minimum right vector magnitude


# =============================================================================
# TEAM COMPONENT CONSTANTS
# =============================================================================

class TeamConstants:
    """Constants for the TeamComponent and related classes."""

    # Default colors (RGB tuples)
    DEFAULT_NEUTRAL_COLOR: Tuple[int, int, int] = (128, 128, 128)

    # Default team ID for "no team"
    NO_TEAM_ID: int = 0

    # Team membership
    UNLIMITED_MEMBERS: int = -1


# =============================================================================
# STATS COMPONENT CONSTANTS
# =============================================================================

class StatsConstants:
    """Constants for the StatsComponent."""

    # Modifier system
    DEFAULT_MODIFIER_PRIORITY: int = 0
    PERMANENT_DURATION: float = -1.0
    DEFAULT_STACKS: int = 1
    DEFAULT_MAX_STACKS: int = 1


# =============================================================================
# STATE MACHINE CONSTANTS
# =============================================================================

class StateMachineConstants:
    """Constants for state machine functionality."""

    # History tracking
    DEFAULT_HISTORY_LIMIT: int = 100


__all__ = [
    "HealthConstants",
    "MovementConstants",
    "TransformConstants",
    "TeamConstants",
    "StatsConstants",
    "StateMachineConstants",
]
