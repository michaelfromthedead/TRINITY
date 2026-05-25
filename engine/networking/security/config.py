"""
Security Configuration Constants for the Game Engine.

This module centralizes ALL security-related thresholds, limits, and constants
to enable easy tuning and prevent magic numbers scattered throughout the codebase.

CRITICAL: All security thresholds MUST be defined here, not inline in code.
"""

from dataclasses import dataclass
from typing import Dict


# =============================================================================
# INPUT VALIDATION CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class InputValidationConfig:
    """Configuration for input validation bounds."""
    # Movement validation
    MAX_SPEED: float = 600.0  # Units per second
    MAX_ROTATION_RATE: float = 720.0  # Degrees per second
    MAX_ACTION_RATE: float = 20.0  # Actions per second
    MAX_TELEPORT_DISTANCE: float = 100.0  # Maximum single-tick position change
    TOLERANCE_MULTIPLIER: float = 1.5  # Lag compensation tolerance

    # World bounds
    WORLD_MIN_X: float = -100000.0
    WORLD_MIN_Y: float = -10000.0
    WORLD_MIN_Z: float = -100000.0
    WORLD_MAX_X: float = 100000.0
    WORLD_MAX_Y: float = 50000.0
    WORLD_MAX_Z: float = 100000.0

    # Sequence validation
    SEQUENCE_WINDOW: int = 100  # Accept sequences within this range

    # Time validation
    MIN_TIME_DELTA: float = 0.001  # Minimum time delta to prevent division issues


INPUT_VALIDATION = InputValidationConfig()


# =============================================================================
# RATE LIMITING CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class RateLimitDefaults:
    """Default rate limit configurations."""
    # Input rate limiting
    INPUT_REQUESTS_PER_SECOND: float = 60.0
    INPUT_BURST_SIZE: int = 10

    # RPC rate limiting
    RPC_REQUESTS_PER_SECOND: float = 10.0
    RPC_BURST_SIZE: int = 5

    # Chat rate limiting
    CHAT_REQUESTS_PER_SECOND: float = 1.0
    CHAT_BURST_SIZE: int = 5

    # Spawn rate limiting
    SPAWN_REQUESTS_PER_SECOND: float = 2.0
    SPAWN_BURST_SIZE: int = 3

    # Item use rate limiting
    ITEM_USE_REQUESTS_PER_SECOND: float = 10.0
    ITEM_USE_BURST_SIZE: int = 5

    # Voice rate limiting
    VOICE_REQUESTS_PER_SECOND: float = 50.0
    VOICE_BURST_SIZE: int = 20

    # Warning threshold
    WARNING_THRESHOLD: float = 0.2


RATE_LIMIT_DEFAULTS = RateLimitDefaults()


# =============================================================================
# ANOMALY DETECTION CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class AnomalyDetectionConfig:
    """Configuration for anomaly detection thresholds."""
    # Accuracy thresholds
    ACCURACY_THRESHOLD: float = 0.95  # 95% accuracy is suspicious
    ACCURACY_SAMPLE_SIZE: int = 50  # Minimum shots to analyze
    HEADSHOT_RATE_THRESHOLD: float = 0.80  # 80% headshot rate is suspicious

    # Reaction time thresholds
    MIN_REACTION_TIME_MS: float = 100.0  # Sub-100ms reactions are suspicious
    REACTION_SAMPLE_SIZE: int = 10  # Minimum reactions to analyze
    IMPOSSIBLE_REACTION_RATIO: float = 0.5  # >50% fast reactions is suspicious

    # Movement thresholds
    SPEED_VARIANCE_THRESHOLD: float = 0.1  # Low variance = constant speed hack
    SPEED_HACK_MIN_SPEED: float = 100.0  # Minimum speed to consider for hack
    MIN_MOVEMENT_SAMPLES: int = 10  # Minimum movement events to analyze

    # Kill/damage thresholds
    CONSECUTIVE_KILLS_THRESHOLD: int = 20  # Kills without deaths
    DAMAGE_MULTIPLIER_THRESHOLD: float = 1.5  # Damage above expected

    # Recoil and wall hack thresholds
    RECOIL_VARIANCE_THRESHOLD: float = 0.05  # Low recoil variance is suspicious
    WALL_HIT_RATE_THRESHOLD: float = 0.3  # Rate of hitting through walls

    # Confidence calculations
    WALLHACK_CONFIDENCE_DIVISOR: float = 0.5  # For confidence normalization
    DAMAGE_CONFIDENCE_DIVISOR: float = 2.0  # For confidence normalization

    # Rolling window
    RECENT_WINDOW_SECONDS: float = 300.0  # 5 minutes


ANOMALY_DETECTION = AnomalyDetectionConfig()


# =============================================================================
# RESPONSE/BAN MANAGEMENT CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class ResponseConfig:
    """Configuration for anti-cheat response management."""
    # Escalation thresholds
    WARNINGS_BEFORE_KICK: int = 3
    KICKS_BEFORE_TEMP_BAN: int = 2
    TEMP_BANS_BEFORE_LONGER_BAN: int = 2
    TEMP_BANS_BEFORE_PERMANENT: int = 4

    # Ban durations (seconds)
    FIRST_TEMP_BAN_DURATION: float = 3600.0  # 1 hour
    SECOND_TEMP_BAN_DURATION: float = 86400.0  # 24 hours

    # Cooldowns (seconds)
    WARNING_COOLDOWN: float = 60.0
    KICK_COOLDOWN: float = 300.0  # 5 minutes
    BAN_COOLDOWN: float = 0.0  # No cooldown for bans

    # Risk score calculation
    RISK_SCORE_DECAY_HOURS: float = 1.0  # Hours for age decay
    RISK_SCORE_MIN_AGE_FACTOR: float = 0.1  # Minimum age factor
    RISK_SCORE_NORMALIZATION: float = 5.0  # Normalize to 0-1 range

    # Severity weights for risk calculation
    SEVERITY_WEIGHT_LOW: float = 0.1
    SEVERITY_WEIGHT_MEDIUM: float = 0.3
    SEVERITY_WEIGHT_HIGH: float = 0.6
    SEVERITY_WEIGHT_CRITICAL: float = 1.0


RESPONSE_CONFIG = ResponseConfig()


# =============================================================================
# ADAPTIVE RATE LIMITER CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class AdaptiveRateLimitConfig:
    """Configuration for adaptive rate limiting."""
    LOAD_THRESHOLD: float = 0.8  # Server load threshold to trigger reduction
    REDUCTION_FACTOR: float = 0.5  # Factor to reduce limits by when overloaded
    MIN_LOAD: float = 0.0
    MAX_LOAD: float = 1.0


ADAPTIVE_RATE_LIMIT = AdaptiveRateLimitConfig()


# =============================================================================
# THREAD SAFETY CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class ThreadSafetyConfig:
    """Configuration for thread safety."""
    # Lock acquisition timeout (for deadlock prevention)
    LOCK_TIMEOUT_SECONDS: float = 5.0


THREAD_SAFETY = ThreadSafetyConfig()


# =============================================================================
# VALIDATION LIMITS (Prevent integer overflow)
# =============================================================================

@dataclass(frozen=True)
class ValidationLimits:
    """Limits to prevent integer overflow and other edge cases."""
    MAX_SEQUENCE_NUMBER: int = 2**31 - 1  # Max 32-bit signed int
    MAX_VIOLATION_COUNT: int = 10000  # Prevent unbounded growth
    MAX_PLAYER_STATE_ENTRIES: int = 100000  # Prevent memory exhaustion
    MAX_TOKENS_PER_REQUEST: int = 100  # Prevent token bucket abuse
    MAX_BAN_DURATION_SECONDS: float = 31536000.0  # 1 year max
    MAX_EVENTS_PER_PLAYER: int = 10000  # Limit event history size


VALIDATION_LIMITS = ValidationLimits()
