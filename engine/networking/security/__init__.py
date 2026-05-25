"""
Security and Anti-Cheat Systems for the Game Engine.

This module provides comprehensive security features for networked games:

- Authority Validation: Control who can modify game state
- Input Validation: Detect speed hacks, teleports, and impossible inputs
- Rate Limiting: Prevent abuse and ensure fair resource usage
- Anomaly Detection: Identify cheating patterns like aimbot and wallhacks
- Response Management: Escalating responses from warnings to bans

Example usage:
    from engine.networking.security import (
        AuthorityValidator,
        InputValidator,
        RateLimiter,
        AnomalyDetector,
        ResponseManager
    )

    # Create validators
    authority = AuthorityValidator()
    input_validator = InputValidator()
    rate_limiter = RateLimiter()
    anomaly_detector = AnomalyDetector()
    response_manager = ResponseManager()

    # Check authority for an entity write
    if authority.validate_write(entity, "position", caller):
        # Validate the input
        result = input_validator.validate_movement(player_id, new_position)
        if result.result == ValidationResult.VALID:
            # Apply the change
            entity.position = new_position
"""

# Authority validation
from engine.networking.security.authority_validator import (
    Authority,
    AuthorityError,
    AuthorityValidator,
    Caller,
    Entity,
    EntityAuthority,
    FieldAuthority,
)

# Input validation
from engine.networking.security.input_validator import (
    InputBounds,
    InputValidator,
    PlayerState,
    ValidationReport,
    ValidationResult,
    Vector3,
)

# Rate limiting
from engine.networking.security.rate_limiter import (
    AdaptiveRateLimiter,
    DEFAULT_LIMITS,
    RateLimitConfig,
    RateLimitResult,
    RateLimiter,
    RateLimitStats,
    TokenBucket,
)

# Anomaly detection
from engine.networking.security.anomaly_detector import (
    AnomalyDetector,
    AnomalyReport,
    AnomalySeverity,
    AnomalyThresholds,
    AnomalyType,
    DamageEvent,
    KillEvent,
    MovementEvent,
    PlayerStats,
    ReactionEvent,
    ShotEvent,
)

# Response management
from engine.networking.security.response import (
    BanRecord,
    CheatResponse,
    DEFAULT_ESCALATION_RULES,
    EscalationRule,
    PlayerViolationHistory,
    ResponseManager,
    ResponseSeverity,
    ViolationRecord,
    generate_hwid_hash,
    generate_secure_token,
)

# Security configuration
from engine.networking.security.config import (
    INPUT_VALIDATION,
    RATE_LIMIT_DEFAULTS,
    ANOMALY_DETECTION,
    RESPONSE_CONFIG,
    ADAPTIVE_RATE_LIMIT,
    VALIDATION_LIMITS,
)

__all__ = [
    # Authority validation
    "Authority",
    "AuthorityError",
    "AuthorityValidator",
    "Caller",
    "Entity",
    "EntityAuthority",
    "FieldAuthority",
    # Input validation
    "InputBounds",
    "InputValidator",
    "PlayerState",
    "ValidationReport",
    "ValidationResult",
    "Vector3",
    # Rate limiting
    "AdaptiveRateLimiter",
    "DEFAULT_LIMITS",
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimiter",
    "RateLimitStats",
    "TokenBucket",
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyReport",
    "AnomalySeverity",
    "AnomalyThresholds",
    "AnomalyType",
    "DamageEvent",
    "KillEvent",
    "MovementEvent",
    "PlayerStats",
    "ReactionEvent",
    "ShotEvent",
    # Response management
    "BanRecord",
    "CheatResponse",
    "DEFAULT_ESCALATION_RULES",
    "EscalationRule",
    "PlayerViolationHistory",
    "ResponseManager",
    "ResponseSeverity",
    "ViolationRecord",
    "generate_hwid_hash",
    "generate_secure_token",
    # Security configuration
    "INPUT_VALIDATION",
    "RATE_LIMIT_DEFAULTS",
    "ANOMALY_DETECTION",
    "RESPONSE_CONFIG",
    "ADAPTIVE_RATE_LIMIT",
    "VALIDATION_LIMITS",
]
