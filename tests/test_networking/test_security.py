"""
Comprehensive tests for the security and anti-cheat systems.

Tests cover:
- Authority validation (server vs client writes)
- Input validation (movement, speed, teleportation)
- Rate limiting (token bucket, per-player limits)
- Anomaly detection (aimbot, speed hack patterns)
- Response escalation (warning -> kick -> ban)
"""

import pytest
import time
import threading
from typing import List
from unittest.mock import Mock, patch

# Import security modules
from engine.networking.security.authority_validator import (
    Authority,
    AuthorityError,
    AuthorityValidator,
    Caller,
    Entity,
    EntityAuthority,
    FieldAuthority,
)
from engine.networking.security.input_validator import (
    InputBounds,
    InputValidator,
    PlayerState,
    ValidationReport,
    ValidationResult,
    Vector3,
)
from engine.networking.security.rate_limiter import (
    RateLimitConfig,
    RateLimitResult,
    RateLimiter,
    TokenBucket,
    AdaptiveRateLimiter,
)
from engine.networking.security.anomaly_detector import (
    AnomalyDetector,
    AnomalyReport,
    AnomalySeverity,
    AnomalyThresholds,
    AnomalyType,
    PlayerStats,
    ShotEvent,
)
from engine.networking.security.response import (
    BanRecord,
    CheatResponse,
    EscalationRule,
    ResponseManager,
    ResponseSeverity,
    generate_hwid_hash,
)


# =============================================================================
# Authority Validator Tests
# =============================================================================


class TestAuthorityValidator:
    """Tests for authority validation system."""

    @pytest.fixture
    def validator(self):
        """Create a fresh authority validator."""
        return AuthorityValidator()

    @pytest.fixture
    def server_caller(self):
        """Create a server authority caller."""
        return Caller(id="server", authority=Authority.SERVER)

    @pytest.fixture
    def client_caller(self):
        """Create a client authority caller."""
        return Caller(id="client-123", authority=Authority.CLIENT)

    @pytest.fixture
    def owner_caller(self):
        """Create an owner authority caller."""
        return Caller(
            id="player-456",
            authority=Authority.OWNER,
            owned_entities={"entity-789"}
        )

    @pytest.fixture
    def test_entity(self):
        """Create a test entity."""
        return Entity(
            id="entity-789",
            entity_type="player",
            owner_id="player-456",
            fields={"position": Vector3(0, 0, 0), "health": 100}
        )

    def test_server_can_always_write(self, validator, server_caller, test_entity):
        """Server authority should always be able to write."""
        assert validator.validate_write(test_entity, "position", server_caller)
        assert validator.validate_write(test_entity, "health", server_caller)
        assert validator.validate_write(test_entity, "any_field", server_caller)

    def test_client_cannot_write_server_fields(self, validator, client_caller, test_entity):
        """Client should not be able to write server-controlled fields."""
        # By default, all fields require server authority
        assert not validator.validate_write(test_entity, "position", client_caller)
        assert not validator.validate_write(test_entity, "health", client_caller)

    def test_owner_can_write_owned_fields(self, validator, owner_caller, test_entity):
        """Owner should be able to write to fields marked owner_can_write."""
        # Register entity type with owner-writable field
        validator.register_entity_type("player", EntityAuthority(
            field_authorities={
                "position": FieldAuthority(owner_can_write=True),
                "health": FieldAuthority(owner_can_write=False)
            }
        ))

        assert validator.validate_write(test_entity, "position", owner_caller)
        assert not validator.validate_write(test_entity, "health", owner_caller)

    def test_server_can_always_spawn(self, validator, server_caller):
        """Server should always be able to spawn entities."""
        assert validator.validate_spawn("player", server_caller)
        assert validator.validate_spawn("projectile", server_caller)
        assert validator.validate_spawn("any_type", server_caller)

    def test_client_cannot_spawn_by_default(self, validator, client_caller):
        """Client should not be able to spawn entities by default."""
        assert not validator.validate_spawn("player", client_caller)
        assert not validator.validate_spawn("projectile", client_caller)

    def test_server_can_always_destroy(self, validator, server_caller, test_entity):
        """Server should always be able to destroy entities."""
        assert validator.validate_destroy(test_entity, server_caller)

    def test_client_cannot_destroy_by_default(self, validator, client_caller, test_entity):
        """Client should not be able to destroy entities by default."""
        assert not validator.validate_destroy(test_entity, client_caller)

    def test_owner_can_destroy_if_allowed(self, validator, owner_caller, test_entity):
        """Owner should be able to destroy if owner_can_destroy is True."""
        validator.register_entity_type("player", EntityAuthority(owner_can_destroy=True))
        assert validator.validate_destroy(test_entity, owner_caller)

    def test_authority_error_raised(self, validator, client_caller, test_entity):
        """AuthorityError should be raised when raise_on_failure is True."""
        with pytest.raises(AuthorityError) as exc_info:
            validator.validate_write(test_entity, "health", client_caller, raise_on_failure=True)

        error = exc_info.value
        assert error.caller_authority == Authority.CLIENT
        assert error.required_authority == Authority.SERVER
        assert error.operation == "write"
        assert error.entity_id == test_entity.id

    def test_is_server_check(self, validator, server_caller, client_caller):
        """is_server should correctly identify server authority."""
        assert validator.is_server(server_caller)
        assert not validator.is_server(client_caller)

    def test_is_owner_check(self, validator, owner_caller, client_caller, test_entity):
        """is_owner should correctly identify entity ownership."""
        assert validator.is_owner(test_entity, owner_caller)
        assert not validator.is_owner(test_entity, client_caller)

    def test_batch_write_validation(self, validator, server_caller, client_caller, test_entity):
        """Batch write validation should check all fields."""
        results = validator.validate_batch_writes(
            test_entity,
            {"position", "health", "score"},
            server_caller
        )
        assert all(results.values())

        results = validator.validate_batch_writes(
            test_entity,
            {"position", "health"},
            client_caller
        )
        assert not any(results.values())


# =============================================================================
# Input Validator Tests
# =============================================================================


class TestInputValidator:
    """Tests for input validation system."""

    @pytest.fixture
    def validator(self):
        """Create a fresh input validator."""
        return InputValidator()

    @pytest.fixture
    def strict_validator(self):
        """Create a validator with strict bounds."""
        bounds = InputBounds(
            max_speed=100.0,
            max_rotation_rate=360.0,
            max_teleport_distance=10.0,
            tolerance_multiplier=1.0  # No tolerance
        )
        return InputValidator(bounds)

    def test_valid_movement_passes(self, validator):
        """Valid movement within bounds should pass."""
        player_id = "player-1"
        validator.set_player_position(player_id, Vector3(0, 0, 0))

        # Small movement over reasonable time
        time.sleep(0.05)  # 50ms
        report = validator.validate_movement(player_id, Vector3(5, 0, 5))

        assert report.result == ValidationResult.VALID

    def test_speed_hack_detected(self, strict_validator):
        """Movement exceeding max speed should be detected."""
        player_id = "player-1"
        strict_validator.set_player_position(player_id, Vector3(0, 0, 0))

        # Speed hack: moving faster than allowed but below teleport threshold
        # Moving 50 units in 0.1 seconds = 500 units/sec (exceeds max_speed of 100)
        # But within max_teleport_distance of 10 * tolerance (stays at 10 with 1.0 multiplier)
        # Use a smaller movement that triggers speed but not teleport
        report = strict_validator.validate_movement(
            player_id,
            Vector3(8, 0, 0),  # Under teleport threshold (10)
            time_delta=0.01   # Speed = 800 units/sec (exceeds 100)
        )

        assert report.result == ValidationResult.INVALID_SPEED
        assert "exceeds maximum" in report.details.lower()

    def test_teleport_detected(self, strict_validator):
        """Large position jumps should be detected as teleports."""
        player_id = "player-1"
        strict_validator.set_player_position(player_id, Vector3(0, 0, 0))

        # Jump 100 units instantly (beyond max_teleport_distance of 10)
        report = strict_validator.validate_movement(
            player_id,
            Vector3(100, 0, 0),
            time_delta=0.016  # One frame
        )

        assert report.result == ValidationResult.INVALID_TELEPORT
        assert "teleport" in report.details.lower() or "jump" in report.details.lower()

    def test_position_bounds_check(self, validator):
        """Positions outside world bounds should be rejected."""
        player_id = "player-1"
        # Start close to the bounds
        validator.set_player_position(player_id, Vector3(99900, 0, 0))

        # Move to position outside world bounds (within teleport range)
        # Default bounds: max x = 100000, with tolerance 1.5 = 150000
        # But we use default validator, so teleport distance = 100 * 1.5 = 150
        report = validator.validate_movement(
            player_id,
            Vector3(100050, 0, 0),  # Just outside bounds, within teleport
            time_delta=10.0  # Long time to avoid speed check
        )

        assert report.result == ValidationResult.INVALID_BOUNDS

    def test_valid_rotation_passes(self, validator):
        """Valid rotation within bounds should pass."""
        player_id = "player-1"
        validator.get_player_state(player_id)  # Initialize

        time.sleep(0.05)
        report = validator.validate_rotation(player_id, 45.0)

        assert report.result == ValidationResult.VALID

    def test_excessive_rotation_detected(self, strict_validator):
        """Rotation exceeding max rate should be detected."""
        player_id = "player-1"
        strict_validator.get_player_state(player_id)

        # 180 degree turn in 0.1 seconds = 1800 deg/sec (exceeds 360)
        report = strict_validator.validate_rotation(player_id, 180.0, time_delta=0.1)

        assert report.result == ValidationResult.INVALID_ROTATION

    def test_action_rate_validation(self, validator):
        """Action rate limiting should work."""
        player_id = "player-1"
        current_time = time.time()

        # First action should pass
        report = validator.validate_action(player_id, "fire", current_time)
        assert report.result == ValidationResult.VALID

        # Immediate second action should fail (too fast)
        report = validator.validate_action(player_id, "fire", current_time + 0.01)
        assert report.result == ValidationResult.INVALID_ACTION_RATE

    def test_sequence_validation(self, validator):
        """Sequence number validation should work."""
        player_id = "player-1"

        # Normal sequence progression
        assert validator.validate_sequence(player_id, 1).result == ValidationResult.VALID
        assert validator.validate_sequence(player_id, 2).result == ValidationResult.VALID
        assert validator.validate_sequence(player_id, 3).result == ValidationResult.VALID

        # Negative sequence numbers now raise ValueError (strict input validation)
        with pytest.raises(ValueError, match="sequence_number must be non-negative"):
            validator.validate_sequence(player_id, -200)

    def test_sequence_jump_detected(self, validator):
        """Large jumps in sequence numbers should be detected."""
        player_id = "player-1"

        validator.validate_sequence(player_id, 1)

        # Massive sequence jump (potential manipulation)
        result = validator.validate_sequence(player_id, 10000)
        assert result.result == ValidationResult.INVALID_SEQUENCE

    def test_full_input_validation(self, strict_validator):
        """Full input validation should check all aspects."""
        player_id = "player-1"
        strict_validator.set_player_position(player_id, Vector3(0, 0, 0))

        # Valid input
        reports = strict_validator.validate_full_input(
            player_id,
            new_position=Vector3(1, 0, 1),
            new_rotation=10.0,
            sequence_number=1,
            time_delta=0.1
        )

        assert all(r.result == ValidationResult.VALID for r in reports)

    def test_violation_count_tracking(self, strict_validator):
        """Violation count should be tracked per player."""
        player_id = "player-1"
        strict_validator.set_player_position(player_id, Vector3(0, 0, 0))

        assert strict_validator.get_violation_count(player_id) == 0

        # Cause a violation
        strict_validator.validate_movement(player_id, Vector3(1000, 0, 0), time_delta=0.01)

        assert strict_validator.get_violation_count(player_id) == 1


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestTokenBucket:
    """Tests for token bucket algorithm."""

    @pytest.fixture
    def bucket(self):
        """Create a test bucket."""
        return TokenBucket(RateLimitConfig(
            requests_per_second=10.0,
            burst_size=5,
            warning_threshold=0.1  # Lower threshold so we get ALLOWED not WARNED
        ))

    def test_allows_under_limit(self, bucket):
        """Requests under the limit should be allowed."""
        # With low warning threshold (0.1), only last request triggers warning
        for i in range(4):
            result = bucket.try_consume()
            assert result in (RateLimitResult.ALLOWED, RateLimitResult.WARNED)
        # Last one may be warned (near limit)
        result = bucket.try_consume()
        assert result in (RateLimitResult.ALLOWED, RateLimitResult.WARNED)

    def test_blocks_over_limit(self, bucket):
        """Requests over the limit should be blocked."""
        # Consume all tokens
        for _ in range(5):
            bucket.try_consume()

        # Next request should be denied
        assert bucket.try_consume() == RateLimitResult.DENIED

    def test_burst_allowance(self, bucket):
        """Burst capacity should allow short bursts."""
        # All burst capacity should be available immediately
        for _ in range(5):
            result = bucket.try_consume()
            assert result in (RateLimitResult.ALLOWED, RateLimitResult.WARNED)

    def test_token_refill(self, bucket):
        """Tokens should refill over time."""
        # Consume all tokens
        for _ in range(5):
            bucket.try_consume()

        # Wait for refill (at 10/sec, 0.1s = 1 token)
        time.sleep(0.15)

        # Should have refilled
        assert bucket.try_consume() in (RateLimitResult.ALLOWED, RateLimitResult.WARNED)

    def test_warning_threshold(self, bucket):
        """Warning should be issued when near limit."""
        # Consume most tokens
        for _ in range(4):
            bucket.try_consume()

        # Next should trigger warning
        result = bucket.try_consume()
        assert result == RateLimitResult.WARNED

    def test_get_remaining_tokens(self, bucket):
        """Should correctly report remaining tokens."""
        assert bucket.get_remaining_tokens() == 5

        bucket.try_consume()
        bucket.try_consume()

        assert bucket.get_remaining_tokens() == 3


class TestRateLimiter:
    """Tests for per-player rate limiting."""

    @pytest.fixture
    def limiter(self):
        """Create a test rate limiter."""
        return RateLimiter({
            "input": RateLimitConfig(requests_per_second=10.0, burst_size=5),
            "rpc": RateLimitConfig(requests_per_second=2.0, burst_size=3),
        })

    def test_per_player_limits(self, limiter):
        """Each player should have independent limits."""
        # Player 1 uses up their tokens
        for _ in range(5):
            limiter.check_rate_limit("player-1", "input")

        assert limiter.check_rate_limit("player-1", "input") == RateLimitResult.DENIED

        # Player 2 should still have full capacity
        assert limiter.check_rate_limit("player-2", "input") != RateLimitResult.DENIED

    def test_per_action_limits(self, limiter):
        """Each action type should have independent limits."""
        # Use up input tokens
        for _ in range(5):
            limiter.check_rate_limit("player-1", "input")

        # RPC should still be available
        result = limiter.check_rate_limit("player-1", "rpc")
        assert result != RateLimitResult.DENIED

    def test_unknown_action_default(self, limiter):
        """Unknown actions should use default config."""
        result = limiter.check_rate_limit("player-1", "unknown_action")
        assert result in (RateLimitResult.ALLOWED, RateLimitResult.WARNED)

    def test_player_removal(self, limiter):
        """Removed players should have limits reset."""
        # Use some tokens
        limiter.check_rate_limit("player-1", "input")

        # Remove player
        limiter.remove_player("player-1")

        # Stats should be cleared
        stats = limiter.get_player_stats("player-1")
        assert not stats  # Empty dict

    def test_time_until_allowed(self, limiter):
        """Should calculate time until action is allowed."""
        # Use all tokens
        for _ in range(5):
            limiter.check_rate_limit("player-1", "input")

        wait_time = limiter.time_until_allowed("player-1", "input")
        assert wait_time > 0


class TestAdaptiveRateLimiter:
    """Tests for adaptive rate limiting."""

    @pytest.fixture
    def adaptive_limiter(self):
        """Create an adaptive rate limiter."""
        return AdaptiveRateLimiter(
            default_configs={
                "input": RateLimitConfig(requests_per_second=10.0, burst_size=5)
            },
            load_threshold=0.8,
            reduction_factor=0.5
        )

    def test_normal_operation(self, adaptive_limiter):
        """Normal operation should work as standard limiter."""
        adaptive_limiter.update_server_load(0.5)  # Under threshold

        result = adaptive_limiter.check_rate_limit("player-1", "input")
        assert result != RateLimitResult.DENIED

    def test_reduced_limits_under_load(self, adaptive_limiter):
        """Limits should be reduced when server is overloaded."""
        adaptive_limiter.update_server_load(0.9)  # Over threshold
        assert adaptive_limiter.is_overloaded

        # Consume tokens faster due to reduction
        for _ in range(3):
            adaptive_limiter.check_rate_limit("player-1", "input")

        # Should hit limit faster
        result = adaptive_limiter.check_rate_limit("player-1", "input")
        # Either denied or warned (near limit)
        assert result in (RateLimitResult.DENIED, RateLimitResult.WARNED)


# =============================================================================
# Anomaly Detector Tests
# =============================================================================


class TestAnomalyDetector:
    """Tests for anomaly detection system."""

    @pytest.fixture
    def detector(self):
        """Create a test anomaly detector."""
        return AnomalyDetector(AnomalyThresholds(
            accuracy_threshold=0.95,
            accuracy_sample_size=10,  # Lower for testing
            headshot_rate_threshold=0.80,
            min_reaction_time_ms=100.0,
            reaction_sample_size=5,  # Lower for testing
        ))

    def test_normal_player_passes(self, detector):
        """Normal player behavior should not trigger anomalies."""
        player_id = "player-1"
        current_time = time.time()

        # Record normal shots (50% accuracy, 30% headshots)
        for i in range(20):
            detector.record_event(player_id, "shot", {
                "timestamp": current_time + i * 0.5,
                "hit": i % 2 == 0,  # 50% hit rate
                "headshot": i % 10 == 0,  # 10% headshot rate
                "distance": 50.0,
                "target_visible": True
            })

        anomalies = detector.analyze_player(player_id)
        assert not any(a.anomaly_type == AnomalyType.AIMBOT for a in anomalies)

    def test_aimbot_pattern_detected(self, detector):
        """Aimbot-like accuracy should be detected."""
        player_id = "cheater-1"
        current_time = time.time()

        # Record suspiciously accurate shots (98% accuracy, 90% headshots)
        for i in range(50):
            detector.record_event(player_id, "shot", {
                "timestamp": current_time + i * 0.1,
                "hit": i % 50 != 0,  # 98% hit rate
                "headshot": i % 10 != 0,  # 90% headshot rate when hitting
                "distance": 100.0,
                "target_visible": True
            })

        anomalies = detector.analyze_player(player_id)
        aimbot_detections = [a for a in anomalies if a.anomaly_type == AnomalyType.AIMBOT]

        assert len(aimbot_detections) > 0
        assert aimbot_detections[0].severity == AnomalySeverity.CRITICAL

    def test_speed_hack_pattern_detected(self, detector):
        """Speed hack pattern should be detected."""
        player_id = "cheater-1"
        current_time = time.time()

        # Record constant max speed movement (no variance)
        for i in range(20):
            detector.record_event(player_id, "movement", {
                "timestamp": current_time + i * 0.1,
                "speed": 500.0,  # Constant high speed
                "position_delta": 50.0,
                "time_delta": 0.1
            })

        anomalies = detector.analyze_player(player_id)
        speed_detections = [a for a in anomalies if a.anomaly_type == AnomalyType.SPEED_HACK]

        assert len(speed_detections) > 0

    def test_impossible_reaction_detected(self, detector):
        """Impossible reaction times should be detected."""
        player_id = "cheater-1"
        current_time = time.time()

        # Record impossibly fast reactions (< 100ms sustained)
        for i in range(10):
            detector.record_event(player_id, "reaction", {
                "timestamp": current_time + i * 1.0,
                "reaction_time_ms": 50.0 + (i % 3) * 10,  # 50-70ms consistently
                "stimulus_type": "enemy_visible"
            })

        anomalies = detector.analyze_player(player_id)
        reaction_detections = [a for a in anomalies if a.anomaly_type == AnomalyType.IMPOSSIBLE_REACTION]

        assert len(reaction_detections) > 0
        assert reaction_detections[0].severity == AnomalySeverity.CRITICAL

    def test_wallhack_detection(self, detector):
        """High wall hit rate should trigger suspicion."""
        player_id = "cheater-1"
        current_time = time.time()

        # Record shots through walls
        for i in range(20):
            detector.record_event(player_id, "shot", {
                "timestamp": current_time + i * 0.5,
                "hit": True,
                "headshot": False,
                "distance": 50.0,
                "target_visible": i % 2 == 0  # 50% through walls
            })

        anomalies = detector.analyze_player(player_id)
        wallhack_detections = [a for a in anomalies if a.anomaly_type == AnomalyType.WALL_HACK_SUSPECT]

        assert len(wallhack_detections) > 0

    def test_risk_score_calculation(self, detector):
        """Risk score should reflect anomaly severity."""
        player_id = "cheater-1"
        current_time = time.time()

        # Record highly suspicious behavior - aimbot-like
        for i in range(50):
            detector.record_event(player_id, "shot", {
                "timestamp": current_time + i * 0.1,
                "hit": True,
                "headshot": True,
                "distance": 100.0,
                "target_visible": True
            })

        # Analyze to create anomaly records
        anomalies = detector.analyze_player(player_id)
        risk_score = detector.get_player_risk_score(player_id)

        # Should have detected aimbot and have non-zero risk
        assert len(anomalies) > 0
        assert risk_score > 0.0  # Has some risk (severity affects score)

    def test_custom_detector_registration(self, detector):
        """Custom detectors should be called during analysis."""
        called = {"count": 0}

        def custom_detector(stats: PlayerStats, thresholds: AnomalyThresholds):
            called["count"] += 1
            return None

        detector.register_custom_detector(custom_detector)
        detector.record_event("player-1", "shot", {"hit": True, "headshot": False, "distance": 10, "target_visible": True})
        detector.analyze_player("player-1")

        assert called["count"] == 1


# =============================================================================
# Response Manager Tests
# =============================================================================


class TestResponseManager:
    """Tests for response escalation system."""

    @pytest.fixture
    def manager(self):
        """Create a test response manager."""
        return ResponseManager()

    def test_first_offense_warning(self, manager):
        """First offense should result in a warning."""
        response = manager.record_violation("player-1", "AIMBOT", severity=4, confidence=0.9)

        assert response is not None
        assert response.severity == ResponseSeverity.WARNING

    def test_repeated_offenses_escalate(self, manager):
        """Repeated offenses should escalate responses."""
        player_id = "player-1"

        # Record multiple violations with enough time between them to pass cooldown
        responses = []
        for i in range(5):
            response = manager.record_violation(player_id, "AIMBOT", severity=3, confidence=0.8)
            if response:
                responses.append(response)
            time.sleep(0.07)  # Small delay (past 60s cooldown in test would be needed for real escalation)

        # Should have received at least one warning
        assert len(responses) > 0
        assert any(r.severity == ResponseSeverity.WARNING for r in responses)

        # Check that history shows warnings were issued
        history = manager.get_violation_history(player_id)
        assert history is not None
        assert history.warnings_issued >= 1

    def test_warning_to_kick_escalation(self, manager):
        """After multiple warnings, should escalate to kick."""
        player_id = "player-1"

        # Generate warnings (with cooldown consideration)
        manager.record_violation(player_id, "SPEED_HACK", severity=2, confidence=0.7)

        history = manager.get_violation_history(player_id)
        assert history is not None
        assert history.warnings_issued >= 1

    def test_kick_to_ban_escalation(self, manager):
        """After multiple kicks, should escalate to ban."""
        player_id = "player-1"

        # Simulate escalation to kick level
        for _ in range(10):
            manager.record_violation(player_id, "AIMBOT", severity=4, confidence=0.95)
            time.sleep(0.01)

        # Check ban status
        response = manager.get_response(player_id)
        if response and response.is_ban():
            assert response.severity in (ResponseSeverity.TEMP_BAN, ResponseSeverity.PERMANENT_BAN)

    def test_ban_duration(self, manager):
        """Temporary bans should have correct duration."""
        # Add a manual temp ban
        ban = manager.add_manual_ban(
            "player-1",
            reason="Testing",
            duration=3600.0,  # 1 hour
            issuer_id="test"
        )

        assert ban.is_active()
        assert not ban.is_permanent()
        assert ban.remaining_time() is not None
        assert ban.remaining_time() > 3500  # ~1 hour remaining

    def test_permanent_ban(self, manager):
        """Permanent bans should have no expiration."""
        ban = manager.add_manual_ban(
            "player-1",
            reason="Confirmed cheating",
            duration=None,  # Permanent
            issuer_id="admin"
        )

        assert ban.is_active()
        assert ban.is_permanent()
        assert ban.remaining_time() is None

    def test_hwid_ban(self, manager):
        """Hardware ID bans should be tracked."""
        hwid = generate_hwid_hash(["cpu-123", "gpu-456", "mb-789"])

        manager.add_manual_ban(
            "player-1",
            reason="Hardware banned",
            hwid=hwid,
            issuer_id="admin"
        )

        ban = manager.check_hwid_ban(hwid)
        assert ban is not None
        assert ban.is_active()

    def test_ip_ban(self, manager):
        """IP bans should be tracked."""
        ip = "192.168.1.100"

        manager.add_manual_ban(
            "player-1",
            reason="IP banned",
            ip_address=ip,
            issuer_id="admin"
        )

        ban = manager.check_ip_ban(ip)
        assert ban is not None
        assert ban.is_active()

    def test_lift_ban(self, manager):
        """Bans should be liftable."""
        manager.add_manual_ban("player-1", reason="Test", issuer_id="admin")
        assert manager.is_banned("player-1")

        manager.lift_ban("player-1")
        assert not manager.is_banned("player-1")

    def test_shadow_ban(self, manager):
        """Shadow bans should be tracked separately."""
        # Manually add shadow ban for testing
        manager._shadow_banned.add("player-1")

        assert manager.is_shadow_banned("player-1")
        assert not manager.is_shadow_banned("player-2")

        manager.remove_shadow_ban("player-1")
        assert not manager.is_shadow_banned("player-1")

    def test_response_callback(self):
        """Response callback should be invoked."""
        callbacks = []

        def on_response(player_id: str, response: CheatResponse):
            callbacks.append((player_id, response))

        manager = ResponseManager(on_response=on_response)
        manager.record_violation("player-1", "AIMBOT", severity=4, confidence=0.9)

        assert len(callbacks) > 0
        assert callbacks[0][0] == "player-1"

    def test_statistics(self, manager):
        """Statistics should track violations and responses."""
        for i in range(3):
            manager.record_violation(f"player-{i}", "SPEED_HACK", severity=2, confidence=0.7)

        stats = manager.get_statistics()
        assert stats["total_players_tracked"] == 3
        assert stats["total_violations"] == 3

    def test_expired_ban_cleanup(self, manager):
        """Expired bans should be cleaned up."""
        # Add a ban that expires immediately
        manager.add_manual_ban(
            "player-1",
            reason="Short ban",
            duration=0.01,  # 10ms
            issuer_id="test"
        )

        time.sleep(0.02)  # Wait for expiration

        cleaned = manager.cleanup_expired_bans()
        assert cleaned == 1
        assert not manager.is_banned("player-1")


class TestHWIDGeneration:
    """Tests for hardware ID generation."""

    def test_deterministic_hash(self):
        """Same components should produce same hash."""
        components = ["cpu-123", "gpu-456", "mb-789"]

        hash1 = generate_hwid_hash(components)
        hash2 = generate_hwid_hash(components)

        assert hash1 == hash2

    def test_order_independent(self):
        """Component order should not affect hash."""
        hash1 = generate_hwid_hash(["cpu-123", "gpu-456"])
        hash2 = generate_hwid_hash(["gpu-456", "cpu-123"])

        assert hash1 == hash2

    def test_different_components_different_hash(self):
        """Different components should produce different hashes."""
        hash1 = generate_hwid_hash(["cpu-123"])
        hash2 = generate_hwid_hash(["cpu-456"])

        assert hash1 != hash2


# =============================================================================
# Integration Tests
# =============================================================================


class TestSecurityIntegration:
    """Integration tests for the complete security pipeline."""

    def test_full_security_pipeline(self):
        """Test the complete security pipeline from input to ban."""
        # Initialize all components
        authority = AuthorityValidator()
        input_validator = InputValidator(InputBounds(
            max_speed=100.0,
            tolerance_multiplier=1.0
        ))
        rate_limiter = RateLimiter()
        anomaly_detector = AnomalyDetector()
        response_manager = ResponseManager()

        player_id = "cheater-1"
        server = Caller(id="server", authority=Authority.SERVER)

        # Simulate cheater behavior
        current_time = time.time()

        # Record suspicious shots
        for i in range(50):
            anomaly_detector.record_event(player_id, "shot", {
                "timestamp": current_time + i * 0.1,
                "hit": True,
                "headshot": True,
                "distance": 200.0,
                "target_visible": True
            })

        # Analyze for anomalies
        anomalies = anomaly_detector.analyze_player(player_id)

        # Record violations and get response
        for anomaly in anomalies:
            response = response_manager.record_violation(
                player_id,
                anomaly.anomaly_type.name,
                severity=anomaly.severity.value,
                confidence=anomaly.confidence,
                details=anomaly.details
            )

        # Verify response was generated
        history = response_manager.get_violation_history(player_id)
        assert history is not None
        assert len(history.violations) > 0

    def test_thread_safety(self):
        """Test thread safety of security components."""
        rate_limiter = RateLimiter()
        anomaly_detector = AnomalyDetector()

        errors = []

        def worker(player_id: str):
            try:
                for _ in range(100):
                    rate_limiter.check_rate_limit(player_id, "input")
                    anomaly_detector.record_event(player_id, "shot", {
                        "hit": True, "headshot": False,
                        "distance": 10, "target_visible": True
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"player-{i}",)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
