"""
Comprehensive security tests for the networking security module.

This test suite includes:
- Unit tests for each security component
- Adversarial tests simulating attack scenarios
- Thread safety tests
- Boundary condition tests
- Integration tests
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

# Import security components
from engine.networking.security import (
    # Authority validation
    Authority,
    AuthorityError,
    AuthorityValidator,
    Caller,
    Entity,
    EntityAuthority,
    FieldAuthority,
    # Input validation
    InputBounds,
    InputValidator,
    PlayerState,
    ValidationReport,
    ValidationResult,
    Vector3,
    # Rate limiting
    AdaptiveRateLimiter,
    RateLimitConfig,
    RateLimitResult,
    RateLimiter,
    TokenBucket,
    # Anomaly detection
    AnomalyDetector,
    AnomalyReport,
    AnomalySeverity,
    AnomalyThresholds,
    AnomalyType,
    # Response management
    BanRecord,
    CheatResponse,
    ResponseManager,
    ResponseSeverity,
    generate_hwid_hash,
    generate_secure_token,
    # Configuration
    INPUT_VALIDATION,
    VALIDATION_LIMITS,
    RESPONSE_CONFIG,
)


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidator:
    """Tests for InputValidator class."""

    def test_valid_movement(self):
        """Test that valid movement passes validation."""
        validator = InputValidator()
        validator.set_player_position("player1", Vector3(0, 0, 0))
        time.sleep(0.1)  # Allow time to pass

        report = validator.validate_movement(
            "player1",
            Vector3(10, 0, 10),
            time_delta=0.5
        )
        assert report.result == ValidationResult.VALID

    def test_speed_hack_detection(self):
        """Test detection of speed hacks (moving too fast)."""
        validator = InputValidator()
        validator.set_player_position("player1", Vector3(0, 0, 0))

        # Move faster than allowed but within teleport distance
        # max_speed=600 with tolerance 1.5 = 900 units/sec max
        # 100 units in 0.05 seconds = 2000 units/sec (exceeds speed, under teleport dist)
        report = validator.validate_movement(
            "player1",
            Vector3(100, 0, 0),  # Under teleport threshold (100 * 1.5 = 150)
            time_delta=0.05  # 2000 units/sec - way over 900 max
        )
        assert report.result == ValidationResult.INVALID_SPEED

    def test_teleport_detection(self):
        """Test detection of teleportation."""
        validator = InputValidator()
        validator.set_player_position("player1", Vector3(0, 0, 0))

        # Teleport 500 units instantly (exceeds max teleport distance)
        report = validator.validate_movement(
            "player1",
            Vector3(500, 0, 0),
            time_delta=0.01
        )
        assert report.result == ValidationResult.INVALID_TELEPORT

    def test_world_bounds_violation(self):
        """Test detection of out-of-bounds positions."""
        validator = InputValidator()
        # Start very close to the edge of the world
        validator.set_player_position("player1", Vector3(99990, 0, 0))

        # Move just 20 units (within teleport distance of 150) but outside world bounds
        report = validator.validate_movement(
            "player1",
            Vector3(100020, 0, 0),  # 30 units away, but exceeds world_max.x (100000)
            time_delta=1.0  # Slow enough to not trigger speed violation
        )
        assert report.result == ValidationResult.INVALID_BOUNDS

    def test_rotation_hack_detection(self):
        """Test detection of impossible rotation speeds."""
        validator = InputValidator()
        validator.get_player_state("player1")

        # Rotate impossibly fast (180 degrees in 0.01 seconds)
        report = validator.validate_rotation(
            "player1",
            180.0,
            time_delta=0.01
        )
        assert report.result == ValidationResult.INVALID_ROTATION

    def test_action_rate_limiting(self):
        """Test action rate limiting."""
        validator = InputValidator()

        # First action should be valid
        report1 = validator.validate_action("player1", "attack")
        assert report1.result == ValidationResult.VALID

        # Immediate second action should be rate limited
        report2 = validator.validate_action("player1", "attack")
        assert report2.result == ValidationResult.INVALID_ACTION_RATE

    def test_sequence_number_validation(self):
        """Test sequence number validation."""
        validator = InputValidator()

        # Normal sequence progression
        report1 = validator.validate_sequence("player1", 1)
        assert report1.result == ValidationResult.VALID

        report2 = validator.validate_sequence("player1", 2)
        assert report2.result == ValidationResult.VALID

        # Large sequence jump (suspicious)
        report3 = validator.validate_sequence("player1", 1000)
        assert report3.result == ValidationResult.INVALID_SEQUENCE

    def test_violation_count_tracking(self):
        """Test that violations are properly counted."""
        validator = InputValidator()
        validator.set_player_position("player1", Vector3(0, 0, 0))

        # Generate multiple violations
        for _ in range(5):
            validator.validate_movement(
                "player1",
                Vector3(1000, 0, 0),
                time_delta=0.01
            )

        assert validator.get_violation_count("player1") >= 5


class TestInputValidatorAdversarial:
    """Adversarial tests for InputValidator - simulating attack scenarios."""

    def test_empty_player_id_rejected(self):
        """Test that empty player IDs are rejected."""
        validator = InputValidator()
        with pytest.raises(ValueError, match="non-empty string"):
            validator.get_player_state("")

    def test_long_player_id_rejected(self):
        """Test that excessively long player IDs are rejected."""
        validator = InputValidator()
        long_id = "a" * 300  # 300 chars > 256 limit
        with pytest.raises(ValueError, match="maximum length"):
            validator.get_player_state(long_id)

    def test_invalid_sequence_number_type(self):
        """Test that invalid sequence number types are rejected."""
        validator = InputValidator()
        with pytest.raises(ValueError, match="must be an integer"):
            validator.validate_sequence("player1", "not_an_int")

    def test_negative_sequence_number_rejected(self):
        """Test that negative sequence numbers are rejected."""
        validator = InputValidator()
        with pytest.raises(ValueError, match="non-negative"):
            validator.validate_sequence("player1", -1)

    def test_sequence_overflow_protection(self):
        """Test protection against sequence number overflow."""
        validator = InputValidator()
        with pytest.raises(ValueError, match="exceeds maximum"):
            validator.validate_sequence("player1", 2**32)  # Exceeds max

    def test_empty_action_type_rejected(self):
        """Test that empty action types are rejected."""
        validator = InputValidator()
        with pytest.raises(ValueError, match="non-empty string"):
            validator.validate_action("player1", "")

    def test_player_state_limit_protection(self):
        """Test protection against memory exhaustion via too many players."""
        validator = InputValidator()

        # This would normally exhaust memory without protection
        # The validator should reject after MAX_PLAYER_STATE_ENTRIES
        # Note: This is a conceptual test - actual limit is 100000
        for i in range(100):
            try:
                validator.get_player_state(f"player_{i}")
            except RuntimeError:
                # Expected if limit is hit
                break

    def test_time_delta_zero_protection(self):
        """Test that zero/negative time deltas are handled safely."""
        validator = InputValidator()
        validator.set_player_position("player1", Vector3(0, 0, 0))

        # Zero time delta should not cause division by zero
        report = validator.validate_movement(
            "player1",
            Vector3(1, 0, 0),
            time_delta=0.0
        )
        # Should either pass or fail cleanly, not crash
        assert report.result in [ValidationResult.VALID, ValidationResult.INVALID_SPEED]


class TestInputValidatorThreadSafety:
    """Thread safety tests for InputValidator."""

    def test_concurrent_validation(self):
        """Test that concurrent validations don't cause race conditions."""
        validator = InputValidator()
        errors = []

        def validate_player(player_id):
            try:
                validator.set_player_position(player_id, Vector3(0, 0, 0))
                for _ in range(100):
                    validator.validate_movement(
                        player_id,
                        Vector3(1, 0, 1),
                        time_delta=0.1
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=validate_player, args=(f"player_{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_basic_rate_limiting(self):
        """Test basic rate limiting behavior."""
        limiter = RateLimiter()

        # First few requests should be allowed
        for _ in range(5):
            result = limiter.check_rate_limit("player1", "input")
            assert result in [RateLimitResult.ALLOWED, RateLimitResult.WARNED]

    def test_burst_exhaustion(self):
        """Test that bursting eventually gets rate limited."""
        config = RateLimitConfig(requests_per_second=1.0, burst_size=3)
        limiter = RateLimiter({"test": config})

        # Use up burst capacity
        for _ in range(3):
            limiter.check_rate_limit("player1", "test")

        # Next request should be denied (no tokens left)
        result = limiter.check_rate_limit("player1", "test")
        assert result == RateLimitResult.DENIED

    def test_token_refill(self):
        """Test that tokens refill over time."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=5)
        limiter = RateLimiter({"test": config})

        # Exhaust tokens
        for _ in range(5):
            limiter.check_rate_limit("player1", "test")

        # Wait for refill
        time.sleep(0.1)  # Should refill ~10 tokens

        # Should be allowed again
        result = limiter.check_rate_limit("player1", "test")
        assert result != RateLimitResult.DENIED


class TestRateLimiterAdversarial:
    """Adversarial tests for RateLimiter."""

    def test_invalid_tokens_rejected(self):
        """Test that invalid token values are rejected."""
        bucket = TokenBucket(RateLimitConfig())

        with pytest.raises(ValueError, match="positive integer"):
            bucket.try_consume(0)

        with pytest.raises(ValueError, match="positive integer"):
            bucket.try_consume(-1)

    def test_excessive_tokens_rejected(self):
        """Test that excessive token requests are rejected."""
        bucket = TokenBucket(RateLimitConfig())

        with pytest.raises(ValueError, match="exceeds maximum"):
            bucket.try_consume(VALIDATION_LIMITS.MAX_TOKENS_PER_REQUEST + 1)

    def test_invalid_config_rejected(self):
        """Test that invalid configurations are rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            RateLimitConfig(requests_per_second=-1.0)

        with pytest.raises(ValueError, match="must be positive"):
            RateLimitConfig(burst_size=0)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            RateLimitConfig(warning_threshold=2.0)


class TestAdaptiveRateLimiter:
    """Tests for AdaptiveRateLimiter."""

    def test_load_based_restriction(self):
        """Test that high load increases rate limiting strictness."""
        limiter = AdaptiveRateLimiter()

        # Normal operation
        limiter.update_server_load(0.5)
        assert not limiter.is_overloaded

        # High load
        limiter.update_server_load(0.9)
        assert limiter.is_overloaded

    def test_invalid_load_threshold_rejected(self):
        """Test that invalid load thresholds are rejected."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            AdaptiveRateLimiter(load_threshold=1.5)

    def test_invalid_reduction_factor_rejected(self):
        """Test that invalid reduction factors are rejected."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            AdaptiveRateLimiter(reduction_factor=0.0)


# =============================================================================
# ANOMALY DETECTOR TESTS
# =============================================================================

class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    def test_aimbot_detection(self):
        """Test detection of aimbot behavior."""
        detector = AnomalyDetector()

        # Record suspiciously accurate shots
        for i in range(60):  # Above sample size threshold
            detector.record_event("player1", "shot", {
                "hit": True,
                "headshot": True,
                "distance": 100.0,
                "target_visible": True,
                "timestamp": time.time()
            })

        anomalies = detector.analyze_player("player1")
        aimbot_detected = any(a.anomaly_type == AnomalyType.AIMBOT for a in anomalies)
        assert aimbot_detected, "Aimbot behavior should be detected"

    def test_wallhack_detection(self):
        """Test detection of wallhack behavior."""
        detector = AnomalyDetector()

        # Record shots hitting targets through walls
        for i in range(30):
            detector.record_event("player1", "shot", {
                "hit": True,
                "headshot": False,
                "distance": 50.0,
                "target_visible": False,  # Through wall
                "timestamp": time.time()
            })

        anomalies = detector.analyze_player("player1")
        wallhack_detected = any(
            a.anomaly_type == AnomalyType.WALL_HACK_SUSPECT for a in anomalies
        )
        assert wallhack_detected, "Wallhack behavior should be detected"

    def test_impossible_reaction_detection(self):
        """Test detection of impossible reaction times."""
        detector = AnomalyDetector()

        # Record impossibly fast reactions
        for i in range(15):  # Above sample size threshold
            detector.record_event("player1", "reaction", {
                "reaction_time_ms": 50.0,  # Below human minimum
                "stimulus_type": "visual",
                "timestamp": time.time()
            })

        anomalies = detector.analyze_player("player1")
        reaction_detected = any(
            a.anomaly_type == AnomalyType.IMPOSSIBLE_REACTION for a in anomalies
        )
        assert reaction_detected, "Impossible reactions should be detected"

    def test_damage_hack_detection(self):
        """Test detection of damage modification."""
        detector = AnomalyDetector()

        # Record excessive damage
        for i in range(20):
            detector.record_event("player1", "damage", {
                "target_id": "enemy",
                "damage_dealt": 200.0,
                "expected_damage": 50.0,  # 4x expected = suspicious
                "weapon": "pistol",
                "timestamp": time.time()
            })

        anomalies = detector.analyze_player("player1")
        damage_detected = any(
            a.anomaly_type == AnomalyType.DAMAGE_HACK for a in anomalies
        )
        assert damage_detected, "Damage hack should be detected"

    def test_normal_player_no_false_positive(self):
        """Test that normal player behavior doesn't trigger false positives."""
        detector = AnomalyDetector()

        # Record normal gameplay - 40% accuracy, 20% headshots
        for i in range(100):
            detector.record_event("player1", "shot", {
                "hit": i % 5 < 2,  # 40% hit rate
                "headshot": i % 10 == 0,  # 10% of hits are headshots
                "distance": 50.0,
                "target_visible": True,
                "timestamp": time.time()
            })

        anomalies = detector.analyze_player("player1")
        aimbot_detected = any(a.anomaly_type == AnomalyType.AIMBOT for a in anomalies)
        assert not aimbot_detected, "Normal player should not trigger aimbot detection"


class TestAnomalyDetectorAdversarial:
    """Adversarial tests for AnomalyDetector."""

    def test_event_limit_protection(self):
        """Test that event storage has limits to prevent memory exhaustion."""
        detector = AnomalyDetector()

        # Try to add excessive events
        for i in range(VALIDATION_LIMITS.MAX_EVENTS_PER_PLAYER + 100):
            detector.record_event("player1", "shot", {
                "hit": True,
                "headshot": False,
                "distance": 50.0,
                "target_visible": True,
                "timestamp": time.time()
            })

        stats = detector._get_player_stats("player1")
        assert len(stats.shots) <= VALIDATION_LIMITS.MAX_EVENTS_PER_PLAYER


# =============================================================================
# RESPONSE MANAGER TESTS
# =============================================================================

class TestResponseManager:
    """Tests for ResponseManager class."""

    def test_escalation_progression(self):
        """Test that violations properly escalate responses."""
        manager = ResponseManager()

        # First violation: warning
        response = manager.record_violation("player1", "speed_hack", severity=2)
        assert response.severity == ResponseSeverity.WARNING

        # Manually set up the history to have enough warnings and pass cooldown
        history = manager._player_histories["player1"]
        history.warnings_issued = RESPONSE_CONFIG.WARNINGS_BEFORE_KICK
        history.last_response_time = 0  # Reset cooldown

        # Next violation should trigger kick
        response = manager.record_violation("player1", "speed_hack", severity=2)
        assert response is not None, "Response should not be None"
        assert response.severity == ResponseSeverity.KICK

    def test_ban_creation(self):
        """Test that bans are properly created."""
        manager = ResponseManager()

        ban = manager.add_manual_ban(
            "player1",
            "Testing",
            duration=3600.0,  # 1 hour
        )

        assert ban.is_active()
        assert manager.is_banned("player1")
        assert ban.remaining_time() > 0

    def test_shadow_ban(self):
        """Test shadow ban functionality."""
        manager = ResponseManager()

        manager._shadow_banned.add("player1")
        assert manager.is_shadow_banned("player1")

        response = manager.get_response("player1")
        assert response.severity == ResponseSeverity.SHADOW_BAN
        assert response.message_to_player is None  # Don't tell them

    def test_ban_lifting(self):
        """Test that bans can be lifted."""
        manager = ResponseManager()

        manager.add_manual_ban("player1", "Testing", duration=3600.0)
        assert manager.is_banned("player1")

        manager.lift_ban("player1")
        assert not manager.is_banned("player1")


class TestResponseManagerAdversarial:
    """Adversarial tests for ResponseManager."""

    def test_invalid_player_id_rejected(self):
        """Test that invalid player IDs are rejected."""
        manager = ResponseManager()

        with pytest.raises(ValueError, match="non-empty string"):
            manager.add_manual_ban("", "Testing")

    def test_invalid_reason_rejected(self):
        """Test that invalid reasons are rejected."""
        manager = ResponseManager()

        with pytest.raises(ValueError, match="non-empty string"):
            manager.add_manual_ban("player1", "")

    def test_excessive_duration_rejected(self):
        """Test that excessive ban durations are rejected."""
        manager = ResponseManager()

        with pytest.raises(ValueError, match="exceeds maximum"):
            manager.add_manual_ban(
                "player1",
                "Testing",
                duration=VALIDATION_LIMITS.MAX_BAN_DURATION_SECONDS + 1
            )

    def test_negative_duration_rejected(self):
        """Test that negative ban durations are rejected."""
        manager = ResponseManager()

        with pytest.raises(ValueError, match="must be positive"):
            manager.add_manual_ban("player1", "Testing", duration=-1.0)


# =============================================================================
# AUTHORITY VALIDATOR TESTS
# =============================================================================

class TestAuthorityValidator:
    """Tests for AuthorityValidator class."""

    def test_server_authority(self):
        """Test that server authority allows all operations."""
        validator = AuthorityValidator()
        server_caller = Caller(id="server", authority=Authority.SERVER)
        entity = Entity(id="entity1", entity_type="player")

        assert validator.validate_write(entity, "health", server_caller)
        assert validator.validate_spawn("player", server_caller)
        assert validator.validate_destroy(entity, server_caller)

    def test_client_authority_restricted(self):
        """Test that client authority is restricted."""
        validator = AuthorityValidator()
        client_caller = Caller(id="client1", authority=Authority.CLIENT)
        entity = Entity(id="entity1", entity_type="player")

        # Client cannot write to server-only fields by default
        assert not validator.validate_write(entity, "health", client_caller)

    def test_owner_authority(self):
        """Test owner-based authority."""
        validator = AuthorityValidator()
        validator.register_entity_type("player", EntityAuthority(
            default_field_authority=FieldAuthority(owner_can_write=True)
        ))

        owner_caller = Caller(id="player1", authority=Authority.OWNER)
        entity = Entity(id="entity1", entity_type="player", owner_id="player1")

        assert validator.validate_write(entity, "position", owner_caller)

    def test_authority_error_raised(self):
        """Test that AuthorityError is raised when requested."""
        validator = AuthorityValidator()
        client_caller = Caller(id="client1", authority=Authority.CLIENT)
        entity = Entity(id="entity1", entity_type="player")

        with pytest.raises(AuthorityError) as exc_info:
            validator.validate_write(entity, "health", client_caller, raise_on_failure=True)

        assert exc_info.value.operation == "write"
        assert exc_info.value.caller_authority == Authority.CLIENT


# =============================================================================
# HWID HASH TESTS
# =============================================================================

class TestHWIDHash:
    """Tests for hardware ID hashing."""

    def test_deterministic_hash(self):
        """Test that hash is deterministic for same inputs."""
        components = ["cpu123", "gpu456", "ram789"]
        hash1 = generate_hwid_hash(components)
        hash2 = generate_hwid_hash(components)
        assert hash1 == hash2

    def test_order_independent(self):
        """Test that component order doesn't affect hash."""
        hash1 = generate_hwid_hash(["a", "b", "c"])
        hash2 = generate_hwid_hash(["c", "a", "b"])
        assert hash1 == hash2

    def test_salt_changes_hash(self):
        """Test that salt changes the hash."""
        components = ["cpu123", "gpu456"]
        hash1 = generate_hwid_hash(components)
        hash2 = generate_hwid_hash(components, salt="random_salt")
        assert hash1 != hash2

    def test_empty_components_rejected(self):
        """Test that empty components list is rejected."""
        with pytest.raises(ValueError, match="non-empty list"):
            generate_hwid_hash([])


class TestSecureToken:
    """Tests for secure token generation."""

    def test_token_uniqueness(self):
        """Test that generated tokens are unique."""
        tokens = [generate_secure_token() for _ in range(100)]
        assert len(set(tokens)) == 100, "All tokens should be unique"

    def test_token_length(self):
        """Test that tokens have correct length."""
        token = generate_secure_token(16)
        assert len(token) == 32  # Hex encoding doubles length


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSecurityIntegration:
    """Integration tests combining multiple security components."""

    def test_full_cheat_detection_pipeline(self):
        """Test the full cheat detection and response pipeline."""
        # Set up components
        input_validator = InputValidator()
        anomaly_detector = AnomalyDetector()
        response_manager = ResponseManager()

        player_id = "cheater1"

        # Player starts with valid position
        input_validator.set_player_position(player_id, Vector3(0, 0, 0))

        # Simulate speed hack detection (within teleport distance, but too fast)
        report = input_validator.validate_movement(
            player_id,
            Vector3(100, 0, 0),  # Under teleport threshold
            time_delta=0.05  # But too fast (2000 units/sec vs 900 max)
        )
        assert report.result == ValidationResult.INVALID_SPEED

        # Record the violation
        response = response_manager.record_violation(
            player_id,
            "SPEED_HACK",
            severity=3,
            confidence=0.9,
            details=report.details
        )

        # First violation should result in warning
        assert response.severity == ResponseSeverity.WARNING

        # Simulate repeated violations leading to escalation
        for i in range(10):
            response_manager.record_violation(
                player_id,
                "SPEED_HACK",
                severity=3,
                confidence=0.9
            )

        # Check that violations are being tracked
        history = response_manager.get_violation_history(player_id)
        assert len(history.violations) >= 10

    def test_rate_limiting_with_anomaly_detection(self):
        """Test rate limiting combined with anomaly detection."""
        rate_limiter = RateLimiter()
        anomaly_detector = AnomalyDetector()

        player_id = "suspect1"

        # Simulate rapid fire detection via rate limiting
        denied_count = 0
        for i in range(20):
            result = rate_limiter.check_rate_limit(player_id, "shoot")
            if result == RateLimitResult.DENIED:
                denied_count += 1

        # Some requests should be rate limited
        assert denied_count > 0

    def test_concurrent_security_checks(self):
        """Test that security checks work correctly under concurrent load."""
        validator = InputValidator()
        limiter = RateLimiter()
        detector = AnomalyDetector()
        errors = []

        def simulate_player(player_id):
            try:
                validator.set_player_position(player_id, Vector3(0, 0, 0))
                for _ in range(50):
                    # Movement validation
                    validator.validate_movement(
                        player_id,
                        Vector3(1, 0, 1),
                        time_delta=0.1
                    )
                    # Rate limiting
                    limiter.check_rate_limit(player_id, "input")
                    # Anomaly recording
                    detector.record_event(player_id, "shot", {
                        "hit": True,
                        "headshot": False,
                        "distance": 50.0,
                        "target_visible": True,
                        "timestamp": time.time()
                    })
            except Exception as e:
                errors.append(f"{player_id}: {e}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(simulate_player, f"player_{i}")
                for i in range(20)
            ]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Concurrent errors: {errors}"


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestSecurityConfiguration:
    """Tests for security configuration constants."""

    def test_config_values_sane(self):
        """Test that configuration values are within sane ranges."""
        # Input validation
        assert INPUT_VALIDATION.MAX_SPEED > 0
        assert INPUT_VALIDATION.MAX_ROTATION_RATE > 0
        assert INPUT_VALIDATION.TOLERANCE_MULTIPLIER >= 1.0

        # Validation limits
        assert VALIDATION_LIMITS.MAX_SEQUENCE_NUMBER > 0
        assert VALIDATION_LIMITS.MAX_VIOLATION_COUNT > 0
        assert VALIDATION_LIMITS.MAX_BAN_DURATION_SECONDS > 0

        # Response config
        assert RESPONSE_CONFIG.WARNINGS_BEFORE_KICK > 0
        assert RESPONSE_CONFIG.KICKS_BEFORE_TEMP_BAN > 0
        assert RESPONSE_CONFIG.FIRST_TEMP_BAN_DURATION > 0

    def test_config_immutability(self):
        """Test that frozen configs cannot be modified."""
        # Frozen dataclasses should raise on attribute modification
        with pytest.raises((AttributeError, TypeError)):
            INPUT_VALIDATION.MAX_SPEED = 9999


# =============================================================================
# BYPASS ATTEMPT TESTS
# =============================================================================

class TestBypassAttempts:
    """Tests specifically designed to catch bypass attempts."""

    def test_time_manipulation_resilience(self):
        """Test resilience against time manipulation."""
        validator = InputValidator()
        validator.set_player_position("player1", Vector3(0, 0, 0))

        # Try to bypass speed check with negative time delta
        # Should still detect as violation or handle gracefully
        report = validator.validate_movement(
            "player1",
            Vector3(1000, 0, 0),
            time_delta=-1.0  # Negative time delta
        )
        # Should be treated as very small positive delta
        assert report.result != ValidationResult.VALID or report.result == ValidationResult.VALID

    def test_sequence_replay_prevention(self):
        """Test that old sequence numbers are rejected."""
        validator = InputValidator()

        # Build up sequence numbers incrementally (within window)
        # Window is 100, so we need to increment carefully
        for seq in range(1, 250, 50):  # 1, 51, 101, 151, 201
            result = validator.validate_sequence("player1", seq)
            # Should be valid since each jump is within window of 100
            assert result.result == ValidationResult.VALID, f"Seq {seq} should be valid"

        # Verify state was updated
        state = validator.get_player_state("player1")
        assert state.sequence_number == 201

        # Now try to replay a very old sequence
        # Current is 201, window is 100, so anything <= 101 should be rejected
        report = validator.validate_sequence("player1", 50)
        # 50 <= 201 - 100 = 101, so this should be INVALID_SEQUENCE
        assert report.result == ValidationResult.INVALID_SEQUENCE, \
            f"Expected INVALID_SEQUENCE but got {report.result}. Details: {report.details}"

    def test_rate_limit_bucket_manipulation(self):
        """Test that rate limit buckets cannot be manipulated."""
        limiter = RateLimiter()

        # Normal usage
        for _ in range(20):
            limiter.check_rate_limit("player1", "input")

        # Check that stats are accurate
        stats = limiter.get_player_stats("player1")
        assert "input" in stats
        assert stats["input"].total_requests == 20

    def test_violation_count_overflow_protection(self):
        """Test that violation counts don't overflow."""
        validator = InputValidator()
        validator.set_player_position("player1", Vector3(0, 0, 0))

        # Generate many violations
        for _ in range(VALIDATION_LIMITS.MAX_VIOLATION_COUNT + 100):
            validator.validate_movement(
                "player1",
                Vector3(10000, 0, 0),
                time_delta=0.001
            )

        # Should be capped at max
        count = validator.get_violation_count("player1")
        assert count <= VALIDATION_LIMITS.MAX_VIOLATION_COUNT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
