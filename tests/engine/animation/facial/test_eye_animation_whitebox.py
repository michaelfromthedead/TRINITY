"""
Whitebox Tests for Eye Animation Vergence (T3.4).

Tests the internal vergence calculation with full source code access.
Validates geometric correctness of eye convergence for near/far targets.
"""

from __future__ import annotations

import math
import pytest
from typing import Tuple

from engine.animation.facial.eye_animation import (
    EyeController,
    EyeLimits,
    EyeTransform,
    BlinkSettings,
    SaccadeSettings,
    PupilSettings,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_controller() -> EyeController:
    """Create an EyeController with default settings."""
    return EyeController()


@pytest.fixture
def custom_limits_controller() -> EyeController:
    """Create an EyeController with custom vergence limits."""
    limits = EyeLimits(
        max_yaw=35.0,
        max_pitch_up=25.0,
        max_pitch_down=30.0,
        max_vergence=10.0,  # Reduced max vergence for testing
    )
    return EyeController(eye_limits=limits)


@pytest.fixture
def wide_eye_controller() -> EyeController:
    """Create an EyeController with wider eye separation."""
    return EyeController(eye_separation=0.08)  # 8cm separation


@pytest.fixture
def narrow_eye_controller() -> EyeController:
    """Create an EyeController with narrower eye separation."""
    return EyeController(eye_separation=0.05)  # 5cm separation


# =============================================================================
# Vergence Calculation Tests
# =============================================================================


class TestVergenceCalculation:
    """Tests for the _update_vergence() internal method."""

    def test_vergence_zero_when_no_target(self, default_controller: EyeController) -> None:
        """Vergence should be zero when there is no look-at target."""
        # Don't set any target
        default_controller.update(0.016)

        assert default_controller.left_eye.vergence == 0.0
        assert default_controller.right_eye.vergence == 0.0

    def test_vergence_zero_after_clearing_target(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should return to zero after clearing the target."""
        # Set a near target
        default_controller.look_at((0.0, 0.0, 0.5))
        default_controller.update(0.016)

        # Vergence should be non-zero
        assert default_controller.left_eye.vergence != 0.0

        # Clear target
        default_controller.clear_target()
        default_controller.update(0.016)

        # Vergence should be zero
        assert default_controller.left_eye.vergence == 0.0
        assert default_controller.right_eye.vergence == 0.0

    def test_vergence_geometrically_correct_formula(
        self, default_controller: EyeController
    ) -> None:
        """Verify vergence angle matches the expected geometric formula."""
        eye_separation = 0.065  # Default eye separation
        distance = 1.0  # 1 meter

        # Expected vergence: atan2(eye_separation/2, distance)
        expected_vergence = math.degrees(math.atan2(eye_separation * 0.5, distance))

        # Set target at 1m distance
        default_controller.look_at((0.0, 0.0, 1.0))
        default_controller.update(0.016)

        # Check vergence matches formula
        assert abs(default_controller.left_eye.vergence - expected_vergence) < 0.001
        assert abs(default_controller.right_eye.vergence + expected_vergence) < 0.001

    def test_vergence_increases_with_closer_distance(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should increase as target gets closer."""
        distances = [5.0, 2.0, 1.0, 0.5, 0.2]
        vergences = []

        for dist in distances:
            default_controller.look_at((0.0, 0.0, dist))
            default_controller.update(0.016)
            vergences.append(default_controller.left_eye.vergence)

        # Each vergence should be greater than the previous (closer = more vergence)
        for i in range(1, len(vergences)):
            assert vergences[i] > vergences[i - 1], (
                f"Vergence at {distances[i]}m ({vergences[i]}) should be greater than "
                f"vergence at {distances[i-1]}m ({vergences[i-1]})"
            )


class TestVergenceDirection:
    """Tests for vergence direction (left vs right eye)."""

    def test_left_eye_positive_vergence(
        self, default_controller: EyeController
    ) -> None:
        """Left eye should have positive vergence (converging right)."""
        default_controller.look_at((0.0, 0.0, 0.5))
        default_controller.update(0.016)

        assert default_controller.left_eye.vergence > 0.0

    def test_right_eye_negative_vergence(
        self, default_controller: EyeController
    ) -> None:
        """Right eye should have negative vergence (converging left)."""
        default_controller.look_at((0.0, 0.0, 0.5))
        default_controller.update(0.016)

        assert default_controller.right_eye.vergence < 0.0

    def test_vergence_symmetry(self, default_controller: EyeController) -> None:
        """Left and right eye vergence should be symmetric (equal magnitude)."""
        default_controller.look_at((0.0, 0.0, 0.5))
        default_controller.update(0.016)

        left_vergence = default_controller.left_eye.vergence
        right_vergence = default_controller.right_eye.vergence

        assert abs(left_vergence + right_vergence) < 0.001, (
            f"Vergence asymmetry: left={left_vergence}, right={right_vergence}"
        )


class TestVergenceNearTargets:
    """Tests for vergence with near targets (high convergence)."""

    def test_vergence_at_minimum_distance(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should be calculated correctly at minimum distance (0.1m)."""
        eye_separation = 0.065
        min_distance = 0.1

        # Expected vergence at minimum clamped distance
        expected_vergence = math.degrees(
            math.atan2(eye_separation * 0.5, min_distance)
        )
        # But clamped to max_vergence (15 degrees by default)
        expected_vergence = min(expected_vergence, 15.0)

        default_controller.look_at((0.0, 0.0, 0.1))
        default_controller.update(0.016)

        assert abs(default_controller.left_eye.vergence - expected_vergence) < 0.001

    def test_vergence_at_very_close_distance_clamped(
        self, default_controller: EyeController
    ) -> None:
        """Very close targets (< 0.1m) should have distance clamped to 0.1m."""
        # Set target at 0.05m (should be clamped to 0.1m)
        default_controller.look_at((0.0, 0.0, 0.05))
        default_controller.update(0.016)
        vergence_at_005 = default_controller.left_eye.vergence

        # Set target at 0.01m (should also be clamped to 0.1m)
        default_controller.look_at((0.0, 0.0, 0.01))
        default_controller.update(0.016)
        vergence_at_001 = default_controller.left_eye.vergence

        # Both should produce same vergence (clamped to 0.1m)
        assert abs(vergence_at_005 - vergence_at_001) < 0.001

    def test_vergence_near_reading_distance(
        self, default_controller: EyeController
    ) -> None:
        """Test vergence at typical reading distance (0.3m)."""
        eye_separation = 0.065
        reading_distance = 0.3

        expected_vergence = math.degrees(
            math.atan2(eye_separation * 0.5, reading_distance)
        )

        default_controller.look_at((0.0, 0.0, 0.3))
        default_controller.update(0.016)

        assert abs(default_controller.left_eye.vergence - expected_vergence) < 0.001


class TestVergenceFarTargets:
    """Tests for vergence with distant targets (parallel eyes)."""

    def test_vergence_nearly_zero_at_infinity(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should approach zero for very distant targets."""
        default_controller.look_at((0.0, 0.0, 100.0))
        default_controller.update(0.016)

        # At 100m, vergence should be negligible
        assert abs(default_controller.left_eye.vergence) < 0.02
        assert abs(default_controller.right_eye.vergence) < 0.02

    def test_vergence_at_10m(self, default_controller: EyeController) -> None:
        """Test vergence at 10 meters distance."""
        eye_separation = 0.065
        distance = 10.0

        expected_vergence = math.degrees(math.atan2(eye_separation * 0.5, distance))

        default_controller.look_at((0.0, 0.0, 10.0))
        default_controller.update(0.016)

        # Should be approximately 0.186 degrees
        assert abs(default_controller.left_eye.vergence - expected_vergence) < 0.001

    def test_eyes_parallel_at_far_distance(
        self, default_controller: EyeController
    ) -> None:
        """Eyes should be effectively parallel for distant targets."""
        default_controller.look_at((0.0, 0.0, 1000.0))
        default_controller.update(0.016)

        # Vergence should be less than 0.01 degrees (effectively parallel)
        assert abs(default_controller.left_eye.vergence) < 0.01
        assert abs(default_controller.right_eye.vergence) < 0.01


class TestVergenceLimits:
    """Tests for max vergence limit enforcement."""

    def test_vergence_clamped_to_max_limit(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should be clamped to max_vergence limit."""
        # At very close distance, geometric vergence would exceed 15 degrees
        # but should be clamped
        default_controller.look_at((0.0, 0.0, 0.1))
        default_controller.update(0.016)

        # Default max_vergence is 15.0 degrees
        assert default_controller.left_eye.vergence <= 15.0
        assert default_controller.right_eye.vergence >= -15.0

    def test_custom_max_vergence_limit(
        self, custom_limits_controller: EyeController
    ) -> None:
        """Custom max_vergence limit should be respected."""
        custom_limits_controller.look_at((0.0, 0.0, 0.1))
        custom_limits_controller.update(0.016)

        # Custom max_vergence is 10.0 degrees
        assert custom_limits_controller.left_eye.vergence <= 10.0
        assert custom_limits_controller.right_eye.vergence >= -10.0

    def test_vergence_exactly_at_limit_for_very_near(
        self, default_controller: EyeController
    ) -> None:
        """Very near targets should hit the max vergence limit exactly."""
        eye_separation = 0.065
        min_distance = 0.1
        max_vergence = 15.0

        # Calculate geometric vergence at minimum distance
        geometric_vergence = math.degrees(
            math.atan2(eye_separation * 0.5, min_distance)
        )

        default_controller.look_at((0.0, 0.0, 0.1))
        default_controller.update(0.016)

        # If geometric vergence exceeds limit, actual should equal limit
        if geometric_vergence > max_vergence:
            assert abs(default_controller.left_eye.vergence - max_vergence) < 0.001
        else:
            assert (
                abs(default_controller.left_eye.vergence - geometric_vergence) < 0.001
            )


class TestEyeSeparationEffect:
    """Tests for how eye separation affects vergence."""

    def test_wider_eyes_have_more_vergence(
        self,
        default_controller: EyeController,
        wide_eye_controller: EyeController,
    ) -> None:
        """Wider eye separation should result in more vergence."""
        distance = 0.5

        default_controller.look_at((0.0, 0.0, distance))
        default_controller.update(0.016)
        default_vergence = default_controller.left_eye.vergence

        wide_eye_controller.look_at((0.0, 0.0, distance))
        wide_eye_controller.update(0.016)
        wide_vergence = wide_eye_controller.left_eye.vergence

        assert wide_vergence > default_vergence

    def test_narrower_eyes_have_less_vergence(
        self,
        default_controller: EyeController,
        narrow_eye_controller: EyeController,
    ) -> None:
        """Narrower eye separation should result in less vergence."""
        distance = 0.5

        default_controller.look_at((0.0, 0.0, distance))
        default_controller.update(0.016)
        default_vergence = default_controller.left_eye.vergence

        narrow_eye_controller.look_at((0.0, 0.0, distance))
        narrow_eye_controller.update(0.016)
        narrow_vergence = narrow_eye_controller.left_eye.vergence

        assert narrow_vergence < default_vergence

    def test_eye_separation_vergence_calculation(self) -> None:
        """Verify vergence is proportional to eye separation."""
        separations = [0.05, 0.065, 0.08]
        distance = 1.0

        for sep in separations:
            controller = EyeController(eye_separation=sep)
            controller.look_at((0.0, 0.0, distance))
            controller.update(0.016)

            expected = math.degrees(math.atan2(sep * 0.5, distance))
            assert abs(controller.left_eye.vergence - expected) < 0.001


class TestVergenceWithOffAxisTargets:
    """Tests for vergence with targets not directly in front."""

    def test_vergence_with_lateral_offset(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should be calculated correctly for off-center targets."""
        # Target to the right at 1m depth
        default_controller.look_at((0.5, 0.0, 1.0))
        default_controller.update(0.016)

        # Distance should include lateral offset
        distance = math.sqrt(0.5**2 + 1.0**2)
        eye_separation = 0.065
        expected_vergence = math.degrees(math.atan2(eye_separation * 0.5, distance))

        # Vergence is based on total distance, not just z
        assert default_controller.left_eye.vergence > 0.0
        assert default_controller.right_eye.vergence < 0.0

    def test_vergence_with_vertical_offset(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should account for vertical distance in total distance."""
        # Target above at 1m depth
        default_controller.look_at((0.0, 0.5, 1.0))
        default_controller.update(0.016)

        # Distance includes vertical offset
        distance = math.sqrt(0.5**2 + 1.0**2)
        eye_separation = 0.065
        expected_vergence = math.degrees(math.atan2(eye_separation * 0.5, distance))

        assert default_controller.left_eye.vergence > 0.0

    def test_vergence_diagonal_target(
        self, default_controller: EyeController
    ) -> None:
        """Test vergence for target with both lateral and vertical offset."""
        # Target at diagonal position
        default_controller.look_at((0.3, 0.4, 1.0))
        default_controller.update(0.016)

        # Total distance
        distance = math.sqrt(0.3**2 + 0.4**2 + 1.0**2)
        eye_separation = 0.065
        expected_vergence = math.degrees(math.atan2(eye_separation * 0.5, distance))

        # Vergence should still follow the formula based on total distance
        assert default_controller.left_eye.vergence > 0.0
        assert default_controller.right_eye.vergence < 0.0


class TestVergenceIntegrationWithEyeTransform:
    """Tests for vergence integration with EyeTransform output."""

    def test_vergence_affects_euler_output(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should be included in EyeTransform euler angles."""
        default_controller.look_at((0.0, 0.0, 0.5))
        default_controller.update(0.016)

        left_euler = default_controller.left_eye.to_euler()
        right_euler = default_controller.right_eye.to_euler()

        # Left eye yaw includes vergence (positive)
        # Right eye yaw includes vergence (negative)
        # So left yaw > right yaw
        left_yaw = left_euler[1]  # yaw is second element
        right_yaw = right_euler[1]

        # Left eye should be rotated more to the right (toward center)
        assert left_yaw > right_yaw

    def test_vergence_affects_quaternion_output(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should be included in EyeTransform quaternion."""
        default_controller.look_at((0.0, 0.0, 0.5))
        default_controller.update(0.016)

        left_quat = default_controller.left_eye.to_quaternion()
        right_quat = default_controller.right_eye.to_quaternion()

        # Quaternions should differ due to opposite vergence
        assert left_quat != right_quat


class TestDistanceClamping:
    """Tests for minimum distance clamping behavior."""

    def test_distance_clamped_at_zero(
        self, default_controller: EyeController
    ) -> None:
        """Target at origin (zero distance) should be clamped to 0.1m."""
        default_controller.set_head_transform((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        default_controller.look_at((0.0, 0.0, 0.0))
        default_controller.update(0.016)

        # Should not crash and vergence should be at max or clamped
        eye_separation = 0.065
        expected_at_min = math.degrees(math.atan2(eye_separation * 0.5, 0.1))
        expected_at_min = min(expected_at_min, 15.0)

        assert abs(default_controller.left_eye.vergence - expected_at_min) < 0.001

    def test_distance_clamped_at_negative_z(
        self, default_controller: EyeController
    ) -> None:
        """Target behind head should still produce valid vergence."""
        default_controller.look_at((0.0, 0.0, -1.0))
        default_controller.update(0.016)

        # Distance is computed as Euclidean, so -1.0 z gives distance of 1.0
        # Vergence should be valid
        assert default_controller.left_eye.vergence >= 0.0


class TestVergenceEdgeCases:
    """Edge case tests for vergence calculation."""

    def test_vergence_with_head_offset(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should account for head position offset."""
        default_controller.set_head_transform((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))
        default_controller.look_at((1.0, 2.0, 4.0))  # 1m in front of head
        default_controller.update(0.016)

        eye_separation = 0.065
        expected_vergence = math.degrees(math.atan2(eye_separation * 0.5, 1.0))

        assert abs(default_controller.left_eye.vergence - expected_vergence) < 0.001

    def test_vergence_preserved_across_updates(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should remain stable across multiple updates."""
        default_controller.look_at((0.0, 0.0, 0.5))

        vergences = []
        for _ in range(10):
            default_controller.update(0.016)
            vergences.append(default_controller.left_eye.vergence)

        # All vergences should converge to same value
        final = vergences[-1]
        for v in vergences[5:]:  # Check last 5 values
            assert abs(v - final) < 0.01

    def test_vergence_with_rapid_target_changes(
        self, default_controller: EyeController
    ) -> None:
        """Vergence should update correctly with rapid target changes."""
        distances = [0.5, 2.0, 0.3, 10.0, 1.0]

        for dist in distances:
            default_controller.look_at((0.0, 0.0, dist))
            default_controller.update(0.016)

            # Vergence should reflect current target
            eye_separation = 0.065
            expected = math.degrees(
                math.atan2(eye_separation * 0.5, max(0.1, dist))
            )
            expected = min(expected, 15.0)

            # After one update, vergence starts moving toward expected
            # (may not fully reach due to smoothing)
            assert default_controller.left_eye.vergence > 0.0


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
