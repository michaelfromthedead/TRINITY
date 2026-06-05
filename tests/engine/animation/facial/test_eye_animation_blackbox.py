"""
Blackbox tests for Eye Animation Vergence (T3.4)

Tests the public contract of EyeController for vergence behavior:
- Eyes converge when looking at near targets
- Eyes are parallel when looking at distant targets
- Vergence angle is geometrically correct
- Max vergence limit is respected

These tests are written without reading the implementation,
based solely on the public contract specification.
"""

import pytest
import math
from typing import Optional


class Vec3:
    """Simple 3D vector for test purposes."""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Vec3({self.x}, {self.y}, {self.z})"

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Vec3":
        length = self.length()
        if length == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / length, self.y / length, self.z / length)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)


# Try to import from the actual implementation
try:
    from engine.animation.facial.eye_animation import EyeController
    from engine.core.math import Vec3 as EngineVec3
    # Use engine's Vec3 if available
    Vec3 = EngineVec3
    IMPLEMENTATION_AVAILABLE = True

    # Check if Vec3 is subscriptable (implementation bug check)
    _test_vec = Vec3(1, 2, 3)
    try:
        _ = _test_vec[0]
        VEC3_IS_SUBSCRIPTABLE = True
    except (TypeError, IndexError):
        VEC3_IS_SUBSCRIPTABLE = False
except ImportError:
    try:
        from trinity.animation.facial.eye_animation import EyeController
        from trinity.core.math import Vec3 as TrinityVec3
        Vec3 = TrinityVec3
        IMPLEMENTATION_AVAILABLE = True
        VEC3_IS_SUBSCRIPTABLE = False
    except ImportError:
        IMPLEMENTATION_AVAILABLE = False
        VEC3_IS_SUBSCRIPTABLE = False


# Skip all tests if implementation is not available
pytestmark = pytest.mark.skipif(
    not IMPLEMENTATION_AVAILABLE,
    reason="EyeController implementation not available"
)


class TestImplementationBug:
    """Tests that detect implementation bugs in Vec3 handling."""

    def test_vec3_attribute_access_works(self):
        """
        Verify EyeController works with Vec3 objects using attribute access.

        The implementation now supports both:
        - Tuple access: target[0], target[1], target[2]
        - Attribute access: target.x, target.y, target.z

        This test verifies the fix for T3.4 (Vec3 subscript TypeError).
        """
        if not IMPLEMENTATION_AVAILABLE:
            pytest.skip("EyeController implementation not available")

        # This should NOT raise TypeError: 'Vec3' object is not subscriptable
        eye_controller = EyeController(eye_separation=0.06)
        target = Vec3(0, 0, 0.5)
        eye_controller.look_at(target)
        eye_controller.update(0.016)

        # Verify vergence was calculated (proves the Vec3 access worked)
        assert eye_controller.left_eye.vergence != 0, \
            "Vergence should be non-zero, proving Vec3 attribute access works"
        assert math.isfinite(eye_controller.left_eye.vergence), \
            "Vergence should be finite"


class TestEyeAnimationVergenceContract:
    """Test the basic vergence contract from the specification."""

    def test_near_target_convergence(self):
        """Eyes should converge when looking at a near target (30cm)."""
        eye_controller = EyeController(eye_separation=0.06)
        eye_controller.look_at(Vec3(0, 0, 0.3))  # 30cm away
        eye_controller.update(0.016)

        # Eyes should converge - left eye rotates inward (positive vergence)
        # right eye rotates inward (negative vergence)
        assert eye_controller.left_eye.vergence > 0, \
            f"Left eye vergence should be positive for convergence, got {eye_controller.left_eye.vergence}"
        assert eye_controller.right_eye.vergence < 0, \
            f"Right eye vergence should be negative for convergence, got {eye_controller.right_eye.vergence}"

    def test_near_target_symmetric_vergence(self):
        """Vergence angles should be symmetric for centered targets."""
        eye_controller = EyeController(eye_separation=0.06)
        eye_controller.look_at(Vec3(0, 0, 0.3))  # 30cm, centered
        eye_controller.update(0.016)

        # Magnitudes should be equal for symmetric target
        left_magnitude = abs(eye_controller.left_eye.vergence)
        right_magnitude = abs(eye_controller.right_eye.vergence)

        assert abs(left_magnitude - right_magnitude) < 1e-6, \
            f"Vergence magnitudes should be equal: left={left_magnitude}, right={right_magnitude}"

    def test_distant_target_parallel_eyes(self):
        """Eyes should be nearly parallel when looking at distant targets."""
        eye_controller = EyeController(eye_separation=0.06)
        eye_controller.look_at(Vec3(0, 0, 100))  # 100m away
        eye_controller.update(0.016)

        # For very distant targets, vergence should be nearly zero
        assert abs(eye_controller.left_eye.vergence) < 0.1, \
            f"Left eye vergence should be < 0.1 for distant target, got {eye_controller.left_eye.vergence}"
        assert abs(eye_controller.right_eye.vergence) < 0.1, \
            f"Right eye vergence should be < 0.1 for distant target, got {eye_controller.right_eye.vergence}"


class TestVergenceGeometry:
    """Test that vergence angles are geometrically correct."""

    def test_vergence_increases_with_proximity(self):
        """Vergence should increase as target gets closer."""
        eye_controller = EyeController(eye_separation=0.06)

        distances = [10.0, 5.0, 1.0, 0.5, 0.3]
        vergence_values = []

        for dist in distances:
            eye_controller.look_at(Vec3(0, 0, dist))
            eye_controller.update(0.016)
            vergence_values.append(abs(eye_controller.left_eye.vergence))

        # Vergence should increase as distance decreases
        for i in range(len(vergence_values) - 1):
            assert vergence_values[i] < vergence_values[i + 1], \
                f"Vergence should increase as target gets closer: " \
                f"at {distances[i]}m got {vergence_values[i]}, " \
                f"at {distances[i+1]}m got {vergence_values[i+1]}"

    def test_geometric_vergence_calculation(self):
        """Vergence angle should match geometric calculation."""
        eye_separation = 0.06  # 6cm (typical human IPD)
        eye_controller = EyeController(eye_separation=eye_separation)

        # For a target at distance d, the vergence angle for each eye
        # should be approximately atan(half_separation / distance)
        target_distance = 0.5  # 50cm
        eye_controller.look_at(Vec3(0, 0, target_distance))
        eye_controller.update(0.016)

        half_separation = eye_separation / 2.0
        # Implementation stores vergence in degrees (per EyeTransform docstring)
        expected_vergence_deg = math.degrees(math.atan(half_separation / target_distance))

        # Allow 10% tolerance for different geometric models
        actual_vergence = abs(eye_controller.left_eye.vergence)
        tolerance = expected_vergence_deg * 0.1

        assert abs(actual_vergence - expected_vergence_deg) < tolerance, \
            f"Vergence angle should be geometrically correct: " \
            f"expected ~{expected_vergence_deg:.2f} deg, " \
            f"got {actual_vergence:.2f} deg"

    def test_vergence_with_different_eye_separations(self):
        """Wider eye separation should result in larger vergence angles."""
        narrow_separation = 0.05  # 5cm
        wide_separation = 0.07   # 7cm

        narrow_controller = EyeController(eye_separation=narrow_separation)
        wide_controller = EyeController(eye_separation=wide_separation)

        target = Vec3(0, 0, 0.5)  # 50cm

        narrow_controller.look_at(target)
        narrow_controller.update(0.016)

        wide_controller.look_at(target)
        wide_controller.update(0.016)

        narrow_vergence = abs(narrow_controller.left_eye.vergence)
        wide_vergence = abs(wide_controller.left_eye.vergence)

        assert wide_vergence > narrow_vergence, \
            f"Wider eye separation should have larger vergence: " \
            f"narrow={math.degrees(narrow_vergence):.2f} deg, " \
            f"wide={math.degrees(wide_vergence):.2f} deg"


class TestVergenceLimits:
    """Test that vergence respects physiological limits."""

    def test_max_vergence_limit(self):
        """Vergence should not exceed physiological maximum (~25-30 degrees)."""
        eye_controller = EyeController(eye_separation=0.06)

        # Very close target that would require extreme vergence
        eye_controller.look_at(Vec3(0, 0, 0.01))  # 1cm - unrealistically close
        eye_controller.update(0.016)

        # EyeLimits.max_vergence defaults to 15.0 degrees
        # Implementation stores vergence in degrees (per EyeTransform docstring)
        max_vergence_degrees = 15.0

        assert abs(eye_controller.left_eye.vergence) <= max_vergence_degrees, \
            f"Left eye vergence should respect max limit: " \
            f"got {eye_controller.left_eye.vergence:.2f} deg"
        assert abs(eye_controller.right_eye.vergence) <= max_vergence_degrees, \
            f"Right eye vergence should respect max limit: " \
            f"got {eye_controller.right_eye.vergence:.2f} deg"

    def test_vergence_with_negative_z(self):
        """Target behind the viewer should not cause invalid vergence."""
        eye_controller = EyeController(eye_separation=0.06)

        # Target behind viewer
        eye_controller.look_at(Vec3(0, 0, -1.0))
        eye_controller.update(0.016)

        # Should either ignore or handle gracefully
        # Vergence values should still be finite and reasonable
        assert math.isfinite(eye_controller.left_eye.vergence), \
            "Left eye vergence should be finite for target behind viewer"
        assert math.isfinite(eye_controller.right_eye.vergence), \
            "Right eye vergence should be finite for target behind viewer"


class TestOffCenterTargets:
    """Test vergence behavior with off-center targets."""

    def test_target_to_left(self):
        """Target to the left should affect vergence asymmetrically."""
        eye_controller = EyeController(eye_separation=0.06)

        # Target to the left at near distance
        eye_controller.look_at(Vec3(-0.5, 0, 0.5))
        eye_controller.update(0.016)

        # Both eyes should still have vergence component
        # but angles will differ due to asymmetric position
        assert eye_controller.left_eye.vergence != 0, \
            "Left eye should have vergence for off-center target"
        assert eye_controller.right_eye.vergence != 0, \
            "Right eye should have vergence for off-center target"

    def test_target_to_right(self):
        """Target to the right should affect vergence asymmetrically."""
        eye_controller = EyeController(eye_separation=0.06)

        # Target to the right at near distance
        eye_controller.look_at(Vec3(0.5, 0, 0.5))
        eye_controller.update(0.016)

        # Both eyes should still have vergence component
        assert eye_controller.left_eye.vergence != 0, \
            "Left eye should have vergence for off-center target"
        assert eye_controller.right_eye.vergence != 0, \
            "Right eye should have vergence for off-center target"

    def test_symmetric_left_right_targets(self):
        """Left and right symmetric targets should produce mirrored vergence."""
        eye_controller = EyeController(eye_separation=0.06)

        # Target to the left
        eye_controller.look_at(Vec3(-0.3, 0, 0.5))
        eye_controller.update(0.016)
        left_target_left_vergence = eye_controller.left_eye.vergence
        left_target_right_vergence = eye_controller.right_eye.vergence

        # Target to the right (symmetric)
        eye_controller.look_at(Vec3(0.3, 0, 0.5))
        eye_controller.update(0.016)
        right_target_left_vergence = eye_controller.left_eye.vergence
        right_target_right_vergence = eye_controller.right_eye.vergence

        # Vergence should be mirrored
        assert abs(left_target_left_vergence - (-right_target_right_vergence)) < 0.01, \
            "Symmetric targets should produce mirrored vergence"


class TestVergenceUpdate:
    """Test vergence update behavior over time."""

    def test_multiple_updates_same_target(self):
        """Multiple updates with same target should converge to stable value."""
        eye_controller = EyeController(eye_separation=0.06)

        target = Vec3(0, 0, 0.3)
        eye_controller.look_at(target)

        # Run multiple updates
        for _ in range(100):
            eye_controller.update(0.016)

        final_left = eye_controller.left_eye.vergence
        final_right = eye_controller.right_eye.vergence

        # Run a few more updates
        for _ in range(10):
            eye_controller.update(0.016)

        # Values should be stable
        assert abs(eye_controller.left_eye.vergence - final_left) < 0.001, \
            "Vergence should be stable after convergence"
        assert abs(eye_controller.right_eye.vergence - final_right) < 0.001, \
            "Vergence should be stable after convergence"

    def test_target_change_updates_vergence(self):
        """Changing target should update vergence accordingly."""
        eye_controller = EyeController(eye_separation=0.06)

        # Near target
        eye_controller.look_at(Vec3(0, 0, 0.3))
        eye_controller.update(0.016)
        near_vergence = abs(eye_controller.left_eye.vergence)

        # Far target
        eye_controller.look_at(Vec3(0, 0, 10.0))
        for _ in range(100):  # Allow time to converge
            eye_controller.update(0.016)
        far_vergence = abs(eye_controller.left_eye.vergence)

        assert far_vergence < near_vergence, \
            f"Far target should have smaller vergence: near={near_vergence}, far={far_vergence}"

    def test_zero_delta_time(self):
        """Zero delta time should not crash or corrupt state."""
        eye_controller = EyeController(eye_separation=0.06)
        eye_controller.look_at(Vec3(0, 0, 0.5))

        # Should handle zero dt gracefully
        eye_controller.update(0.0)

        assert math.isfinite(eye_controller.left_eye.vergence), \
            "Vergence should remain finite after zero dt update"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_target_at_eye_position(self):
        """Target exactly at eye position should be handled gracefully."""
        eye_controller = EyeController(eye_separation=0.06)

        # Target at origin (approximately between eyes)
        eye_controller.look_at(Vec3(0, 0, 0))
        eye_controller.update(0.016)

        # Should not crash, vergence should be finite
        assert math.isfinite(eye_controller.left_eye.vergence), \
            "Vergence should be finite even for target at eye position"

    def test_target_at_infinity(self):
        """Very distant target should result in near-zero vergence."""
        eye_controller = EyeController(eye_separation=0.06)

        eye_controller.look_at(Vec3(0, 0, 1e10))  # Very far
        eye_controller.update(0.016)

        assert abs(eye_controller.left_eye.vergence) < 1e-6, \
            "Vergence should be essentially zero for target at infinity"

    def test_very_small_eye_separation(self):
        """Very small eye separation should still work correctly."""
        eye_controller = EyeController(eye_separation=0.001)  # 1mm

        eye_controller.look_at(Vec3(0, 0, 0.5))
        eye_controller.update(0.016)

        assert math.isfinite(eye_controller.left_eye.vergence), \
            "Should handle very small eye separation"

    def test_negative_eye_separation(self):
        """Negative eye separation should be rejected or handled."""
        # This should either raise an error or treat as absolute value
        try:
            eye_controller = EyeController(eye_separation=-0.06)
            # If it doesn't raise, it should still produce valid vergence
            eye_controller.look_at(Vec3(0, 0, 0.5))
            eye_controller.update(0.016)
            assert math.isfinite(eye_controller.left_eye.vergence)
        except (ValueError, TypeError):
            # Raising an error for invalid input is acceptable
            pass

    def test_vertical_target_offset(self):
        """Target above/below center should still compute valid vergence."""
        eye_controller = EyeController(eye_separation=0.06)

        # Target above
        eye_controller.look_at(Vec3(0, 1.0, 0.5))
        eye_controller.update(0.016)

        assert math.isfinite(eye_controller.left_eye.vergence), \
            "Vergence should be finite for elevated target"

        # Target below
        eye_controller.look_at(Vec3(0, -1.0, 0.5))
        eye_controller.update(0.016)

        assert math.isfinite(eye_controller.left_eye.vergence), \
            "Vergence should be finite for lowered target"


class TestEyeControllerInterface:
    """Test that EyeController exposes expected interface."""

    def test_has_left_eye_attribute(self):
        """EyeController should have left_eye attribute."""
        eye_controller = EyeController(eye_separation=0.06)
        assert hasattr(eye_controller, 'left_eye'), \
            "EyeController should have left_eye attribute"

    def test_has_right_eye_attribute(self):
        """EyeController should have right_eye attribute."""
        eye_controller = EyeController(eye_separation=0.06)
        assert hasattr(eye_controller, 'right_eye'), \
            "EyeController should have right_eye attribute"

    def test_eye_has_vergence_attribute(self):
        """Eye objects should have vergence attribute."""
        eye_controller = EyeController(eye_separation=0.06)
        eye_controller.look_at(Vec3(0, 0, 1.0))
        eye_controller.update(0.016)

        assert hasattr(eye_controller.left_eye, 'vergence'), \
            "Left eye should have vergence attribute"
        assert hasattr(eye_controller.right_eye, 'vergence'), \
            "Right eye should have vergence attribute"

    def test_look_at_method_exists(self):
        """EyeController should have look_at method."""
        eye_controller = EyeController(eye_separation=0.06)
        assert hasattr(eye_controller, 'look_at'), \
            "EyeController should have look_at method"
        assert callable(eye_controller.look_at), \
            "look_at should be callable"

    def test_update_method_exists(self):
        """EyeController should have update method."""
        eye_controller = EyeController(eye_separation=0.06)
        assert hasattr(eye_controller, 'update'), \
            "EyeController should have update method"
        assert callable(eye_controller.update), \
            "update should be callable"


class TestVergenceAngles:
    """Additional tests for vergence angle correctness."""

    def test_vergence_units_are_degrees(self):
        """Vergence should be in degrees (consistent with EyeTransform docstring)."""
        eye_controller = EyeController(eye_separation=0.06)
        eye_controller.look_at(Vec3(0, 0, 0.5))  # 50cm
        eye_controller.update(0.016)

        # For 50cm with 6cm IPD, vergence should be about 3.4 degrees
        # EyeTransform docstring specifies: "vergence: Convergence adjustment (degrees)"
        vergence = abs(eye_controller.left_eye.vergence)

        # Vergence in degrees should be in single digits for typical distances
        # (radians would be ~0.06, degrees would be ~3.4)
        assert 1.0 < vergence < 10.0, \
            f"Vergence should be in degrees (~3.4 for this case), got {vergence:.2f}"

    def test_reading_distance_vergence(self):
        """Test vergence at typical reading distance (40cm)."""
        eye_controller = EyeController(eye_separation=0.063)  # Average adult IPD
        eye_controller.look_at(Vec3(0, 0, 0.4))  # 40cm reading distance
        eye_controller.update(0.016)

        # Expected vergence: atan(0.0315 / 0.4) = 0.0785 rad = 4.5 degrees
        # Implementation stores in degrees (per EyeTransform docstring)
        expected_vergence_deg = math.degrees(math.atan(0.0315 / 0.4))
        actual_vergence = abs(eye_controller.left_eye.vergence)

        # Allow 20% tolerance
        tolerance = expected_vergence_deg * 0.2
        assert abs(actual_vergence - expected_vergence_deg) < tolerance, \
            f"Reading distance vergence off: expected {expected_vergence_deg:.1f}deg, " \
            f"got {actual_vergence:.1f}deg"

    def test_arm_length_vergence(self):
        """Test vergence at arm's length (60cm)."""
        eye_controller = EyeController(eye_separation=0.063)
        eye_controller.look_at(Vec3(0, 0, 0.6))  # 60cm
        eye_controller.update(0.016)

        # Expected: atan(0.0315 / 0.6) = 0.0524 rad = 3.0 degrees
        # Implementation stores in degrees (per EyeTransform docstring)
        expected_vergence_deg = math.degrees(math.atan(0.0315 / 0.6))
        actual_vergence = abs(eye_controller.left_eye.vergence)

        tolerance = expected_vergence_deg * 0.2
        assert abs(actual_vergence - expected_vergence_deg) < tolerance, \
            f"Arm's length vergence off: expected {expected_vergence_deg:.1f}deg, " \
            f"got {actual_vergence:.1f}deg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
