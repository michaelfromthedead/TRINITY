"""
Blackbox tests for BalanceController (T-FB-4.4).

CLEANROOM TEST SUITE - Written from specification only, without reading implementation.

BalanceController maintains character balance by:
- Checking if center of mass is within support polygon
- Computing correction vectors to restore balance
- Applying corrections to pelvis and spine bones
- Supporting configurable correction strength and bone weights

Public Interface (specification):
    @dataclass
    class BalanceController:
        com_calculator: COMCalculator
        support_polygon: SupportPolygon
        correction_strength: float  # 0-1
        pelvis_weight: float  # vs spine

        def is_balanced(self, bone_positions: Dict[str, Vec3]) -> bool
        def get_correction(self, bone_positions: Dict[str, Vec3]) -> Vec3
        def apply_correction(self, bone_positions, pelvis_name, spine_name) -> Dict[str, Vec3]
        def set_correction_strength(self, strength: float) -> None
        def update_support_polygon(self, foot_positions: List[Vec3]) -> None

Test Strategy:
- Test balance detection behavior (is_balanced)
- Test correction vector computation (get_correction)
- Test pose adjustment behavior (apply_correction)
- Test configuration methods
- Test realistic balance scenarios
"""

import math
import pytest
from typing import Dict, List

from engine.animation.ik.fullbody import BalanceController, COMCalculator, SupportPolygon
from engine.core.math.vec import Vec3


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def vec3_distance(a: Vec3, b: Vec3) -> float:
    """Calculate Euclidean distance between two Vec3 points."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def vec3_magnitude(v: Vec3) -> float:
    """Calculate magnitude of a vector."""
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def vec3_nearly_equal(a: Vec3, b: Vec3, eps: float = 1e-6) -> bool:
    """Check if two vectors are nearly equal within epsilon."""
    return vec3_distance(a, b) <= eps


def vec3_nearly_zero(v: Vec3, eps: float = 1e-6) -> bool:
    """Check if vector is nearly zero."""
    return vec3_magnitude(v) <= eps


def compute_simple_com(positions: Dict[str, Vec3]) -> Vec3:
    """Compute simple unweighted center of mass."""
    if not positions:
        return Vec3(0, 0, 0)

    sum_x = sum(p.x for p in positions.values())
    sum_y = sum(p.y for p in positions.values())
    sum_z = sum(p.z for p in positions.values())
    n = len(positions)

    return Vec3(sum_x / n, sum_y / n, sum_z / n)


def project_to_ground(v: Vec3) -> Vec3:
    """Project vector to ground plane (XZ, y=0)."""
    return Vec3(v.x, 0, v.z)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def com_calculator() -> COMCalculator:
    """Create a fresh COMCalculator instance."""
    return COMCalculator()


@pytest.fixture
def bipedal_support_polygon() -> SupportPolygon:
    """Support polygon for bipedal stance."""
    return SupportPolygon.from_foot_positions([
        Vec3(-0.15, 0, 0),  # Left foot
        Vec3(0.15, 0, 0)    # Right foot
    ])


@pytest.fixture
def wide_support_polygon() -> SupportPolygon:
    """Wide square support polygon for easy balance."""
    return SupportPolygon(vertices=[
        Vec3(-1, 0, -1),
        Vec3(1, 0, -1),
        Vec3(1, 0, 1),
        Vec3(-1, 0, 1)
    ])


@pytest.fixture
def narrow_support_polygon() -> SupportPolygon:
    """Narrow support polygon for difficult balance."""
    return SupportPolygon(vertices=[
        Vec3(-0.05, 0, -0.05),
        Vec3(0.05, 0, -0.05),
        Vec3(0.05, 0, 0.05),
        Vec3(-0.05, 0, 0.05)
    ])


@pytest.fixture
def balanced_standing_pose() -> Dict[str, Vec3]:
    """
    Humanoid skeleton in balanced standing pose.
    COM should be roughly above the feet.
    """
    return {
        "pelvis": Vec3(0.0, 1.0, 0.0),
        "spine": Vec3(0.0, 1.3, 0.0),
        "chest": Vec3(0.0, 1.5, 0.0),
        "head": Vec3(0.0, 1.8, 0.0),
        "left_shoulder": Vec3(-0.2, 1.5, 0.0),
        "right_shoulder": Vec3(0.2, 1.5, 0.0),
        "left_hand": Vec3(-0.4, 1.0, 0.0),
        "right_hand": Vec3(0.4, 1.0, 0.0),
        "left_hip": Vec3(-0.1, 1.0, 0.0),
        "right_hip": Vec3(0.1, 1.0, 0.0),
        "left_knee": Vec3(-0.1, 0.5, 0.0),
        "right_knee": Vec3(0.1, 0.5, 0.0),
        "left_foot": Vec3(-0.1, 0.0, 0.0),
        "right_foot": Vec3(0.1, 0.0, 0.0),
    }


@pytest.fixture
def leaning_right_pose() -> Dict[str, Vec3]:
    """
    Humanoid skeleton leaning heavily to the right.
    COM should be outside the support polygon.
    """
    return {
        "pelvis": Vec3(0.5, 1.0, 0.0),
        "spine": Vec3(0.6, 1.3, 0.0),
        "chest": Vec3(0.7, 1.5, 0.0),
        "head": Vec3(0.8, 1.8, 0.0),
        "left_shoulder": Vec3(0.5, 1.5, 0.0),
        "right_shoulder": Vec3(0.9, 1.5, 0.0),
        "left_hand": Vec3(0.4, 1.0, 0.0),
        "right_hand": Vec3(1.2, 1.0, 0.0),
        "left_hip": Vec3(0.4, 1.0, 0.0),
        "right_hip": Vec3(0.6, 1.0, 0.0),
        "left_knee": Vec3(0.0, 0.5, 0.0),
        "right_knee": Vec3(0.3, 0.5, 0.0),
        "left_foot": Vec3(-0.1, 0.0, 0.0),
        "right_foot": Vec3(0.1, 0.0, 0.0),
    }


@pytest.fixture
def leaning_forward_pose() -> Dict[str, Vec3]:
    """Humanoid skeleton leaning forward (positive Z)."""
    return {
        "pelvis": Vec3(0.0, 1.0, 0.3),
        "spine": Vec3(0.0, 1.3, 0.4),
        "chest": Vec3(0.0, 1.5, 0.5),
        "head": Vec3(0.0, 1.7, 0.6),
        "left_hip": Vec3(-0.1, 1.0, 0.3),
        "right_hip": Vec3(0.1, 1.0, 0.3),
        "left_foot": Vec3(-0.1, 0.0, 0.0),
        "right_foot": Vec3(0.1, 0.0, 0.0),
    }


@pytest.fixture
def slightly_unbalanced_pose() -> Dict[str, Vec3]:
    """Pose that is just slightly off balance."""
    return {
        "pelvis": Vec3(0.18, 1.0, 0.0),  # Slightly past right foot
        "spine": Vec3(0.18, 1.3, 0.0),
        "chest": Vec3(0.18, 1.5, 0.0),
        "head": Vec3(0.18, 1.8, 0.0),
        "left_foot": Vec3(-0.15, 0.0, 0.0),
        "right_foot": Vec3(0.15, 0.0, 0.0),
    }


@pytest.fixture
def default_balance_controller(
    com_calculator: COMCalculator,
    bipedal_support_polygon: SupportPolygon
) -> BalanceController:
    """Create a default balance controller with bipedal stance."""
    return BalanceController(
        com_calculator=com_calculator,
        support_polygon=bipedal_support_polygon,
        correction_strength=1.0,
        pelvis_weight=0.7
    )


@pytest.fixture
def wide_balance_controller(
    com_calculator: COMCalculator,
    wide_support_polygon: SupportPolygon
) -> BalanceController:
    """Create a balance controller with wide support."""
    return BalanceController(
        com_calculator=com_calculator,
        support_polygon=wide_support_polygon,
        correction_strength=1.0,
        pelvis_weight=0.7
    )


# =============================================================================
# TEST CLASS: Balance Detection Behavior
# =============================================================================

class TestBalanceBehavior:
    """Test is_balanced() behavior from specification."""

    def test_standing_straight_is_balanced(
        self,
        wide_balance_controller: BalanceController,
        balanced_standing_pose: Dict[str, Vec3]
    ):
        """
        A character standing straight with COM over support should be balanced.

        Specification: is_balanced returns True when COM is within support polygon.
        """
        result = wide_balance_controller.is_balanced(balanced_standing_pose)
        assert result is True, "Standing straight should be balanced"

    def test_leaning_too_far_is_unbalanced(
        self,
        default_balance_controller: BalanceController,
        leaning_right_pose: Dict[str, Vec3]
    ):
        """
        A character leaning far outside support polygon is unbalanced.

        Specification: is_balanced returns False when COM is outside support.
        """
        result = default_balance_controller.is_balanced(leaning_right_pose)
        assert result is False, "Leaning far right should be unbalanced"

    def test_leaning_forward_beyond_support_is_unbalanced(
        self,
        default_balance_controller: BalanceController,
        leaning_forward_pose: Dict[str, Vec3]
    ):
        """Leaning forward beyond feet should be unbalanced."""
        result = default_balance_controller.is_balanced(leaning_forward_pose)
        assert result is False, "Leaning forward beyond support should be unbalanced"

    def test_symmetric_pose_on_symmetric_support_is_balanced(
        self, com_calculator: COMCalculator
    ):
        """Symmetric pose centered on symmetric support should be balanced."""
        # Create perfectly symmetric scenario
        polygon = SupportPolygon(vertices=[
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(1, 0, 1),
            Vec3(-1, 0, 1)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Symmetric pose - COM at origin
        pose = {
            "left": Vec3(-0.5, 1.0, 0.0),
            "right": Vec3(0.5, 1.0, 0.0),
            "center": Vec3(0.0, 1.5, 0.0),
        }

        assert controller.is_balanced(pose) is True

    def test_com_at_edge_of_support(self, com_calculator: COMCalculator):
        """COM right at edge of support may or may not be balanced (edge case)."""
        # This tests boundary behavior - implementation may vary
        polygon = SupportPolygon(vertices=[
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(1, 0, 1),
            Vec3(0, 0, 1)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Pose with COM at center of support
        pose = {"bone": Vec3(0.5, 1.0, 0.5)}

        # Should be balanced when COM is clearly inside
        assert controller.is_balanced(pose) is True

    def test_empty_pose_handling(
        self, default_balance_controller: BalanceController
    ):
        """Empty bone positions should be handled gracefully."""
        # Empty dict - implementation should handle without crashing
        empty_pose: Dict[str, Vec3] = {}

        # Should not raise an exception
        try:
            result = default_balance_controller.is_balanced(empty_pose)
            # Result can be True or False depending on implementation
            assert isinstance(result, bool)
        except (ValueError, ZeroDivisionError):
            # Also acceptable to raise for invalid input
            pass

    def test_single_bone_pose(self, com_calculator: COMCalculator):
        """Single bone pose should work correctly."""
        polygon = SupportPolygon(vertices=[
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(1, 0, 1),
            Vec3(-1, 0, 1)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Single bone at center
        pose = {"only_bone": Vec3(0.0, 1.5, 0.0)}
        assert controller.is_balanced(pose) is True

        # Single bone outside support
        pose_outside = {"only_bone": Vec3(5.0, 1.5, 5.0)}
        assert controller.is_balanced(pose_outside) is False


# =============================================================================
# TEST CLASS: Correction Vector Behavior
# =============================================================================

class TestCorrectionBehavior:
    """Test get_correction() behavior from specification."""

    def test_no_correction_when_balanced(
        self,
        wide_balance_controller: BalanceController,
        balanced_standing_pose: Dict[str, Vec3]
    ):
        """
        Balanced pose should require no correction (or minimal).

        Specification: get_correction returns vector to restore balance.
        When balanced, this should be zero or near-zero.
        """
        correction = wide_balance_controller.get_correction(balanced_standing_pose)

        # Correction should be zero or very small
        magnitude = vec3_magnitude(correction)
        assert magnitude < 0.01, f"Balanced pose should need no correction, got {magnitude}"

    def test_correction_pushes_toward_support(
        self,
        default_balance_controller: BalanceController,
        leaning_right_pose: Dict[str, Vec3]
    ):
        """
        Correction vector should push COM back toward support polygon.

        If leaning right, correction should have negative X component
        (pushing left, toward center).
        """
        correction = default_balance_controller.get_correction(leaning_right_pose)

        # When leaning right, correction should push left (negative X)
        # The exact value depends on implementation, but direction should be correct
        assert correction.x < 0, (
            f"Leaning right should produce leftward correction, got x={correction.x}"
        )

    def test_correction_direction_for_forward_lean(
        self,
        default_balance_controller: BalanceController,
        leaning_forward_pose: Dict[str, Vec3]
    ):
        """Leaning forward should produce backward correction (negative Z)."""
        correction = default_balance_controller.get_correction(leaning_forward_pose)

        # When leaning forward (positive Z), correction should push back (negative Z)
        assert correction.z < 0, (
            f"Leaning forward should produce backward correction, got z={correction.z}"
        )

    def test_stronger_correction_with_higher_strength(
        self, com_calculator: COMCalculator, leaning_right_pose: Dict[str, Vec3]
    ):
        """
        Higher correction_strength should produce larger correction vectors.

        Specification: correction_strength is 0-1, affects correction magnitude.
        """
        polygon = SupportPolygon.from_foot_positions([
            Vec3(-0.15, 0, 0),
            Vec3(0.15, 0, 0)
        ])

        # Controller with low strength
        low_strength = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=0.3,
            pelvis_weight=0.7
        )

        # Controller with high strength
        high_strength = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.7
        )

        low_correction = low_strength.get_correction(leaning_right_pose)
        high_correction = high_strength.get_correction(leaning_right_pose)

        low_mag = vec3_magnitude(low_correction)
        high_mag = vec3_magnitude(high_correction)

        assert high_mag > low_mag, (
            f"Higher strength should produce larger correction: "
            f"high={high_mag}, low={low_mag}"
        )

    def test_zero_strength_produces_zero_correction(
        self, com_calculator: COMCalculator, leaning_right_pose: Dict[str, Vec3]
    ):
        """Zero correction strength should produce no correction."""
        polygon = SupportPolygon.from_foot_positions([
            Vec3(-0.15, 0, 0),
            Vec3(0.15, 0, 0)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=0.0,
            pelvis_weight=0.7
        )

        correction = controller.get_correction(leaning_right_pose)

        assert vec3_nearly_zero(correction), (
            f"Zero strength should produce zero correction, got {correction}"
        )

    def test_correction_magnitude_proportional_to_imbalance(
        self, com_calculator: COMCalculator
    ):
        """Larger imbalance should produce larger correction."""
        polygon = SupportPolygon(vertices=[
            Vec3(-0.5, 0, -0.5),
            Vec3(0.5, 0, -0.5),
            Vec3(0.5, 0, 0.5),
            Vec3(-0.5, 0, 0.5)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Small lean
        small_lean = {"bone": Vec3(0.6, 1.0, 0.0)}

        # Large lean
        large_lean = {"bone": Vec3(2.0, 1.0, 0.0)}

        small_correction = controller.get_correction(small_lean)
        large_correction = controller.get_correction(large_lean)

        small_mag = vec3_magnitude(small_correction)
        large_mag = vec3_magnitude(large_correction)

        assert large_mag > small_mag, (
            f"Larger imbalance should need larger correction: "
            f"large={large_mag}, small={small_mag}"
        )


# =============================================================================
# TEST CLASS: Pose Adjustment Behavior
# =============================================================================

class TestPoseAdjustment:
    """Test apply_correction() behavior from specification."""

    def test_pelvis_moves_more_than_spine(
        self,
        default_balance_controller: BalanceController,
        leaning_right_pose: Dict[str, Vec3]
    ):
        """
        Pelvis should receive more of the correction than spine.

        Specification: pelvis_weight controls distribution (pelvis vs spine).
        With pelvis_weight=0.7, pelvis gets 70% of correction.
        """
        original_pelvis = leaning_right_pose["pelvis"]
        original_spine = leaning_right_pose["spine"]

        corrected = default_balance_controller.apply_correction(
            leaning_right_pose,
            pelvis_name="pelvis",
            spine_name="spine"
        )

        pelvis_delta = vec3_distance(original_pelvis, corrected["pelvis"])
        spine_delta = vec3_distance(original_spine, corrected["spine"])

        # Pelvis should move more (weight = 0.7)
        assert pelvis_delta > spine_delta, (
            f"Pelvis should move more than spine: "
            f"pelvis_delta={pelvis_delta}, spine_delta={spine_delta}"
        )

    def test_y_position_unchanged(
        self,
        default_balance_controller: BalanceController,
        leaning_right_pose: Dict[str, Vec3]
    ):
        """
        Balance correction should only affect XZ plane, not height (Y).

        Vertical position should remain unchanged after correction.
        """
        original_pelvis_y = leaning_right_pose["pelvis"].y
        original_spine_y = leaning_right_pose["spine"].y

        corrected = default_balance_controller.apply_correction(
            leaning_right_pose,
            pelvis_name="pelvis",
            spine_name="spine"
        )

        corrected_pelvis_y = corrected["pelvis"].y
        corrected_spine_y = corrected["spine"].y

        assert abs(original_pelvis_y - corrected_pelvis_y) < 1e-6, (
            f"Pelvis Y should be unchanged: "
            f"original={original_pelvis_y}, corrected={corrected_pelvis_y}"
        )

        assert abs(original_spine_y - corrected_spine_y) < 1e-6, (
            f"Spine Y should be unchanged: "
            f"original={original_spine_y}, corrected={corrected_spine_y}"
        )

    def test_unaffected_bones_unchanged(
        self,
        default_balance_controller: BalanceController,
        leaning_right_pose: Dict[str, Vec3]
    ):
        """Bones other than pelvis/spine should remain at original positions."""
        original_head = leaning_right_pose["head"]
        original_left_foot = leaning_right_pose["left_foot"]

        corrected = default_balance_controller.apply_correction(
            leaning_right_pose,
            pelvis_name="pelvis",
            spine_name="spine"
        )

        # Head and feet should not be modified
        assert vec3_nearly_equal(original_head, corrected["head"]), (
            "Head position should be unchanged"
        )
        assert vec3_nearly_equal(original_left_foot, corrected["left_foot"]), (
            "Foot position should be unchanged"
        )

    def test_balanced_pose_remains_unchanged(
        self,
        wide_balance_controller: BalanceController,
        balanced_standing_pose: Dict[str, Vec3]
    ):
        """Applying correction to balanced pose should not change it."""
        original_pelvis = balanced_standing_pose["pelvis"]
        original_spine = balanced_standing_pose["spine"]

        corrected = wide_balance_controller.apply_correction(
            balanced_standing_pose,
            pelvis_name="pelvis",
            spine_name="spine"
        )

        # Positions should be nearly unchanged
        pelvis_delta = vec3_distance(original_pelvis, corrected["pelvis"])
        spine_delta = vec3_distance(original_spine, corrected["spine"])

        assert pelvis_delta < 0.01, (
            f"Balanced pelvis should barely move: delta={pelvis_delta}"
        )
        assert spine_delta < 0.01, (
            f"Balanced spine should barely move: delta={spine_delta}"
        )

    def test_correction_moves_toward_balance(
        self,
        default_balance_controller: BalanceController,
        leaning_right_pose: Dict[str, Vec3]
    ):
        """
        After correction, pose should be closer to balanced state.

        The corrected pose's COM should be closer to center of support.
        """
        # Get the correction direction
        correction = default_balance_controller.get_correction(leaning_right_pose)

        corrected = default_balance_controller.apply_correction(
            leaning_right_pose,
            pelvis_name="pelvis",
            spine_name="spine"
        )

        # If there was a correction needed (leaning right = negative X correction)
        # then the pelvis should have moved in that direction
        if correction.x < -0.001:  # Significant leftward correction
            original_pelvis_x = leaning_right_pose["pelvis"].x
            corrected_pelvis_x = corrected["pelvis"].x

            assert corrected_pelvis_x < original_pelvis_x, (
                "Pelvis should move left when correction is leftward"
            )

    def test_equal_weights_distribute_evenly(
        self, com_calculator: COMCalculator, leaning_right_pose: Dict[str, Vec3]
    ):
        """With pelvis_weight=0.5, both bones should move equally."""
        polygon = SupportPolygon.from_foot_positions([
            Vec3(-0.15, 0, 0),
            Vec3(0.15, 0, 0)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.5  # Equal weight
        )

        original_pelvis = leaning_right_pose["pelvis"]
        original_spine = leaning_right_pose["spine"]

        corrected = controller.apply_correction(
            leaning_right_pose,
            pelvis_name="pelvis",
            spine_name="spine"
        )

        pelvis_delta = vec3_distance(original_pelvis, corrected["pelvis"])
        spine_delta = vec3_distance(original_spine, corrected["spine"])

        # Deltas should be approximately equal (within tolerance)
        ratio = pelvis_delta / spine_delta if spine_delta > 0.001 else 1.0
        assert 0.8 < ratio < 1.2, (
            f"Equal weights should produce similar deltas: ratio={ratio}"
        )


# =============================================================================
# TEST CLASS: Configuration Methods
# =============================================================================

class TestConfigurationMethods:
    """Test configuration methods from specification."""

    def test_set_correction_strength_valid_range(
        self, default_balance_controller: BalanceController
    ):
        """set_correction_strength should accept values 0-1."""
        # Should not raise for valid values
        default_balance_controller.set_correction_strength(0.0)
        default_balance_controller.set_correction_strength(0.5)
        default_balance_controller.set_correction_strength(1.0)

    def test_set_correction_strength_affects_behavior(
        self,
        default_balance_controller: BalanceController,
        leaning_right_pose: Dict[str, Vec3]
    ):
        """Changing correction_strength should affect get_correction results."""
        # Get correction with default strength (1.0)
        correction_full = default_balance_controller.get_correction(leaning_right_pose)

        # Reduce strength
        default_balance_controller.set_correction_strength(0.5)
        correction_half = default_balance_controller.get_correction(leaning_right_pose)

        full_mag = vec3_magnitude(correction_full)
        half_mag = vec3_magnitude(correction_half)

        # Half strength should produce smaller correction
        assert half_mag < full_mag, (
            f"Half strength should produce smaller correction: "
            f"half={half_mag}, full={full_mag}"
        )

    def test_update_support_polygon_affects_balance(
        self, com_calculator: COMCalculator
    ):
        """update_support_polygon should change balance calculations."""
        # Start with narrow support
        narrow_poly = SupportPolygon(vertices=[
            Vec3(-0.1, 0, -0.1),
            Vec3(0.1, 0, -0.1),
            Vec3(0.1, 0, 0.1),
            Vec3(-0.1, 0, 0.1)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=narrow_poly,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Pose slightly outside narrow support
        pose = {"bone": Vec3(0.15, 1.0, 0.0)}

        # Should be unbalanced with narrow support
        assert controller.is_balanced(pose) is False

        # Update to wider support - must form a proper polygon (4 points)
        wider_positions = [
            Vec3(-0.5, 0, -0.1),
            Vec3(0.5, 0, -0.1),
            Vec3(0.5, 0, 0.1),
            Vec3(-0.5, 0, 0.1)
        ]
        controller.update_support_polygon(wider_positions)

        # Now should be balanced
        assert controller.is_balanced(pose) is True

    def test_update_support_polygon_with_new_foot_positions(
        self, com_calculator: COMCalculator
    ):
        """Update support polygon should work with foot position list."""
        # Initial narrow polygon - must form proper area
        initial_poly = SupportPolygon(vertices=[
            Vec3(-0.1, 0, -0.1),
            Vec3(0.1, 0, -0.1),
            Vec3(0.1, 0, 0.1),
            Vec3(-0.1, 0, 0.1)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=initial_poly,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Pose outside narrow support
        pose = {"bone": Vec3(0.5, 1.0, 0.0)}
        assert controller.is_balanced(pose) is False

        # Move feet to new positions - wider support polygon
        new_feet = [
            Vec3(-1, 0, -0.1),
            Vec3(1, 0, -0.1),
            Vec3(1, 0, 0.1),
            Vec3(-1, 0, 0.1)
        ]
        controller.update_support_polygon(new_feet)

        # Now pose should be balanced
        assert controller.is_balanced(pose) is True


# =============================================================================
# TEST CLASS: Balance Scenarios
# =============================================================================

class TestBalanceScenarios:
    """Test realistic balance scenarios from specification."""

    def test_bipedal_stance_balance(self, com_calculator: COMCalculator):
        """
        Standard bipedal stance should maintain balance.

        Two feet side by side create a support polygon.
        Character standing straight should be balanced.
        """
        polygon = SupportPolygon.from_foot_positions([
            Vec3(-0.15, 0, 0.1),   # Left foot toe
            Vec3(-0.15, 0, -0.1),  # Left foot heel
            Vec3(0.15, 0, 0.1),    # Right foot toe
            Vec3(0.15, 0, -0.1),   # Right foot heel
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.7
        )

        # Standing straight pose
        standing = {
            "pelvis": Vec3(0.0, 1.0, 0.0),
            "spine": Vec3(0.0, 1.3, 0.0),
            "chest": Vec3(0.0, 1.5, 0.0),
        }

        assert controller.is_balanced(standing) is True

    def test_recover_from_lean(self, com_calculator: COMCalculator):
        """
        Character should be able to recover from a lean.

        Apply correction to bring COM back over support.
        """
        # Use explicit vertices for proper polygon
        polygon = SupportPolygon(vertices=[
            Vec3(-0.2, 0, -0.1),
            Vec3(0.2, 0, -0.1),
            Vec3(0.2, 0, 0.1),
            Vec3(-0.2, 0, 0.1),
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.7
        )

        # Leaning pose - only upper body bones (not feet) to keep COM outside
        leaning = {
            "pelvis": Vec3(0.5, 1.0, 0.0),
            "spine": Vec3(0.55, 1.3, 0.0),
        }

        # Should be unbalanced - COM is at ~(0.525, 1.15, 0) which is outside [-0.2, 0.2]
        assert controller.is_balanced(leaning) is False

        # Apply correction
        corrected = controller.apply_correction(
            leaning,
            pelvis_name="pelvis",
            spine_name="spine"
        )

        # After correction, should be closer to balanced
        # The pelvis should have moved toward center
        assert corrected["pelvis"].x < leaning["pelvis"].x, (
            "Pelvis should move toward center"
        )

    def test_one_foot_stance(self, com_calculator: COMCalculator):
        """Single foot stance has smaller support polygon."""
        # One foot only - use explicit vertices in proper winding order
        polygon = SupportPolygon(vertices=[
            Vec3(-0.05, 0, -0.1),  # Heel inner
            Vec3(0.05, 0, -0.1),   # Heel outer
            Vec3(0.05, 0, 0.1),    # Toe outer
            Vec3(-0.05, 0, 0.1),   # Toe inner
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # COM directly over foot
        balanced = {"pelvis": Vec3(0.0, 1.0, 0.0)}
        assert controller.is_balanced(balanced) is True

        # COM slightly outside tiny support
        unbalanced = {"pelvis": Vec3(0.15, 1.0, 0.0)}
        assert controller.is_balanced(unbalanced) is False

    def test_wide_stance_more_stable(self, com_calculator: COMCalculator):
        """Wider stance should provide more stability."""
        # Narrow stance - proper polygon with area
        narrow = SupportPolygon(vertices=[
            Vec3(-0.1, 0, -0.05),
            Vec3(0.1, 0, -0.05),
            Vec3(0.1, 0, 0.05),
            Vec3(-0.1, 0, 0.05),
        ])

        # Wide stance - proper polygon with area
        wide = SupportPolygon(vertices=[
            Vec3(-0.5, 0, -0.05),
            Vec3(0.5, 0, -0.05),
            Vec3(0.5, 0, 0.05),
            Vec3(-0.5, 0, 0.05),
        ])

        narrow_ctrl = BalanceController(
            com_calculator=com_calculator,
            support_polygon=narrow,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        wide_ctrl = BalanceController(
            com_calculator=com_calculator,
            support_polygon=wide,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Pose at x=0.3
        pose = {"pelvis": Vec3(0.3, 1.0, 0.0)}

        # Narrow should be unbalanced, wide should be balanced
        assert narrow_ctrl.is_balanced(pose) is False
        assert wide_ctrl.is_balanced(pose) is True

    def test_tripod_stance_stability(self, com_calculator: COMCalculator):
        """
        Tripod stance (two feet + cane/third point) provides larger support.
        """
        # Two feet + cane
        tripod = SupportPolygon.from_foot_positions([
            Vec3(-0.15, 0, 0),   # Left foot
            Vec3(0.15, 0, 0),    # Right foot
            Vec3(0, 0, 0.4),     # Cane in front
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=tripod,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Leaning forward should still be balanced with tripod
        forward_lean = {"pelvis": Vec3(0.0, 1.0, 0.2)}
        assert controller.is_balanced(forward_lean) is True

    def test_quadruped_stance(self, com_calculator: COMCalculator):
        """Quadruped (four-legged) stance should have large stable area."""
        # Four corners forming a proper convex polygon
        quadruped = SupportPolygon(vertices=[
            Vec3(-0.3, 0, -0.4),   # Front left
            Vec3(0.3, 0, -0.4),    # Front right
            Vec3(0.3, 0, 0.4),     # Back right
            Vec3(-0.3, 0, 0.4),    # Back left
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=quadruped,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Various positions within quadruped support
        positions_to_test = [
            Vec3(0.0, 1.0, 0.0),    # Center
            Vec3(0.2, 1.0, 0.0),    # Slight right
            Vec3(-0.2, 1.0, 0.0),   # Slight left
            Vec3(0.0, 1.0, 0.3),    # Forward
            Vec3(0.0, 1.0, -0.3),   # Back
        ]

        for pos in positions_to_test:
            pose = {"pelvis": pos}
            assert controller.is_balanced(pose) is True, (
                f"Quadruped should be balanced at {pos}"
            )


# =============================================================================
# TEST CLASS: Edge Cases and Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_high_pelvis_weight(self, com_calculator: COMCalculator):
        """Pelvis weight near 1.0 should put almost all correction on pelvis."""
        polygon = SupportPolygon.from_foot_positions([
            Vec3(-0.15, 0, 0),
            Vec3(0.15, 0, 0)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.95  # Almost all on pelvis
        )

        leaning = {
            "pelvis": Vec3(0.5, 1.0, 0.0),
            "spine": Vec3(0.5, 1.3, 0.0),
        }

        original_pelvis = leaning["pelvis"]
        original_spine = leaning["spine"]

        corrected = controller.apply_correction(leaning, "pelvis", "spine")

        pelvis_delta = vec3_distance(original_pelvis, corrected["pelvis"])
        spine_delta = vec3_distance(original_spine, corrected["spine"])

        # Pelvis should move much more than spine
        if spine_delta > 0.001:  # Avoid division by zero
            ratio = pelvis_delta / spine_delta
            assert ratio > 5, f"High pelvis weight should move pelvis much more: ratio={ratio}"

    def test_very_low_pelvis_weight(self, com_calculator: COMCalculator):
        """Pelvis weight near 0.0 should put almost all correction on spine."""
        polygon = SupportPolygon.from_foot_positions([
            Vec3(-0.15, 0, 0),
            Vec3(0.15, 0, 0)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.05  # Almost all on spine
        )

        leaning = {
            "pelvis": Vec3(0.5, 1.0, 0.0),
            "spine": Vec3(0.5, 1.3, 0.0),
        }

        original_pelvis = leaning["pelvis"]
        original_spine = leaning["spine"]

        corrected = controller.apply_correction(leaning, "pelvis", "spine")

        pelvis_delta = vec3_distance(original_pelvis, corrected["pelvis"])
        spine_delta = vec3_distance(original_spine, corrected["spine"])

        # Spine should move much more than pelvis
        if pelvis_delta > 0.001:  # Avoid division by zero
            ratio = spine_delta / pelvis_delta
            assert ratio > 5, f"Low pelvis weight should move spine much more: ratio={ratio}"

    def test_correction_with_missing_bone_names(
        self, default_balance_controller: BalanceController
    ):
        """apply_correction with non-existent bone names should handle gracefully."""
        pose = {
            "existing_bone": Vec3(0.5, 1.0, 0.0),
        }

        # Using bone names that don't exist in pose
        try:
            result = default_balance_controller.apply_correction(
                pose,
                pelvis_name="nonexistent_pelvis",
                spine_name="nonexistent_spine"
            )
            # If it doesn't raise, check that original bones are preserved
            assert "existing_bone" in result
        except (KeyError, ValueError):
            # Also acceptable to raise for missing bones
            pass

    def test_extreme_imbalance(self, com_calculator: COMCalculator):
        """Extremely unbalanced pose should still produce valid correction."""
        polygon = SupportPolygon(vertices=[
            Vec3(-0.5, 0, -0.5),
            Vec3(0.5, 0, -0.5),
            Vec3(0.5, 0, 0.5),
            Vec3(-0.5, 0, 0.5)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Extremely far from support
        extreme_pose = {
            "pelvis": Vec3(100.0, 1.0, 100.0),
            "spine": Vec3(100.0, 1.3, 100.0),
        }

        assert controller.is_balanced(extreme_pose) is False

        correction = controller.get_correction(extreme_pose)

        # Correction should be finite and point toward support
        assert math.isfinite(correction.x)
        assert math.isfinite(correction.y)
        assert math.isfinite(correction.z)

        # Should point back toward origin (negative X and Z)
        assert correction.x < 0, "Should correct toward support (negative X)"
        assert correction.z < 0, "Should correct toward support (negative Z)"

    def test_bones_at_same_position(self, com_calculator: COMCalculator):
        """All bones at same position should still work."""
        polygon = SupportPolygon(vertices=[
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(1, 0, 1),
            Vec3(-1, 0, 1)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # All bones at same spot
        coincident = {
            "pelvis": Vec3(0.0, 1.0, 0.0),
            "spine": Vec3(0.0, 1.0, 0.0),
            "chest": Vec3(0.0, 1.0, 0.0),
        }

        # Should be balanced (COM at center)
        assert controller.is_balanced(coincident) is True

    def test_negative_coordinates_all_around(self, com_calculator: COMCalculator):
        """Support polygon and pose in negative coordinate space should work."""
        polygon = SupportPolygon(vertices=[
            Vec3(-3, 0, -3),
            Vec3(-1, 0, -3),
            Vec3(-1, 0, -1),
            Vec3(-3, 0, -1)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=1.0,
            pelvis_weight=0.5
        )

        # Pose in negative space, centered in polygon
        pose = {"pelvis": Vec3(-2.0, 1.0, -2.0)}

        assert controller.is_balanced(pose) is True

        # Pose outside negative polygon
        outside = {"pelvis": Vec3(0.0, 1.0, 0.0)}
        assert controller.is_balanced(outside) is False


# =============================================================================
# TEST CLASS: Dataclass Behavior
# =============================================================================

class TestDataclassBehavior:
    """Test BalanceController dataclass structure."""

    def test_initialization_with_required_fields(self, com_calculator: COMCalculator):
        """BalanceController should initialize with required fields."""
        polygon = SupportPolygon(vertices=[
            Vec3(-1, 0, -1),
            Vec3(1, 0, -1),
            Vec3(1, 0, 1),
            Vec3(-1, 0, 1)
        ])

        controller = BalanceController(
            com_calculator=com_calculator,
            support_polygon=polygon,
            correction_strength=0.8,
            pelvis_weight=0.6
        )

        assert controller.com_calculator is com_calculator
        assert controller.support_polygon is polygon
        assert controller.correction_strength == 0.8
        assert controller.pelvis_weight == 0.6

    def test_correction_strength_attribute_accessible(
        self, default_balance_controller: BalanceController
    ):
        """correction_strength attribute should be accessible."""
        # Default is 1.0 based on fixture
        assert default_balance_controller.correction_strength == 1.0

        # After setting
        default_balance_controller.set_correction_strength(0.5)
        assert default_balance_controller.correction_strength == 0.5

    def test_pelvis_weight_attribute_accessible(
        self, default_balance_controller: BalanceController
    ):
        """pelvis_weight attribute should be accessible."""
        # Default is 0.7 based on fixture
        assert default_balance_controller.pelvis_weight == 0.7
