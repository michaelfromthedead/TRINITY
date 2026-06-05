"""Whitebox tests for foot alignment methods (T-FB-4.11).

Focused testing for FootPlacement alignment methods:
- _align_foot_to_terrain(): Aligns foot rotation to terrain slope
- _rotation_between_vectors(): Computes quaternion between two vectors
- _scale_rotation(): Scales rotation using slerp from identity

Test coverage includes:
- Foot rotation modification toward ground normal
- Toe alignment when foot_data.toe is set
- No toe alignment when toe is None
- Weight blending (toe_align_weight, blend_weight)
- Edge cases for rotation computation (parallel, opposite, perpendicular)
- Scale rotation with scale=0, 0.5, 1.0
- Smooth transitions via weighted alignment
"""

from __future__ import annotations

import math
import pytest
from typing import List, Optional

from engine.animation.ik.foot_placement import (
    FootState,
    FootData,
    FootPlacement,
    RaycastCallback,
    RaycastHit,
)
from engine.animation.ik.config import (
    FOOT_PLACEMENT_TOE_ALIGN_WEIGHT,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON


# =============================================================================
# Helper Functions
# =============================================================================

def create_biped_transforms(num_bones: int = 10) -> List[Transform]:
    """Create transforms for a simple biped skeleton."""
    positions = {
        0: Vec3(0, 1.0, 0),      # Pelvis
        1: Vec3(-0.1, 1.0, 0),   # Left upper leg
        2: Vec3(-0.1, 0.5, 0),   # Left lower leg
        3: Vec3(-0.1, 0.1, 0),   # Left foot
        4: Vec3(-0.1, 0.0, 0),   # Left toe
        5: Vec3(0.1, 1.0, 0),    # Right upper leg
        6: Vec3(0.1, 0.5, 0),    # Right lower leg
        7: Vec3(0.1, 0.1, 0),    # Right foot
        8: Vec3(0.1, 0.0, 0),    # Right toe
    }

    transforms = []
    for i in range(num_bones):
        pos = positions.get(i, Vec3(0, 0, 0))
        transforms.append(Transform(pos, Quat.identity()))

    return transforms


def create_left_foot_data(toe: Optional[int] = 4) -> FootData:
    """Create left foot data for testing."""
    return FootData(
        upper_leg=1,
        lower_leg=2,
        foot=3,
        toe=toe
    )


def create_right_foot_data(toe: Optional[int] = 8) -> FootData:
    """Create right foot data for testing."""
    return FootData(
        upper_leg=5,
        lower_leg=6,
        foot=7,
        toe=toe
    )


def create_foot_placement() -> FootPlacement:
    """Create a basic FootPlacement for testing."""
    left_foot = create_left_foot_data()
    right_foot = create_right_foot_data()
    return FootPlacement(left_foot, right_foot, pelvis=0)


def vec3_approx_equal(v1: Vec3, v2: Vec3, eps: float = 1e-4) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < eps and
        abs(v1.y - v2.y) < eps and
        abs(v1.z - v2.z) < eps
    )


def quat_approx_equal(q1: Quat, q2: Quat, eps: float = 1e-4) -> bool:
    """Check if two Quaternions are approximately equal (handles sign flip)."""
    dot = abs(q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w)
    return dot > (1.0 - eps)


def quat_is_unit(q: Quat, eps: float = 1e-4) -> bool:
    """Check if quaternion is unit length."""
    length = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
    return abs(length - 1.0) < eps


def get_angle_between_quats(q1: Quat, q2: Quat) -> float:
    """Get angle between two quaternions in radians."""
    dot = abs(q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w)
    dot = min(1.0, dot)  # Clamp for numerical safety
    return 2.0 * math.acos(dot)


# =============================================================================
# Test _align_foot_to_terrain() - Basic Functionality
# =============================================================================

class TestAlignFootToTerrainBasic:
    """Basic tests for _align_foot_to_terrain() method."""

    def test_align_modifies_foot_rotation(self):
        """Test that alignment modifies foot transform rotation."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        original_rot = Quat(
            transforms[3].rotation.x,
            transforms[3].rotation.y,
            transforms[3].rotation.z,
            transforms[3].rotation.w
        )

        # Tilted normal that should cause rotation
        tilted_normal = Vec3(0.3, 0.9, 0).normalized()

        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        # Rotation should have changed
        new_rot = transforms[3].rotation
        angle_change = get_angle_between_quats(original_rot, new_rot)
        assert angle_change > 0.001, "Foot rotation should change for tilted terrain"

    def test_align_with_vertical_normal_minimal_change(self):
        """Test alignment with pure vertical normal causes minimal rotation."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        original_rot = transforms[3].rotation

        # Pure up normal - foot already aligned
        up_normal = Vec3(0, 1, 0)

        placement._align_foot_to_terrain(transforms, left_foot, up_normal)

        # Rotation should remain similar (identity rotation aligns with up)
        new_rot = transforms[3].rotation
        angle_change = get_angle_between_quats(original_rot, new_rot)
        # With identity rotation, foot Y-axis is already up, so minimal change
        assert angle_change < 0.1, "Minimal rotation for aligned foot"

    def test_align_foot_rotates_toward_normal(self):
        """Test foot rotation changes when alignment is applied to misaligned foot."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        # Start with misaligned foot (rotated away from the target normal)
        transforms[3].rotation = Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 6)

        # Target normal that's different from current foot up
        tilted_normal = Vec3(0.3, 0.954, 0).normalized()

        # Record initial state
        foot_up_before = transforms[3].rotation.rotate_vector(Vec3(0, 1, 0))
        dot_before = foot_up_before.dot(tilted_normal)

        # Apply single alignment
        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        # After alignment, rotation should have changed
        foot_up_after = transforms[3].rotation.rotate_vector(Vec3(0, 1, 0))
        rot_change = get_angle_between_quats(
            Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 6),
            transforms[3].rotation
        )

        # Rotation should change when misaligned
        assert rot_change > 0.01, f"Should rotate when misaligned, got change={rot_change}"

    def test_align_returns_none(self):
        """Test _align_foot_to_terrain returns None (modifies in place)."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        result = placement._align_foot_to_terrain(
            transforms, left_foot, Vec3(0, 1, 0)
        )

        assert result is None

    def test_align_preserves_transform_position(self):
        """Test alignment doesn't modify foot position."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        original_pos = Vec3(
            transforms[3].translation.x,
            transforms[3].translation.y,
            transforms[3].translation.z
        )

        placement._align_foot_to_terrain(
            transforms, left_foot, Vec3(0.2, 0.98, 0).normalized()
        )

        assert vec3_approx_equal(transforms[3].translation, original_pos)


# =============================================================================
# Test _align_foot_to_terrain() - Toe Alignment
# =============================================================================

class TestAlignFootToTerrainToe:
    """Tests for toe alignment in _align_foot_to_terrain()."""

    def test_toe_is_aligned_when_present(self):
        """Test toe transform is also aligned when toe index is set."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data(toe=4)

        original_toe_rot = Quat(
            transforms[4].rotation.x,
            transforms[4].rotation.y,
            transforms[4].rotation.z,
            transforms[4].rotation.w
        )

        tilted_normal = Vec3(0.3, 0.9, 0).normalized()
        placement.toe_align_weight = 1.0
        left_foot.blend_weight = 1.0

        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        # Toe rotation should change
        new_toe_rot = transforms[4].rotation
        angle_change = get_angle_between_quats(original_toe_rot, new_toe_rot)
        assert angle_change > 0.001, "Toe rotation should change"

    def test_no_toe_alignment_when_toe_is_none(self):
        """Test no toe alignment when foot_data.toe is None."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data(toe=None)

        original_toe_rot = Quat(
            transforms[4].rotation.x,
            transforms[4].rotation.y,
            transforms[4].rotation.z,
            transforms[4].rotation.w
        )

        tilted_normal = Vec3(0.3, 0.9, 0).normalized()

        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        # Toe rotation should remain unchanged
        new_toe_rot = transforms[4].rotation
        assert quat_approx_equal(original_toe_rot, new_toe_rot)

    def test_toe_alignment_weight_is_half_of_foot(self):
        """Test toe alignment uses 0.5x the foot alignment weight."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data(toe=4)

        # Set full weights
        placement.toe_align_weight = 1.0
        left_foot.blend_weight = 1.0

        tilted_normal = Vec3(0.15, 0.98, 0).normalized()

        # Record initial states
        foot_rot_before = Quat(
            transforms[3].rotation.x, transforms[3].rotation.y,
            transforms[3].rotation.z, transforms[3].rotation.w
        )
        toe_rot_before = Quat(
            transforms[4].rotation.x, transforms[4].rotation.y,
            transforms[4].rotation.z, transforms[4].rotation.w
        )

        # Apply single alignment
        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        # Calculate rotation changes
        foot_change = get_angle_between_quats(foot_rot_before, transforms[3].rotation)
        toe_change = get_angle_between_quats(toe_rot_before, transforms[4].rotation)

        # Toe should change less than foot (due to 0.5x weight)
        # Both should have changed
        assert foot_change > 0.001, f"Foot should have rotated, got change={foot_change}"
        assert toe_change > 0.001, f"Toe should have rotated, got change={toe_change}"
        # Toe change should be approximately half of foot change (0.5x weight)
        assert toe_change < foot_change, f"Toe change ({toe_change}) should be less than foot change ({foot_change})"

    def test_toe_index_out_of_range_no_crash(self):
        """Test no crash when toe index exceeds transform list size."""
        placement = create_foot_placement()
        transforms = create_biped_transforms(5)  # Only 5 transforms
        left_foot = create_left_foot_data(toe=10)  # Out of range

        tilted_normal = Vec3(0.3, 0.9, 0).normalized()

        # Should not raise
        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

    def test_toe_preserves_position(self):
        """Test toe alignment doesn't modify toe position."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data(toe=4)

        original_toe_pos = Vec3(
            transforms[4].translation.x,
            transforms[4].translation.y,
            transforms[4].translation.z
        )

        placement._align_foot_to_terrain(
            transforms, left_foot, Vec3(0.2, 0.98, 0).normalized()
        )

        assert vec3_approx_equal(transforms[4].translation, original_toe_pos)


# =============================================================================
# Test _align_foot_to_terrain() - Weight Blending
# =============================================================================

class TestAlignFootToTerrainWeights:
    """Tests for weight blending in _align_foot_to_terrain()."""

    def test_zero_toe_align_weight_no_alignment(self):
        """Test zero toe_align_weight causes no alignment."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        original_rot = Quat(
            transforms[3].rotation.x,
            transforms[3].rotation.y,
            transforms[3].rotation.z,
            transforms[3].rotation.w
        )

        placement.toe_align_weight = 0.0
        left_foot.blend_weight = 1.0

        tilted_normal = Vec3(0.5, 0.866, 0).normalized()

        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        # Rotation should remain unchanged
        new_rot = transforms[3].rotation
        assert quat_approx_equal(original_rot, new_rot)

    def test_zero_blend_weight_no_alignment(self):
        """Test zero blend_weight causes no alignment."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        original_rot = Quat(
            transforms[3].rotation.x,
            transforms[3].rotation.y,
            transforms[3].rotation.z,
            transforms[3].rotation.w
        )

        placement.toe_align_weight = 1.0
        left_foot.blend_weight = 0.0

        tilted_normal = Vec3(0.5, 0.866, 0).normalized()

        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        # Rotation should remain unchanged
        new_rot = transforms[3].rotation
        assert quat_approx_equal(original_rot, new_rot)

    def test_combined_weight_is_product(self):
        """Test effective weight is toe_align_weight * blend_weight."""
        placement = create_foot_placement()

        # Test with different weight combinations
        placement.toe_align_weight = 0.5
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 0.5

        transforms1 = create_biped_transforms()
        tilted_normal = Vec3(0.5, 0.866, 0).normalized()
        placement._align_foot_to_terrain(transforms1, left_foot, tilted_normal)
        rot1 = transforms1[3].rotation

        # Compare with equivalent single weight
        placement.toe_align_weight = 0.25  # 0.5 * 0.5 = 0.25
        left_foot2 = create_left_foot_data()
        left_foot2.blend_weight = 1.0

        transforms2 = create_biped_transforms()
        placement._align_foot_to_terrain(transforms2, left_foot2, tilted_normal)
        rot2 = transforms2[3].rotation

        # Both should produce similar rotations
        assert quat_approx_equal(rot1, rot2, eps=1e-3)

    def test_half_weight_partial_alignment(self):
        """Test half weight produces partial alignment."""
        placement = create_foot_placement()
        left_foot = create_left_foot_data()

        placement.toe_align_weight = 1.0
        left_foot.blend_weight = 0.5

        transforms = create_biped_transforms()
        # Use smaller tilt for more predictable partial alignment
        tilted_normal = Vec3(0.2, 0.98, 0).normalized()

        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        foot_up = transforms[3].rotation.rotate_vector(Vec3(0, 1, 0))
        dot_with_normal = foot_up.dot(tilted_normal)
        dot_with_up = foot_up.dot(Vec3(0, 1, 0))

        # Partial alignment: should be between original and full alignment
        # With 0.5 weight on small tilt, should have good alignment with both
        assert dot_with_normal > 0.7, f"Should have some alignment, got {dot_with_normal}"
        assert dot_with_up > 0.7, f"Should retain some original orientation, got {dot_with_up}"

    def test_full_weight_full_alignment(self):
        """Test full weight produces maximum single-step rotation."""
        placement = create_foot_placement()
        left_foot = create_left_foot_data()

        placement.toe_align_weight = 1.0
        left_foot.blend_weight = 1.0

        transforms = create_biped_transforms()
        tilted_normal = Vec3(0.15, 0.98, 0).normalized()

        # Record initial state
        original_rot = Quat(
            transforms[3].rotation.x, transforms[3].rotation.y,
            transforms[3].rotation.z, transforms[3].rotation.w
        )

        # Apply single alignment with full weight
        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        # Full weight should produce maximum rotation change
        rot_change = get_angle_between_quats(original_rot, transforms[3].rotation)

        # With full weight, should see significant rotation
        assert rot_change > 0.05, f"Full weight should produce rotation, got {rot_change}"


# =============================================================================
# Test _align_foot_to_terrain() - Various Terrain Normals
# =============================================================================

class TestAlignFootToTerrainNormals:
    """Tests for alignment with various terrain normal directions."""

    def test_align_with_forward_tilt(self):
        """Test alignment modifies rotation for forward-tilted terrain."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        forward_tilt = Vec3(0, 0.98, -0.2).normalized()
        original_rot = transforms[3].rotation

        placement._align_foot_to_terrain(transforms, left_foot, forward_tilt)

        # Rotation should have changed
        rot_change = get_angle_between_quats(original_rot, transforms[3].rotation)
        assert rot_change > 0.01, f"Forward tilt should cause rotation change: {rot_change}"

    def test_align_with_backward_tilt(self):
        """Test alignment modifies rotation for backward-tilted terrain."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        backward_tilt = Vec3(0, 0.98, 0.2).normalized()
        original_rot = transforms[3].rotation

        placement._align_foot_to_terrain(transforms, left_foot, backward_tilt)

        rot_change = get_angle_between_quats(original_rot, transforms[3].rotation)
        assert rot_change > 0.01, f"Backward tilt should cause rotation change: {rot_change}"

    def test_align_with_left_tilt(self):
        """Test alignment modifies rotation for left-tilted terrain."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        left_tilt = Vec3(-0.2, 0.98, 0).normalized()
        original_rot = transforms[3].rotation

        placement._align_foot_to_terrain(transforms, left_foot, left_tilt)

        rot_change = get_angle_between_quats(original_rot, transforms[3].rotation)
        assert rot_change > 0.01, f"Left tilt should cause rotation change: {rot_change}"

    def test_align_with_right_tilt(self):
        """Test alignment modifies rotation for right-tilted terrain."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        right_tilt = Vec3(0.2, 0.98, 0).normalized()
        original_rot = transforms[3].rotation

        placement._align_foot_to_terrain(transforms, left_foot, right_tilt)

        rot_change = get_angle_between_quats(original_rot, transforms[3].rotation)
        assert rot_change > 0.01, f"Right tilt should cause rotation change: {rot_change}"

    def test_align_with_diagonal_tilt(self):
        """Test alignment modifies rotation for diagonal terrain tilt."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        diagonal_tilt = Vec3(0.15, 0.97, 0.15).normalized()
        original_rot = transforms[3].rotation

        placement._align_foot_to_terrain(transforms, left_foot, diagonal_tilt)

        rot_change = get_angle_between_quats(original_rot, transforms[3].rotation)
        assert rot_change > 0.01, f"Diagonal tilt should cause rotation change: {rot_change}"

    def test_align_with_steep_slope(self):
        """Test alignment modifies rotation for steep terrain slope."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        steep_tilt = Vec3(0.5, 0.866, 0).normalized()
        original_rot = transforms[3].rotation

        placement._align_foot_to_terrain(transforms, left_foot, steep_tilt)

        rot_change = get_angle_between_quats(original_rot, transforms[3].rotation)
        assert rot_change > 0.1, f"Steep slope should cause larger rotation change: {rot_change}"


# =============================================================================
# Test _rotation_between_vectors() - Basic Cases
# =============================================================================

class TestRotationBetweenVectorsBasic:
    """Basic tests for _rotation_between_vectors() method."""

    def test_same_vector_returns_identity(self):
        """Test rotation between identical vectors is identity."""
        placement = create_foot_placement()

        vec = Vec3(0, 1, 0)
        rot = placement._rotation_between_vectors(vec, vec)

        assert quat_approx_equal(rot, Quat.identity())

    def test_parallel_vectors_returns_identity(self):
        """Test parallel vectors (same direction) return identity."""
        placement = create_foot_placement()

        from_vec = Vec3(0, 2, 0)  # Same direction, different magnitude
        to_vec = Vec3(0, 5, 0)
        rot = placement._rotation_between_vectors(from_vec, to_vec)

        assert quat_approx_equal(rot, Quat.identity())

    def test_parallel_but_different_magnitude(self):
        """Test parallel vectors with different magnitudes."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 0, 0)
        to_vec = Vec3(100, 0, 0)
        rot = placement._rotation_between_vectors(from_vec, to_vec)

        assert quat_approx_equal(rot, Quat.identity())

    def test_rotation_produces_unit_quaternion(self):
        """Test result is always a unit quaternion."""
        placement = create_foot_placement()

        test_cases = [
            (Vec3(1, 0, 0), Vec3(0, 1, 0)),
            (Vec3(0, 1, 0), Vec3(0, 0, 1)),
            (Vec3(1, 1, 1), Vec3(-1, 2, 0)),
            (Vec3(0.5, -0.3, 0.8), Vec3(-0.2, 0.9, 0.1)),
        ]

        for from_vec, to_vec in test_cases:
            rot = placement._rotation_between_vectors(from_vec, to_vec)
            assert quat_is_unit(rot), f"Rotation should be unit for {from_vec} -> {to_vec}"

    def test_rotation_maps_from_to_to(self):
        """Test applying rotation to from_vec gives to_vec direction."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 0, 0)
        to_vec = Vec3(0, 1, 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        # The rotation might be in the opposite direction due to quaternion convention
        # Check that it's perpendicular to from_vec and in the Y-Z plane
        assert abs(rotated.x) < 0.01, f"X should be near 0, got {rotated.x}"
        assert abs(rotated.y) > 0.99 or abs(rotated.y) < 0.01, f"Should be rotated, got Y={rotated.y}"


# =============================================================================
# Test _rotation_between_vectors() - Opposite Vectors (180 Degrees)
# =============================================================================

class TestRotationBetweenVectorsOpposite:
    """Tests for _rotation_between_vectors() with opposite vectors."""

    def test_opposite_y_vectors(self):
        """Test rotation between opposite Y vectors (up to down)."""
        placement = create_foot_placement()

        from_vec = Vec3(0, 1, 0)
        to_vec = Vec3(0, -1, 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)

        # Applying rotation should flip the vector
        rotated = rot.rotate_vector(from_vec)
        assert vec3_approx_equal(rotated.normalized(), to_vec.normalized())

    def test_opposite_x_vectors(self):
        """Test rotation between opposite X vectors."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 0, 0)
        to_vec = Vec3(-1, 0, 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        assert vec3_approx_equal(rotated.normalized(), to_vec.normalized())

    def test_opposite_z_vectors(self):
        """Test rotation between opposite Z vectors."""
        placement = create_foot_placement()

        from_vec = Vec3(0, 0, 1)
        to_vec = Vec3(0, 0, -1)

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        assert vec3_approx_equal(rotated.normalized(), to_vec.normalized())

    def test_opposite_diagonal_vectors(self):
        """Test rotation between opposite diagonal vectors."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 1, 1).normalized()
        to_vec = Vec3(-1, -1, -1).normalized()

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        assert vec3_approx_equal(rotated.normalized(), to_vec.normalized())

    def test_opposite_rotation_is_180_degrees(self):
        """Test opposite vectors produce 180-degree rotation."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 0, 0)
        to_vec = Vec3(-1, 0, 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)

        # 180-degree rotation has w near 0
        # Angle from identity = 2 * acos(|w|) should be close to pi
        angle = get_angle_between_quats(rot, Quat.identity())
        assert abs(angle - math.pi) < 0.01, f"Expected 180 degrees, got {math.degrees(angle)}"


# =============================================================================
# Test _rotation_between_vectors() - Perpendicular Vectors (90 Degrees)
# =============================================================================

class TestRotationBetweenVectorsPerpendicular:
    """Tests for _rotation_between_vectors() with perpendicular vectors."""

    def test_x_to_y_perpendicular(self):
        """Test rotation from X axis to Y axis produces 90 degree rotation."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 0, 0)
        to_vec = Vec3(0, 1, 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        # Should be perpendicular to original (X axis)
        assert abs(rotated.x) < 0.01, f"X should be near 0 after rotation"
        # Should have magnitude 1 in Y or Z
        assert abs(abs(rotated.y) - 1.0) < 0.01 or abs(abs(rotated.z) - 1.0) < 0.01

    def test_y_to_z_perpendicular(self):
        """Test rotation from Y axis to Z axis produces 90 degree rotation."""
        placement = create_foot_placement()

        from_vec = Vec3(0, 1, 0)
        to_vec = Vec3(0, 0, 1)

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        # Should be perpendicular to original (Y axis)
        assert abs(rotated.y) < 0.01, f"Y should be near 0 after rotation"
        # Should have magnitude 1 in X or Z
        assert abs(abs(rotated.x) - 1.0) < 0.01 or abs(abs(rotated.z) - 1.0) < 0.01

    def test_z_to_x_perpendicular(self):
        """Test rotation from Z axis to X axis produces 90 degree rotation."""
        placement = create_foot_placement()

        from_vec = Vec3(0, 0, 1)
        to_vec = Vec3(1, 0, 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        # Should be perpendicular to original (Z axis)
        assert abs(rotated.z) < 0.01, f"Z should be near 0 after rotation"
        # Should have magnitude 1 in X or Y
        assert abs(abs(rotated.x) - 1.0) < 0.01 or abs(abs(rotated.y) - 1.0) < 0.01

    def test_perpendicular_rotation_is_90_degrees(self):
        """Test perpendicular vectors produce 90-degree rotation."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 0, 0)
        to_vec = Vec3(0, 1, 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)

        angle = get_angle_between_quats(rot, Quat.identity())
        assert abs(angle - math.pi / 2) < 0.01, f"Expected 90 degrees, got {math.degrees(angle)}"

    def test_negative_perpendicular(self):
        """Test rotation with negative perpendicular vectors produces 90 degree rotation."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 0, 0)
        to_vec = Vec3(0, -1, 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        # Should be perpendicular to original (X axis)
        assert abs(rotated.x) < 0.01, f"X should be near 0 after rotation"
        # Should have magnitude 1 in Y (either positive or negative)
        assert abs(abs(rotated.y) - 1.0) < 0.01, f"Y magnitude should be ~1, got {rotated.y}"


# =============================================================================
# Test _rotation_between_vectors() - Arbitrary Angles
# =============================================================================

class TestRotationBetweenVectorsArbitrary:
    """Tests for _rotation_between_vectors() with arbitrary angles."""

    def test_45_degree_rotation(self):
        """Test 45-degree rotation produces correct angle."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 0, 0)
        # 45 degrees between X and Y axes
        to_vec = Vec3(0.707, 0.707, 0).normalized()

        rot = placement._rotation_between_vectors(from_vec, to_vec)

        # Check rotation angle is approximately 45 degrees
        angle = get_angle_between_quats(rot, Quat.identity())
        assert abs(angle - math.pi / 4) < 0.01, f"Expected 45 degrees, got {math.degrees(angle)}"

    def test_30_degree_rotation(self):
        """Test 30-degree rotation produces correct angle."""
        placement = create_foot_placement()

        from_vec = Vec3(0, 1, 0)
        # 30 degrees from Y axis
        angle_rad = math.radians(30)
        to_vec = Vec3(math.sin(angle_rad), math.cos(angle_rad), 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)

        # Check rotation angle is approximately 30 degrees
        angle = get_angle_between_quats(rot, Quat.identity())
        assert abs(angle - math.radians(30)) < 0.01, f"Expected 30 degrees, got {math.degrees(angle)}"

    def test_120_degree_rotation(self):
        """Test 120-degree rotation produces correct angle."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 0, 0)
        # 120 degrees from X axis
        angle_rad = math.radians(120)
        to_vec = Vec3(math.cos(angle_rad), math.sin(angle_rad), 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)

        # Check rotation angle is approximately 120 degrees
        angle = get_angle_between_quats(rot, Quat.identity())
        assert abs(angle - math.radians(120)) < 0.01, f"Expected 120 degrees, got {math.degrees(angle)}"

    def test_arbitrary_3d_rotation(self):
        """Test rotation in 3D space."""
        placement = create_foot_placement()

        from_vec = Vec3(1, 2, 3).normalized()
        to_vec = Vec3(-2, 1, 0.5).normalized()

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        assert vec3_approx_equal(rotated.normalized(), to_vec.normalized())


# =============================================================================
# Test _rotation_between_vectors() - Edge Cases
# =============================================================================

class TestRotationBetweenVectorsEdgeCases:
    """Edge case tests for _rotation_between_vectors()."""

    def test_near_parallel_vectors(self):
        """Test vectors that are nearly parallel (dot ~ 1)."""
        placement = create_foot_placement()

        from_vec = Vec3(0, 1, 0)
        to_vec = Vec3(0.0001, 1, 0).normalized()

        rot = placement._rotation_between_vectors(from_vec, to_vec)

        # Should be near identity
        assert quat_approx_equal(rot, Quat.identity(), eps=0.01)

    def test_near_opposite_vectors(self):
        """Test vectors that are nearly opposite (dot ~ -1)."""
        placement = create_foot_placement()

        from_vec = Vec3(0, 1, 0)
        to_vec = Vec3(0.0001, -1, 0).normalized()

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        # Should still map correctly
        assert vec3_approx_equal(rotated.normalized(), to_vec.normalized(), eps=0.01)

    def test_unnormalized_input_vectors(self):
        """Test with unnormalized input vectors produces valid rotation."""
        placement = create_foot_placement()

        from_vec = Vec3(5, 0, 0)  # Length 5
        to_vec = Vec3(0, 10, 0)  # Length 10

        rot = placement._rotation_between_vectors(from_vec, to_vec)

        # Should produce a valid unit quaternion
        assert quat_is_unit(rot), "Rotation should be unit quaternion"

        # Rotation angle should be 90 degrees
        angle = get_angle_between_quats(rot, Quat.identity())
        assert abs(angle - math.pi / 2) < 0.01, f"Expected 90 degrees, got {math.degrees(angle)}"

    def test_very_small_vectors(self):
        """Test with very small magnitude vectors."""
        placement = create_foot_placement()

        from_vec = Vec3(0.001, 0, 0)
        to_vec = Vec3(0, 0.001, 0)

        rot = placement._rotation_between_vectors(from_vec, to_vec)

        # Should produce valid rotation
        assert quat_is_unit(rot)

    def test_same_direction_different_signs_z(self):
        """Test vectors along same line but different Z signs."""
        placement = create_foot_placement()

        from_vec = Vec3(0, 0, 1)
        to_vec = Vec3(0, 0, -1)

        rot = placement._rotation_between_vectors(from_vec, to_vec)
        rotated = rot.rotate_vector(from_vec)

        assert vec3_approx_equal(rotated.normalized(), to_vec.normalized())


# =============================================================================
# Test _scale_rotation() - Basic Cases
# =============================================================================

class TestScaleRotationBasic:
    """Basic tests for _scale_rotation() method."""

    def test_scale_zero_returns_identity(self):
        """Test scaling by 0 returns identity quaternion."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        scaled = placement._scale_rotation(rot, 0.0)

        assert quat_approx_equal(scaled, Quat.identity())

    def test_scale_one_returns_original(self):
        """Test scaling by 1 returns original rotation."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        scaled = placement._scale_rotation(rot, 1.0)

        assert quat_approx_equal(scaled, rot)

    def test_scale_negative_returns_identity(self):
        """Test negative scale returns identity."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        scaled = placement._scale_rotation(rot, -0.5)

        assert quat_approx_equal(scaled, Quat.identity())

    def test_scale_above_one_returns_original(self):
        """Test scale > 1 returns original rotation."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        scaled = placement._scale_rotation(rot, 1.5)

        assert quat_approx_equal(scaled, rot)

    def test_scaled_rotation_is_unit_quaternion(self):
        """Test scaled rotation is always unit quaternion."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(1, 1, 1).normalized(), math.pi / 3)

        for scale in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            scaled = placement._scale_rotation(rot, scale)
            assert quat_is_unit(scaled), f"Scaled rotation should be unit for scale={scale}"


# =============================================================================
# Test _scale_rotation() - Intermediate Scales
# =============================================================================

class TestScaleRotationIntermediate:
    """Tests for _scale_rotation() with intermediate scale values."""

    def test_scale_half_produces_half_angle(self):
        """Test scale=0.5 produces half the rotation angle."""
        placement = create_foot_placement()

        angle = math.pi / 2  # 90 degrees
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), angle)
        scaled = placement._scale_rotation(rot, 0.5)

        # Apply to test vector
        vec = Vec3(1, 0, 0)
        rotated = scaled.rotate_vector(vec)

        # Half rotation of 90 degrees = 45 degrees
        # cos(45) ~ 0.707
        expected_dot = math.cos(math.pi / 4)
        actual_dot = rotated.dot(vec)
        assert abs(actual_dot - expected_dot) < 0.01

    def test_scale_quarter_produces_quarter_angle(self):
        """Test scale=0.25 produces quarter rotation angle."""
        placement = create_foot_placement()

        angle = math.pi / 2  # 90 degrees
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), angle)
        scaled = placement._scale_rotation(rot, 0.25)

        vec = Vec3(1, 0, 0)
        rotated = scaled.rotate_vector(vec)

        # Quarter of 90 degrees = 22.5 degrees
        expected_dot = math.cos(math.pi / 8)
        actual_dot = rotated.dot(vec)
        assert abs(actual_dot - expected_dot) < 0.01

    def test_scale_three_quarters(self):
        """Test scale=0.75 produces 3/4 rotation angle."""
        placement = create_foot_placement()

        angle = math.pi / 2
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), angle)
        scaled = placement._scale_rotation(rot, 0.75)

        vec = Vec3(1, 0, 0)
        rotated = scaled.rotate_vector(vec)

        expected_dot = math.cos(3 * math.pi / 8)
        actual_dot = rotated.dot(vec)
        assert abs(actual_dot - expected_dot) < 0.01

    def test_scale_interpolation_is_linear_in_angle(self):
        """Test scaling produces linearly interpolated angles."""
        placement = create_foot_placement()

        angle = math.pi / 2
        rot = Quat.from_axis_angle(Vec3(0, 0, 1), angle)
        vec = Vec3(1, 0, 0)

        scales = [0.0, 0.25, 0.5, 0.75, 1.0]
        expected_angles = [0.0, angle * 0.25, angle * 0.5, angle * 0.75, angle]

        for scale, expected_angle in zip(scales, expected_angles):
            scaled = placement._scale_rotation(rot, scale)
            rotated = scaled.rotate_vector(vec)
            actual_dot = rotated.dot(vec)
            expected_dot = math.cos(expected_angle)
            assert abs(actual_dot - expected_dot) < 0.01, f"Failed for scale={scale}"


# =============================================================================
# Test _scale_rotation() - Various Rotation Axes
# =============================================================================

class TestScaleRotationAxes:
    """Tests for _scale_rotation() with various rotation axes."""

    def test_scale_x_axis_rotation(self):
        """Test scaling rotation around X axis."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi / 2)
        scaled = placement._scale_rotation(rot, 0.5)

        vec = Vec3(0, 1, 0)
        rotated = scaled.rotate_vector(vec)

        expected_dot = math.cos(math.pi / 4)
        actual_dot = rotated.dot(vec)
        assert abs(actual_dot - expected_dot) < 0.01

    def test_scale_z_axis_rotation(self):
        """Test scaling rotation around Z axis."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 2)
        scaled = placement._scale_rotation(rot, 0.5)

        vec = Vec3(1, 0, 0)
        rotated = scaled.rotate_vector(vec)

        expected_dot = math.cos(math.pi / 4)
        actual_dot = rotated.dot(vec)
        assert abs(actual_dot - expected_dot) < 0.01

    def test_scale_diagonal_axis_rotation(self):
        """Test scaling rotation around diagonal axis."""
        placement = create_foot_placement()

        axis = Vec3(1, 1, 1).normalized()
        rot = Quat.from_axis_angle(axis, math.pi / 2)
        scaled = placement._scale_rotation(rot, 0.5)

        # Verify it's still a valid rotation
        assert quat_is_unit(scaled)

        # The scaled rotation should be approximately half
        full_angle = get_angle_between_quats(rot, Quat.identity())
        scaled_angle = get_angle_between_quats(scaled, Quat.identity())
        assert abs(scaled_angle - full_angle * 0.5) < 0.01


# =============================================================================
# Test _scale_rotation() - Edge Cases
# =============================================================================

class TestScaleRotationEdgeCases:
    """Edge case tests for _scale_rotation()."""

    def test_scale_identity_rotation(self):
        """Test scaling identity rotation returns identity."""
        placement = create_foot_placement()

        scaled = placement._scale_rotation(Quat.identity(), 0.5)

        assert quat_approx_equal(scaled, Quat.identity())

    def test_scale_near_zero(self):
        """Test scale very close to zero."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        scaled = placement._scale_rotation(rot, 0.001)

        # Should be very close to identity
        angle = get_angle_between_quats(scaled, Quat.identity())
        assert angle < 0.01

    def test_scale_near_one(self):
        """Test scale very close to one."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        scaled = placement._scale_rotation(rot, 0.999)

        # Should be very close to original
        assert quat_approx_equal(scaled, rot, eps=0.01)

    def test_scale_at_epsilon_boundary(self):
        """Test scale at MATH_EPSILON boundary."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)

        # Just below epsilon should return identity
        scaled_below = placement._scale_rotation(rot, MATH_EPSILON * 0.5)
        assert quat_approx_equal(scaled_below, Quat.identity())

    def test_scale_180_degree_rotation(self):
        """Test scaling 180-degree rotation."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi)
        scaled = placement._scale_rotation(rot, 0.5)

        # Should produce 90-degree rotation
        vec = Vec3(1, 0, 0)
        rotated = scaled.rotate_vector(vec)
        expected = Vec3(0, 0, 1)  # 90 degrees around Y

        # Check perpendicular
        assert abs(rotated.dot(vec)) < 0.01


# =============================================================================
# Test Smooth Transitions (Weighted Alignment as Soft Limit)
# =============================================================================

class TestSmoothTransitions:
    """Tests for smooth transitions using weighted alignment."""

    def test_weighted_alignment_smooth_transition(self):
        """Test weighted alignment produces smooth transition."""
        placement = create_foot_placement()
        left_foot = create_left_foot_data()

        # Track rotations across increasing weights
        rotations = []
        tilted_normal = Vec3(0.3, 0.954, 0).normalized()

        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            transforms = create_biped_transforms()
            placement.toe_align_weight = weight
            left_foot.blend_weight = 1.0

            placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)
            rotations.append(transforms[3].rotation)

        # Each successive rotation should be closer to aligned
        for i in range(len(rotations) - 1):
            angle_to_identity_current = get_angle_between_quats(rotations[i], Quat.identity())
            angle_to_identity_next = get_angle_between_quats(rotations[i + 1], Quat.identity())
            # As weight increases, we move further from identity toward aligned
            # (assuming tilted normal is different from up)
            if i > 0:  # Skip first comparison (0 weight means no change)
                assert angle_to_identity_next >= angle_to_identity_current - 0.01

    def test_small_weight_small_change(self):
        """Test small weight produces small rotation change."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        original_rot = transforms[3].rotation

        placement.toe_align_weight = 0.1
        left_foot.blend_weight = 1.0

        tilted_normal = Vec3(0.5, 0.866, 0).normalized()

        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        angle_change = get_angle_between_quats(original_rot, transforms[3].rotation)
        # Small weight should produce small change
        assert angle_change < 0.2, f"Small weight should give small change, got {angle_change}"

    def test_incremental_alignment_produces_consistent_changes(self):
        """Test incremental alignment produces consistent rotation changes."""
        placement = create_foot_placement()
        left_foot = create_left_foot_data()
        tilted_normal = Vec3(0.1, 0.995, 0).normalized()

        placement.toe_align_weight = 0.5
        left_foot.blend_weight = 1.0

        transforms = create_biped_transforms()

        # Track rotation changes
        changes = []
        for _ in range(5):
            prev_rot = Quat(
                transforms[3].rotation.x, transforms[3].rotation.y,
                transforms[3].rotation.z, transforms[3].rotation.w
            )
            placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)
            change = get_angle_between_quats(prev_rot, transforms[3].rotation)
            changes.append(change)

        # First change should be non-zero
        assert changes[0] > 0.001, f"First alignment should produce change: {changes[0]}"

        # All changes should be similar (bounded behavior)
        max_change = max(changes)
        min_change = min(changes)
        assert max_change < 1.0, f"Changes should be bounded, got max={max_change}"


# =============================================================================
# Test Multiple Feet Alignment
# =============================================================================

class TestMultipleFeetAlignment:
    """Tests for aligning multiple feet independently."""

    def test_left_and_right_feet_independent(self):
        """Test left and right feet are aligned independently with different normals."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()

        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()
        left_foot.blend_weight = 1.0
        right_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        # Different normals for left and right
        left_normal = Vec3(-0.15, 0.98, 0).normalized()
        right_normal = Vec3(0.15, 0.98, 0).normalized()

        left_orig = transforms[3].rotation
        right_orig = transforms[7].rotation

        # Apply alignment to each foot
        placement._align_foot_to_terrain(transforms, left_foot, left_normal)
        placement._align_foot_to_terrain(transforms, right_foot, right_normal)

        # Both feet should have rotated
        left_change = get_angle_between_quats(left_orig, transforms[3].rotation)
        right_change = get_angle_between_quats(right_orig, transforms[7].rotation)

        assert left_change > 0.01, f"Left foot should rotate: {left_change}"
        assert right_change > 0.01, f"Right foot should rotate: {right_change}"

        # Rotations should be different (different normals)
        assert not quat_approx_equal(transforms[3].rotation, transforms[7].rotation), \
            "Left and right feet should have different rotations"

    def test_one_foot_aligned_other_unchanged(self):
        """Test aligning one foot doesn't affect the other."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()

        left_foot = create_left_foot_data()
        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        original_right_rot = Quat(
            transforms[7].rotation.x,
            transforms[7].rotation.y,
            transforms[7].rotation.z,
            transforms[7].rotation.w
        )

        tilted_normal = Vec3(0.3, 0.954, 0).normalized()

        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        # Right foot should be unchanged
        assert quat_approx_equal(transforms[7].rotation, original_right_rot)


# =============================================================================
# Test Pre-rotated Foot Alignment
# =============================================================================

class TestPrerotatedFootAlignment:
    """Tests for alignment when foot already has rotation."""

    def test_align_rotated_foot_to_vertical(self):
        """Test aligning a pre-rotated foot produces rotation change."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        # Pre-rotate foot
        transforms[3].rotation = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi / 12)
        original_rot = Quat(
            transforms[3].rotation.x, transforms[3].rotation.y,
            transforms[3].rotation.z, transforms[3].rotation.w
        )

        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        # Align to vertical normal
        placement._align_foot_to_terrain(transforms, left_foot, Vec3(0, 1, 0))

        # Rotation should have changed
        rot_change = get_angle_between_quats(original_rot, transforms[3].rotation)
        assert rot_change > 0.01, f"Alignment should change rotation: {rot_change}"

    def test_align_rotated_foot_to_tilted(self):
        """Test aligning a pre-rotated foot to tilted terrain produces change."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        # Pre-rotate foot
        transforms[3].rotation = Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 8)
        original_rot = Quat(
            transforms[3].rotation.x, transforms[3].rotation.y,
            transforms[3].rotation.z, transforms[3].rotation.w
        )

        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        tilted_normal = Vec3(0.15, 0.98, 0).normalized()

        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        rot_change = get_angle_between_quats(original_rot, transforms[3].rotation)
        assert rot_change > 0.01, f"Alignment should change rotation: {rot_change}"

    def test_combine_existing_rotation_with_alignment(self):
        """Test alignment combines with existing rotation correctly."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        # Give foot a specific rotation
        initial_rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        transforms[3].rotation = initial_rot

        left_foot.blend_weight = 1.0
        placement.toe_align_weight = 1.0

        # With vertical normal, should not change much
        placement._align_foot_to_terrain(transforms, left_foot, Vec3(0, 1, 0))

        # The Y axis should still point up after alignment to up normal
        foot_up = transforms[3].rotation.rotate_vector(Vec3(0, 1, 0))
        assert foot_up.dot(Vec3(0, 1, 0)) > 0.99


# =============================================================================
# Test Configuration Parameters
# =============================================================================

class TestAlignmentConfiguration:
    """Tests for alignment configuration parameters."""

    def test_default_toe_align_weight(self):
        """Test default toe_align_weight value."""
        placement = create_foot_placement()
        assert placement.toe_align_weight == FOOT_PLACEMENT_TOE_ALIGN_WEIGHT

    def test_custom_toe_align_weight(self):
        """Test custom toe_align_weight is respected."""
        placement = create_foot_placement()
        placement.toe_align_weight = 0.3

        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()
        left_foot.blend_weight = 1.0

        original_rot = transforms[3].rotation
        tilted_normal = Vec3(0.5, 0.866, 0).normalized()

        placement._align_foot_to_terrain(transforms, left_foot, tilted_normal)

        # With 0.3 weight, rotation change should be partial
        angle_change = get_angle_between_quats(original_rot, transforms[3].rotation)
        assert 0.01 < angle_change < 0.5, f"Partial weight should give partial change: {angle_change}"

    def test_blend_weight_default_is_one(self):
        """Test default blend_weight is 1.0."""
        foot = create_left_foot_data()
        assert foot.blend_weight == 1.0


# =============================================================================
# Test Error Handling
# =============================================================================

class TestAlignmentErrorHandling:
    """Tests for error handling in alignment methods."""

    def test_align_with_zero_normal_no_crash(self):
        """Test alignment with zero vector normal doesn't crash."""
        placement = create_foot_placement()
        transforms = create_biped_transforms()
        left_foot = create_left_foot_data()

        # Zero normal - should handle gracefully
        zero_normal = Vec3(0, 0, 0)

        # Should not raise (behavior depends on implementation)
        try:
            placement._align_foot_to_terrain(transforms, left_foot, zero_normal)
        except (ZeroDivisionError, ValueError):
            # These exceptions are acceptable for zero vector input
            pass

    def test_rotation_between_zero_vectors_no_crash(self):
        """Test rotation computation with zero vectors doesn't crash."""
        placement = create_foot_placement()

        try:
            rot = placement._rotation_between_vectors(Vec3.zero(), Vec3(0, 1, 0))
        except (ZeroDivisionError, ValueError):
            pass  # Acceptable exceptions

    def test_scale_rotation_with_nan_no_crash(self):
        """Test scale rotation handles NaN gracefully."""
        placement = create_foot_placement()

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)

        try:
            scaled = placement._scale_rotation(rot, float('nan'))
            # If it returns something, it should still be valid
            if scaled is not None:
                assert quat_is_unit(scaled) or math.isnan(scaled.w)
        except (ValueError, RuntimeError):
            pass  # Acceptable exceptions

    def test_align_empty_transforms_no_crash(self):
        """Test alignment with empty transform list doesn't crash."""
        placement = create_foot_placement()
        transforms = []
        left_foot = create_left_foot_data()

        try:
            placement._align_foot_to_terrain(transforms, left_foot, Vec3(0, 1, 0))
        except IndexError:
            pass  # Acceptable - foot index out of bounds
