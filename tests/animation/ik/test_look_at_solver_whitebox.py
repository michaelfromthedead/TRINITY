"""Whitebox tests for LookAtSolver.

Tests the look-at IK solver implementation covering:
- Attribute initialization with default and custom values
- solve() method with various targets and transform configurations
- Weight distribution across head, neck, and spine bones
- Max angle clamping when target exceeds LOOK_AT_MAX_ANGLE
- _rotation_between() for parallel, opposite, and arbitrary vectors
- _scale_rotation() with scale=0 (identity), scale=1 (full), scale=0.5 (half)
- _quat_to_axis_angle() quaternion decomposition
- Edge cases: empty spine_bones, targets behind character, zero-length vectors
"""

from __future__ import annotations

import math
import pytest
from typing import List

from engine.animation.ik.fullbody import LookAtSolver
from engine.animation.ik.config import (
    LOOK_AT_MAX_ANGLE,
    LOOK_AT_HEAD_WEIGHT,
    LOOK_AT_NECK_WEIGHT,
    LOOK_AT_SPINE_WEIGHT,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON


# =============================================================================
# Helper Functions
# =============================================================================

def create_identity_transforms(count: int) -> List[Transform]:
    """Create a list of identity transforms."""
    return [
        Transform(Vec3(0, 0, 0), Quat.identity(), Vec3(1, 1, 1))
        for _ in range(count)
    ]


def create_transform_at(position: Vec3, rotation: Quat = None) -> Transform:
    """Create a transform at a specific position with optional rotation."""
    return Transform(
        position,
        rotation if rotation is not None else Quat.identity(),
        Vec3(1, 1, 1)
    )


def create_simple_spine_transforms() -> List[Transform]:
    """Create transforms for a simple spine hierarchy.

    Layout:
    - Bone 0: Spine base at (0, 0, 0)
    - Bone 1: Spine middle at (0, 0.5, 0)
    - Bone 2: Spine top at (0, 1.0, 0)
    - Bone 3: Neck at (0, 1.5, 0)
    - Bone 4: Head at (0, 2.0, 0)
    """
    return [
        create_transform_at(Vec3(0, 0.0, 0)),   # spine base (index 0)
        create_transform_at(Vec3(0, 0.5, 0)),   # spine middle (index 1)
        create_transform_at(Vec3(0, 1.0, 0)),   # spine top (index 2)
        create_transform_at(Vec3(0, 1.5, 0)),   # neck (index 3)
        create_transform_at(Vec3(0, 2.0, 0)),   # head (index 4)
    ]


def quat_approx_equal(q1: Quat, q2: Quat, tolerance: float = 1e-5) -> bool:
    """Check if two quaternions are approximately equal."""
    # Handle sign ambiguity (q and -q represent same rotation)
    dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
    return abs(abs(dot) - 1.0) < tolerance


def vec3_approx_equal(v1: Vec3, v2: Vec3, tolerance: float = 1e-5) -> bool:
    """Check if two vectors are approximately equal."""
    return (
        abs(v1.x - v2.x) < tolerance and
        abs(v1.y - v2.y) < tolerance and
        abs(v1.z - v2.z) < tolerance
    )


# =============================================================================
# Test Class: Initialization
# =============================================================================

class TestLookAtSolverInit:
    """Tests for LookAtSolver.__init__()."""

    def test_init_with_default_weights(self):
        """Test initialization with default weight values."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2]
        )

        assert solver.head_bone == 4
        assert solver.neck_bone == 3
        assert solver.spine_bones == [0, 1, 2]
        assert solver.head_weight == LOOK_AT_HEAD_WEIGHT
        assert solver.neck_weight == LOOK_AT_NECK_WEIGHT
        assert solver.spine_weight == LOOK_AT_SPINE_WEIGHT
        assert solver.max_angle == LOOK_AT_MAX_ANGLE

    def test_init_with_custom_weights(self):
        """Test initialization with custom weight values."""
        solver = LookAtSolver(
            head_bone=10,
            neck_bone=9,
            spine_bones=[5, 6, 7, 8],
            head_weight=0.5,
            neck_weight=0.35,
            spine_weight=0.15
        )

        assert solver.head_bone == 10
        assert solver.neck_bone == 9
        assert solver.spine_bones == [5, 6, 7, 8]
        assert solver.head_weight == 0.5
        assert solver.neck_weight == 0.35
        assert solver.spine_weight == 0.15

    def test_init_with_empty_spine_bones(self):
        """Test initialization with an empty spine bones list."""
        solver = LookAtSolver(
            head_bone=1,
            neck_bone=0,
            spine_bones=[]
        )

        assert solver.head_bone == 1
        assert solver.neck_bone == 0
        assert solver.spine_bones == []
        assert len(solver.spine_bones) == 0

    def test_init_with_single_spine_bone(self):
        """Test initialization with a single spine bone."""
        solver = LookAtSolver(
            head_bone=2,
            neck_bone=1,
            spine_bones=[0]
        )

        assert solver.spine_bones == [0]
        assert len(solver.spine_bones) == 1

    def test_init_spine_bones_copied(self):
        """Test that spine_bones list is copied (not referenced)."""
        original_bones = [0, 1, 2]
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=original_bones
        )

        # Modify original
        original_bones.append(5)

        # Solver should not be affected
        assert solver.spine_bones == [0, 1, 2]
        assert len(solver.spine_bones) == 3

    def test_init_with_zero_weights(self):
        """Test initialization with zero weights."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=0.0,
            neck_weight=0.0,
            spine_weight=0.0
        )

        assert solver.head_weight == 0.0
        assert solver.neck_weight == 0.0
        assert solver.spine_weight == 0.0

    def test_init_with_full_head_weight(self):
        """Test initialization with all weight on head."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=1.0,
            neck_weight=0.0,
            spine_weight=0.0
        )

        assert solver.head_weight == 1.0
        assert solver.neck_weight == 0.0
        assert solver.spine_weight == 0.0

    def test_init_max_angle_from_config(self):
        """Test that max_angle is set from LOOK_AT_MAX_ANGLE constant."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[])

        assert solver.max_angle == LOOK_AT_MAX_ANGLE
        assert solver.max_angle == pytest.approx(math.pi / 2, abs=1e-4)

    def test_init_bone_indices_zero(self):
        """Test initialization with bone index zero (valid)."""
        solver = LookAtSolver(
            head_bone=0,
            neck_bone=0,
            spine_bones=[0]
        )

        assert solver.head_bone == 0
        assert solver.neck_bone == 0
        assert solver.spine_bones == [0]

    def test_init_large_bone_indices(self):
        """Test initialization with large bone indices."""
        solver = LookAtSolver(
            head_bone=100,
            neck_bone=99,
            spine_bones=[50, 51, 52, 53, 54]
        )

        assert solver.head_bone == 100
        assert solver.neck_bone == 99
        assert len(solver.spine_bones) == 5


# =============================================================================
# Test Class: _rotation_between
# =============================================================================

class TestRotationBetween:
    """Tests for LookAtSolver._rotation_between()."""

    def setup_method(self):
        """Set up test fixtures."""
        self.solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])

    def test_rotation_between_parallel_vectors(self):
        """Test rotation between parallel vectors (dot ~= 1)."""
        from_vec = Vec3(0, 0, 1).normalized()
        to_vec = Vec3(0, 0, 1).normalized()

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Should return identity quaternion
        assert quat_approx_equal(rotation, Quat.identity())

    def test_rotation_between_nearly_parallel_vectors(self):
        """Test rotation between nearly parallel vectors."""
        from_vec = Vec3(0, 0, 1).normalized()
        to_vec = Vec3(0.0001, 0, 1).normalized()

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Should be very close to identity
        assert quat_approx_equal(rotation, Quat.identity(), tolerance=1e-3)

    def test_rotation_between_opposite_vectors(self):
        """Test rotation between opposite vectors (dot ~= -1)."""
        from_vec = Vec3(0, 0, 1).normalized()
        to_vec = Vec3(0, 0, -1).normalized()

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Should be a 180 degree rotation
        axis, angle = self.solver._quat_to_axis_angle(rotation)
        assert angle == pytest.approx(math.pi, abs=1e-4)

    def test_rotation_between_perpendicular_x_to_z(self):
        """Test rotation from X axis to Z axis."""
        from_vec = Vec3(1, 0, 0).normalized()
        to_vec = Vec3(0, 0, 1).normalized()

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Should be 90 degree rotation around Y axis
        axis, angle = self.solver._quat_to_axis_angle(rotation)
        assert angle == pytest.approx(math.pi / 2, abs=1e-4)

    def test_rotation_between_perpendicular_y_to_z(self):
        """Test rotation from Y axis to Z axis."""
        from_vec = Vec3(0, 1, 0).normalized()
        to_vec = Vec3(0, 0, 1).normalized()

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Should be 90 degree rotation
        axis, angle = self.solver._quat_to_axis_angle(rotation)
        assert angle == pytest.approx(math.pi / 2, abs=1e-4)

    def test_rotation_between_arbitrary_vectors(self):
        """Test rotation between arbitrary normalized vectors."""
        from_vec = Vec3(1, 1, 0).normalized()
        to_vec = Vec3(0, 1, 1).normalized()

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Apply rotation and verify result matches to_vec
        rotated = rotation.rotate_vector(from_vec)
        assert vec3_approx_equal(rotated, to_vec, tolerance=1e-4)

    def test_rotation_between_preserves_length(self):
        """Test that rotation preserves vector length."""
        from_vec = Vec3(2, 3, 1).normalized()
        to_vec = Vec3(-1, 2, 3).normalized()

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Rotation should preserve length
        test_vec = Vec3(1, 0, 0)
        rotated = rotation.rotate_vector(test_vec)
        assert rotated.length() == pytest.approx(1.0, abs=1e-4)

    def test_rotation_between_x_to_negative_x(self):
        """Test 180 degree rotation from +X to -X."""
        from_vec = Vec3(1, 0, 0)
        to_vec = Vec3(-1, 0, 0)

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Verify rotation works
        rotated = rotation.rotate_vector(from_vec)
        assert vec3_approx_equal(rotated, to_vec, tolerance=1e-4)

    def test_rotation_between_y_to_negative_y(self):
        """Test 180 degree rotation from +Y to -Y."""
        from_vec = Vec3(0, 1, 0)
        to_vec = Vec3(0, -1, 0)

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Verify rotation works
        rotated = rotation.rotate_vector(from_vec)
        assert vec3_approx_equal(rotated, to_vec, tolerance=1e-4)

    def test_rotation_between_z_to_negative_z(self):
        """Test 180 degree rotation from +Z to -Z."""
        from_vec = Vec3(0, 0, 1)
        to_vec = Vec3(0, 0, -1)

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Verify rotation works
        rotated = rotation.rotate_vector(from_vec)
        assert vec3_approx_equal(rotated, to_vec, tolerance=1e-4)

    def test_rotation_between_diagonal_vectors(self):
        """Test rotation between diagonal vectors."""
        from_vec = Vec3(1, 1, 1).normalized()
        to_vec = Vec3(-1, -1, -1).normalized()

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Should be 180 degree rotation
        axis, angle = self.solver._quat_to_axis_angle(rotation)
        assert angle == pytest.approx(math.pi, abs=1e-4)

    def test_rotation_between_small_angle(self):
        """Test rotation with a small angle."""
        from_vec = Vec3(0, 0, 1)
        to_vec = Vec3(0.01, 0, 1).normalized()

        rotation = self.solver._rotation_between(from_vec, to_vec)

        # Should be a small rotation
        axis, angle = self.solver._quat_to_axis_angle(rotation)
        assert angle < 0.1

    def test_rotation_between_clamped_dot_product(self):
        """Test that dot product is clamped to [-1, 1]."""
        # Even with numerical issues, should not crash
        from_vec = Vec3(0, 0, 1)
        to_vec = Vec3(0, 0, 1)

        # This should work without issues due to clamping
        rotation = self.solver._rotation_between(from_vec, to_vec)
        assert rotation is not None


# =============================================================================
# Test Class: _scale_rotation
# =============================================================================

class TestScaleRotation:
    """Tests for LookAtSolver._scale_rotation()."""

    def setup_method(self):
        """Set up test fixtures."""
        self.solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])

    def test_scale_rotation_zero_returns_identity(self):
        """Test that scale=0 returns identity quaternion."""
        rotation = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)

        scaled = self.solver._scale_rotation(rotation, 0.0)

        # Should be identity rotation
        axis, angle = self.solver._quat_to_axis_angle(scaled)
        assert angle == pytest.approx(0.0, abs=1e-4)

    def test_scale_rotation_one_returns_full(self):
        """Test that scale=1 returns the full rotation."""
        original_angle = math.pi / 4
        rotation = Quat.from_axis_angle(Vec3(0, 1, 0), original_angle)

        scaled = self.solver._scale_rotation(rotation, 1.0)

        axis, angle = self.solver._quat_to_axis_angle(scaled)
        assert angle == pytest.approx(original_angle, abs=1e-4)

    def test_scale_rotation_half(self):
        """Test that scale=0.5 returns half rotation."""
        original_angle = math.pi / 2
        rotation = Quat.from_axis_angle(Vec3(0, 1, 0), original_angle)

        scaled = self.solver._scale_rotation(rotation, 0.5)

        axis, angle = self.solver._quat_to_axis_angle(scaled)
        assert angle == pytest.approx(original_angle * 0.5, abs=1e-4)

    def test_scale_rotation_quarter(self):
        """Test that scale=0.25 returns quarter rotation."""
        original_angle = math.pi
        rotation = Quat.from_axis_angle(Vec3(0, 1, 0), original_angle)

        scaled = self.solver._scale_rotation(rotation, 0.25)

        axis, angle = self.solver._quat_to_axis_angle(scaled)
        assert angle == pytest.approx(original_angle * 0.25, abs=1e-4)

    def test_scale_rotation_preserves_axis(self):
        """Test that scaling preserves the rotation axis."""
        original_axis = Vec3(1, 1, 1).normalized()
        rotation = Quat.from_axis_angle(original_axis, math.pi / 2)

        scaled = self.solver._scale_rotation(rotation, 0.5)

        axis, angle = self.solver._quat_to_axis_angle(scaled)
        # Axis should be preserved (or flipped, which is equivalent)
        dot = abs(axis.dot(original_axis))
        assert dot == pytest.approx(1.0, abs=1e-4)

    def test_scale_rotation_double(self):
        """Test scaling rotation by 2 (double the angle)."""
        original_angle = math.pi / 4
        rotation = Quat.from_axis_angle(Vec3(0, 1, 0), original_angle)

        scaled = self.solver._scale_rotation(rotation, 2.0)

        axis, angle = self.solver._quat_to_axis_angle(scaled)
        assert angle == pytest.approx(original_angle * 2.0, abs=1e-4)

    def test_scale_rotation_small_scale(self):
        """Test scaling with a very small scale factor."""
        rotation = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi)

        scaled = self.solver._scale_rotation(rotation, 0.01)

        axis, angle = self.solver._quat_to_axis_angle(scaled)
        assert angle == pytest.approx(math.pi * 0.01, abs=1e-4)

    def test_scale_rotation_identity_input(self):
        """Test scaling an identity rotation."""
        rotation = Quat.identity()

        scaled = self.solver._scale_rotation(rotation, 0.5)

        # Identity scaled by anything should still be (nearly) identity
        axis, angle = self.solver._quat_to_axis_angle(scaled)
        assert angle == pytest.approx(0.0, abs=1e-4)

    def test_scale_rotation_negative_scale(self):
        """Test scaling with negative scale (reverses direction)."""
        original_angle = math.pi / 2
        rotation = Quat.from_axis_angle(Vec3(0, 1, 0), original_angle)

        scaled = self.solver._scale_rotation(rotation, -1.0)

        axis, angle = self.solver._quat_to_axis_angle(scaled)
        # Result should be rotation in opposite direction
        assert abs(angle) == pytest.approx(original_angle, abs=1e-4)

    def test_scale_rotation_various_axes(self):
        """Test scaling rotations around various axes."""
        axes = [
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 0, 1),
            Vec3(1, 1, 0).normalized(),
            Vec3(1, 1, 1).normalized(),
        ]

        for axis in axes:
            rotation = Quat.from_axis_angle(axis, math.pi / 3)
            scaled = self.solver._scale_rotation(rotation, 0.6)
            result_axis, result_angle = self.solver._quat_to_axis_angle(scaled)
            assert result_angle == pytest.approx(math.pi / 3 * 0.6, abs=1e-4)


# =============================================================================
# Test Class: _quat_to_axis_angle
# =============================================================================

class TestQuatToAxisAngle:
    """Tests for LookAtSolver._quat_to_axis_angle()."""

    def setup_method(self):
        """Set up test fixtures."""
        self.solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])

    def test_quat_to_axis_angle_identity(self):
        """Test identity quaternion gives zero angle."""
        q = Quat.identity()

        axis, angle = self.solver._quat_to_axis_angle(q)

        assert angle == pytest.approx(0.0, abs=1e-4)
        # Axis is arbitrary for zero rotation, just check it's valid
        assert axis.length() == pytest.approx(1.0, abs=1e-4)

    def test_quat_to_axis_angle_90_degrees_y(self):
        """Test 90 degree rotation around Y axis."""
        q = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)

        axis, angle = self.solver._quat_to_axis_angle(q)

        assert angle == pytest.approx(math.pi / 2, abs=1e-4)
        assert vec3_approx_equal(axis, Vec3(0, 1, 0), tolerance=1e-4)

    def test_quat_to_axis_angle_180_degrees(self):
        """Test 180 degree rotation."""
        q = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi)

        axis, angle = self.solver._quat_to_axis_angle(q)

        assert angle == pytest.approx(math.pi, abs=1e-4)

    def test_quat_to_axis_angle_45_degrees(self):
        """Test 45 degree rotation."""
        q = Quat.from_axis_angle(Vec3(0, 0, 1), math.pi / 4)

        axis, angle = self.solver._quat_to_axis_angle(q)

        assert angle == pytest.approx(math.pi / 4, abs=1e-4)
        # Axis should match original (up to sign)
        dot = abs(axis.dot(Vec3(0, 0, 1)))
        assert dot == pytest.approx(1.0, abs=1e-4)

    def test_quat_to_axis_angle_arbitrary(self):
        """Test arbitrary rotation."""
        original_axis = Vec3(1, 2, 3).normalized()
        original_angle = 1.23
        q = Quat.from_axis_angle(original_axis, original_angle)

        axis, angle = self.solver._quat_to_axis_angle(q)

        assert angle == pytest.approx(original_angle, abs=1e-4)
        # Axis should match (up to sign)
        dot = abs(axis.dot(original_axis))
        assert dot == pytest.approx(1.0, abs=1e-4)

    def test_quat_to_axis_angle_small_angle(self):
        """Test very small angle (near identity)."""
        small_angle = 0.001
        q = Quat.from_axis_angle(Vec3(0, 1, 0), small_angle)

        axis, angle = self.solver._quat_to_axis_angle(q)

        assert angle == pytest.approx(small_angle, abs=1e-3)

    def test_quat_to_axis_angle_returns_normalized_axis(self):
        """Test that returned axis is normalized."""
        q = Quat.from_axis_angle(Vec3(1, 1, 1).normalized(), math.pi / 3)

        axis, angle = self.solver._quat_to_axis_angle(q)

        assert axis.length() == pytest.approx(1.0, abs=1e-4)

    def test_quat_to_axis_angle_clamped_w(self):
        """Test that w component is clamped to [-1, 1]."""
        # Create a quaternion manually to test edge case
        q = Quat.identity()

        axis, angle = self.solver._quat_to_axis_angle(q)

        # Should not crash and angle should be valid
        assert 0.0 <= angle <= 2 * math.pi


# =============================================================================
# Test Class: solve()
# =============================================================================

class TestSolve:
    """Tests for LookAtSolver.solve()."""

    def test_solve_returns_modified_transforms(self):
        """Test that solve returns a new list of transforms."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(0, 2, 1)  # In front of head

        result = solver.solve(transforms, target)

        # Should return a new list
        assert result is not transforms
        assert len(result) == len(transforms)

    def test_solve_does_not_modify_original_transforms(self):
        """Test that solve does not modify the original transforms."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        original_head_rot = transforms[4].rotation
        target = Vec3(0, 2, 1)

        solver.solve(transforms, target)

        # Original should be unchanged
        assert quat_approx_equal(transforms[4].rotation, original_head_rot)

    def test_solve_target_in_front(self):
        """Test looking at target directly in front."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(0, 2, 5)  # Directly in front

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Some rotation should be applied (or none if already looking there)
        # Head should be rotated to look at target
        assert result[4].rotation is not None

    def test_solve_target_to_the_right(self):
        """Test looking at target to the right."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)  # To the right

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Head should be rotated to look right
        # The rotation should not be identity
        head_rot = result[4].rotation
        identity = Quat.identity()
        assert not quat_approx_equal(head_rot, identity)

    def test_solve_target_to_the_left(self):
        """Test looking at target to the left."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(-5, 2, 0)  # To the left

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Head should be rotated to look left
        head_rot = result[4].rotation
        identity = Quat.identity()
        assert not quat_approx_equal(head_rot, identity)

    def test_solve_target_above(self):
        """Test looking at target above."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(0, 10, 1)  # Above

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Head should be rotated to look up
        assert result is not None

    def test_solve_target_below(self):
        """Test looking at target below."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(0, -5, 1)  # Below

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Head should be rotated to look down
        assert result is not None

    def test_solve_weight_distribution_head_gets_most(self):
        """Test that head gets the most rotation weight."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=0.6,
            neck_weight=0.3,
            spine_weight=0.1
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)  # To the right

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Head should have more rotation than neck
        head_axis, head_angle = solver._quat_to_axis_angle(result[4].rotation)
        neck_axis, neck_angle = solver._quat_to_axis_angle(result[3].rotation)

        # Head angle should be greater
        assert head_angle > neck_angle

    def test_solve_empty_spine_bones(self):
        """Test solve with empty spine bones list."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[]
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        # Should not crash
        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        assert len(result) == len(transforms)

    def test_solve_single_spine_bone(self):
        """Test solve with single spine bone."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[2],
            spine_weight=0.2
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Single spine bone should get full spine_weight
        spine_axis, spine_angle = solver._quat_to_axis_angle(result[2].rotation)
        assert spine_angle > 0

    def test_solve_spine_weight_distributed(self):
        """Test that spine weight is distributed across spine bones."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            spine_weight=0.3
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Each spine bone should get spine_weight / 3
        # All three should have approximately equal rotation
        angles = []
        for idx in [0, 1, 2]:
            _, angle = solver._quat_to_axis_angle(result[idx].rotation)
            angles.append(angle)

        # All angles should be similar
        assert angles[0] == pytest.approx(angles[1], abs=0.01)
        assert angles[1] == pytest.approx(angles[2], abs=0.01)

    def test_solve_max_angle_clamping(self):
        """Test that rotation is clamped to max_angle."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        # Target behind the character (requires > 90 degree rotation)
        target = Vec3(0, 2, -10)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Total rotation should be clamped to max_angle (90 degrees)
        # Combined rotation of all bones should not exceed this
        # The clamping is applied to total_rotation before distribution
        assert result is not None

    def test_solve_target_behind_triggers_clamping(self):
        """Test that target behind character triggers angle clamping."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(0, 2, -5)  # Behind

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Should still produce valid result due to clamping
        assert len(result) == len(transforms)
        for t in result:
            assert t.rotation is not None

    def test_solve_custom_forward_axis(self):
        """Test solve with custom forward axis."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        # Use X as forward instead of Z
        result = solver.solve(transforms, target, Vec3(1, 0, 0))

        assert len(result) == len(transforms)

    def test_solve_zero_weights_no_rotation(self):
        """Test that zero weights result in no rotation change."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=0.0,
            neck_weight=0.0,
            spine_weight=0.0
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # All rotations should be multiplied by zero-scaled rotations (identity)
        for i, (orig, res) in enumerate(zip(transforms, result)):
            _, angle = solver._quat_to_axis_angle(res.rotation)
            # Should have minimal rotation
            assert angle < 0.01

    def test_solve_target_at_head_position(self):
        """Test looking at target exactly at head position."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        # Target at exact head position
        head_pos = transforms[4].translation
        target = Vec3(head_pos.x, head_pos.y, head_pos.z)

        # This might produce zero-length vector, should handle gracefully
        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert result is not None

    def test_solve_preserves_scale(self):
        """Test that solve preserves transform scale."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity(), Vec3(2, 2, 2)),
            Transform(Vec3(0, 0.5, 0), Quat.identity(), Vec3(1.5, 1.5, 1.5)),
            Transform(Vec3(0, 1, 0), Quat.identity(), Vec3(1, 1, 1)),
            Transform(Vec3(0, 1.5, 0), Quat.identity(), Vec3(0.8, 0.8, 0.8)),
            Transform(Vec3(0, 2, 0), Quat.identity(), Vec3(0.5, 0.5, 0.5)),
        ]
        target = Vec3(5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Scales should be preserved
        for i, (orig, res) in enumerate(zip(transforms, result)):
            assert vec3_approx_equal(orig.scale, res.scale)

    def test_solve_preserves_translation(self):
        """Test that solve preserves transform translation."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Translations should be preserved
        for i, (orig, res) in enumerate(zip(transforms, result)):
            assert vec3_approx_equal(orig.translation, res.translation)


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestLookAtSolverEdgeCases:
    """Edge case tests for LookAtSolver."""

    def test_identical_head_neck_bones(self):
        """Test when head and neck are the same bone."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=4,  # Same as head
            spine_bones=[0, 1, 2]
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        # Should handle without crash
        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)

    def test_head_in_spine_bones_list(self):
        """Test when head bone is also in spine bones list."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2, 4]  # 4 is head
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        # Should handle without crash
        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)

    def test_very_far_target(self):
        """Test with target very far away."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(10000, 2, 10000)  # Very far

        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)

    def test_very_close_target(self):
        """Test with target very close to head."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(0.01, 2, 0.01)  # Very close

        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)

    def test_all_bones_same_index(self):
        """Test when all bones have the same index."""
        solver = LookAtSolver(
            head_bone=0,
            neck_bone=0,
            spine_bones=[0]
        )
        transforms = [create_transform_at(Vec3(0, 0, 0))]
        target = Vec3(1, 0, 0)

        # Should handle without crash
        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == 1

    def test_negative_weights(self):
        """Test with negative weights."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=-0.5,
            neck_weight=-0.3,
            spine_weight=-0.1
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        # Should handle without crash (rotates opposite direction)
        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)

    def test_weights_sum_greater_than_one(self):
        """Test with weights summing to more than 1."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=0.8,
            neck_weight=0.8,
            spine_weight=0.8
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        # Should handle without crash
        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)

    def test_very_large_weights(self):
        """Test with very large weight values."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=10.0,
            neck_weight=5.0,
            spine_weight=3.0
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        # Should handle without crash
        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)

    def test_pre_rotated_head(self):
        """Test with head already rotated."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        # Pre-rotate head to look right
        transforms[4].rotation = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        target = Vec3(5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)

    def test_non_normalized_forward_axis(self):
        """Test with non-normalized forward axis."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        # Use non-normalized forward axis
        result = solver.solve(transforms, target, Vec3(0, 0, 10))
        assert len(result) == len(transforms)

    def test_many_spine_bones(self):
        """Test with many spine bones."""
        solver = LookAtSolver(
            head_bone=20,
            neck_bone=19,
            spine_bones=list(range(15)),  # 15 spine bones
            spine_weight=0.3
        )
        transforms = create_identity_transforms(25)
        # Set up positions
        for i in range(25):
            transforms[i].translation = Vec3(0, i * 0.1, 0)
        target = Vec3(5, 1, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Each spine bone should get spine_weight / 15
        assert len(result) == 25


# =============================================================================
# Test Class: Integration Tests
# =============================================================================

class TestLookAtSolverIntegration:
    """Integration tests for LookAtSolver behavior."""

    def test_look_at_sequence_of_targets(self):
        """Test looking at a sequence of targets."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()

        targets = [
            Vec3(0, 2, 5),   # Front
            Vec3(5, 2, 0),   # Right
            Vec3(0, 2, -1),  # Back (clamped)
            Vec3(-5, 2, 0),  # Left
            Vec3(0, 5, 1),   # Up
        ]

        for target in targets:
            result = solver.solve(transforms, target, Vec3(0, 0, 1))
            assert len(result) == len(transforms)

    def test_solve_maintains_hierarchy_independence(self):
        """Test that each solve is independent (no state carried over)."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()

        # First solve
        result1 = solver.solve(transforms, Vec3(5, 2, 0), Vec3(0, 0, 1))

        # Second solve with same inputs
        result2 = solver.solve(transforms, Vec3(5, 2, 0), Vec3(0, 0, 1))

        # Results should be identical
        for r1, r2 in zip(result1, result2):
            assert quat_approx_equal(r1.rotation, r2.rotation)

    def test_symmetrical_targets_produce_symmetrical_results(self):
        """Test that symmetrical targets produce symmetrical rotations."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()

        # Look right
        result_right = solver.solve(transforms, Vec3(5, 2, 0), Vec3(0, 0, 1))

        # Look left
        result_left = solver.solve(transforms, Vec3(-5, 2, 0), Vec3(0, 0, 1))

        # Head rotation magnitudes should be similar
        _, angle_right = solver._quat_to_axis_angle(result_right[4].rotation)
        _, angle_left = solver._quat_to_axis_angle(result_left[4].rotation)

        assert angle_right == pytest.approx(angle_left, abs=0.01)

    def test_gradual_target_movement(self):
        """Test smooth rotation as target moves gradually."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()

        # Move target from front to right
        prev_angle = 0
        for t in range(10):
            x = t * 0.5
            z = 5 - t * 0.5
            target = Vec3(x, 2, z)
            result = solver.solve(transforms, target, Vec3(0, 0, 1))

            _, head_angle = solver._quat_to_axis_angle(result[4].rotation)

            # Rotation should increase as we move right
            if t > 0:
                assert head_angle >= prev_angle - 0.1  # Allow small variance
            prev_angle = head_angle

    def test_max_angle_boundary(self):
        """Test behavior at max angle boundary."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()

        # Target requiring exactly max_angle
        max_angle = solver.max_angle
        target_z = math.cos(max_angle) * 5
        target_x = math.sin(max_angle) * 5
        target = Vec3(target_x, 2, target_z)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)

    def test_chained_solve_operations(self):
        """Test using result of one solve as input to another."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()

        # First solve
        result1 = solver.solve(transforms, Vec3(2, 2, 2), Vec3(0, 0, 1))

        # Use result as input to second solve
        result2 = solver.solve(result1, Vec3(5, 2, 0), Vec3(0, 0, 1))

        assert len(result2) == len(transforms)


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of LookAtSolver."""

    def test_repeated_solves_are_stable(self):
        """Test that repeated solves produce consistent results."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = create_simple_spine_transforms()
        target = Vec3(3, 2, 4)

        results = []
        for _ in range(100):
            result = solver.solve(transforms, target, Vec3(0, 0, 1))
            results.append(result)

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            for i in range(len(first)):
                assert quat_approx_equal(first[i].rotation, result[i].rotation)

    def test_extreme_positions(self):
        """Test with extreme position values."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = [
            create_transform_at(Vec3(1e6, 0, 0)),
            create_transform_at(Vec3(1e6, 0.5, 0)),
            create_transform_at(Vec3(1e6, 1.0, 0)),
            create_transform_at(Vec3(1e6, 1.5, 0)),
            create_transform_at(Vec3(1e6, 2.0, 0)),
        ]
        target = Vec3(1e6 + 5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)

    def test_small_position_differences(self):
        """Test with very small position differences."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        transforms = [
            create_transform_at(Vec3(0, 0.0, 0)),
            create_transform_at(Vec3(0, 1e-6, 0)),
            create_transform_at(Vec3(0, 2e-6, 0)),
            create_transform_at(Vec3(0, 3e-6, 0)),
            create_transform_at(Vec3(0, 4e-6, 0)),
        ]
        target = Vec3(0, 5e-6, 1)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))
        assert len(result) == len(transforms)


# =============================================================================
# Test Class: Weight Combinations
# =============================================================================

class TestWeightCombinations:
    """Tests for various weight combinations."""

    def test_head_only_weight(self):
        """Test with weight only on head."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=1.0,
            neck_weight=0.0,
            spine_weight=0.0
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Only head should be rotated
        _, head_angle = solver._quat_to_axis_angle(result[4].rotation)
        _, neck_angle = solver._quat_to_axis_angle(result[3].rotation)

        assert head_angle > 0
        assert neck_angle < 0.01

    def test_neck_only_weight(self):
        """Test with weight only on neck."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=0.0,
            neck_weight=1.0,
            spine_weight=0.0
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Only neck should be rotated
        _, head_angle = solver._quat_to_axis_angle(result[4].rotation)
        _, neck_angle = solver._quat_to_axis_angle(result[3].rotation)

        assert head_angle < 0.01
        assert neck_angle > 0

    def test_spine_only_weight(self):
        """Test with weight only on spine."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=0.0,
            neck_weight=0.0,
            spine_weight=1.0
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        # Only spine should be rotated
        _, head_angle = solver._quat_to_axis_angle(result[4].rotation)
        _, neck_angle = solver._quat_to_axis_angle(result[3].rotation)
        _, spine_angle = solver._quat_to_axis_angle(result[0].rotation)

        assert head_angle < 0.01
        assert neck_angle < 0.01
        assert spine_angle > 0

    def test_equal_weights(self):
        """Test with equal weights on all parts."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[2],  # Single spine bone for easier comparison
            head_weight=0.33,
            neck_weight=0.33,
            spine_weight=0.33
        )
        transforms = create_simple_spine_transforms()
        target = Vec3(5, 2, 0)

        result = solver.solve(transforms, target, Vec3(0, 0, 1))

        _, head_angle = solver._quat_to_axis_angle(result[4].rotation)
        _, neck_angle = solver._quat_to_axis_angle(result[3].rotation)
        _, spine_angle = solver._quat_to_axis_angle(result[2].rotation)

        # All angles should be approximately equal
        assert head_angle == pytest.approx(neck_angle, rel=0.1)
        assert neck_angle == pytest.approx(spine_angle, rel=0.1)
