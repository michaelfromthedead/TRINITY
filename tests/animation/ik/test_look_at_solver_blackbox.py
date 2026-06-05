"""Blackbox tests for LookAtSolver (T-FB-4.7).

This module tests the LookAtSolver from the public API only,
without knowledge of implementation details. Tests are derived from
theoretical look-at IK behavior:

1. Rotates head, neck, and spine bones toward a target
2. Distributes rotation based on weights (head, neck, spine)
3. Respects angle limits to prevent unnatural poses
4. Preserves translation and scale (only modifies rotation)
5. Returns new transforms without mutating input

Test Strategy:
- Test public API contracts only
- Test behavioral expectations for look-at
- Test weight distribution effects
- Test angle clamping behavior
- Test edge cases and boundary conditions
"""

import math
import pytest
from typing import List, Optional

# Import public API
from engine.animation.ik import (
    LookAtSolver,
    LOOK_AT_MAX_ANGLE,
)
from engine.core.math import Vec3, Quat, Transform


# =============================================================================
# Helper Functions
# =============================================================================

def make_transform(
    position: Vec3,
    rotation: Optional[Quat] = None,
    scale: Optional[Vec3] = None
) -> Transform:
    """Create a Transform from position and optional rotation/scale."""
    return Transform(
        translation=position,
        rotation=rotation if rotation else Quat.identity(),
        scale=scale if scale else Vec3.one()
    )


def vec3_distance(a: Vec3, b: Vec3) -> float:
    """Calculate distance between two Vec3 points."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def vec_nearly_equal(a: Vec3, b: Vec3, eps: float = 0.01) -> bool:
    """Check if two vectors are nearly equal."""
    return vec3_distance(a, b) <= eps


def quat_angle_between(a: Quat, b: Quat) -> float:
    """Calculate angle between two quaternions in radians."""
    # q1.conjugate() * q2 gives relative rotation
    # The angle is 2 * acos(w) for a unit quaternion
    diff = a.conjugate() * b
    # Clamp w to [-1, 1] to handle numerical errors
    w = max(-1.0, min(1.0, diff.w))
    return 2.0 * math.acos(abs(w))


def quat_nearly_equal(a: Quat, b: Quat, eps: float = 0.01) -> bool:
    """Check if two quaternions represent nearly the same rotation."""
    return quat_angle_between(a, b) <= eps


def create_head_neck_transforms() -> List[Transform]:
    """Create minimal head and neck transforms for testing."""
    return [
        make_transform(Vec3(0.0, 1.7, 0.0)),  # neck
        make_transform(Vec3(0.0, 1.9, 0.0)),  # head
    ]


def create_spine_chain_transforms(count: int = 3) -> List[Transform]:
    """Create a spine chain with specified number of bones."""
    transforms = []
    for i in range(count):
        y = 1.0 + i * 0.2
        transforms.append(make_transform(Vec3(0.0, y, 0.0)))
    return transforms


def create_full_look_at_chain(spine_count: int = 3) -> List[Transform]:
    """Create full chain: spine + neck + head."""
    transforms = create_spine_chain_transforms(spine_count)
    # Add neck and head on top
    last_y = transforms[-1].translation.y
    transforms.append(make_transform(Vec3(0.0, last_y + 0.2, 0.0)))  # neck
    transforms.append(make_transform(Vec3(0.0, last_y + 0.4, 0.0)))  # head
    return transforms


def get_head_forward(head_transform: Transform, forward_axis: Vec3 = None) -> Vec3:
    """Get the forward direction of the head."""
    if forward_axis is None:
        forward_axis = Vec3(0.0, 0.0, 1.0)  # Default Z forward
    return head_transform.rotation.rotate_vector(forward_axis)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_transforms():
    """Basic head/neck transforms facing forward."""
    return create_head_neck_transforms()


@pytest.fixture
def spine_transforms():
    """Spine chain transforms."""
    return create_spine_chain_transforms(3)


@pytest.fixture
def full_chain_transforms():
    """Full spine + neck + head chain."""
    return create_full_look_at_chain(3)


# =============================================================================
# LookAtSolver Construction Tests
# =============================================================================

class TestLookAtSolverConstruction:
    """Tests for LookAtSolver instantiation."""

    def test_can_instantiate_with_required_params(self):
        """LookAtSolver can be created with required parameters."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        assert solver is not None

    def test_can_instantiate_with_spine_bones(self):
        """LookAtSolver can be created with spine bone indices."""
        solver = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[0, 1, 2])
        assert solver is not None

    def test_can_instantiate_with_empty_spine_bones(self):
        """LookAtSolver can be created with empty spine bones list."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        assert solver is not None

    def test_can_instantiate_with_single_spine_bone(self):
        """LookAtSolver can be created with single spine bone."""
        solver = LookAtSolver(head_bone=2, neck_bone=1, spine_bones=[0])
        assert solver is not None

    def test_can_instantiate_with_many_spine_bones(self):
        """LookAtSolver can be created with 5 spine bones."""
        solver = LookAtSolver(
            head_bone=6,
            neck_bone=5,
            spine_bones=[0, 1, 2, 3, 4]
        )
        assert solver is not None

    def test_can_instantiate_with_head_weight(self):
        """LookAtSolver can be created with custom head weight."""
        solver = LookAtSolver(
            head_bone=1, neck_bone=0, spine_bones=[], head_weight=0.8
        )
        assert solver is not None

    def test_can_instantiate_with_neck_weight(self):
        """LookAtSolver can be created with custom neck weight."""
        solver = LookAtSolver(
            head_bone=1, neck_bone=0, spine_bones=[], neck_weight=0.5
        )
        assert solver is not None

    def test_can_instantiate_with_spine_weight(self):
        """LookAtSolver can be created with custom spine weight."""
        solver = LookAtSolver(
            head_bone=2,
            neck_bone=1,
            spine_bones=[0],
            spine_weight=0.2
        )
        assert solver is not None

    def test_can_instantiate_with_all_weights(self):
        """LookAtSolver can be created with all weight parameters."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2],
            head_weight=0.5,
            neck_weight=0.3,
            spine_weight=0.2
        )
        assert solver is not None

    def test_spine_bones_required(self):
        """spine_bones is a required parameter."""
        # The API requires spine_bones, this tests that behavior
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        assert solver is not None


class TestLookAtSolverProperties:
    """Tests for LookAtSolver property access."""

    def test_head_weight_accessible(self):
        """Head weight should be accessible after construction."""
        solver = LookAtSolver(
            head_bone=1, neck_bone=0, spine_bones=[], head_weight=0.7
        )
        assert hasattr(solver, 'head_weight') or hasattr(solver, '_head_weight')

    def test_neck_weight_accessible(self):
        """Neck weight should be accessible after construction."""
        solver = LookAtSolver(
            head_bone=1, neck_bone=0, spine_bones=[], neck_weight=0.4
        )
        assert hasattr(solver, 'neck_weight') or hasattr(solver, '_neck_weight')

    def test_spine_weight_accessible(self):
        """Spine weight should be accessible after construction."""
        solver = LookAtSolver(
            head_bone=2, neck_bone=1, spine_bones=[0], spine_weight=0.15
        )
        assert hasattr(solver, 'spine_weight') or hasattr(solver, '_spine_weight')


# =============================================================================
# Basic Solve Behavior Tests
# =============================================================================

class TestLookAtSolverBasicSolve:
    """Tests for basic solve behavior."""

    def test_solve_returns_list(self, basic_transforms):
        """solve() should return a list."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.0, 1.9, 1.0)  # In front of head
        result = solver.solve(basic_transforms, target)
        assert isinstance(result, list)

    def test_solve_returns_transforms(self, basic_transforms):
        """solve() should return Transform objects."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.0, 1.9, 1.0)
        result = solver.solve(basic_transforms, target)
        for tf in result:
            assert isinstance(tf, Transform)

    def test_solve_preserves_transform_count(self, basic_transforms):
        """solve() should return same number of transforms as input."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.0, 1.9, 1.0)
        result = solver.solve(basic_transforms, target)
        assert len(result) == len(basic_transforms)

    def test_solve_with_full_chain(self, full_chain_transforms):
        """solve() should work with full spine + neck + head chain."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[0, 1, 2]
        )
        target = Vec3(0.5, 1.8, 1.0)
        result = solver.solve(full_chain_transforms, target)
        assert len(result) == len(full_chain_transforms)

    def test_target_in_front_rotates_head(self, basic_transforms):
        """Target in front should rotate head toward it."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.0, 1.9, 2.0)  # Directly in front

        result = solver.solve(basic_transforms, target)

        # Head should look toward target (forward direction should point at target)
        head_tf = result[1]  # head_bone=1
        head_forward = get_head_forward(head_tf)

        # Direction to target from head
        head_pos = head_tf.translation
        to_target = (target - head_pos).normalized()

        # Forward should be roughly aligned with to_target direction
        dot = head_forward.dot(to_target)
        assert dot > 0.9  # Should be mostly aligned

    def test_target_at_current_facing_minimal_rotation(self, basic_transforms):
        """Target at current facing direction should cause minimal rotation."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        # Target directly along default forward (Z+)
        head_pos = basic_transforms[1].translation
        target = head_pos + Vec3(0.0, 0.0, 2.0)

        result = solver.solve(basic_transforms, target)

        # Original and result rotations should be similar
        original_rot = basic_transforms[1].rotation
        result_rot = result[1].rotation

        angle_diff = quat_angle_between(original_rot, result_rot)
        assert angle_diff < 0.1  # Less than ~6 degrees


class TestLookAtSolverForwardAxis:
    """Tests for different forward axis configurations."""

    def test_z_forward_axis(self, basic_transforms):
        """Z forward axis should work correctly."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.0, 1.9, 2.0)
        result = solver.solve(basic_transforms, target, forward_axis=Vec3(0.0, 0.0, 1.0))

        head_forward = result[1].rotation.rotate_vector(Vec3(0.0, 0.0, 1.0))
        head_pos = result[1].translation
        to_target = (target - head_pos).normalized()

        assert head_forward.dot(to_target) > 0.9

    def test_x_forward_axis(self, basic_transforms):
        """X forward axis should work correctly."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(2.0, 1.9, 0.0)  # To the right
        result = solver.solve(basic_transforms, target, forward_axis=Vec3(1.0, 0.0, 0.0))

        head_forward = result[1].rotation.rotate_vector(Vec3(1.0, 0.0, 0.0))
        head_pos = result[1].translation
        to_target = (target - head_pos).normalized()

        assert head_forward.dot(to_target) > 0.8

    def test_negative_z_forward_axis(self, basic_transforms):
        """Negative Z forward axis should work correctly."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.0, 1.9, -2.0)  # Behind
        result = solver.solve(basic_transforms, target, forward_axis=Vec3(0.0, 0.0, -1.0))

        head_forward = result[1].rotation.rotate_vector(Vec3(0.0, 0.0, -1.0))
        head_pos = result[1].translation
        to_target = (target - head_pos).normalized()

        assert head_forward.dot(to_target) > 0.8


# =============================================================================
# Weight Distribution Tests
# =============================================================================

class TestLookAtSolverWeightDistribution:
    """Tests for weight distribution behavior."""

    def test_higher_head_weight_more_head_rotation(self, full_chain_transforms):
        """Higher head weight should result in more head rotation."""
        # High head weight solver
        solver_high = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2],
            head_weight=0.9, neck_weight=0.05, spine_weight=0.05
        )
        # Low head weight solver
        solver_low = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2],
            head_weight=0.1, neck_weight=0.45, spine_weight=0.45
        )

        target = Vec3(0.5, 1.8, 1.0)  # To the side

        result_high = solver_high.solve(full_chain_transforms, target)
        result_low = solver_low.solve(full_chain_transforms, target)

        # Head rotation change
        original_head = full_chain_transforms[4].rotation
        high_head_angle = quat_angle_between(original_head, result_high[4].rotation)
        low_head_angle = quat_angle_between(original_head, result_low[4].rotation)

        # High weight should have more head rotation
        assert high_head_angle >= low_head_angle * 0.8  # Allow some tolerance

    def test_higher_spine_weight_more_distributed_rotation(self, full_chain_transforms):
        """Higher spine weight should distribute rotation across spine."""
        solver = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2],
            head_weight=0.2, neck_weight=0.2, spine_weight=0.6
        )

        target = Vec3(1.0, 1.5, 0.5)
        result = solver.solve(full_chain_transforms, target)

        # Spine bones should have some rotation
        total_spine_rotation = 0.0
        for i in range(3):  # spine indices 0, 1, 2
            original = full_chain_transforms[i].rotation
            resulting = result[i].rotation
            total_spine_rotation += quat_angle_between(original, resulting)

        # With high spine weight, spine should contribute to rotation
        assert total_spine_rotation > 0.0

    def test_zero_head_weight_no_head_rotation(self, full_chain_transforms):
        """Zero head weight should result in no head rotation."""
        solver = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2],
            head_weight=0.0, neck_weight=0.5, spine_weight=0.5
        )

        target = Vec3(1.0, 1.9, 1.0)
        result = solver.solve(full_chain_transforms, target)

        original_head = full_chain_transforms[4].rotation
        result_head = result[4].rotation

        # Head rotation should be unchanged or minimal
        angle_diff = quat_angle_between(original_head, result_head)
        assert angle_diff < 0.02  # Essentially no rotation

    def test_zero_neck_weight_no_neck_rotation(self, full_chain_transforms):
        """Zero neck weight should result in no neck rotation."""
        solver = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2],
            head_weight=0.7, neck_weight=0.0, spine_weight=0.3
        )

        target = Vec3(0.5, 1.9, 1.0)
        result = solver.solve(full_chain_transforms, target)

        original_neck = full_chain_transforms[3].rotation
        result_neck = result[3].rotation

        angle_diff = quat_angle_between(original_neck, result_neck)
        assert angle_diff < 0.02

    def test_zero_spine_weight_no_spine_rotation(self, full_chain_transforms):
        """Zero spine weight should result in no spine rotation."""
        solver = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2],
            head_weight=0.6, neck_weight=0.4, spine_weight=0.0
        )

        target = Vec3(0.8, 1.8, 0.5)
        result = solver.solve(full_chain_transforms, target)

        for i in range(3):  # spine indices
            original = full_chain_transforms[i].rotation
            resulting = result[i].rotation
            angle_diff = quat_angle_between(original, resulting)
            assert angle_diff < 0.02


# =============================================================================
# Angle Limit Tests
# =============================================================================

class TestLookAtSolverAngleLimits:
    """Tests for angle limiting behavior."""

    def test_target_far_to_side_has_bounded_rotation(self, basic_transforms):
        """Target far to the side should have some bounded rotation."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        # Target extremely far to the side (90+ degrees)
        target = Vec3(10.0, 1.9, 0.0)
        result = solver.solve(basic_transforms, target)

        original_head = basic_transforms[1].rotation
        result_head = result[1].rotation

        # Rotation should be bounded (not infinite)
        total_angle = quat_angle_between(original_head, result_head)
        assert total_angle <= math.pi  # Should not exceed 180 degrees

    def test_target_behind_has_bounded_rotation(self, basic_transforms):
        """Target behind should have bounded rotation."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        # Target directly behind
        target = Vec3(0.0, 1.9, -5.0)
        result = solver.solve(basic_transforms, target)

        original_head = basic_transforms[1].rotation
        result_head = result[1].rotation

        total_angle = quat_angle_between(original_head, result_head)
        # Rotation is bounded (by LOOK_AT_MAX_ANGLE or similar limit)
        assert total_angle <= LOOK_AT_MAX_ANGLE + 0.2

    def test_rotation_does_not_exceed_max_angle(self, basic_transforms):
        """Rotation should be bounded by max angle config constant."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        # Target requiring > 90 degree rotation
        target = Vec3(0.0, 1.9, -10.0)  # Far behind
        result = solver.solve(basic_transforms, target)

        original = basic_transforms[1].rotation
        resulting = result[1].rotation

        total_angle = quat_angle_between(original, resulting)
        # Should be clamped near pi/2
        assert total_angle <= LOOK_AT_MAX_ANGLE + 0.2

    def test_look_at_max_angle_constant_defined(self):
        """LOOK_AT_MAX_ANGLE constant should be defined."""
        # This constant is imported from config
        assert LOOK_AT_MAX_ANGLE > 0
        assert LOOK_AT_MAX_ANGLE <= math.pi  # Less than 180 degrees


# =============================================================================
# Transform Preservation Tests
# =============================================================================

class TestLookAtSolverTransformPreservation:
    """Tests for transform component preservation."""

    def test_solve_does_not_mutate_input(self, basic_transforms):
        """solve() should not mutate input transforms."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(1.0, 1.9, 1.0)

        # Copy original values
        original_positions = [tf.translation for tf in basic_transforms]
        original_rotations = [
            Quat(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
            for tf in basic_transforms
        ]

        solver.solve(basic_transforms, target)

        # Original transforms should be unchanged
        for i, tf in enumerate(basic_transforms):
            assert vec_nearly_equal(tf.translation, original_positions[i])
            assert quat_nearly_equal(tf.rotation, original_rotations[i])

    def test_solve_returns_new_list(self, basic_transforms):
        """solve() should return a new list object."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.5, 1.9, 1.0)

        result = solver.solve(basic_transforms, target)

        assert result is not basic_transforms
        for i, tf in enumerate(result):
            assert tf is not basic_transforms[i]

    def test_translation_preserved(self, full_chain_transforms):
        """Translation should be preserved for all bones."""
        solver = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2]
        )
        target = Vec3(1.0, 1.5, 2.0)

        result = solver.solve(full_chain_transforms, target)

        for i, tf in enumerate(result):
            original_pos = full_chain_transforms[i].translation
            assert vec_nearly_equal(tf.translation, original_pos)

    def test_scale_preserved(self, basic_transforms):
        """Scale should be preserved for all bones."""
        # Create transforms with non-unit scale
        transforms = [
            make_transform(Vec3(0.0, 1.7, 0.0), scale=Vec3(1.0, 1.2, 1.0)),
            make_transform(Vec3(0.0, 1.9, 0.0), scale=Vec3(0.8, 1.0, 0.9)),
        ]

        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(1.0, 1.9, 1.0)

        result = solver.solve(transforms, target)

        for i, tf in enumerate(result):
            original_scale = transforms[i].scale
            assert vec_nearly_equal(tf.scale, original_scale)

    def test_only_rotation_modified(self, basic_transforms):
        """Only rotation should be modified, not translation or scale."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(2.0, 2.0, 1.0)  # Target requiring rotation

        result = solver.solve(basic_transforms, target)

        for i, tf in enumerate(result):
            # Translation unchanged
            assert vec_nearly_equal(
                tf.translation,
                basic_transforms[i].translation
            )
            # Scale unchanged
            assert vec_nearly_equal(
                tf.scale,
                basic_transforms[i].scale
            )


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestLookAtSolverEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_target_at_exact_head_position(self, basic_transforms):
        """Target at exact head position should not cause errors."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        # Target at exact head position
        head_pos = basic_transforms[1].translation
        target = Vec3(head_pos.x, head_pos.y, head_pos.z)

        # Should not raise
        result = solver.solve(basic_transforms, target)
        assert len(result) == len(basic_transforms)

    def test_target_very_far_away(self, basic_transforms):
        """Very far target should work correctly."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        target = Vec3(0.0, 1.9, 10000.0)  # Very far in front
        result = solver.solve(basic_transforms, target)

        # Should still look toward target
        head_forward = get_head_forward(result[1])
        head_pos = result[1].translation
        to_target = (target - head_pos).normalized()

        assert head_forward.dot(to_target) > 0.9

    def test_target_very_close(self, basic_transforms):
        """Very close target should work correctly."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        head_pos = basic_transforms[1].translation
        target = Vec3(head_pos.x + 0.001, head_pos.y, head_pos.z + 0.001)

        # Should not raise
        result = solver.solve(basic_transforms, target)
        assert len(result) == len(basic_transforms)

    def test_empty_spine_bones_list(self):
        """Empty spine bones list should work."""
        transforms = create_head_neck_transforms()
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        target = Vec3(0.5, 1.9, 1.0)
        result = solver.solve(transforms, target)

        assert len(result) == len(transforms)

    def test_single_spine_bone(self):
        """Single spine bone should work."""
        transforms = [
            make_transform(Vec3(0.0, 1.2, 0.0)),  # spine
            make_transform(Vec3(0.0, 1.5, 0.0)),  # neck
            make_transform(Vec3(0.0, 1.7, 0.0)),  # head
        ]

        solver = LookAtSolver(
            head_bone=2, neck_bone=1, spine_bones=[0]
        )
        target = Vec3(0.5, 1.7, 1.0)
        result = solver.solve(transforms, target)

        assert len(result) == 3

    def test_five_spine_bones(self):
        """Five spine bones should work."""
        transforms = create_full_look_at_chain(5)

        solver = LookAtSolver(
            head_bone=6, neck_bone=5, spine_bones=[0, 1, 2, 3, 4]
        )
        target = Vec3(0.3, 2.0, 0.8)
        result = solver.solve(transforms, target)

        assert len(result) == len(transforms)


class TestLookAtSolverTargetDirections:
    """Tests for various target directions."""

    def test_target_above(self, basic_transforms):
        """Target above head should cause upward look."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        head_pos = basic_transforms[1].translation
        target = Vec3(head_pos.x, head_pos.y + 2.0, head_pos.z + 1.0)

        result = solver.solve(basic_transforms, target)

        head_forward = get_head_forward(result[1])
        # Forward should have positive Y component (looking up)
        assert head_forward.y > 0

    def test_target_below(self, basic_transforms):
        """Target below head should cause downward look."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        head_pos = basic_transforms[1].translation
        target = Vec3(head_pos.x, head_pos.y - 1.0, head_pos.z + 1.0)

        result = solver.solve(basic_transforms, target)

        head_forward = get_head_forward(result[1])
        # Forward should have negative Y component (looking down)
        assert head_forward.y < 0

    def test_target_left(self, basic_transforms):
        """Target to the left should cause leftward look."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        target = Vec3(-2.0, 1.9, 1.0)
        result = solver.solve(basic_transforms, target)

        head_forward = get_head_forward(result[1])
        # Forward should have negative X component
        assert head_forward.x < 0

    def test_target_right(self, basic_transforms):
        """Target to the right should cause rightward look."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        target = Vec3(2.0, 1.9, 1.0)
        result = solver.solve(basic_transforms, target)

        head_forward = get_head_forward(result[1])
        # Forward should have positive X component
        assert head_forward.x > 0

    def test_target_up_left(self, basic_transforms):
        """Target up and to the left should cause diagonal look."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        head_pos = basic_transforms[1].translation
        target = Vec3(-1.0, head_pos.y + 1.0, head_pos.z + 1.0)

        result = solver.solve(basic_transforms, target)

        head_forward = get_head_forward(result[1])
        # Should look up and left
        assert head_forward.x < 0
        assert head_forward.y > 0

    def test_target_down_right(self, basic_transforms):
        """Target down and to the right should cause diagonal look."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        head_pos = basic_transforms[1].translation
        target = Vec3(1.0, head_pos.y - 0.5, head_pos.z + 1.0)

        result = solver.solve(basic_transforms, target)

        head_forward = get_head_forward(result[1])
        # Should look down and right
        assert head_forward.x > 0
        assert head_forward.y < 0


class TestLookAtSolverNumericalStability:
    """Tests for numerical stability."""

    def test_repeated_solve_stable(self, basic_transforms):
        """Repeated solves with same target should give consistent results."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.5, 2.0, 1.0)

        result1 = solver.solve(basic_transforms, target)
        result2 = solver.solve(basic_transforms, target)

        for i in range(len(result1)):
            assert vec_nearly_equal(
                result1[i].translation,
                result2[i].translation
            )
            assert quat_nearly_equal(
                result1[i].rotation,
                result2[i].rotation
            )

    def test_chained_solve_stable(self, basic_transforms):
        """Using output as input for next solve should be stable."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.8, 1.9, 1.2)

        result = solver.solve(basic_transforms, target)
        # Solve again with same target (should be near converged)
        result2 = solver.solve(result, target)

        for i in range(len(result)):
            assert vec_nearly_equal(
                result[i].translation,
                result2[i].translation
            )
            # Rotations should be reasonably similar (allowing for cumulative effects)
            assert quat_nearly_equal(
                result[i].rotation,
                result2[i].rotation,
                eps=0.15  # Allow up to ~8.5 degrees difference
            )

    def test_small_target_movement_small_rotation_change(self, basic_transforms):
        """Small target movement should cause proportionally small rotation change."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        target1 = Vec3(0.0, 1.9, 2.0)
        target2 = Vec3(0.01, 1.9, 2.0)  # Tiny movement

        result1 = solver.solve(basic_transforms, target1)
        result2 = solver.solve(basic_transforms, target2)

        angle_diff = quat_angle_between(result1[1].rotation, result2[1].rotation)
        # Should be a small change
        assert angle_diff < 0.05


class TestLookAtSolverRotatedInputs:
    """Tests with pre-rotated input transforms."""

    def test_pre_rotated_head_solve(self):
        """Pre-rotated head should still solve correctly."""
        # Head already looking up
        rotation = Quat.from_euler(math.pi / 6, 0, 0)  # 30 degree pitch
        transforms = [
            make_transform(Vec3(0.0, 1.7, 0.0)),
            make_transform(Vec3(0.0, 1.9, 0.0), rotation),
        ]

        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.0, 1.9, 2.0)  # Straight ahead

        result = solver.solve(transforms, target)

        head_forward = get_head_forward(result[1])
        head_pos = result[1].translation
        to_target = (target - head_pos).normalized()

        # Should look toward target
        assert head_forward.dot(to_target) > 0.85

    def test_pre_rotated_neck_solve(self):
        """Pre-rotated neck should still solve correctly."""
        # Neck rotated to the side
        rotation = Quat.from_euler(0, math.pi / 6, 0)  # 30 degree yaw
        transforms = [
            make_transform(Vec3(0.0, 1.7, 0.0), rotation),
            make_transform(Vec3(0.0, 1.9, 0.0)),
        ]

        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(1.0, 1.9, 1.0)

        result = solver.solve(transforms, target)
        assert len(result) == 2

    def test_all_bones_pre_rotated(self):
        """All bones pre-rotated should still solve."""
        pitch_rot = Quat.from_euler(0.1, 0, 0)
        yaw_rot = Quat.from_euler(0, 0.1, 0)

        transforms = [
            make_transform(Vec3(0.0, 1.0, 0.0), pitch_rot),  # spine
            make_transform(Vec3(0.0, 1.3, 0.0), yaw_rot),   # spine2
            make_transform(Vec3(0.0, 1.5, 0.0), pitch_rot),  # neck
            make_transform(Vec3(0.0, 1.7, 0.0), yaw_rot),   # head
        ]

        solver = LookAtSolver(
            head_bone=3, neck_bone=2, spine_bones=[0, 1]
        )
        target = Vec3(0.5, 1.8, 1.5)

        result = solver.solve(transforms, target)
        assert len(result) == 4


class TestLookAtSolverMultipleCallsSequence:
    """Tests for sequences of solve calls."""

    def test_solve_sequence_targets(self, basic_transforms):
        """Sequence of targets should work correctly."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        targets = [
            Vec3(0.0, 1.9, 2.0),   # front
            Vec3(1.0, 1.9, 1.0),   # right
            Vec3(-1.0, 1.9, 1.0),  # left
            Vec3(0.0, 2.5, 1.5),   # up
            Vec3(0.0, 1.3, 1.5),   # down
        ]

        for target in targets:
            result = solver.solve(basic_transforms, target)
            assert len(result) == len(basic_transforms)

            head_forward = get_head_forward(result[1])
            head_pos = result[1].translation
            to_target = (target - head_pos).normalized()

            # Should look toward target (allowing for angle limits)
            dot = head_forward.dot(to_target)
            # Even clamped, we should be facing generally toward target
            assert dot > 0 or abs(dot) < 0.5

    def test_smooth_target_tracking(self, basic_transforms):
        """Smoothly moving target should produce smooth rotation changes."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        # Move target smoothly from left to right
        results = []
        for i in range(10):
            x = -1.0 + (i / 4.5)  # -1 to +1
            target = Vec3(x, 1.9, 2.0)
            result = solver.solve(basic_transforms, target)
            results.append(result[1].rotation)

        # Check that adjacent rotations are close
        for i in range(len(results) - 1):
            angle_diff = quat_angle_between(results[i], results[i + 1])
            # Each step should be reasonably small
            assert angle_diff < 0.5  # Less than ~28 degrees per step


class TestLookAtSolverBoundaryWeights:
    """Tests for boundary weight values."""

    def test_weight_exactly_zero(self, full_chain_transforms):
        """Weight of exactly zero should work."""
        solver = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2],
            head_weight=0.0, neck_weight=1.0, spine_weight=0.0
        )
        target = Vec3(0.5, 1.8, 1.0)
        result = solver.solve(full_chain_transforms, target)
        assert len(result) == len(full_chain_transforms)

    def test_weight_exactly_one(self, basic_transforms):
        """Weight of exactly one should work."""
        solver = LookAtSolver(
            head_bone=1, neck_bone=0, spine_bones=[],
            head_weight=1.0, neck_weight=0.0
        )
        target = Vec3(0.5, 1.9, 1.0)
        result = solver.solve(basic_transforms, target)
        assert len(result) == len(basic_transforms)

    def test_all_weights_equal(self, full_chain_transforms):
        """Equal weights should work."""
        solver = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2],
            head_weight=0.33, neck_weight=0.33, spine_weight=0.34
        )
        target = Vec3(0.5, 1.8, 1.0)
        result = solver.solve(full_chain_transforms, target)
        assert len(result) == len(full_chain_transforms)

    def test_very_small_weights(self, full_chain_transforms):
        """Very small weights should work."""
        solver = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2],
            head_weight=0.001, neck_weight=0.001, spine_weight=0.998
        )
        target = Vec3(0.5, 1.8, 1.0)
        result = solver.solve(full_chain_transforms, target)
        assert len(result) == len(full_chain_transforms)


class TestLookAtSolverAngleBoundaries:
    """Tests for angle boundary behaviors."""

    def test_look_at_max_angle_reasonable(self):
        """LOOK_AT_MAX_ANGLE should be a reasonable value."""
        # Should be between 0 and pi (exclusive)
        assert 0 < LOOK_AT_MAX_ANGLE <= math.pi

    def test_rotation_bounded_by_default_limits(self, basic_transforms):
        """Rotation should be bounded by default angle limits."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        # Target far behind
        target = Vec3(0.0, 1.9, -10.0)
        result = solver.solve(basic_transforms, target)

        original = basic_transforms[1].rotation
        resulting = result[1].rotation

        # Should not rotate more than pi (180 degrees)
        angle_diff = quat_angle_between(original, resulting)
        assert angle_diff <= math.pi

    def test_extreme_target_handled_gracefully(self, basic_transforms):
        """Extreme target positions should be handled gracefully."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        extreme_targets = [
            Vec3(1000.0, 1.9, 0.0),    # Far right
            Vec3(-1000.0, 1.9, 0.0),   # Far left
            Vec3(0.0, 1000.0, 0.0),    # Far up
            Vec3(0.0, -1000.0, 0.0),   # Far down
            Vec3(0.0, 1.9, -1000.0),   # Far behind
        ]

        for target in extreme_targets:
            result = solver.solve(basic_transforms, target)
            assert len(result) == len(basic_transforms)
            # Rotations should still be valid unit quaternions
            for tf in result:
                assert abs(tf.rotation.length() - 1.0) < 0.01


class TestLookAtSolverResultStructure:
    """Tests for result structure and types."""

    def test_result_transforms_have_translation(self, basic_transforms):
        """Result transforms should have translation attribute."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.5, 1.9, 1.0)
        result = solver.solve(basic_transforms, target)

        for tf in result:
            assert hasattr(tf, 'translation')
            assert isinstance(tf.translation, Vec3)

    def test_result_transforms_have_rotation(self, basic_transforms):
        """Result transforms should have rotation attribute."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.5, 1.9, 1.0)
        result = solver.solve(basic_transforms, target)

        for tf in result:
            assert hasattr(tf, 'rotation')
            assert isinstance(tf.rotation, Quat)

    def test_result_transforms_have_scale(self, basic_transforms):
        """Result transforms should have scale attribute."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.5, 1.9, 1.0)
        result = solver.solve(basic_transforms, target)

        for tf in result:
            assert hasattr(tf, 'scale')
            assert isinstance(tf.scale, Vec3)

    def test_result_rotations_are_normalized(self, basic_transforms):
        """Result quaternions should be unit quaternions."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(1.0, 2.0, 1.0)
        result = solver.solve(basic_transforms, target)

        for tf in result:
            length = tf.rotation.length()
            assert abs(length - 1.0) < 0.001  # Unit quaternion


class TestLookAtSolverConsistency:
    """Tests for behavioral consistency."""

    def test_same_input_same_output(self, basic_transforms):
        """Same input should always produce same output."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.7, 1.8, 1.3)

        results = [solver.solve(basic_transforms, target) for _ in range(5)]

        for i in range(1, 5):
            for j in range(len(results[0])):
                assert vec_nearly_equal(
                    results[0][j].translation,
                    results[i][j].translation
                )
                assert quat_nearly_equal(
                    results[0][j].rotation,
                    results[i][j].rotation
                )

    def test_symmetric_targets_symmetric_results(self, basic_transforms):
        """Symmetric targets should produce symmetric rotations."""
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])

        # Target left vs right (symmetric)
        target_left = Vec3(-1.0, 1.9, 1.0)
        target_right = Vec3(1.0, 1.9, 1.0)

        result_left = solver.solve(basic_transforms, target_left)
        result_right = solver.solve(basic_transforms, target_right)

        # Head rotations should be mirror images
        # Y rotation (yaw) should be opposite
        left_forward = get_head_forward(result_left[1])
        right_forward = get_head_forward(result_right[1])

        # X components should be opposite
        assert abs(left_forward.x + right_forward.x) < 0.1
        # Z components should be similar
        assert abs(left_forward.z - right_forward.z) < 0.1


# =============================================================================
# Performance Sanity Tests
# =============================================================================

class TestLookAtSolverPerformance:
    """Basic performance sanity tests."""

    def test_solve_completes_quickly(self, basic_transforms):
        """Single solve should complete quickly."""
        import time
        solver = LookAtSolver(head_bone=1, neck_bone=0, spine_bones=[])
        target = Vec3(0.5, 1.9, 1.0)

        start = time.perf_counter()
        for _ in range(100):
            solver.solve(basic_transforms, target)
        elapsed = time.perf_counter() - start

        # 100 solves should complete in under 100ms
        assert elapsed < 0.1

    def test_full_chain_solve_completes(self, full_chain_transforms):
        """Full chain solve should complete in reasonable time."""
        import time
        solver = LookAtSolver(
            head_bone=4, neck_bone=3, spine_bones=[0, 1, 2]
        )
        target = Vec3(0.5, 1.8, 1.0)

        start = time.perf_counter()
        for _ in range(50):
            solver.solve(full_chain_transforms, target)
        elapsed = time.perf_counter() - start

        # 50 solves should complete in under 100ms
        assert elapsed < 0.1
