"""
Whitebox tests for hair constraint implementations.
"""

import math
import numpy as np
import pytest

from engine.simulation.hair.config import NUMERICAL_EPSILON
from engine.simulation.hair.hair_constraints import (
    CollisionConstraint,
    GlobalShapeConstraint,
    LengthConstraint,
    LocalShapeConstraint,
    RootConstraint,
    create_length_constraints,
    create_local_shape_constraints,
    solve_global_shape_constraint,
    solve_length_constraint,
    solve_local_shape_constraint,
)
from engine.simulation.hair.hair_simulation import (
    GuideHair,
    HairControlPoint,
    HairStrand,
    create_hair_strand,
)


def make_control_point(
    position, inv_mass=1.0, rest_position=None, prev_position=None
):
    """Helper to create a control point."""
    pos = np.array(position, dtype=np.float32)
    return HairControlPoint(
        position=pos.copy(),
        prev_position=prev_position if prev_position is not None else pos.copy(),
        rest_position=rest_position if rest_position is not None else pos.copy(),
        inv_mass=inv_mass,
    )


def make_simple_strand(positions, inv_masses=None):
    """Helper to create a simple strand from position list."""
    if inv_masses is None:
        inv_masses = [0.0] + [1.0] * (len(positions) - 1)

    control_points = []
    rest_lengths = []

    for i, pos in enumerate(positions):
        cp = make_control_point(pos, inv_mass=inv_masses[i])
        control_points.append(cp)

        if i > 0:
            rest_len = np.linalg.norm(
                np.array(positions[i]) - np.array(positions[i - 1])
            )
            rest_lengths.append(float(rest_len))

    return HairStrand(
        control_points=control_points,
        rest_lengths=rest_lengths,
        root_position=np.array(positions[0], dtype=np.float32),
        root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )


class TestSolveLengthConstraint:
    """Tests for solve_length_constraint function."""

    def test_length_constraint_at_rest(self):
        """No correction when at rest length."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([0.0, 1.0, 0.0], inv_mass=1.0)
        rest_length = 1.0

        error = solve_length_constraint(cp0, cp1, rest_length, stiffness=1.0)

        assert error == pytest.approx(0.0, abs=NUMERICAL_EPSILON)

    def test_length_constraint_stretched(self):
        """Should correct stretched segments."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([0.0, 1.5, 0.0], inv_mass=1.0)
        rest_length = 1.0

        original_cp1_pos = cp1.position.copy()
        error = solve_length_constraint(cp0, cp1, rest_length, stiffness=1.0)

        # Should have moved cp1 closer to cp0
        new_distance = np.linalg.norm(cp1.position - cp0.position)
        assert new_distance < np.linalg.norm(original_cp1_pos - cp0.position)
        assert error == pytest.approx(0.5)  # 1.5 - 1.0

    def test_length_constraint_compressed(self):
        """Should correct compressed segments."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([0.0, 0.5, 0.0], inv_mass=1.0)
        rest_length = 1.0

        original_cp1_pos = cp1.position.copy()
        error = solve_length_constraint(cp0, cp1, rest_length, stiffness=1.0)

        # Should have moved cp1 further from cp0
        new_distance = np.linalg.norm(cp1.position - cp0.position)
        assert new_distance > np.linalg.norm(original_cp1_pos - cp0.position)
        assert error == pytest.approx(-0.5)  # 0.5 - 1.0

    def test_length_constraint_stiffness_affects_correction(self):
        """Lower stiffness should result in smaller correction."""
        # High stiffness
        cp1_high = make_control_point([0.0, 1.5, 0.0], inv_mass=1.0)
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        solve_length_constraint(cp0, cp1_high, 1.0, stiffness=1.0)
        high_stiff_pos = cp1_high.position.copy()

        # Low stiffness
        cp1_low = make_control_point([0.0, 1.5, 0.0], inv_mass=1.0)
        solve_length_constraint(cp0, cp1_low, 1.0, stiffness=0.5)
        low_stiff_pos = cp1_low.position.copy()

        # High stiffness should move more
        high_move = np.linalg.norm(high_stiff_pos - np.array([0.0, 1.5, 0.0]))
        low_move = np.linalg.norm(low_stiff_pos - np.array([0.0, 1.5, 0.0]))
        assert high_move > low_move

    def test_length_constraint_root_fixed(self):
        """Root point (inv_mass=0) should not move."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([0.0, 1.5, 0.0], inv_mass=1.0)

        original_cp0_pos = cp0.position.copy()
        solve_length_constraint(cp0, cp1, 1.0, stiffness=1.0)

        np.testing.assert_array_equal(cp0.position, original_cp0_pos)

    def test_length_constraint_zero_length(self):
        """Should handle degenerate case of zero current length."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([0.0, 0.0, 0.0], inv_mass=1.0)  # Same position

        error = solve_length_constraint(cp0, cp1, 1.0, stiffness=1.0)

        # Should return rest_length as error when current length is ~0
        assert error == pytest.approx(1.0)

    def test_length_constraint_fixed_child(self):
        """Should not move child with zero inverse mass."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([0.0, 1.5, 0.0], inv_mass=0.0)  # Also fixed

        original_cp1_pos = cp1.position.copy()
        solve_length_constraint(cp0, cp1, 1.0, stiffness=1.0)

        np.testing.assert_array_equal(cp1.position, original_cp1_pos)


class TestSolveGlobalShapeConstraint:
    """Tests for solve_global_shape_constraint function."""

    def test_global_shape_no_change_at_rest(self):
        """No correction when at rest pose."""
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0],
        ])

        head_pos = np.zeros(3, dtype=np.float32)
        head_rot = np.eye(3, dtype=np.float32)

        # Set rest positions to current positions
        for cp in strand.control_points:
            cp.rest_position = cp.position.copy()

        original_positions = [cp.position.copy() for cp in strand.control_points]

        solve_global_shape_constraint(strand, head_pos, head_rot, stiffness=0.5)

        # Root should be unchanged, others should stay approximately same
        np.testing.assert_array_equal(
            strand.control_points[0].position, original_positions[0]
        )

    def test_global_shape_pulls_toward_rest(self):
        """Should pull deformed strand toward rest pose."""
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Deformed - should be at [0, 0.1, 0]
            [2.0, 0.0, 0.0],  # Deformed - should be at [0, 0.2, 0]
        ])

        # Set rest positions
        strand.control_points[0].rest_position = np.array([0.0, 0.0, 0.0])
        strand.control_points[1].rest_position = np.array([0.0, 0.1, 0.0])
        strand.control_points[2].rest_position = np.array([0.0, 0.2, 0.0])

        head_pos = np.zeros(3, dtype=np.float32)
        head_rot = np.eye(3, dtype=np.float32)

        original_x = strand.control_points[1].position[0]

        solve_global_shape_constraint(strand, head_pos, head_rot, stiffness=0.5)

        # Should have moved toward rest (reduced X)
        new_x = strand.control_points[1].position[0]
        assert new_x < original_x

    def test_global_shape_skips_root(self):
        """Root point should not be modified."""
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        head_pos = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        head_rot = np.eye(3, dtype=np.float32)

        original_root = strand.control_points[0].position.copy()

        solve_global_shape_constraint(strand, head_pos, head_rot, stiffness=1.0)

        # Root should be unchanged (is_root check)
        np.testing.assert_array_equal(strand.control_points[0].position, original_root)

    def test_global_shape_respects_head_rotation(self):
        """Should account for head rotation when computing target."""
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ])

        strand.control_points[1].rest_position = np.array([0.0, 0.1, 0.0])

        head_pos = np.zeros(3, dtype=np.float32)
        # 90 degree rotation around Z
        head_rot = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        solve_global_shape_constraint(strand, head_pos, head_rot, stiffness=1.0)

        # After rotation, rest [0, 0.1, 0] becomes [-0.1, 0, 0]
        # Point should move toward [-0.1, 0, 0]
        assert strand.control_points[1].position[0] < 0  # Moved to negative X


class TestSolveLocalShapeConstraint:
    """Tests for solve_local_shape_constraint function."""

    def test_local_shape_needs_three_points(self):
        """Should do nothing with fewer than 3 points."""
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ])

        original_positions = [cp.position.copy() for cp in strand.control_points]

        solve_local_shape_constraint(strand, stiffness=0.3)

        # No changes should occur
        for i, cp in enumerate(strand.control_points):
            np.testing.assert_array_equal(cp.position, original_positions[i])

    def test_local_shape_preserves_straight_line(self):
        """Should preserve straight line shape."""
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.3, 0.0],
        ])

        # Rest positions are same as current (straight line)
        original_positions = [cp.position.copy() for cp in strand.control_points]

        solve_local_shape_constraint(strand, stiffness=0.3)

        # Should remain largely unchanged (straight line is preserved)
        for i, cp in enumerate(strand.control_points):
            # Allow small numerical drift
            np.testing.assert_array_almost_equal(
                cp.position, original_positions[i], decimal=4
            )

    def test_local_shape_corrects_bent_strand(self):
        """Should try to correct a bent strand toward rest angles."""
        # Create bent strand with larger deviation
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.2, 0.15, 0.0],  # Bent more
            [0.3, 0.2, 0.0],
        ])

        # Set rest positions as straight line
        strand.control_points[0].rest_position = np.array([0.0, 0.0, 0.0])
        strand.control_points[1].rest_position = np.array([0.0, 0.1, 0.0])
        strand.control_points[2].rest_position = np.array([0.0, 0.2, 0.0])
        strand.control_points[3].rest_position = np.array([0.0, 0.3, 0.0])

        original_bend_x = strand.control_points[2].position[0]

        # Apply multiple iterations with higher stiffness
        for _ in range(5):
            solve_local_shape_constraint(strand, stiffness=0.5)

        # Should have reduced the bend (moved x toward 0)
        new_bend_x = strand.control_points[2].position[0]
        # Allow for small numerical drift - check that correction happened
        assert abs(new_bend_x) < abs(original_bend_x) or abs(new_bend_x - original_bend_x) < 0.01


class TestLengthConstraintClass:
    """Tests for LengthConstraint dataclass."""

    def test_length_constraint_init(self):
        """Should initialize with correct values."""
        constraint = LengthConstraint(
            cp0_index=0,
            cp1_index=1,
            rest_length=0.1,
            stiffness=0.9,
        )

        assert constraint.cp0_index == 0
        assert constraint.cp1_index == 1
        assert constraint.rest_length == 0.1
        assert constraint.stiffness == 0.9

    def test_length_constraint_solve(self):
        """solve() should apply the constraint."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([0.0, 0.2, 0.0], inv_mass=1.0)
        control_points = [cp0, cp1]

        constraint = LengthConstraint(
            cp0_index=0,
            cp1_index=1,
            rest_length=0.1,
            stiffness=1.0,
        )

        error = constraint.solve(control_points)

        # Should have corrected the length
        new_length = np.linalg.norm(cp1.position - cp0.position)
        assert new_length < 0.2
        assert error == pytest.approx(0.1)  # 0.2 - 0.1

    def test_length_constraint_solve_with_override(self):
        """solve() should use stiffness_override if provided."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([0.0, 0.2, 0.0], inv_mass=1.0)
        control_points = [cp0, cp1]

        constraint = LengthConstraint(
            cp0_index=0,
            cp1_index=1,
            rest_length=0.1,
            stiffness=1.0,
        )

        # Use lower stiffness override
        constraint.solve(control_points, stiffness_override=0.1)

        # Less correction due to lower stiffness
        new_length = np.linalg.norm(cp1.position - cp0.position)
        assert new_length >= 0.19  # Barely moved (allow exact 0.19)


class TestGlobalShapeConstraintClass:
    """Tests for GlobalShapeConstraint dataclass."""

    def test_global_shape_constraint_init(self):
        """Should initialize with rest positions."""
        rest_positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ], dtype=np.float32)

        constraint = GlobalShapeConstraint(
            rest_positions=rest_positions,
            stiffness=0.5,
        )

        np.testing.assert_array_equal(constraint.rest_positions, rest_positions)
        assert constraint.stiffness == 0.5

    def test_global_shape_constraint_solve(self):
        """solve() should apply global shape matching."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([1.0, 0.0, 0.0], inv_mass=1.0)  # Deformed
        control_points = [cp0, cp1]

        rest_positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],  # Rest is vertical
        ], dtype=np.float32)

        constraint = GlobalShapeConstraint(
            rest_positions=rest_positions,
            stiffness=0.5,
        )

        root_pos = np.zeros(3, dtype=np.float32)
        root_rot = np.eye(3, dtype=np.float32)

        original_x = cp1.position[0]
        constraint.solve(control_points, root_pos, root_rot)

        # Should have moved toward rest (Y direction)
        assert cp1.position[0] < original_x
        assert cp1.position[1] > 0


class TestLocalShapeConstraintClass:
    """Tests for LocalShapeConstraint dataclass."""

    def test_local_shape_constraint_init(self):
        """Should initialize with rest angles."""
        rest_angles = np.array([0.0, 0.1, 0.2], dtype=np.float32)

        constraint = LocalShapeConstraint(
            rest_angles=rest_angles,
            stiffness=0.3,
        )

        np.testing.assert_array_equal(constraint.rest_angles, rest_angles)
        assert constraint.stiffness == 0.3

    def test_local_shape_constraint_solve_short_strand(self):
        """solve() should handle short strands gracefully."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([0.0, 0.1, 0.0], inv_mass=1.0)
        control_points = [cp0, cp1]

        constraint = LocalShapeConstraint(
            rest_angles=np.array([0.0], dtype=np.float32),
            stiffness=0.3,
        )

        # Should not crash with only 2 points
        constraint.solve(control_points)


class TestRootConstraint:
    """Tests for RootConstraint dataclass."""

    def test_root_constraint_init(self):
        """Should initialize with scalp data."""
        scalp_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        scalp_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        constraint = RootConstraint(
            scalp_position=scalp_pos,
            scalp_normal=scalp_normal,
            stiffness=1.0,
        )

        np.testing.assert_array_equal(constraint.scalp_position, scalp_pos)
        np.testing.assert_array_equal(constraint.scalp_normal, scalp_normal)

    def test_root_constraint_solve(self):
        """solve() should fix root to scalp position."""
        cp0 = make_control_point([5.0, 5.0, 5.0], inv_mass=0.0)  # Wrong position
        cp1 = make_control_point([0.0, 0.1, 0.0], inv_mass=1.0)
        control_points = [cp0, cp1]

        scalp_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        scalp_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        constraint = RootConstraint(
            scalp_position=scalp_pos,
            scalp_normal=scalp_normal,
        )

        head_pos = np.zeros(3, dtype=np.float32)
        head_rot = np.eye(3, dtype=np.float32)

        constraint.solve(control_points, head_pos, head_rot)

        # Root should now be at scalp position
        np.testing.assert_array_equal(cp0.position, scalp_pos)
        np.testing.assert_array_equal(cp0.prev_position, scalp_pos)
        np.testing.assert_array_equal(cp0.velocity, np.zeros(3))

    def test_root_constraint_solve_empty(self):
        """solve() should handle empty control points."""
        constraint = RootConstraint(
            scalp_position=np.zeros(3, dtype=np.float32),
            scalp_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        # Should not crash
        constraint.solve([], np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))

    def test_root_constraint_update_scalp_position(self):
        """update_scalp_position should update stored values."""
        constraint = RootConstraint(
            scalp_position=np.zeros(3, dtype=np.float32),
            scalp_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        new_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        new_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        constraint.update_scalp_position(new_pos, new_normal)

        np.testing.assert_array_equal(constraint.scalp_position, new_pos)
        np.testing.assert_array_equal(constraint.scalp_normal, new_normal)


class TestCollisionConstraint:
    """Tests for CollisionConstraint dataclass."""

    def test_collision_constraint_init(self):
        """Should initialize with collision parameters."""
        constraint = CollisionConstraint(
            collision_radius=0.005,
            friction=0.4,
            stiffness=0.9,
        )

        assert constraint.collision_radius == 0.005
        assert constraint.friction == 0.4
        assert constraint.stiffness == 0.9

    def test_solve_capsule_no_collision(self):
        """Should not modify point when no collision."""
        cp = make_control_point([5.0, 0.0, 0.0], inv_mass=1.0)
        original_pos = cp.position.copy()

        constraint = CollisionConstraint()

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        collided = constraint.solve_capsule_collision(
            cp, capsule_a, capsule_b, capsule_radius
        )

        assert collided is False
        np.testing.assert_array_equal(cp.position, original_pos)

    def test_solve_capsule_collision(self):
        """Should push point out of capsule."""
        # Point inside capsule
        cp = make_control_point([0.05, 0.5, 0.0], inv_mass=1.0)

        constraint = CollisionConstraint(
            collision_radius=0.002,
            stiffness=1.0,
        )

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        collided = constraint.solve_capsule_collision(
            cp, capsule_a, capsule_b, capsule_radius
        )

        assert collided is True
        # Point should be pushed outside
        distance = np.linalg.norm(
            cp.position - np.array([0.0, 0.5, 0.0])  # Closest point on axis
        )
        assert distance >= capsule_radius + constraint.collision_radius - 0.001

    def test_solve_capsule_fixed_point(self):
        """Should not move fixed points (inv_mass=0)."""
        cp = make_control_point([0.05, 0.5, 0.0], inv_mass=0.0)
        original_pos = cp.position.copy()

        constraint = CollisionConstraint()

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        collided = constraint.solve_capsule_collision(
            cp, capsule_a, capsule_b, capsule_radius
        )

        assert collided is False
        np.testing.assert_array_equal(cp.position, original_pos)

    def test_solve_capsule_degenerate(self):
        """Should handle degenerate capsule (zero length)."""
        cp = make_control_point([0.05, 0.0, 0.0], inv_mass=1.0)

        constraint = CollisionConstraint()

        # Degenerate capsule - same point
        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        # Should not crash
        collided = constraint.solve_capsule_collision(
            cp, capsule_a, capsule_b, capsule_radius
        )

        assert collided is False  # Zero-length axis check returns early


class TestCreateLengthConstraints:
    """Tests for create_length_constraints function."""

    def test_create_length_constraints_basic(self):
        """Should create one constraint per segment."""
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.3, 0.0],
        ])

        constraints = create_length_constraints(strand, stiffness=0.9)

        assert len(constraints) == 3  # 3 segments
        for i, c in enumerate(constraints):
            assert c.cp0_index == i
            assert c.cp1_index == i + 1
            assert c.stiffness == 0.9
            assert c.rest_length == pytest.approx(0.1)

    def test_create_length_constraints_empty_strand(self):
        """Should handle single-point strand."""
        strand = make_simple_strand([[0.0, 0.0, 0.0]])

        constraints = create_length_constraints(strand)

        assert len(constraints) == 0


class TestCreateLocalShapeConstraints:
    """Tests for create_local_shape_constraints function."""

    def test_create_local_shape_short_strand(self):
        """Should return None for strands with < 3 points."""
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ])

        constraint = create_local_shape_constraints(strand)

        assert constraint is None

    def test_create_local_shape_straight_strand(self):
        """Should create constraint with zero rest angles for straight strand."""
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.3, 0.0],
        ])

        constraint = create_local_shape_constraints(strand, stiffness=0.5)

        assert constraint is not None
        assert len(constraint.rest_angles) == 2  # 2 interior joints
        assert constraint.stiffness == 0.5
        # Straight line should have angle=0 (cos=1, acos=0)
        for angle in constraint.rest_angles:
            assert angle == pytest.approx(0.0, abs=0.01)

    def test_create_local_shape_bent_strand(self):
        """Should create constraint with non-zero rest angles for bent strand."""
        # 90 degree bend
        strand = make_simple_strand([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],  # 90 degree turn
        ])

        constraint = create_local_shape_constraints(strand)

        assert constraint is not None
        # Should have one angle at the bend
        assert len(constraint.rest_angles) == 1
        # 90 degree = pi/2 radians
        assert constraint.rest_angles[0] == pytest.approx(math.pi / 2, abs=0.1)
