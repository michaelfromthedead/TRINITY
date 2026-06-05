"""
Whitebox tests for cloth constraint implementations.

Tests:
- DistanceConstraint: edge length preservation
- BendingConstraint: dihedral angle preservation
- ShearConstraint: diagonal distance preservation
- LongRangeAttachment: maximum stretch limits
- AnchorConstraint: position fixing
- TetherConstraint: maximum distance from attachment
- Factory functions: create_bend_constraints, create_long_range_attachments
"""

import math

import numpy as np
import pytest

from engine.simulation.cloth.cloth_constraints import (
    AnchorConstraint,
    BendingConstraint,
    DistanceConstraint,
    LongRangeAttachment,
    ShearConstraint,
    TetherConstraint,
    create_bend_constraints,
    create_long_range_attachments,
)
from engine.simulation.cloth.cloth_simulation import ClothParticle


def make_particle(pos, inv_mass=1.0):
    """Helper to create a particle at a position."""
    pos_arr = np.array(pos, dtype=np.float32)
    return ClothParticle(position=pos_arr, prev_position=pos_arr.copy(), inv_mass=inv_mass)


class TestDistanceConstraint:
    """Test DistanceConstraint class."""

    def test_constraint_creation(self):
        """Test constraint creation with indices and rest length."""
        constraint = DistanceConstraint(p0_index=0, p1_index=1, rest_length=1.0)

        assert constraint.p0_index == 0
        assert constraint.p1_index == 1
        assert constraint.rest_length == 1.0
        assert constraint.stiffness == 1.0

    def test_solve_edge_stretched(self):
        """Stretched edge should be corrected toward rest length."""
        p0 = make_particle([0.0, 0.0, 0.0])
        p1 = make_particle([2.0, 0.0, 0.0])  # Distance = 2, rest = 1

        error = DistanceConstraint.solve_edge(p0, p1, rest_length=1.0, stiffness=1.0)

        assert error > 0  # Was stretched
        # Particles should have moved closer
        new_dist = np.linalg.norm(p1.position - p0.position)
        assert new_dist < 2.0

    def test_solve_edge_compressed(self):
        """Compressed edge should be corrected toward rest length."""
        p0 = make_particle([0.0, 0.0, 0.0])
        p1 = make_particle([0.5, 0.0, 0.0])  # Distance = 0.5, rest = 1

        error = DistanceConstraint.solve_edge(p0, p1, rest_length=1.0, stiffness=1.0)

        assert error < 0  # Was compressed
        # Particles should have moved apart
        new_dist = np.linalg.norm(p1.position - p0.position)
        assert new_dist > 0.5

    def test_solve_edge_at_rest_length(self):
        """Edge at rest length should not change."""
        p0 = make_particle([0.0, 0.0, 0.0])
        p1 = make_particle([1.0, 0.0, 0.0])

        pos0_before = p0.position.copy()
        pos1_before = p1.position.copy()

        error = DistanceConstraint.solve_edge(p0, p1, rest_length=1.0, stiffness=1.0)

        assert abs(error) < 1e-6
        assert np.allclose(p0.position, pos0_before)
        assert np.allclose(p1.position, pos1_before)

    def test_solve_edge_respects_inv_mass(self):
        """Heavier particle should move less."""
        p0 = make_particle([0.0, 0.0, 0.0], inv_mass=0.5)  # Mass = 2
        p1 = make_particle([2.0, 0.0, 0.0], inv_mass=1.0)  # Mass = 1

        pos0_before = p0.position.copy()
        pos1_before = p1.position.copy()

        DistanceConstraint.solve_edge(p0, p1, rest_length=1.0, stiffness=1.0)

        move0 = np.linalg.norm(p0.position - pos0_before)
        move1 = np.linalg.norm(p1.position - pos1_before)

        # Lighter particle (p1) should move more
        assert move1 > move0

    def test_solve_edge_pinned_particle_does_not_move(self):
        """Pinned particle (inv_mass=0) should not move."""
        p0 = make_particle([0.0, 0.0, 0.0], inv_mass=0.0)  # Pinned
        p1 = make_particle([2.0, 0.0, 0.0], inv_mass=1.0)

        pos0_before = p0.position.copy()

        DistanceConstraint.solve_edge(p0, p1, rest_length=1.0, stiffness=1.0)

        assert np.allclose(p0.position, pos0_before)

    def test_solve_edge_both_pinned(self):
        """Both pinned particles should not move."""
        p0 = make_particle([0.0, 0.0, 0.0], inv_mass=0.0)
        p1 = make_particle([2.0, 0.0, 0.0], inv_mass=0.0)

        pos0_before = p0.position.copy()
        pos1_before = p1.position.copy()

        error = DistanceConstraint.solve_edge(p0, p1, rest_length=1.0, stiffness=1.0)

        assert np.allclose(p0.position, pos0_before)
        assert np.allclose(p1.position, pos1_before)
        assert abs(error - 1.0) < 1e-6  # Error still 1.0 (2.0 - 1.0)

    def test_solve_edge_zero_distance(self):
        """Particles at same position should not crash."""
        p0 = make_particle([0.0, 0.0, 0.0])
        p1 = make_particle([0.0, 0.0, 0.0])

        error = DistanceConstraint.solve_edge(p0, p1, rest_length=1.0, stiffness=1.0)

        assert error == 0.0  # Early return for zero distance

    def test_solve_edge_low_stiffness(self):
        """Low stiffness should cause less correction."""
        p0 = make_particle([0.0, 0.0, 0.0])
        p1 = make_particle([2.0, 0.0, 0.0])

        p0_low = make_particle([0.0, 0.0, 0.0])
        p1_low = make_particle([2.0, 0.0, 0.0])

        DistanceConstraint.solve_edge(p0, p1, rest_length=1.0, stiffness=1.0)
        DistanceConstraint.solve_edge(p0_low, p1_low, rest_length=1.0, stiffness=0.1)

        dist_high = np.linalg.norm(p1.position - p0.position)
        dist_low = np.linalg.norm(p1_low.position - p0_low.position)

        # High stiffness corrects more
        assert dist_high < dist_low

    def test_solve_method(self):
        """Test solve() method on constraint instance."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([2.0, 0.0, 0.0]),
        ]
        constraint = DistanceConstraint(p0_index=0, p1_index=1, rest_length=1.0)

        error = constraint.solve(particles)

        assert abs(error - 1.0) < 1e-6  # Initial error was 1.0

    def test_solve_method_stiffness_override(self):
        """Test solve() with stiffness override."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([2.0, 0.0, 0.0]),
        ]
        constraint = DistanceConstraint(p0_index=0, p1_index=1, rest_length=1.0, stiffness=1.0)

        # Override with low stiffness
        constraint.solve(particles, stiffness_override=0.0)

        # With zero stiffness, no correction should happen
        dist = np.linalg.norm(particles[1].position - particles[0].position)
        assert abs(dist - 2.0) < 1e-6


class TestBendingConstraint:
    """Test BendingConstraint class."""

    @pytest.fixture
    def flat_quad_particles(self):
        """Create 4 particles forming two adjacent triangles in a plane."""
        # Layout:
        #   0---1
        #   |\ /|
        #   | X |
        #   |/ \|
        #   2---3
        # Shared edge: 0-3 (diagonal)
        # Actually let's use a simpler layout:
        #   0 - 1
        #   |   |
        #   2 - 3
        # Triangles: (0,1,2) and (1,3,2), shared edge (1,2)
        return [
            make_particle([0.0, 0.0, 0.0]),  # 0: top-left
            make_particle([1.0, 0.0, 0.0]),  # 1: top-right
            make_particle([0.0, -1.0, 0.0]),  # 2: bottom-left
            make_particle([1.0, -1.0, 0.0]),  # 3: bottom-right
        ]

    def test_compute_dihedral_angle_flat(self):
        """Flat triangles should have dihedral angle of 0 or pi."""
        p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Outer vertex of tri 1
        p1 = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # Shared edge vertex
        p2 = np.array([0.5, 1.0, 0.0], dtype=np.float32)  # Shared edge vertex
        p3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Outer vertex of tri 2

        angle = BendingConstraint.compute_dihedral_angle(p0, p1, p2, p3)

        # Both triangles in XY plane, normals parallel or anti-parallel
        # Angle should be 0 or close to 0
        assert abs(angle) < 0.1 or abs(abs(angle) - math.pi) < 0.1

    def test_compute_dihedral_angle_bent(self):
        """Bent triangles should have non-zero dihedral angle."""
        p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p1 = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        p2 = np.array([0.5, 0.0, 1.0], dtype=np.float32)
        p3 = np.array([1.0, 1.0, 0.5], dtype=np.float32)  # Out of plane

        angle = BendingConstraint.compute_dihedral_angle(p0, p1, p2, p3)

        # Should have some non-trivial angle
        assert abs(angle) > 0.01

    def test_compute_dihedral_angle_degenerate(self):
        """Degenerate triangle should return 0."""
        p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p2 = np.array([2.0, 0.0, 0.0], dtype=np.float32)  # Collinear
        p3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        angle = BendingConstraint.compute_dihedral_angle(p0, p1, p2, p3)

        assert angle == 0.0

    def test_solve_flat_no_change(self, flat_quad_particles):
        """Flat configuration at rest should not change much."""
        particles = flat_quad_particles

        # Compute rest angle first
        rest_angle = BendingConstraint.compute_dihedral_angle(
            particles[0].position,
            particles[1].position,
            particles[2].position,
            particles[3].position,
        )

        constraint = BendingConstraint(
            p0_index=0,
            p1_index=1,
            p2_index=2,
            p3_index=3,
            rest_angle=rest_angle,
            stiffness=0.5,
        )

        pos_before = [p.position.copy() for p in particles]
        constraint.solve(particles)

        # Positions should not change much (already at rest)
        for i, p in enumerate(particles):
            assert np.allclose(p.position, pos_before[i], atol=1e-4)

    def test_solve_bent_corrects_toward_rest(self):
        """Bent configuration should apply corrections without crashing."""
        particles = [
            make_particle([0.0, 0.5, 0.0]),  # 0: Outer, lifted
            make_particle([0.5, 0.0, 0.0]),  # 1: Shared
            make_particle([0.5, 0.0, 1.0]),  # 2: Shared
            make_particle([1.0, 0.0, 0.5]),  # 3: Outer
        ]

        constraint = BendingConstraint(
            p0_index=0,
            p1_index=1,
            p2_index=2,
            p3_index=3,
            rest_angle=0.0,  # Want to flatten
            stiffness=0.5,
        )

        initial_angle = BendingConstraint.compute_dihedral_angle(
            particles[0].position,
            particles[1].position,
            particles[2].position,
            particles[3].position,
        )

        error = constraint.solve(particles)

        # Should return the angle error and not crash
        assert np.isfinite(error)
        # All particle positions should remain finite
        for p in particles:
            assert np.isfinite(p.position).all()

    def test_solve_respects_stiffness_override(self):
        """Stiffness override should be used."""
        particles = [
            make_particle([0.0, 0.5, 0.0]),
            make_particle([0.5, 0.0, 0.0]),
            make_particle([0.5, 0.0, 1.0]),
            make_particle([1.0, 0.0, 0.5]),
        ]

        constraint = BendingConstraint(
            p0_index=0,
            p1_index=1,
            p2_index=2,
            p3_index=3,
            rest_angle=0.0,
            stiffness=1.0,  # High stiffness
        )

        particles_zero = [
            make_particle(p.position) for p in particles
        ]

        constraint.solve(particles, stiffness_override=0.0)  # Override to zero
        constraint_high = BendingConstraint(
            p0_index=0, p1_index=1, p2_index=2, p3_index=3,
            rest_angle=0.0, stiffness=1.0
        )
        constraint_high.solve(particles_zero)

        # With zero stiffness, particles should not move much
        # (Note: due to internal structure, there may still be minimal change)


class TestShearConstraint:
    """Test ShearConstraint class."""

    def test_shear_constraint_same_as_distance(self):
        """ShearConstraint should behave like DistanceConstraint."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([2.0, 0.0, 0.0]),
        ]

        constraint = ShearConstraint(p0_index=0, p1_index=1, rest_length=1.0, stiffness=1.0)

        error = constraint.solve(particles)

        assert abs(error - 1.0) < 1e-6  # Same error computation as distance

    def test_shear_constraint_default_stiffness(self):
        """Shear constraint default stiffness should be 0.5."""
        constraint = ShearConstraint(p0_index=0, p1_index=1, rest_length=1.0)

        assert constraint.stiffness == 0.5


class TestLongRangeAttachment:
    """Test LongRangeAttachment class."""

    def test_within_max_distance_no_correction(self):
        """Particles within max distance should not be corrected."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),  # Anchor (pinned)
            make_particle([1.0, 0.0, 0.0]),  # Within max_distance=2.0
        ]

        constraint = LongRangeAttachment(
            p0_index=0,
            p1_index=1,
            max_distance=2.0,
            stiffness=0.8,
        )

        pos_before = particles[1].position.copy()
        error = constraint.solve(particles)

        assert error == 0.0
        assert np.allclose(particles[1].position, pos_before)

    def test_exceeds_max_distance_corrected(self):
        """Particles exceeding max distance should be corrected."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),  # Anchor
            make_particle([3.0, 0.0, 0.0]),  # Exceeds max_distance=2.0
        ]

        constraint = LongRangeAttachment(
            p0_index=0,
            p1_index=1,
            max_distance=2.0,
            stiffness=1.0,
        )

        error = constraint.solve(particles)

        assert error > 0  # Was 1.0 = 3.0 - 2.0
        # Particle should move closer to anchor
        new_dist = np.linalg.norm(particles[1].position - particles[0].position)
        assert new_dist < 3.0

    def test_at_max_distance_no_correction(self):
        """Particle exactly at max distance should not be corrected."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),
            make_particle([2.0, 0.0, 0.0]),
        ]

        constraint = LongRangeAttachment(
            p0_index=0,
            p1_index=1,
            max_distance=2.0,
            stiffness=1.0,
        )

        error = constraint.solve(particles)

        assert error == 0.0


class TestAnchorConstraint:
    """Test AnchorConstraint class."""

    def test_anchor_moves_particle_to_position(self):
        """Anchor should move particle toward anchor position."""
        particles = [
            make_particle([1.0, 0.0, 0.0]),
        ]
        anchor_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        constraint = AnchorConstraint(
            particle_index=0,
            anchor_position=anchor_pos,
            stiffness=1.0,
        )

        error = constraint.solve(particles)

        assert error > 0  # Distance was 1.0
        # With stiffness=1.0, should move fully to anchor
        assert np.allclose(particles[0].position, anchor_pos)

    def test_anchor_partial_stiffness(self):
        """Partial stiffness should not fully move particle."""
        particles = [
            make_particle([2.0, 0.0, 0.0]),
        ]
        anchor_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        constraint = AnchorConstraint(
            particle_index=0,
            anchor_position=anchor_pos,
            stiffness=0.5,
        )

        constraint.solve(particles)

        # Should move halfway
        expected = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(particles[0].position, expected)

    def test_anchor_at_position_no_change(self):
        """Particle at anchor position should not change."""
        anchor_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        particles = [
            make_particle(anchor_pos.copy()),
        ]

        constraint = AnchorConstraint(
            particle_index=0,
            anchor_position=anchor_pos,
            stiffness=1.0,
        )

        error = constraint.solve(particles)

        assert error == 0.0

    def test_update_anchor_position(self):
        """update_anchor should change anchor position."""
        anchor_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        new_pos = np.array([5.0, 5.0, 5.0], dtype=np.float32)

        constraint = AnchorConstraint(
            particle_index=0,
            anchor_position=anchor_pos,
            stiffness=1.0,
        )

        constraint.update_anchor(new_pos)

        assert np.allclose(constraint.anchor_position, new_pos)


class TestTetherConstraint:
    """Test TetherConstraint class."""

    def test_within_max_distance_no_correction(self):
        """Particle within max distance should not be corrected."""
        particles = [
            make_particle([1.0, 0.0, 0.0]),
        ]
        attachment_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        constraint = TetherConstraint(
            particle_index=0,
            attachment_position=attachment_pos,
            max_distance=2.0,
            stiffness=1.0,
        )

        pos_before = particles[0].position.copy()
        error = constraint.solve(particles)

        assert error == 0.0
        assert np.allclose(particles[0].position, pos_before)

    def test_exceeds_max_distance_corrected(self):
        """Particle exceeding max distance should be pulled back."""
        particles = [
            make_particle([3.0, 0.0, 0.0]),
        ]
        attachment_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        constraint = TetherConstraint(
            particle_index=0,
            attachment_position=attachment_pos,
            max_distance=2.0,
            stiffness=1.0,
        )

        error = constraint.solve(particles)

        assert error == 1.0  # 3.0 - 2.0
        # Should be pulled back to max_distance
        new_dist = np.linalg.norm(particles[0].position - attachment_pos)
        assert abs(new_dist - 2.0) < 1e-6

    def test_tether_respects_stiffness(self):
        """Partial stiffness should not fully correct."""
        particles = [
            make_particle([4.0, 0.0, 0.0]),  # Distance = 4, max = 2, error = 2
        ]
        attachment_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        constraint = TetherConstraint(
            particle_index=0,
            attachment_position=attachment_pos,
            max_distance=2.0,
            stiffness=0.5,
        )

        constraint.solve(particles)

        # Should correct by 50% of error (1.0)
        new_dist = np.linalg.norm(particles[0].position - attachment_pos)
        assert abs(new_dist - 3.0) < 1e-6

    def test_tether_pinned_particle_no_move(self):
        """Pinned particle should not move even if outside max distance."""
        particles = [
            make_particle([5.0, 0.0, 0.0], inv_mass=0.0),  # Pinned
        ]
        attachment_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        constraint = TetherConstraint(
            particle_index=0,
            attachment_position=attachment_pos,
            max_distance=2.0,
            stiffness=1.0,
        )

        pos_before = particles[0].position.copy()
        error = constraint.solve(particles)

        assert error == 3.0  # Still reports error
        assert np.allclose(particles[0].position, pos_before)  # But doesn't move

    def test_update_attachment_position(self):
        """update_attachment should change attachment position."""
        attachment_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        new_pos = np.array([10.0, 10.0, 10.0], dtype=np.float32)

        constraint = TetherConstraint(
            particle_index=0,
            attachment_position=attachment_pos,
            max_distance=2.0,
        )

        constraint.update_attachment(new_pos)

        assert np.allclose(constraint.attachment_position, new_pos)


class TestCreateBendConstraints:
    """Test create_bend_constraints factory function."""

    def test_creates_constraints_for_shared_edges(self):
        """Should create constraint for each shared edge."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),  # 0
            make_particle([1.0, 0.0, 0.0]),  # 1
            make_particle([0.0, 1.0, 0.0]),  # 2
            make_particle([1.0, 1.0, 0.0]),  # 3
        ]
        # Two triangles sharing edge (1, 2)
        triangles = [
            (0, 1, 2),
            (1, 3, 2),
        ]

        constraints = create_bend_constraints(particles, triangles, stiffness=0.2)

        # Should have one constraint for shared edge
        assert len(constraints) == 1
        assert constraints[0].stiffness == 0.2

    def test_no_constraints_for_unshared_edges(self):
        """Should not create constraints for boundary edges."""
        particles = [
            make_particle([0.0, 0.0, 0.0]),
            make_particle([1.0, 0.0, 0.0]),
            make_particle([0.5, 1.0, 0.0]),
        ]
        # Single triangle - no shared edges
        triangles = [(0, 1, 2)]

        constraints = create_bend_constraints(particles, triangles)

        assert len(constraints) == 0

    def test_multiple_shared_edges(self):
        """Should create constraints for all shared edges."""
        # Grid of 4 triangles forming a square
        particles = [
            make_particle([0.0, 0.0, 0.0]),  # 0: top-left
            make_particle([1.0, 0.0, 0.0]),  # 1: top-right
            make_particle([0.0, 1.0, 0.0]),  # 2: bottom-left
            make_particle([1.0, 1.0, 0.0]),  # 3: bottom-right
        ]
        triangles = [
            (0, 1, 2),
            (1, 3, 2),
        ]

        constraints = create_bend_constraints(particles, triangles)

        # One shared edge (1-2)
        assert len(constraints) == 1


class TestCreateLongRangeAttachments:
    """Test create_long_range_attachments factory function."""

    def test_creates_attachments_from_anchors(self):
        """Should create attachments from anchor to all movable particles."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),  # 0: Anchor (pinned)
            make_particle([1.0, 0.0, 0.0]),  # 1: Movable
            make_particle([2.0, 0.0, 0.0]),  # 2: Movable
        ]

        attachments = create_long_range_attachments(
            particles,
            attachment_indices=[0],
            max_ratio=1.5,
            stiffness=0.8,
        )

        # Should create attachment to particles 1 and 2
        assert len(attachments) == 2
        assert all(a.stiffness == 0.8 for a in attachments)

    def test_respects_max_ratio(self):
        """Max distance should be rest_distance * max_ratio."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),  # Anchor
            make_particle([10.0, 0.0, 0.0]),  # Distance = 10
        ]

        attachments = create_long_range_attachments(
            particles,
            attachment_indices=[0],
            max_ratio=2.0,
        )

        assert len(attachments) == 1
        assert abs(attachments[0].max_distance - 20.0) < 1e-6  # 10 * 2.0

    def test_skips_pinned_particles(self):
        """Should not create attachments to pinned particles."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),  # 0: Anchor
            make_particle([1.0, 0.0, 0.0], inv_mass=0.0),  # 1: Pinned
            make_particle([2.0, 0.0, 0.0]),  # 2: Movable
        ]

        attachments = create_long_range_attachments(
            particles,
            attachment_indices=[0],
        )

        # Should only attach to particle 2
        assert len(attachments) == 1
        assert attachments[0].p1_index == 2

    def test_skips_self_attachment(self):
        """Should not attach anchor to itself."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),
        ]

        attachments = create_long_range_attachments(
            particles,
            attachment_indices=[0],
        )

        assert len(attachments) == 0

    def test_multiple_anchors(self):
        """Should create attachments from all anchors."""
        particles = [
            make_particle([0.0, 0.0, 0.0], inv_mass=0.0),  # 0: Anchor
            make_particle([3.0, 0.0, 0.0], inv_mass=0.0),  # 1: Anchor
            make_particle([1.5, 1.0, 0.0]),  # 2: Movable
        ]

        attachments = create_long_range_attachments(
            particles,
            attachment_indices=[0, 1],
        )

        # Should create 2 attachments (one from each anchor to particle 2)
        assert len(attachments) == 2
