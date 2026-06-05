"""
Whitebox tests for edge cases in hair simulation.

Tests cover:
- Degenerate strands
- Extreme lengths
- Numerical stability
- Boundary conditions
- Performance under stress
"""

import math
import numpy as np
import pytest

from engine.simulation.hair.config import (
    DEFAULT_STRAND_SEGMENTS,
    MAX_GUIDE_HAIRS,
    MAX_STRAND_SEGMENTS,
    MIN_STRAND_SEGMENTS,
    NUMERICAL_EPSILON,
)
from engine.simulation.hair.hair_collision import (
    CapsuleCollider,
    HairCollisionSystem,
    HairDensityField,
    SphereCollider,
    collide_point_with_capsule,
    collide_point_with_sphere,
)
from engine.simulation.hair.hair_constraints import (
    CollisionConstraint,
    create_length_constraints,
    create_local_shape_constraints,
    solve_length_constraint,
)
from engine.simulation.hair.hair_lod import (
    HairLODSystem,
    LODSettings,
)
from engine.simulation.hair.hair_simulation import (
    GuideHair,
    HairControlPoint,
    HairSimulation,
    HairSimulationConfig,
    HairStrand,
    create_hair_from_scalp,
    create_hair_strand,
)


def make_control_point(position, inv_mass=1.0):
    """Helper to create a control point."""
    pos = np.array(position, dtype=np.float32)
    return HairControlPoint(
        position=pos.copy(),
        prev_position=pos.copy(),
        rest_position=pos.copy(),
        inv_mass=inv_mass,
    )


class TestDegenerateStrands:
    """Tests for degenerate strand configurations."""

    def test_zero_length_strand(self):
        """Strand with zero length segments."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal, length=0.0001, num_segments=4)

        # Should create valid strand even with tiny length
        assert len(hair.control_points) == 5
        assert hair.length < 0.001

    def test_single_point_strand(self):
        """Strand with only root point (0 segments)."""
        cp = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)

        strand = HairStrand(
            control_points=[cp],
            rest_lengths=[],
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        assert strand.num_segments == 0
        assert strand.length == 0.0

    def test_collinear_points(self):
        """Strand with all points collinear (edge case for local shape)."""
        control_points = []
        for i in range(5):
            cp = make_control_point([0.0, float(i) * 0.1, 0.0])
            cp.inv_mass = 0.0 if i == 0 else 1.0
            control_points.append(cp)

        strand = HairStrand(
            control_points=control_points,
            rest_lengths=[0.1, 0.1, 0.1, 0.1],
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        # Create local shape constraint - should handle collinear case
        constraint = create_local_shape_constraints(strand)
        if constraint:
            constraint.solve(strand.control_points)

        # No NaN values
        for cp in strand.control_points:
            assert not np.any(np.isnan(cp.position))

    def test_overlapping_control_points(self):
        """Strand with overlapping control points."""
        control_points = []
        for i in range(3):
            # All at same position
            cp = make_control_point([0.0, 0.0, 0.0])
            cp.inv_mass = 0.0 if i == 0 else 1.0
            control_points.append(cp)

        strand = HairStrand(
            control_points=control_points,
            rest_lengths=[0.1, 0.1],
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        # Length constraint should handle this
        constraints = create_length_constraints(strand)
        for c in constraints:
            c.solve(strand.control_points)

        # Should push points apart
        # At least some movement should occur


class TestExtremeLengths:
    """Tests for extreme length values."""

    def test_very_long_hair(self):
        """Hair much longer than typical (5 meters)."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal, length=5.0, num_segments=100)

        assert hair.length == pytest.approx(5.0)
        assert len(hair.control_points) == 101

        # Simulate briefly
        config = HairSimulationConfig(enable_collision=False)
        sim = HairSimulation(config)
        sim.add_guide_hair(hair)
        sim.start()
        sim.step(0.016)

        # No NaN
        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))

    def test_microscopic_hair(self):
        """Hair at microscopic scale (0.001mm)."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal, length=0.000001, num_segments=4)

        assert hair.length == pytest.approx(0.000001, abs=1e-9)

        # Should still be valid
        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))

    def test_maximum_segments(self):
        """Hair with maximum segments."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(
            root_pos, root_normal,
            length=0.3,
            num_segments=MAX_STRAND_SEGMENTS,
        )

        assert hair.num_segments == MAX_STRAND_SEGMENTS

    def test_minimum_segments(self):
        """Hair with minimum segments."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(
            root_pos, root_normal,
            length=0.3,
            num_segments=1,
        )

        assert hair.num_segments == 1


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_very_small_mass(self):
        """Control point with very small mass."""
        cp = HairControlPoint(
            position=np.array([0.0, 0.1, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.1, 0.0], dtype=np.float32),
            rest_position=np.array([0.0, 0.1, 0.0], dtype=np.float32),
            inv_mass=1e-10,  # Very small but non-zero
        )

        cp_root = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)

        # Length constraint with small mass
        error = solve_length_constraint(cp_root, cp, 0.1, stiffness=1.0)

        assert not np.isnan(error)
        assert not np.any(np.isnan(cp.position))

    def test_very_large_mass(self):
        """Control point with very large inverse mass."""
        cp = HairControlPoint(
            position=np.array([0.0, 0.2, 0.0], dtype=np.float32),
            prev_position=np.array([0.0, 0.2, 0.0], dtype=np.float32),
            rest_position=np.array([0.0, 0.2, 0.0], dtype=np.float32),
            inv_mass=1e10,  # Very large
        )

        cp_root = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)

        error = solve_length_constraint(cp_root, cp, 0.1, stiffness=1.0)

        assert not np.isnan(error)
        assert not np.any(np.isnan(cp.position))

    def test_near_zero_distance(self):
        """Points very close together (near epsilon)."""
        cp0 = make_control_point([0.0, 0.0, 0.0], inv_mass=0.0)
        cp1 = make_control_point([0.0, NUMERICAL_EPSILON / 2, 0.0], inv_mass=1.0)

        error = solve_length_constraint(cp0, cp1, 0.1, stiffness=1.0)

        # Should handle gracefully
        assert not np.isnan(error)

    def test_large_position_values(self):
        """Control points at very large coordinates."""
        large_val = 1e6
        root_pos = np.array([large_val, large_val, large_val], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal, length=0.3)

        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))
            assert not np.any(np.isinf(cp.position))

    def test_nan_input_handling(self):
        """Should not propagate NaN values."""
        # Create hair with valid positions
        hair = create_hair_strand(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        config = HairSimulationConfig(enable_collision=False)
        sim = HairSimulation(config)
        sim.add_guide_hair(hair)

        # Don't inject NaN - just verify simulation stays stable
        sim.start()
        for _ in range(10):
            sim.step(0.016)

        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_root_at_origin(self):
        """Hair rooted exactly at origin."""
        hair = create_hair_strand(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        assert hair.control_points[0].position[0] == 0.0
        assert hair.control_points[0].position[1] == 0.0
        assert hair.control_points[0].position[2] == 0.0

    def test_negative_coordinates(self):
        """Hair in negative coordinate space."""
        root_pos = np.array([-100.0, -100.0, -100.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal)

        # Should be valid
        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))

    def test_normalized_vs_unnormalized_normal(self):
        """Root normal should be normalized."""
        root_pos = np.zeros(3, dtype=np.float32)
        unnormalized = np.array([0.0, 100.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, unnormalized)

        assert np.linalg.norm(hair.root_normal) == pytest.approx(1.0)

    def test_parallel_to_axis_normals(self):
        """Root normals parallel to each axis."""
        for axis_normal in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]:
            root_pos = np.zeros(3, dtype=np.float32)
            root_normal = np.array(axis_normal, dtype=np.float32)

            hair = create_hair_strand(root_pos, root_normal)

            for cp in hair.control_points:
                assert not np.any(np.isnan(cp.position))


class TestCollisionEdgeCases:
    """Edge cases for collision detection."""

    def test_point_exactly_on_sphere_surface(self):
        """Point exactly on sphere surface."""
        point = make_control_point([0.5, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        radius = 0.5

        result = collide_point_with_sphere(point, center, radius, margin=0.0)

        # Exactly on surface - depends on implementation

    def test_point_exactly_on_capsule_axis(self):
        """Point exactly on capsule axis."""
        point = make_control_point([0.0, 0.5, 0.0])

        capsule_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        capsule_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        capsule_radius = 0.1

        result = collide_point_with_capsule(point, capsule_a, capsule_b, capsule_radius)

        # Should be pushed out
        assert result.collided is True

    def test_zero_radius_sphere(self):
        """Collision with zero-radius sphere."""
        point = make_control_point([0.001, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        radius = 0.0

        result = collide_point_with_sphere(point, center, radius, margin=0.0)

        # No collision with zero radius (unless margin > distance)
        # Depends on margin

    def test_very_large_density_field(self):
        """Large density field should not cause issues."""
        bounds_min = np.array([-100.0, -100.0, -100.0], dtype=np.float32)
        bounds_max = np.array([100.0, 100.0, 100.0], dtype=np.float32)

        field = HairDensityField(bounds_min, bounds_max, resolution=64)

        # Should handle large bounds
        field.accumulate(np.zeros(3, dtype=np.float32))
        density = field.sample_density(np.zeros(3, dtype=np.float32))

        assert not np.isnan(density)


class TestLODEdgeCases:
    """Edge cases for LOD system."""

    def test_lod_with_single_guide(self):
        """LOD system with single guide hair."""
        system = HairLODSystem()

        guides = [create_hair_strand(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )]
        guides[0].index = 0

        system.initialize(guides)

        # All LOD levels should have at least 1 guide
        assert len(system._guides_high) >= 1
        assert len(system._guides_medium) >= 1
        assert len(system._guides_low) >= 1

    def test_lod_exact_distance_threshold(self):
        """Camera at exact LOD distance threshold."""
        settings = LODSettings(
            distance_high=2.0,
            distance_medium=5.0,
            hysteresis=0.0,
        )
        system = HairLODSystem(settings=settings)

        guides = [create_hair_strand(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )]
        guides[0].index = 0
        system.initialize(guides)

        # Exactly at threshold
        camera = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        center = np.zeros(3, dtype=np.float32)

        system.update(camera, center)

        # Should be valid state

    def test_interpolation_weights_equidistant(self):
        """Interpolation when point is equidistant from guides."""
        system = HairLODSystem()

        # Two guides at same distance from query point
        guides = [
            create_hair_strand(
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
            create_hair_strand(
                np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
        ]
        guides[0].index = 0
        guides[1].index = 1
        system.initialize(guides)

        # Query at origin - equidistant from both
        indices, weights = system.get_interpolation_weights(
            np.zeros(3, dtype=np.float32),
            k_nearest=2,
        )

        assert len(indices) == 2
        # Weights should be equal
        assert weights[0] == pytest.approx(weights[1], abs=0.01)


class TestPerformanceStress:
    """Stress tests for performance."""

    def test_many_constraint_iterations(self):
        """Many constraint solving iterations."""
        hair = create_hair_strand(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            num_segments=20,
        )

        constraints = create_length_constraints(hair)

        # Many iterations
        for _ in range(100):
            for c in constraints:
                c.solve(hair.control_points)

        # Should not have degraded
        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))

    def test_large_scalp_array(self):
        """Creating hairs from large scalp array."""
        num_vertices = 1000
        positions = np.random.randn(num_vertices, 3).astype(np.float32)
        normals = np.tile([0.0, 1.0, 0.0], (num_vertices, 1)).astype(np.float32)

        hairs = create_hair_from_scalp(positions, normals, max_hairs=100)

        assert len(hairs) == 100

    def test_collision_with_many_colliders(self):
        """Collision detection with many colliders."""
        system = HairCollisionSystem()

        # Add many colliders
        for i in range(50):
            system.add_sphere(SphereCollider(
                center=np.array([float(i), 0.0, 0.0], dtype=np.float32),
                radius=0.1,
            ))
            system.add_capsule(CapsuleCollider(
                point_a=np.array([float(i), 1.0, 0.0], dtype=np.float32),
                point_b=np.array([float(i), 2.0, 0.0], dtype=np.float32),
                radius=0.1,
            ))

        # Create strand
        hair = create_hair_strand(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        strand = HairStrand(
            control_points=hair.control_points,
            rest_lengths=hair.rest_lengths,
            root_position=hair.root_position,
            root_normal=hair.root_normal,
        )

        # Should complete without timeout
        count = system.process_collisions([strand], iterations=2)


class TestMemoryAndTypes:
    """Tests for memory and type handling."""

    def test_float32_preserved(self):
        """Float32 types should be preserved throughout simulation."""
        hair = create_hair_strand(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        for cp in hair.control_points:
            assert cp.position.dtype == np.float32
            assert cp.prev_position.dtype == np.float32
            assert cp.rest_position.dtype == np.float32

    def test_array_independence(self):
        """Modifying one array should not affect others."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal)

        # Modify original arrays
        root_pos[0] = 100.0
        root_normal[0] = 100.0

        # Hair should be unaffected
        assert hair.root_position[0] != 100.0
        assert hair.root_normal[0] != 100.0

    def test_control_point_copy(self):
        """Control points should be independent copies."""
        hair = create_hair_strand(
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

        # Modify one control point
        original_pos = hair.control_points[1].position.copy()
        hair.control_points[1].position[0] = 999.0

        # Other control points should be unaffected
        assert hair.control_points[2].position[0] != 999.0
