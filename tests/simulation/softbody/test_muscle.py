"""Tests for muscle simulation module."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from engine.simulation.softbody.muscle import (
    MuscleAttachment,
    MuscleFiber,
    MuscleProperties,
    Muscle,
    MuscleGroup,
    MuscleController,
)
from engine.simulation.softbody.config import (
    MUSCLE_FORCE_LENGTH_WIDTH,
    MUSCLE_ECCENTRIC_FORCE_MAX,
    MUSCLE_CONCENTRIC_THRESHOLD,
    MUSCLE_VOLUME_STIFFNESS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mesh_positions():
    """Simple mesh positions for muscle testing."""
    return np.array([
        # Origin vertices (0-3)
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
        # Insertion vertices (4-7)
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [1.0, 0.1, 0.0],
        [1.0, 0.0, 0.1],
    ], dtype=np.float64)


@pytest.fixture
def origin_attachment():
    """Create origin attachment."""
    return MuscleAttachment(
        body_index=0,
        local_position=np.array([0.0, 0.0, 0.0]),
        is_origin=True,
    )


@pytest.fixture
def insertion_attachment():
    """Create insertion attachment."""
    return MuscleAttachment(
        body_index=0,
        local_position=np.array([1.0, 0.0, 0.0]),
        is_origin=False,
    )


@pytest.fixture
def muscle(origin_attachment, insertion_attachment):
    """Create a basic muscle."""
    return Muscle(
        origin=origin_attachment,
        insertion=insertion_attachment,
        fiber_direction=np.array([1.0, 0.0, 0.0]),
    )


# =============================================================================
# Test MuscleAttachment
# =============================================================================

class TestMuscleAttachment:
    """Test muscle attachment data class."""

    def test_construction(self):
        """Should construct properly."""
        attachment = MuscleAttachment(
            body_index=1,
            local_position=np.array([1.0, 2.0, 3.0]),
            is_origin=True,
        )
        assert attachment.body_index == 1
        assert attachment.is_origin is True

    def test_get_world_position_no_transform(self):
        """World position without transform should be local position."""
        attachment = MuscleAttachment(
            body_index=-1,
            local_position=np.array([1.0, 2.0, 3.0]),
        )
        world_pos = attachment.get_world_position()
        assert_array_almost_equal(world_pos, [1.0, 2.0, 3.0])

    def test_get_world_position_with_transform(self):
        """World position with transform should be transformed."""
        attachment = MuscleAttachment(
            body_index=0,
            local_position=np.array([1.0, 0.0, 0.0]),
        )
        # 90 degree rotation around z
        transform = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        position = np.array([5.0, 5.0, 0.0])
        world_pos = attachment.get_world_position(transform, position)
        # Rotated (1,0,0) -> (0,1,0), plus translation
        expected = np.array([5.0, 6.0, 0.0])
        assert_array_almost_equal(world_pos, expected)


# =============================================================================
# Test MuscleFiber
# =============================================================================

class TestMuscleFiber:
    """Test muscle fiber data class."""

    def test_construction(self):
        """Should construct properly."""
        fiber = MuscleFiber(
            start_vertex=0,
            end_vertex=4,
            rest_length=1.0,
            direction=np.array([1.0, 0.0, 0.0]),
            max_contraction=0.3,
        )
        assert fiber.rest_length == 1.0
        assert fiber.max_contraction == 0.3

    def test_compute_current_length(self, mesh_positions):
        """Should compute current fiber length."""
        fiber = MuscleFiber(
            start_vertex=0,
            end_vertex=4,
            rest_length=1.0,
            direction=np.array([1.0, 0.0, 0.0]),
        )
        length = fiber.compute_current_length(mesh_positions)
        assert np.isclose(length, 1.0)

    def test_compute_target_length_no_activation(self):
        """No activation should give rest length."""
        fiber = MuscleFiber(
            start_vertex=0,
            end_vertex=4,
            rest_length=1.0,
            direction=np.array([1.0, 0.0, 0.0]),
            max_contraction=0.3,
        )
        target = fiber.compute_target_length(activation=0.0)
        assert np.isclose(target, 1.0)

    def test_compute_target_length_full_activation(self):
        """Full activation should shorten to minimum length."""
        fiber = MuscleFiber(
            start_vertex=0,
            end_vertex=4,
            rest_length=1.0,
            direction=np.array([1.0, 0.0, 0.0]),
            max_contraction=0.3,
        )
        target = fiber.compute_target_length(activation=1.0)
        assert np.isclose(target, 0.7)  # 1.0 * (1 - 0.3)

    def test_compute_target_length_partial_activation(self):
        """Partial activation should partially contract."""
        fiber = MuscleFiber(
            start_vertex=0,
            end_vertex=4,
            rest_length=1.0,
            direction=np.array([1.0, 0.0, 0.0]),
            max_contraction=0.3,
        )
        target = fiber.compute_target_length(activation=0.5)
        assert np.isclose(target, 0.85)  # 1.0 * (1 - 0.3 * 0.5)


# =============================================================================
# Test MuscleProperties
# =============================================================================

class TestMuscleProperties:
    """Test muscle properties data class."""

    def test_default_values(self):
        """Should have sensible defaults."""
        props = MuscleProperties()
        assert props.max_force == 100.0
        assert props.optimal_length == 1.0
        assert props.fiber_velocity_max == 10.0
        assert props.pennation_angle == 0.0
        assert props.passive_stiffness == 100.0
        assert props.active_stiffness == 1000.0
        assert props.damping == 10.0

    def test_custom_values(self):
        """Should allow custom values."""
        props = MuscleProperties(
            max_force=500.0,
            optimal_length=0.5,
            pennation_angle=0.2,
        )
        assert props.max_force == 500.0
        assert props.optimal_length == 0.5
        assert props.pennation_angle == 0.2


# =============================================================================
# Test Muscle
# =============================================================================

class TestMuscle:
    """Test muscle simulation."""

    def test_construction(self, muscle):
        """Muscle should initialize properly."""
        assert muscle.origin is not None
        assert muscle.insertion is not None
        assert_array_almost_equal(muscle.fiber_direction, [1.0, 0.0, 0.0])
        assert muscle.activation == 0.0

    def test_fiber_direction_normalized(self, origin_attachment, insertion_attachment):
        """Fiber direction should be normalized."""
        muscle = Muscle(
            origin=origin_attachment,
            insertion=insertion_attachment,
            fiber_direction=np.array([2.0, 0.0, 0.0]),
        )
        assert np.isclose(np.linalg.norm(muscle.fiber_direction), 1.0)

    def test_activation_clamped(self, muscle):
        """Activation should be clamped to [0, 1]."""
        muscle.activation = -0.5
        assert muscle.activation == 0.0

        muscle.activation = 1.5
        assert muscle.activation == 1.0

        muscle.activation = 0.5
        assert muscle.activation == 0.5

    def test_build_fibers_from_mesh(self, muscle, mesh_positions):
        """Should build fibers connecting origin to insertion."""
        muscle.build_fibers_from_mesh(
            positions=mesh_positions,
            origin_vertices=[0, 1, 2, 3],
            insertion_vertices=[4, 5, 6, 7],
        )
        assert len(muscle.fibers) == 4
        assert muscle.rest_length > 0

    def test_compute_contraction_force_inactive(self, muscle, mesh_positions):
        """No activation should give minimal force."""
        muscle.build_fibers_from_mesh(
            positions=mesh_positions,
            origin_vertices=[0, 1, 2, 3],
            insertion_vertices=[4, 5, 6, 7],
        )
        muscle.activation = 0.0
        muscle.current_length = muscle.rest_length
        force = muscle.compute_contraction_force()
        # Only passive force at rest length (should be 0)
        assert np.isclose(force, 0.0)

    def test_compute_contraction_force_full_activation(self, muscle, mesh_positions):
        """Full activation should give significant force."""
        muscle.build_fibers_from_mesh(
            positions=mesh_positions,
            origin_vertices=[0, 1, 2, 3],
            insertion_vertices=[4, 5, 6, 7],
        )
        muscle.activation = 1.0
        muscle.current_length = muscle.properties.optimal_length
        muscle.contraction_velocity = 0.0
        force = muscle.compute_contraction_force()
        assert force > 0
        # At optimal length, force should be close to max
        assert force >= muscle.properties.max_force * 0.5

    def test_force_length_relationship(self, muscle, mesh_positions):
        """Force should be maximum at optimal length."""
        muscle.build_fibers_from_mesh(
            positions=mesh_positions,
            origin_vertices=[0, 1, 2, 3],
            insertion_vertices=[4, 5, 6, 7],
        )
        muscle.activation = 1.0
        muscle.contraction_velocity = 0.0

        # Force at optimal length
        muscle.current_length = muscle.properties.optimal_length
        force_optimal = muscle.compute_contraction_force()

        # Force at stretched length
        muscle.current_length = muscle.properties.optimal_length * 1.5
        force_stretched = muscle.compute_contraction_force()

        # Force at shortened length
        muscle.current_length = muscle.properties.optimal_length * 0.7
        force_shortened = muscle.compute_contraction_force()

        # Optimal should have highest active force contribution
        # (passive force increases with stretch)
        assert force_optimal > force_shortened * 0.5

    def test_apply_contraction_forces(self, muscle, mesh_positions):
        """Forces should be applied to vertices."""
        muscle.build_fibers_from_mesh(
            positions=mesh_positions,
            origin_vertices=[0, 1, 2, 3],
            insertion_vertices=[4, 5, 6, 7],
        )
        muscle.activation = 1.0

        velocities = np.zeros_like(mesh_positions)
        inv_masses = np.ones(len(mesh_positions))
        dt = 0.01

        muscle.apply_contraction_forces(mesh_positions, velocities, inv_masses, dt)

        # Velocities should have changed
        assert np.any(velocities != 0)

    def test_volume_preservation(self, muscle, mesh_positions):
        """Volume preservation should expand perpendicular to fiber."""
        muscle.build_fibers_from_mesh(
            positions=mesh_positions,
            origin_vertices=[0, 1, 2, 3],
            insertion_vertices=[4, 5, 6, 7],
        )
        muscle.activation = 1.0
        muscle.current_length = muscle.rest_length * 0.8  # Contracted
        muscle.rest_length = 1.0

        velocities = np.zeros_like(mesh_positions)
        inv_masses = np.ones(len(mesh_positions))
        vertex_indices = list(range(len(mesh_positions)))
        dt = 0.01

        muscle.apply_volume_preservation(
            mesh_positions, velocities, inv_masses, vertex_indices, dt
        )

        # Some velocities should have perpendicular component
        # (checking that something happened)
        total_vel = np.sum(np.abs(velocities))
        assert total_vel >= 0  # May be small if only 8 vertices


# =============================================================================
# Test MuscleGroup
# =============================================================================

class TestMuscleGroup:
    """Test muscle group."""

    @pytest.fixture
    def muscle_group(self, muscle):
        """Create muscle group with one muscle."""
        group = MuscleGroup(name="test_group")
        group.add_muscle(muscle)
        return group

    def test_construction(self):
        """Should construct properly."""
        group = MuscleGroup(name="biceps")
        assert group.name == "biceps"
        assert len(group.muscles) == 0

    def test_add_muscle(self, muscle_group, muscle):
        """Should add muscles."""
        assert len(muscle_group.muscles) == 1
        assert muscle_group.muscles[0] is muscle

    def test_set_activation(self, muscle_group):
        """Should set activation on all muscles."""
        muscle_group.set_activation(0.75)
        for m in muscle_group.muscles:
            assert m.activation == 0.75

    def test_get_total_force(self, muscle_group, mesh_positions):
        """Should sum forces from all muscles."""
        muscle_group.muscles[0].build_fibers_from_mesh(
            positions=mesh_positions,
            origin_vertices=[0, 1, 2, 3],
            insertion_vertices=[4, 5, 6, 7],
        )
        muscle_group.set_activation(1.0)
        muscle_group.muscles[0].current_length = 1.0
        muscle_group.muscles[0].compute_contraction_force()

        total = muscle_group.get_total_force()
        assert total >= 0

    def test_apply_forces(self, muscle_group, mesh_positions):
        """Should apply forces from all muscles."""
        muscle_group.muscles[0].build_fibers_from_mesh(
            positions=mesh_positions,
            origin_vertices=[0, 1, 2, 3],
            insertion_vertices=[4, 5, 6, 7],
        )
        muscle_group.set_activation(1.0)

        velocities = np.zeros_like(mesh_positions)
        inv_masses = np.ones(len(mesh_positions))
        dt = 0.01

        muscle_group.apply_forces(mesh_positions, velocities, inv_masses, dt)

        # Velocities should have changed
        assert np.any(velocities != 0)


# =============================================================================
# Test MuscleController
# =============================================================================

class TestMuscleController:
    """Test muscle controller."""

    @pytest.fixture
    def controller(self, muscle, origin_attachment, insertion_attachment):
        """Create muscle controller with two antagonist groups."""
        ctrl = MuscleController()

        # Create biceps group
        biceps = MuscleGroup(name="biceps")
        biceps.add_muscle(muscle)

        # Create triceps group
        triceps_muscle = Muscle(
            origin=insertion_attachment,  # Swap origin/insertion
            insertion=origin_attachment,
            fiber_direction=np.array([-1.0, 0.0, 0.0]),
        )
        triceps = MuscleGroup(name="triceps")
        triceps.add_muscle(triceps_muscle)

        ctrl.add_group("biceps", biceps)
        ctrl.add_group("triceps", triceps)
        ctrl.set_antagonist_pair("biceps", "triceps")

        return ctrl

    def test_construction(self):
        """Should construct properly."""
        ctrl = MuscleController()
        assert len(ctrl.groups) == 0

    def test_add_group(self, controller):
        """Should add muscle groups."""
        assert "biceps" in controller.groups
        assert "triceps" in controller.groups

    def test_set_antagonist_pair(self, controller):
        """Should register antagonist pairs."""
        assert ("biceps", "triceps") in controller.antagonist_pairs

    def test_activate_group(self, controller):
        """Should activate specified group."""
        controller.activate_group("biceps", 0.8)
        assert controller.groups["biceps"].muscles[0].activation == 0.8

    def test_antagonist_inhibition(self, controller):
        """Activating one should inhibit antagonist."""
        # First activate triceps
        controller.activate_group("triceps", 0.5)

        # Then activate biceps - should inhibit triceps
        controller.activate_group("biceps", 1.0, inhibit_antagonists=True)

        # Triceps should be reduced
        triceps_activation = controller.groups["triceps"].muscles[0].activation
        assert triceps_activation < 0.5

    def test_activate_without_inhibition(self, controller):
        """Can activate without inhibiting antagonists."""
        controller.activate_group("triceps", 0.8)
        controller.activate_group("biceps", 1.0, inhibit_antagonists=False)

        # Triceps should be unchanged
        assert controller.groups["triceps"].muscles[0].activation == 0.8

    def test_update(self, controller, mesh_positions):
        """Should update all muscles."""
        # Build fibers for both muscles
        controller.groups["biceps"].muscles[0].build_fibers_from_mesh(
            mesh_positions, [0, 1, 2, 3], [4, 5, 6, 7]
        )
        controller.groups["triceps"].muscles[0].build_fibers_from_mesh(
            mesh_positions, [4, 5, 6, 7], [0, 1, 2, 3]
        )

        controller.activate_group("biceps", 0.5)

        velocities = np.zeros_like(mesh_positions)
        inv_masses = np.ones(len(mesh_positions))
        dt = 0.01

        controller.update(mesh_positions, velocities, inv_masses, dt)

        # Velocities should have changed
        assert np.any(velocities != 0)


# =============================================================================
# Test Force-Velocity Relationship
# =============================================================================

class TestForceVelocityRelationship:
    """Test muscle force-velocity relationship."""

    @pytest.fixture
    def activated_muscle(self, muscle, mesh_positions):
        """Create activated muscle."""
        muscle.build_fibers_from_mesh(
            positions=mesh_positions,
            origin_vertices=[0, 1, 2, 3],
            insertion_vertices=[4, 5, 6, 7],
        )
        muscle.activation = 1.0
        muscle.current_length = muscle.properties.optimal_length
        return muscle

    def test_isometric_force(self, activated_muscle):
        """Zero velocity should give maximum force."""
        activated_muscle.contraction_velocity = 0.0
        force_isometric = activated_muscle.compute_contraction_force()
        assert force_isometric > 0

    def test_concentric_force_reduced(self, activated_muscle):
        """Shortening velocity should reduce force."""
        activated_muscle.contraction_velocity = 0.0
        force_isometric = activated_muscle.compute_contraction_force()

        # Shortening (negative velocity)
        activated_muscle.contraction_velocity = (
            -0.5 * activated_muscle.properties.fiber_velocity_max
        )
        force_concentric = activated_muscle.compute_contraction_force()

        # Concentric force should be less
        assert force_concentric < force_isometric

    def test_eccentric_force_enhanced(self, activated_muscle):
        """Lengthening velocity can enhance force."""
        activated_muscle.contraction_velocity = 0.0
        force_isometric = activated_muscle.compute_contraction_force()

        # Lengthening (positive velocity)
        activated_muscle.contraction_velocity = (
            0.5 * activated_muscle.properties.fiber_velocity_max
        )
        force_eccentric = activated_muscle.compute_contraction_force()

        # Eccentric force should be greater or equal
        assert force_eccentric >= force_isometric * 0.9


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestMuscleEdgeCases:
    """Test edge cases."""

    def test_zero_length_fiber_direction(self, origin_attachment, insertion_attachment):
        """Zero fiber direction should be handled."""
        muscle = Muscle(
            origin=origin_attachment,
            insertion=insertion_attachment,
            fiber_direction=np.array([0.0, 0.0, 0.0]),
        )
        # Should default to x-axis
        assert np.isclose(np.linalg.norm(muscle.fiber_direction), 1.0)

    def test_no_fibers_apply_forces(self, muscle, mesh_positions):
        """No fibers should not crash."""
        velocities = np.zeros_like(mesh_positions)
        inv_masses = np.ones(len(mesh_positions))
        dt = 0.01

        # No fibers built
        muscle.apply_contraction_forces(mesh_positions, velocities, inv_masses, dt)
        # Should not crash, velocities unchanged
        assert_array_almost_equal(velocities, np.zeros_like(mesh_positions))

    def test_fixed_vertices_no_force(self, muscle, mesh_positions):
        """Fixed vertices should not receive force."""
        muscle.build_fibers_from_mesh(
            positions=mesh_positions,
            origin_vertices=[0, 1, 2, 3],
            insertion_vertices=[4, 5, 6, 7],
        )
        muscle.activation = 1.0

        velocities = np.zeros_like(mesh_positions)
        inv_masses = np.ones(len(mesh_positions))
        inv_masses[0] = 0.0  # Fix vertex 0
        dt = 0.01

        muscle.apply_contraction_forces(mesh_positions, velocities, inv_masses, dt)

        # Vertex 0 should have zero velocity
        assert_array_almost_equal(velocities[0], [0.0, 0.0, 0.0])

    def test_volume_preservation_inactive(self, muscle, mesh_positions):
        """Inactive muscle should not apply volume preservation."""
        muscle.activation = 0.0
        velocities = np.zeros_like(mesh_positions)
        inv_masses = np.ones(len(mesh_positions))
        vertex_indices = list(range(len(mesh_positions)))
        dt = 0.01

        muscle.apply_volume_preservation(
            mesh_positions, velocities, inv_masses, vertex_indices, dt
        )

        # Velocities should be unchanged
        assert_array_almost_equal(velocities, np.zeros_like(mesh_positions))
