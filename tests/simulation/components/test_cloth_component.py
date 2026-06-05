"""
Whitebox tests for ClothComponent.

Tests cover:
- Cloth creation and configuration
- Grid and mesh creation
- Particle and constraint management
- Pinning functionality
- Wind interaction
- Collision handling
- Tearing behavior
- Serialization
"""

import pytest

from engine.simulation.character.character_controller import Vector3
from engine.simulation.components.cloth_component import (
    ClothComponent,
    ClothConfig,
    ClothConstraint,
    ClothParticle,
    ClothSolverType,
    CollisionMode,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cloth_config() -> ClothConfig:
    """Default cloth configuration."""
    return ClothConfig(
        solver_type=ClothSolverType.POSITION_BASED,
        iteration_count=4,
        stretch_stiffness=0.9,
        bend_stiffness=0.5,
    )


@pytest.fixture
def cloth_component(cloth_config) -> ClothComponent:
    """Create a cloth component."""
    return ClothComponent(entity_id=1, config=cloth_config)


@pytest.fixture
def grid_cloth() -> ClothComponent:
    """Create a grid-based cloth."""
    cloth = ClothComponent(entity_id=2)
    cloth.create_grid(width=5, height=5, spacing=0.1)
    return cloth


@pytest.fixture
def mesh_cloth() -> ClothComponent:
    """Create a mesh-based cloth (simple quad)."""
    cloth = ClothComponent(entity_id=3)
    vertices = [
        Vector3(0, 0, 0), Vector3(1, 0, 0),
        Vector3(1, 0, 1), Vector3(0, 0, 1),
    ]
    indices = [0, 1, 2, 0, 2, 3]
    cloth.create_from_mesh(vertices, indices)
    return cloth


# =============================================================================
# ClothSolverType Tests
# =============================================================================


class TestClothSolverType:
    """Tests for ClothSolverType enum."""

    def test_all_solver_types(self):
        """Test all solver types exist."""
        assert ClothSolverType.POSITION_BASED.value == "pbd"
        assert ClothSolverType.MASS_SPRING.value == "mass_spring"
        assert ClothSolverType.FEM.value == "fem"


class TestCollisionMode:
    """Tests for CollisionMode enum."""

    def test_all_collision_modes(self):
        """Test all collision modes exist."""
        assert CollisionMode.VERTEX.value == "vertex"
        assert CollisionMode.CONTINUOUS.value == "continuous"
        assert CollisionMode.HYBRID.value == "hybrid"


# =============================================================================
# ClothConfig Tests
# =============================================================================


class TestClothConfig:
    """Tests for ClothConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ClothConfig()

        assert config.solver_type == ClothSolverType.POSITION_BASED
        assert config.iteration_count == 4
        assert config.stretch_stiffness == 0.9
        assert config.bend_stiffness == 0.5
        assert config.damping == 0.05
        assert config.friction == 0.5
        assert config.self_collision is False
        assert config.self_collision_distance == 0.02
        assert config.gravity_scale == 1.0
        assert config.wind_enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ClothConfig(
            solver_type=ClothSolverType.FEM,
            iteration_count=8,
            stretch_stiffness=1.0,
            self_collision=True,
        )

        assert config.solver_type == ClothSolverType.FEM
        assert config.iteration_count == 8
        assert config.stretch_stiffness == 1.0
        assert config.self_collision is True


# =============================================================================
# ClothParticle Tests
# =============================================================================


class TestClothParticle:
    """Tests for ClothParticle dataclass."""

    def test_default_values(self):
        """Test default particle values."""
        particle = ClothParticle()

        assert particle.position.magnitude() == 0.0
        assert particle.velocity.magnitude() == 0.0
        assert particle.mass == 1.0
        assert particle.pinned is False
        assert particle.target_position is None

    def test_custom_values(self):
        """Test custom particle values."""
        particle = ClothParticle(
            position=Vector3(1.0, 2.0, 3.0),
            velocity=Vector3(0.1, 0.0, 0.0),
            mass=0.5,
            pinned=True,
            target_position=Vector3(1.0, 2.0, 3.0),
        )

        assert particle.position.x == 1.0
        assert particle.mass == 0.5
        assert particle.pinned is True


# =============================================================================
# ClothConstraint Tests
# =============================================================================


class TestClothConstraint:
    """Tests for ClothConstraint dataclass."""

    def test_default_values(self):
        """Test default constraint values."""
        constraint = ClothConstraint()

        assert constraint.particle_a == 0
        assert constraint.particle_b == 0
        assert constraint.rest_length == 1.0
        assert constraint.stiffness == 1.0
        assert constraint.constraint_type == "stretch"

    def test_custom_values(self):
        """Test custom constraint values."""
        constraint = ClothConstraint(
            particle_a=5,
            particle_b=10,
            rest_length=0.1,
            stiffness=0.8,
            constraint_type="bend",
        )

        assert constraint.particle_a == 5
        assert constraint.particle_b == 10
        assert constraint.rest_length == 0.1
        assert constraint.constraint_type == "bend"


# =============================================================================
# ClothComponent Creation Tests
# =============================================================================


class TestClothComponentCreation:
    """Tests for cloth component creation."""

    def test_create_with_default_config(self):
        """Test creating cloth with default config."""
        cloth = ClothComponent(entity_id=1)

        assert cloth.entity_id == 1
        assert cloth.cloth_id is None
        assert cloth.particle_count == 0
        assert cloth.constraint_count == 0
        assert cloth.enabled is True

    def test_create_with_custom_config(self, cloth_component, cloth_config):
        """Test creating cloth with custom config."""
        assert cloth_component.config.stretch_stiffness == 0.9
        assert cloth_component.config.bend_stiffness == 0.5


# =============================================================================
# Grid Creation Tests
# =============================================================================


class TestGridCreation:
    """Tests for grid cloth creation."""

    def test_create_grid(self, grid_cloth):
        """Test creating grid cloth."""
        # 5x5 grid = 25 particles
        assert grid_cloth.particle_count == 25

    def test_grid_constraints(self, grid_cloth):
        """Test grid creates constraints."""
        # Grid has stretch, shear, and bend constraints
        assert grid_cloth.constraint_count > 0

    def test_grid_particle_positions(self, grid_cloth):
        """Test grid particle positions are correct."""
        # First particle at origin
        pos0 = grid_cloth.get_particle_position(0)
        assert pos0 is not None
        assert pos0.x == 0.0
        assert pos0.y == 0.0
        assert pos0.z == 0.0

        # Second particle offset by spacing
        pos1 = grid_cloth.get_particle_position(1)
        assert pos1 is not None
        assert abs(pos1.x - 0.1) < 0.001

    def test_grid_with_origin(self):
        """Test grid with custom origin."""
        cloth = ClothComponent(entity_id=1)
        cloth.create_grid(
            width=3,
            height=3,
            spacing=1.0,
            origin=Vector3(10.0, 5.0, 0.0),
        )

        pos0 = cloth.get_particle_position(0)
        assert pos0.x == 10.0
        assert pos0.y == 5.0

    def test_grid_clears_previous_data(self):
        """Test creating grid clears previous data."""
        cloth = ClothComponent(entity_id=1)
        cloth.create_grid(width=10, height=10, spacing=1.0)
        assert cloth.particle_count == 100

        cloth.create_grid(width=2, height=2, spacing=1.0)
        assert cloth.particle_count == 4


# =============================================================================
# Mesh Creation Tests
# =============================================================================


class TestMeshCreation:
    """Tests for mesh cloth creation."""

    def test_create_from_mesh(self, mesh_cloth):
        """Test creating cloth from mesh."""
        assert mesh_cloth.particle_count == 4

    def test_mesh_constraints(self, mesh_cloth):
        """Test mesh creates constraints from edges."""
        # Quad has 5 edges (4 outer + 1 diagonal)
        assert mesh_cloth.constraint_count > 0

    def test_mesh_with_uvs(self):
        """Test creating cloth with UVs."""
        cloth = ClothComponent(entity_id=1)
        vertices = [
            Vector3(0, 0, 0), Vector3(1, 0, 0),
            Vector3(1, 0, 1), Vector3(0, 0, 1),
        ]
        indices = [0, 1, 2, 0, 2, 3]
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]

        cloth.create_from_mesh(vertices, indices, uvs)

        assert cloth.particle_count == 4


# =============================================================================
# Pinning Tests
# =============================================================================


class TestPinning:
    """Tests for particle pinning."""

    def test_pin_particle(self, grid_cloth):
        """Test pinning a particle."""
        grid_cloth.pin_particle(0)

        # Access internal state for verification
        assert grid_cloth._particles[0].pinned is True
        assert grid_cloth._particles[0].mass == 0.0

    def test_pin_with_target(self, grid_cloth):
        """Test pinning with target position."""
        target = Vector3(5.0, 0.0, 0.0)
        grid_cloth.pin_particle(0, target=target)

        assert grid_cloth._particles[0].target_position is not None
        assert grid_cloth._particles[0].target_position.x == 5.0

    def test_unpin_particle(self, grid_cloth):
        """Test unpinning a particle."""
        grid_cloth.pin_particle(0)
        grid_cloth.unpin_particle(0)

        assert grid_cloth._particles[0].pinned is False
        assert grid_cloth._particles[0].mass == 1.0
        assert grid_cloth._particles[0].target_position is None

    def test_pin_out_of_range(self, grid_cloth):
        """Test pinning out of range particle does nothing."""
        grid_cloth.pin_particle(999)
        # Should not raise

    def test_unpin_out_of_range(self, grid_cloth):
        """Test unpinning out of range particle does nothing."""
        grid_cloth.unpin_particle(999)
        # Should not raise

    def test_pin_row(self, grid_cloth):
        """Test pinning entire row."""
        grid_cloth.pin_row(0, width=5)

        for x in range(5):
            assert grid_cloth._particles[x].pinned is True

    def test_pin_column(self, grid_cloth):
        """Test pinning entire column."""
        grid_cloth.pin_column(0, width=5, height=5)

        for y in range(5):
            assert grid_cloth._particles[y * 5].pinned is True

    def test_pin_to_transform(self, grid_cloth):
        """Test pinning particles to transforms (skinning)."""
        indices = [0, 1, 2]
        transforms = [
            Vector3(0, 1, 0),
            Vector3(1, 1, 0),
            Vector3(2, 1, 0),
        ]

        grid_cloth.pin_to_transform(indices, transforms)

        assert grid_cloth._particles[0].pinned is True
        assert grid_cloth._particles[0].target_position.y == 1.0
        assert grid_cloth._particles[1].target_position.x == 1.0


# =============================================================================
# Wind Tests
# =============================================================================


class TestWind:
    """Tests for wind interaction."""

    def test_set_wind(self, grid_cloth):
        """Test setting wind parameters."""
        grid_cloth.set_wind(
            velocity=Vector3(10.0, 0.0, 0.0),
            turbulence=0.5,
        )

        assert grid_cloth.wind_velocity.x == 10.0
        assert grid_cloth._wind_turbulence == 0.5

    def test_wind_property(self, grid_cloth):
        """Test wind velocity property."""
        grid_cloth.wind_velocity = Vector3(5.0, 0.0, 5.0)

        assert grid_cloth.wind_velocity.x == 5.0
        assert grid_cloth.wind_velocity.z == 5.0

    def test_turbulence_clamped(self, grid_cloth):
        """Test turbulence is clamped to 0-1."""
        grid_cloth.set_wind(Vector3.zero(), turbulence=2.0)
        assert grid_cloth._wind_turbulence == 1.0

        grid_cloth.set_wind(Vector3.zero(), turbulence=-0.5)
        assert grid_cloth._wind_turbulence == 0.0

    def test_apply_wind_force(self, grid_cloth):
        """Test applying wind force to particles."""
        grid_cloth.set_wind(Vector3(10.0, 0.0, 0.0), turbulence=0.0)

        initial_vel = Vector3(
            grid_cloth._particles[5].velocity.x,
            grid_cloth._particles[5].velocity.y,
            grid_cloth._particles[5].velocity.z,
        )

        grid_cloth.apply_wind_force(dt=0.1)

        # Non-pinned particle should have velocity changed
        assert grid_cloth._particles[5].velocity.x != initial_vel.x

    def test_wind_disabled(self, grid_cloth):
        """Test wind doesn't apply when disabled."""
        grid_cloth._config.wind_enabled = False
        grid_cloth.set_wind(Vector3(100.0, 0.0, 0.0))

        initial_vel = grid_cloth._particles[5].velocity.x

        grid_cloth.apply_wind_force(dt=0.1)

        assert grid_cloth._particles[5].velocity.x == initial_vel

    def test_wind_doesnt_affect_pinned(self, grid_cloth):
        """Test wind doesn't affect pinned particles."""
        grid_cloth.pin_particle(5)
        grid_cloth.set_wind(Vector3(100.0, 0.0, 0.0))

        initial_vel = grid_cloth._particles[5].velocity.x

        grid_cloth.apply_wind_force(dt=0.1)

        assert grid_cloth._particles[5].velocity.x == initial_vel


# =============================================================================
# Collision Tests
# =============================================================================


class TestCollision:
    """Tests for collision handling."""

    def test_add_collider(self, grid_cloth):
        """Test adding collider."""
        grid_cloth.add_collider(100)

        assert 100 in grid_cloth._colliders

    def test_add_duplicate_collider(self, grid_cloth):
        """Test adding duplicate collider only adds once."""
        grid_cloth.add_collider(100)
        grid_cloth.add_collider(100)

        assert len(grid_cloth._colliders) == 1

    def test_remove_collider(self, grid_cloth):
        """Test removing collider."""
        grid_cloth.add_collider(100)
        grid_cloth.remove_collider(100)

        assert 100 not in grid_cloth._colliders

    def test_remove_nonexistent_collider(self, grid_cloth):
        """Test removing nonexistent collider does nothing."""
        grid_cloth.remove_collider(999)
        # Should not raise

    def test_set_collision_mode(self, grid_cloth):
        """Test setting collision mode."""
        grid_cloth.set_collision_mode(CollisionMode.CONTINUOUS)

        assert grid_cloth._collision_mode == CollisionMode.CONTINUOUS


# =============================================================================
# Tearing Tests
# =============================================================================


class TestTearing:
    """Tests for cloth tearing."""

    def test_set_tearable(self, grid_cloth):
        """Test enabling tearing."""
        grid_cloth.set_tearable(enabled=True, threshold=15.0)

        assert grid_cloth._tearable is True
        assert grid_cloth._tear_threshold == 15.0

    def test_tear_callback(self, grid_cloth):
        """Test tear callback is called."""
        torn_particles = []
        grid_cloth.set_tear_callback(
            lambda a, b: torn_particles.append((a, b))
        )

        # Get first constraint
        if grid_cloth.constraint_count > 0:
            constraint = grid_cloth._constraints[0]
            expected = (constraint.particle_a, constraint.particle_b)

            result = grid_cloth.tear_constraint(0)

            assert result is True
            assert len(torn_particles) == 1
            assert torn_particles[0] == expected

    def test_tear_constraint(self, grid_cloth):
        """Test tearing a constraint."""
        initial_count = grid_cloth.constraint_count

        result = grid_cloth.tear_constraint(0)

        assert result is True
        assert grid_cloth.constraint_count == initial_count - 1

    def test_tear_out_of_range(self, grid_cloth):
        """Test tearing out of range constraint."""
        result = grid_cloth.tear_constraint(999)

        assert result is False


# =============================================================================
# Query Tests
# =============================================================================


class TestQueries:
    """Tests for cloth query methods."""

    def test_get_particle_position(self, grid_cloth):
        """Test getting particle position."""
        pos = grid_cloth.get_particle_position(0)

        assert pos is not None
        assert pos.x == 0.0

    def test_get_particle_position_out_of_range(self, grid_cloth):
        """Test getting position of nonexistent particle."""
        pos = grid_cloth.get_particle_position(999)

        assert pos is None

    def test_get_particle_velocity(self, grid_cloth):
        """Test getting particle velocity."""
        vel = grid_cloth.get_particle_velocity(0)

        assert vel is not None

    def test_get_particle_velocity_out_of_range(self, grid_cloth):
        """Test getting velocity of nonexistent particle."""
        vel = grid_cloth.get_particle_velocity(999)

        assert vel is None

    def test_get_positions(self, grid_cloth):
        """Test getting all positions."""
        positions = grid_cloth.get_positions()

        assert len(positions) == 25

    def test_get_mesh_positions(self, grid_cloth):
        """Test getting mesh positions."""
        positions = grid_cloth.get_mesh_positions()

        assert len(positions) == 25


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for cloth lifecycle management."""

    def test_initialize(self, cloth_component):
        """Test initialization with physics ID."""
        cloth_component.initialize(cloth_id=42)

        assert cloth_component.cloth_id == 42

    def test_cleanup(self, grid_cloth):
        """Test cleanup clears all data."""
        grid_cloth.initialize(cloth_id=42)
        grid_cloth.add_collider(100)

        grid_cloth.cleanup()

        assert grid_cloth.cloth_id is None
        assert grid_cloth.particle_count == 0
        assert grid_cloth.constraint_count == 0

    def test_enabled_property(self, cloth_component):
        """Test enabled property."""
        assert cloth_component.enabled is True

        cloth_component.enabled = False
        assert cloth_component.enabled is False


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for state serialization."""

    def test_get_state(self, grid_cloth):
        """Test getting serializable state."""
        grid_cloth.set_wind(Vector3(5.0, 0.0, 0.0))
        grid_cloth.set_tearable(True)

        state = grid_cloth.get_state()

        assert state["entity_id"] == 2
        assert state["particle_count"] == 25
        assert state["constraint_count"] > 0
        assert state["enabled"] is True
        assert state["wind"] == (5.0, 0.0, 0.0)
        assert state["tearable"] is True
        assert "config" in state
        assert state["config"]["solver_type"] == "pbd"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_cloth(self):
        """Test operations on empty cloth."""
        cloth = ClothComponent(entity_id=1)

        assert cloth.particle_count == 0
        assert cloth.constraint_count == 0
        assert cloth.get_positions() == []

    def test_single_particle(self):
        """Test cloth with single particle."""
        cloth = ClothComponent(entity_id=1)
        cloth.create_grid(width=1, height=1, spacing=1.0)

        assert cloth.particle_count == 1
        assert cloth.constraint_count == 0

    def test_2x1_grid(self):
        """Test minimal 2x1 grid."""
        cloth = ClothComponent(entity_id=1)
        cloth.create_grid(width=2, height=1, spacing=1.0)

        assert cloth.particle_count == 2
        # Should have one stretch constraint
        assert cloth.constraint_count >= 1

    def test_1x2_grid(self):
        """Test minimal 1x2 grid."""
        cloth = ClothComponent(entity_id=1)
        cloth.create_grid(width=1, height=2, spacing=1.0)

        assert cloth.particle_count == 2

    def test_large_grid(self):
        """Test large grid creation."""
        cloth = ClothComponent(entity_id=1)
        cloth.create_grid(width=50, height=50, spacing=0.01)

        assert cloth.particle_count == 2500
        # Many constraints from stretch, shear, and bend
        assert cloth.constraint_count > 5000

    def test_zero_spacing(self):
        """Test grid with zero spacing (all particles at same position)."""
        cloth = ClothComponent(entity_id=1)
        cloth.create_grid(width=3, height=3, spacing=0.0)

        # All particles at origin
        for i in range(9):
            pos = cloth.get_particle_position(i)
            assert pos.x == 0.0
            assert pos.z == 0.0

    def test_negative_spacing(self):
        """Test grid with negative spacing."""
        cloth = ClothComponent(entity_id=1)
        cloth.create_grid(width=3, height=3, spacing=-0.1)

        # Particles should have negative offsets
        pos = cloth.get_particle_position(1)
        assert pos.x < 0.0

    def test_wind_with_zero_dt(self, grid_cloth):
        """Test wind application with zero dt."""
        grid_cloth.set_wind(Vector3(100.0, 0.0, 0.0))
        initial_vel = grid_cloth._particles[5].velocity.x

        grid_cloth.apply_wind_force(dt=0.0)

        # No change with zero dt
        assert grid_cloth._particles[5].velocity.x == initial_vel
