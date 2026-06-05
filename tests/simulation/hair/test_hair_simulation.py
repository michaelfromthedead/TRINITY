"""
Whitebox tests for main hair simulation system.
"""

import math
import numpy as np
import pytest

from engine.simulation.hair.config import (
    DEFAULT_HAIR_LENGTH,
    DEFAULT_HAIR_THICKNESS,
    DEFAULT_STRAND_SEGMENTS,
    MAX_GUIDE_HAIRS,
    NUMERICAL_EPSILON,
)
from engine.simulation.hair.hair_simulation import (
    GuideHair,
    HairControlPoint,
    HairSimulation,
    HairSimulationConfig,
    HairState,
    HairStrand,
    InterpolatedHair,
    create_hair_from_scalp,
    create_hair_strand,
    create_interpolated_hairs,
)


class TestHairControlPoint:
    """Tests for HairControlPoint dataclass."""

    def test_create_control_point(self):
        """Control point should initialize with correct values."""
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        prev_pos = np.array([0.9, 1.9, 2.9], dtype=np.float32)
        rest_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        cp = HairControlPoint(
            position=pos,
            prev_position=prev_pos,
            rest_position=rest_pos,
            inv_mass=1.0,
        )

        np.testing.assert_array_equal(cp.position, pos)
        np.testing.assert_array_equal(cp.prev_position, prev_pos)
        np.testing.assert_array_equal(cp.rest_position, rest_pos)
        assert cp.inv_mass == 1.0

    def test_control_point_is_root(self):
        """Root points should have zero inverse mass."""
        root_cp = HairControlPoint(
            position=np.zeros(3, dtype=np.float32),
            prev_position=np.zeros(3, dtype=np.float32),
            rest_position=np.zeros(3, dtype=np.float32),
            inv_mass=0.0,
        )
        assert root_cp.is_root is True

    def test_control_point_is_not_root(self):
        """Non-root points should have non-zero inverse mass."""
        non_root_cp = HairControlPoint(
            position=np.zeros(3, dtype=np.float32),
            prev_position=np.zeros(3, dtype=np.float32),
            rest_position=np.zeros(3, dtype=np.float32),
            inv_mass=1.0,
        )
        assert non_root_cp.is_root is False

    def test_default_velocity_is_zero(self):
        """Default velocity should be zero vector."""
        cp = HairControlPoint(
            position=np.zeros(3, dtype=np.float32),
            prev_position=np.zeros(3, dtype=np.float32),
            rest_position=np.zeros(3, dtype=np.float32),
        )
        np.testing.assert_array_equal(cp.velocity, np.zeros(3, dtype=np.float32))


class TestHairStrand:
    """Tests for HairStrand dataclass."""

    def _create_simple_strand(self, num_segments=3):
        """Helper to create a simple test strand."""
        control_points = []
        for i in range(num_segments + 1):
            cp = HairControlPoint(
                position=np.array([0.0, float(i) * 0.1, 0.0], dtype=np.float32),
                prev_position=np.array([0.0, float(i) * 0.1, 0.0], dtype=np.float32),
                rest_position=np.array([0.0, float(i) * 0.1, 0.0], dtype=np.float32),
                inv_mass=0.0 if i == 0 else 1.0,
            )
            control_points.append(cp)

        rest_lengths = [0.1] * num_segments

        return HairStrand(
            control_points=control_points,
            rest_lengths=rest_lengths,
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )

    def test_strand_num_segments(self):
        """Strand should report correct number of segments."""
        strand = self._create_simple_strand(num_segments=5)
        assert strand.num_segments == 5

    def test_strand_length(self):
        """Strand should compute total length from rest lengths."""
        strand = self._create_simple_strand(num_segments=4)
        # 4 segments * 0.1 length each = 0.4
        assert strand.length == pytest.approx(0.4)

    def test_strand_get_positions_array(self):
        """get_positions_array should return Nx3 array."""
        strand = self._create_simple_strand(num_segments=3)
        positions = strand.get_positions_array()

        assert positions.shape == (4, 3)
        assert positions.dtype == np.float32

    def test_strand_set_positions_from_array(self):
        """set_positions_from_array should update all control points."""
        strand = self._create_simple_strand(num_segments=2)
        new_positions = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float32)

        strand.set_positions_from_array(new_positions)

        for i, cp in enumerate(strand.control_points):
            np.testing.assert_array_equal(cp.position, new_positions[i])

    def test_strand_default_thickness(self):
        """Strand should use default thickness if not specified."""
        strand = self._create_simple_strand()
        assert strand.thickness == DEFAULT_HAIR_THICKNESS

    def test_strand_custom_thickness(self):
        """Strand should accept custom thickness."""
        control_points = [
            HairControlPoint(
                position=np.zeros(3, dtype=np.float32),
                prev_position=np.zeros(3, dtype=np.float32),
                rest_position=np.zeros(3, dtype=np.float32),
            )
        ]
        strand = HairStrand(
            control_points=control_points,
            rest_lengths=[],
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            thickness=0.001,
        )
        assert strand.thickness == 0.001


class TestGuideHair:
    """Tests for GuideHair dataclass."""

    def test_guide_hair_index(self):
        """Guide hair should have an index."""
        control_points = [
            HairControlPoint(
                position=np.zeros(3, dtype=np.float32),
                prev_position=np.zeros(3, dtype=np.float32),
                rest_position=np.zeros(3, dtype=np.float32),
            )
        ]
        guide = GuideHair(
            control_points=control_points,
            rest_lengths=[],
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            index=5,
        )
        assert guide.index == 5

    def test_guide_hair_uv(self):
        """Guide hair should store UV coordinates."""
        control_points = [
            HairControlPoint(
                position=np.zeros(3, dtype=np.float32),
                prev_position=np.zeros(3, dtype=np.float32),
                rest_position=np.zeros(3, dtype=np.float32),
            )
        ]
        guide = GuideHair(
            control_points=control_points,
            rest_lengths=[],
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            uv=(0.5, 0.7),
        )
        assert guide.uv == (0.5, 0.7)

    def test_guide_hair_neighbors(self):
        """Guide hair should store neighbor indices."""
        control_points = [
            HairControlPoint(
                position=np.zeros(3, dtype=np.float32),
                prev_position=np.zeros(3, dtype=np.float32),
                rest_position=np.zeros(3, dtype=np.float32),
            )
        ]
        guide = GuideHair(
            control_points=control_points,
            rest_lengths=[],
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            neighbor_indices=[1, 2, 3],
        )
        assert guide.neighbor_indices == [1, 2, 3]


class TestInterpolatedHair:
    """Tests for InterpolatedHair dataclass."""

    def test_interpolated_hair_guide_indices(self):
        """Interpolated hair should store guide hair indices."""
        control_points = [
            HairControlPoint(
                position=np.zeros(3, dtype=np.float32),
                prev_position=np.zeros(3, dtype=np.float32),
                rest_position=np.zeros(3, dtype=np.float32),
            )
        ]
        hair = InterpolatedHair(
            control_points=control_points,
            rest_lengths=[],
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            guide_hair_indices=[0, 1, 2],
        )
        assert hair.guide_hair_indices == [0, 1, 2]

    def test_interpolated_hair_weights(self):
        """Interpolated hair should store interpolation weights."""
        control_points = [
            HairControlPoint(
                position=np.zeros(3, dtype=np.float32),
                prev_position=np.zeros(3, dtype=np.float32),
                rest_position=np.zeros(3, dtype=np.float32),
            )
        ]
        weights = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        hair = InterpolatedHair(
            control_points=control_points,
            rest_lengths=[],
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            interpolation_weights=weights,
        )
        np.testing.assert_array_equal(hair.interpolation_weights, weights)


class TestHairSimulationConfig:
    """Tests for HairSimulationConfig dataclass."""

    def test_default_config_values(self):
        """Config should have sensible defaults."""
        config = HairSimulationConfig()

        assert config.timestep > 0
        assert config.solver_iterations > 0
        assert 0.0 <= config.damping <= 1.0
        assert config.length_stiffness >= 0.0
        assert config.shape_stiffness >= 0.0
        assert config.local_shape_stiffness >= 0.0

    def test_default_gravity(self):
        """Default gravity should point downward."""
        config = HairSimulationConfig()
        assert config.gravity[1] < 0  # Y is down
        assert config.gravity[1] == pytest.approx(-9.81)

    def test_custom_config_values(self):
        """Config should accept custom values."""
        custom_gravity = np.array([0.0, -5.0, 0.0], dtype=np.float32)
        config = HairSimulationConfig(
            timestep=0.01,
            solver_iterations=8,
            damping=0.8,
            gravity=custom_gravity,
            length_stiffness=0.9,
            enable_collision=False,
            enable_wind=False,
        )

        assert config.timestep == 0.01
        assert config.solver_iterations == 8
        assert config.damping == 0.8
        np.testing.assert_array_equal(config.gravity, custom_gravity)
        assert config.enable_collision is False
        assert config.enable_wind is False


class TestHairSimulation:
    """Tests for HairSimulation class."""

    def _create_test_guide_hair(self, index=0):
        """Create a simple guide hair for testing."""
        control_points = []
        for i in range(5):
            cp = HairControlPoint(
                position=np.array([0.0, float(i) * 0.05, 0.0], dtype=np.float32),
                prev_position=np.array([0.0, float(i) * 0.05, 0.0], dtype=np.float32),
                rest_position=np.array([0.0, float(i) * 0.05, 0.0], dtype=np.float32),
                inv_mass=0.0 if i == 0 else 1.0,
            )
            control_points.append(cp)

        return GuideHair(
            control_points=control_points,
            rest_lengths=[0.05, 0.05, 0.05, 0.05],
            root_position=np.zeros(3, dtype=np.float32),
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            index=index,
        )

    def test_simulation_initial_state(self):
        """Simulation should start inactive."""
        sim = HairSimulation()
        assert sim.state == HairState.INACTIVE

    def test_simulation_start_stop(self):
        """Simulation should transition between states."""
        sim = HairSimulation()

        sim.start()
        assert sim.state == HairState.SIMULATING

        sim.pause()
        assert sim.state == HairState.PAUSED

        sim.resume()
        assert sim.state == HairState.SIMULATING

        sim.stop()
        assert sim.state == HairState.INACTIVE

    def test_add_guide_hair(self):
        """Should be able to add guide hairs."""
        sim = HairSimulation()
        hair = self._create_test_guide_hair()

        sim.add_guide_hair(hair)

        assert sim.num_guide_hairs == 1
        assert len(sim.guide_hairs) == 1

    def test_add_guide_hair_assigns_index(self):
        """Adding guide hair should assign sequential indices."""
        sim = HairSimulation()

        for i in range(3):
            hair = self._create_test_guide_hair()
            sim.add_guide_hair(hair)
            assert hair.index == i

    def test_max_guide_hairs_limit(self):
        """Should raise error when exceeding max guide hairs."""
        sim = HairSimulation()

        # Add MAX_GUIDE_HAIRS hairs
        for i in range(MAX_GUIDE_HAIRS):
            sim.add_guide_hair(self._create_test_guide_hair())

        # Adding one more should raise
        with pytest.raises(ValueError, match="Maximum guide hairs"):
            sim.add_guide_hair(self._create_test_guide_hair())

    def test_clear_hairs(self):
        """clear_hairs should remove all hairs."""
        sim = HairSimulation()
        sim.add_guide_hair(self._create_test_guide_hair())
        sim.add_guide_hair(self._create_test_guide_hair())

        sim.clear_hairs()

        assert sim.num_guide_hairs == 0
        assert len(sim.interpolated_hairs) == 0

    def test_set_head_transform(self):
        """Should be able to set head transform."""
        sim = HairSimulation()
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rot = np.eye(3, dtype=np.float32)

        sim.set_head_transform(pos, rot)
        # Should not raise, transforms are stored internally

    def test_set_wind(self):
        """Should be able to set wind velocity."""
        sim = HairSimulation()
        wind = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        sim.set_wind(wind)
        # Should not raise

    def test_add_collision_capsule(self):
        """Should be able to add collision capsules."""
        sim = HairSimulation()
        point_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        point_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        radius = 0.1

        sim.add_collision_capsule(point_a, point_b, radius)
        # Should not raise

    def test_clear_collision_capsules(self):
        """Should be able to clear collision capsules."""
        sim = HairSimulation()
        point_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        point_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        sim.add_collision_capsule(point_a, point_b, 0.1)
        sim.clear_collision_capsules()
        # Should not raise

    def test_step_inactive_does_nothing(self):
        """Stepping while inactive should do nothing."""
        sim = HairSimulation()
        hair = self._create_test_guide_hair()
        sim.add_guide_hair(hair)

        original_positions = hair.get_positions_array().copy()

        sim.step(0.016)  # One frame

        # Positions should be unchanged
        np.testing.assert_array_equal(hair.get_positions_array(), original_positions)

    def test_step_simulating_updates_positions(self):
        """Stepping while simulating should update positions."""
        config = HairSimulationConfig(
            gravity=np.array([0.0, -9.81, 0.0], dtype=np.float32),
            shape_stiffness=0.0,  # Disable shape matching for clearer gravity test
            local_shape_stiffness=0.0,
            enable_collision=False,
            damping=0.99,  # High damping to preserve velocity
        )
        sim = HairSimulation(config)

        # Create a horizontal hair that will droop under gravity
        hair = create_hair_strand(
            root_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            root_normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Horizontal direction
            length=0.2,
            num_segments=4,
        )
        sim.add_guide_hair(hair)

        original_tip_pos = hair.control_points[-1].position.copy()

        sim.start()
        # Step multiple times to ensure simulation progresses
        for _ in range(50):
            sim.step(0.016)

        # Tip should have moved down due to gravity (since hair is horizontal)
        new_tip_pos = hair.control_points[-1].position
        # Y position should decrease (gravity is -Y)
        assert new_tip_pos[1] < original_tip_pos[1]

    def test_simulation_with_wind(self):
        """Wind should affect hair positions."""
        config = HairSimulationConfig(
            gravity=np.zeros(3, dtype=np.float32),  # No gravity
            enable_wind=True,
        )
        sim = HairSimulation(config)
        hair = self._create_test_guide_hair()
        sim.add_guide_hair(hair)

        # Strong wind in X direction
        sim.set_wind(np.array([10.0, 0.0, 0.0], dtype=np.float32))

        original_tip_x = hair.control_points[-1].position[0]

        sim.start()
        for _ in range(20):
            sim.step(0.016)

        # Tip should have moved in X direction
        new_tip_x = hair.control_points[-1].position[0]
        assert new_tip_x > original_tip_x


class TestCreateHairStrand:
    """Tests for create_hair_strand function."""

    def test_create_basic_strand(self):
        """Should create a valid hair strand."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal)

        assert isinstance(hair, GuideHair)
        assert len(hair.control_points) == DEFAULT_STRAND_SEGMENTS + 1
        assert len(hair.rest_lengths) == DEFAULT_STRAND_SEGMENTS

    def test_create_strand_with_custom_length(self):
        """Should create strand with specified length."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        length = 0.5

        hair = create_hair_strand(root_pos, root_normal, length=length)

        assert hair.length == pytest.approx(length)

    def test_create_strand_with_custom_segments(self):
        """Should create strand with specified segments."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        num_segments = 8

        hair = create_hair_strand(root_pos, root_normal, num_segments=num_segments)

        assert hair.num_segments == num_segments
        assert len(hair.control_points) == num_segments + 1

    def test_create_strand_root_is_fixed(self):
        """Root control point should have zero inverse mass."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal)

        assert hair.control_points[0].is_root
        assert hair.control_points[0].inv_mass == 0.0

    def test_create_strand_non_root_points_have_mass(self):
        """Non-root control points should have non-zero inverse mass."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal)

        for cp in hair.control_points[1:]:
            assert not cp.is_root
            assert cp.inv_mass > 0.0

    def test_create_strand_with_curl(self):
        """Curl factor should affect strand shape."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        straight = create_hair_strand(root_pos, root_normal, curl_factor=0.0)
        curly = create_hair_strand(root_pos, root_normal, curl_factor=1.0)

        # Curly hair tip should be at different position than straight
        straight_tip = straight.control_points[-1].position
        curly_tip = curly.control_points[-1].position

        # They should differ (at least in X or Z due to curl)
        assert not np.allclose(straight_tip, curly_tip)

    def test_create_strand_normalizes_root_normal(self):
        """Root normal should be normalized."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        unnormalized = np.array([0.0, 5.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, unnormalized)

        assert np.linalg.norm(hair.root_normal) == pytest.approx(1.0)


class TestCreateHairFromScalp:
    """Tests for create_hair_from_scalp function."""

    def test_create_hairs_from_scalp(self):
        """Should create hairs from scalp positions."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ], dtype=np.float32)
        normals = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        hairs = create_hair_from_scalp(positions, normals)

        assert len(hairs) == 3
        for hair in hairs:
            assert isinstance(hair, GuideHair)

    def test_create_hairs_respects_max_hairs(self):
        """Should limit number of hairs to max_hairs."""
        num_positions = 100
        positions = np.random.randn(num_positions, 3).astype(np.float32)
        normals = np.tile([0.0, 1.0, 0.0], (num_positions, 1)).astype(np.float32)
        max_hairs = 10

        hairs = create_hair_from_scalp(positions, normals, max_hairs=max_hairs)

        assert len(hairs) == max_hairs

    def test_create_hairs_with_length_variation(self):
        """Length variation should create different length hairs."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ], dtype=np.float32)
        normals = np.tile([0.0, 1.0, 0.0], (5, 1)).astype(np.float32)

        hairs = create_hair_from_scalp(
            positions, normals,
            hair_length=0.3,
            length_variation=0.2,
        )

        lengths = [h.length for h in hairs]
        # With variation, not all lengths should be identical
        assert len(set(lengths)) > 1  # Should have some variation

    def test_create_hairs_empty_input(self):
        """Should handle empty input gracefully."""
        positions = np.array([], dtype=np.float32).reshape(0, 3)
        normals = np.array([], dtype=np.float32).reshape(0, 3)

        hairs = create_hair_from_scalp(positions, normals)

        assert len(hairs) == 0


class TestCreateInterpolatedHairs:
    """Tests for create_interpolated_hairs function."""

    def _create_guide_hair(self, root_pos):
        """Helper to create a guide hair."""
        return create_hair_strand(
            root_position=root_pos,
            root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            num_segments=4,
        )

    def test_create_interpolated_hairs(self):
        """Should create interpolated hairs from guides."""
        guides = [
            self._create_guide_hair(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
            self._create_guide_hair(np.array([0.1, 0.0, 0.0], dtype=np.float32)),
        ]
        guides[0].index = 0
        guides[1].index = 1

        interpolated = create_interpolated_hairs(guides, num_interpolated=3)

        # 2 guides * 3 interpolated each = 6
        assert len(interpolated) == 6
        for hair in interpolated:
            assert isinstance(hair, InterpolatedHair)

    def test_interpolated_hair_references_guide(self):
        """Interpolated hair should reference its guide."""
        guide = self._create_guide_hair(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        guide.index = 0

        interpolated = create_interpolated_hairs([guide], num_interpolated=1)

        assert len(interpolated) == 1
        assert 0 in interpolated[0].guide_hair_indices

    def test_interpolated_hair_has_offset(self):
        """Interpolated hairs should have offset from guide."""
        guide = self._create_guide_hair(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        guide.index = 0

        interpolated = create_interpolated_hairs([guide], num_interpolated=10, radius=0.05)

        # Check that interpolated roots are offset from guide root
        guide_root = guide.root_position
        offsets = []
        for hair in interpolated:
            offset = np.linalg.norm(hair.root_position - guide_root)
            offsets.append(offset)

        # Most should have some offset (random, so check average is > 0)
        avg_offset = sum(offsets) / len(offsets)
        assert avg_offset > 0  # Should have some offset on average


class TestHairStateEnum:
    """Tests for HairState enum."""

    def test_hair_state_values(self):
        """HairState should have expected values."""
        assert HairState.INACTIVE is not None
        assert HairState.SIMULATING is not None
        assert HairState.PAUSED is not None

    def test_hair_states_distinct(self):
        """HairState values should be distinct."""
        states = [HairState.INACTIVE, HairState.SIMULATING, HairState.PAUSED]
        assert len(states) == len(set(states))


class TestEdgeCases:
    """Tests for edge cases in hair simulation."""

    def test_single_segment_strand(self):
        """Should handle single segment strand."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal, num_segments=1)

        assert hair.num_segments == 1
        assert len(hair.control_points) == 2

    def test_very_long_strand(self):
        """Should handle very long strands."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal, length=5.0, num_segments=100)

        assert hair.length == pytest.approx(5.0)
        assert hair.num_segments == 100

    def test_very_short_strand(self):
        """Should handle very short strands."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal, length=0.001, num_segments=2)

        assert hair.length == pytest.approx(0.001)

    def test_diagonal_root_normal(self):
        """Should handle diagonal root normals."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal)

        # Should normalize and create valid hair
        assert np.linalg.norm(hair.root_normal) == pytest.approx(1.0)
        assert len(hair.control_points) > 0

    def test_z_aligned_root_normal(self):
        """Should handle Z-aligned root normal (edge case for cross product)."""
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        hair = create_hair_strand(root_pos, root_normal)

        # Should still create valid hair
        assert len(hair.control_points) > 0
        for cp in hair.control_points:
            assert not np.any(np.isnan(cp.position))

    def test_simulation_with_no_hairs(self):
        """Simulation should handle having no hairs."""
        sim = HairSimulation()
        sim.start()
        sim.step(0.016)  # Should not raise
        assert sim.num_guide_hairs == 0

    def test_pause_resume_cycle(self):
        """Should handle pause/resume cycles correctly."""
        sim = HairSimulation()
        sim.start()

        for _ in range(5):
            sim.pause()
            assert sim.state == HairState.PAUSED
            sim.resume()
            assert sim.state == HairState.SIMULATING

    def test_resume_from_inactive(self):
        """Resume should not change inactive state."""
        sim = HairSimulation()
        sim.resume()  # Should not change state
        assert sim.state == HairState.INACTIVE

    def test_pause_while_inactive(self):
        """Pause should not change inactive state."""
        sim = HairSimulation()
        sim.pause()  # Should not change state
        assert sim.state == HairState.INACTIVE
